// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The on-disk Vulkan pipeline cache.
//!
//! One `VkPipelineCache` is shared by every pipeline the provider creates.
//! It is loaded from `RPS5E_PIPELINE_CACHE` (a file path; `0`/`off`
//! disables it) or `$XDG_CACHE_HOME/rps5e/pipelines/` (`~/.cache/...`; `%LOCALAPPDATA%\rps5e\pipelines` on Windows),
//! one file per game named by its title id (no title id: no cache), and
//! written back periodically and on drop.
//!
//! Periodic saves run on a background thread (the driver data of a game
//! is tens of MiB: copying and writing it on the command thread stalled a
//! frame each time); the final save on drop runs inline.
//!
//! The file starts with a signature line naming the emulator version and
//! the device (vendor, device id, driver version, pipeline-cache UUID),
//! followed by a 64-bit checksum of the driver data and the data itself. A
//! file whose signature or checksum does not match is ignored (a new
//! driver, emulator build or a torn write), and data the driver rejects is
//! dropped: the cache then starts empty.

use crate::device::GpuDevice;
use ash::vk;
use std::path::PathBuf;
use std::sync::Arc;

/// Pipelines created between two saves.
const SAVE_INTERVAL: u32 = 64;

pub struct PipelineDiskCache {
    dev: Arc<GpuDevice>,
    pub cache: vk::PipelineCache,
    path: Option<PathBuf>,
    /// Bytes accepted from disk at creation.
    pub loaded_bytes: usize,
    unsaved: u32,
    /// The background save in progress.
    saving: Option<std::thread::JoinHandle<()>>,
}

/// The title id of the running game (pipeline cache file name).
static TITLE_ID: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// A title id usable as a file name: letters, digits, `-` and `_` only.
pub fn valid_title_id(id: &str) -> bool {
    !id.is_empty() && id.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
}

/// Name the running game (param.json title id, or content id): pipeline
/// caches created from now on are kept per game. Ids that are no plain
/// file name are ignored.
pub fn set_title_id(id: &str) {
    let id = id.trim();
    *TITLE_ID.lock().unwrap_or_else(|e| e.into_inner()) = valid_title_id(id).then(|| id.to_string());
}

/// The cache file of the game `title` under the cache directory `base`.
pub fn title_path(base: &std::path::Path, title: &str) -> PathBuf {
    ps5vm::appenv::app_dir(base).join("pipelines").join(format!("{title}.bin"))
}

/// The default cache file (None when disabled or no game is named). The
/// device is not part of the name: the signature check rejects data of
/// another device or driver, which is then replaced.
pub fn default_path(_dev: &GpuDevice) -> Option<PathBuf> {
    match std::env::var("RPS5E_PIPELINE_CACHE") {
        Ok(v) if v == "0" || v.eq_ignore_ascii_case("off") => return None,
        Ok(v) if !v.is_empty() => return Some(PathBuf::from(v)),
        _ => {}
    }
    let title = TITLE_ID.lock().unwrap_or_else(|e| e.into_inner()).clone()?;
    let base = ps5vm::appenv::cache_base()?;
    Some(title_path(&base, &title))
}

/// The signature line of cache files written for `props`. It names the
/// shader translator sources too: pipelines of another translator are never
/// asked for again, and keeping them would grow the file without bound.
pub fn signature(props: &vk::PhysicalDeviceProperties) -> String {
    let uuid: String = props.pipeline_cache_uuid.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "rps5ePC2:{}:{}:{:08x}:{:08x}:{:08x}:{uuid}\n",
        env!("CARGO_PKG_VERSION"),
        env!("PS5GPU_SHADER_SOURCE_VERSION"),
        props.vendor_id,
        props.device_id,
        props.driver_version
    )
}

/// 64-bit FNV-1a of the driver data.
pub fn checksum(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

/// The file contents for driver data `payload`.
pub fn encode(signature: &str, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(signature.len() + 8 + payload.len());
    out.extend_from_slice(signature.as_bytes());
    out.extend_from_slice(&checksum(payload).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// The driver data of a cache file when its signature and checksum match.
pub fn decode<'a>(signature: &str, file: &'a [u8]) -> Option<&'a [u8]> {
    let rest = file.strip_prefix(signature.as_bytes())?;
    if rest.len() < 8 {
        return None;
    }
    let (sum, payload) = rest.split_at(8);
    (u64::from_le_bytes(sum.try_into().ok()?) == checksum(payload)).then_some(payload)
}

/// True when `data` starts with a version-one pipeline cache header for
/// this device.
fn header_matches(dev: &GpuDevice, data: &[u8]) -> bool {
    if data.len() < 32 {
        return false;
    }
    let u = |o: usize| u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
    u(0) >= 32
        && u(4) == vk::PipelineCacheHeaderVersion::ONE.as_raw() as u32
        && u(8) == dev.props.vendor_id
        && u(12) == dev.props.device_id
        && data[16..32] == dev.props.pipeline_cache_uuid
}

impl PipelineDiskCache {
    /// Create the cache, seeded from `path` when it holds data for this
    /// device.
    pub fn new(dev: &Arc<GpuDevice>, path: Option<PathBuf>) -> Result<PipelineDiskCache, vk::Result> {
        if let Some(p) = &path {
            remove_stale_temporaries(p);
        }
        let sig = signature(&dev.props);
        let file = path.as_ref().and_then(|p| std::fs::read(p).ok());
        // The driver data is handed to the driver in the file buffer
        // itself: a game's cache is tens of MiB, not worth a copy.
        let mut data: &[u8] = file.as_deref().and_then(|f| decode(&sig, f)).filter(|d| header_matches(dev, d)).unwrap_or_default();
        if file.is_some() && data.is_empty() {
            log::info!("ps5gpu: pipeline cache: ignoring {:?} (driver, emulator or data mismatch)", path.as_ref().unwrap());
        }
        let info = vk::PipelineCacheCreateInfo::default().initial_data(data);
        // SAFETY: the data carries this device's header (checked above).
        let cache = match unsafe { dev.device.create_pipeline_cache(&info, None) } {
            Ok(c) => c,
            Err(e) => {
                if !data.is_empty() {
                    log::info!("ps5gpu: pipeline cache: the driver rejected the saved data ({e:?}); starting empty");
                    data = &[];
                }
                // SAFETY: empty cache.
                unsafe { dev.device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)? }
            }
        };
        if !data.is_empty() {
            log::info!("ps5gpu: pipeline cache: {} bytes from {:?}", data.len(), path.as_ref().unwrap());
        }
        Ok(PipelineDiskCache { dev: dev.clone(), cache, path, loaded_bytes: data.len(), unsaved: 0, saving: None })
    }

    /// Note a created pipeline; saves every few creations (in the
    /// background).
    pub fn note_created(&mut self) {
        self.unsaved += 1;
        if self.unsaved >= SAVE_INTERVAL {
            self.save_in_background();
        }
    }

    /// Start a background save unless one is still running (then the next
    /// creation tries again).
    pub fn save_in_background(&mut self) {
        if self.saving.as_ref().is_some_and(|h| !h.is_finished()) {
            return;
        }
        self.join_saving();
        self.unsaved = 0;
        let Some(path) = self.path.clone() else { return };
        let (dev, cache) = (self.dev.clone(), self.cache);
        // The cache stays valid until drop, which joins this thread first;
        // pipeline caches are internally synchronised, so creations may
        // continue meanwhile.
        self.saving = std::thread::Builder::new().name("pipeline-cache-wr".into()).spawn(move || write_cache(&dev, cache, &path)).ok();
    }

    /// Write the cache to disk now (atomically through a temporary file).
    pub fn save(&mut self) {
        self.join_saving();
        self.unsaved = 0;
        if let Some(path) = &self.path {
            write_cache(&self.dev, self.cache, path);
        }
    }

    /// [`Self::save`] when pipelines were created since the last save (the
    /// file is left alone otherwise: it may hold another build's data).
    pub fn save_changes(&mut self) {
        if self.unsaved > 0 {
            self.save();
        }
    }

    /// Join the background save, if any (bounded: a stuck writer is
    /// logged and detached). Must run before the cache is destroyed.
    fn join_saving(&mut self) {
        let Some(h) = self.saving.take() else { return };
        const JOIN_SAVE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
        let deadline = std::time::Instant::now() + JOIN_SAVE_TIMEOUT;
        while !h.is_finished() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        if h.is_finished() {
            let _ = h.join();
        } else {
            log::warn!("ps5gpu: the pipeline cache writer is still running after {JOIN_SAVE_TIMEOUT:?}; continuing without it");
        }
    }
}

/// The process id of a temporary save file named `<base>.tmp<pid>` (the
/// pipeline cache `<title>.bin` saves through `<title>.tmp<pid>`).
fn temporary_owner_of(base: &str, candidate: &std::path::Path) -> Option<u32> {
    let name = candidate.file_name()?.to_str()?;
    name.strip_prefix(base)?.strip_prefix(".tmp")?.parse().ok()
}

/// Delete the temporary save files of `path` left by processes that ended
/// during a save (each holds a whole cache: gigabytes over many runs).
fn remove_stale_temporaries(path: &std::path::Path) {
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        remove_stale_temporaries_of(path, stem);
    }
}

/// Delete the files `<base>.tmp<pid>` next to `path` of processes that are
/// gone.
pub(super) fn remove_stale_temporaries_of(path: &std::path::Path, base: &str) {
    let Some(dir) = path.parent() else { return };
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let candidate = e.path();
        let Some(pid) = temporary_owner_of(base, &candidate) else { continue };
        if pid != std::process::id() && !std::path::Path::new(&format!("/proc/{pid}")).exists() {
            let _ = std::fs::remove_file(&candidate);
        }
    }
}

/// Write the driver data of `cache` to `path` (through a temporary file).
fn write_cache(dev: &GpuDevice, cache: vk::PipelineCache, path: &std::path::Path) {
    // SAFETY: valid cache handle (see `save_in_background`).
    let Ok(data) = (unsafe { dev.device.get_pipeline_cache_data(cache) }) else { return };
    if data.is_empty() {
        return;
    }
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    let file = encode(&signature(&dev.props), &data);
    if std::fs::write(&tmp, &file).is_ok() && std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

impl Drop for PipelineDiskCache {
    fn drop(&mut self) {
        // Nothing created since the last save: the file holds it all (a
        // game's cache is hundreds of MiB).
        if self.unsaved > 0 {
            self.save();
        } else {
            self.join_saving();
        }
        // SAFETY: no pipeline creation is in flight.
        unsafe { self.dev.device.destroy_pipeline_cache(self.cache, None) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_temporaries_are_removed() {
        let dir = std::env::temp_dir().join(format!("rps5e-cache-tmp-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("PPSA00001.bin");
        let dead = dir.join("PPSA00001.tmp4000000000");
        let mine = dir.join(format!("PPSA00001.tmp{}", std::process::id()));
        let other = dir.join("PPSA00002.tmp4000000000");
        for f in [&path, &dead, &mine, &other] {
            std::fs::write(f, b"x").unwrap();
        }
        assert_eq!(temporary_owner_of("PPSA00001", &dead), Some(4_000_000_000));
        assert_eq!(temporary_owner_of("PPSA00001", &other), None);
        remove_stale_temporaries(&path);
        assert!(path.exists() && mine.exists() && other.exists());
        assert!(!dead.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_files_are_named_by_title() {
        assert!(valid_title_id("PPSA10112"));
        assert!(valid_title_id("UP9000-PPSA01341_00-ASTROBOTPS5GAME0"));
        assert!(!valid_title_id(""));
        assert!(!valid_title_id("../x"));
        assert!(!valid_title_id("A B"));
        let p = title_path(std::path::Path::new("/c"), "PPSA10112");
        assert_eq!(p, PathBuf::from("/c/rps5e/pipelines/PPSA10112.bin"));
    }

    #[test]
    fn cache_files_round_trip_and_reject_mismatches() {

        let props = vk::PhysicalDeviceProperties { vendor_id: 0x1002, device_id: 0x73bf, driver_version: 7, ..Default::default() };
        let sig = signature(&props);
        assert!(sig.starts_with("rps5ePC2:") && sig.ends_with('\n'));
        let payload = b"driver data".to_vec();
        let file = encode(&sig, &payload);
        assert_eq!(decode(&sig, &file), Some(&payload[..]));
        // Another driver version: another signature.
        let other = signature(&vk::PhysicalDeviceProperties { driver_version: 8, ..props });
        assert_eq!(decode(&other, &file), None);
        // A torn or corrupted payload fails the checksum.
        let mut bad = file.clone();
        *bad.last_mut().unwrap() ^= 1;
        assert_eq!(decode(&sig, &bad), None);
        assert_eq!(decode(&sig, &file[..sig.len() + 4]), None);
    }

    /// Two compute pipelines (each stores its own constant) and a way to
    /// tell whether a cache holds one: building it does not grow the
    /// cache's data.
    struct Pipelines {
        dev: Arc<GpuDevice>,
        set: vk::DescriptorSetLayout,
        layout: vk::PipelineLayout,
        modules: [vk::ShaderModule; 2],
    }

    impl Pipelines {
        fn new(dev: &Arc<GpuDevice>) -> Pipelines {
            let kernel = |k: u32| {
                let mut s = crate::tiler::Spv::new();
                let (main, x, _, _) = s.begin_main().unwrap();
                let v = s.c(k);
                let p = s.word_ptr(s.dst, x).unwrap();
                s.b.store(p, v, None, []).unwrap();
                s.finish(main, 64, 1).unwrap()
            };
            let d = &dev.device;
            let bindings = [0u32, 1].map(|i| {
                vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)
            });
            // SAFETY: valid create infos; destroyed in Drop.
            unsafe {
                let set = d
                    .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR).bindings(&bindings), None)
                    .unwrap();
                let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: (crate::tiler::ARGUMENT_COUNT * 4) as u32 }];
                let layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set)).push_constant_ranges(&ranges), None).unwrap();
                let modules = [0x1111, 0x2222].map(|k| d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&kernel(k)), None).unwrap());
                Pipelines { dev: dev.clone(), set, layout, modules }
            }
        }

        fn build(&self, cache: vk::PipelineCache, index: usize) {
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(self.modules[index]).name(c"main");
            let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(self.layout);
            // SAFETY: valid handles; the pipeline is destroyed at once.
            unsafe {
                let p = self.dev.device.create_compute_pipelines(cache, &[info], None).unwrap()[0];
                self.dev.device.destroy_pipeline(p, None);
            }
        }

        fn bytes(&self, cache: vk::PipelineCache) -> usize {
            // SAFETY: valid cache handle.
            unsafe { self.dev.device.get_pipeline_cache_data(cache) }.unwrap().len()
        }

        fn holds(&self, cache: vk::PipelineCache, index: usize) -> bool {
            let before = self.bytes(cache);
            self.build(cache, index);
            self.bytes(cache) == before
        }
    }

    impl Drop for Pipelines {
        fn drop(&mut self) {
            let d = &self.dev.device;
            // SAFETY: no pipeline of these objects is alive.
            unsafe {
                for m in self.modules {
                    d.destroy_shader_module(m, None);
                }
                d.destroy_pipeline_layout(self.layout, None);
                d.destroy_descriptor_set_layout(self.set, None);
            }
        }
    }

    /// The cache file through the objects that use it: a cache seeded from
    /// the file and extended saves both the file's pipelines and its own,
    /// the file's driver data is what the next cache starts with, and an
    /// invalid file is replaced.
    #[test]
    #[ignore]
    fn cache_file_keeps_pipelines_across_saves() {
        let dev = GpuDevice::create(false).expect("device");
        let objects = Pipelines::new(&dev);
        {
            // SAFETY: empty cache, destroyed below.
            let probe = unsafe { dev.device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) }.unwrap();
            let stored = !objects.holds(probe, 0);
            // SAFETY: no pipeline creation uses it.
            unsafe { dev.device.destroy_pipeline_cache(probe, None) };
            if !stored {
                eprintln!("skipped: the device does not keep pipelines in a pipeline cache");
                return;
            }
        }
        let dir = std::env::temp_dir().join(format!("rps5e-pipeline-cache-file-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("TEST00000.bin");
        let open = || PipelineDiskCache::new(&dev, Some(path.clone())).unwrap();
        // Each build is noted as the provider notes its pipelines: a cache
        // with nothing created since its last save leaves the file alone.
        {
            let mut writer = open();
            assert_eq!(writer.loaded_bytes, 0, "no file yet");
            objects.build(writer.cache, 0);
            writer.note_created();
        }
        {
            let mut extender = open();
            let file = std::fs::read(&path).unwrap();
            let payload = decode(&signature(&dev.props), &file).expect("a valid file");
            assert_eq!(extender.loaded_bytes, payload.len(), "the file's driver data seeds the cache");
            assert!(objects.holds(extender.cache, 0), "the cache file lost a pipeline");
            objects.build(extender.cache, 1);
            extender.note_created();
        }
        {
            let reloaded = open();
            assert!(objects.holds(reloaded.cache, 0) && objects.holds(reloaded.cache, 1), "a cache that saved over an unchanged file lost pipelines");
        }
        std::fs::write(&path, b"not a pipeline cache").unwrap();
        {
            let mut replacing = open();
            assert_eq!(replacing.loaded_bytes, 0, "an invalid file seeds nothing");
            objects.build(replacing.cache, 1);
            replacing.note_created();
        }
        {
            // Nothing created: the file stays as the last save left it.
            let before = std::fs::read(&path).unwrap();
            let idle = open();
            assert!(idle.loaded_bytes > 0);
            drop(idle);
            assert_eq!(std::fs::read(&path).unwrap(), before, "a cache with nothing new rewrote the file");
        }
        {
            let reloaded = open();
            assert!(objects.holds(reloaded.cache, 1), "an invalid cache file was not replaced by the saving object's pipelines");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
