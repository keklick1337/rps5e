// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! On-disk cache of compiled modules: a permutation compiled once is read
//! back on later runs instead of being translated again, so a pipeline seen
//! before does not stall the frame that first needs it.
//!
//! The directory is `RPS5E_SHADER_CACHE` (`0`/`off` disables it) or
//! `$XDG_CACHE_HOME/rps5e/` (`~/.cache/rps5e/`, `%LOCALAPPDATA%\rps5e` on Windows), with one subdirectory
//! per shader-compiler source version and device feature set
//! (`spirv-<stamp>`), one file per permutation key. The source version is
//! a build-time hash of the compiler's sources (`build.rs`), so a rebuild
//! that leaves the compiler alone keeps its modules. An entry is the
//! SPIR-V words, the reflection and the code words the translation read
//! (kept with the module in the prewarm list); entries that fail to parse are ignored
//! (and rewritten). Entries are written by a background thread; module
//! directories unused for [`PRUNE_DAYS`] days are deleted.

use crate::pipeline::compiler::{ComponentKind, ReflectedBinding, ReflectedVariable, ShaderReflection};
use ash::vk;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

const MAGIC: u32 = 0x5053_5643; // "CVSP"
const VERSION: u32 = 5;

/// Module directories not used for this many days are deleted.
pub const PRUNE_DAYS: u64 = 14;

/// Mark `dir` as used now and delete the sibling module directories
/// (`spirv-*`) unused for [`PRUNE_DAYS`] days, on a background thread.
pub fn mark_used_and_prune(dir: &Path) {
    let dir = dir.to_path_buf();
    let _ = std::thread::Builder::new().name("shader-cache-gc".into()).spawn(move || {
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join("used"), b"");
        // Temporary files of runs that ended mid-write.
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                let old = e.metadata().and_then(|m| m.modified()).is_ok_and(|t| t.elapsed().is_ok_and(|a| a.as_secs() > 3600));
                if old && e.path().extension().is_some_and(|x| x.to_string_lossy().starts_with("tmp")) {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
        let Some(base) = dir.parent() else { return };
        let Ok(rd) = std::fs::read_dir(base) else { return };
        let limit = std::time::Duration::from_secs(PRUNE_DAYS * 86400);
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name();
            if p == dir || !name.to_string_lossy().starts_with("spirv-") || !p.is_dir() {
                continue;
            }
            let stamp = std::fs::metadata(p.join("used")).or_else(|_| std::fs::metadata(&p)).and_then(|m| m.modified());
            if stamp.is_ok_and(|t| t.elapsed().is_ok_and(|age| age > limit)) {
                log::info!("[shader] removing unused module cache {}", p.display());
                let _ = std::fs::remove_dir_all(&p);
            }
        }
    });
}

/// The cache directory for this compiler build and `features` (None when
/// disabled or no cache location exists).
pub fn default_dir(features: &super::Features) -> Option<PathBuf> {
    let base = match std::env::var("RPS5E_SHADER_CACHE") {
        Ok(v) if v == "0" || v.eq_ignore_ascii_case("off") => return None,
        Ok(v) if !v.is_empty() => PathBuf::from(v),
        _ => ps5vm::appenv::app_dir(&ps5vm::appenv::cache_base()?),
    };
    Some(base.join(format!("spirv-{:016x}", super::translator_stamp(features))))
}

/// The identity of the translator on a device with `features`: the
/// compiler sources, the features and every switch that changes what it
/// emits (the module cache directory and the prewarm list are keyed by it).
pub fn translator_stamp(features: &super::Features) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    env!("PS5GPU_SHADER_SOURCE_VERSION").hash(&mut h);
    (features.barycentric, features.demote, features.subgroup_size, features.sample_offset).hash(&mut h);
    (features.image_int64_atomics, features.shared_int64_atomics, features.compute_derivatives, features.compute_derivatives_linear).hash(&mut h);
    (features.preserve_inf_nan, features.denorm_preserve, features.f16_rte).hash(&mut h);
    (features.denorm_flush_to_zero32, features.no_float64).hash(&mut h);
    (features.device_clock, features.subgroup_clock).hash(&mut h);
    features.amd_proprietary.hash(&mut h);
    features.emit.hash(&mut h);
    super::exec_run_minimum().hash(&mut h);
    super::storage_resinfo_mips_enabled().hash(&mut h);
    super::float_rules().hash(&mut h);
    super::log_series_enabled().hash(&mut h);
    super::mesh::wave_layout(32).hash(&mut h);
    super::vertex_wave_info_enabled().hash(&mut h);
    super::image::const_texel_offsets_enabled().hash(&mut h);
    super::emit::data_cache_invalidate_enabled().hash(&mut h);
    (features.subgroup64_compute, features.max_compute_workgroup_subgroups, features.max_shared_bytes, super::wave_split::enabled(), super::wave_split::min_vgprs(), super::wave_split::lane_slots_enabled(), super::wave_split::ballot_riders_enabled()).hash(&mut h);
    VERSION.hash(&mut h);
    h.finish()
}

fn entry(dir: &Path, key: u64) -> PathBuf {
    dir.join(format!("{key:016x}.spv"))
}

struct Writer(Vec<u32>);

impl Writer {
    fn u(&mut self, v: u32) {
        self.0.push(v);
    }
    fn vars(&mut self, v: &[ReflectedVariable]) {
        self.u(v.len() as u32);
        for x in v {
            let kind = match x.kind {
                ComponentKind::Float => 0,
                ComponentKind::Uint => 1,
                ComponentKind::Sint => 2,
            };
            self.0.extend([x.location, x.components, kind, x.flat as u32]);
        }
    }
}

struct Reader<'a>(&'a [u32]);

impl Reader<'_> {
    fn u(&mut self) -> Option<u32> {
        let (&v, rest) = self.0.split_first()?;
        self.0 = rest;
        Some(v)
    }
    fn vars(&mut self) -> Option<Vec<ReflectedVariable>> {
        let n = self.u()?;
        let mut out = Vec::new();
        for _ in 0..n {
            let (location, components, kind, flat) = (self.u()?, self.u()?, self.u()?, self.u()?);
            let kind = match kind {
                0 => ComponentKind::Float,
                1 => ComponentKind::Uint,
                2 => ComponentKind::Sint,
                _ => return None,
            };
            out.push(ReflectedVariable { location, components, kind, flat: flat != 0 });
        }
        Some(out)
    }
}

/// The words of an entry.
fn encode(spirv: &[u32], r: &ShaderReflection, reads: &[(u64, u32)]) -> Vec<u32> {
    let mut w = Writer(vec![MAGIC, VERSION]);
    let name = r.entry_point.as_bytes();
    w.u(name.len() as u32);
    for c in name.chunks(4) {
        let mut b = [0u8; 4];
        b[..c.len()].copy_from_slice(c);
        w.u(u32::from_le_bytes(b));
    }
    w.u(r.bindings.len() as u32);
    for b in &r.bindings {
        w.0.extend([b.binding, b.descriptor_type.as_raw() as u32, b.count]);
    }
    match r.push_constants {
        Some((o, s)) => w.0.extend([1, o, s]),
        None => w.0.extend([0, 0, 0]),
    }
    w.vars(&r.vertex_inputs);
    w.vars(&r.outputs);
    w.0.extend([r.writes_point_size as u32 | (r.writes_viewport_index as u32) << 1, r.clip_distances, r.writes_depth as u32]);
    w.0.extend(r.local_size);
    w.u(r.private_lds_bytes);
    w.u(r.required_subgroup_size);
    w.u(r.own_buffer_checks as u32);
    w.u(reads.len() as u32);
    for &(a, v) in reads {
        w.0.extend([a as u32, (a >> 32) as u32, v]);
    }
    w.u(spirv.len() as u32);
    w.0.extend_from_slice(spirv);
    w.0
}

type Entry = (Vec<u32>, ShaderReflection, Vec<(u64, u32)>);

fn decode(words: &[u32]) -> Option<Entry> {
    let mut r = Reader(words);
    if r.u()? != MAGIC || r.u()? != VERSION {
        return None;
    }
    let len = r.u()? as usize;
    let mut name = Vec::new();
    for _ in 0..len.div_ceil(4) {
        name.extend(r.u()?.to_le_bytes());
    }
    name.truncate(len);
    let entry_point = String::from_utf8(name).ok()?;
    let n = r.u()?;
    let mut bindings = Vec::new();
    for _ in 0..n {
        let (binding, ty, count) = (r.u()?, r.u()?, r.u()?);
        bindings.push(ReflectedBinding { binding, descriptor_type: vk::DescriptorType::from_raw(ty as i32), count });
    }
    let (has, o, s) = (r.u()?, r.u()?, r.u()?);
    let push_constants = (has != 0).then_some((o, s));
    let vertex_inputs = r.vars()?;
    let outputs = r.vars()?;
    let (flags, clip_distances, writes_depth) = (r.u()?, r.u()?, r.u()? != 0);
    let (writes_point_size, writes_viewport_index) = (flags & 1 != 0, flags & 2 != 0);
    let local_size = [r.u()?, r.u()?, r.u()?];
    let private_lds_bytes = r.u()?;
    let required_subgroup_size = r.u()?;
    let own_buffer_checks = r.u()? != 0;
    let n = r.u()? as usize;
    if n > r.0.len() / 3 {
        return None;
    }
    let reads = (0..n).map(|_| Some((r.u()? as u64 | (r.u()? as u64) << 32, r.u()?))).collect::<Option<Vec<_>>>()?;
    let n = r.u()? as usize;
    if r.0.len() != n {
        return None;
    }
    let reflection = ShaderReflection {
        entry_point,
        bindings,
        push_constants,
        vertex_inputs,
        outputs,
        writes_point_size,
        clip_distances,
        writes_viewport_index,
        writes_depth,
        local_size,
        private_lds_bytes,
        required_subgroup_size,
        own_buffer_checks,
    };
    Some((r.0.to_vec(), reflection, reads))
}

/// The cached module of `key`, if any.
pub fn load(dir: &Path, key: u64) -> Option<Entry> {
    let bytes = std::fs::read(entry(dir, key)).ok()?;
    if bytes.len() % 4 != 0 {
        return None;
    }
    let words: Vec<u32> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    decode(&words)
}

/// Delete the cached module of `key`.
pub fn remove(dir: &Path, key: u64) {
    let _ = std::fs::remove_file(entry(dir, key));
}

/// Store the module of `key` (written to a temporary file, then renamed,
/// so readers never see a partial entry).
pub fn store(dir: &Path, key: u64, spirv: &[u32], reflection: &ShaderReflection, reads: &[(u64, u32)]) {
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let bytes: Vec<u8> = encode(spirv, reflection, reads).iter().flat_map(|w| w.to_le_bytes()).collect();
    let path = entry(dir, key);
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&tmp, bytes).is_ok() && std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// [`store`] on the background writer thread (the compiling thread does
/// not wait for the file system).
pub fn store_async(dir: &Path, key: u64, spirv: Vec<u32>, reflection: ShaderReflection, reads: Vec<(u64, u32)>) {
    type Job = (PathBuf, u64, Vec<u32>, ShaderReflection, Vec<(u64, u32)>);
    static WRITER: std::sync::OnceLock<Option<std::sync::Mutex<std::sync::mpsc::Sender<Job>>>> = std::sync::OnceLock::new();
    let writer = WRITER.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("shader-cache-wr".into())
            .spawn(move || {
                for (dir, key, spirv, r, reads) in rx {
                    store(&dir, key, &spirv, &r, &reads);
                }
            })
            .ok()
            .map(|_| std::sync::Mutex::new(tx))
    });
    match writer {
        Some(tx) => {
            let job = (dir.to_path_buf(), key, spirv, reflection, reads);
            if let Err(std::sync::mpsc::SendError((dir, key, spirv, r, reads))) = tx.lock().unwrap_or_else(|e| e.into_inner()).send(job) {
                store(&dir, key, &spirv, &r, &reads);
            }
        }
        None => store(dir, key, &spirv, &reflection, &reads),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_round_trip() {
        let r = ShaderReflection {
            entry_point: "main".into(),
            bindings: vec![ReflectedBinding { binding: 3, descriptor_type: vk::DescriptorType::STORAGE_BUFFER, count: 7 }],
            push_constants: Some((0, 128)),
            vertex_inputs: vec![ReflectedVariable { location: 1, components: 4, kind: ComponentKind::Sint, flat: true }],
            outputs: vec![ReflectedVariable { location: 0, components: 4, kind: ComponentKind::Float, flat: false }],
            writes_point_size: true,
            clip_distances: 2,
            writes_viewport_index: true,
            writes_depth: false,
            local_size: [64, 1, 1],
            private_lds_bytes: 12,
            required_subgroup_size: 64,
            own_buffer_checks: true,
        };
        let spirv = vec![0x0723_0203, 1, 2, 3];
        let reads = vec![(0x1_0000_0040, 7), (0x1_0000_0044, 9)];
        let (s, back, rd) = decode(&encode(&spirv, &r, &reads)).unwrap();
        assert_eq!(s, spirv);
        assert_eq!(back, r);
        assert_eq!(rd, reads);
        let dir = std::env::temp_dir().join(format!("rps5e-spirv-cache-test-{}", std::process::id()));
        store(&dir, 0x1234, &spirv, &r, &reads);
        assert_eq!(load(&dir, 0x1234), Some((spirv, r, reads)));
        assert_eq!(load(&dir, 0x1235), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
