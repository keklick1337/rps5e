// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Building a title's prewarm list before the game needs it.
//!
//! [`wait_before_guest`] runs at the start of `rps5e-cli run`, before any
//! guest code: it loads the title's prewarm list, translates the modules of
//! another translator again and builds every pipeline on all cores, showing
//! the progress in the log, the window title and as the compile overlay
//! over the black window (title, done/total, percent, time left, bar:
//! [`crate::osd::set_compile_progress`]). The provider that did it becomes
//! the back-end's provider ([`take_warm`]), so the game starts with every
//! pipeline of the list created. `RPS5E_PREWARM_WAIT=0`
//! (`--no-prewarm-wait`) leaves the list to the background as before: the
//! game starts at once and the list is built behind the work its draws
//! wait for.
//!
//! [`precompile`] does the same without a game (`rps5e-cli precompile`),
//! writing the module cache, the list and the pipeline cache; with `clear`
//! it first drops the title's pipeline cache and the cached modules of the
//! list, so everything is translated and built again ([`clear_title`]).
//! [`start_recompile`] does that from a running game (the in-game menu's
//! "Recompile shaders"), behind the same overlay.
//!
//! What a game will compile cannot be known before it runs: its shaders
//! are created at run time from data it loads, and each module is a
//! permutation of the program with the resources and state of the draws
//! that use it. The list is what a run collected; a first run without one
//! compiles during play (draws whose pipeline is not ready are skipped in
//! interactive play) and every module and pipeline it makes is added to
//! the list for the next start.

use super::prewarm::ListStatus;
use super::provider::{PrewarmProgress, PrewarmSummary, ShaderPipelines};
use super::ShaderCompiler;
use crate::device::GpuDevice;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// `RPS5E_PREWARM_WAIT=0`: the game starts without waiting for its
/// prewarm list (built in the background instead).
pub fn wait_enabled() -> bool {
    !std::env::var("RPS5E_PREWARM_WAIT").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off"))
}

/// The provider that built the list before the game started.
static WARM: Mutex<Option<(usize, ShaderPipelines)>> = Mutex::new(None);

/// The provider [`wait_before_guest`] built on `dev`, if any (taken once).
pub(crate) fn take_warm(dev: &Arc<GpuDevice>) -> Option<ShaderPipelines> {
    let mut w = WARM.lock().unwrap_or_else(|e| e.into_inner());
    match w.take() {
        Some((d, p)) if d == Arc::as_ptr(dev) as usize => Some(p),
        other => {
            *w = other;
            None
        }
    }
}

/// All cores (the driver compiles each pipeline on the thread creating it).
pub fn all_cores() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

/// `1m 05s`, `42s`.
pub fn format_duration(d: std::time::Duration) -> String {
    crate::osd::format_duration(d)
}

/// One progress line: `1234/5678 (21%), 1m 05s left`.
pub fn progress_line(p: &PrewarmProgress) -> String {
    let pct = if p.total == 0 { 100 } else { p.done * 100 / p.total };
    let eta = p.eta().map_or(String::new(), |t| format!(", {} left", format_duration(t)));
    format!("{}/{} ({pct}%){eta}", p.done, p.total)
}

/// Progress for the log (at most once a second, and the last report), the
/// window title and the compile overlay; false once the process shuts
/// down (the build stops).
struct Reporter {
    note: String,
    logged: Option<std::time::Instant>,
}

impl Reporter {
    fn new(note: &str) -> Reporter {
        crate::osd::set_status(Some("Compiling shaders...".into()));
        crate::osd::set_compile_progress(Some(crate::osd::CompileProgress { done: 0, total: 0, eta: None, note: note.into() }));
        Reporter { note: note.into(), logged: None }
    }

    fn report(&mut self, p: &PrewarmProgress) -> bool {
        let line = progress_line(p);
        if self.logged.is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(1)) || p.done == p.total {
            self.logged = Some(std::time::Instant::now());
            log::info!("ps5gpu: shader prewarm: {line}");
        }
        crate::osd::set_status(Some(format!("Compiling shaders: {line}")));
        crate::osd::set_compile_progress(Some(crate::osd::CompileProgress { done: p.done, total: p.total, eta: p.eta(), note: self.note.clone() }));
        !ps5vm::shutdown::requested()
    }
}

impl Drop for Reporter {
    fn drop(&mut self) {
        crate::osd::set_compile_progress(None);
        crate::osd::set_status(None);
    }
}

/// A provider with `compiler` and `workers` compile workers over the
/// pipeline cache `cache_path`, its prewarm list built.
fn build_with(
    dev: &Arc<GpuDevice>,
    memory: &'static crate::memory::GuestGpuMemory,
    compiler: Arc<dyn ShaderCompiler>,
    cache_path: PathBuf,
    workers: usize,
    progress: &mut dyn FnMut(&PrewarmProgress) -> bool,
) -> Result<(ShaderPipelines, PrewarmSummary), String> {
    let mut p = ShaderPipelines::with_workers(dev, memory, Some(compiler), Some(cache_path), Some(workers.max(1))).map_err(|e| format!("pipeline provider: {e:?}"))?;
    let summary = p.prewarm_now(progress);
    Ok((p, summary))
}

/// The SPIR-V compiler of `dev` with its module cache.
fn spirv_compiler(dev: &Arc<GpuDevice>, memory: &'static crate::memory::GuestGpuMemory) -> Arc<dyn ShaderCompiler> {
    Arc::new(crate::shader::SpirvCompiler::for_device(dev, memory))
}

/// Check the list next to `cache_path` before building it: an older
/// format is moved aside (nothing can be built from it). The list file
/// when it is one of this format; else why nothing can be built.
fn usable_list(cache_path: &Path) -> Result<PathBuf, String> {
    let list = super::prewarm::list_path(cache_path);
    match super::prewarm::list_status(&list) {
        ListStatus::Current => Ok(list),
        ListStatus::Missing => Err(format!("no shader prewarm list at {} yet (a game run makes it)", list.display())),
        ListStatus::Old(n) => {
            super::prewarm::retire_old_list(&list);
            Err(format!("the shader prewarm list {} was of the older format PW{n} without guest inputs; it was moved aside and the next game run makes a new one", list.display()))
        }
        ListStatus::Damaged => Err(format!("the shader prewarm list {} is damaged; the next game run makes a new one", list.display())),
    }
}

/// Build the running title's prewarm list before the guest starts (see
/// the module notes). Nothing to do without a list, a title id, the
/// shader compiler or a device.
pub fn wait_before_guest() {
    if !wait_enabled() || !super::prewarm::enabled() || !crate::shader::enabled() {
        return;
    }
    let Some(dev) = GpuDevice::global() else { return };
    let Some(cache_path) = super::cache::default_path(dev) else { return };
    let list = match usable_list(&cache_path) {
        Ok(l) => l,
        Err(e) => {
            log::info!("ps5gpu: shader prewarm: {e}; shaders compile during play this time");
            return;
        }
    };
    if !crate::env_headless() && crate::display_available() {
        crate::presenter().ensure_started(1920, 1080);
    }
    let workers = all_cores();
    log::info!("ps5gpu: shader prewarm: building {list:?} on {workers} workers before the game starts (RPS5E_PREWARM_WAIT=0 or --no-prewarm-wait: in the background instead)");
    let mut reporter = Reporter::new("The game starts when this finishes.");
    let built = build_with(dev, crate::memory::global(), spirv_compiler(dev, crate::memory::global()), cache_path, workers, &mut |p| reporter.report(p));
    drop(reporter);
    match built {
        Ok((mut p, summary)) => {
            log::info!("ps5gpu: shader prewarm: {summary}");
            // The game's share of the cores from now on.
            p.set_compile_workers(None);
            *WARM.lock().unwrap_or_else(|e| e.into_inner()) = Some((Arc::as_ptr(dev) as usize, p));
        }
        Err(e) => log::warn!("ps5gpu: shader prewarm: {e}"),
    }
}

/// Clear the shader and pipeline caches of the title whose pipeline cache
/// file is `cache_path`, so that its list is translated and built again:
/// the pipeline cache file is deleted, every module of the list is marked
/// for translation and dropped from `compiler`'s module cache. Returns the
/// modules of the list (Err: no list of this format; the pipeline cache is
/// deleted anyway).
pub fn clear_title(cache_path: &Path, compiler: &dyn ShaderCompiler) -> Result<usize, String> {
    match std::fs::remove_file(cache_path) {
        Ok(()) => log::info!("ps5gpu: removed the pipeline cache {}", cache_path.display()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("{}: {e}", cache_path.display())),
    }
    let list = usable_list(cache_path)?;
    let keys = super::prewarm::mark_for_recompile(&list)?;
    for &k in &keys {
        compiler.forget_cached(k);
    }
    log::info!("ps5gpu: {} modules of {} marked to translate again", keys.len(), list.display());
    Ok(keys.len())
}

/// Build the prewarm list kept next to the pipeline cache file
/// `cache_path` without running the game: modules translated again into
/// the module cache, every pipeline built into the pipeline cache, both
/// files and the list written. `clear`: everything is translated and
/// built again ([`clear_title`]).
pub fn precompile(cache_path: &Path, workers: usize, clear: bool, progress: &mut dyn FnMut(&PrewarmProgress) -> bool) -> Result<PrewarmSummary, String> {
    let dev = GpuDevice::global().ok_or("no usable Vulkan device")?;
    let memory = crate::memory::global();
    precompile_with(dev, memory, spirv_compiler(dev, memory), cache_path, workers, clear, progress)
}

/// [`precompile`] on `dev` with `compiler`.
pub fn precompile_with(
    dev: &Arc<GpuDevice>,
    memory: &'static crate::memory::GuestGpuMemory,
    compiler: Arc<dyn ShaderCompiler>,
    cache_path: &Path,
    workers: usize,
    clear: bool,
    progress: &mut dyn FnMut(&PrewarmProgress) -> bool,
) -> Result<PrewarmSummary, String> {
    if clear {
        clear_title(cache_path, &*compiler)?;
    } else {
        usable_list(cache_path)?;
    }
    let (p, summary) = build_with(dev, memory, compiler, cache_path.to_path_buf(), workers, progress)?;
    // Joins the workers and saves the list and the pipeline cache.
    drop(p);
    Ok(summary)
}

/// Translate every module of the list at `cache_path`'s list whose program
/// reads its own address as a relocatable module and check that it is
/// the same module at another address (`shader-cache relocation`): how
/// many of the title's programs serve every address they are loaded at.
/// Nothing is written.
pub fn relocation_report(cache_path: &Path, workers: usize) -> Result<String, String> {
    let dev = GpuDevice::global().ok_or("no usable Vulkan device")?;
    let compiler = spirv_compiler(dev, crate::memory::global());
    // Read as is (a list loaded for another translator is written back).
    let path = super::prewarm::list_path(cache_path);
    let file = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (_, modules, _, _) = super::prewarm::decode(&file).ok_or_else(|| format!("{}: not a shader prewarm list of this format", path.display()))?;
    let candidates: Vec<(u64, super::sources::ModuleSource)> =
        modules.iter().filter_map(|m| Some((m.key, super::sources::as_relocatable(m.source.as_deref()?)?))).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = Mutex::new(Vec::new());
    let start = std::time::Instant::now();
    std::thread::scope(|s| {
        for _ in 0..workers.max(1) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some((key, src)) = candidates.get(i) else { break };
                let t = std::time::Instant::now();
                let outcome = match super::sources::translate_uncached(&*compiler, *key, src) {
                    Ok(c) => Some(super::sources::proves_relocatable(&*compiler, *key, src, &c)),
                    Err(_) => None,
                };
                // The control: the plain translations at the two addresses
                // differ when the program's address reaches the module.
                let plain = super::sources::plain_translations_differ(&*compiler, *key, src);
                results.lock().unwrap_or_else(|e| e.into_inner()).push((src.program.address, outcome, t.elapsed(), plain));
            });
        }
    });
    let results = results.into_inner().unwrap_or_else(|e| e.into_inner());
    let proven = results.iter().filter(|r| r.1 == Some(true)).count();
    let dependent: Vec<u64> = results.iter().filter(|r| r.1 == Some(false)).map(|r| r.0).collect();
    let failed = results.iter().filter(|r| r.1.is_none()).count();
    let plain_differ = results.iter().filter(|r| r.3 == Some(true)).count();
    let mut out = format!(
        "{} modules, {} read their own address: {proven} relocatable, {} address dependent, {failed} not translated; {plain_differ} of them name the address in a plain module ({:.1} s)\n",
        modules.len(),
        candidates.len(),
        dependent.len(),
        start.elapsed().as_secs_f64()
    );
    for a in dependent.iter().take(16) {
        out += &format!("  address dependent: program {a:#x}\n");
    }
    Ok(out)
}

/// One module translated by [`translate_list`].
pub struct TranslatedModule {
    pub key: u64,
    /// `ShaderStage as u8`.
    pub stage: u8,
    pub address: u64,
    /// SPIR-V words (0: not translated).
    pub words: usize,
    /// A hash of the SPIR-V and the reflection (0: not translated).
    pub hash: u64,
    /// Wall time of the translation (ns).
    pub ns: u64,
    /// Where the time went ([`crate::shader::stats::LAST_NAMES`], ns).
    pub parts: [u64; 10],
    pub error: Option<String>,
}

/// Translate every module of the prewarm list file `list` (those `only`
/// accepts) past any module cache on `workers` threads, timing each and
/// hashing its output (`shader-cache translate`: the translator's speed
/// and whether a change altered its output). With `dump`, the SPIR-V of
/// each module and a list holding only it (and its compute pipelines) are
/// written there, `<key>.spv` and `<key>.prewarm`. Nothing else is written.
pub fn translate_list(list: &Path, workers: usize, only: &(dyn Fn(u64) -> bool + Sync), dump: Option<&Path>) -> Result<Vec<TranslatedModule>, String> {
    let dev = GpuDevice::global().ok_or("no usable Vulkan device")?;
    translate_list_with(&*spirv_compiler(dev, crate::memory::global()), list, workers, only, dump)
}

/// [`translate_list`] with `compiler` (no device needed).
pub fn translate_list_with(
    compiler: &dyn ShaderCompiler,
    list: &Path,
    workers: usize,
    only: &(dyn Fn(u64) -> bool + Sync),
    dump: Option<&Path>,
) -> Result<Vec<TranslatedModule>, String> {
    use std::hash::{Hash, Hasher};
    let file = std::fs::read(list).map_err(|e| format!("{}: {e}", list.display()))?;
    let (_, modules, _, compute) = super::prewarm::decode(&file).ok_or_else(|| format!("{}: not a shader prewarm list of this format", list.display()))?;
    drop(file);
    if let Some(d) = dump {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let chosen: Vec<&super::prewarm::ModuleRecord> = modules.iter().filter(|m| m.source.is_some() && only(m.key)).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = Mutex::new(Vec::with_capacity(chosen.len()));
    std::thread::scope(|s| {
        for _ in 0..workers.max(1) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(m) = chosen.get(i) else { break };
                let Some(src) = m.source.as_deref() else { continue };
                crate::shader::stats::set_last([0; 10]);
                let t = std::time::Instant::now();
                let r = super::sources::translate_uncached(compiler, m.key, src);
                let ns = t.elapsed().as_nanos() as u64;
                let mut out = TranslatedModule { key: m.key, stage: m.stage, address: src.program.address, words: 0, hash: 0, ns, parts: crate::shader::stats::last(), error: None };
                match r {
                    Ok(c) => {
                        let mut h = std::collections::hash_map::DefaultHasher::new();
                        c.spirv.hash(&mut h);
                        format!("{:?}", c.reflection).hash(&mut h);
                        out.words = c.spirv.len();
                        out.hash = h.finish();
                        if let Some(d) = dump {
                            let bytes: Vec<u8> = c.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
                            let _ = std::fs::write(d.join(format!("{:016x}.spv", m.key)), bytes);
                            let one = super::prewarm::ModuleRecord { key: m.key, stage: m.stage, spirv: Default::default(), reflection: Default::default(), fresh: false, source: m.source.clone() };
                            let pipelines: Vec<super::prewarm::ComputeRecord> = compute.iter().filter(|p| p.cs == m.key).cloned().collect();
                            let _ = std::fs::write(d.join(format!("{:016x}.prewarm", m.key)), super::prewarm::encode(super::prewarm::NO_TRANSLATOR, &[one], &[], &pipelines));
                        }
                    }
                    Err(e) => out.error = Some(format!("{e:?}")),
                }
                results.lock().unwrap_or_else(|e| e.into_inner()).push(out);
            });
        }
    });
    let mut results = results.into_inner().unwrap_or_else(|e| e.into_inner());
    results.sort_by_key(|r| r.key);
    Ok(results)
}

static RECOMPILING: AtomicBool = AtomicBool::new(false);
static RECOMPILE_RESULT: Mutex<Option<String>> = Mutex::new(None);

/// Whether a [`start_recompile`] runs now.
pub fn recompiling() -> bool {
    RECOMPILING.load(Ordering::Acquire)
}

/// The outcome of the last [`start_recompile`] (taken once).
pub fn take_recompile_result() -> Option<String> {
    RECOMPILE_RESULT.lock().unwrap_or_else(|e| e.into_inner()).take()
}

/// Clear the running title's shader and pipeline caches and build its list
/// again on all cores behind the compile overlay, on a thread of its own;
/// `done` runs with the outcome when it finishes (the caller pauses the
/// game around it).
/// The running game keeps the modules and pipelines it holds (the same
/// translation of the same inputs); the rebuilt caches serve every later
/// start. False when one runs already or there is no device or title.
pub fn start_recompile(done: impl FnOnce(&str) + Send + 'static) -> bool {
    let Some(dev) = GpuDevice::global() else { return false };
    let Some(cache_path) = super::cache::default_path(dev) else { return false };
    if RECOMPILING.swap(true, Ordering::AcqRel) {
        return false;
    }
    let dev = dev.clone();
    let spawned = std::thread::Builder::new().name("rps5e-recompile".into()).spawn(move || {
        let t = std::time::Instant::now();
        let mut reporter = Reporter::new("Rebuilding this game's shader cache; the game is paused.");
        let memory = crate::memory::global();
        let result = precompile_with(&dev, memory, spirv_compiler(&dev, memory), &cache_path, all_cores(), true, &mut |p| reporter.report(p));
        drop(reporter);
        let text = match result {
            Ok(s) => format!("Shader cache rebuilt: {s}"),
            Err(e) => format!("Shader cache cleared; nothing rebuilt: {e}"),
        };
        log::info!("ps5gpu: recompile shaders ({:.1} s): {text}", t.elapsed().as_secs_f64());
        *RECOMPILE_RESULT.lock().unwrap_or_else(|e| e.into_inner()) = Some(text.clone());
        RECOMPILING.store(false, Ordering::Release);
        done(&text);
    });
    if spawned.is_err() {
        RECOMPILING.store(false, Ordering::Release);
        return false;
    }
    true
}
