// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Shader objects the game creates, and the prewarm work of their programs.
//!
//! A game creates a shader object (`sceAgcCreateShader`) before the draws
//! and dispatches that run it. The object names the program's code, but
//! not the module: a module is a permutation of the program with the
//! resources and state of the draws that use it, so nothing can be
//! translated for it yet. The prewarm list of an earlier run holds those
//! permutations, keyed by program address. While the list builds on the
//! compile workers (in the background during play, or after the
//! translator changed), the modules and pipelines of a program the game
//! has just created move to the front of the list ([`note`]): they are
//! the ones its next draws need, so they are ready before those draws
//! instead of in list order. Urgent work (draws waiting now) still goes
//! first.
//!
//! The provider that queues the list registers each queued job under the
//! program addresses it builds ([`register`]) and installs the pool's
//! [`PromoteHandle`] ([`install`]); a program created before its jobs are
//! queued is promoted when they are.
//!
//! What the object does determine is the program itself: its code, its
//! stage, the user-data registers it receives (RSRC2 in its register
//! table) and its wave size. A worker thread ([`start`]) decodes and
//! plans each created program ([`note_program`]) and keeps the result;
//! the front end of the first draw or dispatch that runs the program
//! takes it ([`built`]) after comparing the code bytes with guest memory,
//! instead of decoding and planning inside that draw. A program whose
//! draw comes first, or whose draw-time registers differ from the
//! object's, is built by the draw as before.
//!
//! `RPS5E_CREATE_TIME_SHADERS=0` turns both off.

use super::frontend::{ProgramKey, ShaderFrontend, StageProgram};
use super::pool::PromoteHandle;
use crate::memory::GuestGpuMemory;
use ps5shader::resources::ShaderStage;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// `RPS5E_CREATE_TIME_SHADERS` (default on): prewarm work of a program
/// moves to the front of the list when the game creates its shader.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_CREATE_TIME_SHADERS").is_ok_and(|v| v.trim() == "0"))
}

#[derive(Default)]
struct Registry {
    promote: Option<PromoteHandle>,
    /// Queued prewarm jobs by the program addresses they build.
    tags: HashMap<u64, Vec<u64>>,
    /// Programs created so far.
    created: HashSet<u64>,
}

static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);
static CREATED: AtomicU64 = AtomicU64::new(0);
static PROMOTED: AtomicU64 = AtomicU64::new(0);

/// Jobs to promote, and the handle that does it (called without the lock).
type Promotion = Option<(PromoteHandle, Vec<u64>)>;

impl Registry {
    fn note(&mut self, address: u64) -> Promotion {
        if !self.created.insert(address) {
            return None;
        }
        CREATED.fetch_add(1, Ordering::Relaxed);
        let p = self.promote.clone()?;
        Some((p, self.tags.remove(&address).unwrap_or_default()))
    }

    fn install(&mut self, promote: PromoteHandle) {
        self.tags.clear();
        self.promote = Some(promote);
    }

    fn register(&mut self, tag: u64, addresses: &[u64]) -> Promotion {
        let p = self.promote.clone()?;
        if addresses.iter().any(|a| self.created.contains(a)) {
            return Some((p, vec![tag]));
        }
        for &a in addresses.iter().filter(|&&a| a != 0) {
            self.tags.entry(a).or_default().push(tag);
        }
        None
    }
}

fn with(f: impl FnOnce(&mut Registry) -> Promotion) {
    let p = {
        let mut g = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
        f(g.get_or_insert_with(Registry::default))
    };
    if let Some((promote, tags)) = p {
        for tag in tags {
            if promote.promote(tag) {
                PROMOTED.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// The game created a shader object whose program starts at `address`.
pub fn note(address: u64) {
    if enabled() && address != 0 {
        with(|r| r.note(address));
    }
}

/// Use `promote` for the jobs registered from now on (a new provider's
/// pool); jobs registered for an earlier pool are forgotten.
pub fn install(promote: PromoteHandle) {
    if enabled() {
        with(|r| {
            r.install(promote);
            None
        });
    }
}

/// A prewarm job tagged `tag` that builds work of the programs at
/// `addresses` is queued: promoted now when one of them was created.
pub fn register(tag: u64, addresses: &[u64]) {
    if enabled() {
        with(|r| r.register(tag, addresses));
    }
}

/// `(shader objects created, prewarm jobs moved to the front)`.
pub fn counts() -> (u64, u64) {
    (CREATED.load(Ordering::Relaxed), PROMOTED.load(Ordering::Relaxed))
}

/// The stage of a created shader object whose program can be built ahead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CreatedStage {
    /// Compute; the wave word is the object's dispatch modifier.
    Compute,
    /// Pixel; the wave word is the object's SPI_PS_IN_CONTROL.
    Pixel,
    /// The vertex program of merged ES/GS stages; the wave word is the
    /// object's VGT_SHADER_STAGES_EN.
    Vertex,
}

/// RSRC2.USER_SGPR (with the extra high bit).
fn user_sgpr_count(rsrc2: u32) -> u32 {
    ((rsrc2 >> 1) & 0x1F) | (((rsrc2 >> 27) & 1) << 5)
}

/// The program lookups the draws and dispatches of a created shader make
/// (the same values the recorder derives from the register state the
/// object's tables set).
fn keys(address: u64, stage: CreatedStage, rsrc2: u32, wave_word: u32) -> Vec<(ProgramKey, ShaderStage)> {
    let count = user_sgpr_count(rsrc2);
    let key = |stage_tag, user_data_base, user_data_count, wave_size| ProgramKey { address, stage_tag, user_data_base, user_data_count, wave_size, continuation: 0 };
    match stage {
        CreatedStage::Compute => {
            let wave = if wave_word & (1 << 15) != 0 { 32 } else { 64 };
            vec![(key(3, 0, count.min(16), wave), ShaderStage::Compute)]
        }
        CreatedStage::Pixel => vec![(key(2, 0, count.min(32), super::frontend::pixel_wave_size(wave_word)), ShaderStage::Pixel)],
        CreatedStage::Vertex => {
            // Per vertex (user data from s8), and as a primitive subgroup
            // (s[0:1] = the user-data address, user SGPRs from s8).
            let count = count.min(32);
            let subgroup_wave = if wave_word & (1 << 22) != 0 { 32 } else { 64 };
            vec![
                (key(1, 8, count, super::provider::plain_vertex_wave(wave_word)), ShaderStage::Vertex),
                (key(1, 0, 8 + count, subgroup_wave), ShaderStage::Vertex),
            ]
        }
    }
}

/// Programs waiting for the worker (each key once).
#[derive(Default)]
struct Queue {
    jobs: VecDeque<(ProgramKey, ShaderStage)>,
    queued: HashSet<ProgramKey>,
}

/// Built programs nobody dropped yet, oldest first.
#[derive(Default)]
struct Store {
    programs: HashMap<ProgramKey, Arc<StageProgram>>,
    order: VecDeque<ProgramKey>,
}

/// Programs kept and jobs queued at most: a game that creates more
/// shaders than it draws with loses the oldest.
const STORE_LIMIT: usize = 16384;
const QUEUE_LIMIT: usize = 16384;

static QUEUE: Mutex<Option<Queue>> = Mutex::new(None);
static QUEUE_READY: Condvar = Condvar::new();
static STORE: Mutex<Option<Store>> = Mutex::new(None);
static STOP: AtomicBool = AtomicBool::new(false);
static BUILT: AtomicU64 = AtomicU64::new(0);
static USED: AtomicU64 = AtomicU64::new(0);

/// The game created a shader object of `stage` whose program starts at
/// `address`, with `rsrc2` in its register table and `wave_word` (see
/// [`CreatedStage`]): its program is decoded and planned on the worker.
pub fn note_program(address: u64, stage: CreatedStage, rsrc2: u32, wave_word: u32) {
    if !enabled() || address == 0 {
        return;
    }
    let mut g = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    let q = g.get_or_insert_with(Queue::default);
    for (key, stage) in keys(address, stage, rsrc2, wave_word) {
        if q.jobs.len() < QUEUE_LIMIT && q.queued.insert(key) {
            q.jobs.push_back((key, stage));
        }
    }
    drop(g);
    QUEUE_READY.notify_one();
}

/// Decode and plan `key` with `front` and keep the program. A program
/// already kept whose code did not change stays.
fn build(front: &ShaderFrontend, key: ProgramKey, stage: ShaderStage) {
    let kept = STORE.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|s| s.programs.get(&key).cloned());
    if kept.is_some_and(|p| front.code_equals(key.address, &p.code)) {
        return;
    }
    let r = front.build_key(&key, stage);
    let Ok(program) = r else { return };
    BUILT.fetch_add(1, Ordering::Relaxed);
    let mut g = STORE.lock().unwrap_or_else(|e| e.into_inner());
    let s = g.get_or_insert_with(Store::default);
    if s.programs.insert(key, program).is_none() {
        s.order.push_back(key);
        while s.order.len() > STORE_LIMIT {
            if let Some(old) = s.order.pop_front() {
                s.programs.remove(&old);
            }
        }
    }
}

/// The program built for `key` at its shader's creation. The caller
/// compares its code with guest memory before using it.
pub(crate) fn built(key: &ProgramKey) -> Option<Arc<StageProgram>> {
    if !enabled() {
        return None;
    }
    STORE.lock().unwrap_or_else(|e| e.into_inner()).as_ref()?.programs.get(key).cloned()
}

/// A draw or dispatch used a program built at creation.
pub(crate) fn note_used() {
    USED.fetch_add(1, Ordering::Relaxed);
}

/// `(programs built at creation, of them taken by a draw or dispatch)`.
pub fn program_counts() -> (u64, u64) {
    (BUILT.load(Ordering::Relaxed), USED.load(Ordering::Relaxed))
}

/// Start the worker that builds created programs from `memory` (once per
/// process: the live provider's memory).
pub(crate) fn start(memory: &'static GuestGpuMemory) {
    static STARTED: std::sync::Once = std::sync::Once::new();
    if !enabled() {
        return;
    }
    STARTED.call_once(|| {
        let spawned = std::thread::Builder::new().name("shader-create".into()).spawn(move || {
            let front = ShaderFrontend::new(memory);
            loop {
                let job = {
                    let mut g = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
                    loop {
                        if STOP.load(Ordering::Acquire) {
                            return;
                        }
                        let q = g.get_or_insert_with(Queue::default);
                        if let Some(job) = q.jobs.pop_front() {
                            q.queued.remove(&job.0);
                            break job;
                        }
                        g = QUEUE_READY.wait(g).unwrap_or_else(|e| e.into_inner());
                    }
                };
                build(&front, job.0, job.1);
            }
        });
        if let Err(e) = spawned {
            log::warn!("ps5gpu: no thread for programs of created shaders: {e}");
            return;
        }
        ps5vm::shutdown::on_shutdown(ps5vm::shutdown::ORDER_GPU, "created shader programs", || {
            STOP.store(true, Ordering::Release);
            // Under the queue lock: the worker is either before its check
            // or already waiting.
            let _g = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
            QUEUE_READY.notify_all();
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(p: Promotion, log: &Mutex<Vec<u64>>) {
        if let Some((h, tags)) = p {
            for t in tags {
                if h.promote(t) {
                    log.lock().unwrap().push(t);
                }
            }
        }
    }

    /// Jobs of a program created before they were queued are promoted
    /// when queued; jobs queued first are promoted when the program is
    /// created; other programs' jobs stay where they are; a program
    /// created twice promotes once.
    #[test]
    fn created_programs_move_their_prewarm_jobs_forward() {
        let promoted = Arc::new(Mutex::new(Vec::new()));
        let handle = || PromoteHandle::from_fn(|_| true);
        let mut r = Registry::default();
        // Nothing to promote before a pool is installed.
        assert!(r.note(0x1000).is_none());
        r.install(handle());
        run(r.register(1, &[0x1000, 0]), &promoted);
        run(r.register(2, &[0x2000, 0x3000]), &promoted);
        run(r.register(3, &[0x4000]), &promoted);
        assert_eq!(*promoted.lock().unwrap(), vec![1]);
        run(r.note(0x3000), &promoted);
        run(r.note(0x3000), &promoted);
        assert_eq!(*promoted.lock().unwrap(), vec![1, 2]);
        // A new provider's pool forgets the old pool's jobs.
        r.install(handle());
        promoted.lock().unwrap().clear();
        run(r.note(0x4000), &promoted);
        assert!(promoted.lock().unwrap().is_empty());
        run(r.register(4, &[0x4000]), &promoted);
        assert_eq!(*promoted.lock().unwrap(), vec![4]);
    }

    /// The lookups of a created shader are the ones its draws make.
    #[test]
    fn created_shaders_name_the_lookups_of_their_draws() {
        // USER_SGPR = 5, wave64 dispatch.
        let k = keys(0x1000, CreatedStage::Compute, 5 << 1, 0);
        assert_eq!(k.len(), 1);
        assert_eq!((k[0].0.stage_tag, k[0].0.user_data_base, k[0].0.user_data_count, k[0].0.wave_size), (3, 0, 5, 64));
        // More than 16 user SGPRs are not passed to a compute program.
        let k = keys(0x1000, CreatedStage::Compute, 1 << 27, 1 << 15);
        assert_eq!((k[0].0.user_data_count, k[0].0.wave_size), (16, 32));
        let k = keys(0x1000, CreatedStage::Pixel, 3 << 1, 1 << 15);
        assert_eq!((k[0].0.stage_tag, k[0].0.user_data_count, k[0].0.wave_size), (2, 3, 32));
        let k = keys(0x1000, CreatedStage::Vertex, 4 << 1, 1 << 22);
        assert_eq!(k.len(), 2);
        assert_eq!((k[0].0.user_data_base, k[0].0.user_data_count, k[0].0.wave_size), (8, 4, 32));
        assert_eq!((k[1].0.user_data_base, k[1].0.user_data_count, k[1].0.wave_size), (0, 12, 32));
    }

    /// A program built at creation is the one the front end returns for
    /// the same lookup; after the code changed the front end builds its
    /// own.
    #[test]
    fn a_draw_takes_the_program_built_at_creation() {
        use crate::pipeline::testing::asm;
        let memory = crate::testutil::test_memory();
        let (view, _) = crate::testutil::guest_mapping(memory, 1 << 16);
        let write = |words: &[u32]| {
            let bytes = asm::bytes(words);
            // SAFETY: the fresh guest view mapped above.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), view as *mut u8, bytes.len()) };
        };
        write(&[asm::v_mov_from_sgpr(0, 0), asm::S_ENDPGM]);
        let (key, stage) = keys(view, CreatedStage::Pixel, 2 << 1, 0)[0];
        let worker = ShaderFrontend::new(memory);
        build(&worker, key, stage);
        let kept = built(&key).expect("the program was built");
        // Created again with the same code: the kept program stays.
        build(&worker, key, stage);
        assert!(Arc::ptr_eq(&kept, &built(&key).unwrap()));
        let mut front = ShaderFrontend::new(memory);
        let used = front.program(view, stage, key.user_data_base, key.user_data_count, key.wave_size).unwrap();
        assert!(Arc::ptr_eq(&kept, &used));
        // Other registers: another program.
        let other = front.program(view, stage, 0, key.user_data_count + 1, key.wave_size).unwrap();
        assert!(!Arc::ptr_eq(&kept, &other));
        // New code at the address: a fresh front end decodes it.
        write(&[asm::v_mov_from_sgpr(0, 1), asm::v_mov_from_sgpr(1, 0), asm::S_ENDPGM]);
        let mut fresh = ShaderFrontend::new(memory);
        let rebuilt = fresh.program(view, stage, key.user_data_base, key.user_data_count, key.wave_size).unwrap();
        assert!(!Arc::ptr_eq(&kept, &rebuilt));
        assert_ne!(rebuilt.code, kept.code);
    }
}
