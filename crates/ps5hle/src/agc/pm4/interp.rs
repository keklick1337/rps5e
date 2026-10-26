// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Per-queue PM4 command-stream interpreter.
//!
//! Executes one submission slice at a time: walks type-3 packets, keeps the
//! queue's register banks, follows INDIRECT_BUFFER calls/chains, conditional
//! execution and branches, and performs the memory side effects of label
//! writes (WRITE_DATA, RELEASE_MEM, EVENT_WRITE_EOP/EOS), DMA/COPY_DATA,
//! atomics and semaphores. GPU interrupts are delivered to equeues that
//! registered the graphics filter (-14) with `ident` = the queue's event id
//! and `data` = the packet's context id.
//!
//! Labels are written at parse time. Draws and dispatches are decoded into
//! a [`ps5gpu::render::DrawPacket`] / [`ps5gpu::render::DispatchPacket`],
//! resolved against the register banks and handed to the queue's render
//! executor (the GPU back-end). With an executor, GPU interrupts are
//! delivered at end of pipe (after the work recorded before them completed
//! on the GPU); without one they take effect at parse time.
//!
//! A packet that cannot complete yet (unmet WAIT_REG_MEM, WaitFlipDone, a
//! rewind that is not valid, an empty semaphore, a failed compare-swap)
//! suspends the submission at that packet; the queue retries it later and
//! the packet re-executes from the start.

use super::{custom, length, make, nop, op, packet_size, predicated};
use crate::agc::mem;
use crate::sync::equeue;
use ps5gpu::render::regfile::{Bank, ContextState, RegisterFile, Write};
use ps5gpu::render::{DirtyBanks, DispatchPacket, DrawKind, DrawPacket, GdsTransfer, IndirectRegisters, RegisterBanks, RenderExecutor, ResolvedDraw};
use ps5gpu::pipeline::lookahead;
use ps5gpu::slices::{InstanceSource, SliceExecutor};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Compute queues' interrupt idents start here (graphics uses 0).
pub const COMPUTE_QUEUE_BASE: u64 = 0x20;
const CONSTANT_RAM_DWORDS: usize = 0x3000;
const MAX_DEPTH: usize = 16;
const DMA_SKIPPED_DESTINATION: u32 = 0x3022C;
const MARKER_MAGIC: u32 = 0x6875_0000;
const GDS_BYTES: usize = 0x10000;

/// Process-wide counters (diagnostics).
pub static PACKETS: AtomicU64 = AtomicU64::new(0);
pub static DRAWS: AtomicU64 = AtomicU64::new(0);
pub static DISPATCHES: AtomicU64 = AtomicU64::new(0);
pub static FLIPS: AtomicU64 = AtomicU64::new(0);
pub static LABEL_WRITES: AtomicU64 = AtomicU64::new(0);
pub static INTERRUPTS: AtomicU64 = AtomicU64::new(0);

/// `RPS5E_FLIP_EOP_EVENT=0`: a flip packet raises no graphics end-of-pipe
/// event (the earlier behaviour).
fn flip_eop_event_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_FLIP_EOP_EVENT").is_ok_and(|v| v == "0"))
}
/// `RPS5E_FLIP_EOP_LATCH=0`: every flip raises the graphics end-of-pipe
/// event, also on a queue that raises its own release interrupts (the
/// earlier behaviour).
fn flip_eop_latch_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_FLIP_EOP_LATCH").is_ok_and(|v| v == "0"))
}

/// Flips a queue that delivered a release interrupt may take without
/// raising their end-of-pipe event; the next one clears the latch.
const FLIP_EOP_LATCH_LIMIT: u32 = 4;

/// Flips a queue took since its last delivered release interrupt (None:
/// no release interrupt since the latch last cleared). A latched queue
/// raises the end-of-pipe event from its own releases, so its flips do
/// not, until the count passes [`FLIP_EOP_LATCH_LIMIT`] (a title that
/// stopped raising them recovers).
type FlipEopLatch = Arc<Mutex<Option<u32>>>;

/// A release interrupt reached a registered event of the latch's queue.
fn latch_release_interrupt(latch: &FlipEopLatch) {
    if flip_eop_latch_enabled() {
        *latch.lock().unwrap_or_else(|e| e.into_inner()) = Some(0);
    }
}

/// Whether the end-of-pipe event of a flip on the latch's queue is raised.
fn flip_eop_event_due(latch: &FlipEopLatch) -> bool {
    let mut g = latch.lock().unwrap_or_else(|e| e.into_inner());
    match g.as_mut() {
        None => true,
        Some(n) => {
            *n += 1;
            if *n <= FLIP_EOP_LATCH_LIMIT {
                return false;
            }
            *g = None;
            true
        }
    }
}

/// `RPS5E_WAIT64_NARROW=0`: a WAIT_REG_MEM_64 always reads 8 bytes at an
/// 8-byte aligned address, even when its mask has no high bits (the
/// earlier behaviour).
fn wait64_narrow_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_WAIT64_NARROW").is_ok_and(|v| v == "0"))
}
/// `RPS5E_PM4_PROFILE=1`: nanoseconds spent interpreting packets, and the
/// part of it spent in the render executor's draw/dispatch calls.
pub static CP_NANOS: AtomicU64 = AtomicU64::new(0);
pub static EXEC_NANOS: AtomicU64 = AtomicU64::new(0);

pub fn profiling() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_PM4_PROFILE").is_ok_and(|v| v == "1"))
}

/// Serializes guest atomics and semaphore updates performed by GPU packets.
static ATOMIC_GATE: Mutex<()> = Mutex::new(());
/// Global data share contents.
static GDS: Mutex<Vec<u8>> = Mutex::new(Vec::new());

/// Publishes an occlusion result block counted on the GPU, as a
/// command-processor memory write; queues waiting on it (Z-pass
/// predication, WAIT_REG_MEM) poll again.
fn occlusion_write() -> &'static ps5gpu::render::OcclusionWrite {
    static WRITE: std::sync::OnceLock<ps5gpu::render::OcclusionWrite> = std::sync::OnceLock::new();
    WRITE.get_or_init(|| {
        Arc::new(|block: u64, data: &[u8]| {
            if !mem::write(block, data) && rare_warning(&WRITE_WARNINGS) {
                log::warn!("[pm4] cannot write the occlusion result at 0x{block:x}");
            }
            super::queue::global().retry_blocked();
        })
    })
}

/// End-of-pipe labels are written when the packet is processed, ahead of the
/// GPU: the guest runs ahead of the GPU as before. A CPU read of a
/// read-through page after such a label sees the last completed
/// write-through, not necessarily the work in front of the label.
/// `RPS5E_LABELS_AT_EOP=1` writes them once the work before them completed
/// (and its write-throughs reached guest memory), as the console does:
/// exact, but ASTRO BOT loses ~37 % of its frame rate waiting on the GPU.
/// Addresses of labels written by end-of-pipe work (direct-mapped: a
/// collision forgets the older one).
static GPU_LABELS: [AtomicU64; 512] = [const { AtomicU64::new(0) }; 512];

fn label_slot(addr: u64) -> &'static AtomicU64 {
    &GPU_LABELS[((addr >> 3).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 55) as usize % GPU_LABELS.len()]
}

/// `addr` holds a label end-of-pipe work wrote.
pub(crate) fn note_gpu_label(addr: u64) {
    label_slot(addr).store(addr, Ordering::Release);
}

/// True when `addr` was last written as an end-of-pipe label: a wait on it
/// that is satisfied waited for the GPU, not for the CPU to release data
/// (no CPU synchronisation point). `RPS5E_GPU_LABEL_SYNC=1` treats every
/// satisfied wait as one.
pub(crate) fn is_gpu_label(addr: u64) -> bool {
    static ALWAYS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    addr != 0 && ps5gpu::backend::ab::fast() && !*ALWAYS.get_or_init(|| std::env::var("RPS5E_GPU_LABEL_SYNC").is_ok_and(|v| v == "1")) && label_slot(addr).load(Ordering::Acquire) == addr
}

/// End-of-pipe labels are written when the GPU finished the work before
/// them (`RPS5E_LABELS_AT_EOP=1`), not once it was recorded. Required when
/// the GPU reads guest memory where the CPU wrote it (`RPS5E_BDA=1`): a
/// label the guest takes as "the GPU consumed this" lets it reuse that
/// memory, which recorded-but-unexecuted work would then read.
///
/// `RPS5E_LABELS_AT_EOP=through`: only the labels written while GPU
/// results are still on their way to CPU-readable guest memory
/// (write-throughs of read-through pages) wait for the GPU; the others are
/// written once recorded. Without it, a label the CPU sees can come before
/// those results, and the CPU reads the older bytes it was told are done.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LabelMode {
    /// Written once the work before them is recorded.
    Recorded,
    /// Written once the GPU finished the work before them.
    AfterGpu,
    /// After the GPU only while write-throughs are pending.
    AfterWriteThroughs,
}

pub(crate) fn parse_label_mode(v: Option<&str>, bda: bool) -> LabelMode {
    match v.map(str::trim) {
        Some("1") => LabelMode::AfterGpu,
        Some("0") => LabelMode::Recorded,
        Some("through") => LabelMode::AfterWriteThroughs,
        _ if bda => LabelMode::AfterGpu,
        _ => LabelMode::Recorded,
    }
}

fn label_mode() -> LabelMode {
    static MODE: std::sync::OnceLock<LabelMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| parse_label_mode(std::env::var("RPS5E_LABELS_AT_EOP").ok().as_deref(), ps5gpu::buffers::guest::bda_configured()))
}

/// Whether an end-of-pipe label is written after the GPU finished the work
/// recorded before it, for executor `ex`.
fn label_after_gpu(ex: &dyn RenderExecutor) -> bool {
    match label_mode() {
        LabelMode::Recorded => false,
        LabelMode::AfterGpu => true,
        LabelMode::AfterWriteThroughs => ex.writes_through_pending(),
    }
}

/// 100 MHz reference clock (EOP / COPY_DATA timestamps): the clock shaders
/// read with `s_memrealtime`.
pub fn reference_clock() -> u64 {
    ps5gpu::clock::reference_clock()
}

fn gds_read(offset: usize, out: &mut [u8]) {
    let mut g = GDS.lock().unwrap();
    if g.is_empty() {
        g.resize(GDS_BYTES, 0);
    }
    for (i, b) in out.iter_mut().enumerate() {
        *b = g.get(offset + i).copied().unwrap_or(0);
    }
}

fn gds_write(offset: usize, data: &[u8]) {
    let mut g = GDS.lock().unwrap();
    if g.is_empty() {
        g.resize(GDS_BYTES, 0);
    }
    for (i, b) in data.iter().enumerate() {
        if let Some(d) = g.get_mut(offset + i) {
            *d = *b;
        }
    }
}

/// One buffer on the execution stack. Its range is checked against the guest
/// mappings once, when it is entered; packets are then read in place.
#[derive(Debug, Clone, Copy)]
pub struct Cursor {
    pub addr: u64,
    pub dwords: u32,
    pub offset: u32,
}

impl Cursor {
    fn new(addr: u64, dwords: u32) -> Self {
        Self { addr, dwords, offset: 0 }
    }

    fn remaining(&self) -> u32 {
        self.dwords - self.offset
    }
}

/// A buffer a packet asked to run next: a call returns to the packet after
/// it, a chain (or a taken branch) replaces the current buffer.
#[derive(Debug, Clone, Copy)]
struct NextBuffer {
    addr: u64,
    dwords: u32,
    chain: bool,
}

/// Execution state of one submission (survives suspensions).
#[derive(Debug, Default)]
pub struct Execution {
    pub stack: Vec<Cursor>,
    pub suspended: bool,
    pub made_progress: bool,
    started: bool,
    /// Set by INDIRECT_BUFFER / branch packets; entered once the packet
    /// has been consumed.
    next: Option<NextBuffer>,
    /// What the submission waits on when it suspended on a memory wait
    /// (address, value read, reference, mask, compare function).
    pub waiting_on: Option<(u64, u64, u64, u64, u32)>,
}

impl Execution {
    /// Address of the packet the submission is parked on.
    pub fn current_packet(&self) -> u64 {
        self.stack.last().map_or(0, |c| c.addr + c.offset as u64 * 4)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    Complete,
    Blocked,
}

/// Packet being executed.
#[derive(Debug, Clone, Copy)]
pub struct Packet {
    pub header: u32,
    pub addr: u64,
    pub remaining: u32,
}

impl Packet {
    fn len(&self) -> u32 {
        length(self.header)
    }
}

type Result<T> = std::result::Result<T, String>;

fn addr64(lo: u32, hi: u32) -> u64 {
    lo as u64 | (hi as u64) << 32
}

fn compare(value: u64, reference: u64, mask: u64, function: u32) -> Option<bool> {
    let v = value & mask;
    Some(match function {
        0 => true,
        1 => v < reference,
        2 => v <= reference,
        3 => v == reference,
        4 => v != reference,
        5 => v >= reference,
        6 => v > reference,
        _ => return None,
    })
}

fn wait_operation(control: u32, is64: bool) -> u32 {
    if is64 {
        ((control >> 8) & 1) | ((control >> 5) & 6)
    } else {
        ((control >> 8) & 3) | ((control >> 4) & 0xC)
    }
}

/// Interpreter of one hardware queue.
pub struct Interpreter {
    pub queue_id: usize,
    /// Interrupt ident (0 graphics, 0x20.. compute owners).
    pub event_id: u64,
    pub is_compute: bool,
    /// Register banks, context save slot and instance count.
    pub regs: RegisterFile,
    pub index_base: u64,
    pub index_type: u32,
    pub index_buffer_size: u32,
    pub draw_args_base: u64,
    pub dispatch_args_base: u64,
    pub predicate_skip: bool,
    pub cond_wait_enabled: bool,
    ce_count: u32,
    de_count: u32,
    pub ce_complete: bool,
    const_ram: Vec<u32>,
    atomic_return_me: Option<u64>,
    atomic_return_pfp: Option<u64>,
    occlusion_counter: u64,
    /// See [`FlipEopLatch`].
    flip_eop_latch: FlipEopLatch,
    /// Payload of the packet being executed (reused: no per-packet
    /// allocation).
    payload: Vec<u32>,
    pending_flip: (i32, i32, u32, i64),
    pub draws: u64,
    pub dispatches: u64,
    pub flips: u64,
    /// Receives resolved draws/dispatches and end-of-pipe work.
    pub executor: Option<Arc<dyn RenderExecutor>>,
    /// The last resolved draw (the next draw re-resolves only the banks
    /// written since, `regs.dirty`).
    draw_cache: Option<ResolvedDraw>,
}

/// Context registers after a context clear (shared with the slice recorder).
pub use ps5gpu::render::regfile::initial_context_registers;

/// Dwords as bytes (for reads into dword arrays).
fn bytemuck_dwords(w: &mut [u32]) -> &mut [u8] {
    // SAFETY: u8 has no alignment or validity requirements.
    unsafe { std::slice::from_raw_parts_mut(w.as_mut_ptr() as *mut u8, w.len() * 4) }
}

/// A command buffer must be dword-aligned and lie in readable guest memory:
/// checked once when the buffer is entered, then read in place.
fn check_buffer(addr: u64, dwords: u32) -> Result<()> {
    if addr == 0 || addr & 3 != 0 || !mem::accessible(addr, dwords as u64 * 4, false) {
        return Err(format!("command buffer 0x{addr:x} ({dwords} dwords) is not readable guest memory"));
    }
    Ok(())
}

impl Interpreter {
    pub fn new(queue_id: usize) -> Self {
        let event_id = if queue_id == 0 { 0 } else { COMPUTE_QUEUE_BASE + queue_id as u64 - 1 };
        Self {
            queue_id,
            event_id,
            is_compute: queue_id != 0,
            regs: RegisterFile::new(),
            index_base: 0,
            index_type: 0,
            index_buffer_size: 0,
            draw_args_base: 0,
            dispatch_args_base: 0,
            predicate_skip: false,
            cond_wait_enabled: false,
            ce_count: 0,
            de_count: 0,
            ce_complete: false,
            const_ram: vec![0; CONSTANT_RAM_DWORDS],
            atomic_return_me: None,
            atomic_return_pfp: None,
            occlusion_counter: 0,
            flip_eop_latch: Arc::new(Mutex::new(None)),
            payload: Vec::with_capacity(64),
            pending_flip: (0, 0, 0, 0),
            draws: 0,
            dispatches: 0,
            flips: 0,
            executor: crate::agc::render_executor(),
            draw_cache: None,
        }
    }

    fn banks(&self) -> RegisterBanks<'_> {
        self.regs.banks()
    }

    /// The slice sequencer when the executor records in slices: register
    /// writes, draws and dispatches go to its log.
    #[inline]
    fn slices(&self) -> Option<&SliceExecutor> {
        self.executor.as_deref().and_then(|e| e.slices())
    }

    /// Processor reset (frame boundary on the graphics queue, queue-reset
    /// packets): register banks, saved context, draw state, predication and
    /// constant RAM.
    pub fn reset(&mut self) {
        if let Some(s) = self.slices() {
            s.log_reset(self.queue_id, &self.regs);
        }
        self.regs.reset();
        self.index_type = 0;
        self.index_buffer_size = 0;
        self.index_base = 0;
        self.predicate_skip = false;
        self.cond_wait_enabled = false;
        self.draw_args_base = 0;
        self.dispatch_args_base = 0;
        self.const_ram.fill(0);
    }

    /// Per-submission counters.
    pub fn begin_submission(&mut self) {
        self.ce_count = 0;
        self.de_count = 0;
        self.ce_complete = true;
        self.pending_flip = (0, 0, 0, 0);
    }

    /// Run a submission until it completes or suspends.
    pub fn process(&mut self, ex: &mut Execution, addr: u64, dwords: u32) -> Result<Progress> {
        if !ex.started {
            ex.started = true;
            if dwords != 0 {
                check_buffer(addr, dwords)?;
                ex.stack.push(Cursor::new(addr, dwords));
            }
        }
        ex.suspended = false;
        ex.made_progress = false;
        ex.waiting_on = None;
        ex.next = None;
        let mut packets = 0u64;
        let r = self.run_packets(ex, &mut packets);
        PACKETS.fetch_add(packets, Ordering::Relaxed);
        r?;
        Ok(if ex.stack.is_empty() { Progress::Complete } else { Progress::Blocked })
    }

    fn suspend(ex: &mut Execution) {
        ex.suspended = true;
    }

    /// Ask for `dwords` at `addr` to run after the current packet: a call
    /// (`chain` false) returns to the next packet, a chain replaces the
    /// current buffer.
    fn run_next(ex: &mut Execution, addr: u64, dwords: u32, chain: bool) {
        ex.next = Some(NextBuffer { addr, dwords, chain });
    }

    /// Enter the buffer a packet asked for. A call from the last packet of
    /// a buffer replaces it too: nothing is left to return to.
    fn enter(ex: &mut Execution, next: NextBuffer) -> Result<()> {
        check_buffer(next.addr, next.dwords)?;
        let cursor = Cursor::new(next.addr, next.dwords);
        let top = ex.stack.last_mut().expect("a packet runs from a buffer");
        if next.chain || top.offset == top.dwords {
            *top = cursor;
        } else {
            if ex.stack.len() >= MAX_DEPTH {
                return Err(format!("indirect buffer nesting too deep at 0x{:x}", next.addr));
            }
            ex.stack.push(cursor);
        }
        Ok(())
    }

    fn run_packets(&mut self, ex: &mut Execution, packets: &mut u64) -> Result<()> {
        while let Some(&cur) = ex.stack.last() {
            if cur.offset >= cur.dwords {
                if cur.offset > cur.dwords {
                    return Err(format!("cursor past its buffer at 0x{:x}", cur.addr));
                }
                ex.stack.pop();
                continue;
            }
            let paddr = cur.addr + cur.offset as u64 * 4;
            // SAFETY: the buffer's range was checked when it was entered.
            let h = unsafe { std::ptr::read_volatile(paddr as *const u32) };
            *packets += 1;
            if packet_size(h) == 1 {
                ex.stack.last_mut().unwrap().offset += 1;
                ex.made_progress = true;
                continue;
            }
            let remaining = cur.remaining();
            if remaining < 2 || super::packet_type(h) != 3 {
                return Err(format!("bad packet header 0x{h:08x} at 0x{paddr:x} (remaining {remaining})"));
            }
            let len = length(h);
            if len > remaining {
                return Err(format!("packet 0x{h:08x} at 0x{paddr:x} longer ({len}) than its buffer ({remaining})"));
            }
            if predicated(h) && self.predicate_skip {
                ex.stack.last_mut().unwrap().offset += len;
                ex.made_progress = true;
                continue;
            }
            let mut payload = std::mem::take(&mut self.payload);
            payload.clear();
            // SAFETY: inside the checked buffer range (len <= remaining).
            payload.extend_from_slice(unsafe { std::slice::from_raw_parts((paddr + 4) as *const u32, (len - 1) as usize) });
            let pkt = Packet { header: h & !1, addr: paddr, remaining };
            let r = self.dispatch(ex, &pkt, &payload);
            self.payload = payload;
            let consumed = r? + 1;
            if consumed > remaining {
                return Err(format!("packet 0x{h:08x} at 0x{paddr:x} consumed past its buffer"));
            }
            if ex.suspended {
                // The packet runs again from the start when resumed.
                ex.next = None;
                return Ok(());
            }
            ex.stack.last_mut().unwrap().offset += consumed;
            ex.made_progress = true;
            if let Some(next) = ex.next.take() {
                Self::enter(ex, next)?;
            }
        }
        Ok(())
    }

    /// Returns the payload dwords consumed.
    fn dispatch(&mut self, ex: &mut Execution, p: &Packet, pl: &[u32]) -> Result<u32> {
        let rest = p.len() - 1;
        match super::opcode(p.header) {
            op::NOP => self.nop(ex, p, pl),
            op::SET_BASE => Ok(self.set_base(p, pl)),
            op::CLEAR_STATE => {
                self.context_state(ContextState::Clear);
                Ok(rest)
            }
            op::INDEX_BASE => {
                self.index_base = addr64(pl[0], pl.get(1).copied().unwrap_or(0));
                Ok(rest)
            }
            op::INDEX_BUFFER_SIZE => {
                self.index_buffer_size = pl[0];
                Ok(rest)
            }
            op::INDEX_TYPE => {
                self.index_type = pl[0] & 3;
                Ok(rest)
            }
            op::INDEX_ATTRIBUTES_INDIRECT => {
                self.index_attributes_indirect(p, pl);
                Ok(rest)
            }
            op::NUM_INSTANCES => {
                let n = pl[0].max(1);
                if let Some(s) = self.slices() {
                    s.log_num_instances(self.queue_id, &self.regs, n);
                }
                self.regs.num_instances = n;
                Ok(rest)
            }
            op::DISPATCH_DIRECT | op::DISPATCH_INDIRECT => {
                self.dispatch_packet(&ex.stack, p, pl);
                Ok(rest)
            }
            op::DRAW_INDIRECT
            | op::DRAW_INDEX_INDIRECT
            | op::DRAW_INDEX_2
            | op::DRAW_INDIRECT_MULTI
            | op::DRAW_INDEX_AUTO
            | op::DRAW_INDEX_MULTI_AUTO
            | op::DRAW_INDEX_OFFSET_2
            | op::DRAW_INDEX_INDIRECT_MULTI
            | op::DISPATCH_DRAW_PREAMBLE => {
                self.draw_packet(&ex.stack, p, pl);
                Ok(rest)
            }
            op::ATOMIC_MEM => self.atomic_mem(ex, p, pl),
            op::SET_PREDICATION => self.set_predication(ex, p, pl),
            op::COND_EXEC => self.cond_exec(p, pl),
            op::WRITE_DATA => self.write_data_native(p, pl),
            op::MEM_SEMAPHORE => self.mem_semaphore(ex, p, pl),
            op::WAIT_REG_MEM => self.wait_reg_mem(ex, p, pl, false),
            op::WAIT_REG_MEM64 => self.wait_reg_mem(ex, p, pl, true),
            op::INDIRECT_BUFFER => self.indirect_buffer(ex, p, pl),
            op::COPY_DATA => self.copy_data(p, pl),
            op::COND_WRITE => self.cond_write(p, pl),
            op::EVENT_WRITE => self.event_write(p, pl),
            op::EVENT_WRITE_EOP => self.event_write_eop(p, pl),
            op::EVENT_WRITE_EOS => self.event_write_eos(p, pl),
            op::RELEASE_MEM => self.release_mem_native(p, pl),
            op::DMA_DATA => self.dma_native(p, pl),
            op::REWIND => {
                if pl[0] & (1 << 31) == 0 {
                    Self::suspend(ex);
                }
                Ok(rest)
            }
            op::SET_CONFIG_REG => {
                // Config registers (0x2000..) hold no state the back-end
                // draws or dispatches with.
                crate::agc::log_once(&format!("pm4.config.{:x}", pl[0] & 0xFFFF), || {
                    log::info!("[pm4] SET_CONFIG_REG 0x{:x} ({} values) not modelled", pl[0] & 0xFFFF, rest - 1)
                });
                Ok(rest)
            }
            op::SET_CONTEXT_REG => self.set_registers(Bank::Context, p, pl),
            op::SET_SH_REG => self.set_registers(Bank::Shader, p, pl),
            op::SET_UCONFIG_REG | op::SET_UCONFIG_REG_INDEX => self.set_registers(Bank::UserConfig, p, pl),
            op::SET_CONTEXT_REG_INDIRECT => self.register_table_native(Bank::Context, p, pl),
            op::SET_SH_REG_INDIRECT => self.register_table_native(Bank::Shader, p, pl),
            op::SET_UCONFIG_REG_INDIRECT => self.register_table_native(Bank::UserConfig, p, pl),
            op::WRITE_CONST_RAM => {
                let off = (pl[0] / 4) as usize;
                for (i, v) in pl[1..].iter().enumerate() {
                    if let Some(d) = self.const_ram.get_mut(off + i) {
                        *d = *v;
                    }
                }
                Ok(rest)
            }
            op::DUMP_CONST_RAM if pl.len() >= 4 => {
                let (off, n) = ((pl[0] / 4) as usize, pl[1] as usize);
                let words: Vec<u32> = (0..n).map(|i| self.const_ram.get(off + i).copied().unwrap_or(0)).collect();
                self.write_dwords(addr64(pl[2], pl[3]), &words);
                Ok(rest)
            }
            op::INCREMENT_CE_COUNTER => {
                self.ce_count += 1;
                Ok(rest)
            }
            op::INCREMENT_DE_COUNTER => {
                self.de_count += 1;
                Ok(rest)
            }
            op::WAIT_ON_CE_COUNTER => {
                if self.ce_count <= self.de_count && !self.ce_complete {
                    Self::suspend(ex);
                }
                Ok(rest)
            }
            op::WAIT_ON_DE_COUNTER_DIFF => {
                if self.de_count > self.ce_count {
                    return Err(format!("draw-engine counter ahead of the constant engine at 0x{:x}", p.addr));
                }
                if self.ce_count - self.de_count >= pl[0] {
                    Self::suspend(ex);
                }
                Ok(rest)
            }
            op::GET_LOD_STATS => {
                if pl.len() >= 3 {
                    let size = pl[0] as usize;
                    let dst = (pl[1] & !0x3F) as u64 | (pl[2] as u64) << 32;
                    // Cleared, with a non-zero first dword (stats ready).
                    if dst != 0 && size != 0 && (!mem::fill(dst, size as u64, 0) || (size >= 4 && !mem::w32(dst, 1))) && rare_warning(&WRITE_WARNINGS) {
                        log::warn!("[pm4] LOD stats buffer 0x{dst:x} ({size} bytes) not writable");
                    }
                }
                Ok(rest)
            }
            op::ACQUIRE_MEM
            | op::PFP_SYNC_ME
            | op::CONTEXT_CONTROL
            | op::PRIME_UTCL2
            | op::ATOMIC_GDS => Ok(rest),
            other => {
                // Once per opcode (a bit per opcode, no formatting per packet).
                let bit = 1u64 << (other & 63);
                if UNHANDLED_OPCODES[(other >> 6) as usize & 3].fetch_or(bit, Ordering::Relaxed) & bit == 0 {
                    log::warn!("[pm4] unhandled opcode 0x{other:02x} (header 0x{:08x} at 0x{:x}); skipped", p.header, p.addr);
                }
                Ok(rest)
            }
        }
    }

    fn record_draw(&mut self, stack: &[Cursor], at: &Packet, packet: DrawPacket) {
        self.draws += 1;
        DRAWS.fetch_add(1, Ordering::Relaxed);
        dump_registers(self, &packet);
        if let Some(ex) = &self.executor {
            let t = profiling().then(std::time::Instant::now);
            let banks = self.regs.banks();
            let resolved = match &mut self.draw_cache {
                Some(d) => {
                    ps5gpu::render::resolve_draw_into(&banks, &packet, self.regs.dirty, d);
                    d
                }
                cache => cache.insert(ps5gpu::render::resolve_draw(&banks, &packet)),
            };
            self.regs.dirty = DirtyBanks::NONE;
            if lookahead::enabled() {
                let ahead = Ahead { regs: &self.regs, queue: self.queue_id, dispatch_args_base: self.dispatch_args_base, predicate_skip: self.predicate_skip, stack, at: at.addr, len: at.len() };
                lookahead::with_source(&ahead, || ex.draw(resolved));
            } else {
                ex.draw(resolved);
            }
            if let Some(t) = t {
                EXEC_NANOS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
        }
    }

    fn draw_state(&self, kind: DrawKind, addr: u64) -> DrawPacket {
        DrawPacket {
            kind,
            queue: self.queue_id,
            packet_address: addr,
            index_base: self.index_base,
            index_buffer_size: self.index_buffer_size,
            index_type: self.index_type,
            instance_count: self.regs.num_instances.max(1),
            ..Default::default()
        }
    }

    /// Decode a draw packet into its draw arguments.
    fn draw_packet(&mut self, stack: &[Cursor], p: &Packet, pl: &[u32]) {
        let at = |i: usize| pl.get(i).copied().unwrap_or(0);
        let index_bytes = ps5gpu::render::IndexType::from_raw(self.index_type).bytes();
        let packet = match super::opcode(p.header) {
            op::DRAW_INDEX_2 => DrawPacket {
                count: at(3),
                index_address: addr64(at(1), at(2)),
                index_buffer_size: at(0),
                draw_initiator: at(4),
                ..self.draw_state(DrawKind::Index, p.addr)
            },
            op::DRAW_INDEX_OFFSET_2 => DrawPacket {
                count: at(2),
                index_address: self.index_base + at(1) as u64 * index_bytes,
                index_buffer_size: at(0),
                draw_initiator: at(3),
                ..self.draw_state(DrawKind::IndexOffset, p.addr)
            },
            op::DRAW_INDEX_AUTO => {
                DrawPacket { count: at(0), draw_initiator: at(1), ..self.draw_state(DrawKind::Auto, p.addr) }
            }
            op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT => {
                let kind = if super::opcode(p.header) == op::DRAW_INDIRECT { DrawKind::Indirect } else { DrawKind::IndexIndirect };
                DrawPacket {
                    indirect_address: self.draw_args_base + at(0) as u64,
                    indirect_count: 1,
                    draw_initiator: at(3),
                    indirect_registers: IndirectRegisters::decode(at(1), at(2), at(2), kind.indexed()),
                    ..self.draw_state(kind, p.addr)
                }
            }
            op::DRAW_INDIRECT_MULTI | op::DRAW_INDEX_INDIRECT_MULTI => {
                let kind =
                    if super::opcode(p.header) == op::DRAW_INDIRECT_MULTI { DrawKind::IndirectMulti } else { DrawKind::IndexIndirectMulti };
                let from_memory = at(3) & (1 << 30) != 0;
                DrawPacket {
                    indirect_address: self.draw_args_base + at(0) as u64,
                    indirect_count: at(4),
                    count_address: if from_memory { addr64(at(5) & !3, at(6)) } else { 0 },
                    stride: at(7),
                    draw_initiator: at(8),
                    indirect_registers: IndirectRegisters::decode_multi(at(1), at(2), at(3), kind.indexed()),
                    ..self.draw_state(kind, p.addr)
                }
            }
            // [count, index64, maxInstances, _, _, instances, initiator]; no
            // instance count draws NUM_INSTANCES.
            op::DISPATCH_DRAW_PREAMBLE => DrawPacket {
                count: at(0),
                index_address: addr64(at(1), at(2)),
                instance_count: if at(6) != 0 { at(6) } else { self.regs.num_instances },
                draw_initiator: at(7),
                ..self.draw_state(DrawKind::IndexMultiInstanced, p.addr)
            },
            // DRAW_INDEX_MULTI_AUTO: counted, not decoded.
            _ => DrawPacket { count: at(0), ..self.draw_state(DrawKind::Auto, p.addr) },
        };
        if let Some(s) = self.slices() {
            // The slice thread takes the instance count from its own state
            // and applies the argument record at the draw's position.
            let instances = match super::opcode(p.header) {
                op::DISPATCH_DRAW_PREAMBLE if at(6) != 0 => InstanceSource::Packet,
                op::DISPATCH_DRAW_PREAMBLE => InstanceSource::StateRaw,
                _ => InstanceSource::State,
            };
            let fixup = packet.kind.indirect() && self.draw_args_base != 0;
            s.log_draw(self.queue_id, &self.regs, &packet, instances, fixup);
            self.draws += 1;
            DRAWS.fetch_add(1, Ordering::Relaxed);
            dump_registers(self, &packet);
            return;
        }
        if packet.kind.indirect() {
            self.apply_indirect_arguments(&packet);
        }
        self.record_draw(stack, p, packet);
    }

    /// The draw state an indirect draw leaves behind: each argument record
    /// writes its offsets (and a multi-draw its index) to the destination
    /// shader registers and sets the instance count, so the last record
    /// drawn decides (read at parse time;
    /// the draws themselves read the records on the GPU).
    fn apply_indirect_arguments(&mut self, packet: &DrawPacket) {
        if self.draw_args_base == 0 {
            return;
        }
        let indexed = packet.kind.indexed();
        let mut count = packet.indirect_count;
        if packet.count_address != 0 {
            match mem::r32(packet.count_address) {
                Some(c) => count = count.min(c),
                None => return,
            }
        }
        if count == 0 {
            return;
        }
        let args_bytes = if indexed { 20 } else { 16 };
        let stride = if matches!(packet.kind, DrawKind::IndirectMulti | DrawKind::IndexIndirectMulti) { packet.stride as u64 } else { args_bytes };
        if stride < args_bytes {
            return;
        }
        let mut args = [0u32; 5];
        let record = packet.indirect_address + (count as u64 - 1) * stride;
        self.memory_hazard(record, args_bytes);
        if !mem::read(record, bytemuck_dwords(&mut args[..args_bytes as usize / 4])) {
            return;
        }
        // Non-indexed: [vertices, instances, first vertex, first instance];
        // indexed: [indices, instances, first index, base vertex, first instance].
        let (instances, vertex_offset, instance_offset, first_index) =
            if indexed { (args[1], args[3], args[4], args[2]) } else { (args[1], args[2], args[3], 0) };
        let r = packet.indirect_registers;
        self.write_register(Bank::Shader, r.vertex_offset, vertex_offset);
        self.write_register(Bank::Shader, r.instance_offset, instance_offset);
        self.write_register(Bank::Shader, r.first_index, first_index);
        self.write_register(Bank::Shader, r.draw_index_target(), count - 1);
        self.regs.num_instances = instances;
    }

    /// Decode a dispatch packet: DISPATCH_DIRECT `[x, y, z, initiator]`,
    /// DISPATCH_INDIRECT `[offset, initiator]` (graphics, from SET_BASE) or
    /// `[addr64, initiator]` (compute queues).
    /// INDEX_ATTRIBUTES_INDIRECT: the index base, size (in indices) and
    /// type come from the 16-byte record `index` of the table at the
    /// packet's base `{addr_lo, addr_hi, size, type}`.
    fn index_attributes_indirect(&mut self, p: &Packet, pl: &[u32]) {
        if pl.len() < 3 || pl[0] & 0xF != 0 || pl[2] >> 16 != 0 {
            crate::agc::log_once("pm4.index_attributes", || log::warn!("[pm4] INDEX_ATTRIBUTES_INDIRECT at {:#x} in an unsupported mode {:x?}", p.addr, pl));
            return;
        }
        let record = addr64(pl[0], pl[1]) + (pl[2] & 0xFFFF) as u64 * 16;
        self.memory_hazard(record, 16);
        let mut b = [0u8; 16];
        if !self.read_ordered(record, &mut b) {
            log::warn!("[pm4] INDEX_ATTRIBUTES_INDIRECT record {record:#x} unreadable");
            return;
        }
        let w: [u32; 4] = std::array::from_fn(|i| u32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]));
        if w[3] > 3 {
            crate::agc::log_once("pm4.index_attributes.type", || log::warn!("[pm4] INDEX_ATTRIBUTES_INDIRECT record {record:#x} index type {:#x} unsupported", w[3]));
            return;
        }
        self.index_base = addr64(w[0], w[1]);
        self.index_buffer_size = w[2];
        self.index_type = w[3];
    }

    /// The value the queue holds for the register at absolute dword
    /// offset `reg` (shader 0x2C00.., context 0xA000.., user config
    /// 0xC000..).
    fn register_value(&self, reg: u32) -> Option<u32> {
        let (bank, base) = match reg {
            0x2C00..=0x2FFF => (&self.regs.shader, 0x2C00),
            0xA000..=0xA3FF => (&self.regs.context, 0xA000),
            0xC000..=0xFFFF => (&self.regs.uconfig, 0xC000),
            _ => return None,
        };
        bank.get((reg - base) as usize).copied()
    }

    /// A dispatch leaves COMPUTE_DISPATCH_INITIATOR and, when direct, its
    /// group counts in COMPUTE_DIM_X/Y/Z (an indirect one clears them): a
    /// later COPY_DATA from these registers reads them.
    fn note_dispatch_registers(&mut self, initiator: u32, dims: [u32; 3]) {
        const COMPUTE_DISPATCH_INITIATOR: u32 = 0x200;
        let values = [initiator, dims[0], dims[1], dims[2]];
        let at = COMPUTE_DISPATCH_INITIATOR as usize;
        if self.regs.shader.get(at..at + 4) == Some(&values[..]) {
            return;
        }
        if let Some(s) = self.slices() {
            s.log_run(self.queue_id, &self.regs, Bank::Shader, COMPUTE_DISPATCH_INITIATOR, &values);
        }
        self.regs.set_run(Bank::Shader, COMPUTE_DISPATCH_INITIATOR, &values);
    }

    fn dispatch_packet(&mut self, stack: &[Cursor], p: &Packet, pl: &[u32]) {
        self.dispatches += 1;
        DISPATCHES.fetch_add(1, Ordering::Relaxed);
        let packet = decode_dispatch(self.queue_id, self.dispatch_args_base, p, pl);
        let dims = if packet.indirect_address == 0 { packet.groups } else { [0; 3] };
        self.note_dispatch_registers(packet.dispatch_initiator, dims);
        let Some(ex) = &self.executor else { return };
        if let Some(s) = ex.slices() {
            s.log_dispatch(self.queue_id, &self.regs, &packet);
            return;
        }
        // With thread dimensions the back-end reads the arguments once the
        // GPU work writing them is done (ps5gpu `Backend::dispatch`).
        let t = profiling().then(std::time::Instant::now);
        let resolved = ps5gpu::render::resolve_dispatch(&self.banks(), &packet);
        if lookahead::enabled() {
            let ahead = Ahead { regs: &self.regs, queue: self.queue_id, dispatch_args_base: self.dispatch_args_base, predicate_skip: self.predicate_skip, stack, at: p.addr, len: p.len() };
            lookahead::with_source(&ahead, || ex.dispatch(&resolved));
        } else {
            ex.dispatch(&resolved);
        }
        if let Some(t) = t {
            EXEC_NANOS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }

    // ---- memory helpers (failures are logged, not fatal) ----

    /// Transfers queued over `[addr, addr+len)` happen before the
    /// interpreter touches it on the CPU.
    fn memory_hazard(&self, addr: u64, len: u64) {
        if let Some(ex) = &self.executor {
            ex.memory_hazard(addr, len);
        }
    }

    fn write_bytes(&self, addr: u64, data: &[u8]) {
        let queue_id = self.queue_id;
        let bytes = data.to_vec();
        let write = move || {
            if !mem::write(addr, &bytes) && rare_warning(&WRITE_WARNINGS) {
                log::warn!("[pm4] queue {queue_id} cannot write {} bytes at 0x{addr:x}", bytes.len());
            }
        };
        // After the command-processor transfers queued over the range.
        match &self.executor {
            Some(ex) => ex.write_ordered(addr, data, Box::new(write)),
            None => write(),
        }
    }

    fn write32(&self, addr: u64, v: u32) {
        self.write_bytes(addr, &v.to_le_bytes());
    }

    fn write64(&self, addr: u64, v: u64) {
        self.write_bytes(addr, &v.to_le_bytes());
    }

    fn write_dwords(&self, addr: u64, w: &[u32]) {
        let queue_id = self.queue_id;
        let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w = w.to_vec();
        let write = move || {
            if !mem::publish_dwords(addr, &w) && rare_warning(&WRITE_WARNINGS) {
                log::warn!("[pm4] queue {queue_id} cannot write {} dwords at 0x{addr:x}", w.len());
            }
        };
        match &self.executor {
            Some(ex) => ex.write_ordered(addr, &bytes, Box::new(write)),
            None => write(),
        }
    }

    /// Guest memory as the command processor sees it here (transfers
    /// queued before applied).
    fn read_ordered(&self, addr: u64, out: &mut [u8]) -> bool {
        match &self.executor {
            Some(ex) => ex.read_ordered(addr, out, &|o: &mut [u8]| mem::read(addr, o)),
            None => mem::read(addr, out),
        }
    }

    fn read32(&self, addr: u64) -> Result<u32> {
        let mut b = [0u8; 4];
        self.read_ordered(addr, &mut b).then(|| u32::from_le_bytes(b)).ok_or_else(|| format!("cannot read dword at 0x{addr:x}"))
    }

    fn read64(&self, addr: u64) -> Result<u64> {
        let mut b = [0u8; 8];
        self.read_ordered(addr, &mut b).then(|| u64::from_le_bytes(b)).ok_or_else(|| format!("cannot read qword at 0x{addr:x}"))
    }

    /// A read a one-shot decision depends on (conditional execution,
    /// predication, branches, atomics): after the GPU work before it is
    /// recorded (slice mode records it on another thread).
    fn read_control(&self, addr: u64, out: &mut [u8]) -> bool {
        match &self.executor {
            Some(ex) => ex.read_control(addr, out, &|o: &mut [u8]| mem::read(addr, o)),
            None => mem::read(addr, out),
        }
    }

    fn read32c(&self, addr: u64) -> Result<u32> {
        let mut b = [0u8; 4];
        self.read_control(addr, &mut b).then(|| u32::from_le_bytes(b)).ok_or_else(|| format!("cannot read dword at 0x{addr:x}"))
    }

    fn read64c(&self, addr: u64) -> Result<u64> {
        let mut b = [0u8; 8];
        self.read_control(addr, &mut b).then(|| u64::from_le_bytes(b)).ok_or_else(|| format!("cannot read qword at 0x{addr:x}"))
    }

    // ---- interrupts / labels ----

    /// Raise a GPU interrupt. With an executor it is delivered once the
    /// work recorded before it has completed on the GPU (end of pipe).
    fn interrupt(&self, context_id: u32) {
        INTERRUPTS.fetch_add(1, Ordering::Relaxed);
        let (event_id, latch) = (self.event_id, self.flip_eop_latch.clone());
        let deliver = move || {
            if equeue::trigger_registered_events_by_filter(equeue::FILTER_GRAPHICS, context_id as u64, event_id) > 0 {
                latch_release_interrupt(&latch);
            }
        };
        match &self.executor {
            Some(ex) => ex.end_of_pipe(Box::new(deliver)),
            None => deliver(),
        }
    }

    /// The graphics end-of-pipe event of a flip: raised once the work before
    /// it completed, unless the queue raises its own release interrupts
    /// ([`flip_eop_event_due`]).
    fn flip_interrupt(&self) {
        let (event_id, latch) = (self.event_id, self.flip_eop_latch.clone());
        let deliver = move || {
            if flip_eop_event_due(&latch) {
                INTERRUPTS.fetch_add(1, Ordering::Relaxed);
                equeue::trigger_registered_events_by_filter(equeue::FILTER_GRAPHICS, 0, event_id);
            }
        };
        match &self.executor {
            Some(ex) => ex.end_of_pipe(Box::new(deliver)),
            None => deliver(),
        }
    }

    /// Interrupt selectors 5/6: once the work before the packet completed,
    /// compare the label at `dst` with `data` (`wide`: 64 bits, else the low
    /// 32 bits) and raise the interrupt when the label is at or below it.
    fn conditional_interrupt(&self, context_id: u32, dst: u64, data: u64, wide: bool) {
        let (event_id, latch) = (self.event_id, self.flip_eop_latch.clone());
        let deliver = move || {
            if conditional_interrupt_due(dst, data, wide) {
                INTERRUPTS.fetch_add(1, Ordering::Relaxed);
                if equeue::trigger_registered_events_by_filter(equeue::FILTER_GRAPHICS, context_id as u64, event_id) > 0 {
                    latch_release_interrupt(&latch);
                }
            }
        };
        match &self.executor {
            Some(ex) => ex.end_of_pipe(Box::new(deliver)),
            None => deliver(),
        }
    }

    /// A flip label: written with the flip's end of pipe when labels follow
    /// GPU completion ([`label_mode`]).
    fn label32(&self, addr: u64, v: u32) {
        LABEL_WRITES.fetch_add(1, Ordering::Relaxed);
        match &self.executor {
            Some(ex) if label_after_gpu(ex.as_ref()) => {
                note_gpu_label(addr);
                ex.after_pipe(Box::new(move || {
                    let _ = mem::write(addr, &v.to_le_bytes());
                    super::queue::global().retry_blocked();
                }))
            }
            _ => self.write32(addr, v),
        }
    }

    /// End-of-pipe write: `data_sel` 1 = 32-bit value, 2 = 64-bit value,
    /// 3/4 = 64-bit reference clock; `int_sel` 0/3 none, 1 interrupt only
    /// (graphics; compute also writes), 2 write then interrupt, 5/6 an
    /// interrupt without a write, raised only when the label at `dst` is at
    /// or below `value` (5: low 32 bits, 6: 64 bits).
    #[allow(clippy::too_many_arguments)]
    fn end_of_pipe(&self, is64: bool, data_sel: u32, dst: u64, value: u64, int_sel: u32, context_id: u32) {
        let interrupt = match int_sel {
            0 | 3 => false,
            1 if !self.is_compute => {
                self.interrupt(context_id);
                return;
            }
            1 | 2 => true,
            5 | 6 => {
                self.conditional_interrupt(context_id, dst, value, int_sel == 6);
                return;
            }
            other => {
                crate::agc::log_once(&format!("pm4.eop.int.{other}"), || log::warn!("[pm4] unknown EOP interrupt selector {other}"));
                false
            }
        };
        // (bytes, value or None for the clock at write time)
        let label: Option<(usize, Option<u64>)> = match data_sel {
            0 => None,
            1 => Some((4, Some(value))),
            2 => Some((if is64 { 8 } else { 4 }, Some(value))),
            // The clock when the pipe drained.
            3 | 4 => Some((8, None)),
            other => {
                crate::agc::log_once(&format!("pm4.eop.sel.{other}"), || log::warn!("[pm4] unknown EOP data selector {other}"));
                None
            }
        };
        if let Some((len, v)) = label {
            LABEL_WRITES.fetch_add(1, Ordering::Relaxed);
            let queue_id = self.queue_id;
            note_gpu_label(dst);
            let write = move || {
                let bytes = v.unwrap_or_else(reference_clock).to_le_bytes();
                if !mem::write(dst, &bytes[..len]) && rare_warning(&WRITE_WARNINGS) {
                    log::warn!("[pm4] queue {queue_id} cannot write the end-of-pipe label at 0x{dst:x}");
                }
            };
            match &self.executor {
                Some(ex) if label_after_gpu(ex.as_ref()) => ex.after_pipe(Box::new(move || {
                    write();
                    // Queues waiting on the label poll it again.
                    super::queue::global().retry_blocked();
                })),
                Some(ex) => {
                    // Written behind the work queued before it (the clock
                    // as the packet is processed).
                    let bytes = v.unwrap_or_else(reference_clock).to_le_bytes();
                    let data = bytes[..len].to_vec();
                    ex.write_ordered(
                        dst,
                        &bytes[..len],
                        Box::new(move || {
                            if !mem::write(dst, &data) && rare_warning(&WRITE_WARNINGS) {
                                log::warn!("[pm4] queue {queue_id} cannot write the end-of-pipe label at 0x{dst:x}");
                            }
                        }),
                    )
                }
                None => write(),
            }
        }
        if interrupt {
            self.interrupt(context_id);
        }
    }

    fn event_write_eop(&mut self, p: &Packet, pl: &[u32]) -> Result<u32> {
        if pl.len() < 5 {
            return Err(format!("short EVENT_WRITE_EOP at 0x{:x}", p.addr));
        }
        let dst = pl[1] as u64 | ((pl[2] & 0xFFFF) as u64) << 32;
        self.end_of_pipe(true, (pl[2] >> 29) & 7, dst, addr64(pl[3], pl[4]), (pl[2] >> 24) & 7, 0);
        Ok(p.len() - 1)
    }

    fn event_write_eos(&mut self, p: &Packet, pl: &[u32]) -> Result<u32> {
        if pl.len() < 4 {
            return Err(format!("short EVENT_WRITE_EOS at 0x{:x}", p.addr));
        }
        let dst = pl[1] as u64 | ((pl[2] & 0xFFFF) as u64) << 32;
        // EOS: data_sel 1 = GDS words (offset | count<<16), 2 = 32-bit value.
        let sel = (pl[2] >> 29) & 7;
        let int_sel = (pl[2] >> 24) & 7;
        match sel {
            1 => self.end_of_shader_gds(dst, pl[3] as u64, int_sel, 0),
            2 => self.end_of_pipe(false, 1, dst, pl[3] as u64, int_sel, 0),
            other => return Err(format!("EVENT_WRITE_EOS data selector {other} unsupported at 0x{:x}", p.addr)),
        }
        Ok(p.len() - 1)
    }

    /// GDS words to memory at end of shader; interrupt selector 1 on the
    /// graphics queue raises the interrupt only.
    fn end_of_shader_gds(&self, dst: u64, value: u64, int_sel: u32, context_id: u32) {
        if int_sel == 1 && !self.is_compute {
            self.interrupt(context_id);
            return;
        }
        self.copy_gds_words(dst, value);
        if matches!(int_sel, 1 | 2) {
            self.interrupt(context_id);
        }
    }

    fn copy_gds_words(&self, dst: u64, value: u64) {
        let offset = (value & 0xFFFF) as usize;
        let count = ((value >> 16) & 0xFFFF) as usize;
        // GDS lives on the GPU: the read-back runs there, after the work
        // recorded before it.
        let op = GdsTransfer::ToMemory { offset: offset as u64 * 4, dst, size: count as u64 * 4 };
        if count == 0 || self.executor.as_ref().is_some_and(|ex| ex.gds_transfer(op)) {
            return;
        }
        let mut buf = vec![0u8; count * 4];
        gds_read(offset * 4, &mut buf);
        // Into GPU-written memory: written on the GPU, in order with its work.
        if self.executor.as_ref().is_some_and(|ex| ex.write_memory(dst, &buf)) {
            return;
        }
        self.write_bytes(dst, &buf);
    }

    /// RELEASE_MEM semantics shared by the native and private forms.
    #[allow(clippy::too_many_arguments)]
    fn release_mem(&self, dst_sel: u32, data_sel: u32, int_sel: u32, dst: u64, value: u64, context_id: u32) {
        if dst_sel > 1 {
            crate::agc::log_once("pm4.release.dst", || log::warn!("[pm4] RELEASE_MEM destination {dst_sel} unsupported"));
            return;
        }
        if data_sel == 0 || int_sel == 4 || (dst_sel == 0 && dst == 0) {
            if matches!(int_sel, 1 | 2 | 4) {
                self.interrupt(context_id);
            }
            return;
        }
        match data_sel {
            1 | 2 | 3 => self.end_of_pipe(data_sel != 1, data_sel, dst, value, int_sel, context_id),
            5 => self.end_of_shader_gds(dst, value, int_sel, context_id),
            other => crate::agc::log_once(&format!("pm4.release.sel.{other}"), || {
                log::warn!("[pm4] RELEASE_MEM data selector {other} unsupported");
            }),
        }
    }

    fn release_mem_native(&mut self, p: &Packet, pl: &[u32]) -> Result<u32> {
        if pl.len() < 7 {
            return Err(format!("short RELEASE_MEM at 0x{:x}", p.addr));
        }
        let control = pl[1];
        let int_sel = (control >> 24) & 7;
        // The low two address bits are a cache-policy field, not address.
        self.release_mem((control >> 16) & 3, (control >> 29) & 7, int_sel, addr64(pl[2] & !3, pl[3]), addr64(pl[4], pl[5]), pl[6] & 0x07FF_FFFF);
        // An 8-dword graphics RELEASE_MEM that interrupts after its write
        // (selector 2) and names a flip sceVideoOutSubmitEopFlip reserved
        // (the token in its context id) performs that flip after it.
        if !self.is_compute && super::opcode(p.header) == op::RELEASE_MEM && p.len() == 8 && int_sel == 2 {
            if let Some(id) = crate::agc::hook_claim_eop_flip(pl[6]) {
                self.submit_eop_flip(id);
            }
        }
        Ok(p.len() - 1)
    }

    /// A claimed EOP flip, after the work before its release (as a flip
    /// packet, it raises the graphics end-of-pipe event).
    fn submit_eop_flip(&mut self, id: u64) {
        self.flips += 1;
        FLIPS.fetch_add(1, Ordering::Relaxed);
        let flip = move || {
            crate::agc::frame_times::flip();
            if let Err(e) = crate::agc::hook_eop_flip(id) {
                log::warn!("[pm4] EOP flip {id} failed: 0x{:x}", e as u32);
            }
        };
        match &self.executor {
            Some(ex) => ex.run_ordered(Box::new(flip)),
            None => flip(),
        }
        if flip_eop_event_enabled() {
            self.flip_interrupt();
        }
    }

    fn event_write(&mut self, p: &Packet, pl: &[u32]) -> Result<u32> {
        let ty = pl[0] & 0x3F;
        let index = (pl[0] >> 8) & 7;
        if ty == 0x39 && p.len() == 4 {
            let addr = addr64(pl[1], pl[2]);
            if index == 1 && addr != 0 && addr & 7 == 0 {
                // PIXEL_PIPE_STAT_DUMP of an occlusion query: counted on the
                // GPU when the executor can (RPS5E_OCCLUSION_QUERIES=1).
                if self.executor.as_ref().is_some_and(|ex| ex.occlusion_counter_dump(addr, occlusion_write())) {
                    return Ok(p.len() - 1);
                }
                // Publish a ready, always-visible result for every depth block.
                let result = (1u64 << 63) | self.occlusion_counter;
                for block in 0..16u64 {
                    self.write64(addr + block * 16, result);
                }
                self.occlusion_counter = (self.occlusion_counter + 1) & ((1 << 63) - 1);
            }
        }
        Ok(p.len() - 1)
    }

    // ---- NOP-wrapped private packets ----

    /// NOP packets: markers (custom 0) and the libSceAgc private packets.
    fn nop(&mut self, ex: &mut Execution, p: &Packet, pl: &[u32]) -> Result<u32> {
        let rest = p.len() - 1;
        match custom(p.header) {
            nop::ZERO => {
                if let Some(&first) = pl.first() {
                    if first & 0xFFFF_0000 == MARKER_MAGIC {
                        self.marker(ex, pl);
                    }
                }
                Ok(rest)
            }
            nop::DISPATCH_RESET => {
                self.reset();
                Ok(rest)
            }
            nop::WAIT_FLIP_DONE => {
                if pl.len() >= 2 && !crate::agc::hook_is_flip_done(pl[0] as i32, pl[1] as i32) {
                    Self::suspend(ex);
                }
                Ok(rest)
            }
            nop::PUSH_MARKER | nop::POP_MARKER | nop::ACQUIRE_MEM => Ok(rest),
            nop::FLIP => {
                if pl.len() < 5 {
                    return Err(format!("short flip packet at 0x{:x}", p.addr));
                }
                self.pending_flip = (pl[0] as i32, pl[1] as i32, pl[2], addr64(pl[3], pl[4]) as i64);
                self.submit_flip();
                // A flip raises the graphics end-of-pipe event once the work
                // before it completed (titles wait on it after every flip
                // without asking for an interrupt in their releases).
                if flip_eop_event_enabled() {
                    self.flip_interrupt();
                }
                Ok(rest)
            }
            // Same dword layout as the native packet.
            nop::RELEASE_MEM => self.release_mem_native(p, pl),
            nop::CONTEXT_STATE => {
                let operation = match pl.first() {
                    Some(0) => ContextState::Clear,
                    Some(1) => ContextState::Push,
                    Some(2) => ContextState::Pop,
                    Some(3) => ContextState::PushClear,
                    other => return Err(format!("unknown context-state operation {other:?} at 0x{:x}", p.addr)),
                };
                self.context_state(operation);
                Ok(rest)
            }
            other => {
                if rare_warning(&NOP_WARNINGS) {
                    log::warn!("[pm4] unhandled NOP custom code 0x{other:02x} at 0x{:x}; skipped", p.addr);
                }
                Ok(rest)
            }
        }
    }

    fn marker(&mut self, _ex: &mut Execution, pl: &[u32]) {
        match pl[0] & 0xFFF {
            0x777 => self.submit_flip(),
            0x778 if pl.len() >= 4 => {
                self.label32(addr64(pl[1], pl[2]), pl[3]);
                self.submit_flip();
            }
            0x781 if pl.len() >= 4 => {
                self.label32(addr64(pl[1], pl[2]), pl[3]);
                self.submit_flip();
                self.interrupt(0);
            }
            _ => {}
        }
    }

    fn submit_flip(&mut self) {
        let (h, i, m, a) = self.pending_flip;
        self.flips += 1;
        FLIPS.fetch_add(1, Ordering::Relaxed);
        // The flip captures what the draws before it rendered: it runs
        // after them.
        let flip = move || {
            crate::agc::frame_times::flip();
            if let Err(e) = crate::agc::hook_gpu_flip(h, i, m, a) {
                log::warn!("[pm4] GPU flip handle={h} index={i} mode={m} failed: 0x{:x}", e as u32);
            }
        };
        match &self.executor {
            Some(ex) => ex.run_ordered(Box::new(flip)),
            None => flip(),
        }
    }

    /// Context-state operations (CLEAR_STATE and the private packet). A
    /// clear restores the reset values; one context can be saved at a time.
    fn context_state(&mut self, operation: ContextState) {
        if let Some(s) = self.slices() {
            s.log_context_state(self.queue_id, &self.regs, operation);
        }
        match self.regs.context_state(operation) {
            Ok(()) => {}
            Err("push") => crate::agc::log_once("pm4.ctx.push", || log::warn!("[pm4] context-state push with a context already saved; ignored")),
            Err(_) => crate::agc::log_once("pm4.ctx.pop", || log::warn!("[pm4] context-state pop with no saved context; ignored")),
        }
    }

    // ---- waits / conditions ----

    fn wait_on_memory(&mut self, ex: &mut Execution, function: u32, addr: u64, reference: u64, mask: u64, operation: u32, is64: bool) -> Result<()> {
        if addr == 0 {
            return Err("wait on address 0".into());
        }
        if operation == 4 && !self.cond_wait_enabled {
            return Ok(());
        }
        let value = if is64 { self.read64(addr)? } else { self.read32(addr)? as u64 };
        match compare(value, reference, mask, function) {
            // The CPU may have written what follows before releasing it.
            // A label the GPU wrote (end of pipe) releases no CPU data.
            Some(true) if is_gpu_label(addr) => {}
            Some(true) => match &self.executor {
                Some(e) => e.sync_point(),
                None => ps5gpu::memory::note_sync_point(),
            },
            Some(false) => {
                ex.waiting_on = Some((addr, value, reference, mask, function));
                Self::suspend(ex)
            }
            None => return Err(format!("unknown wait compare function {function}")),
        }
        Ok(())
    }

    fn wait_reg_mem(&mut self, ex: &mut Execution, p: &Packet, pl: &[u32], is64: bool) -> Result<u32> {
        let vd = if is64 { 2 } else { 1 };
        let need = 4 + vd * 2;
        if pl.len() < need {
            return Err(format!("short WAIT_REG_MEM at 0x{:x}", p.addr));
        }
        let control = pl[0];
        if control & 0x10 == 0 {
            crate::agc::log_once("pm4.wait.register", || log::warn!("[pm4] register-space WAIT_REG_MEM ignored"));
            return Ok(p.len() - 1);
        }
        let val = |i: usize| if is64 { addr64(pl[i], pl[i + 1]) } else { pl[i] as u64 };
        let (reference, mask) = (val(3), val(3 + vd));
        // A 64-bit wait whose mask has no high bits awaits only the dword at
        // the address: it is aligned to 4 bytes and reads 4 bytes.
        let reads64 = is64 && (mask >> 32 != 0 || !wait64_narrow_enabled());
        let align = if reads64 { !7u32 } else { !3u32 };
        let addr = (pl[1] & align) as u64 | ((pl[2] & 0x3FFFF) as u64) << 32;
        self.wait_on_memory(ex, control & 7, addr, reference, mask, wait_operation(control, is64), reads64)?;
        Ok(p.len() - 1)
    }

    fn cond_exec(&mut self, p: &Packet, pl: &[u32]) -> Result<u32> {
        let rest = p.len() - 1;
        if rest < 4 {
            return Err(format!("short COND_EXEC at 0x{:x}", p.addr));
        }
        let addr = (pl[0] & !3) as u64 | (pl[1] as u64) << 32;
        let count = pl[3] & 0x3FFF;
        if addr == 0 || rest + count >= p.remaining {
            return Err(format!("unsupported COND_EXEC (addr 0x{addr:x}, count {count}) at 0x{:x}", p.addr));
        }
        Ok(if self.read32c(addr)? == 0 { rest + count } else { rest })
    }

    fn set_predication(&mut self, ex: &mut Execution, p: &Packet, pl: &[u32]) -> Result<u32> {
        let rest = p.len() - 1;
        if rest < 2 {
            return Err(format!("short SET_PREDICATION at 0x{:x}", p.addr));
        }
        let (flags, addr) = if rest >= 3 && pl[0] & !0x0007_1100 == 0 && pl[2] <= 0xFFFF {
            (pl[0], (pl[1] & 0xFFFF_FFF0) as u64 | (pl[2] as u64) << 32)
        } else {
            (pl[1], (pl[0] & 0xFFFF_FFF0) as u64 | ((pl[1] & 0xFF) as u64) << 32)
        };
        let (cond, operation, wait) = ((flags >> 8) & 1, (flags >> 16) & 7, (flags >> 12) & 1);
        let value = match operation {
            0 => {
                self.predicate_skip = false;
                return Ok(rest);
            }
            // Z-pass: one begin/end counter pair per depth block, bit 63
            // marks each ready. Not ready: wait for it, or draw everything.
            1 => {
                if addr == 0 {
                    return Err(format!("Z-pass predication without an address at 0x{:x}", p.addr));
                }
                const READY: u64 = 1 << 63;
                let mut total = 0u64;
                for block in 0..16u64 {
                    let begin = self.read64c(addr + block * 16)?;
                    let end = self.read64c(addr + block * 16 + 8)?;
                    if begin & end & READY == 0 {
                        if wait == 0 {
                            Self::suspend(ex);
                        } else {
                            self.predicate_skip = false;
                        }
                        return Ok(rest);
                    }
                    total = total.wrapping_add(end.wrapping_sub(begin));
                }
                total
            }
            // Boolean: the wait selector applies only to Z-pass queries.
            3 => {
                if addr == 0 {
                    return Err(format!("boolean predication without an address at 0x{:x}", p.addr));
                }
                self.read64c(addr)?
            }
            // BOOL32: only the low 32 bits count.
            4 => {
                if addr == 0 {
                    return Err(format!("boolean predication without an address at 0x{:x}", p.addr));
                }
                self.read32c(addr)? as u64
            }
            other => return Err(format!("predication operation {other} unsupported at 0x{:x}", p.addr)),
        };
        self.predicate_skip = if cond == 0 { value != 0 } else { value == 0 };
        Ok(rest)
    }

    fn cond_write(&mut self, p: &Packet, pl: &[u32]) -> Result<u32> {
        if pl.len() < 8 {
            return Err(format!("short COND_WRITE at 0x{:x}", p.addr));
        }
        let control = pl[0];
        let function = control & 7;
        let read_addr = ((pl[2] & 0xFFFF) as u64) << 32 | (pl[1] & !3) as u64;
        let write_addr = ((pl[6] & 0xFFFF) as u64) << 32 | (pl[5] & !3) as u64;
        let writes_memory = control & 0x100 != 0;
        if control & 0x10 == 0 || read_addr == 0 || (writes_memory && write_addr == 0) {
            return Err(format!("unsupported COND_WRITE at 0x{:x}", p.addr));
        }
        let observed = self.read32c(read_addr)? as u64;
        if compare(observed, pl[3] as u64, pl[4] as u64, function) == Some(true) {
            if writes_memory {
                self.write32(write_addr, pl[7]);
            } else {
                self.cond_wait_enabled = pl[7] != 0;
            }
        }
        Ok(p.len() - 1)
    }

    // ---- control flow ----

    fn indirect_buffer(&mut self, ex: &mut Execution, p: &Packet, pl: &[u32]) -> Result<u32> {
        if p.len() == 14 {
            return self.branch(ex, p, pl);
        }
        if p.len() != 4 {
            return Err(format!("INDIRECT_BUFFER length {} unsupported at 0x{:x}", p.len(), p.addr));
        }
        let control = pl[2];
        let addr = pl[0] as u64 | (pl[1] as u64) << 32;
        let dwords = control & 0xFFFFF;
        let chain = (control >> 20) & 1 != 0;
        if dwords != 0 {
            if addr == 0 {
                return Err(format!("indirect buffer at address 0 ({dwords} dwords, control 0x{control:08x}) at 0x{:x}", p.addr));
            }
            Self::run_next(ex, addr, dwords, chain);
        }
        Ok(3)
    }

    /// 14-dword conditional branch (COND_INDIRECT_BUFFER): `[mode |
    /// func<<8, cmpAddr, mask64, ref64, then(addr, dwords|cache<<28),
    /// else(addr, dwords|cache<<28)]`. The taken target is called: its
    /// size dwords have no CHAIN bit, and execution returns to the packet
    /// after the branch.
    fn branch(&mut self, ex: &mut Execution, p: &Packet, pl: &[u32]) -> Result<u32> {
        let cmp_addr = (pl[1] & !7) as u64 | (pl[2] as u64) << 32;
        let mask = addr64(pl[3], pl[4]);
        let reference = addr64(pl[5], pl[6]);
        let mode = pl[0] & 3;
        let function = (pl[0] >> 8) & 7;
        let then_buf = (pl[7] & !3) as u64 | (pl[8] as u64) << 32;
        let then_dw = pl[9] & 0xFFFFF;
        let else_buf = (pl[10] & !3) as u64 | (pl[11] as u64) << 32;
        let else_dw = pl[12] & 0xFFFFF;
        // Mode 0 (or no compare address) branches unconditionally.
        let take_then = if cmp_addr == 0 || mode == 0 {
            true
        } else {
            match compare(self.read64c(cmp_addr)?, reference, mask, function) {
                Some(taken) => taken,
                None => return Err(format!("unknown branch compare function {function} at 0x{:x}", p.addr)),
            }
        };
        if take_then {
            if then_dw != 0 {
                if then_buf == 0 {
                    return Err(format!("branch then-target at address 0 at 0x{:x}", p.addr));
                }
                Self::run_next(ex, then_buf, then_dw, false);
            }
        } else if mode == 2 && else_dw != 0 {
            if else_buf == 0 {
                return Err(format!("branch else-target at address 0 at 0x{:x}", p.addr));
            }
            Self::run_next(ex, else_buf, else_dw, false);
        }
        Ok(13)
    }

    fn set_base(&mut self, p: &Packet, pl: &[u32]) -> u32 {
        if pl.len() >= 3 && pl[0] & 0xF == 1 {
            let base = (pl[1] & !7) as u64 | ((pl[2] & 0xFFFF) as u64) << 32;
            if (p.header >> 1) & 1 == 0 {
                self.draw_args_base = base;
            } else {
                self.dispatch_args_base = base;
            }
        }
        p.len() - 1
    }

    // ---- data movement ----

    fn write_values(&self, dst: u64, src: &[u32], one_address: bool) {
        if src.is_empty() {
            return;
        }
        LABEL_WRITES.fetch_add(1, Ordering::Relaxed);
        if one_address {
            self.write32(dst, *src.last().unwrap());
        } else {
            self.write_dwords(dst, src);
        }
    }

    /// Native WRITE_DATA: DST_SEL bits 11:8 (0 register, memory 1/2/4/5),
    /// WR_ONE_ADDR bit 16 (every value to the same address).
    fn write_data_native(&mut self, p: &Packet, pl: &[u32]) -> Result<u32> {
        if pl.len() < 3 {
            return Err(format!("short WRITE_DATA at 0x{:x}", p.addr));
        }
        let control = pl[0];
        let dst_sel = (control >> 8) & 0xF;
        if dst_sel == 0 {
            // A register at absolute dword offset pl[1] (pl[2] = 0): the
            // values go to consecutive registers, or all to that one.
            let values = &pl[3..];
            match Self::absolute_register(pl[1]).filter(|_| pl[2] == 0) {
                Some((bank, offset)) if control & (1 << 16) != 0 => {
                    if let Some(&last) = values.last() {
                        self.write_run(bank, offset, &[last]);
                    }
                }
                Some((bank, offset)) => self.write_run(bank, offset, values),
                None => crate::agc::log_once(&format!("pm4.wd.reg.{:x}", pl[1]), || {
                    log::warn!("[pm4] WRITE_DATA to register {:#x} (high {:#x}) unsupported", pl[1], pl[2])
                }),
            }
            return Ok(p.len() - 1);
        }
        // DST_SEL 3 is guest memory as well (the library's "memory" write
        // destination encodes to it; titles write their frame labels so).
        if matches!(dst_sel, 1 | 2 | 3 | 4 | 5) {
            self.write_values(addr64(pl[1], pl[2]), &pl[3..], control & (1 << 16) != 0);
        } else {
            crate::agc::log_once(&format!("pm4.wd.dst.{dst_sel}"), || log::warn!("[pm4] WRITE_DATA destination {dst_sel} ignored"));
        }
        Ok(p.len() - 1)
    }

    /// Selectors: 0/3 memory, 1 GDS, 2 immediate (source) / nowhere (dest).
    #[allow(clippy::too_many_arguments)]
    fn transfer(&self, dst_sel: u32, dst: u64, src_sel: u32, src: u64, bytes: u32) {
        if bytes == 0 || dst as u32 == DMA_SKIPPED_DESTINATION || dst_sel == 2 {
            return;
        }
        let dst_gds = match dst_sel {
            0 | 3 => false,
            1 => true,
            other => {
                crate::agc::log_once(&format!("pm4.dma.dst.{other}"), || log::warn!("[pm4] DMA destination selector {other} unsupported"));
                return;
            }
        };
        // GDS lives on the GPU next to the shader programs using it.
        if dst_gds || src_sel == 1 {
            if let Some(ex) = &self.executor {
                let op = match (dst_gds, src_sel) {
                    (true, 2) => GdsTransfer::Fill { offset: dst, size: bytes as u64, value: src as u32 },
                    (true, 0 | 3) => GdsTransfer::FromMemory { offset: dst, src, size: bytes as u64 },
                    (false, 1) => GdsTransfer::ToMemory { offset: src, dst, size: bytes as u64 },
                    _ => {
                        crate::agc::log_once("pm4.dma.gds2gds", || log::warn!("[pm4] DMA from GDS to GDS unsupported"));
                        return;
                    }
                };
                if ex.gds_transfer(op) {
                    return;
                }
            }
        }
        // Copies and fills touching GPU-written memory run on the GPU, in
        // order with the work recorded before them.
        if !dst_gds && src_sel != 1 {
            if let Some(ex) = &self.executor {
                let n = bytes as u64;
                let done = match src_sel {
                    2 => Some(ex.fill_memory_or(dst, n, src as u32, Box::new(move || mem::fill32(dst, n, src as u32)))),
                    0 | 3 => {
                        ex.memory_hazard(src, n);
                        Some(ex.copy_memory_or(dst, src, n, Box::new(move || mem::copy(dst, src, n))))
                    }
                    _ => None,
                };
                if let Some(ok) = done {
                    if !ok && rare_warning(&WRITE_WARNINGS) {
                        log::warn!("[pm4] DMA 0x{src:x} (selector {src_sel}) -> 0x{dst:x} ({bytes} bytes) out of guest memory");
                    }
                    return;
                }
            }
        }
        // Memory to memory, or a fill: straight into guest memory.
        if !dst_gds && matches!(src_sel, 0 | 2 | 3) {
            let ok = if src_sel == 2 { mem::fill32(dst, bytes as u64, src as u32) } else { mem::copy(dst, src, bytes as u64) };
            if !ok {
                if rare_warning(&WRITE_WARNINGS) {
                    log::warn!("[pm4] DMA 0x{src:x} (selector {src_sel}) -> 0x{dst:x} ({bytes} bytes) out of guest memory");
                }
                return;
            }
            if src_sel == 2 {
                if let Some(ex) = &self.executor {
                    ex.memory_filled(dst, bytes as u64, src as u32);
                }
            }
            return;
        }
        let data = if src_sel == 2 {
            (src as u32).to_le_bytes().iter().copied().cycle().take(bytes as usize).collect::<Vec<u8>>()
        } else {
            let mut buf = vec![0u8; bytes as usize];
            match src_sel {
                0 | 3 => {
                    self.memory_hazard(src, bytes as u64);
                    if !mem::read(src, &mut buf) {
                        crate::agc::log_once(&format!("pm4.dma.src.{src:x}"), || log::warn!("[pm4] DMA source 0x{src:x} unreadable"));
                        return;
                    }
                }
                1 => gds_read(src as usize, &mut buf),
                other => {
                    crate::agc::log_once(&format!("pm4.dma.src.{other}"), || log::warn!("[pm4] DMA source selector {other} unsupported"));
                    return;
                }
            }
            buf
        };
        if dst_gds {
            gds_write(dst as usize, &data);
        } else if src_sel == 1 && self.executor.as_ref().is_some_and(|ex| ex.write_memory(dst, &data)) {
            // GDS contents written on the GPU, in order with its work.
        } else {
            self.write_bytes(dst, &data);
            if src_sel == 2 {
                if let Some(ex) = &self.executor {
                    ex.memory_filled(dst, bytes as u64, src as u32);
                }
            }
        }
    }

    fn dma_native(&mut self, p: &Packet, pl: &[u32]) -> Result<u32> {
        if pl.len() < 6 {
            return Err(format!("short DMA_DATA at 0x{:x}", p.addr));
        }
        let (control, control2) = (pl[0], pl[5]);
        let src = addr64(pl[1], pl[2]);
        let dst = addr64(pl[3], pl[4]);
        let dst_sel = ((control >> 20) & 3) | ((control2 >> 25) & 4) | ((control2 >> 26) & 8);
        let src_sel = ((control >> 29) & 3) | ((control2 >> 24) & 4) | ((control2 >> 25) & 8);
        self.transfer(dst_sel, dst, src_sel, src, control2 & 0x03FF_FFFF);
        Ok(p.len() - 1)
    }

    fn copy_atomic_return(&self, dst: u64, bytes: u32, pfp: bool) {
        let v = if pfp { self.atomic_return_pfp } else { self.atomic_return_me };
        match v {
            Some(v) if bytes == 8 => self.write64(dst, v),
            Some(v) => self.write32(dst, v as u32),
            None => crate::agc::log_once("pm4.copy.atomic", || log::warn!("[pm4] COPY_DATA of an atomic result with none latched")),
        }
    }

    fn copy_data(&mut self, p: &Packet, pl: &[u32]) -> Result<u32> {
        if pl.len() < 5 {
            return Err(format!("short COPY_DATA at 0x{:x}", p.addr));
        }
        let control = pl[0];
        let bytes = if (control >> 16) & 1 != 0 { 8 } else { 4 };
        let src = addr64(pl[1], pl[2]);
        let dst = addr64(pl[3], pl[4]);
        // Selectors are doubled with the engine bit folded into the source
        // (compute queues have no engine bit).
        let src_sel = ((control & 0xF) << 1) | ((control >> 30) & 1);
        let dst_sel = ((control >> 8) & 0xF) << 1;
        if dst_sel == 0 {
            self.copy_data_to_register(src_sel, src, bytes / 4, pl[3], pl[4]);
            return Ok(p.len() - 1);
        }
        if matches!(src_sel, 9 | 18) {
            if bytes == 8 {
                self.write64(dst, reference_clock());
            } else {
                self.write32(dst, reference_clock() as u32);
            }
            return Ok(p.len() - 1);
        }
        if src_sel == 12 {
            self.copy_atomic_return(dst, bytes, false);
            return Ok(p.len() - 1);
        }
        if src_sel == 0 {
            // A register the queue holds (absolute dword offset).
            let count = bytes / 4;
            match (0..count).map(|i| self.register_value(pl[1].wrapping_add(i))).collect::<Option<Vec<u32>>>() {
                Some(v) if pl[2] == 0 => {
                    if count == 2 {
                        self.write64(dst, v[0] as u64 | (v[1] as u64) << 32);
                    } else {
                        self.write32(dst, v[0]);
                    }
                }
                _ => crate::agc::log_once(&format!("pm4.copy.reg.{:x}", pl[1]), || log::warn!("[pm4] COPY_DATA from register {:#x} unsupported", pl[1])),
            }
            return Ok(p.len() - 1);
        }
        let dma_src = match src_sel {
            2 | 4 | 5 => 3,
            3 | 6 | 7 => 1,
            10 | 11 => 2,
            other => {
                crate::agc::log_once(&format!("pm4.copy.src.{other}"), || log::warn!("[pm4] COPY_DATA source {other} unsupported"));
                return Ok(p.len() - 1);
            }
        };
        let dma_dst = match dst_sel {
            2 | 4 | 5 => 3,
            3 | 6 | 7 => 1,
            other => {
                crate::agc::log_once(&format!("pm4.copy.dst.{other}"), || log::warn!("[pm4] COPY_DATA destination {other} unsupported"));
                return Ok(p.len() - 1);
            }
        };
        if dma_src == 2 && bytes == 8 {
            self.write64(dst, src);
        } else {
            self.transfer(dma_dst, dst, dma_src, src, bytes);
        }
        Ok(p.len() - 1)
    }

    /// COPY_DATA to a register (DST_SEL 0): `count` dwords (1, or 2 with
    /// COUNT_SEL) from memory (SRC_SEL memory or L2) to the registers from
    /// absolute dword offset `reg` on (`reg_hi` = 0).
    fn copy_data_to_register(&mut self, src_sel: u32, src: u64, count: u32, reg: u32, reg_hi: u32) {
        let target = Self::absolute_register(reg).filter(|_| reg_hi == 0);
        let (Some((bank, offset)), true) = (target, matches!(src_sel, 2 | 4)) else {
            crate::agc::log_once(&format!("pm4.copy.toreg.{src_sel}.{reg:x}"), || {
                log::warn!("[pm4] COPY_DATA from source {src_sel} to register {reg:#x} (high {reg_hi:#x}) unsupported")
            });
            return;
        };
        let mut values = [0u32; 2];
        let values = &mut values[..count as usize];
        self.memory_hazard(src, count as u64 * 4);
        if !self.read_ordered(src, bytemuck_dwords(values)) {
            crate::agc::log_once(&format!("pm4.copy.toreg.src.{src:x}"), || log::warn!("[pm4] COPY_DATA register source 0x{src:x} unreadable"));
            return;
        }
        self.write_run(bank, offset, values);
    }

    /// ATOMIC_MEM `[op[6:0] | cmd<<8 | cache<<25 | engine<<30, addr64,
    /// src64, cmp64, loop]`. Command 1 (compare-swap loop) suspends while the
    /// compare fails.
    fn atomic_mem(&mut self, ex: &mut Execution, p: &Packet, pl: &[u32]) -> Result<u32> {
        if pl.len() < 8 {
            return Err(format!("short ATOMIC_MEM at 0x{:x}", p.addr));
        }
        let control = pl[0];
        let raw = control & 0x7F;
        let base = raw & 0x1F;
        let is64 = raw & 0x20 != 0;
        let returns = raw & 0x40 == 0;
        let command = (control >> 8) & 0xF;
        let engine = (control >> 30) & 3;
        let addr = addr64(pl[1], pl[2]);
        let src = addr64(pl[3], pl[4]);
        let cmp = addr64(pl[5], pl[6]);
        let size = if is64 { 8 } else { 4 };
        let integer = matches!(base, 7 | 8 | 15..=25);
        if !integer || addr == 0 || addr & (size - 1) != 0 {
            crate::agc::log_once(&format!("pm4.atomic.{raw:x}"), || {
                log::warn!("[pm4] ATOMIC_MEM op 0x{raw:x} at 0x{addr:x} unsupported");
            });
            return Ok(p.len() - 1);
        }
        let (prior, passed) = {
            let _g = ATOMIC_GATE.lock().unwrap();
            let cur = if is64 { self.read64c(addr)? } else { self.read32c(addr)? as u64 };
            let (new, passed) = apply_atomic(base, cur, src, cmp, is64);
            if is64 {
                self.write64(addr, new);
            } else {
                self.write32(addr, new as u32);
            }
            (cur, passed)
        };
        if returns {
            if !self.is_compute && engine == 1 {
                self.atomic_return_pfp = Some(prior);
            } else {
                self.atomic_return_me = Some(prior);
            }
        }
        if command == 1 && !passed {
            Self::suspend(ex);
        }
        Ok(p.len() - 1)
    }

    /// MEM_SEMAPHORE `[addr64, control]`: select 6 signals (+1, or set to 1
    /// with the write-signal bit), 7 waits (takes one token or suspends).
    fn mem_semaphore(&mut self, ex: &mut Execution, p: &Packet, pl: &[u32]) -> Result<u32> {
        if pl.len() < 3 {
            return Err(format!("short MEM_SEMAPHORE at 0x{:x}", p.addr));
        }
        let addr = (pl[0] & !7) as u64 | (pl[1] as u64) << 32;
        let control = pl[2];
        let select = (control >> 29) & 7;
        if addr == 0 {
            return Err(format!("MEM_SEMAPHORE on address 0 at 0x{:x}", p.addr));
        }
        let _g = ATOMIC_GATE.lock().unwrap();
        let cur = self.read64c(addr)?;
        match select {
            6 => self.write64(addr, if control & (1 << 20) != 0 { 1 } else { cur.wrapping_add(1) }),
            7 => {
                if cur == 0 {
                    Self::suspend(ex);
                } else {
                    self.write64(addr, cur - 1);
                }
            }
            other => crate::agc::log_once(&format!("pm4.sem.{other}"), || log::warn!("[pm4] semaphore select {other} unsupported")),
        }
        Ok(p.len() - 1)
    }

    // ---- registers ----

    /// One register write (indirect-draw state; out-of-bank offsets are
    /// ignored with a warning).
    fn write_register(&mut self, bank: Bank, offset: u32, value: u32) {
        match self.regs.write(bank, offset, value) {
            Write::Written => {
                if bank == Bank::UserConfig && offset == super::VGT_INDEX_TYPE {
                    self.index_type = value & 3;
                }
            }
            Write::Nop => {}
            Write::OutOfBank => {
                if rare_warning(&OUT_OF_BANK_WARNINGS) {
                    log::warn!("[pm4] {bank:?} register 0x{offset:x} outside its bank; ignored");
                }
            }
        }
    }

    fn set_registers(&mut self, bank: Bank, p: &Packet, pl: &[u32]) -> Result<u32> {
        let rest = p.len() - 1;
        if bank == Bank::UserConfig && super::is_internal_data_packet(p.header, pl) {
            return Ok(rest);
        }
        let raw = pl[0];
        // The selector bits (30:28) encode packet controls, not the
        // register, in every bank; SET_UCONFIG_REG_INDEX uses 31:28.
        let offset = if super::opcode(p.header) == op::SET_UCONFIG_REG_INDEX { raw & 0x0FFF_FFFF } else { raw & !super::REGISTER_SELECTOR_MASK };
        self.write_run(bank, offset, &pl[1..]);
        Ok(rest)
    }

    /// Consecutive register writes from `offset` (logged for the slice
    /// recorder; VGT_INDEX_TYPE updates the index type).
    fn write_run(&mut self, bank: Bank, offset: u32, values: &[u32]) {
        if let Some(s) = self.slices() {
            s.log_run(self.queue_id, &self.regs, bank, offset, values);
        }
        if self.regs.set_run(bank, offset, values) && rare_warning(&OUT_OF_BANK_WARNINGS) {
            log::warn!("[pm4] {bank:?} registers 0x{offset:x}+{} outside its bank; ignored", values.len());
        }
        let end = offset as u64 + values.len() as u64;
        if bank == Bank::UserConfig && (offset as u64..end).contains(&(super::VGT_INDEX_TYPE as u64)) {
            self.index_type = self.regs.uconfig[super::VGT_INDEX_TYPE as usize] & 3;
        }
    }

    /// Native indirect form: `[addr64 (4-aligned), first, count]`. With
    /// `first` = 0x80000000 the memory holds `count` `{offset, value}`
    /// pairs; otherwise `count` values for the registers from `first` on.
    fn register_table_native(&mut self, bank: Bank, p: &Packet, pl: &[u32]) -> Result<u32> {
        if p.len() != 5 {
            return Err(format!("register table length {} unsupported at 0x{:x}", p.len(), p.addr));
        }
        let addr = (pl[0] & !3) as u64 | (pl[1] as u64) << 32;
        let count = pl[3] & 0x3FFF;
        if pl[2] == 0x8000_0000 {
            self.apply_register_table(bank, addr, count)?;
        } else {
            self.apply_register_values(bank, addr, pl[2] & !super::REGISTER_SELECTOR_MASK, count)?;
        }
        Ok(4)
    }

    /// `count` consecutive register values read from guest memory at
    /// `addr`, written from register `first` on.
    fn apply_register_values(&mut self, bank: Bank, addr: u64, first: u32, count: u32) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        if addr == 0 || !mem::accessible(addr, count as u64 * 4, false) {
            return Err(format!("cannot read {bank:?} register values 0x{addr:x} ({count} dwords)"));
        }
        // SAFETY: the whole range was checked readable just above.
        let values = unsafe { std::slice::from_raw_parts(addr as *const u32, count as usize) };
        self.write_run(bank, first, values);
        Ok(())
    }

    /// Bank and in-bank offset of an absolute register dword offset
    /// (shader 0x2C00.., context 0xA000.., user config 0xC000..).
    fn absolute_register(reg: u32) -> Option<(Bank, u32)> {
        match reg {
            0x2C00..=0x2FFF => Some((Bank::Shader, reg - 0x2C00)),
            0xA000..=0xA3FF => Some((Bank::Context, reg - 0xA000)),
            0xC000..=0xFFFF => Some((Bank::UserConfig, reg - 0xC000)),
            _ => None,
        }
    }

    /// Apply a guest table of `{u32 offset, u32 value}` entries in order.
    fn apply_register_table(&mut self, bank: Bank, table: u64, count: u32) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        if table == 0 {
            return Err(format!("{bank:?} register table at address 0"));
        }
        let words = count as usize * 2;
        if !mem::accessible(table, words as u64 * 4, false) {
            return Err(format!("cannot read register table 0x{table:x} ({count} entries)"));
        }
        // SAFETY: the whole table was checked readable just above.
        let entries = unsafe { std::slice::from_raw_parts(table as *const u32, words) };
        if log_tables() {
            log_table(bank, table, entries);
        }
        // Values copied into the slice log: the table is read here, once.
        if let Some(s) = self.slices() {
            s.log_table(self.queue_id, &self.regs, bank, entries);
        }
        // The selector bits encode packet controls, not the register.
        if self.regs.table(bank, entries) && rare_warning(&OUT_OF_BANK_WARNINGS) {
            log::warn!("[pm4] {bank:?} register table 0x{table:x} writes outside its bank; ignored");
        }
        if bank == Bank::UserConfig
            && entries.chunks_exact(2).any(|e| e[0] != u32::MAX && e[0] & !super::REGISTER_SELECTOR_MASK == super::VGT_INDEX_TYPE)
        {
            self.index_type = self.regs.uconfig[super::VGT_INDEX_TYPE as usize] & 3;
        }
        Ok(())
    }
}

/// Decode a dispatch packet: DISPATCH_DIRECT `[x, y, z, initiator]`,
/// DISPATCH_INDIRECT `[offset, initiator]` (graphics, from SET_BASE) or
/// `[addr64, initiator]` (compute queues).
fn decode_dispatch(queue: usize, dispatch_args_base: u64, p: &Packet, pl: &[u32]) -> DispatchPacket {
    let at = |i: usize| pl.get(i).copied().unwrap_or(0);
    let mut packet = DispatchPacket { queue, packet_address: p.addr, ..Default::default() };
    if super::opcode(p.header) == op::DISPATCH_DIRECT {
        packet.groups = [at(0), at(1), at(2)];
        packet.dispatch_initiator = at(3);
    } else if p.len() >= 4 {
        packet.indirect_address = addr64(at(0), at(1));
        packet.dispatch_initiator = at(2);
    } else {
        packet.indirect_address = dispatch_args_base + at(0) as u64;
        packet.dispatch_initiator = at(1);
    }
    packet
}

/// Packets the lookahead decodes at most per scan (a bound on the work of
/// one scan through streams with few dispatches).
const AHEAD_PACKETS: u64 = 1 << 16;

/// The command stream after the packet being executed, for the compile
/// lookahead (`ps5gpu::pipeline::lookahead`): the scan decodes the
/// following packets on a copy of the queue's registers and resolves their
/// dispatches. It executes nothing: no memory, label or register writes,
/// no interrupts or flips. It stops at the first packet whose effect it
/// cannot know without executing (a wait, a conditional, a branch, a
/// register load the command processor produces), at a flip, and at the
/// limit.
struct Ahead<'a> {
    regs: &'a RegisterFile,
    queue: usize,
    dispatch_args_base: u64,
    predicate_skip: bool,
    stack: &'a [Cursor],
    /// The packet being executed and its length (the scan starts after it).
    at: u64,
    len: u32,
}

impl lookahead::Source for Ahead<'_> {
    fn position(&self) -> u64 {
        self.at
    }

    fn scan(&self, limit: usize, out: &mut Vec<ps5gpu::render::ResolvedDispatch>) {
        let Some(top) = self.stack.last() else { return };
        // The packet being executed is the one at the top cursor's offset.
        if top.addr + top.offset as u64 * 4 != self.at {
            return;
        }
        let mut stack: Vec<Cursor> = self.stack.to_vec();
        stack.last_mut().expect("non-empty").offset += self.len;
        let mut regs = self.regs.clone();
        let mut dispatch_args_base = self.dispatch_args_base;
        let start = out.len();
        let mut packets = 0u64;
        while let Some(&cur) = stack.last() {
            if out.len() - start >= limit || packets >= AHEAD_PACKETS {
                return;
            }
            packets += 1;
            if cur.offset >= cur.dwords {
                stack.pop();
                continue;
            }
            let paddr = cur.addr + cur.offset as u64 * 4;
            // SAFETY: the buffer's range was checked when it was entered.
            let h = unsafe { std::ptr::read_volatile(paddr as *const u32) };
            if packet_size(h) == 1 {
                stack.last_mut().expect("non-empty").offset += 1;
                continue;
            }
            let remaining = cur.remaining();
            if remaining < 2 || super::packet_type(h) != 3 {
                return;
            }
            let len = length(h);
            if len > remaining {
                return;
            }
            stack.last_mut().expect("non-empty").offset += len;
            if predicated(h) && self.predicate_skip {
                continue;
            }
            // SAFETY: inside the checked buffer range (len <= remaining).
            let pl = unsafe { std::slice::from_raw_parts((paddr + 4) as *const u32, (len - 1) as usize) };
            let p = Packet { header: h & !1, addr: paddr, remaining };
            match super::opcode(p.header) {
                op::SET_CONTEXT_REG | op::SET_SH_REG | op::SET_UCONFIG_REG | op::SET_UCONFIG_REG_INDEX => {
                    let bank = match super::opcode(p.header) {
                        op::SET_CONTEXT_REG => Bank::Context,
                        op::SET_SH_REG => Bank::Shader,
                        _ => Bank::UserConfig,
                    };
                    if pl.is_empty() || (bank == Bank::UserConfig && super::is_internal_data_packet(p.header, pl)) {
                        continue;
                    }
                    let offset = if super::opcode(p.header) == op::SET_UCONFIG_REG_INDEX { pl[0] & 0x0FFF_FFFF } else { pl[0] & !super::REGISTER_SELECTOR_MASK };
                    regs.set_run(bank, offset, &pl[1..]);
                }
                op::SET_CONTEXT_REG_INDIRECT | op::SET_SH_REG_INDIRECT | op::SET_UCONFIG_REG_INDIRECT => {
                    let bank = match super::opcode(p.header) {
                        op::SET_CONTEXT_REG_INDIRECT => Bank::Context,
                        op::SET_SH_REG_INDIRECT => Bank::Shader,
                        _ => Bank::UserConfig,
                    };
                    if p.len() != 5 {
                        return;
                    }
                    let addr = (pl[0] & !3) as u64 | (pl[1] as u64) << 32;
                    let count = pl[3] & 0x3FFF;
                    let words = if pl[2] == 0x8000_0000 { count as u64 * 2 } else { count as u64 };
                    if count == 0 {
                        continue;
                    }
                    if addr == 0 || !mem::accessible(addr, words * 4, false) {
                        return;
                    }
                    // SAFETY: checked readable just above.
                    let values = unsafe { std::slice::from_raw_parts(addr as *const u32, words as usize) };
                    if pl[2] == 0x8000_0000 {
                        regs.table(bank, values);
                    } else {
                        regs.set_run(bank, pl[2] & !super::REGISTER_SELECTOR_MASK, values);
                    }
                }
                op::CLEAR_STATE => {
                    let _ = regs.context_state(ContextState::Clear);
                }
                op::SET_BASE => {
                    if pl.len() >= 3 && pl[0] & 0xF == 1 && (p.header >> 1) & 1 != 0 {
                        dispatch_args_base = (pl[1] & !7) as u64 | ((pl[2] & 0xFFFF) as u64) << 32;
                    }
                }
                op::DISPATCH_DIRECT | op::DISPATCH_INDIRECT => {
                    let packet = decode_dispatch(self.queue, dispatch_args_base, &p, pl);
                    let dims = if packet.indirect_address == 0 { packet.groups } else { [0; 3] };
                    regs.set_run(Bank::Shader, 0x200, &[packet.dispatch_initiator, dims[0], dims[1], dims[2]]);
                    out.push(ps5gpu::render::resolve_dispatch(&regs.banks(), &packet));
                }
                op::INDIRECT_BUFFER => {
                    if p.len() != 4 {
                        // A conditional branch.
                        return;
                    }
                    let addr = pl[0] as u64 | (pl[1] as u64) << 32;
                    let dwords = pl[2] & 0xFFFFF;
                    let chain = (pl[2] >> 20) & 1 != 0;
                    if dwords == 0 {
                        continue;
                    }
                    if check_buffer(addr, dwords).is_err() {
                        return;
                    }
                    let cursor = Cursor::new(addr, dwords);
                    let top = stack.last_mut().expect("non-empty");
                    if chain || top.offset == top.dwords {
                        *top = cursor;
                    } else if stack.len() >= MAX_DEPTH {
                        return;
                    } else {
                        stack.push(cursor);
                    }
                }
                op::NOP => match custom(p.header) {
                    nop::CONTEXT_STATE => {
                        let operation = match pl.first() {
                            Some(0) => ContextState::Clear,
                            Some(1) => ContextState::Push,
                            Some(2) => ContextState::Pop,
                            Some(3) => ContextState::PushClear,
                            _ => return,
                        };
                        let _ = regs.context_state(operation);
                    }
                    nop::DISPATCH_RESET => {
                        regs.reset();
                        dispatch_args_base = 0;
                    }
                    // A flip ends the frame the lookahead is for.
                    nop::FLIP | nop::WAIT_FLIP_DONE => return,
                    nop::ZERO if pl.first().is_some_and(|&f| f & 0xFFFF_0000 == MARKER_MAGIC && matches!(f & 0xFFF, 0x777 | 0x778 | 0x781)) => return,
                    _ => {}
                },
                // COPY_DATA into a register.
                op::COPY_DATA if pl.first().is_some_and(|&c| (c >> 8) & 0xF == 0) => return,
                // Packets that wait, choose what runs next or load state
                // the command processor itself produces.
                op::WAIT_REG_MEM
                | op::WAIT_REG_MEM64
                | op::MEM_SEMAPHORE
                | op::WAIT_ON_CE_COUNTER
                | op::WAIT_ON_DE_COUNTER_DIFF
                | op::REWIND
                | op::COND_EXEC
                | op::SET_PREDICATION
                | op::INDEX_ATTRIBUTES_INDIRECT => return,
                // Everything else writes memory or signals: no state the
                // dispatches after it are resolved from.
                _ => {}
            }
        }
    }
}

static OUT_OF_BANK_WARNINGS: AtomicU64 = AtomicU64::new(0);
static NOP_WARNINGS: AtomicU64 = AtomicU64::new(0);
static UNHANDLED_OPCODES: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static WRITE_WARNINGS: AtomicU64 = AtomicU64::new(0);

/// The first 16 occurrences of a warning that can repeat per draw.
fn rare_warning(counter: &AtomicU64) -> bool {
    counter.load(Ordering::Relaxed) < 16 && counter.fetch_add(1, Ordering::Relaxed) < 16
}

/// `RPS5E_PM4_TABLES`: log each register table once (diagnostics).
fn log_table(bank: Bank, table: u64, entries: &[u32]) {
    crate::agc::log_once(&format!("pm4.table.{bank:?}.{table:x}"), || {
        let s: Vec<String> = entries.chunks_exact(2).take(64).map(|e| format!("{:x}={:x}", e[0], e[1])).collect();
        log::info!("[pm4-table] {bank:?} @0x{table:x} n={}: {}", entries.len() / 2, s.join(" "));
    });
}

/// `RPS5E_PM4_TABLES` (read once: register tables are applied per draw).
fn log_tables() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("RPS5E_PM4_TABLES").is_some())
}

/// Integer ATOMIC_MEM operations. Returns (new value, compare passed).
fn apply_atomic(op: u32, cur: u64, src: u64, cmp: u64, is64: bool) -> (u64, bool) {
    let mask = if is64 { u64::MAX } else { 0xFFFF_FFFF };
    let (cur, src, cmp) = (cur & mask, src & mask, cmp & mask);
    let signed = |v: u64| if is64 { v as i64 } else { v as u32 as i32 as i64 };
    let passed = op != 8 || cur == cmp;
    let new = match op {
        7 => src,
        8 => {
            if passed {
                src
            } else {
                cur
            }
        }
        15 => cur.wrapping_add(src),
        16 => cur.wrapping_sub(src),
        17 => {
            if signed(src) < signed(cur) {
                src
            } else {
                cur
            }
        }
        18 => src.min(cur),
        19 => {
            if signed(src) > signed(cur) {
                src
            } else {
                cur
            }
        }
        20 => src.max(cur),
        21 => cur & src,
        22 => cur | src,
        23 => cur ^ src,
        24 => {
            if cur >= src {
                0
            } else {
                cur + 1
            }
        }
        25 => {
            if cur == 0 || cur > src {
                src
            } else {
                cur - 1
            }
        }
        _ => cur,
    };
    (new & mask, passed)
}

/// `RPS5E_PM4_DUMP=n`: log the non-zero register state of the first `n`
/// draws (diagnostics).
fn dump_registers(i: &Interpreter, packet: &DrawPacket) {
    static LEFT: std::sync::OnceLock<AtomicU64> = std::sync::OnceLock::new();
    let left = LEFT.get_or_init(|| AtomicU64::new(std::env::var("RPS5E_PM4_DUMP").ok().and_then(|v| v.parse().ok()).unwrap_or(0)));
    if left.load(Ordering::Relaxed) == 0 || left.fetch_sub(1, Ordering::Relaxed) == 0 {
        return;
    }
    let nz = |bank: &[u32]| -> String {
        bank.iter().enumerate().filter(|(_, v)| **v != 0).map(|(o, v)| format!("{o:x}={v:x}")).collect::<Vec<_>>().join(" ")
    };
    log::info!(
        "[pm4-dump] q{} {:?} @0x{:x}\n  cx: {}\n  sh: {}\n  uc: {}",
        i.queue_id,
        packet.kind,
        packet.packet_address,
        nz(&i.regs.context),
        nz(&i.regs.shader),
        nz(&i.regs.uconfig)
    );
}

/// Header of a packet the interpreter treats as a NOP of `dwords` dwords.
pub fn nop_header(dwords: u32) -> u32 {
    make(dwords, op::NOP, nop::ZERO)
}

/// Interrupt selectors 5/6 of RELEASE_MEM: the label at `dst` is readable
/// and at or below `data` (`wide`: 64-bit compare, else the low 32 bits).
fn conditional_interrupt_due(dst: u64, data: u64, wide: bool) -> bool {
    if wide {
        mem::r64(dst).is_some_and(|label| label <= data)
    } else {
        mem::r32(dst).is_some_and(|label| label <= data as u32)
    }
}
