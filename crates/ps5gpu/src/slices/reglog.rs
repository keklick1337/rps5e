// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The slice log: what the sequencer hands a slice thread instead of
//! resolved draws (GPU_PLAN §53: a resolved draw is ~3–4 KiB of cold data;
//! the register writes and packet that produce it are a few hundred
//! bytes, written and read sequentially).

use crate::render::regfile::{Bank, ContextState, RegisterFile};
use crate::render::{DispatchPacket, DrawKind, DrawPacket, ResolvedDispatch, ResolvedDraw, SH_NOP};

/// Where a draw's instance count comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstanceSource {
    /// The packet carries it.
    Packet,
    /// NUM_INSTANCES state, at least 1.
    State,
    /// NUM_INSTANCES state as it is (a preamble draw without its own count).
    StateRaw,
}

/// A draw as the sequencer decoded it: the packet, with the parts that
/// depend on state the slice thread owns (instance count, indirect
/// arguments read from memory) left to the slice thread.
#[derive(Clone, Copy, Debug)]
pub struct DrawRecord {
    pub packet: DrawPacket,
    pub instances: InstanceSource,
    /// Non-zero: the draw's last argument record is read on the slice
    /// thread and written to the shader registers and instance count it
    /// names ([`apply_fixup`]); the id finds the values again when later
    /// slices start from a snapshot that carries them.
    pub fixup: u64,
}

/// One log entry, in sequencer order.
pub enum Entry {
    /// `values[at..at+len]` written from register `offset` on.
    Run { bank: Bank, offset: u32, at: u32, len: u32 },
    /// `values[at..at+len]` as `{offset, value}` table pairs.
    Table { bank: Bank, at: u32, len: u32 },
    ContextState(ContextState),
    Reset,
    NumInstances(u32),
    /// `draws[i]`.
    Draw(u32),
    /// `dispatches[i]`.
    Dispatch(u32),
    /// A draw / dispatch resolved by the caller.
    Resolved(Box<ResolvedDraw>),
    ResolvedDispatch(Box<ResolvedDispatch>),
    EndOfPipe(Box<dyn FnOnce() + Send>),
    AfterPipe(Box<dyn FnOnce() + Send>),
    MemoryFilled(u64, u64, u32),
    Flush,
    SubmissionComplete,
    SyncPoint,
    /// Ordered command-processor work (writes, transfers, flips).
    Call(Box<dyn FnOnce() + Send>),
}

/// The indirect-draw register writes a fixup carries into a snapshot,
/// with the targets the sequencer has not overwritten since.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CarriedFixup {
    pub id: u64,
    /// Vertex offset, instance offset, first index, draw index.
    pub targets: [u32; 4],
    /// Bit i: `targets[i]` still holds the fixup's value; bit 4: the
    /// instance count does.
    pub live: u8,
}

pub const FIXUP_INSTANCES: u8 = 1 << 4;

impl CarriedFixup {
    pub fn new(id: u64, packet: &DrawPacket) -> CarriedFixup {
        let r = packet.indirect_registers;
        let targets = [r.vertex_offset, r.instance_offset, r.first_index, r.draw_index_target()];
        let mut live = FIXUP_INSTANCES;
        for (i, &t) in targets.iter().enumerate() {
            // A write the shader bank ignores is never carried.
            if t != SH_NOP && t < crate::render::regfile::SHADER_REGISTER_COUNT {
                live |= 1 << i;
            }
        }
        CarriedFixup { id, targets, live }
    }

    /// Shader registers `[offset, offset+n)` were written.
    #[inline]
    pub fn overwritten(&mut self, offset: u32, n: u32) {
        for (i, &t) in self.targets.iter().enumerate() {
            if t.wrapping_sub(offset) < n {
                self.live &= !(1 << i);
            }
        }
    }
}

/// The shader registers a fixup targets and the instance count, as the
/// fixup left them (written, or unchanged when it wrote nothing: no
/// record, an unreadable one, a zero count). Carried slots take these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixupValues {
    pub values: [u32; 4],
    pub instances: u32,
}

/// The fixups a snapshot carries: the last one to target each register (or
/// the instance count) since the sequencer wrote it.
pub fn note_fixup(active: &mut Vec<CarriedFixup>, new: CarriedFixup) {
    for f in active.iter_mut() {
        for (i, &t) in new.targets.iter().enumerate() {
            if new.live & (1 << i) != 0 {
                f.overwritten(t, 1);
            }
        }
        f.live &= !FIXUP_INSTANCES;
    }
    active.retain(|f| f.live != 0);
    active.push(new);
}

/// Shader registers `[offset, offset+n)` were written by the sequencer.
pub fn note_shader_writes(active: &mut Vec<CarriedFixup>, offset: u32, n: u32) {
    if active.is_empty() {
        return;
    }
    for f in active.iter_mut() {
        f.overwritten(offset, n);
    }
    active.retain(|f| f.live != 0);
}

/// The sequencer set the instance count.
pub fn note_instances(active: &mut Vec<CarriedFixup>) {
    if active.is_empty() {
        return;
    }
    for f in active.iter_mut() {
        f.live &= !FIXUP_INSTANCES;
    }
    active.retain(|f| f.live != 0);
}

/// Register state of a queue where a slice starts.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub context: Vec<u32>,
    pub shader: Vec<u32>,
    /// Every user-config register written since the reset.
    pub uconfig: Vec<(u16, u32)>,
    pub saved_context: Vec<u32>,
    pub context_pushed: bool,
    pub num_instances: u32,
    /// Indirect-draw writes the sequencer's banks do not hold.
    pub fixups: Vec<CarriedFixup>,
}

impl Snapshot {
    /// Copy the state of `regs` (allocations reused).
    pub fn take(&mut self, regs: &RegisterFile, fixups: &[CarriedFixup]) {
        self.context.clear();
        self.context.extend_from_slice(&regs.context);
        self.shader.clear();
        self.shader.extend_from_slice(&regs.shader);
        self.uconfig.clear();
        self.uconfig.extend(regs.uconfig_written());
        self.context_pushed = regs.context_pushed;
        self.saved_context.clear();
        if regs.context_pushed {
            self.saved_context.extend_from_slice(&regs.saved_context);
        }
        self.num_instances = regs.num_instances;
        self.fixups.clear();
        self.fixups.extend_from_slice(fixups);
    }

    /// The registers a slice starts with: the sequencer's state, then the
    /// carried fixups' live writes (`values` finds a fixup's values).
    pub fn load_into(&self, regs: &mut RegisterFile, values: &dyn Fn(u64) -> Option<FixupValues>) -> bool {
        regs.load(&self.context, &self.shader, &self.uconfig, self.context_pushed.then_some(&self.saved_context[..]), self.num_instances);
        let mut complete = true;
        for f in &self.fixups {
            match values(f.id) {
                Some(v) => {
                    for (i, &t) in f.targets.iter().enumerate() {
                        if f.live & (1 << i) != 0 {
                            regs.write(Bank::Shader, t, v.values[i]);
                        }
                    }
                    if f.live & FIXUP_INSTANCES != 0 {
                        regs.num_instances = v.instances;
                    }
                }
                // Not executed yet (a later stage re-reads it).
                None => complete = false,
            }
        }
        complete
    }
}

/// A run of the sequencer's work for one queue, executed in order by a
/// slice thread.
#[derive(Default)]
pub struct Slice {
    pub queue: usize,
    /// Present once the slice holds register work or draws.
    pub snapshot: Option<Box<Snapshot>>,
    pub entries: Vec<Entry>,
    pub values: Vec<u32>,
    pub draws: Vec<DrawRecord>,
    pub dispatches: Vec<DispatchPacket>,
    /// Why the sequencer closed it.
    pub cut: Cut,
    /// A snapshot allocation kept for the next use.
    pub spare: Option<Box<Snapshot>>,
}

impl Slice {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Empty it for reuse (keeps the allocations, the snapshot's too).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.values.clear();
        self.draws.clear();
        self.dispatches.clear();
        self.cut = Cut::Open;
        if let Some(s) = self.snapshot.take() {
            self.spare = Some(s);
        }
    }

    pub fn push_run(&mut self, bank: Bank, offset: u32, values: &[u32]) {
        let at = self.values.len() as u32;
        self.values.extend_from_slice(values);
        self.entries.push(Entry::Run { bank, offset, at, len: values.len() as u32 });
    }

    pub fn push_table(&mut self, bank: Bank, pairs: &[u32]) {
        let at = self.values.len() as u32;
        self.values.extend_from_slice(pairs);
        self.entries.push(Entry::Table { bank, at, len: pairs.len() as u32 });
    }

    pub fn push_draw(&mut self, d: DrawRecord) {
        self.entries.push(Entry::Draw(self.draws.len() as u32));
        self.draws.push(d);
    }

    pub fn push_dispatch(&mut self, d: DispatchPacket) {
        self.entries.push(Entry::Dispatch(self.dispatches.len() as u32));
        self.dispatches.push(d);
    }

    /// Bytes the slice's log occupies (entries, values, draws).
    pub fn log_bytes(&self) -> usize {
        self.entries.len() * std::mem::size_of::<Entry>()
            + self.values.len() * 4
            + self.draws.len() * std::mem::size_of::<DrawRecord>()
            + self.dispatches.len() * std::mem::size_of::<DispatchPacket>()
    }
}

/// Why a slice ended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Cut {
    #[default]
    Open,
    /// A dispatch, end of pipe or flip (serial in later stages).
    Hard,
    /// The draw budget, at a render-target change.
    Soft,
    /// The budget's grace ran out without a target change.
    Forced,
    /// End of a submission (flush, completion).
    Flush,
    /// Another queue's register work.
    Queue,
    /// Someone waits for the work queued so far.
    Drain,
}

impl Cut {
    pub const COUNT: usize = 7;
    pub const NAMES: [&'static str; Self::COUNT] = ["open", "hard", "soft", "forced", "flush", "queue", "drain"];
}

/// Guest bytes at `addr` (the command processor's view of guest memory).
pub fn guest_read(addr: u64, out: &mut [u8]) -> bool {
    if out.is_empty() {
        return true;
    }
    if addr == 0 || !ps5vm::ranges::accessible(addr, out.len() as u64, false) {
        return false;
    }
    // SAFETY: the range is a tracked, readable guest mapping.
    unsafe { std::ptr::copy_nonoverlapping(addr as *const u8, out.as_mut_ptr(), out.len()) };
    true
}

/// The draw state an indirect draw leaves behind: its last argument record
/// writes its offsets (and a multi-draw its index) to the destination
/// shader registers and sets the instance count (the draws themselves read
/// the records on the GPU). The sequencer logs it only with an argument
/// base set.
pub fn apply_fixup(packet: &DrawPacket, regs: &mut RegisterFile, read: &dyn Fn(u64, &mut [u8]) -> bool) -> FixupValues {
    write_fixup(packet, regs, read);
    let r = packet.indirect_registers;
    let at = |t: u32| regs.shader.get(t as usize).copied().unwrap_or(0);
    FixupValues { values: [at(r.vertex_offset), at(r.instance_offset), at(r.first_index), at(r.draw_index_target())], instances: regs.num_instances }
}

fn write_fixup(packet: &DrawPacket, regs: &mut RegisterFile, read: &dyn Fn(u64, &mut [u8]) -> bool) {
    let indexed = packet.kind.indexed();
    let mut count = packet.indirect_count;
    if packet.count_address != 0 {
        let mut b = [0u8; 4];
        if !read(packet.count_address, &mut b) {
            return;
        }
        count = count.min(u32::from_le_bytes(b));
    }
    if count == 0 {
        return;
    }
    let args_bytes = if indexed { 20 } else { 16 };
    let stride = if matches!(packet.kind, DrawKind::IndirectMulti | DrawKind::IndexIndirectMulti) { packet.stride as u64 } else { args_bytes };
    if stride < args_bytes {
        return;
    }
    let mut raw = [0u8; 20];
    let record = packet.indirect_address + (count as u64 - 1) * stride;
    if !read(record, &mut raw[..args_bytes as usize]) {
        return;
    }
    let args: [u32; 5] = std::array::from_fn(|i| u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap()));
    // Non-indexed: [vertices, instances, first vertex, first instance];
    // indexed: [indices, instances, first index, base vertex, first instance].
    let (instances, vertex_offset, instance_offset, first_index) =
        if indexed { (args[1], args[3], args[4], args[2]) } else { (args[1], args[2], args[3], 0) };
    let r = packet.indirect_registers;
    regs.write(Bank::Shader, r.vertex_offset, vertex_offset);
    regs.write(Bank::Shader, r.instance_offset, instance_offset);
    regs.write(Bank::Shader, r.first_index, first_index);
    regs.write(Bank::Shader, r.draw_index_target(), count - 1);
    regs.num_instances = instances;
}
