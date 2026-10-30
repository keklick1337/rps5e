// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! A plan's per-draw evaluation compiled into a flat register program.
//!
//! The memoising [`RuntimeValueEvaluator`](super::RuntimeValueEvaluator)
//! walks the value graph through hash maps on every draw. For a plan whose
//! materialisation is a pure function of its descriptor words, flattened
//! table words, device-address bases and resource-branch predicates (no
//! indirect images, no runtime V# tables, no clean-only table slots), the
//! same walk is
//! compiled once into a list of operations in dependency order; a draw then
//! runs the list over a register file: user data, the shader base and the
//! guest words the reads reach.
//!
//! The program follows the evaluator exactly: strict operands, the active
//! mask of first-lane reads, invariant phis, the raw-read bounds and the
//! null-base rule, and branch predicates read through the clean reader
//! (an unreadable predicate keeps both edges; descriptors of sources no
//! path reaches are zero and never evaluated). A value the evaluator cannot
//! produce becomes a failing register, and every register derived from it
//! fails too.

use super::evaluator::{resolve_raw_address, GpuWrittenProbe, RawAddress, ResidentReader, WordReader, ADDRESS_MASK};
use super::plan::ShaderResourcePlan;
use super::value::{evaluate_operation, ScalarOp, ValueId, ValueKind};
use crate::inline_constants;
use std::collections::HashMap;

/// One register's computation. Operands name earlier registers.
#[derive(Clone, Copy, Debug)]
enum Op {
    Constant(u64),
    Fail,
    /// User-data dword (index from the plan's base).
    UserData(u32),
    ShaderBase,
    Select(u32, u32, u32),
    Operation { op: ScalarOp, operands: [u32; 4], count: u8 },
    /// A raw scalar read; `records`/`format` only for buffer reads;
    /// `clean`: through the clean reader (a refusal fails the value).
    /// `block`: the number of reads starting here that share every operand
    /// and read consecutive dwords (see [`form_blocks`]; 1: a lone read,
    /// 0: a later read of a block). `deferrable`: the read is the whole
    /// value of table slots nothing else uses (`ReplayProgram::deferrable`).
    Read { kind: ValueKind, low: u32, high: u32, offset: u32, records: u32, format: u32, immediate: i64, clean: bool, block: u8, deferrable: bool },
}

const NONE: u32 = u32::MAX;

/// Reads at most one block covers (dwords).
const MAX_BLOCK: usize = 32;

/// Where one draw's guest words come from.
pub enum ReplayReader<'a> {
    /// Synchronising word reader (may download GPU data) and the clean
    /// reader for branch predicates.
    Words(WordReader<'a>, Option<WordReader<'a>>),
    /// Resident bytes only: a refused general read aborts the run; branch
    /// predicates read it as their clean reader when `has_clean`.
    Resident(ResidentReader<'a>, bool),
}

/// Why a run produced no words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayAbort {
    /// A read through a non-null base could not be served by the reader.
    Unreadable,
    /// A required output failed (the evaluator would fail too).
    Failed,
}

/// The compiled evaluation of one plan.
#[derive(Debug)]
pub struct ReplayProgram {
    ops: Vec<Op>,
    /// Descriptor words of `plan.materialization_sources`, flattened in
    /// order, then the flattened table (`table_len` words).
    outputs: Vec<u32>,
    /// Descriptor source of each descriptor output word.
    output_sources: Vec<u32>,
    /// Resource branches: predicate register (clean reads), successors,
    /// the sources the block uses.
    branches: Vec<(Option<u32>, Vec<usize>, Vec<u32>)>,
    source_count: usize,
    /// Each materialisation source's words in the descriptor outputs
    /// (start, end), in materialisation order.
    descriptor_spans: Vec<(u32, u32)>,
    pub descriptor_words: usize,
    pub table_len: usize,
    /// Per device-address range: (low, high) registers when plannable.
    ranges: Vec<Option<(u32, u32)>>,
    pub user_data_len: usize,
    /// Per op: the table slots it is the whole value of when nothing else
    /// uses it (a plain read that may be deferred to the GPU), or, for a
    /// read in [`Self::records_buffer`], the table slots it fills besides
    /// that NUM_RECORDS; else empty.
    deferrable: Vec<Vec<u32>>,
    /// Per op: the plan buffer whose V# NUM_RECORDS (word 2) is the op's
    /// only use besides table slots ([`NONE`]: none). While GPU work still
    /// writes such a word, the program reads it itself from its table
    /// slots; the host binds the room it has from the buffer's base
    /// ([`RecordsRoom`]) and the word is left to the GPU.
    records_buffer: Vec<u32>,
    /// Per op: it fails on every run (a failing op or an operand that
    /// does), whatever the inputs.
    always_fails: Vec<bool>,
    /// No output word is an op that always fails.
    outputs_can_hold: bool,
    /// The values of the first ops (the constants).
    constants: Vec<u64>,
    /// The ops after the constants: the inputs, each a user-data dword
    /// index or [`SHADER_BASE_INPUT`].
    inputs: Vec<u32>,
}

/// An input op reading the shader base.
const SHADER_BASE_INPUT: u32 = u32::MAX;

/// Scratch of [`ReplayProgram::run_into`], reused across runs.
#[derive(Default)]
pub struct ReplayScratch {
    registers: Vec<u64>,
    failed: Vec<bool>,
    active: Vec<bool>,
    visited: Vec<bool>,
    pending: Vec<usize>,
    /// Reads of NUM_RECORDS words left to the GPU in this run.
    records: Vec<u32>,
}

/// The bytes from a buffer's base (48-bit guest address) that its host
/// binding covers when its NUM_RECORDS (the dword at the third argument,
/// still being written by GPU work) is left to the program, given the V#
/// stride (second argument). None: the host has no such room; the word is
/// read on the CPU.
pub type RecordsRoom<'a> = &'a dyn Fn(u64, u32, u64) -> Option<u64>;

/// `RPS5E_DEFER_RECORDS=check` (set by the host): a NUM_RECORDS word left
/// to the GPU is read on the CPU as well, compared with the room the host
/// would bind ([`RECORDS_STATS`]) and bound as read.
pub static RECORDS_CHECK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// NUM_RECORDS words left to the GPU, and in the check mode: words
/// compared, words whose records reach past the room bound, and the last
/// such (base, records bytes, room).
pub struct RecordsStats {
    pub deferred: std::sync::atomic::AtomicU64,
    pub checked: std::sync::atomic::AtomicU64,
    pub over: std::sync::atomic::AtomicU64,
    pub last_over: std::sync::Mutex<Option<(u64, u64, u64)>>,
    /// The largest (records bytes, room) pair seen by the check, by
    /// records bytes.
    pub largest: std::sync::Mutex<Option<(u64, u64, u64)>>,
    /// Per program address: words checked, words past the room.
    pub programs: std::sync::Mutex<Vec<(u64, u64, u64)>>,
}

pub static RECORDS_STATS: RecordsStats = RecordsStats {
    deferred: std::sync::atomic::AtomicU64::new(0),
    checked: std::sync::atomic::AtomicU64::new(0),
    over: std::sync::atomic::AtomicU64::new(0),
    last_over: std::sync::Mutex::new(None),
    largest: std::sync::Mutex::new(None),
    programs: std::sync::Mutex::new(Vec::new()),
};

/// One run's results.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReplayResult {
    /// Descriptor words then table words (see [`ReplayProgram`]).
    pub words: Vec<u32>,
    /// Per range: the 48-bit base when it evaluated.
    pub range_bases: Vec<Option<u64>>,
    /// Table slots left zero for the GPU to fill: (slot, guest address).
    pub deferred: Vec<(u32, u64)>,
    /// Diagnostics ([`WHY_BLOCKED`]): the read ops of a synchronising run
    /// whose word GPU work is still writing and which could not be
    /// deferred: (op, guest address).
    pub blocked: Vec<(u32, u64)>,
}

/// Diagnostics: synchronising runs note in [`ReplayResult::blocked`] the
/// GPU-written reads they wait for (`RPS5E_DESC_WAIT_WHY`, set by the host).
pub static WHY_BLOCKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Diagnostics: per (program address, op) the blocked reads seen, with
/// the description of the op's uses and the last address.
pub static BLOCKED_READS: std::sync::Mutex<Vec<((u64, u32), String, u64, u64)>> = std::sync::Mutex::new(Vec::new());

/// Count the blocked reads of one synchronising run of `program`.
pub fn note_blocked(program: &ReplayProgram, shader_base: u64, blocked: &[(u32, u64)]) {
    let mut v = BLOCKED_READS.lock().unwrap_or_else(|e| e.into_inner());
    for &(op, address) in blocked {
        if let Some(e) = v.iter_mut().find(|e| e.0 == (shader_base, op)) {
            e.2 += 1;
            e.3 = address;
        } else if v.len() < 256 {
            v.push(((shader_base, op), program.describe_uses(op), 1, address));
        }
    }
}

struct Compiler<'p> {
    plan: &'p ShaderResourcePlan,
    ops: Vec<Op>,
    memo: HashMap<(ValueId, u32, bool), u32>,
    visiting: Vec<(ValueId, u32, bool)>,
    constants: HashMap<u64, u32>,
    fail: Option<u32>,
    depth: usize,
    overflow: bool,
}

const MAX_DEPTH: usize = 4096;
const MAX_OPS: usize = 1 << 16;

impl<'p> Compiler<'p> {
    fn push(&mut self, op: Op) -> u32 {
        self.ops.push(op);
        (self.ops.len() - 1) as u32
    }

    fn constant(&mut self, v: u64) -> u32 {
        if let Some(&r) = self.constants.get(&v) {
            return r;
        }
        let r = self.push(Op::Constant(v));
        self.constants.insert(v, r);
        r
    }

    fn fail(&mut self) -> u32 {
        match self.fail {
            Some(r) => r,
            None => {
                let r = self.push(Op::Fail);
                self.fail = Some(r);
                r
            }
        }
    }

    /// The register of `value` under the active mask `mask` (NONE: none),
    /// read through the clean reader when `clean`.
    fn value(&mut self, value: ValueId, mask: u32, clean: bool) -> u32 {
        let graph = &self.plan.graph;
        let node = graph.node(value);
        if node.is_constant() {
            return self.constant(node.payload);
        }
        if node.kind == ValueKind::Select && mask != NONE && node.operands[0].0 == mask {
            let t = node.operands[1];
            return self.value(t, mask, clean);
        }
        if let Some(&r) = self.memo.get(&(value, mask, clean)) {
            return r;
        }
        if self.visiting.contains(&(value, mask, clean)) {
            return self.fail();
        }
        if self.depth >= MAX_DEPTH || self.ops.len() >= MAX_OPS {
            self.overflow = true;
            return self.fail();
        }
        self.depth += 1;
        self.visiting.push((value, mask, clean));
        let r = self.node(value, mask, clean);
        self.visiting.pop();
        self.depth -= 1;
        self.memo.insert((value, mask, clean), r);
        r
    }

    fn node(&mut self, value: ValueId, mask: u32, clean: bool) -> u32 {
        let plan = self.plan;
        let graph = &plan.graph;
        let node = graph.node(value);
        match node.kind {
            ValueKind::MemoryAperture => self.constant(inline_constants::decode_aperture64(node.payload as u32) >> 32),
            ValueKind::UserData => {
                let register = node.payload as u32;
                let base = graph.user_data_base;
                if register < base {
                    return self.fail();
                }
                self.push(Op::UserData(register - base))
            }
            ValueKind::ShaderBase => self.push(Op::ShaderBase),
            ValueKind::Phi => match graph.resolve_invariant_phi(value) {
                Some(invariant) => self.value(invariant, mask, clean),
                None => self.fail(),
            },
            ValueKind::FirstLane => {
                let (operand, active) = (node.operands[0], node.operands[1]);
                self.value(operand, active.0, clean)
            }
            ValueKind::ResourceTableWord => match plan.table_reads.get(node.payload as usize) {
                Some(read) => self.value(read.value, mask, clean),
                None => self.fail(),
            },
            ValueKind::ScalarAddressWord | ValueKind::ScalarBufferWord => self.read(value, mask, clean),
            ValueKind::Select => {
                let operands = [node.operands[0], node.operands[1], node.operands[2]];
                let c = self.value(operands[0], mask, clean);
                let t = self.value(operands[1], mask, clean);
                let f = self.value(operands[2], mask, clean);
                self.push(Op::Select(c, t, f))
            }
            ValueKind::Operation => {
                if !node.op.is_uniform() {
                    return self.fail();
                }
                let op = node.op;
                let list: Vec<ValueId> = node.operands.iter().copied().take(4).collect();
                let mut operands = [0u32; 4];
                for (i, &o) in list.iter().enumerate() {
                    operands[i] = self.value(o, mask, clean);
                }
                self.push(Op::Operation { op, operands, count: list.len() as u8 })
            }
            _ => self.fail(),
        }
    }

    fn read(&mut self, value: ValueId, mask: u32, clean: bool) -> u32 {
        let graph = &self.plan.graph;
        let node = graph.node(value);
        let kind = node.kind;
        let index = node.memory_index();
        if index >= graph.memory.len() {
            return self.fail();
        }
        let immediate = graph.memory.get(index).offset as i32 as i64;
        let (handle, offset_value) = (node.operands[0], node.operands[1]);
        let handle_operands: Vec<ValueId> = graph.node(handle).operands.to_vec();
        if handle_operands.len() < 2 {
            return self.fail();
        }
        let low = self.value(handle_operands[0], mask, clean);
        let high = self.value(handle_operands[1], mask, clean);
        let offset = self.value(offset_value, mask, clean);
        let (mut records, mut format) = (NONE, NONE);
        if kind == ValueKind::ScalarBufferWord {
            if handle_operands.len() != 4 {
                return self.fail();
            }
            records = self.value(handle_operands[2], mask, clean);
            format = self.value(handle_operands[3], mask, clean);
        }
        self.push(Op::Read { kind, low, high, offset, records, format, immediate, clean, block: 1, deferrable: false })
    }
}

/// Group the reads of `ops` into blocks: reads through the same operand
/// registers (and the same reader) whose immediates are consecutive dwords.
/// Returns the new op order (old indices): each block's reads, in
/// immediate order, take the place of its earliest read, with `block` set
/// on the first. Every read of a block shares the operands of the first,
/// which precede it, so the order still follows the dependencies.
fn form_blocks(ops: &mut [Op]) -> Vec<u32> {
    type Key = (ValueKind, u32, u32, u32, u32, u32, bool);
    let mut groups: HashMap<Key, Vec<(i64, u32)>> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        if let Op::Read { kind, low, high, offset, records, format, immediate, clean, .. } = *op {
            groups.entry((kind, low, high, offset, records, format, clean)).or_default().push((immediate, i as u32));
        }
    }
    // Per op: the block whose earliest read it is.
    let mut leads: Vec<Option<usize>> = vec![None; ops.len()];
    let mut member = vec![false; ops.len()];
    let mut blocks: Vec<Vec<u32>> = Vec::new();
    for (_, mut reads) in groups {
        reads.sort_unstable();
        let mut start = 0;
        while start < reads.len() {
            let mut end = start + 1;
            while end < reads.len() && end - start < MAX_BLOCK && reads[end].0 == reads[end - 1].0 + 4 {
                end += 1;
            }
            if end - start > 1 {
                let run: Vec<u32> = reads[start..end].iter().map(|r| r.1).collect();
                let first = *run.iter().min().unwrap();
                run.iter().for_each(|&r| member[r as usize] = true);
                leads[first as usize] = Some(blocks.len());
                blocks.push(run);
            }
            start = end;
        }
    }
    let mut order = Vec::with_capacity(ops.len());
    for i in 0..ops.len() {
        match leads[i] {
            Some(b) => {
                let run = &blocks[b];
                for (k, &r) in run.iter().enumerate() {
                    if let Op::Read { block, .. } = &mut ops[r as usize] {
                        *block = if k == 0 { run.len() as u8 } else { 0 };
                    }
                }
                order.extend_from_slice(run);
            }
            None if member[i] => {}
            None => order.push(i as u32),
        }
    }
    order
}

impl ReplayProgram {
    /// Whether `plan` can be compiled: its materialisation reads nothing
    /// but the descriptor sources, the flattened table, the device-address
    /// bases and the branch predicates.
    pub fn eligible(plan: &ShaderResourcePlan) -> bool {
        plan.buffer_candidate_tables.is_empty()
            && !plan.requires_specialization_memory
            && plan.indirect_images.is_empty()
            && plan.clean_flat_slots.iter().all(|&c| c == 0)
            && plan.info.images.iter().all(|i| plan.descriptor_sources.get(i.source as usize).is_some_and(|s| s.indirect_image.is_none()))
            && plan.materialization_sources.len() == plan.info.buffers.len() + plan.info.images.len() + plan.info.samplers.len()
            && plan.table_reads.iter().all(|r| (r.flat_offset as usize) < plan.table_reads.len())
    }

    /// Compile `plan` (None when it is not eligible or too large).
    pub fn compile(plan: &ShaderResourcePlan) -> Option<ReplayProgram> {
        if !Self::eligible(plan) {
            return None;
        }
        let mut c = Compiler {
            plan,
            ops: Vec::new(),
            memo: HashMap::new(),
            visiting: Vec::new(),
            constants: HashMap::new(),
            fail: None,
            depth: 0,
            overflow: false,
        };
        let mut outputs = Vec::new();
        let mut output_sources = Vec::new();
        let mut descriptor_spans = Vec::with_capacity(plan.materialization_sources.len());
        for &source in &plan.materialization_sources {
            let s = plan.descriptor_sources.get(source as usize)?;
            let start = outputs.len() as u32;
            for &d in &s.dwords {
                outputs.push(c.value(d, NONE, false));
                output_sources.push(source);
            }
            descriptor_spans.push((start, outputs.len() as u32));
        }
        let descriptor_words = outputs.len();
        let table_len = plan.table_reads.len();
        let zero = c.constant(0);
        let mut table = vec![zero; table_len];
        for read in &plan.table_reads {
            table[read.flat_offset as usize] = c.value(read.value, NONE, false);
        }
        outputs.extend(table);
        let mut ranges = Vec::with_capacity(plan.device_address_ranges.len());
        for range in &plan.device_address_ranges {
            ranges.push(range.plannable.then(|| (c.value(range.base_low, NONE, false), c.value(range.base_high, NONE, false))));
        }
        let mut branches = Vec::with_capacity(plan.resource_branches.len());
        for block in &plan.resource_branches {
            if block.successors.iter().any(|&b| b >= plan.resource_branches.len()) || block.sources.iter().any(|&s| s as usize >= plan.descriptor_sources.len()) {
                return None;
            }
            let condition = block.condition.map(|v| c.value(v, NONE, true));
            branches.push((condition, block.successors.clone(), block.sources.clone()));
        }
        if c.overflow {
            return None;
        }
        // Plain table reads nothing else consumes: their words may be left
        // for the GPU to fill.
        let mut used = vec![false; c.ops.len()];
        for op in &c.ops {
            match *op {
                Op::Select(a, b, d) => [a, b, d].iter().for_each(|&r| used[r as usize] = true),
                Op::Operation { operands, count, .. } => operands[..count as usize].iter().for_each(|&r| used[r as usize] = true),
                Op::Read { low, high, offset, records, format, .. } => {
                    [low, high, offset, records, format].iter().filter(|&&r| r != NONE).for_each(|&r| used[r as usize] = true)
                }
                _ => {}
            }
        }
        outputs[..descriptor_words].iter().for_each(|&r| used[r as usize] = true);
        for &(low, high) in ranges.iter().flatten() {
            used[low as usize] = true;
            used[high as usize] = true;
        }
        for (condition, _, _) in &branches {
            if let Some(r) = condition {
                used[*r as usize] = true;
            }
        }
        let mut deferrable = vec![Vec::new(); c.ops.len()];
        for (slot, &r) in outputs[descriptor_words..].iter().enumerate() {
            if !used[r as usize] && matches!(c.ops[r as usize], Op::Read { clean: false, .. }) {
                deferrable[r as usize].push(slot as u32);
            }
        }
        // Reads whose only use besides table slots is one plan buffer's
        // NUM_RECORDS: uses other than descriptor words, then the
        // descriptor words.
        let mut records_buffer = vec![NONE; c.ops.len()];
        {
            let mut other = vec![false; c.ops.len()];
            for op in &c.ops {
                match *op {
                    Op::Select(a, b, d) => [a, b, d].iter().for_each(|&r| other[r as usize] = true),
                    Op::Operation { operands, count, .. } => operands[..count as usize].iter().for_each(|&r| other[r as usize] = true),
                    Op::Read { low, high, offset, records, format, .. } => {
                        [low, high, offset, records, format].iter().filter(|&&r| r != NONE).for_each(|&r| other[r as usize] = true)
                    }
                    _ => {}
                }
            }
            for &(low, high) in ranges.iter().flatten() {
                other[low as usize] = true;
                other[high as usize] = true;
            }
            for (condition, _, _) in &branches {
                if let Some(r) = condition {
                    other[*r as usize] = true;
                }
            }
            let mut uses = vec![0u32; c.ops.len()];
            let mut word2 = vec![NONE; c.ops.len()];
            let buffers = plan.info.buffers.len().min(descriptor_spans.len());
            for (k, &(start, end)) in descriptor_spans.iter().enumerate() {
                for w in start..end {
                    let r = outputs[w as usize] as usize;
                    uses[r] += 1;
                    if k < buffers && end - start == 4 && w == start + 2 {
                        word2[r] = k as u32;
                    }
                }
            }
            let mut slots = vec![Vec::new(); c.ops.len()];
            for (slot, &r) in outputs[descriptor_words..].iter().enumerate() {
                slots[r as usize].push(slot as u32);
            }
            for r in 0..c.ops.len() {
                if word2[r] != NONE && uses[r] == 1 && !other[r] && !slots[r].is_empty() && matches!(c.ops[r], Op::Read { clean: false, .. }) {
                    records_buffer[r] = word2[r];
                    deferrable[r] = std::mem::take(&mut slots[r]);
                }
            }
        }
        let mut ops = c.ops;
        for ((op, slots), &records) in ops.iter_mut().zip(&deferrable).zip(&records_buffer) {
            if let Op::Read { deferrable, .. } = op {
                *deferrable = !slots.is_empty() && records == NONE;
            }
        }
        // Reads of consecutive dwords through the same operands are
        // placed together and read as one range; constants, then the
        // inputs (user data, shader base) come first: they have no
        // operands, and a run fills them without the general loop.
        let blocked = form_blocks(&mut ops);
        let class = |op: &Op| match op {
            Op::Constant(_) => 0,
            Op::UserData(_) | Op::ShaderBase => 1,
            _ => 2,
        };
        let mut order = Vec::with_capacity(blocked.len());
        for c in 0..3 {
            order.extend(blocked.iter().copied().filter(|&o| class(&ops[o as usize]) == c));
        }
        let constant_count = blocked.iter().filter(|&&o| class(&ops[o as usize]) == 0).count();
        let input_count = blocked.iter().filter(|&&o| class(&ops[o as usize]) == 1).count();
        let mut renamed = vec![0u32; order.len()];
        for (new, &old) in order.iter().enumerate() {
            renamed[old as usize] = new as u32;
        }
        let rename = |r: u32| if r == NONE { NONE } else { renamed[r as usize] };
        let ops: Vec<Op> = order
            .iter()
            .map(|&old| match ops[old as usize] {
                Op::Select(a, b, d) => Op::Select(rename(a), rename(b), rename(d)),
                Op::Operation { op, operands, count } => {
                    let mut o = operands;
                    o[..count as usize].iter_mut().for_each(|r| *r = rename(*r));
                    Op::Operation { op, operands: o, count }
                }
                Op::Read { kind, low, high, offset, records, format, immediate, clean, block, deferrable } => {
                    Op::Read { kind, low: rename(low), high: rename(high), offset: rename(offset), records: rename(records), format: rename(format), immediate, clean, block, deferrable }
                }
                other => other,
            })
            .collect();
        let deferrable: Vec<Vec<u32>> = order.iter().map(|&old| std::mem::take(&mut deferrable[old as usize])).collect();
        let records_buffer: Vec<u32> = order.iter().map(|&old| records_buffer[old as usize]).collect();
        outputs.iter_mut().for_each(|r| *r = rename(*r));
        let ranges = ranges.into_iter().map(|r| r.map(|(low, high)| (rename(low), rename(high)))).collect();
        let branches = branches.into_iter().map(|(condition, successors, sources)| (condition.map(rename), successors, sources)).collect();
        let mut always_fails = vec![false; ops.len()];
        for (i, op) in ops.iter().enumerate() {
            let fails = |r: u32| r != NONE && always_fails[r as usize];
            always_fails[i] = match *op {
                Op::Fail => true,
                Op::Select(a, b, d) => fails(a) || fails(b) || fails(d),
                Op::Operation { operands, count, .. } => operands[..count as usize].iter().any(|&r| fails(r)),
                Op::Read { low, high, offset, records, format, .. } => [low, high, offset, records, format].into_iter().any(fails),
                _ => false,
            };
        }
        let outputs_can_hold = outputs.iter().all(|&r| !always_fails[r as usize]);
        let constants = ops[..constant_count].iter().map(|op| if let Op::Constant(v) = *op { v } else { unreachable!() }).collect();
        let inputs = ops[constant_count..constant_count + input_count]
            .iter()
            .map(|op| match *op {
                Op::UserData(index) => index,
                _ => SHADER_BASE_INPUT,
            })
            .collect();
        Some(ReplayProgram {
            constants,
            inputs,
            always_fails,
            outputs_can_hold,
            ops,
            outputs,
            output_sources,
            branches,
            source_count: plan.descriptor_sources.len(),
            descriptor_spans,
            descriptor_words,
            table_len,
            ranges,
            user_data_len: plan.graph.user_data_count as usize,
            deferrable,
            records_buffer,
        })
    }

    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    /// Each descriptor's words in a result (start, end), in materialisation
    /// order: buffers, images, samplers.
    /// The NUM_RECORDS register of read `i` left to the GPU: the records
    /// the host's room from the buffer's base holds (the check mode reads
    /// the word as well and compares), or the word read through `reader`
    /// when the host has no room for it.
    #[allow(clippy::too_many_arguments)]
    fn records_value(&self, i: usize, shader_base: u64, registers: &[u64], failed: &[bool], reader: &ReplayReader<'_>, room: Option<RecordsRoom<'_>>) -> Result<Option<u64>, ReplayAbort> {
        use std::sync::atomic::Ordering::Relaxed;
        let read = || -> Result<Option<u64>, ReplayAbort> {
            let Op::Read { kind, low, high, offset, records, immediate, clean, .. } = self.ops[i] else { return Ok(None) };
            let records = if records != NONE { registers[records as usize] } else { 0 };
            read_raw(kind, registers[low as usize], registers[high as usize], registers[offset as usize] as u32, records, immediate, reader, clean)
        };
        let (start, _) = self.descriptor_spans[self.records_buffer[i] as usize];
        let (lo, hi) = (self.outputs[start as usize] as usize, self.outputs[start as usize + 1] as usize);
        if failed[lo] || failed[hi] {
            return read();
        }
        let (w0, w1) = (registers[lo] as u32, registers[hi] as u32);
        let base = w0 as u64 | ((w1 & 0xFFFF) as u64) << 32;
        let stride = (w1 >> 16) & 0x3FFF;
        let word = match self.ops[i] {
            Op::Read { kind, low, high, offset, records, immediate, .. } => {
                let records = if records != NONE { registers[records as usize] } else { 0 };
                raw_address(kind, registers[low as usize], registers[high as usize], registers[offset as usize] as u32, records, immediate)
            }
            _ => None,
        };
        let Some(word) = word else { return read() };
        let Some(bytes) = room.and_then(|f| f(base, stride, word)) else { return read() };
        let records = if stride == 0 { bytes } else { bytes / stride as u64 }.min(u32::MAX as u64);
        RECORDS_STATS.deferred.fetch_add(1, Relaxed);
        if RECORDS_CHECK.load(Relaxed) {
            if let Some(actual) = read()? {
                let need = if stride == 0 { actual as u32 as u64 } else { actual as u32 as u64 * stride as u64 };
                let room_bytes = if stride == 0 { records } else { records * stride as u64 };
                RECORDS_STATS.checked.fetch_add(1, Relaxed);
                {
                    let mut p = RECORDS_STATS.programs.lock().unwrap_or_else(|e| e.into_inner());
                    let over = (need > room_bytes) as u64;
                    if let Some(e) = p.iter_mut().find(|e| e.0 == shader_base) {
                        e.1 += 1;
                        e.2 += over;
                    } else if p.len() < 64 {
                        p.push((shader_base, 1, over));
                    }
                }
                if need > room_bytes {
                    RECORDS_STATS.over.fetch_add(1, Relaxed);
                    *RECORDS_STATS.last_over.lock().unwrap_or_else(|e| e.into_inner()) = Some((base, need, room_bytes));
                }
                let mut largest = RECORDS_STATS.largest.lock().unwrap_or_else(|e| e.into_inner());
                if largest.is_none_or(|l| need > l.1) {
                    *largest = Some((base, need, room_bytes));
                }
                // The check binds what the program will read.
                return Ok(Some(actual));
            }
        }
        Ok(Some(records))
    }

    /// Diagnostics: what uses the value of op `op` (descriptor words as
    /// `source.word`, table slots, other ops, range bases, branch
    /// predicates).
    pub fn describe_uses(&self, op: u32) -> String {
        let mut words = Vec::new();
        for (i, &r) in self.outputs[..self.descriptor_words].iter().enumerate() {
            if r == op {
                let (k, &(start, _)) = self.descriptor_spans.iter().enumerate().find(|(_, s)| s.0 as usize <= i && (i as u32) < s.1).unwrap_or((usize::MAX, &(0, 0)));
                words.push(format!("{k}.{}", i as u32 - start));
            }
        }
        let slots = self.outputs[self.descriptor_words..].iter().filter(|&&r| r == op).count();
        let mut ops = 0;
        for o in &self.ops {
            let refs: &[u32] = match o {
                Op::Select(a, b, d) => &[*a, *b, *d],
                Op::Operation { operands, count, .. } => &operands[..*count as usize],
                Op::Read { low, high, offset, records, format, .. } => &[*low, *high, *offset, *records, *format],
                _ => &[],
            };
            ops += refs.iter().filter(|&&r| r == op).count();
        }
        let ranges = self.ranges.iter().flatten().filter(|r| r.0 == op || r.1 == op).count();
        let branches = self.branches.iter().filter(|b| b.0 == Some(op)).count();
        format!("descriptor words [{}] (buffers first) table slots {slots} ops {ops} ranges {ranges} branches {branches}", words.join(" "))
    }

    pub fn descriptor_spans(&self) -> &[(u32, u32)] {
        &self.descriptor_spans
    }

    /// Run the program for one draw.
    pub fn run(
        &self,
        user_data: &[u32],
        shader_base: u64,
        reader: &ReplayReader<'_>,
        gpu_written: Option<GpuWrittenProbe<'_>>,
        scratch: &mut ReplayScratch,
    ) -> Result<ReplayResult, ReplayAbort> {
        let mut out = ReplayResult::default();
        self.run_into(user_data, shader_base, reader, gpu_written, scratch, &mut out)?;
        Ok(out)
    }

    /// [`Self::run`] into `out` (its vectors are reused).
    pub fn run_into(
        &self,
        user_data: &[u32],
        shader_base: u64,
        reader: &ReplayReader<'_>,
        gpu_written: Option<GpuWrittenProbe<'_>>,
        scratch: &mut ReplayScratch,
        out: &mut ReplayResult,
    ) -> Result<(), ReplayAbort> {
        self.run_into_with(user_data, shader_base, reader, gpu_written, None, scratch, out)
    }

    /// [`Self::run_into`]; with `records_room`, a NUM_RECORDS word GPU
    /// work is still writing ([`Self::records_buffer`]) is left to the GPU:
    /// its table slots are deferred and the descriptor gets the records the
    /// room holds.
    #[allow(clippy::too_many_arguments)]
    pub fn run_into_with(
        &self,
        user_data: &[u32],
        shader_base: u64,
        reader: &ReplayReader<'_>,
        gpu_written: Option<GpuWrittenProbe<'_>>,
        records_room: Option<RecordsRoom<'_>>,
        scratch: &mut ReplayScratch,
        out: &mut ReplayResult,
    ) -> Result<(), ReplayAbort> {
        out.deferred.clear();
        out.blocked.clear();
        let why = matches!(reader, ReplayReader::Words(..)) && WHY_BLOCKED.load(std::sync::atomic::Ordering::Relaxed);
        let n = self.ops.len();
        // The register file keeps its values across runs: a register is
        // read only through an operand whose `failed` bit is clear, and a
        // clear bit means the op producing it wrote it in this run
        // (`active_sources` guards its reads the same way). Only growing
        // the file avoids zeroing it per draw.
        let registers = &mut scratch.registers;
        if registers.len() < n {
            registers.resize(n, 0);
        }
        let failed = &mut scratch.failed;
        failed.clear();
        failed.resize(n, false);
        let pending_records = &mut scratch.records;
        pending_records.clear();
        // No register failed but those that fail on every run (which no
        // output word is, when `outputs_can_hold`): the outputs need no
        // failure checks.
        let mut any_failed = !self.outputs_can_hold;
        let (constants, inputs) = (self.constants.len(), self.inputs.len());
        registers[..constants].copy_from_slice(&self.constants);
        for (k, &index) in self.inputs.iter().enumerate() {
            let i = constants + k;
            if index == SHADER_BASE_INPUT {
                registers[i] = shader_base;
            } else {
                match user_data.get(index as usize) {
                    Some(&w) => registers[i] = w as u64,
                    None => {
                        failed[i] = true;
                        any_failed = true;
                    }
                }
            }
        }
        let mut i = constants + inputs;
        while i < n {
            let value = match self.ops[i] {
                Op::Constant(v) => Some(v),
                Op::Fail => None,
                Op::UserData(index) => user_data.get(index as usize).map(|&w| w as u64),
                Op::ShaderBase => Some(shader_base),
                Op::Select(c, t, f) => {
                    if failed[c as usize] || failed[t as usize] || failed[f as usize] {
                        None
                    } else if registers[c as usize] != 0 {
                        Some(registers[t as usize])
                    } else {
                        Some(registers[f as usize])
                    }
                }
                Op::Operation { op, operands, count } => {
                    let n = count as usize;
                    if operands[..n].iter().any(|&o| failed[o as usize]) {
                        None
                    } else {
                        let mut values = [0u64; 4];
                        for k in 0..n {
                            values[k] = registers[operands[k] as usize];
                        }
                        evaluate_operation(op, &values[..n])
                    }
                }
                Op::Read { kind, low, high, offset, records, format, immediate, clean, block, deferrable } => {
                    let to_records = records_room.is_some() && self.records_buffer[i] != NONE;
                    let deferrable = deferrable || to_records;
                    if block > 1 && self.read_block(i, block as usize, reader, gpu_written, records_room.is_some(), registers, failed) {
                        i += block as usize;
                        continue;
                    }
                    let mut inputs = [low, high, offset, records, format].into_iter().filter(|&r| r != NONE);
                    if inputs.any(|r| failed[r as usize]) {
                        None
                    } else {
                        let (low, high, offset) = (registers[low as usize], registers[high as usize], registers[offset as usize] as u32);
                        let records = if records != NONE { registers[records as usize] } else { 0 };
                        match gpu_written.filter(|_| deferrable).and_then(|probe| {
                            let address = raw_address(kind, low, high, offset, records, immediate)?;
                            probe(address).then_some(address)
                        }) {
                            Some(address) => {
                                out.deferred.extend(self.deferrable[i].iter().map(|&slot| (slot, address)));
                                if to_records {
                                    pending_records.push(i as u32);
                                }
                                Some(0)
                            }
                            None => {
                                if why && !deferrable {
                                    if let (Some(probe), Some(address)) = (gpu_written, raw_address(kind, low, high, offset, records, immediate)) {
                                        if probe(address) {
                                            out.blocked.push((i as u32, address));
                                        }
                                    }
                                }
                                read_raw(kind, low, high, offset, records, immediate, reader, clean)?
                            }
                        }
                    }
                }
            };
            match value {
                Some(v) => registers[i] = v,
                None => {
                    failed[i] = true;
                    any_failed |= !self.always_fails[i];
                }
            }
            i += 1;
        }
        for k in 0..pending_records.len() {
            let i = pending_records[k] as usize;
            match self.records_value(i, shader_base, registers, failed, reader, records_room) {
                Ok(Some(v)) => registers[i] = v,
                Ok(None) => {
                    failed[i] = true;
                    any_failed = true;
                }
                Err(e) => return Err(e),
            }
        }
        let ReplayScratch { registers, failed, active, visited, pending, .. } = scratch;
        let any_inactive = self.active_sources(reader, registers, failed, active, visited, pending);
        let words = &mut out.words;
        words.clear();
        if !any_failed && !any_inactive {
            // Every register holds its value: the words are a gather.
            words.extend(self.outputs.iter().map(|&r| registers[r as usize] as u32));
            out.range_bases.clear();
            out.range_bases.extend(self.ranges.iter().map(|r| {
                let (low, high) = (*r)?;
                if failed[low as usize] || failed[high as usize] {
                    return None;
                }
                Some(((registers[low as usize] as u32 as u64) | ((registers[high as usize] as u32 as u64) << 32)) & ADDRESS_MASK)
            }));
            out.deferred.sort_unstable();
            return Ok(());
        }
        words.reserve(self.outputs.len());
        for (i, &r) in self.outputs.iter().enumerate() {
            if any_inactive && i < self.descriptor_words && !active[self.output_sources[i] as usize] {
                words.push(0);
                continue;
            }
            if failed[r as usize] {
                return Err(ReplayAbort::Failed);
            }
            words.push(registers[r as usize] as u32);
        }
        out.range_bases.clear();
        out.range_bases.extend(self.ranges.iter().map(|r| {
            let (low, high) = (*r)?;
            if failed[low as usize] || failed[high as usize] {
                return None;
            }
            Some(((registers[low as usize] as u32 as u64) | ((registers[high as usize] as u32 as u64) << 32)) & ADDRESS_MASK)
        }));
        out.deferred.sort_unstable();
        Ok(())
    }

    /// The `count` reads of the block at op `first` as one read of the
    /// resident reader. False leaves every register of the block to be
    /// produced read by read (exactly as without blocks): some operand
    /// failed, a dword is not a plain read (out of bounds, a failed
    /// address), the dwords are not consecutive, a deferrable dword is
    /// still being written by the GPU, the reader is the word reader, or
    /// the range read refused (any of its words: unmapped, GPU-owned).
    #[allow(clippy::too_many_arguments)]
    fn read_block(&self, first: usize, count: usize, reader: &ReplayReader<'_>, gpu_written: Option<GpuWrittenProbe<'_>>, to_records: bool, registers: &mut [u64], failed: &[bool]) -> bool {
        let ReplayReader::Resident(read, has_clean) = *reader else { return false };
        let Op::Read { kind, low, high, offset, records, format, clean, .. } = self.ops[first] else { return false };
        if clean && !has_clean {
            return false;
        }
        if [low, high, offset, records, format].into_iter().any(|r| r != NONE && failed[r as usize]) {
            return false;
        }
        let (low, high, offset) = (registers[low as usize], registers[high as usize], registers[offset as usize] as u32);
        let records = if records != NONE { registers[records as usize] } else { 0 };
        let base = ((high << 32) | (low as u32 as u64)) & ADDRESS_MASK;
        let mut start = 0;
        for k in 0..count {
            let Op::Read { immediate, deferrable, .. } = self.ops[first + k] else { return false };
            let deferrable = deferrable || (to_records && self.records_buffer[first + k] != NONE);
            let RawAddress::Read(a) = resolve_raw_address(kind, base, high, records, immediate, offset) else { return false };
            if k == 0 {
                start = a;
            } else if Some(a) != start.checked_add(4 * k as u64) {
                return false;
            }
            // As `run_into` probes a deferrable read (through a non-null
            // base): a word the GPU still writes is deferred there.
            if deferrable && base & !3 != 0 && gpu_written.is_some_and(|probe| probe(a)) {
                return false;
            }
        }
        let mut bytes = [0u8; 4 * MAX_BLOCK];
        let bytes = &mut bytes[..4 * count];
        if !read(start, bytes, clean) {
            return false;
        }
        for (k, word) in bytes.chunks_exact(4).enumerate() {
            registers[first + k] = u32::from_le_bytes(word.try_into().unwrap()) as u64;
        }
        true
    }
}

impl ReplayProgram {
    /// Mark in `active` the descriptor sources some path through the
    /// decided branches reaches; false (`active` untouched): no branches,
    /// every source is active.
    fn active_sources(&self, reader: &ReplayReader<'_>, registers: &[u64], failed: &[bool], active: &mut Vec<bool>, visited: &mut Vec<bool>, pending: &mut Vec<usize>) -> bool {
        if self.branches.is_empty() {
            return false;
        }
        let has_clean = match reader {
            ReplayReader::Words(_, clean) => clean.is_some(),
            ReplayReader::Resident(_, has_clean) => *has_clean,
        };
        active.clear();
        active.resize(self.source_count, true);
        for (_, _, sources) in &self.branches {
            for &s in sources {
                active[s as usize] = false;
            }
        }
        visited.clear();
        visited.resize(self.branches.len(), false);
        pending.clear();
        pending.push(0);
        while let Some(index) = pending.pop() {
            if visited[index] {
                continue;
            }
            visited[index] = true;
            let (condition, successors, sources) = &self.branches[index];
            for &s in sources {
                active[s as usize] = true;
            }
            let decided = match condition {
                Some(r) if has_clean && !failed[*r as usize] => Some(registers[*r as usize] as u32),
                _ => None,
            };
            match decided {
                Some(v) => pending.push(successors[if v != 0 { 0 } else { 1 }]),
                None => pending.extend(successors.iter().copied()),
            }
        }
        true
    }
}

/// The address a raw read through a non-null base reads.
fn raw_address(kind: ValueKind, low: u64, high: u64, offset: u32, records: u64, immediate: i64) -> Option<u64> {
    let base = ((high << 32) | (low as u32 as u64)) & ADDRESS_MASK;
    match resolve_raw_address(kind, base, high, records, immediate, offset) {
        RawAddress::Read(a) if base & !3 != 0 => Some(a),
        _ => None,
    }
}

/// [`RuntimeValueEvaluator::read_raw_word`](super::RuntimeValueEvaluator)
/// over a replay reader: `Ok(None)` is a failed value, `Err` aborts.
#[allow(clippy::too_many_arguments)]
fn read_raw(kind: ValueKind, low: u64, high: u64, offset: u32, records: u64, immediate: i64, reader: &ReplayReader<'_>, clean: bool) -> Result<Option<u64>, ReplayAbort> {
    let base = ((high << 32) | (low as u32 as u64)) & ADDRESS_MASK;
    let address = match resolve_raw_address(kind, base, high, records, immediate, offset) {
        RawAddress::Failed => return Ok(None),
        RawAddress::Zero => return Ok(Some(0)),
        RawAddress::Read(a) => a,
    };
    let word = match reader {
        ReplayReader::Words(read, _) if !clean => read(address),
        ReplayReader::Words(_, read_clean) => read_clean.and_then(|read| read(address)),
        ReplayReader::Resident(read, has_clean) => {
            let mut b = [0u8; 4];
            (!clean || *has_clean).then(|| read(address, &mut b, clean).then(|| u32::from_le_bytes(b))).flatten()
        }
    };
    match word {
        Some(w) => Ok(Some(w as u64)),
        // A load through a null base never executes on hardware.
        None if base & !3 == 0 => Ok(Some(0)),
        None => match reader {
            ReplayReader::Resident(..) if !clean => Err(ReplayAbort::Unreadable),
            _ => Ok(None),
        },
    }
}
