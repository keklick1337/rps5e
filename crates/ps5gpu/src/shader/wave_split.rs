// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Wave64 compute programs on a 32-wide host subgroup with one guest lane
//! per host invocation ("split" waves).
//!
//! A guest wave is 64 consecutive invocations of the work-group, so two
//! host subgroups: the low half holds guest lanes 0..31, the high half
//! 32..63. Per-lane work runs as it is; what crosses lanes is rebuilt:
//! * a ballot: each half takes its subgroup's ballot word, and the two
//!   words are exchanged through workgroup memory;
//! * a lane read (`v_readlane`, `v_readfirstlane`): the owning half's
//!   subgroup shuffle, exchanged the same way;
//! * a shuffle whose source lane is in the reading lane's own half for
//!   every lane (DPP rows and quads, permlane, ds_swizzle and ds_bpermute
//!   within 32 lanes): the host subgroup shuffle; any other shuffle: every
//!   lane stores its value and loads its source lane's;
//! * the subgroup LDS barrier: a workgroup barrier;
//! * a lane-indexed register (`v_writelane` of a uniform value at a uniform
//!   lane, `v_readlane` at a uniform lane: a lane stack, SGPR spills) is
//!   kept as lane slots ([`lane_slots`]): every invocation holds the words
//!   of guest lanes j and j + 32 (j: its lane in the half) in both halves,
//!   so its lane reads are shuffles within the half, with no exchange;
//! * a lane read at the first lane of a ballot (`s_ff1` of the mask, the
//!   hit selection of a traversal) rides on the ballot's exchange: each
//!   half sends its candidate (the value at its first set bit) with its
//!   ballot word ([`first_lane_reads`]);
//! * a ballot whose predicate is known where an earlier ballot dominating
//!   it is made rides on that ballot's exchange ([`ballot_riders`]): the
//!   four hit tests of a BVH node take one exchange instead of four.
//!
//! An exchange is a store, a workgroup barrier and a load in a slot of the
//! wave's region of an exchange array. Consecutive exchanges alternate
//! between two slots (the parity lives in a private variable), so a half
//! that runs ahead to the next exchange never overwrites words the other
//! half has not loaded yet.
//!
//! A workgroup barrier needs every invocation of the work-group. A wave's
//! control flow is uniform, so with one wave per work-group every exchange
//! qualifies. With several waves every exchange must lie outside control
//! flow whose condition may differ between the waves ([`plan`] checks it).
//! Programs that fail the checks, or use what is not rebuilt here (wave-wide
//! appends, clocks, 64-bit LDS atomics through the explicit layout), run
//! paired instead (two guest lanes per invocation).
//!
//! Which is faster depends on the program. A paired invocation runs the
//! same operation for its two lanes back to back (the host compiler can
//! pack the pair, e.g. as FP32x2 on the RTX 50 series, which is how we
//! read the measurement below) and keeps two independent chains in
//! flight; it needs twice the registers.
//! Measured on an RTX 5090 (`wave64_split_benchmark`): light ALU kernels
//! run 0-8 % slower split, register-heavy ones up to 3.6x faster (the
//! paired invocation spills). GTA V's per-object compute programs with 22
//! and 23 VGPRs (memory-bound) take 25-33 % less GPU time split. So
//! programs with at least [`min_vgprs`] guest VGPRs run split.
//!
//! `RPS5E_WAVE64_SPLIT=0` keeps every wave64 compute program paired;
//! `RPS5E_WAVE64_SPLIT_VGPRS=<n>` sets the VGPR threshold (0: every
//! program that qualifies).

use super::emit::{custom, Emitter, R, SCOPE_SUBGROUP, SCOPE_WORKGROUP};
use super::Features;
use ps5shader::ssa::*;
use rspirv::dr::{Instruction, Operand};
use rspirv::spirv::{BuiltIn, Capability, GlslStd450Op as GLOp, Op as S, StorageClass as SC, Word};
use std::collections::{HashMap, HashSet};

/// `RPS5E_WAVE64_LANE_SLOTS=0`: lane-indexed registers of split programs
/// are not kept as lane slots (their lane reads are exchanges). Part of
/// the disk-cache key.
pub fn lane_slots_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_WAVE64_LANE_SLOTS").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")))
}

/// `RPS5E_WAVE64_BALLOT_RIDERS=0`: every ballot of a split program makes
/// its own exchange ([`ballot_riders`]). Part of the disk-cache key.
pub fn ballot_riders_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_WAVE64_BALLOT_RIDERS").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")))
}

/// `RPS5E_WAVE64_SPLIT=0`: wave64 compute programs on a 32-wide host
/// subgroup run paired (two guest lanes per invocation). Part of the
/// disk-cache key.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_WAVE64_SPLIT").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")))
}

/// Guest VGPRs from which a program runs split (`RPS5E_WAVE64_SPLIT_VGPRS`,
/// default 16). Part of the disk-cache key.
pub fn min_vgprs() -> u32 {
    static MIN: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| std::env::var("RPS5E_WAVE64_SPLIT_VGPRS").ok().and_then(|v| v.parse().ok()).unwrap_or(16))
}

/// The VGPRs a guest program names (the highest register plus one; a
/// register tuple counts by its first register).
pub fn guest_vgprs(program: &ps5shader::Program) -> u32 {
    program
        .instructions
        .iter()
        .flat_map(|i| i.sources.iter().chain(&i.destinations))
        .filter(|o| o.kind == ps5shader::OperandKind::VectorRegister)
        .map(|o| (o.value & 0xFF) + 1)
        .max()
        .unwrap_or(0)
}

/// Lanes of a guest wave.
const WAVE: u32 = 64;

/// LDS dwords a program that does not size its LDS gets (`memory.rs`).
const DEFAULT_LDS_DWORDS: u32 = 16384;

/// Workgroup memory assumed when the device limit is unknown.
const DEFAULT_SHARED_BYTES: u32 = 48 << 10;

/// What a split program needs.
#[derive(Clone, Debug, Default)]
pub struct SplitPlan {
    /// Waves of a work-group.
    pub waves: u32,
    /// Shuffles whose source lane is in the reading lane's half.
    pub in_half: HashSet<Value>,
    /// Dwords of one exchange slot.
    pub slot_dwords: u32,
    /// Exchanges per kind (statistics): ballots, lane reads, shuffles
    /// through memory, shuffles in the half, LDS barriers, lane reads of
    /// lane slots (no exchange), lane reads riding on a ballot's exchange,
    /// ballots riding on an earlier ballot's exchange.
    pub counts: [u32; 8],
    /// Values kept as lane slots ([`lane_slots`]): a write's (lane, value,
    /// old register), None for a phi.
    pub slots: HashMap<Value, Option<(Value, Value, Value)>>,
    /// Values outside the slots a slot starts from (uniform, defined by an
    /// operation): their slot form is built where they are defined.
    pub slot_entries: HashSet<Value>,
    /// Ballot -> the lane reads at its first set lane that ride on its
    /// exchange ([`first_lane_reads`]).
    pub fused: HashMap<Value, Vec<Value>>,
    /// Ballot -> the later ballots made with its exchange ([`ballot_riders`]).
    pub riders: HashMap<Value, Vec<Value>>,
    /// Riding ballot -> the operations computed again at its host, operands
    /// first.
    pub recompute: HashMap<Value, Vec<Value>>,
}

/// Emission state of a split program.
pub struct Split {
    pub plan: SplitPlan,
    /// The exchange array (workgroup memory) and the parity variable.
    xchg: Option<(Word, Word)>,
    zeros: HashMap<Type, Word>,
    /// Values equal in every lane of the wave (computed on first use).
    uniform: Option<Vec<bool>>,
    /// The slot form of the slot entries.
    entry_slots: HashMap<Value, Word>,
    /// Words of slot values phis outside the slots take: (phi, argument)
    /// -> the word, taken at the end of the predecessor.
    phi_words: HashMap<(Value, usize), Word>,
    /// Results of the lane reads and ballots made with an earlier ballot's
    /// exchange.
    fused_results: HashMap<Value, Word>,
}

impl Split {
    pub fn new(plan: SplitPlan) -> Split {
        Split { plan, xchg: None, zeros: HashMap::new(), uniform: None, entry_slots: HashMap::new(), phi_words: HashMap::new(), fused_results: HashMap::new() }
    }

    /// Whether `v` is kept as lane slots.
    pub fn is_slot(&self, v: Value) -> bool {
        !self.plan.slots.is_empty() && self.plan.slots.contains_key(&v)
    }
}

/// The selection control of a branch on `cond`: a branch on a value that
/// differs between lanes (an EXEC run's) is kept as a branch, so a half
/// whose lanes are all off skips it instead of running it predicated.
pub fn selection_control(e: &mut Emitter, cond: Value) -> rspirv::spirv::SelectionControl {
    use rspirv::spirv::SelectionControl;
    let f = e.f;
    let Some(s) = e.split.as_mut() else { return SelectionControl::NONE };
    let uniform = s.uniform.get_or_insert_with(|| ps5shader::opt::uniform::uniform_values(f));
    if uniform[cond.index()] {
        SelectionControl::NONE
    } else {
        SelectionControl::DONT_FLATTEN
    }
}

/// 32-bit words a value of `ty` takes in an exchange.
fn dwords(ty: Type) -> u32 {
    let per = if ty.component().bits() == 64 { 2 } else { 1 };
    ty.components() * per
}

/// The value of `v` in every lane of a wave, when it depends on the lane
/// index and constants only.
fn lane_values(func: &Function, v: Value, memo: &mut HashMap<Value, Option<Box<[u64; 64]>>>, budget: &mut u32) -> Option<Box<[u64; 64]>> {
    if let Some(m) = memo.get(&v) {
        return m.clone();
    }
    if *budget == 0 {
        return None;
    }
    *budget -= 1;
    let d = func.value(v);
    let r = (|| -> Option<Box<[u64; 64]>> {
        if d.ty.is_vector() || d.ty == Type::VOID {
            return None;
        }
        match d.op {
            Op::Const(c) => return Some(Box::new([c; 64])),
            Op::LaneId => return Some(Box::new(std::array::from_fn(|l| l as u64))),
            _ => {}
        }
        if !d.op.is_pure() || d.op.is_subgroup() || matches!(d.op, Op::Phi | Op::Undef | Op::SystemValue(_) | Op::UserData(_) | Op::InterpParam { .. } | Op::CodeAddress | Op::InverseBallot) {
            return None;
        }
        let args: Vec<Box<[u64; 64]>> = d.args.iter().map(|&a| lane_values(func, a, memo, budget)).collect::<Option<_>>()?;
        if d.args.iter().any(|&a| func.ty(a).is_vector()) {
            return None;
        }
        let mut out = [0u64; 64];
        for (l, o) in out.iter_mut().enumerate() {
            let a: Vec<(Type, ps5shader::interp::eval::Val)> = d.args.iter().zip(&args).map(|(&x, vals)| (func.ty(x), [vals[l], 0, 0, 0])).collect();
            *o = ps5shader::interp::eval::eval(d.op, d.ty, &a).ok()?[0];
        }
        Some(Box::new(out))
    })();
    memo.insert(v, r.clone());
    r
}

/// Whether `v` is below `n` (a power of two) in every lane.
fn below(func: &Function, v: Value, n: u64, depth: u32) -> bool {
    if depth > 16 {
        return false;
    }
    let d = func.value(v);
    if d.ty != Type::U32 {
        return false;
    }
    let a = |i: usize| d.args[i];
    let c = |i: usize| func.as_const(d.args[i]);
    match d.op {
        Op::Const(x) => x < n,
        Op::LaneId => n >= 64,
        Op::And | Op::UMin => below(func, a(0), n, depth + 1) || below(func, a(1), n, depth + 1),
        Op::Or | Op::Xor => below(func, a(0), n, depth + 1) && below(func, a(1), n, depth + 1),
        Op::Select => below(func, a(1), n, depth + 1) && below(func, a(2), n, depth + 1),
        Op::LShr => c(1).is_some_and(|s| s & 31 >= 32 - n.trailing_zeros().min(32) as u64) || below(func, a(0), n, depth + 1),
        Op::BitFieldUExtract => c(2).is_some_and(|w| (w & 31) < 32 && 1u64 << (w & 31) <= n),
        _ => false,
    }
}

/// Whether `v` is, in every lane, a lane of the same half (below 64, bit 5
/// equal to the lane's): the lane with bits of its half kept, or with low
/// bits below 32 or-ed or xor-ed in (DPP rows, ds_bpermute and
/// ds_swizzle sources, permlanes).
fn same_half(func: &Function, v: Value, depth: u32) -> bool {
    if depth > 16 {
        return false;
    }
    let d = func.value(v);
    if d.ty != Type::U32 {
        return false;
    }
    let a = |i: usize| d.args[i];
    match d.op {
        Op::LaneId => true,
        Op::And => (0..2).any(|i| same_half(func, a(i), depth + 1) && func.as_const(a(1 - i)).is_some_and(|m| m & 32 != 0)),
        Op::Or | Op::Xor => (0..2).any(|i| same_half(func, a(i), depth + 1) && below(func, a(1 - i), 32, depth + 1)),
        Op::Select => same_half(func, a(1), depth + 1) && same_half(func, a(2), depth + 1),
        _ => false,
    }
}

/// Whether the source lane `target` of a shuffle is, in every lane, a lane
/// of the reader's own half.
fn source_in_half(func: &Function, target: Value, memo: &mut HashMap<Value, Option<Box<[u64; 64]>>>) -> bool {
    if same_half(func, target, 0) {
        return true;
    }
    let mut budget = 512;
    lane_values(func, target, memo, &mut budget).is_some_and(|t| t.iter().enumerate().all(|(l, &s)| s < 64 && s >> 5 == (l as u64) >> 5))
}

/// Values that may differ between the waves of a work-group, and the blocks
/// under control flow that may differ between them.
fn workgroup_divergence(func: &Function) -> (Vec<bool>, Vec<bool>) {
    use ps5shader::cfg::dom::post_dominators;
    use ps5shader::ssa::build::block_successors;
    let uniform = ps5shader::opt::uniform::uniform_values(func);
    let mut div: Vec<bool> = uniform.iter().map(|u| !u).collect();
    for &b in &func.order {
        for &v in &func.block(b).insts {
            let wave_dependent = match func.op(v) {
                Op::Ballot | Op::ReadLane | Op::LdsAppend { .. } | Op::GdsAppend { .. } | Op::GdsLoad { .. } | Op::GdsAtomic { .. } | Op::Clock { .. } => true,
                Op::SystemValue(s) => !matches!(s, SystemValue::WorkgroupId(_)),
                _ => false,
            };
            div[v.index()] |= wave_dependent;
        }
    }
    let succs = block_successors(func);
    let ipdom = post_dominators(&succs, func.entry.index());
    let nb = func.blocks.len();
    let mut region = vec![false; nb];
    let mut done = vec![false; nb];
    loop {
        let mut changed = true;
        while changed {
            changed = false;
            for &b in &func.order {
                for &v in &func.block(b).insts {
                    if !div[v.index()] && func.args(v).iter().any(|a| div[a.index()]) {
                        div[v.index()] = true;
                        changed = true;
                    }
                }
            }
        }
        let mut grew = false;
        for &b in &func.order {
            let h = b.index();
            let Terminator::CondBranch { cond, .. } = func.block(b).term else { continue };
            if done[h] || !div[cond.index()] {
                continue;
            }
            done[h] = true;
            let stop = ipdom[h];
            let mut seen = vec![false; nb];
            let mut stack = succs[h].clone();
            while let Some(x) = stack.pop() {
                if x >= nb || x == stop || seen[x] {
                    continue;
                }
                seen[x] = true;
                region[x] = true;
                stack.extend(succs[x].iter().copied());
            }
            // The join's phis take each wave's value from its own edge.
            if stop < nb {
                for &v in &func.blocks[stop].insts {
                    if func.op(v) == Op::Phi && !div[v.index()] {
                        div[v.index()] = true;
                        grew = true;
                    }
                }
            }
        }
        // A value computed under divergent control flow (a loop's exit
        // values in particular) differs between the waves.
        for &b in &func.order {
            if region[b.index()] {
                for &v in &func.block(b).insts {
                    if !div[v.index()] {
                        div[v.index()] = true;
                        grew = true;
                    }
                }
            }
        }
        if !grew {
            break;
        }
    }
    (div, region)
}

/// `v` as a lane write (`v_writelane`: `select(lane_id == l, x, old)`) of a
/// uniform `x` at a uniform lane `l` below 64: `(l, x, old)`.
fn lane_write(func: &Function, v: Value, uniform: &[bool]) -> Option<(Value, Value, Value)> {
    let d = func.value(v);
    if d.op != Op::Select || d.ty != Type::U32 {
        return None;
    }
    let c = func.value(d.args[0]);
    if c.op != Op::ICmp(ICmp::Eq) {
        return None;
    }
    let (a, b) = (c.args[0], c.args[1]);
    let l = if func.op(a) == Op::LaneId {
        b
    } else if func.op(b) == Op::LaneId {
        a
    } else {
        return None;
    };
    (uniform[l.index()] && uniform[d.args[1].index()] && below(func, l, WAVE as u64, 0)).then_some((l, d.args[1], d.args[2]))
}

/// Blocks under a branch whose condition differs between the lanes of a
/// wave (its halves may take different paths), and the joins of such
/// branches.
fn lane_divergent_blocks(func: &Function, uniform: &[bool]) -> (Vec<bool>, Vec<bool>) {
    use ps5shader::cfg::dom::post_dominators;
    use ps5shader::ssa::build::block_successors;
    let nb = func.blocks.len();
    let (mut region, mut join) = (vec![false; nb], vec![false; nb]);
    let divergent: Vec<usize> = func
        .order
        .iter()
        .filter(|&&b| matches!(func.block(b).term, Terminator::CondBranch { cond, .. } if !uniform[cond.index()]))
        .map(|b| b.index())
        .collect();
    if divergent.is_empty() {
        return (region, join);
    }
    let succs = block_successors(func);
    let ipdom = post_dominators(&succs, func.entry.index());
    for h in divergent {
        let stop = ipdom[h];
        let mut seen = vec![false; nb];
        let mut stack = succs[h].clone();
        while let Some(x) = stack.pop() {
            if x >= nb || x == stop || seen[x] {
                continue;
            }
            seen[x] = true;
            region[x] = true;
            stack.extend(succs[x].iter().copied());
        }
        if stop < nb {
            join[stop] = true;
        }
    }
    (region, join)
}

/// What [`lane_slots`] returns: the slot values (a lane write's lane,
/// value and old register; None for a phi) and the values they start from
/// that need a slot form built where they are defined.
pub type LaneSlots = (HashMap<Value, Option<(Value, Value, Value)>>, HashSet<Value>);

/// The lane-indexed registers a split program keeps as lane slots: the
/// values made by lane writes ([`lane_write`]) and the phis joining them,
/// with what they start from (constants, undefined values, values defined
/// by an operation: a per-lane one exchanges its words between the halves
/// where it is defined, so it must be where `exchange_ok` allows an
/// exchange). In slot form every invocation holds the
/// words of guest lanes j and j + 32 (j: its lane in its half) in both
/// halves: a lane write is a select in the invocations of lane j, a lane
/// read at a uniform lane a shuffle within the half (no exchange), and
/// any other use takes the word of the invocation's own half.
///
/// Left out: values under control flow that differs between the lanes (a
/// half may skip a write the other half makes), phis at such joins, lazy
/// selects and values starting from per-lane values defined where no
/// exchange can be made. A phi outside the slots that takes a slot value
/// gets its word at the end of the predecessor ([`phi_exits`]).
/// Only groups of connected values holding a lane write are kept.
pub fn lane_slots(func: &Function, exchange_ok: &dyn Fn(Block) -> bool) -> LaneSlots {
    let uniform = ps5shader::opt::uniform::uniform_values(func);
    let n = func.values.len();
    let mut write: Vec<Option<(Value, Value, Value)>> = vec![None; n];
    let mut member = vec![false; n];
    // Why a value is not a slot (diagnostics): 1 under lane-divergent
    // control flow, 2 a source with no slot form.
    let mut why = vec![0u8; n];
    let mut any = false;
    let all: Vec<Value> = func.order.iter().flat_map(|&b| func.block(b).insts.iter().copied()).collect();
    for &v in &all {
        if func.lazy_selects.contains(&v) {
            continue;
        }
        if let Some(w) = lane_write(func, v, &uniform) {
            write[v.index()] = Some(w);
            member[v.index()] = true;
            any = true;
        } else if func.op(v) == Op::Phi && func.ty(v) == Type::U32 {
            member[v.index()] = true;
        }
    }
    if !any {
        return (HashMap::new(), HashSet::new());
    }
    let (region, join) = lane_divergent_blocks(func, &uniform);
    for &v in &all {
        let Some(b) = func.value(v).block else { continue };
        if member[v.index()] && (region[b.index()] || (join[b.index()] && func.op(v) == Op::Phi)) {
            member[v.index()] = false;
            why[v.index()] = 1;
        }
    }
    // A per-lane start gets the other half's words through an exchange
    // where it is defined: both halves (and with several waves every wave)
    // must get there.
    let entry_ok = |a: Value| {
        let d = func.value(a);
        matches!(d.op, Op::Const(_) | Op::Undef) || (d.ty == Type::U32 && d.block.is_some_and(|b| uniform[a.index()] || (!region[b.index()] && exchange_ok(b))))
    };
    let sources = |v: Value| -> Vec<Value> {
        match write[v.index()] {
            Some((_, _, old)) => vec![old],
            None => func.args(v).to_vec(),
        }
    };
    loop {
        let mut changed = false;
        for &v in &all {
            if member[v.index()] && !sources(v).iter().all(|&a| member[a.index()] || entry_ok(a)) {
                member[v.index()] = false;
                why[v.index()] = 2;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // Keep the connected groups holding a lane write.
    let mut adj: HashMap<Value, Vec<Value>> = HashMap::new();
    for &v in &all {
        if member[v.index()] {
            for a in sources(v) {
                if member[a.index()] {
                    adj.entry(v).or_default().push(a);
                    adj.entry(a).or_default().push(v);
                }
            }
        }
    }
    let mut keep = vec![false; n];
    let mut stack: Vec<Value> = all.iter().copied().filter(|v| member[v.index()] && write[v.index()].is_some()).collect();
    while let Some(v) = stack.pop() {
        if std::mem::replace(&mut keep[v.index()], true) {
            continue;
        }
        if let Some(next) = adj.get(&v) {
            stack.extend(next.iter().copied().filter(|a| !keep[a.index()]));
        }
    }
    let mut slots = HashMap::new();
    let mut entries = HashSet::new();
    for &v in &all {
        if keep[v.index()] {
            slots.insert(v, write[v.index()]);
            for a in sources(v) {
                if !keep[a.index()] && !matches!(func.op(a), Op::Const(_) | Op::Undef) {
                    entries.insert(a);
                }
            }
        }
    }
    if log::log_enabled!(log::Level::Trace) {
        for &v in &all {
            if func.op(v) == Op::ReadLane && !keep[func.args(v)[0].index()] {
                let src = func.args(v)[0];
                let w = write[src.index()].map(|(l, x, old)| format!(" write lane %{} value %{} old %{} ({}, why {})", l.0, x.0, old.0, func.op(old).name(), why[old.index()]));
                let args: Vec<String> = if func.op(src) == Op::Phi {
                    func.args(src)
                        .iter()
                        .map(|&a| {
                            let sel = if func.op(a) == Op::Select {
                                let c = func.args(a)[0];
                                let ca: Vec<String> = func.args(c).iter().map(|&x| format!("%{}:{}{}", x.0, func.op(x).name(), if uniform[x.index()] { "/u" } else { "" })).collect();
                                format!("(cond %{} {} [{}] x u{} old %{}:{})", c.0, func.op(c).name(), ca.join(","), uniform[func.args(a)[1].index()] as u8, func.args(a)[2].0, func.op(func.args(a)[2]).name())
                            } else {
                                String::new()
                            };
                            format!("%{}={}{}{}{}", a.0, func.op(a).name(), if member[a.index()] { "/m" } else { "" }, if write[a.index()].is_some() { "/w" } else { "" }, sel)
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                log::trace!("[shader] lane read %{} of %{} ({}) is no lane slot: why {}{} {}", v.0, src.0, func.op(src).name(), why[src.index()], w.unwrap_or_default(), args.join(" "));
            }
        }
    }
    (slots, entries)
}

/// Whether `lane` is, for every mask, the first set lane of ballot `b`'s
/// 64-bit mask (63 for an empty mask): `s_ff1_i32_b64` masked to the wave,
/// checked by evaluating the expression on sample masks.
fn first_lane_of(func: &Function, lane: Value, b: Value) -> bool {
    use ps5shader::interp::eval::eval;
    fn value(func: &Function, v: Value, b: Value, mask: u64, depth: u32) -> Option<u64> {
        if depth > 12 {
            return None;
        }
        let d = func.value(v);
        match d.op {
            Op::Const(c) => return Some(c),
            Op::Extract(i) if d.args[0] == b => return Some(if i == 0 { mask & 0xFFFF_FFFF } else { mask >> 32 }),
            _ => {}
        }
        if d.ty.is_vector() || !d.op.is_pure() || d.op.is_subgroup() || matches!(d.op, Op::Phi | Op::Undef | Op::LaneId) {
            return None;
        }
        let mut args = Vec::with_capacity(d.args.len());
        for &a in &d.args {
            if func.ty(a).is_vector() {
                return None;
            }
            args.push((func.ty(a), [value(func, a, b, mask, depth + 1)?, 0, 0, 0]));
        }
        eval(d.op, d.ty, &args).ok().map(|r| r[0])
    }
    const MASKS: [u64; 9] = [0, 1, 0x8000_0000, 1 << 32, 1 << 63, 0x0000_0005_0000_0006, 0x40 << 32, 0x0000_0007_FFF0_0000, u64::MAX];
    MASKS.iter().all(|&m| value(func, lane, b, m, 0).is_some_and(|l| l & 0xFFFF_FFFF == if m == 0 { 63 } else { m.trailing_zeros() as u64 }))
}

/// Lane reads at the first set lane of a ballot (the selection of a hit
/// in a traversal: `s_ff1` of the mask, then `v_readlane`): each half
/// takes the value at its own first set lane when it makes the ballot, and
/// sends it with its ballot word, so the read needs no exchange of its
/// own. The read value must be defined where the ballot is made
/// (dominating it) and not be a lane slot; the read's 32-bit words go with
/// the ballot word. Returns ballot -> reads.
fn first_lane_reads(func: &Function, slots: &HashMap<Value, Option<(Value, Value, Value)>>) -> HashMap<Value, Vec<Value>> {
    use ps5shader::cfg::dom::{dominates, dominators};
    use ps5shader::ssa::build::block_successors;
    let mut out: HashMap<Value, Vec<Value>> = HashMap::new();
    let mut idom = None;
    let mut position: HashMap<Value, (usize, usize)> = HashMap::new();
    for &b in &func.order {
        for (i, &v) in func.block(b).insts.iter().enumerate() {
            position.insert(v, (b.index(), i));
        }
    }
    for &b in &func.order {
        for &r in &func.block(b).insts {
            if func.op(r) != Op::ReadLane {
                continue;
            }
            let (x, lane) = (func.args(r)[0], func.args(r)[1]);
            if slots.contains_key(&x) || func.ty(r).is_vector() || dwords(func.ty(r)) > 2 {
                continue;
            }
            // The ballot the lane comes from: the one its extracts name.
            let mut stack = vec![(lane, 0u32)];
            let mut ballot = None;
            while let Some((v, depth)) = stack.pop() {
                if depth > 12 {
                    continue;
                }
                let d = func.value(v);
                if let Op::Extract(_) = d.op {
                    if func.op(d.args[0]) == Op::Ballot {
                        ballot = Some(d.args[0]);
                        break;
                    }
                }
                stack.extend(d.args.iter().map(|&a| (a, depth + 1)));
            }
            let Some(bal) = ballot else { continue };
            if !first_lane_of(func, lane, bal) {
                continue;
            }
            // `x` must be available where the ballot is made.
            let (Some(&(xb, xi)), Some(&(bb, bi))) = (position.get(&x), position.get(&bal)) else { continue };
            let ok = if xb == bb {
                xi < bi
            } else {
                let idom = idom.get_or_insert_with(|| dominators(&block_successors(func), func.entry.index()));
                dominates(idom, xb, bb)
            };
            if ok {
                out.entry(bal).or_default().push(r);
            }
        }
    }
    out
}

/// Operations a ballot's predicate may be computed again with where the
/// host ballot is made ([`ballot_riders`]): pure, per lane, no lane
/// exchange of their own.
fn recomputable(func: &Function, v: Value, slots: &HashMap<Value, Option<(Value, Value, Value)>>) -> bool {
    let op = func.op(v);
    op.is_pure() && !op.is_subgroup() && !matches!(op, Op::Phi | Op::Undef | Op::Const(_)) && !slots.contains_key(&v) && !func.lazy_selects.contains(&v)
}

/// Ballots made with the exchange of an earlier ballot that dominates them
/// (the host): the predicate, and the values of the lane reads at its
/// first lane ([`first_lane_reads`]), are defined before the host, or are
/// pure arithmetic of such values (up to 24 operations, computed again at
/// the host: [`ensure_emitted`]). A ballot depends on nothing but its
/// predicate and the lanes present, so it has the same value made at the
/// host. Returns host -> riders, each host the earliest such ballot (a
/// host rides on nothing).
/// Ballots riding on one host at most (`RPS5E_WAVE64_MAX_RIDERS`, default 3).
fn max_riders() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("RPS5E_WAVE64_MAX_RIDERS").ok().and_then(|v| v.parse().ok()).unwrap_or(3))
}

type Riders = (HashMap<Value, Vec<Value>>, HashMap<Value, Vec<Value>>);

fn ballot_riders(func: &Function, fused: &HashMap<Value, Vec<Value>>, slots: &HashMap<Value, Option<(Value, Value, Value)>>) -> Riders {
    use ps5shader::cfg::dom::{dominates, dominators};
    use ps5shader::ssa::build::block_successors;
    let mut position: HashMap<Value, (usize, usize)> = HashMap::new();
    let mut ballots = Vec::new();
    for &b in &func.order {
        for (i, &v) in func.block(b).insts.iter().enumerate() {
            position.insert(v, (b.index(), i));
            if func.op(v) == Op::Ballot {
                ballots.push(v);
            }
        }
    }
    let mut out: HashMap<Value, Vec<Value>> = HashMap::new();
    let mut recompute: HashMap<Value, Vec<Value>> = HashMap::new();
    if ballots.len() < 2 {
        return (out, recompute);
    }
    let idom = dominators(&block_successors(func), func.entry.index());
    let before = |x: Value, at: (usize, usize)| match position.get(&x) {
        // Constants and undefined values are everywhere.
        None => func.value(x).block.is_none(),
        Some(&(xb, xi)) => if xb == at.0 { xi < at.1 } else { dominates(&idom, xb, at.0) },
    };
    // The operations to compute again at `at` for `needs` (operands
    // first): None when one is neither defined before `at` nor
    // recomputable, or there are too many.
    let available = |needs: &[Value], at: (usize, usize)| -> Option<Vec<Value>> {
        fn visit(func: &Function, v: Value, ok: &dyn Fn(Value) -> bool, slots: &HashMap<Value, Option<(Value, Value, Value)>>, seen: &mut HashSet<Value>, order: &mut Vec<Value>) -> bool {
            if ok(v) || !seen.insert(v) {
                return true;
            }
            if seen.len() > 24 || !recomputable(func, v, slots) {
                return false;
            }
            for &a in func.args(v) {
                if !visit(func, a, ok, slots, seen, order) {
                    return false;
                }
            }
            order.push(v);
            true
        }
        let (mut seen, mut order) = (HashSet::new(), Vec::new());
        let ok = |v: Value| before(v, at);
        needs.iter().all(|&x| visit(func, x, &ok, slots, &mut seen, &mut order)).then_some(order)
    };
    let mut hosts: Vec<Value> = Vec::new();
    for &b in &ballots {
        let pb = position[&b];
        let mut needs = vec![func.args(b)[0]];
        needs.extend(fused.get(&b).into_iter().flatten().map(|&r| func.args(r)[0]));
        let host = hosts.iter().copied().find_map(|h| {
            let ph = position[&h];
            let dominated = if ph.0 == pb.0 { ph.1 < pb.1 } else { dominates(&idom, ph.0, pb.0) };
            if !dominated || out.get(&h).is_some_and(|r| r.len() >= max_riders()) {
                return None;
            }
            available(&needs, ph).map(|order| (h, order))
        });
        let host = host.map(|(h, order)| {
            if !order.is_empty() {
                recompute.insert(b, order);
            }
            h
        });
        if host.is_none() && log::log_enabled!(log::Level::Trace) {
            // Diagnostics: what keeps the ballot from the nearest host.
            if let Some(&h) = hosts.iter().rev().find(|&&h| {
                let ph = position[&h];
                if ph.0 == pb.0 { ph.1 < pb.1 } else { dominates(&idom, ph.0, pb.0) }
            }) {
                let ph = position[&h];
                let mut stack = needs.clone();
                let mut seen = HashSet::new();
                let mut blockers = Vec::new();
                while let Some(v) = stack.pop() {
                    if before(v, ph) || !seen.insert(v) {
                        continue;
                    }
                    if !recomputable(func, v, slots) {
                        blockers.push(format!("%{}={}", v.0, func.op(v).name()));
                        continue;
                    }
                    stack.extend(func.args(v).iter().copied());
                }
                log::trace!("[shader] ballot %{} does not ride on %{}: {} values, blocked by {}", b.0, h.0, seen.len(), blockers.join(" "));
            }
        }
        match host {
            Some(h) => out.entry(h).or_default().push(b),
            None => hosts.push(b),
        }
    }
    (out, recompute)
}

/// Decide whether a wave64 compute program runs split; the reason when not.
/// `local_size`: the work-group size; `lds_dwords`: the LDS it allocates
/// (0: the default); `native_shuffles`: shuffles within a half use the host
/// subgroup (else every shuffle goes through memory: tests).
pub fn plan(func: &Function, local_size: [u32; 3], lds_dwords: u32, features: &Features, native_shuffles: bool) -> Result<SplitPlan, String> {
    if func.wave_size != WAVE || func.stage != Stage::Compute {
        return Err("not a wave64 compute program".into());
    }
    let s = local_size;
    let threads = s[0].max(1) * s[1].max(1) * s[2].max(1);
    let waves = threads.div_ceil(WAVE);
    let mut plan = SplitPlan { waves, ..SplitPlan::default() };
    let (mut word_dwords, mut lane_dwords) = (0u32, 0u32);
    if lane_slots_enabled() && native_shuffles {
        let region = (waves > 1).then(|| workgroup_divergence(func).1);
        (plan.slots, plan.slot_entries) = lane_slots(func, &|b: Block| region.as_ref().is_none_or(|r| !r[b.index()]));
        let uniform = ps5shader::opt::uniform::uniform_values(func);
        if plan.slot_entries.iter().any(|e| !uniform[e.index()]) {
            lane_dwords = 1;
        }
    }
    if lane_slots_enabled() && native_shuffles {
        plan.fused = first_lane_reads(func, &plan.slots);
    }
    if ballot_riders_enabled() && native_shuffles {
        (plan.riders, plan.recompute) = ballot_riders(func, &plan.fused, &plan.slots);
    }
    let fused_reads: HashSet<Value> = plan.fused.values().flatten().copied().collect();
    let riding: HashSet<Value> = plan.riders.values().flatten().copied().collect();
    let mut exchanges = Vec::new();
    let mut memo = HashMap::new();
    let (mut uses_lds, mut lds64) = (false, false);
    for &b in &func.order {
        for &v in &func.block(b).insts {
            let op = func.op(v);
            uses_lds |= op.is_lds_read() || op.is_lds_write();
            if matches!(op, Op::LdsAtomic { .. } | Op::GlobalAtomic { flat: true, .. }) && func.args(v).get(1).is_some_and(|&a| func.ty(a) == Type::U64) {
                lds64 = true;
            }
            match op {
                Op::LdsAppend { .. } | Op::GdsAppend { .. } => return Err("wave-wide append".into()),
                Op::Clock { .. } => return Err("clock read".into()),
                Op::Ballot if riding.contains(&v) => {
                    plan.counts[7] += 1;
                }
                Op::Ballot => {
                    let reads = |b: Value| -> u32 { 1 + plan.fused.get(&b).map_or(0, |r| r.iter().map(|&x| dwords(func.ty(x))).sum::<u32>()) };
                    let words = reads(v) + plan.riders.get(&v).map_or(0, |rs| rs.iter().map(|&b| reads(b)).sum());
                    word_dwords = word_dwords.max(words);
                    plan.counts[0] += 1;
                    exchanges.push(v);
                }
                Op::ReadLane if plan.slots.contains_key(&func.args(v)[0]) => {
                    plan.counts[5] += 1;
                }
                Op::ReadLane if fused_reads.contains(&v) => {
                    plan.counts[6] += 1;
                }
                Op::ReadLane => {
                    word_dwords = word_dwords.max(dwords(func.ty(v)));
                    plan.counts[1] += 1;
                    exchanges.push(v);
                }
                Op::Shuffle => {
                    if native_shuffles && source_in_half(func, func.args(v)[1], &mut memo) {
                        plan.in_half.insert(v);
                        plan.counts[3] += 1;
                    } else {
                        lane_dwords = lane_dwords.max(dwords(func.ty(v)));
                        plan.counts[2] += 1;
                        exchanges.push(v);
                    }
                }
                Op::SubgroupLdsBarrier => {
                    plan.counts[4] += 1;
                    exchanges.push(v);
                }
                _ => {}
            }
        }
    }
    if lds64 && features.shared_int64_atomics {
        // The explicitly laid out LDS block would alias the exchange array.
        return Err("64-bit LDS atomics".into());
    }
    plan.slot_dwords = (2 * word_dwords).max(WAVE * lane_dwords);
    let xchg_bytes = waves * 2 * plan.slot_dwords * 4;
    let lds_bytes = if uses_lds {
        4 * match lds_dwords {
            0 => DEFAULT_LDS_DWORDS,
            n => n.min(DEFAULT_LDS_DWORDS),
        }
    } else {
        0
    };
    let limit = if features.max_shared_bytes == 0 { DEFAULT_SHARED_BYTES } else { features.max_shared_bytes };
    if xchg_bytes > 0 && lds_bytes + xchg_bytes > limit {
        return Err(format!("{lds_bytes} bytes of LDS and {xchg_bytes} bytes of exchange slots exceed {limit}"));
    }
    if waves > 1 && !exchanges.is_empty() {
        let (_, region) = workgroup_divergence(func);
        if let Some(&v) = exchanges.iter().find(|v| func.value(**v).block.is_some_and(|b| region[b.index()])) {
            return Err(format!("{} %{} under control flow that differs between the {waves} waves", func.op(v).name(), v.0));
        }
    }
    Ok(plan)
}

// ----------------------------------------------------------------------
// Emission
// ----------------------------------------------------------------------

/// The local invocation index (the guest's: the host work-group is the
/// guest's flattened into whole waves).
fn local_index(e: &mut Emitter) -> Word {
    let tu = e.t_u32();
    e.builtin_value(BuiltIn::LocalInvocationIndex, tu)
}

/// Guest lane (0..63) of this invocation.
pub fn guest_lane(e: &mut Emitter) -> Word {
    let tu = e.t_u32();
    let li = local_index(e);
    let m = e.cu(WAVE - 1);
    e.hoist(S::BitwiseAnd, tu, &[li, m])
}

/// This invocation's half of its wave (0 or 1).
fn half(e: &mut Emitter) -> Word {
    let tu = e.t_u32();
    let li = local_index(e);
    let five = e.cu(5);
    let h = e.hoist(S::ShiftRightLogical, tu, &[li, five]);
    let one = e.cu(1);
    e.hoist(S::BitwiseAnd, tu, &[h, one])
}

/// Index of this invocation's wave in the work-group.
fn wave_index(e: &mut Emitter) -> Word {
    let tu = e.t_u32();
    let li = local_index(e);
    let six = e.cu(6);
    e.hoist(S::ShiftRightLogical, tu, &[li, six])
}

/// The first invocation of each half (it stores the half's words).
fn writer(e: &mut Emitter) -> Word {
    let tu = e.t_u32();
    let tb = e.t_bool();
    let li = local_index(e);
    let m = e.cu(31);
    let l = e.hoist(S::BitwiseAnd, tu, &[li, m]);
    let zero = e.cu(0);
    e.hoist(S::IEqual, tb, &[l, zero])
}

/// A zero of `t` (the split program's undefined values: both halves of a
/// wave must see the same one).
pub fn zero(e: &mut Emitter, t: Type) -> Word {
    if let Some(&w) = e.split.as_ref().and_then(|s| s.zeros.get(&t)) {
        return w;
    }
    let ty = e.ty(t);
    let w = e.b.constant_null(ty);
    if let Some(s) = e.split.as_mut() {
        s.zeros.insert(t, w);
    }
    w
}

/// The exchange array and the parity variable (declared on first use).
fn exchange_vars(e: &mut Emitter) -> (Word, Word) {
    if let Some(v) = e.split.as_ref().and_then(|s| s.xchg) {
        return v;
    }
    let (waves, slot) = e.split.as_ref().map(|s| (s.plan.waves, s.plan.slot_dwords)).unwrap_or((1, 2));
    let tu = e.t_u32();
    let n = e.cu((waves * 2 * slot.max(1)).max(1));
    let id = e.b.id();
    let arr = e.b.type_array_id(Some(id), tu, n);
    let xchg = e.global(arr, SC::Workgroup);
    let zero = e.cu(0);
    let p = e.ptr(SC::Private, tu);
    let parity = e.b.id();
    let inst = Instruction::new(S::Variable, Some(p), Some(parity), vec![Operand::StorageClass(SC::Private), Operand::IdRef(zero)]);
    e.b.module_mut().types_global_values.push(inst);
    e.globals.push(parity);
    if let Some(s) = e.split.as_mut() {
        s.xchg = Some((xchg, parity));
    }
    (xchg, parity)
}

/// Workgroup execution and workgroup memory barrier.
fn barrier(e: &mut Emitter) {
    let (s, sem) = (e.cu(SCOPE_WORKGROUP), e.cu(0x108));
    e.void(S::ControlBarrier, vec![Operand::IdRef(s), Operand::IdRef(s), Operand::IdRef(sem)]);
}

/// Start an exchange: the first dword of this wave's current slot, and the
/// parity loaded for it.
fn begin_exchange(e: &mut Emitter) -> (Word, Word, Word) {
    let (xchg, parity) = exchange_vars(e);
    let slot = e.split.as_ref().map_or(2, |s| s.plan.slot_dwords);
    let tu = e.t_u32();
    let p = e.load(tu, parity);
    let wave = wave_index(e);
    let two = e.cu(2);
    let w2 = e.imul(wave, two);
    let k = e.iadd(w2, p);
    let n = e.cu(slot);
    let base = e.imul(k, n);
    (xchg, base, p)
}

/// End an exchange: the next one uses the other slot.
fn end_exchange(e: &mut Emitter, p: Word) {
    let (_, parity) = exchange_vars(e);
    let tu = e.t_u32();
    let one = e.cu(1);
    let q = e.ins(S::BitwiseXor, tu, &[p, one]);
    e.store(parity, q);
}

fn slot_ptr(e: &mut Emitter, xchg: Word, index: Word) -> Word {
    let tu = e.t_u32();
    let p = e.ptr(SC::Workgroup, tu);
    e.access(p, xchg, &[index])
}

/// Exchange words that are uniform in each half: element `h` of the result
/// holds half `h`'s words.
fn exchange_words(e: &mut Emitter, own: &[Word]) -> [Vec<Word>; 2] {
    let k = own.len() as u32;
    let (xchg, base, p) = begin_exchange(e);
    let h = half(e);
    let kk = e.cu(k);
    let hk = e.imul(h, kk);
    let mine = e.iadd(base, hk);
    let w = writer(e);
    let own = own.to_vec();
    e.if_then(w, |e| {
        for (i, &x) in own.iter().enumerate() {
            let c = e.cu(i as u32);
            let idx = e.iadd(mine, c);
            let ptr = slot_ptr(e, xchg, idx);
            e.store(ptr, x);
        }
    });
    barrier(e);
    let tu = e.t_u32();
    let out = [0u32, 1].map(|from| {
        (0..k)
            .map(|i| {
                let c = e.cu(from * k + i);
                let idx = e.iadd(base, c);
                let ptr = slot_ptr(e, xchg, idx);
                e.load(tu, ptr)
            })
            .collect::<Vec<_>>()
    });
    end_exchange(e, p);
    out
}

/// Every lane's words, read back from lane `source` (any lane of the wave).
fn exchange_lanes(e: &mut Emitter, words: &[Word], source: Word) -> Vec<Word> {
    let (xchg, base, p) = begin_exchange(e);
    let lane = guest_lane(e);
    let m = e.cu(WAVE - 1);
    let src = e.band(source, m);
    let at = |e: &mut Emitter, i: usize, l: Word| {
        let c = e.cu(i as u32 * WAVE);
        let b = e.iadd(base, c);
        let idx = e.iadd(b, l);
        slot_ptr(e, xchg, idx)
    };
    for (i, &x) in words.iter().enumerate() {
        let ptr = at(e, i, lane);
        e.store(ptr, x);
    }
    barrier(e);
    let tu = e.t_u32();
    let out = (0..words.len())
        .map(|i| {
            let ptr = at(e, i, src);
            e.load(tu, ptr)
        })
        .collect();
    end_exchange(e, p);
    out
}

/// A value of `ty` as 32-bit words.
fn to_words(e: &mut Emitter, ty: Type, x: Word) -> Vec<Word> {
    let tu = e.t_u32();
    if ty.is_vector() {
        let s = Type::Scalar(ty.component());
        let st = e.ty(s);
        return (0..ty.components()).flat_map(|i| {
            let c = e.extract(st, x, i);
            to_words(e, s, c)
        }).collect();
    }
    let s = ty.component();
    match (s, s.bits()) {
        (Scalar::Bool, _) => {
            let (one, zero) = (e.cu(1), e.cu(0));
            vec![e.select(tu, x, one, zero)]
        }
        (_, 64) => {
            let t2 = e.t_uvec(2);
            let v = e.bitcast(t2, x);
            vec![e.extract(tu, v, 0), e.extract(tu, v, 1)]
        }
        (_, 16) => {
            let t16 = e.ty(Type::U16);
            let u = if s == Scalar::F16 { e.bitcast(t16, x) } else { x };
            vec![e.ins(S::UConvert, tu, &[u])]
        }
        _ => vec![if s == Scalar::U32 { x } else { e.bitcast(tu, x) }],
    }
}

/// The value of `ty` made of `words` (inverse of [`to_words`]).
fn from_words(e: &mut Emitter, ty: Type, words: &[Word]) -> Word {
    if ty.is_vector() {
        let s = Type::Scalar(ty.component());
        let per = dwords(s) as usize;
        let parts: Vec<Word> = words.chunks(per).map(|c| from_words(e, s, c)).collect();
        let t = e.ty(ty);
        return e.construct(t, &parts);
    }
    let s = ty.component();
    let t = e.ty(ty);
    match (s, s.bits()) {
        (Scalar::Bool, _) => {
            let zero = e.cu(0);
            e.ine(words[0], zero)
        }
        (_, 64) => {
            let t2 = e.t_uvec(2);
            let v = e.construct(t2, &words[..2]);
            e.bitcast(t, v)
        }
        (_, 16) => {
            let t16 = e.ty(Type::U16);
            let u = e.ins(S::UConvert, t16, &[words[0]]);
            if s == Scalar::F16 {
                e.bitcast(t, u)
            } else {
                u
            }
        }
        _ => {
            if s == Scalar::U32 {
                words[0]
            } else {
                e.bitcast(t, words[0])
            }
        }
    }
}

/// Subgroup ballot word of `p` in this half.
fn half_ballot(e: &mut Emitter, p: Word) -> Word {
    e.cap(Capability::GroupNonUniform);
    e.cap(Capability::GroupNonUniformBallot);
    let t4 = e.t_uvec(4);
    let scope = e.cu(SCOPE_SUBGROUP);
    let m = e.ins(S::GroupNonUniformBallot, t4, &[scope, p]);
    let tu = e.t_u32();
    e.extract(tu, m, 0)
}

/// The slot form (`u32x2`: the words of lanes j and j + 32) of a slot value
/// or of a value a slot starts from.
pub fn slot_form(e: &mut Emitter, a: Value) -> R<Word> {
    let Some(s) = e.split.as_ref() else { return Err("lane slots outside a split program".into()) };
    if s.is_slot(a) {
        let w = e.defs[a.index()][0];
        return if w != 0 { Ok(w) } else { Err(format!("lane slot %{} used before its definition", a.0)) };
    }
    if let Some(&w) = s.entry_slots.get(&a) {
        return Ok(w);
    }
    let t2 = e.t_uvec(2);
    match e.f.op(a) {
        Op::Const(bits) => {
            let c = e.konst(Type::U32, bits);
            Ok(e.b.constant_composite(t2, [c, c]))
        }
        Op::Undef => Ok(zero(e, Type::vec(Scalar::U32, 2))),
        other => Err(format!("lane slot start %{} ({}) has no slot form", a.0, other.name())),
    }
}

/// At the end of block `blk`: the phis outside the slots in its successors
/// that take a slot value from it get the value's word here.
pub fn phi_exits(e: &mut Emitter, blk: Block) -> R<()> {
    if e.split.as_ref().is_none_or(|s| s.plan.slots.is_empty()) {
        return Ok(());
    }
    let f = e.f;
    for succ in f.block(blk).term.successors() {
        let preds = &f.block(succ).preds;
        for &p in &f.block(succ).insts {
            if f.op(p) != Op::Phi {
                break;
            }
            if e.split.as_ref().is_some_and(|s| s.is_slot(p)) {
                continue;
            }
            for (k, &a) in f.args(p).iter().enumerate() {
                if preds.get(k) == Some(&blk) && e.split.as_ref().is_some_and(|s| s.is_slot(a)) {
                    let w = slot_lane_word(e, a)?;
                    if let Some(s) = e.split.as_mut() {
                        s.phi_words.insert((p, k), w);
                    }
                }
            }
        }
    }
    Ok(())
}

/// The word [`phi_exits`] took for argument `k` of phi `phi`.
pub fn phi_exit_word(e: &Emitter, phi: Value, k: usize) -> Option<Word> {
    e.split.as_ref().and_then(|s| s.phi_words.get(&(phi, k)).copied())
}

/// The phis of `blk` lane slots start from get their slot form (after
/// the block's phis, where they dominate every use).
pub fn phi_entries(e: &mut Emitter, blk: Block) {
    let Some(s) = e.split.as_ref() else { return };
    if s.plan.slot_entries.is_empty() {
        return;
    }
    let phis: Vec<Value> = e.f.block(blk).insts.iter().copied().filter(|&v| e.f.op(v) == Op::Phi && s.plan.slot_entries.contains(&v)).collect();
    for v in phis {
        let w = e.defs[v.index()][0];
        if w != 0 {
            build_entry(e, v, w);
        }
    }
}

/// `v` (a value a lane slot starts from) was defined as `w`: build its
/// slot form here, where it dominates every use (a per-lane value takes
/// the other half's words through an exchange).
pub fn note_definition(e: &mut Emitter, v: Value, w: Word) {
    if !e.split.as_ref().is_some_and(|s| s.plan.slot_entries.contains(&v)) {
        return;
    }
    build_entry(e, v, w);
}

fn build_entry(e: &mut Emitter, v: Value, w: Word) {
    let t2 = e.t_uvec(2);
    let f = e.f;
    let uniform = e.split.as_mut().is_none_or(|s| s.uniform.get_or_insert_with(|| ps5shader::opt::uniform::uniform_values(f))[v.index()]);
    let pair = if uniform {
        e.construct(t2, &[w, w])
    } else {
        // The word of lane j in the other half.
        let tu = e.t_u32();
        let lane = guest_lane(e);
        let c32 = e.cu(32);
        let other_lane = e.hoist(S::BitwiseXor, tu, &[lane, c32]);
        let other = exchange_lanes(e, &[w], other_lane)[0];
        let h = half(e);
        let zero = e.cu(0);
        let low = e.ieq(h, zero);
        let lo = e.select(tu, low, w, other);
        let hi = e.select(tu, low, other, w);
        e.construct(t2, &[lo, hi])
    };
    if let Some(s) = e.split.as_mut() {
        s.entry_slots.insert(v, pair);
    }
}

/// The word of a slot value in this invocation's own lane.
pub fn slot_lane_word(e: &mut Emitter, v: Value) -> R<Word> {
    let pair = slot_form(e, v)?;
    let h = half(e);
    let tu = e.t_u32();
    Ok(e.ins(S::VectorExtractDynamic, tu, &[pair, h]))
}

/// Lane `lane` (uniform, below 64) of a value in slot form.
fn slot_read(e: &mut Emitter, pair: Word, lane_v: Value, lane: Word) -> Word {
    let tu = e.t_u32();
    let word = match e.f.as_const(lane_v) {
        Some(c) => e.extract(tu, pair, ((c >> 5) & 1) as u32),
        None => {
            let five = e.cu(5);
            let one = e.cu(1);
            let hi = e.shr(lane, five);
            let hi = e.band(hi, one);
            e.ins(S::VectorExtractDynamic, tu, &[pair, hi])
        }
    };
    let m = e.cu(31);
    let l = e.band(lane, m);
    e.shuffle(Type::U32, word, l)
}

/// A lane write of a slot value: the invocations of lane j = `lane` & 31
/// in both halves set the word of their pair `lane` selects.
fn slot_write(e: &mut Emitter, lane_v: Value, x_v: Value, old_v: Value) -> R<Word> {
    let old = slot_form(e, old_v)?;
    let x = e.val(x_v)?;
    let lane = e.val(lane_v)?;
    let tu = e.t_u32();
    let t2 = e.t_uvec(2);
    let m = e.cu(31);
    let me = guest_lane(e);
    let me = e.hoist(S::BitwiseAnd, tu, &[me, m]);
    let j = e.band(lane, m);
    let hit = e.ieq(me, j);
    let words: Vec<Word> = (0..2u32)
        .map(|w| {
            let o = e.extract(tu, old, w);
            let hit_w = match e.f.as_const(lane_v) {
                Some(c) if (c >> 5) & 1 == w as u64 => hit,
                Some(_) => return o,
                None => {
                    let c32 = e.cu(32);
                    let high = e.ule(c32, lane);
                    let side = if w == 1 { high } else { e.lnot(high) };
                    e.land(hit, side)
                }
            };
            e.select(tu, hit_w, x, o)
        })
        .collect();
    Ok(e.construct(t2, &words))
}

/// This half's words of ballot `b` appended to `words`: its ballot word,
/// then for each lane read at its first set lane the read value's words
/// at this half's first set lane (lane 31 of the half when it has none).
/// Returns the reads with their type and word span.
fn ballot_words(e: &mut Emitter, b: Value, words: &mut Vec<Word>) -> R<Vec<(Value, Type, usize, usize)>> {
    let p = e.arg(b, 0)?;
    // A ballot has no bits for lanes past the work-group.
    let out = super::io::system_value(e, SystemValue::Custom(custom::OUT_OF_RANGE))?;
    let zero = e.cu(0);
    let present = e.ieq(out, zero);
    let p = e.land(p, present);
    let own = half_ballot(e, p);
    words.push(own);
    let reads = e.split.as_ref().and_then(|s| s.plan.fused.get(&b).cloned()).unwrap_or_default();
    let mut spans = Vec::new();
    if !reads.is_empty() {
        let tu = e.t_u32();
        let first = e.gl(tu, GLOp::FindILsb, &[own]);
        let m = e.cu(31);
        let first = e.band(first, m);
        for &r in &reads {
            let ty = e.f.ty(r);
            let x = e.val(e.f.args(r)[0])?;
            let xw = to_words(e, ty, x);
            let start = words.len();
            for w in xw {
                let c = e.shuffle(Type::U32, w, first);
                words.push(c);
            }
            spans.push((r, ty, start, words.len()));
        }
    }
    Ok(spans)
}

/// The operations a split program emits differently; `None` for the rest.
pub fn emit_op(e: &mut Emitter, v: Value, d: &ValueData) -> R<Option<Option<Word>>> {
    if e.split.as_ref().is_some_and(|s| !s.plan.slots.is_empty()) {
        match d.op {
            Op::Select => {
                if let Some(Some((lane, x, old))) = e.split.as_ref().and_then(|s| s.plan.slots.get(&v).copied()) {
                    return Ok(Some(Some(slot_write(e, lane, x, old)?)));
                }
            }
            Op::ReadLane if e.split.as_ref().is_some_and(|s| s.fused_results.contains_key(&v)) => {
                return Ok(Some(e.split.as_ref().and_then(|s| s.fused_results.get(&v).copied())));
            }
            Op::ReadLane if e.split.as_ref().is_some_and(|s| s.is_slot(d.args[0])) => {
                let pair = slot_form(e, d.args[0])?;
                let lane = e.arg(v, 1)?;
                return Ok(Some(Some(slot_read(e, pair, d.args[1], lane))));
            }
            _ => {}
        }
    }
    Ok(Some(match d.op {
        Op::Ballot => {
            // A ballot riding on an earlier one's exchange was made there.
            if let Some(w) = e.split.as_ref().and_then(|s| s.fused_results.get(&v).copied()) {
                return Ok(Some(Some(w)));
            }
            let mut words = Vec::new();
            let mut ballots = Vec::new();
            let riding = e.split.as_ref().and_then(|s| s.plan.riders.get(&v).cloned()).unwrap_or_default();
            for b in std::iter::once(v).chain(riding) {
                // A riding ballot's operations not defined here are computed
                // again for it (their definitions are put back after: the
                // values' own positions, or other hosts, define them).
                let again = e.split.as_ref().and_then(|s| s.plan.recompute.get(&b).cloned()).unwrap_or_default();
                let saved: Vec<[Word; 2]> = again.iter().map(|x| e.defs[x.index()]).collect();
                for &x in &again {
                    e.emit_early(x)?;
                }
                let at = words.len();
                let spans = ballot_words(e, b, &mut words);
                for (x, d) in again.iter().zip(saved) {
                    e.defs[x.index()] = d;
                }
                ballots.push((b, at, spans?));
            }
            let [lo, hi] = exchange_words(e, &words);
            let t2 = e.t_uvec(2);
            let tu = e.t_u32();
            let zero = e.cu(0);
            let mut result = None;
            for (b, at, spans) in ballots {
                // The first set lane is in the low half when its word has one.
                let low = e.ine(lo[at], zero);
                for (r, ty, a, z) in spans {
                    let picked: Vec<Word> = (a..z).map(|i| e.select(tu, low, lo[i], hi[i])).collect();
                    let w = from_words(e, ty, &picked);
                    if let Some(s) = e.split.as_mut() {
                        s.fused_results.insert(r, w);
                    }
                }
                let mask = e.construct(t2, &[lo[at], hi[at]]);
                if b == v {
                    result = Some(mask);
                } else if let Some(s) = e.split.as_mut() {
                    s.fused_results.insert(b, mask);
                }
            }
            result
        }
        Op::ReadLane => {
            let x = e.arg(v, 0)?;
            let lane_v = e.f.args(v)[1];
            let lane = e.arg(v, 1)?;
            let words = to_words(e, d.ty, x);
            let m = e.cu(31);
            let l = e.band(lane, m);
            let own: Vec<Word> = words.iter().map(|&w| e.shuffle(Type::U32, w, l)).collect();
            let [lo, hi] = exchange_words(e, &own);
            let picked: Vec<Word> = match e.f.as_const(lane_v) {
                Some(c) => if c & 63 >= 32 { hi } else { lo },
                None => {
                    let c32 = e.cu(32);
                    let high = e.ule(c32, lane);
                    let tu = e.t_u32();
                    lo.iter().zip(&hi).map(|(&a, &b)| e.select(tu, high, b, a)).collect()
                }
            };
            Some(from_words(e, d.ty, &picked))
        }
        Op::Shuffle => {
            let x = e.arg(v, 0)?;
            let target = e.arg(v, 1)?;
            let in_half = e.split.as_ref().is_some_and(|s| s.plan.in_half.contains(&v));
            if in_half {
                let m = e.cu(31);
                let l = e.band(target, m);
                Some(e.shuffle(d.ty, x, l))
            } else {
                let words = to_words(e, d.ty, x);
                let got = exchange_lanes(e, &words, target);
                Some(from_words(e, d.ty, &got))
            }
        }
        Op::SubgroupLdsBarrier => {
            barrier(e);
            None
        }
        _ => return Ok(None),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rig() -> (Function, Block) {
        let mut f = Function::new(64, Stage::Compute, 0);
        let b = f.new_block();
        f.entry = b;
        f.order = vec![b];
        (f, b)
    }

    #[test]
    fn row_and_quad_sources_stay_in_the_half_wave_rotations_do_not() {
        let (mut f, b) = rig();
        let lane = f.append(b, Op::LaneId, Type::U32, vec![]);
        // row_shr:1 within 16-lane rows.
        let rm = f.const_u32(!15);
        let row = f.append(b, Op::And, Type::U32, vec![lane, rm]);
        let fifteen = f.const_u32(15);
        let within = f.append(b, Op::And, Type::U32, vec![lane, fifteen]);
        let one = f.const_u32(1);
        let sub = f.append(b, Op::ISub, Type::U32, vec![within, one]);
        let sub = f.append(b, Op::And, Type::U32, vec![sub, fifteen]);
        let row_shr = f.append(b, Op::Or, Type::U32, vec![row, sub]);
        // Lane + 16 (crosses into the other half for lanes 16..31).
        let sixteen = f.const_u32(16);
        let plus = f.append(b, Op::IAdd, Type::U32, vec![lane, sixteen]);
        let m63 = f.const_u32(63);
        let plus = f.append(b, Op::And, Type::U32, vec![plus, m63]);
        // A run-time source.
        let user = f.append(b, Op::UserData(0), Type::U32, vec![]);
        // ds_bpermute: the lane's half with a run-time index below 32.
        let hm = f.const_u32(!31);
        let base = f.append(b, Op::And, Type::U32, vec![lane, hm]);
        let m31 = f.const_u32(31);
        let two = f.const_u32(2);
        let idx = f.append(b, Op::LShr, Type::U32, vec![user, two]);
        let idx = f.append(b, Op::And, Type::U32, vec![idx, m31]);
        let bperm = f.append(b, Op::Or, Type::U32, vec![base, idx]);
        let bperm = f.append(b, Op::And, Type::U32, vec![bperm, m63]);
        // Without the half's bit the source is in the low half.
        let low = f.append(b, Op::And, Type::U32, vec![lane, m31]);
        let mut memo = HashMap::new();
        assert!(source_in_half(&f, bperm, &mut memo));
        assert!(!source_in_half(&f, low, &mut memo));
        assert!(source_in_half(&f, row_shr, &mut memo));
        assert!(!source_in_half(&f, plus, &mut memo));
        assert!(!source_in_half(&f, user, &mut memo));
        let c40 = f.const_u32(40);
        assert!(!source_in_half(&f, c40, &mut memo));
    }

    /// With several waves, a ballot under a branch on another ballot (an
    /// EXEC-dependent `if`) is under divergent control flow; a ballot
    /// before it, or under a branch on user data, is not.
    #[test]
    fn exchanges_under_wave_dependent_branches_are_found() {
        let mut f = Function::new(64, Stage::Compute, 0);
        let blocks: Vec<Block> = (0..7).map(|_| f.new_block()).collect();
        f.entry = blocks[0];
        f.order = blocks.clone();
        let link = |f: &mut Function, from: Block, term: Terminator| {
            for s in term.successors() {
                f.block_mut(s).preds.push(from);
            }
            f.block_mut(from).term = term;
        };
        let lane = f.append(blocks[0], Op::LaneId, Type::U32, vec![]);
        let user = f.append(blocks[0], Op::UserData(0), Type::U32, vec![]);
        let p = f.append(blocks[0], Op::ICmp(ICmp::ULt), Type::BOOL, vec![lane, user]);
        let first = f.append(blocks[0], Op::Ballot, Type::U32X2, vec![p]);
        let lo = f.append(blocks[0], Op::Extract(0), Type::U32, vec![first]);
        let zero = f.const_u32(0);
        let any = f.append(blocks[0], Op::ICmp(ICmp::Ne), Type::BOOL, vec![lo, zero]);
        f.block_mut(blocks[0]).merge = Merge::Selection { merge: blocks[2] };
        link(&mut f, blocks[0], Terminator::CondBranch { cond: any, then_block: blocks[1], else_block: blocks[2] });
        let inner = f.append(blocks[1], Op::Ballot, Type::U32X2, vec![p]);
        link(&mut f, blocks[1], Terminator::Branch(blocks[2]));
        let five = f.const_u32(5);
        let uniform = f.append(blocks[2], Op::ICmp(ICmp::Eq), Type::BOOL, vec![user, five]);
        f.block_mut(blocks[2]).merge = Merge::Selection { merge: blocks[4] };
        link(&mut f, blocks[2], Terminator::CondBranch { cond: uniform, then_block: blocks[3], else_block: blocks[4] });
        let fine = f.append(blocks[3], Op::Ballot, Type::U32X2, vec![p]);
        link(&mut f, blocks[3], Terminator::Branch(blocks[4]));
        link(&mut f, blocks[4], Terminator::Branch(blocks[5]));
        link(&mut f, blocks[5], Terminator::Branch(blocks[6]));
        f.block_mut(blocks[6]).term = Terminator::Return;
        let (div, region) = workgroup_divergence(&f);
        assert!(div[first.index()] && !div[uniform.index()]);
        assert_eq!(region, vec![false, true, false, false, false, false, false]);
        let _ = (inner, fine);
    }
}
