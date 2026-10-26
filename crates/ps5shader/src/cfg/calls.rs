// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Scalar subroutine calls: S_CALL_B64 and S_SWAPPC_B64 with a link
//! register, each paired with the S_SETPC_B64 that returns through the
//! link.
//!
//! A call region is `[target, return]` in program order. The call transfers
//! to the target; the return transfers to the instruction after the call.
//! Calls are accepted only when that model is exact:
//! - the link is an aligned ordinary SGPR pair (s0..s105);
//! - the target is static (S_CALL_B64's offset, or an S_SWAPPC_B64 whose
//!   address is built from S_GETPC_B64 plus constants) and is an
//!   instruction of the program;
//! - exactly one S_SETPC_B64 returns through the link (with one link shared
//!   by several calls: the first one at or after each call's target);
//! - nothing else in the region touches the link, and no instruction
//!   outside the regions reads a link while it holds a return address;
//! - regions nest or are disjoint, a call never runs inside its own region,
//!   and control enters a region only through its call and leaves it only
//!   through its return.

use super::graph::direct_setpc_target;
use crate::ir::branch;
use crate::ir_types::*;
use crate::opcode::Opcode;
use std::collections::HashMap;
use std::fmt;

/// One call and its return (instruction indices).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScalarCall {
    pub call: usize,
    pub target: usize,
    pub ret: usize,
    pub link: u32,
}

/// Why a call is not accepted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallError {
    pub pc: u32,
    pub reason: String,
}

impl fmt::Display for CallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "scalar call at {:#x}: {}", self.pc, self.reason)
    }
}

/// S_CALL_B64, or S_SWAPPC_B64 writing a link register (S_SWAPPC_B64 with
/// a NULL destination is a plain computed jump).
pub fn is_call(i: &Instruction) -> bool {
    match i.opcode {
        Opcode::SCallB64 => true,
        Opcode::SSwappcB64 => !branch::is_computed_jump(i),
        _ => false,
    }
}

/// Pc an S_CALL_B64 transfers to: the next pc plus simm16 dwords.
pub fn call_offset_target(i: &Instruction) -> Option<u32> {
    if i.opcode != Opcode::SCallB64 || i.words.is_empty() {
        return None;
    }
    let offset = (i.words[0] & 0xFFFF) as u16 as i16 as i64;
    u32::try_from(i.next_pc() as i64 + offset * 4).ok()
}

fn link_of(i: &Instruction) -> Option<u32> {
    i.destinations.first().filter(|d| d.kind == OperandKind::ScalarRegister).map(|d| d.value)
}

/// S_SETPC_B64 through `link`.
fn returns_through(i: &Instruction, link: u32) -> bool {
    i.opcode == Opcode::SSetpcB64 && i.sources.first().is_some_and(|s| s.kind == OperandKind::ScalarRegister && s.value == link)
}

/// Dwords a scalar register operand of `i` covers.
fn operand_width(i: &Instruction, wave64: bool) -> u32 {
    let name = i.opcode.name();
    if name.contains("64") {
        return 2;
    }
    if wave64 && matches!(i.encoding, Encoding::Vop2 | Encoding::Vop3 | Encoding::Vopc | Encoding::Vop3p) {
        return 2;
    }
    1
}

/// Scalar load destination width (dwords).
fn load_width(i: &Instruction) -> u32 {
    let name = i.opcode.name();
    for (suffix, n) in [("x16", 16), ("x8", 8), ("x4", 4), ("x3", 3), ("x2", 2)] {
        if name.to_ascii_lowercase().ends_with(suffix) {
            return n;
        }
    }
    1
}

fn overlaps(first: u32, count: u32, reg: u32) -> bool {
    reg >= first && reg - first < count
}

/// `i` writes SGPR `reg`.
pub(crate) fn writes(i: &Instruction, reg: u32, wave64: bool) -> bool {
    if is_call(i) || i.opcode == Opcode::SGetpcB64 {
        return link_of(i).is_some_and(|l| overlaps(l, 2, reg));
    }
    if i.opcode.is_computed_jump() {
        return false;
    }
    // A relative write can reach any register from its base on.
    if matches!(i.opcode, Opcode::SMovreldB32 | Opcode::SMovreldB64 | Opcode::SMovrelsd2B32) {
        return i.destinations.first().is_some_and(|d| d.kind == OperandKind::ScalarRegister && reg >= d.value);
    }
    let width = match i.encoding {
        // Scalar memory destinations list every register written.
        Encoding::Smem | Encoding::Smrd if i.destinations.len() > 1 => 1,
        Encoding::Smem | Encoding::Smrd => match (&i.control, i.opcode) {
            (_, Opcode::SMemtime | Opcode::SMemrealtime) => 2,
            (Some(Control::ScalarMemory(m)), _) if m.destination_count > 0 => m.destination_count,
            _ => load_width(i),
        },
        Encoding::Vop1 | Encoding::Vop2 | Encoding::Vop3 | Encoding::Vop3p | Encoding::Vopc => {
            if matches!(i.opcode, Opcode::VReadlaneB32 | Opcode::VReadfirstlaneB32) {
                1
            } else {
                operand_width(i, wave64)
            }
        }
        _ => operand_width(i, wave64),
    };
    i.destinations.iter().any(|d| d.kind == OperandKind::ScalarRegister && overlaps(d.value, width, reg))
        || i.control.as_ref().and_then(Control::scalar_destination).is_some_and(|d| overlaps(d, width, reg))
}

/// `i` writes all of SGPR `reg` without reading it.
fn overwrites(i: &Instruction, reg: u32, wave64: bool) -> bool {
    match i.opcode {
        Opcode::SCmovB32 | Opcode::SCmovB64 | Opcode::SBitset0B32 | Opcode::SBitset0B64 | Opcode::SBitset1B32 | Opcode::SBitset1B64
        | Opcode::SMovreldB32 | Opcode::SMovreldB64 => false,
        _ if i.encoding == Encoding::Sopk => i.opcode == Opcode::SMovkI32 && writes(i, reg, wave64),
        _ => writes(i, reg, wave64),
    }
}

/// `i` reads SGPR `reg`.
fn reads(i: &Instruction, reg: u32, wave64: bool) -> bool {
    // A relative read can reach any register.
    if matches!(i.opcode, Opcode::SMovrelsB32 | Opcode::SMovrelsB64 | Opcode::SMovrelsd2B32) {
        return true;
    }
    let width = operand_width(i, wave64);
    if i.sources.iter().any(|s| s.kind == OperandKind::ScalarRegister && overlaps(s.value, width.max(if i.opcode.is_computed_jump() { 2 } else { 1 }), reg)) {
        return true;
    }
    // Instructions that read the register they name as destination.
    let reads_destination = (i.encoding == Encoding::Sopk && i.opcode != Opcode::SMovkI32)
        || matches!(i.opcode, Opcode::SCmovB32 | Opcode::SCmovB64 | Opcode::SBitset0B32 | Opcode::SBitset0B64 | Opcode::SBitset1B32 | Opcode::SBitset1B64);
    if reads_destination && !is_call(i) && i.destinations.iter().any(|d| d.kind == OperandKind::ScalarRegister && overlaps(d.value, width, reg)) {
        return true;
    }
    match &i.control {
        Some(Control::ScalarMemory(m)) => {
            let base = i.sources.first().filter(|s| s.kind == OperandKind::ScalarRegister).map(|s| s.value);
            let base_width = if i.opcode.name().starts_with("SBuffer") { 4 } else { 2 };
            base.is_some_and(|b| overlaps(b, base_width, reg)) || m.dynamic_offset_register == Some(reg)
        }
        Some(Control::Image(image)) => overlaps(image.scalar_resource, 8, reg) || overlaps(image.scalar_sampler, 4, reg),
        Some(Control::BufferMemory(buffer)) => overlaps(buffer.scalar_resource, 4, reg),
        Some(Control::GlobalMemory(global)) => global.scalar_address < 255 && overlaps(global.scalar_address, 2, reg),
        _ => false,
    }
}

fn touches(i: &Instruction, link: u32, wave64: bool) -> bool {
    (0..2).any(|k| writes(i, link + k, wave64) || reads(i, link + k, wave64))
}

/// Instruction indices a computed or direct transfer at `index` goes to
/// (`None`: an unresolvable jump).
fn transfer_targets(p: &Program, index: usize, calls: &[ScalarCall], index_of: &HashMap<u32, usize>, offset_index: &HashMap<u64, usize>) -> Option<Vec<usize>> {
    let i = &p.instructions[index];
    let n = p.instructions.len();
    let next = index + 1;
    let mut out = Vec::new();
    if i.opcode.is_program_end() {
        return Some(out);
    }
    if let Some(call) = calls.iter().find(|c| c.call == index) {
        out.push(call.target);
    } else if let Some(call) = calls.iter().find(|c| c.ret == index) {
        out.push(call.call + 1);
    } else if branch::is_unconditional_branch(i) || branch::is_conditional(i) {
        if let Some(&t) = branch::branch_target(i).and_then(|t| index_of.get(&t)) {
            out.push(t);
        } else if branch::branch_target(i) == p.instructions.last().map(|l| l.next_pc()) {
            out.push(n);
        }
        if branch::is_conditional(i) {
            out.push(next);
        }
    } else if i.opcode.is_computed_jump() {
        let off = direct_setpc_target(p, index)?;
        out.push(offset_index.get(&off).copied().unwrap_or(n));
    } else {
        out.push(next);
    }
    Some(out)
}

fn err(p: &Program, index: usize, reason: impl Into<String>) -> CallError {
    CallError { pc: p.instructions[index].pc, reason: reason.into() }
}

/// The calls of `p` and their returns. `wave64` widens the lane-mask
/// destinations of vector instructions to SGPR pairs.
pub fn analyze(p: &Program, wave64: bool) -> Result<Vec<ScalarCall>, CallError> {
    let ins = &p.instructions;
    if !ins.iter().any(is_call) {
        return Ok(Vec::new());
    }
    let index_of: HashMap<u32, usize> = ins.iter().enumerate().map(|(k, i)| (i.pc, k)).collect();
    let offset_index: HashMap<u64, usize> = ins.iter().enumerate().map(|(k, i)| (i.program_offset(), k)).collect();
    let mut calls: Vec<ScalarCall> = Vec::new();
    for (k, i) in ins.iter().enumerate() {
        if !is_call(i) {
            continue;
        }
        let link = link_of(i).filter(|&l| l % 2 == 0 && l <= 104).ok_or_else(|| {
            err(p, k, "unsupported link register: the link must be an ordinary aligned scalar register pair")
        })?;
        let target = if i.opcode == Opcode::SCallB64 {
            let pc = call_offset_target(i);
            pc.and_then(|t| index_of.get(&t).copied()).ok_or_else(|| err(p, k, format!("s_call_b64 targets invalid instruction boundary {pc:#x?}")))?
        } else {
            let off = direct_setpc_target(p, k).ok_or_else(|| err(p, k, "computed/data-dependent s_swappc_b64 call target is not statically resolvable"))?;
            offset_index.get(&off).copied().ok_or_else(|| err(p, k, format!("s_swappc_b64 targets offset {off:#x}, not an instruction of the program")))?
        };
        calls.push(ScalarCall { call: k, target, ret: usize::MAX, link });
    }

    let shares = |calls: &[ScalarCall], c: &ScalarCall| calls.iter().filter(|o| o.link == c.link).count() > 1;
    let snapshot = calls.clone();
    for c in calls.iter_mut() {
        if shares(&snapshot, c) {
            c.ret = (c.target..ins.len())
                .find(|&k| returns_through(&ins[k], c.link))
                .ok_or_else(|| err(p, c.call, format!("no paired s_setpc_b64 return reads link register s[{}]", c.link)))?;
            continue;
        }
        let returns: Vec<usize> = (0..ins.len()).filter(|&k| k != c.call && returns_through(&ins[k], c.link)).collect();
        // With exactly one return after the target, the link may be reused
        // outside the region; otherwise it must not be touched anywhere.
        let region_end = (returns.len() == 1 && returns[0] >= c.target).then(|| returns[0]);
        let mut seen = 0;
        for (k, i) in ins.iter().enumerate() {
            if k == c.call {
                continue;
            }
            if returns_through(i, c.link) {
                seen += 1;
                if seen > 1 {
                    return Err(err(p, c.call, format!("multiple s_setpc_b64 returns read link register s[{}]", c.link)));
                }
                continue;
            }
            if region_end.is_some_and(|end| k < c.target || k > end) {
                continue;
            }
            if touches(i, c.link, wave64) {
                return Err(err(p, c.call, format!("link register s[{}] escapes the call/return model (used at {:#x})", c.link, i.pc)));
            }
        }
        if returns.is_empty() {
            return Err(err(p, c.call, format!("no paired s_setpc_b64 return reads link register s[{}]", c.link)));
        }
        if returns[0] < c.target {
            return Err(err(p, c.call, "return precedes the call target"));
        }
        c.ret = returns[0];
    }

    // Calls sharing a link: their regions are disjoint and nothing else in
    // them touches the link.
    for c in &calls {
        if !shares(&calls, c) {
            continue;
        }
        for o in calls.iter().filter(|o| o.link == c.link && o.call != c.call) {
            let call_inside = o.call >= c.target && o.call <= c.ret;
            let meet = o.target <= c.ret && c.target <= o.ret;
            if call_inside || meet {
                return Err(err(p, o.call, format!("link register s[{}] is shared with another call", c.link)));
            }
        }
        for k in c.target..c.ret {
            if calls.iter().any(|o| o.call == k && o.link == c.link) {
                continue;
            }
            if touches(&ins[k], c.link, wave64) {
                return Err(err(p, c.call, format!("link register s[{}] escapes the call/return model (used at {:#x})", c.link, ins[k].pc)));
            }
        }
    }

    // A return address never reaches a reader outside its call regions.
    let succ: Vec<Vec<usize>> = (0..ins.len())
        .map(|k| transfer_targets(p, k, &calls, &index_of, &offset_index).unwrap_or_default().into_iter().filter(|&s| s < ins.len()).collect())
        .collect();
    let mut links: Vec<u32> = calls.iter().map(|c| c.link).collect();
    links.sort_unstable();
    links.dedup();
    for &link in &links {
        let defines = |k: usize| calls.iter().any(|c| c.call == k && c.link == link);
        let mut holds = vec![0u8; ins.len()];
        let mut work: Vec<usize> = (0..ins.len()).filter(|&k| defines(k)).collect();
        let after = |k: usize, holds: &[u8]| -> u8 {
            if defines(k) {
                return 3;
            }
            let mut v = holds[k];
            if overwrites(&ins[k], link, wave64) {
                v &= 2;
            }
            if overwrites(&ins[k], link + 1, wave64) {
                v &= 1;
            }
            v
        };
        while let Some(k) = work.pop() {
            let v = after(k, &holds);
            for &s in &succ[k] {
                if holds[s] | v != holds[s] {
                    holds[s] |= v;
                    work.push(s);
                }
            }
        }
        for (k, i) in ins.iter().enumerate() {
            if holds[k] == 0 || defines(k) || i.encoding == Encoding::Sopc || calls.iter().any(|c| c.ret == k) {
                continue;
            }
            let inside = calls.iter().any(|c| c.link == link && k >= c.target && k <= c.ret);
            let reads_held = (holds[k] & 1 != 0 && reads(i, link, wave64)) || (holds[k] & 2 != 0 && reads(i, link + 1, wave64));
            if !inside && reads_held {
                let owner = calls.iter().find(|c| c.link == link).unwrap();
                return Err(err(p, owner.call, format!("link register s[{link}] escapes the call/return model (used at {:#x})", i.pc)));
            }
        }
    }

    // No recursion; regions nest or are disjoint.
    for (a, ca) in calls.iter().enumerate() {
        if ca.call >= ca.target && ca.call <= ca.ret {
            return Err(err(p, ca.call, "recursive scalar call: the call executes inside its own call region"));
        }
        for cb in &calls[a + 1..] {
            let disjoint = ca.ret < cb.target || cb.ret < ca.target;
            let a_holds_b = ca.target <= cb.target && cb.ret <= ca.ret;
            let b_holds_a = cb.target <= ca.target && ca.ret <= cb.ret;
            if !disjoint && !a_holds_b && !b_holds_a {
                return Err(err(p, ca.call, format!("recursive scalar calls at {:#x} and {:#x}: call regions interleave", ins[ca.call].pc, ins[cb.call].pc)));
            }
        }
    }

    // Control enters a region only through its call and leaves only
    // through its return.
    for k in 0..ins.len() {
        let targets = transfer_targets(p, k, &calls, &index_of, &offset_index)
            .ok_or_else(|| err(p, k, "dynamic s_setpc_b64 in a program with calls cannot be confined to a call region"))?;
        for c in &calls {
            let source_inside = k >= c.target && k <= c.ret;
            for &t in &targets {
                let target_inside = t >= c.target && t <= c.ret;
                if !source_inside && target_inside && k != c.call {
                    return Err(err(p, c.call, format!("control flow enters the call region from {:#x}", ins[k].pc)));
                }
                if source_inside && !target_inside && k != c.ret {
                    return Err(err(p, c.call, format!("control flow leaves the call region at {:#x}", ins[k].pc)));
                }
            }
        }
    }
    Ok(calls)
}

/// Pc each call and return transfers to, for the resource analysis' graph:
/// `(instruction pc, target pc)`.
pub fn transfer_pcs(p: &Program, calls: &[ScalarCall]) -> HashMap<u32, u32> {
    let ins = &p.instructions;
    let mut out = HashMap::new();
    for c in calls {
        out.insert(ins[c.call].pc, ins[c.target].pc);
        out.insert(ins[c.ret].pc, ins[c.call].next_pc());
    }
    out
}

/// Branch resolver that also follows calls and their returns.
#[derive(Clone, Debug, Default)]
pub struct CallResolver {
    transfers: HashMap<u32, u32>,
}

impl CallResolver {
    /// The resolver for `p`; a program whose calls are not accepted (its
    /// translation fails) gets the plain branch resolver.
    pub fn new(p: &Program, wave64: bool) -> CallResolver {
        let transfers = analyze(p, wave64).map(|calls| transfer_pcs(p, &calls)).unwrap_or_default();
        CallResolver { transfers }
    }
}

impl branch::BranchResolver for CallResolver {
    fn is_branch(&self, i: &Instruction) -> bool {
        self.transfers.contains_key(&i.pc) || branch::is_branch(i)
    }

    fn is_conditional(&self, i: &Instruction) -> bool {
        !self.transfers.contains_key(&i.pc) && branch::is_conditional(i)
    }

    fn branch_target(&self, i: &Instruction) -> Option<u32> {
        self.transfers.get(&i.pc).copied().or_else(|| branch::branch_target(i))
    }
}
