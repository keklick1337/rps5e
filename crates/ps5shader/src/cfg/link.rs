// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Linking subroutines that live outside the program: an S_SWAPPC_B64
//! whose target is a code address held in user data (directly, or copied
//! there by S_MOV), or loaded from a table of code addresses through a
//! buffer descriptor in user data.
//!
//! The driver reads each callee from guest memory up to its S_SETPC_B64
//! return and appends it to the program; the call becomes an S_CALL_B64 to
//! the copy, which [`super::calls`] then models as a static call. A table
//! call becomes a comparison of the loaded target against every table entry
//! (SCC kept), each match calling its copy; a target matching no entry
//! runs the fallback, which ends the wave (the call would run code that
//! was never linked).

use super::calls::writes;
use super::graph::{direct_setpc_target, resolves_jump_table};
use crate::ir::branch;
use crate::ir_types::*;
use crate::opcode::Opcode;
use crate::program::CodeSlice;
use std::collections::BTreeSet;

/// Where the program's user data lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallSite {
    pub user_data_base: u32,
    pub user_data_count: u32,
    pub wave_size: u32,
}

/// A call through a code address in user data `s[user_data_register:+1]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UserDataCall {
    pub pc: u32,
    pub link: u32,
    pub user_data_register: u32,
}

/// A call through a code address loaded from a table: the descriptor is
/// the 16 bytes at `pointer + descriptor_offset` (`pointer` in user data
/// `s[user_data_register:+1]`); entries are 64-bit addresses from
/// `entry_offset` every `entry_stride` bytes (0: one entry).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TableCall {
    pub pc: u32,
    pub link: u32,
    pub target_register: u32,
    pub user_data_register: u32,
    pub descriptor_offset: u32,
    pub entry_offset: u32,
    pub entry_stride: u32,
}

/// One callee of a table call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableCallee {
    pub target: u64,
    pub code: Vec<u32>,
}

fn scalar(o: Option<&Operand>) -> Option<u32> {
    o.filter(|o| o.kind == OperandKind::ScalarRegister).map(|o| o.value)
}

fn is_scalar(o: Option<&Operand>, reg: u32) -> bool {
    scalar(o) == Some(reg)
}

fn direct_branch(i: &Instruction) -> bool {
    branch::is_unconditional_branch(i) || branch::is_conditional(i)
}

fn label_of(i: &Instruction) -> Option<u32> {
    if direct_branch(i) {
        branch::branch_target(i)
    } else {
        super::calls::call_offset_target(i)
    }
}

fn branch_labels(p: &Program) -> BTreeSet<u32> {
    p.instructions.iter().filter_map(label_of).collect()
}

fn last_writer(p: &Program, end: usize, regs: &[u32], wave64: bool) -> Option<usize> {
    for k in (0..end).rev() {
        let i = &p.instructions[k];
        if direct_branch(i) || i.opcode == Opcode::SSetpcB64 || i.opcode == Opcode::SEndpgm {
            return None;
        }
        if regs.iter().any(|&r| writes(i, r, wave64)) {
            return Some(k);
        }
    }
    None
}

/// The one instruction writing any of `regs` that every path into
/// `reader` passes last: the walk goes back through fall-through, direct
/// branches and S_SETPC_B64 jumps to known targets until each path meets a
/// writer. `None` when two writers reach `reader`, a path reaches the
/// program's entry or a call target without one, or passes a call.
fn reaching_writer(p: &Program, reader: usize, regs: &[u32], wave64: bool) -> Option<usize> {
    reaching_writer_by(p, reader, |i| regs.iter().any(|&r| writes(i, r, wave64)))
}

/// [`reaching_writer`] for the instructions `writes_value` accepts.
fn reaching_writer_by(p: &Program, reader: usize, writes_value: impl Fn(&Instruction) -> bool) -> Option<usize> {
    use std::collections::{BTreeMap, HashMap};
    let ins = &p.instructions;
    let by_pc: HashMap<u32, usize> = ins.iter().enumerate().map(|(k, i)| (i.pc, k)).collect();
    let by_offset: HashMap<u64, usize> = ins.iter().enumerate().map(|(k, i)| (i.program_offset(), k)).collect();
    let mut branches: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut call_targets = BTreeSet::new();
    for (k, i) in ins.iter().enumerate() {
        if direct_branch(i) {
            if let Some(t) = branch::branch_target(i).and_then(|t| by_pc.get(&t)) {
                branches.entry(*t).or_default().push(k);
            }
        } else if i.opcode == Opcode::SSetpcB64 {
            if let Some(t) = direct_setpc_target(p, k).and_then(|t| by_offset.get(&t)) {
                branches.entry(*t).or_default().push(k);
            }
        } else if let Some(t) = super::calls::call_offset_target(i).and_then(|t| by_pc.get(&t)) {
            call_targets.insert(*t);
        }
    }
    let mut pending = Vec::new();
    let predecessors = |index: usize, pending: &mut Vec<usize>| -> bool {
        if index == 0 || call_targets.contains(&index) {
            return false;
        }
        let before = pending.len();
        let previous = &ins[index - 1];
        if !(branch::is_unconditional_branch(previous) || previous.opcode.is_program_end() || matches!(previous.opcode, Opcode::SSetpcB64 | Opcode::SCodeEnd)) {
            pending.push(index - 1);
        }
        if let Some(from) = branches.get(&index) {
            pending.extend_from_slice(from);
        }
        pending.len() != before
    };
    if !predecessors(reader, &mut pending) {
        return None;
    }
    let mut seen = vec![false; ins.len()];
    let mut found = None;
    while let Some(index) = pending.pop() {
        if std::mem::replace(&mut seen[index], true) {
            continue;
        }
        let i = &ins[index];
        if i.opcode == Opcode::SCallB64 || (i.opcode == Opcode::SSwappcB64 && direct_setpc_target(p, index).is_some()) {
            return None;
        }
        if writes_value(i) {
            if found.is_some_and(|f| f != index) {
                return None;
            }
            found = Some(index);
            continue;
        }
        if !predecessors(index, &mut pending) {
            return None;
        }
    }
    found
}

/// The constant lane operand of a V_READLANE_B32 / V_WRITELANE_B32 (the
/// hardware takes it modulo the wave size).
fn constant_lane(o: Option<&Operand>, wave64: bool) -> Option<u32> {
    let lane = match o? {
        Operand { kind: OperandKind::EncodedConstant, value } => crate::inline_constants::decode(*value)?,
        Operand { kind: OperandKind::LiteralConstant, value } => *value,
        _ => return None,
    };
    Some(lane & if wave64 { 63 } else { 31 })
}

/// `i` may write lane `lane` of VGPR `vgpr`: a lane write of that lane (or
/// of an unknown one), a relative write at or past a register below it, or
/// any other write of a destination that may cover it. Vector ALU results
/// name their first register (up to 2 wide), images their first data
/// register (up to 5 with TFE); other memory results name each register.
fn writes_lane(i: &Instruction, vgpr: u32, lane: u32, wave64: bool) -> bool {
    let vector = |o: &Operand| o.kind == OperandKind::VectorRegister;
    if i.opcode == Opcode::VWritelaneB32 {
        return i.destinations.first().is_some_and(|d| vector(d) && d.value == vgpr) && constant_lane(i.sources.get(1), wave64).is_none_or(|l| l == lane);
    }
    if matches!(i.opcode, Opcode::VMovreldB32 | Opcode::VMovrelsdB32 | Opcode::VMovrelsd2B32 | Opcode::VSwaprelB32 | Opcode::VSwapB32) {
        return i.destinations.iter().chain(&i.sources).any(|o| vector(o) && o.value <= vgpr);
    }
    let width = match i.encoding {
        Encoding::Vop1 | Encoding::Vop2 | Encoding::Vop3 | Encoding::Vop3p | Encoding::Vopc => 2,
        Encoding::Mimg => 5,
        _ => 1,
    };
    i.destinations.iter().any(|d| vector(d) && d.value <= vgpr && vgpr - d.value < width)
}

/// The scalar load whose registers `first..first + count` reach `reader`
/// through VGPR lanes: each register's last writer on every path is a
/// V_READLANE_B32 of a constant lane, that lane's last writer a
/// V_WRITELANE_B32 of an SGPR, and that SGPR's last writer the one load,
/// at the same position of its destinations. Returns the load and its
/// first destination register.
fn unspilled_load(p: &Program, reader: usize, first: u32, count: u32, wave64: bool) -> Option<(usize, u32)> {
    let mut found = None;
    for j in 0..count {
        let read = reaching_writer(p, reader, &[first + j], wave64)?;
        let r = &p.instructions[read];
        let vgpr = r.sources.first().filter(|o| o.kind == OperandKind::VectorRegister)?.value;
        let lane = constant_lane(r.sources.get(1), wave64)?;
        if r.opcode != Opcode::VReadlaneB32 || !is_scalar(r.destinations.first(), first + j) {
            return None;
        }
        let write = reaching_writer_by(p, read, |i| writes_lane(i, vgpr, lane, wave64))?;
        let w = &p.instructions[write];
        let spilled = scalar(w.sources.first())?;
        if w.opcode != Opcode::VWritelaneB32 || constant_lane(w.sources.get(1), wave64) != Some(lane) {
            return None;
        }
        let load = reaching_writer(p, write, &[spilled], wave64)?;
        let base = scalar(p.instructions[load].destinations.first())?;
        if spilled.checked_sub(base) != Some(j) || found.is_some_and(|f| f != (load, base)) {
            return None;
        }
        found = Some((load, base));
    }
    found
}

/// No instruction writes `reg` before `reader`, and no write after it can
/// reach it again through a backward branch or call.
fn unchanged_until(p: &Program, reg: u32, reader: usize, wave64: bool) -> bool {
    let reader_pc = p.instructions[reader].pc;
    for (k, i) in p.instructions.iter().enumerate() {
        if !writes(i, reg, wave64) {
            continue;
        }
        if k <= reader {
            return false;
        }
        if p.instructions[k..].iter().any(|l| label_of(l).is_some_and(|t| t <= reader_pc)) {
            return false;
        }
    }
    true
}

/// The scalar buffer load (and its dword) `reg` holds before `before`,
/// through up to four S_MOV copies.
fn loaded_dword(p: &Program, mut reg: u32, mut before: usize, wave64: bool) -> Option<(usize, u32)> {
    for _ in 0..4 {
        let writer = last_writer(p, before, &[reg], wave64)?;
        let i = &p.instructions[writer];
        let dst = scalar(i.destinations.first());
        let src = scalar(i.sources.first());
        match (i.opcode, dst, src) {
            (Opcode::SMovB32, Some(d), Some(s)) if d == reg => reg = s,
            (Opcode::SMovB64, Some(d), Some(s)) if reg >= d && reg - d < 2 => reg = s + (reg - d),
            (op, Some(d), _) if op.name().starts_with("SBufferLoadDword") => {
                let count = match &i.control {
                    Some(Control::ScalarMemory(m)) => m.destination_count,
                    _ => 0,
                };
                return (reg >= d && reg - d < count).then_some((writer, reg - d));
            }
            _ => return None,
        }
        before = writer;
    }
    None
}

fn same_block(p: &Program, labels: &BTreeSet<u32>, first: usize, last: usize) -> bool {
    let (a, b) = (p.instructions[first].pc, p.instructions[last].pc);
    a >= b || labels.range(a + 1..=b).next().is_none()
}

/// S_SWAPPC_B64 calls through a code address in user data.
pub fn find_user_data_calls(p: &Program, site: CallSite) -> Vec<UserDataCall> {
    let wave64 = site.wave_size != 32;
    let labels = branch_labels(p);
    let user_data = |r: u32| r >= site.user_data_base && r - site.user_data_base < site.user_data_count;
    let unchanged = |r: u32, reader: usize| unchanged_until(p, r, reader, wave64);
    let mut out = Vec::new();
    for (k, i) in p.instructions.iter().enumerate() {
        if i.opcode != Opcode::SSwappcB64 || direct_setpc_target(p, k).is_some() {
            continue;
        }
        let (Some(link), Some(source)) = (scalar(i.destinations.first()), scalar(i.sources.first())) else { continue };
        if source % 2 != 0 || link == NULL_SOURCE {
            continue;
        }
        let pair = if source != link && user_data(source) && user_data(source + 1) && unchanged(source, k) && unchanged(source + 1, k) {
            source
        } else {
            // The copy in the call's block, or the one copy every path into
            // the call passes last (made before a branch).
            let writer = |reg: u32| {
                last_writer(p, k, &[reg], wave64).filter(|&w| same_block(p, &labels, w, k)).or_else(|| reaching_writer(p, k, &[reg], wave64))
            };
            let (Some(low), Some(high)) = (writer(source), writer(source + 1)) else { continue };
            let (lw, hw) = (&p.instructions[low], &p.instructions[high]);
            let mut from = None;
            if low == high && lw.opcode == Opcode::SMovB64 && is_scalar(lw.destinations.first(), source) {
                from = scalar(lw.sources.first());
            } else if lw.opcode == Opcode::SMovB32
                && hw.opcode == Opcode::SMovB32
                && is_scalar(lw.destinations.first(), source)
                && is_scalar(hw.destinations.first(), source + 1)
            {
                from = scalar(lw.sources.first()).filter(|&f| is_scalar(hw.sources.first(), f + 1));
            }
            let Some(from) = from else { continue };
            if from % 2 != 0 || !user_data(from) || !user_data(from + 1) || !unchanged(from, low) || !unchanged(from + 1, high) {
                continue;
            }
            from
        };
        out.push(UserDataCall { pc: i.pc, link, user_data_register: pair });
    }
    out
}

/// S_SWAPPC_B64 calls through a code address loaded from a table.
pub fn find_table_calls(p: &Program, site: CallSite) -> Vec<TableCall> {
    let wave64 = site.wave_size != 32;
    let labels = branch_labels(p);
    let user_data = |r: u32| r >= site.user_data_base && r - site.user_data_base < site.user_data_count;
    let mut out = Vec::new();
    for (k, i) in p.instructions.iter().enumerate() {
        if i.opcode != Opcode::SSwappcB64 || direct_setpc_target(p, k).is_some() {
            continue;
        }
        let (Some(link), Some(target)) = (scalar(i.destinations.first()), scalar(i.sources.first())) else { continue };
        if link % 2 != 0 || link > 104 || target % 2 != 0 || target > 104 {
            continue;
        }
        let (Some((low_load, low_dword)), Some((high_load, high_dword))) = (loaded_dword(p, target, k, wave64), loaded_dword(p, target + 1, k, wave64)) else {
            continue;
        };
        if low_load != high_load || high_dword != low_dword + 1 || !same_block(p, &labels, low_load, k) {
            continue;
        }
        let load = &p.instructions[low_load];
        let Some(Control::ScalarMemory(lm)) = &load.control else { continue };
        let Some(descriptor) = scalar(load.sources.first()) else { continue };
        let load_offset = lm.immediate_offset_bytes;
        if descriptor % 4 != 0 || !(0..0x10_0000).contains(&load_offset) || load_offset & 3 != 0 {
            continue;
        }
        // The descriptor load in the call's block, or the one load every
        // path into the entry load passes (hoisted out of a loop).
        let descriptor_regs = [descriptor, descriptor + 1, descriptor + 2, descriptor + 3];
        let loaded = last_writer(p, low_load, &descriptor_regs, wave64)
            .filter(|&d| same_block(p, &labels, d, k))
            .or_else(|| reaching_writer(p, low_load, &descriptor_regs, wave64))
            .filter(|&d| p.instructions[d].opcode == Opcode::SLoadDwordx4)
            .map(|d| (d, descriptor));
        // Or the descriptor was spilled to VGPR lanes and read back.
        let Some((descriptor_load, loaded_into)) = loaded.or_else(|| unspilled_load(p, low_load, descriptor, 4, wave64)) else {
            continue;
        };
        let table = &p.instructions[descriptor_load];
        let Some(Control::ScalarMemory(tm)) = &table.control else { continue };
        let Some(pointer) = scalar(table.sources.first()) else { continue };
        let table_offset = tm.immediate_offset_bytes;
        if table.opcode != Opcode::SLoadDwordx4
            || !is_scalar(table.destinations.first(), loaded_into)
            || tm.dynamic_offset_register.is_some()
            || !(0..0x10_0000).contains(&table_offset)
            || table_offset & 3 != 0
            || !user_data(pointer)
            || !user_data(pointer + 1)
            || !unchanged_until(p, pointer, descriptor_load, wave64)
            || !unchanged_until(p, pointer + 1, descriptor_load, wave64)
        {
            continue;
        }
        let mut stride = 0;
        if let Some(offset) = lm.dynamic_offset_register {
            let Some(shift_index) = last_writer(p, low_load, &[offset], wave64) else { continue };
            if !same_block(p, &labels, shift_index, k) {
                continue;
            }
            let shifted = &p.instructions[shift_index];
            let shift = shifted.sources.get(1).and_then(|o| match o.kind {
                OperandKind::LiteralConstant => Some(o.value),
                OperandKind::EncodedConstant => crate::inline_constants::decode(o.value),
                _ => None,
            });
            match shift {
                Some(s) if shifted.opcode == Opcode::SLshlB32 && is_scalar(shifted.destinations.first(), offset) && (2..=24).contains(&s) => stride = 1 << s,
                _ => continue,
            }
        }
        out.push(TableCall {
            pc: i.pc,
            link,
            target_register: target,
            user_data_register: pointer,
            descriptor_offset: table_offset as u32,
            entry_offset: load_offset as u32 + low_dword * 4,
            entry_stride: stride,
        });
    }
    out
}

/// Words of a user-data callee up to and including its return through
/// `link`; `Ok(None)` when `callee` ends before the return.
pub fn user_data_callee_length(callee: &[u32], link: u32) -> Result<Option<usize>, String> {
    let mut furthest = 0usize;
    let mut word = 0usize;
    while word < callee.len() {
        let pc = (word * 4) as u32;
        let i = match crate::decode::decode_instruction(pc, &callee[word..]) {
            Ok(i) => i,
            Err(e) => {
                if callee.len() - word < 16 {
                    return Ok(None);
                }
                return Err(format!("callee does not decode at offset {pc:#x}: {e:?}"));
            }
        };
        word += i.words.len();
        let at = format!(" at callee offset {pc:#x}");
        if i.opcode == Opcode::SGetpcB64 {
            return Err(format!("user-data callee reads its program counter{at}, which a linked copy does not keep"));
        }
        if super::calls::is_call(&i) {
            return Err(format!("user-data callee makes a call{at}"));
        }
        if i.opcode == Opcode::SEndpgm || i.opcode == Opcode::SCodeEnd {
            return Err(format!("user-data callee ends the program{at} before it returns"));
        }
        if direct_branch(&i) {
            let off = (i.words[0] & 0xFFFF) as u16 as i16 as i64;
            let target = i.next_pc() as i64 + off * 4;
            if target < 0 {
                return Err(format!("user-data callee branches before its start{at}"));
            }
            furthest = furthest.max(target as usize / 4);
        }
        // The return: S_SETPC_B64, or S_SWAPPC_B64 into the null register
        // (a jump that keeps no return address).
        if !branch::is_computed_jump(&i) {
            continue;
        }
        if !is_scalar(i.sources.first(), link) {
            return Err(format!("user-data callee jumps through another register pair than its link s[{link}]{at}"));
        }
        if furthest >= word {
            return Err(format!("user-data callee returns{at} before code it branches to"));
        }
        return Ok(Some(word));
    }
    Ok(None)
}

/// Linked code and the word offsets of the table calls' fallbacks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkedCalls {
    pub code: Vec<u32>,
    pub fallbacks: Vec<u32>,
}

fn branch_word(opcode: u32, from: usize, to: usize, what: &str) -> Result<u32, String> {
    let offset = to as i64 - from as i64 - 1;
    if !(-0x8000..=0x7fff).contains(&offset) {
        return Err(format!("{what} lands out of the 16-bit branch range"));
    }
    Ok(opcode | (offset as u32 & 0xffff))
}

/// Append the callees to `code` (the program's words from its address) and
/// redirect the calls to them.
pub fn link_calls(
    p: &Program,
    code: &[u32],
    user_data_calls: &[UserDataCall],
    user_data_callees: &[Vec<u32>],
    table_calls: &[TableCall],
    table_callees: &[Vec<TableCallee>],
) -> Result<LinkedCalls, String> {
    if user_data_calls.len() != user_data_callees.len() || table_calls.len() != table_callees.len() {
        return Err("calls and callees differ in number".into());
    }
    let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    let source = CodeSlice::new(p.address, &bytes);
    for (k, i) in p.instructions.iter().enumerate() {
        if branch::is_computed_jump(i) && resolves_jump_table(p, k, &source) {
            return Err(format!(
                "a program that jumps through a table in its code (s_setpc_b64 at {:#x}) cannot have callees linked over the words after its last instruction",
                i.pc
            ));
        }
    }
    let end = p.instructions.last().map_or(0, |l| l.next_pc() as usize / 4);
    if end > code.len() {
        return Err("program code is shorter than its instructions".into());
    }
    let mut linked: Vec<u32> = code[..end].to_vec();
    let mut fallbacks = Vec::new();
    let call_word = |pc: u32, kind: &str| -> Result<usize, String> {
        match p.instructions.iter().find(|i| i.pc == pc) {
            Some(i) if i.opcode == Opcode::SSwappcB64 && i.words.len() == 1 => Ok(pc as usize / 4),
            _ => Err(format!("{kind} call does not name an s_swappc_b64")),
        }
    };
    for (call, callee) in user_data_calls.iter().zip(user_data_callees) {
        let word = call_word(call.pc, "user-data")?;
        let offset = linked.len() as i64 - word as i64 - 1;
        if offset > 0x7fff {
            return Err(format!("user-data callee of the call at {:#x} lands out of s_call_b64 range", call.pc));
        }
        linked[word] = 0xbb00_0000 | (call.link << 16) | offset as u32;
        linked.extend_from_slice(callee);
    }
    // The dispatch code of every table call comes first and the callees
    // after all of it, the tables with the least callee code first: the
    // branches between a call and its dispatch stay short, and an
    // S_CALL_B64 reaches as far as possible into the callee code.
    let mut call_sites = Vec::new();
    for (table, callees) in table_calls.iter().zip(table_callees) {
        let word = call_word(table.pc, "table")?;
        let callee_words: usize = callees.iter().map(|c| c.code.len()).sum();
        let at = format!("the table callees of the call at {:#x} ({} callees, {callee_words} words)", table.pc, callees.len());
        let count = callees.len();
        let dispatch = linked.len();
        let zero = dispatch + 1;
        let one = zero + count * 7 + 2;
        let calls = one + count * 8 + 2;
        let fault = calls + count * 2;
        linked[word] = branch_word(0xbf82_0000, word, dispatch, &at)?;
        let branch = |linked: &mut Vec<u32>, opcode: u32, target: usize| -> Result<(), String> {
            let w = branch_word(opcode, linked.len(), target, &at)?;
            linked.push(w);
            Ok(())
        };
        branch(&mut linked, 0xbf85_0000, one)?;
        for scc_one in [false, true] {
            for (k, callee) in callees.iter().enumerate() {
                let next = linked.len() + if scc_one { 8 } else { 7 };
                linked.extend_from_slice(&[0xbf07_ff00 | table.target_register, callee.target as u32]);
                branch(&mut linked, 0xbf85_0000, next)?;
                linked.extend_from_slice(&[0xbf07_ff00 | (table.target_register + 1), (callee.target >> 32) as u32]);
                branch(&mut linked, 0xbf85_0000, next)?;
                if scc_one {
                    linked.push(0xbf06_8080);
                }
                branch(&mut linked, 0xbf82_0000, calls + k * 2)?;
            }
            linked.push(if scc_one { 0xbf06_8080 } else { 0xbf07_8080 });
            branch(&mut linked, 0xbf82_0000, fault)?;
        }
        for _ in callees {
            linked.push(0);
            branch(&mut linked, 0xbf82_0000, word + 1)?;
        }
        fallbacks.push((linked.len() * 4) as u32);
        // S_ENDPGM: an unmatched target stops the wave.
        linked.push(0xbf81_0000);
        branch(&mut linked, 0xbf82_0000, word + 1)?;
        call_sites.push((callee_words, table, callees, calls, at));
    }
    call_sites.sort_by_key(|site| site.0);
    for (_, table, callees, calls, at) in call_sites {
        for (k, callee) in callees.iter().enumerate() {
            let site = calls + k * 2;
            let offset = linked.len() as i64 - site as i64 - 1;
            if offset > 0x7fff {
                return Err(format!("{at} land out of s_call_b64 range"));
            }
            linked[site] = 0xbb00_0000 | (table.link << 16) | offset as u32;
            linked.extend_from_slice(&callee.code);
        }
    }
    linked.push(0xbf9f_0000);
    Ok(LinkedCalls { code: linked, fallbacks })
}

/// S_CODE_END.
pub const S_CODE_END: u32 = 0xbf9f_0000;

/// A program ending in S_SETPC_B64 through a code address in user data
/// `s[user_data_register:+1]` (the high half may be a constant the program
/// moves there).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UserDataTail {
    pub pc: u32,
    pub user_data_register: u32,
    pub constant_high: Option<u32>,
}

/// The tail of `p` when its last instruction is S_SETPC_B64 through an
/// unchanged user-data pair, followed in memory by `word_after` =
/// S_CODE_END, and nothing before branches, calls or ends the program.
pub fn find_user_data_tail(p: &Program, user_data_base: u32, user_data_count: u32, word_after: u32) -> Option<UserDataTail> {
    let tail = p.instructions.last()?;
    if word_after != S_CODE_END || tail.opcode != Opcode::SSetpcB64 {
        return None;
    }
    let reg = scalar(tail.sources.first())?;
    if reg % 2 != 0 || reg < user_data_base || reg - user_data_base >= user_data_count {
        return None;
    }
    let mut high_unchanged = true;
    let mut constant_high = None;
    for i in &p.instructions[..p.instructions.len() - 1] {
        if direct_branch(i) || i.opcode.is_computed_jump() || super::calls::is_call(i) || i.opcode.is_program_end() || i.opcode == Opcode::SCodeEnd {
            return None;
        }
        if writes(i, reg, true) {
            return None;
        }
        if writes(i, reg + 1, true) {
            high_unchanged = false;
            constant_high = None;
            if i.opcode == Opcode::SMovB32 && is_scalar(i.destinations.first(), reg + 1) {
                constant_high = i.sources.first().and_then(|o| match o.kind {
                    OperandKind::LiteralConstant => Some(o.value),
                    OperandKind::EncodedConstant => crate::inline_constants::decode(o.value),
                    _ => None,
                });
            }
        }
    }
    if constant_high.is_none() && (!high_unchanged || reg - user_data_base + 1 >= user_data_count) {
        return None;
    }
    Some(UserDataTail { pc: tail.pc, user_data_register: reg, constant_high })
}

/// The program's code up to its tail, an S_BRANCH to the next word in
/// place of the tail, then the continuation's code.
pub fn link_user_data_tail(p: &Program, code: &[u32], tail: UserDataTail, continuation: &Program, continuation_code: &[u32]) -> Result<Vec<u32>, String> {
    let again = find_user_data_tail(p, tail.user_data_register, 2, S_CODE_END);
    if again.map(|t| (t.pc, t.constant_high)) != Some((tail.pc, tail.constant_high)) {
        return Err("user-data tail is not a linear jump through a proven user-data address".into());
    }
    if continuation.instructions.iter().any(|i| i.opcode == Opcode::SGetpcB64) {
        return Err("user-data tail continuation reads its program counter, which a linked copy does not keep".into());
    }
    let words = tail.pc as usize / 4;
    if words > code.len() {
        return Err("program code is shorter than its instructions".into());
    }
    let end = continuation.instructions.last().map_or(0, |l| l.next_pc() as usize / 4).min(continuation_code.len());
    let mut linked = code[..words].to_vec();
    linked.push(0xbf82_0000);
    linked.extend_from_slice(&continuation_code[..end]);
    Ok(linked)
}

/// Words of guest code a linked program copied (for revalidation).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopiedCode {
    pub address: u64,
    pub words: Vec<u32>,
}

/// A program with its callees linked in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkedProgram {
    pub linked: LinkedCalls,
    /// The callees as read from guest memory.
    pub callees: Vec<CopiedCode>,
}

/// Most words a callee is searched for its return.
pub const CALLEE_LIMIT_WORDS: usize = 16384;
/// Largest call table searched for callees.
pub const CALL_TABLE_LIMIT_BYTES: u64 = 64 << 20;
/// Most distinct callees one table call links.
pub const CALL_TABLE_LIMIT_CALLEES: usize = 64;

/// Reads guest memory: fills the buffer from the address, false when
/// unreadable.
pub type GuestReader<'a> = &'a dyn Fn(u64, &mut [u8]) -> bool;

/// The callee at `address`, read page by page until its return.
pub fn read_callee(address: u64, link: u32, read: GuestReader<'_>) -> Result<Vec<u32>, String> {
    const PAGE: u64 = 4096;
    if address % 4 != 0 {
        return Err(format!("its callee address {address:#x} is not dword aligned"));
    }
    let mut words: Vec<u32> = Vec::new();
    while words.len() < CALLEE_LIMIT_WORDS {
        let next = address + words.len() as u64 * 4;
        let bytes = (PAGE - next % PAGE) as usize;
        let mut buf = vec![0u8; bytes];
        if !read(next, &mut buf) {
            return Err(format!("its callee at {address:#x} is not readable at {next:#x}"));
        }
        words.extend(buf.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])));
        if let Some(length) = user_data_callee_length(&words, link)? {
            words.truncate(length);
            // The copy returns through S_SETPC_B64, the form the call
            // model pairs with its call; a returning S_SWAPPC_B64 into
            // null jumps the same way.
            if let Some(ret) = words.last_mut().filter(|w| **w & 0xFFFF_FF00 == 0xBEFD_2100) {
                *ret = 0xBE80_2000 | (*ret & 0xFF);
            }
            return Ok(words);
        }
    }
    Err(format!("its callee at {address:#x} does not return within {} bytes", CALLEE_LIMIT_WORDS * 4))
}

fn read_bytes(address: u64, bytes: u64, what: &str, read: GuestReader<'_>) -> Result<Vec<u8>, String> {
    let mut data = vec![0u8; bytes as usize];
    if address.checked_add(bytes).is_none() || !read(address, &mut data) {
        return Err(format!("its {what} at {address:#x} + {bytes:#x} is not readable"));
    }
    Ok(data)
}

fn user_pair(user_data: &[u32], site: CallSite, register: u32) -> Result<u64, String> {
    let slot = (register - site.user_data_base) as usize;
    match (user_data.get(slot), user_data.get(slot + 1)) {
        (Some(&lo), Some(&hi)) => Ok(lo as u64 | (hi as u64) << 32),
        _ => Err(format!("user data s[{register}:{}] is not provided", register + 1)),
    }
}

/// The distinct non-zero targets of a table call.
fn table_targets(call: &TableCall, site: CallSite, user_data: &[u32], read: GuestReader<'_>) -> Result<Vec<u64>, String> {
    let pointer = user_pair(user_data, site, call.user_data_register)?;
    let d = read_bytes(pointer.wrapping_add(call.descriptor_offset as u64), 16, "call table descriptor", read)?;
    let word = |k: usize| u32::from_le_bytes([d[k * 4], d[k * 4 + 1], d[k * 4 + 2], d[k * 4 + 3]]);
    let base = (word(0) as u64 | (word(1) as u64) << 32) & 0xFFFF_FFFF_FFFF;
    let stride = (word(1) >> 16) & 0x3FFF;
    let records = word(2) as u64;
    let bytes = if stride == 0 { records } else { stride as u64 * records };
    if bytes > CALL_TABLE_LIMIT_BYTES {
        return Err(format!("its call table of {bytes} bytes is larger than the {CALL_TABLE_LIMIT_BYTES} bytes that are searched for callees"));
    }
    let table = if bytes == 0 { Vec::new() } else { read_bytes(base, bytes, "call table", read)? };
    let mut targets = BTreeSet::new();
    let mut offset = call.entry_offset as u64;
    while offset + 8 <= bytes {
        let o = offset as usize;
        let target = u64::from_le_bytes(table[o..o + 8].try_into().unwrap());
        if target != 0 {
            targets.insert(target);
        }
        if call.entry_stride == 0 {
            break;
        }
        offset += call.entry_stride as u64;
    }
    if targets.len() > CALL_TABLE_LIMIT_CALLEES {
        return Err(format!("its call table holds {} distinct targets; at most {CALL_TABLE_LIMIT_CALLEES} are linked", targets.len()));
    }
    Ok(targets.into_iter().collect())
}

/// Link the user-data and table calls of a compute program (`code`: its
/// words from its address) with the callees `user_data` points to now.
/// `Ok(None)`: no such calls.
pub fn link_program(p: &Program, code: &[u32], site: CallSite, user_data: &[u32], read: GuestReader<'_>) -> Result<Option<LinkedProgram>, String> {
    if !p.instructions.iter().any(|i| i.opcode == Opcode::SSwappcB64) {
        return Ok(None);
    }
    let calls = find_user_data_calls(p, site);
    let tables = find_table_calls(p, site);
    if calls.is_empty() && tables.is_empty() {
        return Ok(None);
    }
    let mut copied = Vec::new();
    let mut callees = Vec::new();
    for call in &calls {
        let address = user_pair(user_data, site, call.user_data_register)?;
        let words = read_callee(address, call.link, read)
            .map_err(|e| format!("the call at {:#x} through user data s[{}:{}]: {e}", call.pc, call.user_data_register, call.user_data_register + 1))?;
        copied.push(CopiedCode { address, words: words.clone() });
        callees.push(words);
    }
    let mut table_callees = Vec::new();
    for call in &tables {
        let at = |e: String| format!("the call at {:#x} through a table: {e}", call.pc);
        let mut list = Vec::new();
        for target in table_targets(call, site, user_data, read).map_err(at)? {
            let words = read_callee(target, call.link, read).map_err(at)?;
            copied.push(CopiedCode { address: target, words: words.clone() });
            list.push(TableCallee { target, code: words });
        }
        table_callees.push(list);
    }
    let linked = link_calls(p, code, &calls, &callees, &tables, &table_callees)?;
    Ok(Some(LinkedProgram { linked, callees: copied }))
}
