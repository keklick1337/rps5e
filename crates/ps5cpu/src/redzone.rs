// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Guest leaf functions that keep data in the System V red zone.
//!
//! The System V ABI lets a function use the 128 bytes below RSP without
//! moving RSP, and compilers do so in functions that make no call. Linux
//! skips that area when it builds a signal frame. Windows (and Wine) build
//! the exception record and `CONTEXT` directly below the RSP of the
//! faulting thread, so every exception the emulator resolves in guest code
//! (a write to a page the GPU memory tracker watches, an emulated
//! instruction, a thread-pointer trap) overwrites whatever such a function
//! kept there.
//!
//! On Windows every function of a guest module that addresses memory below
//! RSP is therefore moved into a copy that runs with RSP lowered by
//! [`SHIFT`]; the original entry jumps to the copy. The copy keeps every
//! memory address of the original: an explicit `rsp`-based operand gets
//! [`SHIFT`] added, `push`/`pop` store and load at the original slots, and
//! RSP is raised back before every exit (`ret`, a jump out of the function,
//! an indirect jump). So the red zone is above the RSP the host sees, and
//! an exception frame lands below it.
//!
//! Function bounds come from the module's `.eh_frame` (through
//! `PT_GNU_EH_FRAME`). A function is left as it is (and stays exposed)
//! when the copy could not keep its behaviour exactly: it calls, uses RSP
//! in a way the rewrite does not cover, addresses the thread pointer, has
//! a branch into the bytes the entry jump replaces, or does not decode.
//! An indirect jump into the original body (a jump table) is still
//! correct: RSP is raised before it, and the original runs unmodified.

use iced_x86::{BlockEncoder, BlockEncoderOptions, Code, Decoder, DecoderOptions, FlowControl, Instruction, InstructionBlock, MemoryOperand, OpKind, Register};

/// How far a copy lowers RSP: the red zone.
pub const SHIFT: i64 = 128;
/// Bytes of the jump that replaces the start of a moved function.
pub const ENTRY_JUMP: usize = 5;
/// Labels of the exit stubs of conditional branches out of a function
/// (non-canonical addresses, so no real branch target matches them).
const STUB_LABEL: u64 = 0x8000_0000_0000_0000;

/// Why a function was not moved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Skip {
    /// It never addresses memory below RSP.
    NoRedZone,
    /// It does not decode as one stream of valid instructions.
    Decode,
    /// It calls (no red zone is live across a call), traps or uses an
    /// instruction with implicit stack effects the copy does not model.
    Unsupported,
    /// It addresses memory through FS or GS (rewritten elsewhere).
    Segment,
    /// It reads or writes RSP in a form the copy does not translate.
    RspUse,
    /// A branch lands inside an instruction, inside the entry jump, or
    /// leaves the function in a form that has no exit stub.
    Branch,
    /// It is shorter than the entry jump.
    TooShort,
    /// The copy does not encode at its address (out of reach).
    Encode,
}

/// Counts of one rewrite pass.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counts {
    pub functions: usize,
    pub moved: usize,
    pub kept: usize,
}

fn displ_size(d: i64) -> u32 {
    if d == 0 {
        0
    } else if i8::try_from(d).is_ok() {
        1
    } else {
        8
    }
}

fn has_memory(insn: &Instruction) -> bool {
    (0..insn.op_count()).any(|i| insn.op_kind(i) == OpKind::Memory)
}

/// Whether `insn` reads or writes RSP as a register operand.
fn rsp_operand(insn: &Instruction) -> bool {
    (0..insn.op_count()).any(|i| insn.op_kind(i) == OpKind::Register && insn.op_register(i).full_register() == Register::RSP)
}

/// Whether `insn` addresses memory below RSP (or below RBP that holds
/// RSP; checked by the caller), i.e. uses the red zone.
fn below(insn: &Instruction, base: Register) -> bool {
    has_memory(insn) && insn.memory_base() == base && (insn.memory_displacement64() as i64) < 0
}

fn add_displ(insn: &mut Instruction, delta: i64) -> Result<(), Skip> {
    let d = (insn.memory_displacement64() as i64).checked_add(delta).ok_or(Skip::Encode)?;
    if i32::try_from(d).is_err() {
        return Err(Skip::Encode);
    }
    insn.set_memory_displacement64(d as u64);
    insn.set_memory_displ_size(displ_size(d));
    Ok(())
}

fn lea(dst: Register, base: Register, displ: i64) -> Instruction {
    Instruction::with2(Code::Lea_r64_m, dst, MemoryOperand::with_base_displ(base, displ)).expect("lea r64, m")
}

fn at_ip(mut insn: Instruction, ip: u64) -> Instruction {
    insn.set_ip(ip);
    insn
}

fn is_mov_rbp_rsp(insn: &Instruction) -> bool {
    matches!(insn.code(), Code::Mov_r64_rm64 | Code::Mov_rm64_r64)
        && insn.op0_kind() == OpKind::Register
        && insn.op1_kind() == OpKind::Register
        && insn.op0_register() == Register::RBP
        && insn.op1_register() == Register::RSP
}

/// Whether the function `insns` addresses memory below RSP: through RSP
/// with a negative displacement, or through a frame pointer below the
/// frame its prologue allocated. Only a function without calls can keep
/// data there (a call writes below RSP).
pub fn uses_red_zone(insns: &[Instruction]) -> bool {
    if insns.iter().any(|i| matches!(i.flow_control(), FlowControl::Call | FlowControl::IndirectCall)) {
        return false;
    }
    // How far RSP is below its entry value along the straight-line
    // prologue (up to the first branch), and RBP once it copies RSP.
    let mut depth = 0i64;
    let mut prologue = true;
    let mut rbp: Option<i64> = None;
    for insn in insns {
        if below(insn, Register::RSP) {
            return true;
        }
        if let Some(r) = rbp {
            if below(insn, Register::RBP) && -(insn.memory_displacement64() as i64) > depth - r {
                return true;
            }
        }
        if is_mov_rbp_rsp(insn) {
            rbp = Some(depth);
        }
        if !prologue {
            continue;
        }
        let op0_rsp = insn.op0_kind() == OpKind::Register && insn.op0_register() == Register::RSP;
        match insn.code() {
            Code::Sub_rm64_imm8 | Code::Sub_rm64_imm32 if op0_rsp => depth += insn.immediate32to64(),
            Code::Add_rm64_imm8 | Code::Add_rm64_imm32 if op0_rsp => depth -= insn.immediate32to64(),
            _ => depth -= insn.stack_pointer_increment() as i64,
        }
        if insn.flow_control() != FlowControl::Next {
            prologue = false;
        }
    }
    false
}

/// The instructions of the copy of the function at `start` (bytes `code`):
/// RSP lowered by [`SHIFT`] for its whole run, every memory address kept.
pub fn rewrite(start: u64, code: &[u8]) -> Result<Vec<Instruction>, Skip> {
    let end = start + code.len() as u64;
    let mut insns = Vec::new();
    let mut decoder = Decoder::with_ip(64, code, start, DecoderOptions::NONE);
    while decoder.can_decode() {
        let insn = decoder.decode();
        if insn.is_invalid() {
            return Err(Skip::Decode);
        }
        insns.push(insn);
    }
    if !uses_red_zone(&insns) {
        return Err(Skip::NoRedZone);
    }
    if code.len() < ENTRY_JUMP {
        return Err(Skip::TooShort);
    }
    let starts: std::collections::HashSet<u64> = insns.iter().map(|i| i.ip()).collect();
    let inside = |t: u64| t >= start && t < end;

    let mut out = vec![lea(Register::RSP, Register::RSP, -SHIFT)];
    let mut stubs: Vec<Instruction> = Vec::new();
    for insn in &insns {
        let ip = insn.ip();
        let seg = insn.segment_prefix();
        if has_memory(insn) && (seg == Register::FS || seg == Register::GS) {
            return Err(Skip::Segment);
        }
        match insn.flow_control() {
            FlowControl::Call | FlowControl::IndirectCall | FlowControl::Interrupt | FlowControl::XbeginXabortXend => return Err(Skip::Unsupported),
            _ => {}
        }
        if insn.is_jcc_short_or_near() || insn.is_jmp_short_or_near() || insn.is_loop() || insn.is_loopcc() || insn.is_jcx_short() {
            let t = insn.near_branch_target();
            if inside(t) {
                if !starts.contains(&t) || (t > start && t < start + ENTRY_JUMP as u64) {
                    return Err(Skip::Branch);
                }
                out.push(*insn);
                continue;
            }
            // Leaves the function: RSP goes back up first.
            if insn.is_jmp_short_or_near() {
                out.push(at_ip(lea(Register::RSP, Register::RSP, SHIFT), ip));
                let jmp = Instruction::with_branch(Code::Jmp_rel32_64, t).map_err(|_| Skip::Encode)?;
                out.push(at_ip(jmp, 0));
            } else if insn.is_jcc_short_or_near() {
                let label = STUB_LABEL + stubs.len() as u64 * 16;
                let mut jcc = *insn;
                jcc.set_near_branch64(label);
                out.push(jcc);
                stubs.push(at_ip(lea(Register::RSP, Register::RSP, SHIFT), label));
                let jmp = Instruction::with_branch(Code::Jmp_rel32_64, t).map_err(|_| Skip::Encode)?;
                stubs.push(at_ip(jmp, 0));
            } else {
                return Err(Skip::Branch);
            }
            continue;
        }
        match insn.flow_control() {
            FlowControl::IndirectBranch | FlowControl::Return => {
                // A tail jump, a jump table or a return: the target sees
                // the original RSP. The operand is read after the raise,
                // so an RSP-based one stays as it is.
                out.push(at_ip(lea(Register::RSP, Register::RSP, SHIFT), ip));
                if rsp_operand(insn) {
                    return Err(Skip::RspUse);
                }
                out.push(at_ip(*insn, 0));
                continue;
            }
            FlowControl::Next | FlowControl::Exception => {}
            _ => return Err(Skip::Unsupported),
        }
        match insn.code() {
            Code::Push_r64 => {
                if insn.op0_register() == Register::RSP {
                    return Err(Skip::RspUse);
                }
                out.push(at_ip(lea(Register::RSP, Register::RSP, -8), ip));
                let st = Instruction::with2(Code::Mov_rm64_r64, MemoryOperand::with_base_displ(Register::RSP, SHIFT), insn.op0_register()).map_err(|_| Skip::Encode)?;
                out.push(at_ip(st, 0));
                continue;
            }
            Code::Pushq_imm8 | Code::Pushq_imm32 => {
                out.push(at_ip(lea(Register::RSP, Register::RSP, -8), ip));
                let st = Instruction::with2(Code::Mov_rm64_imm32, MemoryOperand::with_base_displ(Register::RSP, SHIFT), insn.immediate32to64() as i32).map_err(|_| Skip::Encode)?;
                out.push(at_ip(st, 0));
                continue;
            }
            Code::Pop_r64 => {
                if insn.op0_register() == Register::RSP {
                    return Err(Skip::RspUse);
                }
                let ld = Instruction::with2(Code::Mov_r64_rm64, insn.op0_register(), MemoryOperand::with_base_displ(Register::RSP, SHIFT)).map_err(|_| Skip::Encode)?;
                out.push(at_ip(ld, ip));
                out.push(at_ip(lea(Register::RSP, Register::RSP, 8), 0));
                continue;
            }
            Code::Leaveq => {
                out.push(at_ip(lea(Register::RSP, Register::RBP, -SHIFT), ip));
                let ld = Instruction::with2(Code::Mov_r64_rm64, Register::RBP, MemoryOperand::with_base_displ(Register::RSP, SHIFT)).map_err(|_| Skip::Encode)?;
                out.push(at_ip(ld, 0));
                out.push(at_ip(lea(Register::RSP, Register::RSP, 8), 0));
                continue;
            }
            _ => {}
        }
        if insn.stack_pointer_increment() != 0 {
            return Err(Skip::Unsupported);
        }
        let mut new = *insn;
        if rsp_operand(insn) {
            let op0_rsp = insn.op0_kind() == OpKind::Register && insn.op0_register() == Register::RSP;
            match insn.code() {
                // mov r, rsp -> lea r, [rsp + SHIFT]
                Code::Mov_r64_rm64 | Code::Mov_rm64_r64 if !op0_rsp && insn.op0_kind() == OpKind::Register && insn.op1_kind() == OpKind::Register => {
                    out.push(at_ip(lea(insn.op0_register(), Register::RSP, SHIFT), ip));
                    continue;
                }
                // mov rsp, r -> lea rsp, [r - SHIFT]
                Code::Mov_r64_rm64 | Code::Mov_rm64_r64 if op0_rsp && insn.op1_kind() == OpKind::Register && insn.op1_register() != Register::RSP => {
                    out.push(at_ip(lea(Register::RSP, insn.op1_register(), -SHIFT), ip));
                    continue;
                }
                // RSP moves by a constant: both copies move alike (the
                // flags are those of the lowered value; compilers never
                // read them).
                Code::Add_rm64_imm8 | Code::Add_rm64_imm32 | Code::Sub_rm64_imm8 | Code::Sub_rm64_imm32 if op0_rsp => {}
                // Alignment to a power of two up to SHIFT keeps the
                // distance (SHIFT is a multiple of it).
                Code::And_rm64_imm8 | Code::And_rm64_imm32 if op0_rsp => {
                    let m = insn.immediate32to64();
                    let align = m.unsigned_abs();
                    if m >= 0 || !align.is_power_of_two() || align > SHIFT as u64 {
                        return Err(Skip::RspUse);
                    }
                }
                Code::Lea_r64_m if op0_rsp => {
                    // lea rsp, [rsp + d] moves both alike; lea rsp, [r + d]
                    // sets the original value, lowered here.
                    if insn.memory_base() != Register::RSP {
                        if insn.memory_index() == Register::RSP {
                            return Err(Skip::RspUse);
                        }
                        add_displ(&mut new, -SHIFT)?;
                    }
                    out.push(at_ip(new, ip));
                    continue;
                }
                _ => return Err(Skip::RspUse),
            }
        }
        if has_memory(insn) && insn.memory_base() == Register::RSP {
            add_displ(&mut new, SHIFT)?;
        }
        out.push(at_ip(new, ip));
    }
    if insns.last().is_some_and(|i| matches!(i.flow_control(), FlowControl::Next | FlowControl::ConditionalBranch)) {
        // The last instruction can fall through to the code after the
        // function: continue there with the original RSP.
        out.push(lea(Register::RSP, Register::RSP, SHIFT));
        out.push(Instruction::with_branch(Code::Jmp_rel32_64, end).map_err(|_| Skip::Encode)?);
    }
    out.extend(stubs);
    Ok(out)
}

/// Encode the copy for address `at`.
pub fn encode(list: &[Instruction], at: u64) -> Result<Vec<u8>, Skip> {
    BlockEncoder::encode(64, InstructionBlock::new(list, at), BlockEncoderOptions::NONE).map(|r| r.code_buffer).map_err(|_| Skip::Encode)
}

/// The jump that replaces the first bytes of a moved function.
pub fn entry_jump(from: u64, to: u64) -> Option<[u8; ENTRY_JUMP]> {
    let rel = i32::try_from(to.wrapping_sub(from + ENTRY_JUMP as u64) as i64).ok()?;
    let mut j = [0xE9, 0, 0, 0, 0];
    j[1..].copy_from_slice(&rel.to_le_bytes());
    Some(j)
}

// ---------------------------------------------------------------------------
// Function bounds from .eh_frame
// ---------------------------------------------------------------------------

/// Guest memory a parser may read: `[lo, hi)`.
struct Image {
    lo: u64,
    hi: u64,
}

impl Image {
    fn byte(&self, a: u64) -> Option<u8> {
        // SAFETY: inside the mapped image.
        (a >= self.lo && a < self.hi).then(|| unsafe { std::ptr::read(a as *const u8) })
    }

    fn bytes<const N: usize>(&self, a: u64) -> Option<[u8; N]> {
        if a < self.lo || a.checked_add(N as u64)? > self.hi {
            return None;
        }
        // SAFETY: inside the mapped image.
        Some(unsafe { std::ptr::read_unaligned(a as *const [u8; N]) })
    }

    fn uleb(&self, a: &mut u64) -> Option<u64> {
        let (mut v, mut shift) = (0u64, 0u32);
        loop {
            let b = self.byte(*a)?;
            *a += 1;
            if shift < 64 {
                v |= ((b & 0x7F) as u64) << shift;
            }
            shift += 7;
            if b & 0x80 == 0 {
                return Some(v);
            }
        }
    }

    fn sleb(&self, a: &mut u64) -> Option<i64> {
        let (mut v, mut shift) = (0i64, 0u32);
        loop {
            let b = self.byte(*a)?;
            *a += 1;
            if shift < 64 {
                v |= ((b & 0x7F) as i64) << shift;
            }
            shift += 7;
            if b & 0x80 == 0 {
                if shift < 64 && b & 0x40 != 0 {
                    v |= -1i64 << shift;
                }
                return Some(v);
            }
        }
    }

    /// A `DW_EH_PE_*` encoded value at `*a` (`data`: base of datarel).
    fn encoded(&self, a: &mut u64, enc: u8, data: u64) -> Option<u64> {
        let field = *a;
        let v = match enc & 0x0F {
            0x00 | 0x04 | 0x0C => {
                let v = u64::from_le_bytes(self.bytes::<8>(*a)?);
                *a += 8;
                v
            }
            0x01 => self.uleb(a)?,
            0x02 => {
                let v = u16::from_le_bytes(self.bytes::<2>(*a)?) as u64;
                *a += 2;
                v
            }
            0x03 => {
                let v = u32::from_le_bytes(self.bytes::<4>(*a)?) as u64;
                *a += 4;
                v
            }
            0x09 => self.sleb(a)? as u64,
            0x0A => {
                let v = i16::from_le_bytes(self.bytes::<2>(*a)?) as i64 as u64;
                *a += 2;
                v
            }
            0x0B => {
                let v = i32::from_le_bytes(self.bytes::<4>(*a)?) as i64 as u64;
                *a += 4;
                v
            }
            _ => return None,
        };
        Some(match enc & 0x70 {
            0x00 => v,
            0x10 => field.wrapping_add(v),
            0x30 => data.wrapping_add(v),
            _ => return None,
        })
    }

    /// The pointer encoding of the FDEs of the CIE at `cie`.
    fn cie_fde_encoding(&self, cie: u64) -> Option<u8> {
        let len = u32::from_le_bytes(self.bytes::<4>(cie)?);
        if len == 0 || len == u32::MAX {
            return None;
        }
        let mut a = cie + 8;
        let version = self.byte(a)?;
        a += 1;
        let aug_start = a;
        while self.byte(a)? != 0 {
            a += 1;
        }
        let aug: Vec<u8> = (aug_start..a).filter_map(|p| self.byte(p)).collect();
        a += 1;
        if aug.contains(&b'e') {
            // "eh": an old GCC pointer follows; not produced by current
            // toolchains.
            return None;
        }
        self.uleb(&mut a)?; // code alignment
        self.sleb(&mut a)?; // data alignment
        if version == 1 {
            a += 1;
        } else {
            self.uleb(&mut a)?;
        }
        let mut enc = 0u8;
        if aug.first() == Some(&b'z') {
            self.uleb(&mut a)?;
            for c in &aug[1..] {
                match c {
                    b'R' => {
                        enc = self.byte(a)?;
                        a += 1;
                    }
                    b'P' => {
                        let pe = self.byte(a)?;
                        a += 1;
                        self.encoded(&mut a, pe & 0x7F, 0)?;
                    }
                    b'L' => a += 1,
                    _ => {}
                }
            }
        }
        Some(enc)
    }
}

/// Function ranges `(start, length)` listed by the `.eh_frame_hdr` at
/// `hdr` of a module mapped at `[lo, hi)`, sorted by start.
///
/// # Safety
/// `[lo, hi)` must be mapped and readable.
pub unsafe fn functions(hdr: u64, lo: u64, hi: u64) -> Vec<(u64, u64)> {
    let img = Image { lo, hi };
    let mut out = Vec::new();
    let (Some(version), Some(ptr_enc), Some(count_enc), Some(table_enc)) = (img.byte(hdr), img.byte(hdr + 1), img.byte(hdr + 2), img.byte(hdr + 3)) else {
        return out;
    };
    if version != 1 || table_enc == 0xFF || count_enc == 0xFF {
        return out;
    }
    let mut a = hdr + 4;
    if img.encoded(&mut a, ptr_enc, hdr).is_none() {
        return out;
    }
    let Some(count) = img.encoded(&mut a, count_enc, hdr) else { return out };
    let mut cies: std::collections::HashMap<u64, Option<u8>> = std::collections::HashMap::new();
    for _ in 0..count.min(1 << 24) {
        let (Some(_loc), Some(fde)) = (img.encoded(&mut a, table_enc, hdr), img.encoded(&mut a, table_enc, hdr)) else { break };
        let Some(len) = img.bytes::<4>(fde).map(u32::from_le_bytes) else { continue };
        if len == 0 || len == u32::MAX {
            continue;
        }
        let Some(cie_off) = img.bytes::<4>(fde + 4).map(u32::from_le_bytes) else { continue };
        let cie = (fde + 4).wrapping_sub(cie_off as u64);
        let enc = *cies.entry(cie).or_insert_with(|| img.cie_fde_encoding(cie));
        let Some(enc) = enc else { continue };
        let mut p = fde + 8;
        let (Some(begin), Some(range)) = (img.encoded(&mut p, enc, hdr), img.encoded(&mut p, enc & 0x0F, hdr)) else { continue };
        if range != 0 && range < 1 << 24 {
            out.push((begin, range));
        }
    }
    out.sort_unstable();
    out
}

// ---------------------------------------------------------------------------
// Installing the copies (Windows)
// ---------------------------------------------------------------------------

/// Moved functions: `(copy start, copy end, original start)`, by copy.
#[cfg(windows)]
static COPIES: std::sync::RwLock<Vec<(u64, u64, u64)>> = std::sync::RwLock::new(Vec::new());

/// The original function of an address inside a moved copy.
#[cfg(windows)]
pub fn original_of(addr: u64) -> Option<u64> {
    let v = COPIES.read().ok()?;
    let i = v.partition_point(|c| c.0 <= addr);
    let c = v.get(i.checked_sub(1)?)?;
    (addr < c.1).then_some(c.2)
}

/// Move every red-zone function among `funcs` (sorted `(start, length)`)
/// that lies inside one of `segments` (`(base, size)`, writable guest code
/// no thread runs yet). Runs before the load-time patch of the segments;
/// each copy gets the same in-place replacements (`replace`).
///
/// # Safety
/// The segments must be mapped and writable.
#[cfg(windows)]
pub unsafe fn protect_functions(funcs: &[(u64, u64)], segments: &[(u64, usize)], replace: crate::code_patch::Replace) -> (Counts, std::collections::BTreeMap<&'static str, usize>) {
    let mut counts = Counts::default();
    let mut kept = std::collections::BTreeMap::new();
    let mut copies = Vec::new();
    let mut prev_end = 0u64;
    for &(start, len) in funcs {
        let end = start + len;
        if start < prev_end || !segments.iter().any(|&(b, s)| start >= b && end <= b + s as u64) {
            continue;
        }
        prev_end = end;
        counts.functions += 1;
        let code = std::slice::from_raw_parts_mut(start as *mut u8, len as usize);
        let list = match rewrite(start, code) {
            Ok(l) => l,
            Err(Skip::NoRedZone) => continue,
            Err(e) => {
                counts.kept += 1;
                *kept.entry(skip_name(e)).or_insert(0) += 1;
                continue;
            }
        };
        // No instruction of the copy (a branch in its longest form
        // included) takes more than 16 bytes.
        let bound = 16 * list.len() as u64 + 64;
        let mut copy_len = 0;
        let placed = crate::tls_win::place(start, bound, |at| {
            let bytes = encode(&list, at).ok()?;
            entry_jump(start, at)?;
            copy_len = bytes.len();
            Some(bytes)
        });
        match placed.and_then(|at| Some((at, entry_jump(start, at)?))) {
            Some((at, jump)) => {
                crate::code_patch::patch_guest_code(at, copy_len, replace);
                code[..ENTRY_JUMP].copy_from_slice(&jump);
                copies.push((at, at + copy_len as u64, start));
                counts.moved += 1;
            }
            None => {
                counts.kept += 1;
                *kept.entry(skip_name(Skip::Encode)).or_insert(0) += 1;
            }
        }
    }
    if !copies.is_empty() {
        let mut v = COPIES.write().unwrap_or_else(|e| e.into_inner());
        v.extend(copies);
        v.sort_unstable();
    }
    (counts, kept)
}

#[cfg(windows)]
fn skip_name(s: Skip) -> &'static str {
    match s {
        Skip::NoRedZone => "no red zone",
        Skip::Decode => "decode",
        Skip::Unsupported => "call/trap",
        Skip::Segment => "fs/gs",
        Skip::RspUse => "rsp use",
        Skip::Branch => "branch",
        Skip::TooShort => "too short",
        Skip::Encode => "encode",
    }
}

/// Move the red-zone functions of a module (see the module docs).
///
/// # Safety
/// `[lo, hi)` must be the mapped module and `segments` its writable code
/// segments that no thread runs yet.
#[cfg(windows)]
pub unsafe fn protect_module(name: &str, eh_frame_hdr: u64, lo: u64, hi: u64, segments: &[(u64, usize)], replace: crate::code_patch::Replace) -> Counts {
    let funcs = functions(eh_frame_hdr, lo, hi);
    let (counts, kept) = protect_functions(&funcs, segments, replace);
    if counts.moved + counts.kept != 0 {
        log::info!(
            "[redzone] {name}: {} of {} functions keep data below RSP: {} moved to copies with RSP lowered, {} left as they are {:?}",
            counts.moved + counts.kept,
            counts.functions,
            counts.moved,
            counts.kept,
            kept
        );
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced_x86::{Formatter, IntelFormatter};

    fn text(code: &[u8]) -> Result<Vec<String>, Skip> {
        let list = rewrite(0x1000, code)?;
        let mut f = IntelFormatter::new();
        Ok(list
            .iter()
            .map(|i| {
                let mut s = String::new();
                f.format(i, &mut s);
                s
            })
            .collect())
    }

    #[test]
    fn leaf_with_red_zone_runs_lowered() {
        #[rustfmt::skip]
        let code = [
            0x53,                               // push rbx
            0x48, 0x89, 0x7C, 0x24, 0xD0,       // mov [rsp-0x30], rdi
            0x48, 0x8B, 0x44, 0x24, 0xD0,       // mov rax, [rsp-0x30]
            0x48, 0x8B, 0x4C, 0x24, 0x10,       // mov rcx, [rsp+0x10]
            0x5B,                               // pop rbx
            0xC3,                               // ret
        ];
        assert_eq!(
            text(&code).unwrap(),
            [
                "lea rsp,[rsp-80h]",
                "lea rsp,[rsp-8]",
                "mov [rsp+80h],rbx",
                "mov [rsp+50h],rdi",
                "mov rax,[rsp+50h]",
                "mov rcx,[rsp+90h]",
                "mov rbx,[rsp+80h]",
                "lea rsp,[rsp+8]",
                "lea rsp,[rsp+80h]",
                "ret",
            ]
        );
    }

    #[test]
    fn functions_without_red_zone_or_with_calls_stay() {
        // mov rax, [rsp+8]; ret
        assert_eq!(text(&[0x48, 0x8B, 0x44, 0x24, 0x08, 0xC3]), Err(Skip::NoRedZone));
        // mov [rsp-8], rdi; call +0; ret: a call writes below RSP, so this is
        // no red zone
        assert_eq!(text(&[0x48, 0x89, 0x7C, 0x24, 0xF8, 0xE8, 0, 0, 0, 0, 0xC3]), Err(Skip::NoRedZone));
        // push rbp; mov rbp, rsp; sub rsp, 16; mov [rbp-8], rdi; leave; ret:
        // the slot is inside the allocated frame
        assert_eq!(text(&[0x55, 0x48, 0x89, 0xE5, 0x48, 0x83, 0xEC, 0x10, 0x48, 0x89, 0x7D, 0xF8, 0xC9, 0xC3]), Err(Skip::NoRedZone));
        // push rbp; mov rbp, rsp; mov [rbp-8], rdi; pop rbp; ret: below RSP
        assert_eq!(
            text(&[0x55, 0x48, 0x89, 0xE5, 0x48, 0x89, 0x7D, 0xF8, 0x5D, 0xC3]).unwrap(),
            ["lea rsp,[rsp-80h]", "lea rsp,[rsp-8]", "mov [rsp+80h],rbp", "lea rbp,[rsp+80h]", "mov [rbp-8],rdi", "mov rbp,[rsp+80h]", "lea rsp,[rsp+8]", "lea rsp,[rsp+80h]", "ret"]
        );
        // mov [rsp-8], rdi; mov rax, fs:[0]; ret
        assert_eq!(text(&[0x48, 0x89, 0x7C, 0x24, 0xF8, 0x64, 0x48, 0x8B, 0x04, 0x25, 0, 0, 0, 0, 0xC3]), Err(Skip::Segment));
        // mov [rsp-8], rdi; xchg rsp, rax; ret
        assert_eq!(text(&[0x48, 0x89, 0x7C, 0x24, 0xF8, 0x48, 0x94, 0xC3]), Err(Skip::RspUse));
    }

    #[test]
    fn exits_raise_rsp_first() {
        #[rustfmt::skip]
        let code = [
            0x48, 0x89, 0x7C, 0x24, 0xF8,       // 1000: mov [rsp-8], rdi
            0x48, 0x85, 0xFF,                   // 1005: test rdi, rdi
            0x0F, 0x84, 0xF2, 0x0F, 0, 0,       // 1008: je 2000 (out)
            0x74, 0xF0,                         // 100e: je 1000 (entry)
            0x48, 0x89, 0xE0,                   // 1010: mov rax, rsp
            0x48, 0x89, 0xC4,                   // 1013: mov rsp, rax
            0xFF, 0xE0,                         // 1016: jmp rax
        ];
        let t = text(&code).unwrap();
        assert_eq!(
            t,
            [
                "lea rsp,[rsp-80h]",
                "mov [rsp+78h],rdi",
                "test rdi,rdi",
                "je 8000000000000000h",
                "je short 0000000000001000h",
                "lea rax,[rsp+80h]",
                "lea rsp,[rax-80h]",
                "lea rsp,[rsp+80h]",
                "jmp rax",
                "lea rsp,[rsp+80h]",
                "jmp 0000000000002000h",
            ]
        );
        // The copy encodes away from the original: the loop to the entry
        // stays inside the copy, the exits reach the original targets.
        let at = 0x40_0000u64;
        let bytes = encode(&rewrite(0x1000, &code).unwrap(), at).unwrap();
        let back: Vec<Instruction> = Decoder::with_ip(64, &bytes, at, DecoderOptions::NONE).into_iter().collect();
        let first = back[1].ip();
        assert_eq!(back[4].near_branch_target(), first, "je 1000 loops to the copy's first original instruction");
        let stub = back[3].near_branch_target();
        let stub_at = back.iter().position(|i| i.ip() == stub).unwrap();
        assert_eq!(back[stub_at].code(), Code::Lea_r64_m);
        assert_eq!(back[stub_at + 1].near_branch_target(), 0x2000);
    }

    /// A branch into the bytes the entry jump replaces keeps the function.
    #[test]
    fn branch_into_the_entry_jump_keeps_the_function() {
        #[rustfmt::skip]
        let code = [
            0x53,                               // 1000: push rbx
            0x48, 0x89, 0x7C, 0x24, 0xF8,       // 1001: mov [rsp-8], rdi
            0xEB, 0xF9,                         // 1006: jmp 1001
        ];
        assert_eq!(text(&code), Err(Skip::Branch));
    }

    /// The copy computes what the original computes, with RSP lower while
    /// it runs.
    #[cfg(unix)]
    #[test]
    fn copy_behaves_like_the_original() {
        #[rustfmt::skip]
        let code: [u8; 30] = [
            0x53,                               // push rbx
            0x48, 0x89, 0x7C, 0x24, 0xD0,       // mov [rsp-0x30], rdi
            0x48, 0x89, 0x74, 0x24, 0x80,       // mov [rsp-0x80], rsi
            0x48, 0x89, 0xE3,                   // mov rbx, rsp
            0x48, 0x8B, 0x43, 0xD0,             // mov rax, [rbx-0x30]
            0x48, 0x03, 0x43, 0x80,             // add rax, [rbx-0x80]
            0x48, 0x01, 0xD0,                   // add rax, rdx (the caller's rsp)
            0x48, 0x29, 0xD8,                   // sub rax, rbx
            0x5B,                               // pop rbx
            0xC3,                               // ret
        ];
        let page = ps5vm::os::map_anon(0x2000, ps5vm::MemProtect::RWX).unwrap();
        // SAFETY: the test's own pages.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), page as *mut u8, code.len()) };
        let copy_at = page + 0x1000;
        let bytes = encode(&rewrite(page, &code).unwrap(), copy_at).unwrap();
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), copy_at as *mut u8, bytes.len()) };
        let run = |f: u64| -> u64 {
            let r: u64;
            // SAFETY: a leaf function of the test that preserves rbx.
            unsafe {
                std::arch::asm!(
                    "mov rdx, rsp",
                    "call {f}",
                    f = in(reg) f, in("rdi") 7u64, in("rsi") 5u64, out("rdx") _, lateout("rax") r, clobber_abi("C"),
                );
            }
            r
        };
        // Both find the slots where they stored them, and RSP (as the
        // function reads it) 16 bytes below the caller's: 7 + 5 + 16.
        assert_eq!(run(page), 28);
        assert_eq!(run(copy_at), 28);
        ps5vm::os::unmap(page, 0x2000);
    }

    #[test]
    fn reads_function_ranges_from_eh_frame_hdr() {
        // A hand-built .eh_frame_hdr + .eh_frame: one CIE ("zR", FDE
        // pointers pcrel|sdata4) and two FDEs.
        let mut img = vec![0u8; 0x200];
        let eh = 0x40usize;
        #[rustfmt::skip]
        let cie: [u8; 20] = [
            16, 0, 0, 0,        // length
            0, 0, 0, 0,         // CIE id
            1,                  // version
            b'z', b'R', 0,      // augmentation
            1, 0x78, 16,        // code align 1, data align -8, return register 16
            1, 0x1B,            // augmentation data: FDE encoding pcrel|sdata4
            0, 0, 0,            // DW_CFA_nop
        ];
        img[eh..eh + 20].copy_from_slice(&cie);
        let funcs = [(0x100usize, 0x20u32), (0x180, 0x10)];
        let base = img.as_ptr() as u64;
        let mut fdes = Vec::new();
        for (i, &(start, len)) in funcs.iter().enumerate() {
            let f = eh + 20 + i * 20;
            fdes.push((start, f));
            img[f..f + 4].copy_from_slice(&16u32.to_le_bytes());
            img[f + 4..f + 8].copy_from_slice(&((f + 4 - eh) as u32).to_le_bytes());
            img[f + 8..f + 12].copy_from_slice(&((start as i64 - (f + 8) as i64) as i32).to_le_bytes());
            img[f + 12..f + 16].copy_from_slice(&len.to_le_bytes());
        }
        // Header at 0: version 1, eh_frame_ptr pcrel|sdata4, count udata4,
        // table datarel|sdata4.
        img[0..4].copy_from_slice(&[1, 0x1B, 0x03, 0x3B]);
        img[4..8].copy_from_slice(&((eh - 4) as i32).to_le_bytes());
        img[8..12].copy_from_slice(&2u32.to_le_bytes());
        for (i, &(start, f)) in fdes.iter().enumerate() {
            let e = 12 + i * 8;
            img[e..e + 4].copy_from_slice(&(start as i32).to_le_bytes());
            img[e + 4..e + 8].copy_from_slice(&(f as i32).to_le_bytes());
        }
        // SAFETY: the test's own buffer.
        let f = unsafe { functions(base, base, base + img.len() as u64) };
        assert_eq!(f, [(base + 0x100, 0x20), (base + 0x180, 0x10)]);
    }
}
