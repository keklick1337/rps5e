// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Program decoding: walking a code object to its last reachable
//! instruction, S_SETPC_B64 handling and fused GS/HS shader halves.

use crate::decode::{create_instruction, decode_opcode, InstructionError};
use crate::ir::branch;
use crate::ir_types::*;
use crate::opcode::Opcode;
use std::fmt;

/// Largest declared shader object the decoder accepts (1 MiB). Some shipped
/// shaders exceed 64 KiB before S_ENDPGM.
pub const MAX_DECLARED_SHADER_SIZE: u32 = 1024 * 1024;
const MAX_INSTRUCTIONS: usize = (MAX_DECLARED_SHADER_SIZE / 4) as usize;
/// A non-adjacent continuation is spliced at the entry end rounded up to this.
pub const FUSED_CONTINUATION_ALIGNMENT: u64 = 0x100;

/// Read access to guest code.
pub trait CodeSource {
    /// Little-endian dword at `address`, or `None` if unreadable.
    fn read_u32(&self, address: u64) -> Option<u32>;
}

/// A code blob mapped at `base`.
#[derive(Clone, Copy, Debug)]
pub struct CodeSlice<'a> {
    pub base: u64,
    pub bytes: &'a [u8],
}

impl<'a> CodeSlice<'a> {
    pub fn new(base: u64, bytes: &'a [u8]) -> Self {
        CodeSlice { base, bytes }
    }
}

impl CodeSource for CodeSlice<'_> {
    fn read_u32(&self, address: u64) -> Option<u32> {
        let off = usize::try_from(address.checked_sub(self.base)?).ok()?;
        let b = self.bytes.get(off..off.checked_add(4)?)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

/// Several blobs at distinct addresses (e.g. the two halves of a fused
/// shader).
impl CodeSource for [CodeSlice<'_>] {
    fn read_u32(&self, address: u64) -> Option<u32> {
        self.iter().find_map(|s| s.read_u32(address))
    }
}

impl<F: Fn(u64) -> Option<u32>> CodeSource for F {
    fn read_u32(&self, address: u64) -> Option<u32> {
        self(address)
    }
}

/// Why a program could not be decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeError {
    MissingAddress,
    ReadFailed { pc: u32 },
    Instruction { pc: u32, error: InstructionError },
    /// S_CODE_END before a pending forward branch target (or first).
    CodeEndInsideProgram { pc: u32, branch_target: u32 },
    TruncatedWord { pc: u32, limit: u32 },
    TruncatedInstruction { pc: u32, dwords: usize, limit: u32 },
    Unterminated { pc: u32, word: u32, count: usize, limit: Option<u32> },
    InvalidShaderSize { size: u32 },
    InvalidFusedLayout { entry: u64, continuation: u64 },
    FusedEntry(Box<DecodeError>),
    FusedEntryMissingSetpc { entry: u64 },
    FusedContinuation(Box<DecodeError>),
    FusedContinuationPcOverflow { pc: u32, base: u32 },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::MissingAddress => write!(f, "missing"),
            DecodeError::ReadFailed { pc } => write!(f, "read-failed pc=0x{pc:X}"),
            DecodeError::Instruction { pc, error } => write!(f, "{error} pc=0x{pc:X}"),
            DecodeError::CodeEndInsideProgram { pc, branch_target } => {
                write!(f, "code-end-inside-program pc=0x{pc:X} branchTarget=0x{branch_target:X}")
            }
            DecodeError::TruncatedWord { pc, limit } => write!(f, "truncated-word pc=0x{pc:X} limit=0x{limit:X}"),
            DecodeError::TruncatedInstruction { pc, dwords, limit } => {
                write!(f, "truncated-instruction pc=0x{pc:X} dwords={dwords} limit=0x{limit:X}")
            }
            DecodeError::Unterminated { pc, word, count, limit } => match limit {
                Some(l) => write!(f, "unterminated limit=0x{l:X} pc=0x{pc:X} word=0x{word:08X} count={count}"),
                None => write!(f, "unterminated pc=0x{pc:X} word=0x{word:08X} count={count}"),
            },
            DecodeError::InvalidShaderSize { size } => write!(f, "invalid-shader-size size=0x{size:X}"),
            DecodeError::InvalidFusedLayout { entry, continuation } => {
                write!(f, "invalid-fused-layout entry=0x{entry:X} continuation=0x{continuation:X}")
            }
            DecodeError::FusedEntry(e) => write!(f, "fused-entry: {e}"),
            DecodeError::FusedEntryMissingSetpc { entry } => write!(f, "fused-entry-missing-setpc entry=0x{entry:X}"),
            DecodeError::FusedContinuation(e) => write!(f, "fused-continuation: {e}"),
            DecodeError::FusedContinuationPcOverflow { pc, base } => {
                write!(f, "fused-continuation-pc-overflow pc=0x{pc:X} base=0x{base:X}")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// How a decoded segment ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    /// S_ENDPGM / backward S_BRANCH past every forward target, or S_CODE_END.
    EndProgram,
    /// S_SETPC_B64 (only when stopping there was requested).
    SetProgramCounter,
}

/// Decode a stand-alone program at `address`. Decoding stops at an
/// S_SETPC_B64: an unregistered S_SETPC transfers control to a separately
/// allocated code object and the bytes after it are padding or unrelated
/// data. Fused programs go through [`decode_fused_program`].
pub fn decode_program(code: &(impl CodeSource + ?Sized), address: u64) -> Result<Program, DecodeError> {
    decode_segment(code, address, None, true).map(|(p, _)| p)
}

/// Decode a code blob placed at address 0 (or `base`), e.g. a dumped shader.
pub fn decode_blob(bytes: &[u8], base: u64) -> Result<Program, DecodeError> {
    decode_program(&CodeSlice::new(base, bytes), base)
}

/// Decode one code object. `maximum_bytes` bounds the walk (declared shader
/// size); `stop_at_setpc` ends it at the first S_SETPC_B64.
pub fn decode_segment(
    code: &(impl CodeSource + ?Sized),
    address: u64,
    maximum_bytes: Option<u32>,
    stop_at_setpc: bool,
) -> Result<(Program, Termination), DecodeError> {
    decode_segment_through(code, address, maximum_bytes, stop_at_setpc, 0)
}

/// Decode a stand-alone program at `address` that is known to reach code
/// up to byte offset `required_end` (e.g. jump-table targets after an
/// S_SETPC_B64): decoding does not stop before it.
pub fn decode_program_through(code: &(impl CodeSource + ?Sized), address: u64, required_end: u32) -> Result<Program, DecodeError> {
    decode_segment_through(code, address, None, true, required_end).map(|(p, _)| p)
}

fn decode_segment_through(
    code: &(impl CodeSource + ?Sized),
    address: u64,
    maximum_bytes: Option<u32>,
    stop_at_setpc: bool,
    required_end: u32,
) -> Result<(Program, Termination), DecodeError> {
    if address == 0 {
        return Err(DecodeError::MissingAddress);
    }
    let mut instructions: Vec<Instruction> = Vec::new();
    let mut last_pc = 0u32;
    let mut last_word = 0u32;
    let mut furthest_forward_target = required_end.saturating_sub(4);
    let mut pc = 0u32;
    while instructions.len() < MAX_INSTRUCTIONS && maximum_bytes.is_none_or(|m| pc < m) {
        if let Some(m) = maximum_bytes {
            if m - pc < 4 {
                return Err(DecodeError::TruncatedWord { pc, limit: m });
            }
        }
        let word = code.read_u32(address + pc as u64).ok_or(DecodeError::ReadFailed { pc })?;
        last_pc = pc;
        last_word = word;
        let extra = code.read_u32(address + pc as u64 + 4);
        let (encoding, opcode, size) = decode_opcode(word, extra).map_err(|error| match error {
            InstructionError::MissingExtraWord { .. } => DecodeError::ReadFailed { pc: pc + 4 },
            error => DecodeError::Instruction { pc, error },
        })?;

        // S_CODE_END pads the space after the last instruction. Code placed
        // after the final S_ENDPGM ending in a backward branch runs into it.
        if opcode == Opcode::SCodeEnd {
            if pc < furthest_forward_target || instructions.is_empty() {
                return Err(DecodeError::CodeEndInsideProgram { pc, branch_target: furthest_forward_target });
            }
            return Ok((Program::new(address, instructions), Termination::EndProgram));
        }

        let bytes = size as u32 * 4;
        if let Some(m) = maximum_bytes {
            if bytes > m - pc {
                return Err(DecodeError::TruncatedInstruction { pc, dwords: size, limit: m });
            }
        }
        let mut words = Vec::with_capacity(size);
        words.push(word);
        for i in 1..size as u32 {
            let at = pc + i * 4;
            words.push(code.read_u32(address + at as u64).ok_or(DecodeError::ReadFailed { pc: at })?);
        }

        let mut instruction = create_instruction(pc, encoding, opcode, words);
        if matches!(&instruction.control, Some(Control::GlobalMemory(c)) if c.uses_flat_address) {
            instruction = resolve_flat_address_base(&instructions, instruction);
        }
        // S_CBRANCH_CDBG* only fire under an attached GPU debugger; their
        // targets are debugger stubs after S_ENDPGM and must not extend the
        // decoded program.
        if let Some(target) = branch::branch_target(&instruction) {
            if target > instruction.pc && !instruction.opcode.is_debugger_branch() {
                furthest_forward_target = furthest_forward_target.max(target);
            }
        }
        // A subroutine call's target is code the program reaches too (it
        // may follow the S_ENDPGM).
        if let Some(target) = crate::cfg::calls::call_offset_target(&instruction) {
            if target > instruction.pc {
                furthest_forward_target = furthest_forward_target.max(target);
            }
        }
        let opcode = instruction.opcode;
        let call = crate::cfg::calls::is_call(&instruction) && opcode == Opcode::SSwappcB64;
        instructions.push(instruction);
        if call {
            let window = Program::new(address, instructions.clone());
            if let Some(target) = crate::cfg::graph::direct_setpc_target(&window, instructions.len() - 1) {
                if target > pc as u64 && target < u32::MAX as u64 {
                    furthest_forward_target = furthest_forward_target.max(target as u32);
                }
            }
        }
        pc += bytes;

        // An unconditional backward S_BRANCH with no forward target pending
        // is the last reachable instruction too: compilers put out-of-line
        // loop blocks after S_ENDPGM, followed by the metadata trailer.
        if (opcode.is_program_end() || opcode == Opcode::SBranch) && pc > furthest_forward_target {
            return Ok((Program::new(address, instructions), Termination::EndProgram));
        }
        if stop_at_setpc && branch::is_computed_jump(instructions.last().unwrap()) && pc >= required_end && pc > furthest_forward_target {
            return Ok((Program::new(address, instructions), Termination::SetProgramCounter));
        }
    }
    Err(DecodeError::Unterminated { pc: last_pc, word: last_word, count: instructions.len(), limit: maximum_bytes })
}

fn validate_size(size: u32) -> Result<u32, DecodeError> {
    if size == 0 || !size.is_multiple_of(4) || size > MAX_DECLARED_SHADER_SIZE {
        Err(DecodeError::InvalidShaderSize { size })
    } else {
        Ok(size)
    }
}

/// The two code objects AGC joins into one hardware shader: the entry (front
/// half) transfers control to the continuation (back half) with S_SETPC_B64.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FusedLayout {
    pub entry_address: u64,
    /// Declared code size of the entry object (shader header +0x44).
    pub entry_size: u32,
    pub continuation_address: u64,
    pub continuation_size: u32,
}

/// Decode a fused program into one instruction stream: the entry up to its
/// S_SETPC_B64 (replaced by S_NOP), then the continuation rebased to follow
/// it (at its real distance when it lies after the entry, otherwise at the
/// entry end aligned to [`FUSED_CONTINUATION_ALIGNMENT`], with
/// `address_offset` keeping the real distance).
pub fn decode_fused_program(code: &(impl CodeSource + ?Sized), layout: FusedLayout) -> Result<Program, DecodeError> {
    let entry = layout.entry_address;
    let cont = layout.continuation_address;
    let entry_size = validate_size(layout.entry_size)?;
    let cont_size = validate_size(layout.continuation_size)?;
    let distance = cont.wrapping_sub(entry);
    if cont == entry || distance & 3 != 0 {
        return Err(DecodeError::InvalidFusedLayout { entry, continuation: cont });
    }
    let (entry_program, termination) =
        decode_segment(code, entry, Some(entry_size), true).map_err(|e| DecodeError::FusedEntry(Box::new(e)))?;
    if termination != Termination::SetProgramCounter || entry_program.instructions.is_empty() {
        return Err(DecodeError::FusedEntryMissingSetpc { entry });
    }
    let (cont_program, _) =
        decode_segment(code, cont, Some(cont_size), false).map_err(|e| DecodeError::FusedContinuation(Box::new(e)))?;

    let mut entry_instructions = entry_program.instructions;
    let setpc = entry_instructions.pop().unwrap();
    let entry_end = setpc.pc as u64 + setpc.words.len() as u64 * 4;
    let adjacent = cont > entry && distance <= u32::MAX as u64 && distance >= entry_end;
    let cont_pc = if adjacent {
        distance as u32
    } else {
        ((entry_end + FUSED_CONTINUATION_ALIGNMENT - 1) & !(FUSED_CONTINUATION_ALIGNMENT - 1)) as u32
    };
    let mut instructions = entry_instructions;
    instructions.reserve(cont_program.instructions.len() + 1);
    instructions.push(Instruction {
        encoding: Encoding::Sopp,
        opcode: Opcode::SNop,
        words: vec![0xBF80_0000],
        sources: vec![],
        destinations: vec![],
        control: None,
        ..setpc
    });
    for mut ins in cont_program.instructions {
        let rebased = cont_pc as u64 + ins.pc as u64;
        if rebased > u32::MAX as u64 {
            return Err(DecodeError::FusedContinuationPcOverflow { pc: ins.pc, base: cont_pc });
        }
        if !adjacent {
            ins.address_offset = Some(distance.wrapping_add(ins.program_offset()));
        }
        ins.pc = rebased as u32;
        instructions.push(ins);
    }
    Ok(Program::new(entry, instructions))
}

/// A FLAT access whose 64-bit VGPR address was built from an SGPR pair
/// (`v_mov vN, sK; v_mov vN+1, sK+1`) gets that pair as its scalar base.
fn resolve_flat_address_base(preceding: &[Instruction], mut instruction: Instruction) -> Instruction {
    let Some(Control::GlobalMemory(control)) = &instruction.control else { return instruction };
    let find = |reg: u32| {
        preceding.iter().rev().find(|c| c.destinations.iter().any(|d| d.kind == OperandKind::VectorRegister && d.value == reg))
    };
    let (Some(low), Some(high)) = (find(control.vector_address), find(control.vector_address + 1)) else {
        return instruction;
    };
    for ls in low.sources.iter().filter(|s| s.kind == OperandKind::ScalarRegister) {
        if high.sources.iter().any(|hs| hs.kind == OperandKind::ScalarRegister && hs.value == ls.value + 1) {
            let base = ls.value;
            instruction.sources.push(Operand::scalar(base));
            if let Some(Control::GlobalMemory(c)) = &mut instruction.control {
                c.scalar_address = base;
            }
            return instruction;
        }
    }
    instruction
}
