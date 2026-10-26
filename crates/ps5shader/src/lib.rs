// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Shader front-end for the PS5 GPU (gfx10 / RDNA2 "Gen5" ISA).
//!
//! * [`decode`] — per-instruction decoder: encoding detection, opcode
//!   tables, instruction lengths, operands and control fields.
//! * [`program`] — whole-program decoding from guest memory or a blob
//!   ([`decode_program`], [`decode_blob`]), S_SETPC_B64 handling and fused
//!   GS/HS halves ([`decode_fused_program`]).
//! * [`ir`] — analyses: branch resolution and CFG, EXEC-full analysis,
//!   wave64 half-mask analysis, MOVREL (M0) offset bounding.
//! * [`disasm`] — `Display` implementations for debugging.
//! * [`resources`] — resource planning: the scalar value graph that locates
//!   every V#/T#/S#/address operand, the [`resources::ShaderResourcePlan`]
//!   with its dense binding tables, the binding layout, and per-draw
//!   materialisation of descriptor words (see the module docs).
//!
//! The model ([`Program`] → [`Instruction`] with [`Opcode`], [`Operand`]s
//! and an optional [`Control`]) is what a SPIR-V back-end consumes.

pub mod decode;
pub mod disasm;
pub mod fxhash;
pub mod inline_constants;
pub mod ir;
mod ir_types;
pub mod opcode;
pub mod program;
pub mod resources;
pub mod cfg;
pub mod interp;
pub mod opt;
pub mod ssa;
pub mod testing;
pub mod translate;

pub use decode::{create_instruction, decode_instruction, decode_opcode, InstructionError};
pub use ir_types::*;
pub use opcode::{FlatOp, FlatSegment, Opcode};
pub use program::{
    decode_blob, decode_fused_program, decode_program, decode_program_through, decode_segment, CodeSlice, CodeSource, DecodeError, FusedLayout,
    Termination, FUSED_CONTINUATION_ALIGNMENT, MAX_DECLARED_SHADER_SIZE,
};
