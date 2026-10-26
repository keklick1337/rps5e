// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Typed SSA intermediate representation.
//!
//! * [`types`] — value types (bool, u16/u32/u64, f16/f32/f64, vectors).
//! * [`op`] — operations; semantics follow SPIR-V so emission is direct.
//! * [`func`] — the [`Function`]: value arena, blocks with structured
//!   terminators and merge declarations, and the [`Builder`].
//! * [`build`] — SSA construction from the register-level form.
//! * [`print`] — text form; [`verify`] — validation.
//!
//! Execution model: one IR invocation per guest lane. All guest branches are
//! wave-uniform (they test SCC, VCCZ or EXECZ), so every invocation follows
//! the same path and the guest EXEC mask is data: each invocation keeps its
//! EXEC bit, vector register writes select between the new and old value on
//! it, and lane-mask SGPRs are built with ballots. The one exception is the
//! back-end's last pass, [`crate::opt::exec_runs`], whose branches on the
//! EXEC bit diverge (their arms hold per-invocation arithmetic only).

pub mod build;
pub mod func;
pub mod op;
pub mod print;
pub mod types;
pub mod verify;

pub use func::{Block, BlockData, Builder, Function, Merge, Stage, Terminator, Value, ValueData};
pub use op::*;
pub use types::{Scalar, Type};
