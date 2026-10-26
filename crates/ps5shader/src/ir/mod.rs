// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Analyses over decoded programs.

pub mod branch;
pub mod cfg;
pub mod exec_full;
pub mod movrel;
pub mod wave64_half_mask;

pub use branch::{BranchResolver, Gen5BranchResolver};
pub use cfg::{BlockRange, ControlFlowGraph};
pub use wave64_half_mask::Wave64HalfMaskPlan;
