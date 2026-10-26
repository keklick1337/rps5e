// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Control flow: basic-block graph of a decoded program and the
//! structurizer.

pub mod calls;
pub mod dom;
pub mod graph;
pub mod link;
pub mod structurize;

pub use graph::{BasicBlock, BranchCond, Cfg, CfgError, Exit};
pub use structurize::{structurize, Cond, FlagId, Node, Structured};

#[cfg(test)]
#[path = "structurize_tests.rs"]
mod structurize_tests;
