// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! CPU layer.
//!
//! Guest PS5 code is x86-64 and runs natively on the host CPU, one host
//! thread per guest thread. Import calls are stubs that jump to the direct
//! import entry and are dispatched to HLE callbacks on the host stack
//! (INT3 stubs and the signal handler with `RPS5E_IMPORT_TRAP=1`; see
//! `exec`).
//!
//! HLE dispatch is trait-based: the caller provides an `HleDispatcher`
//! implementation that maps guest stub addresses to NIDs. This keeps
//! `ps5cpu` free of a dependency on the HLE crate.

pub mod code_patch;
pub mod ctx;
pub mod exec;
pub mod host_cpu;
pub mod isa_scan;
#[cfg(any(windows, test))]
pub mod redzone;
pub mod signal;
pub mod tls_patch;
pub mod x64emu;

pub use exec::{GuestExit, GuestRegs, HleDispatcher};
#[cfg(windows)]
mod tls_win;
#[cfg(windows)]
mod win;
