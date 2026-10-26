// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Host operating-system primitives shared by the crates.
//!
//! Each host OS has its own submodule; this module re-exports the common
//! surface (`exit_now`, `current_tid`, anonymous fixed mappings,
//! protection changes, clocks). Linux code that predates this module still
//! calls `libc` directly; new code and the Windows paths go through here.

#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use unix::*;

#[cfg(windows)]
pub mod windows;
#[cfg(windows)]
pub use self::windows::*;

/// Protection bits in the Linux `PROT_*` encoding, used by the memory
/// trackers on every host.
pub const PROT_NONE: i32 = 0;
pub const PROT_READ: i32 = 1;
pub const PROT_WRITE: i32 = 2;
pub const PROT_EXEC: i32 = 4;

impl crate::MemProtect {
    /// From `PROT_*` bits.
    pub fn from_prot_bits(bits: i32) -> Self {
        Self { read: bits & PROT_READ != 0, write: bits & PROT_WRITE != 0, exec: bits & PROT_EXEC != 0 }
    }
}
