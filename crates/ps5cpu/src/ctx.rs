// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Host signal-context types.
//!
//! The execution core works on a Linux-shaped interrupted context: 23
//! general registers indexed by `REG_*` and a pointer to a 512-byte FXSAVE
//! image. On Linux these are the kernel's own types. On Windows the
//! vectored exception handler (`win`) builds the same shape from a
//! `CONTEXT` and writes it back, so the trap, fault and import code is
//! shared.

#[cfg(unix)]
pub use libc::{c_int, c_void, greg_t, siginfo_t, ucontext_t};
#[cfg(unix)]
pub use libc::{
    REG_CSGSFS, REG_EFL, REG_ERR, REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15, REG_R8, REG_R9, REG_RAX, REG_RBP, REG_RBX, REG_RCX, REG_RDI,
    REG_RDX, REG_RIP, REG_RSI, REG_RSP,
};
#[cfg(unix)]
pub use libc::{SIGBUS, SIGFPE, SIGILL, SIGSEGV, SIGTRAP};
/// FXSAVE image of an interrupted context.
#[cfg(unix)]
pub type FpState = libc::_libc_fpstate;

#[cfg(windows)]
pub use self::win_types::*;

#[cfg(windows)]
#[allow(non_camel_case_types)]
mod win_types {
    pub type c_int = i32;
    pub type c_void = core::ffi::c_void;
    pub type greg_t = i64;

    pub const REG_R8: c_int = 0;
    pub const REG_R9: c_int = 1;
    pub const REG_R10: c_int = 2;
    pub const REG_R11: c_int = 3;
    pub const REG_R12: c_int = 4;
    pub const REG_R13: c_int = 5;
    pub const REG_R14: c_int = 6;
    pub const REG_R15: c_int = 7;
    pub const REG_RDI: c_int = 8;
    pub const REG_RSI: c_int = 9;
    pub const REG_RBP: c_int = 10;
    pub const REG_RBX: c_int = 11;
    pub const REG_RDX: c_int = 12;
    pub const REG_RAX: c_int = 13;
    pub const REG_RCX: c_int = 14;
    pub const REG_RSP: c_int = 15;
    pub const REG_RIP: c_int = 16;
    pub const REG_EFL: c_int = 17;
    pub const REG_CSGSFS: c_int = 18;
    pub const REG_ERR: c_int = 19;

    pub const SIGILL: c_int = 4;
    pub const SIGTRAP: c_int = 5;
    pub const SIGBUS: c_int = 7;
    pub const SIGFPE: c_int = 8;
    pub const SIGSEGV: c_int = 11;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct XmmReg {
        pub element: [u32; 4],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct FpxReg {
        pub significand: [u16; 4],
        pub exponent: u16,
        pub padding: [u16; 3],
    }

    /// FXSAVE image (the layout of `CONTEXT::FltSave`).
    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    pub struct FpState {
        pub cwd: u16,
        pub swd: u16,
        pub ftw: u16,
        pub fop: u16,
        pub rip: u64,
        pub rdp: u64,
        pub mxcsr: u32,
        pub mxcr_mask: u32,
        pub _st: [FpxReg; 8],
        pub _xmm: [XmmReg; 16],
        pub padding: [u64; 12],
    }

    const _: () = assert!(std::mem::size_of::<FpState>() == 512);

    #[repr(C)]
    pub struct mcontext_t {
        pub gregs: [greg_t; 23],
        pub fpregs: *mut FpState,
    }

    #[repr(C)]
    pub struct ucontext_t {
        pub uc_mcontext: mcontext_t,
        /// The Windows `CONTEXT` this was built from (null when none).
        pub native: *mut c_void,
    }

    #[repr(C)]
    pub struct siginfo_t {
        pub si_code: c_int,
        pub addr: u64,
    }

    impl siginfo_t {
        /// Faulting data address.
        ///
        /// # Safety
        /// Mirrors the Linux accessor; always safe here.
        pub unsafe fn si_addr(&self) -> *mut c_void {
            self.addr as *mut c_void
        }
    }
}
