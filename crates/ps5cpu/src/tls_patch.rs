// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Thread-pointer redirection for natively executed guest code.
//!
//! Guest code reaches its thread control block through the FS segment
//! (`mov rax, fs:[0]`, `fs:[0x28]` stack canary, ...). The host FS base
//! belongs to the host C runtime, so FS cannot hold the guest thread
//! pointer. GS is unused by host user code on x86-64 Linux, so every guest
//! instruction with an FS segment override is rewritten to use GS instead —
//! a one-byte change of the 0x64 prefix to 0x65 that keeps instruction
//! length and semantics. Each host thread running guest code then sets its
//! GS base to the guest thread pointer (see `set_thread_pointer`).

use iced_x86::{Decoder, DecoderOptions, Register};

const PREFIX_FS: u8 = 0x64;
pub(crate) const PREFIX_GS: u8 = 0x65;

/// Rewrite FS-relative instructions in `[base, base+size)` to GS-relative.
/// Returns the number of instructions patched.
///
/// # Safety
/// The region must be mapped, writable and contain guest code.
pub unsafe fn patch_fs_to_gs(base: u64, size: usize) -> usize {
    let code = std::slice::from_raw_parts_mut(base as *mut u8, size);
    let mut patches = Vec::new();
    {
        let mut decoder = Decoder::with_ip(64, code, base, DecoderOptions::NONE);
        while decoder.can_decode() {
            let insn = decoder.decode();
            if insn.is_invalid() || insn.segment_prefix() != Register::FS {
                continue;
            }
            let start = (insn.ip() - base) as usize;
            let len = insn.len();
            if let Some(off) = fs_prefix_offset(&code[start..start + len]) {
                patches.push(start + off);
            }
        }
    }
    for &p in &patches {
        code[p] = PREFIX_GS;
    }
    patches.len()
}

/// Offset of the effective FS segment prefix inside one instruction: the
/// last FS byte in the legacy prefix block ahead of the opcode (and any REX
/// prefix). Displacement and immediate bytes that happen to equal 0x64 are
/// never considered.
pub(crate) fn fs_prefix_offset(insn: &[u8]) -> Option<usize> {
    let mut found = None;
    for (i, &b) in insn.iter().enumerate() {
        match b {
            PREFIX_FS => found = Some(i),
            // Other legacy prefixes: lock, rep, operand/address size,
            // the remaining segment overrides.
            0xF0 | 0xF2 | 0xF3 | 0x66 | 0x67 | 0x2E | 0x36 | 0x3E | 0x26 | 0x65 => {}
            // REX or the opcode: the prefix block is over.
            _ => return found,
        }
    }
    found
}

#[cfg(windows)]
pub use crate::tls_win::{rewrite as rewrite_fs, trap_target, TlsCounts};

/// Make the guest thread control block of this host thread `tp` (a TEB
/// TLS slot that the rewritten guest accesses read).
#[cfg(windows)]
pub fn set_thread_pointer(tp: u64) -> bool {
    crate::win::set_thread_pointer(tp)
}

/// Current guest thread pointer of this host thread.
#[cfg(windows)]
pub fn thread_pointer() -> u64 {
    crate::win::thread_pointer()
}

/// Point this host thread's GS base at the guest thread control block.
#[cfg(unix)]
pub fn set_thread_pointer(tp: u64) -> bool {
    const ARCH_SET_GS: libc::c_int = 0x1001;
    unsafe { libc::syscall(libc::SYS_arch_prctl, ARCH_SET_GS, tp) == 0 }
}

/// Current guest thread pointer of this host thread.
#[cfg(unix)]
pub fn thread_pointer() -> u64 {
    const ARCH_GET_GS: libc::c_int = 0x1004;
    let mut v: u64 = 0;
    unsafe { libc::syscall(libc::SYS_arch_prctl, ARCH_GET_GS, &mut v as *mut u64) };
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patches_fs_loads_and_canary() {
        // mov rax, fs:[0] ; mov rcx, fs:[0x28] ; nop ; mov eax, [rbx]
        let mut code: Vec<u8> = vec![
            0x64, 0x48, 0x8B, 0x04, 0x25, 0, 0, 0, 0,
            0x64, 0x48, 0x8B, 0x0C, 0x25, 0x28, 0, 0, 0,
            0x90,
            0x8B, 0x03,
        ];
        let n = unsafe { patch_fs_to_gs(code.as_mut_ptr() as u64, code.len()) };
        assert_eq!(n, 2);
        assert_eq!(code[0], 0x65);
        assert_eq!(code[9], 0x65);
        assert_eq!(&code[18..], &[0x90, 0x8B, 0x03]);
    }

    #[test]
    fn fs_byte_in_displacement_is_not_a_prefix() {
        // mov rax, fs:[-0x9C] : the disp32 contains 0x64 but the prefix is the first byte.
        let mut code: Vec<u8> = vec![0x64, 0x48, 0x8B, 0x04, 0x25, 0x64, 0xFF, 0xFF, 0xFF];
        // mov rax, fs:[rdi+0x64] : disp8 0x64 right after the ModRM byte.
        code.extend_from_slice(&[0x64, 0x48, 0x8B, 0x47, 0x64]);
        // mov eax, [rdi+0x64] with no FS: its 0x64 must stay untouched.
        code.extend_from_slice(&[0x8B, 0x47, 0x64]);
        let n = unsafe { patch_fs_to_gs(code.as_mut_ptr() as u64, code.len()) };
        assert_eq!(n, 2);
        assert_eq!(code[0], 0x65);
        assert_eq!(&code[1..9], &[0x48, 0x8B, 0x04, 0x25, 0x64, 0xFF, 0xFF, 0xFF]);
        assert_eq!(code[9], 0x65);
        assert_eq!(&code[10..14], &[0x48, 0x8B, 0x47, 0x64]);
        assert_eq!(&code[14..], &[0x8B, 0x47, 0x64]);
    }

    #[test]
    #[cfg(unix)]
    fn gs_base_round_trip() {

        let handle = std::thread::spawn(|| {
            let block = Box::new([0u64; 4]);
            let tp = block.as_ptr() as u64;
            assert!(set_thread_pointer(tp));
            assert_eq!(thread_pointer(), tp);
            let v: u64;
            unsafe { std::arch::asm!("mov {}, gs:[0]", out(reg) v) };
            assert_eq!(v, 0);
        });
        handle.join().unwrap();
    }
}
