// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Load-time rewriting of guest code (one decode pass per executable
//! segment, before any of it runs):
//!
//!   * FS-relative accesses become GS-relative (`tls_patch`);
//!   * on hosts without the console extension, instructions that have an
//!     equally long native equivalent with the same architectural effect
//!     are replaced in place:
//!       - `MONITORX` -> 3-byte NOP: it only arms address monitoring;
//!       - `MWAITX` -> `PAUSE; NOP`: MWAITX may return at any time (any
//!         interrupt or store ends the wait), its registers are inputs only;
//!       - `MOVNTSS` / `MOVNTSD` (SSE4a scalar non-temporal stores) ->
//!         `MOVSS` / `MOVSD` stores of the same operands: the same bytes
//!         are stored, only the cache hint differs.
//!
//! `EXTRQ` / `INSERTQ` have no equally long native form; they stay and are
//! emulated from SIGILL (`x64emu`).

use crate::host_cpu::host;
use iced_x86::{Decoder, DecoderOptions, Instruction, Mnemonic, Register};

/// What the pass changed.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatchCounts {
    pub fs_to_gs: usize,
    pub monitorx: usize,
    pub mwaitx: usize,
    pub movnt_scalar: usize,
}

/// Host features that decide the in-place replacements.
#[derive(Clone, Copy, Debug)]
pub struct Replace {
    /// The host lacks MONITORX/MWAITX.
    pub monitorx: bool,
    /// The host lacks SSE4a.
    pub sse4a: bool,
}

impl Replace {
    /// From the host CPU.
    pub fn for_host() -> Self {
        let h = host();
        Self { monitorx: !h.has("MONITORX"), sse4a: !h.has("SSE4A") }
    }
}

/// Offset of the opcode escape byte (0x0F) of a legacy-encoded instruction:
/// after the legacy prefixes and an optional REX.
fn escape_offset(bytes: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            0xF0 | 0xF2 | 0xF3 | 0x66 | 0x67 | 0x2E | 0x36 | 0x3E | 0x26 | 0x64 | 0x65 => i += 1,
            b if b & 0xF0 == 0x40 => return (bytes.get(i + 1) == Some(&0x0F)).then_some(i + 1),
            0x0F => return Some(i),
            _ => return None,
        }
    }
    None
}

/// The in-place replacement of one instruction, if any.
fn replacement(insn: &Instruction, bytes: &[u8], r: Replace) -> Option<(Vec<u8>, fn(&mut PatchCounts))> {
    match insn.mnemonic() {
        Mnemonic::Monitorx if r.monitorx && bytes == [0x0F, 0x01, 0xFA] => Some((vec![0x0F, 0x1F, 0x00], |c| c.monitorx += 1)),
        Mnemonic::Mwaitx if r.monitorx && bytes == [0x0F, 0x01, 0xFB] => Some((vec![0xF3, 0x90, 0x90], |c| c.mwaitx += 1)),
        Mnemonic::Movntss | Mnemonic::Movntsd if r.sse4a => {
            // F3/F2 [REX] 0F 2B /r (memory) -> F3/F2 [REX] 0F 11 /r: the
            // mandatory prefix and the operands are kept.
            let esc = escape_offset(bytes)?;
            if bytes.get(esc + 1) != Some(&0x2B) {
                return None;
            }
            let mut out = bytes.to_vec();
            out[esc + 1] = 0x11;
            Some((out, |c| c.movnt_scalar += 1))
        }
        _ => None,
    }
}

/// Rewrite the guest code in `[base, base+size)` (see the module docs).
///
/// # Safety
/// The region must be mapped, writable and contain guest code that no
/// thread executes yet.
pub unsafe fn patch_guest_code(base: u64, size: usize, r: Replace) -> PatchCounts {
    let code = std::slice::from_raw_parts_mut(base as *mut u8, size);
    let mut counts = PatchCounts::default();
    let mut writes: Vec<(usize, Vec<u8>)> = Vec::new();
    #[cfg(windows)]
    let mut fs_insns: Vec<Instruction> = Vec::new();
    {
        let mut decoder = Decoder::with_ip(64, code, base, DecoderOptions::NONE);
        let mut insn = Instruction::default();
        while decoder.can_decode() {
            decoder.decode_out(&mut insn);
            if insn.is_invalid() {
                continue;
            }
            let start = (insn.ip() - base) as usize;
            let bytes = &code[start..start + insn.len()];
            if insn.segment_prefix() == Register::FS {
                #[cfg(windows)]
                fs_insns.push(insn);
                #[cfg(unix)]
                if let Some(off) = crate::tls_patch::fs_prefix_offset(bytes) {
                    writes.push((start + off, vec![crate::tls_patch::PREFIX_GS]));
                    counts.fs_to_gs += 1;
                }
                continue;
            }
            if let Some((new, count)) = replacement(&insn, bytes, r) {
                writes.push((start, new));
                count(&mut counts);
            }
        }
    }
    for (at, bytes) in writes {
        code[at..at + bytes.len()].copy_from_slice(&bytes);
    }
    #[cfg(windows)]
    {
        let c = crate::tls_patch::rewrite_fs(base, code, &fs_insns);
        counts.fs_to_gs = c.jumps + c.traps;
        if c.traps + c.failed != 0 {
            log::info!("[tls] {base:#x}: {} thread-pointer accesses jump to trampolines, {} trap, {} not rewritten", c.jumps, c.traps, c.failed);
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use super::*;

    #[cfg(unix)]
    const ALL: Replace = Replace { monitorx: true, sse4a: true };
    #[cfg(unix)]
    const NONE: Replace = Replace { monitorx: false, sse4a: false };

    #[cfg(unix)]
    #[rustfmt::skip]
    fn sample() -> Vec<u8> {
        vec![
            0x4C, 0x89, 0xF0,                   // mov rax, r14
            0x31, 0xC9,                         // xor ecx, ecx
            0x0F, 0x01, 0xFA,                   // monitorx
            0xB9, 0x02, 0x00, 0x00, 0x00,       // mov ecx, 2
            0x0F, 0x01, 0xFB,                   // mwaitx
            0xF2, 0x0F, 0x2B, 0x07,             // movntsd [rdi], xmm0
            0xF3, 0x44, 0x0F, 0x2B, 0x4E, 0x08, // movntss [rsi+8], xmm9
            0x64, 0x48, 0x8B, 0x04, 0x25, 0, 0, 0, 0, // mov rax, fs:[0]
            0x66, 0x0F, 0x78, 0xC1, 0x08, 0x04, // extrq xmm1, 8, 4 (kept)
        ]
    }

    #[test]
    #[cfg(unix)]
    fn replaces_only_on_hosts_without_the_extension() {
        let mut code = sample();
        let c = unsafe { patch_guest_code(code.as_mut_ptr() as u64, code.len(), NONE) };
        assert_eq!(c, PatchCounts { fs_to_gs: 1, ..Default::default() });
        let mut expect = sample();
        expect[26] = 0x65;
        assert_eq!(code, expect);

        let mut code = sample();
        let c = unsafe { patch_guest_code(code.as_mut_ptr() as u64, code.len(), ALL) };
        assert_eq!(c, PatchCounts { fs_to_gs: 1, monitorx: 1, mwaitx: 1, movnt_scalar: 2 });
        assert_eq!(&code[5..8], &[0x0F, 0x1F, 0x00]);
        assert_eq!(&code[13..16], &[0xF3, 0x90, 0x90]);
        assert_eq!(&code[16..20], &[0xF2, 0x0F, 0x11, 0x07]);
        assert_eq!(&code[20..26], &[0xF3, 0x44, 0x0F, 0x11, 0x4E, 0x08]);
        assert_eq!(&code[35..], &[0x66, 0x0F, 0x78, 0xC1, 0x08, 0x04]);
        // The rewritten stream decodes to the expected instructions.
        let mnemonics: Vec<Mnemonic> = Decoder::new(64, &code, DecoderOptions::NONE).into_iter().map(|i| i.mnemonic()).collect();
        assert_eq!(
            mnemonics,
            [Mnemonic::Mov, Mnemonic::Xor, Mnemonic::Nop, Mnemonic::Mov, Mnemonic::Pause, Mnemonic::Nop, Mnemonic::Movsd, Mnemonic::Movss, Mnemonic::Mov, Mnemonic::Extrq]
        );
    }

    /// The replacements store what the SSE4a instructions store.
    #[test]
    #[cfg(unix)]
    fn scalar_store_replacement_runs_natively() {

        #[rustfmt::skip]
        let mut code = vec![
            0xF2, 0x0F, 0x2B, 0x07,             // movntsd [rdi], xmm0
            0xF3, 0x0F, 0x2B, 0x4E, 0x08,       // movntss [rsi+8], xmm1
            0xC3,
        ];
        let c = unsafe { patch_guest_code(code.as_mut_ptr() as u64, code.len(), ALL) };
        assert_eq!(c.movnt_scalar, 2);
        let page = unsafe {
            libc::mmap(std::ptr::null_mut(), 4096, libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
        };
        assert_ne!(page, libc::MAP_FAILED);
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), page as *mut u8, code.len()) };
        let mut d = [0xAAu8; 8];
        let mut s = [0xBBu8; 16];
        let x0: [u8; 16] = std::array::from_fn(|i| i as u8 + 1);
        let x1: [u8; 16] = std::array::from_fn(|i| i as u8 + 0x41);
        unsafe {
            let f: extern "C" fn() = std::mem::transmute(page);
            std::arch::asm!(
                "movdqu xmm0, [{a}]",
                "movdqu xmm1, [{b}]",
                "call {f}",
                a = in(reg) x0.as_ptr(), b = in(reg) x1.as_ptr(), f = in(reg) f,
                in("rdi") d.as_mut_ptr(), in("rsi") s.as_mut_ptr(),
                clobber_abi("C"),
            );
            libc::munmap(page, 4096);
        }
        assert_eq!(d, x0[..8]);
        assert_eq!(&s[..8], &[0xBB; 8]);
        assert_eq!(&s[8..12], &x1[..4]);
        assert_eq!(&s[12..], &[0xBB; 4]);
    }
}
