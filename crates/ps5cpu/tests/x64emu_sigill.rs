// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Instruction emulation through the real SIGILL path: code containing an
//! instruction that raises #UD on every host is called natively; the signal
//! handler emulates it and execution continues.

use ps5cpu::exec;

/// VRSQRTPS xmm1, xmm0 with the cleared VEX.vvvv bit (c5 f8 -> c5 f0) that
/// marks it for emulation; the upper half of ymm1 is dirtied first and
/// stored afterwards to check the VEX.128 zeroing.
#[rustfmt::skip]
const TRAPPED_VRSQRTPS: [u8; 25] = [
    0x0F, 0x10, 0x06,                         // movups xmm0, [rsi]
    0xC5, 0xFC, 0x10, 0x0E,                   // vmovups ymm1, [rsi]
    0xC5, 0xF0, 0x52, 0xC8,                   // vrsqrtps xmm1, xmm0 (trap mark)
    0x0F, 0x11, 0x0F,                         // movups [rdi], xmm1
    0xC4, 0xE3, 0x7D, 0x19, 0x4F, 0x10, 0x01, // vextractf128 [rdi + 16], ymm1, 1
    0xC5, 0xF8, 0x77,                         // vzeroupper
    0xC3,                                     // ret
];

fn has_avx() -> bool {
    std::arch::x86_64::__cpuid(1).ecx & (1 << 28) != 0
}

#[test]
fn trapped_vrsqrtps_is_emulated_from_sigill() {
    if !has_avx() {
        return;
    }
    exec::init();
    let page = ps5vm::os::map_anon(0x1000, ps5vm::MemProtect::RW).expect("map test page");
    unsafe { std::ptr::copy_nonoverlapping(TRAPPED_VRSQRTPS.as_ptr(), page as *mut u8, TRAPPED_VRSQRTPS.len()) };
    assert!(ps5vm::os::protect(page, 0x1000, ps5vm::MemProtect::RX));
    let input: [u32; 8] = [4.0f32.to_bits(), 0, 0xbf80_0000, 0x7f80_0000, 0x1111_1111, 0x2222_2222, 0x3333_3333, 0x4444_4444];
    let mut out = [0u32; 8];
    let f: extern "sysv64" fn(*mut u32, *const u32) = unsafe { std::mem::transmute(page) };
    f(out.as_mut_ptr(), input.as_ptr());
    assert_eq!(f32::from_bits(out[0]), 0.5);
    assert_eq!(&out[1..4], &[0x7f80_0000, 0xffc0_0000, 0]);
    assert_eq!(&out[4..], &[0, 0, 0, 0], "VEX.128 form zeroes the upper half");
}
