// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! A frame-pacing compute program captured from a game (one wave64
//! invocation): it reads `s_memrealtime` and spins until the clock passes
//! `max(deadline, previous start + interval)`, at most `max_wait` ticks
//! after its first read (`s_cbranch_vccz` exit). The game takes the
//! deadline from a command-processor timestamp.
//!
//! Constant buffer (s[4:7], dwords): deadline lo/hi, max_wait, interval.
//! Output buffer (s[0:3], stride 4): [64:65] start (read as the previous
//! start first), [66:67] deadline, [68:69] last clock read, [70] max_wait;
//! [0..64] = 0..63 at the end.

pub const FRAME_PACING_WAIT: [u32; 97] = [
    0xbfa00003, 0xf4940300, 0xfa000000, 0x7e0002c0, 0x7e0402ff, 0x00000041, //
    0xf4280202, 0xfa000000, 0xbf8cc07f, 0x91059f0b, 0xbe84030b, 0x7e08020c, //
    0xe0302000, 0x80000300, 0xe0302000, 0x80000102, 0xd4e4006a, 0x00010004, //
    0x7e0a020d, 0x7e0c02ff, 0x00000044, 0x7e0e02ff, 0x00000045, 0x7e1002ff, //
    0x00000042, 0x7e120208, 0x7e140209, 0x7e1602ff, 0x00000043, 0x7e18020a, //
    0x7e1a02ff, 0x00000046, 0xe0702000, 0x80000400, 0xe0702000, 0x80000502, //
    0xe0702000, 0x80000406, 0xe0702000, 0x80000507, 0xe0702000, 0x80000908, //
    0xe0702000, 0x80000a0b, 0xe0702000, 0x80000c0d, 0x80060a0c, 0xbe8e0380, //
    0x8207800d, 0xbf13806a, 0xbf8c3f71, 0xd70f6a00, 0x0002060b, 0xbf8c3f70, //
    0x50020205, 0x7dc20008, 0x020502f9, 0x86860680, 0x360404fd, 0x7d8a0480, //
    0x02000008, 0x02020209, 0x7dc20006, 0x02000cf9, 0x86060600, 0x02020ef9, //
    0x86060601, 0x7dc2000c, 0xbf860014, 0xf4940300, 0xfa000000, 0x36041cf9, //
    0x868606bf, 0x7e06020e, 0xbf8cc07f, 0x7e08020c, 0x7e0a02ff, 0x00000044, //
    0x7e0c020d, 0x7e0e02ff, 0x00000045, 0xe0702000, 0x80000302, 0x810e810e, //
    0xe0702000, 0x80000405, 0xe0702000, 0x80000607, 0xbf82ffea, 0xbeea0380, //
    0x7e00026a, 0x816a816a, 0xbf04c06a, 0xe0702000, 0x80000000, 0xbf85fffa, //
    0xbf810000,
];
