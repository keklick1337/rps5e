// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

// Hardware-measured reference rows: sdwa move selectors (code dwords and expectations).
#[allow(dead_code)]
const BROADCAST_EXEC: [u32; 2] = [0xa5a5a5a5, 0x3c3c3c3c];
#[allow(dead_code)]
const SOURCES: [u32; 32] = [0x00000000, 0xffffffff, 0x80808080, 0x7f7f7f7f, 0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000, 0x00000080, 0x00008000, 0x00800000, 0x80000000, 0x0000007f, 0x00007fff, 0x007fffff, 0x7fffffff, 0x01020384, 0x8504f306, 0x12345678, 0x9abcdef0, 0xfedcba98, 0x76543210, 0x00ff00ff, 0xff00ff00, 0x80007fff, 0x7fff8000, 0xc3a5815a, 0x5a81a5c3, 0x0f1e2d3c, 0xf0e1d2c3, 0x40c08001, 0xdeadbeef];
