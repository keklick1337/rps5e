// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

// The reference's LdsMixedWidthAtomics programs (256 invocations, wave32,
// 4 LDS dwords): ds_add_u64 of 0xffffffff and ds_add_u32 of 1 on the
// slot's high dword, uniform, with divergent EXEC halves, and split lanes
// (with operands loaded from s[0:3]); every lane stores the slot's 64 bits
// through s[4:7]. Pulled in with include!.

#[allow(dead_code)]
const THREADS: u32 = 256;
#[allow(dead_code)]
const ITERATIONS: usize = 16;
#[allow(dead_code)]
const INPUT_BYTES: u32 = THREADS * 16;
#[allow(dead_code)]
const OUTPUT_BYTES: u32 = THREADS * 8;

#[allow(dead_code)]
fn unrolled(head: &[u32], body: &[u32], count: usize, tail: &[u32]) -> Vec<u32> {
    let mut code = head.to_vec();
    for _ in 0..count {
        code.extend_from_slice(body);
    }
    code.extend_from_slice(tail);
    code
}

#[allow(dead_code)]
const HEAD: [u32; 12] = [0x34100083, 0x7e020280, 0x7e0402c1, 0x7e060280, 0x7e080281, 0x7e0c0280, 0x7e0e0280, 0xd9340000, 0x00000601, 0xbf8cc07f, 0xbf8a0000, 0xbe89037e];
#[allow(dead_code)]
const TAIL: [u32; 9] = [0xbefe0309, 0xbf8cc07f, 0xbf8a0000, 0xd9d80000, 0x0a000001, 0xbf8cc07f, 0xe0741000, 0x80010a08, 0xbf810000];

#[allow(dead_code)]
fn uniform_code() -> Vec<u32> {
    unrolled(&HEAD[..11], &[0xd9000000, 0x00000201, 0xd8000004, 0x00000401, 0xbf8cc07f], ITERATIONS, &TAIL[1..])
}

#[allow(dead_code)]
fn divergent_code() -> Vec<u32> {
    unrolled(&HEAD, &[0xbefe03ff, 0x55555555, 0xd9000000, 0x00000201, 0xbefe03ff, 0xaaaaaaaa, 0xd8000004, 0x00000401, 0xbf8cc07f], ITERATIONS, &TAIL)
}

#[allow(dead_code)]
fn parity_code() -> Vec<u32> {
    unrolled(&HEAD, &[0xbefe03ff, 0x55555555, 0xd9000000, 0x00000201, 0xbf8cc07f, 0xbefe03ff, 0xaaaaaaaa, 0xd8000004, 0x00000401, 0xbf8cc07f], 1, &TAIL)
}

#[allow(dead_code)]
fn loaded_parity_code() -> Vec<u32> {
    let head = [0x34100083, 0x340a0084, 0x7e020280, 0xe0381000, 0x80000c05, 0x7e0c0280, 0x7e0e0280, 0xd9340000, 0x00000601, 0xbf8c0070, 0xbf8a0000, 0xbe89037e];
    unrolled(&head, &[0xbefe03ff, 0x55555555, 0xd9000000, 0x00000c01, 0xbf8cc07f, 0xbefe03ff, 0xaaaaaaaa, 0xd8000004, 0x00000e01, 0xbf8cc07f], 1, &TAIL)
}
