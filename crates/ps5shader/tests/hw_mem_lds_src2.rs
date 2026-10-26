// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! LDS src2 operations (ds_*_src2_*): MEM[A] = op(MEM[A], MEM[A + 4 *
//! the signed 15-bit offset]) per lane, no return value, against the
//! reference driver's execution test; the f32 forms and an offset taken
//! from the address are refused.

include!("vectors/mem_common.rs");
include!("vectors/mem_lds_src2.rs");

const THREADS: usize = 32;
const INPUTS: usize = 4;
const RESULTS: usize = 16;
const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

fn fill(tid: u32) -> [u32; 4] {
    let seed = (tid as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mixed = seed.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let w0 = if tid % 4 == 0 { tid % 7 } else { seed as u32 };
    let w1 = if tid % 4 == 0 {
        (tid * 3) % 7
    } else if tid % 4 == 1 {
        w0
    } else {
        (mixed >> 32) as u32
    };
    let w2 = (((tid % 9) as i32 - 4) as f32 * 1.75 + 0.5).to_bits();
    let w3 = ((((tid * 5) % 11) as i32 - 5) as f32 * 0.625 + 0.25).to_bits();
    [w0, w1, w2, w3]
}

#[test]
fn lds_src2_operations() {
    let mut input = vec![0u32; THREADS * INPUTS];
    for tid in 0..THREADS {
        input[tid * INPUTS..][..INPUTS].copy_from_slice(&fill(tid as u32));
    }
    let n_out = THREADS * RESULTS;
    let mut user = Vec::new();
    put_desc(&mut user, 0, vdesc(IN_BASE, (input.len() * 4) as u32));
    put_desc(&mut user, 4, vdesc(OUT_BASE, (n_out * 4) as u32));
    let inp = input.clone();
    let setup = move |m: &mut Machine| {
        m.memory.write_words(IN_BASE, &inp);
        m.memory.write_words(OUT_BASE, &vec![0xdeadbeef; n_out]);
    };
    let out = dispatch(&CODE, 32, THREADS as u32, user, &setup, &[(OUT_BASE, n_out)]).unwrap().remove(0);
    let names = [
        "ds_add_src2_u32", "ds_sub_src2_u32", "ds_rsub_src2_u32", "ds_inc_src2_u32", "ds_dec_src2_u32", "ds_min_src2_i32", "ds_max_src2_i32",
        "ds_min_src2_u32", "ds_max_src2_u32", "ds_and_src2_b32", "ds_or_src2_b32", "ds_xor_src2_b32", "ds_write_src2_b32",
    ];
    let mut c = Checker::new();
    for tid in 0..THREADS {
        let (a, b) = (input[tid * INPUTS], input[tid * INPUTS + 1]);
        let (sa, sb) = (a as i32, b as i32);
        let expected = [
            a.wrapping_add(b),
            a.wrapping_sub(b),
            b.wrapping_sub(a),
            if a >= b { 0 } else { a + 1 },
            if a == 0 || a > b { b } else { a - 1 },
            sa.min(sb) as u32,
            sa.max(sb) as u32,
            a.min(b),
            a.max(b),
            a & b,
            a | b,
            a ^ b,
            b,
        ];
        for (op, want) in expected.iter().enumerate() {
            c.eq(|| format!("lane {tid} {}", names[op]), out[tid * RESULTS + op], *want);
        }
    }
    c.finish();
}

#[test]
fn lds_src2_refusals() {
    let compile = |code: &[u32]| ComputeRun::new(code.to_vec(), 32, 32, vec![0; 8]).compile(true);
    for opcode in [0x92u32, 0x93, 0x95] {
        let err = compile(&[0xd800_0000 | (opcode << 18) | 1, 0, 0xbf81_0000]).unwrap_err();
        assert!(err.contains("f32 denormal mode"), "{opcode:#x}: {err}");
    }
    assert!(compile(&ADDRESS_OFFSET_CODE).is_err(), "an offset taken from the address was accepted");
}
