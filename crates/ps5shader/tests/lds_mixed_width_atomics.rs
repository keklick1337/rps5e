// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! ds_add_u64 and ds_add_u32 on one LDS slot (the 32-bit add on its high
//! dword) from a 256-invocation work-group (eight wave32 waves): every add
//! of both widths lands, uniformly, with divergent EXEC halves, and with
//! split lanes whose operands come from memory (the reference's
//! LdsMixedWidthAtomics programs, hardware-equivalent sums).

#[path = "vectors/ref_harness.rs"]
mod harness;

use harness::*;

include!("vectors/mem_lds_mixed_width.rs");

fn raw(base: u32, bytes: u32) -> [u32; 4] {
    [base, 0, bytes, 0x3101_6FAC]
}

/// The program on one work-group; every lane reads the slot's 64 bits
/// after `operations` adds of each width.
fn check(name: &'static str, code: Vec<u32>, operations: u64) {
    let mut user = vec![0u32; 64];
    user[0..4].copy_from_slice(&raw(0, INPUT_BYTES));
    user[4..8].copy_from_slice(&raw(INPUT_BYTES, OUTPUT_BYTES));
    let mut initial: Vec<u32> = (0..THREADS).flat_map(|_| [0xffff_ffff, 0, 1, 0]).collect();
    let inputs = initial.len();
    initial.extend(std::iter::repeat_n(0xdead_beef, THREADS as usize * 2));
    let value = operations * 0xffff_ffff + (operations << 32);
    let mut expected = initial[..inputs].to_vec();
    expected.extend((0..THREADS).flat_map(|_| [value as u32, (value >> 32) as u32]));
    let case = RefCase {
        name,
        code: Box::leak(code.into_boxed_slice()),
        initial: Box::leak(initial.into_boxed_slice()),
        expected: Box::leak(expected.into_boxed_slice()),
        user_data: Some(Box::leak(user.into_boxed_slice())),
        compute: Compute {
            threads: [THREADS, 1, 1],
            lds_dwords: 4,
            wave_size: 32,
            float_mode: 0xc0,
            dispatch_threads: [0, 0, 0],
            workgroup_register: 0,
            thread_ids: 1,
            group_id: [false, false, false],
            dispatch_dims: false,
            tg_size_en: false,
        },
        dispatch: [1, 1, 1],
        compile_only: false,
        tolerance: 0.0,
        bda: &[],
    };
    if let Err(e) = case.check() {
        panic!("{name}: {e}");
    }
}

#[test]
fn uniform() {
    check("uniform", uniform_code(), THREADS as u64 * ITERATIONS as u64);
}

#[test]
fn divergent() {
    check("divergent", divergent_code(), THREADS as u64 / 2 * ITERATIONS as u64);
}

#[test]
fn split_lanes() {
    check("split lanes", parity_code(), THREADS as u64 / 2);
}

#[test]
fn split_lanes_with_per_lane_operands() {
    check("split lanes with per-lane operands", loaded_parity_code(), THREADS as u64 / 2);
}
