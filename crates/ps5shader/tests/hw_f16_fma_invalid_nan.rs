// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Invalid f16 fma operations (inf * 0, inf - inf) give the default NaN
//! 0xfe00 whatever the host fma returns; v_fma_mix_f32 with f16 sources gives
//! 0xffc00000 (hardware rows, RDNA2, every FLOAT_MODE / IEEE_MODE pair).

include!("vectors/fmod_common.rs");

use ps5shader::translate::FloatMode;

const THREADS: usize = 10;

/// v1 = tid * 16, v2 = tid * 8; buffer_load_dwordx4 v[4:7]; v10 = 0;
/// v_fma_f16 v10, v4, v5, v6; v11 = v6; v_fmac_f16 v11, v4, v5;
/// buffer_store_dwordx2 v[10:11].
const FMA_CODE: [u32; 13] = [
    0x34020084, 0x34040083, 0xe03c1000, 0x80000401, 0xbf8c3f70, 0x7e140280, 0xd74b000a, 0x041a0b04, 0x7e160306, 0x6c160b04, 0xe0741000, 0x80010a02,
    0xbf810000,
];
/// v_fma_mix_f32 v10, v4, v5, v4 op_sel_hi:[0,0,1] (three f16 sources);
/// v_fma_mix_f32 v11, v4, v5, v6 op_sel_hi:[0,0,0] with the f16 lanes
/// picked as in the reference program.
const MIX_CODE: [u32; 12] = [
    0x34020084, 0x34040083, 0xe03c1000, 0x80000401, 0xbf8c3f70, 0xcc20600a, 0x1c120b04, 0xcc20600b, 0x0c120d04, 0xe0741000, 0x80010a02, 0xbf810000,
];

const FMA_ROWS: [[u32; 3]; THREADS] = [
    [0x7c00, 0x0000, 0x0000],
    [0x0000, 0xfc00, 0x3c00],
    [0x8000, 0x7c00, 0xfc00],
    [0x8000, 0xfc00, 0x3c00],
    [0x7c00, 0xbc00, 0x7c00],
    [0x7c00, 0x8000, 0xfc00],
    [0x7c00, 0x3c00, 0xfc00],
    [0xfc00, 0x3c00, 0x7c00],
    [0x7c00, 0x0000, 0x7e00],
    [0x8000, 0xfc00, 0x7d00],
];
const MIX_ROWS: [[u32; 3]; THREADS] = [
    [0x00007c00, 0x00000000, 0x00000000],
    [0x3c00fc00, 0x00000000, 0x80000000],
    [0x7e007c00, 0x00008000, 0x80000000],
    [0x7d00fc00, 0x00000000, 0x00000000],
    [0x7c007c00, 0x0000bc00, 0xbf800000],
    [0xfc007c00, 0x00003c00, 0x3f800000],
    [0x7c00fc00, 0x00004000, 0x40000000],
    [0xfc00fc00, 0x0000b800, 0xbf000000],
    [0x7e00fc00, 0x00008000, 0x00000000],
    [0x7fff7c00, 0x00000000, 0x80000000],
];

const MODES: [(&str, FloatMode); 4] = [
    ("denormals flushed, IEEE off", FloatMode { mode: 0x00, dx10_clamp: true, ieee: false, fp16_overflow: false }),
    ("denormals flushed, IEEE on", FloatMode { mode: 0x00, dx10_clamp: true, ieee: true, fp16_overflow: false }),
    ("denormals kept, IEEE off", FloatMode { mode: 0xf0, dx10_clamp: true, ieee: false, fp16_overflow: false }),
    ("denormals kept, IEEE on", FloatMode { mode: 0xf0, dx10_clamp: true, ieee: true, fp16_overflow: false }),
];

fn check(code: &[u32], rows: &[[u32; 3]; THREADS], names: [&str; 2], expected: u32) {
    let mut failures = Vec::new();
    for (mode_name, mode) in MODES {
        let input: Vec<u32> = rows.iter().flat_map(|r| [r[0], r[1], r[2], 0]).collect();
        let out = fmod_dispatch_mode(code, 32, THREADS as u32, &input, THREADS * 2, 0, mode).unwrap_or_else(|e| panic!("{e}"));
        for (lane, row) in rows.iter().enumerate() {
            for (form, name) in names.iter().enumerate() {
                let actual = out[lane * 2 + form];
                if actual != expected {
                    failures.push(format!("{name} {mode_name} of {row:08x?}: {actual:#010x}, expected {expected:#010x}"));
                }
            }
        }
    }
    fmod_report(&failures);
}

#[test]
fn f16_fma_invalid_operation_default_nan() {
    check(&FMA_CODE, &FMA_ROWS, ["v_fma_f16", "v_fmac_f16"], 0xfe00);
}

#[test]
fn fma_mix_f32_of_f16_sources_invalid_operation_default_nan() {
    check(&MIX_CODE, &MIX_ROWS, ["v_fma_mix_f32 of three f16 sources", "v_fma_mix_f32 of an f16, an f32 and an f16 source"], 0xffc0_0000);
}
