// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Native f32 denormal flushing (SPIR-V DenormFlushToZero(32)) replaces the
//! emulated flush for modules whose FLOAT_MODE flushes f32 inputs and
//! results everywhere and whose opcodes agree with the host's flush.

use ps5shader::decode_blob;
use ps5shader::ssa::Function;
use ps5shader::testing::{asm::*, ComputeRun};
use ps5shader::translate::{float_rules, native_ftz_applies, FloatMode};

/// FLOAT_MODE 0xC0: f32 denormals flushed (inputs and results), f16/f64
/// kept.
const FULL_FLUSH: FloatMode = FloatMode { mode: 0xC0, dx10_clamp: true, ieee: false, fp16_overflow: false };
/// f32 denormals kept.
const KEEP: FloatMode = FloatMode { mode: 0xF0, dx10_clamp: true, ieee: false, fp16_overflow: false };

fn program(code: &[u32]) -> ps5shader::Program {
    let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    decode_blob(&bytes, 0x10_0000_0000).expect("decode")
}

/// v_add_f32 v2, v0, v1 then end.
const ADD: &[u32] = &[vop2(0x03, 2, vgpr(0), 1), S_ENDPGM];

#[test]
fn applies_to_full_flush_arithmetic_only() {
    let add = program(ADD);
    assert!(native_ftz_applies(&add, FULL_FLUSH, float_rules::ALL), "arithmetic in the full-flush mode");
    assert!(!native_ftz_applies(&add, KEEP, float_rules::ALL), "kept denormals stay emulated");
    // The denormal output of v_rcp_f32 is the guest's: excluded.
    let rcp = program(&[vop1(0x2A, 2, vgpr(0)), S_ENDPGM]);
    assert!(!native_ftz_applies(&rcp, FULL_FLUSH, float_rules::ALL), "v_rcp_f32 keeps its flush");
    // The same for v_exp_f32 and the f64 -> f32 conversion.
    let exp = program(&[vop1(0x25, 2, vgpr(0)), S_ENDPGM]);
    assert!(!native_ftz_applies(&exp, FULL_FLUSH, float_rules::ALL));
    let cvt = program(&[vop1(0x0F, 2, vgpr(0)), S_ENDPGM]); // v_cvt_f32_f64 reads v[0:1]
    assert!(!native_ftz_applies(&cvt, FULL_FLUSH, float_rules::ALL));
    // v_log_f32 / v_sqrt_f32 agree with the host's flush: the input is
    // flushed unconditionally and the result is never a denormal f32.
    let log = program(&[vop1(0x27, 2, vgpr(0)), S_ENDPGM]);
    assert!(native_ftz_applies(&log, FULL_FLUSH, float_rules::ALL));
    let sqrt = program(&[vop1(0x33, 2, vgpr(0)), S_ENDPGM]);
    assert!(native_ftz_applies(&sqrt, FULL_FLUSH, float_rules::ALL));
    // The median of v_med3_f32 compares the unflushed operands (a host
    // flush turns med3(-0, +denormal, -denormal) from -0 into +0).
    let med3 = program(&[vop3(0x157, 3, vgpr(0), vgpr(1), vgpr(2)), vec![S_ENDPGM]].concat());
    assert!(!native_ftz_applies(&med3, FULL_FLUSH, float_rules::ALL), "v_med3_f32 keeps its flush");
    // min3 / max3 compare flushed operands, as the host does.
    let max3 = program(&[vop3(0x154, 3, vgpr(0), vgpr(1), vgpr(2)), vec![S_ENDPGM]].concat());
    assert!(native_ftz_applies(&max3, FULL_FLUSH, float_rules::ALL));
}

#[test]
fn native_translation_drops_the_flush_code() {
    let count = |f: &Function| f.blocks.iter().map(|b| b.insts.len()).sum::<usize>();
    let mut run = ComputeRun::new(ADD.to_vec(), 32, 1, vec![0; 8]);
    run.float_mode = FULL_FLUSH;
    let emulated = run.compile(false).expect("emulated");
    run.native_ftz = true;
    let native = run.compile(false).expect("native");
    assert!(
        count(&native) < count(&emulated),
        "native {} instructions, emulated {} (the flush selects are gone)",
        count(&native),
        count(&emulated)
    );
    // A module that keeps its f32 denormals ignores the flag: nothing to
    // hand to the host.
    run.native_ftz = false;
    run.float_mode = KEEP;
    let kept = run.compile(false).expect("kept");
    run.native_ftz = true;
    let kept_native = run.compile(false).expect("kept native");
    assert_eq!(count(&kept), count(&kept_native), "no flush code to remove in the keep mode");
}
