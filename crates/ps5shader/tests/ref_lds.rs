// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Reference compute cases: LDS data share (DS reads/writes, atomics, permutes, append/consume).
//!
//! Each test compiles the case, runs it in the interpreter (unoptimized
//! and optimized) and compares the result buffer; see
//! `vectors/ref_harness.rs` for the buffer and user-data mapping.

#[path = "vectors/ref_harness.rs"]
mod harness;

use harness::*;

include!("vectors/ref_lds.rs");

#[test]
fn ds_read_write_variants() {
    run(&DS_READ_WRITE_VARIANTS);
}

#[test]
fn ds_write_b16_d16_hi_captured_uses_high_half() {
    run(&DS_WRITE_B16_D16_HI_CAPTURED_USES_HIGH_HALF);
}

#[test]
fn ds_write_b8_d16_hi_writes_byte_two() {
    run(&DS_WRITE_B8_D16_HI_WRITES_BYTE_TWO);
}

#[test]
fn ds_read_u16_d16_captured_preserves_high_half() {
    run(&DS_READ_U16_D16_CAPTURED_PRESERVES_HIGH_HALF);
}

#[test]
fn ds_read_u16_d16_hi_captured_preserves_low_half() {
    run(&DS_READ_U16_D16_HI_CAPTURED_PRESERVES_LOW_HALF);
}

#[test]
fn ds_append_consume_lds_selector() {
    run(&DS_APPEND_CONSUME_LDS_SELECTOR);
}

#[test]
fn ds_read_write2_variants() {
    run(&DS_READ_WRITE2_VARIANTS);
}

#[test]
fn ds_wide_read_snapshots_overlapping_address() {
    run(&DS_WIDE_READ_SNAPSHOTS_OVERLAPPING_ADDRESS);
}

#[test]
fn ds_read_write2_equal_offsets_use_data0() {
    run(&DS_READ_WRITE2_EQUAL_OFFSETS_USE_DATA0);
}

#[test]
fn ds_wide_lds_partial_bounds() {
    run(&DS_WIDE_LDS_PARTIAL_BOUNDS);
}

#[test]
fn ds_atomic_no_return_variants() {
    run(&DS_ATOMIC_NO_RETURN_VARIANTS);
}

#[test]
#[ignore = "case wrong: it wraps LDS addresses at 64 KiB; the reference driver's execution test LdsCondxchg (hw_mem_bases lds_condxchg) has an address 64 KiB past an in-range one out of range (reads 0, writes dropped), as its recompiler bounds the unwrapped address"]
fn ds_masked_or_lds_bounds() {
    run(&DS_MASKED_OR_LDS_BOUNDS);
}

#[test]
fn ds_masked_or_contention_wave32() {
    run(&DS_MASKED_OR_CONTENTION_WAVE32);
}

#[test]
fn ds_masked_or_contention_wave64() {
    run(&DS_MASKED_OR_CONTENTION_WAVE64);
}

#[test]
fn ds_atomic_return_variants() {
    run(&DS_ATOMIC_RETURN_VARIANTS);
}

#[test]
#[ignore = "case wrong: it wraps LDS addresses at 64 KiB; the reference driver's execution test LdsCondxchg (hw_mem_bases lds_condxchg) has an address 64 KiB past an in-range one out of range (reads 0, writes dropped), as its recompiler bounds the unwrapped address"]
fn ds_or_b64_bounds() {
    run(&DS_OR_B64_BOUNDS);
}

#[test]
fn ds_or_b64_contention_wave32() {
    run(&DS_OR_B64_CONTENTION_WAVE32);
}

#[test]
fn ds_or_b64_contention_wave64() {
    run(&DS_OR_B64_CONTENTION_WAVE64);
}

#[test]
#[ignore = "case wrong: it wraps LDS addresses at 64 KiB; the reference driver's execution test LdsCondxchg (hw_mem_bases lds_condxchg) has an address 64 KiB past an in-range one out of range (reads 0, writes dropped), as its recompiler bounds the unwrapped address"]
fn ds_add_u64_bounds() {
    run(&DS_ADD_U64_BOUNDS);
}

#[test]
fn ds_inc_return_lds_boundaries() {
    run(&DS_INC_RETURN_LDS_BOUNDARIES);
}

#[test]
fn ds_inc_return_lds_contention_wave32() {
    run(&DS_INC_RETURN_LDS_CONTENTION_WAVE32);
}

#[test]
fn ds_inc_return_lds_contention_wave64() {
    run(&DS_INC_RETURN_LDS_CONTENTION_WAVE64);
}

#[test]
fn ds_dec_return_lds_boundaries() {
    run(&DS_DEC_RETURN_LDS_BOUNDARIES);
}

#[test]
fn ds_dec_return_lds_contention_wave32() {
    run(&DS_DEC_RETURN_LDS_CONTENTION_WAVE32);
}

#[test]
fn ds_dec_return_lds_contention_wave64() {
    run(&DS_DEC_RETURN_LDS_CONTENTION_WAVE64);
}

#[test]
fn ds_misc_variants() {
    run(&DS_MISC_VARIANTS);
}

#[test]
fn ds_float_min_max_ignores_data1() {
    run(&DS_FLOAT_MIN_MAX_IGNORES_DATA1);
}

#[test]
fn ds_float_min_max_lds_classes() {
    run(&DS_FLOAT_MIN_MAX_LDS_CLASSES);
}

#[test]
fn ds_swizzle_invalid_source_lane_zero() {
    run(&DS_SWIZZLE_INVALID_SOURCE_LANE_ZERO);
}

#[test]
fn ds_permute_captured_exec_offset_and_wrap() {
    run(&DS_PERMUTE_CAPTURED_EXEC_OFFSET_AND_WRAP);
}

#[test]
fn ds_permute_wave64_uses_independent_halves() {
    run(&DS_PERMUTE_WAVE64_USES_INDEPENDENT_HALVES);
}

#[test]
fn ds_bpermute_captured_exec_offset_and_wrap() {
    run(&DS_BPERMUTE_CAPTURED_EXEC_OFFSET_AND_WRAP);
}

#[test]
fn ds_bpermute_wave64_uses_independent_halves() {
    run(&DS_BPERMUTE_WAVE64_USES_INDEPENDENT_HALVES);
}
