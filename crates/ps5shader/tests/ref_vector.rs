// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Reference compute cases: Vector ALU (VOP1/VOP2/VOP3/VOP3P/VOPC, SDWA, DPP, 16/32/64-bit float and integer).
//!
//! Each test compiles the case, runs it in the interpreter (unoptimized
//! and optimized) and compares the result buffer; see
//! `vectors/ref_harness.rs` for the buffer and user-data mapping.

#[path = "vectors/ref_harness.rs"]
mod harness;

use harness::*;

include!("vectors/ref_vector.rs");
include!("vectors/ref_int16.rs");

#[test]
fn wave32_store_completion_publication() {
    run(&WAVE32_STORE_COMPLETION_PUBLICATION);
}

#[test]
fn vector_dynamic_u64_shift_edges() {
    run(&VECTOR_DYNAMIC_U64_SHIFT_EDGES);
}

#[test]
fn vector_register_moves() {
    run(&VECTOR_REGISTER_MOVES);
}

#[test]
fn vector_vop3_move_applies_float_source_modifiers() {
    run(&VECTOR_VOP3_MOVE_APPLIES_FLOAT_SOURCE_MODIFIERS);
}

#[test]
fn vector_integer_ops() {
    run(&VECTOR_INTEGER_OPS);
}

#[test]
fn vector_ffbh_i32_native_and_vop3_on_gpu() {
    run(&VECTOR_FFBH_I32_NATIVE_AND_VOP3_ON_GPU);
}

#[test]
fn vop1_sdwa_bfrev_source_selectors() {
    run(&VOP1_SDWA_BFREV_SOURCE_SELECTORS);
}

#[test]
fn vop1_sdwa_ffbh_captured_scalar_low_word_source() {
    run(&VOP1_SDWA_FFBH_CAPTURED_SCALAR_LOW_WORD_SOURCE);
}

#[test]
fn vop1_sdwa_ffbl_captured_high_word_source() {
    run(&VOP1_SDWA_FFBL_CAPTURED_HIGH_WORD_SOURCE);
}

#[test]
fn vop1_sdwa_not_captured_byte0_source() {
    run(&VOP1_SDWA_NOT_CAPTURED_BYTE0_SOURCE);
}

#[test]
fn vop1_sdwa_not_preserves_high_word_destination() {
    run(&VOP1_SDWA_NOT_PRESERVES_HIGH_WORD_DESTINATION);
}

#[test]
fn vop1_sdwa_not_partial_sources_and_destinations() {
    run(&VOP1_SDWA_NOT_PARTIAL_SOURCES_AND_DESTINATIONS);
}

#[test]
fn vop1_sdwa_mov_byte_destinations() {
    run(&VOP1_SDWA_MOV_BYTE_DESTINATIONS);
}

#[test]
fn vop1_sdwa_mov_byte_sources_wave32() {
    run(&VOP1_SDWA_MOV_BYTE_SOURCES_WAVE32);
}

#[test]
fn vop1_sdwa_mov_byte_sources_wave64() {
    run(&VOP1_SDWA_MOV_BYTE_SOURCES_WAVE64);
}

#[test]
fn vop2_sdwa_sub_nc_exact_byte2_destination() {
    run(&VOP2_SDWA_SUB_NC_EXACT_BYTE2_DESTINATION);
}

#[test]
fn vop2_sdwa_add_nc_captured_high_word_destination() {
    run(&VOP2_SDWA_ADD_NC_CAPTURED_HIGH_WORD_DESTINATION);
}

#[test]
fn vop2_sdwa_ashrrev_captured_word0_sign_extends() {
    run(&VOP2_SDWA_ASHRREV_CAPTURED_WORD0_SIGN_EXTENDS);
}

#[test]
fn vop2_sdwa_max_i32_high_word_wave32() {
    run(&VOP2_SDWA_MAX_I32_HIGH_WORD_WAVE32);
}

#[test]
fn vop2_sdwa_max_i32_high_word_wave64() {
    run(&VOP2_SDWA_MAX_I32_HIGH_WORD_WAVE64);
}

#[test]
fn vop2_sdwa_mul_i24_destinations_wave32() {
    run(&VOP2_SDWA_MUL_I24_DESTINATIONS_WAVE32);
}

#[test]
fn vop2_sdwa_mul_i24_destinations_wave64() {
    run(&VOP2_SDWA_MUL_I24_DESTINATIONS_WAVE64);
}

#[test]
fn vop2_sdwa_mul_u24_destinations_wave32() {
    run(&VOP2_SDWA_MUL_U24_DESTINATIONS_WAVE32);
}

#[test]
fn vop2_sdwa_mul_u24_destinations_wave64() {
    run(&VOP2_SDWA_MUL_U24_DESTINATIONS_WAVE64);
}

#[test]
fn vop2_sdwa_lshrrev_captured_byte1_source() {
    run(&VOP2_SDWA_LSHRREV_CAPTURED_BYTE1_SOURCE);
}

#[test]
fn vop2_sdwa_sub_nc_preserves_byte_and_word_destinations() {
    run(&VOP2_SDWA_SUB_NC_PRESERVES_BYTE_AND_WORD_DESTINATIONS);
}

#[test]
#[ignore = "case wrong: v_cvt_pk_i16_i32 saturates (hardware rows hw_valu_vop3_integer cvt_pk_i16_i32: (65536, -65536) -> 0x80007fff); the case truncates"]
fn vop3_cvt_pk_i16_i32_captured() {
    run(&VOP3_CVT_PK_I16_I32_CAPTURED);
}

#[test]
fn vop3_mul_lo_u16_captured_and_selectors() {
    run(&VOP3_MUL_LO_U16_CAPTURED_AND_SELECTORS);
}

#[test]
fn vop3_mad_i16_captured_selectors_and_saturation() {
    run(&VOP3_MAD_I16_CAPTURED_SELECTORS_AND_SATURATION);
}

#[test]
fn vop3_min3_u16_captured_and_selectors() {
    run(&VOP3_MIN3_U16_CAPTURED_AND_SELECTORS);
}

#[test]
fn vop3_med3_i16_captured() {
    run(&VOP3_MED3_I16_CAPTURED);
}

#[test]
fn vop2_sdwa_min_u32_preserves_word_destination() {
    run(&VOP2_SDWA_MIN_U32_PRESERVES_WORD_DESTINATION);
}

#[test]
fn vector_shift_counts_mask_low_bits() {
    run(&VECTOR_SHIFT_COUNTS_MASK_LOW_BITS);
}

#[test]
fn vector_vop3_lshrrev_b64_captured() {
    run(&VECTOR_VOP3_LSHRREV_B64_CAPTURED);
}

#[test]
fn vector_vop3_lshlrev_b64_captured() {
    run(&VECTOR_VOP3_LSHLREV_B64_CAPTURED);
}

#[test]
fn vector_vop3_integer_ops() {
    run(&VECTOR_VOP3_INTEGER_OPS);
}

#[test]
fn vector_bfe_i32_sign_extends_field() {
    run(&VECTOR_BFE_I32_SIGN_EXTENDS_FIELD);
}

#[test]
fn vector_align_byte_uses_two_bit_byte_offset() {
    run(&VECTOR_ALIGN_BYTE_USES_TWO_BIT_BYTE_OFFSET);
}

#[test]
fn vector_carry_and_bit_count_ops() {
    run(&VECTOR_CARRY_AND_BIT_COUNT_OPS);
}

#[test]
fn vector_mbcnt_uses_thread_mask() {
    run(&VECTOR_MBCNT_USES_THREAD_MASK);
}

#[test]
fn vector_addc_writes_per_lane_carry_out() {
    run(&VECTOR_ADDC_WRITES_PER_LANE_CARRY_OUT);
}

#[test]
fn vector_addc_uses_per_lane_carry_in() {
    run(&VECTOR_ADDC_USES_PER_LANE_CARRY_IN);
}

#[test]
fn vector_sub_co_ci_u32_compact_and_vop3() {
    run(&VECTOR_SUB_CO_CI_U32_COMPACT_AND_VOP3);
}

#[test]
fn vector_subrev_co_ci_u32_exact_raw_on_gpu() {
    run(&VECTOR_SUBREV_CO_CI_U32_EXACT_RAW_ON_GPU);
}

#[test]
fn vector_vop3_b_subrev_co_ci_uses_encoded_masks() {
    run(&VECTOR_VOP3_B_SUBREV_CO_CI_USES_ENCODED_MASKS);
}

#[test]
fn vector_vop3_b_carry_out_writes_sgpr_mask() {
    run(&VECTOR_VOP3_B_CARRY_OUT_WRITES_SGPR_MASK);
}

#[test]
fn vector_vop3_b_carry_out_uses_encoded_sdst() {
    run(&VECTOR_VOP3_B_CARRY_OUT_USES_ENCODED_SDST);
}

#[test]
fn vector_vop3_b_sub_co_u32_uses_rdna2_opcode310() {
    run(&VECTOR_VOP3_B_SUB_CO_U32_USES_RDNA2_OPCODE310);
}

#[test]
fn vector_mad_u64_u32_unsigned_carry_out() {
    run(&VECTOR_MAD_U64_U32_UNSIGNED_CARRY_OUT);
}

#[test]
#[ignore = "case wrong: v_cvt_pk_u16_u32 saturates (hardware rows hw_valu_vop3_integer cvt_pk_u16_u32: (0xffff, 0x10000) -> 0xffffffff); the case truncates (word 11)"]
fn vector_lane_and_packed_ops() {
    run(&VECTOR_LANE_AND_PACKED_OPS);
}

#[test]
fn vop2_pk_fmac_f16_accumulates_packed_halves_independently() {
    run(&VOP2_PK_FMAC_F16_ACCUMULATES_PACKED_HALVES_INDEPENDENTLY);
}

#[test]
fn vop2_pk_fmac_f16_dpp_negates_both_packed_halves() {
    run(&VOP2_PK_FMAC_F16_DPP_NEGATES_BOTH_PACKED_HALVES);
}

#[test]
fn vop3p_opsel_hi_uses_architectural_source_bits() {
    run(&VOP3P_OPSEL_HI_USES_ARCHITECTURAL_SOURCE_BITS);
}

/// neg_lo / neg_hi flip bit 15 of each 16-bit integer half (hardware rows:
/// `hw_valu_packed::packed_integer_negate`).
#[test]
fn vop3p_integer_negation_captured_and_selected_halves() {
    run(&VOP3P_INTEGER_NEGATION_CAPTURED_AND_SELECTED_HALVES);
}

#[test]
#[ignore = "case wrong: v_cvt_pk_u8_f32 rounds to nearest even (hardware rows hw_valu_vop3_float legacy source modifiers: 1.5 -> 2); the case truncates 12.75 to 12"]
fn cvt_pk_u8_f32_packs_selected_byte() {
    run(&CVT_PK_U8_F32_PACKS_SELECTED_BYTE);
}

#[test]
fn cvt_pkrtz_f16_f32_subnormal_rounds_toward_zero() {
    run(&CVT_PKRTZ_F16_F32_SUBNORMAL_ROUNDS_TOWARD_ZERO);
}

#[test]
#[ignore = "case wrong: hardware rows (hw_fmod_f16_result_modifiers pkrtz_clamp / pkrtz_div2) ignore clamp and omod on v_cvt_pkrtz_f16_f32"]
fn cvt_pkrtz_f16_f32_sdwa_and_output_modifiers() {
    run(&CVT_PKRTZ_F16_F32_SDWA_AND_OUTPUT_MODIFIERS);
}

#[test]
#[ignore = "case wrong: with IEEE_MODE clear v_pk_min/max_f16 return the number for a signaling-NaN operand (hardware rows hw_valu_packed packed_alu IEEE off: pk_min/pk_max(0x7d23, 0x3555) = 0x3555, (0x7c01, 0xc000) = 0xc000); the case quiets the NaN (words 2/3)"]
fn packed_min_max_f16_nan_and_signed_zero_edges() {
    run(&PACKED_MIN_MAX_F16_NAN_AND_SIGNED_ZERO_EDGES);
}

#[test]
fn vector_min_max_f16_ops() {
    run(&VECTOR_MIN_MAX_F16_OPS);
}

#[test]
fn vector_cvt_f16_u16_byte_source() {
    run(&VECTOR_CVT_F16_U16_BYTE_SOURCE);
}

#[test]
fn vector_cvt_u16_f16_sdwa() {
    run(&VECTOR_CVT_U16_F16_SDWA);
}

#[test]
#[ignore = "case wrong: in FLOAT_MODE 0xC0 (f16 denormals kept) omod does not apply to f16 results (hardware rows hw_fmod_f16_result_modifiers cvt_f16_f32_div2 / rcp_mul2); the case doubles words 16 and 18"]
fn native_and_sdwa16_bit_destination_writes() {
    run(&NATIVE_AND_SDWA16_BIT_DESTINATION_WRITES);
}

#[test]
fn vector_min_max_med3_f16_ops() {
    run(&VECTOR_MIN_MAX_MED3_F16_OPS);
}

#[test]
fn vector_special_f16_ops() {
    run(&VECTOR_SPECIAL_F16_OPS);
}

#[test]
#[ignore = "case wrong: v_fract_f16 clamps to 0x3bff (hardware rows hw_valu_vop1 fract_f16 of 0x8400 = 0x3bff) and, with IEEE_MODE clear, returns a signaling NaN unquieted (NaN quieting rows)"]
fn vector_fract_f16_captured_and_edges() {
    run(&VECTOR_FRACT_F16_CAPTURED_AND_EDGES);
}

#[test]
#[ignore = "case wrong: in FLOAT_MODE 0xC0 (f16 denormals kept) omod does not apply to f16 results (hardware rows hw_fmod_f16_result_modifiers rcp_mul2 / fma_mul2 / mul_sdwa_mul2)"]
fn vector_fract_f16_modifiers() {
    run(&VECTOR_FRACT_F16_MODIFIERS);
}

#[test]
fn vector_cos_f16_captured_sdwa_and_edges() {
    run(&VECTOR_COS_F16_CAPTURED_SDWA_AND_EDGES);
}

#[test]
fn vector_sin_f16_sdwa_and_edges() {
    run(&VECTOR_SIN_F16_SDWA_AND_EDGES);
}

#[test]
fn vector_writelane_ignores_exec_mask() {
    run(&VECTOR_WRITELANE_IGNORES_EXEC_MASK);
}

#[test]
fn vector_readlane_inactive_written_lane() {
    run(&VECTOR_READLANE_INACTIVE_WRITTEN_LANE);
}

#[test]
fn vector_lane_wave32_runtime_selector_wraps() {
    run(&VECTOR_LANE_WAVE32_RUNTIME_SELECTOR_WRAPS);
}

#[test]
fn vector_readlane_selects_two_keys_within_wave() {
    run(&VECTOR_READLANE_SELECTS_TWO_KEYS_WITHIN_WAVE);
}

#[test]
fn vector_permlanex16() {
    run(&VECTOR_PERMLANEX16);
}

#[test]
fn vector_permlane16_fetch_inactive_zero() {
    run(&VECTOR_PERMLANE16_FETCH_INACTIVE_ZERO);
}

#[test]
fn vector_permlane16_fetch_inactive_fi() {
    run(&VECTOR_PERMLANE16_FETCH_INACTIVE_FI);
}

#[test]
fn vector_dpp8_captured() {
    run(&VECTOR_DPP8_CAPTURED);
}

#[test]
fn vector_dpp8_captured_masked_exec() {
    run(&VECTOR_DPP8_CAPTURED_MASKED_EXEC);
}

#[test]
fn vector_dpp_quad_permute_reverse() {
    run(&VECTOR_DPP_QUAD_PERMUTE_REVERSE);
}

#[test]
fn vector_dpp_row_xmask() {
    run(&VECTOR_DPP_ROW_XMASK);
}

#[test]
fn vector_dpp_bank_mask_preserves_destination() {
    run(&VECTOR_DPP_BANK_MASK_PRESERVES_DESTINATION);
}

#[test]
fn vector_dpp_bounds_control_zero_preserves_destination() {
    run(&VECTOR_DPP_BOUNDS_CONTROL_ZERO_PRESERVES_DESTINATION);
}

#[test]
fn vop3_fmac_f32_negated_source_accumulates() {
    run(&VOP3_FMAC_F32_NEGATED_SOURCE_ACCUMULATES);
}

#[test]
fn vector_mad_f32_rounds_product() {
    run(&VECTOR_MAD_F32_ROUNDS_PRODUCT);
}

#[test]
fn vector_mad_f32_flushes_denorms() {
    run(&VECTOR_MAD_F32_FLUSHES_DENORMS);
}

#[test]
fn vop3_ldexp_source_modifier() {
    run(&VOP3_LDEXP_SOURCE_MODIFIER);
}

#[test]
fn vop1_move_rel_source() {
    run(&VOP1_MOVE_REL_SOURCE);
}

#[test]
fn vop1_move_rel_destination() {
    run(&VOP1_MOVE_REL_DESTINATION);
}

#[test]
fn vector_float_special_ops() {
    run(&VECTOR_FLOAT_SPECIAL_OPS);
}

#[test]
fn cube_id_captured_negation_and_output_scale() {
    run(&CUBE_ID_CAPTURED_NEGATION_AND_OUTPUT_SCALE);
}

#[test]
fn mad_mix_f16_literal_half_source_uses_opsel() {
    run(&MAD_MIX_F16_LITERAL_HALF_SOURCE_USES_OPSEL);
}

#[test]
fn mad_mix_f16_neg_hi_is_abs_and_neg_is_independent() {
    run(&MAD_MIX_F16_NEG_HI_IS_ABS_AND_NEG_IS_INDEPENDENT);
}

#[test]
fn vector_vop3_fma_f16_uses_rdna2_opcode34b() {
    run(&VECTOR_VOP3_FMA_F16_USES_RDNA2_OPCODE34B);
}

#[test]
fn float_inline_constant_f16_uses_numeric_value() {
    run(&FLOAT_INLINE_CONSTANT_F16_USES_NUMERIC_VALUE);
}

#[test]
fn vector_float_control_contract_preserves_inf_nan() {
    run(&VECTOR_FLOAT_CONTROL_CONTRACT_PRESERVES_INF_NAN);
}

#[test]
fn vector_float_arithmetic_ops() {
    run(&VECTOR_FLOAT_ARITHMETIC_OPS);
}

#[test]
fn vector_min_max_f32_nan_and_signed_zero_edges() {
    run(&VECTOR_MIN_MAX_F32_NAN_AND_SIGNED_ZERO_EDGES);
}

#[test]
fn vector_med3_f32_nan_uses_min3_path() {
    run(&VECTOR_MED3_F32_NAN_USES_MIN3_PATH);
}

#[test]
fn vector_float_conversion_ops() {
    run(&VECTOR_FLOAT_CONVERSION_OPS);
}

#[test]
#[ignore = "case wrong: in FLOAT_MODE 0xC0 v_frexp_exp/mant flush f32 denormal inputs (hardware rows F32DenormalFlush: frexp_mant(0x00000001) = 0, frexp_exp(0x00400000) = 0); the signaling-NaN word now matches"]
fn vector_frexp_f32_edges() {
    run(&VECTOR_FREXP_F32_EDGES);
}

#[test]
fn cvt_f32_to_int_saturates_na_n_and_out_of_range() {
    run(&CVT_F32_TO_INT_SATURATES_NA_N_AND_OUT_OF_RANGE);
}

#[test]
fn vector_special_f32_flushes_denormal_inputs() {
    run(&VECTOR_SPECIAL_F32_FLUSHES_DENORMAL_INPUTS);
}

#[test]
fn vector_rcp_iflag_f32_integer_reciprocal() {
    run(&VECTOR_RCP_IFLAG_F32_INTEGER_RECIPROCAL);
}

#[test]
fn vector_compare_f64_edges() {
    run(&VECTOR_COMPARE_F64_EDGES);
}

#[test]
fn vector_compare_integer64_edges() {
    run(&VECTOR_COMPARE_INTEGER64_EDGES);
}

#[test]
fn vector_compare_exec_wave32_masks() {
    run(&VECTOR_COMPARE_EXEC_WAVE32_MASKS);
}

#[test]
fn vector_compare_exec_wave64_masks() {
    run(&VECTOR_COMPARE_EXEC_WAVE64_MASKS);
}

#[test]
fn vector_compare_f64_wave32_masks() {
    run(&VECTOR_COMPARE_F64_WAVE32_MASKS);
}

#[test]
fn vector_compare_f64_wave64_masks() {
    run(&VECTOR_COMPARE_F64_WAVE64_MASKS);
}

#[test]
fn vector_compare_ne_i64_wave32_masks() {
    run(&VECTOR_COMPARE_NE_I64_WAVE32_MASKS);
}

#[test]
fn vector_compare_ne_i64_wave64_masks() {
    run(&VECTOR_COMPARE_NE_I64_WAVE64_MASKS);
}

#[test]
fn vector_f64_captured_screen_space_shadows() {
    run(&VECTOR_F64_CAPTURED_SCREEN_SPACE_SHADOWS);
}

#[test]
#[ignore = "case wrong: v_fract_f64 clamps to 0x3fefffffffffffff (hardware rows hw_fmod_f64_rounding: fract(-2^-60) = 0x3fefffffffffffff)"]
fn vector_fract_f64_captured_and_edges() {
    run(&VECTOR_FRACT_F64_CAPTURED_AND_EDGES);
}

#[test]
#[ignore = "case wrong: v_min/max_f64 return a signaling NaN operand quieted even with IEEE_MODE clear (hardware rows hw_fmod_f64_arithmetic: min(1.0, 0x7ff0000000000001) = 0x7ff8000000000001)"]
fn vector_min_max_f64_captured_and_edges() {
    run(&VECTOR_MIN_MAX_F64_CAPTURED_AND_EDGES);
}

#[test]
fn vector_round_f64_edges_and_modifiers() {
    run(&VECTOR_ROUND_F64_EDGES_AND_MODIFIERS);
}

#[test]
fn vector_fract_f64_captured_chain_runtime_exec() {
    run(&VECTOR_FRACT_F64_CAPTURED_CHAIN_RUNTIME_EXEC);
}

#[test]
#[ignore = "case wrong on words 106/107 (finite v_cvt_f32_f64 overflow rounds to infinity as IEEE round-to-nearest; OpenCL conversions conformance relies on it); unverified on words 110/111 (f32 denormal result of v_cvt_f32_f64 in FLOAT_MODE 0xC0: no hardware row)"]
fn vector_f64_modes_modifiers_and_exec() {
    run(&VECTOR_F64_MODES_MODIFIERS_AND_EXEC);
}

#[test]
#[ignore = "case wrong: with IEEE_MODE clear v_cvt_f64_f32 returns a signaling NaN unquieted (hardware rows hw_valu_float_mode f64_nan_quieting rounding_ieee_off: cvt_f64_f32(0x7f800001) = 0x7ff0000020000000); the case quiets 0x7f812345 (word 23)"]
fn vector_f64_widening_conversions() {
    run(&VECTOR_F64_WIDENING_CONVERSIONS);
}

#[test]
fn vector_sin_cos_max_finite_special_cases() {
    run(&VECTOR_SIN_COS_MAX_FINITE_SPECIAL_CASES);
}

#[test]
fn vector_compare_f32_exposure_guard() {
    run(&VECTOR_COMPARE_F32_EXPOSURE_GUARD);
}

#[test]
fn vector_compare_f32_denormal_mode_c0() {
    run(&VECTOR_COMPARE_F32_DENORMAL_MODE_C0);
}

#[test]
fn vector_compare_f32_denormal_mode_e0() {
    run(&VECTOR_COMPARE_F32_DENORMAL_MODE_E0);
}

#[test]
fn vector_compare_ops() {
    run(&VECTOR_COMPARE_OPS);
}

#[test]
fn vector_vop3_compare_eq_i64_on_gpu() {
    run(&VECTOR_VOP3_COMPARE_EQ_I64_ON_GPU);
}

#[test]
fn vector_vop3_compare_eq_u64_on_gpu() {
    run(&VECTOR_VOP3_COMPARE_EQ_U64_ON_GPU);
}

#[test]
fn vector_vop3_compare_gt_u64_on_gpu() {
    run(&VECTOR_VOP3_COMPARE_GT_U64_ON_GPU);
}

#[test]
fn vector_vopc_compare_lt_u64_on_gpu() {
    run(&VECTOR_VOPC_COMPARE_LT_U64_ON_GPU);
}

#[test]
fn vector_vop3_compare_ne_u64_on_gpu() {
    run(&VECTOR_VOP3_COMPARE_NE_U64_ON_GPU);
}

#[test]
fn vector_vopc_cmpx_ne_u64_captured_exec_mask() {
    run(&VECTOR_VOPC_CMPX_NE_U64_CAPTURED_EXEC_MASK);
}

#[test]
fn vector_vop3_cmpx_ne_i64_captured_exec_mask() {
    run(&VECTOR_VOP3_CMPX_NE_I64_CAPTURED_EXEC_MASK);
}

#[test]
fn vector_compare_class_f32() {
    run(&VECTOR_COMPARE_CLASS_F32);
}

#[test]
fn vector_compare_class_f32_immediate_and_dynamic_masks() {
    run(&VECTOR_COMPARE_CLASS_F32_IMMEDIATE_AND_DYNAMIC_MASKS);
}

#[test]
fn vector_integer_extrema_runtime_edges() {
    run(&VECTOR_INTEGER_EXTREMA_RUNTIME_EDGES);
}

#[test]
fn vector_vopc_sdwa_cmpx_class_f32_captured_exec_mask() {
    run(&VECTOR_VOPC_SDWA_CMPX_CLASS_F32_CAPTURED_EXEC_MASK);
}

#[test]
fn vector_cmp_class_f16_wave32() {
    run(&VECTOR_CMP_CLASS_F16_WAVE32);
}

#[test]
fn vector_cmp_class_f16_wave64() {
    run(&VECTOR_CMP_CLASS_F16_WAVE64);
}

#[test]
fn vector_cmpx_class_f16_wave32() {
    run(&VECTOR_CMPX_CLASS_F16_WAVE32);
}

#[test]
fn vector_cmpx_class_f16_wave64() {
    run(&VECTOR_CMPX_CLASS_F16_WAVE64);
}

#[test]
fn vector_compare_f16_ops() {
    run(&VECTOR_COMPARE_F16_OPS);
}

#[test]
fn wave32_vcc_masks_preserve_other_half() {
    run(&WAVE32_VCC_MASKS_PRESERVE_OTHER_HALF);
}

#[test]
fn vop2_sdwa_cndmask_source_modifier() {
    run(&VOP2_SDWA_CNDMASK_SOURCE_MODIFIER);
}

#[test]
fn vop2_sdwa_cndmask_captured_byte3_source() {
    run(&VOP2_SDWA_CNDMASK_CAPTURED_BYTE3_SOURCE);
}

#[test]
fn vop3_cndmask_uses_sgpr_mask_lane_bits() {
    run(&VOP3_CNDMASK_USES_SGPR_MASK_LANE_BITS);
}

#[test]
fn vop3_cndmask_allows_data_source_modifier() {
    run(&VOP3_CNDMASK_ALLOWS_DATA_SOURCE_MODIFIER);
}

#[test]
fn vector_compare_exec_ops() {
    run(&VECTOR_COMPARE_EXEC_OPS);
}

#[test]
fn vector_vopc_cmpx_ordered_captured_exec_mask() {
    run(&VECTOR_VOPC_CMPX_ORDERED_CAPTURED_EXEC_MASK);
}

#[test]
fn vector_vop3_float_compare_neg_source_modifier() {
    run(&VECTOR_VOP3_FLOAT_COMPARE_NEG_SOURCE_MODIFIER);
}

#[test]
fn vector_vop3_cmpx_writes_exec_mask() {
    run(&VECTOR_VOP3_CMPX_WRITES_EXEC_MASK);
}

#[test]
fn vector_vopc_sdwa_cmpx_writes_exec_mask() {
    run(&VECTOR_VOPC_SDWA_CMPX_WRITES_EXEC_MASK);
}

#[test]
fn vector_vopc_cmpx_gt_u16_captured_sdwa_exec_mask() {
    run(&VECTOR_VOPC_CMPX_GT_U16_CAPTURED_SDWA_EXEC_MASK);
}

#[test]
fn vector_vopc_cmpx_lt_u16_captured_sdwa_exec_mask() {
    run(&VECTOR_VOPC_CMPX_LT_U16_CAPTURED_SDWA_EXEC_MASK);
}

#[test]
fn vector_vopc_cmpx_lt_i16_captured_sdwa_exec_mask() {
    run(&VECTOR_VOPC_CMPX_LT_I16_CAPTURED_SDWA_EXEC_MASK);
}

#[test]
fn vector_vopc_cmpx_lt_i16_wave32_masks() {
    run(&VECTOR_VOPC_CMPX_LT_I16_WAVE32_MASKS);
}

#[test]
fn vector_vopc_cmpx_lt_i16_wave64_masks() {
    run(&VECTOR_VOPC_CMPX_LT_I16_WAVE64_MASKS);
}

#[test]
fn vector_vopc_cmpx_eq_u16_sdwa_compact_vop3_exec_mask() {
    run(&VECTOR_VOPC_CMPX_EQ_U16_SDWA_COMPACT_VOP3_EXEC_MASK);
}

#[test]
fn vector_vopc_cmpx_ne_u16_sdwa_compact_vop3_exec_mask() {
    run(&VECTOR_VOPC_CMPX_NE_U16_SDWA_COMPACT_VOP3_EXEC_MASK);
}

#[test]
fn vector_vopc_cmp_ngt_f16_captured_sdwa_and_edges() {
    run(&VECTOR_VOPC_CMP_NGT_F16_CAPTURED_SDWA_AND_EDGES);
}

#[test]
fn vector_vopc_cmp_nlt_f16_captured_sdwa_and_edges() {
    run(&VECTOR_VOPC_CMP_NLT_F16_CAPTURED_SDWA_AND_EDGES);
}

#[test]
fn vector_vopc_cmpx_ngt_f16_captured_sdwa_exec_mask() {
    run(&VECTOR_VOPC_CMPX_NGT_F16_CAPTURED_SDWA_EXEC_MASK);
}

#[test]
fn vector_vopc_cmpx_nle_f16_captured_sdwa_exec_mask() {
    run(&VECTOR_VOPC_CMPX_NLE_F16_CAPTURED_SDWA_EXEC_MASK);
}

#[test]
fn vector_compare_inverted_mask_select() {
    run(&VECTOR_COMPARE_INVERTED_MASK_SELECT);
}

#[test]
fn vop3p_int16_ops() {
    run(&VOP3P_INT16_OPS);
}

#[test]
fn vop3_int16_ops() {
    run(&VOP3_INT16_OPS);
}

#[test]
fn vector_compare_i16_u16_ops() {
    run(&VECTOR_COMPARE_I16_U16_OPS);
}
#[test]
fn vop2_sdwa_bitwise_byte_destinations_wave32() {
    run(&VOP2_SDWA_BITWISE_BYTE_DESTINATIONS_WAVE32);
}

#[test]
fn vop2_sdwa_bitwise_byte_destinations_wave64() {
    run(&VOP2_SDWA_BITWISE_BYTE_DESTINATIONS_WAVE64);
}

