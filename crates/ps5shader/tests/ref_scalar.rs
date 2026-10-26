// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Reference compute cases: Scalar ALU (SOP1/SOP2/SOPC/SOPK, lane-mask dataflow, SMEM-free scalar logic).
//!
//! Each test compiles the case, runs it in the interpreter (unoptimized
//! and optimized) and compares the result buffer; see
//! `vectors/ref_harness.rs` for the buffer and user-data mapping.

#[path = "vectors/ref_harness.rs"]
mod harness;

use harness::*;

include!("vectors/ref_scalar.rs");

#[test]
fn integer_add_sub_mul() {
    run(&INTEGER_ADD_SUB_MUL);
}

#[test]
fn bitwise_and_or_xor_not() {
    run(&BITWISE_AND_OR_XOR_NOT);
}

#[test]
fn shifts() {
    run(&SHIFTS);
}

#[test]
fn scalar_zero_shift_with_runtime_count() {
    run(&SCALAR_ZERO_SHIFT_WITH_RUNTIME_COUNT);
}

#[test]
fn exact_push_constant_extent() {
    run(&EXACT_PUSH_CONSTANT_EXTENT);
}

#[test]
fn scalar_shift_counts_mask_low_bits() {
    run(&SCALAR_SHIFT_COUNTS_MASK_LOW_BITS);
}

#[test]
fn rdna2_scalar_opcodes() {
    run(&RDNA2_SCALAR_OPCODES);
}

#[test]
fn scalar_trap_disabled_falls_through_exact_raw() {
    run(&SCALAR_TRAP_DISABLED_FALLS_THROUGH_EXACT_RAW);
}

#[test]
fn scalar_waitcnt_depctr_captured_vm_vsrc() {
    run(&SCALAR_WAITCNT_DEPCTR_CAPTURED_VM_VSRC);
}

#[test]
fn scalar_extended_arithmetic() {
    run(&SCALAR_EXTENDED_ARITHMETIC);
}

#[test]
fn scalar_arithmetic_scc_carry_borrow_overflow() {
    run(&SCALAR_ARITHMETIC_SCC_CARRY_BORROW_OVERFLOW);
}

#[test]
fn scalar_min_max_scc_comparison_edges() {
    run(&SCALAR_MIN_MAX_SCC_COMPARISON_EDGES);
}

#[test]
fn scalar_abs_i32_updates_scc() {
    run(&SCALAR_ABS_I32_UPDATES_SCC);
}

#[test]
fn scalar_absdiff_bitset64_and_setprio() {
    run(&SCALAR_ABSDIFF_BITSET64_AND_SETPRIO);
}

#[test]
fn scalar_shift_left_add_scc_carry_edges() {
    run(&SCALAR_SHIFT_LEFT_ADD_SCC_CARRY_EDGES);
}

#[test]
fn scalar_compare_ops() {
    run(&SCALAR_COMPARE_OPS);
}

#[test]
fn scalar_shift_add_and_mask_ops() {
    run(&SCALAR_SHIFT_ADD_AND_MASK_OPS);
}

#[test]
fn scalar_not_b64_updates_scc() {
    run(&SCALAR_NOT_B64_UPDATES_SCC);
}

#[test]
fn scalar_quadmask_b64() {
    run(&SCALAR_QUADMASK_B64);
}

#[test]
fn scalar_flbit_i32_b64_gpu() {
    run(&SCALAR_FLBIT_I32_B64_GPU);
}

#[test]
fn scalar_flbit_i32_b32_gpu() {
    run(&SCALAR_FLBIT_I32_B32_GPU);
}

#[test]
fn scalar_save_exec_ops() {
    run(&SCALAR_SAVE_EXEC_OPS);
}

#[test]
fn scalar_orn2_saveexec_uses_source_or_not_exec() {
    run(&SCALAR_ORN2_SAVEEXEC_USES_SOURCE_OR_NOT_EXEC);
}

#[test]
fn scalar_orn2_saveexec_b32_wave32() {
    run(&SCALAR_ORN2_SAVEEXEC_B32_WAVE32);
}

#[test]
fn scalar_orn2_saveexec_b32_wave64() {
    run(&SCALAR_ORN2_SAVEEXEC_B32_WAVE64);
}

#[test]
fn scalar_subvector_loops_wave32() {
    run(&SCALAR_SUBVECTOR_LOOPS_WAVE32);
}

#[test]
fn scalar_subvector_loops_wave64() {
    run(&SCALAR_SUBVECTOR_LOOPS_WAVE64);
}

#[test]
fn scalar_bitfield_pack() {
    run(&SCALAR_BITFIELD_PACK);
}

#[test]
fn scalar_bitcmp_b64_dynamic_operands() {
    run(&SCALAR_BITCMP_B64_DYNAMIC_OPERANDS);
}

#[test]
fn scalar_bitcmp_b64_integer_constants() {
    run(&SCALAR_BITCMP_B64_INTEGER_CONSTANTS);
}

#[test]
fn scalar_brev_b32_preserves_scc() {
    run(&SCALAR_BREV_B32_PRESERVES_SCC);
}

#[test]
fn scalar_sext_i8_masks_wave32() {
    run(&SCALAR_SEXT_I8_MASKS_WAVE32);
}

#[test]
fn scalar_sext_i8_masks_wave64() {
    run(&SCALAR_SEXT_I8_MASKS_WAVE64);
}

#[test]
fn scalar_sext_i16_masks_wave32() {
    run(&SCALAR_SEXT_I16_MASKS_WAVE32);
}

#[test]
fn scalar_sext_i16_masks_wave64() {
    run(&SCALAR_SEXT_I16_MASKS_WAVE64);
}

#[test]
fn scalar_sext_constants_and_aliases() {
    run(&SCALAR_SEXT_CONSTANTS_AND_ALIASES);
}

#[test]
fn scalar_brev_b64_operands_and_masks_wave32() {
    run(&SCALAR_BREV_B64_OPERANDS_AND_MASKS_WAVE32);
}

#[test]
fn scalar_brev_b64_operands_and_masks_wave64() {
    run(&SCALAR_BREV_B64_OPERANDS_AND_MASKS_WAVE64);
}

#[test]
fn scalar_bfe_i32_captured_raw_sign_extends() {
    run(&SCALAR_BFE_I32_CAPTURED_RAW_SIGN_EXTENDS);
}

#[test]
fn bitfield_extract_width_past_end_edges() {
    run(&BITFIELD_EXTRACT_WIDTH_PAST_END_EDGES);
}

#[test]
fn scalar64_bit_ops() {
    run(&SCALAR64_BIT_OPS);
}

#[test]
fn scalar_dynamic64_bit_ops() {
    run(&SCALAR_DYNAMIC64_BIT_OPS);
}

#[test]
fn scalar_ashr_i64_constant_edges() {
    run(&SCALAR_ASHR_I64_CONSTANT_EDGES);
}

#[test]
fn scalar_ashr_i64_dynamic_edges() {
    run(&SCALAR_ASHR_I64_DYNAMIC_EDGES);
}

#[test]
fn scalar_ashr_i64_operands_and_aliases() {
    run(&SCALAR_ASHR_I64_OPERANDS_AND_ALIASES);
}

#[test]
fn scalar_ashr_i64_masks_wave32() {
    run(&SCALAR_ASHR_I64_MASKS_WAVE32);
}

#[test]
fn scalar_ashr_i64_masks_wave64() {
    run(&SCALAR_ASHR_I64_MASKS_WAVE64);
}

#[test]
fn scalar_conditional_move_b64() {
    run(&SCALAR_CONDITIONAL_MOVE_B64);
}

#[test]
fn scalar_conditional_move_b64_preserves_masks() {
    run(&SCALAR_CONDITIONAL_MOVE_B64_PRESERVES_MASKS);
}

#[test]
fn scalar_andn2_b64_scc_uses_mask_shadow() {
    run(&SCALAR_ANDN2_B64_SCC_USES_MASK_SHADOW);
}

#[test]
fn scalar_mask_high_write_invalidates_provenance() {
    run(&SCALAR_MASK_HIGH_WRITE_INVALIDATES_PROVENANCE);
}

#[test]
fn scalar_select_b64_preserves_mask_provenance_wave32() {
    run(&SCALAR_SELECT_B64_PRESERVES_MASK_PROVENANCE_WAVE32);
}

#[test]
fn scalar_select_b64_preserves_mask_provenance_wave64() {
    run(&SCALAR_SELECT_B64_PRESERVES_MASK_PROVENANCE_WAVE64);
}

#[test]
fn scalar_wqm_b64_selects_scc_domain() {
    run(&SCALAR_WQM_B64_SELECTS_SCC_DOMAIN);
}

#[test]
fn scalar_wqm_b64_preserves_partial_masks() {
    run(&SCALAR_WQM_B64_PRESERVES_PARTIAL_MASKS);
}

#[test]
fn scalar_wqm_b32_masks_wave32() {
    run(&SCALAR_WQM_B32_MASKS_WAVE32);
}

#[test]
fn scalar_wqm_b32_masks_wave64() {
    run(&SCALAR_WQM_B32_MASKS_WAVE64);
}

#[test]
fn scalar_mask_provenance_overlap_and_mixed_binary() {
    run(&SCALAR_MASK_PROVENANCE_OVERLAP_AND_MIXED_BINARY);
}

#[test]
fn scalar_b32_mask_dataflow_wave32() {
    run(&SCALAR_B32_MASK_DATAFLOW_WAVE32);
}

#[test]
fn scalar_b32_mask_dataflow_wave64() {
    run(&SCALAR_B32_MASK_DATAFLOW_WAVE64);
}

#[test]
fn scalar_literal() {
    run(&SCALAR_LITERAL);
}

#[test]
fn scalar_branch_restores_inactive_wave32() {
    run(&SCALAR_BRANCH_RESTORES_INACTIVE_WAVE32);
}

#[test]
fn scalar_branch_restores_inactive_wave64() {
    run(&SCALAR_BRANCH_RESTORES_INACTIVE_WAVE64);
}

#[test]
fn scalar_branch_restores_partial_wave32() {
    run(&SCALAR_BRANCH_RESTORES_PARTIAL_WAVE32);
}

#[test]
fn scalar_branch_restores_partial_wave64() {
    run(&SCALAR_BRANCH_RESTORES_PARTIAL_WAVE64);
}

#[test]
fn scalar_buffer_offset_alignment_and_carry() {
    run(&SCALAR_BUFFER_OFFSET_ALIGNMENT_AND_CARRY);
}

#[test]
fn scalar_load_signed_immediate_offset_adds_soffset() {
    run(&SCALAR_LOAD_SIGNED_IMMEDIATE_OFFSET_ADDS_SOFFSET);
}

#[test]
fn scalar_load_aligns_components_and_masks_address() {
    run(&SCALAR_LOAD_ALIGNS_COMPONENTS_AND_MASKS_ADDRESS);
}

#[test]
fn scalar_load_aligns_dynamic_base() {
    run(&SCALAR_LOAD_ALIGNS_DYNAMIC_BASE);
}

#[test]
fn scalar_buffer_from_loop_readlane32() {
    run(&SCALAR_BUFFER_FROM_LOOP_READLANE32);
}

#[test]
fn scalar_buffer_from_loop_readlane64() {
    run(&SCALAR_BUFFER_FROM_LOOP_READLANE64);
}
