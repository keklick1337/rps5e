// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Reference compute cases: Control flow, wave64, flat/global/scratch and multi-work-item dispatches.
//!
//! Each test compiles the case, runs it in the interpreter (unoptimized
//! and optimized) and compares the result buffer; see
//! `vectors/ref_harness.rs` for the buffer and user-data mapping.

#[path = "vectors/ref_harness.rs"]
mod harness;

use harness::*;

include!("vectors/ref_control.rs");

#[test]
fn wave64_store_completion_publication() {
    run(&WAVE64_STORE_COMPLETION_PUBLICATION);
}

#[test]
fn branch_select() {
    run(&BRANCH_SELECT);
}

#[test]
fn simple_loop() {
    run(&SIMPLE_LOOP);
}

#[test]
fn shared_return_keeps_selected_values() {
    run(&SHARED_RETURN_KEEPS_SELECTED_VALUES);
}

#[test]
fn sibling_shared_exit_keeps_captured_conditions() {
    run(&SIBLING_SHARED_EXIT_KEEPS_CAPTURED_CONDITIONS);
}

#[test]
fn branch_vccnz_uses_wave_mask() {
    run(&BRANCH_VCCNZ_USES_WAVE_MASK);
}

#[test]
fn branch_vccnz_uses_carry_produced_wave_mask() {
    run(&BRANCH_VCCNZ_USES_CARRY_PRODUCED_WAVE_MASK);
}

#[test]
fn flat_subdword_loads_apply_byte_offset() {
    run(&FLAT_SUBDWORD_LOADS_APPLY_BYTE_OFFSET);
}

#[test]
fn global_load_short_d16_captured_wave32() {
    run(&GLOBAL_LOAD_SHORT_D16_CAPTURED_WAVE32);
}

#[test]
fn global_load_short_d16_captured_wave64() {
    run(&GLOBAL_LOAD_SHORT_D16_CAPTURED_WAVE64);
}

#[test]
fn flat_load_short_d16_address_segments() {
    run(&FLAT_LOAD_SHORT_D16_ADDRESS_SEGMENTS);
}

#[test]
fn global_signed_immediate_rebases_before_saddr() {
    run(&GLOBAL_SIGNED_IMMEDIATE_REBASES_BEFORE_SADDR);
}

#[test]
fn flat_segment_ignores_saddr_and_masks_offset_msb() {
    run(&FLAT_SEGMENT_IGNORES_SADDR_AND_MASKS_OFFSET_MSB);
}

#[test]
fn scratch_is_private_per_invocation() {
    run(&SCRATCH_IS_PRIVATE_PER_INVOCATION);
}

#[test]
fn wave64_raw_masks_and_scalar_branch() {
    run(&WAVE64_RAW_MASKS_AND_SCALAR_BRANCH);
}

#[test]
fn wave64_partial_multidimensional_workgroup() {
    run(&WAVE64_PARTIAL_MULTIDIMENSIONAL_WORKGROUP);
}

#[test]
fn compute_tg_size_sgpr_uses_wave_metadata() {
    run(&COMPUTE_TG_SIZE_SGPR_USES_WAVE_METADATA);
}

#[test]
fn multiple_workitems_global_id() {
    run(&MULTIPLE_WORKITEMS_GLOBAL_ID);
}

#[test]
fn dispatcher_irreducible_control_flow() {
    run(&DISPATCHER_IRREDUCIBLE_CONTROL_FLOW);
}

#[test]
fn loop_selection_arms_leave_the_iteration() {
    run(&LOOP_SELECTION_ARMS_LEAVE_THE_ITERATION);
}
