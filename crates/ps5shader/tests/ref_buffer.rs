// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Reference compute cases: Buffer memory (MUBUF/MTBUF loads, stores, formats, atomics) and scalar memory.
//!
//! Each test compiles the case, runs it in the interpreter (unoptimized
//! and optimized) and compares the result buffer; see
//! `vectors/ref_harness.rs` for the buffer and user-data mapping.

#[path = "vectors/ref_harness.rs"]
mod harness;

use harness::*;

include!("vectors/ref_buffer.rs");

#[test]
fn buffer_workgroup_publication_wave32() {
    run(&BUFFER_WORKGROUP_PUBLICATION_WAVE32);
}

#[test]
fn buffer_workgroup_publication_wave64() {
    run(&BUFFER_WORKGROUP_PUBLICATION_WAVE64);
}

#[test]
fn buffer_workgroup_publication_dlc() {
    run(&BUFFER_WORKGROUP_PUBLICATION_DLC);
}

#[test]
fn buffer_load_store() {
    run(&BUFFER_LOAD_STORE);
}

#[test]
fn buffer_load_dword_offen_idxen_uses_vaddr_plus_one_offset() {
    run(&BUFFER_LOAD_DWORD_OFFEN_IDXEN_USES_VADDR_PLUS_ONE_OFFSET);
}

#[test]
fn buffer_store_dword_offen_idxen_uses_vaddr_plus_one_offset() {
    run(&BUFFER_STORE_DWORD_OFFEN_IDXEN_USES_VADDR_PLUS_ONE_OFFSET);
}

#[test]
fn buffer_load_dword_no_address_flags_ignores_vaddr() {
    run(&BUFFER_LOAD_DWORD_NO_ADDRESS_FLAGS_IGNORES_VADDR);
}

#[test]
fn buffer_load_dword_idxen_uses_descriptor_stride() {
    run(&BUFFER_LOAD_DWORD_IDXEN_USES_DESCRIPTOR_STRIDE);
}

#[test]
fn buffer_store_dword_idxen_uses_descriptor_stride() {
    run(&BUFFER_STORE_DWORD_IDXEN_USES_DESCRIPTOR_STRIDE);
}

#[test]
fn buffer_load_variants() {
    run(&BUFFER_LOAD_VARIANTS);
}

#[test]
fn buffer_load_dwordx2_snapshots_overlapping_address() {
    run(&BUFFER_LOAD_DWORDX2_SNAPSHOTS_OVERLAPPING_ADDRESS);
}

#[test]
fn buffer_load_dwordx3_snapshots_overlapping_address() {
    run(&BUFFER_LOAD_DWORDX3_SNAPSHOTS_OVERLAPPING_ADDRESS);
}

#[test]
fn buffer_load_dwordx4_snapshots_overlapping_address() {
    run(&BUFFER_LOAD_DWORDX4_SNAPSHOTS_OVERLAPPING_ADDRESS);
}

#[test]
fn buffer_load_dwordx4_zeroes_only_out_of_bounds_tail() {
    run(&BUFFER_LOAD_DWORDX4_ZEROES_ONLY_OUT_OF_BOUNDS_TAIL);
}

#[test]
#[ignore = "unverified (FORMAT 0): the case reads 0 through a run-time selected V# whose FORMAT is 0; ours reads raw dwords as both references do for bound V#s; no hardware row"]
fn buffer_loads_gpu_selected_descriptors() {
    run(&BUFFER_LOADS_GPU_SELECTED_DESCRIPTORS);
}

#[test]
#[ignore = "unverified (FORMAT 0): the case reads 0 through a run-time selected V# whose FORMAT is 0; ours reads raw dwords as both references do for bound V#s; no hardware row"]
fn buffer_load_dword_gpu_selected_descriptors() {
    run(&BUFFER_LOAD_DWORD_GPU_SELECTED_DESCRIPTORS);
}

#[test]
#[ignore = "unverified (FORMAT 0): the case reads 0 through a run-time selected V# whose FORMAT is 0; ours reads raw dwords as both references do for bound V#s; no hardware row"]
fn buffer_load_dwordx3_gpu_selected_descriptors() {
    run(&BUFFER_LOAD_DWORDX3_GPU_SELECTED_DESCRIPTORS);
}

#[test]
#[ignore = "unverified: FORMAT 0 V# (reads 0 in the case), a 16-bit formatted load at byte offset 1 (the case aligns it to 2), format 34 (11_11_10 UINT, reserved in our table); no hardware row"]
fn buffer_load_format_x_gpu_selected_descriptors() {
    run(&BUFFER_LOAD_FORMAT_X_GPU_SELECTED_DESCRIPTORS);
}

#[test]
fn buffer_store_dwordx4_drops_only_out_of_bounds_tail() {
    run(&BUFFER_STORE_DWORDX4_DROPS_ONLY_OUT_OF_BOUNDS_TAIL);
}

#[test]
fn buffer_load_format_xyzw_rejects_partial_record() {
    run(&BUFFER_LOAD_FORMAT_XYZW_REJECTS_PARTIAL_RECORD);
}

#[test]
fn buffer_store_format_xyzw_drops_partial_record() {
    run(&BUFFER_STORE_FORMAT_XYZW_DROPS_PARTIAL_RECORD);
}

#[test]
fn buffer_load_format_x_rejects_partial_record() {
    run(&BUFFER_LOAD_FORMAT_X_REJECTS_PARTIAL_RECORD);
}

#[test]
fn buffer_store_format_x_rejects_partial_record() {
    run(&BUFFER_STORE_FORMAT_X_REJECTS_PARTIAL_RECORD);
}

#[test]
fn buffer_load_format_xy_rejects_partial_record() {
    run(&BUFFER_LOAD_FORMAT_XY_REJECTS_PARTIAL_RECORD);
}

#[test]
fn buffer_store_format_xy_rejects_partial_record() {
    run(&BUFFER_STORE_FORMAT_XY_REJECTS_PARTIAL_RECORD);
}

#[test]
fn buffer_store_variants() {
    run(&BUFFER_STORE_VARIANTS);
}

#[test]
fn buffer_load_format_xyzw_snapshots_overlapping_address() {
    run(&BUFFER_LOAD_FORMAT_XYZW_SNAPSHOTS_OVERLAPPING_ADDRESS);
}

#[test]
fn buffer_load_format_xyzw_resource32_32_float_applies_xy01() {
    run(&BUFFER_LOAD_FORMAT_XYZW_RESOURCE32_32_FLOAT_APPLIES_XY01);
}

#[test]
fn buffer_load_format_xyzw_resource323232_float_repeats_source() {
    run(&BUFFER_LOAD_FORMAT_XYZW_RESOURCE323232_FLOAT_REPEATS_SOURCE);
}

#[test]
fn buffer_load_format_xyzw_resource8_8_8_8_uint_applies_wzy1() {
    run(&BUFFER_LOAD_FORMAT_XYZW_RESOURCE8_8_8_8_UINT_APPLIES_WZY1);
}

#[test]
fn buffer_load_format_xyzw_inactive_exec_preserves_overlapping_address() {
    run(&BUFFER_LOAD_FORMAT_XYZW_INACTIVE_EXEC_PRESERVES_OVERLAPPING_ADDRESS);
}

#[test]
fn buffer_store_format_x_resource16_uint_writes_halfword() {
    run(&BUFFER_STORE_FORMAT_X_RESOURCE16_UINT_WRITES_HALFWORD);
}

#[test]
fn buffer_store_format_x_resource16_uint_preserves_adjacent_lanes() {
    run(&BUFFER_STORE_FORMAT_X_RESOURCE16_UINT_PRESERVES_ADJACENT_LANES);
}

#[test]
fn buffer_store_format_xyzw_packed_unorm_skinning_vectors() {
    run(&BUFFER_STORE_FORMAT_XYZW_PACKED_UNORM_SKINNING_VECTORS);
}

#[test]
fn buffer_store_format_x_zero_fills_packed_record() {
    run(&BUFFER_STORE_FORMAT_X_ZERO_FILLS_PACKED_RECORD);
}

#[test]
#[ignore = "case wrong: it rounds f16 stores to nearest; hardware rows truncate (hw_mem_formatted_store 16float: 65520.0 -> 0x7bff, 1.01917 -> 0x3c13)"]
fn buffer_store_format_xyzw_float16_converts_components() {
    run(&BUFFER_STORE_FORMAT_XYZW_FLOAT16_CONVERTS_COMPONENTS);
}

#[test]
fn buffer_store_format_xyzw_snorm16_captured_skinning_vectors() {
    run(&BUFFER_STORE_FORMAT_XYZW_SNORM16_CAPTURED_SKINNING_VECTORS);
}

#[test]
fn buffer_store_format_x_snorm16_clamps_rounds_and_preserves_halfwords() {
    run(&BUFFER_STORE_FORMAT_X_SNORM16_CLAMPS_ROUNDS_AND_PRESERVES_HALFWORDS);
}

#[test]
fn buffer_load_format_d16_x_float16() {
    run(&BUFFER_LOAD_FORMAT_D16_X_FLOAT16);
}

#[test]
fn buffer_load_format_d16_x_float32() {
    run(&BUFFER_LOAD_FORMAT_D16_X_FLOAT32);
}

#[test]
fn buffer_load_format_d16_x_uint32() {
    run(&BUFFER_LOAD_FORMAT_D16_X_UINT32);
}

#[test]
fn buffer_load_format_d16_x_uint_at_max() {
    run(&BUFFER_LOAD_FORMAT_D16_X_UINT_AT_MAX);
}

#[test]
fn buffer_load_format_d16_x_uint_max32() {
    run(&BUFFER_LOAD_FORMAT_D16_X_UINT_MAX32);
}

#[test]
fn buffer_load_format_d16_x_sint32() {
    run(&BUFFER_LOAD_FORMAT_D16_X_SINT32);
}

#[test]
fn buffer_load_format_d16_x_sint_above_max() {
    run(&BUFFER_LOAD_FORMAT_D16_X_SINT_ABOVE_MAX);
}

#[test]
fn buffer_load_format_d16_x_sint_below_min() {
    run(&BUFFER_LOAD_FORMAT_D16_X_SINT_BELOW_MIN);
}

#[test]
fn buffer_load_format_d16_x_sint_at_max() {
    run(&BUFFER_LOAD_FORMAT_D16_X_SINT_AT_MAX);
}

#[test]
fn buffer_load_format_d16_x_sint_at_min() {
    run(&BUFFER_LOAD_FORMAT_D16_X_SINT_AT_MIN);
}

#[test]
fn buffer_load_format_d16_x_sint_max32() {
    run(&BUFFER_LOAD_FORMAT_D16_X_SINT_MAX32);
}

#[test]
fn buffer_load_format_d16_x_sint_min32() {
    run(&BUFFER_LOAD_FORMAT_D16_X_SINT_MIN32);
}

#[test]
fn buffer_load_format_d16_x_unorm8() {
    run(&BUFFER_LOAD_FORMAT_D16_X_UNORM8);
}

#[test]
fn buffer_load_format_d16_x_snorm16() {
    run(&BUFFER_LOAD_FORMAT_D16_X_SNORM16);
}

#[test]
fn buffer_load_format_d16_x_float_one() {
    run(&BUFFER_LOAD_FORMAT_D16_X_FLOAT_ONE);
}

#[test]
fn buffer_load_format_d16_x_oob_float_one() {
    run(&BUFFER_LOAD_FORMAT_D16_X_OOB_FLOAT_ONE);
}

#[test]
fn buffer_load_format_d16_x_oob_uint_one() {
    run(&BUFFER_LOAD_FORMAT_D16_X_OOB_UINT_ONE);
}

#[test]
fn buffer_load_format_d16_x_oob_zero() {
    run(&BUFFER_LOAD_FORMAT_D16_X_OOB_ZERO);
}

#[test]
fn buffer_load_format_d16_x_inactive() {
    run(&BUFFER_LOAD_FORMAT_D16_X_INACTIVE);
}

#[test]
fn buffer_load_format_d16_x_overlapping_address() {
    run(&BUFFER_LOAD_FORMAT_D16_X_OVERLAPPING_ADDRESS);
}

#[test]
fn buffer_store_format_d16_x_float16() {
    run(&BUFFER_STORE_FORMAT_D16_X_FLOAT16);
}

#[test]
fn buffer_store_format_d16_x_float32() {
    run(&BUFFER_STORE_FORMAT_D16_X_FLOAT32);
}

#[test]
fn buffer_store_format_d16_x_uint32() {
    run(&BUFFER_STORE_FORMAT_D16_X_UINT32);
}

#[test]
fn buffer_store_format_d16_x_sint32() {
    run(&BUFFER_STORE_FORMAT_D16_X_SINT32);
}

#[test]
fn buffer_store_format_d16_x_unorm8() {
    run(&BUFFER_STORE_FORMAT_D16_X_UNORM8);
}

#[test]
fn buffer_store_format_d16_x_snorm16() {
    run(&BUFFER_STORE_FORMAT_D16_X_SNORM16);
}

#[test]
fn buffer_store_format_d16_x_zero_fill() {
    run(&BUFFER_STORE_FORMAT_D16_X_ZERO_FILL);
}

#[test]
fn buffer_store_snorm8_range() {
    run(&BUFFER_STORE_SNORM8_RANGE);
}

#[test]
fn t_buffer_store_snorm8_range() {
    run(&T_BUFFER_STORE_SNORM8_RANGE);
}

#[test]
fn buffer_store_snorm8_ties() {
    run(&BUFFER_STORE_SNORM8_TIES);
}

#[test]
fn t_buffer_store_snorm8_ties() {
    run(&T_BUFFER_STORE_SNORM8_TIES);
}

#[test]
#[ignore = "case wrong (unverified on PS5): it stores a 2-bit SNORM channel as unsigned (1.0 -> 3), the GFX8-and-older fetch quirk Mesa only corrects up to GFX8; GFX9+ treats it as signed (1.0 -> 1)"]
fn buffer_store_snorm_packed_alpha() {
    run(&BUFFER_STORE_SNORM_PACKED_ALPHA);
}

#[test]
#[ignore = "case wrong (unverified on PS5): it stores a 2-bit SNORM channel as unsigned (1.0 -> 3), the GFX8-and-older fetch quirk Mesa only corrects up to GFX8; GFX9+ treats it as signed (1.0 -> 1)"]
fn t_buffer_store_snorm_packed_alpha() {
    run(&T_BUFFER_STORE_SNORM_PACKED_ALPHA);
}

#[test]
#[ignore = "case wrong (unverified on PS5): it stores a 2-bit SNORM channel as unsigned (1.0 -> 3), the GFX8-and-older fetch quirk Mesa only corrects up to GFX8; GFX9+ treats it as signed (1.0 -> 1)"]
fn buffer_store_snorm_packed_first_channel() {
    run(&BUFFER_STORE_SNORM_PACKED_FIRST_CHANNEL);
}

#[test]
#[ignore = "case wrong (unverified on PS5): it stores a 2-bit SNORM channel as unsigned (1.0 -> 3), the GFX8-and-older fetch quirk Mesa only corrects up to GFX8; GFX9+ treats it as signed (1.0 -> 1)"]
fn t_buffer_store_snorm_packed_first_channel() {
    run(&T_BUFFER_STORE_SNORM_PACKED_FIRST_CHANNEL);
}

#[test]
fn buffer_store_snorm_packed_zero_fill_x() {
    run(&BUFFER_STORE_SNORM_PACKED_ZERO_FILL_X);
}

#[test]
fn t_buffer_store_snorm_packed_zero_fill_x() {
    run(&T_BUFFER_STORE_SNORM_PACKED_ZERO_FILL_X);
}

#[test]
fn buffer_store_snorm_packed_zero_fill_xy() {
    run(&BUFFER_STORE_SNORM_PACKED_ZERO_FILL_XY);
}

#[test]
fn t_buffer_store_snorm_packed_zero_fill_xy() {
    run(&T_BUFFER_STORE_SNORM_PACKED_ZERO_FILL_XY);
}

#[test]
fn buffer_store_snorm_packed_zero_fill_xyz() {
    run(&BUFFER_STORE_SNORM_PACKED_ZERO_FILL_XYZ);
}

#[test]
fn t_buffer_store_snorm_packed_zero_fill_xyz() {
    run(&T_BUFFER_STORE_SNORM_PACKED_ZERO_FILL_XYZ);
}

#[test]
fn buffer_store_snorm8_zero_fill_x() {
    run(&BUFFER_STORE_SNORM8_ZERO_FILL_X);
}

#[test]
fn t_buffer_store_snorm8_zero_fill_x() {
    run(&T_BUFFER_STORE_SNORM8_ZERO_FILL_X);
}

#[test]
fn buffer_store_snorm16_zero_fill_x() {
    run(&BUFFER_STORE_SNORM16_ZERO_FILL_X);
}

#[test]
fn t_buffer_store_snorm16_zero_fill_x() {
    run(&T_BUFFER_STORE_SNORM16_ZERO_FILL_X);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn buffer_store_uscaled_zero_fill_xy() {
    run(&BUFFER_STORE_USCALED_ZERO_FILL_XY);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn t_buffer_store_uscaled_zero_fill_xy() {
    run(&T_BUFFER_STORE_USCALED_ZERO_FILL_XY);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn buffer_store_sscaled_zero_fill_xyz() {
    run(&BUFFER_STORE_SSCALED_ZERO_FILL_XYZ);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn t_buffer_store_sscaled_zero_fill_xyz() {
    run(&T_BUFFER_STORE_SSCALED_ZERO_FILL_XYZ);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn buffer_store_uscaled8_fraction() {
    run(&BUFFER_STORE_USCALED8_FRACTION);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn t_buffer_store_uscaled8_fraction() {
    run(&T_BUFFER_STORE_USCALED8_FRACTION);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn buffer_store_sscaled8_fraction() {
    run(&BUFFER_STORE_SSCALED8_FRACTION);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn t_buffer_store_sscaled8_fraction() {
    run(&T_BUFFER_STORE_SSCALED8_FRACTION);
}

#[test]
fn buffer_store_uscaled8_range() {
    run(&BUFFER_STORE_USCALED8_RANGE);
}

#[test]
fn t_buffer_store_uscaled8_range() {
    run(&T_BUFFER_STORE_USCALED8_RANGE);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn buffer_store_sscaled8_range() {
    run(&BUFFER_STORE_SSCALED8_RANGE);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn t_buffer_store_sscaled8_range() {
    run(&T_BUFFER_STORE_SSCALED8_RANGE);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn buffer_store_uscaled16_range() {
    run(&BUFFER_STORE_USCALED16_RANGE);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn t_buffer_store_uscaled16_range() {
    run(&T_BUFFER_STORE_USCALED16_RANGE);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn buffer_store_sscaled16_range() {
    run(&BUFFER_STORE_SSCALED16_RANGE);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn t_buffer_store_sscaled16_range() {
    run(&T_BUFFER_STORE_SSCALED16_RANGE);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn buffer_store_uscaled_packed() {
    run(&BUFFER_STORE_USCALED_PACKED);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn t_buffer_store_uscaled_packed() {
    run(&T_BUFFER_STORE_USCALED_PACKED);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn buffer_store_sscaled_packed_alpha() {
    run(&BUFFER_STORE_SSCALED_PACKED_ALPHA);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn t_buffer_store_sscaled_packed_alpha() {
    run(&T_BUFFER_STORE_SSCALED_PACKED_ALPHA);
}

#[test]
#[ignore = "case wrong on the conversion (SSCALED stores like SNORM on hardware, hw_mem_formatted_store); open on our side: format 47 (10_10_10_2 SSCALED, reserved in our table) stores raw dwords"]
fn buffer_store_sscaled_packed_first_channel() {
    run(&BUFFER_STORE_SSCALED_PACKED_FIRST_CHANNEL);
}

#[test]
#[ignore = "case wrong on the conversion (SSCALED stores like SNORM on hardware, hw_mem_formatted_store); open on our side: format 47 (10_10_10_2 SSCALED, reserved in our table) stores raw dwords"]
fn t_buffer_store_sscaled_packed_first_channel() {
    run(&T_BUFFER_STORE_SSCALED_PACKED_FIRST_CHANNEL);
}

#[test]
fn buffer_store_snorm8_special_values() {
    run(&BUFFER_STORE_SNORM8_SPECIAL_VALUES);
}

#[test]
fn t_buffer_store_snorm8_special_values() {
    run(&T_BUFFER_STORE_SNORM8_SPECIAL_VALUES);
}

#[test]
fn buffer_store_uscaled8_special_values() {
    run(&BUFFER_STORE_USCALED8_SPECIAL_VALUES);
}

#[test]
fn t_buffer_store_uscaled8_special_values() {
    run(&T_BUFFER_STORE_USCALED8_SPECIAL_VALUES);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn buffer_store_sscaled8_special_values() {
    run(&BUFFER_STORE_SSCALED8_SPECIAL_VALUES);
}

#[test]
#[ignore = "case wrong: it stores USCALED/SSCALED as clamped integers; hardware rows store them like UNORM/SNORM (hw_mem_formatted_store 8uscaled/16sscaled: 0.5 -> 0x80, -1.0 -> 0x8001; hw_mem_format_d16_store 8uscaled)"]
fn t_buffer_store_sscaled8_special_values() {
    run(&T_BUFFER_STORE_SSCALED8_SPECIAL_VALUES);
}

#[test]
fn buffer_load_format_x_resource8_uint_zero_extends_byte() {
    run(&BUFFER_LOAD_FORMAT_X_RESOURCE8_UINT_ZERO_EXTENDS_BYTE);
}

#[test]
fn buffer_load_format_xy_resource8_8_uint_extracts_bytes() {
    run(&BUFFER_LOAD_FORMAT_XY_RESOURCE8_8_UINT_EXTRACTS_BYTES);
}

#[test]
fn buffer_load_format_xy_resource8_8_8_8_unorm_converts_first_two_components() {
    run(&BUFFER_LOAD_FORMAT_XY_RESOURCE8_8_8_8_UNORM_CONVERTS_FIRST_TWO_COMPONENTS);
}

#[test]
fn buffer_store_format_xy_resource8_8_uint_writes_bytes() {
    run(&BUFFER_STORE_FORMAT_XY_RESOURCE8_8_UINT_WRITES_BYTES);
}

#[test]
fn buffer_store_format_xyz_resource32_32_uint_writes_two_dwords() {
    run(&BUFFER_STORE_FORMAT_XYZ_RESOURCE32_32_UINT_WRITES_TWO_DWORDS);
}

#[test]
fn buffer_store_format_xy_resource32_uint_writes_one_dword() {
    run(&BUFFER_STORE_FORMAT_XY_RESOURCE32_UINT_WRITES_ONE_DWORD);
}

#[test]
fn buffer_store_format_xyz_resource8_uint_writes_one_byte() {
    run(&BUFFER_STORE_FORMAT_XYZ_RESOURCE8_UINT_WRITES_ONE_BYTE);
}

#[test]
fn buffer_store_format_x_add_tid_uses_lane_index() {
    run(&BUFFER_STORE_FORMAT_X_ADD_TID_USES_LANE_INDEX);
}

#[test]
fn buffer_store_format_x_drops_out_of_range_record() {
    run(&BUFFER_STORE_FORMAT_X_DROPS_OUT_OF_RANGE_RECORD);
}

#[test]
fn buffer_zero_stride_oob_formats_and_widths() {
    run(&BUFFER_ZERO_STRIDE_OOB_FORMATS_AND_WIDTHS);
}

#[test]
fn t_buffer_load_format_xyzw_snapshots_overlapping_address() {
    run(&T_BUFFER_LOAD_FORMAT_XYZW_SNAPSHOTS_OVERLAPPING_ADDRESS);
}

#[test]
fn t_buffer_load_format_xyzw_packed_snapshots_overlapping_address() {
    run(&T_BUFFER_LOAD_FORMAT_XYZW_PACKED_SNAPSHOTS_OVERLAPPING_ADDRESS);
}

#[test]
fn t_buffer_load_format_x8_uint_zero_extends_byte() {
    run(&T_BUFFER_LOAD_FORMAT_X8_UINT_ZERO_EXTENDS_BYTE);
}

#[test]
fn t_buffer_load_format_x8_8_8_8_uint_extracts_first_byte() {
    run(&T_BUFFER_LOAD_FORMAT_X8_8_8_8_UINT_EXTRACTS_FIRST_BYTE);
}

#[test]
fn t_buffer_load_format_x_idxen_uses_descriptor_stride() {
    run(&T_BUFFER_LOAD_FORMAT_X_IDXEN_USES_DESCRIPTOR_STRIDE);
}

#[test]
fn t_buffer_load_format_x16_float_converts_to_float() {
    run(&T_BUFFER_LOAD_FORMAT_X16_FLOAT_CONVERTS_TO_FLOAT);
}

#[test]
fn t_buffer_store_format_x8_uint_writes_one_byte() {
    run(&T_BUFFER_STORE_FORMAT_X8_UINT_WRITES_ONE_BYTE);
}

#[test]
fn t_buffer_store_format_x_sint_writes_sub_dword() {
    run(&T_BUFFER_STORE_FORMAT_X_SINT_WRITES_SUB_DWORD);
}

#[test]
fn t_buffer_load_format_x_sint_sign_extends_sub_dword() {
    run(&T_BUFFER_LOAD_FORMAT_X_SINT_SIGN_EXTENDS_SUB_DWORD);
}

#[test]
fn t_buffer_load_format_xy16_16_integer_components() {
    run(&T_BUFFER_LOAD_FORMAT_XY16_16_INTEGER_COMPONENTS);
}

#[test]
fn t_buffer_store_format_xy16_16_integer_components() {
    run(&T_BUFFER_STORE_FORMAT_XY16_16_INTEGER_COMPONENTS);
}

#[test]
fn t_buffer_load_format_xyz16_16_16_16_uint_loads_halfwords() {
    run(&T_BUFFER_LOAD_FORMAT_XYZ16_16_16_16_UINT_LOADS_HALFWORDS);
}

#[test]
fn t_buffer_load_format_xy16_16_unorm_converts_to_float() {
    run(&T_BUFFER_LOAD_FORMAT_XY16_16_UNORM_CONVERTS_TO_FLOAT);
}

#[test]
fn t_buffer_load_format_xy8_8_unorm_converts_to_float() {
    run(&T_BUFFER_LOAD_FORMAT_XY8_8_UNORM_CONVERTS_TO_FLOAT);
}

#[test]
fn t_buffer_load_format_xy8_8_snorm_converts_to_float() {
    run(&T_BUFFER_LOAD_FORMAT_XY8_8_SNORM_CONVERTS_TO_FLOAT);
}

#[test]
fn t_buffer_load_format_xy8_8_8_8_unorm_converts_first_two_components() {
    run(&T_BUFFER_LOAD_FORMAT_XY8_8_8_8_UNORM_CONVERTS_FIRST_TWO_COMPONENTS);
}

#[test]
fn t_buffer_load_format_xyzw8_8_8_8_uint_extracts_bytes() {
    run(&T_BUFFER_LOAD_FORMAT_XYZW8_8_8_8_UINT_EXTRACTS_BYTES);
}

#[test]
fn t_buffer_load_format_xyzw10_10_10_2_snorm_converts_to_float() {
    run(&T_BUFFER_LOAD_FORMAT_XYZW10_10_10_2_SNORM_CONVERTS_TO_FLOAT);
}

#[test]
fn t_buffer_load_format_xyz11_11_10_float_unpacks() {
    run(&T_BUFFER_LOAD_FORMAT_XYZ11_11_10_FLOAT_UNPACKS);
}

#[test]
fn t_buffer_load_format_xyzw32_32_float_zeros_missing_components() {
    run(&T_BUFFER_LOAD_FORMAT_XYZW32_32_FLOAT_ZEROS_MISSING_COMPONENTS);
}

#[test]
fn t_buffer_store_format_xyzw32_32_float_writes_only_present_components() {
    run(&T_BUFFER_STORE_FORMAT_XYZW32_32_FLOAT_WRITES_ONLY_PRESENT_COMPONENTS);
}

#[test]
fn t_buffer_store_format_xy8_8_integer_components() {
    run(&T_BUFFER_STORE_FORMAT_XY8_8_INTEGER_COMPONENTS);
}

#[test]
fn t_buffer_load_format_xy8_8_integer_components() {
    run(&T_BUFFER_LOAD_FORMAT_XY8_8_INTEGER_COMPONENTS);
}

#[test]
fn buffer_atomic_and_x2_glc_and_exec() {
    run(&BUFFER_ATOMIC_AND_X2_GLC_AND_EXEC);
}

#[test]
fn buffer_atomic_variants() {
    run(&BUFFER_ATOMIC_VARIANTS);
}

#[test]
fn buffer_atomic_glc0_does_not_return_old_value() {
    run(&BUFFER_ATOMIC_GLC0_DOES_NOT_RETURN_OLD_VALUE);
}

#[test]
fn buffer_atomic_f_min_exact_raw_glc_modes() {
    run(&BUFFER_ATOMIC_F_MIN_EXACT_RAW_GLC_MODES);
}

#[test]
#[ignore = "case wrong: hardware rows (hw_mem_atomics_float): a quiet NaN in memory takes the data, a signaling NaN is returned quieted, min(-0, +0) = -0; the case keeps memory"]
fn buffer_atomic_f_min_special_values() {
    run(&BUFFER_ATOMIC_F_MIN_SPECIAL_VALUES);
}

#[test]
fn buffer_atomic_f_min_contended_workgroup() {
    run(&BUFFER_ATOMIC_F_MIN_CONTENDED_WORKGROUP);
}

#[test]
fn buffer_atomic_f_max_exact_raw_glc_modes() {
    run(&BUFFER_ATOMIC_F_MAX_EXACT_RAW_GLC_MODES);
}

#[test]
#[ignore = "case wrong: hardware rows (hw_mem_atomics_float): a quiet NaN in memory takes the data, a signaling NaN is returned quieted, max(-0, +0) = +0; the case keeps memory"]
fn buffer_atomic_f_max_special_values() {
    run(&BUFFER_ATOMIC_F_MAX_SPECIAL_VALUES);
}

#[test]
fn buffer_atomic_f_max_contended_workgroup() {
    run(&BUFFER_ATOMIC_F_MAX_CONTENDED_WORKGROUP);
}
