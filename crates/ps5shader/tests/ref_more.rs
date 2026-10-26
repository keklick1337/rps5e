// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Reference compute cases with GDS contents or several guest windows of
//! the result buffer (scalar, buffer, flat, BVH, LDS and GDS).
//!
//! Each test compiles the case, runs it in the interpreter (unoptimized
//! and optimized) and compares the result buffer and, when given, GDS; see
//! `vectors/ref_harness.rs` for the buffer and user-data mapping.
//!
//! `host_offsets[i]` is the byte offset at which buffer resource `i` (in
//! the order the program first uses its V#) starts within the result
//! buffer: its V# base address names that byte, so the buffer is mapped
//! at `base - offset`. Those guest windows are listed per case below.

#[path = "vectors/ref_harness.rs"]
mod harness;

use harness::*;

/// A reference case with GDS and host buffer offsets.
pub struct RefMore {
    pub case: RefCase,
    pub gds_initial: &'static [u32],
    pub expected_gds: &'static [u32],
    /// Bytes of the result buffer each resource may address past its
    /// offset (0: all of it).
    pub host_range: u32,
    pub host_offsets: &'static [u32],
    /// The case widens the shader-data block past the push constants.
    pub expand_shader_data: bool,
    /// Buffer resources that stay bound after dead-code removal.
    pub expected_buffer_resources: Option<&'static [u32]>,
}

include!("vectors/ref_more.rs");

/// Guest addresses of the result buffer: V# base minus host offset, per
/// buffer resource of the case.
fn windows(name: &str) -> Option<&'static [u64]> {
    Some(match name {
        // s[0:3] base 0 at host offset 2: the V# base names byte 2 of the
        // buffer, so it moves to guest address 2 over a buffer at 0 (see
        // `vsharp_base`).
        "BufferStoreDwordAppliesHostOffset" => &[0],
        // s[4i:4i+3] base 0x1000 + 0x4000 i at offsets 0, 12, 20, 28, 36.
        "BufferOffsetsPackedWordsPush" | "BufferOffsetsPackedWordsStorage" => &[0x1000, 0x5000 - 12, 0x9000 - 20, 0xD000 - 28, 0x11000 - 36],
        // s[0:3] base 0x1000 + offset, s[48:51] base 0x2000 at offset 16.
        "BufferUshortHostOffsetTail"
        | "BufferUshortHostOffsetCarry"
        | "BufferUshortHostOffsetFirst"
        | "BufferSshortHostOffsetTail"
        | "BufferUbyteHostOffsetTail"
        | "BufferSbyteHostOffsetTail"
        | "BufferSubwordHostOffsetOverflow" => &[0x1000, 0x2000 - 16],
        "BufferShortStoreHostOffsetTail" | "BufferByteStoreHostOffsetTail" => &[0x1000],
        // s[48:51] and s[0:3] at base 0, s[32:35] at its captured base,
        // s[4:7] base 0x2000 at offset 16.
        "TBufferCapturedZeroStrideOob" | "TBufferZeroStrideRawBoundsStillLoads" | "TBufferZeroStrideOobPreservesScalarRead" => {
            &[0, 0x4a39_8620, 0x2000 - 16]
        }
        // The loads overwrite s[48:51] (the result store V#) with buffer
        // words 27..30; the stores then address the buffer through that V#
        // (base 0x1c_1000_001b, dword-aligned by the stores).
        "ScalarMemoryLoadVariants" => &[0, 0x1c_1000_0018],
        _ => return None,
    })
}

/// Cases whose flat mapping takes guest address 0, or whose V# base is a
/// byte inside the buffer: the result buffer's V#s move to that address.
fn vsharp_base(name: &str) -> Option<u64> {
    match name {
        "FlatLoadVariants" => Some(0x30_0000_0000),
        "BufferStoreDwordAppliesHostOffset" => Some(2),
        _ => None,
    }
}

/// Words that depend on the harness rather than the program: the high
/// dword of S_GETPC_B64 is that of the address the code is mapped at, and
/// S_MEMREALTIME reads a running clock.
fn adjusted_expectation(c: &RefCase) -> (Vec<u32>, Vec<usize>) {
    let mut expected = c.expected.to_vec();
    let mut unchecked = Vec::new();
    match c.name {
        "ScalarGetpcWritesNextInstructionPc" => expected[1] = (CODE_ADDRESS >> 32) as u32,
        "ScalarMemRealtimeCapturedPlaceholder" => unchecked.extend(0..4),
        _ => {}
    }
    (expected, unchecked)
}

/// Cases whose image or ray-intersection operations the interpreter
/// delegates to the embedder: they must decode and compile (the results
/// are checked on a device).
fn compile_more(m: &RefMore) {
    for optimize in [false, true] {
        if let Err(e) = m.case.compile(optimize) {
            panic!("{} [optimize={optimize}]: {e}", m.case.name);
        }
    }
}

fn run_more(m: &RefMore) {
    let c = &m.case;
    let mut errors = Vec::new();
    for optimize in [false, true] {
        if c.compile_only {
            if let Err(e) = c.compile(optimize) {
                errors.push(format!("[optimize={optimize}] {e}"));
            }
            continue;
        }
        let gds_words = m.gds_initial.len().max(m.expected_gds.len());
        let mut gds: Vec<u8> = m.gds_initial.iter().flat_map(|w| w.to_le_bytes()).collect();
        gds.resize(gds_words * 4, 0);
        let mut env = Env { windows: windows(c.name), gds, vsharp_base: vsharp_base(c.name) };
        let (expected, unchecked) = adjusted_expectation(c);
        match c.execute_env(optimize, c.expected.len(), &mut env) {
            Err(e) => errors.push(format!("[optimize={optimize}] {e}")),
            Ok(actual) => {
                let mut diff: Vec<String> = actual
                    .iter()
                    .zip(&expected)
                    .enumerate()
                    .filter(|(i, (a, e))| !unchecked.contains(i) && a != e)
                    .map(|(i, (a, e))| format!("  word {i}: expected {e:#010x} actual {a:#010x}"))
                    .collect();
                let gds: Vec<u32> = env.gds.chunks(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect();
                diff.extend(
                    gds.iter()
                        .zip(m.expected_gds)
                        .enumerate()
                        .filter(|(_, (a, e))| a != e)
                        .map(|(i, (a, e))| format!("  GDS word {i}: expected {e:#010x} actual {a:#010x}")),
                );
                if !diff.is_empty() {
                    errors.push(format!("[optimize={optimize}] {} words differ:\n{}", diff.len(), diff.join("\n")));
                }
            }
        }
    }
    if !errors.is_empty() {
        panic!("{}: {}", c.name, errors.join("\n"));
    }
}

#[test]
fn scalar_orn2_saveexec_b32_partial_wave() {
    run_more(&SCALAR_ORN2_SAVEEXEC_B32_PARTIAL_WAVE);
}

#[test]
fn scalar_getpc_writes_next_instruction_pc() {
    run_more(&SCALAR_GETPC_WRITES_NEXT_INSTRUCTION_PC);
}

#[test]
fn scalar_mem_realtime_captured_placeholder() {
    run_more(&SCALAR_MEM_REALTIME_CAPTURED_PLACEHOLDER);
}

#[test]
#[ignore = "case wrong: it aligns the whole address of a dword buffer access down to a dword; the reference driver's execution test BufferUnalignedBase (hw_mem_bases unaligned_base) keeps a V# base off a dword boundary and drops only the low bits of the offset"]
fn scalar_memory_load_variants() {
    run_more(&SCALAR_MEMORY_LOAD_VARIANTS);
}

#[test]
#[ignore = "case wrong: it aligns the whole address of a dword buffer access down to a dword; the reference driver's execution test BufferUnalignedBase (hw_mem_bases unaligned_base) keeps a V# base off a dword boundary and drops only the low bits of the offset"]
fn buffer_store_dword_applies_host_offset() {
    run_more(&BUFFER_STORE_DWORD_APPLIES_HOST_OFFSET);
}

#[test]
fn buffer_offsets_packed_words_push() {
    run_more(&BUFFER_OFFSETS_PACKED_WORDS_PUSH);
}

#[test]
fn buffer_offsets_packed_words_storage() {
    run_more(&BUFFER_OFFSETS_PACKED_WORDS_STORAGE);
}

#[test]
fn buffer_load_short_d16_captured() {
    run_more(&BUFFER_LOAD_SHORT_D16_CAPTURED);
}

#[test]
fn buffer_load_short_d16_hi_captured() {
    run_more(&BUFFER_LOAD_SHORT_D16_HI_CAPTURED);
}

#[test]
fn flat_store_slc_captured() {
    run_more(&FLAT_STORE_SLC_CAPTURED);
}

#[test]
fn buffer_ushort_host_offset_tail() {
    run_more(&BUFFER_USHORT_HOST_OFFSET_TAIL);
}

#[test]
fn buffer_ushort_host_offset_carry() {
    run_more(&BUFFER_USHORT_HOST_OFFSET_CARRY);
}

#[test]
fn buffer_ushort_host_offset_first() {
    run_more(&BUFFER_USHORT_HOST_OFFSET_FIRST);
}

#[test]
fn buffer_sshort_host_offset_tail() {
    run_more(&BUFFER_SSHORT_HOST_OFFSET_TAIL);
}

#[test]
fn buffer_ubyte_host_offset_tail() {
    run_more(&BUFFER_UBYTE_HOST_OFFSET_TAIL);
}

#[test]
fn buffer_sbyte_host_offset_tail() {
    run_more(&BUFFER_SBYTE_HOST_OFFSET_TAIL);
}

#[test]
fn buffer_subword_host_offset_overflow() {
    run_more(&BUFFER_SUBWORD_HOST_OFFSET_OVERFLOW);
}

#[test]
fn buffer_short_store_host_offset_tail() {
    run_more(&BUFFER_SHORT_STORE_HOST_OFFSET_TAIL);
}

#[test]
fn buffer_byte_store_host_offset_tail() {
    run_more(&BUFFER_BYTE_STORE_HOST_OFFSET_TAIL);
}

#[test]
fn buffer_format_load_variants() {
    run_more(&BUFFER_FORMAT_LOAD_VARIANTS);
}

#[test]
fn buffer_format_store_variants() {
    run_more(&BUFFER_FORMAT_STORE_VARIANTS);
}

#[test]
#[ignore = "case wrong: format stores are range-checked all or nothing (RDNA ISA, buffer range checking); the 12-byte 32_32_32 element does not fit the 8 bytes of NUM_RECORDS, so nothing is written"]
fn buffer_store_format_xyzw_resource32_32_32_uint_writes_three_dwords() {
    run_more(&BUFFER_STORE_FORMAT_XYZW_RESOURCE32_32_32_UINT_WRITES_THREE_DWORDS);
}

#[test]
fn t_buffer_load_variants() {
    run_more(&T_BUFFER_LOAD_VARIANTS);
}

#[test]
fn t_buffer_captured_zero_stride_oob() {
    run_more(&T_BUFFER_CAPTURED_ZERO_STRIDE_OOB);
}

#[test]
fn t_buffer_zero_stride_raw_bounds_still_loads() {
    run_more(&T_BUFFER_ZERO_STRIDE_RAW_BOUNDS_STILL_LOADS);
}

#[test]
fn t_buffer_zero_stride_oob_preserves_scalar_read() {
    run_more(&T_BUFFER_ZERO_STRIDE_OOB_PRESERVES_SCALAR_READ);
}

#[test]
fn t_buffer_store_variants() {
    run_more(&T_BUFFER_STORE_VARIANTS);
}

#[test]
fn flat_load_dlc_captured_wave32() {
    run_more(&FLAT_LOAD_DLC_CAPTURED_WAVE32);
}

#[test]
fn flat_load_dlc_captured_wave64() {
    run_more(&FLAT_LOAD_DLC_CAPTURED_WAVE64);
}

#[test]
fn flat_load_variants() {
    run_more(&FLAT_LOAD_VARIANTS);
}

#[test]
fn flat_virtual_address_rebases_guest_allocation() {
    run_more(&FLAT_VIRTUAL_ADDRESS_REBASES_GUEST_ALLOCATION);
}

#[test]
#[ignore = "unverified: FLAT is now routed by address (4 GiB apertures 0x70000000 scratch, 0x80000000 LDS), but the case's last row expects 0 from an unaligned dword stored and loaded at LDS byte 1, while the FlatApertures table of the other reference (tests/flat_apertures.rs) reads unaligned aperture dwords back byte-exact; no hardware row settles it"]
fn flat_stack_apertures32() {
    run_more(&FLAT_STACK_APERTURES32);
}

#[test]
#[ignore = "unverified: FLAT is now routed by address (4 GiB apertures 0x70000000 scratch, 0x80000000 LDS), but the case's last row expects 0 from an unaligned dword stored and loaded at LDS byte 1, while the FlatApertures table of the other reference (tests/flat_apertures.rs) reads unaligned aperture dwords back byte-exact; no hardware row settles it"]
fn flat_stack_apertures64() {
    run_more(&FLAT_STACK_APERTURES64);
}

#[test]
fn bvh_box_exit_growth() {
    compile_more(&BVH_BOX_EXIT_GROWTH);
}

#[test]
fn bvh64_high_node_pointers() {
    compile_more(&BVH64_HIGH_NODE_POINTERS);
}

#[test]
fn bvh_triangle_id_unsorted() {
    compile_more(&BVH_TRIANGLE_ID_UNSORTED);
}

#[test]
fn bvh_triangle_id_sorted() {
    compile_more(&BVH_TRIANGLE_ID_SORTED);
}

#[test]
fn bvh_barycentric_unsorted() {
    compile_more(&BVH_BARYCENTRIC_UNSORTED);
}

#[test]
fn bvh_barycentric_sorted() {
    compile_more(&BVH_BARYCENTRIC_SORTED);
}

#[test]
fn ds_append_allocates_across_waves32() {
    run_more(&DS_APPEND_ALLOCATES_ACROSS_WAVES32);
}

#[test]
fn ds_append_allocates_across_waves64() {
    run_more(&DS_APPEND_ALLOCATES_ACROSS_WAVES64);
}

#[test]
fn ds_append_gds_selector() {
    run_more(&DS_APPEND_GDS_SELECTOR);
}

#[test]
fn ds_ordered_count_address_and_exec_wave32() {
    run_more(&DS_ORDERED_COUNT_ADDRESS_AND_EXEC_WAVE32);
}

#[test]
fn ds_ordered_count_pixel_address_wave32() {
    run_more(&DS_ORDERED_COUNT_PIXEL_ADDRESS_WAVE32);
}

#[test]
fn ds_ordered_count_address_and_exec_wave64() {
    run_more(&DS_ORDERED_COUNT_ADDRESS_AND_EXEC_WAVE64);
}

#[test]
fn ds_ordered_count_pixel_address_wave64() {
    run_more(&DS_ORDERED_COUNT_PIXEL_ADDRESS_WAVE64);
}

#[test]
fn ds_append_consume_gds_region_bounds() {
    run_more(&DS_APPEND_CONSUME_GDS_REGION_BOUNDS);
}

#[test]
fn ds_gds_subdword_and_atomics() {
    run_more(&DS_GDS_SUBDWORD_AND_ATOMICS);
}

#[test]
fn ds_wide_gds_partial_bounds() {
    run_more(&DS_WIDE_GDS_PARTIAL_BOUNDS);
}

#[test]
fn ds_masked_or_gds_bounds() {
    run_more(&DS_MASKED_OR_GDS_BOUNDS);
}

#[test]
fn ds_add_u64_contention_wave32() {
    run_more(&DS_ADD_U64_CONTENTION_WAVE32);
}

#[test]
fn ds_add_u64_contention_wave64() {
    run_more(&DS_ADD_U64_CONTENTION_WAVE64);
}

#[test]
fn ds_inc_return_gds_boundaries() {
    run_more(&DS_INC_RETURN_GDS_BOUNDARIES);
}

#[test]
fn ds_inc_return_gds_contention_wave32() {
    run_more(&DS_INC_RETURN_GDS_CONTENTION_WAVE32);
}

#[test]
fn ds_inc_return_gds_contention_wave64() {
    run_more(&DS_INC_RETURN_GDS_CONTENTION_WAVE64);
}

#[test]
fn ds_dec_return_gds_boundaries() {
    run_more(&DS_DEC_RETURN_GDS_BOUNDARIES);
}

#[test]
fn ds_dec_return_gds_contention_wave32() {
    run_more(&DS_DEC_RETURN_GDS_CONTENTION_WAVE32);
}

#[test]
fn ds_dec_return_gds_contention_wave64() {
    run_more(&DS_DEC_RETURN_GDS_CONTENTION_WAVE64);
}

#[test]
fn ds_float_min_max_gds_classes() {
    run_more(&DS_FLOAT_MIN_MAX_GDS_CLASSES);
}

#[test]
fn wave64_cross_half_lane_and_lds() {
    run_more(&WAVE64_CROSS_HALF_LANE_AND_LDS);
}

#[test]
fn wave64_append_consume_high_half() {
    run_more(&WAVE64_APPEND_CONSUME_HIGH_HALF);
}

#[test]
#[ignore = "case wrong: its helper stores go through a stride-4 OOB_SELECT 0 V# at offsets 8..20; OOB_SELECT 0 (STRUCTURED_WITH_OFFSET) requires offset < stride, so they are dropped"]
fn buffer_atomic_cmp_swap_exact_raw() {
    run_more(&BUFFER_ATOMIC_CMP_SWAP_EXACT_RAW);
}

#[test]
fn image_load_fmask_uses_native_sample_mapping() {
    compile_more(&IMAGE_LOAD_FMASK_USES_NATIVE_SAMPLE_MAPPING);
}

#[test]
fn image_sample_opcode_alias_uses_normal_coords() {
    run_more(&IMAGE_SAMPLE_OPCODE_ALIAS_USES_NORMAL_COORDS);
}

#[test]
fn image_sample_a16_compare_bias_rdna2_address_order() {
    run_more(&IMAGE_SAMPLE_A16_COMPARE_BIAS_RDNA2_ADDRESS_ORDER);
}

#[test]
fn image_gather_compare_opcodes() {
    run_more(&IMAGE_GATHER_COMPARE_OPCODES);
}

#[test]
fn image_store_mip_selects_ppsa01340_descriptor() {
    run_more(&IMAGE_STORE_MIP_SELECTS_PPSA01340_DESCRIPTOR);
}

