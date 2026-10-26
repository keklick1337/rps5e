// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Reference compute cases that sample, load, store or atomically update
//! images (and the ray-intersection cases, which read BVH nodes through an
//! image instruction).
//!
//! The interpreter delegates image operations to the embedder, so these
//! cases check what the program side decides: each case decodes, plans and
//! materializes its resources from the reference user data (with the
//! result buffer as the memory descriptor tables are read from) and
//! compiles unoptimized and optimized; the image contents and expected
//! texels are kept with the case for a device run.

#[path = "vectors/ref_harness.rs"]
mod harness;

use harness::*;
use ps5shader::program::decode_blob;
use ps5shader::resources::{materialize, ImageMipMode, ResourceRuntimeInputs, ShaderResourcePlan, ShaderStage};

/// A reference case with its images.
#[allow(dead_code)]
pub struct RefImage {
    pub case: RefCase,
    /// The 64 user-data words the case is compiled with.
    pub user_data: &'static [u32],
    pub width: u32,
    pub height: u32,
    /// Sampled image texels (mip 0) or, with `sampled_mips`, one slice per
    /// mip (RGBA32F).
    pub sampled: &'static [u32],
    pub sampled_mips: &'static [&'static [u32]],
    /// Host format (VkFormat) and layout of the sampled image.
    pub sampled_format: u32,
    pub sampled_dwords_per_pixel: u32,
    pub sampled_image_type: u32,
    pub sampled_view_type: u32,
    pub sampled_layers: u32,
    pub sampled_view_base_mip: u32,
    pub sampled_view_base_layer: u32,
    pub sampled_view_layers: u32,
    pub sampler_linear: bool,
    /// Storage image texels before and after the dispatch.
    pub storage: &'static [u32],
    pub expected_storage: &'static [u32],
    pub storage_format: u32,
    pub storage_dwords_per_pixel: u32,
    /// The R32 (or R64) unsigned storage image, before and after.
    pub storage_r32ui: &'static [u32],
    pub expected_storage_r32ui: &'static [u32],
    /// T# destination select the case's descriptor carries.
    pub descriptor_swizzle: u32,
    /// The sampler must be specialized to point filtering.
    pub force_point_sampler: bool,
    /// Descriptors a dynamic-mip image must receive (0: not checked).
    pub expected_mip_descriptors: u32,
}

include!("vectors/ref_image.rs");

fn check_image(m: &RefImage) {
    let c = &m.case;
    let bytes: Vec<u8> = c.code.iter().flat_map(|w| w.to_le_bytes()).collect();
    let program = decode_blob(&bytes, CODE_ADDRESS).unwrap_or_else(|e| panic!("{}: decode: {e}", c.name));
    let plan = ShaderResourcePlan::extract(program, ShaderStage::Compute, 1, 0, 64, None, c.compute.wave_size)
        .unwrap_or_else(|e| panic!("{}: resource plan: {e:?}", c.name));
    let initial = c.initial;
    let read = move |a: u64| -> Option<u32> {
        if a % 4 != 0 {
            return None;
        }
        initial.get((a / 4) as usize).copied()
    };
    let mut inputs = ResourceRuntimeInputs::new(m.user_data, CODE_ADDRESS);
    inputs.read_memory = Some(&read);
    let (_, spec) = materialize(&plan, &inputs).unwrap_or_else(|e| panic!("{}: resource materialization: {e:?}", c.name));
    // An image read per mip (an explicit-LOD gather, a storage access with
    // a mip) gets one descriptor per level of its view.
    if m.expected_mip_descriptors != 0 {
        let dynamic: Vec<u32> = plan
            .info
            .images
            .iter()
            .zip(&spec.images)
            .filter(|(i, _)| i.mip_mode == ImageMipMode::Dynamic)
            .map(|(_, s)| s.mip_count)
            .collect();
        assert_eq!(dynamic, vec![m.expected_mip_descriptors], "{}: mip descriptors", c.name);
    }
    for optimize in [false, true] {
        if let Err(e) = c.compile(optimize) {
            panic!("{} [optimize={optimize}]: {e}", c.name);
        }
    }
}

#[test]
fn image_atomic_s_min_glc_and_exec() {
    check_image(&IMAGE_ATOMIC_S_MIN_GLC_AND_EXEC);
}

#[test]
fn image_atomic_s_max_glc_and_exec() {
    check_image(&IMAGE_ATOMIC_S_MAX_GLC_AND_EXEC);
}

#[test]
fn image_load_variants() {
    check_image(&IMAGE_LOAD_VARIANTS);
}

#[test]
fn image_load_r32_uint_uses_integer_sampled_image() {
    check_image(&IMAGE_LOAD_R32_UINT_USES_INTEGER_SAMPLED_IMAGE);
}

#[test]
fn image_load_r32_sint_uses_signed_sampled_image() {
    check_image(&IMAGE_LOAD_R32_SINT_USES_SIGNED_SAMPLED_IMAGE);
}

#[test]
fn image_load_packed_uint_unpacks_and_swizzles() {
    check_image(&IMAGE_LOAD_PACKED_UINT_UNPACKS_AND_SWIZZLES);
}

#[test]
fn image_sample_packed_uint_converts_sample_and_gather() {
    check_image(&IMAGE_SAMPLE_PACKED_UINT_CONVERTS_SAMPLE_AND_GATHER);
}

#[test]
fn image_sample_r8_u_scaled() {
    check_image(&IMAGE_SAMPLE_R8_U_SCALED);
}

#[test]
fn image_sample_rg8_u_scaled() {
    check_image(&IMAGE_SAMPLE_RG8_U_SCALED);
}

#[test]
fn image_load_r128_ignores_adjacent_mask_sgprs() {
    check_image(&IMAGE_LOAD_R128_IGNORES_ADJACENT_MASK_SGPRS);
}

#[test]
fn image_load1_d_uses_scalar_coordinate() {
    check_image(&IMAGE_LOAD1_D_USES_SCALAR_COORDINATE);
}

#[test]
fn image_gather2_d_instruction_with1_d_descriptor() {
    check_image(&IMAGE_GATHER2_D_INSTRUCTION_WITH1_D_DESCRIPTOR);
}

#[test]
fn image_load1_d_array_uses_layer_coordinate() {
    check_image(&IMAGE_LOAD1_D_ARRAY_USES_LAYER_COORDINATE);
}

#[test]
fn image_load1_d_array_descriptor_uses_selected_layer() {
    check_image(&IMAGE_LOAD1_D_ARRAY_DESCRIPTOR_USES_SELECTED_LAYER);
}

#[test]
fn image_load_mip_uses_vaddr2_lod2_d() {
    check_image(&IMAGE_LOAD_MIP_USES_VADDR2_LOD2_D);
}

#[test]
fn image_load_mip_nsa_uses_selected_address_vgprs() {
    check_image(&IMAGE_LOAD_MIP_NSA_USES_SELECTED_ADDRESS_VGPRS);
}

#[test]
fn image_load_a16_uint_coords_on_gpu() {
    check_image(&IMAGE_LOAD_A16_UINT_COORDS_ON_GPU);
}

#[test]
fn image_get_resinfo_dmask_width_height() {
    check_image(&IMAGE_GET_RESINFO_DMASK_WIDTH_HEIGHT);
}

#[test]
fn image_get_resinfo_dmask_mip_levels() {
    check_image(&IMAGE_GET_RESINFO_DMASK_MIP_LEVELS);
}

#[test]
fn image_sample_and_gather() {
    check_image(&IMAGE_SAMPLE_AND_GATHER);
}

#[test]
fn image_cube_gradients_preserve_derivatives() {
    check_image(&IMAGE_CUBE_GRADIENTS_PRESERVE_DERIVATIVES);
}

#[test]
fn image_gather_explicit_lod() {
    check_image(&IMAGE_GATHER_EXPLICIT_LOD);
}

#[test]
fn image_gather_lod_per_lane() {
    check_image(&IMAGE_GATHER_LOD_PER_LANE);
}

#[test]
fn image_gather_lod_base_mip() {
    check_image(&IMAGE_GATHER_LOD_BASE_MIP);
}

#[test]
fn image_gather_lod_sampler_bounds() {
    check_image(&IMAGE_GATHER_LOD_SAMPLER_BOUNDS);
}

#[test]
fn image_gather_lod_texture_minimum() {
    check_image(&IMAGE_GATHER_LOD_TEXTURE_MINIMUM);
}

#[test]
fn image_gather_lod_mip_none() {
    check_image(&IMAGE_GATHER_LOD_MIP_NONE);
}

#[test]
fn image_gather_lod_point_postclamp_maximum() {
    check_image(&IMAGE_GATHER_LOD_POINT_POSTCLAMP_MAXIMUM);
}

#[test]
fn image_gather_lod_point_preclamp_maximum() {
    check_image(&IMAGE_GATHER_LOD_POINT_PRECLAMP_MAXIMUM);
}

#[test]
fn image_gather_lod_point_postclamp_minimum() {
    check_image(&IMAGE_GATHER_LOD_POINT_POSTCLAMP_MINIMUM);
}

#[test]
fn image_gather_lod_point_preclamp_minimum() {
    check_image(&IMAGE_GATHER_LOD_POINT_PRECLAMP_MINIMUM);
}

/// The point-preclamp case with a linear mip filter (S# word 2
/// 0x0a500000, MIP_POINT_PRECLAMP set or not) and the full LOD range: a
/// linear explicit-LOD gather reads the lower (primary) mip, so the LODs
/// -1, 0, 0.25, 0.75, 1.25, 1.75, 2.75, 9 select mips 0, 0, 0, 0, 1, 1, 2,
/// 4 (texel value = mip + 1), whatever the preclamp bit says.
fn gather_lod_linear(name: &'static str, control: u32) -> &'static RefImage {
    let base = &IMAGE_GATHER_LOD_POINT_PRECLAMP_MINIMUM;
    let with_sampler = |words: &[u32]| -> &'static [u32] {
        let mut w = words.to_vec();
        w[9] = 1024 << 12;
        w[10] = control;
        Box::leak(w.into_boxed_slice())
    };
    let mips = [0u32, 0, 0, 0, 1, 1, 2, 4];
    let mut expected = base.case.initial.to_vec();
    for _ in 0..4 {
        expected.extend(mips.iter().map(|&m| (m as f32 + 1.0).to_bits()));
    }
    Box::leak(Box::new(RefImage {
        case: RefCase { name, expected: Box::leak(expected.into_boxed_slice()), user_data: Some(with_sampler(base.case.user_data.unwrap())), ..base.case },
        user_data: with_sampler(base.user_data),
        ..*base
    }))
}

#[test]
fn image_gather_lod_linear_primary() {
    check_image(gather_lod_linear("ImageGatherLodLinearPrimary", 0x0a50_0000));
}

#[test]
fn image_gather_lod_linear_ignores_preclamp() {
    check_image(gather_lod_linear("ImageGatherLodLinearIgnoresPreclamp", 0x1a50_0000));
}

#[test]
fn image_d16_gather_packs_half_pairs() {
    check_image(&IMAGE_D16_GATHER_PACKS_HALF_PAIRS);
}

#[test]
fn image_sample_a16_sampler_coords_on_gpu() {
    check_image(&IMAGE_SAMPLE_A16_SAMPLER_COORDS_ON_GPU);
}

#[test]
fn image_sample_a16_offset_keeps_texel_offset32_bit_on_gpu() {
    check_image(&IMAGE_SAMPLE_A16_OFFSET_KEEPS_TEXEL_OFFSET32_BIT_ON_GPU);
}

#[test]
fn image_store_variants() {
    check_image(&IMAGE_STORE_VARIANTS);
}

#[test]
fn image_d16_store_unpacks_half_pairs() {
    check_image(&IMAGE_D16_STORE_UNPACKS_HALF_PAIRS);
}

#[test]
fn image_store_single_mip_valid() {
    check_image(&IMAGE_STORE_SINGLE_MIP_VALID);
}

#[test]
fn image_store_single_mip_beyond_end() {
    check_image(&IMAGE_STORE_SINGLE_MIP_BEYOND_END);
}

#[test]
fn image_store_single_mip_negative() {
    check_image(&IMAGE_STORE_SINGLE_MIP_NEGATIVE);
}

#[test]
fn image_store_rgb_one_uses_inverse_swizzle() {
    check_image(&IMAGE_STORE_RGB_ONE_USES_INVERSE_SWIZZLE);
}

#[test]
fn image_store_duplicate_selector_uses_inverse_swizzle() {
    check_image(&IMAGE_STORE_DUPLICATE_SELECTOR_USES_INVERSE_SWIZZLE);
}

#[test]
fn image_store_bgra_uses_inverse_swizzle() {
    check_image(&IMAGE_STORE_BGRA_USES_INVERSE_SWIZZLE);
}

#[test]
fn image_store_yzwx_uses_inverse_swizzle() {
    check_image(&IMAGE_STORE_YZWX_USES_INVERSE_SWIZZLE);
}

#[test]
fn image_store_r32_float_uses_formatless_storage_image() {
    check_image(&IMAGE_STORE_R32_FLOAT_USES_FORMATLESS_STORAGE_IMAGE);
}

#[test]
fn image_store_r32_sint_uses_raw_uint_view() {
    check_image(&IMAGE_STORE_R32_SINT_USES_RAW_UINT_VIEW);
}

#[test]
fn image_store_r32_uint_uses_formatless_storage_image() {
    check_image(&IMAGE_STORE_R32_UINT_USES_FORMATLESS_STORAGE_IMAGE);
}

#[test]
fn image_store_packed_uint_saturates_channels() {
    check_image(&IMAGE_STORE_PACKED_UINT_SATURATES_CHANNELS);
}

#[test]
fn image_store_packed_uint_honors_sparse_dmask() {
    check_image(&IMAGE_STORE_PACKED_UINT_HONORS_SPARSE_DMASK);
}

#[test]
fn image_store_and_atomic_share_typed_binding() {
    check_image(&IMAGE_STORE_AND_ATOMIC_SHARE_TYPED_BINDING);
}

#[test]
fn image_atomic_compare_swap_glc_and_exec() {
    check_image(&IMAGE_ATOMIC_COMPARE_SWAP_GLC_AND_EXEC);
}

#[test]
fn image_atomic_swap_returns_previous_texel() {
    check_image(&IMAGE_ATOMIC_SWAP_RETURNS_PREVIOUS_TEXEL);
}

#[test]
fn image_store_and_atomic_use_separate_bindings() {
    check_image(&IMAGE_STORE_AND_ATOMIC_USE_SEPARATE_BINDINGS);
}

#[test]
fn image_atomic_variants() {
    check_image(&IMAGE_ATOMIC_VARIANTS);
}

#[test]
fn image_atomic_swap32_glc_and_exec() {
    check_image(&IMAGE_ATOMIC_SWAP32_GLC_AND_EXEC);
}

#[test]
fn image_atomic_swap64_glc_and_exec() {
    check_image(&IMAGE_ATOMIC_SWAP64_GLC_AND_EXEC);
}

#[test]
fn image_atomic_add32_glc_and_exec() {
    check_image(&IMAGE_ATOMIC_ADD32_GLC_AND_EXEC);
}

#[test]
fn image_atomic_add64_glc_and_exec() {
    check_image(&IMAGE_ATOMIC_ADD64_GLC_AND_EXEC);
}

#[test]
fn image_atomic_u_min32_glc_and_exec() {
    check_image(&IMAGE_ATOMIC_U_MIN32_GLC_AND_EXEC);
}

#[test]
fn image_atomic_u_min64_glc_and_exec() {
    check_image(&IMAGE_ATOMIC_U_MIN64_GLC_AND_EXEC);
}

#[test]
fn image_atomic_u_max32_glc_and_exec() {
    check_image(&IMAGE_ATOMIC_U_MAX32_GLC_AND_EXEC);
}

#[test]
fn image_atomic_u_max64_glc_and_exec() {
    check_image(&IMAGE_ATOMIC_U_MAX64_GLC_AND_EXEC);
}

#[test]
fn image_atomic_and32_glc_and_exec() {
    check_image(&IMAGE_ATOMIC_AND32_GLC_AND_EXEC);
}

#[test]
fn image_atomic_and64_glc_and_exec() {
    check_image(&IMAGE_ATOMIC_AND64_GLC_AND_EXEC);
}

#[test]
fn image_atomic_or32_glc_and_exec() {
    check_image(&IMAGE_ATOMIC_OR32_GLC_AND_EXEC);
}

#[test]
fn image_atomic_or64_glc_and_exec() {
    check_image(&IMAGE_ATOMIC_OR64_GLC_AND_EXEC);
}

#[test]
fn image_atomic_xor32_glc_and_exec() {
    check_image(&IMAGE_ATOMIC_XOR32_GLC_AND_EXEC);
}

#[test]
fn image_atomic_xor64_glc_and_exec() {
    check_image(&IMAGE_ATOMIC_XOR64_GLC_AND_EXEC);
}

#[test]
fn image_atomic_f_max_captured_glc_and_exec() {
    check_image(&IMAGE_ATOMIC_F_MAX_CAPTURED_GLC_AND_EXEC);
}

#[test]
fn image_atomic_f_min_glc_and_exec() {
    check_image(&IMAGE_ATOMIC_F_MIN_GLC_AND_EXEC);
}

#[test]
fn image_atomic_f_max_special_values() {
    check_image(&IMAGE_ATOMIC_F_MAX_SPECIAL_VALUES);
}

#[test]
fn image_atomic_f_min_special_values() {
    check_image(&IMAGE_ATOMIC_F_MIN_SPECIAL_VALUES);
}

#[test]
fn image_atomic_f_max_contended_workgroup() {
    check_image(&IMAGE_ATOMIC_F_MAX_CONTENDED_WORKGROUP);
}

#[test]
fn image_atomic_f_min_contended_workgroup() {
    check_image(&IMAGE_ATOMIC_F_MIN_CONTENDED_WORKGROUP);
}

#[test]
fn image_atomic_glc0_does_not_return_old_value() {
    check_image(&IMAGE_ATOMIC_GLC0_DOES_NOT_RETURN_OLD_VALUE);
}

