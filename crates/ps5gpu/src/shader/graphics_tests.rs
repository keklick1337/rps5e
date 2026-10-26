// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The reference graphics (rasterisation) cases compiled to SPIR-V: each
//! fragment program is decoded, planned, materialised from its user data,
//! emitted with the case's pixel inputs (SPI_PS_INPUT_ENA/ADDR/CNTL) and
//! checked with `spirv-val`, plus what its module must declare (system
//! value built-ins, flat inputs, discards). The rendered pixels are kept
//! with each case for a device run. The host-side checks follow: misc
//! position exports (layer, viewport index, point size, clip distances),
//! the clip-disable clip-space transform, the second blend source and the
//! rectangle-list tessellation stages.

use super::*;
use crate::pipeline::compiler::{ComponentKind as K, ReflectedVariable};
use ps5shader::resources::{apply_specialization, materialize, BindingLayout, PixelAlphaBlendSource, ResourceRuntimeInputs, ShaderResourcePlan, ShaderStage as St};
use ps5shader::{decode_blob, ColorComponentMapping, PixelOutputBinding};
use rspirv::spirv::{BuiltIn, Decoration, ExecutionMode, Op};

const CODE_ADDRESS: u64 = 0x10_0000_0000;

// ---- encoders (GFX10) ----

const fn inline_u32(v: u32) -> u32 {
    128 + v
}
const fn vgpr(r: u32) -> u32 {
    256 + r
}
const fn s_mov_b32(dst: u32, src: u32) -> u32 {
    0x8000_0000 | 0x7D << 23 | (dst & 0x7F) << 16 | 0x03 << 8 | (src & 0xFF)
}
const fn sop1(op: u32, dst: u32, src: u32) -> u32 {
    0x8000_0000 | 0x7D << 23 | (dst & 0x7F) << 16 | (op & 0xFF) << 8 | (src & 0xFF)
}
const fn sopc(op: u32, src0: u32, src1: u32) -> u32 {
    0x8000_0000 | 0x7E << 23 | (op & 0x7F) << 16 | (src1 & 0xFF) << 8 | (src0 & 0xFF)
}
const fn sopp(op: u32, simm: u32) -> u32 {
    0x8000_0000 | 0x7F << 23 | (op & 0x7F) << 16 | (simm & 0xFFFF)
}
const fn vop1(op: u32, dst: u32, src0: u32) -> u32 {
    0x3F << 25 | (dst & 0xFF) << 17 | (op & 0xFF) << 9 | (src0 & 0x1FF)
}
const fn vop2(op: u32, dst: u32, src0: u32, src1: u32) -> u32 {
    (op & 0x3F) << 25 | (dst & 0xFF) << 17 | (src1 & 0xFF) << 9 | (src0 & 0x1FF)
}
const fn vopc(op: u32, src0: u32, src1: u32) -> u32 {
    0x3E << 25 | (op & 0xFF) << 17 | (src1 & 0xFF) << 9 | (src0 & 0x1FF)
}
const fn vintrp(op: u32, dst: u32, attr: u32, chan: u32, src: u32) -> u32 {
    0x32 << 26 | (dst & 0xFF) << 18 | (op & 3) << 16 | (attr & 0x3F) << 10 | (chan & 3) << 8 | (src & 0xFF)
}
const fn exp0(target: u32, en: u32, done: bool, compr: bool, vm: bool) -> u32 {
    0x3E << 26 | (target & 0x3F) << 4 | (en & 0xF) | (compr as u32) << 10 | (done as u32) << 11 | (vm as u32) << 12
}
const fn exp1(s0: u32, s1: u32, s2: u32, s3: u32) -> u32 {
    (s0 & 0xFF) | (s1 & 0xFF) << 8 | (s2 & 0xFF) << 16 | (s3 & 0xFF) << 24
}
const fn ds0(op: u32, offset: u32) -> u32 {
    0x36 << 26 | (op & 0xFF) << 18 | (offset & 0xFFFF)
}
const fn ds1(vdst: u32, data0: u32, addr: u32) -> u32 {
    (vdst & 0xFF) << 24 | (data0 & 0xFF) << 8 | (addr & 0xFF)
}
const END: u32 = 0xBF81_0000;

fn v_mov_literal(code: &mut Vec<u32>, dst: u32, literal: u32) {
    code.push(vop1(0x01, dst, 255));
    code.push(literal);
}
fn v_mov_u32(code: &mut Vec<u32>, dst: u32, value: u32) {
    if value <= 64 {
        code.push(vop1(0x01, dst, inline_u32(value)));
    } else {
        v_mov_literal(code, dst, value);
    }
}
fn vop3(code: &mut Vec<u32>, op: u32, dst: u32, src0: u32, src1: u32, src2: u32) {
    code.push(0x35 << 26 | (op & 0x3FF) << 16 | (dst & 0xFF));
    code.push((src0 & 0x1FF) | (src1 & 0x1FF) << 9 | (src2 & 0x1FF) << 18);
}

// ---- cases ----

/// SPI_PS_INPUT_ENA bits.
const PERSP_SAMPLE: u32 = 1 << 0;
const PERSP_CENTER: u32 = 1 << 1;
const PERSP_CENTROID: u32 = 1 << 2;
const POS_W: u32 = 1 << 11;
const FRONT_FACE: u32 = 1 << 12;
const ANCILLARY: u32 = 1 << 13;

#[derive(Clone)]
struct GraphicsCase {
    name: &'static str,
    code: Vec<u32>,
    /// The rendered pixel (kept for a device run).
    #[allow(dead_code)]
    expected_pixel: Vec<u32>,
    /// Interpolator settings of inputs 0.. (SPI_PS_INPUT_CNTL; the rest
    /// map input `i` to parameter `i`).
    interpolators: Vec<u32>,
    custom_interpolation: u32,
    input_ena: u32,
    user_data: Option<[u32; 4]>,
    /// MRT0 export format (SPI_SHADER_COL_FORMAT).
    col_format: u32,
    /// SPI_BARYC_CNTL.FRONT_FACE_ALL_BITS: the front-face input reads
    /// 0xFFFFFFFF/0 instead of +1.0f/-1.0f.
    front_face_all_bits: bool,
    /// PS_ITER_SAMPLE on a multisampled target: the program runs per sample.
    sample_rate: bool,
    /// Built-ins the module must read.
    builtins: &'static [BuiltIn],
    /// The module must discard (EXEC-masked VM exports).
    discards: bool,
    flat: bool,
}

impl GraphicsCase {
    fn new(name: &'static str, code: Vec<u32>, expected_pixel: Vec<u32>) -> Self {
        GraphicsCase {
            name,
            code,
            expected_pixel,
            interpolators: Vec::new(),
            custom_interpolation: 0,
            input_ena: PERSP_CENTER,
            user_data: None,
            col_format: 9,
            front_face_all_bits: false,
            sample_rate: false,
            builtins: &[],
            discards: false,
            flat: false,
        }
    }
}

fn interpolation_export() -> GraphicsCase {
    let mut code = vec![vintrp(0x00, 0, 0, 0, 0), vintrp(0x01, 0, 0, 0, 0), vintrp(0x02, 1, 0, 1, 2)];
    v_mov_literal(&mut code, 2, 0x3f40_0000);
    v_mov_literal(&mut code, 3, 0x3f80_0000);
    code.extend([exp0(0, 0xF, true, false, false), exp1(0, 1, 2, 3), END]);
    GraphicsCase::new("GraphicsInterpolationExport", code, vec![0x3e80_0000, 0x3f00_0000, 0x3f40_0000, 0x3f80_0000])
}

fn position_w_export() -> GraphicsCase {
    // POS_W occupies v2.
    let code = vec![exp0(0, 0xF, true, false, false), exp1(2, 2, 2, 2), END];
    GraphicsCase { input_ena: PERSP_CENTER | POS_W, builtins: &[BuiltIn::FragCoord], ..GraphicsCase::new("GraphicsPositionWExport", code, vec![0x4080_0000; 4]) }
}

fn packed_half_centroid() -> GraphicsCase {
    let code = vec![
        0xc812_0002, 0xc80e_0000, // raw vertex0/vertex1 packed halves
        0xcc20_4007, 0x9c12_06f2, // low(vertex1) - low(vertex0)
        0xcc20_7005, 0x9c12_06f2, // high(vertex1) - high(vertex0)
        0xc80a_0001, // raw vertex2
        0xcc20_4006, 0x9c12_04f2, 0xcc20_7003, 0x9c12_04f2, 0xcc20_400e, 0x0412_0f00, // low = I * delta10 + vertex0
        0xcc20_600f, 0x0412_0b00, // high = I * delta10 + vertex0
        0x3e1c_0306, 0x3e1e_0303, // += J * delta20
        exp0(0, 0xF, true, false, false),
        exp1(14, 15, 14, 15),
        END,
    ];
    GraphicsCase {
        input_ena: PERSP_CENTROID,
        interpolators: vec![0x420],
        custom_interpolation: 1,
        ..GraphicsCase::new("GraphicsPackedHalfCentroid", code, vec![0x3f60_0000, 0x3fd0_0000, 0x3f60_0000, 0x3fd0_0000])
    }
}

fn perspective_sample(multisample: bool) -> GraphicsCase {
    let code = vec![vop2(0x08, 2, vgpr(0), 0), vop2(0x08, 3, vgpr(1), 1), exp0(0, 0xF, true, false, false), exp1(2, 3, 0, 1), END];
    let square = if multisample { 21.0f32 / 256.0 } else { 1.0 / 16.0 }.to_bits();
    GraphicsCase {
        input_ena: PERSP_SAMPLE,
        ..GraphicsCase::new(if multisample { "GraphicsPerspectiveSample4x" } else { "GraphicsPerspectiveSample1x" }, code, vec![square, square, 0x3e80_0000, 0x3e80_0000])
    }
}

fn packed_half_input_alias(second_weights: bool) -> GraphicsCase {
    let mut t = packed_half_centroid();
    t.name = if second_weights { "GraphicsPackedHalfAliasSecondWeights" } else { "GraphicsPackedHalfInputAlias" };
    t.custom_interpolation = 2;
    t.interpolators = vec![0, 0x420];
    for w in [0, 1, 6] {
        t.code[w] |= 1 << 10;
    }
    t.code.truncate(t.code.len() - 3);
    t.code.push(vintrp(0x00, 16, 0, 1, 0));
    t.code.push(vintrp(0x01, 16, 0, 1, 1));
    v_mov_literal(&mut t.code, 17, 0x3f80_0000);
    t.code.extend([exp0(0, 0xF, true, false, false), exp1(14, 15, 16, 17), END]);
    t.input_ena = PERSP_CENTER | PERSP_CENTROID;
    t.expected_pixel = if second_weights { vec![0x3f80_0000, 0x3fe0_0000, 0x4080_0000, 0x3f80_0000] } else { vec![0x3f60_0000, 0x3fd0_0000, 0x4070_0000, 0x3f80_0000] };
    t
}

fn smooth_raw_input_alias() -> GraphicsCase {
    let mut t = packed_half_input_alias(false);
    t.name = "GraphicsSmoothRawInputAlias";
    t.input_ena = PERSP_CENTER;
    t.code = vec![vintrp(0x00, 0, 0, 1, 0), vintrp(0x01, 0, 0, 1, 1), vintrp(0x02, 1, 1, 1, 2), exp0(0, 0xF, true, false, false), exp1(0, 1, 0, 1), END];
    t.expected_pixel = vec![0x4070_0000, 0x4000_0000, 0x4070_0000, 0x4000_0000];
    t
}

fn smooth_flat_input_alias(flat_first: bool, provoking_last: bool) -> GraphicsCase {
    let mut t = smooth_raw_input_alias();
    t.name = match (provoking_last, flat_first) {
        (true, true) => "GraphicsFlatSmoothAliasLast",
        (true, false) => "GraphicsSmoothFlatAliasLast",
        (false, true) => "GraphicsFlatSmoothInputAlias",
        (false, false) => "GraphicsSmoothFlatInputAlias",
    };
    t.custom_interpolation = 0;
    t.interpolators = if flat_first { vec![0x400, 0] } else { vec![0, 0x400] };
    let smooth = flat_first as u32;
    let flat = smooth ^ 1;
    t.code = vec![
        vintrp(0x00, 0, smooth, 1, 0),
        vintrp(0x01, 0, smooth, 1, 1),
        vintrp(0x02, 1, flat, 2, 2),
        vop2(0x1b, 1, inline_u32(63), 1),
        vop1(0x06, 1, vgpr(1)),
        vintrp(0x00, 2, flat, 1, 0),
        vintrp(0x01, 2, flat, 1, 1),
        exp0(0, 0xF, true, false, false),
        exp1(0, 1, 2, 0),
        END,
    ];
    t.expected_pixel = vec![0x4070_0000, if provoking_last { 0x4130_0000 } else { 0x40a0_0000 }, if provoking_last { 0x4100_0000 } else { 0x4000_0000 }, 0x4070_0000];
    t.flat = true;
    t
}

fn ancillary_layer(front_face: bool) -> GraphicsCase {
    let ancillary = vgpr(if front_face { 3 } else { 2 });
    let layers = 8;
    let mut code = vec![sop1(0x04, 8, 126), sop1(0x0a, 126, 126)];
    vop3(&mut code, 0x148, 4, ancillary, inline_u32(16), inline_u32(11));
    vop3(&mut code, 0x149, 5, ancillary, inline_u32(16), inline_u32(3));
    code.push(sop1(0x04, 126, 8));
    v_mov_u32(&mut code, 6, layers);
    code.push(vopc(0xc1, vgpr(4), 6)); // dynamic layer < count
    code.push(sop1(0x04, 126, 106));
    code.push(vop1(0x06, 0, vgpr(4)));
    code.push(vop1(0x05, 1, vgpr(5)));
    v_mov_literal(&mut code, 2, 0);
    v_mov_literal(&mut code, 3, 0x3f80_0000);
    code.extend([exp0(0, 0xF, true, false, false), exp1(0, 1, 2, 3), END]);
    let mut expected = Vec::new();
    for layer in 0..layers {
        let signed = if layer < 4 { layer as i32 } else { layer as i32 - 8 };
        expected.extend([(layer as f32).to_bits(), (signed as f32).to_bits(), 0, 0x3f80_0000]);
    }
    GraphicsCase {
        input_ena: PERSP_CENTER | ANCILLARY | if front_face { FRONT_FACE } else { 0 },
        // The ancillary VGPR follows FRONT_FACE when both are enabled.
        builtins: &[BuiltIn::Layer],
        ..GraphicsCase::new(if front_face { "GraphicsAncillaryAfterFrontFace" } else { "GraphicsAncillaryLayer" }, code, expected)
    }
}

/// The front-face input exported unchanged (v2 after PERSP_CENTER).
fn front_face_export(all_bits: bool) -> GraphicsCase {
    let code = vec![exp0(0, 0xF, true, false, false), exp1(2, 2, 2, 2), END];
    GraphicsCase {
        input_ena: PERSP_CENTER | FRONT_FACE,
        front_face_all_bits: all_bits,
        builtins: &[BuiltIn::FrontFacing],
        ..GraphicsCase::new(
            if all_bits { "GraphicsFrontFaceAllBits" } else { "GraphicsFrontFaceExport" },
            code,
            // A front-facing triangle: +1.0f, or all one bits under
            // FRONT_FACE_ALL_BITS (measured on hardware).
            vec![if all_bits { 0xFFFF_FFFF } else { 0x3F80_0000 }; 4],
        )
    }
}

fn ancillary_sample_id() -> GraphicsCase {
    let mut code = Vec::new();
    vop3(&mut code, 0x148, 4, vgpr(2), inline_u32(8), inline_u32(4));
    vop3(&mut code, 0x149, 5, vgpr(2), inline_u32(8), inline_u32(2));
    vop3(&mut code, 0x148, 6, vgpr(2), inline_u32(9), inline_u32(1));
    code.extend([vop1(0x06, 0, vgpr(4)), vop1(0x05, 1, vgpr(5)), vop1(0x06, 2, vgpr(6))]);
    v_mov_literal(&mut code, 3, 0x3f80_0000);
    code.extend([exp0(0, 0xF, true, false, false), exp1(0, 1, 2, 3), END]);
    GraphicsCase {
        input_ena: PERSP_CENTER | ANCILLARY,
        sample_rate: true,
        builtins: &[BuiltIn::SampleId],
        ..GraphicsCase::new("GraphicsAncillarySampleId", code, vec![0x3fc0_0000, 0xbf00_0000, 0x3f00_0000, 0x3f80_0000])
    }
}

fn flat_interpolator_export() -> GraphicsCase {
    let mut code = vec![vintrp(0x02, 0, 0, 0, 2)];
    v_mov_literal(&mut code, 1, 0);
    v_mov_literal(&mut code, 2, 0);
    v_mov_literal(&mut code, 3, 0x3f80_0000);
    code.extend([exp0(0, 0xF, true, false, false), exp1(0, 1, 2, 3), END]);
    GraphicsCase { interpolators: vec![0x400], flat: true, ..GraphicsCase::new("GraphicsFlatInterpolatorExport", code, vec![0x3e80_0000, 0, 0, 0x3f80_0000]) }
}

fn ds_addtid_scratch_export() -> GraphicsCase {
    let mut code = vec![s_mov_b32(124, inline_u32(0))];
    v_mov_literal(&mut code, 0, 0x3f00_0000);
    code.extend([ds0(0xb0, 0), ds1(0, 0, 0), ds0(0xb1, 0), ds1(4, 0, 0), exp0(0, 0xF, true, false, false), exp1(4, 4, 4, 4), END]);
    GraphicsCase::new("GraphicsDsAddtidScratchExport", code, vec![0x3f00_0000; 4])
}

fn direct_sgpr_push_constant_export() -> GraphicsCase {
    let values = [0x3e80_0000, 0x3f00_0000, 0x3f40_0000, 0x3f80_0000];
    let code = vec![vop1(0x01, 0, 0), vop1(0x01, 1, 1), vop1(0x01, 2, 2), vop1(0x01, 3, 3), exp0(0, 0xF, true, false, false), exp1(0, 1, 2, 3), END];
    GraphicsCase { user_data: Some(values), ..GraphicsCase::new("GraphicsDirectSgprPushConstantExport", code, values.to_vec()) }
}

fn inline_srt_scalar_promotion_export() -> GraphicsCase {
    let values = [0, 0, 0, 0x3f80_0000];
    let mut code = Vec::new();
    vop3(&mut code, 0x12f, 0, 0, 1, 0);
    vop3(&mut code, 0x12f, 1, 2, 3, 0);
    code.extend([exp0(0, 0xF, true, true, true), exp1(0, 1, 0, 0), END]);
    GraphicsCase { user_data: Some(values), col_format: 4, ..GraphicsCase::new("GraphicsInlineSrtScalarPromotionExport", code, values.to_vec()) }
}

fn four_values(code: &mut Vec<u32>, first: u32, values: [u32; 4]) {
    for (k, v) in values.into_iter().enumerate() {
        v_mov_literal(code, first + k as u32, v);
    }
}

const HALVES: [u32; 4] = [0x3f00_0000, 0x3f40_0000, 0x3f80_0000, 0x4000_0000];
const LATER: [u32; 4] = [0x4040_0000, 0x4080_0000, 0x40a0_0000, 0x40c0_0000];

fn null_vm_export_discards() -> GraphicsCase {
    let mut code = Vec::new();
    four_values(&mut code, 0, HALVES);
    code.extend([exp0(0, 0xF, false, false, false), exp1(0, 1, 2, 3), sop1(0x04, 126, inline_u32(0)), exp0(9, 0, true, false, true), exp1(0, 0, 0, 0), END]);
    GraphicsCase { discards: true, ..GraphicsCase::new("GraphicsNullVmExportDiscardsInactiveExec", code, vec![0; 4]) }
}

fn mrt0_off_vm_export_discards() -> GraphicsCase {
    let mut code = Vec::new();
    four_values(&mut code, 0, HALVES);
    code.extend([exp0(0, 0xF, false, false, false), exp1(0, 1, 2, 3), sop1(0x04, 126, inline_u32(0)), exp0(0, 0, true, true, true), exp1(0, 0, 0, 0), END]);
    GraphicsCase { discards: true, ..GraphicsCase::new("GraphicsMrt0OffVmExportDiscardsInactiveExec", code, vec![0; 4]) }
}

fn final_vm_export_supersedes() -> GraphicsCase {
    let mut code = Vec::new();
    four_values(&mut code, 0, HALVES);
    code.extend([sop1(0x04, 126, inline_u32(0)), exp0(0, 0xF, false, false, true), exp1(0, 1, 2, 3), s_mov_b32(126, 193), s_mov_b32(127, 193)]);
    four_values(&mut code, 4, LATER);
    code.extend([exp0(0, 0xF, true, false, true), exp1(4, 5, 6, 7), END]);
    GraphicsCase::new("GraphicsFinalVmExportSupersedesEarlierVmMask", code, LATER.to_vec())
}

fn branch_path_final_vm_export_discards() -> GraphicsCase {
    let mut code = Vec::new();
    four_values(&mut code, 0, HALVES);
    code.extend([exp0(0, 0xF, false, false, false), exp1(0, 1, 2, 3), sop1(0x04, 126, inline_u32(0)), exp0(9, 0, true, false, true), exp1(0, 0, 0, 0)]);
    code.push(sopc(0x06, inline_u32(0), inline_u32(1)));
    let branch = code.len();
    code.push(0);
    code.push(END);
    let later = code.len();
    code[branch] = sopp(0x05, (later - branch - 1) as u32);
    code.extend([s_mov_b32(126, 193), s_mov_b32(127, 193)]);
    four_values(&mut code, 4, LATER);
    code.extend([exp0(0, 0xF, true, false, true), exp1(4, 5, 6, 7), END]);
    GraphicsCase { discards: true, ..GraphicsCase::new("GraphicsBranchPathFinalVmExportDiscardsInactiveExec", code, vec![0; 4]) }
}

fn graphics_cases() -> Vec<GraphicsCase> {
    vec![
        interpolation_export(),
        position_w_export(),
        packed_half_centroid(),
        perspective_sample(false),
        perspective_sample(true),
        packed_half_input_alias(false),
        packed_half_input_alias(true),
        smooth_raw_input_alias(),
        smooth_flat_input_alias(false, false),
        smooth_flat_input_alias(true, false),
        smooth_flat_input_alias(false, true),
        smooth_flat_input_alias(true, true),
        ancillary_layer(false),
        ancillary_layer(true),
        ancillary_sample_id(),
        front_face_export(false),
        front_face_export(true),
        flat_interpolator_export(),
        ds_addtid_scratch_export(),
        direct_sgpr_push_constant_export(),
        inline_srt_scalar_promotion_export(),
        null_vm_export_discards(),
        mrt0_off_vm_export_discards(),
        final_vm_export_supersedes(),
        branch_path_final_vm_export_discards(),
    ]
}

// ---- compilation ----

fn spirv_val() -> Option<std::path::PathBuf> {
    ["/usr/bin/spirv-val", "/usr/local/bin/spirv-val"].iter().map(std::path::PathBuf::from).find(|p| p.exists())
}

fn validate(name: &str, spirv: &[u32]) {
    let Some(val) = spirv_val() else { return };
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = std::env::temp_dir().join(format!("rps5e-gfx-{}-{seq}-{name}.spv", std::process::id()));
    let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(&tmp, bytes).unwrap();
    let o = std::process::Command::new(val).arg("--target-env").arg("vulkan1.3").arg(&tmp).output().unwrap();
    assert!(o.status.success(), "{name}: spirv-val: {}", String::from_utf8_lossy(&o.stderr));
    let _ = std::fs::remove_file(&tmp);
}

/// The user data of a case (the reference harness's defaults: V#s for
/// s[0:3] and s[48:51]).
fn native_user_data(values: Option<[u32; 4]>) -> Vec<u32> {
    let mut d = vec![0u32; 64];
    match values {
        Some(v) => d[..4].copy_from_slice(&v),
        None => {
            d[3] = 3 << 28;
            d[50] = 1 << 20;
            d[51] = 3 << 28;
        }
    }
    d
}

struct Stage<'a> {
    code: &'a [u32],
    stage: St,
    user: Vec<u32>,
}

/// Compile one program with `adjust` filling the stage options.
fn compile(s: &Stage, adjust: &dyn Fn(&mut ShaderCompileRequest<'_>)) -> Result<CompiledShader, String> {
    compile_as(s, adjust, None)
}

/// [`compile`], as a mesh workgroup when `mesh` is given.
fn compile_as(s: &Stage, adjust: &dyn Fn(&mut ShaderCompileRequest<'_>), mesh: Option<crate::pipeline::compiler::MeshConfig>) -> Result<CompiledShader, String> {
    let bytes: Vec<u8> = s.code.iter().flat_map(|w| w.to_le_bytes()).collect();
    let program = decode_blob(&bytes, CODE_ADDRESS).map_err(|e| format!("decode: {e}"))?;
    let user = BindingLayout::collect_user_data_registers(&program, 0, 16);
    let gds = BindingLayout::uses_global_data_share(&program);
    let shader_base = BindingLayout::reads_shader_base(&program);
    let plan = ShaderResourcePlan::extract(program, s.stage, 1, 0, 16, None, 32).map_err(|e| format!("plan: {e:?}"))?;
    let code = s.code;
    let read = move |a: u64| -> Option<u32> {
        let off = a.checked_sub(CODE_ADDRESS)?;
        code.get((off / 4) as usize).copied()
    };
    let mut inputs = ResourceRuntimeInputs::new(&s.user, CODE_ADDRESS);
    inputs.read_memory = Some(&read);
    let (_, spec) = materialize(&plan, &inputs).map_err(|e| format!("materialize: {e:?}"))?;
    let resources = apply_specialization(&plan, &spec).map_err(|e| format!("specialize: {e}"))?;
    let flat = ShaderCompileRequest::requires_flattened_table(&plan, &resources);
    let mut layout = BindingLayout::allocate(&resources.info, user, gds, flat, shader_base, 0, false, false).map_err(|e| format!("layout: {e}"))?;
    if mesh.is_some() {
        // The entry point reads the index buffer and its draw words.
        use ps5shader::resources::{DescriptorBinding, DescriptorBindingKind as K};
        for kind in [K::DeviceAddressPageTable, K::FaultBuffer, K::ShaderData] {
            if layout.find(kind).is_none() {
                layout.descriptors.push(DescriptorBinding { kind, resources: vec![] });
            }
        }
    }
    let mut r = ShaderCompileRequest::new(&plan, resources, layout);
    r.wave_size = 32;
    adjust(&mut r);
    let features = Features { barycentric: true, demote: true, subgroup_size: 32, subgroup64_compute: false, max_compute_workgroup_subgroups: 32, max_shared_bytes: 49152, sample_offset: false, image_int64_atomics: true, shared_int64_atomics: true, compute_derivatives: true, compute_derivatives_linear: true, preserve_inf_nan: [true; 3], denorm_preserve: [true; 3], denorm_flush_to_zero32: true, f16_rte: false, device_clock: Some(crate::clock::DeviceClock { num: 1, den: 10 }), subgroup_clock: true, bindless_tables: true, no_float64: false, amd_proprietary: false, emit: super::EmitOptions::ALL };
    SpirvCompiler::new(features, None).compile_mesh_request(s.stage, &r, 0, mesh).map_err(|e| format!("compile: {e}"))
}

fn pixel_request(c: &GraphicsCase, r: &mut ShaderCompileRequest<'_>) {
    r.pixel_outputs = vec![PixelOutputBinding { component_mapping: ColorComponentMapping::IDENTITY, ..PixelOutputBinding::new(0, 0, ps5shader::PixelOutputKind::Float) }];
    r.color_export_format = c.col_format;
    r.pixel_input_enable = c.input_ena;
    r.pixel_input_address = c.input_ena;
    r.pixel_custom_interpolation_mask = c.custom_interpolation;
    let mut cntl: Vec<u32> = (0..32).collect();
    for (i, &s) in c.interpolators.iter().enumerate() {
        cntl[i] = s;
    }
    r.pixel_input_cntl = Some(cntl);
    r.front_face_all_bits = c.front_face_all_bits;
    r.pixel_sample_rate = c.sample_rate;
}

// ---- module inspection ----

/// Every instruction of a module: (opcode, operand words).
fn instructions(spirv: &[u32]) -> Vec<(u32, &[u32])> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < spirv.len() {
        let n = ((spirv[i] >> 16) as usize).max(1);
        out.push((spirv[i] & 0xFFFF, &spirv[i + 1..(i + n).min(spirv.len())]));
        i += n;
    }
    out
}

fn has_op(spirv: &[u32], op: Op) -> bool {
    instructions(spirv).iter().any(|(o, _)| *o == op as u32)
}

fn decorations(spirv: &[u32], d: Decoration) -> Vec<Vec<u32>> {
    instructions(spirv).into_iter().filter(|(o, w)| *o == Op::Decorate as u32 && w.get(1) == Some(&(d as u32))).map(|(_, w)| w[2..].to_vec()).collect()
}

fn has_builtin(spirv: &[u32], b: BuiltIn) -> bool {
    decorations(spirv, Decoration::BuiltIn).iter().any(|v| v.first() == Some(&(b as u32)))
}

/// Every position output (a `BuiltIn Position` variable or block member)
/// is also decorated `Invariant`, and there is at least one.
fn positions_invariant(spirv: &[u32]) -> bool {
    let ops = instructions(spirv);
    let builtin = Decoration::BuiltIn as u32;
    let invariant = Decoration::Invariant as u32;
    let position = BuiltIn::Position as u32;
    let members: Vec<(u32, u32)> = ops
        .iter()
        .filter(|(o, w)| *o == Op::MemberDecorate as u32 && w.get(2) == Some(&builtin) && w.get(3) == Some(&position))
        .map(|(_, w)| (w[0], w[1]))
        .collect();
    let vars: Vec<u32> = ops.iter().filter(|(o, w)| *o == Op::Decorate as u32 && w.get(1) == Some(&builtin) && w.get(2) == Some(&position)).map(|(_, w)| w[0]).collect();
    let member_invariant = |m: &(u32, u32)| ops.iter().any(|(o, w)| *o == Op::MemberDecorate as u32 && w.first() == Some(&m.0) && w.get(1) == Some(&m.1) && w.get(2) == Some(&invariant));
    let var_invariant = |v: &u32| ops.iter().any(|(o, w)| *o == Op::Decorate as u32 && w.first() == Some(v) && w.get(1) == Some(&invariant));
    (!members.is_empty() || !vars.is_empty()) && members.iter().all(member_invariant) && vars.iter().all(var_invariant)
}

fn has_mode(spirv: &[u32], m: ExecutionMode) -> bool {
    instructions(spirv).iter().any(|(o, w)| *o == Op::ExecutionMode as u32 && w.get(1) == Some(&(m as u32)))
}

// ---- tests ----

#[test]
fn graphics_cases_compile_to_spirv() {
    let cases = graphics_cases();
    assert_eq!(cases.len(), 25);
    for c in &cases {
        let s = Stage { code: &c.code, stage: St::Pixel, user: native_user_data(c.user_data) };
        let out = compile(&s, &|r| pixel_request(c, r)).unwrap_or_else(|e| panic!("{}: {e}", c.name));
        validate(c.name, &out.spirv);
        for &b in c.builtins {
            assert!(has_builtin(&out.spirv, b), "{}: no {b:?} input", c.name);
        }
        if c.discards {
            assert!(has_op(&out.spirv, Op::Kill) || has_op(&out.spirv, Op::DemoteToHelperInvocation) || has_op(&out.spirv, Op::TerminateInvocation), "{}: no discard", c.name);
        }
        if c.flat {
            // Flat inputs read the provoking vertex of the per-vertex attribute.
            assert!(!decorations(&out.spirv, Decoration::Flat).is_empty() || !decorations(&out.spirv, Decoration::PerVertexKHR).is_empty(), "{}: no flat input", c.name);
        }
        assert_eq!(out.reflection.outputs.len(), 1, "{}: one color output", c.name);
    }
}

/// The words of every 32-bit SPIR-V constant (front-face select arms:
/// 0x3f800000/0xbf800000, or 0xffffffff/0 under FRONT_FACE_ALL_BITS).
fn u32_constant_bits(spirv: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < spirv.len() {
        let n = ((spirv[i] >> 16) as usize).max(1);
        if spirv[i] & 0xFFFF == Op::Constant as u32 && n == 4 {
            out.push(spirv[i + 3]);
        }
        i += n;
    }
    out
}

#[test]
fn front_face_all_bits_selects_the_measured_arms() {
    for (all_bits, front, back) in [(false, 0x3F80_0000u32, 0xBF80_0000u32), (true, 0xFFFF_FFFF, 0)] {
        let c = front_face_export(all_bits);
        let s = Stage { code: &c.code, stage: St::Pixel, user: native_user_data(c.user_data) };
        let out = compile(&s, &|r| pixel_request(&c, r)).unwrap_or_else(|e| panic!("{}: {e}", c.name));
        validate(c.name, &out.spirv);
        let consts = u32_constant_bits(&out.spirv);
        assert!(consts.contains(&front), "{}: +front arm {front:#x} missing", c.name);
        assert!(consts.contains(&back), "{}: back arm {back:#x} missing", c.name);
        let absent = if all_bits { 0x3F80_0000 } else { 0xFFFF_FFFF };
        assert!(!consts.contains(&absent), "{}: arm {absent:#x} of the other mode present", c.name);
    }
}

/// A vertex program exporting a position and POS1 = (point size, -, layer
/// | viewport << 16, -) and POS2 = four clip / cull distances.
fn misc_vertex_program() -> Vec<u32> {
    let mut code = Vec::new();
    for (k, v) in [0.0f32, 0.0, 0.5, 1.0].into_iter().enumerate() {
        v_mov_literal(&mut code, k as u32, v.to_bits());
    }
    v_mov_literal(&mut code, 4, 2.0f32.to_bits());
    v_mov_literal(&mut code, 5, 0);
    v_mov_literal(&mut code, 6, 3 | 5 << 16);
    v_mov_literal(&mut code, 7, 0);
    code.extend([exp0(12, 0xF, false, false, false), exp1(0, 1, 2, 3), exp0(13, 0xF, false, false, false), exp1(4, 5, 6, 7)]);
    code.extend([exp0(14, 0xF, true, false, false), exp1(0, 1, 2, 3), END]);
    code
}

#[test]
fn misc_position_exports() {
    let code = misc_vertex_program();
    let s = Stage { code: &code, stage: St::Vertex, user: native_user_data(None) };
    // VS_OUT_MISC_VEC_ENA + USE_VTX_POINT_SIZE + RENDER_TARGET_INDX +
    // VIEWPORT_INDX, VS_OUT_CCDIST0_VEC_ENA with clip 0, 1 and cull 2, 3.
    let control = 1 << 21 | 1 << 16 | 1 << 18 | 1 << 19 | 1 << 22 | 0x3 | 0xC << 8;
    let out = compile(&s, &|r| r.position_export_control = control).unwrap();
    validate("misc_position_exports", &out.spirv);
    for b in [BuiltIn::Layer, BuiltIn::ViewportIndex, BuiltIn::PointSize, BuiltIn::ClipDistance, BuiltIn::CullDistance] {
        assert!(has_builtin(&out.spirv, b), "no {b:?}");
    }
    assert!(out.reflection.writes_viewport_index && out.reflection.writes_point_size);
    // A depth pre-pass and an EQUAL-tested pass must compute one position.
    assert!(positions_invariant(&out.spirv), "the vertex position is not Invariant");
    // Without VS_OUT_CNTL the misc vector carries nothing.
    let out = compile(&s, &|r| r.position_export_control = 0).unwrap();
    assert!(!has_builtin(&out.spirv, BuiltIn::ViewportIndex) && !out.reflection.writes_viewport_index);
    // Decoding: POS1 is the misc vector, POS2 the first distance vector.
    use ps5shader::resources::decode_position_export_component as decode;
    assert!(decode(control, 1, 2).layer && decode(control, 1, 2).viewport && decode(control, 1, 0).point_size);
    assert_eq!(decode(control, 2, 1).clip_distance, Some(1));
    assert_eq!(decode(control, 2, 3).cull_distance, Some(1));
    assert_eq!(decode(control, 3, 0), Default::default());
}

/// As a mesh workgroup the layer and viewport index of the misc vector are
/// per-primitive outputs, placed by PA_CL_VS_OUT_CNTL like the vertex path.
#[test]
fn mesh_misc_position_exports() {
    let code = misc_vertex_program();
    let s = Stage { code: &code, stage: St::Vertex, user: native_user_data(None) };
    let mesh = crate::pipeline::compiler::MeshConfig {
        lines: false,
        points: false,
        input_primitive: 4,
        primitives_per_group: 21,
        vertices_per_group: 63,
        max_vertices: 63,
        max_primitives: 21,
        threads: 64,
        esgs_item_size: 4,
        draw_dword: 64,
    };
    let control = 1 << 21 | 1 << 18 | 1 << 19;
    let out = compile_as(&s, &|r| r.position_export_control = control, Some(mesh)).unwrap();
    validate("mesh_misc_position_exports", &out.spirv);
    assert!(positions_invariant(&out.spirv), "the mesh position is not Invariant");
    for b in [BuiltIn::Layer, BuiltIn::ViewportIndex] {
        assert!(has_builtin(&out.spirv, b), "no {b:?}");
    }
    assert!(out.reflection.writes_viewport_index);
    // Without VS_OUT_CNTL the misc vector carries neither.
    let out = compile_as(&s, &|r| r.position_export_control = 0, Some(mesh)).unwrap();
    validate("mesh_misc_position_exports_none", &out.spirv);
    assert!(!has_builtin(&out.spirv, BuiltIn::ViewportIndex) && !has_builtin(&out.spirv, BuiltIn::Layer));
    assert!(!out.reflection.writes_viewport_index);
}

/// Whether the module's position output (a `gl_PerVertex` member or a mesh
/// output array) is decorated Invariant.
fn position_invariant(spirv: &[u32]) -> bool {
    let ins = instructions(spirv);
    let member = |w: &[u32], d: Decoration| w.get(2) == Some(&(d as u32));
    let members: Vec<(u32, u32)> = ins.iter().filter(|(o, w)| *o == Op::MemberDecorate as u32 && member(w, Decoration::BuiltIn) && w.get(3) == Some(&(BuiltIn::Position as u32))).map(|(_, w)| (w[0], w[1])).collect();
    let variables: Vec<u32> = ins.iter().filter(|(o, w)| *o == Op::Decorate as u32 && w.get(1) == Some(&(Decoration::BuiltIn as u32)) && w.get(2) == Some(&(BuiltIn::Position as u32))).map(|(_, w)| w[0]).collect();
    assert!(!members.is_empty() || !variables.is_empty(), "no position output");
    let member_ok = members.iter().all(|&(s, i)| ins.iter().any(|(o, w)| *o == Op::MemberDecorate as u32 && w.first() == Some(&s) && w.get(1) == Some(&i) && member(w, Decoration::Invariant)));
    let variable_ok = variables.iter().all(|&v| ins.iter().any(|(o, w)| *o == Op::Decorate as u32 && w.first() == Some(&v) && w.get(1) == Some(&(Decoration::Invariant as u32))));
    member_ok && variable_ok
}

/// Two vertex programs computing the same position give bit-identical
/// depth only with an Invariant position (depth pre-pass, then an EQUAL
/// redraw), in the vertex and the mesh forms.
#[test]
fn vertex_position_is_invariant() {
    let code = misc_vertex_program();
    let s = Stage { code: &code, stage: St::Vertex, user: native_user_data(None) };
    let out = compile(&s, &|_| {}).unwrap();
    validate("invariant_position", &out.spirv);
    assert!(position_invariant(&out.spirv), "vertex position is not Invariant");
    let mesh = crate::pipeline::compiler::MeshConfig {
        lines: false,
        points: false,
        input_primitive: 4,
        primitives_per_group: 21,
        vertices_per_group: 63,
        max_vertices: 63,
        max_primitives: 21,
        threads: 64,
        esgs_item_size: 4,
        draw_dword: 64,
    };
    let out = compile_as(&s, &|_| {}, Some(mesh)).unwrap();
    validate("invariant_mesh_position", &out.spirv);
    assert!(position_invariant(&out.spirv), "mesh position is not Invariant");
}

/// A geometry stage emitting line strips runs as a mesh workgroup that
/// outputs lines (two vertex indices per primitive).
#[test]
fn mesh_line_strip_output() {
    let code = misc_vertex_program();
    let s = Stage { code: &code, stage: St::Vertex, user: native_user_data(None) };
    let mesh = crate::pipeline::compiler::MeshConfig {
        lines: true,
        points: false,
        input_primitive: 2,
        primitives_per_group: 32,
        vertices_per_group: 64,
        max_vertices: 64,
        max_primitives: 32,
        threads: 64,
        esgs_item_size: 4,
        draw_dword: 64,
    };
    let out = compile_as(&s, &|_| {}, Some(mesh)).unwrap();
    validate("mesh_lines", &out.spirv);
    assert!(has_builtin(&out.spirv, BuiltIn::PrimitiveLineIndicesEXT), "no line indices");
    assert!(!has_builtin(&out.spirv, BuiltIn::PrimitiveTriangleIndicesEXT));
    assert!(has_mode(&out.spirv, ExecutionMode::OutputLinesEXT));
}

/// A geometry stage emitting point lists runs as a mesh workgroup that
/// outputs points (one vertex index per primitive; reference ee81c5112).
#[test]
fn mesh_point_list_output() {
    let code = misc_vertex_program();
    let s = Stage { code: &code, stage: St::Vertex, user: native_user_data(None) };
    let mesh = crate::pipeline::compiler::MeshConfig {
        lines: false,
        points: true,
        input_primitive: 1,
        primitives_per_group: 64,
        vertices_per_group: 64,
        max_vertices: 64,
        max_primitives: 64,
        threads: 64,
        esgs_item_size: 4,
        draw_dword: 64,
    };
    let out = compile_as(&s, &|_| {}, Some(mesh)).unwrap();
    validate("mesh_points", &out.spirv);
    assert!(has_builtin(&out.spirv, BuiltIn::PrimitivePointIndicesEXT), "no point indices");
    assert!(!has_builtin(&out.spirv, BuiltIn::PrimitiveTriangleIndicesEXT));
    assert!(has_mode(&out.spirv, ExecutionMode::OutputPoints));
}

/// With alpha-to-coverage the MRT0 export also writes the sample mask
/// (every sample covered from the start); without it there is none.
#[test]
fn alpha_to_coverage_writes_the_sample_mask() {
    let c = interpolation_export();
    let s = Stage { code: &c.code, stage: St::Pixel, user: native_user_data(c.user_data) };
    let plain = compile(&s, &|r| pixel_request(&c, r)).unwrap();
    assert!(!has_builtin(&plain.spirv, BuiltIn::SampleMask));
    let covered = compile(&s, &|r| {
        pixel_request(&c, r);
        r.pixel_alpha_to_coverage = true;
    })
    .unwrap();
    validate("alpha_to_coverage", &covered.spirv);
    assert!(has_builtin(&covered.spirv, BuiltIn::SampleMask), "no SampleMask output");
    assert!(has_op(&covered.spirv, Op::FOrdGreaterThanEqual), "no alpha threshold");
    assert!(!has_mode(&covered.spirv, ExecutionMode::EarlyFragmentTests));
}

/// An 11_11_10 UNORM target is stored through a 32-bit unsigned output:
/// the export clamps, scales (2047 / 2047 / 1023), rounds and packs its
/// first three components into the first channel.
#[test]
fn packed_unorm_11_11_10_export() {
    let c = interpolation_export();
    let s = Stage { code: &c.code, stage: St::Pixel, user: native_user_data(c.user_data) };
    for truncate in [false, true] {
        let out = compile(&s, &|r| {
            pixel_request(&c, r);
            r.pixel_outputs[0].kind = ps5shader::PixelOutputKind::Uint;
            r.pixel_outputs[0].packed_unorm_11_11_10 = Some(truncate);
        })
        .unwrap();
        validate("packed_unorm_11_11_10", &out.spirv);
        assert!(has_op(&out.spirv, Op::ConvertFToU), "no conversion to fields");
        assert!(has_op(&out.spirv, Op::ShiftLeftLogical), "no field packing");
        assert_eq!(has_op(&out.spirv, Op::FAdd), !truncate, "rounding half up adds one half");
    }
}

/// The stencil test value of the MRTZ export (y of 32_GR / 32_ABGR)
/// becomes FragStencilRefEXT when the DB takes it; otherwise there is none.
#[test]
fn stencil_test_value_export() {
    let mut code = vec![vintrp(0x00, 0, 0, 0, 0), vintrp(0x01, 0, 0, 0, 0)];
    v_mov_literal(&mut code, 1, 0x3f00_0000);
    v_mov_u32(&mut code, 2, 7);
    code.extend([exp0(0, 0xF, false, false, false), exp1(0, 0, 0, 0), exp0(8, 0x3, true, false, false), exp1(1, 2, 0, 0), END]);
    let c = GraphicsCase::new("GraphicsStencilExport", code, vec![0; 4]);
    let s = Stage { code: &c.code, stage: St::Pixel, user: native_user_data(c.user_data) };
    let plain = compile(&s, &|r| pixel_request(&c, r)).unwrap();
    assert!(!has_builtin(&plain.spirv, BuiltIn::FragStencilRefEXT));
    let exported = compile(&s, &|r| {
        pixel_request(&c, r);
        r.pixel_stencil_export = true;
    })
    .unwrap();
    validate("stencil_export", &exported.spirv);
    assert!(has_builtin(&exported.spirv, BuiltIn::FragStencilRefEXT), "no FragStencilRefEXT output");
    assert!(has_mode(&exported.spirv, ExecutionMode::StencilRefReplacingEXT));
    // The stencil operation value (MRTZ.y bits 8..15) is exported the same way.
    let op = compile(&s, &|r| {
        pixel_request(&c, r);
        r.pixel_stencil_op_export = true;
    })
    .unwrap();
    validate("stencil_op_export", &op.spirv);
    assert!(has_builtin(&op.spirv, BuiltIn::FragStencilRefEXT), "no FragStencilRefEXT output");
    assert_ne!(op.spirv, exported.spirv, "the operation value is not the low bits");
}

/// Per-pixel PA_SC_AA_MASK: the sample mask starts from the mask of the
/// pixel's quad position (the window position's parity).
#[test]
fn quad_sample_masks_start_the_sample_mask() {
    let c = interpolation_export();
    let s = Stage { code: &c.code, stage: St::Pixel, user: native_user_data(c.user_data) };
    let masked = compile(&s, &|r| {
        pixel_request(&c, r);
        r.pixel_quad_sample_masks = Some([1, 2, 3, 0]);
        r.pixel_alpha_to_coverage = true;
    })
    .unwrap();
    validate("quad_sample_masks", &masked.spirv);
    assert!(has_builtin(&masked.spirv, BuiltIn::SampleMask), "no SampleMask output");
    assert!(has_builtin(&masked.spirv, BuiltIn::FragCoord), "no window position");
}

/// With the masks in the shader data the module is the same for every
/// mask (a game stepping a dithered fade through the masks compiles one
/// module, not one per step); as constants every mask is a module of its
/// own.
#[test]
fn quad_sample_masks_from_shader_data_make_one_module() {
    let c = interpolation_export();
    let s = Stage { code: &c.code, stage: St::Pixel, user: native_user_data(c.user_data) };
    let with = |masks: [u32; 4], data: bool| {
        compile(&s, &|r| {
            pixel_request(&c, r);
            if data {
                r.bindings.add_quad_masks();
                r.pixel_quad_sample_masks = Some(crate::pipeline::provider::QUAD_MASKS_AT_RUN_TIME);
            } else {
                r.pixel_quad_sample_masks = Some(masks);
            }
        })
        .unwrap()
    };
    let (a, b) = (with([1, 0, 0, 0], true), with([0, 1, 1, 0], true));
    validate("quad_sample_masks_data", &a.spirv);
    assert_eq!(a.spirv, b.spirv, "the masks are not part of the module");
    assert!(has_builtin(&a.spirv, BuiltIn::SampleMask), "no SampleMask output");
    assert_ne!(with([1, 0, 0, 0], false).spirv, with([0, 1, 1, 0], false).spirv);
}

/// Early depth and stencil tests (DEPTH_BEFORE_SHADER or an early Z order)
/// declare EarlyFragmentTests; otherwise the module leaves the choice to
/// the driver.
#[test]
fn early_fragment_tests_execution_mode() {
    let c = interpolation_export();
    let s = Stage { code: &c.code, stage: St::Pixel, user: native_user_data(c.user_data) };
    let plain = compile(&s, &|r| pixel_request(&c, r)).unwrap();
    assert!(!has_mode(&plain.spirv, ExecutionMode::EarlyFragmentTests));
    let early = compile(&s, &|r| {
        pixel_request(&c, r);
        r.pixel_early_fragment_tests = true;
    })
    .unwrap();
    validate("early_fragment_tests", &early.spirv);
    assert!(has_mode(&early.spirv, ExecutionMode::EarlyFragmentTests));
}

/// A primitive-ordered pixel program (PRIMITIVE_ORDERED_PIXEL_SHADER) runs
/// inside a pixel-ordered fragment interlock: one begin, an end before
/// the return.
#[test]
fn ordered_pixel_program_runs_in_an_interlock() {
    let c = interpolation_export();
    let s = Stage { code: &c.code, stage: St::Pixel, user: native_user_data(c.user_data) };
    let plain = compile(&s, &|r| pixel_request(&c, r)).unwrap();
    assert!(!has_mode(&plain.spirv, ExecutionMode::PixelInterlockOrderedEXT));
    assert!(!has_op(&plain.spirv, Op::BeginInvocationInterlockEXT));
    let ordered = compile(&s, &|r| {
        pixel_request(&c, r);
        r.pixel_ordered = true;
    })
    .unwrap();
    validate("ordered_pixel_program", &ordered.spirv);
    assert!(has_mode(&ordered.spirv, ExecutionMode::PixelInterlockOrderedEXT));
    let ops: Vec<u32> = instructions(&ordered.spirv).iter().map(|&(op, _)| op).collect();
    let count = |op: Op| ops.iter().filter(|&&o| o == op as u32).count();
    assert_eq!((count(Op::BeginInvocationInterlockEXT), count(Op::EndInvocationInterlockEXT)), (1, count(Op::Return)));
}

/// Unordered float relations (v_cmp_nge / ngt / nle / nlt) are emitted as
/// the negated inverse ordered relation, never as OpFUnord{Less,Greater}*.
#[test]
fn unordered_relations_emit_negated_ordered_compares() {
    let mut code = vec![vop1(0x06, 4, vgpr(0)), vop1(0x06, 5, vgpr(3))];
    v_mov_literal(&mut code, 11, 1.0f32.to_bits());
    for (k, op) in [0x09u32, 0x0B, 0x0C, 0x0E].into_iter().enumerate() {
        code.push(vopc(op, vgpr(4), 5));
        code.push(vop2(0x01, 6 + k as u32, inline_u32(0), 11));
    }
    code.extend([exp0(12, 0xF, true, false, false), exp1(6, 7, 8, 9), END]);
    let s = Stage { code: &code, stage: St::Vertex, user: native_user_data(None) };
    let out = compile(&s, &|_| {}).unwrap();
    validate("unordered_relations", &out.spirv);
    for op in [Op::FUnordLessThan, Op::FUnordLessThanEqual, Op::FUnordGreaterThan, Op::FUnordGreaterThanEqual] {
        assert!(!has_op(&out.spirv, op), "{op:?} emitted");
    }
    for op in [Op::FOrdLessThan, Op::FOrdLessThanEqual, Op::FOrdGreaterThan, Op::FOrdGreaterThanEqual, Op::LogicalNot] {
        assert!(has_op(&out.spirv, op), "no {op:?}");
    }
}

#[test]
fn clip_disabled_positions_use_the_host_clip_space() {
    let code = misc_vertex_program();
    let s = Stage { code: &code, stage: St::Vertex, user: native_user_data(None) };
    let plain = compile(&s, &|_| {}).unwrap();
    let t = ps5shader::resources::ShaderClipSpaceTransform {
        enabled: true,
        scale_x: 960.0,
        scale_y: -540.0,
        offset_x: 960.0,
        offset_y: 540.0,
        half_extent_x: 8192.0,
        half_extent_y: 8192.0,
    };
    let clipped = compile(&s, &|r| r.clip_space = t).unwrap();
    validate("clip_space", &clipped.spirv);
    assert!(!has_op(&plain.spirv, Op::FDiv) && has_op(&clipped.spirv, Op::FDiv));
}

#[test]
fn second_blend_source() {
    let mut code = Vec::new();
    four_values(&mut code, 0, HALVES);
    code.extend([exp0(0, 0xF, false, false, false), exp1(0, 1, 2, 3), exp0(1, 0xF, true, false, false), exp1(3, 2, 1, 0), END]);
    let s = Stage { code: &code, stage: St::Pixel, user: native_user_data(None) };
    let base = GraphicsCase { col_format: 0x99, ..GraphicsCase::new("SecondSource", code.clone(), vec![]) };
    let index1 = |spirv: &[u32]| decorations(spirv, Decoration::Index).iter().any(|v| v.first() == Some(&1));
    // Guest dual-source blending: MRT1 becomes Location 0, Index 1.
    let dual = compile(&s, &|r| {
        pixel_request(&base, r);
        r.pixel_dual_source_blend = true;
    })
    .unwrap();
    validate("dual_source", &dual.spirv);
    assert!(index1(&dual.spirv));
    // Remapped alpha: target 0's logical alpha as the second source.
    for source in [PixelAlphaBlendSource::SourceAlpha, PixelAlphaBlendSource::SourceAlphaOne, PixelAlphaBlendSource::SourceAlphaZero] {
        let out = compile(&s, &|r| {
            pixel_request(&base, r);
            r.pixel_outputs[0].component_mapping = ColorComponentMapping::new(0x1B);
            r.pixel_alpha_blend_source = source;
        })
        .unwrap();
        validate("alpha_source", &out.spirv);
        assert!(index1(&out.spirv), "{source:?}");
    }
    let plain = compile(&s, &|r| pixel_request(&base, r)).unwrap();
    assert!(!index1(&plain.spirv));
}

/// The rectangle-list stages (the host side of RECT_LIST draws).
#[test]
fn rect_list_shaders() {
    use crate::pipeline::rectlist::{control_shader, evaluation_shader};
    let params = [
        ReflectedVariable { location: 0, components: 4, kind: K::Float, flat: true },
        ReflectedVariable { location: 1, components: 4, kind: K::Float, flat: false },
    ];
    let control = control_shader(&params, 0).unwrap();
    let evaluation = evaluation_shader(&params, 0).unwrap();
    validate("rect_control", &control);
    validate("rect_evaluation", &evaluation);
    // Four control points, a quad domain with equal spacing, clockwise.
    assert!(instructions(&control).iter().any(|(o, w)| *o == Op::ExecutionMode as u32 && w.get(1) == Some(&(ExecutionMode::OutputVertices as u32)) && w.get(2) == Some(&4)));
    for m in [ExecutionMode::Quads, ExecutionMode::SpacingEqual, ExecutionMode::VertexOrderCw] {
        assert!(has_mode(&evaluation, m), "evaluation: no {m:?}");
    }
    // No geometry stage, the parameters keep their locations and the flat
    // one is not interpolated.
    for m in [&control, &evaluation] {
        assert!(!instructions(m).iter().any(|(o, w)| *o == Op::EntryPoint as u32 && w.first() == Some(&(rspirv::spirv::ExecutionModel::Geometry as u32))));
        let locations: Vec<u32> = decorations(m, Decoration::Location).iter().filter_map(|v| v.first().copied()).collect();
        assert!(locations.contains(&0) && locations.contains(&1));
        assert!(!decorations(m, Decoration::Flat).is_empty());
    }
    // The expanded rectangle keeps the vertex stage's invariant position.
    // (its input block, gl_in, carries no Invariant: only outputs may).
    let invariant_members = decorations(&evaluation, Decoration::Invariant).len()
        + instructions(&evaluation).iter().filter(|(o, w)| *o == Op::MemberDecorate as u32 && w.get(2) == Some(&(Decoration::Invariant as u32))).count();
    assert_eq!(invariant_members, 1, "the rectangle's output position is not the one Invariant position");
    // Position-only draws.
    validate("rect_control_empty", &control_shader(&[], 0).unwrap());
    validate("rect_evaluation_empty", &evaluation_shader(&[], 0).unwrap());
    // The vertex stage's clip distances are forwarded through both stages.
    let control = control_shader(&params, 3).unwrap();
    let evaluation = evaluation_shader(&params, 3).unwrap();
    validate("rect_control_clip", &control);
    validate("rect_evaluation_clip", &evaluation);
    for m in [&control, &evaluation] {
        assert!(has_builtin_input(m, BuiltIn::ClipDistance), "no gl_ClipDistance input");
        assert!(has_builtin(m, BuiltIn::ClipDistance));
    }
    assert_eq!(builtin_array_len(&evaluation, BuiltIn::ClipDistance).is_some(), true);
}

/// The evaluation stage rounds gl_TessCoord to the corner (0.5 added before
/// the conversion): a coordinate just below 1 is still corner 1.
#[test]
fn rect_list_corner_rounds_the_tess_coord() {
    let evaluation = crate::pipeline::rectlist::evaluation_shader(&[], 0).unwrap();
    assert!(u32_constant_bits(&evaluation).contains(&0.5f32.to_bits()), "no rounding half");
    assert!(has_op(&evaluation, Op::FAdd));
}

/// The 32-bit constant a value id resolves to through bitcasts and copies.
fn resolve_u32(spirv: &[u32], id: u32) -> Option<u32> {
    let ops = instructions(spirv);
    let mut id = id;
    for _ in 0..16 {
        let (op, w) = ops.iter().find(|(o, w)| {
            let has_result = matches!(Op::from_u32(*o), Some(Op::Constant | Op::Bitcast | Op::CopyObject | Op::ConstantNull));
            has_result && w.get(1) == Some(&id)
        })?;
        match Op::from_u32(*op)? {
            Op::Constant => return w.get(2).copied(),
            Op::ConstantNull => return Some(0),
            _ => id = *w.get(2)?,
        }
    }
    None
}

/// The four component words stored to the fragment's colour output.
fn stored_color(spirv: &[u32]) -> Vec<Option<u32>> {
    let ops = instructions(spirv);
    let outputs: Vec<u32> = ops.iter().filter(|(o, w)| *o == Op::Variable as u32 && w.get(2) == Some(&(rspirv::spirv::StorageClass::Output as u32))).map(|(_, w)| w[1]).collect();
    let (_, store) = ops.iter().find(|(o, w)| *o == Op::Store as u32 && w.first().is_some_and(|p| outputs.contains(p))).expect("a store to the colour output");
    let value = store[1];
    let (op, w) = ops
        .iter()
        .find(|(o, w)| matches!(Op::from_u32(*o), Some(Op::CompositeConstruct | Op::ConstantComposite)) && w.get(1) == Some(&value))
        .expect("a composite colour value");
    let _ = op;
    w[2..].iter().map(|&c| resolve_u32(spirv, c)).collect()
}

/// SPI_SHADER_COL_FORMAT 32_AR on GFX10: red is export channel 0 and alpha
/// export channel 1 (before GFX10 the alpha was channel 3); green and blue
/// are 0. `exp mrt0 v0, v1, off, off` with v0 = 0.25, v1 = 0.75 stores
/// (0.25, 0, 0, 0.75).
#[test]
fn export_32_ar_takes_alpha_from_channel_1() {
    let mut code = Vec::new();
    v_mov_literal(&mut code, 0, 0.25f32.to_bits());
    v_mov_literal(&mut code, 1, 0.75f32.to_bits());
    v_mov_literal(&mut code, 3, 0.5f32.to_bits());
    code.extend([exp0(0, 0xB, true, false, false), exp1(0, 1, 2, 3), END]);
    let s = Stage { code: &code, stage: St::Pixel, user: native_user_data(None) };
    let c = GraphicsCase { col_format: 3, ..GraphicsCase::new("Export32Ar", code.clone(), vec![]) };
    let out = compile(&s, &|r| pixel_request(&c, r)).unwrap();
    validate("export_32_ar", &out.spirv);
    assert_eq!(stored_color(&out.spirv), vec![Some(0.25f32.to_bits()), Some(0), Some(0), Some(0.75f32.to_bits())]);
}

/// The length of the f32 array type of the ClipDistance output.
fn clip_distance_len(spirv: &[u32]) -> Option<u32> {
    let ops = instructions(spirv);
    let var = decorations(spirv, Decoration::BuiltIn).into_iter().zip(ops.iter().filter(|(o, w)| *o == Op::Decorate as u32 && w.get(1) == Some(&(Decoration::BuiltIn as u32)))).find(|(d, _)| d.first() == Some(&(BuiltIn::ClipDistance as u32))).map(|(_, (_, w))| w[0])?;
    let ptr_ty = ops.iter().find(|(o, w)| *o == Op::Variable as u32 && w.get(1) == Some(&var))?.1[0];
    let mut ty = ops.iter().find(|(o, w)| *o == Op::TypePointer as u32 && w.first() == Some(&ptr_ty))?.1[2];
    // Through the per-vertex array of a mesh output to the f32 array.
    loop {
        let (_, w) = ops.iter().find(|(o, w)| *o == Op::TypeArray as u32 && w.first() == Some(&ty))?;
        let (elem, len) = (w[1], w[2]);
        if ops.iter().any(|(o, w)| *o == Op::TypeArray as u32 && w.first() == Some(&elem)) {
            ty = elem;
            continue;
        }
        return resolve_u32(spirv, len);
    }
}

/// A vertex whose clip position is all zeros (either sign) clips its
/// primitive on the hardware; the host draws it at w = 0. The vertex and
/// mesh paths add a clip distance (-1 for such a vertex, 0 otherwise)
/// after the guest's own.
#[test]
fn all_zero_position_clips_its_primitive() {
    let code = misc_vertex_program();
    let s = Stage { code: &code, stage: St::Vertex, user: native_user_data(None) };
    // No guest distances: one distance.
    let out = compile(&s, &|r| r.position_export_control = 0).unwrap();
    validate("zero_position_clip", &out.spirv);
    assert_eq!(clip_distance_len(&out.spirv), Some(1));
    assert!(u32_constant_bits(&out.spirv).contains(&(-1.0f32).to_bits()));
    // Two guest clip distances (and two cull): the third one.
    let control = 1 << 22 | 0x3 | 0xC << 8;
    let out = compile(&s, &|r| r.position_export_control = control).unwrap();
    validate("zero_position_clip_guest", &out.spirv);
    assert_eq!(clip_distance_len(&out.spirv), Some(3));
    // Without device clip distances nothing is added.
    let out = compile(&s, &|r| {
        r.position_export_control = 0;
        r.supports_clip_distance = false;
    })
    .unwrap();
    assert!(!has_builtin(&out.spirv, BuiltIn::ClipDistance));
    let mesh = crate::pipeline::compiler::MeshConfig {
        lines: false,
        points: false,
        input_primitive: 4,
        primitives_per_group: 21,
        vertices_per_group: 63,
        max_vertices: 63,
        max_primitives: 21,
        threads: 64,
        esgs_item_size: 4,
        draw_dword: 64,
    };
    let out = compile_as(&s, &|r| r.position_export_control = 0, Some(mesh)).unwrap();
    validate("mesh_zero_position_clip", &out.spirv);
    assert_eq!(clip_distance_len(&out.spirv), Some(1));
}

/// A wave64 pixel program reducing per-pixel keys with DPP row scans, a
/// cross-row permute and v_readlane of lanes 31 and 63 (the reference's
/// PixelWaveReduction program): each readlane becomes a subgroup reduction
/// of the seed over its half, so lanes the fragment subgroup lacks (or a
/// 32-lane host never has) are not read.
#[test]
fn pixel_wave_reduction_reads_become_subgroup_reductions() {
    for (vector, scalar, identity, op) in [(0x2600_0000u32, 0x8380_0000u32, 0xffff_ffffu32, Op::GroupNonUniformUMin), (0x4a00_0000, 0x8000_0000, 0, Op::GroupNonUniformIAdd)] {
        let dpp = vector | 0x001a_1afa;
        let code = vec![
            0x7e040f00, 0x7e060f01, 0xd5430006, 0x0409ff03, 0x000000c0, 0x7e120506, 0xd569000b, 0x0001ff06, 0x9e3779b1, 0x2c161696, 0xbe98047e, 0x879a187e,
            0xbeea287e, 0xd501000d, if identity == 0 { 0x006a1680 } else { 0x006a16c1 }, dpp, 0xff01110d, dpp, 0xff01120d, dpp, 0xff01140d, dpp, 0xff01180d,
            0xd778100c, 0x0305830d, vector | 0x001a190d, 0xbefe046a, 0xd760000a, 0x00013f0d, 0xd760000b, 0x00017f0d, scalar | 0x000c0b0a, 0x8f099009,
            0x880c090c, 0x7e0a020c, 0xe0702000, 0x80010506, 0x7e0e02f2, 0xf800180f, 0x07070707, 0xbf810000,
        ];
        let s = Stage { code: &code, stage: St::Pixel, user: native_user_data(None) };
        let case = GraphicsCase { input_ena: (1 << 8) | (1 << 9), ..GraphicsCase::new("PixelWaveReduction", code.clone(), vec![]) };
        let out = compile(&s, &|r| {
            pixel_request(&case, r);
            r.wave_size = 64;
        })
        .unwrap();
        validate("pixel_wave_reduction", &out.spirv);
        let reductions = instructions(&out.spirv).iter().filter(|(o, _)| *o == op as u32).count();
        assert_eq!(reductions, 2, "{op:?}: one reduction per readlane");
    }
}

/// Whether the module reads `b` through an Input variable.
fn has_builtin_input(spirv: &[u32], b: BuiltIn) -> bool {
    let ids: Vec<u32> = decorations_with_target(spirv, Decoration::BuiltIn).into_iter().filter(|(_, v)| v.first() == Some(&(b as u32))).map(|(t, _)| t).collect();
    instructions(spirv).iter().any(|(o, w)| *o == Op::Variable as u32 && w.len() >= 3 && ids.contains(&w[1]) && w[2] == rspirv::spirv::StorageClass::Input as u32)
}

fn decorations_with_target(spirv: &[u32], d: Decoration) -> Vec<(u32, Vec<u32>)> {
    instructions(spirv).into_iter().filter(|(o, w)| *o == Op::Decorate as u32 && w.get(1) == Some(&(d as u32))).map(|(_, w)| (w[0], w[2..].to_vec())).collect()
}

/// The ancillary sample number is the host SampleId (which makes the host
/// run the program per sample) only when the guest iterates samples
/// (PS_ITER_SAMPLE); a program run once per pixel reads sample 0.
#[test]
fn ancillary_sample_number_follows_ps_iter_sample() {
    for rate in [false, true] {
        let c = GraphicsCase { sample_rate: rate, ..ancillary_sample_id() };
        let s = Stage { code: &c.code, stage: St::Pixel, user: native_user_data(c.user_data) };
        let out = compile(&s, &|r| pixel_request(&c, r)).unwrap_or_else(|e| panic!("{}: {e}", c.name));
        validate(c.name, &out.spirv);
        assert_eq!(has_builtin(&out.spirv, BuiltIn::SampleId), rate, "sample rate {rate}");
    }
}

/// SPI_PS_INPUT_ENA.SAMPLE_COVERAGE_ENA: the VGPR after the enabled inputs
/// holds the covered samples, read from the SampleMask input.
#[test]
fn sample_coverage_reads_the_sample_mask_input() {
    const SAMPLE_COVERAGE: u32 = 1 << 14;
    let mut code = vec![vop1(0x06, 0, vgpr(2))];
    v_mov_literal(&mut code, 1, 0);
    v_mov_literal(&mut code, 2, 0);
    v_mov_literal(&mut code, 3, 0x3f80_0000);
    code.extend([exp0(0, 0xF, true, false, false), exp1(0, 1, 2, 3), END]);
    let c = GraphicsCase { input_ena: PERSP_CENTER | SAMPLE_COVERAGE, ..GraphicsCase::new("GraphicsSampleCoverage", code, vec![0x3f80_0000, 0, 0, 0x3f80_0000]) };
    let s = Stage { code: &c.code, stage: St::Pixel, user: native_user_data(c.user_data) };
    let out = compile(&s, &|r| pixel_request(&c, r)).unwrap_or_else(|e| panic!("{}: {e}", c.name));
    validate(c.name, &out.spirv);
    assert!(has_builtin_input(&out.spirv, BuiltIn::SampleMask), "no SampleMask input");
}

/// The length of the array behind the `BuiltIn b` variable.
fn builtin_array_len(spirv: &[u32], b: BuiltIn) -> Option<u32> {
    let ins = instructions(spirv);
    let var = decorations_with_target(spirv, Decoration::BuiltIn).into_iter().find(|(_, v)| v.first() == Some(&(b as u32)))?.0;
    let ptr_ty = ins.iter().find(|(o, w)| *o == Op::Variable as u32 && w.get(1) == Some(&var))?.1[0];
    let pointee = ins.iter().find(|(o, w)| *o == Op::TypePointer as u32 && w[0] == ptr_ty)?.1[2];
    let len_id = ins.iter().find(|(o, w)| *o == Op::TypeArray as u32 && w[0] == pointee)?.1[2];
    ins.iter().find(|(o, w)| *o == Op::Constant as u32 && w.get(1) == Some(&len_id)).map(|(_, w)| w[2])
}

/// The all-zero position distance is added to gl_ClipDistance only while
/// the guest's clip and cull distances leave one of the eight free.
#[test]
fn zero_position_distance_fits_beside_clip_and_cull() {
    let code = misc_vertex_program();
    let s = Stage { code: &code, stage: St::Vertex, user: native_user_data(None) };
    // Both distance vectors: 2 clip + 2 cull leaves room, 4 + 4 does not.
    for (clip, cull, len) in [(0x3u32, 0xCu32, 3u32), (0xF, 0xF0, 4)] {
        let control = 1 << 21 | 1 << 22 | 1 << 23 | clip | cull << 8;
        let out = compile(&s, &|r| r.position_export_control = control).unwrap();
        validate("zero_position_distance_fits", &out.spirv);
        if crate::shader::io::zero_position_clip_enabled() {
            assert_eq!(builtin_array_len(&out.spirv, BuiltIn::ClipDistance), Some(len), "clip {clip:#x} cull {cull:#x}");
            // The rect-list stages forward that many distances.
            assert_eq!(out.reflection.clip_distances, len);
        }
    }
}

/// PA_CL_CLIP_CNTL.UCP_ENA: each enabled user clip plane is a clip
/// distance dot(plane, position) before the all-zero position distance.
#[test]
fn user_clip_planes_write_clip_distances() {
    let mut code = Vec::new();
    for (r, v) in [0x3f00_0000u32, 0, 0, 0x3f80_0000].into_iter().enumerate() {
        v_mov_literal(&mut code, r as u32, v);
    }
    code.extend([exp0(12, 0xF, true, false, false), exp1(0, 1, 2, 3), END]);
    let s = Stage { code: &code, stage: St::Vertex, user: native_user_data(None) };
    let none = compile(&s, &|_| {}).unwrap();
    assert!(!has_op(&none.spirv, Op::Dot));
    let mut planes = [0u32; 24];
    planes[0..4].copy_from_slice(&[0x3f80_0000, 0, 0, 0x3e80_0000]); // x + 0.25 >= 0
    planes[8..12].copy_from_slice(&[0, 0xbf80_0000, 0, 0x3e00_0000]); // 0.125 - y >= 0
    let ucp = ps5shader::resources::ShaderUserClipPlanes { mask: 0b101, planes };
    let out = compile(&s, &|r| r.user_clip_planes = ucp).unwrap();
    validate("user_clip_planes", &out.spirv);
    assert!(has_op(&out.spirv, Op::Dot));
    let constants = u32_constant_bits(&out.spirv);
    assert!(constants.contains(&0x3e80_0000) && constants.contains(&0xbf80_0000) && constants.contains(&0x3e00_0000), "plane coefficients");
    let len = 2 + crate::shader::io::zero_position_clip_enabled() as u32;
    assert_eq!(builtin_array_len(&out.spirv, BuiltIn::ClipDistance), Some(len));
    assert_eq!(out.reflection.clip_distances, len, "the rect-list stages forward them");
}

/// A pixel program's per-invocation LDS covers the EXTRA_LDS_SIZE the
/// request carries (an address the program computes at run time would
/// otherwise take the whole 64 KiB per invocation).
#[test]
fn pixel_lds_follows_extra_lds_size() {
    // ds_write_b32 [v0], v0; ds_read_b32 v1, [v0]; export v1.
    let code = vec![ds0(0x0d, 0), ds1(0, 0, 0), ds0(0x36, 0), ds1(1, 0, 0), exp0(0, 0xF, true, false, false), exp1(1, 1, 1, 1), END];
    let c = GraphicsCase::new("GraphicsPixelLds", code, vec![0; 4]);
    let s = Stage { code: &c.code, stage: St::Pixel, user: native_user_data(c.user_data) };
    assert_eq!(crate::pipeline::provider::pixel_lds_dwords(0x0200), 256);
    assert_eq!(crate::pipeline::provider::pixel_lds_dwords(0), 1);
    let sized = compile(&s, &|r| {
        pixel_request(&c, r);
        r.local_data_share_dwords = crate::pipeline::provider::pixel_lds_dwords(0x0200);
    })
    .unwrap();
    validate("pixel_lds_sized", &sized.spirv);
    assert_eq!(sized.reflection.private_lds_bytes, 1024);
    let unsized_ = compile(&s, &|r| pixel_request(&c, r)).unwrap();
    assert!(unsized_.reflection.private_lds_bytes > 1024, "{}", unsized_.reflection.private_lds_bytes);
}
