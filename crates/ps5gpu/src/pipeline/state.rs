// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Fixed-function state of a draw: the static part hashed into the
//! graphics pipeline key and the dynamic part recorded per draw.
//!
//! Static: topology class, primitive restart, vertex input layout,
//! polygon mode, depth clip, attachment formats and sample count, and per
//! target blending and write masks. Dynamic (Vulkan 1.3 core extended
//! dynamic state): viewport, scissor, cull mode, front face, depth
//! test/write/compare/bounds, stencil test/ops/masks/reference, depth bias,
//! blend constants, line width.

use crate::render::{primitive, ColorTarget, ResolvedDraw};
use ash::vk;

pub const MAX_COLOR_ATTACHMENTS: usize = 8;

/// How a RECT_LIST draw is expanded on the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RectMode {
    /// Not a rectangle list.
    None,
    /// 3-vertex patches expanded to quads by generated tessellation stages.
    Tessellation,
    /// Triangle list rasterised as bounding rectangles (VK_NV_fill_rectangle).
    FillRectangle,
    /// Compatibility: one rectangle drawn as a 4-vertex strip, several as a
    /// plain triangle list.
    Strip,
    List,
}

/// One vertex buffer binding of the fixed-function vertex input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexBindingKey {
    pub stride: u32,
    pub per_instance: bool,
}

/// One vertex attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexAttributeKey {
    pub location: u32,
    pub binding: u32,
    pub format: i32,
    pub offset: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct VertexInputKey {
    pub bindings: Vec<VertexBindingKey>,
    pub attributes: Vec<VertexAttributeKey>,
}

/// Blend state of one color attachment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BlendKey {
    pub enable: bool,
    pub src_color: i32,
    pub dst_color: i32,
    pub color_op: i32,
    pub src_alpha: i32,
    pub dst_alpha: i32,
    pub alpha_op: i32,
    pub write_mask: u32,
}

/// Everything that selects a graphics pipeline besides the shader modules.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FixedFunctionKey {
    pub topology: i32,
    pub rect: RectMode,
    pub primitive_restart: bool,
    pub vertex_input: VertexInputKey,
    pub polygon_mode: i32,
    pub depth_clip: bool,
    pub depth_clamp: bool,
    pub negative_one_to_one: bool,
    pub color_formats: [i32; MAX_COLOR_ATTACHMENTS],
    pub color_count: u32,
    pub blend: [BlendKey; MAX_COLOR_ATTACHMENTS],
    pub depth_format: i32,
    pub stencil_format: i32,
    pub samples: u32,
    /// Every sample is rasterized at the pixel center (static sample
    /// locations, VK_EXT_sample_locations): see [`sample_placement`].
    pub centered_samples: bool,
    /// PS_ITER_SAMPLE on a multisampled target: one invocation per sample.
    pub sample_shading: bool,
    /// The static sample mask (PA_SC_AA_MASK when the four pixels of a
    /// quad keep the same samples); all ones otherwise.
    pub sample_mask: u32,
    pub logic_op: Option<i32>,
    /// 1, or [`MAX_VIEWPORTS`] when the vertex stage selects the viewport.
    pub viewport_count: u32,
    /// PA_SU_SC_MODE_CNTL.PROVOKING_VTX_LAST (on a device with
    /// provokingVertexLast): flat attributes come from the last vertex.
    pub provoking_last: bool,
}

/// Viewports a draw may select (gl_ViewportIndex).
pub const MAX_VIEWPORTS: usize = 16;

/// Every viewport and scissor of a draw whose vertex stage selects one.
#[derive(Clone, Copy, Debug)]
pub struct IndexedViewports {
    pub viewports: [vk::Viewport; MAX_VIEWPORTS],
    pub scissors: [vk::Rect2D; MAX_VIEWPORTS],
}

/// Per-draw dynamic state.
#[derive(Clone, Copy, Debug)]
pub struct DynamicState {
    pub viewport: vk::Viewport,
    pub scissor: vk::Rect2D,
    /// All 16 viewports (viewport 0 above) when the draw selects them.
    pub indexed: Option<IndexedViewports>,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_compare: vk::CompareOp,
    pub depth_bounds_test: bool,
    pub depth_bounds: [f32; 2],
    pub stencil_test: bool,
    pub stencil_front: StencilFace,
    pub stencil_back: StencilFace,
    pub depth_bias: Option<[f32; 3]>,
    /// Front and back faces need different depth biases: the draw is
    /// recorded for front faces only (`cull_mode` BACK, `depth_bias`), then
    /// for back faces only with this bias ([`DynamicState::back_faces`]).
    pub back_face_bias: Option<[f32; 3]>,
    pub blend_constants: [f32; 4],
}

impl DynamicState {
    /// The second recording of a draw whose faces need different depth
    /// biases: back faces only, with their bias.
    pub fn back_faces(&self) -> Option<DynamicState> {
        let bias = self.back_face_bias?;
        Some(DynamicState { cull_mode: vk::CullModeFlags::FRONT, depth_bias: Some(bias), back_face_bias: None, ..*self })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilFace {
    pub fail: vk::StencilOp,
    pub pass: vk::StencilOp,
    pub depth_fail: vk::StencilOp,
    pub compare: vk::CompareOp,
    pub compare_mask: u32,
    pub write_mask: u32,
    pub reference: u32,
}

impl Default for StencilFace {
    fn default() -> Self {
        StencilFace {
            fail: vk::StencilOp::KEEP,
            pass: vk::StencilOp::KEEP,
            depth_fail: vk::StencilOp::KEEP,
            compare: vk::CompareOp::ALWAYS,
            compare_mask: 0xFF,
            write_mask: 0,
            reference: 0,
        }
    }
}

/// CB_BLEND*_CONTROL blend factor → host factor.
pub fn blend_factor(f: u32) -> vk::BlendFactor {
    use vk::BlendFactor as B;
    match f {
        0 => B::ZERO,
        1 => B::ONE,
        2 => B::SRC_COLOR,
        3 => B::ONE_MINUS_SRC_COLOR,
        4 => B::SRC_ALPHA,
        5 => B::ONE_MINUS_SRC_ALPHA,
        6 => B::DST_ALPHA,
        7 => B::ONE_MINUS_DST_ALPHA,
        8 => B::DST_COLOR,
        9 => B::ONE_MINUS_DST_COLOR,
        10 => B::SRC_ALPHA_SATURATE,
        13 => B::CONSTANT_COLOR,
        14 => B::ONE_MINUS_CONSTANT_COLOR,
        15 => B::SRC1_COLOR,
        16 => B::ONE_MINUS_SRC1_COLOR,
        17 => B::SRC1_ALPHA,
        18 => B::ONE_MINUS_SRC1_ALPHA,
        19 => B::CONSTANT_ALPHA,
        20 => B::ONE_MINUS_CONSTANT_ALPHA,
        _ => B::ONE,
    }
}

/// Blend combine function → host op.
pub fn blend_op(f: u32) -> vk::BlendOp {
    match f {
        1 => vk::BlendOp::SUBTRACT,
        2 => vk::BlendOp::MIN,
        3 => vk::BlendOp::MAX,
        4 => vk::BlendOp::REVERSE_SUBTRACT,
        _ => vk::BlendOp::ADD,
    }
}

/// `RPS5E_FORCE_DST_ALPHA_1=0` ignores CB_COLOR_ATTRIB.FORCE_DST_ALPHA_1.
fn force_dst_alpha_one_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_FORCE_DST_ALPHA_1").is_ok_and(|v| v == "0"))
}

fn uses_dual_source(f: vk::BlendFactor) -> bool {
    matches!(
        f,
        vk::BlendFactor::SRC1_COLOR | vk::BlendFactor::ONE_MINUS_SRC1_COLOR | vk::BlendFactor::SRC1_ALPHA | vk::BlendFactor::ONE_MINUS_SRC1_ALPHA
    )
}

/// How blending of a target whose export mapping moves components can be
/// expressed on the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMapping {
    /// The physical alpha holds the logical alpha: blend as programmed.
    Direct,
    /// Needs the logical alpha as a second blend source.
    SourceAlpha,
    /// Source alpha for the color, one / zero for alpha, with one shared
    /// destination equation.
    SourceAlphaOne,
    SourceAlphaZero,
    Unsupported,
}

/// Classify the blend state `b` of a target with export mapping `m`.
pub fn classify_blend_mapping(b: &crate::render::BlendState, m: crate::render::formats::ComponentMap) -> BlendMapping {
    let dual = |f: u32| (15..=18).contains(&f);
    let constant_color = |f: u32| f == 13 || f == 14;
    // Constant colors are not swizzled with the exports.
    if m != crate::render::formats::ComponentMap::IDENTITY && (constant_color(b.color_src) || constant_color(b.color_dst)) {
        return BlendMapping::Unsupported;
    }
    if m.map(3) == 3 {
        return BlendMapping::Direct;
    }
    if b.separate_alpha && b.color_src == 4 && b.color_dst == 5 && b.alpha_dst == b.color_dst && b.color_op == 0 && b.alpha_op == b.color_op {
        match b.alpha_src {
            0 => return BlendMapping::SourceAlphaZero,
            1 => return BlendMapping::SourceAlphaOne,
            _ => {}
        }
    }
    // Moving alpha requires one equation for every channel.
    if b.separate_alpha && (b.alpha_src != b.color_src || b.alpha_dst != b.color_dst || b.alpha_op != b.color_op) {
        return BlendMapping::Unsupported;
    }
    let mut support = BlendMapping::Direct;
    for f in [b.color_src, b.color_dst] {
        if dual(f) {
            return BlendMapping::Unsupported;
        }
        match f {
            4 | 5 => support = BlendMapping::SourceAlpha,
            6 | 7 | 10 => return BlendMapping::Unsupported,
            _ => {}
        }
    }
    support
}

/// How the pixel shader feeds the second blend source of target 0: the
/// guest's own dual-source blending (SRC1 factors read MRT1), or the
/// logical source alpha when target 0's export mapping moves alpha to
/// another channel (only when the program exports MRT0 alone, as a float).
pub fn pixel_blend_sources(colors: &[&ColorTarget], col_format: u32, dual_supported: bool) -> (ps5shader::resources::PixelAlphaBlendSource, bool) {
    use ps5shader::resources::PixelAlphaBlendSource as A;
    let Some(t) = colors.iter().find(|t| t.slot == 0) else { return (A::None, false) };
    let b = t.blend;
    let integer = matches!(t.number_type, 4 | 5);
    let bypass = t.info & (1 << 16) != 0;
    if !dual_supported || !b.enable || integer || bypass {
        return (A::None, false);
    }
    let dual = |f: u32| (15..=18).contains(&f);
    if dual(b.color_src) || dual(b.color_dst) || (b.separate_alpha && (dual(b.alpha_src) || dual(b.alpha_dst))) {
        return (A::None, true);
    }
    let mrt0 = col_format & 0xF;
    if mrt0 == 0 || mrt0 == 7 || col_format >> 4 != 0 {
        return (A::None, false);
    }
    let source = match classify_blend_mapping(&b, t.format.export_mapping) {
        BlendMapping::SourceAlpha => A::SourceAlpha,
        BlendMapping::SourceAlphaOne => A::SourceAlphaOne,
        BlendMapping::SourceAlphaZero => A::SourceAlphaZero,
        _ => A::None,
    };
    (source, false)
}

/// Blend key of a color target (`write_mask` is the physical mask);
/// `alpha_source`: target 0's logical alpha arrives as the second source.
pub fn blend_key(t: &ColorTarget, dual_source_supported: bool, alpha_source: ps5shader::resources::PixelAlphaBlendSource) -> BlendKey {
    use ps5shader::resources::PixelAlphaBlendSource as A;
    let mut b = t.blend;
    // Guest factors: 4/5 (ONE_MINUS_)SRC_ALPHA, 15 SRC1_COLOR, 17/18
    // (ONE_MINUS_)SRC1_ALPHA.
    match alpha_source {
        A::SourceAlpha => {
            let remap = |f: u32| match f {
                4 => 17,
                5 => 18,
                f => f,
            };
            b.color_src = remap(b.color_src);
            b.color_dst = remap(b.color_dst);
            b.separate_alpha = false;
        }
        A::SourceAlphaOne | A::SourceAlphaZero => {
            // The second source carries the mapped source factor; its
            // alpha stays the logical source alpha.
            b.color_src = 15;
            b.color_dst = 18;
            b.separate_alpha = false;
        }
        A::None => {}
    }
    let write_mask = crate::render::physical_write_mask(t);
    // Integer attachments never blend; CB_COLOR_INFO.BLEND_BYPASS (bit 16)
    // writes the source as it is.
    let integer = matches!(t.number_type, 4 | 5);
    let bypass = t.info & (1 << 16) != 0;
    // Packed 11_11_10 UNORM is stored through an unsigned host format; a
    // 4_4_4_4 UNORM target is too (the pixel program packs its word).
    let packed = t.format.guest == crate::render::formats::GuestFormat::R11G11B10Unorm
        || t.format.guest == crate::render::formats::GuestFormat::Rgba4Unorm;
    if packed && b.enable && !bypass {
        crate::log_once("ps5gpu: blending into a packed 11_11_10 / 4_4_4_4 UNORM target is not modelled (written unblended)");
    }
    if !b.enable || integer || bypass || packed {
        return BlendKey { write_mask, ..BlendKey::default() };
    }
    // Blending that the export mapping makes inexpressible is disabled
    // (the source-alpha forms read the logical alpha from the second
    // source, see `pixel_blend_sources`).
    if alpha_source == A::None && classify_blend_mapping(&b, t.format.export_mapping) == BlendMapping::Unsupported {
        static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
            log::warn!("ps5gpu: blending disabled for unsupported color mapping {:#04x} (blend {:#010x})", t.format.export_mapping.0, b.raw);
        }
        return BlendKey { write_mask, ..BlendKey::default() };
    }
    let (sa, da, ao) = if b.separate_alpha { (b.alpha_src, b.alpha_dst, b.alpha_op) } else { (b.color_src, b.color_dst, b.color_op) };
    let mut k = BlendKey {
        enable: true,
        src_color: blend_factor(b.color_src).as_raw(),
        dst_color: blend_factor(b.color_dst).as_raw(),
        color_op: blend_op(b.color_op).as_raw(),
        src_alpha: blend_factor(sa).as_raw(),
        dst_alpha: blend_factor(da).as_raw(),
        alpha_op: blend_op(ao).as_raw(),
        write_mask,
    };
    // CB_COLOR_ATTRIB.FORCE_DST_ALPHA_1 (bit 17): the destination alpha
    // reads as one (a target without alpha, or one whose alpha is ignored).
    if t.attrib & (1 << 17) != 0 && force_dst_alpha_one_enabled() {
        use vk::BlendFactor as F;
        let color = |f: &mut i32| {
            *f = match F::from_raw(*f) {
                F::DST_ALPHA => F::ONE,
                F::ONE_MINUS_DST_ALPHA | F::SRC_ALPHA_SATURATE => F::ZERO,
                other => other,
            }
            .as_raw()
        };
        let alpha = |f: &mut i32| {
            *f = match F::from_raw(*f) {
                F::DST_ALPHA | F::DST_COLOR => F::ONE,
                F::ONE_MINUS_DST_ALPHA | F::ONE_MINUS_DST_COLOR => F::ZERO,
                other => other,
            }
            .as_raw()
        };
        color(&mut k.src_color);
        color(&mut k.dst_color);
        alpha(&mut k.src_alpha);
        alpha(&mut k.dst_alpha);
    }
    if !dual_source_supported {
        let fix = |f: &mut i32| {
            if uses_dual_source(vk::BlendFactor::from_raw(*f)) {
                *f = vk::BlendFactor::ONE.as_raw();
            }
        };
        fix(&mut k.src_color);
        fix(&mut k.dst_color);
        fix(&mut k.src_alpha);
        fix(&mut k.dst_alpha);
    }
    k
}

impl BlendKey {
    pub fn attachment(&self) -> vk::PipelineColorBlendAttachmentState {
        vk::PipelineColorBlendAttachmentState {
            blend_enable: self.enable as u32,
            src_color_blend_factor: vk::BlendFactor::from_raw(self.src_color),
            dst_color_blend_factor: vk::BlendFactor::from_raw(self.dst_color),
            color_blend_op: vk::BlendOp::from_raw(self.color_op),
            src_alpha_blend_factor: vk::BlendFactor::from_raw(self.src_alpha),
            dst_alpha_blend_factor: vk::BlendFactor::from_raw(self.dst_alpha),
            alpha_blend_op: vk::BlendOp::from_raw(self.alpha_op),
            color_write_mask: vk::ColorComponentFlags::from_raw(self.write_mask & 0xF),
        }
    }
}

/// DB stencil operation (4 bits) → host op, given the face write mask and
/// the operation value. Operations that leave the masked bits unchanged
/// keep the value; an XOR flipping every writable bit is an inversion.
pub fn stencil_op(op: u32, write_mask: u32, op_value: u32) -> vk::StencilOp {
    use vk::StencilOp as S;
    if write_mask == 0 {
        return S::KEEP;
    }
    match op & 0xF {
        0 => S::KEEP,
        1 => S::ZERO,
        // ONES replaces with all ones, REPLACE_TEST with the test value,
        // REPLACE_OP with the operation value (the reference of the face
        // is chosen by `stencil_face`).
        2 | 3 => S::REPLACE,
        4 => {
            if op_value & write_mask == 0 {
                S::ZERO
            } else {
                S::REPLACE
            }
        }
        // ADD / SUB (clamp or wrap) add or subtract the operation value:
        // 0 keeps the value, 1 is the host's increment / decrement.
        5 | 6 | 8 | 9 if op_value == 0 && stencil_add_opval_enabled() => S::KEEP,
        5 | 6 | 8 | 9 => {
            if op_value != 1 && stencil_add_opval_enabled() {
                unsupported_stencil("add/subtract of an operation value other than 0 or 1 (drawn as 1)");
            }
            match op & 0xF {
                5 => S::INCREMENT_AND_CLAMP,
                6 => S::DECREMENT_AND_CLAMP,
                8 => S::INCREMENT_AND_WRAP,
                _ => S::DECREMENT_AND_WRAP,
            }
        }
        7 => S::INVERT,
        12 => {
            if write_mask & op_value == 0 {
                S::KEEP
            } else {
                if write_mask & !op_value != 0 && !stencil_logic_enabled() {
                    unsupported_stencil("XOR of part of the writable bits");
                }
                S::INVERT
            }
        }
        // AND / OR / XNOR: drawn on the bits they change by `stencil_face`.
        10 | 11 | 15 if stencil_logic_enabled() => S::KEEP,
        // NAND / NOR have no host operation.
        _ => {
            unsupported_stencil("logic operation");
            S::KEEP
        }
    }
}

/// `RPS5E_STENCIL_ADD_OPVAL=0`: the stencil ADD / SUB operations always
/// step by one, whatever DB_STENCILREFMASK.STENCILOPVAL holds.
fn stencil_add_opval_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_STENCIL_ADD_OPVAL").is_ok_and(|v| v == "0"))
}

/// `RPS5E_STENCIL_LOGIC=0`: the stencil logic operations AND, OR and XNOR
/// keep the value and XOR draws only as the inversion of every writable
/// bit, instead of drawing each on the bits it changes.
fn stencil_logic_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_STENCIL_LOGIC").is_ok_and(|v| v == "0"))
}

fn unsupported_stencil(what: &str) {
    static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
        log::warn!("ps5gpu: unsupported stencil state: {what}");
    }
}

/// One host stencil face from the guest compare function, the fail / pass
/// / depth-fail operations, compare mask, write mask, test value and
/// operation value. The host shares one reference between the comparison
/// and every replacement of a face: replacements write their value into
/// the written bits of the reference, which must not change the bits the
/// comparison reads.
///
/// A replacement whose value differs from the reference on compared bits
/// can still be drawn where the compare fixes the stencil value: EQUAL on
/// pass and depth fail, NOTEQUAL on fail, and LEQUAL / GEQUAL / LESS /
/// GREATER against the largest or smallest masked value. When every
/// written bit is compared, the written bits there hold the test value,
/// so the replacement is an INVERT of the bits that differ from the value
/// written (the face's write mask becomes those bits). The operations of
/// a face share its write mask: every active operation must be such an
/// inversion of the same bits (`RPS5E_STENCIL_INVERT=0`: not drawn this
/// way, the reference is kept).
pub fn stencil_face(compare: u32, ops: [u32; 3], compare_mask: u32, write_mask: u32, test_value: u32, op_value: u32) -> StencilFace {
    static INVERT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let invert = *INVERT.get_or_init(|| !std::env::var("RPS5E_STENCIL_INVERT").is_ok_and(|v| v == "0"));
    stencil_face_of(compare, ops, compare_mask, write_mask, test_value, op_value, invert)
}

/// The stencil bits a logic operation (AND 10, OR 11, XOR 12, XNOR 15)
/// changes under `write_mask`: AND clears the bits clear in the operation
/// value, OR sets the bits set in it, XOR inverts the bits set in it and
/// XNOR the bits clear in it. None for other operations (NAND and NOR
/// have no host form).
fn stencil_logic_bits(op: u32, write_mask: u32, op_value: u32) -> Option<u32> {
    match op & 0xF {
        11 | 12 => Some(op_value & write_mask & 0xFF),
        10 | 15 => Some(!op_value & write_mask & 0xFF),
        _ => None,
    }
}

fn stencil_face_of(compare: u32, ops: [u32; 3], compare_mask: u32, write_mask: u32, test_value: u32, op_value: u32, invert: bool) -> StencilFace {
    let mut host = ops.map(|o| stencil_op(o, write_mask, op_value));
    let mut reference = test_value;
    let mut required = if matches!(compare & 7, 0 | 7) { 0 } else { compare_mask };
    // Logic operations draw as ZERO (AND), REPLACE with those bits set in
    // the reference (OR) or INVERT (XOR, XNOR) of the bits they change,
    // under a write mask narrowed to those bits: the logic operations of a
    // face must change the same bits, and the face's other operations
    // must write all of its bits (they share the write mask).
    let logic: Vec<(usize, u32)> = if write_mask != 0 && stencil_logic_enabled() {
        ops.iter().enumerate().filter_map(|(i, &op)| stencil_logic_bits(op, write_mask, op_value).map(|b| (i, b))).collect()
    } else {
        Vec::new()
    };
    let logic_mask = logic.iter().map(|&(_, b)| b).find(|&b| b != 0).unwrap_or(0);
    let others_write = ops.iter().any(|&o| o & 0xF != 0 && stencil_logic_bits(o, write_mask, op_value).is_none());
    let mut narrowed = None;
    if logic_mask != 0 {
        let same_bits = logic.iter().all(|&(_, b)| b == 0 || b == logic_mask);
        let or_conflicts = logic.iter().any(|&(i, b)| ops[i] & 0xF == 11 && b != 0 && !reference & required & b != 0);
        if !same_bits || (others_write && logic_mask != write_mask) || or_conflicts {
            unsupported_stencil("logic operations on different bits, or an OR of compared bits");
        } else {
            for &(i, bits) in &logic {
                host[i] = match ops[i] & 0xF {
                    _ if bits == 0 => vk::StencilOp::KEEP,
                    10 => vk::StencilOp::ZERO,
                    11 => {
                        reference |= bits;
                        required |= bits;
                        vk::StencilOp::REPLACE
                    }
                    _ => vk::StencilOp::INVERT,
                };
            }
            narrowed = Some(logic_mask);
        }
    }
    let (mut invert_mask, mut inverted) = (0u32, Vec::new());
    for (i, &op) in ops.iter().enumerate() {
        if host[i] != vk::StencilOp::REPLACE || (narrowed.is_some() && op & 0xF == 11) {
            continue;
        }
        let replacement = match op & 0xF {
            2 => 0xFF,
            4 => op_value,
            _ => test_value,
        };
        if (reference ^ replacement) & required & write_mask != 0 {
            let compared = test_value & compare_mask;
            let c = compare & 7;
            let pass_known = c == 2 || (c == 3 && compared == compare_mask) || (c == 6 && compared == 0);
            let fail_known = c == 5 || (c == 1 && compared == 0) || (c == 4 && compared == compare_mask);
            let known = if i == 0 { fail_known } else { pass_known };
            let flipped = (test_value ^ replacement) & write_mask;
            if invert && known && write_mask & !compare_mask == 0 && (invert_mask == 0 || invert_mask == flipped) {
                invert_mask = flipped;
                inverted.push(i);
                continue;
            }
            unsupported_stencil("replacement value differs from the compared reference");
            continue;
        }
        reference = (reference & !write_mask) | (replacement & write_mask);
        required |= write_mask;
    }
    let mut write_mask = narrowed.unwrap_or(write_mask);
    if invert_mask != 0 {
        let active = ops.iter().filter(|&&o| o & 0xF != 0).count();
        if inverted.len() == active {
            for &i in &inverted {
                host[i] = vk::StencilOp::INVERT;
            }
            write_mask = invert_mask;
        } else {
            unsupported_stencil("a replacement drawn as an inversion beside another operation");
        }
    }
    StencilFace { fail: host[0], pass: host[1], depth_fail: host[2], compare: compare_op(compare), compare_mask, write_mask, reference: reference & 0xFF }
}

/// Guest compare function (DB / SQ) → host compare op (same numbering).
pub fn compare_op(f: u32) -> vk::CompareOp {
    vk::CompareOp::from_raw((f & 7) as i32)
}

/// Diagnostic only, never a fix: `RPS5E_DEBUG_DEPTH_EQUAL=ge` (or `le`)
/// draws depth tests EQUAL as GREATER_OR_EQUAL (LESS_OR_EQUAL). A pass
/// that tests EQUAL against a depth pre-pass shows only where both passes
/// computed the same depth; when the surfaces it loses come back with the
/// switch, the two passes disagree about their positions (or the pre-pass
/// depth did not reach the second pass).
fn debug_equal_depth(op: vk::CompareOp) -> vk::CompareOp {
    static V: std::sync::OnceLock<Option<vk::CompareOp>> = std::sync::OnceLock::new();
    let replacement = *V.get_or_init(|| parse_debug_equal_depth(std::env::var("RPS5E_DEBUG_DEPTH_EQUAL").ok().as_deref()));
    match replacement {
        Some(r) if op == vk::CompareOp::EQUAL => r,
        _ => op,
    }
}

fn parse_debug_equal_depth(v: Option<&str>) -> Option<vk::CompareOp> {
    match v.map(str::trim) {
        Some("ge") => Some(vk::CompareOp::GREATER_OR_EQUAL),
        Some("le") => Some(vk::CompareOp::LESS_OR_EQUAL),
        _ => None,
    }
}

/// Polygon mode of PA_SU_SC_MODE_CNTL (POLY_MODE 1 with a point or line
/// type). The type is the one of the faces that are drawn: the front type
/// (bits 5..7) unless CULL_FRONT, else the back type (bits 8..10); with
/// both faces culled nothing is rasterized and the mode does not matter.
/// When both faces are drawn with different types the front type is
/// taken (one pipeline has one mode). A point mode needs
/// VK_KHR_maintenance5 for the one-pixel size of a point whose shader
/// writes no PointSize (PA_SU_POINT_SIZE after a context clear); without
/// it, or without fillModeNonSolid, triangles stay filled.
/// `RPS5E_POLYGON_MODE_FACES=0`: always the front type (earlier behaviour).
pub fn polygon_mode(mode_cntl: u32, non_solid_supported: bool, maintenance5: bool) -> vk::PolygonMode {
    static FACES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let faces = *FACES.get_or_init(|| !std::env::var("RPS5E_POLYGON_MODE_FACES").is_ok_and(|v| v == "0"));
    polygon_mode_of(mode_cntl, non_solid_supported, maintenance5 || !faces, faces)
}

fn polygon_mode_of(mode_cntl: u32, non_solid_supported: bool, maintenance5: bool, faces: bool) -> vk::PolygonMode {
    if (mode_cntl >> 3) & 3 != 1 || !non_solid_supported {
        return vk::PolygonMode::FILL;
    }
    let (front, back) = ((mode_cntl >> 5) & 7, (mode_cntl >> 8) & 7);
    let front_culled = mode_cntl & 1 != 0;
    let back_culled = mode_cntl & 2 != 0;
    let ty = if !faces || !front_culled {
        front
    } else if !back_culled {
        back
    } else {
        return vk::PolygonMode::FILL;
    };
    // 0 points, 1 lines, 2 triangles.
    match ty {
        0 if maintenance5 => vk::PolygonMode::POINT,
        1 => vk::PolygonMode::LINE,
        _ => vk::PolygonMode::FILL,
    }
}

/// Topology of a draw on the host after CPU fallbacks: quads become
/// triangle lists (indices generated), rectangle lists use `rect`.
pub fn host_topology(prim: u32, rect: RectMode) -> Option<vk::PrimitiveTopology> {
    use vk::PrimitiveTopology as T;
    Some(match prim {
        primitive::QUAD_LIST_LEGACY => T::TRIANGLE_LIST,
        primitive::RECT_LIST => match rect {
            RectMode::Tessellation => T::PATCH_LIST,
            RectMode::FillRectangle | RectMode::List => T::TRIANGLE_LIST,
            RectMode::Strip => T::TRIANGLE_STRIP,
            RectMode::None => return None,
        },
        _ => crate::render::topology(prim)?,
    })
}

/// Where a draw's rasterizer puts the samples of its target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplePlacement {
    /// The standard locations of the sample count (one sample: the center).
    Standard,
    /// Every sample at the pixel center.
    Centered,
    /// Every sample belongs at the pixel center, but the draw is drawn at
    /// the standard locations (the reason says why).
    CenteredUnsupported(&'static str),
}

/// The sample placement of a draw with `samples` samples.
///
/// PA_SC_MODE_CNTL_0.MSAA_ENABLE is what makes the rasterizer use the
/// programmed sample locations (GFX10 and later). A multisampled target
/// scanned with the bit clear keeps its sample count with every sample at
/// the pixel center, the full-coverage scan GL asks for with multisampling
/// disabled; that the hardware does so with MSAA_NUM_SAMPLES > 0 is
/// inferred, not measured. Filled primitives without a depth target or
/// alpha-to-coverage only: points, lines, a depth target (whose image
/// would need sample-location-compatible depth) and alpha-to-coverage are
/// left at the standard locations.
pub fn sample_placement(samples: u32, msaa_enable: bool, topology: vk::PrimitiveTopology, depth: bool, alpha_to_coverage: bool) -> SamplePlacement {
    if samples <= 1 || msaa_enable {
        return SamplePlacement::Standard;
    }
    use vk::PrimitiveTopology as T;
    if matches!(topology, T::POINT_LIST | T::LINE_LIST | T::LINE_STRIP) {
        return SamplePlacement::CenteredUnsupported("points or lines");
    }
    if depth {
        return SamplePlacement::CenteredUnsupported("a depth target");
    }
    if alpha_to_coverage {
        return SamplePlacement::CenteredUnsupported("alpha-to-coverage");
    }
    SamplePlacement::Centered
}

/// The locations of `samples` centered samples: one pixel's worth (a 1x1
/// grid: every pixel of the 2x2 quad alike), each at (0.5, 0.5).
pub fn centered_sample_locations(samples: u32) -> (vk::Extent2D, Vec<vk::SampleLocationEXT>) {
    (vk::Extent2D { width: 1, height: 1 }, vec![vk::SampleLocationEXT { x: 0.5, y: 0.5 }; samples as usize])
}

/// DB_SHADER_CONTROL.STENCIL_OP_VAL_EXPORT_ENABLE that the host can take:
/// the pixel program supplies the value REPLACE_OP writes (MRTZ.y bits
/// 8..15), drawn as REPLACE with that value as the stencil reference. Taken
/// only without a test value export, with a 32_GR / 32_ABGR Z format, while
/// a REPLACE_OP is reachable through a non-zero write mask and no face's
/// compare reads the reference (NEVER / ALWAYS) nor another operation uses
/// a value (ONES, REPLACE_TEST, the add and subtract operations).
pub fn stencil_op_export(d: &ResolvedDraw) -> bool {
    let ds = &d.depth_state;
    if d.db_shader_control & 4 == 0 || d.db_shader_control & 2 != 0 || !matches!(d.shader_z_format, 2 | 9) || !ds.stencil_test || ds.stencil_clear_enable || ds.stencil_write_disable {
        return false;
    }
    let ctl = ds.stencil_control;
    let mut faces = vec![(ds.stencil_compare, ds.stencil_ref_mask, [ctl & 0xF, (ctl >> 4) & 0xF, (ctl >> 8) & 0xF])];
    if ds.backface {
        faces.push((ds.stencil_compare_back, ds.stencil_ref_mask_back, [(ctl >> 12) & 0xF, (ctl >> 16) & 0xF, (ctl >> 20) & 0xF]));
    }
    let mut reachable = false;
    for (compare, refmask, ops) in faces {
        let write_mask = (refmask >> 16) & 0xFF;
        if write_mask == 0 {
            continue;
        }
        if !matches!(compare & 7, 0 | 7) || ops.iter().any(|&o| matches!(o, 2 | 3 | 5 | 6 | 8 | 9)) {
            return false;
        }
        reachable |= ops.contains(&4);
    }
    reachable
}

/// Host stencil faces from DB_DEPTH_CONTROL / STENCIL_CONTROL / REFMASK.
/// A stencil clear or a read-only stencil view disables the operations.
pub fn stencil_faces(d: &ResolvedDraw) -> (StencilFace, StencilFace) {
    let ds = &d.depth_state;
    let ctl = ds.stencil_control;
    let ops_disabled = ds.stencil_clear_enable || ds.stencil_write_disable;
    // An exported operation value: REPLACE_OP stays a replacement whatever
    // the register's value (the exported one is the host reference).
    let op_export = stencil_op_export(d);
    let face = |compare: u32, refmask: u32, fail: u32, pass: u32, zfail: u32| {
        let write_mask = if ops_disabled { 0 } else { (refmask >> 16) & 0xFF };
        let op_value = (refmask >> 24) & 0xFF;
        let mut f = stencil_face(compare, [fail, pass, zfail], (refmask >> 8) & 0xFF, write_mask, refmask & 0xFF, op_value);
        if op_export && write_mask != 0 {
            // The register's value stays the reference when the program's
            // value cannot be exported (no host stencil export).
            for (op, host) in [(fail, &mut f.fail), (pass, &mut f.pass), (zfail, &mut f.depth_fail)] {
                if op & 0xF == 4 {
                    *host = vk::StencilOp::REPLACE;
                }
            }
            f.reference = (f.reference & !write_mask) | (op_value & write_mask);
        }
        f
    };
    let front = face(ds.stencil_compare, ds.stencil_ref_mask, ctl & 0xF, (ctl >> 4) & 0xF, (ctl >> 8) & 0xF);
    let back = if ds.backface {
        face(ds.stencil_compare_back, ds.stencil_ref_mask_back, (ctl >> 12) & 0xF, (ctl >> 16) & 0xF, (ctl >> 20) & 0xF)
    } else {
        front
    };
    (front, back)
}

/// The aspects (depth, stencil) a draw writes in a depth attachment with
/// the given aspects: depth through clears and enabled, unmasked depth
/// writes; stencil through clears and operations that can change the
/// value on a reachable test outcome.
pub fn depth_attachment_writes(d: &ResolvedDraw, has_depth: bool, has_stencil: bool) -> (bool, bool) {
    let ds = &d.depth_state;
    let depth = has_depth && (ds.depth_clear_enable || effective_depth_write(d));
    if !has_stencil {
        return (depth, false);
    }
    let stencil_clear = ds.stencil_clear_enable && !ds.stencil_write_disable;
    if stencil_clear {
        return (depth, true);
    }
    if !ds.stencil_test {
        return (depth, false);
    }
    let depth_compare = compare_op(ds.depth_compare);
    let face_writes = |f: &StencilFace| {
        if f.write_mask == 0 {
            return false;
        }
        let (mut can_pass, mut can_fail) = (f.compare != vk::CompareOp::NEVER, f.compare != vk::CompareOp::ALWAYS);
        if f.compare_mask == 0 {
            // Every masked comparison sees 0 against 0.
            let passes = matches!(f.compare, vk::CompareOp::EQUAL | vk::CompareOp::LESS_OR_EQUAL | vk::CompareOp::GREATER_OR_EQUAL | vk::CompareOp::ALWAYS);
            (can_pass, can_fail) = (passes, !passes);
        }
        let depth_pass = !ds.depth_test || depth_compare != vk::CompareOp::NEVER;
        let depth_fail = ds.depth_test && depth_compare != vk::CompareOp::ALWAYS;
        (can_fail && f.fail != vk::StencilOp::KEEP)
            || (can_pass && depth_pass && f.pass != vk::StencilOp::KEEP)
            || (can_pass && depth_fail && f.depth_fail != vk::StencilOp::KEEP)
    };
    let (front, back) = stencil_faces(d);
    (depth, face_writes(&front) || face_writes(&back))
}

/// Depth writes of a draw: the depth test and depth writes are enabled,
/// the view accepts writes and no depth clear is in progress.
pub fn effective_depth_write(d: &ResolvedDraw) -> bool {
    let ds = &d.depth_state;
    ds.depth_test && ds.depth_write && !ds.depth_write_disable && !ds.depth_clear_enable
}

/// PA_CL_CLIP_CNTL.CLIP_DISABLE: the host cannot turn clipping off, so the
/// vertex stage maps positions through the guest viewport 0 (as VTE_CNTL
/// enables its scale and offset) into the clip space of a host viewport
/// covering the largest render area (see [`dynamic_state`]).
pub fn clip_space_transform(d: &ResolvedDraw, limits: &vk::PhysicalDeviceLimits) -> ps5shader::resources::ShaderClipSpaceTransform {
    if !clip_disabled(d) {
        return Default::default();
    }
    let v = &d.viewports[0];
    let vte = d.raster.vte_cntl;
    let (w, h) = clip_disabled_extent(limits);
    ps5shader::resources::ShaderClipSpaceTransform {
        enabled: true,
        scale_x: if vte & 1 != 0 { v.scale[0] } else { 1.0 },
        offset_x: if vte & 2 != 0 { v.offset[0] } else { 0.0 },
        scale_y: if vte & 4 != 0 { v.scale[1] } else { 1.0 },
        offset_y: if vte & 8 != 0 { v.offset[1] } else { 0.0 },
        half_extent_x: w * 0.5,
        half_extent_y: h * 0.5,
    }
}

/// The user clip planes of a draw (PA_CL_CLIP_CNTL.UCP_ENA with the
/// PA_CL_UCP registers), written by the vertex stage as clip distances.
/// CLIP_DISABLE disables them too. Taken only in the plain mode (no
/// PS_UCP_MODE, UCP_CULL_ONLY_ENA or PS_UCP_Y_SCALE_NEG) and when the
/// program exports no clip or cull distances of its own (how they combine
/// is unverified); otherwise the planes are not applied (logged once).
/// `enabled`: `RPS5E_USER_CLIP_PLANES` (off: never applied).
pub fn user_clip_planes(d: &ResolvedDraw, enabled: bool) -> ps5shader::resources::ShaderUserClipPlanes {
    let c = d.raster.clip_cntl;
    let mask = c & 0x3F;
    if mask == 0 || clip_disabled(d) || !enabled {
        return Default::default();
    }
    let special = c & (1 << 13 | 3 << 14 | 1 << 17) != 0;
    if special || d.raster.vs_out_cntl & 0xFFFF != 0 {
        crate::log_once(&format!("ps5gpu: user clip planes with PA_CL_CLIP_CNTL {c:#x}, PA_CL_VS_OUT_CNTL {:#x} are not applied", d.raster.vs_out_cntl));
        return Default::default();
    }
    ps5shader::resources::ShaderUserClipPlanes { mask, planes: d.raster.user_clip_planes }
}

pub fn clip_disabled(d: &ResolvedDraw) -> bool {
    d.raster.clip_cntl & (1 << 16) != 0
}

/// The host viewport of a draw without clipping.
pub fn clip_disabled_extent(limits: &vk::PhysicalDeviceLimits) -> (f32, f32) {
    (limits.max_viewport_dimensions[0].clamp(1, 16384) as f32, limits.max_viewport_dimensions[1].clamp(1, 16384) as f32)
}

/// `RPS5E_PER_FACE_BIAS=0`: one depth bias for both faces (a visible front
/// face's), as before per-face biases were drawn in two passes.
fn per_face_depth_bias_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_PER_FACE_BIAS").is_ok_and(|v| v == "0"))
}

/// The dynamic state of a draw into a `width` x `height` render area.
/// `depth_format`: the host depth/stencil attachment format (UNDEFINED
/// without one); `depth_bound` / `stencil_bound`: its aspects.
/// `viewports`: 1, or 16 when the vertex stage selects the viewport;
/// `clip_extent`: the host viewport of a draw without clipping.
#[allow(clippy::too_many_arguments)]
pub fn dynamic_state(
    d: &ResolvedDraw,
    width: u32,
    height: u32,
    depth_format: vk::Format,
    depth_bound: bool,
    stencil_bound: bool,
    rect: bool,
    viewports: u32,
    clip_extent: Option<(f32, f32)>,
) -> DynamicState {
    let (viewport, scissor) = viewport_state(d, 0, width, height, clip_extent);
    let indexed = (viewports > 1).then(|| {
        let mut v = IndexedViewports { viewports: [viewport; MAX_VIEWPORTS], scissors: [scissor; MAX_VIEWPORTS] };
        for i in 1..MAX_VIEWPORTS {
            (v.viewports[i], v.scissors[i]) = viewport_state(d, i, width, height, clip_extent);
        }
        v
    });
    let mut cull = vk::CullModeFlags::NONE;
    let triangles = !rect && d.primitive_type >= primitive::TRI_LIST;
    if triangles {
        if d.raster.cull_front {
            cull |= vk::CullModeFlags::FRONT;
        }
        if d.raster.cull_back {
            cull |= vk::CullModeFlags::BACK;
        }
    }
    // PA_SU_POLY_OFFSET_FRONT_* and BACK_* differ while both faces are
    // drawn: front faces first, then back faces, each with its own bias.
    let face_biases = if depth_bound && triangles && per_face_depth_bias_enabled() { d.raster.face_depth_biases(depth_format) } else { None };
    let mut depth_bias = if depth_bound { d.raster.depth_bias(depth_format) } else { None };
    let mut back_face_bias = None;
    if let Some((front_bias, back_bias)) = face_biases {
        cull = vk::CullModeFlags::BACK;
        depth_bias = Some(front_bias);
        back_face_bias = Some(back_bias);
    }
    let ds = &d.depth_state;
    let (front, back) = if stencil_bound && ds.stencil_test { stencil_faces(d) } else { (StencilFace::default(), StencilFace::default()) };
    DynamicState {
        viewport,
        scissor,
        indexed,
        cull_mode: cull,
        // FACE = 1: clockwise triangles are front-facing.
        front_face: if d.raster.front_face_cw { vk::FrontFace::CLOCKWISE } else { vk::FrontFace::COUNTER_CLOCKWISE },
        depth_test: depth_bound && ds.depth_test,
        depth_write: depth_bound && effective_depth_write(d),
        depth_compare: debug_equal_depth(compare_op(ds.depth_compare)),
        depth_bounds_test: depth_bound && ds.depth_bounds_enable,
        depth_bounds: host_depth_bounds(ds.depth_bounds, DEPTH_RANGE_UNRESTRICTED.load(std::sync::atomic::Ordering::Relaxed)),
        stencil_test: stencil_bound && ds.stencil_test,
        stencil_front: front,
        stencil_back: back,
        depth_bias,
        back_face_bias,
        blend_constants: d.blend_constants,
    }
}

static DEPTH_RANGE_UNRESTRICTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The device takes viewport depth ranges outside [0, 1]
/// (VK_EXT_depth_range_unrestricted): the guest's PA_CL_VPORT depth scale
/// and offset reach the host unclamped (a range such as [-1, 1] of a
/// float depth target). Without it the range is clamped to [0, 1].
pub fn set_depth_range_unrestricted(on: bool) {
    DEPTH_RANGE_UNRESTRICTED.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The host viewport depth range of a guest one (`unrestricted`: the
/// device takes ranges outside [0, 1]).
pub fn host_depth_range(min_d: f32, max_d: f32, unrestricted: bool) -> (f32, f32) {
    if !min_d.is_finite() || !max_d.is_finite() {
        return (0.0, 1.0);
    }
    if unrestricted {
        return (min_d, max_d);
    }
    (min_d.clamp(0.0, 1.0), max_d.clamp(0.0, 1.0))
}

/// The host depth bounds of DB_DEPTH_BOUNDS_MIN/MAX. With
/// VK_EXT_depth_range_unrestricted (depths outside [0, 1] reach the
/// target) the bounds are taken as the guest sets them, infinite ones
/// included; without it they are clamped to [0, 1], which bounds every
/// depth the same way. A NaN bound is replaced by its open end.
pub fn host_depth_bounds(bounds: [f32; 2], unrestricted: bool) -> [f32; 2] {
    let [min, max] = [if bounds[0].is_nan() { f32::NEG_INFINITY } else { bounds[0] }, if bounds[1].is_nan() { f32::INFINITY } else { bounds[1] }];
    if unrestricted {
        [min, max]
    } else {
        [min.clamp(0.0, 1.0), max.clamp(0.0, 1.0)]
    }
}

/// Host viewport and scissor of guest viewport `i`.
fn viewport_state(d: &ResolvedDraw, i: usize, width: u32, height: u32, clip_extent: Option<(f32, f32)>) -> (vk::Viewport, vk::Rect2D) {
    let v = &d.viewports[i];
    let (mut x, mut y, mut w, mut h) = (v.x, v.y, v.width, v.height);
    // VTE_CNTL without viewport scaling: positions are already in window
    // space; an identity viewport over the target keeps them there.
    let vte = d.raster.vte_cntl;
    let scaled = vte & 0x3F != 0;
    // A scaled viewport of zero width rasterises nothing: it keeps a
    // positive host width and an empty scissor.
    let empty = scaled && w == 0.0;
    if let (Some((cw, ch)), false) = (clip_extent, empty) {
        // Without clipping: positions arrive in the clip space of a
        // viewport covering the largest render area.
        x = 0.0;
        y = 0.0;
        w = cw;
        h = ch;
    } else if empty {
        w = 1.0;
    } else if !scaled || h == 0.0 || !w.is_finite() || !h.is_finite() {
        x = 0.0;
        y = 0.0;
        w = width as f32;
        h = height as f32;
    }
    let (min_d, max_d) = host_depth_range(v.min_depth, v.max_depth, DEPTH_RANGE_UNRESTRICTED.load(std::sync::atomic::Ordering::Relaxed));
    let viewport = vk::Viewport { x, y, width: w, height: h, min_depth: min_d, max_depth: max_d };
    let s = d.scissors[i];
    let x0 = s.x0.clamp(0, width as i32);
    let y0 = s.y0.clamp(0, height as i32);
    let x1 = s.x1.clamp(x0, width as i32);
    let y1 = s.y1.clamp(y0, height as i32);
    let scissor = if empty {
        vk::Rect2D { offset: vk::Offset2D { x: x0, y: y0 }, extent: vk::Extent2D::default() }
    } else {
        vk::Rect2D { offset: vk::Offset2D { x: x0, y: y0 }, extent: vk::Extent2D { width: (x1 - x0) as u32, height: (y1 - y0) as u32 } }
    };
    (viewport, scissor)
}

/// PA_CL_CLIP_CNTL: (depth clip enabled, depth clamp needed).
pub fn depth_clip(clip_cntl: u32) -> (bool, bool) {
    let near_disable = clip_cntl & (1 << 26) != 0;
    let far_disable = clip_cntl & (1 << 27) != 0;
    let clip = !(near_disable || far_disable);
    (clip, !clip)
}

/// The DB clamps every fragment depth (after the polygon offset) to the
/// viewport's PA_SC_VPORT_ZMIN/ZMAX, also while primitives are clipped. A
/// host depth clamp clamps to the host viewport's depth range instead, so
/// it reproduces the guest clamp exactly when the first `count` viewports'
/// [ZMIN, ZMAX] equal their host depth ranges; otherwise (a wider or other
/// clamp range) no host clamp is closer than a wrong one.
/// DB_RENDER_OVERRIDE.DISABLE_VIEWPORT_CLAMP (bit 16) turns the guest clamp off.
pub fn viewport_depth_clamp_exact(d: &ResolvedDraw, count: u32) -> bool {
    if d.depth_state.render_override & (1 << 16) != 0 {
        return false;
    }
    let unrestricted = DEPTH_RANGE_UNRESTRICTED.load(std::sync::atomic::Ordering::Relaxed);
    d.viewports.iter().take(count.max(1) as usize).all(|v| depth_clamp_range_matches(v.min_depth, v.max_depth, v.z_min, v.z_max, unrestricted))
}

/// Does a host viewport of guest depth range (`min_d`, `max_d`) clamp to
/// exactly [`z_min`, `z_max`]?
pub fn depth_clamp_range_matches(min_d: f32, max_d: f32, z_min: f32, z_max: f32, unrestricted: bool) -> bool {
    let (n, f) = host_depth_range(min_d, max_d, unrestricted);
    n.min(f) == z_min && n.max(f) == z_max
}

/// The export (MRT index) that feeds color target `slot`: the Nth export
/// SPI_SHADER_COL_FORMAT gives a format goes to the Nth target
/// CB_SHADER_MASK enables; when the two do not pair up, export N feeds
/// slot N. None when no export reaches the slot.
pub fn export_for_slot(col_format: u32, shader_mask: u32, slot: u32) -> Option<u32> {
    if slot >= 8 {
        return None;
    }
    let nibbles = |v: u32| (0..8u32).filter(move |i| (v >> (i * 4)) & 0xF != 0);
    let exports = nibbles(col_format).count();
    if exports == 0 || exports != nibbles(shader_mask).count() {
        return Some(slot);
    }
    nibbles(shader_mask).position(|s| s == slot).and_then(|i| nibbles(col_format).nth(i))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_clamp_only_where_the_host_range_is_the_guest_clamp() {
        // Clipping on: no clamp from the clip register alone.
        assert_eq!(depth_clip(0), (true, false));
        assert_eq!(depth_clip(1 << 26), (false, true));
        // [-1, 1] viewport (OpenGL-style) whose ZMIN/ZMAX are [-1, 1]: a
        // clear at z 1.0 pushed far by a polygon offset lands on 1.0.
        assert!(depth_clamp_range_matches(-1.0, 1.0, -1.0, 1.0, true));
        // [0, 1] viewport with ZMIN/ZMAX [0, 1], also reversed.
        assert!(depth_clamp_range_matches(0.0, 1.0, 0.0, 1.0, false));
        assert!(depth_clamp_range_matches(1.0, 0.0, 0.0, 1.0, true));
        // A clamp range wider than the viewport range: the host clamp would
        // cut depths the guest keeps.
        assert!(!depth_clamp_range_matches(0.0, 1.0, -1.0, 2.0, true));
        assert!(!depth_clamp_range_matches(0.0, 0.5, 0.0, 1.0, true));
        // Without unrestricted ranges a [-1, 1] viewport becomes [0, 1].
        assert!(!depth_clamp_range_matches(-1.0, 1.0, -1.0, 1.0, false));
    }

    #[test]
    fn export_routing() {
        // MRT0 and MRT1 exported to targets 0 and 4.
        assert_eq!(export_for_slot(0x44, 0x000F_000F, 4), Some(1));
        assert_eq!(export_for_slot(0x44, 0x000F_000F, 0), Some(0));
        assert_eq!(export_for_slot(0x44, 0x000F_000F, 1), None);
        // Unpaired registers: identity.
        assert_eq!(export_for_slot(0, 0xF, 3), Some(3));
    }

    #[test]
    fn infinite_depth_bounds_are_kept_where_depths_are_unrestricted() {
        let inf = f32::INFINITY;
        assert_eq!(host_depth_bounds([0.25, 0.75], true), [0.25, 0.75]);
        assert_eq!(host_depth_bounds([0.0, inf], true), [0.0, inf]);
        assert_eq!(host_depth_bounds([-inf, 1.0], true), [-inf, 1.0]);
        assert_eq!(host_depth_bounds([-inf, inf], false), [0.0, 1.0]);
        assert_eq!(host_depth_bounds([-0.5, 2.0], false), [0.0, 1.0]);
        assert_eq!(host_depth_bounds([f32::NAN, f32::NAN], true), [-inf, inf]);
    }

    #[test]
    fn stencil_replacement_against_a_known_value_inverts() {
        // EQUAL against 0xff, REPLACE_OP 0x40 on pass, both masks 0xff: the
        // passing pixels hold 0xff, so the replacement inverts 0xbf.
        let f = stencil_face_of(2, [0, 4, 0], 0xFF, 0xFF, 0xFF, 0x40, true);
        assert_eq!((f.fail, f.pass, f.depth_fail), (vk::StencilOp::KEEP, vk::StencilOp::INVERT, vk::StencilOp::KEEP));
        assert_eq!((f.write_mask, f.reference), (0xBF, 0xFF));
        // NOTEQUAL fixes the value only where the test fails.
        let f = stencil_face_of(5, [4, 0, 0], 0x0F, 0x0F, 0x03, 0x05, true);
        assert_eq!((f.fail, f.write_mask), (vk::StencilOp::INVERT, 0x06));
        let f = stencil_face_of(5, [0, 4, 0], 0x0F, 0x0F, 0x03, 0x05, true);
        assert_eq!((f.pass, f.write_mask, f.reference), (vk::StencilOp::REPLACE, 0x0F, 0x03), "not known: kept as before");
        // GEQUAL against the smallest value: passing pixels hold 0.
        let f = stencil_face_of(6, [0, 4, 4], 0xFF, 0xFF, 0x00, 0x81, true);
        assert_eq!((f.pass, f.depth_fail, f.write_mask), (vk::StencilOp::INVERT, vk::StencilOp::INVERT, 0x81));
        // A written bit outside the compare mask, or another active
        // operation beside the inversion, keeps the reference.
        let f = stencil_face_of(2, [0, 4, 0], 0x0F, 0xFF, 0x0F, 0x40, true);
        assert_eq!((f.pass, f.write_mask), (vk::StencilOp::REPLACE, 0xFF));
        let f = stencil_face_of(2, [1, 4, 0], 0xFF, 0xFF, 0xFF, 0x40, true);
        assert_eq!((f.fail, f.pass, f.write_mask), (vk::StencilOp::ZERO, vk::StencilOp::REPLACE, 0xFF));
        // Switched off: the earlier replacement.
        let f = stencil_face_of(2, [0, 4, 0], 0xFF, 0xFF, 0xFF, 0x40, false);
        assert_eq!((f.pass, f.write_mask, f.reference), (vk::StencilOp::REPLACE, 0xFF, 0xFF));
    }

    #[test]
    fn stencil_logic_operations_draw_the_bits_they_change() {
        use vk::StencilOp as S;
        let face = |compare: u32, ops: [u32; 3], compare_mask: u32, write_mask: u32, test: u32, op_value: u32| stencil_face_of(compare, ops, compare_mask, write_mask, test, op_value, true);
        // AND 0xF0 under write mask 0x3C: clears 0x0C.
        let f = face(7, [0, 10, 0], 0xFF, 0x3C, 0, 0xF0);
        assert_eq!((f.pass, f.write_mask), (S::ZERO, 0x0C));
        // OR 0x05 under 0x0F with compare ALWAYS: sets 0x05 through the reference.
        let f = face(7, [0, 11, 0], 0xFF, 0x0F, 0x80, 0x05);
        assert_eq!((f.pass, f.write_mask, f.reference), (S::REPLACE, 0x05, 0x85));
        // XOR of part of the writable bits inverts those; XNOR the clear ones.
        let f = face(7, [12, 0, 0], 0xFF, 0xFF, 0, 0x81);
        assert_eq!((f.fail, f.write_mask), (S::INVERT, 0x81));
        let f = face(7, [0, 0, 15], 0xFF, 0x0F, 0, 0x0C);
        assert_eq!((f.depth_fail, f.write_mask), (S::INVERT, 0x03));
        // Several logic operations on the same bits share the write mask;
        // on different bits they are not drawn (as without the logic
        // operations: AND keeps, XOR inverts every writable bit).
        let f = face(7, [10, 15, 0], 0xFF, 0xFF, 0, 0xF0);
        assert_eq!((f.fail, f.pass, f.write_mask), (S::ZERO, S::INVERT, 0x0F), "AND 0xf0 clears and XNOR 0xf0 inverts 0x0f");
        let f = face(7, [12, 15, 0], 0xFF, 0xFF, 0, 0xF0);
        assert_eq!((f.fail, f.pass, f.write_mask), (S::INVERT, S::KEEP, 0xFF), "XOR 0xf0 inverts 0xf0, XNOR 0xf0 inverts 0x0f");
        // An operation changing nothing keeps the value.
        let f = face(7, [0, 10, 0], 0xFF, 0x0F, 0, 0xFF);
        assert_eq!(f.pass, S::KEEP);
        // An OR of compared bits clear in the test value changes the
        // comparison: not drawn (kept).
        let f = face(2, [0, 11, 0], 0xFF, 0xFF, 0x00, 0x01);
        assert_eq!(f.pass, S::KEEP);
        // A logic operation on part of the bits beside an operation writing
        // all of them: not drawn.
        let f = face(7, [1, 12, 0], 0xFF, 0xFF, 0, 0x01);
        assert_eq!((f.fail, f.pass, f.write_mask), (S::ZERO, S::INVERT, 0xFF));
        // Beside it, a logic operation on every writable bit draws.
        let f = face(7, [1, 11, 0], 0xFF, 0x0F, 0, 0xFF);
        assert_eq!((f.fail, f.pass, f.write_mask, f.reference), (S::ZERO, S::REPLACE, 0x0F, 0x0F));
    }

    #[test]
    fn polygon_mode_takes_the_drawn_faces_type() {
        let poly = |front: u32, back: u32, cull: u32| cull | 1 << 3 | front << 5 | back << 8;
        let mode = |cntl: u32| polygon_mode_of(cntl, true, true, true);
        // POLY_MODE clear: filled whatever the types say.
        assert_eq!(mode(poly(0, 0, 0) & !(1 << 3)), vk::PolygonMode::FILL);
        // Reserved POLY_MODE 2/3 stay filled.
        assert_eq!(mode(2 << 3 | 1 << 5), vk::PolygonMode::FILL);
        assert_eq!(mode(poly(1, 1, 0)), vk::PolygonMode::LINE);
        assert_eq!(mode(poly(0, 0, 0)), vk::PolygonMode::POINT);
        // Front culled: the back type draws.
        assert_eq!(mode(poly(2, 1, 1)), vk::PolygonMode::LINE);
        assert_eq!(mode(poly(1, 2, 1)), vk::PolygonMode::FILL);
        // Back culled: the front type draws.
        assert_eq!(mode(poly(0, 2, 2)), vk::PolygonMode::POINT);
        // Both culled: nothing rasterizes.
        assert_eq!(mode(poly(1, 1, 3)), vk::PolygonMode::FILL);
        // A point mode needs the one-pixel default point size.
        assert_eq!(polygon_mode_of(poly(0, 0, 0), true, false, true), vk::PolygonMode::FILL);
        assert_eq!(polygon_mode_of(poly(1, 1, 0), false, true, true), vk::PolygonMode::FILL);
        // Earlier behaviour: the front type even when front faces are culled.
        assert_eq!(polygon_mode_of(poly(2, 1, 1), true, true, false), vk::PolygonMode::FILL);
    }

    #[test]
    fn stencil_and_blend_tables() {
        assert_eq!(stencil_op(3, 0xFF, 0), vk::StencilOp::REPLACE);
        assert_eq!(stencil_op(5, 0, 0), vk::StencilOp::KEEP);
        // ADD / SUB step by STENCILOPVAL: 0 changes nothing, 1 is the
        // host increment / decrement.
        assert_eq!(stencil_op(5, 0xFF, 0), vk::StencilOp::KEEP);
        assert_eq!(stencil_op(9, 0xFF, 0), vk::StencilOp::KEEP);
        assert_eq!(stencil_op(5, 0xFF, 1), vk::StencilOp::INCREMENT_AND_CLAMP);
        assert_eq!(stencil_op(6, 0xFF, 1), vk::StencilOp::DECREMENT_AND_CLAMP);
        assert_eq!(stencil_op(8, 0xFF, 1), vk::StencilOp::INCREMENT_AND_WRAP);
        assert_eq!(stencil_op(9, 0xFF, 1), vk::StencilOp::DECREMENT_AND_WRAP);
        assert_eq!(blend_factor(5), vk::BlendFactor::ONE_MINUS_SRC_ALPHA);
        assert_eq!(blend_op(4), vk::BlendOp::REVERSE_SUBTRACT);
        assert_eq!(compare_op(3), vk::CompareOp::LESS_OR_EQUAL);
        assert_eq!(parse_debug_equal_depth(None), None);
        assert_eq!(parse_debug_equal_depth(Some("ge")), Some(vk::CompareOp::GREATER_OR_EQUAL));
        assert_eq!(parse_debug_equal_depth(Some(" le")), Some(vk::CompareOp::LESS_OR_EQUAL));
        assert_eq!(parse_debug_equal_depth(Some("1")), None);
        assert_eq!(host_topology(primitive::QUAD_LIST_LEGACY, RectMode::None), Some(vk::PrimitiveTopology::TRIANGLE_LIST));
        assert_eq!(host_topology(primitive::RECT_LIST, RectMode::Tessellation), Some(vk::PrimitiveTopology::PATCH_LIST));
    }

    #[test]
    fn stencil_reference_merges_replacements() {
        // REPLACE_OP on pass with compare ALWAYS: the reference carries the
        // operation value in the written bits.
        let f = stencil_face(7, [0, 4, 0], 0xFF, 0x0F, 0x30, 0x05);
        assert_eq!(f.pass, vk::StencilOp::REPLACE);
        assert_eq!(f.reference, 0x35);
        // REPLACE_TEST keeps the test value; ONES writes all ones.
        let f = stencil_face(3, [2, 3, 0], 0xF0, 0x0F, 0x20, 0);
        assert_eq!((f.fail, f.pass), (vk::StencilOp::REPLACE, vk::StencilOp::REPLACE));
        assert_eq!(f.reference, 0x2F);
        // No write mask: every operation keeps the value.
        let f = stencil_face(3, [1, 4, 7], 0xFF, 0, 1, 1);
        assert_eq!((f.fail, f.pass, f.depth_fail), (vk::StencilOp::KEEP, vk::StencilOp::KEEP, vk::StencilOp::KEEP));
        // XOR of the writable bits inverts them; XOR outside them keeps.
        assert_eq!(stencil_op(12, 0x0F, 0xFF), vk::StencilOp::INVERT);
        assert_eq!(stencil_op(12, 0x0F, 0xF0), vk::StencilOp::KEEP);
    }

    /// STENCIL_OP_VAL_EXPORT_ENABLE: DB_SHADER_CONTROL 0x45 with
    /// DB_STENCIL_CONTROL 0x40040 (REPLACE_OP on pass, both faces), write
    /// mask 0xff and compare ALWAYS replaces with the exported value: the
    /// host op is REPLACE even when the register's operation value is 0.
    #[test]
    fn stencil_operation_value_export() {
        let b = crate::render::RegisterBanks { context: &[], shader: &[], uconfig: &[] };
        let mut d = crate::render::resolve_draw(&b, &crate::render::DrawPacket::default());
        d.db_shader_control = 0x45;
        d.shader_z_format = 9;
        let ds = &mut d.depth_state;
        ds.stencil_test = true;
        ds.backface = true;
        ds.stencil_control = 0x40040;
        ds.stencil_compare = 7;
        ds.stencil_compare_back = 7;
        ds.stencil_ref_mask = 0xFF << 16;
        ds.stencil_ref_mask_back = 0xFF << 16;
        assert!(stencil_op_export(&d));
        let (front, back) = stencil_faces(&d);
        assert_eq!((front.pass, back.pass), (vk::StencilOp::REPLACE, vk::StencilOp::REPLACE));
        // A compare that reads the reference, a test value export or an
        // increment keep the register's value.
        let mut e = d.clone();
        e.depth_state.stencil_compare = 3;
        assert!(!stencil_op_export(&e));
        let mut e = d.clone();
        e.db_shader_control |= 2;
        assert!(!stencil_op_export(&e));
        let mut e = d.clone();
        e.depth_state.stencil_control = 0x40050;
        assert!(!stencil_op_export(&e));
        // Without the export bit the operation value 0 writes zeros.
        let mut e = d.clone();
        e.db_shader_control = 0x41;
        assert_eq!(stencil_faces(&e).0.pass, vk::StencilOp::ZERO);
    }

    /// A depth range outside [0, 1] (scale 1, offset 0 in -1..1 clip
    /// space: [-1, 1]) reaches a device with VK_EXT_depth_range_unrestricted
    /// as set; otherwise it is clamped.
    #[test]
    fn viewport_depth_range_outside_unit() {
        assert_eq!(host_depth_range(-1.0, 1.0, true), (-1.0, 1.0));
        assert_eq!(host_depth_range(-1.0, 1.0, false), (0.0, 1.0));
        assert_eq!(host_depth_range(0.25, 0.5, false), (0.25, 0.5));
        assert_eq!(host_depth_range(f32::NAN, 1.0, true), (0.0, 1.0));
    }

    #[test]
    fn blend_mappings() {
        use crate::render::formats::ComponentMap;
        use crate::render::BlendState;
        // SRC_ALPHA / ONE_MINUS_SRC_ALPHA, add.
        let over = BlendState::decode(4 | (5 << 8) | (1 << 30));
        assert_eq!(classify_blend_mapping(&over, ComponentMap::IDENTITY), BlendMapping::Direct);
        assert_eq!(classify_blend_mapping(&over, ComponentMap::BGRA), BlendMapping::Direct);
        assert_eq!(classify_blend_mapping(&over, ComponentMap::ABGR), BlendMapping::SourceAlpha);
        // Separate alpha ONE / ONE_MINUS_SRC_ALPHA.
        let sep = BlendState::decode(4 | (5 << 8) | (1 << 16) | (5 << 24) | (1 << 29) | (1 << 30));
        assert_eq!(classify_blend_mapping(&sep, ComponentMap::ABGR), BlendMapping::SourceAlphaOne);
        // Destination alpha cannot move.
        let dst = BlendState::decode(6 | (7 << 8) | (1 << 30));
        assert_eq!(classify_blend_mapping(&dst, ComponentMap::ABGR), BlendMapping::Unsupported);
        // Constant color with a swizzle.
        let constant = BlendState::decode(13 | (1 << 30));
        assert_eq!(classify_blend_mapping(&constant, ComponentMap::BGRA), BlendMapping::Unsupported);
    }

    /// A 2x target scanned without PA_SC_MODE_CNTL_0.MSAA_ENABLE keeps its
    /// two samples, both at the pixel center; with alpha-to-coverage,
    /// points, lines or a depth target the centered placement is not drawn.
    #[test]
    fn multisampled_target_without_msaa_enable_is_centered() {
        use vk::PrimitiveTopology as T;
        assert_eq!(sample_placement(2, false, T::TRIANGLE_LIST, false, false), SamplePlacement::Centered);
        assert_eq!(sample_placement(8, false, T::TRIANGLE_STRIP, false, false), SamplePlacement::Centered);
        // Rectangles expanded by tessellation are filled primitives.
        assert_eq!(sample_placement(4, false, T::PATCH_LIST, false, false), SamplePlacement::Centered);
        let (grid, locations) = centered_sample_locations(2);
        assert_eq!(locations.len(), 2);
        assert!(locations.iter().all(|l| l.x == 0.5 && l.y == 0.5), "every sample at the pixel center");
        // The bit set, or a single sample: the standard locations.
        assert_eq!(sample_placement(2, true, T::TRIANGLE_LIST, false, false), SamplePlacement::Standard);
        assert_eq!(sample_placement(1, false, T::TRIANGLE_LIST, false, false), SamplePlacement::Standard);
        // DB_ALPHA_TO_MASK 0xaa01, points, lines or a depth target.
        assert!(matches!(sample_placement(2, false, T::TRIANGLE_LIST, false, true), SamplePlacement::CenteredUnsupported(_)));
        for t in [T::POINT_LIST, T::LINE_LIST, T::LINE_STRIP] {
            assert!(matches!(sample_placement(2, false, t, false, false), SamplePlacement::CenteredUnsupported(_)));
        }
        assert!(matches!(sample_placement(2, false, T::TRIANGLE_LIST, true, false), SamplePlacement::CenteredUnsupported(_)));
        // Centered samples carry no per-pixel grid, whatever the location
        // registers of the 2x2 quad hold.
        assert_eq!((grid.width, grid.height), (1, 1));
    }
}

/// Whether the depth and stencil tests run before the pixel program:
/// DB_SHADER_CONTROL.DEPTH_BEFORE_SHADER (bit 12), or an early Z order
/// (Z_ORDER bit 4: EARLY_Z_THEN_LATE_Z / EARLY_Z_THEN_RE_Z) when the program
/// neither kills nor exports depth, stencil or a mask (bits 0, 1, 6, 8).
/// Never with alpha-to-coverage (the coverage comes from the program).
pub fn early_fragment_tests(db_shader_control: u32, alpha_to_coverage: bool) -> bool {
    let c = db_shader_control;
    !alpha_to_coverage && (c & 0x1000 != 0 || (c & 0x10 != 0 && c & 0x147 == 0))
}

#[cfg(test)]
mod early_tests {
    #[test]
    fn early_fragment_tests_follow_db_shader_control() {
        for (c, expected) in [(0x0000, false), (0x0020, false), (0x0010, true), (0x0030, true), (0x0050, false), (0x0011, false), (0x1000, true), (0x1050, true), (0x1031, true), (0x0110, false), (0x0012, false), (0x0014, false)] {
            assert_eq!(super::early_fragment_tests(c, false), expected, "{c:#x}");
        }
        assert!(!super::early_fragment_tests(0x0010, true) && !super::early_fragment_tests(0x1000, true), "alpha-to-coverage keeps late tests");
    }
}
