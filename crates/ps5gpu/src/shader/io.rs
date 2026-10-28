// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Stage interface: system values, pixel attribute interpolation, exports
//! (position, parameters, color targets, depth) and pixel discard.

use super::emit::{custom, Emitter, R};
use ps5shader::resources::ShaderStage;
use ps5shader::ssa::*;
use ps5shader::PixelOutputKind;
use rspirv::dr::Operand;
use rspirv::spirv::{BuiltIn, Capability, Decoration, GlslStd450Op as GLOp, Op as S, StorageClass as SC, Word};
use std::collections::BTreeMap;

/// Interface variables declared so far.
#[derive(Default)]
pub struct Interface {
    /// `gl_PerVertex` output block (vertex).
    pub per_vertex: Option<Word>,
    /// Vertex parameter outputs by location.
    pub params: BTreeMap<u32, Word>,
    /// Pixel color outputs by host location: (variable, kind).
    pub colors: BTreeMap<u32, (Word, PixelOutputKind)>,
    pub frag_depth: Option<Word>,
    pub frag_stencil: Option<Word>,
    /// Pixel per-vertex attribute inputs by location.
    pub attributes: BTreeMap<u32, Word>,
    /// Barycentric inputs: (builtin, centroid) -> variable.
    pub barycentrics: BTreeMap<(u32, bool), Word>,
    /// Vertex built-in outputs of the misc position exports (point size,
    /// layer, viewport index, clip and cull distances) by built-in.
    pub misc: BTreeMap<u32, Word>,
    /// Pixel: the second blend source of target 0 (Location 0, Index 1).
    pub second_source: Option<(Word, PixelOutputKind)>,
    /// Pixel: the SampleMask output (alpha-to-coverage).
    pub sample_mask: Option<Word>,
}

/// Render-target array index field of the pixel ancillary VGPR.
pub const ANCILLARY_LAYER_SHIFT: u32 = 16;
pub const ANCILLARY_LAYER_MASK: u32 = 0x1FFF;

/// The pixel ancillary word for a render-target layer and sample.
#[cfg(test)]
pub fn pack_ancillary(layer: u32, sample: u32) -> u32 {
    (layer & ANCILLARY_LAYER_MASK) << ANCILLARY_LAYER_SHIFT | (sample & 0xF) << 8
}

/// Whether a use of the pixel ancillary word can see its sample id bits
/// (8..11): anything but a constant bit-field extract of other bits.
fn ancillary_reads_sample(f: &Function) -> bool {
    let ancillary: Vec<Value> = f
        .order
        .iter()
        .flat_map(|&b| f.block(b).insts.iter().copied())
        .filter(|&v| matches!(f.op(v), Op::SystemValue(SystemValue::Ancillary)))
        .collect();
    if ancillary.is_empty() {
        return false;
    }
    f.order.iter().flat_map(|&b| f.block(b).insts.iter().copied()).any(|v| {
        let args = f.args(v);
        if !args.iter().any(|a| ancillary.contains(a)) {
            return false;
        }
        match f.op(v) {
            Op::BitFieldUExtract | Op::BitFieldSExtract if ancillary.contains(&args[0]) && !ancillary.contains(&args[1]) && !ancillary.contains(&args[2]) => {
                match (f.as_const(args[1]), f.as_const(args[2])) {
                    (Some(o), Some(n)) => {
                        let (o, n) = (o & 31, n & 31);
                        // Bits [o, o + n) meet 8..12.
                        n != 0 && o < 12 && o + n > 8
                    }
                    _ => true,
                }
            }
            _ => true,
        }
    })
}

/// Prologue work: declare the stage outputs the program exports so the
/// interface is complete even when an export is never reached.
pub fn prologue(e: &mut Emitter) -> R<()> {
    let f = e.f;
    let control = e.input.request.position_export_control;
    let mut params = Vec::new();
    let (mut layer, mut viewport) = (false, false);
    for &b in &f.order {
        for &v in &f.block(b).insts {
            if let Op::Export(a) = f.op(v) {
                if (32..64).contains(&a.target) {
                    params.push(a.target as u32 - 32);
                }
                if (13..=16).contains(&a.target) {
                    for c in (0..4).filter(|c| a.enable >> c & 1 != 0) {
                        let out = ps5shader::resources::decode_position_export_component(control, a.target as u32 - 12, c);
                        layer |= out.layer;
                        viewport |= out.viewport;
                    }
                }
            }
        }
    }
    match e.stage {
        ShaderStage::Vertex if e.mesh.is_some() => super::mesh::declare(e, &params, layer, viewport),
        ShaderStage::Vertex => {
            per_vertex(e);
            for p in params {
                param_output(e, p);
            }
        }
        ShaderStage::Pixel => {
            let outputs = e.input.request.pixel_outputs.clone();
            for o in outputs {
                color_output(e, o.host_location, o.kind);
            }
            if e.input.request.pixel_quad_sample_masks.is_some() {
                sample_mask(e);
            }
            if e.input.request.pixel_ordered {
                // The critical section of a primitive-ordered program is
                // the whole program.
                e.cap(Capability::FragmentShaderPixelInterlockEXT);
                e.exts.insert("SPV_EXT_fragment_shader_interlock");
                e.void(S::BeginInvocationInterlockEXT, vec![]);
            }
        }
        _ => {}
    }
    Ok(())
}

fn per_vertex(e: &mut Emitter) -> Word {
    if let Some(v) = e.io.per_vertex {
        return v;
    }
    let v4 = e.t_fvec(4);
    let id = e.b.id();
    let s = e.b.type_struct_id(Some(id), [v4]);
    e.b.member_decorate(s, 0, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::Position)]);
    // The hardware computes a position the same way in every program that
    // runs the same instructions: a depth pre-pass and the pass drawing
    // with an EQUAL depth test must agree to the bit, whatever the host
    // compiler does with the rest of each pipeline.
    e.b.member_decorate(s, 0, Decoration::Invariant, []);
    e.decorate(s, Decoration::Block, &[]);
    let var = e.global(s, SC::Output);
    e.io.per_vertex = Some(var);
    var
}

fn param_output(e: &mut Emitter, location: u32) -> Word {
    if let Some(&v) = e.io.params.get(&location) {
        return v;
    }
    let v4 = e.t_fvec(4);
    let var = e.global(v4, SC::Output);
    e.decorate(var, Decoration::Location, &[location]);
    e.io.params.insert(location, var);
    var
}

fn color_output(e: &mut Emitter, location: u32, kind: PixelOutputKind) -> Word {
    if let Some(&(v, _)) = e.io.colors.get(&location) {
        return v;
    }
    let t = match kind {
        PixelOutputKind::Float => e.t_fvec(4),
        PixelOutputKind::Uint => e.t_uvec(4),
        PixelOutputKind::Sint => e.t_ivec(4),
    };
    let var = e.global(t, SC::Output);
    e.decorate(var, Decoration::Location, &[location]);
    e.io.colors.insert(location, (var, kind));
    var
}

/// Work before a return (nothing yet: outputs are written at exports).
pub fn before_return(e: &mut Emitter) -> R<()> {
    if e.stage == ShaderStage::Pixel && e.input.request.pixel_ordered {
        e.void(S::EndInvocationInterlockEXT, vec![]);
    }
    Ok(())
}

fn u32_bits_of_f32(e: &mut Emitter, f: Word) -> Word {
    let tu = e.t_u32();
    e.bitcast(tu, f)
}

/// The value of a hardware-provided input.
pub fn system_value(e: &mut Emitter, s: SystemValue) -> R<Word> {
    let tu = e.t_u32();
    let uv3 = e.t_uvec(3);
    let comp = |e: &mut Emitter, b: BuiltIn, c: u8| {
        let v = e.builtin_value(b, uv3);
        e.hoist_extract(tu, v, c as u32)
    };
    let size = e.input.request.local_size;
    let size = [size[0].max(1), size[1].max(1), size[2].max(1)];
    // The host work-group is the guest's flattened into whole waves.
    let paired = e.lanes == 2 || e.split.is_some();
    Ok(match s {
        // Compute programs: the lanes that start inactive.
        SystemValue::HelperInvocation if e.stage == ShaderStage::Compute && e.mesh.is_none() => system_value(e, SystemValue::Custom(custom::OUT_OF_RANGE))?,
        SystemValue::LocalInvocationId(c) if paired => {
            let g = guest_index(e);
            local_component(e, g, size, c)
        }
        SystemValue::GlobalInvocationId(c) if paired => {
            let g = guest_index(e);
            let l = local_component(e, g, size, c);
            let w = comp(e, BuiltIn::WorkgroupId, c);
            let n = e.cu(size[c as usize]);
            let base = e.hoist(S::IMul, tu, &[w, n]);
            e.hoist(S::IAdd, tu, &[base, l])
        }
        SystemValue::LocalInvocationId(c) => comp(e, BuiltIn::LocalInvocationId, c),
        SystemValue::WorkgroupId(c) => comp(e, BuiltIn::WorkgroupId, c),
        SystemValue::GlobalInvocationId(c) => comp(e, BuiltIn::GlobalInvocationId, c),
        SystemValue::LocalInvocationIndex => guest_index(e),
        SystemValue::SubgroupId => {
            let i = guest_index(e);
            let w = e.cu(e.f.wave_size);
            e.hoist(S::UDiv, tu, &[i, w])
        }
        SystemValue::Custom(custom::OUT_OF_RANGE) => {
            let g = guest_index(e);
            let n = e.cu(size[0] * size[1] * size[2]);
            let tb = e.t_bool();
            let mut out = e.hoist(S::UGreaterThanEqual, tb, &[g, n]);
            // A dispatch with thread dimensions: invocations at or past the
            // thread counts (in its partial last groups) do not run.
            if e.input.request.bindings.uses_dispatch_thread_limits {
                let first = e.input.request.bindings.dispatch_thread_limits_dword();
                for c in 0..3u8 {
                    let limit = e.shader_data(first + c as u32)?;
                    let id = system_value(e, SystemValue::GlobalInvocationId(c))?;
                    let past = e.hoist(S::UGreaterThanEqual, tb, &[id, limit]);
                    out = e.hoist(S::LogicalOr, tb, &[out, past]);
                }
            }
            let (one, zero) = (e.cu(1), e.cu(0));
            e.hoist(S::Select, tu, &[out, one, zero])
        }
        SystemValue::VertexIndex | SystemValue::InstanceIndex | SystemValue::PrimitiveId => {
            let b = match s {
                SystemValue::VertexIndex => BuiltIn::VertexIndex,
                SystemValue::InstanceIndex => BuiltIn::InstanceIndex,
                _ => BuiltIn::PrimitiveId,
            };
            if b == BuiltIn::PrimitiveId && e.stage == ShaderStage::Pixel {
                e.cap(Capability::Geometry);
            }
            let ti = e.t_i32();
            let var = e.builtin_input(b, ti);
            let x = e.hoist(S::Load, ti, &[var]);
            e.hoist(S::Bitcast, tu, &[x])
        }
        SystemValue::FragCoord(c) => {
            let v4 = e.t_fvec(4);
            let v = e.builtin_value(BuiltIn::FragCoord, v4);
            let tf = e.t_f32();
            let x = e.hoist_extract(tf, v, c as u32);
            e.hoist(S::Bitcast, tu, &[x])
        }
        SystemValue::FrontFacing => {
            let tb = e.t_bool();
            let f = e.builtin_value(BuiltIn::FrontFacing, tb);
            // SPI_BARYC_CNTL.FRONT_FACE_ALL_BITS: 0xFFFFFFFF/0 instead of
            // +1.0f/-1.0f (measured on hardware).
            let (a, b) = if e.input.request.front_face_all_bits { (0xFFFF_FFFF, 0) } else { (0x3F80_0000, 0xBF80_0000) };
            let (one, minus) = (e.cu(a), e.cu(b));
            e.hoist(S::Select, tu, &[f, one, minus])
        }
        SystemValue::SampleId => e.cu(0),
        SystemValue::HelperInvocation => {
            let tb = e.t_bool();
            let h = if e.feat.demote {
                e.cap(Capability::DemoteToHelperInvocation);
                e.exts.insert("SPV_EXT_demote_to_helper_invocation");
                e.hoist(S::IsHelperInvocationEXT, tb, &[])
            } else {
                e.builtin_value(BuiltIn::HelperInvocation, tb)
            };
            let (one, zero) = (e.cu(1), e.cu(0));
            e.hoist(S::Select, tu, &[h, one, zero])
        }
        SystemValue::Barycentric(n) => {
            let i = barycentric(e, n)?;
            u32_bits_of_f32(e, i)
        }
        SystemValue::Ancillary => {
            // Packed pixel ancillary word: the primitive class in bits
            // 0..1, sample id in bits 8..11, render-target array index in
            // bits 16..28 (layered passes such as colour-grading LUTs read
            // it). The sample id is read only when the program looks at
            // those bits: SampleId makes the shader run per sample.
            e.cap(Capability::Geometry);
            let ti = e.t_i32();
            let var = e.builtin_input(BuiltIn::Layer, ti);
            let x = e.hoist(S::Load, ti, &[var]);
            let layer = e.hoist(S::Bitcast, tu, &[x]);
            let mask = e.cu(ANCILLARY_LAYER_MASK);
            let layer = e.hoist(S::BitwiseAnd, tu, &[layer, mask]);
            let shift = e.cu(ANCILLARY_LAYER_SHIFT);
            let word = e.hoist(S::ShiftLeftLogical, tu, &[layer, shift]);
            let class = e.cu(e.input.request.pixel_raster_primitive & 3);
            let word = e.hoist(S::BitwiseOr, tu, &[word, class]);
            // SampleId makes the host run the program per sample, so it is
            // read only when the guest iterates samples (PS_ITER_SAMPLE);
            // otherwise the sample number is 0.
            if !e.input.request.pixel_sample_rate || !ancillary_reads_sample(e.f) {
                return Ok(word);
            }
            e.cap(Capability::SampleRateShading);
            let var = e.builtin_input(BuiltIn::SampleId, ti);
            let s = e.hoist(S::Load, ti, &[var]);
            let s = e.hoist(S::Bitcast, tu, &[s]);
            let m = e.cu(0xF);
            let s = e.hoist(S::BitwiseAnd, tu, &[s, m]);
            let eight = e.cu(8);
            let s = e.hoist(S::ShiftLeftLogical, tu, &[s, eight]);
            e.hoist(S::BitwiseOr, tu, &[word, s])
        }
        SystemValue::SampleCoverage => {
            // SAMPLE_COVERAGE_ENA: the samples the primitive covers in this
            // pixel (gl_SampleMaskIn; per-sample programs see only their own
            // sample there, the whole pixel's coverage is not available).
            let ti = e.t_i32();
            let one = e.cu(1);
            let arr = e.b.type_array(ti, one);
            let var = e.builtin_input(BuiltIn::SampleMask, arr);
            let p = e.ptr(SC::Input, ti);
            e.in_prologue(|e| {
                let zero = e.cu(0);
                let ptr = e.access(p, var, &[zero]);
                let x = e.ins(S::Load, ti, &[ptr]);
                e.ins(S::Bitcast, tu, &[x])
            })
        }
        SystemValue::Custom(c @ (custom::BASE_VERTEX | custom::BASE_INSTANCE | custom::VERTEX_ID_REL | custom::INSTANCE_ID_REL)) => {
            e.cap(Capability::DrawParameters);
            let ti = e.t_i32();
            let load = |e: &mut Emitter, b: BuiltIn| {
                let var = e.builtin_input(b, ti);
                e.hoist(S::Load, ti, &[var])
            };
            let r = match c {
                custom::BASE_VERTEX => load(e, BuiltIn::BaseVertex),
                custom::BASE_INSTANCE => load(e, BuiltIn::BaseInstance),
                custom::VERTEX_ID_REL => {
                    let (v, b) = (load(e, BuiltIn::VertexIndex), load(e, BuiltIn::BaseVertex));
                    e.hoist(S::ISub, ti, &[v, b])
                }
                _ => {
                    let (v, b) = (load(e, BuiltIn::InstanceIndex), load(e, BuiltIn::BaseInstance));
                    e.hoist(S::ISub, ti, &[v, b])
                }
            };
            e.hoist(S::Bitcast, tu, &[r])
        }
        SystemValue::Custom(custom::THREAD_GROUP_SIZE) => {
            // Wave index in bits 6..11? The hardware word: wave count in
            // bits 0..5, this wave's index from bit 6 (the first wave flag
            // in bit 31).
            let total = size[0] * size[1] * size[2];
            let wave = e.f.wave_size;
            let waves = total.div_ceil(wave).min(0x3F);
            let i = guest_index(e);
            let w = e.cu(wave);
            let id = e.hoist(S::UDiv, tu, &[i, w]);
            let six = e.cu(6);
            let bits = e.hoist(S::ShiftLeftLogical, tu, &[id, six]);
            let n = e.cu(waves);
            e.hoist(S::BitwiseOr, tu, &[bits, n])
        }
        SystemValue::Custom(custom::MERGED_WAVE_INFO) => {
            // One wave per subgroup: vertex and primitive counts in bits
            // 0..7 and 8..15 (the active lanes this host subgroup holds,
            // one vertex per invocation), wave index 0, one wave (bit 28).
            let count = vertex_wave_count(e);
            let eight = e.cu(8);
            let prims = e.hoist(S::ShiftLeftLogical, tu, &[count, eight]);
            let counts = e.hoist(S::BitwiseOr, tu, &[count, prims]);
            if !super::vertex_wave_info_enabled() {
                return Ok(counts);
            }
            let one_wave = e.cu(1 << 28);
            e.hoist(S::BitwiseOr, tu, &[counts, one_wave])
        }
        SystemValue::Custom(custom::SUBGROUP_INFO) => {
            // The subgroup's vertex (bits 12..20) and primitive (22..30)
            // counts: those of its one wave.
            let count = vertex_wave_count(e);
            let c12 = e.cu(12);
            let c22 = e.cu(22);
            let v = e.hoist(S::ShiftLeftLogical, tu, &[count, c12]);
            let p = e.hoist(S::ShiftLeftLogical, tu, &[count, c22]);
            e.hoist(S::BitwiseOr, tu, &[v, p])
        }
        SystemValue::Custom(custom::POSITION_W) => {
            let v4 = e.t_fvec(4);
            let v = e.builtin_value(BuiltIn::FragCoord, v4);
            let tf = e.t_f32();
            let w = e.hoist_extract(tf, v, 3);
            let one = e.cf(1.0);
            let r = e.hoist(S::FDiv, tf, &[one, w]);
            e.hoist(S::Bitcast, tu, &[r])
        }
        SystemValue::Custom(custom::POSITION_FIXED) => {
            let v4 = e.t_fvec(4);
            let v = e.builtin_value(BuiltIn::FragCoord, v4);
            let tf = e.t_f32();
            let x = e.hoist_extract(tf, v, 0);
            let y = e.hoist_extract(tf, v, 1);
            let xi = e.hoist(S::ConvertFToU, tu, &[x]);
            let yi = e.hoist(S::ConvertFToU, tu, &[y]);
            let c16 = e.cu(16);
            let ys = e.hoist(S::ShiftLeftLogical, tu, &[yi, c16]);
            let m = e.cu(0xFFFF);
            let xm = e.hoist(S::BitwiseAnd, tu, &[xi, m]);
            e.hoist(S::BitwiseOr, tu, &[xm, ys])
        }
        SystemValue::Custom(n) if e.mesh.is_some() && (custom::MESH_INPUT..custom::MESH_INPUT + 7).contains(&n) => {
            super::mesh::input(e, (n - custom::MESH_INPUT) as usize)?
        }
        SystemValue::Custom(n) => return Err(format!("system value Custom({n})")),
    })
}

/// Vertices of a vertex-path wave: the active invocations of the host
/// subgroup (both ballot words of a wave64 program, whose lanes 32..63 sit
/// in the second word on a 64-wide host) with `RPS5E_NGG_VERTEX_WAVE_INFO=1`;
/// by default the highest active lane of the first word plus one.
fn vertex_wave_count(e: &mut Emitter) -> Word {
    let tu = e.t_u32();
    let t = e.cb(true);
    e.cap(Capability::GroupNonUniformBallot);
    e.cap(Capability::GroupNonUniform);
    let t4 = e.t_uvec(4);
    let scope = e.cu(super::emit::SCOPE_SUBGROUP);
    let m = e.hoist(S::GroupNonUniformBallot, t4, &[scope, t]);
    let lo = e.hoist_extract(tu, m, 0);
    if !super::vertex_wave_info_enabled() {
        let msb = e.in_prologue(|e| e.gl(tu, GLOp::FindUMsb, &[lo]));
        let one = e.cu(1);
        return e.hoist(S::IAdd, tu, &[msb, one]);
    }
    let count = e.hoist(S::BitCount, tu, &[lo]);
    if e.f.wave_size != 64 {
        return count;
    }
    let hi = e.hoist_extract(tu, m, 1);
    let upper = e.hoist(S::BitCount, tu, &[hi]);
    e.hoist(S::IAdd, tu, &[count, upper])
}

/// The guest local invocation index of this lane half: host invocation
/// `j` of a paired (two lanes per invocation) wave64 program holds guest
/// threads `64 * (j / 32) + j % 32` and that plus 32.
pub fn guest_index(e: &mut Emitter) -> Word {
    let tu = e.t_u32();
    let j = e.builtin_value(BuiltIn::LocalInvocationIndex, tu);
    if e.lanes != 2 {
        return j;
    }
    let c31 = e.cu(31);
    let lane = e.hoist(S::BitwiseAnd, tu, &[j, c31]);
    let wave_bits = e.cu(!31);
    let wave = e.hoist(S::BitwiseAnd, tu, &[j, wave_bits]);
    let one = e.cu(1);
    let base = e.hoist(S::ShiftLeftLogical, tu, &[wave, one]);
    let g = e.hoist(S::BitwiseOr, tu, &[base, lane]);
    if e.half == 1 {
        let c32 = e.cu(32);
        e.hoist(S::IAdd, tu, &[g, c32])
    } else {
        g
    }
}

/// Component `c` of the local id of guest index `g`.
fn local_component(e: &mut Emitter, g: Word, size: [u32; 3], c: u8) -> Word {
    let tu = e.t_u32();
    let below: u32 = size[..c as usize].iter().product();
    let d = e.cu(below);
    let q = e.hoist(S::UDiv, tu, &[g, d]);
    if c == 2 {
        return q;
    }
    let n = e.cu(size[c as usize]);
    e.hoist(S::UMod, tu, &[q, n])
}

/// Barycentric component `n` (perspective i/j = 0/1, linear 2/3,
/// perspective centroid 4/5, linear centroid 6/7) as f32.
fn barycentric(e: &mut Emitter, n: u8) -> R<Word> {
    if !e.feat.barycentric {
        return Err("pixel barycentrics need VK_KHR_fragment_shader_barycentric".into());
    }
    e.cap(Capability::FragmentBarycentricKHR);
    e.exts.insert("SPV_KHR_fragment_shader_barycentric");
    let linear = matches!(n, 2 | 3 | 6 | 7);
    let centroid = n >= 4;
    let b = if linear { BuiltIn::BaryCoordNoPerspKHR } else { BuiltIn::BaryCoordKHR };
    let v3 = e.t_fvec(3);
    let var = match e.io.barycentrics.get(&(b as u32, centroid)) {
        Some(&v) => v,
        None => {
            let v = e.global(v3, SC::Input);
            e.decorate(v, Decoration::BuiltIn, &[b as u32]);
            if centroid {
                e.decorate(v, Decoration::Centroid, &[]);
            }
            e.io.barycentrics.insert((b as u32, centroid), v);
            v
        }
    };
    let x = e.hoist(S::Load, v3, &[var]);
    let tf = e.t_f32();
    // i weights vertex 1, j vertex 2.
    Ok(e.hoist_extract(tf, x, 1 + (n as u32 & 1)))
}

/// SPI_PS_INPUT_CNTL facts of attribute `attr`.
struct AttributeInfo {
    location: u32,
    /// DEFAULT_VAL when the attribute reads a constant (OFFSET bit 5
    /// without FLAT_SHADE).
    default: Option<u32>,
    /// OFFSET bit 5 with FLAT_SHADE: the three vertices pass through
    /// unchanged (`v_interp_mov` P10/P20 read vertices 1/2 themselves, the
    /// program interpolates on its own, e.g. packed 16-bit colours).
    pass_through: bool,
}

fn attribute_info(e: &Emitter, attr: u8) -> AttributeInfo {
    let cntl = e.input.request.pixel_input_cntl.as_ref().and_then(|c| c.get(attr as usize).copied()).unwrap_or(attr as u32);
    decode_input_cntl(cntl)
}

fn decode_input_cntl(cntl: u32) -> AttributeInfo {
    let flat = cntl & 0x400 != 0;
    let bit5 = cntl & 0x20 != 0;
    AttributeInfo { location: cntl & 0x1F, default: (bit5 && !flat).then_some((cntl >> 8) & 3), pass_through: bit5 && flat }
}

fn attribute_var(e: &mut Emitter, location: u32) -> Word {
    if let Some(&v) = e.io.attributes.get(&location) {
        return v;
    }
    let v4 = e.t_fvec(4);
    let three = e.cu(3);
    let arr = e.b.type_array(v4, three);
    let var = e.global(arr, SC::Input);
    e.decorate(var, Decoration::Location, &[location]);
    e.decorate(var, Decoration::PerVertexKHR, &[]);
    e.io.attributes.insert(location, var);
    var
}

/// Interpolation parameter `param` of attribute `attr` channel `chan`.
pub fn interp_param(e: &mut Emitter, attr: u8, chan: u8, param: InterpParam) -> R<Word> {
    if let InterpParam::HalfP0 { high } | InterpParam::HalfP10 { high } | InterpParam::HalfP20 { high } = param {
        return interp_param_half(e, attr, chan, param, high);
    }
    let info = attribute_info(e, attr);
    let tf = e.t_f32();
    if let Some(d) = info.default {
        let value = match (d, chan) {
            (0, _) => 0.0,
            (1, 3) => 1.0,
            (1, _) => 0.0,
            (2, 3) => 0.0,
            (2, _) => 1.0,
            _ => 1.0,
        };
        return Ok(match param {
            InterpParam::P0 => e.cf(value),
            _ => e.cf(0.0),
        });
    }
    if !e.feat.barycentric {
        return Err("pixel attributes need VK_KHR_fragment_shader_barycentric".into());
    }
    e.cap(Capability::FragmentBarycentricKHR);
    e.exts.insert("SPV_KHR_fragment_shader_barycentric");
    let var = attribute_var(e, info.location);
    let p = e.ptr(SC::Input, tf);
    let ch = e.cu(chan as u32 & 3);
    let vertex = |e: &mut Emitter, i: u32| {
        let idx = e.cu(i);
        let a = e.hoist(S::AccessChain, p, &[var, idx, ch]);
        e.hoist(S::Load, tf, &[a])
    };
    Ok(match param {
        InterpParam::P0 => vertex(e, 0),
        InterpParam::P10 if info.pass_through => vertex(e, 1),
        InterpParam::P20 if info.pass_through => vertex(e, 2),
        InterpParam::HalfP0 { .. } | InterpParam::HalfP10 { .. } | InterpParam::HalfP20 { .. } => unreachable!("16-bit parameters are read by interp_param_half"),
        InterpParam::P10 | InterpParam::P20 => {
            let k = if param == InterpParam::P10 { 1 } else { 2 };
            let (pk, p0) = (vertex(e, k), vertex(e, 0));
            let d = e.hoist(S::FSub, tf, &[pk, p0]);
            e.decorate(d, Decoration::NoContraction, &[]);
            d
        }
    })
}

/// A 16-bit interpolation parameter (FP16_INTERP_MODE): the f16 half of
/// each vertex's attribute channel; the differences with vertex 0 are
/// rounded to f16. The high half reads its default with USE_DEFAULT_ATTR1
/// (SPI_PS_INPUT_CNTL bit 20, DEFAULT_VAL_ATTR1 in bits 22:21).
fn interp_param_half(e: &mut Emitter, attr: u8, chan: u8, param: InterpParam, high: bool) -> R<Word> {
    let cntl = e.input.request.pixel_input_cntl.as_ref().and_then(|c| c.get(attr as usize).copied()).unwrap_or(attr as u32);
    let info = decode_input_cntl(cntl);
    let default = if high { (cntl & (1 << 20) != 0).then_some((cntl >> 21) & 3) } else { info.default };
    let tf = e.t_f32();
    if let Some(d) = default {
        let value = match (d, chan) {
            (0, _) => 0.0,
            (1, 3) => 1.0,
            (1, _) => 0.0,
            (2, 3) => 0.0,
            (2, _) => 1.0,
            _ => 1.0,
        };
        return Ok(match param {
            InterpParam::HalfP0 { .. } => e.cf(value),
            _ => e.cf(0.0),
        });
    }
    if !e.feat.barycentric {
        return Err("16-bit pixel attributes need VK_KHR_fragment_shader_barycentric".into());
    }
    e.cap(Capability::FragmentBarycentricKHR);
    e.exts.insert("SPV_KHR_fragment_shader_barycentric");
    let var = attribute_var(e, info.location);
    let p = e.ptr(SC::Input, tf);
    let ch = e.cu(chan as u32 & 3);
    let tu = e.t_u32();
    let vertex = |e: &mut Emitter, i: u32| {
        let idx = e.cu(i);
        let a = e.hoist(S::AccessChain, p, &[var, idx, ch]);
        let v = e.hoist(S::Load, tf, &[a]);
        let bits = e.hoist(S::Bitcast, tu, &[v]);
        let bits = if high {
            let c16 = e.cu(16);
            e.hoist(S::ShiftRightLogical, tu, &[bits, c16])
        } else {
            bits
        };
        let f = e.f16_bits_to_f32(bits);
        e.bitcast(tf, f)
    };
    Ok(match param {
        InterpParam::HalfP0 { .. } => vertex(e, 0),
        _ => {
            let k = if matches!(param, InterpParam::HalfP10 { .. }) { 1 } else { 2 };
            let (pk, p0) = (vertex(e, k), vertex(e, 0));
            let d = e.ins(S::FSub, tf, &[pk, p0]);
            e.decorate(d, Decoration::NoContraction, &[]);
            let h = e.f32_to_f16_bits_rte(d);
            let f = e.f16_bits_to_f32(h);
            e.bitcast(tf, f)
        }
    })
}

/// Kill the invocation (demote to a helper when supported).
fn discard(e: &mut Emitter) {
    if e.feat.demote {
        e.cap(Capability::DemoteToHelperInvocation);
        e.exts.insert("SPV_EXT_demote_to_helper_invocation");
        e.void(S::DemoteToHelperInvocation, vec![]);
    } else {
        // OpKill is a terminator: wrap it so the block continues.
        let (kill, cont) = (e.b.id(), e.b.id());
        let t = e.cb(true);
        e.b.selection_merge(cont, rspirv::spirv::SelectionControl::NONE).unwrap();
        e.b.branch_conditional(t, kill, cont, []).unwrap();
        e.begin(kill);
        e.b.kill().unwrap();
        e.begin(cont);
    }
}

/// `exp` instruction.
pub fn export(e: &mut Emitter, v: Value, a: ExportAttr) -> R<()> {
    let n = e.f.args(v).len();
    let exec = e.arg(v, n - 1)?;
    let data: Vec<Word> = (0..n - 1).map(|i| e.arg(v, i)).collect::<R<_>>()?;
    let tu = e.t_u32();
    let tf = e.t_f32();
    let target = a.target as u32;
    match e.stage {
        ShaderStage::Vertex if e.mesh.is_some() => super::mesh::export(e, target, a.enable, &data, exec)?,
        ShaderStage::Vertex => {
            if target == 12 {
                let block = per_vertex(e);
                let v4 = e.t_fvec(4);
                let mut fl: Vec<Word> = (0..4).map(|i| if i < data.len() { e.bitcast(tf, data[i]) } else { e.cf(0.0) }).collect();
                clip_space(e, &mut fl);
                let pos = e.construct(v4, &fl);
                let p = e.ptr(SC::Output, v4);
                let zero = e.cu(0);
                let ptr = e.access(p, block, &[zero]);
                let clip = zero_position_clip(e).map(|(var, index)| {
                    let p = e.ptr(SC::Output, tf);
                    let i = e.cu(index);
                    (e.access(p, var, &[i]), zero_position_distance(e, &fl))
                });
                let user = user_clip_distances(e, pos);
                e.if_then(exec, |e| {
                    e.store(ptr, pos);
                    if let Some((p, d)) = clip {
                        e.store(p, d);
                    }
                    for (p, d) in user {
                        e.store(p, d);
                    }
                });
            } else if (32..64).contains(&target) {
                let var = param_output(e, target - 32);
                let p = e.ptr(SC::Output, tf);
                let comps: Vec<(Word, Word)> = (0..4u32)
                    .filter(|i| a.enable >> i & 1 != 0 && (*i as usize) < data.len())
                    .map(|i| {
                        let idx = e.cu(i);
                        let ptr = e.access(p, var, &[idx]);
                        let val = e.bitcast(tf, data[i as usize]);
                        (ptr, val)
                    })
                    .collect();
                e.if_then(exec, |e| {
                    for (ptr, val) in comps {
                        e.store(ptr, val);
                    }
                });
            }
            else if (13..=16).contains(&target) {
                misc_export(e, target - 12, a.enable, &data, exec);
            }
            // Primitive exports (20) are handled by the host pipeline.
        }
        ShaderStage::Pixel => {
            if target < 8 {
                pixel_color(e, target, &a, &data, exec)?;
            } else if target == 8 && a.enable & 1 != 0 && !data.is_empty() && e.input.request.pixel_depth_export {
                let var = match e.io.frag_depth {
                    Some(v) => v,
                    None => {
                        let v = e.global(tf, SC::Output);
                        e.decorate(v, Decoration::BuiltIn, &[BuiltIn::FragDepth as u32]);
                        e.io.frag_depth = Some(v);
                        v
                    }
                };
                let d = e.bitcast(tf, data[0]);
                e.if_then(exec, |e| e.store(var, d));
            }
            let op_export = e.input.request.pixel_stencil_op_export;
            if target == 8 && a.enable & 2 != 0 && data.len() > 1 && (e.input.request.pixel_stencil_export || op_export) {
                let ti = e.t_i32();
                let var = match e.io.frag_stencil {
                    Some(v) => v,
                    None => {
                        e.cap(rspirv::spirv::Capability::StencilExportEXT);
                        e.exts.insert("SPV_EXT_shader_stencil_export");
                        let v = e.global(ti, SC::Output);
                        e.decorate(v, Decoration::BuiltIn, &[BuiltIn::FragStencilRefEXT as u32]);
                        e.io.frag_stencil = Some(v);
                        v
                    }
                };
                // The operation value sits in bits 8..15 (the test value in
                // the low bits).
                let raw = if op_export { e.bits(data[1], 8, 8) } else { data[1] };
                let s = e.bitcast(ti, raw);
                e.if_then(exec, |e| e.store(var, s));
            }
            if a.valid_mask {
                let dead = e.lnot(exec);
                e.if_then(dead, discard);
            }
        }
        _ => {}
    }
    let _ = tu;
    Ok(())
}

/// With clipping disabled (PA_CL_CLIP_CNTL.CLIP_DISABLE) the host still
/// clips: the position is taken to window space by the guest viewport and
/// back to clip space of a host viewport covering the largest render area
/// (`(p * scale + offset) / half_extent - 1`), so nothing within it clips.
pub fn clip_space(e: &mut Emitter, pos: &mut [Word]) {
    let t = e.input.request.clip_space;
    if !t.enabled {
        return;
    }
    let tf = e.t_f32();
    for (k, (scale, offset, half)) in [(t.scale_x, t.offset_x, t.half_extent_x), (t.scale_y, t.offset_y, t.half_extent_y)].into_iter().enumerate() {
        let (s, o, h, one) = (e.cf(scale), e.cf(offset), e.cf(half), e.cf(1.0));
        let w = e.ins(S::FMul, tf, &[pos[k], s]);
        let b = e.ins(S::FAdd, tf, &[w, o]);
        let d = e.ins(S::FDiv, tf, &[b, h]);
        pos[k] = e.ins(S::FSub, tf, &[d, one]);
    }
}

/// The hardware clips a primitive with a vertex whose clip position is all
/// zeros (both signs), as it does one fetched past its vertex buffer; a
/// host draws such a vertex at w = 0 (an edge to infinity). A clip distance
/// of -1 there and 0 elsewhere clips the whole primitive. The value of
/// `pos` (the four clip-space components as f32).
pub fn zero_position_distance(e: &mut Emitter, pos: &[Word]) -> Word {
    let (tu, tf) = (e.t_u32(), e.t_f32());
    let mm = e.cu(0x7FFF_FFFF);
    let mut any = e.cu(0);
    for &c in pos {
        let bits = e.bitcast(tu, c);
        let mag = e.band(bits, mm);
        any = e.bor(any, mag);
    }
    let zero = e.cu(0);
    let all_zero = e.ieq(any, zero);
    let (neg, none) = (e.cf(-1.0), e.cf(0.0));
    e.select(tf, all_zero, neg, none)
}

/// The guest clip distances PA_CL_VS_OUT_CNTL enables (vertex path).
fn guest_clip_distances(e: &Emitter) -> u32 {
    if e.input.request.supports_clip_distance {
        (e.input.request.position_export_control & 0xFF).count_ones()
    } else {
        0
    }
}

/// The guest's cull distances (PA_CL_VS_OUT_CNTL CULL_DIST_ENA).
fn guest_cull_distances(e: &Emitter) -> u32 {
    if e.input.request.supports_clip_distance {
        ((e.input.request.position_export_control >> 8) & 0xFF).count_ones()
    } else {
        0
    }
}

/// The user clip planes the vertex path writes (PA_CL_CLIP_CNTL.UCP_ENA),
/// as clip distances after the guest's own.
fn user_clip_distance_count(e: &Emitter) -> u32 {
    if e.input.request.supports_clip_distance {
        e.input.request.user_clip_planes.count()
    } else {
        0
    }
}

/// The clip distances of the user clip planes at `pos` (the clip-space
/// position): the clipper keeps the half space where dot(plane, position)
/// >= 0 for each enabled plane. (pointer, value) pairs to store.
fn user_clip_distances(e: &mut Emitter, pos: Word) -> Vec<(Word, Word)> {
    if user_clip_distance_count(e) == 0 {
        return Vec::new();
    }
    let ucp = e.input.request.user_clip_planes;
    let guest = guest_clip_distances(e);
    let var = clip_distance_output(e);
    let (tf, v4) = (e.t_f32(), e.t_fvec(4));
    let p = e.ptr(SC::Output, tf);
    let mut out = Vec::new();
    for (k, plane) in (0..6u32).filter(|i| ucp.mask >> i & 1 != 0).enumerate() {
        let c: Vec<Word> = (0..4).map(|i| e.cf(f32::from_bits(ucp.planes[(plane * 4 + i) as usize]))).collect();
        let plane = e.construct(v4, &c);
        let d = e.ins(S::Dot, tf, &[plane, pos]);
        let i = e.cu(guest + k as u32);
        out.push((e.access(p, var, &[i]), d));
    }
    out
}

/// Whether the all-zero position distance fits beside the guest's clip and
/// cull distances and the user clip planes (eight in all).
fn zero_position_clip_fits(e: &Emitter) -> bool {
    zero_position_clip_enabled() && guest_clip_distances(e) + guest_cull_distances(e) + user_clip_distance_count(e) < 8
}

/// The vertex path's gl_ClipDistance array and the index of the all-zero
/// position distance after the guest's ones (none without clip distances
/// or when the guest's clip and cull distances take all eight).
fn zero_position_clip(e: &mut Emitter) -> Option<(Word, u32)> {
    if !e.input.request.supports_clip_distance || !zero_position_clip_fits(e) {
        return None;
    }
    let index = guest_clip_distances(e) + user_clip_distance_count(e);
    Some((clip_distance_output(e), index))
}

/// The length of the vertex path's gl_ClipDistance output (0 when the
/// program declares none): what later stages must forward.
pub fn clip_distance_count(e: &Emitter) -> u32 {
    if !e.io.misc.contains_key(&(BuiltIn::ClipDistance as u32)) {
        return 0;
    }
    guest_clip_distances(e) + user_clip_distance_count(e) + zero_position_clip_fits(e) as u32
}

/// gl_ClipDistance of the vertex path: the guest's distances, the user
/// clip planes, then the all-zero position distance.
fn clip_distance_output(e: &mut Emitter) -> Word {
    e.cap(Capability::ClipDistance);
    let guest = guest_clip_distances(e);
    let n = guest + user_clip_distance_count(e) + zero_position_clip_fits(e) as u32;
    let tf = e.t_f32();
    let n = e.cu(n);
    let arr = e.b.type_array(tf, n);
    misc_output(e, BuiltIn::ClipDistance, arr)
}

/// `RPS5E_ZERO_POSITION_CLIP=0` leaves all-zero positions to the host.
pub fn zero_position_clip_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_ZERO_POSITION_CLIP").is_ok_and(|v| v == "0"))
}

/// A vertex built-in output (created once).
fn misc_output(e: &mut Emitter, builtin: BuiltIn, ty: Word) -> Word {
    if let Some(&v) = e.io.misc.get(&(builtin as u32)) {
        return v;
    }
    let v = e.global(ty, SC::Output);
    e.decorate(v, Decoration::BuiltIn, &[builtin as u32]);
    e.io.misc.insert(builtin as u32, v);
    v
}

/// POS1..POS4 (`pos_index`): point size, render-target layer, viewport
/// index and clip / cull distances as PA_CL_VS_OUT_CNTL assigns them.
fn misc_export(e: &mut Emitter, pos_index: u32, enable: u8, data: &[Word], exec: Word) {
    let control = e.input.request.position_export_control;
    let tu = e.t_u32();
    let tf = e.t_f32();
    let mut stores: Vec<(Word, Word)> = Vec::new();
    for c in 0..4u32 {
        if enable >> c & 1 == 0 || c as usize >= data.len() {
            continue;
        }
        let raw = data[c as usize];
        let out = ps5shader::resources::decode_position_export_component(control, pos_index, c);
        if out.layer {
            e.cap(Capability::ShaderLayer);
            let var = misc_output(e, BuiltIn::Layer, tu);
            let m = e.cu(0x7FF);
            stores.push((var, e.band(raw, m)));
        }
        if out.viewport {
            // MISC.z packs the viewport index in bits 16..19 next to the layer.
            e.cap(Capability::ShaderViewportIndex);
            let var = misc_output(e, BuiltIn::ViewportIndex, tu);
            stores.push((var, e.bits(raw, 16, 4)));
        }
        if out.point_size {
            let var = misc_output(e, BuiltIn::PointSize, tf);
            stores.push((var, e.bitcast(tf, raw)));
        }
        if !e.input.request.supports_clip_distance {
            continue;
        }
        for (index, builtin, mask, cap) in [
            (out.clip_distance, BuiltIn::ClipDistance, control & 0xFF, Capability::ClipDistance),
            (out.cull_distance, BuiltIn::CullDistance, (control >> 8) & 0xFF, Capability::CullDistance),
        ] {
            let Some(index) = index else { continue };
            e.cap(cap);
            let var = if builtin == BuiltIn::ClipDistance {
                clip_distance_output(e)
            } else {
                let n = e.cu(mask.count_ones());
                let arr = e.b.type_array(tf, n);
                misc_output(e, builtin, arr)
            };
            let p = e.ptr(SC::Output, tf);
            let i = e.cu(index);
            let ptr = e.access(p, var, &[i]);
            stores.push((ptr, e.bitcast(tf, raw)));
        }
    }
    if !stores.is_empty() {
        e.if_then(exec, |e| {
            for (p, v) in stores {
                e.store(p, v);
            }
        });
    }
}

/// The second blend source of target 0 (Location 0, Index 1).
fn second_source(e: &mut Emitter, kind: PixelOutputKind) -> Word {
    if let Some((v, _)) = e.io.second_source {
        return v;
    }
    let t = match kind {
        PixelOutputKind::Float => e.t_fvec(4),
        PixelOutputKind::Uint => e.t_uvec(4),
        PixelOutputKind::Sint => e.t_ivec(4),
    };
    let var = e.global(t, SC::Output);
    e.decorate(var, Decoration::Location, &[0]);
    e.decorate(var, Decoration::Index, &[1]);
    e.io.second_source = Some((var, kind));
    var
}

fn pixel_color(e: &mut Emitter, target: u32, a: &ExportAttr, data: &[Word], exec: Word) -> R<()> {
    use ps5shader::resources::PixelAlphaBlendSource as A;
    let alpha_source = e.input.request.pixel_alpha_blend_source;
    let dual = e.input.request.pixel_dual_source_blend;
    // With a logical-alpha second source only target 0 is exported.
    if alpha_source != A::None && target != 0 {
        return Ok(());
    }
    // Dual-source blending: MRT1 is target 0's second source, in its
    // format and mapping.
    let second = dual && target == 1;
    let outputs: Vec<_> = if second {
        e.input.request.pixel_outputs.iter().filter(|o| o.export_target() == 0).take(1).copied().collect()
    } else {
        e.input.request.pixel_outputs.iter().filter(|o| o.export_target() == target).copied().collect()
    };
    let tu = e.t_u32();
    let fmt = (e.input.request.color_export_format >> (target * 4)) & 0xF;
    let coverage = target == 0 && !second && e.input.request.pixel_alpha_to_coverage && matches!(fmt, 4 | 5 | 6 | 9) && outputs.iter().all(|o| o.kind == PixelOutputKind::Float);
    if outputs.is_empty() && !coverage {
        return Ok(());
    }
    // Components the export actually carries: the `exp` enable mask, narrowed
    // by SPI_SHADER_COL_FORMAT for the 32-bit formats (32_R, 32_GR, 32_AR).
    // The others read as 0, alpha as 1. On GFX10 a 32_AR export carries red
    // in channel 0 and alpha in channel 1 (before GFX10: channel 3).
    let exported = if a.compressed {
        a.enable as u32 & 0xF
    } else {
        a.enable as u32
            & match fmt {
                1 => 0x1,
                2 | 3 => 0x3,
                _ => 0xF,
            }
    };
    // The four exported components as u32 bit patterns (None = not exported).
    let comps: Vec<Option<Word>> = if a.compressed {
        let mut out = Vec::new();
        for k in 0..2u32 {
            let w = data.get(k as usize).copied().unwrap_or_else(|| e.cu(0));
            for half in 0..2u32 {
                let c = k * 2 + half;
                out.push((exported >> c & 1 != 0).then(|| unpack16(e, w, half, fmt)));
            }
        }
        out
    } else {
        let mut out: Vec<Option<Word>> = (0..4u32)
            .map(|i| (exported >> i & 1 != 0).then(|| data.get(i as usize).copied().unwrap_or_else(|| e.cu(0))))
            .collect();
        if fmt == 3 {
            out[3] = out[1].take();
        }
        out
    };
    if coverage {
        // Alpha-to-coverage: the MRT0 alpha (an unexported alpha reads 1)
        // keeps every sample from one half up and none below it or NaN.
        let alpha = comps[3].unwrap_or_else(|| e.cu(0x3f80_0000));
        alpha_coverage(e, alpha, exec);
    }
    for o in outputs {
        let var = if second { second_source(e, o.kind) } else { color_output(e, o.host_location, o.kind) };
        let one = if o.kind == PixelOutputKind::Float { 0x3f80_0000 } else { 1 };
        // A 16-bit float or normalized export (FP16, UNORM16, SNORM16) into
        // an integer target: the CB stores each exported 16-bit field as it
        // is, without converting its value.
        let raw_fields = a.compressed && matches!(fmt, 4..=6) && o.kind != PixelOutputKind::Float;
        let comps: Vec<Option<Word>> = if raw_fields {
            (0..4u32)
                .map(|c| {
                    (exported >> c & 1 != 0).then(|| {
                        let w = data.get((c / 2) as usize).copied().unwrap_or_else(|| e.cu(0));
                        let (off, n) = (e.cu(c % 2 * 16), e.cu(16));
                        let op = if o.kind == PixelOutputKind::Sint { S::BitFieldSExtract } else { S::BitFieldUExtract };
                        e.ins(op, tu, &[w, off, n])
                    })
                })
                .collect()
        } else {
            comps.clone()
        };
        let comps: Vec<Word> = comps
            .iter()
            .enumerate()
            .map(|(i, c)| c.unwrap_or_else(|| e.cu(if i == 3 { one } else { 0 })))
            .collect();
        let (st, vt) = match o.kind {
            PixelOutputKind::Float => (e.t_f32(), e.t_fvec(4)),
            PixelOutputKind::Uint => (tu, e.t_uvec(4)),
            PixelOutputKind::Sint => (e.t_i32(), e.t_ivec(4)),
        };
        let mapped: Vec<Word> = (0..4u32)
            .map(|p| {
                let c = comps[o.component_mapping.map(p) as usize];
                if o.kind == PixelOutputKind::Uint {
                    c
                } else {
                    e.bitcast(st, c)
                }
            })
            .collect();
        let value = match o.packed_unorm_11_11_10 {
            Some(truncate) if o.kind == PixelOutputKind::Uint => pack_unorm_11_11_10(e, &mapped, truncate),
            _ if o.packed_unorm_4_4_4_4 && o.kind == PixelOutputKind::Uint => pack_unorm_4_4_4_4(e, &mapped),
            _ => e.construct(vt, &mapped),
        };
        // The logical source alpha (before the mapping) as target 0's
        // second source: in every channel, or with one / zero in the
        // physical alpha channel.
        let alpha = if alpha_source != A::None && target == 0 && o.kind == PixelOutputKind::Float {
            let a = e.bitcast(st, comps[3]);
            let factor = match alpha_source {
                A::SourceAlphaOne => Some(e.cf(1.0)),
                A::SourceAlphaZero => Some(e.cf(0.0)),
                _ => None,
            };
            let lanes: Vec<Word> = (0..4u32).map(|p| if o.component_mapping.map(p) == 3 { factor.unwrap_or(a) } else { a }).collect();
            let lanes = if factor.is_some() {
                // Only the first channel holding alpha takes the factor.
                let first = (0..4u32).find(|&p| o.component_mapping.map(p) == 3);
                (0..4u32).map(|p| if Some(p) == first { lanes[p as usize] } else { a }).collect::<Vec<_>>()
            } else {
                lanes
            };
            let v = e.construct(vt, &lanes);
            Some((second_source(e, o.kind), v))
        } else {
            None
        };
        e.if_then(exec, |e| {
            e.store(var, value);
            if let Some((v2, a)) = alpha {
                e.store(v2, a);
            }
        });
    }
    Ok(())
}

/// An 11_11_10 UNORM texel from the first three physical components (f32
/// bits): each clamped to [0, 1], scaled to 2047 / 2047 / 1023 and rounded
/// half up (or truncated), packed with the first field lowest. The packed
/// word goes to the first channel of a 32-bit unsigned output.
fn pack_unorm_11_11_10(e: &mut Emitter, physical: &[Word], truncate: bool) -> Word {
    let tf = e.t_f32();
    let tu = e.t_u32();
    let (zero, one) = (e.cf(0.0), e.cf(1.0));
    let mut packed = e.cu(0);
    for (i, (bits, shift)) in [(11u32, 0u32), (11, 11), (10, 22)].into_iter().enumerate() {
        let f = e.bitcast(tf, physical[i]);
        // NaN reads as zero (NMin / NMax return the other operand).
        let c = e.gl(tf, rspirv::spirv::GlslStd450Op::NMax, &[f, zero]);
        let c = e.gl(tf, rspirv::spirv::GlslStd450Op::NMin, &[c, one]);
        let max = e.cf(((1u32 << bits) - 1) as f32);
        let scaled = e.ins(S::FMul, tf, &[c, max]);
        let rounded = if truncate {
            scaled
        } else {
            let half = e.cf(0.5);
            e.ins(S::FAdd, tf, &[scaled, half])
        };
        let field = e.ins(S::ConvertFToU, tu, &[rounded]);
        let field = if shift == 0 {
            field
        } else {
            let s = e.cu(shift);
            e.ins(S::ShiftLeftLogical, tu, &[field, s])
        };
        packed = e.ins(S::BitwiseOr, tu, &[packed, field]);
    }
    let z = e.cu(0);
    let vt = e.t_uvec(4);
    e.construct(vt, &[packed, z, z, z])
}

/// A 4_4_4_4 UNORM texel from the four physical components (f32 bits) the
/// packed-16 host format's R, G, B and A channels hold: each clamped to
/// [0, 1], scaled to 15 and rounded half up (NaN reads as zero), the A
/// channel's field lowest — X in the lowest field, the layout the texture
/// path's packed-channel reads expect. The packed word goes to the first
/// channel of a 16-bit unsigned output.
fn pack_unorm_4_4_4_4(e: &mut Emitter, physical: &[Word]) -> Word {
    let tf = e.t_f32();
    let tu = e.t_u32();
    let (zero, one) = (e.cf(0.0), e.cf(1.0));
    let max = e.cf(15.0);
    let half = e.cf(0.5);
    let mut packed = e.cu(0);
    // The packed-16 format's R, G, B and A occupy fields 3, 2, 1 and 0.
    for (p, shift) in [(0usize, 12u32), (1, 8), (2, 4), (3, 0)] {
        let f = e.bitcast(tf, physical[p]);
        // NaN reads as zero (NMin / NMax return the other operand).
        let c = e.gl(tf, rspirv::spirv::GlslStd450Op::NMax, &[f, zero]);
        let c = e.gl(tf, rspirv::spirv::GlslStd450Op::NMin, &[c, one]);
        let scaled = e.ins(S::FMul, tf, &[c, max]);
        let rounded = e.ins(S::FAdd, tf, &[scaled, half]);
        let field = e.ins(S::ConvertFToU, tu, &[rounded]);
        let field = if shift == 0 {
            field
        } else {
            let s = e.cu(shift);
            e.ins(S::ShiftLeftLogical, tu, &[field, s])
        };
        packed = e.ins(S::BitwiseOr, tu, &[packed, field]);
    }
    let z = e.cu(0);
    let vt = e.t_uvec(4);
    e.construct(vt, &[packed, z, z, z])
}

/// The pixel's sample mask output (SampleMask[0]) (created once): every
/// sample covered from the start of the program, or the samples the
/// pixel's position in its 2x2 quad keeps (PA_SC_AA_MASK).
fn sample_mask(e: &mut Emitter) -> Word {
    if let Some(v) = e.io.sample_mask {
        return v;
    }
    let ti = e.t_i32();
    let one = e.cu(1);
    let arr = e.b.type_array(ti, one);
    let var = e.global(arr, SC::Output);
    e.decorate(var, Decoration::BuiltIn, &[BuiltIn::SampleMask as u32]);
    e.io.sample_mask = Some(var);
    let p = e.ptr(SC::Output, ti);
    let quad = e.input.request.pixel_quad_sample_masks;
    e.in_prologue(|e| {
        let initial = match quad {
            None => e.ci(-1),
            Some(m) => {
                // Index (x & 1) | (y & 1) << 1 of the pixel's window position.
                let v4 = e.t_fvec(4);
                let tf = e.t_f32();
                let tu = e.t_u32();
                let coord = e.builtin_value(BuiltIn::FragCoord, v4);
                let fx = e.hoist_extract(tf, coord, 0);
                let fy = e.hoist_extract(tf, coord, 1);
                let x = e.ins(S::ConvertFToU, tu, &[fx]);
                let y = e.ins(S::ConvertFToU, tu, &[fy]);
                let bit = e.cu(1);
                let x = e.ins(S::BitwiseAnd, tu, &[x, bit]);
                let y = e.ins(S::BitwiseAnd, tu, &[y, bit]);
                let tb = e.t_bool();
                let zero = e.cu(0);
                let xs = e.ins(S::INotEqual, tb, &[x, zero]);
                let ys = e.ins(S::INotEqual, tb, &[y, zero]);
                let (m0, m1, m2, m3) = if e.input.request.bindings.uses_quad_masks() {
                    // The draw's masks, from the shader data (16 bits each).
                    let at = e.input.request.bindings.quad_masks_dword;
                    let all = e.cu(u32::MAX);
                    let w0 = e.shader_data(at).unwrap_or(all);
                    let w1 = e.shader_data(at + 1).unwrap_or(all);
                    let (lo, sixteen) = (e.cu(0), e.cu(16));
                    let mut field = |w: Word, at: Word| {
                        let v = e.ins(S::BitFieldUExtract, tu, &[w, at, sixteen]);
                        e.bitcast(ti, v)
                    };
                    (field(w0, lo), field(w0, sixteen), field(w1, lo), field(w1, sixteen))
                } else {
                    (e.ci(m[0] as i32), e.ci(m[1] as i32), e.ci(m[2] as i32), e.ci(m[3] as i32))
                };
                let top = e.select(ti, xs, m1, m0);
                let bottom = e.select(ti, xs, m3, m2);
                e.select(ti, ys, bottom, top)
            }
        };
        let zero = e.cu(0);
        let ptr = e.access(p, var, &[zero]);
        e.store(ptr, initial);
    });
    var
}

/// Alpha-to-coverage at the MRT0 export: the sample mask keeps every
/// sample when `alpha` (f32 bits) is at least one half, none when it is
/// below or NaN. The pixel keeps running (a zero mask drops its samples).
fn alpha_coverage(e: &mut Emitter, alpha: Word, exec: Word) {
    let var = sample_mask(e);
    let tf = e.t_f32();
    let ti = e.t_i32();
    let tb = e.t_bool();
    let a = e.bitcast(tf, alpha);
    let half = e.cf(0.5);
    let kept = e.ins(S::FOrdGreaterThanEqual, tb, &[a, half]);
    let p = e.ptr(SC::Output, ti);
    let zero = e.cu(0);
    let ptr = e.access(p, var, &[zero]);
    // The samples the quad mask left (all of them without one).
    let all = if e.input.request.pixel_quad_sample_masks.is_some() { e.ins(S::Load, ti, &[ptr]) } else { e.ci(-1) };
    let none = e.ci(0);
    let mask = e.select(ti, kept, all, none);
    e.if_then(exec, |e| e.store(ptr, mask));
}

/// Half `half` of a compressed export dword as a 32-bit component, by
/// SPI_SHADER_COL_FORMAT (FP16, UNORM16, SNORM16, UINT16, SINT16).
fn unpack16(e: &mut Emitter, w: Word, half: u32, fmt: u32) -> Word {
    let tu = e.t_u32();
    let tf = e.t_f32();
    let (o, n) = (e.cu(half * 16), e.cu(16));
    match fmt {
        5 | 6 => {
            let signed = fmt == 6;
            let field = e.ins(if signed { S::BitFieldSExtract } else { S::BitFieldUExtract }, tu, &[w, o, n]);
            let f = e.ins(if signed { S::ConvertSToF } else { S::ConvertUToF }, tf, &[field]);
            let m = e.cf(if signed { 32767.0 } else { 65535.0 });
            let q = e.ins(S::FDiv, tf, &[f, m]);
            let q = if signed {
                let lo = e.cf(-1.0);
                let hi = e.cf(1.0);
                e.gl(tf, GLOp::FClamp, &[q, lo, hi])
            } else {
                q
            };
            e.bitcast(tu, q)
        }
        7 => e.ins(S::BitFieldUExtract, tu, &[w, o, n]),
        8 => e.ins(S::BitFieldSExtract, tu, &[w, o, n]),
        _ => {
            let v2 = e.t_fvec(2);
            let p = e.gl(v2, GLOp::UnpackHalf2x16, &[w]);
            let f = e.extract(tf, p, half);
            e.bitcast(tu, f)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ancillary_layout() {
        // Guest programs read the layer with v_bfe_u32 v, 16, 11 and the
        // sample with v_bfe_u32 v, 8, 4.
        let bfe = |v: u32, o: u32, n: u32| (v >> o) & ((1 << n) - 1);
        for layer in [0, 1, 31, 2047] {
            assert_eq!(bfe(pack_ancillary(layer, 0), 16, 11), layer);
            assert_eq!(bfe(pack_ancillary(layer, 0), 8, 4), 0);
        }
        assert_eq!(bfe(pack_ancillary(5, 3), 8, 4), 3);
        assert_eq!(bfe(pack_ancillary(0x1FFF, 0), 16, 13), 0x1FFF);
        assert_eq!(pack_ancillary(0x2000, 0), 0);
    }

    #[test]
    fn input_cntl_default_and_pass_through() {
        // OFFSET 3, FLAT_SHADE: an ordinary flat parameter.
        let a = decode_input_cntl(0x403);
        assert_eq!((a.location, a.default, a.pass_through), (3, None, false));
        // OFFSET bit 5 without FLAT_SHADE: DEFAULT_VAL 2 = (1, 1, 1, 0).
        let a = decode_input_cntl(0x220);
        assert_eq!((a.default, a.pass_through), (Some(2), false));
        // OFFSET bit 5 with FLAT_SHADE: vertices pass through at slot 3.
        let a = decode_input_cntl(0x423);
        assert_eq!((a.location, a.default, a.pass_through), (3, None, true));
    }
}
