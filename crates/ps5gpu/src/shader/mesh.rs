// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Geometry subgroups (merged ES+GS with GS_EN) as mesh workgroups.
//!
//! One workgroup runs one NGG subgroup: `threads` guest lanes sharing the
//! LDS (Workgroup memory) and barriers. The entry point:
//!
//! 1. computes the hardware inputs of every lane — the subgroup counts in
//!    s2, the wave info in s3, the ES vertex offsets of the lane's input
//!    primitive in v0/v1, the primitive id in v2, the vertex index (fetched
//!    from the index buffer) in v5 and the instance in v8 — into private
//!    variables the guest program reads;
//! 2. calls the guest program, whose position / parameter / primitive
//!    exports go to private per-lane variables and whose GS_ALLOC_REQ
//!    message stores the subgroup's vertex and primitive counts;
//! 3. after a barrier, declares those counts with `SetMeshOutputsEXT` and
//!    writes lane `i`'s exports as vertex `i` and primitive `i` (indices in
//!    10-bit fields, bit 31 culls a null primitive).
//!
//! Draw parameters follow the stage's shader data in the shader-data
//! buffer, from dword `draw_dword`: index count,
//! base vertex, first instance, index size in bytes (0: not indexed), the
//! 64-bit index buffer address (read through the device-address page
//! table) and the shader-data dword of a strip draw's restart table.

use super::emit::{Emitter, R, SCOPE_WORKGROUP};
use crate::pipeline::compiler::MeshConfig;
use ps5shader::ssa::*;
use rspirv::dr::Operand;
use rspirv::spirv::{self, BuiltIn, Capability, Decoration, GlslStd450Op as GLOp, Op as S, StorageClass as SC, Word};
use std::collections::BTreeMap;

/// Hardware inputs the entry point computes per lane half.
pub const IN_S2: usize = 0;
pub const IN_S3: usize = 1;
pub const IN_V0: usize = 2;
pub const IN_V1: usize = 3;
pub const IN_V2: usize = 4;
pub const IN_V5: usize = 5;
pub const IN_V8: usize = 6;
const INPUTS: usize = 7;

/// `s_sendmsg` GS_ALLOC_REQ.
const MSG_GS_ALLOC_REQ: u32 = 9;

/// Draw-parameter dwords (shader-data buffer).
pub const DRAW_INDEX_COUNT: u32 = 0;
pub const DRAW_BASE_VERTEX: u32 = 1;
pub const DRAW_FIRST_INSTANCE: u32 = 2;
pub const DRAW_INDEX_BYTES: u32 = 3;
pub const DRAW_INDEX_ADDRESS: u32 = 4;
/// Shader-data dword of the strip restart table (0: none), see
/// [`restart_table`].
pub const DRAW_RESTART_TABLE: u32 = 6;
pub const DRAW_DWORDS: u32 = 7;

/// The restart table of an indexed strip draw with primitive restart:
/// `[length, restart index, start...]` where `start[p]` is the index
/// position the strip holding position `p` starts at (the position after
/// the last restart index at or before `p`). None when no index restarts.
/// A primitive at position `p` uses positions `p..p+2`; it lies inside one
/// strip when `p >= start[p + 2]`, and its winding counts from that start.
pub fn restart_table(indices: &[u8], index_bytes: u32, restart_index: u32) -> Option<Vec<u32>> {
    let size = index_bytes as usize;
    if size != 2 && size != 4 {
        return None;
    }
    let restart = if size == 2 { restart_index & 0xFFFF } else { restart_index };
    let count = indices.len() / size;
    let mut table = Vec::with_capacity(count + 2);
    table.extend([count as u32, restart]);
    let mut start = 0u32;
    let mut restarted = false;
    for p in 0..count {
        let b = &indices[p * size..p * size + size];
        let index = if size == 2 { u16::from_le_bytes([b[0], b[1]]) as u32 } else { u32::from_le_bytes([b[0], b[1], b[2], b[3]]) };
        if index == restart {
            restarted = true;
            start = p as u32 + 1;
        }
        table.push(start);
    }
    (restarted && count > 0).then_some(table)
}

/// Mesh-mode variables.
pub struct MeshIo {
    /// Per-lane hardware inputs (Private u32[lanes] each).
    inputs: [Word; INPUTS],
    /// Per-lane exports (Private arrays indexed by the lane half).
    pos_data: Word,
    params_data: BTreeMap<u32, Word>,
    prim_data: Word,
    /// Subgroup vertex and primitive counts (Workgroup u32[2]).
    alloc: Word,
    /// Mesh outputs.
    pos_out: Word,
    params_out: BTreeMap<u32, Word>,
    tri_out: Word,
    cull_out: Word,
    /// Per-vertex gl_ClipDistance: the guest's distances PA_CL_VS_OUT_CNTL
    /// enables, then -1 for an all-zero position (see
    /// `io::zero_position_distance`).
    clip_out: Option<Word>,
    /// The guest clip and cull distances of each lane half (Private
    /// f32[lanes][n]), and gl_CullDistance.
    clip_data: Option<(Word, u32)>,
    cull_data: Option<(Word, u32)>,
    cull_distance_out: Option<Word>,
    /// Render-target layer and viewport index (the misc vector as
    /// PA_CL_VS_OUT_CNTL places it): per guest lane in Workgroup memory,
    /// written per primitive from its first vertex.
    layer: Option<(Word, Word)>,
    viewport: Option<(Word, Word)>,
}

fn array(e: &mut Emitter, elem: Word, n: u32) -> Word {
    let c = e.cu(n);
    e.b.type_array(elem, c)
}

/// Declare the mesh interface (`params`: exported parameter slots; `layer`,
/// `viewport`: whether the program exports them).
pub fn declare(e: &mut Emitter, params: &[u32], layer: bool, viewport: bool) {
    let m = e.mesh.expect("mesh mode");
    e.cap(Capability::MeshShadingEXT);
    e.exts.insert("SPV_EXT_mesh_shader");
    let tu = e.t_u32();
    let v4 = e.t_fvec(4);
    let lanes = e.lanes as u32;
    let lane_u = array(e, tu, lanes);
    let lane_v4 = array(e, v4, lanes);
    let inputs = [0; INPUTS].map(|_| e.global(lane_u, SC::Private));
    let pos_data = e.global(lane_v4, SC::Private);
    let prim_data = e.global(lane_u, SC::Private);
    let two = array(e, tu, 2);
    let alloc = e.global(two, SC::Workgroup);
    let verts_v4 = array(e, v4, m.max_vertices);
    let pos_out = e.global(verts_v4, SC::Output);
    e.decorate(pos_out, Decoration::BuiltIn, &[BuiltIn::Position as u32]);
    // Invariant as the vertex path's position (see `io::per_vertex`).
    e.decorate(pos_out, Decoration::Invariant, &[]);
    let mut params_data = BTreeMap::new();
    let mut params_out = BTreeMap::new();
    for &p in params {
        if params_data.contains_key(&p) {
            continue;
        }
        params_data.insert(p, e.global(lane_v4, SC::Private));
        let out = e.global(verts_v4, SC::Output);
        e.decorate(out, Decoration::Location, &[p]);
        params_out.insert(p, out);
    }
    // Triangles (three vertex indices) or, for line-strip output, lines;
    // a point is one index.
    let indices = if m.points { tu } else { e.t_uvec(if m.lines { 2 } else { 3 }) };
    let tris = array(e, indices, m.max_primitives);
    let tri_out = e.global(tris, SC::Output);
    let builtin = if m.points {
        BuiltIn::PrimitivePointIndicesEXT
    } else if m.lines {
        BuiltIn::PrimitiveLineIndicesEXT
    } else {
        BuiltIn::PrimitiveTriangleIndicesEXT
    };
    e.decorate(tri_out, Decoration::BuiltIn, &[builtin as u32]);
    let tb = e.t_bool();
    let culls = array(e, tb, m.max_primitives);
    let cull_out = e.global(culls, SC::Output);
    e.decorate(cull_out, Decoration::BuiltIn, &[BuiltIn::CullPrimitiveEXT as u32]);
    e.decorate(cull_out, Decoration::PerPrimitiveEXT, &[]);
    let per_primitive = |e: &mut Emitter, builtin: BuiltIn| {
        let lanes_all = array(e, tu, m.threads);
        let data = e.global(lanes_all, SC::Workgroup);
        let prims_u = array(e, tu, m.max_primitives);
        let out = e.global(prims_u, SC::Output);
        e.decorate(out, Decoration::BuiltIn, &[builtin as u32]);
        e.decorate(out, Decoration::PerPrimitiveEXT, &[]);
        (data, out)
    };
    let layer = layer.then(|| per_primitive(e, BuiltIn::Layer));
    let viewport = viewport.then(|| per_primitive(e, BuiltIn::ViewportIndex));
    // Clip and cull distances as the vertex path declares them.
    let control = e.input.request.position_export_control;
    let supported = e.input.request.supports_clip_distance;
    let (clips, culls) = if supported { ((control & 0xFF).count_ones(), (control >> 8 & 0xFF).count_ones()) } else { (0, 0) };
    let zero_position = supported && super::io::zero_position_clip_enabled() && clips + culls < 8;
    let tf = e.t_f32();
    let distances = |e: &mut Emitter, n: u32, builtin: BuiltIn| {
        let per_vertex = array(e, tf, n);
        let verts = array(e, per_vertex, m.max_vertices);
        let out = e.global(verts, SC::Output);
        e.decorate(out, Decoration::BuiltIn, &[builtin as u32]);
        out
    };
    let clip_len = clips + zero_position as u32;
    let clip_out = (clip_len > 0).then(|| {
        e.cap(Capability::ClipDistance);
        distances(e, clip_len, BuiltIn::ClipDistance)
    });
    let cull_distance_out = (culls > 0).then(|| {
        e.cap(Capability::CullDistance);
        distances(e, culls, BuiltIn::CullDistance)
    });
    let lane_data = |e: &mut Emitter, n: u32| {
        let per_lane = array(e, tf, n);
        let halves = array(e, per_lane, lanes);
        (e.global(halves, SC::Private), n)
    };
    let clip_data = (clips > 0).then(|| lane_data(e, clips));
    let cull_data = (culls > 0).then(|| lane_data(e, culls));
    e.mesh_io = Some(MeshIo { inputs, pos_data, params_data, prim_data, alloc, pos_out, params_out, tri_out, cull_out, clip_out, clip_data, cull_data, cull_distance_out, layer, viewport });
}

fn io<'e>(e: &'e Emitter<'_, '_>) -> R<&'e MeshIo> {
    e.mesh_io.as_ref().ok_or_else(|| "mesh interface not declared".to_string())
}

/// Whether the program writes the viewport index.
pub fn writes_viewport_index(e: &Emitter) -> bool {
    e.mesh_io.as_ref().is_some_and(|io| io.viewport.is_some())
}

/// Locations of the exported parameters.
pub fn param_locations(e: &Emitter) -> Vec<u32> {
    e.mesh_io.as_ref().map(|io| io.params_out.keys().copied().collect()).unwrap_or_default()
}

/// Hardware input `k` of this lane half (loaded in the guest prologue).
pub fn input(e: &mut Emitter, k: usize) -> R<Word> {
    let var = io(e)?.inputs[k];
    let tu = e.t_u32();
    let p = e.ptr(SC::Private, tu);
    let h = e.cu(e.half as u32);
    let a = e.hoist(S::AccessChain, p, &[var, h]);
    Ok(e.hoist(S::Load, tu, &[a]))
}

/// A position, parameter or primitive export of this lane half.
pub fn export(e: &mut Emitter, target: u32, enable: u8, data: &[Word], exec: Word) -> R<()> {
    let tf = e.t_f32();
    let tu = e.t_u32();
    let v4 = e.t_fvec(4);
    let h = e.cu(e.half as u32);
    if target == 12 {
        let var = io(e)?.pos_data;
        let mut fl: Vec<Word> = (0..4).map(|i| if i < data.len() { e.bitcast(tf, data[i]) } else { e.cf(0.0) }).collect();
        super::io::clip_space(e, &mut fl);
        let pos = e.construct(v4, &fl);
        let p = e.ptr(SC::Private, v4);
        let ptr = e.access(p, var, &[h]);
        e.if_then(exec, |e| e.store(ptr, pos));
    } else if (32..64).contains(&target) {
        let Some(&var) = io(e)?.params_data.get(&(target - 32)) else { return Ok(()) };
        let p = e.ptr(SC::Private, tf);
        let comps: Vec<(Word, Word)> = (0..4u32)
            .filter(|i| enable >> i & 1 != 0 && (*i as usize) < data.len())
            .map(|i| {
                let idx = e.cu(i);
                let ptr = e.access(p, var, &[h, idx]);
                (ptr, e.bitcast(tf, data[i as usize]))
            })
            .collect();
        e.if_then(exec, |e| {
            for (ptr, val) in comps {
                e.store(ptr, val);
            }
        });
    } else if (13..=16).contains(&target) {
        // POS1..POS4: the layer and viewport index of the misc vector, the
        // clip and cull distances.
        let control = e.input.request.position_export_control;
        let (layer, viewport) = (io(e)?.layer, io(e)?.viewport);
        let (clip_data, cull_data) = (io(e)?.clip_data, io(e)?.cull_data);
        let mut stores = Vec::new();
        let mut distances = Vec::new();
        for c in 0..4u32 {
            if enable >> c & 1 == 0 || c as usize >= data.len() {
                continue;
            }
            let out = ps5shader::resources::decode_position_export_component(control, target - 12, c);
            let raw = data[c as usize];
            for (index, data) in [(out.clip_distance, clip_data), (out.cull_distance, cull_data)] {
                if let (Some(i), Some((var, n))) = (index, data) {
                    if i < n {
                        let p = e.ptr(SC::Private, tf);
                        let i = e.cu(i);
                        let ptr = e.access(p, var, &[h, i]);
                        distances.push((ptr, e.bitcast(tf, raw)));
                    }
                }
            }
            if let (true, Some((var, _))) = (out.layer, layer) {
                let m = e.cu(0x7FF);
                stores.push((var, e.band(raw, m)));
            }
            if let (true, Some((var, _))) = (out.viewport, viewport) {
                // MISC.z packs the viewport index in bits 16..19 next to the layer.
                stores.push((var, e.bits(raw, 16, 4)));
            }
        }
        if !distances.is_empty() {
            e.if_then(exec, |e| {
                for (ptr, v) in distances {
                    e.store(ptr, v);
                }
            });
        }
        if !stores.is_empty() {
            let g = super::io::guest_index(e);
            let p = e.ptr(SC::Workgroup, tu);
            let stores: Vec<(Word, Word)> = stores.into_iter().map(|(var, v)| (e.access(p, var, &[g]), v)).collect();
            e.if_then(exec, |e| {
                for (ptr, v) in stores {
                    e.store(ptr, v);
                }
            });
        }
    } else if target == 20 && !data.is_empty() {
        let var = io(e)?.prim_data;
        let p = e.ptr(SC::Private, tu);
        let ptr = e.access(p, var, &[h]);
        let v = data[0];
        e.if_then(exec, |e| e.store(ptr, v));
    }
    Ok(())
}

/// `s_sendmsg` in mesh mode: GS_ALLOC_REQ stores the subgroup's vertex
/// (m0 bits 0..9) and primitive (bits 12..22) counts.
pub fn send_message(e: &mut Emitter, message: u32, m0: Word) -> R<()> {
    if message & 0xF != MSG_GS_ALLOC_REQ {
        return Ok(());
    }
    let alloc = io(e)?.alloc;
    let tu = e.t_u32();
    let var = e.builtin_input(BuiltIn::LocalInvocationIndex, tu);
    let j = e.load(tu, var);
    let zero = e.cu(0);
    let first = e.ieq(j, zero);
    let p = e.ptr(SC::Workgroup, tu);
    e.if_then(first, |e| {
        for (field, (offset, bits)) in [(0u32, 10u32), (12, 11)].into_iter().enumerate() {
            let (o, n) = (e.cu(offset), e.cu(bits));
            let v = e.ins(S::BitFieldUExtract, tu, &[m0, o, n]);
            let f = e.cu(field as u32);
            let ptr = e.access(p, alloc, &[f]);
            e.store(ptr, v);
        }
    });
    Ok(())
}

/// Lanes of a guest wave and the shift from a subgroup lane to its wave
/// index (`RPS5E_MESH_WAVE64_INFO=1`: every program numbered as wave64,
/// the layout before wave32 subgroups were told their own waves).
pub fn wave_layout(wave_size: u32) -> (u32, u32) {
    static WAVE64: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let wave64 = *WAVE64.get_or_init(|| std::env::var("RPS5E_MESH_WAVE64_INFO").is_ok_and(|v| v == "1"));
    if wave_size == 32 && !wave64 {
        (32, 5)
    } else {
        (64, 6)
    }
}

/// Waves of a subgroup of `threads` lanes (the s3 bits 28..31).
pub fn subgroup_waves(threads: u32, wave_size: u32) -> u32 {
    (threads / wave_size).min(15)
}

/// The s3 word lane `local` of a subgroup reads: this wave's vertex
/// (bits 0..7) and primitive (8..15) counts, its wave index (24..27) and
/// the subgroup's wave count (28..31). The entry point computes the same
/// value on the device.
pub fn merged_wave_info(local: u32, vertices: u32, primitives: u32, threads: u32, wave_size: u32) -> u32 {
    let (width, shift) = wave_layout(wave_size);
    let base = local & !(width - 1);
    let vertex_count = vertices.saturating_sub(base).min(width);
    let primitive_count = primitives.saturating_sub(base).min(width);
    (subgroup_waves(threads, width) << 28) | ((local >> shift) << 24) | (primitive_count << 8) | vertex_count
}

/// Saturating `a - b`.
fn sat_sub(e: &mut Emitter, a: Word, b: Word) -> Word {
    let m = e.umin(a, b);
    e.isub(a, m)
}

/// Shader-data buffer dword `d` (in the current block).
fn draw_word(e: &mut Emitter, d: u32) -> R<Word> {
    let idx = e.cu(d);
    e.word_buffer_load(ps5shader::resources::DescriptorBindingKind::ShaderData, idx)
}

/// Emit the entry point around guest function `guest`; returns its id.
pub fn emit_entry(e: &mut Emitter, guest: Word) -> R<Word> {
    let m: MeshConfig = e.mesh.expect("mesh mode");
    let clip_out = io(e)?.clip_out;
    let (clip_data, cull_data, cull_distance_out) = (io(e)?.clip_data, io(e)?.cull_data, io(e)?.cull_distance_out);
    let (inputs, pos_data, params_data, prim_data, alloc, pos_out, params_out, tri_out, cull_out, layer, viewport) = {
        let io = io(e)?;
        (io.inputs, io.pos_data, io.params_data.clone(), io.prim_data, io.alloc, io.pos_out, io.params_out.clone(), io.tri_out, io.cull_out, io.layer, io.viewport)
    };
    let void = e.ty(Type::Void);
    let fnty = e.b.type_function(void, []);
    let entry = e.b.begin_function(void, None, spirv::FunctionControl::NONE, fnty).map_err(|err| format!("{err:?}"))?;
    let label = e.b.id();
    e.begin(label);
    let tu = e.t_u32();
    let tb = e.t_bool();
    let uv3 = e.t_uvec(3);
    let v4 = e.t_fvec(4);
    let lidx = e.builtin_input(BuiltIn::LocalInvocationIndex, tu);
    let j = e.load(tu, lidx);
    let wgv = e.builtin_input(BuiltIn::WorkgroupId, uv3);
    let wg = e.load(uv3, wgv);
    let wg_x = e.extract(tu, wg, 0);
    let wg_y = e.extract(tu, wg, 1);
    let d = |e: &mut Emitter, k: u32| draw_word(e, m.draw_dword + k);
    let index_count = d(e, DRAW_INDEX_COUNT)?;
    let base_vertex = d(e, DRAW_BASE_VERTEX)?;
    let first_instance = d(e, DRAW_FIRST_INSTANCE)?;
    let index_bytes = d(e, DRAW_INDEX_BYTES)?;
    let addr_lo = d(e, DRAW_INDEX_ADDRESS)?;
    let addr_hi = d(e, DRAW_INDEX_ADDRESS + 1)?;
    let index_base = e.join64(addr_lo, addr_hi);
    let (size, step) = (m.input_size(), m.input_step());
    let fan = m.input_primitive == 5;
    let strip = m.input_primitive == 6;
    let ppg = e.cu(m.primitives_per_group);
    let first_primitive = e.imul(wg_x, ppg);
    let stepw = e.cu(step);
    let first_vertex = e.imul(first_primitive, stepw);
    let left = sat_sub(e, index_count, first_vertex);
    let vpg = e.cu(m.vertices_per_group);
    let vertices = e.umin(left, vpg);
    let sizew = e.cu(size);
    let few = e.ult(vertices, sizew);
    let rest = sat_sub(e, vertices, sizew);
    let q = e.ins(S::UDiv, tu, &[rest, stepw]);
    let one = e.cu(1);
    let q1 = e.iadd(q, one);
    let zero = e.cu(0);
    let primitives = e.select(tu, few, zero, q1);
    let c12 = e.cu(12);
    let c22 = e.cu(22);
    let vs = e.shl(vertices, c12);
    let ps = e.shl(primitives, c22);
    let s2 = e.bor(vs, ps);
    let indexed = e.ine(index_bytes, zero);
    let item = e.cu(m.esgs_item_size);
    let pu = e.ptr(SC::Private, tu);
    // The guest wave: s3 counts this wave's vertices and primitives and
    // numbers the wave inside the subgroup (wave32 subgroups hold twice as
    // many waves as wave64 ones).
    let (wave_size, wave_shift) = wave_layout(e.f.wave_size);
    // Strip restart table (shader-data dword of `[length, restart index,
    // starts...]`, 0 without one) and its restart index.
    let restart = if strip {
        let table = d(e, DRAW_RESTART_TABLE)?;
        let has_table = e.ine(table, zero);
        let mut err = None;
        let value = e.if_value(has_table, tu, zero, |e| {
            let at = e.iadd(table, one);
            e.word_buffer_load(ps5shader::resources::DescriptorBindingKind::ShaderData, at).unwrap_or_else(|m| {
                err = Some(m);
                e.cu(0)
            })
        });
        if let Some(m) = err {
            return Err(m);
        }
        Some((table, has_table, value))
    } else {
        None
    };
    for half in 0..e.lanes {
        // Guest lane of this half: host invocation j holds guest lanes
        // 64 * (j / 32) + j % 32 and that plus 32.
        let local = if e.lanes == 2 {
            let c31 = e.cu(31);
            let lane = e.band(j, c31);
            let not31 = e.cu(!31);
            let w = e.band(j, not31);
            let base = e.shl(w, one);
            let g = e.bor(base, lane);
            let off = e.cu(32 * half as u32);
            e.iadd(g, off)
        } else {
            j
        };
        let shift = e.cu(wave_shift);
        let wave = e.shr(local, shift);
        let base_mask = e.cu(!(wave_size - 1));
        let wave_base = e.band(local, base_mask);
        let width = e.cu(wave_size);
        let vl = sat_sub(e, vertices, wave_base);
        let vertex_count = e.umin(vl, width);
        let pl = sat_sub(e, primitives, wave_base);
        let primitive_count = e.umin(pl, width);
        let c24 = e.cu(24);
        let wbits = e.shl(wave, c24);
        let waves = e.cu(subgroup_waves(m.threads, wave_size) << 28);
        let wave_info = e.bor(wbits, waves);
        let c8 = e.cu(8);
        let pcs = e.shl(primitive_count, c8);
        let counts = e.bor(pcs, vertex_count);
        let s3 = e.bor(wave_info, counts);
        // Strips with primitive restart: the strip holding this
        // primitive's last vertex starts at `strip_start`; the winding
        // counts from it, and a primitive reaching back across a restart
        // collapses onto one vertex.
        let strip_position = e.iadd(first_primitive, local);
        let (parity, restarted) = match restart {
            Some((table, has_table, _)) => {
                let mut err = None;
                let start = e.if_value(has_table, tu, zero, |e| {
                    let r = (|| -> R<Word> {
                        let length = e.word_buffer_load(ps5shader::resources::DescriptorBindingKind::ShaderData, table)?;
                        let last = sat_sub(e, length, one);
                        let two = e.cu(2);
                        let p = e.iadd(strip_position, two);
                        let p = e.umin(p, last);
                        let header = e.cu(2);
                        let at = e.iadd(table, header);
                        let at = e.iadd(at, p);
                        e.word_buffer_load(ps5shader::resources::DescriptorBindingKind::ShaderData, at)
                    })();
                    r.unwrap_or_else(|m| {
                        err = Some(m);
                        e.cu(0)
                    })
                });
                if let Some(m) = err {
                    return Err(m);
                }
                let s = e.isub(strip_position, start);
                (e.band(s, one), Some(e.ult(strip_position, start)))
            }
            None if strip => (e.band(strip_position, one), None),
            None => (zero, None),
        };
        let vertex = e.imul(local, stepw);
        let collapsed = e.imul(vertex, item);
        let off = |e: &mut Emitter, add: Word| {
            let v = e.iadd(vertex, add);
            e.imul(v, item)
        };
        let (first, second, third) = if fan {
            let two = e.cu(2);
            (off(e, one), off(e, two), zero)
        } else {
            let f = off(e, parity);
            let s = if size >= 2 {
                let vp = e.iadd(vertex, one);
                let v = e.isub(vp, parity);
                e.imul(v, item)
            } else {
                zero
            };
            let t = if size == 3 {
                let two = e.cu(2);
                off(e, two)
            } else {
                zero
            };
            match restarted {
                Some(r) => (e.select(tu, r, collapsed, f), e.select(tu, r, collapsed, s), e.select(tu, r, collapsed, t)),
                None => (f, s, t),
            }
        };
        let ffff = e.cu(0xFFFF);
        let c16 = e.cu(16);
        let fl = e.band(first, ffff);
        let sh = e.shl(second, c16);
        let v0 = e.bor(fl, sh);
        let v1 = e.band(third, ffff);
        let v2 = e.iadd(first_primitive, local);
        let fl_local = e.iadd(first_vertex, local);
        let input_vertex = if fan {
            let is0 = e.ieq(local, zero);
            e.select(tu, is0, zero, fl_local)
        } else {
            fl_local
        };
        let in_range = e.ult(local, vertices);
        let fetch = e.land(indexed, in_range);
        // The index base is only 2-byte aligned (16-bit indices): align
        // the full address to the dword the index lives in, and take the
        // sub-dword position from the full address too.
        let byte_offset = e.imul(input_vertex, index_bytes);
        let t64 = e.ty(Type::U64);
        let o64 = e.ins(S::UConvert, t64, &[byte_offset]);
        let full = e.ins(S::IAdd, t64, &[index_base, o64]);
        let not3_64 = e.c64(!3);
        let addr = e.ins(S::BitwiseAnd, t64, &[full, not3_64]);
        let full_lo = e.ins(S::UConvert, tu, &[full]);
        let mut err = None;
        let packed = e.if_value(fetch, tu, zero, |e| match e.device_load(addr) {
            Ok(w) => w,
            Err(m) => {
                err = Some(m);
                e.cu(0)
            }
        });
        if let Some(m) = err {
            return Err(m);
        }
        let three = e.cu(3);
        let sub = e.band(full_lo, three);
        let shift = e.imul(sub, c8);
        let bits = e.imul(index_bytes, c8);
        let index = e.ins(S::BitFieldUExtract, tu, &[packed, shift, bits]);
        // A restart index fetches vertex 0 (only collapsed primitives use it).
        let index = match restart {
            Some((_, has_table, value)) => {
                let is = e.ieq(index, value);
                let reset = e.land(has_table, is);
                e.select(tu, reset, zero, index)
            }
            None => index,
        };
        let picked = e.select(tu, indexed, index, input_vertex);
        let v5 = e.iadd(base_vertex, picked);
        let v8 = e.iadd(first_instance, wg_y);
        let h = e.cu(half as u32);
        for (k, v) in [(IN_S2, s2), (IN_S3, s3), (IN_V0, v0), (IN_V1, v1), (IN_V2, v2), (IN_V5, v5), (IN_V8, v8)] {
            let ptr = e.access(pu, inputs[k], &[h]);
            e.store(ptr, v);
        }
        let null = e.cu(0x8000_0000);
        let ptr = e.access(pu, prim_data, &[h]);
        e.store(ptr, null);
    }
    let first_inv = e.ieq(j, zero);
    let pw = e.ptr(SC::Workgroup, tu);
    e.if_then(first_inv, |e| {
        for f in 0..2 {
            let fi = e.cu(f);
            let ptr = e.access(pw, alloc, &[fi]);
            let z = e.cu(0);
            e.store(ptr, z);
        }
    });
    let call = e.b.id();
    e.b.function_call(void, Some(call), guest, []).map_err(|err| format!("{err:?}"))?;
    let (scope, sem) = (e.cu(SCOPE_WORKGROUP), e.cu(0x108));
    e.void(S::ControlBarrier, vec![Operand::IdRef(scope), Operand::IdRef(scope), Operand::IdRef(sem)]);
    let a0 = {
        let f = e.cu(0);
        let ptr = e.access(pw, alloc, &[f]);
        e.load(tu, ptr)
    };
    let a1 = {
        let f = e.cu(1);
        let ptr = e.access(pw, alloc, &[f]);
        e.load(tu, ptr)
    };
    let maxv = e.cu(m.max_vertices);
    let maxp = e.cu(m.max_primitives);
    let verts = e.umin(a0, maxv);
    // Every mesh shader reads NumWorkgroups: RADV sets up the user SGPRs
    // of a mesh dispatch (and the count it announces) only for a shader
    // with the grid-size slot; one without keeps the previous vertex
    // pipeline's layout, the command processor reads the dispatch as
    // register data and the GPU hangs. No dispatch has zero groups.
    let uv3 = e.t_uvec(3);
    let grid = e.builtin_input(BuiltIn::NumWorkgroups, uv3);
    let grid = e.load(uv3, grid);
    let groups_x = e.extract(tu, grid, 0);
    let none = e.ieq(groups_x, zero);
    let verts = e.select(tu, none, zero, verts);
    let prims = e.umin(a1, maxp);
    e.void(S::SetMeshOutputsEXT, vec![Operand::IdRef(verts), Operand::IdRef(prims)]);
    let any = e.gl(tu, GLOp::UMax, &[verts, one]);
    let last = e.isub(any, one);
    let pv4 = e.ptr(SC::Private, v4);
    let ov4 = e.ptr(SC::Output, v4);
    let corners = if m.points {
        1
    } else if m.lines {
        2
    } else {
        3
    };
    let uvn = if m.points { tu } else { e.t_uvec(corners) };
    let ouvn = e.ptr(SC::Output, uvn);
    let ob = e.ptr(SC::Output, tb);
    for half in 0..e.lanes {
        let index = if e.lanes == 2 {
            let c31 = e.cu(31);
            let lane = e.band(j, c31);
            let not31 = e.cu(!31);
            let w = e.band(j, not31);
            let base = e.shl(w, one);
            let g = e.bor(base, lane);
            let off = e.cu(32 * half as u32);
            e.iadd(g, off)
        } else {
            j
        };
        let h = e.cu(half as u32);
        let is_vertex = e.ult(index, verts);
        let pairs: Vec<(Word, Word)> = std::iter::once((pos_data, pos_out)).chain(params_data.iter().map(|(p, &d)| (d, params_out[p]))).collect();
        e.if_then(is_vertex, |e| {
            for (data, out) in pairs {
                let src = e.access(pv4, data, &[h]);
                let v = e.load(v4, src);
                let dst = e.access(ov4, out, &[index]);
                e.store(dst, v);
                if data != pos_data {
                    continue;
                }
                let tf = e.t_f32();
                let (pf, of) = (e.ptr(SC::Private, tf), e.ptr(SC::Output, tf));
                // The guest's clip and cull distances.
                for (lane_data, out) in [(clip_data, clip_out), (cull_data, cull_distance_out)] {
                    let (Some((var, n)), Some(out)) = (lane_data, out) else { continue };
                    for i in 0..n {
                        let k = e.cu(i);
                        let src = e.access(pf, var, &[h, k]);
                        let d = e.load(tf, src);
                        let dst = e.access(of, out, &[index, k]);
                        e.store(dst, d);
                    }
                }
                // The all-zero position distance after them.
                let clips = clip_data.map_or(0, |(_, n)| n);
                let culls = cull_data.map_or(0, |(_, n)| n);
                if let Some(clip) = clip_out.filter(|_| clips + culls < 8 && super::io::zero_position_clip_enabled()) {
                    let comps: Vec<Word> = (0..4).map(|c| e.extract(tf, v, c)).collect();
                    let d = super::io::zero_position_distance(e, &comps);
                    let k = e.cu(clips);
                    let dst = e.access(of, clip, &[index, k]);
                    e.store(dst, d);
                }
            }
        });
        let is_prim = e.ult(index, prims);
        e.if_then(is_prim, |e| {
            let src = e.access(pu, prim_data, &[h]);
            let packed = e.load(tu, src);
            let comps: Vec<Word> = (0..corners)
                .map(|c| {
                    let (o, n) = (e.cu(c * 10), e.cu(9));
                    let f = e.ins(S::BitFieldUExtract, tu, &[packed, o, n]);
                    e.umin(f, last)
                })
                .collect();
            let tri = if m.points { comps[0] } else { e.construct(uvn, &comps) };
            let dst = e.access(ouvn, tri_out, &[index]);
            e.store(dst, tri);
            let top = e.cu(0x8000_0000);
            let bit = e.band(packed, top);
            let z = e.cu(0);
            let culled = e.ine(bit, z);
            let dst = e.access(ob, cull_out, &[index]);
            e.store(dst, culled);
            // Per-primitive values come from the primitive's first vertex.
            for (data, out) in layer.into_iter().chain(viewport) {
                let pw = e.ptr(SC::Workgroup, tu);
                let src = e.access(pw, data, &[comps[0]]);
                let l = e.load(tu, src);
                let po = e.ptr(SC::Output, tu);
                let dst = e.access(po, out, &[index]);
                e.store(dst, l);
            }
        });
    }
    e.b.ret().map_err(|err| format!("{err:?}"))?;
    e.b.end_function().map_err(|err| format!("{err:?}"))?;
    Ok(entry)
}
