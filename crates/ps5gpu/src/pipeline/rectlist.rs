// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Rectangle lists through tessellation.
//!
//! A RECT_LIST draw sends three vertices per rectangle; the hardware adds
//! the fourth corner. The host draws 3-vertex patches: the generated
//! control shader finds the corner vertex (the one whose two edges are
//! axis aligned), emits the patch starting at that corner plus the fourth
//! corner `v1 + v2 - v0` (every interpolated parameter extrapolated the
//! same way, flat and integer parameters copied from the first vertex),
//! and the evaluation shader places the four control points on a quad
//! domain with tessellation level 1, which rasterises the parallelogram as
//! two triangles.
//!
//! Both stages depend only on the vertex shader's parameter outputs and
//! its clip distance count (gl_ClipDistance is forwarded, the fourth
//! corner's extrapolated like a parameter).

use super::compiler::{ComponentKind, ReflectedVariable};
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{self, BuiltIn, Decoration as D, StorageClass as S, Word};
use std::collections::HashMap;

/// Patch size of the input array declarations (gl_MaxPatchVertices).
const MAX_PATCH_VERTICES: u32 = 32;

type R<T> = Result<T, rspirv::dr::Error>;

struct Emitter {
    b: Builder,
    void: Word,
    f32: Word,
    i32: Word,
    u32: Word,
    boolean: Word,
    vec4: Word,
    consts: HashMap<(Word, u32), Word>,
    interfaces: Vec<Word>,
}

impl Emitter {
    fn new(model: spirv::ExecutionModel) -> Emitter {
        let mut b = Builder::new();
        b.set_version(1, 3);
        b.capability(spirv::Capability::Shader);
        if matches!(model, spirv::ExecutionModel::TessellationControl | spirv::ExecutionModel::TessellationEvaluation) {
            b.capability(spirv::Capability::Tessellation);
        }
        b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
        let void = b.type_void();
        let f32 = b.type_float(32, None);
        let i32 = b.type_int(32, 1);
        let u32 = b.type_int(32, 0);
        let boolean = b.type_bool();
        let vec4 = b.type_vector(f32, 4);
        Emitter { b, void, f32, i32, u32, boolean, vec4, consts: HashMap::new(), interfaces: Vec::new() }
    }

    fn c(&mut self, ty: Word, bits: u32) -> Word {
        if let Some(&id) = self.consts.get(&(ty, bits)) {
            return id;
        }
        let id = self.b.constant_bit32(ty, bits);
        self.consts.insert((ty, bits), id);
        id
    }

    fn ci(&mut self, v: i32) -> Word {
        let t = self.i32;
        self.c(t, v as u32)
    }

    fn cf(&mut self, v: f32) -> Word {
        let t = self.f32;
        self.c(t, v.to_bits())
    }

    fn array(&mut self, elem: Word, len: u32) -> Word {
        let n = self.c(self.u32, len);
        self.b.type_array(elem, n)
    }

    /// Scalar or vector type of a parameter.
    fn param_type(&mut self, v: &ReflectedVariable) -> Word {
        let scalar = match v.kind {
            ComponentKind::Float => self.f32,
            ComponentKind::Uint => self.u32,
            ComponentKind::Sint => self.i32,
        };
        if v.components <= 1 {
            scalar
        } else {
            self.b.type_vector(scalar, v.components.min(4))
        }
    }

    /// A `float[clips]` gl_ClipDistance variable of `class`, arrayed per
    /// vertex as `float[vertices][clips]` when `vertices` is given.
    fn clip_var(&mut self, clips: u32, vertices: Option<u32>, class: S) -> Word {
        self.b.capability(spirv::Capability::ClipDistance);
        let f = self.f32;
        let per_vertex = self.array(f, clips);
        let ty = match vertices {
            Some(n) => self.array(per_vertex, n),
            None => per_vertex,
        };
        let v = self.var(ty, class);
        self.b.decorate(v, D::BuiltIn, [Operand::BuiltIn(BuiltIn::ClipDistance)]);
        v
    }

    /// `struct { vec4 gl_Position; }` decorated as a built-in block; an
    /// output position is invariant (as the vertex stage's).
    fn per_vertex(&mut self, output: bool) -> Word {
        // A distinct type per use: struct types are deduplicated otherwise
        // and their decorations would repeat.
        let id = self.b.id();
        let s = self.b.type_struct_id(Some(id), [self.vec4]);
        self.b.member_decorate(s, 0, D::BuiltIn, [Operand::BuiltIn(BuiltIn::Position)]);
        if output {
            self.b.member_decorate(s, 0, D::Invariant, []);
        }
        self.b.decorate(s, D::Block, []);
        s
    }

    fn var(&mut self, ty: Word, class: S) -> Word {
        let p = self.b.type_pointer(None, class, ty);
        let v = self.b.variable(p, None, class, None);
        self.interfaces.push(v);
        v
    }

    fn location(&mut self, v: Word, location: u32) {
        self.b.decorate(v, D::Location, [Operand::LiteralBit32(location)]);
    }

    fn load_at(&mut self, ty: Word, base: Word, class: S, indices: &[Word]) -> R<Word> {
        let p = self.b.type_pointer(None, class, ty);
        let a = self.b.access_chain(p, None, base, indices.iter().copied())?;
        self.b.load(ty, None, a, None, [])
    }

    fn store_at(&mut self, ty: Word, base: Word, class: S, indices: &[Word], value: Word) -> R<()> {
        let p = self.b.type_pointer(None, class, ty);
        let a = self.b.access_chain(p, None, base, indices.iter().copied())?;
        self.b.store(a, value, None, [])
    }

    fn begin(&mut self) -> R<Word> {
        let fnty = self.b.type_function(self.void, []);
        let main = self.b.begin_function(self.void, None, spirv::FunctionControl::NONE, fnty)?;
        self.b.begin_block(None)?;
        Ok(main)
    }

    fn finish(mut self, main: Word, model: spirv::ExecutionModel, modes: &[(spirv::ExecutionMode, Vec<u32>)]) -> R<Vec<u32>> {
        self.b.ret()?;
        self.b.end_function()?;
        for (m, args) in modes {
            self.b.execution_mode(main, *m, args);
        }
        let interfaces = std::mem::take(&mut self.interfaces);
        self.b.entry_point(model, main, "main", interfaces);
        Ok(self.b.module().assemble())
    }

    /// `cond ? a : b` on a scalar or vector of `components` (SPIR-V 1.3
    /// needs a vector condition for vector operands).
    fn select(&mut self, ty: Word, components: u32, cond: Word, a: Word, b: Word) -> R<Word> {
        let cond = if components <= 1 {
            cond
        } else {
            let bv = self.b.type_vector(self.boolean, components.min(4));
            self.b.composite_construct(bv, None, vec![cond; components.min(4) as usize])?
        };
        self.b.select(ty, None, cond, a, b)
    }

    /// `a * wa + b * wb + c * wc` on a float scalar/vector.
    fn weigh(&mut self, ty: Word, components: u32, v: [Word; 3], w: [Word; 3]) -> R<Word> {
        let mut acc = None;
        for i in 0..3 {
            let scaled = if components <= 1 {
                self.b.f_mul(ty, None, v[i], w[i])?
            } else {
                self.b.vector_times_scalar(ty, None, v[i], w[i])?
            };
            acc = Some(match acc {
                Some(a) => self.b.f_add(ty, None, a, scaled)?,
                None => scaled,
            });
        }
        Ok(acc.unwrap())
    }
}

/// The tessellation control shader for `params` (the vertex shader's
/// parameter outputs) and its `clips` clip distances.
pub fn control_shader(params: &[ReflectedVariable], clips: u32) -> Option<Vec<u32>> {
    let build = || -> R<Vec<u32>> {
        let model = spirv::ExecutionModel::TessellationControl;
        let mut e = Emitter::new(model);
        let pv_in = e.per_vertex(false);
        let pv_in_arr = e.array(pv_in, MAX_PATCH_VERTICES);
        let gl_in = e.var(pv_in_arr, S::Input);
        let pv_out = e.per_vertex(true);
        let pv_out_arr = e.array(pv_out, 4);
        let gl_out = e.var(pv_out_arr, S::Output);
        let i32t = e.i32;
        let invocation = e.var(i32t, S::Input);
        e.b.decorate(invocation, D::BuiltIn, [Operand::BuiltIn(BuiltIn::InvocationId)]);
        let f32t = e.f32;
        let outer_t = e.array(f32t, 4);
        let outer = e.var(outer_t, S::Output);
        e.b.decorate(outer, D::BuiltIn, [Operand::BuiltIn(BuiltIn::TessLevelOuter)]);
        e.b.decorate(outer, D::Patch, []);
        let inner_t = e.array(f32t, 2);
        let inner = e.var(inner_t, S::Output);
        e.b.decorate(inner, D::BuiltIn, [Operand::BuiltIn(BuiltIn::TessLevelInner)]);
        e.b.decorate(inner, D::Patch, []);
        let mut io = Vec::new();
        for p in params {
            let ty = e.param_type(p);
            let in_t = e.array(ty, MAX_PATCH_VERTICES);
            let out_t = e.array(ty, 4);
            let vin = e.var(in_t, S::Input);
            e.location(vin, p.location);
            let vout = e.var(out_t, S::Output);
            e.location(vout, p.location);
            if p.flat || p.kind != ComponentKind::Float {
                e.b.decorate(vin, D::Flat, []);
                e.b.decorate(vout, D::Flat, []);
            }
            io.push((p, ty, vin, vout));
        }
        let clip_io = (clips > 0).then(|| (e.clip_var(clips, Some(MAX_PATCH_VERTICES), S::Input), e.clip_var(clips, Some(4), S::Output)));

        let main = e.begin()?;
        let inv = e.b.load(e.i32, None, invocation, None, [])?;
        // Tessellation levels (written by every invocation: same values).
        let one = e.cf(1.0);
        for i in 0..4 {
            let idx = e.ci(i);
            e.store_at(f32t, outer, S::Output, &[idx], one)?;
        }
        for i in 0..2 {
            let idx = e.ci(i);
            e.store_at(f32t, inner, S::Output, &[idx], one)?;
        }
        let zero_i = e.ci(0);
        let mut pos = [0; 3];
        for (i, p) in pos.iter_mut().enumerate() {
            let idx = e.ci(i as i32);
            *p = e.load_at(e.vec4, gl_in, S::Input, &[idx, zero_i])?;
        }
        // Axis-aligned edges: equal x / equal y between vertex i and i+1.
        let mut same_x = [0; 3];
        let mut same_y = [0; 3];
        for i in 0..3 {
            let a = pos[i];
            let b = pos[(i + 1) % 3];
            let (f32t, bt) = (e.f32, e.boolean);
            let ax = e.b.composite_extract(f32t, None, a, [0])?;
            let bx = e.b.composite_extract(f32t, None, b, [0])?;
            let ay = e.b.composite_extract(f32t, None, a, [1])?;
            let by = e.b.composite_extract(f32t, None, b, [1])?;
            same_x[i] = e.b.f_ord_equal(bt, None, ax, bx)?;
            same_y[i] = e.b.f_ord_equal(bt, None, ay, by)?;
        }
        // Vertex i is the corner when its two edges are axis aligned in
        // different axes.
        let mut corner = [0; 3];
        let mut weight = [0; 3];
        let (minus_one, plus_one) = (e.cf(-1.0), e.cf(1.0));
        for i in 0..3 {
            let prev = (i + 2) % 3;
            let bt = e.boolean;
            let xy = e.b.logical_and(bt, None, same_x[i], same_y[prev])?;
            let yx = e.b.logical_and(bt, None, same_y[i], same_x[prev])?;
            corner[i] = e.b.logical_or(bt, None, xy, yx)?;
            weight[i] = e.b.select(e.f32, None, corner[i], minus_one, plus_one)?;
        }
        let (i1, i2) = (e.ci(1), e.ci(2));
        let first = e.b.select(e.i32, None, corner[2], i2, zero_i)?;
        let first = e.b.select(e.i32, None, corner[1], i1, first)?;
        let i3 = e.ci(3);
        let bt = e.boolean;
        let fourth = e.b.i_equal(bt, None, inv, i3)?;
        let sum = e.b.i_add(e.i32, None, first, inv)?;
        let index = e.b.s_mod(e.i32, None, sum, i3)?;

        let vec4 = e.vec4;
        let p3 = e.weigh(vec4, 4, pos, weight)?;
        let picked = e.load_at(vec4, gl_in, S::Input, &[index, zero_i])?;
        let position = e.select(vec4, 4, fourth, p3, picked)?;
        e.store_at(vec4, gl_out, S::Output, &[inv, zero_i], position)?;

        for (p, ty, vin, vout) in io {
            let picked = e.load_at(ty, vin, S::Input, &[index])?;
            let value = if p.flat || p.kind != ComponentKind::Float {
                let first_value = e.load_at(ty, vin, S::Input, &[zero_i])?;
                e.select(ty, p.components, fourth, first_value, picked)?
            } else {
                let mut v = [0; 3];
                for (i, slot) in v.iter_mut().enumerate() {
                    let idx = e.ci(i as i32);
                    *slot = e.load_at(ty, vin, S::Input, &[idx])?;
                }
                let extrapolated = e.weigh(ty, p.components, v, weight)?;
                e.select(ty, p.components, fourth, extrapolated, picked)?
            };
            e.store_at(ty, vout, S::Output, &[inv], value)?;
        }
        if let Some((cin, cout)) = clip_io {
            // Each distance as an interpolated parameter.
            for k in 0..clips {
                let kk = e.ci(k as i32);
                let picked = e.load_at(f32t, cin, S::Input, &[index, kk])?;
                let mut v = [0; 3];
                for (i, slot) in v.iter_mut().enumerate() {
                    let idx = e.ci(i as i32);
                    *slot = e.load_at(f32t, cin, S::Input, &[idx, kk])?;
                }
                let extrapolated = e.weigh(f32t, 1, v, weight)?;
                let value = e.select(f32t, 1, fourth, extrapolated, picked)?;
                e.store_at(f32t, cout, S::Output, &[inv, kk], value)?;
            }
        }
        e.finish(main, model, &[(spirv::ExecutionMode::OutputVertices, vec![4])])
    };
    build().ok()
}

/// The tessellation evaluation shader for `params` and `clips` clip
/// distances.
pub fn evaluation_shader(params: &[ReflectedVariable], clips: u32) -> Option<Vec<u32>> {
    let build = || -> R<Vec<u32>> {
        let model = spirv::ExecutionModel::TessellationEvaluation;
        let mut e = Emitter::new(model);
        let pv_in = e.per_vertex(false);
        let pv_in_arr = e.array(pv_in, MAX_PATCH_VERTICES);
        let gl_in = e.var(pv_in_arr, S::Input);
        let pv_out = e.per_vertex(true);
        let gl_out = e.var(pv_out, S::Output);
        let f32t = e.f32;
        let v3 = e.b.type_vector(f32t, 3);
        let coord = e.var(v3, S::Input);
        e.b.decorate(coord, D::BuiltIn, [Operand::BuiltIn(BuiltIn::TessCoord)]);
        let mut io = Vec::new();
        for p in params {
            let ty = e.param_type(p);
            let in_t = e.array(ty, MAX_PATCH_VERTICES);
            let vin = e.var(in_t, S::Input);
            e.location(vin, p.location);
            let vout = e.var(ty, S::Output);
            e.location(vout, p.location);
            if p.flat || p.kind != ComponentKind::Float {
                e.b.decorate(vin, D::Flat, []);
                e.b.decorate(vout, D::Flat, []);
            }
            io.push((ty, vin, vout));
        }
        let clip_io = (clips > 0).then(|| (e.clip_var(clips, Some(MAX_PATCH_VERTICES), S::Input), e.clip_var(clips, None, S::Output)));
        let main = e.begin()?;
        let tc = e.b.load(v3, None, coord, None, [])?;
        let x = e.b.composite_extract(f32t, None, tc, [0])?;
        let y = e.b.composite_extract(f32t, None, tc, [1])?;
        let i32t = e.i32;
        // The corner coordinates are 0 or 1 up to the tessellator's fixed
        // point: rounded, a coordinate just below 1 is still vertex 1.
        let half = e.cf(0.5);
        let x = e.b.f_add(f32t, None, x, half)?;
        let y = e.b.f_add(f32t, None, y, half)?;
        let xi = e.b.convert_f_to_s(i32t, None, x)?;
        let yi = e.b.convert_f_to_s(i32t, None, y)?;
        let two = e.ci(2);
        let y2 = e.b.i_mul(i32t, None, yi, two)?;
        let index = e.b.i_add(i32t, None, y2, xi)?;
        let zero = e.ci(0);
        let vec4 = e.vec4;
        let position = e.load_at(vec4, gl_in, S::Input, &[index, zero])?;
        e.store_at(vec4, gl_out, S::Output, &[zero], position)?;
        for (ty, vin, vout) in io {
            let v = e.load_at(ty, vin, S::Input, &[index])?;
            e.b.store(vout, v, None, [])?;
        }
        if let Some((cin, cout)) = clip_io {
            for k in 0..clips {
                let kk = e.ci(k as i32);
                let v = e.load_at(f32t, cin, S::Input, &[index, kk])?;
                e.store_at(f32t, cout, S::Output, &[kk], v)?;
            }
        }
        e.finish(
            main,
            model,
            &[
                (spirv::ExecutionMode::Quads, vec![]),
                (spirv::ExecutionMode::SpacingEqual, vec![]),
                (spirv::ExecutionMode::VertexOrderCw, vec![]),
            ],
        )
    };
    build().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stages_assemble() {
        let params = [
            ReflectedVariable { location: 0, components: 4, kind: ComponentKind::Float, flat: false },
            ReflectedVariable { location: 1, components: 2, kind: ComponentKind::Float, flat: false },
            ReflectedVariable { location: 2, components: 1, kind: ComponentKind::Uint, flat: true },
        ];
        for words in [control_shader(&params, 0).unwrap(), evaluation_shader(&params, 0).unwrap(), control_shader(&params, 3).unwrap(), evaluation_shader(&params, 3).unwrap()] {
            assert_eq!(words[0], spirv::MAGIC_NUMBER);
            let mut loader = rspirv::dr::Loader::new();
            rspirv::binary::parse_words(&words, &mut loader).unwrap();
        }
    }
}
