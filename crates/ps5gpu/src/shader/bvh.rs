// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! `image_bvh_intersect_ray` / `image_bvh64_intersect_ray`: one ray against
//! one BVH node, evaluated in software. The node is read through the
//! device-address page table.
//!
//! Descriptor (T# words): base address in 256-byte units (`w0`, `w1[7:0]`),
//! box growing ULPs (`w1[30:23]`), box sorting (`w1[31]`), last node index
//! (`w2`, `w3[9:0]`), barycentric return (`w3[24]`). The node pointer's low
//! 3 bits are its type (0-3 triangle, 4 fp16 box, 5 fp32 box); a node lies
//! at `base * 256 + (pointer & !7) * 8`.
//!
//! Address operands: node pointer (2 dwords for the 64-bit form), ray
//! extent, origin xyz, then direction xyz and inverse direction xyz (packed
//! halves under A16).
//!
//! Result: a triangle returns `{t_num, t_denom, triangle id, hit}` (or the
//! two barycentric numerators in place of id and hit); a box returns its
//! four child pointers, missed children `0xFFFFFFFF`, sorted by entry
//! distance when box sorting is on.
//!
//! With [`super::EmitOptions::hw_rt`] a box query also drops the children
//! that hold no triangle the lane's ray hits, by the host ray query made at
//! the traversal's root ([`crate::rt`]; `rt_window`).

use super::emit::{Emitter, R};
use crate::rt::{table as rtt, ROOT_NODE_POINTER, T_ABSOLUTE, T_RELATIVE};
use ps5shader::resources::DescriptorBindingKind;
use ps5shader::ssa::*;
use rspirv::dr::Operand;
use rspirv::spirv::{Capability, GlslStd450Op as GLOp, Op as S, StorageClass as SC, Word};

const INVALID: u32 = 0xFFFF_FFFF;

struct Ops<'e, 'a, 'p> {
    e: &'e mut Emitter<'a, 'p>,
    tu: Word,
    tf: Word,
    tb: Word,
    t64: Word,
}

impl Ops<'_, '_, '_> {
    fn u(&mut self, v: u32) -> Word {
        self.e.cu(v)
    }
    fn f(&mut self, v: f32) -> Word {
        self.e.cf(v)
    }
    fn op(&mut self, op: S, ty: Word, a: Word, b: Word) -> Word {
        self.e.ins(op, ty, &[a, b])
    }
    fn fbin(&mut self, op: S, a: Word, b: Word) -> Word {
        let tf = self.tf;
        self.op(op, tf, a, b)
    }
    fn cmp(&mut self, op: S, a: Word, b: Word) -> Word {
        let tb = self.tb;
        self.op(op, tb, a, b)
    }
    fn as_f(&mut self, x: Word) -> Word {
        let tf = self.tf;
        self.e.bitcast(tf, x)
    }
    fn as_u(&mut self, x: Word) -> Word {
        let tu = self.tu;
        self.e.bitcast(tu, x)
    }
    fn self_f(&mut self, c: Word, a: Word, b: Word) -> Word {
        let tf = self.tf;
        self.e.select(tf, c, a, b)
    }
    fn sel_u(&mut self, c: Word, a: Word, b: Word) -> Word {
        let tu = self.tu;
        self.e.select(tu, c, a, b)
    }
    fn is_nan(&mut self, x: Word) -> Word {
        let tb = self.tb;
        self.e.ins(S::IsNan, tb, &[x])
    }
    fn widen(&mut self, x: Word) -> Word {
        let t64 = self.t64;
        self.e.ins(S::UConvert, t64, &[x])
    }
    fn pair(&mut self, lo: Word, hi: Word) -> Word {
        let (l, h) = (self.widen(lo), self.widen(hi));
        let s = self.e.c64(32);
        let t64 = self.t64;
        let h = self.op(S::ShiftLeftLogical, t64, h, s);
        self.op(S::BitwiseOr, t64, l, h)
    }
    /// The f32 value of the half in bits [15:0] of `bits`.
    fn half(&mut self, bits: Word) -> Word {
        let v2 = self.e.t_fvec(2);
        let p = self.e.gl(v2, GLOp::UnpackHalf2x16, &[bits]);
        let tf = self.tf;
        self.e.extract(tf, p, 0)
    }
    fn half_at(&mut self, packed: Word, high: bool) -> Word {
        let bits = if high {
            let s = self.u(16);
            self.e.shr(packed, s)
        } else {
            let m = self.u(0xFFFF);
            self.e.band(packed, m)
        };
        self.half(bits)
    }
    fn sub3(&mut self, a: [Word; 3], b: [Word; 3]) -> [Word; 3] {
        [self.fbin(S::FSub, a[0], b[0]), self.fbin(S::FSub, a[1], b[1]), self.fbin(S::FSub, a[2], b[2])]
    }
    fn cross(&mut self, a: [Word; 3], b: [Word; 3]) -> [Word; 3] {
        let term = |o: &mut Self, i: usize, j: usize| {
            let x = o.fbin(S::FMul, a[i], b[j]);
            let y = o.fbin(S::FMul, a[j], b[i]);
            o.fbin(S::FSub, x, y)
        };
        [term(self, 1, 2), term(self, 2, 0), term(self, 0, 1)]
    }
    fn dot(&mut self, a: [Word; 3], b: [Word; 3]) -> Word {
        let x = self.fbin(S::FMul, a[0], b[0]);
        let y = self.fbin(S::FMul, a[1], b[1]);
        let z = self.fbin(S::FMul, a[2], b[2]);
        let xy = self.fbin(S::FAdd, x, y);
        self.fbin(S::FAdd, xy, z)
    }
}

/// Ray and node operands of one query.
struct Query {
    words: [Word; 4],
    ty: Word,
    node_address: Word,
    is_triangle: Word,
    is_box16: Word,
    is_box32: Word,
    box_grow: Word,
    box_sort: Word,
    barycentrics: Word,
    extent: Word,
    origin: [Word; 3],
    direction: [Word; 3],
    inverse: [Word; 3],
}

fn triangle(o: &mut Ops, q: &Query) -> R<Word> {
    let d = o.e.device_load_dwords(q.node_address, 16)?;
    // The node's five vertices; the type picks the triangle: 0 = v0 v1 v2,
    // 1 = v1 v3 v2, 2 = v2 v3 v4, 3 = v2 v4 v0.
    let (one_u, two_u, three_u) = (o.u(1), o.u(2), o.u(3));
    let is1 = o.e.ieq(q.ty, one_u);
    let is2 = o.e.ieq(q.ty, two_u);
    let is3 = o.e.ieq(q.ty, three_u);
    let mut v1 = [0; 3];
    let mut v2 = [0; 3];
    let mut v3 = [0; 3];
    for a in 0..3 {
        let [x0, x1, x2, x3, x4] = [0, 1, 2, 3, 4].map(|k| o.as_f(d[3 * k + a]));
        let pick = |o: &mut Ops, by: [Word; 4]| {
            let r = o.self_f(is1, by[1], by[0]);
            let r = o.self_f(is2, by[2], r);
            o.self_f(is3, by[3], r)
        };
        v1[a] = pick(o, [x0, x1, x2, x2]);
        v2[a] = pick(o, [x1, x3, x3, x4]);
        v3[a] = pick(o, [x2, x2, x4, x0]);
    }
    let e1 = o.sub3(v2, v1);
    let e2 = o.sub3(v3, v1);
    let e3 = o.sub3(q.origin, v1);
    let s1 = o.cross(q.direction, e2);
    let s2 = o.cross(e3, e1);
    let mut t_num = o.dot(e2, s2);
    let mut t_denom = o.dot(s1, e1);
    let i_num = o.dot(e3, s1);
    let j_num = o.dot(q.direction, s2);
    let t = o.fbin(S::FDiv, t_num, t_denom);
    let u = o.fbin(S::FDiv, i_num, t_denom);
    let v = o.fbin(S::FDiv, j_num, t_denom);
    let (zero, one) = (o.f(0.0), o.f(1.0));
    let u_lo = o.cmp(S::FOrdLessThan, u, zero);
    let u_hi = o.cmp(S::FOrdGreaterThan, u, one);
    let v_lo = o.cmp(S::FOrdLessThan, v, zero);
    let uv = o.fbin(S::FAdd, u, v);
    let uv_hi = o.cmp(S::FOrdGreaterThan, uv, one);
    let t_lo = o.cmp(S::FOrdLessThan, t, zero);
    let a = o.e.lor(u_lo, u_hi);
    let b = o.e.lor(v_lo, uv_hi);
    let ab = o.e.lor(a, b);
    let missed = o.e.lor(ab, t_lo);
    let inf = o.f(f32::INFINITY);
    t_num = o.self_f(missed, inf, t_num);
    t_denom = o.self_f(missed, one, t_denom);
    let triangle_id = d[15];
    let three = o.u(3);
    let shift = o.e.shl(q.ty, three);
    let barycentric = |o: &mut Ops, source_shift: u32| {
        let ss = o.u(source_shift);
        let sh = o.e.iadd(shift, ss);
        let src = o.e.shr(triangle_id, sh);
        let three = o.u(3);
        let src = o.e.band(src, three);
        let a = o.fbin(S::FSub, t_denom, i_num);
        let rest = o.fbin(S::FSub, a, j_num);
        let (one, two) = (o.u(1), o.u(2));
        let is1 = o.e.ieq(src, one);
        let is2 = o.e.ieq(src, two);
        let inner = o.self_f(is2, j_num, rest);
        o.self_f(is1, i_num, inner)
    };
    let b0 = barycentric(o, 0);
    let b2 = barycentric(o, 2);
    let hit = o.e.lnot(missed);
    let (one_u, zero_u) = (o.u(1), o.u(0));
    let hit_u = o.sel_u(hit, one_u, zero_u);
    let b0u = o.as_u(b0);
    let b2u = o.as_u(b2);
    let r2 = o.sel_u(q.barycentrics, b0u, triangle_id);
    let r3 = o.sel_u(q.barycentrics, b2u, hit_u);
    let (r0, r1) = (o.as_u(t_num), o.as_u(t_denom));
    let t4 = o.e.t_uvec(4);
    Ok(o.e.construct(t4, &[r0, r1, r2, r3]))
}

fn boxes(o: &mut Ops, q: &Query, fp16: bool, window: Option<(Word, Word)>) -> R<Word> {
    let d = o.e.device_load_dwords(q.node_address, if fp16 { 16 } else { 28 })?;
    let bound = |o: &mut Ops, child: usize, index: usize| {
        let position = child * 6 + index;
        if fp16 {
            o.half_at(d[4 + position / 2], position % 2 == 1)
        } else {
            o.as_f(d[4 + position])
        }
    };
    let nan_max = |o: &mut Ops, a: Word, b: Word| {
        let n = o.is_nan(a);
        let g = o.cmp(S::FOrdGreaterThan, a, b);
        let c = o.e.lor(n, g);
        o.self_f(c, a, b)
    };
    let nan_min = |o: &mut Ops, a: Word, b: Word| {
        let n = o.is_nan(a);
        let l = o.cmp(S::FOrdLessThan, a, b);
        let c = o.e.lor(n, l);
        o.self_f(c, a, b)
    };
    let zero = o.f(0.0);
    let tf = o.tf;
    let grow = o.e.ins(S::ConvertUToF, tf, &[q.box_grow]);
    let ulp = o.f(f32::from_bits(0x3380_0000));
    let grow = o.fbin(S::FMul, grow, ulp);
    let one = o.f(1.0);
    let grow_factor = o.fbin(S::FAdd, one, grow);
    let invalid = o.u(INVALID);
    let mut children = [0; 4];
    let mut keys = [0; 4];
    for child in 0..4 {
        let (mut enter, mut leave) = (0, 0);
        for axis in 0..3 {
            let lo = bound(o, child, axis);
            let hi = bound(o, child, axis + 3);
            let a = o.fbin(S::FSub, lo, q.origin[axis]);
            let plane_min = o.fbin(S::FMul, a, q.inverse[axis]);
            let b = o.fbin(S::FSub, hi, q.origin[axis]);
            let plane_max = o.fbin(S::FMul, b, q.inverse[axis]);
            let positive = o.cmp(S::FOrdGreaterThanEqual, q.inverse[axis], zero);
            let near = o.self_f(positive, plane_min, plane_max);
            let far = o.self_f(positive, plane_max, plane_min);
            if axis == 0 {
                enter = near;
                leave = far;
            } else {
                enter = nan_max(o, enter, near);
                leave = nan_min(o, leave, far);
            }
        }
        let ne = o.is_nan(enter);
        let nl = o.is_nan(leave);
        let nan = o.e.lor(ne, nl);
        let pos = o.cmp(S::FOrdGreaterThan, enter, zero);
        let clamped_enter = o.self_f(pos, enter, zero);
        let inf = o.f(f32::INFINITY);
        let min_t = o.self_f(nan, inf, clamped_enter);
        let below = o.cmp(S::FOrdLessThan, leave, q.extent);
        let clamped_leave = o.self_f(below, leave, q.extent);
        let ninf = o.f(f32::NEG_INFINITY);
        let max_t = o.self_f(nan, ninf, clamped_leave);
        let grown = o.fbin(S::FMul, max_t, grow_factor);
        let mut hit = o.cmp(S::FOrdLessThanEqual, min_t, grown);
        if let Some((lo, hi)) = window {
            // Some hit triangle of the lane's ray lies within the child's
            // span (an unset window is [-inf, +inf]).
            let tf = o.tf;
            let a = o.e.gl(tf, GLOp::FMax, &[min_t, lo]);
            let b = o.e.gl(tf, GLOp::FMin, &[grown, hi]);
            let overlap = o.cmp(S::FOrdLessThanEqual, a, b);
            hit = o.e.land(hit, overlap);
        }
        children[child] = o.sel_u(hit, d[child], invalid);
        keys[child] = min_t;
    }
    let mut sorted = children;
    for (a, b) in [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)] {
        let b_valid = o.e.ine(sorted[b], invalid);
        let a_invalid = o.e.ieq(sorted[a], invalid);
        let closer = o.cmp(S::FOrdLessThan, keys[b], keys[a]);
        let both = o.e.land(b_valid, closer);
        let swap = o.e.lor(both, a_invalid);
        let sa = o.sel_u(swap, sorted[b], sorted[a]);
        let sb = o.sel_u(swap, sorted[a], sorted[b]);
        let ka = o.self_f(swap, keys[b], keys[a]);
        let kb = o.self_f(swap, keys[a], keys[b]);
        sorted[a] = sa;
        sorted[b] = sb;
        keys[a] = ka;
        keys[b] = kb;
    }
    for child in 0..4 {
        children[child] = o.sel_u(q.box_sort, sorted[child], children[child]);
    }
    let t4 = o.e.t_uvec(4);
    Ok(o.e.construct(t4, &children))
}

/// Emit one BVH query (`v`: descriptor, two unused operands, the address
/// dwords, exec).
pub fn emit(e: &mut Emitter, v: Value, d: &ValueData, attr: ImageAttr) -> R<Option<Word>> {
    let ImageOp::BvhIntersect { bvh64, boxes: with_boxes } = attr.op else { return Err("not a BVH query".into()) };
    let n = d.args.len();
    let exec = e.arg(v, n - 1)?;
    let descriptor = e.arg(v, 0)?;
    let comps: Vec<Word> = (3..3 + attr.address_dwords as usize).map(|k| e.arg(v, k)).collect::<R<_>>()?;
    let a16 = attr.a16;
    let ray = if bvh64 { 2 } else { 1 };
    let need = ray + 4 + if a16 { 3 } else { 6 };
    if comps.len() < need {
        return Err(format!("BVH query with {} address dwords", comps.len()));
    }
    let (tu, tf, tb, t64) = (e.t_u32(), e.t_f32(), e.t_bool(), e.ty(Type::U64));
    let t4 = e.t_uvec(4);
    let invalid = e.cu(INVALID);
    let none = e.construct(t4, &[invalid; 4]);
    let result = {
        let mut o = Ops { e, tu, tf, tb, t64 };
        let words = [0u32, 1, 2, 3].map(|i| o.e.extract(tu, descriptor, i));
        let node_lo = comps[0];
        let node = if bvh64 { o.pair(node_lo, comps[1]) } else { o.widen(node_lo) };
        let seven = o.u(7);
        let ty = o.e.band(node_lo, seven);
        let three64 = o.e.c64(3);
        let node_index = o.op(S::ShiftRightLogical, t64, node, three64);
        let m8 = o.u(0xFF);
        let w1_lo = o.e.band(words[1], m8);
        let base_units = o.pair(words[0], w1_lo);
        let m10 = o.u(0x3FF);
        let w3_lo = o.e.band(words[3], m10);
        let last_node = o.pair(words[2], w3_lo);
        let s23 = o.u(23);
        let g = o.e.shr(words[1], s23);
        let box_grow = o.e.band(g, m8);
        let top = o.u(0x8000_0000);
        let zero = o.u(0);
        let sort_bit = o.e.band(words[1], top);
        let box_sort = o.e.ine(sort_bit, zero);
        let bary_mask = o.u(1 << 24);
        let bary_bit = o.e.band(words[3], bary_mask);
        let barycentrics = o.e.ine(bary_bit, zero);
        let eight64 = o.e.c64(8);
        let base_bytes = o.op(S::ShiftLeftLogical, t64, base_units, eight64);
        let not7 = o.e.c64(!7u64);
        let aligned = o.op(S::BitwiseAnd, t64, node, not7);
        let offset = o.op(S::ShiftLeftLogical, t64, aligned, three64);
        let node_address = o.op(S::IAdd, t64, base_bytes, offset);
        let zero64 = o.e.c64(0);
        let based = o.op(S::INotEqual, tb, base_units, zero64);
        let within = |o: &mut Ops, cmp: S| {
            let in_range = o.op(cmp, tb, node_index, last_node);
            if bvh64 {
                in_range
            } else {
                o.e.land(based, in_range)
            }
        };
        let valid = within(&mut o, S::ULessThanEqual);
        let below = within(&mut o, S::ULessThan);
        let three = o.u(3);
        let tri_ty = o.e.ule(ty, three);
        let is_triangle = o.e.land(valid, tri_ty);
        let four = o.u(4);
        let b16 = o.e.ieq(ty, four);
        let is_box16 = o.e.land(valid, b16);
        let five = o.u(5);
        let b32 = o.e.ieq(ty, five);
        let is_box32 = o.e.land(below, b32);
        let extent = o.as_f(comps[ray]);
        let mut origin = [0; 3];
        let mut direction = [0; 3];
        let mut inverse = [0; 3];
        for axis in 0..3 {
            origin[axis] = o.as_f(comps[ray + 1 + axis]);
            if a16 {
                direction[axis] = o.half_at(comps[ray + 4 + axis / 2], axis % 2 == 1);
                let k = 3 + axis;
                inverse[axis] = o.half_at(comps[ray + 4 + k / 2], k % 2 == 1);
            } else {
                direction[axis] = o.as_f(comps[ray + 4 + axis]);
                inverse[axis] = o.as_f(comps[ray + 7 + axis]);
            }
        }
        let q = Query {
            words,
            ty,
            node_address,
            is_triangle,
            is_box16,
            is_box32,
            box_grow,
            box_sort,
            barycentrics,
            extent,
            origin,
            direction,
            inverse,
        };
        let _ = q.words;
        let e = o.e;
        // Compute stages only: the mirrors are brought up to date before
        // dispatches (a draw may run inside a rendering scope).
        let hw = with_boxes && !bvh64 && e.feat.emit.hw_rt && e.feat.emit.hw_rt_prune && e.stage == ps5shader::resources::ShaderStage::Compute && e.input.request.bindings.find(DescriptorBindingKind::FaultBuffer).is_some();
        let mut err: Option<String> = None;
        let r = e.if_value(exec, t4, none, |e| {
            let window = if hw { Some(rt_window(e, &q, node_lo, words[0], w1_lo)) } else { None };
            let o = Ops { e, tu, tf, tb, t64 };
            let tested = o.e.if_value(q.is_triangle, t4, none, |e| {
                let mut o = Ops { e, tu, tf, tb, t64 };
                triangle(&mut o, &q).unwrap_or_else(|m| {
                    err = Some(m);
                    none
                })
            });
            if !with_boxes {
                // A hardware traversal found the nodes: a box returns no
                // children.
                return tested;
            }
            let box16 = o.e.if_value(q.is_box16, t4, tested, |e| {
                let mut o = Ops { e, tu, tf, tb, t64 };
                boxes(&mut o, &q, true, window).unwrap_or_else(|m| {
                    err.get_or_insert(m);
                    tested
                })
            });
            o.e.if_value(q.is_box32, t4, box16, |e| {
                let mut o = Ops { e, tu, tf, tb, t64 };
                let r = boxes(&mut o, &q, false, window).unwrap_or_else(|m| {
                    err.get_or_insert(m);
                    box16
                });
                if hw && o.e.feat.emit.hw_rt_stats {
                    // Box queries and the children they return.
                    let invalid = o.u(INVALID);
                    let mut n = o.u(0);
                    for k in 0..4 {
                        let c = o.e.extract(tu, r, k);
                        let kept = o.e.ine(c, invalid);
                        let (one, zero) = (o.u(1), o.u(0));
                        let add = o.sel_u(kept, one, zero);
                        n = o.e.iadd(n, add);
                    }
                    if let Some(table) = o.e.rt_table() {
                        let one = o.u(1);
                        count_by(o.e, table, 4, one);
                        count_by(o.e, table, 5, n);
                    }
                }
                r
            })
        });
        if let Some(m) = err {
            return Err(m);
        }
        r
    };
    Ok(Some(result))
}

/// The lane half's traversal state: hit interval low and high (f32),
/// the extent bits of the ray it belongs to, and its table entry + 1
/// (0: no state, nothing is dropped).
fn rt_state(e: &mut Emitter) -> [Word; 4] {
    let h = e.half.min(1);
    if let Some(s) = e.res.rt_state[h] {
        return s;
    }
    let (tf, tu) = (e.t_f32(), e.t_u32());
    let init = [e.cf(f32::INFINITY), e.cf(f32::NEG_INFINITY), e.cu(0), e.cu(0)];
    let tys = [tf, tf, tu, tu];
    let vars = [0, 1, 2, 3].map(|i| e.global_init(tys[i], SC::Private, init[i]));
    e.res.rt_state[h] = Some(vars);
    vars
}

/// Word `index` of the ray-tracing table.
fn table_word(e: &mut Emitter, table: Word, index: Word) -> Word {
    let tu = e.t_u32();
    let p = e.ptr(SC::StorageBuffer, tu);
    let zero = e.cu(0);
    let base = e.cu(rtt::BASE);
    let at = e.iadd(index, base);
    let a = e.access(p, table, &[zero, at]);
    e.load(tu, a)
}

fn private_load(e: &mut Emitter, ty: Word, var: Word) -> Word {
    e.load(ty, var)
}

/// The window `[lo, hi]` a child box must overlap to stay in the result.
///
/// At a root query (BOX32 at the first node) of a TLAS the host mirrors,
/// a ray query over the host TLAS collects the `t` of every triangle hit
/// in `[0, extent]` into the state. At the root of a BLAS of the state's
/// TLAS the state stays; at any other root it is dropped (and an unknown
/// BVH is reported in the feedback ring). Queries of the state's ray (same
/// extent) use its interval, widened by the tolerance; others keep every
/// child.
fn rt_window(e: &mut Emitter, q: &Query, node_lo: Word, base_lo: Word, base_hi: Word) -> (Word, Word) {
    e.res.host_bvh = true;
    let (tu, tf, tb) = (e.t_u32(), e.t_f32(), e.t_bool());
    let state = rt_state(e);
    let Some(table) = e.rt_table() else {
        let (a, b) = (e.cf(f32::NEG_INFINITY), e.cf(f32::INFINITY));
        return (a, b);
    };
    let extent_bits = e.bitcast(tu, q.extent);
    let root_ptr = e.cu(ROOT_NODE_POINTER);
    let at_root = e.ieq(node_lo, root_ptr);
    let root = e.land(q.is_box32, at_root);
    e.if_then(root, |e| {
        // The table entry of this BVH (0: none).
        let mut index = e.cu(0);
        let mut flags = e.cu(0);
        let key_mask = e.cu(0x1FF);
        let known = e.cu(rtt::KNOWN);
        let want = e.bor(base_hi, known);
        for k in 0..rtt::ENTRY_COUNT {
            let at = rtt::ENTRIES + k * rtt::ENTRY_WORDS;
            let i0 = e.cu(at + rtt::WORD_BASE);
            let w0 = table_word(e, table, i0);
            let i1 = e.cu(at + rtt::WORD_FLAGS);
            let w1 = table_word(e, table, i1);
            let key = e.band(w1, key_mask);
            let a = e.ieq(w0, base_lo);
            let b = e.ieq(key, want);
            let m = e.land(a, b);
            let n = e.cu(k + 1);
            index = e.select(tu, m, n, index);
            flags = e.select(tu, m, w1, flags);
        }
        let zero = e.cu(0);
        let tlas_bit = e.cu(rtt::TLAS);
        let tlas_flag = e.band(flags, tlas_bit);
        let is_tlas = e.ine(tlas_flag, zero);
        e.if_else_values(
            is_tlas,
            tu,
            0,
            |e| {
                // The entry is in use.
                let one = e.cu(1);
                let slot = e.isub(index, one);
                let words = e.cu(rtt::ENTRY_WORDS);
                let off = e.imul(slot, words);
                let used = e.cu(rtt::BASE + rtt::ENTRIES + rtt::WORD_USED);
                let i = e.iadd(off, used);
                let tu = e.t_u32();
                let p = e.ptr(SC::StorageBuffer, tu);
                let zero = e.cu(0);
                let a = e.access(p, table, &[zero, i]);
                e.store(a, one);
                e.writes_buffers = true;
                let valid_bit = e.cu(rtt::VALID);
                let vf = e.band(flags, valid_bit);
                let valid = e.ine(vf, zero);
                // Vulkan ray parameters: finite origin and direction,
                // 0 <= extent (no NaN).
                let z = e.cf(0.0);
                let mut finite = e.ins(S::FOrdGreaterThanEqual, tb, &[q.extent, z]);
                for &c in q.origin.iter().chain(q.direction.iter()) {
                    let a = e.gl(tf, GLOp::FAbs, &[c]);
                    let inf = e.cf(f32::INFINITY);
                    let ok = e.ins(S::FOrdLessThan, tb, &[a, inf]);
                    finite = e.land(finite, ok);
                }
                let go = e.land(valid, finite);
                e.if_else_values(
                    go,
                    tu,
                    0,
                    |e| {
                        let one = e.cu(1);
                        let slot = e.isub(index, one);
                        let words = e.cu(rtt::ENTRY_WORDS);
                        let off = e.imul(slot, words);
                        let first = e.cu(rtt::ENTRIES + rtt::WORD_AS);
                        let i = e.iadd(off, first);
                        let lo = table_word(e, table, i);
                        let i = e.iadd(i, one);
                        let hi = table_word(e, table, i);
                        let address = e.join64(lo, hi);
                        let (t_lo, t_hi) = ray_query_hits(e, address, q);
                        count(e, table, 0);
                        let inf = e.cf(f32::INFINITY);
                        let tb = e.t_bool();
                        let none = e.ins(S::FOrdEqual, tb, &[t_lo, inf]);
                        e.if_then(none, |e| count(e, table, 3));
                        e.store(state[0], t_lo);
                        e.store(state[1], t_hi);
                        e.store(state[2], extent_bits);
                        e.store(state[3], index);
                        vec![]
                    },
                    |e| {
                        let zero = e.cu(0);
                        e.store(state[3], zero);
                        count(e, table, 1);
                        vec![]
                    },
                );
                vec![]
            },
            |e| {
                // A BLAS of the state's TLAS keeps the state.
                let cur = private_load(e, tu, state[3]);
                let ext = private_load(e, tu, state[2]);
                let set = e.ine(cur, zero);
                let same = e.ieq(ext, extent_bits);
                let traced = e.land(set, same);
                let tb = e.t_bool();
                let f = e.cb(false);
                let member = e.if_value(traced, tb, f, |e| blas_member(e, table, cur, base_lo, base_hi));
                let drop = e.lnot(member);
                e.if_then(drop, |e| {
                    let zero = e.cu(0);
                    e.store(state[3], zero);
                    count(e, table, 2);
                    let unknown = e.ieq(index, zero);
                    e.if_then(unknown, |e| feedback(e, table, base_lo, base_hi));
                });
                vec![]
            },
        );
    });
    let cur = private_load(e, tu, state[3]);
    let ext = private_load(e, tu, state[2]);
    let zero = e.cu(0);
    let set = e.ine(cur, zero);
    let same = e.ieq(ext, extent_bits);
    let active = e.land(set, same);
    let lo = private_load(e, tf, state[0]);
    let hi = private_load(e, tf, state[1]);
    // Tolerance: relative to the distance, plus a small absolute part.
    let rel = e.cf(T_RELATIVE);
    let abs = e.cf(T_ABSOLUTE);
    let alo = e.gl(tf, GLOp::FAbs, &[lo]);
    let elo = e.gl(tf, GLOp::Fma, &[alo, rel, abs]);
    let lo_w = e.ins(S::FSub, tf, &[lo, elo]);
    let ahi = e.gl(tf, GLOp::FAbs, &[hi]);
    let ehi = e.gl(tf, GLOp::Fma, &[ahi, rel, abs]);
    let hi_w = e.ins(S::FAdd, tf, &[hi, ehi]);
    let (ninf, pinf) = (e.cf(f32::NEG_INFINITY), e.cf(f32::INFINITY));
    // No hit at all: an empty window (inf - inf would be NaN).
    let tb = e.t_bool();
    let empty = e.ins(S::FOrdGreaterThan, tb, &[lo, hi]);
    let lo_w = e.select(tf, empty, pinf, lo_w);
    let hi_w = e.select(tf, empty, ninf, hi_w);
    let lo_e = e.select(tf, active, lo_w, ninf);
    let hi_e = e.select(tf, active, hi_w, pinf);
    (lo_e, hi_e)
}

/// The BLAS set of table entry `entry` (1-based) holds `base`.
fn blas_member(e: &mut Emitter, table: Word, entry: Word, base_lo: Word, base_hi: Word) -> Word {
    let (tu, tb) = (e.t_u32(), e.t_bool());
    let t64 = e.ty(Type::U64);
    let one = e.cu(1);
    let slot = e.isub(entry, one);
    let words = e.cu(rtt::ENTRY_WORDS);
    let off = e.imul(slot, words);
    let first = e.cu(rtt::ENTRIES + rtt::WORD_SET);
    let i = e.iadd(off, first);
    let lo = table_word(e, table, i);
    let i1 = e.iadd(i, one);
    let hi = table_word(e, table, i1);
    let two = e.cu(2);
    let i2 = e.iadd(i, two);
    let mask = table_word(e, table, i2);
    let set = e.join64(lo, hi);
    let base = e.join64(base_lo, base_hi);
    let one64 = e.c64(1);
    let key = e.ins(S::IAdd, t64, &[base, one64]);
    let (key_lo, key_hi) = e.split64(key);
    let mult = e.cu(0x9E37_79B1);
    let h = e.imul(base_lo, mult);
    let twelve = e.cu(12);
    let mut h = e.shr(h, twelve);
    let mut found = e.cb(false);
    let mut open = e.cb(true);
    let zero = e.cu(0);
    let eight = e.c64(rtt::SET_WORDS as u64 * 4);
    let four = e.c64(4);
    for _ in 0..rtt::SET_PROBES {
        let s = e.band(h, mask);
        let s64 = e.ins(S::UConvert, t64, &[s]);
        let off = e.ins(S::IMul, t64, &[s64, eight]);
        let a = e.ins(S::IAdd, t64, &[set, off]);
        let p = e.device_word_ptr(a);
        let w0 = e.ins_ops(S::Load, tu, vec![Operand::IdRef(p), Operand::MemoryAccess(rspirv::spirv::MemoryAccess::ALIGNED), Operand::LiteralBit32(4)]);
        let a1 = e.ins(S::IAdd, t64, &[a, four]);
        let p1 = e.device_word_ptr(a1);
        let w1 = e.ins_ops(S::Load, tu, vec![Operand::IdRef(p1), Operand::MemoryAccess(rspirv::spirv::MemoryAccess::ALIGNED), Operand::LiteralBit32(4)]);
        let m0 = e.ieq(w0, key_lo);
        let m1 = e.ieq(w1, key_hi);
        let m = e.land(m0, m1);
        let hit = e.land(open, m);
        found = e.lor(found, hit);
        let e0 = e.ieq(w0, zero);
        let e1 = e.ieq(w1, zero);
        let empty = e.land(e0, e1);
        let stop = e.lor(empty, m);
        let go = e.lnot(stop);
        open = e.land(open, go);
        h = e.iadd(h, one);
    }
    let _ = tb;
    found
}

/// Diagnostics (`RPS5E_HW_RT=2`): count an event of the pruning.
fn count(e: &mut Emitter, table: Word, which: u32) {
    let one = e.cu(1);
    count_by(e, table, which, one);
}

fn count_by(e: &mut Emitter, table: Word, which: u32, by: Word) {
    if !e.feat.emit.hw_rt_stats {
        return;
    }
    let tu = e.t_u32();
    let p = e.ptr(SC::StorageBuffer, tu);
    let (zero, i) = (e.cu(0), e.cu(rtt::BASE + rtt::STATS + which));
    let a = e.access(p, table, &[zero, i]);
    // One atomic per host subgroup: the sum of its invocations.
    e.cap(Capability::GroupNonUniform);
    e.cap(Capability::GroupNonUniformArithmetic);
    let sg = e.cu(3);
    let sum = e.ins_ops(S::GroupNonUniformIAdd, tu, vec![Operand::IdRef(sg), Operand::GroupOperation(rspirv::spirv::GroupOperation::Reduce), Operand::IdRef(by)]);
    let tb = e.t_bool();
    let first = e.ins(S::GroupNonUniformElect, tb, &[sg]);
    e.if_then(first, |e| {
        let tu = e.t_u32();
        let (scope, semantics) = (e.cu(super::emit::SCOPE_DEVICE), e.cu(0));
        e.ins(S::AtomicIAdd, tu, &[a, scope, semantics, sum]);
    });
    e.writes_buffers = true;
}

/// Report `base` in the feedback ring unless its filter bit is set.
fn feedback(e: &mut Emitter, table: Word, base_lo: Word, base_hi: Word) {
    let tu = e.t_u32();
    let p = e.ptr(SC::StorageBuffer, tu);
    let zero = e.cu(0);
    let mult = e.cu(0x9E37_79B1);
    let h = e.imul(base_lo, mult);
    let twelve = e.cu(12);
    let h = e.shr(h, twelve);
    let m = e.cu(rtt::FILTER_WORDS * 32 - 1);
    let h = e.band(h, m);
    let five = e.cu(5);
    let wi = e.shr(h, five);
    let first = e.cu(rtt::BASE + rtt::FILTER);
    let wi = e.iadd(wi, first);
    let fa = e.access(p, table, &[zero, wi]);
    let thirty_one = e.cu(31);
    let b = e.band(h, thirty_one);
    let one = e.cu(1);
    let bit = e.shl(one, b);
    let w = e.load(tu, fa);
    let has = e.band(w, bit);
    let clear = e.ieq(has, zero);
    e.if_then(clear, |e| {
        let tu = e.t_u32();
        let (scope, semantics) = (e.cu(super::emit::SCOPE_DEVICE), e.cu(0));
        let old = e.ins(S::AtomicOr, tu, &[fa, scope, semantics, bit]);
        let zero = e.cu(0);
        let was = e.band(old, bit);
        let first = e.ieq(was, zero);
        e.if_then(first, |e| report(e, table, base_lo, base_hi));
    });
    e.writes_buffers = true;
}

fn report(e: &mut Emitter, table: Word, base_lo: Word, base_hi: Word) {
    let tu = e.t_u32();
    let p = e.ptr(SC::StorageBuffer, tu);
    let zero = e.cu(0);
    let ci = e.cu(rtt::BASE + rtt::FEEDBACK_COUNT);
    let counter = e.access(p, table, &[zero, ci]);
    let (scope, semantics, one) = (e.cu(super::emit::SCOPE_DEVICE), e.cu(0), e.cu(1));
    let n = e.ins(S::AtomicIAdd, tu, &[counter, scope, semantics, one]);
    let m = e.cu(rtt::FEEDBACK_SLOTS - 1);
    let slot = e.band(n, m);
    let two = e.cu(2);
    let off = e.imul(slot, two);
    let first = e.cu(rtt::BASE + rtt::FEEDBACK);
    let i = e.iadd(off, first);
    let a = e.access(p, table, &[zero, i]);
    e.store(a, base_lo);
    let i1 = e.iadd(i, one);
    let a1 = e.access(p, table, &[zero, i1]);
    let marked = e.cu(1 << 31);
    let hi = e.bor(base_hi, marked);
    e.store(a1, hi);
    e.writes_buffers = true;
}

/// Ray query over the host TLAS at `address`: the smallest and largest `t`
/// of the triangles the query's ray hits in `[0, extent]` (+inf / -inf:
/// none). Every triangle is a candidate (no opaque ones); none is
/// committed, so the query visits them all.
fn ray_query_hits(e: &mut Emitter, address: Word, q: &Query) -> (Word, Word) {
    e.cap(Capability::RayQueryKHR);
    e.exts.insert("SPV_KHR_ray_query");
    let (tu, tf, tb) = (e.t_u32(), e.t_f32(), e.t_bool());
    let rq_ty = e.b.type_ray_query_khr();
    let as_ty = e.b.type_acceleration_structure_khr();
    let rq = {
        let p = e.ptr(SC::Private, rq_ty);
        let v = e.b.id();
        let inst = rspirv::dr::Instruction::new(S::Variable, Some(p), Some(v), vec![Operand::StorageClass(SC::Private)]);
        e.b.module_mut().types_global_values.push(inst);
        e.note_global(v);
        v
    };
    let accel = e.ins(S::ConvertUToAccelerationStructureKHR, as_ty, &[address]);
    let v3 = e.t_fvec(3);
    let origin = e.construct(v3, &q.origin);
    let direction = e.construct(v3, &q.direction);
    // NoOpaque: every triangle is a candidate.
    let flags = e.cu(RAY_NO_OPAQUE);
    let mask = e.cu(0xFF);
    let t_min = e.cf(0.0);
    e.void(
        S::RayQueryInitializeKHR,
        vec![
            Operand::IdRef(rq),
            Operand::IdRef(accel),
            Operand::IdRef(flags),
            Operand::IdRef(mask),
            Operand::IdRef(origin),
            Operand::IdRef(t_min),
            Operand::IdRef(direction),
            Operand::IdRef(q.extent),
        ],
    );
    let lo_var = e.res.rt_state[e.half.min(1)].unwrap()[0];
    let hi_var = e.res.rt_state[e.half.min(1)].unwrap()[1];
    let (inf, ninf) = (e.cf(f32::INFINITY), e.cf(f32::NEG_INFINITY));
    e.store(lo_var, inf);
    e.store(hi_var, ninf);
    let (header, check, body, cont, merge) = (e.b.id(), e.b.id(), e.b.id(), e.b.id(), e.b.id());
    e.b.branch(header).unwrap();
    e.begin(header);
    e.open_scope();
    e.b.loop_merge(merge, cont, rspirv::spirv::LoopControl::NONE, []).unwrap();
    e.b.branch(check).unwrap();
    e.begin(check);
    let more = e.ins(S::RayQueryProceedKHR, tb, &[rq]);
    e.b.branch_conditional(more, body, merge, []).unwrap();
    e.begin(body);
    let candidate = e.cu(0);
    // A box candidate stands for a BLAS the host has not mirrored: the
    // whole ray stays (nothing is dropped for this lane).
    let kind = e.ins(S::RayQueryGetIntersectionTypeKHR, tu, &[rq, candidate]);
    let one = e.cu(1);
    let is_box = e.ieq(kind, one);
    let tri = e.lnot(is_box);
    let zero = e.cf(0.0);
    let t = e.if_value(tri, tf, zero, |e| {
        let tf = e.t_f32();
        let candidate = e.cu(0);
        e.ins(S::RayQueryGetIntersectionTKHR, tf, &[rq, candidate])
    });
    let inf = e.cf(f32::INFINITY);
    let t_lo = e.select(tf, is_box, zero, t);
    let t_hi = e.select(tf, is_box, inf, t);
    let lo = e.load(tf, lo_var);
    let hi = e.load(tf, hi_var);
    let lo = e.gl(tf, GLOp::FMin, &[lo, t_lo]);
    let hi = e.gl(tf, GLOp::FMax, &[hi, t_hi]);
    e.store(lo_var, lo);
    e.store(hi_var, hi);
    e.if_then(is_box, |e| e.void(S::RayQueryTerminateKHR, vec![Operand::IdRef(rq)]));
    e.b.branch(cont).unwrap();
    e.begin(cont);
    e.b.branch(header).unwrap();
    e.close_scope();
    e.begin(merge);
    let _ = tu;
    let lo = e.load(tf, lo_var);
    let hi = e.load(tf, hi_var);
    (lo, hi)
}

// ----------------------------------------------------------------------
// Hardware traversals (`RtOp`, from `ps5shader::opt::rt_loops`)
// ----------------------------------------------------------------------

/// Take the query's candidate (`which`: 0 candidate, 1 committed) unless it
/// lies in the leaf list the guest last processed for the invocation:
/// its instance (the record index), its leaf list's first entry (looked up
/// in the BLAS's list starts through the table's starts map and the
/// instance's custom index, the map slot) and leaf base. Returns whether it
/// was taken.
fn take_candidate(e: &mut Emitter, rq: Word, which: u32, slots: (Word, Word, Word, Word, Word, Word, Word)) -> Word {
    let (inst, prim, base, ready, last_inst, last_list, proxy) = slots;
    let (tu, tb) = (e.t_u32(), e.t_bool());
    let t64 = e.ty(Type::U64);
    let w = e.cu(which);
    let i = e.ins(S::RayQueryGetIntersectionInstanceIdKHR, tu, &[rq, w]);
    let slot = e.ins(S::RayQueryGetIntersectionInstanceCustomIndexKHR, tu, &[rq, w]);
    let p = e.ins(S::RayQueryGetIntersectionPrimitiveIndexKHR, tu, &[rq, w]);
    let b = e.ins(S::RayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR, tu, &[rq, w]);
    let Some(table) = e.rt_table() else { return e.cb(false) };
    let i0 = e.cu(rtt::STARTS);
    let lo = table_word(e, table, i0);
    let i1 = e.cu(rtt::STARTS + 1);
    let hi = table_word(e, table, i1);
    let map = e.join64(lo, hi);
    let load = |e: &mut Emitter, a: Word| {
        let tu = e.t_u32();
        let ptr = e.device_word_ptr(a);
        e.ins_ops(S::Load, tu, vec![Operand::IdRef(ptr), Operand::MemoryAccess(rspirv::spirv::MemoryAccess::ALIGNED), Operand::LiteralBit32(4)])
    };
    let s64 = e.ins(S::UConvert, t64, &[slot]);
    let eight = e.c64(8);
    let off = e.ins(S::IMul, t64, &[s64, eight]);
    let at = e.ins(S::IAdd, t64, &[map, off]);
    let a_lo = load(e, at);
    let four = e.c64(4);
    let at1 = e.ins(S::IAdd, t64, &[at, four]);
    let a_hi = load(e, at1);
    let starts = e.join64(a_lo, a_hi);
    let p64 = e.ins(S::UConvert, t64, &[p]);
    let off = e.ins(S::IMul, t64, &[p64, four]);
    let at = e.ins(S::IAdd, t64, &[starts, off]);
    let first = load(e, at);
    let li = e.load(tu, last_inst);
    let ll = e.load(tu, last_list);
    let same_i = e.ieq(li, i);
    let same_l = e.ieq(ll, first);
    let seen = e.land(same_i, same_l);
    let fresh = e.lnot(seen);
    e.if_then(fresh, |e| {
        e.store(inst, i);
        e.store(prim, first);
        e.store(base, b);
        let (t, f) = (e.cb(true), e.cb(false));
        e.store(ready, t);
        e.store(proxy, f);
    });
    let _ = tb;
    fresh
}

/// Take a proxy box candidate (an instance whose BLAS the host lacks: the
/// guest's stack walks it) unless the guest walked that instance for the
/// invocation already. Returns whether it was taken.
fn take_proxy(e: &mut Emitter, rq: Word, slots: (Word, Word, Word, Word, Word, Word, Word)) -> Word {
    let (inst, prim, _, ready, last_inst, last_list, proxy) = slots;
    let tu = e.t_u32();
    let w = e.cu(CANDIDATE);
    let i = e.ins(S::RayQueryGetIntersectionInstanceIdKHR, tu, &[rq, w]);
    let li = e.load(tu, last_inst);
    let ll = e.load(tu, last_list);
    let same_i = e.ieq(li, i);
    let pl = e.cu(PROXY_LIST);
    let same_l = e.ieq(ll, pl);
    let seen = e.land(same_i, same_l);
    let fresh = e.lnot(seen);
    e.if_then(fresh, |e| {
        e.store(inst, i);
        let pl = e.cu(PROXY_LIST);
        e.store(prim, pl);
        let t = e.cb(true);
        e.store(ready, t);
        e.store(proxy, t);
    });
    fresh
}

/// Ray-query intersection kinds and ray flags (SPIR-V `SPV_KHR_ray_query`).
const CANDIDATE: u32 = 0;
const COMMITTED: u32 = 1;
const CANDIDATE_TRIANGLE: u32 = 0;
const CANDIDATE_AABB: u32 = 1;
/// `last_list` of an instance the guest's stack walked (its proxy box).
const PROXY_LIST: u32 = 0xFFFF_FFFE;
const COMMITTED_TRIANGLE: u32 = 1;
const RAY_NO_OPAQUE: u32 = 0x2;
/// The faces a guest facing test drops (`sign(t_num) == flip`): Vulkan
/// calls them front-facing.
const RAY_CULL_FRONT_FACING: u32 = 0x20;
const RAY_TERMINATE_ON_FIRST_HIT: u32 = 0x4;

fn closest_only() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_HW_RT_CLOSEST").is_ok_and(|v| v == "1"))
}
/// Candidates one invocation's traversal takes at most (a guard: a guest
/// loop rejecting candidates without end cannot hang the GPU).
const CANDIDATE_CAP: u32 = 1024;

/// The private state of hardware traversal `slot`: its ray query, done (no
/// candidate left), ready (a candidate fetched, not popped at the top),
/// leaf (a leaf entry waits for the bottom level), the candidate's
/// instance, primitive and leaf base (64-byte units), candidates taken,
/// inside (the top level popped its instance) and that instance, the
/// instance and leaf list the guest loop processed last for it, whether
/// the pending candidate is a proxy box (an instance without a host BLAS)
/// and whether the bottom level walks the guest's stack.
fn rt_slot(e: &mut Emitter, slot: u8) -> [Word; 14] {
    // Per guest lane: the lane halves of a paired invocation have their own.
    let key = slot as u16 * 2 + e.half.min(1) as u16;
    if let Some(&s) = e.res.rt_slots.get(&key) {
        return s;
    }
    e.cap(Capability::RayQueryKHR);
    e.exts.insert("SPV_KHR_ray_query");
    let rq_ty = e.b.type_ray_query_khr();
    let rq = {
        let p = e.ptr(SC::Private, rq_ty);
        let v = e.b.id();
        let inst = rspirv::dr::Instruction::new(S::Variable, Some(p), Some(v), vec![Operand::StorageClass(SC::Private)]);
        e.b.module_mut().types_global_values.push(inst);
        e.note_global(v);
        v
    };
    let (tb, tu) = (e.t_bool(), e.t_u32());
    let (f, z) = (e.cb(false), e.cu(0));
    let t = e.cb(true);
    let done = e.global_init(tb, SC::Private, t);
    let ready = e.global_init(tb, SC::Private, f);
    let leaf = e.global_init(tb, SC::Private, f);
    let inst = e.global_init(tu, SC::Private, z);
    let prim = e.global_init(tu, SC::Private, z);
    let base = e.global_init(tu, SC::Private, z);
    let taken = e.global_init(tu, SC::Private, z);
    let inside = e.global_init(tb, SC::Private, f);
    let here = e.global_init(tu, SC::Private, z);
    let none = e.cu(u32::MAX);
    let last_inst = e.global_init(tu, SC::Private, none);
    let last_list = e.global_init(tu, SC::Private, none);
    let proxy = e.global_init(tb, SC::Private, f);
    let in_guest = e.global_init(tb, SC::Private, f);
    let s = [rq, done, ready, leaf, inst, prim, base, taken, inside, here, last_inst, last_list, proxy, in_guest];
    e.res.rt_slots.insert(key, s);
    s
}

/// Emit one step of a hardware traversal.
pub fn emit_rt(e: &mut Emitter, v: Value, _d: &ValueData, op: RtOp) -> R<Option<Word>> {
    let (tu, tb) = (e.t_u32(), e.t_bool());
    match op {
        RtOp::Lookup { levels } => {
            e.res.host_bvh = true;
            let (lo, hi) = (e.arg(v, 0)?, e.arg(v, 1)?);
            let Some(table) = e.rt_table() else { return Ok(Some(e.cu(0))) };
            let m8 = e.cu(0xFF);
            let base_hi = e.band(hi, m8);
            let key_mask = e.cu(0x1FF);
            let known = e.cu(rtt::KNOWN);
            let want_key = e.bor(base_hi, known);
            let need = rtt::KNOWN | rtt::TLAS | rtt::VALID | rtt::COMPLETE | rtt::SINGLE;
            // A top level with instances whose BLAS the host lacks is fine
            // (the guest's stack walks those); a BLAS traced alone needs its
            // mirror.
            let want_flags = e.cu(rtt::KNOWN | rtt::TLAS | rtt::VALID | if levels == 1 { rtt::SINGLE | rtt::COMPLETE } else { 0 });
            let need = if levels == 1 { need } else { need & !rtt::COMPLETE };
            let need_mask = e.cu(need);
            let mut entry = e.cu(0);
            let mut known_entry = e.cu(0);
            let mut seen = e.cb(false);
            for k in 0..rtt::ENTRY_COUNT {
                let at = rtt::ENTRIES + k * rtt::ENTRY_WORDS;
                let i0 = e.cu(at + rtt::WORD_BASE);
                let w0 = table_word(e, table, i0);
                let i1 = e.cu(at + rtt::WORD_FLAGS);
                let w1 = table_word(e, table, i1);
                let key = e.band(w1, key_mask);
                let a = e.ieq(w0, lo);
                let b = e.ieq(key, want_key);
                let same = e.land(a, b);
                seen = e.lor(seen, same);
                let flags = e.band(w1, need_mask);
                let ok = e.ieq(flags, want_flags);
                let usable = e.land(same, ok);
                let n = e.cu(k + 1);
                entry = e.select(tu, usable, n, entry);
                known_entry = e.select(tu, same, n, known_entry);
            }
            // The entry of the BVH (usable or not) is in use.
            let zero = e.cu(0);
            let found = e.ine(known_entry, zero);
            e.if_then(found, |e| {
                let entry = known_entry;
                let one = e.cu(1);
                let slot = e.isub(entry, one);
                let words = e.cu(rtt::ENTRY_WORDS);
                let off = e.imul(slot, words);
                let used = e.cu(rtt::BASE + rtt::ENTRIES + rtt::WORD_USED);
                let i = e.iadd(off, used);
                let tu = e.t_u32();
                let p = e.ptr(SC::StorageBuffer, tu);
                let zero = e.cu(0);
                let a = e.access(p, table, &[zero, i]);
                e.store(a, one);
            });
            // A BVH the host does not know yet is reported.
            let unknown = e.lnot(seen);
            e.if_then(unknown, |e| feedback(e, table, lo, base_hi));
            e.writes_buffers = true;
            Ok(Some(entry))
        }
        RtOp::Begin { slot } => {
            let [rq, done, ready, leaf, _, _, _, taken, inside, _, last_inst, last_list, proxy, in_guest] = rt_slot(e, slot);
            let entry = e.arg(v, 0)?;
            let tf = e.t_f32();
            let ray: Vec<Word> = (1..8).map(|k| e.arg(v, k).map(|w| e.bitcast(tf, w))).collect::<R<_>>()?;
            let cull = e.arg(v, 8)?;
            let m8 = e.cu(0xFF);
            let cull = e.band(cull, m8);
            let active = e.arg(v, 9)?;
            let ray_flags = e.arg(v, 10)?;
            let z = e.cu(0);
            e.store(taken, z);
            let Some(table) = e.rt_table() else { return Err("hardware traversal without the ray-tracing table".into()) };
            let one = e.cu(1);
            let slot_i = e.isub(entry, one);
            let words = e.cu(rtt::ENTRY_WORDS);
            let off = e.imul(slot_i, words);
            let first = e.cu(rtt::ENTRIES + rtt::WORD_AS);
            let i = e.iadd(off, first);
            let lo = table_word(e, table, i);
            let i1 = e.iadd(i, one);
            let hi = table_word(e, table, i1);
            let address = e.join64(lo, hi);
            // An active invocation with Vulkan's ray parameters: finite
            // origin and direction, 0 <= extent (a NaN fails the compare).
            let z = e.cf(0.0);
            let mut finite = e.ins(S::FOrdGreaterThanEqual, tb, &[ray[6], z]);
            finite = e.land(finite, active);
            let inf = e.cf(f32::INFINITY);
            for &c in &ray[..6] {
                let a = e.gl(tf, GLOp::FAbs, &[c]);
                let ok = e.ins(S::FOrdLessThan, tb, &[a, inf]);
                finite = e.land(finite, ok);
            }
            let f = e.cb(false);
            e.store(ready, f);
            e.store(leaf, f);
            e.store(inside, f);
            e.store(proxy, f);
            e.store(in_guest, f);
            let none = e.cu(u32::MAX);
            e.store(last_inst, none);
            e.store(last_list, none);
            let stop = e.lnot(finite);
            e.store(done, stop);
            if e.feat.emit.hw_rt_stats {
                let one = e.cu(1);
                count_by(e, table, 6, one);
            }
            e.if_then(finite, |e| {
                let as_ty = e.b.type_acceleration_structure_khr();
                let accel = e.ins(S::ConvertUToAccelerationStructureKHR, as_ty, &[address]);
                let v3 = e.t_fvec(3);
                let origin = e.construct(v3, &ray[0..3]);
                let direction = e.construct(v3, &ray[3..6]);
                // (`RPS5E_HW_RT_CLOSEST=1`, diagnostics: never end at the first hit.)
                let allowed = if closest_only() { RAY_CULL_FRONT_FACING } else { RAY_CULL_FRONT_FACING | RAY_TERMINATE_ON_FIRST_HIT };
                let mask = e.cu(allowed);
                let flags = e.band(ray_flags, mask);
                let t_min = e.cf(0.0);
                e.void(
                    S::RayQueryInitializeKHR,
                    vec![
                        Operand::IdRef(rq),
                        Operand::IdRef(accel),
                        Operand::IdRef(flags),
                        Operand::IdRef(cull),
                        Operand::IdRef(origin),
                        Operand::IdRef(t_min),
                        Operand::IdRef(direction),
                        Operand::IdRef(ray[6]),
                    ],
                );
            });
            Ok(None)
        }
        RtOp::Fetch { slot, level } => {
            let [rq, done, ready, leaf, inst, prim, base, taken, inside, here, last_inst, last_list, proxy, _] = rt_slot(e, slot);
            let slots = (inst, prim, base, ready, last_inst, last_list, proxy);
            let d0 = e.load(tb, done);
            let r0 = e.load(tb, ready);
            let l0 = e.load(tb, leaf);
            let busy = e.lor(d0, r0);
            let busy = e.lor(busy, l0);
            let mut fetch = e.lnot(busy);
            if level > 0 {
                let i0 = e.load(tb, inside);
                fetch = e.land(fetch, i0);
            }
            e.if_then(fetch, |e| {
                // The next triangle candidate (a procedural box stands for
                // a BLAS the host lacks: a complete mirror has none, skip).
                let (header, body, cont, merge) = (e.b.id(), e.b.id(), e.b.id(), e.b.id());
                e.b.branch(header).unwrap();
                e.begin(header);
                e.open_scope();
                e.b.loop_merge(merge, cont, rspirv::spirv::LoopControl::NONE, []).unwrap();
                e.b.branch(body).unwrap();
                e.begin(body);
                let tb = e.t_bool();
                let tu = e.t_u32();
                // Past the cap the traversal ends (counted at level 2).
                let n = e.load(tu, taken);
                let one = e.cu(1);
                let n1 = e.iadd(n, one);
                e.store(taken, n1);
                let cap = e.cu(CANDIDATE_CAP);
                let room = e.ult(n, cap);
                if e.feat.emit.hw_rt_stats {
                    if let Some(table) = e.rt_table() {
                        let full = e.lnot(room);
                        let (one, zero) = (e.cu(1), e.cu(0));
                        let c = e.select(tu, full, one, zero);
                        count_by(e, table, 8, c);
                        let c = e.select(tu, room, one, zero);
                        count_by(e, table, 7, c);
                    }
                }
                let f = e.cb(false);
                let more = e.if_value(room, tb, f, |e| {
                    let tb = e.t_bool();
                    e.ins(S::RayQueryProceedKHR, tb, &[rq])
                });
                let cand = e.cu(CANDIDATE);
                let tri_kind = e.cu(CANDIDATE_TRIANGLE);
                // (No candidate: a kind no candidate has.)
                let other = e.cu(u32::MAX);
                let kind = e.if_value(more, tu, other, |e| {
                    let tu = e.t_u32();
                    e.ins(S::RayQueryGetIntersectionTypeKHR, tu, &[rq, cand])
                });
                let is_tri = e.ieq(kind, tri_kind);
                let aabb_kind = e.cu(CANDIDATE_AABB);
                let is_box = e.ieq(kind, aabb_kind);
                let f = e.cb(false);
                let took_tri = e.if_value(is_tri, tb, f, |e| take_candidate(e, rq, CANDIDATE, slots));
                let f = e.cb(false);
                let took_box = if level == 0 { e.if_value(is_box, tb, f, |e| take_proxy(e, rq, slots)) } else { f };
                let took = e.lor(took_tri, took_box);
                let finished = e.lnot(more);
                e.if_then(finished, |e| {
                    let t = e.cb(true);
                    e.store(done, t);
                    // The closest opaque hit the query committed is the last
                    // candidate (past the cap there is none).
                    let tu = e.t_u32();
                    let com = e.cu(COMMITTED);
                    let kind = e.ins(S::RayQueryGetIntersectionTypeKHR, tu, &[rq, com]);
                    let tri = e.cu(COMMITTED_TRIANGLE);
                    let hit = e.ieq(kind, tri);
                    e.if_then(hit, |e| {
                        take_candidate(e, rq, COMMITTED, slots);
                    });
                });
                let is_tri = took;
                let stop = if e.feat.emit.hw_rt_probe { finished } else { e.lor(finished, is_tri) };
                e.b.branch_conditional(stop, merge, cont, []).unwrap();
                e.begin(cont);
                e.b.branch(header).unwrap();
                e.close_scope();
                e.begin(merge);
                if level > 0 {
                    // A candidate in the popped instance stays at the bottom.
                    let tb = e.t_bool();
                    let tu = e.t_u32();
                    let r = e.load(tb, ready);
                    let i = e.load(tu, inst);
                    let h = e.load(tu, here);
                    let same = e.ieq(i, h);
                    let stay = e.land(r, same);
                    e.if_then(stay, |e| {
                        let (t, f) = (e.cb(true), e.cb(false));
                        e.store(ready, f);
                        e.store(leaf, t);
                    });
                }
            });
            Ok(None)
        }
        RtOp::Get { slot, what } => {
            let [_, _, ready, leaf, inst, prim, base, _, _, _, _, _, proxy, in_guest] = rt_slot(e, slot);
            let _ = (inst, prim);
            if e.feat.emit.hw_rt_probe && matches!(what, RtGet::Ready | RtGet::Leaf) {
                return Ok(Some(e.cb(false)));
            }
            Ok(Some(match what {
                RtGet::Ready => e.load(tb, ready),
                RtGet::Leaf => e.load(tb, leaf),
                RtGet::Instance => e.load(tu, inst),
                RtGet::Proxy => e.load(tb, proxy),
                RtGet::InGuest => e.load(tb, in_guest),
                RtGet::Entry => {
                    // 64 * (first node after the triangle nodes) + 8 * primitive.
                    let b = e.load(tu, base);
                    let p = e.load(tu, prim);
                    let six = e.cu(6);
                    let three = e.cu(3);
                    let b = e.shl(b, six);
                    let p = e.shl(p, three);
                    e.iadd(b, p)
                }
            }))
        }
        RtOp::Take { slot, level, levels } => {
            let [_, _, ready, leaf, inst, prim, _, _, inside, here, last_inst, last_list, _, in_guest] = rt_slot(e, slot);
            let take = e.arg(v, 0)?;
            // The leaf list the guest processes for the invocation now: its
            // later candidates in it are not handed out again.
            let processed = level == 1 || levels == 1;
            if processed {
                e.if_then(take, |e| {
                    let tu = e.t_u32();
                    let i = e.load(tu, inst);
                    let p = e.load(tu, prim);
                    e.store(last_inst, i);
                    e.store(last_list, p);
                });
            }
            if level == 0 {
                // Taken at the top: no longer pending there; its leaf waits
                // for the bottom level (two levels), every other leaf is
                // dropped (their instance is not the one popped).
                let r = e.load(tb, ready);
                let keep = e.lnot(take);
                let r = e.land(r, keep);
                e.store(ready, r);
                if levels == 2 {
                    // An instance without a host BLAS: the guest's stack
                    // walks it (the same for the whole wave).
                    let g = e.arg(v, 1)?;
                    let zero = e.cu(0);
                    let guest = e.ine(g, zero);
                    e.store(in_guest, guest);
                    let host = e.lnot(guest);
                    let mine = e.land(take, host);
                    e.store(leaf, mine);
                    e.store(inside, mine);
                    let i = e.load(tu, inst);
                    e.store(here, i);
                    let walked = e.land(take, guest);
                    e.if_then(walked, |e| {
                        let tu = e.t_u32();
                        let i = e.load(tu, inst);
                        e.store(last_inst, i);
                        let pl = e.cu(PROXY_LIST);
                        e.store(last_list, pl);
                    });
                }
            } else {
                let l = e.load(tb, leaf);
                let keep = e.lnot(take);
                let l = e.land(l, keep);
                e.store(leaf, l);
            }
            Ok(None)
        }
    }
}
