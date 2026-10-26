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
//! 3 bits are its type (0/1 triangle, 4 fp16 box, 5 fp32 box); a node lies
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

use super::emit::{Emitter, R};
use ps5shader::ssa::*;
use rspirv::spirv::{GlslStd450Op as GLOp, Op as S, Word};

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
    let one_u = o.u(1);
    let second = o.e.ieq(q.ty, one_u);
    let mut v1 = [0; 3];
    let mut v2 = [0; 3];
    let mut v3 = [0; 3];
    for a in 0..3 {
        let (d0, d3, d6, d9) = (o.as_f(d[a]), o.as_f(d[3 + a]), o.as_f(d[6 + a]), o.as_f(d[9 + a]));
        v1[a] = o.self_f(second, d3, d0);
        v2[a] = o.self_f(second, d9, d3);
        v3[a] = d6;
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

fn boxes(o: &mut Ops, q: &Query, fp16: bool) -> R<Word> {
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
        let hit = o.cmp(S::FOrdLessThanEqual, min_t, grown);
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
    let ImageOp::BvhIntersect { bvh64 } = attr.op else { return Err("not a BVH query".into()) };
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
        let one = o.u(1);
        let tri_ty = o.e.ule(ty, one);
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
        let mut err: Option<String> = None;
        let r = e.if_value(exec, t4, none, |e| {
            let o = Ops { e, tu, tf, tb, t64 };
            let tested = o.e.if_value(q.is_triangle, t4, none, |e| {
                let mut o = Ops { e, tu, tf, tb, t64 };
                triangle(&mut o, &q).unwrap_or_else(|m| {
                    err = Some(m);
                    none
                })
            });
            let box16 = o.e.if_value(q.is_box16, t4, tested, |e| {
                let mut o = Ops { e, tu, tf, tb, t64 };
                boxes(&mut o, &q, true).unwrap_or_else(|m| {
                    err.get_or_insert(m);
                    tested
                })
            });
            o.e.if_value(q.is_box32, t4, box16, |e| {
                let mut o = Ops { e, tu, tf, tb, t64 };
                boxes(&mut o, &q, false).unwrap_or_else(|m| {
                    err.get_or_insert(m);
                    box16
                })
            })
        });
        if let Some(m) = err {
            return Err(m);
        }
        r
    };
    Ok(Some(result))
}
