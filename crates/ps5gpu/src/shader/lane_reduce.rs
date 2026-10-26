// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Wave reductions read from one lane, on host subgroups that may lack
//! lanes the guest wave has.
//!
//! A graphics wave is a host subgroup of the invocations the rasterizer or
//! the vertex assembler launched: a fragment subgroup holds only the
//! covered quads, and a wave64 program on a 32-lane host has no lanes
//! 32..63 at all. Guest code reduces a value over the wave with DPP row
//! scans, a cross-row permute and a v_readlane of the last lane of a row
//! or half; reading a lane the host subgroup lacks is undefined.
//!
//! The reduction is recognised by evaluating the read value symbolically
//! per lane: through shuffles (whose source lanes are functions of the
//! lane), selects on lane conditions, and one associative operation, it
//! must be that operation over a set of lanes of one seed value, the
//! identity where the seed's condition is false. When that condition
//! holds only in lanes live at the program's entry (the entry EXEC, helper
//! lanes excluded), lanes the host lacks contribute the identity, so the
//! value is a subgroup reduction of the seed over the lanes of the set
//! that exist (`RPS5E_LANE_REDUCTIONS=0`: off). Every invocation of a
//! vertex program is live at its entry.

use super::emit::{Emitter, R};
use ps5shader::ssa::*;
use rspirv::spirv::{Capability, Op as S, Word};
use std::collections::HashMap;

/// An associative, commutative reduction operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Reduction {
    op: Op,
    identity: u64,
    /// Every lane may contribute once only (no idempotence).
    disjoint: bool,
    spirv: S,
}

const REDUCTIONS: [Reduction; 7] = [
    Reduction { op: Op::UMin, identity: 0xFFFF_FFFF, disjoint: false, spirv: S::GroupNonUniformUMin },
    Reduction { op: Op::UMax, identity: 0, disjoint: false, spirv: S::GroupNonUniformUMax },
    Reduction { op: Op::SMin, identity: 0x7FFF_FFFF, disjoint: false, spirv: S::GroupNonUniformSMin },
    Reduction { op: Op::SMax, identity: 0x8000_0000, disjoint: false, spirv: S::GroupNonUniformSMax },
    Reduction { op: Op::IAdd, identity: 0, disjoint: true, spirv: S::GroupNonUniformIAdd },
    Reduction { op: Op::And, identity: 0xFFFF_FFFF, disjoint: false, spirv: S::GroupNonUniformBitwiseAnd },
    Reduction { op: Op::Or, identity: 0, disjoint: false, spirv: S::GroupNonUniformBitwiseOr },
];

/// Lane-set evaluation of one candidate reduction.
struct Walk {
    reduction: Reduction,
    seed: Option<Value>,
    memo: HashMap<(Value, u32), Option<u64>>,
    steps: u32,
}

const MAX_STEPS: u32 = 1 << 16;

impl<'a, 'p> Emitter<'a, 'p> {
    /// `v` (a ReadLane of constant lane `lane`) as a subgroup reduction,
    /// when it is one (see the module documentation).
    pub(super) fn lane_reduction(&mut self, v: Value, lane: u32) -> R<Option<Word>> {
        if !self.feat.emit.lane_reductions || self.stage == crate::shader::ShaderStage::Compute || self.lanes != 1 || self.split.is_some() || lane >= 64 {
            return Ok(None);
        }
        let x = self.f.args(v)[0];
        if self.f.ty(x) != Type::U32 {
            return Ok(None);
        }
        for reduction in REDUCTIONS {
            if self.f.value(x).op != reduction.op && !matches!(self.f.value(x).op, Op::Select) {
                continue;
            }
            let mut walk = Walk { reduction, seed: None, memo: HashMap::new(), steps: 0 };
            let Some(set) = self.lanes_of(&mut walk, x, lane) else { continue };
            let Some(seed) = walk.seed else { continue };
            if set == 0 || !self.seed_is_entry_bounded(seed, reduction) {
                continue;
            }
            return self.emit_reduction(seed, set, reduction).map(Some);
        }
        Ok(None)
    }

    /// The lanes whose seed values `x` combines in guest lane `lane` (None:
    /// not a reduction of one seed).
    fn lanes_of(&mut self, walk: &mut Walk, x: Value, lane: u32) -> Option<u64> {
        if let Some(&r) = walk.memo.get(&(x, lane)) {
            return r;
        }
        walk.steps += 1;
        if walk.steps > MAX_STEPS {
            return None;
        }
        let r = self.lanes_of_uncached(walk, x, lane);
        walk.memo.insert((x, lane), r);
        r
    }

    fn lanes_of_uncached(&mut self, walk: &mut Walk, x: Value, lane: u32) -> Option<u64> {
        let d = self.f.value(x).clone();
        if Some(x) == walk.seed {
            return Some(1u64 << lane);
        }
        match d.op {
            Op::Const(c) if c & 0xFFFF_FFFF == walk.reduction.identity => Some(0),
            op if op == walk.reduction.op => {
                let a = self.lanes_of(walk, d.args[0], lane)?;
                let b = self.lanes_of(walk, d.args[1], lane)?;
                if walk.reduction.disjoint && a & b != 0 {
                    return None;
                }
                Some(a | b)
            }
            Op::Shuffle | Op::ReadLane => {
                let t = self.lane_table(d.args[1], 0)?;
                let source = t[lane as usize];
                if source >= self.f.wave_size as u64 {
                    return None;
                }
                self.lanes_of(walk, d.args[0], source as u32)
            }
            Op::Select => {
                if let Some(c) = self.lane_table(d.args[0], 0) {
                    let arm = if c[lane as usize] != 0 { d.args[1] } else { d.args[2] };
                    return self.lanes_of(walk, arm, lane);
                }
                // The seed: a value where a lane condition holds, the
                // identity elsewhere.
                if walk.seed.is_none() && matches!(self.f.value(d.args[2]).op, Op::Const(c) if c & 0xFFFF_FFFF == walk.reduction.identity) {
                    walk.seed = Some(x);
                    return Some(1u64 << lane);
                }
                None
            }
            _ => None,
        }
    }

    /// The seed `select(c, value, identity)` is the identity in every lane
    /// that was not live at the program's entry. Every invocation of a
    /// vertex program enters it (EXEC starts full, no helper lanes): the
    /// only lanes not live at entry are those the host subgroup lacks.
    fn seed_is_entry_bounded(&mut self, seed: Value, _reduction: Reduction) -> bool {
        if self.stage == crate::shader::ShaderStage::Vertex {
            return true;
        }
        let c = self.f.args(seed)[0];
        self.implies_entry(c, 0)
    }

    /// Boolean `c` is false in every lane not live at entry.
    fn implies_entry(&mut self, c: Value, depth: u32) -> bool {
        if depth > 16 {
            return false;
        }
        let d = self.f.value(c).clone();
        match d.op {
            Op::Const(0) => true,
            Op::And => self.implies_entry(d.args[0], depth + 1) || self.implies_entry(d.args[1], depth + 1),
            // The entry EXEC: not a helper (or past-the-end) invocation.
            Op::ICmp(ICmp::Eq) => {
                let is_entry = |v: Value| matches!(self.f.value(v).op, Op::SystemValue(SystemValue::HelperInvocation));
                let zero = |v: Value| matches!(self.f.value(v).op, Op::Const(0));
                (is_entry(d.args[0]) && zero(d.args[1])) || (is_entry(d.args[1]) && zero(d.args[0]))
            }
            // The lane's bit of a mask: `(word >> (lane & 31)) & 1 != 0`
            // with the word of the lane's half of a ballot.
            Op::ICmp(ICmp::Ne) => {
                let zero = |v: Value| matches!(self.f.value(v).op, Op::Const(0));
                let bit = if zero(d.args[1]) { d.args[0] } else if zero(d.args[0]) { d.args[1] } else { return false };
                self.ballot_bit(bit).is_some_and(|b| self.implies_entry(b, depth + 1))
            }
            _ => false,
        }
    }

    /// `bit` = this lane's bit of `ballot(b)`: returns `b`.
    fn ballot_bit(&mut self, bit: Value) -> Option<Value> {
        let d = self.f.value(bit).clone();
        if d.op != Op::And {
            return None;
        }
        let one = |e: &Self, v: Value| matches!(e.f.value(v).op, Op::Const(1));
        let shifted = if one(self, d.args[1]) { d.args[0] } else if one(self, d.args[0]) { d.args[1] } else { return None };
        let s = self.f.value(shifted).clone();
        if s.op != Op::LShr {
            return None;
        }
        let shift = self.lane_table(s.args[1], 0)?;
        if (0..64).any(|g| shift[g] != (g as u64 & 31)) {
            return None;
        }
        // The word: the low ballot word, or by lane half.
        let words = self.ballot_words(s.args[0])?;
        let halves = if self.f.wave_size == 64 { 2 } else { 1 };
        let mut source = None;
        for (half, w) in words.iter().take(halves).enumerate() {
            let (b, k) = (*w)?;
            if k != half || source.is_some_and(|s| s != b) {
                return None;
            }
            source = Some(b);
        }
        source
    }

    /// Per lane half, `(b, component)` when `word` is component of
    /// `ballot(b)` in the lanes of that half (None: lanes not covered).
    fn ballot_words(&mut self, word: Value) -> Option<[Option<(Value, usize)>; 2]> {
        let d = self.f.value(word).clone();
        let extract = |e: &Self, v: Value| -> Option<(Value, usize)> {
            let x = e.f.value(v);
            match x.op {
                Op::Extract(k) if e.f.value(x.args[0]).op == Op::Ballot => Some((e.f.value(x.args[0]).args[0], k as usize)),
                _ => None,
            }
        };
        if let Some(w) = extract(self, word) {
            // One word for every lane: the lanes of its own half.
            return Some(if w.1 == 0 { [Some(w), None] } else { [None, Some(w)] });
        }
        if d.op == Op::Select {
            let c = self.lane_table(d.args[0], 0)?;
            if (0..64).any(|g| (c[g] != 0) != (g >= 32)) {
                return None;
            }
            let hi = extract(self, d.args[1])?;
            let lo = extract(self, d.args[2])?;
            return Some([Some(lo), Some(hi)]);
        }
        None
    }

    /// The subgroup reduction of `seed` over the guest lanes in `set`.
    fn emit_reduction(&mut self, seed: Value, set: u64, reduction: Reduction) -> R<Word> {
        let s = self.val(seed)?;
        let u = self.t_u32();
        let lane = self.lane_id();
        let identity = self.cu(reduction.identity as u32);
        let include = if set == u64::MAX || (self.f.wave_size == 32 && set as u32 == u32::MAX) {
            None
        } else {
            let lo = self.cu(set as u32);
            let hi = self.cu((set >> 32) as u32);
            let c32 = self.cu(32);
            let high = self.ule(c32, lane);
            let word = self.select(u, high, hi, lo);
            let c31 = self.cu(31);
            let shift = self.band(lane, c31);
            let bits = self.shr(word, shift);
            let one = self.cu(1);
            let bit = self.band(bits, one);
            let zero = self.cu(0);
            Some(self.ine(bit, zero))
        };
        let keys = match include {
            Some(c) => self.select(u, c, s, identity),
            None => s,
        };
        self.cap(Capability::GroupNonUniform);
        self.cap(Capability::GroupNonUniformArithmetic);
        let scope = self.cu(super::emit::SCOPE_SUBGROUP);
        Ok(self.ins_ops(
            reduction.spirv,
            u,
            vec![rspirv::dr::Operand::IdRef(scope), rspirv::dr::Operand::GroupOperation(rspirv::spirv::GroupOperation::Reduce), rspirv::dr::Operand::IdRef(keys)],
        ))
    }
}
