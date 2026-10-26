// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Wave64LaneMaskProjection: this invocation's bit of a 64-bit lane mask
//! built from ballots by word-wise scalar mask arithmetic is the same
//! arithmetic on the ballots' predicates.
//!
//! `inverse_ballot(lo, hi)` reads bit `lane` of `lo` in lanes 0..31 and bit
//! `lane - 32` of `hi` in lanes 32..63. When `lo` is built from the ballots'
//! low words (`extract.0`) and `hi` from their high words (`extract.1`) by
//! `and` / `or` / `not` / `select` / phis, with the constants 0 and all ones
//! as the only other leaves, each chain is projected to a per-invocation
//! predicate: a ballot leaf becomes its predicate in the lanes the wave has
//! (`present`: a ballot has no bits for lanes past the work-group). The
//! inverse ballot becomes the projected predicate, or a select on the lane
//! half when the two chains differ.
//!
//! Without the projection, a back-end that runs a wave as two host
//! subgroups has to exchange every ballot word between the halves; with it
//! only the masks scalar code looks at as numbers (branches, counts, first
//! lanes) keep their ballots.

use crate::fxhash::{HashMap, HashSet};
use crate::ssa::*;

/// Where the per-invocation predicates are built.
struct Projector {
    /// `(mask value, word)` -> projected predicate.
    projected: HashMap<(Value, u32), Value>,
    /// Ballot -> its predicate restricted to the present lanes.
    leaves: HashMap<Value, Value>,
    /// The present-lane predicate (built on first use).
    present: Option<Value>,
    present_source: Option<SystemValue>,
}

/// Insert `op` into the block of `after`, right after it (a phi at the
/// block start).
fn insert_after(func: &mut Function, after: Value, op: Op, ty: Type, args: Vec<Value>) -> Value {
    let block = func.value(after).block.expect("placed value");
    let v = func.new_value(op, ty, args);
    func.value_mut(v).block = Some(block);
    let insts = &mut func.block_mut(block).insts;
    let at = if op == Op::Phi { 0 } else { insts.iter().position(|&x| x == after).map_or(insts.len(), |i| i + 1) };
    insts.insert(at, v);
    v
}

/// Insert `op` into `block` right before `before`.
fn insert_before(func: &mut Function, before: Value, op: Op, ty: Type, args: Vec<Value>) -> Value {
    let block = func.value(before).block.expect("placed value");
    let v = func.new_value(op, ty, args);
    func.value_mut(v).block = Some(block);
    let insts = &mut func.block_mut(block).insts;
    let at = insts.iter().position(|&x| x == before).unwrap_or(0);
    insts.insert(at, v);
    v
}

/// The ballot whose word `word` `v` is.
fn ballot_word(func: &Function, v: Value, word: u32) -> Option<Value> {
    let d = func.value(v);
    if d.op != Op::Extract(word as u8) {
        return None;
    }
    let src = d.args[0];
    (func.op(src) == Op::Ballot).then_some(src)
}

fn can_project(func: &Function, v: Value, word: u32, visited: &mut HashSet<(Value, u32)>, grounded: &mut bool) -> bool {
    if let Some(c) = func.as_const(v) {
        let ok = func.ty(v) == Type::U32 && (c == 0 || c == 0xFFFF_FFFF);
        *grounded |= ok;
        return ok;
    }
    if ballot_word(func, v, word).is_some() {
        *grounded = true;
        return true;
    }
    if !visited.insert((v, word)) {
        return true;
    }
    let d = func.value(v);
    if d.ty != Type::U32 || !matches!(d.op, Op::And | Op::Or | Op::Not | Op::Select | Op::Phi) {
        return false;
    }
    let first = if d.op == Op::Select { 1 } else { 0 };
    !d.args.is_empty() && d.args[first..].iter().all(|&a| can_project(func, a, word, visited, grounded))
}

impl Projector {
    /// The predicate "this lane exists" (true without a source).
    fn present(&mut self, func: &mut Function) -> Option<Value> {
        if let Some(p) = self.present {
            return Some(p);
        }
        let sv = self.present_source?;
        let entry = func.entry;
        let at = func.block(entry).insts.iter().position(|&x| func.op(x) != Op::Phi).unwrap_or(func.block(entry).insts.len());
        let s = func.new_value(Op::SystemValue(sv), Type::U32, vec![]);
        let zero = func.const_u32(0);
        let p = func.new_value(Op::ICmp(ICmp::Eq), Type::BOOL, vec![s, zero]);
        for (k, v) in [s, p].into_iter().enumerate() {
            func.value_mut(v).block = Some(entry);
            func.block_mut(entry).insts.insert(at + k, v);
        }
        self.present = Some(p);
        Some(p)
    }

    fn leaf(&mut self, func: &mut Function, ballot: Value) -> Value {
        if let Some(&p) = self.leaves.get(&ballot) {
            return p;
        }
        let pred = func.args(ballot)[0];
        let p = match self.present(func) {
            Some(present) => insert_after(func, ballot, Op::And, Type::BOOL, vec![pred, present]),
            None => pred,
        };
        self.leaves.insert(ballot, p);
        p
    }

    fn project(&mut self, func: &mut Function, v: Value, word: u32) -> Value {
        if let Some(c) = func.as_const(v) {
            return func.const_bool(c != 0);
        }
        if let Some(&p) = self.projected.get(&(v, word)) {
            return p;
        }
        if let Some(b) = ballot_word(func, v, word) {
            return self.leaf(func, b);
        }
        let d = func.value(v).clone();
        let f = func.const_bool(false);
        let result = match d.op {
            Op::Phi => insert_after(func, v, Op::Phi, Type::BOOL, vec![f; d.args.len()]),
            Op::Select => insert_after(func, v, Op::Select, Type::BOOL, vec![d.args[0], f, f]),
            Op::Not => insert_after(func, v, Op::Not, Type::BOOL, vec![f]),
            op => insert_after(func, v, op, Type::BOOL, vec![f, f]),
        };
        self.projected.insert((v, word), result);
        let first = if d.op == Op::Select { 1 } else { 0 };
        for (i, &a) in d.args.iter().enumerate().skip(first) {
            let p = self.project(func, a, word);
            func.value_mut(result).args[i] = p;
        }
        result
    }
}

/// Project the inverse ballots of a wave64 function. `present`: a system
/// value that is 0 exactly in the lanes the wave has (`None`: every lane
/// exists). Returns the number of inverse ballots replaced.
pub fn project_wave64_lane_masks(func: &mut Function, present: Option<SystemValue>) -> usize {
    if func.wave_size != 64 {
        return 0;
    }
    let candidates: Vec<Value> = func
        .order
        .iter()
        .flat_map(|&b| func.block(b).insts.iter().copied())
        .filter(|&v| func.op(v) == Op::InverseBallot)
        .collect();
    let mut p = Projector { projected: HashMap::default(), leaves: HashMap::default(), present: None, present_source: present };
    let mut replace: HashMap<Value, Value> = HashMap::default();
    let mut n = 0;
    for v in candidates {
        let (lo, hi) = (func.args(v)[0], func.args(v)[1]);
        let mut ok = true;
        let mut grounded = false;
        for (mask, word) in [(lo, 0), (hi, 1)] {
            let mut visited = HashSet::default();
            ok &= can_project(func, mask, word, &mut visited, &mut grounded);
        }
        if !ok || !grounded {
            continue;
        }
        let plo = p.project(func, lo, 0);
        let phi = p.project(func, hi, 1);
        n += 1;
        if plo == phi {
            replace.insert(v, plo);
            continue;
        }
        // Lanes 32..63 take the high word's predicate: the inverse ballot
        // becomes that select in place.
        let lane = insert_before(func, v, Op::LaneId, Type::U32, vec![]);
        let c32 = func.const_u32(32);
        let high = insert_before(func, v, Op::ICmp(ICmp::UGe), Type::BOOL, vec![lane, c32]);
        let d = func.value_mut(v);
        d.op = Op::Select;
        d.args = vec![high, phi, plo];
    }
    func.replace_uses(&replace);
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link(f: &mut Function, from: Block, term: Terminator) {
        for s in term.successors() {
            f.block_mut(s).preds.push(from);
        }
        f.block_mut(from).term = term;
    }

    /// `inverse_ballot(and(ballot(a).lo, not(ballot(b).lo)), and(ballot(a).hi, not(ballot(b).hi)))`
    /// becomes `a && present && !(b && present)`, with no subgroup operation left.
    #[test]
    fn parallel_word_chains_project_to_one_predicate() {
        let mut f = Function::new(64, Stage::Compute, 0);
        let b0 = f.new_block();
        f.entry = b0;
        f.order = vec![b0];
        let x = f.append(b0, Op::UserData(0), Type::U32, vec![]);
        let lane = f.append(b0, Op::LaneId, Type::U32, vec![]);
        let a = f.append(b0, Op::ICmp(ICmp::ULt), Type::BOOL, vec![lane, x]);
        let five = f.const_u32(5);
        let b = f.append(b0, Op::ICmp(ICmp::Eq), Type::BOOL, vec![lane, five]);
        let ba = f.append(b0, Op::Ballot, Type::U32X2, vec![a]);
        let bb = f.append(b0, Op::Ballot, Type::U32X2, vec![b]);
        let words: Vec<Value> = (0..2u8)
            .map(|w| {
                let wa = f.append(b0, Op::Extract(w), Type::U32, vec![ba]);
                let wb = f.append(b0, Op::Extract(w), Type::U32, vec![bb]);
                let nb = f.append(b0, Op::Not, Type::U32, vec![wb]);
                f.append(b0, Op::And, Type::U32, vec![wa, nb])
            })
            .collect();
        let inv = f.append(b0, Op::InverseBallot, Type::BOOL, vec![words[0], words[1]]);
        let u = f.append(b0, Op::UConvert, Type::U32, vec![inv]);
        let st = f.append(b0, Op::LdsStore { access: Access::Dwords(1) }, Type::VOID, vec![x, u, inv]);
        f.block_mut(b0).term = Terminator::Return;
        assert_eq!(project_wave64_lane_masks(&mut f, Some(SystemValue::HelperInvocation)), 1);
        // The two chains are alike: CSE merges them and the half select
        // folds away.
        crate::opt::fold::cse(&mut f);
        crate::opt::fold::fold(&mut f);
        crate::opt::dce::dce(&mut f);
        let p = f.args(st)[2];
        assert_eq!(f.op(p), Op::And);
        assert_eq!(f.args(u)[0], p);
        let ops: Vec<Op> = f.block(b0).insts.iter().map(|&v| f.op(v)).collect();
        assert!(!ops.iter().any(|o| matches!(o, Op::Ballot | Op::InverseBallot | Op::Select)), "{f}");
        crate::ssa::verify::verify(&f, true).unwrap();
    }

    /// A low word chain that differs from the high one (`s_mov_b32
    /// exec_lo, 0` style) gives a select on the lane half; a high word
    /// leaf in the low chain is not projected.
    #[test]
    fn differing_chains_select_on_the_half_and_crossed_words_stay() {
        let mut f = Function::new(64, Stage::Compute, 0);
        let b0 = f.new_block();
        f.entry = b0;
        f.order = vec![b0];
        let lane = f.append(b0, Op::LaneId, Type::U32, vec![]);
        let x = f.append(b0, Op::UserData(0), Type::U32, vec![]);
        let a = f.append(b0, Op::ICmp(ICmp::ULt), Type::BOOL, vec![lane, x]);
        let ba = f.append(b0, Op::Ballot, Type::U32X2, vec![a]);
        let lo = f.append(b0, Op::Extract(0), Type::U32, vec![ba]);
        let hi = f.append(b0, Op::Extract(1), Type::U32, vec![ba]);
        let zero = f.const_u32(0);
        let inv = f.append(b0, Op::InverseBallot, Type::BOOL, vec![zero, hi]);
        let crossed = f.append(b0, Op::InverseBallot, Type::BOOL, vec![hi, lo]);
        let both = f.append(b0, Op::And, Type::BOOL, vec![inv, crossed]);
        f.append(b0, Op::LdsStore { access: Access::Dwords(1) }, Type::VOID, vec![x, x, both]);
        f.block_mut(b0).term = Terminator::Return;
        assert_eq!(project_wave64_lane_masks(&mut f, None), 1);
        assert_eq!(f.op(inv), Op::Select);
        assert_eq!(f.args(inv)[1], a);
        assert_eq!(f.op(crossed), Op::InverseBallot);
        crate::ssa::verify::verify(&f, true).unwrap();
    }

    /// A loop-carried EXEC (phi of the words) projects to a phi of
    /// predicates.
    #[test]
    fn loop_carried_masks_project_through_phis() {
        let mut f = Function::new(64, Stage::Compute, 0);
        let blocks: Vec<Block> = (0..3).map(|_| f.new_block()).collect();
        f.entry = blocks[0];
        f.order = blocks.clone();
        let lane = f.append(blocks[0], Op::LaneId, Type::U32, vec![]);
        let x = f.append(blocks[0], Op::UserData(0), Type::U32, vec![]);
        let a = f.append(blocks[0], Op::ICmp(ICmp::ULt), Type::BOOL, vec![lane, x]);
        let ba = f.append(blocks[0], Op::Ballot, Type::U32X2, vec![a]);
        let lo0 = f.append(blocks[0], Op::Extract(0), Type::U32, vec![ba]);
        let hi0 = f.append(blocks[0], Op::Extract(1), Type::U32, vec![ba]);
        link(&mut f, blocks[0], Terminator::Branch(blocks[1]));
        f.block_mut(blocks[1]).merge = Merge::Loop { merge: blocks[2], continue_block: blocks[1] };
        let plo = f.append(blocks[1], Op::Phi, Type::U32, vec![lo0]);
        let phi = f.append(blocks[1], Op::Phi, Type::U32, vec![hi0]);
        let lane1 = f.append(blocks[1], Op::LaneId, Type::U32, vec![]);
        let one = f.const_u32(1);
        let c = f.append(blocks[1], Op::ICmp(ICmp::Ne), Type::BOOL, vec![lane1, one]);
        let bc = f.append(blocks[1], Op::Ballot, Type::U32X2, vec![c]);
        let clo = f.append(blocks[1], Op::Extract(0), Type::U32, vec![bc]);
        let chi = f.append(blocks[1], Op::Extract(1), Type::U32, vec![bc]);
        let nlo = f.append(blocks[1], Op::And, Type::U32, vec![plo, clo]);
        let nhi = f.append(blocks[1], Op::And, Type::U32, vec![phi, chi]);
        f.value_mut(plo).args.push(nlo);
        f.value_mut(phi).args.push(nhi);
        let inv = f.append(blocks[1], Op::InverseBallot, Type::BOOL, vec![nlo, nhi]);
        f.append(blocks[1], Op::LdsStore { access: Access::Dwords(1) }, Type::VOID, vec![x, x, inv]);
        let any = f.append(blocks[1], Op::Or, Type::U32, vec![nlo, nhi]);
        let zero = f.const_u32(0);
        let done = f.append(blocks[1], Op::ICmp(ICmp::Eq), Type::BOOL, vec![any, zero]);
        link(&mut f, blocks[1], Terminator::CondBranch { cond: done, then_block: blocks[2], else_block: blocks[1] });
        f.block_mut(blocks[2]).term = Terminator::Return;
        assert_eq!(project_wave64_lane_masks(&mut f, Some(SystemValue::HelperInvocation)), 1);
        crate::ssa::verify::verify(&f, true).unwrap();
        // The projected loop phi is a bool phi in the header.
        assert!(f.block(blocks[1]).insts.iter().any(|&v| f.op(v) == Op::Phi && f.ty(v) == Type::BOOL));
    }
}
