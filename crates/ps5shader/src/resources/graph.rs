// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The uniform value graph of one program.
//!
//! Every value a descriptor can be assembled from, symbolic in the user-data
//! SGPRs at entry and in the shader base. Nodes live in an arena and are
//! interned: equal structure yields the same [`ValueId`], and operations over
//! constants fold, so revisiting a block during the fixpoint reproduces its
//! values exactly.

use super::memory::MemoryAccessTable;
use super::value::*;
use super::ResourcePlanError;
use crate::inline_constants;
use crate::ir::ControlFlowGraph;
use crate::ir_types::*;
use crate::opcode::Opcode;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::sync::Mutex;

/// The graph values one memory access reads its descriptor from.
///
/// `active` is the EXEC mask the access runs under: a lane outside it
/// performs no access, so a lane-masked register write under the same mask
/// contributes only its new value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryAccessBinding {
    /// V#, T#, or address handle.
    pub handle: Option<ValueId>,
    pub sampler_handle: Option<ValueId>,
    /// The read node of a scalar load (or uniform buffer load).
    pub read: Option<ValueId>,
    /// The uniform vector offset of a global access.
    pub offset: Option<ValueId>,
    pub active: Option<ValueId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct InternKey {
    kind: ValueKind,
    ty: ValueType,
    op: ScalarOp,
    payload: u64,
    identity: u64,
    operands: Box<[ValueId]>,
}

/// The uniform value graph of one program (see the module docs).
#[derive(Debug)]
pub struct ScalarValueGraph {
    pub(super) program: Program,
    pub(super) cfg: ControlFlowGraph,
    pub(super) memory: MemoryAccessTable,
    pub(super) user_data_base: u32,
    pub(super) user_data_count: u32,
    /// A wave32 lane mask fills one SGPR; wave64 an aligned pair.
    pub(super) wave_size: u32,
    nodes: Vec<Node>,
    interned: HashMap<InternKey, ValueId>,
    /// Index-aligned with the memory table; `None` when the access has no
    /// descriptor value.
    pub(super) accesses: Vec<Option<MemoryAccessBinding>>,
    /// The condition of each conditional branch, by branch pc.
    pub(super) branch_conditions: BTreeMap<u32, ValueId>,
    /// The instruction each per-instruction undefined node came from.
    undefined_origins: HashMap<ValueId, (u32, Opcode)>,
    /// S_FF1 results keyed to their instruction (their input guard can
    /// differ between scans).
    bit_scan_instructions: HashMap<ValueId, u32>,
    /// The instruction the builder is translating, for diagnostics.
    pub(super) builder_instruction: Option<(u32, Opcode)>,
    /// Phi → its invariant value (`None`: it merges different values).
    invariant_phis: Mutex<HashMap<ValueId, Option<ValueId>>>,
}

impl Clone for ScalarValueGraph {
    fn clone(&self) -> Self {
        ScalarValueGraph {
            program: self.program.clone(),
            cfg: self.cfg.clone(),
            memory: self.memory.clone(),
            user_data_base: self.user_data_base,
            user_data_count: self.user_data_count,
            wave_size: self.wave_size,
            nodes: self.nodes.clone(),
            interned: self.interned.clone(),
            accesses: self.accesses.clone(),
            branch_conditions: self.branch_conditions.clone(),
            undefined_origins: self.undefined_origins.clone(),
            bit_scan_instructions: self.bit_scan_instructions.clone(),
            builder_instruction: self.builder_instruction,
            invariant_phis: Mutex::new(self.invariant_phis.lock().unwrap().clone()),
        }
    }
}

/// Per-query state of a structural-equivalence walk: the phis whose
/// invariant value is being resolved further up the stack.
#[derive(Default)]
struct EquivalenceContext {
    resolving: HashSet<ValueId>,
}

impl ScalarValueGraph {
    /// Build the graph of `program`: user-data SGPRs `user_data_base ..
    /// user_data_base + user_data_count` are symbolic at entry. Buffer loads at
    /// `fixed_function_vertex_loads` pcs are left out of the memory table.
    pub fn build(
        program: Program,
        user_data_base: u32,
        user_data_count: u32,
        fixed_function_vertex_loads: Option<&HashSet<u32>>,
        wave_size: u32,
    ) -> Result<Self, ResourcePlanError> {
        let cfg = ControlFlowGraph::build(&program.instructions, &crate::cfg::calls::CallResolver::new(&program, true));
        let memory = MemoryAccessTable::build(&program, fixed_function_vertex_loads);
        let mut graph = ScalarValueGraph {
            program,
            cfg,
            memory,
            user_data_base,
            user_data_count,
            wave_size,
            nodes: Vec::new(),
            interned: HashMap::new(),
            accesses: Vec::new(),
            branch_conditions: BTreeMap::new(),
            undefined_origins: HashMap::new(),
            bit_scan_instructions: HashMap::new(),
            builder_instruction: None,
            invariant_phis: Mutex::new(HashMap::new()),
        };
        super::builder::run(&mut graph)?;
        Ok(graph)
    }

    // ---- accessors ----

    pub fn program(&self) -> &Program {
        &self.program
    }

    pub fn control_flow(&self) -> &ControlFlowGraph {
        &self.cfg
    }

    pub fn memory(&self) -> &MemoryAccessTable {
        &self.memory
    }

    pub fn user_data_base(&self) -> u32 {
        self.user_data_base
    }

    pub fn user_data_count(&self) -> u32 {
        self.user_data_count
    }

    pub fn wave_size(&self) -> u32 {
        self.wave_size
    }

    /// Handles each memory access reads, as built (before table flattening).
    pub fn accesses(&self) -> &[Option<MemoryAccessBinding>] {
        &self.accesses
    }

    pub fn branch_conditions(&self) -> &BTreeMap<u32, ValueId> {
        &self.branch_conditions
    }

    pub fn node(&self, id: ValueId) -> &Node {
        &self.nodes[id.index()]
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Every node in creation order.
    pub fn values(&self) -> impl Iterator<Item = ValueId> + '_ {
        (0..self.nodes.len() as u32).map(ValueId)
    }

    pub fn kind(&self, id: ValueId) -> ValueKind {
        self.nodes[id.index()].kind
    }

    pub fn operands(&self, id: ValueId) -> &[ValueId] {
        &self.nodes[id.index()].operands
    }

    pub fn operand(&self, id: ValueId, index: usize) -> ValueId {
        self.nodes[id.index()].operands[index]
    }

    pub fn is_constant(&self, id: ValueId) -> bool {
        self.nodes[id.index()].is_constant()
    }

    pub fn is_undefined(&self, id: ValueId) -> bool {
        self.nodes[id.index()].is_undefined()
    }

    /// The constant value of a U32 constant node.
    pub fn constant_u32(&self, id: ValueId) -> Option<u32> {
        let n = &self.nodes[id.index()];
        (n.is_constant()).then_some(n.payload as u32)
    }

    /// The instruction an undefined node came from, when known.
    pub fn undefined_origin(&self, id: ValueId) -> Option<(u32, Opcode)> {
        self.undefined_origins.get(&id).copied()
    }

    // ---- node creation ----

    fn intern(&mut self, kind: ValueKind, ty: ValueType, payload: u64, op: ScalarOp, operands: &[ValueId], identity: u64) -> ValueId {
        let key = InternKey { kind, ty, op, payload, identity, operands: operands.into() };
        if let Some(&id) = self.interned.get(&key) {
            return id;
        }
        let id = ValueId(self.nodes.len() as u32);
        self.nodes.push(Node { kind, ty, op, payload, operands: key.operands.clone(), phi_predecessors: Box::new([]) });
        self.interned.insert(key, id);
        id
    }

    pub(super) fn constant(&mut self, value: u32) -> ValueId {
        self.intern(ValueKind::Constant, ValueType::U32, value as u64, ScalarOp::None, &[], 0)
    }

    pub(super) fn constant64(&mut self, value: u64) -> ValueId {
        self.intern(ValueKind::Constant, ValueType::U64, value, ScalarOp::None, &[], 0)
    }

    pub(super) fn constant_bool(&mut self, value: bool) -> ValueId {
        self.intern(ValueKind::Constant, ValueType::Bool, value as u64, ScalarOp::None, &[], 0)
    }

    pub(super) fn memory_aperture(&mut self, operand: u32) -> ValueId {
        self.intern(ValueKind::MemoryAperture, ValueType::U32, operand as u64, ScalarOp::None, &[], 0)
    }

    /// Undefined nodes are interned per instruction: revisiting a block
    /// reaches the same instruction, so values stay stable, and a failure can
    /// name the opcode that gave up.
    pub(super) fn undefined(&mut self, ty: ValueType) -> ValueId {
        let Some(instruction) = self.builder_instruction else {
            return self.intern(ValueKind::Undefined, ty, 0, ScalarOp::None, &[], 0);
        };
        let id = self.intern(ValueKind::Undefined, ty, 0, ScalarOp::None, &[], (1u64 << 32) | instruction.0 as u64);
        self.undefined_origins.entry(id).or_insert(instruction);
        id
    }

    pub(super) fn user_data(&mut self, register: u32) -> ValueId {
        self.intern(ValueKind::UserData, ValueType::U32, register as u64, ScalarOp::None, &[], 0)
    }

    pub(super) fn shader_base(&mut self) -> ValueId {
        self.intern(ValueKind::ShaderBase, ValueType::U64, 0, ScalarOp::None, &[], 0)
    }

    pub(super) fn resource_table_word(&mut self, slot: u32) -> ValueId {
        self.intern(ValueKind::ResourceTableWord, ValueType::U32, slot as u64, ScalarOp::None, &[], 0)
    }

    pub(super) fn handle(&mut self, kind: ValueKind, dwords: &[ValueId]) -> ValueId {
        self.intern(kind, ValueType::U32, 0, ScalarOp::None, dwords, 0)
    }

    pub(super) fn memory_read(&mut self, kind: ValueKind, handle: ValueId, offset: ValueId, memory_index: usize) -> ValueId {
        self.intern(kind, ValueType::U32, memory_index as u64, ScalarOp::None, &[handle, offset], 0)
    }

    pub(super) fn first_lane(&mut self, value: ValueId, active_mask: ValueId, pc: u32) -> ValueId {
        self.intern(ValueKind::FirstLane, ValueType::U32, pc as u64, ScalarOp::None, &[value, active_mask], 0)
    }

    /// A new (not interned) phi of `block`; operands are set later.
    pub(super) fn phi(&mut self, block: usize, ty: ValueType) -> ValueId {
        let id = ValueId(self.nodes.len() as u32);
        self.nodes.push(Node {
            kind: ValueKind::Phi,
            ty,
            op: ScalarOp::None,
            payload: block as u64,
            operands: Box::new([]),
            phi_predecessors: Box::new([]),
        });
        id
    }

    pub(super) fn set_phi_operands(&mut self, phi: ValueId, predecessors: Vec<i32>, operands: Vec<ValueId>) {
        let node = &mut self.nodes[phi.index()];
        node.phi_predecessors = predecessors.into();
        node.operands = operands.into();
    }

    pub(super) fn select(&mut self, condition: ValueId, when_true: ValueId, when_false: ValueId) -> ValueId {
        let c = &self.nodes[condition.index()];
        if c.is_constant() {
            return if c.constant_bool() { when_true } else { when_false };
        }
        if when_true == when_false {
            return when_true;
        }
        let ty = self.nodes[when_true.index()].ty;
        self.intern(ValueKind::Select, ty, 0, ScalarOp::None, &[condition, when_true, when_false], 0)
    }

    /// Keeps the finite result range of S_FF1 even when the input cannot be
    /// resolved on the host; each instruction stays separate because its
    /// input guard can differ.
    pub(super) fn find_lowest_set_bit(&mut self, value: ValueId, pc: u32) -> ValueId {
        if self.is_constant(value) {
            return self.operation(ScalarOp::FindLowestBit32, ValueType::U32, &[value]);
        }
        let result =
            self.intern(ValueKind::Operation, ValueType::U32, 0, ScalarOp::FindLowestBit32, &[value], (1u64 << 32) | pc as u64);
        self.bit_scan_instructions.insert(result, pc);
        result
    }

    pub(super) fn operation(&mut self, op: ScalarOp, ty: ValueType, operands: &[ValueId]) -> ValueId {
        // ALU identities that hold even when the other operand has no
        // provenance (masking an unused register before use).
        if let Some(folded) = self.fold_with_undefined(op, operands) {
            return folded;
        }
        if operands.iter().any(|&o| self.is_undefined(o)) {
            // Keep the operation shape: later identities can still eliminate
            // the bits that do not depend on the undefined input.
            return self.intern(ValueKind::Operation, ty, 0, op, operands, 0);
        }
        // An extracted half of a constructed pair is that half; the low half
        // of a carrying add is the plain add.
        if op == ScalarOp::Extract64 && self.is_constant(operands[1]) {
            let pair = &self.nodes[operands[0].index()];
            if pair.kind == ValueKind::Operation {
                if pair.op == ScalarOp::Construct64 {
                    let half = (self.nodes[operands[1].index()].payload as u32).min(1) as usize;
                    return pair.operands[half];
                }
                if pair.op == ScalarOp::AddCarry32 && self.nodes[operands[1].index()].payload as u32 == 0 {
                    let (a, b) = (pair.operands[0], pair.operands[1]);
                    return self.operation(ScalarOp::IAdd32, ValueType::U32, &[a, b]);
                }
            }
        }
        if let Some(folded) = self.try_fold(op, ty, operands) {
            return folded;
        }
        self.intern(ValueKind::Operation, ty, 0, op, operands, 0)
    }

    fn fold_with_undefined(&mut self, op: ScalarOp, operands: &[ValueId]) -> Option<ValueId> {
        if operands.len() != 2 {
            return None;
        }
        let is = |g: &Self, i: usize, ty: ValueType, value: u64| {
            let n = &g.nodes[operands[i].index()];
            n.is_constant() && n.ty == ty && n.payload == value
        };
        let either = |g: &Self, ty: ValueType, value: u64| is(g, 0, ty, value) || is(g, 1, ty, value);
        match op {
            ScalarOp::And32 if either(self, ValueType::U32, 0) => Some(self.constant(0)),
            ScalarOp::And64 if either(self, ValueType::U64, 0) => Some(self.constant64(0)),
            ScalarOp::Or32 if either(self, ValueType::U32, u32::MAX as u64) => Some(self.constant(u32::MAX)),
            ScalarOp::LogicalAnd if either(self, ValueType::Bool, 0) => Some(self.constant_bool(false)),
            ScalarOp::LogicalOr if either(self, ValueType::Bool, 1) => Some(self.constant_bool(true)),
            _ => None,
        }
    }

    fn try_fold(&mut self, op: ScalarOp, ty: ValueType, operands: &[ValueId]) -> Option<ValueId> {
        let mut values = [0u64; 4];
        for (i, &o) in operands.iter().enumerate() {
            let n = &self.nodes[o.index()];
            if !n.is_constant() {
                return None;
            }
            if i < 4 {
                values[i] = n.payload;
            }
        }
        let result = evaluate_operation(op, &values[..operands.len().min(4)])?;
        Some(match ty {
            ValueType::Bool => self.constant_bool(result != 0),
            ValueType::U64 => self.constant64(result),
            ValueType::U32 => self.constant(result as u32),
        })
    }

    /// Rebuild `value` with every replaced node swapped for its replacement.
    /// Phis are recreated first so a cycle through them terminates.
    pub(super) fn substitute(
        &mut self,
        value: ValueId,
        replacements: &HashMap<ValueId, ValueId>,
        memo: &mut HashMap<ValueId, ValueId>,
    ) -> ValueId {
        if let Some(&r) = replacements.get(&value) {
            return r;
        }
        if let Some(&r) = memo.get(&value) {
            return r;
        }
        let node = self.nodes[value.index()].clone();
        if node.operands.is_empty() {
            memo.insert(value, value);
            return value;
        }
        if node.kind == ValueKind::Phi {
            let phi = self.phi(node.phi_block(), node.ty);
            memo.insert(value, phi);
            let mut operands = Vec::with_capacity(node.operands.len());
            for &o in node.operands.iter() {
                operands.push(if o == value { phi } else { self.substitute(o, replacements, memo) });
            }
            self.set_phi_operands(phi, node.phi_predecessors.to_vec(), operands);
            return phi;
        }
        let mut operands = Vec::with_capacity(node.operands.len());
        let mut changed = false;
        for &o in node.operands.iter() {
            let r = self.substitute(o, replacements, memo);
            changed |= r != o;
            operands.push(r);
        }
        if !changed {
            memo.insert(value, value);
            return value;
        }
        let rebuilt = match node.kind {
            ValueKind::Operation => match self.bit_scan_instructions.get(&value) {
                Some(&pc) => self.find_lowest_set_bit(operands[0], pc),
                None => self.operation(node.op, node.ty, &operands),
            },
            ValueKind::Select => self.select(operands[0], operands[1], operands[2]),
            ValueKind::FirstLane => self.first_lane(operands[0], operands[1], node.payload as u32),
            ValueKind::ScalarAddressWord | ValueKind::ScalarBufferWord => {
                self.memory_read(node.kind, operands[0], operands[1], node.memory_index())
            }
            kind => self.handle(kind, &operands),
        };
        memo.insert(value, rebuilt);
        rebuilt
    }

    /// Every value reachable from `roots`, each with the values that use it.
    pub(super) fn collect_uses(&self, roots: &[ValueId]) -> HashMap<ValueId, Vec<ValueId>> {
        let mut uses: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
        let mut pending: Vec<ValueId> = roots.to_vec();
        let mut visited = HashSet::new();
        while let Some(value) = pending.pop() {
            if !visited.insert(value) {
                continue;
            }
            for &o in self.nodes[value.index()].operands.iter() {
                uses.entry(o).or_default().push(value);
                pending.push(o);
            }
        }
        uses
    }

    // ---- equivalence ----

    /// Structural equality: same kind, type, operation, payload and operands;
    /// phis also draw each operand from the same predecessor; memory reads
    /// compare their access records. An undefined value is never equivalent,
    /// not even to itself. Loop phis that only merge an entry value with
    /// themselves compare as that value.
    pub fn equivalent(&self, left: ValueId, right: ValueId) -> bool {
        let mut ctx = EquivalenceContext::default();
        let mut visited = HashSet::new();
        self.equivalent_in(left, right, &mut ctx, &mut visited)
    }

    fn equivalent_in(
        &self,
        left: ValueId,
        right: ValueId,
        ctx: &mut EquivalenceContext,
        visited: &mut HashSet<(ValueId, ValueId)>,
    ) -> bool {
        if self.is_undefined(left) || self.is_undefined(right) {
            return false;
        }
        if left == right {
            return true;
        }
        let left = self.resolve_for_equivalence(left, ctx);
        let right = self.resolve_for_equivalence(right, ctx);
        if self.is_undefined(left) || self.is_undefined(right) {
            return false;
        }
        if left == right {
            return true;
        }
        let (l, r) = (&self.nodes[left.index()], &self.nodes[right.index()]);
        if l.kind != r.kind || l.ty != r.ty || l.op != r.op || l.operands.len() != r.operands.len() {
            return false;
        }
        match l.kind {
            ValueKind::Constant | ValueKind::UserData | ValueKind::ResourceTableWord | ValueKind::MemoryAperture => {
                return l.payload == r.payload
            }
            ValueKind::ShaderBase => return true,
            _ => {}
        }
        if !visited.insert((left, right)) {
            return true;
        }
        if l.kind == ValueKind::FirstLane && l.payload != r.payload {
            return false;
        }
        match l.kind {
            ValueKind::ScalarAddressWord | ValueKind::ScalarBufferWord => {
                let (li, ri) = (l.memory_index(), r.memory_index());
                if li >= self.memory.len() || ri >= self.memory.len() || !self.memory.get(li).same_access(self.memory.get(ri)) {
                    return false;
                }
            }
            ValueKind::Phi => {
                if l.payload != r.payload || l.phi_predecessors != r.phi_predecessors {
                    return false;
                }
            }
            _ => {}
        }
        for i in 0..l.operands.len() {
            let (a, b) = (self.nodes[left.index()].operands[i], self.nodes[right.index()].operands[i]);
            if !self.equivalent_in(a, b, ctx, visited) {
                return false;
            }
        }
        true
    }

    /// A phi's invariant value, or the phi itself when it merges different
    /// values or is already being resolved further up the stack.
    fn resolve_for_equivalence(&self, value: ValueId, ctx: &mut EquivalenceContext) -> ValueId {
        if self.kind(value) != ValueKind::Phi {
            return value;
        }
        if let Some(cached) = self.invariant_phis.lock().unwrap().get(&value) {
            return cached.unwrap_or(value);
        }
        if !ctx.resolving.insert(value) {
            return value;
        }
        let resolved = self.compute_invariant_phi(value, ctx);
        ctx.resolving.remove(&value);
        self.invariant_phis.lock().unwrap().insert(value, resolved);
        resolved.unwrap_or(value)
    }

    /// A phi whose non-phi operands are all equivalent resolves to that
    /// value; a phi that merges different values resolves to `None`. A
    /// non-phi resolves to itself.
    pub fn resolve_invariant_phi(&self, value: ValueId) -> Option<ValueId> {
        if self.kind(value) != ValueKind::Phi {
            return Some(value);
        }
        if let Some(cached) = self.invariant_phis.lock().unwrap().get(&value) {
            return *cached;
        }
        let mut ctx = EquivalenceContext::default();
        let resolved = self.compute_invariant_phi(value, &mut ctx);
        self.invariant_phis.lock().unwrap().insert(value, resolved);
        resolved
    }

    fn compute_invariant_phi(&self, value: ValueId, ctx: &mut EquivalenceContext) -> Option<ValueId> {
        let mut invariant: Option<ValueId> = None;
        let mut pending = vec![value];
        let mut visited = HashSet::new();
        while let Some(current) = pending.pop() {
            if self.kind(current) == ValueKind::Phi {
                if !visited.insert(current) {
                    continue;
                }
                pending.extend(self.nodes[current.index()].operands.iter().copied());
                continue;
            }
            match invariant {
                None => invariant = Some(current),
                Some(i) => {
                    let mut pairs = HashSet::new();
                    if !self.equivalent_in(i, current, ctx, &mut pairs) {
                        return None;
                    }
                }
            }
        }
        invariant
    }

    // ---- bit scans ----

    /// The S_FF1 behind `selector` only runs on a non-zero input: it is the
    /// fall-through of a `s_cmp_lg x, 0; s_cbranch_scc0` (or `eq/scc1`) guard
    /// on the scanned register, on every incoming edge.
    pub(super) fn has_non_zero_bit_scan_input(&self, selector: ValueId) -> bool {
        let Some(&pc) = self.bit_scan_instructions.get(&selector) else { return false };
        let Some(&block) = self.cfg.block_by_start_pc.get(&pc) else { return false };
        if block == 0 {
            return false;
        }
        let Some(scan) = self.program.instructions.iter().find(|i| i.pc == pc) else { return false };
        if scan.sources.len() != 1 || scan.sources[0].kind != OperandKind::ScalarRegister {
            return false;
        }
        let predecessors = &self.cfg.predecessors[block];
        if predecessors.is_empty() {
            return false;
        }
        for &predecessor in predecessors {
            if predecessor + 1 != block || self.cfg.successors[predecessor].len() != 2 {
                return false;
            }
            let range = self.cfg.blocks[predecessor];
            let tail: Vec<&Instruction> =
                self.program.instructions.iter().filter(|i| i.pc >= range.start_pc && i.pc < range.end_pc).collect();
            if tail.len() < 2 {
                return false;
            }
            let (compare, branch) = (tail[tail.len() - 2], tail[tail.len() - 1]);
            let skips_zero = (matches!(compare.opcode, Opcode::SCmpLgU32 | Opcode::SCmpLgI32) && branch.opcode == Opcode::SCbranchScc0)
                || (matches!(compare.opcode, Opcode::SCmpEqU32 | Opcode::SCmpEqI32) && branch.opcode == Opcode::SCbranchScc1);
            if !skips_zero || compare.sources.len() != 2 || branch.next_pc() != pc {
                return false;
            }
            let source = scan.sources[0];
            if !(compare.sources[0] == source && is_zero_operand(compare.sources[1]))
                && !(compare.sources[1] == source && is_zero_operand(compare.sources[0]))
            {
                return false;
            }
        }
        true
    }

    // ---- diagnostics ----

    /// A short structural description of a value, `depth` levels deep.
    pub fn describe(&self, value: ValueId, depth: usize) -> String {
        let mut out = String::new();
        self.describe_into(&mut out, value, depth);
        out
    }

    fn describe_into(&self, out: &mut String, value: ValueId, depth: usize) {
        let n = self.node(value);
        match n.kind {
            ValueKind::Constant => match n.ty {
                ValueType::U64 => {
                    let _ = write!(out, "0x{:X}ul", n.payload);
                }
                ValueType::Bool => out.push_str(if n.payload != 0 { "true" } else { "false" }),
                ValueType::U32 => {
                    let _ = write!(out, "0x{:X}", n.payload as u32);
                }
            },
            ValueKind::UserData => {
                let _ = write!(out, "UserData(s{})", n.payload);
            }
            ValueKind::MemoryAperture => {
                let _ = write!(out, "Aperture({})", n.payload);
            }
            ValueKind::ResourceTableWord => {
                let _ = write!(out, "Table[{}]", n.payload);
            }
            ValueKind::Undefined => match self.undefined_origin(value) {
                Some((pc, op)) => {
                    let _ = write!(out, "Undefined{value}({}@0x{pc:X})", op.name());
                }
                None => {
                    let _ = write!(out, "Undefined{value}");
                }
            },
            kind => {
                if n.kind == ValueKind::Operation {
                    let _ = write!(out, "{:?}{value}", n.op);
                } else {
                    let _ = write!(out, "{kind:?}{value}");
                }
                if kind == ValueKind::ScalarAddressWord || kind == ValueKind::ScalarBufferWord {
                    let _ = write!(out, "[m{}]", n.payload);
                }
                if depth > 0 && !n.operands.is_empty() {
                    out.push('(');
                    for (i, &o) in n.operands.iter().enumerate() {
                        if i != 0 {
                            out.push(',');
                        }
                        self.describe_into(out, o, depth - 1);
                    }
                    out.push(')');
                }
            }
        }
    }

    /// The instructions behind the undefined leaves `value` rests on.
    pub(super) fn describe_undefined_leaves(&self, value: ValueId) -> String {
        let mut origins: Vec<String> = Vec::new();
        let mut seen = HashSet::new();
        let mut pending = vec![value];
        while let Some(current) = pending.pop() {
            if origins.len() >= 8 {
                break;
            }
            if !seen.insert(current) {
                continue;
            }
            if self.is_undefined(current) {
                if let Some((pc, op)) = self.undefined_origin(current) {
                    let d = format!("{}@0x{pc:X}", op.name());
                    if !origins.contains(&d) {
                        origins.push(d);
                    }
                    continue;
                }
            }
            pending.extend(self.operands(current).iter().copied());
        }
        if origins.is_empty() {
            String::new()
        } else {
            format!(" (undefined from: {})", origins.join(", "))
        }
    }

    /// Some undefined leaf of `value` came from an opcode whose name starts
    /// with `prefix`.
    pub(super) fn has_undefined_origin(&self, value: ValueId, prefix: &str) -> bool {
        let mut seen = HashSet::new();
        let mut pending = vec![value];
        while let Some(current) = pending.pop() {
            if !seen.insert(current) {
                continue;
            }
            if self.is_undefined(current) {
                if let Some((_, op)) = self.undefined_origin(current) {
                    if op.name().starts_with(prefix) {
                        return true;
                    }
                }
            }
            pending.extend(self.operands(current).iter().copied());
        }
        false
    }
}

fn is_zero_operand(operand: Operand) -> bool {
    operand == Operand::source(128, None) || (operand.kind == OperandKind::LiteralConstant && operand.value == 0)
}

/// The 32-bit value of an encoded inline constant.
pub(super) fn inline_constant(operand: Operand) -> Option<u32> {
    match operand.kind {
        OperandKind::LiteralConstant => Some(operand.value),
        OperandKind::EncodedConstant => inline_constants::decode(operand.value),
        _ => None,
    }
}
