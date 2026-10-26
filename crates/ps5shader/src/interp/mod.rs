// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! CPU interpreter for the SSA IR: runs all invocations of one wave in
//! lockstep (branches must be uniform, which the translator guarantees),
//! with a flat guest memory, LDS, per-invocation scratch and captured
//! exports. It is the reference for unit tests of instruction semantics.
//!
//! The one divergent branch it accepts is the shape the ExecRunGrouping
//! pass ([`crate::opt::exec_runs`]) creates: a selection whose only arm is
//! per-invocation arithmetic. Every invocation runs the arm (it has no
//! effects) and the join's phis pick per invocation.

pub mod buffer;
pub mod eval;

use crate::ssa::*;
use eval::{eval, EvalError, Val};
use std::collections::{BTreeMap, HashMap};
use std::fmt;

/// Interpreter failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InterpError {
    Eval { value: u32, op: String, error: EvalError },
    /// Invocations disagree on a branch condition.
    DivergentBranch { block: u32 },
    Unreachable { block: u32 },
    StepLimit,
    Unsupported(String),
}

impl fmt::Display for InterpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterpError::Eval { value, op, error } => write!(f, "%{value} ({op}): {error}"),
            InterpError::DivergentBranch { block } => write!(f, "divergent branch at b{block}"),
            InterpError::Unreachable { block } => write!(f, "reached unreachable b{block}"),
            InterpError::StepLimit => write!(f, "step limit exceeded"),
            InterpError::Unsupported(s) => write!(f, "unsupported: {s}"),
        }
    }
}

impl std::error::Error for InterpError {}

/// Sparse byte-addressed guest memory.
#[derive(Clone, Debug, Default)]
pub struct Memory {
    pages: BTreeMap<u64, Vec<u8>>,
    /// `(start, length, target)`: accesses to `start..start + length` go to
    /// the same offset from `target` (the first matching range wins).
    aliases: Vec<(u64, u64, u64)>,
}

const PAGE: u64 = 4096;

impl Memory {
    /// Make `start..start + length` another view of the bytes at `target`.
    pub fn alias(&mut self, start: u64, length: u64, target: u64) {
        self.aliases.push((start, length, target));
    }

    fn resolve(&self, addr: u64) -> u64 {
        for &(start, length, target) in &self.aliases {
            let off = addr.wrapping_sub(start);
            if off < length {
                return target.wrapping_add(off);
            }
        }
        addr
    }

    pub fn read_u8(&self, addr: u64) -> u8 {
        let addr = self.resolve(addr);
        self.pages.get(&(addr / PAGE)).map_or(0, |p| p[(addr % PAGE) as usize])
    }

    pub fn write_u8(&mut self, addr: u64, v: u8) {
        let addr = self.resolve(addr);
        let page = self.pages.entry(addr / PAGE).or_insert_with(|| vec![0; PAGE as usize]);
        page[(addr % PAGE) as usize] = v;
    }

    pub fn read_u32(&self, addr: u64) -> u32 {
        u32::from_le_bytes([self.read_u8(addr), self.read_u8(addr + 1), self.read_u8(addr + 2), self.read_u8(addr + 3)])
    }

    pub fn write_u32(&mut self, addr: u64, v: u32) {
        for (k, b) in v.to_le_bytes().iter().enumerate() {
            self.write_u8(addr + k as u64, *b);
        }
    }

    pub fn write_words(&mut self, addr: u64, words: &[u32]) {
        for (k, w) in words.iter().enumerate() {
            self.write_u32(addr + 4 * k as u64, *w);
        }
    }

    pub fn read_words(&self, addr: u64, n: usize) -> Vec<u32> {
        (0..n).map(|k| self.read_u32(addr + 4 * k as u64)).collect()
    }
}

/// One export instruction's data.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportRecord {
    pub attr: ExportAttr,
    /// Per invocation: the four dwords and the EXEC bit.
    pub lanes: Vec<([u32; 4], bool)>,
}

/// Image operations are delegated to the test.
pub trait ImageHandler {
    /// `args`: texture descriptor (8 dwords), sampler (4 dwords), address
    /// and data dwords, per invocation (all invocations, with their EXEC
    /// bits). Returns per-invocation result dwords.
    fn image(&mut self, attr: &ImageAttr, lanes: &[ImageLane]) -> Result<Vec<[u32; 4]>, String>;
}

#[derive(Clone, Debug)]
pub struct ImageLane {
    pub texture: [u32; 8],
    pub sampler: [u32; 4],
    pub address: Vec<u32>,
    pub data: Vec<u32>,
    pub exec: bool,
}

/// The local memory a FLAT aperture reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlatAperture {
    /// LDS (SHARED_BASE).
    Shared,
    /// The invocation's scratch (PRIVATE_BASE).
    Private,
}

/// Inputs and state of one wave.
pub struct Machine<'a> {
    pub wave_size: u32,
    /// Number of invocations (lanes `0..lanes` exist).
    pub lanes: u32,
    pub user_data: Vec<u32>,
    /// Per system value, per lane.
    pub system: HashMap<SystemValue, Vec<u32>>,
    pub memory: Memory,
    pub lds: Vec<u8>,
    /// Global data share; accesses past its end read 0 and are dropped.
    pub gds: Vec<u8>,
    pub scratch: Vec<Vec<u8>>,
    /// Scratch the dispatch allocated per invocation: a FLAT access through
    /// the private aperture past it reads 0 and is dropped (None: unbounded).
    pub scratch_dwords: Option<u32>,
    pub exports: Vec<ExportRecord>,
    pub interp: Option<Box<dyn Fn(u8, u8, InterpParam, u32) -> f32 + 'a>>,
    pub images: Option<Box<dyn ImageHandler + 'a>>,
    pub messages: Vec<(u16, u32)>,
    pub step_limit: u64,
    pub barriers: u32,
    /// Invocations that exist (`0..present`; default all lanes): lanes
    /// past them keep registers but never take part in memory accesses,
    /// exports or ballots, whatever EXEC holds.
    pub present: Option<u32>,
    /// The work-group this wave runs in with others (see [`WaveGroup`]).
    pub group: Option<std::sync::Arc<WaveGroup>>,
    /// `s_memrealtime` reads `realtime_base` plus the steps run so far (a
    /// clock that advances while the wave runs); `s_memtime` the steps.
    pub realtime_base: u64,
}

/// Memory a work-group's waves share.
#[derive(Default)]
pub struct GroupMemory {
    pub memory: Memory,
    pub lds: Vec<u8>,
    pub gds: Vec<u8>,
}

#[derive(Default)]
struct GroupState {
    /// A wave is running (one at a time).
    busy: bool,
    shared: GroupMemory,
    /// Waves that have not ended.
    alive: u32,
    /// Waves waiting at the current barrier.
    arrived: u32,
    generation: u64,
}

/// Several waves of one work-group, each run by its own thread: one wave
/// runs at a time and owns the shared memory while it runs; a wave hands
/// the memory over at S_BARRIER, which releases once every wave that has
/// not ended arrives, and when it ends.
pub struct WaveGroup {
    state: std::sync::Mutex<GroupState>,
    cv: std::sync::Condvar,
}

impl WaveGroup {
    pub fn new(waves: u32, shared: GroupMemory) -> Self {
        WaveGroup { state: std::sync::Mutex::new(GroupState { alive: waves, shared, ..GroupState::default() }), cv: std::sync::Condvar::new() }
    }

    /// The shared memory after every wave has ended.
    pub fn into_memory(self) -> GroupMemory {
        self.state.into_inner().unwrap().shared
    }

    fn acquire(&self, m: &mut Machine) {
        let mut st = self.state.lock().unwrap();
        while st.busy {
            st = self.cv.wait(st).unwrap();
        }
        st.busy = true;
        m.memory = std::mem::take(&mut st.shared.memory);
        m.lds = std::mem::take(&mut st.shared.lds);
        m.gds = std::mem::take(&mut st.shared.gds);
    }

    fn release(&self, m: &mut Machine, st: &mut GroupState) {
        st.shared.memory = std::mem::take(&mut m.memory);
        st.shared.lds = std::mem::take(&mut m.lds);
        st.shared.gds = std::mem::take(&mut m.gds);
        st.busy = false;
    }

    fn barrier(&self, m: &mut Machine) {
        {
            let mut st = self.state.lock().unwrap();
            self.release(m, &mut st);
            st.arrived += 1;
            let generation = st.generation;
            if st.arrived >= st.alive {
                st.arrived = 0;
                st.generation += 1;
            }
            self.cv.notify_all();
            while st.generation == generation {
                st = self.cv.wait(st).unwrap();
            }
        }
        self.acquire(m);
    }

    fn end(&self, m: &mut Machine) {
        let mut st = self.state.lock().unwrap();
        self.release(m, &mut st);
        st.alive -= 1;
        if st.arrived > 0 && st.arrived >= st.alive {
            st.arrived = 0;
            st.generation += 1;
        }
        self.cv.notify_all();
    }
}

impl<'a> Machine<'a> {
    pub fn new(wave_size: u32, lanes: u32) -> Self {
        Machine {
            wave_size,
            lanes,
            user_data: Vec::new(),
            system: HashMap::new(),
            memory: Memory::default(),
            lds: vec![0; 65536],
            gds: Vec::new(),
            scratch: vec![Vec::new(); lanes as usize],
            scratch_dwords: None,
            exports: Vec::new(),
            interp: None,
            images: None,
            messages: Vec::new(),
            step_limit: 50_000_000,
            barriers: 0,
            present: None,
            group: None,
            realtime_base: 0,
        }
    }

    /// The data share (LDS or GDS) as dwords: a byte address wraps at
    /// 64 KiB and selects dword `addr[15:2]`; dwords past the end read 0
    /// and writes there are dropped.
    fn share(&self, gds: bool) -> &Vec<u8> {
        if gds {
            &self.gds
        } else {
            &self.lds
        }
    }

    fn share_word(&self, gds: bool, index: u32) -> Option<u32> {
        let s = self.share(gds);
        let a = index as usize * 4;
        s.get(a..a + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    }

    fn set_share_word(&mut self, gds: bool, index: u32, v: u32) {
        let s = if gds { &mut self.gds } else { &mut self.lds };
        let a = index as usize * 4;
        if let Some(b) = s.get_mut(a..a + 4) {
            b.copy_from_slice(&v.to_le_bytes());
        }
    }

    /// Dword index of byte `addr`: GDS addresses wrap at 64 KiB; LDS
    /// addresses do not (past the allocation reads 0, writes drop).
    fn share_index(gds: bool, addr: u32) -> u32 {
        if gds {
            (addr & 0xFFFF) >> 2
        } else {
            addr >> 2
        }
    }

    /// `bytes` (1, 2 or 4) at `addr`, zero-extended.
    fn ds_read(&self, gds: bool, addr: u32, bytes: u32) -> u64 {
        let w = self.share_word(gds, Self::share_index(gds, addr)).unwrap_or(0);
        if bytes >= 4 {
            return w as u64;
        }
        let shift = 8 * (addr & 3);
        ((w >> shift) & ((1u32 << (8 * bytes)) - 1)) as u64
    }

    fn ds_write(&mut self, gds: bool, addr: u32, bytes: u32, v: u64) {
        let index = Self::share_index(gds, addr);
        if bytes >= 4 {
            self.set_share_word(gds, index, v as u32);
            return;
        }
        let Some(old) = self.share_word(gds, index) else { return };
        let shift = 8 * (addr & 3);
        let mask = (((1u64 << (8 * bytes)) - 1) as u32) << shift;
        let new = (old & !mask) | (((v as u32) << shift) & mask);
        self.set_share_word(gds, index, new);
    }

    /// Atomic operand of `bytes` (4 or 8) at `addr`: 64-bit operands are
    /// 8-byte aligned and need both dwords in range.
    fn ds_atomic_read(&self, gds: bool, addr: u32, bytes: u32) -> Option<u64> {
        if bytes == 8 {
            let index = Self::share_index(gds, addr & !7);
            let lo = self.share_word(gds, index)?;
            let hi = self.share_word(gds, index + 1)?;
            Some(lo as u64 | (hi as u64) << 32)
        } else {
            self.share_word(gds, Self::share_index(gds, addr)).map(|w| w as u64)
        }
    }

    fn ds_atomic_write(&mut self, gds: bool, addr: u32, bytes: u32, v: u64) {
        if bytes == 8 {
            let index = Self::share_index(gds, addr & !7);
            self.set_share_word(gds, index, v as u32);
            self.set_share_word(gds, index + 1, (v >> 32) as u32);
        } else {
            self.set_share_word(gds, Self::share_index(gds, addr), v as u32);
        }
    }

    fn scratch_read(&self, lane: usize, addr: u32, bytes: u32) -> u64 {
        let mut v = 0u64;
        for k in 0..bytes {
            let a = (addr + k) as usize;
            let b = self.scratch[lane].get(a).copied().unwrap_or(0);
            v |= (b as u64) << (8 * k);
        }
        v
    }

    fn scratch_write(&mut self, lane: usize, addr: u32, bytes: u32, v: u64) {
        let s = &mut self.scratch[lane];
        for k in 0..bytes {
            let a = (addr + k) as usize;
            if a >= s.len() {
                s.resize(a + 1, 0);
            }
            s[a] = (v >> (8 * k)) as u8;
        }
    }

    /// The memory a compute program's FLAT address reaches through an
    /// aperture, and the byte offset in it (None: a device address).
    fn flat_aperture(addr: u64) -> Option<(FlatAperture, u32)> {
        let high = (addr >> 32) as u32;
        if crate::inline_constants::is_shared_aperture_high(high) {
            Some((FlatAperture::Shared, addr as u32))
        } else if crate::inline_constants::is_private_aperture_high(high) {
            Some((FlatAperture::Private, addr as u32))
        } else {
            None
        }
    }

    /// `bytes` at `offset` lie inside the aperture's allocated memory.
    fn aperture_fits(&self, aperture: FlatAperture, offset: u32, bytes: u32) -> bool {
        let end = offset as u64 + bytes as u64;
        let size = match aperture {
            FlatAperture::Shared => (self.lds.len() as u64) & !3,
            FlatAperture::Private => self.scratch_dwords.map_or(1 << 32, |d| d as u64 * 4),
        };
        end <= size
    }

    /// One element of `bytes` through an aperture, zero-extended; 0 when it
    /// does not lie whole inside the memory.
    fn aperture_read(&self, lane: usize, aperture: FlatAperture, offset: u32, bytes: u32) -> u64 {
        if !self.aperture_fits(aperture, offset, bytes) {
            return 0;
        }
        match aperture {
            FlatAperture::Shared => {
                (0..bytes).fold(0u64, |v, k| v | (self.lds[(offset + k) as usize] as u64) << (8 * k))
            }
            FlatAperture::Private => self.scratch_read(lane, offset, bytes),
        }
    }

    /// Store one element through an aperture; dropped whole when it does
    /// not lie inside the memory.
    fn aperture_write(&mut self, lane: usize, aperture: FlatAperture, offset: u32, bytes: u32, v: u64) {
        if !self.aperture_fits(aperture, offset, bytes) {
            return;
        }
        match aperture {
            FlatAperture::Shared => {
                for k in 0..bytes {
                    self.lds[(offset + k) as usize] = (v >> (8 * k)) as u8;
                }
            }
            FlatAperture::Private => self.scratch_write(lane, offset, bytes, v),
        }
    }

    fn mem_read(&self, addr: u64, bytes: u32) -> u64 {
        let mut v = 0u64;
        for k in 0..bytes {
            v |= (self.memory.read_u8(addr + k as u64) as u64) << (8 * k);
        }
        v
    }

    fn mem_write(&mut self, addr: u64, bytes: u32, v: u64) {
        for k in 0..bytes {
            self.memory.write_u8(addr + k as u64, (v >> (8 * k)) as u8);
        }
    }

    /// Run the function. Errors on undefined behaviour, divergent branches
    /// or unsupported operations.
    pub fn run(&mut self, func: &Function) -> Result<(), InterpError> {
        let group = self.group.clone();
        if let Some(g) = &group {
            g.acquire(self);
        }
        let r = Run::new(self, func).execute();
        if let Some(g) = &group {
            g.end(self);
        }
        r
    }
}

struct Run<'m, 'a, 'f> {
    m: &'m mut Machine<'a>,
    func: &'f Function,
    /// Per value, per lane.
    vals: Vec<Option<Vec<Val>>>,
    steps: u64,
}

impl<'m, 'a, 'f> Run<'m, 'a, 'f> {
    fn new(m: &'m mut Machine<'a>, func: &'f Function) -> Self {
        let n = func.values.len();
        Run { m, func, vals: vec![None; n], steps: 0 }
    }

    fn lanes(&self) -> usize {
        self.m.lanes as usize
    }

    /// The trailing EXEC operand of `d`, cleared for absent invocations.
    fn exec_arg(&self, d: &ValueData) -> Vec<Val> {
        let mut e = self.get(*d.args.last().unwrap());
        if let Some(p) = self.m.present {
            for x in e.iter_mut().skip(p as usize) {
                x[0] = 0;
            }
        }
        e
    }

    fn get(&self, v: Value) -> Vec<Val> {
        let d = self.func.value(v);
        match d.op {
            Op::Const(c) => vec![[c, 0, 0, 0]; self.lanes()],
            Op::Undef => vec![[0; 4]; self.lanes()],
            _ => self.vals[v.index()].clone().unwrap_or_else(|| panic!("%{} read before definition", v.0)),
        }
    }

    fn execute(&mut self) -> Result<(), InterpError> {
        let mut block = self.func.entry;
        let mut prev: Option<Block> = None;
        // A divergent selection just run: (header, arm, invocations that
        // took the arm).
        let mut diverged: Option<(Block, Block, Vec<bool>)> = None;
        loop {
            self.steps += 1;
            if self.steps > self.m.step_limit {
                return Err(InterpError::StepLimit);
            }
            let bd = self.func.block(block);
            // Phis read their operands simultaneously.
            let mut phi_updates = Vec::new();
            for &v in &bd.insts {
                if self.func.op(v) != Op::Phi {
                    break;
                }
                if let Some((header, arm, took)) = &diverged {
                    let kh = bd.preds.iter().position(|b| b == header).expect("phi predecessor");
                    let ka = bd.preds.iter().position(|b| b == arm).expect("phi predecessor");
                    let (from_header, from_arm) = (self.get(self.func.args(v)[kh]), self.get(self.func.args(v)[ka]));
                    let vals = took.iter().enumerate().map(|(l, &t)| if t { from_arm[l] } else { from_header[l] }).collect();
                    phi_updates.push((v, vals));
                    continue;
                }
                let p = prev.expect("phi in entry block");
                let k = bd.preds.iter().position(|&b| b == p).expect("phi predecessor");
                phi_updates.push((v, self.get(self.func.args(v)[k])));
            }
            diverged = None;
            for (v, vals) in phi_updates {
                self.vals[v.index()] = Some(vals);
            }
            for &v in &bd.insts {
                if self.func.op(v) == Op::Phi {
                    continue;
                }
                self.step(v)?;
            }
            let next = match &bd.term {
                Terminator::Branch(t) => *t,
                Terminator::CondBranch { cond, then_block, else_block } => {
                    let c = self.get(*cond);
                    let first = c.first().map_or(0, |x| x[0]);
                    if c.iter().any(|x| x[0] != first) {
                        let arm = *then_block;
                        let ad = self.func.block(arm);
                        let local = |v: &Value| {
                            let op = self.func.op(*v);
                            op.is_pure() && !op.is_subgroup() && op != Op::Phi
                        };
                        let grouped = bd.merge == Merge::Selection { merge: *else_block }
                            && ad.term == Terminator::Branch(*else_block)
                            && ad.insts.iter().all(local);
                        if !grouped {
                            return Err(InterpError::DivergentBranch { block: block.0 });
                        }
                        let join = *else_block;
                        for &v in &ad.insts {
                            self.step(v)?;
                        }
                        diverged = Some((block, arm, c.iter().map(|x| x[0] != 0).collect()));
                        prev = Some(arm);
                        block = join;
                        continue;
                    }
                    if first != 0 {
                        *then_block
                    } else {
                        *else_block
                    }
                }
                Terminator::Return => return Ok(()),
                Terminator::Unreachable | Terminator::None => return Err(InterpError::Unreachable { block: block.0 }),
            };
            prev = Some(block);
            block = next;
        }
    }

    fn err(&self, v: Value, error: EvalError) -> InterpError {
        InterpError::Eval { value: v.0, op: crate::ssa::print::instruction(self.func, v), error }
    }

    fn set(&mut self, v: Value, vals: Vec<Val>) {
        self.vals[v.index()] = Some(vals);
    }

    fn step(&mut self, v: Value) -> Result<(), InterpError> {
        let d = self.func.value(v).clone();
        let n = self.lanes();
        let ws = self.m.wave_size;
        let arg = |r: &Self, i: usize| r.get(d.args[i]);
        match d.op {
            Op::GetReg(_) | Op::SetReg(_) => return Err(InterpError::Unsupported("register operations (run SSA construction first)".into())),
            Op::LaneId => self.set(v, (0..n).map(|l| [l as u64, 0, 0, 0]).collect()),
            Op::Ballot => {
                let b = arg(self, 0);
                let mut mask = 0u64;
                let present = self.m.present.map_or(b.len(), |p| p as usize);
                for (l, x) in b.iter().enumerate().take(present) {
                    if x[0] != 0 {
                        mask |= 1 << l;
                    }
                }
                let val = [mask & 0xFFFF_FFFF, mask >> 32, 0, 0];
                self.set(v, vec![val; n]);
            }
            Op::InverseBallot => {
                let lo = arg(self, 0);
                let hi = arg(self, 1);
                self.set(
                    v,
                    (0..n)
                        .map(|l| {
                            let mask = (lo[l][0] & 0xFFFF_FFFF) | ((hi[l][0] & 0xFFFF_FFFF) << 32);
                            [(mask >> l) & 1, 0, 0, 0]
                        })
                        .collect(),
                );
            }
            Op::ReadLane | Op::Shuffle => {
                let src = arg(self, 0);
                let idx = arg(self, 1);
                let mut out = Vec::with_capacity(n);
                for l in 0..n {
                    let i = idx[l][0];
                    if d.op == Op::ReadLane && idx.iter().any(|x| x[0] != i) {
                        return Err(self.err(v, EvalError::Undefined("read_lane index is not uniform".into())));
                    }
                    if i >= ws as u64 {
                        return Err(self.err(v, EvalError::Undefined(format!("lane index {i} >= wave size"))));
                    }
                    out.push(if (i as usize) < n { src[i as usize] } else { [0; 4] });
                }
                self.set(v, out);
            }
            Op::UserData(i) => {
                let x = self.m.user_data.get(i as usize).copied().unwrap_or(0);
                self.set(v, vec![[x as u64, 0, 0, 0]; n]);
            }
            Op::SystemValue(s) => {
                let vals = self.m.system.get(&s).cloned().unwrap_or_default();
                self.set(v, (0..n).map(|l| [vals.get(l).copied().unwrap_or(0) as u64, 0, 0, 0]).collect());
            }
            Op::CodeAddress => self.set(v, vec![[self.func.code_address, 0, 0, 0]; n]),
            Op::Clock { realtime } => {
                let t = if realtime { self.m.realtime_base.wrapping_add(self.steps) } else { self.steps };
                self.set(v, vec![[t, 0, 0, 0]; n])
            }
            Op::InterpParam { attr, chan, param } => {
                let f = self.m.interp.as_ref().ok_or_else(|| InterpError::Unsupported("interpolation without inputs".into()))?;
                let out = (0..n).map(|l| [f(attr, chan, param, l as u32).to_bits() as u64, 0, 0, 0]).collect();
                self.set(v, out);
            }
            Op::ScalarLoad { .. } => {
                let a = arg(self, 0);
                let out = a.iter().map(|x| [self.m.memory.read_u32(x[0]) as u64, 0, 0, 0]).collect();
                self.set(v, out);
            }
            Op::ScalarBufferLoad { .. } => {
                let desc = arg(self, 0);
                let off = arg(self, 1);
                let mut out = Vec::new();
                for l in 0..n {
                    let dsc = [desc[l][0] as u32, desc[l][1] as u32, desc[l][2] as u32, desc[l][3] as u32];
                    let base = dsc[0] as u64 | (((dsc[1] & 0xFFFF) as u64) << 32);
                    let stride = (dsc[1] >> 16) & 0x3FFF;
                    let records = dsc[2] as u64;
                    let limit = if stride == 0 { records } else { records * stride as u64 };
                    let o = off[l][0];
                    // (base & ~3) + (offset & ~3): the base is aligned on
                    // its own (the translator aligns the offset).
                    let x = if o & 3 == 0 && o + 4 <= limit { self.m.memory.read_u32((base & !3) + o) } else { 0 };
                    out.push([x as u64, 0, 0, 0]);
                }
                self.set(v, out);
            }
            Op::BufferLoad { attr, data } | Op::BufferStore { attr, data } => {
                let store = matches!(d.op, Op::BufferStore { .. });
                let desc = arg(self, 0);
                let vindex = arg(self, 1);
                let voffset = arg(self, 2);
                let soffset = arg(self, 3);
                let value = if store { Some(arg(self, 4)) } else { None };
                let exec = self.exec_arg(&d);
                let mut out = Vec::with_capacity(n);
                for l in 0..n {
                    let dsc = buffer::BufferDescriptor::new([desc[l][0] as u32, desc[l][1] as u32, desc[l][2] as u32, desc[l][3] as u32]);
                    let a = buffer::BufferLaneAddress::new(&attr, &dsc, l as u32, vindex[l][0] as u32, voffset[l][0] as u32, soffset[l][0] as u32);
                    let active = exec[l][0] != 0;
                    let data_in = value.as_ref().map(|v| v[l]);
                    let result = match data {
                        BufferData::Raw(access) => self.buffer_raw(&dsc, &a, access, store.then_some(active), data_in),
                        BufferData::Format { components, typed_format, d16 } => {
                            self.buffer_format(&dsc, &a, components as u32, typed_format, d16, store.then_some(active), data_in)?
                        }
                    };
                    out.push(result);
                }
                if !store {
                    self.set(v, out);
                }
            }
            Op::BufferAtomic { attr, op } => {
                let desc = arg(self, 0);
                let vindex = arg(self, 1);
                let voffset = arg(self, 2);
                let soffset = arg(self, 3);
                let data = arg(self, 4);
                let cmp = if matches!(op, AtomicOp::CmpSwap | AtomicOp::FCmpSwap | AtomicOp::MaskOr | AtomicOp::Wrap) { Some(arg(self, 5)) } else { None };
                let exec = self.exec_arg(&d);
                let bytes = if d.ty == Type::U64 { 8 } else { 4 };
                let mut out = vec![[0u64; 4]; n];
                for l in 0..n {
                    if exec[l][0] == 0 {
                        continue;
                    }
                    let dsc = buffer::BufferDescriptor::new([desc[l][0] as u32, desc[l][1] as u32, desc[l][2] as u32, desc[l][3] as u32]);
                    let a = buffer::BufferLaneAddress::new(&attr, &dsc, l as u32, vindex[l][0] as u32, voffset[l][0] as u32, soffset[l][0] as u32);
                    if !buffer::buffer_in_bounds(&dsc, &a, 0, bytes) {
                        continue;
                    }
                    // A 32-bit atomic acts on the dword holding its address
                    // (the two low bits are ignored in DWORD alignment mode).
                    let addr = buffer::buffer_byte_address(&dsc, &a, 0);
                    let addr = if bytes == 4 { addr & !3 } else { addr };
                    let old = self.m.mem_read(addr, bytes);
                    let new = atomic(op, old, data[l][0], cmp.as_ref().map(|c| c[l][0]), bytes * 8);
                    self.m.mem_write(addr, bytes, new);
                    out[l] = [old, 0, 0, 0];
                }
                self.set(v, out);
            }
            Op::GlobalLoad { access, flat: true, .. } if self.func.stage == Stage::Compute => {
                let a = arg(self, 0);
                let exec = self.exec_arg(&d);
                let mut out = Vec::with_capacity(n);
                for l in 0..n {
                    let mut r = [0u64; 4];
                    if exec[l][0] != 0 {
                        let addr = a[l][0];
                        let read = |run: &Self, k: u32, bytes: u32| match Machine::flat_aperture(addr) {
                            Some((aperture, offset)) => run.m.aperture_read(l, aperture, offset.wrapping_add(4 * k), bytes),
                            None => run.m.mem_read(addr.wrapping_add(4 * k as u64), bytes),
                        };
                        match access {
                            Access::Dwords(words) => {
                                for (c, slot) in r.iter_mut().enumerate().take(words as usize) {
                                    *slot = read(self, c as u32, 4);
                                }
                            }
                            Access::U8 => r[0] = read(self, 0, 1),
                            Access::I8 => r[0] = (read(self, 0, 1) as u8 as i8 as i32 as u32) as u64,
                            Access::U16 => r[0] = read(self, 0, 2),
                            Access::I16 => r[0] = (read(self, 0, 2) as u16 as i16 as i32 as u32) as u64,
                        }
                    }
                    out.push(r);
                }
                self.set(v, out);
            }
            Op::GlobalStore { access, flat: true, .. } if self.func.stage == Stage::Compute => {
                let a = arg(self, 0);
                let data = arg(self, 1);
                let exec = self.exec_arg(&d);
                for l in 0..n {
                    if exec[l][0] == 0 {
                        continue;
                    }
                    let addr = a[l][0];
                    let write = |run: &mut Self, k: u32, bytes: u32, val: u64| match Machine::flat_aperture(addr) {
                        Some((aperture, offset)) => run.m.aperture_write(l, aperture, offset.wrapping_add(4 * k), bytes, val),
                        None => run.m.mem_write(addr.wrapping_add(4 * k as u64), bytes, val),
                    };
                    match access {
                        Access::Dwords(k) => {
                            for c in 0..k as usize {
                                write(self, c as u32, 4, data[l][c]);
                            }
                        }
                        a => write(self, 0, a.bytes(), data[l][0]),
                    }
                }
            }
            Op::GlobalAtomic { op, flat: true, .. } if self.func.stage == Stage::Compute => {
                let a = arg(self, 0);
                let data = arg(self, 1);
                let cmp = if matches!(op, AtomicOp::CmpSwap | AtomicOp::FCmpSwap | AtomicOp::MaskOr | AtomicOp::Wrap) { Some(arg(self, 2)) } else { None };
                let exec = self.exec_arg(&d);
                let bytes = if d.ty == Type::U64 { 8 } else { 4 };
                let mut out = vec![[0u64; 4]; n];
                for l in 0..n {
                    if exec[l][0] == 0 {
                        continue;
                    }
                    let addr = a[l][0];
                    let c = cmp.as_ref().map(|c| c[l][0]);
                    match Machine::flat_aperture(addr) {
                        // An aperture operand must be aligned to its size and
                        // lie inside the memory; otherwise nothing happens.
                        Some((aperture, offset)) => {
                            if offset % bytes != 0 || !self.m.aperture_fits(aperture, offset, bytes) {
                                continue;
                            }
                            let old = self.m.aperture_read(l, aperture, offset, bytes);
                            let new = atomic(op, old, data[l][0], c, bytes * 8);
                            self.m.aperture_write(l, aperture, offset, bytes, new);
                            out[l] = [old, 0, 0, 0];
                        }
                        None => {
                            let old = self.m.mem_read(addr, bytes);
                            let new = atomic(op, old, data[l][0], c, bytes * 8);
                            self.m.mem_write(addr, bytes, new);
                            out[l] = [old, 0, 0, 0];
                        }
                    }
                }
                self.set(v, out);
            }
            Op::GlobalLoad { access, .. } | Op::ScratchLoad { access, .. } | Op::LdsLoad { access } | Op::GdsLoad { access } => {
                let a = arg(self, 0);
                let exec = self.exec_arg(&d);
                let mut out = Vec::with_capacity(n);
                for l in 0..n {
                    let mut r = [0u64; 4];
                    let words = access.dwords();
                    let read = |run: &Self, off: u64, bytes: u32| -> u64 {
                        match d.op {
                            Op::GlobalLoad { .. } => run.m.mem_read(a[l][0].wrapping_add(off), bytes),
                            Op::ScratchLoad { .. } => run.m.scratch_read(l, (a[l][0] as u32).wrapping_add(off as u32), bytes),
                            op => run.m.ds_read(matches!(op, Op::GdsLoad { .. }), (a[l][0] as u32).wrapping_add(off as u32), bytes),
                        }
                    };
                    let _ = exec[l];
                    match access {
                        Access::Dwords(_) => {
                            for (c, slot) in r.iter_mut().enumerate().take(words as usize) {
                                *slot = read(self, 4 * c as u64, 4);
                            }
                        }
                        Access::U8 => r[0] = read(self, 0, 1),
                        Access::I8 => r[0] = (read(self, 0, 1) as u8 as i8 as i32 as u32) as u64,
                        Access::U16 => r[0] = read(self, 0, 2),
                        Access::I16 => r[0] = (read(self, 0, 2) as u16 as i16 as i32 as u32) as u64,
                    }
                    out.push(r);
                }
                self.set(v, out);
            }
            Op::GlobalStore { access, .. } | Op::ScratchStore { access, .. } | Op::LdsStore { access } | Op::GdsStore { access } => {
                let a = arg(self, 0);
                let data = arg(self, 1);
                let exec = self.exec_arg(&d);
                for l in 0..n {
                    if exec[l][0] == 0 {
                        continue;
                    }
                    let write = |run: &mut Self, off: u64, bytes: u32, val: u64| match d.op {
                        Op::GlobalStore { .. } => run.m.mem_write(a[l][0].wrapping_add(off), bytes, val),
                        Op::ScratchStore { .. } => run.m.scratch_write(l, (a[l][0] as u32).wrapping_add(off as u32), bytes, val),
                        op => run.m.ds_write(matches!(op, Op::GdsStore { .. }), (a[l][0] as u32).wrapping_add(off as u32), bytes, val),
                    };
                    match access {
                        Access::Dwords(k) => {
                            for c in 0..k as usize {
                                write(self, 4 * c as u64, 4, data[l][c]);
                            }
                        }
                        a => write(self, 0, a.bytes(), data[l][0]),
                    }
                }
            }
            Op::GlobalAtomic { op, .. } | Op::LdsAtomic { op } | Op::GdsAtomic { op } => {
                let a = arg(self, 0);
                let data = arg(self, 1);
                let cmp = if matches!(op, AtomicOp::CmpSwap | AtomicOp::FCmpSwap | AtomicOp::MaskOr | AtomicOp::Wrap) { Some(arg(self, 2)) } else { None };
                let exec = self.exec_arg(&d);
                let bytes = if d.ty == Type::U64 { 8 } else { 4 };
                let mut out = vec![[0u64; 4]; n];
                for l in 0..n {
                    if exec[l][0] == 0 {
                        continue;
                    }
                    let gds = matches!(d.op, Op::GdsAtomic { .. });
                    let old = match d.op {
                        Op::GlobalAtomic { .. } => self.m.mem_read(a[l][0], bytes),
                        _ => match self.m.ds_atomic_read(gds, a[l][0] as u32, bytes) {
                            Some(old) => old,
                            None => continue,
                        },
                    };
                    let new = atomic(op, old, data[l][0], cmp.as_ref().map(|c| c[l][0]), bytes * 8);
                    match d.op {
                        Op::GlobalAtomic { .. } => self.m.mem_write(a[l][0], bytes, new),
                        _ => self.m.ds_atomic_write(gds, a[l][0] as u32, bytes, new),
                    }
                    out[l] = [old, 0, 0, 0];
                }
                self.set(v, out);
            }
            Op::LdsAppend { consume } | Op::GdsAppend { consume } => {
                // The counter dword is addressed without the 64 KiB wrap;
                // past the end the counter is untouched and 0 returned.
                let gds = matches!(d.op, Op::GdsAppend { .. });
                let a = arg(self, 0);
                let c = arg(self, 1);
                let (addr, count) = (a[0][0] as u32, c[0][0] as u32);
                let index = addr >> 2;
                let old = match self.m.share_word(gds, index) {
                    Some(old) => {
                        let new = if consume { old.wrapping_sub(count) } else { old.wrapping_add(count) };
                        self.m.set_share_word(gds, index, new);
                        old
                    }
                    None => 0,
                };
                self.set(v, vec![[old as u64, 0, 0, 0]; n]);
            }
            Op::Image(attr) => {
                let mut lanes_in = Vec::with_capacity(n);
                let t0 = arg(self, 0);
                let t1 = arg(self, 1);
                let s = arg(self, 2);
                let exec = self.exec_arg(&d);
                let addr: Vec<Vec<Val>> = (0..attr.address_dwords as usize).map(|k| arg(self, 3 + k)).collect();
                let data: Vec<Vec<Val>> = (0..attr.data_dwords as usize).map(|k| arg(self, 3 + attr.address_dwords as usize + k)).collect();
                for l in 0..n {
                    let mut texture = [0u32; 8];
                    for k in 0..4 {
                        texture[k] = t0[l][k] as u32;
                        texture[4 + k] = t1[l][k] as u32;
                    }
                    lanes_in.push(ImageLane {
                        texture,
                        sampler: [s[l][0] as u32, s[l][1] as u32, s[l][2] as u32, s[l][3] as u32],
                        address: addr.iter().map(|a| a[l][0] as u32).collect(),
                        data: data.iter().map(|a| a[l][0] as u32).collect(),
                        exec: exec[l][0] != 0,
                    });
                }
                let handler = self.m.images.as_mut().ok_or_else(|| InterpError::Unsupported("image operation without a handler".into()))?;
                let res = handler.image(&attr, &lanes_in).map_err(InterpError::Unsupported)?;
                if d.ty != Type::Void {
                    self.set(v, res.iter().map(|r| [r[0] as u64, r[1] as u64, r[2] as u64, r[3] as u64]).collect());
                }
            }
            Op::Export(attr) => {
                let srcs: Vec<Vec<Val>> = (0..d.args.len() - 1).map(|k| arg(self, k)).collect();
                let exec = self.exec_arg(&d);
                let lanes = (0..n)
                    .map(|l| {
                        let mut w = [0u32; 4];
                        for (k, s) in srcs.iter().enumerate().take(4) {
                            w[k] = s[l][0] as u32;
                        }
                        (w, exec[l][0] != 0)
                    })
                    .collect();
                self.m.exports.push(ExportRecord { attr, lanes });
            }
            Op::Barrier => {
                self.m.barriers += 1;
                if let Some(g) = self.m.group.clone() {
                    g.barrier(self.m);
                }
            }
            Op::SubgroupLdsBarrier | Op::DataCacheInvalidate => {}
            Op::SendMessage { message } => {
                let m0 = arg(self, 0);
                self.m.messages.push((message, m0.first().map_or(0, |x| x[0] as u32)));
            }
            Op::Phi | Op::Const(_) | Op::Undef => {}
            op => {
                let args: Vec<Vec<Val>> = d.args.iter().map(|&a| self.get(a)).collect();
                let types: Vec<Type> = d.args.iter().map(|&a| self.func.ty(a)).collect();
                let mut out = Vec::with_capacity(n);
                for l in 0..n {
                    let lane_args: Vec<(Type, Val)> = types.iter().zip(&args).map(|(t, a)| (*t, a[l])).collect();
                    match eval(op, d.ty, &lane_args) {
                        Ok(r) => out.push(r),
                        Err(EvalError::Undefined(s)) => {
                            // Undefined results are allowed in invocations
                            // whose result is discarded; report only when
                            // every invocation hits it... keep strict: the
                            // translator must avoid UB.
                            return Err(self.err(v, EvalError::Undefined(format!("lane {l}: {s}"))));
                        }
                        Err(e) => return Err(self.err(v, e)),
                    }
                }
                self.set(v, out);
            }
        }
        Ok(())
    }
}

/// Memory float min/max on raw bits: a signalling NaN (source first, then
/// memory) is returned quieted; with two NaNs the source wins; a single
/// quiet NaN yields the other operand; otherwise the total order with
/// -0 < +0 decides and ties keep the memory value.
pub fn atomic_float_min_max(bits: u32, old: u64, src: u64, max: bool) -> u64 {
    let sign = 1u64 << (bits - 1);
    let (infinity, quiet) = if bits == 64 { (0x7FF0_0000_0000_0000u64, 0x0008_0000_0000_0000u64) } else { (0x7F80_0000, 0x0040_0000) };
    let nan = |v: u64| v & (sign - 1) > infinity;
    let signaling = |v: u64| nan(v) && v & quiet == 0;
    let key = |v: u64| if v & sign != 0 { !v & (sign | (sign - 1)) } else { v | sign };
    if signaling(src) {
        return src | quiet;
    }
    if signaling(old) {
        return old | quiet;
    }
    if nan(src) {
        return if nan(old) { src } else { old };
    }
    if nan(old) {
        return src;
    }
    let better = if max { key(src) > key(old) } else { key(src) < key(old) };
    if better {
        src
    } else {
        old
    }
}

/// DS_ADD_F32 result (see [`AtomicOp::DsFAdd`]).
pub fn ds_add_f32(old: u32, data: u32, flush_inputs: bool, flush_result: bool) -> u32 {
    let nan = |x: u32| x & 0x7FFF_FFFF > 0x7F80_0000;
    let flush = |x: u32, on: bool| if on && x & 0x7F80_0000 == 0 { x & 0x8000_0000 } else { x };
    if nan(data) {
        return data | 0x0040_0000;
    }
    if nan(old) {
        return old | 0x0040_0000;
    }
    let sum = f32::from_bits(flush(old, flush_inputs)) + f32::from_bits(flush(data, flush_inputs));
    let r = flush(sum.to_bits(), flush_result);
    if nan(r) {
        0xFFC0_0000
    } else {
        r
    }
}

/// DS_MIN_F32 / DS_MAX_F32 result (see [`AtomicOp::DsFMin`]).
pub fn ds_float_min_max(old: u32, src: u32, max: bool) -> u32 {
    struct Class {
        nan: bool,
        signaling: bool,
        key: u32,
    }
    let classify = |bits: u32| {
        let abs = bits & 0x7FFF_FFFF;
        let nan = abs > 0x7F80_0000;
        let signaling = nan && bits & 0x0040_0000 == 0;
        let sign = bits & 0x8000_0000;
        let key_bits = if abs < 0x0080_0000 {
            if sign == 0 && abs != 0 {
                1
            } else {
                sign
            }
        } else {
            bits
        };
        let key = if key_bits & 0x8000_0000 != 0 { !key_bits } else { key_bits | 0x8000_0000 };
        Class { nan, signaling, key }
    };
    let (l, r) = (classify(old), classify(src));
    let choose_source = if max { r.key > l.key } else { r.key <= l.key };
    let mut result = if choose_source { src } else { old };
    if l.nan {
        result = src;
    }
    if r.nan {
        result = old;
    }
    if r.signaling {
        result = src | 0x0040_0000;
    }
    if l.signaling {
        result = old | 0x0040_0000;
    }
    result
}

/// Read-modify-write result.
pub fn atomic(op: AtomicOp, old: u64, data: u64, cmp: Option<u64>, bits: u32) -> u64 {
    let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let s = |x: u64| -> i64 {
        let sh = 64 - bits;
        ((x << sh) as i64) >> sh
    };
    let r = match op {
        AtomicOp::Swap => data,
        AtomicOp::CmpSwap => {
            if Some(old & mask) == cmp.map(|c| c & mask) {
                data
            } else {
                old
            }
        }
        AtomicOp::Add => old.wrapping_add(data),
        AtomicOp::Sub => old.wrapping_sub(data),
        AtomicOp::SMin => {
            if s(old) <= s(data) {
                old
            } else {
                data
            }
        }
        AtomicOp::SMax => {
            if s(old) >= s(data) {
                old
            } else {
                data
            }
        }
        AtomicOp::UMin => (old & mask).min(data & mask),
        AtomicOp::UMax => (old & mask).max(data & mask),
        AtomicOp::And => old & data,
        AtomicOp::Or => old | data,
        AtomicOp::Xor => old ^ data,
        AtomicOp::Inc => {
            if (old & mask) >= (data & mask) {
                0
            } else {
                old + 1
            }
        }
        AtomicOp::Dec => {
            if old & mask == 0 || (old & mask) > (data & mask) {
                data
            } else {
                old - 1
            }
        }
        AtomicOp::FMin | AtomicOp::FMax => {
            atomic_float_min_max(bits, old & mask, data & mask, op == AtomicOp::FMax)
        }
        AtomicOp::DsFMin | AtomicOp::DsFMax => ds_float_min_max(old as u32, data as u32, op == AtomicOp::DsFMax) as u64,
        AtomicOp::RSub => data.wrapping_sub(old),
        AtomicOp::MaskOr => (old & !data) | cmp.unwrap_or(0),
        AtomicOp::Wrap => {
            if old & mask >= data & mask {
                old.wrapping_sub(data)
            } else {
                old.wrapping_add(cmp.unwrap_or(0))
            }
        }
        AtomicOp::DsFAdd { flush_inputs, flush_result } => ds_add_f32(old as u32, data as u32, flush_inputs, flush_result) as u64,
        AtomicOp::FAdd => {
            if bits == 64 {
                (f64::from_bits(old) + f64::from_bits(data)).to_bits()
            } else {
                (f32::from_bits(old as u32) + f32::from_bits(data as u32)).to_bits() as u64
            }
        }
        AtomicOp::FCmpSwap => {
            let eq = if bits == 64 {
                f64::from_bits(old) == f64::from_bits(cmp.unwrap_or(0))
            } else {
                f32::from_bits(old as u32) == f32::from_bits(cmp.unwrap_or(0) as u32)
            };
            if eq {
                data
            } else {
                old
            }
        }
        AtomicOp::CSub => {
            if (old & mask) >= (data & mask) {
                old - data
            } else {
                0
            }
        }
    };
    r & mask
}
