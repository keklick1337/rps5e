// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Materialisation cache.
//!
//! Materialisation is a pure function of the plan, the draw's user data,
//! the shader base, the compute state and the guest words it reads. The
//! cache records exactly those words and reuses the result while every one
//! is unchanged (checked through a resident-bytes reader that never
//! synchronises). Any failed read or byte difference misses; nothing is
//! guessed. When only words read for the flattened table changed, only the
//! table is evaluated again.
//!
//! Plans whose materialisation depends only on their descriptor words, table
//! words and device-address bases take a cheaper path first
//! ([`ReplayProgram`]): the compiled evaluation runs over the draw's user
//! data and resident guest words, and the result is matched against
//! templates keyed by the plan alone. A template keeps the specialization
//! and snapshot layout of one [`specialization_signature`] (formats, types,
//! strides, swizzles, levels); a draw with the same signature only patches
//! its words in (addresses, sizes, table words), so user data that changes
//! on every draw (constant-buffer pointers in a ring) no longer re-walks
//! descriptor tables or re-decides the specialization. A different
//! signature builds a new template.

use super::evaluator::{ComputeSelectorState, ResidentReader, ResourceRuntimeInputs, WordReader};
use super::materialize::{
    materialize, materialize_replayed, patch_replayed, patch_replayed_into, specialization_signature, try_evaluate_table, MaterializationError, ResourceSnapshot,
    ResourceSpecialization, SnapshotLayout,
};
use super::plan::{ShaderResourcePlan, WRITTEN_RANGE_DWORD_COUNT};
use super::replay::{ReplayAbort, ReplayProgram, ReplayReader, ReplayResult, ReplayScratch};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;

const MAX_VARIANTS: usize = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStatistics {
    pub hits: u64,
    pub misses: u64,
    pub uncacheable: u64,
    pub table_refreshes: u64,
    pub stale: u64,
    pub stale_unreadable: u64,
    /// Replayed draws: words identical to the template's last draw /
    /// patched into a template with the same signature / new templates /
    /// handed to the recording path (unreadable or failing words).
    pub replay_same: u64,
    pub replay_patched: u64,
    pub replay_templates: u64,
    pub replay_fallbacks: u64,
    /// Patched snapshots written into a recycled snapshot (no allocation).
    pub replay_recycled: u64,
}

/// Templates kept per plan (one per specialization signature); 8 with
/// `RPS5E_LAYOUT_INTERN=0` (a plan sampling many texture formats and
/// swizzles rebuilt its templates over and over).
fn max_templates() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| if std::env::var("RPS5E_LAYOUT_INTERN").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")) { 8 } else { 16 })
}

/// Snapshots handed out by patched replays that are kept for reuse: one
/// nobody else holds any more is written over by a later patch.
const RECYCLED_SNAPSHOTS: usize = 32;

/// A snapshot of an earlier patch that nobody holds any more, to be
/// written over (None: none, or recycling is off).
fn take_recycled(recycled: &mut std::collections::VecDeque<Arc<ResourceSnapshot>>) -> Option<Arc<ResourceSnapshot>> {
    if !recycle_enabled() {
        return None;
    }
    // The most recently recycled free snapshot (its memory is the most
    // likely still cached). The count (a plain load) passes over
    // snapshots still held without the atomic exchange of `Arc::get_mut`.
    let i = recycled.iter_mut().rposition(|s| Arc::strong_count(s) == 1 && Arc::get_mut(s).is_some())?;
    recycled.remove(i)
}

fn recycle(recycled: &mut std::collections::VecDeque<Arc<ResourceSnapshot>>, snapshot: Arc<ResourceSnapshot>) {
    if !recycle_enabled() {
        return;
    }
    if recycled.len() >= RECYCLED_SNAPSHOTS {
        recycled.pop_front();
    }
    recycled.push_back(snapshot);
}

/// `RPS5E_PATCH_IN_PLACE=0`: every patched replay allocates its snapshot.
fn recycle_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_PATCH_IN_PLACE").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")))
}

struct Template {
    signature: Vec<u32>,
    layout: SnapshotLayout,
    specialization: Arc<ResourceSpecialization>,
    last: ReplayResult,
    last_user_data: Vec<u32>,
    last_snapshot: Arc<ResourceSnapshot>,
}

type MaterializeResult = Result<(Arc<ResourceSnapshot>, Arc<ResourceSpecialization>), MaterializationError>;

struct Entry {
    plan: u64,
    user_data: Vec<u32>,
    shader_base: u64,
    compute_state: Option<ComputeSelectorState>,
    range_addresses: Vec<u64>,
    range_offsets: Vec<usize>,
    range_lengths: Vec<usize>,
    range_clean: Vec<bool>,
    /// Per recorded dword: read only while the flattened table was
    /// evaluated.
    word_table_only: Vec<bool>,
    /// The table has the plan's plain layout (nothing extended it).
    table_refreshable: bool,
    bytes: Vec<u8>,
    snapshot: Arc<ResourceSnapshot>,
    specialization: Arc<ResourceSpecialization>,
}

impl Entry {
    fn matches(&self, plan: u64, inputs: &ResourceRuntimeInputs<'_>) -> bool {
        self.plan == plan && self.shader_base == inputs.shader_base && self.compute_state == inputs.compute_state && self.user_data == inputs.user_data
    }

    /// The byte offset of a recorded dword.
    fn find_word(&self, address: u64) -> Option<usize> {
        let found = self.range_addresses.partition_point(|&a| a <= address).checked_sub(1)?;
        let delta = address - self.range_addresses[found];
        (delta % 4 == 0 && delta < self.range_lengths[found] as u64).then(|| self.range_offsets[found] + delta as usize)
    }
}

#[derive(Default)]
struct Recorder {
    reads: RefCell<Vec<(u64, u32, bool, bool)>>,
    in_table: Cell<bool>,
    failed: Cell<bool>,
}

impl Recorder {
    fn record(&self, inner: WordReader<'_>, address: u64, clean: bool) -> Option<u32> {
        match inner(address) {
            Some(word) => {
                self.reads.borrow_mut().push((address, word, clean, self.in_table.get()));
                Some(word)
            }
            None => {
                self.failed.set(true);
                None
            }
        }
    }

    fn build(&self, plan: &ShaderResourcePlan, inputs: &ResourceRuntimeInputs<'_>, snapshot: ResourceSnapshot, specialization: ResourceSpecialization) -> Entry {
        let mut reads = self.reads.borrow_mut();
        reads.sort_by_key(|r| r.0);
        let mut entry = Entry {
            plan: plan.id,
            user_data: inputs.user_data.to_vec(),
            shader_base: inputs.shader_base,
            compute_state: inputs.compute_state,
            range_addresses: Vec::new(),
            range_offsets: Vec::new(),
            range_lengths: Vec::new(),
            range_clean: Vec::new(),
            word_table_only: Vec::new(),
            table_refreshable: snapshot.flattened_resource_table.len()
                == plan.table_reads.len() + plan.written_range_count() * WRITTEN_RANGE_DWORD_COUNT,
            bytes: Vec::new(),
            snapshot: Arc::new(snapshot),
            specialization: Arc::new(specialization),
        };
        let mut previous = u64::MAX;
        let mut end = 0u64;
        for &(address, word, clean, table) in reads.iter() {
            if address == previous {
                *entry.range_clean.last_mut().unwrap() |= clean;
                *entry.word_table_only.last_mut().unwrap() &= table;
                continue;
            }
            let word_index = entry.word_table_only.len();
            if entry.range_addresses.is_empty() || address != end {
                entry.range_addresses.push(address);
                entry.range_offsets.push(word_index * 4);
                entry.range_lengths.push(0);
                entry.range_clean.push(false);
            }
            *entry.range_lengths.last_mut().unwrap() += 4;
            *entry.range_clean.last_mut().unwrap() |= clean;
            entry.word_table_only.push(table);
            entry.bytes.extend_from_slice(&word.to_le_bytes());
            previous = address;
            end = address + 4;
        }
        entry
    }
}

/// The replay path of one plan: its compiled evaluation (None: not
/// eligible) and its templates, most recently used first.
struct PlanReplay {
    program: Option<ReplayProgram>,
    templates: Vec<Template>,
}

/// Scratch of the replay path, reused from draw to draw.
#[derive(Default)]
struct ReplayState {
    run: ReplayScratch,
    signature: Vec<u32>,
    /// The replayed result (its vectors keep their capacity).
    scratch: ReplayResult,
    /// Snapshots of earlier patched replays (see [`RECYCLED_SNAPSHOTS`]).
    recycled: std::collections::VecDeque<Arc<ResourceSnapshot>>,
}

/// Recent materialisations keyed by plan, user data, shader base and
/// compute state, with up to four memory-content variants each. Two
/// generations approximate LRU.
pub struct ResourceMaterializationCache {
    generation_capacity: usize,
    young: HashMap<u64, Vec<Entry>>,
    old: HashMap<u64, Vec<Entry>>,
    statistics: CacheStatistics,
    /// Replay state per plan id.
    plans: HashMap<u64, PlanReplay, crate::fxhash::FxBuildHasher>,
    replay: ReplayState,
}

impl Default for ResourceMaterializationCache {
    fn default() -> Self {
        Self::new(16384)
    }
}

impl ResourceMaterializationCache {
    pub fn new(generation_capacity: usize) -> Self {
        ResourceMaterializationCache {
            generation_capacity: generation_capacity.max(1),
            young: HashMap::new(),
            old: HashMap::new(),
            statistics: CacheStatistics::default(),
            plans: HashMap::default(),
            replay: ReplayState::default(),
        }
    }
}

impl ReplayState {
    /// The replay path (see the module comment). None hands the draw to
    /// the recording path.
    fn replayed(
        &mut self,
        statistics: &mut CacheStatistics,
        plan: &ShaderResourcePlan,
        program: &ReplayProgram,
        templates: &mut Vec<Template>,
        inputs: &ResourceRuntimeInputs<'_>,
        resident: ResidentReader<'_>,
    ) -> Option<MaterializeResult> {
        let user_data = inputs.user_data;
        let has_clean = inputs.read_clean_memory.is_some();
        // The result is made in the reused scratch (its vectors keep their
        // capacity from draw to draw).
        let mut result = std::mem::take(&mut self.scratch);
        let mut ran = program.run_into(user_data, inputs.shader_base, &ReplayReader::Resident(resident, has_clean), inputs.gpu_written, &mut self.run, &mut result);
        if ran == Err(ReplayAbort::Unreadable) {
            // GPU-written or unmapped words: read them through the
            // synchronising reader.
            let Some(read) = inputs.read_memory else {
                self.scratch = result;
                return None;
            };
            ran = program.run_into(user_data, inputs.shader_base, &ReplayReader::Words(read, inputs.read_clean_memory), inputs.gpu_written, &mut self.run, &mut result);
        }
        if ran.is_err() {
            self.scratch = result;
            return None;
        }
        if !specialization_signature(plan, program.descriptor_spans(), &result.words[..program.descriptor_words], &mut self.signature) {
            self.scratch = result;
            return None;
        }
        if let Some(index) = templates.iter().position(|t| t.signature == self.signature) {
            let t = &mut templates[index];
            if t.last == result && t.last_user_data == user_data {
                statistics.replay_same += 1;
                self.scratch = result;
            } else {
                // A recycled snapshot is held by no one else: it is
                // written over in place.
                let patched = match take_recycled(&mut self.recycled) {
                    Some(mut s) => {
                        statistics.replay_recycled += 1;
                        let ok = Arc::get_mut(&mut s).is_some_and(|out| patch_replayed_into(plan, program.descriptor_spans(), &t.layout, &result, program.descriptor_words, user_data, out));
                        ok.then_some(s)
                    }
                    None => patch_replayed(plan, program.descriptor_spans(), &t.layout, &result, program.descriptor_words, user_data).map(Arc::new),
                };
                let Some(snapshot) = patched else {
                    self.scratch = result;
                    return None;
                };
                let old = std::mem::replace(&mut t.last_snapshot, snapshot);
                recycle(&mut self.recycled, old);
                // The previous result's vectors become the next scratch.
                self.scratch = std::mem::replace(&mut t.last, result);
                t.last_user_data.clear();
                t.last_user_data.extend_from_slice(user_data);
                statistics.replay_patched += 1;
            }
            let output = (t.last_snapshot.clone(), t.specialization.clone());
            if index != 0 {
                let t = templates.remove(index);
                templates.insert(0, t);
            }
            statistics.hits += 1;
            return Some(Ok(output));
        }
        let (snapshot, specialization, layout) = match materialize_replayed(plan, &result, program.descriptor_words, user_data) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let (snapshot, specialization) = (Arc::new(snapshot), Arc::new(specialization));
        templates.insert(
            0,
            Template {
                signature: self.signature.clone(),
                layout,
                specialization: specialization.clone(),
                last: result,
                last_user_data: user_data.to_vec(),
                last_snapshot: snapshot.clone(),
            },
        );
        templates.truncate(max_templates());
        statistics.replay_templates += 1;
        statistics.misses += 1;
        Some(Ok((snapshot, specialization)))
    }
}

impl ResourceMaterializationCache {
    pub fn statistics(&self) -> CacheStatistics {
        self.statistics
    }

    fn key_of(plan: &ShaderResourcePlan, inputs: &ResourceRuntimeInputs<'_>) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        plan.id.hash(&mut h);
        inputs.shader_base.hash(&mut h);
        inputs.compute_state.hash(&mut h);
        inputs.user_data.hash(&mut h);
        h.finish()
    }

    fn store(&mut self, key: u64, variants: Vec<Entry>) {
        if self.young.len() >= self.generation_capacity && !self.young.contains_key(&key) {
            self.old = std::mem::take(&mut self.young);
        }
        self.young.insert(key, variants);
    }

    fn validate(entry: &Entry, resident: ResidentReader<'_>) -> Result<bool, ()> {
        let mut scratch = Vec::new();
        for i in 0..entry.range_addresses.len() {
            scratch.resize(entry.range_lengths[i], 0);
            if !resident(entry.range_addresses[i], &mut scratch, entry.range_clean[i]) {
                return Err(());
            }
            let offset = entry.range_offsets[i];
            if scratch[..] != entry.bytes[offset..offset + entry.range_lengths[i]] {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Materialise `plan` for `inputs`, reusing a cached result while every
    /// guest word it read is unchanged per `resident`.
    pub fn materialize(
        &mut self,
        plan: &ShaderResourcePlan,
        inputs: &ResourceRuntimeInputs<'_>,
        resident: ResidentReader<'_>,
    ) -> MaterializeResult {
        let replay = self.plans.entry(plan.id).or_insert_with(|| PlanReplay { program: ReplayProgram::compile(plan), templates: Vec::new() });
        if let Some(program) = &replay.program {
            if let Some(result) = self.replay.replayed(&mut self.statistics, plan, program, &mut replay.templates, inputs, resident) {
                return result;
            }
            self.statistics.replay_fallbacks += 1;
        }
        let key = Self::key_of(plan, inputs);
        let plan_id = plan.id;
        let mut variants = match self.young.remove(&key) {
            Some(v) => v,
            None => self.old.remove(&key).unwrap_or_default(),
        };
        variants.retain(|v| v.matches(plan_id, inputs));
        let found = !variants.is_empty();
        if found {
            let mut unreadable = false;
            for index in 0..variants.len() {
                match Self::validate(&variants[index], resident) {
                    Ok(true) => {
                        let hit = variants.remove(index);
                        let result = (hit.snapshot.clone(), hit.specialization.clone());
                        variants.insert(0, hit);
                        self.store(key, variants);
                        self.statistics.hits += 1;
                        return Ok(result);
                    }
                    Ok(false) => {}
                    Err(()) => unreadable = true,
                }
            }
            if let Some(refreshed) = Self::refresh_table(&variants[0], plan, inputs, resident) {
                let result = (refreshed.snapshot.clone(), refreshed.specialization.clone());
                variants[0] = refreshed;
                self.store(key, variants);
                self.statistics.table_refreshes += 1;
                return Ok(result);
            }
            self.statistics.stale += 1;
            if unreadable {
                self.statistics.stale_unreadable += 1;
            }
        }
        self.statistics.misses += 1;
        let recorder = Recorder::default();
        let read = |a: u64| inputs.read_memory.and_then(|r| recorder.record(r, a, false));
        let read_clean = |a: u64| inputs.read_clean_memory.and_then(|r| recorder.record(r, a, true));
        let phase = |in_table: bool| recorder.in_table.set(in_table);
        let resident_record = |address: u64, destination: &mut [u8], clean: bool| {
            let Some(inner) = inputs.read_resident_memory else { return false };
            if !inner(address, destination, clean) {
                return false;
            }
            for (i, chunk) in destination.chunks_exact(4).enumerate() {
                let word = u32::from_le_bytes(chunk.try_into().unwrap());
                recorder.reads.borrow_mut().push((address + i as u64 * 4, word, clean, recorder.in_table.get()));
            }
            true
        };
        let recording = ResourceRuntimeInputs {
            user_data: inputs.user_data,
            shader_base: inputs.shader_base,
            read_memory: inputs.read_memory.map(|_| &read as WordReader<'_>),
            read_clean_memory: inputs.read_clean_memory.map(|_| &read_clean as WordReader<'_>),
            compute_state: inputs.compute_state,
            table_phase: Some(&phase),
            read_resident_memory: inputs.read_resident_memory.map(|_| &resident_record as ResidentReader<'_>),
            reads_clean: inputs.reads_clean,
            gpu_written: inputs.gpu_written,
        };
        let (snapshot, specialization) = match materialize(plan, &recording) {
            Ok(r) => r,
            Err(e) => {
                if found {
                    self.store(key, variants);
                }
                return Err(e);
            }
        };
        // Deferred slots were not read: such a result says nothing about
        // memory a later draw could validate.
        if recorder.failed.get() || !snapshot.deferred_table_reads.is_empty() {
            self.statistics.uncacheable += 1;
            if found {
                self.store(key, variants);
            }
            return Ok((Arc::new(snapshot), Arc::new(specialization)));
        }
        let built = recorder.build(plan, inputs, snapshot, specialization);
        let result = (built.snapshot.clone(), built.specialization.clone());
        variants.insert(0, built);
        variants.truncate(MAX_VARIANTS);
        self.store(key, variants);
        Ok(result)
    }

    /// When every changed word was read only while evaluating the flattened
    /// table, evaluate only the table again. Refused when the table was
    /// extended by specialization, when it now reads a word the entry did not
    /// validate, or when a word moved between the two reads.
    fn refresh_table(
        cached: &Entry,
        plan: &ShaderResourcePlan,
        inputs: &ResourceRuntimeInputs<'_>,
        resident: ResidentReader<'_>,
    ) -> Option<Entry> {
        if !cached.table_refreshable {
            return None;
        }
        let mut current = vec![0u8; cached.bytes.len()];
        for i in 0..cached.range_addresses.len() {
            let o = cached.range_offsets[i];
            if !resident(cached.range_addresses[i], &mut current[o..o + cached.range_lengths[i]], cached.range_clean[i]) {
                return None;
            }
        }
        let mut changed = false;
        for offset in (0..current.len()).step_by(4) {
            if current[offset..offset + 4] == cached.bytes[offset..offset + 4] {
                continue;
            }
            if !cached.word_table_only[offset / 4] {
                return None;
            }
            changed = true;
        }
        if !changed {
            return None;
        }
        let recorder = Recorder::default();
        let read = |a: u64| inputs.read_memory.and_then(|r| recorder.record(r, a, false));
        let read_clean = |a: u64| inputs.read_clean_memory.and_then(|r| recorder.record(r, a, true));
        let recording = ResourceRuntimeInputs {
            read_memory: inputs.read_memory.map(|_| &read as WordReader<'_>),
            read_clean_memory: inputs.read_clean_memory.map(|_| &read_clean as WordReader<'_>),
            table_phase: None,
            read_resident_memory: None,
            ..*inputs
        };
        let mut table = try_evaluate_table(plan, &recording)?;
        if recorder.failed.get() || table.len() != cached.snapshot.flattened_resource_table.len() {
            return None;
        }
        for &(address, word, _, _) in recorder.reads.borrow().iter() {
            let offset = cached.find_word(address)?;
            if u32::from_le_bytes(current[offset..offset + 4].try_into().unwrap()) != word {
                return None;
            }
        }
        // The written device-address slots come from reads validated above.
        for &slot in plan.written_range_slot_by_handle.values() {
            let s = slot as usize;
            table[s..s + WRITTEN_RANGE_DWORD_COUNT].copy_from_slice(&cached.snapshot.flattened_resource_table[s..s + WRITTEN_RANGE_DWORD_COUNT]);
        }
        let snapshot = ResourceSnapshot { flattened_resource_table: table, ..(*cached.snapshot).clone() };
        Some(Entry {
            plan: cached.plan,
            user_data: cached.user_data.clone(),
            shader_base: cached.shader_base,
            compute_state: cached.compute_state,
            range_addresses: cached.range_addresses.clone(),
            range_offsets: cached.range_offsets.clone(),
            range_lengths: cached.range_lengths.clone(),
            range_clean: cached.range_clean.clone(),
            word_table_only: cached.word_table_only.clone(),
            table_refreshable: true,
            bytes: current,
            snapshot: Arc::new(snapshot),
            specialization: cached.specialization.clone(),
        })
    }
}
