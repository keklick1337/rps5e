// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Per-draw materialisation: resolve a plan against one draw's inputs into
//! concrete descriptor words ([`ResourceSnapshot`]) and the module-affecting
//! [`ResourceSpecialization`].

use super::candidates::BufferCandidateTablePlan;
use super::device_ranges::{self, DeviceAddressRange};
use super::evaluator::{evaluate_sources, ResourceRuntimeInputs, RuntimeValueEvaluator, ADDRESS_MASK};
use super::format::{image_format, ImageNumericClass};
use super::info::*;
use super::memory::{ImageDimension, ImageResourceClass, MemoryResourceKind};
use super::plan::{ShaderResourcePlan, WRITTEN_RANGE_DWORD_COUNT};
use super::selector::IndirectSelectorDiagnostic;
use super::ResourcePlanError;
use std::collections::{HashMap, HashSet};
use std::fmt;

const MAX_INDIRECT_IMAGE_PROBES: u64 = 65536;

/// The words of one descriptor (V#, T#, S#), inline up to a T#.
pub type DescriptorWords = smallvec::SmallVec<[u32; 8]>;

/// The descriptor words captured for one program at draw time.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResourceSnapshot {
    /// One 4-dword V# per buffer binding (plan buffers, then candidates).
    pub buffers: Vec<DescriptorWords>,
    /// One 8-dword T# per image binding (plan images, then indirect
    /// candidates).
    pub images: Vec<DescriptorWords>,
    /// One 4-dword S# per sampler binding (plan samplers, then point and
    /// compare duplicates).
    pub samplers: Vec<DescriptorWords>,
    /// Table reads, written-range slots, then indirect key mappings.
    pub flattened_resource_table: Vec<u32>,
    pub user_data: Vec<u32>,
    pub device_address_ranges: Vec<DeviceAddressRange>,
    /// Flattened-table slots left zero because their word is still being
    /// produced by earlier GPU work: (slot, guest address of the dword).
    /// The host copies each dword into its slot on the GPU, ordered after
    /// that work (see `ResourceRuntimeInputs::gpu_written`).
    pub deferred_table_reads: Vec<(u32, u64)>,
    /// Images selected through the host table map.
    pub bindless_tables: Vec<BindlessTable>,
}

/// An image selected through the host table map (a bindless table too
/// large to bind per draw): its T#s are the 32-byte records of the heap V#
/// from `entry_offset` on, and the host fills the
/// [`bindless::TABLE_WORDS`](super::binding::bindless::TABLE_WORDS)
/// flattened-table words at `mapping_offset` with the map of this draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindlessTable {
    pub image: u32,
    pub heap: [u32; 4],
    pub entry_offset: u32,
    /// Records of the heap from `entry_offset` on (keys past them read a
    /// null T#).
    pub entries: u32,
    pub mapping_offset: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferSpecialization {
    pub packed_stride: u32,
    pub descriptor_format: u32,
    pub descriptor_swizzle: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferCandidateTableSpecialization {
    pub first_candidate: u32,
    pub candidate_count: u32,
    pub mapping_offset: u32,
    pub search_iterations: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageSpecialization {
    pub numeric_class: ImageNumericClass,
    pub dimension: ImageDimension,
    pub mip_count: u32,
    pub conversion_format: u32,
    pub shader_swizzle: u32,
    pub indirect_root: u32,
    pub indirect_mapping_offset: u32,
    pub indirect_search_iterations: u32,
    pub cube: bool,
    pub emulated_compare: u32,
    /// Selected through the host table map ([`BindlessTable`]).
    pub bindless: bool,
    /// The T# format of an image accessed as raw texels (PCK, BY forms).
    pub packed_format: u32,
    /// [`super::info::ImageResource::store_quantize`].
    pub store_quantize: u32,
}

/// The module-affecting resource state of one draw. Addresses and
/// descriptor payloads stay in the snapshot, so they never create a
/// permutation.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ResourceSpecialization {
    /// The plan's buffer count; later buffers are materialised candidates.
    pub base_buffer_count: usize,
    pub buffers: Vec<BufferSpecialization>,
    pub images: Vec<ImageSpecialization>,
    pub buffer_candidate_tables: Vec<BufferCandidateTableSpecialization>,
}

impl ResourceSpecialization {
    /// The specialization of a plan before any draw.
    pub fn default_for(info: &ShaderResourceInfo) -> Self {
        ResourceSpecialization {
            base_buffer_count: info.buffers.len(),
            buffers: info
                .buffers
                .iter()
                .map(|_| BufferSpecialization { packed_stride: 0, descriptor_format: INVALID_FORMAT, descriptor_swizzle: IDENTITY_DESTINATION_SELECT })
                .collect(),
            images: info
                .images
                .iter()
                .map(|i| ImageSpecialization {
                    numeric_class: if i.numeric_class == ImageNumericClass::Unsupported {
                        if i.atomic {
                            ImageNumericClass::Uint
                        } else {
                            ImageNumericClass::Float
                        }
                    } else {
                        i.numeric_class
                    },
                    dimension: if i.dimension == ImageDimension::Unknown { ImageDimension::Dim2D } else { i.dimension },
                    mip_count: i.mip_count,
                    conversion_format: i.conversion_format,
                    shader_swizzle: i.shader_swizzle,
                    indirect_root: i.indirect_root,
                    indirect_mapping_offset: i.indirect_mapping_offset,
                    indirect_search_iterations: i.indirect_search_iterations,
                    cube: i.cube,
                    emulated_compare: 0,
                    bindless: i.bindless,
                    packed_format: i.packed_format,
                    store_quantize: i.store_quantize,
                })
                .collect(),
            buffer_candidate_tables: Vec::new(),
        }
    }
}

/// A plan's tables with one draw's specialization applied.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpecializedResourceInfo {
    pub info: ShaderResourceInfo,
    /// Memory index → the sampler (point or compare duplicate) it uses when
    /// it differs from the plan's.
    pub sampler_by_memory_index: HashMap<usize, u32>,
}

/// Why a materialisation failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MaterializationError {
    /// Some descriptor or table word could not be read or evaluated.
    Other(String),
    /// Indirect candidates disagree on class, dimension or format.
    IncompatibleImageCandidates(Box<IndirectImageFailure>),
    ImageCapacityExceeded(String),
}

impl fmt::Display for MaterializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MaterializationError::Other(m) => write!(f, "shader resource specialization failed: {m}"),
            MaterializationError::IncompatibleImageCandidates(failure) => write!(
                f,
                "indirect image table at pc 0x{:08x} has incompatible candidates: exemplar={} candidate={}",
                failure.instruction_address, failure.exemplar_resource, failure.incompatible_resource
            ),
            MaterializationError::ImageCapacityExceeded(m) => write!(f, "image capacity exceeded: {m}"),
        }
    }
}

impl std::error::Error for MaterializationError {}

fn other(message: impl Into<String>) -> MaterializationError {
    MaterializationError::Other(message.into())
}

/// The rejected indirect table, captured without re-reading guest memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndirectImageFailure {
    pub instruction_address: u32,
    pub root_resource: u32,
    pub exemplar_resource: u32,
    pub incompatible_resource: u32,
    pub selector_stride: u32,
    pub selector_offset: u32,
    pub material_descriptor: Vec<u32>,
    pub heap_descriptor: Vec<u32>,
    pub keys: Vec<u32>,
    pub candidate_indices: Vec<u32>,
    pub table_descriptors: Vec<Vec<u32>>,
    pub image_descriptors: Vec<Vec<u32>>,
    pub image_specializations: Vec<ImageSpecialization>,
    pub user_data: Vec<u32>,
    pub selector_diagnostic: Option<IndirectSelectorDiagnostic>,
}

#[derive(Default)]
struct IndirectImageTable {
    resource: u32,
    keys: Vec<u32>,
    candidates: Vec<u32>,
    descriptors: Vec<Vec<u32>>,
    material_descriptor: Vec<u32>,
    heap_descriptor: Vec<u32>,
    selector_stride: u32,
    selector_offset: u32,
    selector_diagnostic: Option<IndirectSelectorDiagnostic>,
    /// Selected through the host table map: (heap V#, entry offset,
    /// entries). The descriptors are then one null T#.
    bindless: Option<([u32; 4], u32, u32)>,
}

#[derive(Default)]
struct BufferCandidateTable {
    keys: Vec<u32>,
    candidates: Vec<u32>,
    descriptors: Vec<Vec<u32>>,
}

#[derive(Default)]
struct MaterializedSnapshot {
    buffers: Vec<DescriptorWords>,
    images: Vec<DescriptorWords>,
    samplers: Vec<DescriptorWords>,
    flattened_table: Vec<u32>,
    user_data: Vec<u32>,
    indirect_images: Vec<IndirectImageTable>,
    buffer_candidate_tables: Vec<BufferCandidateTable>,
    /// Per buffer / image binding: the descriptor was replaced by zeros
    /// (reserved bits, invalid or unsupported T#).
    buffer_zeroed: Vec<bool>,
    image_zeroed: Vec<bool>,
    /// Per sampler binding: the plan sampler whose words it carries
    /// (`NO_INDEX`: none).
    sampler_sources: Vec<u32>,
    deferred_table_reads: Vec<(u32, u64)>,
    /// Images selected through the host table map (their mapping words
    /// are placed by `build_specialization`).
    bindless_tables: Vec<BindlessTable>,
}

/// Resolve `plan` against one draw. Nothing is returned unless the whole
/// materialisation succeeds.
pub fn materialize(
    plan: &ShaderResourcePlan,
    inputs: &ResourceRuntimeInputs<'_>,
) -> Result<(ResourceSnapshot, ResourceSpecialization), MaterializationError> {
    let materialized = materialize_snapshot(plan, inputs, true)?;
    let ranges = device_ranges::evaluate(plan, inputs);
    finish(plan, materialized, ranges).map(|(snapshot, specialization, _)| (snapshot, specialization))
}

/// How a materialisation turned evaluated descriptor words into snapshot
/// words, for draws whose specialization-affecting fields
/// ([`specialization_signature`]) are the same.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct SnapshotLayout {
    pub buffer_zeroed: Vec<bool>,
    pub image_zeroed: Vec<bool>,
    pub sampler_sources: Vec<u32>,
    pub table_len: usize,
}

/// The written ranges, the specialization and the snapshot of evaluated
/// descriptor words.
fn finish(
    plan: &ShaderResourcePlan,
    mut materialized: MaterializedSnapshot,
    ranges: Vec<DeviceAddressRange>,
) -> Result<(ResourceSnapshot, ResourceSpecialization, SnapshotLayout), MaterializationError> {
    // The written ranges follow the table reads so every store checks its
    // own.
    for range in &ranges {
        let Some(&slot) = plan.written_range_slot_by_handle.get(&range.handle) else { continue };
        let offset = slot as usize;
        materialized.flattened_table[offset] = range.base as u32;
        materialized.flattened_table[offset + 1] = (range.base >> 32) as u32;
        materialized.flattened_table[offset + 2] = range.size.min(u32::MAX as u64) as u32;
    }
    let specialization = build_specialization(plan, &mut materialized)?;
    let mut buffer_zeroed = materialized.buffer_zeroed;
    buffer_zeroed.resize(materialized.buffers.len(), false);
    let mut image_zeroed = materialized.image_zeroed;
    image_zeroed.resize(materialized.images.len(), false);
    let layout = SnapshotLayout { buffer_zeroed, image_zeroed, sampler_sources: materialized.sampler_sources, table_len: materialized.flattened_table.len() };
    Ok((
        ResourceSnapshot {
            buffers: materialized.buffers,
            images: materialized.images,
            samplers: materialized.samplers,
            flattened_resource_table: materialized.flattened_table,
            user_data: materialized.user_data,
            device_address_ranges: ranges,
            deferred_table_reads: materialized.deferred_table_reads,
            bindless_tables: materialized.bindless_tables,
        },
        specialization,
        layout,
    ))
}

/// The descriptor words of one replayed draw split per binding: buffers,
/// images, samplers (as the plan's materialisation sources order them).
fn split_descriptors(plan: &ShaderResourcePlan, words: &[u32]) -> Option<(Vec<DescriptorWords>, Vec<DescriptorWords>, Vec<DescriptorWords>)> {
    let mut cursor = 0usize;
    let mut take = |source: u32| -> Option<DescriptorWords> {
        let n = plan.descriptor_sources.get(source as usize)?.dwords.len();
        let v = DescriptorWords::from_slice(words.get(cursor..cursor + n)?);
        cursor += n;
        Some(v)
    };
    let mut sources = plan.materialization_sources.iter().copied();
    let buffers = (0..plan.info.buffers.len()).map(|_| take(sources.next()?)).collect::<Option<Vec<_>>>()?;
    let images = (0..plan.info.images.len()).map(|_| take(sources.next()?)).collect::<Option<Vec<_>>>()?;
    let samplers = (0..plan.info.samplers.len()).map(|_| take(sources.next()?)).collect::<Option<Vec<_>>>()?;
    Some((buffers, images, samplers))
}

/// Full materialisation of a replayed draw (a plan
/// [`ReplayProgram::eligible`](super::replay::ReplayProgram::eligible)):
/// the same steps as [`materialize`] after its evaluation.
pub(super) fn materialize_replayed(
    plan: &ShaderResourcePlan,
    result: &super::replay::ReplayResult,
    descriptor_words: usize,
    user_data: &[u32],
) -> Result<(ResourceSnapshot, ResourceSpecialization, SnapshotLayout), MaterializationError> {
    let (buffers, mut images, samplers) = split_descriptors(plan, &result.words[..descriptor_words]).ok_or_else(|| other("replayed words do not match the plan"))?;
    let mut image_zeroed = vec![false; images.len()];
    for (index, image) in plan.info.images.iter().enumerate() {
        if !valid_image_descriptor(&images[index], image.r128) {
            images[index].iter_mut().for_each(|w| *w = 0);
            image_zeroed[index] = true;
        }
    }
    let mut flattened_table = result.words[descriptor_words..].to_vec();
    flattened_table.resize(flattened_table.len() + plan.written_range_count() * WRITTEN_RANGE_DWORD_COUNT, 0);
    let materialized = MaterializedSnapshot {
        buffers,
        images,
        samplers,
        flattened_table,
        user_data: user_data.to_vec(),
        image_zeroed,
        deferred_table_reads: result.deferred.clone(),
        ..Default::default()
    };
    let ranges = plan.device_address_ranges.iter().zip(&result.range_bases).map(|(r, &b)| device_ranges::from_base(r, b)).collect();
    finish(plan, materialized, ranges)
}

/// The snapshot of a replayed draw whose [`specialization_signature`]
/// equals the one `layout` was recorded with: its words with the same
/// zeroing, sampler copies and written-range slots. `spans`: each
/// descriptor's words in `result` (start, end), in materialisation order
/// ([`ReplayProgram::descriptor_spans`](super::replay::ReplayProgram::descriptor_spans)).
pub(super) fn patch_replayed(
    plan: &ShaderResourcePlan,
    spans: &[(u32, u32)],
    layout: &SnapshotLayout,
    result: &super::replay::ReplayResult,
    descriptor_words: usize,
    user_data: &[u32],
) -> Option<ResourceSnapshot> {
    let mut out = ResourceSnapshot::default();
    patch_replayed_into(plan, spans, layout, result, descriptor_words, user_data, &mut out).then_some(out)
}

/// `dst` made `n` descriptors long, each written by `f` over its earlier
/// contents (inline words: no allocation once `dst` has grown).
fn assign_descriptors(dst: &mut Vec<DescriptorWords>, n: usize, mut f: impl FnMut(usize, &mut DescriptorWords)) {
    dst.truncate(n);
    dst.resize_with(n, DescriptorWords::new);
    for (i, d) in dst.iter_mut().enumerate() {
        d.clear();
        f(i, d);
    }
}

/// [`patch_replayed`] into `out`, whose vectors are reused (no allocation
/// once they have grown). False when the words do not match the plan
/// (`out` is then unspecified).
pub(super) fn patch_replayed_into(
    plan: &ShaderResourcePlan,
    spans: &[(u32, u32)],
    layout: &SnapshotLayout,
    result: &super::replay::ReplayResult,
    descriptor_words: usize,
    user_data: &[u32],
    out: &mut ResourceSnapshot,
) -> bool {
    let words = &result.words[..descriptor_words];
    let (nb, ni) = (plan.info.buffers.len(), plan.info.images.len());
    if spans.len() != nb + ni + plan.info.samplers.len() || spans.iter().any(|&(s, e)| s > e || e as usize > words.len()) {
        return false;
    }
    let copy = |d: &mut DescriptorWords, (s, e): (u32, u32), zero: bool| {
        let w = &words[s as usize..e as usize];
        if zero {
            d.resize(w.len(), 0);
        } else {
            d.extend_from_slice(w);
        }
    };
    assign_descriptors(&mut out.buffers, nb, |i, d| copy(d, spans[i], layout.buffer_zeroed.get(i).copied().unwrap_or(false)));
    assign_descriptors(&mut out.images, ni, |i, d| copy(d, spans[nb + i], layout.image_zeroed.get(i).copied().unwrap_or(false)));
    let samplers = &spans[nb + ni..];
    if layout.sampler_sources.iter().any(|&s| s != NO_INDEX && s as usize >= samplers.len()) {
        return false;
    }
    assign_descriptors(&mut out.samplers, layout.sampler_sources.len(), |i, d| {
        let s = layout.sampler_sources[i];
        if s != NO_INDEX {
            copy(d, samplers[s as usize], false);
        }
    });
    out.device_address_ranges.clear();
    out.device_address_ranges.extend(plan.device_address_ranges.iter().zip(&result.range_bases).map(|(r, &b)| device_ranges::from_base(r, b)));
    let table = &mut out.flattened_resource_table;
    table.clear();
    table.extend_from_slice(&result.words[descriptor_words..]);
    table.resize(layout.table_len, 0);
    for range in &out.device_address_ranges {
        let Some(&slot) = plan.written_range_slot_by_handle.get(&range.handle) else { continue };
        let offset = slot as usize;
        table[offset] = range.base as u32;
        table[offset + 1] = (range.base >> 32) as u32;
        table[offset + 2] = range.size.min(u32::MAX as u64) as u32;
    }
    out.user_data.clear();
    out.user_data.extend_from_slice(user_data);
    out.deferred_table_reads.clear();
    out.deferred_table_reads.extend_from_slice(&result.deferred);
    out.bindless_tables.clear();
    true
}

/// The descriptor fields of a replayed draw that decide its
/// specialization and snapshot layout: per buffer the stride / swizzle
/// and format word, per image whether it is null or valid, its format
/// and its swizzle / level / type word, and the sampler words when a
/// depth compare may be emulated. Addresses, sizes and record counts are
/// left out: draws that differ only there share one specialization.
/// `spans`: as for [`patch_replayed`].
pub(super) fn specialization_signature(plan: &ShaderResourcePlan, spans: &[(u32, u32)], words: &[u32], out: &mut Vec<u32>) -> bool {
    out.clear();
    let (nb, ni) = (plan.info.buffers.len(), plan.info.images.len());
    if spans.len() != nb + ni + plan.info.samplers.len() {
        return false;
    }
    let get = |(s, e): (u32, u32)| words.get(s as usize..e as usize);
    for &span in &spans[..nb] {
        let Some(w) = get(span) else { return false };
        if w.len() != 4 {
            return false;
        }
        out.push(w[1] >> 16);
        out.push(w[3]);
    }
    for (image, &span) in plan.info.images.iter().zip(&spans[nb..nb + ni]) {
        let Some(d) = get(span) else { return false };
        if d.len() < 8 {
            out.push(u32::MAX);
            out.extend_from_slice(d);
            continue;
        }
        let flags = null_image_descriptor(d) as u32 | (valid_image_descriptor(d, image.r128) as u32) << 1;
        out.push(flags);
        out.push(image_format::format_of(d));
        out.push(d[3]);
    }
    let compare = plan.info.images.iter().any(|i| i.depth_compare);
    for &span in &spans[nb + ni..] {
        let Some(s) = get(span) else { return false };
        if compare {
            out.extend_from_slice(s);
        }
    }
    true
}

/// Only the flattened table, laid out as a full materialisation lays it out
/// before specialization (written-range slots left zero).
pub fn try_evaluate_table(plan: &ShaderResourcePlan, inputs: &ResourceRuntimeInputs<'_>) -> Option<Vec<u32>> {
    evaluate_sources(plan, &[], inputs, &plan.clean_flat_slots, true, plan.written_range_count() * WRITTEN_RANGE_DWORD_COUNT).map(|e| e.table)
}

fn materialize_snapshot(
    plan: &ShaderResourcePlan,
    inputs: &ResourceRuntimeInputs<'_>,
    capture_selector_diagnostic: bool,
) -> Result<MaterializedSnapshot, MaterializationError> {
    if plan.requires_specialization_memory && inputs.read_clean_memory.is_none() {
        return Err(other("the plan needs a clean memory reader"));
    }
    let evaluated = evaluate_sources(
        plan,
        &plan.materialization_sources,
        inputs,
        &plan.clean_flat_slots,
        true,
        plan.written_range_count() * WRITTEN_RANGE_DWORD_COUNT,
    )
    .ok_or_else(|| other(diagnose_snapshot_failure(plan, inputs)))?;
    let active = &evaluated.active_sources;
    let mut values = evaluated.descriptors.into_iter();
    let mut snapshot = MaterializedSnapshot::default();
    for _ in 0..plan.info.buffers.len() {
        snapshot.buffers.push(values.next().unwrap().into());
    }
    snapshot.flattened_table = evaluated.table;
    snapshot.deferred_table_reads = evaluated.deferred;
    let clean_inputs = inputs.with_reader(inputs.read_clean_memory);
    for (image_index, image) in plan.info.images.iter().enumerate() {
        let source = &plan.descriptor_sources[image.source as usize];
        let Some(indirect) = &source.indirect_image else {
            let mut descriptor = values.next().unwrap();
            if !valid_image_descriptor(&descriptor, image.r128) {
                descriptor = vec![0; descriptor.len()];
                snapshot.image_zeroed.resize(image_index + 1, false);
                snapshot.image_zeroed[image_index] = true;
            }
            snapshot.images.push(descriptor.into());
            continue;
        };
        if !active.is_empty() && !active[image.source as usize] {
            snapshot.images.push(DescriptorWords::from_elem(0, 8));
            continue;
        }
        let read_failed = || other(format!("indirect image {image_index} tables cannot be read"));
        let table = if let Some(candidates) = &indirect.direct_candidates {
            let sources: Vec<u32> = candidates.iter().map(|c| c.source).collect();
            let descriptors = evaluate_sources(plan, &sources, &clean_inputs, &[], false, 0).ok_or_else(read_failed)?.descriptors;
            let mut table = IndirectImageTable { resource: image_index as u32, ..Default::default() };
            for (i, mut descriptor) in descriptors.into_iter().enumerate() {
                if !usable_image_candidate(&descriptor, image.r128) {
                    descriptor = vec![0; 8];
                }
                let existing = match table.descriptors.iter().position(|d| *d == descriptor) {
                    Some(e) => e,
                    None => {
                        table.descriptors.push(descriptor);
                        table.descriptors.len() - 1
                    }
                };
                table.keys.push(candidates[i].offset);
                table.candidates.push(existing as u32);
            }
            table
        } else if let Some(table_selector) = indirect.table {
            let sources = [indirect.heap_source, indirect.material_source];
            let tables = evaluate_sources(plan, &sources, &clean_inputs, &[], false, 0).ok_or_else(read_failed)?.descriptors;
            let key = draw_uniform_table_key(plan, image_index, &clean_inputs);
            materialize_table_image(indirect, table_selector, &tables[0], &tables[1], image, key, inputs)?
        } else if let Some(wave) = indirect.wave_indexed {
            let heap = evaluate_sources(plan, &[indirect.heap_source], &clean_inputs, &[], false, 0).ok_or_else(read_failed)?.descriptors;
            materialize_wave_indexed_image(indirect, wave, &heap[0], image.r128, inputs)?
        } else if indirect.dense {
            let heap = evaluate_sources(plan, &[indirect.heap_source], &clean_inputs, &[], false, 0).ok_or_else(read_failed)?.descriptors;
            materialize_dense_indirect_image(indirect, &heap[0], image.r128, inputs)?
        } else {
            let tables = evaluate_sources(plan, &[indirect.material_source, indirect.heap_source], &clean_inputs, &[], false, 0)
                .ok_or_else(read_failed)?
                .descriptors;
            materialize_indirect_image(plan, indirect, &tables[0], &tables[1], image, inputs, capture_selector_diagnostic)?
        };
        snapshot.images.push(table.descriptors[table.candidates[0] as usize].as_slice().into());
        if let Some((heap, entry_offset, entries)) = table.bindless {
            snapshot.bindless_tables.push(BindlessTable { image: image_index as u32, heap, entry_offset, entries, mapping_offset: 0 });
            continue;
        }
        if table.descriptors.len() > 1 {
            snapshot.indirect_images.push(IndirectImageTable { resource: image_index as u32, ..table });
        }
    }
    for _ in 0..plan.info.samplers.len() {
        snapshot.samplers.push(values.next().unwrap().into());
    }
    // Bounded runtime V# tables: the whole table is read and validated
    // before it is published.
    for candidate_plan in &plan.buffer_candidate_tables {
        if candidate_plan.source_srt_resource == NO_INDEX {
            return Err(other("buffer candidate table has no SRT source"));
        }
        let srt = evaluate_sources(plan, &[candidate_plan.source_srt_resource], inputs, &[], false, 0)
            .ok_or_else(|| other("buffer candidate SRT cannot be evaluated"))?
            .descriptors;
        snapshot.buffer_candidate_tables.push(materialize_buffer_candidate_table(plan, candidate_plan, &srt[0], inputs)?);
    }
    snapshot.user_data = inputs.user_data.to_vec();
    Ok(snapshot)
}

/// Read every candidate the loop guard can select, validate it, and map
/// its probe key (base-address low dword) to a candidate. Distinct keys are
/// required; identical descriptors share a candidate.
fn materialize_buffer_candidate_table(
    plan: &ShaderResourcePlan,
    table: &BufferCandidateTablePlan,
    srt: &[u32],
    inputs: &ResourceRuntimeInputs<'_>,
) -> Result<BufferCandidateTable, MaterializationError> {
    if srt.len() != 4 {
        return Err(other("buffer candidate SRT has invalid width"));
    }
    let count = if table.is_statically_bounded() {
        table.count
    } else {
        let limit = table.limit.ok_or_else(|| other("buffer candidate table has no limit"))?;
        let evaluator = RuntimeValueEvaluator::new(plan, *inputs);
        let limit = evaluator.evaluate(limit).ok_or_else(|| other("buffer candidate limit cannot be evaluated"))?;
        table.resolve_count(plan.graph.constant_u32(table.initial), limit).ok_or_else(|| other("buffer candidate count overflows"))?
    };
    if count <= 0 || count > table.cap {
        return Err(other(format!("buffer candidate count {count} is outside 1..={}", table.cap)));
    }
    let mut result = BufferCandidateTable::default();
    let mut key_to_candidate: Vec<(u32, u32)> = Vec::new();
    for index in 0..count {
        let offset = table.min_offset.wrapping_add((index as u32).wrapping_mul(table.candidate_spacing()));
        let mut words = vec![0u32; 4];
        for (dword, word) in words.iter_mut().enumerate() {
            *word = read_scalar_buffer_word(srt, offset, dword as u32 * 4, inputs).ok_or_else(|| other("buffer candidate unreadable"))?;
        }
        // A descriptor whose reserved bits are set is unbound.
        if words[3] >> 30 != 0 {
            words = vec![0; 4];
        }
        if let Some(&(_, existing)) = key_to_candidate.iter().find(|(k, _)| *k == words[0]) {
            // Two descriptors sharing one probe key cannot be told apart.
            if result.descriptors[existing as usize] != words {
                return Err(other("buffer candidates share a probe key"));
            }
            continue;
        }
        key_to_candidate.push((words[0], result.descriptors.len() as u32));
        result.descriptors.push(words);
    }
    result.keys = key_to_candidate.iter().map(|k| k.0).collect();
    result.candidates = key_to_candidate.iter().map(|k| k.1).collect();
    Ok(result)
}

fn diagnose_snapshot_failure(plan: &ShaderResourcePlan, inputs: &ResourceRuntimeInputs<'_>) -> String {
    let clean = RuntimeValueEvaluator::new(plan, inputs.with_reader(inputs.read_clean_memory));
    let evaluator = RuntimeValueEvaluator::with(plan, *inputs, &plan.clean_flat_slots, Some(&clean), None);
    for &source in &plan.materialization_sources {
        let Some(s) = plan.descriptor_sources.get(source as usize) else {
            return format!("descriptor source {source} is outside the source table");
        };
        for (dword, &value) in s.dwords.iter().enumerate() {
            if evaluator.evaluate(value).is_none() {
                return format!("descriptor source {source} dword {dword} cannot be evaluated: {}", plan.graph.describe(value, 6));
            }
        }
    }
    for read in &plan.table_reads {
        let is_clean = plan.clean_flat_slots.get(read.flat_offset as usize).is_some_and(|&c| c != 0);
        let selected = if is_clean { &clean } else { &evaluator };
        if selected.evaluate(read.value).is_none() {
            return format!("resource table read {} cannot be evaluated: {}", read.flat_offset, plan.graph.describe(read.value, 6));
        }
    }
    "descriptor snapshot evaluation failed without an isolated source".into()
}

fn null_image_descriptor(d: &[u32]) -> bool {
    d[0] == 0 && d[1] & 0xFF == 0
}

fn reserved_image_bits_clear(d: &[u32]) -> bool {
    const RESERVED: [u32; 8] = [0, 0x2000_0000, 0xF000_3000, 0, 0xE000_E000, 0xF900_0000, 0x0000_7B00, 0];
    d.iter().zip(RESERVED).all(|(&w, r)| w & r == 0)
}

/// A structurally valid T#.
pub fn valid_image_descriptor(d: &[u32], r128: bool) -> bool {
    use image_format::*;
    let ty = type_of(d);
    let format = format_of(d);
    if ty < TYPE_1D || format == INVALID || format > MAX_FORMAT {
        return false;
    }
    if r128 && !matches!(ty, TYPE_1D | TYPE_2D | TYPE_2D_MSAA) {
        return false;
    }
    if matches!(ty, TYPE_CUBE | TYPE_1D_ARRAY | TYPE_2D_ARRAY | TYPE_2D_MSAA_ARRAY) {
        let last_slice = d[4] & 0x1FFF;
        let base_slice = (d[4] >> 16) & 0x1FFF;
        if base_slice > last_slice {
            return false;
        }
    }
    if matches!(ty, TYPE_2D_MSAA | TYPE_2D_MSAA_ARRAY) {
        let base_level = (d[3] >> 12) & 0xF;
        let fragments = (d[3] >> 16) & 0xF;
        let max_mip = (d[5] >> 4) & 0xF;
        return base_level == 0 && (1..=3).contains(&fragments) && (r128 || max_mip == fragments);
    }
    true
}

fn usable_image_candidate(d: &[u32], r128: bool) -> bool {
    !null_image_descriptor(d)
        && valid_image_descriptor(d, r128)
        && reserved_image_bits_clear(d)
        && image_format::sampled_numeric_class(image_format::format_of(d)) != ImageNumericClass::Unsupported
}

fn scalar_buffer_size(d: &[u32]) -> u64 {
    let stride = (d[1] >> 16) & 0x3FFF;
    if stride == 0 {
        d[2] as u64
    } else {
        stride as u64 * d[2] as u64
    }
}

/// One dword of a V# (clean reader); out of bounds reads zero.
fn read_scalar_buffer_word(d: &[u32], dynamic_offset: u32, immediate: u32, inputs: &ResourceRuntimeInputs<'_>) -> Option<u32> {
    let byte_offset = dynamic_offset as u64 + immediate as u64;
    let aligned = byte_offset & !3;
    let size = scalar_buffer_size(d);
    if aligned > size || size - aligned < 4 {
        return Some(0);
    }
    let base = ((d[0] as u64 | ((d[1] as u64) << 32)) & ADDRESS_MASK) & !3;
    if aligned > ADDRESS_MASK - base {
        return None;
    }
    inputs.read_clean_memory.and_then(|read| read(base + aligned))
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Enumerate every material key that can pass the table's bounds and read
/// the heap descriptor each selects; stale or invalid descriptors become
/// null.
fn materialize_indirect_image(
    plan: &ShaderResourcePlan,
    indirect: &IndirectImageSelector,
    material: &[u32],
    heap: &[u32],
    image: &ImageResource,
    inputs: &ResourceRuntimeInputs<'_>,
    capture_selector_diagnostic: bool,
) -> Result<IndirectImageTable, MaterializationError> {
    if material.len() != 4 || heap.len() != 4 {
        return Err(other("indirect image tables have invalid width"));
    }
    // The program selects each record's image through the host table map
    // (the heap's records from offset 0 on), reading its keys itself.
    let entries = scalar_buffer_size(heap) / TABLE_ENTRY_BYTES;
    let host_map = || {
        (bindless_tables_enabled() && bindless_eligible(image) && entries <= BINDLESS_MAP_MAX_ENTRIES).then(|| {
            let mut words = [0u32; 4];
            words.copy_from_slice(heap);
            IndirectImageTable { keys: vec![0], candidates: vec![0], descriptors: vec![vec![0; 8]], bindless: Some((words, 0, entries as u32)), ..Default::default() }
        })
    };
    let material_stride = (material[1] >> 16) & 0x3FFF;
    if material_stride != indirect.selector_stride {
        // The records cannot be enumerated through this V# (a null V#, or
        // records the planned stride does not describe): only the host
        // map reaches every key the program may read.
        if let Some(table) = host_map() {
            return Ok(table);
        }
        return Err(other("material table stride does not match the selector stride"));
    }
    let step = gcd(indirect.selector_stride as u64, 1u64 << 32);
    let residue = indirect.selector_offset as u64 % step;
    let size = scalar_buffer_size(material);
    let limit = (size + 3).min(u32::MAX as u64);
    let probe_count = if residue <= limit { (limit - residue) / step + 1 } else { 0 };
    let mut diagnostic = capture_selector_diagnostic.then(|| IndirectSelectorDiagnostic {
        selection_mode: "full_domain_no_proof".into(),
        ..Default::default()
    });
    let mut proven: Option<Vec<u32>> = None;
    if let Some(values) = &indirect.selector_values {
        let selectors = values.evaluate(plan, inputs, diagnostic.as_mut());
        if let Some(selectors) = &selectors {
            let mut offsets = Vec::new();
            let mut seen = HashSet::new();
            for &s in selectors {
                let o = s.wrapping_mul(indirect.selector_stride).wrapping_add(indirect.selector_offset);
                if seen.insert(o) {
                    offsets.push(o);
                }
            }
            proven = Some(offsets);
        }
        if let Some(d) = diagnostic.as_mut() {
            d.selection_mode = if selectors.is_some() { "bounded" } else { "full_domain_evaluation_failed" }.into();
            d.selector_indices = selectors.unwrap_or_default();
            d.proven_offsets = proven.clone().unwrap_or_default();
        }
    }
    if proven.is_none() && probe_count > MAX_INDIRECT_IMAGE_PROBES {
        return Err(other(format!("indirect image needs {probe_count} probes")));
    }
    let offsets: Vec<u64> = match &proven {
        Some(p) => p.iter().map(|&o| o as u64).collect(),
        None => (0..probe_count).map(|i| residue + i * step).collect(),
    };
    let mut keys = vec![0u32];
    let mut seen = HashSet::from([0u32]);
    for offset in offsets {
        let key = read_scalar_buffer_word(material, offset as u32, indirect.material_immediate, inputs)
            .ok_or_else(|| other("material key unreadable"))?;
        if seen.insert(key) {
            keys.push(key);
            if let Some(d) = diagnostic.as_mut() {
                d.key_probes.push((offset as u32, key));
            }
        }
    }
    let mut probed = Vec::with_capacity(keys.len());
    for &key in &keys {
        let mut candidate = vec![0u32; 8];
        for (dword, word) in candidate.iter_mut().enumerate() {
            *word = read_scalar_buffer_word(heap, key << 5, dword as u32 * 4, inputs).ok_or_else(|| other("heap descriptor unreadable"))?;
        }
        if !usable_image_candidate(&candidate, image.r128) {
            candidate = vec![0; 8];
        }
        probed.push(candidate);
    }
    let mut table = finish_indirect_image(probed, keys)?;
    // Several candidates: the program selects the record's image through
    // the host table map.
    if table.descriptors.len() > 1 {
        if let Some(map) = host_map() {
            return Ok(map);
        }
    }
    table.material_descriptor = material.to_vec();
    table.heap_descriptor = heap.to_vec();
    table.selector_stride = indirect.selector_stride;
    table.selector_offset = indirect.selector_offset;
    table.selector_diagnostic = diagnostic;
    Ok(table)
}

fn read_clean_word(base: u64, offset: u64, inputs: &ResourceRuntimeInputs<'_>) -> Option<u32> {
    if offset > ADDRESS_MASK || base > ADDRESS_MASK - offset {
        return None;
    }
    inputs.read_clean_memory.and_then(|read| read(base + offset))
}

fn materialize_dense_indirect_image(
    indirect: &IndirectImageSelector,
    heap: &[u32],
    r128: bool,
    inputs: &ResourceRuntimeInputs<'_>,
) -> Result<IndirectImageTable, MaterializationError> {
    if heap.len() != 2 || indirect.key_bound == 0 || indirect.key_bound as u64 > MAX_INDIRECT_IMAGE_PROBES || inputs.read_clean_memory.is_none() {
        return Err(other("dense indirect image cannot be enumerated"));
    }
    let base = ((heap[1] as u64) << 32 | heap[0] as u64) & ADDRESS_MASK;
    let mut probed = Vec::with_capacity(indirect.key_bound as usize);
    for key in 0..indirect.key_bound {
        let entry = indirect.table_offset as u64 + ((key as u64) << 5);
        let mut candidate = vec![0u32; 8];
        for (dword, word) in candidate.iter_mut().enumerate() {
            *word = read_clean_word(base, entry + dword as u64 * 4, inputs).ok_or_else(|| other("dense heap descriptor unreadable"))?;
        }
        if !usable_image_candidate(&candidate, r128) {
            candidate = vec![0; 8];
        }
        probed.push(candidate);
    }
    let keys = (0..indirect.key_bound).map(|k| indirect.dynamic_offset_base.wrapping_add(k << 5)).collect();
    finish_indirect_image(probed, keys)
}

fn materialize_wave_indexed_image(
    indirect: &IndirectImageSelector,
    wave: WaveIndexedImageSelector,
    heap: &[u32],
    r128: bool,
    inputs: &ResourceRuntimeInputs<'_>,
) -> Result<IndirectImageTable, MaterializationError> {
    if heap.len() != 2 || inputs.read_clean_memory.is_none() || wave.index_stride == 0 {
        return Err(other("wave-indexed image cannot be enumerated"));
    }
    let base = ((heap[1] as u64) << 32 | heap[0] as u64) & ADDRESS_MASK;
    let active_mask = read_clean_word(base, wave.mask_offset as u64, inputs).ok_or_else(|| other("wave mask unreadable"))?;
    let mut keys = std::collections::BTreeSet::new();
    for bit in 0..32u64 {
        if active_mask & (1 << bit) == 0 {
            continue;
        }
        let key = read_clean_word(base, wave.index_table_offset as u64 + bit * wave.index_stride as u64, inputs)
            .ok_or_else(|| other("wave index unreadable"))?;
        // A signed-negative index is suppressed before the first-lane read.
        if key & 0x8000_0000 == 0 {
            keys.insert(key);
        }
    }
    let mut probed = Vec::new();
    let mut offsets = Vec::new();
    for key in keys {
        let entry = indirect.table_offset as u64 + ((key as u64) << 5);
        let mut candidate = vec![0u32; 8];
        for (dword, word) in candidate.iter_mut().enumerate() {
            *word = read_clean_word(base, entry + dword as u64 * 4, inputs).ok_or_else(|| other("wave heap descriptor unreadable"))?;
        }
        if !usable_image_candidate(&candidate, r128) {
            candidate = vec![0; 8];
        }
        probed.push(candidate);
        offsets.push(indirect.dynamic_offset_base.wrapping_add(key << 5));
    }
    if probed.is_empty() {
        probed.push(vec![0; 8]);
        offsets.push(0);
    }
    finish_indirect_image(probed, offsets)
}

/// Bindless table slots bound for one table image.
pub const BINDLESS_TABLE_SLOTS: usize = 16;
/// Most records of a table selected through the host table map.
pub const BINDLESS_MAP_MAX_ENTRIES: u64 = 1 << 20;

/// `RPS5E_BINDLESS_TABLES=0`: a bindless table too large to bind per draw
/// (whose keys the draw does not narrow) fails the draw instead of being
/// selected through the host table map.
pub fn bindless_tables_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_BINDLESS_TABLES").is_ok_and(|v| v == "0"))
}

/// A table image the host table map can select: sampled and read-only,
/// with a 256-bit T# and a view the program can declare without its
/// descriptor (not multisampled). Its view is the instruction's dimension
/// with a float class (see [`bindless_specialization`]).
fn bindless_eligible(image: &ImageResource) -> bool {
    image.resource_class == ImageResourceClass::Sampled
        && !image.written
        && !image.atomic
        && !image.r128
        && image.mip_mode == super::info::ImageMipMode::None
        && matches!(image.dimension, ImageDimension::Dim1D | ImageDimension::Dim1DArray | ImageDimension::Dim2D | ImageDimension::Dim2DArray | ImageDimension::Dim3D)
        && super::binding::image_binding::for_image(&ImageResource { numeric_class: ImageNumericClass::Float, cube: false, ..image.clone() }).is_some()
}

/// The specialization of an image selected through the host table map:
/// fixed by the instruction, not by a T# (any record of the table may be
/// the one a wave reads). Float class, the instruction's dimension, no
/// cube view (a cube T# is read as its six faces, the 2D-array layers) and
/// no conversion in the program; the host puts in each slot a view of this
/// shape, and the null view where a T# has none.
fn bindless_specialization(image: &ImageResource, base: ImageSpecialization) -> ImageSpecialization {
    ImageSpecialization {
        numeric_class: ImageNumericClass::Float,
        dimension: image.dimension,
        mip_count: 1,
        conversion_format: image_format::INVALID,
        shader_swizzle: IDENTITY_IMAGE_SWIZZLE,
        cube: false,
        emulated_compare: 0,
        bindless: true,
        ..base
    }
}
/// Most material records scanned to enumerate a table's keys; beyond it the
/// whole table is bound instead.
pub const BINDLESS_MATERIAL_SCAN_LIMIT: u64 = 256;
const TABLE_ENTRY_BYTES: u64 = 32;

/// The candidates of a bindless table image: the keys the material
/// records reach (mode M), or every table entry when the table is small
/// (mode T). Null, invalid or differently shaped entries become null
/// descriptors (they sample zero, as a null T# does).
/// The key of table image `image_index` when the draw determines it: the
/// key value evaluates from user data and guest memory alone (a material
/// index in a per-draw constant buffer, say). None when it depends on
/// lanes or on memory that cannot be read clean.
fn draw_uniform_table_key(plan: &ShaderResourcePlan, image_index: usize, inputs: &ResourceRuntimeInputs<'_>) -> Option<u32> {
    let access = plan.indirect_images.iter().find(|a| {
        !a.key_is_address_offset && a.memory_index < plan.graph.memory.len() && plan.graph.memory.get(a.memory_index).resource == image_index as u32
    })?;
    RuntimeValueEvaluator::new(plan, *inputs).evaluate(access.key)
}

#[allow(clippy::too_many_arguments)]
fn materialize_table_image(
    indirect: &IndirectImageSelector,
    table: TableImageSelector,
    heap: &[u32],
    material: &[u32],
    image: &ImageResource,
    draw_key: Option<u32>,
    inputs: &ResourceRuntimeInputs<'_>,
) -> Result<IndirectImageTable, MaterializationError> {
    let r128 = image.r128;
    if heap.len() != 4 || material.len() != 4 || inputs.read_clean_memory.is_none() {
        return Err(other("bindless table cannot be read"));
    }
    let heap_size = scalar_buffer_size(heap);
    let entries = if heap_size > table.entry_offset as u64 { (heap_size - table.entry_offset as u64) / TABLE_ENTRY_BYTES } else { 0 };
    let base_of = |d: &[u32]| (d[0] as u64 | ((d[1] as u64) << 32)) & ADDRESS_MASK;
    // Selected through the host table map: the program reads each
    // record's image, whatever its class or shape.
    let bindless = bindless_tables_enabled() && bindless_eligible(image) && entries <= BINDLESS_MAP_MAX_ENTRIES;
    let host_map = || {
        let mut words = [0u32; 4];
        words.copy_from_slice(heap);
        IndirectImageTable {
            keys: vec![0],
            candidates: vec![0],
            descriptors: vec![vec![0; 8]],
            bindless: Some((words, table.entry_offset, entries as u32)),
            ..Default::default()
        }
    };
    let mut keys: Vec<u32> = Vec::new();
    let mut material_mode = false;
    if table.has_material && indirect.selector_stride != 0 {
        let material_size = scalar_buffer_size(material);
        let records = material_size / indirect.selector_stride as u64;
        if records <= BINDLESS_MATERIAL_SCAN_LIMIT {
            material_mode = true;
            for record in 0..records {
                let offset = record * indirect.selector_stride as u64 + indirect.selector_offset as u64;
                if offset + 4 > material_size {
                    break;
                }
                let key = read_clean_word(base_of(material), offset, inputs).ok_or_else(|| other("material record unreadable"))?;
                if (key as u64) < entries {
                    keys.push(key);
                }
            }
            keys.sort_unstable();
            keys.dedup();
            // The scan only narrows the table to the keys the records
            // hold. Records that name no entry of it (a null or empty
            // material V#: the program reads its key some other way, or
            // reads zero) narrow nothing: the table modes below bind
            // every entry the program may reach.
            if keys.is_empty() {
                material_mode = false;
            }
            if keys.len() > BINDLESS_TABLE_SLOTS {
                if bindless {
                    return Ok(host_map());
                }
                return Err(MaterializationError::ImageCapacityExceeded(format!(
                    "bindless table: {} distinct keys exceed {BINDLESS_TABLE_SLOTS} slots",
                    keys.len()
                )));
            }
        }
    }
    // Mode D: a table too large to bind whose key the draw determines
    // (evaluated from user data and memory): its one entry is the image.
    if !material_mode && entries > BINDLESS_TABLE_SLOTS as u64 {
        if let Some(key) = draw_key.filter(|&k| (k as u64) < entries) {
            keys.push(key);
            material_mode = true;
        }
    }
    if !material_mode {
        if entries > BINDLESS_TABLE_SLOTS as u64 {
            // Every record is reachable: the host selects each record's
            // image through its table map, so nothing of the table is
            // read here (its size alone decides).
            if bindless {
                return Ok(host_map());
            }
            return Err(MaterializationError::ImageCapacityExceeded(format!(
                "bindless table has {entries} entries, limit {BINDLESS_TABLE_SLOTS}"
            )));
        }
        keys = (0..entries as u32).collect();
    }
    if keys.is_empty() {
        return Err(other(format!(
            "bindless table selects no entry (entries {entries}, material {} records of {} bytes, bindless {bindless}, offset {:#x}, material V# {:08x?}, heap V# {:08x?})",
            if indirect.selector_stride != 0 { scalar_buffer_size(material) / indirect.selector_stride as u64 } else { 0 },
            indirect.selector_stride,
            table.entry_offset,
            material,
            heap
        )));
    }
    let mut probed = Vec::with_capacity(keys.len());
    let mut shape: Option<(ImageNumericClass, u32)> = None;
    for &key in &keys {
        let entry = table.entry_offset as u64 + key as u64 * TABLE_ENTRY_BYTES;
        let mut candidate = vec![0u32; 8];
        for (dword, word) in candidate.iter_mut().enumerate() {
            *word = read_clean_word(base_of(heap), entry + dword as u64 * 4, inputs).ok_or_else(|| other("bindless entry unreadable"))?;
        }
        let usable = usable_image_candidate(&candidate, r128);
        let this_shape = (image_format::sampled_numeric_class(image_format::format_of(&candidate)), image_format::type_of(&candidate));
        let converted = image_conversion_format(image_format::format_of(&candidate)) != image_format::INVALID;
        if usable && !converted && shape.is_some_and(|s| s != this_shape) && bindless {
            // Records of several classes or shapes: one binding cannot
            // hold them all.
            return Ok(host_map());
        }
        if !usable || converted || shape.is_some_and(|s| s != this_shape) {
            candidate = vec![0; 8];
        } else if shape.is_none() {
            shape = Some(this_shape);
        }
        probed.push(candidate);
    }
    if shape.is_none() {
        // Every record the draw reaches is null: they read zeros.
        if bindless {
            return Ok(host_map());
        }
        return Err(other("bindless table has no valid entry"));
    }
    let mut result = finish_indirect_image(probed, keys)?;
    // The first candidate must be typed: it is the root binding.
    if null_image_descriptor(&result.descriptors[0]) {
        if let Some(typed) = result.descriptors.iter().position(|d| !null_image_descriptor(d)) {
            result.descriptors.swap(0, typed);
            for c in &mut result.candidates {
                *c = if *c == 0 {
                    typed as u32
                } else if *c == typed as u32 {
                    0
                } else {
                    *c
                };
            }
        }
    }
    result.heap_descriptor = heap.to_vec();
    result.material_descriptor = material.to_vec();
    result.selector_stride = indirect.selector_stride;
    result.selector_offset = indirect.selector_offset;
    Ok(result)
}

fn finish_indirect_image(probed: Vec<Vec<u32>>, keys: Vec<u32>) -> Result<IndirectImageTable, MaterializationError> {
    let mut table = IndirectImageTable { keys, ..Default::default() };
    for candidate in probed {
        let found = match table.descriptors.iter().position(|d| *d == candidate) {
            Some(f) => f,
            None => {
                if table.descriptors.len() >= ShaderResourceInfo::MAX_IMAGES {
                    return Err(MaterializationError::ImageCapacityExceeded("indirect image candidates exceed the image limit".into()));
                }
                table.descriptors.push(candidate);
                table.descriptors.len() - 1
            }
        };
        table.candidates.push(found as u32);
    }
    Ok(table)
}

// ---- specialization ----

/// The binding dimension of a T# for an access of `requested` dimension.
/// A 3D access of a 2D-array T# reads it as a 2D array with r (or z) as
/// the layer (measured on RDNA2).
pub fn descriptor_dimension(d: &[u32], requested: ImageDimension) -> ImageDimension {
    use image_format::*;
    use ImageDimension as D;
    let is_array = matches!(requested, D::Dim1DArray | D::Dim2DArray | D::Dim2DMsaaArray);
    match type_of(d) {
        TYPE_1D => D::Dim1D,
        TYPE_1D_ARRAY => {
            if is_array {
                D::Dim1DArray
            } else {
                D::Dim1D
            }
        }
        TYPE_3D => D::Dim3D,
        TYPE_CUBE => D::Dim2DArray,
        TYPE_2D_ARRAY => {
            if is_array || requested == D::Dim3D {
                D::Dim2DArray
            } else {
                D::Dim2D
            }
        }
        TYPE_2D_MSAA_ARRAY => {
            if is_array {
                D::Dim2DMsaaArray
            } else {
                D::Dim2DMsaa
            }
        }
        TYPE_2D => D::Dim2D,
        TYPE_2D_MSAA => D::Dim2DMsaa,
        _ => D::Unknown,
    }
}

/// An 11_11_10 UNORM image is sampled in the shader (its four footprint
/// words gathered and filtered bilinearly): a centred single-level 2D T#
/// and normalized bilinear clamp-to-edge samplers without anisotropy,
/// reduction, minimum LOD, degamma or coordinate truncation.
fn validate_packed_unorm_sampling(info: &ShaderResourceInfo, samplers: &[DescriptorWords], image: u32, descriptor: &[u32]) -> Result<(), MaterializationError> {
    let base_level = (descriptor[3] >> 12) & 0xF;
    let last_level = (descriptor[3] >> 16) & 0xF;
    let min_lod = (descriptor[1] >> 8) & 0xFFF;
    let corner_sample = (descriptor[5] >> 23) & 1 != 0;
    if image_format::type_of(descriptor) != image_format::TYPE_2D || base_level != last_level || min_lod != 0 || corner_sample {
        return Err(MaterializationError::Other("packed UNORM sampling requires a centered, single-mip 2D image".into()));
    }
    for pair in info.sampled_pairs.iter().filter(|p| p.image == image) {
        let Some(w) = samplers.get(pair.sampler as usize).filter(|s| s.len() >= 4) else {
            return Err(MaterializationError::Other("packed UNORM sampling requires a direct sampler".into()));
        };
        let bilinear = (w[2] >> 20) & 3 == 1 && (w[2] >> 22) & 3 == 1;
        let edge = w[0] & 7 == 2 && (w[0] >> 3) & 7 == 2;
        let plain = (w[0] >> 15) & 1 == 0 // FORCE_UNNORMALIZED
            && (w[0] >> 9) & 7 == 0 // MAX_ANISO_RATIO
            && (w[0] >> 29) & 3 == 0 // FILTER_MODE
            && w[1] & 0xFFF == 0 // MIN_LOD
            && (w[0] >> 20) & 1 == 0 // FORCE_DEGAMMA
            && (w[0] >> 27) & 1 == 0 // TRUNC_COORD
            && (w[0] >> 19) & 1 == 0; // MC_COORD_TRUNC
        if !bilinear || !edge || !plain {
            return Err(MaterializationError::Other(format!("packed UNORM image {image} requires normalized bilinear clamp sampling (sampler {})", pair.sampler)));
        }
    }
    Ok(())
}

/// `swizzle` with its X (4) and Z (6) selectors exchanged.
fn swap_x_z_selectors(swizzle: u32) -> u32 {
    (0..4).fold(0, |acc, c| {
        let s = (swizzle >> (c * 3)) & 7;
        let s = match s {
            4 => 6,
            6 => 4,
            other => other,
        };
        acc | (s << (c * 3))
    })
}

fn image_conversion_format(format: u32) -> u32 {
    if image_format::remap(format) != format {
        format
    } else {
        image_format::INVALID
    }
}

/// Integer images are never filtered (a host linear sampler is invalid for
/// them), and converted formats are read as raw words.
fn requires_point_sampler(numeric_class: ImageNumericClass, conversion_format: u32) -> bool {
    matches!(numeric_class, ImageNumericClass::Sint | ImageNumericClass::Uint) || conversion_format != image_format::INVALID
}

/// The depth compare function (sampler word 0 bits 14:12) of the sampler
/// paired with `image`.
/// The [`emulated_compare`] state of a depth-compare sample of a colour
/// image (no host depth view): the guest compare function, filter and
/// addressing of its sampler and how the reference is clamped. 0 when the
/// shape cannot be emulated (1D, 2D and their array views of float, unorm or snorm
/// formats read through their red channel, point or bilinear filtering,
/// wrap / edge / border addressing without a border table).
pub(super) fn emulated_compare_state(info: &ShaderResourceInfo, samplers: &[DescriptorWords], image: u32, descriptor: &[u32]) -> u32 {
    use emulated_compare::*;
    let format = image_format::format_of(descriptor);
    let reference = match super::format::unified_format::decode(format) {
        // 8, 8_8 and 8_8_8_8 UNORM and sRGB: every comparison is 0
        // (measured on gfx1035 over references -0.5..1.5 and texel codes
        // 0..255).
        Some((1 | 3 | 10, 0 | 9)) => REFERENCE_ZERO,
        Some((_, 0)) => REFERENCE_UNORM,
        Some((_, 1)) => REFERENCE_SNORM,
        Some((_, 7)) => REFERENCE_FLOAT,
        _ => return 0,
    };
    let ty = image_format::type_of(descriptor);
    if !matches!(ty, image_format::TYPE_1D | image_format::TYPE_1D_ARRAY | image_format::TYPE_2D | image_format::TYPE_2D_ARRAY) {
        return 0;
    }
    // The view's X channel must be red.
    if descriptor[3] & 7 != 4 {
        return 0;
    }
    let mut state = None;
    for pair in info.sampled_pairs.iter().filter(|p| p.image == image) {
        let Some(w) = samplers.get(pair.sampler as usize).filter(|s| s.len() >= 4) else { return 0 };
        let address = |clamp: u32| match clamp {
            0 => Some(ADDRESS_WRAP),
            2 => Some(ADDRESS_EDGE),
            6 => Some(ADDRESS_BORDER),
            _ => None,
        };
        let (Some(x), Some(y)) = (address(w[0] & 7), address((w[0] >> 3) & 7)) else { return 0 };
        let function = (w[0] >> 12) & 7;
        let unnormalized = (w[0] >> 15) & 1 != 0;
        let reduction = (w[0] >> 29) & 3;
        let (mag, min) = ((w[2] >> 20) & 3, (w[2] >> 22) & 3);
        let border_type = (w[3] >> 30) & 3;
        let border = x == ADDRESS_BORDER || y == ADDRESS_BORDER;
        if unnormalized || reduction != 0 || mag != min || mag > 1 || (border && border_type == 3) {
            return 0;
        }
        let s = ENABLED
            | function << FUNCTION_SHIFT
            | if mag == 1 { LINEAR } else { 0 }
            | x << CLAMP_X_SHIFT
            | y << CLAMP_Y_SHIFT
            | if border && border_type == 2 { BORDER_WHITE } else { 0 };
        if state.is_some_and(|p| p != s) {
            return 0;
        }
        state = Some(s);
    }
    let Some(state) = state else { return 0 };
    let single_level = (descriptor[3] >> 12) & 0xF == (descriptor[3] >> 16) & 0xF;
    state | reference << REFERENCE_SHIFT | if single_level { SINGLE_LEVEL } else { 0 }
}

fn storage_mip_count(image: &ImageResource, d: &[u32]) -> u32 {
    if image.mip_mode != ImageMipMode::Dynamic || null_image_descriptor(d) {
        return 1;
    }
    let base_level = (d[3] >> 12) & 0xF;
    let last = (d[3] >> 16) & 0xF;
    if base_level <= last {
        last - base_level + 1
    } else {
        0
    }
}

fn buffer_specialization(words: &[u32], formatted: bool) -> BufferSpecialization {
    let stride = (words[1] >> 16) & 0x3FFF;
    let swizzle_enabled = words[1] >> 31 != 0;
    let index_stride = (words[3] >> 21) & 3;
    let add_thread_id = (words[3] >> 23) & 1;
    let mut packed = stride | (swizzle_enabled as u32) << 14 | index_stride << 16 | add_thread_id << 20;
    let swizzle = stride != 0 && (packed >> 14) & 1 != 0;
    if stride == 0 {
        packed &= !((1 << 14) | (3 << 16));
    } else if !swizzle {
        packed &= !(3 << 16);
    }
    // OOB_SELECT picks the range rule the access compiles to.
    packed |= ((words[3] >> 28) & 3) << 24;
    BufferSpecialization {
        packed_stride: packed,
        descriptor_format: if formatted { (words[3] >> 12) & 0x7F } else { INVALID_FORMAT },
        descriptor_swizzle: if formatted { words[3] & 0xFFF } else { IDENTITY_DESTINATION_SELECT },
    }
}

fn search_iterations(keys: usize) -> u32 {
    (keys as u32).ilog2() + 1
}

/// Write a sorted `[count, key0, candidate0, ...]` mapping at `offset`.
fn write_mapping(table: &mut [u32], offset: usize, keys: &[u32], candidates: &[u32]) {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by_key(|&i| keys[i]);
    table[offset] = keys.len() as u32;
    for (entry, &source) in order.iter().enumerate() {
        table[offset + 1 + entry * 2] = keys[source];
        table[offset + 2 + entry * 2] = candidates[source];
    }
}

fn build_specialization(plan: &ShaderResourcePlan, snapshot: &mut MaterializedSnapshot) -> Result<ResourceSpecialization, MaterializationError> {
    let info = &plan.info;
    let mut image_count = info.images.len();
    let mut mapping_words = 0usize;
    for table in &snapshot.indirect_images {
        if table.resource as usize >= info.images.len() || table.descriptors.len() < 2 {
            return Err(other("indirect image table has an invalid root or candidate count"));
        }
        if image_count + table.descriptors.len() - 1 > ShaderResourceInfo::MAX_IMAGES {
            return Err(MaterializationError::ImageCapacityExceeded("indirect image candidates exceed the dense image resource limit".into()));
        }
        image_count += table.descriptors.len() - 1;
        mapping_words += 1 + table.keys.len() * 2;
    }
    for table in &snapshot.buffer_candidate_tables {
        mapping_words += 1 + table.keys.len() * 2;
    }
    mapping_words += snapshot.bindless_tables.len() * super::binding::bindless::TABLE_WORDS as usize;
    let mut mapping_cursor = snapshot.flattened_table.len();
    snapshot.flattened_table.resize(mapping_cursor + mapping_words, 0);
    let mut images: Vec<ImageSpecialization> = info
        .images
        .iter()
        .map(|i| ImageSpecialization {
            numeric_class: i.numeric_class,
            dimension: i.dimension,
            mip_count: i.mip_count,
            conversion_format: i.conversion_format,
            shader_swizzle: i.shader_swizzle,
            indirect_root: i.indirect_root,
            indirect_mapping_offset: i.indirect_mapping_offset,
            indirect_search_iterations: i.indirect_search_iterations,
            cube: i.cube,
            emulated_compare: 0,
            bindless: false,
            packed_format: INVALID_FORMAT,
            store_quantize: 0,
        })
        .collect();
    for table in &snapshot.bindless_tables {
        let index = table.image as usize;
        let Some(base) = info.images.get(index) else {
            return Err(other("bindless table has an invalid image"));
        };
        images[index] = bindless_specialization(base, images[index]);
    }
    for table in &snapshot.indirect_images {
        let root = images[table.resource as usize];
        for candidate in 1..table.descriptors.len() {
            images.push(ImageSpecialization { indirect_root: table.resource, ..root });
            snapshot.images.push(table.descriptors[candidate].as_slice().into());
        }
        let mapping_offset = mapping_cursor;
        images[table.resource as usize] = ImageSpecialization {
            indirect_root: table.resource,
            indirect_mapping_offset: mapping_offset as u32,
            indirect_search_iterations: search_iterations(table.keys.len()),
            ..root
        };
        mapping_cursor += 1 + table.keys.len() * 2;
        write_mapping(&mut snapshot.flattened_table, mapping_offset, &table.keys, &table.candidates);
        snapshot.images[table.resource as usize] = table.descriptors[0].as_slice().into();
    }

    let mut buffers = Vec::with_capacity(info.buffers.len());
    for (index, buffer) in info.buffers.iter().enumerate() {
        let words = &mut snapshot.buffers[index];
        if words.len() != 4 {
            return Err(other(format!("buffer descriptor {index} has invalid width")));
        }
        if words[3] >> 30 != 0 {
            words.iter_mut().for_each(|w| *w = 0);
            snapshot.buffer_zeroed.resize(index + 1, false);
            snapshot.buffer_zeroed[index] = true;
        }
        buffers.push(buffer_specialization(words, buffer.formatted));
    }

    for index in 0..images.len() {
        let mut image = images[index];
        if image.bindless {
            // Its shape is the instruction's (the bound T# is null).
            continue;
        }
        let base_index = if index < info.images.len() { index as u32 } else { image.indirect_root };
        let Some(base) = info.images.get(base_index as usize) else {
            return Err(other(format!("image resource {index} has an invalid root")));
        };
        if base.resource_class == ImageResourceClass::None || (base.atomic && base.resource_class != ImageResourceClass::Storage) {
            return Err(other(format!("image resource {index} has an invalid class")));
        }
        let descriptor = snapshot.images[index].clone();
        let mip_count = storage_mip_count(base, &descriptor);
        if mip_count == 0 {
            return Err(other(format!("image descriptor {index} has an invalid mip range")));
        }
        image.mip_count = mip_count;
        if null_image_descriptor(&descriptor) {
            images[index] = ImageSpecialization {
                // A null-bound atomic keeps the unsigned atomic view.
                numeric_class: if base.atomic { ImageNumericClass::Uint } else { ImageNumericClass::Float },
                dimension: ImageDimension::Dim2D,
                cube: false,
                ..image
            };
            continue;
        }
        let dimension = descriptor_dimension(&descriptor, base.dimension);
        if dimension == ImageDimension::Unknown {
            return Err(other(format!(
                "image descriptor {index} has unsupported type {}: {:08x?}",
                image_format::type_of(&descriptor),
                descriptor
            )));
        }
        let format = image_format::format_of(&descriptor);
        // Integer image atomics use a 32-bit uint view; float atomics (CAS
        // loops on the raw bits) target 32-bit float formats; 64-bit
        // atomics a 32_32 image of any number type (its bits viewed as R64
        // uint).
        let atomic_format_ok = if base.atomic64 {
            matches!(format, image_format::FORMAT_32_32_UINT | image_format::FORMAT_32_32_SINT | image_format::FORMAT_32_32_FLOAT)
        } else {
            matches!(format, image_format::FORMAT_32_UINT | image_format::FORMAT_32_SINT | image_format::FORMAT_32_FLOAT)
        };
        if base.atomic && !atomic_format_ok {
            return Err(other(format!("atomic image descriptor {index} uses unsupported format {format}")));
        }
        let storage = base.resource_class == ImageResourceClass::Storage;
        // Signed integer storage images are viewed as unsigned: the shader
        // saturates and masks the stored fields and sign-extends the loaded
        // ones (the format rides as the conversion format).
        let raw_sint_storage = storage && image_format::raw_sint_storage_format(format);
        let conversion_format = if raw_sint_storage { format } else { image_conversion_format(format) };
        // Storage, converted and raw-texel (PCK, BY) accesses see the T#
        // swizzle in the shader (their views map identically).
        let mut shader_swizzle =
            if storage || conversion_format != image_format::INVALID || base.packed { descriptor[3] & 0xFFF } else { image.shader_swizzle };
        if storage && image_format::host_reverses_channels(format) {
            // The storage view reads and writes the host channel order:
            // guest X and Z live in host Z and X.
            shader_swizzle = swap_x_z_selectors(shader_swizzle);
        }
        // A converted format is viewed as the format it is read as (the
        // shader converts).
        let mut numeric_class = image_format::sampled_numeric_class(image_format::remap(format));
        // Atomics work on the texel bits through a uint view.
        if storage && (base.atomic64 || (base.atomic && format == image_format::FORMAT_32_SINT)) {
            numeric_class = ImageNumericClass::Uint;
        }
        if storage {
            if (!raw_sint_storage && numeric_class == ImageNumericClass::Sint) || numeric_class == ImageNumericClass::Unsupported {
                return Err(other(format!("storage image descriptor {index} uses unsupported format {format}")));
            }
            // Atomics use the uint view; float min/max compare-exchange its
            // bits.
            if raw_sint_storage || (base.atomic && numeric_class == ImageNumericClass::Float) {
                numeric_class = ImageNumericClass::Uint;
            }
        } else if numeric_class == ImageNumericClass::Unsupported {
            // An unsupported sampled format binds as a null texture.
            snapshot.images[index].iter_mut().for_each(|w| *w = 0);
            snapshot.image_zeroed.resize(index + 1, false);
            snapshot.image_zeroed[index] = true;
            images[index] = ImageSpecialization { numeric_class: ImageNumericClass::Float, dimension: ImageDimension::Dim2D, cube: false, ..image };
            continue;
        } else if base.depth_compare && numeric_class != ImageNumericClass::Float {
            return Err(other(format!("sampled image descriptor {index} uses unsupported format {format}")));
        }
        if !storage && format == image_format::FORMAT_11X2X10_UNORM && info.sampled_pairs.iter().any(|p| p.image == base_index) {
            if index >= info.images.len() || image.indirect_root != NO_INDEX {
                return Err(other("packed UNORM sampling requires a direct image"));
            }
            validate_packed_unorm_sampling(info, &snapshot.samplers, base_index, &descriptor)?;
        }
        images[index] = ImageSpecialization {
            dimension,
            packed_format: if base.packed { format } else { image_format::INVALID },
            store_quantize: if storage && base.written && !base.atomic && !base.packed && conversion_format == image_format::INVALID {
                super::info::store_quantize_code(format)
            } else {
                0
            },
            cube: image_format::type_of(&descriptor) == image_format::TYPE_CUBE,
            conversion_format,
            shader_swizzle,
            numeric_class,
            // 1D views have no compare binding: the compares of a 1D T#
            // are emulated whatever its format.
            emulated_compare: if !storage
                && base.depth_compare
                && (!image_format::has_depth_equivalent(format) || matches!(dimension, ImageDimension::Dim1D | ImageDimension::Dim1DArray))
            {
                emulated_compare_state(info, &snapshot.samplers, base_index, &descriptor)
            } else {
                0
            },
            ..image
        };
    }

    for root_index in 0..images.len() {
        let root = images[root_index];
        if root.indirect_root as usize != root_index {
            continue;
        }
        let key_count = snapshot.flattened_table.get(root.indirect_mapping_offset as usize).copied().unwrap_or(0);
        if root.indirect_search_iterations == 0
            || key_count < 2
            || root.indirect_mapping_offset as u64 + 1 + key_count as u64 * 2 > snapshot.flattened_table.len() as u64
        {
            return Err(other("indirect image specialization has an invalid key mapping"));
        }
        let members: Vec<usize> = (0..images.len()).filter(|&r| images[r].indirect_root as usize == root_index).collect();
        let exemplar = members.iter().copied().find(|&r| !null_image_descriptor(&snapshot.images[r]));
        let Some(exemplar) = exemplar.filter(|_| members.len() >= 2) else {
            return Err(other("indirect image specialization has no typed candidate"));
        };
        let class = images[exemplar];
        let separate_sampled_dimensions = info.images[root_index].resource_class == ImageResourceClass::Sampled;
        for &candidate in &members {
            let mut image = images[candidate];
            if null_image_descriptor(&snapshot.images[candidate]) {
                image = ImageSpecialization {
                    numeric_class: class.numeric_class,
                    dimension: class.dimension,
                    mip_count: class.mip_count,
                    conversion_format: class.conversion_format,
                    shader_swizzle: class.shader_swizzle,
                    cube: class.cube,
                    ..image
                };
                images[candidate] = image;
            }
            // Each sampled candidate has its own binding and coordinate
            // width.
            let two_d = |d: ImageDimension| matches!(d, ImageDimension::Dim2D | ImageDimension::Dim2DArray);
            let compatible_dimensions = image.dimension == class.dimension
                || (separate_sampled_dimensions && !image.cube && !class.cube && two_d(image.dimension) && two_d(class.dimension));
            if image.numeric_class != class.numeric_class
                || !compatible_dimensions
                || image.mip_count != class.mip_count
                || image.conversion_format != class.conversion_format
                || image.shader_swizzle != class.shader_swizzle
                || image.cube != class.cube
            {
                let table = snapshot.indirect_images.iter().find(|t| t.resource as usize == root_index);
                return Err(MaterializationError::IncompatibleImageCandidates(Box::new(IndirectImageFailure {
                    instruction_address: info.images[root_index].first_use_pc,
                    root_resource: root_index as u32,
                    exemplar_resource: exemplar as u32,
                    incompatible_resource: candidate as u32,
                    selector_stride: table.map_or(0, |t| t.selector_stride),
                    selector_offset: table.map_or(0, |t| t.selector_offset),
                    material_descriptor: table.map_or(vec![], |t| t.material_descriptor.clone()),
                    heap_descriptor: table.map_or(vec![], |t| t.heap_descriptor.clone()),
                    keys: table.map_or(vec![], |t| t.keys.clone()),
                    candidate_indices: table.map_or(vec![], |t| t.candidates.clone()),
                    table_descriptors: table.map_or(vec![], |t| t.descriptors.clone()),
                    image_descriptors: snapshot.images.iter().map(|d| d.to_vec()).collect(),
                    image_specializations: images.clone(),
                    user_data: snapshot.user_data.clone(),
                    selector_diagnostic: table.and_then(|t| t.selector_diagnostic.clone()),
                })));
            }
        }
    }

    let sampler_plan = build_sampler_plan(info, &images).ok_or_else(|| other("specialized sampler layout does not match the plan"))?;
    snapshot.samplers.resize(sampler_plan.sampler_count as usize, DescriptorWords::new());
    snapshot.sampler_sources = (0..info.samplers.len() as u32).collect();
    snapshot.sampler_sources.resize(snapshot.samplers.len(), NO_INDEX);
    for index in 0..info.samplers.len() {
        let target = sampler_plan.point_sampler[index];
        if target != NO_INDEX && target as usize >= info.samplers.len() {
            snapshot.samplers[target as usize] = snapshot.samplers[index].clone();
            snapshot.sampler_sources[target as usize] = snapshot.sampler_sources[index];
        }
    }
    // apply_specialization appends a depth-compare copy of every sampler
    // shared by ordinary and depth-reference sampling, after the point
    // samplers; the snapshot carries the same words there.
    let mut compare_usage = vec![0u8; snapshot.samplers.len()];
    for pair in &info.sampled_pairs {
        let base = &info.images[pair.image as usize];
        let specialized = images[pair.image as usize];
        let sampler = if requires_point_sampler(specialized.numeric_class, specialized.conversion_format) {
            sampler_plan.point_sampler[pair.sampler as usize]
        } else {
            pair.sampler
        };
        let depth_compare = base.depth_compare && specialized.emulated_compare == 0;
        if (sampler as usize) < compare_usage.len() {
            compare_usage[sampler as usize] |= if depth_compare { 2 } else { 1 };
        }
    }
    let count = snapshot.samplers.len();
    for index in 0..count.min(compare_usage.len()) {
        if compare_usage[index] == 3 {
            let words = snapshot.samplers[index].clone();
            snapshot.samplers.push(words);
            let source = snapshot.sampler_sources[index];
            snapshot.sampler_sources.push(source);
        }
    }

    // Bounded runtime V# candidates follow the plan's buffers.
    let base_buffer_count = buffers.len();
    let mut buffer_cursor = snapshot.buffers.len();
    let mut candidate_tables = Vec::with_capacity(snapshot.buffer_candidate_tables.len());
    for table in &snapshot.buffer_candidate_tables {
        if table.descriptors.is_empty() || table.keys.is_empty() {
            return Err(other("buffer candidate table has no candidates"));
        }
        for descriptor in &table.descriptors {
            let mut words = descriptor.clone();
            if words[3] >> 30 != 0 {
                words = vec![0; 4];
            }
            buffers.push(buffer_specialization(&words, true));
            snapshot.buffers.push(words.into());
        }
        let mapping_offset = mapping_cursor;
        write_mapping(&mut snapshot.flattened_table, mapping_offset, &table.keys, &table.candidates);
        mapping_cursor += 1 + table.keys.len() * 2;
        candidate_tables.push(BufferCandidateTableSpecialization {
            first_candidate: buffer_cursor as u32,
            candidate_count: table.descriptors.len() as u32,
            mapping_offset: mapping_offset as u32,
            search_iterations: search_iterations(table.keys.len()),
        });
        buffer_cursor += table.descriptors.len();
    }
    // The host fills the words of each table map (left zero here).
    for table in &mut snapshot.bindless_tables {
        table.mapping_offset = mapping_cursor as u32;
        images[table.image as usize].indirect_mapping_offset = mapping_cursor as u32;
        mapping_cursor += super::binding::bindless::TABLE_WORDS as usize;
    }
    Ok(ResourceSpecialization { base_buffer_count, buffers, images, buffer_candidate_tables: candidate_tables })
}

/// The samplers a specialization binds: the plan's samplers, then a point
/// copy of each one also used with point-only images. The copies are not
/// limited by the plan's sampler count (bindings are sized per program).
struct SamplerPlan {
    point_sampler: Vec<u32>,
    sampler_count: u32,
}

/// A sampler some pair uses with a point-only image (sint, converted format)
/// needs a point-filtering copy; when every pair does, the sampler itself
/// switches.
fn build_sampler_plan(info: &ShaderResourceInfo, images: &[ImageSpecialization]) -> Option<SamplerPlan> {
    let mut plan = SamplerPlan { point_sampler: vec![NO_INDEX; info.samplers.len()], sampler_count: info.samplers.len() as u32 };
    let mut usage = vec![0u8; info.samplers.len()];
    for pair in &info.sampled_pairs {
        let image = images.get(pair.image as usize)?;
        if pair.sampler as usize >= info.samplers.len() {
            return None;
        }
        usage[pair.sampler as usize] |= if requires_point_sampler(image.numeric_class, image.conversion_format) {
            2
        } else if image.bindless && !info.images.get(pair.image as usize).is_some_and(|i| i.depth_compare) {
            // Float T#s with the sampler, integer ones with a point copy.
            3
        } else {
            1
        };
    }
    for index in 0..info.samplers.len() {
        if usage[index] & 2 == 0 {
            continue;
        }
        if usage[index] & 1 == 0 {
            plan.point_sampler[index] = index as u32;
        } else {
            plan.point_sampler[index] = plan.sampler_count;
            plan.sampler_count += 1;
        }
    }
    Some(plan)
}

/// Apply a specialization to the plan's tables: buffer strides and formats,
/// image classes and indirect candidates, point and compare samplers, and
/// the sampler each access uses.
pub fn apply_specialization(plan: &ShaderResourcePlan, specialization: &ResourceSpecialization) -> Result<SpecializedResourceInfo, ResourcePlanError> {
    let source = &plan.info;
    if source.buffers.len() != specialization.base_buffer_count || source.images.len() > specialization.images.len() {
        return Err(ResourcePlanError::new(format!(
            "shader resource specialization does not match the plan: hash=0x{:016X} stage={:?} buffers={}/{} images={}/{}",
            plan.hash,
            plan.stage,
            source.buffers.len(),
            specialization.base_buffer_count,
            source.images.len(),
            specialization.images.len()
        )));
    }
    let mut info = source.clone();
    for (buffer, s) in info.buffers.iter_mut().zip(&specialization.buffers) {
        buffer.packed_stride = s.packed_stride;
        buffer.descriptor_format = s.descriptor_format;
        buffer.descriptor_swizzle = s.descriptor_swizzle;
    }
    // Candidate buffers follow the plan's buffers, one native binding each.
    for s in &specialization.buffers[source.buffers.len()..] {
        let mut buffer = BufferResource::new(NO_INDEX, 0);
        buffer.read = true;
        buffer.packed_stride = s.packed_stride;
        buffer.descriptor_format = s.descriptor_format;
        buffer.descriptor_swizzle = s.descriptor_swizzle;
        info.buffers.push(buffer);
    }
    for t in &specialization.buffer_candidate_tables {
        info.buffer_candidate_tables.push(BufferCandidateTableInfo {
            first_candidate: t.first_candidate,
            candidate_count: t.candidate_count,
            mapping_offset: t.mapping_offset,
            search_iterations: t.search_iterations,
        });
    }
    for (index, s) in specialization.images.iter().enumerate() {
        if index >= info.images.len() {
            let root = source.images.get(s.indirect_root as usize).ok_or_else(|| {
                ResourcePlanError::new(format!(
                    "shader resource specialization names an invalid indirect root: hash=0x{:016X} image={index}",
                    plan.hash
                ))
            })?;
            info.images.push(root.clone());
        }
        let image = &mut info.images[index];
        image.numeric_class = s.numeric_class;
        image.dimension = s.dimension;
        image.mip_count = s.mip_count;
        image.conversion_format = s.conversion_format;
        image.packed_format = s.packed_format;
        image.store_quantize = s.store_quantize;
        image.shader_swizzle = s.shader_swizzle;
        image.indirect_root = s.indirect_root;
        image.indirect_mapping_offset = s.indirect_mapping_offset;
        image.indirect_search_iterations = s.indirect_search_iterations;
        image.cube = s.cube;
        image.emulated_compare = s.emulated_compare;
        image.bindless = s.bindless;
        if s.emulated_compare != 0 {
            // The shader compares; the host keeps a colour view and a plain
            // sampler.
            image.depth_compare = false;
        }
        image.indirect_resources.clear();
    }
    for index in 0..info.images.len() {
        let root = info.images[index].indirect_root;
        if root != NO_INDEX {
            info.images[root as usize].indirect_resources.push(index as u32);
        }
    }
    let sampler_plan = build_sampler_plan(source, &specialization.images)
        .ok_or_else(|| ResourcePlanError::new(format!("shader resource specialization does not match the plan's samplers: hash=0x{:016X}", plan.hash)))?;
    for index in 0..source.samplers.len() {
        let target = sampler_plan.point_sampler[index];
        if target == NO_INDEX {
            continue;
        }
        if target as usize == index {
            info.samplers[index].force_point_filtering = true;
        } else {
            let mut sampler = source.samplers[index].clone();
            sampler.force_point_filtering = true;
            info.samplers.push(sampler);
        }
    }
    for pair in &mut info.sampled_pairs {
        let image = &info.images[pair.image as usize];
        if requires_point_sampler(image.numeric_class, image.conversion_format) {
            pair.sampler = sampler_plan.point_sampler[pair.sampler as usize];
        } else if image.bindless && !image.depth_compare {
            pair.point_sampler = Some(sampler_plan.point_sampler[pair.sampler as usize]).filter(|&p| p != NO_INDEX);
        }
    }
    // Vulkan bakes compareEnable into the sampler: split samplers shared by
    // ordinary and depth-reference sampling.
    let mut compare_usage = vec![0u8; info.samplers.len()];
    for pair in &info.sampled_pairs {
        compare_usage[pair.sampler as usize] |= if info.images[pair.image as usize].depth_compare { 2 } else { 1 };
    }
    let mut compare_sampler = vec![NO_INDEX; info.samplers.len()];
    for index in 0..info.samplers.len() {
        if compare_usage[index] & 2 == 0 {
            continue;
        }
        if compare_usage[index] & 1 == 0 {
            info.samplers[index].depth_compare = true;
            compare_sampler[index] = index as u32;
            continue;
        }
        let mut sampler = info.samplers[index].clone();
        sampler.depth_compare = true;
        compare_sampler[index] = info.samplers.len() as u32;
        info.samplers.push(sampler);
    }
    for pair in &mut info.sampled_pairs {
        if info.images[pair.image as usize].depth_compare {
            pair.sampler = compare_sampler[pair.sampler as usize];
        }
    }
    let mut sampler_by_memory_index = HashMap::new();
    for (index, memory) in plan.memory().entries().iter().enumerate() {
        if memory.kind != MemoryResourceKind::Image || memory.resource as usize >= info.images.len() {
            continue;
        }
        if !memory.needs_sampler || memory.sampler as usize >= source.samplers.len() {
            continue;
        }
        let image = &info.images[memory.resource as usize];
        let mut sampler = memory.sampler;
        if requires_point_sampler(image.numeric_class, image.conversion_format) {
            sampler = sampler_plan.point_sampler[sampler as usize];
        }
        if image.depth_compare {
            sampler = compare_sampler[sampler as usize];
        }
        if sampler != memory.sampler {
            sampler_by_memory_index.insert(index, sampler);
        }
    }
    Ok(SpecializedResourceInfo { info, sampler_by_memory_index })
}

#[cfg(test)]
mod emulated_compare_tests {
    use super::*;

    fn descriptor(format: u32, ty: u32, dst_sel_x: u32) -> Vec<u32> {
        vec![0, format << 20, 0, ty << 28 | dst_sel_x, 0, 0, 0, 0]
    }

    #[test]
    fn packed_unorm_sampling_shapes() {
        let mut info = ShaderResourceInfo::default();
        info.sampled_pairs.push(SampledImagePair { image: 0, sampler: 0, ..Default::default() });
        let bilinear_edge = vec![0x92, 0xFFF << 12, 1 << 20 | 1 << 22, 0];
        let t = descriptor(image_format::FORMAT_11X2X10_UNORM, image_format::TYPE_2D, 4);
        assert!(validate_packed_unorm_sampling(&info, &[bilinear_edge.as_slice().into()], 0, &t).is_ok());
        // Point filtering, wrap, anisotropy, a reduction, degamma: refused.
        for bad in [vec![0x92, 0, 0, 0], vec![0, 0, 1 << 20 | 1 << 22, 0], vec![0x92 | 1 << 9, 0, 1 << 20 | 1 << 22, 0], vec![0x92 | 1 << 29, 0, 1 << 20 | 1 << 22, 0], vec![0x92 | 1 << 20, 0, 1 << 20 | 1 << 22, 0]] {
            assert!(validate_packed_unorm_sampling(&info, &[bad.as_slice().into()], 0, &t).is_err(), "{bad:x?}");
        }
        // Mipped, or not 2D.
        let mut mipped = t.clone();
        mipped[3] |= 1 << 16;
        assert!(validate_packed_unorm_sampling(&info, &[bilinear_edge.as_slice().into()], 0, &mipped).is_err());
        let cube = descriptor(image_format::FORMAT_11X2X10_UNORM, image_format::TYPE_CUBE, 4);
        assert!(validate_packed_unorm_sampling(&info, &[bilinear_edge.as_slice().into()], 0, &cube).is_err());
    }

    /// Point copies of samplers are bound past the plan's 32 samplers: 32
    /// samplers each used with a float and a point-only (sint) image give
    /// 64 sampler bindings.
    #[test]
    fn point_sampler_copies_are_not_limited() {
        let image = |numeric_class| ImageSpecialization {
            numeric_class,
            dimension: ImageDimension::Dim2D,
            mip_count: 1,
            conversion_format: image_format::INVALID,
            shader_swizzle: crate::resources::IDENTITY_IMAGE_SWIZZLE,
            indirect_root: NO_INDEX,
            indirect_mapping_offset: 0,
            indirect_search_iterations: 0,
            cube: false,
            emulated_compare: 0,
            bindless: false,
            packed_format: 0,
            store_quantize: 0,
        };
        let images = [image(ImageNumericClass::Float), image(ImageNumericClass::Sint)];
        let mut info = ShaderResourceInfo::default();
        for s in 0..ShaderResourceInfo::MAX_SAMPLERS as u32 {
            info.samplers.push(SamplerResource { source: s, ..Default::default() });
            info.sampled_pairs.push(SampledImagePair { image: 0, sampler: s, ..Default::default() });
            info.sampled_pairs.push(SampledImagePair { image: 1, sampler: s, ..Default::default() });
        }
        let plan = build_sampler_plan(&info, &images).expect("sampler plan");
        assert_eq!(plan.sampler_count, 64);
        for s in 0..32 {
            assert_eq!(plan.point_sampler[s], 32 + s as u32);
        }
    }

    /// A sampler paired with an unsigned integer image gets a point copy
    /// as with a signed one (reference 193c54016).
    #[test]
    fn unsigned_integer_images_take_point_samplers() {
        let image = |numeric_class| ImageSpecialization {
            numeric_class,
            dimension: ImageDimension::Dim2D,
            mip_count: 1,
            conversion_format: image_format::INVALID,
            shader_swizzle: crate::resources::IDENTITY_IMAGE_SWIZZLE,
            indirect_root: NO_INDEX,
            indirect_mapping_offset: 0,
            indirect_search_iterations: 0,
            cube: false,
            emulated_compare: 0,
            bindless: false,
            packed_format: 0,
            store_quantize: 0,
        };
        let images = [image(ImageNumericClass::Float), image(ImageNumericClass::Uint)];
        let mut info = ShaderResourceInfo::default();
        info.samplers.push(SamplerResource { source: 0, ..Default::default() });
        info.sampled_pairs.push(SampledImagePair { image: 0, sampler: 0, ..Default::default() });
        info.sampled_pairs.push(SampledImagePair { image: 1, sampler: 0, ..Default::default() });
        let plan = build_sampler_plan(&info, &images).expect("sampler plan");
        assert_eq!(plan.sampler_count, 2);
        assert_eq!(plan.point_sampler[0], 1);
    }

    #[test]
    fn colour_compare_state() {
        use emulated_compare::*;
        let mut info = ShaderResourceInfo::default();
        info.sampled_pairs.push(SampledImagePair { image: 0, sampler: 0, ..Default::default() });
        // Clamp-to-edge both axes, LESSEQUAL, bilinear.
        let sampler = vec![2 | 2 << 3 | 3 << 12, 0, 1 << 20 | 1 << 22, 0];
        // 16_FLOAT (unified 13) 2D array, X = red.
        let s = emulated_compare_state(&info, &[sampler.as_slice().into()], 0, &descriptor(13, image_format::TYPE_2D_ARRAY, 4));
        assert_eq!(s & ENABLED, ENABLED);
        assert_eq!(function(s), 3);
        assert_eq!((address_x(s), address_y(s)), (ADDRESS_EDGE, ADDRESS_EDGE));
        assert_eq!(s & LINEAR, LINEAR);
        assert_eq!(reference(s), REFERENCE_FLOAT);
        // R16G16_UNORM clamps the reference to [0, 1]; every comparison of
        // R8_UNORM is 0 (measured on gfx1035).
        let s = emulated_compare_state(&info, &[sampler.as_slice().into()], 0, &descriptor(23, image_format::TYPE_2D, 4));
        assert_eq!(reference(s), REFERENCE_UNORM);
        let s = emulated_compare_state(&info, &[sampler.as_slice().into()], 0, &descriptor(1, image_format::TYPE_2D, 4));
        assert_eq!(reference(s), REFERENCE_ZERO);
        // 1D and 1D-array lines (16_UNORM, unified 7) are.
        for ty in [image_format::TYPE_1D, image_format::TYPE_1D_ARRAY] {
            let s = emulated_compare_state(&info, &[sampler.as_slice().into()], 0, &descriptor(7, ty, 4));
            assert_eq!((s & ENABLED, reference(s)), (ENABLED, REFERENCE_UNORM));
        }
        // A 3D view, or X not red, is not emulated.
        assert_eq!(emulated_compare_state(&info, &[sampler.as_slice().into()], 0, &descriptor(13, image_format::TYPE_3D, 4)), 0);
        assert_eq!(emulated_compare_state(&info, &[sampler.as_slice().into()], 0, &descriptor(13, image_format::TYPE_2D, 5)), 0);
    }
}

#[cfg(test)]
mod table_image_tests {
    use super::*;

    fn sampled_2d() -> ImageResource {
        ImageResource { resource_class: ImageResourceClass::Sampled, numeric_class: ImageNumericClass::Float, dimension: ImageDimension::Dim2D, read: true, ..ImageResource::new(0, 0) }
    }

    /// Material records that name no entry of a large table (here a null
    /// material V#: the program reads key zero through it) narrow nothing:
    /// the table is selected through the host table map instead of failing
    /// the dispatch.
    #[test]
    fn material_records_naming_no_entry_select_through_the_host_map() {
        let zero = |_: u64| Some(0u32);
        let mut inputs = ResourceRuntimeInputs::new(&[], 0);
        inputs.read_memory = Some(&zero);
        inputs.read_clean_memory = Some(&zero);
        let indirect = IndirectImageSelector::new(1, 0, 188, 0);
        let table = TableImageSelector { entry_offset: 0, has_material: true };
        let heap = [0x8000, 0, 64 * TABLE_ENTRY_BYTES as u32, 0];
        let r = materialize_table_image(&indirect, table, &heap, &[0; 4], &sampled_2d(), None, &inputs).expect("table selected");
        assert_eq!(r.bindless, Some((heap, 0, 64)));
        // Records of a real material V# whose keys all lie past the table
        // narrow nothing either.
        let past = |a: u64| Some(if a >= 0x9000 { 1000u32 } else { 0 });
        inputs.read_clean_memory = Some(&past);
        let r = materialize_table_image(&indirect, table, &heap, &[0x9000, 0, 188 * 4, 0], &sampled_2d(), None, &inputs).expect("table selected");
        assert_eq!(r.bindless, Some((heap, 0, 64)));
    }

    /// A material V# that does not describe the records the program reads
    /// (here null: the program reads key zero through it) cannot be
    /// enumerated; a heap the host table map can hold is selected through
    /// it instead of failing the dispatch.
    #[test]
    fn unenumerable_material_selects_through_the_host_map() {
        let bytes: Vec<u8> = 0xBF81_0000u32.to_le_bytes().to_vec();
        let program = crate::program::decode_blob(&bytes, 0x10_0000).unwrap();
        let plan = ShaderResourcePlan::extract(program, super::super::info::ShaderStage::Compute, 0x1234, 0, 4, None, 64).unwrap();
        let zero = |_: u64| Some(0u32);
        let mut inputs = ResourceRuntimeInputs::new(&[], 0);
        inputs.read_memory = Some(&zero);
        inputs.read_clean_memory = Some(&zero);
        let indirect = IndirectImageSelector::new(0, 1, 188, 0);
        let heap = [0x8000, 0, 64 * TABLE_ENTRY_BYTES as u32, 0];
        let r = materialize_indirect_image(&plan, &indirect, &[0; 4], &heap, &sampled_2d(), &inputs, false).expect("table selected");
        assert_eq!(r.bindless, Some((heap, 0, 64)));
        // An image the map cannot hold (written) still fails.
        let written = ImageResource { written: true, ..sampled_2d() };
        assert!(materialize_indirect_image(&plan, &indirect, &[0; 4], &heap, &written, &inputs, false).is_err());
    }
}
