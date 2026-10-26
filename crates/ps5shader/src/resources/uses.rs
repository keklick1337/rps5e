// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! A flat, IR-independent view of a plan: one [`ResourceUse`] per memory
//! instruction operand, keyed by `(pc, component, operand)`, naming the
//! binding it resolved to and how its descriptor is obtained.
//!
//! The key is what a later SSA pass needs to attach the plan to its own
//! instructions; the descriptor source is an evaluable expression
//! ([`DescriptorSource::dwords`] in the plan's value graph) whose leaves are
//! classified by [`SourceCategory`].

use super::info::NO_INDEX;
use super::memory::{BufferDescriptorProvenance, FlatAddressSpace, MemoryResourceKind};
use super::plan::ShaderResourcePlan;
use super::value::{ValueId, ValueKind};
use std::collections::{BTreeMap, HashSet};
use std::fmt;

/// Which descriptor operand of the instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ResourceOperand {
    /// V# (MUBUF/MTBUF/S_BUFFER_LOAD), T# (MIMG) or address pair.
    Resource,
    /// S# (sampled MIMG).
    Sampler,
}

/// Where an operand's descriptor lives at run time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ResourceBinding {
    /// Dense buffer binding `i` ([`ShaderResourceInfo::buffers`](super::ShaderResourceInfo)).
    Buffer(u32),
    /// Dense image binding `i` (a run-time selected image is its root).
    Image(u32),
    Sampler(u32),
    /// A scalar load the host performs into flattened-table slot `i`.
    FlattenedSlot(u32),
    /// One of the bounded runtime V# candidate tables.
    CandidateTable(usize),
    /// Address-based fallback: the shader reads through a device address
    /// (global/flat access, raw scalar load, or a V# held only in its
    /// registers) via the device-address page table.
    DeviceAddress,
    /// LDS, GDS or scratch: no descriptor.
    Local,
    /// Not planned (e.g. the unreachable tail after a fused S_SETPC).
    Unplanned,
    /// A descriptor read the planner resolved itself (the T# dwords of a
    /// run-time selected image): the words feed only that image's handle,
    /// whose binding stands for them, so the load emits nothing real.
    Suppressed,
}

/// How a descriptor (or address) is obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SourceCategory {
    /// Compile-time constant.
    Constant,
    /// User-data SGPRs (and constants / shader base) only.
    UserData,
    /// Through the shader resource table walk: scalar loads at constant
    /// offsets from user-data pointers, evaluated by the host per draw.
    SrtWalk,
    /// Selected at run time from a host-enumerated candidate set (bindless
    /// table, material/heap pair, finite S_FF1 set, bounded V# table).
    Indirect,
    /// Not host-evaluable: the device resolves it through addresses.
    AddressBased,
    /// No descriptor (LDS/GDS/scratch) or unplanned.
    None,
}

impl fmt::Display for SourceCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SourceCategory::Constant => "constant",
            SourceCategory::UserData => "user-data",
            SourceCategory::SrtWalk => "srt-walk",
            SourceCategory::Indirect => "indirect",
            SourceCategory::AddressBased => "address-based",
            SourceCategory::None => "none",
        };
        f.write_str(s)
    }
}

/// One descriptor operand of one memory instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceUse {
    pub pc: u32,
    /// Dword component (scalar loads have one record per dword).
    pub component: u32,
    pub operand: ResourceOperand,
    pub memory_index: usize,
    pub binding: ResourceBinding,
    pub category: SourceCategory,
    /// Descriptor source index, when the binding has one.
    pub source: Option<u32>,
}

impl ShaderResourcePlan {
    /// Classify the leaves of a descriptor-source expression.
    pub fn classify_values(&self, dwords: &[ValueId]) -> SourceCategory {
        let mut pending: Vec<ValueId> = dwords.to_vec();
        let mut visited = HashSet::new();
        let mut category = SourceCategory::Constant;
        while let Some(v) = pending.pop() {
            if !visited.insert(v) {
                continue;
            }
            let node = self.graph.node(v);
            match node.kind {
                ValueKind::Constant | ValueKind::MemoryAperture => {}
                ValueKind::UserData | ValueKind::ShaderBase => category = category.max(SourceCategory::UserData),
                ValueKind::ResourceTableWord => {
                    category = category.max(SourceCategory::SrtWalk);
                    if let Some(r) = self.table_reads.get(node.payload as usize) {
                        pending.push(r.value);
                    }
                }
                ValueKind::ScalarAddressWord | ValueKind::ScalarBufferWord => {
                    category = category.max(SourceCategory::SrtWalk);
                    pending.extend(node.operands.iter().copied());
                }
                ValueKind::Undefined => return SourceCategory::AddressBased,
                ValueKind::Phi => match self.graph.resolve_invariant_phi(v) {
                    Some(i) => pending.push(i),
                    None => return SourceCategory::AddressBased,
                },
                _ => pending.extend(node.operands.iter().copied()),
            }
        }
        category
    }

    fn source_category(&self, source: u32) -> SourceCategory {
        let s = &self.descriptor_sources[source as usize];
        if s.indirect_image.is_some() {
            SourceCategory::Indirect
        } else {
            self.classify_values(&s.dwords)
        }
    }

    /// Every descriptor operand of every memory instruction, in memory-table
    /// order.
    pub fn resource_uses(&self) -> Vec<ResourceUse> {
        let mut uses = Vec::new();
        for (index, memory) in self.memory().entries().iter().enumerate() {
            let mut push = |operand, binding, category, source| {
                uses.push(ResourceUse { pc: memory.pc, component: memory.component_index, operand, memory_index: index, binding, category, source })
            };
            use MemoryResourceKind as K;
            match memory.kind {
                K::LocalDataShare | K::GlobalDataShare | K::Scratch => {
                    push(ResourceOperand::Resource, ResourceBinding::Local, SourceCategory::None, None)
                }
                K::None if memory.opcode.is_ray_intersect() => {
                    push(ResourceOperand::Resource, ResourceBinding::DeviceAddress, SourceCategory::AddressBased, None)
                }
                K::ScalarAddress | K::ScalarBuffer if memory.planning_only => {
                    let slot = self.flattened_slot_by_memory_index.get(&index).copied();
                    match slot {
                        Some(slot) => push(ResourceOperand::Resource, ResourceBinding::FlattenedSlot(slot), SourceCategory::SrtWalk, None),
                        None if self.suppressed_memory.contains(&index) => {
                            push(ResourceOperand::Resource, ResourceBinding::Suppressed, SourceCategory::None, None)
                        }
                        None => push(ResourceOperand::Resource, ResourceBinding::Unplanned, SourceCategory::None, None),
                    }
                }
                K::Flat | K::Global | K::ScalarAddress => {
                    let binding = if memory.kind == K::Flat && memory.address_space != FlatAddressSpace::Global {
                        ResourceBinding::Local
                    } else {
                        ResourceBinding::DeviceAddress
                    };
                    let category = if binding == ResourceBinding::Local { SourceCategory::None } else { SourceCategory::AddressBased };
                    push(ResourceOperand::Resource, binding, category, None)
                }
                K::Buffer | K::ScalarBuffer => {
                    if let Some(table) = self.buffer_candidate_tables.iter().position(|t| t.memory_indices.contains(&index)) {
                        push(ResourceOperand::Resource, ResourceBinding::CandidateTable(table), SourceCategory::Indirect, None);
                    } else if memory.device_descriptor
                        || memory.buffer_descriptor.is_some_and(|d| d.provenance == BufferDescriptorProvenance::Runtime)
                    {
                        push(ResourceOperand::Resource, ResourceBinding::DeviceAddress, SourceCategory::AddressBased, None);
                    } else if memory.resource != NO_INDEX && (memory.resource as usize) < self.info.buffers.len() {
                        let source = self.info.buffers[memory.resource as usize].source;
                        push(ResourceOperand::Resource, ResourceBinding::Buffer(memory.resource), self.source_category(source), Some(source));
                    } else {
                        push(ResourceOperand::Resource, ResourceBinding::Unplanned, SourceCategory::None, None);
                    }
                }
                K::Image => {
                    if memory.planning_only || memory.resource == NO_INDEX || memory.resource as usize >= self.info.images.len() {
                        push(ResourceOperand::Resource, ResourceBinding::Unplanned, SourceCategory::None, None);
                        continue;
                    }
                    let source = self.info.images[memory.resource as usize].source;
                    push(ResourceOperand::Resource, ResourceBinding::Image(memory.resource), self.source_category(source), Some(source));
                    if memory.needs_sampler && (memory.sampler as usize) < self.info.samplers.len() {
                        let source = self.info.samplers[memory.sampler as usize].source;
                        push(ResourceOperand::Sampler, ResourceBinding::Sampler(memory.sampler), self.source_category(source), Some(source));
                    }
                }
                _ => {}
            }
        }
        uses
    }

    /// Operand counts per category.
    pub fn category_counts(&self) -> BTreeMap<SourceCategory, usize> {
        let mut counts = BTreeMap::new();
        for u in self.resource_uses() {
            *counts.entry(u.category).or_insert(0) += 1;
        }
        counts
    }
}
