// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Shader resource planning.
//!
//! Finds, for every memory instruction of a program, where its descriptor
//! (V# buffer, T# image, S# sampler, or 64-bit address) comes from, and turns
//! that into dense binding tables plus a per-draw materialisation step.
//!
//! # Pipeline
//!
//! 1. [`ScalarValueGraph::build`] walks the control-flow graph to a fixpoint
//!    and gives every SGPR (and uniform VGPR) value a node: user data at
//!    entry, the shader base, constants, scalar loads through address /
//!    buffer handles, arithmetic, phis (selects at uniform if/else joins),
//!    first-lane reads. Values are interned and constant-folded.
//! 2. [`ShaderResourcePlan::extract`] flattens host-evaluable immediate
//!    scalar reads into a *flattened resource table* the host fills per
//!    draw, interns descriptor sources, builds the dense tables
//!    ([`ShaderResourceInfo`]: buffers, images, samplers, sampled pairs),
//!    recognises run-time selected images (material/heap tables, dense key
//!    tables, finite S_FF1 candidate sets) and bounded runtime V# tables,
//!    and plans resource branches and device-address ranges. A plan is
//!    independent of any draw's user data.
//! 3. [`BindingLayout::allocate`] assigns descriptor bindings and the push
//!    data layout (live user-data registers, shader base, buffer offsets).
//! 4. Per draw, [`materialize`] evaluates the plan against the draw's
//!    user-data SGPRs, shader base and guest memory
//!    ([`ResourceRuntimeInputs`]): concrete descriptor words per binding
//!    ([`ResourceSnapshot`]) plus the module-affecting
//!    [`ResourceSpecialization`] (image classes and dimensions, buffer
//!    strides and formats, indirect candidates). [`apply_specialization`]
//!    turns a specialization into the [`SpecializedResourceInfo`] an emitter
//!    compiles against. [`ResourceMaterializationCache`] runs the plan's
//!    compiled evaluation ([`ReplayProgram`]) and reuses the specialization
//!    of the last draw with the same decision-affecting descriptor fields,
//!    patching only addresses, sizes and table words; plans it cannot
//!    compile skip the walk while every guest word it read is unchanged.
//!
//! [`ShaderResourcePlan::resource_uses`] gives an IR-independent view: one
//! [`ResourceUse`] per descriptor operand keyed by `(pc, component, operand)`,
//! with its binding and a [`SourceCategory`] (constant, user data, SRT walk,
//! indirect, or the address-based fallback through the device-address page
//! table).
//!
//! A [`ShaderCompileRequest`] bundles the program, the specialised tables and
//! the binding layout for a back-end.
//!
//! Not ported: the scalar-graph disk cache and the raw-read prefetcher of the
//! reference design (performance only) and the materialisation profiler.

mod binding;
mod branches;
mod builder;
mod cache;
mod candidates;
mod device_ranges;
mod evaluator;
pub mod format;
mod graph;
mod info;
mod materialize;
mod memory;
pub mod metadata;
mod pixel_inputs;
mod plan;
mod replay;
mod request;
mod selector;
mod table_reads;
mod tracker;
mod uses;
mod validator;
mod value;
pub mod vertex_fetch;

pub use binding::*;
pub use branches::ResourceBranchBlock;
pub use cache::{CacheStatistics, ResourceMaterializationCache};
pub use candidates::BufferCandidateTablePlan;
pub use device_ranges::{paging, DeviceAddressRange, DeviceAddressRangePlan, MAX_RANGE_BYTES};
pub use evaluator::{
    evaluate_descriptor_source, evaluate_sources, flatten_resource_table, ComputeSelectorState, EvaluatedSources, ResidentReader,
    ResourceRuntimeInputs, RuntimeValueEvaluator, WordReader, ADDRESS_MASK,
};
pub use format::ImageNumericClass;
pub use graph::{MemoryAccessBinding, ScalarValueGraph};
pub use info::*;
pub use materialize::*;
pub use memory::*;
pub use pixel_inputs::resolve_pixel_input_locations;
pub use replay::{ReplayAbort, ReplayProgram, ReplayReader, ReplayResult, ReplayScratch};
pub use plan::{IndirectImageAccess, PlanOptions, ShaderResourcePlan, WRITTEN_RANGE_DWORD_COUNT};
pub use request::*;
pub use selector::{IndirectSelectorDiagnostic, IndirectSelectorValues};
pub use uses::{ResourceBinding, ResourceOperand, ResourceUse, SourceCategory};
pub use validator::RuntimeValueValidator;
pub use value::{evaluate_operation, Node, ScalarOp, ValueId, ValueKind, ValueType};

use std::fmt;

/// A program whose resources cannot be planned. The message names the
/// shader hash, the stage and the pc where known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourcePlanError {
    pub message: String,
}

impl ResourcePlanError {
    pub fn new(message: impl Into<String>) -> Self {
        ResourcePlanError { message: message.into() }
    }
}

impl fmt::Display for ResourcePlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ResourcePlanError {}

#[cfg(test)]
mod tests;
