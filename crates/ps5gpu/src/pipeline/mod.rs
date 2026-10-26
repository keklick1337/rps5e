// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The shader back-end's pipeline layer: everything between a resolved
//! draw (plus compiled SPIR-V) and recorded Vulkan commands.
//!
//! ```text
//!   ResolvedDraw ─▶ frontend (decode, plan, materialise, binding layout)
//!                 ─▶ ShaderCompiler (SPIR-V + reflection, cached per permutation)
//!                 ─▶ descriptors (V#/T#/S# → buffer cache, image cache, samplers)
//!                 ─▶ state (pipeline key + dynamic state) ─▶ VkPipeline (+ disk cache)
//!                 ─▶ recording (dynamic rendering, push descriptors, draw / dispatch)
//! ```
//!
//! * [`compiler`] — the [`ShaderCompiler`] trait the SPIR-V translator
//!   implements, and the binding contract between modules and the provider.
//! * [`frontend`] — program cache and per-draw resource materialisation.
//! * [`provider`] — [`ShaderPipelines`], the [`PipelineProvider`] installed
//!   into the GPU back-end.
//! * [`state`], [`primitives`], [`rectlist`] — fixed-function state, CPU
//!   index fallbacks, rectangle lists through tessellation.
//! * [`sharps`], [`descriptors`], [`cache`] — descriptor decoding, layout /
//!   sampler caches, the on-disk pipeline cache.
//! * [`testing`] — a test compiler of hand-built SPIR-V programs.
//!
//! [`PipelineProvider`]: crate::backend::PipelineProvider

pub mod bindless;
pub mod cache;
pub mod compiler;
pub mod descriptors;
pub mod frontend;
pub mod hazards;
pub mod indirect;
pub mod lookahead;
pub mod indirect_mesh;
pub mod indirect_threads;
pub mod primitives;
pub mod provider;
pub mod rectlist;
pub mod recording;
pub mod sharps;
pub mod skiplog;
pub mod state;
pub mod testing;
pub mod timing;
pub mod trace;
pub mod pool;
pub mod unlocked_wait;
pub mod portable;
pub mod precompile;
pub mod prewarm;
pub mod sources;
pub mod watch;
pub mod write_extent;
#[cfg(test)]
mod tests;

pub use compiler::{
    CompileError, CompiledShader, ComponentKind, ReflectedBinding, ReflectedVariable, ShaderCompileInput, ShaderCompiler, ShaderReflection,
};
pub use provider::{PrewarmProgress, PrewarmSummary, ProviderStats, ShaderPipelines, VertexAttribute, VertexBufferSource, VertexInputResolver, VertexInputs};

use crate::backend::PipelineProvider;
use crate::device::GpuDevice;
use crate::memory::GuestGpuMemory;
use std::sync::Arc;

/// The provider the back-end starts with: [`ShaderPipelines`] with the
/// SPIR-V compiler ([`crate::shader`]) unless `RPS5E_SHADER_COMPILER=0`;
/// then [`ShaderPipelines`] without a compiler when `RPS5E_GPU_PLAN=1`
/// (front-end diagnostics only), the counting stub otherwise.
pub fn default_provider(dev: &Arc<GpuDevice>, memory: &'static GuestGpuMemory) -> Box<dyn PipelineProvider> {
    if crate::shader::enabled() {
        // The provider that built the prewarm list before the game started.
        if let Some(p) = precompile::take_warm(dev) {
            log::info!("ps5gpu: SPIR-V shader compiler installed (prewarmed)");
            return Box::new(p);
        }
        let compiler: Arc<dyn ShaderCompiler> = Arc::new(crate::shader::SpirvCompiler::for_device(dev, memory));
        match ShaderPipelines::new(dev, memory, Some(compiler), cache::default_path(dev)) {
            Ok(p) => {
                log::info!("ps5gpu: SPIR-V shader compiler installed");
                return Box::new(p);
            }
            Err(e) => log::warn!("ps5gpu: pipeline provider: {e:?}"),
        }
    }
    if std::env::var("RPS5E_GPU_PLAN").is_ok_and(|v| v == "1") {
        match ShaderPipelines::new(dev, memory, None, cache::default_path(dev)) {
            Ok(p) => return Box::new(p),
            Err(e) => log::warn!("ps5gpu: pipeline provider: {e:?}"),
        }
    }
    Box::new(crate::backend::StubPipelines::default())
}

/// Install [`ShaderPipelines`] with `compiler` into the process-wide GPU
/// back-end. False when the back-end is unavailable.
pub fn install(compiler: Arc<dyn ShaderCompiler>) -> bool {
    crate::backend::with_backend(|b| match ShaderPipelines::new(&b.dev, b.memory, Some(compiler), cache::default_path(&b.dev)) {
        Ok(p) => {
            b.pipelines = Box::new(p);
            true
        }
        Err(e) => {
            log::warn!("ps5gpu: pipeline provider: {e:?}");
            false
        }
    })
    .unwrap_or(false)
}
