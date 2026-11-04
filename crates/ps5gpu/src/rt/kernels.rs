// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The compute kernels of the BVH mirrors. Sources are the GLSL files in
//! `kernels/`; the SPIR-V next to each is what `glslc --target-env=vulkan1.3
//! -O` makes of it (the test `kernels_match_their_sources` rebuilds them
//! when glslc is installed). Every kernel has storage-buffer bindings set
//! with push descriptors (binding 0 is the device-address page table of
//! guest memory) and its parameters in push constants.

use crate::device::GpuDevice;
use ash::vk;
use std::sync::Arc;

pub const GATHER: &[u8] = include_bytes!("kernels/gather.spv");
pub const INSTANCES: &[u8] = include_bytes!("kernels/instances.spv");
pub const TRIANGLES: &[u8] = include_bytes!("kernels/triangles.spv");

/// Push-constant bytes every kernel may use.
pub const PUSH_BYTES: u32 = 128;

pub struct Kernel {
    dev: Arc<GpuDevice>,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    bindings: u32,
}

fn words(spv: &[u8]) -> Vec<u32> {
    spv.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

impl Kernel {
    pub fn new(dev: &Arc<GpuDevice>, spv: &[u8], bindings: u32, name: &str) -> Option<Kernel> {
        let code = words(spv);
        let list: Vec<vk::DescriptorSetLayoutBinding> = (0..bindings)
            .map(|i| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        // SAFETY: valid create infos; destroyed in Drop (or here on failure).
        unsafe {
            let d = &dev.device;
            let set_layout = d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR).bindings(&list), None).ok()?;
            let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: PUSH_BYTES }];
            let layout = match d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set_layout)).push_constant_ranges(&ranges), None) {
                Ok(l) => l,
                Err(_) => {
                    d.destroy_descriptor_set_layout(set_layout, None);
                    return None;
                }
            };
            let pipeline = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).ok().and_then(|module| {
                let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
                let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
                let r = d.create_compute_pipelines(vk::PipelineCache::null(), &[info], None);
                d.destroy_shader_module(module, None);
                r.map_err(|(_, e)| log::warn!("ps5gpu: hw-rt kernel {name}: {e:?}")).ok().map(|p| p[0])
            });
            let Some(pipeline) = pipeline else {
                d.destroy_pipeline_layout(layout, None);
                d.destroy_descriptor_set_layout(set_layout, None);
                return None;
            };
            Some(Kernel { dev: dev.clone(), set_layout, layout, pipeline, bindings })
        }
    }

    /// Record one dispatch of `groups` with `buffers` on bindings 0.. and
    /// `push` (at most [`PUSH_BYTES`]) as push constants.
    pub fn record(&self, cb: vk::CommandBuffer, buffers: &[vk::DescriptorBufferInfo], push: &[u8], groups: [u32; 3]) {
        debug_assert_eq!(buffers.len() as u32, self.bindings);
        let writes: Vec<vk::WriteDescriptorSet> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| vk::WriteDescriptorSet::default().dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(b)))
            .collect();
        let mut bytes = [0u8; PUSH_BYTES as usize];
        bytes[..push.len()].copy_from_slice(push);
        crate::scheduler::disturb_state();
        // SAFETY: recording command buffer outside a rendering scope; all handles valid.
        unsafe {
            let d = &self.dev.device;
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.dev.push_descriptor.cmd_push_descriptor_set(cb, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &writes);
            d.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
            if groups.iter().all(|&g| g > 0) {
                d.cmd_dispatch(cb, groups[0], groups[1], groups[2]);
            }
        }
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: the owner waits for the device before dropping.
        unsafe {
            self.dev.device.destroy_pipeline(self.pipeline, None);
            self.dev.device.destroy_pipeline_layout(self.layout, None);
            self.dev.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// Little-endian push-constant bytes.
#[derive(Default)]
pub struct Push(pub Vec<u8>);

impl Push {
    pub fn u64(mut self, v: u64) -> Self {
        while self.0.len() % 8 != 0 {
            self.0.push(0);
        }
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn u32(mut self, v: u32) -> Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn f32(self, v: f32) -> Self {
        self.u32(v.to_bits())
    }
}

/// The page-table words every kernel reads guest memory with.
pub fn paging() -> (u32, u32) {
    (ps5shader::resources::paging::PAGE_BITS, (crate::buffers::BDA_PAGE_COUNT - 1) as u32)
}

#[cfg(test)]
mod tests {
    #[test]
    fn kernels_match_their_sources() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/rt/kernels");
        for (name, spv) in [("gather", super::GATHER), ("instances", super::INSTANCES), ("triangles", super::TRIANGLES)] {
            assert_eq!(&spv[..4], &0x0723_0203u32.to_le_bytes(), "{name}: not SPIR-V");
            let out = std::process::Command::new("glslc").args(["--target-env=vulkan1.3", "-O"]).arg(dir.join(format!("{name}.comp"))).args(["-o", "-"]).output();
            let Ok(out) = out else { return };
            assert!(out.status.success(), "{name}: glslc failed");
            assert!(out.stdout == spv, "{name}.spv is not the build of {name}.comp (rebuild it with glslc --target-env=vulkan1.3 -O)");
        }
    }
}
