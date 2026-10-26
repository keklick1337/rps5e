// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Reinterpretation of a single-sample colour image as a multisampled
//! depth image.
//!
//! A multisampled depth surface whose memory a single-sample colour image
//! holds (each element carrying one depth value per sample in its
//! components) is rebuilt by drawing a full-screen triangle into the depth
//! image: the fragment shader runs per sample and writes, as its depth, the
//! colour component whose index is the sample's.

use crate::device::GpuDevice;
use ash::vk;
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv;
use std::sync::Arc;

/// Layout of the destination while it is drawn.
pub const DESTINATION_LAYOUT: vk::ImageLayout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

type R<T> = Result<T, rspirv::dr::Error>;

/// Vertex shader: a triangle covering the target, `uv` 0..1 across it.
///
/// ```text
/// position = vec2((vertex & 1) << 2, (vertex & 2) << 1)
/// gl_Position = vec4(position - 1, 0, 1); uv = position * 0.5
/// ```
pub fn vertex_spirv() -> Vec<u32> {
    let build = || -> R<Vec<u32>> {
        use spirv::{Decoration as D, StorageClass as S};
        let mut b = Builder::new();
        b.set_version(1, 3);
        b.capability(spirv::Capability::Shader);
        b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let float = b.type_float(32, None);
        let v2 = b.type_vector(float, 2);
        let v4 = b.type_vector(float, 4);
        let p_in_int = b.type_pointer(None, S::Input, int);
        let p_out_v2 = b.type_pointer(None, S::Output, v2);
        let p_out_v4 = b.type_pointer(None, S::Output, v4);
        let vertex = b.variable(p_in_int, None, S::Input, None);
        b.decorate(vertex, D::BuiltIn, [Operand::BuiltIn(spirv::BuiltIn::VertexIndex)]);
        let position = b.variable(p_out_v4, None, S::Output, None);
        b.decorate(position, D::BuiltIn, [Operand::BuiltIn(spirv::BuiltIn::Position)]);
        let uv = b.variable(p_out_v2, None, S::Output, None);
        b.decorate(uv, D::Location, [Operand::LiteralBit32(0)]);
        let one_i = b.constant_bit32(int, 1);
        let two_i = b.constant_bit32(int, 2);
        let zero_f = b.constant_bit32(float, 0f32.to_bits());
        let one_f = b.constant_bit32(float, 1f32.to_bits());
        let half_f = b.constant_bit32(float, 0.5f32.to_bits());
        let fnty = b.type_function(void, []);
        let main = b.begin_function(void, None, spirv::FunctionControl::NONE, fnty)?;
        b.name(main, "main");
        b.begin_block(None)?;
        let vi = b.load(int, None, vertex, None, [])?;
        let x = b.bitwise_and(int, None, vi, one_i)?;
        let x = b.shift_left_logical(int, None, x, two_i)?;
        let y = b.bitwise_and(int, None, vi, two_i)?;
        let y = b.shift_left_logical(int, None, y, one_i)?;
        let fx = b.convert_s_to_f(float, None, x)?;
        let fy = b.convert_s_to_f(float, None, y)?;
        let px = b.f_sub(float, None, fx, one_f)?;
        let py = b.f_sub(float, None, fy, one_f)?;
        let pos = b.composite_construct(v4, None, [px, py, zero_f, one_f])?;
        b.store(position, pos, None, [])?;
        let ux = b.f_mul(float, None, fx, half_f)?;
        let uy = b.f_mul(float, None, fy, half_f)?;
        let coord = b.composite_construct(v2, None, [ux, uy])?;
        b.store(uv, coord, None, [])?;
        b.ret()?;
        b.end_function()?;
        b.entry_point(spirv::ExecutionModel::Vertex, main, "main", [vertex, position, uv]);
        Ok(b.module().assemble())
    };
    build().expect("blit vertex shader assembles")
}

/// Fragment shader: depth of the sample = component `gl_SampleID` of the
/// colour texel under it.
///
/// ```text
/// coord = ivec2(uv * vec2(textureSize(color, 0)))
/// gl_FragDepth = texelFetch(color, coord, 0)[gl_SampleID]
/// ```
pub fn fragment_spirv() -> Vec<u32> {
    let build = || -> R<Vec<u32>> {
        use spirv::{Decoration as D, StorageClass as S};
        let mut b = Builder::new();
        b.set_version(1, 3);
        b.capability(spirv::Capability::Shader);
        b.capability(spirv::Capability::ImageQuery);
        b.capability(spirv::Capability::SampleRateShading);
        b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let float = b.type_float(32, None);
        let v2 = b.type_vector(float, 2);
        let v4 = b.type_vector(float, 4);
        let iv2 = b.type_vector(int, 2);
        let image = b.type_image(float, spirv::Dim::Dim2D, 0, 0, 0, 1, spirv::ImageFormat::Unknown, None);
        let p_image = b.type_pointer(None, S::UniformConstant, image);
        let p_in_v2 = b.type_pointer(None, S::Input, v2);
        let p_in_int = b.type_pointer(None, S::Input, int);
        let p_out_float = b.type_pointer(None, S::Output, float);
        let color = b.variable(p_image, None, S::UniformConstant, None);
        b.decorate(color, D::DescriptorSet, [Operand::LiteralBit32(0)]);
        b.decorate(color, D::Binding, [Operand::LiteralBit32(0)]);
        let uv = b.variable(p_in_v2, None, S::Input, None);
        b.decorate(uv, D::Location, [Operand::LiteralBit32(0)]);
        let sample = b.variable(p_in_int, None, S::Input, None);
        b.decorate(sample, D::BuiltIn, [Operand::BuiltIn(spirv::BuiltIn::SampleId)]);
        b.decorate(sample, D::Flat, []);
        let depth = b.variable(p_out_float, None, S::Output, None);
        b.decorate(depth, D::BuiltIn, [Operand::BuiltIn(spirv::BuiltIn::FragDepth)]);
        let zero_i = b.constant_bit32(int, 0);
        let fnty = b.type_function(void, []);
        let main = b.begin_function(void, None, spirv::FunctionControl::NONE, fnty)?;
        b.name(main, "main");
        b.begin_block(None)?;
        let img = b.load(image, None, color, None, [])?;
        let size = b.image_query_size_lod(iv2, None, img, zero_i)?;
        let fsize = b.convert_s_to_f(v2, None, size)?;
        let at = b.load(v2, None, uv, None, [])?;
        let scaled = b.f_mul(v2, None, at, fsize)?;
        let coord = b.convert_f_to_s(iv2, None, scaled)?;
        let texel = b.image_fetch(v4, None, img, coord, Some(spirv::ImageOperands::LOD), [Operand::IdRef(zero_i)])?;
        let s = b.load(int, None, sample, None, [])?;
        let value = b.vector_extract_dynamic(float, None, texel, s)?;
        b.store(depth, value, None, [])?;
        b.ret()?;
        b.end_function()?;
        b.entry_point(spirv::ExecutionModel::Fragment, main, "main", [uv, sample, depth]);
        b.execution_mode(main, spirv::ExecutionMode::OriginUpperLeft, []);
        b.execution_mode(main, spirv::ExecutionMode::DepthReplacing, []);
        Ok(b.module().assemble())
    };
    build().expect("blit fragment shader assembles")
}

/// The pipelines of the colour → multisampled depth reinterpretation.
pub struct BlitHelper {
    dev: Arc<GpuDevice>,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
    /// Pipelines by (sample count, depth format).
    pipelines: Vec<((u32, vk::Format), vk::Pipeline)>,
}

// SAFETY: plain handles, used by the owning thread.
unsafe impl Send for BlitHelper {}

impl BlitHelper {
    /// None when the device cannot run per-sample fragment shaders.
    pub fn new(dev: &Arc<GpuDevice>) -> Option<BlitHelper> {
        if !dev.caps.sample_rate_shading {
            return None;
        }
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let (vs, fs) = (vertex_spirv(), fragment_spirv());
        // SAFETY: valid create infos; destroyed in Drop.
        unsafe {
            let d = &dev.device;
            let set_layout = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                        .bindings(std::slice::from_ref(&binding)),
                    None,
                )
                .ok()?;
            let layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set_layout)), None).ok()?;
            let vertex = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vs), None).ok()?;
            let fragment = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&fs), None).ok()?;
            Some(BlitHelper { dev: dev.clone(), set_layout, layout, vertex, fragment, pipelines: Vec::new() })
        }
    }

    fn pipeline(&mut self, samples: u32, format: vk::Format) -> Option<vk::Pipeline> {
        if let Some((_, p)) = self.pipelines.iter().find(|(k, _)| *k == (samples, format)) {
            return Some(*p);
        }
        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(self.vertex).name(c"main"),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(self.fragment).name(c"main"),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default().line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(super::samples_flag(samples));
        let depth = vk::PipelineDepthStencilStateCreateInfo::default().depth_test_enable(true).depth_write_enable(true).depth_compare_op(vk::CompareOp::ALWAYS);
        let blend = vk::PipelineColorBlendStateCreateInfo::default();
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let mut rendering = vk::PipelineRenderingCreateInfo::default().depth_attachment_format(format);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .push_next(&mut rendering)
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(self.layout);
        // SAFETY: valid create info; the pipeline is destroyed in Drop.
        let pipeline = match unsafe { self.dev.device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) } {
            Ok(p) => p[0],
            Err((_, e)) => {
                log::error!("ps5gpu: colour to multisampled depth pipeline ({samples} samples, {format:?}): {e:?}");
                return None;
            }
        };
        self.pipelines.push(((samples, format), pipeline));
        Some(pipeline)
    }

    /// Record the reinterpretation of `source` (a sampled view of the
    /// colour image, in SHADER_READ_ONLY layout) into `destination` (a
    /// depth view of the multisampled image, in [`DESTINATION_LAYOUT`]),
    /// both `width` x `height`. False when no pipeline exists for the
    /// destination.
    #[allow(clippy::too_many_arguments)]
    pub fn record(&mut self, cb: vk::CommandBuffer, source: vk::ImageView, destination: vk::ImageView, width: u32, height: u32, samples: u32, format: vk::Format) -> bool {
        let Some(pipeline) = self.pipeline(samples, format) else { return false };
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(destination)
            .image_layout(DESTINATION_LAYOUT)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 } });
        let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width, height } };
        let rendering = vk::RenderingInfo::default().render_area(area).layer_count(1).depth_attachment(&attachment);
        let image = vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: source, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL };
        let write = vk::WriteDescriptorSet::default().dst_binding(0).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(std::slice::from_ref(&image));
        let viewport = vk::Viewport { x: 0.0, y: 0.0, width: width as f32, height: height as f32, min_depth: 0.0, max_depth: 1.0 };
        crate::scheduler::disturb_state();
        // SAFETY: recording buffer outside any rendering scope; the views
        // and the pipeline match the rendering info.
        unsafe {
            let d = &self.dev.device;
            d.cmd_begin_rendering(cb, &rendering);
            self.dev.push_descriptor.cmd_push_descriptor_set(cb, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, std::slice::from_ref(&write));
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
            d.cmd_set_viewport(cb, 0, &[viewport]);
            d.cmd_set_scissor(cb, 0, &[area]);
            d.cmd_draw(cb, 3, 1, 0, 0);
            d.cmd_end_rendering(cb);
        }
        true
    }
}

impl Drop for BlitHelper {
    fn drop(&mut self) {
        // SAFETY: the owner waits for the device before dropping.
        unsafe {
            let d = &self.dev.device;
            for (_, p) in self.pipelines.drain(..) {
                d.destroy_pipeline(p, None);
            }
            d.destroy_shader_module(self.vertex, None);
            d.destroy_shader_module(self.fragment, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// True when a single-sample colour image of `source` shape can be
/// reinterpreted into the multisampled depth image of `destination`
/// shape: same surface and extent, 2 to 4 samples of a depth-only format,
/// one level and layer, a float colour format.
pub fn can_reinterpret(source: &super::ImageKey, destination: &super::ImageKey) -> bool {
    use vk::Format as F;
    let depth_only = matches!(destination.format, F::D16_UNORM | F::D32_SFLOAT | F::X8_D24_UNORM_PACK32);
    let float_color = !source.is_depth()
        && !matches!(
            source.format,
            F::R8_UINT | F::R8_SINT | F::R16_UINT | F::R16_SINT | F::R32_UINT | F::R32_SINT | F::R8G8_UINT | F::R8G8_SINT | F::R16G16_UINT | F::R16G16_SINT
                | F::R32G32_UINT | F::R32G32_SINT | F::R8G8B8A8_UINT | F::R8G8B8A8_SINT | F::R16G16B16A16_UINT | F::R16G16B16A16_SINT
                | F::R32G32B32A32_UINT | F::R32G32B32A32_SINT | F::A2B10G10R10_UINT_PACK32
        );
    float_color
        && depth_only
        && destination.stencil.is_none()
        && source.samples == 1
        && (2..=4).contains(&destination.samples)
        && source.dim == super::ImageDim::D2
        && destination.dim == super::ImageDim::D2
        && source.width == destination.width
        && source.height == destination.height
        && source.main.addr == destination.main.addr
        && destination.levels == 1
        && destination.layers == 1
        && super::block_dim(source.format) == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parses(words: &[u32]) {
        assert_eq!(words[0], rspirv::spirv::MAGIC_NUMBER);
        let mut loader = rspirv::dr::Loader::new();
        rspirv::binary::parse_words(words, &mut loader).unwrap();
        let module = loader.module();
        assert_eq!(module.entry_points.len(), 1);
    }

    #[test]
    fn shaders_assemble() {
        parses(&vertex_spirv());
        parses(&fragment_spirv());
        // RPS5E_DUMP_SPIRV=DIR writes the fixed shaders out for spirv-val.
        if let Some(dir) = std::env::var_os("RPS5E_DUMP_SPIRV") {
            let dir = std::path::PathBuf::from(dir);
            let write = |name: &str, words: &[u32]| std::fs::write(dir.join(name), words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
            write("blit.vert.spv", &vertex_spirv());
            write("blit.frag.spv", &fragment_spirv());
            for t in [crate::tiler::ColorTransform::SwapBgra16, crate::tiler::ColorTransform::Reverse10_11_11] {
                for to_host in [false, true] {
                    write(&format!("transform-{t:?}-{to_host}.spv"), &crate::tiler::transform_spirv(t, to_host).unwrap());
                }
            }
        }
    }

    #[test]
    fn reinterpretation_shapes() {
        use crate::images::{ImageDim, ImageKey, Plane};
        let plane = |aspect| Plane { addr: 0x10_0000, bytes_per_element: 16, swizzle: 0, pitch: 64, slice_size: 0, bytes: 0, aspect };
        let color = ImageKey {
            format: vk::Format::R32G32B32A32_SFLOAT,
            width: 64,
            height: 32,
            depth: 1,
            layers: 1,
            levels: 1,
            samples: 1,
            dim: ImageDim::D2,
            main: plane(vk::ImageAspectFlags::COLOR),
            stencil: None,
            transform: Default::default(),
        };
        let depth = ImageKey { format: vk::Format::D32_SFLOAT, samples: 4, main: plane(vk::ImageAspectFlags::DEPTH), ..color };
        assert!(can_reinterpret(&color, &depth));
        assert!(!can_reinterpret(&color, &ImageKey { samples: 8, ..depth }));
        assert!(!can_reinterpret(&color, &ImageKey { format: vk::Format::D32_SFLOAT_S8_UINT, ..depth }));
        assert!(!can_reinterpret(&ImageKey { format: vk::Format::R32G32B32A32_UINT, ..color }, &depth));
        assert!(!can_reinterpret(&ImageKey { width: 32, ..color }, &depth));
        assert!(!can_reinterpret(&ImageKey { samples: 2, ..color }, &depth));
    }
}
