// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Vulkan renderer of egui output for the game window (render thread).
//!
//! Textures are R8G8B8A8_UNORM images (egui's sRGB-encoded, premultiplied
//! texels), uploaded from a per-frame-slot staging buffer; the meshes go
//! into per-slot host-visible vertex and index buffers. Freed textures are
//! destroyed once the frame slots that may still read them have been
//! waited for. Drawing happens inside the swapchain image's colour
//! attachment pass after the game image and the performance overlay.

use super::shaders;
use ash::vk;
use egui::epaint::{ImageDelta, Primitive};
use egui::{ClippedPrimitive, ImageData, TextureFilter, TextureId, TextureOptions, TextureWrapMode, TexturesDelta};
use std::collections::HashMap;

/// One egui UI frame to draw.
pub(crate) struct UiFrame {
    pub primitives: Vec<ClippedPrimitive>,
    pub pixels_per_point: f32,
}

struct HostBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    ptr: *mut u8,
}

struct Texture {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    width: u32,
    height: u32,
    options: TextureOptions,
}

#[derive(Default)]
struct Slot {
    upload: Option<HostBuffer>,
    vertices: Option<HostBuffer>,
    indices: Option<HostBuffer>,
}

/// One texture region copy of a prepared frame.
struct Upload {
    image: vk::Image,
    offset: u64,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    /// The image was created for this upload (UNDEFINED layout).
    fresh: bool,
}

struct Draw {
    view: vk::ImageView,
    sampler: vk::Sampler,
    scissor: vk::Rect2D,
    first_index: u32,
    index_count: u32,
    vertex_offset: i32,
}

/// What [`UiRenderer::prepare`] set up for one present.
pub(crate) struct UiDraw {
    uploads: Vec<Upload>,
    upload_buffer: vk::Buffer,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    draws: Vec<Draw>,
    screen_points: [f32; 2],
}

pub(crate) struct UiRenderer {
    device: ash::Device,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    format: vk::Format,
    /// The target is an HDR10 swapchain (the UI is converted to PQ).
    hdr: bool,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    samplers: HashMap<(bool, bool, u8), vk::Sampler>,
    textures: HashMap<TextureId, Texture>,
    /// Texture changes not drawn yet, in order.
    pending: Vec<(TextureId, ImageDelta)>,
    pending_free: Vec<TextureId>,
    /// Freed textures and the present count after which they are unused.
    retired: Vec<(Texture, u64)>,
    slots: Vec<Slot>,
    presents: u64,
}

type VkResult<T> = Result<T, vk::Result>;

fn is_srgb(f: vk::Format) -> bool {
    matches!(
        f,
        vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB | vk::Format::A8B8G8R8_SRGB_PACK32 | vk::Format::B8G8R8_SRGB | vk::Format::R8G8B8_SRGB
    )
}

/// Texels of an egui image as RGBA8 bytes (premultiplied, sRGB-encoded).
pub(crate) fn image_bytes(image: &ImageData) -> Vec<u8> {
    match image {
        ImageData::Color(c) => c.pixels.iter().flat_map(|p| p.to_array()).collect(),
        ImageData::Font(f) => f.srgba_pixels(None).flat_map(|p| p.to_array()).collect(),
    }
}

/// A scissor rectangle in pixels for an egui clip rectangle in points,
/// clamped to the target; `None` when nothing is left.
pub(crate) fn scissor(clip: egui::Rect, ppp: f32, width: u32, height: u32) -> Option<vk::Rect2D> {
    let x0 = (clip.min.x * ppp).round().clamp(0.0, width as f32) as u32;
    let y0 = (clip.min.y * ppp).round().clamp(0.0, height as f32) as u32;
    let x1 = (clip.max.x * ppp).round().clamp(0.0, width as f32) as u32;
    let y1 = (clip.max.y * ppp).round().clamp(0.0, height as f32) as u32;
    (x1 > x0 && y1 > y0).then(|| vk::Rect2D {
        offset: vk::Offset2D { x: x0 as i32, y: y0 as i32 },
        extent: vk::Extent2D { width: x1 - x0, height: y1 - y0 },
    })
}

/// Pack egui vertices as the pass reads them (pos, uv, rgba8).
pub(crate) fn pack_vertices(vertices: &[egui::epaint::Vertex], out: &mut Vec<u8>) {
    for v in vertices {
        out.extend_from_slice(&v.pos.x.to_le_bytes());
        out.extend_from_slice(&v.pos.y.to_le_bytes());
        out.extend_from_slice(&v.uv.x.to_le_bytes());
        out.extend_from_slice(&v.uv.y.to_le_bytes());
        out.extend_from_slice(&v.color.to_array());
    }
}

impl UiRenderer {
    pub(crate) fn new(device: &ash::Device, mem_props: vk::PhysicalDeviceMemoryProperties, format: vk::Format, hdr: bool, slots: usize) -> VkResult<Self> {
        let mut r = UiRenderer {
            device: device.clone(),
            mem_props,
            format,
            hdr,
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            samplers: HashMap::new(),
            textures: HashMap::new(),
            pending: Vec::new(),
            pending_free: Vec::new(),
            retired: Vec::new(),
            slots: (0..slots).map(|_| Slot::default()).collect(),
            presents: 0,
        };
        // SAFETY: valid create infos; everything is destroyed in Drop.
        unsafe {
            let binding = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
            r.set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                    .bindings(&binding),
                None,
            )?;
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                .size(shaders::PUSH_CONSTANT_BYTES)];
            let set_layouts = [r.set_layout];
            r.layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts).push_constant_ranges(&ranges), None)?;
            r.pipeline = r.create_pipeline(format, hdr)?;
        }
        Ok(r)
    }

    unsafe fn create_pipeline(&self, format: vk::Format, hdr: bool) -> VkResult<vk::Pipeline> {
        let d = &self.device;
        let module = |words: &[u32]| d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None);
        let vs = module(&shaders::vertex())?;
        let fs = match module(&if hdr { shaders::fragment_hdr() } else { shaders::fragment() }) {
            Ok(m) => m,
            Err(e) => {
                d.destroy_shader_module(vs, None);
                return Err(e);
            }
        };
        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vs).name(c"main"),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(fs).name(c"main"),
        ];
        let bindings = [vk::VertexInputBindingDescription::default().binding(0).stride(shaders::VERTEX_STRIDE).input_rate(vk::VertexInputRate::VERTEX)];
        let attributes = [
            vk::VertexInputAttributeDescription::default().location(0).binding(0).format(vk::Format::R32G32_SFLOAT).offset(0),
            vk::VertexInputAttributeDescription::default().location(1).binding(0).format(vk::Format::R32G32_SFLOAT).offset(8),
            vk::VertexInputAttributeDescription::default().location(2).binding(0).format(vk::Format::R8G8B8A8_UNORM).offset(16),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default().vertex_binding_descriptions(&bindings).vertex_attribute_descriptions(&attributes);
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        // Premultiplied alpha.
        let attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_DST_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let formats = [format];
        let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(self.layout)
            .push_next(&mut rendering);
        let r = d.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None);
        d.destroy_shader_module(vs, None);
        d.destroy_shader_module(fs, None);
        r.map(|p| p[0]).map_err(|(_, e)| e)
    }

    /// The swapchain format or its HDR10 colour space changed (the caller
    /// waited for the frames).
    pub(crate) fn set_format(&mut self, format: vk::Format, hdr: bool) -> VkResult<()> {
        if format == self.format && hdr == self.hdr {
            return Ok(());
        }
        // SAFETY: no submission uses the old pipeline any more.
        unsafe {
            let p = self.create_pipeline(format, hdr)?;
            self.device.destroy_pipeline(self.pipeline, None);
            self.pipeline = p;
        }
        self.format = format;
        self.hdr = hdr;
        Ok(())
    }

    /// Queue egui's texture changes of one UI frame.
    pub(crate) fn queue_textures(&mut self, delta: TexturesDelta) {
        self.pending.extend(delta.set);
        self.pending_free.extend(delta.free);
    }

    fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem_props.memory_type_count)
            .find(|&i| bits & (1 << i) != 0 && self.mem_props.memory_types[i as usize].property_flags.contains(flags))
    }

    unsafe fn ensure_buffer(&self, old: &mut Option<HostBuffer>, size: u64, usage: vk::BufferUsageFlags) -> VkResult<()> {
        if old.as_ref().is_some_and(|b| b.size >= size) {
            return Ok(());
        }
        if let Some(b) = old.take() {
            self.destroy_buffer(b);
        }
        let size = size.max(4096).next_power_of_two();
        let buffer = self.device.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE), None)?;
        let req = self.device.get_buffer_memory_requirements(buffer);
        let Some(ty) = self.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT) else {
            self.device.destroy_buffer(buffer, None);
            return Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        };
        let memory = match self.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None) {
            Ok(m) => m,
            Err(e) => {
                self.device.destroy_buffer(buffer, None);
                return Err(e);
            }
        };
        let ptr = self
            .device
            .bind_buffer_memory(buffer, memory, 0)
            .and_then(|_| self.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()));
        match ptr {
            Ok(p) => {
                *old = Some(HostBuffer { buffer, memory, size, ptr: p as *mut u8 });
                Ok(())
            }
            Err(e) => {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
                Err(e)
            }
        }
    }

    unsafe fn destroy_buffer(&self, b: HostBuffer) {
        self.device.unmap_memory(b.memory);
        self.device.destroy_buffer(b.buffer, None);
        self.device.free_memory(b.memory, None);
    }

    unsafe fn destroy_texture(&self, t: Texture) {
        self.device.destroy_image_view(t.view, None);
        self.device.destroy_image(t.image, None);
        self.device.free_memory(t.memory, None);
    }

    unsafe fn create_texture(&self, width: u32, height: u32, options: TextureOptions) -> VkResult<Texture> {
        let d = &self.device;
        let image = d.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(vk::Extent3D { width, height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            None,
        )?;
        let req = d.get_image_memory_requirements(image);
        let ty = self
            .memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .or_else(|| self.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::empty()));
        let Some(ty) = ty else {
            d.destroy_image(image, None);
            return Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        };
        let memory = match d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None) {
            Ok(m) => m,
            Err(e) => {
                d.destroy_image(image, None);
                return Err(e);
            }
        };
        let view = d.bind_image_memory(image, memory, 0).and_then(|_| {
            d.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(color_range()),
                None,
            )
        });
        match view {
            Ok(view) => Ok(Texture { image, memory, view, width, height, options }),
            Err(e) => {
                d.destroy_image(image, None);
                d.free_memory(memory, None);
                Err(e)
            }
        }
    }

    fn sampler(&mut self, o: TextureOptions) -> VkResult<vk::Sampler> {
        let wrap = match o.wrap_mode {
            TextureWrapMode::ClampToEdge => 0u8,
            TextureWrapMode::Repeat => 1,
            TextureWrapMode::MirroredRepeat => 2,
        };
        let key = (o.magnification == TextureFilter::Nearest, o.minification == TextureFilter::Nearest, wrap);
        if let Some(&s) = self.samplers.get(&key) {
            return Ok(s);
        }
        let filter = |nearest: bool| if nearest { vk::Filter::NEAREST } else { vk::Filter::LINEAR };
        let address = [vk::SamplerAddressMode::CLAMP_TO_EDGE, vk::SamplerAddressMode::REPEAT, vk::SamplerAddressMode::MIRRORED_REPEAT][wrap as usize];
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(filter(key.0))
            .min_filter(filter(key.1))
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(address)
            .address_mode_v(address)
            .address_mode_w(address)
            .max_lod(0.0);
        // SAFETY: valid create info; destroyed in Drop.
        let s = unsafe { self.device.create_sampler(&info, None)? };
        self.samplers.insert(key, s);
        Ok(s)
    }

    /// Set up frame slot `slot` to draw `frame` into a `width` x `height`
    /// target: texture uploads, vertices, indices and the draw list.
    pub(crate) fn prepare(&mut self, slot: usize, frame: &UiFrame, width: u32, height: u32) -> VkResult<Option<UiDraw>> {
        self.presents += 1;
        // Textures freed earlier whose last reader has been waited for
        // (the caller waited for this slot's previous submission).
        let done = self.presents.saturating_sub(self.slots.len() as u64 + 1);
        let (old, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut self.retired).into_iter().partition(|(_, at)| *at <= done);
        self.retired = keep;
        for (t, _) in old {
            // SAFETY: no submission reads it any more.
            unsafe { self.destroy_texture(t) };
        }
        if slot >= self.slots.len() || width == 0 || height == 0 {
            return Ok(None);
        }
        // Texture changes: whole images become new textures, partial
        // ones are copied into the existing image.
        let mut staged: Vec<u8> = Vec::new();
        let mut uploads: Vec<Upload> = Vec::new();
        for (id, delta) in std::mem::take(&mut self.pending) {
            let [w, h] = delta.image.size();
            if w == 0 || h == 0 {
                continue;
            }
            let (w, h) = (w as u32, h as u32);
            let (image, x, y, fresh) = match delta.pos {
                None => {
                    // SAFETY: a new image; the old one is retired.
                    let t = unsafe { self.create_texture(w, h, delta.options)? };
                    let image = t.image;
                    if let Some(old) = self.textures.insert(id, t) {
                        self.retired.push((old, self.presents));
                    }
                    (image, 0, 0, true)
                }
                Some([x, y]) => {
                    let Some(t) = self.textures.get_mut(&id) else { continue };
                    if x as u32 + w > t.width || y as u32 + h > t.height {
                        log::warn!("menu: texture update outside {id:?}");
                        continue;
                    }
                    t.options = delta.options;
                    (t.image, x as u32, y as u32, false)
                }
            };
            let offset = staged.len() as u64;
            staged.extend_from_slice(&image_bytes(&delta.image));
            uploads.push(Upload { image, offset, x, y, width: w, height: h, fresh });
        }
        let mut s = std::mem::take(&mut self.slots[slot]);
        let result = self.prepare_slot(&mut s, frame, width, height, &staged, uploads);
        self.slots[slot] = s;
        // Freed textures are retired after this frame's draws.
        for id in std::mem::take(&mut self.pending_free) {
            if let Some(t) = self.textures.remove(&id) {
                self.retired.push((t, self.presents));
            }
        }
        result
    }

    fn prepare_slot(
        &mut self,
        s: &mut Slot,
        frame: &UiFrame,
        width: u32,
        height: u32,
        staged: &[u8],
        uploads: Vec<Upload>,
    ) -> VkResult<Option<UiDraw>> {
        let ppp = frame.pixels_per_point.max(0.1);
        let mut vertices: Vec<u8> = Vec::new();
        let mut indices: Vec<u8> = Vec::new();
        let mut draws = Vec::new();
        let mut vcount = 0u32;
        let mut icount = 0u32;
        for cp in &frame.primitives {
            let Primitive::Mesh(mesh) = &cp.primitive else { continue };
            if mesh.indices.is_empty() || mesh.vertices.is_empty() {
                continue;
            }
            let Some(rect) = scissor(cp.clip_rect, ppp, width, height) else { continue };
            let Some((view, options)) = self.textures.get(&mesh.texture_id).map(|t| (t.view, t.options)) else { continue };
            let sampler = self.sampler(options)?;
            pack_vertices(&mesh.vertices, &mut vertices);
            for &i in &mesh.indices {
                indices.extend_from_slice(&i.to_le_bytes());
            }
            draws.push(Draw {
                view,
                sampler,
                scissor: rect,
                first_index: icount,
                index_count: mesh.indices.len() as u32,
                vertex_offset: vcount as i32,
            });
            vcount += mesh.vertices.len() as u32;
            icount += mesh.indices.len() as u32;
        }
        // SAFETY: host-visible buffers of this slot; its previous
        // submission has completed.
        unsafe {
            if !staged.is_empty() {
                self.ensure_buffer(&mut s.upload, staged.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC)?;
                let b = s.upload.as_ref().unwrap();
                std::ptr::copy_nonoverlapping(staged.as_ptr(), b.ptr, staged.len());
            }
            if !draws.is_empty() {
                self.ensure_buffer(&mut s.vertices, vertices.len() as u64, vk::BufferUsageFlags::VERTEX_BUFFER)?;
                self.ensure_buffer(&mut s.indices, indices.len() as u64, vk::BufferUsageFlags::INDEX_BUFFER)?;
                std::ptr::copy_nonoverlapping(vertices.as_ptr(), s.vertices.as_ref().unwrap().ptr, vertices.len());
                std::ptr::copy_nonoverlapping(indices.as_ptr(), s.indices.as_ref().unwrap().ptr, indices.len());
            }
        }
        if uploads.is_empty() && draws.is_empty() {
            return Ok(None);
        }
        let null = vk::Buffer::null();
        Ok(Some(UiDraw {
            uploads,
            upload_buffer: s.upload.as_ref().map_or(null, |b| b.buffer),
            vertex_buffer: s.vertices.as_ref().map_or(null, |b| b.buffer),
            index_buffer: s.indices.as_ref().map_or(null, |b| b.buffer),
            draws,
            screen_points: [width as f32 / ppp, height as f32 / ppp],
        }))
    }

    /// Record the texture uploads (outside any rendering pass).
    pub(crate) unsafe fn record_uploads(&self, cmd: vk::CommandBuffer, draw: &UiDraw) {
        if draw.uploads.is_empty() {
            return;
        }
        use vk::AccessFlags as A;
        use vk::ImageLayout as L;
        let d = &self.device;
        let barrier = |image, old, new, src, dst| {
            vk::ImageMemoryBarrier::default()
                .image(image)
                .old_layout(old)
                .new_layout(new)
                .src_access_mask(src)
                .dst_access_mask(dst)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(color_range())
        };
        // One transition per image, from its state before the first copy:
        // a new image has no contents, an updated one keeps them (earlier
        // frames' reads are ordered before by the barrier).
        let mut seen = Vec::new();
        let to_dst: Vec<_> = draw
            .uploads
            .iter()
            .filter(|u| {
                let first = !seen.contains(&u.image);
                seen.push(u.image);
                first
            })
            .map(|u| {
                if u.fresh {
                    barrier(u.image, L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, A::empty(), A::TRANSFER_WRITE)
                } else {
                    barrier(u.image, L::SHADER_READ_ONLY_OPTIMAL, L::TRANSFER_DST_OPTIMAL, A::SHADER_READ, A::TRANSFER_WRITE)
                }
            })
            .collect();
        d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &to_dst);
        for u in &draw.uploads {
            let copy = vk::BufferImageCopy::default()
                .buffer_offset(u.offset)
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                .image_offset(vk::Offset3D { x: u.x as i32, y: u.y as i32, z: 0 })
                .image_extent(vk::Extent3D { width: u.width, height: u.height, depth: 1 });
            d.cmd_copy_buffer_to_image(cmd, draw.upload_buffer, u.image, L::TRANSFER_DST_OPTIMAL, &[copy]);
        }
        seen.clear();
        let to_read: Vec<_> = draw
            .uploads
            .iter()
            .filter(|u| {
                let first = !seen.contains(&u.image);
                seen.push(u.image);
                first
            })
            .map(|u| barrier(u.image, L::TRANSFER_DST_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, A::TRANSFER_WRITE, A::SHADER_READ))
            .collect();
        d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &to_read);
    }

    /// Record the draws into `view` (COLOR_ATTACHMENT_OPTIMAL), `width` x
    /// `height`, inside a rendering pass of their own.
    pub(crate) unsafe fn record_draws(
        &self,
        cmd: vk::CommandBuffer,
        push: &ash::khr::push_descriptor::Device,
        view: vk::ImageView,
        width: u32,
        height: u32,
        draw: &UiDraw,
    ) {
        if draw.draws.is_empty() {
            return;
        }
        let d = &self.device;
        let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width, height } };
        let color = [vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)];
        d.cmd_begin_rendering(cmd, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&color));
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        d.cmd_set_viewport(cmd, 0, &[vk::Viewport { x: 0.0, y: 0.0, width: width as f32, height: height as f32, min_depth: 0.0, max_depth: 1.0 }]);
        d.cmd_bind_vertex_buffers(cmd, 0, &[draw.vertex_buffer], &[0]);
        d.cmd_bind_index_buffer(cmd, draw.index_buffer, 0, vk::IndexType::UINT32);
        let mut pc = [0u8; 16];
        pc[0..4].copy_from_slice(&draw.screen_points[0].to_le_bytes());
        pc[4..8].copy_from_slice(&draw.screen_points[1].to_le_bytes());
        pc[8..12].copy_from_slice(&u32::from(is_srgb(self.format)).to_le_bytes());
        d.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, &pc);
        for dr in &draw.draws {
            d.cmd_set_scissor(cmd, 0, &[dr.scissor]);
            let info = [vk::DescriptorImageInfo { sampler: dr.sampler, image_view: dr.view, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
            let write = vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&info);
            push.cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[write]);
            d.cmd_draw_indexed(cmd, dr.index_count, 1, dr.first_index, dr.vertex_offset, 0);
        }
        d.cmd_end_rendering(cmd);
    }
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }
}

impl Drop for UiRenderer {
    fn drop(&mut self) {
        // SAFETY: the owner waited for every submission that uses these.
        unsafe {
            for s in std::mem::take(&mut self.slots) {
                for b in [s.upload, s.vertices, s.indices].into_iter().flatten() {
                    self.destroy_buffer(b);
                }
            }
            for (_, t) in std::mem::take(&mut self.textures) {
                self.destroy_texture(t);
            }
            for (t, _) in std::mem::take(&mut self.retired) {
                self.destroy_texture(t);
            }
            for (_, s) in self.samplers.drain() {
                self.device.destroy_sampler(s, None);
            }
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Color32, Pos2, Rect};

    #[test]
    fn scissor_in_pixels_and_clamped() {
        let r = scissor(Rect::from_min_max(Pos2::new(10.0, 20.0), Pos2::new(110.0, 70.0)), 2.0, 1920, 1080).unwrap();
        assert_eq!((r.offset.x, r.offset.y, r.extent.width, r.extent.height), (20, 40, 200, 100));
        let r = scissor(Rect::from_min_max(Pos2::new(-5.0, -5.0), Pos2::new(1e9, 1e9)), 1.0, 800, 600).unwrap();
        assert_eq!((r.offset.x, r.offset.y, r.extent.width, r.extent.height), (0, 0, 800, 600));
        assert!(scissor(Rect::from_min_max(Pos2::new(900.0, 0.0), Pos2::new(950.0, 10.0)), 1.0, 800, 600).is_none());
        assert!(scissor(Rect::NOTHING, 1.0, 800, 600).is_none());
    }

    #[test]
    fn vertices_are_packed_as_the_pipeline_reads_them() {
        let v = egui::epaint::Vertex { pos: Pos2::new(1.5, -2.0), uv: Pos2::new(0.25, 0.75), color: Color32::from_rgba_premultiplied(1, 2, 3, 4) };
        let mut out = Vec::new();
        pack_vertices(&[v, v], &mut out);
        assert_eq!(out.len(), 2 * shaders::VERTEX_STRIDE as usize);
        assert_eq!(&out[0..4], &1.5f32.to_le_bytes());
        assert_eq!(&out[4..8], &(-2.0f32).to_le_bytes());
        assert_eq!(&out[8..12], &0.25f32.to_le_bytes());
        assert_eq!(&out[12..16], &0.75f32.to_le_bytes());
        assert_eq!(&out[16..20], &[1, 2, 3, 4]);
    }

    #[test]
    fn egui_textures_become_rgba8() {
        let img = ImageData::Color(std::sync::Arc::new(egui::ColorImage::new([2, 1], Color32::from_rgb(10, 20, 30))));
        assert_eq!(image_bytes(&img), [10, 20, 30, 255, 10, 20, 30, 255]);
        let font = ImageData::Font(egui::epaint::FontImage::new([3, 2]));
        assert_eq!(image_bytes(&font).len(), 3 * 2 * 4);
    }
}

#[cfg(test)]
mod device_tests {
    use super::*;
    use crate::device::GpuDevice;
    use egui::{Color32, Pos2, Rect};

    const W: u32 = 256;

    /// Draws an egui frame into an offscreen image and reads it back:
    /// texture upload (font atlas), vertices, scissor and blending.
    #[test]
    #[ignore]
    fn menu_pass_draws_egui_output() {
        let px = draw_menu_frame(vk::Format::R8G8B8A8_UNORM, false);
        let at = |x: u32, y: u32| {
            let i = ((y * W + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        };
        assert_eq!(at(30, 20), [255, 0, 0, 255], "inside the rectangle");
        assert_eq!(at(5, 5)[..3], [0, 0, 0], "outside it");
        assert_eq!(at(60, 20)[..3], [0, 0, 0], "outside it");
        // The label put some light pixels into its box.
        let lit = (60..90).flat_map(|y| (100..170).map(move |x| (x, y))).filter(|&(x, y)| at(x, y)[0] > 128).count();
        assert!(lit > 20, "text drawn ({lit} pixels)");
    }

    /// The HDR10 pass writes the UI as BT.2020 PQ, matching the CPU
    /// conversion of the same colour.
    #[test]
    #[ignore]
    fn menu_pass_converts_to_hdr10() {
        let format = vk::Format::A2B10G10R10_UNORM_PACK32;
        let px = draw_menu_frame(format, true);
        let at = |x: u32, y: u32| {
            let i = ((y * W + x) * 4) as usize;
            let p = u32::from_le_bytes([px[i], px[i + 1], px[i + 2], px[i + 3]]);
            [p & 0x3FF, (p >> 10) & 0x3FF, (p >> 20) & 0x3FF, p >> 30]
        };
        let want = crate::hdr::sdr_to_hdr10([1.0, 0.0, 0.0], false).map(|v| (v * 1023.0).round() as u32);
        let red = at(30, 20);
        for c in 0..3 {
            assert!(red[c].abs_diff(want[c]) <= 2, "red rectangle {red:?}, expected {want:?}");
        }
        assert_eq!(red[3], 3);
        assert_eq!(at(5, 5)[..3], [0, 0, 0], "black stays black");
        // White text lands at the SDR reference white (code 616), not above.
        let peak = (60..90).flat_map(|y| (100..170).map(move |x| (x, y))).map(|(x, y)| at(x, y)[1]).max().unwrap();
        assert!((600..=620).contains(&peak), "text peak {peak}");
    }

    fn draw_menu_frame(format: vk::Format, hdr: bool) -> Vec<u8> {
        let dev = GpuDevice::create(false).expect("device");
        let d = &dev.device;
        let (w, h) = (W, 128u32);
        // SAFETY: test-owned objects, destroyed at the end.
        unsafe {
            let mem_type = |bits: u32, flags| {
                (0..dev.mem_props.memory_type_count)
                    .find(|&i| bits & (1 << i) != 0 && dev.mem_props.memory_types[i as usize].property_flags.contains(flags))
                    .unwrap()
            };
            let image = d
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(vk::Extent3D { width: w, height: h, depth: 1 })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST),
                    None,
                )
                .unwrap();
            let req = d.get_image_memory_requirements(image);
            let image_mem = d
                .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mem_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)), None)
                .unwrap();
            d.bind_image_memory(image, image_mem, 0).unwrap();
            let view = d
                .create_image_view(&vk::ImageViewCreateInfo::default().image(image).view_type(vk::ImageViewType::TYPE_2D).format(format).subresource_range(color_range()), None)
                .unwrap();
            let bytes = (w * h * 4) as u64;
            let rb = d.create_buffer(&vk::BufferCreateInfo::default().size(bytes).usage(vk::BufferUsageFlags::TRANSFER_DST), None).unwrap();
            let req = d.get_buffer_memory_requirements(rb);
            let rb_mem = d
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(mem_type(req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)),
                    None,
                )
                .unwrap();
            d.bind_buffer_memory(rb, rb_mem, 0).unwrap();

            // One egui frame: a red rectangle and a label, clipped to the image.
            let ctx = egui::Context::default();
            // No fade-in: the area is drawn opaque at once.
            ctx.style_mut(|s| s.animation_time = 0.0);
            let mut ui = UiRenderer::new(d, dev.mem_props, format, hdr, 1).unwrap();
            let mut frame = None;
            for _ in 0..2 {
                let raw = egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(w as f32, h as f32))),
                    ..Default::default()
                };
                let out = ctx.run(raw, |ctx| {
                    egui::Area::new(egui::Id::new("t")).fixed_pos(Pos2::ZERO).show(ctx, |ui| {
                        ui.painter().rect_filled(Rect::from_min_size(Pos2::new(10.0, 10.0), egui::vec2(40.0, 20.0)), 0.0, Color32::RED);
                        ui.painter().text(Pos2::new(100.0, 60.0), egui::Align2::LEFT_TOP, "Menu", egui::FontId::proportional(24.0), Color32::WHITE);
                    });
                });
                ui.queue_textures(out.textures_delta);
                frame = Some(UiFrame { primitives: ctx.tessellate(out.shapes, out.pixels_per_point), pixels_per_point: out.pixels_per_point });
            }
            let draw = ui.prepare(0, frame.as_ref().unwrap(), w, h).unwrap().expect("something to draw");
            assert!(!draw.uploads.is_empty(), "the font atlas is uploaded");

            let pool = d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(dev.queue_family), None).unwrap();
            let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).unwrap()[0];
            d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
            let barrier = |old, new, src, dst| {
                vk::ImageMemoryBarrier::default()
                    .image(image)
                    .old_layout(old)
                    .new_layout(new)
                    .src_access_mask(src)
                    .dst_access_mask(dst)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .subresource_range(color_range())
            };
            use vk::AccessFlags as A;
            use vk::ImageLayout as L;
            use vk::PipelineStageFlags as S;
            let all = S::ALL_COMMANDS;
            d.cmd_pipeline_barrier(cmd, all, all, vk::DependencyFlags::empty(), &[], &[], &[barrier(L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, A::empty(), A::TRANSFER_WRITE)]);
            d.cmd_clear_color_image(cmd, image, L::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] }, &[color_range()]);
            ui.record_uploads(cmd, &draw);
            d.cmd_pipeline_barrier(
                cmd,
                all,
                all,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(L::TRANSFER_DST_OPTIMAL, L::COLOR_ATTACHMENT_OPTIMAL, A::TRANSFER_WRITE, A::COLOR_ATTACHMENT_WRITE | A::COLOR_ATTACHMENT_READ)],
            );
            ui.record_draws(cmd, &dev.push_descriptor, view, w, h, &draw);
            d.cmd_pipeline_barrier(cmd, all, all, vk::DependencyFlags::empty(), &[], &[], &[barrier(L::COLOR_ATTACHMENT_OPTIMAL, L::TRANSFER_SRC_OPTIMAL, A::COLOR_ATTACHMENT_WRITE, A::TRANSFER_READ)]);
            let copy = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
            d.cmd_copy_image_to_buffer(cmd, image, L::TRANSFER_SRC_OPTIMAL, rb, &[copy]);
            d.end_command_buffer(cmd).unwrap();
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            let cmds = [cmd];
            d.queue_submit(*dev.queue(), &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).unwrap();
            d.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            let ptr = d.map_memory(rb_mem, 0, bytes, vk::MemoryMapFlags::empty()).unwrap() as *const u8;
            let px = std::slice::from_raw_parts(ptr, bytes as usize).to_vec();
            d.unmap_memory(rb_mem);
            drop(ui);
            d.destroy_fence(fence, None);
            d.destroy_command_pool(pool, None);
            d.destroy_buffer(rb, None);
            d.free_memory(rb_mem, None);
            d.destroy_image_view(view, None);
            d.destroy_image(image, None);
            d.free_memory(image_mem, None);
            px
        }
    }
}
