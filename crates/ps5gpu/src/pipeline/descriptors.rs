// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Descriptor-set layouts, pipeline layouts, samplers, the descriptor-pool
//! fallback for sets larger than the push-descriptor limit, and the dummy
//! resources bound where a guest resource cannot be represented.

use super::compiler::{descriptor_type, stage_flags};
use super::sharps::SamplerKey;
use crate::device::{GpuDevice, Image, ImageView};
use crate::scheduler::Scheduler;
use ash::vk;
use ps5shader::resources::{native_binding_index, BindingLayout, ImageNumericClass, ShaderStage};
use crate::fxhash::HashMap;
use std::sync::Arc;

/// Bytes of the push-constant block shared by the stages of a pipeline.
pub const PUSH_CONSTANT_BYTES: u32 = ps5shader::resources::push_data::BYTE_SIZE;

/// One binding of a set layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BindingDesc {
    pub binding: u32,
    pub ty: i32,
    pub count: u32,
    pub stages: u32,
}

/// The binding number of the entry [`set_bindings`] adds when a stage selects
/// images through the bindless set (not a binding of set 0: the pipeline
/// layout adds the bindless set).
pub const BINDLESS_SET_MARKER: u32 = u32::MAX;

/// The set layout bindings of `stages` (each with its binding layout).
pub fn set_bindings(stages: &[(ShaderStage, &BindingLayout)]) -> Vec<BindingDesc> {
    let mut out = Vec::new();
    if stages.iter().any(|(_, l)| l.bindless_tables) {
        out.push(BindingDesc { binding: BINDLESS_SET_MARKER, ty: 0, count: 0, stages: 0 });
    }
    for (stage, layout) in stages {
        for b in &layout.descriptors {
            out.push(BindingDesc {
                binding: native_binding_index(*stage, b.kind),
                ty: descriptor_type(b.kind).as_raw(),
                count: (b.resources.len() as u32).max(1),
                stages: stage_flags(*stage).as_raw(),
            });
        }
    }
    out.sort();
    out
}

/// A pipeline layout with its set layout.
#[derive(Clone, Copy, Debug)]
pub struct LayoutEntry {
    pub set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub push_descriptors: bool,
    pub descriptor_count: u32,
    /// The pipeline layout has the bindless set as set 1.
    pub bindless: bool,
}

/// Set and pipeline layouts keyed by their bindings and push stages.
pub struct LayoutCache {
    dev: Arc<GpuDevice>,
    entries: HashMap<(Vec<BindingDesc>, u32), LayoutEntry>,
    /// The bindless set's layout, once the set exists (layouts asking for
    /// it fail before).
    pub bindless: Option<vk::DescriptorSetLayout>,
}

impl LayoutCache {
    pub fn new(dev: &Arc<GpuDevice>) -> LayoutCache {
        LayoutCache { dev: dev.clone(), entries: HashMap::default(), bindless: None }
    }

    /// The layout of `bindings` when made already (read-only).
    pub fn find(&self, bindings: &[BindingDesc], push_stages: vk::ShaderStageFlags) -> Option<LayoutEntry> {
        self.entries.get(&(bindings.to_vec(), push_stages.as_raw())).copied()
    }

    pub fn get(&mut self, bindings: &[BindingDesc], push_stages: vk::ShaderStageFlags) -> Result<LayoutEntry, vk::Result> {
        let key = (bindings.to_vec(), push_stages.as_raw());
        if let Some(e) = self.entries.get(&key) {
            return Ok(*e);
        }
        // The marker sorts last.
        let bindless = bindings.last().is_some_and(|b| b.binding == BINDLESS_SET_MARKER);
        let bindings = if bindless { &bindings[..bindings.len() - 1] } else { bindings };
        let bindless_layout = match (bindless, self.bindless) {
            (false, _) => None,
            (true, Some(l)) => Some(l),
            (true, None) => return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT),
        };
        let descriptor_count: u32 = bindings.iter().map(|b| b.count).sum();
        let push_descriptors = descriptor_count <= self.dev.caps.max_push_descriptors;
        let vk_bindings: Vec<vk::DescriptorSetLayoutBinding> = bindings
            .iter()
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(vk::DescriptorType::from_raw(b.ty))
                    .descriptor_count(b.count)
                    .stage_flags(vk::ShaderStageFlags::from_raw(b.stages))
            })
            .collect();
        let flags = if push_descriptors { vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR } else { vk::DescriptorSetLayoutCreateFlags::empty() };
        let d = &self.dev.device;
        // SAFETY: valid create infos; the handles live for the cache.
        let set_layout = unsafe { d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(flags).bindings(&vk_bindings), None)? };
        let ranges = [vk::PushConstantRange { stage_flags: push_stages, offset: 0, size: PUSH_CONSTANT_BYTES }];
        let set_layouts = [set_layout, bindless_layout.unwrap_or_default()];
        let set_layouts = &set_layouts[..if bindless { 2 } else { 1 }];
        let info = vk::PipelineLayoutCreateInfo::default().set_layouts(set_layouts).push_constant_ranges(&ranges);
        // SAFETY: as above.
        let pipeline_layout = match unsafe { d.create_pipeline_layout(&info, None) } {
            Ok(l) => l,
            Err(e) => {
                // SAFETY: unused layout.
                unsafe { d.destroy_descriptor_set_layout(set_layout, None) };
                return Err(e);
            }
        };
        let e = LayoutEntry { set_layout, pipeline_layout, push_descriptors, descriptor_count, bindless };
        self.entries.insert(key, e);
        Ok(e)
    }
}

impl Drop for LayoutCache {
    fn drop(&mut self) {
        let d = &self.dev.device;
        for e in self.entries.values() {
            // SAFETY: the owner waits for the device before dropping caches.
            unsafe {
                d.destroy_pipeline_layout(e.pipeline_layout, None);
                d.destroy_descriptor_set_layout(e.set_layout, None);
            }
        }
    }
}

/// Host samplers by key.
pub struct SamplerCache {
    dev: Arc<GpuDevice>,
    samplers: HashMap<SamplerKey, vk::Sampler>,
    max_anisotropy: f32,
    anisotropy: bool,
    minmax: bool,
}

impl SamplerCache {
    pub fn new(dev: &Arc<GpuDevice>) -> SamplerCache {
        // SAFETY: plain query.
        let features = unsafe { dev.instance.get_physical_device_features(dev.phys) };
        SamplerCache {
            dev: dev.clone(),
            samplers: HashMap::default(),
            max_anisotropy: dev.props.limits.max_sampler_anisotropy,
            anisotropy: features.sampler_anisotropy != 0,
            minmax: dev.caps.sampler_filter_minmax,
        }
    }

    /// The sampler of `key` when made already (read-only).
    pub fn find(&self, key: &SamplerKey) -> Option<vk::Sampler> {
        self.samplers.get(key).copied()
    }

    pub fn get(&mut self, key: &SamplerKey) -> Result<vk::Sampler, vk::Result> {
        if let Some(&s) = self.samplers.get(key) {
            return Ok(s);
        }
        let max_bias = self.dev.props.limits.max_sampler_lod_bias;
        let mut info = key.create_info(self.max_anisotropy, self.anisotropy);
        // The guest bias is limited by the device.
        info.mip_lod_bias = (key.lod_bias as f32 / 256.0).clamp(-max_bias, max_bias);
        if key.unnormalized {
            info.mip_lod_bias = 0.0;
        }
        let mut reduction = vk::SamplerReductionModeCreateInfo::default().reduction_mode(match key.reduction {
            1 => vk::SamplerReductionMode::MIN,
            2 => vk::SamplerReductionMode::MAX,
            _ => vk::SamplerReductionMode::WEIGHTED_AVERAGE,
        });
        if self.minmax && matches!(key.reduction, 1 | 2) && key.compare.is_none() {
            info = info.push_next(&mut reduction);
        }
        // SAFETY: valid create info.
        let s = unsafe { self.dev.device.create_sampler(&info, None)? };
        self.samplers.insert(*key, s);
        Ok(s)
    }

    pub fn len(&self) -> usize {
        self.samplers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samplers.is_empty()
    }
}

impl Drop for SamplerCache {
    fn drop(&mut self) {
        for &s in self.samplers.values() {
            // SAFETY: device idle (owner waits before dropping).
            unsafe { self.dev.device.destroy_sampler(s, None) };
        }
    }
}

/// A descriptor pool.
struct Pool {
    dev: Arc<GpuDevice>,
    pool: vk::DescriptorPool,
}

impl Drop for Pool {
    fn drop(&mut self) {
        // SAFETY: dropped after every set allocated from it completed
        // (pending pools are only dropped with the heap, after the device
        // went idle).
        unsafe { self.dev.device.destroy_descriptor_pool(self.pool, None) };
    }
}

/// Sets allocated ahead for one layout.
struct Batch {
    sets: Vec<vk::DescriptorSet>,
    /// Sets the next allocation asks for (halved when the pool runs out).
    allocation: u32,
}

/// Descriptor sets for layouts above the push-descriptor limit. Sets are
/// allocated in batches per layout; a full pool waits, with the tick that
/// last used it, until that tick completed and is then reset and reused.
pub struct DescriptorHeap {
    dev: Arc<GpuDevice>,
    current: Option<Pool>,
    pending: std::collections::VecDeque<(Pool, u64)>,
    batches: HashMap<vk::DescriptorSetLayout, Batch>,
}

const HEAP_SETS: u32 = 1024;
/// Sets allocated per layout at a time.
const SET_BATCH: u32 = 32;

impl DescriptorHeap {
    pub fn new(dev: &Arc<GpuDevice>) -> DescriptorHeap {
        DescriptorHeap { dev: dev.clone(), current: None, pending: Default::default(), batches: HashMap::default() }
    }

    fn new_pool(&self) -> Result<Pool, vk::Result> {
        let sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: HEAP_SETS * 16 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLED_IMAGE, descriptor_count: HEAP_SETS * 16 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: HEAP_SETS * 4 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLER, descriptor_count: HEAP_SETS * 8 },
        ];
        let info = vk::DescriptorPoolCreateInfo::default().max_sets(HEAP_SETS).pool_sizes(&sizes);
        // SAFETY: valid create info.
        let pool = unsafe { self.dev.device.create_descriptor_pool(&info, None)? };
        Ok(Pool { dev: self.dev.clone(), pool })
    }

    /// Fill `batch` from the current pool, halving the request while the
    /// pool cannot hold it. False when not even one set fits.
    fn fill(dev: &GpuDevice, pool: vk::DescriptorPool, layout: vk::DescriptorSetLayout, batch: &mut Batch) -> Result<bool, vk::Result> {
        loop {
            let layouts = vec![layout; batch.allocation as usize];
            let info = vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts);
            // SAFETY: valid pool and layout.
            match unsafe { dev.device.allocate_descriptor_sets(&info) } {
                Ok(sets) => {
                    batch.sets = sets;
                    return Ok(true);
                }
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY) | Err(vk::Result::ERROR_FRAGMENTED_POOL) => {
                    if batch.allocation == 1 {
                        return Ok(false);
                    }
                    batch.allocation /= 2;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Pools filled at tick `from` are in use until `to` completes (a slice
    /// segment recorded at a provisional tick was submitted later, R7b).
    pub fn retag(&mut self, from: u64, to: u64) {
        for (_, t) in self.pending.iter_mut() {
            if *t == from {
                *t = to;
            }
        }
    }

    /// A fresh set of `layout`, valid until the current tick completes.
    pub fn allocate(&mut self, sched: &mut Scheduler, layout: vk::DescriptorSetLayout) -> Result<vk::DescriptorSet, vk::Result> {
        if self.current.is_none() {
            self.current = Some(self.new_pool()?);
        }
        let pool = self.current.as_ref().unwrap().pool;
        let batch = self.batches.entry(layout).or_insert_with(|| Batch { sets: Vec::new(), allocation: SET_BATCH });
        if let Some(set) = batch.sets.pop() {
            return Ok(set);
        }
        if Self::fill(&self.dev, pool, layout, batch)? {
            return Ok(batch.sets.pop().unwrap());
        }
        // The pool is full: it waits for the work using it; the oldest
        // waiting pool is reused once its work completed.
        let full = self.current.take().unwrap();
        self.pending.push_back((full, sched.current_tick()));
        let reuse = self.pending.front().is_some_and(|&(_, t)| sched.is_complete(t));
        let next = if reuse {
            let (p, _) = self.pending.pop_front().unwrap();
            // SAFETY: every set of the pool belongs to completed work.
            unsafe { self.dev.device.reset_descriptor_pool(p.pool, vk::DescriptorPoolResetFlags::empty())? };
            p
        } else {
            self.new_pool()?
        };
        let pool = next.pool;
        self.current = Some(next);
        // Sets allocated ahead belonged to the full pool.
        self.batches.clear();
        let batch = self.batches.entry(layout).or_insert_with(|| Batch { sets: Vec::new(), allocation: SET_BATCH });
        if !Self::fill(&self.dev, pool, layout, batch)? {
            return Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY);
        }
        Ok(batch.sets.pop().unwrap())
    }
}

/// Small resources bound where the guest resource is missing or cannot be
/// represented (null V#, unsupported T#), by view type and numeric class.
pub struct DummyResources {
    dev: Arc<GpuDevice>,
    images: HashMap<(i32, u8, bool, bool), (Image, ImageView)>,
    /// Constant `R32_UINT` images (FMASK reads) by (view type, extent,
    /// layers, value).
    constants: HashMap<(i32, u32, u32, u32, u32), (Image, ImageView)>,
}

fn class_tag(c: ImageNumericClass) -> u8 {
    match c {
        ImageNumericClass::Uint => 1,
        ImageNumericClass::Sint => 2,
        _ => 0,
    }
}

impl DummyResources {
    pub fn new(dev: &Arc<GpuDevice>) -> DummyResources {
        DummyResources { dev: dev.clone(), images: HashMap::default(), constants: HashMap::default() }
    }

    /// A `width` x `height` x `layers` `R32_UINT` image holding `value`
    /// everywhere (the FMASK of an expanded multisampled surface), viewed
    /// as `view_type` (2D or 2D array), in GENERAL layout.
    pub fn constant_view(&mut self, sched: &mut Scheduler, view_type: vk::ImageViewType, width: u32, height: u32, layers: u32, value: u32) -> Option<vk::ImageView> {
        let view_type = if view_type == vk::ImageViewType::TYPE_2D_ARRAY { view_type } else { vk::ImageViewType::TYPE_2D };
        let layers = if view_type == vk::ImageViewType::TYPE_2D { 1 } else { layers.max(1) };
        let key = (view_type.as_raw(), width, height, layers, value);
        if let Some((_, v)) = self.constants.get(&key) {
            return Some(v.view);
        }
        let format = vk::Format::R32_UINT;
        let image = Image::new_typed(
            &self.dev,
            vk::ImageType::TYPE_2D,
            format,
            vk::Extent3D { width: width.max(1), height: height.max(1), depth: 1 },
            1,
            layers,
            vk::SampleCountFlags::TYPE_1,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            vk::ImageCreateFlags::empty(),
        )
        .ok()?;
        let range = image.full_range();
        let cb = sched.cmd();
        let barrier = vk::ImageMemoryBarrier2::default()
            .image(image.image)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .subresource_range(range);
        // SAFETY: recording buffer; fresh image.
        unsafe {
            self.dev.device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier)));
            let fill = vk::ClearColorValue { uint32: [value; 4] };
            self.dev.device.cmd_clear_color_image(cb, image.image, vk::ImageLayout::GENERAL, &fill, &[range]);
        }
        sched.full_barrier();
        let view_info = vk::ImageViewCreateInfo::default().image(image.image).view_type(view_type).format(format).subresource_range(range);
        let view = ImageView::new(&self.dev, &view_info).ok()?;
        let handle = view.view;
        self.constants.insert(key, (image, view));
        Some(handle)
    }

    /// A 1x1 (x6 for cubes) zero image view of `view_type` and `class`
    /// (multisampled / storage as requested), in GENERAL layout.
    /// The dummy view [`Self::image_view`] gives when made already.
    pub fn find_image_view(&self, view_type: vk::ImageViewType, class: ImageNumericClass, multisampled: bool, storage: bool) -> Option<vk::ImageView> {
        self.images.get(&(view_type.as_raw(), class_tag(class), multisampled, storage)).map(|(_, v)| v.view)
    }

    pub fn image_view(
        &mut self,
        sched: &mut Scheduler,
        view_type: vk::ImageViewType,
        class: ImageNumericClass,
        multisampled: bool,
        storage: bool,
    ) -> Option<vk::ImageView> {
        let key = (view_type.as_raw(), class_tag(class), multisampled, storage);
        if let Some((_, v)) = self.images.get(&key) {
            return Some(v.view);
        }
        let format = match class {
            ImageNumericClass::Uint => vk::Format::R32_UINT,
            ImageNumericClass::Sint => vk::Format::R32_SINT,
            _ => vk::Format::R8G8B8A8_UNORM,
        };
        let (layers, flags, depth) = match view_type {
            vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY => (6, vk::ImageCreateFlags::CUBE_COMPATIBLE, 1),
            _ => (1, vk::ImageCreateFlags::empty(), 1),
        };
        let mut usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED;
        if storage {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        let samples = if multisampled { vk::SampleCountFlags::TYPE_4 } else { vk::SampleCountFlags::TYPE_1 };
        if multisampled {
            usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        let image_type = match view_type {
            vk::ImageViewType::TYPE_1D | vk::ImageViewType::TYPE_1D_ARRAY => vk::ImageType::TYPE_1D,
            vk::ImageViewType::TYPE_3D => vk::ImageType::TYPE_3D,
            _ => vk::ImageType::TYPE_2D,
        };
        let image = Image::new_typed(
            &self.dev,
            image_type,
            format,
            vk::Extent3D { width: 1, height: 1, depth },
            1,
            layers,
            samples,
            usage,
            flags,
        )
        .ok()?;
        let range = image.full_range();
        let cb = sched.cmd();
        let barrier = vk::ImageMemoryBarrier2::default()
            .image(image.image)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .subresource_range(range);
        // SAFETY: recording buffer; fresh image.
        unsafe {
            self.dev.device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier)));
            if !multisampled {
                let zero = vk::ClearColorValue { uint32: [0; 4] };
                self.dev.device.cmd_clear_color_image(cb, image.image, vk::ImageLayout::GENERAL, &zero, &[range]);
            }
        }
        sched.full_barrier();
        let view_info = vk::ImageViewCreateInfo::default().image(image.image).view_type(view_type).format(format).subresource_range(range);
        let view = ImageView::new(&self.dev, &view_info).ok()?;
        let handle = view.view;
        self.images.insert(key, (image, view));
        Some(handle)
    }
}
