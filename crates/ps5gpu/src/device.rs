// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The shared Vulkan device.
//!
//! One instance / physical device / logical device / queue serves both the
//! presenter (swapchain on the render thread) and the GPU back-end (command
//! recording on the GPU command thread). Required: a Vulkan 1.3 device with
//! dynamic rendering, synchronization2, timeline semaphores, buffer device
//! address, `shaderInt64` and `VK_KHR_push_descriptor`. Optional extensions
//! are enabled when present and reported in [`DeviceCaps`].
//!
//! The queue is shared: every submit or present goes through
//! [`GpuDevice::queue`], which serialises access.
//!
//! `RPS5E_GPU=0` disables the device (presenter falls back to headless, no
//! GPU back-end). `RPS5E_GPU_DEVICE=<index|name substring>` picks a device.

use ash::vk;
use std::ffi::{c_void, CStr};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// Optional capabilities and limits the rest of the crate may consult.
#[derive(Clone, Debug, Default)]
pub struct DeviceCaps {
    pub swapchain: bool,
    pub maintenance5: bool,
    pub maintenance8: bool,
    pub robust_buffer_access2: bool,
    pub robust_image_access2: bool,
    pub null_descriptor: bool,
    pub color_write_enable: bool,
    pub depth_clip_enable: bool,
    pub depth_clip_control: bool,
    /// VK_EXT_extended_dynamic_state enabled. Its dynamic states are core
    /// in Vulkan 1.3 and used from there; the feature is enabled anyway
    /// because some drivers set up per-device state (RADV: its pixel
    /// shader epilog cache) only when it is, and fail draws without it.
    /// `RPS5E_EXT_DYNAMIC_STATE=0` leaves it off.
    pub extended_dynamic_state: bool,
    pub fragment_shader_barycentric: bool,
    /// Pixel-ordered fragment interlock (VK_EXT_fragment_shader_interlock):
    /// primitive-ordered pixel programs.
    pub fragment_shader_pixel_interlock: bool,
    pub image_view_min_lod: bool,
    pub fill_rectangle: bool,
    pub shader_viewport_index_layer: bool,
    pub shader_float64: bool,
    pub shader_int16: bool,
    pub shader_float16: bool,
    pub shader_demote_to_helper_invocation: bool,
    pub shader_shared_int64_atomics: bool,
    /// 64-bit image atomics (VK_EXT_shader_image_atomic_int64).
    pub shader_image_int64_atomics: bool,
    /// Explicitly laid out workgroup memory (VK_KHR_workgroup_memory_explicit_layout).
    pub workgroup_memory_explicit_layout: bool,
    /// Quad derivatives in compute (VK_NV_compute_shader_derivatives).
    pub compute_derivative_group_quads: bool,
    /// Compute derivatives over four consecutive invocations (GFX10's compute quads).
    pub compute_derivative_group_linear: bool,
    /// 16 viewports selected by the vertex stage.
    pub multi_viewport: bool,
    /// pipelineCreationCacheControl: pipeline-cache probes
    /// (FAIL_ON_PIPELINE_COMPILE_REQUIRED).
    pub pipeline_creation_cache_control: bool,
    pub subgroup_size_control: bool,
    /// Default subgroup size.
    pub subgroup_size: u32,
    /// Subgroup sizes a compute pipeline can require (VK_EXT_subgroup_size_control;
    /// 0 when it cannot require one).
    pub min_compute_subgroup_size: u32,
    pub max_compute_subgroup_size: u32,
    /// maxComputeWorkgroupSubgroups (with a required subgroup size).
    pub max_compute_workgroup_subgroups: u32,
    /// Mesh subgroups may run narrower than 32 lanes (variable SIMD width,
    /// Intel) and a mesh stage can require 32.
    pub mesh_subgroup32: bool,
    /// maxComputeSharedMemorySize.
    pub max_compute_shared_memory: u32,
    pub independent_blend: bool,
    pub depth_bounds: bool,
    pub depth_bias_clamp: bool,
    pub texture_compression_bc: bool,
    pub sample_rate_shading: bool,
    pub tessellation_shader: bool,
    pub draw_indirect_count: bool,
    /// shaderDrawParameters: BaseVertex / BaseInstance in vertex shaders
    /// (indirect draws on the GPU).
    pub shader_draw_parameters: bool,
    pub multi_draw_indirect: bool,
    pub fill_mode_non_solid: bool,
    pub dual_src_blend: bool,
    /// samplerFilterMinmax: min/max reduction samplers.
    pub sampler_filter_minmax: bool,
    pub depth_clamp: bool,
    pub max_storage_buffer_range: u32,
    /// Nanoseconds per timestamp tick (0: no timestamps).
    pub timestamp_period: f32,
    pub min_storage_buffer_offset_alignment: u64,
    pub min_uniform_buffer_offset_alignment: u64,
    pub max_push_descriptors: u32,
    pub max_framebuffer_width: u32,
    pub max_framebuffer_height: u32,
    pub non_coherent_atom_size: u64,
    /// shaderSignedZeroInfNanPreserveFloat16/32/64 (float controls).
    pub signed_zero_inf_nan_preserve: [bool; 3],
    /// shaderDenormPreserveFloat16/32/64 (float controls).
    pub denorm_preserve: [bool; 3],
    /// shaderDenormFlushToZeroFloat32 (float controls).
    pub denorm_flush_to_zero32: bool,
    /// Native round-to-nearest-even f32 -> f16 conversions: shaderFloat16
    /// with shaderRoundingModeRTEFloat16 and 16-bit denormal and
    /// inf / NaN / signed-zero preservation.
    pub float16_rte: bool,
    /// VK_EXT_external_memory_host (zero-copy import of host/guest memory).
    pub external_memory_host: bool,
    /// VK_EXT_mesh_shader (geometry subgroups run as mesh workgroups).
    pub mesh_shader: bool,
    pub max_mesh_output_vertices: u32,
    pub max_mesh_output_primitives: u32,
    pub max_mesh_work_group_invocations: u32,
    pub max_mesh_shared_memory: u32,
    /// maxMeshWorkGroupCount and maxMeshWorkGroupTotalCount.
    pub max_mesh_work_group_count: [u32; 3],
    pub max_mesh_work_group_total_count: u32,
    pub min_imported_host_pointer_alignment: u64,
    /// VK_EXT_device_fault (fault details after a device loss).
    pub device_fault: bool,
    /// VK_NV_device_diagnostic_checkpoints, enabled with
    /// `RPS5E_GPU_CHECKPOINTS=1`: every dispatch leaves its program address
    /// as a checkpoint, and a device loss reports the last ones reached.
    pub diagnostic_checkpoints: bool,
    /// VK_EXT_conditional_rendering (GPU-decided metadata clears).
    pub conditional_rendering: bool,
    /// VK_EXT_memory_budget (process VRAM use for the performance overlay).
    pub memory_budget: bool,
    /// occlusionQueryPrecise (exact sample counts for guest occlusion queries).
    pub occlusion_query_precise: bool,
    /// Vulkan 1.2 hostQueryReset (query pools reset from the host).
    pub host_query_reset: bool,
    /// VK_KHR_shader_clock shaderDeviceClock (`s_memrealtime`).
    pub shader_device_clock: bool,
    /// VK_KHR_shader_clock shaderSubgroupClock (`s_memtime`).
    pub shader_subgroup_clock: bool,
    /// VK_EXT_provoking_vertex provokingVertexLast
    /// (PA_SU_SC_MODE_CNTL.PROVOKING_VTX_LAST).
    pub provoking_vertex_last: bool,
    /// provokingVertexModePerPipeline: pipelines of one rendering scope
    /// may use different provoking vertex modes.
    pub provoking_vertex_per_pipeline: bool,
    /// subgroupBroadcastDynamicId: OpGroupNonUniformBroadcast with a
    /// dynamically uniform lane.
    pub subgroup_broadcast_dynamic_id: bool,
    /// VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT (ShuffleUp / ShuffleDown).
    pub subgroup_shuffle_relative: bool,
    /// VK_SUBGROUP_FEATURE_ARITHMETIC_BIT in the vertex and fragment
    /// stages (subgroup reductions).
    pub subgroup_arithmetic: bool,
    /// Slots of the bindless set's image array (0: no bindless set). Needs
    /// runtime descriptor arrays, non-uniform sampled-image indexing and
    /// partially bound sampled images updated after bind and while unused
    /// by pending work; bounded by the update-after-bind limits.
    pub bindless_slots: u32,
    /// VK_EXT_pipeline_robustness: per-stage buffer robustness (stages
    /// that check their own buffer ranges run without the driver's).
    pub pipeline_robustness: bool,
    /// VK_EXT_graphics_pipeline_library with fast linking: graphics
    /// pipelines can be linked from separately compiled stage libraries
    /// (`RPS5E_GPL=0` leaves it off).
    pub graphics_pipeline_library: bool,
    /// VK_EXT_sample_locations: the sample counts (2, 4, 8) whose sample
    /// locations a pipeline may program, on a 1x1 grid at least, with at
    /// least 4 sub-pixel bits over [0, 15/16] (the guest's 1/16 pixel
    /// steps). Empty without the extension.
    pub sample_location_counts: vk::SampleCountFlags,
    /// VK_EXT_shader_stencil_export: pixel programs export the stencil
    /// reference (FragStencilRefEXT).
    pub stencil_export: bool,
    /// VK_EXT_depth_range_unrestricted: viewport depth ranges outside
    /// [0, 1] are taken as the guest sets them (`RPS5E_DEPTH_UNRESTRICTED=0`
    /// clamps them).
    pub depth_range_unrestricted: bool,
    /// VK_KHR_pipeline_executable_properties enabled (`RPS5E_PIPELINE_STATS=1`):
    /// compute pipelines log the driver's statistics (registers, ...).
    pub pipeline_executable_info: bool,
    /// The host driver is AMD's proprietary Vulkan driver
    /// (VkPhysicalDeviceDriverProperties): its compiler turns a
    /// subgroup-scope `OpControlBarrier` into `s_barrier`, which waves in
    /// a branch paired with other waves' barriers hang on, so a wave's LDS
    /// accesses are ordered with a subgroup-scope `OpMemoryBarrier`
    /// instead (and its workgroup-scope memory fences do not order LDS
    /// updates).
    pub amd_proprietary: bool,
}

/// The process-wide (or test-local) Vulkan device.
pub struct GpuDevice {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub phys: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue_family: u32,
    queue: Mutex<vk::Queue>,
    pub props: vk::PhysicalDeviceProperties,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
    pub caps: DeviceCaps,
    pub name: String,
    /// The instance was created with the platform surface extensions.
    pub surface_capable: bool,
    pub push_descriptor: ash::khr::push_descriptor::Device,
    /// Mesh draw commands (when `caps.mesh_shader`).
    pub mesh: Option<ash::ext::mesh_shader::Device>,
    /// Fault queries (when `caps.device_fault`).
    fault: Option<ash::ext::device_fault::Device>,
    /// Checkpoint commands and queries (when `caps.diagnostic_checkpoints`).
    checkpoints: Option<ash::nv::device_diagnostic_checkpoints::Device>,
    /// The queue handle, for the checkpoint query after a loss (it does not
    /// take `queue`: the reporting thread may hold it).
    raw_queue: vk::Queue,
    /// Conditional rendering commands (when `caps.conditional_rendering`).
    pub conditional: Option<ash::ext::conditional_rendering::Device>,
    fault_reported: std::sync::atomic::AtomicBool,
    /// Blocks device-local buffers are placed in (see [`crate::suballoc`]).
    pub arena: crate::suballoc::Arena,
    /// Blocks small images are placed in (`RPS5E_IMAGE_SUBALLOC=1`).
    pub image_arena: crate::suballoc::Arena,
    /// The device shader clock calibrated against the reference clock
    /// (set once by [`crate::clock::calibrate`] when it succeeds).
    pub clock: OnceLock<crate::clock::DeviceClock>,
}

// SAFETY: Vulkan handles are plain identifiers; every externally
// synchronised object reachable from here (the queue) is behind a mutex.
unsafe impl Send for GpuDevice {}
unsafe impl Sync for GpuDevice {}

/// Device features without which guest rendering cannot work (besides the
/// Vulkan 1.3 core features checked first): every interpolated pixel
/// attribute is read per vertex (VK_KHR_fragment_shader_barycentric), and
/// all descriptor sets are pushed (VK_KHR_push_descriptor).
fn missing_required(caps: &DeviceCaps, push_descriptor: bool) -> Vec<&'static str> {
    let mut out = Vec::new();
    if !push_descriptor {
        out.push("VK_KHR_push_descriptor");
    }
    if !caps.fragment_shader_barycentric {
        out.push("VK_KHR_fragment_shader_barycentric (pixel shaders read their attributes per vertex)");
    }
    out
}

/// Optional features the device lacks, with what is lost (logged at start).
pub fn limitations(caps: &DeviceCaps) -> Vec<&'static str> {
    let mut out = Vec::new();
    if !caps.shader_float64 {
        out.push(
            "no shaderFloat64: programs computing in f64 are not run (their draws and dispatches are skipped; \
             Mesa ANV on Arc / Xe-LP / Xe-LPG emulates f64 with the driconf option fp64_workaround_enabled=true)",
        );
    }
    if !caps.mesh_shader {
        out.push("no VK_EXT_mesh_shader: geometry-shader draws and primitive programs exchanging data through LDS are skipped");
    }
    if !caps.shader_demote_to_helper_invocation {
        out.push("no shaderDemoteToHelperInvocation: discarded pixels end their invocation (derivatives after a discard differ)");
    }
    if !caps.shader_image_int64_atomics || !(caps.shader_shared_int64_atomics && caps.workgroup_memory_explicit_layout) {
        out.push("no 64-bit image or workgroup atomics: programs using them are skipped");
    }
    if !caps.compute_derivative_group_quads && !caps.compute_derivative_group_linear {
        out.push("no compute derivatives: implicit-LOD sampling in compute reads LOD 0");
    }
    if caps.bindless_slots == 0 {
        out.push("no descriptor indexing for the bindless set: images selected at run time are not bound");
    }
    if caps.subgroup_size < 32 {
        out.push("subgroups narrower than 32 lanes: guest wave operations need 32");
    }
    if !caps.multi_viewport {
        out.push("fewer than 16 viewports: vertex programs selecting a viewport draw into viewport 0");
    }
    if !caps.texture_compression_bc {
        out.push("no BC texture compression: block-compressed images are decoded on the CPU");
    }
    out
}

/// `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR` (newer than
/// the bundled headers).
const STRUCTURE_TYPE_MAINTENANCE8_FEATURES: i32 = 1_000_574_000;
const MAINTENANCE8_NAME: &CStr = c"VK_KHR_maintenance8";

#[repr(C)]
struct Maintenance8Features {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    maintenance8: vk::Bool32,
}

fn env_disabled() -> bool {
    std::env::var("RPS5E_GPU").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off"))
}

static GLOBAL: OnceLock<Option<Arc<GpuDevice>>> = OnceLock::new();

/// `RPS5E_GPU_CHECKPOINTS=1`: enable VK_NV_device_diagnostic_checkpoints
/// (each dispatch leaves its program address as a checkpoint).
fn checkpoints_requested() -> bool {
    std::env::var("RPS5E_GPU_CHECKPOINTS").is_ok_and(|v| v.trim() == "1")
}

/// One line per checkpoint the queue reached: (marker = program address,
/// pipeline stage bits).
fn checkpoint_lines(points: &[(u64, u64)]) -> Vec<String> {
    if points.is_empty() {
        return vec!["device fault: no checkpoint reached".to_string()];
    }
    points.iter().map(|&(program, stage)| format!("device fault: checkpoint: program {program:#x} at stage {stage:#x}")).collect()
}

fn fault_address_kind(t: vk::DeviceFaultAddressTypeEXT) -> &'static str {
    match t {
        vk::DeviceFaultAddressTypeEXT::NONE => "none",
        vk::DeviceFaultAddressTypeEXT::READ_INVALID => "read invalid",
        vk::DeviceFaultAddressTypeEXT::WRITE_INVALID => "write invalid",
        vk::DeviceFaultAddressTypeEXT::EXECUTE_INVALID => "execute invalid",
        vk::DeviceFaultAddressTypeEXT::INSTRUCTION_POINTER_UNKNOWN => "instruction pointer unknown",
        vk::DeviceFaultAddressTypeEXT::INSTRUCTION_POINTER_INVALID => "instruction pointer invalid",
        vk::DeviceFaultAddressTypeEXT::INSTRUCTION_POINTER_FAULT => "instruction pointer fault",
        _ => "unknown",
    }
}

/// The lines of a VK_EXT_device_fault report: the description, each
/// faulting address with its kind and precision, each vendor record.
fn fault_lines(result: vk::Result, description: &str, addresses: &[(vk::DeviceFaultAddressTypeEXT, u64, u64)], vendor: &[(String, u64, u64)]) -> Vec<String> {
    let mut lines = vec![format!("device fault ({result:?}): \"{description}\", {} address record(s), {} vendor record(s)", addresses.len(), vendor.len())];
    for &(kind, at, precision) in addresses {
        lines.push(format!("device fault: {} at {at:#x} (precision {precision:#x})", fault_address_kind(kind)));
    }
    for (d, code, data) in vendor {
        lines.push(format!("device fault: vendor \"{d}\" (code {code:#x}, data {data:#x})"));
    }
    lines
}

/// A Vulkan call returned `VK_ERROR_DEVICE_LOST` (`what`: where). Nothing
/// the device ran can be recovered, and the guest would only spin on GPU
/// work that never completes: the process shuts down.
/// Most slots of the bindless set's image array.
pub const MAX_BINDLESS_SLOTS: u32 = 1 << 17;

/// Slots of the bindless set's image array within the update-after-bind
/// limits: the set's sampled images count against them in every stage
/// together with a stage's own set (at most 64 images and 32 samplers of
/// a program, two stages of a pipeline sharing one set layout).
fn bindless_slot_capacity(p12: &vk::PhysicalDeviceVulkan12Properties) -> u32 {
    const OWN_IMAGES: u32 = 2 * 64;
    const OWN_RESOURCES: u32 = 2 * (64 + 32 + 32 + 8) + 1;
    let room = |limit: u32, reserve: u32| limit.saturating_sub(reserve);
    let slots = MAX_BINDLESS_SLOTS
        .min(room(p12.max_per_stage_descriptor_update_after_bind_sampled_images, OWN_IMAGES))
        .min(room(p12.max_descriptor_set_update_after_bind_sampled_images, OWN_IMAGES))
        .min(room(p12.max_per_stage_update_after_bind_resources, OWN_RESOURCES))
        .min(room(p12.max_update_after_bind_descriptors_in_all_pools, OWN_RESOURCES));
    // Too few to be worth a set (a table larger than the per-draw slots).
    if slots < 1024 {
        0
    } else {
        slots
    }
}

pub fn device_lost(what: &str) {
    if !LOST.swap(true, std::sync::atomic::Ordering::AcqRel) {
        log::error!("ps5gpu: Vulkan device lost ({what}); stopping GPU work and shutting down (exit code {})", ps5vm::shutdown::EXIT_DEVICE_LOST);
    }
    ps5vm::shutdown::request(ps5vm::shutdown::EXIT_DEVICE_LOST);
}

static LOST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// True once a Vulkan call reported the device lost: nothing is submitted
/// or waited for any more.
pub fn is_lost() -> bool {
    LOST.load(std::sync::atomic::Ordering::Acquire)
}

/// A thread that would submit to or wait on the GPU after the device was
/// lost, or after shutdown started, stops here for good (the process is
/// ending on the shutdown thread).
pub fn stop_if_lost_or_shutting_down() {
    if is_lost() || ps5vm::shutdown::requested() {
        loop {
            std::thread::park();
        }
    }
}

impl GpuDevice {
    /// The device was lost: log what the driver reports about it
    /// (VK_EXT_device_fault, once per device) and end the process.
    pub fn report_device_fault(&self) {
        self.log_device_fault();
        device_lost("queue");
    }

    /// The GPU stopped completing work without the driver reporting a
    /// loss: handled as one (fault report, shutdown).
    pub fn report_hang(&self) {
        self.log_device_fault();
        device_lost("GPU hang");
    }

    fn log_device_fault(&self) {
        if self.fault_reported.swap(true, std::sync::atomic::Ordering::AcqRel) {
            return;
        }
        for line in self.checkpoint_report().into_iter().chain(self.fault_report()) {
            log::error!("ps5gpu: {line}");
        }
    }

    /// Leave `marker` (a program address) as a checkpoint in `cb` when
    /// checkpoints are enabled: after a loss the queue reports the last
    /// markers the GPU reached.
    pub fn checkpoint(&self, cb: vk::CommandBuffer, marker: u64) {
        if let Some(c) = &self.checkpoints {
            // SAFETY: recording command buffer; the marker is an opaque value
            // the driver never dereferences.
            unsafe { c.cmd_set_checkpoint(cb, marker as usize as *const c_void) };
        }
    }

    /// The checkpoints the queue reached before the loss (empty when they
    /// are not enabled).
    fn checkpoint_report(&self) -> Vec<String> {
        let Some(c) = &self.checkpoints else { return Vec::new() };
        // SAFETY: valid queue; the array holds the count the driver reported.
        let points: Vec<(u64, u64)> = unsafe {
            let n = c.get_queue_checkpoint_data_len(self.raw_queue);
            let mut data = vec![vk::CheckpointDataNV::default(); n];
            if n != 0 {
                c.get_queue_checkpoint_data(self.raw_queue, &mut data);
            }
            data.iter().map(|d| (d.p_checkpoint_marker as usize as u64, u64::from(d.stage.as_raw()))).collect()
        };
        checkpoint_lines(&points)
    }

    /// What VK_EXT_device_fault reports about the loss (empty without it).
    fn fault_report(&self) -> Vec<String> {
        let Some(f) = &self.fault else { return Vec::new() };
        let fp = f.fp();
        let mut counts = vk::DeviceFaultCountsEXT::default();
        // SAFETY: valid device; a null info pointer queries the counts.
        let r = unsafe { (fp.get_device_fault_info_ext)(self.device.handle(), &mut counts, std::ptr::null_mut()) };
        if r != vk::Result::SUCCESS && r != vk::Result::INCOMPLETE {
            return vec![format!("device fault query failed: {r:?}")];
        }
        let mut addrs = vec![vk::DeviceFaultAddressInfoEXT::default(); counts.address_info_count as usize];
        let mut vendors = vec![vk::DeviceFaultVendorInfoEXT::default(); counts.vendor_info_count as usize];
        let mut info = vk::DeviceFaultInfoEXT {
            p_address_infos: addrs.as_mut_ptr(),
            p_vendor_infos: vendors.as_mut_ptr(),
            ..Default::default()
        };
        counts.vendor_binary_size = 0;
        // SAFETY: the arrays hold the counts the driver reported.
        let r = unsafe { (fp.get_device_fault_info_ext)(self.device.handle(), &mut counts, &mut info) };
        let text = |d: &[std::ffi::c_char]| {
            // Bounded by the array: a description need not be terminated.
            let bytes: Vec<u8> = d.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
            String::from_utf8_lossy(&bytes).into_owned()
        };
        let addresses: Vec<(vk::DeviceFaultAddressTypeEXT, u64, u64)> =
            addrs.iter().take(counts.address_info_count as usize).map(|a| (a.address_type, a.reported_address, a.address_precision)).collect();
        let vendor: Vec<(String, u64, u64)> =
            vendors.iter().take(counts.vendor_info_count as usize).map(|v| (text(&v.description), v.vendor_fault_code, v.vendor_fault_data)).collect();
        fault_lines(r, &text(&info.description), &addresses, &vendor)
    }

    /// The process-wide device, created on first use. `None` when disabled
    /// or when no suitable device exists.
    pub fn global() -> Option<&'static Arc<GpuDevice>> {
        GLOBAL
            .get_or_init(|| {
                if env_disabled() {
                    log::info!("ps5gpu: Vulkan device disabled (RPS5E_GPU=0)");
                    return None;
                }
                let surface = crate::display_available() && !crate::env_headless();
                match GpuDevice::create(surface) {
                    Ok(d) => {
                        log::info!(
                            "ps5gpu: Vulkan device '{}' (family {}, surface={}) caps: {:?}",
                            d.name,
                            d.queue_family,
                            d.surface_capable,
                            d.caps
                        );
                        Some(d)
                    }
                    Err(e) => {
                        log::warn!("ps5gpu: no usable Vulkan device: {e}");
                        None
                    }
                }
            })
            .as_ref()
    }

    /// True once the global device was created successfully.
    pub fn global_ready() -> bool {
        GLOBAL.get().is_some_and(|d| d.is_some())
    }

    /// Create a device. With `want_surface` the platform surface instance
    /// extensions are enabled (when the loader has them).
    pub fn create(want_surface: bool) -> Result<Arc<GpuDevice>, String> {
        // One creation at a time: drivers fail instance and device creation
        // racing in one process (ERROR_INITIALIZATION_FAILED with the device
        // tests in parallel).
        static CREATING: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _creating = CREATING.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: loading the system Vulkan loader.
        let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("loader: {e}"))?;
        // SAFETY: plain enumeration on a valid entry.
        let available: Vec<vk::ExtensionProperties> =
            unsafe { entry.enumerate_instance_extension_properties(None) }.map_err(|e| format!("{e:?}"))?;
        let has = |name: &CStr| available.iter().any(|e| e.extension_name_as_c_str() == Ok(name));
        let mut inst_exts: Vec<*const i8> = Vec::new();
        let mut surface_capable = false;
        if want_surface && has(ash::khr::surface::NAME) {
            inst_exts.push(ash::khr::surface::NAME.as_ptr());
            for name in [ash::khr::xlib_surface::NAME, ash::khr::xcb_surface::NAME, ash::khr::wayland_surface::NAME, ash::khr::win32_surface::NAME] {
                if has(name) {
                    inst_exts.push(name.as_ptr());
                    surface_capable = true;
                }
            }
            // HDR output: surfaces then also list their HDR10 formats.
            if surface_capable && crate::display::hdr_enabled() && has(ash::ext::swapchain_colorspace::NAME) {
                inst_exts.push(ash::ext::swapchain_colorspace::NAME.as_ptr());
            }
        }
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"rps5e")
            .engine_name(c"rps5e")
            .api_version(vk::API_VERSION_1_3);
        let ici = vk::InstanceCreateInfo::default().application_info(&app_info).enabled_extension_names(&inst_exts);
        // SAFETY: valid create info; destroyed on every error path below and in Drop.
        let instance = unsafe { entry.create_instance(&ici, None) }.map_err(|e| format!("instance: {e:?}"))?;
        match Self::create_on(entry.clone(), instance.clone(), surface_capable) {
            Ok(d) => {
                let d = Arc::new(d);
                // SAFETY: query on the device's physical device.
                let d16s8 = unsafe { d.instance.get_physical_device_format_properties(d.phys, vk::Format::D16_UNORM_S8_UINT) };
                let renders = d16s8.optimal_tiling_features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT);
                crate::render::formats::DEPTH16_STENCIL.store(renders, std::sync::atomic::Ordering::Relaxed);
                if !renders {
                    log::info!("ps5gpu: no D16_UNORM_S8_UINT attachments: 16-bit depth with stencil uses D32_SFLOAT_S8_UINT");
                }
                crate::clock::calibrate(&d);
                Ok(d)
            }
            Err(e) => {
                // SAFETY: no child objects survive a failed create_on.
                unsafe { instance.destroy_instance(None) };
                Err(e)
            }
        }
    }

    fn select_physical(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32), String> {
        // SAFETY: enumeration on a valid instance.
        let devices = unsafe { instance.enumerate_physical_devices() }.map_err(|e| format!("{e:?}"))?;
        let wanted = std::env::var("RPS5E_GPU_DEVICE").ok();
        let mut best: Option<(i32, vk::PhysicalDevice, u32)> = None;
        for (index, pd) in devices.into_iter().enumerate() {
            // SAFETY: valid physical device handle.
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let name = props.device_name_as_c_str().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            if props.api_version < vk::API_VERSION_1_3 {
                continue;
            }
            if let Some(w) = &wanted {
                let matches = w.parse::<usize>().map_or_else(|_| name.to_lowercase().contains(&w.to_lowercase()), |i| i == index);
                if !matches {
                    continue;
                }
            }
            // SAFETY: valid physical device handle.
            let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            let Some(qf) = families
                .iter()
                .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE))
            else {
                continue;
            };
            // SAFETY: valid physical device handle.
            let exts = unsafe { instance.enumerate_device_extension_properties(pd) }.unwrap_or_default();
            if !exts.iter().any(|e| e.extension_name_as_c_str() == Ok(ash::khr::push_descriptor::NAME)) {
                log::info!("ps5gpu: skipping '{name}': no VK_KHR_push_descriptor");
                continue;
            }
            let score = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 4,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::CPU => 0,
                _ => 1,
            };
            if best.is_none_or(|b| score > b.0) {
                best = Some((score, pd, qf as u32));
            }
        }
        best.map(|(_, pd, qf)| (pd, qf)).ok_or_else(|| "no Vulkan 1.3 device with graphics+compute and push descriptors".into())
    }

    fn create_on(entry: ash::Entry, instance: ash::Instance, surface_capable: bool) -> Result<GpuDevice, String> {
        let (phys, queue_family) = Self::select_physical(&instance)?;
        // SAFETY: queries on a valid physical device.
        let (props, mem_props, exts, core) = unsafe {
            (
                instance.get_physical_device_properties(phys),
                instance.get_physical_device_memory_properties(phys),
                instance.enumerate_device_extension_properties(phys).map_err(|e| format!("{e:?}"))?,
                instance.get_physical_device_features(phys),
            )
        };
        let name = props.device_name_as_c_str().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let has = |n: &CStr| exts.iter().any(|e| e.extension_name_as_c_str() == Ok(n));

        // ---- feature query ----
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut robust2 = vk::PhysicalDeviceRobustness2FeaturesEXT::default();
        let mut maint5 = vk::PhysicalDeviceMaintenance5FeaturesKHR::default();
        let mut cwe = vk::PhysicalDeviceColorWriteEnableFeaturesEXT::default();
        let mut eds = vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default();
        let want_eds = has(ash::ext::extended_dynamic_state::NAME) && std::env::var("RPS5E_EXT_DYNAMIC_STATE").map_or(true, |v| v != "0");
        let mut dce = vk::PhysicalDeviceDepthClipEnableFeaturesEXT::default();
        let mut dcc = vk::PhysicalDeviceDepthClipControlFeaturesEXT::default();
        let mut bary = vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR::default();
        let mut interlock = vk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT::default();
        let mut minlod = vk::PhysicalDeviceImageViewMinLodFeaturesEXT::default();
        let mut meshf = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
        let mut faultf = vk::PhysicalDeviceFaultFeaturesEXT::default();
        let mut condf = vk::PhysicalDeviceConditionalRenderingFeaturesEXT::default();
        let mut img64 = vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT::default();
        let mut wgl = vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR::default();
        let mut derivs = vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV::default();
        let mut clockf = vk::PhysicalDeviceShaderClockFeaturesKHR::default();
        let mut provokingf = vk::PhysicalDeviceProvokingVertexFeaturesEXT::default();
        let mut robustf = vk::PhysicalDevicePipelineRobustnessFeaturesEXT::default();
        let mut execf = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default();
        let mut gplf = vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT::default();
        let want_gpl = has(ash::ext::graphics_pipeline_library::NAME) && has(ash::khr::pipeline_library::NAME) && crate::pipeline::library::requested();
        let want_exec = std::env::var("RPS5E_PIPELINE_STATS").is_ok_and(|v| v == "1") && has(ash::khr::pipeline_executable_properties::NAME);
        let mut maint8 = Maintenance8Features {
            s_type: vk::StructureType::from_raw(STRUCTURE_TYPE_MAINTENANCE8_FEATURES),
            p_next: std::ptr::null_mut(),
            maintenance8: 0,
        };
        {
            let mut q = vk::PhysicalDeviceFeatures2::default().push_next(&mut f13).push_next(&mut f12).push_next(&mut f11);
            if has(ash::ext::robustness2::NAME) {
                q = q.push_next(&mut robust2);
            }
            if has(ash::khr::maintenance5::NAME) {
                q = q.push_next(&mut maint5);
            }
            if has(ash::ext::color_write_enable::NAME) {
                q = q.push_next(&mut cwe);
            }
            if want_eds {
                q = q.push_next(&mut eds);
            }
            if has(ash::ext::depth_clip_enable::NAME) {
                q = q.push_next(&mut dce);
            }
            if has(ash::ext::depth_clip_control::NAME) {
                q = q.push_next(&mut dcc);
            }
            if has(ash::khr::fragment_shader_barycentric::NAME) {
                q = q.push_next(&mut bary);
            }
            if has(ash::ext::fragment_shader_interlock::NAME) {
                q = q.push_next(&mut interlock);
            }
            if has(ash::ext::image_view_min_lod::NAME) {
                q = q.push_next(&mut minlod);
            }
            if has(ash::ext::mesh_shader::NAME) {
                q = q.push_next(&mut meshf);
            }
            if has(ash::ext::device_fault::NAME) {
                q = q.push_next(&mut faultf);
            }
            if has(ash::ext::conditional_rendering::NAME) {
                q = q.push_next(&mut condf);
            }
            if has(ash::ext::shader_image_atomic_int64::NAME) {
                q = q.push_next(&mut img64);
            }
            if has(ash::khr::workgroup_memory_explicit_layout::NAME) {
                q = q.push_next(&mut wgl);
            }
            if has(ash::nv::compute_shader_derivatives::NAME) {
                q = q.push_next(&mut derivs);
            }
            if has(ash::khr::shader_clock::NAME) {
                q = q.push_next(&mut clockf);
            }
            if has(ash::ext::provoking_vertex::NAME) {
                q = q.push_next(&mut provokingf);
            }
            if has(ash::ext::pipeline_robustness::NAME) {
                q = q.push_next(&mut robustf);
            }
            if want_exec {
                q = q.push_next(&mut execf);
            }
            if want_gpl {
                q = q.push_next(&mut gplf);
            }
            if has(MAINTENANCE8_NAME) {
                maint8.p_next = q.p_next;
                q.p_next = &mut maint8 as *mut _ as *mut c_void;
            }
            // SAFETY: the chain only holds structures that live in this scope.
            unsafe { instance.get_physical_device_features2(phys, &mut q) };
        }
        let require = |ok: bool, what: &str| if ok { Ok(()) } else { Err(format!("'{name}' lacks {what}")) };
        require(f13.dynamic_rendering != 0, "Vulkan 1.3 dynamicRendering")?;
        require(f13.synchronization2 != 0, "Vulkan 1.3 synchronization2")?;
        require(f12.timeline_semaphore != 0, "timelineSemaphore")?;
        require(f12.buffer_device_address != 0, "bufferDeviceAddress")?;
        require(core.shader_int64 != 0, "shaderInt64")?;

        let mut caps = DeviceCaps {
            swapchain: has(ash::khr::swapchain::NAME),
            maintenance5: maint5.maintenance5 != 0,
            maintenance8: maint8.maintenance8 != 0,
            robust_buffer_access2: robust2.robust_buffer_access2 != 0 && core.robust_buffer_access != 0,
            robust_image_access2: robust2.robust_image_access2 != 0,
            null_descriptor: robust2.null_descriptor != 0,
            color_write_enable: cwe.color_write_enable != 0,
            depth_clip_enable: dce.depth_clip_enable != 0,
            depth_clip_control: dcc.depth_clip_control != 0,
            extended_dynamic_state: eds.extended_dynamic_state != 0,
            fragment_shader_barycentric: bary.fragment_shader_barycentric != 0,
            fragment_shader_pixel_interlock: interlock.fragment_shader_pixel_interlock != 0,
            image_view_min_lod: minlod.min_lod != 0,
            fill_rectangle: has(ash::nv::fill_rectangle::NAME),
            external_memory_host: has(ash::ext::external_memory_host::NAME),
            min_imported_host_pointer_alignment: 0,
            device_fault: faultf.device_fault != 0,
            diagnostic_checkpoints: checkpoints_requested() && has(ash::nv::device_diagnostic_checkpoints::NAME),
            conditional_rendering: condf.conditional_rendering != 0,
            memory_budget: has(ash::ext::memory_budget::NAME),
            occlusion_query_precise: core.occlusion_query_precise != 0,
            host_query_reset: f12.host_query_reset != 0,
            mesh_shader: meshf.mesh_shader != 0 && !std::env::var("RPS5E_MESH").is_ok_and(|v| v == "0"),
            max_mesh_output_vertices: 0,
            max_mesh_output_primitives: 0,
            max_mesh_work_group_invocations: 0,
            max_mesh_shared_memory: 0,
            max_mesh_work_group_count: [0; 3],
            max_mesh_work_group_total_count: 0,
            // Promoted to Vulkan 1.2 features, which must be enabled with it.
            shader_viewport_index_layer: has(ash::ext::shader_viewport_index_layer::NAME)
                && f12.shader_output_viewport_index != 0
                && f12.shader_output_layer != 0,
            shader_float64: core.shader_float64 != 0,
            shader_int16: core.shader_int16 != 0,
            shader_float16: f12.shader_float16 != 0,
            shader_demote_to_helper_invocation: f13.shader_demote_to_helper_invocation != 0,
            shader_shared_int64_atomics: f12.shader_shared_int64_atomics != 0,
            shader_image_int64_atomics: img64.shader_image_int64_atomics != 0,
            workgroup_memory_explicit_layout: wgl.workgroup_memory_explicit_layout != 0,
            compute_derivative_group_quads: derivs.compute_derivative_group_quads != 0,
            compute_derivative_group_linear: derivs.compute_derivative_group_linear != 0,
            multi_viewport: core.multi_viewport != 0 && props.limits.max_viewports >= 16 && f12.shader_output_viewport_index != 0,
            pipeline_creation_cache_control: f13.pipeline_creation_cache_control != 0,
            subgroup_size_control: f13.subgroup_size_control != 0,
            subgroup_size: 32,
            min_compute_subgroup_size: 0,
            max_compute_subgroup_size: 0,
            mesh_subgroup32: false,
            max_compute_workgroup_subgroups: 0,
            max_compute_shared_memory: props.limits.max_compute_shared_memory_size,
            independent_blend: core.independent_blend != 0,
            depth_bounds: core.depth_bounds != 0,
            depth_bias_clamp: core.depth_bias_clamp != 0,
            texture_compression_bc: core.texture_compression_bc != 0,
            sample_rate_shading: core.sample_rate_shading != 0,
            tessellation_shader: core.tessellation_shader != 0,
            draw_indirect_count: f12.draw_indirect_count != 0,
            shader_draw_parameters: f11.shader_draw_parameters != 0,
            multi_draw_indirect: core.multi_draw_indirect != 0,
            fill_mode_non_solid: core.fill_mode_non_solid != 0,
            dual_src_blend: core.dual_src_blend != 0,
            depth_clamp: core.depth_clamp != 0,
            max_storage_buffer_range: props.limits.max_storage_buffer_range,
            timestamp_period: if props.limits.timestamp_compute_and_graphics != 0 { props.limits.timestamp_period } else { 0.0 },
            min_storage_buffer_offset_alignment: props.limits.min_storage_buffer_offset_alignment.max(4),
            min_uniform_buffer_offset_alignment: props.limits.min_uniform_buffer_offset_alignment.max(4),
            max_push_descriptors: 0,
            max_framebuffer_width: props.limits.max_framebuffer_width,
            max_framebuffer_height: props.limits.max_framebuffer_height,
            non_coherent_atom_size: props.limits.non_coherent_atom_size.max(1),
            signed_zero_inf_nan_preserve: [false; 3],
            sampler_filter_minmax: f12.sampler_filter_minmax != 0,
            denorm_preserve: [false; 3],
            denorm_flush_to_zero32: false,
            float16_rte: false,
            shader_device_clock: clockf.shader_device_clock != 0,
            shader_subgroup_clock: clockf.shader_subgroup_clock != 0,
            provoking_vertex_last: provokingf.provoking_vertex_last != 0,
            provoking_vertex_per_pipeline: false,
            subgroup_broadcast_dynamic_id: f12.subgroup_broadcast_dynamic_id != 0,
            subgroup_shuffle_relative: false,
            subgroup_arithmetic: false,
            bindless_slots: 0,
            pipeline_robustness: robustf.pipeline_robustness != 0 && core.robust_buffer_access != 0,
            graphics_pipeline_library: gplf.graphics_pipeline_library != 0,
            sample_location_counts: vk::SampleCountFlags::empty(),
            stencil_export: has(ash::ext::shader_stencil_export::NAME),
            depth_range_unrestricted: has(ash::ext::depth_range_unrestricted::NAME) && !std::env::var("RPS5E_DEPTH_UNRESTRICTED").is_ok_and(|v| v == "0"),
            pipeline_executable_info: want_exec && execf.pipeline_executable_info != 0,
            amd_proprietary: false,
        };
        if has(ash::ext::sample_locations::NAME) {
            caps.sample_location_counts = sample_location_counts(&entry, &instance, phys);
        }
        {
            let mut pd = vk::PhysicalDevicePushDescriptorPropertiesKHR::default();
            let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
            let mut fc = vk::PhysicalDeviceFloatControlsProperties::default();
            let mut p13 = vk::PhysicalDeviceVulkan13Properties::default();
            let mut p12 = vk::PhysicalDeviceVulkan12Properties::default();
            let mut dp = vk::PhysicalDeviceDriverProperties::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut pd).push_next(&mut p11).push_next(&mut fc).push_next(&mut p13).push_next(&mut p12).push_next(&mut dp);
            // SAFETY: valid chain on the stack.
            unsafe { instance.get_physical_device_properties2(phys, &mut p2) };
            caps.max_push_descriptors = pd.max_push_descriptors;
            caps.amd_proprietary = dp.driver_id == vk::DriverId::AMD_PROPRIETARY;
            let indexing = f12.runtime_descriptor_array != 0
                && f12.shader_sampled_image_array_non_uniform_indexing != 0
                && f12.descriptor_binding_partially_bound != 0
                && f12.descriptor_binding_sampled_image_update_after_bind != 0
                && f12.descriptor_binding_update_unused_while_pending != 0;
            if indexing {
                caps.bindless_slots = bindless_slot_capacity(&p12);
            }
            caps.subgroup_size = p11.subgroup_size.max(1);
            if caps.subgroup_size_control && p13.required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE) {
                caps.min_compute_subgroup_size = p13.min_subgroup_size;
                caps.max_compute_subgroup_size = p13.max_subgroup_size;
                caps.max_compute_workgroup_subgroups = p13.max_compute_workgroup_subgroups;
            }
            caps.mesh_subgroup32 = caps.subgroup_size_control
                && p13.min_subgroup_size < 32
                && p13.max_subgroup_size >= 32
                && p13.required_subgroup_size_stages.contains(vk::ShaderStageFlags::MESH_EXT);
            caps.subgroup_shuffle_relative = p11.subgroup_supported_operations.contains(vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE);
            caps.subgroup_arithmetic = p11.subgroup_supported_operations.contains(vk::SubgroupFeatureFlags::ARITHMETIC)
                && p11.subgroup_supported_stages.contains(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);
            caps.signed_zero_inf_nan_preserve = [
                fc.shader_signed_zero_inf_nan_preserve_float16 != 0,
                fc.shader_signed_zero_inf_nan_preserve_float32 != 0,
                fc.shader_signed_zero_inf_nan_preserve_float64 != 0,
            ];
            let preserve = [fc.shader_denorm_preserve_float16 != 0, fc.shader_denorm_preserve_float32 != 0, fc.shader_denorm_preserve_float64 != 0];
            caps.denorm_preserve = denorm_preserve_caps(fc.denorm_behavior_independence, preserve);
            caps.denorm_flush_to_zero32 = fc.shader_denorm_flush_to_zero_float32 != 0;
            caps.float16_rte = caps.shader_float16 && fc.shader_rounding_mode_rte_float16 != 0 && caps.denorm_preserve[0] && caps.signed_zero_inf_nan_preserve[0];
        }
        if caps.mesh_shader {
            let mut mp = vk::PhysicalDeviceMeshShaderPropertiesEXT::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut mp);
            // SAFETY: valid chain on the stack.
            unsafe { instance.get_physical_device_properties2(phys, &mut p2) };
            caps.max_mesh_output_vertices = mp.max_mesh_output_vertices;
            caps.max_mesh_output_primitives = mp.max_mesh_output_primitives;
            caps.max_mesh_work_group_invocations = mp.max_mesh_work_group_invocations;
            caps.max_mesh_shared_memory = mp.max_mesh_shared_memory_size;
            caps.max_mesh_work_group_count = mp.max_mesh_work_group_count;
            caps.max_mesh_work_group_total_count = mp.max_mesh_work_group_total_count;
        }
        if caps.provoking_vertex_last {
            let mut pp = vk::PhysicalDeviceProvokingVertexPropertiesEXT::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut pp);
            // SAFETY: valid chain on the stack.
            unsafe { instance.get_physical_device_properties2(phys, &mut p2) };
            caps.provoking_vertex_per_pipeline = pp.provoking_vertex_mode_per_pipeline != 0;
        }
        if caps.graphics_pipeline_library {
            let mut gp = vk::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut gp);
            // SAFETY: valid chain on the stack.
            unsafe { instance.get_physical_device_properties2(phys, &mut p2) };
            // Without fast linking a link costs as much as a full build.
            caps.graphics_pipeline_library = gp.graphics_pipeline_library_fast_linking != 0;
        }
        if caps.external_memory_host {
            let mut hp = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut hp);
            // SAFETY: valid chain on the stack.
            unsafe { instance.get_physical_device_properties2(phys, &mut p2) };
            caps.min_imported_host_pointer_alignment = hp.min_imported_host_pointer_alignment;
        }

        let missing = missing_required(&caps, has(ash::khr::push_descriptor::NAME));
        if !missing.is_empty() {
            let msg = format!("'{name}' lacks {}", missing.join(", "));
            if !std::env::var("RPS5E_GPU_IGNORE_MISSING").is_ok_and(|v| v == "1") {
                return Err(format!("{msg} (RPS5E_GPU_IGNORE_MISSING=1 runs anyway)"));
            }
            log::error!("ps5gpu: {msg}; continuing (RPS5E_GPU_IGNORE_MISSING=1)");
        }
        for l in limitations(&caps) {
            log::warn!("ps5gpu: '{name}': {l}");
        }

        // ---- device creation ----
        let mut dev_exts: Vec<*const i8> = vec![ash::khr::push_descriptor::NAME.as_ptr()];
        if caps.swapchain && surface_capable {
            dev_exts.push(ash::khr::swapchain::NAME.as_ptr());
        } else {
            caps.swapchain = false;
        }
        crate::pipeline::state::set_depth_range_unrestricted(caps.depth_range_unrestricted);
        let optional: [(bool, &CStr); 30] = [
            (caps.graphics_pipeline_library, ash::khr::pipeline_library::NAME),
            (caps.graphics_pipeline_library, ash::ext::graphics_pipeline_library::NAME),
            (!caps.sample_location_counts.is_empty(), ash::ext::sample_locations::NAME),
            (caps.depth_range_unrestricted, ash::ext::depth_range_unrestricted::NAME),
            (caps.stencil_export, ash::ext::shader_stencil_export::NAME),
            (caps.pipeline_executable_info, ash::khr::pipeline_executable_properties::NAME),
            (caps.pipeline_robustness, ash::ext::pipeline_robustness::NAME),
            (caps.provoking_vertex_last, ash::ext::provoking_vertex::NAME),
            (caps.shader_device_clock || caps.shader_subgroup_clock, ash::khr::shader_clock::NAME),
            (caps.memory_budget, ash::ext::memory_budget::NAME),
            (caps.shader_image_int64_atomics, ash::ext::shader_image_atomic_int64::NAME),
            (caps.workgroup_memory_explicit_layout, ash::khr::workgroup_memory_explicit_layout::NAME),
            (caps.compute_derivative_group_quads || caps.compute_derivative_group_linear, ash::nv::compute_shader_derivatives::NAME),
            (caps.device_fault, ash::ext::device_fault::NAME),
            (caps.diagnostic_checkpoints, ash::nv::device_diagnostic_checkpoints::NAME),
            (caps.conditional_rendering, ash::ext::conditional_rendering::NAME),
            (caps.mesh_shader, ash::ext::mesh_shader::NAME),
            (caps.external_memory_host, ash::ext::external_memory_host::NAME),
            (caps.shader_viewport_index_layer, ash::ext::shader_viewport_index_layer::NAME),
            (caps.fragment_shader_barycentric, ash::khr::fragment_shader_barycentric::NAME),
            (caps.fragment_shader_pixel_interlock, ash::ext::fragment_shader_interlock::NAME),
            (caps.color_write_enable, ash::ext::color_write_enable::NAME),
            (caps.extended_dynamic_state, ash::ext::extended_dynamic_state::NAME),
            (caps.depth_clip_control, ash::ext::depth_clip_control::NAME),
            (caps.depth_clip_enable, ash::ext::depth_clip_enable::NAME),
            (caps.maintenance8, MAINTENANCE8_NAME),
            (caps.maintenance5, ash::khr::maintenance5::NAME),
            (caps.image_view_min_lod, ash::ext::image_view_min_lod::NAME),
            (caps.fill_rectangle, ash::nv::fill_rectangle::NAME),
            (caps.robust_image_access2 || caps.null_descriptor, ash::ext::robustness2::NAME),
        ];
        for (on, n) in optional {
            if on {
                dev_exts.push(n.as_ptr());
            }
        }

        let enabled_core = vk::PhysicalDeviceFeatures {
            depth_bounds: core.depth_bounds,
            independent_blend: core.independent_blend,
            vertex_pipeline_stores_and_atomics: core.vertex_pipeline_stores_and_atomics,
            fragment_stores_and_atomics: core.fragment_stores_and_atomics,
            shader_int64: core.shader_int64,
            shader_float64: core.shader_float64,
            shader_int16: core.shader_int16,
            shader_clip_distance: core.shader_clip_distance,
            shader_cull_distance: core.shader_cull_distance,
            shader_image_gather_extended: core.shader_image_gather_extended,
            // Bindless-table images index their binding array at run time.
            shader_sampled_image_array_dynamic_indexing: core.shader_sampled_image_array_dynamic_indexing,
            shader_storage_image_array_dynamic_indexing: core.shader_storage_image_array_dynamic_indexing,
            shader_storage_image_extended_formats: core.shader_storage_image_extended_formats,
            shader_storage_image_read_without_format: core.shader_storage_image_read_without_format,
            shader_storage_image_write_without_format: core.shader_storage_image_write_without_format,
            texture_compression_bc: core.texture_compression_bc,
            robust_buffer_access: core.robust_buffer_access,
            depth_bias_clamp: core.depth_bias_clamp,
            sample_rate_shading: core.sample_rate_shading,
            sampler_anisotropy: core.sampler_anisotropy,
            multi_draw_indirect: core.multi_draw_indirect,
            draw_indirect_first_instance: core.draw_indirect_first_instance,
            fill_mode_non_solid: core.fill_mode_non_solid,
            wide_lines: core.wide_lines,
            large_points: core.large_points,
            geometry_shader: core.geometry_shader,
            tessellation_shader: core.tessellation_shader,
            image_cube_array: core.image_cube_array,
            dual_src_blend: core.dual_src_blend,
            logic_op: core.logic_op,
            depth_clamp: core.depth_clamp,
            multi_viewport: core.multi_viewport,
            occlusion_query_precise: core.occlusion_query_precise,
            ..Default::default()
        };
        let mut e13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true)
            .subgroup_size_control(caps.subgroup_size_control)
            .pipeline_creation_cache_control(caps.pipeline_creation_cache_control)
            .shader_demote_to_helper_invocation(caps.shader_demote_to_helper_invocation)
            .maintenance4(f13.maintenance4 != 0);
        let mut e12 = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .buffer_device_address(true)
            // Guest samplers use MIRROR_ONCE clamp modes.
            .sampler_mirror_clamp_to_edge(f12.sampler_mirror_clamp_to_edge != 0)
            // Guest samplers select min/max filtering (FILTER_MODE).
            .sampler_filter_minmax(caps.sampler_filter_minmax)
            .shader_shared_int64_atomics(caps.shader_shared_int64_atomics)
            .shader_buffer_int64_atomics(f12.shader_buffer_int64_atomics != 0)
            .scalar_block_layout(f12.scalar_block_layout != 0)
            .storage_buffer8_bit_access(f12.storage_buffer8_bit_access != 0)
            .uniform_and_storage_buffer8_bit_access(f12.uniform_and_storage_buffer8_bit_access != 0)
            .shader_int8(f12.shader_int8 != 0)
            .shader_float16(f12.shader_float16 != 0)
            .descriptor_indexing(f12.descriptor_indexing != 0)
            .runtime_descriptor_array(f12.runtime_descriptor_array != 0)
            .shader_sampled_image_array_non_uniform_indexing(f12.shader_sampled_image_array_non_uniform_indexing != 0)
            .shader_storage_buffer_array_non_uniform_indexing(f12.shader_storage_buffer_array_non_uniform_indexing != 0)
            // The bindless set (images selected through the host table map).
            .descriptor_binding_partially_bound(caps.bindless_slots != 0)
            .descriptor_binding_sampled_image_update_after_bind(caps.bindless_slots != 0)
            .descriptor_binding_update_unused_while_pending(caps.bindless_slots != 0)
            .vulkan_memory_model(f12.vulkan_memory_model != 0)
            .vulkan_memory_model_device_scope(f12.vulkan_memory_model_device_scope != 0)
            .host_query_reset(caps.host_query_reset)
            .draw_indirect_count(f12.draw_indirect_count != 0)
            .shader_output_viewport_index(caps.shader_viewport_index_layer)
            .shader_output_layer(caps.shader_viewport_index_layer)
            .subgroup_broadcast_dynamic_id(caps.subgroup_broadcast_dynamic_id);
        let mut erobust = vk::PhysicalDevicePipelineRobustnessFeaturesEXT::default().pipeline_robustness(true);
        let mut eexec = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default().pipeline_executable_info(true);
        let mut egpl = vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT::default().graphics_pipeline_library(true);
        let mut er2 = vk::PhysicalDeviceRobustness2FeaturesEXT::default()
            .robust_buffer_access2(caps.robust_buffer_access2)
            .robust_image_access2(caps.robust_image_access2)
            .null_descriptor(caps.null_descriptor);
        let mut em5 = vk::PhysicalDeviceMaintenance5FeaturesKHR::default().maintenance5(true);
        let mut ecwe = vk::PhysicalDeviceColorWriteEnableFeaturesEXT::default().color_write_enable(true);
        let mut eeds = vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default().extended_dynamic_state(true);
        let mut edce = vk::PhysicalDeviceDepthClipEnableFeaturesEXT::default().depth_clip_enable(true);
        let mut edcc = vk::PhysicalDeviceDepthClipControlFeaturesEXT::default().depth_clip_control(true);
        let mut ebary = vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR::default().fragment_shader_barycentric(true);
        let mut einterlock = vk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT::default().fragment_shader_pixel_interlock(true);
        let mut eminlod = vk::PhysicalDeviceImageViewMinLodFeaturesEXT::default().min_lod(true);
        let mut emesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default().mesh_shader(true);
        let mut efault = vk::PhysicalDeviceFaultFeaturesEXT::default().device_fault(true);
        let mut econd = vk::PhysicalDeviceConditionalRenderingFeaturesEXT::default().conditional_rendering(true);
        let mut eimg64 = vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT::default().shader_image_int64_atomics(true);
        let mut ewgl = vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR::default().workgroup_memory_explicit_layout(true);
        let mut ederivs = vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV::default()
            .compute_derivative_group_quads(caps.compute_derivative_group_quads)
            .compute_derivative_group_linear(caps.compute_derivative_group_linear);
        let mut eprovoking = vk::PhysicalDeviceProvokingVertexFeaturesEXT::default().provoking_vertex_last(true);
        let mut eclock = vk::PhysicalDeviceShaderClockFeaturesKHR::default()
            .shader_device_clock(caps.shader_device_clock)
            .shader_subgroup_clock(caps.shader_subgroup_clock);
        let mut em8 = Maintenance8Features {
            s_type: vk::StructureType::from_raw(STRUCTURE_TYPE_MAINTENANCE8_FEATURES),
            p_next: std::ptr::null_mut(),
            maintenance8: 1,
        };
        let mut e11 = vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(caps.shader_draw_parameters);
        let mut features2 = vk::PhysicalDeviceFeatures2::default().features(enabled_core).push_next(&mut e13).push_next(&mut e12).push_next(&mut e11);
        if caps.robust_image_access2 || caps.null_descriptor {
            features2 = features2.push_next(&mut er2);
        }
        if caps.maintenance5 {
            features2 = features2.push_next(&mut em5);
        }
        if caps.color_write_enable {
            features2 = features2.push_next(&mut ecwe);
        }
        if caps.extended_dynamic_state {
            features2 = features2.push_next(&mut eeds);
        }
        if caps.depth_clip_enable {
            features2 = features2.push_next(&mut edce);
        }
        if caps.depth_clip_control {
            features2 = features2.push_next(&mut edcc);
        }
        if caps.fragment_shader_barycentric {
            features2 = features2.push_next(&mut ebary);
        }
        if caps.fragment_shader_pixel_interlock {
            features2 = features2.push_next(&mut einterlock);
        }
        if caps.image_view_min_lod {
            features2 = features2.push_next(&mut eminlod);
        }
        if caps.mesh_shader {
            features2 = features2.push_next(&mut emesh);
        }
        if caps.device_fault {
            features2 = features2.push_next(&mut efault);
        }
        if caps.conditional_rendering {
            features2 = features2.push_next(&mut econd);
        }
        if caps.shader_image_int64_atomics {
            features2 = features2.push_next(&mut eimg64);
        }
        if caps.workgroup_memory_explicit_layout {
            features2 = features2.push_next(&mut ewgl);
        }
        if caps.compute_derivative_group_quads || caps.compute_derivative_group_linear {
            features2 = features2.push_next(&mut ederivs);
        }
        if caps.shader_device_clock || caps.shader_subgroup_clock {
            features2 = features2.push_next(&mut eclock);
        }
        if caps.provoking_vertex_last {
            features2 = features2.push_next(&mut eprovoking);
        }
        if caps.pipeline_robustness {
            features2 = features2.push_next(&mut erobust);
        }
        if caps.pipeline_executable_info {
            features2 = features2.push_next(&mut eexec);
        }
        if caps.graphics_pipeline_library {
            features2 = features2.push_next(&mut egpl);
        }
        if caps.maintenance8 {
            em8.p_next = features2.p_next;
            features2.p_next = &mut em8 as *mut _ as *mut c_void;
        }
        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family).queue_priorities(&prio)];
        let dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .enabled_extension_names(&dev_exts)
            .push_next(&mut features2);
        // SAFETY: every structure referenced by the chain lives until the call returns.
        let device = unsafe { instance.create_device(phys, &dci, None) }.map_err(|e| format!("device: {e:?}"))?;
        // SAFETY: the family/index pair was requested at creation.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let push_descriptor = ash::khr::push_descriptor::Device::new(&instance, &device);
        let mesh = caps.mesh_shader.then(|| ash::ext::mesh_shader::Device::new(&instance, &device));
        let fault = caps.device_fault.then(|| ash::ext::device_fault::Device::new(&instance, &device));
        let checkpoints = caps.diagnostic_checkpoints.then(|| ash::nv::device_diagnostic_checkpoints::Device::new(&instance, &device));
        let conditional = caps.conditional_rendering.then(|| ash::ext::conditional_rendering::Device::new(&instance, &device));
        Ok(GpuDevice {
            entry,
            instance,
            phys,
            device,
            queue_family,
            queue: Mutex::new(queue),
            props,
            mem_props,
            caps,
            name,
            surface_capable,
            push_descriptor,
            mesh,
            fault,
            checkpoints,
            raw_queue: queue,
            conditional,
            fault_reported: std::sync::atomic::AtomicBool::new(false),
            arena: crate::suballoc::Arena::default(),
            image_arena: crate::suballoc::Arena::default(),
            clock: OnceLock::new(),
        })
    }

    /// Exclusive access to the shared queue (submits and presents).
    pub fn queue(&self) -> MutexGuard<'_, vk::Queue> {
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// First memory type allowed by `bits` with all of `flags`.
    pub fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem_props.memory_type_count)
            .find(|&i| bits & (1 << i) != 0 && self.mem_props.memory_types[i as usize].property_flags.contains(flags))
    }

    /// Allocate memory for `req`, preferring `preferred` flags and falling
    /// back to `required`. When a memory type is out of memory, the next
    /// allowed one is tried (device-local first, then host memory): a full
    /// device heap (other processes share the GPU) costs bandwidth instead
    /// of failing the resource. A failed image creation binds a dummy and
    /// drops the GPU work that writes it, whose results later passes
    /// cannot recover.
    pub fn allocate(
        &self,
        req: vk::MemoryRequirements,
        preferred: vk::MemoryPropertyFlags,
        required: vk::MemoryPropertyFlags,
        device_address: bool,
    ) -> Result<(vk::DeviceMemory, vk::MemoryPropertyFlags), vk::Result> {
        let types = &self.mem_props.memory_types[..self.mem_props.memory_type_count as usize];
        let order = memory_type_order(types, req.memory_type_bits, preferred, required);
        let ty = allocate_in_order(&order, |ty| self.allocate_type(req.size, ty, device_address))?;
        static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if Some(&ty.0) != order.first() && LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
            log::warn!("ps5gpu: memory type {} out of memory, {} bytes placed in type {} ({:?})", order[0], req.size, ty.0, types[ty.0 as usize].property_flags);
        }
        Ok(ty.1)
    }

    /// Allocate `size` bytes of memory type `ty`.
    pub fn allocate_type(&self, size: u64, ty: u32, device_address: bool) -> Result<(vk::DeviceMemory, vk::MemoryPropertyFlags), vk::Result> {
        let req = vk::MemoryRequirements { size, alignment: 1, memory_type_bits: 1 << ty };
        let mut flags_info = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut info = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty);
        if device_address {
            info = info.push_next(&mut flags_info);
        }
        // SAFETY: valid allocate info.
        let mem = unsafe { self.device.allocate_memory(&info, None) }?;
        Ok((mem, self.mem_props.memory_types[ty as usize].property_flags))
    }

    /// Wait for the whole device to go idle (teardown, tests).
    pub fn wait_idle(&self) {
        let _q = self.queue();
        // SAFETY: queue access is serialised by the guard above.
        unsafe {
            let _ = self.device.device_wait_idle();
        }
    }
}

impl Drop for GpuDevice {
    fn drop(&mut self) {
        // The users (compile pool, cache writer) were joined by their
        // owners first; idle the device under the queue lock — a detached
        // straggler may still be submitting — and only then destroy it.
        // SAFETY: last reference; children were destroyed by their owners.
        let _q = self.queue();
        unsafe {
            let _ = self.device.device_wait_idle();
            for memory in self.arena.drain().into_iter().chain(self.image_arena.drain()) {
                self.device.free_memory(memory, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// A device buffer with its memory (optionally host-mapped and with a
/// device address). Destroyed on drop: retire it through the scheduler
/// while the GPU may still use it.
/// The memory types allowed by `bits` to allocate from, in order: those
/// with all of `preferred` flags, then those with all of `required` flags.
fn memory_type_order(types: &[vk::MemoryType], bits: u32, preferred: vk::MemoryPropertyFlags, required: vk::MemoryPropertyFlags) -> Vec<u32> {
    let allowed = |flags: vk::MemoryPropertyFlags| (0..types.len() as u32).filter(move |&i| bits & (1 << i) != 0 && types[i as usize].property_flags.contains(flags));
    let mut order: Vec<u32> = allowed(preferred).collect();
    for i in allowed(required) {
        if !order.contains(&i) {
            order.push(i);
        }
    }
    order
}

/// Allocate with `alloc` from the first type of `order` that is not out of
/// memory; other errors end the search. Returns the type used.
fn allocate_in_order<T>(order: &[u32], mut alloc: impl FnMut(u32) -> Result<T, vk::Result>) -> Result<(u32, T), vk::Result> {
    let mut last = vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;
    for &ty in order {
        match alloc(ty) {
            Ok(m) => return Ok((ty, m)),
            Err(e @ (vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY)) => last = e,
            Err(e) => return Err(e),
        }
    }
    Err(last)
}

/// Device-local bytes held by live [`Buffer`]s and [`Image`]s (diagnostics).
pub static DEVICE_LOCAL_BYTES: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

fn account(kind: usize, bytes: u64, add: bool) {
    use std::sync::atomic::Ordering;
    if add {
        DEVICE_LOCAL_BYTES[kind].fetch_add(bytes, Ordering::Relaxed);
    } else {
        DEVICE_LOCAL_BYTES[kind].fetch_sub(bytes, Ordering::Relaxed);
    }
}

pub struct Buffer {
    pub dev: Arc<GpuDevice>,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    /// Host pointer for host-visible buffers, else null.
    pub mapped: *mut u8,
    pub coherent: bool,
    /// Buffer device address (0 without SHADER_DEVICE_ADDRESS usage).
    pub address: u64,
    /// Device-local bytes of the allocation (0: host memory).
    device_local: u64,
    /// The place in a shared block (None: `memory` is the buffer's own).
    sub: Option<crate::suballoc::Suballocation>,
}

// SAFETY: the mapped pointer refers to device memory owned by this object.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

/// Memory placement of a [`Buffer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryUse {
    /// Device-local (not mapped).
    Device,
    /// Host-visible, write-combined upload memory.
    Upload,
    /// Host-visible, cached readback memory.
    Download,
}

impl Buffer {
    pub fn new(dev: &Arc<GpuDevice>, size: u64, usage: vk::BufferUsageFlags, mem: MemoryUse) -> Result<Buffer, vk::Result> {
        let size = size.max(16);
        let device_address = usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        let d = &dev.device;
        // SAFETY: plain create/allocate/bind sequence; cleaned up on error.
        unsafe {
            let buffer = d.create_buffer(
                &vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;
            let req = d.get_buffer_memory_requirements(buffer);
            use vk::MemoryPropertyFlags as M;
            let (preferred, required) = match mem {
                MemoryUse::Device => (M::DEVICE_LOCAL, M::empty()),
                MemoryUse::Upload => (M::HOST_VISIBLE | M::HOST_COHERENT, M::HOST_VISIBLE),
                MemoryUse::Download => (M::HOST_VISIBLE | M::HOST_CACHED | M::HOST_COHERENT, M::HOST_VISIBLE),
            };
            // Device-local buffers of moderate size share blocks.
            let shared = (mem == MemoryUse::Device && req.size <= crate::suballoc::MAX_SUBALLOCATION && crate::suballoc::enabled())
                .then(|| dev.memory_type(req.memory_type_bits, preferred).or_else(|| dev.memory_type(req.memory_type_bits, required)))
                .flatten();
            let mut sub = None;
            // A full shared block's memory type: a dedicated allocation
            // from the next memory type (see `GpuDevice::allocate`).
            let placed = shared.map(|ty| (ty, dev.arena.place(ty, req.size, req.alignment, |bytes| dev.allocate_type(bytes, ty, true).map(|m| m.0))));
            let (memory, flags) = if let Some((ty, Ok(s))) = placed {
                sub = Some(s);
                (s.memory, dev.mem_props.memory_types[ty as usize].property_flags)
            } else if let Some((_, Err(e))) = placed.filter(|p| !matches!(p.1, Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY))) {
                d.destroy_buffer(buffer, None);
                return Err(e);
            } else {
                match dev.allocate(req, preferred, required, device_address) {
                    Ok(m) => m,
                    Err(e) => {
                        d.destroy_buffer(buffer, None);
                        return Err(e);
                    }
                }
            };
            if let Err(e) = d.bind_buffer_memory(buffer, memory, sub.map_or(0, |s| s.offset)) {
                d.destroy_buffer(buffer, None);
                match sub {
                    Some(s) => {
                        if let Some(m) = dev.arena.release(s) {
                            d.free_memory(m, None);
                        }
                    }
                    None => d.free_memory(memory, None),
                }
                return Err(e);
            }
            let mut mapped = std::ptr::null_mut();
            if mem != MemoryUse::Device {
                match d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) {
                    Ok(p) => mapped = p as *mut u8,
                    Err(e) => {
                        d.destroy_buffer(buffer, None);
                        d.free_memory(memory, None);
                        return Err(e);
                    }
                }
            }
            let address = if device_address {
                d.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
            } else {
                0
            };
            let device_local = if flags.contains(M::DEVICE_LOCAL) { req.size } else { 0 };
            account(0, device_local, true);
            Ok(Buffer {
                dev: dev.clone(),
                buffer,
                memory,
                size,
                mapped,
                coherent: flags.contains(M::HOST_COHERENT),
                address,
                device_local,
                sub,
            })
        }
    }

    /// Host view of a mapped buffer.
    pub fn bytes_mut(&self) -> &mut [u8] {
        assert!(!self.mapped.is_null(), "buffer is not host-mapped");
        // SAFETY: mapped range of `size` bytes owned by this buffer; callers
        // only touch ranges the GPU is not using.
        unsafe { std::slice::from_raw_parts_mut(self.mapped, self.size as usize) }
    }

    /// Make host writes visible (non-coherent memory).
    pub fn flush(&self, offset: u64, size: u64) {
        if self.coherent || self.mapped.is_null() {
            return;
        }
        let atom = self.dev.caps.non_coherent_atom_size;
        let start = offset / atom * atom;
        let end = (offset + size).div_ceil(atom) * atom;
        let range = vk::MappedMemoryRange::default().memory(self.memory).offset(start).size((end - start).min(self.size - start));
        // SAFETY: range lies within the mapping.
        unsafe {
            let _ = self.dev.device.flush_mapped_memory_ranges(&[range]);
        }
    }

    /// Make device writes visible to the host (non-coherent memory).
    pub fn invalidate(&self, offset: u64, size: u64) {
        if self.coherent || self.mapped.is_null() {
            return;
        }
        let atom = self.dev.caps.non_coherent_atom_size;
        let start = offset / atom * atom;
        let end = (offset + size).div_ceil(atom) * atom;
        let range = vk::MappedMemoryRange::default().memory(self.memory).offset(start).size((end - start).min(self.size - start));
        // SAFETY: range lies within the mapping.
        unsafe {
            let _ = self.dev.device.invalidate_mapped_memory_ranges(&[range]);
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: owners retire buffers only after the GPU finished with them.
        unsafe {
            if !self.mapped.is_null() {
                self.dev.device.unmap_memory(self.memory);
            }
            self.dev.device.destroy_buffer(self.buffer, None);
            match self.sub {
                Some(s) => {
                    if let Some(m) = self.dev.arena.release(s) {
                        self.dev.device.free_memory(m, None);
                    }
                }
                None => self.dev.device.free_memory(self.memory, None),
            }
        }
        account(0, self.device_local, false);
    }
}

/// A device image with dedicated memory.
pub struct Image {
    pub dev: Arc<GpuDevice>,
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
    pub layers: u32,
    pub samples: vk::SampleCountFlags,
    pub usage: vk::ImageUsageFlags,
    pub aspect: vk::ImageAspectFlags,
    /// Bytes of the allocation.
    pub bytes: u64,
    /// Placed in a shared block of the image arena (`memory` is the block's).
    sub: Option<crate::suballoc::Suballocation>,
}

// SAFETY: plain handles.
unsafe impl Send for Image {}
unsafe impl Sync for Image {}

impl Image {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dev: &Arc<GpuDevice>,
        format: vk::Format,
        extent: vk::Extent3D,
        mip_levels: u32,
        layers: u32,
        samples: vk::SampleCountFlags,
        usage: vk::ImageUsageFlags,
        flags: vk::ImageCreateFlags,
    ) -> Result<Image, vk::Result> {
        let ty = if extent.depth > 1 { vk::ImageType::TYPE_3D } else { vk::ImageType::TYPE_2D };
        Self::new_typed(dev, ty, format, extent, mip_levels, layers, samples, usage, flags)
    }

    /// An image of an explicit type (1D / 2D / 3D).
    #[allow(clippy::too_many_arguments)]
    pub fn new_typed(
        dev: &Arc<GpuDevice>,
        ty: vk::ImageType,
        format: vk::Format,
        extent: vk::Extent3D,
        mip_levels: u32,
        layers: u32,
        samples: vk::SampleCountFlags,
        usage: vk::ImageUsageFlags,
        flags: vk::ImageCreateFlags,
    ) -> Result<Image, vk::Result> {
        let d = &dev.device;
        // SAFETY: plain create/allocate/bind sequence; cleaned up on error.
        unsafe {
            let image = d.create_image(
                &vk::ImageCreateInfo::default()
                    .flags(flags)
                    .image_type(ty)
                    .format(format)
                    .extent(extent)
                    .mip_levels(mip_levels.max(1))
                    .array_layers(layers.max(1))
                    .samples(samples)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?;
            let mut dedicated = vk::MemoryDedicatedRequirements::default();
            let mut req2 = vk::MemoryRequirements2::default().push_next(&mut dedicated);
            d.get_image_memory_requirements2(&vk::ImageMemoryRequirementsInfo2::default().image(image), &mut req2);
            let req = req2.memory_requirements;
            let wants_own = dedicated.prefers_dedicated_allocation != 0 || dedicated.requires_dedicated_allocation != 0;
            // Sizes rounded to the alignment: equal alignments leave no
            // padding runs between images (first fit stays short).
            // Small images share blocks (when on); a full device-local
            // type falls back to a dedicated allocation of the next type.
            let placed = (crate::suballoc::images_enabled() && !wants_own && req.size <= crate::suballoc::MAX_IMAGE_SUBALLOCATION)
                .then(|| dev.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL))
                .flatten()
                .and_then(|ty| dev.image_arena.place_in(ty, req.size.next_multiple_of(req.alignment.max(1)), req.alignment, crate::suballoc::IMAGE_BLOCK_BYTES, |bytes| dev.allocate_type(bytes, ty, false).map(|m| m.0)).ok());
            let memory = match placed {
                Some(s) => s.memory,
                None => match dev.allocate(req, vk::MemoryPropertyFlags::DEVICE_LOCAL, vk::MemoryPropertyFlags::empty(), false) {
                    Ok(m) => m.0,
                    Err(e) => {
                        d.destroy_image(image, None);
                        return Err(e);
                    }
                },
            };
            if let Err(e) = d.bind_image_memory(image, memory, placed.map_or(0, |s| s.offset)) {
                d.destroy_image(image, None);
                match placed {
                    Some(s) => {
                        if let Some(m) = dev.image_arena.release(s) {
                            d.free_memory(m, None);
                        }
                    }
                    None => d.free_memory(memory, None),
                }
                return Err(e);
            }
            Ok(Image {
                dev: dev.clone(),
                image,
                memory,
                format,
                extent,
                mip_levels: mip_levels.max(1),
                layers: layers.max(1),
                samples,
                usage,
                aspect: aspect_of(format),
                bytes: {
                    account(1, req.size, true);
                    req.size
                },
                sub: placed,
            })
        }
    }

    pub fn full_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.aspect,
            base_mip_level: 0,
            level_count: self.mip_levels,
            base_array_layer: 0,
            layer_count: self.layers,
        }
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        // SAFETY: owners retire images only after the GPU finished with them.
        unsafe {
            self.dev.device.destroy_image(self.image, None);
            match self.sub {
                Some(s) => {
                    if let Some(m) = self.dev.image_arena.release(s) {
                        self.dev.device.free_memory(m, None);
                    }
                }
                None => self.dev.device.free_memory(self.memory, None),
            }
        }
        account(1, self.bytes, false);
    }
}

impl Buffer {
    /// Import `size` bytes of host memory at `ptr` (aligned to
    /// `min_imported_host_pointer_alignment`) as a buffer without copying
    /// (VK_EXT_external_memory_host). The memory must outlive the buffer.
    ///
    /// # Safety
    /// `ptr..ptr+size` must stay mapped read/write while the buffer exists.
    pub unsafe fn import_host(dev: &Arc<GpuDevice>, ptr: *mut u8, size: u64, usage: vk::BufferUsageFlags) -> Result<Buffer, vk::Result> {
        if !dev.caps.external_memory_host {
            return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT);
        }
        let align = dev.caps.min_imported_host_pointer_alignment.max(1);
        if (ptr as u64) % align != 0 || size % align != 0 {
            return Err(vk::Result::ERROR_INVALID_EXTERNAL_HANDLE);
        }
        let d = &dev.device;
        let ext = ash::ext::external_memory_host::Device::new(&dev.instance, d);
        let ty = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        let mut props = vk::MemoryHostPointerPropertiesEXT::default();
        (ext.fp().get_memory_host_pointer_properties_ext)(d.handle(), ty, ptr as *const std::ffi::c_void, &mut props).result()?;
        let mut ext_info = vk::ExternalMemoryBufferCreateInfo::default().handle_types(ty);
        let buffer = d.create_buffer(
            &vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE).push_next(&mut ext_info),
            None,
        )?;
        let req = d.get_buffer_memory_requirements(buffer);
        let bits = req.memory_type_bits & props.memory_type_bits;
        let Some(type_index) = dev.memory_type(bits, vk::MemoryPropertyFlags::HOST_VISIBLE) else {
            d.destroy_buffer(buffer, None);
            return Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        };
        let mut import = vk::ImportMemoryHostPointerInfoEXT::default().handle_type(ty).host_pointer(ptr as *mut std::ffi::c_void);
        let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut info = vk::MemoryAllocateInfo::default().allocation_size(size).memory_type_index(type_index).push_next(&mut import);
        let bda = usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        if bda {
            info = info.push_next(&mut flags);
        }
        let memory = match d.allocate_memory(&info, None) {
            Ok(m) => m,
            Err(e) => {
                d.destroy_buffer(buffer, None);
                return Err(e);
            }
        };
        if let Err(e) = d.bind_buffer_memory(buffer, memory, 0) {
            d.destroy_buffer(buffer, None);
            d.free_memory(memory, None);
            return Err(e);
        }
        let address = if bda { d.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer)) } else { 0 };
        let coherent = dev.mem_props.memory_types[type_index as usize].property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
        // `mapped` stays null: the memory is not ours to unmap.
        Ok(Buffer { dev: dev.clone(), buffer, memory, size, mapped: std::ptr::null_mut(), coherent, address, device_local: 0, sub: None })
    }
}

/// Aspect mask of a format.
pub fn aspect_of(format: vk::Format) -> vk::ImageAspectFlags {
    use vk::Format as F;
    match format {
        F::D16_UNORM | F::D32_SFLOAT | F::X8_D24_UNORM_PACK32 => vk::ImageAspectFlags::DEPTH,
        F::S8_UINT => vk::ImageAspectFlags::STENCIL,
        F::D16_UNORM_S8_UINT | F::D24_UNORM_S8_UINT | F::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        _ => vk::ImageAspectFlags::COLOR,
    }
}

/// An image view; destroyed on drop.
pub struct ImageView {
    pub dev: Arc<GpuDevice>,
    pub view: vk::ImageView,
}

// SAFETY: plain handle.
unsafe impl Send for ImageView {}
unsafe impl Sync for ImageView {}

impl ImageView {
    pub fn new(dev: &Arc<GpuDevice>, info: &vk::ImageViewCreateInfo) -> Result<ImageView, vk::Result> {
        // SAFETY: valid create info.
        let view = unsafe { dev.device.create_image_view(info, None) }?;
        Ok(ImageView { dev: dev.clone(), view })
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        // SAFETY: retired after GPU use.
        unsafe { self.dev.device.destroy_image_view(self.view, None) };
    }
}

/// The widths (16, 32, 64 bits) whose denormals shaders may declare
/// preserved. Without independence one mode serves every width, so only a
/// device that preserves all three may; 64-bit preservation needs full
/// independence (with 32-bit-only independence 16 and 64 bits share a
/// mode, and a module that sets only one of them left 64-bit subnormal FMA
/// results flushed on one driver).
pub fn denorm_preserve_caps(independence: vk::ShaderFloatControlsIndependence, preserve: [bool; 3]) -> [bool; 3] {
    use vk::ShaderFloatControlsIndependence as I;
    if independence == I::NONE {
        return if preserve.iter().all(|&p| p) { preserve } else { [false; 3] };
    }
    [preserve[0], preserve[1], preserve[2] && independence == I::ALL]
}

/// The sample counts whose locations VK_EXT_sample_locations can program
/// in the guest's resolution: at least 4 sub-pixel bits over a coordinate
/// range covering [0, 15/16], and a sample location grid of any size for
/// that count.
fn sample_location_counts(entry: &ash::Entry, instance: &ash::Instance, phys: vk::PhysicalDevice) -> vk::SampleCountFlags {
    let mut lp = vk::PhysicalDeviceSampleLocationsPropertiesEXT::default();
    let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut lp);
    // SAFETY: valid chain on the stack.
    unsafe { instance.get_physical_device_properties2(phys, &mut p2) };
    if lp.sample_location_sub_pixel_bits < 4 || lp.sample_location_coordinate_range[0] > 0.0 || lp.sample_location_coordinate_range[1] < 15.0 / 16.0 {
        return vk::SampleCountFlags::empty();
    }
    let ext = ash::ext::sample_locations::Instance::new(entry, instance);
    let mut counts = vk::SampleCountFlags::empty();
    for samples in [vk::SampleCountFlags::TYPE_2, vk::SampleCountFlags::TYPE_4, vk::SampleCountFlags::TYPE_8] {
        if !lp.sample_location_sample_counts.contains(samples) {
            continue;
        }
        let mut mp = vk::MultisamplePropertiesEXT::default();
        // SAFETY: valid physical device; the extension is present.
        unsafe { ext.get_physical_device_multisample_properties(phys, samples, &mut mp) };
        if mp.max_sample_location_grid_size.width != 0 && mp.max_sample_location_grid_size.height != 0 {
            counts |= samples;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_fault_report_lines() {
        let lines = fault_lines(
            vk::Result::SUCCESS,
            "page fault",
            &[(vk::DeviceFaultAddressTypeEXT::READ_INVALID, 0x1234000, 0xfff), (vk::DeviceFaultAddressTypeEXT::INSTRUCTION_POINTER_FAULT, 0xabcd, 0x3)],
            &[("mmu".to_string(), 0x77, 0x88)],
        );
        assert_eq!(lines[0], "device fault (SUCCESS): \"page fault\", 2 address record(s), 1 vendor record(s)");
        assert_eq!(lines[1], "device fault: read invalid at 0x1234000 (precision 0xfff)");
        assert_eq!(lines[2], "device fault: instruction pointer fault at 0xabcd (precision 0x3)");
        assert_eq!(lines[3], "device fault: vendor \"mmu\" (code 0x77, data 0x88)");
        assert_eq!(fault_address_kind(vk::DeviceFaultAddressTypeEXT::from_raw(99)), "unknown");
    }

    #[test]
    fn checkpoint_report_lines() {
        let lines = checkpoint_lines(&[(0x10000, 0x1), (0x10100, 0x2000)]);
        assert_eq!(lines, ["device fault: checkpoint: program 0x10000 at stage 0x1", "device fault: checkpoint: program 0x10100 at stage 0x2000"]);
        assert_eq!(checkpoint_lines(&[]), ["device fault: no checkpoint reached"]);
    }

    #[test]
    fn requirements_and_limitations() {
        let full = DeviceCaps {
            fragment_shader_barycentric: true,
            shader_float64: true,
            mesh_shader: true,
            shader_demote_to_helper_invocation: true,
            shader_image_int64_atomics: true,
            shader_shared_int64_atomics: true,
            workgroup_memory_explicit_layout: true,
            compute_derivative_group_quads: true,
            bindless_slots: 1024,
            subgroup_size: 32,
            multi_viewport: true,
            texture_compression_bc: true,
            ..DeviceCaps::default()
        };
        assert!(missing_required(&full, true).is_empty());
        assert!(limitations(&full).is_empty());
        // An Arc-like device: no f64, everything else there.
        let arc = DeviceCaps { shader_float64: false, ..full.clone() };
        assert!(missing_required(&arc, true).is_empty());
        assert_eq!(limitations(&arc).len(), 1);
        assert!(limitations(&arc)[0].contains("shaderFloat64"));
        let old = DeviceCaps { fragment_shader_barycentric: false, ..full };
        assert_eq!(missing_required(&old, false).len(), 2);
    }

    #[test]
    fn denorm_preserve_follows_float_control_independence() {
        use vk::ShaderFloatControlsIndependence as I;
        assert_eq!(denorm_preserve_caps(I::ALL, [true; 3]), [true; 3]);
        assert_eq!(denorm_preserve_caps(I::TYPE_32_ONLY, [true; 3]), [true, true, false], "64-bit preserve needs full independence");
        assert_eq!(denorm_preserve_caps(I::NONE, [true; 3]), [true; 3]);
        assert_eq!(denorm_preserve_caps(I::NONE, [false, true, true]), [false; 3]);
        assert_eq!(denorm_preserve_caps(I::ALL, [false, true, true]), [false, true, true]);
    }

    /// A full device heap: an image allocation falls back to host memory
    /// instead of failing (a failed creation dropped the GPU copy that grows
    /// SILENT HILL's reflection capture array, and every reflection sampled
    /// stale float rows as halves: purple and blue backgrounds).
    #[test]
    fn allocation_falls_back_to_the_next_memory_type_when_out_of_memory() {
        use vk::MemoryPropertyFlags as M;
        let ty = |f: M, heap: u32| vk::MemoryType { property_flags: f, heap_index: heap };
        // Types like a discrete GPU's: system memory, VRAM, BAR, host cached.
        let types = [ty(M::empty(), 1), ty(M::DEVICE_LOCAL, 0), ty(M::DEVICE_LOCAL | M::HOST_VISIBLE | M::HOST_COHERENT, 0), ty(M::HOST_VISIBLE | M::HOST_COHERENT | M::HOST_CACHED, 1)];
        let image_bits = 0b0011;
        assert_eq!(memory_type_order(&types, image_bits, M::DEVICE_LOCAL, M::empty()), vec![1, 0]);
        assert_eq!(memory_type_order(&types, 0b1111, M::HOST_VISIBLE | M::HOST_COHERENT, M::HOST_VISIBLE), vec![2, 3]);
        let order = memory_type_order(&types, image_bits, M::DEVICE_LOCAL, M::empty());
        let vram_full = |t: u32| if types[t as usize].heap_index == 0 { Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) } else { Ok(t * 10) };
        assert_eq!(allocate_in_order(&order, vram_full), Ok((0, 0)));
        assert_eq!(allocate_in_order(&order, |t| Ok::<_, vk::Result>(t)), Ok((1, 1)));
        assert_eq!(allocate_in_order(&order, |_| Err::<u32, _>(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)), Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
        assert_eq!(allocate_in_order(&order, |_| Err::<u32, _>(vk::Result::ERROR_INITIALIZATION_FAILED)), Err(vk::Result::ERROR_INITIALIZATION_FAILED));
    }

    /// The host fallback exists for render-target / storage images on this
    /// GPU: their memory type bits allow a type outside device-local memory.
    #[test]
    #[ignore]
    fn optimal_images_may_live_in_host_memory() {
        use vk::MemoryPropertyFlags as M;
        let d = GpuDevice::create(false).expect("device");
        let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        // SAFETY: plain create/query/destroy on a live device.
        let req = unsafe {
            let image = d
                .device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE | vk::ImageCreateFlags::MUTABLE_FORMAT)
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R16G16B16A16_SFLOAT)
                        .extent(vk::Extent3D { width: 128, height: 128, depth: 1 })
                        .mip_levels(8)
                        .array_layers(516)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .usage(usage),
                    None,
                )
                .expect("image");
            let req = d.device.get_image_memory_requirements(image);
            d.device.destroy_image(image, None);
            req
        };
        let types = &d.mem_props.memory_types[..d.mem_props.memory_type_count as usize];
        let order = memory_type_order(types, req.memory_type_bits, M::DEVICE_LOCAL, M::empty());
        eprintln!("'{}': image memory types in order {order:?}: {:?}", d.name, order.iter().map(|&t| types[t as usize].property_flags).collect::<Vec<_>>());
        assert!(order.iter().any(|&t| !types[t as usize].property_flags.contains(M::DEVICE_LOCAL)), "no host memory type for images");
    }

    /// Needs a Vulkan 1.3 GPU: `cargo test -p ps5gpu --release -- --ignored`.
    #[test]
    #[ignore]
    fn creates_headless_device_with_required_features() {
        let d = GpuDevice::create(false).expect("device");
        assert!(d.caps.max_push_descriptors >= 32);
        let b = Buffer::new(&d, 4096, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS, MemoryUse::Device)
            .unwrap();
        assert_ne!(b.address, 0);
        let u = Buffer::new(&d, 4096, vk::BufferUsageFlags::TRANSFER_SRC, MemoryUse::Upload).unwrap();
        u.bytes_mut()[0] = 7;
        eprintln!("device '{}' caps {:?}", d.name, d.caps);
    }

    /// Zero-copy import of a memfd mapping (how guest RAM is backed) with
    /// VK_EXT_external_memory_host: a GPU fill lands in the guest view and a
    /// CPU write is seen by a GPU copy.
    #[test]
    #[ignore]
    fn imports_memfd_guest_memory_zero_copy() {
        let d = GpuDevice::create(false).expect("device");
        if !d.caps.external_memory_host {
            eprintln!("VK_EXT_external_memory_host not supported; skipped");
            return;
        }
        let len = 4usize << 20;
        // SAFETY: plain memfd + two shared mappings owned by the test.
        let (view, alias) = unsafe {
            let fd = libc::memfd_create(c"import-test".as_ptr(), 0);
            assert_eq!(libc::ftruncate(fd, len as libc::off_t), 0);
            let m = |_| libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0) as *mut u8;
            (m(0), m(1))
        };
        let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        // SAFETY: the alias mapping lives for the test.
        let imported = unsafe { Buffer::import_host(&d, alias, len as u64, usage) };
        let imported = match imported {
            Ok(b) => b,
            Err(e) => panic!("import of memfd memory failed: {e:?}"),
        };
        assert_ne!(imported.address, 0);
        let mut s = crate::scheduler::Scheduler::new(&d).unwrap();
        let cb = s.cmd();
        // SAFETY: recording buffer.
        unsafe { d.device.cmd_fill_buffer(cb, imported.buffer, 4096, 4096, 0xCAFE_F00D) };
        s.flush_and_wait();
        // SAFETY: mapped guest view.
        assert_eq!(unsafe { std::ptr::read_volatile(view.add(4096) as *const u32) }, 0xCAFE_F00D);
        // SAFETY: as above.
        unsafe { std::ptr::write_volatile(view.add(8192) as *mut u32, 0x1234_5678) };
        let out = Buffer::new(&d, 4, vk::BufferUsageFlags::TRANSFER_DST, MemoryUse::Download).unwrap();
        let cb = s.cmd();
        // SAFETY: recording buffer.
        unsafe { d.device.cmd_copy_buffer(cb, imported.buffer, out.buffer, &[vk::BufferCopy { src_offset: 8192, dst_offset: 0, size: 4 }]) };
        s.flush_and_wait();
        out.invalidate(0, 4);
        assert_eq!(u32::from_le_bytes(out.bytes_mut()[..4].try_into().unwrap()), 0x1234_5678);
        eprintln!("memfd import ok (alignment {:#x})", d.caps.min_imported_host_pointer_alignment);
    }
}
