// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Host GPU statistics for the overlay: NVIDIA NVML (loaded at run time,
//! no build dependency), amdgpu sysfs, and VK_EXT_memory_budget of our own
//! Vulkan device. Every source is optional; missing values stay `None`.

use super::proc;
#[cfg(unix)]
use std::ffi::{c_char, CStr};
use std::ffi::{c_int, c_uint, c_void};
use std::path::PathBuf;

/// One GPU sample.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuSample {
    /// Utilisation, percent.
    pub busy: Option<u32>,
    /// Device memory in use / total (whole system), bytes.
    pub vram_used: Option<u64>,
    pub vram_total: Option<u64>,
    /// Device-local memory this process uses (VK_EXT_memory_budget).
    pub vram_process: Option<u64>,
    /// Degrees Celsius.
    pub temp_c: Option<u32>,
    /// Graphics clock, MHz.
    pub clock_mhz: Option<u32>,
}

type NvmlDevice = *mut c_void;

#[repr(C)]
#[derive(Default)]
struct NvmlUtilization {
    gpu: c_uint,
    memory: c_uint,
}

#[repr(C)]
#[derive(Default)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}

/// The NVML entry points the overlay uses.
struct Nvml {
    device: NvmlDevice,
    utilization: unsafe extern "C" fn(NvmlDevice, *mut NvmlUtilization) -> c_int,
    memory: unsafe extern "C" fn(NvmlDevice, *mut NvmlMemory) -> c_int,
    temperature: unsafe extern "C" fn(NvmlDevice, c_int, *mut c_uint) -> c_int,
    clock: unsafe extern "C" fn(NvmlDevice, c_int, *mut c_uint) -> c_int,
}

// SAFETY: NVML is thread-safe; the handle is only used by the sampler.
unsafe impl Send for Nvml {}

const NVML_TEMPERATURE_GPU: c_int = 0;
const NVML_CLOCK_GRAPHICS: c_int = 0;

impl Nvml {
    /// Load `libnvidia-ml.so.1` and pick the device named `want` (our
    /// Vulkan device), else the first one.
    #[cfg(windows)]
    fn load(_want: &str) -> Option<Nvml> {
        None
    }

    /// Load `libnvidia-ml.so.1` and pick the device named `want` (our
    /// Vulkan device), else the first one.
    #[cfg(unix)]
    fn load(want: &str) -> Option<Nvml> {
        // SAFETY: dlopen/dlsym on a NUL-terminated name; every symbol is
        // cast to its documented NVML signature. The library is never
        // closed (the handle lives for the process).
        unsafe {
            let lib = libc::dlopen(c"libnvidia-ml.so.1".as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
            if lib.is_null() {
                return None;
            }
            let sym = |name: &CStr| {
                let p = libc::dlsym(lib, name.as_ptr());
                (!p.is_null()).then_some(p)
            };
            let init: unsafe extern "C" fn() -> c_int = std::mem::transmute(sym(c"nvmlInit_v2")?);
            let count: unsafe extern "C" fn(*mut c_uint) -> c_int = std::mem::transmute(sym(c"nvmlDeviceGetCount_v2")?);
            let by_index: unsafe extern "C" fn(c_uint, *mut NvmlDevice) -> c_int =
                std::mem::transmute(sym(c"nvmlDeviceGetHandleByIndex_v2")?);
            let name: unsafe extern "C" fn(NvmlDevice, *mut c_char, c_uint) -> c_int =
                std::mem::transmute(sym(c"nvmlDeviceGetName")?);
            if init() != 0 {
                return None;
            }
            let mut n: c_uint = 0;
            if count(&mut n) != 0 || n == 0 {
                return None;
            }
            let mut chosen = None;
            for i in 0..n {
                let mut dev: NvmlDevice = std::ptr::null_mut();
                if by_index(i, &mut dev) != 0 {
                    continue;
                }
                let mut buf = [0 as c_char; 96];
                let named = name(dev, buf.as_mut_ptr(), buf.len() as c_uint) == 0
                    && CStr::from_ptr(buf.as_ptr()).to_str().is_ok_and(|s| !want.is_empty() && (s == want || want.contains(s)));
                if chosen.is_none() || named {
                    chosen = Some(dev);
                }
                if named {
                    break;
                }
            }
            Some(Nvml {
                device: chosen?,
                utilization: std::mem::transmute(sym(c"nvmlDeviceGetUtilizationRates")?),
                memory: std::mem::transmute(sym(c"nvmlDeviceGetMemoryInfo")?),
                temperature: std::mem::transmute(sym(c"nvmlDeviceGetTemperature")?),
                clock: std::mem::transmute(sym(c"nvmlDeviceGetClockInfo")?),
            })
        }
    }

    fn sample(&self, out: &mut GpuSample) {
        // SAFETY: valid device handle and out pointers.
        unsafe {
            let mut u = NvmlUtilization::default();
            if (self.utilization)(self.device, &mut u) == 0 {
                out.busy = Some(u.gpu);
            }
            let mut m = NvmlMemory::default();
            if (self.memory)(self.device, &mut m) == 0 {
                out.vram_used = Some(m.used);
                out.vram_total = Some(m.total);
            }
            let mut t: c_uint = 0;
            if (self.temperature)(self.device, NVML_TEMPERATURE_GPU, &mut t) == 0 {
                out.temp_c = Some(t);
            }
            let mut c: c_uint = 0;
            if (self.clock)(self.device, NVML_CLOCK_GRAPHICS, &mut c) == 0 {
                out.clock_mhz = Some(c);
            }
        }
    }
}

/// An amdgpu device in sysfs (`/sys/class/drm/cardN/device`).
struct AmdSysfs {
    dir: PathBuf,
    hwmon: Option<PathBuf>,
}

impl AmdSysfs {
    /// The DRM card with `gpu_busy_percent`, preferring the one whose PCI
    /// ids match our Vulkan device.
    fn find(vendor: u32, device: u32) -> Option<AmdSysfs> {
        let mut found: Option<PathBuf> = None;
        for e in std::fs::read_dir("/sys/class/drm").ok()?.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with("card") || name.contains('-') {
                continue;
            }
            let dir = e.path().join("device");
            if !dir.join("gpu_busy_percent").exists() {
                continue;
            }
            let id = |f: &str| {
                std::fs::read_to_string(dir.join(f)).ok().and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            };
            let matches = id("vendor") == Some(vendor) && id("device") == Some(device);
            if matches || found.is_none() {
                found = Some(dir);
            }
            if matches {
                break;
            }
        }
        let dir = found?;
        let hwmon = std::fs::read_dir(dir.join("hwmon")).ok().and_then(|mut d| d.next()).and_then(|e| e.ok()).map(|e| e.path());
        Some(AmdSysfs { dir, hwmon })
    }

    fn sample(&self, out: &mut GpuSample) {
        let num = |p: PathBuf| std::fs::read_to_string(p).ok().and_then(|s| s.trim().parse::<u64>().ok());
        if let Some(v) = num(self.dir.join("gpu_busy_percent")) {
            out.busy = Some(v as u32);
        }
        if let Some(v) = num(self.dir.join("mem_info_vram_used")) {
            out.vram_used = Some(v);
        }
        if let Some(v) = num(self.dir.join("mem_info_vram_total")) {
            out.vram_total = Some(v);
        }
        if let Some(h) = &self.hwmon {
            if let Some(v) = num(h.join("temp1_input")) {
                out.temp_c = Some((v / 1000) as u32);
            }
            if let Some(v) = num(h.join("freq1_input")) {
                out.clock_mhz = Some((v / 1_000_000) as u32);
            }
        }
        if out.clock_mhz.is_none() {
            out.clock_mhz = std::fs::read_to_string(self.dir.join("pp_dpm_sclk")).ok().and_then(|s| proc::parse_dpm_clock(&s));
        }
    }
}

/// All GPU sources, probed once on the sampler thread.
pub(super) struct GpuSources {
    nvml: Option<Nvml>,
    amd: Option<AmdSysfs>,
    probed: bool,
}

impl GpuSources {
    pub(super) fn new() -> Self {
        Self { nvml: None, amd: None, probed: false }
    }

    pub(super) fn sample(&mut self) -> GpuSample {
        let dev = crate::device::GpuDevice::global_ready().then(crate::device::GpuDevice::global).flatten();
        if !self.probed && dev.is_some() {
            self.probed = true;
            let d = dev.unwrap();
            if std::env::var("RPS5E_PERF_NVML").map_or(true, |v| v != "0") {
                self.nvml = Nvml::load(&d.name);
            }
            if self.nvml.is_none() {
                self.amd = AmdSysfs::find(d.props.vendor_id, d.props.device_id);
            }
            log::info!(
                "perf overlay: GPU statistics from {}",
                if self.nvml.is_some() {
                    "NVML"
                } else if self.amd.is_some() {
                    "amdgpu sysfs"
                } else {
                    "Vulkan memory budget only"
                }
            );
        }
        let mut s = GpuSample::default();
        if let Some(n) = &self.nvml {
            n.sample(&mut s);
        } else if let Some(a) = &self.amd {
            a.sample(&mut s);
        }
        if let Some(d) = dev {
            if d.caps.memory_budget {
                let (used, total) = memory_budget(d);
                s.vram_process = Some(used);
                if s.vram_total.is_none() {
                    s.vram_total = Some(total);
                }
            }
        }
        s
    }
}

/// Device-local memory this process uses and the device-local heap size
/// (VK_EXT_memory_budget).
fn memory_budget(d: &crate::device::GpuDevice) -> (u64, u64) {
    let mut budget = ash::vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let props = {
        let mut p2 = ash::vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        // SAFETY: valid chain on the stack; physical device queries need
        // no external synchronisation.
        unsafe { d.instance.get_physical_device_memory_properties2(d.phys, &mut p2) };
        p2.memory_properties
    };
    let (mut used, mut total) = (0, 0);
    for i in 0..props.memory_heap_count as usize {
        if props.memory_heaps[i].flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL) {
            used += budget.heap_usage[i];
            total += props.memory_heaps[i].size;
        }
    }
    (used, total)
}
