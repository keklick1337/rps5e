// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The GPU reference clock (100 MHz).
//!
//! On the console the command processor's timestamps (end-of-pipe writes,
//! RELEASE_MEM, COPY_DATA from the clock) and a shader's `s_memrealtime`
//! read the same 100 MHz counter, and guest code mixes the two: a frame
//! pacing compute shader gets a deadline computed from a CP timestamp and
//! spins on `s_memrealtime` until the clock passes it (or a bound relative
//! to its own first read). Both sides therefore come from one clock that
//! advances in real time:
//!
//! - shaders read the device clock (`OpReadClockKHR`, device scope) and
//!   scale it to 100 MHz ticks with a fixed ratio ([`DeviceClock`]);
//! - the host side ([`reference_clock`]) is `CLOCK_MONOTONIC` in 10 ns
//!   ticks plus an offset measured once at device creation by a calibration
//!   dispatch that reads the device clock ([`calibrate`]).
//!
//! A shader and the host reading the clock at the same instant agree within
//! the calibration error (the shortest submit round trip, tens of
//! microseconds). Without a device clock (no `shaderDeviceClock`) or when
//! calibration fails, shaders fall back to a per-invocation counter and
//! the host clock has no offset.

use crate::device::{Buffer, GpuDevice, MemoryUse};
use crate::tiler::{Spv, R};
use ash::vk;
use rspirv::spirv;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Reference clock frequency.
pub const REFERENCE_HZ: u64 = 100_000_000;

/// Reference ticks = device clock ticks * `num` / `den`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceClock {
    pub num: u32,
    pub den: u32,
}

impl DeviceClock {
    /// The ratio for a device clock with `period_ns` nanoseconds per tick
    /// (one reference tick is 10 ns), as a reduced fraction with a
    /// denominator of at most 10^6.
    pub fn from_period(period_ns: f64) -> Option<DeviceClock> {
        if !period_ns.is_finite() || period_ns <= 0.0 {
            return None;
        }
        let den = 1_000_000u64;
        let num = (period_ns / 10.0 * den as f64).round() as u64;
        if num == 0 || num > u32::MAX as u64 {
            return None;
        }
        let g = gcd(num, den);
        Some(DeviceClock { num: (num / g) as u32, den: (den / g) as u32 })
    }

    /// Device clock ticks in reference ticks (no intermediate overflow:
    /// `ticks / den * num` is the result itself).
    pub fn to_reference(self, ticks: u64) -> u64 {
        let (num, den) = (self.num as u64, self.den as u64);
        (ticks / den).wrapping_mul(num).wrapping_add((ticks % den) * num / den)
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Reference ticks minus monotonic 10 ns ticks.
static OFFSET: AtomicI64 = AtomicI64::new(0);
/// Error bound (ns) of the calibration `OFFSET` came from. Every device of
/// the process reads the same physical clock, so a later calibration
/// (another device, measured under load) replaces the offset only when it
/// is at least as precise.
static OFFSET_ERROR_NS: std::sync::Mutex<u64> = std::sync::Mutex::new(u64::MAX);

#[cfg(windows)]
fn monotonic_ns() -> u64 {
    ps5vm::os::monotonic_ns()
}

#[cfg(unix)]
fn monotonic_ns() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: valid timespec out pointer.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

/// The 100 MHz reference clock as the GPU sees it (CP timestamps; equal to
/// what `s_memrealtime` reads in a shader at the same instant).
pub fn reference_clock() -> u64 {
    ((monotonic_ns() / 10) as i64).wrapping_add(OFFSET.load(Ordering::Relaxed)) as u64
}

/// How often the calibrated offset is re-measured (ns of CLOCK_MONOTONIC).
const RECAL_PERIOD_NS: u64 = 1_000_000_000;
/// When the next recalibration is due (monotonic ns; 0: at once).
static NEXT_RECAL: AtomicU64 = AtomicU64::new(0);
/// The calibration dispatch, built on the first due recalibration and
/// kept for the process.
static RECAL_RIG: Mutex<Option<Rig>> = Mutex::new(None);

/// `RPS5E_CLOCK_RECAL=0`: keep the offset measured at device creation
/// (it drifts against CLOCK_MONOTONIC by ~7 ppm ≈ 26 ms/hour, which the
/// games' frame limiter waits away).
fn recal_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_CLOCK_RECAL").is_ok_and(|v| v == "0"))
}

/// Decide a recalibrated offset from one sample: None when the sample is
/// unusable — a round trip longer than 100 µs (a busy GPU delays the
/// dispatch by far more than the offset sought) or an offset more than
/// 2 ms from the current one (drift is ~7 µs per period; a jump that
/// large is a glitch, not drift).
fn accept_recal(old_offset: i64, s: Sample, clock: DeviceClock) -> Option<i64> {
    const MAX_RTT_NS: u64 = 100_000;
    const MAX_JUMP_TICKS: i64 = 200_000; // 10 ns reference ticks = 2 ms
    if s.after - s.before > 2 * MAX_RTT_NS {
        return None;
    }
    let offset = (clock.to_reference(s.ticks) as i64).wrapping_sub((s.mid() / 10) as i64);
    (offset.wrapping_sub(old_offset).wrapping_abs() <= MAX_JUMP_TICKS).then_some(offset)
}

/// Re-measure the host offset when its period elapsed: the device-clock
/// ratio is fixed, so the offset calibrated at device creation drifts
/// against CLOCK_MONOTONIC (~7 ppm). One sample per period; a busy GPU
/// spoils it (see [`accept_recal`]) and the next period tries again.
/// The initial calibration's precision bookkeeping ([`OFFSET_ERROR_NS`])
/// is untouched: freshness beats precision here.
pub fn recalibrate_if_due(dev: &Arc<GpuDevice>) {
    if !recal_enabled() || !dev.caps.shader_device_clock {
        return;
    }
    let now = monotonic_ns();
    if now < NEXT_RECAL.load(Ordering::Relaxed) {
        return;
    }
    NEXT_RECAL.store(now.wrapping_add(RECAL_PERIOD_NS), Ordering::Relaxed);
    let Some(clock) = dev.clock.get() else { return };
    let mut rig = RECAL_RIG.lock().unwrap_or_else(|e| e.into_inner());
    if rig.is_none() {
        match Rig::new(dev) {
            Ok(r) => *rig = Some(r),
            Err(e) => {
                // Permanent (the device rejected the kernel): stop retrying.
                log::warn!("ps5gpu: clock recalibration rig unavailable ({e})");
                NEXT_RECAL.store(u64::MAX / 2, Ordering::Relaxed);
                return;
            }
        }
    }
    let Ok(s) = rig.as_ref().unwrap().sample_within(20_000_000) else { return };
    if let Some(offset) = accept_recal(OFFSET.load(Ordering::Relaxed), s, *clock) {
        OFFSET.store(offset, Ordering::Relaxed);
    }
}

/// One thread writes the device clock to output words 0 (low) and 1 (high).
fn kernel() -> R<Vec<u32>> {
    let mut s = Spv::new();
    s.b.capability(spirv::Capability::Int64);
    s.b.capability(spirv::Capability::ShaderClockKHR);
    s.b.extension("SPV_KHR_shader_clock");
    let (main, _, _, _) = s.begin_main()?;
    let u64t = s.b.type_int(64, 0);
    let scope = s.c(spirv::Scope::Device as u32);
    let t = s.b.read_clock_khr(u64t, None, scope)?;
    let lo = s.b.u_convert(s.uint, None, t)?;
    let shift = s.c(32);
    let high = s.b.shift_right_logical(u64t, None, t, shift)?;
    let hi = s.b.u_convert(s.uint, None, high)?;
    for (i, v) in [(0, lo), (1, hi)] {
        let idx = s.c(i);
        let p = s.word_ptr(s.dst, idx)?;
        s.b.store(p, v, None, [])?;
    }
    s.finish(main, 1, 1)
}

/// One calibration sample: monotonic ns before the submit, after the wait,
/// and the device clock the dispatch read.
#[derive(Clone, Copy, Debug)]
struct Sample {
    before: u64,
    after: u64,
    ticks: u64,
}

impl Sample {
    fn mid(&self) -> u64 {
        self.before + (self.after - self.before) / 2
    }
}

/// The kernel's pipeline and a command buffer that runs it once.
struct Rig {
    dev: Arc<GpuDevice>,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::CommandPool,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    out: Buffer,
}

impl Rig {
    fn new(dev: &Arc<GpuDevice>) -> Result<Rig, String> {
        let code = kernel().map_err(|e| format!("kernel: {e}"))?;
        let out = Buffer::new(dev, 16, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryUse::Download).map_err(|e| format!("buffer: {e:?}"))?;
        let d = &dev.device;
        let bindings = [0u32, 1].map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        // SAFETY: valid create infos; every object is destroyed in Drop (the
        // rig is built field by field, so a failure leaks only what the
        // device teardown reclaims).
        unsafe {
            let set_layout = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR).bindings(&bindings),
                    None,
                )
                .map_err(|e| format!("set layout: {e:?}"))?;
            let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: (crate::tiler::ARGUMENT_COUNT * 4) as u32 }];
            let layout = d
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set_layout)).push_constant_ranges(&ranges), None)
                .map_err(|e| format!("layout: {e:?}"))?;
            let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| format!("module: {e:?}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let r = d.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None);
            d.destroy_shader_module(module, None);
            let pipeline = r.map_err(|(_, e)| format!("pipeline: {e:?}"))?[0];
            let pool = d
                .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(dev.queue_family), None)
                .map_err(|e| format!("pool: {e:?}"))?;
            let cb = d
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))
                .map_err(|e| format!("command buffer: {e:?}"))?[0];
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("fence: {e:?}"))?;
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default()).map_err(|e| format!("begin: {e:?}"))?;
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
            let info = vk::DescriptorBufferInfo { buffer: out.buffer, offset: 0, range: vk::WHOLE_SIZE };
            let write = vk::WriteDescriptorSet::default().dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&info));
            dev.push_descriptor.cmd_push_descriptor_set(cb, vk::PipelineBindPoint::COMPUTE, layout, 0, std::slice::from_ref(&write));
            d.cmd_dispatch(cb, 1, 1, 1);
            let barrier = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[barrier], &[], &[]);
            d.end_command_buffer(cb).map_err(|e| format!("end: {e:?}"))?;
            Ok(Rig { dev: dev.clone(), set_layout, layout, pipeline, pool, cb, fence, out })
        }
    }

    fn sample(&self) -> Result<Sample, String> {
        self.sample_within(1_000_000_000)
    }

    /// One sample waiting at most `fence_ns` for the dispatch: Err when
    /// the device was still busy then (the dispatch is drained anyway —
    /// its command buffer must be idle before the next sample resets the
    /// fence — but its timestamps are unusable).
    fn sample_within(&self, fence_ns: u64) -> Result<Sample, String> {
        let d = &self.dev.device;
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.cb));
        // SAFETY: the command buffer is idle (each sample waits for it).
        unsafe {
            d.reset_fences(&[self.fence]).map_err(|e| format!("reset: {e:?}"))?;
            let queue = self.dev.queue();
            let before = monotonic_ns();
            d.queue_submit(*queue, &[submit], self.fence).map_err(|e| format!("submit: {e:?}"))?;
            drop(queue);
            if d.wait_for_fences(&[self.fence], true, fence_ns) == Err(vk::Result::TIMEOUT) {
                d.wait_for_fences(&[self.fence], true, u64::MAX).map_err(|e| format!("wait: {e:?}"))?;
                return Err("the device is busy".into());
            }
            let after = monotonic_ns();
            self.out.invalidate(0, 8);
            let b = self.out.bytes_mut();
            let ticks = u64::from_le_bytes(b[..8].try_into().unwrap());
            Ok(Sample { before, after, ticks })
        }
    }

    /// The sample with the shortest round trip of `n`.
    fn best(&self, n: usize) -> Result<Sample, String> {
        let mut best: Option<Sample> = None;
        for _ in 0..n {
            let s = self.sample()?;
            if best.is_none_or(|b| s.after - s.before < b.after - b.before) {
                best = Some(s);
            }
        }
        Ok(best.unwrap())
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        let d = &self.dev.device;
        // SAFETY: every sample waited for its submit; the objects are unused.
        unsafe {
            d.destroy_fence(self.fence, None);
            d.destroy_command_pool(self.pool, None);
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// Measure the device clock against `CLOCK_MONOTONIC`: its rate (checked
/// against the timestamp period) and the offset that makes
/// [`reference_clock`] read what a shader reads. Sets `dev.clock` and the
/// host offset on success; logs and leaves both unset otherwise.
pub fn calibrate(dev: &Arc<GpuDevice>) {
    if !dev.caps.shader_device_clock {
        log::info!("ps5gpu: no shaderDeviceClock: s_memrealtime is a per-invocation counter");
        return;
    }
    match measure(dev) {
        Ok((clock, offset, error_ns)) => {
            let mut best = OFFSET_ERROR_NS.lock().unwrap_or_else(|e| e.into_inner());
            if error_ns <= *best {
                *best = error_ns;
                OFFSET.store(offset, Ordering::Relaxed);
            }
            drop(best);
            let _ = dev.clock.set(clock);
            log::info!("ps5gpu: device clock calibrated: reference = ticks * {}/{} (±{} µs)", clock.num, clock.den, error_ns / 1000);
        }
        Err(e) => log::warn!("ps5gpu: device clock calibration failed ({e}): s_memrealtime is a per-invocation counter"),
    }
}

fn measure(dev: &Arc<GpuDevice>) -> Result<(DeviceClock, i64, u64), String> {
    let rig = Rig::new(dev)?;
    // The shortest round trip of many samples: a busy GPU (other work on the
    // queue) delays most of them by milliseconds.
    let first = rig.best(32)?;
    std::thread::sleep(std::time::Duration::from_millis(20));
    let last = rig.best(32)?;
    let (dt, dn) = (last.ticks.wrapping_sub(first.ticks), last.mid().saturating_sub(first.mid()));
    if dt == 0 || dn == 0 {
        return Err(format!("the device clock does not advance ({first:?} -> {last:?})"));
    }
    let measured = dn as f64 / dt as f64;
    let period = dev.props.limits.timestamp_period as f64;
    // The device clock normally runs at the timestamp period; when it does
    // not, the measured rate is the better guess.
    let period = if period > 0.0 && (measured / period - 1.0).abs() <= 0.02 {
        period
    } else {
        log::warn!("ps5gpu: device clock runs at {measured:.4} ns/tick, timestamp period {period:.4} ns: using the measured rate");
        measured
    };
    let clock = DeviceClock::from_period(period).ok_or_else(|| format!("unusable clock period {period} ns"))?;
    let offset = (clock.to_reference(last.ticks) as i64).wrapping_sub((last.mid() / 10) as i64);
    Ok((clock, offset, (last.after - last.before) / 2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_assembles() {
        assert_eq!(kernel().expect("kernel")[0], 0x0723_0203);
    }

    #[test]
    fn ratios() {
        assert_eq!(DeviceClock::from_period(1.0), Some(DeviceClock { num: 1, den: 10 }));
        assert_eq!(DeviceClock::from_period(10.0), Some(DeviceClock { num: 1, den: 1 }));
        assert_eq!(DeviceClock::from_period(0.0), None);
        let c = DeviceClock::from_period(52.083_333).unwrap();
        assert_eq!((c.num, c.den), (5_208_333, 1_000_000));
        // No overflow for nanosecond clocks since the epoch.
        let ns = 1_791_523_731_469_200_120u64;
        assert_eq!(DeviceClock { num: 1, den: 10 }.to_reference(ns), ns / 10);
        assert_eq!(c.to_reference(1_000_000), 5_208_333);
    }

    #[test]
    fn recalibration_accepts_only_clean_small_jumps() {
        let clock = DeviceClock { num: 1, den: 1 }; // 1 device tick = 10 ns
        let old = 1_000_000i64;
        let sample = |rtt_ns: u64, jump_ticks: i64| Sample {
            before: 0,
            after: rtt_ns,
            // mid = rtt/2; ticks chosen so the offset differs by `jump_ticks`.
            ticks: ((old + jump_ticks + (rtt_ns / 2 / 10) as i64) as u64).max(1),
        };
        // A clean sample with a small drift: accepted, offset moved.
        let s = sample(40_000, 7);
        assert_eq!(accept_recal(old, s, clock), Some(old + 7));
        // A busy device (round trip over 200 µs): rejected.
        assert_eq!(accept_recal(old, sample(300_000, 7), clock), None);
        // A glitch (jump over 2 ms): rejected.
        assert_eq!(accept_recal(old, sample(40_000, 300_000), clock), None);
        assert_eq!(accept_recal(old, sample(40_000, -300_000), clock), None);
    }

    /// The device's `s_memrealtime` source agrees with the host reference
    /// clock after calibration.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn calibrated_clock_matches_host() {
        let Ok(dev) = GpuDevice::create(false) else { return };
        if !dev.caps.shader_device_clock {
            return;
        }
        let clock = *dev.clock.get().expect("calibrated");
        let rig = Rig::new(&dev).unwrap();
        for _ in 0..4 {
            let s = rig.best(4).unwrap();
            let gpu = clock.to_reference(s.ticks) as i64;
            let (lo, hi) = (((s.before / 10) as i64).wrapping_add(OFFSET.load(Ordering::Relaxed)), ((s.after / 10) as i64).wrapping_add(OFFSET.load(Ordering::Relaxed)));
            // 5 ms of slack for the calibration error (devices of tests running in
            // parallel calibrate under load).
            assert!(gpu >= lo - 500_000 && gpu <= hi + 500_000, "gpu {gpu} host [{lo}, {hi}]");
            std::thread::sleep(std::time::Duration::from_millis(30));
        }
    }

    /// The periodic recalibration keeps the offset within a millisecond of
    /// a fresh calibration (the drift it exists to remove).
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn recalibrate_stays_close_to_a_fresh_measure() {
        let Ok(dev) = GpuDevice::create(false) else { return };
        if !dev.caps.shader_device_clock {
            return;
        }
        calibrate(&dev);
        std::thread::sleep(std::time::Duration::from_millis(50));
        recalibrate_if_due(&dev);
        recalibrate_if_due(&dev);
        let rig = Rig::new(&dev).unwrap();
        let fresh = rig.best(8).unwrap();
        let clock = *dev.clock.get().unwrap();
        let fresh_offset = (clock.to_reference(fresh.ticks) as i64).wrapping_sub((fresh.mid() / 10) as i64);
        let live = OFFSET.load(Ordering::Relaxed);
        assert!((live - fresh_offset).abs() <= 100_000, "live {live} fresh {fresh_offset}");
    }
}
