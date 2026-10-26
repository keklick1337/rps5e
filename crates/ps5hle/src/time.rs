// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Kernel clocks: process time, TSC, wall clock and timezone conversion.
//!
//! The process time counter ticks at the TSC frequency since emulator start
//! (as on the console, where titles scale it by
//! `sceKernelGetTscFrequency`); `sceKernelGetProcessTimeCounterFrequency`
//! reports that rate. The TSC is the host's own
//! `rdtsc` because guest code also reads it natively, so its frequency is
//! calibrated against the monotonic clock once at startup and rounded to
//! 100 kHz. Where the CPU enumerates the exact rate (CPUID leaf 0x15:
//! crystal clock times the TSC/crystal ratio, Intel since Skylake) and it
//! agrees with the calibration, the exact value is reported instead. A CPU
//! without an invariant TSC (CPUID 0x80000007 EDX bit 8) is reported at
//! startup: its TSC rate follows the core clock and guest timing drifts.
//!
//! Kernel clock ids map onto host clocks: the realtime family onto
//! `CLOCK_REALTIME`, the monotonic/uptime/network family onto
//! `CLOCK_MONOTONIC`, `PROCTIME` onto the process time, and the CPU-time
//! clocks onto the host CPU-time clocks.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

const OK: HleResult = 0;
const EINVAL: HleResult = 0x8002_0016u32 as i32 as i64;
const EFAULT: HleResult = 0x8002_000Eu32 as i32 as i64;
const EPERM: HleResult = 0x8002_0001u32 as i32 as i64;


pub const CLOCK_REALTIME: i32 = 0;
pub const CLOCK_VIRTUAL: i32 = 1;
pub const CLOCK_PROF: i32 = 2;
pub const CLOCK_MONOTONIC: i32 = 4;
pub const CLOCK_UPTIME: i32 = 5;
pub const CLOCK_UPTIME_PRECISE: i32 = 7;
pub const CLOCK_UPTIME_FAST: i32 = 8;
pub const CLOCK_REALTIME_PRECISE: i32 = 9;
pub const CLOCK_REALTIME_FAST: i32 = 10;
pub const CLOCK_MONOTONIC_PRECISE: i32 = 11;
pub const CLOCK_MONOTONIC_FAST: i32 = 12;
pub const CLOCK_SECOND: i32 = 13;
pub const CLOCK_THREAD_CPUTIME_ID: i32 = 14;
pub const CLOCK_PROCTIME: i32 = 15;
pub const CLOCK_EXT_NETWORK: i32 = 16;
pub const CLOCK_EXT_DEBUG_NETWORK: i32 = 17;
pub const CLOCK_EXT_AD_NETWORK: i32 = 18;
pub const CLOCK_EXT_RAW_NETWORK: i32 = 19;

/// `tz_dsttime` value reported while daylight saving time is in effect.
const DST_MET: i32 = 4;

static PROCESS_START: OnceLock<Instant> = OnceLock::new();
static TSC_FREQUENCY: OnceLock<u64> = OnceLock::new();

fn process_start() -> Instant {
    *PROCESS_START.get_or_init(Instant::now)
}

/// Host time spent before guest code ran (the shader prewarm), left out of
/// the process clock.
static EXCLUDED_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn process_elapsed() -> Duration {
    guest_elapsed(process_start().elapsed(), EXCLUDED_NS.load(std::sync::atomic::Ordering::Relaxed))
}

/// Process time after `host` since the clock started, of which
/// `excluded_ns` ran before the guest.
fn guest_elapsed(host: Duration, excluded_ns: u64) -> Duration {
    host.saturating_sub(Duration::from_nanos(excluded_ns))
}

/// Leave `d` of host work done before any guest code runs out of the
/// process clock: the title's process time and input recordings and
/// replays count from the start of guest code, whatever the emulator did
/// before (compiling the prewarm list takes seconds to minutes).
pub fn exclude_from_process_time(d: Duration) {
    EXCLUDED_NS.fetch_add(d.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}

fn read_tsc() -> u64 {
    // SAFETY: rdtsc has no side effects and is available on every x86-64 CPU.
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// Pins the calling thread to the CPU it runs on while alive, so that two
/// TSC reads come from one counter; the previous affinity comes back on
/// drop. When the pin cannot be set the thread stays unpinned.
#[cfg(unix)]
struct CurrentCpuPin {
    previous: Option<libc::cpu_set_t>,
}

#[cfg(unix)]
impl CurrentCpuPin {
    fn new() -> Self {
        // SAFETY: affinity calls on the calling thread with local sets.
        unsafe {
            let cpu = libc::sched_getcpu();
            let mut previous: libc::cpu_set_t = std::mem::zeroed();
            let size = std::mem::size_of::<libc::cpu_set_t>();
            if cpu < 0 || cpu as usize >= libc::CPU_SETSIZE as usize || libc::sched_getaffinity(0, size, &mut previous) != 0 {
                return Self { previous: None };
            }
            let mut one: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_SET(cpu as usize, &mut one);
            let pinned = libc::sched_setaffinity(0, size, &one) == 0;
            Self { previous: pinned.then_some(previous) }
        }
    }
}

#[cfg(unix)]
impl Drop for CurrentCpuPin {
    fn drop(&mut self) {
        if let Some(previous) = &self.previous {
            // SAFETY: restores the set read in new() on the same thread.
            unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), previous) };
        }
    }
}

#[cfg(windows)]
struct CurrentCpuPin {
    previous: usize,
}

#[cfg(windows)]
impl CurrentCpuPin {
    fn new() -> Self {
        use windows_sys::Win32::System::Threading::{GetCurrentProcessorNumber, GetCurrentThread, SetThreadAffinityMask};
        // SAFETY: affinity calls on the calling thread's pseudo handle.
        unsafe {
            let cpu = GetCurrentProcessorNumber();
            if cpu as usize >= usize::BITS as usize {
                return Self { previous: 0 };
            }
            Self { previous: SetThreadAffinityMask(GetCurrentThread(), 1usize << cpu) }
        }
    }
}

#[cfg(windows)]
impl Drop for CurrentCpuPin {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};
        if self.previous != 0 {
            // SAFETY: restores the mask returned in new() on the same thread.
            unsafe { SetThreadAffinityMask(GetCurrentThread(), self.previous) };
        }
    }
}

fn calibrate_tsc_sample() -> Option<u64> {
    let _pin = CurrentCpuPin::new();
    let t0 = read_tsc();
    let c0 = Instant::now();
    let min = Duration::from_millis(50);
    let (t1, c1) = loop {
        std::hint::spin_loop();
        let c1 = Instant::now();
        if c1 - c0 >= min {
            break (read_tsc(), c1);
        }
    };
    let dt = t1.checked_sub(t0)?;
    let dc = (c1 - c0).as_nanos() as u64;
    (dt != 0 && dc != 0).then(|| ((dt as u128 * 1_000_000_000u128) / dc as u128) as u64)
}

/// Round a calibrated frequency to the nearest 100 kHz.
fn round_frequency(freq: u64) -> u64 {
    const ROUND_TO: u64 = 100_000;
    let m = freq % ROUND_TO;
    let r = if m >= ROUND_TO / 2 { freq - m + ROUND_TO } else { freq - m };
    if r == 0 { freq } else { r }
}

/// The TSC rate the CPU enumerates in CPUID leaf 0x15 (ECX crystal Hz
/// times EBX/EAX), when all three are reported.
fn cpuid_tsc_frequency() -> Option<u64> {
    use core::arch::x86_64::__cpuid;
    // Leaf 0x15 only where the maximum basic leaf includes it.
    if __cpuid(0).eax < 0x15 {
        return None;
    }
    let leaf = __cpuid(0x15);
    tsc_from_leaf15(leaf.eax, leaf.ebx, leaf.ecx)
}

fn tsc_from_leaf15(denominator: u32, numerator: u32, crystal_hz: u32) -> Option<u64> {
    (denominator != 0 && numerator != 0 && crystal_hz != 0).then(|| crystal_hz as u64 * numerator as u64 / denominator as u64)
}

/// The TSC runs at a constant rate in every P-, C- and T-state.
fn invariant_tsc() -> bool {
    use core::arch::x86_64::__cpuid;
    __cpuid(0x8000_0000).eax >= 0x8000_0007 && __cpuid(0x8000_0007).edx & (1 << 8) != 0
}

/// TSC frequency reported when it can be neither enumerated nor calibrated.
const FALLBACK_TSC_FREQUENCY: u64 = 1_000_000_000;

/// The reported TSC frequency: the enumerated one when the calibration
/// confirms it (within 1%: a hypervisor may enumerate a rate it does not
/// provide), else the calibrated one rounded to 100 kHz.
fn select_tsc_frequency(enumerated: Option<u64>, calibrated: Option<u64>) -> u64 {
    match (enumerated, calibrated) {
        (Some(e), Some(c)) if e.abs_diff(c) * 100 <= c => e,
        (_, Some(c)) => round_frequency(c),
        (Some(e), None) => e,
        (None, None) => FALLBACK_TSC_FREQUENCY,
    }
}

/// Median of three 50 ms samples of the host TSC rate, or the rate the CPU
/// enumerates when the samples confirm it.
pub(crate) fn tsc_frequency() -> u64 {
    *TSC_FREQUENCY.get_or_init(|| {
        let mut samples: Vec<u64> = (0..3).filter_map(|_| calibrate_tsc_sample()).collect();
        samples.sort_unstable();
        let calibrated = samples.get(samples.len() / 2).copied();
        let enumerated = cpuid_tsc_frequency();
        let f = select_tsc_frequency(enumerated, calibrated);
        log::info!("[time] TSC {f} Hz (calibrated {calibrated:?}, CPUID {enumerated:?})");
        if !invariant_tsc() {
            log::warn!("[time] the host TSC is not invariant: its rate follows the core clock and guest timing drifts");
        }
        f
    })
}

/// Nanoseconds since process start (the 1 GHz timeline of the display
/// clock).
pub(crate) fn process_time_counter() -> u64 {
    process_elapsed().as_nanos() as u64
}

/// `ns` nanoseconds as ticks of a `hz` counter, without overflow.
fn ns_to_ticks(ns: u64, hz: u64) -> u64 {
    const NS: u64 = 1_000_000_000;
    (ns / NS) * hz + ((ns % NS) as u128 * hz as u128 / NS as u128) as u64
}

/// sceKernelGetProcessTimeCounter: process time in TSC ticks.
pub(crate) fn process_time_counter_ticks() -> u64 {
    ns_to_ticks(process_time_counter(), tsc_frequency())
}

/// Process time in microseconds.
pub(crate) fn process_time_usec() -> u64 {
    process_elapsed().as_micros() as u64
}

/// Current host TSC value and its calibrated frequency.
pub(crate) fn tsc_now() -> u64 {
    read_tsc()
}

pub(crate) fn tsc_hz() -> u64 {
    tsc_frequency()
}

/// Start the process clock and calibrate the TSC before guest code runs
/// (calibration spins for ~150 ms and must not happen inside a trap).
pub fn init() {
    process_start();
    tsc_frequency();
}

/// The host clock behind a kernel clock id.
pub(crate) fn host_clock(clock_id: i32) -> Option<libc::clockid_t> {
    match clock_id {
        CLOCK_REALTIME | CLOCK_REALTIME_PRECISE | CLOCK_REALTIME_FAST | CLOCK_SECOND => Some(libc::CLOCK_REALTIME),
        CLOCK_MONOTONIC | CLOCK_UPTIME | CLOCK_UPTIME_PRECISE | CLOCK_UPTIME_FAST | CLOCK_MONOTONIC_PRECISE
        | CLOCK_MONOTONIC_FAST | CLOCK_EXT_NETWORK | CLOCK_EXT_DEBUG_NETWORK | CLOCK_EXT_AD_NETWORK
        | CLOCK_EXT_RAW_NETWORK => Some(libc::CLOCK_MONOTONIC),
        _ => None,
    }
}

fn host_now(clock: libc::clockid_t) -> Option<Duration> {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: plain clock read into a local.
    (unsafe { libc::clock_gettime(clock, &mut ts) } == 0).then(|| Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32))
}

/// A CPU-time clock id (`CPUCLOCK_BIT` set): the process clock of the own
/// process or the clock of a running guest thread.
fn cpu_clock_now(clock_id: i32) -> Option<Duration> {
    use crate::pthread::{CPUCLOCK_ID_MASK, CPUCLOCK_PROCESS_BIT};
    let bits = clock_id as u32;
    let id = bits & CPUCLOCK_ID_MASK;
    if bits & CPUCLOCK_PROCESS_BIT != 0 {
        return (id == std::process::id()).then(|| host_now(libc::CLOCK_PROCESS_CPUTIME_ID)).flatten();
    }
    crate::pthread::thread_cpu_time(id)
}

fn is_cpu_clock(clock_id: i32) -> bool {
    clock_id as u32 & crate::pthread::CPUCLOCK_BIT != 0
}

/// Current value of a kernel clock, or None for an unknown clock id.
pub(crate) fn clock_now(clock_id: i32) -> Option<Duration> {
    if is_cpu_clock(clock_id) {
        return cpu_clock_now(clock_id);
    }
    match clock_id {
        CLOCK_PROCTIME => Some(Duration::from_micros(process_time_usec())),
        CLOCK_THREAD_CPUTIME_ID => host_now(libc::CLOCK_THREAD_CPUTIME_ID),
        CLOCK_VIRTUAL | CLOCK_PROF => host_now(libc::CLOCK_PROCESS_CPUTIME_ID),
        CLOCK_SECOND => host_now(libc::CLOCK_REALTIME).map(|d| Duration::from_secs(d.as_secs())),
        _ => host_now(host_clock(clock_id)?),
    }
}

/// (minutes west of UTC, dst flag) like `gettimezone`.
fn timezone() -> (i32, i32) {
    // SAFETY: libc time conversion into locals.
    unsafe {
        let now = libc::time(std::ptr::null_mut());
        let mut local: libc::tm = std::mem::zeroed();
        let mut utc: libc::tm = std::mem::zeroed();
        libc::localtime_r(&now, &mut local);
        libc::gmtime_r(&now, &mut utc);
        let dst = if local.tm_isdst > 0 { DST_MET } else { 0 };
        let l = libc::mktime(&mut local);
        let u = libc::mktime(&mut utc);
        (((u - l) / 60) as i32, dst)
    }
}

fn write_timespec(ctx: &CpuContext, addr: u64, d: Duration) -> bool {
    ctx.write_u64(addr, d.as_secs()) && ctx.write_u64(addr + 8, d.subsec_nanos() as u64)
}

/// sceKernelGetProcessTime() -> microseconds since process start
fn cb_get_process_time(_ctx: &mut CpuContext) -> HleResult {
    process_time_usec() as i64
}

/// sceKernelGetProcessTimeCounter() -> ticks since process start
fn cb_get_process_time_counter(_ctx: &mut CpuContext) -> HleResult {
    process_time_counter_ticks() as i64
}

/// sceKernelGetProcessTimeCounterFrequency(): the TSC frequency.
fn cb_get_process_time_counter_frequency(_ctx: &mut CpuContext) -> HleResult {
    tsc_frequency() as i64
}

fn cb_read_tsc(_ctx: &mut CpuContext) -> HleResult {
    read_tsc() as i64
}

fn cb_get_tsc_frequency(_ctx: &mut CpuContext) -> HleResult {
    tsc_frequency() as i64
}

/// sceKernelClockGettime(clockid, timespec*): the clock is checked first
/// (EINVAL), then the result pointer (EFAULT), as FreeBSD's
/// sys_clock_gettime reads the clock before copying out.
fn clock_gettime(ctx: &CpuContext, clock_id: i32, ts: u64) -> HleResult {
    let Some(now) = clock_now(clock_id) else { return EINVAL };
    if ts == 0 || !write_timespec(ctx, ts, now) {
        return EFAULT;
    }
    OK
}

fn cb_sce_clock_gettime(ctx: &mut CpuContext) -> HleResult {
    clock_gettime(ctx, ctx.rdi() as i32, ctx.rsi())
}

/// clock_gettime(clockid, timespec*): POSIX, errno and -1 on failure.
fn cb_clock_gettime(ctx: &mut CpuContext) -> HleResult {
    let r = clock_gettime(ctx, ctx.rdi() as i32, ctx.rsi());
    crate::kernel::posix_call(ctx, r)
}

/// sceKernelClockGetres(clockid, timespec*): like FreeBSD, a null result
/// asks for nothing and succeeds.
fn clock_getres(ctx: &CpuContext, clock_id: i32, ts: u64) -> HleResult {
    if ts == 0 {
        return OK;
    }
    let res = match clock_id {
        CLOCK_PROCTIME | CLOCK_THREAD_CPUTIME_ID | CLOCK_VIRTUAL | CLOCK_PROF => {
            let f = tsc_frequency();
            Duration::from_nanos(if f != 0 { ((1_000_000_000 + (f >> 1)) / f).max(1) } else { 1 })
        }
        id if is_cpu_clock(id) => {
            // A process clock answers with the CPU-time resolution for any
            // process id (FreeBSD's getres does not look the process up).
            if id as u32 & crate::pthread::CPUCLOCK_PROCESS_BIT == 0 && cpu_clock_now(id).is_none() {
                return EINVAL;
            }
            let f = tsc_frequency();
            Duration::from_nanos(if f != 0 { ((1_000_000_000 + (f >> 1)) / f).max(1) } else { 1 })
        }
        _ => {
            let Some(clock) = host_clock(clock_id) else { return EINVAL };
            let mut t = libc::timespec { tv_sec: 0, tv_nsec: 0 };
            // SAFETY: plain clock query into a local.
            if unsafe { libc::clock_getres(clock, &mut t) } != 0 {
                return EINVAL;
            }
            Duration::new(t.tv_sec as u64, t.tv_nsec as u32)
        }
    };
    if !write_timespec(ctx, ts, res) {
        return EFAULT;
    }
    OK
}

fn cb_sce_clock_getres(ctx: &mut CpuContext) -> HleResult {
    clock_getres(ctx, ctx.rdi() as i32, ctx.rsi())
}

/// clock_getres(clockid, timespec*): POSIX.
fn cb_clock_getres(ctx: &mut CpuContext) -> HleResult {
    let r = clock_getres(ctx, ctx.rdi() as i32, ctx.rsi());
    crate::kernel::posix_call(ctx, r)
}

fn write_timeval(ctx: &CpuContext, tv: u64) -> bool {
    let now = host_now(libc::CLOCK_REALTIME).unwrap_or_default();
    ctx.write_u64(tv, now.as_secs()) && ctx.write_u64(tv + 8, now.subsec_micros() as u64)
}

/// The time zone settimeofday stored for the process: (minuteswest,
/// dsttime), reported by gettimeofday; zero until one is set.
static PROCESS_TIMEZONE: std::sync::Mutex<(u32, u32)> = std::sync::Mutex::new((0, 0));

/// gettimeofday(timeval*, timezone*) — POSIX; the timezone is the one
/// settimeofday stored (zero by default).
fn cb_gettimeofday(ctx: &mut CpuContext) -> HleResult {
    let (tv, tz) = (ctx.rdi(), ctx.rsi());
    if tv != 0 && !write_timeval(ctx, tv) {
        return crate::kernel::posix_call(ctx, EFAULT);
    }
    let (west, dst) = *PROCESS_TIMEZONE.lock().unwrap_or_else(|e| e.into_inner());
    if tz != 0 && !(ctx.write_u32(tz, west) && ctx.write_u32(tz + 4, dst)) {
        return crate::kernel::posix_call(ctx, EFAULT);
    }
    0
}

/// settimeofday(timeval*, timezone*): as FreeBSD's kern_settimeofday, a
/// negative second count or microseconds outside 0..=999999 is EINVAL
/// before anything is stored. The host clock is never set: a valid time is
/// refused with EPERM (a process without PRIV_SETTIMEOFDAY) and its zone is
/// not stored, as FreeBSD stores the zone only after setting the clock. A
/// zone alone is kept for the process.
fn cb_settimeofday(ctx: &mut CpuContext) -> HleResult {
    let (tv, tz) = (ctx.rdi(), ctx.rsi());
    if tv != 0 {
        let (Some(sec), Some(usec)) = (ctx.read_u64(tv), ctx.read_u64(tv + 8)) else {
            return crate::kernel::posix_call(ctx, EFAULT);
        };
        if (sec as i64) < 0 || !(0..1_000_000).contains(&(usec as i64)) {
            return crate::kernel::posix_call(ctx, EINVAL);
        }
        log::warn!("[time] settimeofday({sec}.{usec:06}): the host clock is not changed");
        return crate::kernel::posix_call(ctx, EPERM);
    }
    if tz != 0 {
        let (Some(west), Some(dst)) = (ctx.read_u32(tz), ctx.read_u32(tz + 4)) else {
            return crate::kernel::posix_call(ctx, EFAULT);
        };
        *PROCESS_TIMEZONE.lock().unwrap_or_else(|e| e.into_inner()) = (west, dst);
    }
    0
}

/// sceKernelGettimeofday(timeval*)
fn cb_sce_gettimeofday(ctx: &mut CpuContext) -> HleResult {
    let tv = ctx.rdi();
    if tv == 0 || !write_timeval(ctx, tv) {
        return EFAULT;
    }
    OK
}

/// sceKernelGettimezone(timezone*): { int minuteswest; int dsttime; }
fn cb_gettimezone(ctx: &mut CpuContext) -> HleResult {
    let tz = ctx.rdi();
    if tz == 0 {
        return EFAULT;
    }
    let (west, dst) = timezone();
    if !(ctx.write_u32(tz, west as u32) && ctx.write_u32(tz + 4, dst as u32)) {
        return EFAULT;
    }
    OK
}

/// Offset east of UTC (seconds) and whether daylight saving is in effect,
/// of the host time zone at the UTC instant `t`.
fn host_zone(t: i64) -> (i64, bool) {
    extern "C" {
        fn tzset();
    }
    // SAFETY: tzset reads TZ; localtime_r writes only into the local tm.
    unsafe {
        tzset();
        let mut tm: libc::tm = std::mem::zeroed();
        if libc::localtime_r(&(t as libc::time_t), &mut tm).is_null() {
            return (0, false);
        }
        (tm.tm_gmtoff as i64, tm.tm_isdst > 0)
    }
}

/// A zone at one instant: the standard offset east of UTC, the daylight
/// saving in effect, and the zone's daylight saving amount whether or not
/// it is in effect (found by sampling the year around the instant).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ZoneAt {
    standard_east: i64,
    daylight: i64,
    amount: i64,
}

fn zone_at(t: i64, zone: &dyn Fn(i64) -> (i64, bool)) -> ZoneAt {
    const MONTH: i64 = 2_629_746;
    let (offset, dst) = zone(t);
    let mut amount = 0;
    'search: for month in 1..=12 {
        for direction in [1, -1] {
            let (other, other_dst) = zone(t + direction * month * MONTH);
            if other_dst != dst {
                amount = if dst { offset - other } else { other - offset };
                break 'search;
            }
        }
    }
    let daylight = if dst { amount } else { 0 };
    ZoneAt { standard_east: offset - daylight, daylight, amount }
}

/// `(local, zone)` of the UTC instant `utc`.
fn utc_to_local(utc: i64, zone: &dyn Fn(i64) -> (i64, bool)) -> (i64, ZoneAt) {
    let z = zone_at(utc, zone);
    (utc + z.standard_east + z.daylight, z)
}

/// `(utc, zone)` of the local time `local`: the zone is the one in effect
/// at the instant first guessed with the offset of `local` read as UTC
/// (in a repeated or skipped hour, the offset one standard offset earlier).
fn local_to_utc(local: i64, zone: &dyn Fn(i64) -> (i64, bool)) -> (i64, ZoneAt) {
    let guess = zone_at(local, zone);
    let z = zone_at(local - guess.standard_east - guess.daylight, zone);
    (local - z.standard_east - z.daylight, z)
}

/// sceKernelConvertUtcToLocaltime(utc, local*, SceKernelTimesec*, u64* dstsec):
/// with the host zone at that instant, as libc localtime does. The timesec
/// holds { t, west_sec, dst_sec } where west_sec is the standard offset in
/// seconds east of UTC.
fn cb_convert_utc_to_localtime(ctx: &mut CpuContext) -> HleResult {
    let (utc, local_out, timesec_out, dst_out) = (ctx.rdi() as i64, ctx.rsi(), ctx.rdx(), ctx.rcx());
    let (local, z) = utc_to_local(utc, &host_zone);
    if local_out != 0 && !ctx.write_u64(local_out, local as u64) {
        return EFAULT;
    }
    if timesec_out != 0
        && !(ctx.write_u64(timesec_out, utc as u64)
            && ctx.write_u32(timesec_out + 8, z.standard_east as u32)
            && ctx.write_u32(timesec_out + 12, z.daylight as u32))
    {
        return EFAULT;
    }
    if dst_out != 0 && !ctx.write_u64(dst_out, z.daylight as u64) {
        return EFAULT;
    }
    OK
}

/// sceKernelConvertLocaltimeToUtc(local, reserved, utc*, timezone*, int* dstsec):
/// with the host zone at that instant. The timezone gets { minuteswest,
/// dsttime } of the zone there; dstsec the zone's daylight saving amount
/// whether or not it is in effect, which libc's mktime subtracts when
/// tm_isdst is set outside summer time.
fn cb_convert_localtime_to_utc(ctx: &mut CpuContext) -> HleResult {
    let (local, utc_out, tz_out, dst_out) = (ctx.rdi() as i64, ctx.rdx(), ctx.rcx(), ctx.r8());
    if tz_out == 0 {
        return EINVAL;
    }
    let (utc, z) = local_to_utc(local, &host_zone);
    let dsttime = if z.daylight != 0 { DST_MET } else { 0 };
    if !(ctx.write_u32(tz_out, (-z.standard_east / 60) as u32) && ctx.write_u32(tz_out + 4, dsttime as u32)) {
        return EFAULT;
    }
    if utc_out != 0 && !ctx.write_u64(utc_out, utc as u64) {
        return EFAULT;
    }
    if dst_out != 0 && !ctx.write_u32(dst_out, z.amount as u32) {
        return EFAULT;
    }
    OK
}

pub fn register(reg: &mut HleRegistry) {
    init();
    reg.register_fn("4J2sUJmuHZQ", "sceKernelGetProcessTime", "libKernel", cb_get_process_time);
    reg.register_fn("fgxnMeTNUtY", "sceKernelGetProcessTimeCounter", "libKernel", cb_get_process_time_counter);
    reg.register_fn("BNowx2l588E", "sceKernelGetProcessTimeCounterFrequency", "libKernel", cb_get_process_time_counter_frequency);
    reg.register_fn("-2IRUCO--PM", "sceKernelReadTsc", "libKernel", cb_read_tsc);
    reg.register_fn("1j3S3n-tTW4", "sceKernelGetTscFrequency", "libKernel", cb_get_tsc_frequency);
    reg.register_fn("lLMT9vJAck0", "clock_gettime", "libKernel", cb_clock_gettime);
    reg.register_fn("QBi7HCK03hw", "sceKernelClockGettime", "libKernel", cb_sce_clock_gettime);
    reg.register_fn("smIj7eqzZE8", "clock_getres", "libKernel", cb_clock_getres);
    reg.register_fn("VdXIDAbJ3tQ", "settimeofday", "libKernel", cb_settimeofday);
    reg.register_fn("wRYVA5Zolso", "sceKernelClockGetres", "libKernel", cb_sce_clock_getres);
    reg.register_fn("n88vx3C5nW8", "gettimeofday", "libKernel", cb_gettimeofday);
    // The sceSp* wrapper of libSceSpSysCallWrapper forwards to gettimeofday.
    reg.register_fn("SGSNz6luI2s", "sceSpKernelGettimeofday", "libSceSpSysCallWrapper", cb_gettimeofday);
    reg.register_fn("ejekcaNQNq0", "sceKernelGettimeofday", "libKernel", cb_sce_gettimeofday);
    reg.register_fn("kOcnerypnQA", "sceKernelGettimezone", "libKernel", cb_gettimezone);
    reg.register_fn("-o5uEDpN+oY", "sceKernelConvertUtcToLocaltime", "libKernel", cb_convert_utc_to_localtime);
    reg.register_fn("0NTHN1NKONI", "sceKernelConvertLocaltimeToUtc", "libKernel", cb_convert_localtime_to_utc);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister;
    use ps5vm::{MemProtect, VirtualMemory};

    #[test]
    fn cpu_pin_holds_one_cpu_and_restores_the_affinity() {
        std::thread::spawn(|| {
            let count = || unsafe {
                let mut set: libc::cpu_set_t = std::mem::zeroed();
                assert_eq!(libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set), 0);
                libc::CPU_COUNT(&set)
            };
            let before = count();
            {
                let pin = CurrentCpuPin::new();
                assert!(pin.previous.is_some());
                assert_eq!(count(), 1);
            }
            assert_eq!(count(), before);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn tsc_frequency_is_plausible_and_rounded() {
        let f = tsc_frequency();
        assert!(f > 100_000_000 && f < 10_000_000_000, "tsc {f}");
        assert!(f % 100_000 == 0 || Some(f) == cpuid_tsc_frequency(), "tsc {f}");
        assert_eq!(round_frequency(2_999_949_999), 2_999_900_000);
        assert_eq!(round_frequency(2_999_950_000), 3_000_000_000);
    }

    #[test]
    fn enumerated_tsc_frequency_needs_confirmation() {
        // An Intel leaf 0x15: 38.4 MHz crystal, ratio 125/2 -> 2.4 GHz.
        assert_eq!(tsc_from_leaf15(2, 125, 38_400_000), Some(2_400_000_000));
        assert_eq!(tsc_from_leaf15(2, 125, 0), None);
        assert_eq!(tsc_from_leaf15(0, 125, 38_400_000), None);
        assert_eq!(select_tsc_frequency(Some(2_400_000_000), Some(2_399_987_654)), 2_400_000_000);
        // A hypervisor enumerating a rate that the calibration contradicts.
        assert_eq!(select_tsc_frequency(Some(2_400_000_000), Some(3_000_012_345)), 3_000_000_000);
        assert_eq!(select_tsc_frequency(None, Some(4_299_960_001)), 4_300_000_000);
        assert_eq!(select_tsc_frequency(Some(1_900_000_000), None), 1_900_000_000);
        assert_eq!(select_tsc_frequency(None, None), FALLBACK_TSC_FREQUENCY);
    }

    #[test]
    fn clock_ids_follow_kernel_families() {
        for id in [CLOCK_REALTIME, CLOCK_REALTIME_PRECISE, CLOCK_REALTIME_FAST, CLOCK_SECOND] {
            assert_eq!(host_clock(id), Some(libc::CLOCK_REALTIME));
        }
        for id in [4, 5, 7, 8, 11, 12, 16, 17, 18, 19] {
            assert_eq!(host_clock(id), Some(libc::CLOCK_MONOTONIC));
        }
        for id in [3, 6, 20, -1] {
            assert_eq!(host_clock(id), None);
            assert!(clock_now(id).is_none());
        }
        assert_eq!(clock_now(CLOCK_SECOND).unwrap().subsec_nanos(), 0);
        assert!(clock_now(CLOCK_PROCTIME).is_some());
        assert!(clock_now(CLOCK_THREAD_CPUTIME_ID).is_some());
    }

    #[test]
    fn clock_exports_check_arguments() {
        const BASE: u64 = 0x5CCA_0000_0000;
        let vm = VirtualMemory::new();
        vm.map(BASE, 0x10000, MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(CpuRegister::Rdi, 0);
        ctx.set(CpuRegister::Rsi, 0);
        assert_eq!(cb_sce_clock_gettime(&mut ctx), EFAULT);
        // The clock is checked before the pointer.
        ctx.set(CpuRegister::Rdi, 999);
        assert_eq!(cb_sce_clock_gettime(&mut ctx), EINVAL);
        ctx.set(CpuRegister::Rdi, CLOCK_MONOTONIC as u64);
        assert_eq!(cb_sce_clock_gettime(&mut ctx), EFAULT);
        ctx.set(CpuRegister::Rdi, 0);
        // A null resolution asks for nothing (FreeBSD clock_getres).
        assert_eq!(cb_sce_clock_getres(&mut ctx), OK);
        for id in [CLOCK_REALTIME, CLOCK_VIRTUAL, CLOCK_MONOTONIC, CLOCK_SECOND, CLOCK_PROCTIME] {
            ctx.set(CpuRegister::Rdi, id as u64);
            assert_eq!(cb_clock_getres(&mut ctx), 0);
        }
        // gettimeofday stores only what it is given.
        ctx.set(CpuRegister::Rdi, 0);
        assert_eq!(cb_gettimeofday(&mut ctx), 0);
        ctx.write_u64(BASE + 0x500, u64::MAX);
        ctx.set(CpuRegister::Rsi, BASE + 0x500);
        assert_eq!(cb_gettimeofday(&mut ctx), 0);
        assert_eq!(ctx.read_u64(BASE + 0x500), Some(0));
        // settimeofday: a bad time is EINVAL and stores nothing, a valid
        // one is refused (the host clock is not set) without its zone; a
        // zone alone is kept and gettimeofday reports it.
        let (tv, tz) = (BASE + 0x600, BASE + 0x640);
        ctx.write_u32(tz, 60);
        ctx.write_u32(tz + 4, 1);
        for (sec, usec) in [(0i64, 1_000_000i64), (0, -1), (-1, 0)] {
            ctx.write_u64(tv, sec as u64);
            ctx.write_u64(tv + 8, usec as u64);
            ctx.set(CpuRegister::Rdi, tv);
            ctx.set(CpuRegister::Rsi, tz);
            assert_eq!(cb_settimeofday(&mut ctx), -1);
        }
        ctx.write_u64(tv, 1_000_000_000);
        ctx.write_u64(tv + 8, 0);
        assert_eq!(cb_settimeofday(&mut ctx), -1);
        ctx.set(CpuRegister::Rdi, 0);
        ctx.set(CpuRegister::Rsi, BASE + 0x500);
        assert_eq!(cb_gettimeofday(&mut ctx), 0);
        assert_eq!(ctx.read_u64(BASE + 0x500), Some(0));
        ctx.set(CpuRegister::Rsi, 0);
        assert_eq!(cb_settimeofday(&mut ctx), 0);
        ctx.set(CpuRegister::Rsi, tz);
        assert_eq!(cb_settimeofday(&mut ctx), 0);
        ctx.set(CpuRegister::Rsi, BASE + 0x500);
        assert_eq!(cb_gettimeofday(&mut ctx), 0);
        assert_eq!((ctx.read_u32(BASE + 0x500), ctx.read_u32(BASE + 0x504)), (Some(60), Some(1)));
        *PROCESS_TIMEZONE.lock().unwrap() = (0, 0);
        // A process CPU clock: its resolution is the CPU-time one for any
        // process id; its time only for the own process.
        let other = (crate::pthread::CPUCLOCK_BIT | crate::pthread::CPUCLOCK_PROCESS_BIT | (std::process::id() + 1)) as u64;
        ctx.set(CpuRegister::Rdi, other);
        ctx.set(CpuRegister::Rsi, BASE + 0x680);
        assert_eq!(cb_sce_clock_getres(&mut ctx), OK);
        ctx.set(CpuRegister::Rdi, CLOCK_PROCTIME as u64);
        ctx.set(CpuRegister::Rsi, BASE + 0x6a0);
        assert_eq!(cb_sce_clock_getres(&mut ctx), OK);
        assert_eq!(ctx.read_u64(BASE + 0x688), ctx.read_u64(BASE + 0x6a8));
        ctx.set(CpuRegister::Rdi, other);
        assert_eq!(cb_sce_clock_gettime(&mut ctx), EINVAL);
        ctx.set(CpuRegister::Rsi, 0);
        ctx.set(CpuRegister::Rdi, 3);
        ctx.set(CpuRegister::Rsi, BASE);
        assert_eq!(cb_sce_clock_gettime(&mut ctx), EINVAL);
        assert_eq!(cb_sce_clock_getres(&mut ctx), EINVAL);
        ctx.set(CpuRegister::Rdi, CLOCK_MONOTONIC as u64);
        assert_eq!(cb_sce_clock_gettime(&mut ctx), OK);
        ctx.set(CpuRegister::Rdi, CLOCK_PROCTIME as u64);
        assert_eq!(cb_sce_clock_getres(&mut ctx), OK);
        assert_eq!(ctx.read_u64(BASE), Some(0));
        assert!(ctx.read_u64(BASE + 8).unwrap() >= 1);
        ctx.set(CpuRegister::Rdi, 0);
        assert_eq!(cb_sce_gettimeofday(&mut ctx), EFAULT);
        assert_eq!(cb_gettimezone(&mut ctx), EFAULT);
        ctx.set(CpuRegister::Rdi, BASE);
        assert_eq!(cb_gettimezone(&mut ctx), OK);
        let dst = ctx.read_u32(BASE + 4).unwrap() as i32;
        assert!(dst == 0 || dst == DST_MET);
        // LocaltimeToUtc requires the timezone out-parameter.
        ctx.set(CpuRegister::Rdi, 1_700_000_000);
        ctx.set(CpuRegister::Rdx, 0);
        ctx.set(CpuRegister::Rcx, 0);
        ctx.set(CpuRegister::R8, 0);
        assert_eq!(cb_convert_localtime_to_utc(&mut ctx), EINVAL);
        ctx.set(CpuRegister::Rdx, BASE + 0x100);
        ctx.set(CpuRegister::Rcx, BASE + 0x200);
        assert_eq!(cb_convert_localtime_to_utc(&mut ctx), OK);
        let utc = ctx.read_u64(BASE + 0x100).unwrap() as i64;
        assert_eq!(utc, local_to_utc(1_700_000_000, &host_zone).0);
        // UtcToLocaltime round trip.
        ctx.set(CpuRegister::Rdi, utc as u64);
        ctx.set(CpuRegister::Rsi, BASE + 0x300);
        ctx.set(CpuRegister::Rdx, BASE + 0x400);
        ctx.set(CpuRegister::Rcx, 0);
        assert_eq!(cb_convert_utc_to_localtime(&mut ctx), OK);
        assert_eq!(ctx.read_u64(BASE + 0x300), Some(1_700_000_000));
        assert_eq!(ctx.read_u64(BASE + 0x400), Some(utc as u64));
        let _ = vm.unmap(BASE, 0x10000);
    }


    /// A zone like Central Europe: UTC+1, +1 h of summer time from day 90 to
    /// day 300 of every 365-day "year".
    fn cet(t: i64) -> (i64, bool) {
        let day = t.rem_euclid(365 * 86400) / 86400;
        let summer = (90..300).contains(&day);
        (3600 + if summer { 3600 } else { 0 }, summer)
    }

    /// The conversions use the zone at the instant, not today's: a winter
    /// instant converts with the standard offset, a summer one with summer
    /// time; LocaltimeToUtc reports the zone's summer amount in winter too.
    #[test]
    fn time_zone_conversion_at_the_instant() {
        let winter = 10 * 86400;
        let summer = 200 * 86400;
        assert_eq!(zone_at(winter, &cet), ZoneAt { standard_east: 3600, daylight: 0, amount: 3600 });
        assert_eq!(zone_at(summer, &cet), ZoneAt { standard_east: 3600, daylight: 3600, amount: 3600 });
        assert_eq!(utc_to_local(winter, &cet).0, winter + 3600);
        assert_eq!(utc_to_local(summer, &cet).0, summer + 7200);
        for utc in [winter, summer, 89 * 86400 + 3000, 300 * 86400 + 5000] {
            let (local, _) = utc_to_local(utc, &cet);
            assert_eq!(local_to_utc(local, &cet).0, utc, "round trip at {utc}");
        }
        let (_, z) = local_to_utc(winter + 3600, &cet);
        assert_eq!((z.daylight, z.amount), (0, 3600));
        // A zone without summer time.
        let utc0 = |_t: i64| (0i64, false);
        assert_eq!(zone_at(summer, &utc0), ZoneAt { standard_east: 0, daylight: 0, amount: 0 });
        let tokyo = |_t: i64| (9 * 3600i64, false);
        assert_eq!(local_to_utc(summer + 9 * 3600, &tokyo).0, summer);
    }

    /// The process clock starts before it is first read: the first values
    /// are small and never go backwards (a start taken after the clock
    /// read would wrap to about 2^64).
    #[test]
    fn process_time_starts_before_its_first_read() {
        let first = process_time_usec();
        assert!(first < 60 * 1_000_000 * 60, "first process time {first}");
        assert!(process_time_usec() >= first);
        assert!(process_time_counter() < 60 * 1_000_000_000 * 60);
        assert_eq!(ns_to_ticks(1_500_000_000, 3_400_000_000), 5_100_000_000);
        assert_eq!(ns_to_ticks(u64::MAX / 4, 4_000_000_000), (u64::MAX / 4) / 1_000_000_000 * 4_000_000_000 + ((u64::MAX / 4) % 1_000_000_000) * 4);
        let ticks = process_time_counter_ticks();
        assert!(process_time_counter_ticks() >= ticks);
    }

    /// The process clock counts from emulator start (`init`), not from the
    /// first read: a title that reads it 50 ms after start sees 50 ms.
    #[test]
    fn process_time_counts_from_start_not_from_the_first_read() {
        process_start();
        std::thread::sleep(Duration::from_millis(50));
        assert!(process_time_usec() >= 50_000);
        assert!(process_time_counter() >= 50_000_000);
    }

    /// Host work before the guest starts is not process time.
    #[test]
    fn excluded_host_time_is_not_process_time() {
        assert_eq!(guest_elapsed(Duration::from_secs(70), 60_000_000_000), Duration::from_secs(10));
        assert_eq!(guest_elapsed(Duration::from_secs(5), 60_000_000_000), Duration::ZERO);
        assert_eq!(guest_elapsed(Duration::from_millis(1500), 0), Duration::from_millis(1500));
    }

    /// CPU-time clock ids: the own process clock reads the host process
    /// CPU time; another pid or an unknown thread id is no clock.
    #[test]
    fn cpu_clock_ids() {
        use crate::pthread::{CPUCLOCK_BIT, CPUCLOCK_PROCESS_BIT};
        let own = (CPUCLOCK_BIT | CPUCLOCK_PROCESS_BIT | std::process::id()) as i32;
        assert!(clock_now(own).is_some());
        let other = (CPUCLOCK_BIT | CPUCLOCK_PROCESS_BIT | (std::process::id() + 1)) as i32;
        assert!(clock_now(other).is_none());
        // Thread ids are handed out from 1; this one is never reached.
        assert!(clock_now((CPUCLOCK_BIT | 0x3FFF_FFF0) as i32).is_none());
    }

    #[test]
    fn time_nids_match_names() {
        for (nid, name) in [
            ("wRYVA5Zolso", "sceKernelClockGetres"),
            ("kOcnerypnQA", "sceKernelGettimezone"),
            ("QBi7HCK03hw", "sceKernelClockGettime"),
            ("lLMT9vJAck0", "clock_gettime"),
        ] {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }
}
