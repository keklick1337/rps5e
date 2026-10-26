// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Assorted libkernel services: resource usage, scheduling limits, CPU
//! number, sleeps, GPO/GPI bits, identifiers, core dump user data and
//! thread cancellation.
//!
//! Error convention: `sce*` exports return `0x8002_0000 | errno`; POSIX-named
//! exports return -1 and set the calling thread's errno, except the
//! `pthread_*` ones, which return the error number itself.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use std::cell::Cell;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

mod process;
mod system;

const LIB_KERNEL: &str = "libKernel";
const LIB_COREDUMP: &str = "libSceCoredump";

const OK: HleResult = 0;

// FreeBSD errno values.
const ESRCH: i32 = 3;
const ENOMEM: i32 = 12;
const EFAULT: i32 = 14;
const EINVAL: i32 = 22;
const ENAMETOOLONG: i32 = 63;

const fn sce_error(errno: i32) -> HleResult {
    (0x8002_0000u32 | errno as u32) as i32 as i64
}

fn posix_failure(ctx: &CpuContext, errno: i32) -> HleResult {
    crate::kernel::set_errno(ctx, errno);
    -1
}

/// Logical CPUs a game may run on (available cpumask 0x1fff).
const GUEST_CPU_COUNT: i32 = 13;

// ---------------------------------------------------------------------------
// System mode / GPO
// ---------------------------------------------------------------------------

/// sceKernelIsTrinityMode(): the emulated console is a base PS5.
fn cb_is_trinity_mode(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// Development-kit general purpose output bits. There are no LEDs or
/// switches; the bits are latched and read back through the GPI port.
static GPO_BITS: AtomicU32 = AtomicU32::new(0);

/// sceKernelSetGPO(bits)
fn cb_set_gpo(ctx: &mut CpuContext) -> HleResult {
    GPO_BITS.store(ctx.rdi() as u32, Ordering::Relaxed);
    OK
}

/// sceKernelGetGPI() -> latched GPO bits.
fn cb_get_gpi(_ctx: &mut CpuContext) -> HleResult {
    GPO_BITS.load(Ordering::Relaxed) as HleResult
}

/// Stable 16-byte OpenPSID of the emulated console.
const OPEN_PS_ID: [u8; 16] = *b"rps5e-open-psid\x01";

/// sceKernelGetOpenPsId(SceKernelOpenPsId* out)
fn cb_get_open_ps_id(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return sce_error(EINVAL);
    }
    if ctx.write_bytes(out, &OPEN_PS_ID) { OK } else { sce_error(EFAULT) }
}

/// sceKernelUuidCreate(SceKernelUuid* out): random (version 4) UUID.
fn cb_uuid_create(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return sce_error(EINVAL);
    }
    let mut b = [0u8; 16];
    let n = unsafe { libc::getrandom(b.as_mut_ptr() as *mut libc::c_void, b.len(), 0) };
    if n != b.len() as isize {
        return sce_error(ENOMEM);
    }
    let uuid = make_v4_uuid(b);
    if ctx.write_bytes(out, &uuid) { OK } else { sce_error(EFAULT) }
}

/// sceKernelCreateUuidV4(u8 out[16]): 16 random bytes with the version
/// (byte 6) and variant (byte 8) bits of an RFC 4122 version 4 UUID.
fn cb_create_uuid_v4(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return sce_error(EINVAL);
    }
    let mut b = [0u8; 16];
    crate::misc_services::fill_random(&mut b);
    b[6] = (b[6] & 0x0F) | 0x40;
    b[8] = (b[8] & 0x3F) | 0x80;
    if ctx.write_bytes(out, &b) { OK } else { sce_error(EFAULT) }
}

fn make_v4_uuid(mut b: [u8; 16]) -> [u8; 16] {
    // Fields are stored as in struct uuid: time_low (u32), time_mid (u16),
    // time_hi_and_version (u16) in host order, then the byte fields.
    let thv = (u16::from_le_bytes([b[6], b[7]]) & 0x0FFF) | 0x4000;
    b[6..8].copy_from_slice(&thv.to_le_bytes());
    b[8] = (b[8] & 0x3F) | 0x80;
    b
}

// ---------------------------------------------------------------------------
// CPU / scheduling / time
// ---------------------------------------------------------------------------

/// getargc(): argc passed to the C runtime.
fn cb_getargc(_ctx: &mut CpuContext) -> HleResult {
    crate::cxx::args().0 as i32 as HleResult
}

/// getargv(): argv passed to the C runtime.
fn cb_getargv(_ctx: &mut CpuContext) -> HleResult {
    crate::cxx::args().1 as HleResult
}

/// The CPUs a game may use (`sceKernelGetAvailableCpumask`).
pub(crate) const AVAILABLE_CPUMASK: u64 = 0x1fff;

/// sceKernelGetAvailableCpumask(): the 13 CPUs a game may use.
fn cb_get_available_cpumask(_ctx: &mut CpuContext) -> HleResult {
    AVAILABLE_CPUMASK as HleResult
}

/// sceKernelGetOperationMode(mode*, submode*): PS5 base console, no
/// sub-mode.
fn cb_get_operation_mode(ctx: &mut CpuContext) -> HleResult {
    let (mode, submode) = (ctx.rdi(), ctx.rsi());
    ctx.write_u32(mode, 2);
    ctx.write_u32(submode, 0);
    0
}

/// sceKernelSync(): flush what the guest wrote (open files and changed
/// paths), not the whole host.
fn cb_kernel_sync(_ctx: &mut CpuContext) -> HleResult {
    crate::filesystem::sync_written_paths();
    0
}

/// sceKernelWriteThrottlingStatus(u64 status[4]) (YFC3dBBipj8, library
/// libkernel_write_throttling): no throttling. The console writes
/// {0xffffffff, 0, 0, 0}; a null status is EFAULT.
fn cb_write_throttling_stub(ctx: &mut CpuContext) -> HleResult {
    let status = ctx.rdi();
    if status == 0 {
        return sce_error(EFAULT);
    }
    let mut b = [0u8; 32];
    b[..8].copy_from_slice(&0xFFFF_FFFFu64.to_le_bytes());
    if !ctx.write_bytes(status, &b) {
        return sce_error(EFAULT);
    }
    0
}

/// sceKernelError(int errno): the SCE error code of a POSIX errno value
/// (0x80020000 | errno), 0 for 0.
fn cb_kernel_error(ctx: &mut CpuContext) -> HleResult {
    let v = ctx.rdi() as i32;
    if v == 0 {
        return 0;
    }
    (v as u32).wrapping_add(0x8002_0000) as i32 as i64
}

/// The guest CPU reported for host CPU `host_cpu` to a thread with
/// `affinity`: guest affinities are bookkeeping only, so the host index
/// selects one CPU of the affinity clipped to the available CPUs (all of
/// them when that leaves none). A thread pinned to one CPU always reports
/// that CPU, and the result is always an available CPU.
pub(crate) fn guest_cpu_from_host(host_cpu: u32, affinity: u64) -> i32 {
    let mut mask = affinity & AVAILABLE_CPUMASK;
    if mask == 0 {
        mask = AVAILABLE_CPUMASK;
    }
    for _ in 0..host_cpu % mask.count_ones() {
        mask &= mask - 1;
    }
    mask.trailing_zeros() as i32
}

/// sceKernelGetCurrentCpu(): a CPU of the calling thread's affinity.
fn cb_get_current_cpu(_ctx: &mut CpuContext) -> HleResult {
    let cpu = unsafe { libc::sched_getcpu() }.max(0) as u32;
    guest_cpu_from_host(cpu, crate::pthread::current_affinity()) as HleResult
}

const SCHED_FIFO: i32 = 1;
const SCHED_RR: i32 = 3;
/// Highest priority (numerically lowest) and lowest priority.
pub(crate) const PRIORITY_HIGHEST: HleResult = 256;
const PRIORITY_LOWEST: HleResult = 767;

/// Policies with a priority range: SCHED_FIFO and SCHED_RR. A PS5
/// (firmware 13.60) answers every other policy, SCHED_OTHER included,
/// with EINVAL.
fn valid_policy(policy: i32) -> bool {
    policy == SCHED_FIFO || policy == SCHED_RR
}

/// sched_get_priority_max(policy): 767, the numerically largest priority,
/// as a PS5 (firmware 13.60) reports it.
fn cb_sched_get_priority_max(ctx: &mut CpuContext) -> HleResult {
    if !valid_policy(ctx.rdi() as i32) {
        return posix_failure(ctx, EINVAL);
    }
    PRIORITY_LOWEST
}

/// sched_get_priority_min(policy): 256.
fn cb_sched_get_priority_min(ctx: &mut CpuContext) -> HleResult {
    if !valid_policy(ctx.rdi() as i32) {
        return posix_failure(ctx, EINVAL);
    }
    PRIORITY_HIGHEST
}

/// Requested sleep of a guest timespec, or the kernel error.
fn read_sleep_request(ctx: &CpuContext, req: u64) -> Result<Duration, HleResult> {
    if req == 0 {
        return Err(sce_error(EFAULT));
    }
    let (Some(sec), Some(nsec)) = (ctx.read_u64(req), ctx.read_u64(req + 8)) else {
        return Err(sce_error(EFAULT));
    };
    let (sec, nsec) = (sec as i64, nsec as i64);
    if sec < 0 || !(0..1_000_000_000).contains(&nsec) {
        return Err(sce_error(EINVAL));
    }
    Ok(Duration::new(sec as u64, nsec as u32))
}

/// sceKernelNanosleep(const timespec* req, timespec* rem): the sleep is
/// never interrupted, and as FreeBSD's nanosleep writes the remaining time
/// only when interrupted, `rem` is left untouched.
fn cb_sce_nanosleep(ctx: &mut CpuContext) -> HleResult {
    let d = match read_sleep_request(ctx, ctx.rdi()) {
        Ok(d) => d,
        Err(e) => return e,
    };
    cancellable_sleep(d);
    OK
}

/// FreeBSD `struct rusage` is 2 timevals + 14 longs, the same layout as the
/// x86-64 Linux structure.
const RUSAGE_SIZE: usize = 144;
const RUSAGE_SELF: i32 = 0;
const RUSAGE_THREAD: i32 = 1;
const RUSAGE_CHILDREN: i32 = -1;

/// getrusage(who, struct rusage*)
fn cb_getrusage(ctx: &mut CpuContext) -> HleResult {
    let (who, out) = (ctx.rdi() as i32, ctx.rsi());
    let host_who = match who {
        RUSAGE_SELF => libc::RUSAGE_SELF,
        RUSAGE_THREAD => libc::RUSAGE_THREAD,
        RUSAGE_CHILDREN => libc::RUSAGE_CHILDREN,
        _ => return posix_failure(ctx, EINVAL),
    };
    if out == 0 {
        return posix_failure(ctx, EFAULT);
    }
    // The console reports all zeros for children (measured on hardware),
    // without asking the host.
    if who == RUSAGE_CHILDREN {
        return if ctx.write_bytes(out, &[0u8; RUSAGE_SIZE]) { 0 } else { posix_failure(ctx, EFAULT) };
    }
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(host_who, &mut ru) } != 0 {
        return posix_failure(ctx, EINVAL);
    }
    let fields: [i64; 18] = [
        ru.ru_utime.tv_sec,
        ru.ru_utime.tv_usec,
        ru.ru_stime.tv_sec,
        ru.ru_stime.tv_usec,
        ru.ru_maxrss,
        ru.ru_ixrss,
        ru.ru_idrss,
        ru.ru_isrss,
        ru.ru_minflt,
        ru.ru_majflt,
        ru.ru_nswap,
        ru.ru_inblock,
        ru.ru_oublock,
        ru.ru_msgsnd,
        ru.ru_msgrcv,
        ru.ru_nsignals,
        ru.ru_nvcsw,
        ru.ru_nivcsw,
    ];
    let mut bytes = Vec::with_capacity(RUSAGE_SIZE);
    for f in fields {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    if ctx.write_bytes(out, &bytes) { 0 } else { posix_failure(ctx, EFAULT) }
}

// ---------------------------------------------------------------------------
// Process: credentials, resource limits, sysctl, child processes
// ---------------------------------------------------------------------------

const EPERM: i32 = 1;
const ENOENT: i32 = 2;
const ECHILD: i32 = 10;
const ERANGE: i32 = 34;

/// The guest runs as root (uid 0, gid 0). Real and saved ids stay 0; the
/// effective ids follow seteuid/setegid under FreeBSD's privilege rules.
static EFFECTIVE_UID: AtomicU32 = AtomicU32::new(0);
static EFFECTIVE_GID: AtomicU32 = AtomicU32::new(0);
/// P_SUGID: the credentials have changed since the process started.
static SUGID: AtomicU32 = AtomicU32::new(0);
const REAL_ID: u32 = 0;

fn cb_getuid(_ctx: &mut CpuContext) -> HleResult {
    REAL_ID as HleResult
}

fn cb_geteuid(_ctx: &mut CpuContext) -> HleResult {
    EFFECTIVE_UID.load(Ordering::Relaxed) as HleResult
}

fn cb_getgid(_ctx: &mut CpuContext) -> HleResult {
    REAL_ID as HleResult
}

fn cb_getegid(_ctx: &mut CpuContext) -> HleResult {
    EFFECTIVE_GID.load(Ordering::Relaxed) as HleResult
}

fn cb_issetugid(_ctx: &mut CpuContext) -> HleResult {
    SUGID.load(Ordering::Relaxed) as HleResult
}

/// seteuid/setegid core: the real (= saved) id is always allowed, any other
/// one needs privilege (an effective uid of 0).
fn set_effective(id: u32, slot: &AtomicU32) -> i32 {
    if id != REAL_ID && EFFECTIVE_UID.load(Ordering::Relaxed) != 0 {
        return EPERM;
    }
    if slot.swap(id, Ordering::Relaxed) != id {
        SUGID.store(1, Ordering::Relaxed);
    }
    0
}

/// seteuid(uid)
fn cb_seteuid(ctx: &mut CpuContext) -> HleResult {
    match set_effective(ctx.rdi() as u32, &EFFECTIVE_UID) {
        0 => 0,
        e => posix_failure(ctx, e),
    }
}

/// setegid(gid)
fn cb_setegid(ctx: &mut CpuContext) -> HleResult {
    match set_effective(ctx.rdi() as u32, &EFFECTIVE_GID) {
        0 => 0,
        e => posix_failure(ctx, e),
    }
}

/// FreeBSD resource numbers (RLIM_NLIMITS = 15).
const RLIM_NLIMITS: i32 = 15;
const RLIM_INFINITY: i64 = i64::MAX;

/// The host resource behind a FreeBSD resource number; None for the
/// FreeBSD-only ones (sbsize, npts, swap, kqueues, umtxp), which are
/// unlimited by default.
fn host_resource(resource: i32) -> Option<libc::__rlimit_resource_t> {
    Some(match resource {
        0 => libc::RLIMIT_CPU,
        1 => libc::RLIMIT_FSIZE,
        2 => libc::RLIMIT_DATA,
        3 => libc::RLIMIT_STACK,
        4 => libc::RLIMIT_CORE,
        5 => libc::RLIMIT_RSS,
        6 => libc::RLIMIT_MEMLOCK,
        7 => libc::RLIMIT_NPROC,
        8 => libc::RLIMIT_NOFILE,
        10 => libc::RLIMIT_AS,
        _ => return None,
    })
}

fn guest_limit(v: libc::rlim_t) -> i64 {
    if v == libc::RLIM_INFINITY || v > RLIM_INFINITY as libc::rlim_t { RLIM_INFINITY } else { v as i64 }
}

/// getrlimit(resource, struct rlimit*): { rlim_cur i64, rlim_max i64 }.
fn cb_getrlimit(ctx: &mut CpuContext) -> HleResult {
    let (resource, out) = (ctx.rdi() as i32, ctx.rsi());
    if !(0..RLIM_NLIMITS).contains(&resource) {
        return posix_failure(ctx, EINVAL);
    }
    if out == 0 {
        return posix_failure(ctx, EFAULT);
    }
    let Some((cur, max)) = process::current_limit(resource) else { return posix_failure(ctx, EINVAL) };
    if !(ctx.write_u64(out, cur as u64) && ctx.write_u64(out + 8, max as u64)) {
        return posix_failure(ctx, EFAULT);
    }
    0
}

const CTL_HW: u32 = 6;
const HW_NCPU: u32 = 3;
const CTL_MAXNAME: u32 = 24;

/// Copy a sysctl value out like FreeBSD: with no buffer only the size is
/// reported; a short buffer gets what fits and ENOMEM. Returns the errno.
fn sysctl_copy_out(ctx: &CpuContext, value: &[u8], old: u64, oldlen_ptr: u64) -> i32 {
    if old == 0 {
        if oldlen_ptr != 0 && !ctx.write_u64(oldlen_ptr, value.len() as u64) {
            return EFAULT;
        }
        return 0;
    }
    let room = if oldlen_ptr != 0 {
        match ctx.read_u64(oldlen_ptr) {
            Some(n) => n as usize,
            None => return EFAULT,
        }
    } else {
        0
    };
    let n = room.min(value.len());
    if !ctx.write_bytes(old, &value[..n]) {
        return EFAULT;
    }
    if oldlen_ptr != 0 && !ctx.write_u64(oldlen_ptr, n as u64) {
        return EFAULT;
    }
    if n < value.len() { ENOMEM } else { 0 }
}

/// The host CPU's brand string (CPUID 0x80000002..4), for hw.model.
fn host_cpu_model() -> String {
    let mut bytes = Vec::with_capacity(48);
    for leaf in 0x8000_0002u32..=0x8000_0004 {
        let r = std::arch::x86_64::__cpuid(leaf);
        for w in [r.eax, r.ebx, r.ecx, r.edx] {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
    }
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).trim().to_string()
}

/// The executable path kern.proc.pathname reports.
const PROC_PATHNAME: &[u8] = b"/app0/eboot.bin\0";
/// Most bytes one kern.arandom read returns.
const ARANDOM_MAX: u64 = 256;

/// sysctl over a MIB: hw.ncpu (the CPUs a game may use), hw.pagesize,
/// hw.model (the host CPU), hw.physmem / hw.realmem (host memory),
/// kern.proc.pathname of this process and kern.arandom. All are read-only
/// (EPERM on a write); a longer name under a leaf is ENOTDIR.
fn sysctl_mib(ctx: &CpuContext, mib: &[u32], old: u64, oldlen: u64, new: u64) -> i32 {
    const ENOTDIR: i32 = 20;
    if is_absent_mib(mib) {
        return ENOENT;
    }
    let known = matches!(mib, [6, 1 | 2 | 3 | 5 | 7 | 12] | [1, 1 | 2 | 4 | 10 | 24 | 37 | 50]) || matches!(mib, [1, 14, 12 | 32, ..]);
    if !known {
        let leaf = mib.len() > 2 && (matches!(mib[..2], [6, 1 | 2 | 3 | 5 | 7]) || matches!(mib[..2], [1, 1 | 2 | 4 | 10 | 24 | 37 | 50]));
        if leaf {
            return ENOTDIR;
        }
        log::warn!("[kernel] sysctl: unknown MIB {mib:?}");
        return ENOENT;
    }
    if new != 0 {
        return EPERM;
    }
    // The uname names (kern.ostype, kern.osrelease, kern.version,
    // kern.hostname, hw.machine) as FreeBSD 11.4 answers them.
    let uname: Option<&[u8]> = match mib {
        [1, 1] => Some(b"FreeBSD\0"),
        [1, 2] => Some(b"11.4-RELEASE\0"),
        [1, 4] => Some(b"FreeBSD 11.4-RELEASE\n\0"),
        [1, 10] => Some(crate::libc::GUEST_HOSTNAME),
        [6, 1] => Some(b"amd64\0"),
        _ => None,
    };
    if let Some(text) = uname {
        return sysctl_copy_out(ctx, text, old, oldlen);
    }
    match mib {
        [1, 24] => sysctl_copy_out(ctx, &KERN_OSRELDATE.to_le_bytes(), old, oldlen),
        [1, 50] => rng_pseudo(ctx, old, oldlen),
        [1, 14, 32, rest @ ..] => {
            let [pid] = rest else { return EINVAL };
            if *pid as i32 != std::process::id() as i32 {
                return ESRCH;
            }
            sysctl_copy_out(ctx, &vm_map_entries(), old, oldlen)
        }
        [CTL_HW, HW_NCPU] => sysctl_copy_out(ctx, &GUEST_CPU_COUNT.to_le_bytes(), old, oldlen),
        [6, 7] => sysctl_copy_out(ctx, &(process::GUEST_PAGE_SIZE as i32).to_le_bytes(), old, oldlen),
        [6, 2] => {
            let mut model = host_cpu_model().into_bytes();
            model.push(0);
            sysctl_copy_out(ctx, &model, old, oldlen)
        }
        [6, 5 | 12] => {
            let bytes = process::physical_pages() * process::GUEST_PAGE_SIZE;
            if bytes == 0 {
                return EIO;
            }
            sysctl_copy_out(ctx, &bytes.to_le_bytes(), old, oldlen)
        }
        [1, 14, 12, rest @ ..] => {
            let [pid] = rest else { return EINVAL };
            let pid = *pid as i32;
            if pid != -1 && pid != std::process::id() as i32 {
                return ESRCH;
            }
            sysctl_copy_out(ctx, PROC_PATHNAME, old, oldlen)
        }
        _ => {
            // kern.arandom: up to 256 random bytes.
            if oldlen == 0 {
                return 0;
            }
            let Some(room) = ctx.read_u64(oldlen) else { return EFAULT };
            let count = room.min(ARANDOM_MAX);
            if old != 0 {
                let mut bytes = vec![0u8; count as usize];
                crate::misc_services::fill_random(&mut bytes);
                if !ctx.write_bytes(old, &bytes) {
                    return EFAULT;
                }
            }
            if ctx.write_u64(oldlen, count) { 0 } else { EFAULT }
        }
    }
}

const EIO: i32 = 5;

/// kern.osreldate as the console's FreeBSD 11 kernel reports it.
const KERN_OSRELDATE: i32 = 1_104_001;

/// MIBs the console answers with ENOENT (measured on a PS5, FW 12.70):
/// kern.maxproc, kern.maxfiles, kern.clockrate, kern.maxfilesperproc,
/// kern.ps_strings, hw.byteorder, hw.machine_arch, and four OIDs of the
/// dynamic range.
fn is_absent_mib(mib: &[u32]) -> bool {
    const DYNAMIC: [&[u32]; 4] =
        [&[1, 2147482816, 2147482814], &[1, 2147483151], &[1, 2147483182, 2147483181], &[1, 2147482838, 2147482834]];
    matches!(mib, [1, 6 | 7 | 12 | 27 | 32] | [6, 4 | 11]) || DYNAMIC.contains(&mib)
}

/// Names sysctlbyname answers with ENOENT on the console (measured on a
/// PS5, FW 12.70), leaving the buffer and the length alone.
const ABSENT_SYSCTL_NAMES: &[&str] = &[
    "hw.byteorder",
    "hw.cacheline_size",
    "hw.machine_arch",
    "hw.usermem",
    "kern.clockrate",
    "kern.maxfiles",
    "kern.maxfilesperproc",
    "kern.maxproc",
    "kern.ps_strings",
    "kern.sched.name",
    "kern.smp.maxcpus",
    "kern.stackprot",
    "kern.timecounter.hardware",
    "machdep.cacheline_size",
];

/// Size of one FreeBSD 11 `struct kinfo_vmentry` as kern.proc.vmmap packs
/// it (with an 8-byte path).
const VM_ENTRY_SIZE: usize = 144;

/// kern.proc.vmmap of this process: one packed kinfo_vmentry per guest
/// range (type default, the range's read/write/execute protection).
fn vm_map_entries() -> Vec<u8> {
    const KVME_TYPE_DEFAULT: i32 = 1;
    let ranges = ps5vm::ranges::snapshot();
    let mut out = Vec::with_capacity(ranges.len() * VM_ENTRY_SIZE);
    for (start, end, prot) in ranges {
        let mut e = [0u8; VM_ENTRY_SIZE];
        e[0..4].copy_from_slice(&(VM_ENTRY_SIZE as i32).to_le_bytes());
        e[4..8].copy_from_slice(&KVME_TYPE_DEFAULT.to_le_bytes());
        e[8..16].copy_from_slice(&start.to_le_bytes());
        e[16..24].copy_from_slice(&end.to_le_bytes());
        let bits = prot.read as i32 | (prot.write as i32) << 1 | (prot.exec as i32) << 2;
        e[56..60].copy_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(&e);
    }
    out
}

/// sysctl(name*, namelen, old*, oldlen*, new*, newlen)
fn cb_sysctl(ctx: &mut CpuContext) -> HleResult {
    let (name, namelen, old, oldlen, new) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx(), ctx.r8());
    if !(2..=CTL_MAXNAME).contains(&namelen) {
        return posix_failure(ctx, EINVAL);
    }
    let mib: Option<Vec<u32>> = if name == 0 { None } else { (0..namelen as u64).map(|i| ctx.read_u32(name + i * 4)).collect() };
    let Some(mib) = mib else { return posix_failure(ctx, EFAULT) };
    match sysctl_mib(ctx, &mib, old, oldlen, new) {
        0 => 0,
        e => posix_failure(ctx, e),
    }
}

/// `kern.rng_pseudo`: one 64-byte block of random bytes per call (measured
/// on a PS5, FW 12.70): a size query reports 64, a larger buffer gets 64
/// bytes and its length becomes 64, a smaller one (or a null length) gets
/// what fits and ENOMEM. Returns the errno.
fn rng_pseudo(ctx: &CpuContext, old: u64, oldlen: u64) -> i32 {
    let mut block = [0u8; 64];
    if old != 0 && oldlen != 0 {
        crate::misc_services::fill_random(&mut block);
    }
    sysctl_copy_out(ctx, &block, old, oldlen)
}

/// sysctlbyname(name, old*, oldlen*, new*, newlen)
fn cb_sysctlbyname(ctx: &mut CpuContext) -> HleResult {
    let (name, old, oldlen, new) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    let Some(name) = ctx.read_cstring(name, process::MAX_SYSCTL_NAME + 1) else { return posix_failure(ctx, EFAULT) };
    if name.len() >= process::MAX_SYSCTL_NAME {
        return posix_failure(ctx, ENAMETOOLONG);
    }
    // One trailing dot is dropped; a top-level node is not a value
    // (EINVAL), as on a PS5 (firmware 13.60).
    let name = process::sysctl_name_without_trailing_dot(&name).to_string();
    if process::top_level_sysctl_node(&name).is_some() {
        return posix_failure(ctx, EINVAL);
    }
    if ABSENT_SYSCTL_NAMES.contains(&name.as_str()) {
        return posix_failure(ctx, ENOENT);
    }
    // Values with no MIB of their own.
    let fixed: Option<Vec<u8>> = match name.as_str() {
        "vm.overcommit" => Some(0i32.to_le_bytes().to_vec()),
        "kern.sdk_version" => Some(0x0100_0000u32.to_le_bytes().to_vec()),
        "machdep.tsc_freq" => Some(crate::time::tsc_frequency().to_le_bytes().to_vec()),
        _ => None,
    };
    if let Some(value) = fixed {
        if new != 0 {
            return posix_failure(ctx, EPERM);
        }
        return match sysctl_copy_out(ctx, &value, old, oldlen) {
            0 => 0,
            e => posix_failure(ctx, e),
        };
    }
    let Some(&(_, mib)) = process::KNOWN_SYSCTL_NAMES.iter().find(|(n, _)| *n == name) else {
        log::warn!("[kernel] sysctlbyname: unknown name '{name}'");
        return posix_failure(ctx, ENOENT);
    };
    match sysctl_mib(ctx, mib, old, oldlen, new) {
        0 => 0,
        e => posix_failure(ctx, e),
    }
}

/// system(command): the console has no shell (`/bin/sh`), so a null
/// command reports none, and a command ends like a child whose shell could
/// not be executed (exit status 127). errno is not changed.
fn cb_system(ctx: &mut CpuContext) -> HleResult {
    const SHELL_NOT_EXECUTED: HleResult = 127 << 8;
    if ctx.rdi() == 0 { 0 } else { SHELL_NOT_EXECUTED }
}

/// waitpid(pid, status*, options): the guest never has child processes.
/// Options FreeBSD does not know are EINVAL; the status is not written.
fn cb_waitpid(ctx: &mut CpuContext) -> HleResult {
    // WNOHANG, WUNTRACED, WCONTINUED, WNOWAIT, WEXITED, WTRAPPED, WLINUXCLONE.
    const KNOWN: u32 = 0x8000_003F;
    let options = ctx.rdx() as u32;
    posix_failure(ctx, if options & !KNOWN != 0 { EINVAL } else { ECHILD })
}

/// execvp(file, argv): replacing the running executable is not
/// implemented; the call fails with ENOSYS and the guest keeps running.
fn cb_execvp(ctx: &mut CpuContext) -> HleResult {
    const ENOSYS: i32 = 78;
    let file = ctx.read_cstring(ctx.rdi(), 1024).unwrap_or_default();
    log::error!("[kernel] execvp('{file}'): executable replacement not implemented");
    posix_failure(ctx, ENOSYS)
}

const CPU_LEVEL_WHICH: i32 = 3;
const CPU_WHICH_TID: i32 = 1;
/// Largest cpuset_t the kernel accepts (CPU_MAXSIZE / 8).
const CPUSET_MAX_SIZE: u64 = 256 / 8;

const CPU_WHICH_PID: i32 = 2;

/// cpuset_setaffinity(level, which, id, size, mask*) -> 0 or -1 with errno:
/// the calling thread (which TID, id -1), another thread by id, or the
/// calling process (which PID, id -1 or its pid: every live thread, and
/// threads created later by no guest thread). Other targets are EINVAL.
fn cb_cpuset_setaffinity(ctx: &mut CpuContext) -> HleResult {
    use crate::pthread::CpusetTarget;
    let (level, which, id, size, set) = (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx() as i64, ctx.rcx(), ctx.r8());
    let me = std::process::id() as i64;
    let target = match (level, which, id) {
        (CPU_LEVEL_WHICH, CPU_WHICH_TID, -1) => CpusetTarget::Thread(0),
        (CPU_LEVEL_WHICH, CPU_WHICH_TID, tid) => CpusetTarget::ThreadId(tid),
        (CPU_LEVEL_WHICH, CPU_WHICH_PID, pid) if pid == -1 || pid == me => CpusetTarget::Process,
        _ => {
            log::warn!("[kernel] cpuset_setaffinity: level {level} which {which} id {id} is not supported");
            return posix_failure(ctx, EINVAL);
        }
    };
    match crate::pthread::set_cpuset_affinity(ctx, target, size, set) {
        0 => 0,
        e => posix_failure(ctx, e as i32),
    }
}

/// cpuset_getaffinity(level, which, id, size, mask*): the affinity of a
/// thread (id -1 = the caller, otherwise a thread id) or of the process
/// (id -1 or its pid): the CPUs of its live threads. Checked in the order
/// size (ERANGE), mask (EFAULT), target (ESRCH, EINVAL), level (EINVAL).
fn cb_cpuset_getaffinity(ctx: &mut CpuContext) -> HleResult {
    let (level, which, id, size, out) = (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx() as i64, ctx.rcx(), ctx.r8());
    if !(8..=CPUSET_MAX_SIZE).contains(&size) {
        return posix_failure(ctx, ERANGE);
    }
    if out == 0 {
        return posix_failure(ctx, EFAULT);
    }
    let affinity = match which {
        CPU_WHICH_TID if id == -1 => crate::pthread::current_affinity(),
        CPU_WHICH_TID => match crate::pthread::affinity_of_thread_id(id) {
            Some(a) => a,
            None => return posix_failure(ctx, ESRCH),
        },
        CPU_WHICH_PID if id == -1 || id == std::process::id() as i64 => crate::pthread::process_affinity(),
        CPU_WHICH_PID => return posix_failure(ctx, ESRCH),
        _ => {
            log::warn!("[kernel] cpuset_getaffinity: which {which} is not supported");
            return posix_failure(ctx, EINVAL);
        }
    };
    if level != CPU_LEVEL_WHICH {
        log::warn!("[kernel] cpuset_getaffinity: level {level} is not supported");
        return posix_failure(ctx, EINVAL);
    }
    let mut bytes = vec![0u8; size as usize];
    bytes[..8].copy_from_slice(&affinity.to_le_bytes());
    if !ctx.write_bytes(out, &bytes) {
        return posix_failure(ctx, EFAULT);
    }
    0
}

// ---------------------------------------------------------------------------
// Core dump calls
// ---------------------------------------------------------------------------
//
// The calls that read the stop state or add data to a core dump work only
// inside the handler sceCoredumpRegisterCoredumpHandler registers. The
// handler is never invoked here (host crashes do not become guest core
// dumps), so every call made with well-formed arguments answers
// SCE_COREDUMP_ERROR_NOT_IN_COREDUMP_HANDLER and touches nothing. Whether
// the console checks the arguments before the calling context is unknown:
// malformed ones (null, empty or wrapping regions, a null text, an
// unaligned or wrongly sized buffer) are logged and refused with EINVAL.

/// SCE_COREDUMP_ERROR_NOT_IN_COREDUMP_HANDLER.
const COREDUMP_NOT_IN_HANDLER: HleResult = 0x8118_0003u32 as i32 as i64;

/// Type set by sceCoredumpSetUserDataType.
static COREDUMP_USER_DATA_TYPE: AtomicU32 = AtomicU32::new(0);

/// A region a core dump call names: not null, not empty, not wrapping.
fn coredump_region_ok(func: &str, mem: u64, size: u64) -> bool {
    let ok = mem != 0 && size != 0 && mem.checked_add(size).is_some();
    if !ok {
        log::error!("[coredump] {func}: null, empty or wrapping region {mem:#x}+{size:#x}");
    }
    ok
}

/// A buffer for the handler's state: 8-byte aligned and of `want` bytes.
fn coredump_buffer_ok(func: &str, buf: u64, size: u64, want: u64) -> bool {
    let ok = buf != 0 && buf % 8 == 0 && size == want && buf.checked_add(size).is_some();
    if !ok {
        log::error!("[coredump] {func}: buffer {buf:#x} of {size:#x} bytes (want {want:#x}, 8-byte aligned)");
    }
    ok
}

fn coredump_outside_handler(ok: bool) -> HleResult {
    if ok { COREDUMP_NOT_IN_HANDLER } else { sce_error(EINVAL) }
}

/// sceCoredumpSetUserDataType(type)
fn cb_coredump_set_user_data_type(ctx: &mut CpuContext) -> HleResult {
    COREDUMP_USER_DATA_TYPE.store(ctx.rdi() as u32, Ordering::Relaxed);
    OK
}

/// sceCoredumpWriteUserData(data, size) -> ssize_t: bytes written inside
/// the handler.
fn cb_coredump_write_user_data(ctx: &mut CpuContext) -> HleResult {
    coredump_outside_handler(coredump_region_ok("sceCoredumpWriteUserData", ctx.rdi(), ctx.rsi()))
}

/// sceCoredumpAttachMemoryRegion(u32 userValue, mem, size) and
/// sceCoredumpAttachUserMemoryFile(u32 userValue, mem, size)
fn cb_coredump_attach_memory(ctx: &mut CpuContext) -> HleResult {
    coredump_outside_handler(coredump_region_ok("sceCoredumpAttachMemoryRegion", ctx.rsi(), ctx.rdx()))
}

/// sceCoredumpAttachUserFile(u32 userValue, const char* path)
fn cb_coredump_attach_user_file(ctx: &mut CpuContext) -> HleResult {
    let ok = ctx.rsi() != 0;
    if !ok {
        log::error!("[coredump] sceCoredumpAttachUserFile: null path");
    }
    coredump_outside_handler(ok)
}

/// sceCoredumpAttachMemoryRegionAsUserFile(u32 userValue, mem, size, const char* name)
fn cb_coredump_attach_memory_as_file(ctx: &mut CpuContext) -> HleResult {
    let ok = coredump_region_ok("sceCoredumpAttachMemoryRegionAsUserFile", ctx.rsi(), ctx.rdx()) && ctx.rcx() != 0;
    coredump_outside_handler(ok)
}

/// sceCoredumpGetStopInfoCpu(void* info, size_t size): a 16-byte block.
fn cb_coredump_get_stop_info_cpu(ctx: &mut CpuContext) -> HleResult {
    coredump_outside_handler(coredump_buffer_ok("sceCoredumpGetStopInfoCpu", ctx.rdi(), ctx.rsi(), 16))
}

/// sceCoredumpGetThreadContextInfo(void* info, size_t size): 18 registers
/// of 64 bits (144 bytes).
fn cb_coredump_get_thread_context_info(ctx: &mut CpuContext) -> HleResult {
    coredump_outside_handler(coredump_buffer_ok("sceCoredumpGetThreadContextInfo", ctx.rdi(), ctx.rsi(), 144))
}

/// SCE_COREDUMP_ERROR_INVALID_ARGUMENT.
const COREDUMP_INVALID_ARGUMENT: HleResult = 0x8118_0000u32 as i32 as i64;

/// sceCoredumpConfigDumpMode(int mode): only mode 1 is accepted.
fn cb_coredump_config_dump_mode(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() as i32 == 1 { OK } else { COREDUMP_INVALID_ARGUMENT }
}

/// sceCoredumpWriteUserString(u32 slot, const char* text, int flags): slot
/// 0..15, a text and a third argument 0..256.
fn cb_coredump_write_user_string(ctx: &mut CpuContext) -> HleResult {
    let (slot, text, third) = (ctx.rdi() as u32, ctx.rsi(), ctx.rdx() as i32);
    if text == 0 || slot > 15 || !(0..=256).contains(&third) {
        return COREDUMP_INVALID_ARGUMENT;
    }
    OK
}

/// sceCoredumpGetStopInfoGpu_Agc(void* info, size_t size): no GPU stop
/// outside a core dump handler.
fn cb_coredump_get_stop_info_gpu_agc(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 { COREDUMP_INVALID_ARGUMENT } else { COREDUMP_NOT_IN_HANDLER }
}

/// sceCoredumpDebugTextOut(const char* str, int len) -> void: the text goes
/// into the report on the console; here it is logged.
fn cb_coredump_debug_text_out(ctx: &mut CpuContext) -> HleResult {
    let (text, len) = (ctx.rdi(), ctx.rsi() as i32);
    if len < 0 || (text == 0 && len != 0) {
        log::error!("[coredump] sceCoredumpDebugTextOut: text {text:#x}, length {len}");
        return 0;
    }
    if len > 0 {
        let bytes = ctx.read_bytes(text, len as usize).unwrap_or_default();
        log::warn!("[coredump] {}", String::from_utf8_lossy(&bytes).trim_end_matches('\n'));
    }
    0
}

// ---------------------------------------------------------------------------
// Thread cancellation
// ---------------------------------------------------------------------------
//
// Cancellation is deferred: a request is recorded, wakes the target if it
// is blocked, and is acted upon at the target's next cancellation point:
// pthread_testcancel, the condition waits, the POSIX semaphore waits,
// pthread_join and the sleeps. The thread then runs its cleanup handlers
// (`__pthread_cleanup_push_imp`) and exits with PTHREAD_CANCELED, like
// pthread_exit. Native guest code cannot be interrupted, so the
// asynchronous type behaves like the deferred one.

const PTHREAD_CANCEL_ENABLE: i32 = 0;
const PTHREAD_CANCEL_DISABLE: i32 = 1;
const PTHREAD_CANCEL_DEFERRED: i32 = 0;
const PTHREAD_CANCEL_ASYNCHRONOUS: i32 = 2;
/// Exit value of a cancelled thread (PTHREAD_CANCELED).
const PTHREAD_CANCELED: u64 = 1;

static CANCEL_PENDING: Mutex<Option<HashSet<u64>>> = Mutex::new(None);

thread_local! {
    static CANCEL_STATE: Cell<i32> = const { Cell::new(PTHREAD_CANCEL_ENABLE) };
    static CANCEL_TYPE: Cell<i32> = const { Cell::new(PTHREAD_CANCEL_DEFERRED) };
    /// The thread is exiting: cancellation is no longer acted on.
    static EXITING: Cell<bool> = const { Cell::new(false) };
    /// Innermost cleanup handler node (guest address, 0 = none).
    static CLEANUP_HEAD: Cell<u64> = const { Cell::new(0) };
}

/// Number of pending requests: lets cancellation points skip the lock.
static CANCEL_PENDING_COUNT: AtomicUsize = AtomicUsize::new(0);

fn with_pending<R>(f: impl FnOnce(&mut HashSet<u64>) -> R) -> R {
    let mut g = CANCEL_PENDING.lock().unwrap_or_else(|p| p.into_inner());
    let set = g.get_or_insert_with(HashSet::new);
    let r = f(set);
    CANCEL_PENDING_COUNT.store(set.len(), Ordering::Release);
    r
}

/// Drop any cancellation request for a released thread handle.
pub(crate) fn forget_thread(handle: u64) {
    with_pending(|p| p.remove(&handle));
}

/// Request cancellation of `handle` and wake it if it is blocked in a
/// wait; returns the errno.
fn cancel(handle: u64) -> i32 {
    if handle == 0 || !crate::pthread::thread_exists(handle) {
        return ESRCH;
    }
    log::info!("[pthread] cancel requested for {handle:#x}");
    with_pending(|p| p.insert(handle));
    crate::signals::wake_thread(handle);
    0
}

/// Whether the calling thread has a cancellation request it must act on at
/// its next cancellation point (pending, enabled, not already exiting).
pub(crate) fn cancel_requested() -> bool {
    if CANCEL_PENDING_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    if CANCEL_STATE.with(|s| s.get()) != PTHREAD_CANCEL_ENABLE || EXITING.with(|e| e.get()) {
        return false;
    }
    let me = crate::pthread::current_handle();
    me != 0 && with_pending(|p| p.contains(&me))
}

/// Cancellation point: when a request is pending and enabled, run the
/// cleanup handlers and end the thread with PTHREAD_CANCELED. Must be
/// called with no HLE locks held.
pub(crate) fn test_cancel() {
    if !cancel_requested() {
        return;
    }
    let me = crate::pthread::current_handle();
    with_pending(|p| p.remove(&me));
    log::info!("[pthread] thread {me:#x} cancelled");
    exit_thread(PTHREAD_CANCELED)
}

/// End the calling guest thread with `value` after running its cleanup
/// handlers (innermost first), as pthread_exit does.
pub(crate) fn exit_thread(value: u64) -> ! {
    EXITING.with(|e| e.set(true));
    run_cleanup_handlers();
    if let Some(vm) = crate::pthread::guest_vm() {
        crate::pthread::abandon_running_once(&CpuContext::new(vm));
    }
    ps5cpu::exec::exit_guest_thread(value)
}

/// Pop and run every cleanup handler of the calling thread.
fn run_cleanup_handlers() {
    let Some(vm) = crate::pthread::guest_vm() else { return };
    let ctx = CpuContext::new(vm);
    while CLEANUP_HEAD.with(|h| h.get()) != 0 {
        cleanup_pop(&ctx, true);
    }
}

/// Sleep for `d` as a cancellation point: a pending request is acted on
/// before the sleep, ends it early and is acted on after it.
pub(crate) fn cancellable_sleep(d: Duration) {
    test_cancel();
    crate::signals::sleep_until_unless(std::time::Instant::now() + d, cancel_requested);
    test_cancel();
}

/// FreeBSD `struct _pthread_cleanup_info` node written by
/// `__pthread_cleanup_push_imp`: { prev @0, routine @8, arg @16, onheap @24 }.
const CLEANUP_PREV: u64 = 0;
const CLEANUP_ROUTINE: u64 = 8;
const CLEANUP_ARG: u64 = 16;
const CLEANUP_ONHEAP: u64 = 24;

/// __pthread_cleanup_push_imp(routine, arg, info*)
fn cb_cleanup_push_imp(ctx: &mut CpuContext) -> HleResult {
    let (routine, arg, info) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    let prev = CLEANUP_HEAD.with(|h| h.get());
    let ok = ctx.write_u64(info + CLEANUP_PREV, prev)
        && ctx.write_u64(info + CLEANUP_ROUTINE, routine)
        && ctx.write_u64(info + CLEANUP_ARG, arg)
        && ctx.write_u32(info + CLEANUP_ONHEAP, 0);
    if !ok {
        log::error!("[pthread] __pthread_cleanup_push_imp: bad node {info:#x}");
        return OK;
    }
    CLEANUP_HEAD.with(|h| h.set(info));
    OK
}

/// Unlink the innermost cleanup handler and run it when `execute`.
fn cleanup_pop(ctx: &CpuContext, execute: bool) {
    let node = CLEANUP_HEAD.with(|h| h.get());
    if node == 0 {
        return;
    }
    let prev = ctx.read_u64(node + CLEANUP_PREV).unwrap_or(0);
    let routine = ctx.read_u64(node + CLEANUP_ROUTINE).unwrap_or(0);
    let arg = ctx.read_u64(node + CLEANUP_ARG).unwrap_or(0);
    let on_heap = ctx.read_u32(node + CLEANUP_ONHEAP).unwrap_or(0) != 0;
    CLEANUP_HEAD.with(|h| h.set(prev));
    if execute && routine != 0 {
        ps5cpu::exec::call_guest_nested(routine, [arg, 0, 0, 0, 0, 0]);
    }
    if on_heap {
        crate::font::gfree(node);
    }
}

/// pthread_cleanup_push(routine, arg): the function form allocates the node
/// itself (onheap), released when it is popped.
fn cb_cleanup_push(ctx: &mut CpuContext) -> HleResult {
    let (routine, arg) = (ctx.rdi(), ctx.rsi());
    let Some(node) = crate::font::galloc(ctx, 32) else {
        log::error!("[pthread] pthread_cleanup_push: no memory for the handler");
        return OK;
    };
    let prev = CLEANUP_HEAD.with(|h| h.get());
    ctx.write_u64(node + CLEANUP_PREV, prev);
    ctx.write_u64(node + CLEANUP_ROUTINE, routine);
    ctx.write_u64(node + CLEANUP_ARG, arg);
    ctx.write_u32(node + CLEANUP_ONHEAP, 1);
    CLEANUP_HEAD.with(|h| h.set(node));
    OK
}

/// __pthread_cleanup_pop_imp(execute)
fn cb_cleanup_pop_imp(ctx: &mut CpuContext) -> HleResult {
    cleanup_pop(ctx, ctx.rdi() as i32 != 0);
    OK
}

fn sce_result(errno: i32) -> HleResult {
    if errno == 0 { OK } else { sce_error(errno) }
}

// ---------------------------------------------------------------------------
// Firmware constants (answers read from the firmware's disassembly)
// ---------------------------------------------------------------------------

/// sceKernelAddResource, sceKernelEventLogInit,
/// sceKernelIsGenuineKratosCex and sceKernelReleaseTraceDirectMemory: 0.
fn cb_fw_zero(_ctx: &mut CpuContext) -> HleResult {
    OK
}

/// sceKernelHasNeoMode, sceKernelIsAuthenticNeo and sceKernelIsGenuineN: 1.
fn cb_fw_one(_ctx: &mut CpuContext) -> HleResult {
    1
}

/// pthread_switch_add_np / pthread_switch_delete_np: ENOTSUP.
fn cb_fw_enotsup_posix(_ctx: &mut CpuContext) -> HleResult {
    45
}

/// The eport calls: EPERM.
fn cb_fw_eperm(_ctx: &mut CpuContext) -> HleResult {
    sce_error(0x01)
}

/// sceKernelQueryTraceMemory: ENOENT.
fn cb_fw_enoent(_ctx: &mut CpuContext) -> HleResult {
    sce_error(0x02)
}

/// sceKernelIccGetCPMode / sceKernelIccSetCPMode: ENOTSUP.
fn cb_fw_enotsup(_ctx: &mut CpuContext) -> HleResult {
    sce_error(0x2D)
}

/// The trace-memory calls that need the feature: ENOSYS.
fn cb_fw_enosys(_ctx: &mut CpuContext) -> HleResult {
    sce_error(0x4E)
}

/// scePthreadCancel(thread)
fn cb_sce_cancel(ctx: &mut CpuContext) -> HleResult {
    sce_result(cancel(ctx.rdi()))
}

/// pthread_cancel(thread)
fn cb_posix_cancel(ctx: &mut CpuContext) -> HleResult {
    cancel(ctx.rdi()) as HleResult
}

/// pthread_testcancel() / scePthreadTestcancel(): exit when a request is
/// pending and cancellation is enabled.
fn cb_testcancel(_ctx: &mut CpuContext) -> HleResult {
    test_cancel();
    OK
}

/// Set the calling thread's cancel state/type; returns the errno.
fn set_cancel_field(ctx: &CpuContext, which: &'static std::thread::LocalKey<Cell<i32>>, value: i32, allowed: [i32; 2], old: u64) -> i32 {
    if !allowed.contains(&value) {
        return EINVAL;
    }
    let prev = which.with(|c| c.replace(value));
    if old != 0 && !ctx.write_u32(old, prev as u32) {
        return EFAULT;
    }
    0
}

fn cancel_state(ctx: &CpuContext) -> i32 {
    set_cancel_field(ctx, &CANCEL_STATE, ctx.rdi() as i32, [PTHREAD_CANCEL_ENABLE, PTHREAD_CANCEL_DISABLE], ctx.rsi())
}

fn cancel_type(ctx: &CpuContext) -> i32 {
    set_cancel_field(ctx, &CANCEL_TYPE, ctx.rdi() as i32, [PTHREAD_CANCEL_DEFERRED, PTHREAD_CANCEL_ASYNCHRONOUS], ctx.rsi())
}

/// pthread_setcancelstate(state, old*)
fn cb_posix_setcancelstate(ctx: &mut CpuContext) -> HleResult {
    cancel_state(ctx) as HleResult
}

/// scePthreadSetcancelstate(state, old*)
fn cb_sce_setcancelstate(ctx: &mut CpuContext) -> HleResult {
    sce_result(cancel_state(ctx))
}

/// pthread_setcanceltype(type, old*)
fn cb_posix_setcanceltype(ctx: &mut CpuContext) -> HleResult {
    cancel_type(ctx) as HleResult
}

/// scePthreadSetcanceltype(type, old*)
fn cb_sce_setcanceltype(ctx: &mut CpuContext) -> HleResult {
    sce_result(cancel_type(ctx))
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

const EXPORTS: &[(&str, &str, &str, HleCallback)] = &[
    ("Q9DhUQxNCwI", "pthread_switch_add_np", LIB_KERNEL, cb_fw_enotsup_posix),
    ("neZ60m-+Baw", "pthread_switch_delete_np", LIB_KERNEL, cb_fw_enotsup_posix),
    ("uqJlNd9XIrA", "sceKernelAddResource", "libkernel_sys", cb_fw_zero),
    ("c7noapHDHWU", "sceKernelEventLogInit", LIB_KERNEL, cb_fw_zero),
    ("H54cSDvjryI", "sceKernelIsGenuineKratosCex", LIB_KERNEL, cb_fw_zero),
    ("Alqa9RE0Y5k", "sceKernelReleaseTraceDirectMemory", LIB_KERNEL, cb_fw_zero),
    ("rNRtm1uioyY", "sceKernelHasNeoMode", LIB_KERNEL, cb_fw_one),
    ("cbn9Di01j0k", "sceKernelIsAuthenticNeo", LIB_KERNEL, cb_fw_one),
    ("u3h77G+tFVc", "sceKernelIsGenuineN", LIB_KERNEL, cb_fw_one),
    ("KF25CDxFn48", "sceKernelCreateEport", LIB_KERNEL, cb_fw_eperm),
    ("HhGAQ+ffliw", "sceKernelDeleteEport", LIB_KERNEL, cb_fw_eperm),
    ("wibWed78GyY", "sceKernelOpenEport", LIB_KERNEL, cb_fw_eperm),
    ("J5RqzZ7nMVc", "sceKernelCloseEport", LIB_KERNEL, cb_fw_eperm),
    ("is-XQhYPRaQ", "sceKernelTriggerEport", LIB_KERNEL, cb_fw_eperm),
    ("+8GCeq43o7c", "sceKernelQueryTraceMemory", LIB_KERNEL, cb_fw_enoent),
    ("1NlbNxWzn7U", "sceKernelIccGetCPMode", LIB_KERNEL, cb_fw_enotsup),
    ("hc0GFWwSrR4", "sceKernelIccSetCPMode", LIB_KERNEL, cb_fw_enotsup),
    ("EgJqC+9VmvM", "sceKernelClearGameDirectMemory", LIB_KERNEL, cb_fw_enosys),
    ("DpDnfM6StzA", "sceKernelMapTraceMemory", LIB_KERNEL, cb_fw_enosys),
    ("eESTogkCMPE", "sceKernelTraceMemoryTypeProtect", LIB_KERNEL, cb_fw_enosys),
    ("tU5e3f9gSiU", "sceKernelIsTrinityMode", LIB_KERNEL, cb_is_trinity_mode),
    ("ca7v6Cxulzs", "sceKernelSetGPO", LIB_KERNEL, cb_set_gpo),
    ("4oXYe9Xmk0Q", "sceKernelGetGPI", LIB_KERNEL, cb_get_gpi),
    ("DLORcroUqbc", "sceKernelGetOpenPsId", LIB_KERNEL, cb_get_open_ps_id),
    ("Xjoosiw+XPI", "sceKernelUuidCreate", LIB_KERNEL, cb_uuid_create),
    ("kvD1F56kfm4", "sceKernelCreateUuidV4", LIB_KERNEL, cb_create_uuid_v4),
    ("g0VTBxfJyu0", "sceKernelGetCurrentCpu", LIB_KERNEL, cb_get_current_cpu),
    ("CBNtXOoef-E", "sched_get_priority_max", LIB_KERNEL, cb_sched_get_priority_max),
    ("m0iS6jNsXds", "sched_get_priority_min", LIB_KERNEL, cb_sched_get_priority_min),
    ("QvsZxomvUHs", "sceKernelNanosleep", LIB_KERNEL, cb_sce_nanosleep),
    ("hHlZQUnlxSM", "getrusage", LIB_KERNEL, cb_getrusage),
    ("Uxqkdta7wEg", "sceCoredumpSetUserDataType", LIB_COREDUMP, cb_coredump_set_user_data_type),
    ("Dbbkj6YHWdo", "sceCoredumpWriteUserData", LIB_COREDUMP, cb_coredump_write_user_data),
    ("+YX0z-GUSNw", "sceCoredumpAttachMemoryRegion", LIB_COREDUMP, cb_coredump_attach_memory),
    ("DoKHmUw1yiQ", "sceCoredumpAttachUserMemoryFile", LIB_COREDUMP, cb_coredump_attach_memory),
    ("5nc2gdLNsok", "sceCoredumpAttachUserFile", LIB_COREDUMP, cb_coredump_attach_user_file),
    ("MEJ7tc7ThwM", "sceCoredumpAttachMemoryRegionAsUserFile", LIB_COREDUMP, cb_coredump_attach_memory_as_file),
    ("kK0DUW1Ukgc", "sceCoredumpGetStopInfoCpu", LIB_COREDUMP, cb_coredump_get_stop_info_cpu),
    ("ShChva57wIM", "sceCoredumpGetThreadContextInfo", LIB_COREDUMP, cb_coredump_get_thread_context_info),
    ("dei8oUx6DbU", "sceCoredumpDebugTextOut", LIB_COREDUMP, cb_coredump_debug_text_out),
    ("gzLt9Qrauk0", "sceCoredumpConfigDumpMode", LIB_COREDUMP, cb_coredump_config_dump_mode),
    ("32KQRUK13kI", "sceCoredumpWriteUserString", LIB_COREDUMP, cb_coredump_write_user_string),
    ("Jrs7UUkGOFo", "sceCoredumpGetStopInfoGpu_Agc", LIB_COREDUMP, cb_coredump_get_stop_info_gpu_agc),
    ("D4yla3vx4tY", "sceKernelError", LIB_KERNEL, cb_kernel_error),
    ("0D4-FVvEikw", "pthread_cancel", LIB_KERNEL, cb_posix_cancel),
    ("qBDmpCyGssE", "scePthreadCancel", LIB_KERNEL, cb_sce_cancel),
    ("nYBrkGDqxh8", "pthread_testcancel", LIB_KERNEL, cb_testcancel),
    ("LapIb799SSE", "scePthreadTestcancel", LIB_KERNEL, cb_testcancel),
    ("lZzFeSxPl08", "pthread_setcancelstate", LIB_KERNEL, cb_posix_setcancelstate),
    ("OAmWq+OHSjw", "scePthreadSetcancelstate", LIB_KERNEL, cb_sce_setcancelstate),
    ("2dEhvvjlq30", "pthread_setcanceltype", LIB_KERNEL, cb_posix_setcanceltype),
    ("sCJd99Phct0", "scePthreadSetcanceltype", LIB_KERNEL, cb_sce_setcanceltype),
    ("1xvtUVx1-Sg", "__pthread_cleanup_push_imp", LIB_KERNEL, cb_cleanup_push_imp),
    ("4ZeZWcMsAV0", "pthread_cleanup_push", LIB_KERNEL, cb_cleanup_push),
    ("RVxb0Ssa5t0", "pthread_cleanup_pop", LIB_KERNEL, cb_cleanup_pop_imp),
    ("iWsFlYMf3Kw", "__pthread_cleanup_pop_imp", LIB_KERNEL, cb_cleanup_pop_imp),
    ("iKJMWrAumPE", "getargc", LIB_KERNEL, cb_getargc),
    ("FJmglmTMdr4", "getargv", LIB_KERNEL, cb_getargv),
    ("La9uyZv4Kvw", "sceKernelGetAvailableCpumask", LIB_KERNEL, cb_get_available_cpumask),
    ("NH6xARDOVv8", "sceKernelGetOperationMode", LIB_KERNEL, cb_get_operation_mode),
    ("uvT2iYBBnkY", "sceKernelSync", LIB_KERNEL, cb_kernel_sync),
    ("kg4x8Prhfxw", "getuid", LIB_KERNEL, cb_getuid),
    ("tvpHe5kBO4E", "geteuid", LIB_KERNEL, cb_geteuid),
    ("AfuS23bX6kg", "getgid", LIB_KERNEL, cb_getgid),
    ("72rYuYoDTWk", "getegid", LIB_KERNEL, cb_getegid),
    ("AxUhC3zNrhk", "issetugid", LIB_KERNEL, cb_issetugid),
    ("HTxb6gmexa0", "seteuid", LIB_KERNEL, cb_seteuid),
    ("4oKwKmeOKjM", "setegid", LIB_KERNEL, cb_setegid),
    ("Wh7HbV7JFqc", "getrlimit", LIB_KERNEL, cb_getrlimit),
    ("DFmMT80xcNI", "sysctl", LIB_KERNEL, cb_sysctl),
    ("MhC53TKmjVA", "sysctlbyname", LIB_KERNEL, cb_sysctlbyname),
    ("Jc6E7N+dHz0", "system", LIB_KERNEL, cb_system),
    ("dTnYqqLIQ6c", "waitpid", LIB_KERNEL, cb_waitpid),
    ("0NJDcQ2B8vE", "execvp", LIB_KERNEL, cb_execvp),
    ("Pdgml4rbxYk", "cpuset_getaffinity", LIB_KERNEL, cb_cpuset_getaffinity),
    ("0Ma-LQjv6So", "cpuset_setaffinity", LIB_KERNEL, cb_cpuset_setaffinity),
];

/// Exports of the libkernel_write_throttling library.
const NID_ONLY_EXPORTS: &[(&str, &str, &str, HleCallback)] = &[(
    "YFC3dBBipj8",
    "sceKernelWriteThrottlingStatus",
    "libkernel_write_throttling",
    cb_write_throttling_stub,
)];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, lib, cb) in EXPORTS.iter().chain(NID_ONLY_EXPORTS).chain(process::PROCESS_EXPORTS).chain(system::SYSTEM_EXPORTS) {
        reg.register_fn(nid, name, lib, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister;
    use ps5vm::{MemProtect, VirtualMemory};

    /// Each test gets its own guest window (tests run in parallel).
    fn vm() -> (VirtualMemory, u64) {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let base = 0x7A10_0000_0000 + NEXT.fetch_add(1, Ordering::Relaxed) * 0x10_0000;
        let vm = VirtualMemory::new();
        vm.map(base, 0x10000, MemProtect::RW).unwrap();
        (vm, base)
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name).replace('_', "-"), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn uuid_has_version_and_variant() {
        let u = make_v4_uuid([0xFF; 16]);
        assert_eq!(u16::from_le_bytes([u[6], u[7]]) >> 12, 4);
        assert_eq!(u[8] & 0xC0, 0x80);
        let u = make_v4_uuid([0; 16]);
        assert_eq!(u16::from_le_bytes([u[6], u[7]]), 0x4000);
    }

    /// pthread_cleanup_push links a node of its own (onheap) that
    /// pthread_cleanup_pop unlinks and releases; nodes of the _imp form
    /// stay the caller's.
    #[test]
    fn cleanup_push_and_pop_functions() {
        std::thread::spawn(|| {
            let (vm, base) = vm();
            let mut ctx = CpuContext::new(&vm);
            assert_eq!(CLEANUP_HEAD.with(|h| h.get()), 0);
            ctx.set(CpuRegister::Rdi, 0);
            ctx.set(CpuRegister::Rsi, 0xA5);
            ctx.set(CpuRegister::Rdx, base);
            assert_eq!(cb_cleanup_push_imp(&mut ctx), OK);
            assert_eq!(cb_cleanup_push(&mut ctx), OK);
            let node = CLEANUP_HEAD.with(|h| h.get());
            assert!(node != 0 && node != base);
            assert_eq!(ctx.read_u64(node + CLEANUP_PREV), Some(base));
            assert_eq!(ctx.read_u64(node + CLEANUP_ARG), Some(0xA5));
            assert_eq!(ctx.read_u32(node + CLEANUP_ONHEAP), Some(1));
            ctx.set(CpuRegister::Rdi, 0);
            assert_eq!(cb_cleanup_pop_imp(&mut ctx), OK);
            assert_eq!(CLEANUP_HEAD.with(|h| h.get()), base);
            assert_eq!(cb_cleanup_pop_imp(&mut ctx), OK);
            assert_eq!(CLEANUP_HEAD.with(|h| h.get()), 0);
            assert_eq!(ctx.read_u64(base + CLEANUP_ARG), Some(0xA5), "the caller's node stays");
        })
        .join()
        .unwrap();
    }

    #[test]
    fn create_uuid_v4_sets_version_and_variant_bytes() {
        let (vm, base) = vm();
        let mut ctx = CpuContext::new(&vm);
        for i in 0..8u64 {
            ctx.set(CpuRegister::Rdi, base + i * 16);
            assert_eq!(cb_create_uuid_v4(&mut ctx), OK);
            let u = ctx.read_bytes(base + i * 16, 16).unwrap();
            assert_eq!((u[6] & 0xF0, u[8] & 0xC0), (0x40, 0x80));
        }
        assert_ne!(ctx.read_bytes(base, 16), ctx.read_bytes(base + 16, 16));
        ctx.set(CpuRegister::Rdi, 0);
        assert_eq!(cb_create_uuid_v4(&mut ctx), sce_error(EINVAL));
    }

    #[test]
    fn uuid_create_writes_random_ids() {
        let (vm, base) = vm();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(CpuRegister::Rdi, base);
        assert_eq!(cb_uuid_create(&mut ctx), OK);
        ctx.set(CpuRegister::Rdi, base + 16);
        assert_eq!(cb_uuid_create(&mut ctx), OK);
        let a = ctx.read_bytes(base, 16).unwrap();
        let b = ctx.read_bytes(base + 16, 16).unwrap();
        assert_ne!(a, b);
        ctx.set(CpuRegister::Rdi, 0);
        assert_eq!(cb_uuid_create(&mut ctx), sce_error(EINVAL));
    }

    #[test]
    fn gpo_bits_read_back_through_gpi() {
        let (vm, _) = vm();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(CpuRegister::Rdi, 0xA5);
        assert_eq!(cb_set_gpo(&mut ctx), OK);
        assert_eq!(cb_get_gpi(&mut ctx), 0xA5);
    }

    #[test]
    fn sched_priority_limits() {
        let (vm, _) = vm();
        let mut ctx = CpuContext::new(&vm);
        // SCHED_FIFO and SCHED_RR: 767 / 256 as a PS5 (firmware 13.60)
        // reports them; SCHED_OTHER (2) and the rest are EINVAL.
        for policy in [1u64, 3] {
            ctx.set(CpuRegister::Rdi, policy);
            assert_eq!(cb_sched_get_priority_max(&mut ctx), 767);
            assert_eq!(cb_sched_get_priority_min(&mut ctx), 256);
        }
        for policy in [0u64, 2, 4, 99] {
            ctx.set(CpuRegister::Rdi, policy);
            assert_eq!(cb_sched_get_priority_max(&mut ctx), -1);
            assert_eq!(cb_sched_get_priority_min(&mut ctx), -1);
        }
    }

    /// Every host CPU index maps into the thread's affinity clipped to the
    /// available CPUs, and every CPU of it is reached.
    #[test]
    fn guest_cpu_stays_within_affinity() {
        for affinity in [AVAILABLE_CPUMASK, 0x1FFB, 0x3, 0x3F, 0x1000, 0x1, 0xFFFF, 0x10000] {
            let expected = if affinity & AVAILABLE_CPUMASK != 0 { affinity & AVAILABLE_CPUMASK } else { AVAILABLE_CPUMASK };
            let mut reached = 0u64;
            for host in 0..256u32 {
                let cpu = guest_cpu_from_host(host, affinity);
                assert!((0..64).contains(&cpu) && expected >> cpu & 1 != 0, "{affinity:#x} {host} -> {cpu}");
                assert_eq!(guest_cpu_from_host(host, affinity), cpu);
                reached |= 1 << cpu;
                if host + 1 == expected.count_ones() {
                    assert_eq!(reached, expected);
                }
            }
            assert_eq!(reached, expected);
        }
    }

    #[test]
    fn credentials_follow_freebsd_privilege_rules() {
        let (vm, _) = vm();
        let mut ctx = CpuContext::new(&vm);
        assert_eq!(cb_getuid(&mut ctx), 0);
        assert_eq!(cb_getgid(&mut ctx), 0);
        // Setting the ids the process already has changes nothing.
        ctx.set(CpuRegister::Rdi, 0);
        assert_eq!(cb_seteuid(&mut ctx), 0);
        assert_eq!(cb_setegid(&mut ctx), 0);
        assert_eq!(cb_geteuid(&mut ctx), 0);
        assert_eq!(cb_getegid(&mut ctx), 0);
        assert_eq!(cb_issetugid(&mut ctx), 0);
        // Without privilege only the real id is allowed.
        assert_eq!(set_effective(REAL_ID, &EFFECTIVE_UID), 0);
    }

    #[test]
    fn getrlimit_reports_host_limits() {
        let (vm, base) = vm();
        let mut ctx = CpuContext::new(&vm);
        let call = |ctx: &mut CpuContext, resource: u64, out: u64| {
            ctx.set(CpuRegister::Rdi, resource);
            ctx.set(CpuRegister::Rsi, out);
            cb_getrlimit(ctx)
        };
        let mut host = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
        assert_eq!(unsafe { libc::getrlimit(libc::RLIMIT_DATA, &mut host) }, 0);
        assert_eq!(call(&mut ctx, 2, base), 0);
        assert_eq!(ctx.read_u64(base).map(|v| v as i64), Some(guest_limit(host.rlim_cur)));
        assert_eq!(ctx.read_u64(base + 8).map(|v| v as i64), Some(guest_limit(host.rlim_max)));
        assert_eq!(guest_limit(libc::RLIM_INFINITY), i64::MAX);
        // FreeBSD-only resources are unlimited.
        assert_eq!(call(&mut ctx, 13, base), 0);
        assert_eq!(ctx.read_u64(base), Some(i64::MAX as u64));
        for bad in [15u64, (-1i64) as u64] {
            assert_eq!(call(&mut ctx, bad, base), -1);
            assert_eq!(call(&mut ctx, bad, 0), -1);
        }
        assert_eq!(call(&mut ctx, 2, 0), -1);
    }

    #[test]
    fn sysctl_reports_the_cpu_count() {
        let (vm, base) = vm();
        let mut ctx = CpuContext::new(&vm);
        let (mib, len, val, name) = (base, base + 0x10, base + 0x20, base + 0x40);
        ctx.write_u32(mib, CTL_HW);
        ctx.write_u32(mib + 4, HW_NCPU);
        let sysctl = |ctx: &mut CpuContext, mib: u64, n: u64, old: u64, oldlen: u64, new: u64| {
            for (r, v) in [(CpuRegister::Rdi, mib), (CpuRegister::Rsi, n), (CpuRegister::Rdx, old), (CpuRegister::Rcx, oldlen), (CpuRegister::R8, new)] {
                ctx.set(r, v);
            }
            cb_sysctl(ctx)
        };
        ctx.write_u64(len, 0);
        assert_eq!(sysctl(&mut ctx, mib, 2, 0, len, 0), 0);
        assert_eq!(ctx.read_u64(len), Some(4));
        ctx.write_u64(len, 4);
        assert_eq!(sysctl(&mut ctx, mib, 2, val, len, 0), 0);
        assert_eq!(ctx.read_u32(val), Some(GUEST_CPU_COUNT as u32));
        // A short buffer gets what fits and ENOMEM.
        ctx.write_u32(val, 0xAAAA_AAAA);
        ctx.write_u64(len, 2);
        assert_eq!(sysctl(&mut ctx, mib, 2, val, len, 0), -1);
        assert_eq!(ctx.read_u64(len), Some(2));
        assert_eq!(ctx.read_u32(val), Some(0xAAAA_0000 | (GUEST_CPU_COUNT as u32 & 0xFFFF)));
        assert_eq!(sysctl(&mut ctx, mib, 2, val, 0, 0), -1);
        ctx.write_u64(len, 4);
        assert_eq!(sysctl(&mut ctx, mib, 2, val, len, val), -1);
        assert_eq!(sysctl(&mut ctx, mib, 1, val, len, 0), -1);
        assert_eq!(sysctl(&mut ctx, mib, 25, val, len, 0), -1);
        assert_eq!(sysctl(&mut ctx, 0, 2, val, len, 0), -1);
        ctx.write_u32(mib + 4, 5);
        assert_eq!(sysctl(&mut ctx, mib, 2, val, len, 0), -1);
        // By name.
        ctx.write_bytes(name, b"hw.ncpu\0");
        ctx.write_u64(len, 16);
        ctx.write_u32(val, 0);
        for (r, v) in [(CpuRegister::Rdi, name), (CpuRegister::Rsi, val), (CpuRegister::Rdx, len), (CpuRegister::Rcx, 0)] {
            ctx.set(r, v);
        }
        assert_eq!(cb_sysctlbyname(&mut ctx), 0);
        assert_eq!(ctx.read_u32(val), Some(GUEST_CPU_COUNT as u32));
        assert_eq!(ctx.read_u64(len), Some(4));
        ctx.write_bytes(name, b"kern.osreldate\0");
        ctx.write_u64(len, 16);
        assert_eq!(cb_sysctlbyname(&mut ctx), 0);
        assert_eq!(ctx.read_u32(val), Some(1_104_001));
        assert_eq!(ctx.read_u64(len), Some(4));
        ctx.write_bytes(name, b"kern.no_such_name\0");
        assert_eq!(cb_sysctlbyname(&mut ctx), -1);
        ctx.set(CpuRegister::Rdi, 0);
        assert_eq!(cb_sysctlbyname(&mut ctx), -1);
    }

    /// The uname names: read-only strings with their NUL; a longer MIB under
    /// one of them is ENOTDIR; sysctlnametomib knows them.
    #[test]
    fn sysctl_answers_the_uname_names() {
        let (vm, base) = vm();
        let ctx = CpuContext::new(&vm);
        let (len, val) = (base + 0x10, base + 0x40);
        ctx.write_u64(len, 0);
        assert_eq!(sysctl_mib(&ctx, &[1, 1], 0, len, 0), 0);
        assert_eq!(ctx.read_u64(len), Some(8));
        ctx.write_u64(len, 64);
        assert_eq!(sysctl_mib(&ctx, &[1, 1], val, len, 0), 0);
        assert_eq!(ctx.read_bytes(val, 8).unwrap(), b"FreeBSD\0");
        ctx.write_u64(len, 64);
        assert_eq!(sysctl_mib(&ctx, &[1, 2], val, len, 0), 0);
        assert_eq!(ctx.read_cstring(val, 64).unwrap(), "11.4-RELEASE");
        ctx.write_u64(len, 64);
        assert_eq!(sysctl_mib(&ctx, &[1, 4], val, len, 0), 0);
        assert_eq!(ctx.read_cstring(val, 64).unwrap(), "FreeBSD 11.4-RELEASE\n");
        ctx.write_u64(len, 64);
        assert_eq!(sysctl_mib(&ctx, &[6, 1], val, len, 0), 0);
        assert_eq!(ctx.read_cstring(val, 64).unwrap(), "amd64");
        ctx.write_u64(len, 64);
        assert_eq!(sysctl_mib(&ctx, &[1, 10], val, len, 0), 0);
        assert_eq!((ctx.read_cstring(val, 64).unwrap().as_str(), ctx.read_u64(len)), ("", Some(1)), "kern.hostname is empty");
        assert_eq!(sysctl_mib(&ctx, &[1, 1], val, len, val), EPERM);
        assert_eq!(sysctl_mib(&ctx, &[1, 1, 0], val, len, 0), 20);
        assert_eq!(process::KNOWN_SYSCTL_NAMES.iter().find(|(n, _)| *n == "hw.machine").map(|e| e.1), Some(&[6u32, 1][..]));
    }

    /// sysctlbyname: a top-level node name is EINVAL with the buffer and
    /// length untouched, one trailing dot is dropped, a name of 1024 bytes
    /// or more is ENAMETOOLONG; an absent top-level MIB node is ENOENT.
    #[test]
    fn sysctlbyname_top_level_names_and_long_names() {
        std::thread::spawn(|| {
            let (vm, base) = vm();
            let vm: &'static VirtualMemory = Box::leak(Box::new(vm));
            let mut ctx = CpuContext::new(vm);
            let errno = crate::kernel::bind_test_errno(&ctx);
            let (name, out, len) = (base + 0x1000, base + 0x100, base + 0x108);
            let call = |ctx: &mut CpuContext| {
                for (r, v) in [(CpuRegister::Rdi, name), (CpuRegister::Rsi, out), (CpuRegister::Rdx, len), (CpuRegister::Rcx, 0)] {
                    ctx.set(r, v);
                }
                cb_sysctlbyname(ctx)
            };
            for node in ["sysctl", "kern", "vm", "vfs", "net", "debug", "hw", "machdep", "user", "p1003_1b", "kern."] {
                ctx.write_bytes(name, format!("{node}\0").as_bytes());
                ctx.write_u32(out, 0x5a5a_5a5a);
                ctx.write_u64(len, 4);
                ctx.write_u32(errno, 13);
                assert_eq!(call(&mut ctx), -1, "{node}");
                assert_eq!(ctx.read_u32(errno), Some(EINVAL as u32), "{node}");
                assert_eq!((ctx.read_u32(out), ctx.read_u64(len)), (Some(0x5a5a_5a5a), Some(4)));
            }
            ctx.write_bytes(name, b"hw.ncpu.\0");
            ctx.write_u64(len, 4);
            assert_eq!(call(&mut ctx), 0);
            assert!(ctx.read_u32(out).unwrap() > 0);
            let mut long = vec![b'a'; 1024];
            long.push(0);
            ctx.write_bytes(name, &long);
            assert_eq!(call(&mut ctx), -1);
            assert_eq!(ctx.read_u32(errno), Some(ENAMETOOLONG as u32));
            long[1023] = 0;
            ctx.write_bytes(name, &long);
            assert_eq!(call(&mut ctx), -1);
            assert_eq!(ctx.read_u32(errno), Some(ENOENT as u32));
            for root in [99u32, 10, 255, u32::MAX, (-5i32) as u32] {
                ctx.write_u32(out, 0x5a5a_5a5a);
                ctx.write_u64(len, 4);
                assert_eq!(sysctl_mib(&ctx, &[root, 1], out, len, 0), ENOENT);
                assert_eq!((ctx.read_u32(out), ctx.read_u64(len)), (Some(0x5a5a_5a5a), Some(4)));
            }
        })
        .join()
        .unwrap();
    }

    /// kern.rng_pseudo fills the whole buffer with fresh random bytes and
    /// keeps the length; a write is EPERM; no output buffer is EINVAL
    /// (reference GuestSystemConfiguration case; the reference throws).
    #[test]
    fn sysctlbyname_kern_rng_pseudo() {
        std::thread::spawn(|| {
            let (vm, base) = vm();
            let vm: &'static VirtualMemory = Box::leak(Box::new(vm));
            let mut ctx = CpuContext::new(vm);
            let errno = crate::kernel::bind_test_errno(&ctx);
            let (name, first, second, len) = (base, base + 0x100, base + 0x200, base + 0x300);
            ctx.write_bytes(name, b"kern.rng_pseudo\0");
            let call = |ctx: &mut CpuContext, old: u64, oldlen: u64, new: u64| {
                for (r, v) in [(CpuRegister::Rdi, name), (CpuRegister::Rsi, old), (CpuRegister::Rdx, oldlen), (CpuRegister::Rcx, new)] {
                    ctx.set(r, v);
                }
                cb_sysctlbyname(ctx)
            };
            ctx.write_u32(errno, 13);
            // A buffer of 64 bytes or more gets one 64-byte block.
            for (out, room) in [(first, 64), (second, 200)] {
                ctx.write_bytes(out, &[0; 72]);
                ctx.write_u64(len, room);
                assert_eq!(call(&mut ctx, out, len, 0), 0);
                assert_eq!(ctx.read_u64(len), Some(64));
                assert_eq!(ctx.read_bytes(out + 64, 8).unwrap(), vec![0; 8], "past the block");
            }
            assert_ne!(ctx.read_bytes(first, 64), ctx.read_bytes(second, 64));
            assert_eq!(ctx.read_u32(errno), Some(13));
            // A size query reports 64.
            ctx.write_u64(len, 0);
            assert_eq!(call(&mut ctx, 0, len, 0), 0);
            assert_eq!(ctx.read_u64(len), Some(64));
            // A short buffer gets what fits and ENOMEM; a null length gets none.
            ctx.write_bytes(first, &[0; 16]);
            ctx.write_u64(len, 8);
            assert_eq!(call(&mut ctx, first, len, 0), -1);
            assert_eq!(ctx.read_u32(errno), Some(12));
            assert_eq!(ctx.read_u64(len), Some(8));
            assert_eq!(ctx.read_bytes(first + 8, 8).unwrap(), vec![0; 8]);
            assert_eq!(call(&mut ctx, first, 0, 0), -1);
            assert_eq!(ctx.read_u32(errno), Some(12));
            ctx.write_u32(errno, 13);
            assert_eq!(call(&mut ctx, 0, 0, 0), 0);
            assert_eq!(ctx.read_u32(errno), Some(13));
            ctx.write_u64(len, 64);
            assert_eq!(call(&mut ctx, first, len, first), -1);
            assert_eq!(ctx.read_u32(errno), Some(1));
            // The same block through its MIB.
            ctx.write_u64(len, 64);
            assert_eq!(sysctl_mib(&ctx, &[1, 50], first, len, 0), 0);
            assert_eq!(ctx.read_u64(len), Some(64));
        })
        .join()
        .unwrap();
    }

    /// The names and MIBs the console answers with ENOENT leave the buffer
    /// and the length alone; sysctlnametomib still gives their MIBs
    /// (reference GuestSystemConfiguration cases measured on a PS5).
    #[test]
    fn sysctl_measured_absent_names() {
        std::thread::spawn(|| {
            let (vm, base) = vm();
            let vm: &'static VirtualMemory = Box::leak(Box::new(vm));
            let mut ctx = CpuContext::new(vm);
            let errno = crate::kernel::bind_test_errno(&ctx);
            let (name, val, len) = (base, base + 0x100, base + 0x200);
            for absent in ABSENT_SYSCTL_NAMES {
                ctx.write_bytes(name, format!("{absent}\0").as_bytes());
                ctx.write_u64(len, 8);
                ctx.write_u64(val, 0x5A5A);
                for (r, v) in [(CpuRegister::Rdi, name), (CpuRegister::Rsi, val), (CpuRegister::Rdx, len), (CpuRegister::Rcx, 0)] {
                    ctx.set(r, v);
                }
                assert_eq!(cb_sysctlbyname(&mut ctx), -1, "{absent}");
                assert_eq!(ctx.read_u32(errno), Some(ENOENT as u32));
                assert_eq!((ctx.read_u64(val), ctx.read_u64(len)), (Some(0x5A5A), Some(8)));
            }
            for mib in [&[1u32, 6][..], &[1, 7], &[1, 12], &[1, 27], &[1, 32], &[6, 4], &[6, 11], &[1, 2147483151], &[1, 2147482816, 2147482814]] {
                assert_eq!(sysctl_mib(&ctx, mib, val, len, 0), ENOENT, "{mib:?}");
            }
            assert_eq!(sysctl_mib(&ctx, &[1, 24, 0], val, len, 0), 20);
            assert_eq!(process::KNOWN_SYSCTL_NAMES.iter().find(|(n, _)| *n == "kern.maxproc").map(|e| e.1), Some(&[1u32, 6][..]));
            assert_eq!(process::KNOWN_SYSCTL_NAMES.iter().find(|(n, _)| *n == "kern.rng_pseudo").map(|e| e.1), Some(&[1u32, 50][..]));
        })
        .join()
        .unwrap();
    }

    /// kern.proc.vmmap lists the tracked guest ranges as 144-byte records
    /// for this process only.
    #[test]
    fn sysctl_proc_vmmap() {
        let (vm, base) = vm();
        let ctx = CpuContext::new(&vm);
        ps5vm::ranges::track(0x7_7000_0000, 0x4000, ps5vm::MemProtect::RX);
        let (len, val) = (base, base + 0x100);
        let me = std::process::id();
        ctx.write_u64(len, 0);
        assert_eq!(sysctl_mib(&ctx, &[1, 14, 32, me], 0, len, 0), 0);
        let size = ctx.read_u64(len).unwrap();
        assert!(size >= VM_ENTRY_SIZE as u64 && size % VM_ENTRY_SIZE as u64 == 0);
        let records = vm_map_entries();
        let rec = records.chunks(VM_ENTRY_SIZE).find(|r| u64::from_le_bytes(r[8..16].try_into().unwrap()) == 0x7_7000_0000).unwrap();
        assert_eq!(i32::from_le_bytes(rec[0..4].try_into().unwrap()), 144);
        assert_eq!(u64::from_le_bytes(rec[16..24].try_into().unwrap()), 0x7_7000_4000);
        assert_eq!(i32::from_le_bytes(rec[56..60].try_into().unwrap()), 5);
        ctx.write_u64(len, VM_ENTRY_SIZE as u64 - 1);
        assert_eq!(sysctl_mib(&ctx, &[1, 14, 32, me], val, len, 0), ENOMEM);
        assert_eq!(sysctl_mib(&ctx, &[1, 14, 32, me + 1], val, len, 0), ESRCH);
        assert_eq!(sysctl_mib(&ctx, &[1, 14, 32, u32::MAX], val, len, 0), ESRCH);
        assert_eq!(sysctl_mib(&ctx, &[1, 14, 32], val, len, 0), EINVAL);
        assert_eq!(sysctl_mib(&ctx, &[1, 14, 32, me], val, len, val), EPERM);
        ps5vm::ranges::untrack(0x7_7000_0000, 0x4000);
    }

    #[test]
    fn no_shell_and_no_children() {
        let (vm, base) = vm();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(CpuRegister::Rdi, 0);
        assert_eq!(cb_system(&mut ctx), 0);
        ctx.write_bytes(base, b"exit 0\0");
        ctx.set(CpuRegister::Rdi, base);
        let status = cb_system(&mut ctx);
        assert_eq!(status & 0x7F, 0);
        assert_eq!((status >> 8) & 0xFF, 127);
        let me = std::process::id() as i64;
        for pid in [-1i64, 0, 1, me, me + 1, -me] {
            for options in [0u64, 1, 2, 3, 4, 8, 16, 32, 0x2c, 0x3f, 0x8000_0000, 0x8000_0001, 0x8000_003f] {
                ctx.write_u32(base + 0x100, 0x5a5a_5a5a);
                ctx.set(CpuRegister::Rdi, pid as u64);
                ctx.set(CpuRegister::Rsi, base + 0x100);
                ctx.set(CpuRegister::Rdx, options);
                assert_eq!(cb_waitpid(&mut ctx), -1);
                assert_eq!(ctx.read_u32(base + 0x100), Some(0x5a5a_5a5a));
            }
        }
    }

    /// execvp never replaces the process: it fails (ENOSYS) for any file,
    /// a null one included.
    #[test]
    fn execvp_is_not_implemented() {
        let (vm, base) = vm();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(base, b"sh\0");
        ctx.write_bytes(base + 0x10, b"/bin/sh\0");
        for (file, argv) in [(base, base + 0x100), (base + 0x10, base + 0x100), (0, 0)] {
            ctx.set(CpuRegister::Rdi, file);
            ctx.set(CpuRegister::Rsi, argv);
            assert_eq!(cb_execvp(&mut ctx), -1);
        }
    }

    #[test]
    fn cpuset_getaffinity_of_the_caller() {
        let _g = crate::pthread::AFFINITY_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let (vm, base) = vm();
        let mut ctx = CpuContext::new(&vm);
        let call = |ctx: &mut CpuContext, level: u64, which: u64, id: i64, size: u64, out: u64| {
            for (r, v) in [(CpuRegister::Rdi, level), (CpuRegister::Rsi, which), (CpuRegister::Rdx, id as u64), (CpuRegister::Rcx, size), (CpuRegister::R8, out)] {
                ctx.set(r, v);
            }
            cb_cpuset_getaffinity(ctx)
        };
        let expected = crate::pthread::current_affinity();
        ctx.write_bytes(base, &[0xFF; 32]);
        assert_eq!(call(&mut ctx, 3, 1, -1, 8, base), 0);
        assert_eq!(ctx.read_u64(base), Some(expected));
        assert_eq!(ctx.read_u64(base + 8), Some(u64::MAX));
        assert_eq!(call(&mut ctx, 3, 1, -1, 32, base), 0);
        assert_eq!(ctx.read_bytes(base + 8, 24), Some(vec![0; 24]));
        assert_eq!(call(&mut ctx, 3, 1, -1, 4, base), -1);
        assert_eq!(call(&mut ctx, 3, 1, -1, 33, base), -1);
        assert_eq!(call(&mut ctx, 3, 1, -1, 8, 0), -1);
        assert_eq!(call(&mut ctx, 3, 1, 0x3FFF_FFF0, 8, base), -1);
        // The process: the CPUs of its live threads; another pid is ESRCH.
        let pid = std::process::id() as i64;
        for id in [-1, pid] {
            ctx.write_bytes(base, &[0xFF; 32]);
            assert_eq!(call(&mut ctx, 3, 2, id, 32, base), 0);
            assert_eq!(ctx.read_u64(base), Some(crate::pthread::process_affinity()));
            assert_eq!(ctx.read_bytes(base + 8, 24), Some(vec![0; 24]));
        }
        assert_eq!(call(&mut ctx, 3, 2, pid + 1, 8, base), -1);
        assert_eq!(call(&mut ctx, 3, 7, -1, 8, base), -1);
        assert_eq!(call(&mut ctx, 1, 1, -1, 8, base), -1, "only CPU_LEVEL_WHICH");
    }

    /// cpuset_setaffinity on the calling thread drops unavailable CPUs,
    /// refuses sizes outside 8..=32 (ERANGE), CPUs past 64 (EINVAL), a NULL
    /// set (EFAULT) and a set with no available CPU (EDEADLK, unchanged);
    /// the process target reaches every live thread (reference
    /// GuestAvailableCpumask cases); pthread_setaffinity_np is the same on
    /// one thread and returns the error number.
    #[test]
    fn cpuset_setaffinity_of_the_caller_and_the_process() {
        let _g = crate::pthread::AFFINITY_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        std::thread::spawn(|| {
            let (vm, base) = vm();
            let mut ctx = CpuContext::new(&vm);
            let errno = crate::kernel::bind_test_errno(&ctx);
            let (bystander, _) = std::thread::spawn(|| crate::pthread::fake_thread("bystander", AVAILABLE_CPUMASK)).join().unwrap();
            let (me, _) = crate::pthread::fake_thread("setter", AVAILABLE_CPUMASK);
            let call = |ctx: &mut CpuContext, which: u64, id: i64, size: u64, set: u64| {
                for (r, v) in [(CpuRegister::Rdi, 3), (CpuRegister::Rsi, which), (CpuRegister::Rdx, id as u64), (CpuRegister::Rcx, size), (CpuRegister::R8, set)] {
                    ctx.set(r, v);
                }
                cb_cpuset_setaffinity(ctx)
            };
            let affinity_of = |ctx: &mut CpuContext, h: u64| {
                ctx.set(CpuRegister::Rdi, h);
                ctx.set(CpuRegister::Rsi, base + 0x100);
                assert_eq!(crate::pthread::test_scegetaffinity(ctx), 0);
                ctx.read_u64(base + 0x100).unwrap()
            };
            let last = 1u64 << 12;
            let set = base;
            ctx.write_bytes(set, &[0; 32]);
            ctx.write_u64(set, last | 1 << 40);
            assert_eq!(call(&mut ctx, 1, -1, 8, set), 0);
            assert_eq!(affinity_of(&mut ctx, me), last, "unavailable CPUs dropped");
            ctx.write_u64(set + 8, 1);
            assert_eq!(call(&mut ctx, 1, -1, 16, set), -1);
            assert_eq!(ctx.read_u32(errno), Some(22));
            ctx.write_u64(set + 8, 0);
            assert_eq!(call(&mut ctx, 1, -1, 32, set), 0);
            for size in [4u64, 33] {
                assert_eq!(call(&mut ctx, 1, -1, size, set), -1);
                assert_eq!(ctx.read_u32(errno), Some(34));
            }
            assert_eq!(call(&mut ctx, 1, -1, 8, 0), -1);
            assert_eq!(ctx.read_u32(errno), Some(14));
            ctx.write_u64(set + 0x40, 1 << 40);
            assert_eq!(call(&mut ctx, 1, -1, 8, set + 0x40), -1);
            assert_eq!(ctx.read_u32(errno), Some(11));
            assert_eq!(affinity_of(&mut ctx, me), last, "a refused set changes nothing");
            assert_eq!(affinity_of(&mut ctx, bystander), AVAILABLE_CPUMASK);
            // The process: every live thread.
            let pid = std::process::id() as i64;
            ctx.write_u64(set + 0x80, 0x3);
            assert_eq!(call(&mut ctx, 2, pid, 8, set + 0x80), 0);
            assert_eq!((affinity_of(&mut ctx, me), affinity_of(&mut ctx, bystander)), (0x3, 0x3));
            ctx.write_u64(set + 0x80, AVAILABLE_CPUMASK);
            assert_eq!(call(&mut ctx, 2, -1, 8, set + 0x80), 0);
            assert_eq!(affinity_of(&mut ctx, bystander), AVAILABLE_CPUMASK);
            assert_eq!(call(&mut ctx, 2, pid + 1, 8, set + 0x80), -1, "another process");
            assert_eq!(call(&mut ctx, 1, 0x3FFF_FFF0, 8, set + 0x80), -1, "unknown thread id");
            assert_eq!(ctx.read_u32(errno), Some(3));
            // pthread_setaffinity_np: the same rules, error numbers returned.
            ctx.write_u64(set, last);
            assert_eq!(crate::pthread::test_setaffinity_np(&mut ctx, bystander, 32, set), 0);
            assert_eq!(affinity_of(&mut ctx, bystander), last);
            assert_eq!(crate::pthread::test_setaffinity_np(&mut ctx, bystander, 4, set), 34);
            assert_eq!(crate::pthread::test_setaffinity_np(&mut ctx, 0, 32, set), 3);
            // Back to the default process cpuset.
            ctx.write_u64(set + 0x80, 0x7F);
            assert_eq!(call(&mut ctx, 2, -1, 8, set + 0x80), 0);
            crate::pthread::forget_fake_thread(bystander);
            crate::pthread::forget_fake_thread(me);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn cleanup_handlers_are_a_stack_of_guest_nodes() {
        let (vm, base) = vm();
        let mut ctx = CpuContext::new(&vm);
        let (first, second) = (base, base + 0x40);
        let push = |ctx: &mut CpuContext, routine: u64, arg: u64, node: u64| {
            ctx.set(CpuRegister::Rdi, routine);
            ctx.set(CpuRegister::Rsi, arg);
            ctx.set(CpuRegister::Rdx, node);
            cb_cleanup_push_imp(ctx)
        };
        assert_eq!(push(&mut ctx, 0x1111, 1, first), OK);
        assert_eq!(push(&mut ctx, 0x2222, 2, second), OK);
        assert_eq!(ctx.read_u64(second + CLEANUP_PREV), Some(first));
        assert_eq!(ctx.read_u64(second + CLEANUP_ROUTINE), Some(0x2222));
        assert_eq!(ctx.read_u64(second + CLEANUP_ARG), Some(2));
        assert_eq!(ctx.read_u32(second + CLEANUP_ONHEAP), Some(0));
        assert_eq!(CLEANUP_HEAD.with(|h| h.get()), second);
        // Popped without running (a run needs guest code).
        ctx.set(CpuRegister::Rdi, 0);
        assert_eq!(cb_cleanup_pop_imp(&mut ctx), OK);
        assert_eq!(CLEANUP_HEAD.with(|h| h.get()), first);
        assert_eq!(cb_cleanup_pop_imp(&mut ctx), OK);
        assert_eq!(CLEANUP_HEAD.with(|h| h.get()), 0);
        assert_eq!(cb_cleanup_pop_imp(&mut ctx), OK);
    }

    /// Cancellation is acted on only for a live thread with a pending,
    /// enabled request.
    #[test]
    fn cancel_requests_need_a_live_enabled_thread() {
        let (vm, _) = vm();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(CpuRegister::Rdi, 0);
        assert_eq!(cb_posix_cancel(&mut ctx), ESRCH as i64);
        assert_eq!(cb_sce_cancel(&mut ctx), sce_error(ESRCH));
        // A thread that is not a guest thread never has a request.
        assert!(!cancel_requested());
        let h = 0x6_6600_0000u64;
        with_pending(|p| p.insert(h));
        assert!(!cancel_requested());
        forget_thread(h);
        // sleep and testcancel return normally without a request.
        cancellable_sleep(Duration::from_millis(1));
        assert_eq!(cb_testcancel(&mut ctx), OK);
    }

    #[test]
    fn nanosleep_validates_request() {
        let (vm, base) = vm();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(CpuRegister::Rdi, 0);
        assert_eq!(cb_sce_nanosleep(&mut ctx), sce_error(EFAULT));
        ctx.write_u64(base, 0);
        ctx.write_u64(base + 8, 1_000_000_000);
        ctx.set(CpuRegister::Rdi, base);
        assert_eq!(cb_sce_nanosleep(&mut ctx), sce_error(EINVAL));
        ctx.write_u64(base + 8, 1000);
        ctx.write_u64(base + 16, 7);
        ctx.write_u64(base + 24, 7);
        ctx.set(CpuRegister::Rsi, base + 16);
        assert_eq!(cb_sce_nanosleep(&mut ctx), OK);
        assert_eq!(ctx.read_u64(base + 16), Some(7), "a completed sleep leaves rem alone");
        assert_eq!(ctx.read_u64(base + 24), Some(7));
    }

    #[test]
    fn current_cpu_is_in_guest_range() {
        let (vm, _) = vm();
        let mut ctx = CpuContext::new(&vm);
        let cpu = cb_get_current_cpu(&mut ctx);
        assert!((0..GUEST_CPU_COUNT as i64).contains(&cpu));
    }

    #[test]
    fn getrusage_fills_times() {
        let (vm, base) = vm();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(CpuRegister::Rdi, RUSAGE_SELF as u64);
        ctx.set(CpuRegister::Rsi, base);
        assert_eq!(cb_getrusage(&mut ctx), 0);
        // usec fields are normalised.
        assert!(ctx.read_u64(base + 8).unwrap() < 1_000_000);
        ctx.set(CpuRegister::Rdi, 7);
        assert_eq!(cb_getrusage(&mut ctx), -1);
    }

    /// Outside a core dump handler (the only place calls are made here) the
    /// core dump calls answer NOT_IN_COREDUMP_HANDLER for well-formed
    /// arguments and EINVAL for malformed ones.
    #[test]
    fn coredump_calls_outside_the_handler() {
        let (vm, base) = vm();
        let mut ctx = CpuContext::new(&vm);
        let call = |ctx: &mut CpuContext, cb: fn(&mut CpuContext) -> HleResult, args: [u64; 4]| {
            for (r, v) in [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx].into_iter().zip(args) {
                ctx.set(r, v);
            }
            cb(ctx)
        };
        let not_in = COREDUMP_NOT_IN_HANDLER;
        assert_eq!(call(&mut ctx, cb_coredump_write_user_data, [base, 5, 0, 0]), not_in);
        assert_eq!(call(&mut ctx, cb_coredump_write_user_data, [0, 5, 0, 0]), sce_error(EINVAL));
        assert_eq!(call(&mut ctx, cb_coredump_attach_memory, [7, base, 0x100, 0]), not_in);
        assert_eq!(call(&mut ctx, cb_coredump_attach_memory, [7, base, 0, 0]), sce_error(EINVAL));
        assert_eq!(call(&mut ctx, cb_coredump_attach_memory, [7, u64::MAX, 2, 0]), sce_error(EINVAL));
        assert_eq!(call(&mut ctx, cb_coredump_attach_user_file, [1, base, 0, 0]), not_in);
        assert_eq!(call(&mut ctx, cb_coredump_attach_user_file, [1, 0, 0, 0]), sce_error(EINVAL));
        assert_eq!(call(&mut ctx, cb_coredump_attach_memory_as_file, [999, base, 8, base]), not_in);
        assert_eq!(call(&mut ctx, cb_coredump_attach_memory_as_file, [999, base, 8, 0]), sce_error(EINVAL));
        assert_eq!(call(&mut ctx, cb_coredump_get_stop_info_cpu, [base, 16, 0, 0]), not_in);
        assert_eq!(call(&mut ctx, cb_coredump_get_stop_info_cpu, [base + 4, 16, 0, 0]), sce_error(EINVAL));
        assert_eq!(call(&mut ctx, cb_coredump_get_stop_info_cpu, [base, 8, 0, 0]), sce_error(EINVAL));
        assert_eq!(call(&mut ctx, cb_coredump_get_thread_context_info, [base, 144, 0, 0]), not_in);
        assert_eq!(call(&mut ctx, cb_coredump_get_thread_context_info, [0, 144, 0, 0]), sce_error(EINVAL));
        ctx.write_bytes(base, b"crash\n");
        assert_eq!(call(&mut ctx, cb_coredump_debug_text_out, [base, 6, 0, 0]), 0);
        // The block is left untouched.
        ctx.write_u64(base + 0x100, 0x55);
        call(&mut ctx, cb_coredump_get_stop_info_cpu, [base + 0x100, 16, 0, 0]);
        assert_eq!(ctx.read_u64(base + 0x100), Some(0x55));
    }

    #[test]
    fn coredump_mode_user_string_and_gpu_stop_info() {
        let (vm, base) = vm();
        let mut ctx = CpuContext::new(&vm);
        let call = |ctx: &mut CpuContext, cb: fn(&mut CpuContext) -> HleResult, args: [u64; 3]| {
            for (r, v) in [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx].into_iter().zip(args) {
                ctx.set(r, v);
            }
            cb(ctx)
        };
        let inval = COREDUMP_INVALID_ARGUMENT;
        assert_eq!(call(&mut ctx, cb_coredump_config_dump_mode, [1, 0, 0]), OK);
        assert_eq!(call(&mut ctx, cb_coredump_config_dump_mode, [0, 0, 0]), inval);
        ctx.write_bytes(base, b"x\0");
        assert_eq!(call(&mut ctx, cb_coredump_write_user_string, [0, base, 0]), OK);
        assert_eq!(call(&mut ctx, cb_coredump_write_user_string, [15, base, 256]), OK);
        assert_eq!(call(&mut ctx, cb_coredump_write_user_string, [16, base, 0]), inval);
        assert_eq!(call(&mut ctx, cb_coredump_write_user_string, [0, 0, 0]), inval);
        assert_eq!(call(&mut ctx, cb_coredump_write_user_string, [0, base, 257]), inval);
        assert_eq!(call(&mut ctx, cb_coredump_write_user_string, [0, base, u32::MAX as u64]), inval);
        assert_eq!(call(&mut ctx, cb_coredump_get_stop_info_gpu_agc, [0, 0, 0]), inval);
        assert_eq!(call(&mut ctx, cb_coredump_get_stop_info_gpu_agc, [base, 16, 0]), COREDUMP_NOT_IN_HANDLER);
    }

    #[test]
    fn write_throttling_status_and_kernel_error() {
        let (vm, base) = vm();
        let mut ctx = CpuContext::new(&vm);
        for (i, v) in [1u64, 2, 3, 4].into_iter().enumerate() {
            ctx.write_u64(base + i as u64 * 8, v);
        }
        ctx.set(CpuRegister::Rdi, base);
        assert_eq!(cb_write_throttling_stub(&mut ctx), 0);
        let got: Vec<u64> = (0..4).map(|i| ctx.read_u64(base + i * 8).unwrap()).collect();
        assert_eq!(got, [0xFFFF_FFFF, 0, 0, 0]);
        ctx.set(CpuRegister::Rdi, 0);
        assert_eq!(cb_write_throttling_stub(&mut ctx), 0x8002_000E_u32 as i32 as i64);
        // FreeBSD 11's last errno values (ENOTRECOVERABLE 95, EOWNERDEAD 96)
        // and the wrap of values outside the errno range.
        let cases = [
            (0i32, 0i64),
            (2, 0x8002_0002u32 as i32 as i64),
            (22, 0x8002_0016u32 as i32 as i64),
            (95, 0x8002_005Fu32 as i32 as i64),
            (96, 0x8002_0060u32 as i32 as i64),
            (-1, 0x8001_FFFFu32 as i32 as i64),
            (i32::MAX, 0x1_FFFF),
            (i32::MIN, 0x2_0000),
        ];
        for (e, want) in cases {
            ctx.set(CpuRegister::Rdi, e as u32 as u64);
            assert_eq!(cb_kernel_error(&mut ctx), want);
        }
    }

    #[test]
    fn cancel_unknown_thread_is_esrch() {
        let (vm, _) = vm();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(CpuRegister::Rdi, 0xDEAD_0000);
        assert_eq!(cb_posix_cancel(&mut ctx), ESRCH as i64);
        assert_eq!(cb_sce_cancel(&mut ctx), sce_error(ESRCH));
    }

    #[test]
    fn cancel_state_round_trip() {
        let (vm, base) = vm();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(CpuRegister::Rdi, PTHREAD_CANCEL_DISABLE as u64);
        ctx.set(CpuRegister::Rsi, base);
        assert_eq!(cb_posix_setcancelstate(&mut ctx), 0);
        assert_eq!(ctx.read_u32(base), Some(PTHREAD_CANCEL_ENABLE as u32));
        ctx.set(CpuRegister::Rdi, PTHREAD_CANCEL_ENABLE as u64);
        assert_eq!(cb_sce_setcancelstate(&mut ctx), OK);
        assert_eq!(ctx.read_u32(base), Some(PTHREAD_CANCEL_DISABLE as u32));
        ctx.set(CpuRegister::Rdi, 5);
        assert_eq!(cb_posix_setcanceltype(&mut ctx), EINVAL as i64);
        // No pending request: testcancel returns normally.
        assert_eq!(cb_testcancel(&mut ctx), OK);
    }
}
