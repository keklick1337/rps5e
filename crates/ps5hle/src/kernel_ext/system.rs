// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! System information and notification calls: CPU frequency and mode, SoC
//! id, temperatures (host sensors), power consumption, debug text and
//! notification output, the libkernel text location, internal memory, the
//! core dump mode and the AIO cancel/parameter calls.

use super::*;
use std::path::Path;

const EIO: i32 = 5;

/// sceKernelGetCpuFrequency(): the TSC frequency.
fn cb_get_cpu_frequency(_ctx: &mut CpuContext) -> HleResult {
    crate::time::tsc_frequency() as HleResult
}

/// sceKernelGetCpumode() / sceKernelGetMainSocId(): 0.
fn cb_zero(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// Host temperatures in degrees Celsius: [CPU, board, storage]; -1 when
/// no plausible sensor reports one. Read from `root` (/sys/class).
pub(super) fn host_sensors(root: &Path) -> [i32; 3] {
    fn celsius(file: &Path) -> Option<i32> {
        let milli: i64 = std::fs::read_to_string(file).ok()?.trim().parse().ok()?;
        (-273_150..=2_000_000).contains(&milli).then_some((milli / 1000) as i32)
    }
    fn name(file: &Path) -> String {
        std::fs::read_to_string(file).unwrap_or_default().lines().next().unwrap_or("").to_lowercase()
    }
    fn sorted(dir: &Path) -> Vec<std::path::PathBuf> {
        let mut v: Vec<_> = std::fs::read_dir(dir).map(|r| r.filter_map(|e| e.ok().map(|e| e.path())).collect()).unwrap_or_default();
        v.sort();
        v
    }
    let power_or_network = |n: &str| {
        n.starts_with("bat")
            || n == "ac"
            || n.starts_with("acad")
            || n.starts_with("ucsi")
            || ["iwl", "mt79", "ath", "rtw", "wlan", "wifi", "wireless"].iter().any(|p| n.contains(p))
    };
    let any = |n: &str, parts: &[&str]| parts.iter().any(|p| n.contains(p));
    let mut c = [-1i32; 3];
    let (mut gpu, mut board) = (-1, -1);
    for path in sorted(&root.join("hwmon")) {
        let Some(v) = celsius(&path.join("temp1_input")) else { continue };
        let n = name(&path.join("name"));
        if power_or_network(&n) {
            continue;
        }
        if any(&n, &["coretemp", "k10temp", "zenpower", "cpu"]) {
            if c[0] < 0 {
                c[0] = v;
            }
        } else if any(&n, &["amdgpu", "i915", "radeon", "nvidia", "gpu"]) || n == "xe" {
            if gpu < 0 {
                gpu = v;
            }
        } else if n.contains("nvme") {
            if c[2] < 0 {
                c[2] = v;
            }
        } else if board < 0 {
            board = v;
        }
    }
    for path in sorted(&root.join("thermal")) {
        let Some(v) = celsius(&path.join("temp")) else { continue };
        let t = name(&path.join("type"));
        if any(&t, &["x86_pkg_temp", "cpu-thermal"]) {
            if c[0] < 0 {
                c[0] = v;
            }
        } else if t.contains("gpu") {
            if gpu < 0 {
                gpu = v;
            }
        } else if board < 0 {
            board = v;
        }
    }
    if c[0] < 0 {
        c[0] = gpu;
    }
    if c[1] < 0 {
        c[1] = board;
    }
    for v in c.iter_mut() {
        if !(0..=130).contains(v) {
            *v = -1;
        }
    }
    c
}

/// A failed temperature query: the error is also left in errno, as the
/// console does.
fn temperature_failure(ctx: &CpuContext, errno: i32) -> HleResult {
    crate::kernel::set_errno(ctx, errno);
    sce_error(errno)
}

/// sceKernelGetCpuTemperature(int* celsius): the host CPU's; EINVAL when
/// no sensor reports it.
fn cb_get_cpu_temperature(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return temperature_failure(ctx, EFAULT);
    }
    let c = host_sensors(Path::new("/sys/class"));
    if c[0] < 0 {
        return temperature_failure(ctx, EINVAL);
    }
    if ctx.write_u32(out, c[0] as u32) { OK } else { temperature_failure(ctx, EFAULT) }
}

/// Highest SoC sensor channel the console accepts.
const SOC_SENSOR_LAST: i32 = 64;

/// sceKernelGetSocSensorTemperature(int sensor, int* celsius): the console
/// has channels 0 to 64 and refuses any other with EINVAL before it looks
/// at the output (PS5 firmware 13.60). The host has sensors for 0 (CPU),
/// 1 (board) and 2 (storage) only; the rest are EINVAL here.
fn cb_get_soc_sensor_temperature(ctx: &mut CpuContext) -> HleResult {
    let (sensor, out) = (ctx.rdi() as i32, ctx.rsi());
    if !(0..=SOC_SENSOR_LAST).contains(&sensor) {
        return temperature_failure(ctx, EINVAL);
    }
    if out == 0 {
        return temperature_failure(ctx, EFAULT);
    }
    if sensor >= 3 {
        return temperature_failure(ctx, EINVAL);
    }
    let c = host_sensors(Path::new("/sys/class"));
    if c[sensor as usize] < 0 {
        return temperature_failure(ctx, EINVAL);
    }
    if ctx.write_u32(out, c[sensor as usize] as u32) { OK } else { temperature_failure(ctx, EFAULT) }
}

/// sceKernelGetSocPowerConsumption(out*): no host reading; EINVAL.
fn cb_get_soc_power_consumption(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 { sce_error(EFAULT) } else { sce_error(EINVAL) }
}

fn write_stderr(text: &str) -> bool {
    use std::io::Write;
    static OUT: Mutex<()> = Mutex::new(());
    let _g = OUT.lock().unwrap_or_else(|e| e.into_inner());
    let mut e = std::io::stderr();
    e.write_all(text.as_bytes()).is_ok() && e.flush().is_ok()
}

/// sceKernelDebugOutText(int channel, const char* text): to the host's
/// standard error.
fn cb_debug_out_text(ctx: &mut CpuContext) -> HleResult {
    let (channel, text) = (ctx.rdi() as i32, ctx.rsi());
    if channel < 0 {
        return sce_error(EINVAL);
    }
    if text == 0 {
        return sce_error(EFAULT);
    }
    let Some(s) = ctx.read_cstring(text, 1 << 16) else { return sce_error(EFAULT) };
    if write_stderr(&format!("[debug:{channel}] {s}")) { OK } else { sce_error(EIO) }
}

/// SceNotificationRequest: 3120 bytes, type i32 at +0, message at +45.
const NOTIFICATION_SIZE: u64 = 3120;
const NOTIFICATION_MESSAGE: u64 = 45;
const NOTIFICATION_MESSAGE_MAX: usize = 1024;

/// sceKernelSendNotificationRequest(device, request*, size, flags): a
/// type-0 request on device 0 prints its message to standard error.
fn cb_send_notification_request(ctx: &mut CpuContext) -> HleResult {
    let (device, req, size, flags) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), ctx.rcx() as i32);
    if size != NOTIFICATION_SIZE || device < 0 {
        return sce_error(EINVAL);
    }
    if req == 0 {
        return sce_error(EFAULT);
    }
    if device != 0 || flags != 0 {
        return sce_error(EOPNOTSUPP);
    }
    let Some(head) = ctx.read_bytes(req, NOTIFICATION_MESSAGE as usize) else { return sce_error(EFAULT) };
    if i32::from_le_bytes(head[..4].try_into().unwrap()) != 0 || head[NOTIFICATION_MESSAGE as usize - 1] != 0 {
        return sce_error(EOPNOTSUPP);
    }
    let Some(msg) = ctx.read_bytes(req + NOTIFICATION_MESSAGE, NOTIFICATION_MESSAGE_MAX) else { return sce_error(EFAULT) };
    let Some(end) = msg.iter().position(|&b| b == 0) else { return sce_error(EINVAL) };
    if write_stderr(&format!("[notification] {}\n", String::from_utf8_lossy(&msg[..end]))) { OK } else { sce_error(EIO) }
}

const EOPNOTSUPP: i32 = 45;

/// sceKernelGetLibkernelTextLocation(void** address, size_t* size): the
/// first segment of the main program (libkernel itself is not a guest
/// module here).
fn cb_get_libkernel_text_location(ctx: &mut CpuContext) -> HleResult {
    let (addr_out, size_out) = (ctx.rdi(), ctx.rsi());
    if addr_out == 0 || size_out == 0 {
        log::error!("[kernel] sceKernelGetLibkernelTextLocation: null argument");
        return sce_error(EFAULT);
    }
    let Some(&(addr, size, _)) = crate::modules::main_module().as_ref().and_then(|m| m.segments.first()) else {
        log::error!("[kernel] sceKernelGetLibkernelTextLocation: the main program is not loaded");
        return sce_error(ESRCH);
    };
    if ctx.write_u64(addr_out, addr) && ctx.write_u64(size_out, size) { OK } else { sce_error(EFAULT) }
}

/// sceKernelInternalMemoryGetAvailableSize(size_t*): no internal memory
/// is reserved for the title.
fn cb_internal_memory_available(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return sce_error(EINVAL);
    }
    if ctx.write_u64(out, 0) { OK } else { sce_error(EFAULT) }
}

/// SCE_COREDUMP_ERROR_INVALID_ARGUMENT.
const COREDUMP_INVALID_ARGUMENT: HleResult = 0x8118_0000u32 as i32 as i64;

/// sceCoredumpConfigDumpMode(mode): only mode 1 is accepted.
fn cb_coredump_config_dump_mode(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() as i32 == 1 { OK } else { COREDUMP_INVALID_ARGUMENT }
}

/// sceKernelIsCEX(): 1, a retail unit (sceKernelIsDevKit answers 0).
fn cb_is_cex(_ctx: &mut CpuContext) -> HleResult {
    1
}

pub(super) const SYSTEM_EXPORTS: &[(&str, &str, &str, HleCallback)] = &[
    ("iUohlUDiGac", "sceKernelGetCpuFrequency", LIB_KERNEL, cb_get_cpu_frequency),
    ("VOx8NGmHXTs", "sceKernelGetCpumode", LIB_KERNEL, cb_zero),
    ("0vTn5IDMU9A", "sceKernelGetMainSocId", LIB_KERNEL, cb_zero),
    ("8aCOCGoRkUI", "sceKernelIsCEX", LIB_KERNEL, cb_is_cex),
    ("QNjGUdj1HPM", "sceKernelIsDevKit", LIB_KERNEL, cb_zero),
    ("qiL4fFObAxM", "sceKernelGetCpuTemperature", LIB_KERNEL, cb_get_cpu_temperature),
    ("zzKyvoELf4I", "sceKernelGetSocSensorTemperature", LIB_KERNEL, cb_get_soc_sensor_temperature),
    ("CA5jcDb2vYI", "sceKernelGetSocPowerConsumption", LIB_KERNEL, cb_get_soc_power_consumption),
    ("9JYNqN6jAKI", "sceKernelDebugOutText", LIB_KERNEL, cb_debug_out_text),
    ("zl7hupSO0C0", "sceKernelSendNotificationRequest", LIB_KERNEL, cb_send_notification_request),
    ("AmJ0mn2l4lM", "sceKernelGetLibkernelTextLocation", LIB_KERNEL, cb_get_libkernel_text_location),
    ("rjEuM0nb8xg", "sceKernelInternalMemoryGetAvailableSize", LIB_KERNEL, cb_internal_memory_available),
    ("gzLt9Qrauk0", "sceCoredumpConfigDumpMode", LIB_COREDUMP, cb_coredump_config_dump_mode),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister;
    use ps5vm::{MemProtect, VirtualMemory};

    const BASE: u64 = 0x7E62_0000_0000;

    fn call(ctx: &mut CpuContext, cb: HleCallback, args: &[u64]) -> HleResult {
        let regs = [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx];
        for (r, v) in regs.iter().zip(args.iter().chain(std::iter::repeat(&0))) {
            ctx.set(*r, *v);
        }
        cb(ctx)
    }

    /// The sensor scan picks the CPU, board and storage readings of a fake
    /// /sys/class tree, skips batteries and drops implausible values.
    #[test]
    fn host_sensor_scan() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../local/tmp/sensors-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let hw = |n: &str, name: &str, milli: &str| {
            let d = root.join("hwmon").join(n);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("name"), name).unwrap();
            std::fs::write(d.join("temp1_input"), milli).unwrap();
        };
        hw("hwmon0", "BAT0\n", "40000");
        hw("hwmon1", "k10temp\n", "55500");
        hw("hwmon2", "nvme\n", "41000");
        hw("hwmon3", "acpitz\n", "500000");
        assert_eq!(host_sensors(&root), [55, -1, 41]);
        hw("hwmon3", "acpitz\n", "30000");
        assert_eq!(host_sensors(&root), [55, 30, 41]);
        std::fs::remove_dir_all(&root).unwrap();
        assert_eq!(host_sensors(&root), [-1, -1, -1]);
    }

    #[test]
    fn argument_checks() {
        let vm = VirtualMemory::new();
        vm.map(BASE, 0x2000, MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        assert!(call(&mut ctx, cb_get_cpu_frequency, &[]) > 0);
        assert_eq!(call(&mut ctx, cb_is_cex, &[]), 1, "a retail unit");
        assert_eq!(call(&mut ctx, cb_get_cpu_temperature, &[0]), sce_error(EFAULT));
        assert_eq!(call(&mut ctx, cb_get_soc_sensor_temperature, &[3, BASE]), sce_error(EINVAL));
        assert_eq!(call(&mut ctx, cb_get_soc_sensor_temperature, &[0, 0]), sce_error(EFAULT));
        // A channel outside 0..=64 is EINVAL even with a null output; inside
        // it a null output is EFAULT.
        for bad in [-1i32, 65, 256, i32::MAX, i32::MIN] {
            assert_eq!(call(&mut ctx, cb_get_soc_sensor_temperature, &[bad as u32 as u64, 0]), sce_error(EINVAL), "{bad}");
        }
        for unmapped in [3u64, 15, 32, 64] {
            assert_eq!(call(&mut ctx, cb_get_soc_sensor_temperature, &[unmapped, 0]), sce_error(EFAULT), "{unmapped}");
        }
        assert_eq!(call(&mut ctx, cb_get_soc_power_consumption, &[0]), sce_error(EFAULT));
        assert_eq!(call(&mut ctx, cb_get_soc_power_consumption, &[BASE]), sce_error(EINVAL));
        assert_eq!(call(&mut ctx, cb_debug_out_text, &[u32::MAX as u64, BASE]), sce_error(EINVAL));
        assert_eq!(call(&mut ctx, cb_debug_out_text, &[0, 0]), sce_error(EFAULT));
        ctx.write_bytes(BASE, b"hello\n\0");
        assert_eq!(call(&mut ctx, cb_debug_out_text, &[1, BASE]), OK);
        assert_eq!(call(&mut ctx, cb_internal_memory_available, &[0]), sce_error(EINVAL));
        ctx.write_u64(BASE, 7);
        assert_eq!(call(&mut ctx, cb_internal_memory_available, &[BASE]), OK);
        assert_eq!(ctx.read_u64(BASE), Some(0));
        assert_eq!(call(&mut ctx, cb_coredump_config_dump_mode, &[1]), OK);
        assert_eq!(call(&mut ctx, cb_coredump_config_dump_mode, &[0]), COREDUMP_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, cb_get_libkernel_text_location, &[0, BASE]), sce_error(EFAULT));
        // Notification: size, device, flags, type and terminator checks.
        let req = BASE;
        ctx.write_bytes(req, &vec![0u8; NOTIFICATION_SIZE as usize]);
        ctx.write_bytes(req + NOTIFICATION_MESSAGE, b"saved\0");
        assert_eq!(call(&mut ctx, cb_send_notification_request, &[0, req, 100, 0]), sce_error(EINVAL));
        assert_eq!(call(&mut ctx, cb_send_notification_request, &[0, 0, NOTIFICATION_SIZE, 0]), sce_error(EFAULT));
        assert_eq!(call(&mut ctx, cb_send_notification_request, &[1, req, NOTIFICATION_SIZE, 0]), sce_error(EOPNOTSUPP));
        assert_eq!(call(&mut ctx, cb_send_notification_request, &[0, req, NOTIFICATION_SIZE, 0]), OK);
        ctx.write_u32(req, 1);
        assert_eq!(call(&mut ctx, cb_send_notification_request, &[0, req, NOTIFICATION_SIZE, 0]), sce_error(EOPNOTSUPP));
        ctx.write_u32(req, 0);
        ctx.write_bytes(req + NOTIFICATION_MESSAGE, &[b'x'; NOTIFICATION_MESSAGE_MAX]);
        assert_eq!(call(&mut ctx, cb_send_notification_request, &[0, req, NOTIFICATION_SIZE, 0]), sce_error(EINVAL));
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _, _) in SYSTEM_EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }
}
