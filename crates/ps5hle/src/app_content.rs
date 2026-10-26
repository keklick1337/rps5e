// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceAppContent HLE: boot parameters, app params, add-on content and
//! the temporary-data mount.
//!
//! Mount calls only report guest mount points (e.g. "/temp0"); mapping
//! them to host directories is the filesystem layer's job.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};
use std::path::PathBuf;

const LIB: &str = "libSceAppContent";
const OK: HleResult = 0;
// SCE_APP_CONTENT_ERROR_*.
const ERR_NOT_INITIALIZED: HleResult = 0x80D9_0001u32 as i32 as i64;
const ERR_PARAMETER: HleResult = 0x80D9_0002u32 as i32 as i64;
const ERR_BUSY: HleResult = 0x80D9_0003u32 as i32 as i64;
const ERR_NOT_MOUNTED: HleResult = 0x80D9_0004u32 as i32 as i64;
const ERR_NOT_FOUND: HleResult = 0x80D9_0005u32 as i32 as i64;
const ERR_DRM_NO_ENTITLEMENT: HleResult = 0x80D9_0007u32 as i32 as i64;

/// Reported when the host volume cannot be queried (1 GiB).
const FALLBACK_AVAILABLE_SPACE_KB: u64 = 1024 * 1024;
/// SceAppContentBootParam: u8 reserved1[4]; u32 attr; u8 reserved2[32].
const BOOT_PARAM_SIZE: usize = 40;
pub const TEMP0_MOUNT_POINT: &str = "/temp0";
pub const DOWNLOAD0_MOUNT_POINT: &str = "/download0";

/// Whether the temporary data is mounted (TemporaryDataMount2 until
/// TemporaryDataUnmount).
static TEMP0_MOUNTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Set by sceAppContentInitialize; every other call fails with
/// NOT_INITIALIZED before it, whatever its arguments.
static INITIALIZED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn initialized() -> bool {
    INITIALIZED.load(std::sync::atomic::Ordering::Acquire)
}
const APP_PARAM_SKU_FLAG: u32 = 0;
/// SCE_APP_CONTENT_APPPARAM_SKU_FLAG_FULL.
const APP_PARAM_SKU_FLAG_FULL: i32 = 3;

/// Value for an app param id, or None when the id is invalid.
/// Ids 1..=4 are userDefinedParam1..4 from param.json (0 when absent).
pub fn app_param_value(param_id: u32, param_json: Option<&str>) -> Option<i32> {
    if param_id == APP_PARAM_SKU_FLAG {
        return Some(APP_PARAM_SKU_FLAG_FULL);
    }
    if !(1..=4).contains(&param_id) {
        return None;
    }
    let key = format!("userDefinedParam{param_id}");
    Some(param_json.and_then(|j| crate::app_info::param_json_i32(j, &key)).unwrap_or(0))
}

/// Host directory whose volume backs temporary data.
fn temp_storage_dir() -> PathBuf {
    std::env::var("RPS5E_TEMP0_DIR")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
}

fn host_available_space_kb() -> u64 {
    use crate::host::fs::OsStrExt;
    let mut dir = temp_storage_dir();
    // Walk up to an existing ancestor: nothing is created here.
    while !dir.exists() {
        if !dir.pop() {
            return FALLBACK_AVAILABLE_SPACE_KB;
        }
    }
    let Ok(cpath) = std::ffi::CString::new(dir.as_os_str().as_bytes()) else {
        return FALLBACK_AVAILABLE_SPACE_KB;
    };
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(cpath.as_ptr(), &mut st) } != 0 {
        return FALLBACK_AVAILABLE_SPACE_KB;
    }
    let bytes = (st.f_bavail as u64).saturating_mul(st.f_frsize as u64);
    if bytes == 0 { FALLBACK_AVAILABLE_SPACE_KB } else { bytes / 1024 }
}

fn clear_boot_param(ctx: &CpuContext, boot_param: u64) -> bool {
    boot_param == 0 || ctx.write_bytes(boot_param, &[0u8; BOOT_PARAM_SIZE])
}

/// sceAppContentInitialize(const SceAppContentInitParam*, SceAppContentBootParam*):
/// the boot parameters, when given, are cleared (attr 0: normal boot).
pub(crate) fn cb_initialize(ctx: &mut CpuContext) -> HleResult {
    if !clear_boot_param(ctx, ctx.rsi()) {
        return ERR_PARAMETER;
    }
    INITIALIZED.store(true, std::sync::atomic::Ordering::Release);
    crate::system_service::notify_entitlement_update();
    OK
}

/// (serviceLabel, list, listNum, hitNum*): no add-on content installed.
/// The hit count is required; the list is never written.
fn cb_get_addcont_info_list(ctx: &mut CpuContext) -> HleResult {
    let hit = ctx.rcx();
    if hit == 0 || !ctx.write_u32(hit, 0) {
        return ERR_PARAMETER;
    }
    OK
}

fn cb_app_param_get_int(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as u32;
    let out = ctx.rsi();
    if out == 0 {
        return ERR_PARAMETER;
    }
    let json = if (1..=4).contains(&id) { crate::app_info::read_param_json() } else { None };
    let Some(v) = app_param_value(id, json.as_deref()) else {
        return ERR_PARAMETER;
    };
    if !ctx.write_bytes(out, &v.to_le_bytes()) {
        return ERR_PARAMETER;
    }
    log::debug!("[APPCONTENT] app_param_get_int id={id} value={v}");
    OK
}

/// SCE_APP_CONTENT_TEMPORARY_DATA_OPTION_FORMAT: empty the data on mount.
const TEMPORARY_DATA_OPTION_FORMAT: u32 = 1;

/// Mounts the temporary data once ("/temp0"; BUSY while mounted), emptying
/// it first with the FORMAT option.
fn mount_temporary_data(ctx: &CpuContext, option: u32, out: u64) -> HleResult {
    use std::sync::atomic::Ordering;
    if out == 0 || option > TEMPORARY_DATA_OPTION_FORMAT {
        return ERR_PARAMETER;
    }
    if TEMP0_MOUNTED.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
        return ERR_BUSY;
    }
    if option == TEMPORARY_DATA_OPTION_FORMAT {
        if let Some(root) = crate::filesystem::resolve_guest_path(TEMP0_MOUNT_POINT) {
            match clear_directory(&root) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => log::warn!("[APPCONTENT] temporary data format on mount {}: {e}", root.display()),
            }
        }
    }
    let mut s = [0u8; MOUNT_POINT_SIZE];
    s[..TEMP0_MOUNT_POINT.len()].copy_from_slice(TEMP0_MOUNT_POINT.as_bytes());
    if !ctx.write_bytes(out, &s) {
        TEMP0_MOUNTED.store(false, Ordering::Release);
        return ERR_PARAMETER;
    }
    OK
}

/// sceAppContentTemporaryDataMount(mountPoint*): the FORMAT mount.
fn cb_temporary_data_mount(ctx: &mut CpuContext) -> HleResult {
    mount_temporary_data(ctx, TEMPORARY_DATA_OPTION_FORMAT, ctx.rdi())
}

/// sceAppContentTemporaryDataMount2(option, mountPoint*).
fn cb_temporary_data_mount2(ctx: &mut CpuContext) -> HleResult {
    mount_temporary_data(ctx, ctx.rdi() as u32, ctx.rsi())
}

/// The mount point text of a SceAppContentMountPoint.
fn read_mount_point(ctx: &CpuContext, mp: u64) -> Option<Vec<u8>> {
    let raw = ctx.read_bytes(mp, MOUNT_POINT_SIZE)?;
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    Some(raw[..end].to_vec())
}

/// sceAppContentTemporaryDataUnmount(const SceAppContentMountPoint*): only
/// the mounted "/temp0".
fn cb_temporary_data_unmount(ctx: &mut CpuContext) -> HleResult {
    let mp = ctx.rdi();
    if mp == 0 {
        return ERR_PARAMETER;
    }
    let Some(name) = read_mount_point(ctx, mp) else { return ERR_PARAMETER };
    if name != TEMP0_MOUNT_POINT.as_bytes() {
        return ERR_NOT_MOUNTED;
    }
    let was_mounted = TEMP0_MOUNTED.swap(false, std::sync::atomic::Ordering::AcqRel);
    if was_mounted { OK } else { ERR_NOT_MOUNTED }
}

/// (mountPoint/label, u64* availableSpaceKb).
fn write_available_space(ctx: &CpuContext, kind: &str) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return ERR_PARAMETER;
    }
    let kb = host_available_space_kb();
    if !ctx.write_u64(out, kb) {
        return ERR_PARAMETER;
    }
    log::debug!("[APPCONTENT] {kind}_available_space_kb={kb}");
    OK
}

fn cb_temporary_data_get_available_space_kb(ctx: &mut CpuContext) -> HleResult {
    write_available_space(ctx, "temporary_data")
}

/// sceAppContentDownloadDataGetAvailableSpaceKb(const SceAppContentMountPoint*,
/// u64* availableSpaceKb): the mount point must be "/download0"
/// (NOT_MOUNTED otherwise).
fn cb_download_data_get_available_space_kb(ctx: &mut CpuContext) -> HleResult {
    let mp = ctx.rdi();
    if mp == 0 || ctx.rsi() == 0 {
        return ERR_PARAMETER;
    }
    let Some(name) = read_mount_point(ctx, mp) else { return ERR_PARAMETER };
    if name != DOWNLOAD0_MOUNT_POINT.as_bytes() {
        return ERR_NOT_MOUNTED;
    }
    write_available_space(ctx, "download_data")
}

/// SceAppContentMountPoint: char data[16].
const MOUNT_POINT_SIZE: usize = 16;

/// sceAppContentAddcontMount(serviceLabel, const SceNpUnifiedEntitlementLabel*,
/// SceAppContentMountPoint*): no add-on content is installed, so nothing
/// can be mounted.
fn cb_addcont_mount(ctx: &mut CpuContext) -> HleResult {
    let label = ctx.rsi();
    let mount_point = ctx.rdx();
    if label == 0 || mount_point == 0 {
        return ERR_PARAMETER;
    }
    if !ctx.write_bytes(mount_point, &[0u8; MOUNT_POINT_SIZE]) {
        return ERR_PARAMETER;
    }
    let raw = ctx.read_bytes(label, 17).unwrap_or_default();
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    log::debug!("[APPCONTENT] addcont_mount '{}': not installed", String::from_utf8_lossy(&raw[..end]));
    ERR_DRM_NO_ENTITLEMENT
}

/// sceAppContentAddcontUnmount(const SceAppContentMountPoint*): an empty
/// mount point was never mounted.
fn cb_addcont_unmount(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return ERR_PARAMETER;
    }
    match ctx.read_bytes(ctx.rdi(), 1) {
        None => ERR_PARAMETER,
        Some(b) if b[0] == 0 => ERR_NOT_MOUNTED,
        Some(_) => OK,
    }
}

/// sceAppContentGetAddcontInfo(serviceLabel, const SceNpUnifiedEntitlementLabel*,
/// SceAppContentAddcontInfo*): no add-on content is owned.
fn cb_get_addcont_info(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 || ctx.rdx() == 0 {
        return ERR_PARAMETER;
    }
    ERR_DRM_NO_ENTITLEMENT
}

/// sceAppContentGetEntitlementKey(serviceLabel, const SceNpUnifiedEntitlementLabel*,
/// SceAppContentEntitlementKey*): no entitlement is owned, so there is no
/// key; the 16-byte key is left as it was.
fn cb_get_entitlement_key(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 || ctx.rdx() == 0 {
        return ERR_PARAMETER;
    }
    ERR_DRM_NO_ENTITLEMENT
}

/// AddcontEnqueueDownload / AddcontDelete(serviceLabel, label*): there is
/// no add-on content to download or delete.
fn cb_addcont_not_found(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 {
        return ERR_PARAMETER;
    }
    ERR_NOT_FOUND
}

/// sceAppContentGetAddcontDownloadProgress(serviceLabel, label*, progress*):
/// no download exists.
fn cb_get_addcont_download_progress(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 || ctx.rdx() == 0 {
        return ERR_PARAMETER;
    }
    ERR_NOT_FOUND
}

/// sceAppContentDownloadDataFormat(const SceAppContentMountPoint*): empty
/// the download data directory ("/download0").
fn cb_download_data_format(ctx: &mut CpuContext) -> HleResult {
    let mp = ctx.rdi();
    if mp == 0 {
        return ERR_PARAMETER;
    }
    let Some(name) = read_mount_point(ctx, mp) else { return ERR_PARAMETER };
    if name != DOWNLOAD0_MOUNT_POINT.as_bytes() {
        log::warn!("[APPCONTENT] download data format of '{}': not the download data mount", String::from_utf8_lossy(&name));
        return ERR_NOT_MOUNTED;
    }
    let Some(root) = crate::filesystem::resolve_guest_path(DOWNLOAD0_MOUNT_POINT) else {
        log::warn!("[APPCONTENT] download data format: {DOWNLOAD0_MOUNT_POINT} has no host directory");
        return ERR_NOT_MOUNTED;
    };
    match clear_directory(&root) {
        Ok(()) => OK,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => OK,
        Err(e) => {
            log::warn!("[APPCONTENT] download data format {}: {e}", root.display());
            0x8002_0005u32 as i32 as i64 // EIO
        }
    }
}

/// Delete everything below `dir` (the directory itself stays).
fn clear_directory(dir: &std::path::Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() && !path.is_symlink() {
            std::fs::remove_dir_all(&path)?;
        } else {
            std::fs::remove_file(&path)?;
        }
    }
    Ok(())
}

/// sceAppContentTemporaryDataFormat(const SceAppContentMountPoint*): empty
/// the temporary data mount.
fn cb_temporary_data_format(ctx: &mut CpuContext) -> HleResult {
    let mp = ctx.rdi();
    if mp == 0 {
        return ERR_PARAMETER;
    }
    let Some(raw) = ctx.read_bytes(mp, MOUNT_POINT_SIZE) else {
        return ERR_PARAMETER;
    };
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    if &raw[..end] != TEMP0_MOUNT_POINT.as_bytes() {
        return OK;
    }
    let Some(root) = crate::filesystem::resolve_guest_path(TEMP0_MOUNT_POINT) else {
        return ERR_NOT_MOUNTED;
    };
    match clear_directory(&root) {
        Ok(()) => OK,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => OK,
        Err(e) => {
            log::warn!("[APPCONTENT] temporary data format {}: {e}", root.display());
            0x8002_0005u32 as i32 as i64 // EIO
        }
    }
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("R9lA82OraNs", "sceAppContentInitialize", cb_initialize),
    ("xnd8BJzAxmk", "sceAppContentGetAddcontInfoList", cb_get_addcont_info_list),
    ("99b82IKXpH4", "sceAppContentAppParamGetInt", cb_app_param_get_int),
    ("buYbeLOGWmA", "sceAppContentTemporaryDataMount2", cb_temporary_data_mount2),
    ("7bOLX66Iz-U", "sceAppContentTemporaryDataMount", cb_temporary_data_mount),
    ("SaKib2Ug0yI", "sceAppContentTemporaryDataGetAvailableSpaceKb", cb_temporary_data_get_available_space_kb),
    ("Gl6w5i0JokY", "sceAppContentDownloadDataGetAvailableSpaceKb", cb_download_data_get_available_space_kb),
    ("VANhIWcqYak", "sceAppContentAddcontMount", cb_addcont_mount),
    ("3rHWaV-1KC4", "sceAppContentAddcontUnmount", cb_addcont_unmount),
    ("a5N7lAG0y2Q", "sceAppContentTemporaryDataFormat", cb_temporary_data_format),
    ("bcolXMmp6qQ", "sceAppContentTemporaryDataUnmount", cb_temporary_data_unmount),
    ("m47juOmH0VE", "sceAppContentGetAddcontInfo", cb_get_addcont_info),
    ("XTWR0UXvcgs", "sceAppContentGetEntitlementKey", cb_get_entitlement_key),
    ("7gxh+5QubhY", "sceAppContentAddcontEnqueueDownload", cb_addcont_not_found),
    ("ZiATpP9gEkA", "sceAppContentAddcontDelete", cb_addcont_not_found),
    ("5bvvbUSiFs4", "sceAppContentGetAddcontDownloadProgress", cb_get_addcont_download_progress),
    ("CN7EbEV7MFU", "sceAppContentDownloadDataFormat", cb_download_data_format),
];

/// Export `I` of `EXPORTS` behind the initialisation check.
fn gated<const I: usize>(ctx: &mut CpuContext) -> HleResult {
    run_export(initialized(), I, ctx)
}

/// Not initialised wins over every argument check.
fn run_export(initialized: bool, index: usize, ctx: &mut CpuContext) -> HleResult {
    if !initialized {
        return ERR_NOT_INITIALIZED;
    }
    (EXPORTS[index].2)(ctx)
}

/// What the guest calls: sceAppContentInitialize (export 0) as it is,
/// every other export only once the library is initialised.
const GUEST_ENTRIES: [HleCallback; 17] = [
    cb_initialize,
    gated::<1>,
    gated::<2>,
    gated::<3>,
    gated::<4>,
    gated::<5>,
    gated::<6>,
    gated::<7>,
    gated::<8>,
    gated::<9>,
    gated::<10>,
    gated::<11>,
    gated::<12>,
    gated::<13>,
    gated::<14>,
    gated::<15>,
    gated::<16>,
];
const _: () = assert!(EXPORTS.len() == GUEST_ENTRIES.len());

pub fn register(reg: &mut HleRegistry) {
    for (&(nid, name, _), &cb) in EXPORTS.iter().zip(GUEST_ENTRIES.iter()) {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_params() {
        let json = r#"{"userDefinedParam1": 3, "userDefinedParam3": -2}"#;
        assert_eq!(app_param_value(0, None), Some(3));
        assert_eq!(app_param_value(1, Some(json)), Some(3));
        assert_eq!(app_param_value(2, Some(json)), Some(0));
        assert_eq!(app_param_value(3, Some(json)), Some(-2));
        assert_eq!(app_param_value(4, None), Some(0));
        assert_eq!(app_param_value(5, Some(json)), None);
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn format_clears_directory() {
        let dir = std::env::temp_dir().join(format!("rps5e-format-test-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sub/deeper")).unwrap();
        std::fs::write(dir.join("a.bin"), b"x").unwrap();
        std::fs::write(dir.join("sub/deeper/b.bin"), b"y").unwrap();
        clear_directory(&dir).unwrap();
        assert!(dir.exists());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        std::fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn boot_param_mounts_and_unmount() {
        let _mount = TEMP0_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use crate::context::CpuRegister::*;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D40_0300_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);

        ctx.write_bytes(BASE, &[0xAB; 0x40]);
        assert!(clear_boot_param(&ctx, 0), "boot param may be null");
        assert!(clear_boot_param(&ctx, BASE));
        assert_eq!(ctx.read_bytes(BASE, BOOT_PARAM_SIZE).unwrap(), vec![0; BOOT_PARAM_SIZE]);
        assert_eq!(ctx.read_bytes(BASE + BOOT_PARAM_SIZE as u64, 1).unwrap()[0], 0xAB);

        ctx.write_bytes(BASE, &[0xAB; 0x40]);
        ctx.set(Rdi, 0);
        ctx.set(Rsi, BASE);
        assert_eq!(cb_temporary_data_mount2(&mut ctx), OK);
        let mut expect = [0u8; MOUNT_POINT_SIZE];
        expect[..6].copy_from_slice(b"/temp0");
        assert_eq!(ctx.read_bytes(BASE, MOUNT_POINT_SIZE).unwrap(), expect);

        ctx.set(Rdi, BASE);
        assert_eq!(cb_addcont_unmount(&mut ctx), OK, "a named mount point unmounts");
        ctx.set(Rdi, 1);
        ctx.set(Rsi, BASE + 0x100);
        ctx.set(Rdx, BASE);
        assert_eq!(cb_addcont_mount(&mut ctx), ERR_DRM_NO_ENTITLEMENT);
        assert_eq!(ctx.read_bytes(BASE, MOUNT_POINT_SIZE).unwrap(), vec![0; MOUNT_POINT_SIZE]);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_addcont_unmount(&mut ctx), ERR_NOT_MOUNTED);
        ctx.set(Rdi, 0);
        assert_eq!(cb_addcont_unmount(&mut ctx), ERR_PARAMETER);
        ctx.write_bytes(BASE, b"/temp0\0");
        ctx.set(Rdi, BASE);
        assert_eq!(cb_temporary_data_unmount(&mut ctx), OK);
    }

    /// Before sceAppContentInitialize every other export reports
    /// NOT_INITIALIZED, null arguments or not; once initialised the export
    /// itself runs.
    #[test]
    fn exports_need_initialize() {
        use crate::context::CpuRegister::*;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5FAD_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        for args in [[0u64; 4], [BASE; 4]] {
            for (r, v) in [Rdi, Rsi, Rdx, Rcx].into_iter().zip(args) {
                ctx.set(r, v);
            }
            for (i, &(_, name, _)) in EXPORTS.iter().enumerate().skip(1) {
                assert_eq!(run_export(false, i, &mut ctx), ERR_NOT_INITIALIZED, "{name}");
            }
        }
        assert_eq!(EXPORTS[0].1, "sceAppContentInitialize");
        assert_eq!(EXPORTS[1].1, "sceAppContentGetAddcontInfoList");
        ctx.set(Rcx, 0);
        assert_eq!(run_export(true, 1, &mut ctx), ERR_PARAMETER);
    }

    /// The hit count is required and always 0; the list is left alone.
    #[test]
    fn addcont_info_list_needs_the_hit_count() {
        use crate::context::CpuRegister::*;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5FAD_0100_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, &[0x5A; 24]);
        for (list, num) in [(0, 0), (BASE, 0), (0, 1), (BASE, 1)] {
            ctx.set(Rsi, list);
            ctx.set(Rdx, num);
            ctx.set(Rcx, BASE + 0x100);
            ctx.write_u32(BASE + 0x100, 0x5A5A_5A5A);
            assert_eq!(cb_get_addcont_info_list(&mut ctx), OK);
            assert_eq!(ctx.read_u32(BASE + 0x100), Some(0));
            ctx.set(Rcx, 0);
            assert_eq!(cb_get_addcont_info_list(&mut ctx), ERR_PARAMETER);
        }
        assert_eq!(ctx.read_bytes(BASE, 24).unwrap(), vec![0x5A; 24]);
    }

    /// The temporary data mounts once: a second mount (either call) is
    /// BUSY until it is unmounted; the option must be 0 or FORMAT.
    #[test]
    fn temporary_data_mounts_once() {
        let _mount = TEMP0_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use crate::context::CpuRegister::*;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5FAD_0200_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdi, 0);
        assert_eq!(cb_temporary_data_mount(&mut ctx), ERR_PARAMETER);
        ctx.set(Rdi, 2);
        ctx.set(Rsi, BASE);
        assert_eq!(cb_temporary_data_mount2(&mut ctx), ERR_PARAMETER);
        ctx.write_bytes(BASE, &[0x5A; MOUNT_POINT_SIZE]);
        ctx.set(Rdi, 0);
        assert_eq!(cb_temporary_data_mount2(&mut ctx), OK);
        assert_eq!(&ctx.read_bytes(BASE, 7).unwrap(), b"/temp0\0");
        ctx.set(Rsi, BASE + 0x100);
        assert_eq!(cb_temporary_data_mount2(&mut ctx), ERR_BUSY);
        ctx.set(Rdi, BASE + 0x100);
        assert_eq!(cb_temporary_data_mount(&mut ctx), ERR_BUSY);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_temporary_data_unmount(&mut ctx), OK);
    }

    /// Serialises the tests that mount "/temp0".
    static TEMP0_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Unmount needs the mounted "/temp0"; a second unmount fails.
    #[test]
    fn temporary_data_unmount() {
        let _mount = TEMP0_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use crate::context::CpuRegister::*;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5FA9_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdi, 0);
        assert_eq!(cb_temporary_data_unmount(&mut ctx), ERR_PARAMETER);
        ctx.set(Rsi, BASE);
        assert_eq!(cb_temporary_data_mount2(&mut ctx), OK);
        ctx.write_bytes(BASE + 0x100, b"/temp1\0");
        ctx.set(Rdi, BASE + 0x100);
        assert_eq!(cb_temporary_data_unmount(&mut ctx), ERR_NOT_MOUNTED);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_temporary_data_unmount(&mut ctx), OK);
        assert_eq!(cb_temporary_data_unmount(&mut ctx), ERR_NOT_MOUNTED);
    }

    /// Port of the guest add-on content checks: nothing is owned or
    /// downloadable; a null label is a parameter error.
    #[test]
    fn addcont_queries_without_content() {
        use crate::context::CpuRegister::*;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5FAA_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, b"ADDCONT000000000\0");
        ctx.set(Rdi, 0);
        ctx.set(Rsi, 0);
        ctx.set(Rdx, BASE + 0x100);
        assert_eq!(cb_get_addcont_info(&mut ctx), ERR_PARAMETER);
        assert_eq!(cb_addcont_not_found(&mut ctx), ERR_PARAMETER);
        assert_eq!(cb_get_addcont_download_progress(&mut ctx), ERR_PARAMETER);
        ctx.set(Rsi, BASE);
        assert_eq!(cb_get_addcont_info(&mut ctx), ERR_DRM_NO_ENTITLEMENT);
        assert_eq!(cb_addcont_not_found(&mut ctx), ERR_NOT_FOUND);
        assert_eq!(cb_get_addcont_download_progress(&mut ctx), ERR_NOT_FOUND);
        ctx.set(Rdx, 0);
        assert_eq!(cb_get_addcont_info(&mut ctx), ERR_PARAMETER);
        assert_eq!(cb_get_addcont_download_progress(&mut ctx), ERR_PARAMETER);

        // GetEntitlementKey: null label or key is PARAMETER, otherwise no
        // entitlement and the key untouched.
        assert_eq!(cb_get_entitlement_key(&mut ctx), ERR_PARAMETER);
        ctx.write_bytes(BASE + 0x200, &[0x5A; 16]);
        ctx.set(Rdx, BASE + 0x200);
        assert_eq!(cb_get_entitlement_key(&mut ctx), ERR_DRM_NO_ENTITLEMENT);
        assert_eq!(ctx.read_bytes(BASE + 0x200, 16).unwrap(), vec![0x5A; 16]);
        ctx.set(Rsi, 0);
        assert_eq!(cb_get_entitlement_key(&mut ctx), ERR_PARAMETER);
    }

    /// Only the "/download0" mount point can be formatted.
    #[test]
    fn download_data_format_mount_point() {
        use crate::context::CpuRegister::*;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5FAB_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdi, 0);
        assert_eq!(cb_download_data_format(&mut ctx), ERR_PARAMETER);
        ctx.write_bytes(BASE, b"/temp0\0");
        ctx.set(Rdi, BASE);
        assert_eq!(cb_download_data_format(&mut ctx), ERR_NOT_MOUNTED);
        // The available space needs both pointers, then "/download0".
        ctx.set(Rsi, 0);
        assert_eq!(cb_download_data_get_available_space_kb(&mut ctx), ERR_PARAMETER);
        ctx.set(Rsi, BASE + 0x100);
        assert_eq!(cb_download_data_get_available_space_kb(&mut ctx), ERR_NOT_MOUNTED);
        ctx.write_bytes(BASE, b"/download0\0");
        assert_eq!(cb_download_data_get_available_space_kb(&mut ctx), OK);
        assert!(ctx.read_u64(BASE + 0x100).unwrap() > 0);
        ctx.set(Rdi, 0);
        assert_eq!(cb_download_data_get_available_space_kb(&mut ctx), ERR_PARAMETER);
    }

    #[test]
    fn free_space_is_positive() {
        assert!(host_available_space_kb() > 0);
    }
}
