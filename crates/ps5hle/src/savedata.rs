// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceSaveData: save directories backed by host directories.
//!
//! Layout under the save root (`$RPS5E_SAVEDATA_DIR`, else
//! `<data dir>/savedata`, the data dir being `--data-dir` /
//! `$RPS5E_DATA_DIR`, see [`crate::filesystem::user_data_dir`]):
//!
//! ```text
//! <root>/<titleId>/<dirName>/...              save files (mounted at /savedataN)
//! <root>/<titleId>/<dirName>/sce_sys/param.bin  SceSaveDataParam (0x530 bytes)
//! <root>/<titleId>/<dirName>/sce_sys/blocks.bin allocated blocks (u64 LE)
//! <root>/<titleId>/sce_sdmemory[N]/<userId>/memory.dat  SaveDataMemory slot N
//! <root>/<titleId>/sce_sdmemory[N]/<userId>/param.bin   its SceSaveDataParam
//! ```
//!
//! Files are replaced through a flushed `.tmp` sibling so a failed write
//! never destroys the previous save.

mod slots;
#[cfg(test)]
mod tests;

use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};
use slots::{Available, MountSlots};
use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

const fn err(code: u32) -> HleResult {
    code as i32 as i64
}

const OK: HleResult = 0;
const ERR_PARAMETER: HleResult = err(0x809F_0000);
const ERR_BUSY: HleResult = err(0x809F_0003);
const ERR_NOT_MOUNTED: HleResult = err(0x809F_0004);
const ERR_EXISTS: HleResult = err(0x809F_0007);
const ERR_NOT_FOUND: HleResult = err(0x809F_0008);
const ERR_INTERNAL: HleResult = err(0x809F_000B);
const ERR_MOUNT_FULL: HleResult = err(0x809F_000C);
const ERR_BROKEN: HleResult = err(0x809F_000F);
const ERR_INVALID_LOGIN_USER: HleResult = err(0x809F_0011);
const ERR_MEMORY_NOT_READY: HleResult = err(0x809F_0012);
const ERR_LIMITATION_OVER: HleResult = err(0x809F_0017);

const BLOCKS_MIN: u64 = 48;
const BLOCKS_MAX: u64 = 16384;

/// SceSaveDataTitleId: data[10] + padding[6].
const TITLE_ID_DATA: usize = 10;
const TITLE_ID_SIZE: usize = 16;
/// SceSaveDataDirName.
const DIR_NAME_SIZE: usize = 32;
/// SceSaveDataMountPoint.
const MOUNT_POINT_SIZE: usize = 16;

/// SceSaveDataParam: title[128]@0, subTitle[128]@0x80, detail[1024]@0x100,
/// userParam@0x500, mtime@0x508, reserved[32]@0x510.
const PARAM_SIZE: usize = 0x530;
const PARAM_TITLE: std::ops::Range<usize> = 0x000..0x080;
const PARAM_SUB_TITLE: std::ops::Range<usize> = 0x080..0x100;
const PARAM_DETAIL: std::ops::Range<usize> = 0x100..0x500;
const PARAM_USER_PARAM: usize = 0x500;
const PARAM_MTIME: usize = 0x508;
type Param = [u8; PARAM_SIZE];

/// SceSaveDataMountResult: mountPoint[16]@0, requiredBlocks@0x10,
/// unused@0x18, mountStatus@0x1C, reserved.
const MOUNT_RESULT_SIZE: usize = 0x40;
/// SceSaveDataMountInfo / SceSaveDataSearchInfo: blocks@0, freeBlocks@8, reserved[32].
const BLOCKS_INFO_SIZE: usize = 0x30;
/// SceSaveDataDirNameSearchResult: hitNum@0, dirNames*@8, dirNamesNum@0x10,
/// setNum@0x14, params*@0x18, infos*@0x20, reserved[12]@0x28, pad@0x34.
const SEARCH_RESULT_SIZE: usize = 0x38;
/// SceSaveDataEvent: type@0, errorCode@4, userId@8, titleId@0x10,
/// dirName@0x20, reserved[40]@0x40.
const EVENT_SIZE: usize = 0x68;
/// SceSaveDataMemoryData: buf@0, bufSize@8, offset@0x10, reserved[40].
const MEMORY_DATA_SIZE: u64 = 0x40;
/// SceSaveDataMemorySetupResult: existedMemorySize@0, reserved[16].
const MEMORY_SETUP_RESULT_SIZE: usize = 0x18;
/// SceSaveDataIcon: buf@0, bufSize@8, dataSize@0x10, reserved[32].
const ICON_DATA_SIZE_OFFSET: u64 = 0x10;

const SORT_KEY_DIR_NAME: u32 = 0;
const SORT_KEY_USER_PARAM: u32 = 1;
const SORT_KEY_BLOCKS: u32 = 2;
const SORT_KEY_MTIME: u32 = 3;
const SORT_KEY_FREE_BLOCKS: u32 = 5;
const SORT_ORDER_DESCENT: u32 = 1;

const MOUNT_MODE_OPEN_MASK: u32 = 3;
const MOUNT_MODE_CREATE: u32 = 4;
const MOUNT_MODE_CREATE2: u32 = 32;

const UMOUNT_MODE_BACKUP_ASYNC: u32 = 1 << 16;
const COMMIT_MODE_BACKUP_ASYNC: u32 = 1;

const EVENT_TYPE_UMOUNT_BACKUP_END: u32 = 1;
const EVENT_TYPE_BACKUP_END: u32 = 2;
const EVENT_TYPE_MEMORY_SYNC_END: u32 = 3;
const EVENT_TYPE_COMMIT_BACKUP_END: u32 = 4;
const EVENT_QUEUE_MAX: usize = 20;

const MEMORY_MAX_SIZE: usize = 32 * 1024 * 1024;
const MEMORY_SLOTS: u32 = 4;
const MEMORY_SET_PARAM: u32 = 1;
const MEMORY_DOUBLE_BUFFER: u32 = 2;
const MEMORY_SET_MAX_DATA: u32 = 5;
const MEMORY_DEFAULT_TITLE: &str = "Saved Data";

/// Cap on an icon read from or written to the host (not a platform limit).
const ICON_MAX_SIZE: u64 = 16 * 1024 * 1024;

const UNKNOWN_TITLE: &str = "UNKNOWN";

// ---------------------------------------------------------------------------
// Host storage
// ---------------------------------------------------------------------------

/// Root of all saves: the override (or `$RPS5E_SAVEDATA_DIR`), else
/// `<data dir>/savedata`.
pub fn storage_root(override_dir: Option<&str>) -> PathBuf {
    let configured = override_dir
        .map(str::to_owned)
        .or_else(|| std::env::var("RPS5E_SAVEDATA_DIR").ok())
        .filter(|d| !d.trim().is_empty());
    let root = match configured {
        Some(dir) => PathBuf::from(dir),
        None => crate::filesystem::user_data_dir().join("savedata"),
    };
    std::path::absolute(&root).unwrap_or(root)
}

/// A non-empty name made of `A-Z a-z 0-9 _ - . @` other than `.` / `..`:
/// safe as one host path segment.
fn valid_path_component(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b'@'))
}

/// The NUL-terminated prefix of a fixed field; `None` if it has no NUL.
fn c_str(field: &[u8]) -> Option<&[u8]> {
    field.iter().position(|&b| b == 0).map(|n| &field[..n])
}

/// A fixed field holding a valid path component.
fn component(field: &[u8]) -> Option<String> {
    let s = std::str::from_utf8(c_str(field)?).ok()?;
    valid_path_component(s).then(|| s.to_string())
}

fn memory_dir_name(slot_id: u32) -> String {
    if slot_id == 0 {
        "sce_sdmemory".to_string()
    } else {
        format!("sce_sdmemory{slot_id}")
    }
}

fn is_memory_dir_name(name: &str) -> bool {
    (0..MEMORY_SLOTS).any(|slot| name == memory_dir_name(slot))
}

/// Replace `path` with `data` through a flushed temporary file of its own
/// (`<path>.tmp.<random>`, created exclusively so nothing else in the
/// directory is reused or truncated).
fn write_save_file(path: &Path, data: &[u8]) -> HleResult {
    const ATTEMPTS: usize = 128;
    let mut created = None;
    for _ in 0..ATTEMPTS {
        let mut suffix = [0u8; 8];
        crate::misc_services::fill_random(&mut suffix);
        let mut name = path.as_os_str().to_owned();
        name.push(format!(".tmp.{:016x}", u64::from_le_bytes(suffix)));
        let candidate = PathBuf::from(name);
        match fs::OpenOptions::new().write(true).create_new(true).open(&candidate) {
            Ok(f) => {
                created = Some((candidate, f));
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(_) => return ERR_INTERNAL,
        }
    }
    let Some((temporary, mut file)) = created else {
        return ERR_INTERNAL;
    };
    let written = file.write_all(data).and_then(|()| file.sync_all());
    drop(file);
    if written.is_ok() && fs::rename(&temporary, path).is_ok() {
        return OK;
    }
    let _ = fs::remove_file(&temporary);
    ERR_INTERNAL
}

fn read_save_blocks(directory: &Path) -> Result<u64, HleResult> {
    let path = directory.join("sce_sys").join("blocks.bin");
    let Ok(mut file) = fs::File::open(path) else {
        return Err(ERR_BROKEN);
    };
    if file.metadata().map(|m| m.len()).unwrap_or(0) != 8 {
        return Err(ERR_BROKEN);
    }
    let mut value = [0u8; 8];
    if file.read_exact(&mut value).is_err() {
        return Err(ERR_BROKEN);
    }
    let value = u64::from_le_bytes(value);
    if !(BLOCKS_MIN..=BLOCKS_MAX).contains(&value) {
        return Err(ERR_BROKEN);
    }
    Ok(value)
}

fn read_save_param(path: &Path) -> Result<Param, HleResult> {
    match path.try_exists() {
        Err(_) => return Err(ERR_INTERNAL),
        Ok(false) => return Err(ERR_NOT_FOUND),
        Ok(true) => {}
    }
    let Ok(mut file) = fs::File::open(path) else {
        return Err(ERR_INTERNAL);
    };
    if file.metadata().map(|m| m.len()).unwrap_or(0) != PARAM_SIZE as u64 {
        return Err(ERR_BROKEN);
    }
    let mut param = [0u8; PARAM_SIZE];
    file.read_exact(&mut param).map_err(|_| ERR_BROKEN)?;
    Ok(param)
}

fn save_param_path(directory: &Path) -> PathBuf {
    directory.join("sce_sys").join("param.bin")
}

fn icon_path(directory: &Path) -> PathBuf {
    directory.join("sce_sys").join("icon0.png")
}

fn unix_secs(t: SystemTime) -> i64 {
    match t.duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_secs() as i64,
        Err(e) => -(e.duration().as_secs() as i64),
    }
}

/// Newest modification time (unix seconds) of any file below `directory`.
fn newest_save_time(directory: &Path) -> i64 {
    let mut newest = 0i64;
    let mut pending = vec![directory.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let Ok(ty) = entry.file_type() else { continue };
            if ty.is_dir() {
                pending.push(entry.path());
                continue;
            }
            if let Ok(meta) = fs::metadata(entry.path()) {
                if meta.is_file() {
                    if let Ok(modified) = meta.modified() {
                        newest = newest.max(unix_secs(modified));
                    }
                }
            }
        }
    }
    newest
}

/// The slot's param.bin with `mtime` set to its newest file time.
fn load_save_param(directory: &Path) -> Result<Param, HleResult> {
    let mut param = read_save_param(&save_param_path(directory))?;
    param[PARAM_MTIME..PARAM_MTIME + 8].copy_from_slice(&newest_save_time(directory).to_le_bytes());
    Ok(param)
}

/// Build a SceSaveDataParam (mtime 0). Text is truncated on a character
/// boundary, always leaving the terminator.
fn make_param(title: &str, sub_title: &str, detail: &str, user_param: u32) -> Param {
    fn put(dst: &mut [u8], value: &str) {
        let mut end = value.len().min(dst.len() - 1);
        while !value.is_char_boundary(end) {
            end -= 1;
        }
        dst[..end].copy_from_slice(&value.as_bytes()[..end]);
    }
    let mut p = [0u8; PARAM_SIZE];
    put(&mut p[PARAM_TITLE], title);
    put(&mut p[PARAM_SUB_TITLE], sub_title);
    put(&mut p[PARAM_DETAIL], detail);
    p[PARAM_USER_PARAM..PARAM_USER_PARAM + 4].copy_from_slice(&user_param.to_le_bytes());
    p
}

fn param_text_field(param_type: u32) -> Option<std::ops::Range<usize>> {
    match param_type {
        1 => Some(PARAM_TITLE),
        2 => Some(PARAM_SUB_TITLE),
        3 => Some(PARAM_DETAIL),
        _ => None,
    }
}

/// `%` matches any run, `_` any one character; everything else exactly.
fn dir_name_match(s: &[u8], pattern: &[u8]) -> bool {
    let (mut s, mut pattern) = (s, pattern);
    while let (Some(&c), Some(&p)) = (s.first(), pattern.first()) {
        if p == b'%' {
            return (0..=s.len()).any(|i| dir_name_match(&s[i..], &pattern[1..]));
        }
        if p != b'_' && p != c {
            return false;
        }
        s = &s[1..];
        pattern = &pattern[1..];
    }
    // A pattern ending in '%' also matches the empty rest.
    s.is_empty() && pattern.iter().all(|&p| p == b'%')
}

/// Bytes of one PS5 save data block.
const BLOCK_BYTES: u64 = 64 * 1024;

/// Blocks the files of a save use (each file rounded up to whole blocks);
/// the `sce_sys` metadata of the emulator is not counted.
fn used_blocks(directory: &Path) -> u64 {
    fn walk(dir: &Path, top: bool) -> u64 {
        let Ok(entries) = fs::read_dir(dir) else { return 0 };
        let mut blocks = 0;
        for entry in entries.flatten() {
            let Ok(kind) = entry.file_type() else { continue };
            if kind.is_dir() {
                if !(top && entry.file_name() == "sce_sys") {
                    blocks += walk(&entry.path(), false);
                }
            } else if kind.is_file() {
                blocks += entry.metadata().map_or(0, |m| m.len().div_ceil(BLOCK_BYTES));
            }
        }
        blocks
    }
    walk(directory, true)
}

/// (blocks, free blocks) of a save.
fn save_blocks_info(directory: &Path) -> Result<(u64, u64), HleResult> {
    let blocks = read_save_blocks(directory)?;
    Ok((blocks, blocks - blocks.min(used_blocks(directory))))
}

// ---------------------------------------------------------------------------
// Guest memory helpers
// ---------------------------------------------------------------------------

fn read_i32(ctx: &CpuContext, addr: u64) -> Option<i32> {
    ctx.read_u32(addr).map(|v| v as i32)
}

fn read_array<const N: usize>(ctx: &CpuContext, addr: u64) -> Option<[u8; N]> {
    if addr == 0 {
        return None;
    }
    ctx.read_bytes(addr, N)?.try_into().ok()
}

fn read_param(ctx: &CpuContext, addr: u64) -> Option<Param> {
    read_array::<PARAM_SIZE>(ctx, addr)
}

/// A mount point field that is NUL-terminated within its 16 bytes.
fn read_mount_point(ctx: &CpuContext, addr: u64) -> Option<String> {
    let raw = read_array::<MOUNT_POINT_SIZE>(ctx, addr)?;
    c_str(&raw).map(|s| String::from_utf8_lossy(s).into_owned())
}

/// `snprintf(dst, size, "%s", value)`: copy with truncation plus terminator.
fn write_c_string(ctx: &CpuContext, addr: u64, size: usize, value: &str) -> bool {
    let n = value.len().min(size - 1);
    let mut bytes = value.as_bytes()[..n].to_vec();
    bytes.push(0);
    ctx.write_bytes(addr, &bytes)
}

// ---------------------------------------------------------------------------
// Library state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Event {
    ty: u32,
    error_code: i32,
    user_id: i32,
    title_id: [u8; TITLE_ID_SIZE],
    dir_name: [u8; DIR_NAME_SIZE],
}

impl Event {
    fn to_bytes(&self) -> [u8; EVENT_SIZE] {
        let mut b = [0u8; EVENT_SIZE];
        b[0x00..0x04].copy_from_slice(&self.ty.to_le_bytes());
        b[0x04..0x08].copy_from_slice(&self.error_code.to_le_bytes());
        b[0x08..0x0C].copy_from_slice(&self.user_id.to_le_bytes());
        b[0x10..0x20].copy_from_slice(&self.title_id);
        b[0x20..0x40].copy_from_slice(&self.dir_name);
        b
    }
}

#[derive(Clone)]
struct Memory {
    data: Vec<u8>,
    param: Param,
    option: u32,
}

impl Memory {
    fn footprint(&self) -> usize {
        self.data.len() * if self.option & MEMORY_DOUBLE_BUFFER != 0 { 2 } else { 1 }
    }
}

/// SceSaveDataMemoryData read from the guest.
#[derive(Clone, Copy)]
struct MemoryData {
    buf: u64,
    buf_size: u64,
    offset: i64,
}

impl MemoryData {
    fn read(ctx: &CpuContext, addr: u64) -> Option<Self> {
        Some(Self { buf: ctx.read_u64(addr)?, buf_size: ctx.read_u64(addr + 8)?, offset: ctx.read_u64(addr + 0x10)? as i64 })
    }

    fn valid_for(&self, size: usize) -> bool {
        let size = size as u64;
        self.buf != 0 && self.offset >= 0 && (self.offset as u64) <= size && self.buf_size <= size - self.offset as u64
    }
}

fn save_memory(directory: &Path, memory: &Memory) -> HleResult {
    if fs::create_dir_all(directory).is_err() {
        return ERR_INTERNAL;
    }
    let result = write_save_file(&directory.join("param.bin"), &memory.param);
    if result != OK {
        return result;
    }
    write_save_file(&directory.join("memory.dat"), &memory.data)
}

/// Load `memory.dat` (+ `param.bin`) into `memory`; returns the stored size
/// (0 if nothing was saved yet).
fn load_memory(directory: &Path, memory: &mut Memory) -> Result<usize, HleResult> {
    let path = directory.join("memory.dat");
    match path.try_exists() {
        Err(_) => return Err(ERR_INTERNAL),
        Ok(false) => return Ok(0),
        Ok(true) => {}
    }
    let Ok(mut file) = fs::File::open(&path) else {
        return Err(ERR_INTERNAL);
    };
    let existed = file.metadata().map(|m| m.len()).unwrap_or(0) as usize;
    if existed == 0 || existed > MEMORY_MAX_SIZE {
        return Err(ERR_BROKEN);
    }
    let size = existed.min(memory.data.len());
    file.read_exact(&mut memory.data[..size]).map_err(|_| ERR_BROKEN)?;
    memory.param = read_save_param(&directory.join("param.bin"))?;
    Ok(existed)
}

pub(crate) struct SaveDataLib {
    /// Save root; `None` = [`storage_root`] at the time of use.
    root: Option<PathBuf>,
    /// Running title, if configured.
    title_id: Option<String>,
    slots: MountSlots,
    memory: BTreeMap<PathBuf, Memory>,
    events: VecDeque<Event>,
    next_transaction_resource: i32,
    /// Register mounts with the guest file system.
    register_fs: bool,
    #[cfg(test)]
    last_mounted: Option<PathBuf>,
}

impl SaveDataLib {
    const fn new(root: Option<PathBuf>, register_fs: bool) -> Self {
        Self {
            root,
            title_id: None,
            slots: MountSlots::new(),
            memory: BTreeMap::new(),
            events: VecDeque::new(),
            next_transaction_resource: 1,
            register_fs,
            #[cfg(test)]
            last_mounted: None,
        }
    }

    fn root(&self) -> PathBuf {
        self.root.clone().unwrap_or_else(|| storage_root(None))
    }

    /// Title of the running application: the configured one, else the first
    /// `-`-separated part of the `$RPS5E_APP0_DIR` directory name, else
    /// "UNKNOWN".
    fn title_id(&self) -> String {
        if let Some(t) = self.title_id.as_deref().filter(|t| valid_path_component(t)) {
            return t.to_string();
        }
        if self.root.is_none() {
            if let Ok(app0) = std::env::var("RPS5E_APP0_DIR") {
                let name = Path::new(app0.trim_end_matches('/'))
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                if let Some(candidate) = name.split('-').find(|s| valid_path_component(s)) {
                    return candidate.to_string();
                }
            }
        }
        UNKNOWN_TITLE.to_string()
    }

    /// `<root>/<title>`.
    fn title_dir(&self, title: &str) -> PathBuf {
        self.root().join(title)
    }

    /// The host directory of a save; SaveDataMemory directories are per user.
    fn save_directory(&mut self, title: &str, dir_name: &str, user_id: i32) -> PathBuf {
        let directory = self.title_dir(title).join(dir_name);
        if is_memory_dir_name(dir_name) {
            directory.join(user_id.to_string())
        } else {
            directory
        }
    }

    fn memory_directory(&mut self, user_id: i32, slot_id: u32) -> PathBuf {
        let title = self.title_id();
        self.save_directory(&title, &memory_dir_name(slot_id), user_id)
    }

    fn queue_event(&mut self, ty: u32, user_id: i32, title_id: Option<[u8; TITLE_ID_SIZE]>, dir_name: Option<[u8; DIR_NAME_SIZE]>, error_code: i32) {
        self.events.push_back(Event {
            ty,
            error_code,
            user_id,
            title_id: title_id.unwrap_or([0; TITLE_ID_SIZE]),
            dir_name: dir_name.unwrap_or([0; DIR_NAME_SIZE]),
        });
        if self.events.len() > EVENT_QUEUE_MAX {
            self.events.pop_front();
        }
        log::debug!("savedata: event type={ty} user={user_id} err={error_code:#x}");
    }

    /// Unmount everything and drop all state (new application).
    fn reset(&mut self, title_id: Option<&str>) {
        let mounted: Vec<usize> = self.slots.mounted().map(|(i, _)| i).collect();
        for slot in mounted {
            if self.register_fs {
                crate::filesystem::unregister_guest_path_mount(&MountSlots::mount_point(slot));
            }
            self.slots.release(slot);
        }
        self.title_id = title_id.map(|t| t.trim().to_string()).filter(|t| !t.is_empty());
        self.memory.clear();
        self.events.clear();
        self.next_transaction_resource = 1;
    }

    // -- lifecycle ----------------------------------------------------------

    fn terminate(&mut self) -> HleResult {
        if !self.slots.is_empty() {
            return ERR_BUSY;
        }
        self.events.clear();
        self.memory.clear();
        OK
    }

    fn create_transaction_resource(&mut self, size: u32) -> HleResult {
        let resource = self.next_transaction_resource;
        self.next_transaction_resource += 1;
        log::debug!("savedata: create_transaction_resource size={size:#x} -> {resource}");
        resource as HleResult
    }

    // -- mount --------------------------------------------------------------

    fn attach(&mut self, slot: usize, directory: &Path, status: u32, ctx: &CpuContext, result: u64) -> HleResult {
        let point = MountSlots::mount_point(slot);
        if self.register_fs && !crate::filesystem::register_guest_path_mount(&point, directory) {
            log::error!("savedata: cannot mount {} at {point}", directory.display());
        }
        self.slots.mount(slot, directory);
        #[cfg(test)]
        {
            self.last_mounted = Some(directory.to_path_buf());
        }
        let mut out = [0u8; MOUNT_RESULT_SIZE];
        out[..point.len()].copy_from_slice(point.as_bytes());
        out[0x1C..0x20].copy_from_slice(&status.to_le_bytes());
        ctx.write_bytes(result, &out);
        log::info!("savedata: mount {} -> {point} status={status}", directory.display());
        OK
    }

    /// Mount `dir_name` of the running title. `dir_ptr` points at a
    /// SceSaveDataDirName; `result` at a SceSaveDataMountResult.
    fn mount(&mut self, ctx: &CpuContext, user_id: i32, dir_ptr: u64, blocks: u64, mode: u32, result: u64) -> HleResult {
        if result == 0 || dir_ptr == 0 {
            return ERR_PARAMETER;
        }
        let Some(dir_name) = read_array::<DIR_NAME_SIZE>(ctx, dir_ptr).and_then(|f| component(&f)) else {
            return ERR_PARAMETER;
        };
        if user_id < 0 {
            return ERR_INVALID_LOGIN_USER;
        }
        // The result is written by `attach`; a failed mount leaves it alone.

        let title = self.title_id();
        let mount_dir = self.save_directory(&title, &dir_name, user_id);
        let create = mode & MOUNT_MODE_CREATE != 0;
        let create2 = mode & MOUNT_MODE_CREATE2 != 0;
        let open = !create && !create2 && mode & MOUNT_MODE_OPEN_MASK != 0;

        let slot = match self.slots.find_available(&mount_dir) {
            Available::Busy => return ERR_BUSY,
            Available::Full => return ERR_MOUNT_FULL,
            Available::Slot(s) => s,
        };
        if !create && !create2 && !open {
            log::error!("savedata: unknown mount mode {mode:#x}");
            return ERR_PARAMETER;
        }
        let exists = mount_dir.is_dir();
        if open && !exists {
            return ERR_NOT_FOUND;
        }
        if create && exists {
            return ERR_EXISTS;
        }
        let mut created = false;
        if (create || create2) && !exists {
            if !(BLOCKS_MIN..=BLOCKS_MAX).contains(&blocks) {
                return ERR_PARAMETER;
            }
            let metadata = mount_dir.join("sce_sys");
            if fs::create_dir_all(&metadata).is_err() {
                return ERR_INTERNAL;
            }
            for (path, data) in [
                (metadata.join("blocks.bin"), blocks.to_le_bytes().to_vec()),
                (save_param_path(&mount_dir), vec![0u8; PARAM_SIZE]),
            ] {
                let status = write_save_file(&path, &data);
                if status != OK {
                    let _ = fs::remove_dir_all(&mount_dir);
                    return status;
                }
            }
            created = true;
        }
        self.attach(slot, &mount_dir, u32::from(created), ctx, result)
    }

    /// SceSaveDataMount3: userId@0, dirName*@8, blocks@0x10,
    /// systemBlocks@0x18, mountMode@0x20, resource@0x28.
    fn mount3(&mut self, ctx: &CpuContext, mount: u64, result: u64) -> HleResult {
        if mount == 0 {
            return ERR_PARAMETER;
        }
        let fields = (|| Some((read_i32(ctx, mount)?, ctx.read_u64(mount + 8)?, ctx.read_u64(mount + 0x10)?, ctx.read_u32(mount + 0x20)?)))();
        let Some((user_id, dir_ptr, blocks, mode)) = fields else {
            return ERR_PARAMETER;
        };
        self.mount(ctx, user_id, dir_ptr, blocks, mode, result)
    }

    /// SceSaveDataMount2: userId@0, dirName*@8, blocks@0x10, mountMode@0x18.
    fn mount2(&mut self, ctx: &CpuContext, mount: u64, result: u64) -> HleResult {
        if mount == 0 {
            return ERR_PARAMETER;
        }
        let fields = (|| Some((read_i32(ctx, mount)?, ctx.read_u64(mount + 8)?, ctx.read_u64(mount + 0x10)?, ctx.read_u32(mount + 0x18)?)))();
        let Some((user_id, dir_ptr, blocks, mode)) = fields else {
            return ERR_PARAMETER;
        };
        self.mount(ctx, user_id, dir_ptr, blocks, mode, result)
    }

    /// SceSaveDataMount: userId@0, titleId*@8, dirName*@0x10,
    /// fingerprint*@0x18, blocks@0x20, mountMode@0x28.
    fn mount1(&mut self, ctx: &CpuContext, mount: u64, result: u64) -> HleResult {
        if mount == 0 {
            return ERR_PARAMETER;
        }
        let fields = (|| Some((read_i32(ctx, mount)?, ctx.read_u64(mount + 0x10)?, ctx.read_u64(mount + 0x20)?, ctx.read_u32(mount + 0x28)?)))();
        let Some((user_id, dir_ptr, blocks, mode)) = fields else {
            return ERR_PARAMETER;
        };
        self.mount(ctx, user_id, dir_ptr, blocks, mode, result)
    }

    /// SceSaveDataTransferringMount: userId@0, titleId*@8, dirName*@0x10,
    /// fingerprint*@0x18. Mounts another title's existing save.
    fn transferring_mount(&mut self, ctx: &CpuContext, mount: u64, result: u64) -> HleResult {
        if mount == 0 || result == 0 {
            return ERR_PARAMETER;
        }
        let fields = (|| Some((read_i32(ctx, mount)?, ctx.read_u64(mount + 8)?, ctx.read_u64(mount + 0x10)?)))();
        let Some((user_id, title_ptr, dir_ptr)) = fields else {
            return ERR_PARAMETER;
        };
        let title = read_array::<TITLE_ID_DATA>(ctx, title_ptr).and_then(|f| component(&f));
        let dir_name = read_array::<DIR_NAME_SIZE>(ctx, dir_ptr).and_then(|f| component(&f));
        let (Some(title), Some(dir_name)) = (title, dir_name) else {
            return ERR_PARAMETER;
        };
        if user_id < 0 {
            return ERR_INVALID_LOGIN_USER;
        }
        let mount_dir = self.save_directory(&title, &dir_name, user_id);
        let slot = match self.slots.find_available(&mount_dir) {
            Available::Busy => return ERR_BUSY,
            Available::Full => return ERR_MOUNT_FULL,
            Available::Slot(s) => s,
        };
        if !mount_dir.is_dir() {
            return ERR_NOT_FOUND;
        }
        self.attach(slot, &mount_dir, 0, ctx, result)
    }

    fn umount(&mut self, ctx: &CpuContext, mode: u32, mount_point: u64) -> HleResult {
        let Some(raw) = read_array::<MOUNT_POINT_SIZE>(ctx, mount_point) else {
            return ERR_PARAMETER;
        };
        let point = String::from_utf8_lossy(c_str(&raw).unwrap_or(&raw)).into_owned();
        let Some(slot) = self.slots.find(&point) else {
            return ERR_NOT_FOUND;
        };
        if mode & UMOUNT_MODE_BACKUP_ASYNC != 0 {
            self.queue_event(EVENT_TYPE_UMOUNT_BACKUP_END, 0, None, None, 0);
        }
        if self.register_fs {
            crate::filesystem::unregister_guest_path_mount(&point);
        }
        self.slots.release(slot);
        log::info!("savedata: umount {point} mode={mode:#x}");
        OK
    }

    /// Slot of a mounted, NUL-terminated mount point field.
    fn mounted_slot(&self, ctx: &CpuContext, mount_point: u64) -> Result<usize, HleResult> {
        let point = read_mount_point(ctx, mount_point).ok_or(ERR_PARAMETER)?;
        self.slots.find(&point).ok_or(ERR_NOT_MOUNTED)
    }

    fn get_mount_info(&mut self, ctx: &CpuContext, mount_point: u64, info: u64) -> HleResult {
        if info == 0 {
            return ERR_PARAMETER;
        }
        let slot = match self.mounted_slot(ctx, mount_point) {
            Ok(s) => s,
            Err(e) => return e,
        };
        let (blocks, free) = match save_blocks_info(self.slots.directory(slot)) {
            Ok(b) => b,
            Err(e) => return e,
        };
        let mut out = [0u8; BLOCKS_INFO_SIZE];
        out[0..8].copy_from_slice(&blocks.to_le_bytes());
        out[8..16].copy_from_slice(&free.to_le_bytes());
        ctx.write_bytes(info, &out);
        OK
    }

    // -- params / icons -----------------------------------------------------

    /// The whole fixed field of `param_type` (all of the param, a 128- or
    /// 1024-byte text field, the user param or the mtime); a buffer smaller
    /// than the field is refused.
    fn get_param(&mut self, ctx: &CpuContext, mount_point: u64, param_type: u32, buf: u64, buf_size: u64, got_size: u64) -> HleResult {
        if mount_point == 0 || buf == 0 {
            return ERR_PARAMETER;
        }
        let slot = match self.mounted_slot(ctx, mount_point) {
            Ok(s) => s,
            Err(e) => return e,
        };
        let field = match param_type {
            0 => 0..PARAM_SIZE,
            1..=3 => param_text_field(param_type).unwrap(),
            4 => PARAM_USER_PARAM..PARAM_USER_PARAM + 4,
            5 => PARAM_MTIME..PARAM_MTIME + 8,
            _ => return ERR_PARAMETER,
        };
        if buf_size < field.len() as u64 {
            return ERR_PARAMETER;
        }
        let param = match load_save_param(self.slots.directory(slot)) {
            Ok(p) => p,
            Err(e) => return e,
        };
        let out = param[field].to_vec();
        ctx.write_bytes(buf, &out);
        if got_size != 0 {
            ctx.write_u64(got_size, out.len() as u64);
        }
        OK
    }

    fn set_param(&mut self, ctx: &CpuContext, mount_point: u64, param_type: u32, buf: u64, buf_size: u64) -> HleResult {
        let point_ok = read_mount_point(ctx, mount_point).is_some();
        if !point_ok
            || buf == 0
            || param_type > 4
            || (param_type == 0 && buf_size < PARAM_SIZE as u64)
            || (param_type == 4 && buf_size < 4)
            || ((1..=3).contains(&param_type) && buf_size == 0)
        {
            return ERR_PARAMETER;
        }
        let slot = match self.mounted_slot(ctx, mount_point) {
            Ok(s) => s,
            Err(e) => return e,
        };
        let path = save_param_path(self.slots.directory(slot));
        if param_type == 0 {
            let Some(p) = read_param(ctx, buf) else {
                return ERR_PARAMETER;
            };
            return write_save_file(&path, &p);
        }
        let mut param = match read_save_param(&path) {
            Ok(p) => p,
            Err(e) => return e,
        };
        if param_type == 4 {
            let Some(v) = ctx.read_u32(buf) else {
                return ERR_PARAMETER;
            };
            param[PARAM_USER_PARAM..PARAM_USER_PARAM + 4].copy_from_slice(&v.to_le_bytes());
        } else {
            let range = param_text_field(param_type).unwrap();
            let length = (buf_size as usize).min(range.len() - 1);
            let Some(text) = ctx.read_bytes(buf, length) else {
                return ERR_PARAMETER;
            };
            let count = c_str(&text).map_or(length, <[u8]>::len);
            let field = &mut param[range];
            field.fill(0);
            field[..count].copy_from_slice(&text[..count]);
        }
        write_save_file(&path, &param)
    }

    fn mounted_dir(&self, ctx: &CpuContext, mount_point: u64) -> Option<PathBuf> {
        let point = read_mount_point(ctx, mount_point)?;
        self.slots.find(&point).map(|s| self.slots.directory(s).to_path_buf())
    }

    /// Copy the slot's `sce_sys/icon0.png` into the icon buffer; dataSize
    /// is 0 when there is none.
    fn load_icon(&mut self, ctx: &CpuContext, mount_point: u64, icon: u64) -> HleResult {
        if mount_point == 0 || icon == 0 {
            return ERR_PARAMETER;
        }
        let (Some(buf), Some(buf_size)) = (ctx.read_u64(icon), ctx.read_u64(icon + 8)) else {
            return ERR_PARAMETER;
        };
        let mut size = 0u64;
        if let Some(data) = self.mounted_dir(ctx, mount_point).and_then(|d| fs::read(icon_path(&d)).ok()) {
            if buf != 0 && data.len() as u64 <= buf_size && ctx.write_bytes(buf, &data) {
                size = data.len() as u64;
            }
        }
        ctx.write_u64(icon + ICON_DATA_SIZE_OFFSET, size);
        OK
    }

    /// Store `dataSize` bytes of the icon buffer as `sce_sys/icon0.png`.
    fn save_icon(&mut self, ctx: &CpuContext, mount_point: u64, icon: u64) -> HleResult {
        if mount_point == 0 || icon == 0 {
            return ERR_PARAMETER;
        }
        let (Some(buf), Some(data_size)) = (ctx.read_u64(icon), ctx.read_u64(icon + ICON_DATA_SIZE_OFFSET)) else {
            return ERR_PARAMETER;
        };
        let Some(dir) = self.mounted_dir(ctx, mount_point) else {
            return OK;
        };
        if buf == 0 || data_size == 0 || data_size > ICON_MAX_SIZE {
            return OK;
        }
        let Some(bytes) = ctx.read_bytes(buf, data_size as usize) else {
            return ERR_PARAMETER;
        };
        if fs::create_dir_all(dir.join("sce_sys")).is_err() {
            return ERR_INTERNAL;
        }
        write_save_file(&icon_path(&dir), &bytes)
    }

    /// Copy the guest file at `path` to `sce_sys/icon0.png`.
    fn save_icon_by_path(&mut self, ctx: &CpuContext, mount_point: u64, path: u64) -> HleResult {
        if mount_point == 0 || path == 0 {
            return ERR_PARAMETER;
        }
        let Some(guest_path) = ctx.read_cstring(path, 1024) else {
            return ERR_PARAMETER;
        };
        if let (Some(dir), Some(host)) = (self.mounted_dir(ctx, mount_point), crate::filesystem::resolve_guest_path(&guest_path)) {
            match fs::read(&host) {
                Ok(bytes) if bytes.len() as u64 <= ICON_MAX_SIZE => {
                    if fs::create_dir_all(dir.join("sce_sys")).is_ok() {
                        let _ = write_save_file(&icon_path(&dir), &bytes);
                    }
                }
                _ => log::debug!("savedata: icon {guest_path} not readable"),
            }
        }
        OK
    }

    // -- delete / search ----------------------------------------------------

    /// SceSaveDataDelete: userId@0, titleId*@8, dirName*@0x10.
    fn delete(&mut self, ctx: &CpuContext, del: u64) -> HleResult {
        if del == 0 {
            return ERR_PARAMETER;
        }
        let fields = (|| Some((read_i32(ctx, del)?, ctx.read_u64(del + 8)?, ctx.read_u64(del + 0x10)?)))();
        let Some((user_id, title_ptr, dir_ptr)) = fields else {
            return ERR_PARAMETER;
        };
        let Some(dir_name) = read_array::<DIR_NAME_SIZE>(ctx, dir_ptr).and_then(|f| component(&f)) else {
            return ERR_PARAMETER;
        };
        let title = if title_ptr == 0 {
            self.title_id()
        } else {
            match read_array::<TITLE_ID_DATA>(ctx, title_ptr).and_then(|f| component(&f)) {
                Some(t) => t,
                None => return ERR_PARAMETER,
            }
        };
        if user_id < 0 {
            return ERR_INVALID_LOGIN_USER;
        }
        let dir = self.save_directory(&title, &dir_name, user_id);
        if self.slots.find_available(&dir) == Available::Busy || self.memory.contains_key(&dir) {
            return ERR_BUSY;
        }
        match fs::remove_dir_all(&dir) {
            Ok(()) => {
                log::info!("savedata: delete {}", dir.display());
                OK
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => OK,
            Err(_) => ERR_INTERNAL,
        }
    }

    /// SceSaveDataDirNameSearchCond: userId@0, titleId*@8, dirName*@0x10,
    /// key@0x18, order@0x1C.
    fn dir_name_search(&mut self, ctx: &CpuContext, cond: u64, result: u64) -> HleResult {
        if cond == 0 || result == 0 {
            return ERR_PARAMETER;
        }
        let c = (|| Some((read_i32(ctx, cond)?, ctx.read_u64(cond + 8)?, ctx.read_u64(cond + 0x10)?, ctx.read_u32(cond + 0x18)?, ctx.read_u32(cond + 0x1C)?)))();
        let r = (|| Some((ctx.read_u64(result + 8)?, ctx.read_u32(result + 0x10)?, ctx.read_u64(result + 0x18)?, ctx.read_u64(result + 0x20)?)))();
        let (Some((user_id, title_ptr, dir_ptr, key, order)), Some((dir_names, dir_names_num, params, infos))) = (c, r) else {
            return ERR_PARAMETER;
        };
        let title = if title_ptr == 0 {
            None
        } else {
            match read_array::<TITLE_ID_DATA>(ctx, title_ptr).and_then(|f| component(&f)) {
                Some(t) => Some(t),
                None => return ERR_PARAMETER,
            }
        };
        if dir_names_num != 0 && dir_names == 0 {
            return ERR_PARAMETER;
        }
        let pattern = if dir_ptr == 0 {
            Vec::new()
        } else {
            match read_array::<DIR_NAME_SIZE>(ctx, dir_ptr).as_ref().and_then(|f| c_str(f)) {
                Some(p) => p.to_ascii_lowercase(),
                None => return ERR_PARAMETER,
            }
        };
        if (key > SORT_KEY_MTIME && key != SORT_KEY_FREE_BLOCKS) || order > SORT_ORDER_DESCENT {
            return ERR_PARAMETER;
        }
        if user_id < 0 {
            return ERR_INVALID_LOGIN_USER;
        }

        // hitNum, pad, setNum, reserved and the trailing pad start out zero.
        let mut header = ctx.read_bytes(result, SEARCH_RESULT_SIZE).unwrap_or_else(|| vec![0; SEARCH_RESULT_SIZE]);
        header[0x00..0x08].fill(0);
        header[0x14..0x18].fill(0);
        header[0x28..0x38].fill(0);
        ctx.write_bytes(result, &header);

        let title = title.unwrap_or_else(|| self.title_id());
        let root = self.title_dir(&title);
        let mut list: Vec<String> = Vec::new();
        if let Ok(entries) = fs::read_dir(&root) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if !entry.path().is_dir()
                    || name.starts_with("sce_")
                    || !valid_path_component(&name)
                    || !save_param_path(&root.join(&name)).is_file()
                {
                    continue;
                }
                if pattern.is_empty() || dir_name_match(name.to_ascii_lowercase().as_bytes(), &pattern) {
                    list.push(name);
                }
            }
        }
        // By name, then (stable) by the key; a descending order reverses all.
        list.sort();
        let sort_value = |name: &String| -> u64 {
            let directory = root.join(name);
            let param = || load_save_param(&directory).ok();
            match key {
                SORT_KEY_USER_PARAM => param().map_or(0, |p| u32::from_le_bytes(p[PARAM_USER_PARAM..PARAM_USER_PARAM + 4].try_into().unwrap()) as u64),
                SORT_KEY_BLOCKS => read_save_blocks(&directory).unwrap_or(0),
                SORT_KEY_MTIME => param().map_or(0, |p| u64::from_le_bytes(p[PARAM_MTIME..PARAM_MTIME + 8].try_into().unwrap())),
                SORT_KEY_FREE_BLOCKS => save_blocks_info(&directory).map_or(0, |b| b.1),
                _ => 0,
            }
        };
        if key != SORT_KEY_DIR_NAME {
            let mut keyed: Vec<(u64, String)> = list.into_iter().map(|n| (sort_value(&n), n)).collect();
            keyed.sort_by_key(|(v, _)| *v);
            list = keyed.into_iter().map(|(_, n)| n).collect();
        }
        if order == SORT_ORDER_DESCENT {
            list.reverse();
        }
        let set_num = (dir_names_num as usize).min(list.len());
        ctx.write_u32(result, list.len() as u32);
        ctx.write_u32(result + 0x14, set_num as u32);
        log::debug!("savedata: dir_name_search title={title} hits={} set={set_num}", list.len());

        for (i, name) in list.iter().take(set_num).enumerate() {
            let i = i as u64;
            write_c_string(ctx, dir_names + i * DIR_NAME_SIZE as u64, DIR_NAME_SIZE, name);
            let directory = root.join(name);
            if params != 0 {
                let addr = params + i * PARAM_SIZE as u64;
                match load_save_param(&directory) {
                    Ok(p) => ctx.write_bytes(addr, &p),
                    Err(e) => {
                        ctx.write_bytes(addr, &[0u8; PARAM_SIZE]);
                        return e;
                    }
                };
            }
            if infos != 0 {
                let addr = infos + i * BLOCKS_INFO_SIZE as u64;
                let mut info = [0u8; BLOCKS_INFO_SIZE];
                ctx.write_bytes(addr, &info);
                let (blocks, free) = match save_blocks_info(&directory) {
                    Ok(b) => b,
                    Err(e) => return e,
                };
                info[0..8].copy_from_slice(&blocks.to_le_bytes());
                info[8..16].copy_from_slice(&free.to_le_bytes());
                ctx.write_bytes(addr, &info);
            }
        }
        OK
    }

    // -- events / backup / transactions -------------------------------------

    fn get_event_result(&mut self, ctx: &CpuContext, event: u64) -> HleResult {
        if event == 0 {
            return ERR_PARAMETER;
        }
        let Some(ev) = self.events.pop_front() else {
            return ERR_NOT_FOUND;
        };
        ctx.write_bytes(event, &ev.to_bytes());
        OK
    }

    /// SceSaveDataBackup: userId@0, titleId*@8, dirName*@0x10, fingerprint*@0x18.
    fn backup(&mut self, ctx: &CpuContext, backup: u64) -> HleResult {
        if backup == 0 {
            return ERR_PARAMETER;
        }
        let fields = (|| Some((read_i32(ctx, backup)?, ctx.read_u64(backup + 8)?, ctx.read_u64(backup + 0x10)?)))();
        let Some((user_id, title_ptr, dir_ptr)) = fields else {
            return ERR_PARAMETER;
        };
        let Some(dir_name) = read_array::<DIR_NAME_SIZE>(ctx, dir_ptr) else {
            return ERR_PARAMETER;
        };
        let title = read_array::<TITLE_ID_SIZE>(ctx, title_ptr);
        self.queue_event(EVENT_TYPE_BACKUP_END, user_id, title, Some(dir_name), 0);
        OK
    }

    /// SceSaveDataCommitParam: resource@0, commitMode@4.
    fn commit(&mut self, ctx: &CpuContext, param: u64) -> HleResult {
        if param == 0 {
            return ERR_PARAMETER;
        }
        let (Some(resource), Some(mode)) = (read_i32(ctx, param), ctx.read_u32(param + 4)) else {
            return ERR_PARAMETER;
        };
        log::debug!("savedata: commit resource={resource} mode={mode:#x}");
        if mode & COMMIT_MODE_BACKUP_ASYNC != 0 {
            self.queue_event(EVENT_TYPE_COMMIT_BACKUP_END, 0, None, None, 0);
        }
        OK
    }

    // -- SaveDataMemory -----------------------------------------------------

    /// SceSaveDataMemorySetup2: option@0, userId@4, memorySize@8,
    /// iconMemorySize@0x10, initParam*@0x18, initIcon*@0x20, slotId@0x28.
    fn setup_memory2(&mut self, ctx: &CpuContext, setup: u64, result: u64) -> HleResult {
        if setup == 0 {
            return ERR_PARAMETER;
        }
        let fields = (|| {
            Some((ctx.read_u32(setup)?, read_i32(ctx, setup + 4)?, ctx.read_u64(setup + 8)?, ctx.read_u64(setup + 0x18)?, ctx.read_u32(setup + 0x28)?))
        })();
        let Some((option, user_id, size, init_param, slot_id)) = fields else {
            return ERR_PARAMETER;
        };
        let init_param = if init_param == 0 { None } else { Some(read_param(ctx, init_param)) };
        if init_param == Some(None) {
            return ERR_PARAMETER;
        }
        match self.setup_memory(option, user_id, size, init_param.flatten(), slot_id) {
            Ok(existed) => {
                if result != 0 {
                    let mut out = [0u8; MEMORY_SETUP_RESULT_SIZE];
                    out[..8].copy_from_slice(&(existed as u64).to_le_bytes());
                    ctx.write_bytes(result, &out);
                }
                OK
            }
            Err(e) => e,
        }
    }

    fn setup_memory(&mut self, option: u32, user_id: i32, size: u64, init_param: Option<Param>, slot_id: u32) -> Result<usize, HleResult> {
        if slot_id >= MEMORY_SLOTS || size == 0 || size > MEMORY_MAX_SIZE as u64 || option & !3 != 0 {
            return Err(ERR_PARAMETER);
        }
        if user_id < 0 {
            return Err(ERR_INVALID_LOGIN_USER);
        }
        let directory = self.memory_directory(user_id, slot_id);
        if self.memory.contains_key(&directory) || self.slots.find_available(&directory) == Available::Busy {
            return Err(ERR_BUSY);
        }
        let mut memory = Memory { data: vec![0u8; size as usize], param: make_param(MEMORY_DEFAULT_TITLE, "", "", 0), option };
        let total = memory.footprint() + self.memory.values().map(Memory::footprint).sum::<usize>();
        if self.memory.len() >= MEMORY_SLOTS as usize || total > MEMORY_MAX_SIZE {
            return Err(ERR_LIMITATION_OVER);
        }
        let existed = load_memory(&directory, &mut memory)?;
        if existed == 0 && option & MEMORY_SET_PARAM != 0 {
            if let Some(p) = init_param {
                memory.param = p;
                memory.param[PARAM_MTIME..PARAM_MTIME + 8].fill(0);
            }
        }
        log::debug!("savedata: memory setup user={user_id} slot={slot_id} size={size:#x} existed={existed:#x}");
        self.memory.insert(directory, memory);
        Ok(existed)
    }

    /// SceSaveDataMemoryGet2: userId@0, data*@8, param*@0x10, icon*@0x18, slotId@0x20.
    fn get_memory2(&mut self, ctx: &CpuContext, get: u64) -> HleResult {
        if get == 0 {
            return ERR_PARAMETER;
        }
        let fields = (|| Some((read_i32(ctx, get)?, ctx.read_u64(get + 8)?, ctx.read_u64(get + 0x10)?, ctx.read_u64(get + 0x18)?, ctx.read_u32(get + 0x20)?)))();
        let Some((user_id, data, param, icon, slot_id)) = fields else {
            return ERR_PARAMETER;
        };
        if slot_id >= MEMORY_SLOTS {
            return ERR_PARAMETER;
        }
        if user_id < 0 {
            return ERR_INVALID_LOGIN_USER;
        }
        let directory = self.memory_directory(user_id, slot_id);
        let Some(memory) = self.memory.get(&directory) else {
            return ERR_MEMORY_NOT_READY;
        };
        let data = if data == 0 {
            None
        } else {
            match MemoryData::read(ctx, data) {
                Some(d) => Some(d),
                None => return ERR_PARAMETER,
            }
        };
        if (param != 0 && memory.option & MEMORY_SET_PARAM == 0) || data.is_some_and(|d| !d.valid_for(memory.data.len())) {
            return ERR_PARAMETER;
        }
        if let Some(d) = data {
            let start = d.offset as usize;
            if !ctx.write_bytes(d.buf, &memory.data[start..start + d.buf_size as usize]) {
                return ERR_PARAMETER;
            }
        }
        if param != 0 {
            ctx.write_bytes(param, &memory.param);
        }
        if icon != 0 {
            ctx.write_u64(icon + ICON_DATA_SIZE_OFFSET, 0);
        }
        OK
    }

    /// SceSaveDataMemorySet2: userId@0, data*@8, param*@0x10, icon*@0x18,
    /// dataNum@0x20, slotId@0x24.
    fn set_memory2(&mut self, ctx: &CpuContext, set: u64) -> HleResult {
        if set == 0 {
            return ERR_PARAMETER;
        }
        let fields = (|| {
            Some((read_i32(ctx, set)?, ctx.read_u64(set + 8)?, ctx.read_u64(set + 0x10)?, ctx.read_u32(set + 0x20)?, ctx.read_u32(set + 0x24)?))
        })();
        let Some((user_id, data, param, data_num, slot_id)) = fields else {
            return ERR_PARAMETER;
        };
        if slot_id >= MEMORY_SLOTS || data_num > MEMORY_SET_MAX_DATA || (data == 0 && data_num != 0) {
            return ERR_PARAMETER;
        }
        if user_id < 0 {
            return ERR_INVALID_LOGIN_USER;
        }
        let directory = self.memory_directory(user_id, slot_id);
        let Some(current) = self.memory.get(&directory) else {
            return ERR_MEMORY_NOT_READY;
        };
        if param != 0 && current.option & MEMORY_SET_PARAM == 0 {
            return ERR_PARAMETER;
        }
        // Some games supply one descriptor with an implicit zero count.
        let data_num = if data != 0 && data_num == 0 { 1 } else { data_num };
        let mut descriptors = Vec::with_capacity(data_num as usize);
        for i in 0..u64::from(data_num) {
            match MemoryData::read(ctx, data + i * MEMORY_DATA_SIZE) {
                Some(d) if d.valid_for(current.data.len()) => descriptors.push(d),
                _ => return ERR_PARAMETER,
            }
        }
        if descriptors.is_empty() && param == 0 {
            return OK;
        }
        // Save the complete update before publishing it so progress survives process exit.
        let mut updated = current.clone();
        for d in &descriptors {
            let Some(bytes) = ctx.read_bytes(d.buf, d.buf_size as usize) else {
                return ERR_PARAMETER;
            };
            let start = d.offset as usize;
            updated.data[start..start + bytes.len()].copy_from_slice(&bytes);
        }
        if param != 0 {
            let Some(p) = read_param(ctx, param) else {
                return ERR_PARAMETER;
            };
            updated.param = p;
        }
        updated.param[PARAM_MTIME..PARAM_MTIME + 8].copy_from_slice(&unix_secs(SystemTime::now()).to_le_bytes());
        let status = save_memory(&directory, &updated);
        if status == OK {
            self.memory.insert(directory, updated);
        }
        status
    }

    /// SceSaveDataMemorySync: userId@0, slotId@4, option@8.
    fn sync_memory(&mut self, ctx: &CpuContext, sync: u64) -> HleResult {
        if sync == 0 {
            return ERR_PARAMETER;
        }
        let fields = (|| Some((read_i32(ctx, sync)?, ctx.read_u32(sync + 4)?, ctx.read_u32(sync + 8)?)))();
        let Some((user_id, slot_id, option)) = fields else {
            return ERR_PARAMETER;
        };
        if slot_id >= MEMORY_SLOTS || option > 1 {
            return ERR_PARAMETER;
        }
        if user_id < 0 {
            return ERR_INVALID_LOGIN_USER;
        }
        let directory = self.memory_directory(user_id, slot_id);
        let Some(memory) = self.memory.get(&directory) else {
            return ERR_MEMORY_NOT_READY;
        };
        if self.slots.find_available(&directory) == Available::Busy {
            return ERR_BUSY;
        }
        let status = save_memory(&directory, memory);
        if option == 1 {
            return status;
        }
        let mut title = [0u8; TITLE_ID_SIZE];
        let t = self.title_id();
        let n = t.len().min(TITLE_ID_DATA - 1);
        title[..n].copy_from_slice(&t.as_bytes()[..n]);
        let mut dir = [0u8; DIR_NAME_SIZE];
        let d = memory_dir_name(slot_id);
        dir[..d.len()].copy_from_slice(d.as_bytes());
        self.queue_event(EVENT_TYPE_MEMORY_SYNC_END, user_id, Some(title), Some(dir), status as i32);
        OK
    }

    // -- counts -------------------------------------------------------------

    fn title_save_names(&mut self) -> Vec<PathBuf> {
        let title = self.title_id();
        let root = self.title_dir(&title);
        let Ok(entries) = fs::read_dir(&root) else { return Vec::new() };
        entries
            .flatten()
            .filter(|e| e.path().is_dir() && !e.file_name().to_string_lossy().starts_with("sce_"))
            .map(|e| e.path())
            .collect()
    }
}

fn directory_size(root: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(root) else { return 0 };
    entries
        .flatten()
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => directory_size(&e.path()),
            _ => e.metadata().map(|m| m.len()).unwrap_or(0),
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Global instance
// ---------------------------------------------------------------------------

static LIB: Mutex<SaveDataLib> = Mutex::new(SaveDataLib::new(None, true));

fn lib() -> MutexGuard<'static, SaveDataLib> {
    LIB.lock().unwrap_or_else(|e| e.into_inner())
}

/// Configure the running title (once the title id is known): unmounts all
/// saves and drops SaveDataMemory, events and transaction state.
pub fn configure_application_info(title_id: Option<&str>) {
    lib().reset(title_id);
}

/// Set the title id used for the save directory.
pub fn set_title_id(id: &str) {
    configure_application_info(Some(id));
}

// ---------------------------------------------------------------------------
// HLE entry points
// ---------------------------------------------------------------------------

fn cb_ok(_ctx: &mut CpuContext) -> HleResult {
    OK
}

fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    lib().terminate()
}

fn cb_mount(ctx: &mut CpuContext) -> HleResult {
    lib().mount1(ctx, ctx.rdi(), ctx.rsi())
}

fn cb_mount2(ctx: &mut CpuContext) -> HleResult {
    lib().mount2(ctx, ctx.rdi(), ctx.rsi())
}

fn cb_mount3(ctx: &mut CpuContext) -> HleResult {
    lib().mount3(ctx, ctx.rdi(), ctx.rsi())
}

fn cb_transferring_mount(ctx: &mut CpuContext) -> HleResult {
    lib().transferring_mount(ctx, ctx.rdi(), ctx.rsi())
}

/// rdi: mount point.
fn cb_umount(ctx: &mut CpuContext) -> HleResult {
    lib().umount(ctx, 0, ctx.rdi())
}

/// rdi: mode, rsi: mount point.
fn cb_umount2(ctx: &mut CpuContext) -> HleResult {
    lib().umount(ctx, ctx.rdi() as u32, ctx.rsi())
}

fn cb_is_mounted(ctx: &mut CpuContext) -> HleResult {
    let mounted = !lib().slots.is_empty();
    if ctx.rsi() != 0 {
        ctx.write_u32(ctx.rsi(), u32::from(mounted));
    }
    OK
}

fn cb_get_mount_info(ctx: &mut CpuContext) -> HleResult {
    lib().get_mount_info(ctx, ctx.rdi(), ctx.rsi())
}

fn cb_delete(ctx: &mut CpuContext) -> HleResult {
    lib().delete(ctx, ctx.rdi())
}

/// rdi: mount point, esi: type, rdx: buf, rcx: size, r8: got size.
fn cb_get_param(ctx: &mut CpuContext) -> HleResult {
    lib().get_param(ctx, ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx(), ctx.r8())
}

/// rdi: mount point, esi: type, rdx: buf, rcx: size.
fn cb_set_param(ctx: &mut CpuContext) -> HleResult {
    lib().set_param(ctx, ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx())
}

fn cb_save_icon(ctx: &mut CpuContext) -> HleResult {
    lib().save_icon(ctx, ctx.rdi(), ctx.rsi())
}

fn cb_save_icon_by_path(ctx: &mut CpuContext) -> HleResult {
    lib().save_icon_by_path(ctx, ctx.rdi(), ctx.rsi())
}

fn cb_load_icon(ctx: &mut CpuContext) -> HleResult {
    lib().load_icon(ctx, ctx.rdi(), ctx.rsi())
}

/// rsi: total size of the title's saves in KiB.
fn cb_get_all_size(ctx: &mut CpuContext) -> HleResult {
    let total: u64 = lib().title_save_names().iter().map(|p| directory_size(p)).sum();
    if ctx.rsi() != 0 {
        ctx.write_u64(ctx.rsi(), total.div_ceil(1024));
    }
    OK
}

/// Operations complete synchronously, so progress is always 100%.
fn cb_get_progress(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() != 0 {
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&1.0f32.to_le_bytes());
        ctx.write_bytes(ctx.rdi(), &b);
    }
    OK
}

fn cb_get_save_data_count(ctx: &mut CpuContext) -> HleResult {
    let count = lib().title_save_names().len() as u32;
    if ctx.rsi() != 0 {
        ctx.write_u32(ctx.rsi(), count);
    }
    OK
}

fn cb_get_mounted_save_data_count(ctx: &mut CpuContext) -> HleResult {
    let count = lib().slots.count() as u32;
    if ctx.rsi() != 0 {
        ctx.write_u32(ctx.rsi(), count);
    }
    OK
}

fn cb_dir_name_search(ctx: &mut CpuContext) -> HleResult {
    lib().dir_name_search(ctx, ctx.rdi(), ctx.rsi())
}

fn cb_create_transaction_resource(ctx: &mut CpuContext) -> HleResult {
    lib().create_transaction_resource(ctx.rdi() as u32)
}

/// rdi: mount point, rsi: SceSaveDataPrepareParam {resource@0, prepareMode@4}.
fn cb_prepare(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 || ctx.rsi() == 0 {
        return ERR_PARAMETER;
    }
    OK
}

fn cb_commit(ctx: &mut CpuContext) -> HleResult {
    lib().commit(ctx, ctx.rdi())
}

fn cb_backup(ctx: &mut CpuContext) -> HleResult {
    lib().backup(ctx, ctx.rdi())
}

/// rdi: event param (unused), rsi: SceSaveDataEvent*.
fn cb_get_event_result(ctx: &mut CpuContext) -> HleResult {
    lib().get_event_result(ctx, ctx.rsi())
}

fn cb_setup_memory2(ctx: &mut CpuContext) -> HleResult {
    lib().setup_memory2(ctx, ctx.rdi(), ctx.rsi())
}

fn cb_get_memory2(ctx: &mut CpuContext) -> HleResult {
    lib().get_memory2(ctx, ctx.rdi())
}

fn cb_set_memory2(ctx: &mut CpuContext) -> HleResult {
    lib().set_memory2(ctx, ctx.rdi())
}

fn cb_sync_memory(ctx: &mut CpuContext) -> HleResult {
    lib().sync_memory(ctx, ctx.rdi())
}

/// sceSaveDataSetupSaveDataMemory(userId, memorySize, SceSaveDataParam*):
/// slot 0, parameter support when a param is given.
fn cb_setup_memory(ctx: &mut CpuContext) -> HleResult {
    let param = ctx.rdx();
    let init = if param == 0 {
        None
    } else {
        match read_param(ctx, param) {
            Some(p) => Some(p),
            None => return ERR_PARAMETER,
        }
    };
    let option = if init.is_some() { MEMORY_SET_PARAM } else { 0 };
    match lib().setup_memory(option, ctx.rdi() as i32, ctx.rsi(), init, 0) {
        Ok(_) => OK,
        Err(e) => e,
    }
}

/// sceSaveData{Get,Set}SaveDataMemory(userId, buf, bufSize, offset) on slot 0.
fn transfer_memory_v1(ctx: &CpuContext, write: bool) -> HleResult {
    let (user_id, buf, size, offset) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), ctx.rcx() as i64);
    if user_id < 0 {
        return ERR_INVALID_LOGIN_USER;
    }
    let mut lib = lib();
    let directory = lib.memory_directory(user_id, 0);
    let Some(memory) = lib.memory.get(&directory) else {
        return ERR_MEMORY_NOT_READY;
    };
    let d = MemoryData { buf, buf_size: size, offset };
    if !d.valid_for(memory.data.len()) {
        return ERR_PARAMETER;
    }
    let start = offset as usize;
    if !write {
        return if ctx.write_bytes(buf, &memory.data[start..start + size as usize]) { OK } else { ERR_PARAMETER };
    }
    let Some(bytes) = ctx.read_bytes(buf, size as usize) else {
        return ERR_PARAMETER;
    };
    let mut updated = memory.clone();
    updated.data[start..start + bytes.len()].copy_from_slice(&bytes);
    updated.param[PARAM_MTIME..PARAM_MTIME + 8].copy_from_slice(&unix_secs(SystemTime::now()).to_le_bytes());
    let status = save_memory(&directory, &updated);
    if status == OK {
        lib.memory.insert(directory, updated);
    }
    status
}

fn cb_get_memory(ctx: &mut CpuContext) -> HleResult {
    transfer_memory_v1(ctx, false)
}

fn cb_set_memory(ctx: &mut CpuContext) -> HleResult {
    transfer_memory_v1(ctx, true)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register(reg: &mut HleRegistry) {
    const L: &str = "libSceSaveData";
    reg.register_fn("z1JA8-iJt3k", "sceSaveDataBackup", L, cb_backup);
    reg.register_fn("j8xKtiFj0SY", "sceSaveDataGetEventResult", L, cb_get_event_result);
    reg.register_fn("hsKd5c21sQc", "sceSaveDataRegisterEventCallback", L, cb_ok);
    reg.register_fn("v-AK1AxQhS0", "sceSaveDataUnregisterEventCallback", L, cb_ok);

    reg.register_fn("ZkZhskCPXFw", "sceSaveDataInitialize", L, cb_ok);
    reg.register_fn("l1NmDeDpNGU", "sceSaveDataInitialize2", L, cb_ok);
    reg.register_fn("TywrFKCoLGY", "sceSaveDataInitialize3", L, cb_ok);
    reg.register_fn("yKDy8S5yLA0", "sceSaveDataTerminate", L, cb_terminate);

    reg.register_fn("32HQAQdwM2o", "sceSaveDataMount", L, cb_mount);
    reg.register_fn("0z45PIH+SNI", "sceSaveDataMount2", L, cb_mount2);
    reg.register_fn("ZP4e7rlzOUk", "sceSaveDataMount3", L, cb_mount3);
    reg.register_fn("xz0YMi6BfNk", "sceSaveDataMount5", L, cb_mount3);
    reg.register_fn("WAzWTZm1H+I", "sceSaveDataTransferringMount", L, cb_transferring_mount);
    reg.register_fn("RjMlsR8EXrw", "sceSaveDataTransferringMountPs4", L, cb_transferring_mount);
    reg.register_fn("BMR4F-Uek3E", "sceSaveDataUmount", L, cb_umount);
    reg.register_fn("uW4vfTwMQVo", "sceSaveDataUmount2", L, cb_umount2);
    reg.register_fn("ieP6jP138Qo", "sceSaveDataIsMounted", L, cb_is_mounted);
    reg.register_fn("65VH0Qaaz6s", "sceSaveDataGetMountInfo", L, cb_get_mount_info);

    reg.register_fn("S1GkePI17zQ", "sceSaveDataDelete", L, cb_delete);
    reg.register_fn("SQWusLoK8Pw", "sceSaveDataDelete5", L, cb_delete);

    reg.register_fn("XgvSuIdnMlw", "sceSaveDataGetParam", L, cb_get_param);
    reg.register_fn("85zul--eGXs", "sceSaveDataSetParam", L, cb_set_param);
    reg.register_fn("c88Yy54Mx0w", "sceSaveDataSaveIcon", L, cb_save_icon);
    reg.register_fn("cGjO3wM3V28", "sceSaveDataLoadIcon", L, cb_load_icon);
    reg.register_fn("Z7z6HXWORJY", "sceSaveDataSaveIconByPath", L, cb_save_icon_by_path);

    reg.register_fn("A1ThglSGUwA", "sceSaveDataGetAllSize", L, cb_get_all_size);
    reg.register_fn("ANmSWUiyyGQ", "sceSaveDataGetProgress", L, cb_get_progress);
    reg.register_fn("Wz-4JZfeO9g", "sceSaveDataClearProgress", L, cb_ok);
    reg.register_fn("dQ2GohUHXzk", "sceSaveDataAbort", L, cb_ok);
    reg.register_fn("eBSSNIG6hMk", "sceSaveDataGetEventInfo", L, cb_ok);
    reg.register_fn("52pL2GKkdjA", "sceSaveDataSetEventInfo", L, cb_ok);
    reg.register_fn("SN7rTPHS+Cg", "sceSaveDataGetSaveDataCount", L, cb_get_save_data_count);
    reg.register_fn("pc4guaUPVqA", "sceSaveDataGetMountedSaveDataCount", L, cb_get_mounted_save_data_count);

    reg.register_fn("dyIhnXq-0SM", "sceSaveDataDirNameSearch", L, cb_dir_name_search);
    reg.register_fn("X4MYzukPc3g", "sceSaveDataDirNameSearchPs4", L, cb_dir_name_search);
    reg.register_fn("PHnuI4LhuRk", "sceSaveDataDirNameSearch2", L, cb_dir_name_search);

    reg.register_fn("gjRZNnw0JPE", "sceSaveDataCreateTransactionResource", L, cb_create_transaction_resource);
    reg.register_fn("lJUQuaKqoKY", "sceSaveDataDeleteTransactionResource", L, cb_ok);
    reg.register_fn("sDCBrmc61XU", "sceSaveDataPrepare", L, cb_prepare);
    reg.register_fn("ie7qhZ4X0Cc", "sceSaveDataCommit", L, cb_commit);

    reg.register_fn("v7AAAMo0Lz4", "sceSaveDataSetupSaveDataMemory", L, cb_setup_memory);
    reg.register_fn("7Bt5pBC-Aco", "sceSaveDataGetSaveDataMemory", L, cb_get_memory);
    reg.register_fn("h3YURzXGSVQ", "sceSaveDataSetSaveDataMemory", L, cb_set_memory);
    reg.register_fn("oQySEUfgXRA", "sceSaveDataSetupSaveDataMemory2", L, cb_setup_memory2);
    reg.register_fn("QwOO7vegnV8", "sceSaveDataGetSaveDataMemory2", L, cb_get_memory2);
    reg.register_fn("cduy9v4YmT4", "sceSaveDataSetSaveDataMemory2", L, cb_set_memory2);
    reg.register_fn("wiT9jeC7xPw", "sceSaveDataSyncSaveDataMemory", L, cb_sync_memory);
}
