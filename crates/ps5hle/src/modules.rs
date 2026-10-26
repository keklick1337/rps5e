// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Loaded-module registry and the libkernel module ABI: module load/start,
//! stop/unload, symbol lookup (`sceKernelDlsym`) and module information.
//!
//! The loader registers every mapped image (main executable first, then
//! PRX modules in load order); handles start at 1 and handle 0 names the
//! main executable. The unwinder in libc uses
//! `sceKernelGetModuleInfoForUnwind` to find `.eh_frame` for a PC, so the
//! registry records each image's `PT_GNU_EH_FRAME` header and frame range.

use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};
use std::sync::Mutex;

const OK: HleResult = 0;
const ENOENT: HleResult = 0x8002_0002u32 as i32 as i64;
const ESRCH: HleResult = 0x8002_0003u32 as i32 as i64;
const ENOMEM: HleResult = 0x8002_000Cu32 as i32 as i64;
const EFAULT: HleResult = 0x8002_000Eu32 as i32 as i64;
const EBUSY: HleResult = 0x8002_0010u32 as i32 as i64;
const EINVAL: HleResult = 0x8002_0016u32 as i32 as i64;

const NAME_MAX_BYTES: usize = 256;
const NAME_OFFSET: usize = 0x08;
const MODULE_INFO_SIZE: usize = 0x160;
const MODULE_INFO_SEGMENTS_OFFSET: usize = 0x108;
const MODULE_INFO_SEGMENT_COUNT_OFFSET: usize = 0x148;
const MODULE_INFO_SEGMENTS: usize = 4;
const MODULE_INFO_EX_SIZE: usize = 0x1A8;
const MODULE_INFO_EX_HANDLE_OFFSET: usize = 0x108;
const MODULE_INFO_EX_INIT_PROC_OFFSET: usize = 0x128;
const MODULE_INFO_EX_SEGMENTS_OFFSET: usize = 0x160;
const MODULE_INFO_EX_SEGMENT_COUNT_OFFSET: usize = 0x1A0;
/// `ModuleInfoForUnwind`: size, name[256], eh_frame_hdr, eh_frame,
/// eh_frame size, seg0 address, seg0 size.
const UNWIND_INFO_SIZE: usize = 0x130;
/// Guest executable range; unknown PCs inside it are genuinely unmapped.
const GUEST_EXEC_FLOOR: u64 = 0x0000_0008_0000_0000;
const GUEST_EXEC_CEILING: u64 = 0x0000_0009_0000_0000;
/// Synthetic boundary reported for PCs outside any guest image (host code).
const HOST_UNWIND_BOUNDARY: u64 = 0x10_0000;

/// TCB scratch slots for the sanitizer replace tables.
const TLS_MALLOC_REPLACE_OFFSET: u64 = 0x200;
const TLS_NEW_REPLACE_OFFSET: u64 = 0x300;
const MALLOC_REPLACE_SIZE: usize = 0x70;
const NEW_REPLACE_SIZE: usize = 0x68;

/// Key of the process exit handler the loader passes to the entry point.
pub const EXIT_HANDLER_NID: &str = "rps5e#ProgramExitHandler";

/// Type of an exported symbol, for `sceKernelDlsym` lookup order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymbolKind {
    Func,
    Object,
    NoType,
}

#[derive(Clone, Debug)]
pub struct ModuleEntry {
    pub handle: i32,
    pub name: String,
    pub path: String,
    /// Host file the module was loaded from (matching key for reloads).
    pub host_path: String,
    pub so_name: String,
    pub base: u64,
    pub end: u64,
    /// Exact extent of the mapped segments (PT_LOAD / PT_SCE_RELRO).
    pub seg_start: u64,
    pub seg_end: u64,
    pub entry_point: u64,
    pub init_entry_point: u64,
    pub fini_entry_point: u64,
    pub eh_frame_hdr: u64,
    pub eh_frame: u64,
    pub eh_frame_size: u64,
    /// Mapped segments in program header order: (address, size, protection
    /// with read 1, write 2, execute 4).
    pub segments: Vec<(u64, u64, i32)>,
    pub is_main: bool,
    pub is_system: bool,
    /// Successful sceKernelLoadStartModule calls not yet undone.
    pub load_count: u32,
}

struct Registry {
    modules: Vec<ModuleEntry>,
    next_handle: i32,
    /// Exported symbols per handle: (NID, kind, address) in symbol order.
    symbols: Vec<(i32, Vec<(String, SymbolKind, u64)>)>,
}

static REGISTRY: Mutex<Registry> = Mutex::new(Registry { modules: Vec::new(), next_handle: 1, symbols: Vec::new() });

fn registry() -> std::sync::MutexGuard<'static, Registry> {
    REGISTRY.lock().unwrap_or_else(|e| e.into_inner())
}

/// Record a module's exported symbols for dynamic lookup.
pub fn register_module_symbols(handle: i32, symbols: Vec<(String, SymbolKind, u64)>) {
    let mut r = registry();
    r.symbols.retain(|(h, _)| *h != handle);
    r.symbols.push((handle, symbols));
}

fn find_in(table: &[(String, SymbolKind, u64)], key: &str) -> Option<u64> {
    [SymbolKind::Func, SymbolKind::Object, SymbolKind::NoType]
        .into_iter()
        .find_map(|k| table.iter().find(|(n, kind, _)| *kind == k && n == key).map(|e| e.2))
}

/// Look a symbol up by name in one module's exports (handle 0 = main
/// executable): by the NID of `name` (functions, then objects, then untyped
/// symbols), then by `name` itself taken as a NID.
pub fn resolve_symbol(handle: i32, name: &str) -> Option<u64> {
    let nid = ps5nid::compute_nid(name);
    let r = registry();
    let handle = if handle == 0 { r.modules.iter().find(|m| m.is_main)?.handle } else { handle };
    let (_, table) = r.symbols.iter().find(|(h, _)| *h == handle)?;
    find_in(table, &nid).or_else(|| find_in(table, name))
}

/// Look a symbol up by name in every loaded module, in load order.
pub fn resolve_symbol_any(name: &str) -> Option<u64> {
    let nid = ps5nid::compute_nid(name);
    let r = registry();
    r.symbols.iter().find_map(|(_, t)| find_in(t, &nid).or_else(|| find_in(t, name)))
}

/// Callbacks registered by the C runtime through the rtld hooks.
static THREAD_DTORS: Mutex<u64> = Mutex::new(0);
static THREAD_ATEXIT_COUNT: Mutex<u64> = Mutex::new(0);
static THREAD_ATEXIT_REPORT: Mutex<u64> = Mutex::new(0);

/// Description of one mapped image handed over by the loader.
pub struct ModuleImage<'a> {
    pub path: &'a str,
    pub host_path: &'a str,
    pub so_name: &'a str,
    pub base: u64,
    pub size: u64,
    pub seg_start: u64,
    pub seg_end: u64,
    pub entry_point: u64,
    pub init_entry_point: u64,
    pub fini_entry_point: u64,
    /// Absolute address and size of the `PT_GNU_EH_FRAME` segment, if any.
    pub eh_frame_hdr_segment: Option<(u64, u64)>,
    /// Mapped segments: (address, size, protection with read 1, write 2,
    /// execute 4).
    pub segments: &'a [(u64, u64, i32)],
    pub is_main: bool,
}

/// Decode a DWARF exception-header pointer of `encoding` at `data`
/// (`field_addr` is its guest address, for pc-relative encodings).
fn decode_eh_pointer(data: &[u8], encoding: u8, field_addr: u64) -> Option<u64> {
    if encoding == 0xFF {
        return None;
    }
    let get = |n: usize| data.get(..n).map(|b| { let mut v = [0u8; 8]; v[..n].copy_from_slice(b); u64::from_le_bytes(v) });
    let value = match encoding & 0x0F {
        0x00 | 0x04 | 0x0C => get(8)?,
        0x02 => get(2)?,
        0x03 => get(4)?,
        0x0A => get(2)? as u16 as i16 as i64 as u64,
        0x0B => get(4)? as u32 as i32 as i64 as u64,
        _ => return None,
    };
    match encoding & 0x70 {
        0x00 => Some(value),
        0x10 => Some(field_addr.wrapping_add(value)),
        _ => None,
    }
}

/// Size of the `.eh_frame` records at `frame` up to and including the
/// four-byte zero terminator, all inside `[frame, limit)`; None when the
/// records cannot be read or run past the limit.
fn eh_frame_records_size(read: &dyn Fn(u64, &mut [u8]) -> bool, frame: u64, limit: u64) -> Option<u64> {
    let mut position = frame;
    loop {
        if position.checked_add(4)? > limit {
            return None;
        }
        let mut word = [0u8; 4];
        if !read(position, &mut word) {
            return None;
        }
        let length = u32::from_le_bytes(word) as u64;
        if length == 0 {
            return Some(position + 4 - frame);
        }
        let body = if length == 0xffff_ffff {
            // 64-bit DWARF: the real length follows.
            let mut wide = [0u8; 8];
            if position.checked_add(12)? > limit || !read(position + 4, &mut wide) {
                return None;
            }
            8u64.checked_add(u64::from_le_bytes(wide))?
        } else {
            length
        };
        position = position.checked_add(4)?.checked_add(body)?;
    }
}

/// Locate `.eh_frame` through `.eh_frame_hdr` (version 1): its pointer
/// field gives the start, and the size covers its records through the zero
/// terminator. When the records cannot be walked, the size runs to the
/// header when the frames precede it, otherwise to the end of the image.
/// Returns (hdr, frame, frame size).
fn eh_frame_info(read: &dyn Fn(u64, &mut [u8]) -> bool, img: &ModuleImage) -> (u64, u64, u64) {
    let Some((hdr, hdr_size)) = img.eh_frame_hdr_segment else { return (0, 0, 0) };
    let mut h = [0u8; 12];
    if hdr_size < 4 || !read(hdr, &mut h) || h[0] != 1 {
        return (hdr, 0, 0);
    }
    let Some(frame) = decode_eh_pointer(&h[4..], h[1], hdr + 4) else { return (hdr, 0, 0) };
    let image_end = img.base + img.size;
    let limit = if hdr > frame { hdr } else { image_end };
    let size = eh_frame_records_size(read, frame, limit).unwrap_or_else(|| limit.saturating_sub(frame));
    (hdr, frame, size)
}

fn file_name(path: &str, handle: i32) -> String {
    std::path::Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| format!("module_{handle:04X}.sprx"))
}

/// Register a mapped image; returns its handle.
pub fn register_module(vm: &ps5vm::VirtualMemory, img: &ModuleImage) -> i32 {
    let read = |addr: u64, buf: &mut [u8]| vm.read(addr, buf).is_ok();
    let (eh_frame_hdr, eh_frame, eh_frame_size) = eh_frame_info(&read, img);
    let mut r = registry();
    let handle = r.next_handle;
    r.next_handle += 1;
    let entry = ModuleEntry {
        handle,
        name: file_name(img.path, handle),
        path: img.path.to_string(),
        host_path: img.host_path.to_string(),
        so_name: img.so_name.to_string(),
        base: img.base,
        end: img.base + img.size,
        seg_start: img.seg_start,
        seg_end: img.seg_end,
        entry_point: img.entry_point,
        init_entry_point: img.init_entry_point,
        fini_entry_point: img.fini_entry_point,
        eh_frame_hdr,
        eh_frame,
        eh_frame_size,
        segments: img.segments.to_vec(),
        is_main: img.is_main,
        is_system: false,
        load_count: 0,
    };
    log::info!(
        "[module] handle={} {} base={:#x} size={:#x} eh_frame_hdr={:#x} eh_frame={:#x}+{:#x}",
        handle, entry.name, entry.base, img.size, eh_frame_hdr, eh_frame, eh_frame_size
    );
    let name = entry.name.clone();
    r.modules.push(entry);
    drop(r);
    for &(addr, size, prot) in img.segments {
        crate::memory::register_module_segment(addr, size, prot, &name);
    }
    handle
}

/// Count the reference that a module loaded with the program holds. A later
/// sceKernelLoadStartModule of the same file returns the module without
/// starting it again, and sceKernelStopUnloadModule unloads it only when no
/// other reference is left.
pub fn hold_start_reference(handle: i32) {
    if let Some(m) = registry().modules.iter_mut().find(|m| m.handle == handle) {
        m.load_count = m.load_count.max(1);
    }
}

/// Forget an unloaded module.
pub fn unregister_module(handle: i32) {
    let mut r = registry();
    let segments: Vec<(u64, u64, i32)> = r.modules.iter().filter(|m| m.handle == handle).flat_map(|m| m.segments.iter().copied()).collect();
    r.modules.retain(|m| m.handle != handle);
    r.symbols.retain(|(h, _)| *h != handle);
    drop(r);
    for (addr, size, _) in segments {
        crate::memory::unregister_module_segment(addr, size);
    }
}

/// Loads modules for the kernel module calls. Installed by the runtime.
pub trait ModuleLoader: Send + Sync {
    /// Load and relocate the module at `guest_path` (not started); returns
    /// its handle or a kernel error code.
    fn load(&self, guest_path: &str) -> Result<i32, i64>;
    /// Unmap a stopped module and unbind every reference to it.
    fn unload(&self, handle: i32);
}

static LOADER: std::sync::OnceLock<Box<dyn ModuleLoader>> = std::sync::OnceLock::new();

pub fn set_module_loader(loader: Box<dyn ModuleLoader>) {
    let _ = LOADER.set(loader);
}

fn same_file(a: &str, b: &str) -> bool {
    a.replace('\\', "/").eq_ignore_ascii_case(&b.replace('\\', "/"))
}

/// sceKernelLoadStartModule(path, args, argp, flags, opt, result*) -> handle
fn cb_load_start_module(ctx: &mut CpuContext) -> HleResult {
    let (path_ptr, args, argp, flags, opt, result_out) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8(), ctx.r9());
    let set_result = |ctx: &CpuContext, v: i64| result_out == 0 || ctx.write_u32(result_out, v as u32);
    let Some(path) = ctx.read_cstring(path_ptr, 1024).filter(|p| !p.trim().is_empty()) else {
        return EINVAL;
    };
    if flags != 0 || opt != 0 {
        log::warn!("[module] sceKernelLoadStartModule('{path}'): flags={flags:#x} opt={opt:#x} ignored");
    }
    match load_start(&path, args, argp) {
        Ok((handle, result)) => {
            if !set_result(ctx, result) {
                return EFAULT;
            }
            handle as i64
        }
        Err(e) => {
            set_result(ctx, e);
            e
        }
    }
}

/// Load `path` (once; a loaded module counts one more reference) and run
/// its module_start: (handle, module_start result) or a kernel error.
fn load_start(path: &str, args: u64, argp: u64) -> Result<(i32, i64), i64> {
    let host = match crate::filesystem::resolve_guest_path(path).filter(|h| h.is_file()) {
        Some(h) => h.to_string_lossy().into_owned(),
        None => {
            log::warn!("[module] load '{path}': no such file");
            return Err(ENOENT);
        }
    };
    // Already loaded: count the reference, do not start it again.
    let loaded = {
        let mut r = registry();
        r.modules.iter_mut().find(|m| !m.host_path.is_empty() && same_file(&m.host_path, &host)).map(|m| {
            m.load_count += 1;
            m.handle
        })
    };
    if let Some(handle) = loaded {
        return Ok((handle, OK));
    }
    let Some(loader) = LOADER.get() else { return Err(ENOENT) };
    let handle = match loader.load(path) {
        Ok(h) => h,
        Err(e) => {
            log::warn!("[module] load '{path}' failed: {e:#x}");
            return Err(e);
        }
    };
    let init = {
        let mut r = registry();
        let m = r.modules.iter_mut().find(|m| m.handle == handle);
        m.map(|m| {
            m.load_count = 1;
            m.init_entry_point
        })
        .unwrap_or(0)
    };
    let mut result = 0;
    if init != 0 {
        log::info!("[module] module_start '{path}' at {init:#x}");
        result = ps5cpu::exec::call_guest_nested(init, [args, argp, 0, 0, 0, 0]) as i32 as i64;
        log::info!("[module] module_start '{path}' -> {result}");
    }
    Ok((handle, result))
}

/// sceKernelStopUnloadModule(handle, args, argp, flags, opt, result*)
fn cb_stop_unload_module(ctx: &mut CpuContext) -> HleResult {
    let (handle, args, argp, flags, opt, result_out) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8(), ctx.r9());
    if flags != 0 || opt != 0 {
        log::warn!("[module] sceKernelStopUnloadModule({handle}): flags={flags:#x} opt={opt:#x} ignored");
    }
    stop_unload(ctx, handle, args, argp, result_out)
}

/// Drop one reference to module `handle`; the last one runs module_stop
/// (its result stored at `result_out` when not 0) and unloads it.
fn stop_unload(ctx: &CpuContext, handle: i32, args: u64, argp: u64, result_out: u64) -> HleResult {
    let m = {
        let mut r = registry();
        let Some(m) = r.modules.iter_mut().find(|m| m.handle == handle) else { return ESRCH };
        if m.is_main {
            return EINVAL;
        }
        if m.load_count > 1 {
            m.load_count -= 1;
            return OK;
        }
        m.clone()
    };
    // Modules with pending thread_local destructors cannot go away.
    let count_fn = *THREAD_ATEXIT_COUNT.lock().unwrap();
    if count_fn != 0 && ps5cpu::exec::call_guest_nested(count_fn, [handle as u64, 0, 0, 0, 0, 0]) as i32 > 0 {
        log::warn!("[module] sceKernelStopUnloadModule: cannot unload {} (thread atexit handlers)", m.name);
        let report = *THREAD_ATEXIT_REPORT.lock().unwrap();
        if report != 0 {
            ps5cpu::exec::call_guest_nested(report, [handle as u64, 0, 0, 0, 0, 0]);
        }
        return EBUSY;
    }
    let mut result = 0;
    if m.fini_entry_point != 0 {
        log::info!("[module] module_stop '{}' at {:#x}", m.name, m.fini_entry_point);
        result = ps5cpu::exec::call_guest_nested(m.fini_entry_point, [args, argp, 0, 0, 0, 0]) as i32 as i64;
    }
    if result_out != 0 && !ctx.write_u32(result_out, result as u32) {
        return EFAULT;
    }
    if let Some(loader) = LOADER.get() {
        loader.unload(handle);
    }
    OK
}

// ---------------------------------------------------------------------------
// dlopen / dlsym / dlclose / dlerror
// ---------------------------------------------------------------------------

/// RTLD_LAZY 1, RTLD_NOW 2, RTLD_GLOBAL 0x100; RTLD_DEFAULT handle.
const RTLD_MODE_MASK: i32 = 3;
const RTLD_GLOBAL: i32 = 0x100;
const RTLD_DEFAULT: u64 = -2i64 as u64;

struct DlState {
    /// dlopen handle -> (module handle, RTLD_GLOBAL).
    handles: Vec<(u64, i32, bool)>,
    next: u64,
    /// The message dlerror reports next, if any.
    error: Option<String>,
    /// Guest buffer dlerror returns.
    buffer: u64,
}

static DL: Mutex<DlState> = Mutex::new(DlState { handles: Vec::new(), next: 1, error: None, buffer: 0 });

fn dl() -> std::sync::MutexGuard<'static, DlState> {
    DL.lock().unwrap_or_else(|e| e.into_inner())
}

fn dl_fail(message: impl Into<String>) -> HleResult {
    let message = message.into();
    log::warn!("[module] {message}");
    dl().error = Some(message);
    0
}

/// dlopen(path, mode) -> handle: loads and starts a guest module as
/// sceKernelLoadStartModule does; NULL names the main program. NULL on
/// failure, with the reason for dlerror.
fn cb_dlopen(ctx: &mut CpuContext) -> HleResult {
    let (path_ptr, flags) = (ctx.rdi(), ctx.rsi() as i32);
    if flags & !(RTLD_MODE_MASK | RTLD_GLOBAL) != 0 || flags & RTLD_MODE_MASK == 0 || flags & RTLD_MODE_MASK == 3 {
        return dl_fail("dlopen: unsupported flags");
    }
    let global = flags & RTLD_GLOBAL != 0 || path_ptr == 0;
    let module = if path_ptr == 0 {
        match main_module() {
            Some(m) => m.handle,
            None => return dl_fail("dlopen: the main program is not loaded"),
        }
    } else {
        let Some(path) = ctx.read_cstring(path_ptr, 1024) else { return dl_fail("dlopen: unreadable module path") };
        if path.is_empty() {
            return dl_fail("dlopen: empty module path");
        }
        match load_start(&path, 0, 0) {
            Ok((h, _)) => h,
            Err(e) => return dl_fail(format!("dlopen: cannot load '{path}' (error {e:#x})")),
        }
    };
    let mut s = dl();
    let handle = s.next;
    s.next += 1;
    s.handles.push((handle, module, global));
    handle as HleResult
}

/// dlsym(handle, name) -> address: in one dlopen'ed module, or with NULL /
/// RTLD_DEFAULT in the global ones and then every loaded module.
fn cb_dlsym(ctx: &mut CpuContext) -> HleResult {
    let (handle, name_ptr) = (ctx.rdi(), ctx.rsi());
    let Some(name) = ctx.read_cstring(name_ptr, 1024).filter(|n| !n.is_empty()) else {
        return dl_fail("dlsym: empty symbol name");
    };
    let modules: Vec<i32> = {
        let s = dl();
        if handle == 0 || handle == RTLD_DEFAULT {
            s.handles.iter().filter(|h| h.2).map(|h| h.1).collect()
        } else {
            match s.handles.iter().find(|h| h.0 == handle) {
                Some(h) => vec![h.1],
                None => {
                    drop(s);
                    return dl_fail("dlsym: invalid module handle");
                }
            }
        }
    };
    let found = modules.iter().find_map(|&m| resolve_symbol(m, &name));
    let found = found.or_else(|| if handle == 0 || handle == RTLD_DEFAULT { resolve_symbol_any(&name) } else { None });
    match found {
        Some(addr) => addr as HleResult,
        None => dl_fail(format!("dlsym: symbol '{name}' not found")),
    }
}

/// dlclose(handle): the last dlopen reference stops and unloads the module
/// (never the main program).
fn cb_dlclose(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi();
    let module = {
        let mut s = dl();
        let Some(i) = s.handles.iter().position(|h| h.0 == handle) else {
            drop(s);
            dl_fail("dlclose: invalid module handle");
            return -1;
        };
        s.handles.remove(i).1
    };
    if main_module().is_some_and(|m| m.handle == module) {
        return 0;
    }
    match stop_unload(ctx, module, 0, 0, 0) {
        OK => 0,
        e => {
            dl_fail(format!("dlclose: module {module} cannot be unloaded (error {e:#x})"));
            -1
        }
    }
}

/// dlerror() -> the message of the last failure since the previous call,
/// or NULL.
fn cb_dlerror(ctx: &mut CpuContext) -> HleResult {
    const SIZE: u64 = 256;
    let mut s = dl();
    let Some(message) = s.error.take() else { return 0 };
    if s.buffer == 0 {
        s.buffer = crate::pthread::alloc_guest_block(ctx, SIZE).unwrap_or(0);
    }
    if s.buffer == 0 {
        return 0;
    }
    let mut bytes = message.into_bytes();
    bytes.truncate(SIZE as usize - 1);
    bytes.push(0);
    if ctx.write_bytes(s.buffer, &bytes) { s.buffer as HleResult } else { 0 }
}

const DL_EXPORTS: &[(&str, &str, crate::registry::HleCallback)] = &[
    ("UteVS6B1ZrU", "dlopen", cb_dlopen),
    ("biY+kcVB5D4", "dlsym", cb_dlsym),
    ("oiT0vgzmW0M", "dlclose", cb_dlclose),
    ("ucFJiTO1EUw", "dlerror", cb_dlerror),
];

/// Stop every loaded module (process exit): module_stop of each PRX.
pub fn stop_all_modules() {
    let finis: Vec<(String, u64)> =
        registry().modules.iter().filter(|m| !m.is_main && m.fini_entry_point != 0).map(|m| (m.name.clone(), m.fini_entry_point)).collect();
    for (name, fini) in finis {
        log::info!("[module] module_stop '{name}' at {fini:#x}");
        ps5cpu::exec::call_guest_nested(fini, [0; 6]);
    }
}

/// Process exit handler passed to the main entry point.
fn cb_program_exit_handler(_ctx: &mut CpuContext) -> HleResult {
    stop_all_modules();
    log::info!("[module] exit");
    0
}

pub fn module_by_address(addr: u64) -> Option<ModuleEntry> {
    registry().modules.iter().find(|m| addr >= m.base && addr < m.end).cloned()
}

/// The main program (eboot.bin).
pub fn main_module() -> Option<ModuleEntry> {
    registry().modules.iter().find(|m| m.is_main).cloned()
}

pub fn module_by_handle(handle: i32) -> Option<ModuleEntry> {
    registry().modules.iter().find(|m| m.handle == handle).cloned()
}

/// Module loaded from a file of this name (e.g. "libSceFiber.prx").
pub fn module_by_file_name(name: &str) -> Option<ModuleEntry> {
    registry().modules.iter().find(|m| m.name == name).cloned()
}

/// Module whose files were loaded from `host_path`.
pub fn module_by_host_path(host_path: &str) -> Option<ModuleEntry> {
    registry().modules.iter().find(|m| !m.host_path.is_empty() && same_file(&m.host_path, host_path)).cloned()
}

fn write_name(p: &mut [u8], name: &str) {
    let b = name.as_bytes();
    let n = b.len().min(NAME_MAX_BYTES - 1);
    p[NAME_OFFSET..NAME_OFFSET + n].copy_from_slice(&b[..n]);
}

/// Segment records of the module info structures: up to four
/// `{ u64 address, u32 size, i32 prot }`, in program header order. A module
/// registered without segment details gets one R+X segment over the image.
/// Returns the number written.
fn write_segments(p: &mut [u8], off: usize, m: &ModuleEntry) -> u32 {
    let whole = [(m.base, m.end - m.base, 5)];
    let segments: &[(u64, u64, i32)] = if m.segments.is_empty() { &whole } else { &m.segments };
    let mut count = 0;
    for &(address, size, prot) in segments.iter().take(MODULE_INFO_SEGMENTS) {
        let o = off + count * 16;
        p[o..o + 8].copy_from_slice(&address.to_le_bytes());
        p[o + 8..o + 12].copy_from_slice(&(size.min(u32::MAX as u64) as u32).to_le_bytes());
        p[o + 12..o + 16].copy_from_slice(&prot.to_le_bytes());
        count += 1;
    }
    count as u32
}

/// `ModuleInfo`: size, name[256], segments[4] at 0x108, segment count at
/// 0x148, a zero fingerprint[20] at 0x14C.
fn module_info(m: &ModuleEntry) -> Vec<u8> {
    let mut p = vec![0u8; MODULE_INFO_SIZE];
    p[0..8].copy_from_slice(&(MODULE_INFO_SIZE as u64).to_le_bytes());
    write_name(&mut p, &m.name);
    let count = write_segments(&mut p, MODULE_INFO_SEGMENTS_OFFSET, m);
    p[MODULE_INFO_SEGMENT_COUNT_OFFSET..MODULE_INFO_SEGMENT_COUNT_OFFSET + 4].copy_from_slice(&count.to_le_bytes());
    p
}

fn module_info_ex(m: &ModuleEntry) -> Vec<u8> {
    let mut p = vec![0u8; MODULE_INFO_EX_SIZE];
    p[0..8].copy_from_slice(&(MODULE_INFO_EX_SIZE as u64).to_le_bytes());
    write_name(&mut p, &m.name);
    p[MODULE_INFO_EX_HANDLE_OFFSET..MODULE_INFO_EX_HANDLE_OFFSET + 4].copy_from_slice(&m.handle.to_le_bytes());
    p[MODULE_INFO_EX_INIT_PROC_OFFSET..MODULE_INFO_EX_INIT_PROC_OFFSET + 8].copy_from_slice(&m.entry_point.to_le_bytes());
    let count = write_segments(&mut p, MODULE_INFO_EX_SEGMENTS_OFFSET, m);
    p[MODULE_INFO_EX_SEGMENT_COUNT_OFFSET..MODULE_INFO_EX_SEGMENT_COUNT_OFFSET + 4].copy_from_slice(&count.to_le_bytes());
    p
}

/// `ModuleInfoForUnwind` of a module: its shared-object name (or file
/// name), the `.eh_frame_hdr`/`.eh_frame` location and the extent of its
/// segments.
fn unwind_info(m: &ModuleEntry) -> Vec<u8> {
    let mut p = vec![0u8; UNWIND_INFO_SIZE];
    p[0..8].copy_from_slice(&(UNWIND_INFO_SIZE as u64).to_le_bytes());
    write_name(&mut p, if m.so_name.is_empty() { &m.name } else { &m.so_name });
    p[0x108..0x110].copy_from_slice(&m.eh_frame_hdr.to_le_bytes());
    p[0x110..0x118].copy_from_slice(&m.eh_frame.to_le_bytes());
    p[0x118..0x120].copy_from_slice(&m.eh_frame_size.to_le_bytes());
    let (start, end) = if m.seg_end > m.seg_start { (m.seg_start, m.seg_end) } else { (m.base, m.end) };
    p[0x120..0x128].copy_from_slice(&start.to_le_bytes());
    p[0x128..0x130].copy_from_slice(&(end - start).to_le_bytes());
    p
}

/// sceKernelGetModuleInfoFromAddr(addr, flags, info_ex*). The input size
/// field is not read: a shipped libc.prx passes an uninitialised 0x1A8-byte
/// buffer. The full structure, size field included, is written back.
/// Flags 1 include the system modules, flags 2 only the application's
/// (behaviour measured on hardware); anything else zeroes the structure.
fn cb_get_module_info_from_addr(ctx: &mut CpuContext) -> HleResult {
    let (addr, flags, out) = (ctx.rdi(), ctx.rsi() as i32, ctx.rdx());
    if flags != 1 && flags != 2 {
        if out != 0 {
            ctx.write_bytes(out, &vec![0u8; MODULE_INFO_EX_SIZE]);
        }
        return EINVAL;
    }
    if out == 0 {
        return EFAULT;
    }
    let found = module_by_address(addr).filter(|m| flags == 1 || !m.is_system);
    let Some(m) = found else {
        ctx.write_bytes(out, &vec![0u8; MODULE_INFO_EX_SIZE]);
        return ESRCH;
    };
    if ctx.write_bytes(out, &module_info_ex(&m)) { OK } else { EFAULT }
}

/// sceKernelGetModuleInfoForUnwind(addr, flags, info*)
fn cb_get_module_info_for_unwind(ctx: &mut CpuContext) -> HleResult {
    let (addr, flags, out) = (ctx.rdi(), ctx.rsi() as i32, ctx.rdx());
    if flags >= 3 {
        if out != 0 {
            ctx.write_bytes(out, &[0u8; UNWIND_INFO_SIZE]);
        }
        return EINVAL;
    }
    if out == 0 {
        return EFAULT;
    }
    let Some(size) = ctx.read_u64(out) else { return EFAULT };
    if size < UNWIND_INFO_SIZE as u64 {
        return EINVAL;
    }
    let m = match module_by_address(addr) {
        Some(m) => m,
        None if (GUEST_EXEC_FLOOR..GUEST_EXEC_CEILING).contains(&addr) => return ESRCH,
        None => {
            // A host frame: report an empty pseudo-module so unwinding stops.
            let base = addr & !(HOST_UNWIND_BOUNDARY - 1);
            ModuleEntry {
                handle: 0,
                name: "HostBoundary".into(),
                path: String::new(),
                host_path: String::new(),
                so_name: String::new(),
                base,
                end: base + HOST_UNWIND_BOUNDARY,
                seg_start: base,
                seg_end: base + HOST_UNWIND_BOUNDARY,
                entry_point: 0,
                init_entry_point: 0,
                fini_entry_point: 0,
                eh_frame_hdr: 0,
                eh_frame: 0,
                eh_frame_size: 0,
                segments: Vec::new(),
                is_main: false,
                is_system: false,
                load_count: 0,
            }
        }
    };
    if ctx.write_bytes(out, &unwind_info(&m)) { OK } else { EFAULT }
}

fn info_by_handle(ctx: &mut CpuContext, ex: bool) -> HleResult {
    let (handle, out) = (ctx.rdi() as i32, ctx.rsi());
    if out == 0 {
        return EFAULT;
    }
    // The size is checked before the handle; a rejected call leaves the
    // structure untouched.
    let Some(size) = ctx.read_u64(out) else { return EFAULT };
    if size != if ex { MODULE_INFO_EX_SIZE } else { MODULE_INFO_SIZE } as u64 {
        return EINVAL;
    }
    let Some(m) = module_by_handle(handle) else { return ESRCH };
    let payload = if ex { module_info_ex(&m) } else { module_info(&m) };
    if ctx.write_bytes(out, &payload) { OK } else { EFAULT }
}

/// sceKernelGetModuleInfo / sceKernelGetModuleInfo2(handle, info*)
fn cb_get_module_info(ctx: &mut CpuContext) -> HleResult {
    info_by_handle(ctx, false)
}

/// sceKernelGetModuleInfoInternal(handle, info_ex*)
fn cb_get_module_info_internal(ctx: &mut CpuContext) -> HleResult {
    info_by_handle(ctx, true)
}

/// The `[start, end)` of a module's first executable segment, in program
/// header order. A module registered without segment details has its whole
/// image as one R+X segment, as `write_segments` reports it.
fn exec_segment(m: &ModuleEntry) -> Option<(u64, u64)> {
    if m.segments.is_empty() {
        return Some((m.base, m.end));
    }
    m.segments.iter().find(|&&(_, _, prot)| prot & 4 != 0).map(|&(address, size, _)| (address, address + size))
}

/// The executable extent of the main module, among `modules`.
fn main_exec_segment(modules: &[ModuleEntry]) -> Option<(u64, u64)> {
    modules.iter().find(|m| m.is_main).and_then(exec_segment)
}

/// sceKernelInternalMemoryGetModuleSegmentInfo(u64 out[2]): writes the
/// `[start, end)` of the main module's executable segment and returns 0. The
/// libc frame walker accepts return addresses inside that range; the caller
/// takes no module handle. ESRCH when there is no main module.
fn cb_internal_memory_get_module_segment_info(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return EFAULT;
    }
    let Some((start, end)) = main_exec_segment(&registry().modules) else { return ESRCH };
    let mut record = [0u8; 16];
    record[..8].copy_from_slice(&start.to_le_bytes());
    record[8..].copy_from_slice(&end.to_le_bytes());
    if ctx.write_bytes(out, &record) { OK } else { EFAULT }
}

/// (handles*, capacity, count*): both pointers are required. The handles
/// that fit are written; when they do not all fit the call fails with
/// ENOMEM and leaves `*count` alone, otherwise `*count` is the total.
fn module_list(ctx: &mut CpuContext, include_system: bool) -> HleResult {
    let (handles_out, capacity, count_out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if handles_out == 0 || count_out == 0 {
        return EFAULT;
    }
    let handles: Vec<i32> = registry().modules.iter().filter(|m| include_system || !m.is_system).map(|m| m.handle).collect();
    for (i, h) in handles.iter().take(capacity.min(handles.len() as u64) as usize).enumerate() {
        if !ctx.write_u32(handles_out + i as u64 * 4, *h as u32) {
            return EFAULT;
        }
    }
    if handles.len() as u64 > capacity {
        return ENOMEM;
    }
    if ctx.write_u64(count_out, handles.len() as u64) { OK } else { EFAULT }
}

/// sceKernelGetModuleList(handles*, capacity, count*)
fn cb_get_module_list(ctx: &mut CpuContext) -> HleResult {
    module_list(ctx, true)
}

/// sceKernelGetModuleList2 — excludes system modules.
fn cb_get_module_list2(ctx: &mut CpuContext) -> HleResult {
    module_list(ctx, false)
}

/// __elf_phdr_match_addr(module_info*, dtor_addr): whether the destructor
/// lies in the module the info describes (its handle at +0x108).
fn cb_elf_phdr_match_addr(ctx: &mut CpuContext) -> HleResult {
    let (info, dtor) = (ctx.rdi(), ctx.rsi());
    if info == 0 {
        return 0;
    }
    let Some(handle) = ctx.read_u32(info + MODULE_INFO_EX_HANDLE_OFFSET as u64) else { return 0 };
    module_by_address(dtor).is_some_and(|m| m.handle == handle as i32) as i64
}

/// _sceKernelRtldThreadAtexitIncrement(dso): the `dso` address libc passes
/// for `__cxa_thread_atexit`, which fails when this is nonzero. Measured on
/// hardware: -1 for an address outside every module (null, stack, heap), 0
/// for one inside a module; the module's reference count does not change.
fn cb_rtld_thread_atexit_increment(ctx: &mut CpuContext) -> HleResult {
    if module_by_address(ctx.rdi()).is_some() {
        0
    } else {
        -1
    }
}

/// _sceKernelRtldThreadAtexitDecrement(dso): returns nothing on the console
/// and never fails, whatever the address; 0 here.
fn cb_rtld_thread_atexit_decrement(_ctx: &mut CpuContext) -> HleResult {
    0
}

fn cb_set_thread_dtors(ctx: &mut CpuContext) -> HleResult {
    *THREAD_DTORS.lock().unwrap() = ctx.rdi();
    OK
}

fn cb_set_thread_atexit_count(ctx: &mut CpuContext) -> HleResult {
    *THREAD_ATEXIT_COUNT.lock().unwrap() = ctx.rdi();
    OK
}

fn cb_set_thread_atexit_report(ctx: &mut CpuContext) -> HleResult {
    *THREAD_ATEXIT_REPORT.lock().unwrap() = ctx.rdi();
    OK
}

/// The C runtime's per-thread destructor callback (run on thread exit).
pub fn thread_dtors_callback() -> u64 {
    *THREAD_DTORS.lock().unwrap()
}

/// Zeroed replace table in the TCB scratch area, with its size in word 0.
fn sanitizer_replace(ctx: &mut CpuContext, offset: u64, size: usize) -> HleResult {
    let tcb = ps5cpu::exec::current_thread_pointer();
    if tcb == 0 {
        return 0;
    }
    let addr = tcb + offset;
    let mut table = vec![0u8; size];
    table[0..8].copy_from_slice(&(size as u64).to_le_bytes());
    if !ctx.write_bytes(addr, &table) {
        return EFAULT;
    }
    addr as i64
}

fn cb_get_sanitizer_malloc_replace_external(ctx: &mut CpuContext) -> HleResult {
    sanitizer_replace(ctx, TLS_MALLOC_REPLACE_OFFSET, MALLOC_REPLACE_SIZE)
}

fn cb_get_sanitizer_new_replace_external(ctx: &mut CpuContext) -> HleResult {
    sanitizer_replace(ctx, TLS_NEW_REPLACE_OFFSET, NEW_REPLACE_SIZE)
}

fn cb_is_address_sanitizer_enabled(_ctx: &mut CpuContext) -> HleResult {
    0
}

pub fn register(reg: &mut HleRegistry) {
    reg.register_fn("f7KBOafysXo", "sceKernelGetModuleInfoFromAddr", "libKernel", cb_get_module_info_from_addr);
    reg.register_fn("RpQJJVKTiFM", "sceKernelGetModuleInfoForUnwind", "libKernel", cb_get_module_info_for_unwind);
    reg.register_fn("4fU5yvOkVG4", "sceSysmoduleGetModuleInfoForUnwind", "libSceSysmodule", cb_get_module_info_for_unwind);
    reg.register_fn("-YTW+qXc3CQ", "sceKernelInternalMemoryGetModuleSegmentInfo", "libKernel", cb_internal_memory_get_module_segment_info);
    reg.register_fn("kUpgrXIrz7Q", "sceKernelGetModuleInfo", "libKernel", cb_get_module_info);
    reg.register_fn("QgsKEUfkqMA", "sceKernelGetModuleInfo2", "libKernel", cb_get_module_info);
    reg.register_fn("HZO7xOos4xc", "sceKernelGetModuleInfoInternal", "libKernel", cb_get_module_info_internal);
    reg.register_fn("IuxnUuXk6Bg", "sceKernelGetModuleList", "libKernel", cb_get_module_list);
    reg.register_fn("ZzzC3ZGVAkc", "sceKernelGetModuleList2", "libKernel", cb_get_module_list2);
    reg.register_fn("Fjc4-n1+y2g", "__elf_phdr_match_addr", "libKernel", cb_elf_phdr_match_addr);
    reg.register_fn("wzvqT4UqKX8", "sceKernelLoadStartModule", "libKernel", cb_load_start_module);
    // The system-module loader's entry takes the same arguments.
    reg.register_fn("Gg3+yWL6DWU", "sceKernelLoadStartModuleForSysmodule", "libKernel", cb_load_start_module);
    reg.register_fn("QKd0qM58Qes", "sceKernelStopUnloadModule", "libKernel", cb_stop_unload_module);
    for &(nid, name, cb) in DL_EXPORTS {
        reg.register_fn(nid, name, "libKernel", cb);
    }
    reg.register_fn("Tz4RNUCBbGI", "_sceKernelRtldThreadAtexitIncrement", "libKernel", cb_rtld_thread_atexit_increment);
    reg.register_fn("8OnWXlgQlvo", "_sceKernelRtldThreadAtexitDecrement", "libKernel", cb_rtld_thread_atexit_decrement);
    reg.register_fn("rNhWz+lvOMU", "_sceKernelSetThreadDtors", "libKernel", cb_set_thread_dtors);
    reg.register_fn("pB-yGZ2nQ9o", "_sceKernelSetThreadAtexitCount", "libKernel", cb_set_thread_atexit_count);
    reg.register_fn("WhCc1w3EhSI", "_sceKernelSetThreadAtexitReport", "libKernel", cb_set_thread_atexit_report);
    reg.register_fn("py6L8jiVAN8", "sceKernelGetSanitizerMallocReplaceExternal", "libKernel", cb_get_sanitizer_malloc_replace_external);
    reg.register_fn("bnZxYgAFeA0", "sceKernelGetSanitizerNewReplaceExternal", "libKernel", cb_get_sanitizer_new_replace_external);
    reg.register_fn("jh+8XiK4LeE", "sceKernelIsAddressSanitizerEnabled", "libKernel", cb_is_address_sanitizer_enabled);
    reg.register_fn(EXIT_HANDLER_NID, "ProgramExitHandler", "libKernel", cb_program_exit_handler);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The system-module loader entry is exported next to
    /// sceKernelLoadStartModule, under its own NID.
    #[test]
    fn load_start_module_for_sysmodule_is_exported() {
        let mut reg = HleRegistry::new();
        register(&mut reg);
        let name = "sceKernelLoadStartModuleForSysmodule";
        assert_eq!(ps5nid::compute_nid(name), "Gg3+yWL6DWU");
        assert!(reg.lookup("Gg3+yWL6DWU").is_some(), "{name}");
    }

    fn image<'a>(hdr: Option<(u64, u64)>) -> ModuleImage<'a> {
        ModuleImage {
            path: "/app0/eboot.bin",
            host_path: "",
            so_name: "",
            base: 0x1000_0000,
            size: 0x10000,
            seg_start: 0x1000_0000,
            seg_end: 0x1000_f000,
            entry_point: 0,
            init_entry_point: 0,
            fini_entry_point: 0,
            eh_frame_hdr_segment: hdr,
            segments: &[],
            is_main: true,
        }
    }

    #[test]
    fn eh_frame_before_header() {
        let mem: std::collections::HashMap<u64, [u8; 12]> =
            [(0x1000_2000u64, [1u8, 0x1B, 0x03, 0x3B, 0xF0u8.wrapping_neg(), 0xEF, 0xFF, 0xFF, 0, 0, 0, 0])].into();
        // eh_frame_ptr = -0x1010 relative to hdr+4 -> 0x1000_0FF4
        let read = |a: u64, b: &mut [u8]| match mem.get(&a) {
            Some(v) => {
                b.copy_from_slice(&v[..b.len()]);
                true
            }
            None => false,
        };
        let img = image(Some((0x1000_2000, 0x100)));
        let (hdr, frame, size) = eh_frame_info(&read, &img);
        assert_eq!(hdr, 0x1000_2000);
        assert_eq!(frame, (0x1000_2004i64 + i32::from_le_bytes([0x10, 0xEF, 0xFF, 0xFF]) as i64) as u64);
        assert_eq!(size, hdr - frame);
    }

    #[test]
    fn eh_frame_pointer_encodings() {
        // absptr (udata8)
        assert_eq!(decode_eh_pointer(&0x1234_5678_9ABCu64.to_le_bytes(), 0x00, 0), Some(0x1234_5678_9ABC));
        // udata4, pc-relative
        assert_eq!(decode_eh_pointer(&0x10u32.to_le_bytes(), 0x13, 0x1000), Some(0x1010));
        // sdata4, pc-relative (negative)
        assert_eq!(decode_eh_pointer(&(-0x10i32).to_le_bytes(), 0x1B, 0x1000), Some(0xFF0));
        // sdata8, absolute
        assert_eq!(decode_eh_pointer(&(-1i64).to_le_bytes(), 0x0C, 0), Some(u64::MAX));
        // omitted / unsupported application / short data
        assert_eq!(decode_eh_pointer(&[0; 8], 0xFF, 0), None);
        assert_eq!(decode_eh_pointer(&[0; 8], 0x33, 0), None);
        assert_eq!(decode_eh_pointer(&[0; 2], 0x03, 0), None);
    }

    #[test]
    fn eh_frame_after_header_runs_to_image_end() {
        let mut bytes = [1u8, 0x13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        bytes[4..8].copy_from_slice(&0x100u32.to_le_bytes());
        let read = move |a: u64, b: &mut [u8]| {
            a == 0x1000_2000 && {
                b.copy_from_slice(&bytes[..b.len()]);
                true
            }
        };
        let (_, frame, size) = eh_frame_info(&read, &image(Some((0x1000_2000, 0x40))));
        assert_eq!(frame, 0x1000_2104);
        assert_eq!(size, 0x1001_0000 - frame, "unreadable records: the frames run to the image end");
    }

    #[test]
    fn eh_frame_size_counts_records_through_the_terminator() {
        // Header at 0x1000_2000, frames at 0x1000_2100: a CIE of 0x14 body
        // bytes, a 64-bit-length FDE of 0x10 body bytes, then the zero word.
        let mut mem = vec![0u8; 0x200];
        mem[0..4].copy_from_slice(&[1, 0x1B, 0x03, 0x3B]);
        mem[4..8].copy_from_slice(&(0xFCi32).to_le_bytes());
        mem[0x100..0x104].copy_from_slice(&0x14u32.to_le_bytes());
        mem[0x118..0x11C].copy_from_slice(&0xffff_ffffu32.to_le_bytes());
        mem[0x11C..0x124].copy_from_slice(&0x10u64.to_le_bytes());
        let read = |a: u64, b: &mut [u8]| {
            let Some(off) = a.checked_sub(0x1000_2000).map(|o| o as usize).filter(|&o| o + b.len() <= mem.len()) else { return false };
            b.copy_from_slice(&mem[off..off + b.len()]);
            true
        };
        let (hdr, frame, size) = eh_frame_info(&read, &image(Some((0x1000_2000, 0x40))));
        assert_eq!((hdr, frame), (0x1000_2000, 0x1000_2100));
        assert_eq!(size, 0x18 + 0x1C + 4, "records plus the four-byte terminator");
        // A record running past the image is not trusted.
        assert_eq!(eh_frame_records_size(&read, 0x1000_2100, 0x1000_2120), None);
    }

    fn entry(segments: Vec<(u64, u64, i32)>) -> ModuleEntry {
        ModuleEntry {
            handle: 7,
            name: "segments.prx".into(),
            path: String::new(),
            host_path: String::new(),
            so_name: String::new(),
            base: 0x2_0000_0000,
            end: 0x2_0004_0000,
            seg_start: 0x2_0000_0000,
            seg_end: 0x2_0004_0000,
            entry_point: 0,
            init_entry_point: 0,
            fini_entry_point: 0,
            eh_frame_hdr: 0,
            eh_frame: 0,
            eh_frame_size: 0,
            segments,
            is_main: false,
            is_system: false,
            load_count: 0,
        }
    }

    /// The extent that sceKernelInternalMemoryGetModuleSegmentInfo returns is
    /// the first executable segment of the main module only.
    #[test]
    fn main_segment_info_is_the_first_executable_segment() {
        assert_eq!(ps5nid::compute_nid("sceKernelInternalMemoryGetModuleSegmentInfo"), "-YTW+qXc3CQ");
        let base = 0x2_0000_0000;
        let mut main = entry(vec![(base, 0x2_0000, 4 | 1), (base + 0x2_0000, 0x8000, 2 | 1), (base + 0x2_8000, 0x1_8000, 4)]);
        main.is_main = true;
        let mut library = entry(vec![(base + 0x100_0000, 0x1000, 5)]);
        library.handle = 8;
        // The library comes first and is not the main module.
        assert_eq!(main_exec_segment(&[library.clone(), main.clone()]), Some((base, base + 0x2_0000)));
        assert_eq!(main_exec_segment(&[library.clone()]), None, "no main module");
        // A main module without segment details is one executable image.
        let mut bare = entry(Vec::new());
        bare.is_main = true;
        assert_eq!(main_exec_segment(&[bare.clone()]), Some((bare.base, bare.end)));
        // A main module with data only has no code.
        let mut data = entry(vec![(base, 0x1000, 3)]);
        data.is_main = true;
        assert_eq!(main_exec_segment(&[data]), None);

        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        assert_eq!(cb_internal_memory_get_module_segment_info(&mut ctx), EFAULT, "a NULL record");
    }

    #[test]
    fn module_info_reports_each_segment() {
        let segs = vec![(0x2_0000_0000, 0x2_0000, 5), (0x2_0002_0000, 0x8000, 1), (0x2_0002_8000, 0x1_8000, 3)];
        let p = module_info(&entry(segs.clone()));
        assert_eq!(p.len(), 0x160);
        assert_eq!(u32::from_le_bytes(p[0x148..0x14C].try_into().unwrap()), 3);
        for (i, &(a, s, prot)) in segs.iter().enumerate() {
            let o = 0x108 + i * 16;
            assert_eq!(u64::from_le_bytes(p[o..o + 8].try_into().unwrap()), a);
            assert_eq!(u32::from_le_bytes(p[o + 8..o + 12].try_into().unwrap()) as u64, s);
            assert_eq!(i32::from_le_bytes(p[o + 12..o + 16].try_into().unwrap()), prot);
        }
        assert!(p[0x14C..0x160].iter().all(|&b| b == 0), "fingerprint is zero");
        let ex = module_info_ex(&entry(segs));
        assert_eq!(u32::from_le_bytes(ex[0x1A0..0x1A4].try_into().unwrap()), 3);
        assert_eq!(u64::from_le_bytes(ex[0x170..0x178].try_into().unwrap()), 0x2_0002_0000);
        // At most four segments; none recorded means one R+X image segment.
        let many = module_info(&entry((0..6).map(|i| (0x2_0000_0000 + i * 0x4000, 0x4000, 1)).collect()));
        assert_eq!(u32::from_le_bytes(many[0x148..0x14C].try_into().unwrap()), 4);
        let whole = module_info(&entry(Vec::new()));
        assert_eq!(u32::from_le_bytes(whole[0x148..0x14C].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(whole[0x110..0x114].try_into().unwrap()), 0x4_0000);
        assert_eq!(i32::from_le_bytes(whole[0x114..0x118].try_into().unwrap()), 5);
    }

    /// Serializes the tests that register modules in the global registry.
    static REGISTRY_TESTS: Mutex<()> = Mutex::new(());

    const VM_BASE: u64 = 0x5D7C_0000_0000;

    fn guest() -> &'static ps5vm::VirtualMemory {
        static VM: std::sync::OnceLock<&'static ps5vm::VirtualMemory> = std::sync::OnceLock::new();
        VM.get_or_init(|| {
            let vm: &'static ps5vm::VirtualMemory = Box::leak(Box::new(ps5vm::VirtualMemory::new()));
            vm.map(VM_BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
            vm
        })
    }

    fn call(cb: crate::registry::HleCallback, args: &[u64]) -> HleResult {
        use crate::context::CpuRegister;
        let mut ctx = CpuContext::new(guest());
        for (r, v) in [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx].iter().zip(args) {
            ctx.set(*r, *v);
        }
        cb(&mut ctx)
    }

    #[test]
    fn module_info_checks_size_before_handle() {
        let _g = REGISTRY_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let vm = guest();
        let mut img = image(None);
        img.path = "/app0/module-info-test.prx";
        img.is_main = false;
        let segs = [(0x1000_0000, 0x8000, 5), (0x1000_8000, 0x8000, 3)];
        img.segments = &segs;
        let h = register_module(vm, &img);
        let info = VM_BASE + 0x1000;
        let ctx = CpuContext::new(vm);
        for size in [0x158u64, 0x1a8, 0x1b0, 0x200, 0] {
            ctx.write_u64(info, size);
            ctx.write_bytes(info + 8, b"x");
            assert_eq!(call(cb_get_module_info, &[h as u64, info]), EINVAL, "size {size:#x}");
            assert_eq!(ctx.read_u64(info), Some(size));
            assert_eq!(ctx.read_bytes(info + 8, 1).unwrap()[0], b'x', "rejected calls leave the structure");
        }
        ctx.write_u64(info, 0x160);
        assert_eq!(call(cb_get_module_info, &[h as u64, info]), OK);
        assert_eq!(ctx.read_bytes(info + 8, 22).unwrap(), b"module-info-test.prx\0\0");
        assert_eq!(ctx.read_u32(info + 0x148), Some(2));
        assert_eq!(ctx.read_u32(info + 0x124), Some(3));
        assert_eq!(call(cb_get_module_info, &[h as u64, 0]), EFAULT);
        ctx.write_u64(info, 0x160);
        ctx.write_bytes(info + 8, &[0]);
        assert_eq!(call(cb_get_module_info, &[(-1i32) as u32 as u64, info]), ESRCH);
        assert_eq!(ctx.read_bytes(info + 8, 1).unwrap()[0], 0);
        unregister_module(h);
    }

    /// The thread-atexit references: an address inside a module is 0, one
    /// outside every module (null, stack, heap) -1; Decrement never fails.
    #[test]
    fn rtld_thread_atexit_references() {
        let _g = REGISTRY_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let vm = guest();
        let mut img = image(None);
        img.path = "/app0/thread-atexit-test.prx";
        img.is_main = false;
        let h = register_module(vm, &img);
        let local = 0u64;
        let stack = &local as *const u64 as u64;
        let load_count = module_by_handle(h).map(|m| m.load_count);
        for inside in [0x1000_0000u64, 0x1000_8000, 0x1000_ffff] {
            assert_eq!(call(cb_rtld_thread_atexit_increment, &[inside]), 0, "{inside:#x}");
            assert_eq!(call(cb_rtld_thread_atexit_decrement, &[inside]), 0, "{inside:#x}");
        }
        for outside in [0u64, stack, VM_BASE, 0x1001_0000] {
            assert_eq!(call(cb_rtld_thread_atexit_increment, &[outside]), -1, "{outside:#x}");
            assert_eq!(call(cb_rtld_thread_atexit_decrement, &[outside]), 0, "{outside:#x}");
        }
        assert_eq!(module_by_handle(h).map(|m| m.load_count), load_count, "the reference count does not change");
        unregister_module(h);
    }

    #[test]
    fn module_list_contracts() {
        let _g = REGISTRY_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let vm = guest();
        let mut img = image(None);
        img.path = "/app0/module-list-test.prx";
        img.is_main = false;
        let h = register_module(vm, &img);
        let ctx = CpuContext::new(vm);
        let (handles, count) = (VM_BASE + 0x2000, VM_BASE + 0x3000);
        ctx.write_bytes(handles, &[0xff; 0x800]);
        assert_eq!(call(cb_get_module_list, &[handles, 512, count]), OK);
        let n = ctx.read_u64(count).unwrap();
        assert!(n >= 1 && n < 512);
        let listed: Vec<i32> = (0..n).map(|i| ctx.read_u32(handles + i * 4).unwrap() as i32).collect();
        assert_eq!(listed.iter().filter(|&&x| x == h).count(), 1);
        assert_eq!(ctx.read_u32(handles + n * 4), Some(u32::MAX), "nothing written past the list");
        // Too small: the handles that fit, ENOMEM, the count untouched.
        ctx.write_bytes(handles, &[0xff; 0x800]);
        ctx.write_u64(count, 99);
        assert_eq!(call(cb_get_module_list, &[handles, n - 1, count]), ENOMEM);
        assert_eq!(ctx.read_u64(count), Some(99));
        for (i, &x) in listed.iter().take(n as usize - 1).enumerate() {
            assert_eq!(ctx.read_u32(handles + i as u64 * 4), Some(x as u32));
        }
        assert_eq!(ctx.read_u32(handles + (n - 1) * 4), Some(u32::MAX));
        assert_eq!(call(cb_get_module_list, &[0, 0, count]), EFAULT);
        assert_eq!(call(cb_get_module_list, &[handles, 512, 0]), EFAULT);
        unregister_module(h);
    }

    #[test]
    fn module_info_ex_layout() {
        let m = ModuleEntry {
            handle: 3,
            name: "libc.prx".into(),
            path: String::new(),
            host_path: String::new(),
            so_name: "libc.so".into(),
            base: 0x1_0000_0000,
            end: 0x1_0014_c000,
            seg_start: 0x1_0000_0000,
            seg_end: 0x1_0014_a123,
            entry_point: 0x1_0000_0010,
            init_entry_point: 0,
            fini_entry_point: 0,
            eh_frame_hdr: 0,
            eh_frame: 0,
            eh_frame_size: 0,
            segments: Vec::new(),
            is_main: false,
            is_system: false,
            load_count: 0,
        };
        let p = module_info_ex(&m);
        assert_eq!(u64::from_le_bytes(p[0..8].try_into().unwrap()), 0x1A8);
        assert_eq!(&p[8..16], b"libc.prx");
        assert_eq!(i32::from_le_bytes(p[0x108..0x10C].try_into().unwrap()), 3);
        assert_eq!(u64::from_le_bytes(p[0x160..0x168].try_into().unwrap()), 0x1_0000_0000);
        assert_eq!(u32::from_le_bytes(p[0x1A0..0x1A4].try_into().unwrap()), 1);
        // Unwind info: shared-object name, exact segment extent.
        let u = unwind_info(&m);
        assert_eq!(u.len(), 304);
        assert_eq!(&u[8..16], b"libc.so\0");
        assert_eq!(u64::from_le_bytes(u[0x120..0x128].try_into().unwrap()), 0x1_0000_0000);
        assert_eq!(u64::from_le_bytes(u[0x128..0x130].try_into().unwrap()), 0x14a123);
    }

    /// sceKernelGetModuleInfoFromAddr does not read the caller's size field
    /// (an uninitialised 0xF0000 works) and writes the full 0x1A8 bytes with
    /// the size; flags outside 0..3, a NULL info and an unknown address fail.
    #[test]
    fn module_info_from_addr_ignores_the_input_size() {
        let _g = REGISTRY_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        const BASE: u64 = 0x7C4B_0000_0000;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut img = image(None);
        img.path = "/app0/info-from-addr.prx";
        img.base = 0x7C4C_0000_0000;
        img.seg_start = img.base;
        img.seg_end = img.base + 0x1000;
        img.is_main = false;
        let h = register_module(&vm, &img);
        let mut ctx = CpuContext::new(&vm);
        let call = |ctx: &mut CpuContext, addr: u64, flags: u64, out: u64| {
            ctx.set(crate::context::CpuRegister::Rdi, addr);
            ctx.set(crate::context::CpuRegister::Rsi, flags);
            ctx.set(crate::context::CpuRegister::Rdx, out);
            cb_get_module_info_from_addr(ctx)
        };
        ctx.write_bytes(BASE, &[0x5A; MODULE_INFO_EX_SIZE + 8]);
        ctx.write_u64(BASE, 0xF_0000);
        assert_eq!(call(&mut ctx, img.base + 0x10, 2, BASE), OK);
        assert_eq!(ctx.read_u64(BASE), Some(MODULE_INFO_EX_SIZE as u64));
        assert_eq!(ctx.read_bytes(BASE + 8, 22).unwrap(), b"info-from-addr.prx\0\0\0\0");
        assert_eq!(ctx.read_u32(BASE + 0x108), Some(h as u32));
        assert_eq!(ctx.read_u64(BASE + MODULE_INFO_EX_SIZE as u64), Some(0x5A5A_5A5A_5A5A_5A5A), "nothing past the structure");
        assert_eq!(call(&mut ctx, img.base + 0x10, 3, BASE), EINVAL);
        assert_eq!(call(&mut ctx, img.base + 0x10, 2, 0), EFAULT);
        assert_eq!(call(&mut ctx, 0x7C4D_0000_0000, 2, BASE), ESRCH);
        unregister_module(h);
        vm.unmap(BASE, 0x1000).unwrap();
    }

    #[test]
    fn dlsym_order_and_main_handle() {
        let _g = REGISTRY_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let vm = ps5vm::VirtualMemory::new();
        let mut img = image(None);
        img.path = "/app0/dlsym-test-main.bin";
        img.is_main = false;
        let h = register_module(&vm, &img);
        let nid = ps5nid::compute_nid("dlsym_test_symbol");
        register_module_symbols(
            h,
            vec![(nid.clone(), SymbolKind::Object, 0x20), (nid.clone(), SymbolKind::Func, 0x10), ("rawNid".into(), SymbolKind::NoType, 0x30)],
        );
        assert_eq!(resolve_symbol(h, "dlsym_test_symbol"), Some(0x10), "functions first");
        assert_eq!(resolve_symbol(h, "rawNid"), Some(0x30), "the name itself as a NID");
        assert_eq!(resolve_symbol(h, "missing"), None);
        assert_eq!(resolve_symbol_any("dlsym_test_symbol"), Some(0x10));
        unregister_module(h);
        assert_eq!(resolve_symbol(h, "dlsym_test_symbol"), None);
    }

    /// dlopen(NULL) names the main program; dlsym finds its exports by
    /// name; failures leave a message for dlerror, which is read once.
    #[test]
    fn dlopen_main_program_and_dlerror() {
        let _g = REGISTRY_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let vm = ps5vm::VirtualMemory::new();
        let mut img = image(None);
        img.path = "/app0/dl-test-main.bin";
        img.is_main = true;
        let h = register_module(&vm, &img);
        register_module_symbols(h, vec![(ps5nid::compute_nid("dl_test_symbol"), SymbolKind::Func, 0x1234)]);
        // Mode 0 and both modes are refused.
        assert_eq!(call(cb_dlopen, &[0, 0]), 0);
        assert_eq!(call(cb_dlopen, &[0, 3]), 0);
        let msg = call(cb_dlerror, &[]) as u64;
        assert_ne!(msg, 0);
        assert_eq!(CpuContext::new(guest()).read_cstring(msg, 256).as_deref(), Some("dlopen: unsupported flags"));
        assert_eq!(call(cb_dlerror, &[]), 0, "read once");
        let handle = call(cb_dlopen, &[0, 2]) as u64;
        assert_ne!(handle, 0);
        let name = VM_BASE + 0x800;
        CpuContext::new(guest()).write_bytes(name, b"dl_test_symbol\0");
        assert_eq!(call(cb_dlsym, &[handle, name]), 0x1234);
        assert_eq!(call(cb_dlsym, &[RTLD_DEFAULT, name]), 0x1234);
        CpuContext::new(guest()).write_bytes(name, b"dl_missing\0");
        assert_eq!(call(cb_dlsym, &[handle, name]), 0);
        assert_ne!(call(cb_dlerror, &[]), 0);
        assert_eq!(call(cb_dlsym, &[0x7777, name]), 0);
        assert_eq!(call(cb_dlclose, &[handle]), 0);
        assert_eq!(call(cb_dlclose, &[handle]), -1);
        assert!(main_module().is_some(), "dlclose never unloads the main program");
        unregister_module(h);
    }
}
