// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Top-level emulator glue: the runtime linker.
//!
//! Loading a program (`map_program`):
//!   1. Parse the SELF/ELF image (PT_LOAD / PT_SCE_RELRO payloads, the
//!      dynamic table and PT_SCE_DYNLIBDATA from the file).
//!   2. Map the image (extent of PT_LOAD/PT_SCE_RELRO, each segment's size
//!      aligned to its p_align, the whole to 16 KiB) and copy the payloads.
//!   3. Register PT_TLS, record PT_SCE_PROCPARAM and PT_GNU_EH_FRAME.
//!   4. Read the dynamic metadata: `DT_SCE_*` tables live in
//!      PT_SCE_DYNLIBDATA, plain `DT_*` tables in the mapped image; library
//!      and module ids (`DT_SCE_IMPORT_LIB`, `DT_SCE_NEEDED_MODULE`, ...).
//!   5. Build the export symbol database (defined global/weak symbols whose
//!      `nid#lib#mod` ids resolve).
//!
//! Then every program is relocated (`relocator`); relocations that cannot
//! be resolved yet are retried whenever another program is loaded. Function
//! imports bind to import stubs, which the linking policy (`linker`) points
//! at HLE or at native module code. Finally guest FS accesses are redirected
//! to GS and each segment gets the protection of its program header.
//!
//! At start-up the main executable, `libc.prx` and the game's own modules
//! that the main executable lists as needed (`needed`) are loaded; other
//! modules are loaded by the application (`sceKernelLoadStartModule`).

#[cfg(windows)]
#[allow(unused_imports)]
use ps5hle::host::c as libc;
pub mod dispatcher;
pub mod game_patch;
pub mod linker;
pub mod needed;
pub mod relocator;
pub mod runtime;
pub mod stubs;
pub mod symbols;

use ps5elf::{
    dynamic::{dt, DynamicTable},
    header,
    reloc::Elf64Rela,
    parse_self, ProgramType, SelfImage,
};
use ps5hle::HleRegistry;
use ps5vm::{MemProtect, VirtualMemory};
use std::collections::HashMap;
use symbols::{LibraryId, ModuleId, SymbolDatabase, SymbolResolve, SymbolType};
use thiserror::Error;

/// PS5 guest page size (16 KiB).
const GUEST_PAGE_SIZE: u64 = 0x4000;

/// Region for HLE-provided data symbols (canary, progname, etc.).
const HLE_DATA_REGION: u64 = 0x0000_6D00_0000_0000;
/// Size of the HLE data region (one guest page).
const HLE_DATA_SIZE: u64 = GUEST_PAGE_SIZE;
/// Offset in the region of the HLE libraries' data objects
/// (`ps5hle::data_objects`); the fixed symbols sit below it.
const HLE_OBJECTS_OFFSET: u64 = 0x400;
/// Inaccessible page that unresolved data imports point at, so a use of a
/// missing object faults at a recognisable address.
pub const INVALID_MEMORY: u64 = 0x0000_6D00_0010_0000;

/// The start-up libc.prx is placed here (below the main executable at
/// 0x8_0000_0000 and direct memory); the other modules go after the main
/// image, each after the previous one (`next_module_base`).
#[cfg(unix)]
const MODULE_BASE_START: u64 = 0x0000_0001_0000_0000;
/// Windows keeps its own images and heaps around 4 GiB.
#[cfg(windows)]
const MODULE_BASE_START: u64 = 0x0000_0006_0000_0000;

/// Gap kept after each module, and the address modules must stay below.
pub(crate) const MODULE_GAP: u64 = 0x20_0000;
pub(crate) const MODULE_LIMIT: u64 = 0x9_0000_0000;

/// Base of the module placed after an image that ends at `end`.
pub(crate) fn next_module_base(end: u64) -> u64 {
    align_up(end + MODULE_GAP, MODULE_GAP)
}

#[derive(Debug, Error)]
pub enum EmuError {
    #[error("loader error: {0}")]
    Loader(#[from] ps5elf::SelfLoadError),
    #[error("vm error: {0}")]
    Vm(#[from] ps5vm::VmError),
    #[error("image has no PT_LOAD segments")]
    NoLoadSegment,
    #[error("SELF segment is encrypted; provide a decrypted (fSELF) image")]
    Encrypted,
    #[error("load error: {0}")]
    Load(String),
}

/// Resolved dynamic metadata.
#[derive(Debug, Clone, Default)]
pub struct DynamicInfo {
    pub table: DynamicTable,
    pub strtab: Vec<u8>,
    pub symtab: Vec<u8>,
    pub relas: Vec<Elf64Rela>,
    pub jmp_rels: Vec<Elf64Rela>,
    /// DT_INIT / DT_FINI (image-relative, 0 = none).
    pub init_func: u64,
    pub fini_func: u64,
    pub init_array: Vec<u64>,
    pub fini_array: Vec<u64>,
    pub preinit_array: Vec<u64>,
    pub relative_count: u64,
    pub so_name: String,
    pub import_modules: Vec<ModuleId>,
    pub export_modules: Vec<ModuleId>,
    pub import_libs: Vec<LibraryId>,
    pub export_libs: Vec<LibraryId>,
}

impl DynamicInfo {
    pub fn symbol(&self, index: u32) -> Option<ps5elf::Elf64Sym> {
        let off = index as usize * 24;
        self.symtab.get(off..off + 24).map(bytemuck::pod_read_unaligned)
    }

    pub fn name(&self, off: u32) -> String {
        symbols::c_string(&self.strtab, off as usize)
    }

    fn find_library(&self, id: &str) -> Option<&LibraryId> {
        self.import_libs.iter().find(|l| l.id == id).or_else(|| self.export_libs.iter().find(|l| l.id == id))
    }

    fn find_module(&self, id: &str) -> Option<&ModuleId> {
        self.import_modules.iter().find(|m| m.id == id).or_else(|| self.export_modules.iter().find(|m| m.id == id))
    }

    /// Full identity of symbol name `nid#lib#mod` (None when the ids do not
    /// resolve or the name has another shape).
    pub fn symbol_resolve(&self, name: &str, ty: SymbolType) -> Option<SymbolResolve> {
        let mut it = name.split('#');
        let (nid, lib, module) = (it.next()?, it.next()?, it.next()?);
        if it.next().is_some() {
            return None;
        }
        let l = self.find_library(lib)?;
        let m = self.find_module(module)?;
        Some(SymbolResolve {
            name: nid.to_string(),
            library: l.name.clone(),
            library_version: l.version,
            module: m.name.clone(),
            module_version_major: m.version_major,
            module_version_minor: m.version_minor,
            ty,
        })
    }
}

/// A mapped image segment (PT_LOAD / PT_SCE_RELRO).
#[derive(Clone, Debug)]
pub struct Segment {
    pub addr: u64,
    /// Memory size aligned to the segment alignment.
    pub size: u64,
    pub prot: MemProtect,
    /// Bytes backed by the file (`p_filesz`).
    pub file_size: u64,
    /// False for segments left with the mapping protection (next-generation
    /// PT_LOAD without access flags).
    pub protect: bool,
}

/// An executable code segment within a loaded image.
pub struct ExecSegment {
    pub base: u64,
    pub size: usize,
}

/// A loaded program: the main executable or a PRX module.
pub struct LoadedModule {
    /// Module handle (0 until registered).
    pub handle: i32,
    pub name: String,
    pub guest_path: String,
    pub host_path: String,
    pub is_shared: bool,
    pub is_next_gen: bool,
    pub base: u64,
    /// Mapped size (aligned to 16 KiB).
    pub size: u64,
    /// Extent of the segments.
    pub base_size: u64,
    pub entry: u64,
    /// DT_INIT / DT_FINI offsets (0 = none).
    pub init_func: u64,
    pub fini_func: u64,
    pub dynamic: Option<DynamicInfo>,
    pub exports: SymbolDatabase,
    /// One bit per relocation (DT_RELA entries, then DT_JMPREL): resolved.
    pub rela_bits: Vec<u8>,
    pub tls_id: u64,
    pub tls_static_offset: u64,
    pub segments: Vec<Segment>,
    pub exec_segments: Vec<ExecSegment>,
    pub eh_frame_hdr: Option<(u64, u64)>,
    pub proc_param: u64,
}

impl LoadedModule {
    /// Exported symbol address for `nid` (functions, untyped, then objects).
    pub fn resolve(&self, nid: &str) -> Option<u64> {
        [SymbolType::Func, SymbolType::NoType, SymbolType::Object]
            .into_iter()
            .find_map(|t| self.exports.find_by_nid(nid, t, 0))
            .map(|r| r.vaddr)
    }

    /// Exported data symbol (STT_OBJECT) for `nid`.
    pub fn resolve_data(&self, nid: &str) -> Option<u64> {
        self.exports.find_by_nid(nid, SymbolType::Object, 0).map(|r| r.vaddr)
    }

    /// Whether the program exports library `l` of module `m`.
    fn exports_from(&self, l: &LibraryId, m: &ModuleId) -> bool {
        let Some(d) = &self.dynamic else { return false };
        let lib_eq = |x: &LibraryId| x.name == l.name && x.version == l.version;
        let mod_eq = |x: &ModuleId| x.name == m.name && x.version_major == m.version_major && x.version_minor == m.version_minor;
        d.export_libs.iter().any(lib_eq) && d.export_modules.iter().any(mod_eq)
    }

    /// Exact (library and module aware) export lookup.
    pub fn find_export(&self, s: &SymbolResolve) -> Option<u64> {
        let l = LibraryId { id: String::new(), version: s.library_version, name: s.library.clone() };
        let m = ModuleId { id: String::new(), version_major: s.module_version_major, version_minor: s.module_version_minor, name: s.module.clone() };
        if !self.exports_from(&l, &m) {
            return None;
        }
        self.exports.find(s).map(|r| r.vaddr)
    }

    /// Relocation records not resolved yet.
    pub fn unresolved_relocations(&self) -> usize {
        let n = self.dynamic.as_ref().map_or(0, |d| d.relas.len() + d.jmp_rels.len());
        (0..n).filter(|&i| self.rela_bits[i >> 3] & (1 << (i & 7)) == 0).count()
    }

    pub fn contains(&self, addr: u64) -> bool {
        addr >= self.base && addr < self.base + self.size
    }

    /// Write `bytes` at `addr` inside this image, whatever the protection of
    /// its segment (relocations applied after the image was protected).
    pub(crate) fn patch(&self, addr: u64, bytes: &[u8]) -> bool {
        let seg = self.segments.iter().find(|s| s.protect && !s.prot.write && addr >= s.addr && addr < s.addr + s.size);
        let Some(seg) = seg else {
            if !ps5vm::ranges::covered(addr, bytes.len() as u64) {
                return false;
            }
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), addr as *mut u8, bytes.len()) };
            return true;
        };
        let page = addr & !0xFFF;
        let len = ((addr + bytes.len() as u64 + 0xFFF) & !0xFFF) - page;
        let native = native_prot(seg.prot);
        unsafe {
            if libc::mprotect(page as *mut libc::c_void, len as usize, native | libc::PROT_READ | libc::PROT_WRITE) != 0 {
                return false;
            }
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), addr as *mut u8, bytes.len());
            libc::mprotect(page as *mut libc::c_void, len as usize, native);
        }
        true
    }

    /// Exported symbols for the kernel module registry.
    fn export_list(&self) -> Vec<(String, ps5hle::modules::SymbolKind, u64)> {
        let mut out = Vec::new();
        for (ty, kind) in [
            (SymbolType::Func, ps5hle::modules::SymbolKind::Func),
            (SymbolType::Object, ps5hle::modules::SymbolKind::Object),
            (SymbolType::NoType, ps5hle::modules::SymbolKind::NoType),
        ] {
            let suffix = format!("[{ty:?}]");
            for r in self.exports.iter().filter(|r| r.name.ends_with(&suffix)) {
                let nid = r.name.split('[').next().unwrap_or_default().to_string();
                out.push((nid, kind, r.vaddr));
            }
        }
        out
    }
}

fn native_prot(p: MemProtect) -> i32 {
    (if p.read { libc::PROT_READ } else { 0 }) | (if p.write { libc::PROT_WRITE } else { 0 }) | (if p.exec { libc::PROT_EXEC } else { 0 })
}

/// Protection of a segment with ELF flags `pf` (execute implies read).
fn segment_prot(pf: u32) -> MemProtect {
    let mut p = MemProtect::from_flags(pf);
    if p.exec || p.write {
        p.read = true;
    }
    p
}

/// Result of loading the main executable and its start-up modules.
pub struct LoadedGame {
    pub image_base: u64,
    pub image_size: u64,
    pub entry: u64,
    pub image: SelfImage,
    pub vm: VirtualMemory,
    pub hle: HleRegistry,
    /// Every loaded program, the main executable first.
    pub programs: Vec<LoadedModule>,
    pub import_stubs: stubs::StubTable,
    /// Imports bound to native module code instead of HLE (NID -> target).
    pub lle_imports: HashMap<String, u64>,
    /// HLE-provided data symbols (NID -> address).
    pub hle_data: HleData,
}

impl LoadedGame {
    pub fn dynamic(&self) -> Option<&DynamicInfo> {
        self.programs.first().and_then(|p| p.dynamic.as_ref())
    }
}

fn align_up(v: u64, a: u64) -> u64 {
    if a <= 1 {
        v
    } else {
        (v + a - 1) & !(a - 1)
    }
}

/// An HLE-provided data symbol.
#[derive(Clone, Copy, Debug)]
pub struct HleDataSymbol {
    pub addr: u64,
    /// Used only when no loaded program exports the symbol.
    pub fallback: bool,
}

/// HLE data symbols by NID.
pub type HleData = HashMap<String, HleDataSymbol>;

/// HLE data symbols guest code imports.
fn hle_data_symbols(vm: &VirtualMemory, progname: &str) -> HleData {
    let _ = vm.map(HLE_DATA_REGION, HLE_DATA_SIZE, MemProtect::RW);
    let mut map = HleData::new();
    let mut add = |nid: &str, addr: u64, fallback: bool| {
        map.insert(nid.to_string(), HleDataSymbol { addr, fallback });
    };
    // __stack_chk_guard — the stack canary value.
    let canary = HLE_DATA_REGION;
    let _ = vm.write(canary, &ps5hle::tls::STACK_CANARY.to_le_bytes());
    add("f7uOxY9mM1U", canary, false);
    // __progname — pointer to the program name.
    let name_buf = HLE_DATA_REGION + 0x10;
    let mut name = progname.as_bytes()[..progname.len().min(0xFF)].to_vec();
    name.push(0);
    let _ = vm.write(name_buf, &name);
    let progname_ptr = HLE_DATA_REGION + 0x110;
    let _ = vm.write(progname_ptr, &name_buf.to_le_bytes());
    add("djxxOmW6-aw", progname_ptr, false);
    ps5hle::libc_string::set_progname_symbol(progname_ptr);
    // libc need_flag symbols — must be 1.
    for (i, nid) in ["P330P3dFF68", "ZT4ODD2Ts9o"].iter().enumerate() {
        let addr = HLE_DATA_REGION + 0x118 + i as u64 * 8;
        let _ = vm.write(addr, &1u32.to_le_bytes());
        add(nid, addr, false);
    }
    // Address sanitizer fake-stack flag (u32, 0 = off).
    add("OmecxEgrhCs", HLE_DATA_REGION + 0x130, true);
    // LibcInternal `stdout`: an opaque FILE object that the HLE stdio
    // functions recognise by address (zero-filled, 0x100 bytes).
    add("2sWzhYqFH4E", HLE_DATA_REGION + 0x200, true);
    // Data objects of the HLE libraries, each with its initial contents.
    let mut next = HLE_DATA_REGION + HLE_OBJECTS_OFFSET;
    for o in ps5hle::data_objects::objects() {
        let addr = align_up(next, o.align);
        let end = addr + o.bytes.len() as u64;
        if end > HLE_DATA_REGION + HLE_DATA_SIZE {
            log::error!("[load] no room for HLE data object {} ({})", o.name, o.library);
            break;
        }
        let _ = vm.write(addr, o.bytes);
        add(&ps5nid::compute_nid(o.name), addr, true);
        next = end;
    }
    map
}

/// Read a dynamic table: `DT_SCE_*` from PT_SCE_DYNLIBDATA, `DT_*` from the
/// mapped image. Both forms present is an error.
fn dyn_data(
    table: &DynamicTable,
    image: &SelfImage,
    vm: &VirtualMemory,
    base: u64,
    (os_ptr, std_ptr): (i64, i64),
    (os_size, std_size): (i64, i64),
) -> Result<Vec<u8>, String> {
    if table.has(os_ptr) && table.has(std_ptr) {
        return Err(format!("dynamic tags {os_ptr:#x} and {std_ptr:#x} both present"));
    }
    let size = table.get(std_size).or_else(|| table.get(os_size)).unwrap_or(0);
    if let Some(off) = table.get(os_ptr) {
        let end = off.checked_add(size).ok_or("dynamic table size overflow")?;
        return image
            .dynlib_data
            .get(off as usize..end as usize)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| format!("dynamic table {os_ptr:#x} outside PT_SCE_DYNLIBDATA"));
    }
    if let Some(ptr) = table.get(std_ptr) {
        let mut buf = vec![0u8; size as usize];
        vm.read(base + ptr, &mut buf).map_err(|e| format!("dynamic table {std_ptr:#x}: {e}"))?;
        return Ok(buf);
    }
    Ok(Vec::new())
}

fn read_ptr_array(vm: &VirtualMemory, base: u64, table: &DynamicTable, arr_tag: i64, sz_tag: i64) -> Vec<u64> {
    let (Some(off), Some(sz)) = (table.get(arr_tag), table.get(sz_tag)) else { return Vec::new() };
    let mut bytes = vec![0u8; sz as usize];
    if vm.read(base + off, &mut bytes).is_err() {
        return Vec::new();
    }
    bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn parse_relas(bytes: &[u8]) -> Vec<Elf64Rela> {
    bytes.chunks_exact(24).map(bytemuck::pod_read_unaligned).collect()
}

/// Parse the program's dynamic metadata.
fn parse_dynamic(vm: &VirtualMemory, base: u64, image: &SelfImage) -> Result<Option<DynamicInfo>, String> {
    if image.dynamic.is_empty() {
        return Ok(None);
    }
    let table = DynamicTable::parse(&image.dynamic);
    let data = |p: (i64, i64), s: (i64, i64)| dyn_data(&table, image, vm, base, p, s);
    let strtab = data((dt::DT_SCE_STRTAB, dt::DT_STRTAB), (dt::DT_SCE_STRSZ, dt::DT_STRSZ))?;
    let symtab = data((dt::DT_SCE_SYMTAB, dt::DT_SYMTAB), (dt::DT_SCE_SYMTABSZ, dt::DT_SCE_SYMTABSZ))?;
    let pltrel = table.get(dt::DT_SCE_PLTREL).or_else(|| table.get(dt::DT_PLTREL));
    if pltrel.is_some_and(|t| t != dt::DT_RELA as u64) {
        return Err(format!("unsupported PLT relocation type {pltrel:?}"));
    }
    let jmp_rels = parse_relas(&data((dt::DT_SCE_JMPREL, dt::DT_JMPREL), (dt::DT_SCE_PLTRELSZ, dt::DT_PLTRELSZ))?);
    let relas = parse_relas(&data((dt::DT_SCE_RELA, dt::DT_RELA), (dt::DT_SCE_RELASZ, dt::DT_RELASZ))?);
    for (tag, name) in [(dt::DT_DEBUG, "DT_DEBUG"), (dt::DT_TEXTREL, "DT_TEXTREL")] {
        if table.get(tag).is_some_and(|v| v != 0) {
            log::warn!("[load] {name} is set (not supported)");
        }
    }
    let strs = strtab.clone();
    let modules = |tags: [i64; 2]| tags.iter().flat_map(|&t| table.get_all(t)).map(|v| symbols::module_id(v, &strs)).collect::<Vec<_>>();
    let libs = |tags: [i64; 2]| tags.iter().flat_map(|&t| table.get_all(t)).map(|v| symbols::library_id(v, &strs)).collect::<Vec<_>>();
    Ok(Some(DynamicInfo {
        init_func: table.get(dt::DT_INIT).unwrap_or(0),
        fini_func: table.get(dt::DT_FINI).unwrap_or(0),
        init_array: read_ptr_array(vm, base, &table, dt::DT_INIT_ARRAY, dt::DT_INIT_ARRAYSZ),
        fini_array: read_ptr_array(vm, base, &table, dt::DT_FINI_ARRAY, dt::DT_FINI_ARRAYSZ),
        preinit_array: read_ptr_array(vm, base, &table, dt::DT_PREINIT_ARRAY, dt::DT_PREINIT_ARRAYSZ),
        relative_count: table.get(dt::DT_RELACOUNT).unwrap_or(0),
        so_name: table.get(dt::DT_SONAME).map(|o| symbols::c_string(&strtab, o as usize)).unwrap_or_default(),
        import_modules: modules([dt::DT_SCE_NEEDED_MODULE, dt::DT_SCE_NEEDED_MODULE_1]),
        export_modules: modules([dt::DT_SCE_MODULE_INFO, dt::DT_SCE_MODULE_INFO_1]),
        import_libs: libs([dt::DT_SCE_IMPORT_LIB, dt::DT_SCE_IMPORT_LIB_1]),
        export_libs: libs([dt::DT_SCE_EXPORT_LIB, dt::DT_SCE_EXPORT_LIB_1]),
        table,
        strtab,
        symtab,
        relas,
        jmp_rels,
    }))
}

/// Export database: defined global/weak function, object and untyped
/// symbols whose library and module ids resolve.
fn build_exports(base: u64, d: &DynamicInfo) -> SymbolDatabase {
    let mut db = SymbolDatabase::new();
    for i in 0..(d.symtab.len() / 24) as u32 {
        let Some(sym) = d.symbol(i) else { break };
        let bind = sym.bind();
        if bind != ps5elf::STB_GLOBAL && bind != ps5elf::STB_WEAK || sym.st_value == 0 {
            continue;
        }
        let ty = SymbolType::from_st_type(sym.type_());
        if ty == SymbolType::Unknown {
            continue;
        }
        if let Some(s) = d.symbol_resolve(&d.name(sym.st_name), ty) {
            db.add(&s, base + sym.st_value);
        }
    }
    db
}

/// Map program `host_path` at `base` and read its metadata (not relocated).
pub fn map_program(vm: &VirtualMemory, host_path: &std::path::Path, guest_path: &str, base: u64) -> Result<(LoadedModule, SelfImage), EmuError> {
    let bytes = std::fs::read(host_path).map_err(|e| EmuError::Load(format!("read {}: {e}", host_path.display())))?;
    let mut image = parse_self(&bytes)?;
    drop(bytes);
    let is_shared = image.is_shared();
    if !is_shared && !image.is_next_gen() {
        return Err(EmuError::Load("main executable is not a PS5 (next-generation) image".into()));
    }
    let mapped: Vec<_> = image.segments.iter().filter(|s| s.p_type.is_mapped() && s.memsz != 0).collect();
    if mapped.is_empty() {
        return Err(EmuError::NoLoadSegment);
    }
    let base_size = mapped.iter().map(|s| s.vaddr + align_up(s.memsz, s.align)).max().unwrap_or(0);
    let size = align_up(base_size, GUEST_PAGE_SIZE);
    vm.map(base, size, MemProtect::RWX)?;

    // Everything after the mapping is undone when the load fails: the
    // mapping is released and the module's TLS unregistered.
    let mut tls = (0, 0);
    let body = |tls: &mut (u64, u64), image: &mut SelfImage| -> Result<LoadedModule, EmuError> {
        let mapped: Vec<_> = image.segments.iter().filter(|s| s.p_type.is_mapped() && s.memsz != 0).collect();
        let mut segments = Vec::new();
        let mut exec_segments = Vec::new();
        for s in &mapped {
            if !s.data.is_empty() {
                vm.write(base + s.vaddr, &s.data)?;
            }
            let prot = segment_prot(s.p_flags);
            let no_access = s.p_flags & 7 == 0;
            segments.push(Segment {
                addr: base + s.vaddr,
                size: align_up(s.memsz, s.align),
                prot,
                file_size: s.filesz,
                protect: !(s.p_type == ProgramType::Load && image.is_next_gen() && no_access),
            });
            if s.p_flags & ps5elf::elf::PF_X != 0 && s.filesz != 0 {
                exec_segments.push(ExecSegment { base: base + s.vaddr, size: s.filesz as usize });
            }
        }

        let mut proc_param = 0;
        let mut eh_frame_hdr = None;
        for p in &image.phdrs {
            match ProgramType::from_raw(p.p_type) {
                ProgramType::Tls => {
                    if p.p_vaddr >= base_size {
                        return Err(EmuError::Load("PT_TLS outside the image".into()));
                    }
                    match ps5hle::tls::register_module(p.p_align, p.p_memsz, p.p_filesz, base + p.p_vaddr, p.p_vaddr) {
                        Some(t) => *tls = t,
                        None if p.p_memsz != 0 => return Err(EmuError::Load("cannot register the module's TLS".into())),
                        None => {}
                    }
                }
                ProgramType::SceProcParam => {
                    if proc_param != 0 || p.p_vaddr >= base_size {
                        return Err(EmuError::Load("bad PT_SCE_PROCPARAM".into()));
                    }
                    proc_param = base + p.p_vaddr;
                }
                ProgramType::Other(ps5elf::elf::PT_GNU_EH_FRAME) if p.p_memsz != 0 => {
                    eh_frame_hdr = Some((base + p.p_vaddr, p.p_memsz));
                }
                _ => {}
            }
        }

        let dynamic = parse_dynamic(vm, base, &image).map_err(EmuError::Load)?;
        let exports = dynamic.as_ref().map(|d| build_exports(base, d)).unwrap_or_default();
        let reloc_count = dynamic.as_ref().map_or(0, |d| d.relas.len() + d.jmp_rels.len());
        let name = host_path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        log::info!(
            "[load] {name}: base={base:#x} size={size:#x} segments={} exports={} relocations={reloc_count}",
            segments.len(),
            exports.len()
        );
        // The payloads are in guest memory now.
        for s in &mut image.segments {
            s.data = Vec::new();
        }
        let module = LoadedModule {
            handle: 0,
            name,
            guest_path: guest_path.to_string(),
            host_path: host_path.to_string_lossy().into_owned(),
            is_shared,
            is_next_gen: image.is_next_gen(),
            base,
            size,
            base_size,
            entry: image.entry,
            init_func: dynamic.as_ref().map_or(0, |d| d.init_func),
            fini_func: dynamic.as_ref().map_or(0, |d| d.fini_func),
            exports,
            rela_bits: vec![0; reloc_count.div_ceil(8)],
            dynamic,
            tls_id: tls.0,
            tls_static_offset: tls.1,
            segments,
            exec_segments,
            eh_frame_hdr,
            proc_param,
        };
        Ok(module)
    };
    match body(&mut tls, &mut image) {
        Ok(module) => Ok((module, image)),
        Err(e) => {
            let _ = vm.unmap(base, size);
            if tls.0 != 0 {
                ps5hle::tls::unregister_module(tls.0);
            }
            Err(e)
        }
    }
}

/// Register a relocated program with the kernel module registry.
fn register_program(vm: &VirtualMemory, p: &mut LoadedModule, is_main: bool) {
    if p.tls_id != 0 {
        ps5hle::tls::set_init_image(vm, p.tls_id);
    }
    let seg_start = p.segments.iter().map(|s| s.addr).min().unwrap_or(p.base);
    let seg_end = p.segments.iter().map(|s| s.addr + s.size).max().unwrap_or(p.base);
    let at = |off: u64| if off != 0 { p.base + off } else { 0 };
    let so_name = p.dynamic.as_ref().map(|d| d.so_name.clone()).unwrap_or_default();
    let segments: Vec<(u64, u64, i32)> = p
        .segments
        .iter()
        .map(|s| (s.addr, s.size, s.prot.read as i32 | (s.prot.write as i32) << 1 | (s.prot.exec as i32) << 2))
        .collect();
    p.handle = ps5hle::modules::register_module(
        vm,
        &ps5hle::modules::ModuleImage {
            path: &p.guest_path,
            host_path: &p.host_path,
            so_name: &so_name,
            base: p.base,
            size: p.size,
            seg_start,
            seg_end,
            entry_point: if is_main { p.base + p.entry } else { at(p.init_func) },
            init_entry_point: at(p.init_func),
            fini_entry_point: at(p.fini_func),
            eh_frame_hdr_segment: p.eh_frame_hdr,
            segments: &segments,
            is_main,
        },
    );
    ps5hle::modules::register_module_symbols(p.handle, p.export_list());
}

/// Redirect guest FS accesses to GS, replace console-only instructions the
/// host lacks where an equally long native form exists, publish the code
/// ranges and apply the segment protections (last step of loading a program).
fn finish_program(vm: &VirtualMemory, p: &LoadedModule) -> usize {
    let mut patched = 0;
    let replace = ps5cpu::code_patch::Replace::for_host();
    // Windows builds exception frames over the System V red zone: functions
    // that keep data there move to copies that run with RSP below it.
    #[cfg(windows)]
    if let Some((hdr, _)) = p.eh_frame_hdr {
        let segs: Vec<(u64, usize)> = p.exec_segments.iter().map(|s| (s.base, s.size)).collect();
        // SAFETY: the module is mapped and its code is writable; no thread
        // runs it yet.
        unsafe { ps5cpu::redzone::protect_module(&p.name, hdr, p.base, p.base + p.size, &segs, replace) };
    }
    for seg in &p.exec_segments {
        let c = unsafe { ps5cpu::code_patch::patch_guest_code(seg.base, seg.size, replace) };
        if c.monitorx + c.mwaitx + c.movnt_scalar != 0 {
            log::info!("[load] {}: replaced for this host: monitorx={} mwaitx={} movntss/sd={}", p.name, c.monitorx, c.mwaitx, c.movnt_scalar);
        }
        patched += c.fs_to_gs;
        ps5cpu::exec::add_guest_code_range(seg.base, seg.size as u64, &p.name);
    }
    for s in p.segments.iter().filter(|s| s.protect) {
        let start = s.addr & !0xFFF;
        let end = (s.addr + s.size + 0xFFF) & !0xFFF;
        if let Err(e) = vm.protect(start, end - start, s.prot) {
            log::warn!("[load] {}: protect {:#x}+{:#x}: {e}", p.name, start, end - start);
        }
    }
    patched
}

/// Load and map a decrypted SELF / bare ELF main executable with libc.prx,
/// relocate and link them.
pub fn load_eboot(_bytes: &[u8], path: &str) -> Result<LoadedGame, EmuError> {
    // /app0 root, title id and language for the system-service HLE.
    ps5hle::app_info::configure_from_eboot(std::path::Path::new(path));
    let eboot_dir = std::path::Path::new(path).parent().unwrap_or(std::path::Path::new("."));
    ps5hle::filesystem::set_app0_root(eboot_dir);

    let vm = VirtualMemory::new();
    let file_name = std::path::Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "eboot.bin".into());
    let (main, image) = map_program(&vm, std::path::Path::new(path), &format!("/app0/{file_name}"), header::PS5_MAIN_IMAGE_BASE)?;
    if main.proc_param != 0 {
        ps5hle::kernel::set_proc_param(main.proc_param);
        log::info!("[load] ProcParam at {:#x}", main.proc_param);
    }
    let mut programs = vec![main];

    // The runtime loads libc itself.
    if let Some(libc) = ps5hle::filesystem::resolve_guest_path("/app0/sce_module/libc.prx").filter(|p| p.is_file()) {
        match map_program(&vm, &libc, "/app0/sce_module/libc.prx", MODULE_BASE_START) {
            Ok((m, _)) => programs.push(m),
            Err(e) => log::warn!("[load] libc.prx: {e}"),
        }
    }
    // The game's own modules that the main executable lists as needed are
    // loaded with it, as the system's loader does for DT_NEEDED: before
    // relocation, so that every import can bind to their exports.
    needed::load_needed_modules(&vm, &mut programs);

    let hle_data = hle_data_symbols(&vm, &file_name);
    let _ = vm.map(INVALID_MEMORY, GUEST_PAGE_SIZE, MemProtect::NONE);
    let hle = HleRegistry::with_defaults();
    let mut import_stubs = stubs::StubTable::new();
    import_stubs.set_hle_functions(hle.exports().filter(|&(_, real)| real).map(|(e, _)| e.nid.to_string()));
    relocator::relocate_all(&mut programs, &vm, &mut import_stubs, &hle_data);

    for (i, p) in programs.iter_mut().enumerate() {
        register_program(&vm, p, i == 0);
    }
    for p in programs.iter().skip(1) {
        ps5hle::modules::hold_start_reference(p.handle);
    }

    // Bind function imports: HLE callbacks by default, native module code
    // where the linking policy prefers it.
    let lle_imports = linker::link_native_imports(&vm, &import_stubs, &programs, &hle);
    log::info!("[load] linked {} imports to native module code, {} stay HLE", lle_imports.len(), import_stubs.count() - lle_imports.len());

    // Game patches go in after relocation, before the code is finalized.
    if let Some(plan) = std::env::var_os(game_patch::PATCH_ENV).filter(|p| !p.is_empty()) {
        if let Err(e) = game_patch::apply(std::path::Path::new(&plan), &vm, &programs) {
            log::error!("Game cheat error: {e}");
            return Err(EmuError::Load(format!("failed to apply game cheat: {e}")));
        }
    }

    let patched: usize = programs.iter().map(|p| finish_program(&vm, p)).sum();
    log::info!("[tls] redirected {patched} FS-relative instructions to GS");

    let main = &programs[0];
    Ok(LoadedGame {
        image_base: main.base,
        image_size: main.size,
        entry: main.base + main.entry,
        image,
        vm,
        hle,
        programs,
        import_stubs,
        lle_imports,
        hle_data,
    })
}
