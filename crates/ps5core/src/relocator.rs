// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Relocation processor.
//!
//! Applies DT_RELA then DT_JMPREL entries of a program and remembers which
//! ones resolved (`LoadedModule::rela_bits`); unresolved ones are retried
//! each time the set of programs changes (`relocate_all`). Handles:
//!   - relative relocations (B + A);
//!   - TLS relocations (DTPMOD64 = module id, DTPOFF64, TPOFF64);
//!   - symbol relocations. GLOB_DAT and JUMP_SLOT ignore the addend. Local
//!     symbols and symbols the program defines bind to the program itself.
//!     Imported data (STT_OBJECT) binds to the exporting program: HLE data
//!     symbols first, then an exact library/module match, then any export
//!     with the NID; until one exists it points at an inaccessible page.
//!     Imported untyped symbols bind to an exact export when there is one.
//!     Imported functions bind to the NID's import stub (HLE or native, see
//!     `linker`).

use crate::{stubs::StubTable, symbols::SymbolType, LoadedModule, INVALID_MEMORY};
use ps5elf::reloc::*;
use ps5elf::{compute_value, value_kind, write_kind, ValueKind, WriteKind};
use ps5vm::VirtualMemory;

/// Outcome of one relocation record.
struct Applied {
    addr: u64,
    value: u64,
    kind: WriteKind,
    resolved: bool,
}

/// Imported data symbol `name` (`nid#lib#mod`) of program `importer`.
fn resolve_object(importer: &LoadedModule, programs: &[LoadedModule], hle_data: &crate::HleData, name: &str, ty: SymbolType) -> Option<u64> {
    let nid = name.split('#').next().unwrap_or(name);
    let hle = hle_data.get(nid);
    if let Some(h) = hle.filter(|h| !h.fallback) {
        return Some(h.addr);
    }
    if let Some(s) = importer.dynamic.as_ref().and_then(|d| d.symbol_resolve(name, ty)) {
        if let Some(a) = programs.iter().find_map(|p| p.find_export(&s)) {
            return Some(a);
        }
    }
    programs.iter().find_map(|p| p.exports.find_by_nid(nid, ty, 0)).map(|r| r.vaddr).or(hle.map(|h| h.addr))
}

/// Compute relocation `rela` of program `idx`.
fn relocate_record(
    idx: usize,
    rela: &Elf64Rela,
    programs: &[LoadedModule],
    stubs: &mut StubTable,
    vm: &VirtualMemory,
    hle_data: &crate::HleData,
) -> Option<Applied> {
    let p = &programs[idx];
    let d = p.dynamic.as_ref()?;
    let rtype = rela.rtype();
    let addr = p.base.wrapping_add(rela.r_offset);
    if !is_supported(rtype) {
        log::error!("[reloc] {}: unsupported relocation type {rtype} at {addr:#x}", p.name);
        return None;
    }
    let done = |value: u64, kind: WriteKind| Some(Applied { addr, value, kind, resolved: true });
    match rtype {
        R_X86_64_NONE => return None,
        R_X86_64_RELATIVE | R_X86_64_RELATIVE64 => return done(add_signed(p.base, rela.r_addend), WriteKind::UInt64),
        R_X86_64_DTPMOD64 => return done(p.tls_id, WriteKind::UInt64),
        _ => {}
    }
    let sym = if rela.sym() == 0 { Elf64Sym::default() } else { d.symbol(rela.sym())? };
    match rtype {
        R_X86_64_DTPOFF64 => return done(add_signed(sym.st_value, rela.r_addend), WriteKind::UInt64),
        R_X86_64_TPOFF64 => {
            if p.tls_id != 0 && ps5hle::tls::is_dynamic(p.tls_id) {
                log::error!("[reloc] {}: static TLS relocation in a module with dynamic TLS", p.name);
            }
            return done(add_signed(sym.st_value, rela.r_addend).wrapping_sub(p.tls_static_offset), WriteKind::UInt64);
        }
        _ => {}
    }
    let addend = if rtype == R_X86_64_GLOB_DAT || rtype == R_X86_64_JUMP_SLOT { 0 } else { rela.r_addend };
    let vk = value_kind(rtype);
    let wk = write_kind(rtype);
    if vk == ValueKind::SymbolSize {
        return done(compute_value(vk, sym.st_size, addend, addr), wk);
    }
    let bind = sym.bind();
    let (s, resolved) = if rela.sym() == 0 {
        (0, true)
    } else if bind == STB_LOCAL || sym.st_value != 0 {
        (p.base.wrapping_add(sym.st_value), true)
    } else if bind == STB_GLOBAL || bind == STB_WEAK {
        let name = d.name(sym.st_name);
        let nid = name.split('#').next().unwrap_or_default();
        match SymbolType::from_st_type(sym.type_()) {
            SymbolType::Object => match resolve_object(p, programs, hle_data, &name, SymbolType::Object) {
                Some(a) => (a, true),
                None => (INVALID_MEMORY, false),
            },
            ty => {
                let exact = (ty == SymbolType::NoType)
                    .then(|| d.symbol_resolve(&name, ty).and_then(|s| programs.iter().find_map(|q| q.find_export(&s))))
                    .flatten();
                // A weak import binds like a global one when something
                // defines it (an HLE implementation or a module export).
                let defined = |nid: &str| stubs.has_hle_function(nid) || programs.iter().any(|q| q.resolve(nid).is_some_and(|a| a != 0));
                match exact {
                    Some(a) => (a, true),
                    None if nid.is_empty() => (0, false),
                    // gABI: an undefined weak symbol without a definition is
                    // 0; it stays unresolved so a module loaded later that
                    // defines it is bound.
                    None if bind == STB_WEAK && !defined(nid) => (0, false),
                    None => match stubs.ensure_stub(vm, nid) {
                        Ok(a) => (a, true),
                        Err(e) => {
                            log::warn!("[reloc] {}: no stub for {nid}: {e}", p.name);
                            (0, false)
                        }
                    },
                }
            }
        }
    } else {
        log::warn!("[reloc] {}: symbol {} has unsupported binding {bind}", p.name, rela.sym());
        return None;
    };
    Some(Applied { addr, value: compute_value(vk, s, addend, addr), kind: wk, resolved })
}

fn write(p: &LoadedModule, a: &Applied) -> bool {
    match a.kind {
        WriteKind::UInt64 => p.patch(a.addr, &a.value.to_le_bytes()),
        WriteKind::UInt32 | WriteKind::Int32 => p.patch(a.addr, &(a.value as u32).to_le_bytes()),
    }
}

/// Number of relocation records of a program.
fn count(p: &LoadedModule) -> usize {
    p.dynamic.as_ref().map_or(0, |d| d.relas.len() + d.jmp_rels.len())
}

/// Relocation record `i` (DT_RELA entries first, then DT_JMPREL).
fn record(p: &LoadedModule, i: usize) -> Elf64Rela {
    let d = p.dynamic.as_ref().expect("record of a program without dynamic info");
    if i < d.relas.len() { d.relas[i] } else { d.jmp_rels[i - d.relas.len()] }
}

fn bit(p: &LoadedModule, i: usize) -> bool {
    p.rela_bits[i >> 3] & (1 << (i & 7)) != 0
}

/// Apply the unresolved relocations of program `idx`; returns how many
/// resolved now.
pub fn relocate(idx: usize, programs: &mut [LoadedModule], vm: &VirtualMemory, stubs: &mut StubTable, hle_data: &crate::HleData) -> usize {
    let mut newly = Vec::new();
    let mut unresolved = 0;
    {
        let p = &programs[idx];
        for i in 0..count(p) {
            if bit(p, i) {
                continue;
            }
            let rela = record(p, i);
            if rela.rtype() == R_X86_64_NONE {
                newly.push(i);
                continue;
            }
            let Some(a) = relocate_record(idx, &rela, programs, stubs, vm, hle_data) else { continue };
            if !write(p, &a) {
                log::warn!("[reloc] {}: cannot write relocation at {:#x}", p.name, a.addr);
                continue;
            }
            if a.resolved {
                newly.push(i);
            } else {
                unresolved += 1;
            }
        }
    }
    let p = &mut programs[idx];
    for &i in &newly {
        p.rela_bits[i >> 3] |= 1 << (i & 7);
    }
    log::info!("[reloc] {}: {} relocations applied, {} unresolved", p.name, newly.len(), unresolved);
    newly.len()
}

/// Relocate every program (load order), retrying what did not resolve.
pub fn relocate_all(programs: &mut [LoadedModule], vm: &VirtualMemory, stubs: &mut StubTable, hle_data: &crate::HleData) {
    for i in 0..programs.len() {
        relocate(i, programs, vm, stubs, hle_data);
    }
}

/// Before program `gone` is unmapped: every symbol relocation of the other
/// programs that points into it is marked unresolved and applied again
/// (rebinding to another export, a stub or the inaccessible page).
pub fn unbind(gone: usize, programs: &mut [LoadedModule], vm: &VirtualMemory, stubs: &mut StubTable, hle_data: &crate::HleData) {
    let (lo, hi) = (programs[gone].base, programs[gone].base + programs[gone].size);
    for idx in 0..programs.len() {
        if idx == gone {
            continue;
        }
        let mut cleared = 0;
        {
            let p = &mut programs[idx];
            for i in 0..count(p) {
                if !bit(p, i) {
                    continue;
                }
                let rela = record(p, i);
                let t = rela.rtype();
                if t != R_X86_64_64 && t != R_X86_64_GLOB_DAT && t != R_X86_64_JUMP_SLOT {
                    continue;
                }
                let Some(sym) = p.dynamic.as_ref().and_then(|d| d.symbol(rela.sym())) else { continue };
                if sym.bind() != STB_GLOBAL && sym.bind() != STB_WEAK {
                    continue;
                }
                let slot = p.base.wrapping_add(rela.r_offset);
                if !ps5vm::ranges::covered(slot, 8) {
                    continue;
                }
                let value = unsafe { std::ptr::read_unaligned(slot as *const u64) };
                let target = if t == R_X86_64_64 { value.wrapping_sub(rela.r_addend as u64) } else { value };
                if target >= lo && target < hi {
                    p.rela_bits[i >> 3] &= !(1 << (i & 7));
                    cleared += 1;
                }
            }
        }
        if cleared != 0 {
            log::info!("[reloc] {}: {cleared} references into {} unbound", programs[idx].name, programs[gone].name);
        }
    }
    // The departing program no longer exports anything.
    programs[gone].exports = Default::default();
    for idx in 0..programs.len() {
        if idx != gone {
            relocate(idx, programs, vm, stubs, hle_data);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbols::SymbolDatabase;
    use crate::DynamicInfo;
    use ps5elf::dynamic::DynamicTable;
    use ps5elf::{Elf64Rela, Elf64Sym};

    const STB_WEAK_BIND: u8 = STB_WEAK;
    const STB_GLOBAL_BIND: u8 = STB_GLOBAL;
    const STT_FUNC: u8 = 2;

    /// A module whose symbol 1 is an undefined function import `name`.
    fn module(name: &str, bind: u8) -> LoadedModule {
        let mut strtab = vec![0u8];
        strtab.extend_from_slice(name.as_bytes());
        strtab.push(0);
        let mut symtab = vec![0u8; 24];
        let sym = Elf64Sym { st_name: 1, st_info: (bind << 4) | STT_FUNC, st_other: 0, st_shndx: 0, st_value: 0, st_size: 0 };
        symtab.extend_from_slice(bytemuck::bytes_of(&sym));
        LoadedModule {
            handle: 0,
            name: "test".into(),
            guest_path: String::new(),
            host_path: String::new(),
            is_shared: true,
            is_next_gen: true,
            base: 0x1000_0000,
            size: 0x4000,
            base_size: 0x4000,
            entry: 0,
            init_func: 0,
            fini_func: 0,
            dynamic: Some(DynamicInfo {
                table: DynamicTable::default(),
                strtab,
                symtab,
                relas: Vec::new(),
                jmp_rels: Vec::new(),
                init_func: 0,
                fini_func: 0,
                init_array: Vec::new(),
                fini_array: Vec::new(),
                preinit_array: Vec::new(),
                relative_count: 0,
                so_name: String::new(),
                import_modules: Vec::new(),
                export_modules: Vec::new(),
                import_libs: Vec::new(),
                export_libs: Vec::new(),
            }),
            exports: SymbolDatabase::new(),
            rela_bits: Vec::new(),
            tls_id: 0,
            tls_static_offset: 0,
            segments: Vec::new(),
            exec_segments: Vec::new(),
            eh_frame_hdr: None,
            proc_param: 0,
        }
    }

    fn rela(sym: u32, rtype: u32) -> Elf64Rela {
        Elf64Rela { r_offset: 0x100, r_info: ((sym as u64) << 32) | rtype as u64, r_addend: 0 }
    }

    #[test]
    fn undefined_weak_import_resolves_to_zero() {
        let programs = vec![module("AAAAAAAAAAA#lib#mod", STB_WEAK_BIND)];
        let vm = VirtualMemory::new();
        let mut stubs = StubTable::new();
        let hle = crate::HleData::new();
        let a = relocate_record(0, &rela(1, R_X86_64_JUMP_SLOT), &programs, &mut stubs, &vm, &hle).expect("applied");
        assert_eq!(a.value, 0, "an undefined weak symbol is null, not an import stub");
        assert!(!a.resolved, "it is retried when another module loads");
        assert_eq!(stubs.count(), 0, "no stub is made for it");
    }

    /// `__cxa_finalize` is a weak function import of every game module; the
    /// HLE implements it, so the slot gets its import stub, not 0.
    #[test]
    fn weak_import_with_an_hle_implementation_binds_to_its_stub() {
        let nid = "H2e8t5ScQGc";
        let programs = vec![module(&format!("{nid}#lib#mod"), STB_WEAK_BIND)];
        let vm = VirtualMemory::new();
        let mut stubs = StubTable::new();
        stubs.set_hle_functions([nid.to_string()]);
        let hle = crate::HleData::new();
        let a = relocate_record(0, &rela(1, R_X86_64_JUMP_SLOT), &programs, &mut stubs, &vm, &hle).expect("applied");
        assert!(a.resolved);
        assert_eq!(Some(a.value), stubs.lookup(nid), "the weak import is bound to the import stub");
        assert_ne!(a.value, 0);
    }

    #[test]
    fn unsupported_relocation_type_is_not_written() {
        let programs = vec![module("AAAAAAAAAAA#lib#mod", STB_GLOBAL_BIND)];
        let vm = VirtualMemory::new();
        let mut stubs = StubTable::new();
        let hle = crate::HleData::new();
        // R_X86_64_IRELATIVE (37) and R_X86_64_TLSDESC (36) are not handled.
        for rtype in [36, 37, 23] {
            assert!(relocate_record(0, &rela(0, rtype), &programs, &mut stubs, &vm, &hle).is_none(), "type {rtype}");
        }
    }
}
