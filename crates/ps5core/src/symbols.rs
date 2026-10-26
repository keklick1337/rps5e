// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Symbol identities and per-program symbol databases.
//!
//! An Orbis dynamic symbol is named `nid#lib#mod`, where `lib`/`mod` are
//! short ids that the image's `DT_SCE_IMPORT_LIB`/`DT_SCE_EXPORT_LIB` and
//! `DT_SCE_NEEDED_MODULE`/`DT_SCE_MODULE_INFO` entries map to a library or
//! module name and version. A symbol is identified by the NID, library name
//! and version, module name and version, and its type; the database keys
//! records by the generated name `nid[lib_vN][mod_vM.m][Type]` (a `libSce`
//! prefix is dropped from the names).

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SymbolType {
    Unknown,
    Func,
    Object,
    TlsModule,
    NoType,
}

impl SymbolType {
    pub fn from_st_type(t: u8) -> Self {
        match t {
            ps5elf::reloc::STT_NOTYPE => Self::NoType,
            ps5elf::reloc::STT_FUNC => Self::Func,
            ps5elf::reloc::STT_OBJECT => Self::Object,
            _ => Self::Unknown,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Unknown => "Unknown",
            Self::Func => "Func",
            Self::Object => "Object",
            Self::TlsModule => "TlsModule",
            Self::NoType => "NoType",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleId {
    pub id: String,
    pub version_major: u32,
    pub version_minor: u32,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LibraryId {
    pub id: String,
    pub version: u32,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolResolve {
    pub name: String,
    pub library: String,
    pub library_version: u32,
    pub module: String,
    pub module_version_major: u32,
    pub module_version_minor: u32,
    pub ty: SymbolType,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SymbolRecord {
    pub name: String,
    pub vaddr: u64,
}

/// Encode a 16-bit library/module id as used in symbol names
/// (base64-like digits, most significant first, 1 to 3 digits).
pub fn encode_id64(id: u16) -> String {
    const DIGITS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    let d = |v: u16| DIGITS[(v & 0x3F) as usize] as char;
    let mut s = String::with_capacity(3);
    if id < 0x40 {
        s.push(d(id));
    } else if id < 0x1000 {
        s.push(d(id >> 6));
        s.push(d(id));
    } else {
        s.push(d(id >> 12));
        s.push(d(id >> 6));
        s.push(d(id));
    }
    s
}

/// Module id from a `DT_SCE_NEEDED_MODULE`/`DT_SCE_MODULE_INFO` value.
pub fn module_id(value: u64, strtab: &[u8]) -> ModuleId {
    ModuleId {
        id: encode_id64((value >> 48) as u16),
        version_major: ((value >> 40) & 0xFF) as u32,
        version_minor: ((value >> 32) & 0xFF) as u32,
        name: c_string(strtab, (value & 0xFFFF_FFFF) as usize),
    }
}

/// Library id from a `DT_SCE_IMPORT_LIB`/`DT_SCE_EXPORT_LIB` value.
pub fn library_id(value: u64, strtab: &[u8]) -> LibraryId {
    LibraryId {
        id: encode_id64((value >> 48) as u16),
        version: ((value >> 32) & 0xFFFF) as u32,
        name: c_string(strtab, (value & 0xFFFF_FFFF) as usize),
    }
}

/// NUL-terminated string at `off` (empty when out of range).
pub fn c_string(strtab: &[u8], off: usize) -> String {
    let Some(tail) = strtab.get(off..) else { return String::new() };
    let end = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
    String::from_utf8_lossy(&tail[..end]).into_owned()
}

fn short_name(s: &str) -> &str {
    s.strip_prefix("libSce").unwrap_or(s)
}

/// `nid[lib_vN][mod_vM.m][Type]`.
pub fn generate_name(s: &SymbolResolve) -> String {
    format!(
        "{}[{}_v{}][{}_v{}.{}][{}]",
        s.name,
        short_name(&s.library),
        s.library_version,
        short_name(&s.module),
        s.module_version_major,
        s.module_version_minor,
        s.ty.name()
    )
}

/// Symbols of one program (exports or imports), in insertion order, with an
/// index by generated name and by (NID, type).
#[derive(Default)]
pub struct SymbolDatabase {
    symbols: Vec<SymbolRecord>,
    by_name: HashMap<String, usize>,
    /// First record per (NID, type), in insertion order.
    by_nid: HashMap<(String, SymbolType), Vec<usize>>,
}

impl SymbolDatabase {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, s: &SymbolResolve, vaddr: u64) {
        let name = generate_name(s);
        let index = self.symbols.len();
        self.by_name.insert(name.clone(), index);
        self.by_nid.entry((s.name.clone(), s.ty)).or_default().push(index);
        self.symbols.push(SymbolRecord { name, vaddr });
    }

    /// Exact match (NID, library, module, versions, type).
    pub fn find(&self, s: &SymbolResolve) -> Option<&SymbolRecord> {
        self.by_name.get(&generate_name(s)).map(|&i| &self.symbols[i])
    }

    /// First record with this NID and type (and address, when non-zero).
    pub fn find_by_nid(&self, nid: &str, ty: SymbolType, vaddr: u64) -> Option<&SymbolRecord> {
        let list = self.by_nid.get(&(nid.to_string(), ty))?;
        list.iter().map(|&i| &self.symbols[i]).find(|r| vaddr == 0 || r.vaddr == vaddr)
    }

    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &SymbolRecord> {
        self.symbols.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sr(nid: &str, lib: &str, ty: SymbolType) -> SymbolResolve {
        SymbolResolve {
            name: nid.into(),
            library: lib.into(),
            library_version: 1,
            module: lib.into(),
            module_version_major: 1,
            module_version_minor: 1,
            ty,
        }
    }

    #[test]
    fn encodes_ids() {
        assert_eq!(encode_id64(0), "A");
        assert_eq!(encode_id64(1), "B");
        assert_eq!(encode_id64(0x3F), "-");
        assert_eq!(encode_id64(0x40), "BA");
        assert_eq!(encode_id64(0xFFF), "--");
        assert_eq!(encode_id64(0x1000), "BAA");
    }

    #[test]
    fn decodes_dynamic_ids() {
        let strtab = b"\0libc\0libSceFios2\0";
        let m = module_id(0x0002_0103_0000_0001, strtab);
        assert_eq!((m.id.as_str(), m.version_major, m.version_minor, m.name.as_str()), ("C", 1, 3, "libc"));
        let l = library_id(0x0001_0001_0000_0006, strtab);
        assert_eq!((l.id.as_str(), l.version, l.name.as_str()), ("B", 1, "libSceFios2"));
    }

    #[test]
    fn generated_names_drop_the_sce_prefix() {
        let s = sr("abc", "libSceFios2", SymbolType::Func);
        assert_eq!(generate_name(&s), "abc[Fios2_v1][Fios2_v1.1][Func]");
        assert_eq!(generate_name(&sr("x", "libc", SymbolType::Object)), "x[libc_v1][libc_v1.1][Object]");
    }

    #[test]
    fn database_lookups() {
        let mut db = SymbolDatabase::new();
        db.add(&sr("n1", "libA", SymbolType::Func), 0x1000);
        db.add(&sr("n1", "libB", SymbolType::Func), 0x2000);
        db.add(&sr("n1", "libB", SymbolType::Object), 0x3000);
        // Exact identity.
        assert_eq!(db.find(&sr("n1", "libB", SymbolType::Func)).unwrap().vaddr, 0x2000);
        assert!(db.find(&sr("n1", "libC", SymbolType::Func)).is_none());
        // NID fallback: first record of the type, optionally at an address.
        assert_eq!(db.find_by_nid("n1", SymbolType::Func, 0).unwrap().vaddr, 0x1000);
        assert_eq!(db.find_by_nid("n1", SymbolType::Func, 0x2000).unwrap().vaddr, 0x2000);
        assert_eq!(db.find_by_nid("n1", SymbolType::Object, 0).unwrap().vaddr, 0x3000);
        assert!(db.find_by_nid("n1", SymbolType::NoType, 0).is_none());
        assert!(db.find_by_nid("n1", SymbolType::Func, 0x9999).is_none());
    }
}
