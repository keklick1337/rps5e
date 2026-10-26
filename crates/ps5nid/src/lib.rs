// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! PlayStation 5 NID computation and name database.
//!
//! A NID is derived from a function's symbol name via SHA-1 with the PS5
//! fixed 16-byte salt, taking the first 8 bytes, swapping to big-endian,
//! and base64-encoding (URL-safe, no padding). This matches the algorithm
//! used by the Orbis SDK and every PS5 emulator that resolves imports.

use base64::{engine::general_purpose::STANDARD_NO_PAD, Engine as _};
use sha1::{Digest, Sha1};
use std::collections::HashMap;

/// PS5 NID salt (16 bytes).
const NID_SALT: [u8; 16] = [
    0x51, 0x8D, 0x64, 0xA6, 0x35, 0xDE, 0xD8, 0xC1,
    0xE6, 0xB0, 0x39, 0xB1, 0xC3, 0xE5, 0x52, 0x30,
];

/// Compute a NID string from a symbol name (e.g. "malloc" -> "gQX+4GDQjpM").
pub fn compute_nid(name: &str) -> String {
    let mut hasher = Sha1::new();
    hasher.update(name.as_bytes());
    hasher.update(NID_SALT);
    let digest = hasher.finalize();

    // Take first 8 bytes, interpret as little-endian u64, then big-endian.
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    let value = u64::from_le_bytes(bytes);
    let big_endian = value.to_be_bytes();

    // Standard base64 alphabet without padding, with '/' replaced by '-'.
    STANDARD_NO_PAD.encode(big_endian).replace('/', "-")
}

/// NID database mapping NID strings to symbol names.
pub struct NidDatabase {
    nid_to_name: HashMap<String, String>,
}

impl Default for NidDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl NidDatabase {
    /// Build an empty database.
    pub fn new() -> Self {
        Self {
            nid_to_name: HashMap::new(),
        }
    }

    /// Build a database from an iterable of symbol names (one per line).
    pub fn from_names<'a, I: IntoIterator<Item = &'a str>>(names: I) -> Self {
        let mut nid_to_name = HashMap::new();
        for name in names {
            let name = name.trim();
            if name.is_empty() {
                continue;
            }
            let nid = compute_nid(name);
            nid_to_name.entry(nid).or_insert_with(|| name.to_string());
        }
        Self { nid_to_name }
    }

    /// Load from a newline-delimited text file.
    pub fn from_text_file(path: &str) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(Self::from_names(content.lines()))
    }

    /// Resolve a NID to its symbol name.
    pub fn lookup(&self, nid: &str) -> Option<&str> {
        self.nid_to_name.get(nid).map(|s| s.as_str())
    }

    /// Number of known symbols.
    pub fn len(&self) -> usize {
        self.nid_to_name.len()
    }

    /// Whether the database is empty.
    pub fn is_empty(&self) -> bool {
        self.nid_to_name.is_empty()
    }
}

/// Entry in the built-in HLE export catalog.
#[derive(Debug, Clone, Copy)]
pub struct ExportEntry {
    pub nid: &'static str,
    pub name: &'static str,
    pub library: &'static str,
}

/// Built-in catalog of known HLE exports, embedded at compile time from the
/// merged reference emulator databases. Each line in the TSV data file is:
/// `nid \t name \t library`.
pub fn builtin_exports() -> Vec<ExportEntry> {
    const DATA: &str = include_str!("../data/merged_exports.tsv");
    // The fields borrow the embedded text: nothing is copied or leaked.
    DATA.lines()
        .skip(1) // header
        .filter_map(|line| {
            let mut parts = line.split('\t');
            let nid = parts.next()?;
            let name = parts.next()?;
            let library = parts.next()?;
            Some(ExportEntry { nid, name, library })
        })
        .collect()
}

/// Convenience: build a NidDatabase from the built-in export catalog.
pub fn builtin_database() -> NidDatabase {
    NidDatabase::from_names(builtin_exports().iter().map(|e| e.name))
}

/// Build a NID→ExportEntry lookup from the built-in catalog.
pub fn builtin_export_map() -> HashMap<String, ExportEntry> {
    builtin_exports()
        .iter()
        .map(|e| (e.nid.to_string(), *e))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malloc_nid_matches_known_value() {
        // The PS5 NID for "malloc" — cross-referenced with the game's import
        // table and the reference emulator's known-good NID mappings.
        let nid = compute_nid("malloc");
        assert_eq!(nid, "gQX+4GDQjpM", "malloc NID mismatch: {nid}");
    }

    #[test]
    fn free_nid() {
        let nid = compute_nid("free");
        assert_eq!(nid, "tIhsqj0qsFE", "free NID mismatch: {nid}");
    }

    #[test]
    fn memset_nid() {
        let nid = compute_nid("memset");
        assert_eq!(nid, "8zTFvBIAIN8", "memset NID mismatch: {nid}");
    }

    #[test]
    fn strlen_nid() {
        let nid = compute_nid("strlen");
        assert_eq!(nid, "j4ViWNHEgww", "strlen NID mismatch: {nid}");
    }

    #[test]
    fn nid_with_dash_and_plus() {
        assert_eq!(compute_nid("sceKernelAddUserEvent"), "4R6-OvI2cEA");
        assert_eq!(compute_nid("malloc"), "gQX+4GDQjpM");
    }

    #[test]
    fn builtin_catalog_has_malloc() {
        let map = builtin_export_map();
        let entry = map.get("gQX+4GDQjpM").expect("malloc must be in catalog");
        assert_eq!(entry.name, "malloc");
        assert_eq!(entry.library, "libSceLibcInternal");
    }

    #[test]
    fn builtin_catalog_has_sce_kernel_map_direct_memory() {
        let map = builtin_export_map();
        // The NID for sceKernelMapDirectMemory from the reference emulator.
        assert!(
            map.values().any(|e| e.name == "sceKernelMapDirectMemory"),
            "sceKernelMapDirectMemory must be in catalog"
        );
    }

    #[test]
    fn catalog_nids_are_unique_and_in_the_nid_alphabet() {
        let exports = builtin_exports();
        let mut seen = std::collections::HashSet::new();
        for e in &exports {
            assert!(
                e.nid.len() == 11 && e.nid.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'+' || b == b'-'),
                "{} ({}): not a NID (URL-safe base64 uses '_' where NIDs use '-')",
                e.nid,
                e.name
            );
            assert!(seen.insert(e.nid), "duplicate NID {}", e.nid);
        }
    }

    #[test]
    fn catalog_names_hash_to_their_nids() {
        // Rows whose name is a reference emulator's own label for a NID
        // whose symbol name is still unknown (the NID audit lists
        // them). The number may only go down.
        const MAX_UNHASHED: usize = 7;
        let unhashed: Vec<_> = builtin_exports()
            .into_iter()
            .filter(|e| compute_nid(e.name) != e.nid)
            .map(|e| format!("{} {}", e.nid, e.name))
            .collect();
        assert!(unhashed.len() <= MAX_UNHASHED, "{} rows: {:?}", unhashed.len(), unhashed);
    }

    #[test]
    fn audit_names_resolve_game_imports() {
        // Names found by the NID audit for NIDs the games import.
        let map = builtin_export_map();
        for (nid, name, lib) in [
            ("-KRzWekV120", "sceAgcDcbSetIndexSize_0600", "libSceAgc"),
            ("qj7QZpgr9Uw", "sceAgcDcbContextStateOp_0100", "libSceAgc"),
            ("Ikfdt-rIqCE", "sceAgcJumpPatchSetTarget_0300", "libSceAgc"),
            ("GHna9-DvnUk", "scePsmlMfsrGetMipmapBias", "libScePsml"),
            ("LXq+6mIxpCw", "scePsmlMfsrIsCaptureInProgress", "libScePsml"),
            ("FSGaTQze0UY", "scePsmlMfsrSelectConfig", "libScePsml"),
            ("k0E7vkgqAuE", "sceAgcUpdateInterpolantMapping_0100", "libSceAgc"),
            ("BG26hBGiNlw", "_sceUlobjmgrRegisterObject", "ulobjmgr"),
            (
                "ZXKMi2AM4t8",
                "_ZN3sce2Np9CppWebApi11Matchmaking2V113PlayerForRead19getPlayerAttributesEv",
                "libSceNpCppWebApi",
            ),
        ] {
            assert_eq!(compute_nid(name), nid, "{name}");
            let e = map.get(nid).unwrap_or_else(|| panic!("{nid} {name} missing"));
            assert_eq!((e.name, e.library), (name, lib), "{nid}");
        }
    }

    #[test]
    fn builtin_catalog_size() {
        let exports = builtin_exports();
        assert!(exports.len() > 1500, "expected 1500+ exports, got {}", exports.len());
    }
}
