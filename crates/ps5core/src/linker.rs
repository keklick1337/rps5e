// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Import linking policy: decides, per imported NID, whether a call goes to
//! an HLE callback (INT3 stub) or straight into native code exported by a
//! loaded PRX module (LLE).
//!
//! Rules:
//!   - kernel libraries are always HLE;
//!   - an import with a real HLE implementation stays HLE, except for the
//!     libc allocator family (when the shipped libc.prx exports all of it),
//!     native formatting (`sprintf`) and the safe memory primitives;
//!   - an import without a real HLE implementation binds to a module export
//!     when one exists (by NID, then by name candidates);
//!   - libc FILE-object functions never mix native and HLE layouts.
//!
//! Binding the allocator to native libc matters: libc's malloc consults the
//! game's `_sceLibcMallocReplace` table from the process param and calls the
//! game's own `malloc_init`, which constructs and publishes the game heap.

use crate::{stubs::StubTable, LoadedModule};
use ps5hle::HleRegistry;
use ps5vm::VirtualMemory;
use std::collections::HashMap;

/// NIDs whose shipped libc bodies must not run natively (memset contains a
/// PS5-specific `int 45h` path; memcpy and fputs use HLE-side bookkeeping).
const HLE_PREFERRED_NIDS: &[&str] = &[
    "8zTFvBIAIN8", // memset
    "QrZZdJ8XsX0", // fputs
    "Q3VBxCXhUHs", // memcpy
];

/// memset and memcpy, bound to the module's own exports with
/// `RPS5E_NATIVE_MEMOPS=1` (an A/B check of the HLE copies).
const NATIVE_MEMOP_NIDS: &[&str] = &["8zTFvBIAIN8", "Q3VBxCXhUHs"];

fn native_memops() -> bool {
    std::env::var("RPS5E_NATIVE_MEMOPS").is_ok_and(|v| v == "1")
}

/// Allocator family that must be bound as a unit: mixing a native malloc
/// with an HLE free corrupts both heaps.
const ALLOCATOR_FAMILY: &[(&str, &str)] = &[
    ("gQX+4GDQjpM", "malloc"),
    ("tIhsqj0qsFE", "free"),
    ("2X5agFjKxMc", "calloc"),
    ("Y7aJ1uydPMo", "realloc"),
    ("Ujf3KzMvRmI", "memalign"),
    ("2Btkg8k24Zg", "aligned_alloc"),
    ("cVSk9y8URbc", "posix_memalign"),
];

const ALLOCATOR_EXPORTS: &[&str] = &[
    "malloc",
    "free",
    "calloc",
    "realloc",
    "memalign",
    "aligned_alloc",
    "posix_memalign",
    "malloc_usable_size",
];

const SAFE_LLE_EXPORTS: &[&str] = &["memcpy", "memmove", "memset", "memcmp"];

const FILE_OBJECT_EXPORTS: &[&str] = &[
    "_Fgpos", "_Flocale", "_Fsetlocale", "_Fspos", "_Lockfilelock", "_Unlockfilelock",
    "clearerr", "clearerr_unlocked", "fdopen", "fclose", "feof", "feof_unlocked", "ferror", "ferror_unlocked", "fflush", "fgetc", "fgetpos", "fgets",
    "fgetwc", "fgetws", "fileno", "fileno_unlocked", "fopen", "fopen_s", "fprintf", "fprintf_s", "fputc", "fputs",
    "fputwc", "fputws", "fread", "freopen", "freopen_s", "fscanf", "fscanf_s", "fseek",
    "fsetpos", "ftell", "fwide", "fwrite", "fwprintf", "fwprintf_s", "fwscanf", "fwscanf_s",
    "getc", "getw", "getwc", "putc", "putw", "putwc", "rewind", "setbuf", "setvbuf", "ungetc",
    "ungetwc", "vfprintf", "vfprintf_s", "vfscanf", "vfscanf_s", "vfwprintf", "vfwprintf_s",
    "vfwscanf", "vfwscanf_s",
];

fn is_libc_library(lib: &str) -> bool {
    lib.to_ascii_lowercase().contains("libc")
}

fn is_kernel_library(lib: &str) -> bool {
    lib.contains("Kernel") || lib.eq_ignore_ascii_case("libkernel")
}

/// Process-wide LLE switches (environment):
///   PS5_DISABLE_LLE_LIBC=1     — never bind libc natively
///   PS5_LLE_LIBC_SAFE_ONLY=off — same as disable
///   PS5_LLE_LIBC_ALL=1         — bind every libc export natively
struct LleConfig {
    disabled: bool,
    all: bool,
}

impl LleConfig {
    fn from_env() -> Self {
        let disabled = std::env::var("PS5_DISABLE_LLE_LIBC").is_ok_and(|v| v == "1")
            || std::env::var("PS5_LLE_LIBC_SAFE_ONLY")
                .is_ok_and(|v| matches!(v.to_ascii_lowercase().as_str(), "off" | "false" | "none"));
        let all = std::env::var("PS5_LLE_LIBC_ALL").is_ok_and(|v| v == "1");
        Self { disabled, all }
    }
}

struct Linker<'a> {
    modules: &'a [LoadedModule],
    registry: &'a HleRegistry,
    cfg: LleConfig,
    allocator_ok: bool,
    sprintf_ok: bool,
}

impl<'a> Linker<'a> {
    fn new(modules: &'a [LoadedModule], registry: &'a HleRegistry) -> Self {
        let mut l = Self {
            modules,
            registry,
            cfg: LleConfig::from_env(),
            allocator_ok: false,
            sprintf_ok: false,
        };
        l.allocator_ok = ALLOCATOR_FAMILY
            .iter()
            .all(|(nid, name)| l.module_symbol(nid, name).is_some());
        l.sprintf_ok = l.module_symbol("tcVi5SivF7Q", "sprintf").is_some();
        l
    }

    /// Exported function address for `nid` in any loaded module.
    fn module_export(&self, nid: &str) -> Option<u64> {
        self.modules.iter().find_map(|m| m.resolve(nid)).filter(|&a| a != 0)
    }

    /// Resolve by NID, then by name and its `_`-prefixed/stripped variants.
    fn module_symbol(&self, nid: &str, name: &str) -> Option<u64> {
        if let Some(a) = self.module_export(nid) {
            return Some(a);
        }
        self.name_candidates(name)
            .find_map(|n| self.module_export(&ps5nid::compute_nid(&n)))
    }

    fn name_candidates(&self, name: &str) -> impl Iterator<Item = String> {
        let mut v = Vec::with_capacity(3);
        if !name.is_empty() {
            v.push(name.to_string());
            match name.strip_prefix('_') {
                Some(rest) if !rest.is_empty() => v.push(rest.to_string()),
                _ => v.push(format!("_{name}")),
            }
        }
        v.into_iter()
    }

    fn prefer_lle_for_libc(&self, name: &str) -> bool {
        if name.is_empty() || self.cfg.disabled {
            return false;
        }
        if self.cfg.all {
            return true;
        }
        if ALLOCATOR_EXPORTS.contains(&name) {
            return self.allocator_ok;
        }
        if name == "sprintf" {
            return self.sprintf_ok;
        }
        SAFE_LLE_EXPORTS.contains(&name)
    }

    /// Native target for an import, or None to keep it on HLE.
    fn resolve(&self, nid: &str) -> Option<u64> {
        if HLE_PREFERRED_NIDS.contains(&nid) && !(native_memops() && NATIVE_MEMOP_NIDS.contains(&nid)) {
            return None;
        }
        let catalog = self.registry.lookup(nid);
        if catalog.is_some_and(|e| e.fallback) {
            return self.module_export(nid);
        }
        if self.registry.is_implemented(nid) {
            let e = catalog?;
            if is_kernel_library(e.library) {
                return None;
            }
            if !(is_libc_library(e.library) && self.prefer_lle_for_libc(e.name)) {
                return None;
            }
            return self.module_symbol(nid, e.name);
        }
        if let Some(e) = catalog {
            if is_kernel_library(e.library) {
                return None;
            }
            if FILE_OBJECT_EXPORTS.contains(&e.name) && !self.prefer_lle_for_libc(e.name) {
                return None;
            }
        }
        if let Some(a) = self.module_export(nid) {
            return Some(a);
        }
        let e = catalog?;
        if !self.prefer_lle_for_libc(e.name) {
            return None;
        }
        self.name_candidates(e.name)
            .find_map(|n| self.module_export(&ps5nid::compute_nid(&n)))
    }
}

/// Rewrite import stubs whose NID binds to native module code. Returns the
/// redirected NIDs with their native targets.
pub fn link_native_imports(
    vm: &VirtualMemory,
    stubs: &StubTable,
    modules: &[LoadedModule],
    registry: &HleRegistry,
) -> HashMap<String, u64> {
    let linker = Linker::new(modules, registry);
    log::info!(
        "[link] LLE libc: allocator_family={} sprintf={} disabled={} all={}",
        linker.allocator_ok,
        linker.sprintf_ok,
        linker.cfg.disabled,
        linker.cfg.all
    );
    let mut nids: Vec<String> = stubs.nid_to_addr_iter().map(|(n, _)| n.clone()).collect();
    nids.sort();
    let mut redirected = HashMap::new();
    for nid in nids {
        let Some(target) = linker.resolve(&nid) else { continue };
        match stubs.redirect(vm, &nid, target) {
            Ok(stub) => {
                redirected.insert(nid.clone(), target);
                let name = registry.lookup(&nid).map(|e| e.name).unwrap_or("?");
                log::debug!("[link] LLE {} ({}) stub {:#x} -> {:#x}", name, nid, stub, target);
            }
            Err(e) => log::warn!("[link] failed to redirect {}: {}", nid, e),
        }
    }
    redirected
}
