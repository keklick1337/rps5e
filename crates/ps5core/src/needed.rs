// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Modules a program lists as needed (`DT_SCE_NEEDED_MODULE`).
//!
//! A game lists the PRX modules it depends on, and those it ships itself
//! are loaded with the program, before its imports are bound. This module
//! turns the names into file names and gives the order in which modules
//! are loaded: dependencies first, every module once.

use crate::{map_program, next_module_base, LoadedModule, MODULE_LIMIT};
use ps5vm::VirtualMemory;
use std::collections::{HashMap, HashSet};

/// File name a needed name stands for: the name with `.prx` added when it
/// is not there (`libcohtml.Prospero` → `libcohtml.Prospero.prx`). None for
/// an empty name or one with a directory part.
pub fn module_file_name(name: &str) -> Option<String> {
    let name = name.trim();
    if name.is_empty() || name.contains('/') || name.contains('\\') {
        return None;
    }
    Some(if name.ends_with(".prx") { name.to_string() } else { format!("{name}.prx") })
}

/// File names of the modules `p` lists as needed, in table order (both
/// `DT_SCE_NEEDED_MODULE` tags). Names that are not file names are skipped.
pub fn needed_modules(p: &LoadedModule) -> Vec<String> {
    let Some(d) = &p.dynamic else { return Vec::new() };
    d.import_modules.iter().filter_map(|m| module_file_name(&m.name)).collect()
}

/// Load order for `roots` and everything they need: a depth-first
/// post-order walk, so a name comes after the names it depends on.
///
/// `visit` runs once per name (it loads the module) and returns the names
/// that module needs, or None when the module cannot be loaded. Such a name
/// is left out and its dependents still load. A name met again (a shared
/// dependency, or a cycle) is not visited again.
pub fn dependency_order(roots: &[String], visit: &mut dyn FnMut(&str) -> Option<Vec<String>>) -> Vec<String> {
    fn walk(name: &str, visit: &mut dyn FnMut(&str) -> Option<Vec<String>>, seen: &mut HashSet<String>, out: &mut Vec<String>) {
        if !seen.insert(name.to_string()) {
            return;
        }
        let Some(needed) = visit(name) else { return };
        for dep in &needed {
            walk(dep, visit, seen, out);
        }
        out.push(name.to_string());
    }
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for root in roots {
        walk(root, visit, &mut seen, &mut out);
    }
    out
}

/// Directories a game's own module is looked up in, in order: its
/// `sce_module` folder, then the root of the game.
const SHIPPED_DIRS: &[&str] = &["/app0/sce_module", "/app0"];

/// Directories of the system's own libraries, as on the console. They
/// resolve only when a decrypted dump of the firmware's `/system` and
/// `/system_ex` is mounted (`RPS5E_SYSTEM_DIR`, `RPS5E_SYSTEM_EX_DIR`):
/// the firmware's own files are encrypted and cannot be loaded.
const SYSTEM_DIRS: &[&str] = &["/system/common/lib", "/system_ex/common_ex/lib", "/system/priv/lib"];

/// Host file and guest path of the module `file` (`<name>.prx`): the
/// game's own copy first, then a system library `<name>.sprx` of a mounted
/// firmware dump. libkernel stays HLE: the kernel's own library enters the
/// kernel with raw system calls.
fn find_shipped(file: &str) -> Option<(std::path::PathBuf, String)> {
    let resolve =
        |guest: String| ps5hle::filesystem::resolve_guest_path(&guest).filter(|h| h.is_file()).map(|h| (h, guest));
    if let Some(found) = SHIPPED_DIRS.iter().find_map(|dir| resolve(format!("{dir}/{file}"))) {
        return Some(found);
    }
    let stem = file.strip_suffix(".prx").unwrap_or(file);
    if stem.starts_with("libkernel") {
        return None;
    }
    SYSTEM_DIRS.iter().find_map(|dir| resolve(format!("{dir}/{stem}.sprx")))
}

/// Mount the decrypted firmware dumps named by `RPS5E_SYSTEM_DIR` (host
/// copy of `/system`) and `RPS5E_SYSTEM_EX_DIR` (`/system_ex`), if set.
fn mount_system_dumps() {
    for (var, mount) in [("RPS5E_SYSTEM_DIR", "/system"), ("RPS5E_SYSTEM_EX_DIR", "/system_ex")] {
        let Some(dir) = std::env::var_os(var).filter(|d| !d.is_empty()) else { continue };
        let dir = std::path::PathBuf::from(dir);
        if dir.is_dir() && ps5hle::filesystem::register_guest_path_mount(mount, &dir) {
            log::info!("[load] {mount} is the firmware dump {}", dir.display());
        } else {
            log::warn!("[load] {var}={}: not a directory, {mount} stays unmounted", dir.display());
        }
    }
}

/// Load the modules the main executable (`programs[0]`) lists as needed,
/// and the modules those list, and append them to `programs` in
/// dependency order. Called after the main executable and libc.prx are
/// mapped and before relocation, so that the imports of every program bind
/// to these modules' exports.
///
/// A module that is not found, or cannot be mapped, is logged and left
/// out; the programs that need it still load and their imports stay stubs.
pub fn load_needed_modules(vm: &VirtualMemory, programs: &mut Vec<LoadedModule>) {
    mount_system_dumps();
    let roots = needed_modules(&programs[0]);
    let mut next_base = next_module_base(programs.iter().map(|p| p.base + p.size).max().unwrap_or(0));
    let mut mapped: HashMap<String, LoadedModule> = HashMap::new();
    let order = dependency_order(&roots, &mut |file| {
        let Some((host, guest)) = find_shipped(file) else {
            // The system's own libraries (libSceAgc, libScePad, ...) are listed too and
            // are not shipped with the game: without a mounted firmware dump their
            // functions are provided by HLE.
            log::info!("[load] needed module {file} is not shipped with the game (system module)");
            return None;
        };
        let host_path = host.to_string_lossy();
        if let Some(p) = programs.iter().find(|p| p.host_path == host_path) {
            return Some(needed_modules(p));
        }
        if next_base > MODULE_LIMIT {
            log::warn!("[load] needed module {file}: no address space left");
            return None;
        }
        let (m, _) = match map_program(vm, &host, &guest, next_base) {
            Ok(loaded) => loaded,
            Err(e) => {
                log::warn!("[load] needed module {file}: {e}");
                return None;
            }
        };
        let end = next_module_base(m.base + m.size);
        if end > MODULE_LIMIT {
            let _ = vm.unmap(m.base, m.size);
            log::warn!("[load] needed module {file}: no address space left");
            return None;
        }
        next_base = end;
        let deps = needed_modules(&m);
        mapped.insert(file.to_string(), m);
        Some(deps)
    });
    for file in order {
        if let Some(m) = mapped.remove(&file) {
            log::info!("[load] needed module {file} (before linking)");
            programs.push(m);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Order of a walk over the edges `(module, needed module)`; the names
    /// of `missing` cannot be loaded. Asserts every name is visited once.
    fn order(roots: &[&str], edges: &[(&str, &str)], missing: &[&str]) -> Vec<String> {
        let roots: Vec<String> = roots.iter().map(|s| s.to_string()).collect();
        let mut visits = Vec::new();
        let out = dependency_order(&roots, &mut |name| {
            visits.push(name.to_string());
            if missing.contains(&name) {
                return None;
            }
            Some(edges.iter().filter(|(from, _)| *from == name).map(|(_, to)| to.to_string()).collect())
        });
        let unique: HashSet<_> = visits.iter().collect();
        assert_eq!(unique.len(), visits.len(), "visited twice: {visits:?}");
        out
    }

    #[test]
    fn file_names_get_the_prx_suffix() {
        assert_eq!(module_file_name("libcohtml.Prospero").as_deref(), Some("libcohtml.Prospero.prx"));
        assert_eq!(module_file_name("libSceNpCppWebApi.prx").as_deref(), Some("libSceNpCppWebApi.prx"));
        assert_eq!(module_file_name(" libc ").as_deref(), Some("libc.prx"));
        assert_eq!(module_file_name(""), None);
        assert_eq!(module_file_name("../x"), None);
        assert_eq!(module_file_name("a/b.prx"), None);
    }

    #[test]
    fn dependencies_come_first() {
        assert_eq!(order(&["A"], &[("A", "B"), ("B", "C")], &[]), ["C", "B", "A"]);
    }

    #[test]
    fn siblings_keep_table_order() {
        assert_eq!(order(&["A"], &[("A", "B"), ("A", "C")], &[]), ["B", "C", "A"]);
    }

    #[test]
    fn shared_dependency_loads_once() {
        let edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")];
        assert_eq!(order(&["A"], &edges, &[]), ["D", "B", "C", "A"]);
    }

    #[test]
    fn cycle_terminates() {
        assert_eq!(order(&["A"], &[("A", "B"), ("B", "A")], &[]), ["B", "A"]);
    }

    #[test]
    fn missing_module_is_left_out() {
        assert_eq!(order(&["A"], &[("A", "X"), ("A", "B")], &["X"]), ["B", "A"]);
    }

    #[test]
    fn system_libraries_come_from_a_mounted_dump_only() {
        let dump = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../local/tmp/ps5core-tests")
            .join(format!("system-{}", std::process::id()));
        let lib = dump.join("common/lib");
        std::fs::create_dir_all(&lib).unwrap();
        std::fs::write(lib.join("libSceVshCommon.sprx"), b"x").unwrap();
        std::fs::write(lib.join("libkernel_sys.sprx"), b"x").unwrap();
        assert!(find_shipped("libSceVshCommon.prx").is_none(), "nothing is mounted at /system yet");
        assert!(ps5hle::filesystem::register_guest_path_mount("/system", &dump));
        let (host, guest) = find_shipped("libSceVshCommon.prx").expect("found in the dump");
        assert_eq!(guest, "/system/common/lib/libSceVshCommon.sprx");
        assert_eq!(host.canonicalize().unwrap(), lib.join("libSceVshCommon.sprx").canonicalize().unwrap());
        assert!(find_shipped("libkernel_sys.prx").is_none(), "libkernel stays HLE");
        assert!(find_shipped("libSceMissing.prx").is_none());
        ps5hle::filesystem::unregister_guest_path_mount("/system");
        let _ = std::fs::remove_dir_all(&dump);
    }

    #[test]
    fn root_already_loaded_as_dependency_is_not_repeated() {
        assert_eq!(order(&["B", "A"], &[("B", "A")], &[]), ["A", "B"]);
    }
}
