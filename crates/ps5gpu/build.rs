// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! `PS5GPU_SHADER_SOURCE_VERSION`: a hash of every source file that decides
//! what the shader compiler emits (the `ps5shader` crate, the SPIR-V
//! back-end, the compile request front-end), the lock file and the
//! compiler version. The on-disk module cache is keyed by it, so modules
//! survive rebuilds that leave the shader compiler alone. Comment and blank
//! lines and test-only files do not count: they do not change the output.

use std::path::{Path, PathBuf};

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h = (*h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn files(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        let Ok(rd) = std::fs::read_dir(path) else { return };
        for e in rd.flatten() {
            files(&e.path(), out);
        }
    } else if is_test_file(path) {
    } else if path.extension().is_some_and(|e| e == "rs") || path.file_name().is_some_and(|n| n == "Cargo.lock" || n == "Cargo.toml") {
        out.push(path.to_path_buf());
    }
}

fn is_test_file(path: &Path) -> bool {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    name == "tests.rs" || name.ends_with("_tests.rs") || path.components().any(|c| c.as_os_str() == "tests")
}

/// The bytes that decide the output: for Rust sources, every line without
/// its indentation, minus blank and comment-only lines.
fn code_bytes(path: &Path) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap_or_default();
    if path.extension().is_none_or(|e| e != "rs") {
        return bytes;
    }
    let mut out = Vec::with_capacity(bytes.len());
    for line in String::from_utf8_lossy(&bytes).lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        out.extend_from_slice(line.as_bytes());
        out.push(b'\n');
    }
    out
}

fn main() {
    let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let roots = [
        dir.join("../ps5shader/src"),
        dir.join("../ps5shader/Cargo.toml"),
        dir.join("src/shader"),
        dir.join("src/pipeline/compiler.rs"),
        dir.join("src/pipeline/frontend.rs"),
        dir.join("src/pipeline/sharps.rs"),
        dir.join("Cargo.toml"),
        dir.join("../../Cargo.lock"),
    ];
    let mut list = Vec::new();
    for r in &roots {
        println!("cargo:rerun-if-changed={}", r.display());
        files(r, &mut list);
    }
    list.sort();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for f in &list {
        // Relative names: the same sources in another checkout hash alike.
        let name = f.strip_prefix(&dir).unwrap_or(f);
        fnv(&mut h, name.to_string_lossy().as_bytes());
        fnv(&mut h, &code_bytes(f));
    }
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    if let Ok(o) = std::process::Command::new(rustc).arg("-V").output() {
        fnv(&mut h, &o.stdout);
    }
    fnv(&mut h, std::env::var("PROFILE").unwrap_or_default().as_bytes());
    println!("cargo:rustc-env=PS5GPU_SHADER_SOURCE_VERSION={h:016x}");
}
