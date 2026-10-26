// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Unit tests map guest scratch memory at fixed addresses (`const BASE`)
//! and run in parallel in one process, so two tests sharing a base
//! silently remap each other's memory. Every base must be unique.

use std::collections::HashMap;
use std::path::Path;

fn collect(dir: &Path, out: &mut HashMap<u64, Vec<String>>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            let text = std::fs::read_to_string(&path).unwrap();
            for (n, line) in text.lines().enumerate() {
                let Some(rest) = line.trim_start().strip_prefix("const BASE: u64 = 0x") else { continue };
                let hex: String = rest.chars().take_while(|c| c.is_ascii_hexdigit() || *c == '_').filter(|c| *c != '_').collect();
                let value = u64::from_str_radix(&hex, 16).unwrap();
                out.entry(value).or_default().push(format!("{}:{}", path.display(), n + 1));
            }
        }
    }
}

#[test]
fn test_scratch_bases_are_unique() {
    let mut bases = HashMap::new();
    collect(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut bases);
    let dups: Vec<_> = bases.iter().filter(|(_, v)| v.len() > 1).collect();
    assert!(dups.is_empty(), "test BASE addresses shared between tests: {dups:#x?}");
}
