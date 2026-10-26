// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! CFG + structurizer + translation + SSA + optimization over captured game
//! shaders. The sample in `tests/corpus/` always runs; set
//! `PS5SHADER_CORPUS=<dir>` for a full capture directory (prints a report).

use ps5shader::ssa::Stage;
use ps5shader::translate::{compile, TranslateOptions};
use ps5shader::*;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn json_u64(text: &str, key: &str) -> Option<u64> {
    let at = text.find(&format!("\"{key}\":"))? + key.len() + 3;
    let rest = text[at..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn json_str<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let at = text.find(&format!("\"{key}\":"))? + key.len() + 3;
    let rest = text[at..].trim_start().strip_prefix('"')?;
    Some(&rest[..rest.find('"')?])
}

struct Item {
    name: String,
    stage: Stage,
    program: Program,
    blobs: Vec<(u64, Vec<u8>)>,
}

fn stage_of(s: &str) -> Stage {
    match s {
        "cs" => Stage::Compute,
        "ps" => Stage::Pixel,
        "vs" => Stage::Vertex,
        "gs" => Stage::Geometry,
        "hs" => Stage::Hull,
        _ => Stage::Unknown,
    }
}

fn load(dir: &Path) -> Vec<Item> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    let mut out = Vec::new();
    // Entry halves of fused programs end in a register S_SETPC_B64 to the
    // continuation; they are compiled as part of the fused program only.
    let mut fronts = std::collections::HashSet::new();
    for p in &entries {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        if name.starts_with("fused-") && name.ends_with(".json") {
            if let Some(front) = json_str(&std::fs::read_to_string(p).unwrap(), "front") {
                fronts.insert(format!("{front}.bin"));
            }
        }
    }
    for p in entries {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        if name.starts_with("fused-") && name.ends_with(".json") {
            let text = std::fs::read_to_string(&p).unwrap();
            let (Some(front), Some(back)) = (json_str(&text, "front"), json_str(&text, "back")) else { continue };
            let (Some(fa), Some(ba)) = (json_u64(&text, "front_code_address"), json_u64(&text, "back_code_address")) else { continue };
            let fb = std::fs::read(dir.join(format!("{front}.bin"))).unwrap();
            let bb = std::fs::read(dir.join(format!("{back}.bin"))).unwrap();
            let meta = std::fs::read_to_string(dir.join(format!("{front}.json"))).unwrap_or_default();
            let code = [CodeSlice::new(fa, &fb), CodeSlice::new(ba, &bb)];
            let layout = FusedLayout { entry_address: fa, entry_size: fb.len() as u32, continuation_address: ba, continuation_size: bb.len() as u32 };
            let program = decode_fused_program(&code[..], layout).unwrap();
            out.push(Item { name, stage: stage_of(json_str(&meta, "stage").unwrap_or("")), program, blobs: vec![(fa, fb.clone()), (ba, bb.clone())] });
            continue;
        }
        if p.extension().is_none_or(|x| x != "bin") || fronts.contains(&name) {
            continue;
        }
        let meta = std::fs::read_to_string(p.with_extension("json")).unwrap_or_default();
        let addr = json_u64(&meta, "code_address").unwrap_or(0x1000);
        let bytes = std::fs::read(&p).unwrap();
        let program = decode_blob(&bytes, addr).unwrap();
        out.push(Item { name, stage: stage_of(json_str(&meta, "stage").unwrap_or("")), program, blobs: vec![(addr, bytes)] });
    }
    out
}

#[derive(Default)]
struct Report {
    ok: usize,
    failures: BTreeMap<String, Vec<String>>,
    ir_instructions: usize,
    guest_instructions: usize,
    gotos: u32,
    clones: u32,
    loops: u32,
    /// Optimized IR instructions per operation (first word of its name).
    ops: BTreeMap<String, usize>,
}

fn run(items: &[Item], wave: u32) -> Report {
    let mut r = Report::default();
    for it in items {
        let slices: Vec<CodeSlice> = it.blobs.iter().map(|(a, b)| CodeSlice::new(*a, b)).collect();
        let opts = TranslateOptions::with_user_data(wave, it.stage, 16);
        r.guest_instructions += it.program.instructions.len();
        let reader = |a: u64| slices.iter().find_map(|s| s.read_u32(a));
        match compile(&it.program, &opts, Some(&reader)) {
            Ok(c) => {
                if std::env::var("PS5SHADER_IR_DUMP").is_ok_and(|n| it.name.starts_with(&n)) {
                    eprintln!("{}", c.function);
                }
                // PS5SHADER_IR_GREP=<text>: print functions whose IR contains it.
                if let Ok(t) = std::env::var("PS5SHADER_IR_GREP") {
                    let text = c.function.to_string();
                    if text.contains(&t) {
                        eprintln!("==== {}\n{text}", it.name);
                    }
                }
                r.ok += 1;
                r.ir_instructions += c.stats.instructions;
                r.gotos += c.stats.structure.goto_variables;
                r.clones += c.stats.structure.cloned_blocks;
                r.loops += c.stats.structure.loops;
                let f = &c.function;
                for &b in &f.order {
                    for &v in &f.block(b).insts {
                        let name = f.op(v).name();
                        let key = name.split(|c: char| c == ' ' || c == '.').next().unwrap_or("").to_string();
                        if key == "bitcast" {
                            // Bitcasts by the operation of their source.
                            let src = f.op(f.args(v)[0]).name();
                            let src = src.split(|c: char| c == ' ' || c == '.').next().unwrap_or("").to_string();
                            *r.ops.entry(format!("bitcast<-{src}")).or_default() += 1;
                        }
                        *r.ops.entry(key).or_default() += 1;
                    }
                }
            }
            Err(e) => {
                let msg = e.to_string();
                let key = match msg.find(" at 0x") {
                    Some(i) => msg[..i].to_string(),
                    None => msg.clone(),
                };
                r.failures.entry(key).or_default().push(format!("{}: {msg}", it.name));
            }
        }
    }
    r
}

fn print(r: &Report, wave: u32, total: usize) {
    eprintln!(
        "wave{wave}: {}/{} compiled; guest instructions {} -> IR instructions {}; loops {}, goto variables {}, cloned blocks {}",
        r.ok, total, r.guest_instructions, r.ir_instructions, r.loops, r.gotos, r.clones
    );
    if std::env::var_os("PS5SHADER_IR_OPS").is_some() {
        let mut ops: Vec<_> = r.ops.iter().collect();
        ops.sort_by(|a, b| b.1.cmp(a.1));
        for (k, n) in ops.iter().take(40) {
            eprintln!("  {n:9} {k}");
        }
    }
    for (k, v) in &r.failures {
        eprintln!("  {} x {k}", v.len());
        for s in v.iter().take(3) {
            eprintln!("      {s}");
        }
    }
}

#[test]
fn sample_corpus_compiles() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus");
    let items = load(&dir);
    assert!(!items.is_empty());
    for wave in [32, 64] {
        let r = run(&items, wave);
        print(&r, wave, items.len());
        assert_eq!(r.ok, items.len(), "failures: {:?}", r.failures);
    }
}

#[test]
fn full_corpus_compiles() {
    let Ok(dir) = std::env::var("PS5SHADER_CORPUS") else {
        eprintln!("PS5SHADER_CORPUS not set; skipping");
        return;
    };
    let items = load(Path::new(&dir));
    for wave in [64, 32] {
        let r = run(&items, wave);
        print(&r, wave, items.len());
    }
}
