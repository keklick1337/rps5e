// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Tests over captured game shaders.
//!
//! `tests/corpus/` holds a small sample (captured with
//! `RPS5E_DUMP_SHADERS`). Set `PS5SHADER_CORPUS=<dir>` to also run the
//! decode/coverage checks over a full capture directory.

use ps5shader::ir::{exec_full, movrel, wave64_half_mask, ControlFlowGraph, Gen5BranchResolver};
use ps5shader::*;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

fn sample_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus")
}

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

struct Shader {
    hash: String,
    stage: String,
    code_address: u64,
    bytes: Vec<u8>,
}

fn load(dir: &Path) -> Vec<Shader> {
    let mut out = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.extension().is_none_or(|x| x != "bin") {
            continue;
        }
        let meta = std::fs::read_to_string(p.with_extension("json")).unwrap_or_default();
        out.push(Shader {
            hash: p.file_stem().unwrap().to_string_lossy().into_owned(),
            stage: json_str(&meta, "stage").unwrap_or("unknown").to_string(),
            code_address: json_u64(&meta, "code_address").unwrap_or(0x1000),
            bytes: std::fs::read(&p).unwrap(),
        });
    }
    out
}

fn decode(s: &Shader) -> Program {
    decode_blob(&s.bytes, s.code_address).unwrap_or_else(|e| panic!("{} ({}): {e}", s.hash, s.stage))
}

/// Decode everything, check structural invariants, return opcode counts.
fn check_corpus(shaders: &[Shader]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for s in shaders {
        let p = decode(s);
        assert!(!p.instructions.is_empty());
        // Instructions are contiguous from pc 0 and their words are exactly
        // the blob's bytes (lossless decode).
        let mut pc = 0u32;
        for i in &p.instructions {
            assert_eq!(i.pc, pc, "{}: gap before pc 0x{pc:x}", s.hash);
            for (k, w) in i.words.iter().enumerate() {
                let at = pc as usize + k * 4;
                let b = &s.bytes[at..at + 4];
                assert_eq!(*w, u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            }
            pc = i.next_pc();
            assert!(!i.opcode.is_raw(), "{}: unnamed opcode {} at 0x{:x}", s.hash, i.opcode.name(), i.pc);
            *counts.entry(format!("{:?}/{}", i.encoding, i.opcode.name())).or_insert(0) += 1;
            // Display never panics and starts with the mnemonic.
            assert!(i.to_string().starts_with(&i.opcode.mnemonic()));
        }
        let last = p.instructions.last().unwrap().opcode;
        assert!(
            matches!(last, Opcode::SEndpgm | Opcode::SBranch | Opcode::SSetpcB64),
            "{}: ends with {}",
            s.hash,
            last.name()
        );
        if s.stage == "gs_front" {
            assert_eq!(last, Opcode::SSetpcB64, "{}: front half must end in S_SETPC", s.hash);
        }
    }
    counts
}

#[test]
fn sample_corpus_decodes() {
    let shaders = load(&sample_dir());
    assert!(shaders.len() >= 10, "corpus sample missing");
    let counts = check_corpus(&shaders);
    eprintln!("sample corpus: {} programs, {} distinct opcodes", shaders.len(), counts.len());
}

/// Full capture (opt-in): `PS5SHADER_CORPUS=/path cargo test -p ps5shader -- --nocapture`.
#[test]
fn full_corpus_decodes() {
    let Some(dir) = std::env::var_os("PS5SHADER_CORPUS") else { return };
    let shaders = load(Path::new(&dir));
    let counts = check_corpus(&shaders);
    let movrel_programs = check_analyses(&shaders);
    eprintln!("analyses ok; {movrel_programs} programs use V_MOVREL*");
    let total: usize = counts.values().sum();
    eprintln!("full corpus: {} programs, {} instructions, {} distinct opcodes", shaders.len(), total, counts.len());
    for (k, v) in &counts {
        eprintln!("{k:<40} {v:>9}");
    }
}

#[test]
fn pixel_shaders_export_color() {
    for s in load(&sample_dir()).iter().filter(|s| s.stage == "ps") {
        let p = decode(s);
        let exports: Vec<_> = p.exports().collect();
        assert!(!exports.is_empty(), "{}: no EXP", s.hash);
        assert!(exports.iter().any(|e| e.done), "{}: no EXP done", s.hash);
    }
}

#[test]
fn fused_pair() {
    let dir = sample_dir();
    let desc = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.file_name().unwrap().to_string_lossy().starts_with("fused-"))
        .expect("fused descriptor");
    let text = std::fs::read_to_string(desc).unwrap();
    let front = std::fs::read(dir.join(format!("{}.bin", json_str(&text, "front").unwrap()))).unwrap();
    let back = std::fs::read(dir.join(format!("{}.bin", json_str(&text, "back").unwrap()))).unwrap();
    let fa = json_u64(&text, "front_code_address").unwrap();
    let ba = json_u64(&text, "back_code_address").unwrap();

    // The front half alone stops at its S_SETPC_B64.
    let alone = decode_blob(&front, fa).unwrap();
    assert_eq!(alone.instructions.last().unwrap().opcode, Opcode::SSetpcB64);

    let code = [CodeSlice::new(fa, &front), CodeSlice::new(ba, &back)];
    let layout = FusedLayout { entry_address: fa, entry_size: front.len() as u32, continuation_address: ba, continuation_size: back.len() as u32 };
    let p = decode_fused_program(&code[..], layout).unwrap();
    assert!(p.instructions.iter().all(|i| i.opcode != Opcode::SSetpcB64));
    let n = alone.instructions.len();
    assert_eq!(p.instructions[n - 1].opcode, Opcode::SNop);
    let back_alone = decode_blob(&back, ba).unwrap();
    assert_eq!(p.instructions.len(), n + back_alone.instructions.len());
    let first_back = &p.instructions[n];
    assert!(ba > fa);
    assert_eq!(first_back.pc as u64, ba - fa, "adjacent continuation keeps its real distance");
    assert_eq!(p.instructions.last().unwrap().opcode, Opcode::SEndpgm);
    // The fused program exports positions / parameters.
    assert!(p.exports().count() > 0 || p.instructions.iter().any(|i| i.encoding == Encoding::Ds));
}

#[test]
fn analyses_on_corpus() {
    assert!(check_analyses(&load(&sample_dir())) > 0, "sample must contain a V_MOVRELS shader");
}

/// Run every analysis and check its invariants; returns how many programs
/// use relative moves.
fn check_analyses(shaders: &[Shader]) -> usize {
    let mut movrel_programs = 0;
    for s in shaders {
        let p = decode(&s);
        let computed_jump = p.instructions.iter().any(|i| i.opcode.is_computed_jump());

        let cfg = ControlFlowGraph::build(&p.instructions, &Gen5BranchResolver);
        assert_eq!(cfg.blocks[0].start_pc, 0);
        for w in cfg.blocks.windows(2) {
            assert_eq!(w[0].end_pc, w[1].start_pc);
        }
        for i in &p.instructions {
            assert!(cfg.block_of(i.pc).is_some());
        }

        let full = exec_full::analyze(&p, false);
        if !computed_jump {
            assert!(full.contains(&0), "{}: entry must run with a full EXEC", s.hash);
        } else {
            assert!(full.is_empty());
        }
        let full32 = exec_full::analyze(&p, true);
        // wave32 only adds full points (32-bit EXEC restores count).
        assert!(full.is_subset(&full32));

        let plan = wave64_half_mask::analyze(&p, &HashSet::new());
        assert_eq!(plan.is_none(), computed_jump, "{}", s.hash);
        if let Some(plan) = plan {
            for (pc, pairs) in &plan.exact_pairs_before {
                assert!(p.index_of_pc(*pc).is_some());
                assert!(pairs.iter().all(|r| r % 2 == 0 && *r < 128));
            }
            for (pc, dst) in &plan.half_mask_writes {
                let i = &p.instructions[p.index_of_pc(*pc).unwrap()];
                assert!(i.opcode.is_vector_compare() || i.opcode.writes_vector_carry_out());
                assert_eq!(dst % 2, 0);
            }
        }

        let rel = movrel::analyze(&p);
        if p.instructions.iter().any(|i| i.opcode.is_bounded_relative_move()) {
            movrel_programs += 1;
            for (pc, values) in &rel {
                assert!(p.instructions[p.index_of_pc(*pc).unwrap()].opcode.is_bounded_relative_move());
                assert!(!values.is_empty() && values.len() <= 64);
                assert!(values.windows(2).all(|w| w[0] < w[1]));
            }
        } else {
            assert!(rel.is_empty());
        }
    }
    movrel_programs
}
