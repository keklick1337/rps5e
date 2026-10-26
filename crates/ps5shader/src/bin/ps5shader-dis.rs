// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Disassemble dumped shaders (`RPS5E_DUMP_SHADERS` output).
//!
//! ```text
//! ps5shader-dis <hash>.bin                 disassemble one blob
//! ps5shader-dis fused-<a>-<b>.json         disassemble a fused pair
//! ps5shader-dis --analyses <file>          ... plus EXEC-full / half-mask / MOVREL results
//! ps5shader-dis --stats <dir>              decode every blob, print opcode coverage
//! ps5shader-dis --stats-mods <dir>         ... per opcode: modifiers, constants, memory flags
//! ps5shader-dis --bench <dir>              compile every blob, print the time per phase/pass and the slowest
//! ps5shader-dis --list-opcodes             every opcode mnemonic the decoder names
//! ```

use ps5shader::ir::{exec_full, movrel, wave64_half_mask};
use ps5shader::{decode_blob, decode_fused_program, CodeSlice, CodeSource, FusedLayout, Program};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Unsigned integer value of `"key": N` in a flat JSON text.
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

fn decode_file(path: &Path) -> Result<Program, String> {
    let is_fused = path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("fused-") && n.ends_with(".json"));
    if is_fused {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let dir = path.parent().unwrap_or(Path::new("."));
        let (Some(front), Some(back)) = (json_str(&text, "front"), json_str(&text, "back")) else {
            return Err("malformed fused descriptor".into());
        };
        let (Some(fa), Some(ba)) = (json_u64(&text, "front_code_address"), json_u64(&text, "back_code_address")) else {
            return Err("fused descriptor lacks addresses".into());
        };
        let fb = std::fs::read(dir.join(format!("{front}.bin"))).map_err(|e| e.to_string())?;
        let bb = std::fs::read(dir.join(format!("{back}.bin"))).map_err(|e| e.to_string())?;
        let code = [CodeSlice::new(fa, &fb), CodeSlice::new(ba, &bb)];
        let layout = FusedLayout { entry_address: fa, entry_size: fb.len() as u32, continuation_address: ba, continuation_size: bb.len() as u32 };
        return decode_fused_program(&code[..], layout).map_err(|e| e.to_string());
    }
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    // Decode at the original code address when the sidecar is present, so
    // printed program offsets match guest addresses modulo the base.
    let base = std::fs::read_to_string(path.with_extension("json")).ok().and_then(|t| json_u64(&t, "code_address")).unwrap_or(0x1000);
    decode_blob(&bytes, base.max(4)).map_err(|e| e.to_string())
}

fn print_analyses(program: &Program, wave32: bool) {
    let full = exec_full::analyze(program, wave32);
    println!("; exec-full ({}): {} of {} instructions", if wave32 { "wave32" } else { "wave64" }, full.len(), program.instructions.len());
    match wave64_half_mask::analyze(program, &HashSet::new()) {
        Some(plan) => {
            let mut exact: Vec<_> = plan.exact_pairs_before.iter().collect();
            exact.sort();
            let mut writes: Vec<_> = plan.half_mask_writes.iter().collect();
            writes.sort();
            println!("; wave64 half-mask writes: {writes:x?}");
            println!("; wave64 exact-before: {exact:x?}");
            println!("; wave64 LDS barriers: {:x?}", plan.shared_memory_barriers_before);
        }
        None => println!("; wave64 half-mask: not analysable"),
    }
    let rel = movrel::analyze(program);
    if !rel.is_empty() {
        println!("; movrel M0 values: {rel:x?}");
    }
}

/// Modifier and operand-form tags of one instruction, for `--stats-mods`.
fn modifier_tags(i: &ps5shader::Instruction) -> Vec<String> {
    use ps5shader::{Control, OperandKind};
    let mut tags = Vec::new();
    for s in &i.sources {
        match s.kind {
            OperandKind::LiteralConstant => tags.push("literal".to_string()),
            OperandKind::EncodedConstant => tags.push(format!("const{}", s.value)),
            _ => {}
        }
    }
    match &i.control {
        Some(Control::Vop3(c)) => {
            if c.absolute_mask != 0 {
                tags.push(format!("abs{:x}", c.absolute_mask));
            }
            if c.negate_mask != 0 {
                tags.push(format!("neg{:x}", c.negate_mask));
            }
            if c.output_modifier != 0 {
                tags.push(format!("omod{}", c.output_modifier));
            }
            if c.clamp {
                tags.push("clamp".into());
            }
            if c.operand_select != 0 {
                tags.push(format!("opsel{:x}", c.operand_select));
            }
        }
        Some(Control::Sdwa(c)) => tags.push(format!(
            "sdwa(dst{} unused{} s0:{}{} s1:{}{} abs{:x} neg{:x} omod{} clamp{})",
            c.destination_select,
            c.destination_unused,
            c.source0_select,
            if c.source0_sign_extend { "s" } else { "" },
            c.source1_select,
            if c.source1_sign_extend { "s" } else { "" },
            c.absolute_mask,
            c.negate_mask,
            c.output_modifier,
            c.clamp as u8
        )),
        Some(Control::Vop3p(c)) => tags.push(format!(
            "vop3p(opsel{:x} hi{:x} neglo{:x} neghi{:x} clamp{})",
            c.op_sel_mask, c.op_sel_hi_mask, c.neg_lo_mask, c.neg_hi_mask, c.clamp as u8
        )),
        Some(Control::Dpp(c)) => tags.push(format!("dpp(ctrl{:x} fi{} bc{} abs{:x} neg{:x} bank{:x} row{:x})", c.control, c.fetch_inactive as u8, c.bound_control as u8, c.absolute_mask, c.negate_mask, c.bank_mask, c.row_mask)),
        Some(Control::Dpp8(c)) => tags.push(format!("dpp8({:x})", c.lane_selectors)),
        Some(Control::Image(c)) => tags.push(format!(
            "mimg(dmask{:x} dim{}{}{}{}{}{} naddr{})",
            c.dmask,
            c.dimension,
            if c.is_array { " array" } else { "" },
            if c.a16 { " a16" } else { "" },
            if c.d16 { " d16" } else { "" },
            if c.glc { " glc" } else { "" },
            if c.slc { " slc" } else { "" },
            c.address_registers.len()
        )),
        Some(Control::BufferMemory(c)) => tags.push(format!(
            "buf({}{}{}{}{} off{} soff{:?})",
            if c.index_enabled { "idxen " } else { "" },
            if c.offset_enabled { "offen " } else { "" },
            if c.glc { "glc " } else { "" },
            if c.slc { "slc " } else { "" },
            if c.typed { format!("fmt{}", c.typed_format) } else { "untyped".into() },
            c.offset_bytes,
            i.sources.last().map(|s| (s.kind, s.value))
        )),
        Some(Control::ScalarMemory(c)) => tags.push(format!(
            "smem(imm{} soff{})",
            if c.immediate_offset_bytes >= 0x100 { ">=0x100".to_string() } else { format!("{}", c.immediate_offset_bytes) },
            c.dynamic_offset_register.is_some() as u8
        )),
        Some(Control::Export(c)) => tags.push(format!("exp(t{} m{:x}{})", c.target, c.enable_mask, if c.compressed { " compr" } else { "" })),
        _ => {}
    }
    tags
}

fn stats_mods(dir: &Path) -> ExitCode {
    let files: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "bin")).collect(),
        Err(e) => {
            eprintln!("{}: {e}", dir.display());
            return ExitCode::FAILURE;
        }
    };
    let mut counts: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for f in &files {
        let Ok(p) = decode_file(f) else { continue };
        let mut seen = HashSet::new();
        for i in &p.instructions {
            for t in modifier_tags(i) {
                let key = format!("{:?}/{} {t}", i.encoding, i.opcode.name());
                let e = counts.entry(key.clone()).or_default();
                e.0 += 1;
                if seen.insert(key) {
                    e.1 += 1;
                }
            }
        }
    }
    for (name, (n, p)) in &counts {
        println!("{name:<90} {n:>8} {p:>6}");
    }
    ExitCode::SUCCESS
}

fn stats(dir: &Path) -> ExitCode {
    let mut files: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "bin")).collect(),
        Err(e) => {
            eprintln!("{}: {e}", dir.display());
            return ExitCode::FAILURE;
        }
    };
    files.sort();
    let mut counts: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut failures = Vec::new();
    let mut raw_programs = 0;
    let mut instructions = 0;
    for f in &files {
        match decode_file(f) {
            Ok(p) => {
                let mut seen = HashSet::new();
                let mut has_raw = false;
                for i in &p.instructions {
                    has_raw |= i.opcode.is_raw();
                    let e = counts.entry(format!("{:?}/{}", i.encoding, i.opcode.name())).or_default();
                    e.0 += 1;
                    if seen.insert(i.opcode) {
                        e.1 += 1;
                    }
                }
                instructions += p.instructions.len();
                raw_programs += has_raw as usize;
            }
            Err(e) => failures.push(format!("{}: {e}", f.display())),
        }
    }
    println!("programs: {}  decoded: {}  failed: {}  with-raw-opcodes: {}  instructions: {}", files.len(), files.len() - failures.len(), failures.len(), raw_programs, instructions);
    println!("{:<40} {:>10} {:>9}", "encoding/opcode", "count", "programs");
    for (name, (n, p)) in &counts {
        let raw = if name.contains("Raw") { "  <- unnamed" } else { "" };
        println!("{name:<40} {n:>10} {p:>9}{raw}");
    }
    for f in &failures {
        println!("FAIL {f}");
    }
    if failures.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn print_compiled(path: &Path, p: &Program, wave32: bool, cfg: bool, tree: bool, ir: bool, raw: bool) {
    use ps5shader::ssa::Stage;
    use ps5shader::translate::{compile, TranslateOptions};
    let mut opts = TranslateOptions::with_user_data(if wave32 { 32 } else { 64 }, Stage::Unknown, 16);
    opts.optimize = !raw;
    // Plain blobs: the file is the code object (jump tables are read from it).
    let bytes = (!path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("fused-"))).then(|| std::fs::read(path).ok()).flatten();
    let slice = bytes.as_deref().map(|b| CodeSlice::new(p.address, b));
    let source = slice.as_ref().map(|s| s as &dyn CodeSource);
    match compile(p, &opts, source) {
        Ok(c) => {
            if cfg {
                print!("; cfg\n{}", c.structured.cfg);
            }
            if tree {
                print!("; region tree\n{}", c.structured.tree);
            }
            if ir || raw {
                print!("{}", c.function);
            }
            println!("; stats: {:?}", c.stats);
        }
        Err(e) => println!("; compile error: {e}"),
    }
}

/// `--bench <dir>`: compile every blob of `dir` (optimized, wave64) and
/// print the totals per phase and pass, and the slowest programs.
fn bench(dir: &Path) -> ExitCode {
    use ps5shader::opt::PASSES;
    use ps5shader::ssa::Stage;
    use ps5shader::translate::{compile, TranslateOptions, PHASES};
    let mut files: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "bin")).collect(),
        Err(e) => {
            eprintln!("{}: {e}", dir.display());
            return ExitCode::FAILURE;
        }
    };
    files.sort();
    let mut opts = TranslateOptions::with_user_data(64, Stage::Unknown, 16);
    opts.optimize = true;
    let (mut phases, mut passes) = ([0u64; PHASES.len()], [0u64; PASSES.len()]);
    let mut slowest: Vec<(u64, PathBuf, usize, usize, [u64; PHASES.len()], [u64; PASSES.len()])> = Vec::new();
    let (mut failed, mut total) = (0usize, 0u64);
    for f in &files {
        let Ok(p) = decode_file(f) else {
            failed += 1;
            continue;
        };
        let Ok(bytes) = std::fs::read(f) else { continue };
        let slice = CodeSlice::new(p.address, &bytes);
        let t = std::time::Instant::now();
        let r = compile(&p, &opts, Some(&slice as &dyn CodeSource));
        let ns = t.elapsed().as_nanos() as u64;
        total += ns;
        match r {
            Ok(c) => {
                for (a, b) in phases.iter_mut().zip(c.stats.phase_ns) {
                    *a += b;
                }
                for (a, b) in passes.iter_mut().zip(c.stats.opt.pass_ns) {
                    *a += b;
                }
                slowest.push((ns, f.clone(), p.instructions.len(), c.stats.opt.rounds, c.stats.phase_ns, c.stats.opt.pass_ns));
            }
            Err(_) => failed += 1,
        }
    }
    let ms = |ns: u64| ns / 1_000_000;
    println!("programs: {}  failed: {failed}  total: {} ms", files.len(), ms(total));
    println!("phases ms: {}", PHASES.iter().zip(phases).map(|(n, v)| format!("{n} {}", ms(v))).collect::<Vec<_>>().join(" "));
    println!("passes ms: {}", PASSES.iter().zip(passes).map(|(n, v)| format!("{n} {}", ms(v))).collect::<Vec<_>>().join(" "));
    slowest.sort_by(|a, b| b.0.cmp(&a.0));
    for (ns, f, n, rounds, ph, pa) in slowest.iter().take(15) {
        let ph: Vec<String> = PHASES.iter().zip(ph).filter(|(_, v)| **v >= 1_000_000).map(|(n, v)| format!("{n} {}", ms(*v))).collect();
        let pa: Vec<String> = PASSES.iter().zip(pa).filter(|(_, v)| **v >= 1_000_000).map(|(n, v)| format!("{n} {}", ms(*v))).collect();
        println!("{:>6} ms  {:>6} insts  {rounds:>2} rounds  {}  [{}] [{}]", ms(*ns), n, f.display(), ph.join(" "), pa.join(" "));
    }
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("usage: ps5shader-dis [--analyses] [--wave32] [--cfg] [--tree] [--ir] [--ir-raw] <blob.bin | fused-*.json>...\n       ps5shader-dis --stats <dir>\n       ps5shader-dis --bench <dir>");
        return ExitCode::FAILURE;
    }
    if args[0] == "--list-opcodes" {
        // Every opcode the decoder names (FLAT family per segment), one
        // mnemonic per line: the input of progress/progress.py.
        for op in ps5shader::Opcode::NAMED {
            println!("{}", op.mnemonic());
        }
        for seg in [ps5shader::FlatSegment::Flat, ps5shader::FlatSegment::Scratch, ps5shader::FlatSegment::Global] {
            for f in ps5shader::FlatOp::ALL {
                println!("{}", ps5shader::Opcode::Flat(seg, f).mnemonic());
            }
        }
        return ExitCode::SUCCESS;
    }
    if args[0] == "--stats" {
        return stats(Path::new(args.get(1).map_or(".", |s| s.as_str())));
    }
    if args[0] == "--bench" {
        return bench(Path::new(args.get(1).map_or(".", |s| s.as_str())));
    }
    if args[0] == "--stats-mods" {
        return stats_mods(Path::new(args.get(1).map_or(".", |s| s.as_str())));
    }
    let analyses = args.iter().any(|a| a == "--analyses");
    let wave32 = args.iter().any(|a| a == "--wave32");
    let show_cfg = args.iter().any(|a| a == "--cfg");
    let show_tree = args.iter().any(|a| a == "--tree");
    let show_ir = args.iter().any(|a| a == "--ir");
    let show_raw = args.iter().any(|a| a == "--ir-raw");
    let disasm = !(show_cfg || show_tree || show_ir || show_raw) || analyses;
    args.retain(|a| !a.starts_with("--"));
    let mut status = ExitCode::SUCCESS;
    for a in &args {
        let path = Path::new(a);
        println!("; {}", path.display());
        match decode_file(path) {
            Ok(p) => {
                if disasm {
                    print!("{p}");
                }
                if analyses {
                    print_analyses(&p, wave32);
                }
                if show_cfg || show_tree || show_ir || show_raw {
                    print_compiled(path, &p, wave32, show_cfg, show_tree, show_ir, show_raw);
                }
            }
            Err(e) => {
                println!("; decode error: {e}");
                status = ExitCode::FAILURE;
            }
        }
    }
    status
}
