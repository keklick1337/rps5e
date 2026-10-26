// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The reference compute cases of `ps5shader/tests/ref_*.rs` through the
//! SPIR-V compiler on the device, compared word for word with the IR
//! interpreter (NaN payloads aside: the device returns its canonical NaN).
//! A mismatch is an emitter or host-driver divergence from the
//! interpreter, whatever the case's own expectation says (the ignored
//! cases run too: the interpreter holds our semantics, the device must
//! agree with it).
//!
//! Cases run when they need nothing beyond one work-group with the default
//! user data. Both sides run the same program on the same data: s[0:3] is
//! the result buffer's V# (as the case sizes it) and the helper stores
//! through s[48:51] are redirected to a group of four SGPRs among s4..s15
//! the program does not name (the device has 16 user SGPRs), holding a raw
//! V# of the whole 64 KiB buffer. Work-group ids must follow that group.
//!
//! Every case runs twice: with the compiler as configured, and with every
//! wave64 program that qualifies split over two 32-wide host subgroups
//! (`shader::wave_split`).
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored ref_device`
//! (`REF_DEVICE_CASE=Name[,Name...]` runs only those cases).

use super::*;
use crate::shader::ref_tests::all_cases;
use crate::shader::ref_tests::harness::RefCase;

/// The bytes the helper-store V# spans (and the result buffer has).
const HELPER_BYTES: u32 = 64 << 10;
const RAW_DWORD3: u32 = (3 << 28) | (20 << 12) | 0xFAC;

/// Known divergences: work-groups smaller than the wave, whose missing
/// lanes the host subgroup does not have.
const KNOWN_DIVERGENT: &[&str] = &["VectorCmpClassF16Wave64", "Wave64AppendConsumeHighHalf", "DsPermuteCapturedExecOffsetAndWrap"];

fn eligible(c: &RefCase) -> Result<(), &'static str> {
    if c.name.starts_with("Image") || c.name.starts_with("Bvh") {
        return Err("needs images");
    }
    if c.name.contains("Gds") || c.name.contains("OrderedCount") {
        return Err("needs GDS");
    }
    if c.compile_only {
        return Err("compile only");
    }
    if c.user_data.is_some() {
        return Err("explicit user data");
    }
    if !c.bda.is_empty() {
        return Err("flat/global mappings");
    }
    if c.dispatch.iter().any(|&d| d > 1) || c.compute.dispatch_dims {
        return Err("several work-groups");
    }
    if c.compute.thread_ids > 1 {
        return Err("thread id VGPRs y/z");
    }
    if c.compute.lds_dwords % 128 != 0 {
        return Err("LDS size below the 128-dword allocation granule");
    }
    if c.name.starts_with("ScalarGetpc") || c.name.starts_with("ScalarMemRealtime") {
        return Err("reads the code address or the clock");
    }
    Ok(())
}

/// Whether the program reads the words of the s[0:3] V# as values (an
/// offset or an address): those differ between the device (the buffer's
/// address) and the interpreter (0).
fn reads_buffer_words(program: &ps5shader::Program) -> bool {
    use ps5shader::Opcode::*;
    program.instructions.iter().any(|ins| {
        let low = |o: &ps5shader::Operand| o.kind == ps5shader::OperandKind::ScalarRegister && o.value < 4;
        match ins.encoding {
            ps5shader::Encoding::Smem | ps5shader::Encoding::Smrd => {
                let buffer = matches!(ins.opcode, SBufferLoadDword | SBufferLoadDwordx2 | SBufferLoadDwordx4 | SBufferLoadDwordx8 | SBufferLoadDwordx16);
                ins.sources.iter().enumerate().any(|(i, o)| low(o) && (i > 0 || !buffer))
            }
            ps5shader::Encoding::Mubuf | ps5shader::Encoding::Mtbuf => ins.sources.iter().any(|o| low(o) && o.value != 0),
            _ => ins.sources.iter().any(low),
        }
    })
}

fn is_helper_store(ins: &ps5shader::Instruction) -> bool {
    matches!(ins.encoding, ps5shader::Encoding::Mubuf | ps5shader::Encoding::Mtbuf) && (ins.words[1] >> 16) & 0x1F == 12
}

/// The case's code with MUBUF/MTBUF resources s[48:51] moved to the first
/// group s[4g..4g+3] (g in 1..=3) the program does not name, and the user
/// SGPR count that holds the group.
fn redirect_helper_stores(c: &RefCase) -> Result<(Vec<u32>, u32), String> {
    let bytes: Vec<u8> = c.code.iter().flat_map(|w| w.to_le_bytes()).collect();
    let program = ps5shader::decode_blob(&bytes, 0x10_0000_0000).map_err(|e| format!("decode: {e}"))?;
    // SGPRs the program names (64-bit operands and SMEM destinations
    // spread over the following registers).
    if reads_buffer_words(&program) {
        return Err("reads the s[0:3] words as values".into());
    }
    let mut used = [false; 128];
    for ins in program.instructions.iter().filter(|i| !is_helper_store(i)) {
        let span = match &ins.control {
            Some(ps5shader::Control::ScalarMemory(m)) => m.destination_count.max(2),
            _ => 2,
        };
        for o in ins.sources.iter().chain(&ins.destinations) {
            if o.kind == ps5shader::OperandKind::ScalarRegister {
                for r in o.value..(o.value + span).min(128) {
                    used[r as usize] = true;
                }
            }
        }
    }
    let ids = c.compute.group_id.iter().any(|&b| b) || c.compute.tg_size_en;
    let group = (1..4u32)
        .find(|&g| (4 * g..4 * g + 4).all(|r| !used[r as usize]) && (!ids || c.compute.workgroup_register == 4 * g + 4))
        .ok_or("no free SGPR group for the helper V#")?;
    let mut words = c.code.to_vec();
    for ins in program.instructions.iter().filter(|i| is_helper_store(i)) {
        let at = ins.pc as usize / 4 + 1;
        words[at] = (words[at] & !(0x1F << 16)) | (group << 16);
    }
    Ok((words, 4 * group + 4))
}

fn same(a: u32, e: u32) -> bool {
    let nan = |x: u32| x & 0x7FFF_FFFF > 0x7F80_0000;
    let nan16 = |x: u32| x & 0x7FFF > 0x7C00;
    a == e || (nan(a) && nan(e)) || {
        let h = |x: u32, y: u32| x == y || (nan16(x) && nan16(y));
        h(a & 0xFFFF, e & 0xFFFF) && h(a >> 16, e >> 16)
    }
}

pub(super) fn run_case(g: &mut Guest, rig: &mut Rig, c: &RefCase) -> Result<Vec<String>, String> {
    let words = c.expected.len();
    let (code, user_sgprs) = redirect_helper_stores(c)?;
    let code: &'static [u32] = Box::leak(code.into_boxed_slice());
    let group = (user_sgprs - 4) as usize;
    // The interpreter runs the same program with the same user data.
    let mut interp_user = c.user_data();
    interp_user[group..group + 4].copy_from_slice(&[0, 0, HELPER_BYTES, RAW_DWORD3]);
    let same_setup = RefCase { code, user_data: Some(Box::leak(interp_user.clone().into_boxed_slice())), ..*c };
    // REF_DEVICE_EXPECTED=1 compares with the case's own expectation.
    let want = if std::env::var_os("REF_DEVICE_EXPECTED").is_some() { c.expected.to_vec() } else { same_setup.execute(true, words)? };
    let cs = g.program(code);
    let mut initial: Vec<u8> = c.initial.iter().flat_map(|w| w.to_le_bytes()).collect();
    initial.resize(HELPER_BYTES as usize, 0);
    let region = g.alloc(&initial);
    let mut user = interp_user[..user_sgprs as usize].to_vec();
    for q in [0, group] {
        user[q] = region as u32;
        user[q + 1] |= (region >> 32) as u32 & 0xFFFF;
    }
    let mut shader = stage(cs, &user);
    let mode = c.compute.float_mode & 0xFF;
    shader.rsrc1 = (mode << 12) | (1 << 21);
    let lds_blocks = c.compute.lds_dwords.div_ceil(128).min(0x1FF);
    let g_ids = c.compute.group_id;
    shader.rsrc2 = (user_sgprs << 1)
        | (g_ids[0] as u32) << 7
        | (g_ids[1] as u32) << 8
        | (g_ids[2] as u32) << 9
        | (c.compute.tg_size_en as u32) << 10
        | lds_blocks << 15;
    let t = c.compute.threads;
    let d = ResolvedDispatch {
        packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1 | if c.compute.wave_size == 32 { 1 << 15 } else { 0 }, ..DispatchPacket::default() },
        shader,
        rsrc3: 0,
        threads: [t[0].max(1), t[1].max(1), t[2].max(1)],
        start: [0; 3],
    };
    let outcome = rig.dispatch(&d);
    if outcome != DrawOutcome::Recorded {
        return Err(format!("dispatch {outcome:?}"));
    }
    let bytes = (words * 4).max(4) as u64;
    if !rig.buffers.download(&mut rig.sched, region, bytes) {
        return Err("download failed".into());
    }
    let mut out = vec![0u8; bytes as usize];
    if !g.mem.read_backing(region, &mut out) {
        return Err("read back failed".into());
    }
    let got: Vec<u32> = out.chunks(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect();
    Ok(got
        .iter()
        .zip(&want)
        .enumerate()
        .filter(|(_, (a, e))| !same(**a, **e))
        .map(|(i, (a, e))| format!("{} word {i}: device {a:#010x}, interpreter {e:#010x}", c.name))
        .collect())
}

#[test]
#[ignore]
fn ref_device_cases_match_the_interpreter() {
    let _ = env_logger::builder().is_test(true).try_init();
    let only = std::env::var("REF_DEVICE_CASE").ok();
    let (mut ran, mut skipped) = (0, Vec::new());
    let mut fails = Vec::new();
    // As configured, then with every qualifying wave64 program split over
    // two host subgroups (when the device's subgroups are 32 wide).
    for split_all in [false, true] {
        let mut g = Guest::new(256 << 20);
        let mem = g.mem;
        let mut rig = Rig::with_compiler(
            &g,
            |dev| {
                let mut c = crate::shader::SpirvCompiler::for_device(dev, mem);
                if split_all {
                    c.set_wave64_split(true, true);
                }
                Arc::new(c)
            },
            None,
            None,
        );
        let tag = if split_all { "[split] " } else { "" };
        for c in all_cases() {
            if only.as_deref().is_some_and(|n| !n.split(',').any(|n| n == c.name)) {
                continue;
            }
            if let Err(why) = eligible(c) {
                skipped.push(format!("{} ({why})", c.name));
                continue;
            }
            if let Err(why) = redirect_helper_stores(c) {
                skipped.push(format!("{} ({why})", c.name));
                continue;
            }
            ran += 1;
            match run_case(&mut g, &mut rig, c) {
                Ok(diff) if KNOWN_DIVERGENT.contains(&c.name) => {
                    if diff.is_empty() {
                        fails.push(format!("{tag}{}: matches now, drop it from KNOWN_DIVERGENT", c.name));
                    }
                }
                Ok(diff) => fails.extend(diff.into_iter().map(|d| format!("{tag}{d}"))),
                Err(e) => fails.push(format!("{tag}{}: {e}", c.name)),
            }
        }
    }
    eprintln!("ref_device: {ran} cases ran, {} skipped", skipped.len());
    let mut cases: Vec<&str> = fails.iter().map(|f| f.trim_start_matches("[split] ").split([' ', ':']).next().unwrap_or("")).collect();
    cases.dedup();
    assert!(
        fails.is_empty(),
        "{} mismatches in {} cases ({}):\n{}",
        fails.len(),
        cases.len(),
        cases.join(", "),
        fails.iter().take(120).cloned().collect::<Vec<_>>().join("\n")
    );
}

/// A device without shaderFloat64, simulated on this one: every case
/// whose program does not compute in f64 runs and matches the interpreter;
/// the others are refused.
#[test]
#[ignore]
fn ref_device_cases_without_float64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut g = Guest::new(256 << 20);
    let mem = g.mem;
    let mut rig = Rig::with_compiler(
        &g,
        |dev| Arc::new(crate::shader::SpirvCompiler::new(crate::shader::Features { no_float64: true, ..crate::shader::Features::of(dev) }, Some(mem))),
        None,
        None,
    );
    let (mut ran, mut refused, mut fails) = (0, 0, Vec::new());
    for c in all_cases() {
        if eligible(c).is_err() || redirect_helper_stores(c).is_err() || KNOWN_DIVERGENT.contains(&c.name) {
            continue;
        }
        match run_case(&mut g, &mut rig, c) {
            Ok(diff) => {
                ran += 1;
                fails.extend(diff);
            }
            // Programs computing in f64 are refused (their dispatch skipped).
            Err(e) if e.starts_with("dispatch") && crate::shader::ref_tests::refused_without_float64(c) => refused += 1,
            Err(e) => fails.push(format!("{}: {e}", c.name)),
        }
    }
    eprintln!("without f64: {ran} cases ran, {refused} refused");
    assert!(ran > 0 && refused > 0);
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}
