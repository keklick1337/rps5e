// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

// Shared dispatch helper for the hardware-measured reference rows ports
// (`hw_fmod_*`): s[0:3] = input V#, s[4:7] = output V#, v0 = thread id.

#[allow(unused_imports)]
use ps5shader::testing::{buffer_descriptor, ComputeRun};

#[allow(dead_code)]
const FMOD_INPUT_BASE: u64 = 0x1_0000_0000;
#[allow(dead_code)]
const FMOD_OUTPUT_BASE: u64 = 0x2_0000_0000;

/// Descriptor in one of the two reference layouts: `stride == 0` is a raw
/// buffer sized in bytes, otherwise `stride` bytes per record and one record
/// per word.
#[allow(dead_code)]
fn fmod_descriptor(base: u64, words: usize, stride: u32) -> [u32; 4] {
    let records = if stride == 0 { words as u32 * 4 } else { words as u32 };
    // Raw (OOB_SELECT 3) or index-checked (OOB_SELECT 1), as the reference
    // programs were run.
    buffer_descriptor(base, stride, records, if stride == 0 { 0x3101_6FAC } else { 0x1101_6FAC })
}

/// Run `threads` invocations of one workgroup as consecutive waves of
/// `wave` lanes sharing the same memory.
#[allow(dead_code)]
fn fmod_run_waves(run: &ComputeRun, optimize: bool, threads: u32, input: &[u32], out_words: usize) -> Result<Vec<u32>, String> {
    let f = run.compile(optimize)?;
    let mut memory = ps5shader::interp::Memory::default();
    memory.write_words(FMOD_INPUT_BASE, input);
    memory.write_words(FMOD_OUTPUT_BASE, &vec![0xDEAD_BEEF; out_words]);
    memory.write_words(run.code_address, &run.code);
    let mut first = 0;
    while first < threads {
        let lanes = run.wave_size.min(threads - first);
        let mut m = ps5shader::interp::Machine::new(run.wave_size, lanes);
        m.user_data = run.user_data.clone();
        m.memory = std::mem::take(&mut memory);
        let ids: Vec<u32> = (first..first + lanes).collect();
        m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(0), ids);
        m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(1), vec![0; lanes as usize]);
        m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(2), vec![0; lanes as usize]);
        m.run(&f).map_err(|e| format!("run: {e}"))?;
        memory = std::mem::take(&mut m.memory);
        first += lanes;
    }
    Ok(memory.read_words(FMOD_OUTPUT_BASE, out_words))
}

/// One-workgroup dispatch: `input` is the input buffer contents, the output
/// buffer (`out_words` words) is prefilled with 0xdeadbeef. Runs both the
/// unoptimized and the optimized pipelines and requires them to agree.
#[allow(dead_code)]
fn fmod_dispatch(code: &[u32], wave: u32, threads: u32, input: &[u32], out_words: usize, stride: u32) -> Result<Vec<u32>, String> {
    fmod_dispatch_mode(code, wave, threads, input, out_words, stride, ps5shader::translate::FloatMode::default())
}

/// `fmod_dispatch` compiled for the float controls `mode` (the rows of a
/// reference program measured in another FLOAT_MODE / IEEE_MODE).
#[allow(dead_code)]
fn fmod_dispatch_mode(code: &[u32], wave: u32, threads: u32, input: &[u32], out_words: usize, stride: u32, mode: ps5shader::translate::FloatMode) -> Result<Vec<u32>, String> {
    let mut user = vec![0u32; 8];
    user[..4].copy_from_slice(&fmod_descriptor(FMOD_INPUT_BASE, input.len(), stride));
    user[4..].copy_from_slice(&fmod_descriptor(FMOD_OUTPUT_BASE, out_words, stride));
    let mut run = ComputeRun::new(code.to_vec(), wave, threads.min(wave), user);
    run.float_mode = mode;
    let plain = fmod_run_waves(&run, false, threads, input, out_words)?;
    let opt = fmod_run_waves(&run, true, threads, input, out_words)?;
    if let Some(i) = plain.iter().zip(&opt).position(|(a, b)| a != b) {
        return Err(format!("optimized run differs at word {i}: {:#010x} vs {:#010x}", plain[i], opt[i]));
    }
    Ok(opt)
}

/// The common "Rows[lane][4] in, 16 result words per lane out" layout
/// (raw buffers, wave32, one lane per row).
#[allow(dead_code)]
fn fmod_run_rows(code: &[u32], rows: &[[u32; 4]]) -> Result<Vec<u32>, String> {
    let input: Vec<u32> = rows.iter().flatten().copied().collect();
    fmod_dispatch_mode(code, 32, rows.len() as u32, &input, rows.len() * 16, 0, FMOD_FLOAT_MODE.with(|m| *m.borrow()))
}

thread_local! {
    /// Float controls `fmod_run_rows` compiles for (the test thread sets
    /// them before checking rows measured outside the default mode).
    static FMOD_FLOAT_MODE: std::cell::RefCell<ps5shader::translate::FloatMode> = std::cell::RefCell::new(ps5shader::translate::FloatMode::default());
}

/// FLOAT_MODE 0xF0 with IEEE_MODE and DX10_CLAMP: the controls several
/// reference programs were measured with.
#[allow(dead_code)]
const FMOD_IEEE_F0: ps5shader::translate::FloatMode = ps5shader::translate::FloatMode { mode: 0xF0, dx10_clamp: true, ieee: true, fp16_overflow: false };

#[allow(dead_code)]
fn fmod_set_float_mode(mode: ps5shader::translate::FloatMode) {
    FMOD_FLOAT_MODE.with(|m| *m.borrow_mut() = mode);
}

/// Compare result word `column` of every lane against `expected[lane][column]`
/// (lanes whose `checked` bit for the column is clear are skipped).
#[allow(dead_code)]
fn fmod_check_rows<const N: usize>(code: &[u32], rows: &[[u32; 4]], expected: &[[u32; N]], checked: Option<&[u32]>, column: usize, name: &str, eq: &dyn Fn(u32, u32) -> bool) {
    let out = match fmod_run_rows(code, rows) {
        Ok(o) => o,
        Err(e) => panic!("{}", &e[..e.len().min(600)]),
    };
    let mut failures = Vec::new();
    for (lane, row) in rows.iter().enumerate() {
        if let Some(mask) = checked {
            if (mask[lane] >> column) & 1 == 0 {
                continue;
            }
        }
        let (a, e) = (out[lane * 16 + column], expected[lane][column]);
        if !eq(a, e) {
            failures.push(format!("lane {lane} {:08x?} {name}: {a:#010x}, expected {e:#010x}", row));
        }
    }
    fmod_report(&failures);
}

/// The cross-lane layout: all `rows` (64) are loaded, one workgroup of
/// `wave` threads runs `code` with that wave size, 16 result words per lane;
/// result `column` of lanes `0..expected.len()` is compared exactly.
#[allow(dead_code)]
fn fmod_check_wave<const N: usize>(code: &[u32], wave: u32, rows: &[[u32; 4]], expected: &[[u32; N]], column: usize, name: &str) {
    let input: Vec<u32> = rows.iter().flatten().copied().collect();
    let out = match fmod_dispatch(code, wave, wave, &input, rows.len() * 16, 0) {
        Ok(o) => o,
        Err(e) => panic!("{}", &e[..e.len().min(600)]),
    };
    let mut failures = Vec::new();
    for (lane, e) in expected.iter().enumerate() {
        let a = out[lane * 16 + column];
        if a != e[column] {
            failures.push(format!("wave{wave} lane {lane} {:08x?} {name}: {a:#010x}, expected {:#010x}", rows[lane], e[column]));
        }
    }
    fmod_report(&failures);
}

#[allow(dead_code)]
fn fmod_exact(a: u32, e: u32) -> bool {
    a == e
}

#[allow(dead_code)]
fn fmod_report(failures: &[String]) {
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.iter().take(60).cloned().collect::<Vec<_>>().join("\n"));
}
