// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! MODE register reads and writes (the reference's ShaderModeRegisters
//! cases): the round fields (MODE[3:0]) start from the declared FLOAT_MODE
//! and follow s_round_mode / s_setreg_imm32_b32 / s_setreg_b32 writes
//! through the control flow; s_getreg_b32 reads them. Refused (strict MODE
//! writes, as the tests compile): reads of round bits that differ between
//! the paths reaching them or come from an SGPR, a directed mode set at
//! run time before an instruction that only rounds to nearest, an f16
//! conversion in a run-time round mode with f16 denormals flushed, and
//! writes of other MODE fields or hardware registers.

use ps5shader::testing::asm::*;
use ps5shader::testing::{ComputeRun, IoTest};
use ps5shader::translate::FloatMode;

fn mode(m: u8) -> FloatMode {
    FloatMode { mode: m, dx10_clamp: true, ieee: false, fp16_overflow: false }
}

const END: u32 = S_ENDPGM;
const GETREG_MODE: u32 = 0xB914_1801;
const V_ADD_F32: u32 = 0x0600_0300;
const V_CVT_F32_I32: u32 = 0x7E00_0B00;
const V_CVT_F32_U32: u32 = 0x7E00_0D00;
const V_CVT_F16_F32: u32 = 0x7E00_1500;
const V_ADD_F16: u32 = 0x6400_0300;

fn round_mode(value: u32) -> u32 {
    0xBFA4_0000 | value
}

/// `code` with its final s_endpgm replaced by: s20 stored to output[0].
fn read_s20(code: &[u32], m: FloatMode) -> Result<u32, String> {
    let mut code = code.to_vec();
    assert_eq!(code.pop(), Some(END));
    code.push(vop1(OP_V_MOV_B32, 1, 20));
    code.extend(mubuf(OP_BUFFER_STORE_DWORD, 1, 0, 1, inline_u32(0), 0, true, false, false));
    code.push(END);
    let out = IoTest::new(&code, 1, 1).with_float_mode(m).run(&[0])?;
    Ok(out[0])
}

fn compile(code: &[u32], m: FloatMode) -> Result<String, String> {
    let mut run = ComputeRun::new(code.to_vec(), 32, 1, vec![0; 8]);
    run.float_mode = m;
    run.compile(true).map(|f| f.to_string())
}

fn accepted(code: &[u32], m: FloatMode) {
    if let Err(e) = compile(code, m) {
        panic!("{code:08x?} in mode {:#x} was refused: {e}", m.mode);
    }
}

fn rejected(code: &[u32], m: FloatMode, diagnostic: &str, pc: u32) {
    match compile(code, m) {
        Ok(_) => panic!("{code:08x?} in mode {:#x} was accepted ({diagnostic})", m.mode),
        Err(e) => assert!(e.contains(diagnostic) && e.contains(&format!("at {pc:#x}")), "unexpected diagnostic for {code:08x?}: {e}"),
    }
}

/// v1 = `value` (an integer), `instruction` (v0 = op(v1)-like encodings
/// rewritten to v1 = op(v1)), v1 stored: the result bits.
fn convert(prefix: &[u32], instruction: u32, value: u32, m: FloatMode) -> u32 {
    let mut code = prefix.to_vec();
    code.extend([0x7E02_02FF, value]);
    // vdst v1, src0 v1.
    code.push(instruction & !(0xFF << 17) & !0x1FF | 1 << 17 | 0x101);
    code.extend(mubuf(OP_BUFFER_STORE_DWORD, 1, 0, 1, inline_u32(0), 0, true, false, false));
    code.push(END);
    IoTest::new(&code, 1, 1).with_float_mode(m).run(&[0]).unwrap()[0]
}

#[test]
fn reads_give_the_declared_round_fields() {
    // s_getreg_b32 s20.. hwreg(MODE, offset, size).
    let reads = [(0xB914_0001u32, 0u32, 1u32), (0xB914_0801, 0, 3), (0xB914_1001, 0, 7), (0xB914_1801, 0, 15), (0xB914_0041, 1, 1), (0xB914_0841, 1, 3), (0xB914_1041, 1, 7), (0xB914_0081, 2, 1), (0xB914_0881, 2, 3), (0xB914_00C1, 3, 1)];
    for m in [0xC0u8, 0xC5, 0xCF, 0x0A, 0xF3] {
        for (word, offset, mask) in reads {
            let got = read_s20(&[word, END], mode(m)).unwrap();
            assert_eq!(got, (m as u32 >> offset) & mask, "mode {m:#x} read {word:#010x}");
        }
    }
}

#[test]
fn dynamic_writes() {
    for instruction in [0xB980_0001u32, 0xB980_1801, 0xB980_0041] {
        accepted(&[instruction, END], mode(0xC0));
        rejected(&[instruction, GETREG_MODE, END], mode(0xC0), "reads MODE rounding bits", 4);
        rejected(&[instruction, V_ADD_F32, END], mode(0xC0), "reads MODE rounding bits", 4);
    }
    for instruction in [0xB980_1881u32, 0xB980_F801, 0xB980_0101] {
        rejected(&[instruction, END], mode(0xC0), "other than the MODE rounding fields", 0);
    }
}

#[test]
fn dynamic_write_survives_partial_overwrite() {
    let m = mode(0xC0);
    assert_eq!(read_s20(&[0xB980_1801, 0xBA80_1801, 0, GETREG_MODE, V_ADD_F32, END], m).unwrap(), 0);
    rejected(&[0xB980_1801, 0xBA80_0801, 0, GETREG_MODE, V_ADD_F32, END], m, "reads MODE rounding bits", 12);
    accepted(&[0xB980_1801, 0xBA80_0801, 0, 0xBA80_0881, 0, END], m);
}

#[test]
fn immediate_writes() {
    let instructions = [0xBA80_1801u32, 0xBA80_0801, 0xBA80_0881, 0xBA80_0841];
    let masks = [15u32, 3, 12, 6];
    let offsets = [0u32, 0, 2, 1];
    for rounding in 0..16u32 {
        for value in 0..16u32 {
            for f in 0..4 {
                let code = [instructions[f], 0xFFFF_FFF0 | value, GETREG_MODE, END];
                let expected = (rounding & !masks[f]) | ((value << offsets[f]) & masks[f]);
                assert_eq!(read_s20(&code, mode(0xC0 | rounding as u8)).unwrap(), expected, "rounding {rounding} value {value} field {f}");
            }
        }
    }
    // A denormal field, another hardware register.
    rejected(&[0xBA80_1881, 0, END], mode(0xC0), "other than the MODE rounding fields", 0);
    rejected(&[0xBA80_1802, 0, END], mode(0xC0), "other than the MODE rounding fields", 0);
}

#[test]
fn round_mode_instruction() {
    for rounding in 0..16u32 {
        for value in 0..16u32 {
            for upper in [0u32, 0x8000] {
                let code = [round_mode(upper | value), GETREG_MODE, END];
                assert_eq!(read_s20(&code, mode(0xC0 | rounding as u8)).unwrap(), value);
            }
        }
    }
}

#[test]
fn other_hardware_register_reads_are_refused() {
    for word in [0xB914_0002u32, 0xB914_0101, 0xB914_08C1, 0xB914_F801] {
        let mut run = ComputeRun::new(vec![word, END], 32, 1, vec![0; 8]);
        run.float_mode = mode(0xC0);
        assert!(run.compile(true).is_err(), "{word:#010x} was accepted");
    }
}

#[test]
fn branch() {
    for rounding in 0..16u32 {
        let m = mode(0xC0 | rounding as u8);
        let mut code = [0xBF84_0002, 0xBA80_1801, rounding, GETREG_MODE, END];
        assert_eq!(read_s20(&code, m).unwrap(), rounding);
        code[2] ^= 1;
        rejected(&code, m, "reads MODE rounding bits", 12);
    }
}

#[test]
fn diamond() {
    for rounding in 0..16u32 {
        let m = mode(0xC0 | rounding as u8);
        let other = rounding ^ 8;
        let same = [0xBF84_0003, 0xBA80_1801, other, 0xBF82_0002, 0xBA80_1801, other, GETREG_MODE, END, END];
        let mut same_read = same.to_vec();
        same_read.pop();
        assert_eq!(read_s20(&same_read, m).unwrap(), other);
        let differing = [0xBF84_0003, 0xBA80_1801, other, 0xBF82_0002, 0xBA80_1801, rounding, GETREG_MODE, END, END];
        rejected(&differing, m, "reads MODE rounding bits", 24);
        let differing_use = [0xBF84_0003, 0xBA80_1801, 3, 0xBF82_0002, 0xBA80_1801, 0, V_CVT_F32_I32, END, END];
        if rounding == 0 {
            rejected(&differing_use, m, "reads MODE rounding bits", 24);
        }
    }
}

/// A round mode set at run time before v_cvt_f32_i32 / u32 converts as the
/// same mode declared.
#[test]
fn rounding_before_directed32_use() {
    let values = [0x7FFF_FFFFu32, 0x8000_0001, 0x0100_0001, 0xFEFF_FFFF, 0x0123_4567, 0xFFFF_FFFF, 0x0000_0001];
    for instruction in [V_CVT_F32_I32, V_CVT_F32_U32] {
        for rounding in 1..4u32 {
            let mut changed = false;
            for value in values {
                let runtime = convert(&[round_mode(rounding)], instruction, value, mode(0xC0));
                let declared = convert(&[], instruction, value, mode(0xC0 | rounding as u8));
                assert_eq!(runtime, declared, "{instruction:#x} round {rounding} of {value:#x}");
                changed |= runtime != convert(&[], instruction, value, mode(0xC0));
            }
            assert!(changed, "{instruction:#x}: run-time round mode {rounding} did not change the conversion");
        }
    }
}

#[test]
fn rounding_restored_before_use() {
    for instruction in [V_ADD_F32, V_ADD_F16] {
        for directed in 1..16u32 {
            let restored = [round_mode(directed), round_mode(0), instruction, round_mode(directed), END];
            assert_eq!(compile(&restored, mode(0xC0)).unwrap(), compile(&[instruction, END], mode(0xC0)).unwrap(), "{instruction:#x} directed {directed}");
        }
    }
}

#[test]
fn untouched_fields_stay_accepted() {
    let f16_only = [round_mode(8), V_ADD_F32, round_mode(0), END];
    assert_eq!(compile(&f16_only, mode(0xC0)).unwrap(), compile(&[V_ADD_F32, END], mode(0xC0)).unwrap());
    let f32_only = [round_mode(2), V_ADD_F16, round_mode(0), END];
    assert_eq!(compile(&f32_only, mode(0xC0)).unwrap(), compile(&[V_ADD_F16, END], mode(0xC0)).unwrap());
}

#[test]
fn unmodelled_directed_rounding() {
    for directed in 1..4u32 {
        rejected(&[round_mode(directed), V_ADD_F32, END], mode(0xC0), "does not model directed rounding", 4);
        rejected(&[round_mode(directed << 2), V_ADD_F16, END], mode(0xC0), "does not model directed rounding", 4);
    }
}

#[test]
fn directed_f16_conversion() {
    let values = [0x3F80_1001u32, 0xBF80_1001, 0x477F_F001, 0x3380_0001, 0xC77F_E001];
    for rounding in 1..4u32 {
        for value in values {
            let runtime = convert(&[round_mode(rounding << 2)], V_CVT_F16_F32, value, mode(0xC0));
            let declared = convert(&[], V_CVT_F16_F32, value, mode(0xC0 | (rounding << 2) as u8));
            assert_eq!(runtime, declared, "round {rounding} of {value:#x}");
        }
        rejected(&[round_mode(rounding << 2), V_CVT_F16_F32, END], mode(0x00), "f16 denormals not kept", 4);
    }
}

#[test]
fn rounding_in_loop() {
    accepted(&[round_mode(0), V_ADD_F32, 0xBF84_FFFD, round_mode(0), END, END], mode(0xC0));
    accepted(&[round_mode(0), V_ADD_F32, 0xBF84_FFFD, round_mode(3), END, END], mode(0xC0));
    rejected(&[V_ADD_F32, round_mode(3), 0xBF84_FFFD, GETREG_MODE, END, END], mode(0xC0), "reads MODE rounding bits", 0);
}

/// Without strict MODE writes (the runtime default) nothing is refused:
/// unknown round bits read as the declared ones.
#[test]
fn lenient_mode_runs_unknown_bits_as_declared() {
    let mut run = ComputeRun::new(vec![0xB980_1801, V_ADD_F32, round_mode(3), V_ADD_F32, END], 32, 1, vec![0; 8]);
    run.float_mode = mode(0xC0);
    let mut options = run.options(true);
    options.strict_mode_writes = false;
    let bytes: Vec<u8> = run.code.iter().flat_map(|w| w.to_le_bytes()).collect();
    let program = ps5shader::program::decode_blob(&bytes, run.code_address).unwrap();
    assert!(ps5shader::translate::compile(&program, &options, None).is_ok());
}
