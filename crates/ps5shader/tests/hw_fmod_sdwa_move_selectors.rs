// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! v_mov_b32 SDWA: every src0_sel x sext x dst_sel x dst_unused combination,
//! plus an EXEC-masked in-place byte broadcast, in wave32 and wave64,
//! against the reference model. The code is generated exactly like the
//! reference test builds it. (The reference's third "wave64 split" run only
//! changes the Vulkan subgroup size of its backend and is not applicable.)

include!("vectors/fmod_common.rs");
include!("vectors/fmod_sdwa_move_selectors.rs");

const SELECTORS: u32 = 7;
const UNUSED_MODES: u32 = 3;
const COMBINATIONS: u32 = SELECTORS * 2 * SELECTORS * UNUSED_MODES;
const BROADCAST_RESULT: u32 = COMBINATIONS;
const INPUTS: usize = 2;
const RESULTS: u32 = COMBINATIONS + 1;
const INPUT_INDEX_VECTOR: u32 = 1;
const OUTPUT_INDEX_VECTOR: u32 = 2;
const BROADCAST_VECTOR: u32 = 3;
const SOURCE_VECTOR: u32 = 4;
const PREVIOUS_VECTOR: u32 = 5;
const DATA_VECTOR: u32 = 6;
const THREAD_VECTOR: u32 = 0;
const EXEC_LO: u32 = 0x7e;
const EXEC_HI: u32 = 0x7f;
const INPUT_REGISTER: u32 = 0;
const OUTPUT_REGISTER: u32 = 4;
const CONSTANT_ZERO: u32 = 0x80;
const CONSTANT_ONE: u32 = 0x81;
const CONSTANT_ALL_ONES: u32 = 0xc1;
const LITERAL: u32 = 0xff;
const SDWA: u32 = 0xf9;
const VECTOR_SOURCE: u32 = 0x100;
const ENDPGM: u32 = 0xbf81_0000;

fn sop1(op: u32, d: u32, s: u32) -> u32 {
    0xbe80_0000 | (d << 16) | (op << 8) | s
}

fn vop1(op: u32, d: u32, s: u32) -> u32 {
    0x7e00_0000 | (d << 17) | (op << 9) | s
}

fn vop2(op: u32, d: u32, s0: u32, s1: u32) -> u32 {
    (op << 25) | (d << 17) | (s1 << 9) | s0
}

fn sdwa_modifier(src: u32, dst_sel: u32, unused: u32, src_sel: u32, sext: u32) -> u32 {
    src | (dst_sel << 8) | (unused << 11) | (src_sel << 16) | (sext << 19)
}

fn buffer_access(data: u32, index: u32, descriptor: u32) -> u32 {
    (CONSTANT_ZERO << 24) | ((descriptor / 4) << 16) | (data << 8) | index
}

fn combination_index(src_sel: u32, sext: u32, dst_sel: u32, unused: u32) -> u32 {
    ((src_sel * 2 + sext) * SELECTORS + dst_sel) * UNUSED_MODES + unused
}

fn build_code(wave: u32) -> Vec<u32> {
    let mut code = vec![
        vop2(0x1a, INPUT_INDEX_VECTOR, CONSTANT_ONE, THREAD_VECTOR),
        vop2(0x0b, OUTPUT_INDEX_VECTOR, LITERAL, THREAD_VECTOR),
        RESULTS,
        0xe030_2000,
        buffer_access(SOURCE_VECTOR, INPUT_INDEX_VECTOR, INPUT_REGISTER),
        0xe030_2000 | 4,
        buffer_access(PREVIOUS_VECTOR, INPUT_INDEX_VECTOR, INPUT_REGISTER),
        0xbf8c_3f70,
        vop1(0x01, BROADCAST_VECTOR, VECTOR_SOURCE | SOURCE_VECTOR),
        sop1(0x03, EXEC_LO, LITERAL),
        BROADCAST_EXEC[0],
    ];
    if wave == 64 {
        code.push(sop1(0x03, EXEC_HI, LITERAL));
        code.push(BROADCAST_EXEC[1]);
    }
    for byte in 1..4 {
        code.push(vop1(0x01, BROADCAST_VECTOR, SDWA));
        code.push(sdwa_modifier(BROADCAST_VECTOR, byte, 2, 0, 0));
    }
    code.push(if wave == 64 { sop1(0x04, EXEC_LO, CONSTANT_ALL_ONES) } else { sop1(0x03, EXEC_LO, CONSTANT_ALL_ONES) });
    for src_sel in 0..SELECTORS {
        for sext in 0..2 {
            for dst_sel in 0..SELECTORS {
                for unused in 0..UNUSED_MODES {
                    code.push(vop1(0x01, DATA_VECTOR, VECTOR_SOURCE | PREVIOUS_VECTOR));
                    code.push(vop1(0x01, DATA_VECTOR, SDWA));
                    code.push(sdwa_modifier(SOURCE_VECTOR, dst_sel, unused, src_sel, sext));
                    code.push(0xe070_2000 | (combination_index(src_sel, sext, dst_sel, unused) * 4));
                    code.push(buffer_access(DATA_VECTOR, OUTPUT_INDEX_VECTOR, OUTPUT_REGISTER));
                }
            }
        }
    }
    code.push(0xe070_2000 | (BROADCAST_RESULT * 4));
    code.push(buffer_access(BROADCAST_VECTOR, OUTPUT_INDEX_VECTOR, OUTPUT_REGISTER));
    code.push(ENDPGM);
    code
}

fn source(tid: u32) -> u32 {
    SOURCES[(tid % 32) as usize].rotate_left((tid / 32) * 8)
}

fn previous(tid: u32) -> u32 {
    0xabcd_1234 ^ tid.wrapping_mul(0x0103_0507)
}

fn broadcast_active(tid: u32) -> bool {
    (BROADCAST_EXEC[(tid / 32) as usize] >> (tid % 32)) & 1 != 0
}

fn field_offset(sel: u32) -> u32 {
    if sel < 4 {
        sel * 8
    } else {
        (sel - 4) * 16
    }
}

fn field_mask(sel: u32) -> u32 {
    if sel < 4 {
        0xff
    } else {
        0xffff
    }
}

fn sign_extend_field(field: u32, mask: u32) -> u32 {
    if field & ((mask >> 1) + 1) != 0 {
        field | !mask
    } else {
        field
    }
}

fn select_source(value: u32, sel: u32, sext: bool) -> u32 {
    if sel == 6 {
        return value;
    }
    let field = (value >> field_offset(sel)) & field_mask(sel);
    if sext {
        sign_extend_field(field, field_mask(sel))
    } else {
        field
    }
}

fn place_destination(result: u32, sel: u32, unused: u32, previous: u32) -> u32 {
    if sel == 6 {
        return result;
    }
    let (offset, mask) = (field_offset(sel), field_mask(sel));
    let field = result & mask;
    match unused {
        0 => field << offset,
        1 => sign_extend_field(field, mask) << offset,
        _ => (previous & !(mask << offset)) | (field << offset),
    }
}

const NAMES: [&str; 7] = ["BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3", "WORD_0", "WORD_1", "DWORD"];
const UNUSED_NAMES: [&str; 3] = ["UNUSED_PAD", "UNUSED_SEXT", "UNUSED_PRESERVE"];

fn run(wave: u32) -> Vec<u32> {
    let mut input = vec![0u32; 64 * INPUTS];
    for tid in 0..64u32 {
        input[tid as usize * INPUTS] = source(tid);
        input[tid as usize * INPUTS + 1] = previous(tid);
    }
    let code = build_code(wave);
    match fmod_dispatch(&code, wave, wave, &input, 64 * RESULTS as usize, 4) {
        Ok(o) => o,
        Err(e) => panic!("{}", &e[..e.len().min(600)]),
    }
}

fn check_selectors(wave: u32) {
    let out = run(wave);
    let mut failures = Vec::new();
    for tid in 0..wave {
        let (src, prev) = (source(tid), previous(tid));
        let o = &out[(tid * RESULTS) as usize..];
        for src_sel in 0..SELECTORS {
            for sext in 0..2 {
                let selected = select_source(src, src_sel, sext != 0);
                for dst_sel in 0..SELECTORS {
                    for unused in 0..UNUSED_MODES {
                        let index = combination_index(src_sel, sext, dst_sel, unused) as usize;
                        let expected = place_destination(selected, dst_sel, unused, prev);
                        if o[index] != expected {
                            failures.push(format!(
                                "wave{wave} lane {tid} source {src:#010x} previous {prev:#010x}: v_mov_b32_sdwa dst_sel:{} dst_unused:{} src0_sel:{}{} is {:#010x}, expected {expected:#010x}",
                                NAMES[dst_sel as usize], UNUSED_NAMES[unused as usize], NAMES[src_sel as usize], if sext != 0 { " sext" } else { "" }, o[index]
                            ));
                        }
                    }
                }
            }
        }
    }
    fmod_report(&failures);
}

fn check_broadcast(wave: u32) {
    let out = run(wave);
    let mut failures = Vec::new();
    for tid in 0..wave {
        let src = source(tid);
        let actual = out[(tid * RESULTS + BROADCAST_RESULT) as usize];
        let expected = if broadcast_active(tid) { (src & 0xff).wrapping_mul(0x0101_0101) } else { src };
        if actual != expected {
            failures.push(format!("wave{wave} lane {tid} (active {}) in-place byte broadcast is {actual:#010x}, expected {expected:#010x}", broadcast_active(tid)));
        }
    }
    fmod_report(&failures);
}

#[test]
fn selectors_wave32() {
    check_selectors(32);
}

#[test]
fn selectors_wave64() {
    check_selectors(64);
}

#[test]
fn masked_broadcast_wave32() {
    check_broadcast(32);
}

#[test]
fn masked_broadcast_wave64() {
    check_broadcast(64);
}
