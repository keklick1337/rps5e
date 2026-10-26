// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! SDWA source selects on v_cvt_f32_ubyte1/2/3: the selected (and
//! optionally sign-extended) source is SRC0, then byte N of it converts
//! (RDNA 2 ISA 6.8, 13.3.7, 12.8). WORD_0 / WORD_1 on ubyte1, a sign-
//! extended WORD_1 on ubyte2 and a sign-extended BYTE_3 on ubyte3, wave32
//! over 64 threads (the reference's SdwaCvtF32Ubyte execution test).

use ps5shader::interp::Machine;
use ps5shader::testing::{buffer_descriptor, ComputeRun};

const IN_BASE: u64 = 0x1_0000_0000;
const OUT_BASE: u64 = 0x2_0000_0000;

const CODE: [u32; 30] = [
    0x34020082, 0x34060082, 0xe0302000, 0x80000401, 0x7e1402ff, 0xabcd1234, 0x7e1602ff, 0xabcd1234, 0x7e1802ff, 0xabcd1234, 0x7e1a02ff, 0xabcd1234, 0xbf8c3f70, 0x7e1424f9,
    0x00040604, 0x7e1624f9, 0x00050604, 0x7e1826f9, 0x000d0604, 0x7e1a28f9, 0x000b0604, 0xe0702000, 0x80010a03, 0xe0702004, 0x80010b03, 0xe0702008, 0x80010c03, 0xe070200c,
    0x80010d03, 0xbf810000,
];

#[test]
fn sdwa_source_selects_on_byte_conversions() {
    let mut failures = Vec::new();
    for wave in 0..2u32 {
        let mut input = vec![0u32; 64 * 4];
        for tid in 0..64u32 {
            input[tid as usize * 4] = tid | (tid + 64) << 8 | (255 - tid) << 16 | ((tid * 3 + 128) & 0xff) << 24;
        }
        let mut user = buffer_descriptor(IN_BASE, 4, input.len() as u32, 0x1101_6FAC).to_vec();
        user.extend(buffer_descriptor(OUT_BASE, 4, 64 * 4, 0x1101_6FAC));
        // Each wave's lanes get their own thread ids (v0 = tid).
        let first = wave * 32;
        let mut run = ComputeRun::new(CODE.to_vec(), 32, 32, user);
        run.thread_id_vgprs = 1;
        let input2 = input.clone();
        let setup = move |m: &mut Machine| {
            m.memory.write_words(IN_BASE, &input2);
            m.memory.write_words(OUT_BASE, &vec![0xdead_beef; 64 * 4]);
            m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(0), (first..first + 32).collect());
        };
        let m = run.run_checked(&setup, &[(OUT_BASE, 64 * 4)]).unwrap();
        let out = m.memory.read_words(OUT_BASE, 64 * 4);
        for tid in first..first + 32 {
            let source = input[tid as usize * 4];
            let byte1 = (((source >> 8) & 0xff) as f32).to_bits();
            let byte3 = ((source >> 24) as f32).to_bits();
            let fill = (if source & 0x8000_0000 != 0 { 255.0f32 } else { 0.0 }).to_bits();
            let got = &out[tid as usize * 4..tid as usize * 4 + 4];
            for (i, want) in [byte1, byte3, fill, fill].into_iter().enumerate() {
                if got[i] != want {
                    failures.push(format!("thread {tid} result {i}: {:#010x}, expected {want:#010x}", got[i]));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
}

/// Measured on RDNA2 (gfx1036): v_cvt_f32_ubyte0..3 with every SRC0_SEL
/// (BYTE_0..3, WORD_0/1, DWORD), with and without SRC0_SEXT, over 16
/// sources. Byte N converts from the selected, extended source.
const SELECT_VECTORS: [(u32, [u32; 7], [u32; 7]); 16] = [
    (0x00000000, [0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000], [0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000]),
    (0xffffffff, [0x000000ff, 0x000000ff, 0x000000ff, 0x000000ff, 0x0000ffff, 0x0000ffff, 0xffffffff], [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]),
    (0x12345678, [0x00000078, 0x00000056, 0x00000034, 0x00000012, 0x00005678, 0x00001234, 0x12345678], [0x00000078, 0x00000056, 0x00000034, 0x00000012, 0x00005678, 0x00001234, 0x12345678]),
    (0x80808080, [0x00000080, 0x00000080, 0x00000080, 0x00000080, 0x00008080, 0x00008080, 0x80808080], [0xffffff80, 0xffffff80, 0xffffff80, 0xffffff80, 0xffff8080, 0xffff8080, 0x80808080]),
    (0x7f7f7f7f, [0x0000007f, 0x0000007f, 0x0000007f, 0x0000007f, 0x00007f7f, 0x00007f7f, 0x7f7f7f7f], [0x0000007f, 0x0000007f, 0x0000007f, 0x0000007f, 0x00007f7f, 0x00007f7f, 0x7f7f7f7f]),
    (0x8182a3f4, [0x000000f4, 0x000000a3, 0x00000082, 0x00000081, 0x0000a3f4, 0x00008182, 0x8182a3f4], [0xfffffff4, 0xffffffa3, 0xffffff82, 0xffffff81, 0xffffa3f4, 0xffff8182, 0x8182a3f4]),
    (0x7f00ff80, [0x00000080, 0x000000ff, 0x00000000, 0x0000007f, 0x0000ff80, 0x00007f00, 0x7f00ff80], [0xffffff80, 0xffffffff, 0x00000000, 0x0000007f, 0xffffff80, 0x00007f00, 0x7f00ff80]),
    (0x00008000, [0x00000000, 0x00000080, 0x00000000, 0x00000000, 0x00008000, 0x00000000, 0x00008000], [0x00000000, 0xffffff80, 0x00000000, 0x00000000, 0xffff8000, 0x00000000, 0x00008000]),
    (0x0000007f, [0x0000007f, 0x00000000, 0x00000000, 0x00000000, 0x0000007f, 0x00000000, 0x0000007f], [0x0000007f, 0x00000000, 0x00000000, 0x00000000, 0x0000007f, 0x00000000, 0x0000007f]),
    (0x00000080, [0x00000080, 0x00000000, 0x00000000, 0x00000000, 0x00000080, 0x00000000, 0x00000080], [0xffffff80, 0x00000000, 0x00000000, 0x00000000, 0x00000080, 0x00000000, 0x00000080]),
    (0x80000000, [0x00000000, 0x00000000, 0x00000000, 0x00000080, 0x00000000, 0x00008000, 0x80000000], [0x00000000, 0x00000000, 0x00000000, 0xffffff80, 0x00000000, 0xffff8000, 0x80000000]),
    (0x00800000, [0x00000000, 0x00000000, 0x00000080, 0x00000000, 0x00000000, 0x00000080, 0x00800000], [0x00000000, 0x00000000, 0xffffff80, 0x00000000, 0x00000000, 0x00000080, 0x00800000]),
    (0x0000ff00, [0x00000000, 0x000000ff, 0x00000000, 0x00000000, 0x0000ff00, 0x00000000, 0x0000ff00], [0x00000000, 0xffffffff, 0x00000000, 0x00000000, 0xffffff00, 0x00000000, 0x0000ff00]),
    (0xdeadbeef, [0x000000ef, 0x000000be, 0x000000ad, 0x000000de, 0x0000beef, 0x0000dead, 0xdeadbeef], [0xffffffef, 0xffffffbe, 0xffffffad, 0xffffffde, 0xffffbeef, 0xffffdead, 0xdeadbeef]),
    (0x01020304, [0x00000004, 0x00000003, 0x00000002, 0x00000001, 0x00000304, 0x00000102, 0x01020304], [0x00000004, 0x00000003, 0x00000002, 0x00000001, 0x00000304, 0x00000102, 0x01020304]),
    (0xf0e0d0c0, [0x000000c0, 0x000000d0, 0x000000e0, 0x000000f0, 0x0000d0c0, 0x0000f0e0, 0xf0e0d0c0], [0xffffffc0, 0xffffffd0, 0xffffffe0, 0xfffffff0, 0xffffd0c0, 0xfffff0e0, 0xf0e0d0c0]),
];

#[test]
fn sdwa_every_source_select_on_every_byte_conversion() {
    const NAMES: [&str; 7] = ["BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3", "WORD_0", "WORD_1", "DWORD"];
    let mut failures = Vec::new();
    let mut input = vec![0u32; 32 * 4];
    for tid in 0..32 {
        input[tid * 4] = SELECT_VECTORS[tid % 16].0;
    }
    for sext in [false, true] {
        for sel in 0..7u32 {
            let mut code = CODE;
            // v_cvt_f32_ubyte0..3, all with the same SDWA source select.
            for (word, op) in [(13usize, 0x7e1422f9u32), (15, 0x7e1624f9), (17, 0x7e1826f9), (19, 0x7e1a28f9)] {
                code[word] = op;
                code[word + 1] = 0x0000_0604 | sel << 16 | (sext as u32) << 19;
            }
            let mut user = buffer_descriptor(IN_BASE, 4, input.len() as u32, 0x1101_6FAC).to_vec();
            user.extend(buffer_descriptor(OUT_BASE, 4, 32 * 4, 0x1101_6FAC));
            let mut run = ComputeRun::new(code.to_vec(), 32, 32, user);
            run.thread_id_vgprs = 1;
            let input2 = input.clone();
            let setup = move |m: &mut Machine| {
                m.memory.write_words(IN_BASE, &input2);
                m.memory.write_words(OUT_BASE, &vec![0xdead_beef; 32 * 4]);
            };
            let m = run.run_checked(&setup, &[(OUT_BASE, 32 * 4)]).unwrap();
            let out = m.memory.read_words(OUT_BASE, 32 * 4);
            for tid in 0..32usize {
                let (value, zero, sign) = SELECT_VECTORS[tid % 16];
                let selected = if sext { sign[sel as usize] } else { zero[sel as usize] };
                for byte in 0..4 {
                    let want = (((selected >> (byte * 8)) & 0xff) as f32).to_bits();
                    let got = out[tid * 4 + byte];
                    if got != want {
                        failures.push(format!("ubyte{byte} sel {}{} of {value:#010x}: {got:#010x}, expected {want:#010x}", NAMES[sel as usize], if sext { " sext" } else { "" }));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
}
