// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Instructions decoded by name: ds_condxchg32_rtn_b64 is translated (two
//! masked dword writes: a DATA0 dword with bit 31 set stores its low 31
//! bits, the old pair is returned); s_rfe_b64, s_sendmsghalt, s_call_b64,
//! s_scratch_load_dword* and ds_gws_* decode and are refused.

use ps5shader::testing::asm::*;
use ps5shader::testing::{ComputeRun, IoTest};
use ps5shader::{decode_instruction, Opcode};

#[test]
fn named_opcodes_decode() {
    let cases: [(&[u32], Opcode); 13] = [
        (&[sop1(0x22, 0, 4)], Opcode::SRfeB64),
        (&[sopp(0x11, 0)], Opcode::SSendmsghalt),
        (&[sopk(0x16, 4, 2)], Opcode::SCallB64),
        (&[0xF414_0000, 0], Opcode::SScratchLoadDword),
        (&[0xF418_0000, 0], Opcode::SScratchLoadDwordx2),
        (&[0xF41C_0000, 0], Opcode::SScratchLoadDwordx4),
        (&ds(0x18, 0, 0, 0, 0, 0, 0), Opcode::DsGwsSemaReleaseAll),
        (&ds(0x19, 0, 0, 0, 0, 0, 0), Opcode::DsGwsInit),
        (&ds(0x1A, 0, 0, 0, 0, 0, 0), Opcode::DsGwsSemaV),
        (&ds(0x1B, 0, 0, 0, 0, 0, 0), Opcode::DsGwsSemaBr),
        (&ds(0x1C, 0, 0, 0, 0, 0, 0), Opcode::DsGwsSemaP),
        (&ds(0x1D, 0, 0, 0, 0, 0, 0), Opcode::DsGwsBarrier),
        (&ds(0x7E, 6, 1, 4, 0, 0, 0), Opcode::DsCondxchg32RtnB64),
    ];
    for (words, opcode) in cases {
        assert_eq!(decode_instruction(0, words).unwrap().opcode, opcode, "{words:08x?}");
    }
}

#[test]
fn refused_instructions_do_not_translate() {
    let programs: [Vec<u32>; 5] = [
        vec![sop1(0x22, 0, 4), S_ENDPGM],
        vec![sopp(0x11, 0), S_ENDPGM],
        vec![0xF414_0000, 0, S_ENDPGM],
        [ds(0x1D, 0, 0, 0, 0, 0, 0), vec![S_ENDPGM]].concat(),
        [ds(0x19, 0, 0, 0, 0, 0, 0), vec![S_ENDPGM]].concat(),
    ];
    for code in programs {
        let r = ComputeRun::new(code.clone(), 32, 1, vec![0; 8]).compile(true);
        assert!(r.is_err(), "{code:08x?} translated");
    }
}

#[test]
fn condxchg32_rtn_b64_writes_flagged_dwords() {
    const DS_WRITE_B64: u32 = 0x4D;
    const DS_READ_B64: u32 = 0x76;
    const DS_CONDXCHG32_RTN_B64: u32 = 0x7E;
    let mut code = vec![
        vop2(0x1A, 1, inline_u32(3), 0), // v1 = tid * 8
        vop1(OP_V_MOV_B32, 2, 255),
        0x1111_1111,
        vop1(OP_V_MOV_B32, 3, 255),
        0x2222_2222,
        vop1(OP_V_MOV_B32, 4, 255),
        0x8000_0005,
        vop1(OP_V_MOV_B32, 5, inline_u32(7)),
        vop2(0x1A, 10, inline_u32(2), 0), // v10 = tid * 4
    ];
    code.extend(ds(DS_WRITE_B64, 0, 1, 2, 0, 0, 0));
    code.extend(ds(DS_CONDXCHG32_RTN_B64, 6, 1, 4, 0, 0, 0));
    code.extend(ds(DS_READ_B64, 8, 1, 0, 0, 0, 0));
    for (k, reg) in [6u32, 7, 8, 9].into_iter().enumerate() {
        code.extend(mubuf(OP_BUFFER_STORE_DWORD, reg, 10, 1, inline_u32(0), k as u32 * 4, true, false, false));
    }
    code.push(S_ENDPGM);
    let out = IoTest::new(&code, 4, 16).run(&[0; 16]).unwrap();
    for lane in 0..4 {
        assert_eq!(&out[lane * 4..lane * 4 + 4], &[0x1111_1111, 0x2222_2222, 5, 0x2222_2222], "lane {lane}");
    }
}
