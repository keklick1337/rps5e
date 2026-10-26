// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Encodings that exist only before RDNA are not decoded: VOP2 0x15, 0x17
//! and 0x19 (v_lshr_b32, v_ashr_i32, v_lshl_b32) and 0x22-0x24 (v_bcnt,
//! v_mbcnt_lo/hi) are gfx6-9 opcodes; gfx10 keeps only the *rev shifts in
//! VOP2 and the bit counts in VOP3.

use ps5shader::decode_instruction;
use ps5shader::opcode::Opcode;

const fn vop2(op: u32, dst: u32, src0: u32, src1: u32) -> u32 {
    (op & 0x3F) << 25 | (dst & 0xFF) << 17 | (src1 & 0xFF) << 9 | (src0 & 0x1FF)
}

#[test]
fn pre_rdna_vop2_opcodes_are_not_decoded() {
    for op in [0x15, 0x17, 0x19, 0x22, 0x23, 0x24] {
        let word = vop2(op, 1, 256 + 2, 3);
        assert!(decode_instruction(0, &[word]).is_err(), "VOP2 {op:#x} decoded");
    }
}

#[test]
fn rdna_shifts_and_bit_counts_still_decode() {
    let shifts = [(0x16, Opcode::VLshrrevB32), (0x18, Opcode::VAshrrevI32), (0x1A, Opcode::VLshlrevB32)];
    for (op, opcode) in shifts {
        let ins = decode_instruction(0, &[vop2(op, 1, 256 + 2, 3)]).unwrap();
        assert_eq!(ins.opcode, opcode);
    }
    // VOP3 v_bcnt_u32_b32 v1, v2, v3 (0x364).
    let ins = decode_instruction(0, &[0xD764_0001, 0x0002_0702]).unwrap();
    assert_eq!(ins.opcode, Opcode::VBcntU32B32);
}

/// MUBUF with the LDS bit (word 0 bit 16) loads into LDS at M0, not into
/// VDATA: it is refused instead of being translated as a VGPR load.
#[test]
fn mubuf_lds_modifier_is_refused() {
    use ps5shader::testing::asm::*;
    use ps5shader::testing::ComputeRun;
    let mut code = mubuf(OP_BUFFER_LOAD_DWORD, 1, 0, 0, inline_u32(0), 0, true, false, false);
    code[0] |= 1 << 16;
    code.push(S_ENDPGM);
    let err = ComputeRun::new(code, 32, 32, vec![0; 4]).compile(true).unwrap_err();
    assert!(err.contains("lds"), "{err}");
}
