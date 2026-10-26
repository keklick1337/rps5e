// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! MIMG raw-texel (PCK), multi-texel (BY2 / BY4, PCK2 / PCK4) and packed
//! horizontal gather forms decode and translate with their flags; the BY /
//! PCK2 / PCK4 stores and image_gather8h_pck are refused (no measured
//! behaviour), as are BY / PCKn forms outside the measured subset.

use ps5shader::ssa::{ImageAttr, ImageOp, Op};
use ps5shader::testing::asm::S_ENDPGM;
use ps5shader::testing::ComputeRun;
use ps5shader::{decode_instruction, Opcode};

/// MIMG word 0 for opcode `op`, DIM `dim` (2D = 1), DMASK `dmask`.
fn mimg0(op: u32, dim: u32, dmask: u32) -> u32 {
    0xF000_0000 | (op & 0x7F) << 18 | op >> 7 | dim << 3 | dmask << 8
}

/// MIMG word 1: VADDR v0, VDATA v8, T# s[0:7], S# s[8:11].
const MIMG1: u32 = 8 << 8 | 2 << 21;

fn compile(words: [u32; 2]) -> Result<ImageAttr, String> {
    let code = vec![words[0], words[1], S_ENDPGM];
    let f = ComputeRun::new(code, 32, 1, vec![0; 12]).compile(false)?;
    Ok(f.values.iter().find_map(|v| if let Op::Image(a) = v.op { Some(a) } else { None }).expect("no image op"))
}

#[test]
fn packed_opcodes_decode() {
    let names = [
        (0x02, Opcode::ImageLoadPck),
        (0x03, Opcode::ImageLoadPckSgn),
        (0x04, Opcode::ImageLoadMipPck),
        (0x05, Opcode::ImageLoadMipPckSgn),
        (0x0A, Opcode::ImageStorePck),
        (0x0B, Opcode::ImageStoreMipPck),
        (0x42, Opcode::ImageLoadBy2),
        (0x43, Opcode::ImageLoadBy4),
        (0x4A, Opcode::ImageLoadMipBy2),
        (0x4B, Opcode::ImageLoadMipBy4),
        (0x52, Opcode::ImageStoreBy2),
        (0x53, Opcode::ImageStoreBy4),
        (0x5A, Opcode::ImageStoreMipBy2),
        (0x5B, Opcode::ImageStoreMipBy4),
        (0x62, Opcode::ImageGather4hPck),
        (0x63, Opcode::ImageGather8hPck),
        (0x70, Opcode::ImageLoadPck2),
        (0x71, Opcode::ImageLoadPck4),
        (0x73, Opcode::ImageLoadMipPck2),
        (0x74, Opcode::ImageLoadMipPck4),
        (0x76, Opcode::ImageStorePck2),
        (0x77, Opcode::ImageStorePck4),
        (0x79, Opcode::ImageStoreMipPck2),
        (0x7A, Opcode::ImageStoreMipPck4),
    ];
    for (op, opcode) in names {
        let ins = decode_instruction(0, &[mimg0(op, 1, 0xF), MIMG1]).unwrap();
        assert_eq!(ins.opcode, opcode, "MIMG {op:#x}");
        assert_eq!(ins.destinations.is_empty(), opcode.is_image_store(), "{opcode:?}");
    }
    assert_eq!(Opcode::ImageGather4hPck.mnemonic(), "image_gather4h_pck");
    assert_eq!(Opcode::ImageLoadMipPckSgn.mnemonic(), "image_load_mip_pck_sgn");
}

#[test]
fn packed_forms_translate_with_their_flags() {
    let a = compile([mimg0(0x03, 1, 0xF), MIMG1]).unwrap();
    assert_eq!(a.op, ImageOp::Load);
    assert!(a.flags.packed && a.flags.packed_signed && !a.flags.mip && a.flags.elements == 0);
    let a = compile([mimg0(0x04, 1, 0xA), MIMG1]).unwrap();
    assert!(a.flags.packed && a.flags.mip && !a.flags.packed_signed);
    assert_eq!(a.address_dwords, 3);
    let a = compile([mimg0(0x0B, 1, 0xB), MIMG1]).unwrap();
    assert_eq!(a.op, ImageOp::Store);
    assert!(a.flags.packed && a.flags.mip);
    assert_eq!(a.data_dwords, 3);
    let a = compile([mimg0(0x4A, 1, 0x3), MIMG1]).unwrap();
    assert!(!a.flags.packed && a.flags.elements == 2 && a.flags.mip);
    let a = compile([mimg0(0x71, 1, 0x1), MIMG1]).unwrap();
    assert!(a.flags.packed && a.flags.elements == 4 && !a.flags.mip);
    // The packed horizontal gather returns its DMASK's dwords.
    let a = compile([mimg0(0x62, 1, 0x5), MIMG1]).unwrap();
    assert_eq!(a.op, ImageOp::Gather4);
    assert!(a.flags.packed && a.flags.horizontal);
}

#[test]
fn unmeasured_forms_are_refused() {
    for op in [0x52, 0x53, 0x5A, 0x5B, 0x76, 0x77, 0x79, 0x7A, 0x63] {
        assert!(compile([mimg0(op, 1, 0xF), MIMG1]).is_err(), "MIMG {op:#x}");
    }
    // BY2 needs DMASK 0x3 or 0xF, BY4 0xF, PCK2 / PCK4 0x1; 2D only.
    assert!(compile([mimg0(0x42, 1, 0x1), MIMG1]).is_err());
    assert!(compile([mimg0(0x43, 1, 0x3), MIMG1]).is_err());
    assert!(compile([mimg0(0x70, 1, 0x3), MIMG1]).is_err());
    assert!(compile([mimg0(0x42, 0, 0x3), MIMG1]).is_err());
    // PCK forms have no 16-bit data.
    assert!(compile([mimg0(0x02, 1, 0x1), MIMG1 | 1 << 31]).is_err());
}
