// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Opcodes that have no DPP16/DPP8 form in the ISA must be rejected by the
//! compiler (any error), while ordinary DPP forms compile.

use ps5shader::testing::ComputeRun;

const DPP16: u32 = 0xff00_b104;
const DPP8: u32 = 0x00fa_c604;

/// (is VOP2, opcode, name)
const NO_FORM_OPCODES: [(bool, u32, &str); 23] = [
    (false, 0x02, "v_readfirstlane_b32"),
    (false, 0x03, "v_cvt_i32_f64"),
    (false, 0x04, "v_cvt_f64_i32"),
    (false, 0x0f, "v_cvt_f32_f64"),
    (false, 0x10, "v_cvt_f64_f32"),
    (false, 0x15, "v_cvt_u32_f64"),
    (false, 0x16, "v_cvt_f64_u32"),
    (false, 0x17, "v_trunc_f64"),
    (false, 0x18, "v_ceil_f64"),
    (false, 0x19, "v_rndne_f64"),
    (false, 0x1a, "v_floor_f64"),
    (false, 0x3c, "v_frexp_exp_i32_f64"),
    (false, 0x3d, "v_frexp_mant_f64"),
    (false, 0x3e, "v_fract_f64"),
    (false, 0x65, "v_swap_b32"),
    (false, 0x68, "v_swaprel_b32"),
    (true, 0x06, "v_mac_legacy_f32"),
    (true, 0x20, "v_madmk_f32"),
    (true, 0x21, "v_madak_f32"),
    (true, 0x2c, "v_fmamk_f32"),
    (true, 0x2d, "v_fmaak_f32"),
    (true, 0x37, "v_fmamk_f16"),
    (true, 0x38, "v_fmaak_f16"),
];

fn encode(vop2: bool, opcode: u32, form: u32) -> u32 {
    if vop2 {
        (opcode << 25) | (10 << 17) | (5 << 9) | form
    } else {
        0x7e00_0000 | (10 << 17) | (opcode << 9) | form
    }
}

fn compiles(code: &[u32]) -> Result<(), String> {
    ComputeRun::new(code.to_vec(), 32, 32, vec![0; 8]).compile(false).map(|_| ())
}

/// Every opcode must be refused with an error (a panic is not a refusal).
fn check_refused(form: u32, word1: u32, what: &str) {
    let mut wrong = Vec::new();
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    for (vop2, opcode, name) in NO_FORM_OPCODES {
        let code = [encode(vop2, opcode, form), word1, 0x3c00_3c00, 0xbf81_0000];
        match std::panic::catch_unwind(|| compiles(&code)) {
            Ok(Ok(())) => wrong.push(format!("{name} with {what} ({:#010x}) was not refused", code[0])),
            Ok(Err(_)) => {}
            Err(p) => {
                let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                wrong.push(format!("{name} with {what} ({:#010x}) panicked the compiler: {msg}", code[0]));
            }
        }
    }
    std::panic::set_hook(hook);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn dpp16_refused() {
    check_refused(0xfa, DPP16, "DPP16");
}

#[test]
fn dpp8_refused() {
    check_refused(0xe9, DPP8, "DPP8");
}

#[test]
fn dpp8_fi_refused() {
    check_refused(0xea, DPP8, "DPP8 fi:1");
}

/// The DPP16 controls RDNA leaves reserved (the GCN wave shifts and
/// rotates 0x130..0x13F, row broadcasts 0x142/0x143, and the gaps) are
/// refused, not read as the unchanged source.
#[test]
fn reserved_dpp_controls_are_refused() {
    let word = |control: u32| 0xff00_0004 | (control << 8);
    for control in [0x100u32, 0x110, 0x120, 0x130, 0x134, 0x138, 0x13c, 0x142, 0x143, 0x14f, 0x170, 0x1ff] {
        let err = compiles(&[0x7e14_02fa, word(control), 0xbf81_0000]).unwrap_err();
        assert!(err.contains("DPP control"), "{control:#x}: {err}");
    }
    for control in [0x1bu32, 0x101, 0x11f, 0x121, 0x140, 0x141, 0x150, 0x16f] {
        compiles(&[0x7e14_02fa, word(control), 0xbf81_0000]).unwrap_or_else(|e| panic!("{control:#x}: {e}"));
    }
}

#[test]
fn ordinary_dpp_forms_compile() {
    compiles(&[0x7e14_02fa, DPP16, 0xbf81_0000]).unwrap();
    compiles(&[0x4a14_0afa, DPP16, 0xbf81_0000]).unwrap();
}
