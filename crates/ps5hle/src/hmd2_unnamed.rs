// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The libSceHmd2 exports without a known name, as the console's module
//! answers them before sceHmd2Initialize: a fixed code after the pointer
//! and slot checks it makes first, and the fixed values some of them write.
//! After initialization they are not implemented.

use std::sync::atomic::Ordering;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const fn err(code: u32) -> HleResult {
    code as i32 as i64
}

const OK: HleResult = 0;
const NOT_INITIALIZED: HleResult = err(0x8A72_0002);
const INVALID_HANDLE: HleResult = err(0x8A72_0003);
const ERROR_0004: HleResult = err(0x8A72_0004);
const PARAMETER_NULL: HleResult = err(0x8A72_0008);
const PARAMETER_INVALID: HleResult = err(0x8A72_0009);
const REPROJECTION_NOT_INITIALIZED: HleResult = err(0x8A72_000B);
const ERROR_002C: HleResult = err(0x8A72_002C);
const ERROR_1002: HleResult = err(0x8A72_1002);
const ERROR_1003: HleResult = err(0x8A72_1003);
const ERROR_100F: HleResult = err(0x8A72_100F);
const ERROR_1010: HleResult = err(0x8A72_1010);
const ERROR_800D0203: HleResult = err(0x800D_0203);
const NOT_IMPLEMENTED: HleResult = err(0x8002_0002);

/// Four f32 constants (about 0.0149) written by one query.
const FOUR_CONSTANTS: [u8; 16] = [0xe1, 0xf0, 0x74, 0x3c, 0xd8, 0xf0, 0x74, 0x3c, 0x1f, 0xf1, 0x74, 0x3c, 0xd8, 0xf0, 0x74, 0x3c];
/// { 0, 0.5, 0.5 } twice.
const SIX_HALVES: [u8; 24] = [0, 0, 0, 0, 0, 0, 0, 0x3f, 0, 0, 0, 0x3f, 0, 0, 0, 0, 0, 0, 0, 0x3f, 0, 0, 0, 0x3f];
/// { 0, 0.5, 0.5, 0, 0, 0 } twice.
const TWELVE_HALVES: [u8; 48] = [
    0, 0, 0, 0, 0, 0, 0, 0x3f, 0, 0, 0, 0x3f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0x3f, 0, 0, 0, 0x3f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// The answer measured before initialization; after it, not implemented.
fn before_initialize(answer: HleResult) -> HleResult {
    if super::INITIALIZED.load(Ordering::Acquire) {
        log::error!("[hmd2] unnamed export called after sceHmd2Initialize: not implemented");
        return NOT_IMPLEMENTED;
    }
    answer
}

/// Write `bytes` to a required output, then answer `answer`.
fn write_then(ctx: &CpuContext, out: u64, bytes: &[u8], answer: HleResult) -> HleResult {
    if out == 0 || !ctx.write_bytes(out, bytes) {
        return PARAMETER_NULL;
    }
    before_initialize(answer)
}

fn cb_before_initialize_error0004(_ctx: &mut CpuContext) -> HleResult {
    before_initialize(ERROR_0004)
}

fn cb_before_initialize_error002c(_ctx: &mut CpuContext) -> HleResult {
    before_initialize(ERROR_002C)
}

fn cb_before_initialize_error1002(_ctx: &mut CpuContext) -> HleResult {
    before_initialize(ERROR_1002)
}

fn cb_before_initialize_error1003(_ctx: &mut CpuContext) -> HleResult {
    before_initialize(ERROR_1003)
}

fn cb_before_initialize_minus_one(_ctx: &mut CpuContext) -> HleResult {
    before_initialize(-1)
}

fn cb_before_initialize_not_initialized(_ctx: &mut CpuContext) -> HleResult {
    before_initialize(NOT_INITIALIZED)
}

fn cb_before_initialize_parameter_invalid(_ctx: &mut CpuContext) -> HleResult {
    before_initialize(PARAMETER_INVALID)
}

fn cb_before_initialize_parameter_null(_ctx: &mut CpuContext) -> HleResult {
    before_initialize(PARAMETER_NULL)
}

fn cb_before_initialize_reprojection_not_initialized(_ctx: &mut CpuContext) -> HleResult {
    before_initialize(REPROJECTION_NOT_INITIALIZED)
}

fn cb_before_initialize_value0400(_ctx: &mut CpuContext) -> HleResult {
    before_initialize(0x400)
}

fn cb_before_initialize_value20000(_ctx: &mut CpuContext) -> HleResult {
    before_initialize(0x2_0000)
}

fn cb_before_initialize_zero(_ctx: &mut CpuContext) -> HleResult {
    before_initialize(OK)
}

fn cb_check_all_six_then_zero(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 || ctx.rsi() == 0 || ctx.rdx() == 0 || ctx.rcx() == 0 || ctx.r8() == 0 || ctx.r9() == 0 {
        return PARAMETER_NULL;
    }
    before_initialize(OK)
}

fn cb_check_first_and_third_then_error1003(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 || ctx.rdx() == 0 {
        return PARAMETER_NULL;
    }
    before_initialize(ERROR_1003)
}

fn cb_check_first_and_third_then_zero(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 || ctx.rdx() == 0 {
        return PARAMETER_NULL;
    }
    before_initialize(OK)
}

fn cb_check_first_then_error0004(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return PARAMETER_NULL;
    }
    before_initialize(ERROR_0004)
}

fn cb_check_first_then_error1002(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return PARAMETER_NULL;
    }
    before_initialize(ERROR_1002)
}

fn cb_check_first_then_error1003(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return PARAMETER_NULL;
    }
    before_initialize(ERROR_1003)
}

fn cb_check_first_then_error1010(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return PARAMETER_NULL;
    }
    before_initialize(ERROR_1010)
}

fn cb_check_first_then_error800d0203(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return PARAMETER_NULL;
    }
    before_initialize(ERROR_800D0203)
}

fn cb_check_first_then_minus_one(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return PARAMETER_NULL;
    }
    before_initialize(-1)
}

fn cb_check_first_then_parameter_invalid(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return PARAMETER_NULL;
    }
    before_initialize(PARAMETER_INVALID)
}

fn cb_check_first_then_zero(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return PARAMETER_NULL;
    }
    before_initialize(OK)
}

fn cb_check_first_two_then_error1002(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 || ctx.rsi() == 0 {
        return PARAMETER_NULL;
    }
    before_initialize(ERROR_1002)
}

fn cb_check_first_two_then_error1003(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 || ctx.rsi() == 0 {
        return PARAMETER_NULL;
    }
    before_initialize(ERROR_1003)
}

fn cb_check_first_two_then_parameter_invalid(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 || ctx.rsi() == 0 {
        return PARAMETER_NULL;
    }
    before_initialize(PARAMETER_INVALID)
}

fn cb_check_second_then_error002c(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 {
        return PARAMETER_NULL;
    }
    before_initialize(ERROR_002C)
}

fn cb_check_second_then_error1002(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 {
        return PARAMETER_NULL;
    }
    before_initialize(ERROR_1002)
}

fn cb_check_second_then_invalid_handle(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 {
        return PARAMETER_NULL;
    }
    before_initialize(INVALID_HANDLE)
}

fn cb_check_second_then_parameter_invalid(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 {
        return PARAMETER_NULL;
    }
    before_initialize(PARAMETER_INVALID)
}

fn cb_index_zero_check_second_then_zero(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 {
        return PARAMETER_NULL;
    }
    if ctx.rdi() != 0 {
        return ERROR_100F;
    }
    before_initialize(OK)
}

fn cb_slot_up_to1_check_second_then_error1002(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 {
        return PARAMETER_NULL;
    }
    if ctx.rdi() > 1 {
        return PARAMETER_INVALID;
    }
    before_initialize(ERROR_1002)
}

fn cb_slot_up_to1_then_error1002(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() > 1 {
        return PARAMETER_INVALID;
    }
    before_initialize(ERROR_1002)
}

fn cb_slot_up_to1_then_minus_one(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() > 1 {
        return PARAMETER_INVALID;
    }
    before_initialize(-1)
}

fn cb_slot_up_to2_check_second_then_error002c(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 {
        return PARAMETER_NULL;
    }
    if ctx.rdi() > 2 {
        return PARAMETER_INVALID;
    }
    before_initialize(ERROR_002C)
}

fn cb_slot_up_to2_then_error1002(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() > 2 {
        return PARAMETER_INVALID;
    }
    before_initialize(ERROR_1002)
}

fn cb_slot_up_to2_then_error1003(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() > 2 {
        return PARAMETER_INVALID;
    }
    before_initialize(ERROR_1003)
}

fn cb_slot_up_to2_then_minus_one(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() > 2 {
        return PARAMETER_INVALID;
    }
    before_initialize(-1)
}

fn cb_slot_up_to3_check_second_third_then_zero(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() == 0 || ctx.rdx() == 0 {
        return PARAMETER_NULL;
    }
    if ctx.rdi() > 3 {
        return PARAMETER_INVALID;
    }
    before_initialize(OK)
}

fn cb_write_four_constants(ctx: &mut CpuContext) -> HleResult {
    write_then(ctx, ctx.rdi(), &FOUR_CONSTANTS, OK)
}

fn cb_write_six_halves_then_invalid_handle(ctx: &mut CpuContext) -> HleResult {
    write_then(ctx, ctx.rsi(), &SIX_HALVES, INVALID_HANDLE)
}

fn cb_write_size_x2000_align_x100(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 || ctx.rsi() == 0 {
        return PARAMETER_NULL;
    }
    if !(ctx.write_u64(ctx.rdi(), 0x2000) && ctx.write_u64(ctx.rsi(), 0x100)) {
        return PARAMETER_NULL;
    }
    before_initialize(OK)
}

fn cb_write_state_two(ctx: &mut CpuContext) -> HleResult {
    write_then(ctx, ctx.rdi(), &[0x02, 0, 0, 0, 0], OK)
}

fn cb_write_twelve_halves_then_invalid_handle(ctx: &mut CpuContext) -> HleResult {
    write_then(ctx, ctx.rsi(), &TWELVE_HALVES, INVALID_HANDLE)
}

const UNNAMED: &[(&str, HleCallback)] = &[
    ("vDgpdS2N9Rw", cb_before_initialize_zero),
    ("Ri35gKTZpvE", cb_before_initialize_zero),
    ("9oyW8puiKeI", cb_before_initialize_zero),
    ("1HZNE6jhBnQ", cb_before_initialize_zero),
    ("cdwAKBZLXjs", cb_before_initialize_zero),
    ("Yub1MF58jfY", cb_before_initialize_zero),
    ("JYXAuQFnPyw", cb_before_initialize_zero),
    ("VuPKqCu3wzw", cb_before_initialize_zero),
    ("GSThagVyW5U", cb_before_initialize_zero),
    ("J4qE5N937MY", cb_before_initialize_zero),
    ("9KnNj-dyASg", cb_before_initialize_zero),
    ("Qta2bkstbFk", cb_before_initialize_zero),
    ("5ZcObnDRBXE", cb_before_initialize_zero),
    ("AvPNn0qx4WQ", cb_before_initialize_zero),
    ("jp0apqNGgi8", cb_before_initialize_zero),
    ("-kZT7X7OMg0", cb_before_initialize_zero),
    ("1ewLtng-SVk", cb_before_initialize_zero),
    ("B0b8DeIUGes", cb_before_initialize_zero),
    ("HclLg4Oj5yw", cb_before_initialize_zero),
    ("VbEvrQXsZyY", cb_before_initialize_zero),
    ("bB6MKZi5UCQ", cb_before_initialize_zero),
    ("OXQ1EDf02XU", cb_before_initialize_zero),
    ("IAxfMTvFm0M", cb_before_initialize_zero),
    ("qTJtAtaLeno", cb_before_initialize_zero),
    ("j7Q16UX6l34", cb_before_initialize_zero),
    ("pmmNM06rLqU", cb_before_initialize_zero),
    ("zdQfyDK9Ods", cb_before_initialize_zero),
    ("aPRwp9GnljM", cb_before_initialize_zero),
    ("MjY1gBSaed0", cb_before_initialize_zero),
    ("evAMjJYHqW4", cb_before_initialize_zero),
    ("r93FLEZ3BBk", cb_before_initialize_zero),
    ("IdYURdEdNFo", cb_before_initialize_zero),
    ("CVcL2Cu2AEI", cb_before_initialize_zero),
    ("NmKIfVxr2s0", cb_before_initialize_zero),
    ("KeDLkkp62Gg", cb_before_initialize_zero),
    ("+aadu6-ra+A", cb_before_initialize_zero),
    ("FgnTiwwGSzk", cb_before_initialize_zero),
    ("PWAUwOTHbbo", cb_before_initialize_zero),
    ("WzaEApHnsP8", cb_before_initialize_zero),
    ("dXzQI8QVY7Y", cb_before_initialize_zero),
    ("cCvFKGqRh1k", cb_before_initialize_zero),
    ("uj4-2VEAFPU", cb_before_initialize_zero),
    ("sIaGStI-iLM", cb_before_initialize_zero),
    ("JzwpZOemDO8", cb_before_initialize_zero),
    ("IsC4R4bZFVU", cb_before_initialize_zero),
    ("sXaPEPY-fLA", cb_before_initialize_not_initialized),
    ("3F5inHhKzXg", cb_before_initialize_not_initialized),
    ("sJAiB-6p55U", cb_before_initialize_not_initialized),
    ("E8hDydLjsSs", cb_before_initialize_not_initialized),
    ("xEQkSb2WJ7Y", cb_before_initialize_not_initialized),
    ("aaP4dnR4q-0", cb_before_initialize_not_initialized),
    ("lEU0TeUoGwI", cb_before_initialize_not_initialized),
    ("cFVyjHBpUUU", cb_before_initialize_not_initialized),
    ("zROtb46jTj4", cb_before_initialize_not_initialized),
    ("kAMsjYHBbNQ", cb_before_initialize_error0004),
    ("5m73zmLpsQ0", cb_before_initialize_error0004),
    ("EPeH4xuQ2XQ", cb_before_initialize_error0004),
    ("TJ8lyRsjFY4", cb_before_initialize_error0004),
    ("-XQVcLnyX68", cb_before_initialize_error0004),
    ("giQwQoHhYYA", cb_before_initialize_error0004),
    ("zNc9IkgawMI", cb_before_initialize_error0004),
    ("-tnG5UJPvAc", cb_before_initialize_error0004),
    ("FPhFmG+jXuQ", cb_before_initialize_error0004),
    ("CKlkvNngofQ", cb_before_initialize_error0004),
    ("yL1CZaciFWE", cb_before_initialize_error0004),
    ("qlzgyG3s0RY", cb_before_initialize_error0004),
    ("kTIDrDLzMv0", cb_before_initialize_error0004),
    ("9Gee0ef8a+U", cb_before_initialize_error0004),
    ("ecZVjQdwmBI", cb_before_initialize_error0004),
    ("MNtiJTDOPAU", cb_before_initialize_parameter_null),
    ("RRy9WmqZfgU", cb_before_initialize_parameter_null),
    ("uobR4m9lgwc", cb_before_initialize_parameter_null),
    ("-XPk5KWBYdo", cb_before_initialize_parameter_null),
    ("J44idh9izZM", cb_before_initialize_parameter_null),
    ("Ps0wAvYaa1E", cb_before_initialize_parameter_null),
    ("kLnMR8oQ1MY", cb_before_initialize_parameter_null),
    ("kMz4rbBxvjo", cb_before_initialize_parameter_null),
    ("DIU1sFRRaiE", cb_before_initialize_parameter_null),
    ("ZQvU4vDzZGQ", cb_before_initialize_parameter_null),
    ("TJpyGGP9YEY", cb_before_initialize_parameter_null),
    ("Dih+Odk+RC0", cb_before_initialize_parameter_null),
    ("oaqdSJw+HVY", cb_before_initialize_parameter_null),
    ("m35gtHuanXU", cb_before_initialize_parameter_null),
    ("CTM7QPLJF9Q", cb_before_initialize_parameter_null),
    ("+rExHeC9lP0", cb_before_initialize_parameter_null),
    ("bBh5bCp9coo", cb_before_initialize_parameter_invalid),
    ("G3PNDE2VjfY", cb_before_initialize_parameter_invalid),
    ("nScCT0OPU0o", cb_before_initialize_parameter_invalid),
    ("6Q0Ng7OTN9o", cb_before_initialize_parameter_invalid),
    ("-fJKh4SnFR0", cb_before_initialize_parameter_invalid),
    ("jUA2s-l0npY", cb_before_initialize_parameter_invalid),
    ("+9XhyAIJVj4", cb_before_initialize_parameter_invalid),
    ("eBGK-o6E4RQ", cb_before_initialize_parameter_invalid),
    ("MTuFeL9184E", cb_before_initialize_parameter_invalid),
    ("d3SEIYTP6wc", cb_before_initialize_reprojection_not_initialized),
    ("EVECbTvBezo", cb_before_initialize_reprojection_not_initialized),
    ("ciusPdJKjSU", cb_before_initialize_reprojection_not_initialized),
    ("DIwqTkcZiJo", cb_before_initialize_reprojection_not_initialized),
    ("64ERZWVFBTE", cb_before_initialize_reprojection_not_initialized),
    ("z0+0gXUJWs8", cb_before_initialize_reprojection_not_initialized),
    ("rjAd4nCY4DM", cb_before_initialize_reprojection_not_initialized),
    ("Acyg4vTdGm0", cb_before_initialize_reprojection_not_initialized),
    ("aGOWaEeg04I", cb_before_initialize_reprojection_not_initialized),
    ("9up+EPhw1No", cb_before_initialize_reprojection_not_initialized),
    ("V1kLQmP-17U", cb_before_initialize_error002c),
    ("2LVD1EC5SYE", cb_before_initialize_error1002),
    ("Bf87AL6zWxU", cb_before_initialize_error1002),
    ("XKt7w1MODFY", cb_before_initialize_error1002),
    ("Y5s22M-qUDA", cb_before_initialize_error1002),
    ("JJKY-BtQJLg", cb_before_initialize_error1002),
    ("9kj9vw8UShQ", cb_before_initialize_error1003),
    ("EAkv8Mqt33E", cb_before_initialize_minus_one),
    ("Rv7bkmKvmz0", cb_before_initialize_minus_one),
    ("1oPZizn0SNA", cb_before_initialize_minus_one),
    ("TwCpE-nNrlo", cb_before_initialize_minus_one),
    ("jkqAeQfdtxw", cb_before_initialize_minus_one),
    ("+l0RQph81Z8", cb_before_initialize_minus_one),
    ("qsmAtGw1Ncc", cb_before_initialize_minus_one),
    ("uKhwYoMsIsA", cb_before_initialize_minus_one),
    ("WNdaigr94dI", cb_before_initialize_minus_one),
    ("P6S5ZVWf-H0", cb_before_initialize_minus_one),
    ("sun6NWE1sdg", cb_before_initialize_minus_one),
    ("--TFsJYXo04", cb_before_initialize_minus_one),
    ("BWqcwjCozzI", cb_before_initialize_value0400),
    ("PRz7TdBGUI0", cb_before_initialize_value20000),
    ("9WeZr4kYfzk", cb_check_first_then_error0004),
    ("N+5h1hCOFAg", cb_check_first_then_error0004),
    ("w2vEs2y5ZmE", cb_check_first_then_error0004),
    ("L5gat1pe0y0", cb_check_first_then_error0004),
    ("4GnLHR9NJ1w", cb_check_first_then_error0004),
    ("AHk7nGSzef0", cb_check_first_then_error0004),
    ("ncjX9d3mc0M", cb_check_first_then_error0004),
    ("l63QYAUuogc", cb_check_first_then_error0004),
    ("mrcBySCdXSo", cb_check_first_then_error0004),
    ("ieBra+td8sU", cb_check_first_then_error0004),
    ("s+rzBnewgGs", cb_check_first_then_error0004),
    ("Nfcm0GBdzZI", cb_check_first_then_error0004),
    ("Tc9rasWfrm4", cb_check_first_then_error0004),
    ("ckTbKTX4KZg", cb_check_first_then_error0004),
    ("D7576PUbUjk", cb_check_first_then_error0004),
    ("tTJ04SrBT8U", cb_check_first_then_error0004),
    ("sDP7hZ-P6OI", cb_check_first_then_error0004),
    ("J+eVNbN9ndY", cb_check_first_then_error0004),
    ("kozvHBNyxf0", cb_check_first_then_error0004),
    ("638Tmg5y6U4", cb_check_first_then_error0004),
    ("8ETtFtp5zqY", cb_check_first_then_zero),
    ("CWPkBm3EXQA", cb_check_first_then_zero),
    ("R62liQQIAaA", cb_check_first_then_zero),
    ("-bcWLHsdzo0", cb_check_first_then_zero),
    ("JngJzgNbu8U", cb_check_first_then_zero),
    ("xNBSz8xrGdk", cb_check_first_then_zero),
    ("yUVPfN-jRIU", cb_check_first_then_zero),
    ("BKPR3RZ5aO4", cb_check_first_then_zero),
    ("IG4y+drWAWw", cb_check_first_then_error1002),
    ("rhYUml9vppE", cb_check_first_then_error1002),
    ("6574xHS+jZk", cb_check_first_then_error1002),
    ("mXzaY6xVoxg", cb_check_first_then_error1002),
    ("6af-HIZQQn4", cb_check_first_then_error1002),
    ("6qWtdTHQsTk", cb_check_first_then_error1002),
    ("pUm8t8xWhFg", cb_check_first_then_error1002),
    ("A0XKI-WjxK8", cb_check_first_then_error1002),
    ("A9s2j8OioPQ", cb_check_first_then_minus_one),
    ("YIgtmrL+HLY", cb_check_first_then_minus_one),
    ("Ng8zEkGxDNs", cb_check_first_then_minus_one),
    ("EtslTnqO1xU", cb_check_first_then_minus_one),
    ("hfrnRTryKQQ", cb_check_first_then_minus_one),
    ("PPX68fNToFM", cb_check_first_two_then_parameter_invalid),
    ("LxPIfjHaY4c", cb_check_first_two_then_parameter_invalid),
    ("w8gBTTdg+po", cb_check_first_two_then_parameter_invalid),
    ("tWJPX2ycRMM", cb_check_second_then_error002c),
    ("5UMB9PJA70o", cb_check_second_then_error002c),
    ("xQs62qCk1EI", cb_check_second_then_error002c),
    ("MMc+bSlqkzk", cb_check_first_then_parameter_invalid),
    ("NpZHgvSYgEA", cb_check_first_then_parameter_invalid),
    ("z-serf9vsb0", cb_check_all_six_then_zero),
    ("IRMUFu+uG5s", cb_check_all_six_then_zero),
    ("72U64E+0z6U", cb_check_second_then_parameter_invalid),
    ("LCQzi5ILGrY", cb_check_second_then_parameter_invalid),
    ("Im0qL0j9Ubg", cb_check_first_then_error800d0203),
    ("ng6eI8uA0h4", cb_check_first_then_error1003),
    ("mPPGgjyJrCI", cb_check_first_then_error1010),
    ("ZupCJmV5BDI", cb_check_first_two_then_error1002),
    ("mAQZwOX1IHU", cb_check_first_two_then_error1003),
    ("oacDkQkinLo", cb_check_first_and_third_then_zero),
    ("zn7PHGzzEyI", cb_check_first_and_third_then_error1003),
    ("wEHILc7TP04", cb_check_second_then_invalid_handle),
    ("W-ZIlUXcrNA", cb_check_second_then_error1002),
    ("omimgG5itSs", cb_write_state_two),
    ("6PeHcAaKYg8", cb_write_size_x2000_align_x100),
    ("yur+xPdWPAg", cb_write_four_constants),
    ("RnieDDp5ge4", cb_write_six_halves_then_invalid_handle),
    ("jM3NEJA11yw", cb_write_six_halves_then_invalid_handle),
    ("b0H-n-tUk0U", cb_write_twelve_halves_then_invalid_handle),
    ("MXhloJ6Iw3w", cb_slot_up_to3_check_second_third_then_zero),
    ("19wrIIiqFgM", cb_slot_up_to1_then_error1002),
    ("Z389vAm47Ys", cb_slot_up_to1_check_second_then_error1002),
    ("cDcrvEG8j8k", cb_slot_up_to1_then_minus_one),
    ("tuqJO6O6cb4", cb_slot_up_to1_then_minus_one),
    ("fd4fu1cziaI", cb_slot_up_to1_then_minus_one),
    ("UxMg7iL4nRs", cb_slot_up_to2_then_minus_one),
    ("K1GELwlaaGc", cb_slot_up_to2_then_minus_one),
    ("RFn39VCHjGY", cb_slot_up_to2_then_error1002),
    ("FRoSjceDv10", cb_slot_up_to2_then_error1003),
    ("wQ44gBwBw44", cb_slot_up_to2_check_second_then_error002c),
    ("wiv8+7ogrq0", cb_index_zero_check_second_then_zero),
];

pub fn register(reg: &mut HleRegistry, lib: &'static str) {
    for &(nid, cb) in UNNAMED {
        reg.register_fn(nid, "?", lib, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    #[test]
    fn table_is_complete() {
        assert_eq!(UNNAMED.len(), 205);
        let mut nids: Vec<_> = UNNAMED.iter().map(|e| e.0).collect();
        nids.sort();
        nids.dedup();
        assert_eq!(nids.len(), UNNAMED.len());
    }

    /// Reference GuestHmd2 cases (the console's module before
    /// initialization). The answers that depend on the initialization
    /// state are checked only while no other test has the module
    /// initialized.
    #[test]
    fn answers_before_initialization() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D5B_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let call = |ctx: &mut CpuContext, cb: HleCallback, a: &[u64]| {
            for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].into_iter().zip(a.iter().chain(std::iter::repeat(&0))) {
                ctx.set(r, *v);
            }
            cb(ctx)
        };
        // Checks made before the initialization state is looked at.
        assert_eq!(call(&mut ctx, cb_check_first_then_error0004, &[0]), PARAMETER_NULL);
        assert_eq!(call(&mut ctx, cb_check_second_then_error002c, &[BASE, 0]), PARAMETER_NULL);
        assert_eq!(call(&mut ctx, cb_check_all_six_then_zero, &[BASE, BASE, BASE, BASE, BASE, 0]), PARAMETER_NULL);
        assert_eq!(call(&mut ctx, cb_check_first_and_third_then_zero, &[BASE, 0, 0]), PARAMETER_NULL);
        assert_eq!(call(&mut ctx, cb_slot_up_to1_then_error1002, &[2]), PARAMETER_INVALID);
        assert_eq!(call(&mut ctx, cb_slot_up_to2_check_second_then_error002c, &[9, 0]), PARAMETER_NULL);
        assert_eq!(call(&mut ctx, cb_slot_up_to3_check_second_third_then_zero, &[4, BASE, BASE]), PARAMETER_INVALID);
        assert_eq!(call(&mut ctx, cb_index_zero_check_second_then_zero, &[1, BASE]), ERROR_100F);
        assert_eq!(call(&mut ctx, cb_write_size_x2000_align_x100, &[BASE, 0]), PARAMETER_NULL);
        if super::super::INITIALIZED.load(Ordering::Acquire) {
            return;
        }
        assert_eq!(call(&mut ctx, cb_before_initialize_not_initialized, &[]), NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, cb_before_initialize_minus_one, &[]), -1);
        assert_eq!(call(&mut ctx, cb_before_initialize_value20000, &[]), 0x2_0000);
        assert_eq!(call(&mut ctx, cb_before_initialize_reprojection_not_initialized, &[]), REPROJECTION_NOT_INITIALIZED);
        assert_eq!(call(&mut ctx, cb_check_first_then_error800d0203, &[BASE]), ERROR_800D0203);
        assert_eq!(call(&mut ctx, cb_check_first_then_error1010, &[BASE]), ERROR_1010);
        assert_eq!(call(&mut ctx, cb_check_first_two_then_error1003, &[BASE, BASE]), ERROR_1003);
        assert_eq!(call(&mut ctx, cb_check_second_then_invalid_handle, &[0, BASE]), INVALID_HANDLE);
        assert_eq!(call(&mut ctx, cb_check_first_then_error0004, &[BASE]), ERROR_0004);
        assert_eq!(call(&mut ctx, cb_write_size_x2000_align_x100, &[BASE, BASE + 8]), OK);
        assert_eq!(ctx.read_u64(BASE), Some(0x2000));
        assert_eq!(ctx.read_u64(BASE + 8), Some(0x100));
        assert_eq!(call(&mut ctx, cb_write_state_two, &[BASE + 0x10]), OK);
        assert_eq!(ctx.read_bytes(BASE + 0x10, 5).unwrap(), vec![2, 0, 0, 0, 0]);
        assert_eq!(call(&mut ctx, cb_write_twelve_halves_then_invalid_handle, &[7, BASE + 0x100]), INVALID_HANDLE);
        assert_eq!(ctx.read_u32(BASE + 0x104), Some(0x3f00_0000));
        assert_eq!(ctx.read_u32(BASE + 0x11C), Some(0x3f00_0000));
        assert_eq!(call(&mut ctx, cb_write_four_constants, &[BASE + 0x200]), OK);
        assert_eq!(ctx.read_u32(BASE + 0x200), Some(0x3c74_f0e1));
        assert_eq!(call(&mut ctx, cb_slot_up_to1_then_minus_one, &[1]), -1);
        assert_eq!(call(&mut ctx, cb_index_zero_check_second_then_zero, &[0, BASE]), OK);
    }
}
