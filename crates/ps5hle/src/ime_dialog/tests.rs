// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only
//
// Unit tests of the `ps5hle::ime_dialog` module.

use super::*;
use crate::ime_common::{test_support, utf16, Keycode};
use std::sync::atomic::AtomicI32;

const BASE: u64 = 0x5D30_0000_0000;
const PARAM: u64 = BASE;
const EXT: u64 = BASE + 0x100;
const RESULT: u64 = BASE + 0x200;
const WH: u64 = BASE + 0x300;
const FORM: u64 = BASE + 0x400;
const TEXT: u64 = BASE + 0x1000;

const COPY_FILTER: u64 = BASE + 0xF001;
const SUPPLEMENTARY_FILTER: u64 = BASE + 0xF002;
const KEYBOARD_FILTER: u64 = BASE + 0xF003;

static FILTER_CALLS: AtomicI32 = AtomicI32::new(0);
static KEYBOARD_FILTER_CALLS: AtomicI32 = AtomicI32::new(0);
static BLOCK_KEYBOARD: AtomicI32 = AtomicI32::new(0);
static REMAP_KEYBOARD: AtomicI32 = AtomicI32::new(0);
static LOCK: Mutex<()> = Mutex::new(());

fn copy_filter(ctx: &CpuContext, a: [u64; 6]) -> u64 {
    FILTER_CALLS.fetch_add(1, Ordering::SeqCst);
    let (out, out_len, src, src_len) = (a[0], a[1], a[2], a[3]);
    assert!(ctx.read_u32(out_len).unwrap() as u64 >= src_len);
    let bytes = ctx.read_bytes(src, src_len as usize * 2).unwrap();
    ctx.write_bytes(out, &bytes);
    ctx.write_u32(out_len, src_len as u32);
    0
}

fn supplementary_filter(ctx: &CpuContext, a: [u64; 6]) -> u64 {
    assert!(ctx.read_u32(a[1]).unwrap() >= 2);
    ctx.write_bytes(a[0], &[0x3D, 0xD8, 0x00, 0xDE]);
    ctx.write_u32(a[1], 2);
    0
}

fn keyboard_filter(ctx: &CpuContext, a: [u64; 6]) -> u64 {
    KEYBOARD_FILTER_CALLS.fetch_add(1, Ordering::SeqCst);
    let (src, out_keycode, out_status) = (a[0], a[1], a[2]);
    if BLOCK_KEYBOARD.load(Ordering::SeqCst) != 0 {
        ctx.write_bytes(out_keycode, &[0, 0]);
        ctx.write_u32(out_status, 0);
        return 0;
    }
    let remap = REMAP_KEYBOARD.load(Ordering::SeqCst) as u16;
    if remap != 0 {
        ctx.write_bytes(out_keycode, &remap.to_le_bytes());
        ctx.write_u32(out_status, ctx.read_u32(src + 4).unwrap());
        return 0;
    }
    (-1i64) as u64
}

fn noop_visibility(_: bool, _: u64) {}

struct Env {
    vm: ps5vm::VirtualMemory,
}

impl Env {
    fn new() -> Self {
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        test_support::register_fake(COPY_FILTER, copy_filter);
        test_support::register_fake(SUPPLEMENTARY_FILTER, supplementary_filter);
        test_support::register_fake(KEYBOARD_FILTER, keyboard_filter);
        set_visibility_callback(Some(noop_visibility));
        Self { vm }
    }

    fn ctx(&self) -> CpuContext<'_> {
        CpuContext::new(&self.vm)
    }

    fn set_text(&self, text: &str) {
        let ctx = self.ctx();
        ctx.write_bytes(TEXT, &[0u8; 0x200]);
        common::write_units_nul(&ctx, TEXT, &utf16(text), 64);
    }

    fn text(&self) -> Vec<u16> {
        common::scan_units(&self.ctx(), TEXT, 64).unwrap()
    }

    fn unit(&self, i: u64) -> u16 {
        let b = self.ctx().read_bytes(TEXT + i * 2, 2).unwrap();
        u16::from_le_bytes([b[0], b[1]])
    }
}

/// Raw SceImeDialogParam mirroring the guest layout.
#[derive(Clone, Copy)]
struct P {
    user_id: i32,
    ty: u32,
    languages: u64,
    filter: u64,
    option: u32,
    max_text_length: u32,
    reserved0: u8,
}

fn make_param() -> P {
    P { user_id: 1000, ty: TYPE_DEFAULT, languages: 1, filter: 0, option: 0, max_text_length: 31, reserved0: 0 }
}

fn write_param(env: &Env, p: P) {
    let mut b = [0u8; PARAM_SIZE];
    b[0..4].copy_from_slice(&p.user_id.to_le_bytes());
    b[4..8].copy_from_slice(&p.ty.to_le_bytes());
    b[8..16].copy_from_slice(&p.languages.to_le_bytes());
    b[0x18..0x20].copy_from_slice(&p.filter.to_le_bytes());
    b[0x20..0x24].copy_from_slice(&p.option.to_le_bytes());
    b[0x24..0x28].copy_from_slice(&p.max_text_length.to_le_bytes());
    b[0x28..0x30].copy_from_slice(&TEXT.to_le_bytes());
    b[0x50] = p.reserved0;
    env.ctx().write_bytes(PARAM, &b);
}

fn write_ext(env: &Env, option: u32, filter: u64, disable_device: u32, ext_keyboard_mode: u32) {
    let mut b = [0u8; common::EXTENDED_PARAM_SIZE];
    b[0..4].copy_from_slice(&option.to_le_bytes());
    b[0x38..0x40].copy_from_slice(&filter.to_le_bytes());
    b[0x40..0x44].copy_from_slice(&disable_device.to_le_bytes());
    b[0x44..0x48].copy_from_slice(&ext_keyboard_mode.to_le_bytes());
    env.ctx().write_bytes(EXT, &b);
}

fn panel(env: &Env, p: P, ext: bool) -> (i32, u32, u32) {
    write_param(env, p);
    let ctx = env.ctx();
    let rc = get_panel_size_extended(&ctx, PARAM, if ext { EXT } else { 0 }, WH, WH + 4);
    (rc, ctx.read_u32(WH).unwrap(), ctx.read_u32(WH + 4).unwrap())
}

fn result_endstatus(env: &Env) -> (i32, u32) {
    let ctx = env.ctx();
    ctx.write_bytes(RESULT, &[0u8; RESULT_SIZE]);
    ctx.write_u32(RESULT, 0xFFFF_FFFF);
    let rc = get_result(&ctx, RESULT);
    (rc, ctx.read_u32(RESULT).unwrap())
}

fn snapshot() -> HostSnapshot {
    host_snapshot().expect("dialog running")
}

fn test_layout_and_panel_size(env: &Env) {
    let mut p = make_param();
    assert_eq!(panel(env, p, false), (0, 793, 528));
    p.option = OPTION_MULTILINE;
    assert_eq!(panel(env, p, false), (0, 793, 628));
    p.option = 0;
    p.ty = TYPE_NUMBER;
    assert_eq!(panel(env, p, false), (0, 370, 522));

    let mut p = make_param();
    p.option = OPTION_EXT_KEYBOARD;
    write_ext(env, 0x400, 0, 0, 0);
    assert_eq!(panel(env, p, true), (0, 793, 168));
    p.ty = TYPE_BASIC_LATIN;
    assert_eq!(panel(env, p, true), (0, 793, 103));
}

fn test_accept_lifecycle(env: &Env) {
    let ctx = env.ctx();
    env.set_text("Alice");
    write_param(env, make_param());
    assert_eq!(init(&ctx, PARAM, 0), 0);
    assert_eq!(get_status(&ctx), STATUS_RUNNING as i32);
    assert_eq!(init(&ctx, 0, 0), ERROR_BUSY);

    let s = snapshot();
    assert_eq!(s.text, utf16("Alice"));
    assert!(host_insert_text(s.generation, &utf16(" 2")));
    assert!(host_backspace(s.generation));
    assert!(host_accept(s.generation));
    assert_eq!(get_status(&ctx), STATUS_FINISHED as i32);
    assert_eq!(env.text(), utf16("Alice "));

    assert_eq!(result_endstatus(env), (0, END_STATUS_OK));
    assert_eq!(term(&ctx), 0);
    assert_eq!(get_status(&ctx), STATUS_NONE as i32);
}

fn test_cancel_and_abort_restore_text(env: &Env) {
    let ctx = env.ctx();
    for expected in [END_STATUS_USER_CANCELED, END_STATUS_ABORTED] {
        env.set_text("old");
        write_param(env, make_param());
        assert_eq!(init(&ctx, PARAM, 0), 0);
        let s = snapshot();
        assert!(host_insert_text(s.generation, &utf16(" value")));
        if expected == END_STATUS_USER_CANCELED {
            assert!(host_cancel(s.generation));
        } else {
            assert_eq!(abort(&ctx), 0);
        }
        assert_eq!(result_endstatus(env), (0, expected));
        assert_eq!(env.text(), utf16("old"));
        assert_eq!(term(&ctx), 0);
    }
}

fn test_validation(env: &Env) {
    let ctx = env.ctx();
    env.set_text("");
    let mut p = make_param();
    p.reserved0 = 1;
    write_param(env, p);
    assert_eq!(init(&ctx, PARAM, 0), ERROR_INVALID_RESERVED);
    p.reserved0 = 0;
    p.max_text_length = MAX_TEXT_LENGTH + 1;
    write_param(env, p);
    assert_eq!(init(&ctx, PARAM, 0), ERROR_INVALID_MAX_TEXT_LENGTH);
    assert_eq!(get_result(&ctx, 0), ERROR_NOT_IN_USE);

    write_param(env, make_param());
    write_ext(env, 0, 0, 0, 0x4);
    assert_eq!(init(&ctx, PARAM, EXT), ERROR_INVALID_EXTENDED);
    write_ext(env, 0x200, 0, common::DISABLE_DEVICE_CONTROLLER | common::DISABLE_DEVICE_EXT_KEYBOARD, 0);
    assert_eq!(init(&ctx, PARAM, EXT), 0);
    assert_eq!(snapshot().disable_device, 3);
    assert_eq!(abort(&ctx), 0);
    assert_eq!(result_endstatus(env).0, 0);
    assert_eq!(term(&ctx), 0);

    // ForceClose ends a running dialog without writing its text back.
    assert_eq!(force_close(), ERROR_NOT_IN_USE);
    env.set_text("keep");
    write_param(env, make_param());
    assert_eq!(init(&ctx, PARAM, 0), 0);
    assert_eq!(get_status(&ctx), STATUS_RUNNING as i32);
    assert!(host_insert_text(snapshot().generation, &utf16("x")));
    assert_eq!(force_close(), 0);
    assert_eq!(get_status(&ctx), STATUS_NONE as i32);
    assert_eq!(term(&ctx), ERROR_NOT_IN_USE);
    assert_eq!(env.text(), utf16("keep"));
}

fn test_filtering_and_input_policy(env: &Env) {
    let ctx = env.ctx();
    env.set_text("1");
    let mut p = make_param();
    p.ty = TYPE_NUMBER;
    p.filter = COPY_FILTER;
    write_param(env, p);
    FILTER_CALLS.store(0, Ordering::SeqCst);
    assert_eq!(init(&ctx, PARAM, 0), 0);
    assert_eq!(get_status(&ctx), STATUS_RUNNING as i32);
    assert_eq!(FILTER_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(get_result(&ctx, 0), ERROR_INVALID_ADDRESS);
    assert_eq!(term(&ctx), ERROR_NOT_FINISHED);
    assert_eq!(FILTER_CALLS.load(Ordering::SeqCst), 0);

    let s = snapshot();
    assert!(host_insert_text(s.generation, &utf16("a2.\n")));
    assert_eq!(get_status(&ctx), STATUS_RUNNING as i32);
    assert_eq!(FILTER_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(env.text(), utf16("12."));
    assert!(host_cancel(s.generation));
    assert_eq!(result_endstatus(env).0, 0);
    assert_eq!(term(&ctx), 0);
}

fn key_input(status: u32, keycode: u16, action: ExternalAction, text: &str) -> ExternalInput {
    ExternalInput { key: Keycode { status, keycode, ..Default::default() }, action, text: utf16(text) }
}

fn test_external_keyboard_filter(env: &Env) {
    let ctx = env.ctx();
    env.set_text("");
    write_param(env, make_param());
    write_ext(env, 0, KEYBOARD_FILTER, 0, 0);
    assert_eq!(init(&ctx, PARAM, EXT), 0);
    let s = snapshot();
    let calls = || KEYBOARD_FILTER_CALLS.load(Ordering::SeqCst);

    let input = key_input(3, 4, ExternalAction::Text, "x");
    KEYBOARD_FILTER_CALLS.store(0, Ordering::SeqCst);
    BLOCK_KEYBOARD.store(1, Ordering::SeqCst);
    assert!(host_queue_external_input(s.generation, input.clone()));
    assert_eq!(get_status(&ctx), STATUS_RUNNING as i32);
    assert_eq!(calls(), 1);
    assert_eq!(env.unit(0), 0);

    BLOCK_KEYBOARD.store(0, Ordering::SeqCst);
    assert!(host_queue_external_input(s.generation, input));
    assert_eq!(get_status(&ctx), STATUS_RUNNING as i32);
    assert_eq!(calls(), 2);
    assert_eq!(env.text(), utf16("x"));

    REMAP_KEYBOARD.store(5, Ordering::SeqCst);
    assert!(host_queue_external_input(s.generation, key_input(3, 4, ExternalAction::Text, "x")));
    assert_eq!(get_status(&ctx), STATUS_RUNNING as i32);
    assert_eq!(calls(), 3);
    assert_eq!(env.text(), utf16("xb"));

    REMAP_KEYBOARD.store(50, Ordering::SeqCst);
    assert!(host_queue_external_input(s.generation, key_input(0x201, 4, ExternalAction::Text, "x")));
    assert_eq!(get_status(&ctx), STATUS_RUNNING as i32);
    assert_eq!(calls(), 4);
    assert_eq!(env.text(), utf16("xb~"));

    REMAP_KEYBOARD.store(88, Ordering::SeqCst);
    assert!(host_queue_external_input(s.generation, key_input(3, 4, ExternalAction::Text, "x")));
    assert_eq!(get_status(&ctx), STATUS_FINISHED as i32);
    assert_eq!(calls(), 5);
    assert_eq!(result_endstatus(env), (0, END_STATUS_OK));
    assert_eq!(term(&ctx), 0);

    REMAP_KEYBOARD.store(0, Ordering::SeqCst);
    assert_eq!(init(&ctx, PARAM, EXT), 0);
    let s = snapshot();
    assert!(host_queue_external_input(s.generation, key_input(1, 41, ExternalAction::Cancel, "")));
    assert!(host_queue_external_input(s.generation, key_input(3, 4, ExternalAction::Text, "ignored")));
    assert_eq!(get_status(&ctx), STATUS_FINISHED as i32);
    assert_eq!(calls(), 6);
    assert_eq!(result_endstatus(env).0, 0);
    assert_eq!(term(&ctx), 0);
}

fn test_filtered_cursor_boundary(env: &Env) {
    let ctx = env.ctx();
    env.set_text("");
    let mut p = make_param();
    p.filter = SUPPLEMENTARY_FILTER;
    write_param(env, p);
    assert_eq!(init(&ctx, PARAM, 0), 0);
    let s = snapshot();
    assert!(host_insert_text(s.generation, &utf16("x")));
    assert_eq!(get_status(&ctx), STATUS_RUNNING as i32);
    let s = snapshot();
    assert_eq!(s.cursor, 2);
    assert_eq!((env.unit(0), env.unit(1), env.unit(2)), (0xD83D, 0xDE00, 0));
    assert!(host_cancel(s.generation));
    assert_eq!(result_endstatus(env).0, 0);
    assert_eq!(term(&ctx), 0);
}

fn test_position_and_form(env: &Env) {
    let ctx = env.ctx();
    assert_eq!(get_panel_position_and_form(&ctx, FORM), ERROR_NOT_IN_USE);
    env.set_text("");
    write_param(env, make_param());
    assert_eq!(init(&ctx, PARAM, 0), 0);
    assert_eq!(get_panel_position_and_form(&ctx, 0), ERROR_INVALID_ADDRESS);
    assert_eq!(get_panel_position_and_form(&ctx, FORM), 0);
    assert_eq!(ctx.read_u32(FORM), Some(2));
    assert_eq!(ctx.read_u32(FORM + 0x14), Some(793));
    assert_eq!(ctx.read_u32(FORM + 0x18), Some(528));
    assert_eq!(term(&ctx), ERROR_NOT_FINISHED);
    assert_eq!(abort(&ctx), 0);
    assert_eq!(abort(&ctx), ERROR_NOT_RUNNING);
    assert_eq!(term(&ctx), 0);
}

fn test_headless_host(env: &Env) {
    let ctx = env.ctx();
    set_visibility_callback(None);
    env.set_text("");
    write_param(env, make_param());
    assert_eq!(init(&ctx, PARAM, 0), 0);
    assert_eq!(get_status(&ctx), STATUS_RUNNING as i32);
    assert_eq!(get_status(&ctx), STATUS_FINISHED as i32);
    assert_eq!(env.text(), common::default_input_text());
    assert_eq!(result_endstatus(env), (0, END_STATUS_OK));
    assert_eq!(term(&ctx), 0);
    set_visibility_callback(Some(noop_visibility));
}

#[test]
fn ime_dialog_reference_cases() {
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let env = Env::new();
    test_layout_and_panel_size(&env);
    test_accept_lifecycle(&env);
    test_cancel_and_abort_restore_text(&env);
    test_validation(&env);
    test_filtering_and_input_policy(&env);
    test_external_keyboard_filter(&env);
    test_filtered_cursor_boundary(&env);
    test_position_and_form(&env);
    test_headless_host(&env);
}

#[test]
fn nids_match_names() {
    for &(nid, name, _) in EXPORTS {
        assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
    }
}
