// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceConvertKeycode HLE: keyboard layout queries.

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceConvertKeycode";
const OK: HleResult = 0;

const ERR_INVALID_ADDRESS: HleResult = 0x80BC_0031u32 as i32 as i64;
const ERR_INVALID_USER_ID: HleResult = 0x80BC_0010u32 as i32 as i64;
const ERR_INVALID_KEYBOARD_TYPE: HleResult = 0x80BC_001Du32 as i32 as i64;

/// IME keyboard types run from 1 to 39; 0 is not a layout.
const KEYBOARD_TYPE_MAX: u32 = 39;
/// The US English layout: the type reported for every user and the one
/// whose characters and virtual keys are converted here.
const KEYBOARD_TYPE_US: u32 = 4;

fn check_keyboard_type(ty: u32) -> Result<(), HleResult> {
    if ty == 0 || ty > KEYBOARD_TYPE_MAX { Err(ERR_INVALID_KEYBOARD_TYPE) } else { Ok(()) }
}

/// sceConvertKeycodeGetImeKeyboardType(userId, u32* type)
fn cb_get_ime_keyboard_type(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return ERR_INVALID_ADDRESS;
    }
    if (ctx.rdi() as i32) < 0 {
        return ERR_INVALID_USER_ID;
    }
    if ctx.write_u32(out, KEYBOARD_TYPE_US) { OK } else { ERR_INVALID_ADDRESS }
}

/// State bit that does not change the character of a key.
const IGNORED_STATE_BITS: u32 = 0x400_0000;
/// What an unresolved import returns: the case is not implemented.
const ERR_NOT_IMPLEMENTED: HleResult = 0x8002_0002u32 as i32 as i64;

/// The character of a USB HID keycode on the US layout without modifiers
/// (0 for keys that type none).
fn us_character(keycode: u16) -> u16 {
    const PUNCTUATION: &[u8; 12] = b"-=[]\\\0;'`,./";
    const KEYPAD: &[u8; 16] = b"/*-+\01234567890.";
    let c = match keycode {
        0x04..=0x1D => b'a' + (keycode - 0x04) as u8,
        0x1E..=0x26 => b'1' + (keycode - 0x1E) as u8,
        0x27 => b'0',
        0x2C => b' ',
        0x2D..=0x38 => PUNCTUATION[(keycode - 0x2D) as usize],
        0x54..=0x63 => KEYPAD[(keycode - 0x54) as usize],
        0x64 => b'\\',
        _ => 0,
    };
    c as u16
}

/// sceConvertKeycodeGetCharacter(u16 keycode, u32 state, u32 type,
/// char16_t* character): the address is checked first, then the keyboard
/// type; the US layout without modifiers is converted, other states and
/// layouts are not implemented.
fn cb_get_character(ctx: &mut CpuContext) -> HleResult {
    let (keycode, state, ty, out) = (ctx.rdi() as u16, ctx.rsi() as u32, ctx.rdx() as u32, ctx.rcx());
    if out == 0 {
        return ERR_INVALID_ADDRESS;
    }
    if let Err(e) = check_keyboard_type(ty) {
        return e;
    }
    if state & !IGNORED_STATE_BITS != 0 || ty != KEYBOARD_TYPE_US {
        log::warn!("[convertkeycode] GetCharacter keycode {keycode:#x} state {state:#x} type {ty}: not implemented");
        return ERR_NOT_IMPLEMENTED;
    }
    if ctx.write_bytes(out, &us_character(keycode).to_le_bytes()) { OK } else { ERR_INVALID_ADDRESS }
}

/// Windows-style virtual key of each USB HID keycode on the US layout
/// (255: no virtual key).
const US_VIRTUAL_KEYS: [u8; 256] = {
    let mut t = [255u8; 256];
    let mut k = 0;
    while k < 26 {
        t[0x04 + k] = b'A' + k as u8;
        k += 1;
    }
    let mut k = 0;
    while k < 9 {
        t[0x1E + k] = b'1' + k as u8;
        k += 1;
    }
    t[0x27] = b'0';
    // 0x28..: Enter, Escape, Backspace, Tab, Space, - = [ ] \ (non-US #)
    // ; ' ` , . /, CapsLock, F1, F2.
    let row: [u8; 20] = [13, 27, 8, 9, 32, 189, 187, 219, 221, 220, 220, 186, 222, 192, 188, 190, 191, 20, 112, 113];
    let mut k = 0;
    while k < row.len() {
        t[0x28 + k] = row[k];
        k += 1;
    }
    // 0x3C..: F3..F12, PrintScreen, ScrollLock, Pause, Insert, Home, PageUp,
    // Delete, End, PageDown, Right, Left, Down, Up, NumLock, keypad / * - +
    // Enter 1..9 0 ., non-US \ (none), Application.
    let rest: [u8; 42] = [
        114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 44, 145, 19, 45, 36, 33, 46, 35, 34, 39, 37, 40, 38, 144, 111, 106,
        109, 107, 13, 97, 98, 99, 100, 101, 102, 103, 104, 105, 96, 110, 255, 93,
    ];
    let mut k = 0;
    while k < rest.len() {
        t[0x3C + k] = rest[k];
        k += 1;
    }
    t[0xD8] = 12;
    // 0xE0..: left Ctrl, Shift, Alt, GUI; right Ctrl, Shift, Alt, GUI.
    let modifiers: [u8; 8] = [162, 160, 164, 91, 163, 161, 165, 92];
    let mut k = 0;
    while k < modifiers.len() {
        t[0xE0 + k] = modifiers[k];
        k += 1;
    }
    t
};

/// sceConvertKeycodeGetVirtualKeycode(u16 keycode, u32 keyboardType,
/// u16* virtualKeycode): the address is checked first, then the keyboard
/// type; the US layout is converted, other layouts are not implemented.
fn cb_get_virtual_keycode(ctx: &mut CpuContext) -> HleResult {
    let (keycode, ty, out) = (ctx.rdi() as u16, ctx.rsi() as u32, ctx.rdx());
    if out == 0 {
        return ERR_INVALID_ADDRESS;
    }
    if let Err(e) = check_keyboard_type(ty) {
        return e;
    }
    if ty != KEYBOARD_TYPE_US {
        log::warn!("[convertkeycode] GetVirtualKeycode keycode {keycode:#x} type {ty}: not implemented");
        return ERR_NOT_IMPLEMENTED;
    }
    let key = US_VIRTUAL_KEYS.get(keycode as usize).copied().unwrap_or(255) as u16;
    if ctx.write_bytes(out, &key.to_le_bytes()) { OK } else { ERR_INVALID_ADDRESS }
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("mUuUOWI-C+0", "sceConvertKeycodeGetImeKeyboardType", cb_get_ime_keyboard_type),
    ("QjGCaJbRib4", "sceConvertKeycodeGetCharacter", cb_get_character),
    ("vIsoJsLvvlM", "sceConvertKeycodeGetVirtualKeycode", cb_get_virtual_keycode),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// Null address first, then the keyboard type (0 and above 39 are not
    /// layouts); the US layout maps keys to Windows-style virtual keys.
    #[test]
    fn virtual_keycodes_and_validation() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A6A_0100_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let call = |ctx: &mut CpuContext, key: u16, ty: u32, out: u64| {
            ctx.set(Rdi, key as u64);
            ctx.set(Rsi, ty as u64);
            ctx.set(Rdx, out);
            cb_get_virtual_keycode(ctx)
        };
        assert_eq!(call(&mut ctx, 0, 0, 0), ERR_INVALID_ADDRESS);
        assert_eq!(call(&mut ctx, 0, 0, BASE), ERR_INVALID_KEYBOARD_TYPE);
        assert_eq!(call(&mut ctx, 0, KEYBOARD_TYPE_MAX + 1, BASE), ERR_INVALID_KEYBOARD_TYPE);
        let expect: [(u16, u16); 24] = [
            (0x04, 65),
            (0x1D, 90),
            (0x1E, 49),
            (0x27, 48),
            (0x28, 13),
            (0x2D, 189),
            (0x2E, 187),
            (0x33, 186),
            (0x34, 222),
            (0x35, 192),
            (0x38, 191),
            (0x39, 20),
            (0x3A, 112),
            (0x45, 123),
            (0x4F, 39),
            (0x53, 144),
            (0x58, 13),
            (0x62, 96),
            (0x63, 110),
            (0x64, 255),
            (0x65, 93),
            (0xE0, 162),
            (0xE7, 92),
            (0x100, 255),
        ];
        for (key, vk) in expect {
            assert_eq!(call(&mut ctx, key, KEYBOARD_TYPE_US, BASE), OK, "keycode {key:#x}");
            assert_eq!(ctx.read_bytes(BASE, 2).unwrap(), vk.to_le_bytes().to_vec(), "keycode {key:#x}");
        }
    }

    #[test]
    fn keyboard_type() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A6A_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_u32(BASE, 7);
        ctx.set(Rdi, crate::user_service::PRIMARY_USER_ID as u64);
        ctx.set(Rsi, BASE);
        assert_eq!(cb_get_ime_keyboard_type(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE), Some(KEYBOARD_TYPE_US));
        ctx.set(Rdi, u64::MAX);
        assert_eq!(cb_get_ime_keyboard_type(&mut ctx), ERR_INVALID_USER_ID);
    }

    #[test]
    fn us_characters() {
        let expect = [(0x04, 'a'), (0x1D, 'z'), (0x1E, '1'), (0x26, '9'), (0x27, '0'), (0x2C, ' '), (0x2D, '-'), (0x31, '\\'), (0x33, ';'), (0x38, '/'), (0x54, '/'), (0x59, '1'), (0x62, '0'), (0x63, '.'), (0x64, '\\')];
        for (key, c) in expect {
            assert_eq!(us_character(key), c as u16, "keycode {key:#x}");
        }
        for key in [0x00, 0x32, 0x3A, 0x58, 0x65] {
            assert_eq!(us_character(key), 0, "keycode {key:#x}");
        }
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D46_0300_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        for state in [0, IGNORED_STATE_BITS] {
            for (r, v) in [(Rdi, 0x04), (Rsi, state as u64), (Rdx, KEYBOARD_TYPE_US as u64), (Rcx, BASE)] {
                ctx.set(r, v);
            }
            assert_eq!(cb_get_character(&mut ctx), OK);
            assert_eq!(ctx.read_bytes(BASE, 2).unwrap(), b"a\0".to_vec());
        }
        ctx.set(Rsi, 1);
        assert_eq!(cb_get_character(&mut ctx), ERR_NOT_IMPLEMENTED);
        ctx.set(Rsi, 0);
        for ty in [0, KEYBOARD_TYPE_MAX + 1] {
            ctx.set(Rdx, ty as u64);
            assert_eq!(cb_get_character(&mut ctx), ERR_INVALID_KEYBOARD_TYPE);
        }
        ctx.set(Rcx, 0);
        assert_eq!(cb_get_character(&mut ctx), ERR_INVALID_ADDRESS);
    }
}
