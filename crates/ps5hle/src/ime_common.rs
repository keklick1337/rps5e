// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Shared pieces of libSceIme and libSceImeDialog: the UTF-16 text editor,
//! input policy per IME type, guest text/keyboard filters, the host input
//! queue types and SceImeParamExtended.

use crate::context::CpuContext;
use std::sync::{Mutex, RwLock};

pub const MAX_TEXT_LENGTH: u32 = 2048;
pub const OPTION_MULTILINE: u32 = 0x0000_0001;
pub const OPTION_NO_AUTO_CAPITALIZE: u32 = 0x0000_0002;
pub const OPTION_PASSWORD: u32 = 0x0000_0004;
pub const OPTION_EXT_KEYBOARD: u32 = 0x0000_0010;
pub const OPTION_FIXED_POSITION: u32 = 0x0000_0040;
pub const OPTION_DISABLE_POSITION_ADJUST: u32 = 0x0000_0800;
pub const OPTION_EXPANDED_PREEDIT: u32 = 0x0000_1000;
pub const OPTION_USE_OVER_2K: u32 = 0x0000_4000;
pub const EXT_OPTION_HIDE_KEY_PANEL: u32 = 0x0000_0400;
pub const DISABLE_DEVICE_CONTROLLER: u32 = 0x0000_0001;
pub const DISABLE_DEVICE_EXT_KEYBOARD: u32 = 0x0000_0002;

/// SceImeType.
pub const TYPE_DEFAULT: u32 = 0;
pub const TYPE_BASIC_LATIN: u32 = 1;
pub const TYPE_URL: u32 = 2;
pub const TYPE_MAIL: u32 = 3;
pub const TYPE_NUMBER: u32 = 4;
/// SceImeEnterLabel: DEFAULT, SEND, SEARCH, GO.
pub const ENTER_LABEL_GO: u32 = 3;
/// SceImeHorizontalAlignment / SceImeVerticalAlignment: START, CENTER, END.
pub const ALIGNMENT_END: u32 = 2;

/// Text "typed" by the headless host into an empty dialog field
/// (override: RPS5E_DEFAULT_PROFILE).
pub const DEFAULT_INPUT_TEXT: &str = "rps5e";

pub fn default_input_text() -> Vec<u16> {
    std::env::var("RPS5E_DEFAULT_PROFILE")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_INPUT_TEXT.to_string())
        .encode_utf16()
        .collect()
}

pub fn utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

// ---------------------------------------------------------------------------
// SceImeKeycode (0x20 bytes)
// ---------------------------------------------------------------------------

pub const KEYCODE_SIZE: usize = 0x20;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Keycode {
    pub keycode: u16,
    pub character: u16,
    pub status: u32,
    pub ty: u32,
    pub user_id: i32,
    pub resource_id: u32,
    pub timestamp: u64,
}

impl Keycode {
    pub fn to_bytes(&self) -> [u8; KEYCODE_SIZE] {
        let mut b = [0u8; KEYCODE_SIZE];
        b[0..2].copy_from_slice(&self.keycode.to_le_bytes());
        b[2..4].copy_from_slice(&self.character.to_le_bytes());
        b[4..8].copy_from_slice(&self.status.to_le_bytes());
        b[8..12].copy_from_slice(&self.ty.to_le_bytes());
        b[12..16].copy_from_slice(&self.user_id.to_le_bytes());
        b[16..20].copy_from_slice(&self.resource_id.to_le_bytes());
        b[24..32].copy_from_slice(&self.timestamp.to_le_bytes());
        b
    }
}

// ---------------------------------------------------------------------------
// Host input
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExternalAction {
    #[default]
    None,
    Text,
    Backspace,
    MoveLeft,
    MoveRight,
    Cancel,
    Accept,
    Newline,
}

/// One key or text event from the host keyboard/UI.
#[derive(Clone, Debug, Default)]
pub struct ExternalInput {
    pub key: Keycode,
    pub action: ExternalAction,
    pub text: Vec<u16>,
}

/// What a host UI needs to draw an open IME session.
#[derive(Clone, Debug, Default)]
pub struct HostSnapshot {
    pub generation: u64,
    pub ty: u32,
    pub enter_label: u32,
    pub option: u32,
    pub max_text_length: u32,
    pub cursor: u32,
    pub disable_device: u32,
    pub key_panel_visible: bool,
    pub posx: f32,
    pub posy: f32,
    pub horizontal_alignment: u32,
    pub vertical_alignment: u32,
    pub panel_width: u32,
    pub panel_height: u32,
    pub text: Vec<u16>,
    pub title: Vec<u16>,
    pub placeholder: Vec<u16>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VisualState {
    pub active: bool,
    pub revision: u64,
}

/// Called with (visible, generation) when a session shows or hides. A
/// registered callback means a host UI drives the session.
pub type VisibilityCallback = fn(bool, u64);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EditDelta {
    pub index: u32,
    pub length: i32,
}

// ---------------------------------------------------------------------------
// Text helpers
// ---------------------------------------------------------------------------

const fn is_high_surrogate(v: u16) -> bool {
    v >= 0xD800 && v <= 0xDBFF
}

const fn is_low_surrogate(v: u16) -> bool {
    v >= 0xDC00 && v <= 0xDFFF
}

/// Character of a USB HID usage `keycode` under modifier `status`.
fn hid_character(keycode: u16, status: u32) -> u16 {
    let shift = status & 0x0000_2200 != 0;
    let caps = status & 0x0002_0000 != 0;
    let pick = |plain: &str, shifted: &str, i: usize, use_shift: bool| -> u16 {
        let s = if use_shift { shifted } else { plain };
        s.encode_utf16().nth(i).unwrap_or(0)
    };
    match keycode {
        4..=29 => (if shift != caps { b'A' } else { b'a' }) as u16 + keycode - 4,
        30..=39 => pick("1234567890", "!@#$%^&*()", (keycode - 30) as usize, shift),
        44 => b' ' as u16,
        45..=56 => pick("-=[]\\#;'`,./", "_+{}|~:\"~<>?", (keycode - 45) as usize, shift),
        89..=98 => pick("1234567890", "", (keycode - 89) as usize, false),
        99 => b'.' as u16,
        _ => 0,
    }
}

pub fn is_valid_utf16(text: &[u16]) -> bool {
    let mut i = 0;
    while i < text.len() {
        if is_high_surrogate(text[i]) {
            i += 1;
            if i >= text.len() || !is_low_surrogate(text[i]) {
                return false;
            }
        } else if is_low_surrogate(text[i]) {
            return false;
        }
        i += 1;
    }
    true
}

pub fn is_valid_input_text(text: &[u16], multiline: bool) -> bool {
    is_valid_utf16(text) && !text.iter().any(|&v| v == 0 || (!multiline && (v == 0x0A || v == 0x0D)))
}

pub fn is_allowed_input(value: u16, ty: u32, option: u32) -> bool {
    if value == 0x0D || value == 0 {
        return false;
    }
    if value == 0x0A {
        return option & OPTION_MULTILINE != 0;
    }
    match ty {
        TYPE_NUMBER => (b'0' as u16..=b'9' as u16).contains(&value) || [b',', b'-', b'.'].iter().any(|&c| c as u16 == value),
        TYPE_BASIC_LATIN => (0x20..=0x7E).contains(&value),
        _ => true,
    }
}

pub fn normalize_cursor(text: &[u16], cursor: u32) -> u32 {
    let mut cursor = cursor.min(text.len() as u32);
    let c = cursor as usize;
    if c > 0 && c < text.len() && is_high_surrogate(text[c - 1]) && is_low_surrogate(text[c]) {
        cursor += 1;
    }
    cursor
}

pub fn clamp_text(text: &mut Vec<u16>, limit: u32) {
    if text.len() <= limit as usize {
        return;
    }
    text.truncate(limit as usize);
    if text.last().is_some_and(|&v| is_high_surrogate(v)) {
        text.pop();
    }
}

pub fn compute_edit_delta(before: &[u16], after: &[u16]) -> EditDelta {
    let mut prefix = 0;
    while prefix < before.len() && prefix < after.len() && before[prefix] == after[prefix] {
        prefix += 1;
    }
    let (mut bt, mut at) = (before.len(), after.len());
    while bt > prefix && at > prefix && before[bt - 1] == after[at - 1] {
        bt -= 1;
        at -= 1;
    }
    EditDelta { index: prefix as u32, length: (at - prefix) as i32 - (bt - prefix) as i32 }
}

/// Re-map a host input after the guest keyboard filter rewrote its keycode
/// and status; false drops the input.
pub fn apply_keyboard_filter_output(input: &mut ExternalInput, keycode: u16, status: u32, multiline: bool) -> bool {
    const KEYCODE_VALID: u32 = 0x1;
    const CHARACTER_VALID: u32 = 0x2;
    if keycode == input.key.keycode && status == input.key.status {
        return true;
    }
    if status & KEYCODE_VALID == 0 || keycode == 0 {
        return input.action == ExternalAction::Text && status & CHARACTER_VALID != 0;
    }
    input.key.keycode = keycode;
    input.key.status = status;
    input.text.clear();
    input.action = match keycode {
        40 | 88 | 158 => {
            if multiline {
                ExternalAction::Newline
            } else {
                ExternalAction::Accept
            }
        }
        41 => ExternalAction::Cancel,
        42 | 187 => ExternalAction::Backspace,
        43 | 186 => ExternalAction::None,
        79 => ExternalAction::MoveRight,
        80 => ExternalAction::MoveLeft,
        _ => {
            let ch = hid_character(keycode, status);
            if ch == 0 {
                return false;
            }
            input.key.character = ch;
            input.text.push(ch);
            ExternalAction::Text
        }
    };
    true
}

// ---------------------------------------------------------------------------
// Text edit engine
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TextEditEngine {
    ty: u32,
    option: u32,
    max_length: u32,
    cursor: u32,
    text: Vec<u16>,
}

impl TextEditEngine {
    pub const fn new() -> Self {
        Self { ty: TYPE_DEFAULT, option: 0, max_length: 0, cursor: 0, text: Vec::new() }
    }

    pub fn reset(&mut self, ty: u32, option: u32, max_length: u32, mut text: Vec<u16>) {
        self.ty = ty;
        self.option = option;
        self.max_length = max_length;
        clamp_text(&mut text, max_length);
        self.cursor = text.len() as u32;
        self.text = text;
    }

    /// Insert at the cursor, keeping only characters the type allows.
    pub fn insert(&mut self, text: &[u16]) -> Option<EditDelta> {
        if text.is_empty() || !is_valid_utf16(text) {
            return None;
        }
        let mut allowed: Vec<u16> = text.iter().copied().filter(|&v| is_allowed_input(v, self.ty, self.option)).collect();
        let available = (self.max_length as usize).saturating_sub(self.text.len());
        if allowed.is_empty() || available == 0 {
            return None;
        }
        allowed.truncate(available);
        if allowed.last().is_some_and(|&v| is_high_surrogate(v)) {
            allowed.pop();
        }
        if allowed.is_empty() {
            return None;
        }
        let edit = EditDelta { index: self.cursor, length: allowed.len() as i32 };
        let at = self.cursor as usize;
        self.text.splice(at..at, allowed.iter().copied());
        self.cursor += allowed.len() as u32;
        Some(edit)
    }

    pub fn backspace(&mut self) -> Option<EditDelta> {
        if self.cursor == 0 {
            return None;
        }
        let mut first = self.cursor - 1;
        let f = first as usize;
        if first > 0 && is_low_surrogate(self.text[f]) && is_high_surrogate(self.text[f - 1]) {
            first -= 1;
        }
        let edit = EditDelta { index: first, length: -((self.cursor - first) as i32) };
        self.text.drain(first as usize..self.cursor as usize);
        self.cursor = first;
        Some(edit)
    }

    pub fn move_cursor(&mut self, delta: i32) -> bool {
        if delta == 0 {
            return false;
        }
        let len = self.text.len() as i32;
        let mut next = (self.cursor as i32 + delta).clamp(0, len);
        let n = next as usize;
        if delta < 0 && next > 0 && next < len && is_low_surrogate(self.text[n]) && is_high_surrogate(self.text[n - 1]) {
            next -= 1;
        } else if delta > 0 && next > 0 && next < len && is_high_surrogate(self.text[n - 1]) && is_low_surrogate(self.text[n]) {
            next += 1;
        }
        if next == self.cursor as i32 {
            return false;
        }
        self.cursor = next as u32;
        true
    }

    pub fn replace_text(&mut self, mut text: Vec<u16>, cursor: u32) {
        clamp_text(&mut text, self.max_length);
        self.text = text;
        self.cursor = normalize_cursor(&self.text, cursor);
    }

    pub fn set_cursor(&mut self, cursor: u32) {
        self.cursor = normalize_cursor(&self.text, cursor);
    }

    pub fn ty(&self) -> u32 {
        self.ty
    }
    pub fn option(&self) -> u32 {
        self.option
    }
    pub fn max_length(&self) -> u32 {
        self.max_length
    }
    pub fn cursor(&self) -> u32 {
        self.cursor
    }
    pub fn text(&self) -> &[u16] {
        &self.text
    }
}

// ---------------------------------------------------------------------------
// SceImeParamExtended (0x88 bytes)
// ---------------------------------------------------------------------------

pub const EXTENDED_PARAM_SIZE: usize = 0x88;
const EXT_OPTION_REQUIRES_80: u32 = 0x0000_4000;
const VALID_EXT_KEYBOARD_MODE: u32 = 0x1C00_0003;

/// option @0, nine colors, priority @0x28, additionalDictionaryPath @0x30,
/// extKeyboardFilter @0x38, disableDevice @0x40, extKeyboardMode @0x44,
/// reserved[60] @0x48.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExtendedParam {
    pub option: u32,
    pub priority: u32,
    pub ext_keyboard_filter: u64,
    pub disable_device: u32,
    pub ext_keyboard_mode: u32,
    pub reserved_zero: bool,
}

impl ExtendedParam {
    pub fn parse(raw: &[u8]) -> Self {
        let u32_at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
        Self {
            option: u32_at(0),
            priority: u32_at(0x28),
            ext_keyboard_filter: u64::from_le_bytes(raw[0x38..0x40].try_into().unwrap()),
            disable_device: u32_at(0x40),
            ext_keyboard_mode: u32_at(0x44),
            reserved_zero: raw[0x48..0x48 + 60].iter().all(|&b| b == 0),
        }
    }

    pub fn is_valid(&self, valid_options: u32) -> bool {
        self.option & !valid_options == 0
            && (self.option & EXT_OPTION_REQUIRES_80 == 0 || self.option & 0x80 != 0)
            && self.priority <= 3
            && self.disable_device <= 7
            && self.ext_keyboard_mode & !VALID_EXT_KEYBOARD_MODE == 0
            && self.reserved_zero
    }
}

/// Read an optional SceImeParamExtended; Err on an unreadable pointer.
pub fn read_extended(ctx: &CpuContext, addr: u64) -> Result<Option<ExtendedParam>, ()> {
    if addr == 0 {
        return Ok(None);
    }
    let raw = ctx.read_bytes(addr, EXTENDED_PARAM_SIZE).ok_or(())?;
    Ok(Some(ExtendedParam::parse(&raw)))
}

// ---------------------------------------------------------------------------
// Guest memory helpers
// ---------------------------------------------------------------------------

/// UTF-16 units at `addr` up to (not including) the first NUL, reading at
/// most `max_units`. None if memory faults before a NUL or the limit.
pub fn scan_units(ctx: &CpuContext, addr: u64, max_units: usize) -> Option<Vec<u16>> {
    const CHUNK: usize = 64;
    let mut out = Vec::new();
    while out.len() < max_units {
        let n = CHUNK.min(max_units - out.len());
        let at = addr + out.len() as u64 * 2;
        let bytes = match ctx.read_bytes(at, n * 2) {
            Some(b) => b,
            None => ctx.read_bytes(at, 2)?,
        };
        for c in bytes.chunks_exact(2) {
            let v = u16::from_le_bytes([c[0], c[1]]);
            if v == 0 {
                return Some(out);
            }
            out.push(v);
        }
    }
    Some(out)
}

pub fn read_units(ctx: &CpuContext, addr: u64, count: usize) -> Option<Vec<u16>> {
    let bytes = ctx.read_bytes(addr, count * 2)?;
    Some(bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}

/// Write `text` (first `max` units) plus a NUL terminator.
pub fn write_units_nul(ctx: &CpuContext, addr: u64, text: &[u16], max: u32) -> bool {
    let n = text.len().min(max as usize);
    let mut bytes: Vec<u8> = text[..n].iter().flat_map(|v| v.to_le_bytes()).collect();
    bytes.extend_from_slice(&[0, 0]);
    ctx.write_bytes(addr, &bytes)
}

// ---------------------------------------------------------------------------
// Guest calls (filters, event handlers)
// ---------------------------------------------------------------------------

pub(crate) type GuestCallFn = fn(u64, [u64; 6]) -> u64;

fn default_guest_call(func: u64, args: [u64; 6]) -> u64 {
    ps5cpu::exec::call_guest_nested(func, args)
}

static GUEST_CALL: RwLock<GuestCallFn> = RwLock::new(default_guest_call);

pub(crate) fn guest_call(func: u64, args: [u64; 6]) -> u64 {
    let f = *GUEST_CALL.read().unwrap_or_else(|p| p.into_inner());
    f(func, args)
}

#[cfg(test)]
pub(crate) fn set_guest_call(f: GuestCallFn) {
    *GUEST_CALL.write().unwrap_or_else(|p| p.into_inner()) = f;
}

/// Guest scratch block for callback arguments: filter output text, source
/// text, lengths, a keycode and an event. Blocks are pooled, never freed.
pub(crate) const SCRATCH_OUT_TEXT: u64 = 0x0000;
pub(crate) const SCRATCH_SOURCE: u64 = 0x1100;
pub(crate) const SCRATCH_LENGTH: u64 = 0x2200;
pub(crate) const SCRATCH_KEYCODE: u64 = 0x2210;
pub(crate) const SCRATCH_OUT_KEYCODE: u64 = 0x2230;
pub(crate) const SCRATCH_OUT_STATUS: u64 = 0x2234;
pub(crate) const SCRATCH_EVENT: u64 = 0x2240;
const SCRATCH_SIZE: u64 = 0x2300;

static SCRATCH_POOL: Mutex<Vec<u64>> = Mutex::new(Vec::new());

pub(crate) struct Scratch(pub u64);

impl Scratch {
    pub(crate) fn acquire(ctx: &CpuContext) -> Option<Self> {
        let pooled = SCRATCH_POOL.lock().unwrap_or_else(|p| p.into_inner()).pop();
        match pooled {
            Some(a) => Some(Self(a)),
            None => crate::pthread::alloc_guest_block(ctx, SCRATCH_SIZE).map(Self),
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        SCRATCH_POOL.lock().unwrap_or_else(|p| p.into_inner()).push(self.0);
    }
}

/// Run a guest SceImeTextFilter over `source`; Some(filtered) only when it
/// returns 0 with a valid UTF-16 result of at most `capacity` units.
pub fn run_text_filter(ctx: &CpuContext, filter: u64, source: &[u16], capacity: u32) -> Option<Vec<u16>> {
    if filter == 0 {
        return None;
    }
    let scratch = Scratch::acquire(ctx)?;
    let out = scratch.0 + SCRATCH_OUT_TEXT;
    let src = scratch.0 + SCRATCH_SOURCE;
    let len = scratch.0 + SCRATCH_LENGTH;
    let capacity = capacity.min(MAX_TEXT_LENGTH);
    if !ctx.write_bytes(out, &vec![0u8; (capacity as usize + 1) * 2])
        || !write_units_nul(ctx, src, source, MAX_TEXT_LENGTH)
        || !ctx.write_u32(len, capacity)
    {
        return None;
    }
    let rc = guest_call(filter, [out, len, src, source.len().min(MAX_TEXT_LENGTH as usize) as u64, 0, 0]) as i32;
    let n = ctx.read_u32(len)?;
    if rc != 0 || n > capacity {
        return None;
    }
    let text = read_units(ctx, out, n as usize)?;
    is_valid_utf16(&text).then_some(text)
}

/// Run a guest SceImeExtKeyboardFilter; returns (rc, keycode, status).
pub fn run_keyboard_filter(ctx: &CpuContext, filter: u64, key: &Keycode) -> Option<(i32, u16, u32)> {
    let scratch = Scratch::acquire(ctx)?;
    let k = scratch.0 + SCRATCH_KEYCODE;
    let ok = scratch.0 + SCRATCH_OUT_KEYCODE;
    let os = scratch.0 + SCRATCH_OUT_STATUS;
    if !ctx.write_bytes(k, &key.to_bytes()) || !ctx.write_bytes(ok, &key.keycode.to_le_bytes()) || !ctx.write_u32(os, key.status) {
        return None;
    }
    let rc = guest_call(filter, [k, ok, os, 0, 0, 0]) as i32;
    let kc = ctx.read_bytes(ok, 2)?;
    Some((rc, u16::from_le_bytes([kc[0], kc[1]]), ctx.read_u32(os)?))
}

/// Shared mock for unit tests: fake guest functions keyed by address.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use std::collections::HashMap;

    pub type FakeFn = fn(&CpuContext, [u64; 6]) -> u64;
    static FAKES: Mutex<Option<HashMap<u64, FakeFn>>> = Mutex::new(None);

    fn dispatch(func: u64, args: [u64; 6]) -> u64 {
        let f = FAKES.lock().unwrap().as_ref().and_then(|m| m.get(&func).copied());
        let vm = ps5vm::VirtualMemory::new();
        let ctx = CpuContext::new(&vm);
        f.unwrap_or_else(|| panic!("no fake guest function at {func:#x}"))(&ctx, args)
    }

    pub fn register_fake(func: u64, f: FakeFn) {
        FAKES.lock().unwrap().get_or_insert_with(HashMap::new).insert(func, f);
        set_guest_call(dispatch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_engine_surrogates_and_limits() {
        let mut e = TextEditEngine::new();
        e.reset(TYPE_DEFAULT, 0, 4, utf16("abcdef"));
        assert_eq!(e.text(), &utf16("abcd")[..]);
        assert_eq!(e.cursor(), 4);
        assert_eq!(e.insert(&utf16("x")), None);
        assert_eq!(e.backspace(), Some(EditDelta { index: 3, length: -1 }));
        assert_eq!(e.insert(&[0xD83D, 0xDE00]), None); // pair cut at the limit -> nothing fits
        assert_eq!(e.backspace(), Some(EditDelta { index: 2, length: -1 }));
        assert_eq!(e.insert(&[0xD83D, 0xDE00]), Some(EditDelta { index: 2, length: 2 }));
        assert_eq!(e.backspace(), Some(EditDelta { index: 2, length: -2 }));
        e.reset(TYPE_DEFAULT, 0, 8, vec![b'a' as u16, 0xD83D, 0xDE00]);
        assert!(e.move_cursor(-1));
        assert_eq!(e.cursor(), 1);
        assert!(e.move_cursor(1));
        assert_eq!(e.cursor(), 3);
        e.set_cursor(2);
        assert_eq!(e.cursor(), 3);
    }

    #[test]
    fn input_policy() {
        assert!(is_allowed_input(b'5' as u16, TYPE_NUMBER, 0));
        assert!(!is_allowed_input(b'a' as u16, TYPE_NUMBER, 0));
        assert!(!is_allowed_input(0xE9, TYPE_BASIC_LATIN, 0));
        assert!(is_allowed_input(0x0A, TYPE_DEFAULT, OPTION_MULTILINE));
        assert!(!is_allowed_input(0x0A, TYPE_DEFAULT, 0));
        assert!(!is_allowed_input(0x0D, TYPE_DEFAULT, OPTION_MULTILINE));
        assert!(!is_valid_utf16(&[0xDC00]));
        assert!(!is_valid_utf16(&[0xD800]));
        assert_eq!(compute_edit_delta(&utf16("abc"), &utf16("abXc")), EditDelta { index: 2, length: 1 });
        assert_eq!(compute_edit_delta(&utf16("abc"), &utf16("ac")), EditDelta { index: 1, length: -1 });
    }

    #[test]
    fn keyboard_filter_output_mapping() {
        let mut i = ExternalInput { key: Keycode { keycode: 4, status: 3, ..Default::default() }, action: ExternalAction::Text, text: utf16("x") };
        assert!(apply_keyboard_filter_output(&mut i, 5, 3, false));
        assert_eq!((i.action, i.text.clone()), (ExternalAction::Text, utf16("b")));
        assert!(apply_keyboard_filter_output(&mut i, 4, 0x0002_0001, false));
        assert_eq!(i.text, utf16("A"));
        assert!(apply_keyboard_filter_output(&mut i, 40, 1, true));
        assert_eq!(i.action, ExternalAction::Newline);
        assert!(!apply_keyboard_filter_output(&mut i, 0, 0, false));
        assert!(!apply_keyboard_filter_output(&mut i, 200, 1, false));
    }

    #[test]
    fn extended_param_validation() {
        let mut raw = [0u8; EXTENDED_PARAM_SIZE];
        assert!(ExtendedParam::parse(&raw).is_valid(0x5FDE));
        raw[0x84] = 1; // trailing padding is not part of reserved[60]
        assert!(ExtendedParam::parse(&raw).is_valid(0x5FDE));
        raw[0x50] = 1;
        assert!(!ExtendedParam::parse(&raw).is_valid(0x5FDE));
        let mut raw = [0u8; EXTENDED_PARAM_SIZE];
        raw[0..4].copy_from_slice(&0x4000u32.to_le_bytes());
        assert!(!ExtendedParam::parse(&raw).is_valid(0x5FDE));
        raw[0..4].copy_from_slice(&0x4080u32.to_le_bytes());
        assert!(ExtendedParam::parse(&raw).is_valid(0x5FDE));
        assert!(!ExtendedParam::parse(&raw).is_valid(0x4FDE & !0x80));
    }
}
