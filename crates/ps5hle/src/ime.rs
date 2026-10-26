// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceIme: the in-game IME session and the external keyboard.
//!
//! An open session owns the guest work buffer (20 KiB) and input buffer.
//! Host input (`host_*`, queued) is applied on sceImeUpdate: text edits go
//! through the guest text filter and are reported to the guest event
//! handler as SceImeEvent records (OPEN, UPDATE_TEXT, UPDATE_CARET,
//! PRESS_CLOSE, PRESS_ENTER). No physical keyboard is ever connected.

use crate::context::CpuContext;
use crate::ime_common::{
    self as common, ExtendedParam, ExternalAction, ExternalInput, HostSnapshot, Scratch, TextEditEngine,
    VisibilityCallback, VisualState, ALIGNMENT_END, ENTER_LABEL_GO, OPTION_EXPANDED_PREEDIT, OPTION_EXT_KEYBOARD,
    OPTION_MULTILINE, OPTION_PASSWORD, OPTION_USE_OVER_2K, TYPE_BASIC_LATIN, TYPE_DEFAULT, TYPE_NUMBER,
};
use crate::registry::{HleCallback, HleRegistry, HleResult};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex, MutexGuard, RwLock};

pub const WORK_BUFFER_SIZE: u32 = 20 * 1024;
pub const MAX_TEXT_LENGTH: u32 = common::MAX_TEXT_LENGTH;

const fn err(code: u32) -> i32 {
    code as i32
}

pub const ERROR_BUSY: i32 = err(0x80BC_0001);
pub const ERROR_NOT_OPENED: i32 = err(0x80BC_0002);
pub const ERROR_CONNECTION_FAILED: i32 = err(0x80BC_0004);
pub const ERROR_INVALID_TEXT: i32 = err(0x80BC_0006);
pub const ERROR_EVENT_OVERFLOW: i32 = err(0x80BC_0007);
pub const ERROR_INVALID_USER_ID: i32 = err(0x80BC_0010);
pub const ERROR_INVALID_TYPE: i32 = err(0x80BC_0011);
pub const ERROR_INVALID_LANGUAGES: i32 = err(0x80BC_0012);
pub const ERROR_INVALID_ENTER_LABEL: i32 = err(0x80BC_0013);
pub const ERROR_INVALID_INPUT_METHOD: i32 = err(0x80BC_0014);
pub const ERROR_INVALID_OPTION: i32 = err(0x80BC_0015);
pub const ERROR_INVALID_MAX_TEXT_LENGTH: i32 = err(0x80BC_0016);
pub const ERROR_INVALID_TEXT_BUFFER: i32 = err(0x80BC_0017);
pub const ERROR_INVALID_POSX: i32 = err(0x80BC_0018);
pub const ERROR_INVALID_POSY: i32 = err(0x80BC_0019);
pub const ERROR_INVALID_HALIGN: i32 = err(0x80BC_001A);
pub const ERROR_INVALID_VALIGN: i32 = err(0x80BC_001B);
pub const ERROR_INVALID_EXTENDED: i32 = err(0x80BC_001C);
pub const ERROR_INVALID_WORK: i32 = err(0x80BC_0020);
pub const ERROR_INVALID_HANDLER: i32 = err(0x80BC_0022);
pub const ERROR_NO_RESOURCE_ID: i32 = err(0x80BC_0023);
pub const ERROR_INVALID_MODE: i32 = err(0x80BC_0024);
pub const ERROR_INVALID_PARAM: i32 = err(0x80BC_0030);
pub const ERROR_INVALID_ADDRESS: i32 = err(0x80BC_0031);
pub const ERROR_INVALID_RESERVED: i32 = err(0x80BC_0032);

const VALID_OPTIONS: u32 = 0x0000_7BFF;
const VALID_EXTENDED_OPTIONS: u32 = 0x0000_4FDE;
const VALID_KEYBOARD_OPTIONS: u32 = 0x0000_003F;
const VALID_KEYBOARD_MODE: u32 = 0x0000_007F;
const VALID_LANGUAGES: u64 = 0x0000_0001_FF1F_FFFF;
const EVENT_QUEUE_CAPACITY: usize = 128;

/// SceImeParam (0x60 bytes).
pub const PARAM_SIZE: usize = 0x60;
/// SceImeKeyboardParam (0x20 bytes).
pub const KEYBOARD_PARAM_SIZE: usize = 0x20;
/// SceImeKeyboardResourceIdArray { userId; u32 resourceId[5]; }.
pub const KEYBOARD_RESOURCE_ID_ARRAY_SIZE: usize = 0x18;
/// SceImeKeyboardInfo.
pub const KEYBOARD_INFO_SIZE: usize = 0x24;
/// SceImeEvent { u32 id; SceImeEventParam param @8 (0x40) }.
pub const EVENT_SIZE: usize = 0x48;

/// SceImeEventId.
pub const EVENT_OPEN: u32 = 0;
pub const EVENT_UPDATE_TEXT: u32 = 1;
pub const EVENT_UPDATE_CARET: u32 = 2;
pub const EVENT_PRESS_CLOSE: u32 = 4;
pub const EVENT_PRESS_ENTER: u32 = 5;
pub const EVENT_KEYBOARD_OPEN: u32 = 256;
/// SceImeCaretMovementDirection.
const CARET_MOVE_LEFT: u32 = 1;
const CARET_MOVE_RIGHT: u32 = 2;
/// SceImeTextAreaMode.
const TEXT_AREA_MODE_EDIT: u32 = 1;
const TEXT_AREA_MODE_PREEDIT: u32 = 2;
const TEXT_AREA_MODE_SELECT: u32 = 3;

#[derive(Clone, Copy, Debug, Default)]
struct Param {
    user_id: i32,
    ty: u32,
    supported_languages: u64,
    enter_label: u32,
    input_method: u32,
    filter: u64,
    option: u32,
    max_text_length: u32,
    input_text_buffer: u64,
    posx: f32,
    posy: f32,
    horizontal_alignment: u32,
    vertical_alignment: u32,
    work: u64,
    arg: u64,
    handler: u64,
    reserved_zero: bool,
}

impl Param {
    fn parse(raw: &[u8]) -> Self {
        let u32_at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(raw[o..o + 8].try_into().unwrap());
        Self {
            user_id: u32_at(0) as i32,
            ty: u32_at(0x04),
            supported_languages: u64_at(0x08),
            enter_label: u32_at(0x10),
            input_method: u32_at(0x14),
            filter: u64_at(0x18),
            option: u32_at(0x20),
            max_text_length: u32_at(0x24),
            input_text_buffer: u64_at(0x28),
            posx: f32::from_bits(u32_at(0x30)),
            posy: f32::from_bits(u32_at(0x34)),
            horizontal_alignment: u32_at(0x38),
            vertical_alignment: u32_at(0x3C),
            work: u64_at(0x40),
            arg: u64_at(0x48),
            handler: u64_at(0x50),
            reserved_zero: raw[0x58..0x60].iter().all(|&b| b == 0),
        }
    }
}

/// An event waiting for sceImeUpdate. Text events carry the text that is
/// synced into the guest buffers right before dispatch.
#[derive(Clone, Debug)]
struct QueuedEvent {
    bytes: [u8; EVENT_SIZE],
    text: Option<Vec<u16>>,
}

impl QueuedEvent {
    fn new(id: u32) -> Self {
        let mut bytes = [0u8; EVENT_SIZE];
        bytes[0..4].copy_from_slice(&id.to_le_bytes());
        Self { bytes, text: None }
    }

    fn put_u32(&mut self, param_off: usize, v: u32) {
        self.bytes[8 + param_off..12 + param_off].copy_from_slice(&v.to_le_bytes());
    }

    fn put_u64(&mut self, param_off: usize, v: u64) {
        self.bytes[8 + param_off..16 + param_off].copy_from_slice(&v.to_le_bytes());
    }
}

#[derive(Default)]
struct State {
    open: bool,
    event_overflow: bool,
    generation: u64,
    revision: u64,
    param: Param,
    extended: ExtendedParam,
    editor: TextEditEngine,
    events: VecDeque<QueuedEvent>,
    external_inputs: Vec<ExternalInput>,
}

#[derive(Default)]
struct KeyboardState {
    open: bool,
    arg: u64,
    handler: u64,
    events: VecDeque<QueuedEvent>,
}

#[derive(Default)]
struct Shared {
    ime: State,
    keyboard: KeyboardState,
}

static SHARED: LazyLock<Mutex<Shared>> = LazyLock::new(|| Mutex::new(Shared::default()));
static OPEN: AtomicBool = AtomicBool::new(false);
static REVISION: AtomicU64 = AtomicU64::new(0);
static VISIBILITY: RwLock<Option<VisibilityCallback>> = RwLock::new(None);

fn shared() -> MutexGuard<'static, Shared> {
    SHARED.lock().unwrap_or_else(|p| p.into_inner())
}

fn notify_visibility(visible: bool, generation: u64) {
    let cb = *VISIBILITY.read().unwrap_or_else(|p| p.into_inner());
    if let Some(cb) = cb {
        cb(visible, generation);
    }
}

/// Valid prefix length of `text[..length]` up to the first NUL; None when a
/// line break (single-line) or broken surrogate pair is found.
fn validate_text(text: &[u16], multiline: bool) -> Option<usize> {
    let mut actual = 0;
    let mut i = 0;
    while i < text.len() {
        let v = text[i];
        if v == 0 {
            return Some(actual);
        }
        if !multiline && (v == 0x0A || v == 0x0D) {
            return None;
        }
        if (0xD800..=0xDBFF).contains(&v) {
            i += 1;
            if i >= text.len() || !(0xDC00..=0xDFFF).contains(&text[i]) {
                return None;
            }
        } else if (0xDC00..=0xDFFF).contains(&v) {
            return None;
        }
        actual = i + 1;
        i += 1;
    }
    Some(actual)
}

fn update_revision(s: &mut State) {
    s.revision += 1;
    REVISION.store(s.revision, Ordering::Release);
}

fn sync_guest_text(ctx: &CpuContext, work: u64, input: u64, max_length: u32, text: &[u16]) {
    if work != 0 {
        common::write_units_nul(ctx, work, text, max_length);
    }
    if input != 0 {
        common::write_units_nul(ctx, input, text, max_length);
    }
}

fn queue_event(s: &mut State, e: QueuedEvent) -> bool {
    if s.events.len() == EVENT_QUEUE_CAPACITY {
        s.event_overflow = true;
        return false;
    }
    s.events.push_back(e);
    true
}

/// SceImeEditText event: str @0, caretIndex @8, areaNum @0xC, textArea[0] @0x10.
fn make_text_event(s: &State, id: u32, edit_index: u32, edit_length: i32) -> QueuedEvent {
    let mut e = QueuedEvent::new(id);
    e.put_u64(0, s.param.work);
    e.put_u32(0x08, s.editor.cursor());
    e.put_u32(0x0C, 1);
    e.put_u32(0x10, TEXT_AREA_MODE_EDIT);
    e.put_u32(0x14, edit_index);
    e.put_u32(0x18, edit_length as u32);
    e.text = Some(s.editor.text().to_vec());
    e
}

fn panel_size(ty: u32, option: u32) -> (u32, u32) {
    let (w, h) = if ty == TYPE_NUMBER { (370, 402) } else { (793, 408) };
    if option & OPTION_USE_OVER_2K != 0 { (w * 2, h * 2) } else { (w, h) }
}

fn ranges_overlap(first: u64, first_size: u64, second: u64, second_size: u64) -> bool {
    if first.checked_add(first_size).is_none() || second.checked_add(second_size).is_none() {
        return true;
    }
    first < second + second_size && second < first + first_size
}

fn validate_open(ctx: &CpuContext, param: u64, extended: u64) -> Result<(Param, Option<ExtendedParam>, Vec<u16>), i32> {
    if param == 0 {
        return Err(ERROR_INVALID_ADDRESS);
    }
    let p = Param::parse(&ctx.read_bytes(param, PARAM_SIZE).ok_or(ERROR_INVALID_ADDRESS)?);
    if p.user_id < 0 || p.user_id == 0xFF {
        return Err(ERROR_INVALID_USER_ID);
    }
    if p.ty > TYPE_NUMBER {
        return Err(ERROR_INVALID_TYPE);
    }
    if p.supported_languages & !VALID_LANGUAGES != 0 {
        return Err(ERROR_INVALID_LANGUAGES);
    }
    if p.enter_label > ENTER_LABEL_GO {
        return Err(ERROR_INVALID_ENTER_LABEL);
    }
    if p.input_method != 0 {
        return Err(ERROR_INVALID_INPUT_METHOD);
    }
    if p.option & !VALID_OPTIONS != 0 {
        return Err(ERROR_INVALID_OPTION);
    }
    let multiline = p.option & OPTION_MULTILINE != 0;
    let password = p.option & OPTION_PASSWORD != 0;
    if (multiline && password)
        || (multiline && p.ty != TYPE_DEFAULT && p.ty != TYPE_BASIC_LATIN)
        || (password && p.ty != TYPE_BASIC_LATIN && p.ty != TYPE_NUMBER)
    {
        return Err(ERROR_INVALID_PARAM);
    }
    if p.max_text_length == 0 || p.max_text_length > MAX_TEXT_LENGTH {
        return Err(ERROR_INVALID_MAX_TEXT_LENGTH);
    }
    if p.input_text_buffer == 0 {
        return Err(ERROR_INVALID_TEXT_BUFFER);
    }
    let (max_x, max_y) = if p.option & OPTION_USE_OVER_2K != 0 { (3840.0, 2160.0) } else { (1920.0, 1080.0) };
    if !p.posx.is_finite() || p.posx < 0.0 || p.posx >= max_x {
        return Err(ERROR_INVALID_POSX);
    }
    if !p.posy.is_finite() || p.posy < 0.0 || p.posy >= max_y {
        return Err(ERROR_INVALID_POSY);
    }
    if p.horizontal_alignment > ALIGNMENT_END {
        return Err(ERROR_INVALID_HALIGN);
    }
    if p.vertical_alignment > ALIGNMENT_END {
        return Err(ERROR_INVALID_VALIGN);
    }
    let ext = common::read_extended(ctx, extended).map_err(|_| ERROR_INVALID_ADDRESS)?;
    if ext.is_some_and(|e| !e.is_valid(VALID_EXTENDED_OPTIONS)) {
        return Err(ERROR_INVALID_EXTENDED);
    }
    if p.work == 0 || p.work & 3 != 0 {
        return Err(ERROR_INVALID_WORK);
    }
    if p.handler == 0 {
        return Err(ERROR_INVALID_HANDLER);
    }
    if !p.reserved_zero {
        return Err(ERROR_INVALID_RESERVED);
    }
    let preedit = if p.option & OPTION_EXPANDED_PREEDIT != 0 { 0x79 } else { 0x1F };
    let text_size = (p.max_text_length as u64 + preedit) * 2;
    if ranges_overlap(p.input_text_buffer, text_size, p.work, WORK_BUFFER_SIZE as u64) {
        return Err(ERROR_INVALID_PARAM);
    }
    // Initial text: NUL-terminated within maxTextLength units.
    let units = common::scan_units(ctx, p.input_text_buffer, p.max_text_length as usize + 1).ok_or(ERROR_INVALID_TEXT)?;
    if units.len() > p.max_text_length as usize {
        return Err(ERROR_INVALID_TEXT);
    }
    let actual = validate_text(&units, multiline).ok_or(ERROR_INVALID_TEXT)?;
    Ok((p, ext, units[..actual].to_vec()))
}

// ---------------------------------------------------------------------------
// Guest API
// ---------------------------------------------------------------------------

pub fn param_init(ctx: &CpuContext, param: u64) {
    if param == 0 {
        return;
    }
    let mut b = [0u8; PARAM_SIZE];
    b[0..4].copy_from_slice(&(-1i32).to_le_bytes());
    ctx.write_bytes(param, &b);
}

pub fn get_panel_size(ctx: &CpuContext, param: u64, width: u64, height: u64) -> i32 {
    if param == 0 || width == 0 || height == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let Some(raw) = ctx.read_bytes(param, PARAM_SIZE) else {
        return ERROR_INVALID_ADDRESS;
    };
    let p = Param::parse(&raw);
    if p.ty > TYPE_NUMBER {
        return ERROR_INVALID_TYPE;
    }
    if p.option & !VALID_OPTIONS != 0 {
        return ERROR_INVALID_OPTION;
    }
    let (w, h) = panel_size(p.ty, p.option);
    if ctx.write_u32(width, w) && ctx.write_u32(height, h) { 0 } else { ERROR_INVALID_ADDRESS }
}

pub fn open(ctx: &CpuContext, param: u64, extended: u64) -> i32 {
    let generation = {
        let mut sh = shared();
        if sh.ime.open {
            return ERROR_BUSY;
        }
        let (p, ext, text) = match validate_open(ctx, param, extended) {
            Ok(v) => v,
            Err(e) => return e,
        };
        let s = &mut sh.ime;
        let (generation, revision) = (s.generation + 1, s.revision + 1);
        *s = State::default();
        s.open = true;
        s.generation = generation;
        s.revision = revision;
        s.param = p;
        s.extended = ext.unwrap_or_default();
        s.editor.reset(p.ty, p.option, p.max_text_length, text);
        ctx.write_bytes(p.work, &vec![0u8; WORK_BUFFER_SIZE as usize]);
        sync_guest_text(ctx, p.work, p.input_text_buffer, s.editor.max_length(), s.editor.text());

        // SceImeRect { x, y, width, height }.
        let mut e = QueuedEvent::new(EVENT_OPEN);
        let (w, h) = panel_size(p.ty, p.option);
        e.put_u32(0, p.posx.to_bits());
        e.put_u32(4, p.posy.to_bits());
        e.put_u32(8, w);
        e.put_u32(12, h);
        queue_event(s, e);
        OPEN.store(true, Ordering::Release);
        REVISION.store(revision, Ordering::Release);
        generation
    };
    notify_visibility(true, generation);
    0
}

struct Dispatch {
    arg: u64,
    event: QueuedEvent,
    work: u64,
    input: u64,
    max_length: u32,
    generation: u64,
}

pub fn update(ctx: &CpuContext, handler: u64) -> i32 {
    let selected = |sh: &Shared| -> Result<(bool, bool), i32> {
        if !sh.ime.open && !sh.keyboard.open {
            return Err(ERROR_NOT_OPENED);
        }
        let ime = sh.ime.open && handler == sh.ime.param.handler;
        let keyboard = sh.keyboard.open && handler == sh.keyboard.handler;
        if !ime && !keyboard {
            return Err(ERROR_NOT_OPENED);
        }
        Ok((ime, keyboard))
    };
    let apply_inputs = match selected(&shared()) {
        Ok((ime, _)) => ime,
        Err(e) => return e,
    };
    if apply_inputs {
        apply_external_inputs(ctx);
    }
    let mut dispatch = Vec::new();
    {
        let mut sh = shared();
        let (ime, keyboard) = match selected(&sh) {
            Ok(v) => v,
            Err(e) => return e,
        };
        if ime && std::mem::take(&mut sh.ime.event_overflow) {
            return ERROR_EVENT_OVERFLOW;
        }
        if ime {
            let s = &mut sh.ime;
            while let Some(event) = s.events.pop_front() {
                dispatch.push(Dispatch {
                    arg: s.param.arg,
                    event,
                    work: s.param.work,
                    input: s.param.input_text_buffer,
                    max_length: s.editor.max_length(),
                    generation: s.generation,
                });
            }
        }
        if keyboard {
            let k = &mut sh.keyboard;
            while let Some(event) = k.events.pop_front() {
                dispatch.push(Dispatch { arg: k.arg, event, work: 0, input: 0, max_length: 0, generation: 0 });
            }
        }
    }
    if dispatch.is_empty() {
        return 0;
    }
    let Some(scratch) = Scratch::acquire(ctx) else {
        return 0;
    };
    let event_addr = scratch.0 + common::SCRATCH_EVENT;
    for mut item in dispatch {
        if item.generation != 0 {
            let sh = shared();
            if !sh.ime.open || sh.ime.generation != item.generation {
                continue;
            }
        }
        if let Some(text) = &item.event.text {
            sync_guest_text(ctx, item.work, item.input, item.max_length, text);
            item.event.put_u64(0, item.work);
        }
        if ctx.write_bytes(event_addr, &item.event.bytes) {
            common::guest_call(handler, [item.arg, event_addr, 0, 0, 0, 0]);
        }
    }
    0
}

pub fn close() -> i32 {
    let generation = {
        let mut sh = shared();
        let s = &mut sh.ime;
        if !s.open {
            return ERROR_NOT_OPENED;
        }
        let (generation, revision) = (s.generation, s.revision + 1);
        *s = State::default();
        s.generation = generation;
        s.revision = revision;
        OPEN.store(false, Ordering::Release);
        REVISION.store(revision, Ordering::Release);
        generation
    };
    notify_visibility(false, generation);
    0
}

pub fn set_text(ctx: &CpuContext, text: u64, length: u32) -> i32 {
    let mut sh = shared();
    let s = &mut sh.ime;
    if !s.open {
        return ERROR_NOT_OPENED;
    }
    if text == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let clamped = length.min(s.editor.max_length());
    let Some(units) = common::read_units(ctx, text, clamped as usize) else {
        return ERROR_INVALID_ADDRESS;
    };
    let Some(actual) = validate_text(&units, s.editor.option() & OPTION_MULTILINE != 0) else {
        return ERROR_INVALID_TEXT;
    };
    let cursor = s.editor.cursor();
    s.editor.replace_text(units[..actual].to_vec(), cursor);
    sync_guest_text(ctx, s.param.work, s.param.input_text_buffer, s.editor.max_length(), s.editor.text());
    update_revision(s);
    0
}

/// sceImeGetPanelPositionAndForm(SceImePositionAndForm*): type, posx, posy,
/// horizontal / vertical alignment, width, height of the open panel. A
/// closed panel is reported before a null output, and the output is then
/// left untouched.
pub fn get_panel_position_and_form(ctx: &CpuContext, form: u64) -> i32 {
    let sh = shared();
    let s = &sh.ime;
    if !s.open {
        return ERROR_NOT_OPENED;
    }
    if form == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let (w, h) = panel_size(s.param.ty, s.editor.option());
    let fields = [
        crate::ime_dialog::POSITION_AND_FORM_PANEL_TYPE,
        s.param.posx.to_bits(),
        s.param.posy.to_bits(),
        s.param.horizontal_alignment,
        s.param.vertical_alignment,
        w,
        h,
    ];
    let b: Vec<u8> = fields.iter().flat_map(|v| v.to_le_bytes()).collect();
    if ctx.write_bytes(form, &b) { 0 } else { ERROR_INVALID_ADDRESS }
}

/// sceImeSetCaret(const SceImeCaret*): index @0xC.
pub fn set_caret(ctx: &CpuContext, caret: u64) -> i32 {
    let mut sh = shared();
    let s = &mut sh.ime;
    if !s.open {
        return ERROR_NOT_OPENED;
    }
    if caret == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let Some(index) = ctx.read_u32(caret + 0xC) else {
        return ERROR_INVALID_ADDRESS;
    };
    if index as usize > s.editor.text().len() {
        return ERROR_INVALID_PARAM;
    }
    s.editor.set_cursor(index);
    update_revision(s);
    0
}

/// sceImeSetTextGeometry(mode, const SceImeTextGeometry*): x @0, y @4.
pub fn set_text_geometry(ctx: &CpuContext, mode: u32, geometry: u64) -> i32 {
    let sh = shared();
    if !sh.ime.open {
        return ERROR_NOT_OPENED;
    }
    if geometry == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let (Some(x), Some(y)) = (ctx.read_f32(geometry), ctx.read_f32(geometry + 4)) else {
        return ERROR_INVALID_ADDRESS;
    };
    let (max_x, max_y) =
        if sh.ime.editor.option() & OPTION_USE_OVER_2K != 0 { (3840.0, 2160.0) } else { (1920.0, 1080.0) };
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || x >= max_x
        || y < 0.0
        || y >= max_y
        || (mode != TEXT_AREA_MODE_PREEDIT && mode != TEXT_AREA_MODE_SELECT)
    {
        return ERROR_INVALID_PARAM;
    }
    0
}

/// sceImeKeyboardOpen(userId, const SceImeKeyboardParam*): option @0,
/// reserved1[4] @4, arg @8, handler @0x10, reserved2[8] @0x18.
pub fn keyboard_open(ctx: &CpuContext, user_id: i32, param: u64) -> i32 {
    let mut sh = shared();
    if param == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let Some(raw) = ctx.read_bytes(param, KEYBOARD_PARAM_SIZE) else {
        return ERROR_INVALID_ADDRESS;
    };
    if user_id == -1 {
        return ERROR_INVALID_USER_ID;
    }
    let option = u32::from_le_bytes(raw[0..4].try_into().unwrap());
    let arg = u64::from_le_bytes(raw[8..16].try_into().unwrap());
    let handler = u64::from_le_bytes(raw[16..24].try_into().unwrap());
    if handler == 0 {
        return ERROR_INVALID_HANDLER;
    }
    if option & !VALID_KEYBOARD_OPTIONS != 0 {
        return ERROR_INVALID_OPTION;
    }
    if raw[4..8].iter().chain(&raw[24..32]).any(|&b| b != 0) {
        return ERROR_INVALID_RESERVED;
    }
    if sh.keyboard.open {
        return ERROR_BUSY;
    }
    sh.keyboard = KeyboardState { open: true, arg, handler, events: VecDeque::new() };
    let mut e = QueuedEvent::new(EVENT_KEYBOARD_OPEN);
    e.put_u32(0, user_id as u32);
    sh.keyboard.events.push_back(e);
    0
}

pub fn keyboard_close(user_id: i32) -> i32 {
    let mut sh = shared();
    if !sh.keyboard.open {
        return ERROR_NOT_OPENED;
    }
    if user_id == -1 {
        return ERROR_INVALID_USER_ID;
    }
    sh.keyboard = KeyboardState::default();
    0
}

/// No keyboard ever connects: the ids are cleared and the call fails.
pub fn keyboard_get_resource_id(ctx: &CpuContext, user_id: i32, out: u64) -> i32 {
    if out == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    if user_id == -1 {
        return ERROR_INVALID_USER_ID;
    }
    let mut arr = [0u8; KEYBOARD_RESOURCE_ID_ARRAY_SIZE];
    arr[..4].copy_from_slice(&user_id.to_le_bytes());
    if !ctx.write_bytes(out, &arr) {
        return ERROR_INVALID_ADDRESS;
    }
    if shared().keyboard.open { ERROR_CONNECTION_FAILED } else { ERROR_NOT_OPENED }
}

pub fn keyboard_get_info(info: u64) -> i32 {
    if info == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    if !shared().keyboard.open {
        return ERROR_NOT_OPENED;
    }
    ERROR_NO_RESOURCE_ID
}

pub fn keyboard_set_mode(user_id: i32, mode: u32) -> i32 {
    if user_id == -1 {
        return ERROR_INVALID_USER_ID;
    }
    if !shared().keyboard.open {
        return ERROR_NOT_OPENED;
    }
    if mode & !VALID_KEYBOARD_MODE == 0 { 0 } else { ERROR_INVALID_MODE }
}

// ---------------------------------------------------------------------------
// Host input application
// ---------------------------------------------------------------------------

fn commit_filtered_text(
    ctx: &CpuContext,
    generation: u64,
    revision: u64,
    old_text: Vec<u16>,
    mut candidate: TextEditEngine,
    filter: u64,
    multiline: bool,
) -> bool {
    if filter != 0 {
        if let Some(filtered) = common::run_text_filter(ctx, filter, candidate.text(), candidate.max_length()) {
            if common::is_valid_input_text(&filtered, multiline) {
                let cursor = candidate.cursor();
                candidate.replace_text(filtered, cursor);
            }
        }
    }
    let mut sh = shared();
    let s = &mut sh.ime;
    if !s.open || s.generation != generation || s.revision != revision || s.editor.text() != old_text.as_slice()
        || candidate.text() == old_text.as_slice()
    {
        return false;
    }
    let edit = common::compute_edit_delta(&old_text, candidate.text());
    s.editor = candidate;
    let e = make_text_event(s, EVENT_UPDATE_TEXT, edit.index, edit.length);
    queue_event(s, e);
    update_revision(s);
    true
}

fn apply_edit(ctx: &CpuContext, generation: u64, edit: impl FnOnce(&mut TextEditEngine, bool) -> bool) -> bool {
    let (revision, multiline, filter, old_text, candidate) = {
        let sh = shared();
        let s = &sh.ime;
        if !s.open || s.generation != generation {
            return false;
        }
        let multiline = s.editor.option() & OPTION_MULTILINE != 0;
        let mut candidate = s.editor.clone();
        if !edit(&mut candidate, multiline) {
            return false;
        }
        (s.revision, multiline, s.param.filter, s.editor.text().to_vec(), candidate)
    };
    commit_filtered_text(ctx, generation, revision, old_text, candidate, filter, multiline)
}

fn apply_insert_text(ctx: &CpuContext, generation: u64, text: &[u16]) -> bool {
    if text.is_empty() {
        return false;
    }
    apply_edit(ctx, generation, |c, multiline| common::is_valid_input_text(text, multiline) && c.insert(text).is_some())
}

fn apply_backspace(ctx: &CpuContext, generation: u64) -> bool {
    apply_edit(ctx, generation, |c, _| c.backspace().is_some())
}

fn apply_move_cursor(generation: u64, delta: i32) -> bool {
    let mut sh = shared();
    let s = &mut sh.ime;
    if !s.open || s.generation != generation || delta == 0 {
        return false;
    }
    let old = s.editor.cursor();
    if !s.editor.move_cursor(delta) {
        return false;
    }
    let new = s.editor.cursor();
    let direction = if new > old { CARET_MOVE_RIGHT } else { CARET_MOVE_LEFT };
    for _ in 0..new.abs_diff(old) {
        let mut e = QueuedEvent::new(EVENT_UPDATE_CARET);
        e.put_u32(0, direction);
        queue_event(s, e);
    }
    update_revision(s);
    true
}

fn apply_press(generation: u64, id: u32) -> bool {
    let mut sh = shared();
    let s = &mut sh.ime;
    if !s.open || s.generation != generation {
        return false;
    }
    queue_event(s, QueuedEvent::new(id))
}

fn apply_external_inputs(ctx: &CpuContext) {
    let (generation, filter, user_id, multiline, inputs) = {
        let mut sh = shared();
        let s = &mut sh.ime;
        if !s.open || s.external_inputs.is_empty() {
            return;
        }
        (
            s.generation,
            s.extended.ext_keyboard_filter,
            s.param.user_id,
            s.editor.option() & OPTION_MULTILINE != 0,
            std::mem::take(&mut s.external_inputs),
        )
    };
    for mut input in inputs {
        {
            let sh = shared();
            if !sh.ime.open || sh.ime.generation != generation {
                break;
            }
        }
        input.key.user_id = user_id;
        let mut accepted = true;
        if filter != 0 && input.key.status != 0 {
            if let Some((0, keycode, status)) = common::run_keyboard_filter(ctx, filter, &input.key) {
                accepted = common::apply_keyboard_filter_output(&mut input, keycode, status, multiline);
            }
        }
        if !accepted {
            continue;
        }
        match input.action {
            ExternalAction::None => {}
            ExternalAction::Text => {
                apply_insert_text(ctx, generation, &input.text);
            }
            ExternalAction::Backspace => {
                apply_backspace(ctx, generation);
            }
            ExternalAction::MoveLeft => {
                apply_move_cursor(generation, -1);
            }
            ExternalAction::MoveRight => {
                apply_move_cursor(generation, 1);
            }
            ExternalAction::Cancel => {
                apply_press(generation, EVENT_PRESS_CLOSE);
            }
            ExternalAction::Accept => {
                apply_press(generation, EVENT_PRESS_ENTER);
            }
            ExternalAction::Newline => {
                apply_insert_text(ctx, generation, &[0x0A]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host API (UI frontends)
// ---------------------------------------------------------------------------

pub fn visual_state() -> VisualState {
    VisualState { active: OPEN.load(Ordering::Acquire), revision: REVISION.load(Ordering::Acquire) }
}

pub fn set_visibility_callback(callback: Option<VisibilityCallback>) {
    *VISIBILITY.write().unwrap_or_else(|p| p.into_inner()) = callback;
}

pub fn host_snapshot() -> Option<HostSnapshot> {
    let sh = shared();
    let s = &sh.ime;
    if !s.open {
        return None;
    }
    let (panel_width, panel_height) = panel_size(s.param.ty, s.editor.option());
    Some(HostSnapshot {
        generation: s.generation,
        ty: s.param.ty,
        enter_label: s.param.enter_label,
        option: s.editor.option(),
        max_text_length: s.editor.max_length(),
        cursor: s.editor.cursor(),
        disable_device: s.extended.disable_device,
        key_panel_visible: s.editor.option() & OPTION_EXT_KEYBOARD == 0
            || s.extended.option & common::EXT_OPTION_HIDE_KEY_PANEL == 0,
        posx: s.param.posx,
        posy: s.param.posy,
        horizontal_alignment: s.param.horizontal_alignment,
        vertical_alignment: s.param.vertical_alignment,
        panel_width,
        panel_height,
        text: s.editor.text().to_vec(),
        title: Vec::new(),
        placeholder: Vec::new(),
    })
}

pub fn host_queue_external_input(generation: u64, input: ExternalInput) -> bool {
    let mut sh = shared();
    let s = &mut sh.ime;
    if !s.open || s.generation != generation {
        return false;
    }
    if s.external_inputs.len() == EVENT_QUEUE_CAPACITY {
        s.event_overflow = true;
        return false;
    }
    s.external_inputs.push(input);
    true
}

fn host_action(generation: u64, action: ExternalAction, text: Vec<u16>) -> bool {
    host_queue_external_input(generation, ExternalInput { action, text, ..Default::default() })
}

pub fn host_insert_text(generation: u64, text: &[u16]) -> bool {
    host_action(generation, ExternalAction::Text, text.to_vec())
}

pub fn host_backspace(generation: u64) -> bool {
    host_action(generation, ExternalAction::Backspace, Vec::new())
}

pub fn host_accept(generation: u64) -> bool {
    host_action(generation, ExternalAction::Accept, Vec::new())
}

pub fn host_cancel(generation: u64) -> bool {
    host_action(generation, ExternalAction::Cancel, Vec::new())
}

// ---------------------------------------------------------------------------
// HLE callbacks
// ---------------------------------------------------------------------------

fn r(v: i32) -> HleResult {
    v as HleResult
}

fn cb_set_text(ctx: &mut CpuContext) -> HleResult {
    r(set_text(ctx, ctx.rdi(), ctx.rsi() as u32))
}

fn cb_set_caret(ctx: &mut CpuContext) -> HleResult {
    r(set_caret(ctx, ctx.rdi()))
}

fn cb_get_panel_position_and_form(ctx: &mut CpuContext) -> HleResult {
    r(get_panel_position_and_form(ctx, ctx.rdi()))
}

fn cb_param_init(ctx: &mut CpuContext) -> HleResult {
    param_init(ctx, ctx.rdi());
    0
}

fn cb_get_panel_size(ctx: &mut CpuContext) -> HleResult {
    r(get_panel_size(ctx, ctx.rdi(), ctx.rsi(), ctx.rdx()))
}

fn cb_open(ctx: &mut CpuContext) -> HleResult {
    r(open(ctx, ctx.rdi(), ctx.rsi()))
}

fn cb_update(ctx: &mut CpuContext) -> HleResult {
    r(update(ctx, ctx.rdi()))
}

fn cb_close(_ctx: &mut CpuContext) -> HleResult {
    r(close())
}

fn cb_set_text_geometry(ctx: &mut CpuContext) -> HleResult {
    r(set_text_geometry(ctx, ctx.rdi() as u32, ctx.rsi()))
}

fn cb_keyboard_open(ctx: &mut CpuContext) -> HleResult {
    r(keyboard_open(ctx, ctx.rdi() as i32, ctx.rsi()))
}

fn cb_keyboard_close(ctx: &mut CpuContext) -> HleResult {
    r(keyboard_close(ctx.rdi() as i32))
}

fn cb_keyboard_get_resource_id(ctx: &mut CpuContext) -> HleResult {
    r(keyboard_get_resource_id(ctx, ctx.rdi() as i32, ctx.rsi()))
}

fn cb_keyboard_get_info(ctx: &mut CpuContext) -> HleResult {
    r(keyboard_get_info(ctx.rsi()))
}

fn cb_keyboard_set_mode(ctx: &mut CpuContext) -> HleResult {
    r(keyboard_set_mode(ctx.rdi() as i32, ctx.rsi() as u32))
}

const LIB: &str = "libSceIme";

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("ieCNrVrzKd4", "sceImeSetText", cb_set_text),
    ("WLxUN2WMim8", "sceImeSetCaret", cb_set_caret),
    ("T6FYjZXG93o", "sceImeGetPanelPositionAndForm", cb_get_panel_position_and_form),
    ("WmYDzdC4EHI", "sceImeParamInit", cb_param_init),
    ("ziPDcIjO0Vk", "sceImeGetPanelSize", cb_get_panel_size),
    ("RPydv-Jr1bc", "sceImeOpen", cb_open),
    ("-4GCfYdNF1s", "sceImeUpdate", cb_update),
    ("TmVP8LzcFcY", "sceImeClose", cb_close),
    ("TXYHFRuL8UY", "sceImeSetTextGeometry", cb_set_text_geometry),
    ("eaFXjfJv3xs", "sceImeKeyboardOpen", cb_keyboard_open),
    ("PMVehSlfZ94", "sceImeKeyboardClose", cb_keyboard_close),
    ("dKadqZFgKKQ", "sceImeKeyboardGetResourceId", cb_keyboard_get_resource_id),
    ("VkqLPArfFdc", "sceImeKeyboardGetInfo", cb_keyboard_get_info),
    ("ua+13Hk9kKs", "sceImeKeyboardSetMode", cb_keyboard_set_mode),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
    crate::ime_dialog::register(reg);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ime_common::{test_support, utf16};

    const BASE: u64 = 0x5D31_0000_0000;
    const PARAM: u64 = BASE;
    const OUT: u64 = BASE + 0x200;
    const TEXT: u64 = BASE + 0x1000;
    const WORK: u64 = BASE + 0x2000;
    const HANDLER: u64 = BASE + 0xF_0001;
    const KB_HANDLER: u64 = BASE + 0xF_0002;
    const UPPER_FILTER: u64 = BASE + 0xF_0003;

    static EVENTS: Mutex<Vec<(u64, u32, Vec<u8>)>> = Mutex::new(Vec::new());
    static LOCK: Mutex<()> = Mutex::new(());

    fn record(ctx: &CpuContext, a: [u64; 6]) -> u64 {
        let ev = ctx.read_bytes(a[1], EVENT_SIZE).unwrap();
        let id = u32::from_le_bytes(ev[0..4].try_into().unwrap());
        EVENTS.lock().unwrap().push((a[0], id, ev));
        0
    }

    /// Uppercases ASCII letters.
    fn upper_filter(ctx: &CpuContext, a: [u64; 6]) -> u64 {
        let src = common::read_units(ctx, a[2], a[3] as usize).unwrap();
        let out: Vec<u16> = src.iter().map(|&c| if (b'a' as u16..=b'z' as u16).contains(&c) { c - 32 } else { c }).collect();
        common::write_units_nul(ctx, a[0], &out, 2048);
        ctx.write_u32(a[1], out.len() as u32);
        0
    }

    fn take_events() -> Vec<(u64, u32, Vec<u8>)> {
        std::mem::take(&mut *EVENTS.lock().unwrap())
    }

    fn write_param(ctx: &CpuContext, max: u32, filter: u64, work: u64) {
        let mut b = [0u8; PARAM_SIZE];
        b[0..4].copy_from_slice(&1000i32.to_le_bytes());
        b[0x08..0x10].copy_from_slice(&1u64.to_le_bytes());
        b[0x18..0x20].copy_from_slice(&filter.to_le_bytes());
        b[0x24..0x28].copy_from_slice(&max.to_le_bytes());
        b[0x28..0x30].copy_from_slice(&TEXT.to_le_bytes());
        b[0x40..0x48].copy_from_slice(&work.to_le_bytes());
        b[0x48..0x50].copy_from_slice(&0x77u64.to_le_bytes());
        b[0x50..0x58].copy_from_slice(&HANDLER.to_le_bytes());
        ctx.write_bytes(PARAM, &b);
    }

    #[test]
    fn ime_session_events_and_filter() {
        let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        test_support::register_fake(HANDLER, record);
        test_support::register_fake(KB_HANDLER, record);
        test_support::register_fake(UPPER_FILTER, upper_filter);
        let ctx = CpuContext::new(&vm);
        take_events();

        param_init(&ctx, PARAM);
        assert_eq!(ctx.read_u32(PARAM), Some(u32::MAX));
        assert_eq!(open(&ctx, PARAM, 0), ERROR_INVALID_USER_ID);
        assert_eq!(update(&ctx, HANDLER), ERROR_NOT_OPENED);
        // The panel position needs an open panel, checked before the
        // pointer; nothing is written.
        let form = OUT + 0x100;
        ctx.write_bytes(form, &[0xA5; 0x1C]);
        assert_eq!(get_panel_position_and_form(&ctx, form), ERROR_NOT_OPENED);
        assert_eq!(get_panel_position_and_form(&ctx, 0), ERROR_NOT_OPENED);
        assert_eq!(ctx.read_bytes(form, 0x1C).unwrap(), vec![0xA5; 0x1C]);

        common::write_units_nul(&ctx, TEXT, &utf16("ab"), 16);
        write_param(&ctx, 16, UPPER_FILTER, WORK + 2);
        assert_eq!(open(&ctx, PARAM, 0), ERROR_INVALID_WORK);
        write_param(&ctx, 16, UPPER_FILTER, TEXT + 8);
        assert_eq!(open(&ctx, PARAM, 0), ERROR_INVALID_PARAM);
        write_param(&ctx, 16, UPPER_FILTER, WORK);
        assert_eq!(open(&ctx, PARAM, 0), 0);
        assert_eq!(open(&ctx, PARAM, 0), ERROR_BUSY);
        assert_eq!(common::scan_units(&ctx, WORK, 32).unwrap(), utf16("ab"));
        assert_eq!(update(&ctx, KB_HANDLER), ERROR_NOT_OPENED);
        assert_eq!(update(&ctx, HANDLER), 0);
        let ev = take_events();
        assert_eq!(ev.len(), 1);
        assert_eq!((ev[0].0, ev[0].1), (0x77, EVENT_OPEN));
        assert_eq!(u32::from_le_bytes(ev[0].2[16..20].try_into().unwrap()), 793);
        assert_eq!(u32::from_le_bytes(ev[0].2[20..24].try_into().unwrap()), 408);
        assert_eq!(get_panel_position_and_form(&ctx, 0), ERROR_INVALID_ADDRESS);
        assert_eq!(get_panel_position_and_form(&ctx, form), 0);
        let f: Vec<u32> = (0..7).map(|i| ctx.read_u32(form + i * 4).unwrap()).collect();
        assert_eq!(f, vec![crate::ime_dialog::POSITION_AND_FORM_PANEL_TYPE, 0, 0, 0, 0, 793, 408]);

        let g = host_snapshot().unwrap().generation;
        assert!(host_insert_text(g, &utf16("c")));
        assert_eq!(update(&ctx, HANDLER), 0);
        let ev = take_events();
        assert_eq!(ev.len(), 1);
        assert_eq!(ev[0].1, EVENT_UPDATE_TEXT);
        let p = &ev[0].2[8..];
        assert_eq!(u64::from_le_bytes(p[0..8].try_into().unwrap()), WORK);
        assert_eq!(u32::from_le_bytes(p[8..12].try_into().unwrap()), 3); // caret
        assert_eq!(u32::from_le_bytes(p[12..16].try_into().unwrap()), 1); // areaNum
        // The filter uppercased everything: edit covers the whole text.
        assert_eq!(u32::from_le_bytes(p[0x14..0x18].try_into().unwrap()), 0);
        assert_eq!(common::scan_units(&ctx, WORK, 32).unwrap(), utf16("ABC"));
        assert_eq!(common::scan_units(&ctx, TEXT, 32).unwrap(), utf16("ABC"));

        assert!(host_queue_external_input(g, ExternalInput { action: ExternalAction::MoveLeft, ..Default::default() }));
        assert!(host_accept(g));
        assert_eq!(update(&ctx, HANDLER), 0);
        let ids: Vec<u32> = take_events().iter().map(|e| e.1).collect();
        assert_eq!(ids, vec![EVENT_UPDATE_CARET, EVENT_PRESS_ENTER]);

        ctx.write_u32(OUT + 0xC, 9);
        assert_eq!(set_caret(&ctx, OUT), ERROR_INVALID_PARAM);
        ctx.write_u32(OUT + 0xC, 1);
        assert_eq!(set_caret(&ctx, OUT), 0);
        common::write_units_nul(&ctx, OUT, &utf16("x\ny"), 16);
        assert_eq!(set_text(&ctx, OUT, 3), ERROR_INVALID_TEXT);
        common::write_units_nul(&ctx, OUT, &utf16("xyz"), 16);
        assert_eq!(set_text(&ctx, OUT, 2), 0);
        assert_eq!(common::scan_units(&ctx, TEXT, 32).unwrap(), utf16("xy"));
        ctx.write_f32(OUT, 10.0);
        ctx.write_f32(OUT + 4, 10.0);
        assert_eq!(set_text_geometry(&ctx, TEXT_AREA_MODE_EDIT, OUT), ERROR_INVALID_PARAM);
        assert_eq!(set_text_geometry(&ctx, TEXT_AREA_MODE_PREEDIT, OUT), 0);

        assert_eq!(close(), 0);
        assert_eq!(close(), ERROR_NOT_OPENED);
        assert!(!host_insert_text(g, &utf16("q")));
    }

    #[test]
    fn keyboard_never_connects() {
        let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D32_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        test_support::register_fake(KB_HANDLER, record);
        let ctx = CpuContext::new(&vm);
        take_events();
        let user = crate::user_service::PRIMARY_USER_ID;

        assert_eq!(keyboard_get_info(BASE + 0x100), ERROR_NOT_OPENED);
        assert_eq!(keyboard_close(user), ERROR_NOT_OPENED);
        assert_eq!(keyboard_get_resource_id(&ctx, user, BASE + 0x100), ERROR_NOT_OPENED);
        assert_eq!(ctx.read_u32(BASE + 0x100), Some(user as u32));

        let mut kp = [0u8; KEYBOARD_PARAM_SIZE];
        kp[8..16].copy_from_slice(&5u64.to_le_bytes());
        ctx.write_bytes(BASE, &kp);
        assert_eq!(keyboard_open(&ctx, user, BASE), ERROR_INVALID_HANDLER);
        kp[16..24].copy_from_slice(&KB_HANDLER.to_le_bytes());
        kp[0] = 0x40;
        ctx.write_bytes(BASE, &kp);
        assert_eq!(keyboard_open(&ctx, user, BASE), ERROR_INVALID_OPTION);
        kp[0] = 0;
        kp[0x18] = 1;
        ctx.write_bytes(BASE, &kp);
        assert_eq!(keyboard_open(&ctx, user, BASE), ERROR_INVALID_RESERVED);
        kp[0x18] = 0;
        ctx.write_bytes(BASE, &kp);
        assert_eq!(keyboard_open(&ctx, -1, BASE), ERROR_INVALID_USER_ID);
        assert_eq!(keyboard_open(&ctx, user, BASE), 0);
        assert_eq!(keyboard_open(&ctx, user, BASE), ERROR_BUSY);
        assert_eq!(keyboard_get_resource_id(&ctx, user, BASE + 0x100), ERROR_CONNECTION_FAILED);
        assert_eq!(keyboard_get_info(BASE + 0x100), ERROR_NO_RESOURCE_ID);
        assert_eq!(keyboard_set_mode(user, 0x80), ERROR_INVALID_MODE);
        assert_eq!(keyboard_set_mode(user, 0x7F), 0);
        assert_eq!(update(&ctx, KB_HANDLER), 0);
        let ev = take_events();
        assert_eq!(ev.len(), 1);
        assert_eq!((ev[0].0, ev[0].1), (5, EVENT_KEYBOARD_OPEN));
        assert_eq!(u32::from_le_bytes(ev[0].2[8..12].try_into().unwrap()), user as u32);
        assert_eq!(keyboard_close(user), 0);
    }

    #[test]
    fn panel_size_and_nids() {
        assert_eq!(panel_size(TYPE_NUMBER, 0), (370, 402));
        assert_eq!(panel_size(TYPE_DEFAULT, OPTION_USE_OVER_2K), (1586, 816));
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }
}
