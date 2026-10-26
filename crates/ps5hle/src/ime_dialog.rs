// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceImeDialog: the system text-entry dialog.
//!
//! The dialog is a session (NONE -> RUNNING -> FINISHED -> NONE via Term)
//! edited by a host UI through the `host_*` functions. Edits are filtered by
//! the guest text filter and committed into the guest input buffer on the
//! next status/result poll; cancel and abort restore the original text.
//!
//! When no host UI is attached (no visibility callback), the emulator acts
//! as the user: after the guest has seen the dialog running once, an empty
//! field gets the default text and the dialog is accepted.

use crate::context::CpuContext;
use crate::ime_common::{
    self as common, ExtendedParam, ExternalAction, ExternalInput, HostSnapshot, TextEditEngine, VisibilityCallback,
    VisualState, ALIGNMENT_END, ENTER_LABEL_GO, OPTION_EXT_KEYBOARD, OPTION_MULTILINE, OPTION_PASSWORD,
    OPTION_USE_OVER_2K, TYPE_BASIC_LATIN, TYPE_DEFAULT, TYPE_NUMBER,
};
use crate::registry::{HleCallback, HleRegistry, HleResult};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex, MutexGuard, RwLock};

pub const MAX_TEXT_LENGTH: u32 = common::MAX_TEXT_LENGTH;
pub const MAX_TITLE_LENGTH: u32 = 128;
pub const MAX_PLACEHOLDER_LENGTH: u32 = 64;

/// SceImeDialogStatus.
pub const STATUS_NONE: u32 = 0;
pub const STATUS_RUNNING: u32 = 1;
pub const STATUS_FINISHED: u32 = 2;
/// SceImeDialogEndStatus.
pub const END_STATUS_OK: u32 = 0;
pub const END_STATUS_USER_CANCELED: u32 = 1;
pub const END_STATUS_ABORTED: u32 = 2;

const fn err(code: u32) -> i32 {
    code as i32
}

pub const ERROR_BUSY: i32 = err(0x80BC_0001);
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
pub const ERROR_INVALID_PARAM: i32 = err(0x80BC_0030);
pub const ERROR_INVALID_ADDRESS: i32 = err(0x80BC_0031);
pub const ERROR_INVALID_RESERVED: i32 = err(0x80BC_0032);
pub const ERROR_INVALID_TITLE: i32 = err(0x80BC_0101);
pub const ERROR_NOT_RUNNING: i32 = err(0x80BC_0105);
pub const ERROR_NOT_FINISHED: i32 = err(0x80BC_0106);
pub const ERROR_NOT_IN_USE: i32 = err(0x80BC_0107);

const VALID_OPTIONS: u32 = 0x0000_7BFF;
const VALID_EXTENDED_OPTIONS: u32 = 0x0000_5FDE;
const VALID_LANGUAGES: u64 = 0x0000_0001_FF1F_FFFF;

/// SceImeDialogParam (0x60 bytes).
pub const PARAM_SIZE: usize = 0x60;
/// SceImeDialogResult { endstatus; reserved[12]; }.
pub const RESULT_SIZE: usize = 0x10;
/// SceImeDialogPanelPositionAndForm (0x1C bytes).
pub const POSITION_AND_FORM_SIZE: usize = 0x1C;
/// SceImePositionAndForm.type reported for an open panel.
pub const POSITION_AND_FORM_PANEL_TYPE: u32 = 2;
const MAX_EXTERNAL_INPUTS: usize = 128;

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
    placeholder: u64,
    title: u64,
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
            placeholder: u64_at(0x40),
            title: u64_at(0x48),
            reserved_zero: raw[0x50..0x60].iter().all(|&b| b == 0),
        }
    }

    fn read(ctx: &CpuContext, addr: u64) -> Option<Self> {
        if addr == 0 {
            return None;
        }
        ctx.read_bytes(addr, PARAM_SIZE).map(|raw| Self::parse(&raw))
    }
}

#[derive(Default)]
struct State {
    status: u32,
    end_status: u32,
    generation: u64,
    revision: u64,
    param: Param,
    extended: ExtendedParam,
    input_changed: bool,
    commit_pending: bool,
    original_text: Vec<u16>,
    editor: TextEditEngine,
    title: Vec<u16>,
    placeholder: Vec<u16>,
    external_inputs: Vec<ExternalInput>,
    /// The guest has seen this session running (headless host).
    headless_seen: bool,
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| Mutex::new(State::default()));
static STATUS: AtomicU32 = AtomicU32::new(STATUS_NONE);
static REVISION: AtomicU64 = AtomicU64::new(0);
static VISIBILITY: RwLock<Option<VisibilityCallback>> = RwLock::new(None);

fn state() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|p| p.into_inner())
}

fn notify_visibility(visible: bool, generation: u64) {
    let cb = *VISIBILITY.read().unwrap_or_else(|p| p.into_inner());
    if let Some(cb) = cb {
        cb(visible, generation);
    }
}

/// Bounded NUL-terminated UTF-16 read: Ok(text) when a NUL occurs within
/// `limit` units and the text is valid UTF-16.
fn read_bounded(ctx: &CpuContext, addr: u64, limit: u32) -> Result<Vec<u16>, ()> {
    if addr == 0 {
        return Ok(Vec::new());
    }
    let units = common::scan_units(ctx, addr, limit as usize + 1).ok_or(())?;
    if units.len() > limit as usize || !common::is_valid_utf16(&units) {
        return Err(());
    }
    Ok(units)
}

fn validate_extended(ext: Option<&ExtendedParam>) -> i32 {
    match ext {
        Some(e) if !e.is_valid(VALID_EXTENDED_OPTIONS) => ERROR_INVALID_EXTENDED,
        _ => 0,
    }
}

struct Validated {
    initial: Vec<u16>,
    title: Vec<u16>,
    placeholder: Vec<u16>,
}

fn validate_param(ctx: &CpuContext, param: Option<&Param>, ext: Option<&ExtendedParam>) -> Result<Validated, i32> {
    let p = param.ok_or(ERROR_INVALID_ADDRESS)?;
    if p.ty > TYPE_NUMBER {
        return Err(ERROR_INVALID_TYPE);
    }
    if p.option & !VALID_OPTIONS != 0 {
        return Err(ERROR_INVALID_OPTION);
    }
    if p.supported_languages & !VALID_LANGUAGES != 0 {
        return Err(ERROR_INVALID_LANGUAGES);
    }
    let over_2k = p.option & OPTION_USE_OVER_2K != 0;
    let (max_x, max_y) = if over_2k { (3840.0, 2160.0) } else { (1920.0, 1080.0) };
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
    let multiline = p.option & OPTION_MULTILINE != 0;
    let password = p.option & OPTION_PASSWORD != 0;
    if (multiline && password)
        || (multiline && p.ty != TYPE_DEFAULT && p.ty != TYPE_BASIC_LATIN)
        || (password && p.ty != TYPE_BASIC_LATIN && p.ty != TYPE_NUMBER)
    {
        return Err(ERROR_INVALID_PARAM);
    }
    if p.user_id < 0 || p.user_id == 0xFF {
        return Err(ERROR_INVALID_USER_ID);
    }
    if !p.reserved_zero {
        return Err(ERROR_INVALID_RESERVED);
    }
    if p.input_text_buffer == 0 {
        return Err(ERROR_INVALID_TEXT_BUFFER);
    }
    match validate_extended(ext) {
        0 => {}
        e => return Err(e),
    }
    if p.enter_label > ENTER_LABEL_GO {
        return Err(ERROR_INVALID_ENTER_LABEL);
    }
    if p.input_method != 0 {
        return Err(ERROR_INVALID_INPUT_METHOD);
    }
    if p.max_text_length == 0 || p.max_text_length > MAX_TEXT_LENGTH {
        return Err(ERROR_INVALID_MAX_TEXT_LENGTH);
    }
    let initial = read_bounded(ctx, p.input_text_buffer, p.max_text_length).map_err(|_| ERROR_INVALID_TEXT_BUFFER)?;
    let title = read_bounded(ctx, p.title, MAX_TITLE_LENGTH).map_err(|_| ERROR_INVALID_TITLE)?;
    let placeholder = read_bounded(ctx, p.placeholder, MAX_PLACEHOLDER_LENGTH).map_err(|_| ERROR_INVALID_PARAM)?;
    Ok(Validated { initial, title, placeholder })
}

/// Panel (width, height) of the dialog for `param`.
fn compute_panel_size(ty: u32, option: u32, ext: Option<&ExtendedParam>) -> (u32, u32) {
    let multiline = option & OPTION_MULTILINE != 0;
    let hide_keyboard =
        ext.is_some_and(|e| option & OPTION_EXT_KEYBOARD != 0 && e.option & common::EXT_OPTION_HIDE_KEY_PANEL != 0);
    let (w, h) = match ty {
        TYPE_NUMBER => (370, if hide_keyboard { 102 } else { 522 }),
        TYPE_BASIC_LATIN if hide_keyboard => (793, if multiline { 203 } else { 103 }),
        _ if hide_keyboard => (793, if multiline { 268 } else { 168 }),
        _ => (793, if multiline { 628 } else { 528 }),
    };
    if option & OPTION_USE_OVER_2K != 0 { (w * 2, h * 2) } else { (w, h) }
}

fn write_guest_text(ctx: &CpuContext, p: &Param, text: &[u16]) {
    if p.input_text_buffer != 0 {
        common::write_units_nul(ctx, p.input_text_buffer, text, p.max_text_length);
    }
}

/// Run the guest text filter over pending edits and commit the text (or
/// the original text after cancel/abort) into the guest buffer.
fn apply_filter_and_commit(ctx: &CpuContext) {
    let (filter, mut source, max_length, generation, revision) = {
        let mut s = state();
        if s.status == STATUS_NONE || (!s.input_changed && !s.commit_pending) {
            return;
        }
        let filter = if s.input_changed { s.param.filter } else { 0 };
        s.input_changed = false;
        (filter, s.editor.text().to_vec(), s.param.max_text_length, s.generation, s.revision)
    };

    if filter != 0 {
        if let Some(mut filtered) = common::run_text_filter(ctx, filter, &source, MAX_TEXT_LENGTH) {
            common::clamp_text(&mut filtered, max_length);
            source = filtered;
        }
    }

    let mut s = state();
    if s.generation != generation || s.revision != revision || s.status == STATUS_NONE {
        if s.generation == generation
            && (s.status == STATUS_RUNNING || (s.status == STATUS_FINISHED && s.end_status == END_STATUS_OK))
        {
            s.input_changed = true;
            s.commit_pending = true;
        }
        return;
    }
    if !s.input_changed {
        let cursor = s.editor.cursor();
        s.editor.replace_text(source, cursor);
    }
    let committed = if s.status == STATUS_FINISHED && s.end_status != END_STATUS_OK {
        s.original_text.clone()
    } else {
        s.editor.text().to_vec()
    };
    write_guest_text(ctx, &s.param, &committed);
    s.commit_pending = false;
}

fn match_running(s: &State, generation: u64) -> bool {
    s.status == STATUS_RUNNING && s.generation == generation
}

fn finish_from_host(generation: u64, end_status: u32) -> bool {
    let gen = {
        let mut s = state();
        if !match_running(&s, generation) {
            return false;
        }
        s.status = STATUS_FINISHED;
        s.end_status = end_status;
        s.commit_pending = true;
        s.revision += 1;
        STATUS.store(STATUS_FINISHED, Ordering::Release);
        REVISION.store(s.revision, Ordering::Release);
        s.generation
    };
    notify_visibility(false, gen);
    true
}

/// Apply queued host keyboard input, running the guest keyboard filter.
fn apply_external_inputs(ctx: &CpuContext) {
    let (inputs, filter, generation, user_id, multiline) = {
        let mut s = state();
        if s.status != STATUS_RUNNING || s.external_inputs.is_empty() {
            return;
        }
        let inputs = std::mem::take(&mut s.external_inputs);
        (inputs, s.extended.ext_keyboard_filter, s.generation, s.param.user_id, s.param.option & OPTION_MULTILINE != 0)
    };
    for mut input in inputs {
        if !match_running(&state(), generation) {
            break;
        }
        input.key.user_id = user_id;
        let mut accepted = true;
        if filter != 0 {
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
                host_insert_text(generation, &input.text);
            }
            ExternalAction::Backspace => {
                host_backspace(generation);
            }
            ExternalAction::MoveLeft => {
                host_move_cursor(generation, -1);
            }
            ExternalAction::MoveRight => {
                host_move_cursor(generation, 1);
            }
            ExternalAction::Cancel => {
                host_cancel(generation);
            }
            ExternalAction::Accept => {
                host_accept(generation);
            }
            ExternalAction::Newline => {
                host_insert_text(generation, &[0x0A]);
            }
        }
    }
}

/// Headless host: the second poll of a running dialog types the default
/// text into an empty field and accepts.
fn headless_step() {
    if VISIBILITY.read().unwrap_or_else(|p| p.into_inner()).is_some() {
        return;
    }
    let (generation, empty) = {
        let mut s = state();
        if s.status != STATUS_RUNNING {
            return;
        }
        if !s.headless_seen {
            s.headless_seen = true;
            return;
        }
        (s.generation, s.editor.text().is_empty())
    };
    if empty {
        host_insert_text(generation, &common::default_input_text());
    }
    log::info!("[IMEDIALOG] no host UI: accepting the dialog");
    host_accept(generation);
}

// ---------------------------------------------------------------------------
// Guest API
// ---------------------------------------------------------------------------

pub fn get_panel_size_extended(ctx: &CpuContext, param: u64, extended: u64, width: u64, height: u64) -> i32 {
    if param == 0 || width == 0 || height == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let Some(p) = Param::read(ctx, param) else {
        return ERROR_INVALID_ADDRESS;
    };
    if p.ty > TYPE_NUMBER {
        return ERROR_INVALID_TYPE;
    }
    if p.option & !VALID_OPTIONS != 0 {
        return ERROR_INVALID_OPTION;
    }
    if p.supported_languages & !VALID_LANGUAGES != 0 {
        return ERROR_INVALID_LANGUAGES;
    }
    let Ok(ext) = common::read_extended(ctx, extended) else {
        return ERROR_INVALID_ADDRESS;
    };
    match validate_extended(ext.as_ref()) {
        0 => {}
        e => return e,
    }
    let (w, h) = compute_panel_size(p.ty, p.option, ext.as_ref());
    if ctx.write_u32(width, w) && ctx.write_u32(height, h) { 0 } else { ERROR_INVALID_ADDRESS }
}

pub fn init(ctx: &CpuContext, param: u64, extended: u64) -> i32 {
    if STATUS.load(Ordering::Acquire) != STATUS_NONE {
        return ERROR_BUSY;
    }
    let p = Param::read(ctx, param);
    let ext = match common::read_extended(ctx, extended) {
        Ok(e) => e,
        Err(()) => return ERROR_INVALID_ADDRESS,
    };
    let v = match validate_param(ctx, p.as_ref(), ext.as_ref()) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let p = p.unwrap();
    let generation = {
        let mut s = state();
        if s.status != STATUS_NONE {
            return ERROR_BUSY;
        }
        let (generation, revision) = (s.generation + 1, s.revision + 1);
        *s = State::default();
        s.status = STATUS_RUNNING;
        s.generation = generation;
        s.revision = revision;
        s.param = p;
        s.extended = ext.unwrap_or_default();
        s.original_text = v.initial.clone();
        s.editor.reset(p.ty, p.option, p.max_text_length, v.initial);
        s.title = v.title;
        s.placeholder = v.placeholder;
        s.commit_pending = true;
        STATUS.store(STATUS_RUNNING, Ordering::Release);
        REVISION.store(revision, Ordering::Release);
        generation
    };
    notify_visibility(true, generation);
    0
}

pub fn get_status(ctx: &CpuContext) -> i32 {
    headless_step();
    apply_external_inputs(ctx);
    apply_filter_and_commit(ctx);
    STATUS.load(Ordering::Acquire) as i32
}

pub fn abort(ctx: &CpuContext) -> i32 {
    let generation = {
        let s = state();
        if s.status == STATUS_NONE {
            return ERROR_NOT_IN_USE;
        }
        if s.status != STATUS_RUNNING {
            return ERROR_NOT_RUNNING;
        }
        s.generation
    };
    if !finish_from_host(generation, END_STATUS_ABORTED) {
        return ERROR_NOT_RUNNING;
    }
    apply_filter_and_commit(ctx);
    0
}

pub fn get_result(ctx: &CpuContext, result: u64) -> i32 {
    if state().status == STATUS_NONE {
        return ERROR_NOT_IN_USE;
    }
    if result == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let Some(reserved) = ctx.read_bytes(result + 4, RESULT_SIZE - 4) else {
        return ERROR_INVALID_ADDRESS;
    };
    if reserved.iter().any(|&b| b != 0) {
        return ERROR_INVALID_RESERVED;
    }
    if state().status != STATUS_FINISHED {
        return ERROR_NOT_FINISHED;
    }
    apply_filter_and_commit(ctx);
    let s = state();
    if s.status == STATUS_NONE {
        return ERROR_NOT_IN_USE;
    }
    if s.status != STATUS_FINISHED {
        return ERROR_NOT_FINISHED;
    }
    if ctx.write_u32(result, s.end_status) { 0 } else { ERROR_INVALID_ADDRESS }
}

pub fn term(ctx: &CpuContext) -> i32 {
    {
        let s = state();
        if s.status == STATUS_NONE {
            return ERROR_NOT_IN_USE;
        }
        if s.status != STATUS_FINISHED {
            return ERROR_NOT_FINISHED;
        }
    }
    apply_filter_and_commit(ctx);
    let mut s = state();
    if s.status == STATUS_NONE {
        return ERROR_NOT_IN_USE;
    }
    if s.status != STATUS_FINISHED {
        return ERROR_NOT_FINISHED;
    }
    let (generation, revision) = (s.generation, s.revision);
    *s = State::default();
    s.generation = generation;
    s.revision = revision;
    STATUS.store(STATUS_NONE, Ordering::Release);
    0
}

/// sceImeDialogForceClose(): end the dialog from any state but NONE; the
/// text is not written back.
pub fn force_close() -> i32 {
    let mut s = state();
    if s.status == STATUS_NONE {
        return ERROR_NOT_IN_USE;
    }
    let (generation, revision) = (s.generation, s.revision);
    *s = State::default();
    s.generation = generation;
    s.revision = revision;
    STATUS.store(STATUS_NONE, Ordering::Release);
    0
}

pub fn get_panel_position_and_form(ctx: &CpuContext, form: u64) -> i32 {
    let s = state();
    if s.status == STATUS_NONE {
        return ERROR_NOT_IN_USE;
    }
    if form == 0 {
        return ERROR_INVALID_ADDRESS;
    }
    let (w, h) = compute_panel_size(s.param.ty, s.param.option, Some(&s.extended));
    let mut b = [0u8; POSITION_AND_FORM_SIZE];
    for (i, v) in [POSITION_AND_FORM_PANEL_TYPE, s.param.posx.to_bits(), s.param.posy.to_bits(), s.param.horizontal_alignment, s.param.vertical_alignment, w, h]
        .into_iter()
        .enumerate()
    {
        b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    if ctx.write_bytes(form, &b) { 0 } else { ERROR_INVALID_ADDRESS }
}

// ---------------------------------------------------------------------------
// Host API (UI frontends)
// ---------------------------------------------------------------------------

pub fn visual_state() -> VisualState {
    VisualState { active: STATUS.load(Ordering::Acquire) == STATUS_RUNNING, revision: REVISION.load(Ordering::Acquire) }
}

/// Register the host UI; `None` returns to the headless host.
pub fn set_visibility_callback(callback: Option<VisibilityCallback>) {
    *VISIBILITY.write().unwrap_or_else(|p| p.into_inner()) = callback;
}

pub fn host_snapshot() -> Option<HostSnapshot> {
    let s = state();
    if s.status != STATUS_RUNNING {
        return None;
    }
    let (panel_width, panel_height) = compute_panel_size(s.param.ty, s.param.option, Some(&s.extended));
    Some(HostSnapshot {
        generation: s.generation,
        ty: s.param.ty,
        enter_label: s.param.enter_label,
        option: s.param.option,
        max_text_length: s.param.max_text_length,
        cursor: s.editor.cursor(),
        disable_device: s.extended.disable_device,
        key_panel_visible: s.param.option & OPTION_EXT_KEYBOARD == 0
            || s.extended.option & common::EXT_OPTION_HIDE_KEY_PANEL == 0,
        posx: s.param.posx,
        posy: s.param.posy,
        horizontal_alignment: s.param.horizontal_alignment,
        vertical_alignment: s.param.vertical_alignment,
        panel_width,
        panel_height,
        text: s.editor.text().to_vec(),
        title: s.title.clone(),
        placeholder: s.placeholder.clone(),
    })
}

pub fn host_insert_text(generation: u64, text: &[u16]) -> bool {
    let mut s = state();
    if !match_running(&s, generation) || s.editor.insert(text).is_none() {
        return false;
    }
    s.input_changed = true;
    s.commit_pending = true;
    true
}

pub fn host_backspace(generation: u64) -> bool {
    let mut s = state();
    if !match_running(&s, generation) || s.editor.backspace().is_none() {
        return false;
    }
    s.input_changed = true;
    s.commit_pending = true;
    true
}

pub fn host_move_cursor(generation: u64, delta: i32) -> bool {
    let mut s = state();
    match_running(&s, generation) && s.editor.move_cursor(delta)
}

pub fn host_accept(generation: u64) -> bool {
    finish_from_host(generation, END_STATUS_OK)
}

pub fn host_cancel(generation: u64) -> bool {
    finish_from_host(generation, END_STATUS_USER_CANCELED)
}

pub fn host_queue_external_input(generation: u64, input: ExternalInput) -> bool {
    let mut s = state();
    if !match_running(&s, generation) || s.external_inputs.len() >= MAX_EXTERNAL_INPUTS {
        return false;
    }
    s.external_inputs.push(input);
    true
}

// ---------------------------------------------------------------------------
// HLE callbacks
// ---------------------------------------------------------------------------

fn r(v: i32) -> HleResult {
    v as HleResult
}

fn cb_get_panel_size(ctx: &mut CpuContext) -> HleResult {
    r(get_panel_size_extended(ctx, ctx.rdi(), 0, ctx.rsi(), ctx.rdx()))
}

fn cb_get_panel_size_extended(ctx: &mut CpuContext) -> HleResult {
    r(get_panel_size_extended(ctx, ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx()))
}

fn cb_init(ctx: &mut CpuContext) -> HleResult {
    r(init(ctx, ctx.rdi(), ctx.rsi()))
}

fn cb_get_status(ctx: &mut CpuContext) -> HleResult {
    r(get_status(ctx))
}

fn cb_abort(ctx: &mut CpuContext) -> HleResult {
    r(abort(ctx))
}

fn cb_get_result(ctx: &mut CpuContext) -> HleResult {
    r(get_result(ctx, ctx.rdi()))
}

fn cb_term(ctx: &mut CpuContext) -> HleResult {
    r(term(ctx))
}

fn cb_force_close(_ctx: &mut CpuContext) -> HleResult {
    r(force_close())
}

fn cb_get_panel_position_and_form(ctx: &mut CpuContext) -> HleResult {
    r(get_panel_position_and_form(ctx, ctx.rdi()))
}

const LIB: &str = "libSceImeDialog";

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("CRD+jSErEJQ", "sceImeDialogGetPanelSizeExtended", cb_get_panel_size_extended),
    ("IADmD4tScBY", "sceImeDialogGetStatus", cb_get_status),
    ("gyTyVn+bXMw", "sceImeDialogTerm", cb_term),
    ("bX4H+sxPI-o", "sceImeDialogForceClose", cb_force_close),
    ("x01jxu+vxlc", "sceImeDialogGetResult", cb_get_result),
    ("wqsJvRXwl58", "sceImeDialogGetPanelSize", cb_get_panel_size),
    ("NUeBrN7hzf0", "sceImeDialogInit", cb_init),
    ("oBmw4xrmfKs", "sceImeDialogAbort", cb_abort),
    ("8jqzzPioYl8", "sceImeDialogGetPanelPositionAndForm", cb_get_panel_position_and_form),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests;
