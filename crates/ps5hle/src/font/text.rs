// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Text sources, font strings, writing (per-character layout steps) and
//! writing lines (runs of writing orders laid out on one line).
//!
//! Strings are parsed by the game's own text parser callback; the parsed
//! characters live in a guest array of FontTextCharacter records linked
//! through prev/next, so the game can walk them directly. Writing state is
//! kept on the host keyed by the guest FontWriting block, with the current
//! step and metrics mirrored into that block.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use super::*;

const TEXT_SOURCE_MAGIC: u16 = 0x0F04;
const FONT_STRING_MAGIC: u16 = 0x0F05;
const FONT_WRITING_MAGIC: u16 = 0x0F06;
const WRITING_LINE_MAGIC: u16 = 0x0F08;
const CREATE_STRING_DETAIL_ID: u16 = 0x0FD4;
const CREATE_WRITING_LINE_DETAIL_ID: u16 = 0x0FD5;
const WRITING_FORM_HORIZONTAL: u32 = 0x10;
const WRITING_FORM_HORIZONTAL_RTL: u32 = 0x11;
const WRITING_FORM_HORIZONTAL_LTR: u32 = 0x12;
const WRITING_ATTRIBUTE_ID_MASK: u64 = 0xFF;
const WRITING_ATTRIBUTE_ID: u64 = 0xFA;
const WRITING_ATTRIBUTE_SPACE_BEFORE: u64 = 1 << 30;
const WRITING_ATTRIBUTE_SPACE_AFTER: u64 = 1 << 31;
/// Embedding levels 0..=0x7E.
const WRITING_LEVEL_COUNT: usize = 0x7F;
const PARSER_RESULT_ERROR: i32 = -1;
const PARSER_RESULT_TERMINATE: i32 = 0;
const PARSER_RESULT_FONT_CODE: i32 = 1;
const MAX_PARSED_CHARACTERS: usize = 4096;
const CHARACTER_FLAG_FORMAT: u64 = 1 << 42;
const WHITESPACE_CLASS: u64 = 0x0E;

/// sizeof(FontTextCharacter)
const CHAR_SIZE: u64 = 0xC8;
const CHAR_PREV: u64 = 0x00;
const CHAR_NEXT: u64 = 0x08;
const CHAR_ORDER: u64 = 0x10;
const CHAR_FONT: u64 = 0x18;
const CHAR_CODE: u64 = 0x28;
const CHAR_CLUSTER_SPAN: u64 = 0x30;
const CHAR_CLUSTER_INDEX: u64 = 0x31;
const CHAR_CLUSTER_KIND: u64 = 0x32;
const CHAR_SYNTHETIC: u64 = 0x33;
const CHAR_FLAGS: u64 = 0x38;

const STRING_OBJECT_SIZE: u64 = 0x40;
const WRITING_STEP: u64 = 0x40;
const WRITING_METRICS: u64 = 0x90;
/// The writing's FontWritingLetterStep (0x40 bytes) in the free tail of the
/// 0x100-byte FontWriting, handed out by RefersRenderStepCharacter.
const WRITING_LETTER_STEP: u64 = 0xC0;
const LINE_OBJECT_SIZE: u64 = 0x80;
/// The line's single FontWritingLineStep, handed out on every step.
const LINE_STEP: u64 = 0x40;
const LINE_STEP_SIZE: u64 = 0x40;

// ---------------------------------------------------------------------------
// Host state
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct Char {
    font: u64,
    code: u32,
    order: u64,
    flags: u64,
    cluster_index: u8,
    cluster_kind: u8,
    synthetic: i8,
}

impl Char {
    fn level(&self) -> u32 {
        ((self.flags >> 24) & 0xFF) as u32
    }
}

struct StringData {
    /// Guest FontTextCharacter array (0 when empty).
    block: u64,
    chars: Vec<Char>,
    writing_form: u32,
    default_font: u64,
    terminate_code: u32,
    terminate_order: u64,
}

#[derive(Clone, Copy, Default)]
struct Step {
    x: f32,
    y: f32,
    advance_x: f32,
    advance_y: f32,
    font: u64,
    profile: u32,
    glyph_code: u32,
    pos_x: f32,
    pos_y: f32,
    metrics: GlyphMetrics,
}

impl Step {
    fn bytes(&self) -> [u8; 0x48] {
        let mut b = [0u8; 0x48];
        for (i, v) in [self.x, self.y, self.advance_x, self.advance_y].iter().enumerate() {
            b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        b[0x10..0x18].copy_from_slice(&self.font.to_le_bytes());
        b[0x18..0x1C].copy_from_slice(&self.profile.to_le_bytes());
        b[0x1C..0x20].copy_from_slice(&self.glyph_code.to_le_bytes());
        b[0x20..0x24].copy_from_slice(&self.pos_x.to_le_bytes());
        b[0x24..0x28].copy_from_slice(&self.pos_y.to_le_bytes());
        for (i, v) in self.metrics.iter().enumerate() {
            b[0x28 + i * 4..0x2C + i * 4].copy_from_slice(&v.to_le_bytes());
        }
        b
    }
}

/// FontWritingMetrics { advanceX, advanceY, extent {top, bottom, left, right} }
type WritingMetrics = [f32; 6];

struct WritingData {
    block: u64,
    chars: Vec<Char>,
    default_font: u64,
    next_index: usize,
    current_index: usize,
    pen_x: f32,
    pen_y: f32,
    has_aggregate: bool,
    has_current: bool,
    /// The current step's character has not been handed out yet.
    has_current_character: bool,
    mask_format: bool,
    step: Step,
    raw: Option<GlyphMetrics>,
    metrics: WritingMetrics,
}

/// One writing order of a line, linked into visual order.
#[derive(Clone, Copy, Default)]
struct LineOrder {
    orderer: u64,
    prev: Option<usize>,
    next: Option<usize>,
    attribute: u64,
    metrics: WritingMetrics,
}

/// First and last order (in visual order) of an embedding level run.
type LevelRun = (Option<usize>, Option<usize>);

/// FontWritingLineStep, mirrored in the guest block of the line.
#[derive(Clone, Copy, Default)]
struct LineStep {
    x: f32,
    y: f32,
    advance_x: f32,
    advance_y: f32,
    spacing_progress: f32,
    orderer: u64,
    adjusting_x: f32,
    adjusting_y: f32,
    metrics: WritingMetrics,
}

impl LineStep {
    fn from_bytes(b: &[u8]) -> Self {
        let f = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        LineStep {
            x: f(0x00),
            y: f(0x04),
            advance_x: f(0x08),
            advance_y: f(0x0C),
            spacing_progress: f(0x10),
            orderer: u64::from_le_bytes(b[0x18..0x20].try_into().unwrap()),
            adjusting_x: f(0x20),
            adjusting_y: f(0x24),
            metrics: std::array::from_fn(|i| f(0x28 + i * 4)),
        }
    }

    fn bytes(&self) -> [u8; LINE_STEP_SIZE as usize] {
        let mut b = [0u8; LINE_STEP_SIZE as usize];
        let fields = [self.x, self.y, self.advance_x, self.advance_y, self.spacing_progress];
        for (i, v) in fields.iter().enumerate() {
            b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        b[0x18..0x20].copy_from_slice(&self.orderer.to_le_bytes());
        b[0x20..0x24].copy_from_slice(&self.adjusting_x.to_le_bytes());
        b[0x24..0x28].copy_from_slice(&self.adjusting_y.to_le_bytes());
        for (i, v) in self.metrics.iter().enumerate() {
            b[0x28 + i * 4..0x2C + i * 4].copy_from_slice(&v.to_le_bytes());
        }
        b
    }
}

// Writing metrics fields.
const M_ADVANCE_X: usize = 0;
const M_ADVANCE_Y: usize = 1;
const M_TOP: usize = 2;
const M_BOTTOM: usize = 3;
const M_LEFT: usize = 4;
const M_RIGHT: usize = 5;

/// A writing line: orders are linked in visual order as they arrive,
/// following their embedding levels (even levels run left to right, odd
/// levels right to left), then handed out one render step at a time.
struct LineData {
    form: i32,
    render_metrics: WritingMetrics,
    advance_x: f32,
    advance_y: f32,
    ordering_space: f32,
    stepped_space: f32,
    last_level: usize,
    last: Option<usize>,
    head: Option<usize>,
    tail: Option<usize>,
    cursor: Option<usize>,
    levels: [LevelRun; WRITING_LEVEL_COUNT],
    orders: Vec<LineOrder>,
}

impl LineData {
    fn new(form: i32) -> Self {
        LineData {
            form,
            render_metrics: [0.0; 6],
            advance_x: 0.0,
            advance_y: 0.0,
            ordering_space: 0.0,
            stepped_space: 0.0,
            last_level: 0,
            last: None,
            head: None,
            tail: None,
            cursor: None,
            levels: [(None, None); WRITING_LEVEL_COUNT],
            orders: Vec::new(),
        }
    }

    fn right_to_left(&self) -> bool {
        self.form == WRITING_FORM_HORIZONTAL_RTL as i32
    }

    fn reset_orders(&mut self) {
        self.orders.clear();
        self.levels = [(None, None); WRITING_LEVEL_COUNT];
        self.advance_x = 0.0;
        self.advance_y = 0.0;
        self.ordering_space = 0.0;
        self.last = None;
        self.head = None;
        self.tail = None;
        self.cursor = None;
    }

    fn spaced(&self, before: usize, after: usize) -> bool {
        self.orders[before].attribute & WRITING_ATTRIBUTE_SPACE_AFTER != 0
            || self.orders[after].attribute & WRITING_ATTRIBUTE_SPACE_BEFORE != 0
    }

    fn head_space(&self, o: usize) -> f32 {
        let m = &self.orders[o].metrics;
        if self.right_to_left() { m[M_ADVANCE_X] - m[M_LEFT] } else { m[M_LEFT] }
    }

    fn tail_space(&self, o: usize) -> f32 {
        let m = &self.orders[o].metrics;
        if self.right_to_left() { m[M_RIGHT] } else { m[M_ADVANCE_X] - m[M_RIGHT] }
    }

    fn insert_after(&mut self, anchor: usize, o: usize) {
        let next = self.orders[anchor].next;
        self.orders[o].prev = Some(anchor);
        self.orders[o].next = next;
        if let Some(n) = next {
            self.orders[n].prev = Some(o);
        }
        self.orders[anchor].next = Some(o);
    }

    fn insert_before(&mut self, anchor: usize, o: usize) {
        let prev = self.orders[anchor].prev;
        self.orders[o].next = Some(anchor);
        self.orders[o].prev = prev;
        if let Some(p) = prev {
            self.orders[p].next = Some(o);
        }
        self.orders[anchor].prev = Some(o);
    }

    /// Link a new order behind the previous one at `level`: a deeper or
    /// equal level continues in the direction of the previous level; a
    /// shallower level closes the deeper runs and continues after (even)
    /// or before (odd) the whole closed span.
    fn link(&mut self, o: usize, level: usize) {
        let Some(last) = self.last else {
            self.levels[level] = (Some(o), Some(o));
            return;
        };
        if level >= self.last_level {
            if self.last_level % 2 == 0 {
                self.insert_after(last, o);
            } else {
                self.insert_before(last, o);
            }
            let (prev, next) = (self.orders[o].prev, self.orders[o].next);
            let slot = &mut self.levels[level];
            if slot.1.is_none() || prev == slot.1 {
                slot.1 = Some(o);
            }
            if slot.0.is_none() || next == slot.0 {
                slot.0 = Some(o);
            }
            return;
        }
        let (mut first, mut end) = self.levels[self.last_level];
        self.levels[self.last_level] = (None, None);
        for closing in (level + 1..self.last_level).rev() {
            let closed = self.levels[closing];
            if closing % 2 == 0 {
                if closed.0.is_some() {
                    first = closed.0;
                }
            } else if closed.1.is_some() {
                end = closed.1;
            }
            self.levels[closing] = (None, None);
        }
        if level % 2 == 0 {
            if self.levels[level].0.is_none() {
                self.levels[level].0 = first;
            }
            self.insert_after(end.unwrap_or(last), o);
            self.levels[level].1 = Some(o);
        } else {
            if self.levels[level].1.is_none() {
                self.levels[level].1 = end;
            }
            self.insert_before(first.unwrap_or(last), o);
            self.levels[level].0 = Some(o);
        }
    }

    /// Change of the inline ordering space caused by inserting `o` between
    /// its neighbours.
    fn ordering_space_delta(&self, o: usize) -> f32 {
        let (before, after) = (self.orders[o].prev, self.orders[o].next);
        let split = matches!((before, after), (Some(b), Some(a)) if self.spaced(b, a));
        let sign = if split { -1.0 } else { 1.0 };
        let mut delta = 0.0;
        if let Some(b) = before {
            let spaced = self.spaced(b, o);
            if spaced {
                delta = self.head_space(o);
            }
            if spaced != split {
                delta += sign * self.tail_space(b);
            }
        }
        if let Some(a) = after {
            let spaced = self.spaced(o, a);
            if spaced {
                delta += self.tail_space(o);
            }
            if spaced != split {
                delta += sign * self.head_space(a);
            }
        }
        delta
    }

    /// Fold the step just handed out (with the caller's adjustment) into
    /// the render metrics and advance the pen past it.
    fn extend_render_metrics(&mut self, step: &mut LineStep) {
        let run = step.metrics;
        let x = step.x + step.adjusting_x;
        let y = step.y + step.adjusting_y;
        step.x += run[M_ADVANCE_X];
        step.y += run[M_ADVANCE_Y];
        let total = &mut self.render_metrics;
        total[M_ADVANCE_X] = x + run[M_ADVANCE_X];
        total[M_ADVANCE_Y] = y + run[M_ADVANCE_Y];
        if run[M_LEFT] != run[M_RIGHT] {
            let (left, right) = (x + run[M_LEFT], x + run[M_RIGHT]);
            if total[M_LEFT] == 0.0 && total[M_RIGHT] == 0.0 {
                total[M_LEFT] = left;
                total[M_RIGHT] = right;
            } else {
                if total[M_LEFT] > left {
                    total[M_LEFT] = left;
                }
                if right > total[M_RIGHT] {
                    total[M_RIGHT] = right;
                }
            }
        }
        if run[M_TOP] != run[M_BOTTOM] {
            let (top, bottom) = (y + run[M_TOP], y + run[M_BOTTOM]);
            if total[M_TOP] == 0.0 && total[M_BOTTOM] == 0.0 {
                total[M_TOP] = top;
                total[M_BOTTOM] = bottom;
            } else {
                if total[M_TOP] > top {
                    total[M_TOP] = top;
                }
                if bottom > total[M_BOTTOM] {
                    total[M_BOTTOM] = bottom;
                }
            }
        }
    }

    /// Hand out the next order in reading order; `None` past the end.
    fn next_step(&mut self, step: &mut LineStep) -> bool {
        if self.cursor.is_some() {
            self.extend_render_metrics(step);
        } else {
            self.stepped_space = 0.0;
            step.x = 0.0;
            step.y = 0.0;
            step.spacing_progress = 0.0;
            step.metrics = [0.0; 6];
            self.render_metrics = [0.0; 6];
        }
        let rtl = self.right_to_left();
        let next = match (rtl, self.cursor) {
            (true, Some(c)) => self.orders[c].prev,
            (true, None) => self.tail,
            (false, Some(c)) => self.orders[c].next,
            (false, None) => self.head,
        };
        let Some(next) = next else {
            self.cursor = None;
            step.metrics = [0.0; 6];
            return false;
        };
        let n = self.orders[next].metrics;
        if rtl {
            if let Some(s) = self.orders[next].next {
                if self.spaced(next, s) {
                    let m = self.orders[s].metrics;
                    self.stepped_space += m[M_ADVANCE_X] - m[M_LEFT] + n[M_RIGHT];
                }
            }
        } else if let Some(s) = self.orders[next].prev {
            if self.spaced(s, next) {
                let m = self.orders[s].metrics;
                self.stepped_space += m[M_ADVANCE_X] - m[M_RIGHT] + n[M_LEFT];
            }
        }
        self.cursor = Some(next);
        step.metrics = n;
        step.orderer = self.orders[next].orderer;
        step.adjusting_x = 0.0;
        step.adjusting_y = 0.0;
        step.spacing_progress = 0.0;
        if self.stepped_space != 0.0 && self.ordering_space != 0.0 {
            step.spacing_progress = self.stepped_space / self.ordering_space;
        }
        true
    }
}

#[derive(Default)]
struct TextState {
    strings: HashMap<u64, StringData>,
    writings: HashMap<u64, WritingData>,
    lines: HashMap<u64, LineData>,
    /// Character arrays shared between a string and its writings.
    block_refs: HashMap<u64, u32>,
}

static TEXT: Mutex<Option<TextState>> = Mutex::new(None);

struct TextGuard(MutexGuard<'static, Option<TextState>>);

impl std::ops::Deref for TextGuard {
    type Target = TextState;
    fn deref(&self) -> &TextState {
        self.0.as_ref().unwrap()
    }
}

impl std::ops::DerefMut for TextGuard {
    fn deref_mut(&mut self) -> &mut TextState {
        self.0.as_mut().unwrap()
    }
}

fn text() -> TextGuard {
    let mut g = TEXT.lock().unwrap_or_else(|p| p.into_inner());
    g.get_or_insert_with(TextState::default);
    TextGuard(g)
}

impl TextState {
    fn retain_block(&mut self, block: u64) {
        if block != 0 {
            *self.block_refs.entry(block).or_insert(0) += 1;
        }
    }

    fn release_block(&mut self, block: u64) {
        if block == 0 {
            return;
        }
        let gone = match self.block_refs.get_mut(&block) {
            Some(n) if *n > 1 => {
                *n -= 1;
                false
            }
            Some(_) => true,
            None => false,
        };
        if gone {
            self.block_refs.remove(&block);
            gfree(block);
        }
    }
}

fn is_whitespace(code: u32) -> bool {
    matches!(
        code,
        0x09 | 0x0A | 0x0B | 0x0C | 0x0D | 0x20 | 0x85 | 0xA0 | 0x1680 | 0x2028 | 0x2029 | 0x202F | 0x205F | 0x3000
    ) || (0x2000..=0x200A).contains(&code)
}

fn is_format(code: u32) -> bool {
    matches!(
        code,
        0xAD | 0x061C
            | 0x200B..=0x200F
            | 0x2060..=0x2064
            | 0x2066..=0x2069
            | 0xFE0E
            | 0xFE0F
            | 0xFEFF
    )
}

fn make_char(font: u64, code: u32, order: u64) -> Char {
    let mut flags = if is_whitespace(code) { WHITESPACE_CLASS << 8 } else { 0 };
    if is_format(code) {
        flags |= CHARACTER_FLAG_FORMAT;
    }
    Char { font, code, order, flags, ..Default::default() }
}

/// Whether `code` is a right-to-left letter (the Arabic-Indic digits are
/// not).
fn is_right_to_left_letter(code: u32) -> bool {
    if (0x0660..=0x0669).contains(&code) || (0x06F0..=0x06F9).contains(&code) {
        return false;
    }
    (0x0590..=0x08FF).contains(&code) || (0xFB1D..=0xFDFF).contains(&code) || (0xFE70..=0xFEFF).contains(&code)
        || (0x10800..=0x10FFF).contains(&code) || (0x1E800..=0x1EFFF).contains(&code)
}

/// Whether `code` carries no writing direction of its own.
fn is_direction_neutral(code: u32) -> bool {
    code < 0x41 || (0x5B..=0x60).contains(&code) || (0x7B..=0xBF).contains(&code) || code == 0xD7 || code == 0xF7
        || (0x2000..=0x2BFF).contains(&code) || (0x3000..=0x3004).contains(&code) || (0xFF00..=0xFF20).contains(&code)
}

/// The writing form a plain horizontal string takes: the first strong
/// character decides, isolate runs skipped and the direction marks
/// honoured (behaviour measured on hardware).
fn default_string_form(chars: &[Char]) -> u32 {
    let mut isolate_depth = 0u32;
    for &c in chars {
        let code = c.code;
        if (0x2066..=0x2068).contains(&code) {
            isolate_depth += 1;
            continue;
        }
        if code == 0x2069 {
            isolate_depth = isolate_depth.saturating_sub(1);
            continue;
        }
        if isolate_depth != 0 {
            continue;
        }
        if code == 0x200E {
            return WRITING_FORM_HORIZONTAL_LTR;
        }
        if code == 0x200F || is_right_to_left_letter(code) {
            return WRITING_FORM_HORIZONTAL_RTL;
        }
        if !is_direction_neutral(code) {
            return WRITING_FORM_HORIZONTAL_LTR;
        }
    }
    WRITING_FORM_HORIZONTAL_LTR
}

fn text_source_magic(ctx: &CpuContext, ts: u64) -> bool {
    ts != 0 && read_u16(ctx, ts) == Some(TEXT_SOURCE_MAGIC)
}

// ---------------------------------------------------------------------------
// Text sources
// ---------------------------------------------------------------------------

/// sceFontTextSourceInit(FontTextSource*, textAddress, textSizeByte, parser, textObject)
pub(super) fn cb_text_source_init(ctx: &mut CpuContext) -> HleResult {
    let (ts, addr, size, parser, object) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as u32, ctx.rcx(), ctx.r8());
    if ts == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    ctx.write_bytes(ts, &[0u8; 0x60]);
    if parser == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let end = if size != 0 { addr + size as u64 } else { 0 };
    let head = (WRITING_FORM_HORIZONTAL as u64) << 32 | TEXT_SOURCE_MAGIC as u64;
    let words = [head, addr, end, addr, parser, object, 0, addr, end, parser, object, 0];
    let b: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    if !ctx.write_bytes(ts, &b) {
        return ERROR_INVALID_PARAMETER;
    }
    0
}

/// sceFontTextSourceRewind(FontTextSource*): parsing restarts at the start
/// of the text.
pub(super) fn cb_text_source_rewind(ctx: &mut CpuContext) -> HleResult {
    let ts = ctx.rdi();
    if !text_source_magic(ctx, ts) {
        return ERROR_INVALID_PARAMETER;
    }
    let start = ctx.read_u64(ts + 0x08).unwrap_or(0);
    ctx.write_u64(ts + 0x18, start);
    0
}

/// sceFontTextSourceSetDefaultFont(FontTextSource*, FontHandle)
pub(super) fn cb_text_source_set_default_font(ctx: &mut CpuContext) -> HleResult {
    let (ts, f) = (ctx.rdi(), ctx.rsi());
    if !text_source_magic(ctx, ts) {
        return ERROR_INVALID_PARAMETER;
    }
    ctx.write_u64(ts + 0x30, f);
    ctx.write_u64(ts + 0x58, f);
    0
}

/// sceFontTextSourceSetWritingForm(FontTextSource*, int writingForm)
pub(super) fn cb_text_source_set_writing_form(ctx: &mut CpuContext) -> HleResult {
    let (ts, form) = (ctx.rdi(), ctx.rsi() as i32);
    if !text_source_magic(ctx, ts) {
        return ERROR_INVALID_PARAMETER;
    }
    if form < WRITING_FORM_HORIZONTAL as i32 || form > WRITING_FORM_HORIZONTAL_LTR as i32 {
        return ERROR_INVALID_PARAMETER;
    }
    ctx.write_u32(ts + 4, form as u32);
    0
}

// ---------------------------------------------------------------------------
// Strings
// ---------------------------------------------------------------------------

fn write_chars(ctx: &CpuContext, block: u64, chars: &[Char]) -> bool {
    let mut b = vec![0u8; chars.len() * CHAR_SIZE as usize];
    for (i, c) in chars.iter().enumerate() {
        let o = i * CHAR_SIZE as usize;
        let prev = if i == 0 { 0 } else { block + (i as u64 - 1) * CHAR_SIZE };
        let next = if i + 1 < chars.len() { block + (i as u64 + 1) * CHAR_SIZE } else { 0 };
        let put = |b: &mut Vec<u8>, off: u64, v: &[u8]| b[o + off as usize..o + off as usize + v.len()].copy_from_slice(v);
        put(&mut b, CHAR_PREV, &prev.to_le_bytes());
        put(&mut b, CHAR_NEXT, &next.to_le_bytes());
        put(&mut b, CHAR_ORDER, &c.order.to_le_bytes());
        put(&mut b, CHAR_FONT, &c.font.to_le_bytes());
        put(&mut b, CHAR_CODE, &c.code.to_le_bytes());
        put(&mut b, CHAR_CLUSTER_SPAN, &[1]);
        put(&mut b, CHAR_CLUSTER_INDEX, &[c.cluster_index]);
        put(&mut b, CHAR_CLUSTER_KIND, &[c.cluster_kind]);
        put(&mut b, CHAR_SYNTHETIC, &[c.synthetic as u8]);
        put(&mut b, CHAR_FLAGS, &c.flags.to_le_bytes());
    }
    ctx.write_bytes(block, &b)
}

/// sceFontCreateString(const FontMemory*, FontTextSource*, const FontCreateStringDetail*, FontString*)
pub(super) fn cb_create_string(ctx: &mut CpuContext) -> HleResult {
    let (memory, ts, detail, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    let parser = if ts != 0 { ctx.read_u64(ts + 0x20).unwrap_or(0) } else { 0 };
    if memory == 0 || ts == 0 || out == 0 || parser == 0 {
        if out != 0 {
            ctx.write_u64(out, 0);
        }
        return ERROR_INVALID_PARAMETER;
    }
    ctx.write_u64(out, 0);
    if !text_source_magic(ctx, ts) {
        return ERROR_INVALID_TEXT_SOURCE;
    }
    if !valid_memory(ctx, memory) {
        return ERROR_INVALID_MEMORY;
    }
    let mut default_font = ctx.read_u64(ts + 0x30).unwrap_or(0);
    if detail != 0 {
        let Some(d) = ctx.read_bytes(detail, 0x10) else {
            return ERROR_INVALID_PARAMETER;
        };
        let valid_orders = 1u32 | 2 | 4 | (1 << 31);
        let orders = u32::from_le_bytes(d[4..8].try_into().unwrap());
        if u16::from_le_bytes([d[0], d[1]]) != CREATE_STRING_DETAIL_ID || d[2] > 1 || d[3] & !7 != 0 || orders & !valid_orders != 0 {
            return ERROR_INVALID_PARAMETER;
        }
        if default_font == 0 {
            default_font = u64::from_le_bytes(d[8..16].try_into().unwrap());
        }
    }
    let writing_form = ctx.read_u32(ts + 4).unwrap_or(WRITING_FORM_HORIZONTAL);
    // A plain horizontal form is decided by the characters themselves
    // (measured on hardware).
    let writing_form = if writing_form == WRITING_FORM_HORIZONTAL { None } else { Some(writing_form) };

    // Scratch: textOrder slot at +0, parse result at +0x10.
    let Some(scratch) = galloc(ctx, 0x40) else {
        return ERROR_ALLOCATION_FAILED;
    };
    let mut chars = Vec::new();
    let mut terminate_code = 0;
    let mut terminate_order = 0;
    let mut rc = ERROR_INVALID_TEXT_SOURCE;
    for _ in 0..MAX_PARSED_CHARACTERS {
        ctx.write_bytes(scratch + 0x10, &[0u8; 0x20]);
        let state = guest_call(parser, [ts, scratch, scratch + 0x10, 0, 0, 0]) as i32;
        let order = ctx.read_u64(scratch).unwrap_or(0);
        let r0 = ctx.read_u64(scratch + 0x10).unwrap_or(0);
        let r1 = ctx.read_u32(scratch + 0x18).unwrap_or(0);
        match state {
            PARSER_RESULT_FONT_CODE => {
                let f = if r0 != 0 { r0 } else { default_font };
                if f == 0 {
                    rc = ERROR_INVALID_TEXT_SOURCE;
                    break;
                }
                chars.push(make_char(f, r1, order));
                continue;
            }
            PARSER_RESULT_TERMINATE => {
                terminate_code = r1;
                terminate_order = if order != 0 { order } else { r0 };
                rc = 0;
            }
            PARSER_RESULT_ERROR => {
                rc = if (r1 as i32) < 0 { r1 as i32 as HleResult } else { ERROR_INVALID_TEXT_SOURCE };
            }
            _ => rc = ERROR_INVALID_TEXT_SOURCE,
        }
        break;
    }
    gfree(scratch);
    if rc != 0 {
        return rc;
    }
    let Some(object) = galloc(ctx, STRING_OBJECT_SIZE) else {
        return ERROR_ALLOCATION_FAILED;
    };
    let block = if chars.is_empty() {
        0
    } else {
        match galloc(ctx, chars.len() as u64 * CHAR_SIZE) {
            Some(b) => b,
            None => {
                gfree(object);
                return ERROR_ALLOCATION_FAILED;
            }
        }
    };
    if block != 0 && !write_chars(ctx, block, &chars) {
        gfree(block);
        gfree(object);
        return ERROR_ALLOCATION_FAILED;
    }
    let writing_form = writing_form.unwrap_or_else(|| default_string_form(&chars));
    let mut hdr = [0u8; 0x20];
    hdr[0..2].copy_from_slice(&FONT_STRING_MAGIC.to_le_bytes());
    hdr[4..8].copy_from_slice(&writing_form.to_le_bytes());
    hdr[8..16].copy_from_slice(&block.to_le_bytes());
    hdr[16..20].copy_from_slice(&(chars.len() as u32).to_le_bytes());
    ctx.write_bytes(object, &hdr);
    let mut t = text();
    t.retain_block(block);
    t.strings.insert(object, StringData { block, chars, writing_form, default_font, terminate_code, terminate_order });
    drop(t);
    ctx.write_u64(out, object);
    0
}

/// sceFontDestroyString(FontString*)
pub(super) fn cb_destroy_string(ctx: &mut CpuContext) -> HleResult {
    let p = ctx.rdi();
    if p == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let s = ctx.read_u64(p).unwrap_or(0);
    let mut t = text();
    let Some(data) = t.strings.remove(&s) else {
        drop(t);
        ctx.write_u64(p, 0);
        return ERROR_INVALID_STRING;
    };
    t.release_block(data.block);
    drop(t);
    write_u16(ctx, s, 0);
    gfree(s);
    ctx.write_u64(p, 0);
    0
}

/// uint32 sceFontStringGetTerminateCode(FontString)
pub(super) fn cb_string_get_terminate_code(ctx: &mut CpuContext) -> HleResult {
    text().strings.get(&ctx.rdi()).map_or(0, |s| s.terminate_code as HleResult)
}

/// void* sceFontStringGetTerminateOrder(FontString)
pub(super) fn cb_string_get_terminate_order(ctx: &mut CpuContext) -> HleResult {
    text().strings.get(&ctx.rdi()).map_or(0, |s| s.terminate_order as HleResult)
}

/// int sceFontStringGetWritingForm(FontString)
pub(super) fn cb_string_get_writing_form(ctx: &mut CpuContext) -> HleResult {
    text().strings.get(&ctx.rdi()).map_or(0, |s| s.writing_form as HleResult)
}

/// FontTextCharacter* sceFontStringRefersTextCharacters(FontString, uint32* count)
pub(super) fn cb_string_refers_text_characters(ctx: &mut CpuContext) -> HleResult {
    let (s, count) = (ctx.rdi(), ctx.rsi());
    let (block, n) = text().strings.get(&s).map_or((0, 0), |d| (d.block, d.chars.len() as u32));
    if count != 0 {
        ctx.write_u32(count, n);
    }
    if n == 0 {
        0
    } else {
        block as HleResult
    }
}

/// Index of a guest character pointer inside `block`.
fn char_index(block: u64, n: usize, ptr: u64) -> Option<usize> {
    if block == 0 || ptr < block || (ptr - block) % CHAR_SIZE != 0 {
        return None;
    }
    let i = ((ptr - block) / CHAR_SIZE) as usize;
    (i < n).then_some(i)
}

/// Visual-run selection over `chars` (bidi levels, clusters, synthetic
/// characters). Returns (first index, count).
fn render_characters(chars: &[Char], writing_form: u32, start: usize, last: Option<usize>) -> Option<(usize, u32)> {
    let prev = |i: usize| i.checked_sub(1);
    let next = |i: usize| (i + 1 < chars.len()).then_some(i + 1);
    let mut start = start;
    let mut last = last;
    if chars[start].cluster_kind > 1 {
        let mut probe = Some(start);
        while let Some(p) = probe {
            start = p;
            probe = prev(start);
            match probe {
                Some(q) if chars[q].synthetic < 0 => continue,
                _ => break,
            }
        }
    }
    let mut start_flags = chars[start].flags as u32;
    let paragraph = (writing_form >> 8) & 1;
    if start_flags & 0xFF00 == 0x100 {
        if let Some(p) = prev(start) {
            if chars[p].cluster_kind == 1 {
                start = p;
            }
            start_flags = chars[start].flags as u32;
        }
    }
    let mut start_level = start_flags >> 24;
    let mut last_level = start_level;
    let mut min_level = start_level;
    let mut count: u32 = if chars[start].cluster_index == 0 { 1 } else { 0 };
    let mut probe = start;
    while Some(probe) != last {
        match next(probe) {
            None => {
                if last.is_some() {
                    return None;
                }
                break;
            }
            Some(p) => {
                probe = p;
                if chars[p].cluster_index == 0 {
                    last_level = chars[p].level();
                    if last_level < min_level && (last_level ^ paragraph) & 1 == 0 {
                        min_level = last_level;
                    }
                    count += 1;
                }
            }
        }
    }
    if start_level != last_level || start_level != min_level {
        let mut p = prev(start);
        while let Some(q) = p {
            if chars[q].cluster_index == 0 {
                if chars[q].level() <= min_level {
                    break;
                }
                count += 1;
            }
            start = q;
            p = prev(q);
        }
        if let Some(l) = last {
            let mut p = next(l);
            while let Some(q) = p {
                if chars[q].cluster_index == 0 {
                    if chars[q].level() <= min_level {
                        break;
                    }
                    count += 1;
                    last = Some(q);
                }
                p = next(q);
            }
        }
    }
    if let Some(l) = last {
        let mut p = next(l);
        let mut trim = 0u32;
        while let Some(q) = p {
            let mut next_trim = trim;
            if chars[q].cluster_index == 0 {
                if chars[q].synthetic == 0 {
                    break;
                }
                next_trim = 0;
                let mut next_last = q;
                if chars[q].synthetic < 1 {
                    if chars[q].cluster_kind == 1 {
                        break;
                    }
                    next_trim = trim + 1;
                    next_last = last.unwrap();
                }
                last = Some(next_last);
                count += 1;
            }
            p = next(q);
            count = count.wrapping_sub(next_trim);
            trim = next_trim;
        }
    }
    start_level = chars[start].level();
    let mut result = Some(start);
    if start_level != paragraph {
        let mut s = start;
        let mut probe = start;
        if last.map_or(true, |l| chars[l].level() != start_level) {
            loop {
                match next(probe) {
                    None => {
                        if last.is_some() {
                            result = None;
                        }
                        break;
                    }
                    Some(p) => {
                        probe = p;
                        let mut best = s;
                        let mut best_level = start_level;
                        if chars[p].cluster_index == 0 {
                            let pl = chars[p].level();
                            if pl <= paragraph {
                                break;
                            }
                            if (start_level & 1) == paragraph && pl < start_level && (pl & 1) != paragraph {
                                best = p;
                                best_level = pl;
                            }
                        }
                        start_level = best_level;
                        s = best;
                        if Some(probe) == last {
                            break;
                        }
                    }
                }
            }
            if result.is_some() {
                result = Some(s);
            }
        } else if (chars[start].level() & 1) != paragraph {
            result = last;
        }
    }
    match result {
        Some(r) => Some((r, count)),
        None => Some((0, u32::MAX)),
    }
}

/// const FontTextCharacter* sceFontStringRefersRenderCharacters(FontString,
/// FontTextCharacter* start, FontTextCharacter* last, uint32* count)
pub(super) fn cb_string_refers_render_characters(ctx: &mut CpuContext) -> HleResult {
    let (s, start, last, count) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    let t = text();
    let Some(d) = t.strings.get(&s).filter(|d| !d.chars.is_empty()) else {
        drop(t);
        if count != 0 {
            ctx.write_u32(count, 0);
        }
        return 0;
    };
    if count == 0 {
        return 0;
    }
    let n = d.chars.len();
    let start_i = if start == 0 { Some(0) } else { char_index(d.block, n, start) };
    let last_i = if last == 0 { None } else { char_index(d.block, n, last) };
    let r = match (start_i, last != 0 && last_i.is_none()) {
        (Some(si), false) => render_characters(&d.chars, d.writing_form, si, last_i),
        _ => None,
    };
    let block = d.block;
    drop(t);
    match r {
        Some((i, c)) if c != u32::MAX => {
            ctx.write_u32(count, c);
            (block + i as u64 * CHAR_SIZE) as HleResult
        }
        _ => {
            ctx.write_u32(count, 0);
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Characters (read straight from the guest records)
// ---------------------------------------------------------------------------

/// sceFontCharacterGetBidiLevel(const FontTextCharacter*, int* level)
pub(super) fn cb_character_get_bidi_level(ctx: &mut CpuContext) -> HleResult {
    let (c, out) = (ctx.rdi(), ctx.rsi());
    if c == 0 || out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let flags = ctx.read_u64(c + CHAR_FLAGS).unwrap_or(0);
    ctx.write_u32(out, ((flags >> 24) & 0xFF) as u32);
    0
}

/// sceFontCharacterGetTextFontCode(const FontTextCharacter*, FontHandle*, uint32* code)
pub(super) fn cb_character_get_text_font_code(ctx: &mut CpuContext) -> HleResult {
    let (c, pf, pc) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if c == 0 {
        if pf != 0 {
            ctx.write_u64(pf, 0);
        }
        if pc != 0 {
            ctx.write_u32(pc, 0);
        }
        return ERROR_INVALID_PARAMETER;
    }
    if pf == 0 || pc == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    ctx.write_u64(pf, ctx.read_u64(c + CHAR_FONT).unwrap_or(0));
    ctx.write_u32(pc, ctx.read_u32(c + CHAR_CODE).unwrap_or(0));
    0
}

/// sceFontCharacterGetTextOrder(const FontTextCharacter*, void** order)
pub(super) fn cb_character_get_text_order(ctx: &mut CpuContext) -> HleResult {
    let (c, out) = (ctx.rdi(), ctx.rsi());
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    if c == 0 {
        ctx.write_u64(out, 0);
        return ERROR_INVALID_PARAMETER;
    }
    ctx.write_u64(out, ctx.read_u64(c + CHAR_ORDER).unwrap_or(0));
    0
}

/// uint32 sceFontCharacterLooksWhiteSpace(const FontTextCharacter*)
pub(super) fn cb_character_looks_white_space(ctx: &mut CpuContext) -> HleResult {
    let c = ctx.rdi();
    if c == 0 {
        return 0;
    }
    let flags = ctx.read_u64(c + CHAR_FLAGS).unwrap_or(0);
    if (flags >> 8) & 0xFF == WHITESPACE_CLASS {
        ctx.read_u32(c + CHAR_CODE).unwrap_or(0) as HleResult
    } else {
        0
    }
}

/// uint32 sceFontCharacterLooksFormatCharacters(const FontTextCharacter*):
/// the code of an invisible formatting character (ZWJ, bidi marks, ...).
pub(super) fn cb_character_looks_format_characters(ctx: &mut CpuContext) -> HleResult {
    let c = ctx.rdi();
    if c == 0 {
        return 0;
    }
    let flags = ctx.read_u64(c + CHAR_FLAGS).unwrap_or(0);
    if flags & CHARACTER_FLAG_FORMAT != 0 {
        ctx.read_u32(c + CHAR_CODE).unwrap_or(0) as HleResult
    } else {
        0
    }
}

/// sceFontCharacterGetSyllableStringState(const FontTextCharacter*, int* state):
/// no syllable shaping, every character is a complete syllable.
pub(super) fn cb_character_get_syllable_string_state(ctx: &mut CpuContext) -> HleResult {
    let (c, out) = (ctx.rdi(), ctx.rsi());
    if c == 0 || out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    ctx.write_u32(out, 0);
    0
}

/// FontTextCharacter* sceFontCharacterRefersTextNext(const FontTextCharacter*)
pub(super) fn cb_character_refers_text_next(ctx: &mut CpuContext) -> HleResult {
    refers_text(ctx, ctx.rdi(), CHAR_NEXT)
}

/// FontTextCharacter* sceFontCharacterRefersTextBack(const FontTextCharacter*)
pub(super) fn cb_character_refers_text_back(ctx: &mut CpuContext) -> HleResult {
    refers_text(ctx, ctx.rdi(), CHAR_PREV)
}

/// The nearest cluster-leading, non-synthetic character in direction `link`.
fn refers_text(ctx: &CpuContext, mut c: u64, link: u64) -> HleResult {
    if c == 0 {
        return 0;
    }
    for _ in 0..=MAX_PARSED_CHARACTERS {
        c = ctx.read_u64(c + link).unwrap_or(0);
        if c == 0 {
            return 0;
        }
        let b = ctx.read_bytes(c + CHAR_CLUSTER_INDEX, 3).unwrap_or_default();
        if b.len() == 3 && b[2] == 0 && b[0] == 0 {
            return c as HleResult;
        }
    }
    0
}

// FontTextCodes (0x40): { order ptr, code u32, reserved u32 } overlaid on
// systemReserved[8]; [2] = current character, [3] = terminating character.
const CODES_ORDER: u64 = 0x00;
const CODES_CODE: u64 = 0x08;
const CODES_CURRENT: u64 = 0x10;
const CODES_TERM: u64 = 0x18;
const CODES_SIZE: usize = 0x40;

fn fill_text_codes(ctx: &CpuContext, codes: u64, c: u64, term: u64) {
    let mut b = [0u8; CODES_SIZE];
    let order = ctx.read_u64(c + CHAR_ORDER).unwrap_or(0);
    let code = ctx.read_u32(c + CHAR_CODE).unwrap_or(0);
    let put = |b: &mut [u8; CODES_SIZE], off: u64, v: &[u8]| b[off as usize..off as usize + v.len()].copy_from_slice(v);
    put(&mut b, CODES_ORDER, &order.to_le_bytes());
    put(&mut b, CODES_CODE, &code.to_le_bytes());
    put(&mut b, CODES_CURRENT, &c.to_le_bytes());
    put(&mut b, CODES_TERM, &term.to_le_bytes());
    ctx.write_bytes(codes, &b);
}

/// FontTextCodes* sceFontCharactersRefersTextCodes(const FontTextCharacter*,
/// const FontTextCharacter* term, FontTextCodes*)
pub(super) fn cb_characters_refers_text_codes(ctx: &mut CpuContext) -> HleResult {
    let (c, term, codes) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if c != 0 && codes != 0 {
        fill_text_codes(ctx, codes, c, term);
    }
    codes as HleResult
}

/// FontTextCodes* sceFontTextCodesStepNext(FontTextCodes*): the next
/// character's codes, NULL at the end or at the terminating character.
pub(super) fn cb_text_codes_step_next(ctx: &mut CpuContext) -> HleResult {
    step_text_codes(ctx, ctx.rdi(), CHAR_NEXT)
}

/// FontTextCodes* sceFontTextCodesStepBack(FontTextCodes*)
pub(super) fn cb_text_codes_step_back(ctx: &mut CpuContext) -> HleResult {
    step_text_codes(ctx, ctx.rdi(), CHAR_PREV)
}

fn step_text_codes(ctx: &CpuContext, codes: u64, link: u64) -> HleResult {
    if codes == 0 {
        return 0;
    }
    let current = ctx.read_u64(codes + CODES_CURRENT).unwrap_or(0);
    let term = ctx.read_u64(codes + CODES_TERM).unwrap_or(0);
    if current == 0 {
        return 0;
    }
    let c = ctx.read_u64(current + link).unwrap_or(0);
    if c == 0 || (link == CHAR_NEXT && c == term) {
        return 0;
    }
    fill_text_codes(ctx, codes, c, term);
    codes as HleResult
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

fn glyph_metrics_for(font_handle: u64, code: u32) -> Option<GlyphMetrics> {
    let f = font(font_handle)?;
    if f.renderer != 0 {
        if let Ok(m) = char_metrics(&f.face, code, &f.render_style) {
            return Some(m);
        }
    }
    char_metrics(&f.face, code, &f.style).ok()
}

impl WritingData {
    fn populate(&mut self, index: usize) -> bool {
        let Some(c) = self.chars.get(index).copied() else {
            return false;
        };
        let f = if c.font != 0 { c.font } else { self.default_font };
        let mut step = Step { x: self.pen_x, y: self.pen_y, font: f, profile: 1, glyph_code: c.code, ..Default::default() };
        if c.flags & CHARACTER_FLAG_FORMAT != 0 && self.mask_format {
            step.profile |= 1 << 8;
        }
        self.raw = None;
        if f != 0 && c.code != 0 {
            if let Some(m) = glyph_metrics_for(f, c.code) {
                self.raw = Some(m);
                step.advance_x = m[4];
                step.advance_y = m[7];
                step.metrics = m;
                step.pos_x = step.x;
                step.pos_y = step.y;
            }
        }
        self.pen_x += step.advance_x;
        self.pen_y += step.advance_y;
        self.step = step;
        true
    }

    fn accumulate(&mut self) {
        let g = self.raw.unwrap_or(self.step.metrics);
        let left = self.step.x + g[2];
        let right = left + g[0];
        let top = self.step.y - g[3];
        let bottom = top + g[1];
        let prev = self.metrics;
        self.metrics[0] = self.step.x + self.step.advance_x;
        self.metrics[1] = self.step.y + self.step.advance_y;
        if !self.has_aggregate {
            self.metrics[2..6].copy_from_slice(&[top, bottom, left, right]);
            self.has_aggregate = true;
            return;
        }
        self.metrics[2] = prev[2].min(top);
        self.metrics[3] = prev[3].max(bottom);
        self.metrics[4] = prev[4].min(left);
        self.metrics[5] = prev[5].max(right);
    }
}

/// sceFontWritingInit(FontWriting*, FontString, const FontTextCharacter*)
pub(super) fn cb_writing_init(ctx: &mut CpuContext) -> HleResult {
    let (w, s, c) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    if w == 0 || s == 0 || c == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let mut t = text();
    let Some(d) = t.strings.get(&s) else {
        return ERROR_INVALID_STRING;
    };
    let (block, chars, default_font) = (d.block, d.chars.clone(), d.default_font);
    if let Some(prev) = t.writings.remove(&w) {
        t.release_block(prev.block);
    }
    let Some(index) = char_index(block, chars.len(), c) else {
        drop(t);
        ctx.write_bytes(w, &[0u8; 0x100]);
        return ERROR_INVALID_PARAMETER;
    };
    drop(t);
    let mut wd = WritingData {
        block,
        chars,
        default_font,
        next_index: index,
        current_index: index,
        pen_x: 0.0,
        pen_y: 0.0,
        has_aggregate: false,
        has_current: false,
        has_current_character: false,
        mask_format: true,
        step: Step::default(),
        raw: None,
        metrics: [0.0; 6],
    };
    wd.populate(index);
    wd.pen_x = 0.0;
    wd.pen_y = 0.0;
    let mut hdr = [0u8; 0x100];
    hdr[0..2].copy_from_slice(&FONT_WRITING_MAGIC.to_le_bytes());
    hdr[8..16].copy_from_slice(&s.to_le_bytes());
    hdr[WRITING_STEP as usize..WRITING_STEP as usize + 0x48].copy_from_slice(&wd.step.bytes());
    if !ctx.write_bytes(w, &hdr) {
        return ERROR_INVALID_PARAMETER;
    }
    let mut t = text();
    t.retain_block(block);
    t.writings.insert(w, wd);
    0
}

fn write_writing_metrics(ctx: &CpuContext, addr: u64, m: &WritingMetrics) {
    write_floats(ctx, addr, m);
}

/// sceFontWritingSetMaskInvisible(FontWriting*, int mask): whether
/// formatting characters are reported as invisible glyphs.
pub(super) fn cb_writing_set_mask_invisible(ctx: &mut CpuContext) -> HleResult {
    let (w, mask) = (ctx.rdi(), ctx.rsi() as i32);
    if w == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    match text().writings.get_mut(&w) {
        Some(d) => {
            d.mask_format = mask != 0;
            0
        }
        None => ERROR_INVALID_WRITING,
    }
}

/// sceFontWritingGetRenderMetrics(FontWriting*, FontWritingMetrics*)
pub(super) fn cb_writing_get_render_metrics(ctx: &mut CpuContext) -> HleResult {
    let (w, out) = (ctx.rdi(), ctx.rsi());
    if w == 0 || out == 0 {
        if out != 0 {
            write_writing_metrics(ctx, out, &[0.0; 6]);
        }
        return ERROR_INVALID_PARAMETER;
    }
    let m = text().writings.get(&w).map(|d| d.metrics);
    match m {
        Some(m) => {
            write_writing_metrics(ctx, out, &m);
            0
        }
        None => {
            write_writing_metrics(ctx, out, &[0.0; 6]);
            ERROR_INVALID_WRITING
        }
    }
}

/// const FontWritingStep* sceFontWritingRefersRenderStep(FontWriting*)
pub(super) fn cb_writing_refers_render_step(ctx: &mut CpuContext) -> HleResult {
    let w = ctx.rdi();
    let mut t = text();
    let Some(d) = t.writings.get_mut(&w) else {
        return 0;
    };
    if d.next_index >= d.chars.len() || !d.populate(d.next_index) {
        return 0;
    }
    d.current_index = d.next_index;
    d.has_current = true;
    d.has_current_character = true;
    d.accumulate();
    d.next_index += 1;
    let (step, metrics) = (d.step.bytes(), d.metrics);
    drop(t);
    ctx.write_bytes(w + WRITING_STEP, &step);
    write_writing_metrics(ctx, w + WRITING_METRICS, &metrics);
    (w + WRITING_STEP) as HleResult
}

/// FontTextCharacter* sceFontWritingRefersRenderStepCharacter(FontWriting*,
/// const FontWritingLetterStep** letterStep): the letter step is the
/// writing's own record; `*letterStep` points at it (NULL without a step).
pub(super) fn cb_writing_refers_render_step_character(ctx: &mut CpuContext) -> HleResult {
    let (w, letter) = (ctx.rdi(), ctx.rsi());
    if letter != 0 {
        ctx.write_u64(letter, 0);
    }
    let mut t = text();
    let Some(d) = t.writings.get_mut(&w) else {
        return 0;
    };
    // Each render step hands out its character once; the next call answers
    // NULL until the next step (measured on the PS4 library, firmware
    // 11.00), leaving the letter step as it was.
    if !d.has_current || !d.has_current_character || d.current_index >= d.chars.len() {
        return 0;
    }
    d.has_current_character = false;
    let (step, cur, block) = (d.step, d.current_index as u32, d.block);
    drop(t);
    let mut b = [0u8; 0x40];
    for (i, v) in [step.x, step.y, step.advance_x, step.advance_y].iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in [1u32, cur, 1, cur].iter().enumerate() {
        b[0x10 + i * 4..0x14 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    b[0x25] = ((step.profile & 0xFF) as u8).max(1);
    let record = w + WRITING_LETTER_STEP;
    if ctx.write_bytes(record, &b) && letter != 0 {
        ctx.write_u64(letter, record);
    }
    (block + cur as u64 * CHAR_SIZE) as HleResult
}

// ---------------------------------------------------------------------------
// Writing lines
// ---------------------------------------------------------------------------

/// sceFontCreateWritingLine(const FontMemory*, int writingForm, const void* detail, FontWritingLine*)
pub(super) fn cb_create_writing_line(ctx: &mut CpuContext) -> HleResult {
    let (form, detail, out) = (ctx.rsi() as i32, ctx.rdx(), ctx.rcx());
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    ctx.write_u64(out, 0);
    if form < WRITING_FORM_HORIZONTAL as i32 || form > WRITING_FORM_HORIZONTAL_LTR as i32 {
        return ERROR_INVALID_PARAMETER;
    }
    if detail != 0 && read_u16(ctx, detail) != Some(CREATE_WRITING_LINE_DETAIL_ID) {
        return ERROR_INVALID_PARAMETER;
    }
    let Some(line) = galloc(ctx, LINE_OBJECT_SIZE) else {
        return ERROR_ALLOCATION_FAILED;
    };
    write_u16(ctx, line, WRITING_LINE_MAGIC);
    ctx.write_u32(line + 4, form as u32);
    text().lines.insert(line, LineData::new(form));
    ctx.write_u64(out, line);
    0
}

/// sceFontDestroyWritingLine(FontWritingLine*)
pub(super) fn cb_destroy_writing_line(ctx: &mut CpuContext) -> HleResult {
    let p = ctx.rdi();
    if p == 0 {
        return 0;
    }
    let line = ctx.read_u64(p).unwrap_or(0);
    if line == 0 {
        return 0;
    }
    if text().lines.remove(&line).is_none() {
        return ERROR_INVALID_PARAMETER;
    }
    write_u16(ctx, line, 0);
    gfree(line);
    ctx.write_u64(p, 0);
    0
}

/// sceFontWritingLineClear(FontWritingLine)
pub(super) fn cb_writing_line_clear(ctx: &mut CpuContext) -> HleResult {
    match text().lines.get_mut(&ctx.rdi()) {
        Some(d) => {
            d.reset_orders();
            0
        }
        None => ERROR_INVALID_PARAMETER,
    }
}

/// sceFontWritingLineWritesOrder(FontWritingLine, uint64 attribute,
/// const FontWritingMetrics*, void* writingOrderer)
///
/// The attribute carries the 0xFA id in its low byte (0 means plain
/// level 0), the embedding level in bits 8..15 and the space-before /
/// space-after flags in bits 30 and 31. Writing after a render walk has
/// started begins a new line.
pub(super) fn cb_writing_line_writes_order(ctx: &mut CpuContext) -> HleResult {
    let (line, attribute, metrics, orderer) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    if !text().lines.contains_key(&line) {
        return ERROR_INVALID_PARAMETER;
    }
    let attribute = if attribute == 0 { WRITING_ATTRIBUTE_ID } else { attribute };
    if attribute & WRITING_ATTRIBUTE_ID_MASK != WRITING_ATTRIBUTE_ID {
        return ERROR_INVALID_PARAMETER;
    }
    let level = ((attribute >> 8) & 0xFF) as usize;
    if level >= WRITING_LEVEL_COUNT || orderer == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let mut order_metrics = [0.0f32; 6];
    if metrics != 0 {
        let Some(raw) = ctx.read_bytes(metrics, 0x18) else {
            return ERROR_INVALID_PARAMETER;
        };
        order_metrics = std::array::from_fn(|i| f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap()));
    }
    let mut t = text();
    let Some(d) = t.lines.get_mut(&line) else {
        return ERROR_INVALID_PARAMETER;
    };
    if d.cursor.is_some() {
        d.reset_orders();
    }
    let o = d.orders.len();
    d.orders.push(LineOrder { orderer, attribute, metrics: order_metrics, ..Default::default() });
    d.link(o, level);
    if d.orders[o].prev.is_none() {
        d.head = Some(o);
    }
    if d.orders[o].next.is_none() {
        d.tail = Some(o);
    }
    d.ordering_space += d.ordering_space_delta(o);
    d.advance_x += order_metrics[M_ADVANCE_X];
    d.advance_y += order_metrics[M_ADVANCE_Y];
    d.last = Some(o);
    d.last_level = level;
    0
}

/// const FontWritingLineStep* sceFontWritingLineRefersRenderStep(FontWritingLine)
///
/// Every call returns the line's one step object, refilled with the next
/// order in reading order. Before moving on, the previous step (including
/// any Adjusting the caller wrote into it) is folded into the render
/// metrics. Past the last order the walk ends and NULL is returned; the
/// next call starts over.
pub(super) fn cb_writing_line_refers_render_step(ctx: &mut CpuContext) -> HleResult {
    let line = ctx.rdi();
    if line == 0 {
        return 0;
    }
    let mut t = text();
    let Some(d) = t.lines.get_mut(&line) else {
        return 0;
    };
    let Some(raw) = ctx.read_bytes(line + LINE_STEP, LINE_STEP_SIZE as usize) else {
        return 0;
    };
    let mut step = LineStep::from_bytes(&raw);
    let more = d.next_step(&mut step);
    drop(t);
    if !ctx.write_bytes(line + LINE_STEP, &step.bytes()) || !more {
        return 0;
    }
    (line + LINE_STEP) as HleResult
}

/// sceFontWritingLineGetRenderMetrics(FontWritingLine, FontWritingMetrics*)
pub(super) fn cb_writing_line_get_render_metrics(ctx: &mut CpuContext) -> HleResult {
    let (line, out) = (ctx.rdi(), ctx.rsi());
    let Some(m) = text().lines.get(&line).map(|d| d.render_metrics) else {
        return ERROR_INVALID_PARAMETER;
    };
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    write_writing_metrics(ctx, out, &m);
    0
}

/// sceFontWritingLineGetOrderingSpace(FontWritingLine, float* advance,
/// float* head, float* inline, float* tail): total advance, the blank
/// before the leading order and after the trailing one, and the blank
/// around spaced joins inside the line. Head and tail are written as a
/// pair; any pointer may be NULL.
pub(super) fn cb_writing_line_get_ordering_space(ctx: &mut CpuContext) -> HleResult {
    let line = ctx.rdi();
    let (advance, head, inline, tail) = (ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8());
    let t = text();
    let Some(d) = t.lines.get(&line) else {
        return ERROR_INVALID_PARAMETER;
    };
    if head != 0 && tail == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let rtl = d.right_to_left();
    let (leading, trailing) = if rtl { (d.tail, d.head) } else { (d.head, d.tail) };
    let m = |o: Option<usize>| o.map(|i| d.orders[i].metrics);
    let (head_space, tail_space) = match (rtl, m(leading), m(trailing)) {
        (true, l, r) => (l.map_or(0.0, |l| l[M_RIGHT]), r.map_or(0.0, |r| r[M_ADVANCE_X] - r[M_LEFT])),
        (false, l, r) => (l.map_or(0.0, |l| l[M_LEFT]), r.map_or(0.0, |r| r[M_ADVANCE_X] - r[M_RIGHT])),
    };
    let (advance_x, ordering_space) = (d.advance_x, d.ordering_space);
    drop(t);
    if advance != 0 {
        ctx.write_f32(advance, advance_x);
    }
    if head != 0 {
        ctx.write_f32(head, head_space);
        ctx.write_f32(tail, tail_space);
    }
    if inline != 0 {
        ctx.write_f32(inline, ordering_space);
    }
    0
}

pub(super) const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("6DFUkCwQLa8", "sceFontCharacterGetBidiLevel", cb_character_get_bidi_level),
    ("coCrV6IWplE", "sceFontCharacterGetSyllableStringState", cb_character_get_syllable_string_state),
    ("zN3+nuA0SFQ", "sceFontCharacterGetTextFontCode", cb_character_get_text_font_code),
    ("mxgmMj-Mq-o", "sceFontCharacterGetTextOrder", cb_character_get_text_order),
    ("-P6X35Rq2-E", "sceFontCharacterLooksFormatCharacters", cb_character_looks_format_characters),
    ("SaRlqtqaCew", "sceFontCharacterLooksWhiteSpace", cb_character_looks_white_space),
    ("6Gqlv5KdTbU", "sceFontCharacterRefersTextBack", cb_character_refers_text_back),
    ("BkjBP+YC19w", "sceFontCharacterRefersTextNext", cb_character_refers_text_next),
    ("lVSR5ftvNag", "sceFontCharactersRefersTextCodes", cb_characters_refers_text_codes),
    ("+B-xlbiWDJ4", "sceFontStringGetTerminateOrder", cb_string_get_terminate_order),
    ("IPoYwwlMx-g", "sceFontTextCodesStepBack", cb_text_codes_step_back),
    ("olSmXY+XP1E", "sceFontTextCodesStepNext", cb_text_codes_step_next),
    ("VRFd3diReec", "sceFontTextSourceRewind", cb_text_source_rewind),
    ("BbCZjJizU4A", "sceFontWritingSetMaskInvisible", cb_writing_set_mask_invisible),
    ("MO24vDhmS4E", "sceFontCreateString", cb_create_string),
    ("7rogx92EEyc", "sceFontCreateWritingLine", cb_create_writing_line),
    ("SSCaczu2aMQ", "sceFontDestroyString", cb_destroy_string),
    ("PEjv7CVDRYs", "sceFontDestroyWritingLine", cb_destroy_writing_line),
    ("ObkDGDBsVtw", "sceFontStringGetTerminateCode", cb_string_get_terminate_code),
    ("o1vIEHeb6tw", "sceFontStringGetWritingForm", cb_string_get_writing_form),
    ("hq5LffQjz-s", "sceFontStringRefersRenderCharacters", cb_string_refers_render_characters),
    ("Avv7OApgCJk", "sceFontStringRefersTextCharacters", cb_string_refers_text_characters),
    ("oaJ1BpN2FQk", "sceFontTextSourceInit", cb_text_source_init),
    ("eCRMCSk96NU", "sceFontTextSourceSetDefaultFont", cb_text_source_set_default_font),
    ("OqQKX0h5COw", "sceFontTextSourceSetWritingForm", cb_text_source_set_writing_form),
    ("fljdejMcG1c", "sceFontWritingGetRenderMetrics", cb_writing_get_render_metrics),
    ("fD5rqhEXKYQ", "sceFontWritingInit", cb_writing_init),
    ("1+DgKL0haWQ", "sceFontWritingLineClear", cb_writing_line_clear),
    ("JQKWIsS9joE", "sceFontWritingLineGetOrderingSpace", cb_writing_line_get_ordering_space),
    ("nlU2VnfpqTM", "sceFontWritingLineGetRenderMetrics", cb_writing_line_get_render_metrics),
    ("+FYcYefsVX0", "sceFontWritingLineRefersRenderStep", cb_writing_line_refers_render_step),
    ("wyKFUOWdu3Q", "sceFontWritingLineWritesOrder", cb_writing_line_writes_order),
    ("f4Onl7efPEY", "sceFontWritingRefersRenderStepCharacter", cb_writing_refers_render_step_character),
    ("W-2WOXEHGck", "sceFontWritingRefersRenderStep", cb_writing_refers_render_step),
];
