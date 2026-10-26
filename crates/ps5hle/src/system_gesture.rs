// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceSystemGesture: touch-pad gesture recognition.
//!
//! One gesture handle exists, for the touch pad. The primitive recognizer
//! tracks every touch of the pad reports the title passes to
//! sceSystemGestureUpdatePrimitiveTouchRecognizer (begin, active and end
//! events). Gesture recognizers never produce events: their queries report
//! zero events (or EVENT_DATA_NOT_FOUND for a lookup) after clearing the
//! caller's buffer, as do the primitive queries with no touch tracked.

use std::sync::{Mutex, MutexGuard};

use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};

const LIB: &str = "libSceSystemGesture";
const OK: HleResult = 0;

const fn err(code: u32) -> HleResult {
    code as i32 as i64
}

pub const ERR_INVALID_ARGUMENT: HleResult = err(0x8089_0001);
pub const ERR_EVENT_DATA_NOT_FOUND: HleResult = err(0x8089_0005);
pub const ERR_INVALID_HANDLE: HleResult = err(0x8089_0006);

const INPUT_TOUCH_PAD: i32 = 0;
/// The touch-pad gesture handle.
pub const HANDLE: i32 = 1;

/// SceSystemGesturePrimitiveTouchEvent.
pub const PRIMITIVE_TOUCH_EVENT_SIZE: usize = 80;
/// SceSystemGestureTouchRecognizer (361 x u64).
pub const TOUCH_RECOGNIZER_SIZE: usize = 361 * 8;
/// SceSystemGestureTouchRecognizerInformation.
pub const TOUCH_RECOGNIZER_INFORMATION_SIZE: usize = 296;
/// SceSystemGestureTouchEvent.
pub const TOUCH_EVENT_SIZE: usize = 168;

const TOUCH_STATE_BEGIN: i32 = 1;
const TOUCH_STATE_ACTIVE: i32 = 2;
const TOUCH_STATE_END: i32 = 3;

/// One touch followed by the primitive recognizer.
#[derive(Clone, Copy, Default)]
struct TrackedTouch {
    primitive_id: u16,
    event_state: i32,
    is_updated: bool,
    sampled: bool,
    pressed: [f32; 2],
    current: [f32; 2],
    delta: [f32; 2],
    pressed_time: u64,
    updated_time: u64,
    delta_time: u64,
}

impl TrackedTouch {
    /// SceSystemGesturePrimitiveTouchEvent: i32 state, u16 id, u8 updated,
    /// pressed / current / delta vectors (+8, +16, +24), u64 delta time
    /// (+32) and elapsed time (+40).
    fn event(&self) -> [u8; PRIMITIVE_TOUCH_EVENT_SIZE] {
        let mut e = [0u8; PRIMITIVE_TOUCH_EVENT_SIZE];
        e[0..4].copy_from_slice(&self.event_state.to_le_bytes());
        e[4..6].copy_from_slice(&self.primitive_id.to_le_bytes());
        e[6] = self.is_updated as u8;
        for (i, v) in [self.pressed, self.current, self.delta].iter().enumerate() {
            e[8 + i * 8..12 + i * 8].copy_from_slice(&v[0].to_le_bytes());
            e[12 + i * 8..16 + i * 8].copy_from_slice(&v[1].to_le_bytes());
        }
        e[32..40].copy_from_slice(&self.delta_time.to_le_bytes());
        e[40..48].copy_from_slice(&self.updated_time.wrapping_sub(self.pressed_time).to_le_bytes());
        e
    }
}

static TOUCHES: Mutex<Vec<TrackedTouch>> = Mutex::new(Vec::new());

fn touches() -> MutexGuard<'static, Vec<TrackedTouch>> {
    TOUCHES.lock().unwrap_or_else(|p| p.into_inner())
}

/// The touches of one ScePadData report: (id, x, y) for each of
/// touch_num (+52) points at +60 and +68 (u16 x, u16 y, u8 id).
fn report_touches(report: &[u8]) -> Vec<(u16, [f32; 2])> {
    let count = report[52];
    let point = |off: usize| {
        let x = u16::from_le_bytes([report[off], report[off + 1]]) as f32;
        let y = u16::from_le_bytes([report[off + 2], report[off + 3]]) as f32;
        (report[off + 4] as u16, [x, y])
    };
    [60, 68].into_iter().take(count.min(2) as usize).map(point).collect()
}

/// Follow the touches of one pad report: ended touches are dropped, a new
/// id begins, a known one becomes active (updated when it moved or just
/// began), and a touch missing from the report ends.
fn sample_report(list: &mut Vec<TrackedTouch>, report: &[u8]) {
    list.retain(|t| t.event_state != TOUCH_STATE_END);
    for t in list.iter_mut() {
        t.sampled = false;
        t.is_updated = false;
    }
    let time = u64::from_le_bytes(report[80..88].try_into().unwrap());
    for (id, position) in report_touches(report) {
        let Some(t) = list.iter_mut().find(|t| t.primitive_id == id) else {
            list.push(TrackedTouch {
                primitive_id: id,
                event_state: TOUCH_STATE_BEGIN,
                is_updated: true,
                sampled: true,
                pressed: position,
                current: position,
                pressed_time: time,
                updated_time: time,
                ..Default::default()
            });
            continue;
        };
        t.delta = [position[0] - t.current[0], position[1] - t.current[1]];
        t.current = position;
        t.delta_time = time.saturating_sub(t.updated_time);
        t.is_updated = t.event_state != TOUCH_STATE_ACTIVE || t.delta != [0.0, 0.0];
        t.event_state = TOUCH_STATE_ACTIVE;
        t.sampled = true;
        t.updated_time = time;
    }
    for t in list.iter_mut().filter(|t| !t.sampled) {
        t.delta = [0.0, 0.0];
        t.delta_time = time.saturating_sub(t.updated_time);
        t.event_state = TOUCH_STATE_END;
        t.is_updated = true;
        t.updated_time = time;
    }
}

fn handle(ctx: &CpuContext) -> Result<(), HleResult> {
    if ctx.rdi() as i32 == HANDLE { Ok(()) } else { Err(ERR_INVALID_HANDLE) }
}

/// Zero `size` bytes at `addr` when it is non-null.
fn clear(ctx: &CpuContext, addr: u64, size: usize) -> Result<(), HleResult> {
    if addr != 0 && !ctx.write_bytes(addr, &vec![0u8; size]) {
        return Err(ERR_INVALID_ARGUMENT);
    }
    Ok(())
}

fn status(r: Result<(), HleResult>) -> HleResult {
    r.err().unwrap_or(OK)
}

/// (inputType, param) -> handle
fn cb_open(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() as i32 != INPUT_TOUCH_PAD {
        return ERR_INVALID_ARGUMENT;
    }
    HANDLE as HleResult
}

/// (handle, ...) with no other argument checked.
fn cb_handle_only(ctx: &mut CpuContext) -> HleResult {
    status(handle(ctx))
}

/// sceSystemGestureInitializePrimitiveTouchRecognizer(param) /
/// FinalizePrimitiveTouchRecognizer(): no touch is tracked.
fn cb_forget_touches(_ctx: &mut CpuContext) -> HleResult {
    touches().clear();
    OK
}

/// (handle): no touch is tracked.
fn cb_reset_primitive_touch_recognizer(ctx: &mut CpuContext) -> HleResult {
    status(handle(ctx).map(|()| touches().clear()))
}

/// (handle, const SceSystemGestureTouchPadData*): { i32 padHandle,
/// i32 reportNumber, const ScePadData* reports }; each report is followed
/// in order. No reports (or no param) changes nothing.
fn cb_update_primitive_touch_recognizer(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        handle(ctx)?;
        let param = ctx.rsi();
        if param == 0 {
            return Ok(());
        }
        let count = ctx.read_u32(param + 4).ok_or(ERR_INVALID_ARGUMENT)? as i32;
        let buffer = ctx.read_u64(param + 8).ok_or(ERR_INVALID_ARGUMENT)?;
        if buffer == 0 || count <= 0 {
            return Ok(());
        }
        let size = crate::pad::PAD_DATA_SIZE;
        let reports = ctx.read_bytes(buffer, count as usize * size).ok_or(ERR_INVALID_ARGUMENT)?;
        let mut list = touches();
        for report in reports.chunks_exact(size) {
            sample_report(&mut list, report);
        }
        Ok(())
    })())
}

/// (handle, buffer, capacity, uint32_t* numberOfEvent): the events of the
/// tracked touches that fit; with none, the first entry is cleared.
fn cb_get_primitive_touch_events(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        handle(ctx)?;
        let (buffer, capacity, number) = (ctx.rsi(), ctx.rdx() as u32, ctx.rcx());
        if number == 0 {
            return Err(ERR_INVALID_ARGUMENT);
        }
        let list = touches().clone();
        let reported = list.len().min(capacity as usize);
        if reported > 0 && buffer == 0 {
            return Err(ERR_INVALID_ARGUMENT);
        }
        for (i, t) in list.iter().take(reported).enumerate() {
            if !ctx.write_bytes(buffer + (i * PRIMITIVE_TOUCH_EVENT_SIZE) as u64, &t.event()) {
                return Err(ERR_INVALID_ARGUMENT);
            }
        }
        if !ctx.write_u32(number, reported as u32) {
            return Err(ERR_INVALID_ARGUMENT);
        }
        if reported == 0 && capacity != 0 {
            clear(ctx, buffer, PRIMITIVE_TOUCH_EVENT_SIZE)?;
        }
        Ok(())
    })())
}

/// (handle) -> number of tracked touches
fn cb_get_primitive_touch_events_count(ctx: &mut CpuContext) -> HleResult {
    match handle(ctx) {
        Ok(()) => touches().len() as HleResult,
        Err(e) => e,
    }
}

/// Write the event of a found touch; otherwise clear the output and answer
/// EVENT_DATA_NOT_FOUND.
fn primitive_lookup(ctx: &CpuContext, found: Option<TrackedTouch>) -> HleResult {
    status((|| {
        handle(ctx)?;
        let out = ctx.rdx();
        match found {
            Some(t) if out != 0 => {
                if ctx.write_bytes(out, &t.event()) { Ok(()) } else { Err(ERR_INVALID_ARGUMENT) }
            }
            Some(_) => Err(ERR_INVALID_ARGUMENT),
            None => {
                clear(ctx, out, PRIMITIVE_TOUCH_EVENT_SIZE)?;
                Err(ERR_EVENT_DATA_NOT_FOUND)
            }
        }
    })())
}

/// (handle, u32 index, event*)
fn cb_get_primitive_touch_event_by_index(ctx: &mut CpuContext) -> HleResult {
    let found = touches().get(ctx.rsi() as u32 as usize).copied();
    primitive_lookup(ctx, found)
}

/// (handle, u16 primitiveId, event*)
fn cb_get_primitive_touch_event_by_primitive_id(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rsi() as u16;
    let found = touches().iter().find(|t| t.primitive_id == id).copied();
    primitive_lookup(ctx, found)
}

/// (handle, recognizer*, gestureType, const rectangle*, param)
fn cb_create_touch_recognizer(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        handle(ctx)?;
        let recognizer = ctx.rsi();
        if recognizer == 0 {
            return Err(ERR_INVALID_ARGUMENT);
        }
        clear(ctx, recognizer, TOUCH_RECOGNIZER_SIZE)
    })())
}

/// (handle, recognizer*): append/remove/update need a recognizer.
fn cb_recognizer_required(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        handle(ctx)?;
        if ctx.rsi() == 0 { Err(ERR_INVALID_ARGUMENT) } else { Ok(()) }
    })())
}

/// (handle, recognizer*): the recognizer is cleared.
fn cb_reset_touch_recognizer(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        handle(ctx)?;
        let recognizer = ctx.rsi();
        if recognizer == 0 {
            return Err(ERR_INVALID_ARGUMENT);
        }
        clear(ctx, recognizer, TOUCH_RECOGNIZER_SIZE)
    })())
}

/// (handle, const recognizer*, information*)
fn cb_get_touch_recognizer_information(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        handle(ctx)?;
        let (recognizer, info) = (ctx.rsi(), ctx.rdx());
        if recognizer == 0 || info == 0 {
            return Err(ERR_INVALID_ARGUMENT);
        }
        clear(ctx, info, TOUCH_RECOGNIZER_INFORMATION_SIZE)
    })())
}

/// (handle, recognizer*, const rectangle*)
fn cb_update_touch_recognizer_rectangle(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        handle(ctx)?;
        if ctx.rsi() == 0 || ctx.rdx() == 0 { Err(ERR_INVALID_ARGUMENT) } else { Ok(()) }
    })())
}

/// (handle, const recognizer*, buffer, capacity, uint32_t* numberOfEvent)
fn cb_get_touch_events(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        handle(ctx)?;
        let (recognizer, buffer, capacity, number) = (ctx.rsi(), ctx.rdx(), ctx.rcx() as u32, ctx.r8());
        if recognizer == 0 || number == 0 {
            return Err(ERR_INVALID_ARGUMENT);
        }
        if !ctx.write_u32(number, 0) {
            return Err(ERR_INVALID_ARGUMENT);
        }
        if capacity != 0 {
            clear(ctx, buffer, TOUCH_EVENT_SIZE)?;
        }
        Ok(())
    })())
}

/// (handle, const recognizer*) -> number of events
fn cb_get_touch_events_count(ctx: &mut CpuContext) -> HleResult {
    match handle(ctx) {
        Err(e) => e,
        Ok(()) if ctx.rsi() == 0 => ERR_INVALID_ARGUMENT,
        Ok(()) => 0,
    }
}

/// (handle, const recognizer*, index or event id, event*)
fn cb_get_touch_event_lookup(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        handle(ctx)?;
        if ctx.rsi() == 0 {
            return Err(ERR_INVALID_ARGUMENT);
        }
        clear(ctx, ctx.rcx(), TOUCH_EVENT_SIZE)?;
        Err(ERR_EVENT_DATA_NOT_FOUND)
    })())
}

const EXPORTS: &[(&str, &str, crate::registry::HleCallback)] = &[
    ("qpo-mEOwje0", "sceSystemGestureOpen", cb_open),
    ("j4yXIA2jJ68", "sceSystemGestureClose", cb_handle_only),
    ("3pcAvmwKCvM", "sceSystemGestureInitializePrimitiveTouchRecognizer", cb_forget_touches),
    ("3QYCmMlOlCY", "sceSystemGestureFinalizePrimitiveTouchRecognizer", cb_forget_touches),
    ("o11J529VaAE", "sceSystemGestureResetPrimitiveTouchRecognizer", cb_reset_primitive_touch_recognizer),
    ("GgFMb22sbbI", "sceSystemGestureUpdatePrimitiveTouchRecognizer", cb_update_primitive_touch_recognizer),
    ("L8YmemOeSNY", "sceSystemGestureGetPrimitiveTouchEvents", cb_get_primitive_touch_events),
    ("JhwByySf9FY", "sceSystemGestureGetPrimitiveTouchEventsCount", cb_get_primitive_touch_events_count),
    ("KAeP0+cQPVU", "sceSystemGestureGetPrimitiveTouchEventByIndex", cb_get_primitive_touch_event_by_index),
    ("yBaQ0h9m1NM", "sceSystemGestureGetPrimitiveTouchEventByPrimitiveID", cb_get_primitive_touch_event_by_primitive_id),
    ("FWF8zkhr854", "sceSystemGestureCreateTouchRecognizer", cb_create_touch_recognizer),
    ("1MMK0W-kMgA", "sceSystemGestureAppendTouchRecognizer", cb_recognizer_required),
    ("ELvBVG-LKT0", "sceSystemGestureRemoveTouchRecognizer", cb_recognizer_required),
    ("oBuH3zFWYIg", "sceSystemGestureResetTouchRecognizer", cb_reset_touch_recognizer),
    ("0KrW5eMnrwY", "sceSystemGestureGetTouchRecognizerInformation", cb_get_touch_recognizer_information),
    ("j4h82CQWENo", "sceSystemGestureUpdateTouchRecognizer", cb_recognizer_required),
    ("wPJGwI2RM2I", "sceSystemGestureUpdateAllTouchRecognizer", cb_handle_only),
    ("4WOA1eTx3V8", "sceSystemGestureUpdateTouchRecognizerRectangle", cb_update_touch_recognizer_rectangle),
    ("fLTseA7XiWY", "sceSystemGestureGetTouchEvents", cb_get_touch_events),
    ("h8uongcBNVs", "sceSystemGestureGetTouchEventsCount", cb_get_touch_events_count),
    ("TSKvgSz5ChU", "sceSystemGestureGetTouchEventByIndex", cb_get_touch_event_lookup),
    ("lpsXm7tzeoc", "sceSystemGestureGetTouchEventByEventID", cb_get_touch_event_lookup),
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

    /// The tests share the tracked touches.
    static TOUCH_TESTS: Mutex<()> = Mutex::new(());

    #[test]
    fn handle_and_empty_event_queries() {
        let _serial = TOUCH_TESTS.lock().unwrap_or_else(|p| p.into_inner());
        touches().clear();
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D40_0000_0000;
        vm.map(BASE, 0x4000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);

        ctx.set(Rdi, 1);
        assert_eq!(cb_open(&mut ctx), ERR_INVALID_ARGUMENT, "only the touch pad");
        ctx.set(Rdi, 0);
        assert_eq!(cb_open(&mut ctx), HANDLE as HleResult);

        ctx.set(Rdi, 2);
        assert_eq!(cb_handle_only(&mut ctx), ERR_INVALID_HANDLE);
        assert_eq!(cb_get_primitive_touch_events_count(&mut ctx), ERR_INVALID_HANDLE);

        let events = BASE;
        let number = BASE + 0x1000;
        ctx.write_bytes(events, &[0xAA; PRIMITIVE_TOUCH_EVENT_SIZE]);
        ctx.write_u32(number, 7);
        ctx.set(Rdi, HANDLE as u64);
        ctx.set(Rsi, events);
        ctx.set(Rdx, 4);
        ctx.set(Rcx, 0);
        assert_eq!(cb_get_primitive_touch_events(&mut ctx), ERR_INVALID_ARGUMENT);
        ctx.set(Rcx, number);
        assert_eq!(cb_get_primitive_touch_events(&mut ctx), OK);
        assert_eq!(ctx.read_u32(number), Some(0));
        assert!(ctx.read_bytes(events, PRIMITIVE_TOUCH_EVENT_SIZE).unwrap().iter().all(|&b| b == 0));
        assert_eq!(cb_get_primitive_touch_events_count(&mut ctx), 0);

        ctx.set(Rdx, events);
        assert_eq!(cb_get_primitive_touch_event_by_index(&mut ctx), ERR_EVENT_DATA_NOT_FOUND);
        assert_eq!(cb_get_primitive_touch_event_by_primitive_id(&mut ctx), ERR_EVENT_DATA_NOT_FOUND);

        let recognizer = BASE + 0x2000;
        ctx.write_bytes(recognizer, &[0x55; TOUCH_RECOGNIZER_SIZE]);
        ctx.set(Rsi, 0);
        assert_eq!(cb_create_touch_recognizer(&mut ctx), ERR_INVALID_ARGUMENT);
        assert_eq!(cb_get_touch_events_count(&mut ctx), ERR_INVALID_ARGUMENT);
        ctx.set(Rsi, recognizer);
        assert_eq!(cb_create_touch_recognizer(&mut ctx), OK);
        assert!(ctx.read_bytes(recognizer, TOUCH_RECOGNIZER_SIZE).unwrap().iter().all(|&b| b == 0));
        assert_eq!(cb_recognizer_required(&mut ctx), OK);
        assert_eq!(cb_get_touch_events_count(&mut ctx), 0);

        ctx.set(Rdx, 0);
        assert_eq!(cb_get_touch_recognizer_information(&mut ctx), ERR_INVALID_ARGUMENT);
        assert_eq!(cb_update_touch_recognizer_rectangle(&mut ctx), ERR_INVALID_ARGUMENT);

        ctx.set(Rdx, events);
        ctx.set(Rcx, 1);
        ctx.set(R8, number);
        ctx.write_u32(number, 3);
        assert_eq!(cb_get_touch_events(&mut ctx), OK);
        assert_eq!(ctx.read_u32(number), Some(0));
        ctx.set(Rcx, events);
        assert_eq!(cb_get_touch_event_lookup(&mut ctx), ERR_EVENT_DATA_NOT_FOUND);
    }

    /// Every recognizer call checks the handle first, then its recognizer
    /// (and rectangle) pointer; the primitive-recognizer calls check only
    /// the handle. A bad handle leaves the event count alone.
    #[test]
    fn recognizer_calls_check_handle_and_pointers() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x6A1F_0000_0000;
        vm.map(BASE, 0x4000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (recognizer, rectangle, number) = (BASE, BASE + 0x1000, BASE + 0x2000);
        let call = |ctx: &mut CpuContext, cb: crate::registry::HleCallback, h: i32, a: u64, b: u64| {
            ctx.set(Rdi, h as u64);
            ctx.set(Rsi, a);
            ctx.set(Rdx, b);
            cb(ctx)
        };
        let bad = HANDLE + 1;
        for cb in [cb_recognizer_required, cb_reset_touch_recognizer] {
            assert_eq!(call(&mut ctx, cb, HANDLE, recognizer, 0), OK);
            assert_eq!(call(&mut ctx, cb, HANDLE, 0, 0), ERR_INVALID_ARGUMENT);
            assert_eq!(call(&mut ctx, cb, bad, recognizer, 0), ERR_INVALID_HANDLE);
        }
        assert_eq!(call(&mut ctx, cb_update_touch_recognizer_rectangle, HANDLE, recognizer, rectangle), OK);
        assert_eq!(call(&mut ctx, cb_update_touch_recognizer_rectangle, HANDLE, recognizer, 0), ERR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, cb_update_touch_recognizer_rectangle, HANDLE, 0, rectangle), ERR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, cb_update_touch_recognizer_rectangle, bad, recognizer, rectangle), ERR_INVALID_HANDLE);
        assert_eq!(call(&mut ctx, cb_handle_only, HANDLE, 0, 0), OK);
        assert_eq!(call(&mut ctx, cb_handle_only, bad, 0, 0), ERR_INVALID_HANDLE);

        ctx.write_u32(number, 7);
        ctx.set(Rcx, number);
        assert_eq!(call(&mut ctx, cb_get_primitive_touch_events, bad, BASE + 0x3000, 4), ERR_INVALID_HANDLE);
        assert_eq!(ctx.read_u32(number), Some(7));
        ctx.set(Rcx, 0);
        assert_eq!(call(&mut ctx, cb_get_primitive_touch_events, HANDLE, BASE + 0x3000, 4), ERR_INVALID_ARGUMENT);
        ctx.set(Rcx, 4);
        ctx.set(R8, number);
        assert_eq!(call(&mut ctx, cb_get_touch_events, bad, recognizer, BASE + 0x3000), ERR_INVALID_HANDLE);
        assert_eq!(ctx.read_u32(number), Some(7));
        ctx.set(R8, 0);
        assert_eq!(call(&mut ctx, cb_get_touch_events, HANDLE, recognizer, BASE + 0x3000), ERR_INVALID_ARGUMENT);
        assert_eq!(call(&mut ctx, cb_get_primitive_touch_event_by_index, bad, 0, BASE + 0x3000), ERR_INVALID_HANDLE);
    }

    /// Reference GuestSystemGesture primitive cases: a touch begins, moves
    /// (active, updated), stays (active, not updated) and ends when it is
    /// missing from a report; the ended touch is gone at the next update.
    #[test]
    fn primitive_touches_follow_the_pad_reports() {
        let _serial = TOUCH_TESTS.lock().unwrap_or_else(|p| p.into_inner());
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x6A2F_0000_0000;
        vm.map(BASE, 0x4000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (param, reports, events, number) = (BASE, BASE + 0x100, BASE + 0x1000, BASE + 0x2000);
        let size = crate::pad::PAD_DATA_SIZE as u64;
        // Report i: timestamp, then the touches (id, x, y).
        let report = |ctx: &CpuContext, i: u64, time: u64, points: &[(u8, u16, u16)]| {
            let r = reports + i * size;
            ctx.write_bytes(r, &vec![0u8; size as usize]);
            ctx.write_bytes(r + 52, &[points.len() as u8]);
            for (k, &(id, x, y)) in points.iter().enumerate() {
                let p = r + 60 + 8 * k as u64;
                ctx.write_bytes(p, &[x.to_le_bytes(), y.to_le_bytes()].concat());
                ctx.write_bytes(p + 4, &[id]);
            }
            ctx.write_u64(r + 80, time);
        };
        let update = |ctx: &mut CpuContext, count: u32| {
            ctx.write_u32(param + 4, count);
            ctx.write_u64(param + 8, reports);
            ctx.set(Rdi, HANDLE as u64);
            ctx.set(Rsi, param);
            cb_update_primitive_touch_recognizer(ctx)
        };
        let f32_at = |ctx: &CpuContext, a: u64| f32::from_bits(ctx.read_u32(a).unwrap());
        assert_eq!(cb_forget_touches(&mut ctx), OK);
        report(&ctx, 0, 100, &[(5, 10, 20)]);
        report(&ctx, 1, 150, &[(5, 13, 24), (6, 1, 2)]);
        assert_eq!(update(&mut ctx, 2), OK);
        ctx.set(Rdi, HANDLE as u64);
        assert_eq!(cb_get_primitive_touch_events_count(&mut ctx), 2);
        ctx.set(Rsi, events);
        ctx.set(Rdx, 4);
        ctx.set(Rcx, number);
        assert_eq!(cb_get_primitive_touch_events(&mut ctx), OK);
        assert_eq!(ctx.read_u32(number), Some(2));
        // Touch 5: active and moved by (3, 4) in 50 ticks.
        assert_eq!(ctx.read_u32(events), Some(TOUCH_STATE_ACTIVE as u32));
        assert_eq!(ctx.read_bytes(events + 4, 3).unwrap(), vec![5, 0, 1]);
        assert_eq!((f32_at(&ctx, events + 8), f32_at(&ctx, events + 12)), (10.0, 20.0));
        assert_eq!((f32_at(&ctx, events + 16), f32_at(&ctx, events + 20)), (13.0, 24.0));
        assert_eq!((f32_at(&ctx, events + 24), f32_at(&ctx, events + 28)), (3.0, 4.0));
        assert_eq!(ctx.read_u64(events + 32), Some(50));
        assert_eq!(ctx.read_u64(events + 40), Some(50));
        // Touch 6 just began.
        let second = events + PRIMITIVE_TOUCH_EVENT_SIZE as u64;
        assert_eq!(ctx.read_u32(second), Some(TOUCH_STATE_BEGIN as u32));
        // The same position again: active, not updated; touch 6 ends.
        report(&ctx, 0, 170, &[(5, 13, 24)]);
        assert_eq!(update(&mut ctx, 1), OK);
        ctx.set(Rsi, 5);
        ctx.set(Rdx, events);
        assert_eq!(cb_get_primitive_touch_event_by_primitive_id(&mut ctx), OK);
        assert_eq!(ctx.read_bytes(events + 4, 3).unwrap(), vec![5, 0, 0]);
        assert_eq!(ctx.read_u64(events + 40), Some(70));
        ctx.set(Rsi, 1);
        assert_eq!(cb_get_primitive_touch_event_by_index(&mut ctx), OK);
        assert_eq!(ctx.read_u32(events), Some(TOUCH_STATE_END as u32));
        assert_eq!(ctx.read_u64(events + 32), Some(20));
        // The ended touch is dropped at the next update.
        report(&ctx, 0, 180, &[]);
        assert_eq!(update(&mut ctx, 1), OK);
        ctx.set(Rdi, HANDLE as u64);
        assert_eq!(cb_get_primitive_touch_events_count(&mut ctx), 1);
        ctx.set(Rsi, 6);
        ctx.set(Rdx, events);
        assert_eq!(cb_get_primitive_touch_event_by_primitive_id(&mut ctx), ERR_EVENT_DATA_NOT_FOUND);
        assert_eq!(cb_reset_primitive_touch_recognizer(&mut ctx), OK);
        assert_eq!(cb_get_primitive_touch_events_count(&mut ctx), 0);
    }
}
