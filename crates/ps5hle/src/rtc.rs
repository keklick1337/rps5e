// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceRtc HLE: real-time clock, tick arithmetic and calendar helpers.
//!
//! An RTC "tick" is a count of microseconds since 0001-01-01 00:00:00
//! (proleptic Gregorian). `SceRtcDateTime` is 16 bytes:
//! `u16 year, month, day, hour, minute, second; u32 microsecond`.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};

const OK: HleResult = 0;

/// Returned (positive) by the current-time queries for a NULL output.
const RTC_ERROR_DATETIME_UNINITIALIZED: HleResult = 0x7FFE_F9FE;
const ERR_RTC_INVALID_POINTER: HleResult = 0x80B5_0002u32 as i32 as i64;
const ERR_RTC_INVALID_VALUE: HleResult = 0x80B5_0003u32 as i32 as i64;
const ERR_RTC_BAD_PARSE: HleResult = 0x80B5_0007u32 as i32 as i64;
const ERR_RTC_INVALID_YEAR: HleResult = 0x80B5_0008u32 as i32 as i64;
const ERR_RTC_INVALID_MONTH: HleResult = 0x80B5_0009u32 as i32 as i64;
const ERR_RTC_INVALID_DAY: HleResult = 0x80B5_000Au32 as i32 as i64;
const ERR_RTC_INVALID_HOUR: HleResult = 0x80B5_000Bu32 as i32 as i64;
const ERR_RTC_INVALID_MINUTE: HleResult = 0x80B5_000Cu32 as i32 as i64;
const ERR_RTC_INVALID_SECOND: HleResult = 0x80B5_000Du32 as i32 as i64;
const ERR_RTC_INVALID_MICROSECOND: HleResult = 0x80B5_000Eu32 as i32 as i64;

const US_PER_SECOND: u64 = 1_000_000;
const US_PER_MINUTE: u64 = 60_000_000;
const US_PER_HOUR: u64 = 3_600_000_000;
const US_PER_DAY: u64 = 86_400_000_000;
const US_PER_WEEK: u64 = 604_800_000_000;
/// Tick of 1970-01-01 00:00:00.
const UNIX_EPOCH_TICKS: u64 = 0xDC_BFFE_FF2B_C000;
/// Tick of 1601-01-01 00:00:00 (Win32 FILETIME epoch).
const WIN32_FILETIME_EPOCH_TICKS: u64 = 0xB3_6168_B6A5_8000;
/// Largest representable tick: 9999-12-31 23:59:59.999999.
const MAX_TICK: u64 = 315_537_897_599_999_999;
/// Days from 0001-01-01 to 1970-01-01.
const DAYS_TO_UNIX_EPOCH: i64 = 719_162;
/// Longest RFC 3339 input examined by the parser.
const MAX_RFC3339_INPUT: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct RtcDateTime {
    year: u16,
    month: u16,
    day: u16,
    hour: u16,
    minute: u16,
    second: u16,
    microsecond: u32,
}

impl RtcDateTime {
    fn from_bytes(b: &[u8]) -> Self {
        let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
        Self {
            year: u16_at(0),
            month: u16_at(2),
            day: u16_at(4),
            hour: u16_at(6),
            minute: u16_at(8),
            second: u16_at(10),
            microsecond: u32::from_le_bytes([b[12], b[13], b[14], b[15]]),
        }
    }

    fn to_bytes(self) -> [u8; 16] {
        let mut b = [0u8; 16];
        b[0..2].copy_from_slice(&self.year.to_le_bytes());
        b[2..4].copy_from_slice(&self.month.to_le_bytes());
        b[4..6].copy_from_slice(&self.day.to_le_bytes());
        b[6..8].copy_from_slice(&self.hour.to_le_bytes());
        b[8..10].copy_from_slice(&self.minute.to_le_bytes());
        b[10..12].copy_from_slice(&self.second.to_le_bytes());
        b[12..16].copy_from_slice(&self.microsecond.to_le_bytes());
        b
    }
}

// ---------------------------------------------------------------------------
// Pure calendar math
// ---------------------------------------------------------------------------

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Days in `month` of `year`; caller guarantees 1 <= month <= 12.
fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
    }
}

/// sceRtcGetDaysInMonth semantics: the day count or a year/month error.
fn checked_days_in_month(year: i32, month: i32) -> Result<i64, HleResult> {
    if !(1..=9999).contains(&year) {
        return Err(ERR_RTC_INVALID_YEAR);
    }
    if !(1..=12).contains(&month) {
        return Err(ERR_RTC_INVALID_MONTH);
    }
    Ok(days_in_month(year as i64, month as i64))
}

/// Days since 0001-01-01 for a proleptic Gregorian date.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468 + DAYS_TO_UNIX_EPOCH
}

/// 0 = Sunday ... 6 = Saturday.
fn day_of_week(year: i64, month: i64, day: i64) -> i64 {
    // 0001-01-01 was a Monday.
    (days_from_civil(year, month, day) + 1).rem_euclid(7)
}

/// sceRtcCheckValid order: year, month, day, hour, minute, second, usec.
fn validate(t: &RtcDateTime) -> HleResult {
    if !(1..=9999).contains(&t.year) {
        return ERR_RTC_INVALID_YEAR;
    }
    if !(1..=12).contains(&t.month) {
        return ERR_RTC_INVALID_MONTH;
    }
    if t.day < 1 || t.day as i64 > days_in_month(t.year as i64, t.month as i64) {
        return ERR_RTC_INVALID_DAY;
    }
    if t.hour > 23 {
        return ERR_RTC_INVALID_HOUR;
    }
    if t.minute > 59 {
        return ERR_RTC_INVALID_MINUTE;
    }
    if t.second > 59 {
        return ERR_RTC_INVALID_SECOND;
    }
    if t.microsecond > 999_999 {
        return ERR_RTC_INVALID_MICROSECOND;
    }
    OK
}

/// Tick of a validated date-time (March-based day count).
fn rtc_to_tick(t: &RtcDateTime) -> u64 {
    let (mut year, mut month) = (t.year as u64, t.month as u64);
    if month > 2 {
        month -= 3;
    } else {
        month += 9;
        year -= 1;
    }
    let century = year / 100;
    let year_of_century = year - 100 * century;
    let days = ((146_097 * century) >> 2) + ((1461 * year_of_century) >> 2) + (153 * month + 2) / 5
        + t.day as u64
        - 307;
    let tod = t.hour as u64 * US_PER_HOUR
        + t.minute as u64 * US_PER_MINUTE
        + t.second as u64 * US_PER_SECOND
        + t.microsecond as u64;
    days * US_PER_DAY + tod
}

/// sceRtcSetTick conversion; `None` when the year leaves 1..=9999.
fn tick_to_rtc(tick: u64) -> Option<RtcDateTime> {
    let mut days = tick / US_PER_DAY;
    let mut usec = tick % US_PER_DAY;
    days += 307;
    let mut j = (days << 2) - 1;
    let mut ly = j / 146_097;
    j -= 146_097 * ly;
    let mut ld = j >> 2;
    j = ((ld << 2) + 3) / 1461;
    ld = (((ld << 2) + 7) - 1461 * j) >> 2;
    let mut lm = (5 * ld - 3) / 153;
    ld = (5 * ld + 2 - 153 * lm) / 5;
    ly = 100 * ly + j;
    if lm < 10 {
        lm += 3;
    } else {
        lm -= 9;
        ly += 1;
    }
    if ly == 0 || ly > 9999 {
        return None;
    }
    let hour = usec / US_PER_HOUR;
    usec %= US_PER_HOUR;
    let minute = usec / US_PER_MINUTE;
    usec %= US_PER_MINUTE;
    let second = usec / US_PER_SECOND;
    usec %= US_PER_SECOND;
    Some(RtcDateTime {
        year: ly as u16,
        month: lm as u16,
        day: ld as u16,
        hour: hour as u16,
        minute: minute as u16,
        second: second as u16,
        microsecond: usec as u32,
    })
}

fn dos_time_from_rtc(t: &RtcDateTime) -> u32 {
    let mut v = (t.second as u32 / 2) & 0x1F;
    v |= (t.minute as u32 & 0x3F) << 5;
    v |= (t.hour as u32 & 0x1F) << 11;
    v |= (t.day as u32 & 0x1F) << 16;
    v |= (t.month as u32 & 0x0F) << 21;
    v |= ((t.year as u32).wrapping_sub(1980) & 0x7F) << 25;
    v
}

fn rtc_from_dos_time(dos: u32) -> RtcDateTime {
    RtcDateTime {
        year: (1980 + ((dos >> 25) & 0x7F)) as u16,
        month: ((dos >> 21) & 0x0F) as u16,
        day: ((dos >> 16) & 0x1F) as u16,
        hour: ((dos >> 11) & 0x1F) as u16,
        minute: ((dos >> 5) & 0x3F) as u16,
        second: ((dos << 1) & 0x3E) as u16,
        microsecond: 0,
    }
}

fn clamp_tick(t: i128) -> u64 {
    t.clamp(0, MAX_TICK as i128) as u64
}

// ---------------------------------------------------------------------------
// Host clock / timezone
// ---------------------------------------------------------------------------

fn host_utc_tick() -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    UNIX_EPOCH_TICKS + now.as_secs() * US_PER_SECOND + now.subsec_micros() as u64
}

/// A clock sceRtcSetCurrent*Tick set for this process: the tick and when.
#[derive(Clone, Copy)]
struct SetClock {
    tick: u64,
    at: std::time::Instant,
}

/// The process-local clocks: 0 the current tick, 1 network, 2 ad-network,
/// 3 debug-network, 4 raw network (never set).
static CLOCKS: std::sync::Mutex<[Option<SetClock>; 5]> = std::sync::Mutex::new([None; 5]);

/// The tick of clock `clock`: the set tick plus the time since, or (the
/// ad and debug network clocks falling back to the network clock) the host
/// clock.
fn clock_tick(clock: usize) -> u64 {
    let set = CLOCKS.lock().unwrap_or_else(|p| p.into_inner())[clock];
    if let Some(c) = set {
        return c.tick.saturating_add(c.at.elapsed().as_micros().min(u64::MAX as u128) as u64);
    }
    if clock == 2 || clock == 3 {
        return clock_tick(1);
    }
    host_utc_tick()
}

/// The current UTC tick (sceRtcSetCurrentTick moves it).
fn utc_now_tick() -> u64 {
    clock_tick(0)
}

/// Host local-timezone UTC offset (seconds) and DST flag at a Unix time.
pub(crate) fn host_local_offset(unix_secs: i64) -> Option<(i64, bool)> {
    let t = unix_secs as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let r = unsafe { libc::localtime_r(&t, &mut tm) };
    if r.is_null() {
        return None;
    }
    Some((tm.tm_gmtoff as i64, tm.tm_isdst > 0))
}

fn tick_unix_secs(tick: u64) -> i64 {
    (tick as i128 - UNIX_EPOCH_TICKS as i128).div_euclid(US_PER_SECOND as i128) as i64
}

/// UTC tick to the local wall clock (unchanged when the zone is unknown).
fn utc_to_local_with(tick: u64, offset_at: impl Fn(i64) -> Option<(i64, bool)>) -> u64 {
    match offset_at(tick_unix_secs(tick)) {
        Some((off, _)) => clamp_tick(tick as i128 + off as i128 * US_PER_SECOND as i128),
        None => tick,
    }
}

/// Local wall-clock tick to UTC. Ambiguous times resolve to standard time;
/// times skipped by a DST transition use the offset in force before it.
fn local_to_utc_with(tick: u64, offset_at: impl Fn(i64) -> Option<(i64, bool)>) -> u64 {
    let local = tick_unix_secs(tick);
    let mut chosen: Option<(i64, bool)> = None;
    for probe in [local - 50_400, local, local + 50_400] {
        let Some((off, _)) = offset_at(probe) else { continue };
        let Some((actual, dst)) = offset_at(local - off) else { continue };
        if actual != off {
            continue;
        }
        match chosen {
            None => chosen = Some((off, dst)),
            Some((_, true)) if !dst => chosen = Some((off, dst)),
            _ => {}
        }
    }
    let off = match chosen {
        Some((off, _)) => off,
        None => match offset_at(local - 50_400) {
            Some((off, _)) => off,
            None => return tick,
        },
    };
    clamp_tick(tick as i128 - off as i128 * US_PER_SECOND as i128)
}

/// The host's local date and time, to the millisecond.
fn host_local_now() -> RtcDateTime {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let t = now.as_secs() as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    if unsafe { libc::localtime_r(&t, &mut tm) }.is_null() {
        return RtcDateTime::default();
    }
    RtcDateTime {
        year: (tm.tm_year + 1900) as u16,
        month: (tm.tm_mon + 1) as u16,
        day: tm.tm_mday as u16,
        hour: tm.tm_hour as u16,
        minute: tm.tm_min as u16,
        second: tm.tm_sec.min(59) as u16,
        microsecond: now.subsec_millis() * 1000,
    }
}

// ---------------------------------------------------------------------------
// Guest memory helpers
// ---------------------------------------------------------------------------

fn read_rtc(ctx: &CpuContext, addr: u64) -> Option<RtcDateTime> {
    ctx.read_bytes(addr, 16).map(|b| RtcDateTime::from_bytes(&b))
}

fn write_rtc(ctx: &CpuContext, addr: u64, t: RtcDateTime) -> HleResult {
    if ctx.write_bytes(addr, &t.to_bytes()) {
        OK
    } else {
        ERR_RTC_INVALID_POINTER
    }
}

/// sceRtcGetTick on guest memory: validate, then convert.
fn read_valid_tick(ctx: &CpuContext, addr: u64) -> Result<u64, HleResult> {
    let t = read_rtc(ctx, addr).ok_or(ERR_RTC_INVALID_POINTER)?;
    match validate(&t) {
        OK => Ok(rtc_to_tick(&t)),
        e => Err(e),
    }
}

fn write_u64_result(ctx: &CpuContext, addr: u64, v: u64) -> HleResult {
    if ctx.write_u64(addr, v) {
        OK
    } else {
        ERR_RTC_INVALID_POINTER
    }
}

/// sceRtcSetTick body: store the date-time of `tick` at `dst`.
fn set_tick(ctx: &CpuContext, dst: u64, tick: u64) -> HleResult {
    match tick_to_rtc(tick) {
        Some(t) => write_rtc(ctx, dst, t),
        None => ERR_RTC_INVALID_VALUE,
    }
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

fn cb_check_valid(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 {
        return ERR_RTC_INVALID_POINTER;
    }
    match read_rtc(ctx, addr) {
        Some(t) => validate(&t),
        None => ERR_RTC_INVALID_POINTER,
    }
}

fn cb_compare_tick(ctx: &mut CpuContext) -> HleResult {
    let (a, b) = (ctx.rdi(), ctx.rsi());
    if a == 0 || b == 0 {
        return ERR_RTC_INVALID_POINTER;
    }
    match (ctx.read_u64(a), ctx.read_u64(b)) {
        (Some(t1), Some(t2)) => match t1.cmp(&t2) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Greater => 1,
            std::cmp::Ordering::Equal => 0,
        },
        _ => ERR_RTC_INVALID_POINTER,
    }
}

fn convert_tick(ctx: &mut CpuContext, f: fn(u64) -> u64) -> HleResult {
    let (src, dst) = (ctx.rdi(), ctx.rsi());
    if src == 0 || dst == 0 {
        return ERR_RTC_INVALID_POINTER;
    }
    let Some(tick) = ctx.read_u64(src) else { return ERR_RTC_INVALID_POINTER };
    // Ticks past the last representable date are rejected before the
    // timezone offset is applied; the output is left untouched.
    if tick > MAX_TICK {
        return ERR_RTC_INVALID_VALUE;
    }
    write_u64_result(ctx, dst, f(tick))
}

fn cb_convert_local_time_to_utc(ctx: &mut CpuContext) -> HleResult {
    convert_tick(ctx, |t| local_to_utc_with(t, host_local_offset))
}

fn cb_convert_utc_to_local_time(ctx: &mut CpuContext) -> HleResult {
    convert_tick(ctx, |t| utc_to_local_with(t, host_local_offset))
}

fn cb_ok(_ctx: &mut CpuContext) -> HleResult {
    OK
}

/// sceRtcGetCurrentClock(SceRtcDateTime*, int timeZoneMinutes)
fn cb_get_current_clock(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 {
        return RTC_ERROR_DATETIME_UNINITIALIZED;
    }
    let tz_minutes = ctx.rsi() as i32;
    let tick = utc_now_tick().wrapping_add((tz_minutes as i64 * US_PER_MINUTE as i64) as u64);
    set_tick(ctx, addr, tick)
}

fn cb_get_current_clock_local_time(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 {
        return RTC_ERROR_DATETIME_UNINITIALIZED;
    }
    write_rtc(ctx, addr, host_local_now())
}

/// sceRtcGetCurrentTick and the network variants: their clock's tick.
fn current_tick_of(ctx: &CpuContext, clock: usize) -> HleResult {
    let addr = ctx.rdi();
    if addr == 0 {
        return RTC_ERROR_DATETIME_UNINITIALIZED;
    }
    write_u64_result(ctx, addr, clock_tick(clock))
}

fn cb_get_current_tick(ctx: &mut CpuContext) -> HleResult {
    current_tick_of(ctx, 0)
}
fn cb_get_current_network_tick(ctx: &mut CpuContext) -> HleResult {
    current_tick_of(ctx, 1)
}
fn cb_get_current_ad_network_tick(ctx: &mut CpuContext) -> HleResult {
    current_tick_of(ctx, 2)
}
fn cb_get_current_debug_network_tick(ctx: &mut CpuContext) -> HleResult {
    current_tick_of(ctx, 3)
}
fn cb_get_current_raw_network_tick(ctx: &mut CpuContext) -> HleResult {
    current_tick_of(ctx, 4)
}

/// sceRtcSetCurrent*Tick(const SceRtcTick*): set a process-local clock; a
/// tick before 1970 is INVALID_VALUE.
fn set_current_tick(ctx: &CpuContext, clock: usize) -> HleResult {
    let src = ctx.rdi();
    if src == 0 {
        return ERR_RTC_INVALID_POINTER;
    }
    let Some(tick) = ctx.read_u64(src) else { return ERR_RTC_INVALID_POINTER };
    if tick < UNIX_EPOCH_TICKS {
        return ERR_RTC_INVALID_VALUE;
    }
    CLOCKS.lock().unwrap_or_else(|p| p.into_inner())[clock] = Some(SetClock { tick, at: std::time::Instant::now() });
    OK
}

fn cb_set_current_tick(ctx: &mut CpuContext) -> HleResult {
    set_current_tick(ctx, 0)
}
fn cb_set_current_network_tick(ctx: &mut CpuContext) -> HleResult {
    set_current_tick(ctx, 1)
}
fn cb_set_current_ad_network_tick(ctx: &mut CpuContext) -> HleResult {
    set_current_tick(ctx, 2)
}
fn cb_set_current_debug_network_tick(ctx: &mut CpuContext) -> HleResult {
    set_current_tick(ctx, 3)
}

fn cb_get_day_of_week(ctx: &mut CpuContext) -> HleResult {
    let (y, m, d) = (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx() as i32);
    let days = match checked_days_in_month(y, m) {
        Ok(days) => days,
        Err(e) => return e,
    };
    if d <= 0 || d as i64 > days {
        return ERR_RTC_INVALID_DAY;
    }
    day_of_week(y as i64, m as i64, d as i64)
}

fn cb_get_days_in_month(ctx: &mut CpuContext) -> HleResult {
    match checked_days_in_month(ctx.rdi() as i32, ctx.rsi() as i32) {
        Ok(days) => days,
        Err(e) => e,
    }
}

fn cb_get_dos_time(ctx: &mut CpuContext) -> HleResult {
    let (src, dst) = (ctx.rdi(), ctx.rsi());
    if src == 0 || dst == 0 {
        return ERR_RTC_INVALID_POINTER;
    }
    let Some(t) = read_rtc(ctx, src) else { return ERR_RTC_INVALID_POINTER };
    let v = validate(&t);
    if v != OK {
        return v;
    }
    if ctx.write_u32(dst, dos_time_from_rtc(&t)) {
        OK
    } else {
        ERR_RTC_INVALID_POINTER
    }
}

fn cb_get_tick(ctx: &mut CpuContext) -> HleResult {
    let (src, dst) = (ctx.rdi(), ctx.rsi());
    if src == 0 || dst == 0 {
        return ERR_RTC_INVALID_POINTER;
    }
    match read_valid_tick(ctx, src) {
        Ok(tick) => write_u64_result(ctx, dst, tick),
        Err(e) => e,
    }
}

fn cb_get_tick_resolution(_ctx: &mut CpuContext) -> HleResult {
    US_PER_SECOND as i64
}

fn cb_get_time_t(ctx: &mut CpuContext) -> HleResult {
    let (src, dst) = (ctx.rdi(), ctx.rsi());
    if src == 0 || dst == 0 {
        return ERR_RTC_INVALID_POINTER;
    }
    match read_valid_tick(ctx, src) {
        Ok(tick) => write_u64_result(ctx, dst, tick.saturating_sub(UNIX_EPOCH_TICKS) / US_PER_SECOND),
        Err(e) => e,
    }
}

fn cb_get_win32_file_time(ctx: &mut CpuContext) -> HleResult {
    let (src, dst) = (ctx.rdi(), ctx.rsi());
    if src == 0 || dst == 0 {
        return ERR_RTC_INVALID_POINTER;
    }
    match read_valid_tick(ctx, src) {
        Ok(tick) => write_u64_result(ctx, dst, tick.saturating_sub(WIN32_FILETIME_EPOCH_TICKS) * 10),
        Err(e) => e,
    }
}

fn cb_is_leap_year(ctx: &mut CpuContext) -> HleResult {
    let y = ctx.rdi() as i32;
    if !(1..=9999).contains(&y) {
        return ERR_RTC_INVALID_YEAR;
    }
    is_leap_year(y as i64) as i64
}

/// `dst = src + delta` with two's-complement wrap-around (no range check).
fn tick_add(ctx: &mut CpuContext, delta: i64) -> HleResult {
    let (dst, src) = (ctx.rdi(), ctx.rsi());
    if dst == 0 || src == 0 {
        return ERR_RTC_INVALID_POINTER;
    }
    let Some(tick) = ctx.read_u64(src) else { return ERR_RTC_INVALID_POINTER };
    write_u64_result(ctx, dst, tick.wrapping_add(delta as u64))
}

/// sceRtcTickAddMonths / Years as the console computes them: the date
/// moves by whole months with 32-bit arithmetic on the year (taken as 16
/// bits) and month; a result outside years 1..=9999 leaves `dst` alone and
/// still succeeds; the day is clamped to the month's length.
fn tick_add_calendar(ctx: &mut CpuContext, months: i32) -> HleResult {
    let (dst, src) = (ctx.rdi(), ctx.rsi());
    if dst == 0 || src == 0 {
        return ERR_RTC_INVALID_POINTER;
    }
    let Some(tick) = ctx.read_u64(src) else { return ERR_RTC_INVALID_POINTER };
    match add_months_wrapping(tick, months) {
        Some(r) => write_u64_result(ctx, dst, r),
        None => OK,
    }
}

/// The date of a tick with no upper limit on the year: (year, month, day).
fn tick_civil(tick: u64) -> (i64, i64, i64) {
    let mut days = (tick / US_PER_DAY) as i64 + 306;
    let era = days.div_euclid(146_097);
    days -= era * 146_097;
    let yoe = (days - days / 1460 + days / 36_524 - days / 146_096) / 365;
    let doy = days - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + (month <= 2) as i64, month, day)
}

fn add_months_wrapping(tick: u64, months: i32) -> Option<u64> {
    let (year, month, day) = tick_civil(tick);
    let index = (year as u16 as u32).wrapping_mul(12).wrapping_add(month as u32).wrapping_add(months as u32).wrapping_sub(1) as i32;
    let new_year = (index / 12) as u16;
    let new_month = index % 12;
    if !(1..=9999).contains(&new_year) || new_month < 0 {
        return None;
    }
    let (y, m) = (new_year as i64, new_month as i64 + 1);
    let d = day.min(days_in_month(y, m));
    Some(days_from_civil(y, m, d) as u64 * US_PER_DAY + tick % US_PER_DAY)
}

/// 64-bit count argument.
fn arg_i64(ctx: &CpuContext) -> i64 {
    ctx.rdx() as i64
}

/// 32-bit count argument (hours, days, weeks).
fn arg_i32(ctx: &CpuContext) -> i64 {
    ctx.rdx() as i32 as i64
}

fn cb_tick_add_ticks(ctx: &mut CpuContext) -> HleResult {
    let d = arg_i64(ctx);
    tick_add(ctx, d)
}
fn cb_tick_add_microseconds(ctx: &mut CpuContext) -> HleResult {
    let d = arg_i64(ctx);
    tick_add(ctx, d)
}
fn cb_tick_add_seconds(ctx: &mut CpuContext) -> HleResult {
    let d = arg_i64(ctx).wrapping_mul(US_PER_SECOND as i64);
    tick_add(ctx, d)
}
fn cb_tick_add_minutes(ctx: &mut CpuContext) -> HleResult {
    let d = arg_i64(ctx).wrapping_mul(US_PER_MINUTE as i64);
    tick_add(ctx, d)
}
fn cb_tick_add_hours(ctx: &mut CpuContext) -> HleResult {
    let d = arg_i32(ctx).wrapping_mul(US_PER_HOUR as i64);
    tick_add(ctx, d)
}
fn cb_tick_add_days(ctx: &mut CpuContext) -> HleResult {
    let d = arg_i32(ctx).wrapping_mul(US_PER_DAY as i64);
    tick_add(ctx, d)
}
fn cb_tick_add_weeks(ctx: &mut CpuContext) -> HleResult {
    let d = arg_i32(ctx).wrapping_mul(US_PER_WEEK as i64);
    tick_add(ctx, d)
}
/// sceRtcTickAddMonths(dst, src, int months)
fn cb_tick_add_months(ctx: &mut CpuContext) -> HleResult {
    let months = ctx.rdx() as i32;
    tick_add_calendar(ctx, months)
}
/// sceRtcTickAddYears(dst, src, short years)
fn cb_tick_add_years(ctx: &mut CpuContext) -> HleResult {
    let months = (ctx.rdx() as i16 as i32).wrapping_mul(12);
    tick_add_calendar(ctx, months)
}

fn cb_set_tick(ctx: &mut CpuContext) -> HleResult {
    let (dst, src) = (ctx.rdi(), ctx.rsi());
    if dst == 0 || src == 0 {
        return ERR_RTC_INVALID_POINTER;
    }
    let Some(tick) = ctx.read_u64(src) else { return ERR_RTC_INVALID_POINTER };
    set_tick(ctx, dst, tick)
}

fn cb_set_dos_time(ctx: &mut CpuContext) -> HleResult {
    let dst = ctx.rdi();
    if dst == 0 {
        return ERR_RTC_INVALID_POINTER;
    }
    write_rtc(ctx, dst, rtc_from_dos_time(ctx.rsi() as u32))
}

/// Tick of a time_t, `None` for negative or unrepresentable values.
fn time_t_tick(secs: i64) -> Option<u64> {
    if secs < 0 {
        return None;
    }
    let secs = secs as u64;
    if secs > (u64::MAX - UNIX_EPOCH_TICKS) / US_PER_SECOND {
        return None;
    }
    Some(secs * US_PER_SECOND + UNIX_EPOCH_TICKS)
}

fn cb_set_time_t(ctx: &mut CpuContext) -> HleResult {
    let dst = ctx.rdi();
    if dst == 0 {
        return ERR_RTC_INVALID_POINTER;
    }
    match time_t_tick(ctx.rsi() as i64) {
        Some(tick) => set_tick(ctx, dst, tick),
        None => ERR_RTC_INVALID_VALUE,
    }
}

fn cb_set_win32_file_time(ctx: &mut CpuContext) -> HleResult {
    let dst = ctx.rdi();
    if dst == 0 {
        return ERR_RTC_INVALID_POINTER;
    }
    let ft = ctx.rsi();
    if ft / 10 > u64::MAX - WIN32_FILETIME_EPOCH_TICKS {
        return ERR_RTC_INVALID_VALUE;
    }
    set_tick(ctx, dst, ft / 10 + WIN32_FILETIME_EPOCH_TICKS)
}

// ---------------------------------------------------------------------------
// RFC 3339
// ---------------------------------------------------------------------------

/// `YYYY-MM-DDThh:mm:ss.cc` plus `Z` or `+hh:mm`, for a UTC tick shown in
/// the given zone.
fn format_rfc3339(utc: u64, zone_minutes: i32) -> Result<String, HleResult> {
    let offset = zone_minutes as i64 * US_PER_MINUTE as i64;
    if offset < 0 && utc < offset.unsigned_abs() {
        return Err(ERR_RTC_INVALID_VALUE);
    }
    let t = tick_to_rtc(utc.wrapping_add(offset as u64)).ok_or(ERR_RTC_INVALID_VALUE)?;
    let zone = if zone_minutes == 0 {
        "Z".to_string()
    } else {
        let sign = if zone_minutes < 0 { '-' } else { '+' };
        let m = zone_minutes.unsigned_abs();
        let mut z = format!("{sign}{:02}:{:02}", m / 60, m % 60);
        // The zone text is built in an 8-byte buffer.
        z.truncate(7);
        z
    };
    Ok(format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:02}{zone}",
        t.year,
        t.month,
        t.day,
        t.hour,
        t.minute,
        t.second,
        t.microsecond / 10_000
    ))
}

/// `YYYY-MM-DDThh:mm:ss`, then `.` and `precision` (1..6) fraction digits
/// when asked for, then `Z` or `+hh:mm` (sceRtcFormatRFC3339Precise); the
/// zone is at most a day minus a minute either way.
fn format_rfc3339_precise(utc: u64, zone_minutes: i32, precision: i32) -> Result<String, HleResult> {
    if !(-1439..=1439).contains(&zone_minutes) || !(0..=6).contains(&precision) {
        return Err(ERR_RTC_INVALID_VALUE);
    }
    let offset = zone_minutes as i64 * US_PER_MINUTE as i64;
    let local = utc.checked_add_signed(offset).filter(|&t| t <= MAX_TICK).ok_or(ERR_RTC_INVALID_VALUE)?;
    let t = tick_to_rtc(local).ok_or(ERR_RTC_INVALID_VALUE)?;
    let mut text = format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}", t.year, t.month, t.day, t.hour, t.minute, t.second);
    if precision > 0 {
        let divisor = 10u32.pow(6 - precision as u32);
        text.push_str(&format!(".{:0width$}", t.microsecond / divisor, width = precision as usize));
    }
    if zone_minutes == 0 {
        text.push('Z');
    } else {
        let m = zone_minutes.unsigned_abs();
        text.push_str(&format!("{}{:02}:{:02}", if zone_minutes < 0 { '-' } else { '+' }, m / 60, m % 60));
    }
    Ok(text)
}

fn is_c_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\x0B' | b'\x0C' | b'\r')
}

/// `%hu` conversion: optional white space and sign, then decimal digits
/// (out-of-range magnitudes saturate before narrowing).
fn scan_hu(s: &[u8], pos: &mut usize) -> Option<u16> {
    let mut p = *pos;
    while s.get(p).copied().is_some_and(is_c_space) {
        p += 1;
    }
    let negative = match s.get(p) {
        Some(b'-') => {
            p += 1;
            true
        }
        Some(b'+') => {
            p += 1;
            false
        }
        _ => false,
    };
    let start = p;
    let mut v: Option<u64> = Some(0);
    while let Some(c @ b'0'..=b'9') = s.get(p).copied() {
        v = v.and_then(|v| v.checked_mul(10)).and_then(|v| v.checked_add(u64::from(c - b'0')));
        p += 1;
    }
    if p == start {
        return None;
    }
    *pos = p;
    let v = match v {
        None => u64::MAX,
        Some(v) if negative => v.wrapping_neg(),
        Some(v) => v,
    };
    Some(v as u16)
}

/// `%02d` conversion: optional white space, then at most two characters of
/// sign and digits.
fn scan_d2(s: &[u8], pos: &mut usize) -> Option<i32> {
    let mut p = *pos;
    while s.get(p).copied().is_some_and(is_c_space) {
        p += 1;
    }
    let end = p + 2;
    let negative = match s.get(p) {
        Some(b'-') => {
            p += 1;
            true
        }
        Some(b'+') => {
            p += 1;
            false
        }
        _ => false,
    };
    let start = p;
    let mut v = 0i32;
    while p < end {
        match s.get(p) {
            Some(c @ b'0'..=b'9') => v = v * 10 + i32::from(c - b'0'),
            _ => break,
        }
        p += 1;
    }
    if p == start {
        return None;
    }
    *pos = p;
    Some(if negative { -v } else { v })
}

/// Outcome of parsing: the tick to store (if any) and the return code.
#[derive(Debug, PartialEq)]
struct ParseOutcome {
    tick: Option<u64>,
    result: HleResult,
}

/// sceRtcParseRFC3339: `Y-M-DTh:m:s` read as with `%hu` fields, optional
/// `.fraction` (first six digits), then `Z`, end of text, or `±hh:mm`.
/// The local tick is stored before the zone is examined.
fn parse_rfc3339(text: &[u8]) -> ParseOutcome {
    let fail = |result| ParseOutcome { tick: None, result };
    let mut pos = 0usize;
    let mut fields = [0u16; 6];
    for (i, f) in fields.iter_mut().enumerate() {
        match scan_hu(text, &mut pos) {
            Some(v) => *f = v,
            None => return fail(ERR_RTC_INVALID_VALUE),
        }
        let sep = [b'-', b'-', b'T', b':', b':'];
        if i < 5 {
            if text.get(pos) != Some(&sep[i]) {
                return fail(ERR_RTC_INVALID_VALUE);
            }
            pos += 1;
        }
    }
    let mut t = RtcDateTime {
        year: fields[0],
        month: fields[1],
        day: fields[2],
        hour: fields[3],
        minute: fields[4],
        second: fields[5],
        microsecond: 0,
    };
    if text.get(pos) == Some(&b'.') {
        pos += 1;
        let (mut us, mut digits) = (0u32, 0);
        while let Some(c @ b'0'..=b'9') = text.get(pos).copied() {
            if digits < 6 {
                us = us * 10 + u32::from(c - b'0');
            }
            digits += 1;
            pos += 1;
        }
        while digits > 0 && digits < 6 {
            us *= 10;
            digits += 1;
        }
        t.microsecond = us;
    }
    let v = validate(&t);
    if v != OK {
        return fail(v);
    }
    let local = rtc_to_tick(&t);
    let stored = |result| ParseOutcome { tick: Some(local), result };
    let sign = match text.get(pos) {
        None | Some(b'Z') => return stored(OK),
        Some(&s @ (b'+' | b'-')) => s,
        Some(_) => return stored(ERR_RTC_INVALID_VALUE),
    };
    pos += 1;
    let Some(h) = scan_d2(text, &mut pos) else { return stored(ERR_RTC_INVALID_VALUE) };
    if text.get(pos) != Some(&b':') {
        return stored(ERR_RTC_INVALID_VALUE);
    }
    pos += 1;
    let Some(m) = scan_d2(text, &mut pos) else { return stored(ERR_RTC_INVALID_VALUE) };
    let mut zone = h * 60 + m;
    if sign == b'-' {
        zone = -zone;
    }
    let offset = zone as i64 * US_PER_MINUTE as i64;
    if offset > 0 && local < offset as u64 {
        return stored(ERR_RTC_INVALID_VALUE);
    }
    ParseOutcome { tick: Some(local.wrapping_sub(offset as u64)), result: OK }
}

/// sceRtcFormatRFC3339(char* dateTime, const SceRtcTick* utc, int timeZoneMinutes);
/// a NULL tick formats the current time.
fn cb_format_rfc3339(ctx: &mut CpuContext) -> HleResult {
    let (out, utc_addr, zone) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as i32);
    if out == 0 {
        return ERR_RTC_INVALID_POINTER;
    }
    let utc = if utc_addr == 0 {
        utc_now_tick()
    } else {
        match ctx.read_u64(utc_addr) {
            Some(t) => t,
            None => return ERR_RTC_INVALID_POINTER,
        }
    };
    match format_rfc3339(utc, zone) {
        Ok(text) => {
            let mut bytes = text.into_bytes();
            bytes.truncate(31);
            bytes.push(0);
            if ctx.write_bytes(out, &bytes) { OK } else { ERR_RTC_INVALID_POINTER }
        }
        Err(e) => e,
    }
}

/// sceRtcParseRFC3339(SceRtcTick* utc, const char* dateTime)
fn cb_parse_rfc3339(ctx: &mut CpuContext) -> HleResult {
    let (utc_addr, text_addr) = (ctx.rdi(), ctx.rsi());
    if utc_addr == 0 || text_addr == 0 {
        return ERR_RTC_INVALID_POINTER;
    }
    let Some(text) = ctx.read_cstr_bytes(text_addr, MAX_RFC3339_INPUT) else {
        return ERR_RTC_INVALID_POINTER;
    };
    let outcome = parse_rfc3339(&text);
    if let Some(tick) = outcome.tick {
        if !ctx.write_u64(utc_addr, tick) {
            return ERR_RTC_INVALID_POINTER;
        }
    }
    outcome.result
}

// ---------------------------------------------------------------------------
// RFC 2822, local-time formatting and free-form date parsing
// ---------------------------------------------------------------------------

const WEEKDAY_NAMES: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTH_NAMES: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/// Largest zone offset the RFC 2822 formatter accepts (23:59).
const ZONE_MINUTES_MAX: i32 = 1439;
/// Formatted date buffers hold 32 bytes with the terminator.
const FORMAT_BUFFER_SIZE: usize = 32;

/// `utc` moved by `minutes`, or None outside 0001-01-01..9999-12-31.
fn add_zone(utc: u64, minutes: i32) -> Option<u64> {
    if utc > MAX_TICK {
        return None;
    }
    let local = utc as i128 + minutes as i128 * US_PER_MINUTE as i128;
    (0..=MAX_TICK as i128).contains(&local).then_some(local as u64)
}

/// `Thu, 29 Feb 2024 12:34:56 +0130` for a UTC tick shown in the zone.
fn format_rfc2822(utc: u64, zone_minutes: i32) -> Result<String, HleResult> {
    if !(-ZONE_MINUTES_MAX..=ZONE_MINUTES_MAX).contains(&zone_minutes) {
        return Err(ERR_RTC_INVALID_VALUE);
    }
    let local = add_zone(utc, zone_minutes).ok_or(ERR_RTC_INVALID_VALUE)?;
    let t = tick_to_rtc(local).ok_or(ERR_RTC_INVALID_VALUE)?;
    let weekday = WEEKDAY_NAMES[((local / US_PER_DAY + 1) % 7) as usize];
    let sign = if zone_minutes < 0 { '-' } else { '+' };
    let offset = zone_minutes.unsigned_abs();
    Ok(format!(
        "{weekday}, {:02} {} {:04} {:02}:{:02}:{:02} {sign}{:02}{:02}",
        t.day,
        MONTH_NAMES[t.month as usize - 1],
        t.year,
        t.hour,
        t.minute,
        t.second,
        offset / 60,
        offset % 60
    ))
}

/// The local zone offset at a UTC tick, in whole minutes (0 when unknown).
fn local_zone_minutes_with(utc: u64, offset_at: impl Fn(i64) -> Option<(i64, bool)>) -> i32 {
    offset_at(tick_unix_secs(utc)).map_or(0, |(off, _)| (off / 60) as i32)
}

fn write_formatted(ctx: &CpuContext, out: u64, text: Result<String, HleResult>) -> HleResult {
    match text {
        Ok(text) => {
            let mut bytes = text.into_bytes();
            bytes.truncate(FORMAT_BUFFER_SIZE - 1);
            bytes.push(0);
            if ctx.write_bytes(out, &bytes) { OK } else { ERR_RTC_INVALID_POINTER }
        }
        Err(e) => e,
    }
}

/// Reads the UTC tick argument of a formatter: a null tick formats the
/// current time (measured on a PS5).
fn read_utc_arg(ctx: &CpuContext, out: u64, utc_addr: u64) -> Result<u64, HleResult> {
    if out == 0 {
        return Err(ERR_RTC_INVALID_POINTER);
    }
    read_utc_or_now(ctx, utc_addr)
}

/// sceRtcFormatRFC2822(char* dateTime, const SceRtcTick* utc, int timeZoneMinutes)
fn cb_format_rfc2822(ctx: &mut CpuContext) -> HleResult {
    let (out, zone) = (ctx.rdi(), ctx.rdx() as i32);
    match read_utc_arg(ctx, out, ctx.rsi()) {
        Ok(utc) => write_formatted(ctx, out, format_rfc2822(utc, zone)),
        Err(e) => e,
    }
}

fn format_rfc2822_local_with(utc: u64, offset_at: impl Fn(i64) -> Option<(i64, bool)>) -> Result<String, HleResult> {
    if utc > MAX_TICK {
        return Err(ERR_RTC_INVALID_VALUE);
    }
    format_rfc2822(utc, local_zone_minutes_with(utc, offset_at))
}

fn format_rfc3339_local_with(utc: u64, offset_at: impl Fn(i64) -> Option<(i64, bool)>) -> Result<String, HleResult> {
    if utc > MAX_TICK {
        return Err(ERR_RTC_INVALID_VALUE);
    }
    format_rfc3339(utc, local_zone_minutes_with(utc, offset_at))
}

/// sceRtcFormatRFC2822LocalTime(char* dateTime, const SceRtcTick* utc)
fn cb_format_rfc2822_local_time(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    match read_utc_arg(ctx, out, ctx.rsi()) {
        Ok(utc) => write_formatted(ctx, out, format_rfc2822_local_with(utc, host_local_offset)),
        Err(e) => e,
    }
}

/// sceRtcFormatRFC3339LocalTime(char* dateTime, const SceRtcTick* utc)
fn cb_format_rfc3339_local_time(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    match read_utc_arg(ctx, out, ctx.rsi()) {
        Ok(utc) => write_formatted(ctx, out, format_rfc3339_local_with(utc, host_local_offset)),
        Err(e) => e,
    }
}

/// The UTC tick argument, the current time for a null pointer.
fn read_utc_or_now(ctx: &CpuContext, utc_addr: u64) -> Result<u64, HleResult> {
    if utc_addr == 0 {
        return Ok(utc_now_tick());
    }
    ctx.read_u64(utc_addr).ok_or(ERR_RTC_INVALID_POINTER)
}

/// sceRtcFormatRFC3339Precise(char* dateTime, const SceRtcTick* utc,
/// int timeZoneMinutes, int precision): a null tick formats the current time.
fn cb_format_rfc3339_precise(ctx: &mut CpuContext) -> HleResult {
    let (out, zone, precision) = (ctx.rdi(), ctx.rdx() as i32, ctx.rcx() as i32);
    if out == 0 {
        return ERR_RTC_INVALID_POINTER;
    }
    match read_utc_or_now(ctx, ctx.rsi()) {
        Ok(utc) => write_formatted(ctx, out, format_rfc3339_precise(utc, zone, precision)),
        Err(e) => e,
    }
}

/// sceRtcFormatRFC3339PreciseLocalTime(char* dateTime, const SceRtcTick* utc,
/// int precision): in the host's zone at that time.
fn cb_format_rfc3339_precise_local_time(ctx: &mut CpuContext) -> HleResult {
    let (out, precision) = (ctx.rdi(), ctx.rdx() as i32);
    if out == 0 {
        return ERR_RTC_INVALID_POINTER;
    }
    let utc = match read_utc_or_now(ctx, ctx.rsi()) {
        Ok(t) => t,
        Err(e) => return e,
    };
    if !(0..=6).contains(&precision) || utc > MAX_TICK {
        return ERR_RTC_INVALID_VALUE;
    }
    let zone = local_zone_minutes_with(utc, host_local_offset);
    write_formatted(ctx, out, format_rfc3339_precise(utc, zone, precision))
}

/// Byte at `i`, 0 past the end (the C string terminator).
fn byte_at(s: &[u8], i: usize) -> u8 {
    s.get(i).copied().unwrap_or(0)
}

/// Exactly `count` decimal digits.
fn parse_digits(s: &[u8], pos: &mut usize, count: usize) -> Option<i32> {
    let mut v = 0;
    for _ in 0..count {
        let c = byte_at(s, *pos);
        if !c.is_ascii_digit() {
            return None;
        }
        v = v * 10 + i32::from(c - b'0');
        *pos += 1;
    }
    Some(v)
}

/// Up to `max` decimal digits: (value, digits read).
fn read_digits(s: &[u8], pos: &mut usize, max: usize) -> (i32, usize) {
    let (mut v, mut n) = (0, 0);
    while n < max && byte_at(s, *pos).is_ascii_digit() {
        v = v * 10 + i32::from(byte_at(s, *pos) - b'0');
        *pos += 1;
        n += 1;
    }
    (v, n)
}

/// Case-insensitive match of an upper-case word at `pos`.
fn match_word(s: &[u8], pos: &mut usize, word: &str) -> bool {
    let w = word.as_bytes();
    if !(0..w.len()).all(|i| byte_at(s, *pos + i).to_ascii_uppercase() == w[i]) {
        return false;
    }
    *pos += w.len();
    true
}

/// A full name or its three-letter abbreviation: the index of the name.
fn parse_name(s: &[u8], pos: &mut usize, names: &[&str]) -> Option<usize> {
    names.iter().position(|n| match_word(s, pos, n) || match_word(s, pos, &n[..3]))
}

fn parse_weekday(s: &[u8], pos: &mut usize) -> bool {
    const NAMES: [&str; 7] = ["SUNDAY", "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY"];
    parse_name(s, pos, &NAMES).is_some()
}

fn parse_month(s: &[u8], pos: &mut usize) -> Option<i32> {
    const NAMES: [&str; 12] = [
        "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER",
        "DECEMBER",
    ];
    parse_name(s, pos, &NAMES).map(|i| i as i32 + 1)
}

/// Zone abbreviations and their offsets in minutes, matched in this order.
const ZONE_NAMES: [(&str, i64); 64] = [
    ("WETDST", 60), ("METDST", 120), ("EETDST", 180), ("SWT", 60), ("MEZ", 60), ("AESST", 660), ("ACSST", 630),
    ("CAST", 570), ("CADT", 630), ("SAST", 570), ("SADT", 630), ("WST", 480), ("WDT", 540),
    ("GMT", 0), ("WET", 0), ("BST", 60), ("WAT", -60), ("AST", -240), ("ADT", -180), ("EST", -300), ("EDT", -240),
    ("CST", -360), ("CDT", -300), ("MST", -420), ("MDT", -360), ("PST", -480), ("PDT", -420), ("YST", -540), ("YDT", -480),
    ("HDT", -540), ("CAT", -600), ("AHST", -600), ("NST", -150), ("NFT", -150), ("NDT", -90), ("IDLW", -720), ("CET", 60), ("MET", 60),
    ("DNT", 60), ("NOR", 60), ("SET", 60), ("MEWT", 60), ("MEST", 120), ("CEST", 120), ("SST", 120), ("FWT", 120), ("FST", 60), ("EET", 120), ("IST", 120),
    ("WAST", 420), ("WADT", 480), ("CCT", 480), ("AWST", 480), ("JST", 540), ("KST", 540), ("ACST", 570), ("EAST", 600),
    ("GST", 600), ("AEST", 600), ("NZT", 720), ("NZST", 720), ("NZDT", 780), ("IDLE", 720), ("JT", 450),
];

/// The zone after an RFC 2822 time: end of text, `±hhmm` (malformed
/// digits mean UTC), a known abbreviation or a military letter. Text after
/// the zone is ignored. None: not a zone.
fn parse_zone(s: &[u8], pos: usize) -> Option<i64> {
    let c = byte_at(s, pos);
    if c == 0 {
        return Some(0);
    }
    if c == b'+' || c == b'-' {
        let sign = if c == b'-' { -1 } else { 1 };
        let mut p = pos + 1;
        return Some(match (parse_digits(s, &mut p, 2), parse_digits(s, &mut p, 2)) {
            (Some(h), Some(m)) => sign * i64::from(h * 60 + m),
            _ => 0,
        });
    }
    let letter = c.to_ascii_uppercase();
    if !letter.is_ascii_uppercase() {
        return None;
    }
    if c == b'U' || byte_at(s, pos + 1) == b'T' {
        return Some(0);
    }
    if let Some(&(_, minutes)) = ZONE_NAMES.iter().find(|(name, _)| match_word(s, &mut pos.clone(), name)) {
        return Some(minutes);
    }
    match letter {
        b'J' => None,
        b'A'..=b'M' => Some(i64::from(letter - b'A') * 60),
        b'N'..=b'Y' => Some((i64::from(b'N') - i64::from(letter)) * 60),
        _ => Some(0),
    }
}

fn date_time(year: i32, month: i32, day: i32, hour: i32, minute: i32, second: i32, microsecond: u32) -> RtcDateTime {
    RtcDateTime {
        year: year as u16,
        month: month as u16,
        day: day as u16,
        hour: hour as u16,
        minute: minute as u16,
        second: second as u16,
        microsecond,
    }
}

/// `29 Feb 2024 12:34:56 +0130` (also `29-feb-24`, no seconds, no zone).
fn parse_rfc2822_date_time(s: &[u8], mut pos: usize) -> Option<(RtcDateTime, i64)> {
    let sep = |c: u8| c == b' ' || c == b'-';
    let (day, n) = read_digits(s, &mut pos, 2);
    if n == 0 || !sep(byte_at(s, pos)) {
        return None;
    }
    pos += 1;
    let month = parse_month(s, &mut pos)?;
    if !sep(byte_at(s, pos)) {
        return None;
    }
    pos += 1;
    let (mut year, digits) = read_digits(s, &mut pos, 4);
    if (digits != 2 && digits != 4) || byte_at(s, pos) != b' ' {
        return None;
    }
    pos += 1;
    if digits == 2 {
        year += if year < 50 { 2000 } else { 1900 };
    }
    let (hour, n) = read_digits(s, &mut pos, 2);
    if n == 0 || hour > 25 || byte_at(s, pos) != b':' {
        return None;
    }
    pos += 1;
    let (minute, n) = read_digits(s, &mut pos, 2);
    if n == 0 {
        return None;
    }
    let (mut second, mut zone) = (0, 0);
    if byte_at(s, pos) == b':' {
        pos += 1;
        let (sec, n) = read_digits(s, &mut pos, 2);
        if n == 0 {
            return None;
        }
        second = sec;
        if byte_at(s, pos) == b' ' {
            zone = parse_zone(s, pos + 1)?;
        }
    } else if byte_at(s, pos) != b' ' {
        return None;
    }
    Some((date_time(year, month, day, hour, minute, second, 0), zone))
}

/// asctime layout after the weekday: `Feb 29 12:34:56 2024` (`Jan  1`).
fn parse_asctime_date_time(s: &[u8], mut pos: usize) -> Option<RtcDateTime> {
    let month = parse_month(s, &mut pos)?;
    let expect = |pos: &mut usize, c: u8| {
        let ok = byte_at(s, *pos) == c;
        *pos += 1;
        ok
    };
    if !expect(&mut pos, b' ') {
        return None;
    }
    let (day, n) = if byte_at(s, pos) == b' ' {
        pos += 1;
        read_digits(s, &mut pos, 1)
    } else {
        read_digits(s, &mut pos, 2)
    };
    if n == 0 || !expect(&mut pos, b' ') {
        return None;
    }
    let (hour, n) = read_digits(s, &mut pos, 2);
    if n == 0 || !expect(&mut pos, b':') {
        return None;
    }
    let (minute, n) = read_digits(s, &mut pos, 2);
    if n == 0 || !expect(&mut pos, b':') {
        return None;
    }
    let (second, n) = read_digits(s, &mut pos, 2);
    if n == 0 || !expect(&mut pos, b' ') {
        return None;
    }
    let (year, n) = read_digits(s, &mut pos, 4);
    if n != 4 {
        return None;
    }
    Some(date_time(year, month, day, hour, minute, second, 0))
}

/// Strict RFC 3339 for sceRtcParseDateTime: `YYYY-MM-DDThh:mm:ss`, optional
/// fraction, then `Z` or `±hh:mm` (case-insensitive `T`/`Z`). A leap second
/// (`:60`) is accepted and only applies the zone.
fn parse_strict_rfc3339(s: &[u8], mut pos: usize) -> Result<(RtcDateTime, i64), HleResult> {
    let bad = Err(ERR_RTC_BAD_PARSE);
    let field = |pos: &mut usize, n: usize, sep: Option<u8>| -> Option<i32> {
        let v = parse_digits(s, pos, n)?;
        if let Some(c) = sep {
            if byte_at(s, *pos) != c {
                return None;
            }
            *pos += 1;
        }
        Some(v)
    };
    let (Some(year), Some(month), Some(day)) =
        (field(&mut pos, 4, Some(b'-')), field(&mut pos, 2, Some(b'-')), field(&mut pos, 2, None))
    else {
        return bad;
    };
    if !matches!(byte_at(s, pos), b'T' | b't') {
        return bad;
    }
    pos += 1;
    let (Some(hour), Some(minute), Some(second)) =
        (field(&mut pos, 2, Some(b':')), field(&mut pos, 2, Some(b':')), field(&mut pos, 2, None))
    else {
        return bad;
    };
    let mut microsecond = 0u32;
    if byte_at(s, pos) == b'.' {
        pos += 1;
        let mut scale = 100_000u32;
        while byte_at(s, pos).is_ascii_digit() {
            microsecond += u32::from(byte_at(s, pos) - b'0') * scale;
            scale /= 10;
            pos += 1;
        }
    }
    let mut zone = 0i64;
    match byte_at(s, pos) {
        c @ (b'+' | b'-') => {
            pos += 1;
            let (Some(h), Some(m)) = (field(&mut pos, 2, Some(b':')), field(&mut pos, 2, None)) else {
                return bad;
            };
            zone = i64::from(h * 60 + m);
            if c == b'-' {
                zone = -zone;
            }
        }
        b'Z' | b'z' => {}
        _ => return bad,
    }
    let t = date_time(year, month, day, hour, minute, second, microsecond);
    let v = validate(&t);
    if v != OK && (v != ERR_RTC_INVALID_SECOND || second != 60) {
        return Err(v);
    }
    Ok((t, zone))
}

/// sceRtcParseDateTime text: leading blanks, then RFC 3339, or a weekday
/// followed by an RFC 2822 or asctime date. The date and its zone offset
/// (minutes).
fn parse_date_time(s: &[u8]) -> Result<(RtcDateTime, i64), HleResult> {
    let mut pos = 0;
    while matches!(byte_at(s, pos), b' ' | b'\t') {
        pos += 1;
    }
    if byte_at(s, pos).is_ascii_digit() {
        return parse_strict_rfc3339(s, pos);
    }
    if !parse_weekday(s, &mut pos) {
        return Err(ERR_RTC_BAD_PARSE);
    }
    if byte_at(s, pos) == b',' {
        pos += 1;
    }
    while matches!(byte_at(s, pos), b' ' | b'\t') {
        pos += 1;
    }
    let parsed = if byte_at(s, pos).is_ascii_digit() {
        parse_rfc2822_date_time(s, pos)
    } else {
        parse_asctime_date_time(s, pos).map(|t| (t, 0))
    };
    parsed.ok_or(ERR_RTC_BAD_PARSE)
}

/// The tick stored for a parsed date: its own tick when the date is valid
/// (otherwise the previous value), moved back by the zone offset.
fn parsed_tick(previous: u64, t: &RtcDateTime, zone_minutes: i64) -> u64 {
    let local = if validate(t) == OK { rtc_to_tick(t) } else { previous };
    local.wrapping_sub((zone_minutes * US_PER_MINUTE as i64) as u64)
}

/// sceRtcParseDateTime(SceRtcTick* utc, const char* dateTime)
fn cb_parse_date_time(ctx: &mut CpuContext) -> HleResult {
    let (utc_addr, text_addr) = (ctx.rdi(), ctx.rsi());
    if utc_addr == 0 || text_addr == 0 {
        return ERR_RTC_INVALID_POINTER;
    }
    let Some(text) = ctx.read_cstr_bytes(text_addr, MAX_RFC3339_INPUT) else {
        return ERR_RTC_INVALID_POINTER;
    };
    let (t, zone) = match parse_date_time(&text) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let Some(previous) = ctx.read_u64(utc_addr) else { return ERR_RTC_INVALID_POINTER };
    if ctx.write_u64(utc_addr, parsed_tick(previous, &t, zone)) { OK } else { ERR_RTC_INVALID_POINTER }
}

pub fn register(reg: &mut HleRegistry) {
    reg.register_fn("eiuobaF-hK4", "sceRtcFormatRFC2822", L, cb_format_rfc2822);
    reg.register_fn("AxHBk3eat04", "sceRtcFormatRFC2822LocalTime", L, cb_format_rfc2822_local_time);
    reg.register_fn("DwuHIlLGW8I", "sceRtcFormatRFC3339LocalTime", L, cb_format_rfc3339_local_time);
    reg.register_fn("NxEI1KByvCI", "sceRtcParseDateTime", L, cb_parse_date_time);
    const L: &str = "libSceRtc";
    reg.register_fn("WJ3rqFwymew", "sceRtcFormatRFC3339", L, cb_format_rfc3339);
    reg.register_fn("lja0nNPWojg", "sceRtcFormatRFC3339Precise", L, cb_format_rfc3339_precise);
    reg.register_fn("tOZ6fwwHZOA", "sceRtcFormatRFC3339PreciseLocalTime", L, cb_format_rfc3339_precise_local_time);
    reg.register_fn("99bMGglFW3I", "sceRtcParseRFC3339", L, cb_parse_rfc3339);
    reg.register_fn("lPEBYdVX0XQ", "sceRtcCheckValid", L, cb_check_valid);
    reg.register_fn("fNaZ4DbzHAE", "sceRtcCompareTick", L, cb_compare_tick);
    reg.register_fn("8Yr143yEnRo", "sceRtcConvertLocalTimeToUtc", L, cb_convert_local_time_to_utc);
    reg.register_fn("M1TvFst-jrM", "sceRtcConvertUtcToLocalTime", L, cb_convert_utc_to_local_time);
    reg.register_fn("8SljQx6pDP8", "sceRtcEnd", L, cb_ok);
    reg.register_fn("LN3Zcb72Q0c", "sceRtcGetCurrentAdNetworkTick", L, cb_get_current_ad_network_tick);
    reg.register_fn("d4fHLCGmY80", "sceRtcSetCurrentTick", L, cb_set_current_tick);
    reg.register_fn("qhDBtIo+auw", "sceRtcSetCurrentNetworkTick", L, cb_set_current_network_tick);
    reg.register_fn("sV2tK+yOhBU", "sceRtcSetCurrentAdNetworkTick", L, cb_set_current_ad_network_tick);
    reg.register_fn("VLDUPKmw5L8", "sceRtcSetCurrentDebugNetworkTick", L, cb_set_current_debug_network_tick);
    reg.register_fn("8lfvnRMqwEM", "sceRtcGetCurrentClock", L, cb_get_current_clock);
    reg.register_fn("ZPD1YOKI+Kw", "sceRtcGetCurrentClockLocalTime", L, cb_get_current_clock_local_time);
    reg.register_fn("Ot1DE3gif84", "sceRtcGetCurrentDebugNetworkTick", L, cb_get_current_debug_network_tick);
    reg.register_fn("zO9UL3qIINQ", "sceRtcGetCurrentNetworkTick", L, cb_get_current_network_tick);
    reg.register_fn("HWxHOdbM-Pg", "sceRtcGetCurrentRawNetworkTick", L, cb_get_current_raw_network_tick);
    reg.register_fn("18B2NS1y9UU", "sceRtcGetCurrentTick", L, cb_get_current_tick);
    reg.register_fn("CyIK-i4XdgQ", "sceRtcGetDayOfWeek", L, cb_get_day_of_week);
    reg.register_fn("3O7Ln8AqJ1o", "sceRtcGetDaysInMonth", L, cb_get_days_in_month);
    reg.register_fn("E7AR4o7Ny7E", "sceRtcGetDosTime", L, cb_get_dos_time);
    reg.register_fn("8w-H19ip48I", "sceRtcGetTick", L, cb_get_tick);
    reg.register_fn("jMNwqYr4R-k", "sceRtcGetTickResolution", L, cb_get_tick_resolution);
    reg.register_fn("BtqmpTRXHgk", "sceRtcGetTime_t", L, cb_get_time_t);
    reg.register_fn("jfRO0uTjtzA", "sceRtcGetWin32FileTime", L, cb_get_win32_file_time);
    reg.register_fn("LlodCMDbk3o", "sceRtcInit", L, cb_ok);
    reg.register_fn("Ug8pCwQvh0c", "sceRtcIsLeapYear", L, cb_is_leap_year);
    reg.register_fn("NR1J0N7L2xY", "sceRtcTickAddDays", L, cb_tick_add_days);
    reg.register_fn("MDc5cd8HfCA", "sceRtcTickAddHours", L, cb_tick_add_hours);
    reg.register_fn("XPIiw58C+GM", "sceRtcTickAddMicroseconds", L, cb_tick_add_microseconds);
    reg.register_fn("mn-tf4QiFzk", "sceRtcTickAddMinutes", L, cb_tick_add_minutes);
    reg.register_fn("CL6y9q-XbuQ", "sceRtcTickAddMonths", L, cb_tick_add_months);
    reg.register_fn("07O525HgICs", "sceRtcTickAddSeconds", L, cb_tick_add_seconds);
    reg.register_fn("AqVMssr52Rc", "sceRtcTickAddTicks", L, cb_tick_add_ticks);
    reg.register_fn("gI4t194c2W8", "sceRtcTickAddWeeks", L, cb_tick_add_weeks);
    reg.register_fn("-5y2uJ62qS8", "sceRtcTickAddYears", L, cb_tick_add_years);
    reg.register_fn("ueega6v3GUw", "sceRtcSetTick", L, cb_set_tick);
    reg.register_fn("aYPCd1cChyg", "sceRtcSetDosTime", L, cb_set_dos_time);
    reg.register_fn("bDEVVP4bTjQ", "sceRtcSetTime_t", L, cb_set_time_t);
    reg.register_fn("n5JiAJXsbcs", "sceRtcSetWin32FileTime", L, cb_set_win32_file_time);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    fn dt(year: u16, month: u16, day: u16, hour: u16, minute: u16, second: u16, us: u32) -> RtcDateTime {
        RtcDateTime { year, month, day, hour, minute, second, microsecond: us }
    }

    fn tick(t: RtcDateTime) -> u64 {
        assert_eq!(validate(&t), OK);
        rtc_to_tick(&t)
    }

    #[test]
    fn rfc3339_format() {
        assert_eq!(ps5nid::compute_nid("sceRtcFormatRFC3339"), "WJ3rqFwymew");
        assert_eq!(ps5nid::compute_nid("sceRtcParseRFC3339"), "99bMGglFW3I");
        let utc = tick(dt(2024, 2, 29, 23, 30, 5, 123_456));
        assert_eq!(format_rfc3339(utc, 0).unwrap(), "2024-02-29T23:30:05.12Z");
        assert_eq!(format_rfc3339(utc, 90).unwrap(), "2024-03-01T01:00:05.12+01:30");
        assert_eq!(format_rfc3339(utc, -300).unwrap(), "2024-02-29T18:30:05.12-05:00");
        // No zone range limit; the zone text is cut to seven characters.
        assert_eq!(format_rfc3339(utc, 1440).unwrap(), "2024-03-01T23:30:05.12+24:00");
        assert_eq!(format_rfc3339(utc, 100_000).unwrap(), "2024-05-09T10:10:05.12+1666:4");
        assert_eq!(format_rfc3339(0, -1), Err(ERR_RTC_INVALID_VALUE));
        assert_eq!(format_rfc3339(MAX_TICK, 1), Err(ERR_RTC_INVALID_VALUE));
    }

    #[test]
    fn rfc3339_precise_format() {
        let utc = tick(dt(2026, 1, 2, 3, 4, 5, 123_456));
        assert_eq!(format_rfc3339_precise(utc, 0, 0).unwrap(), "2026-01-02T03:04:05Z");
        assert_eq!(format_rfc3339_precise(utc, 0, 3).unwrap(), "2026-01-02T03:04:05.123Z");
        assert_eq!(format_rfc3339_precise(utc, 0, 6).unwrap(), "2026-01-02T03:04:05.123456Z");
        assert_eq!(format_rfc3339_precise(utc, 90, 1).unwrap(), "2026-01-02T04:34:05.1+01:30");
        assert_eq!(format_rfc3339_precise(utc, -1439, 0).unwrap(), "2026-01-01T03:05:05-23:59");
        assert_eq!(format_rfc3339_precise(utc, 0, 7), Err(ERR_RTC_INVALID_VALUE));
        assert_eq!(format_rfc3339_precise(utc, 1440, 0), Err(ERR_RTC_INVALID_VALUE));
        assert_eq!(format_rfc3339_precise(0, -1, 0), Err(ERR_RTC_INVALID_VALUE));
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(crate::CpuRegister::Rdi, 0);
        assert_eq!(cb_format_rfc3339_precise(&mut ctx), ERR_RTC_INVALID_POINTER);
        assert_eq!(cb_format_rfc3339_precise_local_time(&mut ctx), ERR_RTC_INVALID_POINTER);
    }

    #[test]
    fn rfc3339_parse() {
        let utc = tick(dt(2024, 2, 29, 23, 30, 5, 123_456));
        let ok = |t| ParseOutcome { tick: Some(t), result: OK };
        let cs = utc - utc % 10_000;
        for zone in [0, 90, -300, 1439, -1439] {
            let text = format_rfc3339(utc, zone).unwrap();
            assert_eq!(parse_rfc3339(text.as_bytes()), ok(cs), "{text}");
        }
        // A missing zone means UTC; text after `Z` is ignored; fractions
        // keep their first six digits.
        assert_eq!(parse_rfc3339(b"2024-02-29T23:30:05.123456"), ok(utc));
        assert_eq!(parse_rfc3339(b"2024-02-29T23:30:05.1234567Zjunk"), ok(utc));
        assert_eq!(parse_rfc3339(b"2024-02-29T23:30:05.Z"), ok(utc - 123_456));
        // `%hu` fields: white space, signs and single digits are accepted.
        assert_eq!(parse_rfc3339(b" 2024-2-29T 23:30:+5Z"), ok(utc - 123_456));
        // Lower-case or space separators are rejected.
        let bad = ParseOutcome { tick: None, result: ERR_RTC_INVALID_VALUE };
        assert_eq!(parse_rfc3339(b"2024-02-29t23:30:05Z"), bad);
        assert_eq!(parse_rfc3339(b"2024-02-29 23:30:05Z"), bad);
        assert_eq!(parse_rfc3339(b""), bad);
        // Field validation codes.
        assert_eq!(parse_rfc3339(b"2023-02-29T00:00:00Z").result, ERR_RTC_INVALID_DAY);
        assert_eq!(parse_rfc3339(b"2023-02-01T24:00:00Z").result, ERR_RTC_INVALID_HOUR);
        // A bad zone still stores the local tick.
        let local = tick(dt(2024, 2, 29, 23, 30, 5, 0));
        assert_eq!(parse_rfc3339(b"2024-02-29T23:30:05x"), ParseOutcome { tick: Some(local), result: ERR_RTC_INVALID_VALUE });
        assert_eq!(parse_rfc3339(b"2024-02-29T23:30:05+1"), ParseOutcome { tick: Some(local), result: ERR_RTC_INVALID_VALUE });
        assert_eq!(parse_rfc3339(b"0001-01-01T00:00:00+00:01"), ParseOutcome { tick: Some(0), result: ERR_RTC_INVALID_VALUE });
        // Zone hours are not range checked.
        assert_eq!(parse_rfc3339(b"2024-02-29T23:30:05+99:00"), ok(local - 99 * US_PER_HOUR));
        assert_eq!(parse_rfc3339(b"2024-02-29T23:30:05-1:30"), ok(local + 90 * US_PER_MINUTE));
    }

    #[test]
    fn epoch_ticks() {
        assert_eq!(UNIX_EPOCH_TICKS, 62_135_596_800_000_000);
        assert_eq!(WIN32_FILETIME_EPOCH_TICKS, 50_491_123_200_000_000);
        assert_eq!(tick(dt(1, 1, 1, 0, 0, 0, 0)), 0);
        assert_eq!(tick(dt(1970, 1, 1, 0, 0, 0, 0)), UNIX_EPOCH_TICKS);
        assert_eq!(tick(dt(1601, 1, 1, 0, 0, 0, 0)), WIN32_FILETIME_EPOCH_TICKS);
        assert_eq!(tick(dt(9999, 12, 31, 23, 59, 59, 999_999)), MAX_TICK);
    }

    #[test]
    fn tick_roundtrip() {
        let samples = [
            dt(1, 1, 1, 0, 0, 0, 0),
            dt(1600, 2, 29, 12, 34, 56, 789_012),
            dt(1900, 2, 28, 23, 59, 59, 999_999),
            dt(2000, 2, 29, 0, 0, 0, 1),
            dt(2024, 12, 31, 7, 8, 9, 10),
            dt(9999, 12, 31, 23, 59, 59, 999_999),
        ];
        for s in samples {
            assert_eq!(tick_to_rtc(tick(s)), Some(s));
        }
        assert_eq!(tick_to_rtc(MAX_TICK + 1), None);
        assert_eq!(tick_to_rtc(u64::MAX), None);
        // Every day boundary agrees with the civil day count.
        for days in (0..3_652_059i64).step_by(97) {
            let t = tick_to_rtc(days as u64 * US_PER_DAY).unwrap();
            assert_eq!(days_from_civil(t.year as i64, t.month as i64, t.day as i64), days);
        }
    }

    #[test]
    fn struct_layout() {
        let t = dt(2023, 11, 5, 13, 45, 30, 123_456);
        let b = t.to_bytes();
        assert_eq!(&b[0..2], &2023u16.to_le_bytes());
        assert_eq!(&b[12..16], &123_456u32.to_le_bytes());
        assert_eq!(RtcDateTime::from_bytes(&b), t);
    }

    #[test]
    fn validation_codes() {
        assert_eq!(validate(&dt(2024, 2, 29, 0, 0, 0, 0)), OK);
        assert_eq!(validate(&dt(0, 1, 1, 0, 0, 0, 0)), ERR_RTC_INVALID_YEAR);
        assert_eq!(validate(&dt(10000, 1, 1, 0, 0, 0, 0)), ERR_RTC_INVALID_YEAR);
        assert_eq!(validate(&dt(2024, 13, 1, 0, 0, 0, 0)), ERR_RTC_INVALID_MONTH);
        assert_eq!(validate(&dt(2023, 2, 29, 0, 0, 0, 0)), ERR_RTC_INVALID_DAY);
        assert_eq!(validate(&dt(2023, 4, 0, 0, 0, 0, 0)), ERR_RTC_INVALID_DAY);
        assert_eq!(validate(&dt(2023, 4, 1, 24, 0, 0, 0)), ERR_RTC_INVALID_HOUR);
        assert_eq!(validate(&dt(2023, 4, 1, 0, 60, 0, 0)), ERR_RTC_INVALID_MINUTE);
        assert_eq!(validate(&dt(2023, 4, 1, 0, 0, 60, 0)), ERR_RTC_INVALID_SECOND);
        assert_eq!(validate(&dt(2023, 4, 1, 0, 0, 0, 1_000_000)), ERR_RTC_INVALID_MICROSECOND);
        assert_eq!(ERR_RTC_INVALID_YEAR, 0xFFFF_FFFF_80B5_0008u64 as i64);
    }

    #[test]
    fn calendar_queries() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        let mut q = |f: fn(&mut CpuContext) -> HleResult, y: i64, m: i64, d: i64| {
            ctx.set(Rdi, y as u64);
            ctx.set(Rsi, m as u64);
            ctx.set(Rdx, d as u64);
            f(&mut ctx)
        };
        assert_eq!(q(cb_get_day_of_week, 1, 1, 1), 1);
        assert_eq!(q(cb_get_day_of_week, 1970, 1, 1), 4);
        assert_eq!(q(cb_get_day_of_week, 2026, 10, 7), 3);
        // Errors name the offending field.
        assert_eq!(q(cb_get_day_of_week, 0, 1, 1), ERR_RTC_INVALID_YEAR);
        assert_eq!(q(cb_get_day_of_week, 2024, 13, 1), ERR_RTC_INVALID_MONTH);
        assert_eq!(q(cb_get_day_of_week, 2023, 2, 29), ERR_RTC_INVALID_DAY);
        assert_eq!(q(cb_get_day_of_week, 2023, 2, -1), ERR_RTC_INVALID_DAY);
        assert_eq!(q(cb_get_days_in_month, 2024, 2, 0), 29);
        assert_eq!(q(cb_get_days_in_month, 1900, 2, 0), 28);
        assert_eq!(q(cb_get_days_in_month, 10000, 2, 0), ERR_RTC_INVALID_YEAR);
        assert_eq!(q(cb_get_days_in_month, 2024, 0, 0), ERR_RTC_INVALID_MONTH);
        assert_eq!(q(cb_is_leap_year, 2000, 0, 0), 1);
        assert_eq!(q(cb_is_leap_year, 2100, 0, 0), 0);
        assert_eq!(q(cb_is_leap_year, -4, 0, 0), ERR_RTC_INVALID_YEAR);
    }

    /// sceRtcSetCurrent*Tick move this process's clocks (ad and debug
    /// network clocks follow the network clock until set); a tick before
    /// 1970 is INVALID_VALUE.
    #[test]
    fn process_clock_setters() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D88_4000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (src, out) = (BASE, BASE + 0x100);
        let mut call = |f: fn(&mut CpuContext) -> HleResult, rdi: u64| {
            ctx.set(Rdi, rdi);
            f(&mut ctx)
        };
        let set = tick(dt(2030, 6, 1, 0, 0, 0, 0));
        assert!(vm.write(src, &set.to_le_bytes()).is_ok());
        assert_eq!(call(cb_set_current_network_tick, src), OK);
        assert_eq!(call(cb_get_current_ad_network_tick, out), OK);
        let got = vm_read_u64(&vm, out);
        assert!(got >= set && got < set + 5 * US_PER_SECOND, "follows the network clock");
        assert_eq!(call(cb_get_current_raw_network_tick, out), OK);
        assert!(vm_read_u64(&vm, out) < set, "the raw clock is the host's");
        assert!(vm.write(src, &(UNIX_EPOCH_TICKS - 1).to_le_bytes()).is_ok());
        assert_eq!(call(cb_set_current_debug_network_tick, src), ERR_RTC_INVALID_VALUE);
        assert_eq!(call(cb_set_current_ad_network_tick, 0), ERR_RTC_INVALID_POINTER);
        CLOCKS.lock().unwrap()[1] = None;
    }

    #[test]
    fn guest_tick_functions() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D88_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (t, a, b) = (BASE, BASE + 0x100, BASE + 0x200);
        let mut call = |f: fn(&mut CpuContext) -> HleResult, rdi: u64, rsi: u64, rdx: u64| {
            ctx.set(Rdi, rdi);
            ctx.set(Rsi, rsi);
            ctx.set(Rdx, rdx);
            f(&mut ctx)
        };
        // Current-time queries report NULL outputs as uninitialised.
        assert_eq!(call(cb_get_current_tick, 0, 0, 0), RTC_ERROR_DATETIME_UNINITIALIZED);
        assert_eq!(call(cb_get_current_clock, 0, 0, 0), RTC_ERROR_DATETIME_UNINITIALIZED);
        assert_eq!(call(cb_get_current_clock_local_time, 0, 0, 0), RTC_ERROR_DATETIME_UNINITIALIZED);
        // Any zone offset is accepted.
        assert_eq!(call(cb_get_current_clock, t, 6000, 0), OK);
        assert_eq!(call(cb_get_current_clock_local_time, t, 0, 0), OK);
        assert_eq!(vm_read_rtc(&vm, t).microsecond % 1000, 0);

        // SetTick past 9999 -> INVALID_VALUE.
        assert!(vm.write(a, &(MAX_TICK + 1).to_le_bytes()).is_ok());
        assert_eq!(call(cb_set_tick, t, a, 0), ERR_RTC_INVALID_VALUE);

        // Timezone conversions reject such ticks without writing, also in place.
        for invalid in [MAX_TICK + 1, 1u64 << 63, u64::MAX] {
            assert!(vm.write(a, &invalid.to_le_bytes()).is_ok());
            assert!(vm.write(b, &123u64.to_le_bytes()).is_ok());
            for f in [cb_convert_utc_to_local_time as fn(&mut CpuContext) -> HleResult, cb_convert_local_time_to_utc] {
                assert_eq!(call(f, a, b, 0), ERR_RTC_INVALID_VALUE);
                assert_eq!(vm_read_u64(&vm, b), 123);
                assert_eq!(call(f, a, a, 0), ERR_RTC_INVALID_VALUE);
                assert_eq!(vm_read_u64(&vm, a), invalid);
            }
        }
        // The last valid tick still converts.
        assert!(vm.write(a, &MAX_TICK.to_le_bytes()).is_ok());
        assert_eq!(call(cb_convert_utc_to_local_time, a, b, 0), OK);
        assert_eq!(call(cb_convert_local_time_to_utc, a, b, 0), OK);

        // Tick additions wrap and take 32-bit hour/day/week counts.
        assert!(vm.write(a, &5u64.to_le_bytes()).is_ok());
        assert_eq!(call(cb_tick_add_ticks, b, a, (-6i64) as u64), OK);
        assert_eq!(vm_read_u64(&vm, b), u64::MAX);
        assert_eq!(call(cb_tick_add_days, b, a, 0xFFFF_FFFF_0000_0001), OK);
        assert_eq!(vm_read_u64(&vm, b), 5 + US_PER_DAY);
        assert_eq!(call(cb_tick_add_weeks, b, a, i32::MIN as u32 as u64), OK, "no overflow panic");
        // Months and years: the day is clamped, a result outside 1..=9999
        // leaves dst alone and still succeeds (measured on a PS5).
        let jan31 = tick(dt(2023, 1, 31, 10, 0, 0, 7));
        assert!(vm.write(a, &jan31.to_le_bytes()).is_ok());
        assert_eq!(call(cb_tick_add_months, b, a, 1), OK);
        assert_eq!(vm_read_u64(&vm, b), tick(dt(2023, 2, 28, 10, 0, 0, 7)));
        assert_eq!(call(cb_tick_add_months, b, a, (-13i64) as u64), OK);
        assert_eq!(vm_read_u64(&vm, b), tick(dt(2021, 12, 31, 10, 0, 0, 7)));
        assert_eq!(call(cb_tick_add_years, b, a, 0x1_0001), OK, "a 16-bit year count");
        assert_eq!(vm_read_u64(&vm, b), tick(dt(2024, 1, 31, 10, 0, 0, 7)));
        assert!(vm.write(b, &99u64.to_le_bytes()).is_ok());
        assert_eq!(call(cb_tick_add_years, b, a, 8000), OK);
        assert_eq!(vm_read_u64(&vm, b), 99, "past 9999: untouched");
        assert_eq!(call(cb_tick_add_months, b, a, (-30_000i64) as u64), OK);
        assert_eq!(vm_read_u64(&vm, b), 99, "before year 1: untouched");
        assert_eq!(call(cb_tick_add_months, 0, a, 1), ERR_RTC_INVALID_POINTER);
        assert!(vm.write(a, &5u64.to_le_bytes()).is_ok());
        assert_eq!(call(cb_tick_add_hours, b, a, 0xFFFF_FFFF), OK);
        assert_eq!(vm_read_u64(&vm, b), 5u64.wrapping_sub(US_PER_HOUR));
        assert_eq!(call(cb_tick_add_seconds, b, a, 2), OK);
        assert_eq!(vm_read_u64(&vm, b), 5 + 2 * US_PER_SECOND);

        // time_t / FILETIME round trips and range checks.
        assert_eq!(call(cb_set_time_t, t, 86_400, 0), OK);
        assert_eq!(vm_read_rtc(&vm, t), dt(1970, 1, 2, 0, 0, 0, 0));
        assert_eq!(call(cb_set_time_t, t, (-1i64) as u64, 0), ERR_RTC_INVALID_VALUE);
        assert_eq!(call(cb_set_time_t, t, i64::MAX as u64, 0), ERR_RTC_INVALID_VALUE);
        assert_eq!(time_t_tick(18_446_681_938_112), None);
        assert_eq!(call(cb_get_time_t, t, b, 0), OK);
        assert_eq!(vm_read_u64(&vm, b), 86_400);
        assert_eq!(call(cb_set_win32_file_time, t, u64::MAX, 0), ERR_RTC_INVALID_VALUE);
        assert_eq!(call(cb_set_win32_file_time, t, 10, 0), OK);
        assert_eq!(vm_read_rtc(&vm, t), dt(1601, 1, 1, 0, 0, 0, 1));
        assert_eq!(call(cb_get_win32_file_time, t, b, 0), OK);
        assert_eq!(vm_read_u64(&vm, b), 10);

        // Parse: the local tick is stored even when the zone is bad.
        let text = BASE + 0x300;
        assert!(vm.write(text, b"2000-01-01T00:00:00?\0").is_ok());
        assert_eq!(call(cb_parse_rfc3339, b, text, 0), ERR_RTC_INVALID_VALUE);
        assert_eq!(vm_read_u64(&vm, b), tick(dt(2000, 1, 1, 0, 0, 0, 0)));
        // Format with a NULL tick uses the current time.
        assert_eq!(call(cb_format_rfc3339, text, 0, 0), OK);
        assert_eq!(call(cb_format_rfc3339, 0, b, 0), ERR_RTC_INVALID_POINTER);
    }

    fn vm_read_u64(vm: &ps5vm::VirtualMemory, a: u64) -> u64 {
        let mut b = [0u8; 8];
        vm.read(a, &mut b).unwrap();
        u64::from_le_bytes(b)
    }

    fn vm_read_rtc(vm: &ps5vm::VirtualMemory, a: u64) -> RtcDateTime {
        let mut b = [0u8; 16];
        vm.read(a, &mut b).unwrap();
        RtcDateTime::from_bytes(&b)
    }

    #[test]
    fn dos_time() {
        let t = dt(2021, 6, 15, 14, 33, 58, 0);
        assert_eq!(rtc_from_dos_time(dos_time_from_rtc(&t)), t);
        let t = dt(1980, 1, 1, 0, 0, 59, 0);
        assert_eq!(rtc_from_dos_time(dos_time_from_rtc(&t)).second, 58);
    }

    #[test]
    fn calendar_add() {
        let jan31 = tick(dt(2023, 1, 31, 10, 20, 30, 40));
        assert_eq!(tick_to_rtc(add_months_wrapping(jan31, 1).unwrap()).unwrap(), dt(2023, 2, 28, 10, 20, 30, 40));
        assert_eq!(tick_to_rtc(add_months_wrapping(jan31, -2).unwrap()).unwrap(), dt(2022, 11, 30, 10, 20, 30, 40));
        assert_eq!(tick_to_rtc(add_months_wrapping(jan31, 13).unwrap()).unwrap(), dt(2024, 2, 29, 10, 20, 30, 40));
        let feb29 = tick(dt(2024, 2, 29, 0, 0, 0, 0));
        assert_eq!(tick_to_rtc(add_months_wrapping(feb29, 12).unwrap()).unwrap(), dt(2025, 2, 28, 0, 0, 0, 0));
        assert_eq!(add_months_wrapping(feb29, 96_000), None);
        assert_eq!(add_months_wrapping(0, -12), None);
        assert_eq!(add_months_wrapping(0, 120_001), None);
    }

    #[test]
    fn timezone_conversion() {
        // Synthetic zone: UTC+1, DST (UTC+2) for unix seconds in [1000_000, 2000_000).
        let zone = |u: i64| -> Option<(i64, bool)> {
            Some(if (1_000_000..2_000_000).contains(&u) { (7200, true) } else { (3600, false) })
        };
        let base = UNIX_EPOCH_TICKS;
        let s = US_PER_SECOND;
        assert_eq!(utc_to_local_with(base, zone), base + 3600 * s);
        assert_eq!(local_to_utc_with(base + 3600 * s, zone), base);
        let u = base + 1_500_000 * s;
        assert_eq!(local_to_utc_with(utc_to_local_with(u, zone), zone), u);
        // Spring-forward gap: uses the offset before the transition.
        let gap = base + (1_000_000 + 3600 + 1800) * s;
        assert_eq!(local_to_utc_with(gap, zone), gap - 3600 * s);
        // Fall-back overlap resolves to standard time.
        let l = base + (2_000_000 + 3600 + 1800) * s;
        assert_eq!(local_to_utc_with(l, zone), l - 3600 * s);
        // Clamped at the representable range; never fails.
        assert_eq!(utc_to_local_with(MAX_TICK, zone), MAX_TICK);
        assert_eq!(local_to_utc_with(0, zone), 0);
        assert_eq!(utc_to_local_with(u64::MAX, |_| None), u64::MAX);
    }

    const LEAP_DAY_TICK: u64 = 63_844_806_896_789_000;
    const UNIX_EPOCH_TICK: u64 = 62_135_596_800_000_000;

    #[test]
    fn new_export_nids_match_names() {
        for (nid, name) in [
            ("eiuobaF-hK4", "sceRtcFormatRFC2822"),
            ("AxHBk3eat04", "sceRtcFormatRFC2822LocalTime"),
            ("DwuHIlLGW8I", "sceRtcFormatRFC3339LocalTime"),
            ("lja0nNPWojg", "sceRtcFormatRFC3339Precise"),
            ("tOZ6fwwHZOA", "sceRtcFormatRFC3339PreciseLocalTime"),
            ("NxEI1KByvCI", "sceRtcParseDateTime"),
        ] {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// Port of the guest RFC 2822 formatting checks.
    #[test]
    fn rfc2822_format() {
        assert_eq!(tick(dt(2024, 2, 29, 12, 34, 56, 789_000)), LEAP_DAY_TICK);
        let f = |t, z| format_rfc2822(t, z);
        assert_eq!(f(LEAP_DAY_TICK, 0).unwrap(), "Thu, 29 Feb 2024 12:34:56 +0000");
        assert_eq!(f(LEAP_DAY_TICK, 90).unwrap(), "Thu, 29 Feb 2024 14:04:56 +0130");
        assert_eq!(f(LEAP_DAY_TICK, -300).unwrap(), "Thu, 29 Feb 2024 07:34:56 -0500");
        assert_eq!(f(LEAP_DAY_TICK, -30).unwrap(), "Thu, 29 Feb 2024 12:04:56 -0030");
        assert_eq!(f(LEAP_DAY_TICK, 65).unwrap(), "Thu, 29 Feb 2024 13:39:56 +0105");
        assert_eq!(f(LEAP_DAY_TICK, 1439).unwrap(), "Fri, 01 Mar 2024 12:33:56 +2359");
        assert_eq!(f(LEAP_DAY_TICK, -1439).unwrap(), "Wed, 28 Feb 2024 12:35:56 -2359");
        let week = ["Sun, 03 Mar", "Mon, 04 Mar", "Tue, 05 Mar", "Wed, 06 Mar", "Thu, 07 Mar", "Fri, 08 Mar", "Sat, 09 Mar"];
        for (i, w) in week.iter().enumerate() {
            assert!(f(tick(dt(2024, 3, 3 + i as u16, 0, 0, 0, 0)), 0).unwrap().starts_with(w));
        }
        assert_eq!(f(0, 0).unwrap(), "Mon, 01 Jan 0001 00:00:00 +0000");
        assert_eq!(f(0, -1), Err(ERR_RTC_INVALID_VALUE));
        assert_eq!(f(MAX_TICK, 0).unwrap(), "Fri, 31 Dec 9999 23:59:59 +0000");
        assert_eq!(f(MAX_TICK, 1), Err(ERR_RTC_INVALID_VALUE));
        for z in [1440, -1440, i32::MIN, i32::MAX] {
            assert_eq!(f(LEAP_DAY_TICK, z), Err(ERR_RTC_INVALID_VALUE));
        }
    }

    /// The local-time formatters use the zone offset in force at the tick.
    #[test]
    fn local_time_formats() {
        let zone = |secs: i64| move |_: i64| Some((secs, false));
        assert_eq!(format_rfc2822_local_with(LEAP_DAY_TICK, zone(19_800)).unwrap(), "Thu, 29 Feb 2024 18:04:56 +0530");
        assert_eq!(format_rfc3339_local_with(LEAP_DAY_TICK, zone(19_800)).unwrap(), "2024-02-29T18:04:56.78+05:30");
        assert_eq!(format_rfc2822_local_with(LEAP_DAY_TICK, zone(-10_800)).unwrap(), "Thu, 29 Feb 2024 09:34:56 -0300");
        assert_eq!(format_rfc3339_local_with(LEAP_DAY_TICK, zone(-10_800)).unwrap(), "2024-02-29T09:34:56.78-03:00");
        assert_eq!(format_rfc3339_local_with(LEAP_DAY_TICK, zone(0)).unwrap(), "2024-02-29T12:34:56.78Z");
        for bad in [MAX_TICK + 1, 1 << 63, u64::MAX] {
            assert_eq!(format_rfc2822_local_with(bad, zone(0)), Err(ERR_RTC_INVALID_VALUE));
            assert_eq!(format_rfc3339_local_with(bad, zone(0)), Err(ERR_RTC_INVALID_VALUE));
        }
        // Half a second before a DST switch at the Unix epoch: the offset of
        // the whole second before it applies.
        let dst = |secs: i64| Some(if secs < 0 { (0, false) } else { (3600, true) });
        let t = UNIX_EPOCH_TICK - 500_000;
        assert_eq!(format_rfc3339_local_with(t, dst).unwrap(), "1969-12-31T23:59:59.50Z");
        assert_eq!(format_rfc2822_local_with(t, dst).unwrap(), "Wed, 31 Dec 1969 23:59:59 +0000");
    }

    /// The local-time formatters check the pointers and write 32 bytes at
    /// most; a failed format leaves the buffer untouched.
    #[test]
    fn format_callbacks() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5FA5_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (out, utc) = (BASE, BASE + 0x100);
        ctx.write_u64(utc, LEAP_DAY_TICK);
        ctx.write_bytes(out, &[b'x'; 40]);
        ctx.set(Rdi, out);
        ctx.set(Rsi, utc);
        ctx.set(Rdx, 1440);
        assert_eq!(cb_format_rfc2822(&mut ctx), ERR_RTC_INVALID_VALUE);
        assert_eq!(ctx.read_bytes(out, 40).unwrap(), vec![b'x'; 40]);
        ctx.set(Rdx, 90);
        assert_eq!(cb_format_rfc2822(&mut ctx), OK);
        assert_eq!(ctx.read_cstring(out, 64).unwrap(), "Thu, 29 Feb 2024 14:04:56 +0130");
        assert_eq!(ctx.read_bytes(out + 32, 8).unwrap(), vec![b'x'; 8]);
        // A null tick formats the current time (measured on a PS5).
        ctx.set(Rsi, 0);
        assert_eq!(cb_format_rfc2822(&mut ctx), OK);
        assert_eq!(ctx.read_cstring(out, 64).unwrap().len(), 31);
        assert_eq!(cb_format_rfc2822_local_time(&mut ctx), OK);
        assert_eq!(cb_format_rfc3339_local_time(&mut ctx), OK);
        ctx.set(Rdi, 0);
        ctx.set(Rsi, utc);
        assert_eq!(cb_format_rfc2822_local_time(&mut ctx), ERR_RTC_INVALID_POINTER);
        assert_eq!(cb_format_rfc3339_local_time(&mut ctx), ERR_RTC_INVALID_POINTER);
        ctx.set(Rdi, out);
        assert_eq!(cb_format_rfc2822_local_time(&mut ctx), OK);
        assert_eq!(ctx.read_cstring(out, 64).unwrap().len(), 31);
        assert_eq!(cb_format_rfc3339_local_time(&mut ctx), OK);
        assert!(ctx.read_cstring(out, 64).unwrap().starts_with("2024-0"));
    }

    /// The tick sceRtcParseDateTime stores over `previous` (None: error).
    fn parse_dt(text: &str, previous: u64) -> Result<u64, HleResult> {
        parse_date_time(text.as_bytes()).map(|(t, zone)| parsed_tick(previous, &t, zone))
    }

    /// Port of the guest free-form date parsing checks.
    #[test]
    fn parse_date_time_forms() {
        let at = |y, mo, d, h, mi, s| tick(dt(y, mo, d, h, mi, s, 0));
        const UNTOUCHED: u64 = 123;
        let p = |text: &str| parse_dt(text, UNTOUCHED);
        for text in [
            "2024-02-29T12:34:56.789",
            "2024-02-29 12:34:56",
            "2024-02-29 12:34:56Z",
            "2024-02-29",
            "2024/02/29T12:34:56Z",
            "2024-02-29T12:34:56.",
        ] {
            assert_eq!(p(text), Err(ERR_RTC_BAD_PARSE), "{text}");
        }
        assert_eq!(p(" \t 2024-02-29T12:34:56.789Z"), Ok(LEAP_DAY_TICK));
        assert_eq!(p("2024-02-29T12:34:56.789Zjunk"), Ok(LEAP_DAY_TICK));
        assert_eq!(p("2024-02-29T12:34:56.789+99:99"), Ok(LEAP_DAY_TICK - 6039 * 60_000_000));
        assert_eq!(p("2024-02-29T12:34:60-01:00"), Ok(UNTOUCHED + 3_600_000_000));
        assert_eq!(p("2024-13-01T00:00:00Z"), Err(ERR_RTC_INVALID_MONTH));
        assert_eq!(p("2024-02-29T14:04:56.789+01:30"), Ok(LEAP_DAY_TICK));
        assert_eq!(p("2024-02-29t12:34:56.789z"), Ok(LEAP_DAY_TICK));
        assert_eq!(p("1970-01-01T00:00:00Z"), Ok(UNIX_EPOCH_TICK));
        assert_eq!(p("2023-02-29T00:00:00Z"), Err(ERR_RTC_INVALID_DAY));
        let leap = LEAP_DAY_TICK - 789_000;
        assert_eq!(p("Thu, 29 Feb 2024 12:34:56"), Ok(leap));
        assert_eq!(p("Thu, 29 Feb 2024 12:34:56 GMT"), Ok(leap));
        assert_eq!(p("Thu, 29 Feb 2024 14:04:56 +0130"), Ok(leap));
        assert_eq!(p("Thu, 29 Feb 2024 11:04:56 -0130"), Ok(leap));
        assert_eq!(p("Thu, 01 Jan 1970 00:00:00 +0000"), Ok(UNIX_EPOCH_TICK));
        assert_eq!(p("Thu Feb 29 12:34:56 2024"), Ok(leap));
        assert_eq!(p("Thu Jan  1 00:00:00 1970"), Ok(UNIX_EPOCH_TICK));

        let march = at(2024, 3, 1, 12, 34, 56);
        let min = |m: i64| (m * 60_000_000) as u64;
        let cases: &[(&str, Result<u64, HleResult>)] = &[
            ("thursday,29-feb-24 12:34:56", Ok(leap)),
            ("Mon 29 February 2024 12:34:56 -0130", Ok(leap + min(90))),
            (" \tFri, 01 Mar 2024 12:34:56 GMT", Ok(march)),
            ("Thu, 1-Mar 2024 12:34:56 -9999", Ok(march + min(6039))),
            ("Thu, 01 Mar 2024 12:34:56 +01:30", Ok(march)),
            ("Thu, 01 Mar 2024 12:34:56 +0130x", Ok(march - min(90))),
            ("Thu, 01 Mar 2024 12:34:56 PST8PDT", Ok(march + min(480))),
            ("Thu, 01 Mar 2024 12:34:56 nzdt", Ok(march - min(780))),
            ("Thu, 01 Mar 2024 12:34:56 KST", Ok(march - min(540))),
            ("Thu, 01 Mar 2024 12:34:56 HST", Ok(march - min(420))),
            ("Thu, 01 Mar 2024 12:34:56 Jt", Ok(march - min(450))),
            ("Thu, 01 Mar 2024 12:34:56 ut", Ok(march + min(420))),
            ("Thu, 01 Mar 2024 12:34:56 xT", Ok(march)),
            ("Thu, 01 Mar 2024 12:34:56 Ux", Ok(march)),
            ("Thu, 01 Mar 2024 12:34:56 B", Ok(march - min(60))),
            ("Thu, 01 Mar 2024 12:34:56 m", Ok(march - min(720))),
            ("Thu, 01 Mar 2024 12:34:56 N", Ok(march)),
            ("Thu, 01 Mar 2024 12:34:56 y", Ok(march + min(660))),
            ("Thu, 01 Mar 2024 12:34:56 Z", Ok(march)),
            ("Thu, 01 Mar 2024 12:34:56 J", Err(ERR_RTC_BAD_PARSE)),
            ("Thu, 01 Mar 2024 12:34:56  +0100", Err(ERR_RTC_BAD_PARSE)),
            ("Thu, 01 Mar 2024 12:34:56 (UTC)", Err(ERR_RTC_BAD_PARSE)),
            ("Thu, 01 Mar 2024 12:34:56\tEST", Ok(march)),
            ("Thu, 01 Mar 2024 12:34:56 ", Ok(march)),
            ("Thu, 01 Mar 2024 12:34:567 EST", Ok(march)),
            ("Thu, 01 Mar 2024 12:34 EST", Ok(at(2024, 3, 1, 12, 34, 0))),
            ("Thu, 29 Feb 2024 12:34", Err(ERR_RTC_BAD_PARSE)),
            ("Thu, 29 Feb 2024 12:34 ", Ok(at(2024, 2, 29, 12, 34, 0))),
            ("Thu, 01 Mar 2024 12:34EST", Err(ERR_RTC_BAD_PARSE)),
            ("Thu, 01 Mar 2024 1:2:3 EST", Ok(at(2024, 3, 1, 1, 2, 3) + min(300))),
            ("Thu, 01 Mar 2024 26:00:00 GMT", Err(ERR_RTC_BAD_PARSE)),
            ("Thu, 01 Mar 2024 25:00:00 +0100", Ok(UNTOUCHED - min(60))),
            ("Thu, 29 Feb 2023 12:34:56 EST", Ok(UNTOUCHED + min(300))),
            ("Thu, 01 Mar 49 12:34:56 GMT", Ok(at(2049, 3, 1, 12, 34, 56))),
            ("Thu, 01 Mar 50 12:34:56 GMT", Ok(at(1950, 3, 1, 12, 34, 56))),
            ("Mon, 01 Jan 0001 00:00:00 +0100", Ok(0u64.wrapping_sub(min(60)))),
            ("Thu, 01 Mar 024 12:34:56 GMT", Err(ERR_RTC_BAD_PARSE)),
            ("Thu, 01 Mar 10000 12:34:56 GMT", Err(ERR_RTC_BAD_PARSE)),
            ("Thu, 001 Mar 2024 12:34:56 GMT", Err(ERR_RTC_BAD_PARSE)),
            ("Thu, 01  Mar 2024 12:34:56 GMT", Err(ERR_RTC_BAD_PARSE)),
            ("Thu, 01 Mar 2024\t12:34:56 GMT", Err(ERR_RTC_BAD_PARSE)),
            ("Thu, 01 Sept 2024 12:34:56 GMT", Err(ERR_RTC_BAD_PARSE)),
            ("Thurs, 01 Mar 2024 12:34:56 GMT", Err(ERR_RTC_BAD_PARSE)),
            ("Thu , 01 Mar 2024 12:34:56 GMT", Err(ERR_RTC_BAD_PARSE)),
            (", 01 Mar 2024 12:34:56 GMT", Err(ERR_RTC_BAD_PARSE)),
            ("\nThu, 01 Mar 2024 12:34:56 GMT", Err(ERR_RTC_BAD_PARSE)),
            ("THURSDAY february  9 1:2:3 2024 +0100", Ok(at(2024, 2, 9, 1, 2, 3))),
            ("ThuMar  1 12:34:56 10000", Ok(at(1000, 3, 1, 12, 34, 56))),
            ("Thu,\tFeb 29 12:34:56 2024", Ok(leap)),
            ("Thu Feb 29 99:99:99 2024", Ok(UNTOUCHED)),
            ("Thu Feb  29 12:34:56 2024", Err(ERR_RTC_BAD_PARSE)),
            ("Thu Feb 29 12:34 2024", Err(ERR_RTC_BAD_PARSE)),
            ("Thu Feb 29 12:34:56 24", Err(ERR_RTC_BAD_PARSE)),
            ("Thu Feb 29 12:34:56\t2024", Err(ERR_RTC_BAD_PARSE)),
            ("Feb 29 12:34:56 2024", Err(ERR_RTC_BAD_PARSE)),
            ("\n2024-02-29T12:34:56Z", Err(ERR_RTC_BAD_PARSE)),
            ("+2024-02-29T12:34:56Z", Err(ERR_RTC_BAD_PARSE)),
            ("", Err(ERR_RTC_BAD_PARSE)),
        ];
        for (text, expected) in cases {
            assert_eq!(p(text), *expected, "{text:?}");
        }

        for (name, minutes) in [
            ("SWT", 60), ("MEZ", 60), ("WETDST", 60), ("METDST", 120), ("EETDST", 180), ("AESST", 660), ("ACSST", 630),
            ("CAST", 570), ("CADT", 630), ("SAST", 570), ("SADT", 630), ("WST", 480), ("WDT", 540),
        ] {
            let expected = Ok(leap - min(minutes));
            assert_eq!(p(&format!("Thu, 29 Feb 2024 12:34:56 {name}")), expected, "{name}");
            assert_eq!(p(&format!("Thu, 29 Feb 2024 12:34:56 {}", name.to_lowercase())), expected, "{name}");
            assert_eq!(p(&format!("Thu, 29 Feb 2024 12:34:56 {name}extra")), expected, "{name}");
            assert_eq!(p(&format!("Wed, 01 Mar 2000 08:09:10 {name}")), Ok(at(2000, 3, 1, 8, 9, 10) - min(minutes)));
        }
    }

    /// The callback checks both pointers and leaves the tick untouched on a
    /// parse error.
    #[test]
    fn parse_date_time_callback() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5FA6_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let (utc, text) = (BASE, BASE + 0x100);
        ctx.write_bytes(text, b"1970-01-01T00:00:00Z\0");
        ctx.set(Rdi, 0);
        ctx.set(Rsi, text);
        assert_eq!(cb_parse_date_time(&mut ctx), ERR_RTC_INVALID_POINTER);
        ctx.set(Rdi, utc);
        ctx.set(Rsi, 0);
        assert_eq!(cb_parse_date_time(&mut ctx), ERR_RTC_INVALID_POINTER);
        ctx.set(Rsi, text);
        assert_eq!(cb_parse_date_time(&mut ctx), OK);
        assert_eq!(ctx.read_u64(utc), Some(UNIX_EPOCH_TICK));
        ctx.write_u64(utc, 123);
        ctx.write_bytes(text, b"2024-02-29\0");
        assert_eq!(cb_parse_date_time(&mut ctx), ERR_RTC_BAD_PARSE);
        assert_eq!(ctx.read_u64(utc), Some(123));
    }
}
