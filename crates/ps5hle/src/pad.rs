// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libScePad HLE (controller input), plus libSceMouse and libSceBluetoothHid.
//!
//! Up to four local users each have one pad (user id = primary id + slot,
//! see `user_service`), read through ports the title opens: a port is owned
//! by one handle from scePadOpen until scePadClose (standard port handle =
//! slot + 1; a second open of an owned port fails). The primary user's
//! pad is always reported as connected; the other players' pads are
//! connected while the host input layer has a device for them. Input comes
//! from host-side snapshots (`set_host_pad_state_for`, published by the
//! host input layer); without a host input backend a pad is neutral (no
//! buttons, sticks centred, triggers released, resting motion, no touch).
//! Every change becomes a timestamped
//! sample: `scePadRead` drains the unread samples (so quick taps between two
//! polls are not lost), `scePadReadState` returns the latest one. Output
//! requests (rumble, light bar, adaptive triggers) are forwarded to an
//! optional host sink (`set_host_pad_output`).

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

const OK: HleResult = 0;

// ===========================================================================
// libScePad
// ===========================================================================

const PAD_ERR_INVALID_ARG: HleResult = 0x8092_0001u32 as i32 as i64;
const PAD_ERR_INVALID_HANDLE: HleResult = 0x8092_0003u32 as i32 as i64;
const PAD_ERR_ALREADY_OPENED: HleResult = 0x8092_0004u32 as i32 as i64;
const PAD_ERR_NOT_INITIALIZED: HleResult = 0x8092_0005u32 as i32 as i64;
const PAD_ERR_DEVICE_NO_HANDLE: HleResult = 0x8092_0008u32 as i32 as i64;

/// Port types accepted for a user's own pad.
const STANDARD_PORT_TYPE: i32 = 0;
const SPECIAL_PORT_TYPE: i32 = 2;
/// The system user's remote control port.
const REMOTE_CONTROL_PORT_TYPE: i32 = 16;
const SYSTEM_USER_ID: i32 = 0xff;
const PRIMARY_PAD_HANDLE: i32 = 1;
/// Local players / pads.
pub const MAX_PADS: usize = crate::user_service::MAX_LOCAL_USERS;
/// Guest ports: the standard port of every local user (handles 1..=4), their
/// special ports (handles 5..=8) and the system user's remote control port.
const SPECIAL_PORT_BASE: usize = MAX_PADS;
const REMOTE_CONTROL_PORT: usize = 2 * MAX_PADS;
const PORT_COUNT: usize = 2 * MAX_PADS + 1;
const CONTROLLER_INFORMATION_SIZE: usize = 0x1C;
/// ScePadDeviceClassExtendedInformation: class i32, reserved[4], data[12].
const DEVICE_CLASS_EXT_INFO_SIZE: usize = 0x14;
pub const PAD_DATA_SIZE: usize = 0x78;
/// The caller reserves only 8 bytes for the state (its stack cookie follows).
const TRIGGER_EFFECT_STATE_SIZE: usize = 8;
const STANDARD_GRAVITY: f32 = 9.80665;
/// Unread samples kept for `scePadRead` (its maximum batch).
const SAMPLE_QUEUE_LEN: usize = 64;
/// Touch pad resolution reported by GetControllerInformation.
const TOUCH_RES_X: u16 = 1920;
const TOUCH_RES_Y: u16 = 943;
/// Stick dead zone reported to the title: the host dead zone (8000 of
/// 32767) in 0..255 stick units.
const STICK_DEAD_ZONE: u8 = ((255 * (8000 + 32768)) / 65535 - 128) as u8;
/// Gyro samples further apart than this are not integrated (lost reports).
const MAX_GYRO_INTERVAL_US: u64 = 100_000;

/// SCE_PAD_BUTTON bit values.
pub mod button {
    pub const SHARE: u32 = 0x0001;
    pub const L3: u32 = 0x0002;
    pub const R3: u32 = 0x0004;
    pub const OPTIONS: u32 = 0x0008;
    pub const UP: u32 = 0x0010;
    pub const RIGHT: u32 = 0x0020;
    pub const DOWN: u32 = 0x0040;
    pub const LEFT: u32 = 0x0080;
    pub const L2: u32 = 0x0100;
    pub const R2: u32 = 0x0200;
    pub const L1: u32 = 0x0400;
    pub const R1: u32 = 0x0800;
    pub const TRIANGLE: u32 = 0x1000;
    pub const CIRCLE: u32 = 0x2000;
    pub const CROSS: u32 = 0x4000;
    pub const SQUARE: u32 = 0x8000;
    pub const TOUCH_PAD: u32 = 0x10_0000;
}

/// Host motion sample: acceleration in m/s^2, angular velocity in rad/s,
/// both in the controller frame (x right, y up, z towards the player).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HostMotion {
    pub accel: [f32; 3],
    pub angular_velocity: [f32; 3],
}

/// Host touch point (one finger slot), coordinates normalised to 0..1.
/// Guest touch ids are assigned by the pad on each new touch of a slot.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HostTouchPoint {
    pub x: f32,
    pub y: f32,
}

/// Controller snapshot in pad conventions: SCE_PAD_BUTTON bits, sticks 0..255
/// (128 = centre, Y grows downward), triggers 0..255.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HostPadState {
    pub connected: bool,
    pub buttons: u32,
    pub left_x: u8,
    pub left_y: u8,
    pub right_x: u8,
    pub right_y: u8,
    pub l2: u8,
    pub r2: u8,
    /// `None` when the host device has no motion sensors.
    pub motion: Option<HostMotion>,
    /// Finger slots 0 and 1 of the touch pad.
    pub touch: [Option<HostTouchPoint>; 2],
    /// Host device power state (shown by the frontend; the pad API has no
    /// battery field).
    pub battery: Option<HostBattery>,
}

/// Host controller power and link.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostBattery {
    /// Charge in percent, when known.
    pub percent: Option<u8>,
    pub charging: bool,
    pub wireless: bool,
}

impl HostPadState {
    /// Neutral pad: what the guest sees when no host input is present.
    pub const NEUTRAL: HostPadState = HostPadState {
        connected: true,
        buttons: 0,
        left_x: 128,
        left_y: 128,
        right_x: 128,
        right_y: 128,
        l2: 0,
        r2: 0,
        motion: None,
        touch: [None, None],
        battery: None,
    };

    /// Digital L2/R2 bits and analog trigger values agree: a pressed
    /// trigger sets its bit, a set bit without analog data reads as fully
    /// pressed.
    pub fn normalized(mut self) -> Self {
        if self.l2 > 0 {
            self.buttons |= button::L2;
        } else if self.buttons & button::L2 != 0 {
            self.l2 = 255;
        }
        if self.r2 > 0 {
            self.buttons |= button::R2;
        } else if self.buttons & button::R2 != 0 {
            self.r2 = 255;
        }
        self
    }
}

impl Default for HostPadState {
    fn default() -> Self {
        Self::NEUTRAL
    }
}

/// One guest-visible ScePadData sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PadSample {
    pub buttons: u32,
    /// lx, ly, rx, ry.
    pub sticks: [u8; 4],
    pub l2: u8,
    pub r2: u8,
    /// Orientation quaternion x, y, z, w.
    pub orientation: [f32; 4],
    /// Acceleration in G.
    pub accel: [f32; 3],
    /// Angular velocity in rad/s.
    pub angular_velocity: [f32; 3],
    /// Active touches (compacted), each (x, y, id) in touch pad pixels.
    pub touch: [Option<(u16, u16, u8)>; 2],
    pub connected: bool,
    /// Process time (us) at which this state was sampled.
    pub timestamp: u64,
    pub connected_count: u8,
}

/// Resting controller: lying flat, gravity along +y.
const REST_ACCEL: [f32; 3] = [0.0, 1.0, 0.0];
const IDENTITY: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

impl PadSample {
    pub const NEUTRAL: PadSample = PadSample {
        buttons: 0,
        sticks: [128; 4],
        l2: 0,
        r2: 0,
        orientation: IDENTITY,
        accel: REST_ACCEL,
        angular_velocity: [0.0; 3],
        touch: [None, None],
        connected: true,
        timestamp: 0,
        connected_count: 1,
    };
}

pub use crate::pad_controller::{DualSenseEffects, Setting};
use crate::pad_controller::{ControllerOutput, TriggerEffectParam, SETTINGS, TRIGGER_EFFECT_PARAM_SIZE};

/// Host output sink for force feedback / light bar requests.
/// `slot` is the pad slot (handle - 1).
pub trait HostPadOutput: Send {
    /// Motor strengths already scaled by the vibration intensity, lasting
    /// `duration_ms` (0 stops the motors).
    fn set_vibration(&mut self, slot: usize, large: u8, small: u8, duration_ms: u32);
    fn set_lightbar(&mut self, slot: usize, r: u8, g: u8, b: u8);
    fn reset_lightbar(&mut self, slot: usize);
    /// DualSense effect report; only the triggers whose enable bit is set
    /// change.
    fn set_trigger_effect(&mut self, slot: usize, effect: &DualSenseEffects);
}

// --- orientation tracking ---------------------------------------------------

type Vec3 = [f32; 3];
type Quat = [f32; 4]; // x, y, z, w

fn length(v: Vec3) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Rotate `v` by the unit quaternion `q`.
fn rotate(q: Quat, v: Vec3) -> Vec3 {
    let u = [q[0], q[1], q[2]];
    let t = cross(u, v).map(|c| 2.0 * c);
    let ut = cross(u, t);
    [v[0] + q[3] * t[0] + ut[0], v[1] + q[3] * t[1] + ut[1], v[2] + q[3] * t[2] + ut[2]]
}

fn conjugate(q: Quat) -> Quat {
    [-q[0], -q[1], -q[2], q[3]]
}

/// q * r (Hamilton product, x y z w layout).
fn mul(q: Quat, r: Quat) -> Quat {
    [
        q[3] * r[0] + q[0] * r[3] + q[1] * r[2] - q[2] * r[1],
        q[3] * r[1] - q[0] * r[2] + q[1] * r[3] + q[2] * r[0],
        q[3] * r[2] + q[0] * r[1] - q[1] * r[0] + q[2] * r[3],
        q[3] * r[3] - q[0] * r[0] - q[1] * r[1] - q[2] * r[2],
    ]
}

/// Gyro integration with accelerometer (Mahony) tilt correction.
/// Orientation is relative to the pose at connection / last reset; heading
/// drifts freely (it is unobservable without a magnetometer).
#[derive(Clone, Copy, Debug)]
struct OrientationFilter {
    q: Quat,
    last_us: u64,
    /// Gravity direction in the reference frame, captured once the pad is
    /// at rest (|a| close to 1 G).
    up_ref: Option<Vec3>,
}

impl OrientationFilter {
    const fn new() -> Self {
        Self { q: IDENTITY, last_us: 0, up_ref: None }
    }

    /// Restart integration (motion toggled); the orientation is kept.
    fn restart(&mut self) {
        self.last_us = 0;
        self.up_ref = None;
    }

    fn reset(&mut self) {
        self.q = IDENTITY;
        self.restart();
    }

    /// `accel` in G, `gyro` in rad/s.
    fn update(&mut self, accel: Vec3, gyro: Vec3, now_us: u64) {
        let mag = length(accel);
        if self.up_ref.is_none() && mag > 0.9 && mag < 1.1 {
            self.up_ref = Some(rotate(self.q, accel.map(|c| c / mag)));
        }
        let mut rate = gyro;
        if let Some(up) = self.up_ref {
            if mag > 0.8 && mag < 1.2 {
                let measured = accel.map(|c| c / mag);
                let predicted = rotate(conjugate(self.q), up);
                let e = cross(measured, predicted);
                rate = [rate[0] + e[0], rate[1] + e[1], rate[2] + e[2]];
            }
        }
        // A single sample is not extrapolated across lost reports.
        if self.last_us != 0 && now_us > self.last_us && now_us - self.last_us <= MAX_GYRO_INTERVAL_US {
            let dt = (now_us - self.last_us) as f32 * 1e-6;
            let speed = length(rate);
            let half = speed * dt * 0.5;
            let s = if speed > 0.0 { half.sin() / speed } else { 0.0 };
            let q = mul(self.q, [rate[0] * s, rate[1] * s, rate[2] * s, half.cos()]);
            let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            self.q = q.map(|c| c / n);
        }
        self.last_us = now_us;
    }
}

// --- pad state machine -------------------------------------------------------

struct PadCore {
    /// Last published host state (normalised).
    host: HostPadState,
    /// Current guest-visible sample.
    sample: PadSample,
    /// Samples not yet returned by `scePadRead`.
    unread: VecDeque<PadSample>,
    motion_enabled: bool,
    /// scePadSetAngularVelocityDeadbandState: components below the deadband
    /// read as 0.
    angular_deadband: bool,
    orientation: OrientationFilter,
    touch_ids: [Option<u8>; 2],
    next_touch_id: u8,
    /// Times the pad became connected.
    connected_count: u8,
    /// Cached vibration / trigger requests and trigger effect states.
    output: ControllerOutput,
    /// Buttons pressed since the last scePadReadState (a tap shorter than
    /// the title's read interval still shows), when, and the L2/R2 analog
    /// values they were pressed with.
    latch: PressLatch,
    /// When scePadReadState last read this pad, and the time between its
    /// last two reads.
    last_read_us: Option<u64>,
    read_interval_us: u64,
}

/// How long a latched press waits for scePadReadState at least; a title
/// that reads more slowly gets one and a half of its read interval, up to
/// [`LATCH_EXPIRY_MAX_US`], so a tap between two of its reads still shows.
const LATCH_EXPIRY_US: u64 = 100_000;
const LATCH_EXPIRY_MAX_US: u64 = 500_000;

#[derive(Clone, Copy, Default)]
struct PressLatch {
    buttons: u32,
    at_us: u64,
    l2: u8,
    r2: u8,
}

impl PadCore {
    #[cfg(test)]
    const fn new() -> Self {
        Self::with_connected(true)
    }

    const fn with_connected(connected: bool) -> Self {
        let mut host = HostPadState::NEUTRAL;
        host.connected = connected;
        let mut sample = PadSample::NEUTRAL;
        sample.connected = connected;
        sample.connected_count = connected as u8;
        Self {
            host,
            sample,
            unread: VecDeque::new(),
            motion_enabled: true,
            angular_deadband: false,
            orientation: OrientationFilter::new(),
            touch_ids: [None, None],
            next_touch_id: 1,
            connected_count: connected as u8,
            output: ControllerOutput::new(),
            latch: PressLatch { buttons: 0, at_us: 0, l2: 0, r2: 0 },
            last_read_us: None,
            read_interval_us: 0,
        }
    }

    /// How long a latched press waits for the next read of this pad.
    fn latch_expiry_us(&self) -> u64 {
        (self.read_interval_us + self.read_interval_us / 2).clamp(LATCH_EXPIRY_US, LATCH_EXPIRY_MAX_US)
    }

    /// Latch the buttons `state` presses over the last published state; an
    /// older latch that was not read in time is dropped first.
    fn latch_presses(&mut self, state: &HostPadState, now_us: u64) {
        if self.latch.buttons != 0 && now_us.saturating_sub(self.latch.at_us) > self.latch_expiry_us() {
            self.latch = PressLatch::default();
        }
        let pressed = state.buttons & !self.host.buttons;
        if pressed == 0 {
            return;
        }
        self.latch.buttons |= pressed;
        self.latch.at_us = now_us;
        if pressed & button::L2 != 0 {
            self.latch.l2 = state.l2;
        }
        if pressed & button::R2 != 0 {
            self.latch.r2 = state.r2;
        }
    }

    /// The sample scePadReadState reports: the current one with the
    /// presses latched since the last read (if still fresh); the latch is
    /// cleared.
    fn read_state(&mut self, now_us: u64) -> PadSample {
        let mut s = self.sample;
        let latch = std::mem::take(&mut self.latch);
        if latch.buttons != 0 && now_us.saturating_sub(latch.at_us) <= self.latch_expiry_us() {
            s.buttons |= latch.buttons;
            s.l2 = s.l2.max(latch.l2);
            s.r2 = s.r2.max(latch.r2);
        }
        if let Some(last) = self.last_read_us {
            self.read_interval_us = now_us.saturating_sub(last);
        }
        self.last_read_us = Some(now_us);
        s
    }

    /// Rebuild the guest sample from the host state at `now_us`.
    fn resample(&mut self, now_us: u64) {
        let h = self.host;
        let mut s = PadSample {
            buttons: h.buttons,
            sticks: [h.left_x, h.left_y, h.right_x, h.right_y],
            l2: h.l2,
            r2: h.r2,
            connected: h.connected,
            timestamp: now_us,
            connected_count: self.connected_count,
            touch: [None, None],
            ..self.sample
        };
        // Motion events are ignored while disabled; the last values (and
        // the orientation) stay.
        if let (true, Some(m)) = (self.motion_enabled, h.motion) {
            let accel = m.accel.map(|c| c / STANDARD_GRAVITY);
            let rate = if self.angular_deadband { angular_deadband(m.angular_velocity) } else { m.angular_velocity };
            self.orientation.update(accel, rate, now_us);
            s.accel = accel;
            s.angular_velocity = rate;
        }
        s.orientation = self.orientation.q;
        let mut n = 0;
        for (slot, point) in h.touch.iter().enumerate() {
            let Some(p) = point else {
                self.touch_ids[slot] = None;
                continue;
            };
            let id = *self.touch_ids[slot].get_or_insert_with(|| {
                let id = self.next_touch_id;
                self.next_touch_id = if id == 127 { 1 } else { id + 1 };
                id
            });
            let x = (p.x.clamp(0.0, 1.0) * TOUCH_RES_X as f32) as u16;
            let y = (p.y.clamp(0.0, 1.0) * TOUCH_RES_Y as f32) as u16;
            s.touch[n] = Some((x, y, id));
            n += 1;
        }
        self.output.update_trigger_state([h.l2, h.r2]);
        self.sample = s;
        if self.unread.len() == SAMPLE_QUEUE_LEN {
            self.unread.pop_front();
        }
        self.unread.push_back(s);
    }

    fn publish(&mut self, state: HostPadState, now_us: u64) {
        let state = state.normalized();
        if state == self.host {
            return;
        }
        self.latch_presses(&state, now_us);
        if state.connected && !self.host.connected {
            self.connected_count = self.connected_count.saturating_add(1);
        }
        self.host = state;
        self.resample(now_us);
    }

    /// Another host controller became this pad: its input state, motion
    /// tracking, touch ids, cached requests and trigger states start over.
    fn controller_changed(&mut self, now_us: u64) {
        self.sample = PadSample { connected: self.host.connected, connected_count: self.connected_count, ..PadSample::NEUTRAL };
        self.unread.clear();
        self.orientation.reset();
        self.touch_ids = [None, None];
        self.next_touch_id = 1;
        self.output.reset();
        self.resample(now_us);
    }

    fn set_motion_enabled(&mut self, on: bool, now_us: u64) {
        if self.motion_enabled != on {
            self.motion_enabled = on;
            self.orientation.restart();
            self.sample.accel = REST_ACCEL;
            self.sample.angular_velocity = [0.0; 3];
            self.resample(now_us);
        }
    }

    fn reset_orientation(&mut self, now_us: u64) {
        self.orientation.reset();
        self.resample(now_us);
    }

    /// Up to `max` unread samples (oldest first); the current sample when
    /// nothing changed since the last read. A disconnected pad reads one
    /// neutral sample.
    fn read(&mut self, max: usize) -> Vec<PadSample> {
        if !self.host.connected {
            let s = PadSample { connected: false, timestamp: 0, connected_count: self.connected_count, ..PadSample::NEUTRAL };
            return vec![s];
        }
        if self.unread.is_empty() {
            return vec![self.sample];
        }
        let n = max.min(self.unread.len());
        self.unread.drain(..n).collect()
    }
}

static PADS: Mutex<[PadCore; MAX_PADS]> = Mutex::new([
    PadCore::with_connected(true),
    PadCore::with_connected(false),
    PadCore::with_connected(false),
    PadCore::with_connected(false),
]);
static HOST_OUTPUT: Mutex<Option<Box<dyn HostPadOutput>>> = Mutex::new(None);

fn pads() -> MutexGuard<'static, [PadCore; MAX_PADS]> {
    PADS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Input backend hook: publish the primary pad's snapshot.
pub fn set_host_pad_state(state: HostPadState) {
    set_host_pad_state_for(0, state);
}

/// Input backend hook: publish the snapshot of pad `slot`. Unchanged
/// snapshots are ignored (the sample timestamp stays at the last change).
/// The primary pad always stays connected.
pub fn set_host_pad_state_for(slot: usize, mut state: HostPadState) {
    if slot >= MAX_PADS {
        return;
    }
    if slot == 0 {
        state.connected = true;
    }
    let now = crate::time::process_time_usec();
    if slot == 0 {
        record_input(&state, now);
    }
    pads()[slot].publish(state, now);
}

/// Seconds since process start: the time base of input recordings.
pub fn input_clock_secs() -> f64 {
    crate::time::process_time_usec() as f64 / 1e6
}

/// One line of an input recording: `<seconds> <buttons hex> <lx> <ly> <rx>
/// <ry> <l2> <r2>`. Lines starting with `#` are comments.
pub fn format_input_line(t: f64, s: &HostPadState) -> String {
    format!("{t:.3} {:08x} {} {} {} {} {} {}", s.buttons, s.left_x, s.left_y, s.right_x, s.right_y, s.l2, s.r2)
}

/// Parse a line written by [`format_input_line`].
pub fn parse_input_line(line: &str) -> Option<(f64, HostPadState)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let f: Vec<&str> = line.split_whitespace().collect();
    if f.len() != 8 {
        return None;
    }
    let b = |i: usize| f[i].parse::<u8>().ok();
    let st = HostPadState {
        buttons: u32::from_str_radix(f[1], 16).ok()?,
        left_x: b(2)?,
        left_y: b(3)?,
        right_x: b(4)?,
        right_y: b(5)?,
        l2: b(6)?,
        r2: b(7)?,
        ..HostPadState::NEUTRAL
    };
    Some((f[0].parse().ok()?, st))
}

/// `RPS5E_INPUT_RECORD=<file>`: every change of the primary pad (buttons,
/// sticks, triggers) is appended to the file for a later replay.
fn record_input(state: &HostPadState, now_usec: u64) {
    struct Recorder {
        out: std::io::BufWriter<std::fs::File>,
        last: Option<String>,
    }
    static REC: OnceLock<Option<Mutex<Recorder>>> = OnceLock::new();
    let rec = REC.get_or_init(|| {
        let path = std::env::var_os("RPS5E_INPUT_RECORD").filter(|p| !p.is_empty())?;
        match std::fs::File::create(&path) {
            Ok(f) => {
                log::info!("input: recording the pad to {}", path.to_string_lossy());
                let mut out = std::io::BufWriter::new(f);
                let _ = std::io::Write::write_all(&mut out, b"# rps5e input recording: seconds buttons lx ly rx ry l2 r2\n");
                Some(Mutex::new(Recorder { out, last: None }))
            }
            Err(e) => {
                log::warn!("input: cannot create the recording {}: {e}", path.to_string_lossy());
                None
            }
        }
    });
    let Some(rec) = rec else { return };
    let mut r = rec.lock().unwrap_or_else(|e| e.into_inner());
    let line = format_input_line(now_usec as f64 / 1e6, state);
    // Compare without the timestamp: only changes are recorded.
    let body = line.split_once(' ').map(|x| x.1.to_string()).unwrap_or_default();
    if r.last.as_deref() == Some(body.as_str()) {
        return;
    }
    r.last = Some(body);
    let _ = std::io::Write::write_all(&mut r.out, line.as_bytes());
    let _ = std::io::Write::write_all(&mut r.out, b"\n");
    let _ = std::io::Write::flush(&mut r.out);
}

/// Last published snapshot of the primary pad.
pub fn host_pad_state() -> HostPadState {
    pads()[0].host
}

/// Input backend hook: another host controller now drives pad `slot`
/// (hot-plug / reassignment). Its cached vibration and trigger requests,
/// trigger states and motion tracking start over.
pub fn host_controller_changed(slot: usize) {
    if slot < MAX_PADS {
        pads()[slot].controller_changed(crate::time::process_time_usec());
    }
}

/// Output backend hook: install (or remove) the rumble/light bar sink.
pub fn set_host_pad_output(output: Option<Box<dyn HostPadOutput>>) {
    *HOST_OUTPUT.lock().unwrap_or_else(|e| e.into_inner()) = output;
}

fn with_output(f: impl FnOnce(&mut dyn HostPadOutput)) {
    if let Some(out) = HOST_OUTPUT.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        f(out.as_mut());
    }
}

/// AudioOut hook: motor levels of a vibration port block no haptic
/// actuator played; they drive the first pad's rumble motors (merged with
/// its scePadSetVibration request) for a short while.
pub fn submit_audio_haptics(large: u8, small: u8) {
    let mut pads = pads();
    if let Some(v) = pads[0].output.set_audio_haptics(large, small, monotonic_ms(), &SETTINGS) {
        with_output(|o| o.set_vibration(0, v.large, v.small, v.duration_ms));
    }
}

/// Restore the title's vibration of `slot` once audio haptics ran out.
fn expire_audio_haptics(slot: usize, core: &mut PadCore) {
    if let Some(v) = core.output.expire_audio_haptics(monotonic_ms(), &SETTINGS) {
        with_output(|o| o.set_vibration(slot, v.large, v.small, v.duration_ms));
    }
}

/// Global controller levels (0..=100): pad speaker volume (50 = unity) and
/// vibration intensity (100 = full), plus an optional fixed light bar
/// colour that overrides the title's.
pub fn configure_controller(speaker_volume: u32, vibration_intensity: u32, color: Option<[u8; 3]>) {
    SETTINGS.configure(speaker_volume, vibration_intensity);
    SETTINGS.set_color(color);
}

/// Output scale of a controller setting (pad speaker and vibration ports
/// of AudioOut use it as gain).
pub fn setting_scale(setting: Setting) -> f32 {
    SETTINGS.scale(setting)
}

/// Hotkey: advance a controller setting and re-send the cached requests
/// it scales.
pub fn cycle_setting(setting: Setting) {
    let mut pads = pads();
    SETTINGS.cycle(setting);
    let now = monotonic_ms();
    for (slot, core) in pads.iter_mut().enumerate() {
        match setting {
            Setting::VibrationIntensity => {
                let v = core.output.vibration(now, &SETTINGS);
                with_output(|o| o.set_vibration(slot, v.large, v.small, v.duration_ms));
            }
            Setting::TriggerEffectIntensity => {
                if let Some(e) = core.output.trigger_effect(&SETTINGS) {
                    with_output(|o| o.set_trigger_effect(slot, &e));
                }
            }
            Setting::SpeakerVolume => {}
        }
    }
}

static CONTROLS_LOGGED: AtomicBool = AtomicBool::new(false);

/// Library initialisation and port ownership: a port is owned by the one
/// handle that opened it until that handle is closed. Host connection
/// changes never release a guest-owned port.
#[derive(Debug)]
struct PadPorts {
    initialized: bool,
    open: [bool; PORT_COUNT],
}

impl PadPorts {
    const fn new() -> Self {
        Self { initialized: false, open: [false; PORT_COUNT] }
    }

    /// scePadInit: repeated calls keep the open ports.
    fn init(&mut self) -> HleResult {
        self.initialized = true;
        OK
    }

    fn open(&mut self, user_id: i32, ty: i32, index: i32) -> HleResult {
        if !self.initialized {
            return PAD_ERR_NOT_INITIALIZED;
        }
        // The console checks the port type, then the user id, then the
        // index (behaviour measured on hardware): an invalid user id and a
        // non-zero index have their own errors, an unknown type the plain
        // argument one.
        const PAD_ERR_INVALID_USER_ID: HleResult = 0x809B_0001u32 as i32 as i64;
        const PAD_ERR_INVALID_PORT_INDEX: HleResult = 0x809B_0003u32 as i32 as i64;
        let remote = user_id == SYSTEM_USER_ID && ty == REMOTE_CONTROL_PORT_TYPE;
        let known_type = matches!(ty, STANDARD_PORT_TYPE | SPECIAL_PORT_TYPE) || remote;
        if !matches!(ty, 3 | 4) && !known_type {
            return PAD_ERR_INVALID_ARG;
        }
        if user_id == -1 {
            return PAD_ERR_INVALID_USER_ID;
        }
        if matches!(ty, 3 | 4) {
            log::warn!("scePadOpen: unknown pad port type {ty}");
            return PAD_ERR_INVALID_ARG;
        }
        if index != 0 {
            return PAD_ERR_INVALID_PORT_INDEX;
        }
        let Some(port) = open_args_port(user_id, ty, index) else {
            return PAD_ERR_INVALID_ARG;
        };
        if self.open[port] {
            return PAD_ERR_ALREADY_OPENED;
        }
        self.open[port] = true;
        port_handle(port) as HleResult
    }

    fn get_handle(&self, user_id: i32, ty: i32, index: i32) -> HleResult {
        if !self.initialized {
            return PAD_ERR_NOT_INITIALIZED;
        }
        match open_args_port(user_id, ty, index) {
            Some(port) if self.open[port] => port_handle(port) as HleResult,
            _ => PAD_ERR_DEVICE_NO_HANDLE,
        }
    }

    /// Port owned by `handle`.
    fn check(&self, handle: i32) -> Result<usize, HleResult> {
        if !self.initialized {
            return Err(PAD_ERR_NOT_INITIALIZED);
        }
        let port = usize::try_from(handle.wrapping_sub(PRIMARY_PAD_HANDLE)).ok();
        port.filter(|&p| p < PORT_COUNT && self.open[p]).ok_or(PAD_ERR_INVALID_HANDLE)
    }

    fn close(&mut self, handle: i32) -> HleResult {
        match self.check(handle) {
            Ok(port) => {
                self.open[port] = false;
                OK
            }
            Err(e) => e,
        }
    }
}

static PORTS: Mutex<PadPorts> = Mutex::new(PadPorts::new());

fn ports() -> MutexGuard<'static, PadPorts> {
    PORTS.lock().unwrap_or_else(|e| e.into_inner())
}

fn port_handle(port: usize) -> i32 {
    PRIMARY_PAD_HANDLE + port as i32
}

/// Pad (input source) behind a port: the remote control port reads the
/// primary pad.
fn port_pad_slot(port: usize) -> usize {
    if port == REMOTE_CONTROL_PORT { 0 } else { port % MAX_PADS }
}

/// Pad slot of an open handle, or the pad error for the handle.
fn pad_slot(handle: i32) -> Result<usize, HleResult> {
    ports().check(handle).map(port_pad_slot)
}

/// Pad slot of a signed-in user id.
fn user_slot(user_id: i32) -> Option<usize> {
    crate::user_service::user_index(user_id)
}

/// Host monotonic clock in microseconds.
fn monotonic_us() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    (ts.tv_sec as u64) * 1_000_000 + (ts.tv_nsec as u64) / 1_000
}

/// Host monotonic clock in milliseconds (vibration lifetimes).
fn monotonic_ms() -> u64 {
    monotonic_us() / 1000
}

/// Debug aid: `PS5_AUTO_CROSS="40,52,64"` holds Cross for 0.4 s at each
/// listed second offset from the first pad read.
fn auto_cross_active() -> bool {
    static TIMES: OnceLock<Vec<f64>> = OnceLock::new();
    static START_US: AtomicU64 = AtomicU64::new(0);
    let times = TIMES.get_or_init(|| {
        std::env::var("PS5_AUTO_CROSS")
            .ok()
            .map(|raw| raw.split(',').filter_map(|t| t.trim().parse::<f64>().ok()).collect())
            .unwrap_or_default()
    });
    if times.is_empty() {
        return false;
    }
    let now = monotonic_us();
    let _ = START_US.compare_exchange(0, now, Ordering::AcqRel, Ordering::Acquire);
    let elapsed = (now - START_US.load(Ordering::Acquire)) as f64 / 1_000_000.0;
    times.iter().any(|&t| elapsed >= t && elapsed < t + 0.4)
}

/// Serialise a ScePadData (0x78 bytes).
///
/// Layout: 0x00 buttons u32; 0x04 lx, ly, rx, ry; 0x08 l2, r2; 0x0C orientation
/// quaternion (x, y, z, w f32); 0x1C acceleration (G); 0x28 angular velocity;
/// 0x34 touch count u8; 0x3C touch[2] {x u16, y u16, id u8, pad[3]};
/// 0x4C connected u8; 0x50 timestamp (us); 0x58 extension unit data;
/// 0x68 connected count u8; 0x6B device unique data len; 0x6C unique data.
pub fn serialize_pad_data(s: &PadSample) -> [u8; PAD_DATA_SIZE] {
    let mut d = [0u8; PAD_DATA_SIZE];
    d[0x00..0x04].copy_from_slice(&s.buttons.to_le_bytes());
    d[0x04..0x08].copy_from_slice(&s.sticks);
    d[0x08] = s.l2;
    d[0x09] = s.r2;
    let floats = s.orientation.iter().chain(&s.accel).chain(&s.angular_velocity);
    for (i, v) in floats.enumerate() {
        let o = 0x0C + i * 4;
        d[o..o + 4].copy_from_slice(&v.to_le_bytes());
    }
    let mut count = 0usize;
    for &(x, y, id) in s.touch.iter().flatten() {
        let o = 0x3C + count * 8;
        d[o..o + 2].copy_from_slice(&x.to_le_bytes());
        d[o + 2..o + 4].copy_from_slice(&y.to_le_bytes());
        d[o + 4] = id;
        count += 1;
    }
    d[0x34] = count as u8;
    d[0x4C] = s.connected as u8;
    d[0x50..0x58].copy_from_slice(&s.timestamp.to_le_bytes());
    d[0x68] = s.connected_count;
    d
}

/// Serialise ScePadControllerInformation (0x1C bytes).
pub fn serialize_controller_information() -> [u8; CONTROLLER_INFORMATION_SIZE] {
    controller_information(true, 1)
}

/// ScePadControllerInformation for a pad with the given connection state:
/// touch pixel density f32, touch resolution u16 x2, stick dead zones u8 x2,
/// connection type, connected count, connected @0x0C, device class i32 @0x10.
pub fn controller_information(connected: bool, connected_count: u8) -> [u8; CONTROLLER_INFORMATION_SIZE] {
    let mut d = [0u8; CONTROLLER_INFORMATION_SIZE];
    d[0x00..0x04].copy_from_slice(&44.86f32.to_le_bytes()); // touch pad pixel density
    d[0x04..0x06].copy_from_slice(&TOUCH_RES_X.to_le_bytes());
    d[0x06..0x08].copy_from_slice(&TOUCH_RES_Y.to_le_bytes());
    d[0x08] = STICK_DEAD_ZONE; // left
    d[0x09] = STICK_DEAD_ZONE; // right
    d[0x0A] = 0; // connection type (local)
    d[0x0B] = connected_count;
    d[0x0C] = connected as u8;
    d[0x10..0x14].copy_from_slice(&0i32.to_le_bytes()); // device class (standard)
    d
}

fn guest_sample(slot: usize, mut s: PadSample) -> [u8; PAD_DATA_SIZE] {
    if slot == 0 && auto_cross_active() {
        s.buttons |= button::CROSS;
    }
    serialize_pad_data(&s)
}

fn handle_arg(ctx: &CpuContext) -> i32 {
    ctx.rdi() as i32
}

fn cb_pad_init(_ctx: &mut CpuContext) -> HleResult {
    let mut ports = ports();
    for p in pads().iter_mut() {
        p.latch = PressLatch::default();
    }
    if !ports.initialized {
        let now = crate::time::process_time_usec();
        for p in pads().iter_mut() {
            if p.sample.timestamp == 0 {
                p.sample.timestamp = now;
            }
        }
    }
    ports.init()
}

/// Port named by (user, port type, index): a signed-in user's standard or
/// special port, or the system user's remote control port.
fn open_args_port(user_id: i32, ty: i32, index: i32) -> Option<usize> {
    if index != 0 {
        return None;
    }
    if user_id == SYSTEM_USER_ID && ty == REMOTE_CONTROL_PORT_TYPE {
        return Some(REMOTE_CONTROL_PORT);
    }
    let slot = user_slot(user_id)?;
    match ty {
        STANDARD_PORT_TYPE => Some(slot),
        SPECIAL_PORT_TYPE => Some(SPECIAL_PORT_BASE + slot),
        _ => None,
    }
}

fn pad_open_core(ctx: &CpuContext) -> HleResult {
    let (user_id, ty, index) = (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx() as i32);
    let handle = ports().open(user_id, ty, index);
    if handle < PRIMARY_PAD_HANDLE as HleResult {
        return handle;
    }
    let slot = port_pad_slot((handle - PRIMARY_PAD_HANDLE as HleResult) as usize);
    if slot == 0 && !CONTROLS_LOGGED.swap(true, Ordering::AcqRel) {
        log::info!("pad: primary controller opened (input from host pad state)");
    } else if slot > 0 {
        log::info!("pad: controller of local user {} opened", slot + 1);
    }
    handle
}

/// scePadOpen(userId, type, index, param): the param pointer is ignored.
fn cb_pad_open(ctx: &mut CpuContext) -> HleResult {
    pad_open_core(ctx)
}

fn cb_pad_open_ext(ctx: &mut CpuContext) -> HleResult {
    pad_open_core(ctx)
}

/// Handle owning the port named by the open arguments.
fn cb_pad_get_handle(ctx: &mut CpuContext) -> HleResult {
    let (user_id, ty, index) = (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx() as i32);
    ports().get_handle(user_id, ty, index)
}

/// scePadClose(handle): releases the port; a handle that owns no port is
/// rejected.
fn cb_pad_close(ctx: &mut CpuContext) -> HleResult {
    ports().close(handle_arg(ctx))
}

/// Shared shape for calls that only validate the handle.
fn handle_only(ctx: &CpuContext) -> HleResult {
    pad_slot(handle_arg(ctx)).err().unwrap_or(OK)
}

/// Validate the handle and the output pointer in rsi, then write `data`.
fn write_out(ctx: &CpuContext, data: &[u8]) -> HleResult {
    let addr = ctx.rsi();
    if let Err(e) = pad_slot(handle_arg(ctx)) {
        return e;
    }
    if addr == 0 {
        return PAD_ERR_INVALID_ARG;
    }
    if ctx.write_bytes(addr, data) { OK } else { PAD_ERR_INVALID_ARG }
}

fn cb_pad_set_motion_sensor_state(ctx: &mut CpuContext) -> HleResult {
    let slot = match pad_slot(handle_arg(ctx)) {
        Ok(slot) => slot,
        Err(e) => return e,
    };
    pads()[slot].set_motion_enabled(ctx.rsi() as u8 != 0, crate::time::process_time_usec());
    OK
}

/// Angular velocity below one degree per second (rad/s) is noise.
const ANGULAR_VELOCITY_DEADBAND: f32 = 0.017_453_292;

/// Zero the components whose magnitude is below the deadband.
fn angular_deadband(rate: [f32; 3]) -> [f32; 3] {
    rate.map(|v| if v.abs() < ANGULAR_VELOCITY_DEADBAND { 0.0 } else { v })
}

/// scePadSetAngularVelocityDeadbandState(handle, bool enable).
fn cb_pad_set_angular_velocity_deadband_state(ctx: &mut CpuContext) -> HleResult {
    let slot = match pad_slot(handle_arg(ctx)) {
        Ok(slot) => slot,
        Err(e) => return e,
    };
    pads()[slot].angular_deadband = ctx.rsi() as u8 != 0;
    OK
}

/// scePadSetAngularVelocityBiasCorrectionState(handle, bool enable): the
/// angular velocity is passed on as the host reports it, like the tilt
/// correction setting; only the handle is checked.
fn cb_pad_set_angular_velocity_bias_correction_state(ctx: &mut CpuContext) -> HleResult {
    handle_only(ctx)
}

/// Calls accepted for any open handle (tilt correction, vibration mode).
fn cb_pad_handle_only(ctx: &mut CpuContext) -> HleResult {
    handle_only(ctx)
}

/// Vibration modes scePadSetVibrationMode accepts: 1, 2, 3 and 5.
fn vibration_mode_valid(mode: i32) -> bool {
    matches!(mode, 1 | 2 | 3 | 5)
}

/// scePadSetVibrationMode(handle, mode): the mode has no effect on the host
/// rumble, which plays the motor levels as they are. A mode of 0 or less is
/// refused before the handle is looked at.
fn cb_pad_set_vibration_mode(ctx: &mut CpuContext) -> HleResult {
    let mode = ctx.rsi() as i32;
    if !ports().initialized {
        return PAD_ERR_NOT_INITIALIZED;
    }
    if mode <= 0 {
        return PAD_ERR_INVALID_ARG;
    }
    if let Err(e) = pad_slot(handle_arg(ctx)) {
        return e;
    }
    if !vibration_mode_valid(mode) {
        return PAD_ERR_INVALID_ARG;
    }
    OK
}

fn cb_pad_reset_orientation(ctx: &mut CpuContext) -> HleResult {
    let slot = match pad_slot(handle_arg(ctx)) {
        Ok(slot) => slot,
        Err(e) => return e,
    };
    pads()[slot].reset_orientation(crate::time::process_time_usec());
    OK
}

/// scePadIsRemoteController(handle, bool *isRemote): always a local pad.
fn cb_pad_is_remote_controller(ctx: &mut CpuContext) -> HleResult {
    write_out(ctx, &[0u8])
}

fn slot_information(ctx: &CpuContext) -> [u8; CONTROLLER_INFORMATION_SIZE] {
    let slot = pad_slot(handle_arg(ctx)).unwrap_or(0);
    let pads = pads();
    controller_information(pads[slot].host.connected, pads[slot].connected_count)
}

fn cb_pad_get_controller_information(ctx: &mut CpuContext) -> HleResult {
    let info = slot_information(ctx);
    write_out(ctx, &info)
}

/// scePadGetControllerType(handle, int32_t* type): a special or remote
/// control port reports an unknown controller (0) and a standard one a
/// DualSense (2). The library is checked, then the output pointer, then
/// the handle.
fn cb_pad_get_controller_type(ctx: &mut CpuContext) -> HleResult {
    let (handle, out) = (ctx.rdi() as i32, ctx.rsi());
    if !ports().initialized {
        return PAD_ERR_NOT_INITIALIZED;
    }
    if out == 0 {
        return PAD_ERR_INVALID_ARG;
    }
    let port = match ports().check(handle) {
        Ok(port) => port,
        Err(e) => return e,
    };
    let ty: i32 = if port >= SPECIAL_PORT_BASE { 0 } else { 2 };
    if ctx.write_bytes(out, &ty.to_le_bytes()) { OK } else { PAD_ERR_INVALID_ARG }
}

fn cb_pad_get_feature_report(_ctx: &mut CpuContext) -> HleResult {
    OK
}

/// The extended information starts with ScePadControllerInformation; the
/// extension fields stay as the caller initialised them.
fn cb_pad_get_ext_controller_information(ctx: &mut CpuContext) -> HleResult {
    let info = slot_information(ctx);
    write_out(ctx, &info)
}

/// ScePadDeviceClassExtendedInformation: device class 0 (standard pad) with
/// a zeroed class-data union (no special peripheral).
fn cb_pad_device_class_get_extended_information(ctx: &mut CpuContext) -> HleResult {
    write_out(ctx, &[0u8; DEVICE_CLASS_EXT_INFO_SIZE])
}

/// ScePadDeviceClassData: deviceClass i32 @0, dataValid bool @4, then a
/// 16-byte class-data union @8 (`others`: dataLen u8, reserved[3], data[12]).
const DEVICE_CLASS_DATA_SIZE: usize = 24;
/// SCE_PAD_DEVICE_CLASS_INVALID.
const DEVICE_CLASS_INVALID: i32 = -1;

/// Decode the device-class view of a ScePadData: a standard pad, unless the
/// sample carries device-unique bytes from an unknown peripheral.
pub fn parse_device_class_data(pad_data: &[u8]) -> [u8; DEVICE_CLASS_DATA_SIZE] {
    let mut out = [0u8; DEVICE_CLASS_DATA_SIZE];
    out[4] = (pad_data[0x4C] != 0) as u8; // dataValid = connected
    let len = pad_data[0x6B].min(12) as usize;
    if len > 0 {
        out[..4].copy_from_slice(&DEVICE_CLASS_INVALID.to_le_bytes());
        out[8] = len as u8;
        out[12..12 + len].copy_from_slice(&pad_data[0x6C..0x6C + len]);
    }
    out
}

/// scePadDeviceClassParseData(handle, const ScePadData*, ScePadDeviceClassData*)
fn cb_pad_device_class_parse_data(ctx: &mut CpuContext) -> HleResult {
    let (data, out) = (ctx.rsi(), ctx.rdx());
    if let Err(e) = pad_slot(handle_arg(ctx)) {
        return e;
    }
    if data == 0 || out == 0 {
        return PAD_ERR_INVALID_ARG;
    }
    let Some(pad_data) = ctx.read_bytes(data, PAD_DATA_SIZE) else {
        return PAD_ERR_INVALID_ARG;
    };
    if ctx.write_bytes(out, &parse_device_class_data(&pad_data)) { OK } else { PAD_ERR_INVALID_ARG }
}

fn cb_pad_read_state(ctx: &mut CpuContext) -> HleResult {
    let slot = match pad_slot(handle_arg(ctx)) {
        Ok(slot) => slot,
        Err(e) => return e,
    };
    let sample = {
        let mut pads = pads();
        expire_audio_haptics(slot, &mut pads[slot]);
        pads[slot].read_state(crate::time::process_time_usec())
    };
    write_out(ctx, &guest_sample(slot, sample))
}

/// Zeroes all `count` entries, writes the unread samples (oldest first, at
/// least the current one) and returns their count.
fn cb_pad_read(ctx: &mut CpuContext) -> HleResult {
    let addr = ctx.rsi();
    let count = ctx.rdx() as i32;
    let slot = match pad_slot(handle_arg(ctx)) {
        Ok(slot) => slot,
        Err(e) => return e,
    };
    if addr == 0 || !(1..=64).contains(&count) {
        return PAD_ERR_INVALID_ARG;
    }
    let samples = {
        let mut pads = pads();
        expire_audio_haptics(slot, &mut pads[slot]);
        pads[slot].latch = PressLatch::default();
        pads[slot].read(count as usize)
    };
    let mut buf = vec![0u8; count as usize * PAD_DATA_SIZE];
    for (i, s) in samples.iter().enumerate() {
        buf[i * PAD_DATA_SIZE..(i + 1) * PAD_DATA_SIZE].copy_from_slice(&guest_sample(slot, *s));
    }
    if ctx.write_bytes(addr, &buf) { samples.len() as HleResult } else { PAD_ERR_INVALID_ARG }
}

fn cb_pad_ok(_ctx: &mut CpuContext) -> HleResult {
    OK
}

/// Read a parameter struct from rsi after validating the handle; returns
/// the pad slot and the bytes.
fn read_param(ctx: &CpuContext, len: usize) -> Result<(usize, Vec<u8>), HleResult> {
    let addr = ctx.rsi();
    let slot = pad_slot(handle_arg(ctx))?;
    if addr == 0 {
        return Err(PAD_ERR_INVALID_ARG);
    }
    ctx.read_bytes(addr, len).map(|b| (slot, b)).ok_or(PAD_ERR_INVALID_ARG)
}

/// ScePadVibrationParam: { u8 largeMotor; u8 smallMotor; }. The request
/// lasts 65.535 s and is scaled by the vibration intensity.
fn cb_pad_set_vibration(ctx: &mut CpuContext) -> HleResult {
    match read_param(ctx, 2) {
        Ok((slot, p)) => {
            let mut pads = pads();
            let v = pads[slot].output.set_vibration(p[0], p[1], monotonic_ms(), &SETTINGS);
            with_output(|o| o.set_vibration(slot, v.large, v.small, v.duration_ms));
            OK
        }
        Err(e) => e,
    }
}

/// ScePadColor: { u8 r, g, b, reserved; }; a configured controller colour
/// takes precedence.
fn cb_pad_set_light_bar(ctx: &mut CpuContext) -> HleResult {
    match read_param(ctx, 4) {
        Ok((slot, c)) => {
            let [r, g, b] = SETTINGS.color().unwrap_or([c[0], c[1], c[2]]);
            with_output(|o| o.set_lightbar(slot, r, g, b));
            OK
        }
        Err(e) => e,
    }
}

fn cb_pad_reset_light_bar(ctx: &mut CpuContext) -> HleResult {
    let slot = match pad_slot(handle_arg(ctx)) {
        Ok(slot) => slot,
        Err(e) => return e,
    };
    with_output(|o| o.reset_lightbar(slot));
    OK
}

/// scePadSetTriggerEffect(handle, const ScePadTriggerEffectParam*): an
/// invalid mask or command is rejected as a whole.
fn cb_pad_set_trigger_effect(ctx: &mut CpuContext) -> HleResult {
    let (slot, p) = match read_param(ctx, TRIGGER_EFFECT_PARAM_SIZE) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let param = TriggerEffectParam::parse(&p);
    let mut pads = pads();
    let core = &mut pads[slot];
    match core.output.set_trigger_effect(&param, [core.host.l2, core.host.r2], &SETTINGS) {
        Ok(Some(effect)) => {
            with_output(|o| o.set_trigger_effect(slot, &effect));
            OK
        }
        Ok(None) => OK,
        Err(()) => PAD_ERR_INVALID_ARG,
    }
}

/// ScePadTriggerEffectStateInformation: i32 state[2] (L2, R2).
fn cb_pad_get_trigger_effect_state(ctx: &mut CpuContext) -> HleResult {
    let slot = match pad_slot(handle_arg(ctx)) {
        Ok(slot) => slot,
        Err(e) => return e,
    };
    let state = pads()[slot].output.trigger_state();
    let mut d = [0u8; TRIGGER_EFFECT_STATE_SIZE];
    d[..4].copy_from_slice(&state[0].to_le_bytes());
    d[4..].copy_from_slice(&state[1].to_le_bytes());
    write_out(ctx, &d)
}

/// Undocumented pad export: NOT_INITIALIZED before scePadInit; after it,
/// the system module hands a per-process value to the shell and returns 0
/// for a game process (0x809200FF, measured in a process that is not a
/// game, is the shell refusing it). Pad handles are not looked at.
fn cb_pad_n3k_sx62fg_no(_ctx: &mut CpuContext) -> HleResult {
    if !ports().initialized {
        return PAD_ERR_NOT_INITIALIZED;
    }
    OK
}

/// scePadGetAccessibilityControllerState(handle, u32* state): the pad is
/// not an Access controller (0). The output is checked before the handle.
fn cb_pad_get_accessibility_controller_state(ctx: &mut CpuContext) -> HleResult {
    if !ports().initialized {
        return PAD_ERR_NOT_INITIALIZED;
    }
    let out = ctx.rsi();
    if out == 0 {
        return PAD_ERR_INVALID_ARG;
    }
    if let Err(e) = pad_slot(handle_arg(ctx)) {
        return e;
    }
    if ctx.write_u32(out, 0) { OK } else { PAD_ERR_INVALID_ARG }
}

// ====================================================================
const KEYBOARD_ERR_INVALID_ARG: HleResult = 0x80DA_0001u32 as i32 as i64;
const KEYBOARD_ERR_INVALID_HANDLE: HleResult = 0x80DA_0003u32 as i32 as i64;
const KEYBOARD_MAX_DATA_NUM: i32 = 16;
/// SceKeyboardData: timestamp u64, intercepted, reserve[7], connected,
/// length i32, led u32, modifier u32, key codes u16[16], reserve[32].
const KEYBOARD_DATA_SIZE: usize = 0x68;
/// SceKeyboardCharData: processed, length i32, char code u16, reserve[8].
const KEYBOARD_CHAR_DATA_SIZE: usize = 0x14;
const KEYBOARD_MAPPING_101: i32 = 0;
const KEYBOARD_MAPPING_106: i32 = 1;

fn cb_keyboard_open(ctx: &mut CpuContext) -> HleResult {
    let (ty, index) = (ctx.rsi() as i32, ctx.rdx() as i32);
    if ty != 0 || !(0..2).contains(&index) {
        return KEYBOARD_ERR_INVALID_ARG;
    }
    1
}

fn cb_keyboard_close(ctx: &mut CpuContext) -> HleResult {
    if (ctx.rdi() as i32) <= 0 { KEYBOARD_ERR_INVALID_HANDLE } else { OK }
}

fn cb_keyboard_read(ctx: &mut CpuContext) -> HleResult {
    let (handle, addr, num) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as i32);
    if handle <= 0 {
        return KEYBOARD_ERR_INVALID_HANDLE;
    }
    if addr == 0 || num <= 0 || num > KEYBOARD_MAX_DATA_NUM {
        return KEYBOARD_ERR_INVALID_ARG;
    }
    ctx.write_bytes(addr, &vec![0u8; KEYBOARD_DATA_SIZE * num as usize]);
    OK
}

fn cb_keyboard_read_state(ctx: &mut CpuContext) -> HleResult {
    let (handle, addr) = (ctx.rdi() as i32, ctx.rsi());
    if handle <= 0 {
        return KEYBOARD_ERR_INVALID_HANDLE;
    }
    if addr == 0 {
        return KEYBOARD_ERR_INVALID_ARG;
    }
    ctx.write_bytes(addr, &[0u8; KEYBOARD_DATA_SIZE]);
    OK
}

const KEYBOARD_LED_NUM_LOCK: u32 = 1;
const KEYBOARD_LED_CAPS_LOCK: u32 = 2;
const KEYBOARD_MOD_SHIFT: u32 = 0x02 | 0x20;
/// HID usages a character can be asked for: A .. right GUI.
const KEYBOARD_KEYS: std::ops::RangeInclusive<u16> = 0x04..=0xE7;

/// The character a HID usage types on the US (101) or Japanese (106)
/// arrangement, 0 when it types none. The Kana LED changes nothing.
fn keyboard_character(jis: bool, led: u32, modifier: u32, key: u16) -> u16 {
    const US_NORMAL: &[u8; 27] = b"1234567890\n\0\x08\t -=[]\\\\;'`,./";
    const US_SHIFTED: &[u8; 27] = b"!@#$%^&*()\n\0\x08\t _+{}||:\"~<>?";
    const JIS_NORMAL: &[u8; 27] = b"1234567890\n\0\x08\t -^@[]];:\0,./";
    const JIS_SHIFTED: &[u8; 27] = b"!\"#$%&'()\0\n\0\x08\t =~`{}}+*\0<>?";
    const KEYPAD_DIGITS: &[u8; 11] = b"1234567890.";
    let shift = modifier & KEYBOARD_MOD_SHIFT != 0;
    match key {
        0x04..=0x1D => {
            let upper = shift != (led & KEYBOARD_LED_CAPS_LOCK != 0);
            (if upper { b'A' } else { b'a' } + (key - 0x04) as u8) as u16
        }
        0x1E..=0x38 => {
            let table = match (jis, shift) {
                (false, false) => US_NORMAL,
                (false, true) => US_SHIFTED,
                (true, false) => JIS_NORMAL,
                (true, true) => JIS_SHIFTED,
            };
            table[(key - 0x1E) as usize] as u16
        }
        0x64 if !jis => (if shift { b'|' } else { b'\\' }) as u16,
        0x87 if jis => (if shift { b'_' } else { b'\\' }) as u16,
        0x89 if jis => if shift { b'|' as u16 } else { 0x00A5 },
        0x54 => b'/' as u16,
        0x55 => b'*' as u16,
        0x56 => b'-' as u16,
        0x57 => b'+' as u16,
        0x58 => b'\n' as u16,
        0x59..=0x63 if led & KEYBOARD_LED_NUM_LOCK != 0 => KEYPAD_DIGITS[(key - 0x59) as usize] as u16,
        _ => 0,
    }
}

/// sceKeyboardGetKey2Char(handle, arrange, led, modifier, keyCode, charData*):
/// the key is checked before the handle; a key that types nothing keeps its
/// usage as the character code with `processed` clear.
fn cb_keyboard_get_key2char(ctx: &mut CpuContext) -> HleResult {
    let (handle, arrange, led, modifier, key, addr) =
        (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx() as u32, ctx.rcx() as u32, ctx.r8() as u16, ctx.r9());
    if !KEYBOARD_KEYS.contains(&key) {
        return KEYBOARD_ERR_INVALID_ARG;
    }
    if handle <= 0 {
        return KEYBOARD_ERR_INVALID_HANDLE;
    }
    if addr == 0 || (arrange != KEYBOARD_MAPPING_101 && arrange != KEYBOARD_MAPPING_106) {
        return KEYBOARD_ERR_INVALID_ARG;
    }
    let converted = keyboard_character(arrange == KEYBOARD_MAPPING_106, led, modifier, key);
    let processed = converted != 0;
    let mut data = [0u8; KEYBOARD_CHAR_DATA_SIZE];
    data[0] = processed as u8;
    data[4..8].copy_from_slice(&(processed as i32).to_le_bytes());
    data[8..10].copy_from_slice(&(if processed { converted } else { key }).to_le_bytes());
    if ctx.write_bytes(addr, &data) { OK } else { KEYBOARD_ERR_INVALID_ARG }
}

// ===========================================================================
// libSceMouse
// ===========================================================================
//
// The guest mouse is exposed only when the host input layer enables it
// (`set_host_mouse_enabled`). Disabled: init/open/read succeed with 0 (open
// yields handle 0 without marking it open, read reports zero samples), so
// close never finds an opened handle. Enabled: handles 0..=1 open, every
// host mouse event becomes a sample, read drains them.

const MOUSE_ERR_INVALID_ARG: HleResult = 0x80DF_0001u32 as i32 as i64;
const MOUSE_ERR_INVALID_HANDLE: HleResult = 0x80DF_0003u32 as i32 as i64;
const MOUSE_ERR_NOT_INITIALIZED: HleResult = 0x80DF_0005u32 as i32 as i64;
const MOUSE_DATA_SIZE: usize = 0x28;
const MOUSE_QUEUE_LEN: usize = 64;

/// SCE_MOUSE_BUTTON bit values.
pub mod mouse_button {
    pub const PRIMARY: u32 = 0x01;
    pub const SECONDARY: u32 = 0x02;
    pub const OPTIONAL: u32 = 0x04;
    pub const OPTIONAL2: u32 = 0x08;
    pub const OPTIONAL3: u32 = 0x10;
}

/// One SceMouseData sample.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MouseSample {
    pub timestamp: u64,
    pub connected: bool,
    pub buttons: u32,
    pub x: i32,
    pub y: i32,
    pub wheel: i32,
    pub tilt: i32,
}

/// Serialise SceMouseData (0x28 bytes): timestamp u64, connected u8,
/// pad[3], buttons u32 at 0x0C, x/y/wheel/tilt i32 at 0x10.., reserved[8].
pub fn serialize_mouse_data(s: &MouseSample) -> [u8; MOUSE_DATA_SIZE] {
    let mut d = [0u8; MOUSE_DATA_SIZE];
    d[0x00..0x08].copy_from_slice(&s.timestamp.to_le_bytes());
    d[0x08] = s.connected as u8;
    d[0x0C..0x10].copy_from_slice(&s.buttons.to_le_bytes());
    d[0x10..0x14].copy_from_slice(&s.x.to_le_bytes());
    d[0x14..0x18].copy_from_slice(&s.y.to_le_bytes());
    d[0x18..0x1C].copy_from_slice(&s.wheel.to_le_bytes());
    d[0x1C..0x20].copy_from_slice(&s.tilt.to_le_bytes());
    d
}

struct MouseCore {
    enabled: bool,
    initialized: bool,
    open: [bool; 2],
    buttons: u32,
    unread: VecDeque<MouseSample>,
}

static MOUSE: Mutex<MouseCore> =
    Mutex::new(MouseCore { enabled: false, initialized: false, open: [false; 2], buttons: 0, unread: VecDeque::new() });

fn mouse() -> MutexGuard<'static, MouseCore> {
    MOUSE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Expose (or hide) the host mouse to the guest.
pub fn set_host_mouse_enabled(on: bool) {
    mouse().enabled = on;
}

/// Whether the title reads the host mouse (it has a libSceMouse port open).
pub fn mouse_is_open() -> bool {
    let m = mouse();
    m.enabled && m.open.iter().any(|&o| o)
}

/// Host mouse event: relative motion / wheel / tilt and the new button mask.
pub fn push_host_mouse_event(dx: i32, dy: i32, wheel: i32, tilt: i32, buttons: u32) {
    let mut m = mouse();
    if !m.enabled {
        return;
    }
    m.buttons = buttons;
    // Coalesce pure motion into the newest unread sample with the same buttons.
    if let Some(last) = m.unread.back_mut() {
        if last.buttons == buttons && wheel == 0 && tilt == 0 && last.wheel == 0 && last.tilt == 0 {
            last.x = last.x.saturating_add(dx);
            last.y = last.y.saturating_add(dy);
            return;
        }
    }
    if m.unread.len() == MOUSE_QUEUE_LEN {
        m.unread.pop_front();
    }
    let timestamp = crate::time::process_time_usec();
    m.unread.push_back(MouseSample { timestamp, connected: true, buttons, x: dx, y: dy, wheel, tilt });
}

fn cb_mouse_init(_ctx: &mut CpuContext) -> HleResult {
    let mut m = mouse();
    m.initialized = true;
    m.open = [false; 2];
    OK
}

fn cb_mouse_open(ctx: &mut CpuContext) -> HleResult {
    let mut m = mouse();
    if !m.enabled {
        return OK;
    }
    let (user_id, ty, index) = (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx() as i32);
    if !m.initialized {
        return MOUSE_ERR_NOT_INITIALIZED;
    }
    if user_id != crate::user_service::PRIMARY_USER_ID || ty != 0 || !(0..=1).contains(&index) {
        return MOUSE_ERR_INVALID_ARG;
    }
    m.open[index as usize] = true;
    m.unread.clear();
    index as HleResult
}

fn mouse_handle_open(m: &MouseCore, handle: i32) -> bool {
    (0..=1).contains(&handle) && m.open[handle as usize]
}

/// Writes the unread samples (or one idle sample) and returns their count.
fn cb_mouse_read(ctx: &mut CpuContext) -> HleResult {
    let (handle, addr, num) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as i32);
    let mut m = mouse();
    if !m.enabled {
        return OK;
    }
    if addr == 0 || !(1..=64).contains(&num) {
        return MOUSE_ERR_INVALID_ARG;
    }
    if !mouse_handle_open(&m, handle) {
        return MOUSE_ERR_INVALID_HANDLE;
    }
    let samples: Vec<MouseSample> = if m.unread.is_empty() {
        let timestamp = crate::time::process_time_usec();
        vec![MouseSample { timestamp, connected: true, buttons: m.buttons, ..Default::default() }]
    } else {
        let n = (num as usize).min(m.unread.len());
        m.unread.drain(..n).collect()
    };
    drop(m);
    let mut buf = Vec::with_capacity(samples.len() * MOUSE_DATA_SIZE);
    for s in &samples {
        buf.extend_from_slice(&serialize_mouse_data(s));
    }
    if ctx.write_bytes(addr, &buf) { samples.len() as HleResult } else { MOUSE_ERR_INVALID_ARG }
}

fn cb_mouse_close(ctx: &mut CpuContext) -> HleResult {
    let handle = ctx.rdi() as i32;
    let mut m = mouse();
    if !mouse_handle_open(&m, handle) {
        return MOUSE_ERR_INVALID_HANDLE;
    }
    m.open[handle as usize] = false;
    OK
}

// ===========================================================================
// libSceBluetoothHid
// ===========================================================================
//
// No host Bluetooth passthrough: every call succeeds and input comes
// through the pad path. The registered callback is never invoked.

fn cb_bthid_init(_ctx: &mut CpuContext) -> HleResult {
    OK
}

fn cb_bthid_register_device(_ctx: &mut CpuContext) -> HleResult {
    OK
}

fn cb_bthid_register_callback(_ctx: &mut CpuContext) -> HleResult {
    OK
}

fn cb_bthid_unregister_callback(_ctx: &mut CpuContext) -> HleResult {
    OK
}

fn cb_bthid_unregister_device(_ctx: &mut CpuContext) -> HleResult {
    OK
}

// ===========================================================================
// scePadGetInfo
// ===========================================================================

/// ScePadInfo: maxConnectCount u8, connectedCount[2] u8, reserved0[2],
/// padTypeInfo[8] { connectPort, status, deviceType, connectType u8,
/// bluetoothMacAddress[6] }, reserved1[4].
pub const PAD_INFO_SIZE: usize = 0x59;
const PAD_TYPE_INFO_OFFSET: usize = 5;
const PAD_TYPE_INFO_SIZE: usize = 10;
/// SCE_PAD_DEVICE_TYPE_DUAL_SENSE.
const PAD_DEVICE_TYPE_DUAL_SENSE: u8 = 3;
/// SCE_PAD_CONNECT_TYPE_BLUETOOTH.
const PAD_CONNECT_TYPE_BLUETOOTH: u8 = 2;
/// Bluetooth address of the first emulated controller; the last byte is
/// the slot.
const PAD_BD_ADDRESS: [u8; 6] = [0xDC, 0x8C, 0x37, 0x51, 0x01, 0xF4];

/// The pads of the local users that are connected now: one DualSense over
/// Bluetooth per connected slot.
pub fn pad_info(connected: &[bool]) -> [u8; PAD_INFO_SIZE] {
    let mut b = [0u8; PAD_INFO_SIZE];
    b[0] = MAX_PADS as u8;
    let mut n = 0usize;
    for (slot, _) in connected.iter().enumerate().filter(|(_, &c)| c).take(8) {
        let o = PAD_TYPE_INFO_OFFSET + n * PAD_TYPE_INFO_SIZE;
        b[o] = slot as u8;
        b[o + 1] = 1;
        b[o + 2] = PAD_DEVICE_TYPE_DUAL_SENSE;
        b[o + 3] = PAD_CONNECT_TYPE_BLUETOOTH;
        b[o + 4..o + 10].copy_from_slice(&PAD_BD_ADDRESS);
        b[o + 9] = PAD_BD_ADDRESS[5].wrapping_add(slot as u8);
        n += 1;
    }
    b[1] = n as u8;
    b
}

/// scePadGetInfo(ScePadInfo*)
fn cb_pad_get_info(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return PAD_ERR_INVALID_ARG;
    }
    let connected: Vec<bool> = pads().iter().map(|p| p.host.connected).collect();
    if ctx.write_bytes(out, &pad_info(&connected)) { OK } else { PAD_ERR_INVALID_ARG }
}

/// scePadSetProcessPrivilege(int privilege): the title owns its pads; the
/// privilege is accepted.
fn cb_pad_set_process_privilege(ctx: &mut CpuContext) -> HleResult {
    log::debug!("[pad] process privilege {}", ctx.rdi() as i32);
    OK
}

// ===========================================================================

pub fn register(reg: &mut HleRegistry) {
    const PAD: &str = "libScePad";
    reg.register_fn("hv1luiJrqQM", "scePadInit", PAD, cb_pad_init);
    reg.register_fn("xk0AcarP3V4", "scePadOpen", PAD, cb_pad_open);
    reg.register_fn("WFIiSfXGUq8", "scePadOpenExt", PAD, cb_pad_open_ext);
    reg.register_fn("u1GRHp+oWoY", "scePadGetHandle", PAD, cb_pad_get_handle);
    reg.register_fn("1Q0VRAW2fp0", "scePadGetControllerType", PAD, cb_pad_get_controller_type);
    reg.register_fn("6ncge5+l5Qs", "scePadClose", PAD, cb_pad_close);
    reg.register_fn("clVvL4ZDntw", "scePadSetMotionSensorState", PAD, cb_pad_set_motion_sensor_state);
    reg.register_fn("vDLMoJLde8I", "scePadSetTiltCorrectionState", PAD, cb_pad_handle_only);
    reg.register_fn(
        "r44mAxdSG+U",
        "scePadSetAngularVelocityDeadbandState",
        PAD,
        cb_pad_set_angular_velocity_deadband_state,
    );
    reg.register_fn(
        "KLmYx9ij2h0",
        "scePadSetAngularVelocityBiasCorrectionState",
        PAD,
        cb_pad_set_angular_velocity_bias_correction_state,
    );
    reg.register_fn("gjP9-KQzoUk", "scePadGetControllerInformation", PAD, cb_pad_get_controller_information);
    reg.register_fn("PZSoY8j0Pko", "scePadGetFeatureReport", PAD, cb_pad_get_feature_report);
    reg.register_fn(
        "hGbf2QTBmqc",
        "scePadGetExtControllerInformation",
        PAD,
        cb_pad_get_ext_controller_information,
    );
    reg.register_fn(
        "AcslpN1jHR8",
        "scePadDeviceClassGetExtendedInformation",
        PAD,
        cb_pad_device_class_get_extended_information,
    );
    reg.register_fn("IHPqcbc0zCA", "scePadDeviceClassParseData", PAD, cb_pad_device_class_parse_data);
    reg.register_fn("YndgXqQVV7c", "scePadReadState", PAD, cb_pad_read_state);
    reg.register_fn("q1cHNfGycLI", "scePadRead", PAD, cb_pad_read);
    reg.register_fn("W2G-yoyMF5U", "scePadSetVibrationMode", PAD, cb_pad_set_vibration_mode);
    reg.register_fn("2JgFB2n9oUM", "scePadSetTriggerEffect", PAD, cb_pad_set_trigger_effect);
    reg.register_fn("znaWI0gpuo8", "scePadGetTriggerEffectState", PAD, cb_pad_get_trigger_effect_state);
    reg.register_fn("yFVnOdGxvZY", "scePadSetVibration", PAD, cb_pad_set_vibration);
    reg.register_fn("RR4novUEENY", "scePadSetLightBar", PAD, cb_pad_set_light_bar);
    reg.register_fn("DscD1i9HX1w", "scePadResetLightBar", PAD, cb_pad_reset_light_bar);
    reg.register_fn("rIZnR6eSpvk", "scePadResetOrientation", PAD, cb_pad_reset_orientation);
    reg.register_fn("fCWdlnmB1Ks", "scePadIsRemoteController", PAD, cb_pad_is_remote_controller);
    reg.register_fn("1Odcw19nADw", "scePadGetInfo", PAD, cb_pad_get_info);
    reg.register_fn("DmBx8K+jDWw", "scePadSetProcessPrivilege", PAD, cb_pad_set_process_privilege);
    reg.register_fn(
        "Yq0zOH7YNOM",
        "scePadSetVibrationTriggerEffectWeakWhileEmbeddedMicInUse",
        PAD,
        cb_pad_ok,
    );
    // Exported without a known name.
    let unnamed_pad_nid = "n3kSX62fgNo";
    reg.register_fn(unnamed_pad_nid, "scePad_n3kSX62fgNo", PAD, cb_pad_n3k_sx62fg_no);
    reg.register_fn("zBsgF0q8DIM", "scePadGetAccessibilityControllerState", PAD, cb_pad_get_accessibility_controller_state);

    const KEYBOARD: &str = "libSceKeyboard";
    reg.register_fn("wadT3QBCGY0", "sceKeyboardInit", KEYBOARD, cb_pad_ok);
    reg.register_fn("HJ+KnEHcaxI", "sceKeyboardOpen", KEYBOARD, cb_keyboard_open);
    reg.register_fn("0LWei+c7RNc", "sceKeyboardClose", KEYBOARD, cb_keyboard_close);
    reg.register_fn("6HpE68bzX6M", "sceKeyboardReadState", KEYBOARD, cb_keyboard_read_state);
    reg.register_fn("xybbGMCr738", "sceKeyboardRead", KEYBOARD, cb_keyboard_read);
    reg.register_fn("yO9JwdRhtSA", "sceKeyboardGetKey2Char", KEYBOARD, cb_keyboard_get_key2char);

    const MOUSE: &str = "libSceMouse";
    reg.register_fn("Qs0wWulgl7U", "sceMouseInit", MOUSE, cb_mouse_init);
    reg.register_fn("RaqxZIf6DvE", "sceMouseOpen", MOUSE, cb_mouse_open);
    reg.register_fn("x8qnXqh-tiM", "sceMouseRead", MOUSE, cb_mouse_read);
    reg.register_fn("cAnT0Rw-IwU", "sceMouseClose", MOUSE, cb_mouse_close);

    const BTHID: &str = "libSceBluetoothHid";
    reg.register_fn("tul3-GzejQc", "sceBluetoothHidInit", BTHID, cb_bthid_init);
    reg.register_fn("4FUZ+c52d2k", "sceBluetoothHidRegisterDevice", BTHID, cb_bthid_register_device);
    reg.register_fn("4Ypfo9RIwfM", "sceBluetoothHidRegisterCallback", BTHID, cb_bthid_register_callback);
    reg.register_fn("upEB3XUvH8o", "sceBluetoothHidUnregisterCallback", BTHID, cb_bthid_unregister_callback);
    reg.register_fn("u+lIalCYOho", "sceBluetoothHidUnregisterDevice", BTHID, cb_bthid_unregister_device);
}

#[cfg(test)]
mod tests {
    /// A press that ends between two scePadReadState calls is reported by
    /// the next one (with its L2/R2 analog value), once; a latch not read
    /// within 100 ms, or one older than a newer press, is dropped.
    #[test]
    fn short_presses_are_latched_for_read_state() {
        use super::{button, HostPadState, PadCore};
        let press = |buttons: u32, l2: u8| HostPadState { buttons, l2, ..HostPadState::NEUTRAL };
        let mut core = PadCore::new();
        core.publish(press(button::CROSS | button::L2, 180), 1000);
        core.publish(press(0, 0), 2000);
        let s = core.read_state(3000);
        assert_eq!(s.buttons & (button::CROSS | button::L2), button::CROSS | button::L2);
        assert_eq!(s.l2, 180);
        assert_eq!(core.read_state(4000).buttons, 0, "reported once");
        // Not read in time.
        core.publish(press(button::CIRCLE, 0), 10_000);
        core.publish(press(0, 0), 11_000);
        assert_eq!(core.read_state(10_000 + 100_001).buttons, 0);
        // An expired latch does not survive a newer press.
        core.publish(press(button::CIRCLE, 0), 200_000);
        core.publish(press(0, 0), 200_100);
        core.publish(press(button::CROSS, 0), 400_000);
        core.publish(press(0, 0), 400_100);
        assert_eq!(core.read_state(400_200).buttons, button::CROSS);
    }

    /// A title that reads the pad every 170 ms still sees a tap made 150 ms
    /// before its next read (the latch lasts 1.5 read intervals), but not
    /// one older than that; the latch never lasts more than 500 ms.
    #[test]
    fn latch_follows_a_slow_read_interval() {
        use super::{button, HostPadState, PadCore};
        let press = |buttons: u32| HostPadState { buttons, ..HostPadState::NEUTRAL };
        let mut core = PadCore::new();
        core.read_state(1_000);
        core.read_state(171_000);
        core.publish(press(button::CROSS), 172_000);
        core.publish(press(0), 173_000);
        assert_eq!(core.read_state(323_000).buttons & button::CROSS, button::CROSS);
        core.publish(press(button::CROSS), 324_000);
        core.publish(press(0), 325_000);
        assert_eq!(core.read_state(724_000).buttons & button::CROSS, 0, "older than 1.5 intervals");
        // A read every 2 s still drops a latch after 500 ms.
        core.read_state(2_724_000);
        core.publish(press(button::CROSS), 2_725_000);
        core.publish(press(0), 2_726_000);
        assert_eq!(core.read_state(3_226_000).buttons & button::CROSS, 0);
    }

    #[test]
    fn input_recording_lines_round_trip() {
        let st = HostPadState { buttons: 0x4008, left_x: 0, left_y: 255, right_x: 17, right_y: 128, l2: 200, r2: 3, ..HostPadState::NEUTRAL };
        let line = super::format_input_line(12.345, &st);
        let (t, back) = super::parse_input_line(&line).unwrap();
        assert!((t - 12.345).abs() < 1e-9);
        assert_eq!((back.buttons, back.left_x, back.left_y, back.right_x, back.right_y, back.l2, back.r2), (0x4008, 0, 255, 17, 128, 200, 3));
        assert!(super::parse_input_line("# comment").is_none());
        assert!(super::parse_input_line("1.0 zz 0 0 0 0 0 0").is_none());
    }

    use super::*;

    /// The title reads the mouse while it has a libSceMouse port open (the
    /// host input then keeps motion and the wheel off the pad bindings).
    #[test]
    fn mouse_is_open_follows_the_ports() {
        use crate::context::CpuRegister::*;
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        set_host_mouse_enabled(true);
        assert_eq!(cb_mouse_init(&mut ctx), OK);
        assert!(!mouse_is_open());
        ctx.set(Rdi, crate::user_service::PRIMARY_USER_ID as u64);
        ctx.set(Rsi, 0);
        ctx.set(Rdx, 0);
        let handle = cb_mouse_open(&mut ctx);
        assert!(handle >= 0);
        assert!(mouse_is_open());
        ctx.set(Rdi, handle as u64);
        assert_eq!(cb_mouse_close(&mut ctx), OK);
        assert!(!mouse_is_open());
        set_host_mouse_enabled(false);
    }

    #[test]
    fn info_and_bthid_nids_match_names() {
        for (nid, name) in [
            ("1Odcw19nADw", "scePadGetInfo"),
            ("DmBx8K+jDWw", "scePadSetProcessPrivilege"),
            ("upEB3XUvH8o", "sceBluetoothHidUnregisterCallback"),
            ("u+lIalCYOho", "sceBluetoothHidUnregisterDevice"),
        ] {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// One DualSense over Bluetooth per connected slot, in slot order.
    #[test]
    fn pad_info_lists_connected_pads() {
        let b = pad_info(&[true, false, true, false]);
        assert_eq!((b[0], b[1], b[2]), (4, 2, 0));
        assert_eq!(&b[5..15], &[0, 1, 3, 2, 0xDC, 0x8C, 0x37, 0x51, 0x01, 0xF4]);
        assert_eq!(&b[15..25], &[2, 1, 3, 2, 0xDC, 0x8C, 0x37, 0x51, 0x01, 0xF6]);
        assert!(b[25..].iter().all(|&x| x == 0));
        assert_eq!(pad_info(&[false; 4])[1], 0);
    }

    #[test]
    fn get_info_and_privilege() {
        use crate::context::CpuRegister::*;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5FA4_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdi, 0);
        assert_eq!(cb_pad_get_info(&mut ctx), PAD_ERR_INVALID_ARG);
        ctx.write_bytes(BASE, &[0xAA; 0x60]);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_pad_get_info(&mut ctx), OK);
        let b = ctx.read_bytes(BASE, 0x60).unwrap();
        assert_eq!(b[0], MAX_PADS as u8);
        assert!(b[PAD_INFO_SIZE..].iter().all(|&x| x == 0xAA), "0x59-byte block");
        ctx.set(Rdi, 1);
        assert_eq!(cb_pad_set_process_privilege(&mut ctx), OK);
        assert_eq!(cb_bthid_unregister_device(&mut ctx), OK);
        assert_eq!(cb_bthid_unregister_callback(&mut ctx), OK);
    }

    #[test]
    fn device_class_parse() {
        assert_eq!(ps5nid::compute_nid("scePadDeviceClassParseData"), "IHPqcbc0zCA");
        let mut d = [0u8; PAD_DATA_SIZE];
        d[0x4C] = 1;
        let c = parse_device_class_data(&d);
        assert_eq!(&c[..4], &0i32.to_le_bytes());
        assert_eq!(c[4], 1);
        assert!(c[5..].iter().all(|&b| b == 0));
        d[0x6B] = 3;
        d[0x6C..0x6F].copy_from_slice(&[7, 8, 9]);
        let c = parse_device_class_data(&d);
        assert_eq!(&c[..4], &(-1i32).to_le_bytes());
        assert_eq!(c[8], 3);
        assert_eq!(&c[12..15], &[7, 8, 9]);
    }

    /// Modes 1, 2, 3 and 5 are accepted (the calls on a handle are checked
    /// in `trigger_effect_state_argument_checks`, which owns the ports).
    #[test]
    fn vibration_modes() {
        assert!([1, 2, 3, 5].into_iter().all(vibration_mode_valid));
        assert!([-1, 0, 4, 6].into_iter().all(|m| !vibration_mode_valid(m)));
    }

    /// With the deadband on, components below 1 degree/s read as 0, in the
    /// sample and in what the orientation integrates.
    #[test]
    fn angular_velocity_deadband() {
        assert_eq!(angular_deadband([0.01, 0.0175, -0.017]), [0.0, 0.0175, 0.0]);
        let mut core = PadCore::new();
        let motion = |rate| HostPadState {
            motion: Some(HostMotion { accel: [0.0, STANDARD_GRAVITY, 0.0], angular_velocity: rate }),
            ..HostPadState::NEUTRAL
        };
        core.publish(motion([0.01, 0.0175, -0.017]), 1000);
        assert_eq!(core.latch.buttons, 0, "motion presses nothing");
        assert_eq!(core.sample.angular_velocity, [0.01, 0.0175, -0.017], "off by default");
        core.angular_deadband = true;
        core.publish(motion([0.011, 0.0175, -0.017]), 2000);
        assert_eq!(core.sample.angular_velocity, [0.0, 0.0175, 0.0]);
    }

    fn f32_at(d: &[u8], o: usize) -> f32 {
        f32::from_le_bytes(d[o..o + 4].try_into().unwrap())
    }

    #[test]
    fn neutral_pad_data_layout() {
        let s = PadSample { timestamp: 0x1122_3344_5566_7788, ..PadSample::NEUTRAL };
        let d = serialize_pad_data(&s);
        assert_eq!(d.len(), 0x78);
        assert_eq!(&d[0..4], &[0, 0, 0, 0]);
        assert_eq!(&d[4..8], &[128, 128, 128, 128]);
        assert_eq!(&d[8..10], &[0, 0]);
        assert_eq!(f32_at(&d, 0x0C), 0.0);
        assert_eq!(f32_at(&d, 0x18), 1.0);
        // Resting pad: 1 G along +y, no rotation.
        assert_eq!(f32_at(&d, 0x1C), 0.0);
        assert_eq!(f32_at(&d, 0x20), 1.0);
        assert_eq!(f32_at(&d, 0x24), 0.0);
        assert!(d[0x28..0x34].iter().all(|&b| b == 0));
        assert_eq!(d[0x34], 0);
        assert_eq!(d[0x4C], 1);
        assert_eq!(u64::from_le_bytes(d[0x50..0x58].try_into().unwrap()), 0x1122_3344_5566_7788);
        assert_eq!(d[0x68], 1);
        assert!(d[0x69..].iter().all(|&b| b == 0));
    }

    fn synthetic() -> HostPadState {
        HostPadState {
            buttons: button::CROSS | button::TOUCH_PAD,
            left_x: 0,
            left_y: 255,
            right_x: 10,
            right_y: 20,
            l2: 255,
            r2: 7,
            motion: Some(HostMotion {
                accel: [STANDARD_GRAVITY, 0.0, -STANDARD_GRAVITY],
                angular_velocity: [1.5, 2.5, 3.5],
            }),
            touch: [None, Some(HostTouchPoint { x: 1.0, y: 0.5 })],
            ..HostPadState::NEUTRAL
        }
    }

    #[test]
    fn pad_data_from_synthetic_input() {
        let mut core = PadCore::new();
        core.publish(synthetic(), 1000);
        let d = serialize_pad_data(&core.sample);
        // Analog triggers imply their digital bits.
        let bits = button::CROSS | button::TOUCH_PAD | button::L2 | button::R2;
        assert_eq!(u32::from_le_bytes(d[0..4].try_into().unwrap()), bits);
        assert_eq!(&d[4..10], &[0, 255, 10, 20, 255, 7]);
        assert_eq!(f32_at(&d, 0x1C), 1.0);
        assert_eq!(f32_at(&d, 0x24), -1.0);
        assert_eq!(f32_at(&d, 0x28), 1.5);
        assert_eq!(f32_at(&d, 0x30), 3.5);
        // Only the active point, compacted into slot 0, with the first id.
        assert_eq!(d[0x34], 1);
        assert_eq!(u16::from_le_bytes([d[0x3C], d[0x3D]]), 1920);
        assert_eq!(u16::from_le_bytes([d[0x3E], d[0x3F]]), 471);
        assert_eq!(d[0x40], 1);
        assert!(d[0x44..0x4C].iter().all(|&b| b == 0));
        assert_eq!(u64::from_le_bytes(d[0x50..0x58].try_into().unwrap()), 1000);

        // Motion disabled: resting motion, identity orientation.
        core.set_motion_enabled(false, 2000);
        let d = serialize_pad_data(&core.sample);
        assert_eq!(f32_at(&d, 0x20), 1.0);
        assert!(d[0x28..0x34].iter().all(|&b| b == 0));
        assert_eq!(f32_at(&d, 0x18), 1.0);
    }

    #[test]
    fn digital_trigger_reads_fully_pressed() {
        let s = HostPadState { buttons: button::R2, ..HostPadState::NEUTRAL }.normalized();
        assert_eq!((s.l2, s.r2), (0, 255));
        assert_eq!(s.buttons, button::R2);
    }

    #[test]
    fn timestamp_tracks_changes_and_read_drains_queue() {
        let mut core = PadCore::new();
        let pressed = HostPadState { buttons: button::CROSS, ..HostPadState::NEUTRAL };
        core.publish(pressed, 10);
        core.publish(pressed, 20); // unchanged: no new sample
        core.publish(HostPadState::NEUTRAL, 30);
        let s = core.read(64);
        assert_eq!(s.len(), 2);
        assert_eq!((s[0].buttons, s[0].timestamp), (button::CROSS, 10));
        assert_eq!((s[1].buttons, s[1].timestamp), (0, 30));
        // Nothing new: the current sample, same timestamp.
        let s = core.read(64);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].timestamp, 30);
        // The batch size is honoured, oldest first.
        for i in 0..5u8 {
            core.publish(HostPadState { left_x: i, ..HostPadState::NEUTRAL }, 100 + i as u64);
        }
        assert_eq!(core.read(2).iter().map(|s| s.sticks[0]).collect::<Vec<_>>(), [0, 1]);
        assert_eq!(core.read(64).len(), 3);
    }

    #[test]
    fn touch_ids_advance_per_new_touch() {
        let mut core = PadCore::new();
        let t = |a: bool, b: bool| HostPadState {
            touch: [a.then_some(HostTouchPoint { x: 0.0, y: 0.0 }), b.then_some(HostTouchPoint { x: 1.0, y: 1.0 })],
            ..HostPadState::NEUTRAL
        };
        core.publish(t(true, false), 1);
        assert_eq!(core.sample.touch[0].unwrap().2, 1);
        core.publish(t(true, true), 2);
        assert_eq!(core.sample.touch.map(|p| p.unwrap().2), [1, 2]);
        assert_eq!(core.sample.touch[1].unwrap().0, 1920);
        assert_eq!(core.sample.touch[1].unwrap().1, 943);
        core.publish(t(false, true), 3);
        assert_eq!(core.sample.touch[0].unwrap().2, 2);
        assert_eq!(core.sample.touch[1], None);
        core.publish(t(true, true), 4);
        assert_eq!(core.sample.touch.map(|p| p.unwrap().2), [3, 2]);
    }

    #[test]
    fn orientation_integrates_gyro() {
        let mut core = PadCore::new();
        // 90 deg/s about y for 0.5 s in 5 ms steps -> 45 degrees.
        let rate = std::f32::consts::FRAC_PI_2;
        for i in 0..=100u64 {
            let accel = [0.0, STANDARD_GRAVITY, 0.0];
            let m = HostMotion { accel, angular_velocity: [0.0, rate, i as f32 * 1e-7] };
            core.publish(HostPadState { motion: Some(m), ..HostPadState::NEUTRAL }, 1_000 + i * 5_000);
        }
        let q = core.sample.orientation;
        let angle = 2.0 * q[1].atan2(q[3]);
        assert!((angle - std::f32::consts::FRAC_PI_4).abs() < 0.01, "angle {angle}");
        core.reset_orientation(600_000);
        assert_eq!(core.sample.orientation, IDENTITY);
    }

    #[test]
    fn controller_information_layout() {
        let d = serialize_controller_information();
        assert_eq!(f32_at(&d, 0), 44.86);
        assert_eq!(u16::from_le_bytes([d[4], d[5]]), 1920);
        assert_eq!(u16::from_le_bytes([d[6], d[7]]), 943);
        assert_eq!(&d[8..0x0D], &[30, 30, 0, 1, 1]);
        assert!(d[0x0D..].iter().all(|&b| b == 0));
    }

    #[test]
    fn mouse_data_layout() {
        let s = MouseSample {
            timestamp: 7,
            connected: true,
            buttons: mouse_button::PRIMARY | mouse_button::OPTIONAL,
            x: -3,
            y: 4,
            wheel: 1,
            tilt: -1,
        };
        let d = serialize_mouse_data(&s);
        assert_eq!(d.len(), 0x28);
        assert_eq!(u64::from_le_bytes(d[0..8].try_into().unwrap()), 7);
        assert_eq!(d[8], 1);
        assert_eq!(u32::from_le_bytes(d[0x0C..0x10].try_into().unwrap()), 0x05);
        assert_eq!(i32::from_le_bytes(d[0x10..0x14].try_into().unwrap()), -3);
        assert_eq!(i32::from_le_bytes(d[0x14..0x18].try_into().unwrap()), 4);
        assert_eq!(i32::from_le_bytes(d[0x18..0x1C].try_into().unwrap()), 1);
        assert_eq!(i32::from_le_bytes(d[0x1C..0x20].try_into().unwrap()), -1);
        assert!(d[0x20..].iter().all(|&b| b == 0));
    }

    #[test]
    fn registered_nids_match_names() {
        let mut reg = HleRegistry::new();
        register(&mut reg);
        let src = include_str!("pad.rs");
        let mut n = 0;
        for line in src.lines().map(str::trim) {
            let Some(rest) = line.strip_prefix("reg.register_fn(\"") else { continue };
            let mut parts = rest.split('"');
            let (nid, name) = (parts.next().unwrap(), parts.nth(1).unwrap());
            assert_eq!(crate::sync::test_util::nid(name), nid, "NID mismatch for {name}");
            n += 1;
        }
        // Multi-line registrations.
        for (nid, name) in [
            ("r44mAxdSG+U", "scePadSetAngularVelocityDeadbandState"),
            ("KLmYx9ij2h0", "scePadSetAngularVelocityBiasCorrectionState"),
            ("hGbf2QTBmqc", "scePadGetExtControllerInformation"),
            ("AcslpN1jHR8", "scePadDeviceClassGetExtendedInformation"),
            ("Yq0zOH7YNOM", "scePadSetVibrationTriggerEffectWeakWhileEmbeddedMicInUse"),
        ] {
            assert_eq!(crate::sync::test_util::nid(name), nid, "NID mismatch for {name}");
        }
        assert!(n > 25, "only {n} single-line registrations");
    }

    #[test]
    fn open_arguments() {
        let primary = crate::user_service::PRIMARY_USER_ID;
        assert_eq!(open_args_port(primary, STANDARD_PORT_TYPE, 0), Some(0));
        assert_eq!(open_args_port(primary, SPECIAL_PORT_TYPE, 0), Some(SPECIAL_PORT_BASE));
        assert_eq!(open_args_port(primary, 1, 0), None);
        assert_eq!(open_args_port(primary, STANDARD_PORT_TYPE, 1), None);
        assert_eq!(open_args_port(SYSTEM_USER_ID, REMOTE_CONTROL_PORT_TYPE, 0), Some(REMOTE_CONTROL_PORT));
        assert_eq!(open_args_port(SYSTEM_USER_ID, STANDARD_PORT_TYPE, 0), None);
        assert_eq!(open_args_port(-1, STANDARD_PORT_TYPE, 0), None);
        // Every port of a user reads that user's pad.
        assert_eq!(port_pad_slot(0), 0);
        assert_eq!(port_pad_slot(SPECIAL_PORT_BASE + 1), 1);
        assert_eq!(port_pad_slot(REMOTE_CONTROL_PORT), 0);
        assert_eq!(STICK_DEAD_ZONE, 30);
    }

    #[test]
    fn pad_port_ownership() {
        let user = crate::user_service::PRIMARY_USER_ID;
        let mut ports = PadPorts::new();
        assert_eq!(ports.open(user, 0, 0), PAD_ERR_NOT_INITIALIZED, "an uninitialized library opened a port");
        assert_eq!(ports.get_handle(user, 0, 0), PAD_ERR_NOT_INITIALIZED);
        assert_eq!(ports.check(1), Err(PAD_ERR_NOT_INITIALIZED));
        assert_eq!(ports.init(), OK);
        assert_eq!(ports.get_handle(user, 0, 0), PAD_ERR_DEVICE_NO_HANDLE, "initializing opened a port");
        let handle = ports.open(user, 0, 0);
        assert_eq!(handle, 1);
        assert_eq!(ports.init(), OK);
        assert_eq!(ports.get_handle(user, 0, 0), handle, "reinitializing the library lost the open port");
        for _ in 0..4 {
            let rejected = ports.open(user, 0, 0);
            assert_eq!(rejected, PAD_ERR_ALREADY_OPENED, "a duplicate open exposed another controller");
            assert_eq!(ports.close(rejected as i32), PAD_ERR_INVALID_HANDLE);
            assert_eq!(ports.get_handle(user, 0, 0), handle, "closing a rejected duplicate released the owner");
        }
        assert_eq!(ports.get_handle(user + 1, 0, 0), PAD_ERR_DEVICE_NO_HANDLE);
        assert_eq!(ports.get_handle(user, SPECIAL_PORT_TYPE, 0), PAD_ERR_DEVICE_NO_HANDLE);
        assert_eq!(ports.get_handle(user, 0, 1), PAD_ERR_DEVICE_NO_HANDLE, "another port tuple aliased the pad");
        assert_eq!(ports.open(user, 1, 0), PAD_ERR_INVALID_ARG);
        assert_eq!(ports.open(user, 0, 1), 0x809B_0003u32 as i32 as i64, "a non-zero index has its own error");
        assert_eq!(ports.open(-1, 0, 0), 0x809B_0001u32 as i32 as i64, "an invalid user id has its own error");

        // Input of the shared host pad reaches the owned port; a host
        // disconnection does not release it.
        assert_eq!(ports.check(handle as i32).map(port_pad_slot), Ok(0));
        let mut core = PadCore::new();
        core.publish(HostPadState { buttons: button::CROSS, ..HostPadState::NEUTRAL }, 10);
        assert!(core.sample.connected && core.sample.buttons == button::CROSS);
        core.publish(HostPadState { connected: false, ..HostPadState::NEUTRAL }, 20);
        assert_eq!(ports.get_handle(user, 0, 0), handle);
        assert_eq!(ports.open(user, 0, 0), PAD_ERR_ALREADY_OPENED, "host disconnection released the port");

        assert_eq!(ports.close(handle as i32), OK);
        assert_eq!(ports.get_handle(user, 0, 0), PAD_ERR_DEVICE_NO_HANDLE);
        assert_eq!(ports.check(handle as i32), Err(PAD_ERR_INVALID_HANDLE), "a closed port kept its handle");
        assert_eq!(ports.close(handle as i32), PAD_ERR_INVALID_HANDLE);
        assert_eq!(ports.check(0), Err(PAD_ERR_INVALID_HANDLE));
        assert_eq!(ports.open(user, 0, 0), handle, "a closed port could not reopen");
        assert_eq!(ports.check(handle as i32), Ok(0));

        // The special and remote control ports are owned separately.
        let special = ports.open(user, SPECIAL_PORT_TYPE, 0);
        assert_eq!(special, port_handle(SPECIAL_PORT_BASE) as HleResult);
        assert_eq!(ports.open(user, SPECIAL_PORT_TYPE, 0), PAD_ERR_ALREADY_OPENED);
        let remote = ports.open(SYSTEM_USER_ID, REMOTE_CONTROL_PORT_TYPE, 0);
        assert_eq!(remote, port_handle(REMOTE_CONTROL_PORT) as HleResult);
        assert_eq!(ports.close(special as i32), OK);
        assert_eq!(ports.get_handle(user, 0, 0), handle);
        assert_eq!(ports.check(remote as i32).map(port_pad_slot), Ok(0));
    }

    #[test]
    fn connected_count_counts_connections() {
        let mut core = PadCore::with_connected(false);
        assert_eq!(core.read(4)[0].connected, false);
        core.publish(HostPadState::NEUTRAL, 10);
        assert_eq!(core.sample.connected_count, 1);
        core.publish(HostPadState { connected: false, ..HostPadState::NEUTRAL }, 20);
        let r = core.read(4);
        assert_eq!((r.len(), r[0].connected, r[0].timestamp, r[0].connected_count), (1, false, 0, 1));
        core.publish(HostPadState::NEUTRAL, 30);
        assert_eq!(core.sample.connected_count, 2);
    }

    #[test]
    fn trigger_states_follow_published_triggers() {
        let mut core = PadCore::new();
        let mut b = [0u8; TRIGGER_EFFECT_PARAM_SIZE];
        b[0] = 2;
        b[64..68].copy_from_slice(&2u32.to_le_bytes());
        b[72..75].copy_from_slice(&[2, 7, 6]);
        let param = TriggerEffectParam::parse(&b);
        let e = core.output.set_trigger_effect(&param, [0, 0], &crate::pad_controller::Settings::new()).unwrap().unwrap();
        assert_eq!(e.enable_bits(), crate::pad_controller::DS5_ENABLE_RIGHT_TRIGGER);
        assert_eq!(core.output.trigger_state(), [0, 3]);
        core.publish(HostPadState { r2: 224, ..HostPadState::NEUTRAL }, 1);
        core.publish(HostPadState { r2: 100, ..HostPadState::NEUTRAL }, 2);
        assert_eq!(core.output.trigger_state(), [0, 5]);
        core.controller_changed(3);
        assert_eq!(core.output.trigger_state(), [0, 0]);
    }

    #[test]
    fn motion_disable_keeps_orientation() {
        let mut core = PadCore::new();
        let rate = std::f32::consts::FRAC_PI_2;
        for i in 0..=20u64 {
            let m = HostMotion { accel: [0.0, STANDARD_GRAVITY, 0.0], angular_velocity: [0.0, rate, i as f32 * 1e-7] };
            core.publish(HostPadState { motion: Some(m), ..HostPadState::NEUTRAL }, 1_000 + i * 5_000);
        }
        let q = core.sample.orientation;
        assert_ne!(q, IDENTITY);
        core.set_motion_enabled(false, 200_000);
        assert_eq!(core.sample.orientation, q);
        assert_eq!(core.sample.accel, REST_ACCEL);
        assert_eq!(core.sample.angular_velocity, [0.0; 3]);
        core.controller_changed(300_000);
        assert_eq!(core.sample.orientation, IDENTITY);
    }

    #[test]
    fn keyboard_nids() {
        for (nid, name) in [("0LWei+c7RNc", "sceKeyboardClose"), ("xybbGMCr738", "sceKeyboardRead")] {
            assert_eq!(crate::sync::test_util::nid(name), nid);
        }
    }

    /// Key2Char: usages outside 4..=231 fail before the handle and leave the
    /// output alone; keys that type nothing keep their usage unprocessed;
    /// the Kana LED changes nothing; both arrangements convert.
    #[test]
    fn keyboard_key2char_contract() {
        use crate::CpuRegister::{Rcx, Rdi, Rdx, Rsi, R8, R9};
        const BASE: u64 = 0x5C82_0100_0000;
        const SHIFT: u32 = 0x02;
        const KANA: u32 = 16;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let call = |ctx: &mut CpuContext, handle: i32, arrange: i32, led: u32, modifier: u32, key: u16, out: u64| {
            for (r, v) in [(Rdi, handle as u32 as u64), (Rsi, arrange as u64), (Rdx, led as u64), (Rcx, modifier as u64), (R8, key as u64), (R9, out)] {
                ctx.set(r, v);
            }
            cb_keyboard_get_key2char(ctx)
        };
        // (processed, length, char code)
        let read = |ctx: &CpuContext| {
            let b = ctx.read_bytes(BASE, KEYBOARD_CHAR_DATA_SIZE).unwrap();
            (b[0], i32::from_le_bytes(b[4..8].try_into().unwrap()), u16::from_le_bytes([b[8], b[9]]))
        };
        for key in [0u16, 1, 2, 3, 232, 255, 256, 65535] {
            for handle in [1, 42, 0] {
                ctx.write_bytes(BASE, &[0xA5; KEYBOARD_CHAR_DATA_SIZE]);
                assert_eq!(call(&mut ctx, handle, KEYBOARD_MAPPING_101, 0, 0, key, BASE), KEYBOARD_ERR_INVALID_ARG, "key {key:#x}");
                assert_eq!(ctx.read_bytes(BASE, KEYBOARD_CHAR_DATA_SIZE).unwrap(), vec![0xA5; KEYBOARD_CHAR_DATA_SIZE]);
                assert_eq!(call(&mut ctx, handle, KEYBOARD_MAPPING_106, KANA, 0, key, 0), KEYBOARD_ERR_INVALID_ARG);
            }
        }
        assert_eq!(call(&mut ctx, 0, KEYBOARD_MAPPING_101, 0, 0, 4, BASE), KEYBOARD_ERR_INVALID_HANDLE);
        assert_eq!(call(&mut ctx, 1, KEYBOARD_MAPPING_101, 0, 0, 4, 0), KEYBOARD_ERR_INVALID_ARG);
        assert_eq!(call(&mut ctx, 1, 2, 0, 0, 4, BASE), KEYBOARD_ERR_INVALID_ARG);
        for arrange in [KEYBOARD_MAPPING_101, KEYBOARD_MAPPING_106] {
            for key in [0x29u16, 0x39, 0x3A, 0x4F, 0x59, 0x88, 0xE0, 0xE7] {
                assert_eq!(call(&mut ctx, 1, arrange, 0, 0, key, BASE), OK);
                assert_eq!(read(&ctx), (0, 0, key), "key {key:#x}");
            }
            for modifier in [0, SHIFT, 0x20] {
                for led in [0, KEYBOARD_LED_NUM_LOCK | KEYBOARD_LED_CAPS_LOCK] {
                    for key in KEYBOARD_KEYS {
                        assert_eq!(call(&mut ctx, 1, arrange, led, modifier, key, BASE), OK);
                        let plain = read(&ctx);
                        assert_eq!(call(&mut ctx, 1, arrange, led | KANA, modifier, key, BASE), OK);
                        assert_eq!(read(&ctx), plain);
                        assert!(plain.0 == 1 && plain.1 == 1 && plain.2 != 0 || plain == (0, 0, key));
                    }
                }
            }
        }
        let us = |led, modifier, key| keyboard_character(false, led, modifier, key);
        let jis = |modifier, key| keyboard_character(true, 0, modifier, key);
        assert_eq!((us(0, 0, 0x04), us(0, SHIFT, 0x04), us(KEYBOARD_LED_CAPS_LOCK, SHIFT, 0x04)), (b'a' as u16, b'A' as u16, b'a' as u16));
        assert_eq!((us(0, 0, 0x38), us(0, SHIFT, 0x38)), (b'/' as u16, b'?' as u16));
        assert_eq!((us(0, 0, 0x59), us(KEYBOARD_LED_NUM_LOCK, 0, 0x59), us(KEYBOARD_LED_NUM_LOCK, 0, 0x63)), (0, b'1' as u16, b'.' as u16));
        assert_eq!((us(0, 0, 0x54), us(0, 0, 0x57), us(0, 0, 0x58)), (b'/' as u16, b'+' as u16, b'\n' as u16));
        assert_eq!((us(0, 0, 0x32), us(0, SHIFT, 0x32), us(0, SHIFT, 0x64)), (b'\\' as u16, b'|' as u16, b'|' as u16));
        assert_eq!((us(0, 0, 0x87), us(0, 0, 0x89)), (0, 0));
        assert_eq!((jis(SHIFT, 0x1F), jis(SHIFT, 0x27), jis(0, 0x2E), jis(SHIFT, 0x2F)), (b'"' as u16, 0, b'^' as u16, b'`' as u16));
        assert_eq!((jis(0, 0x34), jis(SHIFT, 0x34), jis(0, 0x35)), (b':' as u16, b'*' as u16, 0));
        assert_eq!((jis(0, 0x87), jis(SHIFT, 0x87), jis(0, 0x89), jis(SHIFT, 0x89)), (b'\\' as u16, b'_' as u16, 0xA5, b'|' as u16));
    }

    #[test]
    fn trigger_effect_state_argument_checks() {
        const BASE: u64 = 0x5C82_0000_0000;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        {
            // The only test using the global ports: the primary pad is open.
            let mut ports = ports();
            ports.init();
            let _ = ports.open(crate::user_service::PRIMARY_USER_ID, STANDARD_PORT_TYPE, 0);
        }
        // Vibration mode: a mode of 0 or less fails before the handle, the
        // handle before the other modes.
        for (handle, mode, expect) in [
            (99u64, -1i64, PAD_ERR_INVALID_ARG),
            (99, 0, PAD_ERR_INVALID_ARG),
            (99, 1, PAD_ERR_INVALID_HANDLE),
            (99, 4, PAD_ERR_INVALID_HANDLE),
            (PRIMARY_PAD_HANDLE as u64, 0, PAD_ERR_INVALID_ARG),
            (PRIMARY_PAD_HANDLE as u64, 1, OK),
            (PRIMARY_PAD_HANDLE as u64, 2, OK),
            (PRIMARY_PAD_HANDLE as u64, 3, OK),
            (PRIMARY_PAD_HANDLE as u64, 5, OK),
            (PRIMARY_PAD_HANDLE as u64, 4, PAD_ERR_INVALID_ARG),
            (PRIMARY_PAD_HANDLE as u64, 6, PAD_ERR_INVALID_ARG),
        ] {
            ctx.set(crate::CpuRegister::Rdi, handle);
            ctx.set(crate::CpuRegister::Rsi, mode as u64);
            assert_eq!(cb_pad_set_vibration_mode(&mut ctx), expect, "handle {handle} mode {mode}");
        }
        // The undocumented export answers 0 once initialized (a game
        // process) and touches no memory.
        ctx.write_bytes(BASE, &[0xEE; 16]);
        for (r, v) in [(crate::CpuRegister::Rdi, BASE), (crate::CpuRegister::Rcx, 16)] {
            ctx.set(r, v);
        }
        assert_eq!(cb_pad_n3k_sx62fg_no(&mut ctx), OK);
        assert_eq!(ctx.read_bytes(BASE, 16).unwrap(), vec![0xEE; 16]);
        // Accessibility controller state: the output, then the handle.
        ctx.set(crate::CpuRegister::Rdi, 0x7777);
        ctx.set(crate::CpuRegister::Rsi, 0);
        assert_eq!(cb_pad_get_accessibility_controller_state(&mut ctx), PAD_ERR_INVALID_ARG);
        ctx.set(crate::CpuRegister::Rsi, BASE);
        assert_eq!(cb_pad_get_accessibility_controller_state(&mut ctx), PAD_ERR_INVALID_HANDLE);
        ctx.set(crate::CpuRegister::Rdi, PRIMARY_PAD_HANDLE as u64);
        assert_eq!(cb_pad_get_accessibility_controller_state(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE), Some(0));
        // Angular velocity settings need an open handle.
        for enable in [0, 1] {
            ctx.set(crate::CpuRegister::Rsi, enable);
            ctx.set(crate::CpuRegister::Rdi, PRIMARY_PAD_HANDLE as u64);
            assert_eq!(cb_pad_set_angular_velocity_bias_correction_state(&mut ctx), OK);
            assert_eq!(cb_pad_set_angular_velocity_deadband_state(&mut ctx), OK);
            ctx.set(crate::CpuRegister::Rdi, PRIMARY_PAD_HANDLE as u64 + 1);
            assert_eq!(cb_pad_set_angular_velocity_bias_correction_state(&mut ctx), PAD_ERR_INVALID_HANDLE);
            assert_eq!(cb_pad_set_angular_velocity_deadband_state(&mut ctx), PAD_ERR_INVALID_HANDLE);
        }
        ctx.write_bytes(BASE, &[0xFF; 8]);
        ctx.set(crate::CpuRegister::Rdi, 99);
        ctx.set(crate::CpuRegister::Rsi, BASE);
        assert_eq!(cb_pad_get_trigger_effect_state(&mut ctx), PAD_ERR_INVALID_HANDLE);
        assert_eq!(ctx.read_bytes(BASE, 8).unwrap(), vec![0xFF; 8], "invalid handle changed the output");
        ctx.set(crate::CpuRegister::Rsi, 0);
        assert_eq!(cb_pad_get_trigger_effect_state(&mut ctx), PAD_ERR_INVALID_HANDLE, "handle checked first");
        ctx.set(crate::CpuRegister::Rdi, PRIMARY_PAD_HANDLE as u64);
        assert_eq!(cb_pad_get_trigger_effect_state(&mut ctx), PAD_ERR_INVALID_ARG);
        ctx.set(crate::CpuRegister::Rsi, BASE);
        assert_eq!(cb_pad_get_trigger_effect_state(&mut ctx), OK);
        // An invalid request is rejected with the pad's invalid-argument code.
        let mut param = [0u8; TRIGGER_EFFECT_PARAM_SIZE];
        param[0] = 2;
        // R2 weapon whose end position equals its start position.
        param[64] = 2;
        param[72..75].copy_from_slice(&[2, 2, 6]);
        ctx.write_bytes(BASE + 0x100, &param);
        ctx.set(crate::CpuRegister::Rsi, BASE + 0x100);
        assert_eq!(cb_pad_set_trigger_effect(&mut ctx), PAD_ERR_INVALID_ARG);
    }
}
