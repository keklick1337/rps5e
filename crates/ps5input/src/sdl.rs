// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Game controllers through SDL3, loaded at run time (`libSDL3.so.0`), so
//! the emulator builds and runs without SDL; controllers are then simply
//! unavailable.
//!
//! Every connected controller is opened (hot-plug through SDL events) and
//! the input layer assigns them to players. SDL's gamepad database maps
//! DualSense/DualShock, Xbox and most other controllers to one layout,
//! which the configurable button layout ([`crate::mapping::PadBinding`])
//! turns into pad buttons. DualSense extras: motion sensors, touch pad,
//! light bar, player LEDs and adaptive triggers through the raw effect
//! report; Xbox One style impulse triggers get the rumble fallback of
//! trigger effects.

use crate::assign::DeviceKey;
use crate::mapping::{PadBinding, PadInput};
use crate::output::OutputState;
use crate::shaping::{shape_stick, shape_trigger, AxisShape};
use libloading::Library;
use ps5hle::pad::{button, HostBattery, HostMotion, HostPadState, HostTouchPoint};
use std::ffi::{c_char, c_int, c_void, CStr};
use ps5hle::pad_controller::{trigger_rumble_strength, DualSenseEffects, TRIGGER_OFF};
use std::time::{Duration, Instant};

type Gamepad = c_void;
type JoystickId = u32;

const SDL_INIT_GAMEPAD: u32 = 0x2000;
const SDL_EVENT_GAMEPAD_ADDED: u32 = 0x653;
const SDL_EVENT_GAMEPAD_REMOVED: u32 = 0x654;

// SDL_GamepadType
const TYPE_XBOX360: c_int = 2;
const TYPE_XBOXONE: c_int = 3;
const TYPE_PS3: c_int = 4;
const TYPE_PS4: c_int = 5;
const TYPE_PS5: c_int = 6;
const TYPE_SWITCH_PRO: c_int = 7;

/// Number of SDL_GamepadButton values read.
const BUTTON_COUNT: u8 = 21;
const BTN_BACK: u8 = 4;

// SDL_GamepadAxis
const AXIS_LEFTX: c_int = 0;
const AXIS_LEFTY: c_int = 1;
const AXIS_RIGHTX: c_int = 2;
const AXIS_RIGHTY: c_int = 3;
const AXIS_LEFT_TRIGGER: c_int = 4;
const AXIS_RIGHT_TRIGGER: c_int = 5;
/// A trigger read as a button is pressed past half travel.
const TRIGGER_BUTTON_THRESHOLD: i16 = 16384;

const SENSOR_ACCEL: c_int = 1;
const SENSOR_GYRO: c_int = 2;

// SDL_PowerState / SDL_JoystickConnectionState
const POWER_NO_BATTERY: c_int = 2;
const POWER_CHARGING: c_int = 3;
const POWER_CHARGED: c_int = 4;
const CONNECTION_WIRELESS: c_int = 2;

/// Trigger rumble (controllers without adaptive triggers) lasts this long.
const TRIGGER_RUMBLE_MS: u32 = 0xffff;
/// Released controllers get this long to receive their last reports.
const RELEASE_FLUSH: Duration = Duration::from_millis(50);
/// A DualSense trigger effect "off".
pub const OFF_TRIGGER: [u8; 11] = [TRIGGER_OFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
/// Light bar colour while the title has none set.
pub const DEFAULT_LIGHTBAR: [u8; 3] = [0, 64, 255];

/// DualSense output report fields (SDL_SendGamepadEffect payload).
const DS5_EFFECT_LEN: usize = 47;
const DS5_ENABLE_RIGHT_TRIGGER: u8 = 0x04;
const DS5_ENABLE_LEFT_TRIGGER: u8 = 0x08;
const DS5_RIGHT_TRIGGER_OFFSET: usize = 10;
const DS5_LEFT_TRIGGER_OFFSET: usize = 21;

#[repr(C, align(8))]
struct Event([u8; 128]);

macro_rules! sdl_api {
    ($($name:ident : fn($($arg:ty),*) $(-> $ret:ty)?;)*) => {
        #[allow(non_snake_case)]
        struct Api {
            _lib: Library,
            $($name: unsafe extern "C" fn($($arg),*) $(-> $ret)?,)*
        }

        impl Api {
            #[allow(non_snake_case)]
            fn load() -> Result<Api, String> {
                let names: &[&str] = if cfg!(windows) {
                    &["SDL3.dll"]
                } else if cfg!(target_os = "macos") {
                    &["libSDL3.0.dylib", "libSDL3.dylib"]
                } else {
                    &["libSDL3.so.0", "libSDL3.so"]
                };
                let mut last = String::new();
                for n in names {
                    // SAFETY: SDL3 has no unsound load-time initialisers.
                    match unsafe { Library::new(n) } {
                        Ok(lib) => unsafe {
                            $(
                                let $name = *lib
                                    .get::<unsafe extern "C" fn($($arg),*) $(-> $ret)?>(
                                        concat!(stringify!($name), "\0").as_bytes(),
                                    )
                                    .map_err(|e| format!("{}: {e}", stringify!($name)))?;
                            )*
                            return Ok(Api { _lib: lib, $($name,)* });
                        },
                        Err(e) => last = e.to_string(),
                    }
                }
                Err(last)
            }
        }
    };
}

sdl_api! {
    SDL_SetHint: fn(*const c_char, *const c_char) -> bool;
    SDL_Init: fn(u32) -> bool;
    SDL_GetError: fn() -> *const c_char;
    SDL_PollEvent: fn(*mut Event) -> bool;
    SDL_GetGamepads: fn(*mut c_int) -> *mut JoystickId;
    SDL_free: fn(*mut c_void);
    SDL_OpenGamepad: fn(JoystickId) -> *mut Gamepad;
    SDL_CloseGamepad: fn(*mut Gamepad);
    SDL_GetGamepadID: fn(*mut Gamepad) -> JoystickId;
    SDL_GetGamepadName: fn(*mut Gamepad) -> *const c_char;
    SDL_GetGamepadSerial: fn(*mut Gamepad) -> *const c_char;
    SDL_GetGamepadType: fn(*mut Gamepad) -> c_int;
    SDL_GetGamepadButton: fn(*mut Gamepad, c_int) -> bool;
    SDL_GetGamepadAxis: fn(*mut Gamepad, c_int) -> i16;
    SDL_GamepadHasSensor: fn(*mut Gamepad, c_int) -> bool;
    SDL_SetGamepadSensorEnabled: fn(*mut Gamepad, c_int, bool) -> bool;
    SDL_GetGamepadSensorData: fn(*mut Gamepad, c_int, *mut f32, c_int) -> bool;
    SDL_GetNumGamepadTouchpads: fn(*mut Gamepad) -> c_int;
    SDL_GetGamepadTouchpadFinger: fn(*mut Gamepad, c_int, c_int, *mut bool, *mut f32, *mut f32, *mut f32) -> bool;
    SDL_GetGamepadPowerInfo: fn(*mut Gamepad, *mut c_int) -> c_int;
    SDL_GetGamepadConnectionState: fn(*mut Gamepad) -> c_int;
    SDL_SetGamepadPlayerIndex: fn(*mut Gamepad, c_int) -> bool;
    SDL_RumbleGamepad: fn(*mut Gamepad, u16, u16, u32) -> bool;
    SDL_RumbleGamepadTriggers: fn(*mut Gamepad, u16, u16, u32) -> bool;
    SDL_SetGamepadLED: fn(*mut Gamepad, u8, u8, u8) -> bool;
    SDL_SendGamepadEffect: fn(*mut Gamepad, *const c_void, c_int) -> bool;
}

fn cstr(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        // SAFETY: SDL returns NUL-terminated strings.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

/// Stick axis -32768..32767 to 0..255 (128 = centre).
pub fn stick_value(v: i16) -> u8 {
    (((v as i32 + 32768) * 255 + 32767) / 65535) as u8
}

/// Trigger axis 0..32767 to 0..255.
pub fn trigger_value(v: i16) -> u8 {
    ((v.max(0) as i32 * 255 + 16383) / 32767) as u8
}

/// Controller family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadKind {
    DualSense,
    DualShock4,
    DualShock3,
    Xbox,
    SwitchPro,
    Other,
}

impl PadKind {
    fn from_sdl(ty: c_int) -> Self {
        match ty {
            TYPE_PS5 => PadKind::DualSense,
            TYPE_PS4 => PadKind::DualShock4,
            TYPE_PS3 => PadKind::DualShock3,
            TYPE_XBOX360 | TYPE_XBOXONE => PadKind::Xbox,
            TYPE_SWITCH_PRO => PadKind::SwitchPro,
            _ => PadKind::Other,
        }
    }

    pub fn is_playstation(self) -> bool {
        matches!(self, PadKind::DualSense | PadKind::DualShock4 | PadKind::DualShock3)
    }

    pub fn label(self) -> &'static str {
        match self {
            PadKind::DualSense => "DualSense",
            PadKind::DualShock4 => "DualShock 4",
            PadKind::DualShock3 => "DualShock 3",
            PadKind::Xbox => "Xbox",
            PadKind::SwitchPro => "Switch Pro",
            PadKind::Other => "Gamepad",
        }
    }
}

/// A connected controller.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceInfo {
    pub id: u32,
    pub name: String,
    pub serial: String,
    pub kind: PadKind,
    pub motion: bool,
    pub touchpad: bool,
    pub battery: Option<HostBattery>,
}

impl DeviceInfo {
    pub fn key(&self) -> DeviceKey {
        DeviceKey { id: self.id, name: self.name.clone(), serial: self.serial.clone() }
    }

    /// What a settings file stores to select this device.
    pub fn selector(&self) -> String {
        if self.serial.is_empty() { self.name.clone() } else { self.serial.clone() }
    }
}

/// How a controller snapshot is built.
#[derive(Clone, Debug)]
pub struct PadConfig {
    pub bindings: Vec<PadBinding>,
    pub left_stick: AxisShape,
    pub right_stick: AxisShape,
    pub triggers: AxisShape,
    pub motion: bool,
    pub touchpad: bool,
}

impl Default for PadConfig {
    fn default() -> Self {
        Self {
            bindings: crate::mapping::default_pad_bindings(),
            left_stick: AxisShape::LINEAR,
            right_stick: AxisShape::LINEAR,
            triggers: AxisShape::LINEAR,
            motion: true,
            touchpad: true,
        }
    }
}

/// Output feature switches applied when sending to a device.
#[derive(Clone, Copy, Debug)]
pub struct OutputFeatures {
    pub rumble: bool,
    pub rumble_strength: f32,
    pub lightbar: bool,
    pub adaptive_triggers: bool,
    /// Fixed light bar colour (also used before the game sets one).
    pub lightbar_color: Option<[u8; 3]>,
}

impl Default for OutputFeatures {
    fn default() -> Self {
        Self { rumble: true, rumble_strength: 1.0, lightbar: true, adaptive_triggers: true, lightbar_color: None }
    }
}

struct Pad {
    ptr: *mut Gamepad,
    info: DeviceInfo,
    applied_seq: u64,
    player_index: Option<usize>,
}

/// The SDL gamepad backend; lives on one thread (the input thread, or
/// the frontend's UI thread).
pub struct SdlGamepads {
    api: Api,
    pads: Vec<Pad>,
}

// SAFETY: used only from the thread that created it.
unsafe impl Send for SdlGamepads {}

impl SdlGamepads {
    pub fn new() -> Result<Self, String> {
        let api = Api::load().map_err(|e| format!("SDL3 not available ({e})"))?;
        unsafe {
            let hint = |k: &CStr, v: &CStr| (api.SDL_SetHint)(k.as_ptr(), v.as_ptr());
            // The emulator owns the process signal handlers.
            hint(c"SDL_NO_SIGNAL_HANDLERS", c"1");
            // The window is not an SDL window: always deliver controller input.
            hint(c"SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", c"1");
            // Full DualSense/DualShock reports (sensors, touch pad, effects).
            hint(c"SDL_JOYSTICK_ENHANCED_REPORTS", c"1");
            if !(api.SDL_Init)(SDL_INIT_GAMEPAD) {
                return Err(format!("SDL_Init failed: {}", cstr((api.SDL_GetError)())));
            }
        }
        let mut s = SdlGamepads { api, pads: Vec::new() };
        s.open_all();
        if s.pads.is_empty() {
            log::info!("input: SDL3 game controller support ready (no controller connected yet)");
        }
        Ok(s)
    }

    fn open_all(&mut self) {
        let mut n: c_int = 0;
        let ids = unsafe { (self.api.SDL_GetGamepads)(&mut n) };
        if ids.is_null() {
            return;
        }
        let list: Vec<JoystickId> = (0..n.max(0) as usize).map(|i| unsafe { *ids.add(i) }).collect();
        unsafe { (self.api.SDL_free)(ids.cast()) };
        for id in list {
            self.open(id);
        }
    }

    fn open(&mut self, id: JoystickId) {
        if self.pads.iter().any(|p| p.info.id == id) {
            return;
        }
        let a = &self.api;
        let ptr = unsafe { (a.SDL_OpenGamepad)(id) };
        if ptr.is_null() {
            log::warn!("input: cannot open game controller {id}: {}", cstr(unsafe { (a.SDL_GetError)() }));
            return;
        }
        let info = unsafe {
            let has_accel = (a.SDL_GamepadHasSensor)(ptr, SENSOR_ACCEL);
            let has_gyro = (a.SDL_GamepadHasSensor)(ptr, SENSOR_GYRO);
            let motion = has_accel
                && has_gyro
                && (a.SDL_SetGamepadSensorEnabled)(ptr, SENSOR_ACCEL, true)
                && (a.SDL_SetGamepadSensorEnabled)(ptr, SENSOR_GYRO, true);
            DeviceInfo {
                id: (a.SDL_GetGamepadID)(ptr),
                name: cstr((a.SDL_GetGamepadName)(ptr)),
                serial: cstr((a.SDL_GetGamepadSerial)(ptr)),
                kind: PadKind::from_sdl((a.SDL_GetGamepadType)(ptr)),
                motion,
                touchpad: (a.SDL_GetNumGamepadTouchpads)(ptr) > 0,
                battery: None,
            }
        };
        log::info!(
            "input: game controller '{}' connected ({}, motion {}, touch pad {})",
            info.name,
            info.kind.label(),
            info.motion,
            info.touchpad
        );
        self.pads.push(Pad { ptr, info, applied_seq: u64::MAX, player_index: None });
    }

    fn close(&mut self, id: JoystickId) {
        if let Some(i) = self.pads.iter().position(|p| p.info.id == id) {
            let p = self.pads.remove(i);
            log::info!("input: game controller '{}' disconnected", p.info.name);
            unsafe { (self.api.SDL_CloseGamepad)(p.ptr) };
        }
    }

    /// Process hot-plug events (also updates SDL's controller state).
    /// Returns true when a controller was connected or disconnected.
    pub fn pump(&mut self) -> bool {
        let mut ev = Event([0; 128]);
        let mut changed = false;
        while unsafe { (self.api.SDL_PollEvent)(&mut ev) } {
            let ty = u32::from_ne_bytes(ev.0[0..4].try_into().unwrap());
            let which = u32::from_ne_bytes(ev.0[16..20].try_into().unwrap());
            match ty {
                SDL_EVENT_GAMEPAD_ADDED => {
                    self.open(which);
                    changed = true;
                }
                SDL_EVENT_GAMEPAD_REMOVED => {
                    self.close(which);
                    changed = true;
                }
                _ => {}
            }
        }
        changed
    }

    fn pad(&self, id: u32) -> Option<&Pad> {
        self.pads.iter().find(|p| p.info.id == id)
    }

    fn battery(&self, g: *mut Gamepad) -> Option<HostBattery> {
        let mut percent: c_int = -1;
        let state = unsafe { (self.api.SDL_GetGamepadPowerInfo)(g, &mut percent) };
        let conn = unsafe { (self.api.SDL_GetGamepadConnectionState)(g) };
        if (state <= 0 && percent < 0) || state == POWER_NO_BATTERY {
            return None;
        }
        Some(HostBattery {
            percent: (0..=100).contains(&percent).then_some(percent as u8),
            charging: matches!(state, POWER_CHARGING | POWER_CHARGED),
            wireless: conn == CONNECTION_WIRELESS,
        })
    }

    /// Connected controllers in connection order (battery refreshed).
    pub fn devices(&self) -> Vec<DeviceInfo> {
        self.pads
            .iter()
            .map(|p| DeviceInfo { battery: self.battery(p.ptr), ..p.info.clone() })
            .collect()
    }

    pub fn keys(&self) -> Vec<DeviceKey> {
        self.pads.iter().map(|p| p.info.key()).collect()
    }

    /// Show the player number on the device (DualSense player LEDs,
    /// Xbox ring).
    pub fn set_player_index(&mut self, id: u32, player: Option<usize>) {
        if let Some(p) = self.pads.iter_mut().find(|p| p.info.id == id) {
            if p.player_index != player {
                p.player_index = player;
                let idx = player.map(|i| i as c_int).unwrap_or(-1);
                unsafe { (self.api.SDL_SetGamepadPlayerIndex)(p.ptr, idx) };
            }
        }
    }

    /// Raw inputs currently held on a device (for previews and capture).
    pub fn pressed_inputs(&self, id: u32) -> Vec<PadInput> {
        let Some(p) = self.pad(id) else { return Vec::new() };
        let (a, g) = (&self.api, p.ptr);
        let mut v: Vec<PadInput> =
            (0..BUTTON_COUNT).filter(|&b| unsafe { (a.SDL_GetGamepadButton)(g, b as c_int) }).map(PadInput).collect();
        if unsafe { (a.SDL_GetGamepadAxis)(g, AXIS_LEFT_TRIGGER) } > TRIGGER_BUTTON_THRESHOLD {
            v.push(PadInput::LEFT_TRIGGER);
        }
        if unsafe { (a.SDL_GetGamepadAxis)(g, AXIS_RIGHT_TRIGGER) } > TRIGGER_BUTTON_THRESHOLD {
            v.push(PadInput::RIGHT_TRIGGER);
        }
        v
    }

    /// Current state of a device in pad conventions, `None` if it is gone.
    pub fn snapshot(&self, id: u32, cfg: &PadConfig) -> Option<HostPadState> {
        let p = self.pad(id)?;
        let a = &self.api;
        let g = p.ptr;
        let axis = |x| unsafe { (a.SDL_GetGamepadAxis)(g, x) };
        let pressed = self.pressed_inputs(id);
        let mut s = HostPadState::NEUTRAL;
        s.buttons = buttons_from_inputs(&pressed, &cfg.bindings, p.info.kind.is_playstation());
        let (lx, ly) = shape_stick(axis(AXIS_LEFTX), axis(AXIS_LEFTY), &cfg.left_stick);
        let (rx, ry) = shape_stick(axis(AXIS_RIGHTX), axis(AXIS_RIGHTY), &cfg.right_stick);
        s.left_x = stick_value(lx);
        s.left_y = stick_value(ly);
        s.right_x = stick_value(rx);
        s.right_y = stick_value(ry);
        if analog_trigger_reported(&cfg.bindings, PadInput::LEFT_TRIGGER, button::L2) {
            s.l2 = trigger_value(shape_trigger(axis(AXIS_LEFT_TRIGGER), &cfg.triggers));
        }
        if analog_trigger_reported(&cfg.bindings, PadInput::RIGHT_TRIGGER, button::R2) {
            s.r2 = trigger_value(shape_trigger(axis(AXIS_RIGHT_TRIGGER), &cfg.triggers));
        }
        if p.info.motion && cfg.motion {
            let mut accel = [0f32; 3];
            let mut gyro = [0f32; 3];
            let ok = unsafe {
                (a.SDL_GetGamepadSensorData)(g, SENSOR_ACCEL, accel.as_mut_ptr(), 3)
                    && (a.SDL_GetGamepadSensorData)(g, SENSOR_GYRO, gyro.as_mut_ptr(), 3)
            };
            if ok {
                s.motion = Some(HostMotion { accel, angular_velocity: gyro });
            }
        }
        if p.info.touchpad && cfg.touchpad {
            for finger in 0..2 {
                let (mut down, mut x, mut y, mut pressure) = (false, 0f32, 0f32, 0f32);
                let ok = unsafe {
                    (a.SDL_GetGamepadTouchpadFinger)(g, 0, finger, &mut down, &mut x, &mut y, &mut pressure)
                };
                if ok && down {
                    s.touch[finger as usize] = Some(HostTouchPoint { x, y });
                }
            }
        }
        // A touch pad click without a finger reading (Back/View on other
        // controllers) touches the centre.
        if s.buttons & button::TOUCH_PAD != 0 && !p.info.touchpad && s.touch.iter().all(Option::is_none) {
            s.touch[0] = Some(HostTouchPoint { x: 0.5, y: 0.5 });
        }
        s.battery = self.battery(g);
        Some(s.normalized())
    }

    /// Send rumble / light bar / trigger effects to a device when they
    /// changed. A DualSense's rumble goes through the haptics layer, which
    /// mutes the motors while its actuators play audio.
    pub fn apply_output(&mut self, id: u32, out: &OutputState, features: &OutputFeatures) {
        let Some(p) = self.pads.iter_mut().find(|p| p.info.id == id) else { return };
        if out.seq == p.applied_seq {
            return;
        }
        let (a, g) = (&self.api, p.ptr);
        let first = p.applied_seq == u64::MAX;
        p.applied_seq = out.seq;
        let scale = |v: u8| -> u8 {
            if !features.rumble {
                return 0;
            }
            (v as f32 * features.rumble_strength.clamp(0.0, 2.0)).round().clamp(0.0, 255.0) as u8
        };
        let (large, small) = (scale(out.large), scale(out.small));
        let duration = out.rumble_until.map_or(0, |t| t.saturating_duration_since(Instant::now()).as_millis().min(u32::MAX as u128) as u32);
        let is_ps5 = p.info.kind == PadKind::DualSense;
        let by_haptics = is_ps5
            && crate::haptics::instance().is_some_and(|h| h.set_vibration(id as i32, large, small, duration));
        unsafe {
            if !by_haptics {
                (a.SDL_RumbleGamepad)(g, large as u16 * 0x101, small as u16 * 0x101, duration);
            }
            if features.lightbar {
                let [r, gr, b] = features.lightbar_color.or(out.lightbar).unwrap_or(DEFAULT_LIGHTBAR);
                (a.SDL_SetGamepadLED)(g, r, gr, b);
            }
            if out.triggers_touched || (first && is_ps5) {
                let triggers = if features.adaptive_triggers { out.triggers } else { [OFF_TRIGGER; 2] };
                if is_ps5 {
                    let report = ds5_trigger_report(&triggers);
                    (a.SDL_SendGamepadEffect)(g, report.as_ptr().cast(), report.len() as c_int);
                } else {
                    let fallback = triggers.map(|t| if features.rumble { trigger_rumble_strength(&t) as u16 * 257 } else { 0 });
                    let on = fallback != [0, 0];
                    (a.SDL_RumbleGamepadTriggers)(g, fallback[0], fallback[1], if on { TRIGGER_RUMBLE_MS } else { 0 });
                }
            }
        }
    }

    /// Set a controller's light bar (colour preview in the frontend).
    pub fn set_led(&mut self, id: u32, color: [u8; 3]) {
        if let Some(p) = self.pads.iter().find(|p| p.info.id == id) {
            unsafe { (self.api.SDL_SetGamepadLED)(p.ptr, color[0], color[1], color[2]) };
        }
    }

    /// Stop the controllers' outputs (trigger effects off, motors and
    /// light bar off) before they are closed.
    pub fn release_all(&mut self) {
        let a = &self.api;
        for p in &self.pads {
            unsafe {
                if p.info.kind == PadKind::DualSense {
                    let off = DualSenseEffects::off();
                    (a.SDL_SendGamepadEffect)(p.ptr, off.0.as_ptr().cast(), off.0.len() as c_int);
                }
                (a.SDL_RumbleGamepad)(p.ptr, 0, 0, 0);
                (a.SDL_SetGamepadLED)(p.ptr, 0, 0, 0);
            }
        }
        if !self.pads.is_empty() {
            // Let the reports reach the controllers before closing them.
            std::thread::sleep(RELEASE_FLUSH);
        }
    }
}

/// Pad buttons for the held controller inputs. On controllers without a
/// touch pad, Back/View acts as the touch pad unless it is bound; on
/// PlayStation controllers the Create/Share button is system-reserved.
pub fn buttons_from_inputs(pressed: &[PadInput], bindings: &[PadBinding], playstation: bool) -> u32 {
    let mut bits = 0;
    for b in bindings {
        if pressed.contains(&b.input) {
            bits |= b.bit;
        }
    }
    let back = PadInput(BTN_BACK);
    if !playstation && pressed.contains(&back) && !bindings.iter().any(|b| b.input == back) {
        bits |= button::TOUCH_PAD;
    }
    bits
}

/// The analog trigger value is reported unless the trigger was rebound to
/// another button.
pub fn analog_trigger_reported(bindings: &[PadBinding], input: PadInput, bit: u32) -> bool {
    !bindings.iter().any(|b| b.input == input && b.bit != bit)
}

/// DualSense effect report setting both adaptive triggers (L2, R2).
pub fn ds5_trigger_report(triggers: &[[u8; 11]; 2]) -> [u8; DS5_EFFECT_LEN] {
    let mut r = [0u8; DS5_EFFECT_LEN];
    r[0] = DS5_ENABLE_RIGHT_TRIGGER | DS5_ENABLE_LEFT_TRIGGER;
    r[DS5_RIGHT_TRIGGER_OFFSET..DS5_RIGHT_TRIGGER_OFFSET + 11].copy_from_slice(&triggers[1]);
    r[DS5_LEFT_TRIGGER_OFFSET..DS5_LEFT_TRIGGER_OFFSET + 11].copy_from_slice(&triggers[0]);
    r
}

impl Drop for SdlGamepads {
    fn drop(&mut self) {
        self.release_all();
        for p in self.pads.drain(..) {
            unsafe { (self.api.SDL_CloseGamepad)(p.ptr) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::default_pad_bindings;

    #[test]
    fn axis_scaling() {
        assert_eq!(stick_value(i16::MIN), 0);
        assert_eq!(stick_value(0), 128);
        assert_eq!(stick_value(i16::MAX), 255);
        assert_eq!(trigger_value(-5), 0);
        assert_eq!(trigger_value(0), 0);
        assert_eq!(trigger_value(i16::MAX), 255);
        assert_eq!(trigger_value(16384), 128);
    }

    #[test]
    fn button_layout_and_remap() {
        let d = default_pad_bindings();
        assert_eq!(buttons_from_inputs(&[PadInput(0), PadInput(11)], &d, true), button::CROSS | button::UP);
        // Back: touch pad on other controllers, nothing on PlayStation ones.
        assert_eq!(buttons_from_inputs(&[PadInput(BTN_BACK)], &d, false), button::TOUCH_PAD);
        assert_eq!(buttons_from_inputs(&[PadInput(BTN_BACK)], &d, true), 0);
        // Swapped Cross/Circle (Nintendo-style).
        let mut cfg = crate::mapping::Config::default();
        cfg.set_pad_inputs(button::CROSS, &[PadInput(1)]);
        cfg.set_pad_inputs(button::CIRCLE, &[PadInput(0)]);
        assert_eq!(buttons_from_inputs(&[PadInput(1)], &cfg.pad_bindings, true), button::CROSS);
        // A trigger bound to a face button no longer reports its analog value.
        cfg.set_pad_inputs(button::SQUARE, &[PadInput::RIGHT_TRIGGER]);
        assert!(!analog_trigger_reported(&cfg.pad_bindings, PadInput::RIGHT_TRIGGER, button::R2));
        assert!(analog_trigger_reported(&cfg.pad_bindings, PadInput::LEFT_TRIGGER, button::L2));
        assert_eq!(buttons_from_inputs(&[PadInput::RIGHT_TRIGGER], &cfg.pad_bindings, true), button::SQUARE);
    }
}
