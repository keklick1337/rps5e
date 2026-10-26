// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Keyboard/mouse binding state and its composition with the game
//! controller snapshot into the pad state seen by the guest: buttons,
//! sticks (keys or mouse-look, optionally swapped), the motion sensors
//! ([`crate::motion`]) and the touch pad ([`crate::touch`]).
//!
//! The mouse does one thing at a time: while the mouse motion action is
//! held it turns the controller, in touch pad mode it moves a finger (the
//! left button touches, the right one clicks), and while captured it
//! drives the mouse-look stick.

use crate::mapping::{Binding, Config, Control, Source, Stick};
use crate::motion::{MotionGen, MotionSettings};
use crate::touch::{self, Finger, Slots, Swipe};
use ps5gpu::input::{KeyCode, MouseButton};
use ps5hle::pad::{button, HostPadState, HostTouchPoint};
use std::time::{Duration, Instant};

/// How long a wheel notch holds its bound button.
pub const WHEEL_PRESS: Duration = Duration::from_millis(80);
/// Mouse-look sampling period.
pub const MOUSE_POLL: Duration = Duration::from_millis(33);

/// Side effects of a key/mouse event the caller must apply.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Effects {
    pub toggle_fullscreen: bool,
    pub toggle_perf_overlay: bool,
    pub toggle_menu: bool,
    /// New mouse capture state, when it changed.
    pub capture_mouse: Option<bool>,
    /// Capture changed for a held mouse motion action (no hint).
    pub hold_capture: Option<bool>,
    /// Controller setting to step (cycle hotkeys).
    pub cycle: Option<ps5hle::pad_controller::Setting>,
    /// Create: take a screenshot.
    pub screenshot: bool,
    /// Mute: toggle the game's sound.
    pub toggle_mute: bool,
    /// Touch pad mode turned on or off.
    pub touch_mode: Option<bool>,
    /// The controls overlay key went down (`true`) or up (`false`).
    pub show_controls: Option<bool>,
}

/// Keyboard/mouse bindings and what is currently held.
pub struct KbmState {
    bindings: Vec<Binding>,
    pressed: Vec<bool>,
    wheel_release: Vec<Option<Instant>>,
    sensitivity: f64,
    invert: (bool, bool),
    /// Mouse-look active (ToggleMouse): mouse motion drives `look_stick`.
    pub mouse_look: bool,
    /// The stick mouse-look drives (`None`: none, capture only).
    look_stick: Option<Stick>,
    mouse_acc: (f64, f64),
    mouse_stick: [u8; 2],
    next_mouse_poll: Option<Instant>,
    motion: MotionGen,
    last_tick: Option<Instant>,
    /// Touch pad mode: numpad digits are zones, the mouse a finger.
    pub touch_mode: bool,
    touch_sensitivity: f32,
    /// Zones held in touch pad mode, in press order.
    zones: Vec<u8>,
    cursor: HostTouchPoint,
    /// Touch pad mode mouse buttons: left touches, right clicks.
    mouse_touch: bool,
    mouse_click: bool,
    swipe: Option<Swipe>,
    slots: Slots,
}

impl KbmState {
    pub fn new(bindings: Vec<Binding>, sensitivity: f64) -> Self {
        let n = bindings.len();
        Self {
            bindings,
            pressed: vec![false; n],
            wheel_release: vec![None; n],
            sensitivity,
            invert: (false, false),
            mouse_look: false,
            look_stick: Some(Stick::Right),
            mouse_acc: (0.0, 0.0),
            mouse_stick: [128, 128],
            next_mouse_poll: None,
            motion: MotionGen::default(),
            last_tick: None,
            touch_mode: false,
            touch_sensitivity: 1.0,
            zones: Vec::new(),
            cursor: touch::point(0.5, 0.5),
            mouse_touch: false,
            mouse_click: false,
            swipe: None,
            slots: Slots::default(),
        }
    }

    /// Player 1's keyboard/mouse state for the settings: the bindings of
    /// its device mode, mouse-look on the stick in mouse mode, the sticks
    /// swapped when set.
    pub fn from_config(cfg: &Config) -> Self {
        let mut bindings = cfg.active_bindings();
        let mut look = cfg.mouse_stick();
        if cfg.swap_sticks {
            for b in &mut bindings {
                b.control = swapped(b.control);
            }
            look = look.map(Stick::other);
        }
        Self::new(bindings, cfg.mouse_sensitivity)
            .with_invert(cfg.mouse_invert_x, cfg.mouse_invert_y)
            .with_look_stick(look)
            .with_motion(cfg.motion)
            .with_touch_sensitivity(cfg.touch_mouse_sensitivity)
    }

    /// The stick mouse-look drives.
    pub fn with_look_stick(mut self, stick: Option<Stick>) -> Self {
        self.look_stick = stick;
        self
    }

    /// Invert mouse-look axes.
    pub fn with_invert(mut self, x: bool, y: bool) -> Self {
        self.invert = (x, y);
        self
    }

    pub fn with_motion(mut self, settings: MotionSettings) -> Self {
        self.motion = MotionGen::new(settings);
        self
    }

    pub fn with_touch_sensitivity(mut self, s: f32) -> Self {
        self.touch_sensitivity = s;
        self
    }

    /// First key bound to `control` (for on-screen hints).
    pub fn key_for(&self, control: Control) -> Option<KeyCode> {
        self.bindings.iter().find_map(|b| match b.source {
            Source::Key(k) if b.control == control => Some(k),
            _ => None,
        })
    }

    fn held(&self, control: Control) -> bool {
        self.bindings.iter().zip(&self.pressed).any(|(b, &p)| p && b.control == control)
    }

    /// The mouse turns the controller (mouse motion held).
    pub fn mouse_motion_held(&self) -> bool {
        self.held(Control::MouseMotion)
    }

    /// The mouse is captured: mouse-look, touch pad mode or held mouse
    /// motion.
    pub fn captured(&self) -> bool {
        self.mouse_look || self.touch_mode || self.mouse_motion_held()
    }

    fn source_event(&mut self, src: Source, down: bool) -> Effects {
        let mut fx = Effects::default();
        let overlay_was = self.held(Control::ShowControls);
        for i in 0..self.bindings.len() {
            if self.bindings[i].source != src {
                continue;
            }
            let rising = down && !self.pressed[i];
            let falling = !down && self.pressed[i];
            match self.bindings[i].control {
                Control::ToggleFullscreen if rising => fx.toggle_fullscreen = true,
                Control::TogglePerfOverlay if rising => fx.toggle_perf_overlay = true,
                Control::ToggleMenu if rising => fx.toggle_menu = true,
                c @ (Control::SpeakerVolume | Control::VibrationIntensity | Control::TriggerEffectIntensity) if rising => {
                    fx.cycle = c.cycled_setting();
                }
                Control::ToggleMouse if rising => {
                    let on = !self.mouse_look;
                    self.set_mouse_look(on);
                    if !self.touch_mode {
                        fx.capture_mouse = Some(on);
                    }
                }
                Control::ReleaseMouse if rising && (self.mouse_look || self.touch_mode) => {
                    self.set_mouse_look(false);
                    if self.touch_mode {
                        self.set_touch_mode(false);
                        fx.touch_mode = Some(false);
                    }
                    fx.capture_mouse = Some(false);
                }
                Control::TouchpadMode if rising => {
                    let on = !self.touch_mode;
                    self.set_touch_mode(on);
                    fx.touch_mode = Some(on);
                }
                Control::Create if rising => fx.screenshot = true,
                Control::Mute if rising => fx.toggle_mute = true,
                Control::Shake if rising => self.motion.shake(),
                c @ (Control::SwipeUp | Control::SwipeDown | Control::SwipeLeft | Control::SwipeRight) if rising => {
                    self.swipe = Some(Swipe { dir: c, t: 0.0 });
                }
                Control::MouseMotion if (rising || falling) && !self.mouse_look && !self.touch_mode => {
                    fx.hold_capture = Some(rising);
                }
                _ => {}
            }
            self.pressed[i] = down;
        }
        let overlay = self.held(Control::ShowControls);
        if overlay != overlay_was {
            fx.show_controls = Some(overlay);
        }
        self.refresh_touch();
        fx
    }

    pub fn key(&mut self, code: KeyCode, down: bool, repeat: bool) -> Effects {
        if repeat {
            return Effects::default();
        }
        // Touch pad mode: the numpad digits touch their zones (unless one
        // is the mode's own key).
        if let Some(zone) = touch::zone_of(code).filter(|_| self.touch_mode) {
            let toggles = self.bindings.iter().any(|b| b.source == Source::Key(code) && b.control == Control::TouchpadMode);
            if !toggles {
                self.zones.retain(|z| *z != zone);
                if down {
                    self.zones.push(zone);
                }
                self.refresh_touch();
                return Effects::default();
            }
        }
        self.source_event(Source::Key(code), down)
    }

    pub fn mouse_button(&mut self, b: MouseButton, down: bool) -> Effects {
        // Touch pad mode: the left button puts the mouse finger down, the
        // right one also clicks.
        if self.touch_mode && matches!(b, MouseButton::Left | MouseButton::Right) {
            match b {
                MouseButton::Left => self.mouse_touch = down,
                _ => self.mouse_click = down,
            }
            self.refresh_touch();
            return Effects::default();
        }
        self.source_event(Source::Mouse(b), down)
    }

    /// A wheel notch presses the buttons bound to its direction (and
    /// releases those of the other one) for `WHEEL_PRESS`.
    pub fn wheel(&mut self, dy: f32, now: Instant) {
        let dir: i8 = if dy > 0.0 {
            1
        } else if dy < 0.0 {
            -1
        } else {
            return;
        };
        for i in 0..self.bindings.len() {
            if let Source::Wheel(d) = self.bindings[i].source {
                self.pressed[i] = d == dir;
                self.wheel_release[i] = (d == dir).then_some(now + WHEEL_PRESS);
            }
        }
    }

    pub fn mouse_motion(&mut self, dx: f64, dy: f64) {
        if self.mouse_motion_held() {
            self.motion.mouse(dx, dy);
        } else if self.touch_mode {
            self.cursor = touch::move_cursor(self.cursor, dx, dy, self.touch_sensitivity);
            self.refresh_touch();
        } else if self.mouse_look {
            self.mouse_acc.0 += if self.invert.0 { -dx } else { dx };
            self.mouse_acc.1 += if self.invert.1 { -dy } else { dy };
        }
    }

    pub fn set_mouse_look(&mut self, on: bool) {
        self.mouse_look = on;
        self.mouse_acc = (0.0, 0.0);
        self.mouse_stick = [128, 128];
        self.next_mouse_poll = None;
    }

    /// Enter or leave touch pad mode. Entering releases the bindings held
    /// on numpad digits (they become zones); leaving lifts the zone and
    /// mouse fingers.
    pub fn set_touch_mode(&mut self, on: bool) {
        self.touch_mode = on;
        self.zones.clear();
        self.mouse_touch = false;
        self.mouse_click = false;
        self.cursor = touch::point(0.5, 0.5);
        if on {
            for (b, p) in self.bindings.iter().zip(self.pressed.iter_mut()) {
                if matches!(b.source, Source::Key(k) if touch::zone_of(k).is_some()) && b.control != Control::TouchpadMode {
                    *p = false;
                }
            }
        }
        self.refresh_touch();
    }

    /// Focus lost: release everything, leave mouse-look and touch pad mode.
    pub fn release_all(&mut self) -> Effects {
        let was = self.captured();
        let touch = self.touch_mode;
        let overlay = self.held(Control::ShowControls);
        self.pressed.iter_mut().for_each(|p| *p = false);
        self.wheel_release.iter_mut().for_each(|w| *w = None);
        self.set_mouse_look(false);
        if touch {
            self.set_touch_mode(false);
        }
        self.refresh_touch();
        Effects {
            capture_mouse: was.then_some(false),
            touch_mode: touch.then_some(false),
            show_controls: overlay.then_some(false),
            ..Effects::default()
        }
    }

    /// Periodic work: wheel releases, the motion sensors, swipes and
    /// mouse-look sampling.
    pub fn tick(&mut self, now: Instant) {
        for i in 0..self.bindings.len() {
            if self.wheel_release[i].is_some_and(|t| now >= t) {
                self.wheel_release[i] = None;
                self.pressed[i] = false;
            }
        }
        let dt = self.last_tick.map_or(0.0, |t| now.saturating_duration_since(t).as_secs_f32());
        self.last_tick = Some(now);
        self.advance(dt);
        if !self.mouse_look {
            return;
        }
        match self.next_mouse_poll {
            Some(t) if now < t => return,
            _ => {}
        }
        self.next_mouse_poll = Some(now + MOUSE_POLL);
        let (dx, dy) = std::mem::take(&mut self.mouse_acc);
        self.mouse_stick = mouse_to_stick(dx, dy, self.sensitivity);
    }

    /// Advance the motion sensors and swipes by `dt` seconds.
    pub fn advance(&mut self, dt: f32) {
        let axis = |neg: bool, pos: bool| (pos as i8 - neg as i8) as f32;
        let pitch = axis(self.held(Control::TiltForward), self.held(Control::TiltBack));
        let roll = axis(self.held(Control::TiltRight), self.held(Control::TiltLeft));
        let yaw = axis(self.held(Control::TurnRight), self.held(Control::TurnLeft));
        self.motion.set_held(pitch, roll, yaw);
        self.motion.advance(dt);
        if let Some(sw) = self.swipe.as_mut() {
            sw.t += dt;
            if sw.point().is_none() {
                self.swipe = None;
            }
        }
        self.refresh_touch();
    }

    /// The simulated motion sensors.
    pub fn motion(&self) -> &MotionGen {
        &self.motion
    }

    /// Recompute the keyboard/mouse fingers.
    fn refresh_touch(&mut self) {
        let mut wanted: Vec<(Finger, HostTouchPoint)> = Vec::new();
        for c in [Control::TouchLeft, Control::TouchRight, Control::TapLeft, Control::TapCenter, Control::TapRight, Control::TwoFingerTouch] {
            if self.held(c) {
                if let Some(p) = touch::action_point(c) {
                    wanted.push((Finger::Action(c), p));
                }
                if c == Control::TwoFingerTouch {
                    wanted.push((Finger::Second, touch::SECOND_FINGER));
                }
            }
        }
        if let Some(p) = self.swipe.and_then(|s| s.point()) {
            wanted.push((Finger::Swipe, p));
        }
        if self.touch_mode {
            wanted.extend(self.zones.iter().map(|&z| (Finger::Zone(z), touch::zone_point(z))));
            if self.mouse_touch || self.mouse_click {
                wanted.push((Finger::Mouse, self.cursor));
            }
        }
        // A click without a finger presses in the middle.
        if wanted.is_empty() && self.held(Control::Button(button::TOUCH_PAD)) {
            wanted.push((Finger::Action(Control::Button(button::TOUCH_PAD)), touch::point(0.5, 0.5)));
        }
        self.slots.update(&wanted);
    }

    /// Overlay the held bindings onto a controller snapshot.
    pub fn compose(&self, pad: &HostPadState) -> HostPadState {
        let mut s = *pad;
        let mut neg = [false; 4];
        let mut pos = [false; 4];
        let mut click = self.touch_mode && self.mouse_click;
        for (b, _) in self.bindings.iter().zip(&self.pressed).filter(|(_, &p)| p) {
            match b.control {
                Control::Button(bit) => {
                    s.buttons |= bit;
                    if bit == button::L2 {
                        s.l2 = 255;
                    }
                    if bit == button::R2 {
                        s.r2 = 255;
                    }
                }
                Control::LeftStickLeft => neg[0] = true,
                Control::LeftStickRight => pos[0] = true,
                Control::LeftStickUp => neg[1] = true,
                Control::LeftStickDown => pos[1] = true,
                Control::RightStickLeft => neg[2] = true,
                Control::RightStickRight => pos[2] = true,
                Control::RightStickUp => neg[3] = true,
                Control::RightStickDown => pos[3] = true,
                Control::TouchLeft | Control::TouchRight => click = true,
                _ => {}
            }
        }
        let mut axes = [s.left_x, s.left_y, s.right_x, s.right_y];
        for i in 0..4 {
            if neg[i] || pos[i] {
                axes[i] = if neg[i] == pos[i] { 128 } else if neg[i] { 0 } else { 255 };
            }
        }
        if let (true, Some(stick)) = (self.mouse_look, self.look_stick) {
            let i = if stick == Stick::Left { 0 } else { 2 };
            axes[i] = self.mouse_stick[0];
            axes[i + 1] = self.mouse_stick[1];
        }
        [s.left_x, s.left_y, s.right_x, s.right_y] = axes;
        if self.slots.any() && s.touch.iter().all(Option::is_none) {
            s.touch = self.slots.points();
            if click {
                s.buttons |= button::TOUCH_PAD;
            }
        }
        // The simulated sensors stand in for a controller without them,
        // and take over from one while they move.
        if pad.motion.is_none() || self.motion.active() {
            s.motion = Some(self.motion.sample());
        }
        s.normalized()
    }
}

/// The same action on the other stick.
pub fn swapped(c: Control) -> Control {
    use Control::*;
    match c {
        LeftStickLeft => RightStickLeft,
        LeftStickRight => RightStickRight,
        LeftStickUp => RightStickUp,
        LeftStickDown => RightStickDown,
        RightStickLeft => LeftStickLeft,
        RightStickRight => LeftStickRight,
        RightStickUp => LeftStickUp,
        RightStickDown => LeftStickDown,
        c => c,
    }
}

/// Mouse-look: a motion delta over one poll period becomes a stick
/// deflection between half and full scale along the motion direction.
pub fn mouse_to_stick(dx: f64, dy: f64, sensitivity: f64) -> [u8; 2] {
    if dx == 0.0 && dy == 0.0 {
        return [128, 128];
    }
    let dist = dx.hypot(dy);
    let scale = (dist * sensitivity + 16.0).clamp(64.0, 128.0) / dist;
    let axis = |d: f64| (128.0 + (d * scale).round()).clamp(0.0, 255.0) as u8;
    [axis(dx), axis(dy)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::default_bindings;
    use ps5hle::pad::PadSample;

    fn kbm() -> KbmState {
        KbmState::new(default_bindings(), 1.0)
    }

    #[test]
    fn keys_drive_buttons_and_sticks() {
        let mut k = kbm();
        k.key(KeyCode::Space, true, false);
        k.key(KeyCode::KeyA, true, false);
        k.key(KeyCode::KeyI, true, false);
        let s = k.compose(&HostPadState::NEUTRAL);
        assert_eq!(s.buttons, button::CROSS);
        assert_eq!((s.left_x, s.left_y, s.right_x, s.right_y), (0, 128, 128, 0));
        // Opposite directions cancel.
        k.key(KeyCode::KeyD, true, false);
        assert_eq!(k.compose(&HostPadState::NEUTRAL).left_x, 128);
        // Release.
        k.key(KeyCode::Space, false, false);
        k.key(KeyCode::KeyA, false, false);
        let s = k.compose(&HostPadState::NEUTRAL);
        assert_eq!((s.buttons, s.left_x), (0, 255));
    }

    #[test]
    fn mouse_trigger_is_analog_full() {
        let mut k = kbm();
        k.mouse_button(MouseButton::Left, true);
        let s = k.compose(&HostPadState::NEUTRAL);
        assert_eq!(s.buttons, button::R2);
        assert_eq!(s.r2, 255);
        k.mouse_button(MouseButton::Right, true);
        assert_eq!(k.compose(&HostPadState::NEUTRAL).l2, 255, "aim on the right button");
    }

    #[test]
    fn keyboard_overlays_controller() {
        let mut k = kbm();
        let pad = HostPadState { buttons: button::CIRCLE, left_x: 200, l2: 40, ..HostPadState::NEUTRAL };
        let s = k.compose(&pad);
        assert_eq!((s.buttons, s.left_x, s.l2), (button::CIRCLE | button::L2, 200, 40));
        k.key(KeyCode::KeyA, true, false);
        k.key(KeyCode::KeyQ, true, false);
        let s = k.compose(&pad);
        assert_eq!(s.buttons, button::CIRCLE | button::L1 | button::L2);
        assert_eq!(s.left_x, 0);
    }

    #[test]
    fn touch_keys_press_the_touch_pad() {
        let mut k = kbm();
        k.key(KeyCode::Tab, true, false);
        let s = k.compose(&HostPadState::NEUTRAL);
        assert_eq!(s.buttons, button::TOUCH_PAD);
        assert_eq!(s.touch, [Some(HostTouchPoint { x: 0.5, y: 0.5 }), None], "a click has a finger in the middle");
        k.key(KeyCode::Tab, false, false);
        assert_eq!(k.compose(&HostPadState::NEUTRAL).touch, [None, None]);
        // Click halves (unbound by default) click where they touch.
        let mut k = KbmState::new(
            vec![Binding { source: Source::Key(KeyCode::KeyZ), control: Control::TouchRight }],
            1.0,
        );
        k.key(KeyCode::KeyZ, true, false);
        let s = k.compose(&HostPadState::NEUTRAL);
        assert_eq!((s.buttons, s.touch[0]), (button::TOUCH_PAD, Some(HostTouchPoint { x: 0.75, y: 0.5 })));
    }

    #[test]
    fn wheel_presses_briefly() {
        let mut k = kbm();
        let t0 = Instant::now();
        k.wheel(1.0, t0);
        assert_eq!(k.compose(&HostPadState::NEUTRAL).buttons, button::UP);
        k.wheel(-1.0, t0);
        assert_eq!(k.compose(&HostPadState::NEUTRAL).buttons, button::DOWN);
        k.tick(t0 + WHEEL_PRESS / 2);
        assert_eq!(k.compose(&HostPadState::NEUTRAL).buttons, button::DOWN);
        k.tick(t0 + WHEEL_PRESS);
        assert_eq!(k.compose(&HostPadState::NEUTRAL).buttons, 0);
    }

    #[test]
    fn toggles_and_repeat() {
        let mut k = kbm();
        assert!(k.key(KeyCode::F11, true, false).toggle_fullscreen);
        assert!(!k.key(KeyCode::F11, true, true).toggle_fullscreen);
        k.key(KeyCode::F11, false, false);
        let fx = k.key(KeyCode::F10, true, false);
        assert!(fx.toggle_perf_overlay && !fx.toggle_fullscreen);
        assert!(!k.key(KeyCode::F10, true, true).toggle_perf_overlay);
        k.key(KeyCode::F10, false, false);
        assert_eq!(k.key(KeyCode::F9, true, false).capture_mouse, Some(true));
        k.key(KeyCode::F9, false, false);
        assert!(k.mouse_look);
        let t0 = Instant::now();
        k.mouse_motion(30.0, 0.0);
        k.tick(t0);
        let s = k.compose(&HostPadState::NEUTRAL);
        assert_eq!((s.right_x, s.right_y), (128 + 64, 128));
        assert_eq!(k.release_all().capture_mouse, Some(false));
        assert!(!k.mouse_look);
        assert_eq!(k.compose(&HostPadState::NEUTRAL).right_x, 128);
    }

    #[test]
    fn hotkeys_capture_and_release_the_mouse() {
        let mut c = KbmState::new(
            vec![Binding { source: Source::Key(KeyCode::Digit8), control: Control::VibrationIntensity }],
            1.0,
        );
        assert_eq!(c.key(KeyCode::Digit8, true, false).cycle, Some(ps5hle::pad_controller::Setting::VibrationIntensity));
        assert_eq!(c.key(KeyCode::Digit8, true, true).cycle, None);
        assert_eq!(c.key(KeyCode::Digit8, false, false).cycle, None);

        let mut k = kbm();
        assert_eq!(k.key(KeyCode::F9, true, false).capture_mouse, Some(true));
        k.key(KeyCode::F9, false, false);
        // Escape leaves capture and still presses Options.
        assert_eq!(k.key(KeyCode::Escape, true, false).capture_mouse, Some(false));
        assert_eq!(k.compose(&HostPadState::NEUTRAL).buttons, button::OPTIONS);
        k.key(KeyCode::Escape, false, false);
        // Without capture Escape is only Options.
        assert_eq!(k.key(KeyCode::Escape, true, false).capture_mouse, None);
        assert_eq!(k.key_for(Control::ToggleMouse), Some(KeyCode::F9));
    }

    #[test]
    fn inverted_mouse_look() {
        let mut k = kbm().with_invert(true, true);
        k.set_mouse_look(true);
        k.mouse_motion(30.0, -30.0);
        k.tick(Instant::now());
        let s = k.compose(&HostPadState::NEUTRAL);
        assert!(s.right_x < 128 && s.right_y > 128);
    }

    #[test]
    fn keyboard_only_layout_needs_no_mouse() {
        let mut k = KbmState::new(crate::mapping::default_keyboard_bindings(), 1.0);
        for (key, bit) in [(KeyCode::KeyR, button::SQUARE), (KeyCode::Digit3, button::R2), (KeyCode::KeyF, button::TRIANGLE)] {
            k.key(key, true, false);
            assert_eq!(k.compose(&HostPadState::NEUTRAL).buttons & bit, bit);
            k.key(key, false, false);
        }
        k.key(KeyCode::KeyL, true, false);
        k.key(KeyCode::KeyI, true, false);
        let s = k.compose(&HostPadState::NEUTRAL);
        assert_eq!((s.right_x, s.right_y), (255, 0));
        // Mouse input has no effect.
        k.mouse_button(MouseButton::Left, true);
        assert_eq!(k.compose(&HostPadState::NEUTRAL).buttons, 0);
    }

    #[test]
    fn bindings_follow_key_positions_not_characters() {
        // The window reports physical keys: with a Russian layout the key
        // right of X types 'с' and still arrives as KeyC, so it is Circle.
        let mut k = kbm();
        k.key(KeyCode::KeyC, true, false);
        assert_eq!(k.compose(&HostPadState::NEUTRAL).buttons, button::CIRCLE);
        k.key(KeyCode::KeyC, false, false);
        // File names are US position names: "C" binds that key, and a
        // character another layout produces there is not a key name.
        let cfg = crate::mapping::parse_config_onto(Default::default(), "[keyboard_bindings]\nTriangle = \"KEY:C\"\n").unwrap();
        let mut k = KbmState::new(cfg.keyboard_bindings.clone(), 1.0);
        k.key(KeyCode::KeyC, true, false);
        assert_eq!(k.compose(&HostPadState::NEUTRAL).buttons & button::TRIANGLE, button::TRIANGLE);
        assert_eq!(crate::mapping::key_from_name("\u{0441}"), None); // Cyrillic 'с'
    }

    #[test]
    fn stick_modes_route_keys_and_mouse() {
        use crate::mapping::StickMode;
        let look = |k: &mut KbmState| {
            k.set_mouse_look(true);
            k.mouse_motion(30.0, 0.0);
            k.tick(Instant::now());
        };
        // Default keyboard + mouse: WASD on the left stick, the mouse on the
        // right one, its keys (I/J/K/L) off.
        let mut cfg = Config { mode: crate::mapping::DeviceMode::KeyboardMouse, ..Config::default() };
        let mut k = KbmState::from_config(&cfg);
        k.key(KeyCode::KeyW, true, false);
        k.key(KeyCode::KeyI, true, false);
        look(&mut k);
        let s = k.compose(&HostPadState::NEUTRAL);
        assert_eq!((s.left_x, s.left_y, s.right_x, s.right_y), (128, 0, 192, 128));
        // Mouse on the left: it takes the left stick, the right one goes to
        // its keys.
        cfg.set_stick_mode(Stick::Left, StickMode::Mouse);
        assert_eq!(cfg.right_stick_mode, StickMode::Keys);
        let mut k = KbmState::from_config(&cfg);
        k.key(KeyCode::KeyW, true, false);
        k.key(KeyCode::KeyI, true, false);
        look(&mut k);
        let s = k.compose(&HostPadState::NEUTRAL);
        assert_eq!((s.left_x, s.left_y, s.right_x, s.right_y), (192, 128, 128, 0));
        // Keys on both: the mouse drives nothing.
        cfg.set_stick_mode(Stick::Left, StickMode::Keys);
        let mut k = KbmState::from_config(&cfg);
        k.key(KeyCode::KeyW, true, false);
        look(&mut k);
        let s = k.compose(&HostPadState::NEUTRAL);
        assert_eq!((s.left_x, s.left_y, s.right_x, s.right_y), (128, 0, 128, 128));
        // Off: the keys leave the stick to the controller.
        cfg.set_stick_mode(Stick::Left, StickMode::Off);
        let mut k = KbmState::from_config(&cfg);
        k.key(KeyCode::KeyW, true, false);
        let pad = HostPadState { left_y: 200, ..HostPadState::NEUTRAL };
        assert_eq!(k.compose(&pad).left_y, 200);
        // Keyboard only: a mouse stick uses its keys (I/J/K/L on the right).
        let mut cfg = Config { mode: crate::mapping::DeviceMode::Keyboard, ..Config::default() };
        assert_eq!(cfg.mouse_stick(), None);
        let mut k = KbmState::from_config(&cfg);
        k.key(KeyCode::KeyI, true, false);
        assert_eq!(k.compose(&HostPadState::NEUTRAL).right_y, 0);
        cfg.right_stick_mode = StickMode::Off;
        let mut k = KbmState::from_config(&cfg);
        k.key(KeyCode::KeyI, true, false);
        assert_eq!(k.compose(&HostPadState::NEUTRAL).right_y, 128);
    }

    #[test]
    fn mouse_stick_curve() {
        assert_eq!(mouse_to_stick(0.0, 0.0, 1.0), [128, 128]);
        assert_eq!(mouse_to_stick(1.0, 0.0, 1.0), [192, 128]); // at least half deflection
        assert_eq!(mouse_to_stick(0.0, -500.0, 1.0), [128, 0]); // capped at full
        assert_eq!(mouse_to_stick(80.0, 0.0, 1.0), [224, 128]);
    }

    #[test]
    fn synthetic_input_encodes_to_pad_data() {
        // End to end: key presses -> pad state -> ScePadData bytes.
        let mut k = kbm();
        k.key(KeyCode::Enter, true, false);
        k.key(KeyCode::KeyW, true, false);
        k.mouse_button(MouseButton::Left, true);
        let s = k.compose(&HostPadState::NEUTRAL);
        let sample = PadSample {
            buttons: s.buttons,
            sticks: [s.left_x, s.left_y, s.right_x, s.right_y],
            l2: s.l2,
            r2: s.r2,
            ..PadSample::NEUTRAL
        };
        let d = ps5hle::pad::serialize_pad_data(&sample);
        assert_eq!(u32::from_le_bytes(d[0..4].try_into().unwrap()), button::CROSS | button::R2);
        assert_eq!(&d[4..10], &[128, 0, 128, 128, 0, 255]);
    }

    #[test]
    fn swapped_sticks_trade_keys_and_mouse() {
        let cfg = Config { mode: crate::mapping::DeviceMode::KeyboardMouse, swap_sticks: true, ..Config::default() };
        let mut k = KbmState::from_config(&cfg);
        k.key(KeyCode::KeyW, true, false);
        k.set_mouse_look(true);
        k.mouse_motion(30.0, 0.0);
        k.tick(Instant::now());
        let s = k.compose(&HostPadState::NEUTRAL);
        assert_eq!((s.left_x, s.left_y, s.right_x, s.right_y), (192, 128, 128, 0), "mouse on the left, WASD on the right");
        assert_eq!(swapped(swapped(Control::LeftStickUp)), Control::LeftStickUp);
    }

    #[test]
    fn numpad_tilts_the_controller() {
        let mut k = kbm();
        let t0 = Instant::now();
        k.tick(t0);
        // At rest without a controller: a flat, still controller.
        let rest = k.compose(&HostPadState::NEUTRAL).motion.unwrap();
        assert_eq!(rest, crate::motion::REST);
        k.key(KeyCode::Numpad8, true, false);
        k.tick(t0 + Duration::from_millis(20));
        let m = k.compose(&HostPadState::NEUTRAL).motion.unwrap();
        assert!(m.angular_velocity[0] < 0.0, "tilting forward");
        for i in 2..200 {
            k.tick(t0 + Duration::from_millis(10 * i));
        }
        let m = k.compose(&HostPadState::NEUTRAL).motion.unwrap();
        assert!(m.accel[2] > 3.0 && m.angular_velocity == [0.0; 3], "held at the tilt: {m:?}");
        // A controller's own sensors win while the keys are idle.
        k.key(KeyCode::Numpad8, false, false);
        for i in 200..400 {
            k.tick(t0 + Duration::from_millis(10 * i));
        }
        let own = ps5hle::pad::HostMotion { accel: [1.0, 2.0, 3.0], angular_velocity: [0.5, 0.0, 0.0] };
        let pad = HostPadState { motion: Some(own), ..HostPadState::NEUTRAL };
        assert_eq!(k.compose(&pad).motion, Some(own));
        // Shake: a burst, then rest.
        k.key(KeyCode::Numpad5, true, false);
        k.tick(t0 + Duration::from_millis(4010));
        assert_ne!(k.compose(&HostPadState::NEUTRAL).motion, Some(crate::motion::REST));
        for i in 402..470 {
            k.tick(t0 + Duration::from_millis(10 * i));
        }
        assert_eq!(k.compose(&HostPadState::NEUTRAL).motion, Some(crate::motion::REST));
    }

    #[test]
    fn middle_button_held_turns_the_controller_with_the_mouse() {
        let mut k = kbm();
        let t0 = Instant::now();
        k.tick(t0);
        let fx = k.mouse_button(MouseButton::Middle, true);
        assert_eq!(fx.hold_capture, Some(true));
        assert!(k.mouse_motion_held() && k.captured());
        k.mouse_motion(200.0, 0.0);
        k.tick(t0 + Duration::from_millis(4));
        let m = k.compose(&HostPadState::NEUTRAL).motion.unwrap();
        assert!(m.angular_velocity[1] < 0.0, "mouse right turns the controller right");
        let fx = k.mouse_button(MouseButton::Middle, false);
        assert_eq!(fx.hold_capture, Some(false));
        // Released: the mouse no longer turns it.
        k.mouse_motion(200.0, 0.0);
        k.tick(t0 + Duration::from_millis(8));
        assert_eq!(k.compose(&HostPadState::NEUTRAL).motion.unwrap().angular_velocity, [0.0; 3]);
        // With mouse-look on, holding it moves the gyro, not the stick.
        k.set_mouse_look(true);
        assert_eq!(k.mouse_button(MouseButton::Middle, true).hold_capture, None);
        k.mouse_motion(30.0, 0.0);
        k.tick(t0 + Duration::from_millis(100));
        assert_eq!(k.compose(&HostPadState::NEUTRAL).right_x, 128);
    }

    #[test]
    fn touchpad_mode_numpad_zones_and_mouse_finger() {
        let mut k = kbm();
        let fx = k.key(KeyCode::Numpad0, true, false);
        k.key(KeyCode::Numpad0, false, false);
        assert_eq!(fx.touch_mode, Some(true));
        // Numpad 7 then 3: two fingers, top left and bottom right, no click.
        k.key(KeyCode::Numpad7, true, false);
        let s = k.compose(&HostPadState::NEUTRAL);
        assert_eq!(s.touch, [Some(touch::zone_point(7)), None]);
        assert_eq!(s.buttons & button::TOUCH_PAD, 0);
        assert_eq!(s.motion.unwrap(), crate::motion::REST, "no tilt in touch pad mode");
        k.key(KeyCode::Numpad3, true, false);
        assert_eq!(k.compose(&HostPadState::NEUTRAL).touch, [Some(touch::zone_point(7)), Some(touch::zone_point(3))]);
        // The first finger lifts, the second keeps its slot.
        k.key(KeyCode::Numpad7, false, false);
        assert_eq!(k.compose(&HostPadState::NEUTRAL).touch, [None, Some(touch::zone_point(3))]);
        // Tab clicks with the finger down.
        k.key(KeyCode::Tab, true, false);
        let s = k.compose(&HostPadState::NEUTRAL);
        assert_eq!((s.buttons & button::TOUCH_PAD, s.touch), (button::TOUCH_PAD, [None, Some(touch::zone_point(3))]));
        k.key(KeyCode::Tab, false, false);
        k.key(KeyCode::Numpad3, false, false);
        // The mouse: the left button touches at the cursor, the right one
        // clicks; neither is L2/R2 here.
        k.mouse_motion(80.0, 0.0);
        k.mouse_button(MouseButton::Left, true);
        let s = k.compose(&HostPadState::NEUTRAL);
        let p = s.touch[0].unwrap();
        assert!((p.x - 0.6).abs() < 1e-5 && (p.y - 0.5).abs() < 1e-5);
        assert_eq!(s.buttons, 0);
        k.mouse_button(MouseButton::Right, true);
        assert_eq!(k.compose(&HostPadState::NEUTRAL).buttons, button::TOUCH_PAD);
        k.mouse_button(MouseButton::Right, false);
        k.mouse_button(MouseButton::Left, false);
        // Leaving: the numpad tilts again.
        assert_eq!(k.key(KeyCode::Numpad0, true, false).touch_mode, Some(false));
        k.key(KeyCode::Numpad0, false, false);
        k.key(KeyCode::Numpad7, true, false);
        assert_eq!(k.compose(&HostPadState::NEUTRAL).touch, [None, None]);
    }

    #[test]
    fn gestures_taps_and_two_fingers() {
        let mut k = kbm();
        let t0 = Instant::now();
        k.tick(t0);
        // Swipe right (Page Down): a finger slides left to right and lifts.
        k.key(KeyCode::PageDown, true, false);
        k.key(KeyCode::PageDown, false, false);
        let first = k.compose(&HostPadState::NEUTRAL).touch[0].unwrap();
        k.tick(t0 + Duration::from_millis(100));
        let mid = k.compose(&HostPadState::NEUTRAL).touch[0].unwrap();
        k.tick(t0 + Duration::from_millis(210));
        let end = k.compose(&HostPadState::NEUTRAL).touch[0].unwrap();
        assert!(first.x < mid.x && mid.x < end.x && end.x > 0.75);
        k.tick(t0 + Duration::from_millis(300));
        assert_eq!(k.compose(&HostPadState::NEUTRAL).touch, [None, None]);
        // Tap: a finger without a click.
        k.key(KeyCode::PageUp, true, false);
        let s = k.compose(&HostPadState::NEUTRAL);
        assert_eq!((s.buttons, s.touch[0].is_some()), (0, true));
        k.key(KeyCode::PageUp, false, false);
        // Two fingers.
        k.key(KeyCode::Insert, true, false);
        let s = k.compose(&HostPadState::NEUTRAL);
        assert!(s.touch[0].is_some() && s.touch[1].is_some());
        k.key(KeyCode::Insert, false, false);
        assert_eq!(k.compose(&HostPadState::NEUTRAL).touch, [None, None]);
    }

    #[test]
    fn create_and_mute_are_hotkeys() {
        let mut k = kbm();
        // The controls overlay shows while its key is held.
        assert_eq!(k.key(KeyCode::F2, true, false).show_controls, Some(true));
        assert_eq!(k.key(KeyCode::F2, true, true).show_controls, None);
        assert_eq!(k.key(KeyCode::F2, false, false).show_controls, Some(false));
        k.key(KeyCode::F2, true, false);
        assert_eq!(k.release_all().show_controls, Some(false));
        assert!(k.key(KeyCode::F12, true, false).screenshot);
        assert!(!k.key(KeyCode::F12, true, true).screenshot);
        assert!(k.key(KeyCode::F8, true, false).toggle_mute);
        assert_eq!(k.compose(&HostPadState::NEUTRAL).buttons, 0, "games never see them");
    }
}
