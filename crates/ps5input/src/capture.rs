// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! "Press a key / button" binding capture used by the settings UI.
//!
//! A capture starts disarmed and arms once every input is released, so
//! the click that started it (or a key still held) is not captured. The
//! first press of a suitable input while armed completes it; inputs that
//! cannot drive the target are rejected with a reason and the capture
//! keeps waiting. Captures expire after [`CAPTURE_TIMEOUT`].

use crate::mapping::{validate_source, Control, PadInput, Source};
use ps5gpu::input::{KeyCode, MouseButton};
use std::time::{Duration, Instant};

pub const CAPTURE_TIMEOUT: Duration = Duration::from_secs(6);

/// What is being rebound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureTarget {
    /// Keyboard + mouse set.
    KeyboardMouse(Control),
    /// Keyboard-only set.
    Keyboard(Control),
    /// Controller layout: pad button bit.
    Gamepad(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureInput {
    Key(KeyCode),
    Mouse(MouseButton),
    /// +1 up, -1 down.
    Wheel(i8),
    Pad(PadInput),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Captured {
    Source(Source),
    Pad(PadInput),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CaptureState {
    /// Waiting for input.
    Pending,
    /// The last input cannot be used; still waiting.
    Rejected(String),
    Done(Captured),
    TimedOut,
}

#[derive(Clone, Debug)]
pub struct Capture {
    pub target: CaptureTarget,
    armed: bool,
    deadline: Instant,
    pub state: CaptureState,
}

impl Capture {
    pub fn new(target: CaptureTarget, now: Instant) -> Self {
        Self { target, armed: false, deadline: now + CAPTURE_TIMEOUT, state: CaptureState::Pending }
    }

    pub fn is_finished(&self) -> bool {
        matches!(self.state, CaptureState::Done(_) | CaptureState::TimedOut)
    }

    pub fn remaining(&self, now: Instant) -> Duration {
        self.deadline.saturating_duration_since(now)
    }

    /// Report whether any input is currently held; arms on "nothing held".
    pub fn set_idle(&mut self, nothing_held: bool) {
        if nothing_held {
            self.armed = true;
        }
    }

    pub fn tick(&mut self, now: Instant) {
        if !self.is_finished() && now >= self.deadline {
            self.state = CaptureState::TimedOut;
        }
    }

    /// Feed one press (releases are not passed in); ignored until armed.
    pub fn press(&mut self, input: CaptureInput, now: Instant) -> &CaptureState {
        self.tick(now);
        if self.is_finished() || !self.armed {
            return &self.state;
        }
        let result = match (self.target, input) {
            (CaptureTarget::Gamepad(_), CaptureInput::Pad(p)) => Ok(Captured::Pad(p)),
            (CaptureTarget::Gamepad(_), _) => Err("press a controller button".to_string()),
            (_, CaptureInput::Pad(_)) => Err("press a key or mouse button".to_string()),
            (CaptureTarget::KeyboardMouse(c) | CaptureTarget::Keyboard(c), i) => {
                let keyboard_only = matches!(self.target, CaptureTarget::Keyboard(_));
                let src = match i {
                    CaptureInput::Key(k) => Source::Key(k),
                    CaptureInput::Mouse(b) => Source::Mouse(b),
                    CaptureInput::Wheel(d) => Source::Wheel(d.signum()),
                    CaptureInput::Pad(_) => unreachable!(),
                };
                validate_source(c, src, keyboard_only).map(|_| Captured::Source(src)).map_err(|e| e.0)
            }
        };
        self.state = match result {
            Ok(c) => CaptureState::Done(c),
            Err(e) => CaptureState::Rejected(e),
        };
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::Control;
    use ps5hle::pad::button;

    #[test]
    fn arms_after_release_and_captures_first_press() {
        let t = Instant::now();
        let mut c = Capture::new(CaptureTarget::KeyboardMouse(Control::Button(button::CROSS)), t);
        // The click that opened the capture is ignored until released.
        c.press(CaptureInput::Mouse(MouseButton::Left), t);
        assert_eq!(c.state, CaptureState::Pending);
        c.set_idle(false);
        c.press(CaptureInput::Key(KeyCode::KeyX), t);
        assert_eq!(c.state, CaptureState::Pending);
        c.set_idle(true);
        c.press(CaptureInput::Key(KeyCode::KeyX), t);
        assert_eq!(c.state, CaptureState::Done(Captured::Source(Source::Key(KeyCode::KeyX))));
        assert!(c.is_finished());
        // Later presses do not change the result.
        c.press(CaptureInput::Key(KeyCode::KeyY), t);
        assert_eq!(c.state, CaptureState::Done(Captured::Source(Source::Key(KeyCode::KeyX))));
    }

    #[test]
    fn rejects_unsuitable_inputs() {
        let t = Instant::now();
        let mut c = Capture::new(CaptureTarget::KeyboardMouse(Control::ToggleFullscreen), t);
        c.set_idle(true);
        assert!(matches!(c.press(CaptureInput::Mouse(MouseButton::Left), t), CaptureState::Rejected(_)));
        assert!(matches!(c.press(CaptureInput::Pad(PadInput(0)), t), CaptureState::Rejected(_)));
        assert_eq!(c.press(CaptureInput::Key(KeyCode::F12), t), &CaptureState::Done(Captured::Source(Source::Key(KeyCode::F12))));

        let mut c = Capture::new(CaptureTarget::Keyboard(Control::Button(button::SQUARE)), t);
        c.set_idle(true);
        assert!(matches!(c.press(CaptureInput::Mouse(MouseButton::Left), t), CaptureState::Rejected(_)));
        assert!(matches!(c.press(CaptureInput::Wheel(1), t), CaptureState::Rejected(_)));

        let mut c = Capture::new(CaptureTarget::KeyboardMouse(Control::LeftStickUp), t);
        c.set_idle(true);
        assert!(matches!(c.press(CaptureInput::Wheel(1), t), CaptureState::Rejected(_)));
        let mut c = Capture::new(CaptureTarget::KeyboardMouse(Control::Button(button::UP)), t);
        c.set_idle(true);
        assert_eq!(c.press(CaptureInput::Wheel(3), t), &CaptureState::Done(Captured::Source(Source::Wheel(1))));
    }

    #[test]
    fn gamepad_target_takes_controller_buttons_only() {
        let t = Instant::now();
        let mut c = Capture::new(CaptureTarget::Gamepad(button::CROSS), t);
        c.set_idle(true);
        assert!(matches!(c.press(CaptureInput::Key(KeyCode::KeyA), t), CaptureState::Rejected(_)));
        assert_eq!(c.press(CaptureInput::Pad(PadInput::RIGHT_TRIGGER), t), &CaptureState::Done(Captured::Pad(PadInput::RIGHT_TRIGGER)));
    }

    #[test]
    fn times_out() {
        let t = Instant::now();
        let mut c = Capture::new(CaptureTarget::Gamepad(button::CROSS), t);
        c.set_idle(true);
        c.tick(t + CAPTURE_TIMEOUT);
        assert_eq!(c.state, CaptureState::TimedOut);
        c.press(CaptureInput::Pad(PadInput(0)), t + CAPTURE_TIMEOUT);
        assert_eq!(c.state, CaptureState::TimedOut);
    }
}
