// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Controller side of the in-game menu (`ps5gpu::menu`): while it is open
//! the guest sees neutral pads, the input thread turns player 1's
//! controller into navigation steps ([`Nav`], with auto-repeat for
//! directions) and publishes the connected controllers and their raw
//! inputs (device choice and binding capture in the menu). The PS button
//! (`Guide`, when no pad button is bound to it) opens and closes the menu.

use crate::mapping::PadInput;
use crate::sdl::DeviceInfo;
use ps5hle::pad::{button, HostPadState};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// The controller button that opens and closes the menu.
pub const MENU_BUTTON: PadInput = PadInput(5);

/// One navigation step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nav {
    Up,
    Down,
    Left,
    Right,
    /// Cross: activate the focused item.
    Accept,
    /// Circle: back / cancel.
    Back,
    /// L1 / R1: previous / next section.
    PrevTab,
    NextTab,
    /// Options: close the menu.
    Close,
}

/// Delay before a held direction repeats, and the repeat period.
pub const FIRST_REPEAT: Duration = Duration::from_millis(350);
pub const REPEAT: Duration = Duration::from_millis(90);

/// Stick deflection (from centre, of 128) that counts as a direction.
const STICK_THRESHOLD: i32 = 64;

const INPUTS: [Nav; 9] = [Nav::Up, Nav::Down, Nav::Left, Nav::Right, Nav::Accept, Nav::Back, Nav::PrevTab, Nav::NextTab, Nav::Close];

fn held(s: &HostPadState) -> [bool; 9] {
    let b = |bit| s.buttons & bit != 0;
    let (x, y) = (s.left_x as i32 - 128, s.left_y as i32 - 128);
    [
        b(button::UP) || y < -STICK_THRESHOLD,
        b(button::DOWN) || y > STICK_THRESHOLD,
        b(button::LEFT) || x < -STICK_THRESHOLD,
        b(button::RIGHT) || x > STICK_THRESHOLD,
        b(button::CROSS),
        b(button::CIRCLE),
        b(button::L1),
        b(button::R1),
        b(button::OPTIONS),
    ]
}

/// Edge detection and auto-repeat of the navigation inputs.
#[derive(Clone, Debug, Default)]
pub struct NavState {
    held: [bool; 9],
    next_repeat: [Option<Instant>; 9],
}

impl NavState {
    /// Start from `s` without reporting what is already held (the button
    /// that opened the menu).
    pub fn reset(&mut self, s: &HostPadState) {
        self.held = held(s);
        self.next_repeat = [None; 9];
    }

    /// Steps for the controller state `s` at `now`.
    pub fn update(&mut self, s: &HostPadState, now: Instant, out: &mut Vec<Nav>) {
        let h = held(s);
        for i in 0..INPUTS.len() {
            let direction = i < 4;
            if h[i] && !self.held[i] {
                out.push(INPUTS[i]);
                self.next_repeat[i] = direction.then_some(now + FIRST_REPEAT);
            } else if h[i] {
                if let Some(t) = self.next_repeat[i].filter(|t| now >= *t) {
                    out.push(INPUTS[i]);
                    self.next_repeat[i] = Some(now.checked_sub(REPEAT).map_or(t, |late| t.max(late)) + REPEAT);
                }
            } else {
                self.next_repeat[i] = None;
            }
        }
        self.held = h;
    }
}

/// What the menu reads from the input thread.
#[derive(Clone, Debug, Default)]
pub struct Feed {
    /// Connected controllers (empty without SDL).
    pub devices: Vec<DeviceInfo>,
    /// Raw inputs held per device id.
    pub pressed: Vec<(u32, Vec<PadInput>)>,
    /// Why controllers are unavailable, if they are.
    pub error: Option<String>,
    nav: Vec<Nav>,
}

static FEED: Mutex<Feed> = Mutex::new(Feed { devices: Vec::new(), pressed: Vec::new(), error: None, nav: Vec::new() });

fn feed() -> std::sync::MutexGuard<'static, Feed> {
    FEED.lock().unwrap_or_else(|e| e.into_inner())
}

/// Navigation steps since the last call.
pub fn take_nav() -> Vec<Nav> {
    std::mem::take(&mut feed().nav)
}

/// The controllers and their held inputs (without the queued steps).
pub fn controllers() -> Feed {
    let f = feed();
    Feed { devices: f.devices.clone(), pressed: f.pressed.clone(), error: f.error.clone(), nav: Vec::new() }
}

/// Publish the controllers and queue navigation steps (the input thread).
pub fn publish(devices: Vec<DeviceInfo>, pressed: Vec<(u32, Vec<PadInput>)>, nav: &[Nav]) {
    let mut f = feed();
    f.devices = devices;
    f.pressed = pressed;
    // A menu that is not reading (closed) does not pile steps up.
    if f.nav.len() < 64 {
        f.nav.extend_from_slice(nav);
    }
}

pub(crate) fn set_error(e: Option<String>) {
    feed().error = e;
}

pub(crate) fn clear() {
    let mut f = feed();
    f.pressed.clear();
    f.nav.clear();
}

/// Whether the PS button toggles the menu (not bound to a pad button).
pub fn menu_button_free(bindings: &[crate::mapping::PadBinding]) -> bool {
    !bindings.iter().any(|b| b.input == MENU_BUTTON)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pad(buttons: u32) -> HostPadState {
        HostPadState { buttons, ..HostPadState::NEUTRAL }
    }

    #[test]
    fn presses_are_edges_and_directions_repeat() {
        let t = Instant::now();
        let mut n = NavState::default();
        let mut out = Vec::new();
        n.update(&pad(button::CROSS), t, &mut out);
        n.update(&pad(button::CROSS), t + Duration::from_secs(2), &mut out);
        assert_eq!(out, [Nav::Accept], "buttons do not repeat");
        out.clear();
        n.update(&pad(button::DOWN), t, &mut out);
        assert_eq!(out, [Nav::Down]);
        n.update(&pad(button::DOWN), t + FIRST_REPEAT - Duration::from_millis(1), &mut out);
        assert_eq!(out.len(), 1);
        n.update(&pad(button::DOWN), t + FIRST_REPEAT, &mut out);
        n.update(&pad(button::DOWN), t + FIRST_REPEAT + REPEAT, &mut out);
        assert_eq!(out, [Nav::Down; 3]);
        n.update(&pad(0), t + FIRST_REPEAT + REPEAT * 2, &mut out);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn stick_navigates_and_reset_ignores_held_buttons() {
        let t = Instant::now();
        let mut n = NavState::default();
        let mut out = Vec::new();
        let left = HostPadState { left_x: 0, ..HostPadState::NEUTRAL };
        n.update(&left, t, &mut out);
        let slight = HostPadState { left_y: 160, ..HostPadState::NEUTRAL };
        n.update(&slight, t, &mut out);
        assert_eq!(out, [Nav::Left], "a small deflection is no step");
        out.clear();
        n.reset(&pad(button::OPTIONS | button::R1));
        n.update(&pad(button::OPTIONS | button::R1 | button::CIRCLE), t, &mut out);
        assert_eq!(out, [Nav::Back]);
        n.update(&pad(button::L1), t, &mut out);
        assert_eq!(out, [Nav::Back, Nav::PrevTab]);
    }

    #[test]
    fn menu_button_only_when_unbound() {
        use crate::mapping::{default_pad_bindings, PadBinding};
        let mut b = default_pad_bindings();
        assert!(menu_button_free(&b));
        b.push(PadBinding { input: MENU_BUTTON, bit: button::TOUCH_PAD });
        assert!(!menu_button_free(&b));
    }

    #[test]
    fn feed_queues_steps() {
        publish(Vec::new(), vec![(7, vec![PadInput(0)])], &[Nav::Up, Nav::Accept]);
        assert_eq!(controllers().pressed, vec![(7, vec![PadInput(0)])]);
        assert_eq!(take_nav(), [Nav::Up, Nav::Accept]);
        assert!(take_nav().is_empty());
        clear();
        assert!(controllers().pressed.is_empty());
    }
}
