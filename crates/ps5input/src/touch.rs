// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Touch pad fingers from the keyboard and mouse: where each touch action
//! puts its finger, swipes over time, the numpad zones of touch pad mode,
//! and the two finger slots of the pad.
//!
//! The pad reports up to two fingers; a finger keeps its slot (and so its
//! touch id, which the pad assigns on each new touch of a slot) from the
//! moment it lands until it lifts, whatever the other finger does. A third
//! finger waits for a free slot. Coordinates are 0..1 across the pad, y
//! downwards.

use crate::mapping::Control;
use ps5gpu::input::KeyCode;
use ps5hle::pad::HostTouchPoint;

/// How long a swipe slides, and how long its finger rests at the end
/// before lifting.
pub const SWIPE_TIME: f32 = 0.2;
pub const SWIPE_HOLD: f32 = 0.03;
/// Touch pad aspect (1920 x 943 touch units): mouse counts per pad width
/// at sensitivity 1, and the matching height.
pub const MOUSE_COUNTS_PER_WIDTH: f32 = 800.0;
const PAD_ASPECT: f32 = 943.0 / 1920.0;

/// What holds a finger on the pad.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finger {
    /// A touch action key (tap, click, the second finger of a two-finger
    /// touch is [`Finger::Second`]).
    Action(Control),
    Second,
    Swipe,
    /// A numpad zone of touch pad mode (1..=9).
    Zone(u8),
    /// The mouse in touch pad mode.
    Mouse,
}

pub fn point(x: f32, y: f32) -> HostTouchPoint {
    HostTouchPoint { x, y }
}

/// Where a touch action's finger rests (the two-finger touch: its first
/// finger).
pub fn action_point(c: Control) -> Option<HostTouchPoint> {
    Some(match c {
        Control::TouchLeft | Control::TapLeft => point(0.25, 0.5),
        Control::TouchRight | Control::TapRight => point(0.75, 0.5),
        Control::TapCenter => point(0.5, 0.5),
        Control::TwoFingerTouch => point(0.35, 0.5),
        _ => return None,
    })
}

/// The second finger of a two-finger touch.
pub const SECOND_FINGER: HostTouchPoint = HostTouchPoint { x: 0.65, y: 0.5 };

/// The zone of a numpad digit key: 7 8 9 top row, 4 5 6 middle, 1 2 3
/// bottom, like the keys.
pub fn zone_of(k: KeyCode) -> Option<u8> {
    use KeyCode::*;
    Some(match k {
        Numpad1 => 1,
        Numpad2 => 2,
        Numpad3 => 3,
        Numpad4 => 4,
        Numpad5 => 5,
        Numpad6 => 6,
        Numpad7 => 7,
        Numpad8 => 8,
        Numpad9 => 9,
        _ => return None,
    })
}

pub fn zone_point(zone: u8) -> HostTouchPoint {
    let i = (zone.clamp(1, 9) - 1) as f32;
    let col = i % 3.0;
    let row = 2.0 - (i / 3.0).floor();
    point((col + 0.5) / 3.0, (row + 0.5) / 3.0)
}

/// A swipe in progress.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Swipe {
    pub dir: Control,
    pub t: f32,
}

impl Swipe {
    /// The finger's position, `None` once it lifted.
    pub fn point(&self) -> Option<HostTouchPoint> {
        if self.t >= SWIPE_TIME + SWIPE_HOLD {
            return None;
        }
        let f = (self.t / SWIPE_TIME).clamp(0.0, 1.0);
        // Smooth start and stop.
        let f = f * f * (3.0 - 2.0 * f);
        let (a, b) = (0.2, 0.8);
        let along = a + (b - a) * f;
        let back = b - (b - a) * f;
        Some(match self.dir {
            Control::SwipeRight => point(along, 0.5),
            Control::SwipeLeft => point(back, 0.5),
            Control::SwipeDown => point(0.5, along),
            Control::SwipeUp => point(0.5, back),
            _ => return None,
        })
    }
}

/// Move a touch pad mode cursor by mouse counts.
pub fn move_cursor(c: HostTouchPoint, dx: f64, dy: f64, sensitivity: f32) -> HostTouchPoint {
    let k = sensitivity / MOUSE_COUNTS_PER_WIDTH;
    point((c.x + dx as f32 * k).clamp(0.0, 1.0), (c.y + dy as f32 * k / PAD_ASPECT).clamp(0.0, 1.0))
}

/// The pad's two finger slots.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Slots {
    slots: [Option<(Finger, HostTouchPoint)>; 2],
}

impl Slots {
    /// Fingers wanted now (in priority order): fingers already down keep
    /// their slots and move, lifted ones free theirs, new ones take free
    /// slots.
    pub fn update(&mut self, wanted: &[(Finger, HostTouchPoint)]) {
        for s in self.slots.iter_mut() {
            *s = s.and_then(|(f, _)| wanted.iter().find(|w| w.0 == f).copied());
        }
        for w in wanted {
            if self.slots.iter().flatten().any(|s| s.0 == w.0) {
                continue;
            }
            if let Some(free) = self.slots.iter_mut().find(|s| s.is_none()) {
                *free = Some(*w);
            }
        }
    }

    pub fn points(&self) -> [Option<HostTouchPoint>; 2] {
        self.slots.map(|s| s.map(|(_, p)| p))
    }

    pub fn any(&self) -> bool {
        self.slots.iter().any(Option::is_some)
    }

    pub fn clear(&mut self) {
        self.slots = [None, None];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zones_follow_the_numpad() {
        assert_eq!(zone_of(KeyCode::Numpad7), Some(7));
        assert_eq!(zone_of(KeyCode::Numpad0), None);
        let p7 = zone_point(7);
        let p3 = zone_point(3);
        let p5 = zone_point(5);
        assert!(p7.x < 0.2 && p7.y < 0.2, "7: top left");
        assert!(p3.x > 0.8 && p3.y > 0.8, "3: bottom right");
        assert!((p5.x - 0.5).abs() < 1e-6 && (p5.y - 0.5).abs() < 1e-6);
        assert!(zone_point(8).y < zone_point(2).y);
    }

    #[test]
    fn fingers_keep_their_slots() {
        let mut s = Slots::default();
        let a = (Finger::Zone(7), zone_point(7));
        let b = (Finger::Zone(3), zone_point(3));
        s.update(&[a]);
        assert_eq!(s.points(), [Some(a.1), None]);
        s.update(&[a, b]);
        assert_eq!(s.points(), [Some(a.1), Some(b.1)]);
        // The first finger lifts: the second stays in slot 1.
        s.update(&[b]);
        assert_eq!(s.points(), [None, Some(b.1)]);
        // A new finger takes the free slot; a third waits.
        let c = (Finger::Mouse, point(0.1, 0.1));
        let d = (Finger::Zone(5), zone_point(5));
        s.update(&[b, c, d]);
        assert_eq!(s.points(), [Some(c.1), Some(b.1)]);
        // Moving keeps the slot.
        let c2 = (Finger::Mouse, point(0.3, 0.1));
        s.update(&[b, c2]);
        assert_eq!(s.points(), [Some(c2.1), Some(b.1)]);
        s.update(&[]);
        assert!(!s.any());
    }

    #[test]
    fn swipes_slide_and_lift() {
        let mut sw = Swipe { dir: Control::SwipeRight, t: 0.0 };
        let start = sw.point().unwrap();
        sw.t = SWIPE_TIME;
        let end = sw.point().unwrap();
        assert!(start.x < 0.25 && end.x > 0.75 && start.y == end.y);
        sw.t = SWIPE_TIME + SWIPE_HOLD;
        assert_eq!(sw.point(), None);
        let up = |t| Swipe { dir: Control::SwipeUp, t }.point().unwrap();
        assert!(up(0.0).y > up(SWIPE_TIME).y, "up: towards y = 0");
    }

    #[test]
    fn cursor_moves_and_clamps() {
        let c = move_cursor(point(0.5, 0.5), 80.0, 0.0, 1.0);
        assert!((c.x - 0.6).abs() < 1e-6 && c.y == 0.5);
        let c = move_cursor(c, -1e6, 1e6, 1.0);
        assert_eq!((c.x, c.y), (0.0, 1.0));
    }
}
