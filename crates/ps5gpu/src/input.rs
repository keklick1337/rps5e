// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Window input hook: forwards keyboard/mouse/focus events of the
//! presentation window to an optional sink, and applies window requests
//! (fullscreen toggle, mouse capture) issued by the input layer.
//!
//! The sink runs on the render thread; it must be quick and must not call
//! back into the presenter.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use winit::event::{DeviceEvent, ElementState, MouseScrollDelta, WindowEvent};
use winit::keyboard::PhysicalKey;
use winit::window::{CursorGrabMode, Fullscreen, Window};

pub use crate::osd::show_hint;
pub use winit::keyboard::KeyCode;

/// Mouse buttons of the window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    /// Back side button.
    X1,
    /// Forward side button.
    X2,
    Other(u16),
}

/// Input event of the presentation window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InputEvent {
    /// Physical key; `repeat` is set for auto-repeated presses.
    Key { code: KeyCode, pressed: bool, repeat: bool },
    MouseButton { button: MouseButton, pressed: bool },
    /// Raw relative mouse motion (device units, not clamped to the window).
    MouseMotion { dx: f64, dy: f64 },
    /// Wheel motion in lines (positive y = away from the user / up).
    Wheel { dx: f32, dy: f32 },
    /// Window keyboard focus gained (true) or lost (false).
    Focus(bool),
}

/// Requests from the input layer to the window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowRequest {
    ToggleFullscreen,
    SetFullscreen(bool),
    /// Confine/hide (true) or release (false) the cursor.
    CaptureMouse(bool),
}

type Sink = Box<dyn Fn(InputEvent) + Send + Sync>;

static SINK: Mutex<Option<Sink>> = Mutex::new(None);
static REQUESTS: Mutex<Vec<WindowRequest>> = Mutex::new(Vec::new());
static FULLSCREEN: AtomicBool = AtomicBool::new(false);

/// Whether the game window is fullscreen (as last set by the render thread).
pub fn fullscreen() -> bool {
    FULLSCREEN.load(Ordering::Relaxed)
}

pub(crate) fn note_fullscreen(on: bool) {
    FULLSCREEN.store(on, Ordering::Relaxed);
}

/// Install (or remove with `None`) the window input sink.
pub fn set_input_sink(sink: Option<Sink>) {
    *SINK.lock().unwrap_or_else(|e| e.into_inner()) = sink;
}

/// Queue a window request; applied by the render thread on its next pump.
pub fn request_window(req: WindowRequest) {
    REQUESTS.lock().unwrap_or_else(|e| e.into_inner()).push(req);
    wake();
}

/// Wake the window's render thread (e.g. new menu input arrived).
pub fn wake_window() {
    wake();
}

/// Wake the render thread's event loop.
pub(crate) fn wake() {
    if let Some(w) = &crate::presenter().lock().waker {
        w();
    }
}

/// Whether the game window should open fullscreen (`RPS5E_FULLSCREEN=1`).
pub(crate) fn start_fullscreen() -> bool {
    std::env::var("RPS5E_FULLSCREEN").is_ok_and(|v| !v.is_empty() && v != "0")
}

fn emit(ev: InputEvent) {
    if let Some(sink) = SINK.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        sink(ev);
    }
}

fn map_button(b: winit::event::MouseButton) -> MouseButton {
    use winit::event::MouseButton as W;
    match b {
        W::Left => MouseButton::Left,
        W::Right => MouseButton::Right,
        W::Middle => MouseButton::Middle,
        W::Back => MouseButton::X1,
        W::Forward => MouseButton::X2,
        W::Other(n) => MouseButton::Other(n),
    }
}

/// A key event for the sink. Bindings name key *positions*: only the
/// physical key (scancode) is used, never the logical key or the text the
/// active layout produces, so `C` stays Circle on a Russian layout (where
/// that key types `с`) and WASD stays WASD on AZERTY.
pub(crate) fn key_input(physical: PhysicalKey, state: ElementState, repeat: bool) -> Option<InputEvent> {
    match physical {
        PhysicalKey::Code(code) => Some(InputEvent::Key { code, pressed: state == ElementState::Pressed, repeat }),
        PhysicalKey::Unidentified(_) => None,
    }
}

/// Translate one window event for the sink.
pub(crate) fn on_window_event(event: &WindowEvent) {
    match event {
        WindowEvent::KeyboardInput { event, .. } => {
            if let Some(ev) = key_input(event.physical_key, event.state, event.repeat) {
                emit(ev);
            }
        }
        WindowEvent::MouseInput { state, button, .. } => {
            emit(InputEvent::MouseButton { button: map_button(*button), pressed: *state == ElementState::Pressed });
        }
        WindowEvent::MouseWheel { delta, .. } => {
            let (dx, dy) = match *delta {
                MouseScrollDelta::LineDelta(x, y) => (x, y),
                MouseScrollDelta::PixelDelta(p) => ((p.x / 40.0) as f32, (p.y / 40.0) as f32),
            };
            emit(InputEvent::Wheel { dx, dy });
        }
        WindowEvent::Focused(f) => emit(InputEvent::Focus(*f)),
        _ => {}
    }
}

/// Translate one device event (raw mouse motion) for the sink.
pub(crate) fn on_device_event(event: &DeviceEvent) {
    if let DeviceEvent::MouseMotion { delta } = event {
        emit(InputEvent::MouseMotion { dx: delta.0, dy: delta.1 });
    }
}

/// Apply queued window requests.
pub(crate) fn apply_requests(window: Option<&Window>) {
    let reqs = std::mem::take(&mut *REQUESTS.lock().unwrap_or_else(|e| e.into_inner()));
    let Some(window) = window else { return };
    for req in reqs {
        match req {
            WindowRequest::ToggleFullscreen | WindowRequest::SetFullscreen(_) => {
                let on = match req {
                    WindowRequest::SetFullscreen(on) => on,
                    _ => window.fullscreen().is_none(),
                };
                window.set_fullscreen(on.then_some(Fullscreen::Borderless(None)));
                note_fullscreen(on);
            }
            WindowRequest::CaptureMouse(on) => {
                if on {
                    let grabbed = window
                        .set_cursor_grab(CursorGrabMode::Locked)
                        .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
                    if let Err(e) = grabbed {
                        log::warn!("ps5gpu: cannot capture the mouse: {e}");
                    }
                } else {
                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                }
                window.set_cursor_visible(!on);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::keyboard::NativeKeyCode;

    #[test]
    fn keys_are_physical_positions() {
        // On a Russian layout the key at the C position types 'с' (and on
        // AZERTY the W position types 'z'): winit still reports the
        // position, which is all the sink sees.
        for code in [KeyCode::KeyC, KeyCode::KeyW, KeyCode::KeyF, KeyCode::KeyR] {
            assert_eq!(
                key_input(PhysicalKey::Code(code), ElementState::Pressed, false),
                Some(InputEvent::Key { code, pressed: true, repeat: false })
            );
        }
        assert_eq!(
            key_input(PhysicalKey::Code(KeyCode::KeyC), ElementState::Released, true),
            Some(InputEvent::Key { code: KeyCode::KeyC, pressed: false, repeat: true })
        );
        // A scancode without a position name cannot match any binding.
        assert_eq!(key_input(PhysicalKey::Unidentified(NativeKeyCode::Xkb(0x1F0)), ElementState::Pressed, false), None);
    }
}
