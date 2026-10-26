// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Frontend keys and mouse buttons to the emulator's binding sources.

use eframe::egui::{Key, Modifiers, PointerButton};
use ps5gpu::input::{KeyCode, MouseButton};

/// The key code a bindable egui key corresponds to (physical position on
/// a US layout); `None` for keys the binding format has no name for.
pub fn key_code(k: Key) -> Option<KeyCode> {
    use KeyCode as C;
    Some(match k {
        Key::ArrowDown => C::ArrowDown,
        Key::ArrowLeft => C::ArrowLeft,
        Key::ArrowRight => C::ArrowRight,
        Key::ArrowUp => C::ArrowUp,
        Key::Escape => C::Escape,
        Key::Tab => C::Tab,
        Key::Backspace => C::Backspace,
        Key::Enter => C::Enter,
        Key::Space => C::Space,
        Key::Insert => C::Insert,
        Key::Delete => C::Delete,
        Key::Home => C::Home,
        Key::End => C::End,
        Key::PageUp => C::PageUp,
        Key::PageDown => C::PageDown,
        Key::Comma => C::Comma,
        Key::Backslash | Key::Pipe => C::Backslash,
        Key::Slash | Key::Questionmark => C::Slash,
        Key::OpenBracket | Key::OpenCurlyBracket => C::BracketLeft,
        Key::CloseBracket | Key::CloseCurlyBracket => C::BracketRight,
        Key::Backtick => C::Backquote,
        Key::Minus => C::Minus,
        Key::Period => C::Period,
        Key::Equals | Key::Plus => C::Equal,
        Key::Semicolon | Key::Colon => C::Semicolon,
        Key::Quote => C::Quote,
        Key::Num0 => C::Digit0,
        Key::Num1 => C::Digit1,
        Key::Num2 => C::Digit2,
        Key::Num3 => C::Digit3,
        Key::Num4 => C::Digit4,
        Key::Num5 => C::Digit5,
        Key::Num6 => C::Digit6,
        Key::Num7 => C::Digit7,
        Key::Num8 => C::Digit8,
        Key::Num9 => C::Digit9,
        Key::A => C::KeyA,
        Key::B => C::KeyB,
        Key::C => C::KeyC,
        Key::D => C::KeyD,
        Key::E => C::KeyE,
        Key::F => C::KeyF,
        Key::G => C::KeyG,
        Key::H => C::KeyH,
        Key::I => C::KeyI,
        Key::J => C::KeyJ,
        Key::K => C::KeyK,
        Key::L => C::KeyL,
        Key::M => C::KeyM,
        Key::N => C::KeyN,
        Key::O => C::KeyO,
        Key::P => C::KeyP,
        Key::Q => C::KeyQ,
        Key::R => C::KeyR,
        Key::S => C::KeyS,
        Key::T => C::KeyT,
        Key::U => C::KeyU,
        Key::V => C::KeyV,
        Key::W => C::KeyW,
        Key::X => C::KeyX,
        Key::Y => C::KeyY,
        Key::Z => C::KeyZ,
        Key::F1 => C::F1,
        Key::F2 => C::F2,
        Key::F3 => C::F3,
        Key::F4 => C::F4,
        Key::F5 => C::F5,
        Key::F6 => C::F6,
        Key::F7 => C::F7,
        Key::F8 => C::F8,
        Key::F9 => C::F9,
        Key::F10 => C::F10,
        Key::F11 => C::F11,
        Key::F12 => C::F12,
        Key::F13 => C::F13,
        Key::F14 => C::F14,
        Key::F15 => C::F15,
        Key::F16 => C::F16,
        Key::F17 => C::F17,
        Key::F18 => C::F18,
        Key::F19 => C::F19,
        Key::F20 => C::F20,
        Key::F21 => C::F21,
        Key::F22 => C::F22,
        Key::F23 => C::F23,
        Key::F24 => C::F24,
        _ => return None,
    })
}

/// Modifier keys newly held in `now` compared with `before` (egui reports
/// modifiers, not key events, for Shift/Ctrl/Alt; the left key is used).
pub fn pressed_modifiers(before: Modifiers, now: Modifiers) -> Vec<KeyCode> {
    let mut v = Vec::new();
    if now.shift && !before.shift {
        v.push(KeyCode::ShiftLeft);
    }
    if now.ctrl && !before.ctrl {
        v.push(KeyCode::ControlLeft);
    }
    if now.alt && !before.alt {
        v.push(KeyCode::AltLeft);
    }
    v
}

pub fn mouse_button(b: PointerButton) -> MouseButton {
    match b {
        PointerButton::Primary => MouseButton::Left,
        PointerButton::Secondary => MouseButton::Right,
        PointerButton::Middle => MouseButton::Middle,
        PointerButton::Extra1 => MouseButton::X1,
        PointerButton::Extra2 => MouseButton::X2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ps5input::mapping::{key_from_name, key_name};

    #[test]
    fn every_mapped_key_has_a_file_name() {
        for k in Key::ALL {
            if let Some(code) = key_code(*k) {
                let name = key_name(code).unwrap_or_else(|| panic!("{k:?} -> {code:?} has no name"));
                assert_eq!(key_from_name(&name), Some(code));
            }
        }
        assert_eq!(key_code(Key::Num7), Some(KeyCode::Digit7));
        assert_eq!(key_code(Key::Copy), None);
    }

    #[test]
    fn modifiers_edge() {
        let none = Modifiers::NONE;
        assert_eq!(pressed_modifiers(none, Modifiers::SHIFT), [KeyCode::ShiftLeft]);
        assert!(pressed_modifiers(Modifiers::SHIFT, Modifiers::SHIFT).is_empty());
        assert_eq!(pressed_modifiers(none, Modifiers::CTRL | Modifiers::ALT), [KeyCode::ControlLeft, KeyCode::AltLeft]);
    }
}
