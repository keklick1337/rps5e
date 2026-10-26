// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Input settings: built-in bindings and the `rps5e-input.toml`
//! configuration file (global file plus optional per-game profiles, see
//! [`crate::profile`]).
//!
//! File format (TOML; every section and key is optional):
//!
//! ```toml
//! [device]
//! mode = "auto"        # auto | keyboard_mouse | keyboard | controller
//! controller = ""      # player 1 controller: serial or name; "" = first free
//!
//! [players]
//! count = 1            # local players (1..4), each a signed-in user
//! player2 = ""         # controller of player 2 ("" = next free one)
//!
//! [bindings]           # keyboard + mouse (modes auto / keyboard_mouse)
//! Cross = ["KEY:F", "KEY:Space"]   # alternates
//! Square = "MOUSE:Left"
//! Up = "WHEEL:Up"
//! TouchLeft = []                   # unbound
//!
//! [keyboard_bindings]  # keyboard-only mode (no mouse sources)
//! Square = "KEY:F"
//!
//! [gamepad_bindings]   # physical controller buttons (PS layout names)
//! Cross = "BUTTON:East"
//!
//! [mouse]
//! sensitivity = 1.0    # mouse-look (ToggleMouse) sensitivity
//! invert_x = false
//! invert_y = false
//! guest_mouse = false  # expose the host mouse to libSceMouse
//!
//! [gamepad]
//! enabled = true       # SDL3 game controllers
//! rumble = true
//! rumble_strength = 1.0
//! lightbar = true
//! adaptive_triggers = true
//! motion = true
//! touchpad = true
//! speaker_volume = 50       # DualSense speaker level 0..100 (50 = unity)
//! vibration_intensity = 100 # vibration level 0..100
//! lightbar_color = ""       # fixed light bar colour "#RRGGBB" (overrides the game's)
//!
//! [sticks]
//! left_mode = "keys"   # keys | mouse | off: what drives the stick besides a
//! right_mode = "mouse" # controller (mouse-look drives at most one stick)
//! left_deadzone = 0.0  # radial, fraction of full deflection
//! left_curve = 1.0     # response exponent
//! right_deadzone = 0.0
//! right_curve = 1.0
//! swap = false         # keyboard/mouse drive the other stick
//!
//! [triggers]
//! deadzone = 0.0
//! curve = 1.0
//!
//! [motion]             # keyboard/mouse motion sensors (crate::motion)
//! tilt_angle = 30.0
//! tilt_time = 0.25
//! mouse_sensitivity = 1.0
//!
//! [touchpad]
//! mouse_sensitivity = 1.0  # touch pad mode mouse finger
//!
//! [profile]            # per-game profile files only
//! enabled = true
//! ```
//!
//! Every action listed under a bindings section replaces its built-in
//! bindings; actions not listed keep them. Section, key and action names,
//! source types and values are case-insensitive. Sources are `KEY:<name>`
//! (key names as in SDL, e.g. `Return`, `Left Shift`, `Keypad 4`, or winit
//! key codes, e.g. `Enter`, `ShiftLeft`, `Numpad4`),
//! `MOUSE:Left|Middle|Right|X1|X2`, `WHEEL:Up|Down` (pad buttons only) and,
//! in `[gamepad_bindings]`, `BUTTON:<name>` (see [`PAD_INPUTS`]). Duplicate
//! sources are ignored.

use crate::shaping::AxisShape;
use ps5gpu::input::{KeyCode, MouseButton};
use ps5hle::pad::button;
use std::fmt;
use std::path::{Path, PathBuf};

/// What a binding drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Control {
    /// SCE_PAD_BUTTON bit.
    Button(u32),
    LeftStickLeft,
    LeftStickRight,
    LeftStickUp,
    LeftStickDown,
    RightStickLeft,
    RightStickRight,
    RightStickUp,
    RightStickDown,
    TouchLeft,
    TouchRight,
    /// Toggle mouse capture (mouse-look drives the stick in mouse mode).
    ToggleMouse,
    /// Leave mouse capture; the key keeps its other bindings.
    ReleaseMouse,
    ToggleFullscreen,
    /// Show or hide the performance overlay.
    TogglePerfOverlay,
    /// Open or close the in-game menu.
    ToggleMenu,
    /// Show the controls overlay over the game while held.
    ShowControls,
    /// Step the controller speaker volume (cycles).
    SpeakerVolume,
    /// Step the vibration intensity (cycles).
    VibrationIntensity,
    /// Step the trigger effect intensity (cycles).
    TriggerEffectIntensity,
    /// Tilt the controller while held (see [`crate::motion`]): the far
    /// edge down / up, the left / right grip down, turn left / right.
    TiltForward,
    TiltBack,
    TiltLeft,
    TiltRight,
    TurnLeft,
    TurnRight,
    /// A short shake burst.
    Shake,
    /// While held, mouse movement turns the controller (gyro aiming).
    MouseMotion,
    /// A finger on the touch pad (no click) while held.
    TapLeft,
    TapCenter,
    TapRight,
    /// A finger sliding across the touch pad.
    SwipeUp,
    SwipeDown,
    SwipeLeft,
    SwipeRight,
    /// Two fingers on the touch pad while held.
    TwoFingerTouch,
    /// Toggle touch pad mode: the numpad digits are touch pad zones and the
    /// mouse drives a finger.
    TouchpadMode,
    /// The Create button: a screenshot of the game window.
    Create,
    /// The mute button: turns the game's sound off and on.
    Mute,
}

impl Control {
    /// Window hotkeys (not pad input).
    pub fn is_hotkey(self) -> bool {
        matches!(
            self,
            Control::ToggleMouse
                | Control::ReleaseMouse
                | Control::ToggleFullscreen
                | Control::TogglePerfOverlay
                | Control::ToggleMenu
                | Control::ShowControls
                | Control::SpeakerVolume
                | Control::VibrationIntensity
                | Control::TriggerEffectIntensity
                | Control::TouchpadMode
                | Control::Create
                | Control::Mute
        )
    }

    /// Actions of the keyboard + mouse set only (they need the mouse).
    pub fn mouse_only(self) -> bool {
        matches!(self, Control::ToggleMouse | Control::ReleaseMouse | Control::MouseMotion)
    }

    /// Motion sensor actions.
    pub fn is_motion(self) -> bool {
        matches!(
            self,
            Control::TiltForward
                | Control::TiltBack
                | Control::TiltLeft
                | Control::TiltRight
                | Control::TurnLeft
                | Control::TurnRight
                | Control::Shake
                | Control::MouseMotion
        )
    }

    /// Touch pad actions (besides the touch pad click button).
    pub fn is_touch(self) -> bool {
        matches!(
            self,
            Control::TouchLeft
                | Control::TouchRight
                | Control::TapLeft
                | Control::TapCenter
                | Control::TapRight
                | Control::SwipeUp
                | Control::SwipeDown
                | Control::SwipeLeft
                | Control::SwipeRight
                | Control::TwoFingerTouch
                | Control::TouchpadMode
        )
    }

    /// The controller setting a cycle hotkey steps.
    pub fn cycled_setting(self) -> Option<ps5hle::pad_controller::Setting> {
        use ps5hle::pad_controller::Setting;
        match self {
            Control::SpeakerVolume => Some(Setting::SpeakerVolume),
            Control::VibrationIntensity => Some(Setting::VibrationIntensity),
            Control::TriggerEffectIntensity => Some(Setting::TriggerEffectIntensity),
            _ => None,
        }
    }

    /// The stick a direction action moves.
    pub fn stick(self) -> Option<Stick> {
        match self {
            Control::LeftStickLeft | Control::LeftStickRight | Control::LeftStickUp | Control::LeftStickDown => Some(Stick::Left),
            Control::RightStickLeft | Control::RightStickRight | Control::RightStickUp | Control::RightStickDown => {
                Some(Stick::Right)
            }
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        ACTIONS.iter().find(|(_, c)| *c == self).map(|(n, _)| *n).unwrap_or("?")
    }
}

/// Physical keyboard/mouse input of a binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    Key(KeyCode),
    Mouse(MouseButton),
    /// +1 = wheel up, -1 = wheel down.
    Wheel(i8),
}

impl Source {
    /// The `TYPE:VALUE` form used in the file, `None` for keys without a
    /// name in the file format.
    pub fn to_config_string(self) -> Option<String> {
        Some(match self {
            Source::Key(k) => format!("KEY:{}", key_name(k)?),
            Source::Mouse(b) => format!("MOUSE:{}", mouse_button_name(b)?),
            Source::Wheel(d) => format!("WHEEL:{}", if d > 0 { "Up" } else { "Down" }),
        })
    }
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_config_string() {
            Some(s) => f.write_str(&s),
            None => write!(f, "{self:?}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Binding {
    pub source: Source,
    pub control: Control,
}

/// A physical controller input: an SDL gamepad button index, or a trigger
/// read as a button ([`PadInput::LEFT_TRIGGER`], [`PadInput::RIGHT_TRIGGER`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PadInput(pub u8);

impl PadInput {
    pub const LEFT_TRIGGER: PadInput = PadInput(0x80);
    pub const RIGHT_TRIGGER: PadInput = PadInput(0x81);

    pub fn name(self) -> &'static str {
        PAD_INPUTS.iter().find(|(_, p)| *p == self).map(|(n, _)| n[0]).unwrap_or("?")
    }

    pub fn to_config_string(self) -> String {
        format!("BUTTON:{}", self.name())
    }

    pub fn from_name(name: &str) -> Option<PadInput> {
        let norm: String = name.chars().filter(|c| !matches!(c, ' ' | '_' | '-')).collect::<String>().to_ascii_lowercase();
        PAD_INPUTS.iter().find(|(names, _)| names.iter().any(|n| n.eq_ignore_ascii_case(&norm))).map(|&(_, p)| p)
    }
}

/// Controller input names (first = canonical, PS layout aliases after).
/// Indices are SDL3 `SDL_GamepadButton` values.
pub const PAD_INPUTS: &[(&[&str], PadInput)] = &[
    (&["South", "cross", "a"], PadInput(0)),
    (&["East", "circle", "b"], PadInput(1)),
    (&["West", "square", "x"], PadInput(2)),
    (&["North", "triangle", "y"], PadInput(3)),
    (&["Back", "create", "share", "select", "view"], PadInput(4)),
    (&["Guide", "ps", "home"], PadInput(5)),
    (&["Start", "options", "menu"], PadInput(6)),
    (&["LeftStick", "l3"], PadInput(7)),
    (&["RightStick", "r3"], PadInput(8)),
    (&["LeftShoulder", "l1", "lb"], PadInput(9)),
    (&["RightShoulder", "r1", "rb"], PadInput(10)),
    (&["DpadUp", "up"], PadInput(11)),
    (&["DpadDown", "down"], PadInput(12)),
    (&["DpadLeft", "left"], PadInput(13)),
    (&["DpadRight", "right"], PadInput(14)),
    (&["Misc1", "mute", "capture"], PadInput(15)),
    (&["RightPaddle1", "p1"], PadInput(16)),
    (&["LeftPaddle1", "p3"], PadInput(17)),
    (&["RightPaddle2", "p2"], PadInput(18)),
    (&["LeftPaddle2", "p4"], PadInput(19)),
    (&["Touchpad", "touch"], PadInput(20)),
    (&["LeftTrigger", "l2", "lt"], PadInput::LEFT_TRIGGER),
    (&["RightTrigger", "r2", "rt"], PadInput::RIGHT_TRIGGER),
];

/// Controller button binding: `input` presses the pad button `bit`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PadBinding {
    pub input: PadInput,
    pub bit: u32,
}

/// Action names accepted in the file, with their controls.
pub const ACTIONS: &[(&str, Control)] = &[
    ("Cross", Control::Button(button::CROSS)),
    ("Circle", Control::Button(button::CIRCLE)),
    ("Triangle", Control::Button(button::TRIANGLE)),
    ("Square", Control::Button(button::SQUARE)),
    ("L1", Control::Button(button::L1)),
    ("R1", Control::Button(button::R1)),
    ("L2", Control::Button(button::L2)),
    ("R2", Control::Button(button::R2)),
    ("L3", Control::Button(button::L3)),
    ("R3", Control::Button(button::R3)),
    ("Options", Control::Button(button::OPTIONS)),
    ("Up", Control::Button(button::UP)),
    ("Right", Control::Button(button::RIGHT)),
    ("Down", Control::Button(button::DOWN)),
    ("Left", Control::Button(button::LEFT)),
    ("TouchPad", Control::Button(button::TOUCH_PAD)),
    ("LeftStickLeft", Control::LeftStickLeft),
    ("LeftStickRight", Control::LeftStickRight),
    ("LeftStickUp", Control::LeftStickUp),
    ("LeftStickDown", Control::LeftStickDown),
    ("RightStickLeft", Control::RightStickLeft),
    ("RightStickRight", Control::RightStickRight),
    ("RightStickUp", Control::RightStickUp),
    ("RightStickDown", Control::RightStickDown),
    ("TouchLeft", Control::TouchLeft),
    ("TouchRight", Control::TouchRight),
    ("ToggleMouse", Control::ToggleMouse),
    ("ReleaseMouse", Control::ReleaseMouse),
    ("ToggleFullscreen", Control::ToggleFullscreen),
    ("TogglePerfOverlay", Control::TogglePerfOverlay),
    ("ToggleMenu", Control::ToggleMenu),
    ("SpeakerVolume", Control::SpeakerVolume),
    ("VibrationIntensity", Control::VibrationIntensity),
    ("TriggerEffectIntensity", Control::TriggerEffectIntensity),
    ("ShowControls", Control::ShowControls),
    ("Create", Control::Create),
    ("Mute", Control::Mute),
    ("TiltForward", Control::TiltForward),
    ("TiltBack", Control::TiltBack),
    ("TiltLeft", Control::TiltLeft),
    ("TiltRight", Control::TiltRight),
    ("TurnLeft", Control::TurnLeft),
    ("TurnRight", Control::TurnRight),
    ("Shake", Control::Shake),
    ("MouseMotion", Control::MouseMotion),
    ("TouchpadMode", Control::TouchpadMode),
    ("TapLeft", Control::TapLeft),
    ("TapCenter", Control::TapCenter),
    ("TapRight", Control::TapRight),
    ("SwipeUp", Control::SwipeUp),
    ("SwipeDown", Control::SwipeDown),
    ("SwipeLeft", Control::SwipeLeft),
    ("SwipeRight", Control::SwipeRight),
    ("TwoFingerTouch", Control::TwoFingerTouch),
];

/// Pad button actions (the ones `[gamepad_bindings]` accepts).
pub fn pad_button_actions() -> impl Iterator<Item = (&'static str, u32)> {
    ACTIONS.iter().filter_map(|&(n, c)| match c {
        Control::Button(bit) => Some((n, bit)),
        _ => None,
    })
}

/// Built-in keyboard/mouse bindings (modes auto and keyboard_mouse): WASD
/// on the left stick, the mouse on the right one (I/J/K/L when it is on
/// its keys), the face buttons on Space/C/R/F, L2/R2 on the right/left
/// mouse button, motion on the numpad.
pub fn default_bindings() -> Vec<Binding> {
    use Control as C;
    use KeyCode as K;
    let key = |k, control| Binding { source: Source::Key(k), control };
    let mouse = |b, control| Binding { source: Source::Mouse(b), control };
    let mut v = vec![
        key(K::F11, C::ToggleFullscreen),
        key(K::F10, C::TogglePerfOverlay),
        key(K::F1, C::ToggleMenu),
        key(K::F2, C::ShowControls),
        key(K::F9, C::ToggleMouse),
        key(K::Escape, C::ReleaseMouse),
        key(K::Space, C::Button(button::CROSS)),
        key(K::Enter, C::Button(button::CROSS)),
        key(K::KeyC, C::Button(button::CIRCLE)),
        key(K::KeyR, C::Button(button::SQUARE)),
        key(K::KeyF, C::Button(button::TRIANGLE)),
        key(K::Escape, C::Button(button::OPTIONS)),
        key(K::KeyQ, C::Button(button::L1)),
        key(K::KeyE, C::Button(button::R1)),
        mouse(MouseButton::Right, C::Button(button::L2)),
        mouse(MouseButton::Left, C::Button(button::R2)),
        key(K::ShiftLeft, C::Button(button::L3)),
        key(K::ShiftRight, C::Button(button::L3)),
        key(K::ControlLeft, C::Button(button::R3)),
        key(K::ControlRight, C::Button(button::R3)),
        key(K::ArrowUp, C::Button(button::UP)),
        key(K::ArrowRight, C::Button(button::RIGHT)),
        key(K::ArrowDown, C::Button(button::DOWN)),
        key(K::ArrowLeft, C::Button(button::LEFT)),
        Binding { source: Source::Wheel(1), control: C::Button(button::UP) },
        Binding { source: Source::Wheel(-1), control: C::Button(button::DOWN) },
        key(K::KeyA, C::LeftStickLeft),
        key(K::KeyD, C::LeftStickRight),
        key(K::KeyW, C::LeftStickUp),
        key(K::KeyS, C::LeftStickDown),
        key(K::KeyJ, C::RightStickLeft),
        key(K::KeyL, C::RightStickRight),
        key(K::KeyI, C::RightStickUp),
        key(K::KeyK, C::RightStickDown),
        mouse(MouseButton::Middle, C::MouseMotion),
    ];
    v.extend(shared_extra_bindings());
    v
}

/// Defaults both keyboard sets share: the touch pad click, Create, mute,
/// motion on the numpad and the touch pad gestures.
fn shared_extra_bindings() -> Vec<Binding> {
    use Control as C;
    use KeyCode as K;
    let key = |k, control| Binding { source: Source::Key(k), control };
    vec![
        key(K::Tab, C::Button(button::TOUCH_PAD)),
        key(K::F12, C::Create),
        key(K::F8, C::Mute),
        key(K::Numpad8, C::TiltForward),
        key(K::Numpad2, C::TiltBack),
        key(K::Numpad4, C::TiltLeft),
        key(K::Numpad6, C::TiltRight),
        key(K::Numpad7, C::TurnLeft),
        key(K::Numpad9, C::TurnRight),
        key(K::Numpad5, C::Shake),
        key(K::Numpad0, C::TouchpadMode),
        key(K::Home, C::SwipeUp),
        key(K::End, C::SwipeDown),
        key(K::Delete, C::SwipeLeft),
        key(K::PageDown, C::SwipeRight),
        key(K::PageUp, C::TapCenter),
        key(K::Insert, C::TwoFingerTouch),
    ]
}

/// Built-in keyboard-only bindings: every pad input on a key, the right
/// stick on I/J/K/L.
pub fn default_keyboard_bindings() -> Vec<Binding> {
    use Control as C;
    use KeyCode as K;
    let key = |k, control| Binding { source: Source::Key(k), control };
    let mut v = vec![
        key(K::F11, C::ToggleFullscreen),
        key(K::F10, C::TogglePerfOverlay),
        key(K::F1, C::ToggleMenu),
        key(K::F2, C::ShowControls),
        key(K::Space, C::Button(button::CROSS)),
        key(K::Enter, C::Button(button::CROSS)),
        key(K::KeyC, C::Button(button::CIRCLE)),
        key(K::KeyR, C::Button(button::SQUARE)),
        key(K::KeyF, C::Button(button::TRIANGLE)),
        key(K::Escape, C::Button(button::OPTIONS)),
        key(K::KeyQ, C::Button(button::L1)),
        key(K::KeyE, C::Button(button::R1)),
        key(K::Digit1, C::Button(button::L2)),
        key(K::Digit3, C::Button(button::R2)),
        key(K::ShiftLeft, C::Button(button::L3)),
        key(K::KeyO, C::Button(button::R3)),
        key(K::ArrowUp, C::Button(button::UP)),
        key(K::ArrowRight, C::Button(button::RIGHT)),
        key(K::ArrowDown, C::Button(button::DOWN)),
        key(K::ArrowLeft, C::Button(button::LEFT)),
        key(K::KeyA, C::LeftStickLeft),
        key(K::KeyD, C::LeftStickRight),
        key(K::KeyW, C::LeftStickUp),
        key(K::KeyS, C::LeftStickDown),
        key(K::KeyJ, C::RightStickLeft),
        key(K::KeyL, C::RightStickRight),
        key(K::KeyI, C::RightStickUp),
        key(K::KeyK, C::RightStickDown),
    ];
    v.extend(shared_extra_bindings());
    v
}

/// Built-in controller layout: SDL's PS-style layout, one to one.
pub fn default_pad_bindings() -> Vec<PadBinding> {
    let b = |i: u8, bit| PadBinding { input: PadInput(i), bit };
    vec![
        b(0, button::CROSS),
        b(1, button::CIRCLE),
        b(2, button::SQUARE),
        b(3, button::TRIANGLE),
        b(9, button::L1),
        b(10, button::R1),
        b(6, button::OPTIONS),
        b(7, button::L3),
        b(8, button::R3),
        b(11, button::UP),
        b(14, button::RIGHT),
        b(12, button::DOWN),
        b(13, button::LEFT),
        b(20, button::TOUCH_PAD),
    ]
}

/// Which devices drive player 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DeviceMode {
    /// Controller, keyboard and mouse together.
    #[default]
    Auto,
    /// Keyboard and mouse act as the pad (no controller).
    KeyboardMouse,
    /// Keyboard only; the mouse is ignored.
    Keyboard,
    /// Physical controller only (the keyboard keeps the window hotkeys).
    Controller,
}

impl DeviceMode {
    pub const ALL: [DeviceMode; 4] =
        [DeviceMode::Auto, DeviceMode::KeyboardMouse, DeviceMode::Keyboard, DeviceMode::Controller];

    pub fn as_str(self) -> &'static str {
        match self {
            DeviceMode::Auto => "auto",
            DeviceMode::KeyboardMouse => "keyboard_mouse",
            DeviceMode::Keyboard => "keyboard",
            DeviceMode::Controller => "controller",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            DeviceMode::Auto => "Automatic (controller + keyboard + mouse)",
            DeviceMode::KeyboardMouse => "Keyboard and mouse as gamepad",
            DeviceMode::Keyboard => "Keyboard only",
            DeviceMode::Controller => "Physical controller",
        }
    }

    pub fn parse(s: &str) -> Option<DeviceMode> {
        let n: String = s.chars().filter(|c| !matches!(c, ' ' | '_' | '-' | '+')).collect::<String>().to_ascii_lowercase();
        Some(match n.as_str() {
            "auto" | "automatic" => DeviceMode::Auto,
            "keyboardmouse" | "kbm" | "keyboardandmouse" => DeviceMode::KeyboardMouse,
            "keyboard" | "keyboardonly" => DeviceMode::Keyboard,
            "controller" | "gamepad" | "pad" => DeviceMode::Controller,
            _ => return None,
        })
    }

    pub fn uses_controller(self) -> bool {
        matches!(self, DeviceMode::Auto | DeviceMode::Controller)
    }

    pub fn uses_keyboard(self) -> bool {
        matches!(self, DeviceMode::Auto | DeviceMode::KeyboardMouse | DeviceMode::Keyboard)
    }

    pub fn uses_mouse(self) -> bool {
        matches!(self, DeviceMode::Auto | DeviceMode::KeyboardMouse)
    }
}

pub const MAX_PLAYERS: usize = 4;

/// One of the two analog sticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stick {
    Left,
    Right,
}

impl Stick {
    pub const BOTH: [Stick; 2] = [Stick::Left, Stick::Right];

    pub fn other(self) -> Stick {
        match self {
            Stick::Left => Stick::Right,
            Stick::Right => Stick::Left,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Stick::Left => "Left stick",
            Stick::Right => "Right stick",
        }
    }
}

/// What drives a stick from the keyboard and mouse (a controller's stick
/// always drives it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StickMode {
    /// Four direction keys (the stick actions of the binding set).
    Keys,
    /// Mouse-look (while the mouse is captured) drives the whole stick.
    Mouse,
    /// Neither: only a controller moves it.
    Off,
}

impl StickMode {
    pub const ALL: [StickMode; 3] = [StickMode::Keys, StickMode::Mouse, StickMode::Off];

    pub fn as_str(self) -> &'static str {
        match self {
            StickMode::Keys => "keys",
            StickMode::Mouse => "mouse",
            StickMode::Off => "off",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            StickMode::Keys => "Keys (4 directions)",
            StickMode::Mouse => "Mouse",
            StickMode::Off => "Controller only",
        }
    }

    pub fn parse(s: &str) -> Option<StickMode> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "keys" | "keyboard" | "key" => StickMode::Keys,
            "mouse" => StickMode::Mouse,
            "off" | "none" | "controller" | "pad" => StickMode::Off,
            _ => return None,
        })
    }
}

/// Controller feature switches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GamepadFeatures {
    pub rumble: bool,
    /// Rumble scale (0..2).
    pub rumble_strength: f32,
    pub lightbar: bool,
    pub adaptive_triggers: bool,
    pub motion: bool,
    pub touchpad: bool,
    /// Global pad speaker level (0..100, 50 = unity gain).
    pub speaker_volume: u32,
    /// Global vibration level (0..100).
    pub vibration_intensity: u32,
    /// Fixed light bar colour overriding the game's requests.
    pub lightbar_color: Option<[u8; 3]>,
}

impl Default for GamepadFeatures {
    fn default() -> Self {
        Self {
            rumble: true,
            rumble_strength: 1.0,
            lightbar: true,
            adaptive_triggers: true,
            motion: true,
            touchpad: true,
            speaker_volume: ps5hle::pad_controller::DEFAULT_SPEAKER_VOLUME,
            vibration_intensity: ps5hle::pad_controller::DEFAULT_VIBRATION_INTENSITY,
            lightbar_color: None,
        }
    }
}

/// Parsed configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Keyboard + mouse bindings (modes auto and keyboard_mouse).
    pub bindings: Vec<Binding>,
    /// Keyboard-only bindings.
    pub keyboard_bindings: Vec<Binding>,
    /// Controller button layout.
    pub pad_bindings: Vec<PadBinding>,
    pub mouse_sensitivity: f64,
    pub mouse_invert_x: bool,
    pub mouse_invert_y: bool,
    pub guest_mouse: bool,
    /// SDL3 game controllers enabled.
    pub gamepad: bool,
    pub features: GamepadFeatures,
    pub mode: DeviceMode,
    /// Local players (1..=4).
    pub players: usize,
    /// Controller selector per player ("" = first free controller).
    pub controllers: [String; MAX_PLAYERS],
    pub left_stick: AxisShape,
    pub right_stick: AxisShape,
    /// Keyboard/mouse drive of each stick (`[sticks] left_mode / right_mode`).
    pub left_stick_mode: StickMode,
    pub right_stick_mode: StickMode,
    pub triggers: AxisShape,
    /// Per-game profile files: `[profile] enabled`.
    pub profile_enabled: bool,
    /// The keyboard and mouse drive the other stick (`[sticks] swap`): the
    /// left stick's keys and mouse-look move the right stick and the other
    /// way round. A controller's sticks are not swapped.
    pub swap_sticks: bool,
    /// Keyboard/mouse motion sensors (`[motion]`).
    pub motion: crate::motion::MotionSettings,
    /// Touch pad mode: how fast the mouse moves its finger
    /// (`[touchpad] mouse_sensitivity`).
    pub touch_mouse_sensitivity: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bindings: default_bindings(),
            keyboard_bindings: default_keyboard_bindings(),
            pad_bindings: default_pad_bindings(),
            mouse_sensitivity: 1.0,
            mouse_invert_x: false,
            mouse_invert_y: false,
            guest_mouse: false,
            gamepad: true,
            features: GamepadFeatures::default(),
            mode: DeviceMode::Auto,
            players: 1,
            controllers: Default::default(),
            left_stick: AxisShape::LINEAR,
            right_stick: AxisShape::LINEAR,
            left_stick_mode: StickMode::Keys,
            right_stick_mode: StickMode::Mouse,
            triggers: AxisShape::LINEAR,
            profile_enabled: true,
            swap_sticks: false,
            motion: crate::motion::MotionSettings::default(),
            touch_mouse_sensitivity: 1.0,
        }
    }
}

impl Config {
    /// The stick mode as set.
    pub fn stick_mode(&self, stick: Stick) -> StickMode {
        match stick {
            Stick::Left => self.left_stick_mode,
            Stick::Right => self.right_stick_mode,
        }
    }

    /// Set a stick's mode. The mouse drives one stick: giving it to one
    /// puts the other on its keys.
    pub fn set_stick_mode(&mut self, stick: Stick, mode: StickMode) {
        if mode == StickMode::Mouse && self.stick_mode(stick.other()) == StickMode::Mouse {
            self.set_stick_mode(stick.other(), StickMode::Keys);
        }
        match stick {
            Stick::Left => self.left_stick_mode = mode,
            Stick::Right => self.right_stick_mode = mode,
        }
    }

    /// Built-in settings of one stick: its mode (keys on the left, the
    /// mouse on the right), dead zone and curve, and for a mouse stick the
    /// mouse-look settings.
    pub fn reset_stick(&mut self, stick: Stick) {
        let d = Config::default();
        self.set_stick_mode(stick, d.stick_mode(stick));
        match stick {
            Stick::Left => self.left_stick = d.left_stick,
            Stick::Right => self.right_stick = d.right_stick,
        }
        if self.stick_mode(stick) == StickMode::Mouse {
            self.mouse_sensitivity = d.mouse_sensitivity;
            self.mouse_invert_x = d.mouse_invert_x;
            self.mouse_invert_y = d.mouse_invert_y;
        }
    }

    /// The stick mode of a binding set (`keyboard_only`: the keyboard-only
    /// set, where a mouse stick falls back to its keys).
    pub fn stick_mode_in(&self, stick: Stick, keyboard_only: bool) -> StickMode {
        match self.stick_mode(stick) {
            StickMode::Mouse if keyboard_only => StickMode::Keys,
            m => m,
        }
    }

    /// The stick mouse-look drives in the device mode of player 1.
    pub fn mouse_stick(&self) -> Option<Stick> {
        if !self.mode.uses_mouse() {
            return None;
        }
        Stick::BOTH.into_iter().find(|&s| self.stick_mode(s) == StickMode::Mouse)
    }

    /// Keyboard/mouse bindings in effect for the device mode of player 1:
    /// the stick directions only of sticks on their keys.
    pub fn active_bindings(&self) -> Vec<Binding> {
        let keys_drive = |b: &Binding, kb: bool| b.control.stick().is_none_or(|s| self.stick_mode_in(s, kb) == StickMode::Keys);
        match self.mode {
            DeviceMode::Auto | DeviceMode::KeyboardMouse => self.bindings.iter().copied().filter(|b| keys_drive(b, false)).collect(),
            DeviceMode::Keyboard => self.keyboard_bindings.iter().copied().filter(|b| keys_drive(b, true)).collect(),
            // The keyboard keeps the window hotkeys and the system buttons.
            DeviceMode::Controller => self
                .bindings
                .iter()
                .copied()
                .filter(|b| {
                    matches!(
                        b.control,
                        Control::ToggleFullscreen
                            | Control::TogglePerfOverlay
                            | Control::ToggleMenu
                            | Control::ShowControls
                            | Control::Create
                            | Control::Mute
                    )
                })
                .collect(),
        }
    }

    /// Players that use a controller, with their selectors: player 1 only
    /// when its mode includes a controller.
    pub fn controller_players(&self) -> Vec<(usize, &str)> {
        (0..self.players.clamp(1, MAX_PLAYERS))
            .filter(|&p| p > 0 || self.mode.uses_controller())
            .map(|p| (p, self.controllers[p].as_str()))
            .collect()
    }

    /// Bindings of `control` in the keyboard/mouse set (`keyboard_only`
    /// selects the keyboard-only set).
    pub fn sources_of(&self, control: Control, keyboard_only: bool) -> Vec<Source> {
        let set = if keyboard_only { &self.keyboard_bindings } else { &self.bindings };
        set.iter().filter(|b| b.control == control).map(|b| b.source).collect()
    }

    /// Replace the sources of `control`; invalid ones are rejected.
    pub fn set_sources(&mut self, control: Control, sources: &[Source], keyboard_only: bool) -> Result<(), ConfigError> {
        for s in sources {
            validate_source(control, *s, keyboard_only)?;
        }
        let set = if keyboard_only { &mut self.keyboard_bindings } else { &mut self.bindings };
        set.retain(|b| b.control != control);
        for &source in sources {
            let b = Binding { source, control };
            if !set.contains(&b) {
                set.push(b);
            }
        }
        Ok(())
    }

    pub fn pad_inputs_of(&self, bit: u32) -> Vec<PadInput> {
        self.pad_bindings.iter().filter(|b| b.bit == bit).map(|b| b.input).collect()
    }

    pub fn set_pad_inputs(&mut self, bit: u32, inputs: &[PadInput]) {
        self.pad_bindings.retain(|b| b.bit != bit);
        for &input in inputs {
            let b = PadBinding { input, bit };
            if !self.pad_bindings.contains(&b) {
                self.pad_bindings.push(b);
            }
        }
    }

    /// Serialise the complete configuration (every action listed).
    pub fn to_toml_string(&self) -> String {
        use toml::{Table, Value};
        let mut root = Table::new();
        let mut device = Table::new();
        device.insert("mode".into(), Value::String(self.mode.as_str().into()));
        device.insert("controller".into(), Value::String(self.controllers[0].clone()));
        root.insert("device".into(), Value::Table(device));
        let mut players = Table::new();
        players.insert("count".into(), Value::Integer(self.players.clamp(1, MAX_PLAYERS) as i64));
        for p in 1..MAX_PLAYERS {
            players.insert(format!("player{}", p + 1), Value::String(self.controllers[p].clone()));
        }
        root.insert("players".into(), Value::Table(players));
        let bindings_table = |set: &[Binding], keyboard_only: bool| {
            let mut t = Table::new();
            for &(name, control) in ACTIONS {
                if keyboard_only && control.mouse_only() {
                    continue;
                }
                let v: Vec<Value> = set
                    .iter()
                    .filter(|b| b.control == control)
                    .filter_map(|b| b.source.to_config_string())
                    .map(Value::String)
                    .collect();
                t.insert(name.into(), Value::Array(v));
            }
            Value::Table(t)
        };
        root.insert("bindings".into(), bindings_table(&self.bindings, false));
        root.insert("keyboard_bindings".into(), bindings_table(&self.keyboard_bindings, true));
        let mut pads = Table::new();
        for (name, bit) in pad_button_actions() {
            let v: Vec<Value> = self.pad_inputs_of(bit).into_iter().map(|p| Value::String(p.to_config_string())).collect();
            pads.insert(name.into(), Value::Array(v));
        }
        root.insert("gamepad_bindings".into(), Value::Table(pads));
        let mut mouse = Table::new();
        mouse.insert("sensitivity".into(), Value::Float(self.mouse_sensitivity));
        mouse.insert("invert_x".into(), Value::Boolean(self.mouse_invert_x));
        mouse.insert("invert_y".into(), Value::Boolean(self.mouse_invert_y));
        mouse.insert("guest_mouse".into(), Value::Boolean(self.guest_mouse));
        root.insert("mouse".into(), Value::Table(mouse));
        let f = &self.features;
        let mut gp = Table::new();
        gp.insert("enabled".into(), Value::Boolean(self.gamepad));
        gp.insert("rumble".into(), Value::Boolean(f.rumble));
        gp.insert("rumble_strength".into(), Value::Float(round3(f.rumble_strength)));
        gp.insert("lightbar".into(), Value::Boolean(f.lightbar));
        gp.insert("adaptive_triggers".into(), Value::Boolean(f.adaptive_triggers));
        gp.insert("motion".into(), Value::Boolean(f.motion));
        gp.insert("touchpad".into(), Value::Boolean(f.touchpad));
        gp.insert("speaker_volume".into(), Value::Integer(f.speaker_volume as i64));
        gp.insert("vibration_intensity".into(), Value::Integer(f.vibration_intensity as i64));
        gp.insert("lightbar_color".into(), Value::String(f.lightbar_color.map(color_string).unwrap_or_default()));
        root.insert("gamepad".into(), Value::Table(gp));
        let mut sticks = Table::new();
        sticks.insert("left_mode".into(), Value::String(self.left_stick_mode.as_str().into()));
        sticks.insert("right_mode".into(), Value::String(self.right_stick_mode.as_str().into()));
        sticks.insert("swap".into(), Value::Boolean(self.swap_sticks));
        sticks.insert("left_deadzone".into(), Value::Float(round3(self.left_stick.deadzone)));
        sticks.insert("left_curve".into(), Value::Float(round3(self.left_stick.curve)));
        sticks.insert("right_deadzone".into(), Value::Float(round3(self.right_stick.deadzone)));
        sticks.insert("right_curve".into(), Value::Float(round3(self.right_stick.curve)));
        root.insert("sticks".into(), Value::Table(sticks));
        let mut trig = Table::new();
        trig.insert("deadzone".into(), Value::Float(round3(self.triggers.deadzone)));
        trig.insert("curve".into(), Value::Float(round3(self.triggers.curve)));
        root.insert("triggers".into(), Value::Table(trig));
        let mut motion = Table::new();
        motion.insert("tilt_angle".into(), Value::Float(round3(self.motion.tilt_angle)));
        motion.insert("tilt_time".into(), Value::Float(round3(self.motion.tilt_time)));
        motion.insert("mouse_sensitivity".into(), Value::Float(round3(self.motion.mouse_sensitivity)));
        root.insert("motion".into(), Value::Table(motion));
        let mut touch = Table::new();
        touch.insert("mouse_sensitivity".into(), Value::Float(round3(self.touch_mouse_sensitivity)));
        root.insert("touchpad".into(), Value::Table(touch));
        let mut profile = Table::new();
        profile.insert("enabled".into(), Value::Boolean(self.profile_enabled));
        root.insert("profile".into(), Value::Table(profile));
        let body = toml::to_string_pretty(&root).unwrap_or_default();
        format!("# rps5e input settings (written by the launcher; see rps5e-input.example.toml)\n\n{body}")
    }
}

fn round3(v: f32) -> f64 {
    (v as f64 * 1000.0).round() / 1000.0
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

fn err<T>(msg: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError(msg.into()))
}

pub fn find_action(name: &str) -> Option<Control> {
    ACTIONS.iter().find(|(n, _)| n.eq_ignore_ascii_case(name.trim())).map(|&(_, c)| c)
}

fn parse_mouse_button(v: &str) -> Option<MouseButton> {
    Some(match v.to_ascii_lowercase().as_str() {
        "left" => MouseButton::Left,
        "middle" => MouseButton::Middle,
        "right" => MouseButton::Right,
        "x1" | "back" => MouseButton::X1,
        "x2" | "forward" => MouseButton::X2,
        _ => return None,
    })
}

pub fn mouse_button_name(b: MouseButton) -> Option<&'static str> {
    Some(match b {
        MouseButton::Left => "Left",
        MouseButton::Middle => "Middle",
        MouseButton::Right => "Right",
        MouseButton::X1 => "X1",
        MouseButton::X2 => "X2",
        MouseButton::Other(_) => return None,
    })
}

/// Whether `source` may drive `control` (`keyboard_only`: the keyboard-only
/// set, which takes no mouse input).
pub fn validate_source(control: Control, source: Source, keyboard_only: bool) -> Result<(), ConfigError> {
    match source {
        Source::Key(k) => {
            if key_name(k).is_none() {
                return err(format!("key {k:?} cannot be bound"));
            }
            Ok(())
        }
        Source::Mouse(b) => {
            if keyboard_only {
                return err("keyboard-only bindings cannot use the mouse");
            }
            if matches!(control, Control::ToggleFullscreen | Control::ToggleMenu | Control::ShowControls) {
                return err(format!("{} can only use a keyboard key", control.name()));
            }
            if mouse_button_name(b).is_none() {
                return err("unsupported mouse button");
            }
            Ok(())
        }
        Source::Wheel(_) => {
            if keyboard_only {
                return err("keyboard-only bindings cannot use the mouse");
            }
            if !matches!(control, Control::Button(_)) {
                return err("mouse wheel can only map to a pad button");
            }
            Ok(())
        }
    }
}

/// Parse one `TYPE:VALUE` source for `control`.
pub fn parse_source(control: Control, source: &str) -> Result<Source, ConfigError> {
    parse_source_in(control, source, false)
}

fn parse_source_in(control: Control, source: &str, keyboard_only: bool) -> Result<Source, ConfigError> {
    let Some((ty, value)) = source.split_once(':') else {
        return err("source must use TYPE:VALUE format");
    };
    let value = value.trim();
    if value.is_empty() {
        return err("source value is empty");
    }
    let src = match ty.trim().to_ascii_uppercase().as_str() {
        "KEY" => match key_from_name(value) {
            Some(k) => Source::Key(k),
            None => return err(format!("unknown key name '{value}'")),
        },
        "MOUSE" => {
            if matches!(control, Control::ToggleFullscreen | Control::ToggleMenu | Control::ShowControls) {
                return err(format!("{} can only use a keyboard key", control.name()));
            }
            Source::Mouse(parse_mouse_button(value).ok_or_else(|| ConfigError(format!("unknown mouse button '{value}'")))?)
        }
        "WHEEL" => {
            if !matches!(control, Control::Button(_)) {
                return err("mouse wheel can only map to a pad button");
            }
            match value.to_ascii_uppercase().as_str() {
                "UP" => Source::Wheel(1),
                "DOWN" => Source::Wheel(-1),
                _ => return err("mouse wheel direction must be UP or DOWN"),
            }
        }
        _ => return err("source type must be KEY, MOUSE or WHEEL"),
    };
    validate_source(control, src, keyboard_only)?;
    Ok(src)
}

/// Parse one `BUTTON:<name>` controller source.
pub fn parse_pad_source(source: &str) -> Result<PadInput, ConfigError> {
    let Some((ty, value)) = source.split_once(':') else {
        return err("source must use BUTTON:<name> format");
    };
    if !ty.trim().eq_ignore_ascii_case("BUTTON") {
        return err("controller source type must be BUTTON");
    }
    PadInput::from_name(value.trim()).ok_or_else(|| ConfigError(format!("unknown controller button '{}'", value.trim())))
}

/// Parse the configuration file contents onto the built-in defaults.
pub fn parse_config(text: &str) -> Result<Config, ConfigError> {
    parse_config_onto(Config::default(), text)
}

/// Parse a `#RRGGBB` colour.
pub fn parse_color(text: &str) -> Option<[u8; 3]> {
    let hex = text.trim().strip_prefix('#')?;
    if hex.len() != 6 || !hex.is_ascii() {
        return None;
    }
    let c = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).ok();
    Some([c(0)?, c(2)?, c(4)?])
}

/// `#rrggbb` text of a colour.
pub fn color_string(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

fn number(v: &toml::Value) -> Option<f64> {
    v.as_float().or_else(|| v.as_integer().map(|i| i as f64))
}

fn get_bool(v: &toml::Value, name: &str) -> Result<bool, ConfigError> {
    v.as_bool().ok_or_else(|| ConfigError(format!("{name} must be a boolean")))
}

fn get_range(v: &toml::Value, name: &str, lo: f64, hi: f64) -> Result<f32, ConfigError> {
    number(v)
        .filter(|x| (lo..=hi).contains(x))
        .map(|x| x as f32)
        .ok_or_else(|| ConfigError(format!("{name} must be a number in {lo}..{hi}")))
}

fn get_string(v: &toml::Value, name: &str) -> Result<String, ConfigError> {
    v.as_str().map(|s| s.trim().to_string()).ok_or_else(|| ConfigError(format!("{name} must be a string")))
}

/// Parse the configuration file contents on top of `base`: listed settings
/// replace those of `base`, the others are kept (how a per-game profile
/// overlays the global file).
pub fn parse_config_onto(base: Config, text: &str) -> Result<Config, ConfigError> {
    let text = text.strip_prefix('\u{FEFF}').unwrap_or(text);
    let table: toml::Table = text.parse().map_err(|e: toml::de::Error| ConfigError(e.to_string()))?;
    let mut cfg = base;
    for (section, value) in &table {
        let Some(t) = value.as_table() else {
            return err(format!("'{section}' must be a [section]"));
        };
        let sec = section.to_ascii_lowercase();
        let unknown = |k: &str| err(format!("unknown setting '{sec}.{k}'"));
        match sec.as_str() {
            "bindings" => apply_bindings(&mut cfg.bindings, t, false)?,
            "keyboard_bindings" => apply_bindings(&mut cfg.keyboard_bindings, t, true)?,
            "gamepad_bindings" => apply_pad_bindings(&mut cfg.pad_bindings, t)?,
            "mouse" => {
                for (k, v) in t {
                    match k.to_ascii_lowercase().as_str() {
                        "sensitivity" => {
                            cfg.mouse_sensitivity = number(v)
                                .filter(|s| *s > 0.0)
                                .ok_or_else(|| ConfigError("mouse.sensitivity must be a positive number".into()))?
                        }
                        "guest_mouse" => cfg.guest_mouse = get_bool(v, "mouse.guest_mouse")?,
                        "invert_x" => cfg.mouse_invert_x = get_bool(v, "mouse.invert_x")?,
                        "invert_y" => cfg.mouse_invert_y = get_bool(v, "mouse.invert_y")?,
                        _ => return unknown(k),
                    }
                }
            }
            "gamepad" => {
                for (k, v) in t {
                    let f = &mut cfg.features;
                    match k.to_ascii_lowercase().as_str() {
                        "enabled" => cfg.gamepad = get_bool(v, "gamepad.enabled")?,
                        "rumble" => f.rumble = get_bool(v, "gamepad.rumble")?,
                        "rumble_strength" => f.rumble_strength = get_range(v, "gamepad.rumble_strength", 0.0, 2.0)?,
                        "lightbar" => f.lightbar = get_bool(v, "gamepad.lightbar")?,
                        "adaptive_triggers" => f.adaptive_triggers = get_bool(v, "gamepad.adaptive_triggers")?,
                        "motion" => f.motion = get_bool(v, "gamepad.motion")?,
                        "touchpad" => f.touchpad = get_bool(v, "gamepad.touchpad")?,
                        "speaker_volume" => {
                            f.speaker_volume = get_range(v, "gamepad.speaker_volume", 0.0, 100.0)?.round() as u32
                        }
                        "vibration_intensity" => {
                            f.vibration_intensity = get_range(v, "gamepad.vibration_intensity", 0.0, 100.0)?.round() as u32
                        }
                        "lightbar_color" => {
                            let s = get_string(v, "gamepad.lightbar_color")?;
                            f.lightbar_color = if s.is_empty() {
                                None
                            } else {
                                Some(parse_color(&s).ok_or_else(|| {
                                    ConfigError("gamepad.lightbar_color must be \"#RRGGBB\" or empty".into())
                                })?)
                            };
                        }
                        _ => return unknown(k),
                    }
                }
            }
            "device" => {
                for (k, v) in t {
                    match k.to_ascii_lowercase().as_str() {
                        "mode" => {
                            let s = get_string(v, "device.mode")?;
                            cfg.mode = DeviceMode::parse(&s).ok_or_else(|| {
                                ConfigError(format!(
                                    "device.mode '{s}' must be auto, keyboard_mouse, keyboard or controller"
                                ))
                            })?;
                        }
                        "controller" => cfg.controllers[0] = get_string(v, "device.controller")?,
                        _ => return unknown(k),
                    }
                }
            }
            "players" => {
                for (k, v) in t {
                    let key = k.to_ascii_lowercase();
                    match key.as_str() {
                        "count" => {
                            cfg.players = v
                                .as_integer()
                                .filter(|n| (1..=MAX_PLAYERS as i64).contains(n))
                                .ok_or_else(|| ConfigError(format!("players.count must be 1..{MAX_PLAYERS}")))?
                                as usize
                        }
                        "player2" | "player3" | "player4" => {
                            let idx = (key.as_bytes()[6] - b'1') as usize;
                            cfg.controllers[idx] = get_string(v, &format!("players.{key}"))?;
                        }
                        _ => return unknown(k),
                    }
                }
            }
            "sticks" => {
                for (k, v) in t {
                    let name = format!("sticks.{k}");
                    match k.to_ascii_lowercase().as_str() {
                        "left_deadzone" => cfg.left_stick.deadzone = get_range(v, &name, 0.0, 0.9)?,
                        "right_deadzone" => cfg.right_stick.deadzone = get_range(v, &name, 0.0, 0.9)?,
                        "left_curve" => cfg.left_stick.curve = get_range(v, &name, 0.2, 5.0)?,
                        "right_curve" => cfg.right_stick.curve = get_range(v, &name, 0.2, 5.0)?,
                        "left_mode" | "right_mode" => {
                            let s = get_string(v, &name)?;
                            let m = StickMode::parse(&s)
                                .ok_or_else(|| ConfigError(format!("{name} '{s}' must be keys, mouse or off")))?;
                            // A profile giving the mouse to one stick takes it
                            // off the other.
                            let stick = if k.eq_ignore_ascii_case("left_mode") { Stick::Left } else { Stick::Right };
                            cfg.set_stick_mode(stick, m);
                        }
                        "swap" => cfg.swap_sticks = get_bool(v, &name)?,
                        _ => return unknown(k),
                    }
                }
            }
            "triggers" => {
                for (k, v) in t {
                    let name = format!("triggers.{k}");
                    match k.to_ascii_lowercase().as_str() {
                        "deadzone" => cfg.triggers.deadzone = get_range(v, &name, 0.0, 0.9)?,
                        "curve" => cfg.triggers.curve = get_range(v, &name, 0.2, 5.0)?,
                        _ => return unknown(k),
                    }
                }
            }
            "motion" => {
                for (k, v) in t {
                    let name = format!("motion.{k}");
                    match k.to_ascii_lowercase().as_str() {
                        "tilt_angle" => cfg.motion.tilt_angle = get_range(v, &name, 1.0, 90.0)?,
                        "tilt_time" => cfg.motion.tilt_time = get_range(v, &name, 0.02, 2.0)?,
                        "mouse_sensitivity" => cfg.motion.mouse_sensitivity = get_range(v, &name, 0.05, 20.0)?,
                        _ => return unknown(k),
                    }
                }
            }
            "touchpad" => {
                for (k, v) in t {
                    let name = format!("touchpad.{k}");
                    match k.to_ascii_lowercase().as_str() {
                        "mouse_sensitivity" => cfg.touch_mouse_sensitivity = get_range(v, &name, 0.05, 20.0)?,
                        _ => return unknown(k),
                    }
                }
            }
            "profile" => {
                for (k, v) in t {
                    match k.to_ascii_lowercase().as_str() {
                        "enabled" => cfg.profile_enabled = get_bool(v, "profile.enabled")?,
                        _ => return unknown(k),
                    }
                }
            }
            _ => return err(format!("unknown section [{section}]")),
        }
    }
    Ok(cfg)
}

fn source_strings<'a>(name: &str, value: &'a toml::Value) -> Result<Vec<&'a str>, ConfigError> {
    match value {
        toml::Value::String(s) => Ok(vec![s.as_str()]),
        toml::Value::Array(a) => a
            .iter()
            .map(|v| v.as_str().ok_or_else(|| ConfigError(format!("{name}: sources must be strings"))))
            .collect(),
        _ => err(format!("{name}: expected \"TYPE:VALUE\" or a list of them")),
    }
}

/// Apply a bindings section. Listed actions replace their inputs. An
/// action the section does not list keeps its inputs, except those a
/// listed action now uses (one input, one action; Escape may both release
/// the mouse and press a button), so actions added after a file was
/// written start with their defaults where those are free.
fn apply_bindings(bindings: &mut Vec<Binding>, t: &toml::Table, keyboard_only: bool) -> Result<(), ConfigError> {
    let seen = apply_listed(bindings, t, keyboard_only)?;
    let used: Vec<Source> = bindings.iter().filter(|b| seen.contains(&b.control) && b.control != Control::ReleaseMouse).map(|b| b.source).collect();
    bindings.retain(|b| seen.contains(&b.control) || b.control == Control::ReleaseMouse || !used.contains(&b.source));
    Ok(())
}

/// Replace the inputs of every action `t` lists; returns those actions.
fn apply_listed(bindings: &mut Vec<Binding>, t: &toml::Table, keyboard_only: bool) -> Result<Vec<Control>, ConfigError> {
    let mut seen = Vec::new();
    for (name, value) in t {
        let Some(control) = find_action(name) else {
            return err(format!("unknown action '{name}'"));
        };
        if seen.contains(&control) {
            return err(format!("action '{name}' is listed twice"));
        }
        seen.push(control);
        let sources = source_strings(name, value)?;
        bindings.retain(|b| b.control != control);
        for s in sources {
            let source =
                parse_source_in(control, s, keyboard_only).map_err(|e| ConfigError(format!("{name} = \"{s}\": {e}")))?;
            let b = Binding { source, control };
            if !bindings.contains(&b) {
                bindings.push(b);
            }
        }
    }
    Ok(seen)
}

fn apply_pad_bindings(bindings: &mut Vec<PadBinding>, t: &toml::Table) -> Result<(), ConfigError> {
    let mut seen = Vec::new();
    for (name, value) in t {
        let Some(Control::Button(bit)) = find_action(name) else {
            return err(format!("'{name}' is not a pad button action"));
        };
        if seen.contains(&bit) {
            return err(format!("action '{name}' is listed twice"));
        }
        seen.push(bit);
        let sources = source_strings(name, value)?;
        bindings.retain(|b| b.bit != bit);
        for s in sources {
            let input = parse_pad_source(s).map_err(|e| ConfigError(format!("{name} = \"{s}\": {e}")))?;
            let b = PadBinding { input, bit };
            if !bindings.contains(&b) {
                bindings.push(b);
            }
        }
    }
    Ok(())
}

pub const FILE_NAME: &str = "rps5e-input.toml";
pub const PATH_ENV: &str = "RPS5E_INPUT_CONFIG";

/// `$RPS5E_CONFIG_DIR`, else `$XDG_CONFIG_HOME/rps5e` (`~/.config/rps5e`);
/// on Windows `%APPDATA%\rps5e`.
pub fn config_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("RPS5E_CONFIG_DIR").filter(|p| !p.is_empty()) {
        return Some(PathBuf::from(d));
    }
    #[cfg(windows)]
    if let Some(d) = std::env::var_os("APPDATA").filter(|p| !p.is_empty()) {
        return Some(ps5vm::appenv::app_dir(Path::new(&d)));
    }
    std::env::var_os("XDG_CONFIG_HOME")
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".config")))
        .map(|d| ps5vm::appenv::app_dir(&d))
}

/// Where the configuration is looked up: `$RPS5E_INPUT_CONFIG` (must
/// exist), else `./rps5e-input.toml`, else
/// `$XDG_CONFIG_HOME/rps5e/rps5e-input.toml` (`~/.config/...`).
pub fn config_candidates() -> (Option<PathBuf>, Vec<PathBuf>) {
    if let Some(p) = std::env::var_os(PATH_ENV).filter(|p| !p.is_empty()) {
        return (Some(PathBuf::from(p)), Vec::new());
    }
    let mut v = vec![PathBuf::from(FILE_NAME)];
    if let Some(dir) = config_dir() {
        v.push(dir.join(FILE_NAME));
    }
    (None, v)
}

/// Read and parse one file on top of `base`.
pub fn load_file_onto(base: Config, path: &Path) -> Result<Config, ConfigError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ConfigError(format!("cannot read input mapping '{}': {e}", path.display())))?;
    parse_config_onto(base, &text).map_err(|e| ConfigError(format!("invalid input mapping '{}': {e}", path.display())))
}

/// The global configuration file in use, if any.
pub fn global_config_path() -> Option<PathBuf> {
    let (explicit, candidates) = config_candidates();
    explicit.or_else(|| candidates.into_iter().find(|p| p.is_file()))
}

/// Load the global configuration; built-in defaults when no file exists.
pub fn load_config() -> Result<(Config, Option<PathBuf>), ConfigError> {
    match global_config_path() {
        Some(p) => Ok((load_file_onto(Config::default(), &p)?, Some(p))),
        None => Ok((Config::default(), None)),
    }
}

/// File-format name of a key (SDL style), `None` for keys the format has
/// no name for.
pub fn key_name(k: KeyCode) -> Option<String> {
    if let Some(i) = LETTERS.iter().position(|&x| x == k) {
        return Some(((b'A' + i as u8) as char).to_string());
    }
    if let Some(i) = DIGITS.iter().position(|&x| x == k) {
        return Some(i.to_string());
    }
    if let Some(i) = NUMPAD_DIGITS.iter().position(|&x| x == k) {
        return Some(format!("Keypad {i}"));
    }
    if let Some(i) = FKEYS.iter().position(|&x| x == k) {
        return Some(format!("F{}", i + 1));
    }
    use KeyCode::*;
    let s = match k {
        Enter => "Return",
        Escape => "Escape",
        Backspace => "Backspace",
        Tab => "Tab",
        Space => "Space",
        Minus => "Minus",
        Equal => "Equals",
        BracketLeft => "Left Bracket",
        BracketRight => "Right Bracket",
        Backslash => "Backslash",
        Semicolon => "Semicolon",
        Quote => "Apostrophe",
        Backquote => "Grave",
        Comma => "Comma",
        Period => "Period",
        Slash => "Slash",
        CapsLock => "CapsLock",
        PrintScreen => "PrintScreen",
        ScrollLock => "ScrollLock",
        Pause => "Pause",
        Insert => "Insert",
        Home => "Home",
        PageUp => "PageUp",
        Delete => "Delete",
        End => "End",
        PageDown => "PageDown",
        ArrowRight => "Right",
        ArrowLeft => "Left",
        ArrowDown => "Down",
        ArrowUp => "Up",
        NumLock => "Numlock",
        NumpadDivide => "Keypad /",
        NumpadMultiply => "Keypad *",
        NumpadSubtract => "Keypad -",
        NumpadAdd => "Keypad +",
        NumpadEnter => "Keypad Enter",
        NumpadDecimal => "Keypad .",
        ContextMenu => "Application",
        ControlLeft => "Left Ctrl",
        ShiftLeft => "Left Shift",
        AltLeft => "Left Alt",
        SuperLeft => "Left GUI",
        ControlRight => "Right Ctrl",
        ShiftRight => "Right Shift",
        AltRight => "Right Alt",
        SuperRight => "Right GUI",
        _ => return None,
    };
    Some(s.to_string())
}

/// Key name lookup: SDL scancode names and winit key code names, ignoring
/// case, spaces and underscores.
pub fn key_from_name(name: &str) -> Option<KeyCode> {
    let norm: String = name.chars().filter(|c| !matches!(c, ' ' | '_')).collect::<String>().to_ascii_lowercase();
    if norm.is_empty() {
        return None;
    }
    let b = norm.as_bytes();
    if b.len() == 1 && b[0].is_ascii_lowercase() {
        return Some(LETTERS[(b[0] - b'a') as usize]);
    }
    if b.len() == 1 && b[0].is_ascii_digit() {
        return Some(DIGITS[(b[0] - b'0') as usize]);
    }
    if let Some(rest) = norm.strip_prefix("key") {
        if rest.len() == 1 && rest.as_bytes()[0].is_ascii_lowercase() {
            return Some(LETTERS[(rest.as_bytes()[0] - b'a') as usize]);
        }
    }
    if let Some(rest) = norm.strip_prefix("digit") {
        if rest.len() == 1 && rest.as_bytes()[0].is_ascii_digit() {
            return Some(DIGITS[(rest.as_bytes()[0] - b'0') as usize]);
        }
    }
    for prefix in ["keypad", "numpad", "kp"] {
        if let Some(rest) = norm.strip_prefix(prefix) {
            if rest.len() == 1 && rest.as_bytes()[0].is_ascii_digit() {
                return Some(NUMPAD_DIGITS[(rest.as_bytes()[0] - b'0') as usize]);
            }
        }
    }
    if let Some(n) = norm.strip_prefix('f').and_then(|r| r.parse::<usize>().ok()) {
        if (1..=24).contains(&n) {
            return Some(FKEYS[n - 1]);
        }
    }
    NAMED.iter().find(|(names, _)| names.contains(&norm.as_str())).map(|&(_, k)| k)
}

const LETTERS: [KeyCode; 26] = {
    use KeyCode::*;
    [
        KeyA, KeyB, KeyC, KeyD, KeyE, KeyF, KeyG, KeyH, KeyI, KeyJ, KeyK, KeyL, KeyM, KeyN, KeyO, KeyP, KeyQ, KeyR,
        KeyS, KeyT, KeyU, KeyV, KeyW, KeyX, KeyY, KeyZ,
    ]
};

const DIGITS: [KeyCode; 10] = {
    use KeyCode::*;
    [Digit0, Digit1, Digit2, Digit3, Digit4, Digit5, Digit6, Digit7, Digit8, Digit9]
};

const NUMPAD_DIGITS: [KeyCode; 10] = {
    use KeyCode::*;
    [Numpad0, Numpad1, Numpad2, Numpad3, Numpad4, Numpad5, Numpad6, Numpad7, Numpad8, Numpad9]
};

const FKEYS: [KeyCode; 24] = {
    use KeyCode::*;
    [
        F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24,
    ]
};

/// Normalised names (lowercase, no spaces/underscores).
const NAMED: &[(&[&str], KeyCode)] = {
    use KeyCode::*;
    &[
        (&["return", "enter"], Enter),
        (&["escape", "esc"], Escape),
        (&["backspace"], Backspace),
        (&["tab"], Tab),
        (&["space"], Space),
        (&["-", "minus"], Minus),
        (&["=", "equals", "equal"], Equal),
        (&["[", "leftbracket", "bracketleft"], BracketLeft),
        (&["]", "rightbracket", "bracketright"], BracketRight),
        (&["\\", "backslash"], Backslash),
        (&[";", "semicolon"], Semicolon),
        (&["'", "apostrophe", "quote"], Quote),
        (&["`", "grave", "backquote"], Backquote),
        (&[",", "comma"], Comma),
        (&[".", "period"], Period),
        (&["/", "slash"], Slash),
        (&["capslock"], CapsLock),
        (&["printscreen"], PrintScreen),
        (&["scrolllock"], ScrollLock),
        (&["pause"], Pause),
        (&["insert"], Insert),
        (&["home"], Home),
        (&["pageup"], PageUp),
        (&["delete"], Delete),
        (&["end"], End),
        (&["pagedown"], PageDown),
        (&["right", "arrowright"], ArrowRight),
        (&["left", "arrowleft"], ArrowLeft),
        (&["down", "arrowdown"], ArrowDown),
        (&["up", "arrowup"], ArrowUp),
        (&["numlock", "numlockclear"], NumLock),
        (&["keypad/", "numpaddivide"], NumpadDivide),
        (&["keypad*", "numpadmultiply"], NumpadMultiply),
        (&["keypad-", "numpadsubtract"], NumpadSubtract),
        (&["keypad+", "numpadadd"], NumpadAdd),
        (&["keypadenter", "numpadenter"], NumpadEnter),
        (&["keypad.", "numpaddecimal"], NumpadDecimal),
        (&["application", "menu", "contextmenu"], ContextMenu),
        (&["leftctrl", "lctrl", "controlleft"], ControlLeft),
        (&["leftshift", "lshift", "shiftleft"], ShiftLeft),
        (&["leftalt", "lalt", "altleft"], AltLeft),
        (&["leftgui", "lgui", "superleft", "metaleft"], SuperLeft),
        (&["rightctrl", "rctrl", "controlright"], ControlRight),
        (&["rightshift", "rshift", "shiftright"], ShiftRight),
        (&["rightalt", "ralt", "altright"], AltRight),
        (&["rightgui", "rgui", "superright", "metaright"], SuperRight),
    ]
};
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_names() {
        assert_eq!(key_from_name("Return"), Some(KeyCode::Enter));
        assert_eq!(key_from_name("enter"), Some(KeyCode::Enter));
        assert_eq!(key_from_name("Left Shift"), Some(KeyCode::ShiftLeft));
        assert_eq!(key_from_name("ShiftLeft"), Some(KeyCode::ShiftLeft));
        assert_eq!(key_from_name("f"), Some(KeyCode::KeyF));
        assert_eq!(key_from_name("KeyF"), Some(KeyCode::KeyF));
        assert_eq!(key_from_name("7"), Some(KeyCode::Digit7));
        assert_eq!(key_from_name("Keypad 4"), Some(KeyCode::Numpad4));
        assert_eq!(key_from_name("F11"), Some(KeyCode::F11));
        assert_eq!(key_from_name("Keypad -"), Some(KeyCode::NumpadSubtract));
        assert_eq!(key_from_name("Up"), Some(KeyCode::ArrowUp));
        assert_eq!(key_from_name("F25"), None);
        assert_eq!(key_from_name("NoSuchKey"), None);
    }

    #[test]
    fn defaults_without_file() {
        let cfg = parse_config("").unwrap();
        assert_eq!(cfg, Config::default());
        assert_eq!(cfg.bindings.len(), 52);
        assert_eq!(cfg.mode, DeviceMode::Auto);
        assert_eq!(cfg.players, 1);
    }

    #[test]
    fn new_sections() {
        let cfg = parse_config(
            r#"
            [Device]
            Mode = "Keyboard+Mouse"
            controller = "DualSense"
            [players]
            count = 3
            player2 = "Xbox"
            [mouse]
            invert_y = true
            [gamepad]
            rumble = false
            rumble_strength = 0.5
            adaptive_triggers = false
            speaker_volume = 25
            [sticks]
            left_deadzone = 0.1
            right_curve = 2
            [triggers]
            deadzone = 0.05
            [keyboard_bindings]
            Square = "KEY:Z"
            [gamepad_bindings]
            Cross = "BUTTON:East"
            Circle = ["button:south", "BUTTON:RightPaddle1"]
            "#,
        )
        .unwrap();
        assert_eq!(cfg.mode, DeviceMode::KeyboardMouse);
        assert_eq!(cfg.controllers[0], "DualSense");
        assert_eq!((cfg.players, cfg.controllers[1].as_str()), (3, "Xbox"));
        assert!(cfg.mouse_invert_y && !cfg.mouse_invert_x);
        assert!(!cfg.features.rumble && !cfg.features.adaptive_triggers && cfg.features.lightbar);
        assert_eq!(cfg.features.rumble_strength, 0.5);
        assert_eq!((cfg.features.speaker_volume, cfg.features.vibration_intensity), (25, 100));
        assert_eq!(cfg.left_stick.deadzone, 0.1);
        assert_eq!(cfg.right_stick.curve, 2.0);
        assert_eq!(cfg.triggers.deadzone, 0.05);
        assert_eq!(cfg.sources_of(Control::Button(button::SQUARE), true), [Source::Key(KeyCode::KeyZ)]);
        assert_eq!(cfg.pad_inputs_of(button::CROSS), [PadInput(1)]);
        assert_eq!(cfg.pad_inputs_of(button::CIRCLE), [PadInput(0), PadInput(16)]);
        assert_eq!(cfg.pad_inputs_of(button::L1), [PadInput(9)]);
        // Player 1 is not a controller player in keyboard+mouse mode.
        assert_eq!(cfg.controller_players(), vec![(1, "Xbox"), (2, "")]);
    }

    #[test]
    fn new_sections_are_validated() {
        let bad = |s: &str| parse_config(s).unwrap_err().0;
        assert!(bad("[device]\nmode = \"joystick\"").contains("device.mode"));
        assert!(bad("[players]\ncount = 5").contains("players.count"));
        assert!(bad("[sticks]\nleft_deadzone = 0.95").contains("sticks.left_deadzone"));
        assert!(bad("[sticks]\nleft_curve = 0").contains("left_curve"));
        assert!(bad("[keyboard_bindings]\nSquare = \"MOUSE:Left\"").contains("mouse"));
        assert!(bad("[gamepad_bindings]\nLeftStickUp = \"BUTTON:South\"").contains("not a pad button"));
        assert!(bad("[gamepad_bindings]\nCross = \"KEY:A\"").contains("BUTTON"));
        assert!(bad("[gamepad_bindings]\nCross = \"BUTTON:Nope\"").contains("unknown controller button"));
        assert!(bad("[gamepad]\nrumble_strength = 3").contains("rumble_strength"));
    }

    #[test]
    fn device_modes_select_bindings() {
        let mut cfg = Config::default();
        let has_mouse = |b: &[Binding]| b.iter().any(|b| matches!(b.source, Source::Mouse(_) | Source::Wheel(_)));
        assert!(has_mouse(&cfg.active_bindings()));
        cfg.mode = DeviceMode::Keyboard;
        let kb = cfg.active_bindings();
        assert!(!has_mouse(&kb));
        // Every pad button is reachable from the keyboard alone.
        for (_, bit) in pad_button_actions() {
            if bit == button::TOUCH_PAD {
                continue; // TouchLeft / TouchRight
            }
            assert!(kb.iter().any(|b| b.control == Control::Button(bit)), "{bit:#x}");
        }
        cfg.mode = DeviceMode::Controller;
        assert_eq!(
            cfg.active_bindings().iter().map(|b| b.control).collect::<Vec<_>>(),
            [Control::ToggleFullscreen, Control::TogglePerfOverlay, Control::ToggleMenu, Control::ShowControls, Control::Create, Control::Mute]
        );
        assert_eq!(cfg.controller_players(), vec![(0, "")]);
    }

    #[test]
    fn serialisation_round_trips() {
        let mut cfg = Config::default();
        cfg.mode = DeviceMode::Controller;
        cfg.players = 2;
        cfg.controllers[1] = "Xbox \"Elite\"".into();
        cfg.mouse_invert_x = true;
        cfg.left_stick.deadzone = 0.125;
        cfg.features.motion = false;
        cfg.set_sources(Control::Button(button::CROSS), &[Source::Key(KeyCode::KeyX), Source::Wheel(-1)], false).unwrap();
        cfg.set_sources(Control::TouchLeft, &[], true).unwrap();
        cfg.set_pad_inputs(button::CROSS, &[PadInput(1), PadInput::LEFT_TRIGGER]);
        let text = cfg.to_toml_string();
        let back = parse_config(&text).unwrap();
        let norm = |c: &Config| {
            let mut c = c.clone();
            let k = |x: &Binding| format!("{x:?}");
            c.bindings.sort_by_key(k);
            c.keyboard_bindings.sort_by_key(k);
            c.pad_bindings.sort_by_key(|b| (b.bit, b.input));
            c
        };
        assert_eq!(norm(&back), norm(&cfg));
        // Defaults round-trip too.
        assert_eq!(norm(&parse_config(&Config::default().to_toml_string()).unwrap()), norm(&Config::default()));
    }

    #[test]
    fn set_sources_validates() {
        let mut cfg = Config::default();
        assert!(cfg.set_sources(Control::ToggleFullscreen, &[Source::Mouse(MouseButton::Left)], false).is_err());
        assert!(cfg.set_sources(Control::Button(button::CROSS), &[Source::Mouse(MouseButton::Left)], true).is_err());
        assert!(cfg.set_sources(Control::LeftStickUp, &[Source::Wheel(1)], false).is_err());
    }

    #[test]
    fn overlay_keeps_base_settings() {
        let base = parse_config("[mouse]\nsensitivity = 2\n[bindings]\nCross = \"KEY:J\"").unwrap();
        let cfg = parse_config_onto(base, "[device]\nmode = \"keyboard\"").unwrap();
        assert_eq!(cfg.mouse_sensitivity, 2.0);
        assert_eq!(cfg.sources_of(Control::Button(button::CROSS), false), [Source::Key(KeyCode::KeyJ)]);
        assert_eq!(cfg.mode, DeviceMode::Keyboard);
    }

    #[test]
    fn every_key_name_parses_back() {
        let all = LETTERS.iter().chain(&DIGITS).chain(&NUMPAD_DIGITS).chain(&FKEYS).copied().chain(NAMED.iter().map(|n| n.1));
        for k in all {
            let name = key_name(k).unwrap_or_else(|| panic!("{k:?} has no name"));
            assert_eq!(key_from_name(&name), Some(k), "{name}");
        }
        for (names, p) in PAD_INPUTS {
            assert_eq!(PadInput::from_name(names[0]), Some(*p));
            assert_eq!(parse_pad_source(&p.to_config_string()), Ok(*p));
        }
    }

    #[test]
    fn override_replaces_only_listed_actions() {
        let cfg = parse_config(
            r#"
            [bindings]
            cross = ["KEY:F", "key:space", "KEY:Space"]
            LeftStickLeft = "KEY:J"
            Square = "MOUSE:Left"
            R2 = "mouse:right"
            ToggleMouse = "MOUSE:Middle"
            L2 = "WHEEL:Down"
            TouchLeft = []

            [mouse]
            sensitivity = 2
            guest_mouse = true
            "#,
        )
        .unwrap();
        let of = |c: Control| cfg.bindings.iter().filter(|b| b.control == c).map(|b| b.source).collect::<Vec<_>>();
        assert_eq!(of(Control::Button(button::CROSS)), [Source::Key(KeyCode::KeyF), Source::Key(KeyCode::Space)]);
        assert_eq!(of(Control::LeftStickLeft), [Source::Key(KeyCode::KeyJ)]);
        assert_eq!(of(Control::Button(button::L2)), [Source::Wheel(-1)]);
        assert!(of(Control::TouchLeft).is_empty());
        // Untouched actions keep their built-in bindings, except the inputs
        // listed actions took: F (Triangle), the middle button (mouse
        // motion).
        assert_eq!(of(Control::LeftStickRight), [Source::Key(KeyCode::KeyD)]);
        assert!(of(Control::Button(button::TRIANGLE)).is_empty());
        assert!(of(Control::MouseMotion).is_empty());
        assert_eq!(of(Control::Shake), [Source::Key(KeyCode::Numpad5)]);
        assert_eq!(of(Control::Button(button::OPTIONS)), [Source::Key(KeyCode::Escape)]);
        assert_eq!(cfg.mouse_sensitivity, 2.0);
        assert!(cfg.guest_mouse);
        assert!(cfg.gamepad);
    }

    #[test]
    fn invalid_files_are_rejected() {
        let bad = |s: &str| parse_config(s).unwrap_err().0;
        assert!(bad("[bindings]\nJump = \"KEY:J\"").contains("unknown action 'Jump'"));
        assert!(bad("[bindings]\nCross = \"F\"").contains("TYPE:VALUE"));
        assert!(bad("[bindings]\nCross = \"KEY:\"").contains("empty"));
        assert!(bad("[bindings]\nCross = \"KEY:Nope\"").contains("unknown key name"));
        assert!(bad("[bindings]\nCross = \"PAD:A\"").contains("KEY, MOUSE or WHEEL"));
        assert!(bad("[bindings]\nCross = \"MOUSE:Side\"").contains("unknown mouse button"));
        assert!(bad("[bindings]\nToggleFullscreen = \"MOUSE:Left\"").contains("keyboard key"));
        assert!(bad("[bindings]\nLeftStickUp = \"WHEEL:Up\"").contains("pad button"));
        assert!(bad("[bindings]\nUp = \"WHEEL:Left\"").contains("UP or DOWN"));
        assert!(bad("[bindings]\nCross = \"KEY:A\"\ncross = \"KEY:B\"").contains("listed twice"));
        assert!(bad("[bindings]\nCross = 5").contains("expected"));
        assert!(bad("[mouse]\nspeed = 1").contains("unknown setting"));
        assert!(bad("[keys]\na = 1").contains("unknown section"));
        assert!(bad("Cross = KEY:F").len() > 0); // not TOML
    }

    #[test]
    fn example_file_parses_to_defaults() {
        let text = include_str!("../../../rps5e-input.example.toml");
        assert_eq!(parse_config(text).unwrap(), Config::default());
        // Uncommenting the documented defaults yields the same bindings set.
        let uncommented: String = text
            .lines()
            .map(|l| match l.strip_prefix("# ") {
                Some(r) if r.contains(" = ") && !r.starts_with(' ') => r.to_string(),
                _ => l.to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n");
        let cfg = parse_config(&uncommented).unwrap();
        let norm = |c: &Config| {
            let mut c = c.clone();
            let k = |x: &Binding| format!("{x:?}");
            c.bindings.sort_by_key(k);
            c.keyboard_bindings.sort_by_key(k);
            c.pad_bindings.sort_by_key(|b| (b.bit, b.input));
            c
        };
        assert_eq!(norm(&cfg), norm(&Config::default()));
    }

    #[test]
    fn motion_touchpad_and_swap_settings() {
        let cfg = parse_config(
            "[motion]\ntilt_angle = 45\ntilt_time = 0.5\nmouse_sensitivity = 2\n[touchpad]\nmouse_sensitivity = 0.5\n[sticks]\nswap = true\n",
        )
        .unwrap();
        assert_eq!((cfg.motion.tilt_angle, cfg.motion.tilt_time, cfg.motion.mouse_sensitivity), (45.0, 0.5, 2.0));
        assert_eq!(cfg.touch_mouse_sensitivity, 0.5);
        assert!(cfg.swap_sticks);
        let back = parse_config(&cfg.to_toml_string()).unwrap();
        assert_eq!((back.motion, back.swap_sticks, back.touch_mouse_sensitivity), (cfg.motion, true, 0.5));
        let bad = |s: &str| parse_config(s).unwrap_err().0;
        assert!(bad("[motion]\ntilt_angle = 120").contains("motion.tilt_angle"));
        assert!(bad("[motion]\nspeed = 1").contains("unknown setting"));
        assert!(bad("[touchpad]\nmouse_sensitivity = 0").contains("touchpad.mouse_sensitivity"));
        // Defaults: the new actions and the face buttons.
        let d = Config::default();
        let key = |c, kb| d.sources_of(c, kb).first().copied();
        assert_eq!(key(Control::Button(button::SQUARE), false), Some(Source::Key(KeyCode::KeyR)));
        assert_eq!(key(Control::Button(button::TRIANGLE), true), Some(Source::Key(KeyCode::KeyF)));
        assert_eq!(key(Control::Button(button::R2), false), Some(Source::Mouse(MouseButton::Left)));
        assert_eq!(key(Control::Button(button::L2), false), Some(Source::Mouse(MouseButton::Right)));
        assert_eq!(key(Control::MouseMotion, false), Some(Source::Mouse(MouseButton::Middle)));
        assert_eq!(key(Control::MouseMotion, true), None);
        assert_eq!(key(Control::TouchpadMode, true), Some(Source::Key(KeyCode::Numpad0)));
        assert_eq!(key(Control::Button(button::TOUCH_PAD), false), Some(Source::Key(KeyCode::Tab)));
        assert_eq!(d.sources_of(Control::ToggleMouse, false), [Source::Key(KeyCode::F9)]);
        // One input, one action in the defaults (Escape also releases the
        // mouse).
        for set in [&d.bindings, &d.keyboard_bindings] {
            for b in set.iter() {
                let clash = set.iter().any(|o| {
                    o.source == b.source && o.control != b.control && o.control != Control::ReleaseMouse && b.control != Control::ReleaseMouse
                });
                assert!(!clash, "{b:?}");
            }
        }
    }

    #[test]
    fn bom_is_accepted() {
        assert!(parse_config("\u{FEFF}[bindings]\nCross = \"KEY:F\"\n").is_ok());
    }

    #[test]
    fn lightbar_colour_and_cycle_actions() {
        assert_eq!(parse_color("#FF8000"), Some([255, 128, 0]));
        assert_eq!(parse_color("ff8000"), None);
        assert_eq!(parse_color("#ff80"), None);
        assert_eq!(color_string([1, 2, 255]), "#0102ff");
        let cfg = parse_config("[gamepad]\nlightbar_color = \"#10a0ff\"\n").unwrap();
        assert_eq!(cfg.features.lightbar_color, Some([0x10, 0xa0, 0xff]));
        assert_eq!(parse_config(&cfg.to_toml_string()).unwrap().features, cfg.features);
        assert_eq!(parse_config("[gamepad]\nlightbar_color = \"\"\n").unwrap().features.lightbar_color, None);
        assert!(parse_config("[gamepad]\nlightbar_color = \"red\"\n").is_err());
        assert_eq!(Config::default().to_toml_string().contains("lightbar_color = \"\""), true);
        let cfg = parse_config("[bindings]\nSpeakerVolume = \"KEY:Digit8\"\nTriggerEffectIntensity = \"MOUSE:X1\"\n").unwrap();
        assert_eq!(cfg.sources_of(Control::SpeakerVolume, false), [Source::Key(KeyCode::Digit8)]);
        assert!(Control::SpeakerVolume.is_hotkey());
        assert_eq!(Control::VibrationIntensity.cycled_setting(), Some(ps5hle::pad_controller::Setting::VibrationIntensity));
        assert_eq!(Control::ToggleMouse.cycled_setting(), None);
        let back = parse_config(&cfg.to_toml_string()).unwrap();
        assert_eq!(back.sources_of(Control::TriggerEffectIntensity, false), [Source::Mouse(MouseButton::X1)]);
    }

    #[test]
    fn stick_modes_in_files() {
        // Files without modes (written before them) keep working: left on
        // its keys, the mouse on the right stick, the stick keys kept.
        let old = "[sticks]\nleft_deadzone = 0.1\n[bindings]\nRightStickUp = \"KEY:U\"\n";
        let cfg = parse_config(old).unwrap();
        assert_eq!((cfg.left_stick_mode, cfg.right_stick_mode), (StickMode::Keys, StickMode::Mouse));
        assert_eq!(cfg.sources_of(Control::RightStickUp, false), [Source::Key(KeyCode::KeyU)]);
        let cfg = Config { mode: DeviceMode::KeyboardMouse, ..cfg };
        assert_eq!(cfg.mouse_stick(), Some(Stick::Right));
        assert!(cfg.active_bindings().iter().all(|b| b.control.stick() != Some(Stick::Right)));
        assert!(cfg.active_bindings().iter().any(|b| b.control == Control::LeftStickUp));
        // Round trip.
        let mut cfg = Config::default();
        cfg.set_stick_mode(Stick::Left, StickMode::Mouse);
        cfg.set_stick_mode(Stick::Right, StickMode::Off);
        let back = parse_config(&cfg.to_toml_string()).unwrap();
        assert_eq!((back.left_stick_mode, back.right_stick_mode), (StickMode::Mouse, StickMode::Off));
        // A profile giving the mouse to the left stick takes it off the
        // right one of the global file.
        let global = parse_config("[sticks]\nright_mode = \"mouse\"\n").unwrap();
        let game = parse_config_onto(global, "[sticks]\nleft_mode = \"MOUSE\"\n").unwrap();
        assert_eq!((game.left_stick_mode, game.right_stick_mode), (StickMode::Mouse, StickMode::Keys));
        assert!(parse_config("[sticks]\nleft_mode = \"wheel\"\n").is_err());
        // Resets: one stick back to its built-in settings, everything back
        // to keys on the left and the mouse on the right.
        let mut cfg = Config::default();
        cfg.set_stick_mode(Stick::Left, StickMode::Mouse);
        cfg.left_stick.deadzone = 0.3;
        cfg.right_stick.curve = 2.0;
        cfg.mouse_sensitivity = 4.0;
        cfg.mouse_invert_y = true;
        cfg.reset_stick(Stick::Left);
        assert_eq!((cfg.left_stick_mode, cfg.right_stick_mode), (StickMode::Keys, StickMode::Keys));
        assert_eq!(cfg.left_stick.deadzone, 0.0);
        assert_eq!((cfg.right_stick.curve, cfg.mouse_sensitivity), (2.0, 4.0), "the other stick and the mouse kept");
        cfg.reset_stick(Stick::Right);
        assert_eq!(cfg.right_stick_mode, StickMode::Mouse);
        assert_eq!((cfg.right_stick.curve, cfg.mouse_sensitivity, cfg.mouse_invert_y), (1.0, 1.0, false));
        cfg.set_stick_mode(Stick::Left, StickMode::Mouse);
        cfg.mouse_sensitivity = 3.0;
        let d = Config::default();
        assert_eq!((d.left_stick_mode, d.right_stick_mode, d.mouse_sensitivity), (StickMode::Keys, StickMode::Mouse, 1.0));
        // Controller mode: no mouse stick.
        assert_eq!(Config { mode: DeviceMode::Controller, ..Config::default() }.mouse_stick(), None);
    }
}
