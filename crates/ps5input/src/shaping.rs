// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Analog shaping of physical controller axes: radial stick dead zones,
//! trigger dead zones and response curves, applied before the values are
//! converted to the pad's 0..255 range.

/// Dead zone and response curve of one stick or trigger.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxisShape {
    /// Inner dead zone as a fraction of full deflection (0..0.9).
    pub deadzone: f32,
    /// Response exponent: 1 = linear, > 1 = finer control near the centre,
    /// < 1 = more sensitive near the centre.
    pub curve: f32,
}

impl AxisShape {
    pub const LINEAR: AxisShape = AxisShape { deadzone: 0.0, curve: 1.0 };
    pub const MAX_DEADZONE: f32 = 0.9;
    pub const CURVE_RANGE: (f32, f32) = (0.2, 5.0);

    pub fn is_identity(&self) -> bool {
        self.deadzone <= 0.0 && (self.curve - 1.0).abs() < 1e-6
    }

    /// Map a magnitude in 0..1 through the dead zone and curve; the dead
    /// zone edge maps to 0 and full deflection stays full.
    pub fn apply(&self, magnitude: f32) -> f32 {
        let m = magnitude.clamp(0.0, 1.0);
        let dz = self.deadzone.clamp(0.0, Self::MAX_DEADZONE);
        if m <= dz {
            return 0.0;
        }
        let t = (m - dz) / (1.0 - dz);
        let curve = self.curve.clamp(Self::CURVE_RANGE.0, Self::CURVE_RANGE.1);
        t.powf(curve).clamp(0.0, 1.0)
    }
}

impl Default for AxisShape {
    fn default() -> Self {
        Self::LINEAR
    }
}

/// Raw SDL stick axis (-32768..32767) to -1..1.
fn unit(v: i16) -> f32 {
    if v < 0 {
        v as f32 / 32768.0
    } else {
        v as f32 / 32767.0
    }
}

fn from_unit(v: f32) -> i16 {
    let v = v.clamp(-1.0, 1.0);
    if v < 0.0 {
        (v * 32768.0).round() as i16
    } else {
        (v * 32767.0).round() as i16
    }
}

/// Radial shaping of a stick (x, y raw SDL values): the dead zone and the
/// curve act on the deflection length, preserving its direction.
pub fn shape_stick(x: i16, y: i16, shape: &AxisShape) -> (i16, i16) {
    if shape.is_identity() {
        return (x, y);
    }
    let (fx, fy) = (unit(x), unit(y));
    let len = fx.hypot(fy);
    if len <= 0.0 {
        return (0, 0);
    }
    let out = shape.apply(len.min(1.0));
    let k = out / len;
    // Square gates report corners beyond the unit circle: keep the
    // direction and let each axis saturate.
    (from_unit(fx * k * len.max(1.0)), from_unit(fy * k * len.max(1.0)))
}

/// Trigger shaping (raw SDL 0..32767).
pub fn shape_trigger(v: i16, shape: &AxisShape) -> i16 {
    if shape.is_identity() {
        return v;
    }
    let t = v.max(0) as f32 / 32767.0;
    (shape.apply(t) * 32767.0).round() as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_untouched() {
        assert_eq!(shape_stick(1234, -32768, &AxisShape::LINEAR), (1234, -32768));
        assert_eq!(shape_trigger(77, &AxisShape::LINEAR), 77);
    }

    #[test]
    fn deadzone_is_radial_and_rescaled() {
        let s = AxisShape { deadzone: 0.2, curve: 1.0 };
        // Inside the dead zone: centred.
        assert_eq!(shape_stick(3000, 3000, &s), (0, 0));
        // Full deflection stays full.
        assert_eq!(shape_stick(32767, 0, &s), (32767, 0));
        assert_eq!(shape_stick(0, -32768, &s), (0, -32768));
        // Half deflection: (0.5 - 0.2) / 0.8 = 0.375.
        let (x, y) = shape_stick(16384, 0, &s);
        assert!((x as f32 / 32767.0 - 0.375).abs() < 0.01, "{x}");
        assert_eq!(y, 0);
        // Direction is kept.
        let (x, y) = shape_stick(-20000, 20000, &s);
        assert_eq!(x, -y);
    }

    #[test]
    fn curve_shapes_response() {
        let s = AxisShape { deadzone: 0.0, curve: 2.0 };
        assert!((s.apply(0.5) - 0.25).abs() < 1e-6);
        assert_eq!(s.apply(1.0), 1.0);
        assert_eq!(s.apply(0.0), 0.0);
        let soft = AxisShape { deadzone: 0.0, curve: 0.5 };
        assert!((soft.apply(0.25) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn trigger_deadzone() {
        let s = AxisShape { deadzone: 0.1, curve: 1.0 };
        assert_eq!(shape_trigger(2000, &s), 0);
        assert_eq!(shape_trigger(32767, &s), 32767);
        assert_eq!(shape_trigger(-5, &s), 0);
    }
}
