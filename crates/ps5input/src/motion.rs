// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Motion sensors driven from the keyboard and mouse: a simulated
//! controller pose that tilt keys, a shake key and the mouse move, turned
//! into the accelerometer and gyroscope readings a DualSense would report.
//!
//! Frame and units are those of [`ps5hle::pad::HostMotion`]: x to the
//! right, y up out of the face of the controller, z towards the player;
//! acceleration in m/s^2 (the specific force: +1 g along y while lying
//! flat), angular velocity in rad/s about those axes in the controller's
//! own frame. The pose is `yaw (about the world vertical) * pitch (about
//! x) * roll (about z)`:
//!
//! - tilting forward (the far edge down) is a negative pitch: the gyro
//!   reads x < 0 while it moves and gravity shows up as +z;
//! - tilting right (the right grip down) is a negative roll: z < 0 while
//!   moving, gravity towards -x;
//! - turning left is a positive yaw (y > 0); gravity does not change.
//!
//! Held tilt keys drive each angle towards its limit with a critically
//! damped spring (no jump at the start or the end of a move) and back to
//! level on release. The gyro reading is the body rate that turns the
//! previous pose into the new one, so integrating it (as the pad's
//! orientation filter does) gives back the same pose. A shake adds a
//! decaying side-to-side oscillation (rotation and linear acceleration)
//! that ends at rest after [`SHAKE_TIME`]. While mouse motion is held the
//! mouse turns the controller: right is a turn to the right, up tilts the
//! far edge up; the pose stays where the mouse left it.

use ps5hle::pad::HostMotion;

pub const GRAVITY: f32 = 9.80665;
/// Length of a shake burst, seconds.
pub const SHAKE_TIME: f32 = 0.5;
const SHAKE_FREQ: f32 = 6.0;
const SHAKE_ROLL: f32 = 14.0 * DEG;
const SHAKE_YAW: f32 = 8.0 * DEG;
/// Peak side-to-side acceleration of a shake, in g.
const SHAKE_ACCEL: f32 = 2.5;
/// Controller rotation per mouse count at sensitivity 1.
pub const MOUSE_RAD_PER_COUNT: f32 = 0.05 * DEG;
/// How far the mouse can tip the controller up or down.
const MOUSE_PITCH_LIMIT: f32 = 80.0 * DEG;
/// Integration sub-step of the tilt spring.
const STEP: f32 = 0.002;
/// Longer gaps (a stalled thread) are not simulated in one go.
const MAX_DT: f32 = 0.1;
const DEG: f32 = std::f32::consts::PI / 180.0;

pub type Vec3 = [f32; 3];
/// x, y, z, w.
pub type Quat = [f32; 4];

/// Motion settings (`[motion]` in the input file).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionSettings {
    /// Tilt of a held key, degrees.
    pub tilt_angle: f32,
    /// Time to reach the tilt (and to come back), seconds.
    pub tilt_time: f32,
    /// Mouse motion sensitivity (1 = 0.05 degrees per count).
    pub mouse_sensitivity: f32,
}

impl Default for MotionSettings {
    fn default() -> Self {
        Self { tilt_angle: 30.0, tilt_time: 0.25, mouse_sensitivity: 1.0 }
    }
}

/// The simulated controller pose and its sensor readings.
#[derive(Clone, Debug)]
pub struct MotionGen {
    settings: MotionSettings,
    /// Held directions of pitch (+ back), roll (+ left), yaw (+ left).
    held: [f32; 3],
    /// Key-driven angles and their rates (spring state).
    angle: [f32; 3],
    rate: [f32; 3],
    /// Mouse-driven yaw and pitch.
    mouse: [f32; 2],
    mouse_acc: [f32; 2],
    /// Time into the current shake.
    shake: Option<f32>,
    q: Quat,
    sample: HostMotion,
}

impl Default for MotionGen {
    fn default() -> Self {
        Self::new(MotionSettings::default())
    }
}

impl MotionGen {
    pub fn new(settings: MotionSettings) -> Self {
        Self {
            settings,
            held: [0.0; 3],
            angle: [0.0; 3],
            rate: [0.0; 3],
            mouse: [0.0; 2],
            mouse_acc: [0.0; 2],
            shake: None,
            q: IDENTITY,
            sample: REST,
        }
    }

    /// Held tilt directions, each -1, 0 or 1: pitch (+1 back, -1 forward),
    /// roll (+1 left, -1 right), yaw (+1 left, -1 right).
    pub fn set_held(&mut self, pitch: f32, roll: f32, yaw: f32) {
        self.held = [pitch.clamp(-1.0, 1.0), roll.clamp(-1.0, 1.0), yaw.clamp(-1.0, 1.0)];
    }

    /// Start (or restart) a shake.
    pub fn shake(&mut self) {
        self.shake = Some(0.0);
    }

    /// Mouse motion in counts (x right, y down).
    pub fn mouse(&mut self, dx: f64, dy: f64) {
        self.mouse_acc[0] += dx as f32;
        self.mouse_acc[1] += dy as f32;
    }

    /// The current reading.
    pub fn sample(&self) -> HostMotion {
        self.sample
    }

    /// The current pose (unit quaternion, controller to world).
    pub fn orientation(&self) -> Quat {
        self.q
    }

    /// Whether the pose or the readings differ from a controller lying
    /// still and flat.
    pub fn active(&self) -> bool {
        self.sample != REST || self.q != IDENTITY || self.shake.is_some() || self.held != [0.0; 3]
    }

    /// Advance by `dt` seconds.
    pub fn advance(&mut self, dt: f32) {
        if !(dt > 0.0) {
            return;
        }
        let dt = dt.min(MAX_DT);
        let s = self.settings;
        let k = MOUSE_RAD_PER_COUNT * s.mouse_sensitivity;
        let [mx, my] = std::mem::take(&mut self.mouse_acc);
        self.mouse[0] = wrap(self.mouse[0] - mx * k);
        self.mouse[1] = (self.mouse[1] - my * k).clamp(-MOUSE_PITCH_LIMIT, MOUSE_PITCH_LIMIT);

        // Critically damped spring towards the held angles: 95 % of the
        // way after `tilt_time`.
        let limit = s.tilt_angle.clamp(0.0, 90.0) * DEG;
        let w = 4.74 / s.tilt_time.max(0.01);
        let n = (dt / STEP).ceil().max(1.0) as usize;
        let h = dt / n as f32;
        for _ in 0..n {
            for i in 0..3 {
                let target = self.held[i] * limit;
                let acc = w * w * (target - self.angle[i]) - 2.0 * w * self.rate[i];
                self.rate[i] += acc * h;
                self.angle[i] += self.rate[i] * h;
            }
        }
        for i in 0..3 {
            if self.held[i] == 0.0 && self.angle[i].abs() < 1e-4 && self.rate[i].abs() < 1e-3 {
                self.angle[i] = 0.0;
                self.rate[i] = 0.0;
            }
        }

        let mut shake_roll = 0.0;
        let mut shake_yaw = 0.0;
        let mut linear = 0.0;
        if let Some(t) = self.shake.as_mut() {
            *t += dt;
            if *t >= SHAKE_TIME {
                self.shake = None;
            } else {
                let env = (1.0 - *t / SHAKE_TIME).powi(2);
                let phase = 2.0 * std::f32::consts::PI * SHAKE_FREQ * *t;
                shake_roll = SHAKE_ROLL * env * phase.sin();
                shake_yaw = SHAKE_YAW * env * (phase * 0.5).sin();
                // Side-to-side: the hands push the controller right and
                // left, so the reading leads the roll.
                linear = SHAKE_ACCEL * GRAVITY * env * phase.cos();
            }
        }

        let yaw = self.angle[2] + self.mouse[0] + shake_yaw;
        let pitch = self.angle[0] + self.mouse[1];
        let roll = self.angle[1] + shake_roll;
        let q = pose(yaw, pitch, roll);
        let gyro = body_rate(self.q, q, dt);
        let force = [linear, GRAVITY, 0.0];
        let accel = rotate(conjugate(q), force);
        self.q = q;
        self.sample = HostMotion { accel: clean(accel), angular_velocity: clean(gyro) };
        if self.sample.angular_velocity == [0.0; 3] && self.q == IDENTITY && linear == 0.0 {
            self.sample = REST;
        }
    }
}

/// A controller lying still and flat.
pub const REST: HostMotion = HostMotion { accel: [0.0, GRAVITY, 0.0], angular_velocity: [0.0; 3] };
pub const IDENTITY: Quat = [0.0, 0.0, 0.0, 1.0];

fn clean(v: Vec3) -> Vec3 {
    v.map(|c| if c.abs() < 1e-6 { 0.0 } else { c })
}

fn wrap(a: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    (a + std::f32::consts::PI).rem_euclid(tau) - std::f32::consts::PI
}

fn axis_angle(axis: Vec3, angle: f32) -> Quat {
    let (s, c) = (angle * 0.5).sin_cos();
    [axis[0] * s, axis[1] * s, axis[2] * s, c]
}

/// Controller-to-world rotation of a pose.
pub fn pose(yaw: f32, pitch: f32, roll: f32) -> Quat {
    if yaw == 0.0 && pitch == 0.0 && roll == 0.0 {
        return IDENTITY;
    }
    let q = mul(mul(axis_angle([0.0, 1.0, 0.0], yaw), axis_angle([1.0, 0.0, 0.0], pitch)), axis_angle([0.0, 0.0, 1.0], roll));
    normalize(q)
}

pub fn normalize(q: Quat) -> Quat {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    q.map(|c| c / n)
}

pub fn conjugate(q: Quat) -> Quat {
    [-q[0], -q[1], -q[2], q[3]]
}

/// q * r (Hamilton product).
pub fn mul(q: Quat, r: Quat) -> Quat {
    [
        q[3] * r[0] + q[0] * r[3] + q[1] * r[2] - q[2] * r[1],
        q[3] * r[1] - q[0] * r[2] + q[1] * r[3] + q[2] * r[0],
        q[3] * r[2] + q[0] * r[1] - q[1] * r[0] + q[2] * r[3],
        q[3] * r[3] - q[0] * r[0] - q[1] * r[1] - q[2] * r[2],
    ]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Rotate `v` by the unit quaternion `q`.
pub fn rotate(q: Quat, v: Vec3) -> Vec3 {
    let u = [q[0], q[1], q[2]];
    let t = cross(u, v).map(|c| 2.0 * c);
    let ut = cross(u, t);
    [v[0] + q[3] * t[0] + ut[0], v[1] + q[3] * t[1] + ut[1], v[2] + q[3] * t[2] + ut[2]]
}

/// The constant body-frame angular velocity that turns `from` into `to`
/// in `dt` (`to = from * exp(w dt / 2)`).
fn body_rate(from: Quat, to: Quat, dt: f32) -> Vec3 {
    let mut d = mul(conjugate(from), to);
    if d[3] < 0.0 {
        d = d.map(|c| -c);
    }
    let s = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if s < 1e-9 {
        return [0.0; 3];
    }
    let angle = 2.0 * s.atan2(d[3]);
    let k = angle / (s * dt);
    [d[0] * k, d[1] * k, d[2] * k]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn len(v: Vec3) -> f32 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    fn run(m: &mut MotionGen, secs: f32) -> Vec<HostMotion> {
        let n = (secs / 0.004).round() as usize;
        (0..n)
            .map(|_| {
                m.advance(0.004);
                m.sample()
            })
            .collect()
    }

    /// Body-rate integration as the pad's orientation filter does it.
    fn integrate(q: Quat, w: Vec3, dt: f32) -> Quat {
        let speed = len(w);
        let half = speed * dt * 0.5;
        let s = if speed > 0.0 { half.sin() / speed } else { 0.0 };
        normalize(mul(q, [w[0] * s, w[1] * s, w[2] * s, half.cos()]))
    }

    #[test]
    fn rest_is_flat_with_one_g() {
        let mut m = MotionGen::default();
        run(&mut m, 0.5);
        assert_eq!(m.sample(), REST);
        assert!((len(m.sample().accel) - GRAVITY).abs() < 1e-4);
        assert_eq!(m.orientation(), IDENTITY);
        assert!(!m.active());
    }

    #[test]
    fn quaternion_stays_normalized() {
        let mut m = MotionGen::default();
        let moves = [(1.0, 0.0, 0.0), (1.0, -1.0, 1.0), (0.0, 1.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0)];
        for (i, (p, r, y)) in moves.into_iter().enumerate() {
            m.set_held(p, r, y);
            if i == 2 {
                m.shake();
            }
            m.mouse(300.0 * i as f64, -120.0);
            for _ in 0..100 {
                m.advance(0.004);
                let q = m.orientation();
                let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
                assert!((n - 1.0).abs() < 1e-5, "{n}");
            }
        }
    }

    #[test]
    fn tilt_forward_signs() {
        let mut m = MotionGen::default();
        m.set_held(-1.0, 0.0, 0.0);
        let ramp = run(&mut m, 0.1);
        assert!(ramp.iter().all(|s| s.angular_velocity[0] < 0.0), "pitching forward turns about -x");
        run(&mut m, 1.5);
        let s = m.sample();
        assert_eq!(s.angular_velocity, [0.0; 3], "still at the held tilt");
        // Gravity: |a| = 1 g, leaning towards the player (+z).
        assert!((len(s.accel) - GRAVITY).abs() < 1e-3);
        let tilt = 30.0 * DEG;
        assert!((s.accel[2] - GRAVITY * tilt.sin()).abs() < 1e-2, "{:?}", s.accel);
        assert!((s.accel[1] - GRAVITY * tilt.cos()).abs() < 1e-2);
        // The face points away from the player.
        assert!(rotate(m.orientation(), [0.0, 1.0, 0.0])[2] < -0.4);
    }

    #[test]
    fn tilt_back_right_and_yaw_signs() {
        let mut m = MotionGen::default();
        m.set_held(1.0, 0.0, 0.0);
        assert!(run(&mut m, 0.05).iter().all(|s| s.angular_velocity[0] > 0.0));
        run(&mut m, 1.0);
        assert!(m.sample().accel[2] < -1.0);

        let mut m = MotionGen::default();
        m.set_held(0.0, -1.0, 0.0);
        assert!(run(&mut m, 0.05).iter().all(|s| s.angular_velocity[2] < 0.0), "right grip down turns about -z");
        run(&mut m, 1.0);
        assert!(m.sample().accel[0] < -1.0, "up leans to the left grip");

        let mut m = MotionGen::default();
        m.set_held(0.0, 1.0, 0.0);
        run(&mut m, 1.0);
        assert!(m.sample().accel[0] > 1.0);

        let mut m = MotionGen::default();
        m.set_held(0.0, 0.0, 1.0);
        assert!(run(&mut m, 0.05).iter().all(|s| s.angular_velocity[1] > 0.0), "turning left turns about +y");
        run(&mut m, 1.0);
        let a = m.sample().accel;
        assert!(a[0].abs() < 1e-3 && a[2].abs() < 1e-3 && (a[1] - GRAVITY).abs() < 1e-3, "yaw keeps gravity: {a:?}");
    }

    #[test]
    fn release_returns_to_level() {
        let mut m = MotionGen::default();
        m.set_held(-1.0, 1.0, 1.0);
        run(&mut m, 0.5);
        m.set_held(0.0, 0.0, 0.0);
        let back = run(&mut m, 0.1);
        assert!(back[0].angular_velocity[0] > 0.0, "coming back from forward turns about +x");
        run(&mut m, 2.0);
        assert_eq!(m.sample(), REST);
        assert_eq!(m.orientation(), IDENTITY);
        assert!(!m.active());
    }

    #[test]
    fn shake_decays_to_rest() {
        let mut m = MotionGen::default();
        m.shake();
        let burst = run(&mut m, SHAKE_TIME - 0.01);
        assert!(burst.iter().any(|s| len(s.accel) > 2.0 * GRAVITY), "a strong jolt");
        assert!(burst.iter().any(|s| len(s.angular_velocity) > 1.0), "and rotation");
        let peak_late = burst[burst.len() - 20..].iter().map(|s| len(s.angular_velocity)).fold(0.0, f32::max);
        let peak_early = burst[..20].iter().map(|s| len(s.angular_velocity)).fold(0.0, f32::max);
        assert!(peak_late < peak_early * 0.2, "decaying: {peak_early} -> {peak_late}");
        run(&mut m, 0.1);
        assert_eq!(m.sample(), REST);
        assert!(!m.active());
    }

    #[test]
    fn gyro_integrates_to_the_pose() {
        let mut m = MotionGen::default();
        let mut q = IDENTITY;
        let steps: [(f32, f32, f32, bool); 4] = [(-1.0, 1.0, 0.0, false), (1.0, -1.0, 1.0, true), (0.0, 0.0, -1.0, false), (0.0, 0.0, 0.0, false)];
        for (p, r, y, shake) in steps {
            m.set_held(p, r, y);
            if shake {
                m.shake();
            }
            for _ in 0..150 {
                m.advance(0.004);
                q = integrate(q, m.sample().angular_velocity, 0.004);
                // The accelerometer agrees with the pose when not shaking.
                let g = rotate(conjugate(m.orientation()), [0.0, GRAVITY, 0.0]);
                if m.shake.is_none() {
                    let a = m.sample().accel;
                    assert!((0..3).all(|i| (a[i] - g[i]).abs() < 1e-3));
                }
            }
            let o = m.orientation();
            let dot = (0..4).map(|i| q[i] * o[i]).sum::<f32>().abs();
            assert!(dot > 0.9999, "integrated gyro matches the pose ({dot})");
        }
    }

    #[test]
    fn mouse_turns_the_controller_and_stays() {
        let mut m = MotionGen::default();
        m.mouse(100.0, 0.0);
        m.advance(0.004);
        assert!(m.sample().angular_velocity[1] < 0.0, "mouse right turns right");
        m.mouse(0.0, -100.0);
        m.advance(0.004);
        assert!(m.sample().angular_velocity[0] > 0.0, "mouse up tips the far edge up");
        run(&mut m, 1.0);
        assert_eq!(m.sample().angular_velocity, [0.0; 3]);
        assert!(m.active(), "the pose stays");
        assert!(m.sample().accel[2] < 0.0);
        // Pitch is limited.
        m.mouse(0.0, -1.0e6);
        m.advance(0.004);
        assert!(m.sample().accel[1] > 0.0);
    }
}
