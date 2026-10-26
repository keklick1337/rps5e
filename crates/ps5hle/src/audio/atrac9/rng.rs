// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Xorshift RNG used for band-extension noise fill.

#[derive(Clone, Debug)]
pub(super) struct Atrac9Rng {
    state_a: u16,
    state_b: u16,
    state_c: u16,
    state_d: u16,
}

impl Atrac9Rng {
    pub fn new(seed: u16) -> Self {
        let seed = seed as i32;
        let start_value = 0x4D93i32.wrapping_mul(seed ^ (seed >> 14));
        Self {
            state_a: 3i32.wrapping_sub(start_value) as u16,
            state_b: 2i32.wrapping_sub(start_value) as u16,
            state_c: 1i32.wrapping_sub(start_value) as u16,
            state_d: 0i32.wrapping_sub(start_value) as u16,
        }
    }

    pub fn next(&mut self) -> u16 {
        let d = self.state_d as i32;
        let t = (d ^ (d << 5)) as u16;
        self.state_d = self.state_c;
        self.state_c = self.state_b;
        self.state_b = self.state_a;
        let a = self.state_a as i32;
        let t = t as i32;
        self.state_a = (t ^ a ^ ((t ^ (a >> 5)) >> 4)) as u16;
        self.state_a
    }
}
