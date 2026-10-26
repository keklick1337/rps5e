// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libc pseudo-random generators, as the console's FreeBSD-derived libc
//! defines them:
//! - `rand`, `rand_r`, `srand`: the Park-Miller "minimal standard" step
//!   (`do_rand`), shared by the global state and the caller's state;
//! - `random`, `srandom`, `initstate`, `setstate`: the additive feedback
//!   generator of random(3), its state table in guest memory (a title may
//!   hand the table `initstate` returned back to `setstate`);
//! - `erand48`, `nrand48`, `jrand48`: the 48-bit linear congruential
//!   generator on a caller's three 16-bit seed words.
//!
//! All are used only when no loaded module exports them
//! (`register_fallback_fn`); the games' libc.prx does.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};
use std::sync::Mutex;

/// One step of the minimal standard generator (16807 * x mod 2^31 - 1,
/// Schrage's method); the state and the result are the step minus one.
fn good_rand(context: u32) -> u32 {
    let mut x = (context % 0x7FFF_FFFE) as i32 + 1;
    let (hi, lo) = (x / 127_773, x % 127_773);
    x = 16_807 * lo - 2_836 * hi;
    if x < 0 {
        x += 0x7FFF_FFFF;
    }
    (x - 1) as u32
}

// ---------------------------------------------------------------------------
// rand / rand_r / srand
// ---------------------------------------------------------------------------

static RAND_STATE: Mutex<u32> = Mutex::new(1);

fn cb_rand(_ctx: &mut CpuContext) -> HleResult {
    let mut s = RAND_STATE.lock().unwrap_or_else(|e| e.into_inner());
    *s = good_rand(*s);
    *s as HleResult
}

/// rand_r(unsigned* state): the rand step on the caller's state.
fn cb_rand_r(ctx: &mut CpuContext) -> HleResult {
    let state = ctx.rdi();
    let Some(value) = ctx.read_u32(state) else {
        log::error!("[libc] rand_r: unreadable state {state:#x}");
        return 0;
    };
    let next = good_rand(value);
    ctx.write_u32(state, next);
    next as HleResult
}

fn cb_srand(ctx: &mut CpuContext) -> HleResult {
    *RAND_STATE.lock().unwrap_or_else(|e| e.into_inner()) = ctx.rdi() as u32;
    0
}

// ---------------------------------------------------------------------------
// random / srandom / initstate / setstate
// ---------------------------------------------------------------------------

const MAX_TYPES: u32 = 5;
const TYPE_0: usize = 0;
/// Shuffles of the linear congruential (type 0) state after seeding.
const TYPE_0_SHUFFLES: u32 = 50;
/// Smallest state size in bytes for each type.
const BREAKS: [i64; 5] = [8, 32, 64, 128, 256];
const DEGREES: [u32; 5] = [0, 7, 15, 31, 63];
const SEPARATIONS: [u32; 5] = [0, 3, 1, 3, 1];

/// The library's default table: the type word (type 3, rear 0) and the 31
/// words of the state after `srandom(1)`.
const DEFAULT_TABLE: [u32; 32] = [
    3, 0x2cf41758, 0x27bb3711, 0x4916d4d1, 0x7b02f59f, 0x9b8e28eb, 0xc0e80269, 0x696f5c16, 0x878f1ff5, 0x52d9c07f,
    0x916a06cd, 0xb50b3a20, 0x2776970a, 0xee4eb2a6, 0xe94640ec, 0xb1d65612, 0x9d1ed968, 0x1043f6b7, 0xa3432a76,
    0x17eacbb9, 0x3c09e2eb, 0x04f8c2b3, 0x708a1f57, 0xee341814, 0x95d0e4d2, 0xb06f216c, 0x8bd2e72e, 0x8f7c38d7,
    0xcfc6a8fc, 0x02a59495, 0xa20d2a69, 0xe29d12d1,
];

/// The active table: guest address of its type word (the state words
/// follow it), type, degree, separation and the front and rear indices.
struct Feedback {
    table: u64,
    ty: usize,
    degree: u32,
    separation: u32,
    front: u32,
    rear: u32,
}

/// `table` 0: the default table, not yet placed in guest memory.
static RANDOM: Mutex<Feedback> = Mutex::new(Feedback { table: 0, ty: 3, degree: 31, separation: 3, front: 3, rear: 0 });

impl Feedback {
    fn word(&self, ctx: &CpuContext, i: u32) -> u32 {
        ctx.read_u32(self.table + 4 * i as u64).unwrap_or(0)
    }

    fn store(&self, ctx: &CpuContext, i: u32, v: u32) {
        ctx.write_u32(self.table + 4 * i as u64, v);
    }

    /// Place the default table in guest memory on first use.
    fn ensure_table(&mut self, ctx: &CpuContext) -> bool {
        if self.table != 0 {
            return true;
        }
        let Some(addr) = crate::pthread::alloc_guest_block(ctx, 4 * DEFAULT_TABLE.len() as u64) else {
            log::error!("[libc] random: no guest memory for the default state");
            return false;
        };
        let bytes: Vec<u8> = DEFAULT_TABLE.iter().flat_map(|w| w.to_le_bytes()).collect();
        if !ctx.write_bytes(addr, &bytes) {
            return false;
        }
        self.table = addr;
        true
    }

    fn next(&mut self, ctx: &CpuContext) -> u32 {
        if self.ty == TYPE_0 {
            let v = good_rand(self.word(ctx, 1));
            self.store(ctx, 1, v);
            return v;
        }
        let sum = self.word(ctx, 1 + self.front).wrapping_add(self.word(ctx, 1 + self.rear));
        self.store(ctx, 1 + self.front, sum);
        self.front += 1;
        if self.front >= self.degree {
            self.front = 0;
            self.rear += 1;
        } else {
            self.rear += 1;
            if self.rear >= self.degree {
                self.rear = 0;
            }
        }
        sum >> 1
    }

    fn seed(&mut self, ctx: &CpuContext, seed: u32) {
        self.store(ctx, 1, seed);
        let mut rounds = TYPE_0_SHUFFLES;
        if self.ty != TYPE_0 {
            for i in 1..self.degree {
                let prev = self.word(ctx, i);
                self.store(ctx, 1 + i, good_rand(prev));
            }
            self.front = self.separation;
            self.rear = 0;
            rounds = 10 * self.degree;
        }
        for _ in 0..rounds {
            self.next(ctx);
        }
    }

    /// Record the type and the rear index in the table's first word.
    fn save_type_word(&self, ctx: &CpuContext) {
        let word = if self.ty == TYPE_0 { 0 } else { MAX_TYPES * self.rear + self.ty as u32 };
        self.store(ctx, 0, word);
    }
}

fn random_state() -> std::sync::MutexGuard<'static, Feedback> {
    RANDOM.lock().unwrap_or_else(|e| e.into_inner())
}

/// random() -> 31-bit value.
fn cb_random(ctx: &mut CpuContext) -> HleResult {
    let mut st = random_state();
    if !st.ensure_table(ctx) {
        return 0;
    }
    st.next(ctx) as HleResult
}

/// srandom(seed): the low 32 bits of the seed count.
fn cb_srandom(ctx: &mut CpuContext) -> HleResult {
    let seed = ctx.rdi() as u32;
    let mut st = random_state();
    if st.ensure_table(ctx) {
        st.seed(ctx, seed);
    }
    0
}

/// initstate(seed, state, size) -> the previous state. The type is the
/// largest whose state fits in `size` bytes; under 8 bytes the call returns
/// NULL before touching anything (errno included).
fn cb_initstate(ctx: &mut CpuContext) -> HleResult {
    let (seed, state, size) = (ctx.rdi() as u32, ctx.rsi(), ctx.rdx() as i64);
    if size < BREAKS[0] {
        return 0;
    }
    if state == 0 {
        log::error!("[libc] initstate: null state");
        return 0;
    }
    let mut st = random_state();
    if !st.ensure_table(ctx) {
        return 0;
    }
    let previous = st.table;
    st.save_type_word(ctx);
    let ty = (0..BREAKS.len()).rev().find(|&t| size >= BREAKS[t]).unwrap_or(TYPE_0);
    *st = Feedback { table: state, ty, degree: DEGREES[ty], separation: SEPARATIONS[ty], front: 0, rear: 0 };
    st.seed(ctx, seed);
    st.save_type_word(ctx);
    previous as HleResult
}

/// setstate(state) -> the previous state, or NULL (and no change) when the
/// table's rear index is past its degree.
fn cb_setstate(ctx: &mut CpuContext) -> HleResult {
    let state = ctx.rdi();
    let Some(word) = (state != 0).then(|| ctx.read_u32(state)).flatten() else {
        log::error!("[libc] setstate: unreadable state {state:#x}");
        return 0;
    };
    let ty = (word % MAX_TYPES) as usize;
    let rear = word / MAX_TYPES;
    let degree = DEGREES[ty];
    if ty != TYPE_0 && rear >= degree {
        return 0;
    }
    let mut st = random_state();
    if !st.ensure_table(ctx) {
        return 0;
    }
    let previous = st.table;
    st.save_type_word(ctx);
    let separation = SEPARATIONS[ty];
    st.table = state;
    st.ty = ty;
    st.degree = degree;
    st.separation = separation;
    if ty != TYPE_0 {
        st.rear = rear;
        st.front = (rear + separation) % degree;
    }
    previous as HleResult
}

// ---------------------------------------------------------------------------
// erand48 / nrand48 / jrand48
// ---------------------------------------------------------------------------

/// Advance the caller's 48-bit state (x = (0x5DEECE66D * x + 11) mod 2^48);
/// None when the seed words cannot be read.
fn next_rand48(ctx: &CpuContext, seed: u64, function: &str) -> Option<u64> {
    let Some(b) = ctx.read_bytes(seed, 6) else {
        log::error!("[libc] {function}: unreadable seed {seed:#x}");
        return None;
    };
    let mut x = 0u64;
    for (i, w) in b.chunks_exact(2).enumerate() {
        x |= (u16::from_le_bytes([w[0], w[1]]) as u64) << (16 * i);
    }
    x = x.wrapping_mul(0x5_DEEC_E66D).wrapping_add(0xB) & ((1 << 48) - 1);
    ctx.write_bytes(seed, &x.to_le_bytes()[..6]);
    Some(x)
}

/// erand48(seed) -> [0, 1): the upper 32 bits of the new state over 2^32,
/// as the games' libc.prx returns it (FreeBSD keeps all 48 bits).
fn cb_erand48(ctx: &mut CpuContext) -> HleResult {
    let v = next_rand48(ctx, ctx.rdi(), "erand48").map_or(0.0, |x| (x >> 16) as f64 / (1u64 << 32) as f64);
    ctx.set_return_f64(v);
    0
}

/// nrand48(seed) -> [0, 2^31).
fn cb_nrand48(ctx: &mut CpuContext) -> HleResult {
    next_rand48(ctx, ctx.rdi(), "nrand48").map_or(0, |x| (x >> 17) as HleResult)
}

/// jrand48(seed) -> the 32 high bits of the state, zero-extended to the
/// 64-bit long as the console returns them (measured on a PS5).
fn cb_jrand48(ctx: &mut CpuContext) -> HleResult {
    next_rand48(ctx, ctx.rdi(), "jrand48").map_or(0, |x| (x >> 16) as u32 as HleResult)
}

type Export = (&'static str, &'static str, fn(&mut CpuContext) -> HleResult);

const EXPORTS: &[Export] = &[
    ("cpCOXWMgha0", "rand", cb_rand),
    ("dW3xsu3EgFI", "rand_r", cb_rand_r),
    ("VPbJwTCgME0", "srand", cb_srand),
    ("w1o05aHJT4c", "random", cb_random),
    ("sPC7XE6hfFY", "srandom", cb_srandom),
    ("IIUY-5hk-4k", "initstate", cb_initstate),
    ("woQta4WRpk0", "setstate", cb_setstate),
    ("Fncgcl1tnXg", "erand48", cb_erand48),
    ("3wcYIMz8LUo", "nrand48", cb_nrand48),
    ("M7KmRg9CERk", "jrand48", cb_jrand48),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fallback_fn(nid, name, "libc", cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::{Rdi, Rdx, Rsi};
    use ps5vm::{MemProtect, VirtualMemory};

    const BASE: u64 = 0x7B35_0000_0000;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// rand_r advances the caller's state with the rand step: from 1 it
    /// gives 33613, then 16806 (reference values).
    #[test]
    fn rand_r_steps_the_callers_state() {
        let vm = VirtualMemory::new();
        vm.map(BASE, 0x1000, MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_u32(BASE, 1);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_rand_r(&mut ctx), 33613);
        assert_eq!(ctx.read_u32(BASE), Some(33613));
        ctx.write_u32(BASE, 0);
        assert_eq!(cb_rand_r(&mut ctx), 16806);
        assert_eq!(ctx.read_u32(BASE), Some(16806));
        // The same generator as rand after srand with the same seed.
        for seed in [0u32, 1, 12345, 0x7FFF_FFFE, 0xFFFF_FFFF] {
            ctx.set(Rdi, seed as u64);
            cb_srand(&mut ctx);
            ctx.write_u32(BASE, seed);
            for _ in 0..8 {
                ctx.set(Rdi, BASE);
                let mine = cb_rand_r(&mut ctx);
                assert_eq!(mine, cb_rand(&mut ctx), "seed {seed:#x}");
            }
        }
        vm.unmap(BASE, 0x1000).unwrap();
    }

    #[cfg_attr(windows, allow(dead_code))]
    fn seed_is(ctx: &CpuContext, at: u64, w: [u16; 3]) -> bool {
        ctx.read_bytes(at, 6).unwrap() == [w[0].to_le_bytes(), w[1].to_le_bytes(), w[2].to_le_bytes()].concat()
    }

    /// The 48-bit generator: the reference's value and seed sequence, and
    /// the host's erand48 (cut to its upper 32 bits)/nrand48/jrand48 for
    /// 64 rounds.
    #[test]
    #[cfg(unix)]
    fn rand48_family() {
        const SEED_BASE: u64 = 0x7B36_0000_0000;
        let vm = VirtualMemory::new();
        vm.map(SEED_BASE, 0x1000, MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let seed = SEED_BASE;
        let set = |ctx: &CpuContext, at: u64, w: [u16; 3]| ctx.write_bytes(at, &[w[0].to_le_bytes(), w[1].to_le_bytes(), w[2].to_le_bytes()].concat());
        let er = |ctx: &mut CpuContext| f64::from_bits(ps5cpu::exec::with_import_xmm(&[], || cb_erand48(ctx)).1.unwrap() as u64);
        set(&ctx, seed, [0x330e, 0xabcd, 0x1234]);
        ctx.set(Rdi, seed);
        assert_eq!(er(&mut ctx), 0x657e_b725u64 as f64 / 2f64.powi(32));
        assert!(seed_is(&ctx, seed, [0x5101, 0xb725, 0x657e]));
        assert_eq!(cb_nrand48(&mut ctx), 1804928587);
        assert!(seed_is(&ctx, seed, [0x6378, 0x0c96, 0xd72a]));
        assert_eq!(cb_jrand48(&mut ctx), 1517566982);
        assert!(seed_is(&ctx, seed, [0x2a23, 0x3c06, 0x5a74]));
        assert_eq!(er(&mut ctx), 0x7253_4abfu64 as f64 / 2f64.powi(32));
        assert!(seed_is(&ctx, seed, [0x62f2, 0x4abf, 0x7253]));
        assert_eq!(cb_nrand48(&mut ctx), 684387517);
        assert!(seed_is(&ctx, seed, [0x8d15, 0xd97a, 0x5195]));
        assert_eq!(cb_jrand48(&mut ctx), (-487786166i32) as u32 as i64);
        assert!(seed_is(&ctx, seed, [0xeffc, 0xf94a, 0xe2ec]));
        set(&ctx, seed, [0, 0, 0]);
        assert_eq!(cb_nrand48(&mut ctx), 0);
        assert!(seed_is(&ctx, seed, [0x000b, 0, 0]), "a zero seed adds the increment");
        set(&ctx, seed, [0xffff; 3]);
        let v = er(&mut ctx);
        assert!((0.0..1.0).contains(&v));
        // The host's generators.
        set(&ctx, seed, [0x1234, 0x5678, 0x9abc]);
        let mut host: [libc::c_ushort; 3] = [0x1234, 0x5678, 0x9abc];
        for _ in 0..64 {
            let host_value = unsafe { libc::erand48(host.as_mut_ptr()) };
            assert_eq!(er(&mut ctx), (host_value * 2f64.powi(32)).floor() / 2f64.powi(32));
            assert_eq!(cb_nrand48(&mut ctx), unsafe { libc::nrand48(host.as_mut_ptr()) } as i64);
            assert_eq!(cb_jrand48(&mut ctx), unsafe { libc::jrand48(host.as_mut_ptr()) } as u32 as i64);
        }
        vm.unmap(SEED_BASE, 0x1000).unwrap();
    }

    /// random(3): the reference's sequences for the default state,
    /// srandom, every initstate size and setstate between tables (one test:
    /// the generator state is process-wide).
    #[test]
    fn random_state_tables() {
        const STATE_BASE: u64 = 0x7B37_0000_0000;
        let vm = VirtualMemory::new();
        vm.map(STATE_BASE, 0x10000, MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        let draw = |ctx: &mut CpuContext| -> [i64; 5] { std::array::from_fn(|_| cb_random(ctx)) };
        let seed_one = [1408992891, 1638182910, 83848093, 145962118, 627851256];
        let seed_12345 = [1052984060, 1639322404, 1971525204, 2097956667, 2012147915];
        let seed_42_by_type: [[i64; 5]; 5] = [
            [2060028338, 1164936638, 481681973, 1763071474, 962919018],
            [660273387, 1356916325, 287587791, 1132382070, 1497594867],
            [1015209508, 12952757, 1418769128, 2005808989, 1212998434],
            [745791173, 418148062, 949332769, 1480536337, 177711995],
            [996638292, 1209745284, 1148231710, 400392963, 1149215164],
        ];
        let seed_7_type_1 = [1271472800i64, 27686551, 1252069488, 1609001632, 28873941, 508017694];

        assert_eq!(draw(&mut ctx), seed_one, "unseeded random matches srandom(1)");
        let srandom = |ctx: &mut CpuContext, s: u64| {
            ctx.set(Rdi, s);
            cb_srandom(ctx);
        };
        srandom(&mut ctx, 1);
        assert_eq!(draw(&mut ctx), seed_one);
        srandom(&mut ctx, 0x1_0000_0001);
        assert_eq!(draw(&mut ctx), seed_one, "only the low 32 bits of the seed");
        srandom(&mut ctx, 12345);
        assert_eq!(draw(&mut ctx), seed_12345);

        let initstate = |ctx: &mut CpuContext, seed: u64, state: u64, size: i64| -> u64 {
            ctx.set(Rdi, seed);
            ctx.set(Rsi, state);
            ctx.set(Rdx, size as u64);
            cb_initstate(ctx) as u64
        };
        let setstate = |ctx: &mut CpuContext, state: u64| -> u64 {
            ctx.set(Rdi, state);
            cb_setstate(ctx) as u64
        };
        let sizes: [(i64, usize, u64); 10] =
            [(8, 0, 8), (31, 0, 8), (32, 1, 32), (63, 1, 32), (64, 2, 64), (127, 2, 64), (128, 3, 128), (255, 3, 128), (256, 4, 256), (1000, 4, 256)];
        let (mut default_table, mut active) = (0u64, 0u64);
        let mut at = STATE_BASE;
        for (size, ty, used) in sizes {
            ctx.write_bytes(at, &vec![0xA5u8; size as usize + 4]);
            let previous = initstate(&mut ctx, 42, at, size);
            if default_table == 0 {
                default_table = previous;
            } else {
                assert_eq!(previous, active, "initstate returns the previous state");
            }
            assert_eq!(ctx.read_u32(at), Some(ty as u32), "type word for {size} bytes");
            assert_eq!(draw(&mut ctx), seed_42_by_type[ty], "initstate(42) on {size} bytes");
            let tail = ctx.read_bytes(at + used, (size as u64 + 4 - used) as usize).unwrap();
            assert!(tail.iter().all(|&b| b == 0xA5), "initstate used more than {used} bytes of {size}");
            active = at;
            at += 0x800;
        }
        assert_ne!(default_table, 0);
        assert_eq!(ctx.read_u32(default_table), Some(5 * 5 + 3), "the default table holds its rear index");
        assert_eq!(setstate(&mut ctx, default_table), active);
        assert_eq!((cb_random(&mut ctx), cb_random(&mut ctx)), (1533042628, 1225903521), "the srandom(12345) sequence continues");

        let (small, large) = (STATE_BASE + 0x8000, STATE_BASE + 0x8100);
        initstate(&mut ctx, 7, small, 32);
        for &v in &seed_7_type_1[..3] {
            assert_eq!(cb_random(&mut ctx), v);
        }
        initstate(&mut ctx, 7, large, 64);
        assert_eq!(setstate(&mut ctx, small), large);
        assert_eq!(setstate(&mut ctx, large), small);
        assert_eq!(ctx.read_u32(small), Some(5 * 3 + 1), "setstate saves the old table's rear index");
        assert_eq!(setstate(&mut ctx, small), large);
        for &v in &seed_7_type_1[3..] {
            assert_eq!(cb_random(&mut ctx), v, "setstate resumes the sequence");
        }

        initstate(&mut ctx, 7, small, 32);
        cb_random(&mut ctx);
        let garbled = STATE_BASE + 0x8200;
        ctx.write_bytes(garbled, &[0u8; 32]);
        ctx.write_u32(garbled, 5 * 7 + 1);
        assert_eq!(setstate(&mut ctx, garbled), 0, "a rear index past the degree is refused");
        assert_eq!(cb_random(&mut ctx), seed_7_type_1[1], "a refused setstate keeps the state");

        let tiny = STATE_BASE + 0x8300;
        ctx.write_bytes(tiny, &[0x5A; 8]);
        for size in [7i64, 0, -1] {
            assert_eq!(initstate(&mut ctx, 1, tiny, size), 0, "initstate below 8 bytes");
        }
        assert_eq!(initstate(&mut ctx, 1, 0, 4), 0);
        assert_eq!(ctx.read_bytes(tiny, 8).unwrap(), vec![0x5A; 8], "a refused initstate wrote the state");
        assert_eq!(cb_random(&mut ctx), seed_7_type_1[2], "a refused initstate keeps the state");
        vm.unmap(STATE_BASE, 0x10000).unwrap();
    }
}
