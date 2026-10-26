// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

// FLAT addresses in the shared (0x80000000) and private (0x70000000)
// apertures and in guest memory: the row table, kernels and expected values
// of the reference's FlatApertures test. Its expectations are the operation
// applied to the inputs (stores read back byte-exact at any alignment, an
// aperture access past the end of LDS dropped whole), not hardware rows.
// Pulled in with `include!`.

pub const SHARED_HIGH: u32 = 0x8000_0000;
pub const PRIVATE_HIGH: u32 = 0x7000_0000;
pub const LDS_DWORDS: u32 = 256;
pub const SCRATCH_DWORDS: u32 = 4;
pub const GUEST_BYTES: usize = 65536;
pub const GUEST_STORE_SLOT: u64 = 0x200;
pub const GUEST_ATOMIC_SLOT: u64 = 0x800;
const WAITCNT0: u32 = 0xbf8c_0070;
const END_PROGRAM: u32 = 0xbf81_0000;
/// v_lshlrev_b32 v1, 5, v0: each lane owns 8 dwords of the table and of
/// the output.
const SHIFT_LANE_BY5: u32 = 0x3402_0085;
const STORE_BYTE: u32 = 0x18;
const STORE_SHORT: u32 = 0x1a;
const STORE_DWORD: u32 = 0x1c;
const STORE_DWORDX2: u32 = 0x1d;
const LOAD_UBYTE: u32 = 0x08;
const LOAD_USHORT: u32 = 0x0a;
const LOAD_DWORD: u32 = 0x0c;
const LOAD_DWORDX2: u32 = 0x0d;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Shared,
    Private,
    Guest,
    /// Lane % 3: shared, private, guest.
    Mixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlatAtomic {
    Swap,
    Cmpswap,
    Add,
    Sub,
    Smin,
    Umin,
    Smax,
    Umax,
    And,
    Or,
    Xor,
    Inc,
    Dec,
    Fcmpswap,
    Fmin,
    Fmax,
}

const ALL_ATOMICS: [FlatAtomic; 16] = [
    FlatAtomic::Swap,
    FlatAtomic::Cmpswap,
    FlatAtomic::Add,
    FlatAtomic::Sub,
    FlatAtomic::Smin,
    FlatAtomic::Umin,
    FlatAtomic::Smax,
    FlatAtomic::Umax,
    FlatAtomic::And,
    FlatAtomic::Or,
    FlatAtomic::Xor,
    FlatAtomic::Inc,
    FlatAtomic::Dec,
    FlatAtomic::Fcmpswap,
    FlatAtomic::Fmin,
    FlatAtomic::Fmax,
];
/// The 32-bit FLAT atomics of the table (no float forms).
const ATOMIC32_OPS: usize = 13;
const ATOMIC_OPCODES32: [u32; 13] = [0x30, 0x31, 0x32, 0x33, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d];
const ATOMIC_OPCODES64: [u32; 16] = [0x50, 0x51, 0x52, 0x53, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f, 0x60];

#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub target: Target,
    pub atomic: bool,
    pub op: FlatAtomic,
    pub width: u32,
    pub offset: u32,
    pub wave: u32,
}

fn flat_word0(op: u32, glc: bool, offset: u32) -> u32 {
    0xdc00_0000 | (op << 18) | if glc { 0x1_0000 } else { 0 } | offset
}

fn flat_word1(vdst: u32, data: u32, address: u32) -> u32 {
    (vdst << 24) | (0x7d << 16) | (data << 8) | address
}

fn emit_flat(code: &mut Vec<u32>, op: u32, glc: bool, vdst: u32, data: u32, address: u32) {
    code.push(flat_word0(op, glc, 0));
    code.push(flat_word1(vdst, data, address));
}

/// buffer_load_dwordx4 v[vdst..], v1, s[4:7] offen offset:`offset`.
fn emit_table_load(code: &mut Vec<u32>, vdst: u32, offset: u32) {
    code.push(0xe038_1000 | offset);
    code.push(0x8001_0001 | (vdst << 8));
}

/// buffer_store_dword v`vdata`, v1, s[0:3] offen offset:`offset`.
fn emit_output_store(code: &mut Vec<u32>, vdata: u32, offset: u32) {
    code.push(0xe070_1000 | offset);
    code.push(0x8000_0001 | (vdata << 8));
}

fn store_op(width: u32) -> u32 {
    match width {
        8 => STORE_BYTE,
        16 => STORE_SHORT,
        _ => STORE_DWORD,
    }
}

fn load_op(width: u32) -> u32 {
    match width {
        8 => LOAD_UBYTE,
        16 => LOAD_USHORT,
        _ => LOAD_DWORD,
    }
}

fn kernel32(row: &Row) -> Vec<u32> {
    let mut code = vec![SHIFT_LANE_BY5];
    emit_table_load(&mut code, 2, 0);
    emit_table_load(&mut code, 6, 16);
    code.push(WAITCNT0);
    if row.atomic {
        emit_flat(&mut code, STORE_DWORD, false, 0, 6, 2);
        code.push(WAITCNT0);
        emit_flat(&mut code, ATOMIC_OPCODES32[row.op as usize], true, 10, 4, 2);
        code.push(WAITCNT0);
        emit_flat(&mut code, LOAD_DWORD, false, 11, 0, 2);
        code.push(WAITCNT0);
        emit_output_store(&mut code, 10, 0);
        emit_output_store(&mut code, 11, 4);
    } else {
        emit_flat(&mut code, store_op(row.width), false, 0, 4, 2);
        code.push(WAITCNT0);
        emit_flat(&mut code, load_op(row.width), false, 10, 0, 2);
        emit_flat(&mut code, LOAD_UBYTE, false, 11, 0, 2);
        code.push(flat_word0(LOAD_UBYTE, false, row.width / 8 - 1));
        code.push(flat_word1(12, 0, 2));
        code.push(WAITCNT0);
        emit_output_store(&mut code, 10, 0);
        emit_output_store(&mut code, 11, 4);
        emit_output_store(&mut code, 12, 8);
    }
    code.push(END_PROGRAM);
    code
}

fn kernel64(row: &Row) -> Vec<u32> {
    let mut code = vec![SHIFT_LANE_BY5];
    emit_table_load(&mut code, 2, 0);
    emit_table_load(&mut code, 6, 16);
    code.push(WAITCNT0);
    emit_flat(&mut code, STORE_DWORDX2, false, 0, 8, 2);
    code.push(WAITCNT0);
    emit_flat(&mut code, ATOMIC_OPCODES64[row.op as usize], true, 10, 4, 2);
    code.push(WAITCNT0);
    emit_flat(&mut code, LOAD_DWORDX2, false, 12, 0, 2);
    code.push(WAITCNT0);
    emit_output_store(&mut code, 10, 0);
    emit_output_store(&mut code, 11, 4);
    emit_output_store(&mut code, 12, 8);
    emit_output_store(&mut code, 13, 12);
    code.push(END_PROGRAM);
    code
}

pub fn kernel(row: &Row) -> Vec<u32> {
    if row.atomic && row.width == 64 {
        kernel64(row)
    } else {
        kernel32(row)
    }
}

pub fn operand_a(lane: u32) -> u32 {
    0x9e37_79b1u32.wrapping_mul(lane + 1) ^ 0x5a5a_0f0f
}

pub fn init_value(lane: u32) -> u32 {
    if lane % 2 != 0 {
        0x8000_0000 | (lane * 0x101)
    } else {
        lane * 0x35 + 7
    }
}

pub fn cmp_value(lane: u32) -> u32 {
    if lane % 2 != 0 {
        init_value(lane) + 1
    } else {
        init_value(lane)
    }
}

pub fn apply32(op: FlatAtomic, init: u32, a: u32, b: u32) -> u32 {
    match op {
        FlatAtomic::Swap => a,
        FlatAtomic::Cmpswap => {
            if init == b {
                a
            } else {
                init
            }
        }
        FlatAtomic::Add => init.wrapping_add(a),
        FlatAtomic::Sub => init.wrapping_sub(a),
        FlatAtomic::Smin => (init as i32).min(a as i32) as u32,
        FlatAtomic::Umin => init.min(a),
        FlatAtomic::Smax => (init as i32).max(a as i32) as u32,
        FlatAtomic::Umax => init.max(a),
        FlatAtomic::And => init & a,
        FlatAtomic::Or => init | a,
        FlatAtomic::Xor => init ^ a,
        FlatAtomic::Inc => {
            if init >= a {
                0
            } else {
                init + 1
            }
        }
        FlatAtomic::Dec => {
            if init == 0 || init > a {
                a
            } else {
                init - 1
            }
        }
        _ => init,
    }
}

fn pack(high: u32, low: u32) -> u64 {
    ((high as u64) << 32) | low as u64
}

fn is_float(op: FlatAtomic) -> bool {
    matches!(op, FlatAtomic::Fcmpswap | FlatAtomic::Fmin | FlatAtomic::Fmax)
}

fn int_init64(lane: u32) -> u64 {
    let high = if lane % 2 != 0 { 0x8000_0000 | (lane * 0x1357) } else { lane * 0x0a0b + 0x11 };
    pack(high, init_value(lane))
}

fn int_a64(lane: u32) -> u64 {
    pack(0x85eb_ca6bu32.wrapping_mul(lane + 1) ^ 0xc2b2_ae35, operand_a(lane))
}

fn int_cmp64(lane: u32) -> u64 {
    if lane % 2 != 0 {
        int_init64(lane).wrapping_add(1)
    } else {
        int_init64(lane)
    }
}

fn float_init(lane: u32) -> f64 {
    if lane % 2 == 0 {
        (lane + 1) as f64 * 3.25
    } else {
        -((lane + 1) as f64) * 7.5
    }
}

fn float_a(lane: u32) -> f64 {
    if lane % 2 == 0 {
        -((lane + 2) as f64) * 1.5
    } else {
        (lane + 3) as f64 * 2.75 - 2.0
    }
}

fn float_cmp(lane: u32) -> f64 {
    if lane % 2 == 0 {
        float_init(lane)
    } else {
        float_init(lane) + 1.0
    }
}

pub fn init_bits64(op: FlatAtomic, lane: u32) -> u64 {
    if is_float(op) {
        float_init(lane).to_bits()
    } else {
        int_init64(lane)
    }
}

pub fn operand_bits64(op: FlatAtomic, lane: u32) -> u64 {
    if is_float(op) {
        float_a(lane).to_bits()
    } else {
        int_a64(lane)
    }
}

pub fn cmp_bits64(op: FlatAtomic, lane: u32) -> u64 {
    if is_float(op) {
        float_cmp(lane).to_bits()
    } else {
        int_cmp64(lane)
    }
}

fn apply64(op: FlatAtomic, init: u64, a: u64, cmp: u64) -> u64 {
    match op {
        FlatAtomic::Swap => a,
        FlatAtomic::Cmpswap => {
            if init == cmp {
                a
            } else {
                init
            }
        }
        FlatAtomic::Add => init.wrapping_add(a),
        FlatAtomic::Sub => init.wrapping_sub(a),
        FlatAtomic::Smin => (init as i64).min(a as i64) as u64,
        FlatAtomic::Umin => init.min(a),
        FlatAtomic::Smax => (init as i64).max(a as i64) as u64,
        FlatAtomic::Umax => init.max(a),
        FlatAtomic::And => init & a,
        FlatAtomic::Or => init | a,
        FlatAtomic::Xor => init ^ a,
        FlatAtomic::Inc => {
            if init >= a {
                0
            } else {
                init + 1
            }
        }
        FlatAtomic::Dec => {
            if init == 0 || init > a {
                a
            } else {
                init - 1
            }
        }
        _ => init,
    }
}

pub fn expected64(op: FlatAtomic, lane: u32) -> u64 {
    if is_float(op) {
        let (init, a) = (float_init(lane), float_a(lane));
        return match op {
            FlatAtomic::Fcmpswap => {
                if init == float_cmp(lane) {
                    a.to_bits()
                } else {
                    init.to_bits()
                }
            }
            FlatAtomic::Fmin => init.min(a).to_bits(),
            _ => init.max(a).to_bits(),
        };
    }
    apply64(op, int_init64(lane), int_a64(lane), int_cmp64(lane))
}

pub fn lane_address(row: &Row, lane: u32, guest_base: u64) -> u64 {
    let target = match row.target {
        Target::Mixed => match lane % 3 {
            0 => Target::Shared,
            1 => Target::Private,
            _ => Target::Guest,
        },
        t => t,
    };
    let private_base = if row.width == 64 { 8 } else { 4 };
    match target {
        Target::Shared => ((SHARED_HIGH as u64) << 32) | (8 * lane + row.offset) as u64,
        Target::Private => ((PRIVATE_HIGH as u64) << 32) | (private_base + row.offset) as u64,
        _ => guest_base + 8 * lane as u64 + row.offset as u64,
    }
}

pub fn describe(row: &Row) -> String {
    let access = if row.atomic {
        format!("{}{:?}", if row.width == 64 { "flat_atomic64_" } else { "flat_atomic_" }, row.op)
    } else {
        format!("flat_store{}", row.width)
    };
    format!("{access} {:?} offset {} wave{}", row.target, row.offset, row.wave)
}

/// The per-lane table: address (2 dwords), then the operand, comparator
/// and initial value (1 dword each, or 2 for 64-bit atomics).
pub fn table(row: &Row, guest_base: u64) -> Vec<u32> {
    let wide = row.atomic && row.width == 64;
    let mut table = vec![0u32; row.wave as usize * 8];
    for lane in 0..row.wave {
        let t = &mut table[lane as usize * 8..lane as usize * 8 + 8];
        let address = lane_address(row, lane, guest_base);
        t[0] = address as u32;
        t[1] = (address >> 32) as u32;
        if wide {
            let (operand, cmp, init) = (operand_bits64(row.op, lane), cmp_bits64(row.op, lane), init_bits64(row.op, lane));
            t[2] = operand as u32;
            t[3] = (operand >> 32) as u32;
            t[4] = cmp as u32;
            t[5] = (cmp >> 32) as u32;
            t[6] = init as u32;
            t[7] = (init >> 32) as u32;
        } else {
            t[2] = operand_a(lane);
            t[3] = if row.atomic { cmp_value(lane) } else { 0 };
            t[4] = if row.atomic { init_value(lane) } else { 0 };
        }
    }
    table
}

/// Compare a row's output (8 dwords per lane) with its expected values.
pub fn check(row: &Row, output: &[u32], failures: &mut Vec<String>) {
    let name = describe(row);
    let mut fail = |m: String| failures.push(format!("{name}: {m}"));
    for lane in 0..row.wave {
        let o = &output[lane as usize * 8..lane as usize * 8 + 8];
        if row.atomic && row.width == 64 {
            let old = pack(o[1], o[0]);
            let init = init_bits64(row.op, lane);
            if old != init {
                fail(format!("lane {lane} returned {old:#018x}, expected {init:#018x}"));
            }
            let left = pack(o[3], o[2]);
            let expected = expected64(row.op, lane);
            if left != expected {
                fail(format!("lane {lane} left {left:#018x}, expected {expected:#018x}"));
            }
            continue;
        }
        let a = operand_a(lane);
        if row.atomic {
            let init = init_value(lane);
            if o[0] != init {
                fail(format!("lane {lane} returned {:#010x}, expected {init:#010x}", o[0]));
            }
            let expected = apply32(row.op, init, a, cmp_value(lane));
            if o[1] != expected {
                fail(format!("lane {lane} left {:#010x}, expected {expected:#010x}", o[1]));
            }
            continue;
        }
        if row.target == Target::Shared && 8 * lane + row.offset + row.width / 8 > LDS_DWORDS * 4 {
            if o[0] != 0 {
                fail(format!("lane {lane} read {:#010x} back from an access past the end of LDS, expected 0 (the store is dropped whole)", o[0]));
            }
            continue;
        }
        let expected = match row.width {
            8 => a & 0xff,
            16 => a & 0xffff,
            _ => a,
        };
        if o[0] != expected {
            fail(format!("lane {lane} read {:#010x} back, expected {expected:#010x}", o[0]));
        }
        if o[1] != a & 0xff {
            fail(format!("lane {lane} has first byte {:#010x} in memory, expected {:#010x}", o[1], a & 0xff));
        }
        let last = (a >> (8 * (row.width / 8 - 1))) & 0xff;
        if o[2] != last {
            fail(format!("lane {lane} has last byte {:#010x} in memory, expected {last:#010x}", o[2]));
        }
    }
}

pub fn rows() -> Vec<Row> {
    let mut rows = Vec::new();
    let single = [Target::Shared, Target::Private, Target::Guest];
    let stores = [(8, 0), (8, 3), (16, 0), (16, 1), (16, 3), (32, 0), (32, 2)];
    let row = |target, atomic, op, width, offset, wave| Row { target, atomic, op, width, offset, wave };
    for target in single {
        for (width, offset) in stores {
            rows.push(row(target, false, FlatAtomic::Swap, width, offset, 32));
        }
    }
    rows.push(row(Target::Mixed, false, FlatAtomic::Swap, 16, 3, 32));
    rows.push(row(Target::Mixed, false, FlatAtomic::Swap, 32, 2, 32));
    for (width, offset) in [(32, 773), (32, 774), (32, 776), (16, 775), (8, 775)] {
        rows.push(row(Target::Shared, false, FlatAtomic::Swap, width, offset, 32));
    }
    for target in single {
        for op in ALL_ATOMICS.iter().take(ATOMIC32_OPS) {
            rows.push(row(target, true, *op, 32, 0, 32));
        }
    }
    for target in single {
        for op in ALL_ATOMICS {
            rows.push(row(target, true, op, 64, 0, 32));
        }
    }
    rows.push(row(Target::Mixed, true, FlatAtomic::Add, 32, 0, 32));
    rows.push(row(Target::Mixed, true, FlatAtomic::Cmpswap, 32, 0, 32));
    rows.push(row(Target::Mixed, true, FlatAtomic::Add, 64, 0, 32));
    rows.push(row(Target::Mixed, true, FlatAtomic::Cmpswap, 64, 0, 32));
    rows.push(row(Target::Mixed, true, FlatAtomic::Fmax, 64, 0, 32));
    rows.push(row(Target::Mixed, false, FlatAtomic::Swap, 16, 3, 64));
    rows.push(row(Target::Mixed, false, FlatAtomic::Swap, 32, 2, 64));
    rows.push(row(Target::Mixed, true, FlatAtomic::Add, 32, 0, 64));
    rows.push(row(Target::Shared, true, FlatAtomic::Cmpswap, 32, 0, 64));
    rows.push(row(Target::Mixed, true, FlatAtomic::Add, 64, 0, 64));
    rows.push(row(Target::Shared, true, FlatAtomic::Fmin, 64, 0, 64));
    rows
}
