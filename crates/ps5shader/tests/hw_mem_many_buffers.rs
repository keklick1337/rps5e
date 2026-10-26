// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Many buffer descriptors loaded from a table with s_load_dwordx4: each
//! buffer gets a per-lane load / add constant / store. Ported for 52 and
//! 124 buffers (the reference used these two counts to exercise its two
//! descriptor-offset paths; here both just check the results).

include!("vectors/mem_common.rs");

const THREADS: u32 = 32;
const PUSH_BUFFERS: u32 = 52;
const DATA_BUFFERS: u32 = 124;
const REGION_WORDS: u32 = 2 * THREADS + 4;
const RANGE_BYTES: u32 = (REGION_WORDS + 4) * 4;
const TABLE_REGISTER: u32 = 0;
const DESCRIPTOR_REGISTER: u32 = 4;
const NULL_REGISTER: u32 = 0x7d;
const THREAD_VECTOR: u32 = 0;
const OFFSET_VECTOR: u32 = 1;
const DATA_VECTOR: u32 = 2;
const CONSTANT_ZERO: u32 = 0x80;
const CONSTANT_TWO: u32 = 0x82;
const LITERAL: u32 = 0xff;
const WAIT_SCALAR_LOADS: u32 = 0xbf8cc07f;
const WAIT_VECTOR_LOADS: u32 = 0xbf8c3f70;
const ENDPGM: u32 = 0xbf810000;

const DATA_BASE: u64 = 0x1_0000_0000;
const TABLE_BASE: u64 = 0x2_0000_0000;

fn vop2(opcode: u32, destination: u32, source0: u32, source1: u32) -> u32 {
    (opcode << 25) | (destination << 17) | (source1 << 9) | source0
}

fn increment(buffer: u32) -> u32 {
    (buffer + 1).wrapping_mul(0x0100_0193)
}

fn build_code(buffers: u32) -> Vec<u32> {
    const V_LSHLREV_B32: u32 = 0x1a;
    const V_ADD_NC_U32: u32 = 0x25;
    let s_load_dwordx4 = 0xf408_0000 | (DESCRIPTOR_REGISTER << 6) | (TABLE_REGISTER / 2);
    let buffer_load_dword = 0xe030_1000;
    let buffer_store_dword = 0xe070_1000 | (THREADS * 4);
    let operands = (CONSTANT_ZERO << 24) | ((DESCRIPTOR_REGISTER / 4) << 16) | (DATA_VECTOR << 8) | OFFSET_VECTOR;
    let mut code = vec![vop2(V_LSHLREV_B32, OFFSET_VECTOR, CONSTANT_TWO, THREAD_VECTOR)];
    for buffer in 0..buffers {
        code.extend_from_slice(&[
            s_load_dwordx4,
            (NULL_REGISTER << 25) | (buffer * 16),
            WAIT_SCALAR_LOADS,
            buffer_load_dword,
            operands,
            WAIT_VECTOR_LOADS,
            vop2(V_ADD_NC_U32, DATA_VECTOR, LITERAL, DATA_VECTOR),
            increment(buffer),
            buffer_store_dword,
            operands,
        ]);
    }
    code.push(ENDPGM);
    assert_eq!(code.len(), 2 + buffers as usize * 10);
    code
}

/// Word index of the buffer's region (regions are misaligned by
/// `buffer % 4` words).
fn buffer_base(buffer: u32) -> u32 {
    buffer * REGION_WORDS + buffer % 4
}

fn input_value(buffer: u32, tid: u32) -> u32 {
    0x4000_0000 | (buffer << 8) | tid
}

fn run(buffers: u32) {
    let code = build_code(buffers);
    let data_words = ((buffers + 1) * REGION_WORDS) as usize;
    let mut data = vec![0xdeadbeefu32; data_words];
    let mut table = vec![0u32; buffers as usize * 4];
    for buffer in 0..buffers {
        let base = buffer_base(buffer);
        for tid in 0..THREADS {
            data[(base + tid) as usize] = input_value(buffer, tid);
        }
        let address = DATA_BASE + base as u64 * 4;
        table[buffer as usize * 4..][..4].copy_from_slice(&[address as u32, ((address >> 32) as u32) & 0xffff, RANGE_BYTES, 0x3101_6fac]);
    }
    let user = vec![TABLE_BASE as u32, (TABLE_BASE >> 32) as u32];
    let d = data.clone();
    let setup = move |m: &mut Machine| {
        m.memory.write_words(DATA_BASE, &d);
        m.memory.write_words(TABLE_BASE, &table);
    };
    let out = dispatch(&code, 32, THREADS, user, &setup, &[(DATA_BASE, data_words)]).unwrap().remove(0);
    let mut c = Checker::new();
    for buffer in 0..buffers {
        let base = buffer_base(buffer) as usize;
        for tid in 0..THREADS {
            let t = tid as usize;
            c.eq(|| format!("{buffers} buffers: buffer {buffer} thread {tid} input"), out[base + t], input_value(buffer, tid));
            c.eq(|| format!("{buffers} buffers: buffer {buffer} thread {tid}"), out[base + THREADS as usize + t], input_value(buffer, tid).wrapping_add(increment(buffer)));
        }
    }
    c.finish();
}

#[test]
fn buffers_52() {
    run(PUSH_BUFFERS);
}

#[test]
fn buffers_124() {
    run(DATA_BUFFERS);
}
