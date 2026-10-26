// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Subroutines linked from outside the program: calls through code
//! addresses in user data and through tables of code addresses (found,
//! linked, then modelled as static calls and executed).

include!("vectors/scalar_common.rs");

use ps5shader::cfg::link::*;
use ps5shader::cfg::{Cfg, CfgError, Exit};
use ps5shader::{decode_blob, Program};

const ADDRESS: u64 = 0x10_0000;
const SITE: CallSite = CallSite { user_data_base: 0, user_data_count: 8, wave_size: 64 };

fn decode(words: &[u32]) -> Program {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    decode_blob(&bytes, ADDRESS).unwrap()
}

fn cfg_of(words: &[u32]) -> Result<Cfg, CfgError> {
    Cfg::build(&decode(words), None)
}

fn expect(needle: &str, r: Result<Cfg, CfgError>) {
    match r {
        Err(e) => assert!(e.to_string().contains(needle), "expected {needle}, got {e}"),
        Ok(cfg) => panic!("expected {needle}, built\n{cfg}"),
    }
}

fn block_at(cfg: &Cfg, pc: u32) -> usize {
    cfg.blocks.iter().position(|b| b.start_pc == pc && b.first < b.end).unwrap_or_else(|| panic!("no block at {pc:#x}\n{cfg}"))
}

/// Two calls through the code addresses in s[0:1] and s[2:3], in a loop
/// of three; each callee adds to v1.
const USER_DATA_CALLER: [u32; 13] =
    [0x7e020280, 0xbe940383, 0xbe8e0300, 0xbe8f0301, 0xbe8e210e, 0xbe8e0302, 0xbe8f0303, 0xbe8e210e, 0x80948114, 0xbf078014, 0xbf85fff7, 0xbf810000, 0x30306c73];
const ADD_FIVE: [u32; 2] = [0x4a020285, 0xbe80200e];
const ADD_SIXTY_FOUR: [u32; 2] = [0x4a0202c0, 0xbe80200e];
const ADD_ONE: [u32; 2] = [0x4a020281, 0xbe80200e];

fn user_data_pcs(code: &[u32], count: u32) -> Vec<u32> {
    find_user_data_calls(&decode(code), CallSite { user_data_count: count, ..SITE }).iter().map(|c| c.pc).collect()
}

#[test]
fn finds_user_data_calls() {
    let program = decode(&USER_DATA_CALLER);
    let calls = find_user_data_calls(&program, SITE);
    assert_eq!(calls, vec![UserDataCall { pc: 0x10, link: 14, user_data_register: 0 }, UserDataCall { pc: 0x1c, link: 14, user_data_register: 2 }]);
    expect("not statically resolvable", Cfg::build(&program, None));

    let mut overwritten = USER_DATA_CALLER;
    overwritten[1] = 0xbe800380;
    assert_eq!(user_data_pcs(&overwritten, 8), vec![0x1c]);
    let mut entered = USER_DATA_CALLER;
    entered[10] = 0xbf85fff8;
    assert_eq!(user_data_pcs(&entered, 8), vec![0x1c]);
    let mut looped = USER_DATA_CALLER;
    looped[8] = 0xbe820380;
    assert_eq!(user_data_pcs(&looped, 8), vec![0x10]);
    assert!(user_data_pcs(&[0xbe842104, 0xbf810000], 6).is_empty());
    assert_eq!(
        find_user_data_calls(&decode(&[0xbe8e2104, 0xbf810000]), CallSite { user_data_count: 6, ..SITE }),
        vec![UserDataCall { pc: 0, link: 14, user_data_register: 4 }]
    );
}

#[test]
fn user_data_callee_lengths() {
    assert_eq!(user_data_callee_length(&ADD_FIVE, 14), Ok(Some(2)));
    assert_eq!(user_data_callee_length(&[0x4a020285], 14), Ok(None));
    let err = |words: &[u32]| user_data_callee_length(words, 14).unwrap_err();
    assert!(err(&[0xbe901f00, 0xbe80200e]).contains("reads its program counter"));
    assert!(err(&[0xbe92210e, 0xbe80200e]).contains("makes a call"));
    assert!(err(&[0xbf810000]).contains("ends the program"));
    assert!(err(&[0xbe802010]).contains("jumps through another register pair"));
    assert!(err(&[0xbf840001, 0xbe80200e, 0x4a020285, 0xbe80200e]).contains("before code it branches to"));
}

#[test]
fn links_user_data_calls() {
    let program = decode(&USER_DATA_CALLER);
    let calls = find_user_data_calls(&program, SITE);
    let callees = vec![ADD_FIVE.to_vec(), ADD_SIXTY_FOUR.to_vec()];
    let linked = link_calls(&program, &USER_DATA_CALLER, &calls, &callees, &[], &[]).unwrap().code;
    assert_eq!(linked.len(), 17);
    assert_eq!((linked[4], linked[7], linked[16]), (0xbb0e0007, 0xbb0e0006, 0xbf9f0000));
    assert_eq!(&linked[12..14], &ADD_FIVE);
    assert_eq!(&linked[14..16], &ADD_SIXTY_FOUR);
    let mut reads_pc = USER_DATA_CALLER;
    reads_pc[1] = 0xbe941f00;
    let reads_pc_program = decode(&reads_pc);
    let c = find_user_data_calls(&reads_pc_program, SITE);
    assert_eq!(link_calls(&reads_pc_program, &reads_pc, &c, &callees, &[], &[]).unwrap().code.len(), 17);

    let cfg = cfg_of(&linked).unwrap();
    let exit = |pc: u32| cfg.blocks[block_at(&cfg, pc)].exit;
    assert_eq!(exit(0x08), Exit::Branch(block_at(&cfg, 0x30)));
    assert_eq!(exit(0x30), Exit::Branch(block_at(&cfg, 0x14)));
    assert_eq!(exit(0x14), Exit::Branch(block_at(&cfg, 0x38)));
    assert_eq!(exit(0x38), Exit::Branch(block_at(&cfg, 0x20)));

    let mut reads_link = linked.clone();
    reads_link[8] = 0xbe94030e;
    expect("escapes", cfg_of(&reads_link));
    let mut writes_inside = linked.clone();
    writes_inside[12] = 0xbe8e0300;
    expect("escapes", cfg_of(&writes_inside));
    let mut same_copy = linked.clone();
    same_copy[7] = 0xbb0e0004;
    expect("is shared with another call", cfg_of(&same_copy));
}

/// The caller of the driver execution test: the loop, then v1 stored per
/// lane through s[4:7].
const STORING_CALLER: [u32; 19] = [
    0x7e020280, 0xbe940383, 0xbe8e0300, 0xbe8f0301, 0xbe8e210e, 0xbe8e0302, 0xbe8f0303, 0xbe8e210e, 0x80948114, 0xbf078014, 0xbf85fff7, 0x34040082,
    0xe0701000, 0x80010102, 0xbf810000, 0xbf9f0000, 0x30306c73, 0x00000040, 0x00000008,
];

#[test]
fn linked_user_data_calls_execute() {
    let program = decode(&STORING_CALLER);
    let calls = find_user_data_calls(&program, SITE);
    assert_eq!(calls.len(), 2);
    for (first, second, want) in [(&ADD_FIVE, &ADD_SIXTY_FOUR, 3 * (5 + 64)), (&ADD_ONE, &ADD_SIXTY_FOUR, 3 * 65), (&ADD_SIXTY_FOUR, &ADD_SIXTY_FOUR, 3 * 128)] {
        let linked = link_calls(&program, &STORING_CALLER, &calls, &[first.to_vec(), second.to_vec()], &[], &[]).unwrap().code;
        for wave in [32, 64] {
            let mut ud = user_data([0x1000, 0, 0x2000, 0], raw_desc(OUT_BASE, 64 * 4), 8);
            ud[..4].copy_from_slice(&[0x1000, 0, 0x2000, 0]);
            let out = Dispatch::new(&linked, wave, 64, ud).output(OUT_BASE, 64).run().unwrap().remove(0);
            assert!(out.iter().all(|&v| v == want), "wave{wave}: {out:?}, expected {want}");
        }
    }
}

/// s_lshl_b32 s20, s20, 4; s_load_dwordx4 s[16:19], s[12:13], 0x270;
/// s_buffer_load_dwordx2 s[8:9], s[16:19], s20; s[14:15] = s[8:9];
/// s_swappc_b64 s[14:15], s[14:15].
const TABLE_CALLER: [u32; 11] = [0x8f148414, 0xf4080406, 0xfa000270, 0xbf8cc07f, 0xf4280208, 0x28000000, 0xbf8cc07f, 0xbe8e0308, 0xbe8f0309, 0xbe8e210e, 0xbf810000];

#[test]
fn finds_table_calls() {
    let site = CallSite { user_data_count: 16, ..SITE };
    let program = decode(&TABLE_CALLER);
    let calls = find_table_calls(&program, site);
    assert_eq!(
        calls,
        vec![TableCall { pc: 0x24, link: 14, target_register: 14, user_data_register: 12, descriptor_offset: 0x270, entry_offset: 0, entry_stride: 16 }]
    );
    assert!(find_user_data_calls(&program, site).is_empty());
    let table = |word: usize, value: u32, count: u32| {
        let mut code = TABLE_CALLER;
        code[word] = value;
        find_table_calls(&decode(&code), CallSite { user_data_count: count, ..SITE })
    };
    assert!(table(0, 0x8f148114, 16).is_empty());
    assert!(table(2, 0x28000270, 16).is_empty());
    assert!(table(8, 0xbe8f030a, 16).is_empty());
    assert!(table(10, 0xbf810000, 12).is_empty());
    assert_eq!(table(5, 0x28000008, 16).first().map(|c| c.entry_offset), Some(8));
    assert!(table(0, 0xbe940380, 16).is_empty());
    let mut reads_lane = vec![0xd4c2000b, 0x00020300];
    reads_lane.extend_from_slice(&TABLE_CALLER);
    let p = decode(&reads_lane);
    assert_eq!(find_table_calls(&p, CallSite { user_data_count: 16, wave_size: 32, ..SITE }).len(), 1);
    assert!(find_table_calls(&p, CallSite { user_data_count: 16, wave_size: 64, ..SITE }).is_empty());
    expect("not statically resolvable", Cfg::build(&program, None));
}

#[test]
fn links_table_calls() {
    let site = CallSite { user_data_count: 16, ..SITE };
    let program = decode(&TABLE_CALLER);
    let calls = find_table_calls(&program, site);
    let first = vec![0xbf068000, 0x4a020285, 0xbe80200e];
    let second = vec![0x4a0202c0, 0xbf068001, 0xbe80200e];
    let callees = vec![vec![TableCallee { target: 0x11_2233_4455, code: first.clone() }, TableCallee { target: 0x66_7788_99aa, code: second.clone() }]];
    let linked = link_calls(&program, &TABLE_CALLER, &[], &[], &calls, &callees).unwrap();
    assert_eq!(linked.fallbacks, vec![50 * 4]);
    let w = &linked.code;
    assert_eq!(w.len(), 59);
    assert_eq!((w[9], w[11], w[58]), (0xbf820001, 0xbf850010, 0xbf9f0000));
    assert_eq!(&w[12..19], &[0xbf07ff0e, 0x22334455, 0xbf850004, 0xbf07ff0f, 0x11, 0xbf850001, 0xbf82001b]);
    assert_eq!((w[20], w[26], w[27], w[34], w[35], w[44], w[45]), (0x778899aa, 0xbf078080, 0xbf820016, 0xbf068080, 0xbf82000a, 0xbf068080, 0xbf820004));
    // The fallback (an unmatched target) ends the wave.
    assert_eq!(&w[46..52], &[0xbb0e0005, 0xbf82ffda, 0xbb0e0006, 0xbf82ffd8, 0xbf810000, 0xbf82ffd6]);
    assert_eq!(&w[52..55], &first[..]);
    assert_eq!(&w[55..58], &second[..]);
    let cfg = cfg_of(w).unwrap();
    let exit = |word: u32| cfg.blocks[block_at(&cfg, word * 4)].exit;
    assert_eq!(exit(46), Exit::Branch(block_at(&cfg, 52 * 4)));
    assert_eq!(exit(52), Exit::Branch(block_at(&cfg, 47 * 4)));
    assert_eq!(exit(48), Exit::Branch(block_at(&cfg, 55 * 4)));
    assert_eq!(exit(55), Exit::Branch(block_at(&cfg, 49 * 4)));

    let large = vec![0xbf800000u32; 0x8000];
    let far = vec![vec![TableCallee { target: 0x10, code: large }, TableCallee { target: 0x20, code: first }]];
    assert!(link_calls(&program, &TABLE_CALLER, &[], &[], &calls, &far).unwrap_err().contains("out of s_call_b64 range"));
}

/// The driver test's caller: s[0:1] points at a buffer descriptor of a
/// table with two 16-byte records holding code addresses, s2 selects the
/// record, s3 the SCC the callee sees; the callee sets v1 by SCC and v1 is
/// stored per lane through s[4:7].
const TABLE_STORING_CALLER: [u32; 20] = [
    0x7e020280, 0xf4080400, 0xfa000000, 0x8f148402, 0xbf8cc07f, 0xf4280208, 0x28000000, 0xbf8cc07f, 0xbe8e0308, 0xbe8f0309, 0xbf068103, 0xbe8e210e,
    0x34040082, 0xe0701000, 0x80010102, 0xbf810000, 0xbf9f0000, 0x30306c73, 0x00000040, 0x00000008,
];
const SELECT_SIXTEEN: [u32; 3] = [0x85159091, 0x7e020215, 0xbe80200e];
const SELECT_THIRTY_TWO: [u32; 3] = [0x8515a0a1, 0x7e020215, 0xbe80200e];

#[test]
fn linked_table_calls_execute() {
    const DESCRIPTORS: u64 = 0x3_0000_0000;
    const TABLE: u64 = 0x3_0000_1000;
    const FIRST: u64 = 0x4_0000_0000;
    const SECOND: u64 = 0x4_0000_2000;
    let mut memory = Memory::default();
    memory.write_words(FIRST, &SELECT_SIXTEEN);
    memory.write_words(SECOND, &SELECT_THIRTY_TWO);
    let table = [FIRST as u32, (FIRST >> 32) as u32, 0, 0, SECOND as u32, (SECOND >> 32) as u32, 0, 0];
    let descriptor = [TABLE as u32, ((TABLE >> 32) as u32 & 0xffff) | (16 << 16), 2, 0x3101_6fac];
    memory.write_words(TABLE, &table);
    memory.write_words(DESCRIPTORS, &descriptor);
    let read = |address: u64, buf: &mut [u8]| {
        for (k, b) in buf.iter_mut().enumerate() {
            *b = memory.read_u8(address + k as u64);
        }
        true
    };
    let program = decode(&TABLE_STORING_CALLER);
    for wave in [32, 64] {
        for (index, scc, want) in [(0, 0, 16), (0, 1, 17), (1, 0, 32), (1, 1, 33), (2, 0, UNTOUCHED)] {
            let mut ud = user_data([0; 4], raw_desc(OUT_BASE, 64 * 4), 8);
            ud[..4].copy_from_slice(&[DESCRIPTORS as u32, (DESCRIPTORS >> 32) as u32, index, scc]);
            let site = CallSite { user_data_base: 0, user_data_count: 8, wave_size: wave };
            let linked = link_program(&program, &TABLE_STORING_CALLER, site, &ud, &read).unwrap().unwrap();
            assert_eq!(linked.callees.iter().map(|c| c.address).collect::<Vec<_>>(), vec![FIRST, SECOND]);
            let out = Dispatch::new(&linked.linked.code, wave, 64, ud)
                .buffer(DESCRIPTORS, descriptor.to_vec())
                .buffer(TABLE, table.to_vec())
                .output(OUT_BASE, 64)
                .run()
                .unwrap()
                .remove(0);
            assert!(out.iter().all(|&v| v == want), "wave{wave} record {index} scc {scc}: {out:?}, expected {want:#x}");
        }
    }
}

#[test]
fn unreadable_callees_are_reported() {
    let program = decode(&STORING_CALLER);
    let ud = [0x1000, 0, 0x2000, 0, 0, 0, 0, 0];
    let read = |address: u64, buf: &mut [u8]| {
        if address >= 0x2000 {
            return false;
        }
        buf.fill(0);
        buf[..8].copy_from_slice(&[0x85, 0x02, 0x02, 0x4a, 0x0e, 0x20, 0x80, 0xbe]);
        true
    };
    let err = link_program(&program, &STORING_CALLER, SITE, &ud, &read).unwrap_err();
    assert!(err.contains("is not readable") && err.contains("s[2:3]"), "{err}");
}

#[test]
fn programs_with_code_jump_tables_are_not_linked() {
    let jump_table: [u32; 26] = [
        0xbe8c1f00, 0x800cff0c, 0x00000058, 0x820d800d, 0x83928202, 0xd4c40016, 0x02000500, 0x8f128212, 0xf4000386, 0x24000000, 0xbf8cc07f, 0x7d820003, 0x808c0e0c,
        0x828d800d, 0xbefd210c, 0x7e020281, 0xbf820003, 0x7e020282, 0xbf820001, 0x7e020283, 0xe0700000, 0x80000100, 0xbf810000, 0x00000020, 0x00000018, 0x00000010,
    ];
    let bytes: Vec<u8> = jump_table.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ps5shader::decode_program_through(&ps5shader::CodeSlice::new(ADDRESS, &bytes), ADDRESS, 0x58).unwrap();
    assert!(link_calls(&p, &jump_table, &[], &[], &[], &[]).unwrap_err().contains("jumps through a table in its code"));
}

/// The tail of `code` (its S_SETPC_B64 is followed by `code`'s next word).
fn tail_of(code: &[u32], base: u32, count: u32) -> Option<UserDataTail> {
    let p = decode(code);
    let after = code.get(p.instructions.last().unwrap().next_pc() as usize / 4).copied().unwrap_or(0);
    find_user_data_tail(&p, base, count, after)
}

#[test]
fn user_data_tails() {
    let code = [0x7e000280, 0xbe802008, S_CODE_END];
    let tail = tail_of(&code, 8, 2).expect("unchanged user-data tail");
    assert_eq!((tail.pc, tail.user_data_register), (4, 8));
    assert!(tail_of(&code, 9, 2).is_none() && tail_of(&code, 8, 1).is_none());
    let refused: [&[u32]; 8] = [
        &[0xbe880380, 0xbe802008, S_CODE_END],
        &[0xbe890300, 0xbe802008, S_CODE_END],
        &[0xbe880480, 0xbe802008, S_CODE_END],
        &[0xf4080200, 0xfa000000, 0xbe802008, S_CODE_END],
        &[0xbf820000, 0xbe802008, S_CODE_END],
        &[0xbe802009, S_CODE_END],
        &[0xbe802008, 0xbf810000],
        &[0xbe802008, 0xbf800000, S_CODE_END],
    ];
    for r in refused {
        assert!(tail_of(r, 8, 4).is_none(), "accepted {r:x?}");
    }
    let continuation = [0xbf820001, 0x7e000281, 0xbf810000];
    let linked = link_user_data_tail(&decode(&code), &code, tail, &decode(&continuation), &continuation).unwrap();
    assert_eq!(linked, vec![0x7e000280, 0xbf820000, 0xbf820001, 0x7e000281, 0xbf810000]);
    cfg_of(&linked).unwrap();
    let reads_pc = [0xbe8c1f00, 0xbf810000];
    let err = link_user_data_tail(&decode(&code), &code, tail, &decode(&reads_pc), &reads_pc).unwrap_err();
    assert!(err.contains("reads its program counter"));
}

#[test]
fn constant_high_tails() {
    let accepted: [(&[u32], u32); 4] = [
        (&[0xbe850382, 0xbe802004, S_CODE_END], 2),
        (&[0xbe8503ff, 0x12345678, 0xbe802004, S_CODE_END], 0x12345678),
        (&[0xbe850380, 0xbe802004, S_CODE_END], 0),
        (&[0xbe850300, 0xbe850382, 0xbe802004, S_CODE_END], 2),
    ];
    let end = [0xbf810000];
    for (code, value) in accepted {
        let tail = tail_of(code, 0, 5).expect("constant high half");
        assert_eq!((tail.user_data_register, tail.constant_high), (4, Some(value)));
        let p = decode(code);
        let linked = link_user_data_tail(&p, code, tail, &decode(&end), &end).unwrap();
        assert_eq!(&linked[..code.len() - 2], &code[..code.len() - 2]);
        let mismatched = UserDataTail { constant_high: Some(value ^ 1), ..tail };
        assert!(link_user_data_tail(&p, code, mismatched, &decode(&end), &end).is_err());
    }
    let refused: [&[u32]; 5] = [
        &[0xbe850382, 0xbe850300, 0xbe802004, S_CODE_END],
        &[0xbe850382, 0x80058105, 0xbe802004, S_CODE_END],
        &[0xbe850382, 0xbe840380, 0xbe802004, S_CODE_END],
        &[0xbe850382, 0xf4000140, 0xfa000000, 0xbe802004, S_CODE_END],
        &[0xbf840001, 0xbe850382, 0xbe802004, S_CODE_END],
    ];
    for r in refused {
        assert!(tail_of(r, 0, 6).is_none(), "accepted {r:x?}");
    }
    assert!(tail_of(accepted[0].0, 0, 4).is_none());
}

/// Where a caller looping over table calls loads the table's descriptor
/// (s_load_dwordx4 s[96:99], s[0:1], 0x10) relative to the loop.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Shape {
    InLoop,
    Hoisted,
    SecondWriterOnArm,
    LoadOnArmOnly,
    WriterAfterCall,
    WriterAfterLoop,
    CallOnArm,
    LongBranchWriter,
}

/// The reference's hoisted table caller: an optional arm before a loop
/// that loads entry s[16:19] of the table and calls s[14:15].
fn hoisted_caller(shape: Shape) -> Vec<u32> {
    let mut code = Vec::new();
    let descriptor_load = |code: &mut Vec<u32>| code.extend_from_slice(&[0xf408_1800, 0xfa00_0010, 0xbf8c_c07f]);
    if shape != Shape::InLoop && shape != Shape::LoadOnArmOnly {
        descriptor_load(&mut code);
    }
    code.push(0xbf06_8002);
    code.push(if shape == Shape::LoadOnArmOnly { 0xbf85_0003 } else { 0xbf85_0001 });
    if shape == Shape::LoadOnArmOnly {
        descriptor_load(&mut code);
    } else {
        code.push(match shape {
            Shape::SecondWriterOnArm => 0xbee0_0380,
            Shape::CallOnArm => 0xbb1e_0000,
            Shape::LongBranchWriter => 0xbf82_0000,
            _ => 0x7e02_0280,
        });
    }
    let arm = code.len() - 1;
    code.push(0xbee5_037e);
    let lp = code.len();
    code.push(0xbee4_037e);
    if shape == Shape::InLoop {
        descriptor_load(&mut code);
    }
    code.extend_from_slice(&[0x7ed4_0500, 0x7da4_006a, 0x8f6a_846a, 0xf428_0430, 0xd400_0000, 0xbf8c_c07f, 0xbe8e_0310, 0xbe8f_0311, 0xbe90_0312, 0xbe91_0313]);
    code.push(0xbe8e_210e);
    if shape == Shape::WriterAfterCall {
        code.push(0xbee1_0301);
    }
    code.push(0x897e_7e64);
    code.push(0xbf89_0000 | ((lp as u32).wrapping_sub(code.len() as u32 + 1) & 0xffff));
    code.push(0xbefe_0365);
    if shape == Shape::WriterAfterLoop {
        code.push(0xbee0_0380);
    }
    code.push(0xbf81_0000);
    if shape == Shape::CallOnArm {
        code[arm] |= (code.len() - arm - 1) as u32;
        code.extend_from_slice(&[0xbe80_201e, 0xbf9f_0000]);
    }
    if shape == Shape::LongBranchWriter {
        code[arm] |= (code.len() - arm - 1) as u32;
        let getpc = code.len() + 3;
        code.extend_from_slice(&[0xf408_1800, 0xfa00_0020, 0xbf8c_c07f, 0xbe84_1f00, 0x8084_ff04, ((getpc - arm) * 4) as u32, 0xbe80_2004, 0xbf9f_0000]);
    }
    code
}

/// A table call whose descriptor load sits before its loop is found when
/// that load is the only writer reaching the call on every path (reference
/// 499e2db90, cc6901685); a second writer, a path without the load, a call
/// on the way, or a writer reached through a long branch leave it unlinked.
#[test]
fn finds_hoisted_table_calls() {
    use Shape::*;
    for shape in [InLoop, Shape::Hoisted, SecondWriterOnArm, LoadOnArmOnly, WriterAfterCall, WriterAfterLoop, CallOnArm, LongBranchWriter] {
        let linkable = matches!(shape, InLoop | Shape::Hoisted | WriterAfterLoop);
        for wave_size in [32, 64] {
            let program = decode(&hoisted_caller(shape));
            let calls = find_table_calls(&program, CallSite { user_data_base: 0, user_data_count: 16, wave_size });
            if !linkable {
                assert!(calls.is_empty(), "{shape:?} wave{wave_size}: {calls:?}");
                continue;
            }
            assert_eq!(calls.len(), 1, "{shape:?} wave{wave_size}");
            let c = calls[0];
            assert_eq!((c.user_data_register, c.descriptor_offset, c.entry_offset, c.entry_stride), (0, 0x10, 0, 16), "{shape:?} wave{wave_size}");
        }
    }
}
