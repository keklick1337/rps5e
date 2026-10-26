// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! ExecRunGrouping: runs of EXEC-masked vector writes under one branch on
//! the EXEC bit. The grouped function must compute what the plain one does
//! (the interpreter runs the divergent arm for every invocation and the
//! join's phis pick per invocation), and anything that is not
//! per-invocation arithmetic must stay outside the branch.

use ps5shader::interp::Machine;
use ps5shader::opt::exec_runs::{group_exec_runs, DEFAULT_MINIMUM};
use ps5shader::ssa::{Function, Op, Terminator};
use ps5shader::testing::asm::*;
use ps5shader::testing::{buffer_descriptor, ComputeRun, INPUT_BASE, OUTPUT_BASE};

const OUTPUTS: u32 = 8;

/// v1 = tid * 8 (output record), v2 = input[tid], v3..v10 seeded with
/// constants under the full EXEC, then EXEC = (16 < v2) saved in s[10:11].
fn prologue() -> Vec<u32> {
    let mut c = vec![vop2(0x1A, 1, inline_u32(3), 0)]; // v_lshlrev_b32 v1, 3, v0
    c.extend(mubuf(OP_BUFFER_LOAD_DWORD, 2, 0, 0, inline_u32(0), 0, true, false, false));
    for k in 0..OUTPUTS {
        c.push(vop1(OP_V_MOV_B32, 3 + k, inline_u32(k + 1)));
    }
    c.push(vopc(0xC1, inline_u32(16), 2)); // v_cmp_lt_u32 vcc, 16, v2
    c.push(sop1(0x24, 10, VCC_LO)); // s_and_saveexec_b64 s[10:11], vcc
    c
}

/// Restore EXEC and store v3..v10 to output[tid * 8 + k].
fn epilogue() -> Vec<u32> {
    let mut c = vec![sop2(0x11, EXEC_LO, EXEC_LO, 10)]; // s_or_b64 exec, exec, s[10:11]
    for k in 0..OUTPUTS {
        c.extend(mubuf(OP_BUFFER_STORE_DWORD, 3 + k, 1, 1, inline_u32(0), k * 4, true, false, false));
    }
    c.push(S_ENDPGM);
    c
}

/// A run of eight vector ALU writes, each register depending on the last.
fn arithmetic() -> Vec<u32> {
    vec![
        vop2(0x25, 3, vgpr(2), 3),   // v_add_nc_u32 v3, v2, v3
        vop2(0x1D, 4, vgpr(3), 4),   // v_xor_b32 v4, v3, v4
        vop2(0x1B, 5, vgpr(4), 2),   // v_and_b32 v5, v4, v2
        vop2(0x1C, 6, vgpr(5), 3),   // v_or_b32 v6, v5, v3
        vop2(0x1A, 7, inline_u32(2), 6), // v_lshlrev_b32 v7, 2, v6
        vop2(0x26, 8, vgpr(7), 2),   // v_sub_nc_u32 v8, v7, v2
        vop2(0x25, 9, vgpr(8), 8),   // v_add_nc_u32 v9, v8, v8
        vop2(0x1D, 10, vgpr(9), 3),  // v_xor_b32 v10, v9, v3
    ]
}

struct Case {
    run: ComputeRun,
    threads: u32,
}

impl Case {
    fn new(body: &[u32], wave: u32, threads: u32) -> Case {
        let mut code = prologue();
        code.extend_from_slice(body);
        code.extend(epilogue());
        let input = buffer_descriptor(INPUT_BASE, 4, threads, 0x1101_6FAC);
        let output = buffer_descriptor(OUTPUT_BASE, 4, threads * OUTPUTS, 0x1101_6FAC);
        let user: Vec<u32> = input.iter().chain(output.iter()).copied().collect();
        Case { run: ComputeRun::new(code, wave, threads, user), threads }
    }

    fn outputs(&self, f: &Function) -> Vec<u32> {
        let n = self.threads as usize;
        let input: Vec<u32> = (0..self.threads).map(|i| i.wrapping_mul(7) % 29).collect();
        let total = n * OUTPUTS as usize;
        let setup = move |m: &mut Machine| {
            m.memory.write_words(INPUT_BASE, &input);
            m.memory.write_words(OUTPUT_BASE, &vec![0xDEAD_BEEF; total]);
        };
        let m = self.run.run_function(f, &setup).unwrap();
        m.memory.read_words(OUTPUT_BASE, total)
    }

    /// (plain outputs, grouped outputs, runs grouped, the grouped function).
    fn compare(&self) -> (Vec<u32>, Vec<u32>, usize, Function) {
        let plain = self.run.compile(true).unwrap();
        let mut grouped = plain.clone();
        let n = group_exec_runs(&mut grouped, DEFAULT_MINIMUM);
        ps5shader::ssa::verify::verify(&grouped, true).unwrap_or_else(|e| panic!("{e}\n{grouped}"));
        (self.outputs(&plain), self.outputs(&grouped), n, grouped)
    }
}

fn selects(f: &Function) -> usize {
    f.order.iter().flat_map(|&b| f.block(b).insts.iter()).filter(|&&v| f.op(v) == Op::Select).count()
}

/// Divergent branches: their arms must hold per-invocation arithmetic only.
fn divergent_arms_are_arithmetic(f: &Function) {
    for &b in &f.order {
        if let Terminator::CondBranch { then_block, .. } = f.block(b).term {
            for &v in &f.block(then_block).insts {
                let op = f.op(v);
                assert!(!op.has_side_effects() && !op.reads_memory() && !op.is_subgroup(), "{} under a branch\n{f}", op.name());
            }
        }
    }
}

#[test]
fn a_run_of_vector_writes_shares_one_branch() {
    for wave in [32, 64] {
        let case = Case::new(&arithmetic(), wave, 32);
        let (plain, grouped, n, f) = case.compare();
        assert_eq!(n, 1, "wave{wave}\n{f}");
        assert_eq!(plain, grouped, "wave{wave}");
        assert!(selects(&f) < OUTPUTS as usize, "wave{wave}: the run still selects per write\n{f}");
        divergent_arms_are_arithmetic(&f);
        // Lanes with EXEC clear kept the seeded values.
        for lane in 0..32usize {
            let x = (lane as u32 * 7) % 29;
            if x <= 16 {
                assert_eq!(&grouped[lane * 8..lane * 8 + 8], &[1, 2, 3, 4, 5, 6, 7, 8], "wave{wave} lane {lane}");
            }
        }
    }
}

#[test]
fn short_runs_keep_their_selects() {
    let case = Case::new(&arithmetic()[..DEFAULT_MINIMUM - 1], 32, 32);
    let (plain, grouped, n, _) = case.compare();
    assert_eq!(n, 0);
    assert_eq!(plain, grouped);
}

#[test]
fn a_store_inside_the_run_stays_outside_the_branch() {
    let mut body = arithmetic();
    // buffer_store_dword v5 at output[tid * 8 + 7], later overwritten by the
    // epilogue's store of v10: the store must still run in order.
    let store = mubuf(OP_BUFFER_STORE_DWORD, 5, 1, 1, inline_u32(0), 7 * 4, true, false, false);
    body.splice(4..4, store);
    let case = Case::new(&body, 32, 32);
    let (plain, grouped, _, f) = case.compare();
    assert_eq!(plain, grouped);
    divergent_arms_are_arithmetic(&f);
}

#[test]
fn cross_lane_reads_end_the_run() {
    let mut body = arithmetic();
    // v_mov_b32_dpp v6, v4 row_shr:1 in the middle: a neighbour's register.
    body.splice(4..4, [vop1(OP_V_MOV_B32, 6, 0xFA), dpp(4, 0x111, 0xF, 0xF, false, false)]);
    // v_readlane_b32 s20, v8, 3 then v_mov_b32 v9, s20.
    body.extend(vop3(0x360, 20, vgpr(8), inline_u32(3), 0));
    body.push(vop1(OP_V_MOV_B32, 9, 20));
    let case = Case::new(&body, 32, 32);
    let (plain, grouped, _, f) = case.compare();
    assert_eq!(plain, grouped);
    divergent_arms_are_arithmetic(&f);
}

#[test]
fn every_exec_pattern_matches_the_plain_function() {
    // Partial waves change which lanes are present and active.
    for threads in [1, 5, 17, 32] {
        let case = Case::new(&arithmetic(), 32, threads);
        let (plain, grouped, _, _) = case.compare();
        assert_eq!(plain, grouped, "{threads} threads");
    }
}

/// A stretch as long as a large game program's: a chain of 20000
/// dependent additions under selects on one condition, its last value
/// stored. Every value of the chain leaves the branch's candidates one
/// after the other (each is used by the next, the last by the store): the
/// classification follows the drops through a worklist instead of one pass
/// over the window per dropped value (minutes for this function).
#[test]
fn a_long_stretch_classifies_in_linear_time() {
    use ps5shader::ssa::{Access, Function, ICmp, Stage, Type};
    let mut f = Function::new(32, Stage::Compute, 0);
    let b = f.new_block();
    f.entry = b;
    f.order = vec![b];
    let x = f.append(b, Op::UserData(0), Type::U32, vec![]);
    let old = f.append(b, Op::UserData(1), Type::U32, vec![]);
    let lane = f.append(b, Op::LaneId, Type::U32, vec![]);
    let exec = f.append(b, Op::ICmp(ICmp::ULt), Type::BOOL, vec![lane, x]);
    let mut c = x;
    let mut selected = Vec::new();
    for i in 0..20000 {
        c = f.append(b, Op::IAdd, Type::U32, vec![c, x]);
        if i % 4 == 0 {
            selected.push(f.append(b, Op::Select, Type::U32, vec![exec, c, old]));
        }
    }
    f.append(b, Op::LdsStore { access: Access::Dwords(1) }, Type::VOID, vec![x, c, exec]);
    for s in selected {
        f.append(b, Op::LdsStore { access: Access::Dwords(1) }, Type::VOID, vec![x, s, exec]);
    }
    f.block_mut(b).term = Terminator::Return;
    let before = f.to_string();
    let t = std::time::Instant::now();
    // The chain must stay before the branch, so no run forms.
    assert_eq!(group_exec_runs(&mut f, DEFAULT_MINIMUM), 0);
    let elapsed = t.elapsed();
    assert!(elapsed.as_secs() < 10, "{elapsed:?}");
    assert_eq!(f.to_string(), before);
}
