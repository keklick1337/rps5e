// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Optimization pass behaviour on small programs.

use ps5shader::ssa::*;
use ps5shader::testing::asm::*;
use ps5shader::testing::IoTest;

fn count(f: &Function, pred: impl Fn(Op) -> bool) -> usize {
    f.order.iter().flat_map(|&b| f.block(b).insts.iter()).filter(|&&v| pred(f.op(v))).count()
}

fn prologue() -> Vec<u32> {
    let mut c = vec![vop2(0x1A, 1, inline_u32(2), 0)];
    c.extend(mubuf(OP_BUFFER_LOAD_DWORD, 2, 0, 0, inline_u32(0), 0, true, false, false));
    c
}

fn store(reg: u32) -> Vec<u32> {
    mubuf(OP_BUFFER_STORE_DWORD, reg, 0, 1, inline_u32(0), 0, true, false, false)
}

#[test]
fn spilled_sgpr_read_back_is_forwarded() {
    let mut code = prologue();
    code.extend(vop3(0x360, 8, 256 + 2, inline_u32(3), 0)); // s8 = readlane(v2, 3)
    code.extend(vop3(0x361, 9, 8, inline_u32(2), 0)); // writelane v9[2] = s8
    code.extend(vop3(0x360, 9, 256 + 9, inline_u32(2), 0)); // s9 = readlane(v9, 2)
    code.push(vop1(OP_V_MOV_B32, 3, 9));
    code.extend(store(3));
    code.push(S_ENDPGM);
    let t = IoTest::new(&code, 32, 32);
    let f = t.run.compile(true).unwrap();
    // Only the first readlane (of a per-lane value) remains.
    assert_eq!(count(&f, |o| o == Op::ReadLane), 1, "{f}");
    let out = t.run(&(0..32).map(|i| i + 50).collect::<Vec<_>>()).unwrap();
    assert!(out.iter().all(|&x| x == 53));
}

#[test]
fn readlane_of_uniform_value_disappears() {
    let mut code = prologue();
    code.push(vop1(OP_V_MOV_B32, 4, 5)); // v4 = s5 (uniform)
    code.extend(vop3(0x360, 10, 256 + 4, inline_u32(7), 0)); // s10 = readlane(v4, 7)
    code.push(vop1(0x02, 11 << 0, 256 + 4)); // v_readfirstlane_b32 s11, v4
    code.push(sop2(0x00, 12, 10, 11));
    code.push(vop1(OP_V_MOV_B32, 3, 12));
    code.extend(store(3));
    code.push(S_ENDPGM);
    let t = IoTest::new(&code, 32, 32);
    let f = t.run.compile(true).unwrap();
    assert_eq!(count(&f, |o| o == Op::ReadLane), 0, "{f}");
}

#[test]
fn entry_exec_selects_fold_away() {
    let mut code = prologue();
    code.push(vop2(0x25, 3, F_ONE, 2)); // v_add_nc_u32 (no denormal flushing selects)
    code.extend(store(3));
    code.push(S_ENDPGM);
    let f = IoTest::new(&code, 32, 32).run.compile(true).unwrap();
    assert_eq!(count(&f, |o| o == Op::Select), 0, "{f}");
}

#[test]
fn lds_write_then_read_gets_a_subgroup_barrier() {
    let mut code = prologue();
    code.extend(ds(0x0D, 0, 1, 2, 0, 0, 0)); // ds_write_b32 v1, v2
    code.extend(ds(0x36, 3, 1, 0, 0, 4, 0)); // ds_read_b32 v3, v1 offset:4
    code.extend(store(3));
    code.push(S_ENDPGM);
    let t = IoTest::new(&code, 32, 32);
    let f = t.run.compile(true).unwrap();
    assert_eq!(count(&f, |o| o == Op::SubgroupLdsBarrier), 1, "{f}");
    // Lane i reads what lane i+1 wrote.
    let input: Vec<u32> = (0..32).map(|i| 100 + i).collect();
    let out = t.run(&input).unwrap();
    for i in 0..31 {
        assert_eq!(out[i], 101 + i as u32);
    }
}

#[test]
fn split_descriptor_phi_duplicates_image_reads() {
    // Hand-built: b0 branches on a uniform condition to two arms choosing a
    // texture descriptor from user data 0..3 or 4..7; the merge samples
    // through a phi.
    let mut f = Function::new(32, Stage::Pixel, 0x1000);
    let b0 = f.new_block();
    let b1 = f.new_block();
    let b2 = f.new_block();
    let b3 = f.new_block();
    f.order = vec![b0, b1, b2, b3];
    f.entry = b0;
    let mut b = Builder::new(&mut f, b0);
    let c = b.emit(Op::UserData(8), Type::U32, vec![]);
    let z = b.u32(0);
    let cond = b.icmp(ICmp::Eq, c, z);
    f.block_mut(b0).term = Terminator::CondBranch { cond, then_block: b1, else_block: b2 };
    f.block_mut(b0).merge = Merge::Selection { merge: b3 };
    let mut descs = Vec::new();
    for (blk, base) in [(b1, 0u32), (b2, 4)] {
        let mut b = Builder::new(&mut f, blk);
        let parts: Vec<Value> = (0..4).map(|k| b.emit(Op::UserData(base + k), Type::U32, vec![])).collect();
        descs.push(b.construct(&parts));
        f.block_mut(blk).term = Terminator::Branch(b3);
    }
    f.rebuild_preds();
    let phi = f.new_value(Op::Phi, Type::U32X4, descs.clone());
    f.value_mut(phi).block = Some(b3);
    f.block_mut(b3).insts.push(phi);
    let mut b = Builder::new(&mut f, b3);
    let undef = b.func.undef(Type::U32X4);
    let x = b.f32(0.5);
    let x = b.bitcast(x, Type::U32);
    let t = b.bool(true);
    let attr = ImageAttr {
        pc: 0x40,
        op: ImageOp::Sample,
        flags: SampleFlags { lod_zero: true, ..Default::default() },
        dim: 1,
        dmask: 0xF,
        a16: false,
        d16: false,
        glc: false,
        slc: false,
        address_dwords: 2,
        data_dwords: 0,
    };
    let s = b.emit(Op::Image(attr), Type::U32X4, vec![phi, undef, undef, x, x, t]);
    b.emit(Op::Export(ExportAttr { target: 0, enable: 0xF, compressed: false, done: true, valid_mask: true }), Type::VOID, vec![s, s, s, s, t]);
    f.block_mut(b3).term = Terminator::Return;
    ps5shader::ssa::verify::verify(&f, true).unwrap();
    let n = ps5shader::opt::split_phis::split_descriptor_phis(&mut f);
    assert_eq!(n, 1);
    ps5shader::ssa::verify::verify(&f, true).unwrap();
    assert_eq!(count(&f, |o| matches!(o, Op::Image(_))), 2, "{f}");
    // Each copy reads a descriptor that is not a phi.
    for &v in &f.block(b3).insts {
        if let Op::Image(_) = f.op(v) {
            assert_ne!(f.op(f.args(v)[0]), Op::Phi, "{f}");
        }
    }
}

/// Wave32: the lane's bit of a mask combined from compare masks by scalar
/// mask arithmetic is computed from the compares themselves.
#[test]
fn lane_masks_project_to_predicates() {
    let mut code = prologue();
    code.push(vopc(0xC4, 256 + 2, 1)); // vcc_lo = v2 > v1 (v_cmp_gt_u32)
    code.extend(vop3(0xC1, 20, 256 + 2, inline_u32(20), 0)); // s20 = v2 < 20
    code.push(sop2(0x0E, 22, VCC_LO, 20)); // s22 = vcc_lo & s20
    code.push(sop1(0x07, 23, 22)); // s23 = ~s22 (s_not_b32)
    code.push(sop2(0x10, 23, 23, 20)); // s23 |= s20 (s_or_b32)
    code.extend(vop3(0x101, 3, inline_u32(0), inline_u32(1), 22)); // v3 = s22 ? 1 : 0
    code.extend(vop3(0x101, 5, inline_u32(0), inline_u32(2), 23)); // v5 = s23 ? 2 : 0
    code.push(vop2(0x25, 3, 256 + 3, 5)); // v3 += v5
    code.extend(store(3));
    code.push(S_ENDPGM);
    let t = IoTest::new(&code, 32, 32);
    let f = t.run.compile(true).unwrap();
    assert_eq!(count(&f, |o| o == Op::InverseBallot), 0, "{f}");
    let input: Vec<u32> = (0..32).map(|i| i * 5).collect();
    let out = t.run(&input).unwrap();
    for (lane, (&x, &o)) in input.iter().zip(&out).enumerate() {
        let (a, b) = (x > lane as u32 * 4, x < 20);
        let s6 = a && b;
        let s7 = !s6 || b;
        assert_eq!(o, s6 as u32 + 2 * s7 as u32, "lane {lane}");
    }
}
