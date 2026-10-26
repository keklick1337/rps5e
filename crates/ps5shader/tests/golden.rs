// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Golden tests on hand-assembled encodings and synthetic programs.

use ps5shader::ir::{exec_full, movrel, wave64_half_mask, ControlFlowGraph, Gen5BranchResolver};
use ps5shader::*;
use std::collections::HashSet;

fn one(words: &[u32]) -> Instruction {
    decode_instruction(0, words).unwrap_or_else(|e| panic!("decode {words:08x?}: {e}"))
}

fn blob(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn program(words: &[u32]) -> Program {
    decode_blob(&blob(words), 0x1000).unwrap_or_else(|e| panic!("decode program: {e}"))
}

const S_ENDPGM: u32 = 0xBF81_0000;

#[test]
fn scalar_alu() {
    let i = one(&[0xBE80_0301]);
    assert_eq!((i.encoding, i.opcode), (Encoding::Sop1, Opcode::SMovB32));
    assert_eq!(i.destinations, vec![Operand::scalar(0)]);
    assert_eq!(i.sources, vec![Operand::scalar(1)]);
    assert_eq!(i.to_string(), "s_mov_b32 s0 <- s1");

    // s_cmp_eq_u32 s1, 0x100 (literal)
    let i = one(&[0xBF06_FF01, 0x100]);
    assert_eq!((i.encoding, i.opcode, i.words.len()), (Encoding::Sopc, Opcode::SCmpEqU32, 2));
    assert_eq!(i.sources, vec![Operand::scalar(1), Operand::literal(0x100)]);

    // s_movk_i32 s5, 0x1234
    let i = one(&[0xB005_1234]);
    assert_eq!((i.encoding, i.opcode), (Encoding::Sopk, Opcode::SMovkI32));
    assert_eq!(i.sources, vec![Operand::encoded(0x1234)]);
    assert_eq!(i.destinations, vec![Operand::scalar(5)]);

    // s_cmov_b64 becomes s_cselect_b64 with the old destination as source 1.
    let i = one(&[0xBE84_0602]);
    assert_eq!((i.encoding, i.opcode), (Encoding::Sop2, Opcode::SCselectB64));
    assert_eq!(i.sources, vec![Operand::scalar(2), Operand::scalar(4)]);

    // s_lshl_b32 vcc_lo, s3, 16 (from a game shader)
    let i = one(&[0x8F6A_9003]);
    assert_eq!(i.opcode, Opcode::SLshlB32);
    assert_eq!(i.sources, vec![Operand::scalar(3), Operand::encoded(144)]);
    assert_eq!(i.to_string(), "s_lshl_b32 vcc_lo <- s3, 16");
}

#[test]
fn sopp_and_branches() {
    let e = one(&[S_ENDPGM]);
    assert_eq!((e.encoding, e.opcode), (Encoding::Sopp, Opcode::SEndpgm));
    let b = one(&[0xBF82_0002]);
    assert_eq!(b.opcode, Opcode::SBranch);
    assert_eq!(ir::branch::branch_target(&b), Some(12));
    // Backward: simm16 = -2 at pc 0 -> 4 - 8 < 0: no target.
    assert_eq!(ir::branch::branch_target(&one(&[0xBF82_FFFE])), None);
    assert!(matches!(
        decode_instruction(0, &[0xBFFF_0000]),
        Err(InstructionError::UnknownOpcode { encoding: Encoding::Sopp, opcode: 0x7F, .. })
    ));
}

#[test]
fn scalar_memory() {
    // s_load_dwordx2 s[24:25], s[14:15], null
    let i = one(&[0xF404_0607, 0xFA00_0000]);
    assert_eq!((i.encoding, i.opcode), (Encoding::Smem, Opcode::SLoadDwordx2));
    assert_eq!(i.sources, vec![Operand::scalar(14), Operand::encoded(125)]);
    assert_eq!(i.destinations, vec![Operand::scalar(24), Operand::scalar(25)]);
    match i.control {
        Some(Control::ScalarMemory(c)) => {
            assert_eq!((c.destination_count, c.immediate_offset_bytes, c.dynamic_offset_register), (2, 0, None))
        }
        c => panic!("{c:?}"),
    }
    // s_buffer_load_dwordx4 s[28:31], s[8:11], 0x10
    let i = one(&[0xF428_0704, 0xFA00_0010]);
    assert_eq!(i.opcode, Opcode::SBufferLoadDwordx4);
    assert!(matches!(i.control, Some(Control::ScalarMemory(ScalarMemoryControl { immediate_offset_bytes: 16, .. }))));
}

#[test]
fn vector_alu() {
    // v_mov_b32 v0, 1.0
    let i = one(&[0x7E00_02F2]);
    assert_eq!((i.encoding, i.opcode), (Encoding::Vop1, Opcode::VMovB32));
    assert_eq!(i.sources, vec![Operand::encoded(242)]);
    assert_eq!(i.to_string(), "v_mov_b32 v0 <- 1.0");

    // v_add_f32 v1, v2, v3
    let i = one(&[0x0602_0702]);
    assert_eq!((i.encoding, i.opcode), (Encoding::Vop2, Opcode::VAddF32));
    assert_eq!(i.sources, vec![Operand::vector(2), Operand::vector(3)]);
    assert_eq!(i.destinations, vec![Operand::vector(1)]);

    // v_fmamk_f32 v1, v2, K, v3: literal sits between the sources.
    let i = one(&[(0x2C << 25) | (1 << 17) | (3 << 9) | 0x102, 0x3F80_0000]);
    assert_eq!(i.opcode, Opcode::VFmaMkF32);
    assert_eq!(i.sources, vec![Operand::vector(2), Operand::literal(0x3F80_0000), Operand::vector(3)]);
    // v_fmaak_f32: literal last.
    let i = one(&[(0x2D << 25) | (1 << 17) | (3 << 9) | 0x102, 0x4000_0000]);
    assert_eq!(i.sources, vec![Operand::vector(2), Operand::vector(3), Operand::literal(0x4000_0000)]);

    // v_readfirstlane_b32 s5, v1: scalar destination.
    let i = one(&[0x7E0A_0501]);
    assert_eq!(i.opcode, Opcode::VReadfirstlaneB32);
    assert_eq!(i.destinations, vec![Operand::scalar(5)]);

    // v_movrels_b32 reads M0.
    let i = one(&[0x7E00_8701]);
    assert_eq!(i.opcode, Opcode::VMovrelsB32);
    assert_eq!(i.sources, vec![Operand::vector(1), Operand::scalar(M0)]);

    // v_mad_f32 v7, v7, s29, s31 (VOP3, from a game shader)
    let i = one(&[0xD541_0007, 0x007C_3B07]);
    assert_eq!((i.encoding, i.opcode), (Encoding::Vop3, Opcode::VMadF32));
    assert_eq!(i.sources, vec![Operand::vector(7), Operand::scalar(29), Operand::scalar(31)]);
    assert_eq!(i.destinations, vec![Operand::vector(7)]);

    // VOP3 with a literal: three dwords.
    let i = one(&[0xD541_0007, 0x007C_3AFF, 0x4000_0000]);
    assert_eq!(i.words.len(), 3);
    assert_eq!(i.sources[0], Operand::literal(0x4000_0000));

    // Unknown VOP3 opcode is accepted raw.
    let i = one(&[0xD5FF_0000, 0x0000_0000]);
    assert_eq!(i.opcode, Opcode::Vop3Raw(0x1FF));
    assert!(i.opcode.is_raw());
}

#[test]
fn vop3b_and_compares() {
    // v_add_co_u32 v0, vcc_lo, v1, v2
    let i = one(&[0xD70F_6A00, 0x0002_0501]);
    assert_eq!(i.opcode, Opcode::VAddCoU32);
    assert_eq!(i.control.as_ref().and_then(Control::scalar_destination), Some(VCC_LO));
    assert_eq!(&i.sources[..2], &[Operand::vector(1), Operand::vector(2)]);

    // VOP3 compare: v_cmp_gt_f32 s[4:5], v1, v2 (opcode 0x004)
    let i = one(&[0xD404_0004, 0x0002_0501]);
    assert_eq!(i.opcode, Opcode::VCmpGtF32);
    assert_eq!(i.sources, vec![Operand::vector(1), Operand::vector(2)]);
    assert_eq!(i.destinations, vec![Operand::scalar(4)]);
    // VOP3 V_CMPX writes EXEC only.
    let i = one(&[0xD414_0000, 0x0002_0501]);
    assert_eq!(i.opcode, Opcode::VCmpxGtF32);
    assert!(i.destinations.is_empty());

    // VOPC SDWA: v_cmp_gt_u32 s0, 64, v18 (from a game shader)
    let i = one(&[0x7D88_24F9, 0x0686_80C0]);
    assert_eq!((i.encoding, i.opcode), (Encoding::Vopc, Opcode::VCmpGtU32));
    assert_eq!(i.sources, vec![Operand::encoded(192), Operand::vector(18)]);
    assert_eq!(i.destinations, vec![Operand::scalar(0)]);
    assert!(matches!(i.control, Some(Control::Sdwa(SdwaControl { scalar_destination: Some(0), .. }))));

    // Plain VOPC writes VCC implicitly.
    let i = one(&[0x7C08_0501]);
    assert_eq!(i.opcode, Opcode::VCmpGtF32);
    assert!(i.destinations.is_empty());
}

#[test]
fn dpp_forms() {
    // v_mov_b32 v0, v1 dpp quad_perm / row_shr:1
    let i = one(&[0x7E00_02FA, 0xFF01_1101]);
    assert!(matches!(i.control, Some(Control::Dpp(DppControl { control: 0x111, row_mask: 0xF, bank_mask: 0xF, .. }))));
    assert_eq!(i.sources, vec![Operand::vector(1)]);
    // DPP8 with fetch-inactive.
    let i = one(&[0x7E00_02EA, 0x0000_0001]);
    assert!(matches!(i.control, Some(Control::Dpp8(Dpp8Control { fetch_inactive: true, .. }))));
}

#[test]
fn vop3p() {
    let i = one(&[0xCC0E_4005, 0x1C0E_0501]);
    assert_eq!((i.encoding, i.opcode), (Encoding::Vop3p, Opcode::VPkFmaF16));
    assert_eq!(i.sources, vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)]);
    assert!(matches!(i.control, Some(Control::Vop3p(Vop3pControl { op_sel_hi_mask: 7, op_sel_mask: 0, .. }))));
}

#[test]
fn memory_and_export() {
    // buffer_load_format_xy v[6:7], v6, s[16:19], s25 idxen (game shader)
    let i = one(&[0xE004_2000, 0x1904_0606]);
    assert_eq!((i.encoding, i.opcode), (Encoding::Mubuf, Opcode::BufferLoadFormatXy));
    assert_eq!(i.destinations, vec![Operand::vector(6), Operand::vector(7)]);
    assert_eq!(i.sources, vec![Operand::vector(6), Operand::scalar(16), Operand::scalar(25)]);
    assert!(matches!(i.control, Some(Control::BufferMemory(BufferMemoryControl { index_enabled: true, typed: false, .. }))));

    // tbuffer_load_format_x: typed with the format field.
    let i = one(&[0xE808_0000 | (0x16 << 19), 0x0000_0000]);
    assert_eq!(i.encoding, Encoding::Mtbuf);
    assert!(matches!(i.control, Some(Control::BufferMemory(BufferMemoryControl { typed: true, .. }))));

    // image_sample v0, v4, s[8:15], s[16:19] dmask 0xf 2D
    let i = one(&[0xF080_0F08, 0x0082_0004]);
    assert_eq!((i.encoding, i.opcode), (Encoding::Mimg, Opcode::ImageSample));
    match &i.control {
        Some(Control::Image(c)) => {
            assert_eq!((c.dmask, c.dimension, c.scalar_resource, c.scalar_sampler), (0xF, 1, 8, 16));
            assert_eq!(c.address_registers, vec![4]);
        }
        c => panic!("{c:?}"),
    }
    // NSA: one extra address dword -> 3 dwords, 5 address registers.
    let i = one(&[0xF080_0F0A, 0x0082_0004, 0x0807_0605]);
    assert_eq!(i.words.len(), 3);
    assert!(matches!(&i.control, Some(Control::Image(c)) if c.address_registers == vec![4, 5, 6, 7, 8]));

    // exp mrt0 v0, v1, v2, v3 done vm
    let i = one(&[0xF800_180F, 0x0302_0100]);
    assert_eq!(i.opcode, Opcode::Exp);
    assert!(matches!(i.control, Some(Control::Export(ExportControl { target: 0, enable_mask: 0xF, done: true, valid_mask: true, .. }))));

    // ds_read2_b32 v[2:3], v1 offset0:1 offset1:2
    let i = one(&[0xD8DC_0201, 0x0200_0001]);
    assert_eq!(i.opcode, Opcode::DsRead2B32);
    assert_eq!(i.destinations, vec![Operand::vector(2), Operand::vector(3)]);
    assert!(matches!(i.control, Some(Control::DataShare(DataShareControl { offset0: 1, offset1: 2, gds: false }))));

    // v_interp_p1_f32 v2, v0, attr3.y
    let i = one(&[0xC808_0D00]);
    assert_eq!(i.opcode, Opcode::VInterpP1F32);
    assert!(matches!(i.control, Some(Control::Interpolation(InterpolationControl { attribute: 3, channel: 1 }))));

    // global_load_dword v1, v[2:3], off
    let i = one(&[0xDC30_8000, 0x017D_0002]);
    assert_eq!(i.opcode, Opcode::Flat(FlatSegment::Global, FlatOp::LoadDword));
    assert_eq!(i.sources, vec![Operand::vector(2), Operand::scalar(125)]);
    assert_eq!(i.destinations, vec![Operand::vector(1)]);
    assert_eq!(i.opcode.mnemonic(), "global_load_dword");
}

#[test]
fn flat_address_base_resolution() {
    // v_mov_b32 v2, s4; v_mov_b32 v3, s5; flat_load_dword v1, v[2:3]; s_endpgm
    let p = program(&[0x7E04_0204, 0x7E06_0205, 0xDC30_0000, 0x0100_0002, S_ENDPGM]);
    let load = &p.instructions[2];
    assert_eq!(load.opcode, Opcode::Flat(FlatSegment::Flat, FlatOp::LoadDword));
    assert!(matches!(&load.control, Some(Control::GlobalMemory(c)) if c.uses_flat_address && c.scalar_address == 4));
    assert_eq!(load.sources.last(), Some(&Operand::scalar(4)));
}

#[test]
fn program_termination() {
    // Out-of-line block after S_ENDPGM reached by a forward branch.
    let p = program(&[0xBF85_0001, S_ENDPGM, 0x7E00_0280, 0xBF82_FFFD, 0xDEAD_BEEF]);
    assert_eq!(p.instructions.len(), 4);
    assert_eq!(p.instructions.last().unwrap().opcode, Opcode::SBranch);
    // S_CODE_END padding after the end.
    let p = program(&[0x7E00_0280, S_ENDPGM, 0xBF9F_0000]);
    assert_eq!(p.instructions.len(), 2);
    // S_CODE_END before the program ends is an error.
    assert!(matches!(
        decode_blob(&blob(&[0xBF9F_0000]), 0x1000),
        Err(DecodeError::CodeEndInsideProgram { .. })
    ));
    // Debugger branches do not extend the program.
    let p = program(&[0xBF97_0001, S_ENDPGM, 0xFFFF_FFFF]);
    assert_eq!(p.instructions.len(), 2);
    // Unregistered S_SETPC stops decoding; the garbage after it is ignored.
    let p = program(&[0xBE80_0301, 0xBE80_2002, 0xFFFF_FFFF]);
    assert_eq!(p.instructions.last().unwrap().opcode, Opcode::SSetpcB64);
    let (_, t) = decode_segment(&CodeSlice::new(0x1000, &blob(&[0xBE80_2002])), 0x1000, Some(4), true).unwrap();
    assert_eq!(t, Termination::SetProgramCounter);
    // Running off the end of the blob.
    assert!(matches!(decode_blob(&blob(&[0x7E00_0280]), 0x1000), Err(DecodeError::ReadFailed { pc: 4 })));
}

#[test]
fn fused_programs() {
    let entry = blob(&[0xBE80_0301, 0xBE80_2002]);
    let cont = blob(&[0x7E00_0280, S_ENDPGM]);
    // Continuation after the entry: spliced at its real distance.
    let code = [CodeSlice::new(0x10000, &entry), CodeSlice::new(0x10100, &cont)];
    let layout = FusedLayout { entry_address: 0x10000, entry_size: 8, continuation_address: 0x10100, continuation_size: 8 };
    let p = decode_fused_program(&code[..], layout).unwrap();
    let pcs: Vec<u32> = p.instructions.iter().map(|i| i.pc).collect();
    assert_eq!(pcs, vec![0, 4, 0x100, 0x104]);
    assert_eq!(p.instructions[1].opcode, Opcode::SNop);
    assert_eq!(p.instructions[1].words, vec![0xBF80_0000]);
    assert!(p.instructions.iter().all(|i| i.address_offset.is_none()));

    // Continuation below the entry: placed at the aligned entry end, with
    // the real distance kept in address_offset.
    let code = [CodeSlice::new(0x10000, &entry), CodeSlice::new(0x8000, &cont)];
    let layout = FusedLayout { entry_address: 0x10000, entry_size: 8, continuation_address: 0x8000, continuation_size: 8 };
    let p = decode_fused_program(&code[..], layout).unwrap();
    assert_eq!(p.instructions[2].pc, 0x100);
    assert_eq!(p.instructions[2].program_offset(), 0x8000u64.wrapping_sub(0x10000));
    assert_eq!(p.instructions[3].program_offset(), 0x8000u64.wrapping_sub(0x10000) + 4);

    // Entry without S_SETPC.
    let code = [CodeSlice::new(0x10000, &cont), CodeSlice::new(0x10100, &cont)];
    let layout = FusedLayout { entry_address: 0x10000, entry_size: 8, continuation_address: 0x10100, continuation_size: 8 };
    assert!(matches!(decode_fused_program(&code[..], layout), Err(DecodeError::FusedEntryMissingSetpc { .. })));
    let bad = FusedLayout { entry_size: 6, ..layout };
    assert!(matches!(decode_fused_program(&code[..], bad), Err(DecodeError::InvalidShaderSize { size: 6 })));
}

#[test]
fn control_flow_graph() {
    // 0: s_cbranch_scc0 -> c ; 4: v_mov ; 8: s_branch -> 4 (loop) ; c: s_endpgm
    let p = program(&[0xBF84_0002, 0x7E00_0280, 0xBF82_FFFE, S_ENDPGM]);
    assert_eq!(p.instructions.len(), 4);
    let cfg = ControlFlowGraph::build(&p.instructions, &Gen5BranchResolver);
    let starts: Vec<u32> = cfg.blocks.iter().map(|b| b.start_pc).collect();
    assert_eq!(starts, vec![0, 4, 0xC]);
    assert_eq!(cfg.successors[0], vec![2, 1]);
    assert_eq!(cfg.successors[1], vec![1]);
    assert!(cfg.successors[2].is_empty());
    assert_eq!(cfg.predecessors[2], vec![0]);
    assert!(cfg.loop_headers.contains(&1));
    assert_eq!(cfg.block_of(8), Some(1));
    // A backward S_BRANCH with no pending forward target ends the program.
    let p = program(&[0x7E00_0280, 0xBF82_FFFE, 0xDEAD_BEEF]);
    assert_eq!(p.instructions.len(), 2);
}

#[test]
fn exec_full_analysis() {
    // 0: s_mov_b64 s[2:3], exec ; 4: v_cmpx_gt_f32 v1, v2 ; 8: v_mov ;
    // c: s_mov_b64 exec, s[2:3] ; 10: v_mov ; 14: s_endpgm
    let p = program(&[0xBE82_047E, 0x7C28_0501, 0x7E00_0280, 0xBEFE_0402, 0x7E00_0280, S_ENDPGM]);
    let full = exec_full::analyze(&p, false);
    let mut pcs: Vec<u32> = full.into_iter().collect();
    pcs.sort();
    assert_eq!(pcs, vec![0, 4, 0x10, 0x14]);
    // A computed jump disables the analysis.
    let p = program(&[0xBE80_2002]);
    assert!(exec_full::analyze(&p, false).is_empty());
}

#[test]
fn wave64_half_masks() {
    // 0: v_cmp_gt_f32 vcc, v1, v2 ; 4: s_cbranch_vccz -> c ; 8: v_mov ; c: s_endpgm
    let p = program(&[0x7C08_0501, 0xBF86_0001, 0x7E00_0280, S_ENDPGM]);
    let plan = wave64_half_mask::analyze(&p, &HashSet::new()).unwrap();
    assert_eq!(plan.half_mask_writes.get(&0), Some(&VCC_LO));
    assert_eq!(plan.exact_pairs_before.get(&4), Some(&vec![VCC_LO]));
    // s_and_b64 s[4:5], vcc, exec stays a half mask when SCC is dead.
    let p = program(&[0x7C08_0501, 0x8784_7E6A, 0x7E00_0280, S_ENDPGM]);
    let plan = wave64_half_mask::analyze(&p, &HashSet::new()).unwrap();
    assert!(plan.exact_pairs_before.is_empty());
    // LDS write then read needs a barrier between the halves.
    let p = program(&[0xD834_0000, 0x0000_0100, 0xD8D8_0000, 0x0200_0000, S_ENDPGM]);
    let plan = wave64_half_mask::analyze(&p, &HashSet::new()).unwrap();
    assert!(plan.shared_memory_barriers_before.contains(&8));
}

#[test]
fn movrel_offsets() {
    // s_and_b32 m0, s4, 3 ; v_movrels_b32 v0, v1 ; s_endpgm
    let p = program(&[0x877C_8304, 0x7E00_8701, S_ENDPGM]);
    let r = movrel::analyze(&p);
    assert_eq!(r.get(&4), Some(&vec![0, 1, 2, 3]));
    // s_bfe_u32 s5, s4, (2 << 16) ; s_lshl_b32 m0, s5, 1 ; v_movrels
    let p = program(&[0x9385_FF04, 0x0002_0000, 0x8F7C_8105, 0x7E00_8701, S_ENDPGM]);
    assert_eq!(movrel::analyze(&p).get(&0xC), Some(&vec![0, 2, 4, 6]));
    // M0 from memory: unbounded.
    let p = program(&[0xF400_1F00, 0xFA00_0000, 0x7E00_8701, S_ENDPGM]);
    assert!(movrel::analyze(&p).is_empty());
}

#[test]
fn inline_constant_display() {
    assert_eq!(Operand::encoded(193).to_string(), "-1");
    assert_eq!(Operand::scalar(126).to_string(), "exec_lo");
    assert_eq!(Operand::scalar(110).to_string(), "ttmp2");
    assert_eq!(Operand::literal(0xABC).to_string(), "0x00000abc");
}
