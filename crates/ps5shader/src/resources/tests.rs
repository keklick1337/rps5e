// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Value-graph and planning tests on hand-assembled programs.

use super::*;
use super::format::image_format;
use crate::{decode_blob, Opcode, Program};
use std::cell::Cell;
use std::collections::HashMap;

// ---- tiny assembler ----

const S_ENDPGM: u32 = 0xBF81_0000;
const NULL: u32 = 0x7D;

fn sop1(op: u32, sdst: u32, ssrc: u32) -> u32 {
    0xBE80_0000 | (sdst << 16) | (op << 8) | ssrc
}

fn sop2(op: u32, sdst: u32, ssrc0: u32, ssrc1: u32) -> u32 {
    0x8000_0000 | (op << 23) | (sdst << 16) | (ssrc1 << 8) | ssrc0
}

fn sopc(op: u32, ssrc0: u32, ssrc1: u32) -> u32 {
    0xBF00_0000 | (op << 16) | (ssrc1 << 8) | ssrc0
}

fn sopp(op: u32, simm: i16) -> u32 {
    0xBF80_0000 | (op << 16) | simm as u16 as u32
}

/// s_load_dwordxN / s_buffer_load_dwordxN with an immediate offset.
fn smem(op: u32, sdata: u32, sbase: u32, offset: u32) -> [u32; 2] {
    [0xF400_0000 | (op << 18) | (sdata << 6) | (sbase >> 1), offset | (NULL << 25)]
}

/// image_sample vdata, vaddr, srsrc, ssamp (2D, dmask 0xF).
fn image_sample(vdata: u32, vaddr: u32, srsrc: u32, ssamp: u32) -> [u32; 2] {
    [0xF080_0F08, vaddr | (vdata << 8) | ((srsrc / 4) << 16) | ((ssamp / 4) << 21)]
}

/// buffer_load_dword vdata, vaddr, srsrc, 0 offen.
fn buffer_load_dword(vdata: u32, vaddr: u32, srsrc: u32) -> [u32; 2] {
    [0xE000_0000 | (0x0C << 18) | (1 << 12), vaddr | (vdata << 8) | ((srsrc / 4) << 16) | (0x80 << 24)]
}

const LITERAL: u32 = 0xFF;
const OP_S_MOV_B32: u32 = 0x03;
const OP_S_ADD_U32: u32 = 0x00;
const OP_S_ADDC_U32: u32 = 0x04;
const OP_S_CSELECT_B32: u32 = 0x0A;
const OP_S_LSHL_B32: u32 = 0x1E;
const OP_S_CMP_EQ_U32: u32 = 0x06;
const OP_S_BRANCH: u32 = 0x02;
const OP_S_CBRANCH_SCC1: u32 = 0x05;
const SMEM_LOAD_DWORD: u32 = 0x00;
const SMEM_LOAD_DWORDX2: u32 = 0x01;
const SMEM_LOAD_DWORDX4: u32 = 0x02;
const SMEM_LOAD_DWORDX8: u32 = 0x03;
const SMEM_BUFFER_LOAD_DWORD: u32 = 0x08;

fn assemble(parts: &[&[u32]]) -> Program {
    let words: Vec<u32> = parts.iter().flat_map(|p| p.iter().copied()).collect();
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    decode_blob(&bytes, 0x10_0000).unwrap_or_else(|e| panic!("decode: {e}"))
}

fn plan(program: Program, stage: ShaderStage, count: u32) -> ShaderResourcePlan {
    ShaderResourcePlan::extract(program, stage, 0x1234, 0, count, None, 64).unwrap_or_else(|e| panic!("plan: {e}"))
}

/// A valid 2D R8G8B8A8 T#.
fn t_sharp(address: u64) -> [u32; 8] {
    [(address >> 8) as u32, (56 << 20) | ((address >> 40) as u32 & 0xFF), 0, (image_format::TYPE_2D << 28) | 0xFAC, 0, 0, 0, 0]
}

fn memory_reader(words: &HashMap<u64, u32>) -> impl Fn(u64) -> Option<u32> + '_ {
    move |a| words.get(&a).copied()
}

#[test]
fn assembler_matches_decoder() {
    let p = assemble(&[
        &[sop1(OP_S_MOV_B32, 4, 5)],
        &[sop2(OP_S_ADD_U32, 0, 0, LITERAL), 0x10],
        &[sop2(OP_S_ADDC_U32, 1, 1, 128)],
        &[sop2(OP_S_CSELECT_B32, 4, 5, 6)],
        &[sopc(OP_S_CMP_EQ_U32, 2, 128)],
        &smem(SMEM_LOAD_DWORDX8, 16, 0, 0x20),
        &smem(SMEM_BUFFER_LOAD_DWORD, 30, 4, 8),
        &image_sample(0, 0, 16, 24),
        &buffer_load_dword(1, 2, 8),
        &[S_ENDPGM],
    ]);
    let ops: Vec<Opcode> = p.instructions.iter().map(|i| i.opcode).collect();
    assert_eq!(
        ops,
        vec![
            Opcode::SMovB32,
            Opcode::SAddU32,
            Opcode::SAddcU32,
            Opcode::SCselectB32,
            Opcode::SCmpEqU32,
            Opcode::SLoadDwordx8,
            Opcode::SBufferLoadDword,
            Opcode::ImageSample,
            Opcode::BufferLoadDword,
            Opcode::SEndpgm
        ]
    );
}

#[test]
fn interning_and_folding() {
    let mut g = ScalarValueGraph::build(assemble(&[&[S_ENDPGM]]), 0, 4, None, 64).unwrap();
    let (a, b) = (g.constant(3), g.constant(4));
    assert_eq!(g.constant(3), a);
    let sum = g.operation(ScalarOp::IAdd32, ValueType::U32, &[a, b]);
    assert_eq!(g.constant_u32(sum), Some(7));
    let u0 = g.user_data(0);
    let x = g.operation(ScalarOp::IAdd32, ValueType::U32, &[u0, a]);
    assert_eq!(g.operation(ScalarOp::IAdd32, ValueType::U32, &[u0, a]), x, "interned");
    assert_eq!(g.kind(x), ValueKind::Operation);
    // Extract of a constructed pair is that half.
    let pair = g.operation(ScalarOp::Construct64, ValueType::U64, &[u0, b]);
    let one = g.constant(1);
    assert_eq!(g.operation(ScalarOp::Extract64, ValueType::U32, &[pair, one]), b);
    // The low half of a carrying add is the plain add.
    let carry = g.operation(ScalarOp::AddCarry32, ValueType::U64, &[u0, a]);
    let zero = g.constant(0);
    assert_eq!(g.operation(ScalarOp::Extract64, ValueType::U32, &[carry, zero]), x);
    // Identities that hold with an undefined input.
    let u = g.undefined(ValueType::U32);
    assert_eq!(g.operation(ScalarOp::And32, ValueType::U32, &[u, zero]), zero);
    let shaped = g.operation(ScalarOp::IAdd32, ValueType::U32, &[u, a]);
    assert_eq!(g.kind(shaped), ValueKind::Operation);
    // Selects on constants fold.
    let t = g.constant_bool(true);
    assert_eq!(g.select(t, a, b), a);
    let c = g.user_data(1);
    assert_eq!(g.select(c, a, a), a);
    // Equivalence: undefined never, structure yes.
    assert!(!g.equivalent(u, u));
    assert!(g.equivalent(x, x));
    // A substituted value is rebuilt and folded.
    let replacements = HashMap::from([(u0, g.constant(10))]);
    let mut memo = HashMap::new();
    let r = g.substitute(x, &replacements, &mut memo);
    assert_eq!(g.constant_u32(r), Some(13));
}

/// Distinct image/sampler combinations are not limited: 9 T#s and 8 S#s
/// of a table sampled in all 72 combinations give 72 pairs, each with its
/// own image, sampler and first use.
#[test]
fn image_sampler_pairs_are_not_limited() {
    let mut parts: Vec<Vec<u32>> = Vec::new();
    for pair in 0..72u32 {
        parts.push(smem(SMEM_LOAD_DWORDX8, 16, 2, (pair / 8) * 0x20).to_vec());
        parts.push(smem(SMEM_LOAD_DWORDX4, 24, 2, 0x200 + (pair % 8) * 0x10).to_vec());
        parts.push(image_sample(0, 0, 16, 24).to_vec());
    }
    parts.push(vec![S_ENDPGM]);
    let refs: Vec<&[u32]> = parts.iter().map(|p| p.as_slice()).collect();
    let p = plan(assemble(&refs), ShaderStage::Pixel, 4);
    assert_eq!((p.info.images.len(), p.info.samplers.len(), p.info.sampled_pairs.len()), (9, 8, 72));
    let mut seen: Vec<(u32, u32)> = p.info.sampled_pairs.iter().map(|q| (q.image, q.sampler)).collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), 72, "every combination keeps its own pair");
}

#[test]
fn image_from_user_data_is_static() {
    // image_sample v0, v0, s[0:7], s[8:11]
    let p = plan(assemble(&[&image_sample(0, 0, 0, 8), &[S_ENDPGM]]), ShaderStage::Pixel, 12);
    assert_eq!(p.info.images.len(), 1);
    assert_eq!(p.info.samplers.len(), 1);
    assert_eq!(p.info.sampled_pairs.len(), 1);
    assert!(p.table_reads.is_empty());
    let source = &p.descriptor_sources[p.info.images[0].source as usize];
    for (i, &d) in source.dwords.iter().enumerate() {
        assert_eq!(p.graph.kind(d), ValueKind::UserData);
        assert_eq!(p.graph.node(d).payload, i as u64);
    }
    let memory = &p.memory().entries()[0];
    assert_eq!((memory.resource, memory.sampler), (0, 0));

    let mut user_data = vec![0u32; 12];
    user_data[..8].copy_from_slice(&t_sharp(0x1_0000_0000));
    user_data[8] = 0x24; // clamp bits set: border colour kept
    user_data[11] = 0x1234_5678;
    let inputs = ResourceRuntimeInputs::new(&user_data, 0);
    let (snapshot, specialization) = materialize(&p, &inputs).unwrap();
    assert_eq!(snapshot.images[0][..], (t_sharp(0x1_0000_0000))[..]);
    assert_eq!(snapshot.samplers[0][..], (user_data[8..12].to_vec())[..]);
    assert_eq!(specialization.images[0].numeric_class, ImageNumericClass::Float);
    assert_eq!(specialization.images[0].dimension, ImageDimension::Dim2D);

    let specialized = apply_specialization(&p, &specialization).unwrap();
    let registers = BindingLayout::collect_user_data_registers(p.program(), 0, 12);
    assert_eq!(registers, (0..12).collect::<Vec<_>>());
    let layout = BindingLayout::allocate(&specialized.info, registers.clone(), false, false, false, 0, false, false).unwrap();
    assert_eq!(layout.push_data_start_dword, 0);
    assert_eq!(layout.shader_data_dword_count(), 12);
    assert!(layout.find(DescriptorBindingKind::Samplers).is_some());
    let image = layout.descriptors.iter().find(|d| matches!(d.kind, DescriptorBindingKind::Image(_))).unwrap();
    assert_eq!(image_binding::describe(image.kind).0, ImageResourceClass::Sampled);
    layout.validate(&specialized.info, &registers, false, false, false, p.hash, p.stage).unwrap();

    // An unsupported format is bound as a null texture.
    user_data[1] = 4 << 20;
    let (snapshot, _) = materialize(&p, &ResourceRuntimeInputs::new(&user_data, 0)).unwrap();
    assert!(snapshot.images[0].iter().all(|&w| w == 0));

    // A T# of TYPE 0 or 1 (no resource type) with a base reads 0: a null
    // texture (as the hardware returns for it), not a failed draw.
    user_data[..8].copy_from_slice(&t_sharp(0x1_0000_0000));
    for ty in [0u32, 1] {
        user_data[3] = (user_data[3] & 0x0FFF_FFFF) | ty << 28;
        let (snapshot, specialization) = materialize(&p, &ResourceRuntimeInputs::new(&user_data, 0)).unwrap();
        assert!(snapshot.images[0].iter().all(|&w| w == 0), "TYPE {ty}");
        assert_eq!((specialization.images[0].numeric_class, specialization.images[0].dimension), (ImageNumericClass::Float, ImageDimension::Dim2D));
    }
}

#[test]
fn undefined_sampler_fails_the_plan() {
    // The sampler comes from s[8:11], outside the user data: an undefined
    // S# with no control-dependent origin is a hard failure.
    let err = ShaderResourcePlan::extract(assemble(&[&image_sample(0, 0, 0, 8), &[S_ENDPGM]]), ShaderStage::Pixel, 1, 0, 8, None, 64)
        .unwrap_err();
    assert!(err.message.contains("SamplerHandle dword 0"), "{err}");
}

/// A V# whose record count the program computes (here a lane value read
/// with v_readfirstlane) binds the buffer unbounded: base and stride come
/// from user data, the access checks its own record count.
#[test]
fn computed_buffer_record_counts_bind_unbounded() {
    // v_readfirstlane_b32 s6, v0 ; buffer_load_dword v1, v2, s[4:7]
    let program = || assemble(&[&[0x7E0C_0500], &buffer_load_dword(1, 2, 4), &[S_ENDPGM]]);
    let p = plan(program(), ShaderStage::Compute, 8);
    assert_eq!(p.info.buffers.len(), 1);
    let source = &p.descriptor_sources[p.info.buffers[0].source as usize];
    assert_eq!(p.graph.constant_u32(source.dwords[2]), Some(u32::MAX));
    assert!(source.dwords.iter().enumerate().filter(|&(i, _)| i != 2).all(|(_, &d)| p.graph.kind(d) == ValueKind::UserData));
}

#[test]
fn extended_user_data_reads_are_flattened() {
    // s_load_dwordx8 s[16:23], s[2:3], 0x20 ; s_load_dwordx4 s[24:27], s[2:3], 0x40
    // image_sample v0, v0, s[16:23], s[24:27]
    let p = plan(
        assemble(&[&smem(SMEM_LOAD_DWORDX8, 16, 2, 0x20), &smem(SMEM_LOAD_DWORDX4, 24, 2, 0x40), &image_sample(0, 0, 16, 24), &[S_ENDPGM]]),
        ShaderStage::Compute,
        4,
    );
    assert_eq!(p.table_reads.len(), 12);
    assert_eq!(p.flattened_slot_by_memory_index.len(), 12);
    assert!(p.memory().entries()[..12].iter().all(|m| m.planning_only));
    let source = &p.descriptor_sources[p.info.images[0].source as usize];
    assert!(source.dwords.iter().all(|&d| p.graph.kind(d) == ValueKind::ResourceTableWord));

    let eud = 0x2_0000_0000u64;
    let mut words = HashMap::new();
    for (i, w) in t_sharp(0x3_0000_0000).iter().enumerate() {
        words.insert(eud + 0x20 + i as u64 * 4, *w);
    }
    for i in 0..4u64 {
        words.insert(eud + 0x40 + i * 4, 0x100 + i as u32);
    }
    let reader = memory_reader(&words);
    let user_data = [0, 0, eud as u32, (eud >> 32) as u32];
    let mut inputs = ResourceRuntimeInputs::new(&user_data, 0);
    inputs.read_memory = Some(&reader);
    inputs.read_clean_memory = Some(&reader);
    let (snapshot, _) = materialize(&p, &inputs).unwrap();
    assert_eq!(snapshot.images[0][..], (t_sharp(0x3_0000_0000))[..]);
    assert_eq!(snapshot.samplers[0][..], (vec![0x100, 0x101, 0x102, 0x103])[..]);
    assert_eq!(snapshot.flattened_resource_table.len(), 12);
    assert_eq!(snapshot.flattened_resource_table[..8], t_sharp(0x3_0000_0000));

    // Unreadable memory fails the materialisation.
    let empty = HashMap::new();
    let none = memory_reader(&empty);
    inputs.read_memory = Some(&none);
    assert!(materialize(&p, &inputs).is_err());
}

#[test]
fn address_arithmetic_is_evaluated() {
    // s_add_u32 s0, s0, 0x10 ; s_addc_u32 s1, s1, 0 ; s_load_dwordx4 s[4:7], s[0:1], 0
    // buffer_load_dword v1, v2, s[4:7]
    let p = plan(
        assemble(&[
            &[sop2(OP_S_ADD_U32, 0, 0, LITERAL), 0x10],
            &[sop2(OP_S_ADDC_U32, 1, 1, 128)],
            &smem(SMEM_LOAD_DWORDX4, 4, 0, 0),
            &buffer_load_dword(1, 2, 4),
            &[S_ENDPGM],
        ]),
        ShaderStage::Compute,
        2,
    );
    assert_eq!(p.info.buffers.len(), 1);
    assert_eq!(p.table_reads.len(), 4);
    let buffer_memory = p.memory().entries().iter().find(|m| m.kind == MemoryResourceKind::Buffer).unwrap();
    assert_eq!(buffer_memory.buffer_descriptor, Some(GuestBufferDescriptor::static_resource(0)));
    // The base crosses a 4 GiB boundary: the carry must reach the high dword.
    let base = 0x1_FFFF_FFF8u64;
    let target = base + 0x10;
    let mut words = HashMap::new();
    let v_sharp = [0x4000_0000u32, 0x0000_0002, 0x100, 0x2000_0000 | IDENTITY_DESTINATION_SELECT];
    for (i, w) in v_sharp.iter().enumerate() {
        words.insert(target + i as u64 * 4, *w);
    }
    let reader = memory_reader(&words);
    let user_data = [base as u32, (base >> 32) as u32];
    let mut inputs = ResourceRuntimeInputs::new(&user_data, 0);
    inputs.read_memory = Some(&reader);
    let (snapshot, specialization) = materialize(&p, &inputs).unwrap();
    assert_eq!(snapshot.buffers[0][..], (v_sharp.to_vec())[..]);
    // Stride 0; OOB_SELECT 2 (dword3 bits 28..29) in bits 24..25.
    assert_eq!(specialization.buffers[0].packed_stride, 2 << 24);
    assert_eq!(specialization.base_buffer_count, 1);
}

#[test]
fn scalar_buffer_reads_bind_the_buffer() {
    // s_buffer_load_dword s30, s[4:7], 8 (the V# is user data s4..s7)
    let p = plan(assemble(&[&smem(SMEM_BUFFER_LOAD_DWORD, 30, 4, 8), &[S_ENDPGM]]), ShaderStage::Compute, 8);
    // The read itself has no consumer, but the V# is a host buffer.
    assert_eq!(p.info.buffers.len(), 1);
    assert!(p.info.buffers[0].scalar);
    assert_eq!(p.info.buffers[0].max_byte_extent, 12);
}

#[test]
fn uniform_join_becomes_a_select() {
    // s_cmp_eq_u32 s2, 0 ; s_cbranch_scc1 +1 ; s_mov_b32 s0, s3 ; (join)
    // image_sample v0, v0, s[0:7], s[8:11]
    let p = plan(
        assemble(&[
            &[sopc(OP_S_CMP_EQ_U32, 2, 128)],
            &[sopp(OP_S_CBRANCH_SCC1, 1)],
            &[sop1(OP_S_MOV_B32, 0, 3)],
            &image_sample(0, 0, 0, 8),
            &[S_ENDPGM],
        ]),
        ShaderStage::Pixel,
        12,
    );
    let source = &p.descriptor_sources[p.info.images[0].source as usize];
    let d0 = source.dwords[0];
    assert_eq!(p.graph.kind(d0), ValueKind::Select, "{}", p.graph.describe(d0, 3));
    assert!(p.validate_runtime_value(d0));
    let mut user_data = vec![0u32; 12];
    user_data[0] = 0xAAAA;
    user_data[3] = 0xBBBB;
    let evaluator = RuntimeValueEvaluator::new(&p, ResourceRuntimeInputs::new(&user_data, 0));
    assert_eq!(evaluator.evaluate(d0), Some(0xAAAA), "s2 == 0: branch taken, s0 kept");
    user_data[2] = 1;
    let evaluator = RuntimeValueEvaluator::new(&p, ResourceRuntimeInputs::new(&user_data, 0));
    assert_eq!(evaluator.evaluate(d0), Some(0xBBBB));
}

/// SOPK instruction (s_movk / s_cmpk_* / s_addk_i32 ...).
fn sopk(op: u32, sdst: u32, simm: u16) -> u32 {
    0xB000_0000 | (op << 23) | (sdst << 16) | simm as u32
}

const OP_S_CMPK_EQ_U32: u32 = 0x09;
const OP_S_CMPK_LG_U32: u32 = 0x0A;
const OP_S_CMPK_LT_I32: u32 = 0x07;
const OP_S_CMPK_LE_U32: u32 = 0x0E;
const OP_S_ADDK_I32: u32 = 0x0F;
const OP_S_BFM_B32: u32 = 0x24;

/// The first T# dword of `program`'s image sample, evaluated for each
/// user-data set.
fn image_dword0(program: Program, cases: &[(&[(usize, u32)], u32)]) {
    let p = plan(program, ShaderStage::Pixel, 12);
    let d0 = p.descriptor_sources[p.info.images[0].source as usize].dwords[0];
    assert!(p.validate_runtime_value(d0), "{}", p.graph.describe(d0, 4));
    for &(sets, expected) in cases {
        let mut user_data = vec![0u32; 12];
        user_data[0] = 0xAAAA;
        user_data[3] = 0xBBBB;
        for &(i, v) in sets {
            user_data[i] = v;
        }
        let evaluator = RuntimeValueEvaluator::new(&p, ResourceRuntimeInputs::new(&user_data, 0));
        assert_eq!(evaluator.evaluate(d0), Some(expected), "user data {sets:x?}: {}", p.graph.describe(d0, 4));
    }
}

#[test]
fn sopk_compares_set_scc_and_keep_their_register() {
    // s_cmpk_eq_u32 s2, 0x8001 (zero-extended) ; s_cbranch_scc1 +1 ;
    // s_mov_b32 s0, s3 ; image_sample v0, v0, s[0:7], s[8:11]
    let branch = |cmpk: u32| {
        assemble(&[&[cmpk], &[sopp(OP_S_CBRANCH_SCC1, 1)], &[sop1(OP_S_MOV_B32, 0, 3)], &image_sample(0, 0, 0, 8), &[S_ENDPGM]])
    };
    image_dword0(branch(sopk(OP_S_CMPK_EQ_U32, 2, 0x8001)), &[(&[(2, 0x8001)], 0xAAAA), (&[(2, 0xFFFF_8001)], 0xBBBB)]);
    // s_cmpk_le_u32 s2, 0xffff: 0xffff <= 0xffff, 0x10000 is not.
    image_dword0(branch(sopk(OP_S_CMPK_LE_U32, 2, 0xFFFF)), &[(&[(2, 0xFFFF)], 0xAAAA), (&[(2, 0x1_0000)], 0xBBBB)]);
    // s_cmpk_lt_i32 s2, -1 (sign-extended): -2 < -1, 0x7fff_ffff is not.
    image_dword0(branch(sopk(OP_S_CMPK_LT_I32, 2, 0xFFFF)), &[(&[(2, 0xFFFF_FFFE)], 0xAAAA), (&[(2, 0x7FFF_FFFF)], 0xBBBB)]);
    // The compared register is only read: s_cmpk_lg_u32 s0, 0 leaves s0
    // (the T# dword) as it was.
    let p = plan(assemble(&[&[sopk(OP_S_CMPK_LG_U32, 0, 0)], &image_sample(0, 0, 0, 8), &[S_ENDPGM]]), ShaderStage::Pixel, 12);
    let d0 = p.descriptor_sources[p.info.images[0].source as usize].dwords[0];
    assert_eq!(p.graph.kind(d0), ValueKind::UserData, "{}", p.graph.describe(d0, 3));
}

#[test]
fn sopk_add_sets_the_signed_overflow() {
    // s_addk_i32 s2, 1 ; s_cbranch_scc1 +1 ; s_mov_b32 s0, s3
    let p = assemble(&[&[sopk(OP_S_ADDK_I32, 2, 1)], &[sopp(OP_S_CBRANCH_SCC1, 1)], &[sop1(OP_S_MOV_B32, 0, 3)], &image_sample(0, 0, 0, 8), &[S_ENDPGM]]);
    image_dword0(p, &[(&[(2, 0x7FFF_FFFF)], 0xAAAA), (&[(2, 5)], 0xBBBB), (&[(2, 0xFFFF_FFFF)], 0xBBBB)]);
    // s_addk_i32 s2, -2 (0xfffe): 0x8000_0000 - 2 overflows.
    let p = assemble(&[&[sopk(OP_S_ADDK_I32, 2, 0xFFFE)], &[sopp(OP_S_CBRANCH_SCC1, 1)], &[sop1(OP_S_MOV_B32, 0, 3)], &image_sample(0, 0, 0, 8), &[S_ENDPGM]]);
    image_dword0(p, &[(&[(2, 0x8000_0000)], 0xAAAA), (&[(2, 1)], 0xBBBB)]);
}

#[test]
fn bfm_b32_truncates_the_field_at_bit_31() {
    // s_bfm_b32 s0, s4, s5: ((1 << s4) - 1) << s5, the bits past 31 dropped.
    let p = assemble(&[&[sop2(OP_S_BFM_B32, 0, 4, 5)], &image_sample(0, 0, 0, 8), &[S_ENDPGM]]);
    image_dword0(p, &[(&[(4, 8), (5, 28)], 0xF000_0000), (&[(4, 4), (5, 4)], 0xF0), (&[(4, 0), (5, 9)], 0)]);
}

#[test]
fn float_inline_constants_of_64_bit_operands_are_f64() {
    // s_mov_b64 s[0:1], 1.0 (inline 242): s1 = 0x3ff00000, s0 = 0.
    let p = plan(assemble(&[&[sop1(0x04, 0, 242)], &image_sample(0, 0, 0, 8), &[S_ENDPGM]]), ShaderStage::Pixel, 12);
    let dwords = p.descriptor_sources[p.info.images[0].source as usize].dwords.clone();
    assert_eq!(p.graph.constant_u32(dwords[0]), Some(0));
    assert_eq!(p.graph.constant_u32(dwords[1]), Some(0x3FF0_0000));
}

#[test]
fn loop_invariant_phi_resolves() {
    // L: s_add_u32 s20, s20, 1 ; s_cmp_eq_u32 s20, 4 ; s_cbranch_scc1 +1 ; s_branch L
    // image_sample v0, v0, s[0:7], s[8:11]
    let p = plan(
        assemble(&[
            &[sop2(OP_S_ADD_U32, 20, 20, 129)],
            &[sopc(OP_S_CMP_EQ_U32, 20, 132)],
            &[sopp(OP_S_CBRANCH_SCC1, 1)],
            &[sopp(OP_S_BRANCH, -4)],
            &image_sample(0, 0, 0, 8),
            &[S_ENDPGM],
        ]),
        ShaderStage::Pixel,
        12,
    );
    let source = &p.descriptor_sources[p.info.images[0].source as usize];
    let d0 = source.dwords[0];
    // The loop header merges the entry user data with itself.
    let resolved = p.graph.resolve_invariant_phi(d0).unwrap();
    assert_eq!(p.graph.kind(resolved), ValueKind::UserData);
    assert!(p.validate_runtime_value(d0));
    // The loop counter is a phi that merges different values.
    let counter = p.graph.values().find(|&v| p.graph.kind(v) == ValueKind::Phi && p.graph.resolve_invariant_phi(v).is_none());
    assert!(counter.is_some());
}

#[test]
fn shader_base_feeds_getpc_addresses() {
    // s_getpc_b64 s[0:1] ; s_load_dwordx4 s[4:7], s[0:1], 0x100 ; buffer_load_dword v1, v2, s[4:7]
    let p = plan(
        assemble(&[&[sop1(0x1F, 0, 0)], &smem(SMEM_LOAD_DWORDX4, 4, 0, 0x100), &buffer_load_dword(1, 2, 4), &[S_ENDPGM]]),
        ShaderStage::Compute,
        0,
    );
    assert_eq!(p.program().instructions[0].opcode, Opcode::SGetpcB64);
    assert!(BindingLayout::reads_shader_base(p.program()));
    let shader_base = 0x5000_0000u64;
    let at = shader_base + 4 + 0x100;
    let mut words = HashMap::new();
    for i in 0..4 {
        words.insert(at + i * 4, 0x77 + i as u32);
    }
    let reader = memory_reader(&words);
    let mut inputs = ResourceRuntimeInputs::new(&[], shader_base);
    inputs.read_memory = Some(&reader);
    let (snapshot, _) = materialize(&p, &inputs).unwrap();
    assert_eq!(snapshot.buffers[0][..], (vec![0x77, 0x78, 0x79, 0x7A])[..]);
}

#[test]
fn undefined_descriptors_fail_with_the_culprit() {
    // buffer_load_dword v1, v2, s[40:43] with no user data: the V# is
    // undefined and the plan names the instruction.
    let err = ShaderResourcePlan::extract(assemble(&[&buffer_load_dword(1, 2, 40), &[S_ENDPGM]]), ShaderStage::Compute, 7, 0, 4, None, 64)
        .unwrap_err();
    assert!(err.message.contains("BufferLoadDword"), "{err}");
    assert!(err.message.contains("hash=0x0000000000000007"), "{err}");
}

/// Guest memory and readers for the materialisation cache tests.
struct TestMemory {
    words: std::cell::RefCell<HashMap<u64, u32>>,
    /// Addresses the resident reader refuses (GPU-owned).
    gpu_owned: std::cell::RefCell<std::collections::HashSet<u64>>,
    word_reads: Cell<u32>,
}

impl TestMemory {
    fn new() -> Self {
        TestMemory { words: Default::default(), gpu_owned: Default::default(), word_reads: Cell::new(0) }
    }

    fn set(&self, a: u64, w: u32) {
        self.words.borrow_mut().insert(a, w);
    }

    fn set_all(&self, a: u64, words: &[u32]) {
        for (i, &w) in words.iter().enumerate() {
            self.set(a + i as u64 * 4, w);
        }
    }

    fn read(&self, a: u64) -> Option<u32> {
        self.word_reads.set(self.word_reads.get() + 1);
        self.words.borrow().get(&a).copied()
    }

    fn resident(&self, a: u64, out: &mut [u8]) -> bool {
        for (i, chunk) in out.chunks_mut(4).enumerate() {
            let at = a + i as u64 * 4;
            if self.gpu_owned.borrow().contains(&at) {
                return false;
            }
            match self.words.borrow().get(&at) {
                Some(w) => chunk.copy_from_slice(&w.to_le_bytes()),
                None => return false,
            }
        }
        true
    }
}

/// image_sample through a T# and S# loaded from the table at s[2:3]
/// (+0x20, +0x40), a table-only dword (+0x60) and a buffer V# in user
/// data s[4:7].
fn srt_test_plan() -> ShaderResourcePlan {
    plan(
        assemble(&[
            &smem(SMEM_LOAD_DWORDX8, 16, 2, 0x20),
            &smem(SMEM_LOAD_DWORDX4, 24, 2, 0x40),
            &smem(SMEM_LOAD_DWORD, 30, 2, 0x60),
            &image_sample(0, 0, 16, 24),
            &buffer_load_dword(1, 2, 4),
            &[S_ENDPGM],
        ]),
        ShaderStage::Compute,
        8,
    )
}

fn v_sharp(address: u64, stride: u32, records: u32) -> [u32; 4] {
    [address as u32, (address >> 32) as u32 | (stride << 16), records, 0x0002_7FAC]
}

fn srt_user_data(table: u64, buffer: [u32; 4]) -> Vec<u32> {
    let mut u = vec![0, 0, table as u32, (table >> 32) as u32];
    u.extend_from_slice(&buffer);
    u
}

fn fill_table(m: &TestMemory, table: u64, t: [u32; 8], s: [u32; 4], constant: u32) {
    m.set_all(table + 0x20, &t);
    m.set_all(table + 0x40, &s);
    m.set(table + 0x60, constant);
}

/// Cached materialisation against a fresh one.
fn cached_matches_fresh(cache: &mut ResourceMaterializationCache, p: &ShaderResourcePlan, m: &TestMemory, user_data: &[u32]) -> (std::sync::Arc<ResourceSnapshot>, std::sync::Arc<ResourceSpecialization>) {
    let read = |a: u64| m.read(a);
    let resident = |a: u64, out: &mut [u8], _clean: bool| m.resident(a, out);
    let mut inputs = ResourceRuntimeInputs::new(user_data, 0);
    inputs.read_memory = Some(&read);
    inputs.read_clean_memory = Some(&read);
    inputs.read_resident_memory = Some(&resident);
    let fresh = materialize(p, &inputs).unwrap();
    let cached = cache.materialize(p, &inputs, &resident).unwrap();
    assert_eq!(*cached.0, fresh.0);
    assert_eq!(*cached.1, fresh.1);
    cached
}

#[test]
fn replay_hits_when_only_addresses_change() {
    let p = srt_test_plan();
    assert!(ReplayProgram::eligible(&p));
    let m = TestMemory::new();
    let mut cache = ResourceMaterializationCache::default();
    let sampler = [0x24, 0, 0, 0];
    // Draw 1: table A, buffer at 0x5000.
    let (a, b) = (0x2_0000_0000u64, 0x2_0001_0000u64);
    fill_table(&m, a, t_sharp(0x3_0000_0000), sampler, 7);
    let (s1, spec1) = cached_matches_fresh(&mut cache, &p, &m, &srt_user_data(a, v_sharp(0x5000, 16, 64)));
    assert_eq!(cache.statistics().replay_templates, 1);
    // The same draw again: identical words, the same snapshot.
    m.word_reads.set(0);
    let (s1b, _) = cached_matches_fresh(&mut cache, &p, &m, &srt_user_data(a, v_sharp(0x5000, 16, 64)));
    assert!(std::sync::Arc::ptr_eq(&s1, &s1b));
    assert_eq!(cache.statistics().replay_same, 1);
    // Draw 2: another table (a ring allocation) holding another texture of
    // the same format, a moved constant buffer with another size, another
    // table constant: patched into the template, same specialization.
    fill_table(&m, b, t_sharp(0x3_0040_0000), sampler, 9);
    let (s2, spec2) = cached_matches_fresh(&mut cache, &p, &m, &srt_user_data(b, v_sharp(0x9000, 16, 128)));
    assert!(std::sync::Arc::ptr_eq(&spec1, &spec2), "the specialization is reused");
    assert_eq!(cache.statistics().replay_patched, 1);
    assert_eq!(cache.statistics().replay_templates, 1);
    assert_eq!(s2.images[0][..], (t_sharp(0x3_0040_0000))[..]);
    assert_eq!(s2.buffers[0][..], (v_sharp(0x9000, 16, 128))[..]);
    assert_eq!(s2.flattened_resource_table[12], 9);
    // Hits never use the synchronising word reader (only the fresh
    // reference materialisations above did).
    let before = m.word_reads.get();
    let resident = |a: u64, out: &mut [u8], _clean: bool| m.resident(a, out);
    let user = srt_user_data(a, v_sharp(0x5000, 16, 64));
    let read = |a: u64| m.read(a);
    let mut inputs = ResourceRuntimeInputs::new(&user, 0);
    inputs.read_memory = Some(&read);
    cache.materialize(&p, &inputs, &resident).unwrap();
    assert_eq!(m.word_reads.get(), before);
    assert_eq!(cache.statistics().replay_fallbacks, 0);
}

/// Patched replays write into snapshots of earlier patches no one holds
/// any more: every result still equals a fresh materialisation, and a
/// snapshot still held is never written over.
#[test]
fn replay_patches_reuse_only_snapshots_no_one_holds() {
    let p = srt_test_plan();
    let m = TestMemory::new();
    let mut cache = ResourceMaterializationCache::default();
    let sampler = [0x24, 0, 0, 0];
    let tables: Vec<u64> = (0..4).map(|i| 0x2_0000_0000u64 + i * 0x1_0000).collect();
    for (i, &t) in tables.iter().enumerate() {
        fill_table(&m, t, t_sharp(0x3_0000_0000 + i as u64 * 0x40_0000), sampler, i as u32);
    }
    let (held, _) = cached_matches_fresh(&mut cache, &p, &m, &srt_user_data(tables[0], v_sharp(0x5000, 16, 64)));
    let kept = (*held).clone();
    for k in 1..200u64 {
        let t = tables[(k % 4) as usize];
        // Every result is dropped at once but `held`.
        cached_matches_fresh(&mut cache, &p, &m, &srt_user_data(t, v_sharp(0x5000 + k * 0x100, 16, 64)));
    }
    assert_eq!(*held, kept, "a snapshot still held is left alone");
    if std::env::var("RPS5E_PATCH_IN_PLACE").is_err() {
        assert!(cache.statistics().replay_recycled > 100, "{:?}", cache.statistics());
    }
}

#[test]
fn replay_specialization_invalidates_on_decision_fields() {
    let p = srt_test_plan();
    let m = TestMemory::new();
    let mut cache = ResourceMaterializationCache::default();
    let table = 0x2_0000_0000u64;
    let sampler = [0x24, 0, 0, 0];
    let buffer = v_sharp(0x5000, 16, 64);
    fill_table(&m, table, t_sharp(0x3_0000_0000), sampler, 7);
    let (_, float) = cached_matches_fresh(&mut cache, &p, &m, &srt_user_data(table, buffer));
    assert_eq!(float.images[0].numeric_class, ImageNumericClass::Float);
    // Format: a sint texture (needs a point-sampling copy of the sampler).
    let mut t = t_sharp(0x3_0000_0000);
    t[1] = (t[1] & !(0x1FF << 20)) | (61 << 20);
    m.set_all(table + 0x20, &t);
    let (snapshot, sint) = cached_matches_fresh(&mut cache, &p, &m, &srt_user_data(table, buffer));
    assert_eq!(sint.images[0].numeric_class, ImageNumericClass::Sint);
    assert_eq!(snapshot.samplers[0][..], (sampler.to_vec())[..]);
    assert_eq!(cache.statistics().replay_templates, 2);
    // Type: a 3D texture.
    let mut t = t_sharp(0x3_0000_0000);
    t[3] = (t[3] & 0x0FFF_FFFF) | (image_format::TYPE_3D << 28);
    m.set_all(table + 0x20, &t);
    let (_, volume) = cached_matches_fresh(&mut cache, &p, &m, &srt_user_data(table, buffer));
    assert_eq!(volume.images[0].dimension, ImageDimension::Dim3D);
    assert_eq!(cache.statistics().replay_templates, 3);
    // An unsupported format binds a null texture; another address with the
    // same unsupported format is still zeroed by the patch.
    let mut t = t_sharp(0x3_0000_0000);
    t[1] = (t[1] & !(0x1FF << 20)) | (4 << 20);
    m.set_all(table + 0x20, &t);
    let (snapshot, _) = cached_matches_fresh(&mut cache, &p, &m, &srt_user_data(table, buffer));
    assert!(snapshot.images[0].iter().all(|&w| w == 0));
    let mut t2 = t;
    t2[0] = 0x0123_4567;
    m.set_all(table + 0x20, &t2);
    let patched = cache.statistics().replay_patched;
    let (snapshot, _) = cached_matches_fresh(&mut cache, &p, &m, &srt_user_data(table, buffer));
    assert!(snapshot.images[0].iter().all(|&w| w == 0));
    assert_eq!(cache.statistics().replay_patched, patched + 1);
    // Buffer stride: a new specialization.
    m.set_all(table + 0x20, &t_sharp(0x3_0000_0000));
    let (_, strided) = cached_matches_fresh(&mut cache, &p, &m, &srt_user_data(table, v_sharp(0x5000, 32, 64)));
    assert_eq!(strided.buffers[0].packed_stride & 0x3FFF, 32);
    // Back to the first shape: its template is found again.
    let templates = cache.statistics().replay_templates;
    let (_, again) = cached_matches_fresh(&mut cache, &p, &m, &srt_user_data(table, buffer));
    assert!(std::sync::Arc::ptr_eq(&again, &float));
    assert_eq!(cache.statistics().replay_templates, templates);
}

#[test]
fn replay_reads_gpu_owned_words_through_the_word_reader() {
    let p = srt_test_plan();
    let m = TestMemory::new();
    let mut cache = ResourceMaterializationCache::default();
    let table = 0x2_0000_0000u64;
    fill_table(&m, table, t_sharp(0x3_0000_0000), [0x24, 0, 0, 0], 7);
    m.gpu_owned.borrow_mut().insert(table + 0x60);
    let (s, _) = cached_matches_fresh(&mut cache, &p, &m, &srt_user_data(table, v_sharp(0x5000, 16, 64)));
    assert_eq!(s.flattened_resource_table[12], 7);
    assert_eq!(cache.statistics().replay_templates, 1);
    // Unreadable everywhere: the recording path reports the failure.
    m.words.borrow_mut().remove(&(table + 0x60));
    let read = |a: u64| m.read(a);
    let resident = |a: u64, out: &mut [u8], _clean: bool| m.resident(a, out);
    let user = srt_user_data(table, v_sharp(0x5000, 16, 64));
    let mut inputs = ResourceRuntimeInputs::new(&user, 0);
    inputs.read_memory = Some(&read);
    inputs.read_clean_memory = Some(&read);
    assert!(cache.materialize(&p, &inputs, &resident).is_err());
    assert_eq!(cache.statistics().replay_fallbacks, 1);
}

/// [`srt_test_plan`] with its table-only dword at `constant` (0x50: right
/// after the S#, in the same block of consecutive reads).
fn srt_test_plan_with_constant_at(constant: u32) -> ShaderResourcePlan {
    plan(
        assemble(&[
            &smem(SMEM_LOAD_DWORDX8, 16, 2, 0x20),
            &smem(SMEM_LOAD_DWORDX4, 24, 2, 0x40),
            &smem(SMEM_LOAD_DWORD, 30, 2, constant),
            &image_sample(0, 0, 16, 24),
            &buffer_load_dword(1, 2, 4),
            &[S_ENDPGM],
        ]),
        ShaderStage::Compute,
        8,
    )
}

/// A replay through a resident reader refusing what `refuse` names, checked
/// against a fresh materialisation: the replayed snapshot, the resident
/// reads it asked for (address, bytes) and the cache counters.
fn replay_resident_calls(p: &ShaderResourcePlan, m: &TestMemory, user: &[u32], refuse: &dyn Fn(u64, usize) -> bool) -> (Result<ResourceSnapshot, ()>, Vec<(u64, usize)>, CacheStatistics) {
    let calls = std::cell::RefCell::new(Vec::new());
    let resident = |a: u64, out: &mut [u8], _clean: bool| {
        calls.borrow_mut().push((a, out.len()));
        !refuse(a, out.len()) && m.resident(a, out)
    };
    let read = |a: u64| m.read(a);
    let mut inputs = ResourceRuntimeInputs::new(user, 0);
    inputs.read_memory = Some(&read);
    inputs.read_clean_memory = Some(&read);
    inputs.read_resident_memory = Some(&resident);
    let fresh = materialize(p, &inputs).map(|r| r.0);
    calls.borrow_mut().clear();
    let mut cache = ResourceMaterializationCache::default();
    let cached = cache.materialize(p, &inputs, &resident).map(|r| (*r.0).clone());
    match (&fresh, &cached) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "replay equals a fresh materialisation"),
        (Err(_), Err(_)) => {}
        _ => panic!("fresh {:?} vs replayed {:?}", fresh.is_ok(), cached.is_ok()),
    }
    let calls = calls.into_inner();
    (cached.map_err(|_| ()), calls, cache.statistics())
}

/// Consecutive descriptor dwords read through one table pointer are one
/// resident read: the T# and S# (12 dwords) in one call, the lone table
/// dword in another.
#[test]
fn replay_reads_consecutive_dwords_as_one_range() {
    let p = srt_test_plan();
    let m = TestMemory::new();
    let table = 0x2_0000_0000u64;
    fill_table(&m, table, t_sharp(0x3_0000_0000), [0x24, 0, 0, 0], 7);
    let (s, mut calls, stats) = replay_resident_calls(&p, &m, &srt_user_data(table, v_sharp(0x5000, 16, 64)), &|_, _| false);
    assert_eq!(s.unwrap().flattened_resource_table[12], 7);
    assert_eq!(stats.replay_templates, 1);
    calls.sort_unstable();
    assert_eq!(calls, vec![(table + 0x20, 48), (table + 0x60, 4)]);
}

/// A range read the reader refuses (it crosses a boundary the reader
/// cannot serve in one piece) is read word by word, with the same words.
#[test]
fn replay_range_refused_is_read_word_by_word() {
    let p = srt_test_plan();
    let m = TestMemory::new();
    let table = 0x2_0000_0000u64 + 0x10;
    fill_table(&m, table, t_sharp(0x3_0000_0000), [0x24, 0, 0, 0], 7);
    // Pieces crossing a 64-byte boundary are refused.
    let straddles = |a: u64, n: usize| a / 64 != (a + n as u64 - 1) / 64;
    let before = m.word_reads.get();
    let (s, calls, stats) = replay_resident_calls(&p, &m, &srt_user_data(table, v_sharp(0x5000, 16, 64)), &straddles);
    // The word reader served the fresh reference alone (13 dwords).
    assert_eq!(m.word_reads.get() - before, 13);
    assert!(s.is_ok());
    assert_eq!(stats.replay_fallbacks, 0);
    assert!(calls.contains(&(table + 0x20, 48)), "{calls:x?}");
    // Every dword then once on its own through the resident reader.
    assert_eq!(calls.iter().filter(|c| c.1 == 4).count(), 13, "{calls:x?}");
}

/// A block holding a word the resident reader refuses (GPU-owned) or one
/// that is not mapped at all: the same result as word-by-word reads (the
/// synchronising reader for the GPU-owned word; a failure for the
/// unmapped one).
#[test]
fn replay_block_with_unreadable_words() {
    let p = srt_test_plan();
    let m = TestMemory::new();
    let table = 0x2_0000_0000u64;
    let sampler = [0x24, 0, 0, 0];
    fill_table(&m, table, t_sharp(0x3_0000_0000), sampler, 7);
    let user = srt_user_data(table, v_sharp(0x5000, 16, 64));
    m.gpu_owned.borrow_mut().insert(table + 0x44);
    let (s, _, stats) = replay_resident_calls(&p, &m, &user, &|_, _| false);
    assert_eq!(s.unwrap().samplers[0][..], sampler[..]);
    assert_eq!(stats.replay_fallbacks, 0);
    m.gpu_owned.borrow_mut().clear();
    // The last S# dword unmapped: both paths fail.
    m.words.borrow_mut().remove(&(table + 0x4C));
    let (s, _, _) = replay_resident_calls(&p, &m, &user, &|_, _| false);
    assert!(s.is_err());
    // The first T# dword unmapped (the block's start).
    m.set(table + 0x4C, 0);
    m.words.borrow_mut().remove(&(table + 0x20));
    let (s, _, _) = replay_resident_calls(&p, &m, &user, &|_, _| false);
    assert!(s.is_err());
}

/// A deferrable table dword inside a block: deferred while the GPU still
/// writes it (the rest of the block read word by word), read with the
/// block otherwise.
#[test]
fn replay_block_with_a_deferrable_dword() {
    let p = srt_test_plan_with_constant_at(0x50);
    assert!(ReplayProgram::eligible(&p));
    let m = TestMemory::new();
    let table = 0x2_0000_0000u64;
    m.set_all(table + 0x20, &t_sharp(0x3_0000_0000));
    m.set_all(table + 0x40, &[0x24, 0, 0, 0]);
    m.set(table + 0x50, 7);
    let user = srt_user_data(table, v_sharp(0x5000, 16, 64));
    let calls = std::cell::RefCell::new(Vec::new());
    let resident = |a: u64, out: &mut [u8], _clean: bool| {
        calls.borrow_mut().push((a, out.len()));
        m.resident(a, out)
    };
    let read = |a: u64| m.read(a);
    let probe = |a: u64| m.gpu_owned.borrow().contains(&a);
    let mut inputs = ResourceRuntimeInputs::new(&user, 0);
    inputs.read_memory = Some(&read);
    inputs.read_clean_memory = Some(&read);
    inputs.read_resident_memory = Some(&resident);
    inputs.gpu_written = Some(&probe);
    let mut cache = ResourceMaterializationCache::default();
    let (s, _) = cache.materialize(&p, &inputs, &resident).unwrap();
    assert_eq!(*s, materialize(&p, &inputs).unwrap().0);
    assert!(s.deferred_table_reads.is_empty());
    assert!(calls.borrow().contains(&(table + 0x20, 52)), "{:x?}", calls.borrow());
    // The GPU still writes the table dword: it is deferred.
    m.gpu_owned.borrow_mut().insert(table + 0x50);
    let (s, _) = cache.materialize(&p, &inputs, &resident).unwrap();
    assert_eq!(*s, materialize(&p, &inputs).unwrap().0);
    assert_eq!(s.deferred_table_reads, vec![(12, table + 0x50)]);
    assert_eq!(s.samplers[0][0], 0x24);
}

/// Over random memory with unmapped words, the program read in blocks
/// through the resident reader and read dword by dword through the word
/// reader agree.
#[test]
fn replay_blocks_agree_with_word_reads() {
    let p = srt_test_plan_with_constant_at(0x50);
    let program = ReplayProgram::compile(&p).unwrap();
    let mut scratch = ReplayScratch::default();
    for seed in 0..64u64 {
        let m = TestMemory::new();
        let table = 0x2_0000_0000u64 + seed * 0x34;
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        for k in 0..32u64 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            // A few words left unmapped.
            if x % 13 != 0 {
                m.set(table + k * 4, x as u32);
            }
        }
        let user = srt_user_data(table, v_sharp(0x5000, 16, 64));
        let resident = |a: u64, out: &mut [u8], _clean: bool| m.resident(a, out);
        let read = |a: u64| m.read(a);
        let blocks = program.run(&user, 0, &ReplayReader::Resident(&resident, true), None, &mut scratch);
        let words = program.run(&user, 0, &ReplayReader::Words(&read, Some(&read)), None, &mut scratch);
        match blocks {
            Ok(b) => assert_eq!(Ok(b), words, "seed {seed}"),
            Err(e) => {
                assert_eq!(e, ReplayAbort::Unreadable);
                assert!(words.is_err(), "seed {seed}");
            }
        }
    }
}

#[test]
fn recording_cache_hits_and_refreshes() {
    // Plans outside the replay path (here: a resource branch on a table
    // word would make it ineligible; the recording cache is exercised
    // directly through an ineligible plan built from an eligible one).
    let mut p = srt_test_plan();
    p.clean_flat_slots = vec![0; p.table_reads.len()];
    p.clean_flat_slots[12] = 1;
    assert!(!ReplayProgram::eligible(&p));
    let m = TestMemory::new();
    let table = 0x2_0000_0000u64;
    fill_table(&m, table, t_sharp(0x3_0000_0000), [0x100, 0x101, 0x102, 0x103], 7);
    let read = |a: u64| m.read(a);
    let resident = |a: u64, out: &mut [u8], _clean: bool| m.resident(a, out);
    let user = srt_user_data(table, v_sharp(0x5000, 16, 64));
    let mut inputs = ResourceRuntimeInputs::new(&user, 0);
    inputs.read_memory = Some(&read);
    inputs.read_clean_memory = Some(&read);
    let mut cache = ResourceMaterializationCache::default();
    let (first, _) = cache.materialize(&p, &inputs, &resident).unwrap();
    let after_first = m.word_reads.get();
    let (second, _) = cache.materialize(&p, &inputs, &resident).unwrap();
    assert_eq!(m.word_reads.get(), after_first, "a hit reads nothing through the word reader");
    assert_eq!(first, second);
    assert_eq!(cache.statistics().hits, 1);
    // A changed constant (read only for the flattened table) refreshes the
    // table alone.
    m.set(table + 0x60, 0x999);
    let (third, _) = cache.materialize(&p, &inputs, &resident).unwrap();
    assert_eq!(cache.statistics().table_refreshes, 1);
    assert_eq!(third.flattened_resource_table[12], 0x999);
    assert_eq!(third.images, first.images);
    // A changed descriptor word is a miss.
    m.set(table + 0x40, 0x555);
    let (fourth, _) = cache.materialize(&p, &inputs, &resident).unwrap();
    assert_eq!(cache.statistics().misses, 2);
    assert_eq!(fourth.samplers[0][0], 0x555);
}

#[test]
fn live_user_data_registers() {
    // s_mov_b32 s0, s5 overwrites s0 before any read: s0 is not live.
    let program = assemble(&[&[sop1(OP_S_MOV_B32, 0, 5)], &image_sample(0, 0, 0, 8), &[S_ENDPGM]]);
    let registers = BindingLayout::collect_user_data_registers(&program, 0, 12);
    assert!(!registers.contains(&0));
    assert!(registers.contains(&5) && registers.contains(&1) && registers.contains(&11));
}

/// A ray query's BVH T# in user data: all four words are live (only the
/// first was, the others read as zero: last node 0, every box a miss).
#[test]
fn ray_query_descriptor_user_data_is_live() {
    let bvh = [0xF198_8F07, 0x0002_1804, 0x0807_0605, 0x0C0B_0A09, 0x0000_0E0D];
    let program = assemble(&[&bvh, &[S_ENDPGM]]);
    let registers = BindingLayout::collect_user_data_registers(&program, 0, 12);
    assert_eq!(registers, [8, 9, 10, 11]);
}

#[test]
fn image_binding_classes() {
    let mut image = ImageResource::new(0, 0);
    image.resource_class = ImageResourceClass::Sampled;
    image.numeric_class = ImageNumericClass::Float;
    image.dimension = ImageDimension::Dim2D;
    let kind = image_binding::for_image(&image).unwrap();
    assert_eq!(image_binding::describe(kind), (ImageResourceClass::Sampled, ImageNumericClass::Float, ImageDimension::Dim2D, false));
    image.depth_compare = true;
    let kind = image_binding::for_image(&image).unwrap();
    assert_eq!(kind.number(), 43);
    image.depth_compare = false;
    image.cube = true;
    assert!(image_binding::is_cube(image_binding::for_image(&image).unwrap()));
    // A cube also read by face binds as a 2D array.
    image.fetched = true;
    assert!(!image_binding::is_cube(image_binding::for_image(&image).unwrap()));
    image.fetched = false;
    image.cube = false;
    image.resource_class = ImageResourceClass::Storage;
    image.dimension = ImageDimension::Dim2DMsaa;
    assert_eq!(image_binding::for_image(&image), None);
    // Every image binding describes itself consistently.
    for i in 0..IMAGE_BINDING_COUNT {
        let kind = DescriptorBindingKind::Image(i);
        let (class, _, _, _) = image_binding::describe(kind);
        assert_eq!(class, image_binding::resource_class(kind), "binding {}", kind.number());
    }
}

mod provenance;

/// A flattened-table word that GPU work is still producing is left for the
/// GPU to fill (no wait); a descriptor word there is still read.
#[test]
fn table_words_gpu_work_still_produces_are_deferred() {
    let p = srt_test_plan();
    let m = TestMemory::new();
    let table = 0x2_0000_0000u64;
    fill_table(&m, table, t_sharp(0x3_0000_0000), [0x24, 0, 0, 0], 7);
    let user = srt_user_data(table, v_sharp(0x5000, 16, 64));
    let read = |a: u64| m.read(a);
    let resident = |a: u64, out: &mut [u8], _clean: bool| m.resident(a, out);
    let probe = |a: u64| m.gpu_owned.borrow().contains(&a);
    let mut inputs = ResourceRuntimeInputs::new(&user, 0);
    inputs.read_memory = Some(&read);
    inputs.read_clean_memory = Some(&read);
    inputs.read_resident_memory = Some(&resident);
    inputs.gpu_written = Some(&probe);
    let baseline = materialize(&p, &inputs).unwrap().0;
    assert!(baseline.deferred_table_reads.is_empty());
    assert_eq!(baseline.flattened_resource_table[12], 7);

    // The table-only dword and a T# word are GPU-owned.
    m.gpu_owned.borrow_mut().insert(table + 0x60);
    m.gpu_owned.borrow_mut().insert(table + 0x20);
    let mut expected = baseline.clone();
    expected.flattened_resource_table[12] = 0;
    expected.deferred_table_reads = vec![(12, table + 0x60)];
    // The evaluator path and the replay path agree.
    let fresh = materialize(&p, &inputs).unwrap().0;
    assert_eq!(fresh, expected);
    let mut cache = ResourceMaterializationCache::default();
    let reads = m.word_reads.get();
    let (replayed, _) = cache.materialize(&p, &inputs, &resident).unwrap();
    assert_eq!(*replayed, expected, "the T# is read through the word reader, the table word deferred");
    assert!(m.word_reads.get() > reads);
    // Without the probe the word is read (synchronising) as before.
    inputs.gpu_written = None;
    let (read_through, _) = cache.materialize(&p, &inputs, &resident).unwrap();
    assert_eq!(*read_through, baseline);
}

/// A 3D access of a 2D-array T# binds it as a 2D array (r is the layer);
/// a 2D one still reads its first layer as a 2D image.
#[test]
fn three_d_access_reads_a_2d_array_t_sharp_as_an_array() {
    use super::materialize::descriptor_dimension;
    let t = |ty: u32| [0u32, 0, 0, ty << 28];
    let array = t(image_format::TYPE_2D_ARRAY);
    assert_eq!(descriptor_dimension(&array, ImageDimension::Dim3D), ImageDimension::Dim2DArray);
    assert_eq!(descriptor_dimension(&array, ImageDimension::Dim2DArray), ImageDimension::Dim2DArray);
    assert_eq!(descriptor_dimension(&array, ImageDimension::Dim2D), ImageDimension::Dim2D);
    assert_eq!(descriptor_dimension(&t(image_format::TYPE_2D), ImageDimension::Dim3D), ImageDimension::Dim2D);
    assert_eq!(descriptor_dimension(&t(image_format::TYPE_3D), ImageDimension::Dim3D), ImageDimension::Dim3D);
}

/// A comparison sample of an 8-bit sRGB or UNORM colour texture returns 0
/// on the hardware (gfx1035): the emulated compare takes the zero
/// reference; a 16-bit UNORM one keeps its clamp.
#[test]
fn srgb_colour_compare_returns_zero() {
    use super::materialize::emulated_compare_state;
    let mut info = ShaderResourceInfo::default();
    info.sampled_pairs.push(SampledImagePair { image: 0, sampler: 0, first_use_pc: 0, point_sampler: None });
    // Clamp-to-edge in x and y, LESS, point filtering.
    let samplers = vec![super::materialize::DescriptorWords::from_slice(&[2 | 2 << 3 | 1 << 12, 0, 0, 0])];
    let t = |format: u32| [0u32, format << 20, 0, 4 | image_format::TYPE_2D << 28];
    let reference = emulated_compare::reference;
    // 8, 8_8, 8_8_8_8 sRGB and UNORM.
    for format in [128u32, 129, 130, 1, 14, 56] {
        let state = emulated_compare_state(&info, &samplers, 0, &t(format));
        assert!(state & emulated_compare::ENABLED != 0, "format {format}");
        assert_eq!(reference(state), emulated_compare::REFERENCE_ZERO, "format {format}");
    }
    // 16_16 UNORM.
    assert_eq!(reference(emulated_compare_state(&info, &samplers, 0, &t(23))), emulated_compare::REFERENCE_UNORM);
}

/// 10_10_10_2 SNORM (51) and SINT (55) textures sample (as float and
/// signed integer), not as null textures.
#[test]
fn ten_ten_ten_two_signed_formats_sample() {
    assert_eq!(image_format::sampled_numeric_class(51), ImageNumericClass::Float);
    assert_eq!(image_format::sampled_numeric_class(55), ImageNumericClass::Sint);
    assert_eq!(image_format::sampled_numeric_class(50), ImageNumericClass::Float);
}

/// A sample-adjust S# whose dword 3 is built only from values inside the
/// reserved bits 12..29 (a wave mask shifted to bit 12) drops that dword
/// whole instead of failing the plan (reference 6e5d708f8).
#[test]
fn sample_adjust_dword3_of_reserved_bits_is_dropped() {
    const OP_S_AND_B32: u32 = 0x0E;
    const EXEC_LO: u32 = 126;
    let [w0, w1] = image_sample(0, 0, 0, 8);
    // image_sample_a: MIMG opcode 0xA0 (bit 7 of the opcode in word 0 bit 0).
    let sample_a = [w0 | 1, w1];
    // A wave mask (EXEC, or VCC no host value stands for) masked to 18 bits.
    for mask in [EXEC_LO, 106] {
        let p = plan(
            assemble(&[&[sop2(OP_S_AND_B32, 11, mask, LITERAL), 0x3_FFFF], &[sop2(OP_S_LSHL_B32, 11, 11, 128 + 12)], &sample_a, &[S_ENDPGM]]),
            ShaderStage::Pixel,
            12,
        );
        assert_eq!(p.info.samplers.len(), 1);
        let source = &p.descriptor_sources[p.info.samplers[0].source as usize];
        assert_eq!(p.graph.constant_u32(source.dwords[3]), Some(0), "mask register {mask}");
    }
}

/// The per-pixel sample masks take two dwords at the end of the shader
/// data; a layout that no longer fits the push data moves it to the
/// shader-data buffer.
#[test]
fn quad_masks_extend_the_shader_data() {
    let info = ShaderResourceInfo::default();
    let mut l = BindingLayout::allocate(&info, (0..4).collect(), false, false, false, 0, false, false).unwrap();
    assert!(!l.uses_quad_masks());
    assert_eq!(l.shader_data_dword_count(), 4);
    l.add_quad_masks();
    assert_eq!((l.quad_masks_dword, l.shader_data_dword_count(), l.push_data_start_dword), (4, 6, 0));
    let again = l.clone();
    l.add_quad_masks();
    assert_eq!(l, again, "added once");
    let mut full = BindingLayout::allocate(&info, (0..16).collect(), false, false, false, 16, false, false).unwrap();
    assert_eq!(full.push_data_start_dword, 16);
    assert!(full.find(DescriptorBindingKind::ShaderData).is_none());
    full.add_quad_masks();
    assert_eq!(full.push_data_start_dword, push_data::NO_START);
    assert!(full.find(DescriptorBindingKind::ShaderData).is_some());
    assert_eq!(full.shader_data_dword_count(), 18);
}

/// A V# whose NUM_RECORDS the program loads from memory (s_load into
/// word 2 of the V# in s[4:7]): while GPU work still writes that word and
/// the host has room for the buffer, the word is left to the GPU (its
/// table slot deferred) and the descriptor gets the records the room
/// holds; without room, or while the word is not GPU-written, it is read.
#[test]
fn replay_defers_a_gpu_written_record_count() {
    let p = plan(assemble(&[&smem(SMEM_LOAD_DWORD, 6, 2, 0x10), &buffer_load_dword(1, 2, 4), &[S_ENDPGM]]), ShaderStage::Compute, 8);
    assert!(ReplayProgram::eligible(&p));
    let program = ReplayProgram::compile(&p).unwrap();
    let m = TestMemory::new();
    let table = 0x2_0000_0000u64;
    let count = table + 0x10;
    m.set(count, 64);
    let user = srt_user_data(table, v_sharp(0x5000, 16, 0));
    let resident = |a: u64, out: &mut [u8], _clean: bool| m.resident(a, out);
    let read = |a: u64| m.read(a);
    let probe = |a: u64| m.gpu_owned.borrow().contains(&a);
    let room = |base: u64, stride: u32, word: u64| (base == 0x5000 && stride == 16 && word == count).then_some(0x1000u64);
    let no_room = |_: u64, _: u32, _: u64| None;
    let mut scratch = ReplayScratch::default();
    let run = |reader: &ReplayReader<'_>, room: Option<RecordsRoom<'_>>, scratch: &mut ReplayScratch| {
        let mut out = ReplayResult::default();
        program.run_into_with(&user, 0, reader, Some(&probe), room, scratch, &mut out).map(|_| out)
    };
    let words = |r: &ReplayResult| r.words[..program.descriptor_words].to_vec();
    // Not GPU-written: read.
    let r = run(&ReplayReader::Resident(&resident, true), Some(&room), &mut scratch).unwrap();
    assert_eq!(words(&r)[2], 64);
    assert!(r.deferred.is_empty());
    // GPU-written, with room: deferred, 0x1000 / 16 records, no CPU read.
    m.gpu_owned.borrow_mut().insert(count);
    let before = m.word_reads.get();
    let r = run(&ReplayReader::Resident(&resident, true), Some(&room), &mut scratch).unwrap();
    assert_eq!(words(&r)[2], 0x100);
    assert_eq!(words(&r)[0], 0x5000);
    assert_eq!(r.deferred.len(), 1, "{:?}", r.deferred);
    assert_eq!(r.deferred[0].1, count);
    assert_eq!(m.word_reads.get(), before);
    // GPU-written, no room: the resident run refuses, the word reader reads it.
    assert_eq!(run(&ReplayReader::Resident(&resident, true), Some(&no_room), &mut scratch), Err(ReplayAbort::Unreadable));
    let r = run(&ReplayReader::Words(&read, Some(&read)), Some(&no_room), &mut scratch).unwrap();
    assert_eq!(words(&r)[2], 64);
    // Without the host's room function nothing changes: read.
    let r = run(&ReplayReader::Words(&read, Some(&read)), None, &mut scratch).unwrap();
    assert_eq!(words(&r)[2], 64);
    assert!(r.deferred.is_empty());
}
