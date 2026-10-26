// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Scalar-provenance cases (ported from another emulator's test suite and
//! re-expressed as gfx10 programs where the decoder can express them).

use super::*;

/// A plan of an empty program, for building and evaluating values by hand.
fn scratch_plan() -> ShaderResourcePlan {
    plan(assemble(&[&[S_ENDPGM]]), ShaderStage::Compute, 4)
}

/// s_mov_b32 s5, 0 ; s_mov_b32 s6, 16 ; s_mov_b32 s7, 0 (finish a V# in s[4:7]).
fn finish_v_sharp() -> Vec<u32> {
    vec![sop1(OP_S_MOV_B32, 5, 128), sop1(OP_S_MOV_B32, 6, LITERAL), 16, sop1(OP_S_MOV_B32, 7, 128)]
}

#[test]
fn equivalent_reads_share_one_slot() {
    // Two identical s_load_dword at different pcs: one flattened slot, both
    // loads planning-only, one host read.
    let p = plan(
        assemble(&[
            &smem(SMEM_LOAD_DWORD, 4, 0, 0x20),
            &smem(SMEM_LOAD_DWORD, 8, 0, 0x20),
            &[sop1(OP_S_MOV_B32, 5, 8)],
            &[sop1(OP_S_MOV_B32, 6, LITERAL), 16],
            &[sop1(OP_S_MOV_B32, 7, 128)],
            &buffer_load_dword(1, 2, 4),
            &[S_ENDPGM],
        ]),
        ShaderStage::Compute,
        2,
    );
    assert_eq!(p.table_reads.len(), 1);
    assert!(p.memory().entries().iter().filter(|m| m.kind == MemoryResourceKind::ScalarAddress).all(|m| m.planning_only));
    let reads = Cell::new(0);
    let reader = |a: u64| {
        reads.set(reads.get() + 1);
        (a == 0x1020).then_some(0xFEED_BEEF)
    };
    let user_data = [0x1000, 0];
    let mut inputs = ResourceRuntimeInputs::new(&user_data, 0);
    inputs.read_memory = Some(&reader);
    assert_eq!(flatten_resource_table(&p, &inputs).unwrap(), vec![0xFEED_BEEF]);
    assert_eq!(reads.get(), 1);
    let (snapshot, _) = materialize(&p, &inputs).unwrap();
    assert_eq!(snapshot.buffers[0][..2], [0xFEED_BEEF, 0xFEED_BEEF]);
}

#[test]
fn raw_scalar_reads_align_base_and_offsets_independently() {
    // Base 0x1003 (low two bits ignored) and immediate 2 (rounded down).
    let p = plan(assemble(&[&smem(SMEM_LOAD_DWORD, 4, 0, 2), &finish_v_sharp(), &buffer_load_dword(1, 2, 4), &[S_ENDPGM]]), ShaderStage::Compute, 2);
    let reader = |a: u64| (a == 0x1000).then_some(0x1234_5678);
    let user_data = [0x1003, 0];
    let mut inputs = ResourceRuntimeInputs::new(&user_data, 0);
    inputs.read_memory = Some(&reader);
    assert_eq!(flatten_resource_table(&p, &inputs).unwrap(), vec![0x1234_5678]);
}

#[test]
fn dynamic_offset_reads_stay_typed() {
    // s_load_dword s4, s[0:1], s2: a register offset is not flattened; the
    // offset is user data, so the host still evaluates the V# per draw.
    let load = [0xF400_0000 | (4 << 6), 2 << 25];
    let p = plan(assemble(&[&load, &finish_v_sharp(), &buffer_load_dword(1, 2, 4), &[S_ENDPGM]]), ShaderStage::Compute, 3);
    assert!(p.table_reads.is_empty());
    assert_eq!(p.dynamic_reads.len(), 1);
    let load = p.memory().entries().iter().find(|m| m.kind == MemoryResourceKind::ScalarAddress).unwrap();
    assert!(!load.planning_only);
    let buffer = p.memory().entries().iter().find(|m| m.kind == MemoryResourceKind::Buffer).unwrap();
    assert_eq!(buffer.buffer_descriptor, Some(GuestBufferDescriptor::static_resource(0)));
    assert_eq!(p.category_counts().get(&SourceCategory::SrtWalk), Some(&1));
    let reader = |a: u64| (a == 0x1008).then_some(0x5000);
    let user_data = [0x1000, 0, 8];
    let mut inputs = ResourceRuntimeInputs::new(&user_data, 0);
    inputs.read_memory = Some(&reader);
    let (snapshot, _) = materialize(&p, &inputs).unwrap();
    assert_eq!(snapshot.buffers[0][..], (vec![0x5000, 0, 16, 0])[..]);
}

#[test]
fn nested_srt_walk_reads_in_dependency_order() {
    // s_load_dwordx2 s[4:5], s[0:1], 0 ; s_load_dwordx4 s[8:11], s[4:5], 0
    let p = plan(
        assemble(&[&smem(SMEM_LOAD_DWORDX2, 4, 0, 0), &smem(SMEM_LOAD_DWORDX4, 8, 4, 0), &buffer_load_dword(1, 2, 8), &[S_ENDPGM]]),
        ShaderStage::Compute,
        2,
    );
    assert_eq!(p.table_reads.len(), 6);
    let memory = HashMap::from([(0x1000u64, 0x2000u32), (0x1004, 0), (0x2000, 0xABCD_EF01), (0x2004, 0), (0x2008, 16), (0x200C, 0)]);
    let reader = memory_reader(&memory);
    let user_data = [0x1000, 0];
    let mut inputs = ResourceRuntimeInputs::new(&user_data, 0);
    inputs.read_memory = Some(&reader);
    let (snapshot, _) = materialize(&p, &inputs).unwrap();
    assert_eq!(snapshot.buffers[0][..], (vec![0xABCD_EF01, 0, 16, 0])[..]);
    assert_eq!(snapshot.flattened_resource_table[..2], [0x2000, 0]);
}

#[test]
fn shader_base_user_data_carry_and_bit_fields() {
    let mut p = scratch_plan();
    let g = &mut p.graph;
    let base = g.shader_base();
    let (zero, one) = (g.constant(0), g.constant(1));
    let low = g.operation(ScalarOp::Extract64, ValueType::U32, &[base, zero]);
    let high = g.operation(ScalarOp::Extract64, ValueType::U32, &[base, one]);
    let user = g.user_data(2);
    let four = g.constant(4);
    let sum = g.operation(ScalarOp::IAdd32, ValueType::U32, &[user, four]);
    let u = g.undefined(ValueType::U32);
    let poisoned = g.operation(ScalarOp::IAdd32, ValueType::U32, &[u, four]);
    let user_data = [0, 0, 0x20, 0];
    let e = RuntimeValueEvaluator::new(&p, ResourceRuntimeInputs::new(&user_data, 0x1234_5678_ABCD_EF00));
    assert_eq!(e.evaluate(low), Some(0xABCD_EF00));
    assert_eq!(e.evaluate(high), Some(0x1234_5678));
    assert_eq!(e.evaluate(sum), Some(0x24));
    // An undefined input never evaluates.
    assert_eq!(e.evaluate(poisoned), None);
    assert!(!p.validate_runtime_value(poisoned));

    let mut p = scratch_plan();
    let g = &mut p.graph;
    let user = g.user_data(0);
    let two = g.constant(2);
    let carry = g.operation(ScalarOp::AddCarry32, ValueType::U64, &[user, two]);
    let (zero, one) = (g.constant(0), g.constant(1));
    let carry_low = g.operation(ScalarOp::Extract64, ValueType::U32, &[carry, zero]);
    let carry_high = g.operation(ScalarOp::Extract64, ValueType::U32, &[carry, one]);
    let (f0, c4) = (g.constant(0xF0), g.constant(4));
    let sign = g.operation(ScalarOp::BitFieldSExtract, ValueType::U32, &[f0, c4, c4]);
    let wide = g.operation(ScalarOp::Construct64, ValueType::U64, &[user, zero]);
    let c32 = g.constant(32);
    let shifted = g.operation(ScalarOp::ShiftLeft64, ValueType::U64, &[wide, c32]);
    let mask = g.constant64(0x0000_FFFF_0000_0000);
    let masked = g.operation(ScalarOp::And64, ValueType::U64, &[shifted, mask]);
    let add = g.constant64(0x0000_0001_0000_ABCD);
    let combined = g.operation(ScalarOp::IAdd64, ValueType::U64, &[masked, add]);
    let combined_low = g.operation(ScalarOp::Extract64, ValueType::U32, &[combined, zero]);
    let combined_high = g.operation(ScalarOp::Extract64, ValueType::U32, &[combined, one]);
    let user_data = [0xFFFF_FFFF, 0, 0, 0];
    let e = RuntimeValueEvaluator::new(&p, ResourceRuntimeInputs::new(&user_data, 0));
    assert_eq!((e.evaluate(carry_low), e.evaluate(carry_high)), (Some(1), Some(1)));
    assert_eq!(e.evaluate(sign), Some(0xFFFF_FFFF));
    let user_data = [0x1234, 0, 0, 0];
    let e = RuntimeValueEvaluator::new(&p, ResourceRuntimeInputs::new(&user_data, 0));
    assert_eq!((e.evaluate(combined_low), e.evaluate(combined_high)), (Some(0xABCD), Some(0x1235)));
}

#[test]
fn invariant_and_divergent_phis() {
    let mut p = scratch_plan();
    let g = &mut p.graph;
    let (seven, nine) = (g.constant(7), g.constant(9));
    let invariant = g.phi(0, ValueType::U32);
    g.set_phi_operands(invariant, vec![-1, 0], vec![seven, invariant]);
    let divergent = g.phi(0, ValueType::U32);
    g.set_phi_operands(divergent, vec![-1, 0], vec![seven, nine]);
    let e = RuntimeValueEvaluator::new(&p, ResourceRuntimeInputs::new(&[], 0));
    assert_eq!(e.evaluate(invariant), Some(7));
    assert_eq!(e.evaluate(divergent), None);
    let mut validator = RuntimeValueValidator::new(&p.graph, 0);
    assert!(!validator.validate(divergent));
    assert!(validator.control_dependent);
}

#[test]
fn control_dependent_load_is_not_flattened() {
    // The load's base low dword changes every loop iteration:
    // L: s_add_u32 s20, s20, 1 ; s_add_u32 s0, s0, 16 ; s_cmp_eq_u32 s20, 4 ;
    //    s_cbranch_scc1 +1 ; s_branch L ; s_load_dword s4, s[0:1], 0
    let p = plan(
        assemble(&[
            &[sop2(OP_S_ADD_U32, 20, 20, 129)],
            &[sop2(OP_S_ADD_U32, 0, 0, 144)],
            &[sopc(OP_S_CMP_EQ_U32, 20, 132)],
            &[sopp(OP_S_CBRANCH_SCC1, 1)],
            &[sopp(OP_S_BRANCH, -5)],
            &smem(SMEM_LOAD_DWORD, 4, 0, 0),
            &finish_v_sharp(),
            &buffer_load_dword(1, 2, 4),
            &[S_ENDPGM],
        ]),
        ShaderStage::Compute,
        4,
    );
    assert!(p.table_reads.is_empty());
    let load = p.memory().entries().iter().find(|m| m.kind == MemoryResourceKind::ScalarAddress).unwrap();
    assert!(!load.planning_only);
    let buffer = p.memory().entries().iter().find(|m| m.kind == MemoryResourceKind::Buffer).unwrap();
    assert_eq!(buffer.buffer_descriptor, Some(GuestBufferDescriptor::runtime()));
    assert!(p.resource_uses().iter().any(|u| u.category == SourceCategory::AddressBased));
}

#[test]
fn resource_uses_are_keyed_by_pc() {
    let p = plan(
        assemble(&[&smem(SMEM_LOAD_DWORDX8, 16, 2, 0x20), &smem(SMEM_LOAD_DWORDX4, 24, 2, 0x40), &image_sample(0, 0, 16, 24), &[S_ENDPGM]]),
        ShaderStage::Compute,
        4,
    );
    let uses = p.resource_uses();
    let sample_pc = p.program().instructions[2].pc;
    let image = uses.iter().find(|u| u.pc == sample_pc && u.operand == ResourceOperand::Resource).unwrap();
    assert_eq!(image.binding, ResourceBinding::Image(0));
    assert_eq!(image.category, SourceCategory::SrtWalk);
    let sampler = uses.iter().find(|u| u.pc == sample_pc && u.operand == ResourceOperand::Sampler).unwrap();
    assert_eq!(sampler.binding, ResourceBinding::Sampler(0));
    assert_eq!(uses.iter().filter(|u| matches!(u.binding, ResourceBinding::FlattenedSlot(_))).count(), 12);
}

#[test]
fn bindless_table_image_binds_every_entry() {
    // s_buffer_load_dword s20, s[4:7], 0       (key, read from a V#)
    // s_lshl_b32 s21, s20, 5
    // s_buffer_load_dwordx8 s[24:31], s[8:11], s21 (heap V# in s[8:11])
    // image_sample v0, v0, s[24:31], s[12:15]
    let sbuffer_x8_dynamic = [0xF400_0000 | (0x0B << 18) | (24 << 6) | (8 >> 1), 21 << 25];
    let p = plan(
        assemble(&[
            &smem(SMEM_BUFFER_LOAD_DWORD, 20, 4, 0),
            &[sop2(OP_S_LSHL_B32, 21, 20, 133)],
            &sbuffer_x8_dynamic,
            &image_sample(0, 0, 24, 12),
            &[S_ENDPGM],
        ]),
        ShaderStage::Pixel,
        16,
    );
    assert_eq!(p.program().instructions[2].opcode, Opcode::SBufferLoadDwordx8);
    assert_eq!(p.indirect_images.len(), 1, "{:?}", p.descriptor_sources);
    let source = &p.descriptor_sources[p.info.images[0].source as usize];
    let selector = source.indirect_image.as_ref().unwrap();
    assert_eq!(selector.table, Some(TableImageSelector { entry_offset: 0, has_material: false }));
    assert!(p.requires_specialization_memory);
    assert!(p.resource_uses().iter().any(|u| u.category == SourceCategory::Indirect));

    // A heap of three entries: two valid 2D textures and a null one.
    let heap = 0x8000u64;
    let mut words = HashMap::new();
    for (entry, address) in [(0u64, 0x1_0000_0000u64), (1, 0x2_0000_0000)] {
        for (i, w) in t_sharp(address).iter().enumerate() {
            words.insert(heap + entry * 32 + i as u64 * 4, *w);
        }
    }
    for i in 0..8 {
        words.insert(heap + 64 + i * 4, 0);
    }
    words.insert(0x9000, 1);
    let reader = memory_reader(&words);
    let mut user_data = vec![0u32; 16];
    user_data[4..8].copy_from_slice(&[0x9000, 0, 4, 0]);
    user_data[8..12].copy_from_slice(&[heap as u32, 0, 96, 0]);
    user_data[12] = 0x24;
    let mut inputs = ResourceRuntimeInputs::new(&user_data, 0);
    inputs.read_memory = Some(&reader);
    inputs.read_clean_memory = Some(&reader);
    let (snapshot, specialization) = materialize(&p, &inputs).unwrap();
    // Root = entry 0, one candidate for entry 1; the null entry maps to a
    // null descriptor.
    assert_eq!(snapshot.images[0][..], (t_sharp(0x1_0000_0000))[..]);
    assert!(snapshot.images.iter().any(|d| d[..] == t_sharp(0x2_0000_0000)[..]));
    let root = specialization.images[0];
    assert_eq!(root.indirect_root, 0);
    let mapping = root.indirect_mapping_offset as usize;
    assert_eq!(snapshot.flattened_resource_table[mapping], 3, "three keys");
    assert_eq!(&snapshot.flattened_resource_table[mapping + 1..mapping + 7], &[0, 0, 1, 1, 2, 2][..]);
}

#[test]
fn large_bindless_table_selects_through_the_host_map() {
    // As above with a heap of 64 entries: too many to bind per draw, and
    // the key is not narrowed (not a material record, not draw-uniform).
    let sbuffer_x8_dynamic = [0xF400_0000 | (0x0B << 18) | (24 << 6) | (8 >> 1), 21 << 25];
    let p = plan(
        assemble(&[
            &[0x7E28_0500], // v_readfirstlane_b32 s20, v0 (a per-wave key)
            &[sop2(OP_S_LSHL_B32, 21, 20, 133)],
            &sbuffer_x8_dynamic,
            &image_sample(0, 0, 24, 12),
            &[S_ENDPGM],
        ]),
        ShaderStage::Pixel,
        16,
    );
    assert_eq!(p.indirect_images.len(), 1, "{:?}", p.descriptor_sources);
    // No table record may be read: the reader knows none of them.
    let heap = 0x8000u64;
    let words: HashMap<u64, u32> = HashMap::new();
    let reader = memory_reader(&words);
    let mut user_data = vec![0u32; 16];
    user_data[8..12].copy_from_slice(&[heap as u32, 0, 64 * 32, 0]);
    let mut inputs = ResourceRuntimeInputs::new(&user_data, 0);
    inputs.read_memory = Some(&reader);
    inputs.read_clean_memory = Some(&reader);
    let (snapshot, specialization) = materialize(&p, &inputs).unwrap();
    assert_eq!(snapshot.bindless_tables.len(), 1);
    let table = snapshot.bindless_tables[0];
    assert_eq!((table.image, table.heap, table.entry_offset, table.entries), (0, [heap as u32, 0, 64 * 32, 0], 0, 64));
    // The bound T# is null; the shape is the instruction's.
    assert!(snapshot.images[0].iter().all(|&w| w == 0));
    let root = specialization.images[0];
    assert!(root.bindless);
    assert_eq!((root.numeric_class, root.dimension, root.cube, root.indirect_search_iterations), (ImageNumericClass::Float, ImageDimension::Dim2D, false, 0));
    assert_eq!(root.indirect_mapping_offset, table.mapping_offset);
    assert!(snapshot.flattened_resource_table.len() >= (table.mapping_offset + bindless::TABLE_WORDS) as usize);
    // The layout reads the flattened table and adds the bindless set.
    let specialized = apply_specialization(&p, &specialization).unwrap();
    assert!(specialized.info.images[0].bindless);
    let layout = BindingLayout::allocate(&specialized.info, vec![], false, false, false, 0, false, false).unwrap();
    assert!(layout.bindless_tables);
    assert!(layout.find(DescriptorBindingKind::FlattenedResourceTable).is_some());

    // Sixteen entries are still bound per draw (the small-table path).
    user_data[8..12].copy_from_slice(&[heap as u32, 0, 16 * 32, 0]);
    let mut words = HashMap::new();
    for key in 0..16u64 {
        for (i, w) in t_sharp(0x1_0000_0000 + key * 0x1_0000).iter().enumerate() {
            words.insert(heap + key * 32 + i as u64 * 4, *w);
        }
    }
    let reader = memory_reader(&words);
    let mut inputs = ResourceRuntimeInputs::new(&user_data, 0);
    inputs.read_memory = Some(&reader);
    inputs.read_clean_memory = Some(&reader);
    let (snapshot, specialization) = materialize(&p, &inputs).unwrap();
    assert!(snapshot.bindless_tables.is_empty());
    assert!(!specialization.images[0].bindless);
    assert_eq!(specialization.images[0].indirect_root, 0);
}
