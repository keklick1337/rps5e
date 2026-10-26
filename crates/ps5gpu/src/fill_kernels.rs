// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! SDK compute kernels that only fill memory, recognised so a dispatch of
//! one runs as a buffer fill (no shader, no resource binding).
//!
//! The pattern-fill kernel (64 threads, one work-group id, user data
//! s[0:3] source V#, s[4:7] destination V#, s[8:11] control V#) stores, for
//! every thread `i` below the control's count, `source[i & mask]` at
//! `destination[i]` (count and mask are the control's first two dwords).
//! A zero mask is a fill with one repeated dword.

/// The kernel's code.
pub const PATTERN_FILL_KERNEL: [u32; 18] = [
    0xbfa0_0001, 0xd746_0000, 0x0401_0c0c, 0xf420_1a84, 0xfa00_0000, 0xbf8c_c07f, 0x7da8_006a, 0xbf88_0009, 0xf420_1a84, 0xfa00_0004, 0xbf8c_c07f, 0x3602_006a, 0xe000_2000,
    0x8000_0101, 0xbf8c_3f70, 0xe010_2000, 0x8001_0100, 0xbf81_0000,
];

/// Compute stage shape of a dispatch (the parts the match depends on).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComputeShape {
    pub threads: [u32; 3],
    /// RSRC2.TGID_X_EN.
    pub group_id_x: bool,
    /// RSRC2.TIDIG_COMP_CNT + 1.
    pub thread_id_components: u32,
}

impl ComputeShape {
    pub fn from_rsrc2(threads: [u32; 3], rsrc2: u32) -> Self {
        ComputeShape { threads, group_id_x: rsrc2 >> 7 & 1 != 0, thread_id_components: (rsrc2 >> 11 & 3) + 1 }
    }
}

/// Whether `code` (at least the kernel's length), `user_data` (exactly the
/// 12 user SGPRs) and `shape` are the pattern-fill kernel with the buffer
/// descriptors it is used with (stride 4 source and destination, stride 16
/// control, unswizzled, formats of the SDK).
pub fn matches_pattern_fill(code: &[u32], user_data: &[u32], shape: &ComputeShape) -> bool {
    if user_data.len() != 12 || shape.threads != [64, 1, 1] || !shape.group_id_x || shape.thread_id_components != 1 {
        return false;
    }
    if code.len() < PATTERN_FILL_KERNEL.len() || code[..PATTERN_FILL_KERNEL.len()] != PATTERN_FILL_KERNEL {
        return false;
    }
    for first in [0usize, 4, 8] {
        let control = first == 8;
        if user_data[first + 1] & 0xFFFF_0000 != if control { 0x0010_0000 } else { 0x0004_0000 } {
            return false;
        }
        if user_data[first + 3] != if control { 0x0004_dfac } else { 0x0001_4004 } {
            return false;
        }
    }
    user_data[2] >= 1 && user_data[10] >= 1
}

/// The memory a pattern-fill dispatch touches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatternFillRange {
    pub source: u64,
    pub destination: u64,
    pub control: u64,
    /// Threads launched.
    pub invocations: u64,
    /// Destination records.
    pub records: u32,
}

impl PatternFillRange {
    /// Bytes stored for the control's `count` and `mask` when the fill is
    /// one repeated dword over whole 16-byte units that overlap neither the
    /// source dword nor the control; None otherwise (the dispatch runs).
    pub fn uniform_bytes(&self, count: u32, mask: u32) -> Option<u64> {
        if mask != 0 {
            return None;
        }
        let bytes = self.invocations.min(count as u64).min(self.records as u64) * 4;
        if self.destination % 16 != 0 || bytes % 16 != 0 {
            return None;
        }
        let d = self.destination;
        if bytes != 0 && ((self.source < d + bytes && d < self.source + 4) || (self.control < d + bytes && d < self.control + 8)) {
            return None;
        }
        Some(bytes)
    }
}

/// Decode a pattern-fill dispatch: `groups` are the packet's dimensions
/// (threads with `thread_dimensions`). None for multidimensional
/// dispatches, thread indices past 32 bits or unaligned source / control.
pub fn decode_pattern_fill(user_data: &[u32], groups: [u32; 3], thread_dimensions: bool) -> Option<PatternFillRange> {
    if user_data.len() != 12 || groups[1] != 1 || groups[2] != 1 {
        return None;
    }
    let invocations = if thread_dimensions { groups[0] as u64 } else { groups[0] as u64 * 64 };
    if invocations > 1 << 32 {
        return None;
    }
    let base = |w: usize| user_data[w] as u64 | ((user_data[w + 1] & 0xFFFF) as u64) << 32;
    if base(0) % 4 != 0 || base(8) % 4 != 0 {
        return None;
    }
    Some(PatternFillRange { source: base(0), destination: base(4), control: base(8), invocations, records: user_data[6] })
}

/// The SDK's indexed pattern-fill kernel: like the pattern-fill kernel the
/// pattern is four dwords repeated over the output, but the count comes
/// from user data (not a control buffer) and the pattern is selected per
/// element (`pattern[index & 3]`) through `v_movrels_b32`. The shape the
/// detector extracts: the work-group id register, the count's user-data
/// word, the pattern's four user-data words and the destination V#'s four
/// user-data words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexedPatternFill {
    /// SGPR holding the work-group id X.
    pub group: u32,
    /// User-data SGPR of the element count.
    pub count: u32,
    /// User-data SGPRs of the four pattern words.
    pub pattern: [u32; 4],
    /// User-data SGPRs of the destination V#.
    pub resource: u32,
}

/// Inline (encoded) constants of the indexed pattern-fill kernel.
const ENCODED_SHIFT_64: u32 = 134;
const ENCODED_MASK_3: u32 = 131;
const ENCODED_ZERO: u32 = 128;
/// The scalar registers the kernel's lane-selection loop runs through.
const EXEC_REGISTER: u32 = 126;
const MARKER_REGISTER: u32 = 124;

/// The pc a SOPP branch of `instruction` jumps to.
fn branch_target(instruction: &ps5shader::Instruction) -> i64 {
    instruction.pc as i64 + 4 + (instruction.words[0] & 0xFFFF) as u16 as i16 as i64 * 4
}

/// Recognise the indexed pattern-fill kernel in a decoded program: twenty
/// instructions that move the pattern into consecutive VGPRs, bound the
/// flat thread index by a user-data count, then select
/// `pattern[index & 3]` per element with a lane loop and store it through
/// an indexed V#. None for any other shape (or a shader that also writes
/// the registers the fill reads).
pub fn detect_indexed_pattern_fill(program: &ps5shader::Program) -> Option<IndexedPatternFill> {
    use ps5shader::{Operand, OperandKind};
    use ps5shader::opcode::Opcode::*;

    const SEQUENCE: [ps5shader::Opcode; 20] = [
        VLshlAddU32, VMovB32, VMovB32, VMovB32, VMovB32, VCmpxGtU32, SCbranchExecz, VAndB32, SMovB64, SMovB64, VReadfirstlaneB32, VCmpxEqU32,
        SMovB32, VMovrelsB32, SAndn2B64, SMovB64, SCbranchScc1, SMovB64, BufferStoreFormatX, SEndpgm,
    ];
    let scalar = |o: &Operand| matches!(o.kind, OperandKind::ScalarRegister).then_some(o.value);
    let vector = |o: &Operand| matches!(o.kind, OperandKind::VectorRegister).then_some(o.value);
    let encoded = |o: &Operand, v: u32| matches!(o.kind, OperandKind::EncodedConstant) && o.value == v;

    let i: Vec<&ps5shader::Instruction> = program
        .instructions
        .iter()
        .filter(|x| !matches!(x.opcode, STtraceData | SInstPrefetch))
        .collect();
    if i.len() != SEQUENCE.len() || i.iter().zip(SEQUENCE).any(|(x, op)| x.opcode != op) {
        return None;
    }
    // Scalar registers any instruction writes.
    let written = |from: usize| i[from..].iter().flat_map(|x| x.destinations.iter().filter_map(scalar)).collect::<std::collections::HashSet<u32>>();

    // v_lshl_add_u32 v_index, s_group, 6, v0: the flat thread index.
    let index = vector(i[0].destinations.first()?)?;
    let group = scalar(&i[0].sources[0])?;
    if i[0].sources.len() != 3 || !encoded(&i[0].sources[1], ENCODED_SHIFT_64) || i[0].sources[2] != Operand::vector(0) {
        return None;
    }

    // Four v_mov_b32 v[pattern_base + k], s_pattern[k]: the pattern in
    // consecutive VGPRs, none of its registers written later.
    let mut pattern = [0u32; 4];
    let mut pattern_base = None;
    let all_written = written(0);
    for (k, m) in i[1..5].iter().enumerate() {
        let moved = vector(m.destinations.first()?)?;
        let p = scalar(m.sources.first()?)?;
        if m.sources.len() != 1 || moved == index || pattern_base.unwrap_or(moved) + k as u32 != moved || all_written.contains(&p) {
            return None;
        }
        pattern_base = Some(pattern_base.unwrap_or(moved));
        pattern[k] = p;
    }
    let pattern_base = pattern_base?;

    // v_cmpx_gt_u32 s_count, v_index: the element bound.
    let count = scalar(&i[5].sources[0])?;
    if i[5].sources.len() != 2 || vector(&i[5].sources[1])? != index || all_written.contains(&count) {
        return None;
    }

    // The lane-selection loop: v_and_b32 (index & 3) picks the pattern
    // word, readfirstlane makes it uniform, m0 selects it through
    // v_movrels_b32, and the loop repeats until every lane stored.
    let lane = vector(i[7].destinations.first()?)?;
    if i[7].sources.len() != 2 || !encoded(&i[7].sources[0], ENCODED_MASK_3) || vector(&i[7].sources[1])? != index {
        return None;
    }
    let saved = scalar(i[8].destinations.first()?)?;
    if i[8].sources.len() != 1 || scalar(&i[8].sources[0])? != EXEC_REGISTER {
        return None;
    }
    let remaining = scalar(i[9].destinations.first()?)?;
    if i[9].sources.len() != 1 || scalar(&i[9].sources[0])? != EXEC_REGISTER {
        return None;
    }
    let uniform = scalar(i[10].destinations.first()?)?;
    if i[10].sources.len() != 1 || vector(&i[10].sources[0])? != lane {
        return None;
    }
    if i[11].sources.len() != 2 || scalar(&i[11].sources[0])? != uniform || vector(&i[11].sources[1])? != lane {
        return None;
    }
    if i[12].destinations.first().and_then(scalar) != Some(MARKER_REGISTER) || scalar(i[12].sources.first()?)? != uniform {
        return None;
    }
    let data = vector(i[13].destinations.first()?)?;
    if i[13].sources.len() != 2 || vector(&i[13].sources[0])? != pattern_base || scalar(&i[13].sources[1])? != MARKER_REGISTER {
        return None;
    }
    if i[14].destinations.first().and_then(scalar) != Some(remaining) || i[14].sources.len() != 2 || scalar(&i[14].sources[0])? != remaining || scalar(&i[14].sources[1])? != EXEC_REGISTER {
        return None;
    }
    if i[15].destinations.first().and_then(scalar) != Some(EXEC_REGISTER) || i[15].sources.len() != 1 || scalar(&i[15].sources[0])? != remaining {
        return None;
    }
    if i[17].destinations.first().and_then(scalar) != Some(EXEC_REGISTER) || i[17].sources.len() != 1 || scalar(&i[17].sources[0])? != saved {
        return None;
    }
    if data == index || lane == index {
        return None;
    }

    // buffer_store_format_x v_data, v_index, s[resource..4], 0: one dword
    // per element through the V#, indexed, untyped, no offset.
    let ps5shader::Control::BufferMemory(c) = i[18].control.as_ref()? else { return None };
    if c.dword_count != 1 || c.offset_bytes != 0 || !c.index_enabled || c.offset_enabled || c.typed {
        return None;
    }
    if i[18].sources.len() != 3 || !encoded(&i[18].sources[2], ENCODED_ZERO) || c.vector_address != index || c.vector_data != data {
        return None;
    }

    // Branches: skip to the end when nothing is in bounds; loop back to
    // the readfirstlane while lanes remain; the saved exec must survive
    // the loop.
    let end_pc = i[19].pc as i64;
    if branch_target(i[6]) != end_pc || branch_target(i[16]) != i[10].pc as i64 {
        return None;
    }
    if (10..17).any(|k| i[k].destinations.iter().any(|d| scalar(d) == Some(saved))) {
        return None;
    }
    // The V# itself must stay intact for the whole program.
    if (0..4u32).any(|w| all_written.contains(&(c.scalar_resource + w))) {
        return None;
    }
    Some(IndexedPatternFill { group, count, pattern, resource: c.scalar_resource })
}

#[cfg(test)]
mod tests {
    use super::*;

    const PATTERN: u64 = 0x10_0000;
    const OUTPUT: u64 = 0x20_0000;
    const CONTROL: u64 = 0x30_0000;

    fn user_data(records: u32) -> [u32; 12] {
        let mut d = [0u32; 12];
        let mut descriptor = |first: usize, address: u64, stride: u32, count: u32, format: u32| {
            d[first] = address as u32;
            d[first + 1] = (address >> 32) as u32 | stride << 16;
            d[first + 2] = count;
            d[first + 3] = format;
        };
        descriptor(0, PATTERN, 4, 4, 0x14004);
        descriptor(4, OUTPUT, 4, records, 0x14004);
        descriptor(8, CONTROL, 16, 1, 0x4dfac);
        d
    }

    fn shape() -> ComputeShape {
        ComputeShape::from_rsrc2([64, 1, 1], 12 << 1 | 1 << 7)
    }

    #[test]
    fn the_kernel_and_its_descriptors_match() {
        let data = user_data(256);
        assert!(matches_pattern_fill(&PATTERN_FILL_KERNEL, &data, &shape()));
        for word in 0..PATTERN_FILL_KERNEL.len() {
            let mut altered = PATTERN_FILL_KERNEL;
            altered[word] ^= 1;
            assert!(!matches_pattern_fill(&altered, &data, &shape()), "a different shader (word {word}) matched the fill kernel");
        }
        assert!(!matches_pattern_fill(&PATTERN_FILL_KERNEL[..17], &data, &shape()), "truncated fill shader matched");
        for word in [1usize, 3, 5, 7, 9, 11] {
            let mut altered = data;
            altered[word] ^= if word % 4 == 1 { 0x8000_0000 } else { 0x1000 };
            assert!(!matches_pattern_fill(&PATTERN_FILL_KERNEL, &altered, &shape()), "unsupported fill descriptor (word {word}) matched");
        }
        let no_group = ComputeShape { group_id_x: false, ..shape() };
        assert!(!matches_pattern_fill(&PATTERN_FILL_KERNEL, &data, &no_group), "fill with no group id matched");
        let narrow = ComputeShape { threads: [32, 1, 1], ..shape() };
        assert!(!matches_pattern_fill(&PATTERN_FILL_KERNEL, &data, &narrow), "different fill group size matched");
        let two_ids = ComputeShape { thread_id_components: 2, ..shape() };
        assert!(!matches_pattern_fill(&PATTERN_FILL_KERNEL, &data, &two_ids));
        assert!(!matches_pattern_fill(&PATTERN_FILL_KERNEL, &data[..11], &shape()));
    }

    #[test]
    fn the_range_is_bounded_by_threads_count_and_records() {
        let data = user_data(256);
        let range = decode_pattern_fill(&data, [2, 1, 1], false).unwrap();
        assert_eq!(range.uniform_bytes(256, 0), Some(512), "fill exceeded launched threads");
        assert_eq!(range.uniform_bytes(80, 0), Some(320), "fill exceeded its control count");
        assert_eq!(range.uniform_bytes(80, 1), None, "nonuniform pattern matched a uniform fill");
        assert_eq!(range.uniform_bytes(79, 0), None, "partial 16-byte fill matched");
        assert_eq!(range.uniform_bytes(0, 0), Some(0), "empty control count was not empty");
        let mut range = decode_pattern_fill(&data, [68, 1, 1], true).unwrap();
        assert_eq!(range.uniform_bytes(256, 0), Some(272), "partial workgroup was rounded up");
        range.records = 64;
        assert_eq!(range.uniform_bytes(256, 0), Some(256), "fill exceeded output descriptor");
        range.source = range.destination + 16;
        assert_eq!(range.uniform_bytes(256, 0), None, "aliased pattern was accepted");
        range.source = 0x1000;
        range.control = range.destination - 4;
        assert_eq!(range.uniform_bytes(256, 0), None, "aliased control was accepted");
        assert_eq!(decode_pattern_fill(&data, [0, 1, 1], true).unwrap().invocations, 0, "empty dispatch gained invocations");
        assert_eq!(decode_pattern_fill(&data, [0x400_0001, 1, 1], false), None, "wrapped thread indices were accepted");
        assert_eq!(decode_pattern_fill(&data, [1, 2, 1], false), None, "multidimensional fill was accepted");
    }

    /// The kernel run in the shader interpreter stores what the decoded
    /// range says (a uniform fill bounded by the count and the threads).
    #[test]
    fn the_interpreted_kernel_agrees_with_the_range() {
        use ps5shader::interp::{Machine, Memory};
        use ps5shader::program::decode_blob;
        use ps5shader::ssa::{Stage, SystemValue};
        use ps5shader::translate::{compile, InputValue, TranslateOptions};
        let data = user_data(256);
        let code_address = 0x10_0000_0000u64;
        let bytes: Vec<u8> = PATTERN_FILL_KERNEL.iter().flat_map(|w| w.to_le_bytes()).collect();
        let program = decode_blob(&bytes, code_address).unwrap();
        let mut o = TranslateOptions::with_user_data(64, Stage::Compute, 12);
        o.sgprs.push((12, InputValue::System(SystemValue::WorkgroupId(0))));
        o.vgprs = vec![(0, InputValue::System(SystemValue::LocalInvocationId(0)))];
        let f = compile(&program, &o, None).unwrap().function;
        for (count, groups) in [(80u32, 2u32), (256, 2), (512, 3), (0, 1)] {
            let mut memory = Memory::default();
            memory.write_words(PATTERN, &[0x1234_5678, 0xAAAA_AAAA, 0xBBBB_BBBB, 0xCCCC_CCCC]);
            memory.write_words(CONTROL, &[count, 0, 0, 0]);
            memory.write_words(OUTPUT, &[0xDEAD_BEEF; 300]);
            for g in 0..groups {
                let mut m = Machine::new(64, 64);
                m.user_data = data.to_vec();
                m.memory = std::mem::take(&mut memory);
                m.system.insert(SystemValue::LocalInvocationId(0), (0..64).collect());
                m.system.insert(SystemValue::WorkgroupId(0), vec![g; 64]);
                m.run(&f).unwrap();
                memory = std::mem::take(&mut m.memory);
            }
            let out = memory.read_words(OUTPUT, 300);
            let expected = decode_pattern_fill(&data, [groups, 1, 1], false).unwrap().uniform_bytes(count, 0).unwrap() / 4;
            for (i, &w) in out.iter().enumerate() {
                let want = if (i as u64) < expected { 0x1234_5678 } else { 0xDEAD_BEEF };
                assert_eq!(w, want, "count {count} groups {groups} dword {i}");
            }
        }
    }
}
