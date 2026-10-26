// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

// Programs of the reference driver's execution tests ported in
// hw_scalar_waves.rs, hw_mem_bases.rs and hw_valu_halves.rs (and run on
// the device by ps5gpu's exec_ref tests): code words as the tests give
// them, and the generators of the programs they assemble. Pulled in with
// `include!`.

#[allow(dead_code)]
pub mod exec_programs {
    pub const ENDPGM: u32 = 0xBF81_0000;

    pub fn sop1(op: u32, dst: u32, src: u32) -> u32 {
        0xBE80_0000 | (dst << 16) | (op << 8) | src
    }

    pub fn sop2(op: u32, dst: u32, src0: u32, src1: u32) -> u32 {
        0x8000_0000 | (op << 23) | (dst << 16) | (src1 << 8) | src0
    }

    pub fn sopc(op: u32, src0: u32, src1: u32) -> u32 {
        0xBF00_0000 | (op << 16) | (src1 << 8) | src0
    }

    pub fn sopk(op: u32, reg: u32, imm: u16) -> u32 {
        0xB000_0000 | (op << 23) | (reg << 16) | imm as u32
    }

    pub fn vop1(op: u32, dst: u32, src: u32) -> u32 {
        0x7E00_0000 | (dst << 17) | (op << 9) | src
    }

    pub fn vop2(op: u32, dst: u32, src0: u32, src1: u32) -> u32 {
        (op << 25) | (dst << 17) | (src1 << 9) | src0
    }

    // ---- ScalarSopkCompare / ScalarSopkWaitcnt: s[4:7] = output V#
    // (strided, 64 records per lane), s8.. = values.

    pub const SOPK_COMPARE_OPS: [u32; 12] = [0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E];
    pub const SOPK_VALUES: [u32; 10] = [0, 1, 0x1234, 0x7FFF, 0x8000, 0xFFFF, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_8000, 0xFFFF_FFFF];
    pub const SOPK_IMMEDIATES: [u16; 6] = [0, 1, 0x1234, 0x7FFF, 0x8000, 0xFFFF];

    const SOPK_STORE: u32 = (0x80 << 24) | (1 << 16) | (10 << 8) | 3;

    /// Per (value, immediate): s30 = OR of (SCC of compare k) << k, stored
    /// at record tid * 64 + result.
    pub fn sopk_compare() -> Vec<u32> {
        let mut code = vec![vop2(0x1A, 3, 0x86, 0)];
        for v in 0..SOPK_VALUES.len() as u32 {
            for (i, &imm) in SOPK_IMMEDIATES.iter().enumerate() {
                code.push(sop1(0x03, 30, 0x80));
                for (k, &op) in SOPK_COMPARE_OPS.iter().enumerate() {
                    code.extend([sopk(op, 8 + v, imm), sop2(0x0A, 31, 0xFF, 0x80), 1 << k, sop2(0x10, 30, 30, 31)]);
                }
                let result = v * SOPK_IMMEDIATES.len() as u32 + i as u32;
                code.extend([vop1(0x01, 10, 30), 0xE070_2000 | (result * 4), SOPK_STORE]);
            }
        }
        code.push(ENDPGM);
        code
    }

    pub const SOPK_WAITS: [u32; 4] = [0x17, 0x18, 0x19, 0x1A];
    pub const SOPK_WAIT_REGISTERS: [u32; 2] = [0x7D, 8];
    pub const SOPK_WAIT_LEVELS: [u16; 3] = [0, 0x3F, 0xFFFF];
    pub const SOPK_WAIT_VALUE: u32 = 0x1234_5678;

    /// Each wait sits between a compare and an s_cselect reading SCC
    /// (first set, then clear); the wait's register is s8 or null. Two
    /// records per case: the flags, then s8.
    pub fn sopk_waitcnt() -> Vec<u32> {
        let mut code = vec![vop2(0x1A, 3, 0x86, 0)];
        let mut result = 0u32;
        for w in SOPK_WAITS {
            for reg in SOPK_WAIT_REGISTERS {
                for level in SOPK_WAIT_LEVELS {
                    code.extend([sop1(0x03, 30, 0x80), sopc(0x00, 8, 8), sopk(w, reg, level), sop2(0x0A, 31, 0x81, 0x80), sop2(0x10, 30, 30, 31)]);
                    code.extend([sopc(0x01, 8, 8), sopk(w, reg, level), sop2(0x0A, 31, 0x82, 0x80), sop2(0x10, 30, 30, 31)]);
                    code.extend([vop1(0x01, 10, 30), 0xE070_2000 | (result * 4), SOPK_STORE, vop1(0x01, 10, 8), 0xE070_2000 | ((result + 1) * 4), SOPK_STORE]);
                    result += 2;
                }
            }
        }
        code.push(ENDPGM);
        code
    }

    // ---- Waves and EXEC.

    /// s[0:3] = raw output V#; 32 lanes.
    pub const WAVE32_EXEC_LANE_BITS: [u32; 17] = [
        0x340c0082, 0x7d88008d, 0x02020080, 0xbe8a3c6a, 0x4a0202c0, 0x887e0a7e, 0x7d820094, 0xbe8b3c6a, 0x4a0202a0, 0xbefe030b, 0x7d82008a, 0x7e0e0281,
        0x02040e80, 0x4a020302, 0xe0701000, 0x80000106, 0xbf810000,
    ];

    /// s[8:11] keys V# (stride 4), s[12:15] table (raw), s[16:19] output
    /// (stride 16); 192 invocations of wave64.
    pub const EXEC_HOST_LANES: [u32; 59] = [
        0xbefe04c1, 0xe0302000, 0x80020c00, 0x7e3c0300, 0x36080081, 0x7e080d04, 0x7e2a02c1, 0xbea4040c, 0xbea6040e, 0xbea80380, 0xbf8c3f70, 0xbe96047e,
        0xbe88047e, 0xbeb0047e, 0x7c0808f9, 0x068698f2, 0xbf880023, 0x80288128, 0xbf0aff28, 0x00000080, 0xbf84001f, 0xbeea1416, 0xd7600020, 0x0000d50c,
        0x7d8418f9, 0x06869a20, 0xd5010004, 0x00690283, 0x7da40881, 0xbf88000b, 0x7e0a0280, 0x87ea7e18, 0x8afe6a7e, 0x8ab06a30, 0xbf880006, 0x7e080283,
        0xb82001b8, 0xf4200792, 0x40000000, 0xbf8cc07f, 0x7e2a021e, 0xbefe0430, 0x7da40883, 0xbf880008, 0x7e0a0281, 0x8a961a16, 0xbf128016, 0xbf800000,
        0x85ea807e, 0x8ab06a30, 0x8afe6a7e, 0xbf82ffdc, 0xbefe0408, 0x7e2c0228, 0x7e2e030c, 0x7e300305, 0xe0782000, 0x8004151e, 0xbf810000,
    ];
    pub const EXEC_HOST_KEYS: u32 = 6;
    pub const EXEC_HOST_KEY_STRIDE: u32 = 440;

    /// s[0:1] = EXEC, s[4:7] = output V# (stride 16); 64 lanes, LDS 128
    /// dwords.
    pub const LDS_WATERFALL: [u32; 58] = [
        0x34240083, 0x7e040280, 0x7e060280, 0xd9340000, 0x00000212, 0xbf8cc07f, 0x16080085, 0x3608088f, 0x34080883, 0xd6ff0002, 0x00010300, 0xd9280000,
        0x00000204, 0xbf8cc07f, 0xd9d80000, 0x06000012, 0xbf8cc07f, 0xd4e50002, 0x00020c80, 0x7e140280, 0x7e160280, 0xbe8a0380, 0xbefe0400, 0xbf138002,
        0xbf840016, 0xbe8b1402, 0xbe821c0b, 0x8f0c830b, 0x7e18020c, 0xd9d80000, 0x0e00000c, 0xbf8cc07f, 0x7dca1c80, 0xbf86fff5, 0x7e1c050e, 0x7e1e050f,
        0xbe90140e, 0x8f921081, 0x3a1c1c12, 0x3a1e1e13, 0x4a141481, 0x4a161610, 0x800a810a, 0xbf08ff0a, 0x00001000, 0xbf850001, 0xbf82fff1, 0xbefe04c1,
        0x7e1a020a, 0xe0702000, 0x80010a00, 0xe0702004, 0x80010b00, 0xe0702008, 0x80010d00, 0xe070200c, 0x80010600, 0xbf810000,
    ];
    pub const LDS_MASKED_OR: [u32; 23] = [
        0x34240083, 0x7e040280, 0x7e060280, 0xd9340000, 0x00000212, 0xbf8cc07f, 0x3608008f, 0x34080883, 0xd6ff0002, 0x00010300, 0xbefe0400, 0xd9280000,
        0x00000204, 0xbf8cc07f, 0xbefe04c1, 0xd9d80000, 0x06000012, 0xbf8cc07f, 0xe0702000, 0x80010600, 0xe0702004, 0x80010700, 0xbf810000,
    ];
    pub const LDS_WATERFALL_EXECS: [u64; 7] = [!0, 0xaaaa_aaaa_aaaa_aaaa, 0xffff_0000_0000_0000, 0xffff, 1 << 63, 1, 1 << 32];
    pub const LDS_MASKED_OR_EXECS: [u64; 5] = [!0, 0xffff_ffff, 0xffff_ffff_0000_0000, 1 << 40, 0x8000_0000_0000_0001];

    /// s[0:1] = guest block address, s2 = work-group id; two work-groups of
    /// 64: the first stores a flag, the second waits for it.
    pub const CROSS_WORKGROUP_FLAG: [u32; 25] = [
        0xbe840380, 0x7e000280, 0xbf068002, 0xbf840004, 0x7e020281, 0xdc718000, 0x00000100, 0xbf810000, 0xdc319000, 0x01000000, 0xbf8c3f70, 0x7d840280,
        0x80048104, 0xbf0aff04, 0x000f4240, 0xbf840005, 0xbf87fff7, 0x7e020282, 0xdc718004, 0x00000100, 0xbf810000, 0x7e020283, 0xdc718004, 0x00000100,
        0xbf810000,
    ];

    /// s[0:3] input (raw), s[4:7] output (raw, 16 words per lane).
    pub const VECTOR_WINDOW_DIRECT: [u32; 65] = [
        0x34020084, 0x34060086, 0xe0301000, 0x80000401, 0xe0301004, 0x80000b01, 0xbe8803ff, 0x5a5a0000, 0xbf8c3f70, 0x4a100881, 0x4a120882, 0x7e140304,
        0xd761000a, 0x00010608, 0xd7600009, 0x0001070a, 0xd760000a, 0x00010b0a, 0x4a180809, 0xbefc0380, 0x7e208708, 0xbefc0381, 0x7e228708, 0xbefc0382,
        0x7e248708, 0xbefc0383, 0x7e268708, 0xbefc0384, 0x7e288708, 0x34120884, 0xbefc0381, 0x7e2a8708, 0xbefc0388, 0x7e2c8708, 0x7e2e0209, 0x7e30020a,
        0xe0701000, 0x80010803, 0xe0701004, 0x80010903, 0xe0701008, 0x80010a03, 0xe070100c, 0x80010b03, 0xe0701010, 0x80010c03, 0xe0701014, 0x80011003,
        0xe0701018, 0x80011103, 0xe070101c, 0x80011203, 0xe0701020, 0x80011303, 0xe0701024, 0x80011403, 0xe0701028, 0x80011503, 0xe070102c, 0x80011603,
        0xe0701030, 0x80011703, 0xe0701034, 0x80011803, 0xbf810000,
    ];

    /// s[4:7] output (raw, 16 words per lane), s8 = bounded index, s9 =
    /// data index.
    pub const VECTOR_WINDOW_NARROWING: [u32; 54] = [
        0x34020086, 0x34060088, 0x4a140680, 0x4a160681, 0x4a180682, 0x4a1a0683, 0x4a1c0684, 0x4a1e0685, 0x4a200686, 0x4a220687, 0x4a240688, 0x4a260689,
        0x4a28068a, 0x4a2a068b, 0x4a2c068c, 0x4a2e068d, 0x4a30068e, 0x4a32068f, 0x4a340690, 0x4a360691, 0x4a380692, 0x4a3a0693, 0x4a3c0694, 0x4a3e0695,
        0x4a400696, 0x4a420697, 0x4a440698, 0x4a460699, 0x4a48069a, 0x4a4a069b, 0x4a4c069c, 0x4a4e069d, 0x4a50069e, 0x4a52069f, 0x877c8308, 0x7e08870a,
        0xbefc0309, 0x7e0a870a, 0x7e148504, 0x877c8308, 0x7e148505, 0xe0701000, 0x80010401, 0xe0701004, 0x80010501, 0xe0701008, 0x80010c01, 0xe070100c,
        0x80010d01, 0xe0701010, 0x80011e01, 0xe0701014, 0x80012901, 0xbf810000,
    ];

    // ---- Memory.

    /// BufferResourceVcc: s[0:3] the ordinary V#, s[4:7] the one moved to
    /// s[104:105] + VCC.
    pub fn resource_vcc(opcode: u32, store: bool) -> [u32; 11] {
        [
            0x34020082,
            0xbee80304,
            0xbee90305,
            0xbeea0306,
            0xbeeb0307,
            if store { 0xe0301000 } else { opcode },
            if store { 0x80000201 } else { 0x801a0201 },
            0xbf8c3f70,
            if store { opcode } else { 0xe0701000 },
            if store { 0x801a0201 } else { 0x80000201 },
            0xbf810000,
        ]
    }

    /// BufferUnalignedBase: s[0:3] input, s[4:7] view, s[8:11] results,
    /// s[12:15] stores; 32 lanes.
    pub const UNALIGNED_BASE: [u32; 56] = [
        0x34020082, 0x34040085, 0x34100081, 0x4a101082, 0x34140081, 0x4a141500, 0x4a141481, 0x34180083, 0x4a1a00ff, 0x5a5abe00, 0x4a1c00ff, 0x7f7f7f91,
        0x4a1e00ff, 0xa1b2c300, 0x4a2000bc, 0xe0301000, 0x80000301, 0xe0301000, 0x80010401, 0xe0341004, 0x80010501, 0xe0281000, 0x80010708, 0xe0201000,
        0x8001090a, 0xe8281002, 0x80011101, 0xf4200402, 0xfa000006, 0xbf8c0000, 0x7e160210, 0xe0701000, 0x80020302, 0xe0701004, 0x80020402, 0xe0701008,
        0x80020502, 0xe070100c, 0x80020602, 0xe0701010, 0x80020702, 0xe0701014, 0x80020902, 0xe0701018, 0x80020b02, 0xe070101c, 0x80021102, 0xe0701000,
        0x80030f0c, 0xe0601004, 0x80030e0c, 0xe0681006, 0x80030d0c, 0xe82c1005, 0x8003100c, 0xbf810000,
    ];
    /// s[0:3] input, s[4:7] view, s[8:11] results.
    pub const UNALIGNED_PAIR: [u32; 11] = [0x34040083, 0x7e0a0281, 0x7e0c0280, 0xe0345000, 0x80010300, 0xe1480100, 0x80020500, 0xbf8c0000, 0xe0741000, 0x80020302, 0xbf810000];
    /// s[0:3] the atomic V#, s[4:7] the returns, s8 = the offset added.
    pub const UNALIGNED_ATOMIC: [u32; 10] = [0x34040082, 0x4a040408, 0x4a020081, 0xe0c85000, 0x80000102, 0xbf8c3f70, 0x34060082, 0xe0701000, 0x80010103, 0xbf810000];

    /// s[0:3] view, s[4:7] results.
    pub fn formatted_load(format: u32, offset: u32) -> [u32; 7] {
        [0x34020082, 0xe8001000 | (format << 19) | offset, 0x80000201, 0xbf8c0000, 0xe0701000, 0x80010201, 0xbf810000]
    }

    pub fn formatted_store(format: u32, offset: u32) -> [u32; 6] {
        [0x34020082, 0x4a0400ff, 0x0000c300, 0xe8041000 | (format << 19) | offset, 0x80000201, 0xbf810000]
    }

    pub const FORMATTED_STORE_FILL: u32 = 0xaaaa_aaaa;
    pub const FORMATTED_STORE_LANE_BYTES: u32 = 64;
    /// (format, width, channels, components, offset)
    pub const FORMATTED_STORES: [(u32, u32, u32, u32, u32); 8] = [(5, 1, 1, 1, 1), (11, 2, 1, 1, 3), (60, 1, 4, 4, 6), (60, 1, 4, 3, 14), (11, 2, 1, 1, 19), (27, 2, 2, 2, 26), (69, 2, 4, 4, 33), (18, 1, 2, 2, 45)];

    /// FormattedStoreUnaligned: s[0:3] input (4 words per lane), s[4:7]
    /// output (64 bytes per lane).
    pub fn formatted_store_unaligned() -> Vec<u32> {
        let mut code = vec![0x34020084, 0x34060086];
        for c in 0..4u32 {
            code.extend([0xe0301000 | (c * 4), 0x80000401 | (c << 8)]);
        }
        code.push(0xbf8c3f70);
        code.extend([0x7e3c02ff, FORMATTED_STORE_FILL]);
        for offset in (0..FORMATTED_STORE_LANE_BYTES).step_by(4) {
            code.extend([0xe0701000 | offset, 0x80011e03]);
        }
        code.push(0xbf8c3f70);
        for (format, _, _, components, offset) in FORMATTED_STORES {
            code.extend([0xe8001000 | (format << 19) | ((3 + components) << 16) | offset, 0x80010403]);
        }
        code.push(0xbf810000);
        code
    }

    pub const CONDXCHG_SENTINEL: u32 = 0xdead_beef;
    pub const CONDXCHG_LDS_DWORDS: u32 = 4096;

    /// LdsCondxchg: s[0:3] input (8 words per lane), s[4:7] output.
    pub fn lds_condxchg(wave: u32, offset: u32, destination: u32, inactive: bool, contended: bool) -> Vec<u32> {
        let save_exec = if wave == 32 { 0xbe94_3c6a } else { 0xbe94_246a };
        let restore_exec = if wave == 32 { 0xbefe_03c1 } else { 0xbefe_04c1 };
        vec![
            0x34020085,
            0xe0381000,
            0x80000201,
            0xe0381010,
            0x80000601,
            0xbf8c3f70,
            0x7e160304,
            0x7e180305,
            0x7e1a02ff,
            CONDXCHG_SENTINEL,
            0x7e140280,
            if contended { 0x7d840080 } else { 0x7d841480 },
            save_exec,
            0xd9340000,
            0x00000206,
            0xd9340008,
            0x00000806,
            0xd9340010,
            0x00000806,
            0xbf8cc07f,
            restore_exec,
            0xbf8a0000,
            if inactive { 0x36140081 } else { 0x36140080 },
            0x7d841480,
            save_exec,
            0xd9f80000 | offset,
            (destination << 24) | 0x00000b07,
            0xbf8cc07f,
            restore_exec,
            0xbf8a0000,
            0x7e280300 | destination,
            0x7e2a0300 | (destination + 1),
            0xd9d80000,
            0x16000006,
            0xd9d80008,
            0x18000006,
            0xd9d80010,
            0x1a000006,
            0xbf8cc07f,
            0xe0781000,
            0x80011401,
            0xe0781010,
            0x80011801,
            0xbf810000,
        ]
    }

    /// The LdsCondxchg runs: (offset, destination, inactive, contended).
    pub const CONDXCHG_RUNS: [(u32, u32, bool, bool); 6] = [(0, 12, false, false), (0x120, 12, false, false), (0xfff8, 12, false, false), (0x120, 12, true, false), (0xfff8, 7, true, false), (0x120, 12, false, true)];

    /// LdsCondxchg's input rows.
    pub fn condxchg_input(threads: u32, offset: u32, contended: bool) -> Vec<u32> {
        const SIGN: u32 = 0x8000_0000;
        let mut input = vec![0u32; (threads * 8) as usize];
        for tid in 0..threads {
            let i = &mut input[(tid * 8) as usize..((tid + 1) * 8) as usize];
            let seed = ((tid + 1) as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            i[0] = if contended { 0xffff_ffff } else { seed as u32 };
            i[1] = if contended { 0xffff_fffe } else { (seed >> 32) as u32 };
            i[2] = if contended { SIGN | (tid + 1) } else { (if tid & 1 != 0 { SIGN } else { 0 }) | (if (tid / 4) % 2 != 0 { 0x7fff_ffff } else { 0 }) };
            i[3] = if contended { SIGN | (tid + 1) } else { (if tid & 2 != 0 { SIGN } else { 0 }) | (if (tid / 8) % 2 != 0 { 0x1357_9bdf } else { 0 }) };
            let out_of_range = !contended && (tid & 8 == 0 || offset >= CONDXCHG_LDS_DWORDS * 4);
            i[4] = if contended { 0 } else if out_of_range { tid * 32 } else { CONDXCHG_LDS_DWORDS * 4 - 24 - tid * 32 };
            i[5] = i[4].wrapping_add(if out_of_range { 0x10000 } else { 0 }).wrapping_sub(offset);
            i[6] = 0xa5a5_a5a5;
            i[7] = 0x5a5a_5a5a;
        }
        input
    }

    pub const APERTURE_SOURCES: [u32; 4] = [235, 236, 237, 238];
    pub const APERTURE_VALUES: [u64; 4] = [0x8000_0000_0000_0000, 0x8000_0000_ffff_ffff, 0x7000_0000_0000_0000, 0x7000_0000_ffff_ffff];
    pub const APERTURE_SHARED_DATA: u32 = 0x5eed_0000;
    pub const APERTURE_PRIVATE_DATA: u32 = 0x51de_0000;
    /// (op, source, pointer register, expected)
    pub const APERTURE_COMPARES: [(u32, u32, u32, u32); 8] = [(0xe3, 235, 3, 1), (0xe6, 236, 3, 1), (0xe3, 235, 7, 0), (0xe3, 237, 7, 1), (0xe6, 238, 7, 1), (0xe6, 238, 3, 0), (0xe6, 236, 12, 1), (0xe6, 236, 14, 0)];

    /// FlatApertureSources: s[0:3] output (32 words per lane); LDS 256
    /// dwords, 4 dwords of scratch.
    pub fn flat_aperture_sources() -> Vec<u32> {
        let (lane_offset, shared_ptr, shared_value, shared_flat, private_ptr, private_value, lds_read, scratch_read, edge_ptr, past_ptr, result, private_flat, one) =
            (1u32, 3u32, 5u32, 6u32, 7u32, 9u32, 10u32, 11u32, 12u32, 14u32, 16u32, 17u32, 20u32);
        let (narrow, wide, past_high) = (8u32, 12u32, 20u32);
        let vmove = |d: u32, s: u32| 0x7e00_0200 | (d << 17) | s;
        let flat = |code: &mut Vec<u32>, seg: u32, op: u32, vdst: u32, data: u32, addr: u32| {
            code.push(0xdc00_0000 | (op << 18) | (seg << 14));
            code.push((vdst << 24) | (0x7d << 16) | (data << 8) | addr);
        };
        let store = |code: &mut Vec<u32>, vdata: u32, slot: u32| {
            code.push(0xe070_1000 | (slot * 4));
            code.push(0x8000_0000 | lane_offset | (vdata << 8));
        };
        let mut code = vec![vop2(0x1a, lane_offset, 135, 0)];
        for (i, &s) in APERTURE_SOURCES.iter().enumerate() {
            code.push(sop1(0x03, narrow + i as u32, s));
            code.push(sop1(0x04, wide + i as u32 * 2, s));
        }
        for i in 0..12 {
            code.push(vmove(result, narrow + i));
            store(&mut code, result, i);
        }
        code.push(vop2(0x1a, shared_ptr, 130, 0));
        code.push(vmove(shared_ptr + 1, wide + 1));
        code.extend([vop2(0x25, shared_value, 255, 0), APERTURE_SHARED_DATA]);
        code.push(vmove(private_ptr, 128 + 4));
        code.push(vmove(private_ptr + 1, wide + 5));
        code.extend([vop2(0x25, private_value, 255, 0), APERTURE_PRIVATE_DATA]);
        flat(&mut code, 0, 0x1c, 0, shared_value, shared_ptr);
        flat(&mut code, 0, 0x1c, 0, private_value, private_ptr);
        code.push(0xbf8c_0070);
        code.extend([0xd8d8_0000, (lds_read << 24) | shared_ptr]);
        flat(&mut code, 1, 0x0c, scratch_read, 0, private_ptr);
        flat(&mut code, 0, 0x0c, shared_flat, 0, shared_ptr);
        flat(&mut code, 0, 0x0c, private_flat, 0, private_ptr);
        code.push(0xbf8c_0070);
        store(&mut code, lds_read, 12);
        store(&mut code, scratch_read, 13);
        store(&mut code, shared_flat, 14);
        store(&mut code, private_flat, 15);
        code.push(vmove(edge_ptr, wide + 2));
        code.push(vmove(edge_ptr + 1, wide + 3));
        code.push(0x8000_0000 | (past_high << 16) | (129 << 8) | (wide + 3));
        code.push(vmove(past_ptr, 128));
        code.push(vmove(past_ptr + 1, past_high));
        code.push(vmove(one, 129));
        for (i, &(op, source, pointer, _)) in APERTURE_COMPARES.iter().enumerate() {
            code.push(0x7c00_0000 | (op << 17) | (pointer << 9) | source);
            code.push(vop2(0x01, result, 128, one));
            store(&mut code, result, 16 + i as u32);
        }
        for (i, &s) in APERTURE_SOURCES.iter().enumerate() {
            code.push(0xd501_0000 | result);
            code.push(128 | (129 << 9) | (s << 18));
            store(&mut code, result, 24 + i as u32);
        }
        code.push(ENDPGM);
        code
    }

    // ---- 16-bit VALU: s[0:3] input, s[4:7] output.

    const MUL_LO_HEAD: [u32; 9] = [0x34020082, 0x34060080, 0xe0302000, 0x80000401, 0xe0302004, 0x80000501, 0xe0302008, 0x80000601, 0xbf8c3f70];
    const MUL_LO_TAIL: [u32; 3] = [0xe0702000, 0x80010a03, 0xbf810000];

    /// MulLoU16HighHalf's six programs (strided V#s, 64 invocations of
    /// wave32): zero high, add then mul, kept high, masked factors, packed
    /// low, packed high.
    pub fn mul_lo_u16() -> [Vec<u32>; 6] {
        let program = |body: &[u32]| [&MUL_LO_HEAD[..], body, &MUL_LO_TAIL[..]].concat();
        [
            program(&[0x36140cff, 0x0000ffff, 0xd705000a, 0x00020b04]),
            program(&[0x7e140280, 0xd703000a, 0x00020906, 0xd705000a, 0x00020b0a]),
            program(&[0x7e140306, 0xd705000a, 0x00020b04]),
            program(&[0x360808ff, 0x0000ffff, 0x360a0aff, 0x0000ffff, 0x36140cff, 0x0000ffff, 0xd705000a, 0x00020b04]),
            program(&[0x360808ff, 0x0000ffff, 0x360a0aff, 0x0000ffff, 0xcc01400a, 0x18020b04]),
            program(&[0x360808ff, 0xffff0000, 0x360a0aff, 0xffff0000, 0xcc01400a, 0x18020b04]),
        ]
    }

    pub const MUL_LO_EDGES: [[u32; 3]; 8] = [
        [0xffffffff, 0xffffffff, 0x00000000],
        [0x0000ffff, 0x0000ffff, 0xabcd1234],
        [0xffff0000, 0xffff0000, 0xffffffff],
        [0x8000ffff, 0x7fff8000, 0x12345678],
        [0x12345678, 0x9abcdef0, 0x80000000],
        [0x00010001, 0x00010001, 0x0000ffff],
        [0x0000ff00, 0x00000100, 0xffff0000],
        [0x00000000, 0xffffffff, 0x7fff7fff],
    ];

    /// MulLoU16HighHalf's input: 64 rows of 4 words.
    pub fn mul_lo_u16_input() -> Vec<u32> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut input = vec![0u32; 64 * 4];
        for tid in 0..64usize {
            for j in 0..3 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                input[tid * 4 + j] = if tid < MUL_LO_EDGES.len() { MUL_LO_EDGES[tid][j] } else { (state >> 32) as u32 };
            }
        }
        input
    }

    /// Float16MinMaxBits (8 results per lane).
    pub const F16_MIN_MAX_SCALAR: [u32; 24] = [
        0x34020084, 0x34060085, 0xe0381000, 0x80000401, 0xbf8c3f70, 0x74140b04, 0x72160b04, 0xd539810c, 0x20020d04, 0xd53a800d, 0x40020d05, 0xd751000e,
        0x041a0b04, 0xd754280f, 0x441a0b04, 0xd7570010, 0x041a0b04, 0xd7579111, 0x841e0b04, 0xe0781000, 0x80010a03, 0xe0781010, 0x80010e03, 0xbf810000,
    ];
    pub const F16_MIN_MAX_PACKED: [u32; 16] = [
        0x34020084, 0x34060085, 0xe0381000, 0x80000401, 0xbf8c3f70, 0xcc11400a, 0x18020b04, 0xcc12400b, 0x18020b04, 0xcc12ca0c, 0x30020d04, 0xcc11c00d,
        0x58020f05, 0xe0781000, 0x80010a03, 0xbf810000,
    ];

    /// VCvtPknormF16SourceModifiers (2 results per lane).
    pub fn cvt_pknorm(abs: u32, neg: u32, op_sel: u32) -> [u32; 14] {
        let sources = 0x00020b04 | neg << 29;
        [0x34020084, 0x34060083, 0xe0381000, 0x80000401, 0xbf8c3f70, 0xd7120000 | op_sel << 11 | abs << 8 | 10, sources, 0xd7130000 | op_sel << 11 | abs << 8 | 11, sources, 0xe0701000, 0x80010a03, 0xe0701004, 0x80010b03, 0xbf810000]
    }

    /// VCmpF32Denormals: one result per lane (the lane's bit of the
    /// captured VCC or EXEC).
    pub fn cmp_f32(compare: u32, exec: bool) -> [u32; 14] {
        [0x34020084, 0x34060082, 0xe0381000, 0x80000401, 0xbf8c3f70, 0xbe94037e, compare, if exec { 0xbe95037e } else { 0xbe95036a }, 0xbefe0314, 0xd501000a, 0x00550280, 0xe0701000, 0x80010a03, 0xbf810000]
    }

    pub const CMP_F32_VALUES: [u32; 16] = [
        0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000,
        0x7fc00000, 0x7f800001, 0xffc00000, 0x7f7fffff,
    ];

    /// VCmpEqF16Clamp: a VOP3 compare into VCC with the clamp bit set
    /// (strided V#s, 4 words per lane).
    pub fn cmp_clamp(op: u32, s0: u32, s1: u32) -> [u32; 18] {
        let w0 = 0xd400_0000 | (op << 16) | 106 | 0x8000;
        let w1 = (256 + s0) | ((256 + s1) << 9);
        [
            0x34020082, 0x34060082, 0xe0302000, 0x80000401, 0xe0302004, 0x80000501, 0xe0302008, 0x80000601, 0xe030200c, 0x80000701, 0xbf8c3f70, w0, w1, 0xd5010014,
            0x01a90280, 0xe0702000, 0x80011403, 0xbf810000,
        ]
    }

    /// F16ToF32Unpack: s8 = work-group id (32 lanes each).
    pub const F16_TO_F32_CONVERT: [u32; 10] = [0x34020082, 0x8f098708, 0x4a020209, 0xe0301000, 0x80000401, 0xbf8c3f70, 0x7e0a1704, 0xe0701000, 0x80010501, 0xbf810000];
    pub const F16_TO_F32_BYTE: [u32; 13] = [0x34020082, 0x8f098708, 0x4a020209, 0xe0301000, 0x80000401, 0x7e0e02ff, 0x00008000, 0xbf8c3f70, 0x640c0ef9, 0x06010604, 0xe0701000, 0x80010601, 0xbf810000];
    pub const F16_TO_F32_COMPARE: [u32; 22] = [
        0x34020082, 0x34040083, 0x8f098708, 0x8f0a8808, 0x4a020209, 0x4a04040a, 0xe0301000, 0x80000401, 0x7e0c02ff, 0x00003c00, 0xbf8c3f70, 0x7dd00904,
        0xd5010005, 0x01a90280, 0x7d920d04, 0xd5010007, 0x01a90280, 0xe0701000, 0x80010502, 0xe0701004, 0x80010702, 0xbf810000,
    ];

    /// Dpp8PkFmac (4 words per lane: fused, separate, plain).
    pub fn dpp8_pk_fmac(lanes: [u32; 8]) -> [u32; 16] {
        let selectors = lanes.iter().enumerate().fold(0u32, |v, (l, &s)| v | (s << (l * 3)));
        let dpp8 = |src: u32| (selectors << 8) | src;
        [0x34080084, 0xe03c1000, 0x80000104, 0xbf8c3f70, 0x7e140301, 0x781406e9, dpp8(2), 0x7e1a02e9, dpp8(2), 0x7e160301, 0x7816070d, 0x7e180301, 0x78180702, 0xe07c1000, 0x80010a04, 0xbf810000]
    }

    pub fn dpp8_half(seed: u32) -> u32 {
        let sign = (seed >> 15) & 1;
        let exponent = 10 + seed % 10;
        let mantissa = (seed >> 4) & 0x3ff;
        (sign << 15) | (exponent << 10) | mantissa
    }

    /// Dpp8PkFmac's input: 64 rows of 4 words.
    pub fn dpp8_input() -> Vec<u32> {
        let mut input = vec![0u32; 64 * 4];
        for lane in 0..64u32 {
            for word in 0..3u32 {
                let seed = lane.wrapping_mul(2654435761).wrapping_add(word * 40503).wrapping_add(12345);
                input[(lane * 4 + word) as usize] = dpp8_half(seed) | (dpp8_half(seed.wrapping_mul(7).wrapping_add(3)) << 16);
            }
        }
        input
    }

    /// Float16Native (16 results per lane) and PackedFloat16Native (8).
    pub const F16_NATIVE_SCALAR: [u32; 41] = [
        0x34020084, 0x34060086, 0xe0381000, 0x80000401, 0xbf8c3f70, 0x64140b04, 0x66160b04, 0x68180b04, 0x6a1a0b04, 0xd74b000e, 0x041a0b04, 0xd74b980f,
        0x041a0b04, 0xd74b3110, 0xc4160d04, 0x7e220306, 0x6c220b04, 0x6e240b04, 0x00003d55, 0x70260b04, 0x00000001, 0xd74ba214, 0x241a0b04, 0x662a08f2,
        0xd5328116, 0x20020b04, 0xd5350217, 0x40020d04, 0xd5338018, 0x00020b04, 0xd5340019, 0x20020b04, 0xe0781000, 0x80010a03, 0xe0781010, 0x80010e03,
        0xe0781020, 0x80011203, 0xe0781030, 0x80011603, 0xbf810000,
    ];
    pub const F16_NATIVE_PACKED: [u32; 26] = [
        0x34020084, 0x34060085, 0xe0381000, 0x80000401, 0xbf8c3f70, 0xcc0f400a, 0x18020b04, 0xcc10400b, 0x18020b04, 0xcc0e400c, 0x1c1a0b04, 0xcc0f4a0d,
        0x30020b04, 0xcc10c00e, 0x18020d04, 0xcc0e330f, 0x8c1a0b04, 0x7e200306, 0x78200b04, 0xcc0ec011, 0x3c1a0f04, 0xe0781000, 0x80010a03, 0xe0781010,
        0x80010e03, 0xbf810000,
    ];

    /// The SplitMix64 sequence of the reference tests.
    pub struct SplitMix(pub u64);

    impl SplitMix {
        pub fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            (z ^ (z >> 31)) as u32
        }
    }

    /// Float16Native's half generator.
    pub fn f16_native_half(rng: &mut SplitMix) -> u32 {
        let bits = rng.next();
        let sign = (bits >> 31) << 15;
        let mantissa = bits & 0x3ff;
        match (bits >> 16) % 10 {
            0 => sign,
            1 => sign | mantissa.max(1),
            2 => sign | 0x7c00,
            3 => sign | 0x7c00 | mantissa.max(1),
            4 => sign | ((0x1d + (bits >> 20) % 2) << 10) | mantissa,
            5 => sign | ((1 + (bits >> 20) % 4) << 10) | mantissa,
            6 => sign | ((0x0d + (bits >> 20) % 5) << 10) | mantissa,
            _ => bits & 0xffff,
        }
    }

    /// Float16MinMaxBits's half generator.
    pub fn f16_min_max_half(rng: &mut SplitMix) -> u32 {
        let bits = rng.next();
        let sign = (bits >> 31) << 15;
        let mantissa = bits & 0x3ff;
        match (bits >> 16) % 9 {
            0 => sign,
            1 => sign | mantissa.max(1),
            2 => sign | 0x7c00,
            3 => sign | 0x7c00 | mantissa.max(1),
            4 => sign | 0x3c00 | (bits & 3),
            _ => bits & 0xffff,
        }
    }
}
