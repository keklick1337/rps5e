// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

// Reference compute cases: 16-bit integer VALU (VOPC I16/U16 compares, VOP3 16-bit ALU, VOP3P packed 16-bit).
// Machine-code words and buffer contents of each case; see ref_harness.rs.
// Each case stores its results through `buffer_store_dword vN, v31, s[48:51] offen`.

pub const VOP3P_INT16_OPS: RefCase = RefCase {
    name: "Vop3pInt16Ops",
    code: &[
        0x7e0202ff, 0x80007fff, 0x7e0402ff, 0x00010001, 0x7e0602ff, 0xfffe0003, 0x7e0802ff, 0x00130004,
        0x7e0a02ff, 0x8010f00f, 0x7e0c02ff, 0x00050007, 0xcc00400a, 0x1c1a0701, 0xcc09400b, 0x1c1a0701,
        0xcc01000c, 0x18020701, 0xcc02000d, 0x18020501, 0xcc0b000e, 0x18020702, 0xcc04000f, 0x18020b04,
        0xcc050010, 0x18020b04, 0xcc060011, 0x18020b04, 0xcc080012, 0x18020501, 0xcc0c0013, 0x18020501,
        0xcc0d0014, 0x18020501, 0x7e3e0280, 0xe0701000, 0x800c0a1f, 0x7e3e0284, 0xe0701000, 0x800c0b1f,
        0x7e3e0288, 0xe0701000, 0x800c0c1f, 0x7e3e028c, 0xe0701000, 0x800c0d1f, 0x7e3e0290, 0xe0701000,
        0x800c0e1f, 0x7e3e0294, 0xe0701000, 0x800c0f1f, 0x7e3e0298, 0xe0701000, 0x800c101f, 0x7e3e029c,
        0xe0701000, 0x800c111f, 0x7e3e02a0, 0xe0701000, 0x800c121f, 0x7e3e02a4, 0xe0701000, 0x800c131f,
        0x7e3e02a8, 0xe0701000, 0x800c141f, 0xbf810000,
    ],
    initial: &[],
    expected: &[
        0x00058004, 0x00058004, 0x00007ffd, 0x80018000, 0x0003fffe, 0x008000f0, 0x10020f00, 0xf002ff00,
        0x80000001, 0x80007fff, 0x00010001,
    ],
    user_data: None,
    compute: Compute {
        threads: [0, 0, 0],
        lds_dwords: 1024,
        wave_size: 64,
        float_mode: 0xc0,
        dispatch_threads: [0, 0, 0],
        workgroup_register: 0,
        thread_ids: 0,
        group_id: [false, false, false],
        dispatch_dims: false,
        tg_size_en: false,
    },
    dispatch: [1, 1, 1],
    compile_only: false,
    tolerance: 0.0,
    bda: &[],
};

pub const VOP3_INT16_OPS: RefCase = RefCase {
    name: "Vop3Int16Ops",
    code: &[
        0x7e0202ff, 0xabcd0001, 0x7e0402ff, 0x12340002, 0x7e0602ff, 0x55557fff, 0x7e0802ff, 0x66668000,
        0x7e0a02ff, 0x77770013, 0x7e0c02ff, 0x88888010, 0x7e1402ff, 0xaaaa0000, 0x7e1602ff, 0xbbbb0000,
        0x7e1802ff, 0xcccc0000, 0x7e1a02ff, 0xdddd0000, 0x7e1c02ff, 0xeeee0000, 0x7e1e02ff, 0x11110000,
        0xd704000a, 0x00020501, 0xd70d000b, 0x00020303, 0xd708000c, 0x00020d05, 0xd709000d, 0x00020903,
        0xd70b000e, 0x00020903, 0xd708000f, 0x00020705, 0x7e3e0280, 0xe0701000, 0x800c0a1f, 0x7e3e0284,
        0xe0701000, 0x800c0b1f, 0x7e3e0288, 0xe0701000, 0x800c0c1f, 0x7e3e028c, 0xe0701000, 0x800c0d1f,
        0x7e3e0290, 0xe0701000, 0x800c0e1f, 0x7e3e0294, 0xe0701000, 0x800c0f1f, 0xbf810000,
    ],
    initial: &[],
    expected: &[
        0xaaaaffff, 0xbbbb8000, 0xccccf002, 0xdddd8000, 0xeeee7fff, 0x11110fff,
    ],
    user_data: None,
    compute: Compute {
        threads: [0, 0, 0],
        lds_dwords: 1024,
        wave_size: 64,
        float_mode: 0xc0,
        dispatch_threads: [0, 0, 0],
        workgroup_register: 0,
        thread_ids: 0,
        group_id: [false, false, false],
        dispatch_dims: false,
        tg_size_en: false,
    },
    dispatch: [1, 1, 1],
    compile_only: false,
    tolerance: 0.0,
    bda: &[],
};

pub const VECTOR_COMPARE_I16_U16_OPS: RefCase = RefCase {
    name: "VectorCompareI16U16Ops",
    code: &[
        0x7e020281, 0x7e0402ff, 0xabcd8000, 0x7e0602ff, 0x00007fff, 0x7e0802ff, 0x12348000, 0x7e0a02ff,
        0xffff0001, 0x7e0c02ff, 0x00000001, 0x7d140902, 0x02140280, 0x7e3e0280, 0xe0701000, 0x800c0a1f,
        0x7d140702, 0x02160280, 0x7e3e0284, 0xe0701000, 0x800c0b1f, 0x7d160702, 0x02180280, 0x7e3e0288,
        0xe0701000, 0x800c0c1f, 0x7d160503, 0x021a0280, 0x7e3e028c, 0xe0701000, 0x800c0d1f, 0x7d160d05,
        0x021c0280, 0x7e3e0290, 0xe0701000, 0x800c0e1f, 0x7d180503, 0x021e0280, 0x7e3e0294, 0xe0701000,
        0x800c0f1f, 0x7d180702, 0x02200280, 0x7e3e0298, 0xe0701000, 0x800c101f, 0x7d1a0702, 0x02220280,
        0x7e3e029c, 0xe0701000, 0x800c111f, 0x7d1a0d05, 0x02240280, 0x7e3e02a0, 0xe0701000, 0x800c121f,
        0x7d540902, 0x02260280, 0x7e3e02a4, 0xe0701000, 0x800c131f, 0x7d540702, 0x02280280, 0x7e3e02a8,
        0xe0701000, 0x800c141f, 0x7d560503, 0x022a0280, 0x7e3e02ac, 0xe0701000, 0x800c151f, 0x7d560702,
        0x022c0280, 0x7e3e02b0, 0xe0701000, 0x800c161f, 0x7d580702, 0x022e0280, 0x7e3e02b4, 0xe0701000,
        0x800c171f, 0x7d580503, 0x02300280, 0x7e3e02b8, 0xe0701000, 0x800c181f, 0x7d5a0702, 0x02320280,
        0x7e3e02bc, 0xe0701000, 0x800c191f, 0x7d5a0d05, 0x02340280, 0x7e3e02c0, 0xe0701000, 0x800c1a1f,
        0x7d5c0702, 0x02360280, 0x7e3e02ff, 0x00000044, 0xe0701000, 0x800c1b1f, 0x7d5c0503, 0x02380280,
        0x7e3e02ff, 0x00000048, 0xe0701000, 0x800c1c1f, 0x7d5c0d05, 0x023a0280, 0x7e3e02ff, 0x0000004c,
        0xe0701000, 0x800c1d1f, 0xbf810000,
    ],
    initial: &[],
    expected: &[
        0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000001, 0x00000000, 0x00000001,
        0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001,
        0x00000000, 0x00000001, 0x00000000, 0x00000001,
    ],
    user_data: None,
    compute: Compute {
        threads: [0, 0, 0],
        lds_dwords: 1024,
        wave_size: 64,
        float_mode: 0xc0,
        dispatch_threads: [0, 0, 0],
        workgroup_register: 0,
        thread_ids: 0,
        group_id: [false, false, false],
        dispatch_dims: false,
        tg_size_en: false,
    },
    dispatch: [1, 1, 1],
    compile_only: false,
    tolerance: 0.0,
    bda: &[],
};

