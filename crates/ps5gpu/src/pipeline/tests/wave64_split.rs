// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Wave64 compute programs split over two 32-wide host subgroups
//! (`shader::wave_split`) on the device, compared word for word with the
//! IR interpreter: every operation a split wave rebuilds across its halves
//! (ballots with partial EXEC, lane reads from the high half, the first
//! active lane in the high half, DPP / permlane / swizzle / bpermute next
//! to the half boundary, saveexec branches and loops with divergent lanes,
//! s_barrier and LDS with several waves, a partial wave). Each program
//! runs split with host shuffles, split with every shuffle through
//! workgroup memory, and paired (two guest lanes per invocation).
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored wave64_split`
//! (they need a device; the split runs need a 32-wide subgroup).
//!
//! The programs (gfx1013, wave64) were assembled with llvm-mc; the source
//! is next to each word.

use super::ref_device::run_case;
use super::*;
use crate::shader::ref_tests::harness::{Compute, RefCase};
use crate::shader::{wave_split, Features, SpirvCompiler};

/// Ballots of compares under a partial EXEC, saved EXEC, lane reads of a
/// lane in the high half (constant and SGPR index, a register written
/// under EXEC), the first active lane when only high lanes are active and
/// a whole-wave bit count. Outputs: 10 regions of 64 dwords.
const BALLOTS_AND_LANE_READS: &[u32] = &[
    0x343c0082, // v_lshlrev_b32_e32 v30, 2, v0
    0x16060087, // v_mul_u32_u24_e32 v3, 7, v0
    0x4a060681, // v_add_nc_u32_e32 v3, 1, v3
    0x7e080280, // v_mov_b32_e32 v4, 0
    0xbe9c047e, // s_mov_b64 s[28:29], exec
    0x36020085, // v_and_b32_e32 v1, 5, v0
    0x7d8a0280, // v_cmp_ne_u32_e32 vcc, 0, v1
    0xbe90246a, // s_and_saveexec_b64 s[16:17], vcc
    0xd4c10012, 0x00020094, // v_cmp_lt_u32_e64 s[18:19], 20, v0
    0x7e080303, // v_mov_b32_e32 v4, v3
    0x7e020212, // v_mov_b32_e32 v1, s18
    0x7e040213, // v_mov_b32_e32 v2, s19
    0xe0701000, 0x800c011e, // buffer_store_dword v1, v30, s[48:51], 0 offen
    0xe0701100, 0x800c021e, // buffer_store_dword v2, v30, s[48:51], 0 offen offset:256
    0xbefe041c, // s_mov_b64 exec, s[28:29]
    0xd7600014, 0x02016503, // v_readlane_b32 s20, v3, 50
    0xbe9503a1, // s_mov_b32 s21, 33
    0xd7600016, 0x02002b03, // v_readlane_b32 s22, v3, s21
    0xd7600018, 0x02015304, // v_readlane_b32 s24, v4, 41
    0xd7600019, 0x02015104, // v_readlane_b32 s25, v4, 40
    0xbefe0380, // s_mov_b32 exec_lo, 0
    0xbeff03ff, 0x00f0f000, // s_mov_b32 exec_hi, 0xf0f000
    0x7e2e0503, // v_readfirstlane_b32 s23, v3
    0xbefe041c, // s_mov_b64 exec, s[28:29]
    0xd4c4001a, 0x000200a5, // v_cmp_gt_u32_e64 s[26:27], 37, v0
    0xbe9a101a, // s_bcnt1_i32_b64 s26, s[26:27]
    0x7e020214, // v_mov_b32_e32 v1, s20
    0xe0701200, 0x800c011e, // buffer_store_dword v1, v30, s[48:51], 0 offen offset:512
    0x7e020216, // v_mov_b32_e32 v1, s22
    0xe0701300, 0x800c011e, // buffer_store_dword v1, v30, s[48:51], 0 offen offset:768
    0x7e020217, // v_mov_b32_e32 v1, s23
    0xe0701400, 0x800c011e, // buffer_store_dword v1, v30, s[48:51], 0 offen offset:1024
    0x7e02021a, // v_mov_b32_e32 v1, s26
    0xe0701500, 0x800c011e, // buffer_store_dword v1, v30, s[48:51], 0 offen offset:1280
    0x7e020218, // v_mov_b32_e32 v1, s24
    0xe0701600, 0x800c011e, // buffer_store_dword v1, v30, s[48:51], 0 offen offset:1536
    0x7e020219, // v_mov_b32_e32 v1, s25
    0xe0701700, 0x800c011e, // buffer_store_dword v1, v30, s[48:51], 0 offen offset:1792
    0x7e020210, // v_mov_b32_e32 v1, s16
    0xe0701800, 0x800c011e, // buffer_store_dword v1, v30, s[48:51], 0 offen offset:2048
    0x7e020211, // v_mov_b32_e32 v1, s17
    0xe0701900, 0x800c011e, // buffer_store_dword v1, v30, s[48:51], 0 offen offset:2304
    0xbf810000, // s_endpgm
];

/// Lane moves with lanes 28..35 inactive (EXEC 0x0fffffff / 0xfffffff0):
/// DPP row_shr, row_ror (bound_ctrl), quad_perm, row_xmask, row_share,
/// row_shl with row/bank masks, DPP8, v_permlanex16, ds_swizzle (swap
/// neighbours) and ds_bpermute. Outputs: 10 regions.
const LANE_MOVES_AT_THE_HALF_BOUNDARY: &[u32] = &[
    0x343c0082, // v_lshlrev_b32_e32 v30, 2, v0
    0x16060083, // v_mul_u32_u24_e32 v3, 3, v0
    0x4a0606ff, 0x00000064, // v_add_nc_u32_e32 v3, 0x64, v3
    0x16140085, // v_mul_u32_u24_e32 v10, 5, v0
    0x3614149f, // v_and_b32_e32 v10, 31, v10
    0x34141482, // v_lshlrev_b32_e32 v10, 2, v10
    0xbe9403ff, 0x76543210, // s_mov_b32 s20, 0x76543210
    0xbe9503ff, 0xfedcba98, // s_mov_b32 s21, 0xfedcba98
    0xbe9c047e, // s_mov_b64 s[28:29], exec
    0xbefe03ff, 0x0fffffff, // s_mov_b32 exec_lo, 0xfffffff
    0xbeff03d0, // s_mov_b32 exec_hi, -16
    0x7e020287, // v_mov_b32_e32 v1, 7
    0x7e0202fa, 0xff011103, // v_mov_b32_dpp v1, v3 row_shr:1 row_mask:0xf bank_mask:0xf
    0x7e040289, // v_mov_b32_e32 v2, 9
    0x7e0402fa, 0xff092303, // v_mov_b32_dpp v2, v3 row_ror:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
    0x7e08028b, // v_mov_b32_e32 v4, 11
    0x7e0802fa, 0xff001b03, // v_mov_b32_dpp v4, v3 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf
    0x7e0a028d, // v_mov_b32_e32 v5, 13
    0x7e0a02fa, 0xff016f03, // v_mov_b32_dpp v5, v3 row_xmask:15 row_mask:0xf bank_mask:0xf
    0x7e0c028f, // v_mov_b32_e32 v6, 15
    0x7e0c02fa, 0xff015503, // v_mov_b32_dpp v6, v3 row_share:5 row_mask:0xf bank_mask:0xf
    0x7e0e0291, // v_mov_b32_e32 v7, 17
    0xd7780007, 0x00542903, // v_permlanex16_b32 v7, v3, s20, s21
    0x7e100293, // v_mov_b32_e32 v8, 19
    0xd8d4041f, 0x08000003, // ds_swizzle_b32 v8, v3 offset:swizzle(SWAP,1)
    0x7e120295, // v_mov_b32_e32 v9, 21
    0xdacc0000, 0x0900030a, // ds_bpermute_b32 v9, v10, v3
    0x7e160297, // v_mov_b32_e32 v11, 23
    0x7e1602e9, 0x05397703, // v_mov_b32_dpp v11, v3 dpp8:[7,6,5,4,3,2,1,0]
    0x7e180299, // v_mov_b32_e32 v12, 25
    0x7e1802fa, 0xa5010f03, // v_mov_b32_dpp v12, v3 row_shl:15 row_mask:0xa bank_mask:0x5
    0xbefe041c, // s_mov_b64 exec, s[28:29]
    0xe0701000, 0x800c011e, // buffer_store_dword v1, v30, s[48:51], 0 offen
    0xe0701100, 0x800c021e, // buffer_store_dword v2, v30, s[48:51], 0 offen offset:256
    0xe0701200, 0x800c041e, // buffer_store_dword v4, v30, s[48:51], 0 offen offset:512
    0xe0701300, 0x800c051e, // buffer_store_dword v5, v30, s[48:51], 0 offen offset:768
    0xe0701400, 0x800c061e, // buffer_store_dword v6, v30, s[48:51], 0 offen offset:1024
    0xe0701500, 0x800c071e, // buffer_store_dword v7, v30, s[48:51], 0 offen offset:1280
    0xe0701600, 0x800c081e, // buffer_store_dword v8, v30, s[48:51], 0 offen offset:1536
    0xe0701700, 0x800c091e, // buffer_store_dword v9, v30, s[48:51], 0 offen offset:1792
    0xe0701800, 0x800c0b1e, // buffer_store_dword v11, v30, s[48:51], 0 offen offset:2048
    0xe0701900, 0x800c0c1e, // buffer_store_dword v12, v30, s[48:51], 0 offen offset:2304
    0xbf810000, // s_endpgm
];

/// A loop whose lanes leave it after `lane & 7` (+ 3 in the high half)
/// iterations (`s_and_b64 exec, exec, vcc` + `s_cbranch_execz`), then an
/// if/else on even lanes through saveexec, xor and or (float arithmetic
/// the host rounds like the hardware).
/// Outputs: 3 regions.
/// A lane stack and SGPR spills (the code a compiler makes for a
/// wave-coherent traversal): 45 pushes with `v_writelane` at an SGPR lane
/// (crossing into the high half, with EXEC 0xffff), pops with
/// `v_readlane` at that lane into an order-dependent hash, spills to lanes
/// 5, 33 and 40 of v12 read back after the loops, both registers used as
/// per-lane values, and a run of six writes under EXEC 0xffff (an EXEC
/// run branch) of which one adds to v12. Outputs: 8 regions.
const LANE_STACK_AND_SPILLS: &[u32] = &[
    0x343c0082, // v_lshlrev_b32_e32 v30, 2, v0
    0x16060087, // v_mul_u32_u24_e32 v3, 7, v0
    0x7e2e02ff, 0x00000055, // v_mov_b32_e32 v23, 0x55
    0x7e180280, // v_mov_b32_e32 v12, 0
    0xbe8c03ff, 0x00001234, // s_mov_b32 s12, 0x1234
    0xbe8d03ff, 0x00005678, // s_mov_b32 s13, 0x5678
    0xd761000c, 0x02010a0c, // v_writelane_b32 v12, s12, 5
    0xd761000c, 0x0201500d, // v_writelane_b32 v12, s13, 40
    0xbe9c047e, // s_mov_b64 s[28:29], exec
    0xbefe03ff, 0x0000ffff, // s_mov_b32 exec_lo, 0xffff
    0xbeff0380, // s_mov_b32 exec_hi, 0
    0xbeb10380, // s_mov_b32 s49, 0
    0xbe8e0380, // s_mov_b32 s14, 0
    0x930f830e, // s_mul_i32 s15, s14, 3
    0x800f810f, // s_add_u32 s15, s15, 1
    0xd7610017, 0x0200620f, // v_writelane_b32 v23, s15, s49
    0x80318131, // s_add_u32 s49, s49, 1
    0x800e810e, // s_add_u32 s14, s14, 1
    0xbf0aad0e, // s_cmp_lt_u32 s14, 45
    0xbf85fff8, // s_cbranch_scc1 push
    0x4a500681, // v_add_nc_u32_e32 v40, 1, v3
    0x4a520682, // v_add_nc_u32_e32 v41, 2, v3
    0x4a181883, // v_add_nc_u32_e32 v12, 3, v12
    0x4a540684, // v_add_nc_u32_e32 v42, 4, v3
    0x4a560685, // v_add_nc_u32_e32 v43, 5, v3
    0x4a580686, // v_add_nc_u32_e32 v44, 6, v3
    0xbefe041c, // s_mov_b64 exec, s[28:29]
    0x80b18731, // s_sub_u32 s49, s49, 7
    0xd761000c, 0x02014231, // v_writelane_b32 v12, s49, 33
    0x4a040717, // v_add_nc_u32_e32 v2, v23, v3
    0xbe900380, // s_mov_b32 s16, 0
    0x80b18131, // s_sub_u32 s49, s49, 1
    0xd7600011, 0x02006317, // v_readlane_b32 s17, v23, s49
    0x93109f10, // s_mul_i32 s16, s16, 31
    0x80101110, // s_add_u32 s16, s16, s17
    0xbf078031, // s_cmp_lg_u32 s49, 0
    0xbf85fff9, // s_cbranch_scc1 pop
    0xd7600012, 0x02010b0c, // v_readlane_b32 s18, v12, 5
    0xd7600013, 0x0201510c, // v_readlane_b32 s19, v12, 40
    0xd7600014, 0x0201430c, // v_readlane_b32 s20, v12, 33
    0x7e020210, // v_mov_b32_e32 v1, s16
    0xe0701000, 0x800c011e, // buffer_store_dword v1, v30, s[48:51], 0 offen
    0x7e020212, // v_mov_b32_e32 v1, s18
    0xe0701100, 0x800c011e, // buffer_store_dword v1, v30, s[48:51], 0 offen offset:256
    0x7e020213, // v_mov_b32_e32 v1, s19
    0xe0701200, 0x800c011e, // buffer_store_dword v1, v30, s[48:51], 0 offen offset:512
    0x7e020214, // v_mov_b32_e32 v1, s20
    0xe0701300, 0x800c011e, // buffer_store_dword v1, v30, s[48:51], 0 offen offset:768
    0xe0701400, 0x800c171e, // buffer_store_dword v23, v30, s[48:51], 0 offen offset:1024
    0xe0701500, 0x800c0c1e, // buffer_store_dword v12, v30, s[48:51], 0 offen offset:1280
    0xe0701600, 0x800c021e, // buffer_store_dword v2, v30, s[48:51], 0 offen offset:1536
    0x4a505328, // v_add_nc_u32_e32 v40, v40, v41
    0x4a505528, // v_add_nc_u32_e32 v40, v40, v42
    0x4a505728, // v_add_nc_u32_e32 v40, v40, v43
    0x4a505928, // v_add_nc_u32_e32 v40, v40, v44
    0xe0701700, 0x800c281e, // buffer_store_dword v40, v30, s[48:51], 0 offen offset:1792
    0xbf810000, // s_endpgm
];

/// Lane reads at the first set lane of a compare mask (`s_ff1_i32_b64`
/// then `v_readlane`): the first lane in the high half, an empty mask
/// (lane 63), the first lane in the low half, a mask made under EXEC
/// 0xff00 in the high word, and a compare of a register written in a run
/// of six writes under EXEC 0xffff (an EXEC run branch: the compare's
/// operand becomes a phi of the run's join). Outputs: 7 regions.
const FIRST_LANE_READS: &[u32] = &[
    0x343c0082, // v_lshlrev_b32_e32 v30, 2, v0
    0x16060087, // v_mul_u32_u24_e32 v3, 7, v0
    0x7e0c02ff, 0x00000100, // v_mov_b32_e32 v6, 0x100
    0x7e0e02ff, 0x00000100, // v_mov_b32_e32 v7, 0x100
    0x7e1002ff, 0x00000100, // v_mov_b32_e32 v8, 0x100
    0x7e1202ff, 0x00000100, // v_mov_b32_e32 v9, 0x100
    0x7e1402ff, 0x00000100, // v_mov_b32_e32 v10, 0x100
    0x7e1602ff, 0x00000100, // v_mov_b32_e32 v11, 0x100
    0x7d8200a8, // v_cmp_lt_u32_e32 vcc, 40, v0
    0x020206c1, // v_cndmask_b32_e32 v1, -1, v3, vcc
    0xd4c50014, 0x000202c1, // v_cmp_ne_u32_e64 s[20:21], -1, v1
    0xbe961414, // s_ff1_i32_b64 s22, s[20:21]
    0xd7600017, 0x02002d03, // v_readlane_b32 s23, v3, s22
    0xd7600018, 0x02002d01, // v_readlane_b32 s24, v1, s22
    0xd4c4001a, 0x00020080, // v_cmp_gt_u32_e64 s[26:27], 0, v0
    0xbe99141a, // s_ff1_i32_b64 s25, s[26:27]
    0xd760001c, 0x02003303, // v_readlane_b32 s28, v3, s25
    0xd4c2001e, 0x00020089, // v_cmp_eq_u32_e64 s[30:31], 9, v0
    0xbe9d141e, // s_ff1_i32_b64 s29, s[30:31]
    0xd7600020, 0x02003b03, // v_readlane_b32 s32, v3, s29
    0xbea2047e, // s_mov_b64 s[34:35], exec
    0xbefe0380, // s_mov_b32 exec_lo, 0
    0xbeff03ff, 0x0000ff00, // s_mov_b32 exec_hi, 0xff00
    0xd4c10024, 0x000200ac, // v_cmp_lt_u32_e64 s[36:37], 44, v0
    0xbea11424, // s_ff1_i32_b64 s33, s[36:37]
    0xbefe0422, // s_mov_b64 exec, s[34:35]
    0xd7600026, 0x02004303, // v_readlane_b32 s38, v3, s33
    0x7e040217, // v_mov_b32_e32 v2, s23
    0xe0701000, 0x800c021e, // buffer_store_dword v2, v30, s[48:51], 0 offen
    0x7e040218, // v_mov_b32_e32 v2, s24
    0xe0701100, 0x800c021e, // buffer_store_dword v2, v30, s[48:51], 0 offen offset:256
    0x7e04021c, // v_mov_b32_e32 v2, s28
    0xe0701200, 0x800c021e, // buffer_store_dword v2, v30, s[48:51], 0 offen offset:512
    0x7e040220, // v_mov_b32_e32 v2, s32
    0xe0701300, 0x800c021e, // buffer_store_dword v2, v30, s[48:51], 0 offen offset:768
    0x7e040226, // v_mov_b32_e32 v2, s38
    0xe0701400, 0x800c021e, // buffer_store_dword v2, v30, s[48:51], 0 offen offset:1024
    0xbefe03ff, 0x0000ffff, // s_mov_b32 exec_lo, 0xffff
    0xbeff0380, // s_mov_b32 exec_hi, 0
    0x4a0c0081, // v_add_nc_u32_e32 v6, 1, v0
    0x4a0e0082, // v_add_nc_u32_e32 v7, 2, v0
    0x4a100083, // v_add_nc_u32_e32 v8, 3, v0
    0x4a120084, // v_add_nc_u32_e32 v9, 4, v0
    0x4a140085, // v_add_nc_u32_e32 v10, 5, v0
    0x4a160086, // v_add_nc_u32_e32 v11, 6, v0
    0xbefe0422, // s_mov_b64 exec, s[34:35]
    0xd4c10028, 0x00020c8a, // v_cmp_lt_u32_e64 s[40:41], 10, v6
    0xbea71428, // s_ff1_i32_b64 s39, s[40:41]
    0xd760002a, 0x02004f03, // v_readlane_b32 s42, v3, s39
    0x7e04022a, // v_mov_b32_e32 v2, s42
    0xe0701500, 0x800c021e, // buffer_store_dword v2, v30, s[48:51], 0 offen offset:1280
    0x4a041107, // v_add_nc_u32_e32 v2, v7, v8
    0x4a041302, // v_add_nc_u32_e32 v2, v2, v9
    0x4a041502, // v_add_nc_u32_e32 v2, v2, v10
    0x4a041702, // v_add_nc_u32_e32 v2, v2, v11
    0xe0701600, 0x800c021e, // buffer_store_dword v2, v30, s[48:51], 0 offen offset:1536
    0xbf810000, // s_endpgm
];

const DIVERGENT_LOOP_AND_IF_ELSE: &[u32] = &[
    0x343c0082, // v_lshlrev_b32_e32 v30, 2, v0
    0x36020087, // v_and_b32_e32 v1, 7, v0
    0x7d82009f, // v_cmp_lt_u32_e32 vcc, 31, v0
    0xd5010004, 0x01a90680, // v_cndmask_b32_e64 v4, 0, 3, vcc
    0x4a020901, // v_add_nc_u32_e32 v1, v1, v4
    0x7e040280, // v_mov_b32_e32 v2, 0
    0xbe90047e, // s_mov_b64 s[16:17], exec
    0xbe980380, // s_mov_b32 s24, 0
    0x7d8a0280, // v_cmp_ne_u32_e32 vcc, 0, v1
    0x87fe6a7e, // s_and_b64 exec, exec, vcc
    0xbf880005, // s_cbranch_execz done
    0x80188118, // s_add_u32 s24, s24, 1
    0x4a040102, // v_add_nc_u32_e32 v2, v2, v0
    0x4a040483, // v_add_nc_u32_e32 v2, 3, v2
    0x4e020281, // v_subrev_nc_u32_e32 v1, 1, v1
    0xbf82fff8, // s_branch loop
    0xbefe0410, // s_mov_b64 exec, s[16:17]
    0x7e0c0d00, // v_cvt_f32_u32_e32 v6, v0
    0x100c0cff, 0x3d800000, // v_mul_f32_e32 v6, 0x3d800000, v6
    0x36080081, // v_and_b32_e32 v4, 1, v0
    0x7d840880, // v_cmp_eq_u32_e32 vcc, 0, v4
    0xbe92246a, // s_and_saveexec_b64 s[18:19], vcc
    0xbf880003, // s_cbranch_execz else_part
    0x100a0cff, 0x3fc00000, // v_mul_f32_e32 v5, 0x3fc00000, v6
    0x100a0b05, // v_mul_f32_e32 v5, v5, v5
    0x89fe127e, // s_xor_b64 exec, exec, s[18:19]
    0xbf880002, // s_cbranch_execz merge
    0x080a0cf4, // v_sub_f32_e32 v5, 2.0, v6
    0x060a0b05, // v_add_f32_e32 v5, v5, v5
    0x88fe127e, // s_or_b64 exec, exec, s[18:19]
    0x7e0e0218, // v_mov_b32_e32 v7, s24
    0xe0701000, 0x800c021e, // buffer_store_dword v2, v30, s[48:51], 0 offen
    0xe0701100, 0x800c051e, // buffer_store_dword v5, v30, s[48:51], 0 offen offset:256
    0xe0701200, 0x800c071e, // buffer_store_dword v7, v30, s[48:51], 0 offen offset:512
    0xbf810000, // s_endpgm
];

/// 256 threads (four waves): LDS written, s_barrier, read from the next
/// wave; LDS written and read from the other half of the same wave without
/// s_barrier (the subgroup LDS barrier); a whole-wave count, a lane read
/// and an `s_cbranch_execz` that some waves take, all in control flow
/// every wave follows. Outputs: 7 regions of 256 dwords.
const WAVES_BARRIER_AND_LDS: &[u32] = &[
    0x343c0082, // v_lshlrev_b32_e32 v30, 2, v0
    0x16060083, // v_mul_u32_u24_e32 v3, 3, v0
    0x4a060685, // v_add_nc_u32_e32 v3, 5, v3
    0xd8340000, 0x0000031e, // ds_write_b32 v30, v3
    0xbf8a0000, // s_barrier
    0x4a023cff, 0x00000100, // v_add_nc_u32_e32 v1, 0x100, v30
    0x360202ff, 0x000003ff, // v_and_b32_e32 v1, 0x3ff, v1
    0xd8d80000, 0x02000001, // ds_read_b32 v2, v1
    0x3a103cff, 0x00000080, // v_xor_b32_e32 v8, 0x80, v30
    0x4a1010ff, 0x00000400, // v_add_nc_u32_e32 v8, 0x400, v8
    0x4a123cff, 0x00000400, // v_add_nc_u32_e32 v9, 0x400, v30
    0x1614008b, // v_mul_u32_u24_e32 v10, 11, v0
    0xd8340000, 0x00000a09, // ds_write_b32 v9, v10
    0xd8d80000, 0x0b000008, // ds_read_b32 v11, v8
    0x36080083, // v_and_b32_e32 v4, 3, v0
    0xd4c20010, 0x00020880, // v_cmp_eq_u32_e64 s[16:17], 0, v4
    0xbe921010, // s_bcnt1_i32_b64 s18, s[16:17]
    0xd7600013, 0x02015b03, // v_readlane_b32 s19, v3, 45
    0x7d8200ff, 0x00000060, // v_cmp_lt_u32_e32 vcc, 0x60, v0
    0xbe94047e, // s_mov_b64 s[20:21], exec
    0x87fe6a7e, // s_and_b64 exec, exec, vcc
    0x7e0a0281, // v_mov_b32_e32 v5, 1
    0xbf880001, // s_cbranch_execz skip
    0x7e0a0282, // v_mov_b32_e32 v5, 2
    0xbefe0414, // s_mov_b64 exec, s[20:21]
    0x7e0c0212, // v_mov_b32_e32 v6, s18
    0x7e0e0213, // v_mov_b32_e32 v7, s19
    0xb0160400, // s_movk_i32 s22, 0x400
    0xe0701000, 0x800c021e, // buffer_store_dword v2, v30, s[48:51], 0 offen
    0xe0701000, 0x160c061e, // buffer_store_dword v6, v30, s[48:51], s22 offen
    0xb0160800, // s_movk_i32 s22, 0x800
    0xe0701000, 0x160c071e, // buffer_store_dword v7, v30, s[48:51], s22 offen
    0xb0160c00, // s_movk_i32 s22, 0xc00
    0xe0701000, 0x160c051e, // buffer_store_dword v5, v30, s[48:51], s22 offen
    0xb0161000, // s_movk_i32 s22, 0x1000
    0xe0701000, 0x160c0b1e, // buffer_store_dword v11, v30, s[48:51], s22 offen
    0x7e180210, // v_mov_b32_e32 v12, s16
    0x7e1a0211, // v_mov_b32_e32 v13, s17
    0xb0161400, // s_movk_i32 s22, 0x1400
    0xe0701000, 0x160c0c1e, // buffer_store_dword v12, v30, s[48:51], s22 offen
    0xb0161800, // s_movk_i32 s22, 0x1800
    0xe0701000, 0x160c0d1e, // buffer_store_dword v13, v30, s[48:51], s22 offen
    0xbf810000, // s_endpgm
];

/// 256 threads: a lane read inside an `s_cbranch_execz` region the first
/// wave skips. The exchange would sit under control flow that differs
/// between the waves, so the program runs paired.
const LANE_READ_UNDER_A_WAVE_DEPENDENT_BRANCH: &[u32] = &[
    0x343c0082, // v_lshlrev_b32_e32 v30, 2, v0
    0x7e0a0280, // v_mov_b32_e32 v5, 0
    0x7d8200ff, 0x00000060, // v_cmp_lt_u32_e32 vcc, 0x60, v0
    0xbe94047e, // s_mov_b64 s[20:21], exec
    0x87fe6a7e, // s_and_b64 exec, exec, vcc
    0xbf880004, // s_cbranch_execz skip
    0x36080083, // v_and_b32_e32 v4, 3, v0
    0xd7600012, 0x02010b04, // v_readlane_b32 s18, v4, 5
    0x4a0a0812, // v_add_nc_u32_e32 v5, s18, v4
    0xbefe0414, // s_mov_b64 exec, s[20:21]
    0xe0701000, 0x800c051e, // buffer_store_dword v5, v30, s[48:51], 0 offen
    0xbf810000, // s_endpgm
];

/// 256 threads: a ballot inside an `s_cbranch_execz` region the first
/// wave skips. Its predicate is known before the region: it rides on the
/// exchange of the branch's own ballot, outside the region, and the
/// program runs split.
const BALLOT_UNDER_A_WAVE_DEPENDENT_BRANCH: &[u32] = &[
    0x343c0082, // v_lshlrev_b32_e32 v30, 2, v0
    0x7e0a0280, // v_mov_b32_e32 v5, 0
    0x7d8200ff, 0x00000060, // v_cmp_lt_u32_e32 vcc, 0x60, v0
    0xbe94047e, // s_mov_b64 s[20:21], exec
    0x87fe6a7e, // s_and_b64 exec, exec, vcc
    0xbf880005, // s_cbranch_execz skip
    0x36080083, // v_and_b32_e32 v4, 3, v0
    0xd4c20010, 0x00020881, // v_cmp_eq_u32_e64 s[16:17], 1, v4
    0xbe921010, // s_bcnt1_i32_b64 s18, s[16:17]
    0x7e0a0212, // v_mov_b32_e32 v5, s18
    0xbefe0414, // s_mov_b64 exec, s[20:21]
    0xe0701000, 0x800c051e, // buffer_store_dword v5, v30, s[48:51], 0 offen
    0xbf810000, // s_endpgm
];

/// The benchmark kernel: 256 iterations of mul / exp / fma, then an
/// if/else on the wave half (lanes 0..31 then, 32..63 else) through
/// saveexec with `s_cbranch_execz` around each arm (exp, sqrt / log, rsq);
/// one dword per thread at `(group * 64 + lane) * 4` of s[0:3]; the group
/// id in s4.
const BENCH_BRANCHES: &[u32] = &[
    0x8f058804, // s_lshl_b32 s5, s4, 8
    0x343c0082, // v_lshlrev_b32_e32 v30, 2, v0
    0x4a3c3c05, // v_add_nc_u32_e32 v30, s5, v30
    0x7e020d00, // v_cvt_f32_u32_e32 v1, v0
    0x100202ff, 0x3c000000, // v_mul_f32_e32 v1, 0x3c000000, v1
    0x7e0402f2, // v_mov_b32_e32 v2, 1.0
    0x360800a0, // v_and_b32_e32 v4, 32, v0
    0xd4c20014, 0x00020880, // v_cmp_eq_u32_e64 s[20:21], 0, v4
    0xb0100100, // s_movk_i32 s16, 0x100
    0x10060302, // loop: v_mul_f32_e32 v3, v2, v1
    0x7e064b03, // v_exp_f32_e32 v3, v3
    0xd54b0002, 0x0409e103, // v_fma_f32 v2, v3, 0.5, v2
    0xbe922414, // s_and_saveexec_b64 s[18:19], s[20:21]
    0xbf880007, // s_cbranch_execz else1
    0x100a0703, // v_mul_f32_e32 v5, v3, v3
    0x7e0a4b05, // v_exp_f32_e32 v5, v5
    0x10040b02, // v_mul_f32_e32 v2, v2, v5
    0x100404ff, 0x3f7fff00, // v_mul_f32_e32 v2, 0x3f7fff00, v2
    0x7e0c6702, // v_sqrt_f32_e32 v6, v2
    0x06040d02, // v_add_f32_e32 v2, v2, v6
    0x89fe127e, // else1: s_xor_b64 exec, exec, s[18:19]
    0xbf880007, // s_cbranch_execz merge1
    0x100a0303, // v_mul_f32_e32 v5, v3, v1
    0x7e0a4f05, // v_log_f32_e32 v5, v5
    0x06040b02, // v_add_f32_e32 v2, v2, v5
    0x100404ff, 0x3f7fff00, // v_mul_f32_e32 v2, 0x3f7fff00, v2
    0x7e0c5d02, // v_rsq_f32_e32 v6, v2
    0x10040d02, // v_mul_f32_e32 v2, v2, v6
    0x88fe127e, // merge1: s_or_b64 exec, exec, s[18:19]
    0x80908110, // s_sub_u32 s16, s16, 1
    0xbf078010, // s_cmp_lg_u32 s16, 0
    0xbf85ffe6, // s_cbranch_scc1 loop
    0xe0701000, 0x8000021e, // buffer_store_dword v2, v30, s[0:3], 0 offen
    0xbf810000, // s_endpgm
];

/// [`BENCH_BRANCHES`] without the `s_cbranch_execz`: both arms run under
/// their EXEC, no scalar code looks at a mask.
const BENCH_MASKED: &[u32] = &[
    0x8f058804, // s_lshl_b32 s5, s4, 8
    0x343c0082, // v_lshlrev_b32_e32 v30, 2, v0
    0x4a3c3c05, // v_add_nc_u32_e32 v30, s5, v30
    0x7e020d00, // v_cvt_f32_u32_e32 v1, v0
    0x100202ff, 0x3c000000, // v_mul_f32_e32 v1, 0x3c000000, v1
    0x7e0402f2, // v_mov_b32_e32 v2, 1.0
    0x360800a0, // v_and_b32_e32 v4, 32, v0
    0xd4c20014, 0x00020880, // v_cmp_eq_u32_e64 s[20:21], 0, v4
    0xb0100100, // s_movk_i32 s16, 0x100
    0x10060302, // loop: v_mul_f32_e32 v3, v2, v1
    0x7e064b03, // v_exp_f32_e32 v3, v3
    0xd54b0002, 0x0409e103, // v_fma_f32 v2, v3, 0.5, v2
    0xbe922414, // s_and_saveexec_b64 s[18:19], s[20:21]
    0x100a0703, // v_mul_f32_e32 v5, v3, v3
    0x7e0a4b05, // v_exp_f32_e32 v5, v5
    0x10040b02, // v_mul_f32_e32 v2, v2, v5
    0x100404ff, 0x3f7fff00, // v_mul_f32_e32 v2, 0x3f7fff00, v2
    0x7e0c6702, // v_sqrt_f32_e32 v6, v2
    0x06040d02, // v_add_f32_e32 v2, v2, v6
    0x89fe127e, // s_xor_b64 exec, exec, s[18:19]
    0x100a0303, // v_mul_f32_e32 v5, v3, v1
    0x7e0a4f05, // v_log_f32_e32 v5, v5
    0x06040b02, // v_add_f32_e32 v2, v2, v5
    0x100404ff, 0x3f7fff00, // v_mul_f32_e32 v2, 0x3f7fff00, v2
    0x7e0c5d02, // v_rsq_f32_e32 v6, v2
    0x10040d02, // v_mul_f32_e32 v2, v2, v6
    0x88fe127e, // s_or_b64 exec, exec, s[18:19]
    0x80908110, // s_sub_u32 s16, s16, 1
    0xbf078010, // s_cmp_lg_u32 s16, 0
    0xbf85ffe8, // s_cbranch_scc1 loop
    0xe0701000, 0x8000021e, // buffer_store_dword v2, v30, s[0:3], 0 offen
    0xbf810000, // s_endpgm
];

/// [`BENCH_BRANCHES`] with the arm condition a compare of each iteration's
/// value (`v_fract` < 0.5): every `s_cbranch_execz` reads a fresh ballot.
const BENCH_BALLOT_LOOP: &[u32] = &[
    0x8f058804, // s_lshl_b32 s5, s4, 8
    0x343c0082, // v_lshlrev_b32_e32 v30, 2, v0
    0x4a3c3c05, // v_add_nc_u32_e32 v30, s5, v30
    0x7e020d00, // v_cvt_f32_u32_e32 v1, v0
    0x100202ff, 0x3c000000, // v_mul_f32_e32 v1, 0x3c000000, v1
    0x7e0402f2, // v_mov_b32_e32 v2, 1.0
    0x360800a0, // v_and_b32_e32 v4, 32, v0
    0xd4c20014, 0x00020880, // v_cmp_eq_u32_e64 s[20:21], 0, v4
    0xb0100100, // s_movk_i32 s16, 0x100
    0x10060302, // v_mul_f32_e32 v3, v2, v1
    0x7e064b03, // v_exp_f32_e32 v3, v3
    0xd54b0002, 0x0409e103, // v_fma_f32 v2, v3, 0.5, v2
    0x7e0e4102, // v_fract_f32_e32 v7, v2
    0xd4040014, 0x00020ef0, // v_cmp_gt_f32_e64 s[20:21], 0.5, v7
    0xbe922414, // s_and_saveexec_b64 s[18:19], s[20:21]
    0xbf880007, // s_cbranch_execz else1
    0x100a0703, // v_mul_f32_e32 v5, v3, v3
    0x7e0a4b05, // v_exp_f32_e32 v5, v5
    0x10040b02, // v_mul_f32_e32 v2, v2, v5
    0x100404ff, 0x3f7fff00, // v_mul_f32_e32 v2, 0x3f7fff00, v2
    0x7e0c6702, // v_sqrt_f32_e32 v6, v2
    0x06040d02, // v_add_f32_e32 v2, v2, v6
    0x89fe127e, // s_xor_b64 exec, exec, s[18:19]
    0xbf880007, // s_cbranch_execz merge1
    0x100a0303, // v_mul_f32_e32 v5, v3, v1
    0x7e0a4f05, // v_log_f32_e32 v5, v5
    0x06040b02, // v_add_f32_e32 v2, v2, v5
    0x100404ff, 0x3f7fff00, // v_mul_f32_e32 v2, 0x3f7fff00, v2
    0x7e0c5d02, // v_rsq_f32_e32 v6, v2
    0x10040d02, // v_mul_f32_e32 v2, v2, v6
    0x88fe127e, // s_or_b64 exec, exec, s[18:19]
    0x80908110, // s_sub_u32 s16, s16, 1
    0xbf078010, // s_cmp_lg_u32 s16, 0
    0xbf85ffe3, // s_cbranch_scc1 loop
    0xe0701000, 0x8000021e, // buffer_store_dword v2, v30, s[0:3], 0 offen
    0xbf810000, // s_endpgm
];

/// Words of the generated benchmark kernels (gfx1013 encodings).
mod enc {
    pub const LIT: u32 = 0xFF;
    pub fn v(n: u32) -> u32 {
        256 + n
    }
    /// VOP2 `op vdst, src0, vsrc1`; a literal source follows.
    pub fn vop2(out: &mut Vec<u32>, op: u32, vdst: u32, src0: u32, vsrc1: u32, literal: Option<u32>) {
        out.push(op << 25 | vdst << 17 | vsrc1 << 9 | src0);
        out.extend(literal);
    }
    /// VOP1 `op vdst, vsrc`.
    pub fn vop1(out: &mut Vec<u32>, op: u32, vdst: u32, vsrc: u32) {
        out.push(0x7E00_0000 | vdst << 17 | op << 9 | v(vsrc));
    }
    /// `v_fma_f32 vdst, src0, src1, src2` with a literal.
    pub fn fma(out: &mut Vec<u32>, vdst: u32, src0: u32, src1: u32, src2: u32, literal: u32) {
        out.extend([0xD54B_0000 | vdst, src0 | src1 << 9 | src2 << 18, literal]);
    }
    pub const ADD: u32 = 3;
    pub const MUL: u32 = 8;
    pub const EXP: u32 = 0x25;
    pub const LOG: u32 = 0x27;
    pub const RSQ: u32 = 0x2E;
    pub const SQRT: u32 = 0x33;
    /// `s_cbranch_*` / `s_branch` word with a dword offset.
    pub fn branch(opcode: u32, offset: i32) -> u32 {
        0xBF80_0000 | opcode << 16 | (offset as u32 & 0xFFFF)
    }
    /// The common start: v30 / v250 = `(group * 64 + lane) * 4`, v1 =
    /// `lane * scale`.
    pub fn prologue(out: &mut Vec<u32>, address: u32, scale: u32) {
        out.push(0x8f058804); // s_lshl_b32 s5, s4, 8
        out.push(0x3400_0082 | address << 17); // v_lshlrev_b32 vA, 2, v0
        out.push(0x4a00_0005 | address << 17 | address << 9); // v_add_nc_u32 vA, s5, vA
        out.push(0x7e020d00); // v_cvt_f32_u32 v1, v0
        vop2(out, MUL, 1, LIT, 1, Some(scale));
    }
    /// v2 = the sum of `acc`, stored at vA through s[0:3]; s_endpgm.
    pub fn epilogue(out: &mut Vec<u32>, address: u32, acc: &[u32]) {
        out.push(0x7e040200 | v(acc[0])); // v_mov_b32 v2, vN
        for &a in &acc[1..] {
            vop2(out, ADD, 2, v(2), a, None);
        }
        out.extend([0xe0701000, 0x80000200 | address, 0xbf810000]); // buffer_store_dword v2, vA, s[0:3], 0 offen; s_endpgm
    }
}

/// `n` accumulators per lane live across a 64-iteration loop, each
/// updated from itself and its neighbour (fma, mul by 0.5): a paired
/// invocation needs twice the registers. (The generated words were checked
/// against llvm-mc's.)
fn bench_registers(n: u32) -> Vec<u32> {
    use enc::*;
    let acc: Vec<u32> = (10..10 + n).collect();
    let mut w = Vec::new();
    prologue(&mut w, 250, 0x3a800000);
    for (k, &a) in acc.iter().enumerate() {
        vop2(&mut w, ADD, a, [0xF0, 0xF2, 0xF4, 0xF6][k % 4], 1, None);
    }
    w.push(0xb0100040); // s_movk_i32 s16, 0x40
    for (k, &a) in acc.iter().enumerate() {
        fma(&mut w, a, v(a), LIT, v(acc[(k + 1) % acc.len()]), 0x3f7ff000);
        vop2(&mut w, MUL, a, 0xF0, a, None);
    }
    w.extend([0x80908110, 0xbf078010, branch(5, -(4 * n as i32 + 3))]); // s_sub_u32 s16, s16, 1; s_cmp_lg_u32 s16, 0; s_cbranch_scc1
    epilogue(&mut w, 250, &acc);
    w
}

/// `n` independent accumulators per lane, each updated in the arm its
/// half takes (fma, exp, sqrt in lanes 0..31; mul, log, rsq in 32..63),
/// 256 iterations, `s_cbranch_execz` around each arm.
fn bench_arms(n: u32) -> Vec<u32> {
    use enc::*;
    let acc: Vec<u32> = (10..10 + n).collect();
    let mut w = Vec::new();
    prologue(&mut w, 30, 0x3c000000);
    for (k, &a) in acc.iter().enumerate() {
        vop2(&mut w, ADD, a, if k % 2 == 1 { 0xF0 } else { 0xF2 }, 1, None);
    }
    // v4 = lane & 32; s[20:21] = lanes of the low half; s16 = 256.
    w.extend([0x360800a0, 0xd4c20014, 0x00020880, 0xb0100100]);
    w.extend([0xbe922414, branch(8, 9 * n as i32)]); // s_and_saveexec_b64 s[18:19], s[20:21]; s_cbranch_execz
    for &a in &acc {
        fma(&mut w, a, v(a), LIT, v(1), 0x3f7ff000);
        vop1(&mut w, EXP, 60, a);
        fma(&mut w, a, v(60), LIT, v(a), 0x3a800000);
        vop2(&mut w, MUL, a, v(a), a, None);
        vop1(&mut w, SQRT, a, a);
    }
    w.extend([0x89fe127e, branch(8, 8 * n as i32)]); // s_xor_b64 exec, exec, s[18:19]; s_cbranch_execz
    for &a in &acc {
        vop2(&mut w, MUL, a, LIT, a, Some(0x3f7ff000));
        vop1(&mut w, LOG, 61, a);
        fma(&mut w, a, v(61), LIT, v(a), 0x3a800000);
        vop2(&mut w, MUL, a, v(a), a, None);
        vop1(&mut w, RSQ, a, a);
    }
    w.extend([0x88fe127e, 0x80908110, 0xbf078010, branch(5, -(17 * n as i32 + 8))]); // s_or_b64 exec, exec, s[18:19]; loop
    epilogue(&mut w, 30, &acc);
    w
}

fn leak(words: Vec<u32>) -> &'static [u32] {
    Box::leak(words.into_boxed_slice())
}

/// How a program is compiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Split, shuffles within a half on the host subgroup.
    Split,
    /// Split, every shuffle through workgroup memory.
    SplitMemory,
    /// Two guest lanes per invocation.
    Paired,
}

const MODES: [Mode; 3] = [Mode::Split, Mode::SplitMemory, Mode::Paired];

fn rig(g: &Guest, mode: Mode) -> Rig {
    let mem = g.mem;
    Rig::with_compiler(
        g,
        |dev| {
            let mut c = SpirvCompiler::for_device(dev, mem);
            c.set_wave64_split(mode != Mode::Paired, mode != Mode::SplitMemory);
            Arc::new(c)
        },
        None,
        None,
    )
}

fn case(name: &'static str, code: &'static [u32], threads: u32, lds_dwords: u32, words: usize) -> RefCase {
    RefCase {
        name,
        code,
        initial: &[],
        expected: Box::leak(vec![0u32; words].into_boxed_slice()),
        user_data: None,
        compute: Compute {
            threads: [threads, 1, 1],
            lds_dwords,
            wave_size: 64,
            float_mode: 0xC0,
            dispatch_threads: [0; 3],
            workgroup_register: 0,
            thread_ids: 1,
            group_id: [false; 3],
            dispatch_dims: false,
            tg_size_en: false,
        },
        dispatch: [1, 1, 1],
        compile_only: false,
        tolerance: 0.0,
        bda: &[],
    }
}

/// Whether the device runs wave64 compute programs on 32-wide subgroups
/// (paired or split).
fn narrow_device() -> bool {
    let dev = GpuDevice::create(false).expect("device");
    let f = Features::of(&dev);
    f.subgroup_size == 32 && !f.subgroup64_compute
}

/// The split decision for `c` (on the interpreter's optimized function,
/// projected as the back-end does).
fn split_plan(c: &RefCase) -> Result<wave_split::SplitPlan, String> {
    let mut f = c.compile(true)?;
    ps5shader::opt::wave64_masks::project_wave64_lane_masks(&mut f, Some(ps5shader::ssa::SystemValue::HelperInvocation));
    ps5shader::opt::fold::cse(&mut f);
    ps5shader::opt::fold::fold(&mut f);
    ps5shader::opt::dce::dce(&mut f);
    let t = c.compute.threads;
    let features = Features { subgroup_size: 32, max_shared_bytes: 48 << 10, ..Features::default() };
    wave_split::plan(&f, [t[0].max(1), t[1].max(1), t[2].max(1)], c.compute.lds_dwords, &features, true)
}

/// Run `c` in every mode; returns the interpreter's words.
fn check(c: &RefCase, split: bool) -> Vec<u32> {
    let _ = env_logger::builder().is_test(true).try_init();
    let plan = split_plan(c);
    assert_eq!(plan.is_ok(), split, "{}: split decision {plan:?}", c.name);
    let want = c.execute(true, c.expected.len()).unwrap();
    let narrow = narrow_device();
    let mut fails = Vec::new();
    for mode in MODES {
        if mode != Mode::Paired && !narrow {
            continue;
        }
        let mut g = Guest::new(64 << 20);
        let mut rig = rig(&g, mode);
        match run_case(&mut g, &mut rig, c) {
            Ok(diff) => fails.extend(diff.into_iter().take(16).map(|d| format!("[{mode:?}] {d}"))),
            Err(e) => fails.push(format!("[{mode:?}] {}: {e}", c.name)),
        }
    }
    assert!(fails.is_empty(), "{}:\n{}", c.name, fails.join("\n"));
    want
}

/// Word `lane` of output region `region` (64 dwords each).
fn at(words: &[u32], region: usize, lane: usize) -> u32 {
    words[region * 64 + lane]
}

fn mask(f: impl Fn(u32) -> bool, present: u32) -> u64 {
    (0..present).filter(|&l| f(l)).fold(0u64, |m, l| m | 1 << l)
}

#[test]
#[ignore]
fn wave64_split_ballots_and_lane_reads() {
    for threads in [64, 40] {
        let c = case(if threads == 64 { "Wave64SplitBallotsAndLaneReads" } else { "Wave64SplitBallotsAndLaneReadsPartialWave" }, BALLOTS_AND_LANE_READS, threads, 0, 640);
        let w = check(&c, true);
        // The interpreter's words are what the device must match; check
        // that they exercise the cases.
        let m = mask(|l| l > 20 && l & 5 != 0, threads);
        assert_eq!((at(&w, 0, 1), at(&w, 1, 1)), (m as u32, (m >> 32) as u32), "{}", c.name);
        if threads == 64 {
            assert_eq!(at(&w, 2, 0), 50 * 7 + 1, "v_readlane lane 50");
        }
        assert_eq!(at(&w, 3, 0), 33 * 7 + 1, "v_readlane lane s21 = 33");
        assert_eq!(at(&w, 4, 0), if threads == 64 { 44 * 7 + 1 } else { 0 }, "v_readfirstlane with EXEC 0x00f0f000 in the high word");
        assert_eq!(at(&w, 5, 0), 37, "s_bcnt1 of lanes below 37");
        // Lane 41 (active) and 40 (inactive) of a register written under
        // EXEC; past a partial wave lane 41 does not exist.
        let lane41 = if threads > 41 { 41 * 7 + 1 } else { 0 };
        assert_eq!((at(&w, 6, 0), at(&w, 7, 0)), (lane41, 0), "v_readlane of a register written under EXEC");
        let exec = mask(|_| true, threads);
        assert_eq!((at(&w, 8, 0), at(&w, 9, 0)), (exec as u32, (exec >> 32) as u32), "saved EXEC");
    }
}

#[test]
#[ignore]
fn wave64_split_lane_stack_and_spills() {
    for threads in [64, 40] {
        let c = case(if threads == 64 { "Wave64SplitLaneStackAndSpills" } else { "Wave64SplitLaneStackAndSpillsPartialWave" }, LANE_STACK_AND_SPILLS, threads, 0, 512);
        // The lane-indexed registers are lane slots: no lane read is an
        // exchange.
        let plan = split_plan(&c).unwrap();
        assert!(plan.counts[5] > 0 && plan.counts[1] == 0, "{}: lane reads {:?}", c.name, plan.counts);
        let w = check(&c, true);
        assert_eq!(at(&w, 0, 0), 0x119585c7, "hash of the popped lanes");
        assert_eq!((at(&w, 1, 0), at(&w, 2, 0), at(&w, 3, 0)), (0x1234 + 3, 0x5678, 38), "spilled lanes 5, 40 and 33");
        for l in 0..threads as usize {
            let v23 = if l < 45 { 3 * l as u32 + 1 } else { 0x55 };
            assert_eq!(at(&w, 4, l), v23, "v23 lane {l}");
            assert_eq!(at(&w, 6, l), v23 + 7 * l as u32, "v23 + 7 * lane, lane {l}");
            let v12 = match l {
                5 => 0x1234 + 3,
                33 => 38,
                40 => 0x5678,
                _ if l < 16 => 3,
                _ => 0,
            };
            assert_eq!(at(&w, 5, l), v12, "v12 lane {l}");
            assert_eq!(at(&w, 7, l), if l < 16 { 18 + 35 * l as u32 } else { 0 }, "writes under EXEC 0xffff, lane {l}");
        }
    }
}

#[test]
#[ignore]
fn wave64_split_first_lane_reads() {
    for threads in [64, 40] {
        let c = case(if threads == 64 { "Wave64SplitFirstLaneReads" } else { "Wave64SplitFirstLaneReadsPartialWave" }, FIRST_LANE_READS, threads, 0, 448);
        // The reads ride on their ballots' exchanges, the later ballots on
        // the first one's.
        let plan = split_plan(&c).unwrap();
        assert!(plan.counts[6] >= 4 && plan.counts[1] == 0 && plan.counts[7] >= 2, "{}: exchanges {:?}", c.name, plan.counts);
        let w = check(&c, true);
        if threads == 64 {
            assert_eq!((at(&w, 0, 0), at(&w, 1, 0)), (41 * 7, 41 * 7), "first lane above 40");
            assert_eq!(at(&w, 2, 0), 63 * 7, "empty mask: lane 63");
            assert_eq!(at(&w, 4, 0), 45 * 7, "mask under EXEC 0xff00 << 32");
        }
        assert_eq!(at(&w, 3, 0), 9 * 7, "first lane 9");
    }
}

#[test]
#[ignore]
fn wave64_split_lane_moves_at_the_half_boundary() {
    let c = case("Wave64SplitLaneMovesAtTheHalfBoundary", LANE_MOVES_AT_THE_HALF_BOUNDARY, 64, 0, 640);
    let w = check(&c, true);
    let v3 = |l: usize| 100 + 3 * l as u32;
    // Lanes 28..35 are inactive: never written. row_shr:1: lane 36 reads
    // inactive lane 35 and keeps its value, lane 37 reads 36; lane 48
    // starts its row.
    assert_eq!((at(&w, 0, 33), at(&w, 0, 36), at(&w, 0, 37), at(&w, 0, 48)), (0, 7, v3(36), 7));
    // quad_perm [3,2,1,0]: lane 36 reads 39.
    assert_eq!((at(&w, 2, 36), at(&w, 2, 30)), (v3(39), 0));
    // ds_bpermute: lane 40 reads lane 32 + (40 * 5) % 32 = 40.
    assert_eq!(at(&w, 7, 40), v3(32 + (40 * 5) % 32));
}

#[test]
#[ignore]
fn wave64_split_divergent_loop_and_if_else() {
    let c = case("Wave64SplitDivergentLoopAndIfElse", DIVERGENT_LOOP_AND_IF_ELSE, 64, 0, 192);
    let w = check(&c, true);
    for l in [0usize, 5, 31, 32, 39, 63] {
        let n = (l as u32 & 7) + if l >= 32 { 3 } else { 0 };
        assert_eq!(at(&w, 0, l), n * (l as u32 + 3), "loop sum of lane {l}");
    }
    // The loop runs as long as its longest lane (7 + 3).
    assert_eq!(at(&w, 2, 0), 10);
}

#[test]
#[ignore]
fn wave64_split_waves_barrier_and_lds() {
    let c = case("Wave64SplitWavesBarrierAndLds", WAVES_BARRIER_AND_LDS, 256, 512, 1792);
    let plan = split_plan(&c).unwrap();
    assert_eq!(plan.waves, 4);
    let w = check(&c, true);
    let r = |region: usize, lane: usize| w[region * 256 + lane];
    for l in [0usize, 63, 64, 200, 255] {
        assert_eq!(r(0, l), ((l as u32 + 64) % 256) * 3 + 5, "LDS of the next wave, lane {l}");
        assert_eq!(r(4, l), (l as u32 ^ 32) * 11, "LDS of the other half, lane {l}");
        assert_eq!(r(1, l), 16, "lanes with lane % 4 == 0 per wave");
        assert_eq!(r(2, l), (l as u32 & !63 | 45) * 3 + 5, "v_readlane 45 of its wave");
        assert_eq!(r(3, l), if l > 0x60 { 2 } else { 0 }, "written under EXEC, lane {l}");
    }
}

#[test]
#[ignore]
fn wave64_split_ballot_under_a_wave_dependent_branch_rides_outside() {
    let c = case("Wave64SplitBallotUnderAWaveDependentBranch", BALLOT_UNDER_A_WAVE_DEPENDENT_BRANCH, 256, 0, 256);
    let w = check(&c, true);
    assert_eq!((w[0], w[100], w[200]), (0, 8, 16));
}

#[test]
#[ignore]
fn wave64_split_falls_back_for_a_lane_read_under_a_wave_dependent_branch() {
    let c = case("Wave64SplitLaneReadUnderAWaveDependentBranch", LANE_READ_UNDER_A_WAVE_DEPENDENT_BRANCH, 256, 0, 256);
    check(&c, false);
}

// ----------------------------------------------------------------------
// Benchmark
// ----------------------------------------------------------------------

/// GPU time (ns) of `n` dispatches of `groups` work-groups of `program`.
fn bench(mode: Mode, program: &'static [u32], groups: u32, n: u32) -> (f64, Vec<u32>) {
    let mut g = Guest::new(64 << 20);
    let mut rig = rig(&g, mode);
    let code = g.program(program);
    let bytes = groups as usize * 256;
    let out = g.zeroed(bytes);
    let mut s = stage(code, &[out as u32, (out >> 32) as u32 & 0xFFFF, bytes as u32, (3 << 28) | (20 << 12) | 0xFAC]);
    s.rsrc1 = (0xC0 << 12) | (1 << 21);
    s.rsrc2 = (4 << 1) | (1 << 7);
    let d = ResolvedDispatch {
        packet: DispatchPacket { groups: [groups, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() },
        shader: s,
        rsrc3: 0,
        threads: [64, 1, 1],
        start: [0; 3],
    };
    // Compile and warm up.
    assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
    rig.sched.flush_and_wait();
    let dev = rig.dev.clone();
    let info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2);
    // SAFETY: valid create info.
    let pool = unsafe { dev.device.create_query_pool(&info, None) }.unwrap();
    let mut best = f64::MAX;
    for _ in 0..5 {
        let cb = rig.sched.cmd();
        // SAFETY: recording buffer; the pool lives through the waits.
        unsafe {
            dev.device.cmd_reset_query_pool(cb, pool, 0, 2);
            dev.device.cmd_write_timestamp2(cb, vk::PipelineStageFlags2::ALL_COMMANDS, pool, 0);
        }
        for _ in 0..n {
            assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
        }
        let cb = rig.sched.cmd();
        // SAFETY: as above.
        unsafe { dev.device.cmd_write_timestamp2(cb, vk::PipelineStageFlags2::ALL_COMMANDS, pool, 1) };
        rig.sched.flush_and_wait();
        let mut ts = [0u64; 2];
        // SAFETY: both queries were written and the work completed.
        unsafe { dev.device.get_query_pool_results(pool, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT) }.unwrap();
        best = best.min((ts[1] - ts[0]) as f64 * dev.caps.timestamp_period as f64);
    }
    // SAFETY: no pending use.
    unsafe { dev.device.destroy_query_pool(pool, None) };
    assert!(rig.buffers.download(&mut rig.sched, out, bytes as u64));
    let mut back = vec![0u8; bytes];
    assert!(g.mem.read_backing(out, &mut back));
    (best, back.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// GPU time of the benchmark kernels split and paired (printed; the split
/// and paired results must agree).
/// `RPS5E_WAVE64_BENCH=1 cargo test -p ps5gpu --release -- --ignored --nocapture wave64_split_benchmark`
#[test]
#[ignore]
fn wave64_split_benchmark() {
    // A benchmark: it loads the GPU and would disturb the timing tests that
    // run beside it, so it runs only when asked for.
    if std::env::var_os("RPS5E_WAVE64_BENCH").is_none() {
        return;
    }
    let _ = env_logger::builder().is_test(true).try_init();
    if !narrow_device() {
        eprintln!("wave64_split_benchmark: the device runs wave64 on 64-wide subgroups");
        return;
    }
    let groups = 16384;
    for (name, program) in [("branches", BENCH_BRANCHES), ("ballot_loop", BENCH_BALLOT_LOOP), ("masked", BENCH_MASKED), ("arms", leak(bench_arms(8))), ("arms32", leak(bench_arms(32))), ("regs48", leak(bench_registers(48))), ("regs160", leak(bench_registers(160)))] {
        let (paired, a) = bench(Mode::Paired, program, groups, 4);
        let (split, b) = bench(Mode::Split, program, groups, 4);
        let same = a.iter().zip(&b).filter(|(x, y)| x == y || (f32::from_bits(**x).is_nan() && f32::from_bits(**y).is_nan())).count();
        eprintln!("wave64 split benchmark {name}: {groups} groups x 4 dispatches: paired {:.3} ms, split {:.3} ms ({:.2}x); {same} of {} words alike", paired / 1e6, split / 1e6, paired / split, a.len());
        assert_eq!(same, a.len(), "{name}: split and paired results differ");
    }
}
