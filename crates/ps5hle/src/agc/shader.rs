// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceAgc shader-object exports: `sceAgcCreateShader`, fused shader
//! halves, primitive state and interpolant (SPI_PS_INPUT_CNTL) mapping.
//!
//! Shader object (0x60 bytes, pointers stored relative to their field until
//! `sceAgcCreateShader` relocates them):
//!
//! | Offset | Field |
//! |---|---|
//! | +0x00 | u32 file header `0x34333231` |
//! | +0x04 | u32 version `0x18` |
//! | +0x08 | user-data table pointer (5 relative pointers inside) |
//! | +0x10 | code address (set by CreateShader) |
//! | +0x18 | context register table `{u32 offset, u32 value}[]` |
//! | +0x20 | shader register table |
//! | +0x28 | specials (GE_CNTL +0x00, VGT_SHADER_STAGES_EN +0x08, VGT_GS_OUT_PRIM_TYPE +0x20, GE_USER_VGPR_EN +0x28) |
//! | +0x30 | input semantics |
//! | +0x38 | output semantics |
//! | +0x44 | u32 code size |
//! | +0x50 | u32 input semantic count |
//! | +0x56 | u16 output semantic count |
//! | +0x5A | u8 shader type |
//! | +0x5B | u8 context register count |
//! | +0x5C | u8 shader register count |

use super::{arg, mem, Export, ERR_INCOMPLETE_SHADER_REGISTERS, ERR_INVALID_ARGUMENT, ERR_MEMORY_FAULT, LIB_AGC, OK};

/// SCE_AGC_ERROR_INVALID_SHADER_HALVES: not a front/back pair, or halves
/// built for different wave sizes.
const ERR_INVALID_SHADER_HALVES: HleResult = 0x8A6C_0008u32 as i32 as i64;
use crate::context::CpuContext;
use crate::registry::HleResult;
use std::collections::HashMap;
use std::sync::Mutex;

const FILE_HEADER: u32 = 0x3433_3231;
const VERSION: u32 = 0x18;

const OFF_USER_DATA: u64 = 0x08;
const OFF_CODE: u64 = 0x10;
const OFF_CX_REGS: u64 = 0x18;
const OFF_SH_REGS: u64 = 0x20;
const OFF_SPECIALS: u64 = 0x28;
const OFF_INPUT_SEMANTICS: u64 = 0x30;
const OFF_OUTPUT_SEMANTICS: u64 = 0x38;
const OFF_NUM_INPUT_SEMANTICS: u64 = 0x50;
const OFF_NUM_OUTPUT_SEMANTICS: u64 = 0x56;
const OFF_TYPE: u64 = 0x5A;
const OFF_NUM_CX_REGS: u64 = 0x5B;
const OFF_NUM_SH_REGS: u64 = 0x5C;
const SHADER_STRUCT_BYTES: usize = 0x60;

const TYPE_CS: u8 = 0;
const TYPE_PS: u8 = 1;
const TYPE_GS: u8 = 2;
const TYPE_HS: u8 = 3;
const TYPE_GS_FRONT: u8 = 4;
const TYPE_HS_FRONT: u8 = 5;
const TYPE_GS_BACK: u8 = 6;
const TYPE_HS_BACK: u8 = 7;

const SPECIAL_GE_CNTL: u64 = 0x00;
const SPECIAL_VGT_SHADER_STAGES_EN: u64 = 0x08;
const SPECIAL_VGT_GS_OUT_PRIM_TYPE: u64 = 0x20;
const SPECIAL_GE_USER_VGPR_EN: u64 = 0x28;
const STAGES_HS_W32_EN: u32 = 1 << 21;
const STAGES_GS_W32_EN: u32 = 1 << 22;
const STAGES_GS_EN: u32 = 1 << 5;

const VGT_GS_OUT_PRIM_TYPE: u32 = 0x29B;
const VGT_PRIMITIVE_TYPE: u32 = 0x242;
const SPI_PS_INPUT_CNTL_0: u32 = 0x191;

// Program address registers (shader bank).
const SPI_SHADER_PGM_LO_PS: u32 = 0x08;
const SPI_SHADER_PGM_CHKSUM_GS: u32 = 0x80;
const SPI_SHADER_PGM_LO_GS: u32 = 0x88;
const SPI_SHADER_PGM_RSRC1_GS: u32 = 0x8A;
const SPI_SHADER_PGM_RSRC2_GS: u32 = 0x8B;
const SPI_SHADER_PGM_LO_ES: u32 = 0xC8;
const SPI_SHADER_PGM_CHKSUM_HS: u32 = 0x100;
const SPI_SHADER_PGM_LO_HS: u32 = 0x108;
const SPI_SHADER_PGM_RSRC1_HS: u32 = 0x10A;
const SPI_SHADER_PGM_RSRC2_HS: u32 = 0x10B;
const SPI_SHADER_PGM_LO_LS: u32 = 0x148;
const COMPUTE_PGM_LO: u32 = 0x20C;

const INTERPOLANT_COUNT: usize = 32;

/// Shader code address -> shader object, for the draw path.
static SHADERS_BY_CODE: Mutex<Option<HashMap<u64, u64>>> = Mutex::new(None);

/// Shader object created for `code`, if any.
#[allow(dead_code)]
pub fn shader_for_code(code: u64) -> Option<u64> {
    SHADERS_BY_CODE.lock().unwrap().as_ref()?.get(&code).copied()
}

fn relocate(field: u64) -> bool {
    match mem::r64(field) {
        Some(0) => true,
        Some(rel) => mem::w64(field, field.wrapping_add(rel)),
        None => false,
    }
}

/// The register holding a shader type's program address (front halves and
/// function shaders have none).
fn program_register(ty: u8) -> Option<u32> {
    match ty {
        TYPE_CS => Some(COMPUTE_PGM_LO),
        TYPE_PS => Some(SPI_SHADER_PGM_LO_PS),
        TYPE_GS => Some(SPI_SHADER_PGM_LO_ES),
        TYPE_HS => Some(SPI_SHADER_PGM_LO_LS),
        TYPE_GS_BACK => Some(SPI_SHADER_PGM_LO_GS),
        TYPE_HS_BACK => Some(SPI_SHADER_PGM_LO_HS),
        _ => None,
    }
}

/// Relocate the program address register pair: the table holds the code's
/// offset (lo = bits 8..39, hi bits 0..7 = bits 40..47), which becomes
/// `code + offset`. The hi register must directly follow lo.
fn patch_program_registers(header: u64, code: u64) -> HleResult {
    let (Some(regs), Some(ty), Some(count)) = (mem::r64(header + OFF_SH_REGS), mem::r8(header + OFF_TYPE), mem::r8(header + OFF_NUM_SH_REGS))
    else {
        return ERR_MEMORY_FAULT;
    };
    let Some(lo_reg) = program_register(ty) else { return OK };
    if regs == 0 || count == 0 {
        return ERR_INCOMPLETE_SHADER_REGISTERS;
    }
    let Some(lo) = find_register(regs, count as usize, lo_reg, 0) else { return ERR_INCOMPLETE_SHADER_REGISTERS };
    let hi = lo + 8;
    if hi >= regs + count as u64 * 8 || mem::r32(hi) != Some(lo_reg + 1) {
        return ERR_INCOMPLETE_SHADER_REGISTERS;
    }
    let (Some(lv), Some(hv)) = (mem::r32(lo + 4), mem::r32(hi + 4)) else { return ERR_MEMORY_FAULT };
    let addr = code.wrapping_add((lv as u64) << 8 | ((hv & 0xFF) as u64) << 40);
    if mem::w32(lo + 4, (addr >> 8) as u32) && mem::w32(hi + 4, (hv & 0xFFFF_FF00) | ((addr >> 40) & 0xFF) as u32) {
        OK
    } else {
        ERR_MEMORY_FAULT
    }
}

/// `sceAgcCreateShader(Shader** out, header, code)`.
fn create_shader(ctx: &mut CpuContext) -> HleResult {
    let (out, header, code) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2));
    if out == 0 || header == 0 || code == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let fields = [OFF_CX_REGS, OFF_SH_REGS, OFF_USER_DATA, OFF_SPECIALS, OFF_INPUT_SEMANTICS, OFF_OUTPUT_SEMANTICS];
    if !fields.iter().all(|&f| relocate(header + f)) || !mem::w64(header + OFF_CODE, code) {
        return ERR_MEMORY_FAULT;
    }
    let Some(ud) = mem::r64(header + OFF_USER_DATA) else { return ERR_MEMORY_FAULT };
    if ud != 0 && !(0..5).all(|i| relocate(ud + i * 8)) {
        return ERR_MEMORY_FAULT;
    }
    let (Some(fh), Some(ver)) = (mem::r32(header), mem::r32(header + 4)) else { return ERR_MEMORY_FAULT };
    if fh != FILE_HEADER || ver != VERSION {
        log::warn!("[agc] CreateShader: bad header 0x{fh:08x} version 0x{ver:x}");
        return ERR_INVALID_ARGUMENT;
    }
    if code & 0xFFFF_0000_0000_00FF != 0 {
        log::warn!("[agc] CreateShader: code 0x{code:x} is not a 256-byte aligned 48-bit address");
    }
    let r = patch_program_registers(header, code);
    if r != OK {
        return r;
    }
    if !mem::w64(out, header) {
        return ERR_MEMORY_FAULT;
    }
    SHADERS_BY_CODE.lock().unwrap().get_or_insert_with(HashMap::new).insert(code, header);
    super::shader_dump::on_create_shader(header);
    OK
}

fn is_fused_pair(front: u8, back: u8) -> bool {
    (front == TYPE_GS_FRONT && back == TYPE_GS_BACK) || (front == TYPE_HS_FRONT && back == TYPE_HS_BACK)
}

/// `GetFusedShaderSize(SizeAlign* out, front, back)`: room for the fused
/// object's register table (a copy of the back half's).
fn get_fused_shader_size(ctx: &mut CpuContext) -> HleResult {
    let (out, front, back) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2));
    if out == 0 || front == 0 || back == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let (Some(ft), Some(bt), Some(n)) = (mem::r8(front + OFF_TYPE), mem::r8(back + OFF_TYPE), mem::r8(back + OFF_NUM_SH_REGS)) else {
        return ERR_MEMORY_FAULT;
    };
    if !is_fused_pair(ft, bt) {
        return ERR_INVALID_SHADER_HALVES;
    }
    if mem::w64(out, n as u64 * 8) && mem::w64(out + 8, 4) {
        OK
    } else {
        ERR_MEMORY_FAULT
    }
}

fn find_register(table: u64, count: usize, offset: u32, mut occurrence: usize) -> Option<u64> {
    if table == 0 {
        return None;
    }
    for i in 0..count {
        let e = table + i as u64 * 8;
        if mem::r32(e)? == offset {
            if occurrence == 0 {
                return Some(e);
            }
            occurrence -= 1;
        }
    }
    None
}

fn merge_max(dst: u32, src: u32, shift: u32, mask: u32) -> u32 {
    (dst & !(mask << shift)) | (((dst >> shift) & mask).max((src >> shift) & mask) << shift)
}

/// Merge the front half's resource registers into the fused (back) ones:
/// the larger VGPR/SGPR/scratch-related fields, the shared VGPRs either
/// recomputed or the larger of the two, and the front half's user SGPRs.
fn merge_resources(front: u64, fc: usize, fused: u64, n: usize, geometry: bool, recompute_shared_vgprs: bool) -> bool {
    let (r1, r2) = if geometry {
        (SPI_SHADER_PGM_RSRC1_GS, SPI_SHADER_PGM_RSRC2_GS)
    } else {
        (SPI_SHADER_PGM_RSRC1_HS, SPI_SHADER_PGM_RSRC2_HS)
    };
    let (Some(f1e), Some(f2e), Some(u1e), Some(u2e)) = (
        find_register(front, fc, r1, 0),
        find_register(front, fc, r2, 0),
        find_register(fused, n, r1, 0),
        find_register(fused, n, r2, 0),
    ) else {
        return false;
    };
    let (Some(f1), Some(f2), Some(mut u1), Some(mut u2)) = (mem::r32(f1e + 4), mem::r32(f2e + 4), mem::r32(u1e + 4), mem::r32(u2e + 4)) else {
        return false;
    };
    if recompute_shared_vgprs {
        let front_vgprs = ((f1 & 0x3F) + 1) * 4;
        let back_vgprs = ((u1 & 0x3F) + 1) * 4;
        let front_total = front_vgprs + (f2 >> 28) * 8;
        let back_total = back_vgprs + (u2 >> 28) * 8;
        let max_total = front_total.max(back_total);
        let shared = if front_vgprs.max(back_vgprs) >= max_total { 0 } else { (max_total - front_total.min(back_total) + 7) / 64 };
        u2 = (u2 & 0x0FFF_FFFF) | ((shared & 0xF) << 28);
    } else {
        u2 = merge_max(u2, f2, 28, 0xF);
    }
    u1 = merge_max(u1, f1, 0, 0x3F);
    if geometry {
        u1 = merge_max(u1, f1, 29, 3);
        u2 = merge_max(u2, f2, 16, 3);
        u2 = (u2 & !0x0004_0000) | (f2 & 0x0004_0000);
    } else {
        u1 = merge_max(u1, f1, 28, 3);
    }
    u2 = (u2 & 0xF7FF_FFC1) | (f2 & 0x0800_003E);
    mem::w32(u1e + 4, u1) && mem::w32(u2e + 4, u2)
}

fn patch_fused_program(regs: u64, n: usize, lo_reg: u32, code: u64) -> bool {
    let Some(le) = find_register(regs, n, lo_reg, 0) else { return true };
    let he = le + 8;
    if he >= regs + n as u64 * 8 || mem::r32(he) != Some(lo_reg + 1) {
        return true;
    }
    let Some(hv) = mem::r32(he + 4) else { return false };
    mem::w32(le + 4, (code >> 8) as u32) && mem::w32(he + 4, (hv & 0xFFFF_FF00) | ((code >> 40) & 0xFF) as u32)
}

/// `FuseShaderHalves(fused, front, back, scratchRegs)`: the fused object is
/// the back half with its type promoted, its registers (copied to the
/// scratch memory when given) carrying the front half's checksum, merged
/// resources and program address (LO_ES / LO_LS).
fn fuse_shader_halves_core(ctx: &CpuContext, recompute_shared_vgprs: bool) -> HleResult {
    let (fused, front, back, scratch) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2), arg(ctx, 3));
    if fused == 0 || front == 0 || back == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let (Some(ft), Some(bt)) = (mem::r8(front + OFF_TYPE), mem::r8(back + OFF_TYPE)) else { return ERR_MEMORY_FAULT };
    if !is_fused_pair(ft, bt) {
        return ERR_INVALID_SHADER_HALVES;
    }
    let geometry = ft == TYPE_GS_FRONT;
    let mut hdr = [0u8; SHADER_STRUCT_BYTES];
    if !mem::read(back, &mut hdr) || !mem::write(fused, &hdr) || !mem::w8(fused + OFF_TYPE, if geometry { TYPE_GS } else { TYPE_HS }) {
        return ERR_MEMORY_FAULT;
    }
    let (Some(fs), Some(bs)) = (mem::r64(front + OFF_SPECIALS), mem::r64(back + OFF_SPECIALS)) else { return ERR_MEMORY_FAULT };
    let (Some(front_stages), Some(back_stages)) = (mem::r32(fs + SPECIAL_VGT_SHADER_STAGES_EN + 4), mem::r32(bs + SPECIAL_VGT_SHADER_STAGES_EN + 4)) else {
        return ERR_MEMORY_FAULT;
    };
    let bit = if geometry { STAGES_GS_W32_EN } else { STAGES_HS_W32_EN };
    if (front_stages ^ back_stages) & bit != 0 {
        return ERR_INVALID_SHADER_HALVES;
    }
    let (Some(back_regs), Some(n), Some(front_code)) = (mem::r64(back + OFF_SH_REGS), mem::r8(back + OFF_NUM_SH_REGS), mem::r64(front + OFF_CODE))
    else {
        return ERR_MEMORY_FAULT;
    };
    let n = n as usize;
    let mut fused_regs = back_regs;
    if scratch != 0 {
        if !mem::copy(scratch, back_regs, n as u64 * 8) || !mem::w64(fused + OFF_SH_REGS, scratch) {
            return ERR_MEMORY_FAULT;
        }
        fused_regs = scratch;
    }
    let (Some(front_regs), Some(fc)) = (mem::r64(front + OFF_SH_REGS), mem::r8(front + OFF_NUM_SH_REGS)) else { return ERR_MEMORY_FAULT };
    let fc = fc as usize;
    let chk = if geometry { SPI_SHADER_PGM_CHKSUM_GS } else { SPI_SHADER_PGM_CHKSUM_HS };
    for occ in 0..2 {
        let (Some(ue), Some(fe)) = (find_register(fused_regs, n, chk, occ), find_register(front_regs, fc, chk, occ)) else { continue };
        match mem::r32(fe + 4) {
            Some(v) if mem::w32(ue + 4, v) => {}
            _ => return ERR_MEMORY_FAULT,
        }
    }
    if !merge_resources(front_regs, fc, fused_regs, n, geometry, recompute_shared_vgprs) {
        return ERR_INVALID_SHADER_HALVES;
    }
    let lo_reg = if geometry { SPI_SHADER_PGM_LO_ES } else { SPI_SHADER_PGM_LO_LS };
    if !patch_fused_program(fused_regs, n, lo_reg, front_code) {
        return ERR_MEMORY_FAULT;
    }
    let user_data = if recompute_shared_vgprs {
        0
    } else {
        match mem::r64(front + OFF_USER_DATA) {
            Some(v) => v,
            None => return ERR_MEMORY_FAULT,
        }
    };
    if !mem::w64(fused + OFF_USER_DATA, user_data) {
        return ERR_MEMORY_FAULT;
    }
    super::shader_dump::on_fuse(front, back);
    OK
}

fn fuse_shader_halves(ctx: &mut CpuContext) -> HleResult {
    fuse_shader_halves_core(ctx, true)
}

fn fuse_shader_halves_legacy(ctx: &mut CpuContext) -> HleResult {
    fuse_shader_halves_core(ctx, false)
}

/// Draw primitive type -> VGT_GS_OUT_PRIM_TYPE (points 0, lines 1,
/// triangles 2, rect 3, legacy rect list 4).
pub(crate) fn primitive_to_gs_out(prim: u32) -> u32 {
    match prim {
        1 => 0,
        2 | 3 | 10 | 11 | 18 => 1,
        7 => 3,
        0x11 => 4,
        _ => 2,
    }
}

fn read_register(src: u64) -> Result<[u32; 2], HleResult> {
    match (mem::r32(src), mem::r32(src + 4)) {
        (Some(o), Some(v)) => Ok([o, v]),
        _ => Err(ERR_MEMORY_FAULT),
    }
}

/// The tables CreatePrimState fills: context `{VGT_SHADER_STAGES_EN,
/// VGT_GS_OUT_PRIM_TYPE}` (with `cx`) and user config `{GE_CNTL,
/// GE_USER_VGPR_EN, VGT_PRIMITIVE_TYPE}` (with `uc`) register pairs.
fn prim_state(hs: u64, gs: u64, prim: u32, cx: bool, uc: bool) -> Result<([u32; 4], [u32; 6]), HleResult> {
    if gs == 0 {
        return Err(ERR_INVALID_ARGUMENT);
    }
    let ty = mem::r8(gs + OFF_TYPE);
    let specials = mem::r64(gs + OFF_SPECIALS).unwrap_or(0);
    if ty != Some(TYPE_GS) || specials == 0 {
        return Err(ERR_INVALID_ARGUMENT);
    }
    let mut hs_specials = 0;
    if hs != 0 {
        hs_specials = mem::r64(hs + OFF_SPECIALS).unwrap_or(0);
        if mem::r8(hs + OFF_TYPE) != Some(TYPE_HS) || hs_specials == 0 {
            return Err(ERR_INVALID_ARGUMENT);
        }
    }
    let mut cx_values = [0u32; 4];
    if cx {
        let [st_off, mut st_val] = read_register(specials + SPECIAL_VGT_SHADER_STAGES_EN)?;
        let [mut out_off, mut out_val] = if st_val & STAGES_GS_EN != 0 {
            read_register(specials + SPECIAL_VGT_GS_OUT_PRIM_TYPE)?
        } else {
            [VGT_GS_OUT_PRIM_TYPE, primitive_to_gs_out(prim)]
        };
        if hs_specials != 0 {
            let Some(hv) = mem::r32(hs_specials + SPECIAL_VGT_SHADER_STAGES_EN + 4) else { return Err(ERR_MEMORY_FAULT) };
            st_val |= hv;
            if st_val & STAGES_GS_EN == 0 {
                [out_off, out_val] = read_register(hs_specials + SPECIAL_VGT_GS_OUT_PRIM_TYPE)?;
            }
        }
        cx_values = [st_off, st_val, out_off, out_val];
    }
    let mut uc_values = [0u32; 6];
    if uc {
        let vgpr_src = if hs_specials != 0 { hs_specials } else { specials } + SPECIAL_GE_USER_VGPR_EN;
        let [cntl_off, cntl_val] = read_register(specials + SPECIAL_GE_CNTL)?;
        let [vgpr_off, vgpr_val] = read_register(vgpr_src)?;
        uc_values = [cntl_off, cntl_val, vgpr_off, vgpr_val, VGT_PRIMITIVE_TYPE, prim];
    }
    Ok((cx_values, uc_values))
}

/// `sceAgcCreatePrimState(cxRegs[2], ucRegs[3], hs, gs, primType)`.
fn create_prim_state(ctx: &mut CpuContext) -> HleResult {
    let (cx, uc, hs, gs, prim) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2), arg(ctx, 3), arg(ctx, 4) as u32);
    if cx == 0 && uc == 0 {
        return OK;
    }
    let (cx_values, uc_values) = match prim_state(hs, gs, prim, cx != 0, uc != 0) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if (cx != 0 && !mem::write_dwords(cx, &cx_values)) || (uc != 0 && !mem::write_dwords(uc, &uc_values)) {
        return ERR_MEMORY_FAULT;
    }
    OK
}

/// `sceAgcUpdatePrimState(cxRegs, ucRegs, primType)`: refresh the output
/// primitive (unless a GS/tessellation stage owns it) and VGT_PRIMITIVE_TYPE
/// in the tables CreatePrimState filled.
fn update_prim_state(ctx: &mut CpuContext) -> HleResult {
    let (cx, uc, prim) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2) as u32);
    if cx != 0 {
        let Some(stages) = mem::r32(cx + 4) else { return ERR_MEMORY_FAULT };
        if stages & 0x24 == 0 {
            let Some(v) = mem::r32(cx + 12) else { return ERR_MEMORY_FAULT };
            if !mem::w32(cx + 12, (v & !7) | primitive_to_gs_out(prim)) {
                return ERR_MEMORY_FAULT;
            }
        }
    }
    if uc != 0 {
        let Some(v) = mem::r32(uc + 20) else { return ERR_MEMORY_FAULT };
        if !mem::w32(uc + 20, (v & !0x1F) | prim) {
            return ERR_MEMORY_FAULT;
        }
    }
    OK
}

/// Context registers the GS occupancy derives from.
const VGT_GS_ONCHIP_CNTL: u32 = 0x291;
const GE_NGG_SUBGRP_CNTL: u32 = 0x2D3;
const SPI_VS_OUT_CONFIG: u32 = 0x1B1;
const PA_CL_VS_OUT_CNTL: u32 = 0x207;
const GE_MAX_OUTPUT_PER_SUBGROUP: u32 = 0x1FF;
/// Registers GetGsOversubscription fills.
const UC_PARAMETER_OVERSUBSCRIPTION: u32 = 0x260;
const SPI_SHADER_PGM_RSRC4_GS: u32 = 0x81;
const FULL_PC_OVERSUBSCRIPTION: u32 = 0x7FF;
const FULL_SH_OVERSUBSCRIPTION: u32 = 0x007F_0000;

/// Vertices and exports a GS keeps in flight for the given parameter-cache
/// and export capacities.
fn gs_occupancy_limits(gs: u64, vertex_capacity: u32, export_capacity: u32) -> Option<(u32, u32)> {
    let (regs, n) = (mem::r64(gs + OFF_CX_REGS)?, mem::r8(gs + OFF_NUM_CX_REGS)? as usize);
    let reg = |o: u32| find_register(regs, n, o, 0).and_then(|e| mem::r32(e + 4));
    let (onchip, subgroup, vs_out, cl_out, max_output) =
        (reg(VGT_GS_ONCHIP_CNTL)?, reg(GE_NGG_SUBGRP_CNTL)?, reg(SPI_VS_OUT_CONFIG)?, reg(PA_CL_VS_OUT_CNTL)?, reg(GE_MAX_OUTPUT_PER_SUBGROUP)?);
    let stages = mem::r32(mem::r64(gs + OFF_SPECIALS)? + SPECIAL_VGT_SHADER_STAGES_EN + 4)?;
    let output_words = ((max_output & 0x3FF) + 31) >> 5;
    let subgroup_work = ((((onchip >> 11) & 0x7FF) * (subgroup & 0x1FF)) + 31) >> 5;
    let mut waves = subgroup_work.max(output_words);
    if stages & STAGES_GS_W32_EN == 0 {
        waves >>= 1;
    }
    let waves = waves.max(1);
    if output_words == 0 {
        return None;
    }
    let exports = 1 + ((cl_out >> 21) & 1) + ((cl_out >> 22) & 1) + ((cl_out >> 23) & 1);
    let export_limit = (export_capacity / exports) * 4;
    let vertex_limit = if vs_out & 0x80 != 0 { 2048 } else { vertex_capacity / (((vs_out >> 2) & 0xF) + 1) };
    Some((waves * (vertex_limit / output_words), waves * (export_limit / output_words)))
}

/// `sceAgcGetGsOversubscription(regs[2], gs, budget, float factor)`: the
/// parameter-cache / shader oversubscription a GS may use, as
/// UC_PARAMETER_OVERSUBSCRIPTION and SPI_SHADER_PGM_RSRC4_GS. The factor
/// (xmm0) places the target between the base and the budget-limited
/// occupancy.
fn get_gs_oversubscription(ctx: &mut CpuContext) -> HleResult {
    let factor = f32::from_bits(ps5cpu::exec::trap_xmm_low(0).unwrap_or(0) as u32);
    gs_oversubscription(arg(ctx, 0), arg(ctx, 1), arg(ctx, 2) as u32, factor)
}

pub(super) fn gs_oversubscription(regs: u64, gs: u64, budget: u32, factor: f32) -> HleResult {
    if regs == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let mut pc = 0u32;
    let mut sh = 0u32;
    let write = |pc: u32, sh: u32| {
        if mem::write_dwords(regs, &[UC_PARAMETER_OVERSUBSCRIPTION, pc, SPI_SHADER_PGM_RSRC4_GS, sh]) {
            OK
        } else {
            ERR_MEMORY_FAULT
        }
    };
    if budget == 0 {
        return write(pc, sh);
    }
    if budget == u32::MAX {
        return write(FULL_PC_OVERSUBSCRIPTION, FULL_SH_OVERSUBSCRIPTION);
    }
    if gs == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let (Some(base), Some(expanded), Some(specials)) = (gs_occupancy_limits(gs, 1024, 128), gs_occupancy_limits(gs, 2048, 382), mem::r64(gs + OFF_SPECIALS))
    else {
        return ERR_INVALID_ARGUMENT;
    };
    let Some(stages) = mem::r32(specials + SPECIAL_VGT_SHADER_STAGES_EN + 4) else { return ERR_MEMORY_FAULT };
    let base_min = base.0.min(base.1);
    let shift = if stages & STAGES_GS_W32_EN != 0 { 5 } else { 6 };
    let expanded_limit = expanded.0.min(expanded.1).min(budget >> shift).min(1024);
    let headroom = expanded_limit.saturating_sub(base_min);
    let target = factor.mul_add(headroom as f32, base_min as f32) as u32;
    if target <= base_min {
        return write(pc, sh);
    }
    if target < base.1 {
        let range = expanded.0.wrapping_sub(base.0).max(1);
        let value = ((((target.wrapping_sub(base.0) as u64) << 10) / range as u64).min(1024) as u32).max(1);
        pc = (pc & !FULL_PC_OVERSUBSCRIPTION) | (((value << 1) - 1) & FULL_PC_OVERSUBSCRIPTION);
        sh |= FULL_SH_OVERSUBSCRIPTION;
    } else {
        let range = expanded.1.wrapping_sub(base.1).max(1);
        let value = (((target - base.1) as u64 * 127) / range as u64).min(127) as u32;
        pc |= FULL_PC_OVERSUBSCRIPTION;
        sh = (sh & !FULL_SH_OVERSUBSCRIPTION) | (value << 16);
    }
    write(pc, sh)
}

fn interpolant_table() -> Vec<u32> {
    let mut t = vec![0u32; INTERPOLANT_COUNT * 2];
    for i in 0..INTERPOLANT_COUNT {
        t[i * 2] = SPI_PS_INPUT_CNTL_0 + i as u32;
        t[i * 2 + 1] = i as u32;
    }
    t
}

/// The output semantic word whose semantic id (low byte) matches.
fn find_output_semantic(outputs: &[u32], input: u32) -> Option<u32> {
    outputs.iter().copied().find(|o| *o as u8 == input as u8)
}

fn two_bits(v: u32, field: u32, shift: u32) -> u32 {
    (v & !(3 << shift)) | ((field & 3) << shift)
}

fn interpolant_f16(ps: u32, gs: Option<u32>) -> u32 {
    let mut v = (ps << 4) & 0x0300_0000;
    match gs {
        None => v |= 0x0018_0020,
        Some(g) => {
            let common = ps & g;
            v &= 0xFFF7_FFDF;
            v |= (common >> 15) & 0x20;
            v ^= 0x0008_0020;
            v &= !0x0010_0000;
            v |= (!common >> 1) & 0x0010_0000;
        }
    }
    (v & !0x0060_0000) | (((ps >> 30) & 3) << 21)
}

fn interpolant_value(ps: u32, gs: Option<u32>) -> u32 {
    let base = if ps & 0x0030_0000 != 0 {
        interpolant_f16(ps, gs)
    } else if ps & 0x0100_0000 != 0 || gs.is_none() {
        0x20
    } else {
        0
    };
    let mut v = base & !0x1F & !0x400;
    if let Some(g) = gs {
        v |= (g >> 8) & 0x1F;
        if ps & 0x0040_0000 != 0 || ps & 0x0100_0000 != 0 {
            v |= 0x400;
        }
    }
    (v & !0x300) | (((ps >> 28) & 3) << 8)
}

fn read_semantics(ps: u64, gs: u64, cap: Option<u32>) -> Result<(Vec<u32>, Vec<u32>), HleResult> {
    if ps == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let (Some(in_addr), Some(in_count)) = (mem::r64(ps + OFF_INPUT_SEMANTICS), mem::r32(ps + OFF_NUM_INPUT_SEMANTICS)) else {
        return Err(ERR_MEMORY_FAULT);
    };
    if in_count == 0 || in_addr == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let in_count = match cap {
        Some(c) => in_count.min(c),
        None if in_count > 32 => return Err(ERR_INVALID_ARGUMENT),
        None => in_count,
    };
    if gs == 0 {
        return Err(ERR_INVALID_ARGUMENT);
    }
    let (Some(out_addr), Some(out_count)) = (mem::r64(gs + OFF_OUTPUT_SEMANTICS), mem::r16(gs + OFF_NUM_OUTPUT_SEMANTICS)) else {
        return Err(ERR_MEMORY_FAULT);
    };
    let inputs = mem::read_dwords(in_addr, in_count as usize).ok_or(ERR_MEMORY_FAULT)?;
    let outputs = if out_addr != 0 && out_count != 0 {
        mem::read_dwords(out_addr, out_count as usize).ok_or(ERR_MEMORY_FAULT)?
    } else {
        Vec::new()
    };
    Ok((inputs, outputs))
}

/// The 32 `{SPI_PS_INPUT_CNTL_i, value}` pairs of
/// `sceAgcCreateInterpolantMapping`: one per PS input, matching GS outputs
/// by semantic id (location in bits 4:0, flat shading at bit 10), the rest
/// the identity.
fn interpolant_mapping(gs: u64, ps: u64) -> Result<Vec<u32>, HleResult> {
    let (inputs, outputs) = read_semantics(ps, gs, Some(INTERPOLANT_COUNT as u32))?;
    let mut t = interpolant_table();
    for (i, &w) in inputs.iter().enumerate() {
        t[i * 2 + 1] = interpolant_value(w, find_output_semantic(&outputs, w));
    }
    Ok(t)
}

/// `sceAgcCreateInterpolantMapping(regs[32], gs, ps)`.
fn create_interpolant_mapping(ctx: &mut CpuContext) -> HleResult {
    let (regs, gs, ps) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2));
    if regs == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    match interpolant_mapping(gs, ps) {
        Ok(t) if mem::write_dwords(regs, &t) => OK,
        Ok(_) => ERR_MEMORY_FAULT,
        Err(e) => e,
    }
}

/// SPI_PS_INPUT_CNTL value of the second-generation mapping for PS input
/// semantic `src` and the GS output it matched (f16 inputs whose low half
/// is not interpolated, mode 2, keep the high half's defaults).
fn interpolant_value2(src: u32, found: Option<u32>) -> u32 {
    let mode = (src >> 20) & 3;
    let mut f;
    if mode == 0 {
        f = (((src >> 24) & 1) | found.is_none() as u32) << 5;
        f = two_bits(f, src >> 28, 8);
    } else {
        f = ((src << 4) & 0x0300_0000) + 0x0008_0000;
        if mode == 2 {
            f &= 0xFFEF_FFDF;
            f |= match found {
                Some(m) => (!(m & src) >> 16) & 0x20,
                None => 0x20,
            };
            f = two_bits(f, src >> 30, 8);
            f = two_bits(f, src >> 30, 21);
        } else {
            match found {
                Some(m) => {
                    let masked = m & src;
                    f = (f & 0xFFFF_FFDF) | ((masked >> 15) & 0x20);
                    f ^= 0x20;
                    f = (f & 0xFFEF_FFFF) | ((!masked >> 1) & 0x0010_0000);
                }
                None => f |= 0x0010_0020,
            }
            f = two_bits(f, src >> 28, 8);
            f = two_bits(f, src >> 30, 21);
        }
    }
    match found {
        Some(m) => {
            let g = (f & 0xFFFF_FFE0) | ((m >> 8) & 0x1F);
            (g & 0xFFFF_FBFF) | if src & 0x0040_0000 != 0 { 0x400 } else { (src >> 14) & 0x400 }
        }
        None => f & 0xFFFF_FBE0,
    }
}

/// Second-generation mapping (different flag packing).
fn create_interpolant_mapping2(ctx: &mut CpuContext) -> HleResult {
    let (regs, gs, ps) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2));
    if regs == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let (inputs, outputs) = match read_semantics(ps, gs, None) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut t = interpolant_table();
    for (i, &src) in inputs.iter().enumerate() {
        t[i * 2 + 1] = interpolant_value2(src, find_output_semantic(&outputs, src));
    }
    if mem::write_dwords(regs, &t) {
        OK
    } else {
        ERR_MEMORY_FAULT
    }
}

/// `sceAgcUpdateInterpolantMapping_10000(regs, gs, ps)`: the
/// second-generation mapping of the PS inputs only (no identity tail; a
/// PS without inputs writes nothing). An f16 input interpolating its low
/// half whose low and high defaults differ is refused: which default the
/// update keeps is not known.
fn update_interpolant_mapping_10000(ctx: &mut CpuContext) -> HleResult {
    let (regs, gs, ps) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2));
    if regs == 0 || ps == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let (inputs, outputs) = match read_semantics(ps, gs, None) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if let Some(w) = inputs.iter().find(|&&w| matches!((w >> 20) & 3, 1 | 3) && (w >> 28) & 3 != (w >> 30) & 3) {
        log::error!("[agc] sceAgcUpdateInterpolantMapping_10000: f16 input {w:#010x} has different low and high defaults");
        return ERR_INVALID_ARGUMENT;
    }
    let mut t = Vec::with_capacity(inputs.len() * 2);
    for (i, &src) in inputs.iter().enumerate() {
        t.extend_from_slice(&[SPI_PS_INPUT_CNTL_0 + i as u32, interpolant_value2(src, find_output_semantic(&outputs, src))]);
    }
    if t.is_empty() || mem::write_dwords(regs, &t) {
        OK
    } else {
        ERR_MEMORY_FAULT
    }
}

/// SCE_AGC_ERROR_INVALID_SHADER_PROGRAM.
const ERR_INVALID_SHADER_PROGRAM: HleResult = 0x8A6C_0005u32 as i32 as i64;

/// `sceAgcLinkShaders(cxRegs[34], ucRegs[3], reserved, gs, ps, primType)`:
/// the primitive state of the geometry shader (two context and three user
/// config registers) followed by the interpolant mapping of `ps` (32
/// context registers). Nothing is written unless both succeed.
fn link_shaders(ctx: &mut CpuContext) -> HleResult {
    let (cx, uc, reserved, gs, ps, prim) = (arg(ctx, 0), arg(ctx, 1), arg(ctx, 2), arg(ctx, 3), arg(ctx, 4), arg(ctx, 5) as u32);
    let gs_ok = gs != 0
        && mem::r8(gs + OFF_TYPE) == Some(TYPE_GS)
        && (mem::r16(gs + OFF_NUM_OUTPUT_SEMANTICS) == Some(0) || mem::r64(gs + OFF_OUTPUT_SEMANTICS).is_some_and(|a| a != 0));
    let inputs = if ps == 0 { Some(0) } else { mem::r32(ps + OFF_NUM_INPUT_SEMANTICS) };
    let ps_ok = ps == 0
        || (mem::r8(ps + OFF_TYPE) == Some(TYPE_PS)
            && inputs.is_some_and(|n| n <= INTERPOLANT_COUNT as u32)
            && (inputs == Some(0) || mem::r64(ps + OFF_INPUT_SEMANTICS).is_some_and(|a| a != 0)));
    if cx == 0 || uc == 0 || reserved != 0 || !gs_ok || !ps_ok {
        return ERR_INVALID_SHADER_PROGRAM;
    }
    let (cx_values, uc_values) = match prim_state(0, gs, prim, true, true) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mapping = match interpolant_mapping(gs, ps) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let context: Vec<u32> = cx_values.iter().chain(mapping.iter()).copied().collect();
    if mem::write_dwords(cx, &context) && mem::write_dwords(uc, &uc_values) {
        OK
    } else {
        ERR_MEMORY_FAULT
    }
}

/// `sceAgcGetGsPrimPayload(payload, gs)`: 8 bytes of primitive payload
/// when a context register of the geometry shader sets
/// SPI_SHADER_IDX_FORMAT's IDX0 export format to 2 components, else 0.
fn get_gs_prim_payload(ctx: &mut CpuContext) -> HleResult {
    const SPI_SHADER_IDX_FORMAT: u32 = 0x1C2;
    let (payload, gs) = (arg(ctx, 0), arg(ctx, 1));
    if payload == 0 || gs == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let (Some(regs), Some(n)) = (mem::r64(gs + OFF_CX_REGS), mem::r8(gs + OFF_NUM_CX_REGS)) else { return ERR_MEMORY_FAULT };
    if n != 0 && regs == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let mut bytes = 0;
    for i in 0..n as u64 {
        let (Some(offset), Some(value)) = (mem::r32(regs + i * 8), mem::r32(regs + i * 8 + 4)) else { return ERR_MEMORY_FAULT };
        if offset == SPI_SHADER_IDX_FORMAT && value & 0xF == 2 {
            bytes = 8;
            break;
        }
    }
    if mem::w32(payload, bytes) {
        OK
    } else {
        ERR_MEMORY_FAULT
    }
}

pub(crate) const EXPORTS: &[Export] = &[
    ("ICkECTBxrMw", "sceAgcGetGsPrimPayload", LIB_AGC, get_gs_prim_payload),
    ("f3dg2CSgRKY", "sceAgcCreateShader", LIB_AGC, create_shader),
    ("dolOmWH+huQ", "sceAgcGetFusedShaderSize_0080", LIB_AGC, get_fused_shader_size),
    ("fd5Bp5tGTgo", "sceAgcFuseShaderHalves_0200", LIB_AGC, fuse_shader_halves),
    ("nApJjpKNBl4", "sceAgcFuseShaderHalves", LIB_AGC, fuse_shader_halves_legacy),
    ("D9sr1xGUriE", "sceAgcCreatePrimState", LIB_AGC, create_prim_state),
    ("Y3ymLfZ1384", "sceAgcUpdatePrimState", LIB_AGC, update_prim_state),
    ("NKIzURsgV7I", "sceAgcGetGsOversubscription", LIB_AGC, get_gs_oversubscription),
    ("HV4j+E0MBHE", "sceAgcCreateInterpolantMapping_0100", LIB_AGC, create_interpolant_mapping),
    ("dbOlWdppb4o", "sceAgcCreateInterpolantMapping_10000", LIB_AGC, create_interpolant_mapping2),
    ("vieBRwlh1Lw", "sceAgcUpdateInterpolantMapping_10000", LIB_AGC, update_interpolant_mapping_10000),
    ("MqAdbRMdNz4", "sceAgcLinkShaders", LIB_AGC, link_shaders),
];
