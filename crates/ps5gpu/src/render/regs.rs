// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Register offsets (dword index inside each bank) the draw state
//! resolution reads.

/// Context bank (0xA000-based register block).
pub mod cx {
    pub const DB_RENDER_CONTROL: u32 = 0x000;
    pub const DB_DEPTH_VIEW: u32 = 0x002;
    pub const DB_RENDER_OVERRIDE: u32 = 0x003;
    pub const DB_HTILE_DATA_BASE: u32 = 0x005;
    pub const DB_DEPTH_SIZE_XY: u32 = 0x007;
    pub const DB_DEPTH_BOUNDS_MIN: u32 = 0x008;
    pub const DB_DEPTH_BOUNDS_MAX: u32 = 0x009;
    pub const DB_STENCIL_CLEAR: u32 = 0x00A;
    pub const DB_DEPTH_CLEAR: u32 = 0x00B;
    pub const PA_SC_SCREEN_SCISSOR_TL: u32 = 0x00C;
    pub const PA_SC_SCREEN_SCISSOR_BR: u32 = 0x00D;
    pub const DB_Z_INFO: u32 = 0x010;
    pub const DB_STENCIL_INFO: u32 = 0x011;
    pub const DB_Z_READ_BASE: u32 = 0x012;
    pub const DB_STENCIL_READ_BASE: u32 = 0x013;
    pub const DB_Z_WRITE_BASE: u32 = 0x014;
    pub const DB_STENCIL_WRITE_BASE: u32 = 0x015;
    pub const DB_Z_READ_BASE_HI: u32 = 0x01A;
    pub const DB_STENCIL_READ_BASE_HI: u32 = 0x01B;
    pub const DB_Z_WRITE_BASE_HI: u32 = 0x01C;
    pub const DB_STENCIL_WRITE_BASE_HI: u32 = 0x01D;
    pub const DB_HTILE_DATA_BASE_HI: u32 = 0x01E;
    pub const PA_SC_WINDOW_OFFSET: u32 = 0x080;
    pub const PA_SC_WINDOW_SCISSOR_TL: u32 = 0x081;
    pub const PA_SC_WINDOW_SCISSOR_BR: u32 = 0x082;
    pub const PA_SC_CLIPRECT_RULE: u32 = 0x083;
    /// PA_SC_CLIPRECT_n_TL / _BR pairs (n = 0..3) from here.
    pub const PA_SC_CLIPRECT_0_TL: u32 = 0x084;
    pub const CB_TARGET_MASK: u32 = 0x08E;
    pub const CB_SHADER_MASK: u32 = 0x08F;
    pub const PA_SC_GENERIC_SCISSOR_TL: u32 = 0x090;
    pub const PA_SC_GENERIC_SCISSOR_BR: u32 = 0x091;
    pub const PA_SC_VPORT_SCISSOR_0_TL: u32 = 0x094;
    pub const PA_SC_VPORT_ZMIN_0: u32 = 0x0B4;
    pub const VGT_MULTI_PRIM_IB_RESET_INDX: u32 = 0x103;
    pub const CB_BLEND_RED: u32 = 0x105;
    pub const DB_STENCIL_CONTROL: u32 = 0x10B;
    pub const DB_STENCILREFMASK: u32 = 0x10C;
    pub const DB_STENCILREFMASK_BF: u32 = 0x10D;
    pub const PA_CL_VPORT_XSCALE: u32 = 0x10F;
    pub const SPI_PS_INPUT_CNTL_0: u32 = 0x191;
    pub const SPI_VS_OUT_CONFIG: u32 = 0x1B1;
    pub const SPI_PS_INPUT_ENA: u32 = 0x1B3;
    pub const SPI_PS_INPUT_ADDR: u32 = 0x1B4;
    pub const SPI_PS_IN_CONTROL: u32 = 0x1B6;
    pub const SPI_BARYC_CNTL: u32 = 0x1B8;
    pub const SPI_SHADER_POS_FORMAT: u32 = 0x1C3;
    pub const SPI_SHADER_Z_FORMAT: u32 = 0x1C4;
    pub const SPI_SHADER_COL_FORMAT: u32 = 0x1C5;
    pub const CB_BLEND0_CONTROL: u32 = 0x1E0;
    pub const DB_DEPTH_CONTROL: u32 = 0x200;
    pub const DB_EQAA: u32 = 0x201;
    pub const CB_COLOR_CONTROL: u32 = 0x202;
    pub const DB_SHADER_CONTROL: u32 = 0x203;
    pub const DB_ALPHA_TO_MASK: u32 = 0x2DC;
    pub const PA_CL_CLIP_CNTL: u32 = 0x204;
    /// PA_CL_UCP_0_X..PA_CL_UCP_5_W: six planes of four floats.
    pub const PA_CL_UCP_0_X: u32 = 0x16F;
    pub const PA_SU_SC_MODE_CNTL: u32 = 0x205;
    pub const PA_CL_VTE_CNTL: u32 = 0x206;
    pub const PA_CL_VS_OUT_CNTL: u32 = 0x207;
    pub const PA_SU_LINE_CNTL: u32 = 0x282;
    pub const PA_SC_MODE_CNTL_0: u32 = 0x292;
    pub const PA_SC_MODE_CNTL_1: u32 = 0x293;
    pub const PA_SC_AA_MASK_X0Y0_X1Y0: u32 = 0x30E;
    pub const PA_SC_AA_MASK_X0Y1_X1Y1: u32 = 0x30F;
    pub const GE_MAX_OUTPUT_PER_SUBGROUP: u32 = 0x1FF;
    pub const VGT_GS_OUT_PRIM_TYPE: u32 = 0x29B;
    pub const VGT_ESGS_RING_ITEMSIZE: u32 = 0x2AB;
    pub const VGT_GS_MAX_VERT_OUT: u32 = 0x2CE;
    pub const DB_HTILE_SURFACE: u32 = 0x2AF;
    pub const VGT_SHADER_STAGES_EN: u32 = 0x2D5;
    pub const PA_SU_POLY_OFFSET_DB_FMT_CNTL: u32 = 0x2DE;
    pub const PA_SU_POLY_OFFSET_CLAMP: u32 = 0x2DF;
    pub const PA_SU_POLY_OFFSET_FRONT_SCALE: u32 = 0x2E0;
    pub const PA_SU_POLY_OFFSET_FRONT_OFFSET: u32 = 0x2E1;
    pub const PA_SU_POLY_OFFSET_BACK_SCALE: u32 = 0x2E2;
    pub const PA_SU_POLY_OFFSET_BACK_OFFSET: u32 = 0x2E3;
    pub const PA_SC_AA_CONFIG: u32 = 0x2F8;
    pub const CB_COLOR0_BASE: u32 = 0x318;
    pub const CB_COLOR0_VIEW: u32 = 0x31B;
    pub const CB_COLOR0_INFO: u32 = 0x31C;
    pub const CB_COLOR0_ATTRIB: u32 = 0x31D;
    pub const CB_COLOR0_DCC_CONTROL: u32 = 0x31E;
    pub const CB_COLOR0_CMASK: u32 = 0x31F;
    pub const CB_COLOR0_FMASK: u32 = 0x321;
    pub const CB_COLOR0_CLEAR_WORD0: u32 = 0x323;
    pub const CB_COLOR0_CLEAR_WORD1: u32 = 0x324;
    pub const CB_COLOR0_DCC_BASE: u32 = 0x325;
    /// Stride between the per-slot CB_COLORn blocks.
    pub const CB_COLOR_STRIDE: u32 = 0xF;
    pub const CB_COLOR0_BASE_EXT: u32 = 0x390;
    pub const CB_COLOR0_CMASK_BASE_EXT: u32 = 0x398;
    pub const CB_COLOR0_FMASK_BASE_EXT: u32 = 0x3A0;
    pub const CB_COLOR0_DCC_BASE_EXT: u32 = 0x3A8;
    pub const CB_COLOR0_ATTRIB2: u32 = 0x3B0;
    pub const CB_COLOR0_ATTRIB3: u32 = 0x3B8;
}

/// Shader (persistent state) bank.
pub mod sh {
    pub const SPI_SHADER_PGM_LO_PS: u32 = 0x008;
    pub const SPI_SHADER_PGM_RSRC1_PS: u32 = 0x00A;
    pub const SPI_SHADER_USER_DATA_PS_0: u32 = 0x00C;
    pub const SPI_SHADER_PGM_LO_VS: u32 = 0x048;
    pub const SPI_SHADER_PGM_RSRC1_VS: u32 = 0x04A;
    pub const SPI_SHADER_USER_DATA_VS_0: u32 = 0x04C;
    pub const SPI_SHADER_PGM_LO_GS: u32 = 0x088;
    pub const SPI_SHADER_PGM_RSRC1_GS: u32 = 0x08A;
    pub const SPI_SHADER_USER_DATA_GS_0: u32 = 0x08C;
    pub const SPI_SHADER_USER_DATA_ADDR_LO_GS: u32 = 0x082;
    pub const SPI_SHADER_PGM_LO_ES: u32 = 0x0C8;
    pub const SPI_SHADER_PGM_RSRC1_ES: u32 = 0x0CA;
    pub const SPI_SHADER_USER_DATA_ES_0: u32 = 0x0CC;
    pub const SPI_SHADER_PGM_LO_HS: u32 = 0x108;
    pub const SPI_SHADER_PGM_RSRC1_HS: u32 = 0x10A;
    pub const SPI_SHADER_USER_DATA_HS_0: u32 = 0x10C;
    pub const SPI_SHADER_PGM_LO_LS: u32 = 0x148;
    pub const SPI_SHADER_PGM_RSRC1_LS: u32 = 0x14A;
    pub const COMPUTE_START_X: u32 = 0x204;
    pub const COMPUTE_NUM_THREAD_X: u32 = 0x207;
    pub const COMPUTE_PGM_LO: u32 = 0x20C;
    pub const COMPUTE_PGM_RSRC1: u32 = 0x212;
    pub const COMPUTE_PGM_RSRC2: u32 = 0x213;
    pub const COMPUTE_PGM_RSRC3: u32 = 0x228;
    pub const COMPUTE_USER_DATA_0: u32 = 0x240;
}

/// User-config bank.
pub mod uc {
    pub const VGT_PRIMITIVE_TYPE: u32 = 0x242;
    pub const VGT_INDEX_TYPE: u32 = 0x243;
    pub const GE_INDEX_OFFSET: u32 = 0x24A;
    pub const GE_MULTI_PRIM_IB_RESET_EN: u32 = 0x24B;
    pub const VGT_NUM_INSTANCES: u32 = 0x24D;
    pub const GE_CNTL: u32 = 0x25B;
}
