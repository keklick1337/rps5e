// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceAudiodec: frame decoders addressed by handle. MPEG-4 AAC is
//! decoded with the AJM AAC decoder; ATRAC9 and MP3 frames produce
//! silence of one frame's size.

use super::ajm::aac::{self, AjmAacDecoder};
use super::ajm::decoder::{
    AjmDecoder, GaplessState, PcmEncoding, RESULT_CODEC_ERROR, RESULT_FATAL, RESULT_INVALID_PARAMETER,
    RESULT_NOT_ENOUGH_ROOM, RESULT_PARTIAL_INPUT, RESULT_TOO_MANY_CHANNELS,
};
use super::Export;
use crate::context::CpuContext;
use crate::registry::HleResult;
use std::sync::Mutex;

const fn err(code: u32) -> HleResult {
    code as i32 as i64
}

pub const AUDIODEC_ERROR_API_FAIL: HleResult = err(0x807F_0000);
pub const AUDIODEC_ERROR_INVALID_TYPE: HleResult = err(0x807F_0001);
pub const AUDIODEC_ERROR_ARG: HleResult = err(0x807F_0002);
pub const AUDIODEC_ERROR_INVALID_PARAM_SIZE: HleResult = err(0x807F_0004);
pub const AUDIODEC_ERROR_INVALID_BSI_INFO_SIZE: HleResult = err(0x807F_0005);
pub const AUDIODEC_ERROR_INVALID_AU_INFO_SIZE: HleResult = err(0x807F_0006);
pub const AUDIODEC_ERROR_INVALID_PCM_ITEM_SIZE: HleResult = err(0x807F_0007);
pub const AUDIODEC_ERROR_INVALID_CTRL_POINTER: HleResult = err(0x807F_0008);
pub const AUDIODEC_ERROR_INVALID_PARAM_POINTER: HleResult = err(0x807F_0009);
pub const AUDIODEC_ERROR_INVALID_BSI_INFO_POINTER: HleResult = err(0x807F_000A);
pub const AUDIODEC_ERROR_INVALID_AU_INFO_POINTER: HleResult = err(0x807F_000B);
pub const AUDIODEC_ERROR_INVALID_PCM_ITEM_POINTER: HleResult = err(0x807F_000C);
pub const AUDIODEC_ERROR_INVALID_AU_POINTER: HleResult = err(0x807F_000D);
pub const AUDIODEC_ERROR_INVALID_PCM_POINTER: HleResult = err(0x807F_000E);
pub const AUDIODEC_ERROR_INVALID_HANDLE: HleResult = err(0x807F_000F);
pub const AUDIODEC_ERROR_INVALID_WORD_LENGTH: HleResult = err(0x807F_0010);
pub const AUDIODEC_ERROR_INVALID_AU_SIZE: HleResult = err(0x807F_0011);
pub const AUDIODEC_ERROR_INVALID_PCM_SIZE: HleResult = err(0x807F_0012);

pub const TYPE_AT9: u32 = 1;
pub const TYPE_MP3: u32 = 2;
pub const TYPE_M4AAC: u32 = 3;

const WORD_SZ_24BIT: i32 = 0;
const WORD_SZ_16BIT: i32 = 1;
const WORD_SZ_FLOAT: i32 = 2;

const AT9_MAX_FRAME_SIZE: u32 = 2048;
const AT9_MAX_FRAME_SAMPLES: u32 = 256;
const MP3_MAX_FRAME_SIZE: u32 = 1441;
const MP3_MAX_FRAME_SAMPLES: u32 = 1152;

const M4AAC_RESULT_DECODE_ERROR: i32 = -4;
const M4AAC_RESULT_INSUFFICIENT_DATA: i32 = -5;

/// SceAudiodecAuInfo / SceAudiodecPcmItem { u32 size; void *addr; u32 bytes; }.
const ITEM_SIZE: u32 = 24;
const PARAM_AT9_SIZE: u32 = 12;
const AT9_INFO_SIZE: u32 = 36;
const PARAM_MP3_SIZE: u32 = 8;
const MP3_INFO_SIZE: u32 = 20;
const PARAM_M4AAC_SIZE: u32 = 24;
const PARAM_M4AAC_EX_SIZE: u32 = 28;
const M4AAC_INFO_SIZE: u32 = 20;

const MAX_DECODERS: usize = 64;

struct Decoder {
    codec_type: u32,
    word_size: i32,
    channels: u32,
    sample_rate: u32,
    frame_bytes: u32,
    frame_samples: u32,
    aac: Option<AjmAacDecoder>,
}

struct State {
    init_count: [u32; 4],
    decoders: Vec<Option<Decoder>>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(|| State { init_count: [0; 4], decoders: (0..MAX_DECODERS).map(|_| None).collect() }))
}

fn is_valid_type(t: u32) -> bool {
    matches!(t, TYPE_AT9 | TYPE_MP3 | TYPE_M4AAC)
}

fn word_size_bytes(w: i32) -> u32 {
    match w {
        WORD_SZ_24BIT | WORD_SZ_FLOAT => 4,
        WORD_SZ_16BIT => 2,
        _ => 0,
    }
}

/// The guest SceAudiodecCtrl, read once.
struct Ctrl {
    param: u64,
    bsi: u64,
    au_info: u64,
    pcm_item: u64,
}

impl Ctrl {
    fn read(ctx: &CpuContext, addr: u64) -> Self {
        let r = |o| ctx.read_u64(addr + o).unwrap_or(0);
        Self { param: r(0), bsi: r(8), au_info: r(16), pcm_item: r(24) }
    }
}

fn validate_ctrl(ctx: &CpuContext, addr: u64, codec_type: u32, decode: bool) -> Result<Ctrl, HleResult> {
    if addr == 0 {
        return Err(AUDIODEC_ERROR_INVALID_CTRL_POINTER);
    }
    let c = Ctrl::read(ctx, addr);
    if c.param == 0 {
        return Err(AUDIODEC_ERROR_INVALID_PARAM_POINTER);
    }
    if c.bsi == 0 {
        return Err(AUDIODEC_ERROR_INVALID_BSI_INFO_POINTER);
    }
    let u32_at = |a: u64| ctx.read_u32(a).unwrap_or(0);
    if decode {
        if c.au_info == 0 {
            return Err(AUDIODEC_ERROR_INVALID_AU_INFO_POINTER);
        }
        if c.pcm_item == 0 {
            return Err(AUDIODEC_ERROR_INVALID_PCM_ITEM_POINTER);
        }
        if u32_at(c.au_info) != ITEM_SIZE {
            return Err(AUDIODEC_ERROR_INVALID_AU_INFO_SIZE);
        }
        if u32_at(c.pcm_item) != ITEM_SIZE {
            return Err(AUDIODEC_ERROR_INVALID_PCM_ITEM_SIZE);
        }
        if ctx.read_u64(c.au_info + 8).unwrap_or(0) == 0 {
            return Err(AUDIODEC_ERROR_INVALID_AU_POINTER);
        }
        if ctx.read_u64(c.pcm_item + 8).unwrap_or(0) == 0 {
            return Err(AUDIODEC_ERROR_INVALID_PCM_POINTER);
        }
    }
    let (param_size, info_size) = (u32_at(c.param), u32_at(c.bsi));
    let (param_ok, info_ok) = match codec_type {
        TYPE_AT9 => (param_size == PARAM_AT9_SIZE, info_size == AT9_INFO_SIZE),
        TYPE_MP3 => (param_size == PARAM_MP3_SIZE, info_size == MP3_INFO_SIZE),
        TYPE_M4AAC => (param_size >= PARAM_M4AAC_SIZE, info_size == M4AAC_INFO_SIZE),
        _ => return Err(AUDIODEC_ERROR_INVALID_TYPE),
    };
    if !param_ok {
        return Err(AUDIODEC_ERROR_INVALID_PARAM_SIZE);
    }
    if !info_ok {
        return Err(AUDIODEC_ERROR_INVALID_BSI_INFO_SIZE);
    }
    // The PS5 decoders take 16-bit PCM, and float except for MP3; 24-bit
    // and anything else is INVALID_WORD_LENGTH.
    match u32_at(c.param + 4) as i32 {
        WORD_SZ_16BIT => {}
        WORD_SZ_FLOAT if codec_type != TYPE_MP3 => {}
        _ => return Err(AUDIODEC_ERROR_INVALID_WORD_LENGTH),
    }
    if decode && u32_at(c.au_info + 16) == 0 {
        return Err(AUDIODEC_ERROR_INVALID_AU_SIZE);
    }
    if decode && u32_at(c.pcm_item + 16) == 0 {
        return Err(AUDIODEC_ERROR_INVALID_PCM_SIZE);
    }
    Ok(c)
}

/// Writes the codec's BSI info block (after its size field).
fn fill_info(ctx: &CpuContext, bsi: u64, d: &Decoder, result: i32) {
    match d.codec_type {
        TYPE_AT9 => {
            let words = [d.channels, 0, d.sample_rate, d.frame_bytes, 1, d.frame_bytes, d.frame_samples, result as u32];
            let b: Vec<u8> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
            ctx.write_bytes(bsi + 4, &b);
        }
        TYPE_MP3 => {
            let mut b = [0u8; 16];
            b[5] = if d.channels == 1 { 3 } else { 0 };
            b[12..16].copy_from_slice(&result.to_le_bytes());
            ctx.write_bytes(bsi + 4, &b);
        }
        TYPE_M4AAC => {
            let f = d.aac.as_ref().map(|a| a.format()).unwrap_or_default();
            let words = [f.sample_rate, f.channels, 0, result as u32];
            let b: Vec<u8> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
            ctx.write_bytes(bsi + 4, &b);
        }
        _ => {}
    }
}

/// sceAudiodecInitLibrary(u32 codecType)
fn cb_init_library(ctx: &mut CpuContext) -> HleResult {
    let t = ctx.rdi() as u32;
    if !is_valid_type(t) {
        return AUDIODEC_ERROR_INVALID_TYPE;
    }
    with_state(|s| s.init_count[t as usize] += 1);
    0
}

/// sceAudiodecTermLibrary(u32 codecType)
fn cb_term_library(ctx: &mut CpuContext) -> HleResult {
    let t = ctx.rdi() as u32;
    if !is_valid_type(t) {
        return AUDIODEC_ERROR_INVALID_TYPE;
    }
    with_state(|s| s.init_count[t as usize] = s.init_count[t as usize].saturating_sub(1));
    0
}

/// sceAudiodecCreateDecoder(SceAudiodecCtrl *ctrl, u32 codecType) -> handle
fn cb_create_decoder(ctx: &mut CpuContext) -> HleResult {
    let (ctrl, t) = (ctx.rdi(), ctx.rsi() as u32);
    if !is_valid_type(t) {
        return AUDIODEC_ERROR_INVALID_TYPE;
    }
    let c = match validate_ctrl(ctx, ctrl, t, false) {
        Ok(c) => c,
        Err(e) => return e,
    };
    let u32_at = |o: u64| ctx.read_u32(c.param + o).unwrap_or(0);
    with_state(|s| {
        if s.init_count[t as usize] == 0 {
            return AUDIODEC_ERROR_ARG;
        }
        let Some(slot) = s.decoders.iter().position(|d| d.is_none()) else { return AUDIODEC_ERROR_ARG };
        let word_size = u32_at(4) as i32;
        let mut d = Decoder {
            codec_type: t,
            word_size,
            channels: 2,
            sample_rate: 48000,
            frame_bytes: MP3_MAX_FRAME_SIZE,
            frame_samples: MP3_MAX_FRAME_SAMPLES,
            aac: None,
        };
        match t {
            TYPE_AT9 => {
                d.frame_bytes = AT9_MAX_FRAME_SIZE;
                d.frame_samples = AT9_MAX_FRAME_SAMPLES;
            }
            TYPE_MP3 => {}
            _ => {
                let max_channels = match u32_at(16) {
                    0 => 2,
                    n => n,
                }
                .min(8);
                let encoding = match word_size {
                    WORD_SZ_16BIT => PcmEncoding::S16,
                    WORD_SZ_FLOAT => PcmEncoding::Float,
                    _ => PcmEncoding::S32,
                };
                let mut flags = if u32_at(20) != 0 { aac::FLAG_ENABLE_SBR_DECODE } else { 0 };
                if u32_at(0) >= PARAM_M4AAC_EX_SIZE && u32_at(24) != 0 {
                    flags |= aac::FLAG_ENABLE_NONDELAY_OUTPUT;
                }
                let mut a = AjmAacDecoder::new(max_channels, d.sample_rate, encoding, flags);
                let mut params = [0u8; aac::INITIALIZE_PARAMETERS_BYTES];
                params[0..4].copy_from_slice(&u32_at(8).to_le_bytes());
                params[4..8].copy_from_slice(&u32_at(12).to_le_bytes());
                if a.initialize(&params).result != 0 {
                    return AUDIODEC_ERROR_API_FAIL;
                }
                d.aac = Some(a);
            }
        }
        fill_info(ctx, c.bsi, &d, 0);
        s.decoders[slot] = Some(d);
        (slot + 1) as HleResult
    })
}

fn handle_slot(handle: i32) -> Option<usize> {
    (handle > 0 && handle as usize <= MAX_DECODERS).then(|| handle as usize - 1)
}

/// sceAudiodecDeleteDecoder(int handle)
fn cb_delete_decoder(ctx: &mut CpuContext) -> HleResult {
    let Some(slot) = handle_slot(ctx.rdi() as i32) else { return AUDIODEC_ERROR_INVALID_HANDLE };
    with_state(|s| match s.decoders[slot].take() {
        Some(_) => 0,
        None => AUDIODEC_ERROR_INVALID_HANDLE,
    })
}

/// sceAudiodecDecode(int handle, SceAudiodecCtrl *ctrl)
fn cb_decode(ctx: &mut CpuContext) -> HleResult {
    let (handle, ctrl) = (ctx.rdi() as i32, ctx.rsi());
    let Some(slot) = handle_slot(handle) else { return AUDIODEC_ERROR_INVALID_HANDLE };
    with_state(|s| {
        let Some(d) = s.decoders[slot].as_mut() else { return AUDIODEC_ERROR_INVALID_HANDLE };
        let c = match validate_ctrl(ctx, ctrl, d.codec_type, true) {
            Ok(c) => c,
            Err(e) => return e,
        };
        let au_addr = ctx.read_u64(c.au_info + 8).unwrap_or(0);
        let au_size = ctx.read_u32(c.au_info + 16).unwrap_or(0);
        let pcm_addr = ctx.read_u64(c.pcm_item + 8).unwrap_or(0);
        let pcm_size = ctx.read_u32(c.pcm_item + 16).unwrap_or(0);

        if let Some(a) = d.aac.as_mut() {
            let input = ctx.read_bytes(au_addr, au_size as usize).unwrap_or_default();
            let mut output = vec![0u8; pcm_size as usize];
            let r = a.decode(&input, &mut output, false, &mut GaplessState::default());
            ctx.write_bytes(pcm_addr, &output[..r.output_written]);
            ctx.write_u32(c.au_info + 16, r.input_consumed as u32);
            ctx.write_u32(c.pcm_item + 16, r.output_written as u32);
            let info_result = if r.result & (RESULT_CODEC_ERROR | RESULT_TOO_MANY_CHANNELS) != 0 {
                M4AAC_RESULT_DECODE_ERROR
            } else if r.result & RESULT_PARTIAL_INPUT != 0 {
                M4AAC_RESULT_INSUFFICIENT_DATA
            } else {
                0
            };
            fill_info(ctx, c.bsi, d, info_result);
            if r.result & RESULT_INVALID_PARAMETER != 0 {
                return AUDIODEC_ERROR_ARG;
            }
            if r.result & RESULT_FATAL != 0 {
                return AUDIODEC_ERROR_API_FAIL;
            }
            return if r.result & RESULT_NOT_ENOUGH_ROOM != 0 { AUDIODEC_ERROR_INVALID_PCM_SIZE } else { 0 };
        }

        let wanted = d.frame_samples * d.channels * word_size_bytes(d.word_size);
        let produced = pcm_size.min(wanted);
        let consumed = au_size.min(d.frame_bytes);
        ctx.write_bytes(pcm_addr, &vec![0u8; produced as usize]);
        ctx.write_u32(c.pcm_item + 16, produced);
        ctx.write_u32(c.au_info + 16, consumed);
        fill_info(ctx, c.bsi, d, 0);
        0
    })
}

/// sceAudiodecClearContext(int handle)
fn cb_clear_context(ctx: &mut CpuContext) -> HleResult {
    let Some(slot) = handle_slot(ctx.rdi() as i32) else { return AUDIODEC_ERROR_INVALID_HANDLE };
    with_state(|s| match s.decoders[slot].as_mut() {
        None => AUDIODEC_ERROR_INVALID_HANDLE,
        Some(d) => {
            if let Some(a) = d.aac.as_mut() {
                a.reset();
            }
            0
        }
    })
}

const LIB: &str = "libSceAudiodec";

pub(crate) static EXPORTS: &[Export] = &[
    ("VjhsmxpcezI", "sceAudiodecInitLibrary", LIB, cb_init_library),
    ("h5jSB2QIDV0", "sceAudiodecTermLibrary", LIB, cb_term_library),
    ("O3f1sLMWRvs", "sceAudiodecCreateDecoder", LIB, cb_create_decoder),
    ("Tp+ZEy69mLk", "sceAudiodecDeleteDecoder", LIB, cb_delete_decoder),
    ("KHXHMDLkILw", "sceAudiodecDecode", LIB, cb_decode),
    ("6Vf9WTLDoss", "sceAudiodecClearContext", LIB, cb_clear_context),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister;

    #[test]
    fn mp3_decoder_lifecycle_and_validation() {
        const BASE: u64 = 0x5CE0_5000_0000;
        let mut ctx = CpuContext::new(crate::sync::test_util::vm_arena(BASE));
        let set = |ctx: &mut CpuContext, a: u64, b: u64| {
            ctx.set(CpuRegister::Rdi, a);
            ctx.set(CpuRegister::Rsi, b);
        };
        let (ctrl, param, bsi, au, pcm) = (BASE, BASE + 0x100, BASE + 0x200, BASE + 0x300, BASE + 0x400);
        set(&mut ctx, 0, TYPE_MP3 as u64);
        assert_eq!(cb_create_decoder(&mut ctx), AUDIODEC_ERROR_INVALID_CTRL_POINTER);
        set(&mut ctx, 7, 0);
        assert_eq!(cb_init_library(&mut ctx), AUDIODEC_ERROR_INVALID_TYPE);

        for (o, v) in [(0, param), (8, bsi), (16, au), (24, pcm)] {
            ctx.write_u64(ctrl + o, v);
        }
        ctx.write_u32(param, PARAM_MP3_SIZE);
        ctx.write_u32(param + 4, WORD_SZ_16BIT as u32);
        ctx.write_u32(bsi, MP3_INFO_SIZE + 1);
        set(&mut ctx, ctrl, TYPE_MP3 as u64);
        assert_eq!(cb_create_decoder(&mut ctx), AUDIODEC_ERROR_INVALID_BSI_INFO_SIZE);
        ctx.write_u32(bsi, MP3_INFO_SIZE);
        // Word sizes a PS5 refuses: 24-bit, float for MP3, unknown ones.
        for w in [WORD_SZ_24BIT, WORD_SZ_FLOAT, 3] {
            ctx.write_u32(param + 4, w as u32);
            assert_eq!(cb_create_decoder(&mut ctx), AUDIODEC_ERROR_INVALID_WORD_LENGTH, "word size {w}");
        }
        ctx.write_u32(param + 4, WORD_SZ_16BIT as u32);
        // The library must be initialized first.
        assert_eq!(cb_create_decoder(&mut ctx), AUDIODEC_ERROR_ARG);
        set(&mut ctx, TYPE_MP3 as u64, 0);
        assert_eq!(cb_init_library(&mut ctx), 0);
        set(&mut ctx, ctrl, TYPE_MP3 as u64);
        let handle = cb_create_decoder(&mut ctx);
        assert!(handle > 0);

        // Decode: one silent frame, sizes written back.
        ctx.write_u32(au, ITEM_SIZE);
        ctx.write_u64(au + 8, BASE + 0x1000);
        ctx.write_u32(au + 16, 5000);
        ctx.write_u32(pcm, ITEM_SIZE);
        ctx.write_u64(pcm + 8, BASE + 0x4000);
        ctx.write_u32(pcm + 16, 0x10000);
        ctx.write_bytes(BASE + 0x4000, &[0x55; 1152 * 4 + 4]);
        set(&mut ctx, handle as u64, ctrl);
        assert_eq!(cb_decode(&mut ctx), 0);
        assert_eq!(ctx.read_u32(au + 16), Some(MP3_MAX_FRAME_SIZE));
        assert_eq!(ctx.read_u32(pcm + 16), Some(1152 * 4));
        let out = ctx.read_bytes(BASE + 0x4000, 1152 * 4 + 4).unwrap();
        assert!(out[..1152 * 4].iter().all(|&b| b == 0));
        assert_eq!(out[1152 * 4], 0x55);
        ctx.write_u32(au + 16, 0);
        assert_eq!(cb_decode(&mut ctx), AUDIODEC_ERROR_INVALID_AU_SIZE);

        set(&mut ctx, handle as u64, 0);
        assert_eq!(cb_clear_context(&mut ctx), 0);
        assert_eq!(cb_delete_decoder(&mut ctx), 0);
        assert_eq!(cb_delete_decoder(&mut ctx), AUDIODEC_ERROR_INVALID_HANDLE);
        set(&mut ctx, 0, 0);
        assert_eq!(cb_decode(&mut ctx), AUDIODEC_ERROR_INVALID_HANDLE);
    }

    #[test]
    fn at9_info_block() {
        const BASE: u64 = 0x5CE0_6000_0000;
        let mut ctx = CpuContext::new(crate::sync::test_util::vm_arena(BASE));
        let (ctrl, param, bsi) = (BASE, BASE + 0x100, BASE + 0x200);
        ctx.write_u64(ctrl, param);
        ctx.write_u64(ctrl + 8, bsi);
        ctx.write_u32(param, PARAM_AT9_SIZE);
        ctx.write_u32(param + 4, 7);
        ctx.write_u32(bsi, AT9_INFO_SIZE);
        ctx.set(CpuRegister::Rdi, TYPE_AT9 as u64);
        assert_eq!(cb_init_library(&mut ctx), 0);
        ctx.set(CpuRegister::Rdi, ctrl);
        ctx.set(CpuRegister::Rsi, TYPE_AT9 as u64);
        assert_eq!(cb_create_decoder(&mut ctx), AUDIODEC_ERROR_INVALID_WORD_LENGTH);
        ctx.write_u32(param + 4, WORD_SZ_FLOAT as u32);
        let h = cb_create_decoder(&mut ctx);
        assert!(h > 0);
        let b = ctx.read_bytes(bsi, 36).unwrap();
        let w = |i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!((w(1), w(3), w(4), w(5), w(7)), (2, 48000, 2048, 1, 256));
        ctx.set(CpuRegister::Rdi, h as u64);
        assert_eq!(cb_delete_decoder(&mut ctx), 0);
    }
}
