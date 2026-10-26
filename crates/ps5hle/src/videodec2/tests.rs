// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Access-unit decoding of the reference clips (H.264, HEVC, VP9), Annex B
//! conversion of length-prefixed units and the failure paths.

use super::clips::*;
use super::*;
use crate::avplayer::{arena_alloc, arena_free, host};
use crate::context::CpuRegister;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 40;
const FRAME_BYTES: u64 = 1 << 20;

/// The decoders are host FFmpeg contexts; one clip at a time keeps the
/// pictures map free of other tests' frame buffers.
pub(super) static LOCK: Mutex<()> = Mutex::new(());

struct Env {
    vm: ps5vm::VirtualMemory,
    scratch: u64,
    au: u64,
    frame: u64,
}

impl Env {
    fn new() -> Self {
        Self {
            vm: ps5vm::VirtualMemory::new(),
            scratch: arena_alloc(0x1000).unwrap(),
            au: arena_alloc(0x10000).unwrap(),
            frame: arena_alloc(FRAME_BYTES).unwrap(),
        }
    }

    fn ctx(&self) -> CpuContext<'_> {
        CpuContext::new(&self.vm)
    }

    fn call(&self, name: &str, args: &[u64]) -> HleResult {
        let cb = EXPORTS.iter().find(|e| e.1 == name).unwrap().3;
        let mut ctx = self.ctx();
        const REGS: [CpuRegister; 4] = [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx];
        for (r, v) in REGS.iter().zip(args) {
            ctx.set(*r, *v);
        }
        cb(&mut ctx)
    }

    /// A decoder for `codec`, at most 64x40.
    fn create(&self, codec: u32, profile: u32) -> Result<u64, HleResult> {
        let ctx = self.ctx();
        let (config, memory, handle) = (self.scratch, self.scratch + 0x100, self.scratch + 0x200);
        ctx.write_bytes(self.scratch, &[0u8; 0x300]);
        ctx.write_u64(config, DECODER_CONFIG_INFO_SIZE);
        ctx.write_u32(config + 8, RESOURCE_TYPE_COMPUTE);
        ctx.write_u32(config + 12, codec);
        ctx.write_u32(config + 16, profile);
        ctx.write_u32(config + 24, WIDTH);
        ctx.write_u32(config + 28, HEIGHT);
        ctx.write_u32(config + 32, 4);
        ctx.write_u32(config + 36, 1);
        ctx.write_u64(config + 40, 0xC0FFEE);
        ctx.write_u64(memory, DECODER_MEMORY_INFO_SIZE);
        let r = self.call("sceVideodec2QueryDecoderMemoryInfo", &[config, memory]);
        if r != OK {
            return Err(r);
        }
        for off in [16, 32, 48] {
            ctx.write_u64(memory + off, 0x1000);
        }
        match self.call("sceVideodec2CreateDecoder", &[config, memory, handle]) {
            OK => Ok(ctx.read_u64(handle).unwrap()),
            e => Err(e),
        }
    }

    fn frame_buffer(&self, size: u64) -> u64 {
        let ctx = self.ctx();
        let fb = self.scratch + 0x300;
        ctx.write_u64(fb, FRAME_BUFFER_SIZE);
        ctx.write_u64(fb + 8, self.frame);
        ctx.write_u64(fb + 16, size);
        ctx.write_bytes(fb + 24, &[0xAA]);
        let out = self.scratch + 0x340;
        ctx.write_bytes(out, &[0u8; OUTPUT_INFO_SIZE as usize]);
        ctx.write_u64(out, OUTPUT_INFO_SIZE);
        fb
    }

    fn decode(&self, handle: u64, unit: &[u8], pts: u64, dts: u64, attached: u64) -> HleResult {
        let ctx = self.ctx();
        ctx.write_bytes(self.au, unit);
        let input = self.scratch + 0x380;
        ctx.write_u64(input, INPUT_DATA_SIZE);
        ctx.write_u64(input + 8, self.au);
        ctx.write_u64(input + 16, unit.len() as u64);
        ctx.write_u64(input + 24, pts);
        ctx.write_u64(input + 32, dts);
        ctx.write_u64(input + 40, attached);
        let fb = self.frame_buffer(FRAME_BYTES);
        self.call("sceVideodec2Decode", &[handle, input, fb, self.scratch + 0x340])
    }

    fn flush(&self, handle: u64) -> HleResult {
        let fb = self.frame_buffer(FRAME_BYTES);
        self.call("sceVideodec2Flush", &[handle, fb, self.scratch + 0x340])
    }

    /// The output info of the last call, if it carries a picture.
    fn picture(&self, codec: u32) -> Option<u64> {
        let ctx = self.ctx();
        let out = self.scratch + 0x340;
        let valid = ctx.read_bytes(out + 8, 1).unwrap()[0] != 0;
        let accepted = ctx.read_bytes(self.scratch + 0x300 + 24, 1).unwrap()[0] != 0;
        assert_eq!(valid, accepted, "frame buffer acceptance disagrees with the output");
        if !valid {
            return None;
        }
        assert_eq!(ctx.read_bytes(out + 9, 1).unwrap()[0], 0, "error frame");
        assert_eq!(ctx.read_u32(out + 12), Some(codec));
        assert_eq!((ctx.read_u32(out + 16), ctx.read_u32(out + 20), ctx.read_u32(out + 24)), (Some(WIDTH), Some(256), Some(HEIGHT)));
        assert_eq!(ctx.read_u64(out + 32), Some(self.frame));
        Some(out)
    }

    /// FNV-1a over the visible NV12 rows.
    fn hash_nv12(&self) -> u64 {
        let ctx = self.ctx();
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let rows = (0..HEIGHT).map(|r| r as u64 * 256).chain((0..HEIGHT / 2).map(|r| (HEIGHT as u64 + r as u64) * 256));
        for row in rows {
            for b in ctx.read_bytes(self.frame + row, WIDTH as usize).unwrap() {
                hash = (hash ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    /// Decodes `units` and the flushed tail; returns the picture hashes and
    /// the output info address after each picture is checked by `check`.
    fn run_clip(&self, codec: u32, profile: u32, units: &[Vec<u8>], mut check: impl FnMut(&Env, u64, usize)) -> Vec<u64> {
        let handle = self.create(codec, profile).expect("decoder creation failed");
        let mut hashes = Vec::new();
        let mut take = |env: &Env, hashes: &mut Vec<u64>| {
            let Some(out) = env.picture(codec) else { return false };
            check(env, out, hashes.len());
            hashes.push(env.hash_nv12());
            true
        };
        for (n, unit) in units.iter().enumerate() {
            assert_eq!(self.decode(handle, unit, 1000 + n as u64, n as u64, 0xA0 + n as u64), OK, "decode failed");
            take(self, &mut hashes);
        }
        loop {
            assert_eq!(self.flush(handle), OK, "flush failed");
            if !take(self, &mut hashes) {
                break;
            }
        }
        assert_eq!(self.call("sceVideodec2DeleteDecoder", &[handle]), OK);
        hashes
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        arena_free(self.scratch);
        arena_free(self.au);
        arena_free(self.frame);
    }
}

/// Splits `stream` before each 4-byte start code whose NAL `is_delimiter`.
pub(super) fn split_units(stream: &[u8], is_delimiter: impl Fn(u8) -> bool) -> Vec<Vec<u8>> {
    let mut starts: Vec<usize> =
        (0..stream.len().saturating_sub(4)).filter(|&i| stream[i..i + 4] == [0, 0, 0, 1] && is_delimiter(stream[i + 4])).collect();
    starts.push(stream.len());
    starts.windows(2).map(|w| stream[w[0]..w[1]].to_vec()).collect()
}

/// The same access unit with 4-byte NAL lengths instead of start codes.
fn length_prefixed(unit: &[u8]) -> Vec<u8> {
    let mut nals = Vec::new();
    let mut i = 0;
    while i + 3 < unit.len() {
        if unit[i..i + 3] == [0, 0, 1] {
            nals.push(i + 3);
            i += 2;
        }
        i += 1;
    }
    let mut out = Vec::new();
    for (n, &start) in nals.iter().enumerate() {
        let mut end = nals.get(n + 1).map_or(unit.len(), |&next| next - 3);
        while end > start && unit[end - 1] == 0 {
            end -= 1;
        }
        out.extend_from_slice(&((end - start) as u32).to_be_bytes());
        out.extend_from_slice(&unit[start..end]);
    }
    out
}

pub(super) fn avc_units() -> Vec<Vec<u8>> {
    split_units(&AVC_STREAM, |nal| nal & 0x1F == 9)
}

fn hevc_units() -> Vec<Vec<u8>> {
    split_units(&HEVC_STREAM, |nal| (nal >> 1) & 0x3F == 35)
}

fn ivf_frames() -> Vec<Vec<u8>> {
    let mut frames = Vec::new();
    let mut i = 32;
    while i + 12 <= VP9_STREAM.len() {
        let len = u32::from_le_bytes(VP9_STREAM[i..i + 4].try_into().unwrap()) as usize;
        assert!(i + 12 + len <= VP9_STREAM.len(), "truncated IVF frame");
        frames.push(VP9_STREAM[i + 12..i + 12 + len].to_vec());
        i += 12 + len;
    }
    frames
}

#[test]
fn annex_b_conversion() {
    let to_annex_b = |data: &[u8]| super::to_annex_b(data, &mut Framing::default()).map(|c| c.into_owned());
    let start = [0, 0, 0, 1, 0x65, 1, 2];
    assert_eq!(to_annex_b(&start).as_deref(), Some(&start[..]));
    assert_eq!(to_annex_b(&[0, 0, 1, 0x65]).as_deref(), Some(&[0, 0, 1, 0x65][..]));
    let prefixed = [0, 0, 0, 2, 0x67, 0x64, 0, 0, 0, 1, 0x68];
    assert_eq!(to_annex_b(&prefixed).as_deref(), Some(&[0, 0, 0, 1, 0x67, 0x64, 0, 0, 0, 1, 0x68][..]));
    // Lengths past the end, zero lengths and truncated length fields.
    assert_eq!(to_annex_b(&[0, 0, 0, 8, 0x65]), None);
    assert_eq!(to_annex_b(&[0, 0, 0, 0, 0x65]), None);
    assert_eq!(to_annex_b(&[0, 0, 0, 1]).as_deref(), Some(&[0, 0, 0, 1][..]));
    assert_eq!(to_annex_b(&[0, 0, 0, 2, 0x65]), None);
    // A converted unit starts with a 4-byte start code before every NAL.
    for unit in avc_units().iter().chain(&hevc_units()) {
        let prefixed = length_prefixed(unit);
        let converted = to_annex_b(&prefixed).unwrap();
        assert_eq!(converted.len(), prefixed.len());
        assert_eq!(converted[..4], [0, 0, 0, 1]);
    }
}

/// A unit that reads both as Annex B and as length-prefixed NAL units is
/// taken the way the decoder's earlier units were framed.
#[test]
fn ambiguous_units_follow_the_stream() {
    // [0,0,1,1] [x]: a 4-byte start code? no; a 3-byte one, and also a
    // length of 0x101 ... use a unit whose length prefix is 0x00000001.
    let ambiguous = [0, 0, 0, 1, 0x65];
    let mut f = Framing::default();
    assert_eq!(to_annex_b(&ambiguous, &mut f).as_deref(), Some(&ambiguous[..]), "Annex B by default");
    let mut f = Framing::default();
    let prefixed = [0, 0, 0, 2, 0x67, 0x64];
    assert_eq!(to_annex_b(&prefixed, &mut f).as_deref(), Some(&[0, 0, 0, 1, 0x67, 0x64][..]));
    assert_eq!(to_annex_b(&ambiguous, &mut f).as_deref(), Some(&[0, 0, 0, 1, 0x65][..]), "length-prefixed after length-prefixed units");
    // A short start code that also tiles: length-prefixed unless the
    // stream used 3-byte start codes.
    let short = [0, 0, 1, 0x01, 0x65];
    let tiled: Vec<u8> = [&[0u8, 0, 1, 0x01][..], &[0x41; 0x101]].concat();
    assert!(tiles_as_avcc(&tiled));
    let mut f = Framing::default();
    assert_eq!(to_annex_b(&tiled, &mut f).map(|c| c.len()), Some(4 + 0x101), "converted");
    assert_eq!(to_annex_b(&short, &mut f).as_deref(), Some(&short[..]));
    assert_eq!(to_annex_b(&tiled, &mut f).as_deref(), Some(&tiled[..]), "3-byte start codes were seen");
    assert_eq!(to_annex_b(&[1, 2, 3], &mut f), None);
}

#[test]
fn malformed_units_and_unsupported_codecs() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let env = Env::new();
    assert_eq!(env.create(2, 100), Err(err(ERROR_CODEC_TYPE)));
    if !host::available() {
        return;
    }
    let handle = env.create(CODEC_AVC, 100).unwrap();
    assert_eq!(env.decode(handle, &[0, 0, 0, 8, 0x65], 0, 0, 0), err(ERROR_ACCESS_UNIT));
    assert_eq!(env.decode(handle, &[0, 0], 0, 0, 0), err(ERROR_ACCESS_UNIT));
    assert_eq!(env.call("sceVideodec2Reset", &[handle]), OK);
    assert_eq!(env.call("sceVideodec2DeleteDecoder", &[handle]), OK);
    assert_eq!(env.call("sceVideodec2DeleteDecoder", &[handle]), err(ERROR_DECODER_INSTANCE));
}

/// H.264 in start-code and length-prefixed form: pictures in display order
/// with their timestamps, AVC picture info and the reference pixels.
#[test]
fn decodes_avc_clip() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !host::available() {
        return;
    }
    const DISPLAY_ORDER: [u64; 6] = [0, 2, 3, 1, 5, 4];
    let env = Env::new();
    let units = avc_units();
    assert_eq!(units.len(), AVC_HASHES.len());
    for units in [units.clone(), units.iter().map(|u| length_prefixed(u)).collect()] {
        let hashes = env.run_clip(CODEC_AVC, 100, &units, |env, out, n| {
            let ctx = env.ctx();
            let info = env.scratch + 0x400;
            ctx.write_bytes(info, &[0u8; AVC_PICTURE_INFO_SIZE as usize]);
            ctx.write_u64(info, AVC_PICTURE_INFO_SIZE);
            assert_eq!(env.call("sceVideodec2GetAvcPictureInfo", &[out, info, 0]), OK);
            let b = ctx.read_bytes(info, AVC_PICTURE_INFO_SIZE as usize).unwrap();
            let unit = DISPLAY_ORDER[n];
            let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
            assert_eq!(b[8], 1, "picture info missing");
            assert_eq!((u64_at(16), u64_at(24), u64_at(32)), (1000 + unit, unit, 0xA0 + unit), "picture {n} out of display order");
            assert_eq!((b[40], b[41]), ((n == 0) as u8, 100), "picture {n} flags");
        });
        assert_eq!(hashes, AVC_HASHES, "AVC pictures differ from the reference");
    }
}

/// HEVC (start codes and length-prefixed) and VP9 (IVF frames as they are).
#[test]
fn decodes_hevc_and_vp9_clips() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !host::available() {
        return;
    }
    let env = Env::new();
    let hevc = hevc_units();
    assert_eq!(hevc.len(), HEVC_HASHES.len());
    let check = |env: &Env, out: u64, _: usize| {
        // The common picture info carries the timestamps.
        let ctx = env.ctx();
        let info = env.scratch + 0x400;
        ctx.write_bytes(info, &[0u8; 64]);
        ctx.write_u64(info, 64);
        assert_eq!(env.call("sceVideodec2GetPictureInfo", &[out, info, 0]), OK);
        assert_eq!(ctx.read_bytes(info + 8, 1).unwrap()[0], 1);
    };
    assert_eq!(env.run_clip(CODEC_HEVC, 1, &hevc, check), HEVC_HASHES, "HEVC pictures differ");
    let prefixed: Vec<Vec<u8>> = hevc.iter().map(|u| length_prefixed(u)).collect();
    assert_eq!(env.run_clip(CODEC_HEVC, 1, &prefixed, check), HEVC_HASHES, "length-prefixed HEVC pictures differ");
    let vp9 = ivf_frames();
    assert_eq!(vp9.len(), VP9_HASHES.len());
    // A VP9 picture answers the generic query with its VP9 information
    // (the size of SceVideodec2Vp9PictureInfo at least).
    let check_vp9 = |env: &Env, out: u64, _: usize| {
        let ctx = env.ctx();
        let info = env.scratch + 0x400;
        ctx.write_u64(info, 64);
        assert_eq!(env.call("sceVideodec2GetPictureInfo", &[out, info, 0]), err(ERROR_STRUCT_SIZE));
        ctx.write_bytes(info, &[0u8; 96]);
        ctx.write_u64(info, vp9::PICTURE_INFO_SIZE);
        assert_eq!(env.call("sceVideodec2GetPictureInfo", &[out, info, 0]), OK);
        let generic = ctx.read_bytes(info + 8, 80).unwrap();
        let specific = env.scratch + 0x500;
        ctx.write_bytes(specific, &[0u8; 96]);
        ctx.write_u64(specific, vp9::PICTURE_INFO_SIZE);
        assert_eq!(env.call("sceVideodec2GetVp9PictureInfo", &[out, specific]), OK);
        assert_eq!(ctx.read_bytes(specific + 8, 80).unwrap(), generic);
    };
    assert_eq!(env.run_clip(CODEC_VP9, 0, &vp9, check_vp9), VP9_HASHES, "VP9 pictures differ");
}

/// A Main 10 HEVC stream: the picture is delivered as 8-bit NV12.
#[test]
fn ten_bit_hevc_is_delivered_as_nv12() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !host::available() {
        return;
    }
    let env = Env::new();
    for codec in [CODEC_HEVC, CODEC_VP9] {
        let handle = env.create(codec, 2).expect("ten-bit decoder creation failed");
        if codec == CODEC_HEVC {
            assert_eq!(env.decode(handle, &HEVC_MAIN10_STREAM, 0, 0, 0), OK);
            let mut pictures = env.picture(codec).is_some() as usize;
            loop {
                assert_eq!(env.flush(handle), OK);
                if env.picture(codec).is_none() {
                    break;
                }
                pictures += 1;
            }
            assert_eq!(pictures, 1);
        }
        assert_eq!(env.call("sceVideodec2DeleteDecoder", &[handle]), OK);
    }
}

impl Env {
    /// sceVideodec2GetVp9PictureInfo of the output at `out` into a guarded
    /// 88-byte structure; (result, structure).
    fn vp9_info(&self, out: u64) -> (HleResult, Vec<u8>) {
        let ctx = self.ctx();
        let info = self.scratch + 0x400;
        ctx.write_bytes(info, &[0xEE; 96]);
        ctx.write_u64(info, vp9::PICTURE_INFO_SIZE);
        let r = self.call("sceVideodec2GetVp9PictureInfo", &[out, info]);
        assert_eq!(ctx.read_bytes(info + 88, 8).unwrap(), [0xEE; 8], "wrote past the structure");
        assert_eq!(ctx.read_u64(info), Some(vp9::PICTURE_INFO_SIZE), "thisSize changed");
        (r, ctx.read_bytes(info, 88).unwrap())
    }
}

fn at32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn at64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

/// Header metadata of every picture of the VP9 clip, a superframe of two
/// shown frames, "show existing frame" and the rejected headers.
#[test]
fn vp9_picture_info() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !host::available() {
        return;
    }
    let env = Env::new();
    let units = ivf_frames();
    let check = |env: &Env, out: u64, n: usize| {
        let (r, info) = env.vp9_info(out);
        assert_eq!(r, OK);
        assert_eq!(info[8], 1, "picture {n} not valid");
        assert_eq!((at64(&info, 16), at64(&info, 24), at64(&info, 32)), (1000 + n as u64, n as u64, 0xA0 + n as u64));
        assert_eq!((at32(&info, 40), info[56], at32(&info, 60), info[57], at32(&info, 64), at32(&info, 52)), (0, 8, 0, 0, 1, 0));
        assert_eq!((at32(&info, 68), at32(&info, 72), at32(&info, 76), at32(&info, 80)), (WIDTH, HEIGHT, WIDTH, HEIGHT));
        assert_eq!(at32(&info, 48), (n == 0) as u32, "key frame flag of picture {n}");
    };
    assert_eq!(env.run_clip(CODEC_VP9, 0, &units, check), VP9_HASHES);

    let ctx = env.ctx();
    let handle = env.create(CODEC_VP9, 0).unwrap();
    let out = env.scratch + 0x340;
    assert_eq!(env.call("sceVideodec2GetVp9PictureInfo", &[0, env.scratch + 0x400]), err(ERROR_ARGUMENT_POINTER));
    assert_eq!(env.call("sceVideodec2GetVp9PictureInfo", &[out, 0]), err(ERROR_ARGUMENT_POINTER));
    ctx.write_u64(env.scratch + 0x400, vp9::PICTURE_INFO_SIZE - 1);
    env.frame_buffer(FRAME_BYTES);
    assert_eq!(env.call("sceVideodec2GetVp9PictureInfo", &[out, env.scratch + 0x400]), err(ERROR_STRUCT_SIZE));
    // A frame buffer no decoder wrote: not valid.
    ctx.write_u64(out + 32, 0x1234);
    let (r, info) = env.vp9_info(out);
    assert_eq!((r, info[8]), (OK, 0));

    // A superframe of the first two frames: one picture now, one at the flush.
    let mut superframe = [units[0].clone(), units[1].clone()].concat();
    superframe.push(0xC9);
    for u in &units[..2] {
        superframe.extend_from_slice(&(u.len() as u16).to_le_bytes());
    }
    superframe.push(0xC9);
    assert_eq!(env.decode(handle, &superframe, 9000, 9001, 9002), OK);
    assert!(env.picture(CODEC_VP9).is_some());
    assert_eq!(env.hash_nv12(), VP9_HASHES[0]);
    let (_, first) = env.vp9_info(out);
    assert_eq!((first[8], at32(&first, 64), at32(&first, 48)), (1, 2, 1));
    assert_eq!((at64(&first, 16), at64(&first, 24), at64(&first, 32)), (9000, 9001, 9002));
    assert_eq!(env.flush(handle), OK);
    assert!(env.picture(CODEC_VP9).is_some());
    assert_eq!(env.hash_nv12(), VP9_HASHES[1]);
    let (_, second) = env.vp9_info(out);
    assert_eq!((at32(&second, 64), at32(&second, 48), at64(&second, 16)), (2, 0, 9000));
    assert_eq!(env.call("sceVideodec2Reset", &[handle]), OK);
    let (_, info) = env.vp9_info(out);
    assert_eq!(info[8], 0, "reset kept the picture");

    // A hidden key frame is not presented; "show existing frame" shows it.
    let mut hidden = units[0].clone();
    hidden[0] &= !2;
    assert_eq!(env.decode(handle, &hidden, 9100, 0, 0), OK);
    assert!(env.picture(CODEC_VP9).is_none(), "hidden key frame presented");
    assert_eq!(env.decode(handle, &[0x88], 9200, 0, 0), OK);
    assert!(env.picture(CODEC_VP9).is_some());
    assert_eq!(env.hash_nv12(), VP9_HASHES[0]);
    let (_, info) = env.vp9_info(out);
    assert_eq!((info[8], at64(&info, 16), at32(&info, 48), at32(&info, 68)), (1, 9200, 0, WIDTH));

    // Headers the parser rejects.
    assert_eq!(env.call("sceVideodec2Reset", &[handle]), OK);
    for unit in [&[0x88u8][..], &[0], &[0x82], &[0xC1, 0, 0, 0xC1]] {
        assert_eq!(env.decode(handle, unit, 9300, 0, 0), err(ERROR_ACCESS_UNIT), "{unit:x?}");
    }
    assert_eq!(env.call("sceVideodec2DeleteDecoder", &[handle]), OK);
}
