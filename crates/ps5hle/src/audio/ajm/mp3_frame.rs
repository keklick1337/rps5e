// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! sceAjmDecMp3ParseFrame: the length and format of one MPEG audio frame,
//! and with `parseOfl` the original file length record (Xing/Info with an
//! optional LAME tag, VBRI, or the block that ends a Fraunhofer file).
//!
//! Result layout (AjmDecMp3ParseFrame, 40 bytes): u64 frame_size; u32
//! num_channels, samples_per_channel, bitrate, sample_rate, encoder_delay,
//! num_frames, total_samples, ofl_type.
//!
//! Like the reference model, the frame length is the layer III one and the
//! layer field is not checked. Two places differ from it: the MPEG-2.5
//! bitrate table is the full one (the reference stops at 64 kbit/s), and
//! the original-length records the reference cannot parse (VBRI outside
//! MPEG-1 stereo, layers other than III, a CRC-protected header) are left
//! out, so `ofl_type` stays NONE instead of throwing.

use crate::context::CpuContext;
use crate::registry::HleResult;

use super::AJM_ERROR_INVALID_PARAMETER;

pub const OFL_NONE: u32 = 0;
pub const OFL_LAME: u32 = 1;
pub const OFL_VBRI: u32 = 2;
pub const OFL_FGH: u32 = 3;
pub const OFL_VBRI_AND_FGH: u32 = 4;

/// Bytes of the title's buffer read at most (a frame and the next header fit easily).
const MAX_READ: u32 = 0x4_0000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParsedFrame {
    pub frame_size: u64,
    pub num_channels: u32,
    pub samples_per_channel: u32,
    pub bitrate: u32,
    pub sample_rate: u32,
    pub encoder_delay: u32,
    pub num_frames: u32,
    pub total_samples: u32,
    pub ofl_type: u32,
}

/// Sample rates by MPEG version index (0 = 2.5, 2 = 2, 3 = 1; 1 is reserved).
const SAMPLE_RATES: [[u32; 4]; 4] = [[11025, 12000, 8000, 0], [0; 4], [22050, 24000, 16000, 0], [44100, 48000, 32000, 0]];
/// Bitrates in kbit/s by MPEG version index, layer III columns.
const KBPS: [[u32; 16]; 4] = [
    [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0],
    [0; 16],
    [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0],
    [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0],
];

fn be(b: &[u8], n: usize) -> u32 {
    b[..n].iter().fold(0u32, |v, &x| (v << 8) | x as u32)
}

/// The frame header fields and the frame length (no original-length record).
fn parse_frame_header(stream: &[u8], stream_size: u32) -> Option<(ParsedFrame, u32)> {
    if stream.len() < 4 || stream_size < 4 {
        return None;
    }
    let header = be(stream, 4);
    let version = ((header >> 19) & 3) as usize;
    let sample_rate = SAMPLE_RATES[version][((header >> 10) & 3) as usize];
    let bitrate = KBPS[version][((header >> 12) & 15) as usize] * 1000;
    let valid = (header >> 21) == 0x7FF && sample_rate != 0 && bitrate != 0;
    if !valid {
        return None;
    }
    let mpeg1 = version == 3;
    let frame = ParsedFrame {
        frame_size: (if mpeg1 { 144u64 } else { 72 }) * bitrate as u64 / sample_rate as u64 + ((header >> 9) & 1) as u64,
        num_channels: if ((header >> 6) & 3) == 3 { 1 } else { 2 },
        samples_per_channel: if mpeg1 { 1152 } else { 576 },
        bitrate,
        sample_rate,
        ..ParsedFrame::default()
    };
    Some((frame, header))
}

/// Parses one frame; `stream` holds the bytes from the frame start (at most
/// `stream_size` of them are meaningful). With `parse_ofl` the original length
/// record is read too.
pub fn parse_frame(stream: &[u8], stream_size: u32, parse_ofl: bool) -> Option<ParsedFrame> {
    let (mut frame, header) = parse_frame_header(stream, stream_size)?;
    if parse_ofl {
        let layer3 = (header >> 17) & 3 == 1;
        let crc_free = (header >> 16) & 1 == 1;
        if layer3 && crc_free {
            parse_ofl_record(stream, stream_size, header, &mut frame);
        } else {
            log::warn!("[ajm] sceAjmDecMp3ParseFrame: original length record of this frame is not parsed");
        }
    }
    Some(frame)
}

/// The original length record after the frame header (offset 4 onwards).
fn parse_ofl_record(stream: &[u8], stream_size: u32, header: u32, frame: &mut ParsedFrame) {
    let end = (stream_size as u64).min(frame.frame_size).min(stream.len() as u64) as usize;
    if end <= 4 {
        return;
    }
    let available = end - 4;
    let mut bit = 0usize;
    let read_bits = |count: u32, bit: &mut usize| -> u32 {
        let mut value = 0u32;
        for _ in 0..count {
            let b = if *bit / 8 < available { (stream[4 + *bit / 8] >> (7 - *bit % 8)) & 1 } else { 0 };
            value = (value << 1) | b as u32;
            *bit += 1;
        }
        value
    };
    let mpeg1 = (header >> 19) & 3 == 3;
    let mono = (header >> 6) & 3 == 3;
    // Side information up to the per-granule parts, which hold part2_3_length.
    bit += match (mpeg1, mono) {
        (true, true) => 18,
        (true, false) => 20,
        (false, true) => 9,
        (false, false) => 10,
    };
    let mut main_data_bits = 0u32;
    for _ in 0..(if mpeg1 { 2 } else { 1 }) {
        for _ in 0..frame.num_channels {
            main_data_bits = main_data_bits.wrapping_add(read_bits(12, &mut bit));
            bit += if mpeg1 { 47 } else { 51 };
        }
    }
    bit += main_data_bits as usize;
    // (bit + 7) / 8 bytes, i.e. bit.div_ceil(8).
    let at = 4 + bit.div_ceil(8).min(available);

    if at + 8 <= end && (&stream[at..at + 4] == b"Xing" || &stream[at..at + 4] == b"Info") {
        let flags = stream[at + 7];
        let mut field = at + 8;
        let mut frames = 0u32;
        if flags & 0x01 != 0 {
            if field + 4 > end {
                return;
            }
            frames = be(&stream[field..], 4);
            frame.num_frames = frames;
            field += 4;
        }
        for (flag, skip) in [(0x02u8, 4usize), (0x04, 100), (0x08, 4)] {
            if flags & flag != 0 {
                field += skip;
                if field > end {
                    return;
                }
            }
        }
        if field + 0x18 <= end && &stream[field..field + 4] == b"LAME" {
            let delay = ((stream[field + 0x15] as u32) << 4) | (stream[field + 0x16] as u32 >> 4);
            let padding = (((stream[field + 0x16] as u32) & 0x0F) << 8) | stream[field + 0x17] as u32;
            if flags & 0x01 != 0 && frames > 0 {
                frame.total_samples = frames
                    .wrapping_mul(frame.samples_per_channel)
                    .wrapping_sub(delay + padding);
            }
            frame.encoder_delay = frame.samples_per_channel.wrapping_add(delay).wrapping_add(529);
            frame.ofl_type = OFL_LAME;
        }
    } else if at != 36 && end >= 40 && &stream[36..40] == b"VBRI" {
        log::warn!("[ajm] sceAjmDecMp3ParseFrame: VBRI header outside MPEG-1 stereo is not parsed");
    } else if at + 26 <= end && &stream[at..at + 4] == b"VBRI" {
        frame.encoder_delay = be(&stream[at + 6..], 2);
        frame.ofl_type = OFL_VBRI;
        if frame.frame_size <= stream_size as u64 && frame.frame_size as usize <= stream.len() {
            let next_at = frame.frame_size as usize;
            if let Some(next) = parse_frame(&stream[next_at..], stream_size - next_at as u32, true) {
                if next.ofl_type == OFL_FGH {
                    frame.encoder_delay = frame.encoder_delay.wrapping_add(next.encoder_delay);
                    frame.total_samples = next.total_samples;
                    frame.ofl_type = OFL_VBRI_AND_FGH;
                }
            }
        }
    } else if at + 10 <= end {
        let mut at = at;
        while at + 9 < end && stream[at] != 0xB4 {
            at += 1;
        }
        if at + 9 >= end {
            return;
        }
        let mut crc = 0xFFu8;
        for &byte in &stream[at..at + 9] {
            for shift in (0..8).rev() {
                let same = ((crc >> 7) & 1) == ((byte >> shift) & 1);
                crc = if same { (crc << 1) ^ 0x45 } else { crc << 1 };
            }
        }
        if stream[at + 9] != crc {
            return;
        }
        frame.encoder_delay = be(&stream[at + 1..], 2);
        frame.total_samples = be(&stream[at + 3..], 4);
        frame.ofl_type = OFL_FGH;
    }
}

/// sceAjmDecMp3ParseFrame(const u8* stream, u32 streamSize, int parseOfl, AjmDecMp3ParseFrame* frame)
pub(crate) fn cb_parse_frame(ctx: &mut CpuContext) -> HleResult {
    let (stream, size, parse_ofl, out) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx() as i32, ctx.rcx());
    if stream == 0 || out == 0 || size < 4 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let Some(data) = ctx.read_bytes(stream, size.min(MAX_READ) as usize) else {
        return AJM_ERROR_INVALID_PARAMETER;
    };
    let Some(f) = parse_frame(&data, size, parse_ofl != 0) else {
        return AJM_ERROR_INVALID_PARAMETER;
    };
    let mut buf = [0u8; 40];
    buf[0..8].copy_from_slice(&f.frame_size.to_le_bytes());
    let fields = [
        f.num_channels,
        f.samples_per_channel,
        f.bitrate,
        f.sample_rate,
        f.encoder_delay,
        f.num_frames,
        f.total_samples,
        f.ofl_type,
    ];
    for (i, v) in fields.iter().enumerate() {
        buf[8 + i * 4..12 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    if ctx.write_bytes(out, &buf) {
        0
    } else {
        AJM_ERROR_INVALID_PARAMETER
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MPEG-1 Layer III, 128 kbit/s, 44.1 kHz, joint stereo: frame length 417.
    const HEADER: [u8; 4] = [0xFF, 0xFB, 0x90, 0x64];

    fn frame_bytes(len: usize) -> Vec<u8> {
        let mut v = vec![0u8; len];
        v[..4].copy_from_slice(&HEADER);
        v
    }

    #[test]
    fn nid_matches_name() {
        assert_eq!(ps5nid::compute_nid("sceAjmDecMp3ParseFrame"), "eDFeTyi+G3Y");
    }

    #[test]
    fn header_fields_and_frame_length() {
        let f = parse_frame(&frame_bytes(417), 417, false).unwrap();
        assert_eq!(f.frame_size, 417);
        assert_eq!((f.num_channels, f.samples_per_channel), (2, 1152));
        assert_eq!((f.bitrate, f.sample_rate), (128_000, 44_100));
        assert_eq!(f.ofl_type, OFL_NONE);
        // The padding bit adds one byte.
        let mut padded = frame_bytes(418);
        padded[2] |= 0x02;
        assert_eq!(parse_frame(&padded, 418, false).unwrap().frame_size, 418);
    }

    #[test]
    fn invalid_headers_are_rejected() {
        assert!(parse_frame(&[0xFF, 0xFB, 0xF0, 0x00, 0, 0], 6, false).is_none(), "bitrate index 15");
        assert!(parse_frame(&[0xFF, 0xFB, 0x9C, 0x00, 0, 0], 6, false).is_none(), "reserved sample rate");
        assert!(parse_frame(&[0xFF, 0xFB, 0x90], 3, false).is_none(), "shorter than a header");
        assert!(parse_frame(&[0x00; 417], 417, false).is_none(), "no sync word");
    }

    #[test]
    fn xing_header_with_lame_tag() {
        let mut b = frame_bytes(417);
        b[36..40].copy_from_slice(b"Xing");
        b[43] = 0x01; // frames field only
        b[44..48].copy_from_slice(&1000u32.to_be_bytes());
        b[48..52].copy_from_slice(b"LAME");
        b[0x15 + 48] = 0x00;
        b[0x16 + 48] = 0x30; // delay 3
        b[0x17 + 48] = 0x05; // padding 5
        let f = parse_frame(&b, 417, true).unwrap();
        assert_eq!(f.ofl_type, OFL_LAME);
        assert_eq!(f.num_frames, 1000);
        assert_eq!(f.total_samples, 1000 * 1152 - (3 + 5));
        assert_eq!(f.encoder_delay, 1152 + 3 + 529);
    }

    #[test]
    fn fraunhofer_block_needs_a_good_crc() {
        let mut b = frame_bytes(417);
        b[40] = 0xB4;
        b[41..43].copy_from_slice(&291u16.to_be_bytes());
        b[43..47].copy_from_slice(&0x00AB_CDEFu32.to_be_bytes());
        let crc_of = |buf: &[u8]| {
            let mut crc = 0xFFu8;
            for &byte in &buf[40..49] {
                for shift in (0..8).rev() {
                    let same = ((crc >> 7) & 1) == ((byte >> shift) & 1);
                    crc = if same { (crc << 1) ^ 0x45 } else { crc << 1 };
                }
            }
            crc
        };
        let good = crc_of(&b);
        let mut bad = b.clone();
        bad[49] = good ^ 1;
        assert_eq!(parse_frame(&bad, 417, true).unwrap().ofl_type, OFL_NONE, "block with a wrong CRC accepted");
        b[49] = good;
        let f = parse_frame(&b, 417, true).unwrap();
        assert_eq!(f.ofl_type, OFL_FGH);
        assert_eq!(f.encoder_delay, 291);
        assert_eq!(f.total_samples, 0x00AB_CDEF);
    }

    #[test]
    fn vbri_header_is_read_at_its_offset() {
        let mut b = frame_bytes(417);
        b[36..40].copy_from_slice(b"VBRI");
        b[42..44].copy_from_slice(&77u16.to_be_bytes());
        let f = parse_frame(&b, 417, true).unwrap();
        assert_eq!(f.ofl_type, OFL_VBRI);
        assert_eq!(f.encoder_delay, 77);
    }

    #[test]
    fn guest_call_writes_the_result() {
        const BASE: u64 = 0x7E30_0000_0000;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        assert!(ctx.write_bytes(BASE, &frame_bytes(417)));
        ctx.set(crate::context::CpuRegister::Rdi, BASE);
        ctx.set(crate::context::CpuRegister::Rsi, 417);
        ctx.set(crate::context::CpuRegister::Rdx, 0);
        ctx.set(crate::context::CpuRegister::Rcx, BASE + 0x100);
        assert_eq!(cb_parse_frame(&mut ctx), 0);
        let out = ctx.read_bytes(BASE + 0x100, 40).unwrap();
        assert_eq!(u64::from_le_bytes(out[0..8].try_into().unwrap()), 417);
        assert_eq!(u32::from_le_bytes(out[8..12].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(out[12..16].try_into().unwrap()), 1152);
        assert_eq!(u32::from_le_bytes(out[16..20].try_into().unwrap()), 128_000);
        assert_eq!(u32::from_le_bytes(out[20..24].try_into().unwrap()), 44_100);
        assert_eq!(u32::from_le_bytes(out[36..40].try_into().unwrap()), OFL_NONE);
        ctx.set(crate::context::CpuRegister::Rsi, 3);
        assert_eq!(cb_parse_frame(&mut ctx), AJM_ERROR_INVALID_PARAMETER, "buffer shorter than a header");
        let _ = vm.unmap(BASE, 0x10000);
    }
}
