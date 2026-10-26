// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! PCM conversion helpers for the host output streams.

/// Float [-1, 1] to s16 with asymmetric scale (32768 below zero, 32767
/// above), rounding half to even.
pub fn float_to_pcm16(value: f32) -> i16 {
    let scale = if value < 0.0 { 32768.0 } else { i16::MAX as f32 };
    (value * scale).round_ties_even().clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

/// Decode interleaved guest bytes (s16 or f32) to f32 samples in [-1, 1].
pub fn bytes_to_f32(source: &[u8], is_float: bool, out: &mut Vec<f32>) {
    out.clear();
    extend_f32(source, is_float, out);
}

/// Append interleaved guest bytes (s16 or f32), decoded as by `bytes_to_f32`.
pub fn extend_f32(source: &[u8], is_float: bool, out: &mut impl Extend<f32>) {
    if is_float {
        out.extend(source.chunks_exact(4).map(|c| {
            let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            if v.is_finite() { v } else { 0.0 }
        }));
    } else {
        out.extend(source.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix_to_pcm16() {
        assert_eq!(float_to_pcm16(1.0), 32767);
        assert_eq!(float_to_pcm16(-1.0), -32768);
        assert_eq!(float_to_pcm16(0.0), 0);
        assert_eq!(float_to_pcm16(0.5), 16384); // 16383.5 rounds to even
    }

    #[test]
    fn bytes_decode() {
        let mut out = Vec::new();
        let s16: Vec<u8> = [16384i16, -32768].iter().flat_map(|s| s.to_le_bytes()).collect();
        bytes_to_f32(&s16, false, &mut out);
        assert_eq!(out, vec![0.5, -1.0]);
        let f: Vec<u8> = [0.25f32, f32::NAN].iter().flat_map(|s| s.to_le_bytes()).collect();
        bytes_to_f32(&f, true, &mut out);
        assert_eq!(out, vec![0.25, 0.0]);
    }
}
