// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Opus encoder (libopus, loaded at run time) for the DualSense Bluetooth
//! speaker packets.

use libloading::Library;
use std::ffi::{c_int, c_void};
use std::sync::{Arc, OnceLock};

const OPUS_OK: c_int = 0;
const OPUS_APPLICATION_AUDIO: c_int = 2049;
const OPUS_SET_BITRATE_REQUEST: c_int = 4002;
const OPUS_SET_VBR_REQUEST: c_int = 4006;
const OPUS_SET_COMPLEXITY_REQUEST: c_int = 4010;

/// Opus packet size of the controller's speaker sub-packet.
pub const PACKET_BYTES: usize = 200;

struct Api {
    _lib: Library,
    create: unsafe extern "C" fn(i32, c_int, c_int, *mut c_int) -> *mut c_void,
    ctl: unsafe extern "C" fn(*mut c_void, c_int, ...) -> c_int,
    encode_float: unsafe extern "C" fn(*mut c_void, *const f32, c_int, *mut u8, i32) -> i32,
    destroy: unsafe extern "C" fn(*mut c_void),
    decoder_create: unsafe extern "C" fn(i32, c_int, *mut c_int) -> *mut c_void,
    decode_float: unsafe extern "C" fn(*mut c_void, *const u8, i32, *mut f32, c_int, c_int) -> c_int,
    decoder_destroy: unsafe extern "C" fn(*mut c_void),
}

fn api() -> Option<Arc<Api>> {
    static API: OnceLock<Option<Arc<Api>>> = OnceLock::new();
    API.get_or_init(|| {
        let names: &[&str] = if cfg!(windows) {
            &["opus.dll", "libopus-0.dll"]
        } else if cfg!(target_os = "macos") {
            &["libopus.0.dylib", "libopus.dylib"]
        } else {
            &["libopus.so.0", "libopus.so"]
        };
        for n in names {
            // SAFETY: libopus has no load-time initialisers.
            let Ok(lib) = (unsafe { Library::new(n) }) else { continue };
            // SAFETY: the symbol types match the libopus API.
            let api = unsafe {
                Api {
                    create: *lib.get(b"opus_encoder_create\0").ok()?,
                    ctl: *lib.get(b"opus_encoder_ctl\0").ok()?,
                    encode_float: *lib.get(b"opus_encode_float\0").ok()?,
                    destroy: *lib.get(b"opus_encoder_destroy\0").ok()?,
                    decoder_create: *lib.get(b"opus_decoder_create\0").ok()?,
                    decode_float: *lib.get(b"opus_decode_float\0").ok()?,
                    decoder_destroy: *lib.get(b"opus_decoder_destroy\0").ok()?,
                    _lib: lib,
                }
            };
            return Some(Arc::new(api));
        }
        None
    })
    .clone()
}

/// Encodes stereo 48 kHz frames into fixed-size packets.
pub trait Encoder: Send {
    /// Encode `frames` stereo frames; returns the packet size or an error.
    fn encode(&mut self, pcm: &[f32], frames: usize, packet: &mut [u8]) -> i32;
}

pub struct OpusEncoder {
    api: Arc<Api>,
    enc: *mut c_void,
}

// SAFETY: the encoder state is used by one thread at a time.
unsafe impl Send for OpusEncoder {}

impl OpusEncoder {
    /// 48 kHz stereo, constant 160 kbit/s (200-byte packets per 10 ms),
    /// lowest complexity.
    pub fn new() -> Option<Self> {
        let api = api()?;
        let mut error = OPUS_OK;
        // SAFETY: valid arguments; the result is checked.
        let enc = unsafe { (api.create)(48000, 2, OPUS_APPLICATION_AUDIO, &mut error) };
        if enc.is_null() || error != OPUS_OK {
            return None;
        }
        let e = OpusEncoder { api, enc };
        // SAFETY: valid encoder and request codes.
        let ok = unsafe {
            (e.api.ctl)(e.enc, OPUS_SET_BITRATE_REQUEST, (PACKET_BYTES * 8 * 100) as i32) == OPUS_OK
                && (e.api.ctl)(e.enc, OPUS_SET_VBR_REQUEST, 0i32) == OPUS_OK
                && (e.api.ctl)(e.enc, OPUS_SET_COMPLEXITY_REQUEST, 0i32) == OPUS_OK
        };
        ok.then_some(e)
    }
}

impl Encoder for OpusEncoder {
    fn encode(&mut self, pcm: &[f32], frames: usize, packet: &mut [u8]) -> i32 {
        if pcm.len() < frames * 2 {
            return -1;
        }
        // SAFETY: buffers are large enough for `frames` and `packet.len()`.
        unsafe { (self.api.encode_float)(self.enc, pcm.as_ptr(), frames as c_int, packet.as_mut_ptr(), packet.len() as i32) }
    }
}

impl Drop for OpusEncoder {
    fn drop(&mut self) {
        // SAFETY: created by opus_encoder_create.
        unsafe { (self.api.destroy)(self.enc) };
    }
}

/// Decode one packet to stereo 48 kHz (tests); `None` without libopus.
pub fn decode(packet: &[u8], frames: usize) -> Option<Result<Vec<f32>, i32>> {
    let api = api()?;
    let mut error = OPUS_OK;
    // SAFETY: valid arguments; checked below.
    let dec = unsafe { (api.decoder_create)(48000, 2, &mut error) };
    if dec.is_null() || error != OPUS_OK {
        return Some(Err(error));
    }
    let mut out = vec![0.0f32; frames * 2];
    // SAFETY: `out` holds `frames` stereo frames.
    let n = unsafe { (api.decode_float)(dec, packet.as_ptr(), packet.len() as i32, out.as_mut_ptr(), frames as c_int, 0) };
    // SAFETY: created above.
    unsafe { (api.decoder_destroy)(dec) };
    Some(if n == frames as c_int { Ok(out) } else { Err(n) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_bitrate_packets_round_trip() {
        let Some(mut enc) = OpusEncoder::new() else { return };
        let pcm = vec![0.25f32; 960];
        let mut packet = [0u8; PACKET_BYTES];
        for _ in 0..3 {
            assert_eq!(enc.encode(&pcm, 480, &mut packet), PACKET_BYTES as i32, "packets are not 200 bytes");
        }
        let decoded = decode(&packet, 480).unwrap().expect("decode failed");
        let mean = decoded.iter().sum::<f32>() / decoded.len() as f32;
        assert!(mean > 0.05 && mean < 0.35, "mean {mean}");
    }
}
