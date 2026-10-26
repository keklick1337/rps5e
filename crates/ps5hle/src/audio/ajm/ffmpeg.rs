// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Raw audio packet decoding on the host FFmpeg shim (`csrc/media.c`,
//! psa_*), used by the AAC and Opus decoders. Without FFmpeg every open
//! fails and those codecs report a fatal codec error.

use super::decoder::PcmEncoding;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostCodec {
    Aac = 0,
    Opus = 1,
}

/// One decoded frame, already interleaved in the requested encoding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostFrame {
    pub channels: u32,
    pub sample_rate: u32,
    pub samples: u32,
    pub bit_rate: u64,
    /// Stream profile (AAC: 1 LC, 4 HE, 28 HE v2; -99 unknown).
    pub profile: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostError {
    Api,
    InvalidData,
    Other,
}

#[cfg(ps5_ffmpeg)]
mod ffi {
    #[repr(C)]
    #[derive(Default)]
    pub struct PsaFrame {
        pub channels: i32,
        pub sample_rate: i32,
        pub nb_samples: i32,
        pub bit_rate: i64,
        pub profile: i32,
    }

    #[repr(C)]
    pub struct PsaDecoder {
        _private: [u8; 0],
    }

    extern "C" {
        pub fn psa_open(codec: i32, channels: i32, sample_rate: i32) -> *mut PsaDecoder;
        pub fn psa_open_ex(codec: i32, channels: i32, sample_rate: i32, extradata: *const u8, extradata_size: i32) -> *mut PsaDecoder;
        pub fn psa_close(a: *mut PsaDecoder);
        pub fn psa_flush(a: *mut PsaDecoder);
        pub fn psa_set_output_rate(a: *mut PsaDecoder, rate: i32);
        pub fn psa_send(a: *mut PsaDecoder, data: *const u8, size: i32, extradata: *const u8, extradata_size: i32) -> i32;
        pub fn psa_receive(a: *mut PsaDecoder, fmt: i32, out: *mut PsaFrame) -> i32;
        pub fn psa_data(a: *const PsaDecoder) -> *const u8;
    }
}

pub struct HostAudioDecoder {
    #[cfg(ps5_ffmpeg)]
    raw: *mut ffi::PsaDecoder,
}

// The decoder is only used behind the AJM instance mutex.
unsafe impl Send for HostAudioDecoder {}

#[cfg(ps5_ffmpeg)]
impl HostAudioDecoder {
    pub fn open(codec: HostCodec, channels: u32, sample_rate: u32) -> Option<Self> {
        let raw = unsafe { ffi::psa_open(codec as i32, channels as i32, sample_rate as i32) };
        (!raw.is_null()).then_some(Self { raw })
    }

    /// Open with the codec's stream header (an OpusHead for multichannel Opus).
    pub fn open_with_header(codec: HostCodec, channels: u32, sample_rate: u32, header: &[u8]) -> Option<Self> {
        let raw = unsafe { ffi::psa_open_ex(codec as i32, channels as i32, sample_rate as i32, header.as_ptr(), header.len() as i32) };
        (!raw.is_null()).then_some(Self { raw })
    }

    pub fn flush(&mut self) {
        unsafe { ffi::psa_flush(self.raw) }
    }

    /// Resample the following frames to `rate` Hz (0 keeps the decoder's
    /// rate); the resampler's delay carries over between frames.
    pub fn set_output_rate(&mut self, rate: u32) {
        unsafe { ffi::psa_set_output_rate(self.raw, rate as i32) }
    }

    pub fn send(&mut self, packet: &[u8], extradata: Option<&[u8]>) -> Result<(), HostError> {
        if packet.len() > i32::MAX as usize {
            return Err(HostError::Other);
        }
        let (e, n) = extradata.map_or((std::ptr::null(), 0), |e| (e.as_ptr(), e.len() as i32));
        match unsafe { ffi::psa_send(self.raw, packet.as_ptr(), packet.len() as i32, e, n) } {
            0 => Ok(()),
            -1 => Err(HostError::Api),
            -2 => Err(HostError::InvalidData),
            _ => Err(HostError::Other),
        }
    }

    /// Next decoded frame; the PCM is in [`HostAudioDecoder::data`] until
    /// the next call.
    pub fn receive(&mut self, encoding: PcmEncoding) -> Result<Option<HostFrame>, HostError> {
        let mut f = ffi::PsaFrame::default();
        match unsafe { ffi::psa_receive(self.raw, encoding as i32, &mut f) } {
            1 => Ok(Some(HostFrame {
                channels: f.channels.max(1) as u32,
                sample_rate: f.sample_rate.max(1) as u32,
                samples: f.nb_samples.max(0) as u32,
                bit_rate: f.bit_rate.max(0) as u64,
                profile: f.profile,
            })),
            0 => Ok(None),
            -1 => Err(HostError::Api),
            _ => Err(HostError::InvalidData),
        }
    }

    pub fn data(&self, frame: &HostFrame, encoding: PcmEncoding) -> &[u8] {
        let len = frame.samples as usize * frame.channels as usize * encoding.bytes_per_sample();
        let p = unsafe { ffi::psa_data(self.raw) };
        if p.is_null() || len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(p, len) }
    }
}

#[cfg(ps5_ffmpeg)]
impl Drop for HostAudioDecoder {
    fn drop(&mut self) {
        unsafe { ffi::psa_close(self.raw) }
    }
}

#[cfg(not(ps5_ffmpeg))]
impl HostAudioDecoder {
    pub fn open(_codec: HostCodec, _channels: u32, _sample_rate: u32) -> Option<Self> {
        None
    }

    pub fn open_with_header(_codec: HostCodec, _channels: u32, _sample_rate: u32, _header: &[u8]) -> Option<Self> {
        None
    }

    pub fn flush(&mut self) {}

    pub fn set_output_rate(&mut self, _rate: u32) {}

    pub fn send(&mut self, _packet: &[u8], _extradata: Option<&[u8]>) -> Result<(), HostError> {
        Err(HostError::Api)
    }

    pub fn receive(&mut self, _encoding: PcmEncoding) -> Result<Option<HostFrame>, HostError> {
        Err(HostError::Api)
    }

    pub fn data(&self, _frame: &HostFrame, _encoding: PcmEncoding) -> &[u8] {
        &[]
    }
}
