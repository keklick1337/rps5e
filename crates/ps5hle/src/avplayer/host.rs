// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Safe wrapper over the host media back end (`csrc/media.c`, FFmpeg).
//!
//! When the crate is built without FFmpeg every open fails, so callers
//! report "operation failed" and titles skip their movies.

use super::stream::FileStreamer;
use std::os::raw::{c_char, c_int, c_void};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaType {
    Video,
    Audio,
    Subtitle,
    Other,
}

#[derive(Debug, Clone)]
pub struct StreamInfo {
    pub media_type: MediaType,
    pub decodable: bool,
    /// The codec is allowed in a WebM source.
    pub webm_ok: bool,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub sample_rate: u32,
    pub frame_size: u32,
    pub fps: f64,
    /// Sample aspect ratio (0 when unknown).
    pub sar: f32,
    pub full_range: bool,
    pub color_primaries: u32,
    pub color_trc: u32,
    pub duration_ms: u64,
    pub language: [u8; 4],
}

/// Description of the frame a stream decoder just produced.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameMeta {
    pub width: u32,
    pub height: u32,
    pub ts_ms: u64,
    pub sar: f32,
    pub full_range: bool,
    pub crop: [u32; 4], // left, right, top, bottom
    pub nb_samples: u32,
    pub channels: u32,
    pub sample_rate: u32,
}

/// Picture produced by the raw access-unit decoder.
#[derive(Debug, Clone, Copy, Default)]
pub struct RawPicture {
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub crop: [u32; 4],
    pub profile: u32,
    pub level: u32,
    pub sar: (u32, u32),
    pub color_range: u8,
    pub color_primaries: u8,
    pub color_trc: u8,
    pub color_space: u8,
    pub key_frame: bool,
    pub corrupt: bool,
    pub pts: u64,
    pub dts: u64,
    pub attached: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawError {
    ApiFail,
    AccessUnit,
    FrameBufferSize,
}

/// True when host decoding is compiled in.
pub fn available() -> bool {
    cfg!(ps5_ffmpeg)
}

#[repr(C)]
#[derive(Default)]
pub(crate) struct PsmStreamInfo {
    media_type: i32,
    decodable: i32,
    webm_ok: i32,
    width: i32,
    height: i32,
    channels: i32,
    sample_rate: i32,
    frame_size: i32,
    fps: f64,
    sar: f32,
    full_range: i32,
    color_primaries: u32,
    color_trc: u32,
    duration_ms: u64,
    lang: [u8; 4],
}

#[repr(C)]
#[derive(Default)]
pub(crate) struct PsmFrame {
    width: i32,
    height: i32,
    ts_ms: u64,
    sar: f32,
    full_range: i32,
    crop_left: i32,
    crop_right: i32,
    crop_top: i32,
    crop_bottom: i32,
    nb_samples: i32,
    channels: i32,
    sample_rate: i32,
}

#[repr(C)]
#[derive(Default)]
pub(crate) struct PsvFrame {
    width: i32,
    height: i32,
    pitch: i32,
    crop_left: i32,
    crop_right: i32,
    crop_top: i32,
    crop_bottom: i32,
    profile: i32,
    level: i32,
    sar_num: i32,
    sar_den: i32,
    color_range: i32,
    color_primaries: i32,
    color_trc: i32,
    color_space: i32,
    key_frame: i32,
    corrupt: i32,
    pts: u64,
    dts: u64,
    attached: u64,
}

type ReadFn = extern "C" fn(*mut c_void, *mut u8, c_int) -> c_int;
type SeekFn = extern "C" fn(*mut c_void, i64, c_int) -> i64;

#[cfg(ps5_ffmpeg)]
mod ffi {
    use super::*;
    extern "C" {
        pub fn psm_io_new(opaque: *mut c_void, rd: ReadFn, sk: SeekFn, bufsize: c_int) -> *mut c_void;
        pub fn psm_io_free(io: *mut c_void);
        pub fn psm_io_read(io: *mut c_void, buf: *mut u8, len: c_int) -> c_int;
        pub fn psm_io_seek(io: *mut c_void, offset: i64, whence: c_int) -> i64;

        pub fn psm_open(
            name: *const c_char,
            opaque: *mut c_void,
            rd: ReadFn,
            sk: SeekFn,
            bufsize: c_int,
            err: *mut c_char,
            errlen: c_int,
        ) -> *mut c_void;
        pub fn psm_close(m: *mut c_void);
        pub fn psm_stream_count(m: *const c_void) -> c_int;
        pub fn psm_stream_info(m: *const c_void, index: c_int, out: *mut PsmStreamInfo) -> c_int;
        pub fn psm_duration_ms(m: *const c_void) -> u64;
        pub fn psm_set_discard(m: *mut c_void, video: c_int, audio: c_int);
        pub fn psm_seek(m: *mut c_void, stream: c_int, ms: u64) -> c_int;
        pub fn psm_read(m: *mut c_void, out: *mut *mut c_void, stream: *mut c_int, size: *mut c_int) -> c_int;
        pub fn psm_packet_free(p: *mut c_void);

        pub fn psm_decoder_open(m: *const c_void, index: c_int) -> *mut c_void;
        pub fn psm_decoder_close(d: *mut c_void);
        pub fn psm_decoder_params(d: *const c_void, channels: *mut c_int, frame_size: *mut c_int);
        pub fn psm_decoder_flush(d: *mut c_void);
        pub fn psm_decoder_send(d: *mut c_void, p: *const c_void) -> c_int;
        pub fn psm_decoder_receive(d: *mut c_void, out: *mut PsmFrame) -> c_int;
        pub fn psm_decoder_nv12(d: *mut c_void, dst: *mut u8, pitch: u32, rows: u32, full_range: c_int) -> c_int;
        pub fn psm_decoder_s16(d: *mut c_void, out: *mut *const i16) -> c_int;

        pub fn psv_open(codec: c_int) -> *mut c_void;
        pub fn psv_close(v: *mut c_void);
        pub fn psv_decode(
            v: *mut c_void,
            data: *const u8,
            size: i64,
            pts: u64,
            dts: u64,
            attached: u64,
            out: *mut PsvFrame,
        ) -> c_int;
        pub fn psv_flush(v: *mut c_void, out: *mut PsvFrame) -> c_int;
        pub fn psv_reset(v: *mut c_void);
        pub fn psv_copy_nv12(v: *mut c_void, dst: *mut u8, size: u64) -> c_int;
    }
}

/// Without FFmpeg: every open fails, nothing else is reachable.
#[cfg(not(ps5_ffmpeg))]
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
mod ffi {
    use super::*;
    pub unsafe fn psm_io_new(_: *mut c_void, _: ReadFn, _: SeekFn, _: c_int) -> *mut c_void {
        std::ptr::null_mut()
    }
    pub unsafe fn psm_io_free(_: *mut c_void) {}
    pub unsafe fn psm_io_read(_: *mut c_void, _: *mut u8, _: c_int) -> c_int {
        -1
    }
    pub unsafe fn psm_io_seek(_: *mut c_void, _: i64, _: c_int) -> i64 {
        -1
    }
    pub unsafe fn psm_open(
        _: *const c_char,
        _: *mut c_void,
        _: ReadFn,
        _: SeekFn,
        _: c_int,
        err: *mut c_char,
        errlen: c_int,
    ) -> *mut c_void {
        let msg = b"host media decoding (FFmpeg) is not compiled in\0";
        let n = msg.len().min(errlen.max(1) as usize);
        std::ptr::copy_nonoverlapping(msg.as_ptr() as *const c_char, err, n);
        *err.add(n - 1) = 0;
        std::ptr::null_mut()
    }
    pub unsafe fn psm_close(_: *mut c_void) {}
    pub unsafe fn psm_stream_count(_: *const c_void) -> c_int {
        0
    }
    pub unsafe fn psm_stream_info(_: *const c_void, _: c_int, _: *mut PsmStreamInfo) -> c_int {
        -1
    }
    pub unsafe fn psm_duration_ms(_: *const c_void) -> u64 {
        0
    }
    pub unsafe fn psm_set_discard(_: *mut c_void, _: c_int, _: c_int) {}
    pub unsafe fn psm_seek(_: *mut c_void, _: c_int, _: u64) -> c_int {
        -1
    }
    pub unsafe fn psm_read(_: *mut c_void, _: *mut *mut c_void, _: *mut c_int, _: *mut c_int) -> c_int {
        -1
    }
    pub unsafe fn psm_packet_free(_: *mut c_void) {}
    pub unsafe fn psm_decoder_open(_: *const c_void, _: c_int) -> *mut c_void {
        std::ptr::null_mut()
    }
    pub unsafe fn psm_decoder_close(_: *mut c_void) {}
    pub unsafe fn psm_decoder_params(_: *const c_void, _: *mut c_int, _: *mut c_int) {}
    pub unsafe fn psm_decoder_flush(_: *mut c_void) {}
    pub unsafe fn psm_decoder_send(_: *mut c_void, _: *const c_void) -> c_int {
        -1
    }
    pub unsafe fn psm_decoder_receive(_: *mut c_void, _: *mut PsmFrame) -> c_int {
        -1
    }
    pub unsafe fn psm_decoder_nv12(_: *mut c_void, _: *mut u8, _: u32, _: u32, _: c_int) -> c_int {
        -1
    }
    pub unsafe fn psm_decoder_s16(_: *mut c_void, _: *mut *const i16) -> c_int {
        -1
    }
    pub unsafe fn psv_open(_: c_int) -> *mut c_void {
        std::ptr::null_mut()
    }
    pub unsafe fn psv_close(_: *mut c_void) {}
    pub unsafe fn psv_decode(_: *mut c_void, _: *const u8, _: i64, _: u64, _: u64, _: u64, _: *mut PsvFrame) -> c_int {
        -1
    }
    pub unsafe fn psv_flush(_: *mut c_void, _: *mut PsvFrame) -> c_int {
        -1
    }
    pub unsafe fn psv_reset(_: *mut c_void) {}
    pub unsafe fn psv_copy_nv12(_: *mut c_void, _: *mut u8, _: u64) -> c_int {
        -1
    }
}

extern "C" fn io_read(opaque: *mut c_void, buf: *mut u8, len: c_int) -> c_int {
    // SAFETY: opaque is the boxed streamer owned by the caller of psm_*.
    let s = unsafe { &mut *(opaque as *mut FileStreamer) };
    let buf = unsafe { std::slice::from_raw_parts_mut(buf, len.max(0) as usize) };
    s.read(buf)
}

extern "C" fn io_seek(opaque: *mut c_void, offset: i64, whence: c_int) -> i64 {
    // SAFETY: as in io_read.
    let s = unsafe { &mut *(opaque as *mut FileStreamer) };
    s.seek(offset, whence)
}

/// Buffered (AVIO) access to a streamer.
pub struct BufferedIo {
    raw: *mut c_void,
    streamer: Option<Box<FileStreamer>>,
}

impl BufferedIo {
    pub fn new(streamer: FileStreamer) -> Result<Self, FileStreamer> {
        let mut streamer = Box::new(streamer);
        let size = streamer.buffer_size() as c_int;
        let opaque = &mut *streamer as *mut FileStreamer as *mut c_void;
        let raw = unsafe { ffi::psm_io_new(opaque, io_read, io_seek, size) };
        if raw.is_null() {
            return Err(*streamer);
        }
        Ok(Self { raw, streamer: Some(streamer) })
    }

    pub fn read(&mut self, buf: &mut [u8]) -> i32 {
        unsafe { ffi::psm_io_read(self.raw, buf.as_mut_ptr(), buf.len() as c_int) }
    }

    pub fn seek(&mut self, offset: i64, whence: i32) -> i64 {
        unsafe { ffi::psm_io_seek(self.raw, offset, whence) }
    }

    /// Drop the buffering and give the streamer back.
    pub fn into_streamer(mut self) -> FileStreamer {
        unsafe { ffi::psm_io_free(self.raw) };
        self.raw = std::ptr::null_mut();
        *self.streamer.take().expect("streamer present")
    }
}

impl Drop for BufferedIo {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ffi::psm_io_free(self.raw) }
        }
    }
}

/// An open container.
pub struct Media {
    raw: *mut c_void,
    /// Must outlive `raw` (it is the AVIO opaque).
    streamer: Box<FileStreamer>,
}

// SAFETY: the FFmpeg contexts are owned exclusively by this value and only
// used through &mut self.
unsafe impl Send for Media {}

impl Media {
    /// Probe the container; `name` (a path) only helps format detection.
    pub fn open(streamer: FileStreamer, name: Option<&str>) -> Result<Self, String> {
        let mut streamer = Box::new(streamer);
        let cname = name.and_then(|n| std::ffi::CString::new(n).ok());
        let mut err = [0 as c_char; 256];
        let opaque = &mut *streamer as *mut FileStreamer as *mut c_void;
        let raw = unsafe {
            ffi::psm_open(
                cname.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                opaque,
                io_read,
                io_seek,
                streamer.buffer_size() as c_int,
                err.as_mut_ptr(),
                err.len() as c_int,
            )
        };
        if raw.is_null() {
            let msg = unsafe { std::ffi::CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
            return Err(if msg.is_empty() { "open failed".into() } else { msg });
        }
        Ok(Self { raw, streamer })
    }

    pub fn streamer(&self) -> &FileStreamer {
        &self.streamer
    }

    /// All container streams, indexed by stream id.
    pub fn streams(&self) -> Vec<StreamInfo> {
        let n = unsafe { ffi::psm_stream_count(self.raw) };
        (0..n)
            .map(|i| {
                let mut s = PsmStreamInfo::default();
                unsafe { ffi::psm_stream_info(self.raw, i, &mut s) };
                StreamInfo {
                    media_type: match s.media_type {
                        0 => MediaType::Video,
                        1 => MediaType::Audio,
                        2 => MediaType::Subtitle,
                        _ => MediaType::Other,
                    },
                    decodable: s.decodable != 0,
                    webm_ok: s.webm_ok != 0,
                    width: s.width.max(0) as u32,
                    height: s.height.max(0) as u32,
                    channels: s.channels.max(0) as u32,
                    sample_rate: s.sample_rate.max(0) as u32,
                    frame_size: s.frame_size.max(0) as u32,
                    fps: s.fps,
                    sar: s.sar,
                    full_range: s.full_range != 0,
                    color_primaries: s.color_primaries,
                    color_trc: s.color_trc,
                    duration_ms: s.duration_ms,
                    language: s.lang,
                }
            })
            .collect()
    }

    pub fn duration_ms(&self) -> u64 {
        unsafe { ffi::psm_duration_ms(self.raw) }
    }

    pub fn set_discard(&mut self, video: Option<usize>, audio: Option<usize>) {
        let i = |s: Option<usize>| s.map_or(-1, |v| v as c_int);
        unsafe { ffi::psm_set_discard(self.raw, i(video), i(audio)) }
    }

    /// Seek every stream to `ms` on the time base of `stream`.
    pub fn seek(&mut self, stream: Option<usize>, ms: u64) -> i32 {
        unsafe { ffi::psm_seek(self.raw, stream.map_or(-1, |v| v as c_int), ms) }
    }

    /// Next packet, or an AVERROR code.
    pub fn read(&mut self) -> Result<Packet, i32> {
        let (mut p, mut stream, mut size) = (std::ptr::null_mut(), 0, 0);
        let r = unsafe { ffi::psm_read(self.raw, &mut p, &mut stream, &mut size) };
        if r < 0 || p.is_null() {
            return Err(r.min(-1));
        }
        Ok(Packet { raw: p, stream: stream.max(0) as usize, size: size.max(0) as usize })
    }

    pub fn decoder(&self, stream: usize) -> Option<StreamDecoder> {
        let raw = unsafe { ffi::psm_decoder_open(self.raw, stream as c_int) };
        (!raw.is_null()).then_some(StreamDecoder { raw })
    }
}

impl Drop for Media {
    fn drop(&mut self) {
        unsafe { ffi::psm_close(self.raw) }
    }
}

/// A demuxed packet.
pub struct Packet {
    raw: *mut c_void,
    pub stream: usize,
    pub size: usize,
}

// SAFETY: an AVPacket owned by this value.
unsafe impl Send for Packet {}

impl Drop for Packet {
    fn drop(&mut self) {
        unsafe { ffi::psm_packet_free(self.raw) }
    }
}

/// Decoder of one container stream.
pub struct StreamDecoder {
    raw: *mut c_void,
}

// SAFETY: exclusively owned FFmpeg context.
unsafe impl Send for StreamDecoder {}

impl StreamDecoder {
    /// (channels, frame size) of an audio decoder.
    pub fn params(&self) -> (u32, u32) {
        let (mut c, mut f) = (0, 0);
        unsafe { ffi::psm_decoder_params(self.raw, &mut c, &mut f) };
        (c.max(0) as u32, f.max(0) as u32)
    }

    pub fn flush(&mut self) {
        unsafe { ffi::psm_decoder_flush(self.raw) }
    }

    /// avcodec_send_packet (None = drain).
    pub fn send(&mut self, p: Option<&Packet>) -> i32 {
        unsafe { ffi::psm_decoder_send(self.raw, p.map_or(std::ptr::null(), |p| p.raw)) }
    }

    /// avcodec_receive_frame.
    pub fn receive(&mut self, out: &mut FrameMeta) -> i32 {
        let mut f = PsmFrame::default();
        let r = unsafe { ffi::psm_decoder_receive(self.raw, &mut f) };
        if r == 0 {
            *out = FrameMeta {
                width: f.width.max(0) as u32,
                height: f.height.max(0) as u32,
                ts_ms: f.ts_ms,
                sar: f.sar,
                full_range: f.full_range != 0,
                crop: [
                    f.crop_left.max(0) as u32,
                    f.crop_right.max(0) as u32,
                    f.crop_top.max(0) as u32,
                    f.crop_bottom.max(0) as u32,
                ],
                nb_samples: f.nb_samples.max(0) as u32,
                channels: f.channels.max(0) as u32,
                sample_rate: f.sample_rate.max(0) as u32,
            };
        }
        r
    }

    /// Write the current picture as NV12 at `dst`.
    ///
    /// # Safety
    /// `dst` must be writable for `pitch * rows + pitch * ceil(rows / 2)` bytes.
    pub unsafe fn write_nv12(&mut self, dst: *mut u8, pitch: u32, rows: u32, full_range: bool) -> bool {
        ffi::psm_decoder_nv12(self.raw, dst, pitch, rows, full_range as c_int) == 0
    }

    /// The current audio frame as interleaved s16 (`samples * channels`
    /// values), valid until the next `receive`.
    pub fn s16(&mut self, channels: u32) -> Option<&[i16]> {
        let mut p = std::ptr::null();
        let n = unsafe { ffi::psm_decoder_s16(self.raw, &mut p) };
        if n < 0 || p.is_null() {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(p, n as usize * channels.max(1) as usize) })
    }
}

impl Drop for StreamDecoder {
    fn drop(&mut self) {
        unsafe { ffi::psm_decoder_close(self.raw) }
    }
}

/// Raw access-unit video decoder (H.264 / HEVC / VP9).
pub struct RawDecoder {
    raw: *mut c_void,
}

// SAFETY: exclusively owned FFmpeg context, used through &mut self.
unsafe impl Send for RawDecoder {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawCodec {
    H264,
    Hevc,
    Vp9,
}

impl RawDecoder {
    pub fn new(codec: RawCodec) -> Option<Self> {
        let c = match codec {
            RawCodec::H264 => 0,
            RawCodec::Hevc => 1,
            RawCodec::Vp9 => 2,
        };
        let raw = unsafe { ffi::psv_open(c) };
        (!raw.is_null()).then_some(Self { raw })
    }

    fn picture(f: &PsvFrame) -> RawPicture {
        RawPicture {
            width: f.width.max(0) as u32,
            height: f.height.max(0) as u32,
            pitch: f.pitch.max(0) as u32,
            crop: [
                f.crop_left.max(0) as u32,
                f.crop_right.max(0) as u32,
                f.crop_top.max(0) as u32,
                f.crop_bottom.max(0) as u32,
            ],
            profile: f.profile.max(0) as u32,
            level: f.level.max(0) as u32,
            sar: (f.sar_num.max(0) as u32, f.sar_den.max(0) as u32),
            color_range: f.color_range as u8,
            color_primaries: f.color_primaries as u8,
            color_trc: f.color_trc as u8,
            color_space: f.color_space as u8,
            key_frame: f.key_frame != 0,
            corrupt: f.corrupt != 0,
            pts: f.pts,
            dts: f.dts,
            attached: f.attached,
        }
    }

    /// Feed one access unit; returns a picture when one became available.
    pub fn decode(&mut self, au: &[u8], pts: u64, dts: u64, attached: u64) -> Result<Option<RawPicture>, RawError> {
        let mut f = PsvFrame::default();
        let r = unsafe { ffi::psv_decode(self.raw, au.as_ptr(), au.len() as i64, pts, dts, attached, &mut f) };
        match r {
            1 => Ok(Some(Self::picture(&f))),
            0 => Ok(None),
            -2 => Err(RawError::AccessUnit),
            _ => Err(RawError::ApiFail),
        }
    }

    /// Drain the decoder: the first call ends the stream, each call returns
    /// one remaining picture. The next access unit restarts the decoder.
    pub fn flush(&mut self) -> Result<Option<RawPicture>, RawError> {
        let mut f = PsvFrame::default();
        match unsafe { ffi::psv_flush(self.raw, &mut f) } {
            1 => Ok(Some(Self::picture(&f))),
            0 => Ok(None),
            _ => Err(RawError::ApiFail),
        }
    }

    pub fn reset(&mut self) {
        unsafe { ffi::psv_reset(self.raw) }
    }

    /// Write the current picture as NV12 (pitch = width aligned to 256,
    /// chroma after `height` rows) at `dst`.
    ///
    /// # Safety
    /// `dst` must be writable for `size` bytes.
    pub unsafe fn copy_nv12(&mut self, dst: *mut u8, size: u64) -> Result<(), RawError> {
        match ffi::psv_copy_nv12(self.raw, dst, size) {
            0 => Ok(()),
            -3 => Err(RawError::FrameBufferSize),
            _ => Err(RawError::ApiFail),
        }
    }
}

impl Drop for RawDecoder {
    fn drop(&mut self) {
        unsafe { ffi::psv_close(self.raw) }
    }
}
