// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! SDL3 audio streams (libSDL3 loaded at run time).
//!
//! SDL picks the desktop's sound server (PipeWire, PulseAudio, then ALSA,
//! ...) and converts channels, sample format and rate to whatever the
//! device plays, so a guest 7.1 port reaches a stereo desktop downmixed
//! instead of being written to whichever raw ALSA card happens to be
//! `default` (often a loopback or HDMI device nobody listens to). Each
//! port gets one `SDL_AudioStream` bound to the default playback device;
//! SDL's stream queue is the host queue AudioOut paces against.

use super::host::{HostAudioStream, PcmFormat, PcmStats};
use libloading::Library;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::{Mutex, OnceLock};

const SDL_INIT_AUDIO: u32 = 0x0000_0010;
const SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK: u32 = 0xFFFF_FFFF;
const SDL_AUDIO_S16LE: c_int = 0x8010;
const SDL_AUDIO_F32LE: c_int = 0x8120;

/// SDL_AudioSpec { SDL_AudioFormat format; int channels; int freq; }.
#[repr(C)]
#[derive(Default)]
struct AudioSpec {
    format: c_int,
    channels: c_int,
    freq: c_int,
}

type Stream = c_void;

macro_rules! sdl_audio_api {
    ($($name:ident : fn($($arg:ty),*) $(-> $ret:ty)?;)*) => {
        #[allow(non_snake_case)]
        struct Api {
            _lib: Library,
            $($name: unsafe extern "C" fn($($arg),*) $(-> $ret)?,)*
        }

        impl Api {
            #[allow(non_snake_case)]
            fn load() -> Result<Api, String> {
                let names: &[&str] = if cfg!(windows) {
                    &["SDL3.dll"]
                } else if cfg!(target_os = "macos") {
                    &["libSDL3.0.dylib", "libSDL3.dylib"]
                } else {
                    &["libSDL3.so.0", "libSDL3.so"]
                };
                let mut last = String::from("not found");
                for n in names {
                    // SAFETY: SDL3 has no unsound load-time initialisers.
                    match unsafe { Library::new(n) } {
                        Ok(lib) => unsafe {
                            $(
                                let $name = *lib
                                    .get::<unsafe extern "C" fn($($arg),*) $(-> $ret)?>(
                                        concat!(stringify!($name), "\0").as_bytes(),
                                    )
                                    .map_err(|e| format!("{}: {e}", stringify!($name)))?;
                            )*
                            return Ok(Api { _lib: lib, $($name,)* });
                        },
                        Err(e) => last = e.to_string(),
                    }
                }
                Err(last)
            }
        }
    };
}

sdl_audio_api! {
    SDL_InitSubSystem: fn(u32) -> bool;
    SDL_GetError: fn() -> *const c_char;
    SDL_GetCurrentAudioDriver: fn() -> *const c_char;
    SDL_OpenAudioDeviceStream: fn(u32, *const AudioSpec, *const c_void, *mut c_void) -> *mut Stream;
    SDL_GetAudioStreamDevice: fn(*mut Stream) -> u32;
    SDL_GetAudioDeviceName: fn(u32) -> *const c_char;
    SDL_GetAudioDeviceFormat: fn(u32, *mut AudioSpec, *mut c_int) -> bool;
    SDL_ResumeAudioStreamDevice: fn(*mut Stream) -> bool;
    SDL_PutAudioStreamData: fn(*mut Stream, *const c_void, c_int) -> bool;
    SDL_GetAudioStreamQueued: fn(*mut Stream) -> c_int;
    SDL_ClearAudioStream: fn(*mut Stream) -> bool;
    SDL_DestroyAudioStream: fn(*mut Stream);
}

unsafe fn c_str(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

impl Api {
    fn error(&self) -> String {
        unsafe { c_str((self.SDL_GetError)()) }
    }
}

/// The library with its audio subsystem up; None when SDL3 is missing or
/// has no working audio driver.
fn api() -> Option<&'static Api> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    API.get_or_init(|| {
        let api = match Api::load() {
            Ok(a) => a,
            Err(e) => {
                log::info!("[AUDIO] SDL3 not available ({e}); using cpal");
                return None;
            }
        };
        if !unsafe { (api.SDL_InitSubSystem)(SDL_INIT_AUDIO) } {
            log::warn!("[AUDIO] SDL audio init failed: {}; using cpal", api.error());
            return None;
        }
        let driver = unsafe { c_str((api.SDL_GetCurrentAudioDriver)()) };
        log::info!("[AUDIO] host backend SDL3, driver {driver:?}");
        Some(api)
    })
    .as_ref()
}

struct SdlStream {
    api: &'static Api,
    stream: *mut Stream,
    stats: Mutex<PcmStats>,
    format: PcmFormat,
    report_every: u64,
    id: u64,
}

// SAFETY: SDL audio streams are internally locked; every entry point used
// here may be called from any thread.
unsafe impl Send for SdlStream {}
unsafe impl Sync for SdlStream {}

impl HostAudioStream for SdlStream {
    fn put(&self, pcm: &[u8]) -> bool {
        if self.report_every != 0 {
            let mut st = self.stats.lock().unwrap_or_else(|e| e.into_inner());
            st.add_pcm(pcm, self.format);
            if st.put_samples >= self.report_every {
                let done = std::mem::take(&mut *st);
                drop(st);
                log::info!(
                    "[AUDIO] stream {}: queued rms {:.5} peak {:.4} ({} samples), backlog {} bytes (SDL)",
                    self.id,
                    done.put_rms(),
                    done.peak,
                    done.put_samples,
                    self.queued_bytes()
                );
            }
        }
        let gain = super::host::output_gain();
        if gain != 1.0 {
            thread_local! {
                static SCALED: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
            }
            return SCALED.with(|b| {
                let mut b = b.borrow_mut();
                super::host::scale_pcm(pcm, self.format, gain, &mut b);
                let len = b.len().min(c_int::MAX as usize) as c_int;
                unsafe { (self.api.SDL_PutAudioStreamData)(self.stream, b.as_ptr().cast(), len) }
            });
        }
        let len = pcm.len().min(c_int::MAX as usize) as c_int;
        unsafe { (self.api.SDL_PutAudioStreamData)(self.stream, pcm.as_ptr().cast(), len) }
    }

    fn queued_bytes(&self) -> usize {
        unsafe { (self.api.SDL_GetAudioStreamQueued)(self.stream) }.max(0) as usize
    }

    fn clear(&self) {
        unsafe { (self.api.SDL_ClearAudioStream)(self.stream) };
    }

    fn backend_name(&self) -> &'static str {
        "sdl3"
    }
}

impl Drop for SdlStream {
    fn drop(&mut self) {
        unsafe { (self.api.SDL_DestroyAudioStream)(self.stream) };
    }
}

/// Open a stream on the default playback device; None when SDL3 cannot.
pub fn open_stream(
    rate: u32,
    channels: usize,
    format: PcmFormat,
    id: u64,
    stats_every_secs: u64,
) -> Option<Box<dyn HostAudioStream>> {
    let api = api()?;
    let spec = AudioSpec {
        format: match format {
            PcmFormat::S16 => SDL_AUDIO_S16LE,
            PcmFormat::F32 => SDL_AUDIO_F32LE,
        },
        channels: channels as c_int,
        freq: rate as c_int,
    };
    let stream = unsafe {
        (api.SDL_OpenAudioDeviceStream)(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, std::ptr::null(), std::ptr::null_mut())
    };
    if stream.is_null() {
        log::warn!("[AUDIO] SDL_OpenAudioDeviceStream failed: {}", api.error());
        return None;
    }
    // Owned from here: dropped (destroyed) on any later failure.
    let s = SdlStream {
        api,
        stream,
        stats: Mutex::new(PcmStats::default()),
        format,
        report_every: stats_every_secs * rate as u64 * channels as u64,
        id,
    };
    if !unsafe { (api.SDL_ResumeAudioStreamDevice)(stream) } {
        log::warn!("[AUDIO] SDL_ResumeAudioStreamDevice failed: {}", api.error());
        return None;
    }
    let dev = unsafe { (api.SDL_GetAudioStreamDevice)(stream) };
    let mut dev_spec = AudioSpec::default();
    let mut frames: c_int = 0;
    let name = unsafe { c_str((api.SDL_GetAudioDeviceName)(dev)) };
    unsafe { (api.SDL_GetAudioDeviceFormat)(dev, &mut dev_spec, &mut frames) };
    log::info!(
        "[AUDIO] device stream {id}: guest {rate} Hz {channels} ch {format:?} -> SDL device {name:?} {} Hz {} ch (format {:#x}, {frames} frames)",
        dev_spec.freq,
        dev_spec.channels,
        dev_spec.format
    );
    Some(Box::new(s))
}
