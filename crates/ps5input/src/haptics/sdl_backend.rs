// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! [`Backend`] on SDL3 (loaded at run time like the gamepad backend; the
//! same library instance is shared).

use super::opus::{Encoder, OpusEncoder};
use super::{AudioDevice, Backend, Connection, DeviceStream, HidDevice, HidInfo, PullCallback};
use libloading::Library;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::Arc;
use std::thread::JoinHandle;

type Gamepad = c_void;
type AudioStream = c_void;
type HidHandle = c_void;
#[cfg(windows)]
type WChar = u16;
#[cfg(not(windows))]
type WChar = i32;

const SDL_INIT_AUDIO: u32 = 0x10;
const SDL_AUDIO_F32: u32 = 0x8120;
const GAMEPAD_TYPE_PS5: c_int = 6;
const CONNECTION_WIRED: c_int = 1;
const CONNECTION_WIRELESS: c_int = 2;
const HID_BUS_BLUETOOTH: c_int = 2;

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct AudioSpec {
    format: u32,
    channels: c_int,
    freq: c_int,
}

#[repr(C)]
struct HidDeviceInfo {
    path: *mut c_char,
    vendor_id: u16,
    product_id: u16,
    serial_number: *mut WChar,
    release_number: u16,
    manufacturer_string: *mut WChar,
    product_string: *mut WChar,
    usage_page: u16,
    usage: u16,
    interface_number: c_int,
    interface_class: c_int,
    interface_subclass: c_int,
    interface_protocol: c_int,
    bus_type: c_int,
    next: *mut HidDeviceInfo,
}

type StreamCallback = unsafe extern "C" fn(*mut c_void, *mut AudioStream, c_int, c_int);

macro_rules! api {
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
                let mut last = String::new();
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

api! {
    SDL_InitSubSystem: fn(u32) -> bool;
    SDL_QuitSubSystem: fn(u32);
    SDL_GetTicks: fn() -> u64;
    SDL_GetTicksNS: fn() -> u64;
    SDL_free: fn(*mut c_void);
    SDL_LockJoysticks: fn();
    SDL_UnlockJoysticks: fn();
    SDL_GetGamepads: fn(*mut c_int) -> *mut u32;
    SDL_GetGamepadFromID: fn(u32) -> *mut Gamepad;
    SDL_GetGamepadTypeForID: fn(u32) -> c_int;
    SDL_GetGamepadConnectionState: fn(*mut Gamepad) -> c_int;
    SDL_GetGamepadVendor: fn(*mut Gamepad) -> u16;
    SDL_GetGamepadProduct: fn(*mut Gamepad) -> u16;
    SDL_GetGamepadSerial: fn(*mut Gamepad) -> *const c_char;
    SDL_RumbleGamepad: fn(*mut Gamepad, u16, u16, u32) -> bool;
    SDL_SendGamepadEffect: fn(*mut Gamepad, *const c_void, c_int) -> bool;
    SDL_GetAudioPlaybackDevices: fn(*mut c_int) -> *mut u32;
    SDL_GetAudioDeviceName: fn(u32) -> *const c_char;
    SDL_GetAudioDeviceFormat: fn(u32, *mut AudioSpec, *mut c_int) -> bool;
    SDL_OpenAudioDeviceStream: fn(u32, *const AudioSpec, Option<StreamCallback>, *mut c_void) -> *mut AudioStream;
    SDL_GetAudioStreamDevice: fn(*mut AudioStream) -> u32;
    SDL_ResumeAudioStreamDevice: fn(*mut AudioStream) -> bool;
    SDL_GetAudioStreamQueued: fn(*mut AudioStream) -> c_int;
    SDL_PutAudioStreamData: fn(*mut AudioStream, *const c_void, c_int) -> bool;
    SDL_GetAudioStreamData: fn(*mut AudioStream, *mut c_void, c_int) -> c_int;
    SDL_GetAudioStreamFormat: fn(*mut AudioStream, *mut AudioSpec, *mut AudioSpec) -> bool;
    SDL_ClearAudioStream: fn(*mut AudioStream) -> bool;
    SDL_DestroyAudioStream: fn(*mut AudioStream);
    SDL_hid_enumerate: fn(u16, u16) -> *mut HidDeviceInfo;
    SDL_hid_free_enumeration: fn(*mut HidDeviceInfo);
    SDL_hid_open_path: fn(*const c_char) -> *mut HidHandle;
    SDL_hid_write: fn(*mut HidHandle, *const u8, usize) -> c_int;
    SDL_hid_close: fn(*mut HidHandle) -> c_int;
}

fn cstr(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        // SAFETY: SDL returns NUL-terminated strings.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

fn wstr(p: *const WChar) -> String {
    let mut s = String::new();
    if p.is_null() {
        return s;
    }
    let mut i = 0;
    // SAFETY: NUL-terminated wide string from the HID enumeration.
    while let Some(c) = unsafe { (*p.add(i) != 0).then(|| *p.add(i)) } {
        s.push(char::from_u32(c as u32).unwrap_or('?'));
        i += 1;
    }
    s
}

pub struct SdlBackend {
    api: Arc<Api>,
}

impl SdlBackend {
    pub fn new() -> Result<Self, String> {
        Ok(Self { api: Arc::new(Api::load()?) })
    }

    /// Run `f` on an open gamepad with the joystick lock held.
    fn with_pad<T>(&self, controller: i32, f: impl FnOnce(*mut Gamepad) -> T) -> Option<T> {
        if controller < 0 {
            return None;
        }
        let a = &self.api;
        // SAFETY: SDL gamepad calls under SDL's joystick lock.
        unsafe {
            (a.SDL_LockJoysticks)();
            let pad = (a.SDL_GetGamepadFromID)(controller as u32);
            let r = (!pad.is_null()).then(|| f(pad));
            (a.SDL_UnlockJoysticks)();
            r
        }
    }
}

struct SdlStream {
    api: Arc<Api>,
    stream: *mut AudioStream,
    _callback: Box<PullCallback>,
}

// SAFETY: SDL audio streams are thread-safe.
unsafe impl Send for SdlStream {}

impl DeviceStream for SdlStream {
    fn device_channels(&self) -> Option<i32> {
        let mut spec = AudioSpec::default();
        // SAFETY: valid stream.
        unsafe {
            let dev = (self.api.SDL_GetAudioStreamDevice)(self.stream);
            (self.api.SDL_GetAudioDeviceFormat)(dev, &mut spec, std::ptr::null_mut()).then_some(spec.channels)
        }
    }

    fn resume(&mut self) -> bool {
        // SAFETY: valid stream.
        unsafe { (self.api.SDL_ResumeAudioStreamDevice)(self.stream) }
    }

    fn queued(&self) -> i32 {
        // SAFETY: valid stream.
        unsafe { (self.api.SDL_GetAudioStreamQueued)(self.stream) }
    }

    fn put(&mut self, samples: &[f32]) -> bool {
        // SAFETY: valid stream and buffer.
        unsafe { (self.api.SDL_PutAudioStreamData)(self.stream, samples.as_ptr().cast(), (samples.len() * 4) as c_int) }
    }

    fn get(&mut self, buf: &mut [u8]) -> i32 {
        // SAFETY: valid stream and buffer.
        unsafe { (self.api.SDL_GetAudioStreamData)(self.stream, buf.as_mut_ptr().cast(), buf.len() as c_int) }
    }

    fn target(&self) -> Option<(i32, i32)> {
        let mut dst = AudioSpec::default();
        // SAFETY: valid stream.
        let ok = unsafe { (self.api.SDL_GetAudioStreamFormat)(self.stream, std::ptr::null_mut(), &mut dst) };
        // F32 samples.
        ok.then_some((dst.channels * 4, dst.freq))
    }

    fn clear(&mut self) {
        // SAFETY: valid stream.
        unsafe { (self.api.SDL_ClearAudioStream)(self.stream) };
    }
}

impl Drop for SdlStream {
    fn drop(&mut self) {
        // SAFETY: SDL waits for a running callback before returning.
        unsafe { (self.api.SDL_DestroyAudioStream)(self.stream) };
    }
}

unsafe extern "C" fn on_pull(userdata: *mut c_void, _stream: *mut AudioStream, _additional: c_int, _total: c_int) {
    // SAFETY: userdata is the stream's boxed callback, alive until destroy.
    let cb = unsafe { &*(userdata as *const PullCallback) };
    cb();
}

struct SdlHid {
    api: Arc<Api>,
    dev: *mut HidHandle,
}

// SAFETY: writes are serialised by the session's I/O lock.
unsafe impl Send for SdlHid {}
unsafe impl Sync for SdlHid {}

impl HidDevice for SdlHid {
    fn write(&self, data: &[u8]) -> i32 {
        // SAFETY: valid device and buffer.
        unsafe { (self.api.SDL_hid_write)(self.dev, data.as_ptr(), data.len()) }
    }
}

impl Drop for SdlHid {
    fn drop(&mut self) {
        // SAFETY: opened by SDL_hid_open_path.
        unsafe { (self.api.SDL_hid_close)(self.dev) };
    }
}

impl Backend for SdlBackend {
    fn ticks(&self) -> u64 {
        // SAFETY: plain query.
        unsafe { (self.api.SDL_GetTicks)() }
    }

    fn ticks_ns(&self) -> u64 {
        // SAFETY: plain query.
        unsafe { (self.api.SDL_GetTicksNS)() }
    }

    fn init_audio(&self) -> bool {
        // SAFETY: reference-counted subsystem init.
        unsafe { (self.api.SDL_InitSubSystem)(SDL_INIT_AUDIO) }
    }

    fn quit_audio(&self) {
        // SAFETY: balances init_audio.
        unsafe { (self.api.SDL_QuitSubSystem)(SDL_INIT_AUDIO) }
    }

    fn is_ps5(&self, controller: i32) -> bool {
        // SAFETY: plain query.
        controller >= 0 && unsafe { (self.api.SDL_GetGamepadTypeForID)(controller as u32) } == GAMEPAD_TYPE_PS5
    }

    fn connection(&self, controller: i32) -> Option<Connection> {
        let a = self.api.clone();
        self.with_pad(controller, |p| {
            // SAFETY: open gamepad.
            if unsafe { (a.SDL_GetGamepadConnectionState)(p) } == CONNECTION_WIRELESS {
                Connection::Wireless
            } else {
                Connection::Wired
            }
        })
    }

    fn gamepads(&self) -> Vec<(i32, bool, Connection)> {
        let a = &self.api;
        let mut out = Vec::new();
        // SAFETY: SDL gamepad calls under the joystick lock.
        unsafe {
            (a.SDL_LockJoysticks)();
            let mut n: c_int = 0;
            let ids = (a.SDL_GetGamepads)(&mut n);
            if !ids.is_null() {
                for i in 0..n.max(0) as usize {
                    let id = *ids.add(i);
                    let pad = (a.SDL_GetGamepadFromID)(id);
                    if pad.is_null() {
                        continue;
                    }
                    let ps5 = (a.SDL_GetGamepadTypeForID)(id) == GAMEPAD_TYPE_PS5;
                    let conn = match (a.SDL_GetGamepadConnectionState)(pad) {
                        CONNECTION_WIRED => Connection::Wired,
                        CONNECTION_WIRELESS => Connection::Wireless,
                        _ => continue,
                    };
                    out.push((id as i32, ps5, conn));
                }
                (a.SDL_free)(ids.cast());
            }
            (a.SDL_UnlockJoysticks)();
        }
        out
    }

    fn rumble(&self, controller: i32, large: u16, small: u16, duration_ms: u32) -> bool {
        let a = self.api.clone();
        // SAFETY: open gamepad.
        self.with_pad(controller, |p| unsafe { (a.SDL_RumbleGamepad)(p, large, small, duration_ms) }).unwrap_or(false)
    }

    fn send_effect(&self, controller: i32, data: &[u8]) -> bool {
        let a = self.api.clone();
        // SAFETY: open gamepad and valid buffer.
        self.with_pad(controller, |p| unsafe { (a.SDL_SendGamepadEffect)(p, data.as_ptr().cast(), data.len() as c_int) })
            .unwrap_or(false)
    }

    fn playback_devices(&self) -> Vec<AudioDevice> {
        let a = &self.api;
        let mut out = Vec::new();
        // SAFETY: SDL audio queries; the list is freed.
        unsafe {
            let mut n: c_int = 0;
            let ids = (a.SDL_GetAudioPlaybackDevices)(&mut n);
            if ids.is_null() {
                return out;
            }
            for i in 0..n.max(0) as usize {
                let id = *ids.add(i);
                let mut spec = AudioSpec::default();
                let channels = (a.SDL_GetAudioDeviceFormat)(id, &mut spec, std::ptr::null_mut()).then_some(spec.channels);
                out.push(AudioDevice { id, name: cstr((a.SDL_GetAudioDeviceName)(id)), channels });
            }
            (a.SDL_free)(ids.cast());
        }
        out
    }

    fn open_device(&self, device: u32, freq: u32, on_pull_cb: PullCallback) -> Option<Box<dyn DeviceStream>> {
        let spec = AudioSpec { format: SDL_AUDIO_F32, channels: 4, freq: freq as c_int };
        let callback = Box::new(on_pull_cb);
        let userdata = &*callback as *const PullCallback as *mut c_void;
        // SAFETY: the callback box outlives the stream (dropped after destroy).
        let stream = unsafe { (self.api.SDL_OpenAudioDeviceStream)(device, &spec, Some(on_pull), userdata) };
        if stream.is_null() {
            return None;
        }
        Some(Box::new(SdlStream { api: self.api.clone(), stream, _callback: callback }))
    }

    fn wireless_identity(&self, controller: i32) -> Option<(u16, u16, String)> {
        let a = self.api.clone();
        self.with_pad(controller, |p| {
            // SAFETY: open gamepad.
            unsafe {
                let wireless = (a.SDL_GetGamepadTypeForID)(controller as u32) == GAMEPAD_TYPE_PS5
                    && (a.SDL_GetGamepadConnectionState)(p) == CONNECTION_WIRELESS;
                wireless.then(|| {
                    ((a.SDL_GetGamepadVendor)(p), (a.SDL_GetGamepadProduct)(p), cstr((a.SDL_GetGamepadSerial)(p)))
                })
            }
        })
        .flatten()
    }

    fn hid_enumerate(&self, vendor: u16, product: u16) -> Vec<HidInfo> {
        let a = &self.api;
        let mut out = Vec::new();
        // SAFETY: walk and free the enumeration list.
        unsafe {
            let list = (a.SDL_hid_enumerate)(vendor, product);
            let mut d = list;
            while !d.is_null() {
                let info = &*d;
                out.push(HidInfo {
                    path: cstr(info.path),
                    serial: wstr(info.serial_number),
                    bluetooth: info.bus_type == HID_BUS_BLUETOOTH,
                    usage_page: info.usage_page,
                    usage: info.usage,
                });
                d = info.next;
            }
            if !list.is_null() {
                (a.SDL_hid_free_enumeration)(list);
            }
        }
        out
    }

    fn hid_open(&self, path: &str) -> Option<Box<dyn HidDevice>> {
        let path = CString::new(path).ok()?;
        // SAFETY: valid C string.
        let dev = unsafe { (self.api.SDL_hid_open_path)(path.as_ptr()) };
        (!dev.is_null()).then(|| Box::new(SdlHid { api: self.api.clone(), dev }) as Box<dyn HidDevice>)
    }

    fn create_encoder(&self) -> Option<Box<dyn Encoder>> {
        OpusEncoder::new().map(|e| Box::new(e) as Box<dyn Encoder>)
    }

    fn spawn_sender(&self, run: Box<dyn FnOnce() + Send>) -> Option<Option<JoinHandle<()>>> {
        std::thread::Builder::new().name("rps5e-ds-bluetooth".into()).spawn(run).ok().map(Some)
    }
}
