// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only
//
// Unit tests of the `ps5hle::avplayer` module.

use super::host::BufferedIo;
use super::stream::*;
use super::*;
use crate::context::CpuRegister;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

const FN_ALLOC: u64 = 0xA110_0001;
const FN_FREE: u64 = 0xA110_0002;
const FN_ALLOC_TEX: u64 = 0xA110_0003;
const FN_FREE_TEX: u64 = 0xA110_0004;
const FN_EVENT: u64 = 0xA110_0005;
const EVENT_OBJECT: u64 = 0x5EED;
const MEM_OBJECT: u64 = 0x3E3;
/// File callbacks backed by a host movie (playback tests).
const FN_OPEN: u64 = 0xF11E_0001;
const FN_CLOSE: u64 = 0xF11E_0002;
const FN_READ: u64 = 0xF11E_0003;
const FN_SIZE: u64 = 0xF11E_0004;
const FILE_OBJECT: u64 = 0xF00D;
/// File callbacks backed by an in-memory payload (streamer tests).
const FN_CB_OPEN: u64 = 0xCB00_0001;
const FN_CB_CLOSE: u64 = 0xCB00_0002;
const FN_CB_READ: u64 = 0xCB00_0003;
const FN_CB_SIZE: u64 = 0xCB00_0004;
const CB_OBJECT: u64 = 0xCB0B;

/// Host file behind the movie file callbacks.
static REPLACEMENT: Mutex<Option<(PathBuf, Option<Vec<u8>>)>> = Mutex::new(None);

#[derive(Default)]
struct Callbacks {
    data: Vec<u8>,
    opens: u32,
    closes: u32,
    premature_eof: bool,
}
static CALLBACKS: Mutex<Option<Callbacks>> = Mutex::new(None);

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static FREES: AtomicU64 = AtomicU64::new(0);
static EVENTS: Mutex<Vec<(i32, u32)>> = Mutex::new(Vec::new());
static TEST_LOCK: Mutex<()> = Mutex::new(());
/// Player the READY event callback starts.
static START_ON_READY: Mutex<Option<u64>> = Mutex::new(None);

fn stub_guest(func: u64, args: [u64; 6]) -> u64 {
    match func {
        FN_ALLOC | FN_ALLOC_TEX => {
            assert_eq!(args[0], MEM_OBJECT);
            assert_eq!(args[1], 0x100, "alignment");
            ALLOCS.fetch_add(1, Ordering::SeqCst);
            arena_alloc(args[2]).unwrap()
        }
        FN_FREE | FN_FREE_TEX => {
            assert_eq!(args[0], MEM_OBJECT);
            FREES.fetch_add(1, Ordering::SeqCst);
            arena_free(args[1]);
            0
        }
        FN_EVENT => {
            assert_eq!(args[0], EVENT_OBJECT);
            let warning = if args[3] != 0 { unsafe { std::ptr::read(args[3] as *const u32) } } else { 0 };
            EVENTS.lock().unwrap().push((args[1] as i32, warning));
            // A title that starts playback itself from its READY callback.
            if args[1] as i32 == EVENT_STATE_READY {
                let handle = START_ON_READY.lock().unwrap().take();
                if let Some(h) = handle {
                    assert_eq!(start_impl(&player(h).unwrap(), 0, false, false), OK);
                }
            }
            0
        }
        FN_OPEN | FN_CLOSE | FN_READ | FN_SIZE => {
            assert_eq!(args[0], FILE_OBJECT);
            let mut g = REPLACEMENT.lock().unwrap();
            let (host, data) = g.as_mut().unwrap();
            match func {
                FN_OPEN => {
                    let name = unsafe { std::ffi::CStr::from_ptr(args[1] as *const std::ffi::c_char) };
                    assert_eq!(name.to_str().unwrap(), "virtual:clip");
                    *data = Some(std::fs::read(&*host).unwrap());
                    0
                }
                FN_CLOSE => {
                    *data = None;
                    0
                }
                FN_SIZE => data.as_ref().unwrap().len() as u64,
                _ => {
                    let d = data.as_ref().unwrap();
                    let (pos, len) = (args[2] as usize, args[3] as usize);
                    let n = len.min(d.len() - pos);
                    unsafe { std::ptr::copy_nonoverlapping(d[pos..].as_ptr(), args[1] as *mut u8, n) };
                    n as u64
                }
            }
        }
        FN_CB_OPEN | FN_CB_CLOSE | FN_CB_READ | FN_CB_SIZE => {
            assert_eq!(args[0], CB_OBJECT);
            let mut g = CALLBACKS.lock().unwrap();
            let c = g.as_mut().unwrap();
            match func {
                FN_CB_OPEN => {
                    c.opens += 1;
                    0
                }
                FN_CB_CLOSE => {
                    c.closes += 1;
                    0
                }
                FN_CB_SIZE => c.data.len() as u64,
                _ => {
                    if c.premature_eof {
                        return 0;
                    }
                    let (pos, len) = (args[2] as usize, args[3] as usize);
                    let n = len.min(c.data.len() - pos);
                    unsafe { std::ptr::copy_nonoverlapping(c.data[pos..].as_ptr(), args[1] as *mut u8, n) };
                    n as u64
                }
            }
        }
        _ => panic!("unexpected guest call {func:#x}"),
    }
}

fn events() -> Vec<i32> {
    EVENTS.lock().unwrap().iter().map(|e| e.0).collect()
}

struct Env {
    vm: ps5vm::VirtualMemory,
    scratch: u64,
}

impl Env {
    fn new() -> Self {
        set_guest_call(stub_guest);
        Self { vm: ps5vm::VirtualMemory::new(), scratch: arena_alloc(0x10000).unwrap() }
    }

    fn call(&self, cb: HleCallback, args: &[u64]) -> HleResult {
        let mut ctx = CpuContext::new(&self.vm);
        const REGS: [CpuRegister; 4] = [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx];
        for (i, v) in args.iter().enumerate() {
            ctx.set(REGS[i], *v);
        }
        cb(&mut ctx)
    }

    fn ctx(&self) -> CpuContext<'_> {
        CpuContext::new(&self.vm)
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        arena_free(self.scratch);
    }
}

fn write_allocators(ctx: &CpuContext, at: u64) {
    ctx.write_u64(at, MEM_OBJECT);
    ctx.write_u64(at + 8, FN_ALLOC);
    ctx.write_u64(at + 16, FN_FREE);
    ctx.write_u64(at + 24, FN_ALLOC_TEX);
    ctx.write_u64(at + 32, FN_FREE_TEX);
}

/// A one-second 320x180 H.264 + AAC clip, made with the host ffmpeg tool.
fn test_clip() -> Option<PathBuf> {
    let dir = std::env::temp_dir().join(format!("rps5e-avplayer-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let out = dir.join("clip.mp4");
    if out.is_file() {
        return Some(dir);
    }
    let tmp = dir.join("clip.tmp.mp4");
    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=1"])
        .args(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1"])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2", "-shortest"])
        .arg(&tmp)
        .status()
        .ok()?;
    (status.success() && std::fs::rename(&tmp, &out).is_ok()).then_some(dir)
}

fn mount_clip() -> Option<String> {
    let dir = test_clip()?;
    assert!(crate::filesystem::register_guest_path_mount("/avtest", &dir));
    Some("/avtest/clip.mp4".to_string())
}

fn init_ex(env: &Env, buffers: u32) -> u64 {
    init_ex_with(env, buffers, false)
}

/// PS5 SceAvPlayerInitDataEx: auto start at 0x74, nonzero thread
/// parameters in 0x78..0x228, the buffer count at 0x228.
fn init_ex_with(env: &Env, buffers: u32, auto_start: bool) -> u64 {
    let init = env.scratch;
    let ctx = env.ctx();
    ctx.write_bytes(init, &[0u8; 0x240]);
    ctx.write_u64(init, 0x238);
    write_allocators(&ctx, init + 8);
    ctx.write_u64(init + 88, EVENT_OBJECT);
    ctx.write_u64(init + 96, FN_EVENT);
    ctx.write_bytes(init + 0x74, &[auto_start as u8]);
    ctx.write_bytes(init + 0x78, &[1u8; 0x228 - 0x78]);
    ctx.write_u32(init + 552, buffers);
    let out = init + 0x300;
    assert_eq!(env.call(cb_init_ex, &[init, out]), OK);
    ctx.read_u64(out).unwrap()
}

fn add_source_ex(env: &Env, h: u64, path: &str) -> HleResult {
    let ctx = env.ctx();
    let details = env.scratch + 0x400;
    let name = env.scratch + 0x600;
    ctx.write_bytes(details, &[0u8; 128]);
    ctx.write_bytes(name, path.as_bytes());
    ctx.write_u64(details, name);
    ctx.write_u32(details + 8, path.len() as u32);
    env.call(cb_add_source_ex, &[h, 0, details])
}

#[test]
fn exports_match_nids() {
    let mut seen = std::collections::HashSet::new();
    for &(nid, name, _, _) in EXPORTS.iter().chain(crate::videodec2::EXPORTS) {
        assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        assert!(seen.insert(nid), "duplicate {nid}");
    }
}

fn video_stream(width: u32, height: u32) -> StreamInfo {
    StreamInfo {
        media_type: MediaType::Video,
        decodable: true,
        webm_ok: false,
        width,
        height,
        channels: 0,
        sample_rate: 0,
        frame_size: 0,
        fps: 29.97,
        sar: 1.0,
        full_range: false,
        color_primaries: 1,
        color_trc: 1,
        duration_ms: 1234,
        language: *b"und\0",
    }
}

#[test]
fn layouts_and_paths() {
    assert_eq!(video_dims(false, &video_stream(1920, 1080)), (1920, 1088, 2048));
    assert_eq!(video_dims(false, &video_stream(378, 150)), (384, 160, 512));
    // Software2 decoder: no alignment of the picture.
    assert_eq!(video_dims(true, &video_stream(378, 150)), (378, 150, 512));
    assert_eq!(source_type_from_path("/app0/movie.MP4"), SOURCE_TYPE_FILE_MP4);
    assert_eq!(source_type_from_path("/app0/a.webm"), SOURCE_TYPE_FILE_WEBM);
    assert_eq!(source_type_from_path("http://x/y.m3u8?a"), SOURCE_TYPE_HLS);
    assert_eq!(source_type_from_path("/app0/movie"), SOURCE_TYPE_UNKNOWN);
    let c = path_candidates("file://../../../shpc/Content/Movies/a%20b.mp4");
    assert!(c.contains(&"/app0/shpc/Content/Movies/a b.mp4".to_string()), "{c:?}");
    let s = video_stream(1920, 1080);
    let b = encode_stream_info_ex(false, &s);
    assert_eq!(u64::from_le_bytes(b[0..8].try_into().unwrap()), 104);
    assert_eq!(u32::from_le_bytes(b[8..12].try_into().unwrap()), STREAM_TYPE_VIDEO);
    assert_eq!(u32::from_le_bytes(b[16..20].try_into().unwrap()), 1920);
    assert_eq!(u32::from_le_bytes(b[20..24].try_into().unwrap()), 1088);
    assert_eq!(f32::from_le_bytes(b[24..28].try_into().unwrap()), 1.0);
    assert_eq!(&b[28..31], b"und");
    assert!(b[36..52].iter().all(|&x| x == 0), "stream info carries no crop");
    assert_eq!(u32::from_le_bytes(b[52..56].try_into().unwrap()), 2048); // pitch
    assert_eq!(&b[56..58], &[8, 8]);
    assert_eq!(f64::from_le_bytes(b[64..72].try_into().unwrap()), 29.97);
    assert_eq!(u64::from_le_bytes(b[96..104].try_into().unwrap()), 1234);
    let l = encode_stream_info(false, &s);
    assert_eq!(u32::from_le_bytes(l[8..12].try_into().unwrap()), 1920);
    assert_eq!(u64::from_le_bytes(l[24..32].try_into().unwrap()), 1234);
}

// ---------------------------------------------------------------------------
// File streamer (port of the AvPlayer file tests)
// ---------------------------------------------------------------------------

fn payload() -> Vec<u8> {
    (0..128 * 1024 + 37).map(|i| (i * 37) as u8).collect()
}

/// Raw read/seek semantics, then buffered reads; returns the streamer.
fn check_stream(mut s: FileStreamer, data: &[u8]) -> FileStreamer {
    let size = data.len() as i64;
    assert_eq!(s.seek(0, AVSEEK_SIZE), size, "query size");
    assert_eq!(s.seek(65531, SEEK_SET | AVSEEK_FORCE), 65531, "seek across block boundary");
    let mut bytes = [0u8; 32];
    assert_eq!(s.read(&mut bytes), 32, "read across block boundary");
    assert_eq!(&bytes[..], &data[65531..65531 + 32]);
    assert_eq!(s.seek(-16, SEEK_CUR), 65547, "seek relative to cursor");
    assert!(
        s.seek(i64::MIN, SEEK_CUR) < 0 && s.seek(i64::MAX, SEEK_END) < 0 && s.seek(0, 12345) < 0,
        "reject invalid and overflowing seeks"
    );
    assert_eq!(s.seek(0, SEEK_CUR), 65547, "rejected seeks preserve the cursor");
    assert_eq!(s.seek(-7, SEEK_END), size - 7, "seek near end");
    assert_eq!(s.read(&mut bytes), 7, "short read at end");
    assert_eq!(&bytes[..7], &data[data.len() - 7..]);
    assert_eq!(s.read(&mut bytes), AVERROR_EOF, "report EOF");
    assert_eq!(s.seek(11, SEEK_END), size + 11);
    assert_eq!(s.read(&mut bytes), AVERROR_EOF, "read beyond end");
    assert_eq!(s.seek(0, SEEK_SET), 0, "rewind");
    let mut io = match BufferedIo::new(s) {
        Ok(io) => io,
        Err(s) => {
            assert!(!host::available());
            return s;
        }
    };
    let mut buffered = vec![0u8; data.len()];
    assert_eq!(io.read(&mut buffered), data.len() as i32);
    assert!(buffered == data, "FFmpeg buffered reads preserve content");
    assert_eq!(io.seek(123, SEEK_SET), 123);
    assert_eq!(io.read(&mut bytes), 32);
    assert_eq!(&bytes[..], &data[123..155], "FFmpeg seeks reset buffered reads");
    io.into_streamer()
}

#[test]
fn file_streamer_native() {
    let data = payload();
    let dir = std::env::temp_dir().join(format!("rps5e-avio-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("movie.bin"), &data).unwrap();
    assert!(crate::filesystem::register_guest_path_mount("/avio", &dir));
    let s = open_streamer(FileReplacement::default(), "/avio/movie.bin").expect("open native stream");
    drop(check_stream(s, &data));
    assert!(open_streamer(FileReplacement::default(), "/unmounted/movie.bin").is_none(), "reject unmapped media");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn file_streamer_guest_callbacks() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    set_guest_call(stub_guest);
    let data = payload();
    *CALLBACKS.lock().unwrap() = Some(Callbacks { data: data.clone(), ..Default::default() });
    let file = FileReplacement { object: CB_OBJECT, open: FN_CB_OPEN, close: FN_CB_CLOSE, read_offset: FN_CB_READ, size: FN_CB_SIZE };
    {
        let s = open_streamer(file, "callback").expect("open guest callback stream");
        let mut s = check_stream(s, &data);
        CALLBACKS.lock().unwrap().as_mut().unwrap().premature_eof = true;
        assert_eq!(s.seek(0, SEEK_SET), 0, "rewind callback");
        let mut byte = [0u8; 1];
        assert_eq!(s.read(&mut byte), AVERROR_EIO, "reject premature EOF");
    }
    let c = CALLBACKS.lock().unwrap().take().unwrap();
    assert!(c.opens == 1 && c.closes == 1, "close callback exactly once ({} / {})", c.opens, c.closes);

    // Incomplete callback sets and empty files are rejected.
    let partial = FileReplacement { close: 0, ..file };
    assert!(open_streamer(partial, "callback").is_none());
    *CALLBACKS.lock().unwrap() = Some(Callbacks::default());
    assert!(open_streamer(file, "callback").is_none(), "empty file");
    let c = CALLBACKS.lock().unwrap().take().unwrap();
    assert_eq!((c.opens, c.closes), (1, 1));
}

// ---------------------------------------------------------------------------
// Library semantics without a movie
// ---------------------------------------------------------------------------

#[test]
fn init_and_parameter_errors() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let env = Env::new();
    let ctx = env.ctx();
    let init = env.scratch;
    // A memory replacement is mandatory.
    ctx.write_bytes(init, &[0u8; 0x240]);
    assert_eq!(env.call(cb_init, &[init]), 0);
    assert_eq!(env.call(cb_init_ex, &[init, init + 0x300]), err(ERROR_INVALID_PARAMS));
    assert_eq!(env.call(cb_init_ex, &[0, init + 0x300]), err(ERROR_INVALID_PARAMS));

    let h = init_ex(&env, 0);
    assert_eq!(lock(&player(h).unwrap()).video_buffers, 2);
    let h2 = init_ex(&env, 40);
    assert_eq!(lock(&player(h2).unwrap()).video_buffers, 16);
    assert_eq!(env.call(cb_close, &[h2]), OK);

    assert_eq!(env.call(cb_post_init, &[h, 0]), OK);
    assert_eq!(env.call(cb_post_init, &[0xDEAD, 0]), err(ERROR_INVALID_PARAMS));
    assert_eq!(env.call(cb_stream_count, &[h]), 0);
    assert_eq!(env.call(cb_is_active, &[h]), 1, "no source counts as active");
    assert_eq!(env.call(cb_is_active, &[0xDEAD]), 0);
    assert_eq!(env.call(cb_current_time, &[0xDEAD]), 0);
    assert_eq!(env.call(cb_current_time, &[h]), 0);
    let info = env.scratch + 0x800;
    assert_eq!(env.call(cb_get_stream_info, &[h, 0, info]), err(ERROR_OPERATION_FAILED));
    assert_eq!(env.call(cb_get_stream_info, &[h, 0, 0]), err(ERROR_INVALID_PARAMS));
    let no_source: [HleCallback; 7] = [cb_start, cb_stop, cb_pause, cb_resume, cb_enable_stream, cb_disable_stream, cb_set_looping];
    for cb in no_source {
        assert_eq!(env.call(cb, &[h, 0]), err(ERROR_INVALID_PARAMS));
    }
    assert_eq!(env.call(cb_jump_to_time, &[h, 0]), err(ERROR_INVALID_PARAMS));
    assert_eq!(env.call(cb_set_trick_speed, &[h, 100]), err(ERROR_OPERATION_FAILED));
    assert_eq!(env.call(cb_set_trick_speed, &[h, 0]), err(ERROR_OPERATION_FAILED), "the source is checked before the speed");
    assert_eq!(env.call(cb_set_trick_speed, &[0xDEAD, 100]), err(ERROR_INVALID_PARAMS));
    assert_eq!(env.call(cb_set_av_sync_mode, &[h, 1]), OK);
    assert_eq!(env.call(cb_set_av_sync_mode, &[h, 2]), err(ERROR_NOT_SUPPORTED));
    assert_eq!(env.call(cb_set_available_bandwidth, &[h, 0, 10, 5]), err(ERROR_INVALID_PARAMS));
    assert_eq!(env.call(cb_set_available_bandwidth, &[h, 0, 5, 10]), OK);
    assert_eq!(env.call(cb_set_available_bw, &[h, 8_000_000]), OK);
    assert_eq!(env.call(cb_set_available_bw, &[0xDEAD, 8_000_000]), err(ERROR_INVALID_PARAMS));
    assert_eq!(env.call(cb_get_video_data_ex, &[h, env.scratch + 0x900]), 0);
    assert_eq!(env.call(cb_get_audio_data, &[h, env.scratch + 0x900]), 0);

    // AddSourceEx parameter checks.
    let details = env.scratch + 0x400;
    ctx.write_bytes(details, &[0u8; 128]);
    assert_eq!(env.call(cb_add_source_ex, &[h, 1, details]), err(ERROR_INVALID_PARAMS));
    assert_eq!(env.call(cb_add_source_ex, &[h, 0, details]), err(ERROR_INVALID_PARAMS), "no uri");
    let name = env.scratch + 0x600;
    ctx.write_bytes(name, b"/x/a.m3u8");
    ctx.write_u64(details, name);
    ctx.write_u32(details + 8, 9);
    ctx.write_u32(details + 80, 5);
    assert_eq!(env.call(cb_add_source_ex, &[h, 0, details]), err(ERROR_INVALID_PARAMS), "bad source type");
    ctx.write_u32(details + 80, 0);
    assert_eq!(env.call(cb_add_source_ex, &[h, 0, details]), err(ERROR_NOT_SUPPORTED), "HLS");
    assert_eq!(add_source_ex(&env, h, "/app0/does/not/exist.mp4"), err(ERROR_OPERATION_FAILED));
    assert_eq!(env.call(cb_close, &[h]), OK);
    assert_eq!(env.call(cb_close, &[h]), err(ERROR_INVALID_PARAMS));
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------

#[test]
fn plays_clip_into_guest_memory() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !host::available() {
        eprintln!("skipping: built without FFmpeg");
        return;
    }
    let Some(path) = mount_clip() else {
        eprintln!("skipping: ffmpeg tool with libx264/aac unavailable");
        return;
    };
    let env = Env::new();
    EVENTS.lock().unwrap().clear();
    let allocs_before = ALLOCS.load(Ordering::SeqCst);
    let frees_before = FREES.load(Ordering::SeqCst);

    let h = init_ex(&env, 3);
    assert_ne!(h, 0);
    assert_eq!(add_source_ex(&env, h, &path), OK);
    assert_eq!(add_source_ex(&env, h, &path), err(ERROR_OPERATION_FAILED), "one source per player");
    assert_eq!(events(), vec![EVENT_STATE_READY]);
    assert_eq!(env.call(cb_is_active, &[h]), 1);
    assert_eq!(env.call(cb_stream_count, &[h]), 2);
    assert_eq!(ALLOCS.load(Ordering::SeqCst), allocs_before, "buffers are allocated on start");

    let ctx = env.ctx();
    let info = env.scratch + 0x800;
    ctx.write_u64(info, 0);
    assert_eq!(env.call(cb_get_stream_info_ex, &[h, 0, info]), OK);
    assert_eq!(ctx.read_u64(info), Some(104));
    assert_eq!(ctx.read_u32(info + 8), Some(STREAM_TYPE_VIDEO));
    assert_eq!(ctx.read_u32(info + 16), Some(320));
    assert_eq!(ctx.read_u32(info + 20), Some(192));
    assert_eq!(ctx.read_u32(info + 52), Some(512));
    assert_eq!(env.call(cb_get_stream_info, &[h, 1, info]), OK);
    assert_eq!(ctx.read_u32(info), Some(STREAM_TYPE_AUDIO));
    assert_eq!(ctx.read_bytes(info + 8, 2).unwrap(), vec![2, 0]);
    assert_eq!(ctx.read_u32(info + 12), Some(48000));
    assert_eq!(ctx.read_u32(info + 16), Some(0));
    let duration = ctx.read_u64(info + 24).unwrap();
    assert!((900..=1100).contains(&duration), "duration {duration}");
    assert_eq!(env.call(cb_get_stream_info, &[h, 2, info]), err(ERROR_OPERATION_FAILED));

    assert_eq!(env.call(cb_enable_stream, &[h, 0]), OK);
    assert_eq!(env.call(cb_enable_stream, &[h, 1]), OK);
    assert_eq!(env.call(cb_enable_stream, &[h, 2]), err(ERROR_OPERATION_FAILED));
    assert_eq!(env.call(cb_start, &[h]), OK);
    assert_eq!(events(), vec![EVENT_STATE_READY, EVENT_STATE_PLAY]);
    // 3 texture buffers, the decode-ahead pictures and 8 audio buffers
    // through the title allocator.
    assert_eq!(ALLOCS.load(Ordering::SeqCst) - allocs_before, (3 + VIDEO_DECODE_AHEAD_FRAMES + AUDIO_BUFFERS) as u64);
    assert_eq!(env.call(cb_enable_stream, &[h, 0]), err(ERROR_OPERATION_FAILED));
    assert_eq!(env.call(cb_disable_stream, &[h, 0]), err(ERROR_OPERATION_FAILED));

    let frame = env.scratch + 0x900;
    let audio = env.scratch + 0xA00;
    let mut pictures = Vec::new();
    let mut audio_bytes = 0u64;
    let mut audio_energy = 0u64;
    let started = Instant::now();
    let deadline = started + Duration::from_secs(10);
    while env.call(cb_is_active, &[h]) == 1 {
        assert!(Instant::now() < deadline, "playback did not finish");
        // Pull audio at the rate an audio output port would consume it.
        let due = started.elapsed().as_secs_f64() * 48000.0 * 4.0 + 8192.0;
        while (audio_bytes as f64) < due && env.call(cb_get_audio_data, &[h, audio]) == 1 {
            let data = ctx.read_u64(audio).unwrap();
            let size = ctx.read_u32(audio + 32).unwrap();
            assert_eq!(ctx.read_bytes(audio + 24, 2).unwrap(), vec![2, 0]);
            assert_eq!(ctx.read_u32(audio + 28), Some(48000));
            assert!(size > 0 && size <= 2 * 1024 * 2, "audio size {size}");
            let pcm = ctx.read_bytes(data, size as usize).unwrap();
            audio_energy += pcm.chunks(2).map(|s| i16::from_le_bytes([s[0], s[1]]).unsigned_abs() as u64).sum::<u64>();
            audio_bytes += size as u64;
        }
        if env.call(cb_get_video_data_ex, &[h, frame]) == 1 {
            let data = ctx.read_u64(frame).unwrap();
            let ts = ctx.read_u64(frame + 16).unwrap();
            assert_eq!(ctx.read_u32(frame + 24), Some(320)); // width
            assert_eq!(ctx.read_u32(frame + 28), Some(192)); // height
            assert_eq!(ctx.read_u32(frame + 48), Some(512 - 320)); // crop right
            assert_eq!(ctx.read_u32(frame + 56), Some(192 - 180)); // crop bottom
            assert_eq!(ctx.read_u32(frame + 60), Some(512)); // pitch
            assert_eq!(ctx.read_bytes(frame + 64, 2).unwrap(), vec![8, 8]);
            // testsrc: varied luma inside the picture, black padding.
            let row = ctx.read_bytes(data + 90 * 512, 512).unwrap();
            let distinct: std::collections::HashSet<_> = row[..320].iter().collect();
            assert!(distinct.len() > 4, "flat luma row");
            assert!(row[320..].iter().all(|&y| y == 16), "padding not black");
            let chroma = ctx.read_bytes(data + 512 * 192 + 45 * 512, 320).unwrap();
            assert!(chroma.iter().any(|&c| c != 128), "no chroma");
            pictures.push((ts, data));
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(started.elapsed() >= Duration::from_millis(800), "pictures not paced: {:?}", started.elapsed());
    assert!(pictures.len() >= 20, "only {} pictures", pictures.len());
    assert!(pictures.windows(2).all(|w| w[0].0 < w[1].0), "timestamps not increasing");
    let buffers: std::collections::HashSet<_> = pictures.iter().map(|p| p.1).collect();
    assert_eq!(buffers.len(), 3 + VIDEO_DECODE_AHEAD_FRAMES, "pictures should rotate through the output buffers");
    assert!(audio_bytes >= 48000 * 4 * 8 / 10, "only {audio_bytes} audio bytes");
    assert!(audio_energy > 0, "silent audio");
    assert_eq!(env.call(cb_get_video_data_ex, &[h, frame]), 0);
    assert_eq!(events(), vec![EVENT_STATE_READY, EVENT_STATE_PLAY, EVENT_STATE_STOP], "STOP at the end of the stream");
    assert_eq!(env.call(cb_is_active, &[h]), 0);
    assert_eq!(events().len(), 3, "the end is reported once");

    assert_eq!(env.call(cb_close, &[h]), OK);
    assert_eq!(events().last(), Some(&EVENT_STATE_STOP), "closing a playing player stops it");
    assert_eq!(FREES.load(Ordering::SeqCst) - frees_before, (3 + VIDEO_DECODE_AHEAD_FRAMES + AUDIO_BUFFERS) as u64);
    assert_eq!(env.call(cb_close, &[h]), err(ERROR_INVALID_PARAMS));
}

#[test]
fn looping_jump_and_pause() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !host::available() {
        return;
    }
    let Some(path) = mount_clip() else { return };
    let env = Env::new();
    EVENTS.lock().unwrap().clear();
    let h = init_ex(&env, 2);
    assert_eq!(add_source_ex(&env, h, &path), OK);
    assert_eq!(env.call(cb_set_looping, &[h, 1]), OK);
    assert_eq!(env.call(cb_set_av_sync_mode, &[h, SYNC_MODE_NONE as u64]), OK);
    assert_eq!(env.call(cb_set_av_sync_mode, &[h, 7]), err(ERROR_NOT_SUPPORTED));
    assert_eq!(env.call(cb_set_trick_speed, &[h, 200]), err(ERROR_NOT_SUPPORTED));
    assert_eq!(env.call(cb_start, &[h]), OK); // auto-enables video + audio
    assert_eq!(env.call(cb_current_time, &[h]) as u64 / 1000, 0);

    // Without A/V sync audio does not drive the clock, but pictures still
    // come on presentation time; 75 pictures span the 30-picture clip more
    // than twice. A title polling in a tight loop never gets a whole pass at
    // once: the looped pass starts one frame after the previous one ended.
    let frame = env.scratch + 0x900;
    let audio = env.scratch + 0xA00;
    let mut stamps = Vec::new();
    let mut times = Vec::new();
    let start = Instant::now();
    let deadline = start + Duration::from_secs(20);
    while stamps.len() < 75 {
        assert!(Instant::now() < deadline, "only {} pictures", stamps.len());
        while env.call(cb_get_audio_data, &[h, audio]) == 1 {}
        if env.call(cb_get_video_data_ex, &[h, frame]) == 1 {
            stamps.push(env.ctx().read_u64(frame + 16).unwrap());
            times.push(Instant::now());
        } else {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    let wraps = stamps.windows(2).filter(|w| w[1] < w[0]).count();
    assert!(wraps >= 2, "never looped: {stamps:?}");
    let wrap = stamps.windows(2).position(|w| w[1] < w[0]).unwrap();
    assert!(times[wrap + 1] - times[wrap] >= Duration::from_millis(20), "looped pass delivered early");
    assert!(start.elapsed() >= Duration::from_millis(2000), "pictures not paced: {:?}", start.elapsed());
    let ev = EVENTS.lock().unwrap().clone();
    let loops = ev.iter().filter(|&&(id, w)| id == EVENT_WARNING_ID && w == WARNING_LOOPING_BACK).count();
    assert!(loops >= 2, "{ev:?}");
    assert_eq!(env.call(cb_is_active, &[h]), 1);

    // Trick play blocks audio.
    assert_eq!(env.call(cb_set_trick_speed, &[h, 800]), err(ERROR_NOT_SUPPORTED));
    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(env.call(cb_get_audio_data, &[h, audio]), 0);
    assert_eq!(env.call(cb_set_trick_speed, &[h, 100]), OK);

    // Pause, then jump: exactly one picture while paused.
    assert_eq!(env.call(cb_pause, &[h]), OK);
    assert_eq!(env.call(cb_jump_to_time, &[h, 500]), OK);
    let ev = EVENTS.lock().unwrap().clone();
    assert!(ev.iter().any(|&(id, w)| id == EVENT_WARNING_ID && w == WARNING_JUMP_COMPLETE));
    assert_eq!(env.call(cb_current_time, &[h]), 500);
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        assert!(Instant::now() < deadline, "no seek picture");
        if env.call(cb_get_video_data_ex, &[h, frame]) == 1 {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let ts = env.ctx().read_u64(frame + 16).unwrap();
    assert!(ts <= 500, "seek landed after the target: {ts}");
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(env.call(cb_get_video_data_ex, &[h, frame]), 0, "paused player delivered a picture");
    assert_eq!(env.call(cb_get_audio_data, &[h, audio]), 0);
    assert_eq!(env.call(cb_resume, &[h]), OK);
    assert_eq!(env.call(cb_stop, &[h]), OK);
    assert_eq!(events().last(), Some(&EVENT_STATE_STOP));
    assert_eq!(env.call(cb_is_active, &[h]), 0);
    assert_eq!(env.call(cb_jump_to_time, &[h, 0]), err(ERROR_OPERATION_FAILED));
    assert_eq!(env.call(cb_pause, &[h]), err(ERROR_OPERATION_FAILED));
    assert_eq!(env.call(cb_close, &[h]), OK);
}

#[test]
fn reads_through_file_replacement() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !host::available() {
        return;
    }
    let Some(dir) = test_clip() else { return };
    *REPLACEMENT.lock().unwrap() = Some((dir.join("clip.mp4"), None));
    let env = Env::new();
    EVENTS.lock().unwrap().clear();
    let init = env.scratch;
    let ctx = env.ctx();
    // Legacy init data: allocators, file callbacks, no event callback (so
    // the player starts on its own).
    ctx.write_bytes(init, &[0u8; 0x80]);
    write_allocators(&ctx, init);
    ctx.write_u64(init + 40, FILE_OBJECT);
    ctx.write_u64(init + 48, FN_OPEN);
    ctx.write_u64(init + 56, FN_CLOSE);
    ctx.write_u64(init + 64, FN_READ);
    ctx.write_u64(init + 72, FN_SIZE);
    ctx.write_u32(init + 104, 4);
    let h = env.call(cb_init, &[init]) as u64;
    assert_ne!(h, 0);
    let name = env.scratch + 0x600;
    ctx.write_bytes(name, b"virtual:clip\0");
    assert_eq!(env.call(cb_add_source, &[h, name]), OK);
    // Auto-started; the demuxer's reads are served by these calls.
    let frame = env.scratch + 0x900;
    let mut n = 0;
    let started = Instant::now();
    let deadline = started + Duration::from_secs(10);
    let mut audio_bytes = 0u64;
    while env.call(cb_is_active, &[h]) == 1 {
        assert!(Instant::now() < deadline, "playback did not finish");
        // Playback only ends once the audio was consumed too; it is pulled
        // at the rate an audio output port would consume it (audio drives
        // the clock, so draining it at once would make the pictures late).
        let due = started.elapsed().as_secs_f64() * 48000.0 * 4.0 + 8192.0;
        while (audio_bytes as f64) < due && env.call(cb_get_audio_data, &[h, env.scratch + 0xA00]) == 1 {
            audio_bytes += ctx.read_u32(env.scratch + 0xA00 + 32).unwrap() as u64;
        }
        if env.call(cb_get_video_data, &[h, frame]) == 1 {
            assert_eq!(ctx.read_u32(frame + 24), Some(320));
            assert_eq!(ctx.read_u32(frame + 28), Some(192));
            let data = ctx.read_u64(frame).unwrap();
            let row = ctx.read_bytes(data + 90 * 512, 320).unwrap();
            assert!(row.iter().collect::<std::collections::HashSet<_>>().len() > 4);
            n += 1;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(n >= 20, "only {n} pictures");
    assert!(events().is_empty());
    assert!(REPLACEMENT.lock().unwrap().as_ref().unwrap().1.is_some(), "file closed early");
    assert_eq!(env.call(cb_close, &[h]), OK);
    assert!(REPLACEMENT.lock().unwrap().as_ref().unwrap().1.is_none(), "file not closed");
}

/// Raw H.264 access units through libSceVideodec2.
#[test]
fn videodec2_decodes_access_units() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !host::available() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("rps5e-vdec2-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let es = dir.join("clip.h264");
    let ok = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=176x144:rate=25:duration=1"])
        .args(["-c:v", "libx264", "-bf", "2", "-pix_fmt", "yuv420p", "-bsf:v", "h264_metadata=aud=insert", "-f", "h264"])
        .arg(&es)
        .status()
        .is_ok_and(|s| s.success());
    if !ok {
        return;
    }
    let stream = std::fs::read(&es).unwrap();
    // Split into access units at access unit delimiters (NAL type 9).
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 4 < stream.len() {
        if stream[i..i + 3] == [0, 0, 1] && stream[i + 3] & 0x1F == 9 {
            starts.push(if i > 0 && stream[i - 1] == 0 { i - 1 } else { i });
            i += 3;
        }
        i += 1;
    }
    starts.push(stream.len());
    let aus: Vec<&[u8]> = starts.windows(2).map(|w| &stream[w[0]..w[1]]).collect();
    assert_eq!(aus.len(), 25);

    let vd = crate::videodec2::EXPORTS;
    let f = |name: &str| vd.iter().find(|e| e.1 == name).unwrap().3;
    let env = Env::new();
    let ctx = env.ctx();
    let s = env.scratch;
    let (config, memory, handle_out, input, fb, out, pic) = (s, s + 0x100, s + 0x200, s + 0x300, s + 0x400, s + 0x500, s + 0x600);
    ctx.write_bytes(s, &[0u8; 0x800]);
    ctx.write_u64(config, 72);
    ctx.write_u32(config + 8, 1); // compute
    ctx.write_u32(config + 12, 1); // AVC
    ctx.write_u32(config + 24, 1920);
    ctx.write_u32(config + 28, 1088);
    ctx.write_u32(config + 32, u32::MAX); // -1 dpb
    ctx.write_u32(config + 36, 1);
    ctx.write_u64(config + 40, 0xC0FFEE);
    ctx.write_u64(memory, 72);
    assert_eq!(env.call(f("sceVideodec2QueryDecoderMemoryInfo"), &[config, memory]), OK);
    let mem_size = ctx.read_u64(memory + 8).unwrap();
    assert_eq!(mem_size, 16 << 20);
    for off in [16, 32, 48] {
        ctx.write_u64(memory + off, 0x1000);
    }
    assert_eq!(env.call(f("sceVideodec2CreateDecoder"), &[config, memory, handle_out]), OK);
    let dec = ctx.read_u64(handle_out).unwrap();

    let au_buf = arena_alloc(1 << 20).unwrap();
    let frame_buf = arena_alloc(1 << 20).unwrap();
    // The access unit is checked before the frame buffer.
    ctx.write_u64(input, 48);
    ctx.write_u64(fb, 32);
    ctx.write_u64(out, 56);
    assert_eq!(env.call(f("sceVideodec2Decode"), &[dec, input, fb, out]), err(0x811D_010D));
    ctx.write_u64(input + 16, 16);
    assert_eq!(env.call(f("sceVideodec2Decode"), &[dec, input, fb, out]), err(0x811D_010E));
    ctx.write_u64(input + 8, au_buf);
    assert_eq!(env.call(f("sceVideodec2Decode"), &[dec, input, fb, out]), err(0x811D_0106));
    ctx.write_u64(out, 40);
    assert_eq!(env.call(f("sceVideodec2Decode"), &[dec, input, fb, out]), err(0x811D_0101));

    let mut pictures = Vec::new();
    let mut collect = |env: &Env| {
        let ctx = env.ctx();
        if ctx.read_bytes(out + 8, 1).unwrap()[0] == 1 {
            assert_eq!(ctx.read_u32(out + 16), Some(176));
            assert_eq!(ctx.read_u32(out + 20), Some(256));
            assert_eq!(ctx.read_u32(out + 24), Some(144));
            assert_eq!(ctx.read_bytes(fb + 24, 1).unwrap()[0], 1);
            ctx.write_u64(pic, 120);
            assert_eq!(env.call(f("sceVideodec2GetAvcPictureInfo"), &[out, pic, 0]), OK);
            pictures.push(ctx.read_u64(pic + 16).unwrap());
            let luma = ctx.read_bytes(frame_buf + 70 * 256, 176).unwrap();
            assert!(luma.iter().collect::<std::collections::HashSet<_>>().len() > 4);
            // Padding is left zero.
            assert!(ctx.read_bytes(frame_buf + 70 * 256 + 176, 80).unwrap().iter().all(|&b| b == 0));
        }
    };
    for (n, au) in aus.iter().enumerate() {
        ctx.write_bytes(au_buf, au);
        ctx.write_u64(input, 48);
        ctx.write_u64(input + 8, au_buf);
        ctx.write_u64(input + 16, au.len() as u64);
        ctx.write_u64(input + 24, n as u64 * 1000); // pts
        ctx.write_u64(input + 32, n as u64);
        ctx.write_u64(fb, 32);
        ctx.write_u64(fb + 8, frame_buf);
        ctx.write_u64(fb + 16, 1 << 20);
        ctx.write_u64(out, 56);
        assert_eq!(env.call(f("sceVideodec2Decode"), &[dec, input, fb, out]), OK);
        collect(&env);
    }
    loop {
        ctx.write_u64(out, 56);
        assert_eq!(env.call(f("sceVideodec2Flush"), &[dec, fb, out]), OK);
        if ctx.read_bytes(out + 8, 1).unwrap()[0] == 0 {
            break;
        }
        collect(&env);
    }
    // Flush drains the decoder: every access unit came out as a picture.
    assert_eq!(pictures.len(), 25, "only {} pictures", pictures.len());
    assert!(pictures.windows(2).all(|w| w[0] < w[1]), "pictures out of order: {pictures:?}");
    assert_eq!(env.call(f("sceVideodec2Reset"), &[dec]), OK);
    ctx.write_u64(pic, 120);
    assert_eq!(env.call(f("sceVideodec2GetPictureInfo"), &[out, pic, 0]), err(0x811D_010F), "reset drops picture info");
    assert_eq!(env.call(f("sceVideodec2DeleteDecoder"), &[dec]), OK);
    assert_eq!(env.call(f("sceVideodec2DeleteDecoder"), &[dec]), err(0x811D_0103));
    arena_free(au_buf);
    arena_free(frame_buf);
}

/// SILENT HILL's title-screen movie (SILENT HILL PPSA10112 under `GAMES_DIR`; skipped when unset or absent).
#[test]
fn decodes_game_title_movie() {
    let Some(dir) = ps5vm::test_games::game_dir("PPSA10112") else { return };
    let path = PathBuf::from(dir).join("shpc/content/movies/ui_titlemovie01.mp4");
    if !host::available() || !path.is_file() {
        return;
    }
    let media = Media::open(FileStreamer::open_native(&path).unwrap(), path.to_str()).unwrap();
    let streams = media.streams();
    let (vi, v) = streams.iter().enumerate().find(|s| s.1.media_type == MediaType::Video).unwrap();
    assert_eq!((v.width, v.height), (378, 150));
    let (_, rows, pitch) = video_dims(false, v);
    assert_eq!((rows, pitch), (160, 512));
    assert!(media.decoder(vi).is_some());
}

// ---------------------------------------------------------------------------
// SceAvPlayerInitDataEx layouts, auto start, late pictures
// ---------------------------------------------------------------------------

#[test]
fn init_ex_layout_follows_this_size() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(init_ex_offsets(0x230), (0x74, 0x228));
    assert_eq!(init_ex_offsets(0x238), (0x74, 0x228));
    assert_eq!(init_ex_offsets(0xB0), (0xA8, 0xA4));
    let env = Env::new();
    let ctx = env.ctx();
    // PS5 layout: thread priorities of 700 (and the old auto start / count
    // offsets inside them) are not read as auto start or buffer count.
    let init = env.scratch;
    ctx.write_bytes(init, &[0u8; 0x240]);
    ctx.write_u64(init, 0x230);
    write_allocators(&ctx, init + 8);
    ctx.write_u64(init + 88, EVENT_OBJECT);
    ctx.write_u64(init + 96, FN_EVENT);
    for o in (0x78..0x228).step_by(4) {
        ctx.write_u32(init + o, 700);
    }
    ctx.write_u32(init + 0x228, 3);
    assert_eq!(env.call(cb_init_ex, &[init, init + 0x300]), OK);
    let h = ctx.read_u64(init + 0x300).unwrap();
    {
        let p = player(h).unwrap();
        let g = lock(&p);
        assert_eq!(g.video_buffers, 3, "buffer count not read at 0x228");
        assert!(!g.auto_start, "auto start not read at 0x74");
    }
    assert_eq!(env.call(cb_close, &[h]), OK);
    ctx.write_bytes(init + 0x74, &[1]);
    assert_eq!(env.call(cb_init_ex, &[init, init + 0x300]), OK);
    let h = ctx.read_u64(init + 0x300).unwrap();
    assert!(lock(&player(h).unwrap()).auto_start, "auto start at 0x74 ignored");
    assert_eq!(env.call(cb_close, &[h]), OK);

    // A smaller structure keeps the older offsets.
    ctx.write_bytes(init, &[0u8; 0x240]);
    ctx.write_u64(init, 0xB0);
    write_allocators(&ctx, init + 8);
    ctx.write_u64(init + 88, EVENT_OBJECT);
    ctx.write_u64(init + 96, FN_EVENT);
    ctx.write_u32(init + 0xA4, 5);
    ctx.write_bytes(init + 0xA8, &[1]);
    assert_eq!(env.call(cb_init_ex, &[init, init + 0x300]), OK);
    let h = ctx.read_u64(init + 0x300).unwrap();
    {
        let p = player(h).unwrap();
        let g = lock(&p);
        assert_eq!(g.video_buffers, 5, "buffer count not read at 0xA4 of the older layout");
        assert!(g.auto_start, "auto start not read at 0xA8 of the older layout");
    }
    assert_eq!(env.call(cb_close, &[h]), OK);
}

#[test]
fn auto_start_keeps_playback_the_title_started() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !host::available() {
        return;
    }
    let Some(path) = mount_clip() else { return };
    let env = Env::new();
    EVENTS.lock().unwrap().clear();
    let h = init_ex_with(&env, 3, true);
    assert!(lock(&player(h).unwrap()).auto_start);
    *START_ON_READY.lock().unwrap() = Some(h);
    let frees_before = FREES.load(Ordering::SeqCst);
    assert_eq!(add_source_ex(&env, h, &path), OK);
    assert!(START_ON_READY.lock().unwrap().is_none(), "the READY callback did not run");
    assert_eq!(FREES.load(Ordering::SeqCst), frees_before, "auto start restarted playback the title had started");
    assert_eq!(events(), vec![EVENT_STATE_READY], "auto start reported a second start");
    let frame = env.scratch + 0x900;
    let deadline = Instant::now() + Duration::from_secs(5);
    while env.call(cb_get_video_data_ex, &[h, frame]) != 1 {
        assert!(Instant::now() < deadline, "no picture after the title's own start");
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(env.call(cb_close, &[h]), OK);

    // Without a start in the callback the auto start plays the movie.
    EVENTS.lock().unwrap().clear();
    let h = init_ex_with(&env, 3, true);
    assert_eq!(add_source_ex(&env, h, &path), OK);
    assert_eq!(events(), vec![EVENT_STATE_READY, EVENT_STATE_PLAY]);
    assert_eq!(env.call(cb_close, &[h]), OK);
}

#[test]
fn unsynced_video_keeps_up_with_audio() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !host::available() {
        return;
    }
    let Some(path) = mount_clip() else { return };
    let env = Env::new();
    let ctx = env.ctx();
    let h = init_ex(&env, 3);
    assert_eq!(add_source_ex(&env, h, &path), OK);
    assert_eq!(env.call(cb_set_av_sync_mode, &[h, SYNC_MODE_NONE as u64]), OK);
    assert_eq!(env.call(cb_start, &[h]), OK);
    let (frame, audio) = (env.scratch + 0x900, env.scratch + 0xA00);
    let deadline = Instant::now() + Duration::from_secs(5);
    while env.call(cb_get_audio_data, &[h, audio]) != 1 {
        assert!(Instant::now() < deadline, "no audio");
        std::thread::sleep(Duration::from_millis(1));
    }
    let first = ctx.read_u64(audio + 16).unwrap();
    let mut sound = first;
    let start = Instant::now();
    let (mut worst, mut pictures) = (0u64, 0);
    // The title asks for a picture every 66 ms, half the clip's rate, and
    // pulls audio in real time.
    while env.call(cb_is_active, &[h]) == 1 && start.elapsed() < Duration::from_millis(900) {
        let elapsed = start.elapsed().as_millis() as u64;
        while sound <= first + elapsed && env.call(cb_get_audio_data, &[h, audio]) == 1 {
            sound = ctx.read_u64(audio + 16).unwrap();
        }
        if env.call(cb_get_video_data_ex, &[h, frame]) == 1 {
            let ts = ctx.read_u64(frame + 16).unwrap();
            worst = worst.max(sound.saturating_sub(ts));
            pictures += 1;
        }
        std::thread::sleep(Duration::from_millis(66));
    }
    assert!(pictures >= 5, "only {pictures} pictures");
    assert!(worst <= 150, "video fell {worst} ms behind the audio");
    assert_eq!(env.call(cb_close, &[h]), OK);
}

/// The buffer of a picture goes back to the decoder as soon as the next
/// picture is handed out, so every output buffer but the one shown can hold
/// a decoded picture: after a stall, a title that retries at once gets that
/// many successive pictures. A failed get while paused keeps the shown
/// picture intact; closing frees every buffer.
#[test]
fn decoder_runs_ahead_into_every_buffer_but_the_shown_one() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !host::available() {
        return;
    }
    let Some(path) = mount_clip() else { return };
    let env = Env::new();
    let ctx = env.ctx();
    let (allocs_before, frees_before) = (ALLOCS.load(Ordering::SeqCst), FREES.load(Ordering::SeqCst));
    const BUFFERS: usize = 6;
    let h = init_ex(&env, BUFFERS as u32);
    assert_eq!(add_source_ex(&env, h, &path), OK);
    // No A/V sync and no audio pulled: due pictures are not skipped.
    assert_eq!(env.call(cb_set_av_sync_mode, &[h, SYNC_MODE_NONE as u64]), OK);
    assert_eq!(env.call(cb_start, &[h]), OK);
    let frame = env.scratch + 0x900;
    let mut last = None;
    for _ in 0..3 {
        let deadline = Instant::now() + Duration::from_secs(5);
        while env.call(cb_get_video_data_ex, &[h, frame]) != 1 {
            assert!(Instant::now() < deadline, "no steady-state picture");
            std::thread::sleep(Duration::from_millis(1));
        }
        last = ctx.read_u64(frame + 16);
    }
    // The title stalls long enough for more pictures to fall due than there
    // are buffers, then retries without waiting.
    std::thread::sleep(Duration::from_millis(450));
    let ready = BUFFERS + VIDEO_DECODE_AHEAD_FRAMES - 1;
    for i in 0..ready {
        assert_eq!(env.call(cb_get_video_data_ex, &[h, frame]), 1, "retry {i}: only {i} pictures were decoded ahead");
        let ts = ctx.read_u64(frame + 16);
        assert!(ts > last, "retry {i} did not return the next picture");
        last = ts;
    }
    // Paused: the get fails and the shown picture stays as it was.
    let shown = ctx.read_u64(frame).unwrap();
    let pixels = ctx.read_bytes(shown, 256).unwrap();
    assert_eq!(env.call(cb_pause, &[h]), OK);
    assert_eq!(env.call(cb_get_video_data_ex, &[h, frame]), 0);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(ctx.read_bytes(shown, 256).unwrap(), pixels, "the shown picture was reused while paused");
    assert_eq!(env.call(cb_close, &[h]), OK);
    assert_eq!(ALLOCS.load(Ordering::SeqCst) - allocs_before, FREES.load(Ordering::SeqCst) - frees_before, "close releases every buffer");
}

/// A fence the test moves by hand: tickets are `queued`, a ticket passed
/// when `done` reached it.
struct ManualFence {
    queued: AtomicU64,
    done: AtomicU64,
}

impl source::RecordFence for ManualFence {
    fn ticket(&self) -> u64 {
        self.queued.load(Ordering::SeqCst)
    }
    fn passed(&self, ticket: u64) -> bool {
        self.done.load(Ordering::SeqCst) >= ticket
    }
}

/// The buffer of a retired picture goes back to the decoder only after the
/// GPU queue processed the commands queued while it was shown (they may
/// sample it), so a command thread that lags the title never records parts
/// of a later picture.
#[test]
fn retired_picture_is_kept_until_the_gpu_queue_processed_it() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !host::available() {
        return;
    }
    let Some(path) = mount_clip() else { return };
    let fence = Arc::new(ManualFence { queued: AtomicU64::new(1), done: AtomicU64::new(0) });
    source::set_record_fence(Some(fence.clone()));
    let env = Env::new();
    let ctx = env.ctx();
    let h = init_ex(&env, 3);
    assert_eq!(add_source_ex(&env, h, &path), OK);
    assert_eq!(env.call(cb_set_av_sync_mode, &[h, SYNC_MODE_NONE as u64]), OK);
    assert_eq!(env.call(cb_start, &[h]), OK);
    let frame = env.scratch + 0x900;
    let next_picture = || {
        let deadline = Instant::now() + Duration::from_secs(5);
        while env.call(cb_get_video_data_ex, &[h, frame]) != 1 {
            assert!(Instant::now() < deadline, "no picture");
            std::thread::sleep(Duration::from_millis(1));
        }
        ctx.read_u64(frame).unwrap()
    };
    // The luma plane: the clip's frame counter changes it every picture.
    const LUMA: usize = 320 * 180;
    let first = next_picture();
    let pixels = ctx.read_bytes(first, LUMA).unwrap();
    // The title moves on; the commands queued while `first` was shown are
    // not processed yet.
    let second = next_picture();
    assert_ne!(second, first);
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(ctx.read_bytes(first, LUMA).unwrap(), pixels, "the decoder reused a picture the GPU queue may still read");
    // The command thread catches up: the buffer is decoded into again.
    fence.done.store(1, Ordering::SeqCst);
    let deadline = Instant::now() + Duration::from_secs(2);
    while ctx.read_bytes(first, LUMA).unwrap() == pixels {
        assert!(Instant::now() < deadline, "the processed picture never went back to the decoder");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(env.call(cb_close, &[h]), OK);
    source::set_record_fence(None);
}
