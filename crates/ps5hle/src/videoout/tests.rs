// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only
//
// Unit tests of the `ps5hle::videoout` module.

use super::exports::{
    buffer_attribute2_bytes, hdr_allowed, initial_output_options, output_status_bytes, parse_buffer_attribute,
    resolution_status_bytes, EXPORTS,
};
use super::*;
use crate::context::{CpuContext, CpuRegister::*};
use crate::sync::equeue;
use ps5gpu::tiling;

fn call(name: &str, ctx: &mut CpuContext, args: &[u64]) -> i64 {
    let cb = EXPORTS.iter().find(|e| e.1 == name).unwrap_or_else(|| panic!("{name}")).3;
    for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
        ctx.set(*r, *v);
    }
    cb(ctx)
}

fn headless() {
    ps5gpu::presenter().force_headless();
}

/// The ports, the flip table and the vblank clock are process-wide: tests
/// that open ports run one at a time (at most four ports may be open).
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn nids_match_names() {
    let mut n = 0;
    for &(nid, name, _, _) in EXPORTS {
        assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        n += 1;
    }
    assert!(n >= 34);
}

#[test]
fn attribute_layouts() {
    let a2 = buffer_attribute2_bytes(0x8000_0000_2200_0000, 0, 1920, 1080, 0x20, 7, 0x1234);
    assert_eq!(&a2[0x20..0x28], &0x8000_0000_2200_0000u64.to_le_bytes());
    assert_eq!(u32::from_le_bytes(a2[0x30..0x34].try_into().unwrap()), 7);
    assert_eq!(u64::from_le_bytes(a2[0x28..0x30].try_into().unwrap()), 0x1234);
    let p = parse_buffer_attribute(&a2, true).unwrap();
    assert_eq!(
        p,
        BufferAttribute {
            pixel_format: 0x8000_0000_2200_0000,
            tiling_mode: 0,
            aspect_ratio: 0,
            width: 1920,
            height: 1080,
            pitch_in_pixel: 1920,
            option: 0x20,
        }
    );

    // v1 through the export into guest memory.
    let vm = ps5vm::VirtualMemory::new();
    const BASE: u64 = 0x5B10_0000_0000;
    vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
    let mut ctx = CpuContext::new(&vm);
    // pitch is the first stack argument at [rsp+8].
    ctx.set(Rsp, BASE + 0x800);
    assert!(ctx.write_u64(BASE + 0x808, 2048));
    assert_eq!(call("sceVideoOutSetBufferAttribute", &mut ctx, &[BASE, 0x8000_2200, 0, 1, 1280, 720]), 0);
    let raw = ctx.read_bytes(BASE, 0x28).unwrap();
    let v1 = parse_buffer_attribute(&raw, false).unwrap();
    assert_eq!((v1.pixel_format, v1.aspect_ratio, v1.width, v1.height, v1.pitch_in_pixel), (0x8000_2200, 1, 1280, 720, 2048));

    // v2 through the export: dccControl at [rsp+8], dccClearColor at [rsp+0x10].
    assert!(ctx.write_u64(BASE + 0x808, 0x48));
    assert!(ctx.write_u64(BASE + 0x810, 0));
    assert_eq!(
        call("sceVideoOutSetBufferAttribute2", &mut ctx, &[BASE + 0x100, 0x8000_0000_0000_0000, 0, 640, 360, 0]),
        0
    );
    let raw = ctx.read_bytes(BASE + 0x100, 0x50).unwrap();
    let v2 = parse_buffer_attribute(&raw, true).unwrap();
    assert_eq!((v2.pixel_format, v2.width, v2.height), (0x8000_0000_0000_0000, 640, 360));
    assert_eq!(u32::from_le_bytes(raw[0x30..0x34].try_into().unwrap()), 0x48);

    let s = output_status_bytes(1920, 1080, 60, false);
    assert_eq!(u32::from_le_bytes(s[0..4].try_into().unwrap()), 1);
    assert_eq!(u64::from_le_bytes(s[8..16].try_into().unwrap()), 3);
    assert_eq!(u32::from_le_bytes(output_status_bytes(3840, 2160, 120, true)[0..4].try_into().unwrap()), 2);
}

#[test]
fn output_status_dynamic_range() {
    let sdr = output_status_bytes(1920, 1080, 60, false);
    let hdr = output_status_bytes(1920, 1080, 60, true);
    assert_eq!(u32::from_le_bytes(sdr[4..8].try_into().unwrap()), 1);
    assert_eq!(u32::from_le_bytes(hdr[4..8].try_into().unwrap()), 2);
    // Only the dynamic range differs; flags and reserved words stay zero.
    assert_eq!(sdr[..4], hdr[..4]);
    assert_eq!(sdr[8..], hdr[8..]);
    assert!(hdr[0x10..].iter().all(|&b| b == 0));
}

#[test]
fn hdr_needs_the_param_json_attribute() {
    // No param.json at all: allowed.
    assert!(hdr_allowed(None));
    // Bit 29 of "attribute" (SILENT HILL: The Short Message has 536870912).
    assert!(hdr_allowed(Some(r#"{"titleId": "PPSA10112", "attribute": 536870912, "attribute3": 0}"#)));
    assert!(hdr_allowed(Some(r#"{"attribute": 4026531840}"#)));
    // Without the bit, or without the member (defaults to 0): SDR.
    assert!(!hdr_allowed(Some(r#"{"attribute": 268435456}"#)));
    assert!(!hdr_allowed(Some(r#"{"attribute": 0}"#)));
    assert!(!hdr_allowed(Some(r#"{"titleId": "PPSA00001"}"#)));
    assert!(!hdr_allowed(Some(r#"{"attribute": "536870912"}"#)));
}

#[test]
fn output_status_stays_sdr_with_hdr_off() {
    let _serial = serial();
    headless();
    // RPS5E_HDR is not set by the tests: the window is never asked.
    assert!(!ps5gpu::display::hdr_enabled());
    let vm = ps5vm::VirtualMemory::new();
    const BASE: u64 = 0x5BB0_0000_0000;
    vm.map(BASE, 0x1_0000, ps5vm::MemProtect::RW).unwrap();
    let mut ctx = CpuContext::new(&vm);
    ctx.set(Rsp, BASE + 0x800);
    let h = call("sceVideoOutOpen", &mut ctx, &[255, 0, 0, 0]);
    assert!(h > 0);
    assert_eq!(call("sceVideoOutGetOutputStatus", &mut ctx, &[h as u64, BASE + 0x100]), 0);
    assert_eq!(ctx.read_bytes(BASE + 0x100, 0x30).unwrap(), output_status_bytes(1920, 1080, 60, false));
    assert_eq!(call("sceVideoOutClose", &mut ctx, &[h as u64]), 0);
}

#[test]
fn open_register_flip_and_events() {
    let _serial = serial();
    headless();
    let vm = ps5vm::VirtualMemory::new();
    const BASE: u64 = 0x5B20_0000_0000;
    const SURF: u64 = BASE + 0x10_0000;
    vm.map(BASE, 0x40_0000, ps5vm::MemProtect::RW).unwrap();
    let mut ctx = CpuContext::new(&vm);
    ctx.set(Rsp, BASE + 0x800);

    // Bad open parameters.
    assert_eq!(call("sceVideoOutOpen", &mut ctx, &[0, 3, 0, 0]), ERROR_INVALID_VALUE);
    let h = call("sceVideoOutOpen", &mut ctx, &[255, 0, 0, 0]);
    assert!(h > 0);
    let hu = h as u64;

    // Output options / status.
    assert_eq!(call("sceVideoOutInitializeOutputOptions", &mut ctx, &[BASE + 0x200]), 0);
    assert_eq!(call("sceVideoOutIsOutputSupported", &mut ctx, &[hu, 1, BASE + 0x200, 0, 0]), 1);
    assert_eq!(call("sceVideoOutIsOutputSupported", &mut ctx, &[hu, 5, 0, 0, 0]), ERROR_UNKNOWN_OUTPUT_MODE);
    assert_eq!(call("sceVideoOutIsOutputSupported", &mut ctx, &[999, 1, 0, 0, 0]), ERROR_INVALID_HANDLE);
    assert_eq!(call("sceVideoOutGetOutputStatus", &mut ctx, &[hu, BASE + 0x300]), 0);
    assert_eq!(call("sceVideoOutSetFlipRate", &mut ctx, &[hu, 3]), ERROR_INVALID_VALUE);
    assert_eq!(call("sceVideoOutSetFlipRate", &mut ctx, &[hu, 0]), 0);

    // Event queue registrations; output mode fires immediately.
    let eq = equeue::create();
    assert_eq!(call("sceVideoOutAddFlipEvent", &mut ctx, &[0xdead, hu, 1]), ERROR_INVALID_EVENT_QUEUE);
    assert_eq!(call("sceVideoOutAddFlipEvent", &mut ctx, &[eq, hu, 0x77]), 0);
    assert_eq!(call("sceVideoOutAddOutputModeEvent", &mut ctx, &[eq, hu, 0x88]), 0);
    let evs = equeue::wait(eq, 8, Some(0)).unwrap();
    assert_eq!(evs.len(), 1);
    assert_eq!((evs[0].ident, evs[0].filter, evs[0].udata), (8, -13, 0x88));
    assert_eq!(evs[0].data >> 16, 1);

    // Register two 256x128 render-target buffers (RGBA8 sRGB).
    let (w, hgt) = (256u32, 128u32);
    let params = tiling::DetileParams::new(tiling::SWIZZLE_64KB_R_X, 4, w).unwrap();
    let size = params.surface_size(hgt) as u64;
    let attr = BASE + 0x400;
    assert!(ctx.write_u64(BASE + 0x808, 0));
    assert!(ctx.write_u64(BASE + 0x810, 0));
    assert_eq!(call("sceVideoOutSetBufferAttribute2", &mut ctx, &[attr, 0x8000_0000_2200_0000, 0, w as u64, hgt as u64, 0]), 0);
    let entries = BASE + 0x600;
    assert!(ctx.write_u64(entries, SURF));
    assert!(ctx.write_u64(entries + 0x20, SURF + size));
    // category (stale upper half) and option on the stack.
    assert!(ctx.write_u64(BASE + 0x808, 0xC0DE_0000_0000_0000));
    assert!(ctx.write_u64(BASE + 0x810, 0));
    assert_eq!(call("sceVideoOutRegisterBuffers2", &mut ctx, &[hu, 0, 0, entries, 2, attr]), 0);
    // Slots are taken now.
    assert_eq!(call("sceVideoOutRegisterBuffers2", &mut ctx, &[hu, 1, 0, entries, 2, attr]), ERROR_RESOURCE_BUSY);
    let info = display_buffer_info(h as i32, 1).unwrap();
    assert_eq!(info.address, SURF + size);
    assert_eq!(info.attribute.width, w);

    // Fill buffer 0 with a tiled pattern.
    let lin: Vec<u8> = (0..w * hgt).flat_map(|i| [(i % 256) as u8, (i / 256) as u8, 0x40, 0xFF]).collect();
    let mut tiled = vec![0u8; size as usize];
    tiling::tile(&params, &lin, &mut tiled, w, hgt);
    assert!(ctx.write_bytes(SURF, &tiled));
    let converted = ps5gpu::read_guest_surface(
        &ps5gpu::SurfaceDesc { width: w, height: hgt, pitch_pixels: w, pixel_format: 0x8000_0000_2200_0000, tiling: 0 },
        SURF,
    )
    .unwrap();
    assert_eq!(&converted[..8], &[0x40, 0, 0, 0xFF, 0x40, 0, 1, 0xFF]);

    // GPU flip: counters, events and presenter completion.
    assert_eq!(gpu_flip(h as i32, 16, 1, 0), Err(ERROR_INVALID_INDEX));
    assert_eq!(gpu_flip(h as i32, -2, 1, 0), Err(ERROR_INVALID_INDEX));
    assert_eq!(gpu_flip(999, 0, 1, 0), Err(ERROR_INVALID_HANDLE));
    let before = ps5gpu::presenter().stats().presented;
    assert_eq!(gpu_flip(h as i32, 0, FLIP_MODE_VSYNC as u32, 42), Ok(()));
    let fs = BASE + 0x900;
    assert_eq!(call("sceVideoOutGetFlipStatus", &mut ctx, &[hu, fs]), 0);
    assert_eq!(ctx.read_u64(fs), Some(1));
    assert_eq!(ctx.read_u64(fs + 0x18), Some(42));
    assert_eq!(ctx.read_u32(fs + 0x30), Some(0));
    assert_eq!(ctx.read_u32(fs + 0x34), Some(0));
    assert_eq!(ctx.read_u32(fs + 0x38), Some(0));
    let evs = equeue::wait(eq, 8, Some(0)).unwrap();
    assert_eq!(evs.len(), 1);
    assert_eq!((evs[0].ident, evs[0].filter, evs[0].udata), (6, -13, 0x77));
    // GetEventId / GetEventData on the delivered kevent.
    let ev = BASE + 0xA00;
    assert!(ctx.write_bytes(ev, &evs[0].to_bytes()));
    assert_eq!(call("sceVideoOutGetEventId", &mut ctx, &[ev]), 0);
    assert_eq!(call("sceVideoOutGetEventData", &mut ctx, &[ev, BASE + 0xA80]), 0);
    assert_eq!(ctx.read_u64(BASE + 0xA80), Some(42));
    let mut bogus = evs[0];
    bogus.filter = -1;
    assert!(ctx.write_bytes(ev, &bogus.to_bytes()));
    assert_eq!(call("sceVideoOutGetEventId", &mut ctx, &[ev]), ERROR_INVALID_EVENT);

    // The first vsync flip is shown at once; a second is paced to the next refresh.
    assert!(is_flip_done(h as i32, 0));
    assert_eq!(gpu_flip(h as i32, 1, FLIP_MODE_VSYNC as u32, 43), Ok(()));
    let t0 = Instant::now();
    while !is_flip_done(h as i32, 1) {
        assert!(t0.elapsed() < Duration::from_secs(1), "flip never presented");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(ps5gpu::presenter().stats().presented >= before + 2);

    // Blank flip and CPU flip.
    assert_eq!(gpu_flip(h as i32, -1, FLIP_MODE_ASAP as u32, 44), Ok(()));
    assert!(is_flip_done(h as i32, -1));
    assert_eq!(call("sceVideoOutSubmitFlip", &mut ctx, &[hu, 0, FLIP_MODE_VSYNC as u64, 45]), 0);
    assert_eq!(call("sceVideoOutGetFlipStatus", &mut ctx, &[hu, fs]), 0);
    assert_eq!(ctx.read_u64(fs), Some(4));
    assert_eq!(ctx.read_u64(fs + 0x18), Some(45));
    assert_eq!(call("sceVideoOutIsFlipPending", &mut ctx, &[hu]), 0);
    let t0 = Instant::now();
    while pending_flip_request_count() != 0 && t0.elapsed() < Duration::from_secs(1) {
        std::thread::sleep(Duration::from_millis(1));
    }

    // Flip event removal: only this queue's registration goes away.
    assert_eq!(call("sceVideoOutDeleteFlipEvent", &mut ctx, &[eq, 999]), ERROR_INVALID_HANDLE);
    assert_eq!(lock().ports[&(h as i32)].flip_events.len(), 1);
    assert_eq!(call("sceVideoOutDeleteFlipEvent", &mut ctx, &[0xDEAD_BEEF_0000, hu]), 0);
    assert_eq!(lock().ports[&(h as i32)].flip_events.len(), 1);
    assert_eq!(call("sceVideoOutDeleteFlipEvent", &mut ctx, &[eq, hu]), 0);
    assert!(lock().ports[&(h as i32)].flip_events.is_empty());

    // Unregister, close.
    assert_eq!(call("sceVideoOutUnregisterBuffers", &mut ctx, &[hu, 1]), ERROR_INVALID_VALUE);
    assert_eq!(call("sceVideoOutUnregisterBuffers", &mut ctx, &[hu, 0]), 0);
    assert!(display_buffer_info(h as i32, 0).is_none());
    assert_eq!(call("sceVideoOutClose", &mut ctx, &[hu]), 0);
    assert_eq!(call("sceVideoOutGetFlipStatus", &mut ctx, &[hu, fs]), ERROR_INVALID_HANDLE);
}

#[test]
fn vblank_events_and_wait() {
    let _serial = serial();
    let vm = ps5vm::VirtualMemory::new();
    const BASE: u64 = 0x5B30_0000_0000;
    vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
    let mut ctx = CpuContext::new(&vm);
    let h = call("sceVideoOutOpen", &mut ctx, &[0, 0, 0, 0]) as u64;
    let st = BASE + 0x100;
    assert_eq!(call("sceVideoOutWaitVblank", &mut ctx, &[h]), 0);
    assert_eq!(call("sceVideoOutGetVblankStatus", &mut ctx, &[h, st]), 0);
    let c1 = ctx.read_u64(st).unwrap();
    assert!(c1 >= 1);
    let t = Instant::now();
    assert_eq!(call("sceVideoOutWaitVblank", &mut ctx, &[h]), 0);
    assert!(t.elapsed() <= Duration::from_millis(40));
    assert_eq!(call("sceVideoOutGetVblankStatus", &mut ctx, &[h, st]), 0);
    assert!(ctx.read_u64(st).unwrap() > c1);

    let eq = equeue::create();
    assert_eq!(call("sceVideoOutAddVblankEvent", &mut ctx, &[eq, h, 0x99]), 0);
    let evs = equeue::wait(eq, 4, Some(200_000)).unwrap();
    assert_eq!(evs.len(), 1);
    assert_eq!((evs[0].ident, evs[0].udata), (0x40, 0x99));
    let ev = BASE + 0x200;
    assert!(ctx.write_bytes(ev, &evs[0].to_bytes()));
    assert_eq!(call("sceVideoOutGetEventId", &mut ctx, &[ev]), 1);
    assert_eq!(call("sceVideoOutDeleteVblankEvent", &mut ctx, &[eq, h]), 0);
    assert_eq!(call("sceVideoOutClose", &mut ctx, &[h]), 0);
}

/// Manual smoke test of the whole flip path on the real window: a
/// render-target-tiled 1920x1080 RGBA8 buffer pair is animated and flipped
/// through `gpu_flip` for ~2 s. Run alone:
/// `cargo test -p ps5hle --release window_flip_smoke -- --ignored`.
#[test]
#[ignore]
fn window_flip_smoke() {
    let _serial = serial();
    let vm = ps5vm::VirtualMemory::new();
    const BASE: u64 = 0x5B40_0000_0000;
    let (w, hgt) = (1920u32, 1080u32);
    let params = tiling::DetileParams::new(tiling::SWIZZLE_64KB_R_X, 4, w).unwrap();
    let size = params.surface_size(hgt) as u64;
    vm.map(BASE, 0x10000 + 2 * size, ps5vm::MemProtect::RW).unwrap();
    let mut ctx = CpuContext::new(&vm);
    ctx.set(Rsp, BASE + 0x800);
    let h = call("sceVideoOutOpen", &mut ctx, &[0, 0, 0, 0]);
    assert!(h > 0);
    let attr = BASE + 0x400;
    assert!(ctx.write_u64(BASE + 0x808, 0));
    assert!(ctx.write_u64(BASE + 0x810, 0));
    call("sceVideoOutSetBufferAttribute2", &mut ctx, &[attr, 0x8000_0000_2200_0000, 0, w as u64, hgt as u64, 0]);
    let entries = BASE + 0x600;
    let surf = [BASE + 0x10000, BASE + 0x10000 + size];
    assert!(ctx.write_u64(entries, surf[0]));
    assert!(ctx.write_u64(entries + 0x20, surf[1]));
    assert_eq!(call("sceVideoOutRegisterBuffers2", &mut ctx, &[h as u64, 0, 0, entries, 2, attr]), 0);
    let start = Instant::now();
    let mut frame = 0u32;
    let mut lin = vec![0u8; (w * hgt * 4) as usize];
    let mut tiled = vec![0u8; size as usize];
    while start.elapsed() < Duration::from_secs(2) {
        let idx = (frame % 2) as i32;
        while !is_flip_done(h as i32, idx) {
            std::thread::sleep(Duration::from_millis(1));
        }
        for y in 0..hgt {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let checker = (((x + frame * 8) / 64) ^ (y / 64)) & 1;
                lin[i] = (x * 255 / w) as u8;
                lin[i + 1] = (y * 255 / hgt) as u8;
                lin[i + 2] = if checker == 1 { 255 } else { 0 };
                lin[i + 3] = 255;
            }
        }
        tiling::tile(&params, &lin, &mut tiled, w, hgt);
        assert!(ctx.write_bytes(surf[idx as usize], &tiled));
        gpu_flip(h as i32, idx, FLIP_MODE_VSYNC as u32, frame as i64).unwrap();
        frame += 1;
    }
    let s = ps5gpu::presenter().stats();
    eprintln!("flip smoke: frames={frame} headless={} presented={} discarded={}", s.headless, s.presented, s.discarded);
    assert!(s.presented > 0);
    call("sceVideoOutClose", &mut ctx, &[h as u64]);
}

#[test]
fn flip_master_attribute_option_and_event_count() {
    let _serial = serial();
    const BASE: u64 = 0x5B50_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
    let mut ctx = CpuContext::new(&vm);

    // AddBufferAttributeOption ORs into the option word at +0x18.
    assert!(ctx.write_u64(BASE + 0x18, 0x8));
    assert_eq!(call("sceVideoOutAddBufferAttributeOption", &mut ctx, &[BASE, 0x20]), 0);
    assert_eq!(ctx.read_u64(BASE + 0x18), Some(0x28));
    // RemoveBufferAttributeOption clears only the named bits.
    assert!(ctx.write_u64(BASE + 0x10, 0x1234));
    assert_eq!(call("sceVideoOutRemoveBufferAttributeOption", &mut ctx, &[BASE, 0x21]), 0);
    assert_eq!((ctx.read_u64(BASE + 0x18), ctx.read_u64(BASE + 0x10)), (Some(0x8), Some(0x1234)));
    assert_eq!(call("sceVideoOutRemoveBufferAttributeOption", &mut ctx, &[BASE, 0x8]), 0);
    assert_eq!(ctx.read_u64(BASE + 0x18), Some(0));
    assert_eq!(call("sceVideoOutRemoveBufferAttributeOption", &mut ctx, &[0, 0x8]), 0);

    // GetEventCount: bits 12..15 of the event data.
    let ev = BASE + 0x100;
    assert_eq!(call("sceVideoOutGetEventCount", &mut ctx, &[0]), ERROR_INVALID_ADDRESS);
    assert!(ctx.write_u64(ev, EVENT_PRE_VBLANK_START));
    assert!(ctx.write_bytes(ev + 8, &(-1i16).to_le_bytes()));
    assert_eq!(call("sceVideoOutGetEventCount", &mut ctx, &[ev]), ERROR_INVALID_EVENT);
    assert!(ctx.write_bytes(ev + 8, &FILTER_VIDEO_OUT.to_le_bytes()));
    assert!(ctx.write_u64(ev + 0x10, (7 << 16) | (3 << 12) | 0x123));
    assert_eq!(call("sceVideoOutGetEventCount", &mut ctx, &[ev]), 3);
    assert_eq!(call("sceVideoOutGetEventId", &mut ctx, &[ev]), 2);

    let h = call("sceVideoOutOpen", &mut ctx, &[0, 0, 0, 0]) as u64;
    assert_eq!(call("sceVideoOutAllowOutputResolutionWqhdDetection", &mut ctx, &[h]), 0);
    assert_eq!(call("sceVideoOutAllowOutputResolutionWqhdDetection", &mut ctx, &[0x7fff]), ERROR_INVALID_HANDLE);
    assert_eq!(call("sceVideoOutSetFlipMaster", &mut ctx, &[h, 0x7fff]), ERROR_INVALID_HANDLE);
    assert_eq!(call("sceVideoOutSetFlipMaster", &mut ctx, &[h, h]), ERROR_UNSUPPORTED_OPERATION);
    // Slave/master flip modes need a joined port.
    assert_eq!(call("sceVideoOutSubmitFlip", &mut ctx, &[h, u64::MAX, 8, 0]), ERROR_INVALID_FLIP_MODE);
    assert_eq!(call("sceVideoOutSubmitFlip", &mut ctx, &[h, u64::MAX, 9, 0]), ERROR_INVALID_FLIP_MODE);

    let eq = equeue::create();
    assert_eq!(call("sceVideoOutAddPreVblankStartEvent", &mut ctx, &[eq, 0x7fff, 0]), ERROR_INVALID_HANDLE);
    assert_eq!(call("sceVideoOutAddPreVblankStartEvent", &mut ctx, &[eq, h, 0x55]), 0);
    assert_eq!(call("sceVideoOutDeletePreVblankStartEvent", &mut ctx, &[eq, h]), 0);
    assert_eq!(call("sceVideoOutClose", &mut ctx, &[h]), 0);
    assert_eq!(call("sceVideoOutDeletePreVblankStartEvent", &mut ctx, &[eq, h]), ERROR_INVALID_HANDLE);
}

#[test]
fn output_mode_queries_answer_console_codes() {
    let _serial = serial();
    const BASE: u64 = 0x5B60_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
    let mut ctx = CpuContext::new(&vm);
    let opts = BASE + 0x100;

    // InitializeOutputOptions writes 0xFF to byte 2 and zeroes the rest.
    assert!(ctx.write_bytes(opts, &[0xAA; 0x40]));
    assert_eq!(call("sceVideoOutInitializeOutputOptions", &mut ctx, &[opts]), 0);
    let raw = ctx.read_bytes(opts, 0x40).unwrap();
    for (i, &b) in raw.iter().enumerate() {
        assert_eq!(b, if i == 2 { 0xFF } else { 0 }, "options byte {i}");
    }
    assert_eq!(raw, initial_output_options());
    assert_eq!(call("sceVideoOutInitializeOutputOptions", &mut ctx, &[0]), ERROR_INVALID_ADDRESS);

    // The handle is checked first.
    for bad in [u64::MAX, 0, 0x7FFF_FFFF] {
        assert_eq!(call("sceVideoOutIsOutputSupported", &mut ctx, &[bad, 1, opts, 0, 0]), ERROR_INVALID_HANDLE);
        assert_eq!(call("sceVideoOutIsOutputSupported", &mut ctx, &[bad, 2, 0, opts, 1]), ERROR_INVALID_HANDLE);
    }

    let h = call("sceVideoOutOpen", &mut ctx, &[255, 0, 0, 0]);
    assert!(h > 0);
    let h = h as u64;
    assert_eq!(call("sceVideoOutIsOutputSupported", &mut ctx, &[h, 1, opts, 0, 0]), 1);
    assert_eq!(call("sceVideoOutIsOutputSupported", &mut ctx, &[h, 1, 0, 0, 0]), 1);
    // Modes the measured display lacks (119.88 Hz only on a 120 Hz output).
    let high_rate = guest_refresh_rate() >= 119;
    for mode in [0x4u64, 0x7, 0x8, 0xC, 0xD, 0xE, 0xF, 0x10, 0x11, 0x12, 0x13] {
        if mode == 0xF && high_rate {
            continue;
        }
        assert_eq!(call("sceVideoOutIsOutputSupported", &mut ctx, &[h, mode, opts, 0, 0]), ERROR_UNSUPPORTED_OUTPUT_MODE, "mode {mode:#x}");
        assert_eq!(call("sceVideoOutConfigureOutput", &mut ctx, &[h, mode, opts, 0, 0]), ERROR_UNSUPPORTED_OUTPUT_MODE, "mode {mode:#x}");
    }
    // Values that are not modes.
    for mode in [0x0u64, 0x2, 0x3, 0x5, 0x6, 0x9, 0xA, 0xB, 0x14, 0x20, 0x3F, 0x100, 0xD000_000A, 0x8000_0000_0000_0001, u64::MAX] {
        assert_eq!(call("sceVideoOutIsOutputSupported", &mut ctx, &[h, mode, opts, 0, 0]), ERROR_UNKNOWN_OUTPUT_MODE, "mode {mode:#x}");
    }
    assert_eq!(call("sceVideoOutConfigureOutput", &mut ctx, &[h, 0xD000_000A, 0, 0, 0]), ERROR_UNKNOWN_OUTPUT_MODE);

    // Reserved arguments come before the mode, the mode before the options.
    assert_eq!(call("sceVideoOutIsOutputSupported", &mut ctx, &[h, 1, opts, opts, 0]), ERROR_INVALID_VALUE);
    assert_eq!(call("sceVideoOutIsOutputSupported", &mut ctx, &[h, 1, opts, 0, 1]), ERROR_INVALID_VALUE);
    assert_eq!(call("sceVideoOutIsOutputSupported", &mut ctx, &[h, 2, opts, 0, 1]), ERROR_INVALID_VALUE);
    let zeroed = BASE + 0x200;
    assert!(ctx.write_bytes(zeroed, &[0; 0x40]));
    assert_eq!(call("sceVideoOutIsOutputSupported", &mut ctx, &[h, 1, zeroed, 0, 0]), ERROR_INVALID_VALUE);
    assert_eq!(call("sceVideoOutIsOutputSupported", &mut ctx, &[h, 2, zeroed, 0, 0]), ERROR_UNKNOWN_OUTPUT_MODE);
    assert_eq!(call("sceVideoOutIsOutputSupported", &mut ctx, &[h, 4, zeroed, 0, 0]), ERROR_UNSUPPORTED_OUTPUT_MODE);

    // Every word but word 3 must keep its initialized value.
    let changed = BASE + 0x300;
    for word in 0..16usize {
        let mut b = initial_output_options();
        b[word * 4] ^= 1;
        assert!(ctx.write_bytes(changed, &b));
        let expected = if word == 3 { 1 } else { ERROR_INVALID_VALUE };
        assert_eq!(call("sceVideoOutIsOutputSupported", &mut ctx, &[h, 1, changed, 0, 0]), expected, "word {word}");
    }
    let mut b = initial_output_options();
    b[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(ctx.write_bytes(changed, &b));
    assert_eq!(call("sceVideoOutIsOutputSupported", &mut ctx, &[h, 1, changed, 0, 0]), 1);
    assert_eq!(call("sceVideoOutConfigureOutput", &mut ctx, &[h, 1, opts, 0, 0]), 0);

    assert_eq!(call("sceVideoOutClose", &mut ctx, &[h]), 0);
    assert_eq!(call("sceVideoOutIsOutputSupported", &mut ctx, &[h, 1, opts, 0, 0]), ERROR_INVALID_HANDLE);
}

#[test]
fn resolution_status_layout() {
    let _serial = serial();
    const BASE: u64 = 0x5B70_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
    let mut ctx = CpuContext::new(&vm);
    let h = call("sceVideoOutOpen", &mut ctx, &[255, 0, 0, 0]);
    assert!(h > 0);
    let h = h as u64;

    let st = BASE + 0x100;
    assert!(ctx.write_bytes(st, &[0xFF; 0x40]));
    assert_eq!(call("sceVideoOutGetResolutionStatus", &mut ctx, &[h, st]), 0);
    let raw = ctx.read_bytes(st, 0x40).unwrap();
    let u32_at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
    // Full and pane size are the port's output size.
    assert_eq!((u32_at(0x0), u32_at(0x4), u32_at(0x8), u32_at(0xC)), (1920, 1080, 1920, 1080));
    // Screen size, flags and reserved words are zero; the status is 48 bytes.
    assert!(raw[0x18..0x30].iter().all(|&b| b == 0));
    assert!(raw[0x30..0x40].iter().all(|&b| b == 0xFF));
    // The refresh-rate code is the output status's.
    let out = BASE + 0x200;
    assert_eq!(call("sceVideoOutGetOutputStatus", &mut ctx, &[h, out]), 0);
    assert_eq!(ctx.read_u64(out + 8), Some(u64::from_le_bytes(raw[0x10..0x18].try_into().unwrap())));

    let s = resolution_status_bytes(3840, 2160, 120);
    assert_eq!(u32::from_le_bytes(s[8..12].try_into().unwrap()), 3840);
    assert_eq!(u64::from_le_bytes(s[0x10..0x18].try_into().unwrap()), 13);

    assert_eq!(call("sceVideoOutGetResolutionStatus", &mut ctx, &[h, 0]), ERROR_INVALID_ADDRESS);
    assert_eq!(call("sceVideoOutGetResolutionStatus", &mut ctx, &[0, st]), ERROR_INVALID_HANDLE);
    assert_eq!(call("sceVideoOutGetResolutionStatus", &mut ctx, &[u64::MAX, st]), ERROR_INVALID_HANDLE);
    assert_eq!(call("sceVideoOutClose", &mut ctx, &[h]), 0);
    assert_eq!(call("sceVideoOutGetResolutionStatus", &mut ctx, &[h, st]), ERROR_INVALID_HANDLE);
}

#[test]
fn gpu_flip_of_unregistered_buffer_presents_black_and_completes() {
    let _serial = serial();
    headless();
    const BASE: u64 = 0x5B80_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
    let mut ctx = CpuContext::new(&vm);
    let h = call("sceVideoOutOpen", &mut ctx, &[255, 0, 0, 0]);
    assert!(h > 0);
    let hu = h as u64;
    let eq = equeue::create();
    assert_eq!(call("sceVideoOutAddFlipEvent", &mut ctx, &[eq, hu, 0x31]), 0);

    // A CPU flip names its buffer at submit and still rejects it.
    assert_eq!(call("sceVideoOutSubmitFlip", &mut ctx, &[hu, 3, FLIP_MODE_VSYNC as u64, 1]), ERROR_INVALID_INDEX);
    // A command-buffer flip resolves the buffer when the GPU reaches it.
    assert_eq!(gpu_flip(h as i32, 3, FLIP_MODE_VSYNC as u32, 0x55), Ok(()));
    let fs = BASE + 0x100;
    assert_eq!(call("sceVideoOutGetFlipStatus", &mut ctx, &[hu, fs]), 0);
    assert_eq!(ctx.read_u64(fs), Some(1));
    assert_eq!(ctx.read_u64(fs + 0x18), Some(0x55));
    assert_eq!(ctx.read_u32(fs + 0x34), Some(0));
    assert_eq!(ctx.read_u32(fs + 0x38), Some(3));
    let evs = equeue::wait(eq, 8, Some(0)).unwrap();
    assert_eq!(evs.len(), 1);
    assert_eq!((evs[0].ident, evs[0].udata, evs[0].data >> 16), (EVENT_FLIP, 0x31, 0x55));
    let t0 = Instant::now();
    while !is_flip_done(h as i32, 3) {
        assert!(t0.elapsed() < Duration::from_secs(1), "black flip never presented");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(call("sceVideoOutClose", &mut ctx, &[hu]), 0);
    // Flips on a closed port are refused, not stored as failures.
    assert_eq!(gpu_flip(h as i32, 3, FLIP_MODE_VSYNC as u32, 0x56), Err(ERROR_INVALID_HANDLE));
    assert!(is_flip_done(h as i32, 3));
}

#[test]
fn overlay_bus_flips_complete_without_being_shown() {
    let _serial = serial();
    headless();
    const BASE: u64 = 0x5B90_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    vm.map(BASE, 0x40_0000, ps5vm::MemProtect::RW).unwrap();
    let mut ctx = CpuContext::new(&vm);
    ctx.set(Rsp, BASE + 0x800);

    let main = call("sceVideoOutOpen", &mut ctx, &[255, BUS_TYPE_MAIN as u64, 0, 0]);
    let overlay = call("sceVideoOutOpen", &mut ctx, &[255, BUS_TYPE_OVERLAY as u64, 0, 0]);
    let sub = call("sceVideoOutOpen", &mut ctx, &[255, BUS_TYPE_SUB as u64, 0, 0]);
    assert!(main > 0 && overlay > 0 && sub > 0);
    assert_ne!(main, overlay);
    assert_eq!(call("sceVideoOutClose", &mut ctx, &[sub as u64]), 0);
    let ov = overlay as u64;

    let attr = BASE + 0x400;
    assert!(ctx.write_u64(BASE + 0x808, 0));
    assert!(ctx.write_u64(BASE + 0x810, 0));
    assert_eq!(call("sceVideoOutSetBufferAttribute2", &mut ctx, &[attr, 0x8000_0000_2200_0000, 0, 64, 64, 0]), 0);
    let entries = BASE + 0x600;
    assert!(ctx.write_u64(entries, BASE + 0x10_0000));
    assert_eq!(call("sceVideoOutRegisterBuffers2", &mut ctx, &[ov, 0, 0, entries, 1, attr]), 0);
    let eq = equeue::create();
    assert_eq!(call("sceVideoOutAddFlipEvent", &mut ctx, &[eq, ov, 0x41]), 0);

    // Completed and signalled like a main-bus flip, but never scheduled
    // for presentation.
    assert_eq!(gpu_flip(overlay as i32, 0, FLIP_MODE_VSYNC as u32, 7), Ok(()));
    {
        let g = lock();
        let p = &g.ports[&(overlay as i32)];
        assert_eq!((p.flip_count, p.flip_arg, p.current_buffer, p.flip_pending_count), (1, 7, 0, 0));
        assert_eq!(p.last_scheduled_ts, -1);
        assert_eq!(p.last_presentation_ts, -1);
    }
    let evs = equeue::wait(eq, 8, Some(0)).unwrap();
    assert_eq!(evs.len(), 1);
    assert_eq!((evs[0].ident, evs[0].udata, evs[0].data >> 16), (EVENT_FLIP, 0x41, 7));
    assert!(is_flip_done(overlay as i32, 0));

    assert_eq!(call("sceVideoOutSubmitFlip", &mut ctx, &[ov, 0, FLIP_MODE_VSYNC as u64, 8]), 0);
    {
        let g = lock();
        let p = &g.ports[&(overlay as i32)];
        assert_eq!((p.flip_count, p.flip_arg), (2, 8));
        assert_eq!(p.last_presentation_ts, -1);
    }
    assert!(is_flip_done(overlay as i32, 0));
    assert_eq!(call("sceVideoOutClose", &mut ctx, &[ov]), 0);
    assert_eq!(call("sceVideoOutClose", &mut ctx, &[main as u64]), 0);
}

#[test]
fn pre_vblank_and_vrr_events_and_close_removes_events() {
    let _serial = serial();
    const BASE: u64 = 0x5BA0_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
    let mut ctx = CpuContext::new(&vm);
    let h = call("sceVideoOutOpen", &mut ctx, &[255, 0, 0, 0]);
    assert!(h > 0);
    let h = h as u64;

    // Each vblank tick delivers the pre-vblank start event and the vblank
    // event, both carrying the vblank count.
    let eq = equeue::create();
    assert_eq!(call("sceVideoOutAddPreVblankStartEvent", &mut ctx, &[eq, h, 0x21]), 0);
    assert_eq!(call("sceVideoOutAddVblankEvent", &mut ctx, &[eq, h, 0x22]), 0);
    let (mut pre, mut vb) = (None, None);
    let t0 = Instant::now();
    while (pre.is_none() || vb.is_none()) && t0.elapsed() < Duration::from_secs(1) {
        for e in equeue::wait(eq, 8, Some(50_000)).unwrap() {
            match e.ident {
                EVENT_PRE_VBLANK_START => pre = Some(e),
                EVENT_VBLANK => vb = Some(e),
                _ => {}
            }
        }
    }
    let (pre, vb) = (pre.expect("pre-vblank event"), vb.expect("vblank event"));
    assert_eq!((pre.filter, pre.udata, vb.udata), (FILTER_VIDEO_OUT, 0x21, 0x22));
    assert!(pre.data >> 16 >= 1);
    let ev = BASE + 0x100;
    assert!(ctx.write_bytes(ev, &pre.to_bytes()));
    assert_eq!(call("sceVideoOutGetEventId", &mut ctx, &[ev]), 2);
    assert_eq!(call("sceVideoOutDeletePreVblankStartEvent", &mut ctx, &[eq, h]), 0);
    assert_eq!(call("sceVideoOutDeleteVblankEvent", &mut ctx, &[eq, h]), 0);

    // VRR status (event id 16): registered, replaced per queue, never fired.
    assert_eq!(call("sceVideoOutAddVrrActiveStatusEvent", &mut ctx, &[eq, 0x7fff, 0]), ERROR_INVALID_HANDLE);
    assert_eq!(call("sceVideoOutAddVrrActiveStatusEvent", &mut ctx, &[0xDEAD_BEEF_0000, h, 0]), ERROR_INVALID_EVENT_QUEUE);
    assert_eq!(call("sceVideoOutAddVrrActiveStatusEvent", &mut ctx, &[eq, h, 0x23]), 0);
    assert_eq!(call("sceVideoOutAddVrrActiveStatusEvent", &mut ctx, &[eq, h, 0x24]), 0);
    assert_eq!(lock().ports[&(h as i32)].vrr_status_events, vec![EventReg { equeue: eq, udata: 0x24 }]);
    let vrr = equeue::KernelEvent { ident: EVENT_VRR_STATUS, filter: FILTER_VIDEO_OUT, ..Default::default() };
    assert!(ctx.write_bytes(ev, &vrr.to_bytes()));
    assert_eq!(call("sceVideoOutGetEventId", &mut ctx, &[ev]), 16);
    assert_eq!(call("sceVideoOutAddVrrStatusFlagsPrivilege", &mut ctx, &[]), 0);

    // VRR peg and unpeg on the fixed-rate output: the handle is checked
    // (reference Flip.cpp cases; non-zero peg arguments are only logged).
    assert_eq!(call("sceVideoOutVrrPegToFixedRate", &mut ctx, &[h, 0, 0]), 0);
    assert_eq!(call("sceVideoOutVrrUnpegFromFixedRate", &mut ctx, &[h]), 0);
    assert_eq!(call("sceVideoOutVrrPegToFixedRate", &mut ctx, &[h, 0x1_0000_0000, 1]), 0);
    assert_eq!(call("sceVideoOutVrrPegToFixedRate", &mut ctx, &[0x7fff, 0, 0]), ERROR_INVALID_HANDLE);
    assert_eq!(call("sceVideoOutVrrUnpegFromFixedRate", &mut ctx, &[0x7fff]), ERROR_INVALID_HANDLE);

    // Output mode events: deleting the registration clears it; a second
    // delete succeeds; a NULL queue or an unknown handle fails (reference
    // Flip.cpp cases).
    let eqm = equeue::create();
    assert_eq!(call("sceVideoOutAddOutputModeEvent", &mut ctx, &[eqm, h, 0x26]), 0);
    assert_eq!(lock().ports[&(h as i32)].output_mode_events.len(), 1);
    assert_eq!(call("sceVideoOutDeleteOutputModeEvent", &mut ctx, &[eqm, h]), 0);
    assert!(lock().ports[&(h as i32)].output_mode_events.is_empty());
    assert_eq!(equeue::pending_count(eqm), 0, "the pending event is taken off the queue");
    assert_eq!(call("sceVideoOutDeleteOutputModeEvent", &mut ctx, &[eqm, h]), 0);
    assert_eq!(call("sceVideoOutDeleteOutputModeEvent", &mut ctx, &[0, h]), ERROR_INVALID_EVENT_QUEUE);
    assert_eq!(call("sceVideoOutDeleteOutputModeEvent", &mut ctx, &[eqm, 0x7fff]), ERROR_INVALID_HANDLE);

    // Closing the port takes its pending events off the queue.
    let eq2 = equeue::create();
    assert_eq!(call("sceVideoOutAddOutputModeEvent", &mut ctx, &[eq2, h, 0x25]), 0);
    assert_eq!(equeue::pending_count(eq2), 1);
    assert_eq!(call("sceVideoOutClose", &mut ctx, &[h]), 0);
    assert_eq!(equeue::pending_count(eq2), 0);
}

/// Open a main-bus port with `count` registered 64x64 buffers.
fn port_with_buffers(ctx: &mut CpuContext, base: u64, count: u64) -> u64 {
    let h = call("sceVideoOutOpen", ctx, &[255, BUS_TYPE_MAIN as u64, 0, 0]);
    assert!(h > 0);
    let attr = base + 0x400;
    assert!(ctx.write_u64(base + 0x808, 0));
    assert!(ctx.write_u64(base + 0x810, 0));
    assert_eq!(call("sceVideoOutSetBufferAttribute2", ctx, &[attr, 0x8000_0000_2200_0000, 0, 64, 64, 0]), 0);
    let entries = base + 0x600;
    for i in 0..count {
        assert!(ctx.write_u64(entries + i * 0x20, base + 0x10_0000 + i * 0x1_0000));
    }
    assert_eq!(call("sceVideoOutRegisterBuffers2", ctx, &[h as u64, 0, 0, entries, count, attr]), 0);
    h as u64
}

fn wait_flip_done(h: u64, index: i32) {
    let t0 = Instant::now();
    while !is_flip_done(h as i32, index) {
        assert!(t0.elapsed() < Duration::from_secs(2), "flip of buffer {index} never presented");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// sceVideoOutSubmitEopFlip reserves a flip (pending at once) and hands out
/// a single-use token in 0x04000000..0x07FFFFFF; claiming it performs the
/// flip; at most 16 flips wait; closing cancels the reservations
/// (reference VideoOutDriver.cpp SubmitEopFlip, Driver ReserveEopFlip).
#[test]
fn eop_flip_reservations() {
    let _serial = serial();
    headless();
    const BASE: u64 = 0x5BE0_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    vm.map(BASE, 0x40_0000, ps5vm::MemProtect::RW).unwrap();
    let mut ctx = CpuContext::new(&vm);
    ctx.set(Rsp, BASE + 0x800);
    let h = port_with_buffers(&mut ctx, BASE, 2);
    let token = BASE + 0x900;
    let vsync = FLIP_MODE_VSYNC as u64;
    assert!(ctx.write_u64(token, 0xFFFF_FFFF_FFFF_FFFF));
    for (args, err) in [
        ([h, 0, vsync, 1, 0], ERROR_INVALID_ADDRESS),
        ([0x7fff, 0, vsync, 1, token], ERROR_INVALID_HANDLE),
        ([h, 0, 0, 1, token], ERROR_INVALID_VALUE),
        ([h, 0, 8, 1, token], ERROR_INVALID_VALUE),
        ([h, 5, vsync, 1, token], ERROR_INVALID_INDEX),
        ([h, 16, vsync, 1, token], ERROR_INVALID_INDEX),
        ([h, -2i64 as u64, vsync, 1, token], ERROR_INVALID_INDEX),
    ] {
        assert_eq!(call("sceVideoOutSubmitEopFlip", &mut ctx, &args), err, "{args:x?}");
    }
    assert_eq!(ctx.read_u64(token), Some(u64::MAX), "a refused reservation wrote a token");
    assert_eq!(call("sceVideoOutIsFlipPending", &mut ctx, &[h]), 0);

    assert_eq!(call("sceVideoOutSubmitEopFlip", &mut ctx, &[h, 0, vsync, 0x66, token]), 0);
    let t = ctx.read_u32(token).unwrap();
    assert_eq!(ctx.read_u32(token + 4), Some(0xFFFF_FFFF), "the token is one dword");
    assert!((0x0400_0000..=0x07FF_FFFF).contains(&t));
    assert_eq!(call("sceVideoOutIsFlipPending", &mut ctx, &[h]), 1);
    assert!(!is_flip_done(h as i32, 0), "a reserved flip keeps its buffer busy");
    assert_eq!(claim_eop_flip(t + 1), None);
    let id = claim_eop_flip(t).expect("reserved token");
    assert_eq!(claim_eop_flip(t), None, "tokens are single use");
    assert_eq!(eop_flip(id), Ok(()));
    let fs = BASE + 0xA00;
    assert_eq!(call("sceVideoOutGetFlipStatus", &mut ctx, &[h, fs]), 0);
    assert_eq!((ctx.read_u64(fs), ctx.read_u64(fs + 0x18), ctx.read_u32(fs + 0x38)), (Some(1), Some(0x66), Some(0)));
    wait_flip_done(h, 0);

    // Sixteen flips may wait; the seventeenth reservation is refused.
    let mut tokens = Vec::new();
    for i in 0..16u64 {
        assert_eq!(call("sceVideoOutSubmitEopFlip", &mut ctx, &[h, 1, vsync, i, token]), 0);
        tokens.push(ctx.read_u32(token).unwrap());
    }
    assert!(tokens.windows(2).all(|w| w[1] > w[0] && w[0] > t), "tokens never repeat");
    assert_eq!(call("sceVideoOutSubmitEopFlip", &mut ctx, &[h, 1, vsync, 99, token]), ERROR_FLIP_QUEUE_FULL);
    // Closing cancels the reservations: their tokens claim nothing.
    assert_eq!(call("sceVideoOutClose", &mut ctx, &[h]), 0);
    assert!(tokens.iter().all(|&t| claim_eop_flip(t).is_none()));
    assert_eq!(pending_flip_request_count(), 0);
}

/// sceVideoOutGetBufferLabelAddress: 16 zeroed u64 labels per port;
/// registration zeroes the registered buffers' labels, and a buffer's
/// label is released once the flip after it has been presented (reference
/// tests/Flip.cpp testLabels).
#[test]
fn buffer_labels() {
    let _serial = serial();
    headless();
    const BASE: u64 = 0x5BC0_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    vm.map(BASE, 0x40_0000, ps5vm::MemProtect::RW).unwrap();
    let mut ctx = CpuContext::new(&vm);
    ctx.set(Rsp, BASE + 0x800);
    let out = BASE + 0x900;
    assert!(ctx.write_u64(out, 0x1234));
    for invalid in [-1i64 as u64, 0, 0x7fff] {
        assert_eq!(call("sceVideoOutGetBufferLabelAddress", &mut ctx, &[invalid, out]), ERROR_INVALID_HANDLE);
    }
    assert_eq!(ctx.read_u64(out), Some(0x1234));
    let h = call("sceVideoOutOpen", &mut ctx, &[255, BUS_TYPE_MAIN as u64, 0, 0]) as u64;
    assert_eq!(call("sceVideoOutGetBufferLabelAddress", &mut ctx, &[h, 0]), ERROR_INVALID_ADDRESS);
    assert_eq!(call("sceVideoOutGetBufferLabelAddress", &mut ctx, &[h, out]), 16);
    let labels = ctx.read_u64(out).unwrap();
    assert!(labels != 0 && labels % 8 == 0);
    let label = |i: u64| crate::agc::mem::r64(labels + i * 8).unwrap();
    assert!((0..16).all(|i| label(i) == 0));
    assert_eq!(call("sceVideoOutGetBufferLabelAddress", &mut ctx, &[h, out + 0x10]), 16);
    assert_eq!(ctx.read_u64(out + 0x10), Some(labels), "the array stays where it is");
    for i in [0, 1, 15] {
        assert!(crate::agc::mem::w64(labels + i * 8, 55));
    }
    // Registering buffers 0 and 1 resets their labels only.
    let attr = BASE + 0x400;
    assert!(ctx.write_u64(BASE + 0x808, 0) && ctx.write_u64(BASE + 0x810, 0));
    assert_eq!(call("sceVideoOutSetBufferAttribute2", &mut ctx, &[attr, 0x8000_0000_2200_0000, 0, 64, 64, 0]), 0);
    let entries = BASE + 0x600;
    assert!(ctx.write_u64(entries, BASE + 0x10_0000) && ctx.write_u64(entries + 0x20, BASE + 0x11_0000));
    assert_eq!(call("sceVideoOutRegisterBuffers2", &mut ctx, &[h, 0, 0, entries, 2, attr]), 0);
    assert_eq!((label(0), label(1), label(15)), (0, 0, 55));
    assert!(crate::agc::mem::w64(labels, 1) && crate::agc::mem::w64(labels + 8, 1));
    let vsync = FLIP_MODE_VSYNC as u64;
    assert_eq!(call("sceVideoOutSubmitFlip", &mut ctx, &[h, 0, vsync, 0]), 0);
    wait_flip_done(h, 0);
    assert_eq!((label(0), label(1)), (1, 1), "the first flip released a label");
    assert_eq!(call("sceVideoOutSubmitFlip", &mut ctx, &[h, 1, vsync, 1]), 0);
    wait_flip_done(h, 1);
    assert_eq!((label(0), label(1)), (0, 1), "the replacement released the wrong label");
    assert_eq!(call("sceVideoOutSubmitFlip", &mut ctx, &[h, -1i64 as u64, vsync, 2]), 0);
    wait_flip_done(h, -1);
    assert_eq!((label(1), label(15)), (0, 55), "a black flip releases only the buffer shown before it");
    // Ports have their own labels; a closed port has none.
    let other = call("sceVideoOutOpen", &mut ctx, &[255, BUS_TYPE_OVERLAY as u64, 0, 0]) as u64;
    assert_eq!(call("sceVideoOutGetBufferLabelAddress", &mut ctx, &[other, out]), 16);
    assert_ne!(ctx.read_u64(out), Some(labels));
    assert_eq!(call("sceVideoOutClose", &mut ctx, &[other]), 0);
    assert_eq!(call("sceVideoOutClose", &mut ctx, &[h]), 0);
    assert!(ctx.write_u64(out, 0x1234));
    assert_eq!(call("sceVideoOutGetBufferLabelAddress", &mut ctx, &[h, out]), ERROR_INVALID_HANDLE);
    assert_eq!(ctx.read_u64(out), Some(0x1234));
}

/// Output status with VRR, VRR event removal, color settings, color space
/// conversion and blend space exports (reference Output.cpp / Event.cpp).
#[test]
fn output_vrr_and_color_exports() {
    let _serial = serial();
    headless();
    const BASE: u64 = 0x5BD0_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    vm.map(BASE, 0x10000, ps5vm::MemProtect::RW).unwrap();
    let mut ctx = CpuContext::new(&vm);
    let h = call("sceVideoOutOpen", &mut ctx, &[255, 0, 0, 0]) as u64;
    let (st, st_vrr) = (BASE + 0x100, BASE + 0x200);
    assert!(ctx.write_bytes(st_vrr, &[0xCC; 0x58]));
    assert_eq!(call("sceVideoOutGetOutputStatusWithVrr", &mut ctx, &[h, 0]), ERROR_INVALID_ADDRESS);
    assert_eq!(call("sceVideoOutGetOutputStatusWithVrr", &mut ctx, &[0x7fff, st_vrr]), ERROR_INVALID_HANDLE);
    assert_eq!(call("sceVideoOutGetOutputStatusWithVrr", &mut ctx, &[h, st_vrr]), 0);
    assert_eq!(call("sceVideoOutGetOutputStatus", &mut ctx, &[h, st]), 0);
    let vrr = ctx.read_bytes(st_vrr, 0x58).unwrap();
    assert_eq!(vrr[..0x30], ctx.read_bytes(st, 0x30).unwrap()[..]);
    assert!(vrr[0x30..0x50].iter().all(|&b| b == 0));
    assert_eq!(vrr[0x50..], [0xCC; 8], "wrote past 0x50 bytes");

    let eq = equeue::create();
    assert_eq!(call("sceVideoOutAddVrrActiveStatusEvent", &mut ctx, &[eq, h, 0x23]), 0);
    assert_eq!(call("sceVideoOutDeleteVrrActiveStatusEvent", &mut ctx, &[eq, 0x7fff]), ERROR_INVALID_HANDLE);
    assert_eq!(call("sceVideoOutDeleteVrrActiveStatusEvent", &mut ctx, &[eq, h]), 0);
    assert!(lock().ports[&(h as i32)].vrr_status_events.is_empty());

    // Gamma: 0.1..=2.0; the sized form needs the 16-byte settings.
    let cs = BASE + 0x300;
    let gamma = |ctx: &mut CpuContext, name: &str, args: &[u64], g: f32| ps5cpu::exec::with_import_xmm(&[g.to_bits() as u64], || call(name, ctx, args)).0;
    assert_eq!(gamma(&mut ctx, "sceVideoOutColorSettingsSetGamma", &[cs], 0.5), 0);
    assert_eq!(ctx.read_f32(cs), Some(0.5));
    assert_eq!(gamma(&mut ctx, "sceVideoOutColorSettingsSetGamma", &[cs], 3.0), ERROR_INVALID_VALUE);
    assert_eq!(gamma(&mut ctx, "sceVideoOutColorSettingsSetGamma", &[0], 1.0), ERROR_INVALID_ADDRESS);
    assert_eq!(gamma(&mut ctx, "sceVideoOutColorSettingsSetGamma_", &[cs, 15], 1.5), ERROR_INVALID_VALUE);
    assert_eq!(gamma(&mut ctx, "sceVideoOutColorSettingsSetGamma_", &[cs, 16], 1.5), 0);
    assert_eq!(ctx.read_f32(cs), Some(1.5));
    assert_eq!(call("sceVideoOutAdjustColor", &mut ctx, &[h, cs]), 0);
    assert_eq!(lock().ports[&(h as i32)].gamma, 1.5);
    assert_eq!(call("sceVideoOutAdjustColor", &mut ctx, &[0x7fff, cs]), ERROR_INVALID_HANDLE);
    assert_eq!(call("sceVideoOutAdjustColor_", &mut ctx, &[h, cs, 8]), ERROR_INVALID_VALUE);
    assert_eq!(call("sceVideoOutAdjustColor_", &mut ctx, &[h, cs, 15]), ERROR_INVALID_VALUE);
    assert_eq!(call("sceVideoOutAdjustColor_", &mut ctx, &[h, cs, 16]), 0);
    // Color space conversion: 32-byte settings, kept by the port.
    let csc = BASE + 0x400;
    assert!(ctx.write_bytes(csc, &[0xAB; 32]));
    assert_eq!(call("sceVideoOutColorSpaceConversionSettingsSetDefaultSdrToHdrConversion_", &mut ctx, &[csc, 31]), ERROR_INVALID_VALUE);
    assert_eq!(ctx.read_bytes(csc, 32).unwrap(), vec![0xAB; 32]);
    assert_eq!(call("sceVideoOutColorSpaceConversionSettingsSetDefaultSdrToHdrConversion", &mut ctx, &[0]), ERROR_INVALID_ADDRESS);
    assert_eq!(call("sceVideoOutColorSpaceConversionSettingsSetDefaultSdrToHdrConversion_", &mut ctx, &[csc, 32]), 0);
    assert_eq!(ctx.read_bytes(csc, 32).unwrap(), vec![0; 32]);
    assert_eq!(call("sceVideoOutColorSpaceConversionSettingsSetSdrToHdrConversion", &mut ctx, &[0, 0]), ERROR_INVALID_ADDRESS);
    assert_eq!(call("sceVideoOutColorSpaceConversionSettingsSetSdrToHdrConversion_", &mut ctx, &[csc, 0, 16]), ERROR_INVALID_VALUE);
    assert_eq!(call("sceVideoOutColorSpaceConversionSettingsSetSdrToHdrConversion_", &mut ctx, &[csc, 0, 32]), 0);
    assert!(ctx.write_bytes(csc, &[7; 32]));
    assert_eq!(call("sceVideoOutAdjustColorSpaceConversion", &mut ctx, &[h, 0]), ERROR_INVALID_ADDRESS);
    assert_eq!(call("sceVideoOutAdjustColorSpaceConversion", &mut ctx, &[0x7fff, csc]), ERROR_INVALID_HANDLE);
    assert_eq!(call("sceVideoOutAdjustColorSpaceConversion_", &mut ctx, &[h, csc, 4]), ERROR_INVALID_VALUE);
    assert_eq!(call("sceVideoOutAdjustColorSpaceConversion_", &mut ctx, &[h, csc, 32]), 0);
    assert_eq!(lock().ports[&(h as i32)].color_space_conversion, [7; 32]);
    assert_eq!(call("sceVideoOutSetGlobalBlendSpace", &mut ctx, &[h, 2, 1]), 0);
    assert_eq!(lock().ports[&(h as i32)].blend_space, (2, 1));
    assert_eq!(call("sceVideoOutSetGlobalBlendSpace", &mut ctx, &[0x7fff, 2, 1]), ERROR_INVALID_HANDLE);
    assert_eq!(call("sceVideoOutClose", &mut ctx, &[h]), 0);
}
