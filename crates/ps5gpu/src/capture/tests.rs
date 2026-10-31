// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Frame capture: the file format round trip, the recorder over a test
//! memory manager, and the replay loop over a recording sink (no device).

use super::format::{self, Capture, Event, Record, Totals};
use super::replay::{self, Sink, Space};
use super::*;
use crate::render::formats::{depth_format, render_target_format};
use crate::render::*;
use crate::scheduler::gpu_timer::GpuTime;
use crate::testutil::{guest_mapping, test_memory};

/// One capture at a time records (the recorder is process-wide).
static SERIAL: Mutex<()> = Mutex::new(());

fn stage(seed: u32) -> ShaderStage {
    ShaderStage { address: 0x10_0000_0000 + seed as u64 * 0x100, rsrc1: seed * 3, rsrc2: seed * 5 + 1, user_data: std::array::from_fn(|i| seed ^ (i as u32 * 0x0101_0101)) }
}

fn color(slot: u32) -> ColorTarget {
    ColorTarget {
        slot,
        base: 0x20_0000_0000 + slot as u64 * 0x10_0000,
        format: render_target_format(10, 0, 0).unwrap(),
        layout: 10,
        number_type: 0,
        order: 0,
        width: 1920,
        height: 1080,
        pitch: 1920,
        swizzle: crate::tiling::SWIZZLE_64KB_R_X,
        samples: 1,
        mip_level: 1,
        mip_levels: 3,
        base_layer: 2,
        layers: 4,
        slice_size: 0x84_0000,
        write_mask: 0xB,
        fast_clear: true,
        dcc_enabled: slot == 1,
        dcc_address: 0x30_0000_0000,
        cmask_address: 0x31_0000_0000,
        fmask_address: 0x32_0000_0000,
        clear_word0: 0xDEAD_BEEF,
        clear_word1: 0x0BAD_F00D,
        dcc_control: 0x1234,
        blend: BlendState::decode(0x6001_0203 | slot),
        info: 0x1000_0028,
        attrib: 0x2_0000,
        attrib2: 0x1234_5678,
        attrib3: 0x8765_4321,
        view: 0x00FF_0001,
    }
}

fn sample_draw() -> ResolvedDraw {
    let mut viewports = [Viewport::default(); 16];
    for (i, v) in viewports.iter_mut().enumerate() {
        let f = i as f32;
        *v = Viewport { x: f, y: f * 2.0, width: 1920.0 - f, height: 1080.0, min_depth: 0.0, max_depth: 1.0, scale: [f, -f, 0.5], offset: [960.0, 540.0, 0.5], z_min: -0.25, z_max: 1.5 };
    }
    let scissors: [Rect; 16] = std::array::from_fn(|i| Rect { x0: -(i as i32), y0: i as i32, x1: 1920 - i as i32, y1: 1080 });
    let (host_format, bpe) = depth_format(3, true).unwrap();
    ResolvedDraw {
        packet: DrawPacket {
            kind: DrawKind::IndexIndirectMulti,
            queue: 3,
            packet_address: 0x40_0000_1234,
            count: 77,
            index_address: 0x41_0000_0000,
            index_base: 0x42_0000_0000,
            index_buffer_size: 999,
            index_type: 1,
            instance_count: 4,
            indirect_address: 0x43_0000_0010,
            indirect_count: 12,
            count_address: 0x44_0000_0020,
            stride: 20,
            draw_initiator: 0x8000_0002,
            indirect_registers: IndirectRegisters { vertex_offset: 0x24C, instance_offset: 0x24D, first_index: SH_NOP, draw_index: 0x98 },
        },
        index: Some(IndexBuffer { address: 0x41_0000_0000, kind: IndexType::U8, count: 999 }),
        primitive_type: primitive::TRI_STRIP,
        topology: topology(primitive::TRI_STRIP),
        primitive_restart: true,
        restart_index: 0xFFFF,
        index_offset: 17,
        color_targets: smallvec::smallvec![color(0), color(1)],
        resolve_targets: Some((color(0), color(1))),
        depth: Some(DepthTarget {
            z_format: 3,
            host_format,
            bytes_per_element: bpe,
            has_stencil: true,
            width: 1920,
            height: 1080,
            pitch: 1920,
            swizzle: crate::tiling::SWIZZLE_64KB_Z_X,
            samples: 1,
            base_layer: 0,
            layers: 1,
            z_read_base: 0x50_0000_0000,
            z_write_base: 0x50_0000_0000,
            stencil_read_base: 0x51_0000_0000,
            stencil_write_base: 0x51_0001_0000,
            htile_base: 0x52_0000_0000,
            htile_enabled: true,
            depth_slice_size: 0x80_0000,
            stencil_slice_size: 0x20_0000,
            z_info: 0x2000_0013,
            stencil_info: 1,
            depth_view: 0x55,
        }),
        depth_state: DepthStencilState {
            depth_test: true,
            depth_write: true,
            depth_compare: 3,
            depth_bounds_enable: true,
            depth_bounds: [0.125, 0.875],
            stencil_test: true,
            stencil_compare: 7,
            stencil_compare_back: 2,
            backface: true,
            stencil_control: 0x1111,
            stencil_ref_mask: 0x2222,
            stencil_ref_mask_back: 0x3333,
            depth_clear_enable: false,
            stencil_clear_enable: true,
            depth_clear_value: 1.0,
            stencil_clear_value: 0x80,
            render_control: 4,
            render_override: 5,
            depth_control: 0x776,
            depth_write_disable: false,
            stencil_write_disable: true,
        },
        color_mode: color_mode::NORMAL,
        rop3: 0xCC,
        target_mask: 0xFF,
        shader_mask: 0x0F,
        blend_constants: [0.1, 0.2, 0.3, 0.4],
        viewports,
        scissor: scissors[0],
        scissors,
        raster: RasterState {
            cull_front: false,
            cull_back: true,
            front_face_cw: true,
            polygon_mode: 2,
            provoking_vertex_last: true,
            mode_cntl: 0x1801,
            clip_cntl: 1 << 19,
            vte_cntl: 0x43F,
            vs_out_cntl: 0x10,
            dx_clip_space: true,
            poly_offset_db_fmt: 0x1E8,
            poly_offset: [1, 2, 3, 4, 5],
            user_clip_planes: std::array::from_fn(|i| 0x3F80_0000 + i as u32),
        },
        shaders: GraphicsShaders {
            ps: stage(1),
            gs: stage(2),
            es: stage(3),
            vs: stage(4),
            hs: stage(5),
            ls: stage(6),
            stages: 0x2_0000,
            geometry: GeometryRegs { ge_cntl: 1, max_output_per_subgroup: 2, max_vert_out: 3, esgs_ring_item_size: 4, out_prim_type: 5, user_data_address: 0x60_0000_0000 },
        },
        ps_input_cntl: std::array::from_fn(|i| i as u32 * 7),
        ps_input_ena: 0x22,
        ps_input_addr: 0x33,
        ps_in_control: 4,
        shader_z_format: 1,
        shader_col_format: 0x4444,
        shader_pos_format: 4,
        aa_config: 0x10,
        db_shader_control: 0x741,
        alpha_to_mask: 1,
        front_face_all_bits: false,
        msaa_enable: true,
        ps_iter_sample: false,
        aa_mask: [u32::MAX; 2],
    }
}

fn sample_dispatch() -> ResolvedDispatch {
    ResolvedDispatch {
        packet: DispatchPacket { queue: 9, packet_address: 0x70_0000_0040, groups: [12, 34, 1], indirect_address: 0x71_0000_0000, dispatch_initiator: 0x21, unsynced: false },
        shader: stage(7),
        rsrc3: 0x55,
        threads: [8, 8, 1],
        start: [1, 2, 3],
    }
}

fn all_events() -> Vec<Event> {
    vec![
        Event::Map { addr: 0x1000, size: 0x10000, backing: Some(0x2000), prot: 3 },
        Event::Map { addr: 0x90000, size: 0x1000, backing: None, prot: 1 },
        Event::Unmap { addr: 0x1000, size: 0x1000 },
        Event::CpuAccess { addr: 0x5000, write: true },
        Event::CpuAccess { addr: 0x6000, write: false },
        Event::SyncPoint,
        Event::Draw(Box::new(sample_draw())),
        Event::Dispatch(Box::new(sample_dispatch())),
        Event::EndOfPipe,
        Event::AfterPipe,
        Event::MemoryFilled { addr: 0x8000, size: 0x400, value: 0xFFFF_0000 },
        Event::CopyMemory { dst: 1, src: 2, size: 3, gpu: true },
        Event::FillMemory { dst: 4, size: 5, value: 6, gpu: false },
        Event::WriteMemory { dst: 7, data: vec![1, 2, 3, 4, 5], gpu: true },
        Event::Gds(GdsTransfer::Write { offset: 8, data: vec![9; 12] }),
        Event::Gds(GdsTransfer::Fill { offset: 4, size: 64, value: 0x77 }),
        Event::Gds(GdsTransfer::FromMemory { offset: 0, src: 0x1234, size: 16 }),
        Event::Gds(GdsTransfer::ToMemory { offset: 16, dst: 0x5678, size: 32 }),
        Event::Flush,
        Event::SubmissionComplete,
        Event::Flip { desc: SurfaceDesc { width: 3840, height: 2160, pitch_pixels: 3840, pixel_format: crate::format::A8B8G8R8_SRGB, tiling: 0 }, addr: 0x80_0000_0000, gpu: true },
    ]
}

#[test]
fn every_record_round_trips() {
    let mut bytes = format::header_bytes(&[("format".into(), "rps5e-frame-capture".into()), ("frames".into(), "2".into())]);
    let events = all_events();
    for e in &events {
        format::event_record(&mut bytes, e);
    }
    let page: Vec<u8> = (0..4096u32).map(|i| (i * 7) as u8).collect();
    format::memory_record(&mut bytes, 0x1234_5000, &page);
    format::memory_record(&mut bytes, 0x1234_6000, &[0u8; 100]);
    // A record of a tag from a newer writer is skipped.
    format::record(&mut bytes, 200, &[1, 2, 3]);
    let totals = Totals { flips: 1, draws: 2, dispatches: 3, items: 4, patches: 5, patch_bytes: 6, zero_bytes: 7, cpu_accesses: 8, maps: 9, dropped_accesses: 10 };
    format::event_record(&mut bytes, &Event::End(totals));
    let c = Capture::from_bytes(bytes).unwrap();
    assert_eq!(c.header_value("frames"), Some("2"));
    assert_eq!(c.totals, Some(totals));
    assert_eq!(c.unknown, 1);
    assert_eq!(c.records.len(), events.len() + 2);
    for (r, e) in c.records.iter().zip(&events) {
        assert_eq!(r, &Record::Event(e.clone()));
    }
    match &c.records[events.len()] {
        Record::Patch { addr, data } => {
            assert_eq!(*addr, 0x1234_5000);
            assert_eq!(c.data(data), &page[..]);
        }
        r => panic!("{r:?}"),
    }
    assert_eq!(c.records[events.len() + 1], Record::Zero { addr: 0x1234_6000, len: 100 });
}

#[test]
fn draw_round_trip_keeps_every_field() {
    // Each field of the sample differs from its default: a field the
    // encoder forgot would come back as the default.
    let d = sample_draw();
    let mut e = format::Enc::default();
    format::put_draw(&mut e, &d);
    let back = format::get_draw(&mut format::Dec::new(&e.0)).unwrap();
    assert_eq!(back, d);
    assert_eq!(back.color_targets[1].format, d.color_targets[1].format);
    let x = sample_dispatch();
    let mut e = format::Enc::default();
    format::put_dispatch(&mut e, &x);
    assert_eq!(format::get_dispatch(&mut format::Dec::new(&e.0)).unwrap(), x);
}

#[test]
fn version_one_draws_have_no_draw_index_register() {
    // Version 1 wrote three indirect registers (the packet's 77 bytes of
    // fields come first): the draw reads back without a draw-index
    // destination, the rest unchanged.
    let d = sample_draw();
    let mut e = format::Enc::default();
    format::put_draw(&mut e, &d);
    let mut v1 = e.0.clone();
    v1.drain(77 + 12..77 + 16);
    let back = format::get_draw(&mut format::Dec::with_version(&v1, 1)).unwrap();
    let mut expect = d.clone();
    expect.packet.indirect_registers.draw_index = SH_NOP;
    // Nor user clip planes (version 6).
    expect.raster.user_clip_planes = [0; 24];
    assert_eq!(back, expect);
}

#[test]
fn damaged_files_are_rejected_or_cut() {
    assert!(Capture::from_bytes(b"not a capture at all".to_vec()).is_err());
    let mut bytes = format::header_bytes(&[]);
    bytes[8] = 99;
    assert!(Capture::from_bytes(bytes).err().unwrap().0.contains("version 99"));
    // A capture cut in the middle of a record keeps the complete ones and
    // has no totals.
    let mut bytes = format::header_bytes(&[]);
    format::event_record(&mut bytes, &Event::Flush);
    format::event_record(&mut bytes, &Event::Draw(Box::new(sample_draw())));
    bytes.truncate(bytes.len() - 10);
    let c = Capture::from_bytes(bytes).unwrap();
    assert_eq!(c.records, vec![Record::Event(Event::Flush)]);
    assert_eq!(c.totals, None);
    // A record whose payload does not match its tag is an error.
    let mut bytes = format::header_bytes(&[]);
    format::record(&mut bytes, format::tag::FLIP, &[1, 2, 3]);
    assert!(Capture::from_bytes(bytes).is_err());
}

#[test]
fn content_hash_tells_contents_apart() {
    let a = vec![0u8; 4096];
    let mut b = a.clone();
    assert_eq!(content_hash(&a), content_hash(&b));
    for i in [0usize, 31, 32, 4000, 4095] {
        b[i] ^= 1;
        assert_ne!(content_hash(&a), content_hash(&b), "byte {i}");
        b[i] ^= 1;
    }
    assert_ne!(content_hash(&a[..100]), content_hash(&a[..101]));
}

/// The recorder over a test memory manager: memory records at first read
/// and on changed contents only, GPU writes updating without a record,
/// items after the memory they read, nested entry points left out, CPU
/// accesses from the fault ring, mappings.
#[test]
fn recorder_records_reads_items_and_accesses() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mem = test_memory();
    let (view, _) = guest_mapping(mem, 1 << 20);
    let dir = std::env::temp_dir().join(format!("rps5e-capture-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("recorder.ps5cap");
    // SAFETY: an unwatched test view.
    unsafe { std::ptr::write_bytes((view + 0x1000) as *mut u8, 0x11, 0x2000) };
    start(mem, &path, 0).unwrap();
    assert!(start(mem, &path, 0).is_err(), "one capture at a time");
    let mut buf = [0u8; 16];
    // First read: a record of the whole page.
    let r = around(mem, || mem.read_backing(view + 0x1010, &mut buf), |_| Event::Draw(Box::new(sample_draw())));
    assert!(r);
    // Same contents: no record; the read spans two pages.
    let mut wide = vec![0u8; 0x1000];
    around(mem, || mem.read_backing(view + 0x1800, &mut wide), |_| Event::Flush);
    // The CPU changes the first page: recorded again at the next read.
    // SAFETY: as above.
    unsafe { *((view + 0x1004) as *mut u8) = 0x22 };
    note_fault(mem, view + 0x1004, true);
    around(mem, || mem.read_backing(view + 0x1000, &mut buf), |_| Event::Dispatch(Box::new(sample_dispatch())));
    // GPU results written by the back-end: no record, and reading them
    // back records nothing either.
    assert!(mem.write_backing(view + 0x2000, &[0x33; 64]));
    around(mem, || mem.read_backing(view + 0x2000, &mut buf), |_| Event::EndOfPipe);
    // A nested entry point belongs to the outer one.
    around(mem, || around(mem, || (), |_| Event::Flush), |_| Event::AfterPipe);
    // Another memory manager's reads are not recorded.
    let other = test_memory();
    let (oview, _) = guest_mapping(other, 1 << 16);
    other.read_backing(oview, &mut buf);
    around(other, || (), |_| Event::Flush);
    mem.note_mapped(view + 0x8000, 0x1000, libc::PROT_READ | libc::PROT_WRITE, Some(0x8000));
    flip_begin(&SurfaceDesc { width: 4, height: 4, pitch_pixels: 4, pixel_format: crate::format::A8B8G8R8_SRGB, tiling: 1 }, view, false);
    let summary = finish().unwrap();
    assert!(finish().is_err());
    let c = Capture::open(&summary.path).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(c.header_value("format"), Some("rps5e-frame-capture"));
    assert_eq!(c.header_value("alias_bytes"), Some((1u64 << 20).to_string().as_str()));
    // Mappings at the start, then the stream (other tests' sync points aside).
    let maps = c.records.iter().take_while(|r| matches!(r, Record::Event(Event::Map { .. }))).count();
    assert_eq!(maps, 1);
    assert_eq!(c.records[0], Record::Event(Event::Map { addr: view, size: 1 << 20, backing: Some(0), prot: libc::PROT_READ | libc::PROT_WRITE }));
    let stream: Vec<String> = c.records[maps..]
        .iter()
        .filter(|r| !matches!(r, Record::Event(Event::SyncPoint)))
        .map(|r| match r {
            Record::Patch { addr, data } => format!("patch+{:#x}:{:#x}:{:#x}", addr - view, data.len(), c.data(data)[4]),
            Record::Zero { addr, len } => format!("zero+{:#x}:{len:#x}", addr - view),
            Record::Event(Event::Draw(_)) => "draw".into(),
            Record::Event(Event::Dispatch(_)) => "dispatch".into(),
            Record::Event(Event::CpuAccess { addr, write }) => format!("access+{:#x}:{write}", addr - view),
            Record::Event(Event::Map { addr, .. }) => format!("map+{:#x}", addr - view),
            Record::Event(Event::Flip { .. }) => "flip".into(),
            Record::Event(e) => format!("{e:?}"),
        })
        .collect();
    assert_eq!(
        stream,
        [
            "patch+0x1000:0x1000:0x11",
            "draw",
            "patch+0x2000:0x1000:0x11",
            "Flush",
            "access+0x1000:true",
            "patch+0x1000:0x1000:0x22",
            "dispatch",
            "EndOfPipe",
            "AfterPipe",
            "map+0x8000",
            "flip",
        ]
    );
    let t = c.totals.unwrap();
    assert_eq!((t.draws, t.dispatches, t.flips, t.cpu_accesses, t.patches, t.maps), (1, 1, 1, 1, 3, 2));
    assert_eq!(summary.totals, t);
}

/// What a replay hands to the back-end.
#[derive(Default)]
struct Log(Vec<String>);

impl Sink for Log {
    fn draw(&mut self, d: &ResolvedDraw) {
        self.0.push(format!("draw {:#x}", d.packet.packet_address));
    }
    fn dispatch(&mut self, d: &ResolvedDispatch) {
        self.0.push(format!("dispatch {:?}", d.packet.groups));
    }
    fn end_of_pipe(&mut self) {
        self.0.push("eop".into());
    }
    fn after_pipe(&mut self) {
        self.0.push("after".into());
    }
    fn memory_filled(&mut self, addr: u64, _: u64, _: u32) {
        self.0.push(format!("filled {addr:#x}"));
    }
    fn copy_memory(&mut self, _: u64, _: u64, _: u64) -> bool {
        self.0.push("copy".into());
        false
    }
    fn fill_memory(&mut self, _: u64, _: u64, _: u32) -> bool {
        self.0.push("fill".into());
        false
    }
    fn write_memory(&mut self, _: u64, _: &[u8]) -> bool {
        false
    }
    fn gds(&mut self, _: GdsTransfer) -> bool {
        true
    }
    fn flush(&mut self) {
        self.0.push("flush".into());
    }
    fn submission_complete(&mut self) {}
    fn sync_point(&mut self) {
        self.0.push("sync".into());
    }
    fn flip(&mut self, _: &SurfaceDesc, addr: u64) {
        self.0.push(format!("flip {addr:#x}"));
    }
    fn settle(&mut self) {
        self.0.push("settle".into());
    }
    fn gpu(&mut self) -> Option<(GpuTime, replay::ShaderTimes)> {
        None
    }
    fn writeback_display(&mut self, _: &SurfaceDesc, _: u64) -> bool {
        false
    }
}

/// The replay loop: memory records written only where memory differs,
/// transfers the back-end declines done on the CPU, items in order, every
/// loop ending with a settle.
#[test]
fn replay_restores_memory_and_runs_items_in_order() {
    let mem = test_memory();
    let (view, _) = guest_mapping(mem, 1 << 20);
    let mut bytes = format::header_bytes(&[]);
    format::event_record(&mut bytes, &Event::Map { addr: view, size: 1 << 20, backing: Some(0), prot: 3 });
    format::memory_record(&mut bytes, view, &[0x5A; 4096]);
    format::memory_record(&mut bytes, view + 0x1000, &[7; 4096]);
    let mut d = sample_draw();
    d.packet.packet_address = 0xD1;
    format::event_record(&mut bytes, &Event::Draw(Box::new(d)));
    format::event_record(&mut bytes, &Event::SyncPoint);
    format::memory_record(&mut bytes, view, &[0u8; 4096]);
    format::event_record(&mut bytes, &Event::FillMemory { dst: view + 0x3000, size: 8, value: 0x0403_0201, gpu: false });
    format::event_record(&mut bytes, &Event::CopyMemory { dst: view + 0x4000, src: view + 0x3000, size: 8, gpu: false });
    format::event_record(&mut bytes, &Event::Dispatch(Box::new(sample_dispatch())));
    format::event_record(&mut bytes, &Event::Flip { desc: SurfaceDesc { width: 4, height: 4, pitch_pixels: 4, pixel_format: 0, tiling: 1 }, addr: 0xF0, gpu: true });
    format::event_record(&mut bytes, &Event::Flip { desc: SurfaceDesc { width: 4, height: 4, pitch_pixels: 4, pixel_format: 0, tiling: 1 }, addr: 0xF1, gpu: false });
    let c = Capture::from_bytes(bytes).unwrap();
    let mut space = Space::existing(mem, &c);
    assert_eq!(space.body_start, 1);
    let mut log = Log::default();
    let report = replay::run(&c, &mut space, &mut log, &replay::Options { loops: 2, present: true, dump_dir: None, slices: 0 });
    let one = ["draw 0xd1", "sync", "fill", "copy", "dispatch [12, 34, 1]", "flip 0xf0", "settle"];
    assert_eq!(log.0, [one, one].concat());
    let mut out = [0u8; 8];
    assert!(mem.read_backing(view + 0x4000, &mut out));
    assert_eq!(out, [1, 2, 3, 4, 1, 2, 3, 4], "the CPU fill and copy ran");
    assert!(mem.read_backing(view, &mut out));
    assert_eq!(out, [0; 8], "the last memory record of the loop");
    assert_eq!((report.frames, report.draws, report.dispatches), (2, 1, 1));
    // Loop 1 writes the 0x5A page (memory held zeros), the page of 7s and
    // the zero page back. Loop 2 starts from what loop 1 left: the page of
    // 7s (recorded once) carries over, the page the CPU rewrote within
    // the capture goes through both versions again.
    assert_eq!(report.loops[0].patches_written, 3);
    assert_eq!(report.loops[1].patches_written, 2);
    assert!(mem.read_backing(view + 0x1000, &mut out));
    assert_eq!(out, [7; 8]);
    assert_eq!(report.loops[0].flips, 2);
    assert_eq!(report.loops[0].transfer_mismatches, 0);
    let text = report.to_string();
    assert!(text.contains("warm loops 2..2"), "{text}");
    // Slice mode: the same items in the same order, cut into slices at the
    // dispatch and the flips.
    let mut space = Space::existing(mem, &c);
    let mut log = Log::default();
    let report = replay::run(&c, &mut space, &mut log, &replay::Options { loops: 2, present: true, dump_dir: None, slices: 1 });
    assert_eq!(log.0, [one, one].concat());
    assert!(mem.read_backing(view + 0x4000, &mut out));
    assert_eq!(out, [1, 2, 3, 4, 1, 2, 3, 4]);
    assert!(report.loops.iter().all(|l| l.slices == 3), "{:?}", report.loops.iter().map(|l| l.slices).collect::<Vec<_>>());
    assert!(report.to_string().contains("slices: 3 per warm loop"));
}

/// A pass mixing a 4x colour with single-sample colour and depth keeps the
/// attachments of the raster count; consistent passes keep everything.
#[test]
fn mixed_sample_attachments_keep_the_raster_count() {
    use crate::backend::DrawTargets;
    let d = sample_draw();
    let z = d.depth.unwrap();
    let msaa = |slot| ColorTarget { samples: 4, ..color(slot) };
    let mut t = DrawTargets { colors: smallvec::smallvec![(msaa(0), 1), (color(1), 2)], depth: Some((z, 3)) };
    assert!(t.drop_mixed_samples(1));
    assert_eq!(t.colors.iter().map(|c| c.1).collect::<Vec<_>>(), [2]);
    assert!(t.depth.is_some());
    // Raster 4x: the single-sample colour and depth go.
    let mut t = DrawTargets { colors: smallvec::smallvec![(msaa(0), 1), (color(1), 2)], depth: Some((z, 3)) };
    assert!(t.drop_mixed_samples(4));
    assert_eq!(t.colors.iter().map(|c| c.1).collect::<Vec<_>>(), [1]);
    assert!(t.depth.is_none());
    // Colours agreeing with the depth: nothing is dropped, whatever the raster.
    let mut t = DrawTargets { colors: smallvec::smallvec![(color(0), 1), (color(1), 2)], depth: Some((z, 3)) };
    assert!(!t.drop_mixed_samples(4));
    assert_eq!(t.colors.len(), 2);
    // Without depth the first colour sets the pass count.
    let mut t = DrawTargets { colors: smallvec::smallvec![(msaa(0), 1), (msaa(1), 2)], depth: None };
    assert!(!t.drop_mixed_samples(1));
}
