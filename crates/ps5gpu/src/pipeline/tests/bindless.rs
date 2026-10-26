// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Bindless image tables: the T# is read from a heap V# at `key * 32` with
//! a run-time key. A few reachable entries are bound and the shader picks
//! its candidate at run time (`shader::image::table_element`); other tables
//! are selected through the host table map (`pipeline::bindless`, GPU_PLAN
//! §66).
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored bindless`.

use super::*;
use ps5shader::testing::{buffer_descriptor, raw_buffer_descriptor};

/// A V# of `records` records of `stride` bytes (0: raw) at `address`.
fn v_sharp(address: u64, stride: u32, records: u32) -> [u32; 4] {
    [address as u32, (address >> 32) as u32 | (stride << 16), records, if stride == 0 { 0x3101_6FAC } else { 0x0101_6FAC }]
}

/// Ported hardware test (`BindlessImages`): a compute program reads a key
/// from a material record (`readfirstlane(v0) * 16 + 4`), loads the T# at
/// `key * 32` of a 4-record table and samples it. Twelve dispatches: every
/// sixth skips everything (a user-data branch); keys 0 and 1 hold a
/// 32_UINT and a 32_FLOAT texture (one table of two classes); key 2 is a
/// null record and key 3 (0xFFFFFFFF) lies past the table: both read 0.
#[test]
#[ignore]
fn bindless_images_hw() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut g = Guest::new(16 << 20);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    const CODE: [u32; 23] = [
        0xbf06_8002, 0xbf85_0013, 0xf408_0100, 0xfa00_0000, 0xf408_0200, 0xfa00_0010, 0xf408_0300, 0xfa00_0020, 0xf408_0700, 0xfa00_0030, 0x7e20_0500, 0x9310_9010,
        0xf420_0406, 0x2000_0004, 0x8f10_8510, 0xf42c_0502, 0x2000_0000, 0xf09c_0f08, 0x0045_0000, 0xe070_0000, 0x8007_0000, 0xbf81_0000, 0xbf80_0000,
    ];
    let cs = g.program(&CODE);
    let textures = [g.alloc(&asm::bytes(&[0x4000_0000u32; 64])), g.alloc(&asm::bytes(&[0x4040_0000u32; 64]))];
    let texels = [0x4000_0000u32, 0x4040_0000];
    let threads = rig.dev.caps.subgroup_size.min(32);
    let mut outputs = Vec::new();
    for iteration in 0..12u32 {
        let key = iteration % 4;
        let mut heap = [[0u32; 8]; 4];
        if key < 2 {
            for (texture, &address) in textures.iter().enumerate() {
                let format = if texture == 0 { 20 } else { 22 };
                heap[texture] = [(address >> 8) as u32, (address >> 40) as u32 | (format << 20) | (3 << 30), 7, 0x9000_0FAC, 0, 0, 0, 0];
            }
        }
        let heap = g.alloc(&asm::bytes(heap.as_flattened()));
        let mut material = [0u32; 12];
        material[1] = if key == 3 { 0xFFFF_FFFF } else { key };
        material[5] = 0;
        material[9] = 1;
        let material = g.alloc(&asm::bytes(&material));
        let output = g.alloc(&asm::bytes(&[0xDEAD_BEEFu32; 64]));
        let mut srt = [0u32; 16];
        srt[..4].copy_from_slice(&v_sharp(heap, 32, 4));
        srt[8..12].copy_from_slice(&v_sharp(material, 16, 3));
        srt[12..].copy_from_slice(&v_sharp(output, 0, 256));
        let srt = g.alloc(&asm::bytes(&srt));
        let user = [srt as u32, (srt >> 32) as u32, if iteration % 6 == 0 { 0 } else { 1 }];
        let d = ResolvedDispatch {
            packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() },
            shader: stage(cs, &user),
            rsrc3: 0,
            threads: [threads, 1, 1],
            start: [0; 3],
        };
        assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded, "iteration {iteration}");
        outputs.push(output);
    }
    for (iteration, &output) in outputs.iter().enumerate() {
        assert!(rig.buffers.download(&mut rig.sched, output, 4));
        let mut bytes = [0u8; 4];
        assert!(g.mem.read_backing(output, &mut bytes));
        let value = u32::from_le_bytes(bytes);
        let expected = if iteration % 6 == 0 {
            0xDEAD_BEEF
        } else if iteration % 4 < 2 {
            texels[iteration % 2]
        } else {
            0
        };
        assert_eq!(value, expected, "bindless resource selection or lifetime failed at {iteration}: {value:#x}, expected {expected:#x}");
    }
}

/// s9 = s8 << 5 (s8: the work-group id); s[12:19] = heap[s9]; image_load
/// v[0:3] at (0, 0); the red channel stored at output dword s8 (heap V# in
/// s[0:3], output V# in s[4:7]).
fn group_program() -> Vec<u32> {
    group_program_dim(1)
}

/// [`group_program`] with an image_load of MIMG dimension `dim` (1: 2D,
/// 5: 2D array, at slice 0).
fn group_program_dim(dim: u32) -> Vec<u32> {
    let mut code = vec![
        0x8f09_8508, // s_lshl_b32 s9, s8, 5
        0xf42c_0300, 0x1200_0000, // s_buffer_load_dwordx8 s[12:19], s[0:3], s9
        0xbf8c_c07f, // s_waitcnt lgkmcnt(0)
        0x7e08_0280, // v_mov_b32 v4, 0
        0x7e0a_0280, // v_mov_b32 v5, 0
        0x7e0c_0280, // v_mov_b32 v6, 0
        0xf000_0f00 | dim << 3, 0x0003_0004, // image_load v[0:3], v[4:..], s[12:19] dmask 0xf
        0xbf8c_3f70, // s_waitcnt vmcnt(0)
        asm::v_mov_from_sgpr(6, 8),
        0x3400_0000 | (6 << 17) | (6 << 9) | 0x82, // v_lshlrev_b32 v6, 2, v6
    ];
    code.extend(asm::buffer_store_dword(0, 6, 4));
    code.push(asm::S_ENDPGM);
    code
}

/// A 1x1 RGBA8 T# at `tex`.
fn rgba8_texel_sharp(tex: u64) -> [u32; 8] {
    [(tex >> 8) as u32, ((tex >> 40) as u32 & 0xFF) | (56 << 20), 0, 0xFAC | (sharps::image_type::TYPE_2D << 28), 0, 0, 0, 0]
}

struct LargeTable {
    rig: Rig,
    cs: u64,
    heap: u64,
    textures: u64,
    out: u64,
}

const LARGE_KEYS: u32 = 64;

impl LargeTable {
    /// Texture `k` (256 bytes apart) holds `k * 7 + 1`; every record `k`
    /// with `k % 7 == 3` is null.
    fn new(g: &mut Guest) -> LargeTable {
        let rig = Rig::with_compiler(g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
        let cs = g.program(&group_program());
        let mut texels = Vec::new();
        for k in 0..LARGE_KEYS {
            texels.extend(std::iter::repeat_n(k * 7 + 1, 64));
        }
        let textures = g.alloc(&asm::bytes(&texels));
        let mut words = Vec::new();
        for k in 0..LARGE_KEYS as u64 {
            words.extend_from_slice(&if k % 7 == 3 { [0; 8] } else { rgba8_texel_sharp(textures + k * 256) });
        }
        let heap = g.alloc(&asm::bytes(&words));
        let out = g.zeroed(LARGE_KEYS as usize * 4);
        LargeTable { rig, cs, heap, textures, out }
    }

    /// One work group per key; the red channel each group read.
    fn run(&mut self, g: &Guest) -> Vec<f32> {
        let mut user = vec![0u32; 8];
        user[..4].copy_from_slice(&raw_buffer_descriptor(self.heap, LARGE_KEYS * 32));
        user[4..8].copy_from_slice(&buffer_descriptor(self.out, 0, LARGE_KEYS * 4, (3 << 28) | 0x0001_6FAC));
        let mut shader = stage(self.cs, &user);
        shader.rsrc2 |= 1 << 7; // TGID_X_EN: s8 = group id
        let d = ResolvedDispatch {
            packet: DispatchPacket { groups: [LARGE_KEYS, 1, 1], dispatch_initiator: 1 | (1 << 15), ..DispatchPacket::default() },
            shader,
            rsrc3: 0,
            threads: [1, 1, 1],
            start: [0; 3],
        };
        assert_eq!(self.rig.dispatch(&d), DrawOutcome::Recorded);
        assert!(self.rig.buffers.download(&mut self.rig.sched, self.out, LARGE_KEYS as u64 * 4));
        let mut bytes = vec![0u8; LARGE_KEYS as usize * 4];
        assert!(g.mem.read_backing(self.out, &mut bytes));
        bytes.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
    }
}

fn expected_red(value: u32) -> f32 {
    (value & 0xFF) as f32 / 255.0
}

fn assert_reds(reds: &[f32], expected: impl Fn(u32) -> f32, what: &str) {
    for (k, &red) in reds.iter().enumerate() {
        let want = expected(k as u32);
        assert!((red - want).abs() < 1e-3, "{what}: group {k} read {red}, want {want}");
    }
}

/// A 64-record table no draw-time key narrows (each work group reads its
/// own record) is selected through the host table map: every group reads
/// its record's texture, null records read zero, and a repeated dispatch
/// reads nothing of the table again.
#[test]
#[ignore]
fn bindless_large_table_selects_per_group() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut g = Guest::new(16 << 20);
    let mut t = LargeTable::new(&mut g);
    let reds = t.run(&g);
    let table = |k: u32| if k % 7 == 3 { 0.0 } else { expected_red(k * 7 + 1) };
    assert_reds(&reds, table, "first dispatch");
    let stats = t.rig.provider.bindless_stats().expect("the bindless set was used");
    assert_eq!((stats.binds, stats.reads, stats.maps), (1, 1, 1));
    assert_eq!(stats.revalidated, (0..LARGE_KEYS).filter(|k| k % 7 != 3).count() as u64, "one look-up per distinct non-null T#");
    // Nothing changed: the table is neither read nor mapped again. The
    // synchronisation point is process-wide: a device test running in
    // parallel that passes one makes the table's unwatched bytes due for a
    // read again (as for any guest memory), so the read count is checked
    // only when none passed meanwhile.
    let sync = crate::memory::sync_generation();
    assert_reds(&t.run(&g), table, "second dispatch");
    let again = t.rig.provider.bindless_stats().unwrap();
    let reads = if crate::memory::sync_generation() == sync { 1 } else { again.reads.min(2) };
    assert_eq!((again.binds, again.reads, again.maps, again.revalidated), (2, reads, 1, stats.revalidated));
}

/// The host map follows its table: a record the CPU points at another
/// texture (seen at the next synchronisation point, as all unwatched
/// guest memory) and a texture the CPU rewrites (its watch faults).
#[test]
#[ignore]
fn bindless_large_table_follows_cpu_writes() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut g = Guest::new(16 << 20);
    let mut t = LargeTable::new(&mut g);
    g.mem.attach(
        Arc::new(super::memo_invalidation::TrackerFaults(t.rig.buffers.tracker.clone())),
        Arc::new(super::memo_invalidation::TableFaults(t.rig.images.table.clone())),
        Arc::new(super::memo_invalidation::NoUnmapping),
    );
    t.run(&g);
    // Record 5 takes texture 9's T#.
    let words = asm::bytes(&rgba8_texel_sharp(t.textures + 9 * 256));
    // SAFETY: the table's bytes in the guest view.
    unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), (t.heap + 5 * 32) as *mut u8, words.len()) };
    crate::memory::note_sync_point();
    let before = t.rig.provider.bindless_stats().unwrap();
    let reds = t.run(&g);
    let after = t.rig.provider.bindless_stats().unwrap();
    assert_eq!((after.reads - before.reads, after.resolved - before.resolved, after.maps - before.maps), (1, 1, 1), "one record read and mapped again");
    let table = |k: u32| {
        let k = if k == 5 { 9 } else { k };
        if k % 7 == 3 { 0.0 } else { expected_red(k * 7 + 1) }
    };
    assert_reds(&reds, table, "after the CPU rewrote record 5");
    // Texture 12 gets other texels.
    let texels = asm::bytes(&[200u32; 64]);
    // SAFETY: the texture's bytes in the guest view (watched: the write faults).
    unsafe { std::ptr::copy_nonoverlapping(texels.as_ptr(), (t.textures + 12 * 256) as *mut u8, texels.len()) };
    let reds = t.run(&g);
    assert_reds(&reds, |k| if k == 12 { expected_red(200) } else { table(k) }, "after the CPU rewrote texture 12");
}

/// One table holds a 2D-array T# of one layer and one of four layers of the
/// same surface: looking up the second makes an image of four layers that
/// absorbs (deletes) the image the first one took a slot of in the same
/// bind. The first T# is looked at again before the map is written: no slot
/// holds a view of a deleted image (its view is destroyed once a submit
/// completes the tick, and a later write or read of it crashed the driver),
/// and both groups read the surface.
#[test]
#[ignore]
fn bindless_table_lookup_absorbing_an_earlier_image() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut g = Guest::new(16 << 20);
    let rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    let cs = g.program(&group_program_dim(5));
    let surface = g.alloc(&asm::bytes(&[0x51u32; 16384]));
    let array = |layers: u32| {
        let mut w = rgba8_texel_sharp(surface);
        w[3] = 0xFAC | (sharps::image_type::TYPE_2D_ARRAY << 28);
        w[4] = layers - 1;
        w
    };
    let mut words = vec![0u32; LARGE_KEYS as usize * 8];
    words[..8].copy_from_slice(&array(1));
    words[8..16].copy_from_slice(&array(4));
    let heap = g.alloc(&asm::bytes(&words));
    let out = g.zeroed(LARGE_KEYS as usize * 4);
    let mut t = LargeTable { rig, cs, heap, textures: surface, out };
    let deleted_before = t.rig.images.stats().deleted;
    let reds = t.run(&g);
    assert!(t.rig.images.stats().deleted > deleted_before, "the four-layer image absorbed the one-layer image");
    assert_reds(&reds, |k| if k < 2 { expected_red(0x51) } else { 0.0 }, "both T#s of the surface");
    let held = t.rig.provider.bindless_held_images();
    assert_eq!(held.len(), 2, "both T#s hold a view");
    for id in held {
        assert!(t.rig.images.get(id).is_some(), "a slot holds a view of deleted image {id}");
    }
}

/// s9 = s8 << 5; s[12:19] = heap[s9]; image_load v[0:3] at (0, 0); store
/// v0..v3 at output dwords 0..3 (output V# in s[4:7]).
fn program() -> Vec<u32> {
    let mut code = vec![
        0x8f09_8508, // s_lshl_b32 s9, s8, 5
        0xf42c_0300, 0x1200_0000, // s_buffer_load_dwordx8 s[12:19], s[0:3], s9
        0xbf8c_c07f, // s_waitcnt lgkmcnt(0)
        0x7e08_0280, // v_mov_b32 v4, 0
        0x7e0a_0280, // v_mov_b32 v5, 0
        0xf000_0f08, 0x0003_0004, // image_load v[0:3], v[4:5], s[12:19] dmask 0xf 2D
        0xbf8c_3f70, // s_waitcnt vmcnt(0)
    ];
    for c in 0..4u32 {
        code.push(0x7e0c_02ff); // v_mov_b32 v6, c*4
        code.push(c * 4);
        code.extend(asm::buffer_store_dword(c, 6, 4));
    }
    code.push(asm::S_ENDPGM);
    code
}

/// A 1x1 linear RGBA8 T# at `tex`.
fn texture(tex: u64) -> [u32; 8] {
    [(tex >> 8) as u32, ((tex >> 40) as u32 & 0xFF) | (56 << 20), 0, 0xFAC | (sharps::image_type::TYPE_2D << 28), 0, 0, 0, 0]
}

#[test]
#[ignore]
fn bindless_table_selects_the_keyed_texture() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut g = Guest::new(16 << 20);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    let cs = g.program(&program());
    let texels = [0x1122_3344u32, 0x5566_7788, 0x99AA_BBCC];
    let mut heap_words = Vec::new();
    for &t in &texels {
        let tex = g.alloc(&asm::bytes(&[t; 64]));
        heap_words.extend_from_slice(&texture(tex));
    }
    let heap = g.alloc(&asm::bytes(&heap_words));
    for (key, &t) in texels.iter().enumerate().rev() {
        let out = g.zeroed(16);
        let mut user = vec![0u32; 9];
        user[..4].copy_from_slice(&raw_buffer_descriptor(heap, heap_words.len() as u32 * 4));
        user[4..8].copy_from_slice(&buffer_descriptor(out, 0, 16, (3 << 28) | 0x0001_6FAC));
        user[8] = key as u32;
        let d = ResolvedDispatch {
            packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1 | (1 << 15), ..DispatchPacket::default() },
            shader: stage(cs, &user),
            rsrc3: 0,
            threads: [1, 1, 1],
            start: [0; 3],
        };
        assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded, "key {key}");
        assert!(rig.buffers.download(&mut rig.sched, out, 16));
        let mut bytes = [0u8; 16];
        assert!(g.mem.read_backing(out, &mut bytes));
        let red = f32::from_le_bytes(bytes[..4].try_into().unwrap());
        let want = (t & 0xFF) as f32 / 255.0;
        assert!((red - want).abs() < 1e-3, "key {key}: red {red}, want {want}");
    }
}

/// A table too large to bind whole (32 entries) whose key the draw sets in
/// a constant buffer: the evaluated key's entry is bound alone.
#[test]
#[ignore]
fn bindless_large_table_with_a_draw_key() {
    let mut g = Guest::new(16 << 20);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    let mut code = vec![
        0xf420_0304, 0xfa00_0000, // s_buffer_load_dword s12, s[8:11], 0
        0xbf8c_c07f,              // s_waitcnt lgkmcnt(0)
        0x8f0d_850c,              // s_lshl_b32 s13, s12, 5
        0xf42c_0400, 0x1a00_0000, // s_buffer_load_dwordx8 s[16:23], s[0:3], s13
        0xbf8c_c07f,
        0x7e08_0280,
        0x7e0a_0280,
        0xf000_0f08, 0x0004_0004, // image_load v[0:3], v[4:5], s[16:23]
        0xbf8c_3f70,
    ];
    code.extend([0x7e0c_0280]); // v6 = 0
    code.extend(asm::buffer_store_dword(0, 6, 4));
    code.push(asm::S_ENDPGM);
    let cs = g.program(&code);
    let mut heap_words = Vec::new();
    for k in 0..32u32 {
        let tex = g.alloc(&asm::bytes(&[k * 7 + 1; 64]));
        heap_words.extend_from_slice(&texture(tex));
    }
    let heap = g.alloc(&asm::bytes(&heap_words));
    for key in [20u32, 3] {
        let cb = g.alloc(&asm::bytes(&[key, 0, 0, 0]));
        let out = g.zeroed(16);
        let mut user = vec![0u32; 12];
        user[..4].copy_from_slice(&raw_buffer_descriptor(heap, heap_words.len() as u32 * 4));
        user[4..8].copy_from_slice(&buffer_descriptor(out, 0, 16, (3 << 28) | 0x0001_6FAC));
        user[8..12].copy_from_slice(&raw_buffer_descriptor(cb, 16));
        let d = ResolvedDispatch {
            packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1 | (1 << 15), ..DispatchPacket::default() },
            shader: stage(cs, &user),
            rsrc3: 0,
            threads: [1, 1, 1],
            start: [0; 3],
        };
        assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded, "key {key}");
        assert!(rig.buffers.download(&mut rig.sched, out, 4));
        let mut bytes = [0u8; 4];
        assert!(g.mem.read_backing(out, &mut bytes));
        let red = f32::from_le_bytes(bytes);
        let want = ((key * 7 + 1) & 0xFF) as f32 / 255.0;
        assert!((red - want).abs() < 1e-3, "key {key}: red {red}, want {want}");
    }
}

/// The cost of making the bindless set (GTA V froze 1.4 s on its first
/// table, a 131072-slot set).
#[test]
#[ignore]
fn bindless_set_creation_time() {
    let dev = GpuDevice::create(false).expect("device");
    if dev.caps.bindless_slots == 0 {
        eprintln!("bindless_set_creation_time: no bindless set on this device");
        return;
    }
    // What one small image costs the driver (create, allocation, bind):
    // its own allocation, or placed in a shared block.
    let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC;
    for shared in [false, true, false, true] {
        crate::suballoc::force_images(Some(shared));
        let t0 = std::time::Instant::now();
        let images: Vec<_> = (0..2048)
            .map(|_| crate::device::Image::new(&dev, vk::Format::R8G8B8A8_UNORM, vk::Extent3D { width: 1, height: 1, depth: 1 }, 1, 1, vk::SampleCountFlags::TYPE_1, usage, vk::ImageCreateFlags::MUTABLE_FORMAT).unwrap())
            .collect();
        let made = t0.elapsed();
        let t1 = std::time::Instant::now();
        drop(images);
        eprintln!("bindless_set_creation_time: 2048 1x1 images (shared blocks {shared}): create {:.1} us, destroy {:.1} us each", made.as_secs_f64() * 1e6 / 2048.0, t1.elapsed().as_secs_f64() * 1e6 / 2048.0);
    }
    crate::suballoc::force_images(None);
    for round in 0..3 {
        let t0 = std::time::Instant::now();
        let set = crate::pipeline::bindless::BindlessSet::new(&dev).expect("bindless set");
        let made = t0.elapsed();
        let t1 = std::time::Instant::now();
        drop(set);
        eprintln!("bindless_set_creation_time: round {round}: {} slots: create {:.1} ms, destroy {:.1} ms", dev.caps.bindless_slots, made.as_secs_f64() * 1e3, t1.elapsed().as_secs_f64() * 1e3);
    }
}

/// The first dispatch through a large table (`RPS5E_BINDLESS_BENCH_KEYS`
/// distinct 1x1 textures, default 8192) against the next ones: the time a
/// game's first frame with such a table stalls.
#[test]
#[ignore]
fn bindless_large_table_first_bind_time() {
    let Some(keys) = std::env::var("RPS5E_BINDLESS_BENCH_KEYS").ok().and_then(|v| v.parse::<u32>().ok()) else { return };
    let _ = env_logger::builder().is_test(true).try_init();
    // Textures `RPS5E_BINDLESS_BENCH_STRIDE` bytes apart (default 4096: one page each).
    let stride = std::env::var("RPS5E_BINDLESS_BENCH_STRIDE").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(4096).max(256);
    let mut g = Guest::new(((keys as usize * stride as usize * 2) + (16 << 20)).next_power_of_two());
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    let cs = g.program(&group_program());
    let mut texels = Vec::with_capacity(keys as usize * 64);
    for k in 0..keys {
        texels.extend(std::iter::repeat_n(k * 7 + 1, stride as usize / 4));
    }
    let textures = g.alloc(&asm::bytes(&texels));
    let mut words = Vec::new();
    for k in 0..keys as u64 {
        words.extend_from_slice(&rgba8_texel_sharp(textures + k * stride));
    }
    let heap = g.alloc(&asm::bytes(&words));
    let out = g.zeroed(keys as usize * 4);
    let mut user = vec![0u32; 8];
    user[..4].copy_from_slice(&raw_buffer_descriptor(heap, keys * 32));
    user[4..8].copy_from_slice(&buffer_descriptor(out, 0, keys * 4, (3 << 28) | 0x0001_6FAC));
    let mut shader = stage(cs, &user);
    shader.rsrc2 |= 1 << 7;
    let d = ResolvedDispatch { packet: DispatchPacket { groups: [keys, 1, 1], dispatch_initiator: 1 | (1 << 15), ..DispatchPacket::default() }, shader, rsrc3: 0, threads: [1, 1, 1], start: [0; 3] };
    // The program compiles on a 16-record table of another heap first.
    let small = g.alloc(&asm::bytes(&words[..16 * 8]));
    let mut warm = d.clone();
    warm.packet.groups = [16, 1, 1];
    warm.shader.user_data[..4].copy_from_slice(&raw_buffer_descriptor(small, 17 * 32));
    assert_eq!(rig.dispatch(&warm), DrawOutcome::Recorded);
    rig.sched.flush_and_wait();
    for round in 0..3 {
        let t0 = std::time::Instant::now();
        assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
        let cpu = t0.elapsed();
        rig.sched.flush_and_wait();
        eprintln!("bindless_large_table_first_bind_time: {keys} keys, dispatch {round}: record {:.1} ms, with GPU {:.1} ms", cpu.as_secs_f64() * 1e3, t0.elapsed().as_secs_f64() * 1e3);
    }
    assert!(rig.buffers.download(&mut rig.sched, out, keys as u64 * 4));
    let mut bytes = vec![0u8; keys as usize * 4];
    assert!(g.mem.read_backing(out, &mut bytes));
    for (k, b) in bytes.chunks(4).enumerate().step_by(97) {
        let red = f32::from_le_bytes(b.try_into().unwrap());
        assert!((red - expected_red(k as u32 * 7 + 1)).abs() < 1e-3, "key {k}: {red}");
    }
    let s = rig.provider.bindless_stats().unwrap();
    eprintln!("bindless_large_table_first_bind_time: {s:?}");
}
