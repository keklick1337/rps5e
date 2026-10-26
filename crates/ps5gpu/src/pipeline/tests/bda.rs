// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Read-only stages read guest memory through its imports:
//! no mirror, upload or buffer descriptor per use; CPU writes are seen by
//! the next use, GPU writes into mirrors once written through.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored bda`.

use super::*;
use ps5shader::testing::{asm as sa, buffer_descriptor, raw_buffer_descriptor};

const IMAGE_STORE: [u32; 2] = [0xf020_0f08, 0x0000_0014];
const S_WAITCNT_0: u32 = 0xBF8C_0000;

/// T# in s[0:7], V# in s[8:11]: texel (1, 2) of the storage image =
/// (s_buffer_load_dword at 0, buffer_load_dword at 4, 0, 0).
fn reader(g: &mut Guest) -> u64 {
    let mut code = Vec::new();
    code.extend(sa::smem(0x08, 12, 4, 0, sa::NULL));
    code.push(S_WAITCNT_0);
    code.push(sa::vop1(sa::OP_V_MOV_B32, 0, 12));
    code.extend(sa::mubuf(sa::OP_BUFFER_LOAD_DWORD, 1, 0, 2, sa::inline_u32(0), 4, false, false, false));
    code.push(sa::vop1(sa::OP_V_MOV_B32, 2, sa::inline_u32(0)));
    code.push(sa::vop1(sa::OP_V_MOV_B32, 3, sa::inline_u32(0)));
    code.push(sa::vop1(sa::OP_V_MOV_B32, 20, sa::inline_u32(1)));
    code.push(sa::vop1(sa::OP_V_MOV_B32, 21, sa::inline_u32(2)));
    code.push(S_WAITCNT_0);
    code.extend(IMAGE_STORE);
    code.push(sa::S_ENDPGM);
    g.program(&code)
}

/// T# in s[0:7], V# (32_32 UINT) in s[8:11]: texel (1, 2) of the storage
/// image = (buffer_load_format_xy at 0, 0, 0). A formatted load needs the
/// V#'s format on the host: its V# is materialised.
fn formatted_reader(g: &mut Guest) -> u64 {
    let mut code = Vec::new();
    code.extend(sa::mubuf(0x01, 0, 0, 2, sa::inline_u32(0), 0, false, false, false));
    code.push(sa::vop1(sa::OP_V_MOV_B32, 2, sa::inline_u32(0)));
    code.push(sa::vop1(sa::OP_V_MOV_B32, 3, sa::inline_u32(0)));
    code.push(sa::vop1(sa::OP_V_MOV_B32, 20, sa::inline_u32(1)));
    code.push(sa::vop1(sa::OP_V_MOV_B32, 21, sa::inline_u32(2)));
    code.push(S_WAITCNT_0);
    code.extend(IMAGE_STORE);
    code.push(sa::S_ENDPGM);
    g.program(&code)
}

/// 4x4 linear R32G32B32A32_UINT storage texture (row pitch 16 texels).
fn texture(tex: u64) -> [u32; 8] {
    [(tex >> 8) as u32, ((tex >> 40) as u32 & 0xFF) | (75 << 20) | (3 << 30), (3 << 14), 0xFAC | (sharps::image_type::TYPE_2D << 28), 0, 0, 0, 0]
}

fn dispatch(shader: Stage) -> ResolvedDispatch {
    ResolvedDispatch {
        packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() },
        shader,
        rsrc3: 0,
        threads: [64, 1, 1],
        start: [0; 3],
    }
}

/// Texel (1, 2) after a write-back.
fn texel(rig: &mut Rig, g: &Guest, tex: u64) -> [u32; 4] {
    let size = 64 * 4 * 16;
    rig.images.writeback_range(&mut rig.sched, &mut rig.buffers, tex, size);
    assert!(rig.buffers.download(&mut rig.sched, tex, size));
    let mut out = vec![0u8; size as usize];
    assert!(g.mem.read_backing(tex, &mut out));
    // Linear rows of 16-byte texels are 256-byte aligned: 16 texels.
    let o = (2 * 16 + 1) * 16;
    std::array::from_fn(|i| u32::from_le_bytes(out[o + 4 * i..o + 4 * i + 4].try_into().unwrap()))
}

#[test]
#[ignore]
fn bda_reads_guest_memory_without_mirrors() {
    let mut g = Guest::new(16 << 20);
    let cs = reader(&mut g);
    // v1 = s4 * 64 + v0; v2 = 1; buffer_store_dword v2, v1 idxen (V# s[0:3]).
    let writer = g.program(&[0xD746_0001, 0x0401_0C04, 0x7E04_0281, 0xE070_2000, 0x8000_0201, asm::S_ENDPGM]);
    let tex = g.zeroed(64 * 4 * 16);
    let data = g.alloc(&asm::bytes(&[0x1111, 0x2222, 0, 0]));
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    rig.buffers.set_bda(std::env::var("BDA_TEST_OFF").is_err());
    let mut user: Vec<u32> = texture(tex).to_vec();
    user.extend(raw_buffer_descriptor(data, 16));
    let mut shader = stage(cs, &user);
    shader.rsrc1 = (0xC0 << 12) | (1 << 21);
    shader.rsrc2 = (user.len() as u32) << 1;
    let read = dispatch(shader);

    assert_eq!(rig.dispatch(&read), DrawOutcome::Recorded);
    assert!(rig.buffers.bda_active() || std::env::var("BDA_TEST_OFF").is_ok(), "guest memory was not imported");
    assert!(rig.buffers.find_containing(data, 16).is_none(), "a read-only stage made a mirror");
    assert_eq!(texel(&mut rig, &g, tex), [0x1111, 0x2222, 0, 0]);
    assert_eq!(rig.provider.stats().bda_stages, 1);

    // A CPU write is seen by the next use, with nothing uploaded.
    // SAFETY: an unwatched chunk of the guest view.
    unsafe { std::ptr::copy_nonoverlapping(asm::bytes(&[0x3333, 0x4444]).as_ptr(), data as *mut u8, 8) };
    let uploads = rig.buffers.stats().uploads;
    assert_eq!(rig.dispatch(&read), DrawOutcome::Recorded);
    assert_eq!(texel(&mut rig, &g, tex), [0x3333, 0x4444, 0, 0]);
    assert_eq!(rig.buffers.stats().uploads, uploads);

    // A GPU write into the buffer's mirror reaches the reader.
    let mut w = stage(writer, &buffer_descriptor(data, 4, 4, 0x0101_6FAC));
    w.rsrc2 |= 1 << 7;
    let mut wd = dispatch(w);
    wd.threads = [4, 1, 1];
    assert_eq!(rig.dispatch(&wd), DrawOutcome::Recorded);
    assert!(rig.buffers.find_containing(data, 16).is_some(), "the writing stage binds a mirror");
    assert_eq!(rig.dispatch(&read), DrawOutcome::Recorded);
    assert_eq!(texel(&mut rig, &g, tex), [1, 1, 0, 0]);
    // The CPU sees the GPU write too.
    assert!(rig.buffers.download(&mut rig.sched, data, 16));
    let mut out = [0u8; 8];
    assert!(g.mem.read_backing(data, &mut out));
    assert_eq!(out, [1, 0, 0, 0, 1, 0, 0, 0]);
}

/// A large range a reading stage uses lives in its mirror (the import table
/// points at it) until the CPU writes it; then the stage reads guest memory
/// until the range is current in the mirror again.
#[test]
#[ignore]
fn bda_large_ranges_live_in_mirrors_while_the_cpu_leaves_them() {
    let mut g = Guest::new(16 << 20);
    let cs = formatted_reader(&mut g);
    let tex = g.zeroed(64 * 4 * 16);
    let len = 256 << 10;
    let mut words = vec![0u32; len / 4];
    words[0] = 0x5555;
    words[1] = 0x6666;
    let data = g.alloc(&asm::bytes(&words));
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    rig.buffers.set_bda(true);
    rig.buffers.set_bda_residency(true);
    let mut user: Vec<u32> = texture(tex).to_vec();
    user.extend(buffer_descriptor(data, 0, len as u32, (3 << 28) | (62 << 12) | 0xFAC));
    let mut shader = stage(cs, &user);
    shader.rsrc1 = (0xC0 << 12) | (1 << 21);
    shader.rsrc2 = (user.len() as u32) << 1;
    let read = dispatch(shader);

    assert_eq!(rig.dispatch(&read), DrawOutcome::Recorded);
    assert!(rig.buffers.bda_resident(data), "a large range is mirrored");
    assert_eq!(texel(&mut rig, &g, tex), [0x5555, 0x6666, 0, 0]);
    let flips = rig.buffers.stats().resident_flips;

    // A CPU write points the page back at guest memory: the next use sees
    // it, and the range is current in its mirror again afterwards.
    assert!(g.mem.write_backing(data, &asm::bytes(&[0x7777, 0x8888])));
    rig.buffers.tracker.mark_cpu_dirty_pages(data, 8);
    assert_eq!(rig.dispatch(&read), DrawOutcome::Recorded);
    assert_eq!(texel(&mut rig, &g, tex), [0x7777, 0x8888, 0, 0]);
    assert!(rig.buffers.stats().resident_evictions > 0);
    assert!(rig.buffers.bda_resident(data));
    assert!(rig.buffers.stats().resident_flips > flips);

    // Unchanged: no upload, no table update.
    let (uploads, flips) = (rig.buffers.stats().uploads, rig.buffers.stats().resident_flips);
    assert_eq!(rig.dispatch(&read), DrawOutcome::Recorded);
    assert_eq!(texel(&mut rig, &g, tex), [0x7777, 0x8888, 0, 0]);
    assert_eq!(rig.buffers.stats().uploads, uploads);
    assert_eq!(rig.buffers.stats().resident_flips, flips);
}

/// T# in s[0:7]; texel (1, 2) = buffer_load_format_xyzw of element 1 of
/// the V# in s[8:11] (`runtime` false: specialised by its format) or of
/// the V# read from the address in s[8:9] (`runtime`: one switch over the
/// formats).
fn format_xyzw_reader(g: &mut Guest, runtime: bool) -> u64 {
    let mut code = Vec::new();
    let srsrc = if runtime {
        // s_load_dwordx4 s[12:15], s[8:9], s10 (the group id, 0: an
        // address only known at run time).
        code.extend(sa::smem(0x02, 12, 4, 0, 10));
        code.push(S_WAITCNT_0);
        3
    } else {
        2
    };
    code.push(sa::vop1(sa::OP_V_MOV_B32, 1, sa::inline_u32(1)));
    // buffer_load_format_xyzw v[0:3], v1, s[..], 0 idxen.
    code.extend(sa::mubuf(0x03, 0, 1, srsrc, sa::inline_u32(0), 0, true, false, false));
    code.push(sa::vop1(sa::OP_V_MOV_B32, 20, sa::inline_u32(1)));
    code.push(sa::vop1(sa::OP_V_MOV_B32, 21, sa::inline_u32(2)));
    code.push(S_WAITCNT_0);
    code.extend(IMAGE_STORE);
    code.push(sa::S_ENDPGM);
    g.program(&code)
}

/// A formatted load through a descriptor read at run time returns what the
/// same load through the specialised descriptor returns, for every valid
/// format, two destination selects and two strides (an unaligned element
/// among them); an invalid format returns the element's dword 0 in every
/// component.
#[test]
#[ignore]
fn runtime_format_loads_match_specialised_ones() {
    let mut g = Guest::new(16 << 20);
    let fixed = format_xyzw_reader(&mut g, false);
    let runtime = format_xyzw_reader(&mut g, true);
    let tex = g.zeroed(64 * 4 * 16);
    let bytes: Vec<u8> = (0..64u32).map(|i| (i.wrapping_mul(0x9D) ^ 0x5A) as u8).collect();
    let data = g.alloc(&bytes);
    let vsharp = g.zeroed(16);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    rig.buffers.set_bda(true);
    let run = |rig: &mut Rig, cs: u64, user: Vec<u32>| {
        let mut shader = stage(cs, &user);
        shader.rsrc1 = (0xC0 << 12) | (1 << 21);
        // The group id x after the user data.
        shader.rsrc2 = ((user.len() as u32) << 1) | (1 << 7);
        assert_eq!(rig.dispatch(&dispatch(shader)), DrawOutcome::Recorded);
        texel(rig, &g, tex)
    };
    let mut checked = 0;
    for format in 0..128u32 {
        for sel in [0xFAC, 6 | (1 << 3) | (4 << 9)] {
            // Element 1 in range, or past the records (zeros and ones).
            for (stride, records) in [(16, 4), (6, 4), (16, 1)] {
                let v = buffer_descriptor(data, stride, records, (format << 12) | sel);
                assert!(g.mem.write_backing(vsharp, &asm::bytes(&v)));
                rig.buffers.tracker.mark_cpu_dirty_pages(vsharp, 16);
                let mut user = texture(tex).to_vec();
                user.extend([vsharp as u32, (vsharp >> 32) as u32]);
                let got = run(&mut rig, runtime, user);
                if ps5shader::interp::buffer::FormatInfo::new(format).is_none() && records > 1 {
                    let o = stride as usize;
                    let d0 = u32::from_le_bytes(bytes[o & !3..(o & !3) + 4].try_into().unwrap());
                    assert_eq!(got, [d0; 4], "invalid format {format}");
                    continue;
                }
                let mut user = texture(tex).to_vec();
                user.extend(v);
                let want = run(&mut rig, fixed, user);
                if ps5shader::interp::buffer::FormatInfo::new(format).is_none() {
                    // Out of range: raw dword 0 is zero.
                    assert_eq!(got, [0; 4], "invalid format {format} out of range");
                    continue;
                }
                assert_eq!(got, want, "format {format} sel {sel:#x} stride {stride} records {records}");
                checked += 1;
            }
        }
    }
    assert!(checked > 100, "{checked} valid cases");
}
