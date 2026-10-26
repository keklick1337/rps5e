// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Indirect draws (DRAW_INDIRECT, DRAW_INDEX_INDIRECT and their multi-draw
//! forms) as the command processor executes them: every argument record
//! writes its offsets to the user-data registers the packet names and
//! becomes one direct draw.
//!
//! The vertex and instance IDs a guest vertex shader receives do not
//! include the record's base vertex, first vertex or first instance: a
//! program that needs them reads the user-data registers written here and
//! adds them in its own vertex fetch. The host draw therefore starts at
//! vertex 0 and instance 0 (no host vertex offset or first instance), so
//! the offsets are applied exactly once. The first index only moves the
//! index range.
//!
//! The records are read when the draw is recorded (GPU-written records are
//! downloaded first).

use crate::backend::{DrawContext, DrawOutcome};
use crate::memory::GuestGpuMemory;
use crate::render::{DrawKind, IndexBuffer, IndexType, ResolvedDraw, SH_NOP};

/// Bytes of a DRAW_INDIRECT record: vertex count, instance count, first
/// vertex, first instance.
pub const AUTO_RECORD: u64 = 16;
/// Bytes of a DRAW_INDEX_INDIRECT record: index count, instance count,
/// first index, base vertex, first instance.
pub const INDEXED_RECORD: u64 = 20;

/// Read `len` bytes of guest memory, downloading GPU-written data first.
fn read(ctx: &mut DrawContext, memory: &GuestGpuMemory, addr: u64, len: u64) -> Option<Vec<u8>> {
    if ctx.buffers.has_gpu_dirty(addr, len) {
        ctx.buffers.download(ctx.sched, addr, len);
    }
    let mut bytes = vec![0u8; len as usize];
    memory.read_backing(addr, &mut bytes).then_some(bytes)
}

/// The argument records of an indirect draw: the draw count (from memory
/// when the packet gives a count address, clamped to the packet's
/// maximum) and the record words.
pub fn records(ctx: &mut DrawContext, memory: &GuestGpuMemory, d: &ResolvedDraw) -> Option<Vec<[u32; 5]>> {
    let p = &d.packet;
    let indexed = p.kind.indexed();
    let size = if indexed { INDEXED_RECORD } else { AUTO_RECORD };
    let multi = matches!(p.kind, DrawKind::IndirectMulti | DrawKind::IndexIndirectMulti);
    let mut count = if multi { p.indirect_count } else { 1 };
    if multi && p.count_address != 0 {
        let c = read(ctx, memory, p.count_address, 4)?;
        count = count.min(u32::from_le_bytes(c[..4].try_into().unwrap()));
    }
    if count == 0 {
        return Some(Vec::new());
    }
    let stride = if multi { p.stride as u64 } else { size };
    if stride < size {
        crate::log_once("ps5gpu: indirect draw stride below the record size");
        return None;
    }
    let bytes = read(ctx, memory, p.indirect_address, stride * (count as u64 - 1) + size)?;
    let words = (size / 4) as usize;
    Some(
        (0..count as usize)
            .map(|i| {
                let at = i * stride as usize;
                let mut r = [0u32; 5];
                for (k, w) in r.iter_mut().enumerate().take(words) {
                    *w = u32::from_le_bytes(bytes[at + k * 4..at + k * 4 + 4].try_into().unwrap());
                }
                r
            })
            .collect(),
    )
}

/// The direct draw of one argument record, with the record's offsets
/// written to the packet's destination registers. `draw_index` is the
/// record's position in a multi-draw (0 for single records), written to
/// the packet's draw-index register. None when the record draws nothing.
pub fn direct_draw(d: &ResolvedDraw, record: &[u32; 5], draw_index: u32) -> Option<ResolvedDraw> {
    let mut sub = d.clone();
    let regs = d.packet.indirect_registers;
    let mut write = |reg: u32, value: u32| {
        if reg != SH_NOP && !sub.shaders.write_user_data(reg, value) {
            crate::log_once(&format!("ps5gpu: indirect draw writes non-user-data register {reg:#x}"));
        }
    };
    write(regs.draw_index_target(), draw_index);
    let (count, instances) = (record[0], record[1]);
    let mut packet = d.packet;
    packet.indirect_address = 0;
    packet.indirect_count = 0;
    packet.count_address = 0;
    packet.stride = 0;
    packet.instance_count = instances;
    if d.packet.kind.indexed() {
        let (first_index, base_vertex, first_instance) = (record[2], record[3], record[4]);
        write(regs.vertex_offset, base_vertex);
        write(regs.instance_offset, first_instance);
        write(regs.first_index, first_index);
        let kind = IndexType::from_raw(d.packet.index_type);
        // INDEX_BUFFER_SIZE bounds the indices one record may read.
        let count = if d.packet.index_buffer_size != 0 { count.min(d.packet.index_buffer_size) } else { count };
        packet.kind = DrawKind::Index;
        packet.count = count;
        packet.index_address = d.packet.index_base + first_index as u64 * kind.bytes();
        sub.index = Some(IndexBuffer { address: packet.index_address, kind, count });
    } else {
        let (first_vertex, first_instance) = (record[2], record[3]);
        write(regs.vertex_offset, first_vertex);
        write(regs.instance_offset, first_instance);
        packet.kind = DrawKind::Auto;
        packet.count = count;
        sub.index = None;
    }
    // GE_INDEX_OFFSET is not added to the vertex IDs of indirect draws.
    sub.index_offset = 0;
    sub.packet = packet;
    (count != 0 && instances != 0).then_some(sub)
}

/// The outcome of several draws recorded for one packet.
pub fn merge(a: DrawOutcome, b: DrawOutcome) -> DrawOutcome {
    match (a, b) {
        (DrawOutcome::NeedsShader, _) | (_, DrawOutcome::NeedsShader) => DrawOutcome::NeedsShader,
        (DrawOutcome::Recorded, _) | (_, DrawOutcome::Recorded) => DrawOutcome::Recorded,
        _ => DrawOutcome::Skipped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{DrawPacket, IndirectRegisters};

    const GS_0: u32 = 0x8C;
    const HS_0: u32 = 0x10C;

    #[test]
    fn decodes_destination_words() {
        // Single indexed packet: first index in the high half of word 1,
        // enabled by word 2 bit 28 (bit 27 does not enable it).
        let w1 = (GS_0 + 9) | ((GS_0 + 11) << 16);
        let r = IndirectRegisters::decode(w1, (GS_0 + 10) | (1 << 28), (GS_0 + 10) | (1 << 28), true);
        assert_eq!(r, IndirectRegisters { vertex_offset: GS_0 + 9, instance_offset: GS_0 + 10, first_index: GS_0 + 11, draw_index: SH_NOP });
        let r = IndirectRegisters::decode(w1, (GS_0 + 10) | (1 << 27), (GS_0 + 10) | (1 << 27), true);
        assert_eq!(r.first_index, SH_NOP);
        // Multi-draw: the enable moves to word 3.
        let r = IndirectRegisters::decode(w1, GS_0 + 10, (1 << 30) | (1 << 28) | SH_NOP, true);
        assert_eq!(r.first_index, GS_0 + 11);
        let r = IndirectRegisters::decode(w1, (GS_0 + 10) | (1 << 28), SH_NOP, true);
        assert_eq!(r.first_index, SH_NOP);
        // Non-indexed packets have no first-index destination.
        let r = IndirectRegisters::decode((HS_0 + 9) | ((HS_0 + 11) << 16), HS_0 + 10, 1 << 28, false);
        assert_eq!(r, IndirectRegisters { vertex_offset: HS_0 + 9, instance_offset: HS_0 + 10, first_index: SH_NOP, draw_index: SH_NOP });
        // Multi-draw word 3: DRAW_INDEX_ENABLE (bit 31) with the draw-index
        // location in the low half, next to the count-from-memory (30) and
        // first-index (28) enables. Words as a title's packets carry them.
        let r = IndirectRegisters::decode_multi(SH_NOP, SH_NOP, 0x8000_0098, true);
        assert_eq!(r, IndirectRegisters { vertex_offset: SH_NOP, instance_offset: SH_NOP, first_index: SH_NOP, draw_index: GS_0 + 12 });
        let r = IndirectRegisters::decode_multi(w1, GS_0 + 10, (1 << 31) | (1 << 30) | (1 << 28) | (GS_0 + 16), true);
        assert_eq!(r, IndirectRegisters { vertex_offset: GS_0 + 9, instance_offset: GS_0 + 10, first_index: GS_0 + 11, draw_index: GS_0 + 16 });
        // Without the enable (or with location 0) nothing is written.
        assert_eq!(IndirectRegisters::decode_multi(w1, GS_0 + 10, GS_0 + 12, true).draw_index, SH_NOP);
        assert_eq!(IndirectRegisters::decode_multi(w1, GS_0 + 10, 1 << 31, true).draw_index, SH_NOP);
        // Single draws have no draw-index destination.
        assert_eq!(IndirectRegisters::decode(w1, GS_0 + 10, 0x8000_0098, true).draw_index, SH_NOP);
    }

    #[test]
    fn records_write_their_draw_index() {
        let regs = IndirectRegisters { draw_index: GS_0 + 12, ..IndirectRegisters::default() };
        let mut d = draw(DrawPacket { kind: DrawKind::IndexIndirectMulti, index_base: 0x1000, index_buffer_size: 64, indirect_registers: regs, ..DrawPacket::default() });
        d.shaders.gs.user_data[12] = 0x8000_00A4;
        for i in [0, 1, 20] {
            let s = direct_draw(&d, &[6, 1, 0, 0, 0], i).unwrap();
            assert_eq!(s.shaders.gs.user_data[12], i, "record {i}");
        }
        // No destination: the register keeps its value.
        let d = draw(DrawPacket { kind: DrawKind::IndexIndirectMulti, index_base: 0x1000, index_buffer_size: 64, ..DrawPacket::default() });
        let s = direct_draw(&d, &[6, 1, 0, 0, 0], 3).unwrap();
        assert_eq!(s.shaders.gs.user_data, d.shaders.gs.user_data);
    }

    fn draw(packet: DrawPacket) -> ResolvedDraw {
        let b = crate::render::RegisterBanks { context: &[], shader: &[], uconfig: &[] };
        crate::render::resolve_draw(&b, &packet)
    }

    #[test]
    fn records_become_direct_draws_with_patched_user_data() {
        let regs = IndirectRegisters { vertex_offset: GS_0 + 9, instance_offset: GS_0 + 10, first_index: GS_0 + 11, draw_index: SH_NOP };
        let mut d = draw(DrawPacket {
            kind: DrawKind::IndexIndirect,
            index_base: 0x1000,
            index_type: 1,
            index_buffer_size: 4,
            indirect_registers: regs,
            ..DrawPacket::default()
        });
        d.index_offset = 3;
        let s = direct_draw(&d, &[6, 2, 5, (-17i32) as u32, 3], 0).unwrap();
        assert_eq!(s.packet.kind, DrawKind::Index);
        assert_eq!((s.packet.count, s.packet.instance_count), (4, 2), "index count clamped to INDEX_BUFFER_SIZE");
        assert_eq!(s.index.unwrap().address, 0x1000 + 5 * 4);
        assert_eq!(s.index_offset, 0);
        assert_eq!(&s.shaders.gs.user_data[9..12], &[(-17i32) as u32, 3, 5]);
        assert!(direct_draw(&d, &[6, 0, 5, 0, 0], 0).is_none(), "no instances");

        let d = draw(DrawPacket {
            kind: DrawKind::Indirect,
            indirect_registers: IndirectRegisters { vertex_offset: HS_0 + 9, ..IndirectRegisters::default() },
            ..DrawPacket::default()
        });
        let s = direct_draw(&d, &[3, 1, 31, 5, 0], 0).unwrap();
        assert_eq!((s.packet.kind, s.packet.count, s.packet.instance_count), (DrawKind::Auto, 3, 1));
        assert!(s.index.is_none());
        assert_eq!(s.shaders.hs.user_data[9], 31);
        assert_eq!(s.shaders.hs.user_data[10], 0, "disabled destination left alone");
        assert_eq!(s.shaders.gs.user_data, d.shaders.gs.user_data);
    }

    #[test]
    fn outcomes_merge() {
        use DrawOutcome::*;
        assert_eq!(merge(Skipped, Recorded), Recorded);
        assert_eq!(merge(Recorded, NeedsShader), NeedsShader);
        assert_eq!(merge(Skipped, Skipped), Skipped);
    }
}
