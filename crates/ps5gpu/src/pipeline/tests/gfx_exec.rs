// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The bench of the reference driver's graphics execution tests: guest
//! vertex and pixel programs both compiled to SPIR-V, drawn into linear
//! targets, pixels and guest buffers read back.
//!
//! The reference runs its vertex programs with user data from s0; ours
//! run in the merged layout (user data from s8, the vertex index in v5),
//! so the vertex programs read their V# from s[8:11].

use super::*;

/// The colour a target holds where nothing was drawn.
pub(super) const KEPT: u8 = 0x40;
/// SPI_PS_INPUT_ENA/ADDR: POS_X_FLOAT and POS_Y_FLOAT (v0, v1).
pub(super) const POS_XY: u32 = (1 << 8) | (1 << 9);
/// SPI_PS_INPUT_ENA/ADDR: POS_Z_FLOAT and POS_W_FLOAT.
pub(super) const POS_ZW: u32 = (1 << 10) | (1 << 11);
/// SPI_PS_INPUT_ENA/ADDR: ANCILLARY.
pub(super) const ANCILLARY: u32 = 1 << 13;
/// SPI_PS_INPUT_ENA/ADDR: PERSP_CENTER.
pub(super) const PERSP_CENTER: u32 = 1 << 1;

/// buffer_load_format_xyzw v[0:3], v5, s[8:11], 0 idxen; s_waitcnt
/// vmcnt(0); exp pos0 v0, v1, v2, v3 done; s_endpgm.
pub(super) const VERTEX: [u32; 6] = [0xe0382000, 0x80020005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000];
/// v_mov_b32 v7, 1.0; exp mrt0 v7, v7, v7, v7 done vm; s_endpgm.
pub(super) const WHITE: [u32; 4] = [0x7e0e02f2, 0xf800180f, 0x07070707, 0xbf810000];
/// One triangle over the whole target.
pub(super) const COVER: [f32; 12] = [-1.0, -1.0, 0.5, 1.0, 3.0, -1.0, 0.5, 1.0, -1.0, 3.0, 0.5, 1.0];

/// A V# of `count` records of `stride` bytes at `addr` (32_32_32_32_FLOAT).
pub(super) fn v_sharp(addr: u64, stride: u32, count: u32) -> [u32; 4] {
    [addr as u32, ((addr >> 32) as u32 & 0xffff) | (stride << 16), count, 0x0104_dfac]
}

pub(super) struct Bench {
    pub g: Guest,
    pub rig: Rig,
}

impl Bench {
    pub fn new() -> Bench {
        // A refused program logs why.
        let _ = env_logger::builder().is_test(true).try_init();
        let g = Guest::new(128 << 20);
        let rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
        Bench { g, rig }
    }

    /// A linear RGBA8 target of `w` x `h` holding `fill` (rows are 64
    /// texels apart).
    pub fn target(&mut self, w: u32, h: u32, fill: u8) -> ColorTarget {
        let bytes = (w.next_multiple_of(64) * h * 4) as usize;
        rgba8_target(self.g.alloc(&vec![fill; bytes]), w, h)
    }

    /// A linear target of CB_COLOR_INFO `layout`, `number_type` and
    /// `order`, `w` x `h`, every byte `fill`.
    pub fn target_of(&mut self, w: u32, h: u32, layout: u32, number_type: u32, order: u32, fill: u8) -> ColorTarget {
        let format = render_target_format(layout, number_type, order).expect("target format");
        let bpe = format.bytes_per_element;
        let bytes = (w.next_multiple_of(64) * h * bpe) as usize;
        let mut t = rgba8_target(self.g.alloc(&vec![fill; bytes]), w, h);
        (t.format, t.layout, t.number_type, t.order, t.slice_size) = (format, layout, number_type, order, (w * h * bpe) as u64);
        t
    }

    /// The texels of a target of `bpe` bytes an element, row after row.
    pub fn texels(&mut self, t: &ColorTarget, bpe: u32) -> Vec<u8> {
        let id = self.rig.images.find_or_create(&mut self.rig.sched, &mut self.rig.buffers, &color_key(t)).unwrap();
        self.rig.images.transit(&mut self.rig.sched, id, crate::images::TRANSFER_SRC, None);
        let image = self.rig.images.get(id).unwrap().image.image;
        let size = (t.width * t.height * bpe) as u64;
        let out = Buffer::new(&self.rig.dev, size, vk::BufferUsageFlags::TRANSFER_DST, MemoryUse::Download).unwrap();
        self.rig.sched.full_barrier();
        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
            .image_extent(vk::Extent3D { width: t.width, height: t.height, depth: 1 });
        let cb = self.rig.sched.cmd();
        // SAFETY: recording buffer; the image is in TRANSFER_SRC layout.
        unsafe { self.rig.dev.device.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, out.buffer, &[region]) };
        self.rig.sched.flush_and_wait();
        out.invalidate(0, size);
        out.bytes_mut()[..size as usize].to_vec()
    }

    /// Words in guest memory.
    pub fn words(&mut self, words: &[u32]) -> u64 {
        self.g.alloc(&asm::bytes(words))
    }

    /// The positions as a vertex buffer: the vertex program's user data.
    pub fn positions(&mut self, positions: &[f32]) -> Vec<u32> {
        let at = self.g.alloc(&floats(positions));
        v_sharp(at, 16, (positions.len() / 4) as u32).to_vec()
    }

    /// A draw of `count` vertices of `vs` and `ps` into `target`, the
    /// viewport's y pointing up (as the reference's), both stages wave32
    /// when `wave32`.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(&mut self, target: ColorTarget, vs: u64, vs_user: &[u32], ps: u64, ps_user: &[u32], count: u32, wave32: bool, edit: impl FnOnce(&mut ResolvedDraw)) {
        let mut d = draw_into(target, stage(vs, vs_user), stage(ps, ps_user), primitive::TRI_LIST, auto(count, 1), None);
        // PA_CL_VPORT_YSCALE negated, as the registers resolve.
        let v = &mut d.viewports[0];
        v.scale[1] = -v.scale[1];
        (v.y, v.height) = (v.offset[1] - v.scale[1], 2.0 * v.scale[1]);
        if wave32 {
            d.ps_in_control |= 1 << 15;
            d.shaders.stages |= 1 << 22;
        }
        edit(&mut d);
        assert_eq!(self.rig.draw(&d), DrawOutcome::Recorded);
    }

    /// `n` words of guest memory at `addr` after the GPU's writes.
    pub fn read(&mut self, addr: u64, n: usize) -> Vec<u32> {
        assert!(self.rig.buffers.download(&mut self.rig.sched, addr, n as u64 * 4));
        let mut out = vec![0u8; n * 4];
        assert!(self.g.mem.read_backing(addr, &mut out));
        out.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
    }
}
