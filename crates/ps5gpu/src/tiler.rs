// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! GPU (de)tiling: compute shaders that convert between a tiled guest
//! surface and tightly packed linear rows, both held in storage buffers.
//!
//! One shader per (swizzle mode, element size, direction), assembled with
//! `rspirv` from the same address equations the CPU path uses
//! ([`crate::tiling::address_equation`]), so the two stay in lockstep.
//! Each invocation moves one element:
//!
//! ```text
//!   s      = firstSlice + z                       (array layer / depth slice)
//!   (u, v) = (x + tailX, y + tailY)               (position in the mip tail)
//!   tiled  = mipOffset + (s >> sliceShift) * layerBytes
//!          + ((v / BH) * blocksPerRow + u / BW) * blockBytes
//!          + inBlock(u, v, s)                     (XOR equation)
//!   linear = z * sliceBytes + y * rowBytes + x * bpe
//! ```
//!
//! `layerBytes` is the stride of an array layer (thin surfaces: the whole
//! mip chain of one layer) or of a slab of `2^sliceShift` slices (thick
//! volumes); the slice index also enters the in-block XOR (the `_X` modes'
//! pipe/bank bits, or the depth bits of a thick block). See
//! [`crate::tiling_layout`] for where each level lives.
//!
//! Elements narrower than a dword are merged into the destination with
//! atomic AND/OR so neighbouring invocations do not race.
//!
//! Arguments (push constants, 16 dwords): source base, destination base
//! (bytes), width, height, slices, first slice, linear row bytes, linear
//! slice bytes, blocks per row, layer bytes, tail x, tail y, slice shift,
//! mip offset and two reserved dwords.

use crate::device::GpuDevice;
use ash::vk;
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{self, Word};
use std::collections::HashMap;
use std::sync::Arc;

pub const ARGUMENT_COUNT: usize = 16;

/// One (de)tile of a mip level: a `width` x `height` x `slices` box of
/// elements (see the module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TileRegion {
    pub width: u32,
    pub height: u32,
    pub slices: u32,
    pub first_slice: u32,
    /// Linear side: bytes per row and per slice.
    pub row_bytes: u32,
    pub slice_bytes: u32,
    /// Tiled side.
    pub blocks_per_row: u32,
    pub layer_bytes: u32,
    pub tail_x: u32,
    pub tail_y: u32,
    /// log2 of the slices per layer stride (thick block depth; 0 thin).
    pub slice_shift: u32,
    /// Offset of the level from the start of the tiled binding.
    pub mip_offset: u32,
}

impl TileRegion {
    /// Mip 0 of a plain `width` x `height` x `layers` surface, tightly
    /// packed on the linear side, `layer_bytes` apart on the tiled side.
    pub fn simple(width: u32, height: u32, layers: u32, bytes_per_element: u32, blocks_per_row: u32, layer_bytes: u32) -> TileRegion {
        TileRegion {
            width,
            height,
            slices: layers.max(1),
            first_slice: 0,
            row_bytes: width * bytes_per_element,
            slice_bytes: width * height * bytes_per_element,
            blocks_per_row,
            layer_bytes,
            tail_x: 0,
            tail_y: 0,
            slice_shift: 0,
            mip_offset: 0,
        }
    }

    fn words(&self, src_base: u32, dst_base: u32) -> [u32; ARGUMENT_COUNT] {
        [
            src_base,
            dst_base,
            self.width,
            self.height,
            self.slices,
            self.first_slice,
            self.row_bytes,
            self.slice_bytes,
            self.blocks_per_row,
            self.layer_bytes,
            self.tail_x,
            self.tail_y,
            self.slice_shift,
            self.mip_offset,
            0,
            0,
        ]
    }
}

/// Minimal SPIR-V module context shared by the fixed compute shaders.
pub(crate) struct Spv {
    pub(crate) b: Builder,
    pub(crate) uint: Word,
    pub(crate) boolean: Word,
    pub(crate) v3: Word,
    pub(crate) p_sb_uint: Word,
    p_sb_struct: Word,
    pub(crate) src: Word,
    pub(crate) dst: Word,
    pub(crate) args: Word,
    pub(crate) p_pc_uint: Word,
    pub(crate) gid: Word,
    pub(crate) consts: HashMap<u32, Word>,
}

pub(crate) type R<T> = Result<T, rspirv::dr::Error>;

impl Spv {
    pub(crate) fn new() -> Spv {
        Spv::with_arguments(ARGUMENT_COUNT)
    }

    /// A module whose push constants are `arguments` dwords.
    pub(crate) fn with_arguments(arguments: usize) -> Spv {
        use spirv::{Decoration as D, StorageClass as S};
        let mut b = Builder::new();
        b.set_version(1, 3);
        b.capability(spirv::Capability::Shader);
        b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
        let uint = b.type_int(32, 0);
        let boolean = b.type_bool();
        let v3 = b.type_vector(uint, 3);
        let rta = b.type_runtime_array(uint);
        b.decorate(rta, D::ArrayStride, [Operand::LiteralBit32(4)]);
        let sb_struct = b.type_struct([rta]);
        b.decorate(sb_struct, D::Block, []);
        b.member_decorate(sb_struct, 0, D::Offset, [Operand::LiteralBit32(0)]);
        let p_sb_struct = b.type_pointer(None, S::StorageBuffer, sb_struct);
        let p_sb_uint = b.type_pointer(None, S::StorageBuffer, uint);
        let src = b.variable(p_sb_struct, None, S::StorageBuffer, None);
        b.name(src, "input_buffer");
        b.decorate(src, D::DescriptorSet, [Operand::LiteralBit32(0)]);
        b.decorate(src, D::Binding, [Operand::LiteralBit32(0)]);
        b.decorate(src, D::NonWritable, []);
        let dst = b.variable(p_sb_struct, None, S::StorageBuffer, None);
        b.name(dst, "output_buffer");
        b.decorate(dst, D::DescriptorSet, [Operand::LiteralBit32(0)]);
        b.decorate(dst, D::Binding, [Operand::LiteralBit32(1)]);
        let pc_struct = b.type_struct(vec![uint; arguments]);
        b.decorate(pc_struct, D::Block, []);
        for i in 0..arguments as u32 {
            b.member_decorate(pc_struct, i, D::Offset, [Operand::LiteralBit32(i * 4)]);
        }
        let p_pc_struct = b.type_pointer(None, S::PushConstant, pc_struct);
        let p_pc_uint = b.type_pointer(None, S::PushConstant, uint);
        let args = b.variable(p_pc_struct, None, S::PushConstant, None);
        b.name(args, "params");
        let p_in_v3 = b.type_pointer(None, S::Input, v3);
        let gid = b.variable(p_in_v3, None, S::Input, None);
        b.name(gid, "gl_GlobalInvocationID");
        b.decorate(gid, D::BuiltIn, [Operand::BuiltIn(spirv::BuiltIn::GlobalInvocationId)]);
        Spv { b, uint, boolean, v3, p_sb_uint, p_sb_struct, src, dst, args, p_pc_uint, gid, consts: HashMap::new() }
    }

    /// Another storage buffer of words at `binding` of set 0.
    pub(crate) fn storage_buffer(&mut self, binding: u32, name: &str, read_only: bool) -> Word {
        use spirv::{Decoration as D, StorageClass as S};
        let v = self.b.variable(self.p_sb_struct, None, S::StorageBuffer, None);
        self.b.name(v, name);
        self.b.decorate(v, D::DescriptorSet, [Operand::LiteralBit32(0)]);
        self.b.decorate(v, D::Binding, [Operand::LiteralBit32(binding)]);
        if read_only {
            self.b.decorate(v, D::NonWritable, []);
        }
        v
    }

    /// Store `value` to word `index` of `buf`.
    pub(crate) fn store_word(&mut self, buf: Word, index: Word, value: Word) -> R<()> {
        let p = self.word_ptr(buf, index)?;
        self.b.store(p, value, None, [])
    }

    /// `if cond { a } else { b }` (u32).
    pub(crate) fn select(&mut self, cond: Word, a: Word, b: Word) -> R<Word> {
        self.b.select(self.uint, None, cond, a, b)
    }

    pub(crate) fn c(&mut self, v: u32) -> Word {
        if let Some(&id) = self.consts.get(&v) {
            return id;
        }
        let id = self.b.constant_bit32(self.uint, v);
        self.consts.insert(v, id);
        id
    }

    pub(crate) fn arg(&mut self, i: u32) -> R<Word> {
        let idx = self.c(i);
        let p = self.b.access_chain(self.p_pc_uint, None, self.args, [idx])?;
        self.b.load(self.uint, None, p, None, [])
    }

    pub(crate) fn word_ptr(&mut self, buf: Word, index: Word) -> R<Word> {
        let zero = self.c(0);
        self.b.access_chain(self.p_sb_uint, None, buf, [zero, index])
    }

    pub(crate) fn load_word(&mut self, buf: Word, index: Word) -> R<Word> {
        let p = self.word_ptr(buf, index)?;
        self.b.load(self.uint, None, p, None, [])
    }

    pub(crate) fn shl(&mut self, v: Word, bits: u32) -> R<Word> {
        if bits == 0 {
            return Ok(v);
        }
        let c = self.c(bits);
        self.b.shift_left_logical(self.uint, None, v, c)
    }

    pub(crate) fn shr(&mut self, v: Word, bits: u32) -> R<Word> {
        if bits == 0 {
            return Ok(v);
        }
        let c = self.c(bits);
        self.b.shift_right_logical(self.uint, None, v, c)
    }

    pub(crate) fn and_c(&mut self, v: Word, mask: u32) -> R<Word> {
        let c = self.c(mask);
        self.b.bitwise_and(self.uint, None, v, c)
    }

    pub(crate) fn add(&mut self, a: Word, b: Word) -> R<Word> {
        self.b.i_add(self.uint, None, a, b)
    }

    pub(crate) fn add_c(&mut self, a: Word, k: u32) -> R<Word> {
        let c = self.c(k);
        self.b.i_add(self.uint, None, a, c)
    }

    pub(crate) fn mul(&mut self, a: Word, b: Word) -> R<Word> {
        self.b.i_mul(self.uint, None, a, b)
    }

    pub(crate) fn mul_c(&mut self, a: Word, k: u32) -> R<Word> {
        let c = self.c(k);
        self.b.i_mul(self.uint, None, a, c)
    }

    pub(crate) fn div_c(&mut self, a: Word, k: u32) -> R<Word> {
        if k.is_power_of_two() {
            return self.shr(a, k.trailing_zeros());
        }
        let c = self.c(k);
        self.b.u_div(self.uint, None, a, c)
    }

    pub(crate) fn xor(&mut self, a: Word, b: Word) -> R<Word> {
        self.b.bitwise_xor(self.uint, None, a, b)
    }

    /// `((v >> src_bit) & 1) << dst_bit`
    pub(crate) fn bit(&mut self, v: Word, src_bit: u32, dst_bit: u32) -> R<Word> {
        let s = self.shr(v, src_bit)?;
        let m = self.and_c(s, 1)?;
        self.shl(m, dst_bit)
    }

    /// End the invocation when `cond` holds; the body continues in a new block.
    pub(crate) fn return_when(&mut self, cond: Word) -> R<()> {
        let exit = self.b.id();
        let body = self.b.id();
        self.b.selection_merge(body, spirv::SelectionControl::NONE)?;
        self.b.branch_conditional(cond, exit, body, [])?;
        self.b.begin_block(Some(exit))?;
        self.b.ret()?;
        self.b.begin_block(Some(body))?;
        Ok(())
    }

    pub(crate) fn begin_main(&mut self) -> R<(Word, Word, Word, Word)> {
        let void = self.b.type_void();
        let fnty = self.b.type_function(void, []);
        let main = self.b.begin_function(void, None, spirv::FunctionControl::NONE, fnty)?;
        self.b.name(main, "main");
        self.b.begin_block(None)?;
        let g = self.b.load(self.v3, None, self.gid, None, [])?;
        let x = self.b.composite_extract(self.uint, None, g, [0])?;
        let y = self.b.composite_extract(self.uint, None, g, [1])?;
        let z = self.b.composite_extract(self.uint, None, g, [2])?;
        Ok((main, x, y, z))
    }

    pub(crate) fn finish(mut self, main: Word, lx: u32, ly: u32) -> R<Vec<u32>> {
        self.b.ret()?;
        self.b.end_function()?;
        self.b.execution_mode(main, spirv::ExecutionMode::LocalSize, [lx, ly, 1]);
        self.b.entry_point(spirv::ExecutionModel::GLCompute, main, "main", [self.gid]);
        Ok(self.b.module().assemble())
    }

    /// Copy one element of `bytes` from byte offset `src` (input buffer) to
    /// byte offset `dst` (output buffer).
    pub(crate) fn copy_element(&mut self, src: Word, dst: Word, bytes: u32) -> R<()> {
        if bytes >= 4 {
            for w in (0..bytes).step_by(4) {
                let wc = self.c(w);
                let s = self.add(src, wc)?;
                let si = self.shr(s, 2)?;
                let v = self.load_word(self.src, si)?;
                let d = self.add(dst, wc)?;
                let di = self.shr(d, 2)?;
                let p = self.word_ptr(self.dst, di)?;
                self.b.store(p, v, None, [])?;
            }
            return Ok(());
        }
        let mask = self.c(if bytes == 1 { 0xFF } else { 0xFFFF });
        let si = self.shr(src, 2)?;
        let loaded = self.load_word(self.src, si)?;
        let sb = self.and_c(src, 3)?;
        let sshift = self.mul_c(sb, 8)?;
        let shifted = self.b.shift_right_logical(self.uint, None, loaded, sshift)?;
        let value = self.b.bitwise_and(self.uint, None, shifted, mask)?;
        let db = self.and_c(dst, 3)?;
        let dshift = self.mul_c(db, 8)?;
        let di = self.shr(dst, 2)?;
        let target = self.word_ptr(self.dst, di)?;
        let dmask = self.b.shift_left_logical(self.uint, None, mask, dshift)?;
        let clear = self.b.not(self.uint, None, dmask)?;
        let scope = self.c(spirv::Scope::Device as u32);
        let sem = self.c(0);
        self.b.atomic_and(self.uint, None, target, scope, sem, clear)?;
        let placed = self.b.shift_left_logical(self.uint, None, value, dshift)?;
        self.b.atomic_or(self.uint, None, target, scope, sem, placed)?;
        Ok(())
    }
}

/// SPIR-V of the block copy for `mode` / `bytes_per_element`, towards the
/// tiled layout when `to_tiled`, with the thick (3D) equation when
/// `thick`. None for unsupported combinations.
pub fn block_copy_spirv(mode: u32, bytes_per_element: u32, to_tiled: bool, thick: bool) -> Option<Vec<u32>> {
    let (eq, bw, bh, block_bytes) = if thick {
        crate::tiling::thick_address_equation(mode, bytes_per_element as usize)?
    } else {
        crate::tiling::address_equation(mode, bytes_per_element as usize)?
    };
    let build = || -> R<Vec<u32>> {
        let mut s = Spv::new();
        let (main, x, y, z) = s.begin_main()?;
        let (w, h, d) = (s.arg(2)?, s.arg(3)?, s.arg(4)?);
        let ox = s.b.u_greater_than_equal(s.boolean, None, x, w)?;
        let oy = s.b.u_greater_than_equal(s.boolean, None, y, h)?;
        let oz = s.b.u_greater_than_equal(s.boolean, None, z, d)?;
        let oxy = s.b.logical_or(s.boolean, None, ox, oy)?;
        let out = s.b.logical_or(s.boolean, None, oxy, oz)?;
        s.return_when(out)?;

        let first = s.arg(5)?;
        let slice = s.add(first, z)?;
        let tx = s.arg(10)?;
        let ty = s.arg(11)?;
        let u = s.add(x, tx)?;
        let v = s.add(y, ty)?;
        // In-block offset: XOR of the equation's coordinate bits.
        let mut inner: Option<Word> = None;
        for (dst_bit, &(xm, ym, zm)) in eq.iter().enumerate() {
            for (coord, m) in [(u, xm), (v, ym), (slice, zm)] {
                let mut m = m;
                while m != 0 {
                    let sbit = m.trailing_zeros();
                    m &= m - 1;
                    let t = s.bit(coord, sbit, dst_bit as u32)?;
                    inner = Some(match inner {
                        Some(acc) => s.xor(acc, t)?,
                        None => t,
                    });
                }
            }
        }
        let inner = match inner {
            Some(v) => v,
            None => s.c(0),
        };
        let bx = s.div_c(u, bw)?;
        let by = s.div_c(v, bh)?;
        let bpr = s.arg(8)?;
        let row = s.mul(by, bpr)?;
        let block = s.add(row, bx)?;
        let block_off = s.mul_c(block, block_bytes as u32)?;
        let shift = s.arg(12)?;
        let layer = s.b.shift_right_logical(s.uint, None, slice, shift)?;
        let layer_bytes = s.arg(9)?;
        let layer_off = s.mul(layer, layer_bytes)?;
        let mip_off = s.arg(13)?;
        let base = s.add(mip_off, layer_off)?;
        let in_level = s.add(block_off, inner)?;
        let tiled = s.add(base, in_level)?;
        let row_bytes = s.arg(6)?;
        let slice_bytes = s.arg(7)?;
        let ly = s.mul(y, row_bytes)?;
        let lz = s.mul(z, slice_bytes)?;
        let lx = s.mul_c(x, bytes_per_element)?;
        let lyz = s.add(ly, lz)?;
        let linear = s.add(lyz, lx)?;
        let sbase = s.arg(0)?;
        let dbase = s.arg(1)?;
        let (src, dst) = if to_tiled {
            (s.add(sbase, linear)?, s.add(dbase, tiled)?)
        } else {
            (s.add(sbase, tiled)?, s.add(dbase, linear)?)
        };
        s.copy_element(src, dst, bytes_per_element)?;
        s.finish(main, 8, 8)
    };
    match build() {
        Ok(words) => Some(words),
        Err(e) => {
            log::error!("ps5gpu: tiler shader assembly failed (mode {mode}, bpe {bytes_per_element}): {e:?}");
            None
        }
    }
}

/// A rearrangement of the bits of each element between a guest format's
/// storage and the host format holding it: the 16-bit components B and R
/// swap (BGRA16 display buffers in an RGBA16 image), or the 10-bit field of
/// a 10_11_11 float element moves to the top so the element is a B10G11R11
/// host value (views then swap R and B).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ColorTransform {
    #[default]
    None,
    SwapBgra16,
    Reverse10_11_11,
}

impl ColorTransform {
    /// Bytes of one element the transform applies to (0 for none).
    pub fn element_bytes(self) -> u32 {
        match self {
            ColorTransform::None => 0,
            ColorTransform::SwapBgra16 => 8,
            ColorTransform::Reverse10_11_11 => 4,
        }
    }

    /// Transform one element (its little-endian words), from the guest
    /// storage to the host format when `to_host`, else back.
    pub fn apply(self, words: &mut [u32], to_host: bool) {
        match self {
            ColorTransform::None => {}
            ColorTransform::SwapBgra16 => {
                let (x, y) = (words[0], words[1]);
                words[0] = (y & 0xFFFF) | (x & 0xFFFF_0000);
                words[1] = (x & 0xFFFF) | (y & 0xFFFF_0000);
            }
            ColorTransform::Reverse10_11_11 => {
                let p = words[0];
                words[0] = if to_host {
                    (p >> 21) | ((p >> 10) & 0x7FF) << 11 | (p & 0x3FF) << 22
                } else {
                    (p >> 22) | ((p >> 11) & 0x7FF) << 10 | (p & 0x7FF) << 21
                };
            }
        }
    }
}

/// A box of elements a colour transform covers: `width` x `height` x
/// `slices`, rows and slices `*_row` / `*_slice` bytes apart on each side.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransformRegion {
    pub width: u32,
    pub height: u32,
    pub slices: u32,
    pub src_row: u32,
    pub src_slice: u32,
    pub dst_row: u32,
    pub dst_slice: u32,
}

impl TransformRegion {
    /// Tightly packed elements on both sides.
    pub fn packed(width: u32, height: u32, slices: u32, bytes_per_element: u32) -> TransformRegion {
        let row = width * bytes_per_element;
        TransformRegion { width, height, slices, src_row: row, src_slice: row * height, dst_row: row, dst_slice: row * height }
    }

    fn words(&self, src_base: u32, dst_base: u32) -> [u32; ARGUMENT_COUNT] {
        let mut w = [0u32; ARGUMENT_COUNT];
        w[..9].copy_from_slice(&[src_base, dst_base, self.width, self.height, self.slices, self.src_row, self.src_slice, self.dst_row, self.dst_slice]);
        w
    }
}

/// SPIR-V of the colour transform `t` (towards the host format when
/// `to_host`): one invocation per element of a [`TransformRegion`]
/// (arguments: source base, destination base, width, height, slices,
/// source row and slice bytes, destination row and slice bytes). The
/// source and the destination may be the same bytes.
pub fn transform_spirv(t: ColorTransform, to_host: bool) -> Option<Vec<u32>> {
    let bytes = t.element_bytes();
    if bytes == 0 {
        return None;
    }
    let build = || -> R<Vec<u32>> {
        let mut s = Spv::new();
        let (main, x, y, z) = s.begin_main()?;
        let (w, h, d) = (s.arg(2)?, s.arg(3)?, s.arg(4)?);
        let ox = s.b.u_greater_than_equal(s.boolean, None, x, w)?;
        let oy = s.b.u_greater_than_equal(s.boolean, None, y, h)?;
        let oz = s.b.u_greater_than_equal(s.boolean, None, z, d)?;
        let oxy = s.b.logical_or(s.boolean, None, ox, oy)?;
        let out = s.b.logical_or(s.boolean, None, oxy, oz)?;
        s.return_when(out)?;
        let element = s.mul_c(x, bytes)?;
        let address = |s: &mut Spv, base: u32, row: u32, slice: u32| -> R<Word> {
            let b = s.arg(base)?;
            let r = s.arg(row)?;
            let sl = s.arg(slice)?;
            let ry = s.mul(y, r)?;
            let sz = s.mul(z, sl)?;
            let a = s.add(b, ry)?;
            let a = s.add(a, sz)?;
            let a = s.add(a, element)?;
            s.shr(a, 2)
        };
        let si = address(&mut s, 0, 5, 6)?;
        let di = address(&mut s, 1, 7, 8)?;
        let or = |s: &mut Spv, a: Word, b: Word| s.b.bitwise_or(s.uint, None, a, b);
        let w0 = s.load_word(s.src, si)?;
        let one = s.c(1);
        match t {
            ColorTransform::SwapBgra16 => {
                let si1 = s.add(si, one)?;
                let w1 = s.load_word(s.src, si1)?;
                let lo1 = s.and_c(w1, 0xFFFF)?;
                let hi0 = s.and_c(w0, 0xFFFF_0000)?;
                let lo0 = s.and_c(w0, 0xFFFF)?;
                let hi1 = s.and_c(w1, 0xFFFF_0000)?;
                let x2 = or(&mut s, lo1, hi0)?;
                let y2 = or(&mut s, lo0, hi1)?;
                let p0 = s.word_ptr(s.dst, di)?;
                s.b.store(p0, x2, None, [])?;
                let di1 = s.add(di, one)?;
                let p1 = s.word_ptr(s.dst, di1)?;
                s.b.store(p1, y2, None, [])?;
            }
            ColorTransform::Reverse10_11_11 => {
                let (a_shift, b_shift, b_dst, c_mask, c_dst) = if to_host { (21, 10, 11, 0x3FF, 22) } else { (22, 11, 10, 0x7FF, 21) };
                let a = s.shr(w0, a_shift)?;
                let b = s.shr(w0, b_shift)?;
                let b = s.and_c(b, 0x7FF)?;
                let b = s.shl(b, b_dst)?;
                let c = s.and_c(w0, c_mask)?;
                let c = s.shl(c, c_dst)?;
                let ab = or(&mut s, a, b)?;
                let v = or(&mut s, ab, c)?;
                let p = s.word_ptr(s.dst, di)?;
                s.b.store(p, v, None, [])?;
            }
            ColorTransform::None => unreachable!(),
        }
        s.finish(main, 8, 8)
    };
    match build() {
        Ok(words) => Some(words),
        Err(e) => {
            log::error!("ps5gpu: colour transform shader assembly failed ({t:?}): {e:?}");
            None
        }
    }
}

/// Guest 16-bit unorm depth held in a 32-bit float host depth aspect (the
/// D16_UNORM_S8_UINT fallback): `to_float` widens `count` packed u16
/// elements to f32 (`v / 65535`), else narrows f32 elements back to packed
/// u16 (clamped, rounded to nearest). Arguments: source base, destination
/// base (bytes), count, row width in invocations (2D dispatch).
pub fn depth16_spirv(to_float: bool) -> Option<Vec<u32>> {
    let build = || -> R<Vec<u32>> {
        let mut s = Spv::new();
        let (main, x, y, _) = s.begin_main()?;
        let (count, width) = (s.arg(2)?, s.arg(3)?);
        let row = s.mul(y, width)?;
        let index = s.add(row, x)?;
        let f32t = s.b.type_float(32, None);
        let sb = s.arg(0)?;
        let sw = s.shr(sb, 2)?;
        let db = s.arg(1)?;
        let dw = s.shr(db, 2)?;
        if to_float {
            let out = s.b.u_greater_than_equal(s.boolean, None, index, count)?;
            s.return_when(out)?;
            let half = s.shr(index, 1)?;
            let wi = s.add(sw, half)?;
            let w = s.load_word(s.src, wi)?;
            let odd = s.and_c(index, 1)?;
            let shift = s.mul_c(odd, 16)?;
            let v = s.b.shift_right_logical(s.uint, None, w, shift)?;
            let v = s.and_c(v, 0xFFFF)?;
            let f = s.b.convert_u_to_f(f32t, None, v)?;
            let scale = s.b.constant_bit32(f32t, (1.0f32 / 65535.0).to_bits());
            let f = s.b.f_mul(f32t, None, f, scale)?;
            let bits = s.b.bitcast(s.uint, None, f)?;
            let di = s.add(dw, index)?;
            let p = s.word_ptr(s.dst, di)?;
            s.b.store(p, bits, None, [])?;
        } else {
            // One invocation per output word (two elements).
            let pairs_plus = s.add_c(count, 1)?;
            let pairs = s.shr(pairs_plus, 1)?;
            let out = s.b.u_greater_than_equal(s.boolean, None, index, pairs)?;
            s.return_when(out)?;
            let zero_f = s.b.constant_bit32(f32t, 0);
            let one_f = s.b.constant_bit32(f32t, 1.0f32.to_bits());
            let max = s.b.constant_bit32(f32t, 65535.0f32.to_bits());
            let half_f = s.b.constant_bit32(f32t, 0.5f32.to_bits());
            let zero = s.c(0);
            let mut packed = zero;
            for k in 0..2u32 {
                let twice = s.shl(index, 1)?;
                let kc = s.c(k);
                let i = s.add(twice, kc)?;
                let inside = s.b.u_less_than(s.boolean, None, i, count)?;
                let si = s.add(sw, i)?;
                let w = s.load_word(s.src, si)?;
                let f = s.b.bitcast(f32t, None, w)?;
                let lo = s.b.f_ord_less_than(s.boolean, None, f, zero_f)?;
                let f = s.b.select(f32t, None, lo, zero_f, f)?;
                let hi = s.b.f_ord_greater_than(s.boolean, None, f, one_f)?;
                let f = s.b.select(f32t, None, hi, one_f, f)?;
                let scaled = s.b.f_mul(f32t, None, f, max)?;
                let rounded = s.b.f_add(f32t, None, scaled, half_f)?;
                let u = s.b.convert_f_to_u(s.uint, None, rounded)?;
                let u = s.b.select(s.uint, None, inside, u, zero)?;
                let u = s.shl(u, k * 16)?;
                packed = s.b.bitwise_or(s.uint, None, packed, u)?;
            }
            let di = s.add(dw, index)?;
            let p = s.word_ptr(s.dst, di)?;
            s.b.store(p, packed, None, [])?;
        }
        s.finish(main, 64, 1)
    };
    match build() {
        Ok(words) => Some(words),
        Err(e) => {
            log::error!("ps5gpu: depth16 conversion shader assembly failed: {e:?}");
            None
        }
    }
}

/// Compute pipelines for the tiler shaders (push descriptors + push
/// constants; created on first use).
pub struct GpuTiler {
    dev: Arc<GpuDevice>,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipelines: HashMap<(u32, u32, bool, bool), vk::Pipeline>,
    transforms: HashMap<(ColorTransform, bool), vk::Pipeline>,
    depth16: [Option<vk::Pipeline>; 2],
}

/// A storage buffer range bound to one tiler binding.
#[derive(Clone, Copy, Debug)]
pub struct TilerBuffer {
    pub buffer: vk::Buffer,
    /// Byte offset of the data in `buffer`.
    pub offset: u64,
    /// Bytes from `offset` the shader may touch.
    pub size: u64,
}

impl TilerBuffer {
    /// The descriptor of the buffer with its offset aligned down to
    /// `align`, and the byte offset of the data within it.
    pub fn binding(self, align: u64) -> (vk::DescriptorBufferInfo, u32) {
        let base = self.offset / align * align;
        (vk::DescriptorBufferInfo { buffer: self.buffer, offset: base, range: (self.offset - base + self.size).div_ceil(4) * 4 }, (self.offset - base) as u32)
    }
}

impl GpuTiler {
    pub fn new(dev: &Arc<GpuDevice>) -> Result<GpuTiler, vk::Result> {
        let bindings = [0u32, 1].map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        // SAFETY: valid create infos; destroyed in Drop.
        unsafe {
            let set_layout = dev.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                    .bindings(&bindings),
                None,
            )?;
            let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: (ARGUMENT_COUNT * 4) as u32 }];
            let layout = dev.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set_layout)).push_constant_ranges(&ranges),
                None,
            )?;
            Ok(GpuTiler { dev: dev.clone(), set_layout, layout, pipelines: HashMap::new(), transforms: HashMap::new(), depth16: [None; 2] })
        }
    }

    fn pipeline(&mut self, mode: u32, bpe: u32, to_tiled: bool, thick: bool) -> Option<vk::Pipeline> {
        if let Some(&p) = self.pipelines.get(&(mode, bpe, to_tiled, thick)) {
            return Some(p);
        }
        let code = block_copy_spirv(mode, bpe, to_tiled, thick)?;
        let pipeline = self.create(&code, &format!("mode {mode}, bpe {bpe}"))?;
        self.pipelines.insert((mode, bpe, to_tiled, thick), pipeline);
        Some(pipeline)
    }

    fn transform_pipeline(&mut self, t: ColorTransform, to_host: bool) -> Option<vk::Pipeline> {
        if let Some(&p) = self.transforms.get(&(t, to_host)) {
            return Some(p);
        }
        let code = transform_spirv(t, to_host)?;
        let pipeline = self.create(&code, &format!("{t:?}"))?;
        self.transforms.insert((t, to_host), pipeline);
        Some(pipeline)
    }

    fn create(&self, code: &[u32], what: &str) -> Option<vk::Pipeline> {
        // SAFETY: valid SPIR-V words and create infos.
        unsafe {
            let module = self.dev.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None).ok()?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(self.layout);
            let r = self.dev.device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None);
            self.dev.device.destroy_shader_module(module, None);
            match r {
                Ok(p) => Some(p[0]),
                Err((_, e)) => {
                    log::error!("ps5gpu: tiler pipeline ({what}) failed: {e:?}");
                    None
                }
            }
        }
    }

    /// Record a (de)tile of a `width` x `height` x `layers` surface with
    /// `pitch` elements per tiled row (mip 0, layers one surface apart).
    /// `src`/`dst` are the tiled and the linear buffer (swapped when
    /// `to_tiled`). The linear side is tightly packed. Returns false when
    /// the combination is unsupported.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        cb: vk::CommandBuffer,
        mode: u32,
        bytes_per_element: u32,
        to_tiled: bool,
        src: TilerBuffer,
        dst: TilerBuffer,
        width: u32,
        height: u32,
        layers: u32,
        pitch: u32,
    ) -> bool {
        let Some(params) = crate::tiling::DetileParams::new(mode, bytes_per_element as usize, pitch.max(width)) else {
            return false;
        };
        if mode == crate::tiling::SWIZZLE_LINEAR {
            return false;
        }
        let layer_bytes = params.surface_size(height) as u32;
        let region = TileRegion::simple(width, height, layers, bytes_per_element, params.blocks_per_row, layer_bytes);
        self.record_region(cb, mode, bytes_per_element, false, to_tiled, src, dst, &region)
    }

    /// Record a (de)tile of one region (a mip level of some slices; see
    /// [`TileRegion`]), with the thick equation when `thick`.
    #[allow(clippy::too_many_arguments)]
    pub fn record_region(
        &mut self,
        cb: vk::CommandBuffer,
        mode: u32,
        bytes_per_element: u32,
        thick: bool,
        to_tiled: bool,
        src: TilerBuffer,
        dst: TilerBuffer,
        region: &TileRegion,
    ) -> bool {
        if mode == crate::tiling::SWIZZLE_LINEAR || region.width == 0 || region.height == 0 || region.slices == 0 {
            return false;
        }
        let Some(pipeline) = self.pipeline(mode, bytes_per_element, to_tiled, thick) else { return false };
        let groups = [region.width.div_ceil(8), region.height.div_ceil(8), region.slices];
        self.dispatch(cb, pipeline, src, dst, |sb, db| region.words(sb, db), groups);
        true
    }

    /// Record the colour transform `t` (towards the host format when
    /// `to_host`) of `region` from `src` into `dst` (which may be the same
    /// bytes). False when there is nothing to do or no pipeline.
    pub fn record_transform(&mut self, cb: vk::CommandBuffer, t: ColorTransform, to_host: bool, src: TilerBuffer, dst: TilerBuffer, region: &TransformRegion) -> bool {
        if t == ColorTransform::None || region.width == 0 || region.height == 0 || region.slices == 0 {
            return false;
        }
        let Some(pipeline) = self.transform_pipeline(t, to_host) else { return false };
        let groups = [region.width.div_ceil(8), region.height.div_ceil(8), region.slices];
        self.dispatch(cb, pipeline, src, dst, |sb, db| region.words(sb, db), groups);
        true
    }

    /// Record the conversion of `count` depth elements between packed
    /// 16-bit unorm (guest) and 32-bit float (host): widening when
    /// `to_float`. False without a pipeline.
    pub fn record_depth16(&mut self, cb: vk::CommandBuffer, to_float: bool, src: TilerBuffer, dst: TilerBuffer, count: u32) -> bool {
        if count == 0 {
            return true;
        }
        let slot = to_float as usize;
        let pipeline = match self.depth16[slot] {
            Some(p) => p,
            None => {
                let Some(code) = depth16_spirv(to_float) else { return false };
                let Some(p) = self.create(&code, "depth16") else { return false };
                self.depth16[slot] = Some(p);
                p
            }
        };
        let invocations = if to_float { count } else { count.div_ceil(2) };
        let groups = invocations.div_ceil(64);
        let gx = groups.min(65535);
        let gy = groups.div_ceil(gx);
        let width = gx * 64;
        self.dispatch(cb, pipeline, src, dst, |sb, db| {
            let mut w = [0u32; ARGUMENT_COUNT];
            w[..4].copy_from_slice(&[sb, db, count, width]);
            w
        }, [gx, gy, 1]);
        true
    }

    /// Bind `pipeline` with `src` / `dst` and the arguments `words(source
    /// base, destination base)`, and dispatch `groups`.
    fn dispatch(&self, cb: vk::CommandBuffer, pipeline: vk::Pipeline, src: TilerBuffer, dst: TilerBuffer, words: impl Fn(u32, u32) -> [u32; ARGUMENT_COUNT], groups: [u32; 3]) {
        let align = self.dev.caps.min_storage_buffer_offset_alignment;
        let (si, sbase) = src.binding(align);
        let (di, dbase) = dst.binding(align);
        let words = words(sbase, dbase);
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let infos = [si, di];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&infos[0])),
            vk::WriteDescriptorSet::default()
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&infos[1])),
        ];
        crate::scheduler::disturb_state();
        // SAFETY: recording command buffer; all handles valid.
        unsafe {
            let d = &self.dev.device;
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.dev.push_descriptor.cmd_push_descriptor_set(cb, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &writes);
            d.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
            d.cmd_dispatch(cb, groups[0], groups[1], groups[2]);
        }
    }

    /// Bytes of the tiled surface for a `pitch`-element-wide layout.
    pub fn tiled_size(mode: u32, bytes_per_element: u32, pitch: u32, height: u32) -> Option<u64> {
        crate::tiling::DetileParams::new(mode, bytes_per_element as usize, pitch).map(|p| p.surface_size(height) as u64)
    }
}

impl Drop for GpuTiler {
    fn drop(&mut self) {
        // SAFETY: owner waits for the device before dropping.
        unsafe {
            for (_, p) in self.pipelines.drain() {
                self.dev.device.destroy_pipeline(p, None);
            }
            for (_, p) in self.transforms.drain() {
                self.dev.device.destroy_pipeline(p, None);
            }
            for p in self.depth16.iter_mut().filter_map(|p| p.take()) {
                self.dev.device.destroy_pipeline(p, None);
            }
            self.dev.device.destroy_pipeline_layout(self.layout, None);
            self.dev.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Buffer, MemoryUse};
    use crate::scheduler::Scheduler;
    use crate::tiling::*;

    #[test]
    fn spirv_assembles_for_every_mode() {
        for mode in TILED_MODES {
            for bpe in [1u32, 2, 4, 8, 16] {
                for to_tiled in [false, true] {
                    let w = block_copy_spirv(mode, bpe, to_tiled, false).expect("assembles");
                    assert_eq!(w[0], rspirv::spirv::MAGIC_NUMBER);
                    // Round-trips through the parser.
                    let mut loader = rspirv::dr::Loader::new();
                    rspirv::binary::parse_words(&w, &mut loader).unwrap();
                }
            }
        }
        for mode in [SWIZZLE_4KB_S, SWIZZLE_4KB_S_X, SWIZZLE_64KB_S, SWIZZLE_64KB_S_X, SWIZZLE_64KB_D_X] {
            for bpe in [1u32, 2, 4, 8, 16] {
                assert!(has_thick_layout(mode));
                assert!(block_copy_spirv(mode, bpe, false, true).is_some());
            }
        }
        assert!(block_copy_spirv(SWIZZLE_64KB_R_X, 4, false, true).is_none());
        assert!(!has_thick_layout(SWIZZLE_64KB_Z_X) && !has_thick_layout(SWIZZLE_64KB_R_X));
        assert!(block_copy_spirv(SWIZZLE_LINEAR, 4, false, false).is_none());
    }

    #[test]
    fn colour_transforms_round_trip() {
        // A 10_11_11 float element: fields X (10 bits, low), Y, Z (11 bits).
        let (x, y, z) = (0x2A5u32, 0x5C3u32, 0x7F1u32);
        let guest = x | y << 10 | z << 21;
        let mut w = [guest];
        ColorTransform::Reverse10_11_11.apply(&mut w, true);
        // Host B10G11R11: R (11 bits, low) = Z, G = Y, B (10 bits, top) = X.
        assert_eq!(w[0], z | y << 11 | x << 22);
        ColorTransform::Reverse10_11_11.apply(&mut w, false);
        assert_eq!(w[0], guest);
        // BGRA16: components 0 and 2 swap, 1 and 3 stay.
        let (b, g, r, a) = (0x1111u32, 0x2222u32, 0x3333u32, 0x4444u32);
        let mut w = [b | g << 16, r | a << 16];
        ColorTransform::SwapBgra16.apply(&mut w, true);
        assert_eq!(w, [r | g << 16, b | a << 16]);
        ColorTransform::SwapBgra16.apply(&mut w, false);
        assert_eq!(w, [b | g << 16, r | a << 16]);
        assert_eq!(ColorTransform::None.element_bytes(), 0);
        for t in [ColorTransform::SwapBgra16, ColorTransform::Reverse10_11_11] {
            for to_host in [false, true] {
                let words = transform_spirv(t, to_host).expect("assembles");
                let mut loader = rspirv::dr::Loader::new();
                rspirv::binary::parse_words(&words, &mut loader).unwrap();
            }
        }
        assert!(transform_spirv(ColorTransform::None, true).is_none());
        let r = TransformRegion::packed(5, 3, 2, 8);
        assert_eq!((r.src_row, r.src_slice, r.dst_row, r.dst_slice), (40, 120, 40, 120));
    }

    fn synth(len: usize, seed: u64) -> Vec<u8> {

        (0..len).map(|i| ((i as u64 + seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 37) as u8).collect()
    }

    /// GPU detile/tile against the CPU implementation on the real device.
    #[test]
    #[ignore]
    fn gpu_tiling_matches_cpu() {
        let dev = GpuDevice::create(false).expect("device");
        let mut sched = Scheduler::new(&dev).unwrap();
        let mut tiler = GpuTiler::new(&dev).unwrap();
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
        let mut checked = 0;
        for mode in TILED_MODES {
            for bpe in [1u32, 2, 4, 8, 16] {
                for (w, h) in [(37u32, 19u32), (300, 170), (1920 / 4, 1080 / 4)] {
                    let p = DetileParams::new(mode, bpe as usize, w).unwrap();
                    let tiled_len = p.surface_size(h);
                    let lin_len = (w * h * bpe) as usize;
                    let tiled = synth(tiled_len, mode as u64 * 31 + bpe as u64);
                    let mut cpu_lin = vec![0u8; lin_len];
                    assert!(detile(&p, &tiled, &mut cpu_lin, w, h));
                    // Upload the tiled data at an unaligned offset to exercise bases.
                    let off = 4 * 3;
                    let src = Buffer::new(&dev, (tiled_len + off) as u64, usage, MemoryUse::Upload).unwrap();
                    src.bytes_mut()[off..off + tiled_len].copy_from_slice(&tiled);
                    let dst = Buffer::new(&dev, lin_len as u64 + 64, usage, MemoryUse::Download).unwrap();
                    dst.bytes_mut().fill(0);
                    let cb = sched.cmd();
                    assert!(tiler.record(
                        cb,
                        mode,
                        bpe,
                        false,
                        TilerBuffer { buffer: src.buffer, offset: off as u64, size: tiled_len as u64 },
                        TilerBuffer { buffer: dst.buffer, offset: 0, size: lin_len as u64 },
                        w,
                        h,
                        1,
                        w,
                    ));
                    sched.flush_and_wait();
                    dst.invalidate(0, dst.size);
                    assert!(dst.bytes_mut()[..lin_len] == cpu_lin[..], "detile mode={mode} bpe={bpe} {w}x{h}");

                    // Tile it back on the GPU and compare with the CPU tiler.
                    let mut cpu_tiled = vec![0u8; tiled_len];
                    assert!(tile(&p, &cpu_lin, &mut cpu_tiled, w, h));
                    let back = Buffer::new(&dev, tiled_len as u64, usage, MemoryUse::Download).unwrap();
                    back.bytes_mut().fill(0);
                    let cb = sched.cmd();
                    assert!(tiler.record(
                        cb,
                        mode,
                        bpe,
                        true,
                        TilerBuffer { buffer: dst.buffer, offset: 0, size: lin_len as u64 },
                        TilerBuffer { buffer: back.buffer, offset: 0, size: tiled_len as u64 },
                        w,
                        h,
                        1,
                        w,
                    ));
                    sched.flush_and_wait();
                    back.invalidate(0, back.size);
                    assert!(back.bytes_mut()[..tiled_len] == cpu_tiled[..], "tile mode={mode} bpe={bpe} {w}x{h}");
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, TILED_MODES.len() * 15);
        drop(tiler);
        dev.wait_idle();
    }
}
