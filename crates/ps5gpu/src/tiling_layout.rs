// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! GFX10 surface layout: where every mip level, array layer and depth
//! slice of a guest surface lives (addrlib's mip chain rules).
//!
//! * Thin (2D) tiled surfaces: each array layer holds the whole mip chain.
//!   Levels are stored from the smallest to the largest; the levels small
//!   enough for the *mip tail* share the first block of the layer (each at
//!   a fixed element position inside it), then the other levels follow in
//!   decreasing level order, level 0 last. Each level is padded to whole
//!   blocks (`blocks_per_row` x rows).
//! * Linear surfaces: levels from the smallest to level 0, rows padded to
//!   256 bytes.
//! * Volumes (3D) of `SW_4KB_S` / `SW_64KB_S` / `SW_64KB_S_X` /
//!   `SW_64KB_D_X` are *thick*: blocks span `block_depth` slices and each
//!   slab of `block_depth` slices holds the whole mip chain; slice z of a
//!   level lives in slab `z / block_depth` and the slice index selects the
//!   depth bits inside the block. Volumes of `_Z_X` and `_R_X` are stored
//!   as thin slices, one mip chain per slice.
//!
//! Sizes are in *elements*: texels, or 4x4 blocks of block-compressed
//! formats (`texel_scale` = 4).

use crate::tiling::{self, DetileParams, SWIZZLE_LINEAR};

/// One mip level inside a layer (thin) or slab (thick).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MipLayout {
    /// Byte offset of the level from the start of its layer / slab.
    pub offset: u64,
    /// Tiled bytes of the level (the shared tail block for tail levels).
    pub size: u64,
    /// Extent of the level in elements.
    pub width: u32,
    pub height: u32,
    /// Blocks per tiled row (elements per row when linear).
    pub blocks_per_row: u32,
    /// Row pitch in elements.
    pub pitch: u32,
    /// The level lives in the mip tail block, at element (`tail_x`, `tail_y`).
    pub tail: bool,
    pub tail_x: u32,
    pub tail_y: u32,
}

/// The layout of a whole surface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SurfaceLayout {
    pub mode: u32,
    pub bytes_per_element: u32,
    pub mips: Vec<MipLayout>,
    /// Array layers, or depth slices of a volume.
    pub slices: u32,
    pub volume: bool,
    /// Thick blocks (a volume of `SW_4KB_S` / `SW_64KB_S` / `SW_64KB_S_X` /
    /// `SW_64KB_D_X`).
    pub thick: bool,
    /// Slices per block (1 unless thick).
    pub block_depth: u32,
    /// Bytes between consecutive layers (thin) or slabs (thick).
    pub layer_bytes: u64,
    /// Bytes of the whole surface.
    pub guest_bytes: u64,
}

impl SurfaceLayout {
    /// Byte offset of the layer / slab holding `slice`.
    pub fn slice_offset(&self, slice: u32) -> u64 {
        if self.thick {
            (slice / self.block_depth) as u64 * self.layer_bytes
        } else {
            slice as u64 * self.layer_bytes
        }
    }

    /// Slices level `level` has (volumes shrink with the level).
    pub fn level_slices(&self, level: u32) -> u32 {
        if self.volume {
            (self.slices >> level).max(1)
        } else {
            self.slices
        }
    }

    /// The addressing of the swizzle blocks (thin or thick), None for
    /// linear surfaces.
    pub fn params(&self) -> Option<DetileParams> {
        if self.mode == SWIZZLE_LINEAR {
            return None;
        }
        let bpe = self.bytes_per_element as usize;
        if self.thick {
            DetileParams::new_thick(self.mode, bpe, 1)
        } else {
            DetileParams::new(self.mode, bpe, 1)
        }
    }

    /// Byte offset of element (x, y) of `slice` of `level` from the start
    /// of the surface (the reference the GPU tiler follows).
    pub fn element_offset(&self, level: u32, slice: u32, x: u32, y: u32) -> Option<u64> {
        let m = self.mips.get(level as usize)?;
        let base = self.slice_offset(slice) + m.offset;
        let Some(p) = self.params() else {
            return Some(base + (y as u64 * m.pitch as u64 + x as u64) * self.bytes_per_element as u64);
        };
        let (sx, sy) = (x + m.tail_x, y + m.tail_y);
        let block = (sy / p.block_height) as u64 * m.blocks_per_row as u64 + (sx / p.block_width) as u64;
        let inner = p.in_block(sx, sy, slice) as u64;
        Some(base + block * p.block_bytes as u64 + inner)
    }
}

fn shift_ceil(v: u32, s: u32) -> u32 {
    ((v as u64 + (1u64 << s) - 1) >> s) as u32
}

/// Elements of a level: `ceil(max(texels >> level, 1) / scale)`.
fn level_elements(texels: u32, level: u32, scale: u32) -> u32 {
    (texels >> level).max(1).div_ceil(scale).max(1)
}

/// Mip tail element positions of thin 4 KiB / 64 KiB blocks, per log2
/// element size, from the last tail level backwards.
const TAIL_4K: [[(u32, u32); 8]; 5] = [
    [(32, 0), (16, 32), (0, 48), (0, 32), (16, 16), (16, 0), (0, 16), (0, 0)],
    [(32, 0), (16, 16), (0, 24), (0, 16), (16, 8), (16, 0), (0, 8), (0, 0)],
    [(16, 0), (8, 16), (0, 24), (0, 16), (8, 8), (8, 0), (0, 8), (0, 0)],
    [(16, 0), (8, 8), (0, 12), (0, 8), (8, 4), (8, 0), (0, 4), (0, 0)],
    [(8, 0), (4, 8), (0, 12), (0, 8), (4, 4), (4, 0), (0, 4), (0, 0)],
];

const TAIL_64K: [[(u32, u32); 12]; 5] = [
    [(128, 0), (0, 128), (64, 0), (0, 64), (32, 0), (16, 32), (0, 48), (0, 32), (16, 16), (16, 0), (0, 16), (0, 0)],
    [(128, 0), (0, 64), (64, 0), (0, 32), (32, 0), (16, 16), (0, 24), (0, 16), (16, 8), (16, 0), (0, 8), (0, 0)],
    [(64, 0), (0, 64), (32, 0), (0, 32), (16, 0), (8, 16), (0, 24), (0, 16), (8, 8), (8, 0), (0, 8), (0, 0)],
    [(64, 0), (0, 32), (32, 0), (0, 16), (16, 0), (8, 8), (0, 12), (0, 8), (8, 4), (8, 0), (0, 4), (0, 0)],
    [(32, 0), (0, 32), (16, 0), (0, 16), (8, 0), (4, 8), (0, 12), (0, 8), (4, 4), (4, 0), (0, 4), (0, 0)],
];

/// Linear mip chain (levels from the smallest, rows padded to 256 bytes).
fn linear_mips(bpe: u32, scale: u32, width: u32, height: u32, levels: u32) -> Vec<MipLayout> {
    let (ew0, eh0) = (width.div_ceil(scale), height.div_ceil(scale));
    let align = (256 / bpe).max(1);
    let mut mips = vec![MipLayout::default(); levels as usize];
    let mut offset = 0u64;
    for level in (0..levels).rev() {
        let ew = shift_ceil(ew0, level).max(1);
        let eh = shift_ceil(eh0, level).max(1);
        let padded = ew.next_multiple_of(align);
        let size = padded as u64 * eh as u64 * bpe as u64;
        mips[level as usize] = MipLayout {
            offset,
            size,
            width: level_elements(width, level, scale),
            height: level_elements(height, level, scale),
            blocks_per_row: padded,
            pitch: padded,
            tail: false,
            tail_x: 0,
            tail_y: 0,
        };
        offset += size;
    }
    mips
}

/// Thin tiled mip chain.
fn thin_mips(mode: u32, bpe: u32, scale: u32, width: u32, height: u32, levels: u32) -> Option<Vec<MipLayout>> {
    // The block shape only (building the address tables is far costlier).
    let (bw, bh, bb) = tiling::thin_block_extent(mode, bpe as usize)?;
    let bb = bb as u64;
    let (ew0, eh0) = (width.div_ceil(scale), height.div_ceil(scale));
    let index = bpe.trailing_zeros() as usize;
    // Tail: half a block wide; none for 256-byte blocks.
    let tail_table: Option<&[(u32, u32)]> = match bb {
        4096 => Some(&TAIL_4K[index]),
        65536 => Some(&TAIL_64K[index]),
        _ => None,
    };
    // SW_64KB_Z_X keeps 1- and 2-byte elements out of the tail one level
    // longer: its limit is half as wide (and, for 1 byte, half as high).
    let (mut tail_w, mut tail_h) = (bw >> 1, bh);
    if mode == tiling::SWIZZLE_64KB_Z_X && bpe <= 2 {
        tail_w >>= 1;
        if bpe == 1 {
            tail_h >>= 1;
        }
    }
    let mut first_tail = levels;
    if let Some(t) = tail_table {
        if levels > 1 {
            for level in 0..levels {
                if shift_ceil(ew0, level) <= tail_w && shift_ceil(eh0, level) <= tail_h && levels - level <= t.len() as u32 {
                    first_tail = level;
                    break;
                }
            }
        }
    }
    let mut mips = vec![MipLayout::default(); levels as usize];
    for level in 0..levels {
        let m = &mut mips[level as usize];
        m.width = level_elements(width, level, scale);
        m.height = level_elements(height, level, scale);
        if level < first_tail {
            let pw = shift_ceil(ew0, level).max(1).next_multiple_of(bw);
            let ph = shift_ceil(eh0, level).max(1).next_multiple_of(bh);
            m.blocks_per_row = pw / bw;
            m.pitch = pw;
            m.size = pw as u64 * ph as u64 * bpe as u64;
        } else {
            let (tx, ty) = tail_table.unwrap()[(level - first_tail) as usize];
            m.blocks_per_row = 1;
            m.pitch = bw;
            m.size = bb;
            m.tail = true;
            m.tail_x = tx;
            m.tail_y = ty;
        }
    }
    let mut offset = if first_tail < levels { bb } else { 0 };
    for level in (0..first_tail).rev() {
        mips[level as usize].offset = offset;
        offset += mips[level as usize].size;
    }
    Some(mips)
}

/// Thick (or linear) volume: the mip chain of one slab and its block depth.
fn thick_mips(mode: u32, bpe: u32, width: u32, height: u32, levels: u32) -> Option<(Vec<MipLayout>, u32, u64)> {
    let linear = mode == SWIZZLE_LINEAR;
    let (block, block_bytes) = if linear {
        ([(256 / bpe).max(1), 1, 1], 0u64)
    } else {
        (tiling::thick_block_extent(mode, bpe)?, if matches!(mode, tiling::SWIZZLE_4KB_S | tiling::SWIZZLE_4KB_S_X) { 4096 } else { 65536 })
    };
    let mut max_tail = 0u32;
    let mut first_tail = levels;
    if !linear {
        let log2 = block_bytes.trailing_zeros();
        let tail_w = if log2 % 3 == 1 { block[0] >> 1 } else { block[0] };
        let tail_h = if log2 % 3 == 0 { block[1] >> 1 } else { block[1] };
        let effective = log2 - (log2 - 8) / 3;
        max_tail = if effective <= 11 { 1 + (1 << (effective - 9)) } else { effective - 4 };
        if levels > 1 {
            for level in 0..levels {
                if shift_ceil(width, level) <= tail_w && shift_ceil(height, level) <= tail_h && levels - level <= max_tail {
                    first_tail = level;
                    break;
                }
            }
        }
    }
    const MICRO: [[u32; 2]; 5] = [[8, 4], [4, 4], [4, 4], [4, 2], [2, 2]];
    let element = bpe.trailing_zeros() as usize;
    let mut mips = vec![MipLayout::default(); levels as usize];
    let mut offset = if first_tail < levels { block_bytes } else { 0 };
    for level in (0..levels).rev() {
        let m = &mut mips[level as usize];
        m.width = (width >> level).max(1);
        m.height = (height >> level).max(1);
        let lw = if linear { m.width } else { shift_ceil(width, level).max(1) };
        let lh = if linear { m.height } else { shift_ceil(height, level).max(1) };
        let pw = lw.next_multiple_of(block[0]);
        m.pitch = pw;
        if level >= first_tail {
            let slot = max_tail - 1 - (level - first_tail);
            let slot_offset = if slot > 6 { 16u32 << slot } else { slot << 8 };
            let (mut tx, mut ty) = (0u32, 0u32);
            for bit in 0..6 {
                tx |= ((slot_offset >> (9 + 2 * bit)) & 1) << bit;
                ty |= ((slot_offset >> (8 + 2 * bit)) & 1) << bit;
            }
            m.tail = true;
            m.tail_x = tx * MICRO[element][0];
            m.tail_y = ty * MICRO[element][1];
            m.blocks_per_row = 1;
            m.pitch = block[0];
            m.offset = 0;
            m.size = block_bytes;
        } else {
            if linear {
                m.blocks_per_row = pw;
                m.size = pw as u64 * m.height as u64 * bpe as u64;
            } else {
                m.blocks_per_row = pw / block[0];
                m.size = m.blocks_per_row as u64 * (lh.next_multiple_of(block[1]) / block[1]) as u64 * block_bytes;
            }
            m.offset = offset;
            offset += m.size;
        }
    }
    Some((mips, block[2], offset))
}

/// The layout of a surface of `mode` with `bytes_per_element`-byte
/// elements of `texel_scale` x `texel_scale` texels: `width` x `height`
/// texels at level 0, `levels` mip levels and `slices` array layers (or
/// depth slices when `volume`). None for unsupported combinations
/// (unknown modes, block-compressed thick volumes, volumes of modes with
/// neither a thick nor a thin-slice layout).
pub fn describe(mode: u32, bytes_per_element: u32, texel_scale: u32, width: u32, height: u32, slices: u32, levels: u32, volume: bool) -> Option<SurfaceLayout> {
    let bpe = bytes_per_element;
    if width == 0 || height == 0 || slices == 0 || levels == 0 || levels > 16 || !bpe.is_power_of_two() || bpe > 16 {
        return None;
    }
    let thin_volume = volume && matches!(mode, tiling::SWIZZLE_64KB_Z_X | tiling::SWIZZLE_64KB_R_X);
    if volume && !thin_volume {
        if texel_scale != 1 {
            return None;
        }
        let (mips, block_depth, slab) = thick_mips(mode, bpe, width, height, levels)?;
        let thick = mode != SWIZZLE_LINEAR;
        let guest_bytes = slab * slices.div_ceil(block_depth) as u64;
        return Some(SurfaceLayout { mode, bytes_per_element: bpe, mips, slices, volume, thick, block_depth, layer_bytes: slab, guest_bytes });
    }
    let mips = if mode == SWIZZLE_LINEAR { linear_mips(bpe, texel_scale, width, height, levels) } else { thin_mips(mode, bpe, texel_scale, width, height, levels)? };
    let layer_bytes = mips.iter().map(|m| m.offset + m.size).max().unwrap_or(0);
    Some(SurfaceLayout {
        mode,
        bytes_per_element: bpe,
        mips,
        slices,
        volume,
        thick: false,
        block_depth: 1,
        layer_bytes,
        guest_bytes: layer_bytes * slices as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Element addresses computed by addrlib (GFX10, 16 pipes) for volumes,
    /// mip chains and arrays of every swizzle family.
    /// (volume, mode, bytes per element, width, height, slices, levels, first tail level, guest bytes, layer bytes, [(level, x, y, slice, address)], what)
    #[rustfmt::skip]
    const ADDRLIB_VECTORS: &[(bool, u32, u32, u32, u32, u32, u32, Option<u32>, u64, Option<u64>, &[(u32, u32, u32, u32, u64)], &str)] = &[
        (true, 21, 1, 33, 20, 12, 6, Some(2), 36864, None, &[(0, 0, 0, 0, 0x3000), (0, 32, 19, 11, 0x843c), (0, 8, 9, 8, 0x3e08), (1, 0, 0, 0, 0x1000), (1, 15, 9, 5, 0x1acf), (1, 12, 4, 0, 0x1340), (2, 0, 0, 0, 0x800), (2, 7, 4, 2, 0x953), (2, 2, 2, 0, 0x822), (3, 0, 0, 0, 0x300), (3, 3, 1, 0, 0x30b), (3, 3, 1, 0, 0x30b), (4, 0, 0, 0, 0x200), (4, 1, 0, 0, 0x201), (4, 0, 0, 0, 0x200), (5, 0, 0, 0, 0x100), (5, 0, 0, 0, 0x100), (5, 0, 0, 0, 0x100)], "SW_4KB_S_X 8 bpp 33x20x12, 6 levels"),
        (true, 21, 2, 33, 20, 12, 6, Some(3), 65536, None, &[(0, 0, 0, 0, 0x6000), (0, 32, 19, 11, 0xf13c), (0, 8, 9, 8, 0x7408), (1, 0, 0, 0, 0x3000), (1, 15, 9, 5, 0x42ce), (1, 12, 4, 0, 0x4b00), (2, 0, 0, 0, 0x1000), (2, 7, 4, 2, 0x1352), (2, 2, 2, 0, 0x1060), (3, 0, 0, 0, 0x800), (3, 3, 1, 0, 0x84a), (3, 3, 1, 0, 0x84a), (4, 0, 0, 0, 0x300), (4, 1, 0, 0, 0x302), (4, 0, 0, 0, 0x300), (5, 0, 0, 0, 0x200), (5, 0, 0, 0, 0x200), (5, 0, 0, 0, 0x200)], "SW_4KB_S_X 16 bpp 33x20x12, 6 levels"),
        (true, 21, 4, 128, 128, 8, 2, Some(2), 655360, None, &[(0, 0, 0, 0, 0x20000), (0, 127, 127, 7, 0x9f4fc), (0, 24, 93, 0, 0x73108), (1, 0, 0, 0, 0x0), (1, 63, 63, 3, 0x1f0fc), (1, 28, 38, 2, 0x13da0)], "SW_4KB_S_X 32 bpp 128x128x8, 2 levels"),
        (true, 21, 4, 33, 20, 12, 6, Some(3), 131072, None, &[(0, 0, 0, 0, 0x6000), (0, 32, 19, 11, 0x1fdb8), (0, 8, 9, 8, 0x17808), (1, 0, 0, 0, 0x3000), (1, 15, 9, 5, 0x465c), (1, 12, 4, 0, 0x4b00), (2, 0, 0, 0, 0x1000), (2, 7, 4, 2, 0x13c4), (2, 2, 2, 0, 0x1060), (3, 0, 0, 0, 0x800), (3, 3, 1, 0, 0x84c), (3, 3, 1, 0, 0x84c), (4, 0, 0, 0, 0x300), (4, 1, 0, 0, 0x304), (4, 0, 0, 0, 0x300), (5, 0, 0, 0, 0x200), (5, 0, 0, 0, 0x200), (5, 0, 0, 0, 0x200)], "SW_4KB_S_X 32 bpp 33x20x12, 6 levels"),
        (true, 21, 8, 33, 20, 12, 6, Some(3), 196608, None, &[(0, 0, 0, 0, 0x9000), (0, 32, 19, 11, 0x2fab0), (0, 8, 9, 8, 0x27420), (1, 0, 0, 0, 0x3000), (1, 15, 9, 5, 0x7a78), (1, 12, 4, 0, 0x4200), (2, 0, 0, 0, 0x1000), (2, 7, 4, 2, 0x1ac8), (2, 2, 2, 0, 0x1140), (3, 0, 0, 0, 0x800), (3, 3, 1, 0, 0x868), (3, 3, 1, 0, 0x868), (4, 0, 0, 0, 0x300), (4, 1, 0, 0, 0x308), (4, 0, 0, 0, 0x300), (5, 0, 0, 0, 0x200), (5, 0, 0, 0, 0x200), (5, 0, 0, 0, 0x200)], "SW_4KB_S_X 64 bpp 33x20x12, 6 levels"),
        (true, 21, 16, 33, 20, 12, 6, Some(4), 352256, None, &[(0, 0, 0, 0, 0x10000), (0, 32, 19, 11, 0x55bb0), (0, 8, 9, 8, 0x46820), (1, 0, 0, 0, 0x6000), (1, 15, 9, 5, 0xee70), (1, 12, 4, 0, 0x9400), (2, 0, 0, 0, 0x3000), (2, 7, 4, 2, 0x42c0), (2, 2, 2, 0, 0x3300), (3, 0, 0, 0, 0x1000), (3, 3, 1, 0, 0x1260), (3, 3, 1, 0, 0x1260), (4, 0, 0, 0, 0x800), (4, 1, 0, 0, 0x840), (4, 0, 0, 0, 0x800), (5, 0, 0, 0, 0x300), (5, 0, 0, 0, 0x300), (5, 0, 0, 0, 0x300)], "SW_4KB_S_X 128 bpp 33x20x12, 6 levels"),
        (true, 5, 4, 8, 8, 8, 2, Some(0), 4096, None, &[(0, 0, 0, 0, 0x800), (0, 7, 7, 7, 0xffc), (0, 4, 2, 4, 0xe20), (0, 2, 7, 0, 0x968), (0, 7, 0, 7, 0xed4), (1, 0, 0, 0, 0x300), (1, 3, 3, 3, 0x3fc), (1, 2, 1, 2, 0x3c8), (1, 1, 3, 0, 0x32c), (1, 3, 0, 3, 0x3d4)], "SW_4KB_S 32 bpp 8x8x8, 2 levels"),
        (true, 5, 4, 33, 20, 12, 6, Some(3), 131072, None, &[(0, 0, 0, 0, 0x6000), (0, 32, 19, 11, 0x1f0b8), (0, 16, 6, 6, 0x85a0), (0, 11, 19, 0, 0xc06c), (0, 32, 0, 11, 0x1a090), (1, 0, 0, 0, 0x3000), (1, 15, 9, 5, 0x4e5c), (1, 8, 3, 3, 0x40b8), (1, 5, 9, 0, 0x3a0c), (1, 15, 0, 5, 0x4654), (2, 0, 0, 0, 0x1000), (2, 7, 4, 2, 0x13c4), (2, 4, 1, 1, 0x1218), (2, 2, 4, 0, 0x1140), (2, 7, 0, 2, 0x12c4), (3, 0, 0, 0, 0x800), (3, 3, 1, 0, 0x84c), (3, 2, 0, 0, 0x840), (3, 1, 1, 0, 0x80c), (3, 3, 0, 0, 0x844), (4, 0, 0, 0, 0x300), (4, 1, 0, 0, 0x304), (4, 0, 0, 0, 0x300), (5, 0, 0, 0, 0x200)], "SW_4KB_S 32 bpp 33x20x12, 6 levels"),
        (true, 9, 8, 40, 24, 20, 6, Some(2), 786432, None, &[(0, 0, 0, 0, 0x20000), (0, 39, 23, 19, 0xb0bf8), (0, 20, 8, 10, 0x2e280), (0, 13, 23, 0, 0x41b28), (0, 39, 0, 19, 0x902d8), (1, 0, 0, 0, 0x10000), (1, 19, 11, 9, 0x1e178), (1, 10, 4, 5, 0x11c50), (1, 6, 11, 0, 0x14360), (1, 19, 0, 9, 0x1a058), (2, 0, 0, 0, 0x8000), (2, 9, 5, 4, 0x9c28), (2, 5, 2, 2, 0x8388), (2, 3, 5, 0, 0x8868), (2, 9, 0, 4, 0x9408), (3, 0, 0, 0, 0x4000), (3, 4, 2, 1, 0x4310), (3, 2, 1, 1, 0x4070), (3, 1, 2, 0, 0x4108), (3, 4, 0, 1, 0x4210), (4, 0, 0, 0, 0x1000), (4, 1, 0, 0, 0x1008), (5, 0, 0, 0, 0xa00)], "SW_64KB_S 64 bpp 40x24x20, 6 levels"),
        (true, 9, 1, 100, 60, 70, 7, Some(2), 1179648, None, &[(0, 0, 0, 0, 0x20000), (0, 99, 59, 69, 0x11c8af), (0, 50, 20, 35, 0x8d116), (0, 33, 59, 0, 0x4c829), (0, 99, 0, 69, 0xf8087), (1, 0, 0, 0, 0x10000), (1, 49, 29, 34, 0x7d919), (1, 25, 10, 17, 0x13a25), (1, 16, 29, 0, 0x15908), (1, 49, 0, 34, 0x79011), (2, 0, 0, 0, 0x8000), (2, 24, 14, 16, 0xbb20), (2, 12, 5, 8, 0x8748), (2, 8, 14, 0, 0x8b20), (2, 24, 0, 16, 0xb200), (3, 0, 0, 0, 0x4000), (3, 11, 6, 7, 0x43b7), (3, 6, 2, 4, 0x40e2), (3, 4, 6, 0, 0x4160), (3, 11, 0, 7, 0x4297), (4, 0, 0, 0, 0x1000), (4, 5, 2, 3, 0x1075), (4, 3, 1, 2, 0x101b), (4, 2, 2, 0, 0x1022), (4, 5, 0, 3, 0x1055), (5, 0, 0, 0, 0xa00), (5, 2, 0, 1, 0xa06), (5, 1, 0, 0, 0xa01), (6, 0, 0, 0, 0x900)], "SW_64KB_S 8 bpp 100x60x70, 7 levels"),
        (true, 5, 16, 9, 5, 3, 4, Some(2), 24576, None, &[(0, 0, 0, 0, 0x3000), (0, 8, 4, 2, 0x5880), (0, 4, 1, 1, 0x4030), (0, 3, 4, 0, 0x3a40), (0, 8, 0, 2, 0x5080), (1, 0, 0, 0, 0x1000), (1, 3, 1, 0, 0x1260), (1, 2, 0, 0, 0x1200), (1, 1, 1, 0, 0x1060), (1, 3, 0, 0, 0x1240), (2, 0, 0, 0, 0x800), (2, 1, 0, 0, 0x840), (3, 0, 0, 0, 0x300)], "SW_4KB_S 128 bpp 9x5x3, 4 levels"),
        (true, 25, 1, 100, 60, 70, 7, Some(2), 1179648, None, &[(0, 0, 0, 0, 0x20000), (0, 99, 59, 69, 0x11caaf), (0, 50, 20, 35, 0x8db16), (0, 33, 59, 0, 0x4ca29), (0, 99, 0, 69, 0xf8487), (1, 0, 0, 0, 0x10000), (1, 49, 29, 34, 0x7d319), (1, 25, 10, 17, 0x13a25), (1, 16, 29, 0, 0x15508), (1, 49, 0, 34, 0x79e11), (2, 0, 0, 0, 0x8400), (2, 24, 14, 16, 0xbf20), (2, 12, 5, 8, 0x8348), (2, 8, 14, 0, 0x8f20), (2, 24, 0, 16, 0xb600), (3, 0, 0, 0, 0x4400), (3, 11, 6, 7, 0x47b7), (3, 6, 2, 4, 0x44e2), (3, 4, 6, 0, 0x4560), (3, 11, 0, 7, 0x4697), (4, 0, 0, 0, 0x1800), (4, 5, 2, 3, 0x1875), (4, 3, 1, 2, 0x181b), (4, 2, 2, 0, 0x1822), (4, 5, 0, 3, 0x1855), (5, 0, 0, 0, 0xa00), (5, 2, 0, 1, 0xa06), (5, 1, 0, 1, 0xa05), (5, 1, 0, 0, 0xa01), (5, 2, 0, 1, 0xa06), (6, 0, 0, 0, 0x900), (6, 0, 0, 0, 0x900), (6, 0, 0, 0, 0x900), (6, 0, 0, 0, 0x900), (6, 0, 0, 0, 0x900)], "SW_64KB_S_X 8 bpp 100x60x70, 7 levels"),
        (true, 25, 16, 9, 5, 3, 4, Some(1), 131072, None, &[(0, 0, 0, 0, 0x10000), (0, 8, 4, 2, 0x18c80), (0, 4, 1, 1, 0x11830), (0, 3, 4, 0, 0x10a40), (0, 8, 0, 2, 0x18480), (1, 0, 0, 0, 0x8400), (1, 3, 1, 0, 0x8660), (1, 2, 0, 0, 0x8600), (1, 1, 1, 0, 0x8460), (1, 3, 0, 0, 0x8640), (2, 0, 0, 0, 0x4400), (2, 1, 0, 0, 0x4440), (2, 1, 0, 0, 0x4440), (2, 0, 0, 0, 0x4400), (2, 1, 0, 0, 0x4440), (3, 0, 0, 0, 0x1800), (3, 0, 0, 0, 0x1800), (3, 0, 0, 0, 0x1800), (3, 0, 0, 0, 0x1800), (3, 0, 0, 0, 0x1800)], "SW_64KB_S_X 128 bpp 9x5x3, 4 levels"),
        (true, 24, 4, 40, 24, 20, 4, None, 1310720, None, &[(0, 0, 0, 0, 0x8400), (0, 39, 23, 19, 0x1382fc), (0, 20, 8, 10, 0xab240), (0, 13, 23, 0, 0x87ec), (0, 39, 0, 19, 0x138054), (1, 0, 0, 0, 0x4800), (1, 19, 11, 9, 0x9723c), (1, 10, 4, 5, 0x54390), (1, 6, 11, 0, 0x5978), (1, 19, 0, 9, 0x96314), (2, 0, 0, 0, 0x800), (2, 9, 5, 4, 0x40b8c), (2, 5, 2, 2, 0x20c64), (2, 3, 5, 0, 0x89c), (2, 9, 0, 4, 0x40b04), (3, 0, 0, 0, 0x400), (3, 4, 2, 1, 0x10c60), (3, 2, 1, 1, 0x10c18), (3, 1, 2, 0, 0x424), (3, 4, 0, 1, 0x10c40)], "thin SW_64KB_Z_X 32 bpp 40x24x20, 4 levels"),
        (true, 24, 8, 33, 20, 12, 6, None, 786432, None, &[(0, 0, 0, 0, 0x8400), (0, 32, 19, 11, 0xbc350), (0, 16, 6, 6, 0x69040), (0, 11, 19, 0, 0xe778), (0, 32, 0, 11, 0xb8100), (1, 0, 0, 0, 0x400), (1, 15, 9, 5, 0x52eb8), (1, 8, 3, 3, 0x32950), (1, 5, 9, 0, 0x598), (1, 15, 0, 5, 0x52fa8), (2, 0, 0, 0, 0x800), (2, 7, 4, 2, 0x21ca8), (2, 4, 1, 1, 0x10090), (2, 2, 4, 0, 0x1820), (2, 7, 0, 2, 0x20ca8), (3, 0, 0, 0, 0x4200), (3, 3, 1, 0, 0x4238), (3, 2, 0, 0, 0x4220), (3, 1, 1, 0, 0x4218), (3, 3, 0, 0, 0x4228), (4, 0, 0, 0, 0x200), (4, 1, 0, 0, 0x208), (4, 1, 0, 0, 0x208), (4, 0, 0, 0, 0x200), (4, 1, 0, 0, 0x208), (5, 0, 0, 0, 0x2000), (5, 0, 0, 0, 0x2000), (5, 0, 0, 0, 0x2000), (5, 0, 0, 0, 0x2000), (5, 0, 0, 0, 0x2000)], "thin SW_64KB_Z_X 64 bpp 33x20x12, 6 levels"),
        (true, 26, 4, 64, 64, 16, 1, Some(1), 262144, None, &[(0, 0, 0, 0, 0x0), (0, 63, 63, 15, 0x3f0fc), (0, 47, 4, 9, 0x13134), (0, 54, 3, 3, 0x19af8), (0, 23, 39, 12, 0x2ba5c), (0, 57, 14, 7, 0x1c6e4)], "SW_64KB_D_X 32 bpp 64x64x16, 1 level"),
        (true, 26, 1, 100, 60, 70, 7, Some(2), 1179648, None, &[(0, 0, 0, 0, 0x20000), (0, 99, 59, 69, 0x112b9f), (0, 15, 52, 41, 0xa1347), (0, 86, 43, 67, 0x11637e), (0, 87, 59, 38, 0xb61fb), (1, 0, 0, 0, 0x10000), (1, 49, 29, 34, 0x76d29), (1, 11, 28, 6, 0x126a3), (1, 8, 27, 2, 0x12238), (1, 30, 6, 17, 0x1c756), (2, 0, 0, 0, 0x800), (2, 24, 14, 16, 0xee10), (2, 8, 3, 12, 0x1998), (2, 15, 5, 9, 0x1d4f), (2, 8, 8, 11, 0x3824), (3, 0, 0, 0, 0x200), (3, 11, 6, 7, 0x7b7), (3, 7, 3, 6, 0x2fb), (3, 3, 5, 4, 0x68b), (3, 7, 2, 0, 0x253), (4, 0, 0, 0, 0x4200), (4, 5, 2, 3, 0x4275), (4, 0, 2, 1, 0x4214), (4, 2, 1, 1, 0x420e), (4, 0, 2, 0, 0x4210), (5, 0, 0, 0, 0x2000), (5, 2, 0, 1, 0x2006), (5, 1, 0, 0, 0x2001), (6, 0, 0, 0, 0x2500)], "SW_64KB_D_X 8 bpp 100x60x70, 7 levels"),
        (true, 26, 16, 9, 5, 3, 4, Some(1), 131072, None, &[(0, 0, 0, 0, 0x10000), (0, 8, 4, 2, 0x1c180), (0, 0, 2, 1, 0x10820), (0, 2, 3, 0, 0x10a40), (0, 8, 4, 1, 0x1c120), (1, 0, 0, 0, 0x8100), (1, 3, 1, 0, 0x8350), (1, 1, 0, 0, 0x8110), (1, 0, 1, 0, 0x8140), (1, 2, 0, 0, 0x8300), (2, 0, 0, 0, 0x100), (2, 1, 0, 0, 0x110), (3, 0, 0, 0, 0x1000)], "SW_64KB_D_X 128 bpp 9x5x3, 4 levels"),
        (false, 24, 1, 256, 256, 1, 9, None, 196608, Some(196608), &[(0, 0, 0, 0, 0x20000), (1, 0, 0, 0, 0x10000)], "swizzle mode 24, 1 bytes, 256x256, 9 levels, 1 slices"),
        (false, 24, 2, 128, 128, 1, 8, None, 131072, Some(131072), &[(0, 0, 0, 0, 0x10000)], "swizzle mode 24, 2 bytes, 128x128, 8 levels, 1 slices"),
        (false, 2, 1, 40, 20, 1, 3, None, 2304, Some(2304), &[(0, 0, 0, 0, 0x300), (0, 39, 19, 0, 0x81f), (0, 20, 6, 0, 0x42c), (0, 39, 0, 0, 0x507), (0, 0, 19, 0, 0x618), (1, 0, 0, 0, 0x100), (1, 19, 9, 0, 0x293), (1, 10, 3, 0, 0x15a), (1, 19, 0, 0, 0x203), (1, 0, 9, 0, 0x190), (2, 0, 0, 0, 0x0), (2, 9, 4, 0, 0x61), (2, 5, 1, 0, 0x15), (2, 9, 0, 0, 0x41), (2, 0, 4, 0, 0x20)], "swizzle mode 2, 1 bytes, 40x20, 3 levels, 1 slices"),
        (false, 6, 8, 100, 60, 1, 7, None, 90112, Some(90112), &[(0, 0, 0, 0, 0x6000), (0, 99, 59, 0, 0x154b8), (0, 50, 20, 0, 0xb920), (0, 99, 0, 0, 0x9028), (0, 0, 59, 0, 0x12490), (1, 0, 0, 0, 0x2000), (1, 49, 29, 0, 0x5d18), (1, 25, 10, 0, 0x2e88), (1, 49, 0, 0, 0x3808), (1, 0, 29, 0, 0x4510), (2, 0, 0, 0, 0x1000), (2, 24, 14, 0, 0x1f80), (2, 12, 5, 0, 0x1350), (2, 24, 0, 0, 0x1a00), (2, 0, 14, 0, 0x1580), (3, 0, 0, 0, 0x800), (3, 11, 6, 0, 0xba8), (3, 6, 2, 0, 0x8e0), (3, 11, 0, 0, 0xa28), (3, 0, 6, 0, 0x980), (4, 0, 0, 0, 0x600), (4, 5, 2, 0, 0x6c8), (4, 3, 1, 0, 0x638), (4, 5, 0, 0, 0x648), (4, 0, 2, 0, 0x680), (5, 0, 0, 0, 0x500), (5, 2, 0, 0, 0x520), (5, 1, 0, 0, 0x508), (5, 2, 0, 0, 0x520), (5, 0, 0, 0, 0x500), (6, 0, 0, 0, 0x400), (6, 0, 0, 0, 0x400), (6, 0, 0, 0, 0x400), (6, 0, 0, 0, 0x400), (6, 0, 0, 0, 0x400)], "swizzle mode 6, 8 bytes, 100x60, 7 levels, 1 slices"),
        (false, 10, 4, 200, 150, 1, 6, None, 393216, Some(393216), &[(0, 0, 0, 0, 0x20000), (0, 199, 149, 0, 0x584dc), (0, 100, 50, 0, 0x2b4a0), (0, 199, 0, 0, 0x3808c), (0, 0, 149, 0, 0x40450), (1, 0, 0, 0, 0x10000), (1, 99, 74, 0, 0x1e12c), (1, 50, 25, 0, 0x12d18), (1, 99, 0, 0, 0x1a00c), (1, 0, 74, 0, 0x14120), (2, 0, 0, 0, 0x8000), (2, 49, 36, 0, 0xb844), (2, 25, 12, 0, 0x8b44), (2, 49, 0, 0, 0xa804), (2, 0, 36, 0, 0x9040), (3, 0, 0, 0, 0x4000), (3, 24, 17, 0, 0x4e10), (3, 12, 6, 0, 0x42e0), (3, 24, 0, 0, 0x4a00), (3, 0, 17, 0, 0x4410), (4, 0, 0, 0, 0x2000), (4, 11, 8, 0, 0x230c), (4, 6, 3, 0, 0x20b8), (4, 11, 0, 0, 0x220c), (4, 0, 8, 0, 0x2100), (5, 0, 0, 0, 0x1000), (5, 5, 3, 0, 0x10b4), (5, 3, 1, 0, 0x101c), (5, 5, 0, 0, 0x1084), (5, 0, 3, 0, 0x1030)], "swizzle mode 10, 4 bytes, 200x150, 6 levels, 1 slices"),
        (false, 17, 4, 520, 260, 3, 4, None, 4718592, Some(1572864), &[(0, 0, 0, 0, 0x90000), (0, 519, 259, 0, 0x1700bc), (0, 260, 86, 0, 0xb46e0), (0, 519, 0, 0, 0xd008c), (0, 0, 259, 0, 0x130030), (0, 0, 0, 1, 0x210000), (0, 519, 259, 1, 0x2f00bc), (0, 260, 86, 1, 0x2346e0), (0, 519, 0, 1, 0x25008c), (0, 0, 259, 1, 0x2b0030), (0, 0, 0, 2, 0x390000), (0, 519, 259, 2, 0x4700bc), (0, 260, 86, 2, 0x3b46e0), (0, 519, 0, 2, 0x3d008c), (0, 0, 259, 2, 0x430030), (1, 0, 0, 0, 0x30000), (1, 259, 129, 0, 0x8001c), (1, 130, 43, 0, 0x41938), (1, 259, 0, 0, 0x5000c), (1, 0, 129, 0, 0x60010), (1, 0, 0, 1, 0x1b0000), (1, 259, 129, 1, 0x20001c), (1, 130, 43, 1, 0x1c1938), (1, 259, 0, 1, 0x1d000c), (1, 0, 129, 1, 0x1e0010), (1, 0, 0, 2, 0x330000), (1, 259, 129, 2, 0x38001c), (1, 130, 43, 2, 0x341938), (1, 259, 0, 2, 0x35000c), (1, 0, 129, 2, 0x360010), (2, 0, 0, 0, 0x10000), (2, 129, 64, 0, 0x24204), (2, 65, 21, 0, 0x18554), (2, 129, 0, 0, 0x20004), (2, 0, 64, 0, 0x14200), (2, 0, 0, 1, 0x190000), (2, 129, 64, 1, 0x1a4204), (2, 65, 21, 1, 0x198554), (2, 129, 0, 1, 0x1a0004), (2, 0, 64, 1, 0x194200), (2, 0, 0, 2, 0x310000), (2, 129, 64, 2, 0x324204), (2, 65, 21, 2, 0x318554), (2, 129, 0, 2, 0x320004), (2, 0, 64, 2, 0x314200), (3, 0, 0, 0, 0x0), (3, 64, 31, 0, 0x8470), (3, 32, 10, 0, 0x2520), (3, 64, 0, 0, 0x8100), (3, 0, 31, 0, 0x570), (3, 0, 0, 1, 0x180000), (3, 64, 31, 1, 0x188470), (3, 32, 10, 1, 0x182520), (3, 64, 0, 1, 0x188100), (3, 0, 31, 1, 0x180570), (3, 0, 0, 2, 0x300000), (3, 64, 31, 2, 0x308470), (3, 32, 10, 2, 0x302520), (3, 64, 0, 2, 0x308100), (3, 0, 31, 2, 0x300570)], "swizzle mode 17, 4 bytes, 520x260, 4 levels, 3 slices"),
        (false, 18, 16, 64, 40, 2, 5, None, 262144, Some(131072), &[(0, 0, 0, 0, 0x10000), (0, 63, 39, 0, 0x1ecf0), (0, 32, 13, 0, 0x18420), (0, 63, 0, 0, 0x1af50), (0, 0, 39, 0, 0x143a0), (0, 0, 0, 1, 0x30000), (0, 63, 39, 1, 0x3ecf0), (0, 32, 13, 1, 0x38420), (0, 63, 0, 1, 0x3af50), (0, 0, 39, 1, 0x343a0), (1, 0, 0, 0, 0x8100), (1, 31, 19, 0, 0xb7f0), (1, 16, 6, 0, 0xa480), (1, 31, 0, 0, 0xaf50), (1, 0, 19, 0, 0x99a0), (1, 0, 0, 1, 0x28100), (1, 31, 19, 1, 0x2b7f0), (1, 16, 6, 1, 0x2a480), (1, 31, 0, 1, 0x2af50), (1, 0, 19, 1, 0x299a0), (2, 0, 0, 0, 0x4200), (2, 15, 9, 0, 0x4c70), (2, 8, 3, 0, 0x4aa0), (2, 15, 0, 0, 0x4850), (2, 0, 9, 0, 0x4620), (2, 0, 0, 1, 0x24200), (2, 15, 9, 1, 0x24c70), (2, 8, 3, 1, 0x24aa0), (2, 15, 0, 1, 0x24850), (2, 0, 9, 1, 0x24620), (3, 0, 0, 0, 0x2400), (3, 7, 4, 0, 0x2750), (3, 4, 1, 0, 0x2620), (3, 7, 0, 0, 0x2650), (3, 0, 4, 0, 0x2500), (3, 0, 0, 1, 0x22400), (3, 7, 4, 1, 0x22750), (3, 4, 1, 1, 0x22620), (3, 7, 0, 1, 0x22650), (3, 0, 4, 1, 0x22500), (4, 0, 0, 0, 0x1800), (4, 3, 1, 0, 0x1870), (4, 2, 0, 0, 0x1840), (4, 3, 0, 0, 0x1850), (4, 0, 1, 0, 0x1820), (4, 0, 0, 1, 0x21800), (4, 3, 1, 1, 0x21870), (4, 2, 0, 1, 0x21840), (4, 3, 0, 1, 0x21850), (4, 0, 1, 1, 0x21820)], "swizzle mode 18, 16 bytes, 64x40, 5 levels, 2 slices"),
        (false, 21, 1, 100, 70, 3, 6, None, 73728, Some(24576), &[(0, 0, 0, 0, 0x2000), (0, 99, 69, 0, 0x5453), (0, 50, 23, 0, 0x2b72), (0, 99, 0, 0, 0x3c03), (0, 0, 69, 0, 0x4850), (0, 0, 0, 1, 0x8800), (0, 99, 69, 1, 0xbc53), (0, 50, 23, 1, 0x8372), (0, 99, 0, 1, 0x9403), (0, 0, 69, 1, 0xa050), (0, 0, 0, 2, 0xe400), (0, 99, 69, 2, 0x11053), (0, 50, 23, 2, 0xef72), (0, 99, 0, 2, 0xf803), (0, 0, 69, 2, 0x10c50), (1, 0, 0, 0, 0x1000), (1, 49, 34, 0, 0x1e21), (1, 25, 11, 0, 0x12b9), (1, 49, 0, 0, 0x1a01), (1, 0, 34, 0, 0x1420), (1, 0, 0, 1, 0x7800), (1, 49, 34, 1, 0x7621), (1, 25, 11, 1, 0x7ab9), (1, 49, 0, 1, 0x7201), (1, 0, 34, 1, 0x7c20), (1, 0, 0, 2, 0xd400), (1, 49, 34, 2, 0xda21), (1, 25, 11, 2, 0xd6b9), (1, 49, 0, 2, 0xde01), (1, 0, 34, 2, 0xd020), (2, 0, 0, 0, 0x800), (2, 24, 16, 0, 0xb08), (2, 12, 5, 0, 0x85c), (2, 24, 0, 0, 0xa08), (2, 0, 16, 0, 0x900), (2, 0, 0, 1, 0x6000), (2, 24, 16, 1, 0x6308), (2, 12, 5, 1, 0x605c), (2, 24, 0, 1, 0x6208), (2, 0, 16, 1, 0x6100), (2, 0, 0, 2, 0xcc00), (2, 24, 16, 2, 0xcf08), (2, 12, 5, 2, 0xcc5c), (2, 24, 0, 2, 0xce08), (2, 0, 16, 2, 0xcd00), (3, 0, 0, 0, 0x600), (3, 11, 7, 0, 0x67b), (3, 6, 2, 0, 0x626), (3, 11, 0, 0, 0x60b), (3, 0, 7, 0, 0x670), (3, 0, 0, 1, 0x6e00), (3, 11, 7, 1, 0x6e7b), (3, 6, 2, 1, 0x6e26), (3, 11, 0, 1, 0x6e0b), (3, 0, 7, 1, 0x6e70), (3, 0, 0, 2, 0xc200), (3, 11, 7, 2, 0xc27b), (3, 6, 2, 2, 0xc226), (3, 11, 0, 2, 0xc20b), (3, 0, 7, 2, 0xc270), (4, 0, 0, 0, 0x500), (4, 5, 3, 0, 0x535), (4, 3, 1, 0, 0x513), (4, 5, 0, 0, 0x505), (4, 0, 3, 0, 0x530), (4, 0, 0, 1, 0x6d00), (4, 5, 3, 1, 0x6d35), (4, 3, 1, 1, 0x6d13), (4, 5, 0, 1, 0x6d05), (4, 0, 3, 1, 0x6d30), (4, 0, 0, 2, 0xc100), (4, 5, 3, 2, 0xc135), (4, 3, 1, 2, 0xc113), (4, 5, 0, 2, 0xc105), (4, 0, 3, 2, 0xc130), (5, 0, 0, 0, 0x400), (5, 2, 1, 0, 0x412), (5, 1, 0, 0, 0x401), (5, 2, 0, 0, 0x402), (5, 0, 1, 0, 0x410), (5, 0, 0, 1, 0x6c00), (5, 2, 1, 1, 0x6c12), (5, 1, 0, 1, 0x6c01), (5, 2, 0, 1, 0x6c02), (5, 0, 1, 1, 0x6c10), (5, 0, 0, 2, 0xc000), (5, 2, 1, 2, 0xc012), (5, 1, 0, 2, 0xc001), (5, 2, 0, 2, 0xc002), (5, 0, 1, 2, 0xc010)], "swizzle mode 21, 1 bytes, 100x70, 6 levels, 3 slices"),
        (false, 22, 4, 33, 20, 2, 4, None, 32768, Some(16384), &[(0, 0, 0, 0, 0x2000), (0, 32, 19, 0, 0x3030), (0, 16, 6, 0, 0x2860), (0, 32, 0, 0, 0x3400), (0, 0, 19, 0, 0x2430), (0, 0, 0, 1, 0x6800), (0, 32, 19, 1, 0x7830), (0, 16, 6, 1, 0x6060), (0, 32, 0, 1, 0x7c00), (0, 0, 19, 1, 0x6c30), (1, 0, 0, 0, 0x1000), (1, 15, 9, 0, 0x139c), (1, 8, 3, 0, 0x1230), (1, 15, 0, 0, 0x128c), (1, 0, 9, 0, 0x1110), (1, 0, 0, 1, 0x5800), (1, 15, 9, 1, 0x5b9c), (1, 8, 3, 1, 0x5a30), (1, 15, 0, 1, 0x5a8c), (1, 0, 9, 1, 0x5910), (2, 0, 0, 0, 0x800), (2, 7, 4, 0, 0x8cc), (2, 4, 1, 0, 0x890), (2, 7, 0, 0, 0x88c), (2, 0, 4, 0, 0x840), (2, 0, 0, 1, 0x4000), (2, 7, 4, 1, 0x40cc), (2, 4, 1, 1, 0x4090), (2, 7, 0, 1, 0x408c), (2, 0, 4, 1, 0x4040), (3, 0, 0, 0, 0x600), (3, 3, 1, 0, 0x61c), (3, 2, 0, 0, 0x608), (3, 3, 0, 0, 0x60c), (3, 0, 1, 0, 0x610), (3, 0, 0, 1, 0x4e00), (3, 3, 1, 1, 0x4e1c), (3, 2, 0, 1, 0x4e08), (3, 3, 0, 1, 0x4e0c), (3, 0, 1, 1, 0x4e10)], "swizzle mode 22, 4 bytes, 33x20, 4 levels, 2 slices"),
    ];

    #[test]
    fn matches_addrlib_vectors() {
        for &(volume, mode, bpe, w, h, slices, levels, first_tail, bytes, layer, points, what) in ADDRLIB_VECTORS {
            let l = describe(mode, bpe, 1, w, h, slices, levels, volume).unwrap_or_else(|| panic!("{what}: no layout"));
            assert_eq!(l.guest_bytes, bytes, "{what}: guest bytes");
            if let Some(layer) = layer {
                assert_eq!(l.layer_bytes, layer, "{what}: layer bytes");
            }
            if let Some(ft) = first_tail {
                assert!(l.thick, "{what}: thick");
                for level in 0..levels {
                    assert_eq!(l.mips[level as usize].tail, level >= ft, "{what}: mip {level} tail");
                }
            }
            for &(level, x, y, z, want) in points {
                assert!(z < l.level_slices(level), "{what}: slice {z} outside mip {level}");
                assert_eq!(l.element_offset(level, z, x, y), Some(want), "{what}: mip {level} ({x}, {y}, {z})");
            }
        }
    }

    #[test]
    fn linear_chain() {
        // 4x4 8-bit, 2 levels: smallest level first, rows padded to 256 bytes.
        let l = describe(SWIZZLE_LINEAR, 1, 1, 4, 4, 1, 2, false).unwrap();
        assert_eq!((l.mips[0].offset, l.mips[0].size, l.mips[0].pitch), (512, 1024, 256));
        assert_eq!((l.mips[0].width, l.mips[0].height), (4, 4));
        assert_eq!((l.mips[1].offset, l.mips[1].size, l.mips[1].width), (0, 512, 2));
        // 8x8 BC1 (8-byte 4x4 blocks): 2x2 elements, 32 elements per row.
        let bc = describe(SWIZZLE_LINEAR, 8, 4, 8, 8, 1, 1, false).unwrap();
        assert_eq!((bc.mips[0].width, bc.mips[0].height, bc.mips[0].pitch, bc.mips[0].size), (2, 2, 32, 512));
    }

    #[test]
    fn thin_chains_and_tails() {
        // 256-byte blocks never use a tail.
        let s = describe(tiling::SWIZZLE_256B_S, 1, 1, 64, 64, 3, 6, false).unwrap();
        assert!(s.mips.iter().all(|m| !m.tail));
        assert_eq!((s.mips[0].size, s.mips[0].blocks_per_row), (4096, 4));
        // A deep 64 KiB chain ends in the tail block at offset 0.
        let d = describe(tiling::SWIZZLE_64KB_S, 4, 1, 1024, 1024, 1, 11, false).unwrap();
        assert!(!d.mips[0].tail && d.mips[10].tail);
        for m in d.mips.iter().filter(|m| m.tail) {
            assert_eq!((m.offset, m.size, m.blocks_per_row), (0, 65536, 1));
        }
        assert_eq!(d.mips[0].offset + d.mips[0].size, d.layer_bytes);
        // Render target 257x129 32 bpp: three 128x128 blocks per row; six
        // cube faces keep the padded slice stride.
        let rt = describe(tiling::SWIZZLE_64KB_R_X, 4, 1, 257, 129, 6, 1, false).unwrap();
        assert_eq!((rt.mips[0].blocks_per_row, rt.mips[0].size), (3, 393216));
        assert_eq!(rt.guest_bytes, 2359296);
        // BC7 (16-byte blocks) 1024x1024 with a full chain: 256x256 elements at level 0.
        let bc = describe(tiling::SWIZZLE_64KB_S, 16, 4, 1024, 1024, 2, 11, false).unwrap();
        assert_eq!((bc.mips[0].width, bc.mips[0].height), (256, 256));
        assert_eq!((bc.mips[3].width, bc.mips[10].width), (32, 1));
        assert!(bc.mips[10].tail);
        assert_eq!(bc.guest_bytes, bc.layer_bytes * 2);
    }

    /// A block-compressed texture in the render-target swizzle (R_X):
    /// BC1 2048x2048, 8 layers, 1 level, as captured from a title (64 KiB
    /// blocks of 128x64 8-byte elements, 512x512 elements a layer, 2 MiB
    /// per layer); a small odd-sized array round-trips its elements.
    #[test]
    fn block_compressed_render_target_swizzle() {
        let l = describe(tiling::SWIZZLE_64KB_R_X, 8, 4, 2048, 2048, 8, 1, false).unwrap();
        assert_eq!((l.guest_bytes, l.layer_bytes), (0x100_0000, 0x20_0000));
        assert_eq!((l.mips[0].pitch, l.mips[0].height, l.mips[0].tail), (512, 512, false));
        let p = l.params().unwrap();
        assert_eq!((p.block_width, p.block_height), (128, 64));
        let small = describe(tiling::SWIZZLE_64KB_R_X, 8, 4, 67, 51, 8, 1, false).unwrap();
        let (w, h) = (small.mips[0].width, small.mips[0].height);
        assert_eq!((w, h), (17, 13));
        let mut seen = std::collections::HashSet::new();
        for z in 0..8 {
            for y in 0..h {
                for x in 0..w {
                    let o = small.element_offset(0, z, x, y).unwrap();
                    assert!(o + 8 <= small.guest_bytes && o % 8 == 0, "({x}, {y}, {z}) at {o:#x}");
                    assert!(seen.insert(o), "({x}, {y}, {z}) shares {o:#x}");
                }
            }
        }
    }

    #[test]
    fn thick_volumes() {
        let v = describe(tiling::SWIZZLE_4KB_S, 4, 1, 64, 64, 32, 3, true).unwrap();
        assert!(v.thick && v.block_depth == 8);
        assert_eq!((v.mips[2].offset, v.mips[2].size), (0, 8192));
        assert_eq!((v.mips[1].offset, v.mips[1].size), (8192, 32768));
        assert_eq!((v.mips[0].offset, v.mips[0].size), (40960, 131072));
        assert_eq!((v.layer_bytes, v.guest_bytes), (172032, 172032 * 4));
        assert_eq!((v.level_slices(1), v.level_slices(2)), (16, 8));
        let odd = describe(tiling::SWIZZLE_4KB_S, 4, 1, 33, 20, 12, 2, true).unwrap();
        assert_eq!((odd.mips[1].width, odd.mips[1].blocks_per_row, odd.mips[1].size), (16, 3, 12288));
        assert_eq!((odd.mips[0].blocks_per_row, odd.mips[0].offset, odd.mips[0].size), (5, 12288, 40960));
        // Thin-slice volume (the colour-grading LUT layout): 64^3 RGBA16F R_X.
        let lut = describe(tiling::SWIZZLE_64KB_R_X, 8, 1, 64, 64, 64, 1, true).unwrap();
        assert!(!lut.thick);
        assert_eq!((lut.layer_bytes, lut.guest_bytes), (65536, 64 * 65536));
        // Block-compressed volumes have no thick layout.
        assert!(describe(tiling::SWIZZLE_64KB_S, 16, 4, 64, 64, 4, 1, true).is_none());
    }
}
