// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! CPU (de)tiling of the PS5 GPU's (gfx1013) swizzled 2D surfaces.
//!
//! A tiled surface is a row-major grid of power-of-two swizzle blocks
//! (256 B, 4 KiB or 64 KiB). Inside a block every address bit is the XOR of
//! a set of X and Y coordinate bits (the AddrLib "equation"). Because parity
//! distributes over XOR, the in-block byte offset factors into two independent
//! axis terms:
//!
//! ```text
//!   inBlock = xTerm[x & xMask] ^ yTerm[y & yMask]
//!   src     = ((y / blockH) * blocksPerRow + x / blockW) * blockBytes + inBlock
//! ```
//!
//! The equations are the GFX10 *non*-RB+ ones of a 16-pipe part (gfx1013 is
//! Navi10-class; RB+ only exists on Navi14 and GFX10.3+). The pipe/bank XOR
//! of the `_X` modes uses coordinate bits inside the block only. They were
//! confirmed against the console's own `libSceAgcGpuAddress` tiler, which
//! titles link statically: its per-mode element-offset kernels reproduce
//! these tables bit for bit for all 14 modes and element sizes (see the
//! golden vectors in the tests).
//!
//! The slice index (array layer, or depth slice of a volume) enters the
//! `_X` modes' pipe/bank XOR: [`DetileParams::offset_in_slice`]. Volumes of
//! `SW_4KB_S` / `SW_64KB_S` / `SW_64KB_S_X` / `SW_64KB_D_X` use thick blocks spanning several
//! slices ([`DetileParams::new_thick`]); the mip-chain layout lives in
//! [`crate::tiling_layout`].
//!
//! Supported swizzle modes (1/2/4/8/16-byte elements):
//!   * 1/2   `SW_256B_S` / `SW_256B_D`
//!   * 5/6   `SW_4KB_S` / `SW_4KB_D`, 21/22 `SW_4KB_S_X` / `SW_4KB_D_X`
//!   * 9/10  `SW_64KB_S` / `SW_64KB_D`, 17/18 `SW_64KB_S_T` / `SW_64KB_D_T`
//!   * 24    `SW_64KB_Z_X` (depth targets)
//!   * 25/26 `SW_64KB_S_X` / `SW_64KB_D_X`
//!   * 27    `SW_64KB_R_X` (colour render targets, the layout of every
//!     VideoOut display buffer with tiling mode 0)
//!   * 0     linear

/// GFX10 swizzle mode numbers.
pub const SWIZZLE_LINEAR: u32 = 0;
pub const SWIZZLE_256B_S: u32 = 1;
pub const SWIZZLE_4KB_S: u32 = 5;
pub const SWIZZLE_64KB_S: u32 = 9;
pub const SWIZZLE_64KB_Z_X: u32 = 24;
pub const SWIZZLE_64KB_R_X: u32 = 27;
/// Further modes from the generated equation table ([`crate::tiling_equations`]).
pub const SWIZZLE_256B_D: u32 = 2;
pub const SWIZZLE_4KB_D: u32 = 6;
pub const SWIZZLE_64KB_D: u32 = 10;
pub const SWIZZLE_64KB_S_T: u32 = 17;
pub const SWIZZLE_64KB_D_T: u32 = 18;
pub const SWIZZLE_4KB_S_X: u32 = 21;
pub const SWIZZLE_4KB_D_X: u32 = 22;
pub const SWIZZLE_64KB_S_X: u32 = 25;
pub const SWIZZLE_64KB_D_X: u32 = 26;

/// Every supported tiled mode.
pub const TILED_MODES: [u32; 14] = [
    SWIZZLE_256B_S,
    SWIZZLE_256B_D,
    SWIZZLE_4KB_S,
    SWIZZLE_4KB_D,
    SWIZZLE_64KB_S,
    SWIZZLE_64KB_D,
    SWIZZLE_64KB_S_T,
    SWIZZLE_64KB_D_T,
    SWIZZLE_4KB_S_X,
    SWIZZLE_4KB_D_X,
    SWIZZLE_64KB_Z_X,
    SWIZZLE_64KB_S_X,
    SWIZZLE_64KB_D_X,
    SWIZZLE_64KB_R_X,
];

/// One address bit: XOR of the X bits in `x`, the Y bits in `y` and the
/// slice bits in `z` (array layer or depth slice; the `_X` pipe/bank XOR,
/// or the depth coordinate inside a thick block).
#[derive(Clone, Copy)]
struct AddressBit {
    x: u32,
    y: u32,
    z: u32,
}

const Z: AddressBit = AddressBit { x: 0, y: 0, z: 0 };
const fn x(b: u32) -> AddressBit {
    AddressBit { x: 1 << b, y: 0, z: 0 }
}
const fn y(b: u32) -> AddressBit {
    AddressBit { x: 0, y: 1 << b, z: 0 }
}

// 64 KiB S (standard).
const STANDARD_64K: [[AddressBit; 16]; 5] = [
    [x(0), x(1), x(2), x(3), y(0), y(1), y(2), y(3),
     y(4), x(4), y(5), x(5), y(6), x(6), y(7), x(7)],
    [Z, x(0), x(1), x(2), y(0), y(1), y(2), x(3),
     y(3), x(4), y(4), x(5), y(5), x(6), y(6), x(7)],
    [Z, Z, x(0), x(1), y(0), y(1), y(2), x(2),
     y(3), x(3), y(4), x(4), y(5), x(5), y(6), x(6)],
    [Z, Z, Z, x(0), y(0), y(1), x(1), x(2),
     y(2), x(3), y(3), x(4), y(4), x(5), y(5), x(6)],
    [Z, Z, Z, Z, y(0), y(1), x(0), x(1),
     y(2), x(2), y(3), x(3), y(4), x(4), y(5), x(5)],
];

// 256 B S: the low octet of the 4 KiB S micro-tile.
const STANDARD_256: [[AddressBit; 8]; 5] = [
    [x(0), x(1), x(2), x(3), y(0), y(1), y(2), y(3)],
    [Z, x(0), x(1), x(2), y(0), y(1), y(2), x(3)],
    [Z, Z, x(0), x(1), y(0), y(1), y(2), x(2)],
    [Z, Z, Z, x(0), y(0), y(1), x(1), x(2)],
    [Z, Z, Z, Z, y(0), y(1), x(0), x(1)],
];

// 4 KiB S.
const STANDARD_4K: [[AddressBit; 12]; 5] = [
    [x(0), x(1), x(2), x(3), y(0), y(1), y(2), y(3), y(4), x(4), y(5), x(5)],
    [Z, x(0), x(1), x(2), y(0), y(1), y(2), x(3), y(3), x(4), y(4), x(5)],
    [Z, Z, x(0), x(1), y(0), y(1), y(2), x(2), y(3), x(3), y(4), x(4)],
    [Z, Z, Z, x(0), y(0), y(1), x(1), x(2), y(2), x(3), y(3), x(4)],
    [Z, Z, Z, Z, y(0), y(1), x(0), x(1), y(2), x(2), y(3), x(3)],
];

fn pattern(mode: u32, bpe_log2: usize) -> Option<&'static [AddressBit]> {
    if bpe_log2 >= 5 {
        return None;
    }
    Some(match mode {
        SWIZZLE_256B_S => &STANDARD_256[bpe_log2],
        SWIZZLE_4KB_S => &STANDARD_4K[bpe_log2],
        SWIZZLE_64KB_S => &STANDARD_64K[bpe_log2],
        _ => return generated_pattern(mode, 1 << bpe_log2),
    })
}

type EquationTable = Vec<(u32, u32, Vec<AddressBit>)>;

fn convert(table: &[(u32, u32, [(u32, u32, u32); 16])]) -> EquationTable {
    table.iter().map(|(m, b, bits)| (*m, *b, bits.iter().map(|&(x, y, z)| AddressBit { x, y, z }).collect())).collect()
}

/// Equations of the generated table, converted once.
fn generated_pattern(mode: u32, bpe: u32) -> Option<&'static [AddressBit]> {
    static TABLE: std::sync::OnceLock<EquationTable> = std::sync::OnceLock::new();
    let t = TABLE.get_or_init(|| convert(crate::tiling_equations::EQUATIONS));
    t.iter().find(|(m, b, _)| *m == mode && *b == bpe).map(|(_, _, v)| v.as_slice())
}

/// The pipe/bank XOR a surface address of an XOR swizzle mode carries: its
/// offset inside the swizzle block (4 KiB for `SW_4KB_*_X`, 64 KiB for
/// `SW_64KB_*_X`); 0 for the other modes.
pub fn pipe_bank_xor(address: u64, swizzle: u32) -> u64 {
    match swizzle {
        SWIZZLE_4KB_S_X | SWIZZLE_4KB_D_X => address & 0xFFF,
        SWIZZLE_64KB_Z_X | SWIZZLE_64KB_S_X | SWIZZLE_64KB_D_X | SWIZZLE_64KB_R_X => address & 0xFFFF,
        _ => 0,
    }
}

/// Report the first surface with a pipe/bank XOR (`what` names its use):
/// the XOR is not applied, so its detiled contents are wrong.
pub fn report_pipe_bank_xor(what: &str, address: u64, swizzle: u32) {
    static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        log::warn!("ps5gpu: {what} {address:#x} swizzle {swizzle} carries a pipe/bank XOR {:#x}, which is not modelled", pipe_bank_xor(address, swizzle));
    }
}

/// Thick (3D) equation of `mode`, None for modes without a thick layout.
fn thick_pattern(mode: u32, bpe: u32) -> Option<&'static [AddressBit]> {
    static TABLE: std::sync::OnceLock<EquationTable> = std::sync::OnceLock::new();
    let t = TABLE.get_or_init(|| convert(crate::tiling_equations::THICK_EQUATIONS));
    t.iter().find(|(m, b, _)| *m == mode && *b == bpe).map(|(_, _, v)| v.as_slice())
}

/// True when 3D surfaces of `mode` use thick (multi-slice) blocks:
/// `SW_4KB_S`, `SW_4KB_S_X`, `SW_64KB_S`, `SW_64KB_S_X` and `SW_64KB_D_X`. `SW_64KB_Z_X`
/// and `SW_64KB_R_X` lay a volume out as thin slices, each holding its own
/// mip chain.
pub fn has_thick_layout(mode: u32) -> bool {
    matches!(mode, SWIZZLE_4KB_S | SWIZZLE_4KB_S_X | SWIZZLE_64KB_S | SWIZZLE_64KB_S_X | SWIZZLE_64KB_D_X)
}

/// Thick block extent `[width, height, depth]` in elements (addrlib
/// `Block4K_Log2_3d` / `Block64K_Log2_3d`).
pub fn thick_block_extent(mode: u32, bytes_per_element: u32) -> Option<[u32; 3]> {
    const THICK_4K: [[u32; 3]; 5] = [[4, 4, 4], [3, 4, 4], [3, 4, 3], [3, 3, 3], [2, 3, 3]];
    const THICK_64K: [[u32; 3]; 5] = [[6, 5, 5], [5, 5, 5], [5, 5, 4], [5, 4, 4], [4, 4, 4]];
    if !bytes_per_element.is_power_of_two() || bytes_per_element > 16 {
        return None;
    }
    let i = bytes_per_element.trailing_zeros() as usize;
    let l = match mode {
        SWIZZLE_4KB_S | SWIZZLE_4KB_S_X => THICK_4K[i],
        SWIZZLE_64KB_S | SWIZZLE_64KB_S_X | SWIZZLE_64KB_D_X => THICK_64K[i],
        _ => return None,
    };
    Some(l.map(|b| 1 << b))
}

fn block_bytes(mode: u32) -> usize {
    match mode {
        SWIZZLE_256B_S | SWIZZLE_256B_D => 256,
        SWIZZLE_4KB_S | SWIZZLE_4KB_D | SWIZZLE_4KB_S_X | SWIZZLE_4KB_D_X => 4096,
        _ => 65536,
    }
}

/// Thin block extent `(width, height, bytes)` in elements of a tiled
/// `mode` (None for linear or unknown modes): the block shape of
/// [`DetileParams::new`] without building its address tables.
pub fn thin_block_extent(mode: u32, bytes_per_element: usize) -> Option<(u32, u32, usize)> {
    if mode == SWIZZLE_LINEAR || bytes_per_element == 0 || !bytes_per_element.is_power_of_two() || bytes_per_element > 16 {
        return None;
    }
    let bpe_log2 = bytes_per_element.trailing_zeros() as usize;
    pattern(mode, bpe_log2)?;
    let bb = block_bytes(mode);
    let total_bits = ((bb >> bpe_log2) as u32).trailing_zeros();
    let wbits = total_bits.div_ceil(2);
    Some((1 << wbits, 1 << (total_bits - wbits), bb))
}

/// Pitch (elements) and bytes of a thin `width` x `height` surface of
/// `mode`: [`DetileParams::new`] + [`DetileParams::surface_size`] without
/// building the address tables (resolved per draw for every target).
pub fn thin_surface(mode: u32, bytes_per_element: usize, width: u32, height: u32) -> Option<(u32, usize)> {
    if bytes_per_element == 0 || !bytes_per_element.is_power_of_two() || bytes_per_element > 16 || width == 0 {
        return None;
    }
    if mode == SWIZZLE_LINEAR {
        return Some((width, width as usize * height as usize * bytes_per_element));
    }
    let (bw, bh, bb) = thin_block_extent(mode, bytes_per_element)?;
    let pitch = width.div_ceil(bw) * bw;
    Some((pitch, height.div_ceil(bh) as usize * (pitch / bw) as usize * bb))
}

/// Resolved addressing for one (mode, element size, pitch) combination.
#[derive(Clone, Debug)]
pub struct DetileParams {
    pub mode: u32,
    pub bytes_per_element: usize,
    pub block_width: u32,
    pub block_height: u32,
    pub block_bytes: usize,
    /// Pitch in elements (row length of the linear layout for mode 0).
    pub pitch: u32,
    pub blocks_per_row: u32,
    /// Depth of a block in slices (1 unless thick).
    pub block_depth: u32,
    pub thick: bool,
    x_terms: Vec<u32>,
    x_mask: u32,
    y_terms: Vec<u32>,
    y_mask: u32,
    z_terms: Vec<u32>,
    z_mask: u32,
}

fn axis_period(mask: u32) -> usize {
    if mask == 0 {
        1
    } else {
        1usize << (32 - mask.leading_zeros())
    }
}

fn axis_term(coord: u32, pat: &[AddressBit], axis: usize) -> u32 {
    let mut off = 0;
    for (bit, b) in pat.iter().enumerate() {
        let m = [b.x, b.y, b.z][axis];
        off |= ((coord & m).count_ones() & 1) << bit;
    }
    off
}

impl DetileParams {
    /// Build the addressing for `mode` with `bytes_per_element`-byte
    /// elements and a surface pitch of `pitch` elements (the pitch is rounded
    /// up to whole swizzle blocks for tiled modes).
    pub fn new(mode: u32, bytes_per_element: usize, pitch: u32) -> Option<Self> {
        Self::build(mode, bytes_per_element, pitch, false)
    }

    /// The thick (3D) addressing of `mode`: blocks span
    /// [`block_depth`](Self::block_depth) slices and the slice coordinate
    /// selects the depth bits inside the block. None for modes without a
    /// thick layout.
    pub fn new_thick(mode: u32, bytes_per_element: usize, pitch: u32) -> Option<Self> {
        Self::build(mode, bytes_per_element, pitch, true)
    }

    fn build(mode: u32, bytes_per_element: usize, pitch: u32, thick: bool) -> Option<Self> {
        if bytes_per_element == 0 || !bytes_per_element.is_power_of_two() || bytes_per_element > 16 || pitch == 0 {
            return None;
        }
        if mode == SWIZZLE_LINEAR {
            return Some(Self {
                mode,
                bytes_per_element,
                block_width: 1,
                block_height: 1,
                block_bytes: bytes_per_element,
                pitch,
                blocks_per_row: pitch,
                block_depth: 1,
                thick: false,
                x_terms: vec![0],
                x_mask: 0,
                y_terms: vec![0],
                y_mask: 0,
                z_terms: vec![0],
                z_mask: 0,
            });
        }
        let bpe_log2 = bytes_per_element.trailing_zeros() as usize;
        let bb = block_bytes(mode);
        let (pat, bw, bh, bd) = if thick {
            let pat = thick_pattern(mode, bytes_per_element as u32)?;
            let [w, h, d] = thick_block_extent(mode, bytes_per_element as u32)?;
            (pat, w, h, d)
        } else {
            let pat = pattern(mode, bpe_log2)?;
            let elements = (bb >> bpe_log2) as u32;
            let total_bits = elements.trailing_zeros();
            let wbits = total_bits.div_ceil(2);
            let hbits = total_bits - wbits;
            (pat, 1u32 << wbits, 1u32 << hbits, 1)
        };
        let (mut xm, mut ym, mut zm) = (0u32, 0u32, 0u32);
        for b in pat {
            xm |= b.x;
            ym |= b.y;
            zm |= b.z;
        }
        let (xl, yl, zl) = (axis_period(xm), axis_period(ym), axis_period(zm));
        let x_terms = (0..xl as u32).map(|c| axis_term(c, pat, 0)).collect();
        let y_terms = (0..yl as u32).map(|c| axis_term(c, pat, 1)).collect();
        let z_terms = (0..zl as u32).map(|c| axis_term(c, pat, 2)).collect();
        let pitch = pitch.div_ceil(bw) * bw;
        Some(Self {
            mode,
            bytes_per_element,
            block_width: bw,
            block_height: bh,
            block_bytes: bb,
            pitch,
            blocks_per_row: pitch / bw,
            block_depth: bd,
            thick,
            x_terms,
            x_mask: (xl - 1) as u32,
            y_terms,
            y_mask: (yl - 1) as u32,
            z_terms,
            z_mask: (zl - 1) as u32,
        })
    }

    /// Bytes occupied by a surface `height` rows tall.
    pub fn surface_size(&self, height: u32) -> usize {
        if self.mode == SWIZZLE_LINEAR {
            return self.pitch as usize * height as usize * self.bytes_per_element;
        }
        let rows = height.div_ceil(self.block_height) as usize;
        rows * self.blocks_per_row as usize * self.block_bytes
    }

    /// Byte offset of element (x, y) inside the tiled surface.
    #[inline]
    pub fn offset(&self, x: u32, y: u32) -> usize {
        if self.mode == SWIZZLE_LINEAR {
            return (y as usize * self.pitch as usize + x as usize) * self.bytes_per_element;
        }
        let block = (y / self.block_height) as usize * self.blocks_per_row as usize + (x / self.block_width) as usize;
        let inner = self.x_terms[(x & self.x_mask) as usize] ^ self.y_terms[(y & self.y_mask) as usize];
        block * self.block_bytes + inner as usize
    }

    /// Byte offset of element (x, y) of slice `z`: the slice only enters
    /// the in-block XOR (the `_X` pipe/bank bits of a layer, or the depth
    /// bits of a thick block); the caller adds the offset of the layer or
    /// slab.
    #[inline]
    pub fn offset_in_slice(&self, x: u32, y: u32, z: u32) -> usize {
        self.offset(x, y) ^ self.z_terms[(z & self.z_mask) as usize] as usize
    }

    /// In-block byte offset of element (x, y) of slice `z` (the XOR
    /// equation alone; the coordinates may reach past the block, whose
    /// bits feed the pipe/bank XOR).
    #[inline]
    pub fn in_block(&self, x: u32, y: u32, z: u32) -> u32 {
        self.x_terms[(x & self.x_mask) as usize] ^ self.y_terms[(y & self.y_mask) as usize] ^ self.z_terms[(z & self.z_mask) as usize]
    }

    fn row_offsets(&self, y: u32) -> (usize, u32) {
        let block_row = (y / self.block_height) as usize * self.blocks_per_row as usize * self.block_bytes;
        (block_row, self.y_terms[(y & self.y_mask) as usize])
    }
}

/// The `SW_64KB_R_X` in-block offset written out per element size as shift
/// and mask terms (an independent form of the table entry; both must agree
/// for every coordinate).
pub fn render_target_offset(x: u32, y: u32, bytes_per_element: u32) -> u32 {
    match bytes_per_element {
        1 => ((y << 2) & 0x008) ^ ((y << 4) & 0x010) ^ ((y << 3) & 0x0a0) ^ ((y << 5) & 0xf00) ^ ((y << 6) & 0x1000) ^ ((y << 7) & 0x4000)
            ^ (x & 7) ^ ((x << 3) & 0x040) ^ ((x << 5) & 0x300) ^ ((x << 4) & 0x400) ^ ((x << 6) & 0x800) ^ ((x << 7) & 0x2000) ^ ((x << 8) & 0x8000),
        2 => ((y << 4) & 0x070) ^ ((y << 5) & 0xf00) ^ ((y << 8) & 0x5000)
            ^ ((x << 1) & 0x00e) ^ ((x << 4) & 0x480) ^ ((x << 5) & 0x300) ^ ((x << 6) & 0x800) ^ ((x << 7) & 0x2000) ^ ((x << 8) & 0x8000),
        4 => ((y << 4) & 0x070) ^ ((y << 5) & 0xf00) ^ ((y << 9) & 0x1000) ^ ((y << 8) & 0x4000)
            ^ ((x << 2) & 0x00c) ^ ((x << 5) & 0x380) ^ ((x << 4) & 0x400) ^ ((x << 6) & 0x800) ^ ((x << 9) & 0xa000),
        8 => ((y << 4) & 0x010) ^ ((y << 6) & 0x080) ^ ((y << 5) & 0xf00) ^ ((y << 10) & 0x5000)
            ^ ((x << 3) & 0x008) ^ ((x << 4) & 0x460) ^ ((x << 5) & 0x300) ^ ((x << 6) & 0x800) ^ ((x << 10) & 0x2000) ^ ((x << 9) & 0x8000),
        16 => ((x << 4) & 0x410) ^ ((x << 5) & 0x340) ^ ((x << 6) & 0x800) ^ ((x << 11) & 0xa000)
            ^ ((y << 5) & 0xf20) ^ ((y << 6) & 0x080) ^ ((y << 10) & 0x1000) ^ ((y << 11) & 0x4000),
        _ => 0,
    }
}

/// The in-block address equation of a tiled mode: for every byte-offset bit
/// (index = destination bit) the X and Y coordinate bits XORed into it, and
/// the block geometry `(equation, block_width, block_height, block_bytes)`.
/// The GPU tiler builds its shaders from this table, so both paths share
/// one definition.
pub fn address_equation(mode: u32, bytes_per_element: usize) -> Option<(Vec<(u32, u32, u32)>, u32, u32, usize)> {
    if mode == SWIZZLE_LINEAR {
        return None;
    }
    let p = DetileParams::new(mode, bytes_per_element, 1)?;
    let pat = pattern(mode, bytes_per_element.trailing_zeros() as usize)?;
    Some((pat.iter().map(|b| (b.x, b.y, b.z)).collect(), p.block_width, p.block_height, p.block_bytes))
}

/// The thick (3D) equation of `mode` in the form of [`address_equation`]
/// (the third mask selects depth bits inside the block).
pub fn thick_address_equation(mode: u32, bytes_per_element: usize) -> Option<(Vec<(u32, u32, u32)>, u32, u32, usize)> {
    let p = DetileParams::new_thick(mode, bytes_per_element, 1)?;
    let pat = thick_pattern(mode, bytes_per_element as u32)?;
    Some((pat.iter().map(|b| (b.x, b.y, b.z)).collect(), p.block_width, p.block_height, p.block_bytes))
}

const PARALLEL_THRESHOLD: usize = 512 * 512;

fn for_rows(height: u32, width: u32, row_bytes: usize, out: &mut [u8], f: impl Fn(u32, &mut [u8]) + Sync) {
    let rows = height as usize;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(8);
    if (width as usize) * rows < PARALLEL_THRESHOLD || threads <= 1 {
        for (yy, row) in out.chunks_mut(row_bytes).take(rows).enumerate() {
            f(yy as u32, row);
        }
        return;
    }
    let per = rows.div_ceil(threads);
    std::thread::scope(|s| {
        for (ci, chunk) in out[..rows * row_bytes].chunks_mut(per * row_bytes).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (i, row) in chunk.chunks_mut(row_bytes).enumerate() {
                    f((ci * per + i) as u32, row);
                }
            });
        }
    });
}

/// Copy a `width` x `height` region out of `tiled` into tightly packed
/// `linear` rows. Elements whose source lies outside `tiled` are left
/// untouched. Returns false when `linear` is too small.
pub fn detile(p: &DetileParams, tiled: &[u8], linear: &mut [u8], width: u32, height: u32) -> bool {
    let bpe = p.bytes_per_element;
    let row_bytes = width as usize * bpe;
    if width == 0 || height == 0 || linear.len() < row_bytes * height as usize {
        return false;
    }
    for_rows(height, width, row_bytes, linear, |yy, row| {
        if p.mode == SWIZZLE_LINEAR {
            let s = yy as usize * p.pitch as usize * bpe;
            let n = row_bytes.min(tiled.len().saturating_sub(s));
            row[..n].copy_from_slice(&tiled[s..s + n]);
            return;
        }
        let (base, yterm) = p.row_offsets(yy);
        for xx in 0..width {
            let src = base
                + (xx / p.block_width) as usize * p.block_bytes
                + (p.x_terms[(xx & p.x_mask) as usize] ^ yterm) as usize;
            if src + bpe <= tiled.len() {
                let d = xx as usize * bpe;
                row[d..d + bpe].copy_from_slice(&tiled[src..src + bpe]);
            }
        }
    });
    true
}

/// Inverse of [`detile`]: scatter tightly packed `linear` rows into `tiled`.
pub fn tile(p: &DetileParams, linear: &[u8], tiled: &mut [u8], width: u32, height: u32) -> bool {
    let bpe = p.bytes_per_element;
    let row_bytes = width as usize * bpe;
    if linear.len() < row_bytes * height as usize {
        return false;
    }
    for yy in 0..height {
        for xx in 0..width {
            let dst = p.offset(xx, yy);
            let s = yy as usize * row_bytes + xx as usize * bpe;
            if dst + bpe <= tiled.len() {
                tiled[dst..dst + bpe].copy_from_slice(&linear[s..s + bpe]);
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8).collect()
    }

    #[test]
    fn pipe_bank_xor_of_xor_swizzles() {
        // The reference driver's pipe/bank XOR test: a 64 KiB-aligned base
        // plus the XOR in the T# address.
        assert_eq!(pipe_bank_xor(0x12_3450_0000 + 0x3400, SWIZZLE_64KB_R_X), 0x3400);
        assert_eq!(pipe_bank_xor(0x12_3450_0000 + 0x3400, SWIZZLE_64KB_S_X), 0x3400);
        assert_eq!(pipe_bank_xor(0x12_3450_0000 + 0x3400, SWIZZLE_4KB_S_X), 0x400, "4 KiB XOR blocks");
        assert_eq!(pipe_bank_xor(0x12_3450_0000 + 0x3400, SWIZZLE_64KB_S), 0, "no XOR outside the _X modes");
        assert_eq!(pipe_bank_xor(0x12_3450_0000, SWIZZLE_64KB_Z_X), 0);
    }

    #[test]
    fn thin_block_extent_matches_params() {
        for mode in 0..32 {
            for bpe in [1usize, 2, 4, 8, 16] {
                let p = DetileParams::new(mode, bpe, 1).filter(|_| mode != SWIZZLE_LINEAR).map(|p| (p.block_width, p.block_height, p.block_bytes));
                assert_eq!(thin_block_extent(mode, bpe), p, "mode {mode} bpe {bpe}");
            }
        }
    }

    #[test]
    fn render_target_table_matches_bit_equation() {
        for bpe in [1u32, 2, 4, 8, 16] {
            let p = DetileParams::new(SWIZZLE_64KB_R_X, bpe as usize, 1024).unwrap();
            for yy in 0..600 {
                for xx in 0..600 {
                    let inner = p.x_terms[(xx & p.x_mask) as usize] ^ p.y_terms[(yy & p.y_mask) as usize];
                    assert_eq!(inner, render_target_offset(xx, yy, bpe), "bpe={bpe} x={xx} y={yy}");
                }
            }
        }
    }

    #[test]
    fn block_is_bijective() {
        for mode in TILED_MODES {
            for bpe in [1usize, 2, 4, 8, 16] {
                let p = DetileParams::new(mode, bpe, 4096).unwrap();
                for (bx, by) in [(0u32, 0u32), (1, 1), (3, 2)] {
                    let mut seen = vec![false; p.block_bytes / bpe];
                    let base = (by as usize * p.blocks_per_row as usize + bx as usize) * p.block_bytes;
                    for yy in 0..p.block_height {
                        for xx in 0..p.block_width {
                            let o = p.offset(bx * p.block_width + xx, by * p.block_height + yy);
                            assert!(o >= base && o < base + p.block_bytes, "mode={mode} bpe={bpe}");
                            assert_eq!((o - base) % bpe, 0);
                            let e = (o - base) / bpe;
                            assert!(!seen[e], "mode={mode} bpe={bpe} duplicate");
                            seen[e] = true;
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn round_trip_tile_detile() {
        for mode in std::iter::once(SWIZZLE_LINEAR).chain(TILED_MODES) {
            for bpe in [1usize, 2, 4, 8, 16] {
                for (w, h) in [(37u32, 19u32), (300, 170), (1920 / 4, 1080 / 4)] {
                    let p = DetileParams::new(mode, bpe, w).unwrap();
                    let lin = synth(w as usize * h as usize * bpe);
                    let mut tiled = vec![0u8; p.surface_size(h)];
                    assert!(tile(&p, &lin, &mut tiled, w, h));
                    let mut back = vec![0u8; lin.len()];
                    assert!(detile(&p, &tiled, &mut back, w, h));
                    assert_eq!(lin, back, "mode={mode} bpe={bpe} {w}x{h}");
                }
            }
        }
    }

    #[test]
    fn full_hd_render_target_round_trip_parallel() {
        let (w, h) = (1920u32, 1080u32);
        let p = DetileParams::new(SWIZZLE_64KB_R_X, 4, w).unwrap();
        assert_eq!(p.block_width, 128);
        assert_eq!(p.block_height, 128);
        assert_eq!(p.surface_size(h), 15 * 9 * 65536);
        let lin = synth((w * h * 4) as usize);
        let mut tiled = vec![0u8; p.surface_size(h)];
        tile(&p, &lin, &mut tiled, w, h);
        assert_ne!(&tiled[..lin.len()], &lin[..]);
        let mut back = vec![0u8; lin.len()];
        detile(&p, &tiled, &mut back, w, h);
        assert_eq!(lin, back);
    }

    #[test]
    fn known_render_target_offsets() {
        // 257 x 129 x 4-byte SW_64KB_R_X surface: three 128 x 128 blocks per row.
        let p = DetileParams::new(SWIZZLE_64KB_R_X, 4, 257).unwrap();
        assert_eq!((p.offset(0, 0), p.offset(1, 0), p.offset(0, 1)), (0, 4, 16));
        assert_eq!(p.offset(128, 0), 65536);
        assert_eq!(p.offset(0, 128), 3 * 65536);
        assert_eq!(p.offset(256, 0), 2 * 65536);
        // Pipe/bank XOR: x bit 4 sets bits 9 and 13, y bit 3 bits 8 and 12.
        assert_eq!(p.offset(16, 0), 0x2200);
        assert_eq!(p.offset(0, 8), 0x1100);
        assert_eq!(render_target_offset(16, 0, 4), 0x2200);
        assert_eq!(render_target_offset(0, 8, 4), 0x1100);
        // 256 x 256 x 4-byte SW_4KB_S: 32 x 32 blocks, eight per row.
        let s = DetileParams::new(SWIZZLE_4KB_S, 4, 256).unwrap();
        assert_eq!((s.offset(1, 0), s.offset(0, 1)), (4, 16));
        assert_eq!((s.offset(32, 0), s.offset(0, 32)), (4096, 8 * 4096));
        assert_eq!(s.surface_size(256), 64 * 4096);
        // 3840 x 2160 x 4-byte render target: 30 x 17 blocks.
        let screen = DetileParams::new(SWIZZLE_64KB_R_X, 4, 3840).unwrap();
        for (mode, bpe, w, h) in [(SWIZZLE_64KB_R_X, 4, 3840, 2160), (SWIZZLE_4KB_S, 1, 77, 33), (SWIZZLE_LINEAR, 8, 100, 7), (SWIZZLE_64KB_Z_X, 2, 1921, 1081)] {
            let p = DetileParams::new(mode, bpe, w).unwrap();
            assert_eq!(thin_surface(mode, bpe, w, h), Some((p.pitch, p.surface_size(h))));
        }
        assert_eq!(screen.surface_size(2160), 33_423_360);
    }

    /// In-block offsets produced by the console's `libSceAgcGpuAddress`
    /// element-offset kernels (statically linked into titles), sampled for
    /// every supported mode and element size: `(mode, bytes, [(x, y, offset)])`.
    /// The coordinates reach past the block so the XOR terms are exercised.
    #[rustfmt::skip]
    const SYSTEM_TILER_VECTORS: &[(u32, usize, [(u32, u32, usize); 12])] = &[
        (1, 1, [(1, 0, 0x1), (0, 1, 0x10), (5, 3, 0x35), (16, 0, 0x0), (0, 8, 0x80), (37, 19, 0x35), (100, 77, 0xd4), (127, 63, 0xff), (129, 200, 0x81), (255, 255, 0xff), (300, 450, 0x2c), (511, 511, 0xff)]),
        (1, 2, [(1, 0, 0x2), (0, 1, 0x10), (5, 3, 0x3a), (16, 0, 0x0), (0, 8, 0x0), (37, 19, 0x3a), (100, 77, 0x58), (127, 63, 0xfe), (129, 200, 0x2), (255, 255, 0xfe), (300, 450, 0xa8), (511, 511, 0xfe)]),
        (1, 4, [(1, 0, 0x4), (0, 1, 0x10), (5, 3, 0xb4), (16, 0, 0x0), (0, 8, 0x0), (37, 19, 0xb4), (100, 77, 0xd0), (127, 63, 0xfc), (129, 200, 0x4), (255, 255, 0xfc), (300, 450, 0xa0), (511, 511, 0xfc)]),
        (1, 8, [(1, 0, 0x8), (0, 1, 0x10), (5, 3, 0xb8), (16, 0, 0x0), (0, 8, 0x0), (37, 19, 0xb8), (100, 77, 0x90), (127, 63, 0xf8), (129, 200, 0x8), (255, 255, 0xf8), (300, 450, 0xa0), (511, 511, 0xf8)]),
        (1, 16, [(1, 0, 0x40), (0, 1, 0x10), (5, 3, 0x70), (16, 0, 0x0), (0, 8, 0x0), (37, 19, 0x70), (100, 77, 0x10), (127, 63, 0xf0), (129, 200, 0x40), (255, 255, 0xf0), (300, 450, 0x20), (511, 511, 0xf0)]),
        (2, 1, [(1, 0, 0x1), (0, 1, 0x10), (5, 3, 0x1d), (16, 0, 0x0), (0, 8, 0x80), (37, 19, 0x1d), (100, 77, 0xb4), (127, 63, 0xff), (129, 200, 0x81), (255, 255, 0xff), (300, 450, 0x4c), (511, 511, 0xff)]),
        (2, 2, [(1, 0, 0x2), (0, 1, 0x10), (5, 3, 0x3a), (16, 0, 0x0), (0, 8, 0x0), (37, 19, 0x3a), (100, 77, 0x58), (127, 63, 0xfe), (129, 200, 0x2), (255, 255, 0xfe), (300, 450, 0xa8), (511, 511, 0xfe)]),
        (2, 4, [(1, 0, 0x4), (0, 1, 0x10), (5, 3, 0xb4), (16, 0, 0x0), (0, 8, 0x0), (37, 19, 0xb4), (100, 77, 0xd0), (127, 63, 0xfc), (129, 200, 0x4), (255, 255, 0xfc), (300, 450, 0xa0), (511, 511, 0xfc)]),
        (2, 8, [(1, 0, 0x8), (0, 1, 0x10), (5, 3, 0xd8), (16, 0, 0x0), (0, 8, 0x0), (37, 19, 0xd8), (100, 77, 0x50), (127, 63, 0xf8), (129, 200, 0x8), (255, 255, 0xf8), (300, 450, 0xc0), (511, 511, 0xf8)]),
        (2, 16, [(1, 0, 0x10), (0, 1, 0x20), (5, 3, 0xb0), (16, 0, 0x0), (0, 8, 0x0), (37, 19, 0xb0), (100, 77, 0x20), (127, 63, 0xf0), (129, 200, 0x10), (255, 255, 0xf0), (300, 450, 0x80), (511, 511, 0xf0)]),
        (5, 1, [(1, 0, 0x1), (0, 1, 0x10), (5, 3, 0x35), (16, 0, 0x200), (0, 8, 0x80), (37, 19, 0x935), (100, 77, 0x8d4), (127, 63, 0xfff), (129, 200, 0x81), (255, 255, 0xfff), (300, 450, 0x82c), (511, 511, 0xfff)]),
        (5, 2, [(1, 0, 0x2), (0, 1, 0x10), (5, 3, 0x3a), (16, 0, 0x200), (0, 8, 0x100), (37, 19, 0xc3a), (100, 77, 0x958), (127, 63, 0xffe), (129, 200, 0x102), (255, 255, 0xffe), (300, 450, 0x8a8), (511, 511, 0xffe)]),
        (5, 4, [(1, 0, 0x4), (0, 1, 0x10), (5, 3, 0xb4), (16, 0, 0x800), (0, 8, 0x100), (37, 19, 0x4b4), (100, 77, 0x1d0), (127, 63, 0xffc), (129, 200, 0x104), (255, 255, 0xffc), (300, 450, 0x2a0), (511, 511, 0xffc)]),
        (5, 8, [(1, 0, 0x8), (0, 1, 0x10), (5, 3, 0xb8), (16, 0, 0x800), (0, 8, 0x400), (37, 19, 0xb8), (100, 77, 0x590), (127, 63, 0xff8), (129, 200, 0x408), (255, 255, 0xff8), (300, 450, 0x2a0), (511, 511, 0xff8)]),
        (5, 16, [(1, 0, 0x40), (0, 1, 0x10), (5, 3, 0x270), (16, 0, 0x0), (0, 8, 0x400), (37, 19, 0x270), (100, 77, 0x710), (127, 63, 0xff0), (129, 200, 0x440), (255, 255, 0xff0), (300, 450, 0xa20), (511, 511, 0xff0)]),
        (6, 1, [(1, 0, 0x1), (0, 1, 0x10), (5, 3, 0x1d), (16, 0, 0x200), (0, 8, 0x80), (37, 19, 0x91d), (100, 77, 0x8b4), (127, 63, 0xfff), (129, 200, 0x81), (255, 255, 0xfff), (300, 450, 0x84c), (511, 511, 0xfff)]),
        (6, 2, [(1, 0, 0x2), (0, 1, 0x10), (5, 3, 0x3a), (16, 0, 0x200), (0, 8, 0x100), (37, 19, 0xc3a), (100, 77, 0x958), (127, 63, 0xffe), (129, 200, 0x102), (255, 255, 0xffe), (300, 450, 0x8a8), (511, 511, 0xffe)]),
        (6, 4, [(1, 0, 0x4), (0, 1, 0x10), (5, 3, 0xb4), (16, 0, 0x800), (0, 8, 0x100), (37, 19, 0x4b4), (100, 77, 0x1d0), (127, 63, 0xffc), (129, 200, 0x104), (255, 255, 0xffc), (300, 450, 0x2a0), (511, 511, 0xffc)]),
        (6, 8, [(1, 0, 0x8), (0, 1, 0x10), (5, 3, 0xd8), (16, 0, 0x800), (0, 8, 0x400), (37, 19, 0xd8), (100, 77, 0x550), (127, 63, 0xff8), (129, 200, 0x408), (255, 255, 0xff8), (300, 450, 0x2c0), (511, 511, 0xff8)]),
        (6, 16, [(1, 0, 0x10), (0, 1, 0x20), (5, 3, 0x2b0), (16, 0, 0x0), (0, 8, 0x400), (37, 19, 0x2b0), (100, 77, 0x720), (127, 63, 0xff0), (129, 200, 0x410), (255, 255, 0xff0), (300, 450, 0xa80), (511, 511, 0xff0)]),
        (9, 1, [(1, 0, 0x1), (0, 1, 0x10), (5, 3, 0x35), (16, 0, 0x200), (0, 8, 0x80), (37, 19, 0x935), (100, 77, 0x38d4), (127, 63, 0x2fff), (129, 200, 0xd081), (255, 255, 0xffff), (300, 450, 0x582c), (511, 511, 0xffff)]),
        (9, 2, [(1, 0, 0x2), (0, 1, 0x10), (5, 3, 0x3a), (16, 0, 0x200), (0, 8, 0x100), (37, 19, 0xc3a), (100, 77, 0x6958), (127, 63, 0x3ffe), (129, 200, 0xc102), (255, 255, 0xfffe), (300, 450, 0x48a8), (511, 511, 0xfffe)]),
        (9, 4, [(1, 0, 0x4), (0, 1, 0x10), (5, 3, 0xb4), (16, 0, 0x800), (0, 8, 0x100), (37, 19, 0x24b4), (100, 77, 0xe1d0), (127, 63, 0xbffc), (129, 200, 0x4104), (255, 255, 0xfffc), (300, 450, 0x62a0), (511, 511, 0xfffc)]),
        (9, 8, [(1, 0, 0x8), (0, 1, 0x10), (5, 3, 0xb8), (16, 0, 0x800), (0, 8, 0x400), (37, 19, 0x30b8), (100, 77, 0xa590), (127, 63, 0xfff8), (129, 200, 0x408), (255, 255, 0xfff8), (300, 450, 0x22a0), (511, 511, 0xfff8)]),
        (9, 16, [(1, 0, 0x40), (0, 1, 0x10), (5, 3, 0x270), (16, 0, 0x2000), (0, 8, 0x400), (37, 19, 0x9270), (100, 77, 0x8710), (127, 63, 0xfff0), (129, 200, 0x440), (255, 255, 0xfff0), (300, 450, 0x8a20), (511, 511, 0xfff0)]),
        (10, 1, [(1, 0, 0x1), (0, 1, 0x10), (5, 3, 0x1d), (16, 0, 0x200), (0, 8, 0x80), (37, 19, 0x91d), (100, 77, 0x38b4), (127, 63, 0x2fff), (129, 200, 0xd081), (255, 255, 0xffff), (300, 450, 0x584c), (511, 511, 0xffff)]),
        (10, 2, [(1, 0, 0x2), (0, 1, 0x10), (5, 3, 0x3a), (16, 0, 0x200), (0, 8, 0x100), (37, 19, 0xc3a), (100, 77, 0x6958), (127, 63, 0x3ffe), (129, 200, 0xc102), (255, 255, 0xfffe), (300, 450, 0x48a8), (511, 511, 0xfffe)]),
        (10, 4, [(1, 0, 0x4), (0, 1, 0x10), (5, 3, 0xb4), (16, 0, 0x800), (0, 8, 0x100), (37, 19, 0x24b4), (100, 77, 0xe1d0), (127, 63, 0xbffc), (129, 200, 0x4104), (255, 255, 0xfffc), (300, 450, 0x62a0), (511, 511, 0xfffc)]),
        (10, 8, [(1, 0, 0x8), (0, 1, 0x10), (5, 3, 0xd8), (16, 0, 0x800), (0, 8, 0x400), (37, 19, 0x30d8), (100, 77, 0xa550), (127, 63, 0xfff8), (129, 200, 0x408), (255, 255, 0xfff8), (300, 450, 0x22c0), (511, 511, 0xfff8)]),
        (10, 16, [(1, 0, 0x10), (0, 1, 0x20), (5, 3, 0x2b0), (16, 0, 0x2000), (0, 8, 0x400), (37, 19, 0x92b0), (100, 77, 0x8720), (127, 63, 0xfff0), (129, 200, 0x410), (255, 255, 0xfff0), (300, 450, 0x8a80), (511, 511, 0xfff0)]),
        (17, 1, [(1, 0, 0x1), (0, 1, 0x10), (5, 3, 0x35), (16, 0, 0x200), (0, 8, 0x80), (37, 19, 0x935), (100, 77, 0x34d4), (127, 63, 0x2bff), (129, 200, 0xdb81), (255, 255, 0xf0ff), (300, 450, 0x522c), (511, 511, 0xf0ff)]),
        (17, 2, [(1, 0, 0x2), (0, 1, 0x10), (5, 3, 0x3a), (16, 0, 0x200), (0, 8, 0x100), (37, 19, 0xc3a), (100, 77, 0x6f58), (127, 63, 0x33fe), (129, 200, 0xc202), (255, 255, 0xf0fe), (300, 450, 0x4aa8), (511, 511, 0xf0fe)]),
        (17, 4, [(1, 0, 0x4), (0, 1, 0x10), (5, 3, 0xb4), (16, 0, 0x800), (0, 8, 0x100), (37, 19, 0x20b4), (100, 77, 0xe6d0), (127, 63, 0xb2fc), (129, 200, 0x4304), (255, 255, 0xf0fc), (300, 450, 0x64a0), (511, 511, 0xf0fc)]),
        (17, 8, [(1, 0, 0x8), (0, 1, 0x10), (5, 3, 0xb8), (16, 0, 0x800), (0, 8, 0x400), (37, 19, 0x3cb8), (100, 77, 0xa090), (127, 63, 0xf0f8), (129, 200, 0x408), (255, 255, 0xf0f8), (300, 450, 0x26a0), (511, 511, 0xf0f8)]),
        (17, 16, [(1, 0, 0x40), (0, 1, 0x10), (5, 3, 0x270), (16, 0, 0x2400), (0, 8, 0x400), (37, 19, 0x9b70), (100, 77, 0x8610), (127, 63, 0xf0f0), (129, 200, 0x440), (255, 255, 0xf0f0), (300, 450, 0x8b20), (511, 511, 0xf0f0)]),
        (18, 1, [(1, 0, 0x1), (0, 1, 0x10), (5, 3, 0x1d), (16, 0, 0x200), (0, 8, 0x80), (37, 19, 0x91d), (100, 77, 0x34b4), (127, 63, 0x2bff), (129, 200, 0xdb81), (255, 255, 0xf0ff), (300, 450, 0x524c), (511, 511, 0xf0ff)]),
        (18, 2, [(1, 0, 0x2), (0, 1, 0x10), (5, 3, 0x3a), (16, 0, 0x200), (0, 8, 0x100), (37, 19, 0xc3a), (100, 77, 0x6f58), (127, 63, 0x33fe), (129, 200, 0xc202), (255, 255, 0xf0fe), (300, 450, 0x4aa8), (511, 511, 0xf0fe)]),
        (18, 4, [(1, 0, 0x4), (0, 1, 0x10), (5, 3, 0xb4), (16, 0, 0x800), (0, 8, 0x100), (37, 19, 0x20b4), (100, 77, 0xe6d0), (127, 63, 0xb2fc), (129, 200, 0x4304), (255, 255, 0xf0fc), (300, 450, 0x64a0), (511, 511, 0xf0fc)]),
        (18, 8, [(1, 0, 0x8), (0, 1, 0x10), (5, 3, 0xd8), (16, 0, 0x800), (0, 8, 0x400), (37, 19, 0x3cd8), (100, 77, 0xa050), (127, 63, 0xf0f8), (129, 200, 0x408), (255, 255, 0xf0f8), (300, 450, 0x26c0), (511, 511, 0xf0f8)]),
        (18, 16, [(1, 0, 0x10), (0, 1, 0x20), (5, 3, 0x2b0), (16, 0, 0x2400), (0, 8, 0x400), (37, 19, 0x9bb0), (100, 77, 0x8620), (127, 63, 0xf0f0), (129, 200, 0x410), (255, 255, 0xf0f0), (300, 450, 0x8b80), (511, 511, 0xf0f0)]),
        (21, 1, [(1, 0, 0x1), (0, 1, 0x10), (5, 3, 0x35), (16, 0, 0x200), (0, 8, 0x80), (37, 19, 0x935), (100, 77, 0x4d4), (127, 63, 0xbff), (129, 200, 0xb81), (255, 255, 0xff), (300, 450, 0x22c), (511, 511, 0xff)]),
        (21, 2, [(1, 0, 0x2), (0, 1, 0x10), (5, 3, 0x3a), (16, 0, 0x200), (0, 8, 0x100), (37, 19, 0xc3a), (100, 77, 0xf58), (127, 63, 0x3fe), (129, 200, 0x202), (255, 255, 0xfe), (300, 450, 0xaa8), (511, 511, 0xfe)]),
        (21, 4, [(1, 0, 0x4), (0, 1, 0x10), (5, 3, 0xb4), (16, 0, 0x800), (0, 8, 0x100), (37, 19, 0xb4), (100, 77, 0x6d0), (127, 63, 0x2fc), (129, 200, 0x304), (255, 255, 0xfc), (300, 450, 0x4a0), (511, 511, 0xfc)]),
        (21, 8, [(1, 0, 0x8), (0, 1, 0x10), (5, 3, 0xb8), (16, 0, 0x800), (0, 8, 0x400), (37, 19, 0xcb8), (100, 77, 0x90), (127, 63, 0xf8), (129, 200, 0x408), (255, 255, 0xf8), (300, 450, 0x6a0), (511, 511, 0xf8)]),
        (21, 16, [(1, 0, 0x40), (0, 1, 0x10), (5, 3, 0x270), (16, 0, 0x400), (0, 8, 0x400), (37, 19, 0xb70), (100, 77, 0x610), (127, 63, 0xf0), (129, 200, 0x440), (255, 255, 0xf0), (300, 450, 0xb20), (511, 511, 0xf0)]),
        (22, 1, [(1, 0, 0x1), (0, 1, 0x10), (5, 3, 0x1d), (16, 0, 0x200), (0, 8, 0x80), (37, 19, 0x91d), (100, 77, 0x4b4), (127, 63, 0xbff), (129, 200, 0xb81), (255, 255, 0xff), (300, 450, 0x24c), (511, 511, 0xff)]),
        (22, 2, [(1, 0, 0x2), (0, 1, 0x10), (5, 3, 0x3a), (16, 0, 0x200), (0, 8, 0x100), (37, 19, 0xc3a), (100, 77, 0xf58), (127, 63, 0x3fe), (129, 200, 0x202), (255, 255, 0xfe), (300, 450, 0xaa8), (511, 511, 0xfe)]),
        (22, 4, [(1, 0, 0x4), (0, 1, 0x10), (5, 3, 0xb4), (16, 0, 0x800), (0, 8, 0x100), (37, 19, 0xb4), (100, 77, 0x6d0), (127, 63, 0x2fc), (129, 200, 0x304), (255, 255, 0xfc), (300, 450, 0x4a0), (511, 511, 0xfc)]),
        (22, 8, [(1, 0, 0x8), (0, 1, 0x10), (5, 3, 0xd8), (16, 0, 0x800), (0, 8, 0x400), (37, 19, 0xcd8), (100, 77, 0x50), (127, 63, 0xf8), (129, 200, 0x408), (255, 255, 0xf8), (300, 450, 0x6c0), (511, 511, 0xf8)]),
        (22, 16, [(1, 0, 0x10), (0, 1, 0x20), (5, 3, 0x2b0), (16, 0, 0x400), (0, 8, 0x400), (37, 19, 0xbb0), (100, 77, 0x620), (127, 63, 0xf0), (129, 200, 0x410), (255, 255, 0xf0), (300, 450, 0xb80), (511, 511, 0xf0)]),
        (24, 1, [(1, 0, 0x1), (0, 1, 0x2), (5, 3, 0x1b), (16, 0, 0x200), (0, 8, 0x100), (37, 19, 0xa9b), (100, 77, 0x3532), (127, 63, 0x28ff), (129, 200, 0xd901), (255, 255, 0xf0ff), (300, 450, 0x5158), (511, 511, 0xf0ff)]),
        (24, 2, [(1, 0, 0x2), (0, 1, 0x4), (5, 3, 0x36), (16, 0, 0x200), (0, 8, 0x100), (37, 19, 0x1a36), (100, 77, 0x6564), (127, 63, 0x38fe), (129, 200, 0xc902), (255, 255, 0xf0fe), (300, 450, 0x41b0), (511, 511, 0xf0fe)]),
        (24, 4, [(1, 0, 0x4), (0, 1, 0x8), (5, 3, 0x6c), (16, 0, 0x2200), (0, 8, 0x1100), (37, 19, 0xa6c), (100, 77, 0xd5c8), (127, 63, 0xb8fc), (129, 200, 0x5904), (255, 255, 0xf0fc), (300, 450, 0x4160), (511, 511, 0xf0fc)]),
        (24, 8, [(1, 0, 0x8), (0, 1, 0x10), (5, 3, 0xd8), (16, 0, 0x200), (0, 8, 0x100), (37, 19, 0x4ad8), (100, 77, 0x9590), (127, 63, 0xf8f8), (129, 200, 0x908), (255, 255, 0xf0f8), (300, 450, 0x21c0), (511, 511, 0xf0f8)]),
        (24, 16, [(1, 0, 0x10), (0, 1, 0x20), (5, 3, 0x20b0), (16, 0, 0x8200), (0, 8, 0x4100), (37, 19, 0x2ab0), (100, 77, 0x7520), (127, 63, 0xf8f0), (129, 200, 0x4910), (255, 255, 0xf0f0), (300, 450, 0x2180), (511, 511, 0xf0f0)]),
        (25, 1, [(1, 0, 0x1), (0, 1, 0x10), (5, 3, 0x35), (16, 0, 0x200), (0, 8, 0x80), (37, 19, 0x935), (100, 77, 0x34d4), (127, 63, 0x2bff), (129, 200, 0xdb81), (255, 255, 0xf0ff), (300, 450, 0x522c), (511, 511, 0xf0ff)]),
        (25, 2, [(1, 0, 0x2), (0, 1, 0x10), (5, 3, 0x3a), (16, 0, 0x200), (0, 8, 0x100), (37, 19, 0xc3a), (100, 77, 0x6f58), (127, 63, 0x33fe), (129, 200, 0xc202), (255, 255, 0xf0fe), (300, 450, 0x4aa8), (511, 511, 0xf0fe)]),
        (25, 4, [(1, 0, 0x4), (0, 1, 0x10), (5, 3, 0xb4), (16, 0, 0x800), (0, 8, 0x100), (37, 19, 0x20b4), (100, 77, 0xe6d0), (127, 63, 0xb2fc), (129, 200, 0x4304), (255, 255, 0xf0fc), (300, 450, 0x64a0), (511, 511, 0xf0fc)]),
        (25, 8, [(1, 0, 0x8), (0, 1, 0x10), (5, 3, 0xb8), (16, 0, 0x800), (0, 8, 0x400), (37, 19, 0x3cb8), (100, 77, 0xa090), (127, 63, 0xf0f8), (129, 200, 0x408), (255, 255, 0xf0f8), (300, 450, 0x26a0), (511, 511, 0xf0f8)]),
        (25, 16, [(1, 0, 0x40), (0, 1, 0x10), (5, 3, 0x270), (16, 0, 0x2400), (0, 8, 0x400), (37, 19, 0x9b70), (100, 77, 0x8610), (127, 63, 0xf0f0), (129, 200, 0x440), (255, 255, 0xf0f0), (300, 450, 0x8b20), (511, 511, 0xf0f0)]),
        (26, 1, [(1, 0, 0x1), (0, 1, 0x10), (5, 3, 0x1d), (16, 0, 0x200), (0, 8, 0x80), (37, 19, 0x91d), (100, 77, 0x34b4), (127, 63, 0x2bff), (129, 200, 0xdb81), (255, 255, 0xf0ff), (300, 450, 0x524c), (511, 511, 0xf0ff)]),
        (26, 2, [(1, 0, 0x2), (0, 1, 0x10), (5, 3, 0x3a), (16, 0, 0x200), (0, 8, 0x100), (37, 19, 0xc3a), (100, 77, 0x6f58), (127, 63, 0x33fe), (129, 200, 0xc202), (255, 255, 0xf0fe), (300, 450, 0x4aa8), (511, 511, 0xf0fe)]),
        (26, 4, [(1, 0, 0x4), (0, 1, 0x10), (5, 3, 0xb4), (16, 0, 0x800), (0, 8, 0x100), (37, 19, 0x20b4), (100, 77, 0xe6d0), (127, 63, 0xb2fc), (129, 200, 0x4304), (255, 255, 0xf0fc), (300, 450, 0x64a0), (511, 511, 0xf0fc)]),
        (26, 8, [(1, 0, 0x8), (0, 1, 0x10), (5, 3, 0xd8), (16, 0, 0x800), (0, 8, 0x400), (37, 19, 0x3cd8), (100, 77, 0xa050), (127, 63, 0xf0f8), (129, 200, 0x408), (255, 255, 0xf0f8), (300, 450, 0x26c0), (511, 511, 0xf0f8)]),
        (26, 16, [(1, 0, 0x10), (0, 1, 0x20), (5, 3, 0x2b0), (16, 0, 0x2400), (0, 8, 0x400), (37, 19, 0x9bb0), (100, 77, 0x8620), (127, 63, 0xf0f0), (129, 200, 0x410), (255, 255, 0xf0f0), (300, 450, 0x8b80), (511, 511, 0xf0f0)]),
        (27, 1, [(1, 0, 0x1), (0, 1, 0x10), (5, 3, 0x1d), (16, 0, 0x200), (0, 8, 0x100), (37, 19, 0xa9d), (100, 77, 0x3534), (127, 63, 0x28ff), (129, 200, 0xd901), (255, 255, 0xf0ff), (300, 450, 0x514c), (511, 511, 0xf0ff)]),
        (27, 2, [(1, 0, 0x2), (0, 1, 0x10), (5, 3, 0x3a), (16, 0, 0x200), (0, 8, 0x100), (37, 19, 0x1a3a), (100, 77, 0x6558), (127, 63, 0x38fe), (129, 200, 0xc902), (255, 255, 0xf0fe), (300, 450, 0x41a8), (511, 511, 0xf0fe)]),
        (27, 4, [(1, 0, 0x4), (0, 1, 0x10), (5, 3, 0xb4), (16, 0, 0x2200), (0, 8, 0x1100), (37, 19, 0xab4), (100, 77, 0xd5d0), (127, 63, 0xb8fc), (129, 200, 0x5904), (255, 255, 0xf0fc), (300, 450, 0x41a0), (511, 511, 0xf0fc)]),
        (27, 8, [(1, 0, 0x8), (0, 1, 0x10), (5, 3, 0xd8), (16, 0, 0x200), (0, 8, 0x100), (37, 19, 0x4ad8), (100, 77, 0x9550), (127, 63, 0xf8f8), (129, 200, 0x908), (255, 255, 0xf0f8), (300, 450, 0x21c0), (511, 511, 0xf0f8)]),
        (27, 16, [(1, 0, 0x10), (0, 1, 0x20), (5, 3, 0x20b0), (16, 0, 0x8200), (0, 8, 0x4100), (37, 19, 0x2ab0), (100, 77, 0x7520), (127, 63, 0xf8f0), (129, 200, 0x4910), (255, 255, 0xf0f0), (300, 450, 0x2180), (511, 511, 0xf0f0)]),
    ];

    #[test]
    fn matches_the_system_tiler() {
        assert_eq!(SYSTEM_TILER_VECTORS.len(), TILED_MODES.len() * 5);
        for &(mode, bpe, ref points) in SYSTEM_TILER_VECTORS {
            let p = DetileParams::new(mode, bpe, 4096).unwrap();
            for &(x, y, want) in points {
                assert_eq!(p.offset(x, y) % p.block_bytes, want, "mode={mode} bpe={bpe} ({x}, {y})");
            }
        }
    }
}
