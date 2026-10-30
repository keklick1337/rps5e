// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! DCC keys of surfaces read through image descriptors.
//!
//! A T# with COMPRESSION_EN (word 6 bit 21) reads its surface through the
//! DCC keys at META_DATA_ADDRESS. A surface fast-cleared by its keys (a
//! fill of a clear code) and then only sampled — never bound as a colour
//! target again — reads the clear colour on the hardware, not the bytes
//! in memory. Keys of 0x10 (comp-to-single) say each 256-byte block of the
//! surface holds one value, stored once as its first element.
//!
//! The keys are decided where a colour-target bind decides them: a pending
//! fill of the keys, or, when the image is loaded from guest memory, the
//! key bytes themselves (fetched from the GPU when it wrote them). Only
//! uniform keys are modelled. A fixed clear code (0x00, 0x40, 0x80, 0xC0)
//! clears the image; comp-to-single keys expand every block's first
//! element over the block in guest memory before the load. Either way the
//! keys are then spent (0xFF, uncompressed): the image, or memory, holds
//! what they described. The register clear (0x20) needs the target's
//! CB_COLOR_CLEAR_WORD, which a descriptor does not carry: such surfaces
//! read as stored.
//!
//! `RPS5E_SAMPLED_DCC=0` turns this off.

use super::ImageCache;
use crate::buffers::BufferCache;
use crate::scheduler::Scheduler;

/// Bytes of surface one DCC key byte describes.
pub const KEY_BLOCK: u64 = 256;
/// The comp-to-single key.
pub const SINGLE: u8 = 0x10;
/// The uncompressed key.
pub const UNCOMPRESSED: u8 = 0xFF;

pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_SAMPLED_DCC").map_or(true, |v| v != "0"))
}

/// The DCC metadata of an image descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescriptorDcc {
    /// META_DATA_ADDRESS.
    pub address: u64,
    /// ALPHA_IS_ON_MSB (word 6 bit 22).
    pub alpha_on_msb: bool,
}

/// The DCC metadata of the T# `words` (None without COMPRESSION_EN or
/// without a metadata address).
pub fn descriptor_dcc(words: &[u32]) -> Option<DescriptorDcc> {
    let (w6, w7) = (*words.get(6)?, *words.get(7)?);
    if (w6 >> 21) & 1 == 0 {
        return None;
    }
    let address = (((w6 >> 24) as u64) | ((w7 as u64) << 8)) << 8;
    (address != 0).then_some(DescriptorDcc { address, alpha_on_msb: (w6 >> 22) & 1 != 0 })
}

/// Bytes of keys of a surface of `bytes` (one per 256 surface bytes; the
/// 64 KiB render-target layout keeps whole 4 KiB metadata blocks per
/// layer, `slice` bytes each).
pub fn key_bytes(surface_bytes: u64, slice: u64, layers: u32) -> u64 {
    if slice != 0 {
        slice * layers.max(1) as u64
    } else {
        surface_bytes / KEY_BLOCK
    }
}

/// The key every byte of `keys` holds (None when they differ or there are
/// none).
pub fn uniform_key(keys: &[u8]) -> Option<u8> {
    let (&first, rest) = keys.split_first()?;
    rest.iter().all(|&k| k == first).then_some(first)
}

/// `RPS5E_DCC_IN_PLACE=0`: comp-to-single blocks are expanded in a copy of
/// the surface written back afterwards instead of in guest memory itself
/// (A/B scope `release` toggles it; the bytes are the same).
fn in_place_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_DCC_IN_PLACE")) && crate::backend::ab::release_fast("RPS5E_DCC_IN_PLACE")
}

/// Comp-to-single blocks: the first element of each 256-byte block of
/// `bytes` is repeated over the block.
pub fn expand_single_blocks(bytes: &mut [u8], element_bytes: usize) {
    if element_bytes == 0 || KEY_BLOCK as usize % element_bytes != 0 {
        return;
    }
    // Element sizes known at compile time copy as plain stores.
    match element_bytes {
        1 => return expand_blocks_of::<1>(bytes),
        2 => return expand_blocks_of::<2>(bytes),
        4 => return expand_blocks_of::<4>(bytes),
        8 => return expand_blocks_of::<8>(bytes),
        16 => return expand_blocks_of::<16>(bytes),
        _ => {}
    }
    expand_blocks_any(bytes, element_bytes);
}

fn expand_blocks_any(bytes: &mut [u8], element_bytes: usize) {
    for block in bytes.chunks_mut(KEY_BLOCK as usize) {
        let n = element_bytes.min(block.len());
        let (first, rest) = block.split_at_mut(n);
        for e in rest.chunks_mut(n) {
            e.copy_from_slice(&first[..e.len()]);
        }
    }
}

/// [`expand_blocks_any`] for `N`-byte elements.
fn expand_blocks_of<const N: usize>(bytes: &mut [u8]) {
    for block in bytes.chunks_mut(KEY_BLOCK as usize) {
        if block.len() <= N {
            continue;
        }
        let (first, rest) = block.split_at_mut(N);
        let first: [u8; N] = (*first).try_into().expect("N bytes");
        let mut whole = rest.chunks_exact_mut(N);
        for e in &mut whole {
            e.copy_from_slice(&first);
        }
        let tail = whole.into_remainder();
        let n = tail.len();
        tail.copy_from_slice(&first[..n]);
    }
}

/// What the keys of a sampled surface asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampledKeys {
    /// Nothing to do (no clear code, mixed keys, keys not read).
    Stored,
    /// The image was cleared to the colour of this code.
    Cleared(u8),
    /// Comp-to-single blocks were expanded in guest memory.
    Expanded,
}

impl ImageCache {
    /// Decide the DCC keys of image `id` read through a descriptor with
    /// `dcc` (see the module documentation). Call before the image is
    /// prepared for the read.
    pub fn sampled_dcc(&mut self, sched: &mut Scheduler, buffers: &mut BufferCache, id: u64, dcc: DescriptorDcc) -> SampledKeys {
        let Some(img) = self.images.get(&id) else { return SampledKeys::Stored };
        let key = img.key;
        if key.is_depth() || key.samples != 1 {
            return SampledKeys::Stored;
        }
        if self.sampled_dcc_owners.len() >= 4096 {
            self.sampled_dcc_owners.clear();
        }
        self.sampled_dcc_owners.insert(dcc.address, id);
        let slice = crate::render::dcc_slice_bytes(key.width, key.height, key.main.bytes_per_element, key.main.swizzle, key.samples, key.levels);
        let bytes = key_bytes(key.main.bytes, slice, key.slices());
        if bytes == 0 {
            return SampledKeys::Stored;
        }
        let fill = self.metadata.get(&dcc.address).copied();
        let code = if let Some(fill) = fill {
            let b = fill.value as u8;
            if fill.value != u32::from_ne_bytes([b; 4]) || fill.size < bytes {
                return SampledKeys::Stored;
            }
            b
        } else {
            if !self.needs_upload(id) {
                return SampledKeys::Stored;
            }
            // Keys a GPU fill wrote last hold its value: no download.
            let filled = buffers.has_gpu_dirty(dcc.address, bytes).then(|| buffers.known_fill(dcc.address, bytes));
            match filled {
                Some(Some(v)) => {
                    let b = v as u8;
                    if v != u32::from_ne_bytes([b; 4]) || !(b == SINGLE || matches!(b, 0x00 | 0x40 | 0x80 | 0xC0)) {
                        return SampledKeys::Stored;
                    }
                    self.stats.known_fill_keys += 1;
                    b
                }
                _ => {
                    if filled.is_some() {
                        crate::log_once(&format!("ps5gpu: sampled DCC keys {:#x} written by the GPU: fetched", dcc.address));
                        if !buffers.download(sched, dcc.address, bytes) {
                            return SampledKeys::Stored;
                        }
                    }
                    // The head first: most surfaces read here are not compressed.
                    let head = bytes.min(64);
                    let mut first = vec![0u8; head as usize];
                    if !self.memory.read_backing(dcc.address, &mut first) {
                        return SampledKeys::Stored;
                    }
                    let Some(code) = uniform_key(&first).filter(|&c| c == SINGLE || matches!(c, 0x00 | 0x40 | 0x80 | 0xC0)) else { return SampledKeys::Stored };
                    let mut all = vec![0u8; bytes as usize];
                    if !self.memory.read_backing(dcc.address, &mut all) || uniform_key(&all) != Some(code) {
                        return SampledKeys::Stored;
                    }
                    code
                }
            }
        };
        let done = match code {
            SINGLE => {
                let element = key.main.bytes_per_element as usize;
                if element == 0 || KEY_BLOCK as usize % element != 0 {
                    return SampledKeys::Stored;
                }
                let in_place = in_place_enabled() && self.memory.modify_backing(key.main.addr, key.main.bytes, |b| expand_single_blocks(b, element));
                if !in_place {
                    let mut texels = vec![0u8; key.main.bytes as usize];
                    if !self.memory.read_backing(key.main.addr, &mut texels) {
                        return SampledKeys::Stored;
                    }
                    expand_single_blocks(&mut texels, element);
                    if !self.memory.write_backing(key.main.addr, &texels) {
                        return SampledKeys::Stored;
                    }
                }
                buffers.tracker.mark_cpu_dirty_pages(key.main.addr, key.main.bytes);
                self.invalidate_range(key.main.addr, key.main.bytes);
                SampledKeys::Expanded
            }
            _ => {
                let alpha = dcc.alpha_on_msb;
                let Some(value) = crate::render::formats::decode_dcc_clear(code, key.format, None, alpha) else { return SampledKeys::Stored };
                self.prepare(sched, buffers, id, true);
                self.clear_color(sched, id, value, None);
                SampledKeys::Cleared(code)
            }
        };
        self.metadata.remove(&dcc.address);
        self.metadata_generation += 1;
        spend_keys(self.memory, buffers, sched, dcc.address, bytes);
        done
    }
}

/// The keys of `[addr, addr+bytes)` become uncompressed (0xFF): on the GPU
/// when it holds the newest bytes, else in guest memory.
fn spend_keys(memory: &crate::memory::GuestGpuMemory, buffers: &mut BufferCache, sched: &mut Scheduler, addr: u64, bytes: u64) {
    if bytes % 4 == 0 && buffers.has_gpu_dirty(addr, bytes) {
        match buffers.fill(sched, addr, bytes, u32::MAX) {
            Ok(true) => return,
            Ok(false) => {}
            Err(e) => crate::log_once(&format!("ps5gpu: sampled DCC key fill {addr:#x}+{bytes:#x}: {e:?}")),
        }
    }
    if memory.write_backing(addr, &vec![UNCOMPRESSED; bytes as usize]) {
        buffers.tracker.mark_cpu_dirty_pages(addr, bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_metadata_fields() {
        let mut w = [0u32; 8];
        assert_eq!(descriptor_dcc(&w), None);
        w[6] = 1 << 21 | 1 << 22 | 0x34 << 24;
        w[7] = 0x12;
        assert_eq!(descriptor_dcc(&w), Some(DescriptorDcc { address: 0x1234 << 8, alpha_on_msb: true }));
        w[6] &= !(1 << 21);
        assert_eq!(descriptor_dcc(&w), None, "COMPRESSION_EN off");
    }

    #[test]
    fn single_blocks_repeat_their_first_element() {
        let mut b: Vec<u8> = (0..512u32).map(|i| (i % 251) as u8).collect();
        expand_single_blocks(&mut b, 4);
        assert!(b[..256].chunks(4).all(|e| e == [0, 1, 2, 3]));
        assert!(b[256..].chunks(4).all(|e| e == [5, 6, 7, 8]), "block 2 starts at byte 256");
        let mut b: Vec<u8> = (0..256u32).map(|i| (i / 8) as u8 + 1).collect();
        b[0..8].copy_from_slice(&[9, 8, 7, 6, 5, 4, 3, 2]);
        expand_single_blocks(&mut b, 8);
        assert!(b.chunks(8).all(|e| e == [9, 8, 7, 6, 5, 4, 3, 2]));
        assert_eq!(uniform_key(&[0x10; 5]), Some(0x10));
        assert_eq!(uniform_key(&[0x10, 0xFF]), None);
        assert_eq!(key_bytes(64 * 64 * 4, 0, 1), 64);
    }

    /// The fixed-size expansions write the bytes of the general one, for
    /// whole and partial last blocks.
    #[test]
    fn fixed_size_expansion_matches_the_general_one() {
        for element in [1usize, 2, 4, 8, 16] {
            for len in [0usize, 1, 3, 16, 17, 255, 256, 257, 300, 1024, 1029] {
                let src: Vec<u8> = (0..len as u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
                let (mut a, mut b) = (src.clone(), src.clone());
                expand_single_blocks(&mut a, element);
                expand_blocks_any(&mut b, element);
                assert_eq!(a, b, "element {element}, {len} bytes");
            }
        }
    }
}
