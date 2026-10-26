// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The SDK image-copy kernel, recognised so a dispatch of it runs as an
//! image-to-image copy on the GPU timeline (no shader, no descriptors).
//!
//! The kernel (8x8x1 threads, user data s[0:7] source T#, s[8:15]
//! destination T#, work-group ids x / y in s16 / s17) loads texel
//! `(group.x * 8 + tid.x, group.y * 8 + tid.y)` of the source with
//! `image_load` (dmask 0xF, 2D) and stores it at the same coordinate of the
//! destination with `image_store`. Loads past the source's extent read 0,
//! stores past the destination's are dropped.
//!
//! The copy runs only when it reproduces the kernel's bits exactly: both
//! T#s plain 2D, the same format whose load/store round trip is the
//! identity (unorm, uint, sint, 16/32-bit float), destination swizzles that
//! store every channel of the source unchanged, and a copied area inside
//! the source (otherwise part of the destination would take the zeros of
//! out-of-bounds loads). Everything else runs as the dispatch.

use crate::buffers::BufferCache;
use crate::images::ImageCache;
use crate::pipeline::sharps::{image_type, texel_format, texture_key, TextureSharp};
use crate::render::ResolvedDispatch;
use crate::scheduler::Scheduler;

/// The kernel's code.
pub const IMAGE_COPY_KERNEL: [u32; 10] = [
    0xd746_0004, 0x0401_0610, // v_lshl_add_u32 v4, s16, 3, v0
    0xd746_0005, 0x0405_0611, // v_lshl_add_u32 v5, s17, 3, v1
    0xf000_0f08, 0x0000_0004, // image_load v[0:3], v[4:5], s[0:7] dmask 0xF 2D
    0xbf8c_3f70, // s_waitcnt vmcnt(0)
    0xf020_0f08, 0x0002_0004, // image_store v[0:3], v[4:5], s[8:15] dmask 0xF 2D
    0xbf81_0000, // s_endpgm
];

/// User SGPRs of the kernel (the two T#s); the work-group ids follow.
pub const USER_SGPRS: u32 = 16;

/// Whether `code` (at least the kernel's length) and the compute stage
/// (`threads` per group, `user_sgprs` user SGPRs, RSRC2) are the copy
/// kernel launched as it expects: 8x8x1 threads, work-group ids x and y
/// right after the 16 user SGPRs (s16 / s17), thread ids x and y in v0 / v1.
pub fn matches_image_copy(code: &[u32], threads: [u32; 3], user_sgprs: u32, rsrc2: u32) -> bool {
    let group_x = rsrc2 >> 7 & 1 != 0;
    let group_y = rsrc2 >> 8 & 1 != 0;
    let thread_id_components = (rsrc2 >> 11 & 3) + 1;
    if threads != [8, 8, 1] || user_sgprs != USER_SGPRS || !group_x || !group_y || thread_id_components < 2 {
        return false;
    }
    code.len() >= IMAGE_COPY_KERNEL.len() && code[..IMAGE_COPY_KERNEL.len()] == IMAGE_COPY_KERNEL
}

/// Unified image formats whose `image_load` / `image_store` round trip
/// gives back the stored bits: unorm (k / (2^n - 1) converts back to k),
/// integers, 16- and 32-bit floats. Not snorm (the most negative code
/// reads as -1.0 and stores as the code above it), sRGB, scaled, shared
/// exponent, the small floats, packed 16-bit formats or block-compressed
/// ones.
pub fn exact_round_trip(format: u32) -> bool {
    matches!(
        format,
        // Unorm: 8, 16, 8_8, 16_16, 2_10_10_10, 8_8_8_8, 16_16_16_16.
        1 | 7 | 14 | 23 | 50 | 56 | 65
        // Uint.
        | 5 | 11 | 18 | 20 | 27 | 54 | 60 | 62 | 69 | 75
        // Sint.
        | 6 | 12 | 19 | 21 | 28 | 61 | 63 | 70 | 76
        // Float: 16, 32, 16_16, 32_32, 16_16_16_16, 32_32_32_32.
        | 13 | 22 | 29 | 64 | 71 | 77
    )
}

/// Components of a unified image format (0 when unknown).
fn components(format: u32) -> u32 {
    use ps5shader::resources::format::unified_format;
    unified_format::decode(format).map_or(0, |(data, _)| unified_format::component_count(data))
}

/// Whether loading through DST_SEL `load` and storing through DST_SEL
/// `store` writes each of the first `components` channels of the source
/// into the same channel of the destination. A load returns component `k`
/// as the source channel its selector names (4..7) or a constant; a store
/// writes channel `c` from the lowest component whose selector is `4 + c`.
pub fn swizzles_copy_channels(load: u32, store: u32, components: u32) -> bool {
    let sel = |sw: u32, k: u32| sw >> (3 * k) & 7;
    components != 0
        && (0..components).all(|c| match (0..4).find(|&k| sel(store, k) == 4 + c) {
            Some(k) => sel(load, k) == 4 + c,
            None => false,
        })
}

/// The copy a dispatch of the kernel performs: the `width` x `height`
/// texels at the origin of level `src_level` of the source go to the same
/// texels of level `dst_level` of the destination (array slice 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageCopy {
    pub src_level: u32,
    pub dst_level: u32,
    pub width: u32,
    pub height: u32,
}

impl ImageCopy {
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// The level an `image_load` / `image_store` without a mip operand reads
/// or writes (BASE_LEVEL) and that level's extent; None when BASE_LEVEL is
/// past LAST_LEVEL or past the surface's levels.
fn base_level_extent(t: &TextureSharp) -> Option<(u32, u32, u32)> {
    let levels = t.levels().clamp(1, crate::images::max_levels(t.width(), t.height(), 1));
    let level = t.base_level();
    if level > t.last_level() || level >= levels {
        return None;
    }
    Some((level, (t.width() >> level).max(1), (t.height() >> level).max(1)))
}

/// Decode a dispatch of the kernel over `groups` work groups (a direct
/// dispatch counting groups) with source T# `src` and destination T#
/// `dst`. `Err` says why the copy would not reproduce the kernel exactly.
pub fn decode_image_copy(src: &TextureSharp, dst: &TextureSharp, groups: [u32; 3]) -> Result<ImageCopy, &'static str> {
    for t in [src, dst] {
        if t.is_null() {
            return Err("null T#");
        }
        if t.ty() != image_type::TYPE_2D || t.base_array() != 0 {
            return Err("not a plain 2D T#");
        }
    }
    if src.format() != dst.format() {
        return Err("different formats");
    }
    if !exact_round_trip(src.format()) {
        return Err("format conversion is not exact");
    }
    if !swizzles_copy_channels(src.dst_sel(), dst.dst_sel(), components(src.format())) {
        return Err("swizzles do not copy every channel");
    }
    let (Some((src_level, sw, sh)), Some((dst_level, dw, dh))) = (base_level_extent(src), base_level_extent(dst)) else {
        return Err("base level outside the surface");
    };
    // MIN_LOD above the base level clamps the level a fetch reads.
    if src.min_lod() > src_level * 256 {
        return Err("MIN_LOD clamps the source level");
    }
    let grid = |g: u32| if groups[2] == 0 { 0 } else { g as u64 * 8 };
    let width = grid(groups[0]).min(dw as u64) as u32;
    let height = grid(groups[1]).min(dh as u64) as u32;
    let copy = ImageCopy { src_level, dst_level, width, height };
    // Destination texels past the source's extent would store the zeros
    // of out-of-bounds loads.
    if !copy.is_empty() && (width > sw || height > sh) {
        return Err("the source is smaller than the copied area");
    }
    Ok(copy)
}

/// Whether dispatch `d` runs the copy kernel (its code read from guest
/// memory through `read`): direct, counting groups, starting at group 0.
pub fn dispatch_matches(d: &ResolvedDispatch, read: impl FnOnce(u64, &mut [u8]) -> bool) -> bool {
    let s = &d.shader;
    if d.packet.indirect_address != 0 || d.uses_thread_dimensions() || d.start != [0, 0, 0] || !s.bound() {
        return false;
    }
    // Cheap checks before the code read.
    if !matches_image_copy(&IMAGE_COPY_KERNEL, d.threads, s.user_sgpr_count(), s.rsrc2) {
        return false;
    }
    let mut bytes = [0u8; IMAGE_COPY_KERNEL.len() * 4];
    if !read(s.address, &mut bytes) {
        return false;
    }
    let words = bytes.as_chunks::<4>().0;
    let code: [u32; IMAGE_COPY_KERNEL.len()] = std::array::from_fn(|i| u32::from_le_bytes(words[i]));
    matches_image_copy(&code, d.threads, s.user_sgpr_count(), s.rsrc2)
}

/// Run a dispatch of the copy kernel (see [`dispatch_matches`]) as a copy
/// between the cached images of its T#s, looked up as the dispatch's
/// bindings would be (the source as a sampled image, the destination as a
/// storage image). The destination is marked GPU-written as a dispatch
/// storing to it is. `Err` says why the dispatch must run instead (nothing
/// was recorded then, apart from loading the images).
pub fn run_image_copy(sched: &mut Scheduler, buffers: &mut BufferCache, images: &mut ImageCache, d: &ResolvedDispatch) -> Result<ImageCopy, String> {
    let u = &d.shader.user_data;
    let src_t = TextureSharp::from_words(&u[0..8]);
    let dst_t = TextureSharp::from_words(&u[8..16]);
    let copy = decode_image_copy(&src_t, &dst_t, d.packet.groups)?;
    if copy.is_empty() {
        return Ok(copy);
    }
    let fmt = texel_format(src_t.format()).ok_or("no host format")?;
    let src_key = texture_key(&src_t, &fmt)?;
    let dst_key = texture_key(&dst_t, &fmt)?;
    // The source as a sampled binding finds it: every level up to
    // LAST_LEVEL is viewed. An 8-bit texture over a stencil plane would
    // read the stencil aspect of the depth target.
    if images.stencil_owner(&src_key).is_some() {
        return Err("the source is a stencil plane".into());
    }
    let src_levels = src_t.last_level().min(src_key.levels - 1) + 1;
    let src = match images.find_alias(&src_key, src_levels) {
        Some(id) => id,
        None => images.find_or_create(sched, buffers, &src_key).map_err(|e| format!("source image: {e:?}"))?,
    };
    // The destination as a storage binding: a depth surface is written
    // through a colour image of its memory.
    let mut dst = match images.find_alias(&dst_key, copy.dst_level + 1) {
        Some(id) => id,
        None => images.find_or_create(sched, buffers, &dst_key).map_err(|e| format!("destination image: {e:?}"))?,
    };
    if images.get(dst).is_some_and(|i| i.key.is_depth()) {
        dst = images.find_or_create(sched, buffers, &dst_key).map_err(|e| format!("destination image: {e:?}"))?;
    }
    if src == dst {
        return Err("source and destination are one image".into());
    }
    images.prepare(sched, buffers, src, false);
    images.prepare(sched, buffers, dst, false);
    let usable = |id: u64, level: u32, usage: ash::vk::ImageUsageFlags| {
        images.get(id).is_some_and(|i| {
            i.key.samples == 1
                && !i.key.is_depth()
                && i.key.transform == fmt.transform
                && i.image.aspect == ash::vk::ImageAspectFlags::COLOR
                && i.image.usage.contains(usage)
                && level < i.image.mip_levels
                && crate::images::block_dim(i.key.format) == 1
                && i.key.main.bytes_per_element == fmt.bytes_per_element
        })
    };
    if !usable(src, copy.src_level, ash::vk::ImageUsageFlags::TRANSFER_SRC) || !usable(dst, copy.dst_level, ash::vk::ImageUsageFlags::TRANSFER_DST) {
        return Err("cached images cannot be copied".into());
    }
    // As before a dispatch: work recorded without precise barriers may
    // have written anything. The images order themselves through their
    // layouts.
    if sched.has_unordered_work() {
        sched.full_barrier();
    }
    if !images.copy_region(sched, src, (copy.src_level, 0), dst, (copy.dst_level, 0), (copy.width, copy.height)) {
        return Err("copy refused by the image cache".into());
    }
    images.note_storage_write(dst);
    Ok(copy)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RSRC2 of the kernel as the title sets it (16 user SGPRs, TGID_X/Y,
    /// two thread-id components).
    const RSRC2: u32 = 0x9A0;

    /// A 2D T# of `w` x `h` (unified format `format`, DST_SEL `sel`,
    /// levels `base`..=`last`, MAX_MIP `max_mip`).
    fn t(format: u32, w: u32, h: u32, sel: u32, base: u32, last: u32, max_mip: u32) -> TextureSharp {
        let addr = 0x10_0000u64;
        TextureSharp([
            (addr >> 8) as u32,
            ((addr >> 40) as u32 & 0xFF) | format << 20 | ((w - 1) & 3) << 30,
            (w - 1) >> 2 | (h - 1) << 14,
            sel | base << 12 | last << 16 | image_type::TYPE_2D << 28,
            0,
            max_mip << 4,
            0,
            0,
        ])
    }

    fn rgba8(w: u32, h: u32) -> TextureSharp {
        t(56, w, h, 0xFAC, 0, 0, 0)
    }

    #[test]
    fn the_kernel_matches_with_its_launch_shape() {
        assert!(matches_image_copy(&IMAGE_COPY_KERNEL, [8, 8, 1], 16, RSRC2));
        let mut longer = IMAGE_COPY_KERNEL.to_vec();
        longer.extend_from_slice(&[0, 0, 0]);
        assert!(matches_image_copy(&longer, [8, 8, 1], 16, RSRC2), "trailing bytes after s_endpgm rejected");
        for word in 0..IMAGE_COPY_KERNEL.len() {
            for bit in [0u32, 8, 16, 31] {
                let mut altered = IMAGE_COPY_KERNEL;
                altered[word] ^= 1 << bit;
                assert!(!matches_image_copy(&altered, [8, 8, 1], 16, RSRC2), "a different shader (word {word} bit {bit}) matched");
            }
        }
        assert!(!matches_image_copy(&IMAGE_COPY_KERNEL[..9], [8, 8, 1], 16, RSRC2), "truncated kernel matched");
        for threads in [[8, 4, 1], [4, 8, 1], [16, 8, 1], [8, 8, 2], [64, 1, 1]] {
            assert!(!matches_image_copy(&IMAGE_COPY_KERNEL, threads, 16, RSRC2), "{threads:?} threads matched");
        }
        assert!(!matches_image_copy(&IMAGE_COPY_KERNEL, [8, 8, 1], 15, RSRC2), "group ids elsewhere matched");
        assert!(!matches_image_copy(&IMAGE_COPY_KERNEL, [8, 8, 1], 16, RSRC2 & !(1 << 7)), "no group id x matched");
        assert!(!matches_image_copy(&IMAGE_COPY_KERNEL, [8, 8, 1], 16, RSRC2 & !(1 << 8)), "no group id y matched");
        assert!(!matches_image_copy(&IMAGE_COPY_KERNEL, [8, 8, 1], 16, RSRC2 & !(3 << 11)), "one thread-id component matched");
        // A group id z after them changes nothing.
        assert!(matches_image_copy(&IMAGE_COPY_KERNEL, [8, 8, 1], 16, RSRC2 | 1 << 9));
        assert!(matches_image_copy(&IMAGE_COPY_KERNEL, [8, 8, 1], 16, RSRC2 | 1 << 11 | 1 << 12));
    }

    #[test]
    fn the_raw_bytes_are_the_kernel() {
        let bytes: [u8; 40] = [
            0x04, 0x00, 0x46, 0xd7, 0x10, 0x06, 0x01, 0x04, 0x05, 0x00, 0x46, 0xd7, 0x11, 0x06, 0x05, 0x04, 0x08, 0x0f, 0x00, 0xf0, 0x04, 0x00, 0x00, 0x00, 0x70, 0x3f, 0x8c,
            0xbf, 0x08, 0x0f, 0x20, 0xf0, 0x04, 0x00, 0x02, 0x00, 0x00, 0x00, 0x81, 0xbf,
        ];
        let code: Vec<u32> = bytes.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(code, IMAGE_COPY_KERNEL);
    }

    #[test]
    fn the_copied_area_is_the_grid_inside_both_images() {
        // Grid larger than the images (odd sizes).
        let c = decode_image_copy(&rgba8(37, 21), &rgba8(37, 21), [5, 3, 1]).unwrap();
        assert_eq!(c, ImageCopy { src_level: 0, dst_level: 0, width: 37, height: 21 });
        // Grid smaller than the images.
        let c = decode_image_copy(&rgba8(100, 60), &rgba8(100, 60), [3, 2, 1]).unwrap();
        assert_eq!((c.width, c.height), (24, 16));
        // Destination smaller than the source: bounded by it.
        let c = decode_image_copy(&rgba8(64, 64), &rgba8(13, 9), [8, 8, 1]).unwrap();
        assert_eq!((c.width, c.height), (13, 9));
        // Source smaller than the copied area: the rest would store zeros.
        assert!(decode_image_copy(&rgba8(13, 9), &rgba8(64, 64), [8, 8, 1]).is_err());
        assert!(decode_image_copy(&rgba8(64, 9), &rgba8(64, 64), [8, 8, 1]).is_err());
        // ... unless the grid stops inside the source.
        let c = decode_image_copy(&rgba8(16, 9), &rgba8(64, 64), [2, 1, 1]).unwrap();
        assert_eq!((c.width, c.height), (16, 8));
        // Empty dispatches copy nothing (and need no source).
        for groups in [[0, 4, 1], [4, 0, 1], [4, 4, 0]] {
            assert!(decode_image_copy(&rgba8(64, 64), &rgba8(64, 64), groups).unwrap().is_empty(), "{groups:?}");
        }
        // More groups along z repeat the same stores.
        assert_eq!(decode_image_copy(&rgba8(64, 64), &rgba8(64, 64), [2, 2, 7]).unwrap().width, 16);
        // Group counts whose texel coordinates pass 32 bits cover the image.
        assert_eq!(decode_image_copy(&rgba8(64, 64), &rgba8(64, 64), [u32::MAX, 1, 1]).unwrap().width, 64);
    }

    #[test]
    fn base_levels_select_the_level_and_its_extent() {
        // 64x40 with 4 levels, base level 2: 16x10.
        let src = t(56, 64, 40, 0xFAC, 2, 3, 3);
        let dst = t(56, 64, 40, 0xFAC, 1, 3, 3);
        // Destination level 1 is 32x20; the source level holds 16x10 only.
        assert!(decode_image_copy(&src, &dst, [8, 8, 1]).is_err());
        assert_eq!(decode_image_copy(&src, &dst, [2, 1, 1]).unwrap(), ImageCopy { src_level: 2, dst_level: 1, width: 16, height: 8 });
        let dst = t(56, 64, 40, 0xFAC, 2, 3, 3);
        let c = decode_image_copy(&src, &dst, [8, 8, 1]).unwrap();
        assert_eq!(c, ImageCopy { src_level: 2, dst_level: 2, width: 16, height: 10 });
        // Odd sizes round down, never below 1.
        let src = t(22, 7, 5, 0x204, 2, 2, 2);
        let c = decode_image_copy(&src, &src, [1, 1, 1]).unwrap();
        assert_eq!((c.width, c.height), (1, 1));
        // Base level past LAST_LEVEL or the surface's levels.
        assert!(decode_image_copy(&t(56, 64, 64, 0xFAC, 2, 1, 3), &rgba8(64, 64), [1, 1, 1]).is_err());
        assert!(decode_image_copy(&t(56, 64, 64, 0xFAC, 7, 15, 15), &rgba8(64, 64), [1, 1, 1]).is_err());
        // MIN_LOD above the base level.
        let mut clamped = t(56, 64, 64, 0xFAC, 0, 3, 3);
        clamped.0[1] |= 0x100 << 8;
        assert!(decode_image_copy(&clamped, &rgba8(64, 64), [1, 1, 1]).is_err());
    }

    #[test]
    fn only_exact_formats_and_swizzles_copy() {
        for format in [1, 7, 14, 23, 50, 56, 65, 5, 20, 60, 75, 6, 21, 13, 22, 29, 71, 77] {
            let s = t(format, 32, 32, 0xFAC, 0, 0, 0);
            assert!(decode_image_copy(&s, &s, [4, 4, 1]).is_ok(), "format {format}");
        }
        // Snorm, sRGB, scaled, 10_11_11, 5_9_9_9, packed 16-bit, BC.
        for format in [2, 57, 66, 130, 128, 3, 58, 36, 43, 132, 133, 136, 169, 181] {
            let s = t(format, 32, 32, 0xFAC, 0, 0, 0);
            assert!(decode_image_copy(&s, &s, [4, 4, 1]).is_err(), "format {format}");
        }
        // Different formats.
        assert!(decode_image_copy(&t(56, 32, 32, 0xFAC, 0, 0, 0), &t(60, 32, 32, 0xFAC, 0, 0, 0), [4, 4, 1]).is_err());
        // Not 2D, or a slice past the first.
        let mut array = rgba8(32, 32);
        array.0[3] = array.0[3] & 0x0FFF_FFFF | image_type::TYPE_2D_ARRAY << 28;
        assert!(decode_image_copy(&array, &rgba8(32, 32), [4, 4, 1]).is_err());
        let mut slice = rgba8(32, 32);
        slice.0[4] |= 1 << 16;
        assert!(decode_image_copy(&rgba8(32, 32), &slice, [4, 4, 1]).is_err());
        let mut msaa = rgba8(32, 32);
        msaa.0[3] = msaa.0[3] & 0x0FFF_FFFF | image_type::TYPE_2D_MSAA << 28;
        assert!(decode_image_copy(&msaa, &rgba8(32, 32), [4, 4, 1]).is_err());
    }

    #[test]
    fn swizzles_must_carry_every_channel() {
        // Identity, one-channel X001 / X000 on both sides.
        assert!(swizzles_copy_channels(0xFAC, 0xFAC, 4));
        assert!(swizzles_copy_channels(0x204, 0x204, 1));
        assert!(swizzles_copy_channels(0x004, 0xFAC, 1));
        // BGRA on both sides cancels: load ZYXW, store inverse.
        assert!(swizzles_copy_channels(0xF2E, 0xF2E, 4));
        // BGRA on one side swaps red and blue.
        assert!(!swizzles_copy_channels(0xF2E, 0xFAC, 4));
        assert!(!swizzles_copy_channels(0xFAC, 0xF2E, 4));
        // Alpha forced to one on load (XYZ1) is not the source's alpha.
        assert!(!swizzles_copy_channels(0x3AC, 0xFAC, 4));
        assert!(swizzles_copy_channels(0x3AC, 0xFAC, 3));
        // Alpha not stored (XYZ0 destination) stores 0.
        assert!(!swizzles_copy_channels(0xFAC, 0x1AC, 4));
        // Duplicated store selectors: the lowest component wins.
        assert!(swizzles_copy_channels(0xFA4, 0xFA4, 1));
        assert!(!swizzles_copy_channels(0xFA4, 0xFA4, 2));
        assert!(!swizzles_copy_channels(0xFAC, 0xFAC, 0));
    }
}
