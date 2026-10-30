// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Image cache on the real device: GPU-detiled upload, CPU-write
//! invalidation through the fault handler, clears, write-back and aliasing.
//! Run with `cargo test -p ps5gpu --release -- --ignored`.

use super::*;
use crate::memory::tracker::GuestPageTracker;
use crate::memory::{BufferStore, ImageStore, UnmapListener};
use crate::testutil::{guest_mapping, map_guest, perms, test_memory, unmap_guest};
use crate::tiling::{self, DetileParams};

struct Images(Arc<WatchTable>);
impl ImageStore for Images {
    fn mark_cpu_write(&self, addr: u64, size: u64) -> bool {
        self.0.mark_cpu_write(addr, size, true)
    }
    fn mapped(&self, addr: u64, size: u64) {
        self.0.mark_mapped(addr, size);
    }
}
struct NoBuffers;
impl BufferStore for NoBuffers {
    fn mark_cpu_write(&self, _: u64, _: u64) -> bool {
        false
    }
    fn download_to_cpu(&self, _: u64, _: u64) -> bool {
        false
    }
}
struct NoUnmap;
impl UnmapListener for NoUnmap {
    fn unmapped(&self, _: u64, _: u64) {}
}

fn read_image(dev: &Arc<GpuDevice>, sched: &mut Scheduler, images: &mut ImageCache, id: u64, bpe: u32) -> Vec<u8> {
    images.transit(sched, id, super::TRANSFER_SRC, None);
    let img = images.get(id).unwrap();
    let (w, h) = (img.key.width, img.key.height);
    let size = (w * h * bpe) as u64;
    let out = Buffer::new(dev, size, vk::BufferUsageFlags::TRANSFER_DST, MemoryUse::Download).unwrap();
    sched.full_barrier();
    let region = vk::BufferImageCopy::default()
        .image_subresource(vk::ImageSubresourceLayers { aspect_mask: img.key.main.aspect, mip_level: 0, base_array_layer: 0, layer_count: 1 })
        .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
    let cb = sched.cmd();
    // SAFETY: recording buffer; image in TRANSFER_SRC layout.
    unsafe { dev.device.cmd_copy_image_to_buffer(cb, img.image.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, out.buffer, &[region]) };
    sched.flush_and_wait();
    out.invalidate(0, size);
    out.bytes_mut()[..size as usize].to_vec()
}

#[test]
#[ignore]
fn image_cache_upload_invalidate_clear_writeback() {
    let mem = test_memory();
    let (w, h) = (300u32, 200u32);
    let p = DetileParams::new(tiling::SWIZZLE_64KB_R_X, 4, w).unwrap();
    let size = p.surface_size(h);
    let (view, _) = guest_mapping(mem, 8 << 20);
    let addr = view + (1 << 20);
    // Guest surface: tiled RGBA8 pattern.
    let lin: Vec<u8> = (0..(w * h * 4) as usize).map(|i| (i * 7 % 251) as u8).collect();
    // SAFETY: freshly mapped, unwatched guest view.
    let surface = unsafe { std::slice::from_raw_parts_mut(addr as *mut u8, size) };
    tiling::tile(&p, &lin, surface, w, h);

    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));

    let key = ImageKey {
        format: vk::Format::R8G8B8A8_UNORM,
        width: w,
        height: h,
        depth: 1,
        layers: 1,
        levels: 1,
        samples: 1,
        dim: ImageDim::D2,
        main: Plane {
            addr,
            bytes_per_element: 4,
            swizzle: tiling::SWIZZLE_64KB_R_X,
            pitch: p.pitch,
            slice_size: size as u64,
            bytes: size as u64,
            aspect: vk::ImageAspectFlags::COLOR,
        },
        stencil: None,
        transform: Default::default(),
    };
    let id = images.find_or_create(&mut sched, &mut buffers, &key).unwrap();
    assert!(images.needs_upload(id));
    assert!(images.prepare(&mut sched, &mut buffers, id, false));
    assert!(!images.needs_upload(id));
    assert_eq!(read_image(&dev, &mut sched, &mut images, id, 4), lin, "GPU-detiled upload matches");
    assert_eq!(perms(addr), "r--", "uploaded image pages are write-watched");

    // A CPU write into the surface invalidates the image; the next use reloads it.
    let (x, y) = (123u32, 45u32);
    let off = p.offset(x, y);
    // SAFETY: faulting write resolved by the image watch table.
    unsafe { std::ptr::write_volatile((addr + off as u64) as *mut u32, 0x0102_0304) };
    assert!(images.needs_upload(id));
    assert_eq!(perms(addr), "rw-");
    assert!(images.prepare(&mut sched, &mut buffers, id, false));
    let back = read_image(&dev, &mut sched, &mut images, id, 4);
    let i = ((y * w + x) * 4) as usize;
    assert_eq!(&back[i..i + 4], &0x0102_0304u32.to_le_bytes());
    assert_eq!(images.stats().uploads, 2);

    // Clear on the GPU: the image becomes GPU-modified and is the display image.
    let red = vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 1.0] };
    images.clear_color(&mut sched, id, red, None);
    assert!(images.get(id).unwrap().gpu_modified);
    assert_eq!(images.find_display(addr, w, h), Some(id));
    assert_eq!(images.find_display(addr + 4096, w, h), None);
    assert!(!images.needs_upload(id), "GPU contents are kept while watched");

    // A partial clear through dynamic rendering.
    let blue = vk::ClearColorValue { float32: [0.0, 0.0, 1.0, 1.0] };
    images.clear_color(&mut sched, id, blue, Some(vk::Rect2D { offset: vk::Offset2D { x: 10, y: 20 }, extent: vk::Extent2D { width: 30, height: 40 } }));
    let px = read_image(&dev, &mut sched, &mut images, id, 4);
    let at = |x: u32, y: u32| px[((y * w + x) * 4) as usize..((y * w + x) * 4 + 4) as usize].to_vec();
    assert_eq!(at(0, 0), vec![255, 0, 0, 255]);
    assert_eq!(at(15, 25), vec![0, 0, 255, 255]);
    assert_eq!(at(40, 60), vec![255, 0, 0, 255]);

    // An aliasing request with another format of the same geometry takes
    // the GPU contents over without a trip through guest memory.
    let alias = ImageKey { format: vk::Format::R32_UINT, ..key };
    let id2 = images.find_or_create(&mut sched, &mut buffers, &alias).unwrap();
    assert!(images.get(id).is_some(), "overlapping images coexist");
    assert!(images.needs_refresh(id2));
    assert!(images.prepare(&mut sched, &mut buffers, id2, false));
    assert_eq!((images.stats().conversions, images.stats().writebacks, images.stats().uploads), (1, 0, 2));
    assert!(images.get(id2).unwrap().gpu_modified);
    assert!(!images.needs_refresh(id2));
    assert!(!images.needs_refresh(id), "both images hold the same contents");
    let converted = read_image(&dev, &mut sched, &mut images, id2, 4);
    assert!(images.writeback(&mut sched, &mut buffers, id2));
    // The write-back is lazy: the pages are GPU-dirty in the buffer cache
    // until the CPU needs them.
    assert!(buffers.has_gpu_dirty(addr, size as u64));
    assert!(buffers.download(&mut sched, addr, size as u64));
    // Guest memory now holds the cleared surface (tiled).
    let mut guest = vec![0u8; size];
    assert!(mem.read_backing(addr, &mut guest));
    let mut detiled = vec![0u8; (w * h * 4) as usize];
    tiling::detile(&p, &guest, &mut detiled, w, h);
    assert_eq!(&detiled[0..4], &[255, 0, 0, 255]);
    let j = ((25 * w + 15) * 4) as usize;
    assert_eq!(&detiled[j..j + 4], &[0, 0, 255, 255]);
    assert_eq!(converted, detiled, "the converted image holds the written-back bytes");

    // Depth: clear, write back, alias as color.
    let dp = DetileParams::new(tiling::SWIZZLE_64KB_Z_X, 4, 128).unwrap();
    let daddr = view + (4 << 20);
    let dkey = ImageKey {
        format: vk::Format::D32_SFLOAT,
        width: 128,
        height: 64,
        depth: 1,
        layers: 1,
        levels: 1,
        samples: 1,
        dim: ImageDim::D2,
        main: Plane {
            addr: daddr,
            bytes_per_element: 4,
            swizzle: tiling::SWIZZLE_64KB_Z_X,
            pitch: dp.pitch,
            slice_size: dp.surface_size(64) as u64,
            bytes: dp.surface_size(64) as u64,
            aspect: vk::ImageAspectFlags::DEPTH,
        },
        stencil: None,
        transform: Default::default(),
    };
    let did = images.find_or_create(&mut sched, &mut buffers, &dkey).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, did, true));
    images.clear_depth_stencil(&mut sched, did, Some(0.5), None, None, None);
    assert!(images.writeback(&mut sched, &mut buffers, did));
    assert!(buffers.download(&mut sched, daddr, images.get(did).unwrap().key.size()));
    let mut d = [0u8; 4];
    assert!(mem.read_backing(daddr + dp.offset(77, 33) as u64, &mut d));
    assert_eq!(f32::from_le_bytes(d), 0.5);
    // A storage-capable colour image of the depth surface: converted on the GPU.
    let ckey = ImageKey { format: vk::Format::R32_SFLOAT, main: Plane { aspect: vk::ImageAspectFlags::COLOR, ..dkey.main }, ..dkey };
    images.clear_depth_stencil(&mut sched, did, Some(0.25), None, None, None);
    let cid = images.find_or_create(&mut sched, &mut buffers, &ckey).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, cid, false));
    let px = read_image(&dev, &mut sched, &mut images, cid, 4);
    assert_eq!(f32::from_le_bytes(px[0..4].try_into().unwrap()), 0.25);
    // A compute pass writes the colour image; the next depth pass sees it.
    images.clear_color(&mut sched, cid, vk::ClearColorValue { float32: [0.75, 0.0, 0.0, 0.0] }, None);
    assert_eq!(images.find_or_create(&mut sched, &mut buffers, &dkey).unwrap(), did);
    assert!(images.needs_refresh(did));
    assert!(images.prepare(&mut sched, &mut buffers, did, false));
    assert!(images.get(did).unwrap().gpu_modified);
    assert_eq!(images.stats().conversions, 3);
    let px = read_image(&dev, &mut sched, &mut images, did, 4);
    assert_eq!(f32::from_le_bytes(px[0..4].try_into().unwrap()), 0.75);
    sched.flush_and_wait();
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

/// Element pattern of (level, slice, x, y), `bpe` bytes.
fn pattern(level: u32, slice: u32, x: u32, y: u32, bpe: u32) -> Vec<u8> {
    let seed = (level as u64) << 48 | (slice as u64) << 32 | (y as u64) << 16 | x as u64;
    (0..bpe as u64).map(|b| ((seed.wrapping_add(b).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 40) as u8).collect()
}

/// Write the pattern of every element of every level and slice of `key`
/// into guest memory at `addr` through the CPU layout reference.
fn fill_surface(addr: u64, key: &ImageKey) -> crate::tiling_layout::SurfaceLayout {
    let layout = key.layout(&key.main).expect("layout");
    let bpe = key.main.bytes_per_element;
    // SAFETY: the caller's fresh guest mapping covers the surface.
    let surface = unsafe { std::slice::from_raw_parts_mut(addr as *mut u8, layout.guest_bytes as usize) };
    for (level, m) in layout.mips.iter().enumerate() {
        let level = level as u32;
        for slice in 0..layout.level_slices(level) {
            for y in 0..m.height {
                for x in 0..m.width {
                    let o = layout.element_offset(level, slice, x, y).unwrap() as usize;
                    surface[o..o + bpe as usize].copy_from_slice(&pattern(level, slice, x, y, bpe));
                }
            }
        }
    }
    layout
}

/// Every level of the image, packed (elements of slice 0, slice 1, ...).
fn read_levels(dev: &Arc<GpuDevice>, sched: &mut Scheduler, images: &mut ImageCache, id: u64) -> Vec<Vec<u8>> {
    images.transit(sched, id, super::TRANSFER_SRC, None);
    let img = images.get(id).unwrap();
    let key = img.key;
    let bd = block_dim(key.format);
    let bpe = key.main.bytes_per_element as u64;
    let mut out = Vec::new();
    for level in 0..img.image.mip_levels {
        let e = key.level_extent(level);
        let slices = if key.dim == ImageDim::D3 { e.depth } else { key.layers };
        let size = e.width.div_ceil(bd) as u64 * e.height.div_ceil(bd) as u64 * slices as u64 * bpe;
        let buf = Buffer::new(dev, size, vk::BufferUsageFlags::TRANSFER_DST, MemoryUse::Download).unwrap();
        sched.full_barrier();
        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: key.main.aspect,
                mip_level: level,
                base_array_layer: 0,
                layer_count: if key.dim == ImageDim::D3 { 1 } else { slices },
            })
            .image_extent(e);
        let cb = sched.cmd();
        // SAFETY: recording buffer; image in TRANSFER_SRC layout.
        unsafe { dev.device.cmd_copy_image_to_buffer(cb, img.image.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buf.buffer, &[region]) };
        sched.flush_and_wait();
        buf.invalidate(0, size);
        out.push(buf.bytes_mut()[..size as usize].to_vec());
    }
    out
}

/// Upload surfaces of every layout family (mip chains with tails, arrays
/// with the `_X` slice XOR, thick and thin-slice volumes, linear chains,
/// block-compressed chains, 1D arrays) and compare each element with the
/// CPU layout reference; then write one back after a GPU clear.
#[test]
#[ignore]
fn image_cache_mips_arrays_volumes() {
    let mem = test_memory();
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));
    use vk::Format as F;
    // (format, bpe, swizzle, dim, width, height, depth, layers, levels)
    let cases: &[(F, u32, u32, ImageDim, u32, u32, u32, u32, u32)] = &[
        (F::R8G8B8A8_UNORM, 4, tiling::SWIZZLE_64KB_S, ImageDim::D2, 100, 60, 1, 3, 7),
        (F::R16G16B16A16_SFLOAT, 8, tiling::SWIZZLE_64KB_R_X, ImageDim::D2, 300, 170, 1, 2, 9),
        (F::R8_UNORM, 1, tiling::SWIZZLE_4KB_S_X, ImageDim::D2, 100, 70, 1, 3, 6),
        (F::R32G32B32A32_SFLOAT, 16, tiling::SWIZZLE_64KB_D_T, ImageDim::D2, 64, 40, 1, 2, 5),
        (F::R8G8B8A8_UNORM, 4, tiling::SWIZZLE_LINEAR, ImageDim::D2, 37, 19, 1, 2, 4),
        (F::BC1_RGBA_UNORM_BLOCK, 8, tiling::SWIZZLE_64KB_S, ImageDim::D2, 256, 256, 1, 2, 9),
        (F::BC7_UNORM_BLOCK, 16, tiling::SWIZZLE_4KB_S, ImageDim::D2, 120, 72, 1, 1, 6),
        (F::R8G8B8A8_UNORM, 4, tiling::SWIZZLE_4KB_S, ImageDim::D3, 33, 20, 12, 1, 6),
        (F::R8_UNORM, 1, tiling::SWIZZLE_64KB_S_X, ImageDim::D3, 100, 60, 70, 1, 7),
        (F::R16G16B16A16_SFLOAT, 8, tiling::SWIZZLE_64KB_R_X, ImageDim::D3, 64, 64, 64, 1, 1),
        (F::R16G16B16A16_SFLOAT, 8, tiling::SWIZZLE_64KB_Z_X, ImageDim::D3, 33, 20, 12, 1, 6),
        (F::R32_SFLOAT, 4, tiling::SWIZZLE_64KB_R_X, ImageDim::D1, 300, 1, 1, 4, 9),
    ];
    let (view, _) = guest_mapping(mem, 64 << 20);
    let mut addr = view;
    for &(format, bpe, swizzle, dim, width, height, depth, layers, levels) in cases {
        let mut key = ImageKey { format, width, height, depth, layers, levels, samples: 1, dim, main: Plane { addr: 0, bytes_per_element: 0, swizzle: 0, pitch: 0, slice_size: 0, bytes: 0, aspect: vk::ImageAspectFlags::COLOR }, stencil: None, transform: Default::default() };
        key.main = plane(&key, addr, bpe, swizzle, vk::ImageAspectFlags::COLOR);
        let what = format!("{format:?} swizzle {swizzle} {dim:?} {width}x{height}x{depth} [{layers}] {levels} levels");
        let layout = fill_surface(addr, &key);
        let id = images.find_or_create(&mut sched, &mut buffers, &key).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        assert!(images.prepare(&mut sched, &mut buffers, id, false), "{what}");
        let got = read_levels(&dev, &mut sched, &mut images, id);
        for (level, data) in got.iter().enumerate() {
            let level = level as u32;
            let m = layout.mips[level as usize];
            let mut i = 0;
            for slice in 0..layout.level_slices(level) {
                for y in 0..m.height {
                    for x in 0..m.width {
                        assert_eq!(&data[i..i + bpe as usize], &pattern(level, slice, x, y, bpe)[..], "{what}: level {level} slice {slice} ({x}, {y})");
                        i += bpe as usize;
                    }
                }
            }
        }
        addr = (addr + layout.guest_bytes).next_multiple_of(65536);
        // The views the T# path asks for exist (validated by the layers).
        let full = |ty, levels, layers| ViewRange { ty, base_level: 0, levels, base_layer: 0, layers, min_lod: 0.0 };
        let id_views: &[(vk::ImageViewType, u32, u32, bool)] = match dim {
            ImageDim::D1 => &[(vk::ImageViewType::TYPE_1D_ARRAY, 99, 99, true), (vk::ImageViewType::TYPE_1D, 1, 1, true), (vk::ImageViewType::TYPE_2D, 1, 1, false)],
            ImageDim::D3 => &[(vk::ImageViewType::TYPE_3D, 99, 1, true), (vk::ImageViewType::CUBE_ARRAY, 1, 6, false)],
            ImageDim::D2 => &[(vk::ImageViewType::TYPE_2D_ARRAY, 99, 99, true), (vk::ImageViewType::TYPE_2D, 99, 1, true), (vk::ImageViewType::TYPE_3D, 1, 1, false)],
        };
        for &(ty, levels, layers, ok) in id_views {
            let v = images.view_range(id, format, vk::ImageAspectFlags::COLOR, ViewRange { min_lod: 0.5, ..full(ty, levels, layers) }, vk::ComponentMapping::default());
            assert_eq!(v.is_some(), ok, "{what}: {ty:?} view");
        }
        if dim == ImageDim::D3 && block_dim(format) == 1 {
            // Render-target view: depth slices of level 1.
            assert!(images.view_level(id, format, vk::ImageAspectFlags::COLOR, levels.min(2) - 1, 0, depth).is_some(), "{what}: slices view");
        }
    }

    // Cube views of a square array of 12 layers (two cubes), with mips.
    let mut cube = ImageKey { format: F::R8G8B8A8_UNORM, width: 64, height: 64, depth: 1, layers: 12, levels: 7, samples: 1, dim: ImageDim::D2, main: Plane { addr: 0, bytes_per_element: 0, swizzle: 0, pitch: 0, slice_size: 0, bytes: 0, aspect: vk::ImageAspectFlags::COLOR }, stencil: None, transform: Default::default() };
    cube.main = plane(&cube, addr, 4, tiling::SWIZZLE_64KB_S_X, vk::ImageAspectFlags::COLOR);
    let layout = fill_surface(addr, &cube);
    let id = images.find_or_create(&mut sched, &mut buffers, &cube).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, id, false));
    let r = |ty, base_layer, layers| ViewRange { ty, base_level: 1, levels: 6, base_layer, layers, min_lod: 0.0 };
    assert!(images.view_range(id, cube.format, vk::ImageAspectFlags::COLOR, r(vk::ImageViewType::CUBE_ARRAY, 0, 12), vk::ComponentMapping::default()).is_some());
    assert!(images.view_range(id, cube.format, vk::ImageAspectFlags::COLOR, r(vk::ImageViewType::CUBE_ARRAY, 6, 6), vk::ComponentMapping::default()).is_some());
    assert!(images.view_range(id, cube.format, vk::ImageAspectFlags::COLOR, r(vk::ImageViewType::CUBE, 6, 6), vk::ComponentMapping::default()).is_some());
    assert!(images.view_range(id, cube.format, vk::ImageAspectFlags::COLOR, r(vk::ImageViewType::CUBE_ARRAY, 9, 3), vk::ComponentMapping::default()).is_none(), "fewer than six faces");
    addr = (addr + layout.guest_bytes).next_multiple_of(65536);

    // Write-back of a mipmapped array after a GPU clear of the whole image:
    // every element of every level reaches its guest address, the bytes in
    // between (padding, the free part of the tail block) are kept.
    let (format, bpe) = (F::R8G8B8A8_UNORM, 4u32);
    let mut key = ImageKey { format, width: 100, height: 60, depth: 1, layers: 3, levels: 7, samples: 1, dim: ImageDim::D2, main: Plane { addr: 0, bytes_per_element: 0, swizzle: 0, pitch: 0, slice_size: 0, bytes: 0, aspect: vk::ImageAspectFlags::COLOR }, stencil: None, transform: Default::default() };
    key.main = plane(&key, addr, bpe, tiling::SWIZZLE_64KB_S_X, vk::ImageAspectFlags::COLOR);
    let layout = fill_surface(addr, &key);
    let id = images.find_or_create(&mut sched, &mut buffers, &key).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, id, false));
    images.clear_color(&mut sched, id, vk::ClearColorValue { uint32: [0x11, 0x22, 0x33, 0x44] }, None);
    images.clear_color(&mut sched, id, vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 1.0] }, None);
    assert!(images.writeback(&mut sched, &mut buffers, id));
    assert!(buffers.download(&mut sched, addr, layout.guest_bytes));
    let mut guest = vec![0u8; layout.guest_bytes as usize];
    assert!(mem.read_backing(addr, &mut guest));
    let mut covered = vec![false; guest.len()];
    for (level, m) in layout.mips.iter().enumerate() {
        for slice in 0..3 {
            for y in 0..m.height {
                for x in 0..m.width {
                    let o = layout.element_offset(level as u32, slice, x, y).unwrap() as usize;
                    assert_eq!(&guest[o..o + 4], &[255, 0, 0, 255], "level {level} slice {slice} ({x}, {y})");
                    covered[o..o + 4].iter_mut().for_each(|c| *c = true);
                }
            }
        }
    }
    assert!(covered.iter().any(|c| !c), "the surface has padding");
    assert!(covered.iter().zip(&guest).all(|(c, b)| *c || *b == 0), "padding bytes are kept");
    sched.flush_and_wait();
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

/// Key of a single-level 2D colour surface.
fn key2d(format: vk::Format, bpe: u32, swizzle: u32, width: u32, height: u32, addr: u64) -> ImageKey {
    let mut key = ImageKey {
        format,
        width,
        height,
        depth: 1,
        layers: 1,
        levels: 1,
        samples: 1,
        dim: ImageDim::D2,
        main: Plane { addr: 0, bytes_per_element: 0, swizzle: 0, pitch: 0, slice_size: 0, bytes: 0, aspect: vk::ImageAspectFlags::COLOR },
        stencil: None,
        transform: Default::default(),
    };
    key.main = plane(&key, addr, bpe, swizzle, vk::ImageAspectFlags::COLOR);
    key
}

/// Tile `lin` (`w` x `h` elements of layout `p`) into `guest` at `offset`.
fn tile_at(p: &DetileParams, lin: &[u8], w: u32, h: u32, guest: &mut [u8], offset: usize) {
    let size = p.surface_size(h);
    assert!(tiling::tile(p, lin, &mut guest[offset..offset + size], w, h));
}

/// Read `guest` at `offset` as `w` x `h` elements of layout `p`.
fn detile_at(p: &DetileParams, guest: &[u8], offset: usize, w: u32, h: u32, bpe: usize) -> Vec<u8> {
    let mut out = vec![0u8; w as usize * h as usize * bpe];
    assert!(tiling::detile(p, &guest[offset..offset + p.surface_size(h)], &mut out, w, h));
    out
}

fn rect(x: i32, y: i32, w: u32, h: u32) -> Option<vk::Rect2D> {
    Some(vk::Rect2D { offset: vk::Offset2D { x, y }, extent: vk::Extent2D { width: w, height: h } })
}

/// Render-target memory reused by surfaces of other formats and sizes: each
/// image takes the newer GPU contents of the others on the GPU, as the
/// hardware would read the shared bytes, and guest memory is never touched.
#[test]
#[ignore]
fn image_cache_aliasing_other_format_and_size() {
    use vk::Format as F;
    let mem = test_memory();
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));

    let (view, _) = guest_mapping(mem, 8 << 20);
    let addr = view + (1 << 20);
    let sw = tiling::SWIZZLE_64KB_R_X;
    // Guest memory holds stale bytes the images must not pick up.
    let span = 256 << 10;
    let stale: Vec<u8> = (0..span).map(|i| (i * 13 % 241) as u8).collect();
    assert!(mem.write_backing(addr, &stale));

    // X: 256x128 RGBA8, rendered (a pattern of clears).
    let xkey = key2d(F::R8G8B8A8_UNORM, 4, sw, 256, 128, addr);
    let xp = DetileParams::new(sw, 4, 256).unwrap();
    let x = images.find_or_create(&mut sched, &mut buffers, &xkey).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, x, true));
    images.clear_color(&mut sched, x, vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 1.0] }, None);
    images.clear_color(&mut sched, x, vk::ClearColorValue { float32: [0.0, 1.0, 0.0, 1.0] }, rect(10, 20, 50, 30));
    images.clear_color(&mut sched, x, vk::ClearColorValue { float32: [0.0, 0.0, 1.0, 0.5] }, rect(200, 100, 40, 20));
    let xlin = read_image(&dev, &mut sched, &mut images, x, 4);

    // Y: 128x128 RGBA16 over the same memory reads X's bytes.
    let ykey = key2d(F::R16G16B16A16_UINT, 8, sw, 128, 128, addr);
    let yp = DetileParams::new(sw, 8, 128).unwrap();
    assert_eq!(ykey.size(), xkey.size());
    let y = images.find_or_create(&mut sched, &mut buffers, &ykey).unwrap();
    assert!(images.get(x).is_some(), "overlapping images coexist");
    assert!(images.prepare(&mut sched, &mut buffers, y, false));
    let mut guest = stale.clone();
    tile_at(&xp, &xlin, 256, 128, &mut guest, 0);
    assert_eq!(read_image(&dev, &mut sched, &mut images, y, 8), detile_at(&yp, &guest, 0, 128, 128, 8), "Y reinterprets X's bytes");
    let s = images.stats();
    assert_eq!((s.retiles, s.writebacks, s.uploads), (1, 0, 0));

    // Y is rendered; X takes Y's bytes back.
    images.clear_color(&mut sched, y, vk::ClearColorValue { uint32: [1, 2, 3, 4] }, None);
    images.clear_color(&mut sched, y, vk::ClearColorValue { uint32: [0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD] }, rect(30, 40, 20, 70));
    let ylin = read_image(&dev, &mut sched, &mut images, y, 8);
    assert!(images.needs_refresh(x));
    assert!(images.prepare(&mut sched, &mut buffers, x, false));
    tile_at(&yp, &ylin, 128, 128, &mut guest, 0);
    let xlin = read_image(&dev, &mut sched, &mut images, x, 4);
    assert_eq!(xlin, detile_at(&xp, &guest, 0, 256, 128, 4), "X reinterprets Y's bytes");
    assert!(!images.needs_refresh(y), "Y is not older than X");

    // Z: a small RGBA8 surface in X's second 64 KiB block. X keeps its own
    // bytes outside Z.
    let zkey = key2d(F::R8G8B8A8_UNORM, 4, sw, 64, 64, addr + (64 << 10));
    let zp = DetileParams::new(sw, 4, 64).unwrap();
    let z = images.find_or_create(&mut sched, &mut buffers, &zkey).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, z, true));
    images.clear_color(&mut sched, z, vk::ClearColorValue { float32: [1.0, 1.0, 1.0, 1.0] }, None);
    images.clear_color(&mut sched, z, vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] }, rect(5, 6, 7, 8));
    let zlin = read_image(&dev, &mut sched, &mut images, z, 4);
    assert!(images.prepare(&mut sched, &mut buffers, x, false));
    let mut guest = stale.clone();
    tile_at(&xp, &xlin, 256, 128, &mut guest, 0);
    tile_at(&zp, &zlin, 64, 64, &mut guest, 64 << 10);
    let xlin = read_image(&dev, &mut sched, &mut images, x, 4);
    assert_eq!(xlin, detile_at(&xp, &guest, 0, 256, 128, 4), "X = its own bytes with Z's on top");
    let s = images.stats();
    assert_eq!((s.retiles, s.writebacks, s.uploads), (3, 0, 0));

    // Guest memory was never written.
    let mut now = vec![0u8; span];
    assert!(mem.read_backing(addr, &mut now));
    assert!(now == stale, "no GPU data went through guest memory");

    // Y overwrites X completely; X, idle, is dropped and its host image
    // pooled. Recreating X takes the pooled image and Y's bytes.
    images.clear_color(&mut sched, y, vk::ClearColorValue { uint32: [7, 8, 9, 10] }, rect(0, 0, 64, 64));
    let ylin = read_image(&dev, &mut sched, &mut images, y, 8);
    for _ in 0..=IDLE_TICKS {
        sched.flush();
    }
    sched.flush_and_wait();
    images.collect(&mut sched, &mut buffers, u64::MAX);
    assert!(images.get(x).is_none(), "the superseded idle image was dropped");
    assert!(images.get(z).is_none(), "Z lies inside Y too");
    sched.flush_and_wait();
    let created = images.stats().created;
    let x2 = images.find_or_create(&mut sched, &mut buffers, &xkey).unwrap();
    assert_eq!((images.stats().created, images.stats().reused), (created, 1), "the host image came from the pool");
    assert!(images.prepare(&mut sched, &mut buffers, x2, false));
    let mut guest = stale.clone();
    tile_at(&yp, &ylin, 128, 128, &mut guest, 0);
    assert_eq!(read_image(&dev, &mut sched, &mut images, x2, 4), detile_at(&xp, &guest, 0, 256, 128, 4));
    assert_eq!(images.stats().writebacks, 0);

    // A CPU-visible consumer still gets the newest bytes, lazily: through
    // the buffer cache's mirror, GPU-dirty until the CPU needs it.
    assert!(!buffers.has_gpu_dirty(addr, xkey.size()), "aliasing never wrote guest pages");
    assert!(images.writeback(&mut sched, &mut buffers, x2));
    assert!(buffers.has_gpu_dirty(addr, xkey.size()));
    assert!(buffers.download(&mut sched, addr, xkey.size()));
    let mut now = vec![0u8; xkey.size() as usize];
    assert!(mem.read_backing(addr, &mut now));
    assert!(now == guest[..now.len()], "write-back stores the current bytes");
    sched.flush_and_wait();
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

/// Over the budget, only images idle for `EVICT_FRAMES` frames go: an image
/// used every frame stays (no reload per frame), an idle clean image is
/// dropped, and an idle rendered image is written back into guest memory
/// (its mirror settled and released) before it is dropped; recreated, it
/// loads the rendered bytes from guest memory.
#[test]
#[ignore]
fn image_cache_budget_evicts_only_idle_images() {
    use vk::Format as F;
    let mem = test_memory();
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));

    let (view, _) = guest_mapping(mem, 8 << 20);
    let sw = tiling::SWIZZLE_64KB_R_X;
    let (a_addr, b_addr, c_addr) = (view + (1 << 20), view + (2 << 20), view + (4 << 20));
    let a = images.find_or_create(&mut sched, &mut buffers, &key2d(F::R8G8B8A8_UNORM, 4, sw, 64, 64, a_addr)).unwrap();
    let b = images.find_or_create(&mut sched, &mut buffers, &key2d(F::R8G8B8A8_UNORM, 4, sw, 64, 64, b_addr)).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, a, false));
    assert!(images.prepare(&mut sched, &mut buffers, b, false));
    let c = images.find_or_create(&mut sched, &mut buffers, &key2d(F::R16G16B16A16_UINT, 8, sw, 64, 64, c_addr)).unwrap();
    images.prepare(&mut sched, &mut buffers, c, true);
    images.clear_color(&mut sched, c, vk::ClearColorValue { uint32: [5, 6, 7, 8] }, rect(0, 0, 32, 32));
    let ckey = images.get(c).unwrap().key;
    for _ in 0..=EVICT_FRAMES + 1 {
        // B is used every frame.
        assert!(images.prepare(&mut sched, &mut buffers, b, false));
        for _ in 0..4 {
            sched.flush();
        }
        images.end_frame(sched.current_tick());
        images.collect(&mut sched, &mut buffers, 0);
        assert!(images.get(b).is_some(), "an image used every frame stays");
    }
    sched.flush_and_wait();
    images.collect(&mut sched, &mut buffers, 0);
    assert!(images.get(a).is_none(), "the idle clean image was evicted");
    assert!(images.get(c).is_none(), "the idle rendered image was written back and evicted");
    assert!(buffers.has_gpu_dirty(c_addr, 4096), "its mirror holds the bytes until the copy-out completes");
    sched.full_barrier();
    sched.flush_and_wait();
    let idle = sched.current_tick();
    buffers.set_idle_threshold(0);
    buffers.collect(&mut sched, idle);
    assert!(!buffers.has_gpu_dirty(c_addr, ckey.size()), "settled: guest memory holds the rendered bytes");
    assert!(buffers.find_containing(c_addr, 4096).is_none(), "the idle clean mirror was released");
    // Recreated from guest memory: the cleared quarter is back.
    let c2 = images.find_or_create(&mut sched, &mut buffers, &ckey).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, c2, false));
    let lin = read_image(&dev, &mut sched, &mut images, c2, 8);
    let texel = |x: usize, y: usize| lin[(y * 64 + x) * 8..(y * 64 + x) * 8 + 8].to_vec();
    let cleared: Vec<u8> = [5u16, 6, 7, 8].iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(texel(3, 5), cleared);
    assert_eq!(texel(31, 31), cleared);
    sched.flush_and_wait();
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

/// Evicting overlapping GPU-modified images keeps the newest bytes in guest
/// memory: the older, larger alias (R32 over the same memory) is evicted first, the
/// newer RGBA8 image is written back after it, and an image recreated from
/// guest memory sees the newer contents.
#[test]
#[ignore]
fn image_cache_eviction_keeps_newest_alias_bytes() {
    use vk::Format as F;
    let mem = test_memory();
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));

    let (view, _) = guest_mapping(mem, 4 << 20);
    let sw = tiling::SWIZZLE_64KB_R_X;
    let addr = view + (1 << 20);
    let old_key = key2d(F::R32_UINT, 4, sw, 256, 256, addr);
    let new_key = key2d(F::R8G8B8A8_UINT, 4, sw, 64, 64, addr);
    let old = images.find_or_create(&mut sched, &mut buffers, &old_key).unwrap();
    images.prepare(&mut sched, &mut buffers, old, true);
    images.clear_color(&mut sched, old, vk::ClearColorValue { uint32: [0x1111_1111, 0, 0, 0] }, None);
    sched.flush();
    let new = images.find_or_create(&mut sched, &mut buffers, &new_key).unwrap();
    images.prepare(&mut sched, &mut buffers, new, true);
    images.clear_color(&mut sched, new, vk::ClearColorValue { uint32: [1, 2, 3, 4] }, rect(0, 0, 16, 16));
    for _ in 0..=EVICT_FRAMES + 1 {
        for _ in 0..4 {
            sched.flush();
        }
        images.end_frame(sched.current_tick());
    }
    sched.flush_and_wait();
    images.collect(&mut sched, &mut buffers, 0);
    images.collect(&mut sched, &mut buffers, 0);
    assert!(images.get(old).is_none() && images.get(new).is_none(), "both idle modified images were evicted");
    assert_eq!(images.stats().drops[4], 2);
    sched.full_barrier();
    sched.flush_and_wait();
    let idle = sched.current_tick();
    buffers.collect(&mut sched, idle);
    assert!(!buffers.has_gpu_dirty(addr, old_key.size()));
    // Recreated as R32: the newer clear in the corner, the older one elsewhere.
    let back = images.find_or_create(&mut sched, &mut buffers, &old_key).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, back, false));
    let lin = read_image(&dev, &mut sched, &mut images, back, 4);
    let texel = |x: usize, y: usize| u32::from_le_bytes(lin[(y * 256 + x) * 4..(y * 256 + x) * 4 + 4].try_into().unwrap());
    assert_eq!(texel(2, 3), 0x0403_0201);
    assert_eq!(texel(200, 200), 0x1111_1111);
    sched.flush_and_wait();
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

/// A CPU write into a GPU-modified image keeps its GPU contents: only the
/// page written is unwatched, and the next use takes that page from guest
/// memory over the image's own bytes.
#[test]
#[ignore]
fn image_cache_cpu_write_into_gpu_contents_merges_pages() {
    let mem = test_memory();
    let (w, h) = (300u32, 200u32);
    let p = DetileParams::new(tiling::SWIZZLE_64KB_R_X, 4, w).unwrap();
    let size = p.surface_size(h);
    let (view, _) = guest_mapping(mem, 8 << 20);
    let addr = view + (1 << 20);
    let lin: Vec<u8> = (0..(w * h * 4) as usize).map(|i| (i * 7 % 251) as u8).collect();
    // SAFETY: freshly mapped, unwatched guest view.
    let surface = unsafe { std::slice::from_raw_parts_mut(addr as *mut u8, size) };
    tiling::tile(&p, &lin, surface, w, h);

    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));
    let key = key2d(vk::Format::R8G8B8A8_UNORM, 4, tiling::SWIZZLE_64KB_R_X, w, h, addr);
    let id = images.find_or_create(&mut sched, &mut buffers, &key).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, id, false));
    // GPU contents newer than guest memory.
    images.clear_color(&mut sched, id, vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 1.0] }, None);
    assert!(images.get(id).unwrap().gpu_modified);

    let (x, y) = (123u32, 45u32);
    let off = p.offset(x, y) as u64;
    let page = (addr + off) & !(PAGE_BYTES - 1);
    let other = if page == addr { addr + PAGE_BYTES } else { addr };
    // SAFETY: faulting write resolved by the image watch table.
    unsafe { std::ptr::write_volatile((addr + off) as *mut u32, 0x0102_0304) };
    assert_eq!(perms(page), "rw-", "the page written is unwatched");
    assert_eq!(perms(other), "r--", "the other pages stay watched");
    assert!(images.needs_upload(id));
    assert!(images.prepare(&mut sched, &mut buffers, id, false));
    assert_eq!(images.stats().merges, 1);
    assert!(images.get(id).unwrap().gpu_modified, "the image still holds GPU contents");
    assert_eq!(perms(page), "r--", "watched again after the merge");
    let back = read_image(&dev, &mut sched, &mut images, id, 4);
    let mut guest = vec![0u8; size];
    assert!(mem.read_backing(addr, &mut guest));
    let at = |x: u32, y: u32| back[((y * w + x) * 4) as usize..((y * w + x) * 4 + 4) as usize].to_vec();
    assert_eq!(at(x, y), 0x0102_0304u32.to_le_bytes().to_vec(), "the CPU's bytes win");
    let mut written_page = 0;
    for yy in 0..h {
        for xx in 0..w {
            let o = p.offset(xx, yy) as u64;
            let in_page = (addr + o) & !(PAGE_BYTES - 1) == page;
            let want = if in_page { guest[o as usize..o as usize + 4].to_vec() } else { vec![255, 0, 0, 255] };
            assert_eq!(at(xx, yy), want, "texel ({xx}, {yy})");
            written_page += in_page as u32;
        }
    }
    assert!(written_page > 0 && written_page < w * h);
    // Its memory partly belongs to the CPU now: GPU bytes never go back
    // over it (not every guest write passes the tracker).
    assert!(!images.writeback(&mut sched, &mut buffers, id));
    assert!(!buffers.has_gpu_dirty(addr, size as u64));
    let mut after = vec![0u8; size];
    assert!(mem.read_backing(addr, &mut after));
    assert_eq!(after, guest, "guest memory untouched");
    // GPU work rendering into it again: the memory is still used as this
    // image, so it may be written back.
    images.clear_color(&mut sched, id, vk::ClearColorValue { float32: [0.0, 1.0, 0.0, 1.0] }, None);
    assert!(images.writeback(&mut sched, &mut buffers, id));
    sched.flush_and_wait();
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

/// An alias of a GPU-modified image that the CPU wrote one page of is
/// refilled from that image (with the page merged), not from guest memory:
/// the image still holds the newest bytes everywhere else (a texture view
/// of a render target the CPU wrote next to must not lose the rendering).
#[test]
#[ignore]
fn alias_refills_from_a_newer_image_with_cpu_written_pages() {
    let mem = test_memory();
    let (w, h) = (300u32, 200u32);
    let p = DetileParams::new(tiling::SWIZZLE_64KB_R_X, 4, w).unwrap();
    let size = p.surface_size(h);
    let (view, _) = guest_mapping(mem, 8 << 20);
    let addr = view + (1 << 20);
    let lin: Vec<u8> = (0..(w * h * 4) as usize).map(|i| (i * 5 % 241) as u8).collect();
    // SAFETY: freshly mapped, unwatched guest view.
    let surface = unsafe { std::slice::from_raw_parts_mut(addr as *mut u8, size) };
    tiling::tile(&p, &lin, surface, w, h);

    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));
    // The render target and a texture of the same memory in another format.
    let target = images.find_or_create(&mut sched, &mut buffers, &key2d(vk::Format::R8G8B8A8_UNORM, 4, tiling::SWIZZLE_64KB_R_X, w, h, addr)).unwrap();
    let texture = images.find_or_create(&mut sched, &mut buffers, &key2d(vk::Format::R32_UINT, 4, tiling::SWIZZLE_64KB_R_X, w, h, addr)).unwrap();
    assert_ne!(target, texture);
    assert!(images.prepare(&mut sched, &mut buffers, target, false));
    assert!(images.prepare(&mut sched, &mut buffers, texture, false));
    // The GPU renders the target.
    images.clear_color(&mut sched, target, vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 1.0] }, None);
    assert!(images.get(target).unwrap().gpu_modified);

    // The CPU writes one texel of the shared memory.
    let (x, y) = (123u32, 45u32);
    let off = p.offset(x, y) as u64;
    let page = (addr + off) & !(PAGE_BYTES - 1);
    // SAFETY: faulting write resolved by the image watch table.
    unsafe { std::ptr::write_volatile((addr + off) as *mut u32, 0x0102_0304) };
    assert!(images.needs_refresh(texture));
    assert!(images.prepare(&mut sched, &mut buffers, texture, false));
    // The written page was taken from memory: merged into the target
    // first, or laid over its bytes in the refill.
    assert_eq!(images.stats().merges + images.stats().alias_overlays, 1, "the written page was taken");
    let back = read_image(&dev, &mut sched, &mut images, texture, 4);
    let mut guest = vec![0u8; size];
    assert!(mem.read_backing(addr, &mut guest));
    let at = |x: u32, y: u32| back[((y * w + x) * 4) as usize..((y * w + x) * 4 + 4) as usize].to_vec();
    assert_eq!(at(x, y), 0x0102_0304u32.to_le_bytes().to_vec(), "the CPU's bytes win on their page");
    for yy in 0..h {
        for xx in 0..w {
            let o = p.offset(xx, yy) as u64;
            let want = if (addr + o) & !(PAGE_BYTES - 1) == page { guest[o as usize..o as usize + 4].to_vec() } else { vec![255, 0, 0, 255] };
            assert_eq!(at(xx, yy), want, "texel ({xx}, {yy}): the target's rendering everywhere else");
        }
    }
    sched.flush_and_wait();
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

/// Preparing an image refreshes first only the newer aliases whose
/// CPU-written pages lie over it (only those bytes are read from them),
/// and none when the image is discarded (UE5 transient targets: a large
/// target reusing memory with many small surfaces).
#[test]
#[ignore]
fn alias_pages_written_outside_the_image_are_not_merged_first() {
    let mem = test_memory();
    let sw = tiling::SWIZZLE_64KB_R_X;
    let (view, _) = guest_mapping(mem, 8 << 20);
    let addr = view + (1 << 20);
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));
    // The target: four dense 64 KiB blocks; the texture: its second block.
    let target = images.find_or_create(&mut sched, &mut buffers, &key2d(vk::Format::R8G8B8A8_UNORM, 4, sw, 256, 256, addr)).unwrap();
    let block = 64u64 << 10;
    let texture = images.find_or_create(&mut sched, &mut buffers, &key2d(vk::Format::R32_UINT, 4, sw, 128, 128, addr + block)).unwrap();
    let other = images.find_or_create(&mut sched, &mut buffers, &key2d(vk::Format::R32_UINT, 4, sw, 128, 128, addr + 2 * block)).unwrap();
    for id in [target, texture, other] {
        assert!(images.prepare(&mut sched, &mut buffers, id, false));
    }
    images.clear_color(&mut sched, target, vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 1.0] }, None);

    // The CPU writes into the target's last block, outside the texture.
    // SAFETY: faulting write resolved by the image watch table.
    unsafe { std::ptr::write_volatile((addr + 3 * block + 0x100) as *mut u32, 0x0102_0304) };
    assert!(images.prepare(&mut sched, &mut buffers, texture, false));
    assert_eq!(images.stats().merges, 0, "the written page is not over the texture");
    let back = read_image(&dev, &mut sched, &mut images, texture, 4);
    assert!(back.chunks(4).all(|c| c == [255, 0, 0, 255]), "the texture holds the target's rendering");

    // The CPU writes into the third block, then that image is overwritten
    // whole: nothing is taken from the target first.
    // SAFETY: as above.
    unsafe { std::ptr::write_volatile((addr + 2 * block + 0x100) as *mut u32, 0x0506_0708) };
    assert!(images.prepare(&mut sched, &mut buffers, other, true));
    assert_eq!(images.stats().merges, 0, "a discarded image takes nothing from its aliases");
    sched.flush_and_wait();
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

/// An image refilled from a newer alias that has a page written into it
/// over the image takes that page from memory on top of the alias's bytes
/// (`RPS5E_ALIAS_PAGE_OVERLAY`): the alias itself is not merged or refilled
/// first, and keeps the page for its own next use. With the switch at `0`
/// the alias is merged first (the assertion on `merges` fails).
#[test]
#[ignore]
fn alias_pages_are_laid_over_the_refill_without_preparing_the_alias() {
    let mem = test_memory();
    let sw = tiling::SWIZZLE_64KB_R_X;
    let (view, _) = guest_mapping(mem, 8 << 20);
    let addr = view + (1 << 20);
    let block = 64u64 << 10;
    assert!(mem.write_backing(addr, &vec![0x5Au8; (4 * block) as usize]));
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));
    // The texture: the target's second and third blocks, loaded first.
    let texture = images.find_or_create(&mut sched, &mut buffers, &key2d(vk::Format::R32_UINT, 4, sw, 256, 128, addr + block)).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, texture, false));
    // The target: four dense blocks, rendered after (newer).
    let target = images.find_or_create(&mut sched, &mut buffers, &key2d(vk::Format::R8G8B8A8_UNORM, 4, sw, 256, 256, addr)).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, target, false));
    images.clear_color(&mut sched, target, vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 1.0] }, None);

    // The CPU writes into the second block (the texture's first).
    // SAFETY: faulting write resolved by the image watch table.
    unsafe { std::ptr::write_volatile((addr + block + 0x100) as *mut u32, 0x0102_0304) };
    assert!(images.prepare(&mut sched, &mut buffers, texture, false));
    assert_eq!(images.stats().merges, 0, "the target is not merged for the texture's refill");
    assert_eq!(images.stats().alias_overlays, 1);
    let back = read_image(&dev, &mut sched, &mut images, texture, 4);
    let count = |v: &[u8], t: [u8; 4]| v.chunks(4).filter(|c| *c == t).count();
    let page = (PAGE_BYTES / 4) as usize;
    assert_eq!(count(&back, [255, 0, 0, 255]), 2 * 128 * 128 - page, "outside the written page the texture holds the target's rendering");
    assert_eq!(count(&back, [4, 3, 2, 1]), 1, "the written dword");
    assert_eq!(count(&back, [0x5A; 4]), page - 1, "the rest of the written page is memory's");

    // The target takes its page at its own next use.
    assert!(images.prepare(&mut sched, &mut buffers, target, false));
    assert_eq!(images.stats().merges, 1);
    let back = read_image(&dev, &mut sched, &mut images, target, 4);
    assert_eq!(count(&back, [4, 3, 2, 1]), 1);
    assert_eq!(count(&back, [255, 0, 0, 255]), 4 * 128 * 128 - page);
    sched.flush_and_wait();
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

/// A large image that holds an older alias's GPU bytes and then gets a
/// CPU-written page keeps them: its next refill (from a newer alias
/// elsewhere in its range) merges the written page into its own contents
/// instead of reloading guest memory under the newer alias. Before, the
/// reload replaced the older alias's bytes with stale memory and, being the
/// newest image there, handed those back to the older alias (a render pool
/// that lost a depth buffer this way drew tile-shaped holes).
#[test]
#[ignore]
fn stale_image_refill_keeps_older_alias_bytes() {
    use vk::Format as F;
    let mem = test_memory();
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));

    let (view, _) = guest_mapping(mem, 8 << 20);
    let addr = view + (1 << 20);
    let sw = tiling::SWIZZLE_64KB_R_X;
    let span = 256 << 10;
    let stale: Vec<u8> = (0..span).map(|i| (i * 13 % 241) as u8).collect();
    assert!(mem.write_backing(addr, &stale));

    // D: rendered in the pool's first 64 KiB block.
    let dkey = key2d(F::R8G8B8A8_UINT, 4, sw, 128, 128, addr);
    let d = images.find_or_create(&mut sched, &mut buffers, &dkey).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, d, true));
    images.clear_color(&mut sched, d, vk::ClearColorValue { uint32: [1, 2, 3, 4] }, None);
    // B: the whole pool, takes D's bytes.
    let bkey = key2d(F::R32_UINT, 4, sw, 256, 256, addr);
    assert_eq!(bkey.size(), span as u64);
    let b = images.find_or_create(&mut sched, &mut buffers, &bkey).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, b, false));
    // S: rendered in the third block, newer than B.
    let skey = key2d(F::R8G8B8A8_UINT, 4, sw, 128, 128, addr + (128 << 10));
    let s = images.find_or_create(&mut sched, &mut buffers, &skey).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, s, true));
    images.clear_color(&mut sched, s, vk::ClearColorValue { uint32: [5, 6, 7, 8] }, None);
    // The CPU writes into B's last block (not D's or S's).
    let off = 3 * (64 << 10) + 100;
    // SAFETY: faulting write resolved by the image watch table.
    unsafe { std::ptr::write_volatile((addr + off) as *mut u32, 0x0102_0304) };
    assert!(images.needs_upload(b));
    assert!(!images.needs_upload(d));
    // B is refilled (S on top), then D is asked for again.
    assert!(images.prepare(&mut sched, &mut buffers, b, false));
    assert!(images.prepare(&mut sched, &mut buffers, d, false));
    let back = read_image(&dev, &mut sched, &mut images, d, 4);
    assert!(back.chunks(4).all(|t| t == [1, 2, 3, 4]), "D lost its rendering: {:?}", &back[..16]);
    // B holds D's bytes, S's bytes and the CPU's word.
    let bp = DetileParams::new(sw, 4, 256).unwrap();
    let mut guest = stale.clone();
    tile_at(&DetileParams::new(sw, 4, 128).unwrap(), &vec![1, 2, 3, 4].repeat(128 * 128), 128, 128, &mut guest, 0);
    tile_at(&DetileParams::new(sw, 4, 128).unwrap(), &vec![5, 6, 7, 8].repeat(128 * 128), 128, 128, &mut guest, 128 << 10);
    guest[off as usize..off as usize + 4].copy_from_slice(&0x0102_0304u32.to_le_bytes());
    let mut now = vec![0u8; span];
    assert!(mem.read_backing(addr, &mut now));
    guest[3 * (64 << 10)..].copy_from_slice(&now[3 * (64 << 10)..]);
    assert_eq!(read_image(&dev, &mut sched, &mut images, b, 4), detile_at(&bp, &guest, 0, 256, 256, 4));
    // B's memory still partly belongs to the CPU: no write-back over it.
    assert!(!images.writeback(&mut sched, &mut buffers, b));
    sched.flush_and_wait();
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

/// A depth/stencil image refilled from a newer colour alias over part of
/// its depth plane goes through guest memory (the GPU path does not take
/// stencil planes): its own rendering is written there first, so it keeps
/// its depth wherever the alias does not lie.
#[test]
#[ignore]
fn depth_stencil_refill_through_memory_keeps_its_own_depth() {
    use vk::Format as F;
    let mem = test_memory();
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));

    let (view, _) = guest_mapping(mem, 8 << 20);
    let addr = view + (1 << 20);
    let (w, h) = (256u32, 256u32);
    let zp = DetileParams::new(tiling::SWIZZLE_64KB_Z_X, 4, w).unwrap();
    let size = zp.surface_size(h);
    assert!(mem.write_backing(addr, &vec![0x55u8; size]));
    let mut ykey = ImageKey {
        format: F::D32_SFLOAT_S8_UINT,
        width: w,
        height: h,
        depth: 1,
        layers: 1,
        levels: 1,
        samples: 1,
        dim: ImageDim::D2,
        main: Plane { addr: 0, bytes_per_element: 0, swizzle: 0, pitch: 0, slice_size: 0, bytes: 0, aspect: vk::ImageAspectFlags::COLOR },
        stencil: None,
        transform: Default::default(),
    };
    ykey.main = plane(&ykey, addr, 4, tiling::SWIZZLE_64KB_Z_X, vk::ImageAspectFlags::DEPTH);
    ykey.stencil = Some(plane(&ykey, addr + (2 << 20), 1, tiling::SWIZZLE_64KB_Z_X, vk::ImageAspectFlags::STENCIL));
    let y = images.find_or_create(&mut sched, &mut buffers, &ykey).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, y, true));
    images.clear_depth_stencil(&mut sched, y, Some(0.25), Some(7), None, None);
    // X: a newer colour surface over the first 64 KiB block only.
    let xkey = key2d(F::R32_UINT, 4, tiling::SWIZZLE_64KB_R_X, 128, 128, addr);
    let x = images.find_or_create(&mut sched, &mut buffers, &xkey).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, x, true));
    images.clear_color(&mut sched, x, vk::ClearColorValue { uint32: [0x3f00_0000, 0, 0, 0] }, None);
    assert!(images.needs_refresh(y));
    assert!(images.prepare(&mut sched, &mut buffers, y, false));
    let back = read_image(&dev, &mut sched, &mut images, y, 4);
    let at = |x: u32, y: u32| f32::from_le_bytes(back[((y * w + x) * 4) as usize..((y * w + x) * 4 + 4) as usize].try_into().unwrap());
    let (mut inside, mut outside) = (0, 0);
    for yy in (0..h).step_by(7) {
        for xx in (0..w).step_by(5) {
            if zp.offset(xx, yy) < (64 << 10) {
                assert_eq!(at(xx, yy), 0.5, "texel ({xx}, {yy}) under X");
                inside += 1;
            } else {
                assert_eq!(at(xx, yy), 0.25, "texel ({xx}, {yy}): Y's own depth");
                outside += 1;
            }
        }
    }
    assert!(inside > 0 && outside > 0);
    sched.flush_and_wait();
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

/// A depth/stencil image refilled through guest memory does not first
/// prepare a newer alias that has a page written over it: that alias keeps
/// the CPU's claim on its memory and is not written back either way
/// (`RPS5E_ALIAS_PAGE_OVERLAY`; with `=0` the alias is merged first and the
/// assertion on `merges` fails).
#[test]
#[ignore]
fn depth_stencil_refill_through_memory_leaves_written_aliases_alone() {
    use vk::Format as F;
    let mem = test_memory();
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));

    let (view, _) = guest_mapping(mem, 8 << 20);
    let addr = view + (1 << 20);
    let (w, h) = (256u32, 256u32);
    let zp = DetileParams::new(tiling::SWIZZLE_64KB_Z_X, 4, w).unwrap();
    let size = zp.surface_size(h);
    assert!(mem.write_backing(addr, &vec![0x55u8; size]));
    let mut ykey = ImageKey {
        format: F::D32_SFLOAT_S8_UINT,
        width: w,
        height: h,
        depth: 1,
        layers: 1,
        levels: 1,
        samples: 1,
        dim: ImageDim::D2,
        main: Plane { addr: 0, bytes_per_element: 0, swizzle: 0, pitch: 0, slice_size: 0, bytes: 0, aspect: vk::ImageAspectFlags::COLOR },
        stencil: None,
        transform: Default::default(),
    };
    ykey.main = plane(&ykey, addr, 4, tiling::SWIZZLE_64KB_Z_X, vk::ImageAspectFlags::DEPTH);
    ykey.stencil = Some(plane(&ykey, addr + (2 << 20), 1, tiling::SWIZZLE_64KB_Z_X, vk::ImageAspectFlags::STENCIL));
    let y = images.find_or_create(&mut sched, &mut buffers, &ykey).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, y, true));
    images.clear_depth_stencil(&mut sched, y, Some(0.25), Some(7), None, None);
    // X: a newer colour surface over the first two 64 KiB blocks.
    let xkey = key2d(F::R32_UINT, 4, tiling::SWIZZLE_64KB_R_X, 256, 128, addr);
    let x = images.find_or_create(&mut sched, &mut buffers, &xkey).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, x, true));
    images.clear_color(&mut sched, x, vk::ClearColorValue { uint32: [0x3f00_0000, 0, 0, 0] }, None);
    // The CPU writes into X's second block.
    // SAFETY: faulting write resolved by the image watch table.
    unsafe { std::ptr::write_volatile((addr + (64 << 10) + 0x40) as *mut u32, 0x3e80_0000) };
    assert!(images.needs_refresh(y));
    assert!(images.prepare(&mut sched, &mut buffers, y, false));
    assert_eq!(images.stats().merges, 0, "X is not merged for Y's refill");
    // X takes its page at its own next use.
    assert!(images.prepare(&mut sched, &mut buffers, x, false));
    assert_eq!(images.stats().merges, 1);
    sched.flush_and_wait();
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

/// The DCC keys of a sampled image that a GPU fill wrote last are decided
/// from the fill's value (`RPS5E_KNOWN_FILLS`): the image is cleared to the
/// code's colour without downloading the keys. With `=0` the keys are
/// downloaded first (the assertion on `known_fill_keys` fails).
#[test]
#[ignore]
fn sampled_keys_written_by_a_gpu_fill_need_no_download() {
    let mem = test_memory();
    let sw = tiling::SWIZZLE_64KB_R_X;
    let (view, _) = guest_mapping(mem, 8 << 20);
    let addr = view + (1 << 20);
    let keys = view + (4 << 20);
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));
    let id = images.find_or_create(&mut sched, &mut buffers, &key2d(vk::Format::R8G8B8A8_UNORM, 4, sw, 256, 256, addr)).unwrap();
    // The CPU wrote the texels: the image must be loaded before its read.
    assert!(mem.write_backing(addr, &vec![0x33u8; 256 << 10]));
    images.invalidate_range(addr, 256 << 10);
    // The keys: cleared to code 0x00 (black, alpha 0) by a GPU fill.
    assert!(buffers.fill(&mut sched, keys, 64 << 10, 0).unwrap());
    let r = images.sampled_dcc(&mut sched, &mut buffers, id, super::sampled_dcc::DescriptorDcc { address: keys, alpha_on_msb: false });
    assert_eq!(images.stats().known_fill_keys, 1, "decided from the fill, not from downloaded keys");
    assert_eq!(r, super::sampled_dcc::SampledKeys::Cleared(0));
    let back = read_image(&dev, &mut sched, &mut images, id, 4);
    assert!(back.chunks(4).all(|c| c == [0, 0, 0, 0]), "cleared to the code's colour");
    sched.flush_and_wait();
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

/// CPU writes and re-watches of a partially resident image racing on two
/// threads keep the page's watch count equal to what the entry holds. A
/// re-watch that added its watches after releasing the table lock let a
/// write in between remove a watch that was not there yet; the one added
/// next belonged to nobody, and later stores to the page faulted for ever.
#[test]
fn watch_table_rewatch_racing_cpu_writes_leaks_no_watch() {
    let mem = test_memory();
    let (view, _) = guest_mapping(mem, 4 << 20);
    let table = Arc::new(WatchTable::new(mem.pages.clone()));
    let a = view + 16 * PAGE_BYTES;
    let page = |i: u64| a + i * PAGE_BYTES;
    // Partially resident: pages 4..6 uncommitted, so the entry is watched
    // per page.
    unmap_guest(mem, page(4), 2 * PAGE_BYTES);
    table.insert(1, a, page(16));
    table.rearm(1, true);
    assert!(table.is_sparse(1));
    let target = page(1);
    std::thread::scope(|s| {
        let t = table.clone();
        s.spawn(move || {
            for _ in 0..200_000 {
                t.mark_cpu_write(target + 8, 8, false);
            }
        });
        for _ in 0..200_000 {
            table.rearm_with(1, true, true, None);
        }
    });
    // Quiet now: the entry watches the page once, and a write unwatches it.
    table.rearm_with(1, true, true, None);
    assert_eq!(mem.pages.watchers(target).1, 1, "watch count after re-watching");
    table.mark_cpu_write(target + 8, 8, false);
    assert_eq!(mem.pages.watchers(target).1, 0, "a watch nobody holds was left on the page");
    table.remove(1);
}

/// A partially resident image (some pages of its range not committed) is
/// watched on its committed pages, and a page committed, written or
/// unmapped later makes exactly that page stale.
#[test]
fn watch_table_tracks_partially_resident_images_per_page() {
    let mem = test_memory();
    let (view, _) = guest_mapping(mem, 4 << 20);
    let table = Arc::new(WatchTable::new(mem.pages.clone()));
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(table.clone())), Arc::new(NoUnmap));
    let a = view + 16 * PAGE_BYTES;
    let page = |i: u64| a + i * PAGE_BYTES;
    unmap_guest(mem, page(4), 4 * PAGE_BYTES);
    unmap_guest(mem, page(12), PAGE_BYTES);
    table.insert(1, a, page(16));
    assert!(mem.is_watchable(a, 16 * PAGE_BYTES));
    table.rearm(1, true);
    assert!(table.is_sparse(1));
    assert_eq!(table.state(1), (true, false), "watched although partly unmapped");
    assert_eq!(perms(page(0)), "r--");
    assert_eq!(perms(page(15)), "r--");
    assert_eq!(perms(page(4)), "---");

    // Committing a page: only that page is stale, the others stay watched.
    map_guest(mem, page(12), PAGE_BYTES);
    assert_eq!(table.state(1), (true, true));
    assert!(!table.stale_whole(1));
    assert_eq!(table.dirty_runs(1), Some(vec![(page(12), page(13))]));
    assert_eq!(perms(page(12)), "rw-", "the committed page is unwatched until the image takes it");
    assert_eq!(perms(page(11)), "r--");
    assert!(!table.may_write_back(1));
    // The image took the page (merged): watched again, its memory still
    // its own (a mapping is not a CPU store).
    table.rearm_with(1, true, true, None);
    assert_eq!(table.state(1), (true, false));
    assert_eq!(perms(page(12)), "r--");
    assert!(table.may_write_back(1));
    assert!(table.is_sparse(1), "pages 4..8 are still uncommitted");

    // A CPU write: that page only, and the memory is the CPU's now.
    // SAFETY: faulting write resolved by the image watch table.
    unsafe { std::ptr::write_volatile((page(1) + 8) as *mut u32, 7) };
    assert_eq!(table.dirty_runs(1), Some(vec![(page(1), page(2))]));
    assert_eq!(perms(page(0)), "r--");
    table.rearm_with(1, true, true, None);
    assert!(!table.may_write_back(1));

    // Unmapping a page (what ImageCache::unmap does for an image keeping
    // other pages): that page is stale and reads as zeros from now on.
    unmap_guest(mem, page(13), PAGE_BYTES);
    table.mark_mapped(page(13), PAGE_BYTES);
    assert_eq!(table.dirty_runs(1), Some(vec![(page(13), page(14))]));
    table.rearm(1, true);
    assert_eq!(table.state(1), (true, false));

    // A wholly committed image keeps the whole-image mode.
    table.insert(2, view, view + 8 * PAGE_BYTES);
    table.rearm(2, true);
    assert!(!table.is_sparse(2));
    map_guest(mem, page(5), PAGE_BYTES);
    assert_eq!(table.state(2), (true, false), "a mapping outside its range");
    table.remove(1);
    table.remove(2);
}

/// A partially resident texture on the device: loaded with zeros for its
/// uncommitted pages, watched, a page committed later is merged on its own
/// (not a reload), and a write-back never stores into uncommitted pages.
#[test]
#[ignore]
fn image_cache_partially_resident_texture() {
    let mem = test_memory();
    let (w, h) = (300u32, 200u32);
    let p = DetileParams::new(tiling::SWIZZLE_64KB_R_X, 4, w).unwrap();
    let size = p.surface_size(h);
    let (view, alias) = guest_mapping(mem, 8 << 20);
    let addr = view + (1 << 20);
    let backing = |a: u64| alias + (a - view);
    let lin: Vec<u8> = (0..(w * h * 4) as usize).map(|i| (i * 7 % 251) as u8 | 1).collect();
    // SAFETY: freshly mapped, unwatched guest view.
    let surface = unsafe { std::slice::from_raw_parts_mut(addr as *mut u8, size) };
    tiling::tile(&p, &lin, surface, w, h);
    // Pages 5..8 and 20 of the surface are not committed.
    let pg = |i: u64| addr + i * PAGE_BYTES;
    unmap_guest(mem, pg(5), 3 * PAGE_BYTES);
    unmap_guest(mem, pg(20), PAGE_BYTES);

    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));
    let expected = |mem: &GuestGpuMemory| {
        let mut guest = vec![0u8; size];
        assert!(mem.read_committed(addr, &mut guest));
        let mut d = vec![0u8; (w * h * 4) as usize];
        tiling::detile(&p, &guest, &mut d, w, h);
        d
    };
    let key = key2d(vk::Format::R8G8B8A8_UNORM, 4, tiling::SWIZZLE_64KB_R_X, w, h, addr);
    let id = images.find_or_create(&mut sched, &mut buffers, &key).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, id, false));
    let first = read_image(&dev, &mut sched, &mut images, id, 4);
    assert_eq!(first, expected(mem), "uncommitted pages load as zeros");
    assert!(first.chunks(4).any(|t| t == [0; 4]) && first.chunks(4).any(|t| t != [0; 4]));
    assert!(!images.needs_upload(id), "watched: not reloaded at the next use");
    assert_eq!(perms(pg(0)), "r--");
    assert_eq!(images.stats().uploads, 1);

    // The guest commits page 6 (new bytes in its memory): only that page
    // is taken at the next use.
    // SAFETY: the alias maps the whole test memfd.
    unsafe { std::ptr::write_bytes(backing(pg(6)) as *mut u8, 0x77, PAGE_BYTES as usize) };
    map_guest(mem, pg(6), PAGE_BYTES);
    assert!(images.needs_upload(id));
    let before = images.stats();
    assert!(images.prepare(&mut sched, &mut buffers, id, false));
    let after = images.stats();
    assert_eq!(after.merges - before.merges, 1, "merged, not reloaded");
    assert_eq!(after.uploaded_bytes - before.uploaded_bytes, PAGE_BYTES);
    let second = read_image(&dev, &mut sched, &mut images, id, 4);
    assert_eq!(second, expected(mem));
    assert!(second.chunks(4).any(|t| t == [0x77; 4]));
    assert_eq!(perms(pg(6)), "r--", "watched again");

    // GPU contents written back: the uncommitted pages' memory is left alone.
    // SAFETY: the alias maps the whole test memfd.
    unsafe { std::ptr::write_bytes(backing(pg(20)) as *mut u8, 0x3C, PAGE_BYTES as usize) };
    images.clear_color(&mut sched, id, vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 1.0] }, None);
    assert!(images.writeback(&mut sched, &mut buffers, id));
    assert!(buffers.download(&mut sched, addr, size as u64));
    // SAFETY: as above.
    let hole = unsafe { std::slice::from_raw_parts(backing(pg(20)) as *const u8, PAGE_BYTES as usize) };
    assert!(hole.iter().all(|&b| b == 0x3C), "nothing stored into an uncommitted page");
    let back = expected(mem);
    for (i, t) in back.chunks(4).enumerate() {
        let (x, y) = (i as u32 % w, i as u32 / w);
        let a = addr + p.offset(x, y) as u64;
        let uncommitted = ((pg(5)..pg(8)).contains(&a) && !(pg(6)..pg(7)).contains(&a)) || (pg(20)..pg(21)).contains(&a);
        assert_eq!(t, if uncommitted { [0; 4] } else { [255, 0, 0, 255] }, "texel ({x}, {y})");
    }
    sched.flush_and_wait();
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

/// 16-bit guest depth in a D32_SFLOAT_S8_UINT image (the fallback for
/// devices without D16_UNORM_S8_UINT): uploads widen to float, write-backs
/// narrow to the same 16-bit values.
#[test]
#[ignore]
fn image_cache_depth16_in_float_depth_round_trips() {
    let mem = test_memory();
    let (w, h) = (128u32, 64u32);
    let p = DetileParams::new(tiling::SWIZZLE_64KB_Z_X, 2, w).unwrap();
    let size = p.surface_size(h);
    let (view, _) = guest_mapping(mem, 8 << 20);
    let addr = view + (1 << 20);
    let lin: Vec<u8> = (0..w * h).flat_map(|i| (((i * 2731) ^ (i >> 3)) as u16).to_le_bytes()).collect();
    // SAFETY: freshly mapped, unwatched guest view.
    let surface = unsafe { std::slice::from_raw_parts_mut(addr as *mut u8, size) };
    tiling::tile(&p, &lin, surface, w, h);

    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));
    let key = ImageKey {
        format: vk::Format::D32_SFLOAT_S8_UINT,
        width: w,
        height: h,
        depth: 1,
        layers: 1,
        levels: 1,
        samples: 1,
        dim: ImageDim::D2,
        main: Plane {
            addr,
            bytes_per_element: 2,
            swizzle: tiling::SWIZZLE_64KB_Z_X,
            pitch: p.pitch,
            slice_size: size as u64,
            bytes: size as u64,
            aspect: vk::ImageAspectFlags::DEPTH,
        },
        stencil: None,
        transform: Default::default(),
    };
    let id = images.find_or_create(&mut sched, &mut buffers, &key).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, id, false));
    // Round trip: the uploaded floats narrow back to the guest values.
    let tick = sched.current_tick();
    images.mark_gpu_written(id, tick);
    assert!(mem.write_backing(addr, &vec![0u8; size]));
    assert!(images.writeback(&mut sched, &mut buffers, id));
    assert!(buffers.download(&mut sched, addr, size as u64));
    let mut guest = vec![0u8; size];
    assert!(mem.read_backing(addr, &mut guest));
    let mut back = vec![0u8; lin.len()];
    tiling::detile(&p, &guest, &mut back, w, h);
    assert!(back == lin, "16-bit depth changed through the float image");
    // A clear to 0.5 narrows to 32768.
    images.clear_depth_stencil(&mut sched, id, Some(0.5), None, None, None);
    assert!(images.writeback(&mut sched, &mut buffers, id));
    assert!(buffers.download(&mut sched, addr, size as u64));
    let mut d = [0u8; 2];
    assert!(mem.read_backing(addr + p.offset(77, 33) as u64, &mut d));
    assert_eq!(u16::from_le_bytes(d), 32768);
    sched.flush_and_wait();
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

/// A small image inside a large GPU-modified one takes only the large
/// one's block rows over it, and the large one takes the small one's
/// bytes back without losing its others; views of fewer layers of an
/// array are the array itself.
#[test]
#[ignore]
fn image_cache_windows_and_array_layers() {
    use vk::Format as F;
    let mem = test_memory();
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));
    let (view, _) = guest_mapping(mem, 16 << 20);
    let addr = view + (1 << 20);
    let sw = tiling::SWIZZLE_64KB_R_X;

    // X: 1024x512 RGBA8 (four rows of 64 KiB blocks, 512 KiB each), rendered.
    let xkey = key2d(F::R8G8B8A8_UNORM, 4, sw, 1024, 512, addr);
    let xp = DetileParams::new(sw, 4, 1024).unwrap();
    let x = images.find_or_create(&mut sched, &mut buffers, &xkey).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, x, true));
    images.clear_color(&mut sched, x, vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 1.0] }, None);
    images.clear_color(&mut sched, x, vk::ClearColorValue { float32: [0.0, 1.0, 0.0, 1.0] }, rect(100, 260, 300, 100));
    let xlin = read_image(&dev, &mut sched, &mut images, x, 4);
    let mut guest = vec![0u8; xkey.size() as usize];
    tile_at(&xp, &xlin, 1024, 512, &mut guest, 0);

    // Z: 128x128 RGBA16 in X's third block row.
    let zoff = 2 * (512 << 10);
    let zkey = key2d(F::R16G16B16A16_UINT, 8, sw, 128, 128, addr + zoff as u64);
    let zp = DetileParams::new(sw, 8, 128).unwrap();
    let z = images.find_or_create(&mut sched, &mut buffers, &zkey).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, z, false));
    assert_eq!(read_image(&dev, &mut sched, &mut images, z, 8), detile_at(&zp, &guest, zoff, 128, 128, 8), "Z reinterprets X's bytes over it");

    // Z rendered; X takes Z's bytes and keeps its own elsewhere.
    images.clear_color(&mut sched, z, vk::ClearColorValue { uint32: [7, 8, 9, 10] }, rect(10, 10, 50, 60));
    let zlin = read_image(&dev, &mut sched, &mut images, z, 8);
    assert!(images.prepare(&mut sched, &mut buffers, x, false));
    tile_at(&zp, &zlin, 128, 128, &mut guest, zoff);
    assert_eq!(read_image(&dev, &mut sched, &mut images, x, 4), detile_at(&xp, &guest, 0, 1024, 512, 4), "X = its bytes with Z's on top");

    // A 4-layer array, then views of 2 and 4 layers.
    let base = view + (8 << 20);
    let mut a4 = key2d(F::R32_UINT, 4, sw, 256, 256, base);
    a4.layers = 4;
    a4.main = plane(&a4, base, 4, sw, vk::ImageAspectFlags::COLOR);
    let mut a2 = a4;
    a2.layers = 2;
    a2.main = plane(&a2, base, 4, sw, vk::ImageAspectFlags::COLOR);
    assert!(a2.layers_within(&a4) && !a4.layers_within(&a2));
    // The 2-layer view first, rendered; the 4-layer array takes it over.
    let i2 = images.find_or_create(&mut sched, &mut buffers, &a2).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, i2, true));
    images.clear_color(&mut sched, i2, vk::ClearColorValue { uint32: [0x1234, 0, 0, 0] }, None);
    let i4 = images.find_or_create(&mut sched, &mut buffers, &a4).unwrap();
    assert!(images.get(i2).is_none(), "the 2-layer image went into the array");
    assert_eq!(images.stats().absorbed, 1);
    assert_eq!(images.find_or_create(&mut sched, &mut buffers, &a2).unwrap(), i4, "a view of fewer layers is the array");
    let layer0 = read_image(&dev, &mut sched, &mut images, i4, 4);
    assert!(layer0.chunks(4).all(|t| t == 0x1234u32.to_le_bytes()), "the rendered layer moved over");
    sched.flush_and_wait();
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

/// sRGB images have no storage support of their own, but their storage
/// views use the UNORM format of the same layout: they are created with
/// STORAGE usage and EXTENDED_USAGE, and a UNORM view of them is made (a
/// compute pass writing an sRGB texture through a storage view bound a
/// dummy and lost its writes).
#[test]
#[ignore]
fn image_cache_srgb_images_take_storage_views() {
    use vk::Format as F;
    let mem = test_memory();
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    let (view, _) = guest_mapping(mem, 4 << 20);
    let key = key2d(F::R8G8B8A8_SRGB, 4, tiling::SWIZZLE_64KB_R_X, 64, 64, view + (1 << 20));
    let id = images.find_or_create(&mut sched, &mut buffers, &key).unwrap();
    assert!(images.get(id).unwrap().image.usage.contains(vk::ImageUsageFlags::STORAGE));
    let range = ViewRange { ty: vk::ImageViewType::TYPE_2D, base_level: 0, levels: 1, base_layer: 0, layers: 1, min_lod: 0.0 };
    assert!(images.view_range(id, F::R8G8B8A8_UNORM, vk::ImageAspectFlags::COLOR, range, vk::ComponentMapping::default()).is_some());
    sched.flush_and_wait();
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

/// GPU work writing an image's memory through a buffer finds the image's
/// GPU contents written back first (the store lands over them, the rest of
/// the image stays); one write-back per change of the image.
#[test]
#[ignore]
fn buffer_write_over_a_gpu_modified_image_writes_it_back_first() {
    use vk::Format as F;
    let mem = test_memory();
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));
    let (view, _) = guest_mapping(mem, 4 << 20);
    let base = view + (1 << 20);
    let id = images.find_or_create(&mut sched, &mut buffers, &key2d(F::R8G8B8A8_UNORM, 4, tiling::SWIZZLE_64KB_R_X, 64, 64, base)).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, id, true));
    images.clear_color(&mut sched, id, vk::ClearColorValue { float32: [1.0, 0.0, 1.0, 1.0] }, None);
    let size = images.get(id).unwrap().key.size();
    // A store into the image's last 256 bytes.
    assert_eq!(images.writeback_before_buffer_write(&mut sched, &mut buffers, base + size - 256, 256), 1);
    assert_eq!(images.writeback_before_buffer_write(&mut sched, &mut buffers, base + size - 256, 256), 0, "unchanged since");
    assert!(buffers.download(&mut sched, base, size));
    let mut bytes = vec![0u8; size as usize];
    assert!(mem.read_backing(base, &mut bytes));
    // Every texel of the 64x64 image (the rest of the 64 KiB block is padding).
    assert_eq!(bytes.chunks(4).filter(|p| *p == [255, 0, 255, 255]).count(), 64 * 64, "the mirror holds the image");
    images.invalidate_shader_write(base + size - 256, 256);
    assert_eq!(images.writeback_before_buffer_write(&mut sched, &mut buffers, base, 256), 0, "a stale image is not written back");
    sched.flush_and_wait();
    dev.wait_idle();
}

/// A GPU write through a buffer makes exactly the images over its range
/// stale (looked up through the page index), also one that starts pages
/// before the range.
#[test]
#[ignore]
fn buffer_write_invalidates_the_images_over_its_range() {
    use vk::Format as F;
    let mem = test_memory();
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));
    let (view, _) = guest_mapping(mem, 16 << 20);
    let sw = tiling::SWIZZLE_64KB_R_X;
    // A large image (4 MiB) and two small ones after it.
    let big = images.find_or_create(&mut sched, &mut buffers, &key2d(F::R8G8B8A8_UNORM, 4, sw, 1024, 1024, view + (1 << 20))).unwrap();
    let a = images.find_or_create(&mut sched, &mut buffers, &key2d(F::R8G8B8A8_UNORM, 4, sw, 64, 64, view + (8 << 20))).unwrap();
    let b = images.find_or_create(&mut sched, &mut buffers, &key2d(F::R8G8B8A8_UNORM, 4, sw, 64, 64, view + (10 << 20))).unwrap();
    for id in [big, a, b] {
        assert!(images.prepare(&mut sched, &mut buffers, id, false));
        assert!(!images.needs_upload(id));
    }
    let big_end = view + (1 << 20) + images.get(big).unwrap().key.size();
    // The last bytes of the large image, three pages past its start.
    images.invalidate_shader_write(big_end - 256, 128);
    assert!(images.needs_upload(big), "a write into its last page");
    assert!(!images.needs_upload(a) && !images.needs_upload(b));
    assert!(images.prepare(&mut sched, &mut buffers, big, false));
    // A range between the images hits none.
    images.invalidate_range(big_end + 4096, 4096);
    assert!(!images.needs_upload(big) && !images.needs_upload(a) && !images.needs_upload(b));
    // A range over A and B.
    images.invalidate_range(view + (8 << 20), 3 << 20);
    assert!(images.needs_upload(a) && images.needs_upload(b));
    assert!(!images.needs_upload(big));
    sched.flush_and_wait();
    dev.wait_idle();
}

/// Lookup generations (memoised texture lookups): a change to an image
/// moves the generations of the image and of the images overlapping it,
/// and of no other image; the lookup epoch stays.
#[test]
#[ignore]
fn lookup_generations_follow_the_image_and_its_aliases() {
    use vk::Format as F;
    let mem = test_memory();
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));
    let (view, _) = guest_mapping(mem, 8 << 20);
    let sw = tiling::SWIZZLE_64KB_R_X;
    let (a_addr, b_addr) = (view + (1 << 20), view + (2 << 20));
    let a = images.find_or_create(&mut sched, &mut buffers, &key2d(F::R8G8B8A8_UNORM, 4, sw, 64, 64, a_addr)).unwrap();
    let b = images.find_or_create(&mut sched, &mut buffers, &key2d(F::R8G8B8A8_UNORM, 4, sw, 64, 64, b_addr)).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, a, false));
    assert!(images.prepare(&mut sched, &mut buffers, b, false));
    let epoch = images.lookup_epoch();
    let g = |images: &ImageCache, id: u64| images.image_generation(id).unwrap();
    let (ga, gb) = (g(&images, a), g(&images, b));
    // A new image over A's memory.
    let c = images.find_or_create(&mut sched, &mut buffers, &key2d(F::R32_UINT, 4, sw, 64, 64, a_addr)).unwrap();
    assert_ne!(g(&images, a), ga, "an image created over it");
    assert_eq!(g(&images, b), gb, "an image created elsewhere");
    // C rendered into: it holds the newest contents of A's memory now.
    let (ga, gc) = (g(&images, a), g(&images, c));
    images.mark_gpu_written(c, sched.current_tick());
    assert_ne!(g(&images, a), ga, "an alias written by the GPU");
    assert_ne!(g(&images, c), gc);
    assert_eq!(g(&images, b), gb);
    // Rendered into again while newest: no lookup changes.
    let (ga, gc) = (g(&images, a), g(&images, c));
    images.mark_gpu_written(c, sched.current_tick());
    assert_eq!((g(&images, a), g(&images, c)), (ga, gc), "the newest image written again");
    // A is refreshed from C at its next use.
    assert!(images.needs_refresh(a));
    assert!(images.prepare(&mut sched, &mut buffers, a, false));
    assert_ne!(g(&images, a), ga, "refreshed");
    assert_ne!(g(&images, c), gc, "its alias");
    assert_eq!(g(&images, b), gb);
    let (ga, gc) = (g(&images, a), g(&images, c));
    assert!(images.prepare(&mut sched, &mut buffers, a, false));
    assert_eq!((g(&images, a), g(&images, c)), (ga, gc), "a use that does nothing");
    // B written for the first time, then cleared: B only.
    images.mark_gpu_written(b, sched.current_tick());
    assert_ne!(g(&images, b), gb);
    let gb = g(&images, b);
    images.clear_color(&mut sched, b, vk::ClearColorValue { float32: [1.0; 4] }, None);
    assert_ne!(g(&images, b), gb, "cleared");
    assert_eq!((g(&images, a), g(&images, c)), (ga, gc), "an unrelated image cleared");
    // C goes: A's lookups may find another image.
    images.delete(&mut sched, c);
    assert_ne!(g(&images, a), ga, "an alias deleted");
    if image_generations_enabled() {
        assert_eq!(images.lookup_epoch(), epoch, "no change reached every lookup");
    }
    sched.flush_and_wait();
    dev.wait_idle();
}

/// A compute pass writes the blocks of a BC3 surface through an
/// uncompressed R32G32B32A32_UINT image of the same memory: the BC3 image
/// takes the texels as its blocks in one image copy, nothing goes through
/// guest memory, and the blocks are what the guest layout makes of them.
#[test]
#[ignore]
fn compressed_image_refills_from_its_block_view_by_one_copy() {
    use vk::Format as F;
    let mem = test_memory();
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));
    let (view, _) = guest_mapping(mem, 8 << 20);
    let addr = view + (1 << 20);
    let sw = tiling::SWIZZLE_64KB_S;
    let stale: Vec<u8> = (0..(1 << 20)).map(|i| (i * 13 % 241) as u8).collect();
    assert!(mem.write_backing(addr, &stale));

    let xkey = key2d(F::R32G32B32A32_UINT, 16, sw, 64, 64, addr);
    let ykey = key2d(F::BC3_UNORM_BLOCK, 16, sw, 256, 256, addr);
    assert!(super::block_view_of(&xkey, &ykey));
    assert!(!super::block_view_of(&ykey, &xkey));
    let x = images.find_or_create(&mut sched, &mut buffers, &xkey).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, x, true));
    images.clear_color(&mut sched, x, vk::ClearColorValue { uint32: [1, 2, 3, 4] }, None);
    images.clear_color(&mut sched, x, vk::ClearColorValue { uint32: [0x1111, 0x2222, 0x3333, 0x4444] }, rect(5, 7, 20, 9));
    let xlin = read_image(&dev, &mut sched, &mut images, x, 16);

    let y = images.find_or_create(&mut sched, &mut buffers, &ykey).unwrap();
    assert!(images.needs_refresh(y));
    assert!(images.prepare(&mut sched, &mut buffers, y, false));
    let s = images.stats();
    let direct = super::direct_copy_enabled();
    assert_eq!((s.retiles, s.direct_copies, s.writebacks, s.uploads), (1, direct as u64, 0, 0));
    // One byte per texel of a BC3 image: its 64x64 blocks of 16 bytes.
    assert_eq!(read_image(&dev, &mut sched, &mut images, y, 1), xlin, "block (x, y) is texel (x, y)");
    assert!(images.get(y).unwrap().gpu_modified);
    assert!(!images.needs_refresh(y));
    let mut now = vec![0u8; stale.len()];
    assert!(mem.read_backing(addr, &mut now));
    assert!(now == stale, "no GPU data went through guest memory");
    sched.flush_and_wait();
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

/// Array key of `layers` layers at `addr`.
fn key_layers(format: vk::Format, bpe: u32, swizzle: u32, width: u32, height: u32, layers: u32, addr: u64) -> ImageKey {
    let mut key = key2d(format, bpe, swizzle, width, height, addr);
    key.layers = layers;
    key.main = plane(&key, addr, bpe, swizzle, vk::ImageAspectFlags::COLOR);
    key
}

#[test]
fn layer_shift_matches_views_of_one_array() {
    use vk::Format as F;
    let sw = tiling::SWIZZLE_64KB_S;
    let a = key_layers(F::R32_UINT, 4, sw, 256, 128, 4, 0x1000_0000);
    let s = a.main.slice_size;
    assert_eq!(super::layer_shift(&a, &key_layers(F::R32_UINT, 4, sw, 256, 128, 1, 0x1000_0000 + 3 * s)), Some(3));
    assert_eq!(super::layer_shift(&a, &key_layers(F::R32_UINT, 4, sw, 256, 128, 3, 0x1000_0000 + s)), Some(1));
    assert_eq!(super::layer_shift(&key_layers(F::R32_UINT, 4, sw, 256, 128, 1, 0x1000_0000 + 3 * s), &a), Some(-3));
    assert_eq!(super::layer_shift(&a, &key_layers(F::R32_UINT, 4, sw, 256, 128, 1, 0x1000_0000 + 4 * s)), None, "past the last layer");
    assert_eq!(super::layer_shift(&a, &key_layers(F::R32_UINT, 4, sw, 256, 128, 1, 0x1000_0000 + s / 2)), None, "not a layer boundary");
    assert_eq!(super::layer_shift(&a, &key_layers(F::R32_SFLOAT, 4, sw, 256, 128, 1, 0x1000_0000 + s)), None, "another format");
    assert_eq!(super::layer_shift(&a, &key_layers(F::R32_UINT, 4, sw, 256, 64, 1, 0x1000_0000 + s)), None, "another extent");
}

/// A 4-layer array is rendered into; single-layer and 3-layer views of
/// the same memory starting at layers 3 and 1 take their layers from it
/// with image copies: the right layers, nothing through guest memory.
#[test]
#[ignore]
fn layer_views_refill_from_their_array_by_layer_copies() {
    use vk::Format as F;
    let mem = test_memory();
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));
    let (view, _) = guest_mapping(mem, 8 << 20);
    let addr = view + (1 << 20);
    let sw = tiling::SWIZZLE_64KB_S;
    let akey = key_layers(F::R32_UINT, 4, sw, 256, 128, 4, addr);
    let s = akey.main.slice_size;
    let stale: Vec<u8> = (0..akey.size() as usize).map(|i| (i * 13 % 241) as u8).collect();
    assert!(mem.write_backing(addr, &stale));

    let a = images.find_or_create(&mut sched, &mut buffers, &akey).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, a, true));
    images.clear_color(&mut sched, a, vk::ClearColorValue { uint32: [0x1111_1111; 4] }, None);
    images.clear_color_layers(&mut sched, a, vk::ClearColorValue { uint32: [0x3333_3333; 4] }, 3, 1);
    images.mark_gpu_written(a, sched.current_tick());

    let y = images.find_or_create(&mut sched, &mut buffers, &key_layers(F::R32_UINT, 4, sw, 256, 128, 1, addr + 3 * s)).unwrap();
    assert!(images.needs_refresh(y));
    assert!(images.prepare(&mut sched, &mut buffers, y, false));
    let z = images.find_or_create(&mut sched, &mut buffers, &key_layers(F::R32_UINT, 4, sw, 256, 128, 3, addr + s)).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, z, false));
    let st = images.stats();
    let on = super::layer_copy_enabled();
    assert_eq!((st.retiles, st.direct_copies, st.writebacks), (2, 2 * on as u64, 0));
    let ylin = read_image(&dev, &mut sched, &mut images, y, 4);
    assert!(ylin.chunks(4).all(|c| c == [0x33; 4]), "layer 0 of the view is layer 3 of the array");
    let zlin = read_image(&dev, &mut sched, &mut images, z, 4);
    assert!(zlin.chunks(4).all(|c| c == [0x11; 4]), "layer 0 of the view is layer 1 of the array");
    assert!(!images.needs_refresh(y) && !images.needs_refresh(z));
    let mut now = vec![0u8; stale.len()];
    assert!(mem.read_backing(addr, &mut now));
    assert!(now == stale, "no GPU data went through guest memory");
    sched.flush_and_wait();
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

#[test]
fn direct_copies_pair_only_size_compatible_aspects() {
    use vk::{Format as F, ImageAspectFlags as A};
    assert!(super::copy_compatible(false, (F::R32_UINT, A::COLOR), (F::R8G8B8A8_UNORM, A::COLOR)));
    assert!(super::copy_compatible(true, (F::D16_UNORM, A::DEPTH), (F::R16_UNORM, A::COLOR)));
    assert!(super::copy_compatible(true, (F::R32_SFLOAT, A::COLOR), (F::D32_SFLOAT, A::DEPTH)));
    assert!(!super::copy_compatible(false, (F::D16_UNORM, A::DEPTH), (F::R16_UNORM, A::COLOR)), "depth to colour needs maintenance8");
    assert!(!super::copy_compatible(true, (F::D16_UNORM, A::DEPTH), (F::R8G8_UNORM, A::COLOR)), "not a listed pair");
    assert!(!super::copy_compatible(true, (F::D32_SFLOAT_S8_UINT, A::DEPTH), (F::R32_SFLOAT, A::COLOR)));
    assert!(super::copy_compatible(false, (F::D32_SFLOAT, A::DEPTH), (F::D32_SFLOAT, A::DEPTH)));
}

#[test]
fn block_views_need_the_same_element_grid_and_layout() {
    use vk::Format as F;
    let sw = tiling::SWIZZLE_64KB_S;
    let at = 0x20_0000_0000;
    let y = key2d(F::BC3_UNORM_BLOCK, 16, sw, 7800, 7800, at);
    assert!(super::block_view_of(&key2d(F::R32G32B32A32_UINT, 16, sw, 1950, 1950, at), &y));
    assert!(!super::block_view_of(&key2d(F::R32G32B32A32_UINT, 16, sw, 1950, 1951, at), &y), "other grid");
    assert!(!super::block_view_of(&key2d(F::R32G32B32A32_UINT, 16, sw, 1950, 1950, at + 0x1_0000), &y), "other memory");
    assert!(!super::block_view_of(&key2d(F::R32G32B32A32_UINT, 16, tiling::SWIZZLE_64KB_R_X, 1950, 1950, at), &y), "other swizzle");
    assert!(!super::block_view_of(&key2d(F::R32G32_UINT, 8, sw, 1950, 1950, at), &key2d(F::BC1_RGBA_UNORM_BLOCK, 8, sw, 7801, 7800, at)), "a partial block column");
}

/// Benchmark (`RPS5E_RETILE_BENCH=1`): GPU time of refilling a 2848x1600
/// RGBA16 target from a newer B10G11R11 image over the same memory (the
/// UE5 transient-target shape), per refill and per guest byte.
#[test]
#[ignore]
fn retile_bench() {
    if std::env::var("RPS5E_RETILE_BENCH").is_err() {
        return;
    }
    let mem = test_memory();
    let sw = tiling::SWIZZLE_64KB_R_X;
    let (view, _) = guest_mapping(mem, 64 << 20);
    let addr = view + (1 << 20);
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));
    let a = images.find_or_create(&mut sched, &mut buffers, &key2d(vk::Format::R16G16B16A16_SFLOAT, 8, sw, 2848, 1600, addr)).unwrap();
    let b = images.find_or_create(&mut sched, &mut buffers, &key2d(vk::Format::B10G11R11_UFLOAT_PACK32, 4, sw, 2848, 1600, addr)).unwrap();
    let bytes = images.get(a).unwrap().key.size();
    assert!(images.prepare(&mut sched, &mut buffers, a, true));
    assert!(images.prepare(&mut sched, &mut buffers, b, true));
    sched.flush_and_wait();
    let n = 40;
    let t = std::time::Instant::now();
    for i in 0..n {
        let (dst, src) = if i % 2 == 0 { (a, b) } else { (b, a) };
        images.clear_color(&mut sched, src, vk::ClearColorValue { float32: [i as f32, 0.0, 0.0, 1.0] }, None);
        assert!(images.prepare(&mut sched, &mut buffers, dst, false));
        sched.flush();
    }
    sched.flush_and_wait();
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    let s = images.stats();
    eprintln!("retile bench: {n} refills of {:.1} MiB: {ms:.1} ms wall, {:.2} ms each, {:.1} GB/s of target bytes (retiles {}, direct {})", bytes as f64 / 1048576.0, ms / n as f64, bytes as f64 * n as f64 / ms / 1e6, s.retiles, s.direct_copies);
    dev.wait_idle();
    drop(images);
    drop(buffers);
}

fn key2d_levels(levels: u32, addr: u64) -> ImageKey {
    let sw = tiling::SWIZZLE_64KB_R_X;
    let mut key = key2d(vk::Format::R8G8B8A8_UNORM, 4, sw, 256, 256, addr);
    key.levels = levels;
    key.main = plane(&key, addr, 4, sw, vk::ImageAspectFlags::COLOR);
    key
}

/// GFX10 stores the levels of a layer from the smallest to the largest:
/// level 0 of a one-level surface is not where level 0 of an eight-level
/// surface at the same address lies.
#[test]
fn shorter_mip_chains_place_level_zero_elsewhere() {
    let long = key2d_levels(8, 0x10_0000_0000);
    let short = key2d_levels(1, 0x10_0000_0000);
    let l = long.layout(&long.main).unwrap();
    let s = short.layout(&short.main).unwrap();
    assert_ne!(l.mips[0].offset, s.mips[0].offset);
    assert!(!same_level_placement(&long, &short, 1), "a one-level texture does not read the long chain's level 0");
    assert!(same_level_placement(&long, &long, 8));
}

#[test]
#[ignore]
fn alias_lookup_skips_images_with_another_mip_chain() {
    let mem = test_memory();
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));
    let (view, _) = guest_mapping(mem, 8 << 20);
    let addr = view + (1 << 20);
    let long = key2d_levels(8, addr);
    let id = images.find_or_create(&mut sched, &mut buffers, &long).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, id, true));
    assert_eq!(images.find_alias(&long, 3), Some(id), "the same chain is found");
    assert_eq!(images.find_alias(&key2d_levels(1, addr), 1), None, "a one-level texture at the address is another surface");
}

/// Small colour targets the GPU wrote reach guest memory for the CPU: a
/// linear one at the next collection, a tiny tiled one on the CPU access
/// to its pages.
#[test]
#[ignore]
fn small_gpu_written_targets_reach_guest_memory() {
    let mem = test_memory();
    let dev = GpuDevice::create(false).expect("device");
    let mut sched = Scheduler::new(&dev).unwrap();
    let tracker = Arc::new(GuestPageTracker::new(mem.pages.clone()));
    let mut buffers = BufferCache::new(&dev, mem, tracker.clone()).unwrap();
    let mut images = ImageCache::new(&dev, mem).unwrap();
    mem.attach(Arc::new(NoBuffers), Arc::new(Images(images.table.clone())), Arc::new(NoUnmap));
    images.readback_linear = LinearReadback::Eager;
    let (view, _) = guest_mapping(mem, 8 << 20);

    // A 128x128 linear RGBA8 target, cleared on the GPU.
    let lin = view + (1 << 20);
    assert!(mem.write_backing(lin, &vec![0x55u8; 128 * 128 * 4]));
    let lkey = key2d(vk::Format::R8G8B8A8_UNORM, 4, tiling::SWIZZLE_LINEAR, 128, 128, lin);
    let l = images.find_or_create(&mut sched, &mut buffers, &lkey).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, l, true));
    images.clear_color(&mut sched, l, vk::ClearColorValue { float32: [1.0, 0.0, 1.0, 1.0] }, None);
    images.mark_gpu_written(l, 1);
    images.process_downloads(&mut sched, &mut buffers);
    buffers.download(&mut sched, lin, 128 * 128 * 4);
    let mut out = vec![0u8; 16];
    assert!(mem.read_backing(lin + 128 * 4 * 77, &mut out));
    assert_eq!(&out[..4], &[255, 0, 255, 255], "linear target written back");

    // A 1x1 tiled RG32F target: read-watched until the CPU touches it.
    let tiny = view + (4 << 20);
    assert!(mem.write_backing(tiny, &[0x55u8; 8]));
    let tkey = key2d(vk::Format::R32G32_SFLOAT, 8, tiling::SWIZZLE_64KB_R_X, 1, 1, tiny);
    let t = images.find_or_create(&mut sched, &mut buffers, &tkey).unwrap();
    assert!(images.prepare(&mut sched, &mut buffers, t, true));
    images.clear_color(&mut sched, t, vk::ClearColorValue { float32: [2.0, -3.0, 0.0, 0.0] }, None);
    images.mark_gpu_written(t, 2);
    let hits = images.table.take_read_backs(tiny).expect("the tiny target is read-watched");
    images.resolve_read_backs(&mut sched, &mut buffers, &tracker, &hits, false);
    let mut out = [0u8; 8];
    assert!(mem.read_backing(tiny, &mut out));
    assert_eq!(out, [2.0f32.to_le_bytes(), (-3.0f32).to_le_bytes()].concat().as_slice(), "tiled target written back");
}
