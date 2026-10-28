// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only
//
// Unit tests of the `ps5gpu::buffers::guest` module.

use super::*;
use crate::scheduler::Scheduler;
use crate::testutil;
use crate::tiler::{Spv, ARGUMENT_COUNT};
use rspirv::spirv;

/// Entries `[first, first+count)` of the import table.
fn read_table(dev: &Arc<GpuDevice>, s: &mut Scheduler, g: &GuestImports, first: u64, count: u64) -> Vec<u64> {
    let out = Buffer::new(dev, count * 8, vk::BufferUsageFlags::TRANSFER_DST, MemoryUse::Download).unwrap();
    s.full_barrier();
    let cb = s.cmd();
    // SAFETY: recording buffer.
    unsafe { dev.device.cmd_copy_buffer(cb, g.table.buffer, out.buffer, &[vk::BufferCopy { src_offset: first * 8, dst_offset: 0, size: count * 8 }]) };
    s.flush_and_wait();
    out.invalidate(0, count * 8);
    out.bytes_mut()[..(count * 8) as usize].chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()
}

/// The table maps a guest mapping's pages into the import of its backing,
/// follows unmaps and remaps, and a write-through lands in guest memory.
#[test]
#[ignore]
fn import_table_follows_guest_mappings() {
    let dev = crate::device::GpuDevice::create(false).expect("device");
    if !dev.caps.external_memory_host {
        eprintln!("VK_EXT_external_memory_host not supported; skipped");
        return;
    }
    let mem = testutil::test_memory();
    let len = 8u64 << 20;
    let (view, _alias) = testutil::guest_mapping(mem, len as usize);
    let mut s = Scheduler::new(&dev).unwrap();
    let mut g = GuestImports::new(&dev, mem, &mut s).unwrap().expect("imports");
    assert!(g.sync(&mut s).is_empty());
    let first = view >> CACHING_PAGE_BITS;
    let pages = len >> CACHING_PAGE_BITS;
    let entries = read_table(&dev, &mut s, &g, first, pages);
    let (_, base, _) = g.chunk(0).unwrap();
    for (i, e) in entries.iter().enumerate() {
        assert_eq!(*e, base + ((i as u64) << CACHING_PAGE_BITS), "page {i}");
    }
    assert!(g.is_backed(view, len));

    // An unmapped hole reads zero entries; a remap restores them.
    testutil::unmap_guest(mem, view + CACHING_PAGE * 4, CACHING_PAGE * 2);
    let cleared = g.sync(&mut s);
    assert_eq!(cleared, vec![(view + CACHING_PAGE * 4, CACHING_PAGE * 2)]);
    let entries = read_table(&dev, &mut s, &g, first, 8);
    assert_eq!(&entries[4..6], &[0, 0]);
    assert_ne!(entries[6], 0);
    assert!(!g.is_backed(view, len));
    testutil::map_guest(mem, view + CACHING_PAGE * 4, CACHING_PAGE * 2);
    g.sync(&mut s);
    let entries = read_table(&dev, &mut s, &g, first, 8);
    assert_eq!(entries[4], base + CACHING_PAGE * 4);
    assert!(g.is_backed(view, len));

    // A device-local mirror written by the GPU reaches guest memory.
    let mirror = Buffer::new(&dev, CACHING_PAGE * 2, BUFFER_USAGE, MemoryUse::Device).unwrap();
    let cb = s.cmd();
    // SAFETY: recording buffer.
    unsafe { dev.device.cmd_fill_buffer(cb, mirror.buffer, 0, vk::WHOLE_SIZE, 0x5EED_F00D) };
    s.full_barrier();
    let at = view + CACHING_PAGE * 3;
    assert!(g.write_through(&mut s, mirror.buffer, at, at + 64, at + CACHING_PAGE));
    s.flush_and_wait();
    // SAFETY: the guest view is mapped read/write.
    let words = unsafe { std::slice::from_raw_parts((at) as *const u32, 4096) };
    assert_eq!(words[15], 0, "bytes before the range stay");
    assert!(words[16..].iter().all(|&w| w == 0x5EED_F00D));
    assert_eq!(g.stats.chunks, 1);
}

fn pipeline(dev: &Arc<GpuDevice>, code: &[u32]) -> (vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline) {
    let d = &dev.device;
    let bindings = [0u32, 1].map(|i| {
        vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)
    });
    // SAFETY: valid create infos; destroyed by the caller.
    unsafe {
        let set = d
            .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR).bindings(&bindings), None)
            .unwrap();
        let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: (ARGUMENT_COUNT * 4) as u32 }];
        let layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set)).push_constant_ranges(&ranges), None).unwrap();
        let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None).unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let p = d.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None).unwrap()[0];
        d.destroy_shader_module(module, None);
        (set, layout, p)
    }
}

const READS: u32 = 64;

/// Each invocation sums `READS` words `threads` apart (arg 0: threads,
/// arg 1: word mask) and stores the sum.
fn stream_kernel() -> Vec<u32> {
    let mut s = Spv::new();
    let (main, x, _, _) = s.begin_main().unwrap();
    let threads = s.arg(0).unwrap();
    let mask = s.arg(1).unwrap();
    let mut sum = s.c(0);
    let mut idx = x;
    for _ in 0..READS {
        let i = s.b.bitwise_and(s.uint, None, idx, mask).unwrap();
        let v = s.load_word(s.src, i).unwrap();
        sum = s.add(sum, v).unwrap();
        idx = s.add(idx, threads).unwrap();
    }
    let p = s.word_ptr(s.dst, x).unwrap();
    s.b.store(p, sum, None, []).unwrap();
    s.finish(main, 256, 1).unwrap()
}

/// One invocation follows `READS` dependent words (`next = src[next]`).
fn chase_kernel() -> Vec<u32> {
    let mut s = Spv::new();
    let (main, x, _, _) = s.begin_main().unwrap();
    let mut idx = x;
    for _ in 0..READS {
        idx = s.load_word(s.src, idx).unwrap();
    }
    let zero = s.c(0);
    let p = s.word_ptr(s.dst, zero).unwrap();
    s.b.store(p, idx, None, []).unwrap();
    let _ = spirv::Scope::Device;
    s.finish(main, 1, 1).unwrap()
}

/// Best of `runs` wall times of one dispatch over `src`.
fn time_dispatch(dev: &Arc<GpuDevice>, s: &mut Scheduler, code: &[u32], src: vk::Buffer, dst: vk::Buffer, groups: u32, args: [u32; 2], runs: usize) -> f64 {
    let (set, layout, p) = pipeline(dev, code);
    let mut best = f64::MAX;
    for _ in 0..runs {
        s.flush_and_wait();
        let cb = s.cmd();
        let infos = [vk::DescriptorBufferInfo { buffer: src, offset: 0, range: vk::WHOLE_SIZE }, vk::DescriptorBufferInfo { buffer: dst, offset: 0, range: vk::WHOLE_SIZE }];
        let writes = [0u32, 1].map(|i| vk::WriteDescriptorSet::default().dst_binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&infos[i as usize])));
        let mut words = [0u32; ARGUMENT_COUNT];
        words[..2].copy_from_slice(&args);
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let t0 = std::time::Instant::now();
        // SAFETY: recording buffer; the pipeline objects outlive the wait.
        unsafe {
            dev.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, p);
            dev.push_descriptor.cmd_push_descriptor_set(cb, vk::PipelineBindPoint::COMPUTE, layout, 0, &writes);
            dev.device.cmd_push_constants(cb, layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
            dev.device.cmd_dispatch(cb, groups, 1, 1);
        }
        s.flush_and_wait();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    // SAFETY: idle after the waits.
    unsafe {
        dev.device.destroy_pipeline(p, None);
        dev.device.destroy_pipeline_layout(layout, None);
        dev.device.destroy_descriptor_set_layout(set, None);
    }
    best
}

/// Device limits for importing guest memory and what reading it costs:
/// import time per chunk, shader read bandwidth and dependent-read latency
/// of imported host memory against device-local memory.
#[test]
#[ignore]
fn import_limits_and_read_costs() {
    let dev = crate::device::GpuDevice::create(false).expect("device");
    let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
    let mut p13 = vk::PhysicalDeviceVulkan13Properties::default();
    let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut p11).push_next(&mut p13);
    // SAFETY: valid chain on the stack.
    unsafe { dev.instance.get_physical_device_properties2(dev.phys, &mut p2) };
    eprintln!(
        "device '{}': maxMemoryAllocationSize {} MiB, maxBufferSize {} MiB, maxStorageBufferRange {} MiB, minImportedHostPointerAlignment {:#x}, external_memory_host {}",
        dev.name,
        p11.max_memory_allocation_size >> 20,
        p13.max_buffer_size >> 20,
        dev.caps.max_storage_buffer_range >> 20,
        dev.caps.min_imported_host_pointer_alignment,
        dev.caps.external_memory_host
    );
    for (i, t) in dev.mem_props.memory_types[..dev.mem_props.memory_type_count as usize].iter().enumerate() {
        eprintln!("  memory type {i}: heap {} flags {:?}", t.heap_index, t.property_flags);
    }
    if !dev.caps.external_memory_host {
        return;
    }
    // A 4 GiB memfd imported in 256 MiB chunks and as one allocation.
    let len: u64 = 4 << 30;
    let rss = || std::fs::read_to_string("/proc/self/status").unwrap().lines().find(|l| l.starts_with("VmRSS")).unwrap().to_string();
    // SAFETY: plain memfd + shared mapping owned by the test.
    let alias = unsafe {
        let fd = libc::memfd_create(c"import-bench".as_ptr(), 0);
        assert_eq!(libc::ftruncate(fd, len as libc::off_t), 0);
        libc::mmap(std::ptr::null_mut(), len as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0) as *mut u8
    };
    eprintln!("before imports: {}", rss());
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    let mut chunks = Vec::new();
    let t0 = std::time::Instant::now();
    for i in 0..len / CHUNK {
        // SAFETY: the mapping lives for the test.
        let t = std::time::Instant::now();
        let b = unsafe { Buffer::import_host(&dev, alias.add((i * CHUNK) as usize), CHUNK, usage) }.expect("chunk import");
        if i == 0 {
            eprintln!("first {} MiB chunk import: {:?} (coherent {})", CHUNK >> 20, t.elapsed(), b.coherent);
        }
        chunks.push(b);
    }
    eprintln!("{} chunks ({} GiB) imported in {:?}; {}", chunks.len(), len >> 30, t0.elapsed(), rss());
    let t = std::time::Instant::now();
    // SAFETY: as above.
    match unsafe { Buffer::import_host(&dev, alias, len, usage) } {
        Ok(_) => eprintln!("one {} GiB import: {:?} ok", len >> 30, t.elapsed()),
        Err(e) => eprintln!("one {} GiB import: {e:?} after {:?}", len >> 30, t.elapsed()),
    }

    let mut s = Scheduler::new(&dev).unwrap();
    // Read bandwidth: 256 MiB read per dispatch out of a 256 MiB window.
    let words: u64 = CHUNK / 4;
    let threads: u32 = 1 << 20;
    let out = Buffer::new(&dev, threads as u64 * 4, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryUse::Device).unwrap();
    let device = Buffer::new(&dev, CHUNK, usage, MemoryUse::Device).unwrap();
    let stream = stream_kernel();
    let bytes = threads as f64 * READS as f64 * 4.0;
    for (name, src) in [("device-local", device.buffer), ("imported host", chunks[0].buffer)] {
        let t = time_dispatch(&dev, &mut s, &stream, src, out.buffer, threads / 256, [threads, (words - 1) as u32], 5);
        eprintln!("stream read {name}: {:.1} GB/s ({:.2} ms for {} MiB)", bytes / t / 1e9, t * 1e3, bytes as u64 >> 20);
    }
    // Dependent reads 64 KiB + 4 bytes apart (no reuse in any cache line).
    let step: u64 = (64 << 10) / 4 + 1;
    let n = words / step;
    let chain: Vec<u32> = (0..words).map(|i| if i % step == 0 { (((i / step + 1) % n) * step) as u32 } else { 0 }).collect();
    // SAFETY: the alias maps the first chunk.
    unsafe { std::ptr::copy_nonoverlapping(chain.as_ptr() as *const u8, alias, (words * 4) as usize) };
    let staging = Buffer::new(&dev, CHUNK, vk::BufferUsageFlags::TRANSFER_SRC, MemoryUse::Upload).unwrap();
    staging.bytes_mut().copy_from_slice(&unsafe { std::slice::from_raw_parts(alias, CHUNK as usize) }[..]);
    staging.flush(0, CHUNK);
    let cb = s.cmd();
    // SAFETY: recording buffer.
    unsafe { dev.device.cmd_copy_buffer(cb, staging.buffer, device.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: CHUNK }]) };
    s.flush_and_wait();
    let chase = chase_kernel();
    let empty = {
        let mut sp = Spv::new();
        let (main, _, _, _) = sp.begin_main().unwrap();
        sp.finish(main, 1, 1).unwrap()
    };
    let base = time_dispatch(&dev, &mut s, &empty, device.buffer, out.buffer, 1, [0, 0], 9);
    for (name, src) in [("device-local", device.buffer), ("imported host", chunks[0].buffer)] {
        let t = time_dispatch(&dev, &mut s, &chase, src, out.buffer, 1, [0, 0], 9);
        eprintln!("dependent read {name}: {:.2} us each ({:.1} us dispatch, {:.1} us empty)", (t - base) * 1e6 / READS as f64, t * 1e6, base * 1e6);
    }
    // DMA copy bandwidth host -> device.
    let t = std::time::Instant::now();
    let cb = s.cmd();
    // SAFETY: recording buffer.
    unsafe { dev.device.cmd_copy_buffer(cb, chunks[1].buffer, device.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: CHUNK }]) };
    s.flush_and_wait();
    eprintln!("copy imported -> device: {:.1} GB/s", CHUNK as f64 / t.elapsed().as_secs_f64() / 1e9);
    drop(chunks);
    eprintln!("after release: {}", rss());
}

/// A private read/write guest mapping (no memfd backing) is imported where
/// it lives; its unmap drops the import and the entries.
#[test]
#[ignore]
fn private_mappings_are_imported_in_place() {
    let dev = crate::device::GpuDevice::create(false).expect("device");
    if !dev.caps.external_memory_host {
        return;
    }
    let mem = testutil::test_memory();
    let (_view, _) = testutil::guest_mapping(mem, 1 << 20);
    let len: u64 = 1 << 20;
    let addr = testutil::private_mapping(len as usize);
    ps5vm::ranges::track(addr, len, ps5vm::MemProtect::RW);
    mem.note_mapped(addr, len, libc::PROT_READ | libc::PROT_WRITE, None);
    let mut s = Scheduler::new(&dev).unwrap();
    let mut g = GuestImports::new(&dev, mem, &mut s).unwrap().expect("imports");
    g.sync(&mut s);
    assert!(g.is_backed(addr, len), "the private mapping has entries");
    assert_eq!(g.stats.private_imports, 1);
    let first = addr >> CACHING_PAGE_BITS;
    let e = read_table(&dev, &mut s, &g, first, 2);
    assert_ne!(e[0], 0);
    assert_eq!(e[1], e[0] + CACHING_PAGE);
    mem.note_unmapped(addr, len);
    let cleared = g.sync(&mut s);
    assert!(cleared.contains(&(addr, len)));
    assert!(!g.is_backed(addr, CACHING_PAGE));
    assert_eq!(read_table(&dev, &mut s, &g, first, 1)[0], 0);
    s.flush_and_wait();
}

/// A range inside one mapping is a transfer source in its import (the
/// copy reads the guest bytes at their offset); a range over an unmapped
/// hole is not.
#[test]
#[ignore]
fn source_slice_reads_guest_bytes() {
    let dev = crate::device::GpuDevice::create(false).expect("device");
    if !dev.caps.external_memory_host {
        eprintln!("VK_EXT_external_memory_host not supported; skipped");
        return;
    }
    let mem = testutil::test_memory();
    let len = 4u64 << 20;
    let (view, _alias) = testutil::guest_mapping(mem, len as usize);
    let mut s = Scheduler::new(&dev).unwrap();
    let mut g = GuestImports::new(&dev, mem, &mut s).unwrap().expect("imports");
    g.sync(&mut s);
    let at = view + CACHING_PAGE * 2 + 0x300;
    // SAFETY: the guest view is mapped read/write.
    let words = unsafe { std::slice::from_raw_parts_mut(at as *mut u32, 1024) };
    for (i, w) in words.iter_mut().enumerate() {
        *w = 0xA000_0000 | i as u32;
    }
    let (src, off) = g.source_slice(at, 4096).expect("one import covers the range");
    let out = Buffer::new(&dev, 4096, vk::BufferUsageFlags::TRANSFER_DST, MemoryUse::Download).unwrap();
    let cb = s.cmd();
    // SAFETY: recording buffer; the import lives for the process.
    unsafe { dev.device.cmd_copy_buffer(cb, src, out.buffer, &[vk::BufferCopy { src_offset: off, dst_offset: 0, size: 4096 }]) };
    s.flush_and_wait();
    out.invalidate(0, 4096);
    let got: Vec<u32> = out.bytes_mut()[..4096].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    assert!(got.iter().enumerate().all(|(i, &w)| w == 0xA000_0000 | i as u32));
    testutil::unmap_guest(mem, view + CACHING_PAGE * 8, CACHING_PAGE);
    g.sync(&mut s);
    assert!(g.source_slice(view + CACHING_PAGE * 7, CACHING_PAGE * 3).is_none(), "a hole in the range");
    s.flush_and_wait();
}
