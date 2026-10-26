// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Dispatches with thread dimensions (DISPATCH_INITIATOR bit 5): the
//! dimensions count threads; the last group is partial and its lanes past
//! the count do not run. A ray-tracing BVH refit kernel without a bounds
//! check read past its leaf array (and looped forever) when the count was
//! taken as a group count.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored thread_dims`.

use super::*;
use ps5shader::testing::buffer_descriptor;

const POISON: u32 = 0xDEAD_BEEF;

#[test]
fn thread_dims_group_counts_and_limits() {
    let d = |groups: [u32; 3], initiator: u32, indirect: u64| ResolvedDispatch {
        packet: DispatchPacket { groups, dispatch_initiator: initiator, indirect_address: indirect, ..DispatchPacket::default() },
        shader: unbound(),
        rsrc3: 0,
        threads: [64, 2, 1],
        start: [0; 3],
    };
    assert_eq!(d([100, 4, 1], 1, 0).group_counts(), [100, 4, 1]);
    assert_eq!(d([100, 4, 1], 1, 0).thread_limits(), None);
    assert_eq!(d([100, 4, 1], 0x21, 0).group_counts(), [2, 2, 1]);
    assert_eq!(d([100, 4, 1], 0x21, 0).thread_limits(), Some([100, 4, 1]));
    assert_eq!(d([128, 4, 1], 0x21, 0).thread_limits(), None);
    assert_eq!(d([128, 3, 1], 0x21, 0).thread_limits(), Some([128, 3, 1]));
}

/// 100 threads in groups of 64: two groups launch, 100 lanes store.
#[test]
#[ignore]
fn thread_dims_partial_group_stops_at_the_count() {
    let mut g = Guest::new(16 << 20);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    // v1 = s4 * 64 + v0; v2 = 1; buffer_store_dword v2, v1 idxen.
    let cs = g.program(&[0xD746_0001, 0x0401_0C04, 0x7E04_0281, 0xE070_2000, 0x8000_0201, asm::S_ENDPGM]);
    for (wave32, threads) in [(false, 100u32), (true, 100), (false, 128), (true, 37)] {
        let words = 192usize;
        let ob = g.alloc(&asm::bytes(&vec![POISON; words]));
        let user = buffer_descriptor(ob, 4, words as u32, 0x0101_6FAC);
        let mut shader = stage(cs, &user);
        shader.rsrc2 |= 1 << 7; // TGID_X_EN: s4 = group id
        let d = ResolvedDispatch {
            packet: DispatchPacket { groups: [threads, 1, 1], dispatch_initiator: 0x21 | if wave32 { 1 << 15 } else { 0 }, ..DispatchPacket::default() },
            shader,
            rsrc3: 0,
            threads: [64, 1, 1],
            start: [0; 3],
        };
        assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
        assert!(rig.buffers.download(&mut rig.sched, ob, words as u64 * 4));
        let mut out = vec![0u8; words * 4];
        assert!(g.mem.read_backing(ob, &mut out));
        let w: Vec<u32> = out.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let n = threads as usize;
        assert!(w[..n].iter().all(|&x| x == 1), "wave32 {wave32} threads {threads}: {:x?}", &w[..n]);
        assert!(w[n..].iter().all(|&x| x == POISON), "wave32 {wave32} threads {threads}: lanes past the count ran: {:x?}", &w[n..]);
    }
}

/// Output region of the indirect thread-dimension tests: 32 x 16 x 8
/// words, the program stores word index x + 32 y + 512 z at its global
/// invocation id.
const REGION: [u32; 3] = [32, 16, 8];
const REGION_WORDS: usize = (REGION[0] * REGION[1] * REGION[2]) as usize;
/// Threads per group of the tests' dispatches.
const GROUP: [u32; 3] = [8, 4, 2];

/// gx = s4 * 8 + v0, gy = s5 * 4 + v1, gz = s6 * 2 + v2;
/// i = gx + 32 gy + 512 gz; buffer_store_dword i, i (idxen, s[0:3]).
fn region_store_program(g: &mut Guest) -> u64 {
    use ps5shader::testing::asm as a;
    const V_LSHL_ADD_U32: u32 = 0x346;
    let (v, k) = (a::vgpr, a::inline_u32);
    let mut code = Vec::new();
    code.extend(a::vop3(V_LSHL_ADD_U32, 3, 4, k(3), v(0)));
    code.extend(a::vop3(V_LSHL_ADD_U32, 4, 5, k(2), v(1)));
    code.extend(a::vop3(V_LSHL_ADD_U32, 5, 6, k(1), v(2)));
    code.extend(a::vop3(V_LSHL_ADD_U32, 6, v(4), k(5), v(3)));
    code.extend(a::vop3(V_LSHL_ADD_U32, 6, v(5), k(9), v(6)));
    code.extend(a::mubuf(a::OP_BUFFER_STORE_DWORD, 6, 6, 0, k(0), 0, true, false, false));
    code.push(asm::S_ENDPGM);
    g.program(&code)
}

/// v1 = src[v0] (s[0:3]); dst[v0] = v1 (s[4:7]).
fn copy_program(g: &mut Guest) -> u64 {
    use ps5shader::testing::asm as a;
    let mut code = Vec::new();
    code.extend(a::mubuf(a::OP_BUFFER_LOAD_DWORD, 1, 0, 0, a::inline_u32(0), 0, true, false, false));
    code.push(a::sopp(0x0C, 0x3F70)); // s_waitcnt vmcnt(0)
    code.extend(a::mubuf(a::OP_BUFFER_STORE_DWORD, 1, 0, 1, a::inline_u32(0), 0, true, false, false));
    code.push(asm::S_ENDPGM);
    g.program(&code)
}

/// The words the region program leaves for `threads` threads per axis.
fn region_expected(threads: [u32; 3]) -> Vec<u32> {
    (0..REGION_WORDS as u32)
        .map(|i| {
            let (x, y, z) = (i % REGION[0], i / REGION[0] % REGION[1], i / (REGION[0] * REGION[1]));
            if x < threads[0] && y < threads[1] && z < threads[2] {
                i
            } else {
                POISON
            }
        })
        .collect()
}

fn words_at(b: &mut Backend, mem: &crate::memory::GuestGpuMemory, a: u64, n: usize) -> Vec<u32> {
    assert!(b.buffers.download(&mut b.sched, a, n as u64 * 4));
    let mut out = vec![0u8; n * 4];
    assert!(mem.read_backing(a, &mut out));
    out.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// An indirect dispatch with thread dimensions whose record an earlier
/// dispatch of the same submission writes (guest memory still holds stale
/// counts) runs from the record on the GPU: the same stores as the direct
/// dispatch of those counts (partial groups on each axis, an axis without
/// threads, exact multiples), with no read-back and no submit in between.
#[test]
#[ignore]
fn thread_dims_indirect_record_written_by_the_gpu() {
    let mut g = Guest::new(32 << 20);
    let dev = GpuDevice::create(false).expect("device");
    let mut b = Backend::new(&dev, g.mem).unwrap();
    let compiler: Arc<dyn ShaderCompiler> = Arc::new(crate::shader::SpirvCompiler::for_device(&dev, g.mem));
    let mut p = ShaderPipelines::new(&dev, g.mem, Some(compiler), None).unwrap();
    p.set_async_compile(false);
    b.pipelines = Box::new(p);
    let cs = region_store_program(&mut g);
    let copy = copy_program(&mut g);
    // TGID_X/Y/Z_EN (s4..s6 after four user SGPRs), TIDIG_COMP_CNT = 2.
    let system = (1 << 7) | (1 << 8) | (1 << 9) | (2 << 11);
    let cases: [[u32; 3]; 7] = [[13, 5, 3], [16, 8, 4], [0, 8, 4], [16, 0, 4], [32, 16, 8], [1, 1, 1], [27, 13, 7]];
    let mut gpu_runs = 0;
    for wave32 in [false, true] {
        let wave = if wave32 { 1 << 15 } else { 0 };
        for threads in cases {
            // The record 20 bytes into its chunk, holding stale counts that
            // would fill the whole region.
            let mut stale = vec![0x5A5A_5A5Au32; 16];
            stale[5..8].copy_from_slice(&REGION);
            let chunk = g.alloc(&asm::bytes(&stale));
            let record = chunk + 20;
            let src = g.alloc(&asm::bytes(&threads));
            let out_indirect = g.alloc(&asm::bytes(&vec![POISON; REGION_WORDS]));
            let out_direct = g.alloc(&asm::bytes(&vec![POISON; REGION_WORDS]));
            // The record, written on the GPU.
            let mut user = buffer_descriptor(src, 4, 3, 0x0101_6FAC).to_vec();
            user.extend(buffer_descriptor(record, 4, 3, 0x0101_6FAC));
            let writer = ResolvedDispatch {
                packet: DispatchPacket { groups: [1, 1, 1], dispatch_initiator: 1, ..DispatchPacket::default() },
                shader: stage(copy, &user),
                rsrc3: 0,
                threads: [3, 1, 1],
                start: [0; 3],
            };
            let dispatch = |out: u64, indirect: bool| {
                let mut shader = stage(cs, &buffer_descriptor(out, 4, REGION_WORDS as u32, 0x0101_6FAC));
                shader.rsrc2 |= system;
                let packet = if indirect {
                    DispatchPacket { indirect_address: record, dispatch_initiator: 0x21 | wave, ..DispatchPacket::default() }
                } else {
                    DispatchPacket { groups: threads, dispatch_initiator: 0x21 | wave, ..DispatchPacket::default() }
                };
                ResolvedDispatch { packet, shader, rsrc3: 0, threads: GROUP, start: [0; 3] }
            };
            // Submits other than the early ones of a long tick
            // (`RPS5E_EARLY_SUBMIT`) would be waits for a read-back.
            let submits_of = |b: &Backend| b.sched.stats().submits - b.stats.early_submits;
            let (downloads, through, submits) = (b.buffers.stats().downloads, b.buffers.stats().written_through_waits, submits_of(&b));
            b.dispatch(&writer);
            b.dispatch(&dispatch(out_indirect, true));
            assert_eq!(
                (b.buffers.stats().downloads, b.buffers.stats().written_through_waits, submits_of(&b)),
                (downloads, through, submits),
                "wave32 {wave32} {threads:?}: the record was read back"
            );
            gpu_runs += 1;
            b.dispatch(&dispatch(out_direct, false));
            let indirect = words_at(&mut b, g.mem, out_indirect, REGION_WORDS);
            let direct = words_at(&mut b, g.mem, out_direct, REGION_WORDS);
            let expected = region_expected(threads);
            let wrong = |w: &[u32]| (0..REGION_WORDS).filter(|&i| w[i] != expected[i]).take(8).map(|i| (i, w[i], expected[i])).collect::<Vec<_>>();
            assert!(direct == expected, "wave32 {wave32} {threads:?}: direct dispatch (index, got, expected): {:x?}", wrong(&direct));
            assert!(indirect == direct, "wave32 {wave32} {threads:?}: indirect dispatch (index, got, expected): {:x?}", wrong(&indirect));
            assert_eq!(words_at(&mut b, g.mem, record, 3), threads.to_vec(), "the record as the GPU wrote it");
        }
    }
    let summary = b.pipelines.summary().unwrap_or_default();
    assert!(summary.contains(&format!("cs-thread-dims={gpu_runs}gpu")), "every indirect dispatch launched on the GPU: {summary}");
    drop(b);
    dev.wait_idle();
}

/// Records the dispatches it receives; reports whether it runs indirect
/// thread dimensions on the GPU as configured.
struct SeenDispatches {
    gpu: bool,
    seen: Arc<std::sync::Mutex<Vec<ResolvedDispatch>>>,
}

impl PipelineProvider for SeenDispatches {
    fn draw(&mut self, _ctx: &mut DrawContext, _d: &ResolvedDraw) -> DrawOutcome {
        DrawOutcome::Skipped
    }

    fn dispatch(&mut self, _ctx: &mut DrawContext, d: &ResolvedDispatch) -> DrawOutcome {
        self.seen.lock().unwrap().push(*d);
        DrawOutcome::Recorded
    }

    fn indirect_thread_dimensions(&self, _d: &ResolvedDispatch) -> bool {
        self.gpu
    }
}

/// The back-end passes an indirect dispatch with thread dimensions on as
/// it is when the provider resolves the record on the GPU, and otherwise
/// reads the record (the `RPS5E_GPU_INDIRECT_THREADS=0` path) and passes
/// the direct dispatch of its counts.
#[test]
#[ignore]
fn thread_dims_indirect_backend_path_selection() {
    let mut g = Guest::new(16 << 20);
    let record = g.alloc(&asm::bytes(&[100, 3, 2]));
    let dev = GpuDevice::create(false).expect("device");
    let mut b = Backend::new(&dev, g.mem).unwrap();
    let packet = DispatchPacket { indirect_address: record, dispatch_initiator: 0x21, ..DispatchPacket::default() };
    let d = ResolvedDispatch { packet, shader: stage(0x1000, &[]), rsrc3: 0, threads: [64, 1, 1], start: [0; 3] };
    for gpu in [true, false] {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        b.pipelines = Box::new(SeenDispatches { gpu, seen: seen.clone() });
        b.dispatch(&d);
        let got = seen.lock().unwrap().clone();
        assert_eq!(got.len(), 1);
        if gpu {
            assert_eq!(got[0], d, "passed on unchanged");
        } else {
            assert_eq!(got[0].packet.indirect_address, 0);
            assert_eq!(got[0].packet.groups, [100, 3, 2]);
            assert_eq!(got[0].group_counts(), [2, 3, 2]);
            assert_eq!(got[0].thread_limits(), Some([100, 3, 2]));
        }
    }
    drop(b);
    dev.wait_idle();
}

/// An indirect dispatch whose CPU-written record holds a zero count is
/// skipped before its program is prepared; a non-empty one is recorded.
#[test]
#[ignore]
fn empty_cpu_written_indirect_dispatch_is_skipped() {
    let mut g = Guest::new(16 << 20);
    let dev = GpuDevice::create(false).expect("device");
    let mut b = Backend::new(&dev, g.mem).unwrap();
    let compiler: Arc<dyn ShaderCompiler> = Arc::new(crate::shader::SpirvCompiler::for_device(&dev, g.mem));
    let mut p = ShaderPipelines::new(&dev, g.mem, Some(compiler), None).unwrap();
    p.set_async_compile(false);
    b.pipelines = Box::new(p);
    let cs = g.program(&[asm::S_ENDPGM]);
    let empty = g.alloc(&asm::bytes(&[0, 1, 1]));
    let full = g.alloc(&asm::bytes(&[1, 1, 1]));
    for (record, skipped) in [(empty, true), (full, false)] {
        let packet = DispatchPacket { indirect_address: record, dispatch_initiator: 0x1, ..DispatchPacket::default() };
        b.dispatch(&ResolvedDispatch { packet, shader: stage(cs, &[]), rsrc3: 0, threads: [64, 1, 1], start: [0; 3] });
        let summary = b.pipelines.summary().unwrap_or_default();
        assert!(summary.contains(&format!("cs-empty-indirect={}", 1)), "{summary}");
        assert_eq!(summary.contains("cs=0 "), skipped, "no compute pipeline for the empty dispatch: {summary}");
    }
    drop(b);
    dev.wait_idle();
}

/// The group-record clamp (`RPS5E_INDIRECT_GROUP_CLAMP=1`) runs the record
/// kernel with one thread per group: counts within the device limits pass
/// unchanged, one count over them launches nothing.
#[test]
fn indirect_group_records_clamp_to_the_device_limits() {
    use crate::pipeline::indirect_threads::{launch, ThreadRules};
    let r = ThreadRules::new([1, 1, 1], [0x7FFF_FFFF, 65535, 65535]);
    assert_eq!(launch([5, 6, 7], &r), [5, 6, 7]);
    assert_eq!(launch([0x7FFF_FFFF, 65535, 65535], &r), [0x7FFF_FFFF, 65535, 65535]);
    assert_eq!(launch([5, 70000, 1], &r), [0, 0, 0]);
    assert_eq!(launch([0, 1, 1], &r), [0, 1, 1]);
}

/// A group record that is not dword-aligned is no valid indirect host
/// dispatch argument: the back-end reads it and passes the direct dispatch
/// of its counts; an aligned one is passed on as it is.
#[test]
#[ignore]
fn misaligned_indirect_group_record_is_read_on_the_cpu() {
    let mut g = Guest::new(16 << 20);
    let mut bytes = vec![0u8, 0];
    bytes.extend_from_slice(&asm::bytes(&[7, 3, 2]));
    let at = g.alloc(&bytes);
    let dev = GpuDevice::create(false).expect("device");
    let mut b = Backend::new(&dev, g.mem).unwrap();
    for (record, cpu) in [(at + 2, true), (at, false)] {
        let packet = DispatchPacket { indirect_address: record, dispatch_initiator: 0x1, ..DispatchPacket::default() };
        let d = ResolvedDispatch { packet, shader: stage(0x1000, &[]), rsrc3: 0, threads: [64, 1, 1], start: [0; 3] };
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        b.pipelines = Box::new(SeenDispatches { gpu: true, seen: seen.clone() });
        b.dispatch(&d);
        let got = seen.lock().unwrap().clone();
        assert_eq!(got.len(), 1);
        if cpu {
            assert_eq!((got[0].packet.indirect_address, got[0].group_counts()), (0, [7, 3, 2]));
        } else {
            assert_eq!(got[0], d, "passed on unchanged");
        }
    }
    drop(b);
    dev.wait_idle();
}

/// The layout of a program run by an indirect dispatch with thread
/// dimensions: thread limits in the shader-data buffer (the GPU writes
/// them), a key of its own; other dispatches keep push data.
#[test]
fn thread_dims_indirect_layout_selection() {
    use ps5shader::resources::{push_data, BindingLayout, DescriptorBindingKind, ShaderResourceInfo};
    let d = |groups: [u32; 3], initiator: u32, indirect: u64| ResolvedDispatch {
        packet: DispatchPacket { groups, dispatch_initiator: initiator, indirect_address: indirect, ..DispatchPacket::default() },
        shader: unbound(),
        rsrc3: 0,
        threads: [64, 2, 1],
        start: [0; 3],
    };
    use super::provider::dispatch_layout;
    assert_eq!(dispatch_layout(&d([0; 3], 0x21, 0x1000)), (push_data::DWORD_COUNT, Some([0; 3])), "indirect thread counts");
    assert_eq!(dispatch_layout(&d([0; 3], 1, 0x1000)), (0, None), "indirect group counts");
    assert_eq!(dispatch_layout(&d([100, 4, 1], 0x21, 0)), (0, Some([100, 4, 1])), "direct, partial");
    assert_eq!(dispatch_layout(&d([128, 4, 1], 0x21, 0)), (0, None), "direct, whole groups");
    // Allocated at those cursors: the indirect layout reads its limits
    // from the shader-data buffer, the direct one from push constants.
    let info = ShaderResourceInfo::default();
    let gpu = BindingLayout::allocate(&info, vec![0, 1], false, false, false, push_data::DWORD_COUNT, true, false).unwrap();
    let cpu = BindingLayout::allocate(&info, vec![0, 1], false, false, false, 0, true, false).unwrap();
    assert!(!gpu.uses_push_data() && gpu.find(DescriptorBindingKind::ShaderData).is_some());
    assert!(cpu.uses_push_data() && cpu.find(DescriptorBindingKind::ShaderData).is_none());
    assert_eq!((gpu.dispatch_thread_limits_dword(), gpu.shader_data_dword_count()), (2, 5));
    assert!(gpu != cpu, "the layouts (and so the module keys) differ");
}

/// A direct dispatch of more groups than the device launches at once (its
/// Y limit plus 3) runs every group once, split into chunks at their base
/// groups (`RPS5E_DISPATCH_SPLIT=0` skips it).
#[test]
#[ignore]
fn dispatch_over_the_device_group_limit_is_split() {
    let mut g = Guest::new(64 << 20);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(crate::shader::SpirvCompiler::for_device(dev, g.mem)), None, None);
    let limit = rig.dev.props.limits.max_compute_work_group_count[1];
    if limit > 1 << 18 {
        eprintln!("skipped: the device launches {limit} groups along Y at once");
        return;
    }
    let groups = limit + 3;
    // v1 = s4 * 64 + v0; v2 = 1; buffer_store_dword v2, v1 idxen (lane 0
    // of each one-thread group).
    let cs = g.program(&[0xD746_0001, 0x0401_0C04, 0x7E04_0281, 0xE070_2000, 0x8000_0201, asm::S_ENDPGM]);
    let words = groups as usize * 64;
    let ob = g.alloc(&asm::bytes(&vec![0u32; words]));
    let user = buffer_descriptor(ob, 4, words as u32, 0x0101_6FAC);
    let mut shader = stage(cs, &user);
    shader.rsrc2 |= 1 << 8; // TGID_Y_EN: s4 = group id y
    let d = ResolvedDispatch {
        packet: DispatchPacket { groups: [1, groups, 1], dispatch_initiator: 1, ..DispatchPacket::default() },
        shader,
        rsrc3: 0,
        threads: [1, 1, 1],
        start: [0; 3],
    };
    assert_eq!(rig.dispatch(&d), DrawOutcome::Recorded);
    assert!(rig.buffers.download(&mut rig.sched, ob, words as u64 * 4));
    let mut out = vec![0u8; words * 4];
    assert!(g.mem.read_backing(ob, &mut out));
    let w: Vec<u32> = out.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let missing: Vec<usize> = (0..groups as usize).filter(|&k| w[k * 64] != 1).take(8).collect();
    assert!(missing.is_empty(), "groups that did not run: {missing:?}");
}
