// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The driver's compiler on SPIR-V files from disk, outside a game: the
//! compute pipeline of each file is created the way
//! `provider::create_compute_pipeline` creates it (same device, flags and
//! stage structures), with the set layout and the push-constant range
//! derived from the module. For modules saved by `RPS5E_COMPILE_NOTES`
//! when a compile takes the process down inside the driver.
//!
//! `RPS5E_DC_SPV=<file or directory>[,...] cargo test -p ps5gpu --release -- --ignored driver_compiles_spirv_files --nocapture`
//!
//! * `RPS5E_DC_N` (1): creations of each file per thread;
//! * `RPS5E_DC_THREADS` (1): threads creating at once (thread `t` starts at
//!   file `t`);
//! * `RPS5E_DC_CACHE` (1): one `VkPipelineCache` shared by all threads, 0:
//!   none;
//! * `RPS5E_DC_STATS` (1), `RPS5E_DC_ROBUST` (1): CAPTURE_STATISTICS and
//!   the robustness structure, when the device has them;
//! * `RPS5E_DC_SUBGROUP` (0): the required subgroup size;
//! * `RPS5E_DC_STACK_KB` (2048): the stack of the creating threads.
//!
//! A compile that dies in the driver ends the test process; the SIGILL
//! handler installed here first prints the text the driver's compiler
//! passed to its fatal-error handler.

use crate::device::GpuDevice;
use ash::vk;
use rspirv::dr::Operand;
use rspirv::spirv::{Decoration, Op, StorageClass};
use std::collections::HashMap;

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// What the harness needs of a module: its entry name, the bindings of
/// set 0 and the size of the push-constant block.
struct Shape {
    entry: String,
    bindings: Vec<(u32, vk::DescriptorType, u32)>,
    push_bytes: u32,
}

fn shape(code: &[u32]) -> Shape {
    let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    let m = rspirv::dr::load_bytes(&bytes).expect("SPIR-V parses");
    let types: HashMap<u32, &rspirv::dr::Instruction> = m.types_global_values.iter().filter_map(|i| i.result_id.map(|id| (id, i))).collect();
    let id = |o: &Operand| if let Operand::IdRef(i) = o { *i } else { panic!("id operand") };
    let constant = |i: u32| match types[&i].operands[0] {
        Operand::LiteralBit32(v) => v,
        _ => panic!("array length"),
    };
    let mut binding = HashMap::new();
    let mut set = HashMap::new();
    let mut stride = HashMap::new();
    let mut offsets: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut buffer_block = std::collections::HashSet::new();
    for a in &m.annotations {
        match (a.class.opcode, &a.operands[..]) {
            (Op::Decorate, [Operand::IdRef(t), Operand::Decoration(Decoration::Binding), Operand::LiteralBit32(v)]) => {
                binding.insert(*t, *v);
            }
            (Op::Decorate, [Operand::IdRef(t), Operand::Decoration(Decoration::DescriptorSet), Operand::LiteralBit32(v)]) => {
                set.insert(*t, *v);
            }
            (Op::Decorate, [Operand::IdRef(t), Operand::Decoration(Decoration::ArrayStride), Operand::LiteralBit32(v)]) => {
                stride.insert(*t, *v);
            }
            (Op::Decorate, [Operand::IdRef(t), Operand::Decoration(Decoration::BufferBlock)]) => {
                buffer_block.insert(*t);
            }
            (Op::MemberDecorate, [Operand::IdRef(t), Operand::LiteralBit32(_), Operand::Decoration(Decoration::Offset), Operand::LiteralBit32(v)]) => offsets.entry(*t).or_default().push(*v),
            _ => {}
        }
    }
    // Byte size of a push-constant member type (scalars, vectors and arrays
    // of them, which is all the emitter puts there).
    fn size(t: u32, types: &HashMap<u32, &rspirv::dr::Instruction>, stride: &HashMap<u32, u32>, constant: &dyn Fn(u32) -> u32) -> u32 {
        let i = types[&t];
        let idr = |o: &Operand| if let Operand::IdRef(i) = o { *i } else { 0 };
        match i.class.opcode {
            Op::TypeInt | Op::TypeFloat => match i.operands[0] {
                Operand::LiteralBit32(w) => w / 8,
                _ => 4,
            },
            Op::TypeVector => match i.operands[1] {
                Operand::LiteralBit32(n) => n * size(idr(&i.operands[0]), types, stride, constant),
                _ => 16,
            },
            Op::TypeArray => constant(idr(&i.operands[1])) * stride.get(&t).copied().unwrap_or_else(|| size(idr(&i.operands[0]), types, stride, constant)),
            _ => 16,
        }
    }
    let mut out = Shape { entry: String::new(), bindings: Vec::new(), push_bytes: 0 };
    for e in &m.entry_points {
        if let Operand::LiteralString(s) = &e.operands[2] {
            out.entry = s.clone();
        }
    }
    let mut seen: HashMap<u32, (vk::DescriptorType, u32)> = HashMap::new();
    for v in &m.types_global_values {
        if v.class.opcode != Op::Variable {
            continue;
        }
        let Operand::StorageClass(class) = v.operands[0] else { continue };
        let pointee = id(&types[&v.result_type.unwrap()].operands[1]);
        if class == StorageClass::PushConstant {
            let st = types[&pointee];
            let offs = offsets.get(&pointee).cloned().unwrap_or_default();
            for (k, member) in st.operands.iter().enumerate() {
                let end = offs.get(k).copied().unwrap_or(0) + size(id(member), &types, &stride, &constant);
                out.push_bytes = out.push_bytes.max(end);
            }
            continue;
        }
        let var = v.result_id.unwrap();
        let (Some(&b), Some(&s)) = (binding.get(&var), set.get(&var)) else { continue };
        assert_eq!(s, 0, "the harness handles set 0 only (binding {b})");
        let (mut ty, mut count) = (pointee, 1);
        match types[&ty].class.opcode {
            Op::TypeArray => {
                count = constant(id(&types[&ty].operands[1]));
                ty = id(&types[&ty].operands[0]);
            }
            Op::TypeRuntimeArray => panic!("runtime descriptor arrays (the bindless set) are not handled"),
            _ => {}
        }
        let t = types[&ty];
        let kind = match (class, t.class.opcode) {
            (StorageClass::StorageBuffer, _) => vk::DescriptorType::STORAGE_BUFFER,
            (StorageClass::Uniform, _) if buffer_block.contains(&ty) => vk::DescriptorType::STORAGE_BUFFER,
            (StorageClass::Uniform, _) => vk::DescriptorType::UNIFORM_BUFFER,
            (_, Op::TypeSampler) => vk::DescriptorType::SAMPLER,
            (_, Op::TypeSampledImage) => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            (_, Op::TypeAccelerationStructureKHR) => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            (_, Op::TypeImage) => {
                let buffer = matches!(t.operands[1], Operand::Dim(rspirv::spirv::Dim::DimBuffer));
                let sampled = matches!(t.operands[5], Operand::LiteralBit32(1));
                match (buffer, sampled) {
                    (false, true) => vk::DescriptorType::SAMPLED_IMAGE,
                    (false, false) => vk::DescriptorType::STORAGE_IMAGE,
                    (true, true) => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                    (true, false) => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
                }
            }
            other => panic!("binding {b}: {other:?}"),
        };
        // Two variables may view one binding (a read-only and a writable
        // block type over the same buffers).
        let e = seen.entry(b).or_insert((kind, count));
        assert_eq!(e.0, kind, "binding {b} has one descriptor type");
        e.1 = e.1.max(count);
    }
    out.bindings = seen.into_iter().map(|(b, (k, c))| (b, k, c)).collect();
    out.bindings.sort_by_key(|b| b.0);
    out
}

/// SIGILL in the test process: print the string RSI points at (the reason
/// the driver's compiler hands its fatal-error handler, which is `ud2`),
/// then die of the signal.
extern "C" fn on_sigill(sig: libc::c_int, _info: *mut libc::siginfo_t, uc: *mut libc::c_void) {
    // SAFETY: kernel-provided context; memory is read through
    // process_vm_readv, which fails instead of faulting.
    unsafe {
        let uc = uc as *mut libc::ucontext_t;
        let rip = (*uc).uc_mcontext.gregs[libc::REG_RIP as usize] as u64;
        let rsi = (*uc).uc_mcontext.gregs[libc::REG_RSI as usize] as u64;
        let rsp = (*uc).uc_mcontext.gregs[libc::REG_RSP as usize] as u64;
        let mut buf = [0u8; 512];
        let mut got = 0;
        // Up to the end of the page, then the next one (either may be absent).
        let first = (4096 - (rsi & 0xfff) as usize).min(buf.len());
        for (off, len) in [(0, first), (first, buf.len() - first)] {
            if len == 0 {
                continue;
            }
            let local = libc::iovec { iov_base: buf.as_mut_ptr().add(off).cast(), iov_len: len };
            let remote = libc::iovec { iov_base: (rsi as usize + off) as *mut _, iov_len: len };
            let n = libc::process_vm_readv(libc::getpid(), &local, 1, &remote, 1, 0);
            if n <= 0 {
                break;
            }
            got += n as usize;
        }
        let text = &buf[..got];
        let text = &text[..text.iter().position(|&c| c == 0).unwrap_or(text.len())];
        let mut name = [0u8; 32];
        libc::pthread_getname_np(libc::pthread_self(), name.as_mut_ptr().cast(), name.len());
        let name = std::ffi::CStr::from_ptr(name.as_ptr().cast()).to_string_lossy();
        let line = format!("\n[driver-compile] SIGILL on thread {name} at {rip:#x} rsp {rsp:#x}; rsi -> {:?}\n", String::from_utf8_lossy(text));
        libc::write(2, line.as_ptr().cast(), line.len());
        // The mapping the instruction is in.
        if let Ok(maps) = std::fs::read_to_string("/proc/self/maps") {
            for l in maps.lines() {
                let mut it = l.split(['-', ' ']);
                let (Some(a), Some(b)) = (it.next().and_then(|v| u64::from_str_radix(v, 16).ok()), it.next().and_then(|v| u64::from_str_radix(v, 16).ok())) else { continue };
                if (a..b).contains(&rip) || (a..b).contains(&rsp) {
                    let line = format!("[driver-compile] {l}\n");
                    libc::write(2, line.as_ptr().cast(), line.len());
                }
            }
        }
        libc::signal(sig, libc::SIG_DFL);
        libc::raise(sig);
    }
}

#[test]
#[ignore]
fn driver_compiles_spirv_files() {
    let Some(list) = std::env::var_os("RPS5E_DC_SPV") else {
        eprintln!("RPS5E_DC_SPV is not set: nothing to compile");
        return;
    };
    let mut files = Vec::new();
    for p in list.to_string_lossy().split(',').filter(|p| !p.is_empty()) {
        let p = std::path::PathBuf::from(p);
        if p.is_dir() {
            let mut inside: Vec<_> = std::fs::read_dir(&p).unwrap().flatten().map(|e| e.path()).filter(|f| f.extension().is_some_and(|e| e == "spv")).collect();
            inside.sort();
            files.extend(inside);
        } else {
            files.push(p);
        }
    }
    assert!(!files.is_empty(), "no .spv file in RPS5E_DC_SPV");
    // SAFETY: installing a handler that only reads and then re-raises.
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = on_sigill as *const () as usize;
        sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
        libc::sigaction(libc::SIGILL, &sa, std::ptr::null_mut());
    }
    let n = env_u32("RPS5E_DC_N", 1);
    let threads = env_u32("RPS5E_DC_THREADS", 1).max(1) as usize;
    let with_cache = env_u32("RPS5E_DC_CACHE", 1) != 0;
    let stats = env_u32("RPS5E_DC_STATS", 1) != 0;
    let robust = env_u32("RPS5E_DC_ROBUST", 1) != 0;
    let subgroup = env_u32("RPS5E_DC_SUBGROUP", 0);
    let stack = env_u32("RPS5E_DC_STACK_KB", 2048) as usize * 1024;

    let dev = GpuDevice::create(false).expect("device");
    eprintln!("[driver-compile] {} | executable info {} robustness {}", dev.name, dev.caps.pipeline_executable_info, dev.caps.pipeline_robustness);
    struct Unit {
        name: String,
        words: usize,
        entry: std::ffi::CString,
        module: vk::ShaderModule,
        layout: vk::PipelineLayout,
    }
    let d = &dev.device;
    let mut units = Vec::new();
    for f in &files {
        let bytes = std::fs::read(f).unwrap();
        let code: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let s = shape(&code);
        let bindings: Vec<_> =
            s.bindings.iter().map(|&(b, k, c)| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(k).descriptor_count(c).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
        let ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(s.push_bytes.next_multiple_of(4))];
        // SAFETY: valid create infos; the handles live until the process ends.
        let (module, layout) = unsafe {
            let set = d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR).bindings(&bindings), None).unwrap();
            let mut info = vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set));
            if s.push_bytes != 0 {
                info = info.push_constant_ranges(&ranges);
            }
            (d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).unwrap(), d.create_pipeline_layout(&info, None).unwrap())
        };
        eprintln!("[driver-compile] {}: {} words, entry {:?}, push {} bytes, bindings {:?}", f.display(), code.len(), s.entry, s.push_bytes, s.bindings);
        units.push(Unit { name: f.file_name().unwrap().to_string_lossy().into_owned(), words: code.len(), entry: std::ffi::CString::new(s.entry).unwrap(), module, layout });
    }
    // SAFETY: valid create info.
    let cache = if with_cache { unsafe { d.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None).unwrap() } } else { vk::PipelineCache::null() };
    let units = &units;
    let dev = &dev;
    std::thread::scope(|scope| {
        for t in 0..threads {
            std::thread::Builder::new()
                .name(format!("dc-compile-{t}"))
                .stack_size(stack)
                .spawn_scoped(scope, move || {
                    for round in 0..n {
                        for k in 0..units.len() {
                            let u = &units[(k + t) % units.len()];
                            let mut required = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(subgroup);
                            let mut stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(u.module).name(&u.entry);
                            if subgroup != 0 {
                                stage = stage.push_next(&mut required);
                            }
                            let mut robustness = vk::PipelineRobustnessCreateInfoEXT::default()
                                .storage_buffers(vk::PipelineRobustnessBufferBehaviorEXT::DISABLED)
                                .uniform_buffers(vk::PipelineRobustnessBufferBehaviorEXT::DEVICE_DEFAULT)
                                .vertex_inputs(vk::PipelineRobustnessBufferBehaviorEXT::DEVICE_DEFAULT)
                                .images(vk::PipelineRobustnessImageBehaviorEXT::DEVICE_DEFAULT);
                            if dev.caps.pipeline_robustness && robust {
                                stage = stage.push_next(&mut robustness);
                            }
                            let mut flags = vk::PipelineCreateFlags::DISPATCH_BASE;
                            if dev.caps.pipeline_executable_info && stats {
                                flags |= vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR;
                            }
                            let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(u.layout).flags(flags);
                            let at = std::time::Instant::now();
                            // SAFETY: valid create info; the pipeline is destroyed at once.
                            match unsafe { dev.device.create_compute_pipelines(cache, &[info], None) } {
                                Ok(p) => unsafe { dev.device.destroy_pipeline(p[0], None) },
                                Err((_, e)) => eprintln!("[driver-compile] {}: {e:?}", u.name),
                            }
                            eprintln!("[driver-compile] thread {t} round {round} {} ({} words) {:.0} ms", u.name, u.words, at.elapsed().as_secs_f64() * 1e3);
                        }
                    }
                })
                .unwrap();
        }
    });
    eprintln!("[driver-compile] done: {} files x {n} x {threads} threads", units.len());
}
