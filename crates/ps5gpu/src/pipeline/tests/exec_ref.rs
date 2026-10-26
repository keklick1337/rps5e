// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The reference driver's execution tests ported to the interpreter in
//! `ps5shader` (hw_scalar_waves, hw_mem_bases, hw_valu_halves) through the
//! SPIR-V compiler on the device, compared word for word with the
//! interpreter (which those tests hold to the reference's expectations).
//! NaN payloads are not compared.
//!
//! Run one group at a time:
//! `cargo test -p ps5gpu --release exec_ref_<group> -- --ignored`
//! (`EXEC_REF_JOB=text` runs the jobs whose name contains it,
//! `EXEC_REF_VERBOSE=1` names each job as it runs).

use super::*;
use ps5shader::testing::ComputeRun;
use ps5shader::translate::FloatMode;

include!("../../../../ps5shader/tests/vectors/exec_programs.rs");

use exec_programs::{self as ep, SplitMix};

const POISON: u32 = 0xdead_beef;
/// Interpreter address of region `k`.
const REGION: u64 = 0x1_0000_0000;
const REGION_STRIDE: u64 = 0x1000_0000;

const RAW: u32 = 0x3101_6fac;
const STRIDED: u32 = 0x1101_6fac;

fn desc(base: u64, stride: u32, records: u32, dword3: u32) -> [u32; 4] {
    [base as u32, ((base >> 32) as u32 & 0xffff) | (stride << 16), records, dword3]
}

fn raw(base: u64, bytes: u32) -> [u32; 4] {
    desc(base, 0, bytes, RAW)
}

fn strided(base: u64, records: u32) -> [u32; 4] {
    desc(base, 4, records, STRIDED)
}

fn mode(mode: u8, dx10_clamp: bool, ieee: bool, fp16_overflow: bool) -> FloatMode {
    FloatMode { mode, dx10_clamp, ieee, fp16_overflow }
}

/// One dispatch: `regions` hold the initial words of the memory the
/// program uses, `user` builds the user data from their base addresses,
/// `outputs` are the regions compared afterwards.
struct Job {
    name: String,
    code: Vec<u32>,
    wave: u32,
    threads: u32,
    groups: u32,
    regions: Vec<Vec<u32>>,
    user: Box<dyn Fn(&[u64]) -> Vec<u32>>,
    outputs: Vec<usize>,
    float_mode: FloatMode,
    lds_dwords: u32,
    /// Work-group id x in the SGPR after the user data.
    group_id: bool,
    scratch_dwords: Option<u32>,
}

impl Job {
    fn new(name: impl Into<String>, code: &[u32], wave: u32, threads: u32, regions: Vec<Vec<u32>>, outputs: &[usize], user: impl Fn(&[u64]) -> Vec<u32> + 'static) -> Job {
        Job {
            name: name.into(),
            code: code.to_vec(),
            wave,
            threads,
            groups: 1,
            regions,
            user: Box::new(user),
            outputs: outputs.to_vec(),
            float_mode: FloatMode::default(),
            lds_dwords: 0,
            group_id: false,
            scratch_dwords: None,
        }
    }

    fn float_mode(mut self, m: FloatMode) -> Job {
        self.float_mode = m;
        self
    }

    fn lds(mut self, dwords: u32) -> Job {
        self.lds_dwords = dwords;
        self
    }

    fn groups(mut self, n: u32) -> Job {
        self.groups = n;
        self.group_id = true;
        self
    }

    fn scratch(mut self, dwords: u32) -> Job {
        self.scratch_dwords = Some(dwords);
        self
    }

    /// The output regions after the interpreter ran every work-group (in
    /// order, each wave after the other).
    fn interpret(&self) -> Result<Vec<Vec<u32>>, String> {
        let bases: Vec<u64> = (0..self.regions.len() as u64).map(|k| REGION + k * REGION_STRIDE).collect();
        let user = (self.user)(&bases);
        let mut run = ComputeRun::new(self.code.clone(), self.wave, self.wave, user.clone());
        run.float_mode = self.float_mode;
        run.workgroup_id_sgprs = self.group_id as u32;
        let f = run.compile(true)?;
        let mut memory = ps5shader::interp::Memory::default();
        for (r, &b) in self.regions.iter().zip(&bases) {
            memory.write_words(b, r);
        }
        memory.write_words(run.code_address, &run.code);
        for group in 0..self.groups {
            let mut lds = vec![0u8; if self.lds_dwords > 0 { self.lds_dwords as usize * 4 } else { 65536 }];
            let mut first = 0;
            while first < self.threads {
                let n = self.wave.min(self.threads - first);
                let mut m = ps5shader::interp::Machine::new(self.wave, n);
                m.user_data = user.clone();
                m.memory = std::mem::take(&mut memory);
                m.lds = std::mem::take(&mut lds);
                m.scratch_dwords = self.scratch_dwords;
                m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(0), (first..first + n).collect());
                m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(1), vec![0; n as usize]);
                m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationId(2), vec![0; n as usize]);
                m.system.insert(ps5shader::ssa::SystemValue::LocalInvocationIndex, (first..first + n).collect());
                m.system.insert(ps5shader::ssa::SystemValue::WorkgroupId(0), vec![group; n as usize]);
                m.run(&f).map_err(|e| format!("run: {e}"))?;
                memory = std::mem::take(&mut m.memory);
                lds = std::mem::take(&mut m.lds);
                first += n;
            }
        }
        Ok(self.outputs.iter().map(|&k| memory.read_words(bases[k], self.regions[k].len())).collect())
    }

    /// Copy the program and the regions into fresh chunks of the guest
    /// view (before any dispatch: pages near those the GPU wrote may be
    /// write-protected). Returns the code address and the region bases.
    fn place(&self, g: &mut Guest) -> (u64, Vec<u64>) {
        (g.program(&self.code), self.regions.iter().map(|r| g.alloc(&asm::bytes(r))).collect())
    }

    /// Guest view bytes [`Self::place`] takes.
    fn footprint(&self) -> u64 {
        let chunk = |bytes: usize| (bytes as u64).max(1).div_ceil(64 << 10) * (64 << 10);
        chunk(self.code.len() * 4) + self.regions.iter().map(|r| chunk(r.len() * 4)).sum::<u64>()
    }

    /// The output regions after the device ran the dispatch over the
    /// memory [`Self::place`] prepared.
    fn execute(&self, g: &mut Guest, rig: &mut Rig, (cs, bases): (u64, Vec<u64>)) -> Result<Vec<Vec<u32>>, String> {
        let user = (self.user)(&bases);
        if user.len() > 32 {
            return Err(format!("{} user SGPRs", user.len()));
        }
        let mut shader = stage(cs, &user);
        let fm = self.float_mode;
        shader.rsrc1 = (fm.mode as u32) << 12 | (fm.dx10_clamp as u32) << 21 | (fm.ieee as u32) << 23 | (fm.fp16_overflow as u32) << 26;
        shader.rsrc2 = (user.len() as u32) << 1 | (self.group_id as u32) << 7 | self.lds_dwords.div_ceil(128) << 15;
        let d = ResolvedDispatch {
            packet: DispatchPacket { groups: [self.groups, 1, 1], dispatch_initiator: 1 | if self.wave == 32 { 1 << 15 } else { 0 }, ..DispatchPacket::default() },
            shader,
            rsrc3: 0,
            threads: [self.threads, 1, 1],
            start: [0; 3],
        };
        let outcome = rig.dispatch(&d);
        if outcome != DrawOutcome::Recorded {
            return Err(format!("dispatch {outcome:?}"));
        }
        let mut out = Vec::new();
        for &k in &self.outputs {
            let bytes = self.regions[k].len() as u64 * 4;
            if !rig.buffers.download(&mut rig.sched, bases[k], bytes) {
                return Err("download failed".into());
            }
            let mut b = vec![0u8; bytes as usize];
            if !g.mem.read_backing(bases[k], &mut b) {
                return Err("read back failed".into());
            }
            out.push(b.chunks(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect());
        }
        Ok(out)
    }
}

/// Jobs (name prefixes) the device runs differently: V#s whose base is
/// not dword-aligned. The SPIR-V back-end addresses a buffer in dwords from
/// the aligned-down base (its bound length too), where the hardware keeps
/// the base's misalignment and drops only the offset's low bits
/// (BufferUnalignedBase; the interpreter models it).
const KNOWN_DIVERGENT: &[&str] = &["unaligned base view", "unaligned glc pair", "tbuffer_load_format_x 16_UINT view +2"];

/// Equal, or both NaN as f32, or half by half equal or both f16 NaN.
fn same(a: u32, e: u32) -> bool {
    let nan32 = |v: u32| v & 0x7fff_ffff > 0x7f80_0000;
    let nan16 = |v: u32| v & 0x7fff > 0x7c00;
    a == e || (nan32(a) && nan32(e)) || [0, 16].iter().all(|&s| (a >> s) & 0xffff == (e >> s) & 0xffff || (nan16(a >> s) && nan16(e >> s)))
}

fn run_jobs(jobs: Vec<Job>) {
    run_jobs_split(jobs, false)
}

/// [`run_jobs`]; `split_all`: every wave64 program that qualifies runs
/// split over two 32-wide host subgroups (`shader::wave_split`).
fn run_jobs_split(jobs: Vec<Job>, split_all: bool) {
    // EXEC_REF_JOB=text runs only the jobs whose name contains it.
    let only = std::env::var("EXEC_REF_JOB").ok();
    let jobs: Vec<Job> = jobs.into_iter().filter(|j| only.as_deref().is_none_or(|o| j.name.contains(o))).collect();
    let footprint: u64 = jobs.iter().map(Job::footprint).sum();
    let mut g = Guest::new(footprint as usize + (1 << 20));
    let placed: Vec<(u64, Vec<u64>)> = jobs.iter().map(|j| j.place(&mut g)).collect();
    let mem = g.mem;
    let mut rig = Rig::with_compiler(
        &g,
        |dev| {
            let mut c = crate::shader::SpirvCompiler::for_device(dev, mem);
            if split_all {
                c.set_wave64_split(true, true);
            }
            Arc::new(c)
        },
        None,
        None,
    );
    let mut fails = Vec::new();
    for (job, placed) in jobs.iter().zip(placed) {
        if std::env::var_os("EXEC_REF_VERBOSE").is_some() {
            eprintln!("{}", job.name);
        }
        let want = match job.interpret() {
            Ok(w) => w,
            Err(e) => {
                fails.push(format!("{}: interpreter: {e}", job.name));
                continue;
            }
        };
        let got = match job.execute(&mut g, &mut rig, placed) {
            Ok(o) => o,
            Err(e) => {
                fails.push(format!("{}: device: {e}", job.name));
                continue;
            }
        };
        let mut bad = Vec::new();
        for (k, (gr, wr)) in got.iter().zip(&want).enumerate() {
            bad.extend(gr.iter().zip(wr).enumerate().filter(|(_, (a, e))| !same(**a, **e)).map(|(i, (a, e))| format!("{}: region {k} word {i}: device {a:#010x}, interpreter {e:#010x}", job.name)).take(8));
        }
        if KNOWN_DIVERGENT.iter().any(|p| job.name.starts_with(p)) {
            if bad.is_empty() {
                fails.push(format!("{}: matches now, drop it from KNOWN_DIVERGENT", job.name));
            }
        } else {
            fails.extend(bad);
        }
    }
    assert!(fails.is_empty(), "{} mismatches:\n{}", fails.len(), fails.iter().take(120).cloned().collect::<Vec<_>>().join("\n"));
}

fn scalar_wave_jobs() -> Vec<Job> {
    let mut jobs = Vec::new();
    // The device has 16 user SGPRs: the compares see the first 8 values
    // (s8..s15), s16 and s17 read 0 on both sides.
    for (name, code, values) in [("sopk compare", ep::sopk_compare(), ep::SOPK_VALUES[..8].to_vec()), ("sopk waitcnt", ep::sopk_waitcnt(), vec![ep::SOPK_WAIT_VALUE])] {
        jobs.push(Job::new(name, &code, 32, 32, vec![vec![POISON; 32 * 64]], &[0], move |b| {
            let mut u = vec![0u32; 8];
            u[4..8].copy_from_slice(&strided(b[0], 32 * 64));
            u.extend(&values);
            u
        }));
    }
    jobs.push(Job::new("wave32 exec lane bits", &ep::WAVE32_EXEC_LANE_BITS, 32, 32, vec![vec![POISON; 32]], &[0], |b| raw(b[0], 128).to_vec()));
    // ExecHostLanes keeps its user data in s[8:19], past the device's 16
    // user SGPRs: it runs in the interpreter only (hw_scalar_waves).
    for (code, execs) in [(&ep::LDS_WATERFALL[..], &ep::LDS_WATERFALL_EXECS[..]), (&ep::LDS_MASKED_OR[..], &ep::LDS_MASKED_OR_EXECS[..])] {
        for &exec in execs {
            jobs.push(
                Job::new(format!("lds mask waterfall {} exec {exec:#018x}", code.len()), code, 64, 64, vec![vec![POISON; 256]], &[0], move |b| {
                    let mut u = vec![exec as u32, (exec >> 32) as u32, 0, 0];
                    u.extend(desc(b[0], 16, 64, STRIDED));
                    u
                })
                .lds(128),
            );
        }
    }
    for wave in [64u32, 32] {
        for preset in [1u32, 0] {
            let mut block = vec![0u32; 16384];
            block[0] = preset;
            jobs.push(Job::new(format!("cross workgroup flag wave{wave} preset {preset}"), &ep::CROSS_WORKGROUP_FLAG, wave, 64, vec![block], &[0], |b| vec![b[0] as u32, (b[0] >> 32) as u32]).groups(2));
        }
    }
    for wave in [32u32, 64] {
        let mut input = vec![0u32; 256];
        for t in 0..wave as usize {
            input[t * 4] = 0x1000 * (t as u32 + 1);
            input[t * 4 + 1] = 0x0077_0000 + t as u32 * 0x101;
        }
        jobs.push(Job::new(format!("vector window direct wave{wave}"), &ep::VECTOR_WINDOW_DIRECT, wave, wave, vec![input, vec![POISON; 64 * 16]], &[1], |b| {
            let mut u = raw(b[0], 1024).to_vec();
            u.extend(raw(b[1], 4096));
            u
        }));
        for data in [20u32, 2, 31] {
            jobs.push(Job::new(format!("vector window narrowing wave{wave} data {data}"), &ep::VECTOR_WINDOW_NARROWING, wave, wave, vec![vec![POISON; 64 * 16]], &[0], move |b| {
                let mut u = vec![0u32; 4];
                u.extend(raw(b[0], 4096));
                u.extend([7, data]);
                u
            }));
        }
    }
    jobs
}

/// ScalarSopkCompare, ScalarSopkWaitcnt, Wave32ExecLaneBits,
/// ExecHostLanes (compute), LdsMaskWaterfall, CrossWorkgroupFlag,
/// VectorWindowDirectAccess and VectorWindowNarrowing.
#[test]
#[ignore]
fn exec_ref_scalar_waves_on_device() {
    run_jobs(scalar_wave_jobs());
}

/// [`exec_ref_scalar_waves_on_device`] with every wave64 program split
/// over two host subgroups (LdsMaskWaterfall's spanned mode).
#[test]
#[ignore]
fn exec_ref_scalar_waves_split_on_device() {
    run_jobs_split(scalar_wave_jobs(), true);
}

fn memory_jobs() -> Vec<Job> {
    let mut jobs = Vec::new();
    let vcc_input: Vec<u32> = (0..64).map(|l| 0x3f80_0000 | (l * 0x10001)).collect();
    let mut vcc_runs = Vec::new();
    for load in [0xe8b0_1000u32, 0xe030_1000, 0xe000_1000] {
        vcc_runs.push((load, 256u32, RAW, false));
        vcc_runs.push((load, 12, RAW, false));
    }
    vcc_runs.push((0xe000_1000, 256, 0x3103_8fac, false));
    for store in [0xe070_1000u32, 0xe8b4_1000] {
        vcc_runs.push((store, 256, RAW, true));
        vcc_runs.push((store, 12, RAW, true));
    }
    for wave in [32u32, 64] {
        for &(opcode, bytes, format, store) in &vcc_runs {
            jobs.push(Job::new(format!("resource vcc wave{wave} {opcode:#010x} bytes {bytes} store={store}"), &ep::resource_vcc(opcode, store), wave, 64, vec![vcc_input.clone(), vec![POISON; 72]], &[1], move |b| {
                let output = desc(b[1] + 16, 0, if store { bytes } else { 256 }, format);
                let input = desc(b[0], 0, if store { 256 } else { bytes }, format);
                let (ordinary, mixed) = if store { (input, output) } else { (output, input) };
                [ordinary, mixed].concat()
            }));
        }
    }
    let bytes_words = |f: &dyn Fn(u32) -> u8| -> Vec<u32> { (0..64u32).map(|w| u32::from_le_bytes([f(w * 4), f(w * 4 + 1), f(w * 4 + 2), f(w * 4 + 3)])).collect() };
    for (view, store) in [(2u32, 1u32), (3, 3), (1, 2)] {
        let input = bytes_words(&|i| (i.wrapping_mul(37) + 11 + (i >> 3)) as u8);
        jobs.push(Job::new(format!("unaligned base view +{view} stores +{store}"), &ep::UNALIGNED_BASE, 32, 32, vec![input, vec![POISON; 384]], &[1], move |b| {
            [raw(b[0], 256), raw(b[0] + view as u64, 60), raw(b[1], 1024), raw(b[1] + 1024 + store as u64, 224)].concat()
        }));
    }
    for view in [1u32, 2, 3] {
        let input = bytes_words(&|i| (i.wrapping_mul(53) + 7 + (i >> 2)) as u8);
        jobs.push(Job::new(format!("unaligned glc pair view +{view}"), &ep::UNALIGNED_PAIR, 32, 32, vec![input, vec![POISON; 384]], &[1], move |b| [raw(b[0], 256), raw(b[0] + view as u64, 60), raw(b[1], 1024)].concat()));
        jobs.push(Job::new(format!("unaligned atomic view +{view}"), &ep::UNALIGNED_ATOMIC, 32, 32, vec![vec![POISON; 384]], &[0], move |b| {
            let mut u = [raw(b[0] + view as u64, 34 * 4), raw(b[0] + 512, 128)].concat();
            u.push(4 - view);
            u
        }));
    }
    let input = bytes_words(&|i| (i.wrapping_mul(29) + 5 + (i >> 4)) as u8);
    jobs.push(Job::new("tbuffer_load_format_x 16_UINT view +2", &ep::formatted_load(11, 2), 32, 32, vec![input, vec![POISON; 384]], &[1], |b| [raw(b[0] + 2, 60), raw(b[1], 1024)].concat()));
    jobs.push(Job::new("tbuffer_store_format_x 16_UINT view +2", &ep::formatted_store(11, 0), 32, 32, vec![vec![POISON; 384]], &[0], |b| [raw(b[0] + 2, 128), raw(b[0], 1024)].concat()));
    let value = |tid: u32, c: u32| {
        let mixed = (tid + 1).wrapping_mul(2654435761).wrapping_add(c * 40503);
        if tid % 5 == 3 {
            mixed | 0x10000
        } else {
            (mixed >> 3) & 0xffff
        }
    };
    let input: Vec<u32> = (0..32).flat_map(|t| (0..4).map(move |c| value(t, c))).collect();
    jobs.push(Job::new("formatted store unaligned", &ep::formatted_store_unaligned(), 32, 32, vec![input, vec![0xdede_dede; 512]], &[1], |b| [raw(b[0], 512), raw(b[1], 2048)].concat()));
    for wave in [32u32, 64] {
        for (offset, destination, inactive, contended) in ep::CONDXCHG_RUNS {
            let input = ep::condxchg_input(wave, offset, contended);
            let mut job = Job::new(
                format!("lds condxchg wave{wave} offset {offset:#x} v{destination} inactive={inactive} contended={contended}"),
                &ep::lds_condxchg(wave, offset, destination, inactive, contended),
                wave,
                wave,
                vec![input, vec![ep::CONDXCHG_SENTINEL; wave as usize * 8]],
                &[1],
                |b| [raw(b[0], 2048), raw(b[1], 2048)].concat(),
            )
            .lds(ep::CONDXCHG_LDS_DWORDS);
            if contended {
                // Which lane wins each exchange is the device's order.
                job.outputs.clear();
            }
            jobs.push(job);
        }
    }
    for wave in [32u32, 64] {
        jobs.push(Job::new(format!("flat aperture sources wave{wave}"), &ep::flat_aperture_sources(), wave, wave, vec![vec![POISON; wave as usize * 32]], &[0], move |b| raw(b[0], wave * 128).to_vec()).lds(256).scratch(4));
    }
    jobs
}

/// BufferResourceVcc, BufferUnalignedBase, FormattedStoreUnaligned,
/// LdsCondxchg and FlatApertureSources.
#[test]
#[ignore]
fn exec_ref_memory_on_device() {
    run_jobs(memory_jobs());
}

fn valu_jobs(f16_flushed: bool) -> Vec<Job> {
    let mut jobs = Vec::new();
    let io = |inp: usize, out: usize| move |b: &[u64]| [raw(b[0], inp as u32 * 4), raw(b[1], out as u32 * 4)].concat();
    if !f16_flushed {
        let input = ep::mul_lo_u16_input();
        for (k, code) in ep::mul_lo_u16().iter().enumerate() {
            jobs.push(Job::new(format!("mul_lo_u16 program {k}"), code, 32, 64, vec![input.clone(), vec![POISON; 64]], &[1], |b| [strided(b[0], 256), strided(b[1], 64)].concat()));
        }
    }
    let min_max_modes: Vec<(&str, FloatMode)> = if f16_flushed {
        vec![("f16 flushed dx10 ieee", mode(0x30, true, true, false)), ("f16 input flushed", mode(0xb0, false, false, false))]
    } else {
        vec![("default", FloatMode::default()), ("dx10 ieee", mode(0xc0, true, true, false)), ("dx10", mode(0xc0, true, false, false)), ("ieee", mode(0xc0, false, true, false)), ("none", mode(0xc0, false, false, false))]
    };
    let mut rng = SplitMix(0x2545_f491_4f6c_dd1d);
    for (name, fm) in &min_max_modes {
        let mut input = vec![0u32; 256];
        for tid in 0..64 {
            let row = &mut input[tid * 4..tid * 4 + 4];
            for w in row.iter_mut() {
                *w = ep::f16_min_max_half(&mut rng) | (ep::f16_min_max_half(&mut rng) << 16);
            }
            match tid % 8 {
                0 => row[1] = row[0],
                1 => row[1] = row[0] ^ 0x8000_8000,
                2 => row[2] = row[1],
                3 => row[2] = row[0] ^ 0x8000_8000,
                _ => {}
            }
        }
        for (what, code) in [("scalar", &ep::F16_MIN_MAX_SCALAR[..]), ("packed", &ep::F16_MIN_MAX_PACKED[..])] {
            jobs.push(Job::new(format!("f16 min/max {what} {name}"), code, 32, 64, vec![input.clone(), vec![POISON; 512]], &[1], io(256, 512)).float_mode(*fm));
        }
    }
    let native_modes: Vec<(&str, FloatMode)> = if f16_flushed {
        vec![("f16 denormals flushed", mode(0x30, true, false, false))]
    } else {
        vec![
            ("default", FloatMode::default()),
            ("dx10 ieee ovfl", mode(0xc0, true, true, true)),
            ("dx10 ieee", mode(0xc0, true, true, false)),
            ("dx10 ovfl", mode(0xc0, true, false, true)),
            ("ieee ovfl", mode(0xc0, false, true, true)),
            ("ieee", mode(0xc0, false, true, false)),
            ("ovfl", mode(0xc0, false, false, true)),
            ("none", mode(0xc0, false, false, false)),
        ]
    };
    let mut rng = SplitMix(0x9e37_79b9_7f4a_7c15);
    for (name, fm) in &native_modes {
        for batch in 0..4 {
            let input: Vec<u32> = (0..256).map(|_| ep::f16_native_half(&mut rng) | (ep::f16_native_half(&mut rng) << 16)).collect();
            jobs.push(Job::new(format!("f16 native scalar {name} batch {batch}"), &ep::F16_NATIVE_SCALAR, 32, 64, vec![input.clone(), vec![POISON; 1024]], &[1], io(256, 1024)).float_mode(*fm));
            jobs.push(Job::new(format!("f16 native packed {name} batch {batch}"), &ep::F16_NATIVE_PACKED, 32, 64, vec![input, vec![POISON; 512]], &[1], io(256, 512)).float_mode(*fm));
        }
    }
    if f16_flushed {
        return jobs;
    }
    let pknorm_rows: [[u32; 2]; 8] = [[0x400, 0x400], [0x83ff, 0x83ff], [0x3555, 0xd640], [0xbc00, 0x3c00], [0x7bff, 0x400], [0xbeef0000, 0xbeef3c00], [0xbeef8000, 0xbeef0000], [0xbeef0000, 0xbeeffc00]];
    let mut pknorm_input = vec![0u32; 128];
    for (i, r) in pknorm_rows.iter().enumerate() {
        pknorm_input[i * 4..i * 4 + 2].copy_from_slice(r);
    }
    for (abs, neg, op_sel) in [(1, 0, 0), (2, 0, 0), (0, 1, 0), (0, 2, 0), (1, 1, 0), (2, 2, 0), (0, 0, 1), (0, 0, 2)] {
        for fm in [mode(0xf0, true, true, false), mode(0xc0, true, false, false)] {
            jobs.push(Job::new(format!("cvt_pknorm abs {abs} neg {neg} op_sel {op_sel} mode {:#x}", fm.mode), &ep::cvt_pknorm(abs, neg, op_sel), 32, 32, vec![pknorm_input.clone(), vec![POISON; 64]], &[1], io(128, 64)).float_mode(fm));
        }
    }
    // (cmpx, cmp) of eq, ge, gt, le, lg, lt, neq, nge, ngt, nle, nlg, nlt.
    let compares: [u32; 12] = [0x12, 0x16, 0x14, 0x13, 0x15, 0x11, 0x1d, 0x19, 0x1b, 0x1c, 0x1a, 0x1e];
    for op in compares {
        for exec in [true, false] {
            let word = 0x7c00_0b04 | (if exec { op } else { op - 0x10 }) << 17;
            for fm in [mode(0xf0, true, true, false), mode(0xf0, false, false, false), mode(0xc0, true, true, false), mode(0xc0, false, false, false)] {
                for group in 0..8usize {
                    let mut input = vec![0u32; 128];
                    for lane in 0..32 {
                        let pair = group * 32 + lane;
                        input[lane * 4] = ep::CMP_F32_VALUES[pair / 16];
                        input[lane * 4 + 1] = ep::CMP_F32_VALUES[pair % 16];
                    }
                    jobs.push(Job::new(format!("v_cmp{} {word:#010x} mode {:#x} ieee {} group {group}", if exec { "x" } else { "" }, fm.mode, fm.ieee), &ep::cmp_f32(word, exec), 32, 32, vec![input, vec![POISON; 32]], &[1], io(128, 32)).float_mode(fm));
                }
            }
        }
    }
    let clamp_rows: [(u32, u32, u32, FloatMode, Vec<[u32; 4]>); 3] = [
        (0xca, 4, 5, mode(0xf0, true, true, false), vec![[0, 0, 0, 0], [0xbc00, 0, 0, 0], [0x3555, 0, 0, 0], [1, 0x8000, 0, 0], [0xfc00, 0x8000, 0, 0], [0x7c01, 0x3c00, 0, 0], [0x6400, 0x3c00, 0, 0], [0xbeef8000, 0xbeefbc00, 0xbeef0000, 0]]),
        (0x23, 4, 6, mode(0xc0, true, false, false), vec![[0, 0, 0, 0], [0, 0xbff00000, 0, 0xfff00000], [0, 0xc0040000, 0, 0xbff00000], [0, 0x00100000, 1, 0x7ff00000], [0, 0xfff00000, 0, 0x40040000], [1, 0x7ff00000, 0, 0x41e00000]]),
        (0x2b, 4, 6, mode(0xf0, true, true, false), vec![[0, 0, 0, 0], [0, 0xbff00000, 0, 0xfff00000], [0, 0xc0040000, 0, 0xbff00000], [0, 0x00100000, 1, 0x7ff00000]]),
    ];
    for (op, s0, s1, fm, rows) in clamp_rows {
        let mut input = vec![0u32; 128];
        for (i, r) in rows.iter().enumerate() {
            input[i * 4..i * 4 + 4].copy_from_slice(r);
        }
        jobs.push(Job::new(format!("vop3 compare {op:#x} clamp"), &ep::cmp_clamp(op, s0, s1), 32, 32, vec![input, vec![POISON; 128]], &[1], |b| [strided(b[0], 128), strided(b[1], 128)].concat()).float_mode(fm));
    }
    let halves: Vec<u32> = (0..65536u32).map(|l| l | ((l ^ 0xa5a5) << 16)).collect();
    for (name, code, out) in [("cvt_f32_f16", &ep::F16_TO_F32_CONVERT[..], 65536usize), ("add_f16 sdwa byte", &ep::F16_TO_F32_BYTE[..], 65536), ("cmp f16", &ep::F16_TO_F32_COMPARE[..], 131072)] {
        jobs.push(Job::new(format!("f16 to f32 {name}"), code, 32, 32, vec![halves.clone(), vec![POISON; out]], &[1], move |b| {
            let mut u = raw(b[0], 65536 * 4).to_vec();
            u.extend(raw(b[1], out as u32 * 4));
            u
        }).float_mode(mode(0xc0, true, false, false)).groups(2048));
    }
    let dpp8_input = ep::dpp8_input();
    for wave in [32u32, 64] {
        for lanes in [[7u32, 6, 5, 4, 3, 2, 1, 0], [1, 0, 3, 2, 5, 4, 7, 6]] {
            jobs.push(Job::new(format!("dpp8 pk_fmac wave{wave} {lanes:?}"), &ep::dpp8_pk_fmac(lanes), wave, wave, vec![dpp8_input.clone(), vec![POISON; 256]], &[1], io(256, 256)));
        }
    }
    jobs
}

/// MulLoU16HighHalf, Float16MinMaxBits, Float16Native,
/// PackedFloat16Native, VCvtPknormF16SourceModifiers, VCmpF32Denormals,
/// VCmpEqF16Clamp, F16ToF32Unpack and Dpp8PkFmac.
#[test]
#[ignore]
fn exec_ref_valu_on_device() {
    run_jobs(valu_jobs(false));
}

/// The f16 min/max and arithmetic programs with f16 denormals flushed in
/// FLOAT_MODE.
#[test]
#[ignore]
fn exec_ref_valu_f16_flushed_on_device() {
    run_jobs(valu_jobs(true));
}
