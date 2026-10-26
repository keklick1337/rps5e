// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Resource planning over captured game shaders.
//!
//! Plans every shader of `tests/corpus/` (and of `PS5SHADER_CORPUS=<dir>`,
//! opt-in) using the dumped AGC metadata for the stage, user-data base and
//! count, wave size and the embedded vertex-fetch tables; reports how many
//! plans are static, need per-draw memory walks, select descriptors at run
//! time, fall back to device addresses, or fail. Also checks every plan
//! against the AGC sharp-resource tables and smoke-tests materialisation.

use ps5shader::resources::metadata::{self, AgcShaderType, DirectResource, ShaderUserDataLayout};
use ps5shader::resources::vertex_fetch;
use ps5shader::resources::*;
use ps5shader::*;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

// ---- minimal JSON ----

#[derive(Clone, Debug)]
enum Json {
    Null,
    Bool,
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    fn parse(text: &str) -> Option<Json> {
        let mut p = Parser { s: text.as_bytes(), i: 0 };
        p.value()
    }

    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(o) => o.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn u64(&self) -> Option<u64> {
        match self {
            Json::Number(n) => Some(*n as u64),
            _ => None,
        }
    }

    fn str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    fn array(&self) -> &[Json] {
        match self {
            Json::Array(a) => a,
            _ => &[],
        }
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn value(&mut self) -> Option<Json> {
        self.ws();
        match *self.s.get(self.i)? {
            b'{' => {
                self.i += 1;
                let mut o = Vec::new();
                loop {
                    self.ws();
                    if self.s[self.i] == b'}' {
                        self.i += 1;
                        return Some(Json::Object(o));
                    }
                    let Json::String(k) = self.value()? else { return None };
                    self.ws();
                    self.i += 1; // ':'
                    let v = self.value()?;
                    o.push((k, v));
                    self.ws();
                    if self.s[self.i] == b',' {
                        self.i += 1;
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut a = Vec::new();
                loop {
                    self.ws();
                    if self.s[self.i] == b']' {
                        self.i += 1;
                        return Some(Json::Array(a));
                    }
                    a.push(self.value()?);
                    self.ws();
                    if self.s[self.i] == b',' {
                        self.i += 1;
                    }
                }
            }
            b'"' => {
                self.i += 1;
                let start = self.i;
                while self.s[self.i] != b'"' {
                    self.i += 1;
                }
                let s = String::from_utf8_lossy(&self.s[start..self.i]).into_owned();
                self.i += 1;
                Some(Json::String(s))
            }
            b't' => {
                self.i += 4;
                Some(Json::Bool)
            }
            b'f' => {
                self.i += 5;
                Some(Json::Bool)
            }
            b'n' => {
                self.i += 4;
                Some(Json::Null)
            }
            _ => {
                let start = self.i;
                while self.i < self.s.len() && matches!(self.s[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    self.i += 1;
                }
                std::str::from_utf8(&self.s[start..self.i]).ok()?.parse().ok().map(Json::Number)
            }
        }
    }
}

// ---- corpus ----

struct Shader {
    hash: String,
    ty: AgcShaderType,
    code_address: u64,
    bytes: Vec<u8>,
    meta: Json,
}

fn sample_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus")
}

fn load(dir: &Path) -> (Vec<Shader>, Vec<Json>) {
    let mut shaders = Vec::new();
    let mut fused = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        if name.starts_with("fused-") {
            fused.push(Json::parse(&std::fs::read_to_string(&p).unwrap()).unwrap());
            continue;
        }
        if p.extension().is_none_or(|x| x != "bin") {
            continue;
        }
        let meta = Json::parse(&std::fs::read_to_string(p.with_extension("json")).unwrap()).expect("metadata");
        let ty = AgcShaderType::from_raw(meta.get("type").and_then(Json::u64).unwrap() as u8).unwrap();
        shaders.push(Shader {
            hash: p.file_stem().unwrap().to_string_lossy().into_owned(),
            ty,
            code_address: meta.get("code_address").and_then(Json::u64).unwrap(),
            bytes: std::fs::read(&p).unwrap(),
            meta,
        });
    }
    (shaders, fused)
}

fn sh_register(meta: &Json, offset: u32) -> Option<u32> {
    meta.get("sh_registers")?.array().iter().rev().find_map(|r| {
        let a = r.array();
        (a.first()?.u64()? == offset as u64).then(|| a.get(1)?.u64().map(|v| v as u32)).flatten()
    })
}

fn user_data_layout(meta: &Json) -> ShaderUserDataLayout {
    let ud = meta.get("user_data").unwrap();
    let list = |j: Option<&Json>| -> Vec<u16> { j.map_or(vec![], |j| j.array().iter().filter_map(|v| v.u64()).map(|v| v as u16).collect()) };
    let direct = list(ud.get("direct_resource_offsets"));
    let sharps: Vec<Vec<u16>> = ud.get("sharp_resource_offsets").map_or(vec![], |s| s.array().iter().map(|a| list(Some(a))).collect());
    let empty = Vec::new();
    let sharp = |i: usize| sharps.get(i).unwrap_or(&empty).as_slice();
    ShaderUserDataLayout::new(
        &direct,
        [sharp(0), sharp(1), sharp(2), sharp(3)],
        ud.get("eud_size_dw").and_then(Json::u64).unwrap_or(0) as u32,
        ud.get("srt_size_dw").and_then(Json::u64).unwrap_or(0) as u32,
    )
}

struct StageParameters {
    stage: ShaderStage,
    base: u32,
    count: u32,
    wave_size: u32,
    layout: ShaderUserDataLayout,
}

fn parameters(s: &Shader) -> StageParameters {
    let rsrc2 = s.ty.rsrc2_register().and_then(|r| sh_register(&s.meta, r)).unwrap_or(0);
    let count = metadata::user_sgpr_count(s.ty, rsrc2);
    let range = s.meta.get("specials").and_then(|sp| sp.get("user_data_range")).map(|r| r.array().to_vec()).unwrap_or_default();
    if let (Some(start), Some(end)) = (range.first().and_then(Json::u64), range.get(1).and_then(Json::u64)) {
        assert_eq!(end - start, count as u64, "{}: PGM_RSRC2.USER_SGPR disagrees with the user-data range", s.hash);
    }
    let modifier = s.meta.get("specials").and_then(|sp| sp.get("dispatch_modifier")).and_then(Json::u64).unwrap_or(0) as u32;
    StageParameters { stage: s.ty.stage(), base: s.ty.user_data_base(), count, wave_size: s.ty.wave_size(modifier), layout: user_data_layout(&s.meta) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Outcome {
    /// Every descriptor comes from user data or constants.
    Static,
    /// Some descriptor or flattened read walks guest memory per draw.
    SrtWalk,
    /// Some image or V# is selected at run time from host-enumerated
    /// candidates.
    Indirect,
    /// Some access uses the address-based fallback.
    AddressBased,
    Failed,
}

#[derive(Default)]
struct Report {
    outcomes: BTreeMap<(String, Outcome), usize>,
    failures: BTreeMap<String, Vec<String>>,
    operands: BTreeMap<SourceCategory, usize>,
    bindings: BTreeMap<String, usize>,
    sharp_matches: BTreeMap<String, usize>,
    vertex_fetch: usize,
    table_reads: usize,
    materialized: usize,
    materialize_errors: BTreeMap<String, usize>,
    plans: usize,
    /// Plans the replay path covers, and draws it reproduced.
    replay_eligible: usize,
    replay_draws: usize,
    replay_patched: u64,
}

fn failure_bucket(message: &str) -> String {
    let m = message.split(" pc=0x").nth(1).map(|r| r.split_once(' ').map_or(r, |x| x.1)).unwrap_or(message);
    let m = m.split(" (undefined from").next().unwrap();
    let m = m.split(" (value:").next().unwrap();
    m.chars().take(90).collect()
}

/// Check plan descriptors that sit at a fixed user-data location against
/// the AGC sharp tables: a user-data T#/V#/S# at SGPR r is flat dword
/// r - base; one read through the EUD pointer at byte offset o is flat dword
/// 16 + o / 4.
fn check_sharp_tables(plan: &ShaderResourcePlan, p: &StageParameters, report: &mut Report) {
    let eud = p.layout.direct(DirectResource::PtrExtendedUserData).map(|r| r + p.base);
    let location = |dword: ValueId| -> Option<u32> {
        let mut v = dword;
        let n = plan.graph.node(v);
        if n.kind == ValueKind::ResourceTableWord {
            v = plan.table_reads.get(n.payload as usize)?.value;
        }
        let n = plan.graph.node(v);
        match n.kind {
            ValueKind::UserData => Some(n.payload as u32 - p.base),
            ValueKind::ScalarAddressWord => {
                let handle = plan.graph.node(n.operands[0]);
                let low = plan.graph.node(handle.operands[0]);
                if low.kind != ValueKind::UserData || Some(low.payload as u32) != eud || !plan.graph.is_constant(n.operands[1]) {
                    return None;
                }
                let offset = plan.memory().get(n.memory_index()).offset.wrapping_add(plan.graph.constant_u32(n.operands[1]).unwrap());
                Some(metadata::USER_SGPR_DWORDS + offset / 4)
            }
            _ => None,
        }
    };
    let mut check = |kind: &str, source: u32| {
        let s = &plan.descriptor_sources[source as usize];
        if s.indirect_image.is_some() {
            return;
        }
        let Some(first) = location(s.dwords[0]) else {
            *report.sharp_matches.entry(format!("{kind}: not at a fixed location")).or_insert(0) += 1;
            return;
        };
        let class = p.layout.sharp_class_at(first);
        *report.sharp_matches.entry(format!("{kind}: sharp class {class:?}")).or_insert(0) += 1;
    };
    for b in &plan.info.buffers {
        check("buffer", b.source);
    }
    for i in &plan.info.images {
        check("image", i.source);
    }
    for s in &plan.info.samplers {
        check("sampler", s.source);
    }
}

fn plan_shader(program: Program, s: &Shader, report: &mut Report) {
    let p = parameters(s);
    let label = format!("{:?}", s.ty);
    let mut program = program;
    let mut fixed = None;
    let (attributes, buffers) = (p.layout.direct(DirectResource::PtrVertexAttribDescTable), p.layout.direct(DirectResource::PtrVertexBufferTable));
    if p.stage == ShaderStage::Vertex {
        if let (Some(a), Some(b)) = (attributes, buffers) {
            let fetch = vertex_fetch::detect(&program, (p.base + a) as i32, (p.base + b) as i32, p.base, p.count, 32);
            if !fetch.loads.is_empty() {
                report.vertex_fetch += 1;
                program = fetch.remove_replaced_table_loads(&program);
                fixed = Some(fetch.load_pcs());
            }
        }
    }
    let hash = u64::from_str_radix(&s.hash, 16).unwrap_or(0);
    report.plans += 1;
    let plan = match ShaderResourcePlan::extract(program, p.stage, hash, p.base, p.count, fixed.as_ref(), p.wave_size) {
        Ok(plan) => plan,
        Err(e) => {
            *report.outcomes.entry((label, Outcome::Failed)).or_insert(0) += 1;
            report.failures.entry(failure_bucket(&e.message)).or_default().push(s.hash.clone());
            return;
        }
    };
    let uses = plan.resource_uses();
    let mut outcome = Outcome::Static;
    for u in &uses {
        *report.operands.entry(u.category).or_insert(0) += 1;
        let binding = format!("{:?}", u.binding).split('(').next().unwrap().to_string();
        *report.bindings.entry(binding).or_insert(0) += 1;
        outcome = outcome.max(match u.category {
            SourceCategory::Constant | SourceCategory::UserData | SourceCategory::None => Outcome::Static,
            SourceCategory::SrtWalk => Outcome::SrtWalk,
            SourceCategory::Indirect => Outcome::Indirect,
            SourceCategory::AddressBased => Outcome::AddressBased,
        });
        // Every use is keyed by an instruction of the program.
        assert!(plan.program().index_of_pc(u.pc).is_some());
    }
    if !plan.table_reads.is_empty() {
        report.table_reads += 1;
        outcome = outcome.max(Outcome::SrtWalk);
    }
    *report.outcomes.entry((label, outcome)).or_insert(0) += 1;
    check_sharp_tables(&plan, &p, report);
    assert_eq!(plan.materialization_sources.len() + plan.info.images.iter().filter(|i| plan.descriptor_sources[i.source as usize].indirect_image.is_some()).count(),
        plan.info.buffers.len() + plan.info.images.len() + plan.info.samplers.len());

    // Default layout for the unspecialised plan (narrow offsets; the wide
    // variant only widens the slots and re-keys).
    let specialization = ResourceSpecialization::default_for(&plan.info);
    let specialized = apply_specialization(&plan, &specialization).unwrap();
    let registers = BindingLayout::collect_user_data_registers(plan.program(), p.base, p.count);
    assert!(registers.iter().all(|&r| r >= p.base && r < p.base + p.count));
    let flattened = ShaderCompileRequest::requires_flattened_table(&plan, &specialized);
    let allocate = |wide: bool| {
        BindingLayout::allocate(&specialized.info, registers.clone(), BindingLayout::uses_global_data_share(plan.program()), flattened, BindingLayout::reads_shader_base(plan.program()), 0, false, wide)
    };
    if let Ok(layout) = allocate(false) {
        layout
            .validate(&specialized.info, &registers, BindingLayout::uses_global_data_share(plan.program()), flattened, BindingLayout::reads_shader_base(plan.program()), plan.hash, plan.stage)
            .unwrap();
        let _ = ShaderCompileRequest::new(&plan, specialized.clone(), layout);
    }
    if let (Ok(narrow), Ok(wide)) = (allocate(false), allocate(true)) {
        assert!(narrow.memory_offset_dwords() <= wide.memory_offset_dwords());
        assert_eq!(narrow.memory_offset_slot(0).0, wide.memory_offset_slot(0).0);
        assert_eq!(wide.memory_offset_slot(0), (wide.memory_offset_dword, 0, 32));
        assert_eq!(narrow.memory_offset_slot(5), (narrow.memory_offset_dword + 1, 8, 8));
        assert_eq!(wide.dispatch_thread_limits_dword(), wide.memory_offset_dword + wide.memory_offset_count);
    }

    // Materialisation smoke test: zeroed user data and memory.
    let user_data = vec![0u32; p.count as usize];
    let zero = |_: u64| Some(0u32);
    let mut inputs = ResourceRuntimeInputs::new(&user_data, s.code_address);
    inputs.read_memory = Some(&zero);
    inputs.read_clean_memory = Some(&zero);
    match materialize(&plan, &inputs) {
        Ok((snapshot, spec)) => {
            report.materialized += 1;
            assert_eq!(snapshot.buffers.len(), spec.buffers.len());
            assert_eq!(snapshot.images.len(), spec.images.len());
            assert!(snapshot.flattened_resource_table.len() >= plan.flattened_table_reserved_count());
            apply_specialization(&plan, &spec).unwrap();
        }
        Err(e) => {
            let text = e.to_string();
            let bucket: String = text.split(':').take(2).collect::<Vec<_>>().join(":").chars().take(70).collect();
            *report.materialize_errors.entry(bucket).or_insert(0) += 1;
        }
    }
    check_replay(&plan, &p, s, report);
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

/// The cached materialisation (replay templates for eligible plans) gives
/// exactly what a fresh materialisation gives, over draws whose user data
/// and memory vary: random words, pointer-like words, and memory that
/// repeats one valid T# / V# pattern so that signatures match and the
/// patch path runs.
fn check_replay(plan: &ShaderResourcePlan, p: &StageParameters, s: &Shader, report: &mut Report) {
    if !ReplayProgram::eligible(plan) {
        return;
    }
    report.replay_eligible += 1;
    let mut cache = ResourceMaterializationCache::default();
    // A 2D RGBA8 T# / a plain V# pattern (8 dwords, the V# in the low 4).
    let pattern = |a: u64| -> u32 {
        match (a / 4) % 8 {
            0 => 0x0010_0000,
            1 => 56 << 20,
            2 => 0x00ff_00ff,
            3 => (9 << 28) | 0xFAC,
            _ => 0,
        }
    };
    for round in 0..12u64 {
        let kind = round % 3;
        let user_data: Vec<u32> = (0..p.count as u64)
            .map(|i| {
                let r = mix(round * 1000 + i + 17);
                match kind {
                    0 => r as u32,
                    // Pointer-like: a 4 GiB-aligned base with a varying low dword.
                    1 => if i % 2 == 1 { 2 } else { (r as u32) & !0xFF },
                    _ => if i % 2 == 1 { 2 } else { ((r as u32) & 0xFFF0) | 0x1_0000 },
                }
            })
            .collect();
        let memory = move |a: u64| -> Option<u32> {
            Some(match kind {
                0 => mix(a ^ 0x5555) as u32,
                1 => if a % 64 < 16 { mix(a / 64) as u32 & 0xFFFF_FF00 } else { pattern(a) },
                _ => pattern(a),
            })
        };
        let resident = move |a: u64, out: &mut [u8], _clean: bool| -> bool {
            for (i, chunk) in out.chunks_mut(4).enumerate() {
                chunk.copy_from_slice(&memory(a + i as u64 * 4).unwrap().to_le_bytes());
            }
            true
        };
        let mut inputs = ResourceRuntimeInputs::new(&user_data, s.code_address);
        inputs.read_memory = Some(&memory);
        inputs.read_clean_memory = Some(&memory);
        inputs.read_resident_memory = Some(&resident);
        let fresh = materialize(plan, &inputs);
        let cached = cache.materialize(plan, &inputs, &resident);
        match (fresh, cached) {
            (Ok((a, sa)), Ok((b, sb))) => {
                assert_eq!(a, *b, "{}: replayed snapshot differs (round {round})", s.hash);
                assert_eq!(sa, *sb, "{}: replayed specialization differs (round {round})", s.hash);
                report.replay_draws += 1;
            }
            (Err(_), Err(_)) => {}
            (a, b) => panic!("{}: fresh {:?} vs cached {:?} (round {round})", s.hash, a.map(|_| ()), b.map(|_| ())),
        }
    }
    report.replay_patched += cache.statistics().replay_patched;
}

fn run(dir: &Path) -> Report {
    let (shaders, fused) = load(dir);
    let by_hash: HashMap<&str, &Shader> = shaders.iter().map(|s| (s.hash.as_str(), s)).collect();
    let mut report = Report::default();
    for s in &shaders {
        let program = decode_blob(&s.bytes, s.code_address).unwrap_or_else(|e| panic!("{}: {e}", s.hash));
        plan_shader(program, s, &mut report);
    }
    // Fused GS halves are planned as one program with the front half's
    // metadata.
    for f in &fused {
        let (Some(front), Some(back)) = (f.get("front").and_then(Json::str), f.get("back").and_then(Json::str)) else { continue };
        let (Some(front), Some(back)) = (by_hash.get(front), by_hash.get(back)) else { continue };
        let code = [CodeSlice::new(front.code_address, &front.bytes), CodeSlice::new(back.code_address, &back.bytes)];
        let layout = FusedLayout {
            entry_address: front.code_address,
            entry_size: front.bytes.len() as u32,
            continuation_address: back.code_address,
            continuation_size: back.bytes.len() as u32,
        };
        let program = decode_fused_program(&code[..], layout).unwrap();
        let fused_shader = Shader { hash: format!("fused-{}", front.hash), ty: AgcShaderType::Geometry, code_address: front.code_address, bytes: vec![], meta: front.meta.clone() };
        plan_shader(program, &fused_shader, &mut report);
    }
    report
}

fn print(report: &Report, name: &str) {
    eprintln!("== resource plans: {name}: {} programs", report.plans);
    let mut totals: BTreeMap<Outcome, usize> = BTreeMap::new();
    for ((stage, outcome), n) in &report.outcomes {
        eprintln!("  {stage:<14} {outcome:?}: {n}");
        *totals.entry(*outcome).or_insert(0) += n;
    }
    eprintln!("  totals: {totals:?}");
    eprintln!("  operands by category: {:?}", report.operands);
    eprintln!("  operands by binding: {:?}", report.bindings);
    eprintln!("  plans with flattened reads: {}, with fixed-function vertex fetch: {}", report.table_reads, report.vertex_fetch);
    eprintln!("  sharp-table locations: {:?}", report.sharp_matches);
    eprintln!("  materialised (zero memory): {} ok, errors {:?}", report.materialized, report.materialize_errors);
    eprintln!("  replay: {} eligible plans, {} draws equal to fresh materialisation ({} patched)", report.replay_eligible, report.replay_draws, report.replay_patched);
    for (bucket, hashes) in &report.failures {
        eprintln!("  FAIL x{}: {bucket} (e.g. {})", hashes.len(), hashes.iter().take(3).cloned().collect::<Vec<_>>().join(", "));
    }
}

#[test]
fn sample_corpus_plans() {
    let report = run(&sample_dir());
    print(&report, "sample");
    let failed: usize = report.outcomes.iter().filter(|((_, o), _)| *o == Outcome::Failed).map(|(_, n)| n).sum();
    assert_eq!(failed, 0, "{:?}", report.failures);
    assert!(report.plans >= 14);
}

/// Full capture (opt-in): `PS5SHADER_CORPUS=/path cargo test -p ps5shader --test resources -- --nocapture`.
#[test]
fn full_corpus_plans() {
    let Some(dir) = std::env::var_os("PS5SHADER_CORPUS") else { return };
    let report = run(Path::new(&dir));
    print(&report, &dir.to_string_lossy());
}
