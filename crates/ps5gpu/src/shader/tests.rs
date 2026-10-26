// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Offline compilation of captured game shaders: every program of the
//! sample corpus (and of `PS5SHADER_CORPUS=<dir>` when set) is planned,
//! specialised with default resources, compiled and checked with
//! `spirv-val` when it is installed.

use super::*;
use ps5shader::resources::{apply_specialization, BindingLayout, ResourceSpecialization, ShaderResourcePlan};
use ps5shader::{decode_blob, ColorComponentMapping, ComputeSystemRegisters, PixelOutputBinding};
use std::path::{Path, PathBuf};

fn json_u64(text: &str, key: &str) -> Option<u64> {
    let at = text.find(&format!("\"{key}\":"))? + key.len() + 3;
    let rest = text[at..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn json_str<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let at = text.find(&format!("\"{key}\":"))? + key.len() + 3;
    let rest = text[at..].trim_start().strip_prefix('"')?;
    Some(&rest[..rest.find('"')?])
}

fn user_range(text: &str) -> (u32, u32) {
    let Some(at) = text.find("\"user_data_range\": [") else { return (0, 16) };
    let rest = &text[at + 20..];
    let end = rest.find(']').unwrap_or(0);
    let v: Vec<u32> = rest[..end].split(',').filter_map(|s| s.trim().parse().ok()).collect();
    (v.first().copied().unwrap_or(0), v.get(1).copied().unwrap_or(16))
}

fn spirv_val() -> Option<PathBuf> {
    ["/usr/bin/spirv-val", "/usr/local/bin/spirv-val"].iter().map(PathBuf::from).find(|p| p.exists())
}

/// Compile one blob; Err(message) on any failure (translation, emission
/// or validation).
fn compile_one(path: &Path, validator: Option<&Path>) -> Result<usize, String> {
    compile_ext(path, validator, |_, _| true, None).map(|m| m.map_or(0, |m| m.spirv.len()))
}

/// [`compile_one`] with the default specialization adjusted by `adjust`
/// (which returns false to skip the shader).
fn compile_with(path: &Path, validator: Option<&Path>, adjust: impl Fn(&mut ResourceSpecialization, &ps5shader::resources::ShaderResourceInfo) -> bool) -> Result<(), String> {
    compile_ext(path, validator, adjust, None).map(|_| ())
}

/// [`compile_with`] returning the module; `mesh`: compile a vertex-stage
/// program as a primitive subgroup (mesh workgroup).
fn compile_ext(
    path: &Path,
    validator: Option<&Path>,
    adjust: impl Fn(&mut ResourceSpecialization, &ps5shader::resources::ShaderResourceInfo) -> bool,
    mesh: Option<crate::pipeline::compiler::MeshConfig>,
) -> Result<Option<crate::pipeline::compiler::CompiledShader>, String> {
    // PS5SHADER_EMIT_BEFORE=1: the code emitted before the GPU-cost
    // switches (for size comparisons).
    let emit = if std::env::var_os("PS5SHADER_EMIT_BEFORE").is_some() { super::EmitOptions::BEFORE } else { super::EmitOptions::ALL };
    compile_emit(path, validator, adjust, mesh, emit)
}

/// [`compile_ext`] with the emitter switches `emit`.
fn compile_emit(
    path: &Path,
    validator: Option<&Path>,
    adjust: impl Fn(&mut ResourceSpecialization, &ps5shader::resources::ShaderResourceInfo) -> bool,
    mesh: Option<crate::pipeline::compiler::MeshConfig>,
    emit: super::EmitOptions,
) -> Result<Option<crate::pipeline::compiler::CompiledShader>, String> {
    let meta = std::fs::read_to_string(path.with_extension("json")).unwrap_or_default();
    let stage = match json_str(&meta, "stage") {
        Some("cs") => ShaderStage::Compute,
        Some("ps") => ShaderStage::Pixel,
        Some("gs") | Some("vs") | Some("es") => ShaderStage::Vertex,
        _ => return Ok(None),
    };
    let addr = json_u64(&meta, "code_address").unwrap_or(0x1000);
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let program = decode_blob(&bytes, addr).map_err(|e| format!("decode: {e}"))?;
    let (_, count) = user_range(&meta);
    let (base, count) = if mesh.is_some() {
        (0, 8 + count.min(32))
    } else if stage == ShaderStage::Vertex && json_str(&meta, "stage") == Some("gs") {
        (8, count.min(32))
    } else {
        (0, count.min(32))
    };
    let wave = if stage == ShaderStage::Compute || mesh.is_some() { 64 } else { 32 };
    let user = BindingLayout::collect_user_data_registers(&program, base, count);
    let gds = BindingLayout::uses_global_data_share(&program);
    let shader_base = BindingLayout::reads_shader_base(&program);
    let plan = ShaderResourcePlan::extract(program, stage, 1, base, count, None, wave).map_err(|e| format!("plan: {e}"))?;
    let mut spec = ResourceSpecialization::default_for(&plan.info);
    for b in &mut spec.buffers {
        b.descriptor_format = 77;
    }
    if !adjust(&mut spec, &plan.info) {
        return Ok(None);
    }
    let resources = apply_specialization(&plan, &spec).map_err(|e| format!("specialize: {e}"))?;
    let flat = ShaderCompileRequest::requires_flattened_table(&plan, &resources);
    let mut layout = BindingLayout::allocate(&resources.info, user, gds, flat, shader_base, 0, false, false).map_err(|e| format!("layout: {e}"))?;
    if mesh.is_some() {
        // The entry point reads the index buffer and its draw words.
        use ps5shader::resources::{DescriptorBinding, DescriptorBindingKind as K};
        for kind in [K::DeviceAddressPageTable, K::FaultBuffer, K::ShaderData] {
            if layout.find(kind).is_none() {
                layout.descriptors.push(DescriptorBinding { kind, resources: vec![] });
            }
        }
    }
    let mut r = ShaderCompileRequest::new(&plan, resources, layout);
    r.wave_size = wave;
    match stage {
        ShaderStage::Compute => {
            r.local_size = [64, 1, 1];
            r.compute_system_registers =
                Some(ComputeSystemRegisters { work_group_x: Some(count), work_group_y: Some(count + 1), work_group_z: Some(count + 2), thread_group_size: None });
        }
        ShaderStage::Pixel => {
            r.pixel_outputs = (0..8).map(|i| PixelOutputBinding { component_mapping: ColorComponentMapping::IDENTITY, ..PixelOutputBinding::new(i, i, PixelOutputKind::Float) }).collect();
            r.pixel_input_enable = 0x2;
            r.pixel_input_address = 0x2;
            r.color_export_format = 0x4444_4444;
            r.pixel_input_cntl = Some((0..32).collect());
        }
        _ => {}
    }
    if let Some(m) = &mesh {
        r.local_size = [m.threads, 1, 1];
        r.local_data_share_dwords = 2304;
    }
    let compiler = SpirvCompiler::new(Features { barycentric: true, demote: true, subgroup_size: 32, subgroup64_compute: false, max_compute_workgroup_subgroups: 32, max_shared_bytes: 49152, sample_offset: false, image_int64_atomics: true, shared_int64_atomics: true, compute_derivatives: true, compute_derivatives_linear: true, preserve_inf_nan: [true; 3], denorm_preserve: [true; 3], denorm_flush_to_zero32: true, f16_rte: false, device_clock: Some(crate::clock::DeviceClock { num: 1, den: 10 }), subgroup_clock: true, bindless_tables: true, no_float64: false, amd_proprietary: false, emit }, None);
    let out = compiler.compile_mesh_request(stage, &r, 0, mesh).map_err(|e| format!("compile: {e}"))?;
    // PS5SHADER_DUMP_SPV=<dir>: write every module for inspection.
    if let Some(dir) = std::env::var_os("PS5SHADER_DUMP_SPV") {
        let bytes: Vec<u8> = out.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
        let _ = std::fs::write(Path::new(&dir).join(path.file_stem().unwrap()).with_extension("spv"), bytes);
    }
    if let Some(val) = validator {
        // Tests validate the same corpus in parallel: keep file names unique.
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = std::env::temp_dir().join(format!(
            "rps5e-spv-{}-{seq}-{}.spv",
            std::process::id(),
            path.file_stem().unwrap().to_string_lossy()
        ));
        let bytes: Vec<u8> = out.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&tmp, bytes).map_err(|e| e.to_string())?;
        let o = std::process::Command::new(val).arg("--target-env").arg("vulkan1.3").arg(&tmp).output().map_err(|e| e.to_string())?;
        if !o.status.success() {
            return Err(format!("spirv-val ({}): {}", tmp.display(), String::from_utf8_lossy(&o.stderr).lines().take(4).collect::<Vec<_>>().join(" | ")));
        }
        let _ = std::fs::remove_file(&tmp);
    }
    Ok(Some(out))
}

/// Compiled count, failures and the total SPIR-V words of a directory.
fn run_dir(dir: &Path) -> (usize, Vec<(String, String)>) {
    let (ok, failures, _) = run_dir_sized(dir);
    (ok, failures)
}

fn run_dir_sized(dir: &Path) -> (usize, Vec<(String, String)>, usize) {
    // PS5SHADER_NOVAL=1 skips spirv-val (to time the compiler alone).
    let validator = if std::env::var_os("PS5SHADER_NOVAL").is_some() { None } else { spirv_val() };
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    let mut ok = 0;
    let mut words = 0;
    let mut failures = Vec::new();
    let mut slowest: Vec<(u128, String)> = Vec::new();
    for p in entries {
        if p.extension().is_none_or(|x| x != "bin") {
            continue;
        }
        let t = std::time::Instant::now();
        let r = compile_one(&p, validator.as_deref());
        slowest.push((t.elapsed().as_micros(), p.file_name().unwrap().to_string_lossy().into_owned()));
        match r {
            Ok(n) => {
                ok += 1;
                words += n;
            }
            Err(m) => failures.push((p.file_name().unwrap().to_string_lossy().into_owned(), m)),
        }
    }
    if std::env::var_os("PS5SHADER_SLOWEST").is_some() {
        slowest.sort_unstable_by(|a, b| b.cmp(a));
        for (us, n) in slowest.iter().take(15) {
            eprintln!("slow: {n} {} ms", us / 1000);
        }
    }
    (ok, failures, words)
}

#[test]
fn sample_corpus_compiles() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../ps5shader/tests/corpus");
    let (ok, failures) = run_dir(&dir);
    for (n, m) in &failures {
        eprintln!("{n}: {m}");
    }
    eprintln!("compiled {ok}, failed {}", failures.len());
    assert!(failures.is_empty(), "{} corpus shaders failed", failures.len());
}

/// The same request gives the same SPIR-V words (ids are allocated in
/// program order, never in the order of a hash map with a per-instance
/// seed), so module keys and disk-cache contents are stable across runs.
#[test]
fn corpus_modules_are_deterministic() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../ps5shader/tests/corpus");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    let mut checked = 0;
    for p in entries.iter().filter(|p| p.extension().is_some_and(|x| x == "bin")) {
        let Ok(Some(a)) = compile_emit(p, None, |_, _| true, None, super::EmitOptions::ALL) else { continue };
        let b = compile_emit(p, None, |_, _| true, None, super::EmitOptions::ALL).unwrap().unwrap();
        assert!(a.spirv == b.spirv, "{}: two compiles differ", p.display());
        checked += 1;
    }
    assert!(checked > 0);
}

/// `PS5SHADER_CORPUS=<dir> cargo test -p ps5gpu --release full_corpus -- --ignored --nocapture`
#[test]
#[ignore]
fn full_corpus() {
    let Some(dir) = std::env::var_os("PS5SHADER_CORPUS") else { return };
    let started = std::time::Instant::now();
    let (ok, failures, words) = run_dir_sized(Path::new(&dir));
    eprintln!("SPIR-V words {words} ({} KiB), {:.1} s", words * 4 / 1024, started.elapsed().as_secs_f64());
    let mut reasons: std::collections::BTreeMap<String, usize> = Default::default();
    for (n, m) in &failures {
        eprintln!("{n}: {m}");
        let key: String = m.chars().take(70).collect();
        *reasons.entry(key).or_default() += 1;
    }
    let mut r: Vec<_> = reasons.into_iter().collect();
    r.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, n) in r.iter().take(30) {
        eprintln!("{n:5}  {k}");
    }
    eprintln!("compiled {ok}, failed {}", failures.len());
    eprintln!("phases (ms): {}", super::stats::phases_text());
    eprintln!("passes (ms): {}", super::stats::passes_text());
}

/// Every sampled image of the corpus shaders specialised as a cube (array):
/// sampled through directions, the emitted modules validate.
#[test]
fn sample_corpus_compiles_as_cubes() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../ps5shader/tests/corpus");
    let validator = spirv_val();
    let mut cubes = 0;
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries.iter().filter(|p| p.extension().is_some_and(|x| x == "bin")) {
        let found = std::cell::Cell::new(false);
        let r = compile_with(p, validator.as_deref(), |spec, info| {
            for (i, image) in info.images.iter().enumerate() {
                if image.resource_class == ps5shader::resources::ImageResourceClass::Sampled && !image.fetched {
                    if let Some(s) = spec.images.get_mut(i) {
                        s.cube = true;
                        s.dimension = ps5shader::resources::ImageDimension::Dim2DArray;
                        found.set(true);
                    }
                }
            }
            found.get()
        });
        if found.get() {
            cubes += 1;
        }
        assert!(r.is_ok(), "{}: {}", p.display(), r.unwrap_err());
    }
    eprintln!("compiled {cubes} shaders with cube images");
    assert!(cubes > 0);
}

fn corpus(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../ps5shader/tests/corpus").join(name)
}

/// A pixel program spilling through ADDTID LDS (slots 256 bytes apart,
/// lane * 4 within a slot) keeps one private dword per slot it touches.
#[test]
fn addtid_spills_are_lane_private() {
    let out = compile_ext(&corpus("0fe6027d4cc5f9fc.bin"), spirv_val().as_deref(), |_, _| true, None).unwrap().unwrap();
    let bytes = out.reflection.private_lds_bytes;
    assert!(bytes > 0 && bytes <= 64 * 4, "{bytes} bytes of private LDS");
}

/// A primitive subgroup without a geometry stage that culls and compacts
/// its vertices through LDS compiles as a mesh workgroup with shared LDS.
#[test]
fn culling_vertex_program_as_mesh() {
    let mesh = crate::pipeline::compiler::MeshConfig {
        lines: false,
        points: false,
        input_primitive: 4,
        primitives_per_group: 21,
        vertices_per_group: 63,
        max_vertices: 63,
        max_primitives: 21,
        threads: 64,
        esgs_item_size: 4,
        draw_dword: 64,
    };
    let out = compile_ext(&corpus("13a5b2561f07d0b4.bin"), spirv_val().as_deref(), |_, _| true, Some(mesh)).unwrap().unwrap();
    assert_eq!(out.reflection.private_lds_bytes, 0);
    assert_eq!(out.reflection.local_size, [32, 1, 1]);
}

#[test]
fn float_rules_switch_parses_masks() {
    use ps5shader::translate::float_rules::{ALL, DIV_FMAS_NAN, F16_DENORMALS};
    assert_eq!(super::parse_float_rules(None), ALL);
    assert_eq!(super::parse_float_rules(Some("")), ALL);
    assert_eq!(super::parse_float_rules(Some("0")), 0);
    assert_eq!(super::parse_float_rules(Some("OFF")), 0);
    assert_eq!(super::parse_float_rules(Some("4")), DIV_FMAS_NAN);
    assert_eq!(super::parse_float_rules(Some("0xC")), DIV_FMAS_NAN | F16_DENORMALS);
    assert_eq!(super::parse_float_rules(Some("0xFFFF")), ALL);
    assert_eq!(super::parse_float_rules(Some("junk")), ALL);
}

/// Compute modules whose buffer accesses all stay in range by their own
/// checks say so (their pipelines may run without the driver's buffer
/// robustness), only with guarded loads; the code emitted before the
/// switches never does; and every module validates both ways.
#[test]
fn compute_modules_report_their_own_buffer_checks() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../ps5shader/tests/corpus");
    let validator = spirv_val();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    let (mut checked, mut compute) = (0, 0);
    for p in entries.iter().filter(|p| p.extension().is_some_and(|x| x == "bin")) {
        let meta = std::fs::read_to_string(p.with_extension("json")).unwrap_or_default();
        if json_str(&meta, "stage") != Some("cs") {
            continue;
        }
        let new = compile_emit(p, validator.as_deref(), |_, _| true, None, super::EmitOptions::ALL).unwrap().unwrap();
        let unguarded = compile_emit(p, validator.as_deref(), |_, _| true, None, super::EmitOptions { guarded_loads: false, ..super::EmitOptions::ALL }).unwrap().unwrap();
        let before = compile_emit(p, validator.as_deref(), |_, _| true, None, super::EmitOptions::BEFORE).unwrap().unwrap();
        compute += 1;
        checked += new.reflection.own_buffer_checks as usize;
        assert!(!unguarded.reflection.own_buffer_checks && !before.reflection.own_buffer_checks, "{}", p.display());
    }
    eprintln!("{checked} of {compute} compute modules check their own buffer ranges");
    assert!(compute > 0 && checked > 0);
}

/// Paired wave64 compute modules (two guest lanes per invocation) load each
/// scalar memory read once for both lane halves: never larger than the
/// per-half loads, smaller where a program has scalar loads.
#[test]
fn paired_wave64_scalar_loads_are_shared_by_the_halves() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../ps5shader/tests/corpus");
    let validator = spirv_val();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    let (mut compute, mut smaller) = (0, 0);
    for p in entries.iter().filter(|p| p.extension().is_some_and(|x| x == "bin")) {
        let meta = std::fs::read_to_string(p.with_extension("json")).unwrap_or_default();
        if json_str(&meta, "stage") != Some("cs") {
            continue;
        }
        let shared = compile_emit(p, validator.as_deref(), |_, _| true, None, super::EmitOptions::ALL).unwrap().unwrap();
        let per_half = compile_emit(p, validator.as_deref(), |_, _| true, None, super::EmitOptions { smem_once: false, ..super::EmitOptions::ALL }).unwrap().unwrap();
        compute += 1;
        assert!(shared.spirv.len() <= per_half.spirv.len(), "{}: {} > {}", p.display(), shared.spirv.len(), per_half.spirv.len());
        smaller += (shared.spirv.len() < per_half.spirv.len()) as usize;
    }
    eprintln!("{smaller} of {compute} compute modules shrink");
    assert!(compute > 0 && smaller > 0);
}

/// `RPS5E_TRANSLATE_BENCH=<list.prewarm>` (keys: `RPS5E_TRANSLATE_BENCH_KEYS`
/// = hex,hex): translate the modules of a shader prewarm list without a
/// device (a 32-wide subgroup device's features) on one thread and print
/// where each one's time went. A workload for profiling the translator.
#[test]
#[ignore]
fn translate_list_bench() {
    let Some(list) = std::env::var_os("RPS5E_TRANSLATE_BENCH") else { return };
    let keys: Vec<u64> = std::env::var("RPS5E_TRANSLATE_BENCH_KEYS").unwrap_or_default().split(',').filter_map(|k| u64::from_str_radix(k.trim(), 16).ok()).collect();
    let emit = super::EmitOptions::ALL;
    let compiler = SpirvCompiler::new(Features { barycentric: true, demote: true, subgroup_size: 32, subgroup64_compute: false, max_compute_workgroup_subgroups: 32, max_shared_bytes: 49152, sample_offset: false, image_int64_atomics: true, shared_int64_atomics: true, compute_derivatives: true, compute_derivatives_linear: true, preserve_inf_nan: [true; 3], denorm_preserve: [true; 3], denorm_flush_to_zero32: true, f16_rte: false, device_clock: Some(crate::clock::DeviceClock { num: 1, den: 10 }), subgroup_clock: true, bindless_tables: true, no_float64: false, amd_proprietary: false, emit }, None);
    let only = |k: u64| keys.is_empty() || keys.contains(&k);
    let modules = crate::pipeline::precompile::translate_list_with(&compiler, Path::new(&list), 1, &only, None).unwrap();
    for m in &modules {
        let parts: Vec<String> = super::stats::LAST_NAMES.iter().zip(m.parts).map(|(n, v)| format!("{n} {:.1}", v as f64 / 1e6)).collect();
        eprintln!("{:016x}: {:.1} ms, {} words, hash {:016x} [{}]", m.key, m.ns as f64 / 1e6, m.words, m.hash, parts.join(" "));
    }
}
