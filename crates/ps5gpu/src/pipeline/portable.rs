// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Portable shader caches: a title's prewarm list without anything tied
//! to this machine, so players can hand each other a game's shaders and
//! skip the first-run compiles (`rps5e-cli shader-cache export|import`,
//! the launcher's "Import shader cache…").
//!
//! The file keeps what the translator works from: every program's guest
//! code words, the request each module was translated with, the code words
//! the translation read, and every pipeline as its module keys, binding
//! layout and fixed-function key. It holds no SPIR-V, no reflection, no
//! driver data (the `VkPipelineCache` file is specific to the GPU and
//! driver and is never shared) and no host paths. An import merges the
//! file into the title's list (union, one record per key): its modules are
//! marked for translation, so they only ever reach the device through the
//! normal path (decode → plan → translate → `vkCreateShaderModule`), on
//! the next start or `precompile`.
//!
//! An imported file is untrusted input. It is read with bounds checks
//! everywhere (no field is trusted to size an allocation beyond the file),
//! must name the title it is imported for and this format version, and
//! every record is checked against the values this emulator writes
//! (stages, wave sizes, descriptor types, formats, blend state, ...)
//! before anything is merged: a file that fails any check is rejected
//! whole.

use super::descriptors::{BindingDesc, BINDLESS_SET_MARKER};
use super::prewarm::{self, ComputeRecord, GraphicsRecord, ModuleRecord, Reader, Writer};
use super::sources::{CodeImage, ProgramSource};
use super::state::{FixedFunctionKey, MAX_COLOR_ATTACHMENTS, MAX_VIEWPORTS};
use super::ShaderReflection;
use std::path::Path;


/// The signature line of a portable file.
const SIGNATURE: &str = "rps5eSC1\n";
const MAGIC: u32 = 0x4353_3550; // "P5SC"
/// The format version (a file of another version is refused).
pub const VERSION: u32 = 1;
/// Files larger than this are refused before they are read.
pub const MAX_FILE_BYTES: u64 = 1 << 30;

/// What an import added.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImportSummary {
    /// Records in the file.
    pub modules: usize,
    pub pipelines: usize,
    /// Records the title's list did not have yet.
    pub new_modules: usize,
    pub new_pipelines: usize,
}

impl std::fmt::Display for ImportSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} modules and {} pipelines in the file, {} modules and {} pipelines new (translated and built on the next start or precompile)",
            self.modules, self.pipelines, self.new_modules, self.new_pipelines
        )
    }
}

/// The portable file of the list at `list_path` for `title`.
pub fn export(list_path: &Path, title: &str) -> Result<Vec<u8>, String> {
    let file = std::fs::read(list_path).map_err(|e| format!("{}: {e}", list_path.display()))?;
    let (_, modules, graphics, compute) = prewarm::decode(&file).ok_or_else(|| format!("{}: not a shader prewarm list of this format", list_path.display()))?;
    Ok(encode(title, modules, graphics, compute))
}

/// Encode `modules` and the pipelines whose modules it holds as a portable
/// file: only modules with guest inputs, without their SPIR-V and
/// reflection.
fn encode(title: &str, modules: Vec<ModuleRecord>, graphics: Vec<GraphicsRecord>, compute: Vec<ComputeRecord>) -> Vec<u8> {
    let modules: Vec<ModuleRecord> = modules
        .into_iter()
        .filter(|m| m.source.is_some())
        .map(|m| ModuleRecord { spirv: Default::default(), reflection: ShaderReflection::default(), fresh: false, ..m })
        .collect();
    let keys: std::collections::HashSet<u64> = modules.iter().map(|m| m.key).collect();
    let graphics: Vec<GraphicsRecord> = graphics.into_iter().filter(|g| keys.contains(&g.vs) && (g.ps == 0 || keys.contains(&g.ps))).collect();
    let compute: Vec<ComputeRecord> = compute.into_iter().filter(|c| keys.contains(&c.cs)).collect();
    let inner = prewarm::encode(prewarm::NO_TRANSLATOR, &modules, &graphics, &compute);
    let mut w = Writer(Vec::new());
    w.u(MAGIC);
    w.u(VERSION);
    w.string(title);
    w.u(inner.len() as u32);
    let mut payload: Vec<u8> = w.0.iter().flat_map(|v| v.to_le_bytes()).collect();
    payload.extend_from_slice(&inner);
    super::cache::encode(SIGNATURE, &payload)
}

/// The records of a portable file for `title`, checked.
type Records = (Vec<ModuleRecord>, Vec<GraphicsRecord>, Vec<ComputeRecord>);

fn decode(file: &[u8], title: &str) -> Result<Records, String> {
    if file.len() as u64 > MAX_FILE_BYTES {
        return Err("the file is too large".into());
    }
    let payload = super::cache::decode(SIGNATURE, file).ok_or("not a rps5e shader cache file (or damaged: bad signature or checksum)")?;
    let head_words = |n: usize| -> Option<Vec<u32>> { Some(payload.get(..n * 4)?.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()) };
    let head = head_words(3).ok_or("truncated header")?;
    if head[0] != MAGIC {
        return Err("not a rps5e shader cache file".into());
    }
    if head[1] != VERSION {
        return Err(format!("shader cache format version {} (this build reads version {VERSION})", head[1]));
    }
    let title_len = head[2] as usize;
    if title_len > 64 {
        return Err("bad title id".into());
    }
    let header = 3 + title_len.div_ceil(4) + 1;
    let words = head_words(header).ok_or("truncated header")?;
    let mut r = Reader(&words[2..]);
    let file_title = r.string().ok_or("bad title id")?;
    let inner_len = r.u().ok_or("truncated header")? as usize;
    let inner = &payload[header * 4..];
    if inner.len() != inner_len {
        return Err("truncated or damaged file".into());
    }
    if file_title != title {
        return Err(format!("the file is the shader cache of {file_title}, not of {title}"));
    }
    let (_, modules, graphics, compute) = prewarm::decode(inner).ok_or("damaged shader records")?;
    let mut keys = std::collections::HashSet::new();
    for m in &modules {
        check_module(m).map_err(|e| format!("module {:016x}: {e}", m.key))?;
        if !keys.insert(m.key) {
            return Err(format!("module {:016x} twice", m.key));
        }
    }
    for g in &graphics {
        if !keys.contains(&g.vs) || (g.ps != 0 && !keys.contains(&g.ps)) {
            return Err("a graphics pipeline names a module the file does not hold".into());
        }
        check_graphics(g)?;
    }
    for c in &compute {
        if !keys.contains(&c.cs) {
            return Err("a compute pipeline names a module the file does not hold".into());
        }
        check_layout(&c.layout)?;
    }
    Ok((modules, graphics, compute))
}

fn check_image(image: &CodeImage) -> Result<(), String> {
    if !image.runs_valid() {
        return Err("bad code image".into());
    }
    Ok(())
}

fn check_program(p: &ProgramSource) -> Result<(), String> {
    if super::sources::stage_of(p.stage).is_none_or(|s| s == ps5shader::resources::ShaderStage::Unknown) {
        return Err("bad stage".into());
    }
    if !matches!(p.wave_size, 32 | 64) {
        return Err("bad wave size".into());
    }
    if p.user_data_count > 64 || p.user_data_base > 0x10000 {
        return Err("bad user data".into());
    }
    if p.address % 4 != 0 || p.address >> 48 != 0 || p.continuation % 4 != 0 || p.continuation >> 48 != 0 {
        return Err("bad program address".into());
    }
    if p.image.read(p.address).is_none() {
        return Err("the code image does not hold the program".into());
    }
    check_image(&p.image)
}

fn check_module(m: &ModuleRecord) -> Result<(), String> {
    let Some(src) = &m.source else { return Err("no guest inputs".into()) };
    if m.fresh || !m.spirv.is_empty() {
        return Err("carries SPIR-V".into());
    }
    if m.stage != src.program.stage {
        return Err("stage mismatch".into());
    }
    if src.request.len() > 1 << 20 {
        return Err("request too large".into());
    }
    check_program(&src.program)?;
    check_image(&src.reads)
}

/// Descriptor types the provider's layouts use.
fn descriptor_type_known(ty: i32) -> bool {
    (0..=10).contains(&ty) || ty == ash::vk::DescriptorType::INLINE_UNIFORM_BLOCK.as_raw() || ty == ash::vk::DescriptorType::ACCELERATION_STRUCTURE_KHR.as_raw()
}

fn check_layout(layout: &[BindingDesc]) -> Result<(), String> {
    use ash::vk::ShaderStageFlags as S;
    let stages = S::VERTEX | S::TESSELLATION_CONTROL | S::TESSELLATION_EVALUATION | S::GEOMETRY | S::FRAGMENT | S::COMPUTE | S::TASK_EXT | S::MESH_EXT;
    if layout.len() > 1024 {
        return Err("layout too large".into());
    }
    for b in layout {
        if !descriptor_type_known(b.ty) || b.count > 1 << 20 || (b.binding > 4096 && b.binding != BINDLESS_SET_MARKER) || b.stages & !stages.as_raw() != 0 {
            return Err("bad binding layout".into());
        }
    }
    Ok(())
}

/// Formats the provider can name: core formats and the 4_4_4_4 ones.
fn format_known(f: i32) -> bool {
    (0..=184).contains(&f) || f == ash::vk::Format::A4R4G4B4_UNORM_PACK16.as_raw() || f == ash::vk::Format::A4B4G4R4_UNORM_PACK16.as_raw()
}

fn check_graphics(g: &GraphicsRecord) -> Result<(), String> {
    check_layout(&g.layout)?;
    let f: &FixedFunctionKey = &g.ff;
    let bad = |what: &str| Err(format!("graphics pipeline: bad {what}"));
    if !(0..=10).contains(&f.topology) {
        return bad("topology");
    }
    if !(0..=2).contains(&f.polygon_mode) {
        return bad("polygon mode");
    }
    if f.vertex_input.bindings.len() > 32 || f.vertex_input.attributes.len() > 32 {
        return bad("vertex input");
    }
    if f.vertex_input.bindings.iter().any(|b| b.stride > 4096) {
        return bad("vertex stride");
    }
    if f.vertex_input.attributes.iter().any(|a| a.location >= 32 || a.binding as usize >= f.vertex_input.bindings.len() || !format_known(a.format) || a.format == 0 || a.offset > 4096) {
        return bad("vertex attribute");
    }
    if f.color_count as usize > MAX_COLOR_ATTACHMENTS || f.color_formats.iter().any(|&c| !format_known(c)) {
        return bad("color formats");
    }
    if !format_known(f.depth_format) || !format_known(f.stencil_format) {
        return bad("depth format");
    }
    for b in &f.blend {
        let factor = |v: i32| (0..=18).contains(&v);
        let op = |v: i32| (0..=4).contains(&v);
        if !factor(b.src_color) || !factor(b.dst_color) || !factor(b.src_alpha) || !factor(b.dst_alpha) || !op(b.color_op) || !op(b.alpha_op) || b.write_mask > 15 {
            return bad("blend state");
        }
    }
    if !matches!(f.samples, 1 | 2 | 4 | 8 | 16) {
        return bad("sample count");
    }
    if f.logic_op.is_some_and(|op| !(0..=15).contains(&op)) {
        return bad("logic op");
    }
    if f.viewport_count == 0 || f.viewport_count as usize > MAX_VIEWPORTS {
        return bad("viewport count");
    }
    use ash::vk::ShaderStageFlags as S;
    if g.push_stages & !(S::ALL_GRAPHICS | S::MESH_EXT | S::TASK_EXT).as_raw() != 0 {
        return bad("push constant stages");
    }
    Ok(())
}

/// Merge the portable file `bytes` into the list at `list_path` of
/// `title` (created when missing; a list of an older format is moved
/// aside first). Records the list has are kept as they are; new modules
/// are marked for translation.
pub fn import(list_path: &Path, title: &str, bytes: &[u8]) -> Result<ImportSummary, String> {
    let (modules, graphics, compute) = decode(bytes, title)?;
    let mut summary = ImportSummary { modules: modules.len(), pipelines: graphics.len() + compute.len(), ..Default::default() };
    prewarm::retire_old_list(list_path);
    let (stamp, mut have_modules, mut have_graphics, mut have_compute) = match std::fs::read(list_path) {
        Ok(file) => prewarm::decode(&file).ok_or_else(|| format!("{}: the title's shader list is damaged; delete it to import", list_path.display()))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (prewarm::NO_TRANSLATOR, Vec::new(), Vec::new(), Vec::new()),
        Err(e) => return Err(format!("{}: {e}", list_path.display())),
    };
    let mut keys: std::collections::HashSet<u64> = have_modules.iter().map(|m| m.key).collect();
    for m in modules {
        if keys.insert(m.key) {
            summary.new_modules += 1;
            have_modules.push(m);
        }
    }
    let mut seen: std::collections::HashSet<GraphicsRecord> = have_graphics.iter().cloned().collect();
    for g in graphics {
        if seen.insert(g.clone()) {
            summary.new_pipelines += 1;
            have_graphics.push(g);
        }
    }
    let mut seen: std::collections::HashSet<ComputeRecord> = have_compute.iter().cloned().collect();
    for c in compute {
        if seen.insert(c.clone()) {
            summary.new_pipelines += 1;
            have_compute.push(c);
        }
    }
    if summary.new_modules + summary.new_pipelines == 0 {
        return Ok(summary);
    }
    if let Some(dir) = list_path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let out = prewarm::encode(stamp, &have_modules, &have_graphics, &have_compute);
    let tmp = prewarm::temporary_path(list_path);
    std::fs::write(&tmp, &out).and_then(|_| std::fs::rename(&tmp, list_path)).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{}: {e}", list_path.display())
    })?;
    Ok(summary)
}

/// What the list at `list_path` holds and where its permutations come
/// from (`rps5e-cli shader-cache info`): programs found at several
/// addresses, the request parts in which a program's modules differ, and
/// the fixed-function fields in which a module pair's pipelines differ.
pub fn describe(list_path: &Path) -> Result<String, String> {
    use std::collections::{BTreeMap, HashMap};
    use std::fmt::Write as _;
    let file = std::fs::read(list_path).map_err(|e| format!("{}: {e}", list_path.display()))?;
    let (stamp, modules, graphics, compute) = prewarm::decode(&file).ok_or_else(|| format!("{}: not a shader prewarm list of this format", list_path.display()))?;
    let mut out = String::new();
    let fresh = modules.iter().filter(|m| m.fresh).count();
    let _ = writeln!(out, "translator {stamp:016x}: {} modules ({fresh} translated by it), {} graphics and {} compute pipelines", modules.len(), graphics.len(), compute.len());
    // Programs by code (plan hash and stage): the addresses they were at,
    // and their modules.
    let mut by_code: HashMap<(u64, u8), (std::collections::BTreeSet<u64>, Vec<&ModuleRecord>)> = HashMap::new();
    for m in &modules {
        if let Some(s) = &m.source {
            let e = by_code.entry((s.program.plan_hash, s.program.stage)).or_default();
            e.0.insert(s.program.address);
            e.1.push(m);
        }
    }
    let relocated = by_code.values().filter(|(a, _)| a.len() > 1).count();
    // The same permutation (inputs equal but for the address) made again
    // for another address: a program that reads its own address.
    let mut same_inputs: HashMap<(u64, u8, &[u32]), std::collections::BTreeSet<u64>> = HashMap::new();
    for m in &modules {
        if let Some(s) = &m.source {
            same_inputs.entry((s.program.plan_hash, s.program.stage, &s.request[..s.request.len().saturating_sub(1)])).or_default().insert(s.program.address);
        }
    }
    let again: usize = same_inputs.values().map(|a| a.len() - 1).sum();
    let _ = writeln!(out, "{} programs; {relocated} of them at more than one address; {again} modules are a permutation made again for another address", by_code.len());
    let mut part_diffs: BTreeMap<String, usize> = BTreeMap::new();
    let mut permuted = 0;
    for (_, ms) in by_code.values() {
        if ms.len() < 2 {
            continue;
        }
        permuted += 1;
        let decoded: Vec<Vec<String>> = ms.iter().filter_map(|m| super::sources::request_fields(m.source.as_ref()?).ok()).collect();
        let mut names: BTreeMap<&str, std::collections::HashSet<&str>> = BTreeMap::new();
        for fields in &decoded {
            for f in fields {
                let (name, value) = f.split_once(" = ").unwrap_or((f, ""));
                names.entry(name).or_default().insert(value);
            }
        }
        for (name, values) in names {
            if values.len() > 1 {
                *part_diffs.entry(name.to_string()).or_default() += 1;
                if verbose_describe() {
                    let _ = writeln!(out, "  {name}: {values:?}");
                }
            }
        }
    }
    let _ = writeln!(out, "{permuted} programs with several modules; programs whose modules differ in:");
    for (k, n) in &part_diffs {
        let _ = writeln!(out, "  {k}: {n}");
    }
    // Pipelines by module pair.
    let mut pairs: HashMap<(u64, u64), Vec<&GraphicsRecord>> = HashMap::new();
    for g in &graphics {
        pairs.entry((g.vs, g.ps)).or_default().push(g);
    }
    let multi: Vec<_> = pairs.values().filter(|v| v.len() > 1).collect();
    let _ = writeln!(out, "{} module pairs, {} with several graphics pipelines ({} pipelines); pairs whose pipelines differ in:", pairs.len(), multi.len(), multi.iter().map(|v| v.len()).sum::<usize>());
    let mut ff_diffs: BTreeMap<&str, usize> = BTreeMap::new();
    for v in multi {
        let mut note = |name: &'static str, differs: bool| {
            if differs {
                *ff_diffs.entry(name).or_default() += 1;
            }
        };
        let first = v[0];
        let any = |f: &dyn Fn(&GraphicsRecord) -> bool| v.iter().any(|g| f(g));
        note("layout", any(&|g| g.layout != first.layout || g.push_stages != first.push_stages));
        note("mesh", any(&|g| g.mesh != first.mesh));
        let (a, ff) = (&first.ff, |g: &GraphicsRecord| g.ff.clone());
        note("topology", any(&|g| ff(g).topology != a.topology));
        note("rect", any(&|g| ff(g).rect != a.rect));
        note("primitive restart", any(&|g| ff(g).primitive_restart != a.primitive_restart));
        note("vertex input", any(&|g| ff(g).vertex_input != a.vertex_input));
        note("polygon mode", any(&|g| ff(g).polygon_mode != a.polygon_mode));
        note("depth clip/clamp", any(&|g| (ff(g).depth_clip, ff(g).depth_clamp, ff(g).negative_one_to_one) != (a.depth_clip, a.depth_clamp, a.negative_one_to_one)));
        note("color formats", any(&|g| (ff(g).color_formats, ff(g).color_count) != (a.color_formats, a.color_count)));
        note("blend", any(&|g| ff(g).blend != a.blend));
        note("depth/stencil format", any(&|g| (ff(g).depth_format, ff(g).stencil_format) != (a.depth_format, a.stencil_format)));
        note("samples", any(&|g| (ff(g).samples, ff(g).centered_samples, ff(g).sample_shading, ff(g).sample_mask) != (a.samples, a.centered_samples, a.sample_shading, a.sample_mask)));
        note("logic op", any(&|g| ff(g).logic_op != a.logic_op));
        note("viewports", any(&|g| ff(g).viewport_count != a.viewport_count));
        note("provoking vertex", any(&|g| ff(g).provoking_last != a.provoking_last));
    }
    for (k, n) in &ff_diffs {
        let _ = writeln!(out, "  {k}: {n}");
    }
    let mut by_cs: HashMap<u64, usize> = HashMap::new();
    for c in &compute {
        *by_cs.entry(c.cs).or_default() += 1;
    }
    let _ = writeln!(out, "{} compute modules, {} with several pipelines", by_cs.len(), by_cs.values().filter(|&&n| n > 1).count());
    Ok(out)
}

/// `RPS5E_SHADER_INFO_VERBOSE=1`: `describe` prints the differing values.
fn verbose_describe() -> bool {
    std::env::var("RPS5E_SHADER_INFO_VERBOSE").is_ok_and(|v| v == "1")
}

/// Read a portable file from `path` (size-checked before it is read).
pub fn read_file(path: &Path) -> Result<Vec<u8>, String> {
    let len = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?.len();
    if len > MAX_FILE_BYTES {
        return Err(format!("{}: too large for a shader cache ({len} bytes)", path.display()));
    }
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// The title id a portable file names (None: not such a file).
pub fn title_of(bytes: &[u8]) -> Option<String> {
    let payload = super::cache::decode(SIGNATURE, bytes)?;
    let words: Vec<u32> = payload.get(..payload.len().min(4 * 20) / 4 * 4)?.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let mut r = Reader(&words);
    (r.u()? == MAGIC && r.u()? == VERSION).then_some(())?;
    r.string()
}

#[cfg(test)]
mod tests {
    use super::super::sources::{CodeImage, ModuleSource};
    use std::sync::Arc;
    use super::super::state::{BlendKey, RectMode, VertexAttributeKey, VertexBindingKey, VertexInputKey};
    use super::*;

    fn program(address: u64) -> Arc<ProgramSource> {
        Arc::new(ProgramSource {
            address,
            continuation: 0,
            stage: 3,
            user_data_base: 0,
            user_data_count: 16,
            wave_size: 64,
            plan_hash: address,
            guest_plan: false,
            image: Arc::new(CodeImage::from_words(vec![(address, 0xBF81_0000)])),
        })
    }

    fn module(key: u64, address: u64, fresh: bool) -> ModuleRecord {
        let source = Arc::new(ModuleSource { program: program(address), request: vec![5, key as u32].into(), reads: Default::default() });
        ModuleRecord {
            key,
            stage: 3,
            spirv: if fresh { vec![0x0723_0203, key as u32].into() } else { Arc::default() },
            reflection: ShaderReflection { entry_point: "main".into(), ..Default::default() },
            fresh,
            source: Some(source),
        }
    }

    fn ff() -> FixedFunctionKey {
        FixedFunctionKey {
            topology: 3,
            rect: RectMode::None,
            primitive_restart: false,
            vertex_input: VertexInputKey { bindings: vec![VertexBindingKey { stride: 12, per_instance: false }], attributes: vec![VertexAttributeKey { location: 0, binding: 0, format: 106, offset: 0 }] },
            polygon_mode: 0,
            depth_clip: true,
            depth_clamp: false,
            negative_one_to_one: false,
            color_formats: [37, 0, 0, 0, 0, 0, 0, 0],
            color_count: 1,
            blend: [BlendKey { enable: false, src_color: 1, dst_color: 0, color_op: 0, src_alpha: 1, dst_alpha: 0, alpha_op: 0, write_mask: 15 }; MAX_COLOR_ATTACHMENTS],
            depth_format: 126,
            stencil_format: 0,
            samples: 1,
            centered_samples: false,
            sample_shading: false,
            sample_mask: u32::MAX,
            logic_op: None,
            viewport_count: 1,
            provoking_last: false,
        }
    }

    fn layout() -> Vec<BindingDesc> {
        vec![BindingDesc { binding: 0, ty: 7, count: 1, stages: ash::vk::ShaderStageFlags::COMPUTE.as_raw() }]
    }

    fn dir(name: &str) -> std::path::PathBuf {
        let d = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../local/tmp/portable-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_list(path: &Path, stamp: u64, modules: &[ModuleRecord], graphics: &[GraphicsRecord], compute: &[ComputeRecord]) {
        std::fs::write(path, prewarm::encode(stamp, modules, graphics, compute)).unwrap();
    }

    fn graphics(vs: u64, ps: u64) -> GraphicsRecord {
        GraphicsRecord { vs, ps, layout: layout(), push_stages: ash::vk::ShaderStageFlags::VERTEX.as_raw(), ff: ff(), mesh: false }
    }

    /// Export strips the SPIR-V and anything of this machine; importing
    /// into an empty title gives every record back, marked for
    /// translation, and a second import adds nothing.
    #[test]
    fn export_import_round_trip() {
        let d = dir("roundtrip");
        let list = d.join("PPSA00001.prewarm");
        let modules = vec![module(1, 0x1000, true), module(2, 0x2000, true), ModuleRecord { source: None, ..module(3, 0x3000, true) }];
        let compute = vec![ComputeRecord { cs: 1, layout: layout(), subgroup32: false }, ComputeRecord { cs: 3, layout: layout(), subgroup32: false }];
        write_list(&list, 77, &modules, &[graphics(1, 2)], &compute);
        let file = export(&list, "PPSA00001").unwrap();
        assert_eq!(title_of(&file).as_deref(), Some("PPSA00001"));
        // No SPIR-V: the magic number of a module is nowhere in the file.
        assert!(!file.windows(4).any(|w| w == 0x0723_0203u32.to_le_bytes()));
        let target = d.join("other/PPSA00001.prewarm");
        let s = import(&target, "PPSA00001", &file).unwrap();
        assert_eq!(s, ImportSummary { modules: 2, pipelines: 2, new_modules: 2, new_pipelines: 2 });
        let (stamp, m, g, c) = prewarm::decode(&std::fs::read(&target).unwrap()).unwrap();
        assert_eq!(stamp, prewarm::NO_TRANSLATOR);
        assert_eq!(m.iter().map(|m| m.key).collect::<Vec<_>>(), vec![1, 2]);
        assert!(m.iter().all(|m| !m.fresh && m.spirv.is_empty() && m.source.is_some()));
        assert_eq!(*m[0].source.as_ref().unwrap().program, *program(0x1000));
        assert!(g == vec![graphics(1, 2)]);
        // The pipeline of the module without inputs is not exported.
        assert_eq!(c.len(), 1);
        let again = import(&target, "PPSA00001", &file).unwrap();
        assert_eq!((again.new_modules, again.new_pipelines), (0, 0));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Two players' lists combine: the union of both, one record per key;
    /// the importer's own translated modules stay as they are.
    #[test]
    fn import_merges_two_lists() {
        let d = dir("merge");
        let a = d.join("a/PPSA00002.prewarm");
        let b = d.join("b/PPSA00002.prewarm");
        std::fs::create_dir_all(a.parent().unwrap()).unwrap();
        std::fs::create_dir_all(b.parent().unwrap()).unwrap();
        write_list(&a, 9, &[module(1, 0x1000, true), module(2, 0x2000, true)], &[graphics(1, 2)], &[]);
        write_list(&b, 5, &[module(2, 0x2000, true), module(4, 0x4000, true)], &[graphics(4, 2)], &[ComputeRecord { cs: 4, layout: layout(), subgroup32: true }]);
        let s = import(&a, "PPSA00002", &export(&b, "PPSA00002").unwrap()).unwrap();
        assert_eq!((s.modules, s.new_modules, s.pipelines, s.new_pipelines), (2, 1, 2, 2));
        let (stamp, m, g, c) = prewarm::decode(&std::fs::read(&a).unwrap()).unwrap();
        assert_eq!(stamp, 9, "the importer's translator stamp stays");
        let mut keys: Vec<u64> = m.iter().map(|m| m.key).collect();
        keys.sort();
        assert_eq!(keys, vec![1, 2, 4]);
        // Its own modules keep their SPIR-V; the new one is to translate.
        assert!(m.iter().filter(|m| m.key != 4).all(|m| m.fresh && !m.spirv.is_empty()));
        assert!(m.iter().any(|m| m.key == 4 && !m.fresh));
        assert_eq!(g.len(), 2);
        assert_eq!(c.len(), 1);
        // The list loads, with the imported module marked for translation.
        let list = prewarm::PrewarmList::load(Some(&a.with_extension("bin")), 9);
        assert_eq!(list.stale().map(|m| m.key).collect::<Vec<_>>(), vec![4]);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Damaged, truncated, foreign and hostile files are refused without
    /// a panic and leave the title's list untouched.
    #[test]
    fn corrupted_files_are_rejected() {
        let d = dir("corrupt");
        let src = d.join("src.prewarm");
        write_list(&src, 3, &[module(1, 0x1000, true), module(2, 0x2000, true)], &[graphics(1, 2)], &[ComputeRecord { cs: 1, layout: layout(), subgroup32: false }]);
        let file = export(&src, "PPSA00003").unwrap();
        let target = d.join("PPSA00003.prewarm");
        write_list(&target, 3, &[module(9, 0x9000, true)], &[], &[]);
        let before = std::fs::read(&target).unwrap();
        let refused = |bytes: &[u8]| {
            assert!(import(&target, "PPSA00003", bytes).is_err());
            assert_eq!(std::fs::read(&target).unwrap(), before);
        };
        // Wrong title.
        assert!(import(&target, "PPSA00004", &file).unwrap_err().contains("PPSA00003"));
        // Every truncation, and a flipped bit anywhere (the checksum).
        for n in 0..file.len() {
            refused(&file[..n]);
        }
        for i in (0..file.len()).step_by(7) {
            let mut bad = file.clone();
            bad[i] ^= 0x10;
            refused(&bad);
        }
        refused(b"");
        refused(b"rps5eSC1\n");
        // A well-formed file of another version.
        let mut w = Writer(Vec::new());
        w.u(MAGIC);
        w.u(VERSION + 1);
        w.string("PPSA00003");
        w.u(0);
        let payload: Vec<u8> = w.0.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert!(import(&target, "PPSA00003", &super::super::cache::encode(SIGNATURE, &payload)).unwrap_err().contains("version"));
        // Correct framing and checksum around hostile records.
        let hostile = |modules: Vec<ModuleRecord>, graphics: Vec<GraphicsRecord>, compute: Vec<ComputeRecord>| {
            let inner = prewarm::encode(0, &modules, &graphics, &compute);
            let mut w = Writer(Vec::new());
            w.u(MAGIC);
            w.u(VERSION);
            w.string("PPSA00003");
            w.u(inner.len() as u32);
            let mut payload: Vec<u8> = w.0.iter().flat_map(|v| v.to_le_bytes()).collect();
            payload.extend_from_slice(&inner);
            super::super::cache::encode(SIGNATURE, &payload)
        };
        let ok = || (vec![module(1, 0x1000, false), module(2, 0x2000, false)], vec![graphics(1, 2)], vec![]);
        let (m, g, c) = ok();
        assert!(import(&d.join("x/PPSA00003.prewarm"), "PPSA00003", &hostile(m, g, c)).is_ok(), "the baseline is accepted");
        let mut cases: Vec<Records> = Vec::new();
        let (mut m, g, c) = ok();
        m[0].spirv = vec![1, 2, 3].into();
        m[0].fresh = true;
        cases.push((m, g, c));
        let (m, _, c) = ok();
        cases.push((m, vec![graphics(1, 99)], c));
        let (m, g, _) = ok();
        cases.push((m, g, vec![ComputeRecord { cs: 1, layout: vec![BindingDesc { binding: 0, ty: 12345, count: 1, stages: 0x20 }], subgroup32: false }]));
        let (m, _, c) = ok();
        let mut bad = graphics(1, 2);
        bad.ff.color_formats[0] = -5;
        cases.push((m, vec![bad], c));
        let (m, _, c) = ok();
        let mut bad = graphics(1, 2);
        bad.ff.vertex_input.attributes[0].binding = 3;
        cases.push((m, vec![bad], c));
        let (m, _, c) = ok();
        let mut bad = graphics(1, 2);
        bad.ff.samples = 3;
        cases.push((m, vec![bad], c));
        let (mut m, g, c) = ok();
        let mut p = (*m[0].source.as_ref().unwrap().program).clone();
        p.wave_size = 48;
        m[0].source = Some(Arc::new(ModuleSource { program: Arc::new(p), request: vec![1].into(), reads: Default::default() }));
        cases.push((m, g, c));
        let (mut m, g, c) = ok();
        m.push(module(1, 0x1000, false));
        cases.push((m, g, c));
        let (mut m, g, c) = ok();
        m[1].source = None;
        cases.push((m, g, c));
        for (m, g, c) in cases {
            refused(&hostile(m, g, c));
        }
        let _ = std::fs::remove_dir_all(&d);
    }
}
