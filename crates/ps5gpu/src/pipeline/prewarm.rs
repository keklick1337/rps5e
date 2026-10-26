// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The shader prewarm list: every module and pipeline a game created, kept
//! per title next to the pipeline cache (`<title>.prewarm`), so the next
//! run creates them before the game asks for them.
//!
//! A module is kept as its SPIR-V and reflection under its permutation
//! key, with the guest inputs of its translation ([`super::sources`]); a
//! pipeline as the keys of its modules, the bindings of its layout and its
//! fixed-function key. The provider loads the list once (modules and
//! layouts on its thread, pipelines on the compile workers), so the keys
//! it later looks up are the ones the list recreated.
//!
//! The list names the translator stamp its SPIR-V was made by
//! ([`super::compiler::ShaderCompiler::translator_stamp`]). A list of
//! another translator (an emulator update, another driver or device
//! features) keeps its modules: they are translated again from their guest
//! inputs by the current translator; only modules without inputs or whose
//! translation fails are dropped.

use super::compiler::{ComponentKind, ReflectedBinding, ReflectedVariable, ShaderReflection};
use super::descriptors::BindingDesc;
use super::sources::{ModuleSource, ProgramSource};
use super::state::{BlendKey, FixedFunctionKey, RectMode, VertexAttributeKey, VertexBindingKey, VertexInputKey, MAX_COLOR_ATTACHMENTS};
use ash::vk;
use std::path::PathBuf;
use std::sync::Arc;

/// Records that start a save once [`SAVE_PERIOD`] has passed since the
/// last one; fewer are saved by [`PrewarmList::tick`] after the period
/// (a list is hundreds of MiB: saves run on their own thread).
const SAVE_INTERVAL: u32 = 64;
const SAVE_PERIOD: std::time::Duration = std::time::Duration::from_secs(30);

/// The signature line of every list file (the device, driver and
/// translator are named by the stamp inside).
const SIGNATURE: &str = "rps5ePW4\n";

#[derive(Clone)]
pub struct ModuleRecord {
    pub key: u64,
    /// `ShaderStage as u8`.
    pub stage: u8,
    /// The SPIR-V of the list's translator (empty while `!fresh`); shared
    /// with the copy a background save writes.
    pub spirv: Arc<[u32]>,
    pub reflection: ShaderReflection,
    /// False: made by another translator, to be translated again from
    /// `source`.
    pub fresh: bool,
    /// The guest inputs of the translation (None: unknown).
    pub source: Option<Arc<ModuleSource>>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct GraphicsRecord {
    pub vs: u64,
    /// 0: no pixel stage.
    pub ps: u64,
    pub layout: Vec<BindingDesc>,
    pub push_stages: u32,
    pub ff: FixedFunctionKey,
    pub mesh: bool,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ComputeRecord {
    pub cs: u64,
    pub layout: Vec<BindingDesc>,
    pub subgroup32: bool,
}

#[derive(Default)]
pub struct PrewarmList {
    path: Option<PathBuf>,
    /// The translator the fresh modules were made by.
    stamp: u64,
    pub modules: Vec<ModuleRecord>,
    /// Index into `modules` by key.
    module_index: crate::fxhash::HashMap<u64, usize>,
    pub graphics: Vec<GraphicsRecord>,
    graphics_seen: crate::fxhash::HashSet<GraphicsRecord>,
    pub compute: Vec<ComputeRecord>,
    compute_seen: crate::fxhash::HashSet<ComputeRecord>,
    unsaved: u32,
    last_save: Option<std::time::Instant>,
    /// Saves started, and the number of the last one written (saves write
    /// one at a time and never over a later one).
    saves: u64,
    written: Arc<std::sync::Mutex<u64>>,
}

/// `RPS5E_SHADER_PREWARM=0`: no prewarm list is read or written.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_SHADER_PREWARM").is_ok_and(|v| v == "0"))
}

/// The list file of the pipeline cache file `cache_path`.
pub fn list_path(cache_path: &std::path::Path) -> PathBuf {
    cache_path.with_extension("prewarm")
}

/// What a list file holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ListStatus {
    Missing,
    /// This format (it may name another translator).
    Current,
    /// An older format (`rps5ePW<n>`): no guest inputs, nothing can be
    /// built from it.
    Old(u32),
    Damaged,
}

/// The status of the list file `path`.
pub fn list_status(path: &std::path::Path) -> ListStatus {
    let Ok(file) = std::fs::read(path) else { return ListStatus::Missing };
    if decode(&file).is_some() {
        return ListStatus::Current;
    }
    let Some(rest) = file.strip_prefix(b"rps5ePW") else { return ListStatus::Damaged };
    let digits = &rest[..rest.iter().position(|b| !b.is_ascii_digit()).unwrap_or(rest.len())];
    match std::str::from_utf8(digits).ok().and_then(|v| v.parse::<u32>().ok()) {
        Some(n) if n < 4 => ListStatus::Old(n),
        _ => ListStatus::Damaged,
    }
}

/// Move a list of an older format aside (`<title>.prewarm.pw<n>.old`): it
/// holds no guest inputs, so nothing can be translated from it; the next
/// game run collects a new list. Returns the new name.
pub fn retire_old_list(path: &std::path::Path) -> Option<PathBuf> {
    let ListStatus::Old(n) = list_status(path) else { return None };
    let to = path.with_extension(format!("prewarm.pw{n}.old"));
    std::fs::rename(path, &to).ok()?;
    log::warn!(
        "ps5gpu: shader prewarm list {} is of the older format PW{n} (no guest inputs: nothing can be built from it); moved to {} - the game run collects a new list",
        path.display(),
        to.display()
    );
    Some(to)
}

/// Mark every module of the list file `path` for translation (a full
/// recompile): the file is written again naming no translator. Returns the
/// module keys.
pub fn mark_for_recompile(path: &std::path::Path) -> Result<Vec<u64>, String> {
    let file = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (_, modules, graphics, compute) = decode(&file).ok_or_else(|| format!("{}: not a prewarm list of this format", path.display()))?;
    let keys = modules.iter().map(|m| m.key).collect();
    let bytes = encode(NO_TRANSLATOR, &modules, &graphics, &compute);
    let tmp = temporary_path(path);
    std::fs::write(&tmp, &bytes).and_then(|_| std::fs::rename(&tmp, path)).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{}: {e}", path.display())
    })?;
    Ok(keys)
}

/// The file a save of the list `path` writes before renaming it over the
/// list (`<title>.prewarm.tmp<pid>`: not the name the pipeline cache next
/// to it saves through, `<title>.tmp<pid>`, which may be written at the
/// same time).
pub(super) fn temporary_path(path: &std::path::Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp{}", std::process::id()));
    path.with_file_name(name)
}

/// Delete temporary save files of the list `path` whose process is gone.
fn remove_stale_temporaries(path: &std::path::Path) {
    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
        super::cache::remove_stale_temporaries_of(path, name);
    }
}

/// The stamp of a list marked for a full recompile (no translator has it).
pub(super) const NO_TRANSLATOR: u64 = 0;

impl PrewarmList {
    /// The list kept next to the pipeline cache file `cache_path`, for the
    /// translator `stamp`: empty when missing, of an older format or
    /// damaged. Modules of another stamp stay, marked for translation.
    pub fn load(cache_path: Option<&std::path::Path>, stamp: u64) -> PrewarmList {
        let path = cache_path.filter(|_| enabled()).map(list_path);
        let mut list = PrewarmList::default();
        list.path = path;
        list.stamp = stamp;
        let Some(path) = list.path.clone() else { return list };
        remove_stale_temporaries(&path);
        let Ok(file) = std::fs::read(&path) else { return list };
        match decode(&file) {
            Some((file_stamp, modules, graphics, compute)) => {
                let retranslate = file_stamp != stamp;
                let (mut stale, mut dropped) = (0, 0);
                for mut m in modules {
                    if retranslate {
                        m.fresh = false;
                        m.spirv = Arc::default();
                    }
                    if !m.fresh && m.source.is_none() {
                        dropped += 1;
                        continue;
                    }
                    if !list.module_index.contains_key(&m.key) {
                        stale += !m.fresh as usize;
                        list.module_index.insert(m.key, list.modules.len());
                        list.modules.push(m);
                    }
                }
                // Records read from the file are not new: counting them as
                // unsaved saved the list from inside this loop once 64 were
                // in (its first 64 graphics pipelines and no compute
                // pipeline), and the count reset below kept the rest from
                // ever being written back.
                for g in graphics {
                    list.insert_graphics(g);
                }
                for c in compute {
                    list.insert_compute(c);
                }
                // A list taken over from another translator is written
                // with this one's stamp.
                list.unsaved = retranslate as u32;
                log::info!(
                    "ps5gpu: shader prewarm list: {} modules ({stale} to translate again, {dropped} without inputs dropped), {} graphics and {} compute pipelines from {path:?}",
                    list.modules.len(),
                    list.graphics.len(),
                    list.compute.len()
                );
            }
            None => {
                if retire_old_list(&path).is_none() {
                    log::warn!("ps5gpu: shader prewarm list: ignoring {path:?} (damaged); the game run makes it again");
                }
            }
        }
        list
    }

    pub fn is_active(&self) -> bool {
        self.path.is_some()
    }

    /// Modules to translate again.
    pub fn stale(&self) -> impl Iterator<Item = &ModuleRecord> {
        self.modules.iter().filter(|m| !m.fresh)
    }

    /// Note a module made by the current translator (replacing a stale
    /// record of the same key).
    pub fn add_module(&mut self, key: u64, stage: u8, spirv: Vec<u32>, reflection: ShaderReflection, source: Option<Arc<ModuleSource>>) {
        if self.path.is_none() {
            return;
        }
        let record = ModuleRecord { key, stage, spirv: spirv.into(), reflection, fresh: true, source };
        match self.module_index.get(&key) {
            Some(&i) if self.modules[i].fresh => {}
            Some(&i) => {
                let source = record.source.clone().or_else(|| self.modules[i].source.take());
                self.modules[i] = ModuleRecord { source, ..record };
                self.note();
            }
            None => {
                self.module_index.insert(key, self.modules.len());
                self.modules.push(record);
                self.note();
            }
        }
    }

    /// Forget a stale module that could not be translated again.
    pub fn remove_module(&mut self, key: u64) {
        let Some(i) = self.module_index.remove(&key) else { return };
        self.modules.swap_remove(i);
        if let Some(moved) = self.modules.get(i) {
            self.module_index.insert(moved.key, i);
        }
        self.unsaved += 1;
    }

    pub fn add_graphics(&mut self, g: GraphicsRecord) {
        if self.insert_graphics(g) {
            self.note();
        }
    }

    pub fn add_compute(&mut self, c: ComputeRecord) {
        if self.insert_compute(c) {
            self.note();
        }
    }

    /// Enter a graphics record unless known. True when new.
    fn insert_graphics(&mut self, g: GraphicsRecord) -> bool {
        let new = self.graphics_seen.insert(g.clone());
        if new {
            self.graphics.push(g);
        }
        new
    }

    /// Enter a compute record unless known. True when new.
    fn insert_compute(&mut self, c: ComputeRecord) -> bool {
        let new = self.compute_seen.insert(c.clone());
        if new {
            self.compute.push(c);
        }
        new
    }

    fn note(&mut self) {
        self.unsaved += 1;
        if self.unsaved >= SAVE_INTERVAL && self.save_due() {
            self.save_in_background();
        }
    }

    fn save_due(&self) -> bool {
        self.last_save.is_none_or(|t| t.elapsed() >= SAVE_PERIOD)
    }

    /// Called now and then by the running game (once a frame): records
    /// left unsaved are saved once [`SAVE_PERIOD`] has passed since the
    /// last save, so a run that stops adding records keeps them even when
    /// its exit is cut short.
    pub fn tick(&mut self) {
        if self.unsaved != 0 && self.save_due() {
            self.save_in_background();
        }
    }

    /// Write the list now (atomically through a temporary file), after any
    /// save still running on its thread.
    pub fn save(&mut self) {
        if let Some((data, number)) = self.start_save() {
            write_save(&self.written, number, data);
        }
    }

    /// Write the list on a thread of its own: encoding and writing a list
    /// of hundreds of MiB takes most of a second, which the command thread
    /// (where records are added) must not wait for. The thread writes a
    /// copy of the records (the SPIR-V is shared, not copied).
    pub fn save_in_background(&mut self) {
        let Some((data, number)) = self.start_save() else { return };
        let (written, copy) = (self.written.clone(), data.clone());
        if std::thread::Builder::new().name("rps5e-prewarm-save".into()).spawn(move || write_save(&written, number, copy)).is_err() {
            write_save(&self.written, number, data);
        }
    }

    /// The copy to write and its save number (None: nothing to write).
    fn start_save(&mut self) -> Option<(Arc<SaveData>, u64)> {
        if self.unsaved == 0 {
            return None;
        }
        self.unsaved = 0;
        self.last_save = Some(std::time::Instant::now());
        let path = self.path.clone()?;
        self.saves += 1;
        let data = SaveData { path, stamp: self.stamp, modules: self.modules.clone(), graphics: self.graphics.clone(), compute: self.compute.clone() };
        Some((Arc::new(data), self.saves))
    }
}

/// What one save writes.
struct SaveData {
    path: PathBuf,
    stamp: u64,
    modules: Vec<ModuleRecord>,
    graphics: Vec<GraphicsRecord>,
    compute: Vec<ComputeRecord>,
}

/// Write save `number` unless a later one was written already (saves take
/// the lock one at a time).
fn write_save(written: &std::sync::Mutex<u64>, number: u64, data: Arc<SaveData>) {
    let mut last = written.lock().unwrap_or_else(|e| e.into_inner());
    if *last >= number {
        return;
    }
    *last = number;
    let path = &data.path;
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let bytes = encode(data.stamp, &data.modules, &data.graphics, &data.compute);
    let tmp = temporary_path(path);
    if std::fs::write(&tmp, &bytes).is_err() || std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

impl Drop for PrewarmList {
    fn drop(&mut self) {
        self.save();
    }
}

// ---------------------------------------------------------------------
// Encoding: the signature line, a checksum, then little-endian words.
// ---------------------------------------------------------------------

pub(super) struct Writer(pub Vec<u32>);

impl Writer {
    pub(super) fn u(&mut self, v: u32) {
        self.0.push(v);
    }
    pub(super) fn q(&mut self, v: u64) {
        self.0.push(v as u32);
        self.0.push((v >> 32) as u32);
    }
    pub(super) fn i(&mut self, v: i32) {
        self.0.push(v as u32);
    }
    pub(super) fn b(&mut self, v: bool) {
        self.0.push(v as u32);
    }
    pub(super) fn words(&mut self, w: &[u32]) {
        self.u(w.len() as u32);
        self.0.extend_from_slice(w);
    }
    pub(super) fn string(&mut self, s: &str) {
        let bytes = s.as_bytes();
        self.u(bytes.len() as u32);
        for c in bytes.chunks(4) {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            self.u(u32::from_le_bytes(w));
        }
    }
}

pub(super) struct Reader<'a>(pub &'a [u32]);

impl Reader<'_> {
    pub(super) fn u(&mut self) -> Option<u32> {
        let (&v, rest) = self.0.split_first()?;
        self.0 = rest;
        Some(v)
    }
    pub(super) fn q(&mut self) -> Option<u64> {
        Some(self.u()? as u64 | (self.u()? as u64) << 32)
    }
    pub(super) fn i(&mut self) -> Option<i32> {
        self.u().map(|v| v as i32)
    }
    pub(super) fn b(&mut self) -> Option<bool> {
        self.u().map(|v| v != 0)
    }
    pub(super) fn len(&mut self, max: usize) -> Option<usize> {
        let n = self.u()? as usize;
        (n <= max && n <= self.0.len().max(1) * 4).then_some(n)
    }
    pub(super) fn words(&mut self) -> Option<Vec<u32>> {
        let n = self.len(self.0.len())?;
        let (w, rest) = self.0.split_at(n);
        self.0 = rest;
        Some(w.to_vec())
    }
    pub(super) fn string(&mut self) -> Option<String> {
        let n = self.len(4096)?;
        let mut bytes = Vec::with_capacity(n);
        for _ in 0..n.div_ceil(4) {
            bytes.extend_from_slice(&self.u()?.to_le_bytes());
        }
        bytes.truncate(n);
        String::from_utf8(bytes).ok()
    }
}

fn kind_code(k: ComponentKind) -> u32 {
    match k {
        ComponentKind::Float => 0,
        ComponentKind::Uint => 1,
        ComponentKind::Sint => 2,
    }
}

fn kind_of(v: u32) -> Option<ComponentKind> {
    Some(match v {
        0 => ComponentKind::Float,
        1 => ComponentKind::Uint,
        2 => ComponentKind::Sint,
        _ => return None,
    })
}

fn put_variables(w: &mut Writer, v: &[ReflectedVariable]) {
    w.u(v.len() as u32);
    for x in v {
        w.u(x.location);
        w.u(x.components);
        w.u(kind_code(x.kind));
        w.b(x.flat);
    }
}

fn get_variables(r: &mut Reader) -> Option<Vec<ReflectedVariable>> {
    let n = r.len(1024)?;
    (0..n).map(|_| Some(ReflectedVariable { location: r.u()?, components: r.u()?, kind: kind_of(r.u()?)?, flat: r.b()? })).collect()
}

fn put_reflection(w: &mut Writer, x: &ShaderReflection) {
    w.string(&x.entry_point);
    w.u(x.bindings.len() as u32);
    for b in &x.bindings {
        w.u(b.binding);
        w.i(b.descriptor_type.as_raw());
        w.u(b.count);
    }
    match x.push_constants {
        Some((o, s)) => {
            w.u(1);
            w.u(o);
            w.u(s);
        }
        None => w.u(0),
    }
    put_variables(w, &x.vertex_inputs);
    put_variables(w, &x.outputs);
    w.b(x.writes_point_size);
    w.u(x.clip_distances);
    w.b(x.writes_viewport_index);
    w.b(x.writes_depth);
    for v in x.local_size {
        w.u(v);
    }
    w.u(x.private_lds_bytes);
    w.u(x.required_subgroup_size);
    w.b(x.own_buffer_checks);
}

fn get_reflection(r: &mut Reader) -> Option<ShaderReflection> {
    let entry_point = r.string()?;
    let n = r.len(1024)?;
    let bindings = (0..n).map(|_| Some(ReflectedBinding { binding: r.u()?, descriptor_type: vk::DescriptorType::from_raw(r.i()?), count: r.u()? })).collect::<Option<Vec<_>>>()?;
    let push_constants = if r.b()? { Some((r.u()?, r.u()?)) } else { None };
    Some(ShaderReflection {
        entry_point,
        bindings,
        push_constants,
        vertex_inputs: get_variables(r)?,
        outputs: get_variables(r)?,
        writes_point_size: r.b()?,
        clip_distances: r.u()?,
        writes_viewport_index: r.b()?,
        writes_depth: r.b()?,
        local_size: [r.u()?, r.u()?, r.u()?],
        private_lds_bytes: r.u()?,
        required_subgroup_size: r.u()?,
        own_buffer_checks: r.b()?,
    })
}

fn put_layout(w: &mut Writer, l: &[BindingDesc]) {
    w.u(l.len() as u32);
    for b in l {
        w.u(b.binding);
        w.i(b.ty);
        w.u(b.count);
        w.u(b.stages);
    }
}

fn get_layout(r: &mut Reader) -> Option<Vec<BindingDesc>> {
    let n = r.len(4096)?;
    (0..n).map(|_| Some(BindingDesc { binding: r.u()?, ty: r.i()?, count: r.u()?, stages: r.u()? })).collect()
}

fn rect_code(m: RectMode) -> u32 {
    match m {
        RectMode::None => 0,
        RectMode::Tessellation => 1,
        RectMode::FillRectangle => 2,
        RectMode::Strip => 3,
        RectMode::List => 4,
    }
}

fn rect_of(v: u32) -> Option<RectMode> {
    Some(match v {
        0 => RectMode::None,
        1 => RectMode::Tessellation,
        2 => RectMode::FillRectangle,
        3 => RectMode::Strip,
        4 => RectMode::List,
        _ => return None,
    })
}

fn put_ff(w: &mut Writer, f: &FixedFunctionKey) {
    w.i(f.topology);
    w.u(rect_code(f.rect));
    w.b(f.primitive_restart);
    w.u(f.vertex_input.bindings.len() as u32);
    for b in &f.vertex_input.bindings {
        w.u(b.stride);
        w.b(b.per_instance);
    }
    w.u(f.vertex_input.attributes.len() as u32);
    for a in &f.vertex_input.attributes {
        w.u(a.location);
        w.u(a.binding);
        w.i(a.format);
        w.u(a.offset);
    }
    w.i(f.polygon_mode);
    w.b(f.depth_clip);
    w.b(f.depth_clamp);
    w.b(f.negative_one_to_one);
    for &c in &f.color_formats {
        w.i(c);
    }
    w.u(f.color_count);
    for b in &f.blend {
        w.b(b.enable);
        w.i(b.src_color);
        w.i(b.dst_color);
        w.i(b.color_op);
        w.i(b.src_alpha);
        w.i(b.dst_alpha);
        w.i(b.alpha_op);
        w.u(b.write_mask);
    }
    w.i(f.depth_format);
    w.i(f.stencil_format);
    w.u(f.samples);
    w.b(f.centered_samples);
    w.b(f.sample_shading);
    w.u(f.sample_mask);
    match f.logic_op {
        Some(op) => {
            w.u(1);
            w.i(op);
        }
        None => w.u(0),
    }
    w.u(f.viewport_count);
    w.b(f.provoking_last);
}

fn get_ff(r: &mut Reader) -> Option<FixedFunctionKey> {
    let topology = r.i()?;
    let rect = rect_of(r.u()?)?;
    let primitive_restart = r.b()?;
    let n = r.len(64)?;
    let bindings = (0..n).map(|_| Some(VertexBindingKey { stride: r.u()?, per_instance: r.b()? })).collect::<Option<Vec<_>>>()?;
    let n = r.len(64)?;
    let attributes = (0..n).map(|_| Some(VertexAttributeKey { location: r.u()?, binding: r.u()?, format: r.i()?, offset: r.u()? })).collect::<Option<Vec<_>>>()?;
    let polygon_mode = r.i()?;
    let depth_clip = r.b()?;
    let depth_clamp = r.b()?;
    let negative_one_to_one = r.b()?;
    let mut color_formats = [0i32; MAX_COLOR_ATTACHMENTS];
    for c in color_formats.iter_mut() {
        *c = r.i()?;
    }
    let color_count = r.u()?;
    let mut blend = [BlendKey::default(); MAX_COLOR_ATTACHMENTS];
    for b in blend.iter_mut() {
        *b = BlendKey { enable: r.b()?, src_color: r.i()?, dst_color: r.i()?, color_op: r.i()?, src_alpha: r.i()?, dst_alpha: r.i()?, alpha_op: r.i()?, write_mask: r.u()? };
    }
    let depth_format = r.i()?;
    let stencil_format = r.i()?;
    let samples = r.u()?;
    let centered_samples = r.b()?;
    let sample_shading = r.b()?;
    let sample_mask = r.u()?;
    let logic_op = if r.b()? { Some(r.i()?) } else { None };
    let viewport_count = r.u()?;
    let provoking_last = r.b()?;
    (color_count as usize <= MAX_COLOR_ATTACHMENTS).then_some(FixedFunctionKey {
        topology,
        rect,
        primitive_restart,
        vertex_input: VertexInputKey { bindings, attributes },
        polygon_mode,
        depth_clip,
        depth_clamp,
        negative_one_to_one,
        color_formats,
        color_count,
        blend,
        depth_format,
        stencil_format,
        samples,
        centered_samples,
        sample_shading,
        sample_mask,
        logic_op,
        viewport_count,
        provoking_last,
    })
}

const MAGIC: u32 = 0x5057_3450; // "P4WP"

pub(super) type Decoded = (u64, Vec<ModuleRecord>, Vec<GraphicsRecord>, Vec<ComputeRecord>);

pub(super) fn encode(stamp: u64, modules: &[ModuleRecord], graphics: &[GraphicsRecord], compute: &[ComputeRecord]) -> Vec<u8> {
    let mut w = Writer(Vec::new());
    w.u(MAGIC);
    w.q(stamp);
    // The programs, once each (a program has many permutations).
    let mut programs: Vec<&ProgramSource> = Vec::new();
    let mut by_ptr: crate::fxhash::HashMap<usize, u32> = Default::default();
    let mut by_value: std::collections::HashMap<&ProgramSource, u32> = Default::default();
    for src in modules.iter().filter_map(|m| m.source.as_ref()) {
        let ptr = Arc::as_ptr(&src.program) as usize;
        if by_ptr.contains_key(&ptr) {
            continue;
        }
        let i = *by_value.entry(&*src.program).or_insert_with(|| {
            programs.push(&src.program);
            programs.len() as u32 - 1
        });
        by_ptr.insert(ptr, i);
    }
    w.u(programs.len() as u32);
    for p in &programs {
        p.put(&mut w);
    }
    w.u(modules.len() as u32);
    for m in modules {
        w.q(m.key);
        w.u(m.stage as u32);
        w.b(m.fresh);
        w.words(if m.fresh { &m.spirv[..] } else { &[] });
        put_reflection(&mut w, &m.reflection);
        match &m.source {
            Some(src) => {
                w.u(1);
                w.u(by_ptr[&(Arc::as_ptr(&src.program) as usize)]);
                w.words(&src.request);
                src.reads.put(&mut w);
            }
            None => w.u(0),
        }
    }
    w.u(graphics.len() as u32);
    for g in graphics {
        w.q(g.vs);
        w.q(g.ps);
        put_layout(&mut w, &g.layout);
        w.u(g.push_stages);
        put_ff(&mut w, &g.ff);
        w.b(g.mesh);
    }
    w.u(compute.len() as u32);
    for c in compute {
        w.q(c.cs);
        put_layout(&mut w, &c.layout);
        w.b(c.subgroup32);
    }
    let payload: Vec<u8> = w.0.iter().flat_map(|v| v.to_le_bytes()).collect();
    super::cache::encode(SIGNATURE, &payload)
}

pub(super) fn decode(file: &[u8]) -> Option<Decoded> {
    let payload = super::cache::decode(SIGNATURE, file)?;
    if payload.len() % 4 != 0 {
        return None;
    }
    let words: Vec<u32> = payload.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let mut r = Reader(&words);
    if r.u()? != MAGIC {
        return None;
    }
    let stamp = r.q()?;
    let n = r.len(1 << 22)?;
    let programs = (0..n).map(|_| ProgramSource::get(&mut r).map(Arc::new)).collect::<Option<Vec<_>>>()?;
    let n = r.len(1 << 22)?;
    let mut modules = Vec::with_capacity(n);
    for _ in 0..n {
        let key = r.q()?;
        let stage = u8::try_from(r.u()?).ok()?;
        let fresh = r.b()?;
        let spirv: Arc<[u32]> = r.words()?.into();
        let reflection = get_reflection(&mut r)?;
        let source = if r.b()? {
            let program = programs.get(r.u()? as usize)?.clone();
            let request: Arc<[u32]> = r.words()?.into();
            let reads = Arc::new(super::sources::CodeImage::get(&mut r)?);
            Some(Arc::new(ModuleSource { program, request, reads }))
        } else {
            None
        };
        modules.push(ModuleRecord { key, stage, spirv, reflection, fresh, source });
    }
    let n = r.len(1 << 22)?;
    let graphics = (0..n)
        .map(|_| Some(GraphicsRecord { vs: r.q()?, ps: r.q()?, layout: get_layout(&mut r)?, push_stages: r.u()?, ff: get_ff(&mut r)?, mesh: r.b()? }))
        .collect::<Option<Vec<_>>>()?;
    let n = r.len(1 << 22)?;
    let compute = (0..n).map(|_| Some(ComputeRecord { cs: r.q()?, layout: get_layout(&mut r)?, subgroup32: r.b()? })).collect::<Option<Vec<_>>>()?;
    r.0.is_empty().then_some((stamp, modules, graphics, compute))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ff() -> FixedFunctionKey {
        FixedFunctionKey {
            topology: 3,
            rect: RectMode::Tessellation,
            primitive_restart: true,
            vertex_input: VertexInputKey {
                bindings: vec![VertexBindingKey { stride: 12, per_instance: false }],
                attributes: vec![VertexAttributeKey { location: 0, binding: 0, format: 106, offset: 4 }],
            },
            polygon_mode: 0,
            depth_clip: true,
            depth_clamp: false,
            negative_one_to_one: true,
            color_formats: [97, 0, 0, 0, 0, 0, 0, 0],
            color_count: 1,
            blend: [BlendKey { enable: true, src_color: 6, dst_color: 7, color_op: 0, src_alpha: 1, dst_alpha: 0, alpha_op: 0, write_mask: 15 }; MAX_COLOR_ATTACHMENTS],
            depth_format: 126,
            stencil_format: 0,
            samples: 1,
            centered_samples: true,
            sample_shading: false,
            sample_mask: u32::MAX,
            logic_op: Some(3),
            viewport_count: 16,
            provoking_last: true,
        }
    }

    #[test]
    fn lists_round_trip_and_reject_damage() {
        let reflection = ShaderReflection {
            entry_point: "main_vs".into(),
            bindings: vec![ReflectedBinding { binding: 3, descriptor_type: vk::DescriptorType::STORAGE_BUFFER, count: 2 }],
            push_constants: Some((0, 128)),
            vertex_inputs: vec![],
            outputs: vec![ReflectedVariable { location: 1, components: 4, kind: ComponentKind::Sint, flat: true }],
            writes_point_size: true,
            clip_distances: 2,
            writes_viewport_index: true,
            writes_depth: false,
            local_size: [64, 1, 1],
            private_lds_bytes: 512,
            required_subgroup_size: 0,
            own_buffer_checks: true,
        };
        let program = Arc::new(ProgramSource {
            address: 0x1000,
            continuation: 0,
            stage: 3,
            user_data_base: 0,
            user_data_count: 16,
            wave_size: 64,
            plan_hash: 77,
            guest_plan: true,
            image: Arc::new(super::super::sources::CodeImage::from_words(vec![(0x1000, 0xBF81_0000)])),
        });
        let source = |reads: Vec<(u64, u32)>| {
            Some(Arc::new(ModuleSource { program: program.clone(), request: vec![1, 2, 3].into(), reads: Arc::new(super::super::sources::CodeImage::from_words(reads)) }))
        };
        let modules = vec![
            ModuleRecord { key: 0x1234_5678_9abc_def0, stage: 1, spirv: vec![0x0723_0203, 1, 2, 3].into(), reflection: reflection.clone(), fresh: true, source: source(vec![(0x2000, 5)]) },
            ModuleRecord { key: 7, stage: 3, spirv: Arc::default(), reflection: reflection.clone(), fresh: false, source: source(vec![]) },
            ModuleRecord { key: 8, stage: 2, spirv: vec![0x0723_0203].into(), reflection: reflection.clone(), fresh: true, source: None },
        ];
        let layout = vec![BindingDesc { binding: 0, ty: 7, count: 4, stages: 1 }];
        let graphics = vec![GraphicsRecord { vs: 1, ps: 2, layout: layout.clone(), push_stages: 17, ff: ff(), mesh: false }];
        let compute = vec![ComputeRecord { cs: 9, layout, subgroup32: true }];
        let file = encode(42, &modules, &graphics, &compute);
        let (stamp, m, g, c) = decode(&file).expect("decodes");
        assert_eq!(stamp, 42);
        assert_eq!(m.len(), 3);
        assert_eq!(m[0].key, modules[0].key);
        assert_eq!(m[0].spirv, modules[0].spirv);
        assert_eq!(m[0].reflection, reflection);
        assert!(m[0].fresh && !m[1].fresh && m[2].source.is_none());
        let (a, b) = (m[0].source.as_ref().unwrap(), m[1].source.as_ref().unwrap());
        // One program record shared by both permutations.
        assert!(Arc::ptr_eq(&a.program, &b.program));
        assert_eq!(*a.program, *program);
        assert_eq!(&a.request[..], &[1, 2, 3]);
        assert_eq!(a.reads.read(0x2000), Some(5));
        assert!(g == graphics);
        assert!(c == compute);
        let mut bad = file.clone();
        *bad.last_mut().unwrap() ^= 1;
        assert!(decode(&bad).is_none());
        // A list of the previous format is not read.
        assert!(decode(&super::super::cache::encode("rps5ePW3:x\n", &file[SIGNATURE.len() + 8..])).is_none());
    }

    /// A list written by another translator keeps the modules that carry
    /// their guest inputs (marked for translation) instead of being
    /// dropped; a module translated again replaces its stale record.
    #[test]
    fn another_translators_list_is_kept_for_translation() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../local/tmp/prewarm-list-test-{}", std::process::id()));
        let bin = dir.join("TEST00001.bin");
        let program = Arc::new(ProgramSource {
            address: 0x1000,
            continuation: 0,
            stage: 3,
            user_data_base: 0,
            user_data_count: 16,
            wave_size: 64,
            plan_hash: 1,
            guest_plan: false,
            image: Arc::new(super::super::sources::CodeImage::from_words(vec![(0x1000, 0xBF81_0000)])),
        });
        let source = Some(Arc::new(ModuleSource { program, request: vec![9].into(), reads: Default::default() }));
        let r = ShaderReflection::default();
        {
            let mut list = PrewarmList::load(Some(&bin), 1);
            list.add_module(10, 3, vec![0x0723_0203, 10], r.clone(), source.clone());
            list.add_module(11, 3, vec![0x0723_0203, 11], r.clone(), None);
            list.add_compute(ComputeRecord { cs: 10, layout: vec![], subgroup32: false });
        }
        // The same translator: both modules as made.
        let list = PrewarmList::load(Some(&bin), 1);
        assert_eq!(list.modules.len(), 2);
        assert_eq!(list.stale().count(), 0);
        drop(list);
        // Another translator: the module with inputs stays, to be
        // translated again; the one without is dropped; pipelines stay.
        let mut list = PrewarmList::load(Some(&bin), 2);
        assert_eq!(list.modules.len(), 1);
        let stale: Vec<u64> = list.stale().map(|m| m.key).collect();
        assert_eq!(stale, vec![10]);
        assert!(list.modules[0].spirv.is_empty() && list.modules[0].source.is_some());
        assert_eq!(list.compute.len(), 1);
        list.add_module(10, 3, vec![0x0723_0203, 20], r.clone(), None);
        assert_eq!(list.stale().count(), 0);
        // The inputs stay with the translated module.
        assert!(list.modules[0].source.is_some());
        drop(list);
        let list = PrewarmList::load(Some(&bin), 2);
        assert_eq!(list.modules.len(), 1);
        assert!(list.modules[0].fresh);
        assert_eq!(&list.modules[0].spirv[..], &[0x0723_0203, 20]);
        drop(list);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Loading a list leaves its file as it was: every pipeline record is
    /// read back after any number of loads (a load used to write the file
    /// from inside its loop, keeping only the first 64 graphics pipelines
    /// and no compute pipeline).
    #[test]
    fn loading_keeps_every_pipeline_of_the_file() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../local/tmp/prewarm-load-test-{}", std::process::id()));
        let bin = dir.join("TEST00003.bin");
        let graphics = |i: u64| GraphicsRecord { vs: i, ps: 0, layout: vec![], push_stages: 0, ff: ff(), mesh: false };
        {
            let mut list = PrewarmList::load(Some(&bin), 7);
            for i in 0..150 {
                list.add_graphics(graphics(i));
            }
            for i in 0..90 {
                list.add_compute(ComputeRecord { cs: i, layout: vec![], subgroup32: false });
            }
        }
        for _ in 0..3 {
            let list = PrewarmList::load(Some(&bin), 7);
            assert_eq!((list.graphics.len(), list.compute.len()), (150, 90));
        }
        let (_, _, g, c) = decode(&std::fs::read(list_path(&bin)).unwrap()).unwrap();
        assert_eq!((g.len(), c.len()), (150, 90), "the file after the loads");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Saves on their own thread write the records as they were when the
    /// save started, one save at a time, and never over a later save.
    #[test]
    fn background_saves_write_in_order() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../local/tmp/prewarm-bg-test-{}", std::process::id()));
        let bin = dir.join("TEST00004.bin");
        let compute = |i: u64| ComputeRecord { cs: i, layout: vec![], subgroup32: false };
        let mut list = PrewarmList::load(Some(&bin), 3);
        for round in 0..20u64 {
            for i in 0..10 {
                list.add_compute(compute(round * 10 + i));
            }
            list.add_module(round, 3, vec![0x0723_0203; 50_000], ShaderReflection::default(), None);
            list.save_in_background();
        }
        list.add_compute(compute(1000));
        // The last save waits for the ones still writing and is not undone
        // by them.
        list.save();
        let (_, m, _, c) = decode(&std::fs::read(list_path(&bin)).unwrap()).unwrap();
        assert_eq!((m.len(), c.len()), (20, 201));
        assert!(m.iter().all(|m| m.spirv.len() == 50_000));
        drop(list);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A list saves through a temporary file of its own (the pipeline
    /// cache next to it saves through `<title>.tmp<pid>`, possibly at the
    /// same time); loading removes the list's temporaries of processes that
    /// are gone and nothing of the cache's.
    #[test]
    fn list_temporaries_are_the_lists_own() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../local/tmp/prewarm-tmp-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = dir.join("TEST00005.bin");
        let path = list_path(&bin);
        let tmp = temporary_path(&path);
        assert_eq!(tmp.file_name().unwrap().to_str().unwrap(), format!("TEST00005.prewarm.tmp{}", std::process::id()));
        let dead_list = dir.join("TEST00005.prewarm.tmp4000000000");
        let dead_cache = dir.join("TEST00005.tmp4000000000");
        std::fs::write(&dead_list, b"x").unwrap();
        std::fs::write(&dead_cache, b"x").unwrap();
        drop(PrewarmList::load(Some(&bin), 1));
        assert!(!dead_list.exists() && dead_cache.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Lists of an older format are recognised and moved aside (never
    /// silently read as empty); a recompile marks every module of a current
    /// list for translation.
    #[test]
    fn old_lists_are_retired_and_recompile_retranslates_everything() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../local/tmp/prewarm-status-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = dir.join("TEST00002.bin");
        let path = list_path(&bin);
        assert_eq!(list_status(&path), ListStatus::Missing);
        std::fs::write(&path, super::super::cache::encode("rps5ePW3\n", &[1, 2, 3, 4])).unwrap();
        assert_eq!(list_status(&path), ListStatus::Old(3));
        let moved = retire_old_list(&path).expect("moved aside");
        assert!(moved.ends_with("TEST00002.prewarm.pw3.old"), "{moved:?}");
        assert_eq!(list_status(&path), ListStatus::Missing);
        std::fs::write(&path, b"garbage").unwrap();
        assert_eq!(list_status(&path), ListStatus::Damaged);
        assert!(retire_old_list(&path).is_none(), "only older formats are moved");
        std::fs::remove_file(&path).unwrap();
        let program = Arc::new(ProgramSource {
            address: 0x1000,
            continuation: 0,
            stage: 3,
            user_data_base: 0,
            user_data_count: 16,
            wave_size: 64,
            plan_hash: 1,
            guest_plan: false,
            image: Arc::new(super::super::sources::CodeImage::from_words(vec![(0x1000, 0xBF81_0000)])),
        });
        let source = Some(Arc::new(ModuleSource { program, request: vec![9].into(), reads: Default::default() }));
        {
            let mut list = PrewarmList::load(Some(&bin), 5);
            list.add_module(10, 3, vec![0x0723_0203, 10], ShaderReflection::default(), source.clone());
            list.add_module(11, 3, vec![0x0723_0203, 11], ShaderReflection::default(), source);
        }
        assert_eq!(list_status(&path), ListStatus::Current);
        assert_eq!(PrewarmList::load(Some(&bin), 5).stale().count(), 0);
        let mut keys = mark_for_recompile(&path).unwrap();
        keys.sort();
        assert_eq!(keys, vec![10, 11]);
        // The same translator now translates both again.
        let list = PrewarmList::load(Some(&bin), 5);
        assert_eq!(list.stale().count(), 2);
        drop(list);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
