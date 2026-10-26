// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! A [`ShaderCompiler`] for tests: small hand-built SPIR-V programs
//! registered per guest program address, emitted against the request's
//! real binding layout (descriptor bindings, push-data dwords, buffer
//! offset adjustments, pixel outputs), so the whole provider path can be
//! exercised on the device before the real translator exists.

use super::compiler::{layout_bindings, CompileError, CompiledShader, ComponentKind, ReflectedVariable, ShaderCompileInput, ShaderCompiler, ShaderReflection};
use ps5shader::resources::{native_binding_index, DescriptorBindingKind, ShaderCompileRequest, ShaderStage};
use ps5shader::PixelOutputKind;
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{self, BuiltIn, Decoration as D, StorageClass as S, Word};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// What a registered program does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TestProgram {
    /// `gl_Position = in[0]`; `out[i - 1] = in[i]` for `1 <= i < inputs`
    /// (all `vec4` attributes at locations `0..inputs`).
    PassthroughVertex { inputs: u32 },
    /// `gl_Position = in[0] + in[1]`; `out[0] = in[2]` (three `vec4`
    /// attributes, e.g. a per-instance offset and color).
    OffsetVertex,
    /// Every color output = the four user-data registers `s0..s3`
    /// (bit patterns: floats for float targets, integers otherwise).
    SolidPixel,
    /// Color output 0 = input location 0.
    VaryingPixel,
    /// Color output 0 = input location 0, flat (the provoking vertex's).
    FlatPixel,
    /// Color output 0 = the first image sampled with the first sampler at
    /// input location 0 `.xy`.
    TexturedPixel,
    /// `buffer0[gid.x] = s4 + gid.x` (32-bit words, honouring the
    /// buffer's offset adjustment).
    StoreCompute { local_size: u32 },
}

/// The test compiler: programs by guest address.
#[derive(Default)]
pub struct TestCompiler {
    programs: HashMap<u64, TestProgram>,
    /// Compilations performed (cache-miss counter for tests).
    pub compiled: Mutex<u64>,
    /// Compilations wait until the gate is open ([`TestCompiler::gated`]).
    gate: Option<Arc<CompileGate>>,
}

/// Holds compilations back (a slow compile, ended at a chosen moment).
#[derive(Default)]
pub struct CompileGate {
    open: Mutex<bool>,
    changed: std::sync::Condvar,
}

impl CompileGate {
    pub fn open(&self) {
        *self.open.lock().unwrap() = true;
        self.changed.notify_all();
    }

    pub fn is_open(&self) -> bool {
        *self.open.lock().unwrap()
    }

    fn pass(&self) {
        let mut open = self.open.lock().unwrap();
        while !*open {
            open = self.changed.wait(open).unwrap();
        }
    }
}

impl TestCompiler {
    pub fn new() -> TestCompiler {
        TestCompiler::default()
    }

    pub fn with(mut self, address: u64, program: TestProgram) -> TestCompiler {
        self.programs.insert(address, program);
        self
    }

    pub fn compiled(&self) -> u64 {
        *self.compiled.lock().unwrap()
    }

    /// Every compilation waits until `gate` opens.
    pub fn gated(mut self, gate: Arc<CompileGate>) -> TestCompiler {
        self.gate = Some(gate);
        self
    }
}

impl ShaderCompiler for TestCompiler {
    fn compile(&self, input: &ShaderCompileInput<'_, '_>) -> Result<CompiledShader, CompileError> {
        let program = *self.programs.get(&input.guest_address).ok_or_else(|| CompileError::Unsupported(format!("no test program at {:#x}", input.guest_address)))?;
        if let Some(g) = &self.gate {
            g.pass();
        }
        *self.compiled.lock().unwrap() += 1;
        let r = input.request;
        let stage = input.stage;
        let built = match program {
            TestProgram::PassthroughVertex { inputs } => passthrough_vertex(inputs, false),
            TestProgram::OffsetVertex => passthrough_vertex(3, true),
            TestProgram::SolidPixel => solid_pixel(r),
            TestProgram::VaryingPixel => varying_pixel(r, false),
            TestProgram::FlatPixel => varying_pixel(r, true),
            TestProgram::TexturedPixel => textured_pixel(r),
            TestProgram::StoreCompute { local_size } => store_compute(r, local_size),
        }
        .map_err(|e| CompileError::Failed(e.0))?;
        let (spirv, mut reflection) = built;
        reflection.bindings = layout_bindings(stage, r);
        Ok(CompiledShader { spirv, reflection, code_reads: Vec::new() })
    }

    fn name(&self) -> &str {
        "test-compiler"
    }
}

/// A build failure (rspirv error or a missing binding).
#[derive(Debug)]
struct E(String);

impl From<rspirv::dr::Error> for E {
    fn from(e: rspirv::dr::Error) -> E {
        E(format!("{e:?}"))
    }
}

type R<T> = Result<T, E>;

struct Spv {
    b: Builder,
    void: Word,
    f32: Word,
    u32: Word,
    i32: Word,
    vec4: Word,
    consts: HashMap<(Word, u32), Word>,
    interfaces: Vec<Word>,
    push: Option<Word>,
}

impl Spv {
    fn new() -> Spv {
        let mut b = Builder::new();
        b.set_version(1, 3);
        b.capability(spirv::Capability::Shader);
        b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
        let void = b.type_void();
        let f32 = b.type_float(32, None);
        let u32 = b.type_int(32, 0);
        let i32 = b.type_int(32, 1);
        let vec4 = b.type_vector(f32, 4);
        Spv { b, void, f32, u32, i32, vec4, consts: HashMap::new(), interfaces: Vec::new(), push: None }
    }

    fn c(&mut self, ty: Word, v: u32) -> Word {
        if let Some(&id) = self.consts.get(&(ty, v)) {
            return id;
        }
        let id = self.b.constant_bit32(ty, v);
        self.consts.insert((ty, v), id);
        id
    }

    fn cu(&mut self, v: u32) -> Word {
        let t = self.u32;
        self.c(t, v)
    }

    fn var(&mut self, ty: Word, class: S, interface: bool) -> Word {
        let p = self.b.type_pointer(None, class, ty);
        let v = self.b.variable(p, None, class, None);
        if interface {
            self.interfaces.push(v);
        }
        v
    }

    fn binding(&mut self, v: Word, binding: u32) {
        self.b.decorate(v, D::DescriptorSet, [Operand::LiteralBit32(0)]);
        self.b.decorate(v, D::Binding, [Operand::LiteralBit32(binding)]);
    }

    /// The 32-dword push-constant block (declare before `begin`).
    fn push_block(&mut self) -> Word {
        if let Some(p) = self.push {
            return p;
        }
        let n = self.cu(32);
        let arr = self.b.type_array(self.u32, n);
        self.b.decorate(arr, D::ArrayStride, [Operand::LiteralBit32(4)]);
        let id = self.b.id();
        let s = self.b.type_struct_id(Some(id), [arr]);
        self.b.decorate(s, D::Block, []);
        self.b.member_decorate(s, 0, D::Offset, [Operand::LiteralBit32(0)]);
        let v = self.var(s, S::PushConstant, false);
        self.push = Some(v);
        v
    }

    fn push_word(&mut self, dword: u32) -> R<Word> {
        let block = self.push_block();
        let p = self.b.type_pointer(None, S::PushConstant, self.u32);
        let (z, i) = (self.cu(0), self.cu(dword));
        let a = self.b.access_chain(p, None, block, [z, i])?;
        Ok(self.b.load(self.u32, None, a, None, [])?)
    }

    fn begin(&mut self) -> R<Word> {
        let fnty = self.b.type_function(self.void, []);
        let main = self.b.begin_function(self.void, None, spirv::FunctionControl::NONE, fnty)?;
        self.b.begin_block(None)?;
        Ok(main)
    }

    fn finish(mut self, main: Word, model: spirv::ExecutionModel, modes: &[(spirv::ExecutionMode, Vec<u32>)]) -> R<Vec<u32>> {
        self.b.ret()?;
        self.b.end_function()?;
        for (m, a) in modes {
            self.b.execution_mode(main, *m, a);
        }
        let interfaces = std::mem::take(&mut self.interfaces);
        self.b.entry_point(model, main, "main", interfaces);
        Ok(self.b.module().assemble())
    }
}

fn user_dword(r: &ShaderCompileRequest<'_>, register: u32) -> Option<u32> {
    let l = &r.bindings;
    if !l.uses_push_data() {
        return None;
    }
    let index = l.user_data_registers.iter().position(|&x| x == register)? as u32;
    Some(l.push_data_start_dword + index)
}

fn need<T>(v: Option<T>, what: &str) -> R<T> {
    v.ok_or_else(|| E(format!("the request lacks {what}")))
}

fn passthrough_vertex(inputs: u32, offset: bool) -> R<(Vec<u32>, ShaderReflection)> {
    let mut s = Spv::new();
    let id = s.b.id();
    let pv = s.b.type_struct_id(Some(id), [s.vec4]);
    s.b.member_decorate(pv, 0, D::BuiltIn, [Operand::BuiltIn(BuiltIn::Position)]);
    s.b.decorate(pv, D::Block, []);
    let out_pv = s.var(pv, S::Output, true);
    let mut ins = Vec::new();
    for l in 0..inputs.max(1) {
        let v = s.var(s.vec4, S::Input, true);
        s.b.decorate(v, D::Location, [Operand::LiteralBit32(l)]);
        ins.push(v);
    }
    let mut outs = Vec::new();
    let first_out = if offset { 2 } else { 1 };
    for l in first_out..inputs.max(1) {
        let v = s.var(s.vec4, S::Output, true);
        s.b.decorate(v, D::Location, [Operand::LiteralBit32(l - first_out)]);
        outs.push(v);
    }
    let main = s.begin()?;
    let mut pos = s.b.load(s.vec4, None, ins[0], None, [])?;
    if offset {
        let o = s.b.load(s.vec4, None, ins[1], None, [])?;
        pos = s.b.f_add(s.vec4, None, pos, o)?;
    }
    let p = s.b.type_pointer(None, S::Output, s.vec4);
    let z = s.c(s.i32, 0);
    let a = s.b.access_chain(p, None, out_pv, [z])?;
    s.b.store(a, pos, None, [])?;
    for (i, o) in outs.iter().enumerate() {
        let v = s.b.load(s.vec4, None, ins[i + first_out as usize], None, [])?;
        s.b.store(*o, v, None, [])?;
    }
    let words = s.finish(main, spirv::ExecutionModel::Vertex, &[])?;
    let reflection = ShaderReflection {
        vertex_inputs: (0..inputs.max(1)).map(|l| ReflectedVariable { location: l, components: 4, kind: ComponentKind::Float, flat: false }).collect(),
        outputs: (first_out..inputs.max(1)).map(|l| ReflectedVariable { location: l - first_out, components: 4, kind: ComponentKind::Float, flat: false }).collect(),
        ..ShaderReflection::default()
    };
    Ok((words, reflection))
}

/// Declare the pixel outputs of `r`; returns (variable, value type, kind).
fn pixel_outputs(s: &mut Spv, r: &ShaderCompileRequest<'_>) -> Vec<(Word, Word, PixelOutputKind)> {
    let mut v = Vec::new();
    for o in &r.pixel_outputs {
        let scalar = match o.kind {
            PixelOutputKind::Float => s.f32,
            PixelOutputKind::Uint => s.u32,
            PixelOutputKind::Sint => s.i32,
        };
        let ty = s.b.type_vector(scalar, 4);
        let var = s.var(ty, S::Output, true);
        s.b.decorate(var, D::Location, [Operand::LiteralBit32(o.host_location)]);
        v.push((var, ty, o.kind));
    }
    v
}

fn pixel_reflection(r: &ShaderCompileRequest<'_>) -> ShaderReflection {
    ShaderReflection {
        outputs: r
            .pixel_outputs
            .iter()
            .map(|o| ReflectedVariable {
                location: o.host_location,
                components: 4,
                kind: match o.kind {
                    PixelOutputKind::Float => ComponentKind::Float,
                    PixelOutputKind::Uint => ComponentKind::Uint,
                    PixelOutputKind::Sint => ComponentKind::Sint,
                },
                flat: false,
            })
            .collect(),
        ..ShaderReflection::default()
    }
}

fn fragment_modes(main: Word) -> Vec<(spirv::ExecutionMode, Vec<u32>)> {
    let _ = main;
    vec![(spirv::ExecutionMode::OriginUpperLeft, vec![])]
}

fn solid_pixel(r: &ShaderCompileRequest<'_>) -> R<(Vec<u32>, ShaderReflection)> {
    let mut s = Spv::new();
    let outs = pixel_outputs(&mut s, r);
    s.push_block();
    let main = s.begin()?;
    let mut words = Vec::new();
    for reg in 0..4 {
        let d = need(user_dword(r, reg), "user data s0..s3 in push data")?;
        words.push(s.push_word(d)?);
    }
    let uvec4 = s.b.type_vector(s.u32, 4);
    let raw = s.b.composite_construct(uvec4, None, words)?;
    for (var, ty, kind) in outs {
        let v = if kind == PixelOutputKind::Uint { raw } else { s.b.bitcast(ty, None, raw)? };
        s.b.store(var, v, None, [])?;
    }
    let modes = fragment_modes(main);
    Ok((s.finish(main, spirv::ExecutionModel::Fragment, &modes)?, pixel_reflection(r)))
}

fn varying_pixel(r: &ShaderCompileRequest<'_>, flat: bool) -> R<(Vec<u32>, ShaderReflection)> {
    let mut s = Spv::new();
    let outs = pixel_outputs(&mut s, r);
    let input = s.var(s.vec4, S::Input, true);
    s.b.decorate(input, D::Location, [Operand::LiteralBit32(0)]);
    if flat {
        s.b.decorate(input, D::Flat, []);
    }
    let main = s.begin()?;
    let v = s.b.load(s.vec4, None, input, None, [])?;
    if let Some(&(var, _, PixelOutputKind::Float)) = outs.first() {
        s.b.store(var, v, None, [])?;
    }
    let modes = fragment_modes(main);
    Ok((s.finish(main, spirv::ExecutionModel::Fragment, &modes)?, pixel_reflection(r)))
}

fn textured_pixel(r: &ShaderCompileRequest<'_>) -> R<(Vec<u32>, ShaderReflection)> {
    let stage = ShaderStage::Pixel;
    let image_binding = need(r.bindings.descriptors.iter().find(|b| matches!(b.kind, DescriptorBindingKind::Image(_))), "an image binding")?;
    let sampler_binding = need(r.bindings.find(DescriptorBindingKind::Samplers), "a sampler binding")?;
    let mut s = Spv::new();
    let outs = pixel_outputs(&mut s, r);
    let input = s.var(s.vec4, S::Input, true);
    s.b.decorate(input, D::Location, [Operand::LiteralBit32(0)]);
    let image_t = s.b.type_image(s.f32, spirv::Dim::Dim2D, 0, 0, 0, 1, spirv::ImageFormat::Unknown, None);
    let n_img = s.cu(image_binding.resources.len().max(1) as u32);
    let image_arr = s.b.type_array(image_t, n_img);
    let images = s.var(image_arr, S::UniformConstant, false);
    s.binding(images, native_binding_index(stage, image_binding.kind));
    let sampler_t = s.b.type_sampler();
    let n_smp = s.cu(sampler_binding.resources.len().max(1) as u32);
    let sampler_arr = s.b.type_array(sampler_t, n_smp);
    let samplers = s.var(sampler_arr, S::UniformConstant, false);
    s.binding(samplers, native_binding_index(stage, DescriptorBindingKind::Samplers));
    let main = s.begin()?;
    let z = s.cu(0);
    let pi = s.b.type_pointer(None, S::UniformConstant, image_t);
    let ai = s.b.access_chain(pi, None, images, [z])?;
    let img = s.b.load(image_t, None, ai, None, [])?;
    let ps = s.b.type_pointer(None, S::UniformConstant, sampler_t);
    let a_s = s.b.access_chain(ps, None, samplers, [z])?;
    let smp = s.b.load(sampler_t, None, a_s, None, [])?;
    let si_t = s.b.type_sampled_image(image_t);
    let si = s.b.sampled_image(si_t, None, img, smp)?;
    let uv4 = s.b.load(s.vec4, None, input, None, [])?;
    let vec2 = s.b.type_vector(s.f32, 2);
    let uv = s.b.vector_shuffle(vec2, None, uv4, uv4, [0, 1])?;
    let texel = s.b.image_sample_implicit_lod(s.vec4, None, si, uv, None, [])?;
    if let Some(&(var, _, PixelOutputKind::Float)) = outs.first() {
        s.b.store(var, texel, None, [])?;
    }
    let modes = fragment_modes(main);
    Ok((s.finish(main, spirv::ExecutionModel::Fragment, &modes)?, pixel_reflection(r)))
}

fn store_compute(r: &ShaderCompileRequest<'_>, local_size: u32) -> R<(Vec<u32>, ShaderReflection)> {
    let stage = ShaderStage::Compute;
    let buffers = need(r.bindings.find(DescriptorBindingKind::Buffers), "a buffer binding")?;
    let s4 = need(user_dword(r, 4), "user data s4 in push data")?;
    let offsets = need(r.bindings.uses_push_data().then_some(r.bindings.push_data_start_dword + r.bindings.memory_offset_dword), "push data")?;
    let mut s = Spv::new();
    let rta = s.b.type_runtime_array(s.u32);
    s.b.decorate(rta, D::ArrayStride, [Operand::LiteralBit32(4)]);
    let id = s.b.id();
    let block = s.b.type_struct_id(Some(id), [rta]);
    s.b.decorate(block, D::Block, []);
    s.b.member_decorate(block, 0, D::Offset, [Operand::LiteralBit32(0)]);
    let n = s.cu(buffers.resources.len().max(1) as u32);
    let arr = s.b.type_array(block, n);
    let buf = s.var(arr, S::StorageBuffer, false);
    s.binding(buf, native_binding_index(stage, DescriptorBindingKind::Buffers));
    let uvec3 = s.b.type_vector(s.u32, 3);
    let gid = s.var(uvec3, S::Input, true);
    s.b.decorate(gid, D::BuiltIn, [Operand::BuiltIn(BuiltIn::GlobalInvocationId)]);
    s.push_block();
    let main = s.begin()?;
    let g = s.b.load(uvec3, None, gid, None, [])?;
    let x = s.b.composite_extract(s.u32, None, g, [0])?;
    let base = s.push_word(s4)?;
    let value = s.b.i_add(s.u32, None, base, x)?;
    let packed = s.push_word(offsets)?;
    let mask = s.cu(0xFF);
    let adjust = s.b.bitwise_and(s.u32, None, packed, mask)?;
    let two = s.cu(2);
    let adjust_words = s.b.shift_right_logical(s.u32, None, adjust, two)?;
    let index = s.b.i_add(s.u32, None, adjust_words, x)?;
    let p = s.b.type_pointer(None, S::StorageBuffer, s.u32);
    let z = s.cu(0);
    let a = s.b.access_chain(p, None, buf, [z, z, index])?;
    s.b.store(a, value, None, [])?;
    let words = s.finish(main, spirv::ExecutionModel::GLCompute, &[(spirv::ExecutionMode::LocalSize, vec![local_size, 1, 1])])?;
    Ok((words, ShaderReflection { local_size: [local_size, 1, 1], ..ShaderReflection::default() }))
}

/// Tiny RDNA assembler for test programs.
pub mod asm {
    pub const S_ENDPGM: u32 = 0xBF81_0000;

    /// v_mov_b32 vdst, s<src>.
    pub fn v_mov_from_sgpr(vdst: u32, sgpr: u32) -> u32 {
        0x7E00_0000 | (vdst << 17) | (1 << 9) | sgpr
    }

    /// image_sample vdata, vaddr, srsrc, ssamp (2D, dmask 0xF).
    pub fn image_sample(vdata: u32, vaddr: u32, srsrc: u32, ssamp: u32) -> [u32; 2] {
        [0xF080_0F08, vaddr | (vdata << 8) | ((srsrc / 4) << 16) | ((ssamp / 4) << 21)]
    }

    /// buffer_store_dword vdata, vaddr, srsrc, 0 offen.
    pub fn buffer_store_dword(vdata: u32, vaddr: u32, srsrc: u32) -> [u32; 2] {
        [0xE000_0000 | (0x1C << 18) | (1 << 12), vaddr | (vdata << 8) | ((srsrc / 4) << 16) | (0x80 << 24)]
    }

    pub fn bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}
