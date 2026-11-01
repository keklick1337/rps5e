// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! What the host device does with f32 denormals in a module without an
//! f32 denormal execution mode (the modules the translator emits where the
//! device has neither DenormPreserve(32) nor DenormFlushToZero(32)).
//!
//! The guest flushes f32 denormals per FLOAT_MODE and the translator
//! emulates that flush. [`measure`] runs the host operations the flush
//! elision relies on (`ps5shader::opt::denorm::HostFloat`) over denormal
//! operands and operands whose exact results are denormal, once per device:
//! the host flushes when every result equals the result on the operands
//! flushed in integer code and no result is a denormal.
//!
//! The tests print the float-controls properties and a table of f32
//! operations with no mode, DenormPreserve(32) and DenormFlushToZero(32)
//! (`cargo test -p ps5gpu --release -- --ignored host_denormals --nocapture`).

use crate::device::{Buffer, GpuDevice, MemoryUse};
use crate::tiler::{Spv, R};
use ash::vk;
use rspirv::dr::Operand;
use rspirv::spirv;
use ps5shader::opt::denorm::HostFloat;
use rspirv::binary::Assemble;
use std::sync::Arc;

/// The f32 denormal behaviour a module asks for.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Mode {
    /// No denormal execution mode (implementation-defined).
    Default,
    Preserve,
    FlushToZero,
}

/// Operations of the table, in output order.
#[cfg(test)]
pub(crate) const OPS: &[&str] = &[
    "fadd", "fsub", "fmul", "fma", "fmin", "fmax", "fneg", "fabs", "rcp", "sqrt", "floor", "fract", "f64rt", "lt", "eq", "clamp", "select", "copy", "exp2", "log2",
    "rsq", "ldexp", "mul+add", "lt0", "trunc", "nmin", "f16rt", "div", "mul1", "mul-1", "add0", "add-0", "sub0", "fma1-0", "fma1+0", "minxx", "maxxx", "mul.5", "ceil",
];

/// One row of three inputs per invocation; `OPS.len()` result words per row.
#[cfg(test)]
fn kernel(mode: Mode, nan_preserve: bool, rows: u32) -> R<Vec<u32>> {
    use spirv::GlslStd450Op as G;
    let mut s = Spv::new();
    s.b.capability(spirv::Capability::Float64);
    s.b.capability(spirv::Capability::Float16);
    match mode {
        Mode::Default => {}
        Mode::Preserve => s.b.capability(spirv::Capability::DenormPreserve),
        Mode::FlushToZero => s.b.capability(spirv::Capability::DenormFlushToZero),
    }
    if nan_preserve {
        s.b.capability(spirv::Capability::SignedZeroInfNanPreserve);
    }
    if mode != Mode::Default || nan_preserve {
        s.b.extension("SPV_KHR_float_controls");
    }
    let glsl = s.b.ext_inst_import("GLSL.std.450");
    let (main, x, _, _) = s.begin_main()?;
    let f32t = s.b.type_float(32, None);
    let f64t = s.b.type_float(64, None);
    let f16t = s.b.type_float(16, None);
    let i32t = s.b.type_int(32, 1);
    let rows_c = s.c(rows);
    let past = s.b.u_greater_than_equal(s.boolean, None, x, rows_c)?;
    s.return_when(past)?;
    let base = s.mul_c(x, 3)?;
    let mut input = Vec::new();
    for k in 0..3 {
        let i = s.add_c(base, k)?;
        let w = s.load_word(s.src, i)?;
        input.push(s.b.bitcast(f32t, None, w)?);
    }
    let (a, b, c) = (input[0], input[1], input[2]);
    let one = s.b.constant_bit32(f32t, 1.0f32.to_bits());
    let zero = s.b.constant_bit32(f32t, 0);
    let ext = |s: &mut Spv, op: G, args: &[u32]| s.b.ext_inst(f32t, None, glsl, op as u32, args.iter().map(|&i| Operand::IdRef(i)));
    let mut out = Vec::new();
    out.push(s.b.f_add(f32t, None, a, b)?);
    out.push(s.b.f_sub(f32t, None, a, b)?);
    out.push(s.b.f_mul(f32t, None, a, b)?);
    out.push(ext(&mut s, G::Fma, &[a, b, c])?);
    out.push(ext(&mut s, G::FMin, &[a, b])?);
    out.push(ext(&mut s, G::FMax, &[a, b])?);
    out.push(s.b.f_negate(f32t, None, a)?);
    out.push(ext(&mut s, G::FAbs, &[a])?);
    out.push(s.b.f_div(f32t, None, one, a)?);
    out.push(ext(&mut s, G::Sqrt, &[a])?);
    out.push(ext(&mut s, G::Floor, &[a])?);
    out.push(ext(&mut s, G::Fract, &[a])?);
    let wide = s.b.f_convert(f64t, None, a)?;
    out.push(s.b.f_convert(f32t, None, wide)?);
    let bool_word = |s: &mut Spv, cond: u32| {
        let (t, f) = (s.c(1), s.c(0));
        s.select(cond, t, f)
    };
    let lt = s.b.f_ord_less_than(s.boolean, None, a, b)?;
    let lt = bool_word(&mut s, lt)?;
    let eq = s.b.f_ord_equal(s.boolean, None, a, b)?;
    let eq = bool_word(&mut s, eq)?;
    let mut words = Vec::new();
    for v in out.drain(..) {
        words.push(s.b.bitcast(s.uint, None, v)?);
    }
    words.push(lt);
    words.push(eq);
    let f = ext(&mut s, G::FClamp, &[a, b, c])?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    let pick = s.b.f_ord_greater_than(s.boolean, None, c, zero)?;
    let f = s.b.select(f32t, None, pick, b, a)?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    let f = s.b.copy_object(f32t, None, a)?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    for op in [G::Exp2, G::Log2, G::InverseSqrt] {
        let f = ext(&mut s, op, &[a])?;
        words.push(s.b.bitcast(s.uint, None, f)?);
    }
    // ldexp(a, -(bits of c) as i32): exponent from c.
    let e = s.b.bitcast(i32t, None, c)?;
    let f = s.b.ext_inst(f32t, None, glsl, G::Ldexp as u32, [Operand::IdRef(a), Operand::IdRef(e)])?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    let p = s.b.f_mul(f32t, None, a, b)?;
    let f = s.b.f_add(f32t, None, p, c)?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    let lt0 = s.b.f_ord_less_than(s.boolean, None, a, zero)?;
    words.push(bool_word(&mut s, lt0)?);
    let f = ext(&mut s, G::Trunc, &[a])?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    let f = ext(&mut s, G::NMin, &[a, b])?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    let h = s.b.f_convert(f16t, None, a)?;
    let f = s.b.f_convert(f32t, None, h)?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    let f = s.b.f_div(f32t, None, a, b)?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    // Operations with constants a compiler may fold to an identity.
    let m_one = s.b.constant_bit32(f32t, (-1.0f32).to_bits());
    let m_zero = s.b.constant_bit32(f32t, 0x8000_0000);
    let half = s.b.constant_bit32(f32t, 0.5f32.to_bits());
    let f = s.b.f_mul(f32t, None, a, one)?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    let f = s.b.f_mul(f32t, None, a, m_one)?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    let f = s.b.f_add(f32t, None, a, zero)?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    let f = s.b.f_add(f32t, None, a, m_zero)?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    let f = s.b.f_sub(f32t, None, a, zero)?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    let f = ext(&mut s, G::Fma, &[a, one, m_zero])?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    let f = ext(&mut s, G::Fma, &[a, one, zero])?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    let f = ext(&mut s, G::FMin, &[a, a])?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    let f = ext(&mut s, G::FMax, &[a, a])?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    let f = s.b.f_mul(f32t, None, a, half)?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    let f = ext(&mut s, G::Ceil, &[a])?;
    words.push(s.b.bitcast(s.uint, None, f)?);
    assert_eq!(words.len(), OPS.len());
    let out_base = s.mul_c(x, OPS.len() as u32)?;
    for (k, w) in words.into_iter().enumerate() {
        let i = s.add_c(out_base, k as u32)?;
        s.store_word(s.dst, i, w)?;
    }
    match mode {
        Mode::Default => {}
        Mode::Preserve => s.b.execution_mode(main, spirv::ExecutionMode::DenormPreserve, [32]),
        Mode::FlushToZero => s.b.execution_mode(main, spirv::ExecutionMode::DenormFlushToZero, [32]),
    }
    if nan_preserve {
        s.b.execution_mode(main, spirv::ExecutionMode::SignedZeroInfNanPreserve, [32]);
    }
    s.finish(main, 64, 1)
}

/// Run the table kernel over `rows` (three input words each) and return
/// `OPS.len()` words per row.
#[cfg(test)]
pub(crate) fn run(dev: &Arc<GpuDevice>, mode: Mode, nan_preserve: bool, rows: &[[u32; 3]]) -> Result<Vec<Vec<u32>>, String> {
    let code = kernel(mode, nan_preserve, rows.len() as u32).map_err(|e| format!("kernel: {e}"))?;
    let input: Vec<u32> = rows.iter().flatten().copied().collect();
    let words = dispatch(dev, &code, &input, rows.len() * OPS.len(), rows.len().div_ceil(64) as u32)?;
    Ok(words.chunks(OPS.len()).map(|c| c.to_vec()).collect())
}

/// Run the compute module `code` (input buffer at binding 0, output at 1)
/// over `groups` work-groups and return `out_words` output words.
fn dispatch(dev: &Arc<GpuDevice>, code: &[u32], input_words: &[u32], out_words: usize, groups: u32) -> Result<Vec<u32>, String> {
    let input = Buffer::new(dev, (input_words.len() * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryUse::Download).map_err(|e| format!("{e:?}"))?;
    let output = Buffer::new(dev, (out_words * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryUse::Download).map_err(|e| format!("{e:?}"))?;
    for (i, w) in input_words.iter().enumerate() {
        input.bytes_mut()[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    input.flush(0, vk::WHOLE_SIZE);
    let d = &dev.device;
    let bindings = [0u32, 1].map(|i| {
        vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)
    });
    // SAFETY: valid create infos; every object is destroyed before return,
    // after the fence wait (an error path leaks only what device teardown
    // reclaims).
    unsafe {
        let set_layout = d
            .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR).bindings(&bindings), None)
            .map_err(|e| format!("{e:?}"))?;
        let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: (crate::tiler::ARGUMENT_COUNT * 4) as u32 }];
        let layout = d
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set_layout)).push_constant_ranges(&ranges), None)
            .map_err(|e| format!("{e:?}"))?;
        let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None).map_err(|e| format!("{e:?}"))?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let r = d.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None);
        d.destroy_shader_module(module, None);
        let pipeline = r.map_err(|(_, e)| format!("pipeline: {e:?}"))?[0];
        let pool = d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(dev.queue_family), None).map_err(|e| format!("{e:?}"))?;
        let cb = d
            .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))
            .map_err(|e| format!("{e:?}"))?[0];
        let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("{e:?}"))?;
        d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default()).map_err(|e| format!("{e:?}"))?;
        d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
        let infos = [input.buffer, output.buffer].map(|buffer| vk::DescriptorBufferInfo { buffer, offset: 0, range: vk::WHOLE_SIZE });
        let writes = [0usize, 1].map(|i| vk::WriteDescriptorSet::default().dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&infos[i])));
        dev.push_descriptor.cmd_push_descriptor_set(cb, vk::PipelineBindPoint::COMPUTE, layout, 0, &writes);
        d.cmd_dispatch(cb, groups, 1, 1);
        let barrier = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
        d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[barrier], &[], &[]);
        d.end_command_buffer(cb).map_err(|e| format!("{e:?}"))?;
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cb));
        let submitted = {
            let queue = dev.queue();
            d.queue_submit(*queue, &[submit], fence)
        };
        let waited = submitted.and_then(|_| d.wait_for_fences(&[fence], true, u64::MAX));
        output.invalidate(0, vk::WHOLE_SIZE);
        let bytes = output.bytes_mut();
        let words: Vec<u32> = (0..out_words).map(|i| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())).collect();
        d.destroy_fence(fence, None);
        d.destroy_command_pool(pool, None);
        d.destroy_pipeline(pipeline, None);
        d.destroy_pipeline_layout(layout, None);
        d.destroy_descriptor_set_layout(set_layout, None);
        waited.map_err(|e| format!("{e:?}"))?;
        Ok(words)
    }
}

/// The host operations the flush elision relies on, in output order.
pub(crate) const CHECKED: &[&str] = &["add", "sub", "mul", "div", "fma", "rcp", "sqrt", "rsq", "log2"];

/// The float execution modes of a module (as the emitter declares them).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Modes {
    /// SignedZeroInfNanPreserve for 16/32/64-bit floats.
    pub nan_preserve: [bool; 3],
    /// DenormPreserve(16).
    pub preserve16: bool,
    /// RoundingModeRTE(16).
    pub rte16: bool,
}

/// For every row of three f32 operands: each [`CHECKED`] operation on the
/// operands and on the operands flushed in integer code (two words each).
/// A compute module (64 rows per work-group), a vertex module (row = the
/// vertex index) or the fragment module of a `width`-wide target whose
/// pixel (x, y) does row `y * width + x`.
fn check_kernel(rows: u32, modes: Modes, stage: Stage) -> R<Vec<u32>> {
    use spirv::GlslStd450Op as G;
    let mut s = Spv::new();
    s.b.capability(spirv::Capability::Float16);
    s.b.capability(spirv::Capability::Float64);
    if modes.nan_preserve.iter().any(|&b| b) {
        s.b.capability(spirv::Capability::SignedZeroInfNanPreserve);
    }
    if modes.preserve16 {
        s.b.capability(spirv::Capability::DenormPreserve);
    }
    if modes.rte16 {
        s.b.capability(spirv::Capability::RoundingModeRTE);
    }
    if modes != Modes::default() {
        s.b.extension("SPV_KHR_float_controls");
    }
    let glsl = s.b.ext_inst_import("GLSL.std.450");
    let f32t = s.b.type_float(32, None);
    let (main, x, coord) = begin_stage(&mut s, stage, f32t)?;
    let rows_c = s.c(rows);
    let past = s.b.u_greater_than_equal(s.boolean, None, x, rows_c)?;
    s.return_when(past)?;
    let base = s.mul_c(x, 3)?;
    let one = s.b.constant_bit32(f32t, 1.0f32.to_bits());
    let mut raw = Vec::new();
    let mut flushed = Vec::new();
    for k in 0..3 {
        let i = s.add_c(base, k)?;
        let w = s.load_word(s.src, i)?;
        raw.push(s.b.bitcast(f32t, None, w)?);
        let mag = s.and_c(w, 0x7FFF_FFFF)?;
        let sign = s.and_c(w, 0x8000_0000)?;
        let lim = s.c(0x0080_0000);
        let small = s.b.u_less_than(s.boolean, None, mag, lim)?;
        let f = s.select(small, sign, w)?;
        flushed.push(s.b.bitcast(f32t, None, f)?);
    }
    let mut words = Vec::new();
    for v in [&raw, &flushed] {
        let (a, b, c) = (v[0], v[1], v[2]);
        let ext = |s: &mut Spv, op: G, args: &[u32]| s.b.ext_inst(f32t, None, glsl, op as u32, args.iter().map(|&i| Operand::IdRef(i)));
        let r = [
            s.b.f_add(f32t, None, a, b)?,
            s.b.f_sub(f32t, None, a, b)?,
            s.b.f_mul(f32t, None, a, b)?,
            s.b.f_div(f32t, None, a, b)?,
            ext(&mut s, G::Fma, &[a, b, c])?,
            s.b.f_div(f32t, None, one, a)?,
            ext(&mut s, G::Sqrt, &[a])?,
            ext(&mut s, G::InverseSqrt, &[a])?,
            ext(&mut s, G::Log2, &[a])?,
        ];
        for (k, &id) in r.iter().enumerate() {
            if k < 4 {
                s.b.decorate(id, spirv::Decoration::NoContraction, []);
            }
        }
        words.push(r);
    }
    let n = CHECKED.len() as u32;
    let out_base = s.mul_c(x, 2 * n)?;
    for k in 0..n as usize {
        for (j, set) in words.iter().enumerate() {
            let w = s.b.bitcast(s.uint, None, set[k])?;
            let i = s.add_c(out_base, 2 * k as u32 + j as u32)?;
            s.store_word(s.dst, i, w)?;
        }
    }
    for (i, width) in [16, 32, 64].into_iter().enumerate() {
        if modes.nan_preserve[i] {
            s.b.execution_mode(main, spirv::ExecutionMode::SignedZeroInfNanPreserve, [width]);
        }
    }
    if modes.preserve16 {
        s.b.execution_mode(main, spirv::ExecutionMode::DenormPreserve, [16]);
    }
    if modes.rte16 {
        s.b.execution_mode(main, spirv::ExecutionMode::RoundingModeRTE, [16]);
    }
    finish_stage(s, stage, main, coord)
}

/// The entry function of a probe module of `stage`: (function, row index,
/// the input variable of the row's built-in outside compute).
fn begin_stage(s: &mut Spv, stage: Stage, f32t: u32) -> R<(u32, u32, Option<u32>)> {
    Ok(match stage {
        Stage::Compute => {
            let (main, x, _, _) = s.begin_main()?;
            (main, x, None)
        }
        Stage::Vertex => {
            let i32t = s.b.type_int(32, 1);
            let p_in = s.b.type_pointer(None, spirv::StorageClass::Input, i32t);
            let index = s.b.variable(p_in, None, spirv::StorageClass::Input, None);
            s.b.decorate(index, spirv::Decoration::BuiltIn, [Operand::BuiltIn(spirv::BuiltIn::VertexIndex)]);
            let void = s.b.type_void();
            let fnty = s.b.type_function(void, []);
            let main = s.b.begin_function(void, None, spirv::FunctionControl::NONE, fnty)?;
            s.b.begin_block(None)?;
            let i = s.b.load(i32t, None, index, None, [])?;
            (main, s.b.bitcast(s.uint, None, i)?, Some(index))
        }
        Stage::Fragment(width) => {
            let v4 = s.b.type_vector(f32t, 4);
            let p_in = s.b.type_pointer(None, spirv::StorageClass::Input, v4);
            let coord = s.b.variable(p_in, None, spirv::StorageClass::Input, None);
            s.b.decorate(coord, spirv::Decoration::BuiltIn, [Operand::BuiltIn(spirv::BuiltIn::FragCoord)]);
            let void = s.b.type_void();
            let fnty = s.b.type_function(void, []);
            let main = s.b.begin_function(void, None, spirv::FunctionControl::NONE, fnty)?;
            s.b.begin_block(None)?;
            let c = s.b.load(v4, None, coord, None, [])?;
            let fx = s.b.composite_extract(f32t, None, c, [0])?;
            let fy = s.b.composite_extract(f32t, None, c, [1])?;
            let px = s.b.convert_f_to_u(s.uint, None, fx)?;
            let py = s.b.convert_f_to_u(s.uint, None, fy)?;
            let line = s.mul_c(py, width)?;
            (main, s.add(line, px)?, Some(coord))
        }
    })
}

/// End the probe module begun by [`begin_stage`].
fn finish_stage(mut s: Spv, stage: Stage, main: u32, coord: Option<u32>) -> R<Vec<u32>> {
    match (stage, coord) {
        (Stage::Compute, _) | (_, None) => s.finish(main, 64, 1),
        (Stage::Vertex, Some(index)) => {
            s.b.ret()?;
            s.b.end_function()?;
            s.b.entry_point(spirv::ExecutionModel::Vertex, main, "main", [index]);
            Ok(s.b.module().assemble())
        }
        (Stage::Fragment(_), Some(coord)) => {
            s.b.ret()?;
            s.b.end_function()?;
            s.b.execution_mode(main, spirv::ExecutionMode::OriginUpperLeft, []);
            s.b.entry_point(spirv::ExecutionModel::Fragment, main, "main", [coord]);
            Ok(s.b.module().assemble())
        }
    }
}

/// The shader stage a probe runs in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Stage {
    Compute,
    Vertex,
    /// A target of this width.
    Fragment(u32),
}

/// A vertex module drawing one triangle over the whole target.
fn cover_vertex() -> R<Vec<u32>> {
    use rspirv::dr::Builder;
    use spirv::{Decoration as D, StorageClass as S};
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(spirv::Capability::Shader);
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let f32t = b.type_float(32, None);
    let i32t = b.type_int(32, 1);
    let boolean = b.type_bool();
    let v4 = b.type_vector(f32t, 4);
    let p_in = b.type_pointer(None, S::Input, i32t);
    let p_out = b.type_pointer(None, S::Output, v4);
    let index = b.variable(p_in, None, S::Input, None);
    b.decorate(index, D::BuiltIn, [Operand::BuiltIn(spirv::BuiltIn::VertexIndex)]);
    let position = b.variable(p_out, None, S::Output, None);
    b.decorate(position, D::BuiltIn, [Operand::BuiltIn(spirv::BuiltIn::Position)]);
    let void = b.type_void();
    let fnty = b.type_function(void, []);
    let main = b.begin_function(void, None, spirv::FunctionControl::NONE, fnty)?;
    b.begin_block(None)?;
    let i = b.load(i32t, None, index, None, [])?;
    let (one, two) = (b.constant_bit32(i32t, 1), b.constant_bit32(i32t, 2));
    let (low, high) = (b.constant_bit32(f32t, (-1.0f32).to_bits()), b.constant_bit32(f32t, 3.0f32.to_bits()));
    let (zero, w) = (b.constant_bit32(f32t, 0), b.constant_bit32(f32t, 1.0f32.to_bits()));
    let is1 = b.i_equal(boolean, None, i, one)?;
    let is2 = b.i_equal(boolean, None, i, two)?;
    let x = b.select(f32t, None, is1, high, low)?;
    let y = b.select(f32t, None, is2, high, low)?;
    let p = b.composite_construct(v4, None, [x, y, zero, w])?;
    b.store(position, p, None, [])?;
    b.ret()?;
    b.end_function()?;
    b.entry_point(spirv::ExecutionModel::Vertex, main, "main", [index, position]);
    Ok(b.module().assemble())
}

/// Draw [`cover_vertex`] and the fragment module `fs` (input buffer at
/// binding 0, output at 1) over a `width` x `height` target without
/// attachments, or with `points` that many points of the vertex module
/// `fs` alone (rasterizer discard), and return `out_words` output words.
fn draw(dev: &Arc<GpuDevice>, fs: &[u32], input_words: &[u32], out_words: usize, width: u32, height: u32, points: Option<u32>) -> Result<Vec<u32>, String> {
    let vs = match points {
        Some(_) => fs.to_vec(),
        None => cover_vertex().map_err(|e| format!("vertex: {e}"))?,
    };
    let input = Buffer::new(dev, (input_words.len() * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryUse::Download).map_err(|e| format!("{e:?}"))?;
    let output = Buffer::new(dev, (out_words * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryUse::Download).map_err(|e| format!("{e:?}"))?;
    for (i, w) in input_words.iter().enumerate() {
        input.bytes_mut()[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    input.flush(0, vk::WHOLE_SIZE);
    let d = &dev.device;
    let stages = vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX;
    let bindings = [0u32, 1].map(|i| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(stages));
    // SAFETY: valid create infos; every object is destroyed before return,
    // after the fence wait (an error path leaks only what device teardown
    // reclaims).
    unsafe {
        let set_layout = d
            .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR).bindings(&bindings), None)
            .map_err(|e| format!("{e:?}"))?;
        let ranges = [vk::PushConstantRange { stage_flags: stages, offset: 0, size: (crate::tiler::ARGUMENT_COUNT * 4) as u32 }];
        let layout = d
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set_layout)).push_constant_ranges(&ranges), None)
            .map_err(|e| format!("{e:?}"))?;
        let vm = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vs), None).map_err(|e| format!("{e:?}"))?;
        let fm = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(fs), None).map_err(|e| format!("{e:?}"))?;
        let all_stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vm).name(c"main"),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(fm).name(c"main"),
        ];
        let shader_stages = if points.is_some() { &all_stages[..1] } else { &all_stages[..] };
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let topology = if points.is_some() { vk::PrimitiveTopology::POINT_LIST } else { vk::PrimitiveTopology::TRIANGLE_LIST };
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(topology);
        let viewports = [vk::Viewport { x: 0.0, y: 0.0, width: width as f32, height: height as f32, min_depth: 0.0, max_depth: 1.0 }];
        let scissors = [vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width, height } }];
        let viewport = vk::PipelineViewportStateCreateInfo::default().viewports(&viewports).scissors(&scissors);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0)
            .rasterizer_discard_enable(points.is_some());
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend = vk::PipelineColorBlendStateCreateInfo::default();
        let mut rendering = vk::PipelineRenderingCreateInfo::default();
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .layout(layout)
            .push_next(&mut rendering);
        let r = d.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None);
        d.destroy_shader_module(vm, None);
        d.destroy_shader_module(fm, None);
        let pipeline = r.map_err(|(_, e)| format!("pipeline: {e:?}"))?[0];
        let pool = d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(dev.queue_family), None).map_err(|e| format!("{e:?}"))?;
        let cb = d
            .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))
            .map_err(|e| format!("{e:?}"))?[0];
        let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("{e:?}"))?;
        d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default()).map_err(|e| format!("{e:?}"))?;
        let infos = [input.buffer, output.buffer].map(|buffer| vk::DescriptorBufferInfo { buffer, offset: 0, range: vk::WHOLE_SIZE });
        let writes = [0usize, 1].map(|i| vk::WriteDescriptorSet::default().dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&infos[i])));
        d.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(scissors[0]).layer_count(1));
        d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
        dev.push_descriptor.cmd_push_descriptor_set(cb, vk::PipelineBindPoint::GRAPHICS, layout, 0, &writes);
        d.cmd_draw(cb, points.unwrap_or(3), 1, 0, 0);
        d.cmd_end_rendering(cb);
        let barrier = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
        d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::VERTEX_SHADER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[barrier], &[], &[]);
        d.end_command_buffer(cb).map_err(|e| format!("{e:?}"))?;
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cb));
        let submitted = {
            let queue = dev.queue();
            d.queue_submit(*queue, &[submit], fence)
        };
        let waited = submitted.and_then(|_| d.wait_for_fences(&[fence], true, u64::MAX));
        output.invalidate(0, vk::WHOLE_SIZE);
        let bytes = output.bytes_mut();
        let words: Vec<u32> = (0..out_words).map(|i| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())).collect();
        d.destroy_fence(fence, None);
        d.destroy_command_pool(pool, None);
        d.destroy_pipeline(pipeline, None);
        d.destroy_pipeline_layout(layout, None);
        d.destroy_descriptor_set_layout(set_layout, None);
        waited.map_err(|e| format!("{e:?}"))?;
        Ok(words)
    }
}

fn is_denormal(bits: u32) -> bool {
    bits & 0x7F80_0000 == 0 && bits & 0x007F_FFFF != 0
}

fn is_nan(bits: u32) -> bool {
    bits & 0x7FFF_FFFF > 0x7F80_0000
}

/// Rows of three operands: the hand-picked [`ROWS`] and `random` rows from
/// a fixed seed with exponents spread over the denormals, the small
/// normals (products and quotients that underflow), and the whole range.
pub(crate) fn probe_rows(random: usize, seed: u64) -> Vec<[u32; 3]> {
    let mut rows: Vec<[u32; 3]> = ROWS.to_vec();
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let word = |r: u64| -> u32 {
        let sign = ((r >> 63) as u32) << 31;
        let mantissa = (r as u32) & 0x7F_FFFF;
        let exponent = match (r >> 32) % 4 {
            0 => 0,
            1 => 1 + ((r >> 40) % 40) as u32,
            2 => 40 + ((r >> 40) % 90) as u32,
            _ => ((r >> 40) % 255) as u32,
        };
        sign | exponent << 23 | mantissa
    };
    for _ in 0..random {
        let (a, b, c) = (next(), next(), next());
        rows.push([word(a), word(b), word(c)]);
    }
    rows
}

/// The rows (operation, operands, raw result, result on flushed operands)
/// where the host does not flush; empty when it does.
pub(crate) fn check(dev: &Arc<GpuDevice>, rows: &[[u32; 3]], modes: Modes) -> Result<Vec<String>, String> {
    check_in(dev, rows, modes, Stage::Compute)
}

/// [`check`] in a module of `stage` (the fragment width is chosen here).
pub(crate) fn check_in(dev: &Arc<GpuDevice>, rows: &[[u32; 3]], modes: Modes, stage: Stage) -> Result<Vec<String>, String> {
    let input: Vec<u32> = rows.iter().flatten().copied().collect();
    let n = CHECKED.len();
    let out = rows.len() * 2 * n;
    let words = match stage {
        Stage::Fragment(_) => {
            let width = 256u32;
            let code = check_kernel(rows.len() as u32, modes, Stage::Fragment(width)).map_err(|e| format!("kernel: {e}"))?;
            draw(dev, &code, &input, out, width, (rows.len() as u32).div_ceil(width), None)?
        }
        Stage::Vertex => {
            let code = check_kernel(rows.len() as u32, modes, Stage::Vertex).map_err(|e| format!("kernel: {e}"))?;
            draw(dev, &code, &input, out, 1, 1, Some(rows.len() as u32))?
        }
        Stage::Compute => {
            let code = check_kernel(rows.len() as u32, modes, Stage::Compute).map_err(|e| format!("kernel: {e}"))?;
            dispatch(dev, &code, &input, out, rows.len().div_ceil(64) as u32)?
        }
    };
    // The run computed: 2^127 * 1 of the hand-picked rows.
    let mul = CHECKED.iter().position(|&op| op == "mul").unwrap();
    if rows.get(5) == Some(&ROWS[5]) && words[(5 * n + mul) * 2] != 0x7F00_0000 {
        return Err(format!("the probe did not run (2^127 * 1 = {:08x})", words[(5 * n + mul) * 2]));
    }
    let mut bad = Vec::new();
    for (r, row) in rows.iter().enumerate() {
        for (k, op) in CHECKED.iter().enumerate() {
            let (raw, on_flushed) = (words[(r * n + k) * 2], words[(r * n + k) * 2 + 1]);
            let same = raw == on_flushed || is_nan(raw) && is_nan(on_flushed);
            if !same || is_denormal(raw) {
                bad.push(format!("{op}({:08x}, {:08x}, {:08x}) = {raw:08x}, on flushed operands {on_flushed:08x}", row[0], row[1], row[2]));
            }
        }
    }
    Ok(bad)
}

/// What the device does with denormals where the flush elision relies on
/// it (measured once per device, [`measure`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HostDenormals {
    /// The f32 operations that flush in a module with the 32-bit modes the
    /// emitter declares.
    pub plain: HostFloat,
    /// The same in a module that also declares DenormPreserve(16).
    pub with_preserve16: HostFloat,
    /// f16 keeps its denormals without DenormPreserve(16) exactly as with
    /// it (conversions, add, mul, fma).
    pub f16_kept: bool,
}

/// f16 rows for [`f16_kept`]: f32 values over the f16 range (and its
/// denormals) in the words, f16 bits with small exponents in the low halves.
fn f16_rows(n: usize, seed: u64) -> Vec<[u32; 3]> {
    let mut state = seed | 1;
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let mut r = [0u32; 3];
        for w in r.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let e = 100 + ((state >> 40) % 30) as u32;
            let half = ((state >> 20) as u32 & 0x83FF) | ((((state >> 50) % 4) as u32) << 10);
            let word = ((state >> 63) as u32) << 31 | e << 23 | (state as u32 & 0x7F_FFFF);
            *w = (word & 0xFFFF_0000) | half;
        }
        rows.push(r);
    }
    rows
}

/// Whether f16 gives the same bits without DenormPreserve(16) as with it,
/// in the modes the emitter declares otherwise, with denormal results
/// among them (`Err` when the probe cannot run).
fn f16_kept(dev: &Arc<GpuDevice>, rows: &[[u32; 3]]) -> Result<bool, String> {
    let nan = dev.caps.signed_zero_inf_nan_preserve;
    let rte16 = dev.caps.float16_rte;
    let without = Modes { nan_preserve: nan, preserve16: false, rte16 };
    let with = Modes { preserve16: true, ..without };
    let input: Vec<u32> = rows.iter().flatten().copied().collect();
    let denormal = |v: u32| v & 0x7C00 == 0 && v & 0x3FF != 0;
    for stage in [Stage::Compute, Stage::Vertex, Stage::Fragment(256)] {
        let run = |m: Modes| -> Result<Vec<u32>, String> {
            let code = f16_kernel(rows.len() as u32, m, stage).map_err(|e| format!("kernel: {e}"))?;
            let out = rows.len() * 5;
            match stage {
                Stage::Compute => dispatch(dev, &code, &input, out, rows.len().div_ceil(64) as u32),
                Stage::Vertex => draw(dev, &code, &input, out, 1, 1, Some(rows.len() as u32)),
                Stage::Fragment(width) => draw(dev, &code, &input, out, width, (rows.len() as u32).div_ceil(width), None),
            }
        };
        let (a, b) = (run(without)?, run(with)?);
        let kept = (0..rows.len()).filter(|&r| denormal(b[r * 5 + 3])).count();
        if a != b || kept == 0 {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Which f32 operations of this device flush denormals the way the flush
/// elision relies on (`ps5shader::opt::denorm::HostFloat`): in a module
/// with the 32-bit modes the emitter declares, and in one that also
/// declares DenormPreserve(16) (a driver may keep f32 denormals then); and
/// whether f16 keeps its denormals without DenormPreserve(16). Nothing on
/// a device with DenormPreserve(32): the translator then asks for
/// preserved denormals and flushes in code. Logged once per device.
pub fn measure(dev: &Arc<GpuDevice>) -> HostDenormals {
    if dev.caps.denorm_preserve[1] {
        return HostDenormals::default();
    }
    let t = std::time::Instant::now();
    let rows = probe_rows(4096 - ROWS.len(), 0x9E37_79B9_7F4A_7C15);
    let nan = dev.caps.signed_zero_inf_nan_preserve;
    let plain = Modes { nan_preserve: nan, preserve16: false, rte16: dev.caps.float16_rte };
    let wide = Modes { preserve16: true, ..plain };
    let f16 = dev.caps.shader_float16;
    let mut out = [HostFloat::NONE; 2];
    for (k, modes) in [plain, wide].into_iter().enumerate() {
        if k == 1 && !(dev.caps.denorm_preserve[0] && f16) {
            out[1] = out[0];
            continue;
        }
        // Every stage the translator emits f32 code for (mesh programs run
        // the vertex stage's code).
        match check(dev, &rows, modes).and_then(|mut bad| {
            bad.extend(check_in(dev, &rows, modes, Stage::Vertex)?);
            bad.extend(check_in(dev, &rows, modes, Stage::Fragment(0))?);
            Ok(bad)
        }) {
            Ok(bad) => {
                let flushes = |ops: &[&str]| !bad.iter().any(|b| ops.iter().any(|op| b.starts_with(&format!("{op}("))));
                out[k] = HostFloat { arith: flushes(&["add", "sub", "mul", "div"]), fused: flushes(&["fma", "rcp", "sqrt", "rsq", "log2"]) };
                if let Some(first) = bad.first() {
                    log::info!("ps5gpu: f32 denormals kept by the host in {} of the probe operations ({modes:?}; first: {first})", bad.len());
                }
            }
            Err(e) => {
                log::warn!("ps5gpu: f32 denormal probe failed: {e}");
                return HostDenormals::default();
            }
        }
    }
    let f16_kept = f16
        && dev.caps.denorm_preserve[0]
        && match f16_kept(dev, &f16_rows(4096, 0x2545_F491_4F6C_DD1D)) {
            Ok(k) => k,
            Err(e) => {
                log::warn!("ps5gpu: f16 denormal probe failed: {e}");
                false
            }
        };
    let h = HostDenormals { plain: out[0], with_preserve16: out[1], f16_kept };
    log::info!("ps5gpu: denormals on the host ({} ms): f32 flushed by {:?}, with DenormPreserve(16) {:?}; f16 kept without DenormPreserve(16): {}", t.elapsed().as_millis(), h.plain, h.with_preserve16, h.f16_kept);
    h
}

/// Inputs: denormal sources and normal sources whose exact results are
/// denormal.
pub(crate) const ROWS: &[[u32; 3]] = &[
    [0x0000_0001, 0x0000_0000, 0x0000_0000],
    [0x807F_FFFF, 0x3F80_0000, 0x0000_0000],
    // 2^-70 * 2^-70 = 2^-140 (denormal product of normals).
    [0x1C80_0000, 0x1C80_0000, 0x0000_0000],
    // 1.5 * 2^-126 + -(2^-126) = 2^-127 (denormal sum of normals).
    [0x00C0_0000, 0x8080_0000, 0x0000_0000],
    // Denormal + denormal = smallest normal.
    [0x0040_0000, 0x0040_0000, 0x0080_0000],
    // 1 / 2^127 = 2^-127; 2^127 * 1.
    [0x7F00_0000, 0x3F80_0000, 0x0000_0000],
    // fma(2^-70, 2^-60, -0) = 2^-130.
    [0x1C80_0000, 0x2180_0000, 0x8000_0000],
    // denormal * 1.0; ldexp(denormal, 0).
    [0x0040_0000, 0x3F80_0000, 0x0000_0000],
    // 2^-149 * 2^24 = 2^-125 (normal from a denormal input).
    [0x0000_0001, 0x4B80_0000, 0x0000_0000],
    // Two negative / positive denormals.
    [0x8000_0001, 0x0000_0001, 0x3F80_0000],
    // ldexp(1.0, -130) = 2^-130.
    [0x3F80_0000, 0x3F80_0000, (-130i32) as u32],
    // 2^-63 * 2^-64 (rounded product exactly 2^-127), / 2^126 -> 2^-126*...
    [0x2000_0000, 0x1F80_0000, 0x0000_0000],
    // 0x3F21E58F * 0x00CA6691 = 2^-126 - 2^-173 exactly: tiny before
    // rounding, the smallest normal after rounding to 24 bits.
    [0x3F21_E58F, 0x00CA_6691, 0x0000_0000],
    // -2^-126 * (1 - 2^-24): exactly representable as a denormal.
    [0x8080_0000, 0x3F7F_FFFF, 0x0000_0000],
    // A negative denormal (ceil -> -0, mul .5 -> -denormal).
    [0x8040_0000, 0x0000_0000, 0x0000_0000],
    // (2^47 - 2^22) * 2^-173 = 2^-126 - 2^-151: a tie between the largest
    // denormal and 2^-126 at 24 bits (rounds to even, 2^-126).
    [0x3F04_2108, 0x00F8_0000, 0x0000_0000],
    // (2^47 - 4194306) * 2^-173: just past that tie, tiny after rounding
    // at 24 bits, but within half a denormal step of 2^-126.
    [0x3F09_B29A, 0x00ED_F87B, 0x0000_0000],
];

/// f16 operations on rows of three words (f32 values for the conversion,
/// the low halves as f16 operands for the rest): f32 -> f16, f16 -> f32,
/// f16 add, mul and fma, as bits (five words per row).
fn f16_kernel(rows: u32, modes: Modes, stage: Stage) -> R<Vec<u32>> {
    use spirv::GlslStd450Op as G;
    let mut s = Spv::new();
    s.b.capability(spirv::Capability::Float16);
    s.b.capability(spirv::Capability::Int16);
    if modes.nan_preserve.iter().any(|&b| b) {
        s.b.capability(spirv::Capability::SignedZeroInfNanPreserve);
    }
    if modes.preserve16 {
        s.b.capability(spirv::Capability::DenormPreserve);
    }
    if modes.rte16 {
        s.b.capability(spirv::Capability::RoundingModeRTE);
    }
    if modes != Modes::default() {
        s.b.extension("SPV_KHR_float_controls");
    }
    let glsl = s.b.ext_inst_import("GLSL.std.450");
    let f32t = s.b.type_float(32, None);
    let (main, x, coord) = begin_stage(&mut s, stage, f32t)?;
    let f16t = s.b.type_float(16, None);
    let u16t = s.b.type_int(16, 0);
    let rows_c = s.c(rows);
    let past = s.b.u_greater_than_equal(s.boolean, None, x, rows_c)?;
    s.return_when(past)?;
    let base = s.mul_c(x, 3)?;
    let mut w = Vec::new();
    let mut h = Vec::new();
    for k in 0..3 {
        let i = s.add_c(base, k)?;
        let word = s.load_word(s.src, i)?;
        w.push(word);
        let lo = s.b.u_convert(u16t, None, word)?;
        h.push(s.b.bitcast(f16t, None, lo)?);
    }
    let to16 = |s: &mut Spv, v: u32| -> R<u32> {
        let b = s.b.bitcast(u16t, None, v)?;
        s.b.u_convert(s.uint, None, b)
    };
    let af = s.b.bitcast(f32t, None, w[0])?;
    let c = s.b.f_convert(f16t, None, af)?;
    let mut out = vec![to16(&mut s, c)?];
    let wide = s.b.f_convert(f32t, None, h[0])?;
    out.push(s.b.bitcast(s.uint, None, wide)?);
    let sum = s.b.f_add(f16t, None, h[0], h[1])?;
    out.push(to16(&mut s, sum)?);
    let product = s.b.f_mul(f16t, None, h[0], h[1])?;
    out.push(to16(&mut s, product)?);
    let fma = s.b.ext_inst(f16t, None, glsl, G::Fma as u32, [Operand::IdRef(h[0]), Operand::IdRef(h[1]), Operand::IdRef(h[2])])?;
    out.push(to16(&mut s, fma)?);
    let out_base = s.mul_c(x, 5)?;
    for (k, v) in out.into_iter().enumerate() {
        let i = s.add_c(out_base, k as u32)?;
        s.store_word(s.dst, i, v)?;
    }
    for (i, width) in [16, 32, 64].into_iter().enumerate() {
        if modes.nan_preserve[i] {
            s.b.execution_mode(main, spirv::ExecutionMode::SignedZeroInfNanPreserve, [width]);
        }
    }
    if modes.preserve16 {
        s.b.execution_mode(main, spirv::ExecutionMode::DenormPreserve, [16]);
    }
    if modes.rte16 {
        s.b.execution_mode(main, spirv::ExecutionMode::RoundingModeRTE, [16]);
    }
    finish_stage(s, stage, main, coord)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host flushes on a million rows (the startup check uses 4096).
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn host_denormals_flush_on_many_rows() {
        let Ok(dev) = GpuDevice::create(false) else { return };
        if dev.caps.denorm_preserve[1] {
            return;
        }
        let modes = Modes { nan_preserve: dev.caps.signed_zero_inf_nan_preserve, rte16: dev.caps.float16_rte, ..Modes::default() };
        let mut bad = Vec::new();
        for seed in 1..=16u64 {
            let rows = probe_rows(65536, seed.wrapping_mul(0x2545_F491_4F6C_DD1D));
            bad.extend(check(&dev, &rows, modes).unwrap());
            for stage in [Stage::Vertex, Stage::Fragment(0)] {
                let other = check_in(&dev, &rows, modes, stage).unwrap();
                if seed == 1 {
                    println!("{stage:?} stage: {} rows not flushed", other.len());
                }
                bad.extend(other);
            }
        }
        for b in bad.iter().take(20) {
            println!("{b}");
        }
        println!("{} rows not flushed", bad.len());
        assert_eq!(dev.f32_host_ftz.get().map(|h| h.plain.arith && h.plain.fused), Some(bad.is_empty()));
    }

    /// What f16 does with and without DenormPreserve(16) (the guest keeps
    /// f16 denormals): the rows whose f16 results differ.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn host_f16_denormals_by_modes() {
        let Ok(dev) = GpuDevice::create(false) else { return };
        let rows = f16_rows(65536, 0x1234_5678_9ABC_DEF1);
        let n = [true, true, true];
        let sets = [
            ("nan+rte16", Modes { nan_preserve: n, rte16: true, ..Modes::default() }),
            ("nan", Modes { nan_preserve: n, ..Modes::default() }),
            ("nan+preserve16+rte16", Modes { nan_preserve: n, preserve16: true, rte16: true }),
        ];
        let input: Vec<u32> = rows.iter().flatten().copied().collect();
        let results: Vec<Vec<u32>> = sets
            .iter()
            .map(|(_, m)| dispatch(&dev, &f16_kernel(rows.len() as u32, *m, Stage::Compute).unwrap(), &input, rows.len() * 5, rows.len().div_ceil(64) as u32).unwrap())
            .collect();
        let names = ["f32->f16", "f16->f32", "add16", "mul16", "fma16"];
        let reference = results.len() - 1;
        for (k, (name, _)) in sets.iter().enumerate() {
            let counts: Vec<String> = names
                .iter()
                .enumerate()
                .map(|(j, op)| {
                    let differ = (0..rows.len()).filter(|&r| results[k][r * 5 + j] != results[reference][r * 5 + j]).count();
                    format!("{op} {differ}")
                })
                .collect();
            println!("{name:>22} vs preserve16: {}", counts.join(", "));
        }
        let denormal16 = (0..rows.len()).filter(|&r| { let v = results[reference][r * 5 + 3]; v & 0x7C00 == 0 && v & 0x3FF != 0 }).count();
        println!("{denormal16} f16 denormal products with DenormPreserve(16)");
        assert_eq!(f16_kept(&dev, &f16_rows(65536, 3)), Ok(true), "every stage keeps f16 denormals without DenormPreserve(16)");
    }

    /// Which checked operations keep denormals under each set of modes the
    /// emitter declares.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn host_denormals_by_modes() {
        let Ok(dev) = GpuDevice::create(false) else { return };
        let rows = probe_rows(65536, 7);
        let n = [true, true, true];
        let sets = [
            ("none", Modes::default()),
            ("nan32", Modes { nan_preserve: [false, true, false], ..Modes::default() }),
            ("nan16/32/64", Modes { nan_preserve: n, ..Modes::default() }),
            ("preserve16", Modes { preserve16: true, ..Modes::default() }),
            ("rte16", Modes { rte16: true, ..Modes::default() }),
            ("nan+rte16", Modes { nan_preserve: n, rte16: true, ..Modes::default() }),
            ("nan+preserve16", Modes { nan_preserve: n, preserve16: true, ..Modes::default() }),
            ("nan+preserve16+rte16", Modes { nan_preserve: n, preserve16: true, rte16: true }),
        ];
        for stage in [Stage::Compute, Stage::Vertex, Stage::Fragment(0)] {
            for (name, modes) in sets {
                let bad = check_in(&dev, &rows, modes, stage).unwrap();
                let counts: Vec<String> = CHECKED.iter().map(|op| format!("{op} {}", bad.iter().filter(|b| b.starts_with(&format!("{op}("))).count())).collect();
                println!("{:>11}{name:>22}: {}", format!("{stage:?}"), counts.join(", "));
            }
        }
    }

    /// Prints the device's float controls and the probe table for every
    /// mode (the measurement behind the translator's native-flush choice).
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn host_denormals_probe() {
        let Ok(dev) = GpuDevice::create(false) else { return };
        let mut fc = vk::PhysicalDeviceFloatControlsProperties::default();
        let mut dp = vk::PhysicalDeviceDriverProperties::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut fc).push_next(&mut dp);
        // SAFETY: valid chain on the stack.
        unsafe { dev.instance.get_physical_device_properties2(dev.phys, &mut p2) };
        let name = dp.driver_info_as_c_str().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        println!("driver {:?} {name}", dp.driver_id);
        println!(
            "denormBehaviorIndependence {:?} roundingModeIndependence {:?}\nDenormPreserve 16/32/64 {} {} {}\nDenormFlushToZero 16/32/64 {} {} {}\nSignedZeroInfNanPreserve 16/32/64 {} {} {}\nRTE 16/32/64 {} {} {} RTZ {} {} {}",
            fc.denorm_behavior_independence,
            fc.rounding_mode_independence,
            fc.shader_denorm_preserve_float16,
            fc.shader_denorm_preserve_float32,
            fc.shader_denorm_preserve_float64,
            fc.shader_denorm_flush_to_zero_float16,
            fc.shader_denorm_flush_to_zero_float32,
            fc.shader_denorm_flush_to_zero_float64,
            fc.shader_signed_zero_inf_nan_preserve_float16,
            fc.shader_signed_zero_inf_nan_preserve_float32,
            fc.shader_signed_zero_inf_nan_preserve_float64,
            fc.shader_rounding_mode_rte_float16,
            fc.shader_rounding_mode_rte_float32,
            fc.shader_rounding_mode_rte_float64,
            fc.shader_rounding_mode_rtz_float16,
            fc.shader_rounding_mode_rtz_float32,
            fc.shader_rounding_mode_rtz_float64,
        );
        let modes = [(Mode::Default, false), (Mode::Default, true), (Mode::Preserve, false), (Mode::Preserve, true), (Mode::FlushToZero, false), (Mode::FlushToZero, true)];
        let results: Vec<_> = modes.iter().map(|&(m, n)| run(&dev, m, n, ROWS).unwrap()).collect();
        print!("{:>8} {:>26}", "op", "row (a b c)");
        for (m, n) in modes {
            print!(" {:>10}", format!("{m:?}{}", if n { "+N" } else { "" }).chars().take(10).collect::<String>());
        }
        println!();
        for (k, op) in OPS.iter().enumerate() {
            for (r, row) in ROWS.iter().enumerate() {
                let vals: Vec<u32> = results.iter().map(|res| res[r][k]).collect();
                let differ = vals.iter().any(|&v| v != vals[0]);
                print!("{op:>8} {:08x} {:08x} {:08x}", row[0], row[1], row[2]);
                for v in &vals {
                    print!(" {v:>10x}");
                }
                println!("{}", if differ { "  *" } else { "" });
            }
        }
    }
}
