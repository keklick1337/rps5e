// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! SPIR-V of the menu pass (egui meshes), built with rspirv.
//!
//! Vertex inputs: location 0 `vec2` position in points, 1 `vec2` texture
//! coordinate, 2 `vec4` colour (R8G8B8A8_UNORM: sRGB-encoded, alpha
//! premultiplied). Push constants (vertex and fragment): `vec2 screen`
//! (window size in points), `uint srgb` (1: the target is an sRGB format,
//! the output is converted to linear). The fragment shader multiplies the
//! colour with the texel of the combined image sampler at set 0, binding 0;
//! the pipeline blends premultiplied alpha.

use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{self, BuiltIn, Decoration as D, GlslStd450Op, StorageClass as S};

type R<T> = Result<T, rspirv::dr::Error>;

/// Size of the push constants.
pub const PUSH_CONSTANT_BYTES: u32 = 16;

/// Bytes per vertex: pos (2 x f32), uv (2 x f32), colour (4 x u8).
pub const VERTEX_STRIDE: u32 = 20;

fn module() -> Builder {
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(spirv::Capability::Shader);
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    b
}

/// The push constant block `{ vec2 screen; uint srgb; uint pad; }`.
fn push_block(b: &mut Builder) -> (u32, u32) {
    let f32t = b.type_float(32, None);
    let u32t = b.type_int(32, 0);
    let vec2 = b.type_vector(f32t, 2);
    let ty = b.type_struct([vec2, u32t, u32t]);
    b.decorate(ty, D::Block, []);
    for (m, off) in [0u32, 8, 12].into_iter().enumerate() {
        b.member_decorate(ty, m as u32, D::Offset, [Operand::LiteralBit32(off)]);
    }
    let ptr = b.type_pointer(None, S::PushConstant, ty);
    let var = b.variable(ptr, None, S::PushConstant, None);
    (ty, var)
}

pub fn vertex() -> Vec<u32> {
    let build = || -> R<Vec<u32>> {
        let mut b = module();
        let void = b.type_void();
        let f32t = b.type_float(32, None);
        let i32t = b.type_int(32, 1);
        let vec2 = b.type_vector(f32t, 2);
        let vec4 = b.type_vector(f32t, 4);
        let in_vec2 = b.type_pointer(None, S::Input, vec2);
        let in_vec4 = b.type_pointer(None, S::Input, vec4);
        let out_vec2 = b.type_pointer(None, S::Output, vec2);
        let out_vec4 = b.type_pointer(None, S::Output, vec4);
        let (_, pc) = push_block(&mut b);
        let pc_vec2 = b.type_pointer(None, S::PushConstant, vec2);
        let var = |b: &mut Builder, ty, class, loc: u32| {
            let v = b.variable(ty, None, class, None);
            b.decorate(v, D::Location, [Operand::LiteralBit32(loc)]);
            v
        };
        let pos_in = var(&mut b, in_vec2, S::Input, 0);
        let uv_in = var(&mut b, in_vec2, S::Input, 1);
        let color_in = var(&mut b, in_vec4, S::Input, 2);
        let color_out = var(&mut b, out_vec4, S::Output, 0);
        let uv_out = var(&mut b, out_vec2, S::Output, 1);
        let position = b.variable(out_vec4, None, S::Output, None);
        b.decorate(position, D::BuiltIn, [Operand::BuiltIn(BuiltIn::Position)]);
        let c0 = b.constant_bit32(i32t, 0);
        let zero = b.constant_bit32(f32t, 0f32.to_bits());
        let one = b.constant_bit32(f32t, 1f32.to_bits());
        let two = b.constant_bit32(f32t, 2f32.to_bits());
        let fnty = b.type_function(void, []);
        let main = b.begin_function(void, None, spirv::FunctionControl::NONE, fnty)?;
        b.begin_block(None)?;
        let pos = b.load(vec2, None, pos_in, None, [])?;
        let sp = b.access_chain(pc_vec2, None, pc, [c0])?;
        let screen = b.load(vec2, None, sp, None, [])?;
        // ndc = 2 * pos / screen - 1 (egui's y grows downwards, like Vulkan's).
        let two2 = b.composite_construct(vec2, None, [two, two])?;
        let one2 = b.composite_construct(vec2, None, [one, one])?;
        let scaled = b.f_mul(vec2, None, pos, two2)?;
        let norm = b.f_div(vec2, None, scaled, screen)?;
        let ndc = b.f_sub(vec2, None, norm, one2)?;
        let x = b.composite_extract(f32t, None, ndc, [0])?;
        let y = b.composite_extract(f32t, None, ndc, [1])?;
        let p = b.composite_construct(vec4, None, [x, y, zero, one])?;
        b.store(position, p, None, [])?;
        let uv = b.load(vec2, None, uv_in, None, [])?;
        b.store(uv_out, uv, None, [])?;
        let color = b.load(vec4, None, color_in, None, [])?;
        b.store(color_out, color, None, [])?;
        b.ret()?;
        b.end_function()?;
        b.entry_point(spirv::ExecutionModel::Vertex, main, "main", [pos_in, uv_in, color_in, color_out, uv_out, position]);
        Ok(b.module().assemble())
    };
    build().expect("menu vertex shader")
}

pub fn fragment() -> Vec<u32> {
    fragment_for(false)
}

/// The fragment shader for an HDR10 swapchain: the premultiplied sRGB
/// colour is un-premultiplied, converted to BT.2020 PQ
/// ([`crate::hdr::sdr_to_hdr10`]) and premultiplied again; `srgb` is not
/// read.
pub fn fragment_hdr() -> Vec<u32> {
    fragment_for(true)
}

fn fragment_for(hdr: bool) -> Vec<u32> {
    let build = || -> R<Vec<u32>> {
        let mut b = module();
        let glsl = b.ext_inst_import("GLSL.std.450");
        let void = b.type_void();
        let f32t = b.type_float(32, None);
        let i32t = b.type_int(32, 1);
        let u32t = b.type_int(32, 0);
        let vec2 = b.type_vector(f32t, 2);
        let vec3 = b.type_vector(f32t, 3);
        let vec4 = b.type_vector(f32t, 4);
        let in_vec2 = b.type_pointer(None, S::Input, vec2);
        let in_vec4 = b.type_pointer(None, S::Input, vec4);
        let out_vec4 = b.type_pointer(None, S::Output, vec4);
        let (_, pc) = push_block(&mut b);
        let pc_u32 = b.type_pointer(None, S::PushConstant, u32t);
        let image = b.type_image(f32t, spirv::Dim::Dim2D, 0, 0, 0, 1, spirv::ImageFormat::Unknown, None);
        let sampled = b.type_sampled_image(image);
        let tex_ptr = b.type_pointer(None, S::UniformConstant, sampled);
        let tex = b.variable(tex_ptr, None, S::UniformConstant, None);
        b.decorate(tex, D::DescriptorSet, [Operand::LiteralBit32(0)]);
        b.decorate(tex, D::Binding, [Operand::LiteralBit32(0)]);
        let color_in = b.variable(in_vec4, None, S::Input, None);
        b.decorate(color_in, D::Location, [Operand::LiteralBit32(0)]);
        let uv_in = b.variable(in_vec2, None, S::Input, None);
        b.decorate(uv_in, D::Location, [Operand::LiteralBit32(1)]);
        let out = b.variable(out_vec4, None, S::Output, None);
        b.decorate(out, D::Location, [Operand::LiteralBit32(0)]);
        let c1 = b.constant_bit32(i32t, 1);
        let gamma = b.constant_bit32(f32t, 2.2f32.to_bits());
        let fnty = b.type_function(void, []);
        let main = b.begin_function(void, None, spirv::FunctionControl::NONE, fnty)?;
        b.begin_block(None)?;
        let color = b.load(vec4, None, color_in, None, [])?;
        let uv = b.load(vec2, None, uv_in, None, [])?;
        let t = b.load(sampled, None, tex, None, [])?;
        let texel = b.image_sample_implicit_lod(vec4, None, t, uv, None, [])?;
        let c = b.f_mul(vec4, None, color, texel)?;
        let rgb = b.vector_shuffle(vec3, None, c, c, [0, 1, 2])?;
        if hdr {
            let a = b.composite_extract(f32t, None, c, [3])?;
            let zero = b.constant_bit32(f32t, 0f32.to_bits());
            let zero3 = b.constant_composite(vec3, [zero, zero, zero]);
            let a3 = b.composite_construct(vec3, None, [a, a, a])?;
            let straight = b.f_div(vec3, None, rgb, a3)?;
            let boolt = b.type_bool();
            let bvec3 = b.type_vector(boolt, 3);
            let covered = b.f_ord_greater_than(bvec3, None, a3, zero3)?;
            let straight = b.select(vec3, None, covered, straight, zero3)?;
            let pq = crate::hdr::emit_sdr_to_hdr10(&mut b, glsl, straight, false)?;
            let pm = b.f_mul(vec3, None, pq, a3)?;
            let r = b.composite_extract(f32t, None, pm, [0])?;
            let g = b.composite_extract(f32t, None, pm, [1])?;
            let bl = b.composite_extract(f32t, None, pm, [2])?;
            let result = b.composite_construct(vec4, None, [r, g, bl, a])?;
            b.store(out, result, None, [])?;
            b.ret()?;
            b.end_function()?;
            b.execution_mode(main, spirv::ExecutionMode::OriginUpperLeft, []);
            b.entry_point(spirv::ExecutionModel::Fragment, main, "main", [color_in, uv_in, out]);
            return Ok(b.module().assemble());
        }
        // sRGB target: rgb^2.2 (the colours are sRGB-encoded).
        let g3 = b.composite_construct(vec3, None, [gamma, gamma, gamma])?;
        let lin = b.ext_inst(vec3, None, glsl, GlslStd450Op::Pow as u32, [Operand::IdRef(rgb), Operand::IdRef(g3)])?;
        let fp = b.access_chain(pc_u32, None, pc, [c1])?;
        let flag = b.load(u32t, None, fp, None, [])?;
        let f = b.convert_u_to_f(f32t, None, flag)?;
        let f3 = b.composite_construct(vec3, None, [f, f, f])?;
        let mixed = b.ext_inst(vec3, None, glsl, GlslStd450Op::FMix as u32, [Operand::IdRef(rgb), Operand::IdRef(lin), Operand::IdRef(f3)])?;
        let a = b.composite_extract(f32t, None, c, [3])?;
        let r = b.composite_extract(f32t, None, mixed, [0])?;
        let g = b.composite_extract(f32t, None, mixed, [1])?;
        let bl = b.composite_extract(f32t, None, mixed, [2])?;
        let result = b.composite_construct(vec4, None, [r, g, bl, a])?;
        b.store(out, result, None, [])?;
        b.ret()?;
        b.end_function()?;
        b.execution_mode(main, spirv::ExecutionMode::OriginUpperLeft, []);
        b.entry_point(spirv::ExecutionModel::Fragment, main, "main", [color_in, uv_in, out]);
        Ok(b.module().assemble())
    };
    build().expect("menu fragment shader")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaders_assemble_and_parse() {
        for words in [vertex(), fragment(), fragment_hdr()] {
            assert_eq!(words[0], spirv::MAGIC_NUMBER);
            let module = rspirv::dr::load_words(&words).unwrap();
            assert_eq!(module.entry_points.len(), 1);
        }
        // `RPS5E_SPIRV_DUMP=<dir>` writes the binaries for spirv-val.
        if let Some(dir) = std::env::var_os("RPS5E_SPIRV_DUMP") {
            let bytes = |w: Vec<u32>| w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            let dir = std::path::PathBuf::from(dir);
            std::fs::write(dir.join("menu.vert.spv"), bytes(vertex())).unwrap();
            std::fs::write(dir.join("menu.frag.spv"), bytes(fragment())).unwrap();
            std::fs::write(dir.join("menu_hdr.frag.spv"), bytes(fragment_hdr())).unwrap();
        }
    }
}
