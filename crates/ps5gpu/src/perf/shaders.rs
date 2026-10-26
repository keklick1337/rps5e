// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! SPIR-V of the overlay pass, built with rspirv (no shader toolchain).
//!
//! The vertex shader emits a triangle covering the viewport (the viewport
//! and scissor are the panel rectangle). The fragment shader reads the
//! panel texel under the fragment from a storage buffer of packed RGBA8
//! words (`R` in the low byte) and outputs it; the pipeline blends it over
//! the game image with its alpha.
//!
//! Fragment push constants: `int x, int y` (panel origin in the swapchain
//! image), `uint width`.

use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{self, BuiltIn, Decoration as D, GlslStd450Op, StorageClass as S};

type R<T> = Result<T, rspirv::dr::Error>;

/// Size of the fragment push constants.
pub const PUSH_CONSTANT_BYTES: u32 = 16;

fn module() -> Builder {
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(spirv::Capability::Shader);
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    b
}

/// The full-viewport triangle.
pub fn vertex() -> Vec<u32> {
    let build = || -> R<Vec<u32>> {
        let mut b = module();
        let void = b.type_void();
        let f32t = b.type_float(32, None);
        let i32t = b.type_int(32, 1);
        let vec4 = b.type_vector(f32t, 4);
        let in_i32 = b.type_pointer(None, S::Input, i32t);
        let out_vec4 = b.type_pointer(None, S::Output, vec4);
        let index = b.variable(in_i32, None, S::Input, None);
        b.decorate(index, D::BuiltIn, [Operand::BuiltIn(BuiltIn::VertexIndex)]);
        let position = b.variable(out_vec4, None, S::Output, None);
        b.decorate(position, D::BuiltIn, [Operand::BuiltIn(BuiltIn::Position)]);
        let one_i = b.constant_bit32(i32t, 1);
        let two_i = b.constant_bit32(i32t, 2);
        let zero_f = b.constant_bit32(f32t, 0f32.to_bits());
        let one_f = b.constant_bit32(f32t, 1f32.to_bits());
        let two_f = b.constant_bit32(f32t, 2f32.to_bits());
        let fnty = b.type_function(void, []);
        let main = b.begin_function(void, None, spirv::FunctionControl::NONE, fnty)?;
        b.begin_block(None)?;
        // (x, y) = ((i << 1) & 2, i & 2) * 2 - 1: (-1,-1), (3,-1), (-1,3).
        let i = b.load(i32t, None, index, None, [])?;
        let shl = b.shift_left_logical(i32t, None, i, one_i)?;
        let xi = b.bitwise_and(i32t, None, shl, two_i)?;
        let yi = b.bitwise_and(i32t, None, i, two_i)?;
        let mut coord = |v| -> R<u32> {
            let f = b.convert_s_to_f(f32t, None, v)?;
            let m = b.f_mul(f32t, None, f, two_f)?;
            b.f_sub(f32t, None, m, one_f)
        };
        let x = coord(xi)?;
        let y = coord(yi)?;
        let pos = b.composite_construct(vec4, None, [x, y, zero_f, one_f])?;
        b.store(position, pos, None, [])?;
        b.ret()?;
        b.end_function()?;
        b.entry_point(spirv::ExecutionModel::Vertex, main, "main", [index, position]);
        Ok(b.module().assemble())
    };
    build().expect("overlay vertex shader")
}

/// Panel texel fetch.
pub fn fragment() -> Vec<u32> {
    fragment_for(false)
}

/// Panel texel fetch for an HDR10 swapchain: the sRGB texel is converted
/// to BT.2020 PQ ([`crate::hdr::sdr_to_hdr10`]); alpha is unchanged.
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
        let vec4 = b.type_vector(f32t, 4);
        // Interface.
        let in_vec4 = b.type_pointer(None, S::Input, vec4);
        let out_vec4 = b.type_pointer(None, S::Output, vec4);
        let frag_coord = b.variable(in_vec4, None, S::Input, None);
        b.decorate(frag_coord, D::BuiltIn, [Operand::BuiltIn(BuiltIn::FragCoord)]);
        let color = b.variable(out_vec4, None, S::Output, None);
        b.decorate(color, D::Location, [Operand::LiteralBit32(0)]);
        // Push constants { int x; int y; uint width; uint pad; }.
        let pc_ty = b.type_struct([i32t, i32t, u32t, u32t]);
        b.decorate(pc_ty, D::Block, []);
        for (m, off) in [0u32, 4, 8, 12].into_iter().enumerate() {
            b.member_decorate(pc_ty, m as u32, D::Offset, [Operand::LiteralBit32(off)]);
        }
        let pc_ptr = b.type_pointer(None, S::PushConstant, pc_ty);
        let pc = b.variable(pc_ptr, None, S::PushConstant, None);
        // Storage buffer { uint texels[]; } at set 0, binding 0.
        let rt = b.type_runtime_array(u32t);
        b.decorate(rt, D::ArrayStride, [Operand::LiteralBit32(4)]);
        let sb_ty = b.type_struct([rt]);
        b.decorate(sb_ty, D::Block, []);
        b.member_decorate(sb_ty, 0, D::Offset, [Operand::LiteralBit32(0)]);
        b.member_decorate(sb_ty, 0, D::NonWritable, []);
        let sb_ptr = b.type_pointer(None, S::StorageBuffer, sb_ty);
        let sb = b.variable(sb_ptr, None, S::StorageBuffer, None);
        b.decorate(sb, D::DescriptorSet, [Operand::LiteralBit32(0)]);
        b.decorate(sb, D::Binding, [Operand::LiteralBit32(0)]);
        let pc_i32 = b.type_pointer(None, S::PushConstant, i32t);
        let pc_u32 = b.type_pointer(None, S::PushConstant, u32t);
        let sb_u32 = b.type_pointer(None, S::StorageBuffer, u32t);
        let c0 = b.constant_bit32(i32t, 0);
        let c1 = b.constant_bit32(i32t, 1);
        let c2 = b.constant_bit32(i32t, 2);

        let fnty = b.type_function(void, []);
        let main = b.begin_function(void, None, spirv::FunctionControl::NONE, fnty)?;
        b.begin_block(None)?;
        let fc = b.load(vec4, None, frag_coord, None, [])?;
        let fx = b.composite_extract(f32t, None, fc, [0])?;
        let fy = b.composite_extract(f32t, None, fc, [1])?;
        let px = b.convert_f_to_s(i32t, None, fx)?;
        let py = b.convert_f_to_s(i32t, None, fy)?;
        let ox_p = b.access_chain(pc_i32, None, pc, [c0])?;
        let ox = b.load(i32t, None, ox_p, None, [])?;
        let oy_p = b.access_chain(pc_i32, None, pc, [c1])?;
        let oy = b.load(i32t, None, oy_p, None, [])?;
        let w_p = b.access_chain(pc_u32, None, pc, [c2])?;
        let w = b.load(u32t, None, w_p, None, [])?;
        let lx = b.i_sub(i32t, None, px, ox)?;
        let ly = b.i_sub(i32t, None, py, oy)?;
        let lx = b.bitcast(u32t, None, lx)?;
        let ly = b.bitcast(u32t, None, ly)?;
        let row = b.i_mul(u32t, None, ly, w)?;
        let idx = b.i_add(u32t, None, row, lx)?;
        let tp = b.access_chain(sb_u32, None, sb, [c0, idx])?;
        let texel = b.load(u32t, None, tp, None, [])?;
        let rgba = b.ext_inst(vec4, None, glsl, GlslStd450Op::UnpackUnorm4x8 as u32, [Operand::IdRef(texel)])?;
        let rgba = if hdr {
            let vec3 = b.type_vector(f32t, 3);
            let rgb = b.vector_shuffle(vec3, None, rgba, rgba, [0, 1, 2])?;
            let pq = crate::hdr::emit_sdr_to_hdr10(&mut b, glsl, rgb, false)?;
            let alpha = b.composite_extract(f32t, None, rgba, [3])?;
            let r = b.composite_extract(f32t, None, pq, [0])?;
            let g = b.composite_extract(f32t, None, pq, [1])?;
            let bl = b.composite_extract(f32t, None, pq, [2])?;
            b.composite_construct(vec4, None, [r, g, bl, alpha])?
        } else {
            rgba
        };
        b.store(color, rgba, None, [])?;
        b.ret()?;
        b.end_function()?;
        b.execution_mode(main, spirv::ExecutionMode::OriginUpperLeft, []);
        b.entry_point(spirv::ExecutionModel::Fragment, main, "main", [frag_coord, color]);
        Ok(b.module().assemble())
    };
    build().expect("overlay fragment shader")
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
        // Optional: `RPS5E_SPIRV_DUMP=<dir>` writes the binaries for
        // offline validation with spirv-val.
        if let Some(dir) = std::env::var_os("RPS5E_SPIRV_DUMP") {
            let bytes = |w: Vec<u32>| w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            let dir = std::path::PathBuf::from(dir);
            std::fs::write(dir.join("overlay.vert.spv"), bytes(vertex())).unwrap();
            std::fs::write(dir.join("overlay.frag.spv"), bytes(fragment())).unwrap();
            std::fs::write(dir.join("overlay_hdr.frag.spv"), bytes(fragment_hdr())).unwrap();
        }
    }
}
