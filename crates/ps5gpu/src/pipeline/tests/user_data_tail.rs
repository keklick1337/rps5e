// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! A pixel program that ends in s_setpc_b64 through a code address in its
//! user data continues in that code (the reference driver's
//! FragmentUserDataTail test): the linked program draws what the same
//! code joined into one program draws, with the high address half a
//! constant the body moves into place, and follows a change of the
//! continuation's address or code; SCC and EXEC carry across the jump.
//!
//! Run with `cargo test -p ps5gpu --release -- --ignored user_data_tail`.

use super::*;
use std::sync::{Arc, Mutex};

const KEPT: u8 = 0x40;
const W: u32 = 16;
const H: u32 = 16;
/// s_cbranch_scc1 +1; v_mov_b32 v0, 0; v_mov_b32 v2, s8; v_mov_b32 v3,
/// 1.0; exp mrt0 (v2, v0, v1, v3) done vm; s_endpgm.
const EPILOG: [u32; 7] = [0xbf850001, 0x7e000280, 0x7e040208, 0x7e0602f2, 0xf800180f, 0x03010002, 0xbf810000];
/// The same with v_mov_b32 v2, s1.
const OTHER_EPILOG: [u32; 7] = [0xbf850001, 0x7e000280, 0x7e040201, 0x7e0602f2, 0xf800180f, 0x03010002, 0xbf810000];
const S_SETPC_S4: u32 = 0xbe80_2004;
const S_CODE_END: u32 = 0xbf9f_0000;

/// Compiles the programs listed in `spirv_for` through SPIR-V, the rest
/// with the test compiler.
struct Listed {
    spirv: crate::shader::SpirvCompiler,
    spirv_for: Arc<Mutex<Vec<u64>>>,
    rest: TestCompiler,
}

impl ShaderCompiler for Listed {
    fn compile(&self, input: &compiler::ShaderCompileInput<'_, '_>) -> Result<compiler::CompiledShader, compiler::CompileError> {
        if self.spirv_for.lock().unwrap().contains(&input.guest_address) {
            self.spirv.compile(input)
        } else {
            self.rest.compile(input)
        }
    }
}

struct Bench {
    g: Guest,
    rig: Rig,
    vs: u64,
    spirv_for: Arc<Mutex<Vec<u64>>>,
}

impl Bench {
    /// Pixels of a draw of a triangle over part of the target with pixel
    /// program `ps`, user data s0 = 1.0, s1 = 0.5, s4 = `next` (low half)
    /// and s5 = a junk word the program replaces (when `user_count` is 6).
    fn draw(&mut self, ps: u64, next: u64, user_count: usize) -> Vec<u8> {
        // Linear rows of the target are 64 texels apart.
        let target = rgba8_target(self.g.alloc(&vec![KEPT; (64 * H * 4) as usize]), W, H);
        let user = [1.0f32.to_bits(), 0.5f32.to_bits(), 0, 0, next as u32, 0x0080_3fc1];
        let d = draw_into(target, stage(self.vs, &[]), stage(ps, &user[..user_count]), primitive::TRI_LIST, auto(3, 1), None);
        assert_eq!(self.rig.draw(&d), DrawOutcome::Recorded);
        self.rig.pixels(&target)
    }

    /// Draw the linked body (`prefix`, s_setpc_b64 s[4:5]) and the
    /// reference (`prefix` then `epilog`'s code); both must agree.
    fn check(&mut self, prefix: &[u32], next: u64, epilog: &[u32], user_count: usize) -> Vec<u8> {
        let mut body = prefix.to_vec();
        body.push(S_SETPC_S4);
        body.resize(64, S_CODE_END);
        let mut reference = prefix.to_vec();
        reference.extend_from_slice(epilog);
        reference.resize(64, S_CODE_END);
        let (body, reference) = (self.g.program(&body), self.g.program(&reference));
        self.spirv_for.lock().unwrap().extend([body, reference]);
        let expected = self.draw(reference, next, user_count);
        for _ in 0..2 {
            let actual = self.draw(body, next, user_count);
            assert!(actual == expected, "linked fragment tail pixels differ from the joined program");
        }
        expected
    }
}

/// A vertex program (merged layout: user data from s8, the vertex index
/// in v5) calling its fetch subroutine through s[8:9] (s_swappc_b64
/// s[2:3], s[8:9]): the subroutine loads the position with
/// buffer_load_format_xyzw from the V# in s[12:15] and returns through
/// s[2:3]; the program exports it. The callee is linked in.
#[test]
#[ignore]
fn user_data_tail_vertex_fetch_subroutine() {
    let mut g = Guest::new(64 << 20);
    let vs = g.program(&[0xbe82_2108, 0xf800_08cf, 0x0b0a_0908, asm::S_ENDPGM]);
    // buffer_load_format_xyzw v[8:11], v5, s[12:15], 0 idxen; s_waitcnt
    // vmcnt(0); s_setpc_b64 s[2:3].
    let fetch = g.program(&[0xe00c_2000, 0x8003_0805, 0xbf8c_3f70, 0xbe80_2002, S_CODE_END]);
    let ps = solid_program(&mut g);
    let verts = g.alloc(&floats(&[-1.0, -1.0, 0.5, 1.0, 1.0, -1.0, 0.5, 1.0, -1.0, 1.0, 0.5, 1.0]));
    let spirv_for = Arc::new(Mutex::new(vec![vs]));
    let listed = spirv_for.clone();
    let rest = TestCompiler::new().with(ps, TestProgram::SolidPixel);
    let mut rig = Rig::with_compiler(&g, |dev| Arc::new(Listed { spirv: crate::shader::SpirvCompiler::for_device(dev, g.mem), spirv_for: listed, rest }), None, None);
    let target = rgba8_target(g.alloc(&vec![KEPT; (64 * H * 4) as usize]), W, H);
    let v_sharp = [verts as u32, ((verts >> 32) as u32 & 0xffff) | (16 << 16), 3, 0x0104_dfac];
    let user = [fetch as u32, (fetch >> 32) as u32, 0, 0, v_sharp[0], v_sharp[1], v_sharp[2], v_sharp[3]];
    let d = draw_into(target, stage(vs, &user), stage(ps, &color_words([0.0, 1.0, 0.0, 1.0])), primitive::TRI_LIST, auto(3, 1), None);
    assert_eq!(rig.draw(&d), DrawOutcome::Recorded);
    let p = rig.pixels(&target);
    assert_eq!(px(&p, W, 2, 2), [0, 255, 0, 255], "inside the fetched triangle");
    assert_eq!(px(&p, W, 14, 14), [KEPT; 4], "outside the fetched triangle");
    drop(spirv_for);
}

fn write_words(addr: u64, words: &[u32]) {
    let bytes = asm::bytes(words);
    // SAFETY: a chunk of the test's guest view, read by the GPU only
    // through uploads made after this write.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), addr as *mut u8, bytes.len()) };
}

#[test]
#[ignore]
fn user_data_tail_fragment_programs_continue_in_user_data_code() {
    let mut g = Guest::new(64 << 20);
    let vs = g.program(&[asm::S_ENDPGM]);
    let padded = |words: &[u32]| {
        let mut w = words.to_vec();
        w.resize(64, S_CODE_END);
        w
    };
    let (epilog, other) = (g.program(&padded(&EPILOG)), g.program(&padded(&OTHER_EPILOG)));
    let verts = g.alloc(&floats(&[-1.0, -1.0, 0.5, 1.0, 1.0, -1.0, 0.5, 1.0, -1.0, 1.0, 0.5, 1.0]));
    let spirv_for = Arc::new(Mutex::new(Vec::new()));
    let listed = spirv_for.clone();
    let rest = TestCompiler::new().with(vs, TestProgram::PassthroughVertex { inputs: 1 });
    let rig = Rig::with_compiler(
        &g,
        |dev| Arc::new(Listed { spirv: crate::shader::SpirvCompiler::for_device(dev, g.mem), spirv_for: listed, rest }),
        Some(vec4_attrs(&[(verts, 48, 16, false)], &[(0, 0, 0)])),
        None,
    );
    let mut b = Bench { g, rig, vs, spirv_for };
    let high = (epilog >> 32) as u32;
    // s_mov_b32 s8, s0; v_mov_b32 v0, s1; v_mov_b32 v1, 0;
    // s_cmp_eq_u32 s5, s5; s_mov_b32 s5, high.
    let mut prefix = vec![0xbe880300, 0x7e000201, 0x7e020280, 0xbf060505, 0xbe8503ff, high];
    let original = b.check(&prefix, epilog, &EPILOG, 6);
    let kept = original.chunks(4).filter(|p| p[0] == KEPT).count();
    assert!(kept != 0 && kept != (W * H) as usize, "the triangle did not cross the target");
    assert!(original.chunks(4).any(|p| p[0] == 0xff), "the continuation did not export its carried scalar colour");
    assert!(b.check(&prefix, epilog, &EPILOG, 5) == original, "a constant high half needed a second user-data word");
    // Another continuation.
    *prefix.last_mut().unwrap() = (other >> 32) as u32;
    let changed = b.check(&prefix, other, &OTHER_EPILOG, 6);
    assert!(changed != original, "another continuation did not change the pixels");
    // The continuation's code rewritten in place, then restored.
    *prefix.last_mut().unwrap() = high;
    write_words(epilog, &OTHER_EPILOG);
    crate::memory::note_sync_point();
    assert!(b.check(&prefix, epilog, &OTHER_EPILOG, 6) == changed, "the rewritten continuation kept the old pixels");
    write_words(epilog, &EPILOG);
    crate::memory::note_sync_point();
    assert!(b.check(&prefix, epilog, &EPILOG, 6) == original, "the restored continuation did not restore the pixels");
    // SCC and EXEC carry across the jump.
    prefix[3] = 0xbf070505;
    assert!(b.check(&prefix, epilog, &EPILOG, 6) != original, "SCC was not kept across the jump");
    prefix[3] = 0xbf060505;
    prefix.push(0xbefe0480);
    let empty = b.check(&prefix, epilog, &EPILOG, 6);
    let written: Vec<&[u8]> = empty.chunks(4).filter(|p| p[0] != KEPT).take(4).collect();
    assert!(written.is_empty(), "EXEC was not kept across the jump: {} written, e.g. {written:?}", empty.chunks(4).filter(|p| p[0] != KEPT).count());
}

