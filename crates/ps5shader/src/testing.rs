// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Test support: a small RDNA2 instruction encoder and a runner that
//! compiles raw machine code and executes it in the IR interpreter, so
//! instruction semantics can be checked against reference vectors.

use crate::interp::Machine;
use crate::program::decode_blob;
use crate::ssa::{Function, Stage, SystemValue};
use crate::translate::{compile, FloatMode, InputValue, TranslateOptions};

/// Instruction encoders (field layouts of the gfx10 ISA).
pub mod asm {
    pub const fn inline_u32(value: u32) -> u32 {
        128 + value
    }

    /// Negative integer inline constant -1..=-16.
    pub const fn inline_neg(value: u32) -> u32 {
        192 + value
    }

    pub const fn vgpr(reg: u32) -> u32 {
        256 + reg
    }

    pub const LITERAL: u32 = 255;
    pub const VCC_LO: u32 = 106;
    pub const VCC_HI: u32 = 107;
    pub const M0: u32 = 124;
    pub const NULL: u32 = 125;
    pub const EXEC_LO: u32 = 126;
    pub const EXEC_HI: u32 = 127;
    pub const SCC: u32 = 253;
    pub const F_HALF: u32 = 240;
    pub const F_NEG_HALF: u32 = 241;
    pub const F_ONE: u32 = 242;
    pub const F_NEG_ONE: u32 = 243;
    pub const F_TWO: u32 = 244;
    pub const F_NEG_TWO: u32 = 245;
    pub const F_FOUR: u32 = 246;
    pub const F_NEG_FOUR: u32 = 247;

    pub const fn s_mov_b32(dst: u32, src: u32) -> u32 {
        sop1(0x03, dst, src)
    }

    pub const fn sop1(op: u32, dst: u32, src: u32) -> u32 {
        0x8000_0000 | (0x7D << 23) | ((dst & 0x7F) << 16) | ((op & 0xFF) << 8) | (src & 0xFF)
    }

    pub const fn sop2(op: u32, dst: u32, src0: u32, src1: u32) -> u32 {
        0x8000_0000 | ((op & 0x7F) << 23) | ((dst & 0x7F) << 16) | ((src1 & 0xFF) << 8) | (src0 & 0xFF)
    }

    pub const fn sopc(op: u32, src0: u32, src1: u32) -> u32 {
        0x8000_0000 | (0x7E << 23) | ((op & 0x7F) << 16) | ((src1 & 0xFF) << 8) | (src0 & 0xFF)
    }

    pub const fn sopp(op: u32, simm: u32) -> u32 {
        0x8000_0000 | (0x7F << 23) | ((op & 0x7F) << 16) | (simm & 0xFFFF)
    }

    pub const fn sopk(op: u32, dst: u32, imm: u32) -> u32 {
        0x8000_0000 | (((op + 0x60) & 0x7F) << 23) | ((dst & 0x7F) << 16) | (imm & 0xFFFF)
    }

    pub const fn vop1(op: u32, dst: u32, src0: u32) -> u32 {
        (0x3F << 25) | ((dst & 0xFF) << 17) | ((op & 0xFF) << 9) | (src0 & 0x1FF)
    }

    pub const fn vop2(op: u32, dst: u32, src0: u32, vsrc1: u32) -> u32 {
        ((op & 0x3F) << 25) | ((dst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF)
    }

    pub const fn vopc(op: u32, src0: u32, vsrc1: u32) -> u32 {
        (0x3E << 25) | ((op & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF)
    }

    /// VOP3 (VOP3A) word 0.
    pub const fn vop3_w0(op: u32, dst: u32, abs: u32, op_sel: u32, clamp: bool) -> u32 {
        (0x35 << 26) | ((op & 0x3FF) << 16) | ((abs & 7) << 8) | ((op_sel & 0xF) << 11) | if clamp { 1 << 15 } else { 0 } | (dst & 0xFF)
    }

    /// VOP3B word 0 (scalar carry destination).
    pub const fn vop3b_w0(op: u32, vdst: u32, sdst: u32, clamp: bool) -> u32 {
        (0x35 << 26) | ((op & 0x3FF) << 16) | ((sdst & 0x7F) << 8) | if clamp { 1 << 15 } else { 0 } | (vdst & 0xFF)
    }

    pub const fn vop3_w1(src0: u32, src1: u32, src2: u32, omod: u32, neg: u32) -> u32 {
        (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((src2 & 0x1FF) << 18) | ((omod & 3) << 27) | ((neg & 7) << 29)
    }

    /// Plain VOP3: `[w0, w1]`.
    pub fn vop3(op: u32, dst: u32, src0: u32, src1: u32, src2: u32) -> Vec<u32> {
        vec![vop3_w0(op, dst, 0, 0, false), vop3_w1(src0, src1, src2, 0, 0)]
    }

    pub fn vop3_mod(op: u32, dst: u32, src: [u32; 3], abs: u32, neg: u32, op_sel: u32, clamp: bool, omod: u32) -> Vec<u32> {
        vec![vop3_w0(op, dst, abs, op_sel, clamp), vop3_w1(src[0], src[1], src[2], omod, neg)]
    }

    pub fn vop3b(op: u32, vdst: u32, sdst: u32, src0: u32, src1: u32, src2: u32) -> Vec<u32> {
        vec![vop3b_w0(op, vdst, sdst, false), vop3_w1(src0, src1, src2, 0, 0)]
    }

    pub const fn vop3p_w0(op: u32, dst: u32, op_sel_hi: u32, op_sel: u32, neg_hi: u32, clamp: bool) -> u32 {
        (0x33 << 26) | ((op & 0x7F) << 16) | ((neg_hi & 7) << 8) | ((op_sel & 7) << 11) | (((op_sel_hi >> 2) & 1) << 14) | if clamp { 1 << 15 } else { 0 } | (dst & 0xFF)
    }

    pub const fn vop3p_w1(src0: u32, src1: u32, src2: u32, op_sel_hi: u32, neg: u32) -> u32 {
        (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((src2 & 0x1FF) << 18) | ((op_sel_hi & 3) << 27) | ((neg & 7) << 29)
    }

    pub fn vop3p(op: u32, dst: u32, src: [u32; 3], op_sel: u32, op_sel_hi: u32, neg: u32, neg_hi: u32) -> Vec<u32> {
        vec![vop3p_w0(op, dst, op_sel_hi, op_sel, neg_hi, false), vop3p_w1(src[0], src[1], src[2], op_sel_hi, neg)]
    }

    /// SDWA second dword for VOP1/VOP2.
    #[allow(clippy::too_many_arguments)]
    pub const fn sdwa(src0: u32, dst_sel: u32, dst_unused: u32, src0_sel: u32, src1_sel: u32, src0_sext: bool, src1_sext: bool, clamp: bool) -> u32 {
        (src0 & 0xFF) | ((dst_sel & 7) << 8) | ((dst_unused & 3) << 11) | if clamp { 1 << 13 } else { 0 } | ((src0_sel & 7) << 16) | if src0_sext { 1 << 19 } else { 0 } | ((src1_sel & 7) << 24) | if src1_sext { 1 << 27 } else { 0 }
    }

    /// DPP16 second dword.
    pub const fn dpp(src0: u32, ctrl: u32, row_mask: u32, bank_mask: u32, bound_ctrl: bool, fetch_inactive: bool) -> u32 {
        (src0 & 0xFF) | ((ctrl & 0x1FF) << 8) | if fetch_inactive { 1 << 18 } else { 0 } | if bound_ctrl { 1 << 19 } else { 0 } | ((bank_mask & 0xF) << 24) | ((row_mask & 0xF) << 28)
    }

    pub const fn mubuf_w0(op: u32, offset: u32, idxen: bool, offen: bool, glc: bool) -> u32 {
        (0x38 << 26) | ((op & 0x7F) << 18) | (((op >> 7) & 1) << 25) | if offen { 1 << 12 } else { 0 } | if idxen { 1 << 13 } else { 0 } | if glc { 1 << 14 } else { 0 } | (offset & 0xFFF)
    }

    /// `srsrc` in units of four SGPRs.
    pub const fn mubuf_w1(vdata: u32, srsrc: u32, vaddr: u32, soffset: u32) -> u32 {
        ((soffset & 0xFF) << 24) | ((srsrc & 0x1F) << 16) | ((vdata & 0xFF) << 8) | (vaddr & 0xFF)
    }

    pub fn mubuf(op: u32, vdata: u32, vaddr: u32, srsrc: u32, soffset: u32, offset: u32, idxen: bool, offen: bool, glc: bool) -> Vec<u32> {
        vec![mubuf_w0(op, offset, idxen, offen, glc), mubuf_w1(vdata, srsrc, vaddr, soffset)]
    }

    pub const fn mtbuf_w0(op: u32, format: u32, offset: u32, idxen: bool, offen: bool) -> u32 {
        (0x3A << 26) | (offset & 0xFFF) | if offen { 1 << 12 } else { 0 } | if idxen { 1 << 13 } else { 0 } | ((op & 7) << 16) | ((format & 0x7F) << 19)
    }

    pub const fn mtbuf_w1(op: u32, vdata: u32, srsrc: u32, vaddr: u32, soffset: u32) -> u32 {
        (((op >> 3) & 1) << 21) | ((soffset & 0xFF) << 24) | ((srsrc & 0x1F) << 16) | ((vdata & 0xFF) << 8) | (vaddr & 0xFF)
    }

    /// SMEM: `sbase` is the SGPR pair index (register / 2).
    pub fn smem(op: u32, dst: u32, sbase_pair: u32, offset: u32, soffset: u32) -> Vec<u32> {
        vec![(0x3D << 26) | ((op & 0xFF) << 18) | ((dst & 0x7F) << 6) | (sbase_pair & 0x3F), (offset & 0x1F_FFFF) | ((soffset & 0x7F) << 25)]
    }

    pub fn ds(op: u32, vdst: u32, addr: u32, data0: u32, data1: u32, offset0: u32, offset1: u32) -> Vec<u32> {
        vec![(0x36 << 26) | ((op & 0xFF) << 18) | ((offset1 & 0xFF) << 8) | (offset0 & 0xFF), ((vdst & 0xFF) << 24) | ((data1 & 0xFF) << 16) | ((data0 & 0xFF) << 8) | (addr & 0xFF)]
    }

    /// FLAT/GLOBAL/SCRATCH (`segment` 0 flat, 1 scratch, 2 global).
    pub fn flat(op: u32, segment: u32, vdst: u32, vaddr: u32, vdata: u32, saddr: u32, offset: i32, glc: bool) -> Vec<u32> {
        vec![(0x37 << 26) | ((op & 0x7F) << 18) | ((segment & 3) << 14) | if glc { 1 << 16 } else { 0 } | (offset as u32 & 0xFFF), ((vdst & 0xFF) << 24) | ((saddr & 0x7F) << 16) | ((vdata & 0xFF) << 8) | (vaddr & 0xFF)]
    }

    pub const S_ENDPGM: u32 = 0xBF81_0000;

    /// s_branch / s_cbranch_*: `op` SOPP opcode, `delta` dwords from the next
    /// instruction.
    pub const fn branch(op: u32, delta: i32) -> u32 {
        sopp(op, delta as u32 & 0xFFFF)
    }

    // Common opcodes.
    pub const OP_S_MOV_B32: u32 = 0x03;
    pub const OP_V_MOV_B32: u32 = 0x01;
    pub const OP_BUFFER_LOAD_DWORD: u32 = 0x0C;
    pub const OP_BUFFER_STORE_DWORD: u32 = 0x1C;
    pub const SOPP_BRANCH: u32 = 0x02;
    pub const SOPP_CBRANCH_SCC0: u32 = 0x04;
    pub const SOPP_CBRANCH_SCC1: u32 = 0x05;
    pub const SOPP_CBRANCH_VCCZ: u32 = 0x06;
    pub const SOPP_CBRANCH_VCCNZ: u32 = 0x07;
    pub const SOPP_CBRANCH_EXECZ: u32 = 0x08;
    pub const SOPP_CBRANCH_EXECNZ: u32 = 0x09;
}

/// A buffer descriptor (V#) with a stride and record count. `dword3`
/// supplies DST_SEL / FORMAT / OOB_SELECT bits.
pub fn buffer_descriptor(base: u64, stride: u32, num_records: u32, dword3: u32) -> [u32; 4] {
    [base as u32, ((base >> 32) as u32 & 0xFFFF) | ((stride & 0x3FFF) << 16), num_records, dword3]
}

/// A raw (untyped, stride 0) buffer descriptor of `bytes` bytes.
pub fn raw_buffer_descriptor(base: u64, bytes: u32) -> [u32; 4] {
    buffer_descriptor(base, 0, bytes, (3 << 28) | 0x0001_6FAC)
}

/// A compute dispatch of one wave.
#[derive(Clone, Debug)]
pub struct ComputeRun {
    pub code: Vec<u32>,
    pub wave_size: u32,
    /// Invocations in the wave.
    pub threads: u32,
    pub user_data: Vec<u32>,
    /// Address the code object is mapped at (S_GETPC_B64 bases).
    pub code_address: u64,
    /// v0..v(n-1) receive the local invocation id components.
    pub thread_id_vgprs: u32,
    /// Extra initial SGPRs after the user data (work-group ids).
    pub workgroup_id_sgprs: u32,
    /// RSRC1 float controls the program is compiled for.
    pub float_mode: FloatMode,
    /// Float rules the program is translated with (`float_rules::ALL`).
    pub float_rules: u32,
    /// Equivalent rewrites the program is translated with
    /// (`rewrites::ALL`).
    pub rewrites: u32,
    /// The host flushes f32 denormals (see
    /// [`crate::translate::native_ftz_applies`]): the interpreter then runs
    /// without the flush code (tests of the emulated semantics keep it
    /// off).
    pub native_ftz: bool,
}

impl ComputeRun {
    pub fn new(code: Vec<u32>, wave_size: u32, threads: u32, user_data: Vec<u32>) -> Self {
        ComputeRun { code, wave_size, threads, user_data, code_address: 0x10_0000_0000, thread_id_vgprs: 1, workgroup_id_sgprs: 0, float_mode: FloatMode::default(), float_rules: crate::translate::float_rules::ALL, rewrites: crate::translate::rewrites::ALL, native_ftz: false }
    }

    pub fn options(&self, optimize: bool) -> TranslateOptions {
        let mut o = TranslateOptions::with_user_data(self.wave_size, Stage::Compute, self.user_data.len() as u32);
        let n = self.user_data.len() as u8;
        for k in 0..self.workgroup_id_sgprs as u8 {
            o.sgprs.push((n + k, InputValue::System(SystemValue::WorkgroupId(k))));
        }
        o.vgprs = (0..self.thread_id_vgprs as u16).map(|a| (a, InputValue::System(SystemValue::LocalInvocationId(a as u8)))).collect();
        o.optimize = optimize;
        o.float_mode = self.float_mode;
        o.float_rules = self.float_rules;
        o.rewrites = self.rewrites;
        o.native_ftz = self.native_ftz;
        // Tests check the modelled semantics: unmodelled MODE writes refuse.
        o.strict_mode_writes = true;
        o
    }

    /// Compile (`optimize` selects the optimization pipeline).
    pub fn compile(&self, optimize: bool) -> Result<Function, String> {
        let bytes: Vec<u8> = self.code.iter().flat_map(|w| w.to_le_bytes()).collect();
        let program = decode_blob(&bytes, self.code_address).map_err(|e| format!("decode: {e}"))?;
        let addr = self.code_address;
        let code = self.code.clone();
        let reader = move |a: u64| -> Option<u32> {
            let off = a.checked_sub(addr)?;
            if off % 4 != 0 {
                return None;
            }
            code.get((off / 4) as usize).copied()
        };
        let c = compile(&program, &self.options(optimize), Some(&reader)).map_err(|e| format!("compile: {e}"))?;
        Ok(c.function)
    }

    fn machine(&self) -> Machine<'static> {
        let mut m = Machine::new(self.wave_size, self.threads);
        m.user_data = self.user_data.clone();
        m.memory.write_words(self.code_address, &self.code);
        let ids: Vec<u32> = (0..self.threads).collect();
        m.system.insert(SystemValue::LocalInvocationId(0), ids);
        m.system.insert(SystemValue::LocalInvocationId(1), vec![0; self.threads as usize]);
        m.system.insert(SystemValue::LocalInvocationId(2), vec![0; self.threads as usize]);
        m
    }

    /// Compile and run with `setup` preparing memory; returns the machine.
    pub fn run_with(&self, optimize: bool, setup: &dyn Fn(&mut Machine)) -> Result<Machine<'static>, String> {
        let f = self.compile(optimize)?;
        self.run_function(&f, setup)
    }

    /// Run an already compiled (and possibly further transformed) function
    /// with `setup` preparing memory; returns the machine.
    pub fn run_function(&self, f: &Function, setup: &dyn Fn(&mut Machine)) -> Result<Machine<'static>, String> {
        let mut m = self.machine();
        setup(&mut m);
        m.run(f).map_err(|e| format!("run: {e}\n{f}"))?;
        Ok(m)
    }

    /// Run both unoptimized and optimized; the memory must agree. Returns
    /// the optimized run.
    pub fn run_checked(&self, setup: &dyn Fn(&mut Machine), outputs: &[(u64, usize)]) -> Result<Machine<'static>, String> {
        let plain = self.run_with(false, setup)?;
        let opt = self.run_with(true, setup)?;
        for &(addr, n) in outputs {
            let a = plain.memory.read_words(addr, n);
            let b = opt.memory.read_words(addr, n);
            if a != b {
                let i = a.iter().zip(&b).position(|(x, y)| x != y).unwrap();
                return Err(format!("optimized run differs at word {i}: {:#010x} vs {:#010x}", a[i], b[i]));
            }
        }
        Ok(opt)
    }
}

/// The execution test layout: s[0:3] = input V#, s[4:7] =
/// output V# (stride 4 bytes, `dwords` records each), v0 = thread id,
/// wave32 with `threads` invocations.
pub struct IoTest {
    pub run: ComputeRun,
    pub input_base: u64,
    pub output_base: u64,
    pub dwords: usize,
}

pub const INPUT_BASE: u64 = 0x1_0000_0000;
pub const OUTPUT_BASE: u64 = 0x2_0000_0000;

impl IoTest {
    pub fn new(code: &[u32], threads: u32, dwords: usize) -> Self {
        Self::with_wave(code, 32, threads, dwords)
    }

    pub fn with_wave(code: &[u32], wave_size: u32, threads: u32, dwords: usize) -> Self {
        let input = buffer_descriptor(INPUT_BASE, 4, dwords as u32, 0x1101_6FAC);
        let output = buffer_descriptor(OUTPUT_BASE, 4, dwords as u32, 0x1101_6FAC);
        let mut user = vec![0u32; 8];
        user[..4].copy_from_slice(&input);
        user[4..].copy_from_slice(&output);
        IoTest { run: ComputeRun::new(code.to_vec(), wave_size, threads, user), input_base: INPUT_BASE, output_base: OUTPUT_BASE, dwords }
    }

    /// The rows were measured with these float controls.
    pub fn with_float_mode(mut self, mode: FloatMode) -> Self {
        self.run.float_mode = mode;
        self
    }

    /// Run with `input`; returns the output words (prefilled 0xdeadbeef),
    /// after checking the optimized and unoptimized runs agree.
    pub fn run(&self, input: &[u32]) -> Result<Vec<u32>, String> {
        let (ib, ob, n) = (self.input_base, self.output_base, self.dwords);
        let setup = move |m: &mut Machine| {
            m.memory.write_words(ib, input);
            m.memory.write_words(ob, &vec![0xDEAD_BEEF; n]);
        };
        let m = self.run.run_checked(&setup, &[(ob, n)])?;
        Ok(m.memory.read_words(ob, n))
    }
}

/// Guest programs and input rows that reach the edges of the f32 / f16
/// special-value handling (shared by the interpreter and device tests of
/// the equivalent rewrites).
pub mod float_rows {
    /// A chain of dependent f32 operations (each result is a source of the
    /// next, so a flushed result meets the next instruction's input flush),
    /// with output modifiers, clamp, min/max/med3, selects, a class compare,
    /// an integer conversion and a partial EXEC; v8..v31 stored per lane at
    /// tid*128. Assembled with llvm-mc (gfx1013, wave32):
    /// v1 = tid*16, v3 = tid*128; buffer_load_dwordx4 v[4:7]; v8 = v4*v5;
    /// v9 = v8+v6; v10 = v9*v8; v11 = fma(v10,v5,v7); v12 = mad(v11,v4,v8);
    /// v13 = min(v12,v4); v14 = max(v13,v9); v15 = v14-v10; v16 = v15*v15;
    /// v17 = floor(v16); v18 = v17*v7; v19 = mul_legacy(v18,v6); v20 =
    /// exp(v19); v21 = rcp(v20); v22 = v21*v8 div:2; v23 = v22+v5 clamp;
    /// v24 = med3(v23,v11,v12); v25 = v24 < v8 ? v10 : v9; v26 = v25*-v4;
    /// v26 += v5*v6 (mac); v27 = sqrt(v26); v28 = |v27|*v7 mul:4; v29 =
    /// fract(v28); v30 = cvt_f32_i32(v4)*v29; v31 = class(v30, NaN) ? v30 :
    /// v28; lanes with v8 > v5: v12 *= v13, v13 = v12+v14, v14 = v13*v13;
    /// then v15 = v14*v12, v16 = v15+v13.
    pub const CHAIN_W32: [u32; 62] = [
        0x34020084, 0x34060087, 0xe0381000, 0x80000401, 0xbf8c3f70, 0x10100b04, 0x06120d08, 0x10141109, 0xd54b000b, 0x041e0b0a, 0xd541000c, 0x0422090b,
        0x1e1a090c, 0x201c130d, 0x081e150e, 0x10201f0f, 0x7e224910, 0x10240f11, 0x0e260d12, 0x7e284b13, 0x7e2a5514, 0xd5080016, 0x18021115, 0xd5038017,
        0x00020b16, 0xd5570018, 0x04321717, 0x7c021118, 0x02321509, 0xd508001a, 0x40020919, 0x3e340d05, 0x7e36671a, 0xd508011c, 0x10020f1b, 0x7e3a411c,
        0x7e3c0b04, 0x103c3b1e, 0xd488006a, 0x0001071e, 0x023e3d1c, 0x7c080b08, 0xbe943c6a, 0x10181b0c, 0x061a1d0c, 0x101c1b0d, 0xbefe0314, 0x101e190e,
        0x06201b0f, 0xe0781000, 0x80010803, 0xe0781010, 0x80010c03, 0xe0781020, 0x80011003, 0xe0781030, 0x80011403, 0xe0781040, 0x80011803, 0xe0781050,
        0x80011c03, 0xbf810000,
    ];

    /// f32 inputs for the exactness checks: a third special values (zeros,
    /// denormals at both ends, the smallest normals, infinities, quiet and
    /// signaling NaNs of both signs and payloads, the largest finite value,
    /// powers of two around 2^±64 whose products land at the denormal edge),
    /// a third values with exponents near the denormal range (products and
    /// sums underflow or round to the smallest normal), a third random bits.
    pub fn special_rows(words: usize, seed: u64) -> Vec<u32> {
        const SPECIAL: [u32; 36] = [
            0, 0x8000_0000, 0x0000_0001, 0x8000_0001, 0x007F_FFFF, 0x807F_FFFF, 0x0080_0000, 0x8080_0000, 0x0080_0001, 0x0040_0000, 0x3F80_0000, 0xBF80_0000,
            0x7F80_0000, 0xFF80_0000, 0x7FC0_0000, 0xFFC0_0000, 0x7F80_0001, 0xFF80_0001, 0x7FBF_FFFF, 0x7FFF_FFFF, 0xFFFF_FFFF, 0x7FC1_2345, 0x7F7F_FFFF,
            0xFF7F_FFFF, 0x1F80_0000, 0x2000_0000, 0x2080_0000, 0x5F80_0000, 0xDF80_0000, 0x3F00_0000, 0x3F7F_FFFF, 0x1FFF_FFFF, 0x0000_7FFF, 0x0012_3456,
            0x3380_0000, 0x4B00_0000,
        ];
        let mut x = seed | 1;
        (0..words)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                match x % 3 {
                    0 => SPECIAL[(x >> 8) as usize % SPECIAL.len()],
                    1 => {
                        // Exponent 0..=0x50 or 0x2F..=0x7F: near the denormal
                        // range or near 2^-64 (products of two land near it).
                        let e = if x & 0x100 != 0 { (x >> 9) % 0x51 } else { 0x2F + (x >> 9) % 0x51 } as u32;
                        ((x >> 32) as u32 & 0x8000_0000) | (e << 23) | ((x >> 20) as u32 & 0x7F_FFFF)
                    }
                    _ => (x >> 20) as u32,
                }
            })
            .collect()
    }

    /// v_fma_mix forms (v4/v5 f16 halves, v6/v7 f32; every op_sel_hi
    /// combination, op_sel, negate / abs, clamp, lo and hi destinations, and
    /// two v_fma_mix_f32): v[8:15] stored at tid*32. Assembled with llvm-mc
    /// (gfx1013): mixlo v8 = v4*v5+v6 [1,1,0]; mixhi v8 = v4*v5+v7 [1,1,0]
    /// clamp; mixlo v9 = v6*v7+v4 [0,0,1]; mixhi v9 = v4.hi*v5.hi+v6.hi
    /// [1,1,1]; mixlo v10 = v6*v7+v5 [0,0,0]; mixhi v10 = v4*v6+v7 [1,0,0];
    /// mixlo v11 = -v4*v5+|v6| [1,1,0]; mixhi v11 = v6*v4.hi+v7 [0,1,0]
    /// clamp; mix_f32 v12 = v4*v5+v6 [1,1,0]; mix_f32 v13 = v6*v7+v4 [0,0,1].
    pub const MIX_W32: [u32; 36] = [
        0x34020084, 0x34060085, 0xe0381000, 0x80000401, 0xbf8c3f70, 0x7e100280, 0x7e120280, 0x7e140280, 0x7e160280, 0xcc210008, 0x1c1a0b04, 0xcc228008,
        0x1c1e0b04, 0xcc214009, 0x04120f06, 0xcc227809, 0x1c1a0b04, 0xcc21000a, 0x04160f06, 0xcc22000a, 0x0c1e0d04, 0xcc21040b, 0x3c1a0b04, 0xcc22900b,
        0x141e0906, 0xcc20000c, 0x1c1a0b04, 0xcc20400d, 0x04120f06, 0x7e1c0280, 0x7e1e0280, 0xe0781000, 0x80010803, 0xe0781010, 0x80010c03, 0xbf810000,
    ];

    /// The value of the f16 bits `h`.
    pub fn half_value(h: u32) -> f64 {
        let e = (h >> 10) & 0x1F;
        let m = (h & 0x3FF) as f64;
        let v = if e == 0 { m * 2f64.powi(-24) } else { (m + 1024.0) * 2f64.powi(e as i32 - 25) };
        if h & 0x8000 != 0 {
            -v
        } else {
            v
        }
    }

    /// Rows of four words for [`MIX_W32`] whose fma sums land on f16
    /// midpoints (where the f16 rounding of the f32 fma can differ from that
    /// of the exact sum) or a hair to either side: v4 = (h, a'), v5 = (1.0,
    /// b'), v6 = (midpoint above h) - h and v7 = (midpoint below h) - h, each
    /// exact or one f32 unit off, for f16 values h of either sign (denormal,
    /// normal, around 1.0 and 65504); every fourth row is special values.
    pub fn midpoint_rows(lanes: usize, seed: u64) -> Vec<u32> {
        let mut x = seed | 1;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut out = Vec::new();
        for lane in 0..lanes {
            if lane % 4 == 3 {
                out.extend(special_rows(4, next()));
                continue;
            }
            let r = next();
            let h = match r % 7 {
                0 => 0x3BFF,
                1 => 0x3C00,
                2 => 0x7BFF,
                3 => 1 + (r >> 8) as u32 % 0x3FF,
                4 => 0x0400 + (r >> 8) as u32 % 0x400,
                5 => 0x0400 * (1 + (r >> 8) as u32 % 0x1E),
                _ => 0x0400 + (r >> 8) as u32 % 0x7800,
            };
            let sign = if r & 0x80 != 0 { 0x8000 } else { 0 };
            let a = half_value(h | sign);
            let ulp = |h: u32| if h < 0x400 { 2f64.powi(-24) } else { 2f64.powi(((h >> 10) as i32) - 25) };
            let up = ulp(h) / 2.0;
            let down = if h == 0x400 || h & 0x3FF != 0 || h < 0x400 { ulp(h - 1) / 2.0 } else { ulp(h) / 4.0 };
            let s = if sign != 0 { -1.0 } else { 1.0 };
            let step = |c: f64, k: u64| -> u32 {
                let b = (c as f32).to_bits();
                match k % 3 {
                    1 => b + 1,
                    2 if b & 0x7FFF_FFFF != 0 => b - 1,
                    _ => b,
                }
            };
            let c = step(s * up, next());
            let c2 = step(-s * down, next());
            let a2 = (next() >> 24) as u32 & 0xFFFF;
            let b2 = (next() >> 24) as u32 & 0xFFFF;
            let _ = a;
            out.extend([h | sign | a2 << 16, 0x3C00 | b2 << 16, c, c2]);
        }
        out
    }
}
