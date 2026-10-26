// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Instruction translation: decoded RDNA program + structured control flow
//! → register-level IR → SSA.
//!
//! Model (see [`crate::ssa`]): one IR invocation per guest lane. Branches
//! are wave-uniform: SCC is uniform, and VCCZ / EXECZ are computed from the
//! lane-mask words, which are ballots. Every invocation therefore follows
//! the same path; EXEC is data. Each invocation keeps its own EXEC bit
//! (`Reg::SgprBit(126)`), vector register writes select the new value only
//! where it is set, memory side effects take it as a predicate, and
//! lane-mask SGPRs (VCC, saved EXEC, compare results) keep both their words
//! and, while valid, this invocation's bit, so a compare feeding a
//! `v_cndmask` needs no ballot round trip.
//!
//! Exactness notes: float results follow hardware-measured RDNA2 semantics
//! (denormal flushing of transcendental inputs, output
//! modifiers flushing denormals, NaN quieting, saturating conversions,
//! min/max with -0 < +0); transcendental functions themselves use the
//! host's precision. Shader float denormals are otherwise preserved.

pub mod bind;
mod extra;
mod float;
mod memory;
pub(crate) mod operand;
pub mod rounding;
mod scalar;
mod vector;

use crate::cfg::{self, structurize, BranchCond, Cfg, CfgError, Cond, Node, Structured};
use crate::ir_types::*;
use crate::opcode::Opcode;
use crate::program::CodeSource;
use crate::ssa::*;
use std::collections::BTreeSet;
use std::fmt;

/// Source of an initial register value at wave launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputValue {
    /// User data SGPR `n` of the dispatch / draw.
    UserData(u32),
    System(SystemValue),
    Const(u32),
}

/// Initial EXEC.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitialExec {
    /// Every invocation that runs is active (compute, vertex).
    All,
    /// Active where the system value is zero (pixel helper invocations).
    NotSystem(SystemValue),
}

/// The float controls of SPI_SHADER_PGM_RSRC1 / COMPUTE_PGM_RSRC1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FloatMode {
    /// FLOAT_MODE (bits 19:12): [3:0] round modes, [5:4] f32 denormals,
    /// [7:6] f16/f64 denormals (0 flush, 3 keep).
    pub mode: u8,
    /// DX10_CLAMP (bit 21): the clamp modifier sends NaN to 0.
    pub dx10_clamp: bool,
    /// IEEE_MODE (bit 23): signaling NaNs are quieted and returned by
    /// min/max; output modifiers are ignored.
    pub ieee: bool,
    /// FP16_OVFL (bit 26 of the compute RSRC1, 29 pixel, 30 hull, 31
    /// vertex / geometry): f16 results that would round to infinity from
    /// finite sources are the largest finite value instead.
    pub fp16_overflow: bool,
}

impl Default for FloatMode {
    /// What compiled shaders use: f32 denormals flushed, f16/f64 kept,
    /// DX10 clamp, no IEEE mode.
    fn default() -> Self {
        FloatMode { mode: 0xC0, dx10_clamp: true, ieee: false, fp16_overflow: false }
    }
}

impl FloatMode {
    /// The float controls of an SPI_SHADER_PGM_RSRC1 / COMPUTE_PGM_RSRC1
    /// value; `fp16_overflow_bit` is the stage's FP16_OVFL bit (see
    /// [`FloatMode::fp16_overflow_bit`]).
    pub fn from_rsrc1(rsrc1: u32, fp16_overflow_bit: u32) -> Self {
        FloatMode { mode: (rsrc1 >> 12) as u8, dx10_clamp: rsrc1 & (1 << 21) != 0, ieee: rsrc1 & (1 << 23) != 0, fp16_overflow: rsrc1 >> fp16_overflow_bit & 1 != 0 }
    }

    /// FP16_OVFL's bit in the RSRC1 register of a stage (gfx10): 26 in
    /// COMPUTE_PGM_RSRC1, 29 for pixel, 30 for hull, 31 for vertex,
    /// export and geometry (NGG) programs.
    pub fn fp16_overflow_bit(stage: Stage) -> u32 {
        match stage {
            Stage::Compute => 26,
            Stage::Pixel => 29,
            Stage::Hull => 30,
            _ => 31,
        }
    }

    /// The f32 denormal mode (FLOAT_MODE[5:4]).
    pub fn f32_denormals(self) -> u8 {
        (self.mode >> 4) & 3
    }

    /// Output modifiers apply only without IEEE mode and with f32 output
    /// denormals flushed (FLOAT_MODE[5] clear).
    pub fn output_modifier_applies(self) -> bool {
        !self.ieee && self.f32_denormals() & 2 == 0
    }
}

/// Translation parameters.
#[derive(Clone, Debug)]
pub struct TranslateOptions {
    pub wave_size: u32,
    pub stage: Stage,
    /// Initial SGPR values.
    pub sgprs: Vec<(u8, InputValue)>,
    /// Initial VGPR values.
    pub vgprs: Vec<(u16, InputValue)>,
    pub initial_exec: InitialExec,
    /// Run the optimization pipeline after SSA construction.
    pub optimize: bool,
    /// Replace S_GETPC_B64 bases by the known code address (enables folding
    /// constant tables embedded in the code object).
    pub fold_code_address: bool,
    pub float_mode: FloatMode,
    /// Round-field writes (`s_round_mode`, `s_setreg_imm32_b32` /
    /// `s_setreg_b32` of MODE[3:0]) are tracked through the control flow
    /// ([`rounding`]). Strict: refuse what is not modelled (an instruction
    /// reading round bits that differ between paths or come from an SGPR,
    /// a directed mode set at run time for an instruction that only rounds
    /// to nearest, an f16 conversion in a run-time round mode with f16
    /// denormals flushed, writes of other MODE fields or hardware
    /// registers). Off: those instructions run with the declared bits in
    /// place of the unknown ones, and other writes are ignored (a refused
    /// program loses its draws).
    pub strict_mode_writes: bool,
    /// Which of the hardware-measured float rules apply ([`float_rules`]
    /// bits; [`float_rules::ALL`] by default). A cleared bit keeps the
    /// rule's earlier translation.
    pub float_rules: u32,
    /// Equivalent rewrites that only make the translation cheaper on the
    /// device ([`rewrites`] bits; [`rewrites::ALL`] by default). A cleared
    /// bit keeps the earlier translation.
    pub rewrites: u32,
    /// v_exp/log/rcp/rsq/sqrt_f32 as the host's operation with one NaN
    /// select, without the hardware's special-value and range rules (not
    /// exact: opt-in only).
    pub fast_transcendental: bool,
    /// The host flushes f32 denormals (SPIR-V DenormFlushToZero(32)): no
    /// flush code is translated. Only set for a module whose FLOAT_MODE
    /// flushes f32 inputs and results everywhere and whose opcodes all
    /// agree with the native semantics ([`native_ftz_applies`]).
    pub native_ftz: bool,
}

/// Bits of [`TranslateOptions::rewrites`]: translations equal bit for bit
/// to the ones they replace.
pub mod rewrites {
    /// The f32 denormal flush tests `|x| < 2^-126` alone: a zero is its own
    /// sign, so the separate nonzero test is redundant.
    pub const LEAN_FLUSH: u32 = 1 << 0;
    /// Whether a DPP source lane has EXEC set is a bit test of the EXEC
    /// words, not a shuffle of every lane's EXEC bit.
    pub const LANE_ACTIVE_EXEC: u32 = 1 << 1;
    /// The f32 denormal flush of a value that is never a denormal (the
    /// result of an earlier flush through EXEC selects and phis, see
    /// [`crate::opt::denorm`]) is removed.
    pub const FLUSH_ELISION: u32 = 1 << 2;
    /// The f32 denormal flush tests `|x| < 2^-126` and the NaN tests of the
    /// f32 arithmetic results are float compares (one host instruction
    /// with the absolute value folded) instead of integer tests of the
    /// magnitude bits. Needs a host that preserves NaNs in compares
    /// (SignedZeroInfNanPreserve).
    pub const FLOAT_CLASS_COMPARE: u32 = 1 << 3;
    /// v_fma_mix_lo/hi_f16 round the f32 fma to f16 where that gives the
    /// f16 of the exact sum (the fma is not an f16 midpoint, not a NaN, and
    /// no f32 source is a kept denormal); the f64 round-to-odd sum only
    /// otherwise, as the lazy arm of a select
    /// ([`crate::ssa::Function::lazy_selects`]).
    pub const HALF_FMA_FAST: u32 = 1 << 4;
    pub const ALL: u32 = LEAN_FLUSH | LANE_ACTIVE_EXEC | FLUSH_ELISION | FLOAT_CLASS_COMPARE | HALF_FMA_FAST;
}

/// Bits of [`TranslateOptions::float_rules`].
pub mod float_rules {
    /// A flushed tiny product that rounds up to the smallest normal (in
    /// the current round mode) is that normal, not zero.
    pub const TINY_AFTER_ROUNDING: u32 = 1 << 0;
    /// v_div_scale_f32 flushes its input denormals in the flush modes.
    pub const DIV_SCALE_FLUSH: u32 = 1 << 1;
    /// v_div_fmas_f32 picks its NaN result like v_fma_f32 (quieted only
    /// with IEEE_MODE, a NaN addend not returned for inf * 0).
    pub const DIV_FMAS_NAN: u32 = 1 << 2;
    /// The f16/f64 denormal mode (FLOAT_MODE[7:6]: bit 6 keeps input
    /// denormals, bit 7 results) applies where it was measured: f16
    /// min/max/min3/max3/med3 inputs, v_add/sub/mul_f16 and the f16 fused
    /// multiply-adds (round to nearest), f16 unary operations, frexp,
    /// conversions and compares, packed f16, v_pack_b32_f16 /
    /// v_cvt_pkrtz_f16_f32, v_fma_mix f16 sources, v_div_fixup_f16/f64
    /// denominator and numerator, v_ldexp_f16, v_cvt_pknorm_*_f16, and the
    /// inputs of v_min/max_f64 and the f64 compares.
    pub const F16_DENORMALS: u32 = 1 << 3;
    /// v_ldexp_f32 keeps denormal inputs and results when FLOAT_MODE keeps
    /// f32 denormals, rounds tiny results in the f32 round mode and
    /// saturates overflow to the largest finite value in the modes that
    /// round away from that infinity.
    pub const LDEXP_DENORMALS: u32 = 1 << 4;
    /// v_cvt_f32_i32 / v_cvt_f32_u32 round in the f32 round mode (not
    /// always to nearest even).
    pub const INT_TO_F32_ROUND: u32 = 1 << 5;
    /// Not a rule but an equivalent translation: an f32 compare against a
    /// constant folds the denormal flush of its other operand into the
    /// compare (one compare instead of a flush and a compare).
    pub const COMPARE_CONSTANT_FOLD: u32 = 1 << 6;
    /// Not a rule but an equivalent translation: to nearest even, the
    /// tininess of [`TINY_AFTER_ROUNDING`] is decided by one scaled
    /// multiply instead of the integer significand product.
    pub const TINY_ONE_MULTIPLY: u32 = 1 << 7;
    /// Not a rule but an equivalent translation: v_cvt_pknorm_i16/u16_f32
    /// round the scaled value in f32 (exact error by Fast2Sum) instead of
    /// a 64-bit integer product with variable shifts.
    pub const PKNORM_F32: u32 = 1 << 8;
    /// The directed f16 round modes (FLOAT_MODE[3:2] != 0) apply to the
    /// f16 conversions (v_cvt_f16_f32/u16/i16), v_ldexp_f16 and
    /// v_fma_mixlo/mixhi_f16 (measured on RDNA2); the result denormal
    /// flush of the conversions follows the round mode.
    pub const F16_ROUNDING: u32 = 1 << 9;
    /// v_sin_f32 / v_cos_f32 (and the f16 forms computed through them)
    /// take a polynomial in turns where the host's sine / cosine is more
    /// than 4 ulp from it (drivers whose sin / cos lose precision).
    pub const SIN_COS_SERIES: u32 = 1 << 10;
    pub const ALL: u32 = TINY_AFTER_ROUNDING | DIV_SCALE_FLUSH | DIV_FMAS_NAN | F16_DENORMALS | LDEXP_DENORMALS | INT_TO_F32_ROUND | COMPARE_CONSTANT_FOLD | TINY_ONE_MULTIPLY | PKNORM_F32 | F16_ROUNDING | SIN_COS_SERIES;
}

impl TranslateOptions {
    /// Compute shader: `user_sgprs` user-data SGPRs from s0, thread ids in
    /// v0..v2, work-group ids in the following SGPRs.
    pub fn compute(wave_size: u32, user_sgprs: u32) -> Self {
        let mut sgprs: Vec<(u8, InputValue)> = (0..user_sgprs).map(|i| (i as u8, InputValue::UserData(i))).collect();
        for axis in 0..3u8 {
            sgprs.push(((user_sgprs as u8) + axis, InputValue::System(SystemValue::WorkgroupId(axis))));
        }
        let vgprs = (0..3u16).map(|a| (a, InputValue::System(SystemValue::LocalInvocationId(a as u8)))).collect();
        TranslateOptions { wave_size, stage: Stage::Compute, sgprs, vgprs, initial_exec: InitialExec::All, optimize: true, fold_code_address: true, float_mode: FloatMode::default(), strict_mode_writes: false, float_rules: float_rules::ALL, rewrites: rewrites::ALL, fast_transcendental: false, native_ftz: false }
    }

    /// Generic options with only user data (no system values).
    pub fn with_user_data(wave_size: u32, stage: Stage, user_sgprs: u32) -> Self {
        let sgprs = (0..user_sgprs).map(|i| (i as u8, InputValue::UserData(i))).collect();
        TranslateOptions { wave_size, stage, sgprs, vgprs: Vec::new(), initial_exec: InitialExec::All, optimize: true, fold_code_address: true, float_mode: FloatMode::default(), strict_mode_writes: false, float_rules: float_rules::ALL, rewrites: rewrites::ALL, fast_transcendental: false, native_ftz: false }
    }
}

/// Which f32 denormals an instruction flushes (bit 0 inputs, bit 1
/// results). With FLOAT_MODE[5:4] = 0: arithmetic, min/max, rounding and
/// the fused multiply-adds flush both; compares and conversions from f32
/// flush their inputs; moves and everything else keep denormals. Bit 0 of
/// the mode keeps input denormals, bit 1 result denormals.
///
/// The separately rounded multiply-adds (v_mad_f32, v_mac_f32,
/// v_madak_f32, v_madmk_f32) do not support denormals: they flush inputs,
/// product and result in every mode (LLVM forms them only with f32
/// denormals flushed and ACO switches to v_fma_f32 when they are kept).
pub fn f32_denormal_flush(op: Opcode, mode: FloatMode) -> u32 {
    f32_denormal_flush_with(op, mode, float_rules::ALL)
}

/// [`f32_denormal_flush`] under the float rules `rules`.
pub fn f32_denormal_flush_with(op: Opcode, mode: FloatMode, rules: u32) -> u32 {
    use Opcode::*;
    let natural = match op {
        VDivScaleF32 if rules & float_rules::DIV_SCALE_FLUSH != 0 => 1,
        // v_mad_legacy_f32 / v_mac_legacy_f32: measured on gfx1010 in f32
        // denormal modes 0 to 3.
        VMadF32 | VMacF32 | VMadMkF32 | VMadAkF32 | VMadLegacyF32 | VMacLegacyF32 => return 3,
        VAddF32 | VSubF32 | VSubrevF32 | VMulF32 | VMulLegacyF32 | VMullitF32 | VMinF32 | VMaxF32 | VMin3F32 | VMax3F32 | VMed3F32 | VFractF32 | VFloorF32 | VCeilF32
        | VFrexpMantF32 | VSinF32 | VCosF32 | VFmaF32 | VFmacF32 | VFmaAkF32 | VFmaMkF32 | VFmaMixF32 => 3,
        VCmpEqF32 | VCmpGeF32 | VCmpGtF32 | VCmpLeF32 | VCmpLgF32 | VCmpLtF32 | VCmpNeqF32 | VCmpNgeF32 | VCmpNgtF32 | VCmpNleF32 | VCmpNlgF32 | VCmpNltF32
        | VCmpxEqF32 | VCmpxGeF32 | VCmpxGtF32 | VCmpxLeF32 | VCmpxLgF32 | VCmpxLtF32 | VCmpxNeqF32 | VCmpxNgeF32 | VCmpxNgtF32 | VCmpxNleF32 | VCmpxNlgF32
        | VCmpxNltF32 | VCvtI32F32 | VCvtU32F32 | VCvtRpiI32F32 | VCvtFlrI32F32 | VFrexpExpI32F32 | VCvtF64F32 | VCvtPknormI16F32 | VCvtPknormU16F32 | VCvtPkU8F32
        | VCvtPkrtzF16F32 | VCvtF16F32 | VDivFixupF32 | VFmaMixloF16 | VFmaMixhiF16 => 1,
        _ => 0,
    };
    let kept = mode.f32_denormals() as u32;
    natural & !kept & 3
}

/// Opcodes whose f32 semantics differ from a host DenormFlushToZero(32)
/// even in the full-flush FLOAT_MODE: they keep a denormal f32 output (or
/// intermediates) the guest preserves (natural flush 0 with a float
/// result), or read unflushed denormal inputs. With one of them a module
/// keeps its explicit flush code.
pub const NATIVE_FTZ_EXCEPTIONS: &[Opcode] = &[
    Opcode::VRcpF32,
    Opcode::VRcpIflagF32,
    Opcode::VRsqF32,
    Opcode::VExpF32,
    Opcode::VCvtF32F64,
    Opcode::VDivFmasF32,
    Opcode::VDivScaleF32,
    // The median compares the operands before the input flush: a host
    // flush makes a denormal a zero there (med3(-0, +denormal, -denormal)
    // is -0 on the hardware, +0 with flushed comparisons).
    Opcode::VMed3F32,
];

/// A module whose FLOAT_MODE flushes f32 inputs and results everywhere
/// can drop the emulated flush and let the host flush (DenormFlushToZero)
/// when none of its opcodes disagrees with the native semantics:
/// instructions that keep a denormal f32 output (rcp, rsq, exp, the f64
/// conversion, the division intermediates) or read their denormal inputs
/// unflushed (v_div_scale_f32 without the hardware rule, the median of
/// v_med3_f32) are excluded.
/// Transcendental inputs are flushed unconditionally (the guest and the
/// host agree), log/sqrt never produce a denormal f32, comparisons and
/// conversions from f32 have no f32 result, and the mad family flushes in
/// every mode.
pub fn native_ftz_applies(program: &crate::ir_types::Program, mode: FloatMode, rules: u32) -> bool {
    if mode.f32_denormals() != 0 {
        return false;
    }
    let div_scale_unflushed = rules & float_rules::DIV_SCALE_FLUSH == 0;
    !program.instructions.iter().any(|i| {
        NATIVE_FTZ_EXCEPTIONS.contains(&i.opcode) && !matches!(i.opcode, Opcode::VDivScaleF32) || (div_scale_unflushed && matches!(i.opcode, Opcode::VDivScaleF32))
    })
}

/// Why translation failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranslateError {
    Cfg(CfgError),
    /// Instruction (or form of it) without a translation.
    Unsupported { pc: u32, opcode: String, reason: String },
    /// Malformed instruction for its opcode.
    Invalid { pc: u32, reason: String },
    /// Internal consistency check failed (a bug).
    Verify(String),
}

impl fmt::Display for TranslateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TranslateError::Cfg(e) => write!(f, "{e}"),
            TranslateError::Unsupported { pc, opcode, reason } => write!(f, "unsupported {opcode} at {pc:#x}: {reason}"),
            TranslateError::Invalid { pc, reason } => write!(f, "invalid instruction at {pc:#x}: {reason}"),
            TranslateError::Verify(e) => write!(f, "IR verification failed: {e}"),
        }
    }
}

impl std::error::Error for TranslateError {}

impl From<CfgError> for TranslateError {
    fn from(e: CfgError) -> Self {
        TranslateError::Cfg(e)
    }
}

pub type Result<T> = std::result::Result<T, TranslateError>;

/// Statistics of one compilation.
#[derive(Clone, Debug, Default)]
pub struct CompileStats {
    pub cfg_blocks: usize,
    pub structure: cfg::structurize::StructureStats,
    pub ir_blocks: usize,
    pub instructions_before_opt: usize,
    pub instructions: usize,
    pub opt: crate::opt::OptStats,
    /// Wall time per phase (ns): [`PHASES`].
    pub phase_ns: [u64; PHASES.len()],
}

/// The phases timed in [`CompileStats::phase_ns`].
pub const PHASES: [&str; 6] = ["cfg", "structurize", "translate", "verify", "ssa", "opt"];

/// A compiled shader.
#[derive(Clone, Debug)]
pub struct Compiled {
    pub function: Function,
    pub structured: Structured,
    pub stats: CompileStats,
}

/// Whole pipeline: CFG, structurizer, translation, SSA, optimization.
pub fn compile(program: &Program, options: &TranslateOptions, code: Option<&dyn CodeSource>) -> Result<Compiled> {
    // A jump table after an S_SETPC_B64 targets code the decoder stopped
    // before: decode a stand-alone program again through the furthest
    // target.
    let mut extended: Option<Program> = None;
    let mut extensions = 0;
    let mut phase_ns = [0u64; PHASES.len()];
    let mut t = std::time::Instant::now();
    let mut lap = |k: usize| {
        let now = std::time::Instant::now();
        phase_ns[k] += (now - t).as_nanos() as u64;
        t = now;
    };
    let cfg = loop {
        let current = extended.as_ref().unwrap_or(program);
        match Cfg::build(current, code) {
            Err(CfgError::TargetPastEnd { pc, offset }) => {
                let plain = current.instructions.iter().all(|i| i.pc as u64 == i.program_offset());
                let required = u32::try_from(offset + 4).ok().filter(|&r| r > current.instructions.last().map_or(0, |i| i.next_pc()));
                extensions += 1;
                match (code, plain && extensions <= 64, required) {
                    (Some(source), true, Some(required)) => {
                        let p = crate::program::decode_program_through(source, current.address, required)
                            .map_err(|e| TranslateError::Invalid { pc, reason: format!("decoding jump-table targets: {e}") })?;
                        extended = Some(p);
                    }
                    _ => return Err(TranslateError::Cfg(CfgError::TargetPastEnd { pc, offset })),
                }
            }
            other => break other?,
        }
    };
    let program = extended.as_ref().unwrap_or(program);
    lap(0);
    let structured = structurize(&cfg)?;
    lap(1);
    let mut func = translate_structured(program, &structured, options)?;
    func.rebuild_preds();
    lap(2);
    let mut stats = CompileStats { cfg_blocks: cfg.blocks.len(), structure: structured.stats, ir_blocks: func.order.len(), ..Default::default() };
    crate::ssa::verify::verify(&func, false).map_err(TranslateError::Verify)?;
    lap(3);
    build_ssa(&mut func);
    lap(4);
    crate::ssa::verify::verify(&func, true).map_err(TranslateError::Verify)?;
    lap(3);
    stats.instructions_before_opt = func.instruction_count();
    if options.optimize {
        stats.opt = crate::opt::optimize_with(&mut func, code, options.rewrites & rewrites::FLUSH_ELISION != 0);
        lap(5);
        crate::ssa::verify::verify(&func, true).map_err(TranslateError::Verify)?;
        lap(3);
    }
    stats.phase_ns = phase_ns;
    stats.instructions = func.instruction_count();
    Ok(Compiled { function: func, structured, stats })
}

/// SSA construction with the translator's register defaults: a register
/// read before any write is zero (false for the per-invocation bits, the
/// mask-bit validity and the goto variables). It is not `undef`: an
/// instruction that keeps the old value of a register in some invocations
/// (an EXEC-masked write, a permlane/DPP read of an inactive lane) must
/// observe that zero, and `undef` would let the optimizer replace the kept
/// value by the newly computed one.
pub fn build_ssa(func: &mut Function) {
    crate::ssa::build::construct_ssa_with(func, &|f, r| f.constant(r.ty(), 0));
}

/// Translate a structured program into register-level IR.
pub fn translate_structured(program: &Program, structured: &Structured, options: &TranslateOptions) -> Result<Function> {
    if options.wave_size != 32 && options.wave_size != 64 {
        return Err(TranslateError::Invalid { pc: 0, reason: format!("wave size {}", options.wave_size) });
    }
    let mut func = Function::new(options.wave_size, options.stage, program.address);
    let entry = func.new_block();
    func.entry = entry;
    func.order.push(entry);
    // The window also reaches the stage's input VGPRs (vertex and instance
    // indices, pixel inputs, thread ids) the program never names.
    let inputs = options.vgprs.iter().map(|&(r, _)| r as u32 + 1).max().unwrap_or(0);
    let vector_limit = vector_register_limit(program).max(inputs).min(256);
    let entry_rounding = rounding::entry_states(&structured.cfg, program, options.float_mode.mode);
    let mut t = Translator {
        func,
        block: entry,
        program,
        structured,
        options,
        loops: Vec::new(),
        vector_limit,
        current_pc: 0,
        f32_flush: 0,
        not_denormal: Default::default(),
        float_mode: options.float_mode,
        round: rounding::RoundState { value: options.float_mode.mode & rounding::ROUND_BITS, unknown: 0 },
        entry_rounding,
    };
    t.prologue();
    let open = t.emit_node(&structured.tree)?;
    if open {
        t.terminate(Terminator::Return);
    }
    Ok(t.func)
}

/// One past the highest VGPR any instruction names (with the stage's
/// input VGPRs, it bounds M0-relative moves).
fn vector_register_limit(program: &Program) -> u32 {
    let mut limit = 1;
    for i in &program.instructions {
        for o in i.sources.iter().chain(i.destinations.iter()) {
            if o.kind == OperandKind::VectorRegister {
                limit = limit.max(o.value + 1);
            }
        }
        if let Some(Control::Image(c)) = &i.control {
            limit = limit.max(c.vector_data + 4).max(c.vector_address + 16);
        }
    }
    limit.min(256)
}

pub(crate) struct Translator<'a> {
    pub func: Function,
    pub block: Block,
    pub program: &'a Program,
    pub structured: &'a Structured,
    pub options: &'a TranslateOptions,
    /// (merge, continue) of enclosing loops.
    loops: Vec<(Block, Block)>,
    pub vector_limit: u32,
    pub current_pc: u32,
    /// f32 denormal flushing of the current instruction: bit 0 inputs,
    /// bit 1 results ([`f32_denormal_flush`]).
    pub f32_flush: u32,
    /// f32 values known never to be denormals (a tiny-checked product and
    /// the NaN selections of it): their result flush is skipped with
    /// [`rewrites::FLUSH_ELISION`].
    pub not_denormal: crate::fxhash::HashSet<Value>,
    /// The float controls of the current instruction: the declared
    /// FLOAT_MODE with the round bits set at this point ([`rounding`]).
    pub float_mode: FloatMode,
    /// The MODE round fields at this point of the current block.
    pub round: rounding::RoundState,
    /// The round fields at the entry of each CFG block.
    entry_rounding: Vec<rounding::RoundState>,
}

impl<'a> Translator<'a> {
    pub fn b(&mut self) -> Builder<'_> {
        Builder::new(&mut self.func, self.block)
    }

    pub fn wave64(&self) -> bool {
        self.func.wave_size == 64
    }

    /// `wave_size - 1` as a u32 constant.
    pub fn wave_mask(&mut self) -> Value {
        let m = self.func.wave_size - 1;
        self.b().u32(m)
    }

    fn new_block(&mut self) -> Block {
        let b = self.func.new_block();
        self.func.order.push(b);
        b
    }

    fn terminate(&mut self, t: Terminator) {
        let bd = self.func.block_mut(self.block);
        debug_assert_eq!(bd.term, Terminator::None);
        bd.term = t;
    }

    fn prologue(&mut self) {
        let opts = self.options;
        for &(reg, input) in &opts.sgprs {
            let v = self.input_value(input);
            self.b().set_reg(Reg::Sgpr(reg), v);
        }
        for &(reg, input) in &opts.vgprs {
            let v = self.input_value(input);
            self.b().set_reg(Reg::Vgpr(reg), v);
        }
        let exec = match opts.initial_exec {
            InitialExec::All => self.b().bool(true),
            InitialExec::NotSystem(sv) => {
                let s = self.b().emit(Op::SystemValue(sv), Type::U32, vec![]);
                let zero = self.b().u32(0);
                self.b().icmp(ICmp::Eq, s, zero)
            }
        };
        self.b().set_reg(Reg::SgprBit(Reg::EXEC), exec);
        let (lo, hi) = self.b().ballot(exec);
        let hi = if self.wave64() { hi } else { self.b().u32(0) };
        self.b().set_reg(Reg::Sgpr(Reg::EXEC), lo);
        self.b().set_reg(Reg::Sgpr(Reg::EXEC + 1), hi);
    }

    fn input_value(&mut self, input: InputValue) -> Value {
        match input {
            InputValue::UserData(i) => self.b().emit(Op::UserData(i), Type::U32, vec![]),
            InputValue::System(s) => self.b().emit(Op::SystemValue(s), Type::U32, vec![]),
            InputValue::Const(c) => self.b().u32(c),
        }
    }

    /// Emit a region; returns whether control falls out of it (the current
    /// block is still open).
    fn emit_node(&mut self, node: &Node) -> Result<bool> {
        match node {
            Node::Code(b) => {
                self.translate_block(*b)?;
                Ok(true)
            }
            Node::Seq(items) => {
                for item in items {
                    if !self.emit_node(item)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Node::SetFlag(f, v) => {
                let c = self.b().bool(*v);
                self.b().set_reg(Reg::Goto(*f), c);
                Ok(true)
            }
            Node::SetState(b) => {
                let c = self.b().u32(*b);
                self.b().set_reg(Reg::Dispatch, c);
                Ok(true)
            }
            Node::If { cond, then_branch, else_branch } => {
                let c = self.eval_cond(cond);
                let header = self.block;
                let then_block = if then_branch.is_empty() { None } else { Some(self.new_block()) };
                let else_block = if else_branch.is_empty() { None } else { Some(self.new_block()) };
                let merge = self.func.new_block();
                let then_target = then_block.unwrap_or(merge);
                let else_target = else_block.unwrap_or(merge);
                self.func.block_mut(header).term = Terminator::CondBranch { cond: c, then_block: then_target, else_block: else_target };
                self.func.block_mut(header).merge = Merge::Selection { merge };
                let mut reaches_merge = then_block.is_none() || else_block.is_none();
                for (arm, block) in [(then_branch, then_block), (else_branch, else_block)] {
                    if let Some(b) = block {
                        self.block = b;
                        if self.emit_node(arm)? {
                            self.terminate(Terminator::Branch(merge));
                            reaches_merge = true;
                        }
                    }
                }
                self.func.order.push(merge);
                self.block = merge;
                if !reaches_merge {
                    self.terminate(Terminator::Unreachable);
                    return Ok(false);
                }
                Ok(true)
            }
            Node::Loop(body) => {
                let header = self.new_block();
                self.terminate(Terminator::Branch(header));
                let body_block = self.new_block();
                let cont = self.func.new_block();
                let merge = self.func.new_block();
                self.func.block_mut(header).merge = Merge::Loop { merge, continue_block: cont };
                self.func.block_mut(header).term = Terminator::Branch(body_block);
                self.loops.push((merge, cont));
                self.block = body_block;
                if self.emit_node(body)? {
                    self.terminate(Terminator::Branch(merge));
                }
                self.loops.pop();
                self.func.order.push(cont);
                self.func.block_mut(cont).term = Terminator::Branch(header);
                self.func.order.push(merge);
                self.block = merge;
                // The merge is reachable iff some block branches to it.
                let reachable = self.func.order.iter().any(|&b| self.func.block(b).term.successors().contains(&merge));
                if !reachable {
                    self.terminate(Terminator::Unreachable);
                    return Ok(false);
                }
                Ok(true)
            }
            Node::Break => {
                let (merge, _) = *self.loops.last().expect("break outside loop");
                self.terminate(Terminator::Branch(merge));
                Ok(false)
            }
            Node::Continue => {
                let (_, cont) = *self.loops.last().expect("continue outside loop");
                self.terminate(Terminator::Branch(cont));
                Ok(false)
            }
            Node::Return => {
                self.terminate(Terminator::Return);
                Ok(false)
            }
        }
    }

    fn eval_cond(&mut self, cond: &Cond) -> Value {
        match cond {
            Cond::Branch { cond, .. } => self.branch_condition(*cond),
            Cond::Flag(f) => self.b().get_reg(Reg::Goto(*f)),
            Cond::State(b) => {
                let s = self.b().get_reg(Reg::Dispatch);
                let c = self.b().u32(*b);
                self.b().icmp(ICmp::Eq, s, c)
            }
            Cond::Not(c) => {
                let v = self.eval_cond(c);
                self.b().lnot(v)
            }
            Cond::Or(cs) => {
                let mut acc = self.eval_cond(&cs[0]);
                for c in &cs[1..] {
                    let v = self.eval_cond(c);
                    acc = self.b().lor(acc, v);
                }
                acc
            }
        }
    }

    /// Wave-uniform value of a branch condition.
    fn branch_condition(&mut self, cond: BranchCond) -> Value {
        match cond {
            BranchCond::Scc1 => self.b().get_reg(Reg::Scc),
            BranchCond::Scc0 => {
                let s = self.b().get_reg(Reg::Scc);
                self.b().lnot(s)
            }
            BranchCond::Vccz | BranchCond::Vccnz | BranchCond::Execz | BranchCond::Execnz => {
                let base = if matches!(cond, BranchCond::Vccz | BranchCond::Vccnz) { Reg::VCC } else { Reg::EXEC };
                let zero = self.mask_is_zero(base);
                if matches!(cond, BranchCond::Vccz | BranchCond::Execz) {
                    zero
                } else {
                    self.b().lnot(zero)
                }
            }
            BranchCond::Instruction => self.b().get_reg(Reg::InstructionBranch),
            BranchCond::SgprEquals { sgpr, value } => {
                let s = self.b().get_reg(Reg::Sgpr(sgpr));
                let c = self.b().u32(value);
                self.b().icmp(ICmp::Eq, s, c)
            }
        }
    }

    /// Whether the lane mask starting at SGPR `base` is zero (wave-wide).
    pub fn mask_is_zero(&mut self, base: u8) -> Value {
        let lo = self.b().get_reg(Reg::Sgpr(base));
        let word = if self.wave64() {
            let hi = self.b().get_reg(Reg::Sgpr(base + 1));
            self.b().or(lo, hi)
        } else {
            lo
        };
        let zero = self.b().u32(0);
        self.b().icmp(ICmp::Eq, word, zero)
    }

    fn translate_block(&mut self, b: usize) -> Result<()> {
        let bb = &self.structured.cfg.blocks[b];
        if let Some(&state) = self.entry_rounding.get(b) {
            self.round = state;
        }
        for i in bb.first..bb.end {
            let ins = &self.program.instructions[i];
            self.current_pc = ins.pc;
            self.begin_rounding(ins)?;
            self.f32_flush = if self.options.native_ftz { 0 } else { f32_denormal_flush_with(ins.opcode, self.float_mode, self.options.float_rules) };
            self.translate(ins)?;
        }
        Ok(())
    }

    /// The float controls `ins` runs with: the declared FLOAT_MODE with the
    /// round bits tracked to this point (unknown bits: the declared ones).
    /// With strict MODE writes, refuses what is not modelled.
    fn begin_rounding(&mut self, ins: &Instruction) -> Result<()> {
        let declared = self.options.float_mode;
        let initial = declared.mode & rounding::ROUND_BITS;
        let state = self.round;
        if let Some((fields, kind)) = rounding::rounding_use(ins.opcode) {
            if self.options.strict_mode_writes {
                if state.unknown & fields != 0 {
                    return self.unsupported(ins, "reads MODE rounding bits that differ between the paths reaching it or were written from an SGPR");
                }
                if (state.value ^ initial) & fields != 0 {
                    match kind {
                        rounding::RoundUse::Directed16 if declared.mode & 0xC0 != 0xC0 => {
                            return self.unsupported(ins, "runs with an f16 rounding mode set at run time and f16 denormals not kept");
                        }
                        rounding::RoundUse::Unmodelled if rounding::sets_directed(state.value, initial, fields) => {
                            return self.unsupported(ins, "runs with a MODE rounding set at run time, and the instruction does not model directed rounding");
                        }
                        _ => {}
                    }
                }
            }
        }
        let bits = (state.value & !state.unknown) | (initial & state.unknown);
        self.float_mode = FloatMode { mode: (declared.mode & !rounding::ROUND_BITS) | (bits & rounding::ROUND_BITS), ..declared };
        Ok(())
    }

    /// Apply the round-field write of `ins` (s_round_mode,
    /// s_setreg_imm32_b32 / s_setreg_b32 of MODE[3:0]) to the tracked
    /// state. Strict: other MODE fields, and other registers written by
    /// s_setreg_imm32_b32, are refused.
    pub(super) fn write_rounding(&mut self, ins: &Instruction) -> Result<()> {
        match rounding::decode_write(ins) {
            Some(w) => {
                self.round = rounding::apply(self.round, w);
                Ok(())
            }
            None if self.options.strict_mode_writes => self.unsupported(ins, "writes a hardware register field other than the MODE rounding fields"),
            None => Ok(()),
        }
    }

    /// The MODE round bits `mask` as tracked here (unknown ones refused when
    /// strict, otherwise the declared bits).
    pub(super) fn read_rounding(&mut self, ins: &Instruction, mask: u8) -> Result<u8> {
        if self.round.unknown & mask != 0 && self.options.strict_mode_writes {
            return self.unsupported(ins, "reads MODE rounding bits that differ between the paths reaching it or were written from an SGPR");
        }
        let initial = self.options.float_mode.mode & rounding::ROUND_BITS;
        Ok(((self.round.value & !self.round.unknown) | (initial & self.round.unknown)) & mask)
    }

    pub fn unsupported<T>(&self, ins: &Instruction, reason: &str) -> Result<T> {
        Err(TranslateError::Unsupported { pc: ins.pc, opcode: ins.opcode.mnemonic(), reason: reason.into() })
    }

    pub fn invalid<T>(&self, ins: &Instruction, reason: &str) -> Result<T> {
        Err(TranslateError::Invalid { pc: ins.pc, reason: format!("{}: {reason}", ins.opcode.mnemonic()) })
    }

    /// Translate one instruction.
    fn translate(&mut self, ins: &Instruction) -> Result<()> {
        use Opcode::*;
        if ins.opcode.is_raw() {
            return self.unsupported(ins, "opcode has no name in the decoder");
        }
        // DPP16 controls RDNA defines: quad_perm, row_shl/shr/ror 1..15,
        // row_mirror, row_half_mirror, row_share, row_xmask (the GCN wave
        // shifts and row broadcasts are reserved).
        if let Some(Control::Dpp(d)) = &ins.control {
            let c = d.control;
            let defined = c <= 0xFF
                || matches!(c, 0x101..=0x10F | 0x111..=0x11F | 0x121..=0x12F | 0x140 | 0x141 | 0x150..=0x15F | 0x160..=0x16F);
            if !defined {
                return self.unsupported(ins, &format!("DPP control {c:#x}"));
            }
        }
        match ins.encoding {
            Encoding::Sopp => self.translate_sopp(ins),
            Encoding::Sop1 | Encoding::Sop2 | Encoding::Sopc | Encoding::Sopk => self.translate_salu(ins),
            Encoding::Smem if !matches!(ins.control, Some(Control::ScalarMemory(_))) => self.translate_smem_misc(ins),
            Encoding::Smrd | Encoding::Smem => self.translate_smem(ins),
            Encoding::Vop3 if matches!(ins.opcode, VInterpP1F32 | VInterpP2F32 | VInterpMovF32) => self.translate_interp(ins),
            Encoding::Vop1 | Encoding::Vop2 | Encoding::Vopc | Encoding::Vop3 | Encoding::Vop3p => {
                if matches!(ins.opcode, VNop) {
                    return Ok(());
                }
                self.translate_valu(ins)
            }
            Encoding::Vintrp => self.translate_interp(ins),
            Encoding::Ds if ins.opcode == DsNop => Ok(()),
            Encoding::Ds => self.translate_ds(ins),
            // Vector cache invalidation: later reads see device memory.
            Encoding::Mubuf if matches!(ins.opcode, BufferGl0Inv | BufferGl1Inv) => {
                self.b().emit(Op::DataCacheInvalidate, Type::VOID, vec![]);
                Ok(())
            }
            Encoding::Mubuf | Encoding::Mtbuf => self.translate_buffer(ins),
            Encoding::Flat => self.translate_flat(ins),
            Encoding::Mimg => self.translate_image(ins),
            Encoding::Exp => self.translate_export(ins),
        }
    }

    fn translate_sopp(&mut self, ins: &Instruction) -> Result<()> {
        use Opcode::*;
        match ins.opcode {
            SNop | SWaitcnt | SWaitcntDepctr | SSetprio | SInstPrefetch | SClause | STtraceData | STrap | SCodeEnd
            | SSleep | SWakeup | SIcacheInv | SIncperflevel | SDecperflevel | STtracedataImm | SEndpgmSaved | SWaitIdle => Ok(()),
            SSendmsghalt => self.unsupported(ins, "s_sendmsghalt halts the wave until a debugger resumes it"),
            SRoundMode | SDenormMode => {
                if self.translate_extra_salu(ins)? {
                    return Ok(());
                }
                self.unsupported(ins, "SOPP")
            }
            // Control transfer is in the CFG.
            SEndpgm | SEndpgmOrderedPsDone | SSetkill | SBranch | SCbranchScc0 | SCbranchScc1 | SCbranchVccz | SCbranchVccnz | SCbranchExecz
            | SCbranchExecnz | SCbranchCdbgsys | SCbranchCdbguser | SCbranchCdbgsysOrUser | SCbranchCdbgsysAndUser => Ok(()),
            SBarrier => {
                self.b().emit(Op::Barrier, Type::VOID, vec![]);
                Ok(())
            }
            // s_sethalt 0 resumes a wave that is running anyway.
            SSethalt if ins.words[0] & 1 == 0 => Ok(()),
            SSethalt => self.unsupported(ins, "s_sethalt halting the wave"),
            SSendmsg => {
                let m0 = self.b().get_reg(Reg::Sgpr(Reg::M0));
                self.b().emit(Op::SendMessage { message: (ins.words[0] & 0xFFFF) as u16 }, Type::VOID, vec![m0]);
                Ok(())
            }
            _ => self.unsupported(ins, "SOPP"),
        }
    }
}

/// Distinct opcodes of `program` that [`compile`] cannot translate (for
/// coverage reports): returns `(mnemonic, reason)` of the first failure.
pub fn first_failure(program: &Program, options: &TranslateOptions) -> Option<String> {
    match compile(program, options, None) {
        Ok(_) => None,
        Err(e) => Some(e.to_string()),
    }
}

/// Opcode names used by a program (helper for coverage tables).
pub fn opcode_set(program: &Program) -> BTreeSet<String> {
    program.instructions.iter().map(|i| format!("{:?}/{}", i.encoding, i.opcode.name())).collect()
}
