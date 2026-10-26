// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The SPIR-V shader compiler: guest program (decoded and resource-planned
//! by `ps5shader`) → structured SSA IR → SPIR-V module plus reflection, as
//! a [`ShaderCompiler`] for the pipeline layer.
//!
//! * [`emit`] — module setup, types, control flow, phis, ALU, subgroups.
//! * [`memory`] — push / shader data, buffers, device addresses, LDS,
//!   scratch, atomics.
//! * [`image`] — image and sampler operations.
//! * [`io`] — system values, attribute interpolation, exports.
//!
//! Stage model:
//! * compute: one invocation per lane, workgroup = dispatch group;
//! * vertex: the guest's merged (ES+GS, NGG pass-through) or plain vertex
//!   program runs once per host vertex; the wave-info SGPR reports the host
//!   subgroup's lanes, the primitive export is left to the host assembler;
//! * pixel: barycentrics and per-vertex attributes come from
//!   `VK_KHR_fragment_shader_barycentric`, so the guest's own interpolation
//!   arithmetic runs unchanged.
//!
//! `RPS5E_SHADER_COMPILER=0` keeps the compiler out of the pipeline
//! provider; `RPS5E_SPIRV_DUMP=<dir>` writes every module as
//! `<dir>/<stage>_<address>_<key>.spv`; compiled modules are kept on disk
//! across runs ([`disk`], `RPS5E_SHADER_CACHE`).

pub mod bvh;
pub mod convert;
pub mod disk;
pub use disk::translator_stamp;
pub mod emit;
pub mod image;
pub mod io;
pub mod lane_reduce;
pub mod memory;
pub mod mesh;
pub mod stats;
pub mod wave_split;

use crate::device::GpuDevice;
use crate::memory::GuestGpuMemory;
use crate::pipeline::compiler::{layout_bindings, CompileError, CompiledShader, ComponentKind, ReflectedVariable, ShaderCompileInput, ShaderCompiler, ShaderReflection};
use ps5shader::resources::{ResourceUse, ShaderCompileRequest, ShaderStage};
use ps5shader::ssa::{Function, Stage, SystemValue, Value};
use ps5shader::translate::{self, InitialExec, InputValue, TranslateOptions};
use ps5shader::PixelOutputKind;
use rspirv::spirv;
use std::collections::HashMap;

/// Device features the emitter may use.
#[derive(Clone, Copy, Debug, Default)]
pub struct Features {
    pub barycentric: bool,
    pub demote: bool,
    /// Host subgroup size (32 or 64).
    pub subgroup_size: u32,
    /// A compute pipeline can require 64-wide subgroups when the default
    /// is narrower (VK_EXT_subgroup_size_control): wave64 compute programs
    /// then run one guest lane per invocation on 64-wide subgroups.
    pub subgroup64_compute: bool,
    /// maxComputeWorkgroupSubgroups with a required subgroup size.
    pub max_compute_workgroup_subgroups: u32,
    /// maxComputeSharedMemorySize (0: unknown).
    pub max_shared_bytes: u32,
    /// Offset image operands on non-gather samples (maintenance8).
    pub sample_offset: bool,
    /// 64-bit image atomics (VK_EXT_shader_image_atomic_int64).
    pub image_int64_atomics: bool,
    /// 64-bit workgroup atomics through an explicitly laid out u64 view of
    /// the LDS (shaderSharedInt64Atomics + VK_KHR_workgroup_memory_explicit_layout).
    pub shared_int64_atomics: bool,
    /// Quad derivatives in compute (VK_KHR/NV_compute_shader_derivatives).
    pub compute_derivatives: bool,
    /// Compute derivatives over four consecutive invocations
    /// (DerivativeGroupLinearNV): the GFX10 compute quad.
    pub compute_derivatives_linear: bool,
    /// SignedZeroInfNanPreserve execution modes for 16/32/64-bit floats.
    pub preserve_inf_nan: [bool; 3],
    /// DenormPreserve execution modes for 16/32/64-bit floats: the
    /// translator flushes the denormals the guest's FLOAT_MODE flushes, the
    /// host keeps the rest.
    pub denorm_preserve: [bool; 3],
    /// DenormFlushToZero(32) is available: modules whose FLOAT_MODE flushes
    /// f32 everywhere run without the emulated flush code.
    pub denorm_flush_to_zero32: bool,
    /// Round-to-nearest-even f32 -> f16 conversions as native OpFConvert
    /// under the 16-bit RoundingModeRTE / DenormPreserve /
    /// SignedZeroInfNanPreserve modes (`RPS5E_NATIVE_F16_RTE=0`: the exact
    /// integer conversion everywhere).
    pub f16_rte: bool,
    /// `s_memrealtime` reads the device clock scaled by this ratio
    /// (calibrated against the host reference clock, [`crate::clock`]);
    /// `None`: a per-invocation counter.
    pub device_clock: Option<crate::clock::DeviceClock>,
    /// `s_memtime` reads the subgroup clock (shaderSubgroupClock).
    pub subgroup_clock: bool,
    /// The bindless set exists (descriptor indexing: runtime arrays,
    /// non-uniform sampled-image indexing, partially bound update-after-bind
    /// sampled images): images selected through the host table map.
    pub bindless_tables: bool,
    /// The device lacks shaderFloat64 (Intel Arc / Xe-LPG): the
    /// translator's own exact conversions use integer code, and a module
    /// whose guest program computes in f64 is refused.
    pub no_float64: bool,
    /// The host driver is AMD's proprietary one: a wave's LDS ordering
    /// barrier is a subgroup-scope `OpMemoryBarrier`, not a subgroup-scope
    /// `OpControlBarrier` (that driver compiles it to `s_barrier`, which
    /// waves in a branch paired with other waves' barriers hang on) —
    /// only in a work-group of several waves; one wave keeps the control
    /// barrier ([`crate::device::DeviceCaps::amd_proprietary`]).
    pub amd_proprietary: bool,
    /// GPU-cost switches of the emitted code.
    pub emit: EmitOptions,
}

/// GPU-cost switches of the emitted code and of the translation it asks
/// for, read once from the environment by [`Features::of`] (part of the
/// disk-cache key). Each exact one that measured faster on the device is on
/// by default and `RPS5E_<name>=0` keeps the code emitted before it; the
/// inexact fast transcendentals, the fault log and the two bound-buffer
/// switches that measured slower on NVIDIA (the branches cost the loads
/// their overlap, and the driver's robustness is cheaper than the module's
/// own clamps) are opt-in (`=1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EmitOptions {
    /// `RPS5E_LOG_SERIES_BRANCH`: v_log_f32's series near 1.0 is computed
    /// only in invocations whose input is near 1.0 (a branch), not always
    /// and selected; the result is the same.
    pub log_series_branch: bool,
    /// `RPS5E_SHADER_READONLY`: bindings the module never writes (page
    /// table, shader data, resource table, buffers without stores) are
    /// NonWritable (and Restrict when the module writes no buffer memory).
    pub readonly: bool,
    /// `RPS5E_DEVICE_ACCESS_ONCE`: a device-address access of several
    /// dwords within one page translates its address once and loads the
    /// dwords as one vector (scalar loads of one instruction as a group).
    pub device_once: bool,
    /// `RPS5E_SHADER_FAULTS=1` (opt-in): device-address reads of pages
    /// without a page-table entry are counted in the fault buffer (the
    /// `unmapped-reads` of the `[gpu]` line).
    pub fault_log: bool,
    /// `RPS5E_BUFFER_ACCESS_ONCE=1` (opt-in): a bound-buffer load of
    /// several dwords (raw loads, the dwords of one `s_buffer_load`) checks
    /// the host range once (a branch) when every dword is in range; a
    /// store of several dwords is one branch per dword. Every guest range
    /// rule stays per dword.
    pub bound_once: bool,
    /// `RPS5E_SUBGROUP_OPS`: v_readlane with a constant lane (any uniform
    /// lane with subgroupBroadcastDynamicId) is a broadcast; a shuffle whose
    /// source lane is a fixed function of the lane, in every lane whose
    /// result is used, is ShuffleXor / ShuffleUp / ShuffleDown / a broadcast
    /// / nothing; the DPP source lane's EXEC bit is a bit test of the EXEC
    /// words, not a shuffle.
    pub subgroup_ops: bool,
    /// subgroupBroadcastDynamicId: Broadcast takes a uniform, non-constant
    /// lane (device feature).
    pub broadcast_dynamic: bool,
    /// ShuffleUp / ShuffleDown are supported (device property).
    pub shuffle_relative: bool,
    /// `RPS5E_PIPELINE_ROBUSTNESS=1` (opt-in) where the device has
    /// VK_EXT_pipeline_robustness: bound-buffer loads read only in-range
    /// words (an index past the range reads word 0, selected away), so
    /// compute pipelines of modules that keep every buffer index in range
    /// run without the driver's storage-buffer robustness (images keep
    /// theirs).
    pub guarded_loads: bool,
    /// `RPS5E_LEAN_FLUSH`: the f32 denormal flush is one compare and one
    /// select (a zero is its own sign).
    pub lean_flush: bool,
    /// `RPS5E_EMIT_CSE`: the emitter's common subexpressions live along
    /// the dominator tree, not only within one block.
    pub global_cse: bool,
    /// `RPS5E_FAST_TRANSCENDENTAL=1` (opt-in, inexact): v_exp/log/rcp/
    /// rsq/sqrt_f32 are the host's operations with one NaN select.
    pub fast_transcendental: bool,
    /// `RPS5E_NATIVE_FTZ=0`: modules whose FLOAT_MODE flushes f32 inputs
    /// and results everywhere (and whose opcodes agree, see
    /// `ps5shader::translate::native_ftz_applies`) keep the emulated flush
    /// instead of the host's DenormFlushToZero(32).
    pub native_ftz: bool,
    /// `RPS5E_FLUSH_ELISION`: the f32 denormal flush of a value that is
    /// never a denormal (an earlier flush's result through EXEC selects
    /// and phis) is not emitted (`ps5shader::opt::denorm`).
    pub flush_elision: bool,
    /// `RPS5E_FULL_WAVE_EXEC=1` (opt-in): a compute program whose
    /// work-group is whole waves (and whose dispatch has no thread limits)
    /// starts with every lane active, so its EXEC-masked writes fold in the
    /// IR (`full_waves`). Measured neutral on NVIDIA (the driver already
    /// knows the lanes are in range; same binary, < 0.1 % fewer words).
    pub full_wave_exec: bool,
    /// `RPS5E_FLOAT_CLASS_COMPARE`: the f32 denormal-flush and NaN tests
    /// are float compares (where the device preserves NaNs), not integer
    /// tests of the magnitude bits.
    pub float_class_compare: bool,
    /// `RPS5E_EMIT_FOLD`: u32/bool instructions the emitter makes on
    /// constants (range checks of constant offsets, `x + 0`, selects on a
    /// constant) are folded instead of emitted, and the dwords of one
    /// bound-buffer load index from the first dword's word index.
    pub const_fold: bool,
    /// `RPS5E_LAZY_SELECT`: the operations a lazy select
    /// (`ps5shader::ssa::Function::lazy_selects`) needs only for its true
    /// arm run under a branch on its condition.
    pub lazy_select: bool,
    /// `RPS5E_HALF_FMA_FAST`: v_fma_mix_lo/hi_f16 round the f32 fma to f16
    /// except at f16 midpoints (and NaNs, kept f32 denormal sources),
    /// where the f64 round-to-odd sum is the lazy arm
    /// (`ps5shader::translate::rewrites::HALF_FMA_FAST`).
    pub half_fma_fast: bool,
    /// `RPS5E_LANE_REDUCTIONS`: a graphics v_readlane of a wave reduction
    /// (DPP row scans, a cross-row permute) is a subgroup reduction over
    /// the lanes the host subgroup has (`lane_reduce`), not a read of a
    /// lane it may lack (needs subgroup arithmetic).
    pub lane_reductions: bool,
    /// `RPS5E_SMEM_ONCE`: a paired wave64 program (two guest lanes per
    /// invocation) loads each scalar memory read (`s_load`,
    /// `s_buffer_load`: SGPR operands, no EXEC) once and both lane halves
    /// share the words, instead of loading them per half.
    pub smem_once: bool,
}

impl EmitOptions {
    /// The defaults: the switches that measured faster on, the opt-in ones
    /// off.
    pub const DEFAULT: EmitOptions = EmitOptions {
        log_series_branch: true,
        readonly: true,
        device_once: true,
        fault_log: false,
        bound_once: false,
        subgroup_ops: true,
        broadcast_dynamic: false,
        shuffle_relative: false,
        guarded_loads: false,
        lean_flush: true,
        global_cse: true,
        fast_transcendental: false,
        native_ftz: true,
        flush_elision: false,
        full_wave_exec: false,
        float_class_compare: false,
        const_fold: false,
        lazy_select: false,
        half_fma_fast: false,
        lane_reductions: true,
        smem_once: true,
    };
    /// Every exact switch on, on a device with every feature they use
    /// (tests).
    pub const ALL: EmitOptions = EmitOptions { broadcast_dynamic: true, shuffle_relative: true, guarded_loads: true, bound_once: true, full_wave_exec: true, flush_elision: true, float_class_compare: true, const_fold: true, lazy_select: true, half_fma_fast: true, ..EmitOptions::DEFAULT };
    /// The code emitted before these switches existed.
    pub const BEFORE: EmitOptions = EmitOptions {
        log_series_branch: false,
        readonly: false,
        device_once: false,
        fault_log: true,
        bound_once: false,
        subgroup_ops: false,
        broadcast_dynamic: false,
        shuffle_relative: false,
        guarded_loads: false,
        lean_flush: false,
        global_cse: false,
        fast_transcendental: false,
        native_ftz: false,
        flush_elision: false,
        full_wave_exec: false,
        float_class_compare: false,
        const_fold: false,
        lazy_select: false,
        half_fma_fast: false,
        lane_reductions: false,
        smem_once: false,
    };

    /// The switches from the environment (read once).
    pub fn from_env() -> EmitOptions {
        static OPTIONS: std::sync::OnceLock<EmitOptions> = std::sync::OnceLock::new();
        *OPTIONS.get_or_init(|| {
            let on = crate::backend::ab::switch_on;
            let opt_in = |name: &str| std::env::var(name).is_ok_and(|v| v == "1");
            let o = EmitOptions {
                log_series_branch: on("RPS5E_LOG_SERIES_BRANCH"),
                readonly: on("RPS5E_SHADER_READONLY"),
                device_once: on("RPS5E_DEVICE_ACCESS_ONCE"),
                fault_log: opt_in("RPS5E_SHADER_FAULTS"),
                bound_once: opt_in("RPS5E_BUFFER_ACCESS_ONCE"),
                subgroup_ops: on("RPS5E_SUBGROUP_OPS"),
                broadcast_dynamic: false,
                shuffle_relative: false,
                guarded_loads: opt_in("RPS5E_PIPELINE_ROBUSTNESS"),
                lean_flush: on("RPS5E_LEAN_FLUSH"),
                global_cse: on("RPS5E_EMIT_CSE"),
                fast_transcendental: opt_in("RPS5E_FAST_TRANSCENDENTAL"),
                native_ftz: on("RPS5E_NATIVE_FTZ"),
                // Opt-in until proven exact: SILENT HILL frames showed
                // garbage at the bottom/right screen edges with these three
                // on (GPU_PLAN §62 addendum).
                flush_elision: opt_in("RPS5E_FLUSH_ELISION"),
                full_wave_exec: opt_in("RPS5E_FULL_WAVE_EXEC"),
                float_class_compare: opt_in("RPS5E_FLOAT_CLASS_COMPARE"),
                const_fold: opt_in("RPS5E_EMIT_FOLD"),
                // Opt-in until the reference captures are checked with them.
                lazy_select: opt_in("RPS5E_LAZY_SELECT"),
                half_fma_fast: opt_in("RPS5E_HALF_FMA_FAST"),
                lane_reductions: on("RPS5E_LANE_REDUCTIONS"),
                smem_once: on("RPS5E_SMEM_ONCE"),
            };
            if o.fast_transcendental {
                log::warn!("[shader] RPS5E_FAST_TRANSCENDENTAL=1: v_exp/log/rcp/rsq/sqrt_f32 use the host's precision and special values");
            }
            o
        })
    }
}

impl EmitOptions {
    /// [`Self::from_env`] with what the device supports.
    pub fn for_device(dev: &GpuDevice) -> EmitOptions {
        let o = EmitOptions::from_env();
        EmitOptions {
            broadcast_dynamic: dev.caps.subgroup_broadcast_dynamic_id,
            shuffle_relative: dev.caps.subgroup_shuffle_relative,
            lane_reductions: o.lane_reductions && dev.caps.subgroup_arithmetic,
            guarded_loads: o.guarded_loads && dev.caps.pipeline_robustness,
            ..o
        }
    }
}

impl Default for EmitOptions {
    fn default() -> Self {
        EmitOptions::DEFAULT
    }
}

impl Features {
    pub fn of(dev: &GpuDevice) -> Features {
        Features {
            barycentric: dev.caps.fragment_shader_barycentric,
            demote: dev.caps.shader_demote_to_helper_invocation,
            subgroup_size: dev.caps.subgroup_size,
            subgroup64_compute: dev.caps.subgroup_size < 64 && dev.caps.min_compute_subgroup_size <= 64 && dev.caps.max_compute_subgroup_size >= 64,
            max_compute_workgroup_subgroups: dev.caps.max_compute_workgroup_subgroups,
            max_shared_bytes: dev.caps.max_compute_shared_memory,
            sample_offset: dev.caps.maintenance8,
            image_int64_atomics: dev.caps.shader_image_int64_atomics,
            shared_int64_atomics: dev.caps.shader_shared_int64_atomics && dev.caps.workgroup_memory_explicit_layout,
            compute_derivatives: dev.caps.compute_derivative_group_quads,
            compute_derivatives_linear: dev.caps.compute_derivative_group_linear,
            preserve_inf_nan: dev.caps.signed_zero_inf_nan_preserve,
            denorm_preserve: dev.caps.denorm_preserve,
            denorm_flush_to_zero32: dev.caps.denorm_flush_to_zero32,
            f16_rte: dev.caps.float16_rte && native_f16_rte_enabled(),
            device_clock: dev.clock.get().copied(),
            subgroup_clock: dev.caps.shader_subgroup_clock,
            bindless_tables: dev.caps.bindless_slots != 0,
            no_float64: !dev.caps.shader_float64,
            amd_proprietary: dev.caps.amd_proprietary,
            emit: EmitOptions::for_device(dev),
        }
    }
}

/// Everything the emitter reads.
pub struct EmitInput<'a, 'p> {
    pub func: &'a Function,
    pub stage: ShaderStage,
    pub request: &'a ShaderCompileRequest<'p>,
    pub uses: &'a HashMap<Value, Vec<ResourceUse>>,
    pub features: Features,
    /// Guest lanes per host invocation (see [`emit::Emitter::lanes`]).
    pub lanes: usize,
    /// Geometry subgroup run as a mesh workgroup (vertex stage only).
    pub mesh: Option<crate::pipeline::compiler::MeshConfig>,
    /// A wave64 compute program split over two 32-wide host subgroups
    /// ([`wave_split`]), with `lanes` 1.
    pub split: Option<wave_split::SplitPlan>,
    /// The module's f32 denormals are the host's (DenormFlushToZero(32)):
    /// the translation carries no flush code (see
    /// `ps5shader::translate::native_ftz_applies`).
    pub native_ftz: bool,
}

/// Native f32 -> f16 conversions where the device guarantees them, unless
/// `RPS5E_NATIVE_F16_RTE=0`.
fn native_f16_rte_enabled() -> bool {
    !std::env::var("RPS5E_NATIVE_F16_RTE").is_ok_and(|v| v == "0")
}

/// Is the compiler enabled (`RPS5E_SHADER_COMPILER` is not `0`)?
pub fn enabled() -> bool {
    !std::env::var("RPS5E_SHADER_COMPILER").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off"))
}

/// Pixel input VGPRs in SPI_PS_INPUT_ADDR order: (enable bit, VGPR count).
const PIXEL_INPUTS: [(u32, u32); 16] = [
    (0, 2),
    (1, 2),
    (2, 2),
    (3, 3),
    (4, 2),
    (5, 2),
    (6, 2),
    (7, 1),
    (8, 1),
    (9, 1),
    (10, 1),
    (11, 1),
    (12, 1),
    (13, 1),
    (14, 1),
    (15, 1),
];

/// Value of VGPR `k` of pixel input `bit`.
fn pixel_input_value(bit: u32, k: u32) -> InputValue {
    use emit::custom;
    let sys = InputValue::System;
    match bit {
        0 | 1 => sys(SystemValue::Barycentric(k as u8)),
        2 => sys(SystemValue::Barycentric(4 + k as u8)),
        4 | 5 => sys(SystemValue::Barycentric(2 + k as u8)),
        6 => sys(SystemValue::Barycentric(6 + k as u8)),
        8 => sys(SystemValue::FragCoord(0)),
        9 => sys(SystemValue::FragCoord(1)),
        10 => sys(SystemValue::FragCoord(2)),
        11 => sys(SystemValue::Custom(custom::POSITION_W)),
        12 => sys(SystemValue::FrontFacing),
        13 => sys(SystemValue::Ancillary),
        14 => sys(SystemValue::SampleCoverage),
        15 => sys(SystemValue::Custom(custom::POSITION_FIXED)),
        _ => InputValue::Const(0),
    }
}

/// Translation options of one permutation.
pub fn translate_options(stage: ShaderStage, r: &ShaderCompileRequest<'_>, mesh: bool) -> TranslateOptions {
    let base = r.user_data_base();
    let count = r.user_data_count();
    let wave = if r.wave_size == 64 { 64 } else { 32 };
    let mut sgprs: Vec<(u8, InputValue)> = (base..base + count).filter(|&s| s < 106).map(|s| (s as u8, InputValue::UserData(s))).collect();
    let mut vgprs: Vec<(u16, InputValue)> = Vec::new();
    let mut initial_exec = InitialExec::All;
    let ssa_stage = match stage {
        ShaderStage::Compute => {
            if let Some(sys) = r.compute_system_registers {
                for (axis, reg) in [sys.work_group_x, sys.work_group_y, sys.work_group_z].into_iter().enumerate() {
                    if let Some(reg) = reg {
                        sgprs.push((reg as u8, InputValue::System(SystemValue::WorkgroupId(axis as u8))));
                    }
                }
                if let Some(reg) = sys.thread_group_size {
                    sgprs.push((reg as u8, InputValue::System(SystemValue::Custom(emit::custom::THREAD_GROUP_SIZE))));
                }
            }
            vgprs.extend((0..3u16).map(|a| (a, InputValue::System(SystemValue::LocalInvocationId(a as u8)))));
            // The lanes past the group's threads (or the dispatch's) start
            // inactive. HelperInvocation stands for them (`io.rs`): the
            // optimizer treats it as varying, unlike the custom values.
            initial_exec = InitialExec::NotSystem(SystemValue::HelperInvocation);
            Stage::Compute
        }
        ShaderStage::Vertex if mesh => {
            // Geometry subgroup: s0/s1 (the user-data address) arrive as
            // user data from s0; the entry point computes the rest.
            let input = |k: usize| InputValue::System(SystemValue::Custom(emit::custom::MESH_INPUT + k as u32));
            sgprs.retain(|(s, _)| *s < 2 || *s >= 8);
            sgprs.push((2, input(mesh::IN_S2)));
            sgprs.push((3, input(mesh::IN_S3)));
            for s in 4..8u8 {
                sgprs.push((s, InputValue::Const(0)));
            }
            for k in 0..9u16 {
                let v = match k {
                    0 => input(mesh::IN_V0),
                    1 => input(mesh::IN_V1),
                    2 => input(mesh::IN_V2),
                    5 => input(mesh::IN_V5),
                    8 => input(mesh::IN_V8),
                    _ => InputValue::Const(0),
                };
                vgprs.push((k, v));
            }
            Stage::Vertex
        }
        ShaderStage::Vertex => {
            if base >= 8 {
                // Merged ES+GS: system SGPRs s0..s7, ES inputs from v5.
                for s in 0..8u8 {
                    let v = match s {
                        2 if vertex_wave_info_enabled() => InputValue::System(SystemValue::Custom(emit::custom::SUBGROUP_INFO)),
                        3 => InputValue::System(SystemValue::Custom(emit::custom::MERGED_WAVE_INFO)),
                        _ => InputValue::Const(0),
                    };
                    sgprs.push((s, v));
                }
                for k in 0..9u16 {
                    let v = match k {
                        5 => InputValue::System(SystemValue::VertexIndex),
                        8 => InputValue::System(SystemValue::InstanceIndex),
                        _ => InputValue::Const(0),
                    };
                    vgprs.push((k, v));
                }
            } else {
                vgprs.push((0, InputValue::System(SystemValue::VertexIndex)));
                vgprs.push((1, InputValue::Const(0)));
                vgprs.push((2, InputValue::Const(0)));
                vgprs.push((3, InputValue::System(SystemValue::InstanceIndex)));
            }
            if let Some(regs) = r.indirect_draw {
                let custom = |c: u32| InputValue::System(SystemValue::Custom(c));
                for (_, v) in vgprs.iter_mut() {
                    match v {
                        InputValue::System(SystemValue::VertexIndex) => *v = custom(emit::custom::VERTEX_ID_REL),
                        InputValue::System(SystemValue::InstanceIndex) => *v = custom(emit::custom::INSTANCE_ID_REL),
                        _ => {}
                    }
                }
                for (k, c) in [(regs[0], emit::custom::BASE_VERTEX), (regs[1], emit::custom::BASE_INSTANCE)] {
                    if k == 0xFF {
                        continue;
                    }
                    let s = base + k as u32;
                    if let Some(e) = sgprs.iter_mut().find(|(r, _)| *r as u32 == s) {
                        e.1 = custom(c);
                    }
                }
            }
            Stage::Vertex
        }
        ShaderStage::Pixel => {
            let ena = r.pixel_input_enable;
            let addr = if r.pixel_input_address != 0 { r.pixel_input_address | ena } else { ena };
            let mut next = 0u16;
            for (bit, n) in PIXEL_INPUTS {
                if addr >> bit & 1 == 0 {
                    continue;
                }
                for k in 0..n {
                    if ena >> bit & 1 != 0 {
                        vgprs.push((next, pixel_input_value(bit, k)));
                    }
                    next += 1;
                }
            }
            initial_exec = InitialExec::NotSystem(SystemValue::HelperInvocation);
            Stage::Pixel
        }
        ShaderStage::Unknown => Stage::Unknown,
    };
    TranslateOptions {
        wave_size: wave,
        stage: ssa_stage,
        sgprs,
        vgprs,
        initial_exec,
        optimize: true,
        fold_code_address: true,
        float_mode: r.float_mode,
        strict_mode_writes: strict_mode_writes(),
        float_rules: float_rules(),
        rewrites: translate::rewrites::ALL,
        fast_transcendental: false,
        native_ftz: false,
    }
}

/// Whether every lane of a compute program's waves runs: the work-group is
/// a whole number of guest waves and the dispatch has no thread limits
/// (partial last groups), so no lane starts inactive (the lanes the
/// `HelperInvocation` initial EXEC stands for, `io.rs`).
pub fn full_waves(r: &ShaderCompileRequest<'_>) -> bool {
    let s = r.local_size;
    let n = s[0].max(1) * s[1].max(1) * s[2].max(1);
    let wave = if r.wave_size == 64 { 64 } else { 32 };
    n % wave == 0 && !r.bindings.uses_dispatch_thread_limits
}

/// The translation switches of `emit` applied to `options`.
fn apply_emit_options(options: &mut TranslateOptions, emit: &EmitOptions) {
    use translate::rewrites::{FLUSH_ELISION, HALF_FMA_FAST, LANE_ACTIVE_EXEC, LEAN_FLUSH};
    options.rewrites = (if emit.lean_flush { LEAN_FLUSH } else { 0 })
        | (if emit.subgroup_ops { LANE_ACTIVE_EXEC } else { 0 })
        | (if emit.flush_elision { FLUSH_ELISION } else { 0 })
        | (if emit.half_fma_fast { HALF_FMA_FAST } else { 0 });
    options.fast_transcendental = emit.fast_transcendental;
}

/// `RPS5E_NGG_VERTEX_WAVE_INFO=1` (opt-in): merged (ES+GS) programs on
/// the vertex path read a one-wave subgroup: s3 counts the subgroup's
/// active invocations and sets the wave count to 1, s2 carries the
/// subgroup vertex and primitive counts. Off (the default), s3 comes from
/// the highest active lane, with no wave count, and s2 is 0: with the
/// one-wave inputs ASTRO BOT's intro drew large flat polygons over the
/// space scene and ran ~35 % slower. Part of the disk-cache
/// key.
pub fn vertex_wave_info_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_NGG_VERTEX_WAVE_INFO").is_ok_and(|v| v == "1"))
}

/// `RPS5E_STRICT_MODE_WRITES=1`: MODE round-field changes are tracked
/// through the control flow either way; strict, what is not modelled
/// (round bits that differ between paths or come from an SGPR, a directed
/// mode set at run time before an instruction that rounds to nearest
/// only, writes of other MODE fields) fails translation (the draws are
/// skipped) instead of running with the declared bits. Modules in the disk
/// cache keep the mode they were compiled with: clear it when switching.
fn strict_mode_writes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RPS5E_STRICT_MODE_WRITES").is_ok_and(|v| v == "1"))
}

/// `RPS5E_FLOAT_RULES`: which hardware-measured float rules the
/// translator applies (`ps5shader::translate::float_rules` bits: 1 tiny
/// products rounded before the flush, 2 v_div_scale_f32 input flush, 4
/// v_div_fmas_f32 NaN selection, 8 f16/f64 denormal modes, 0x200 directed
/// f16 rounding, 0x400 the v_sin/v_cos series where the host strays; see
/// the module for the rest). Unset: all; `0`
/// or `off`: none (the translation before them); otherwise a mask
/// (decimal or `0x` hex). Part of the disk-cache key.
pub fn float_rules() -> u32 {
    static RULES: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *RULES.get_or_init(|| parse_float_rules(std::env::var("RPS5E_FLOAT_RULES").ok().as_deref()))
}

fn parse_float_rules(value: Option<&str>) -> u32 {
    use ps5shader::translate::float_rules::ALL;
    match value.map(str::trim) {
        None | Some("") => ALL,
        Some(v) if v.eq_ignore_ascii_case("off") => 0,
        Some(v) => {
            let parsed = match v.strip_prefix("0x").or_else(|| v.strip_prefix("0X")) {
                Some(hex) => u32::from_str_radix(hex, 16).ok(),
                None => v.parse::<u32>().ok(),
            };
            match parsed {
                Some(mask) => mask & ALL,
                None => {
                    log::warn!("RPS5E_FLOAT_RULES={v}: not a mask, all rules apply");
                    ALL
                }
            }
        }
    }
}

/// `RPS5E_LOG_SERIES=0`: v_log_f32 (and the f16 form computed through it)
/// is the host's log2 everywhere, without the series near 1.0. Part of the
/// disk-cache key.
pub fn log_series_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_LOG_SERIES").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off")))
}

/// `RPS5E_STORAGE_RESINFO_MIPS=0`: image_get_resinfo of a storage binding
/// reports the bound view's base-level size and one level whatever its mip
/// operand, as before. (Part of the disk-cache key.)
pub fn storage_resinfo_mips_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_STORAGE_RESINFO_MIPS").is_ok_and(|v| v == "0"))
}

/// Vector-write runs at least this long share one EXEC branch in compute
/// modules (see `ps5shader::opt::exec_runs`): `RPS5E_EXEC_RUN=0` turns the
/// grouping off, `RPS5E_EXEC_RUN_MIN=<n>` (n >= 2) sets the length. 0 when
/// off.
pub fn exec_run_minimum() -> usize {
    static MINIMUM: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MINIMUM.get_or_init(|| {
        if !crate::backend::ab::switch_on("RPS5E_EXEC_RUN") {
            return 0;
        }
        std::env::var("RPS5E_EXEC_RUN_MIN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 2)
            .unwrap_or(ps5shader::opt::exec_runs::DEFAULT_MINIMUM)
    })
}

/// The SPIR-V compiler.
pub struct SpirvCompiler {
    features: Features,
    memory: Option<&'static GuestGpuMemory>,
    dump: Option<std::path::PathBuf>,
    /// On-disk module cache (see [`disk`]).
    disk: Option<std::path::PathBuf>,
    /// Shortest vector-write run grouped under one EXEC branch (0: off).
    pub exec_run_minimum: usize,
    /// Split wave64 compute programs over two 32-wide host subgroups where
    /// possible ([`wave_split`]; `RPS5E_WAVE64_SPLIT`).
    pub wave64_split: bool,
    /// Shuffles within a half of a split wave use the host subgroup (off:
    /// every shuffle goes through workgroup memory, for tests).
    pub wave64_native_shuffles: bool,
    /// Guest VGPRs from which a program runs split ([`wave_split::min_vgprs`]).
    pub wave64_split_min_vgprs: u32,
}

impl SpirvCompiler {
    pub fn new(features: Features, memory: Option<&'static GuestGpuMemory>) -> SpirvCompiler {
        let dump = std::env::var_os("RPS5E_SPIRV_DUMP").map(std::path::PathBuf::from);
        if let Some(d) = &dump {
            let _ = std::fs::create_dir_all(d);
        }
        SpirvCompiler { features, memory, dump, disk: None, exec_run_minimum: exec_run_minimum(), wave64_split: wave_split::enabled(), wave64_native_shuffles: true, wave64_split_min_vgprs: wave_split::min_vgprs() }
    }

    /// The compiler of the pipeline provider: with the on-disk module cache.
    pub fn for_device(dev: &GpuDevice, memory: &'static GuestGpuMemory) -> SpirvCompiler {
        let features = Features::of(dev);
        let mut c = SpirvCompiler::new(features, Some(memory));
        c.disk = disk::default_dir(&features);
        if let Some(d) = &c.disk {
            disk::mark_used_and_prune(d);
            log::info!("[shader] module cache: {}", d.display());
        }
        c
    }

    /// Split every wave64 compute program that qualifies (`on`, whatever
    /// its VGPR count) or none, regardless of `RPS5E_WAVE64_SPLIT`; the
    /// modules stay out of the disk cache (its key holds the environment's
    /// choice).
    pub fn set_wave64_split(&mut self, on: bool, native_shuffles: bool) {
        self.wave64_split = on;
        self.wave64_native_shuffles = native_shuffles;
        self.wave64_split_min_vgprs = 0;
        self.disk = None;
    }

    /// Decide whether the wave64 compute program `func` runs split over two
    /// 32-wide host subgroups ([`wave_split`]); on success `func` is the
    /// program with its lane masks projected.
    fn split_wave64(&self, func: &mut Function, r: &ShaderCompileRequest<'_>) -> Result<wave_split::SplitPlan, String> {
        let vgprs = wave_split::guest_vgprs(r.program());
        if vgprs < self.wave64_split_min_vgprs {
            return Err(format!("{vgprs} VGPRs"));
        }
        let mut f = func.clone();
        if ps5shader::opt::wave64_masks::project_wave64_lane_masks(&mut f, Some(SystemValue::HelperInvocation)) > 0 {
            ps5shader::opt::fold::cse(&mut f);
            ps5shader::opt::fold::fold(&mut f);
            ps5shader::opt::dce::dce(&mut f);
        }
        let plan = wave_split::plan(&f, r.local_size, r.local_data_share_dwords, &self.features, self.wave64_native_shuffles)?;
        *func = f;
        Ok(plan)
    }

    /// Compile one permutation (the trait entry point without the input
    /// wrapper, for tests and tools).
    pub fn compile_request(&self, stage: ShaderStage, r: &ShaderCompileRequest<'_>, key: u64) -> Result<CompiledShader, CompileError> {
        self.compile_mesh_request(stage, r, key, None)
    }

    /// [`Self::compile_request`] with an optional geometry-subgroup
    /// configuration (vertex stage run as a mesh shader).
    pub fn compile_mesh_request(
        &self,
        stage: ShaderStage,
        r: &ShaderCompileRequest<'_>,
        key: u64,
        mesh: Option<crate::pipeline::compiler::MeshConfig>,
    ) -> Result<CompiledShader, CompileError> {
        self.compile_from(stage, r, key, mesh, None)
    }

    /// [`Self::compile_mesh_request`] reading code words (jump tables) from
    /// `code` instead of guest memory when given. The words read are
    /// returned with the module.
    fn compile_from(
        &self,
        stage: ShaderStage,
        r: &ShaderCompileRequest<'_>,
        key: u64,
        mesh: Option<crate::pipeline::compiler::MeshConfig>,
        code: Option<&(dyn ps5shader::CodeSource + Sync)>,
    ) -> Result<CompiledShader, CompileError> {
        let mesh = mesh.filter(|_| stage == ShaderStage::Vertex);
        let feats = self.features;
        let mut options = translate_options(stage, r, mesh.is_some());
        apply_emit_options(&mut options, &feats.emit);
        if feats.emit.float_class_compare && feats.preserve_inf_nan[1] {
            options.rewrites |= translate::rewrites::FLOAT_CLASS_COMPARE;
        }
        if self.features.emit.full_wave_exec && stage == ShaderStage::Compute && mesh.is_none() && full_waves(r) {
            options.initial_exec = InitialExec::All;
        }
        // One decision for the translation and the emitted module: the
        // module's FLOAT_MODE flushes f32 everywhere and its opcodes agree
        // with the host's flush.
        let native_ftz = self.features.emit.native_ftz
            && self.features.denorm_flush_to_zero32
            && translate::native_ftz_applies(r.program(), r.float_mode, options.float_rules);
        options.native_ftz = native_ftz;
        let memory = self.memory;
        let reads = std::cell::RefCell::new(Vec::new());
        let read = |a: u64| -> Option<u32> {
            let w = match code {
                Some(c) => c.read_u32(a),
                None => {
                    let m = memory?;
                    let mut b = [0u8; 4];
                    m.read_backing(a, &mut b).then(|| u32::from_le_bytes(b))
                }
            };
            if let Some(v) = w {
                reads.borrow_mut().push((a, v));
            }
            w
        };
        let source: Option<&dyn ps5shader::CodeSource> = if memory.is_some() || code.is_some() { Some(&read) } else { None };
        let mut compiled = translate::compile(r.program(), &options, source).map_err(|e| match e {
            translate::TranslateError::Unsupported { .. } => CompileError::Unsupported(e.to_string()),
            other => CompileError::Failed(other.to_string()),
        })?;
        if r.relocatable_code {
            ps5shader::opt::fold::unfold_code_address(&mut compiled.function, RELOCATION_SPAN);
        }
        let mut last = [0u64; 10];
        for (k, ns) in compiled.stats.phase_ns.iter().enumerate() {
            stats::PHASE_NS[k].fetch_add(*ns, std::sync::atomic::Ordering::Relaxed);
            last[k] = *ns;
        }
        for (k, ns) in compiled.stats.opt.pass_ns.iter().enumerate() {
            stats::PASS_NS[k].fetch_add(*ns, std::sync::atomic::Ordering::Relaxed);
        }
        let t = std::time::Instant::now();
        let compute = stage == ShaderStage::Compute && mesh.is_none();
        let wave64 = compiled.function.wave_size == 64;
        let s = r.local_size;
        let waves = (s[0].max(1) * s[1].max(1) * s[2].max(1)).div_ceil(64);
        // 64-wide subgroups the pipeline requires (a device whose default
        // subgroup is narrower).
        let required64 = compute && wave64 && self.features.subgroup64_compute && waves <= self.features.max_compute_workgroup_subgroups;
        let narrow = wave64 && self.features.subgroup_size == 32 && !required64;
        let mut split = None;
        if compute && narrow && self.wave64_split {
            let t = std::time::Instant::now();
            let planned = self.split_wave64(&mut compiled.function, r);
            last[6] = t.elapsed().as_nanos() as u64;
            match planned {
                Ok(p) => {
                    log::debug!("[shader] cs {:#x}: wave64 split ({} waves, exchanges {:?})", r.program().address, p.waves, p.counts);
                    split = Some(p);
                }
                Err(why) => log::debug!("[shader] cs {:#x}: wave64 paired: {why}", r.program().address),
            }
            stats::note_split(split.is_some());
        }
        let paired_stage = stage == ShaderStage::Compute || mesh.is_some();
        let lanes = if paired_stage && narrow && split.is_none() { 2 } else { 1 };
        // Runs of vector writes under one EXEC branch: compute only (a pixel
        // wave already runs with helper lanes, and the extra branch per run
        // costs it more than the selects it removes), and only with one guest
        // lane per host invocation (a branch cannot diverge between the two
        // halves of a paired invocation).
        let t_runs = std::time::Instant::now();
        if compute && lanes == 1 {
            let runs = ps5shader::opt::exec_runs::group_exec_runs(&mut compiled.function, self.exec_run_minimum);
            // Inside a run the written values are read without their EXEC
            // selects: flushes of flushed values show again.
            if runs > 0 && options.rewrites & translate::rewrites::FLUSH_ELISION != 0 {
                let n = ps5shader::opt::denorm::elide_flushes(&mut compiled.function);
                if n > 0 {
                    ps5shader::opt::dce::dce(&mut compiled.function);
                    compiled.stats.opt.elided_flushes += n;
                }
            }
        }
        last[7] = t_runs.elapsed().as_nanos() as u64;
        stats::PHASE_NS[5].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let t = std::time::Instant::now();
        // A sample takes the sampler its specialization assigns (the point
        // copy of a sampler shared by integer and float images, the
        // compare copy of one shared by depth-reference and plain samples).
        let mut plan_uses = r.plan.resource_uses();
        for u in plan_uses.iter_mut().filter(|u| u.operand == ps5shader::resources::ResourceOperand::Sampler) {
            if let Some(&s) = r.resources.sampler_by_memory_index.get(&u.memory_index) {
                u.binding = ps5shader::resources::ResourceBinding::Sampler(s);
            }
        }
        let uses = translate::bind::attach_resource_uses(&compiled.function, &plan_uses);
        last[8] = t.elapsed().as_nanos() as u64;
        stats::PHASE_NS[6].fetch_add(last[8], std::sync::atomic::Ordering::Relaxed);
        let t = std::time::Instant::now();
        let input = EmitInput { func: &compiled.function, stage, request: r, uses: &uses, features: feats, lanes, mesh, split, native_ftz };
        let (spirv, mut reflection) = emit_module(&input).map_err(CompileError::Unsupported)?;
        if required64 {
            reflection.required_subgroup_size = 64;
        }
        last[9] = t.elapsed().as_nanos() as u64;
        stats::PHASE_NS[7].fetch_add(last[9], std::sync::atomic::Ordering::Relaxed);
        stats::set_last(last);
        if let Some(dir) = &self.dump {
            let tag = match stage {
                ShaderStage::Vertex if mesh.is_some() => "ms",
                ShaderStage::Vertex => "vs",
                ShaderStage::Pixel => "ps",
                ShaderStage::Compute => "cs",
                ShaderStage::Unknown => "xx",
            };
            let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
            let stem = format!("{tag}_{:x}_{key:016x}", r.program().address);
            let _ = std::fs::write(dir.join(format!("{stem}.spv")), bytes);
            // The guest program and the IR the module was emitted from.
            let _ = std::fs::write(dir.join(format!("{stem}.s")), r.program().to_string());
            let _ = std::fs::write(dir.join(format!("{stem}.ir")), format!("; lanes {lanes} split {} required64 {required64} float_mode {:?} native_ftz {native_ftz} rewrites {:#x} opt {:?}\n{}", input.split.is_some(), r.float_mode, options.rewrites, compiled.stats.opt, compiled.function));
        }
        if reflection.private_lds_bytes != 0 {
            log::info!("[shader] {stage:?} {:#x}: {} bytes of private LDS per invocation", r.program().address, reflection.private_lds_bytes);
        }
        let mut code_reads = reads.into_inner();
        code_reads.sort_unstable();
        code_reads.dedup();
        Ok(CompiledShader { spirv, reflection: ShaderReflection { bindings: layout_bindings(stage, r), ..reflection }, code_reads })
    }
}

impl ShaderCompiler for SpirvCompiler {
    fn compile(&self, input: &ShaderCompileInput<'_, '_>) -> Result<CompiledShader, CompileError> {
        // The permutation key identifies the module (program, stage,
        // specialization, layout and options), so a module compiled by an
        // earlier run is reused as is.
        if let Some(dir) = &self.disk {
            let t = std::time::Instant::now();
            if let Some((spirv, reflection, code_reads)) = disk::load(dir, input.key) {
                stats::DISK_HITS.add(t.elapsed().as_nanos() as u64);
                return Ok(CompiledShader { spirv, reflection: ShaderReflection { bindings: layout_bindings(input.stage, input.request), ..reflection }, code_reads });
            }
        }
        let c = stats::TRANSLATE.time(|| self.compile_from(input.stage, input.request, input.key, input.mesh, input.code))?;
        if let Some(dir) = &self.disk {
            disk::store_async(dir, input.key, c.spirv.clone(), c.reflection.clone(), c.code_reads.clone());
        }
        Ok(c)
    }

    fn compile_uncached(&self, input: &ShaderCompileInput<'_, '_>) -> Result<CompiledShader, CompileError> {
        self.compile_from(input.stage, input.request, input.key, input.mesh, input.code)
    }

    fn name(&self) -> &str {
        "spirv"
    }

    fn translator_stamp(&self) -> Option<u64> {
        Some(translator_stamp(&self.features))
    }

    fn forget_cached(&self, key: u64) {
        if let Some(dir) = &self.disk {
            disk::remove(dir, key);
        }
    }
}

/// How far past its entry a 64-bit constant counts as derived from a
/// relocatable program's code address (code and the tables after it).
pub const RELOCATION_SPAN: u64 = 4 << 20;

/// Emit the module of a translated function.
pub fn emit_module(input: &EmitInput<'_, '_>) -> Result<(Vec<u32>, ShaderReflection), String> {
    let mut e = emit::Emitter::new(input);
    e.emit_function()?;
    let r = input.request;
    let mut reflection = ShaderReflection { entry_point: "main".into(), ..ShaderReflection::default() };
    let (model, modes) = match input.stage {
        ShaderStage::Vertex if input.mesh.is_some() => {
            let m = input.mesh.unwrap();
            reflection.outputs =
                mesh::param_locations(&e).into_iter().map(|l| ReflectedVariable { location: l, components: 4, kind: ComponentKind::Float, flat: false }).collect();
            reflection.writes_viewport_index = mesh::writes_viewport_index(&e);
            let local = m.threads / input.lanes as u32;
            reflection.local_size = [local, 1, 1];
            (
                spirv::ExecutionModel::MeshEXT,
                vec![
                    (spirv::ExecutionMode::LocalSize, vec![local, 1, 1]),
                    (spirv::ExecutionMode::OutputVertices, vec![m.max_vertices]),
                    (spirv::ExecutionMode::OutputPrimitivesEXT, vec![m.max_primitives]),
                    (if m.points { spirv::ExecutionMode::OutputPoints } else if m.lines { spirv::ExecutionMode::OutputLinesEXT } else { spirv::ExecutionMode::OutputTrianglesEXT }, vec![]),
                ],
            )
        }
        ShaderStage::Vertex => {
            reflection.outputs =
                e.io.params.keys().map(|&l| ReflectedVariable { location: l, components: 4, kind: ComponentKind::Float, flat: false }).collect();
            reflection.writes_viewport_index = e.io.misc.contains_key(&(spirv::BuiltIn::ViewportIndex as u32));
            reflection.writes_point_size = e.io.misc.contains_key(&(spirv::BuiltIn::PointSize as u32));
            reflection.clip_distances = io::clip_distance_count(&e);
            (spirv::ExecutionModel::Vertex, vec![])
        }
        ShaderStage::Pixel => {
            reflection.outputs = r
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
                .collect();
            let mut modes = vec![(spirv::ExecutionMode::OriginUpperLeft, vec![])];
            if r.pixel_early_fragment_tests {
                modes.push((spirv::ExecutionMode::EarlyFragmentTests, vec![]));
            }
            if r.pixel_ordered {
                modes.push((spirv::ExecutionMode::PixelInterlockOrderedEXT, vec![]));
            }
            if e.io.frag_depth.is_some() {
                reflection.writes_depth = true;
                modes.push((spirv::ExecutionMode::DepthReplacing, vec![]));
            }
            if e.io.frag_stencil.is_some() {
                modes.push((spirv::ExecutionMode::StencilRefReplacingEXT, vec![]));
            }
            (spirv::ExecutionModel::Fragment, modes)
        }
        ShaderStage::Compute => {
            let s = r.local_size;
            // Quad derivatives need even x and y (a missing size is 2).
            let unset = if e.compute_derivatives { 2 } else { 1 };
            let g = [s[0].max(1), s[1].max(1), s[2].max(1)];
            let mut s = [if s[0] == 0 { unset } else { s[0] }, if s[1] == 0 { unset } else { s[1] }, s[2].max(1)];
            if input.lanes == 2 || input.split.is_some() {
                // Whole waves: two guest lanes per invocation take half the
                // waves' width, split waves all of it.
                let n = (g[0] * g[1] * g[2]).div_ceil(64) * if input.lanes == 2 { 32 } else { 64 };
                // Quads of consecutive invocations: a 2-wide grid.
                s = if e.compute_derivatives { [2, n / 2, 1] } else { [n, 1, 1] };
            }
            reflection.local_size = s;
            let mut modes = vec![(spirv::ExecutionMode::LocalSize, s.to_vec())];
            if e.compute_derivatives_linear {
                modes.push((spirv::ExecutionMode::DerivativeGroupLinearNV, vec![]));
            } else if e.compute_derivatives {
                modes.push((spirv::ExecutionMode::DerivativeGroupQuadsNV, vec![]));
            }
            (spirv::ExecutionModel::GLCompute, modes)
        }
        ShaderStage::Unknown => return Err("unknown shader stage".into()),
    };
    if e.res.lds.is_some() && e.lds_class() == spirv::StorageClass::Private {
        reflection.private_lds_bytes = e.res.lds_dwords * 4;
    }
    if e.res.uses_push {
        reflection.push_constants = Some((0, ps5shader::resources::push_data::BYTE_SIZE));
    }
    reflection.own_buffer_checks = e.feat.emit.guarded_loads && !e.unchecked_buffers;
    if e.feat.no_float64 && e.caps.contains_key(&(spirv::Capability::Float64 as u32)) {
        return Err("the program computes in 64-bit floats and the device lacks shaderFloat64".into());
    }
    let words = e.finish(model, &modes);
    Ok((words, reflection))
}

#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) mod ref_tests;
#[cfg(test)]
mod graphics_tests;
