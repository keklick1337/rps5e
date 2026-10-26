// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! IR operations.
//!
//! Semantics follow SPIR-V (and GLSL.std.450 for the extended math) so a
//! back-end can emit each operation with one or a few instructions. Guest
//! (RDNA) semantics are composed from these by the translator. Where SPIR-V
//! leaves a result undefined (a shift by the bit width or more, a float to
//! integer conversion out of range, a bit field past the width) the operation
//! is undefined here too; the interpreter reports it as an error so the
//! translator cannot silently rely on it.
//!
//! Every value is one invocation's value. An invocation is one guest lane.

use std::fmt;

/// Guest register, used by the register-level IR the translator produces
/// before SSA construction ([`Op::GetReg`] / [`Op::SetReg`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Reg {
    /// SGPR 0..=127 (VCC = 106/107, M0 = 124, EXEC = 126/127), `u32`.
    Sgpr(u8),
    /// VGPR 0..=255, `u32`.
    Vgpr(u16),
    /// This invocation's bit of the lane mask whose low word is SGPR `n`
    /// (wave64: the pair `n, n+1`), `bool`. For EXEC (126) it is always valid.
    SgprBit(u8),
    /// Whether [`Reg::SgprBit`] `n` is in sync with the SGPR words, `bool`
    /// (wave-uniform).
    SgprBitValid(u8),
    /// SCC, `bool`.
    Scc,
    /// Structurizer goto variable, `bool` (wave-uniform).
    Goto(u32),
    /// Dispatcher state of an irreducible graph: the next CFG block, `u32`
    /// (wave-uniform).
    Dispatch,
    /// Branch condition computed by the instruction ending a block
    /// (S_SUBVECTOR_LOOP_BEGIN / END), `bool` (wave-uniform).
    InstructionBranch,
}

impl Reg {
    pub const EXEC: u8 = 126;
    pub const VCC: u8 = 106;
    pub const M0: u8 = 124;

    pub fn ty(self) -> super::Type {
        match self {
            Reg::Sgpr(_) | Reg::Vgpr(_) | Reg::Dispatch => super::Type::U32,
            _ => super::Type::BOOL,
        }
    }
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reg::Sgpr(106) => f.write_str("vcc_lo"),
            Reg::Sgpr(107) => f.write_str("vcc_hi"),
            Reg::Sgpr(124) => f.write_str("m0"),
            Reg::Sgpr(126) => f.write_str("exec_lo"),
            Reg::Sgpr(127) => f.write_str("exec_hi"),
            Reg::Sgpr(n) => write!(f, "s{n}"),
            Reg::Vgpr(n) => write!(f, "v{n}"),
            Reg::SgprBit(126) => f.write_str("exec.bit"),
            Reg::SgprBit(n) => write!(f, "s{n}.bit"),
            Reg::SgprBitValid(n) => write!(f, "s{n}.valid"),
            Reg::Scc => f.write_str("scc"),
            Reg::Goto(n) => write!(f, "goto{n}"),
            Reg::Dispatch => f.write_str("dispatch"),
            Reg::InstructionBranch => f.write_str("branch"),
        }
    }
}

/// Integer comparison predicates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ICmp {
    Eq,
    Ne,
    ULt,
    ULe,
    UGt,
    UGe,
    SLt,
    SLe,
    SGt,
    SGe,
}

/// Float comparison predicates (SPIR-V Ford*/Funord*).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FCmp {
    OEq,
    ONe,
    OLt,
    OLe,
    OGt,
    OGe,
    /// Neither operand is NaN.
    Ord,
    /// Either operand is NaN.
    Uno,
    UEq,
    UNe,
    ULt,
    ULe,
    UGt,
    UGe,
}

/// Rounding of a narrowing float conversion or an integer to float
/// conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rounding {
    /// Round to nearest, ties to even.
    Rte,
    /// Round toward zero.
    Rtz,
}

/// Read-modify-write memory operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomicOp {
    Swap,
    /// Data operands are (new value, comparator); stores the new value when
    /// memory equals the comparator.
    CmpSwap,
    Add,
    Sub,
    SMin,
    UMin,
    SMax,
    UMax,
    And,
    Or,
    Xor,
    /// `old >= data ? 0 : old + 1`.
    Inc,
    /// `old == 0 || old > data ? data : old - 1`.
    Dec,
    FMin,
    FMax,
    /// DS_MIN_F32 / DS_MAX_F32: denormals order as zero (negative ones tie
    /// with -0, positive ones only with each other) but keep their bits,
    /// min keeps DATA on a tie, a NaN operand yields the other one and a
    /// signaling NaN is returned quieted.
    DsFMin,
    DsFMax,
    /// `data - old`.
    RSub,
    /// Data operands (mask, value): `(old & !mask) | value`.
    MaskOr,
    /// Float add.
    FAdd,
    /// DS_ADD_F32 (measured on RDNA1/RDNA2): both operands flushed to a
    /// signed zero with `flush_inputs`, the sum (nearest even) with
    /// `flush_result`; a NaN data operand is stored quieted, else a NaN old
    /// value quieted, and an invalid sum (+inf + -inf) stores 0xffc00000.
    DsFAdd { flush_inputs: bool, flush_result: bool },
    /// Float compare-and-swap: data operands (new value, comparator),
    /// compared as floats.
    FCmpSwap,
    /// Clamped subtract: `old >= data ? old - data : 0`.
    CSub,
    /// Data operands (sub, add): `old >= sub ? old - sub : old + add`.
    Wrap,
}

/// Width and extension of a sub-dword or multi-dword memory access.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Access {
    /// Zero-extended byte.
    U8,
    /// Sign-extended byte.
    I8,
    U16,
    I16,
    /// 1..=4 dwords.
    Dwords(u8),
}

impl Access {
    pub fn bytes(self) -> u32 {
        match self {
            Access::U8 | Access::I8 => 1,
            Access::U16 | Access::I16 => 2,
            Access::Dwords(n) => 4 * n as u32,
        }
    }

    /// Register dwords moved per invocation.
    pub fn dwords(self) -> u32 {
        match self {
            Access::Dwords(n) => n as u32,
            _ => 1,
        }
    }
}

/// MUBUF / MTBUF access fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferAttr {
    /// Guest instruction pc (resource planning key).
    pub pc: u32,
    pub idxen: bool,
    pub offen: bool,
    /// Instruction immediate offset in bytes.
    pub offset: u32,
    pub glc: bool,
    pub slc: bool,
}

/// How a buffer access interprets data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferData {
    /// Raw bytes.
    Raw(Access),
    /// Converted through the descriptor's format (MUBUF *_FORMAT_*) or the
    /// instruction's unified format (MTBUF), `components` 1..=4.
    ///
    /// Loads: untyped (`typed_format: None`) output component `i` follows
    /// the descriptor's DST_SEL field `i` (dword3 bits `3i..3i+2`: 0 -> 0,
    /// 1 -> 1 (1.0 for float formats, 0x3C00 with `d16`), 4..7 -> format
    /// component `(sel - 4) % count`); typed loads return the format's
    /// components in order and 0 past its component count. Format accesses
    /// are range-checked all or nothing: an element is in range only when
    /// every component of it is (loads and stores alike); an out-of-range
    /// load returns 0 for memory sources and the DST_SEL constants
    /// otherwise, a store is dropped.
    ///
    /// Stores write the whole element: components past `components` are
    /// encoded as 0. Conversions (see `interp::buffer`): UNORM/USCALED and
    /// SNORM/SSCALED round the exact product with the field maximum to
    /// nearest even and saturate; UINT/SINT saturate; f16 and the 11/10-bit
    /// floats truncate toward zero, clamp to the largest finite value and
    /// keep NaN payloads.
    ///
    /// `d16`: every register component is a 16-bit value in the low half of
    /// its u32 (loads convert to f16 / 16-bit integers, stores convert from
    /// them); the translator packs and unpacks register halves.
    Format { components: u8, typed_format: Option<u32>, d16: bool },
}

/// Image operation kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageOp {
    Sample,
    Gather4,
    Load,
    Store,
    GetResinfo,
    GetLod,
    Atomic(AtomicOp),
    /// BVH ray intersection (`bvh64` when the node pointer is 64-bit).
    BvhIntersect { bvh64: bool },
}

/// Image sample / gather address modifiers (the instruction suffixes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SampleFlags {
    /// `_c`: depth comparison reference.
    pub compare: bool,
    /// `_b`: LOD bias.
    pub bias: bool,
    /// `_l`: explicit LOD.
    pub lod: bool,
    /// `_lz`: LOD zero.
    pub lod_zero: bool,
    /// `_d` / `_cd`: explicit derivatives.
    pub derivatives: bool,
    /// `_cl`: LOD clamp.
    pub clamp: bool,
    /// `_o`: texel offsets.
    pub offset: bool,
    /// `_mip` for load/store.
    pub mip: bool,
    /// `_g16`: 16-bit derivatives (the other address components keep their
    /// width).
    pub g16: bool,
    /// `image_gather4h`: the component of four horizontally adjacent
    /// texels of one row, the left texel of the bilinear footprint shifted
    /// by -1, 0, 1 and 2 texels.
    pub horizontal: bool,
    /// `_pck`: raw texel bits (loads: the texel's bytes as dwords; stores:
    /// dwords written as the texel's bytes).
    pub packed: bool,
    /// `_pck_sgn`: raw bits sign-extended from the texel's width.
    pub packed_signed: bool,
    /// `_by2` / `_by4` / `_pck2` / `_pck4`: consecutive texels read at once
    /// (0 for one).
    pub elements: u8,
}

/// MIMG fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageAttr {
    pub pc: u32,
    pub op: ImageOp,
    pub flags: SampleFlags,
    /// MIMG DIM field (0 1D, 1 2D, 2 3D, 3 cube, 4 1D array, 5 2D array,
    /// 6 2D MSAA, 7 2D MSAA array).
    pub dim: u8,
    pub dmask: u8,
    pub a16: bool,
    pub d16: bool,
    pub glc: bool,
    pub slc: bool,
    /// Number of raw address dwords in the operand list.
    pub address_dwords: u8,
    /// Number of data dwords (store / atomic) after the address.
    pub data_dwords: u8,
}

/// EXP fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExportAttr {
    pub target: u8,
    pub enable: u8,
    pub compressed: bool,
    pub done: bool,
    pub valid_mask: bool,
}

/// Interpolation parameter of the current primitive for one attribute
/// channel: `P0` is the value at vertex 0, `P10` = P1 - P0, `P20` = P2 - P0.
/// The hardware interpolates as `fma(j, P20, fma(i, P10, P0))`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InterpParam {
    P0,
    P10,
    P20,
    /// FP16_INTERP_MODE: the f16 half (`high` selects bits 31:16) of the
    /// attribute channel, as f32: `HalfP0` is vertex 0's value, `HalfP10` /
    /// `HalfP20` the difference of vertex 1 / 2 and vertex 0 rounded to f16.
    HalfP0 { high: bool },
    HalfP10 { high: bool },
    HalfP20 { high: bool },
}

/// Values the hardware provides at wave launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SystemValue {
    LocalInvocationId(u8),
    WorkgroupId(u8),
    LocalInvocationIndex,
    GlobalInvocationId(u8),
    /// Wave index within the workgroup.
    SubgroupId,
    VertexIndex,
    InstanceIndex,
    PrimitiveId,
    /// Fragment coordinate component (x, y, z as float bits; w is 1/w).
    FragCoord(u8),
    FrontFacing,
    SampleId,
    HelperInvocation,
    /// Barycentric coordinate component (perspective `i`/`j` = 0/1,
    /// linear = 2/3, perspective centroid = 4/5, linear centroid = 6/7).
    Barycentric(u8),
    /// Pixel shader ancillary dword.
    Ancillary,
    /// Pixel sample coverage.
    SampleCoverage,
    /// Value of an SGPR whose meaning the caller defines (e.g. a
    /// hardware-generated SGPR such as the compute TG_SIZE word).
    Custom(u32),
}

/// An IR operation. `args` conventions are documented per variant; `exec`
/// is the invocation's guest EXEC bit (side effects happen only where it is
/// set; loads may be speculated).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    // ---- constants (module scope) ----
    /// Constant scalar; bits in the low bits of the payload.
    Const(u64),
    Undef,

    // ---- structure ----
    /// Arguments align with the block's predecessors.
    Phi,
    /// Register read (pre-SSA only).
    GetReg(Reg),
    /// `(value)` register write (pre-SSA only).
    SetReg(Reg),

    // ---- generic ----
    /// `(cond: bool, a, b)` -> a where cond, else b.
    Select,
    /// `(x)` reinterpret as the result type of the same bit width.
    Bitcast,
    /// `(c0, c1, ...)` -> vector.
    Construct,
    /// `(vector)` -> component.
    Extract(u8),
    /// `(vector, component)` -> vector with component replaced.
    Insert(u8),

    // ---- integer (u16/u32/u64; And/Or/Xor/Not also on bool) ----
    IAdd,
    ISub,
    IMul,
    /// High half of the unsigned / signed product.
    UMulHi,
    SMulHi,
    /// Undefined for a zero divisor.
    UDiv,
    UMod,
    /// `(a, b)` -> `u32x2 (a + b, carry)`.
    IAddCarry,
    /// `(a, b)` -> `u32x2 (a - b, borrow)`.
    ISubBorrow,
    INeg,
    SMin,
    SMax,
    UMin,
    UMax,
    /// `(x, count)`; count must be below the bit width.
    Shl,
    LShr,
    AShr,
    And,
    Or,
    Xor,
    Not,
    BitReverse,
    /// Population count, result u32.
    BitCount,
    /// Index of the lowest set bit, -1 (all ones) when zero; result u32.
    FindLsb,
    /// Index of the highest set bit, -1 when zero; result u32.
    FindUMsb,
    /// Index of the highest bit differing from the sign bit, -1 for 0 / -1.
    FindSMsb,
    /// `(base, offset, count)`; offset + count must not exceed the width.
    BitFieldUExtract,
    BitFieldSExtract,
    /// `(base, insert, offset, count)`.
    BitFieldInsert,
    /// Integer width conversion (zero-extend / truncate).
    UConvert,
    /// Integer width conversion (sign-extend / truncate).
    SConvert,
    ICmp(ICmp),

    // ---- float (f16/f32/f64) ----
    FAdd,
    FSub,
    FMul,
    FDiv,
    /// Fused multiply-add, single rounding.
    FFma,
    FNeg,
    FAbs,
    /// Minimum returning the non-NaN operand when one is NaN (NMin).
    FMin,
    FMax,
    FFloor,
    FCeil,
    FTrunc,
    FRoundEven,
    FSqrt,
    /// 1 / sqrt(x).
    FInvSqrt,
    /// 1 / x.
    FRcp,
    FExp2,
    FLog2,
    /// Sine of radians.
    FSin,
    FCos,
    /// `(x, exp: u32 as signed)` -> x * 2^exp, rounded once to the result
    /// type (any exponent; zeros, infinities and NaNs are unchanged).
    FLdexp,
    FCmp(FCmp),
    /// Float width conversion; the rounding applies when narrowing.
    FConvert(Rounding),
    /// Float to signed / unsigned integer, rounding toward zero. The value
    /// must be representable.
    FToS,
    FToU,
    SToF(Rounding),
    UToF(Rounding),

    // ---- invocation / subgroup ----
    /// Guest lane index (subgroup invocation id), u32.
    LaneId,
    /// `(bool)` -> `u32x2` mask of the invocations where it is true (low and
    /// high lane words).
    Ballot,
    /// `(lo: u32, hi: u32)` -> this invocation's bit of the 64-bit mask.
    InverseBallot,
    /// `(value, lane: u32)` -> value of invocation `lane`; `lane` is the same
    /// for every invocation and below the wave size.
    ReadLane,
    /// `(value, lane: u32)` -> value of invocation `lane` (per invocation,
    /// below the wave size).
    Shuffle,

    // ---- inputs ----
    /// Shader user data SGPR value, u32 (wave-uniform).
    UserData(u32),
    SystemValue(SystemValue),
    /// Guest code base address (the program's entry address), u64.
    CodeAddress,
    /// Attribute interpolation parameter, f32.
    InterpParam { attr: u8, chan: u8, param: InterpParam },

    // ---- memory (all take a guest pc for resource planning) ----
    /// `(address: u64)` -> u32, wave-uniform constant-cache load.
    ScalarLoad { pc: u32, dword: u8 },
    /// `(desc: u32x4, offset: u32)` -> u32, wave-uniform buffer load. The
    /// offset is dword aligned; an unaligned offset (the translator uses
    /// `u32::MAX`) marks an offset sum that overflowed 32 bits and reads 0.
    ScalarBufferLoad { pc: u32, dword: u8 },
    /// `(desc: u32x4, vindex, voffset, soffset, exec)` -> data.
    ///
    /// Range checking is per access unit, never all-or-nothing: each dword
    /// of a raw multi-dword access (and each sub-dword access) is checked on
    /// its own at byte `extra` (`4 * k`) of the access, so a dwordx2 load
    /// straddling the end returns `[data, 0]` and a straddling dwordx3 store
    /// writes only its in-range dwords. With `offset = (offen ? voffset : 0)
    /// + attr.offset + extra` (32-bit wrapping) and `index = idxen ? vindex :
    /// 0`, an access of `bytes` bytes is in range by OOB_SELECT (dword3 bits
    /// 28..29) (formatted accesses additionally need a nonzero format):
    /// 0 with stride 0, or 3: `soffset <= records && offset + bytes <=
    /// records - soffset`; 0 with a stride: `index * stride + offset +
    /// soffset + bytes <= stride * records`; 1: `index < records`; 2:
    /// `records != 0`. Out-of-range loads read 0 and stores are dropped.
    BufferLoad { attr: BufferAttr, data: BufferData },
    /// `(desc, vindex, voffset, soffset, data: u32 or u32xN, exec)`; range
    /// checked per access unit like `BufferLoad`.
    BufferStore { attr: BufferAttr, data: BufferData },
    /// `(desc, vindex, voffset, soffset, data, [comparator,] exec)` -> old;
    /// range checked like `BufferLoad` (4 or 8 bytes).
    BufferAtomic { attr: BufferAttr, op: AtomicOp },
    /// `(address: u64, exec)` -> data. `flat`: a FLAT-segment access. In a
    /// compute program its address is routed per invocation by the high
    /// dword: [`SHARED_APERTURE_HIGH`] reaches LDS and
    /// [`PRIVATE_APERTURE_HIGH`] this invocation's scratch, at the byte
    /// offset in the low dword (each aperture is a 4 GiB window). There an
    /// element (1, 2 or 4 bytes; each dword of a wider access on its own)
    /// is read or written whole when all its bytes lie inside the allocated
    /// memory, at any byte alignment; otherwise it reads 0 and a store is
    /// dropped. Every other address is a device address. `glc`: a globally
    /// coherent access (it sees stores of other work-groups made while the
    /// program runs).
    ///
    /// [`SHARED_APERTURE_HIGH`]: crate::inline_constants::SHARED_APERTURE_HIGH
    /// [`PRIVATE_APERTURE_HIGH`]: crate::inline_constants::PRIVATE_APERTURE_HIGH
    GlobalLoad { pc: u32, access: Access, flat: bool, glc: bool },
    /// `(address: u64, data, exec)`; `flat` as for `GlobalLoad`.
    GlobalStore { pc: u32, access: Access, flat: bool, glc: bool },
    /// `(address: u64, data, [comparator,] exec)` -> old; `flat` as for
    /// `GlobalLoad`, except that an aperture operand must be aligned to its
    /// size (4 or 8 bytes) and lie inside the allocated memory, or the
    /// atomic does nothing and returns 0.
    GlobalAtomic { pc: u32, op: AtomicOp, flat: bool },
    /// `(byte offset: u32, exec)` -> data, per-invocation private memory.
    ScratchLoad { pc: u32, access: Access },
    /// `(byte offset: u32, data, exec)`.
    ScratchStore { pc: u32, access: Access },
    /// `(byte address: u32, exec)` -> data, workgroup shared memory.
    LdsLoad { access: Access },
    /// `(byte address: u32, data, exec)`.
    LdsStore { access: Access },
    /// `(byte address: u32, data, [comparator,] exec)` -> old.
    LdsAtomic { op: AtomicOp },
    /// `(byte address: u32, count: u32)` -> old counter. One wave-wide
    /// atomic add (`consume`: subtract) of `count`; the old value is
    /// returned to every invocation.
    LdsAppend { consume: bool },
    /// `(byte address: u32, exec)` -> data, global data share (GDS). Like
    /// the LDS forms: the address wraps at 64 KiB and dwords past the end of
    /// the GDS read 0 (stores and atomics there are dropped).
    GdsLoad { access: Access },
    /// `(byte address: u32, data, exec)`.
    GdsStore { access: Access },
    /// `(byte address: u32, data, [comparator,] exec)` -> old.
    GdsAtomic { op: AtomicOp },
    /// `(byte address: u32, count: u32)` -> old counter: one wave-wide
    /// atomic add (`consume`: subtract) on a GDS dword, the old value
    /// returned to every invocation.
    GdsAppend { consume: bool },
    /// `(texture desc: u32x8, sampler desc: u32x4, address dwords...,
    /// data dwords..., exec)` -> result dwords (void for stores).
    Image(ImageAttr),

    // ---- outputs / control ----
    /// `(x, y, z, w: u32, exec)`.
    Export(ExportAttr),
    /// Workgroup execution and memory barrier (s_barrier).
    Barrier,
    /// Subgroup-scope shared-memory barrier: makes this wave's LDS writes
    /// visible to its other invocations.
    SubgroupLdsBarrier,
    /// Vector data cache invalidation (buffer_gl0_inv / buffer_gl1_inv):
    /// later buffer and image reads of the wave see device memory, with no
    /// rendezvous of the workgroup.
    DataCacheInvalidate,
    /// `(m0: u32)`, s_sendmsg.
    SendMessage { message: u16 },
    /// `() -> u64`: wave-uniform counter, non-decreasing between reads of
    /// one invocation (shader clock, or the constant-rate real-time clock
    /// when `realtime`). Never merged or removed.
    Clock { realtime: bool },
}

impl Op {
    /// Writes memory, registers or outputs (never removed when unused).
    pub fn has_side_effects(self) -> bool {
        match self {
            Op::SetReg(_)
            | Op::BufferStore { .. }
            | Op::BufferAtomic { .. }
            | Op::GlobalStore { .. }
            | Op::GlobalAtomic { .. }
            | Op::ScratchStore { .. }
            | Op::LdsStore { .. }
            | Op::LdsAtomic { .. }
            | Op::LdsAppend { .. }
            | Op::GdsStore { .. }
            | Op::GdsAtomic { .. }
            | Op::GdsAppend { .. }
            | Op::Export(_)
            | Op::Barrier
            | Op::SubgroupLdsBarrier
            | Op::DataCacheInvalidate
            | Op::SendMessage { .. }
            | Op::Clock { .. } => true,
            Op::Image(a) => matches!(a.op, ImageOp::Store | ImageOp::Atomic(_)),
            _ => false,
        }
    }

    /// Reads memory that a side effect may change (cannot be reordered across
    /// stores or deduplicated).
    pub fn reads_memory(self) -> bool {
        matches!(
            self,
            Op::ScalarLoad { .. }
                | Op::ScalarBufferLoad { .. }
                | Op::BufferLoad { .. }
                | Op::GlobalLoad { .. }
                | Op::ScratchLoad { .. }
                | Op::LdsLoad { .. }
                | Op::GdsLoad { .. }
                | Op::Image(_)
        )
    }

    /// Accesses workgroup shared memory (a FLAT-segment access may, through
    /// the shared aperture).
    pub fn is_lds_read(self) -> bool {
        matches!(
            self,
            Op::LdsLoad { .. } | Op::LdsAtomic { .. } | Op::LdsAppend { .. } | Op::GlobalLoad { flat: true, .. } | Op::GlobalAtomic { flat: true, .. }
        )
    }

    pub fn is_lds_write(self) -> bool {
        matches!(
            self,
            Op::LdsStore { .. } | Op::LdsAtomic { .. } | Op::LdsAppend { .. } | Op::GlobalStore { flat: true, .. } | Op::GlobalAtomic { flat: true, .. }
        )
    }

    /// Takes the trailing `exec` predicate operand.
    pub fn has_exec_operand(self) -> bool {
        matches!(
            self,
            Op::BufferLoad { .. }
                | Op::BufferStore { .. }
                | Op::BufferAtomic { .. }
                | Op::GlobalLoad { .. }
                | Op::GlobalStore { .. }
                | Op::GlobalAtomic { .. }
                | Op::ScratchLoad { .. }
                | Op::ScratchStore { .. }
                | Op::LdsLoad { .. }
                | Op::LdsStore { .. }
                | Op::LdsAtomic { .. }
                | Op::GdsLoad { .. }
                | Op::GdsStore { .. }
                | Op::GdsAtomic { .. }
                | Op::Image(_)
                | Op::Export(_)
        )
    }

    /// Result depends on other invocations (cannot be moved across control
    /// flow that changes which invocations reach it; here all invocations
    /// run in step, so these are ordinary values).
    pub fn is_subgroup(self) -> bool {
        matches!(self, Op::Ballot | Op::ReadLane | Op::Shuffle | Op::LdsAppend { .. } | Op::GdsAppend { .. })
    }

    /// Pure computation that may be removed when unused and folded when its
    /// operands are constant.
    pub fn is_pure(self) -> bool {
        !self.has_side_effects() && !self.reads_memory() && !matches!(self, Op::GetReg(_) | Op::Phi)
    }

    pub fn name(self) -> String {
        match self {
            Op::Const(v) => format!("const {v:#x}"),
            Op::Undef => "undef".into(),
            Op::Phi => "phi".into(),
            Op::GetReg(r) => format!("get {r}"),
            Op::SetReg(r) => format!("set {r}"),
            Op::Select => "select".into(),
            Op::Bitcast => "bitcast".into(),
            Op::Construct => "construct".into(),
            Op::Extract(i) => format!("extract.{i}"),
            Op::Insert(i) => format!("insert.{i}"),
            Op::ICmp(p) => format!("icmp.{}", format!("{p:?}").to_lowercase()),
            Op::FCmp(p) => format!("fcmp.{}", format!("{p:?}").to_lowercase()),
            Op::FConvert(r) => format!("fconvert.{}", format!("{r:?}").to_lowercase()),
            Op::SToF(r) => format!("stof.{}", format!("{r:?}").to_lowercase()),
            Op::UToF(r) => format!("utof.{}", format!("{r:?}").to_lowercase()),
            Op::UserData(i) => format!("user_data {i}"),
            Op::SystemValue(s) => format!("system_value {s:?}"),
            Op::InterpParam { attr, chan, param } => format!("interp_param attr{attr}.{} {param:?}", ["x", "y", "z", "w"][chan as usize & 3]),
            Op::ScalarLoad { pc, dword } => format!("scalar_load @{pc:#x}.{dword}"),
            Op::ScalarBufferLoad { pc, dword } => format!("scalar_buffer_load @{pc:#x}.{dword}"),
            Op::BufferLoad { attr, data } => format!("buffer_load @{:#x} {data:?} {}", attr.pc, buffer_flags(&attr)),
            Op::BufferStore { attr, data } => format!("buffer_store @{:#x} {data:?} {}", attr.pc, buffer_flags(&attr)),
            Op::BufferAtomic { attr, op } => format!("buffer_atomic.{op:?} @{:#x} {}", attr.pc, buffer_flags(&attr)),
            Op::GlobalLoad { pc, access, flat, glc } => format!("{}_load @{pc:#x} {access:?}{}", if flat { "flat" } else { "global" }, if glc { " glc" } else { "" }),
            Op::GlobalStore { pc, access, flat, glc } => format!("{}_store @{pc:#x} {access:?}{}", if flat { "flat" } else { "global" }, if glc { " glc" } else { "" }),
            Op::GlobalAtomic { pc, op, flat } => format!("{}_atomic.{op:?} @{pc:#x}", if flat { "flat" } else { "global" }),
            Op::ScratchLoad { pc, access } => format!("scratch_load @{pc:#x} {access:?}"),
            Op::ScratchStore { pc, access } => format!("scratch_store @{pc:#x} {access:?}"),
            Op::LdsLoad { access } => format!("lds_load {access:?}"),
            Op::LdsStore { access } => format!("lds_store {access:?}"),
            Op::LdsAtomic { op } => format!("lds_atomic.{op:?}"),
            Op::LdsAppend { consume } => if consume { "lds_consume" } else { "lds_append" }.into(),
            Op::GdsLoad { access } => format!("gds_load {access:?}"),
            Op::GdsStore { access } => format!("gds_store {access:?}"),
            Op::GdsAtomic { op } => format!("gds_atomic.{op:?}"),
            Op::GdsAppend { consume } => if consume { "gds_consume" } else { "gds_append" }.into(),
            Op::Image(a) => format!(
                "image.{:?} @{:#x} dim{} dmask{:#x}{}{}{}",
                a.op,
                a.pc,
                a.dim,
                a.dmask,
                sample_flags(&a.flags),
                if a.a16 { " a16" } else { "" },
                if a.d16 { " d16" } else { "" }
            ),
            Op::Export(e) => format!(
                "export target{} en{:#x}{}{}{}",
                e.target,
                e.enable,
                if e.compressed { " compr" } else { "" },
                if e.done { " done" } else { "" },
                if e.valid_mask { " vm" } else { "" }
            ),
            Op::SendMessage { message } => format!("sendmsg {message:#x}"),
            other => {
                let s = format!("{other:?}");
                let mut out = String::new();
                for (i, c) in s.chars().enumerate() {
                    if c.is_ascii_uppercase() {
                        if i != 0 {
                            out.push('_');
                        }
                        out.push(c.to_ascii_lowercase());
                    } else {
                        out.push(c);
                    }
                }
                out
            }
        }
    }
}

fn buffer_flags(a: &BufferAttr) -> String {
    let mut s = String::new();
    if a.idxen {
        s.push_str("idxen ");
    }
    if a.offen {
        s.push_str("offen ");
    }
    if a.offset != 0 {
        s.push_str(&format!("offset:{} ", a.offset));
    }
    if a.glc {
        s.push_str("glc ");
    }
    if a.slc {
        s.push_str("slc ");
    }
    s.trim_end().to_string()
}

fn sample_flags(f: &SampleFlags) -> String {
    let mut s = String::new();
    for (on, name) in [
        (f.compare, "c"),
        (f.bias, "b"),
        (f.lod, "l"),
        (f.lod_zero, "lz"),
        (f.derivatives, "d"),
        (f.clamp, "cl"),
        (f.offset, "o"),
        (f.mip, "mip"),
    ] {
        if on {
            s.push(' ');
            s.push_str(name);
        }
    }
    s
}
