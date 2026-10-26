// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Nodes of the uniform (scalar) value graph and the semantics of their
//! operations.

use std::fmt;

/// The width of a graph value: a lane-mask bit, one dword, or a dword pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ValueType {
    Bool,
    U32,
    U64,
}

/// Node kinds of the value graph. Leaves have no operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ValueKind {
    /// Constant bits (payload).
    Constant,
    /// A value the graph cannot model (lane-varying, unknown opcode, ...).
    Undefined,
    /// User-data SGPR `payload` at program entry.
    UserData,
    /// The program's entry address (S_GETPC_B64).
    ShaderBase,
    /// A join of the values of `phi_predecessors` (payload: owning block).
    Phi,
    Operation,
    /// `operands[0] ? operands[1] : operands[2]`.
    Select,
    /// V_READFIRSTLANE of `operands[0]` under the active mask `operands[1]`
    /// (payload: instruction pc).
    FirstLane,
    /// A 4-dword V# held in SGPRs (operands: the dwords).
    BufferHandle,
    /// A 64-bit address held in an SGPR (or VGPR) pair.
    AddressHandle,
    /// An 8-dword T#.
    ImageHandle,
    /// A 4-dword S#.
    SamplerHandle,
    /// A dword S_LOAD read through an address handle: operands `[handle,
    /// dynamic offset]`, payload: memory-table index.
    ScalarAddressWord,
    /// A dword S_BUFFER_LOAD (or uniform buffer load) through a buffer handle.
    ScalarBufferWord,
    /// A flattened resource-table slot the host fills per draw (payload: slot).
    ResourceTableWord,
    /// The high dword of a gfx10 LDS/scratch aperture (payload: the inline
    /// operand).
    MemoryAperture,
}

/// Operations a value node can apply to its operands. The runtime validator
/// accepts only the uniform subset; every operation still builds so a
/// rejection names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ScalarOp {
    None,
    ConvertU32F32,
    ConvertF32U32,
    Construct64,
    Extract64,
    BitFieldInsert,
    BitFieldUExtract,
    BitFieldSExtract,
    IAdd32,
    IAdd64,
    AddCarry32,
    ISub32,
    ISub64,
    IMul32,
    IMul64,
    UMin32,
    ShiftLeft32,
    ShiftLeft64,
    ShiftRightLogical32,
    ShiftRightLogical64,
    ShiftRightArithmetic32,
    ShiftRightArithmetic64,
    And32,
    And64,
    Or32,
    Xor32,
    Not32,
    ULessThan32,
    IEqual32,
    UGreaterThan32,
    INotEqual32,
    LogicalOr,
    LogicalAnd,
    LogicalXor,
    LogicalNot,
    FLessThanEqual,
    FGreaterThanEqual,
    FIsNan,
    FMul,
    FTrunc,
    UMax32,
    SMin32,
    SMax32,
    IAbs32,
    UMulHi32,
    SMulHi32,
    BitCount32,
    BitReverse32,
    FindLowestBit32,
    FindHighestBit32,
    ULessThanEqual32,
    UGreaterThanEqual32,
    SLessThan32,
    SLessThanEqual32,
    SGreaterThan32,
    SGreaterThanEqual32,
    IEqual64,
    INotEqual64,
    QuadMask32,
    FAdd,
    FSub,
    FMin,
    FMax,
    FLessThan,
    FGreaterThan,
    FEqual,
    FNotEqual,
    ConvertS32F32,
    ConvertF32S32,
}

impl ScalarOp {
    /// Operations the host can evaluate per draw: pure functions of their
    /// operands that are the same for every lane.
    pub fn is_uniform(self) -> bool {
        !matches!(self, ScalarOp::None)
    }
}

/// Index of a node in its [`ScalarValueGraph`](super::ScalarValueGraph).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub u32);

impl ValueId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// One node. Nodes are immutable once built, except that a phi's operands are
/// filled in while the graph converges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    pub kind: ValueKind,
    pub ty: ValueType,
    pub op: ScalarOp,
    /// Constant bits, user-data register, phi block, instruction pc, memory
    /// index or table slot, depending on `kind`.
    pub payload: u64,
    pub operands: Box<[ValueId]>,
    /// For a phi: the predecessor block of each operand (-1 = program entry).
    pub phi_predecessors: Box<[i32]>,
}

impl Node {
    pub fn is_constant(&self) -> bool {
        self.kind == ValueKind::Constant
    }

    pub fn is_undefined(&self) -> bool {
        self.kind == ValueKind::Undefined
    }

    pub fn constant_u32(&self) -> u32 {
        self.payload as u32
    }

    pub fn constant_bool(&self) -> bool {
        self.payload != 0
    }

    /// Memory-table index of a scalar read.
    pub fn memory_index(&self) -> usize {
        self.payload as usize
    }

    pub fn phi_block(&self) -> usize {
        self.payload as usize
    }

    pub fn is_operation(&self, op: ScalarOp) -> bool {
        self.kind == ValueKind::Operation && self.op == op
    }
}

fn float(bits: u64) -> f32 {
    f32::from_bits(bits as u32)
}

fn bits(value: f32) -> u64 {
    value.to_bits() as u64
}

/// IEEE-754 minNum / maxNum as `MathF.Min/Max`: NaN propagates.
fn fmin(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == b {
        // -0 < +0
        if a.is_sign_negative() {
            a
        } else {
            b
        }
    } else if a < b {
        a
    } else {
        b
    }
}

fn fmax(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == b {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else if a > b {
        a
    } else {
        b
    }
}

/// Evaluate one operation over concrete operands. Bool values are 0 or 1,
/// U32 values use the low dword, F32 values are the bits of the low dword.
/// `None` when the operation is not defined for these inputs.
pub fn evaluate_operation(op: ScalarOp, operands: &[u64]) -> Option<u64> {
    let at = |i: usize| operands.get(i).copied().unwrap_or(0);
    let (first, second, third, fourth) = (at(0), at(1), at(2), at(3));
    let (a, b) = (first as u32, second as u32);
    let flag = |c: bool| c as u64;
    use ScalarOp::*;
    Some(match op {
        ConvertF32U32 => bits(a as f32),
        ConvertU32F32 => {
            let v = float(first);
            if !v.is_finite() || v < 0.0 || v as f64 > u32::MAX as f64 {
                return Option::None;
            }
            v as u32 as u64
        }
        ConvertF32S32 => bits(a as i32 as f32),
        ConvertS32F32 => {
            let v = float(first);
            if !v.is_finite() || (v as f64) < i32::MIN as f64 || v as f64 > i32::MAX as f64 {
                return Option::None;
            }
            v as i32 as u32 as u64
        }
        Construct64 => a as u64 | ((b as u64) << 32),
        Extract64 => {
            if b >= 2 {
                return Option::None;
            }
            (first >> (b * 32)) as u32 as u64
        }
        AddCarry32 => {
            let sum = a as u64 + b as u64;
            (sum as u32 as u64) | ((sum >> 32) << 32)
        }
        IAdd32 => a.wrapping_add(b) as u64,
        IAdd64 => first.wrapping_add(second),
        ISub32 => a.wrapping_sub(b) as u64,
        ISub64 => first.wrapping_sub(second),
        IMul32 => a.wrapping_mul(b) as u64,
        IMul64 => first.wrapping_mul(second),
        UMin32 => a.min(b) as u64,
        UMax32 => a.max(b) as u64,
        SMin32 => (a as i32).min(b as i32) as u32 as u64,
        SMax32 => (a as i32).max(b as i32) as u32 as u64,
        IAbs32 => ((a as i32 as i64).abs() as u32) as u64,
        UMulHi32 => ((a as u64 * b as u64) >> 32) as u32 as u64,
        SMulHi32 => (((a as i32 as i64) * (b as i32 as i64)) >> 32) as u32 as u64,
        BitCount32 => a.count_ones() as u64,
        BitReverse32 => a.reverse_bits() as u64,
        FindLowestBit32 => {
            if a == 0 {
                u32::MAX as u64
            } else {
                a.trailing_zeros() as u64
            }
        }
        FindHighestBit32 => {
            if a == 0 {
                u32::MAX as u64
            } else {
                (31 - a.leading_zeros()) as u64
            }
        }
        ShiftLeft32 => (a << (b & 31)) as u64,
        ShiftLeft64 => first << (b & 63),
        ShiftRightLogical32 => (a >> (b & 31)) as u64,
        ShiftRightLogical64 => first >> (b & 63),
        ShiftRightArithmetic32 => ((a as i32) >> (b & 31)) as u32 as u64,
        ShiftRightArithmetic64 => ((first as i64) >> (b & 63)) as u64,
        And32 => (a & b) as u64,
        And64 => first & second,
        Or32 => (a | b) as u64,
        Xor32 => (a ^ b) as u64,
        Not32 => (!a) as u64,
        QuadMask32 => (((a | (a >> 1) | (a >> 2) | (a >> 3)) & 0x1111_1111).wrapping_mul(0xF)) as u64,
        BitFieldUExtract | BitFieldSExtract => {
            let (offset, width) = (b, third as u32);
            if offset > 32 || width > 32 - offset {
                return Option::None;
            }
            if width == 0 {
                return Some(0);
            }
            let mask = if width == 32 { u32::MAX } else { (1u32 << width) - 1 };
            let mut v = a.checked_shr(offset).unwrap_or(0) & mask;
            if op == BitFieldSExtract && width < 32 && v & (1 << (width - 1)) != 0 {
                v |= !mask;
            }
            v as u64
        }
        BitFieldInsert => {
            let (offset, width) = (third as u32, fourth as u32);
            if offset > 32 || width > 32 - offset {
                return Option::None;
            }
            if width == 0 {
                return Some(a as u64);
            }
            let mask = if width == 32 { u32::MAX } else { ((1u32 << width) - 1) << offset };
            ((a & !mask) | (b.checked_shl(offset).unwrap_or(0) & mask)) as u64
        }
        ULessThan32 => flag(a < b),
        ULessThanEqual32 => flag(a <= b),
        UGreaterThan32 => flag(a > b),
        UGreaterThanEqual32 => flag(a >= b),
        SLessThan32 => flag((a as i32) < (b as i32)),
        SLessThanEqual32 => flag((a as i32) <= (b as i32)),
        SGreaterThan32 => flag((a as i32) > (b as i32)),
        SGreaterThanEqual32 => flag((a as i32) >= (b as i32)),
        IEqual32 => flag(a == b),
        INotEqual32 => flag(a != b),
        IEqual64 => flag(first == second),
        INotEqual64 => flag(first != second),
        LogicalAnd => flag(first != 0 && second != 0),
        LogicalOr => flag(first != 0 || second != 0),
        LogicalXor => flag((first != 0) != (second != 0)),
        LogicalNot => flag(first == 0),
        FMul => bits(float(first) * float(second)),
        FAdd => bits(float(first) + float(second)),
        FSub => bits(float(first) - float(second)),
        FMin => bits(fmin(float(first), float(second))),
        FMax => bits(fmax(float(first), float(second))),
        FTrunc => bits(float(first).trunc()),
        FIsNan => flag(float(first).is_nan()),
        FLessThanEqual => flag(float(first) <= float(second)),
        FGreaterThanEqual => flag(float(first) >= float(second)),
        FLessThan => flag(float(first) < float(second)),
        FGreaterThan => flag(float(first) > float(second)),
        FEqual => flag(float(first) == float(second)),
        FNotEqual => flag(float(first) != float(second)),
        None => return Option::None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantics() {
        use ScalarOp::*;
        assert_eq!(evaluate_operation(AddCarry32, &[u32::MAX as u64, 2]), Some((1u64 << 32) | 1));
        assert_eq!(evaluate_operation(Extract64, &[0x1111_2222_3333_4444, 1]), Some(0x1111_2222));
        assert_eq!(evaluate_operation(Extract64, &[0, 2]), Option::None);
        assert_eq!(evaluate_operation(BitFieldUExtract, &[0xABCD_1234, 16, 8]), Some(0xCD));
        assert_eq!(evaluate_operation(BitFieldSExtract, &[0x0000_8000, 8, 8]), Some(0xFFFF_FF80));
        assert_eq!(evaluate_operation(BitFieldUExtract, &[1, 20, 16]), Option::None);
        assert_eq!(evaluate_operation(BitFieldInsert, &[0, u32::MAX as u64, 4, 8]), Some(0xFF0));
        assert_eq!(evaluate_operation(BitFieldInsert, &[0, u32::MAX as u64, 0, 32]), Some(u32::MAX as u64));
        assert_eq!(evaluate_operation(FindLowestBit32, &[0]), Some(u32::MAX as u64));
        assert_eq!(evaluate_operation(FindLowestBit32, &[0x80]), Some(7));
        assert_eq!(evaluate_operation(FindHighestBit32, &[0x80]), Some(7));
        assert_eq!(evaluate_operation(QuadMask32, &[0x0000_0102]), Some(0xF0F));
        assert_eq!(evaluate_operation(ShiftRightArithmetic32, &[0x8000_0000, 4]), Some(0xF800_0000));
        assert_eq!(evaluate_operation(SMulHi32, &[(-2i32) as u32 as u64, 3]), Some(u32::MAX as u64));
        assert_eq!(evaluate_operation(ConvertU32F32, &[(-1.0f32).to_bits() as u64]), Option::None);
        assert_eq!(evaluate_operation(ConvertU32F32, &[(7.9f32).to_bits() as u64]), Some(7));
        assert_eq!(evaluate_operation(ConvertF32U32, &[3]), Some(3.0f32.to_bits() as u64));
        assert_eq!(evaluate_operation(IAbs32, &[i32::MIN as u32 as u64]), Some(0x8000_0000));
        assert_eq!(evaluate_operation(FMin, &[1.0f32.to_bits() as u64, (-2.0f32).to_bits() as u64]), Some((-2.0f32).to_bits() as u64));
        assert_eq!(evaluate_operation(None, &[]), Option::None);
    }
}
