// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! IR value types.

use std::fmt;

/// Scalar component type. Integers carry no signedness: operations say how
/// they interpret their operands (as in SPIR-V).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Scalar {
    Bool,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,
}

impl Scalar {
    pub fn bits(self) -> u32 {
        match self {
            Scalar::Bool => 1,
            Scalar::U16 | Scalar::F16 => 16,
            Scalar::U32 | Scalar::F32 => 32,
            Scalar::U64 | Scalar::F64 => 64,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, Scalar::F16 | Scalar::F32 | Scalar::F64)
    }

    pub fn is_int(self) -> bool {
        matches!(self, Scalar::U16 | Scalar::U32 | Scalar::U64)
    }

    /// Mask of the value bits in a 64-bit container.
    pub fn mask(self) -> u64 {
        match self.bits() {
            64 => u64::MAX,
            b => (1u64 << b) - 1,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Scalar::Bool => "bool",
            Scalar::U16 => "u16",
            Scalar::U32 => "u32",
            Scalar::U64 => "u64",
            Scalar::F16 => "f16",
            Scalar::F32 => "f32",
            Scalar::F64 => "f64",
        }
    }
}

/// A value type: void, a scalar or a 2..4 component vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Type {
    Void,
    Scalar(Scalar),
    Vector(Scalar, u8),
}

impl Type {
    pub const VOID: Type = Type::Void;
    pub const BOOL: Type = Type::Scalar(Scalar::Bool);
    pub const U16: Type = Type::Scalar(Scalar::U16);
    pub const U32: Type = Type::Scalar(Scalar::U32);
    pub const U64: Type = Type::Scalar(Scalar::U64);
    pub const F16: Type = Type::Scalar(Scalar::F16);
    pub const F32: Type = Type::Scalar(Scalar::F32);
    pub const F64: Type = Type::Scalar(Scalar::F64);
    pub const U32X2: Type = Type::Vector(Scalar::U32, 2);
    pub const U32X3: Type = Type::Vector(Scalar::U32, 3);
    pub const U32X4: Type = Type::Vector(Scalar::U32, 4);
    pub const F32X2: Type = Type::Vector(Scalar::F32, 2);
    pub const F32X4: Type = Type::Vector(Scalar::F32, 4);

    /// `n` components of `s` (a scalar when `n == 1`).
    pub fn vec(s: Scalar, n: u32) -> Type {
        assert!((1..=4).contains(&n), "vector width {n}");
        if n == 1 {
            Type::Scalar(s)
        } else {
            Type::Vector(s, n as u8)
        }
    }

    pub fn u32xn(n: u32) -> Type {
        Type::vec(Scalar::U32, n)
    }

    pub fn scalar(self) -> Option<Scalar> {
        match self {
            Type::Void => None,
            Type::Scalar(s) | Type::Vector(s, _) => Some(s),
        }
    }

    /// Component scalar; panics on void.
    pub fn component(self) -> Scalar {
        self.scalar().expect("void type has no component")
    }

    pub fn components(self) -> u32 {
        match self {
            Type::Void => 0,
            Type::Scalar(_) => 1,
            Type::Vector(_, n) => n as u32,
        }
    }

    pub fn is_vector(self) -> bool {
        matches!(self, Type::Vector(..))
    }

    pub fn is_bool(self) -> bool {
        self == Type::BOOL
    }

    pub fn is_float(self) -> bool {
        self.scalar().is_some_and(Scalar::is_float)
    }

    pub fn is_int(self) -> bool {
        self.scalar().is_some_and(Scalar::is_int)
    }

    /// Total bit width (components * component bits).
    pub fn bits(self) -> u32 {
        self.scalar().map_or(0, |s| s.bits() * self.components())
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Void => f.write_str("void"),
            Type::Scalar(s) => f.write_str(s.name()),
            Type::Vector(s, n) => write!(f, "{}x{n}", s.name()),
        }
    }
}

impl From<Scalar> for Type {
    fn from(s: Scalar) -> Type {
        Type::Scalar(s)
    }
}
