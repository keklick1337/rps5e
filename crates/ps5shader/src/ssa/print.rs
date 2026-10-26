// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Text form of a function.

use super::func::*;
use super::op::Op;
use super::types::Type;
use std::fmt::{self, Write};

/// Operand text: constants inline, others as `%n`.
pub fn operand(func: &Function, v: Value) -> String {
    let d = func.value(v);
    match d.op {
        Op::Const(c) => match d.ty {
            Type::BOOL => (c != 0).to_string(),
            Type::F32 => format!("{}f", f32::from_bits(c as u32)),
            Type::F64 => format!("{}d", f64::from_bits(c)),
            ty => format!("{c:#x}:{ty}"),
        },
        Op::Undef => format!("undef:{}", d.ty),
        _ => format!("%{}", v.0),
    }
}

/// One instruction line (without indentation).
pub fn instruction(func: &Function, v: Value) -> String {
    let d = func.value(v);
    let mut s = String::new();
    if d.ty != Type::Void {
        let _ = write!(s, "%{}:{} = ", v.0, d.ty);
    }
    s.push_str(&d.op.name());
    for a in &d.args {
        s.push(' ');
        s.push_str(&operand(func, *a));
    }
    s
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "function wave{} {:?} code@{:#x}", self.wave_size, self.stage, self.code_address)?;
        for &b in &self.order {
            let bd = self.block(b);
            write!(f, "b{}:", b.0)?;
            if !bd.preds.is_empty() {
                write!(f, " ; preds")?;
                for p in &bd.preds {
                    write!(f, " b{}", p.0)?;
                }
            }
            writeln!(f)?;
            for &v in &bd.insts {
                writeln!(f, "  {}", instruction(self, v))?;
            }
            match bd.merge {
                Merge::None => {}
                Merge::Selection { merge } => writeln!(f, "  selection_merge b{}", merge.0)?,
                Merge::Loop { merge, continue_block } => {
                    writeln!(f, "  loop_merge b{} continue b{}", merge.0, continue_block.0)?
                }
            }
            match &bd.term {
                Terminator::None => writeln!(f, "  <unterminated>")?,
                Terminator::Branch(t) => writeln!(f, "  branch b{}", t.0)?,
                Terminator::CondBranch { cond, then_block, else_block } => {
                    writeln!(f, "  branch_cond {} b{} b{}", operand(self, *cond), then_block.0, else_block.0)?
                }
                Terminator::Return => writeln!(f, "  return")?,
                Terminator::Unreachable => writeln!(f, "  unreachable")?,
            }
        }
        Ok(())
    }
}
