// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Uniformity: values equal in every invocation of the wave. Control flow
//! is always uniform in this IR, so a phi of uniform values is uniform.

use crate::ssa::*;

/// Per value: true when the value is the same in every invocation.
pub fn uniform_values(func: &Function) -> Vec<bool> {
    let n = func.values.len();
    // Optimistic fixed point: start uniform, demote.
    let mut uniform = vec![true; n];
    let source_varying = |op: Op| match op {
        Op::LaneId | Op::Shuffle | Op::InverseBallot | Op::InterpParam { .. } => true,
        Op::SystemValue(s) => !matches!(s, SystemValue::WorkgroupId(_) | SystemValue::SubgroupId | SystemValue::Custom(_)),
        Op::BufferLoad { .. }
        | Op::BufferAtomic { .. }
        | Op::GlobalLoad { .. }
        | Op::GlobalAtomic { .. }
        | Op::ScratchLoad { .. }
        | Op::LdsLoad { .. }
        | Op::LdsAtomic { .. }
        | Op::Image(_) => true,
        _ => false,
    };
    let always_uniform = |op: Op| matches!(op, Op::Ballot | Op::ReadLane | Op::LdsAppend { .. } | Op::GdsAppend { .. });
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &func.order {
            for &v in &func.block(b).insts {
                if !uniform[v.index()] {
                    continue;
                }
                let d = func.value(v);
                let u = if always_uniform(d.op) {
                    true
                } else if source_varying(d.op) {
                    false
                } else {
                    d.args.iter().all(|a| uniform[a.index()])
                };
                if !u {
                    uniform[v.index()] = false;
                    changed = true;
                }
            }
        }
    }
    uniform
}
