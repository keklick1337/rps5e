// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Attaching a resource plan to IR memory operations.
//!
//! Every IR memory operation carries the guest pc of the instruction it
//! came from (and, for scalar loads, the dword index); the resource plan
//! keys its [`ResourceUse`] records the same way, so the back-end can find
//! the binding of each descriptor operand.

use crate::resources::{ResourceOperand, ResourceUse};
use crate::ssa::*;
use std::collections::HashMap;

/// Plan key of an IR memory operation: `(pc, component)`.
pub fn memory_key(op: Op) -> Option<(u32, u32)> {
    Some(match op {
        Op::ScalarLoad { pc, dword } | Op::ScalarBufferLoad { pc, dword } => (pc, dword as u32),
        Op::BufferLoad { attr, .. } | Op::BufferStore { attr, .. } | Op::BufferAtomic { attr, .. } => (attr.pc, 0),
        Op::GlobalLoad { pc, .. } | Op::GlobalStore { pc, .. } | Op::GlobalAtomic { pc, .. } => (pc, 0),
        Op::ScratchLoad { pc, .. } | Op::ScratchStore { pc, .. } => (pc, 0),
        Op::Image(a) => (a.pc, 0),
        _ => return None,
    })
}

/// Resource uses of every memory operation in `func`, by value. Scalar
/// loads match on `(pc, dword)`; other operations take every use of their pc
/// (resource and, for sampled images, sampler).
pub fn attach_resource_uses(func: &Function, uses: &[ResourceUse]) -> HashMap<Value, Vec<ResourceUse>> {
    let mut by_key: HashMap<(u32, u32), Vec<ResourceUse>> = HashMap::new();
    let mut by_pc: HashMap<u32, Vec<ResourceUse>> = HashMap::new();
    for u in uses {
        by_key.entry((u.pc, u.component)).or_default().push(*u);
        by_pc.entry(u.pc).or_default().push(*u);
    }
    let mut out = HashMap::new();
    for &b in &func.order {
        for &v in &func.block(b).insts {
            let op = func.op(v);
            let Some((pc, component)) = memory_key(op) else { continue };
            let found = match op {
                Op::ScalarLoad { .. } | Op::ScalarBufferLoad { .. } => by_key.get(&(pc, component)).cloned(),
                _ => by_pc.get(&pc).cloned(),
            };
            if let Some(list) = found {
                out.insert(v, list);
            }
        }
    }
    out
}

/// The resource (not sampler) use of a value, if planned.
pub fn resource_use(map: &HashMap<Value, Vec<ResourceUse>>, v: Value) -> Option<ResourceUse> {
    map.get(&v)?.iter().find(|u| u.operand == ResourceOperand::Resource).copied()
}
