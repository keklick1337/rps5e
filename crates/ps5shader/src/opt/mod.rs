// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Optimization passes over SSA IR:
//!
//! * [`fold`] — ConstantFolder (evaluation shared with the interpreter,
//!   algebraic identities, inverse-ballot of a ballot) and CSE.
//! * [`dce`] — DeadCodeEliminator.
//! * [`lanes`] — ReadLaneEliminator, MaskedSelectEliminator, LaneMaskProjection,
//!   SharedMemoryBarrierInserter.
//! * [`split_phis`] — SplitDescriptorPhis.
//! * [`uniform`] — uniformity analysis.
//! * [`wave64_masks`] — Wave64LaneMaskProjection (not part of [`optimize`]:
//!   the back-end runs it for wave64 programs it splits over two host
//!   subgroups).
//! * [`exec_runs`] — ExecRunGrouping (not part of [`optimize`]: it creates
//!   the only divergent branches, so it runs after it, in the back-end).

pub mod dce;
pub mod denorm;
pub mod exec_runs;
pub mod fold;
pub mod lanes;
pub mod split_phis;
pub mod uniform;
pub mod wave64_masks;

use crate::program::CodeSource;
use crate::ssa::Function;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OptStats {
    pub folded: usize,
    pub common: usize,
    pub removed: usize,
    pub read_lanes: usize,
    pub masked_selects: usize,
    pub projected_masks: usize,
    pub code_loads: usize,
    pub split_descriptor_phis: usize,
    pub lds_barriers: usize,
    /// f32 denormal flushes of values that are never denormals, removed
    /// ([`denorm`]).
    pub elided_flushes: usize,
    pub rounds: usize,
    /// Wall time per pass (ns): [`PASSES`].
    pub pass_ns: [u64; PASSES.len()],
}

/// The passes timed in [`OptStats::pass_ns`].
pub const PASSES: [&str; 10] = ["fold", "cse", "code_loads", "read_lanes", "project_masks", "masked_selects", "dce", "split_phis", "lds_barriers", "flushes"];

/// Run the pipeline to a fixed point (bounded), then insert LDS barriers.
pub fn optimize(func: &mut Function, code: Option<&dyn CodeSource>) -> OptStats {
    optimize_with(func, code, false)
}

/// [`optimize`]; `elide_flushes` also removes the f32 denormal flushes of
/// values that are never denormals ([`denorm`]) in every round.
pub fn optimize_with(func: &mut Function, code: Option<&dyn CodeSource>, elide_flushes: bool) -> OptStats {
    let mut s = OptStats::default();
    let mut t = std::time::Instant::now();
    let mut lap = |s: &mut OptStats, k: usize| {
        let now = std::time::Instant::now();
        s.pass_ns[k] += (now - t).as_nanos() as u64;
        t = now;
    };
    for round in 0..16 {
        s.rounds = round + 1;
        let mut changes = 0;
        let f = fold::fold(func);
        s.folded += f;
        changes += f;
        lap(&mut s, 0);
        let c = fold::cse(func);
        s.common += c;
        changes += c;
        lap(&mut s, 1);
        if let Some(code) = code {
            let k = fold::fold_code_loads(func, code, crate::program::MAX_DECLARED_SHADER_SIZE as u64 * 2);
            s.code_loads += k;
            changes += k;
        }
        lap(&mut s, 2);
        let r = lanes::eliminate_read_lanes(func);
        s.read_lanes += r;
        changes += r;
        lap(&mut s, 3);
        let p = lanes::project_lane_masks(func);
        s.projected_masks += p;
        changes += p;
        lap(&mut s, 4);
        let m = lanes::eliminate_masked_selects(func);
        s.masked_selects += m;
        changes += m;
        lap(&mut s, 5);
        let d = dce::dce(func);
        s.removed += d;
        changes += d;
        lap(&mut s, 6);
        if elide_flushes {
            let e = denorm::elide_flushes(func);
            s.elided_flushes += e;
            changes += e;
        }
        lap(&mut s, 9);
        if changes == 0 {
            break;
        }
    }
    s.split_descriptor_phis = split_phis::split_descriptor_phis(func);
    if s.split_descriptor_phis > 0 {
        fold::fold(func);
        dce::dce(func);
    }
    lap(&mut s, 7);
    s.lds_barriers = lanes::insert_lds_barriers(func);
    lap(&mut s, 8);
    s
}
