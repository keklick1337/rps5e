// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! GDS ds_append / ds_consume. The hardware-measured reference rows run
//! this program against GDS counters set through PM4, which the CPU
//! interpreter does not model; the port checks that the GDS counters
//! translate to wave-wide GDS append / consume operations.

include!("vectors/mem_common.rs");

const UNTOUCHED_APPEND: u32 = 0x5eed_0001;
const UNTOUCHED_CONSUME: u32 = 0x5eed_0002;
const ACTIVE_LANES_LITERAL: usize = 12;
const OUT_BASE: u64 = 0x2_0000_0000;

const CODE: [u32; 28] = [
    0x34060082, 0x7e0c02ff, UNTOUCHED_APPEND, 0x7e0e02ff, UNTOUCHED_CONSUME, 0xbefc03ff, 0x0000ffff, 0xd8fa0010,
    0x04000000, 0xd8f60014, 0x05000000, 0x7da800ff, 0, 0xd8fa0018, 0x06000000, 0xd8f6001c,
    0x07000000, 0xbefe04c1, 0xbf8cc07f, 0xe0702000, 0x80000403, 0xe0702004, 0x80000503, 0xe0702008,
    0x80000603, 0xe070200c, 0x80000703, 0xbf810000,
];

/// (threads, wave size, active lanes) of the reference workgroups.
const WORKGROUPS: [(u32, u32, u32); 4] = [(32, 32, 32), (32, 32, 20), (64, 32, 40), (64, 64, 40)];

#[test]
fn gds_append_consume_translates() {
    for (threads, wave, active) in WORKGROUPS {
        let mut code = CODE;
        code[ACTIVE_LANES_LITERAL] = active;
        let mut user = Vec::new();
        put_desc(&mut user, 0, vdesc_full(OUT_BASE, 4, 64 * 4, 0x1101_6fac));
        // One wave of the workgroup is enough to reach the GDS instructions.
        let run = ComputeRun::new(code.to_vec(), wave, threads.min(wave), user);
        for optimize in [false, true] {
            let f = run.compile(optimize).unwrap_or_else(|e| panic!("wave{wave} {threads} threads, {active} active: {e}"));
            let ops: Vec<ps5shader::ssa::Op> = f.order.iter().flat_map(|&b| f.block(b).insts.iter().map(|&v| f.op(v))).collect();
            let appends = ops.iter().filter(|o| matches!(o, ps5shader::ssa::Op::GdsAppend { consume: false })).count();
            let consumes = ops.iter().filter(|o| matches!(o, ps5shader::ssa::Op::GdsAppend { consume: true })).count();
            assert_eq!((appends, consumes), (2, 2), "wave{wave} {threads} threads: GDS counters");
        }
    }
}
