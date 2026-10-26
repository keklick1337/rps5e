// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Scalar memory: wide s_buffer_load / s_load forms with out-of-range
//! offsets, and a decoupled look-back scan across workgroups, against
//! hardware-measured reference rows.

include!("vectors/scalar_common.rs");

mod wide_loads {
    use super::*;
    include!("vectors/scalar_wide_loads.rs");

    /// s_buffer_load_dwordx4 / x16 with SGPR and immediate offsets (some
    /// past num_records), an s_buffer_load_dword through a descriptor built
    /// from lane data, and s_load_dwordx16 with an unaligned immediate. The
    /// input sits at the start of a zeroed 64 KiB block.
    fn run() -> Vec<u32> {
        let mut input: Vec<u32> = ROWS.iter().flatten().copied().collect();
        let in_bytes = (input.len() * 4) as u32;
        input.resize(65536 / 4, 0);
        let out_words = 32 * 96;
        let ud = user_data(raw_desc(IN_BASE, in_bytes), raw_desc(OUT_BASE, (out_words * 4) as u32), 8);
        Dispatch::new(&CODE, 32, 32, ud).buffer(IN_BASE, input).output(OUT_BASE, out_words).run().unwrap().remove(0)
    }

    #[test]
    fn wide_loads() {
        assert_clean(check_rows("scalar wide loads", &run(), 96, &EXPECTED, &NAMES, &|_| true));
    }

    /// Every lane / column whose SGPR offset (s8 = row word 0, s9 = row
    /// word 1) does not make imm + soffset wrap past 2^32.
    #[test]
    fn wide_loads_without_wrapping_offsets() {
        let out = run();
        let mut failures = Vec::new();
        for (lane, row) in EXPECTED.iter().enumerate() {
            for (i, &want) in row.iter().enumerate() {
                let Some(name) = NAMES[i] else { continue };
                let wraps = |s: u32| s >= 0xFFF0_0000;
                if (name.contains(" s8") && wraps(ROWS[lane][0])) || (name.contains(" s9") && wraps(ROWS[lane][1])) {
                    continue;
                }
                let got = out[lane * 96 + i];
                if got != want {
                    failures.push(format!("scalar wide loads: lane {lane} {name} is {got:#010x}, expected {want:#010x}"));
                }
            }
        }
        assert_clean(failures);
    }
}

mod look_back_scan {
    use super::*;
    include!("vectors/scalar_look_back_scan.rs");

    const LANES: u32 = 64;
    const BLOCKS: usize = 4096;
    const COUNTER: u64 = 0x1_0000_0000;
    const AGGREGATES: u64 = 0x1_0001_0000;
    const PAIRS: u64 = 0x1_0002_0000;
    const PREFIXES: u64 = 0x1_0004_0000;

    fn value(index: u32, seed: u32) -> u32 {
        (index.wrapping_mul(2654435761).wrapping_add(seed.wrapping_mul(40503)) >> 13) % 1000
    }

    fn desc(base: u64, bytes: u32) -> [u32; 4] {
        [base as u32, (base >> 32) as u32 & 0xFFFF, bytes, 0x3101_6FAC]
    }

    fn user() -> Vec<u32> {
        let mut u = Vec::new();
        u.extend(desc(COUNTER, 16));
        u.extend(desc(AGGREGATES, BLOCKS as u32 * 4));
        u.extend(desc(PAIRS, BLOCKS as u32 * 8));
        u.extend(desc(PREFIXES, BLOCKS as u32 * 4));
        u
    }

    struct State {
        counter: Vec<u32>,
        aggregates: Vec<u32>,
        pairs: Vec<u32>,
        prefixes: Vec<u32>,
    }

    /// Run `groups` single-wave64 workgroups one after another on shared
    /// memory (the scan takes its block index from an atomic counter).
    fn run(state: &mut State, groups: usize, optimize: bool) -> Result<(), String> {
        let run = ComputeRun::new(SCAN_CODE.to_vec(), 64, LANES, user());
        let f = run.compile(optimize)?;
        let mut memory = Memory::default();
        memory.write_words(COUNTER, &state.counter);
        memory.write_words(AGGREGATES, &state.aggregates);
        memory.write_words(PAIRS, &state.pairs);
        memory.write_words(PREFIXES, &state.prefixes);
        for g in 0..groups {
            let mut m = Machine::new(64, LANES);
            m.user_data = user();
            m.memory = std::mem::take(&mut memory);
            m.system.insert(SystemValue::LocalInvocationId(0), (0..LANES).collect());
            m.system.insert(SystemValue::LocalInvocationId(1), vec![0; LANES as usize]);
            m.system.insert(SystemValue::LocalInvocationId(2), vec![0; LANES as usize]);
            m.run(&f).map_err(|e| format!("group {g}: {e}"))?;
            memory = m.memory;
        }
        state.counter = memory.read_words(COUNTER, 4);
        state.pairs = memory.read_words(PAIRS, BLOCKS * 2);
        state.prefixes = memory.read_words(PREFIXES, BLOCKS);
        Ok(())
    }

    /// One workgroup whose predecessors already published their
    /// (flag, value) pairs: it must take block `block`, find the inclusive
    /// prefix by looking back and publish its own.
    #[test]
    fn look_back_single_block() {
        let mut failures = Vec::new();
        for optimize in [false, true] {
            for block in [0u32, 1, 2, 63, 64, 1000, BLOCKS as u32 - 1] {
                let aggregates: Vec<u32> = (0..BLOCKS as u32).map(|b| value(b, block)).collect();
                let mut pairs = vec![0u32; BLOCKS * 2];
                let mut prefix = 0u32;
                for b in 0..block as usize {
                    prefix += aggregates[b];
                    pairs[2 * b] = if b == 0 { 2 } else { 1 };
                    pairs[2 * b + 1] = if b == 0 { prefix } else { aggregates[b] };
                }
                let mut s = State { counter: vec![block, 0, 0, 0], aggregates: aggregates.clone(), pairs, prefixes: vec![UNTOUCHED; BLOCKS] };
                if let Err(e) = run(&mut s, 1, optimize) {
                    failures.push(format!("block {block}: {e}"));
                    continue;
                }
                let b = block as usize;
                let name = format!("look-back from block {block} (optimize {optimize})");
                if s.counter[0] != block + 1 {
                    failures.push(format!("{name}: the counter is {}", s.counter[0]));
                }
                if s.prefixes[b] != prefix {
                    failures.push(format!("{name}: its prefix is {}, expected {prefix}", s.prefixes[b]));
                }
                if s.pairs[2 * b] != 2 {
                    failures.push(format!("{name}: its flag is {}, expected 2", s.pairs[2 * b]));
                }
                if s.pairs[2 * b + 1] != prefix + aggregates[b] {
                    failures.push(format!("{name}: it published {}, expected {}", s.pairs[2 * b + 1], prefix + aggregates[b]));
                }
            }
        }
        assert_clean(failures);
    }

    /// A whole 4096-block scan (blocks run in counter order).
    #[test]
    fn look_back_whole_scan() {
        let mut failures = Vec::new();
        for seed in 0..3u32 {
            let aggregates: Vec<u32> = (0..BLOCKS as u32).map(|b| value(b, seed)).collect();
            let mut s = State { counter: vec![0; 4], aggregates: aggregates.clone(), pairs: vec![0; BLOCKS * 2], prefixes: vec![UNTOUCHED; BLOCKS] };
            if let Err(e) = run(&mut s, BLOCKS, true) {
                failures.push(format!("seed {seed}: {e}"));
                continue;
            }
            let mut running = 0u32;
            for b in 0..BLOCKS {
                if s.prefixes[b] != running {
                    failures.push(format!("whole scan {seed}: block {b}'s prefix is {}, expected {running}", s.prefixes[b]));
                }
                running += aggregates[b];
                if s.pairs[2 * b] != 2 || s.pairs[2 * b + 1] != running {
                    failures.push(format!("whole scan {seed}: block {b} published {{{}, {}}}, expected {{2, {running}}}", s.pairs[2 * b], s.pairs[2 * b + 1]));
                }
            }
        }
        assert_clean(failures);
    }
}
