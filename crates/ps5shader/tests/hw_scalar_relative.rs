// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! M0-relative register access (s_movrel*, v_movrel*, v_swaprel / v_swap)
//! against hardware-measured reference rows.

include!("vectors/scalar_common.rs");

const THREADS: usize = 32;

fn run(code: &[u32], input: Vec<u32>, results: usize) -> Vec<u32> {
    let out_words = THREADS * results;
    let ud = user_data(raw_desc(IN_BASE, (input.len() * 4) as u32), raw_desc(OUT_BASE, (out_words * 4) as u32), 8);
    Dispatch::new(code, 32, THREADS as u32, ud).buffer(IN_BASE, input).output(OUT_BASE, out_words).run().unwrap().remove(0)
}

mod scalar {
    use super::*;
    include!("vectors/scalar_relative_indexing.rs");

    /// s_movrels_b32/_b64, s_movreld_b32/_b64, s_movrelsd_2_b32 with M0
    /// offsets, including an out-of-range s_movreld.
    #[test]
    fn scalar_relative_indexing() {
        let out = run(&CODE, vec![0; THREADS * 4], 16);
        let mut failures = Vec::new();
        for tid in 0..THREADS {
            for i in 0..16 {
                let got = out[tid * 16 + i];
                if got != EXPECTED[i] {
                    failures.push(format!("scalar relative indexing: lane {tid} {} is {got:#010x}, expected {:#010x}", NAMES[i].unwrap(), EXPECTED[i]));
                }
            }
        }
        assert_clean(failures);
    }
}

mod vector {
    use super::*;
    include!("vectors/scalar_vector_relative_indexing.rs");

    /// v_movrelsd_b32, v_movrelsd_2_b32, v_swaprel_b32, v_swap_b32 and an
    /// out-of-range VOP3 v_movrelsd.
    fn input() -> Vec<u32> {
        let mut input = vec![0u32; THREADS * 4];
        for tid in 0..THREADS {
            input[tid * 4] = 0x1000 * (tid as u32 + 1);
        }
        input
    }

    #[test]
    fn vector_relative_indexing() {
        let out = run(&CODE, input(), 16);
        assert_clean(check_rows("relative indexing", &out, 16, &EXPECTED, &NAMES, &|_| true));
    }

}

/// The relative-addressing window reaches the stage's input VGPRs the
/// program never names (the reference driver's RelativeStageInputs
/// compute case): m0 = 2; v_movrels_b32 v1, v0 reads thread z from v2,
/// stored at (x + z * 8) * 4.
#[test]
fn vector_relative_reads_reach_stage_inputs() {
    const CODE: [u32; 9] = [0xbefc0382, 0x7e028700, 0x34020283, 0x4a000300, 0x34000082, 0x7e028700, 0xe0701000, 0x80000100, 0xbf810000];
    let mut d = Dispatch::new(&CODE, 32, 16, raw_desc(OUT_BASE, 64).to_vec()).output(OUT_BASE, 16);
    d.local = [8, 1, 2];
    d.thread_id_vgprs = 3;
    let out = d.run().unwrap().remove(0);
    let want: Vec<u32> = (0..16).map(|i| i / 8).collect();
    assert_eq!(out, want, "relative reads of the implicit thread z input");
}

/// Out-of-range relative VGPR accesses, measured on RDNA2 (gfx1035): M0 is
/// unsigned, so an index at or past the relative-addressing window (which
/// ends at the program's last named register) reads v0 and drops its
/// write; v_swaprel_b32 and v_movrelsd_2_b32 write neither side when either
/// index is out of range; registers never written read 0. Wave32 and
/// wave64.
mod vector_out_of_range {
    use super::*;

    const RESULTS: usize = 32;
    const CHECKED: usize = 19;
    const CODE: [u32; 108] = [
        0x34020084, 0x34060087, 0xe0301000, 0x80000401, 0x7e140280, 0x7e160280, 0x7e180280, 0x7e1a0280,
        0x7e1c0280, 0x7e1e0280, 0x7e200280, 0x7e220280, 0x7e240280, 0x7e260280, 0x7e280280, 0xbf8c3f70,
        0x7e3c0304, 0xd525001f, 0x00010304, 0xd5250020, 0x00010504, 0xd5250021, 0x00010704, 0xd5250022,
        0x00010904, 0xd5250023, 0x00010b04, 0xd5250024, 0x00010d04, 0xd5250025, 0x00010f04, 0xbefc0382,
        0x7e3e891e, 0x7e140321, 0xbefc03ff, 0xfc03fc02, 0x7e3e911e, 0x7e160322, 0x7e18031f, 0x4a4008a0,
        0x4a4208b0, 0xbefc03ff, 0x00010003, 0x7e3ed11e, 0x7e1a0320, 0x7e1c0321, 0x7e1e031e, 0x7e48cb25,
        0x7e200324, 0x7e220325, 0xbefc03ff, 0x00020002, 0x7e44d122, 0x7e240324, 0x7e260323, 0xd5c40023,
        0x0000011f, 0x7e280323, 0xe0701000, 0x80010a03, 0xe0701004, 0x80010b03, 0xe0701008, 0x80010c03,
        0xe070100c, 0x80010d03, 0xe0701010, 0x80010e03, 0xe0701014, 0x80010f03, 0xe0701018, 0x80011003,
        0xe070101c, 0x80011103, 0xe0701020, 0x80011203, 0xe0701024, 0x80011303, 0xe0701028, 0x80011403,
        0xbefc03c1, 0x7e2a8720, 0xe070102c, 0x80011503, 0x7e428504, 0xe0701030, 0x80012003, 0x7e468922,
        0xe0701034, 0x80012203, 0xe0701038, 0x80012103, 0xbefc03ff, 0x000103ff, 0x7e3ed120, 0xe070103c,
        0x80012003, 0x7e48911e, 0xe0701040, 0x80012503, 0x7e2c0327, 0xe0701044, 0x80011603, 0xbefc0388,
        0x7e2e871e, 0xe0701048, 0x80011703, 0xbf810000,
    ];
    const NAMES: [&str; CHECKED] = [
        "movrelsd", "swapped_2", "sd2_untouched", "swaprel_dst", "swaprel_src", "swaprel_other", "swap_dst", "swap_src",
        "swaprel_same", "swaprel_same_other", "movrelsd_e64_dst", "movrels_out_of_range", "movreld_out_of_range_neighbour",
        "movrelsd_out_of_range_source", "movrelsd_out_of_range_other", "swaprel_one_side_out_of_range",
        "movrelsd_2_source_out_of_range", "unwritten_direct", "unwritten_relative",
    ];

    fn expected(tid: u32) -> [u32; CHECKED] {
        let x = 0x1000 * (tid + 1);
        [x + 2, x + 2, x + 1, x + 0x30, x + 0x20, x, x + 7, x + 6, x + 7, x + 5, x + 5, tid, x + 0x30, x + 2, x + 0x20, x + 0x30, tid, 0, 0]
    }

    fn check(wave: u32) {
        let threads = wave as usize;
        let mut input = vec![0u32; threads * 4];
        for tid in 0..threads {
            input[tid * 4] = 0x1000 * (tid as u32 + 1);
        }
        let out_words = threads * RESULTS;
        let ud = user_data(raw_desc(IN_BASE, (input.len() * 4) as u32), raw_desc(OUT_BASE, (out_words * 4) as u32), 8);
        let out = Dispatch::new(&CODE, wave, threads as u32, ud).buffer(IN_BASE, input).output(OUT_BASE, out_words).run().unwrap().remove(0);
        let mut failures = Vec::new();
        for tid in 0..threads {
            for (i, want) in expected(tid as u32).into_iter().enumerate() {
                let got = out[tid * RESULTS + i];
                if got != want {
                    failures.push(format!("relative indexing: wave{wave} lane {tid} {} is {got:#010x}, expected {want:#010x}", NAMES[i]));
                }
            }
        }
        assert_clean(failures);
    }

    #[test]
    fn vector_relative_out_of_range_wave32() {
        check(32);
    }

    #[test]
    fn vector_relative_out_of_range_wave64() {
        check(64);
    }
}
