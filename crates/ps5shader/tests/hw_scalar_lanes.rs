// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Cross-lane scalar reads / writes and wave identity (v_readfirstlane,
//! v_readlane, v_writelane, wave32 subgroups, s_get_waveid_in_workgroup)
//! against hardware-measured reference rows.

include!("vectors/scalar_common.rs");

/// Strided (stride 4, one record per dword) input at s[0:3] and output at
/// s[4:7], `words` dwords each; `input` None leaves s[0:3] zero.
fn strided_io(code: &[u32], wave: u32, threads: u32, input: Option<Vec<u32>>, words: usize) -> Vec<u32> {
    let in_desc = if input.is_some() { strided_desc(IN_BASE, words as u32) } else { [0; 4] };
    let ud = user_data(in_desc, strided_desc(OUT_BASE, words as u32), 8);
    let mut d = Dispatch::new(code, wave, threads, ud);
    if let Some(input) = input {
        d = d.buffer(IN_BASE, input);
    }
    d.output(OUT_BASE, words).run().unwrap().remove(0)
}

fn compare(label: &str, lanes: usize, stride: usize, out: &[u32], expected: &dyn Fn(usize) -> Vec<u32>) {
    let mut failures = Vec::new();
    for tid in 0..lanes {
        for (j, want) in expected(tid).into_iter().enumerate() {
            let got = out[tid * stride + j];
            if got != want {
                failures.push(format!("{label}: lane {tid} result {j} is {got:#x}, expected {want:#x}"));
            }
        }
    }
    assert_clean(failures);
}

mod read_first_lane {
    use super::*;
    include!("vectors/scalar_read_first_lane.rs");

    /// v_readfirstlane_b32 with EXEC = 0 (reads lane 0) and with a partial
    /// EXEC (first active lane 4).
    #[test]
    fn read_first_lane_wave32() {
        let out = strided_io(&READ_FIRST_LANE_WAVE32_CODE, 32, 32, None, 64 * 4);
        compare("read first lane wave32", 32, 4, &out, &|_| vec![0x100, 0x104]);
    }

    /// Wave64: EXEC = 0, then EXEC_HI = 0x300 (first active lane 40).
    #[test]
    fn read_first_lane_wave64() {
        let out = strided_io(&READ_FIRST_LANE_WAVE64_CODE, 64, 64, None, 64 * 4);
        compare("read first lane wave64", 64, 4, &out, &|_| vec![0x100, 0x100 + 40]);
    }
}

mod read_lane {
    use super::*;
    include!("vectors/scalar_read_lane.rs");

    /// v_readlane_b32 with an inline lane select, an SGPR lane select and
    /// v_readfirstlane.
    #[test]
    fn read_lane() {
        let mut input = vec![0u32; 32 * 4];
        for tid in 0..32u32 {
            input[tid as usize * 4] = 0xa000_0000u32.wrapping_add(tid.wrapping_mul(0x0101_0101));
            input[tid as usize * 4 + 1] = 69 + tid;
        }
        let out = strided_io(&READ_LANE_CODE, 32, 32, Some(input.clone()), 32 * 4);
        compare("read lane", 32, 4, &out, &|_| vec![input[5 * 4], input[5 * 4], 69]);
    }
}

mod write_lane {
    use super::*;
    include!("vectors/scalar_write_lane.rs");

    /// v_writelane_b32 in a loop reversing the lanes, writelane under
    /// EXEC = 1 / -1 and a literal-from-SOPK write to lane 5.
    #[test]
    fn write_lane() {
        let mut input = vec![0u32; 32 * 4];
        for tid in 0..32u32 {
            input[tid as usize * 4] = 0xa000_0000u32.wrapping_add(tid.wrapping_mul(0x0101_0101));
            input[tid as usize * 4 + 1] = 0x5000_0000 ^ tid.wrapping_mul(0x0010_2030);
        }
        let out = strided_io(&WRITE_LANE_CODE, 32, 32, Some(input.clone()), 32 * 4);
        compare("write lane", 32, 4, &out, &|tid| vec![input[(31 - tid) * 4], input[tid * 4 + 1], if tid == 5 { 0x1234 } else { tid as u32 }]);
    }
}

mod wave32_subgroup {
    use super::*;
    include!("vectors/scalar_wave32_subgroup.rs");

    /// A 64-invocation workgroup of two wave32 waves: v_mbcnt_lo, s_bcnt1
    /// of a VCC compare, v_readlane lane 3 and v_cmpx EXEC.
    fn expected(tid: usize) -> Vec<u32> {
        let lane = (tid % 32) as u32;
        vec![lane, 8, tid as u32 - lane + 3, (lane < 8) as u32]
    }

    /// The reference layout: the stores are `offen` (byte offset tid * 16 +
    /// 0..12) through a stride-4 descriptor with OOB_SELECT 0.
    #[test]
    fn wave32_subgroup() {
        let out = strided_io(&WAVE_CODE, 32, 64, None, 64 * 4);
        compare("wave32 subgroup", 64, 4, &out, &expected);
    }

    /// The same program through a raw (stride 0) output descriptor.
    #[test]
    fn wave32_subgroup_raw_output() {
        let ud = user_data([0; 4], raw_desc(OUT_BASE, 64 * 4 * 4), 8);
        let out = Dispatch::new(&WAVE_CODE, 32, 64, ud).output(OUT_BASE, 64 * 4).run().unwrap().remove(0);
        compare("wave32 subgroup", 64, 4, &out, &expected);
    }
}

mod wave_id {
    use super::*;
    include!("vectors/scalar_wave_id.rs");

    /// s_get_waveid_in_workgroup in 1D and 3D workgroups of several waves:
    /// thread `t` (linearised from v0..v2 for 8-wide rows) stores t / wave.
    #[test]
    fn wave_id_in_workgroup() {
        let groups: [([u32; 3], u32); 6] = [([128, 1, 1], 32), ([128, 1, 1], 64), ([8, 8, 2], 32), ([8, 8, 2], 64), ([8, 13, 1], 32), ([8, 13, 1], 64)];
        let mut failures = Vec::new();
        for (local, wave) in groups {
            let mut d = Dispatch::new(&WAVE_ID_CODE, wave, 0, strided_desc(OUT_BASE, 128).to_vec()).output(OUT_BASE, 128);
            d.local = local;
            d.thread_id_vgprs = 3;
            let out = match d.run() {
                Ok(mut o) => o.remove(0),
                Err(e) => {
                    failures.push(format!("{local:?} wave{wave}: {e}"));
                    continue;
                }
            };
            let count = local[0] * local[1] * local[2];
            for tid in 0..128u32 {
                let want = if tid < count { tid / wave } else { UNTOUCHED };
                if out[tid as usize] != want {
                    failures.push(format!("{local:?} wave{wave}: thread {tid} is {:#x}, expected {want:#x}", out[tid as usize]));
                }
            }
        }
        assert_clean(failures);
    }
}
