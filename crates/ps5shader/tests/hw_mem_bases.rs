// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Buffer, LDS and FLAT addressing edge cases from the reference driver's
//! execution tests (BufferResourceVcc, BufferUnalignedBase,
//! FormattedStoreUnaligned, LdsCondxchg, FlatApertureSources), with their
//! expected values computed the same way.

include!("vectors/mem_common.rs");
include!("vectors/exec_programs.rs");

use exec_programs as ep;

use ps5shader::ssa::SystemValue;

const IN: u64 = 0x1_0000_0000;
const OUT: u64 = 0x2_0000_0000;
const GUARD: u32 = 0xdead_beef;

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// A V# of `bytes` bytes at `base` with the given dword3.
fn desc(base: u64, bytes: u32, dword3: u32) -> [u32; 4] {
    [base as u32, ((base >> 32) as u32) & 0xffff, bytes, dword3]
}

fn user(descs: &[[u32; 4]]) -> Vec<u32> {
    descs.iter().flatten().copied().collect()
}

/// Run a work-group of `threads` invocations as consecutive waves (local
/// ids continue across waves) on shared memory; returns the output words.
fn waves(code: &[u32], wave: u32, threads: u32, user: Vec<u32>, setup: &dyn Fn(&mut Machine), output: (u64, usize)) -> Result<Vec<u32>, String> {
    let mut memory: Option<ps5shader::interp::Memory> = None;
    for k in 0..threads.div_ceil(wave) {
        let lanes = (threads - k * wave).min(wave);
        let carried = memory.clone();
        let wave_setup = |m: &mut Machine| {
            match &carried {
                Some(mem) => m.memory = mem.clone(),
                None => setup(m),
            }
            m.system.insert(SystemValue::LocalInvocationId(0), (k * wave..k * wave + lanes).collect());
        };
        let m = ComputeRun::new(code.to_vec(), wave, lanes, user.clone()).run_checked(&wave_setup, &[output])?;
        memory = Some(m.memory.clone());
    }
    Ok(memory.unwrap().read_words(output.0, output.1))
}

mod resource_vcc {
    use super::*;

    const THREADS: u32 = 64;

    /// The V# of one side of the copy lives in s[104:105] and VCC
    /// (s[104:107] = s[4:7]), the other in s[0:3].
    fn run(wave: u32, opcode: u32, bytes: u32, format: u32, store: bool) -> Vec<String> {
        let input: Vec<u32> = (0..THREADS).map(|l| 0x3f80_0000 | (l * 0x10001)).collect();
        let code = ep::resource_vcc(opcode, store);
        // Output words 4..68 are the lanes; 0..4 and 68..72 guards.
        let out_lanes = OUT + 16;
        let output = desc(out_lanes, if store { bytes } else { THREADS * 4 }, format);
        let inp = desc(IN, if store { THREADS * 4 } else { bytes }, format);
        let (ordinary, mixed) = if store { (inp, output) } else { (output, inp) };
        let setup = |m: &mut Machine| {
            m.memory.write_words(IN, &input);
            m.memory.write_words(OUT, &vec![GUARD; THREADS as usize + 8]);
        };
        let out = match waves(&code, wave, THREADS, user(&[ordinary, mixed]), &setup, (OUT, THREADS as usize + 8)) {
            Ok(o) => o,
            Err(e) => return vec![format!("wave{wave} {opcode:#010x} store={store}: {e}")],
        };
        let mut failures = Vec::new();
        for i in 0..4 {
            if out[i] != GUARD || out[THREADS as usize + 4 + i] != GUARD {
                failures.push(format!("wave{wave} {opcode:#010x} store={store}: wrote past the output"));
            }
        }
        for lane in 0..THREADS {
            let mut want = if lane * 4 < bytes { input[lane as usize] } else if store { GUARD } else { 0 };
            if !store && format == 0x3103_8fac && opcode == 0xe000_1000 && lane * 4 < bytes {
                want = ((input[lane as usize] & 0xff) as f32 / 255.0).to_bits();
            }
            let got = out[lane as usize + 4];
            if got != want {
                failures.push(format!("wave{wave} {opcode:#010x} store={store} bytes={bytes} format={format:#010x} lane {lane}: {got:#010x}, expected {want:#010x}"));
            }
        }
        failures
    }

    #[test]
    fn buffer_resource_in_vcc() {
        let mut failures = Vec::new();
        for wave in [32u32, 64] {
            for load in [0xe8b0_1000u32, 0xe030_1000, 0xe000_1000] {
                failures.extend(run(wave, load, THREADS * 4, 0x3101_6fac, false));
                failures.extend(run(wave, load, 12, 0x3101_6fac, false));
            }
            failures.extend(run(wave, 0xe000_1000, THREADS * 4, 0x3103_8fac, false));
            for store in [0xe070_1000u32, 0xe8b4_1000] {
                failures.extend(run(wave, store, THREADS * 4, 0x3101_6fac, true));
                failures.extend(run(wave, store, 12, 0x3101_6fac, true));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}

mod unaligned_base {
    use super::*;

    const LANES: u32 = 32;
    const SLOTS: u32 = 8;
    const INPUT_BYTES: u32 = 256;
    const VIEW_BYTES: u32 = 60;
    const RESULT_BYTES: u32 = LANES * SLOTS * 4;
    const STORED_BYTES: u32 = 8 * 28;
    const FILL: u32 = 0xdead_beef;
    const OUTPUT_WORDS: usize = 384;
    const NAMES: [&str; 8] = ["aligned dword", "dword", "dwordx2[0]", "dwordx2[1]", "ushort", "ubyte", "s_buffer_load_dword", "tbuffer_load_format_x 8_UINT"];




    fn view_dword(view: &[u8], dword: u32) -> u32 {
        if dword >= VIEW_BYTES / 4 {
            return 0;
        }
        let d = dword as usize * 4;
        u32::from_le_bytes(view[d..d + 4].try_into().unwrap())
    }

    fn view_field(view: &[u8], byte: u32, mask: u32) -> u32 {
        (view_dword(view, byte / 4) >> ((byte % 4) * 8)) & mask
    }

    fn fill_byte(offset: usize) -> u8 {
        (FILL >> ((offset % 4) * 8)) as u8
    }

    fn raw(base: u64, bytes: u32) -> [u32; 4] {
        desc(base, bytes, 0x3101_6fac)
    }

    /// Runs `code` with `input` at IN and the output (FILL) at OUT;
    /// returns the output bytes.
    fn go(code: &[u32], descs: &[[u32; 4]], extra: &[u32], input: &[u8]) -> Result<Vec<u8>, String> {
        let mut u = user(descs);
        u.extend_from_slice(extra);
        let words = bytes_to_words(input);
        let setup = |m: &mut Machine| {
            m.memory.write_words(IN, &words);
            m.memory.write_words(OUT, &[FILL; OUTPUT_WORDS]);
        };
        let out = dispatch(code, LANES, LANES, u, &setup, &[(OUT, OUTPUT_WORDS)])?;
        Ok(words_to_bytes(&out[0]))
    }

    fn run(view_offset: u32, store_offset: u32) -> Vec<String> {
        let input: Vec<u8> = (0..INPUT_BYTES).map(|i| (i.wrapping_mul(37) + 11 + (i >> 3)) as u8).collect();
        let view = &input[view_offset as usize..];
        let descs = [raw(IN, INPUT_BYTES), raw(IN + view_offset as u64, VIEW_BYTES), raw(OUT, RESULT_BYTES), raw(OUT + (RESULT_BYTES + store_offset) as u64, STORED_BYTES)];
        let name = format!("view +{view_offset}, stores +{store_offset}");
        let out = match go(&ep::UNALIGNED_BASE, &descs, &[], &input) {
            Ok(o) => o,
            Err(e) => return vec![format!("{name}: {e}")],
        };
        let word = |i: usize| u32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
        let mut failures = Vec::new();
        for lane in 0..LANES {
            let aligned = u32::from_le_bytes(input[lane as usize * 4..lane as usize * 4 + 4].try_into().unwrap());
            let s = (view_offset & !3) as usize + 4;
            let scalar = u32::from_le_bytes(input[s..s + 4].try_into().unwrap());
            let want = [
                aligned,
                view_dword(view, lane),
                view_dword(view, lane + 1),
                view_dword(view, lane + 2),
                view_field(view, lane * 2 + 2, 0xffff),
                view_field(view, lane * 3 + 1, 0xff),
                scalar,
                view_field(view, lane * 4 + 2, 0xff),
            ];
            for (slot, &w) in want.iter().enumerate() {
                let got = word((lane * SLOTS) as usize + slot);
                if got != w {
                    failures.push(format!("{name}: lane {lane} {} is {got:#010x}, expected {w:#010x}", NAMES[slot]));
                }
            }
        }
        for offset in RESULT_BYTES as usize..OUTPUT_WORDS * 4 {
            let relative = offset as i64 - (RESULT_BYTES + store_offset) as i64;
            let mut want = fill_byte(offset);
            if (0..STORED_BYTES as i64).contains(&relative) {
                let lane = (relative / 8) as u32;
                let byte = (relative % 8) as u32;
                want = match byte {
                    0..=3 => ((0xa1b2_c300u32 + lane) >> (byte * 8)) as u8,
                    4 => (0x91 + lane) as u8,
                    5 => (0x3c + lane) as u8,
                    _ => ((0xbe00 + lane) >> ((byte - 6) * 8)) as u8,
                };
            }
            if out[offset] != want {
                failures.push(format!("{name}: stored byte {} is {:#04x}, expected {want:#04x}", offset - RESULT_BYTES as usize, out[offset]));
            }
        }
        failures
    }

    fn run_pair(view_offset: u32) -> Vec<String> {
        let input: Vec<u8> = (0..INPUT_BYTES).map(|i| (i.wrapping_mul(53) + 7 + (i >> 2)) as u8).collect();
        let view = &input[view_offset as usize..];
        let descs = [raw(IN, INPUT_BYTES), raw(IN + view_offset as u64, VIEW_BYTES), raw(OUT, RESULT_BYTES)];
        let name = format!("glc dwordx2, view +{view_offset}");
        let out = match go(&ep::UNALIGNED_PAIR, &descs, &[], &input) {
            Ok(o) => o,
            Err(e) => return vec![format!("{name}: {e}")],
        };
        let word = |i: usize| u32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
        let mut failures = Vec::new();
        for lane in 0..LANES {
            for half in 0..2 {
                let (got, want) = (word((lane * 2 + half) as usize), view_dword(view, lane / 4 + half));
                if got != want {
                    failures.push(format!("{name}: lane {lane} dword {half} is {got:#010x}, expected {want:#010x}"));
                }
            }
        }
        let (lo, hi) = (word(LANES as usize * 2), word(LANES as usize * 2 + 1));
        if lo != FILL.wrapping_add(LANES) || hi != FILL {
            failures.push(format!("{name}: the 64-bit atomic is {hi:#010x}{lo:08x}"));
        }
        failures
    }

    fn run_atomic(misalignment: u32) -> Vec<String> {
        let descs = [raw(OUT + misalignment as u64, (LANES + 2) * 4), raw(OUT + 128 * 4, LANES * 4)];
        let name = format!("atomic, view +{misalignment}");
        let out = match go(&ep::UNALIGNED_ATOMIC, &descs, &[4 - misalignment], &[]) {
            Ok(o) => o,
            Err(e) => return vec![format!("{name}: {e}")],
        };
        let word = |i: usize| u32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
        let mut failures = Vec::new();
        for w in 0..128u32 {
            let want = if (1..=LANES).contains(&w) { FILL + w } else { FILL };
            if word(w as usize) != want {
                failures.push(format!("{name}: dword {w} is {:#010x}, expected {want:#010x}", word(w as usize)));
            }
        }
        for lane in 0..LANES as usize {
            if word(128 + lane) != FILL {
                failures.push(format!("{name}: lane {lane} returned {:#010x}, expected {FILL:#010x}", word(128 + lane)));
            }
        }
        failures
    }

    fn run_formatted_load(view_offset: u32) -> Vec<String> {
        let input: Vec<u8> = (0..INPUT_BYTES).map(|i| (i.wrapping_mul(29) + 5 + (i >> 4)) as u8).collect();
        let view = &input[view_offset as usize..];
        let descs = [raw(IN + view_offset as u64, VIEW_BYTES), raw(OUT, RESULT_BYTES)];
        let out = match go(&ep::formatted_load(11, 2), &descs, &[], &input) {
            Ok(o) => o,
            Err(e) => return vec![format!("tbuffer_load_format_x 16_UINT: {e}")],
        };
        (0..LANES)
            .filter_map(|lane| {
                let got = u32::from_le_bytes(out[lane as usize * 4..lane as usize * 4 + 4].try_into().unwrap());
                let want = view_field(view, lane * 4 + 2, 0xffff);
                (got != want).then(|| format!("tbuffer_load_format_x 16_UINT, view +{view_offset}: lane {lane} is {got:#010x}, expected {want:#010x}"))
            })
            .collect()
    }

    fn run_formatted_store(view_offset: u32) -> Vec<String> {
        let descs = [raw(OUT + view_offset as u64, LANES * 4), raw(OUT, RESULT_BYTES)];
        let out = match go(&ep::formatted_store(11, 0), &descs, &[], &[]) {
            Ok(o) => o,
            Err(e) => return vec![format!("tbuffer_store_format_x 16_UINT: {e}")],
        };
        let mut failures = Vec::new();
        for (offset, &got) in out.iter().enumerate() {
            let relative = offset as i64 - view_offset as i64;
            let mut want = fill_byte(offset);
            if (0..LANES as i64 * 4).contains(&relative) && relative % 4 < 2 {
                want = ((0xc300 + (relative / 4) as u32) >> ((relative % 4) * 8)) as u8;
            }
            if got != want {
                failures.push(format!("tbuffer_store_format_x 16_UINT, view +{view_offset}: byte {offset} is {got:#04x}, expected {want:#04x}"));
            }
        }
        failures
    }

    #[test]
    fn unaligned_bases() {
        let mut failures = Vec::new();
        failures.extend(run(2, 1));
        failures.extend(run(3, 3));
        failures.extend(run(1, 2));
        assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures[..failures.len().min(80)].join("\n"));
    }

    #[test]
    fn unaligned_glc_pair_and_atomic64() {
        let failures: Vec<String> = [1, 2, 3].into_iter().flat_map(run_pair).collect();
        assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures[..failures.len().min(80)].join("\n"));
    }

    #[test]
    fn unaligned_atomics() {
        let failures: Vec<String> = [1, 2, 3].into_iter().flat_map(run_atomic).collect();
        assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures[..failures.len().min(80)].join("\n"));
    }

    /// 16-bit formatted accesses through a V# whose base is aligned to the
    /// element (2 bytes) but not to a dword. (The reference rejects
    /// formatted accesses through a V# misaligned to its element, and
    /// atomics off a dword boundary, which fault the wave on the hardware;
    /// our compiler does not see the V# words, so those cases are not
    /// ported.)
    #[test]
    fn unaligned_formatted_16() {
        let mut failures = run_formatted_load(2);
        failures.extend(run_formatted_store(2));
        assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures[..failures.len().min(80)].join("\n"));
    }
}

/// tbuffer_store_format_x/xy/xyz/xyzw of 8/16-bit unsigned formats at
/// byte offsets that are not multiples of the element: each channel lands
/// at its offset, clamped to the format's range, missing components 0.
#[test]
fn formatted_store_unaligned() {
    const THREADS: u32 = 32;
    const LANE_BYTES: u32 = ep::FORMATTED_STORE_LANE_BYTES;
    const FILL: u32 = ep::FORMATTED_STORE_FILL;
    const STORES: [(u32, u32, u32, u32, u32); 8] = ep::FORMATTED_STORES;
    let value = |tid: u32, c: u32| {
        let mixed = (tid + 1).wrapping_mul(2654435761).wrapping_add(c * 40503);
        if tid % 5 == 3 {
            mixed | 0x10000
        } else {
            (mixed >> 3) & 0xffff
        }
    };
    let code = ep::formatted_store_unaligned();
    let input: Vec<u32> = (0..THREADS).flat_map(|t| (0..4).map(move |c| value(t, c))).collect();
    let out_words = (THREADS * LANE_BYTES / 4) as usize;
    let setup = |m: &mut Machine| {
        m.memory.write_words(IN, &input);
        m.memory.write_words(OUT, &vec![0xdede_dede; out_words]);
    };
    let descs = [desc(IN, THREADS * 16, 0x3101_6fac), desc(OUT, THREADS * LANE_BYTES, 0x3101_6fac)];
    let out = words_to_bytes(&dispatch(&code, 32, THREADS, user(&descs), &setup, &[(OUT, out_words)]).unwrap()[0]);
    let mut failures = Vec::new();
    for tid in 0..THREADS {
        let mut want = [FILL as u8; LANE_BYTES as usize];
        for (_, width, channels, components, offset) in STORES {
            let limit = if width == 1 { 0xff } else { 0xffff };
            for c in 0..channels {
                let v = if c < components { value(tid, c).min(limit) } else { 0 };
                for b in 0..width {
                    want[(offset + c * width + b) as usize] = (v >> (8 * b)) as u8;
                }
            }
        }
        for (b, &w) in want.iter().enumerate() {
            let got = out[(tid * LANE_BYTES) as usize + b];
            if got != w {
                failures.push(format!("lane {tid} byte {b} is {got:#04x}, expected {w:#04x}"));
            }
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures[..failures.len().min(80)].join("\n"));
}

mod lds_condxchg {
    use super::*;

    const WORDS: u32 = 8;
    const LDS_DWORDS: u32 = 4096;
    const SIGN: u32 = 0x8000_0000;
    const SENTINEL: u32 = 0xdead_beef;



    fn run(wave: u32, offset: u32, destination: u32, inactive: bool, contended: bool) -> Vec<String> {
        let input = ep::condxchg_input(wave, offset, contended);
        let n = (wave * WORDS) as usize;
        let descs = [desc(IN, 64 * WORDS * 4, 0x3101_6fac), desc(OUT, 64 * WORDS * 4, 0x3101_6fac)];
        let setup = |m: &mut Machine| {
            m.lds = vec![0; LDS_DWORDS as usize * 4];
            m.memory.write_words(IN, &input);
            m.memory.write_words(OUT, &vec![SENTINEL; n]);
        };
        let name = format!("wave{wave} offset {offset:#x} v{destination} inactive={inactive} contended={contended}");
        let out = match dispatch(&ep::lds_condxchg(wave, offset, destination, inactive, contended), wave, wave, user(&descs), &setup, &[(OUT, n)]) {
            Ok(o) => o.into_iter().next().unwrap(),
            Err(e) => return vec![format!("{name}: {e}")],
        };
        let mut failures = Vec::new();
        let mut expect = |tid: u32, index: u32, want: u32| {
            let got = out[(tid * WORDS + index) as usize];
            if got != want {
                failures.push(format!("{name}: lane {tid} word {index} is {got:#010x}, expected {want:#010x}"));
            }
        };
        for tid in 0..wave {
            let i = &input[(tid * WORDS) as usize..((tid + 1) * WORDS) as usize];
            if !contended {
                let active = !inactive || tid % 2 == 0;
                let in_bounds = i[5].wrapping_add(offset) < LDS_DWORDS * 4;
                expect(tid, 0, if active { if in_bounds { i[0] } else { 0 } } else if destination == 12 { i[3] } else { i[5] });
                expect(tid, 1, if active { if in_bounds { i[1] } else { 0 } } else if destination == 12 { SENTINEL } else { i[6] });
                expect(tid, 2, if active && in_bounds && i[2] & SIGN != 0 { i[2] & !SIGN } else { i[0] });
                expect(tid, 3, if active && in_bounds && i[3] & SIGN != 0 { i[3] & !SIGN } else { i[1] });
            }
            for index in 4..WORDS {
                expect(tid, index, i[6 + (index % 2) as usize]);
            }
        }
        if contended {
            for word in 0..2usize {
                let last = out[2 + word];
                if !(1..=wave).contains(&last) {
                    failures.push(format!("{name}: invalid final contended dword {last:#x}"));
                    continue;
                }
                let mut seen = vec![false; 65];
                for tid in 0..wave {
                    if out[(tid * WORDS) as usize + 2 + word] != last {
                        failures.push(format!("{name}: lane {tid} final dword {word} differs"));
                    }
                    let old = out[(tid * WORDS) as usize + word];
                    let index = if old == input[word] { 0 } else { old };
                    if index > wave || index == tid + 1 || seen[index as usize] {
                        failures.push(format!("{name}: lane {tid} returned {old:#x} (invalid or duplicate)"));
                    } else {
                        seen[index as usize] = true;
                    }
                }
                if !seen[0] || seen[last as usize] {
                    failures.push(format!("{name}: the returned contended dwords lost an exchange"));
                }
            }
        }
        failures
    }

    #[test]
    fn condxchg32_rtn_b64() {
        let mut failures = Vec::new();
        for wave in [32u32, 64] {
            for (offset, destination, inactive, contended) in ep::CONDXCHG_RUNS {
                failures.extend(run(wave, offset, destination, inactive, contended));
            }
        }
        assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures[..failures.len().min(80)].join("\n"));
    }

    fn compiles(code: &[u32]) -> Result<(), String> {
        ps5shader::testing::ComputeRun::new(code.to_vec(), 32, 32, vec![0; 8]).compile(true).map(|_| ())
    }

    /// Register ranges past v255 and GDS are refused; the instruction
    /// right before the end of the code still decodes.
    #[test]
    fn condxchg_rejections() {
        compiles(&[0xd9f8_0000, 0xfe00_fe00, 0xbf81_0000]).unwrap();
        for (code, what) in [
            (&[0xd9f8_0000u32][..], "truncated"),
            (&[0xd9f8_0000, 0x0200_ff00, 0xbf81_0000][..], "source range"),
            (&[0xd9f8_0000, 0xff00_0400, 0xbf81_0000][..], "destination range"),
            (&[0xd9fa_0000, 0x0200_0400, 0xbf81_0000][..], "GDS"),
        ] {
            assert!(compiles(code).is_err(), "{what}: compiled");
        }
    }
}

/// src_shared_base / limit and src_private_base / limit as scalar and
/// 64-bit sources, as lane masks, and as the high words of FLAT addresses
/// reaching LDS and scratch.
#[test]
fn flat_aperture_sources() {
    const SOURCES: [u32; 4] = ep::APERTURE_SOURCES;
    const VALUES: [u64; 4] = ep::APERTURE_VALUES;
    const LANE_DWORDS: u32 = 32;
    const SHARED_DATA: u32 = ep::APERTURE_SHARED_DATA;
    const PRIVATE_DATA: u32 = ep::APERTURE_PRIVATE_DATA;
    let compares = ep::APERTURE_COMPARES;
    let code = ep::flat_aperture_sources();
    let mut failures = Vec::new();
    for wave in [32u32, 64] {
        let n = (wave * LANE_DWORDS) as usize;
        let setup = |m: &mut Machine| {
            m.lds = vec![0; 256 * 4];
            m.scratch_dwords = Some(4);
            m.memory.write_words(OUT, &vec![GUARD; n]);
            m.system.insert(SystemValue::LocalInvocationId(0), (0..wave).collect());
        };
        let mut u = desc(OUT, n as u32 * 4, 0x3101_6fac).to_vec();
        u.resize(8, 0);
        let out = match dispatch(&code, wave, wave, u, &setup, &[(OUT, n)]) {
            Ok(o) => o.into_iter().next().unwrap(),
            Err(e) => {
                failures.push(format!("wave{wave}: {e}"));
                continue;
            }
        };
        for lane in 0..wave {
            let mut expect = |slot: u32, want: u32, what: &str| {
                let got = out[(lane * LANE_DWORDS + slot) as usize];
                if got != want {
                    failures.push(format!("wave{wave} lane {lane} {what}: {got:#010x}, expected {want:#010x}"));
                }
            };
            for (i, &v) in VALUES.iter().enumerate() {
                let i = i as u32;
                expect(i, v as u32, &format!("s_mov_b32 of source {}", SOURCES[i as usize]));
                expect(4 + i * 2, v as u32, &format!("s_mov_b64 of source {} low", SOURCES[i as usize]));
                expect(5 + i * 2, (v >> 32) as u32, &format!("s_mov_b64 of source {} high", SOURCES[i as usize]));
                let bit = if wave == 32 { (v as u32 >> lane) & 1 } else { ((v >> lane) & 1) as u32 };
                expect(24 + i, bit, &format!("v_cndmask with source {} as the mask", SOURCES[i as usize]));
            }
            expect(12, SHARED_DATA + lane, "ds_read_b32 after a flat store through src_shared_base");
            expect(13, PRIVATE_DATA + lane, "scratch_load_dword after a flat store through src_private_base");
            expect(14, SHARED_DATA + lane, "flat_load_dword through src_shared_base");
            expect(15, PRIVATE_DATA + lane, "flat_load_dword through src_private_base");
            for (i, c) in compares.iter().enumerate() {
                expect(16 + i as u32, c.3, &format!("compare {i}"));
            }
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures[..failures.len().min(80)].join("\n"));
}
