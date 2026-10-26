// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! FLAT loads, stores and atomics of a compute program whose addresses lie
//! in the shared aperture (LDS), the private aperture (scratch) or guest
//! memory, per lane: the reference's FlatApertures table. Each lane owns 8
//! dwords of a table (s[4:7]) holding its address and operands, and 8
//! dwords of the output (s[0:3]).

include!("vectors/mem_common.rs");
include!("vectors/flat_apertures.rs");

const OUTPUT: u64 = 0x1_0000_0000;
const TABLE: u64 = 0x2_0000_0000;
const GUEST: u64 = 0x3_0000_0000;

fn run(row: &Row) -> Result<Vec<u32>, String> {
    let lanes = row.wave;
    let slot = if row.atomic { GUEST_ATOMIC_SLOT } else { GUEST_STORE_SLOT };
    let table = table(row, GUEST + slot);
    let words = lanes as usize * 8;
    let mut user = Vec::new();
    put_desc(&mut user, 0, vdesc(OUTPUT, words as u32 * 4));
    put_desc(&mut user, 4, vdesc(TABLE, words as u32 * 4));
    let setup = move |m: &mut Machine| {
        m.lds = vec![0; LDS_DWORDS as usize * 4];
        m.scratch_dwords = Some(SCRATCH_DWORDS);
        m.memory.write_words(TABLE, &table);
        m.memory.write_words(OUTPUT, &vec![0xdead_beef; words]);
        m.memory.write_words(GUEST, &vec![0; GUEST_BYTES / 4]);
    };
    let out = dispatch(&kernel(row), row.wave, lanes, user, &setup, &[(OUTPUT, words)])?;
    Ok(out.into_iter().next().unwrap())
}

fn run_rows(select: impl Fn(&Row) -> bool) {
    let mut failures = Vec::new();
    let mut checked = 0;
    for row in rows().iter().filter(|r| select(r)) {
        match run(row) {
            Ok(output) => check(row, &output, &mut failures),
            Err(e) => failures.push(format!("{}: {e}", describe(row))),
        }
        checked += 1;
    }
    assert!(checked > 0, "no rows selected");
    let shown: Vec<_> = failures.iter().take(64).cloned().collect();
    assert!(failures.is_empty(), "{} failures in {checked} rows:\n{}", failures.len(), shown.join("\n"));
}

#[test]
fn flat_aperture_stores() {
    run_rows(|r| !r.atomic && r.wave == 32);
}

#[test]
fn flat_aperture_atomics32() {
    run_rows(|r| r.atomic && r.width == 32 && r.wave == 32);
}

#[test]
fn flat_aperture_atomics64() {
    run_rows(|r| r.atomic && r.width == 64 && r.wave == 32);
}

#[test]
fn flat_aperture_wave64() {
    run_rows(|r| r.wave == 64);
}

/// A carry out of the low dword leaves the aperture (each is a 4 GiB
/// window): 0x80000000_fffffffc + 4 is the device address
/// 0x80000001_00000000, not LDS byte 0.
#[test]
fn flat_aperture_is_a_4gib_window() {
    use ps5shader::testing::asm::*;
    // s[0:3]: output V#; v[2:3] = 0x80000000_fffffffc.
    let mut code = Vec::new();
    code.push(vop1(OP_V_MOV_B32, 2, LITERAL));
    code.push(0xffff_fffc);
    code.push(vop1(OP_V_MOV_B32, 3, LITERAL));
    code.push(SHARED_HIGH);
    code.push(vop1(OP_V_MOV_B32, 4, LITERAL));
    code.push(0x1234_5678);
    // flat_store_dword v[2:3], v4 offset:4 (address 0x80000001_00000000).
    code.push(flat_word0(STORE_DWORD, false, 4));
    code.push(flat_word1(0, 4, 2));
    code.push(WAITCNT0);
    // flat_load_dword v6, v[4:5]: LDS byte 0 (v4 = 0, v5 = SHARED_HIGH).
    code.push(vop1(OP_V_MOV_B32, 4, inline_u32(0)));
    code.push(vop1(OP_V_MOV_B32, 5, LITERAL));
    code.push(SHARED_HIGH);
    code.push(flat_word0(LOAD_DWORD, false, 0));
    code.push(flat_word1(6, 0, 4));
    code.push(WAITCNT0);
    // buffer_store_dword v6, off, s[0:3].
    code.extend(mubuf(OP_BUFFER_STORE_DWORD, 6, 0, 0, 0x80, 0, false, false, false));
    code.push(END_PROGRAM);
    let mut user = Vec::new();
    put_desc(&mut user, 0, vdesc(OUTPUT, 4));
    let device = 0x8000_0001_0000_0000u64;
    let setup = move |m: &mut Machine| {
        m.lds = vec![0; LDS_DWORDS as usize * 4];
        m.memory.write_words(OUTPUT, &[0xdead_beef]);
    };
    let m = dispatch_machine(&code, 32, 1, user, &setup, &[(OUTPUT, 1), (device, 1)]).unwrap();
    assert_eq!(m.memory.read_u32(OUTPUT), 0, "LDS byte 0 was written through a carried-out aperture address");
    assert_eq!(m.memory.read_u32(device), 0x1234_5678, "the store did not reach the device address");
}
