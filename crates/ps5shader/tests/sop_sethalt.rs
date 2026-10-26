// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! s_sethalt: 0 (resume) is a no-op for a running wave; halting is not
//! modelled.

use ps5shader::testing::asm::*;
use ps5shader::testing::IoTest;

fn program(halt: u32) -> Vec<u32> {
    let mut code = vec![sopp(0x0D, halt), vop1(OP_V_MOV_B32, 1, inline_u32(5))];
    code.extend(mubuf(OP_BUFFER_STORE_DWORD, 1, 0, 1, inline_u32(0), 0, true, false, false));
    code.push(S_ENDPGM);
    code
}

#[test]
fn sethalt_zero_runs_on() {
    let t = IoTest::new(&program(0), 4, 4);
    assert_eq!(t.run(&[0; 4]).unwrap(), vec![5; 4]);
}

#[test]
fn sethalt_one_is_unsupported() {
    let t = IoTest::new(&program(1), 4, 4);
    let e = t.run.compile(true).unwrap_err();
    assert!(e.contains("s_sethalt"), "{e}");
}
