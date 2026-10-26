// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceTextToSpeech2: no speech is ever produced, so there is nothing to
//! report or cancel; the library initialises but opening a voice fails.

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceTextToSpeech2";

fn cb_ok(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// No speech engine exists, so no voice channel can be opened.
const ERROR_NOT_IMPLEMENTED: HleResult = 0x8002_FFFFu32 as i32 as i64;

/// sceTextToSpeech2Open(...): fails, the caller keeps speech disabled.
fn cb_open(_ctx: &mut CpuContext) -> HleResult {
    ERROR_NOT_IMPLEMENTED
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("08JSg9p6bgQ", "sceTextToSpeech2GetSpeechStatus", cb_ok),
    ("2jiIxUmcsGo", "sceTextToSpeech2Cancel", cb_ok),
    ("UOjiprYwVNw", "sceTextToSpeech2Initialize", cb_ok),
    ("X0HZNbSiqyg", "sceTextToSpeech2Open", cb_open),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in super::EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }
}
