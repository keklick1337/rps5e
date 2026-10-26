// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! FMOD (`libfmod`, shipped by Unity titles): `System::setOutput` override.
//!
//! Some builds hand Unity an FMOD system object whose "initialised" byte is
//! already set before Unity applies its startup output configuration, and
//! FMOD Studio then skips the real core initialisation. The override records
//! the output type in the object and clears that byte for the first two
//! calls per object, so the next init runs fully.
//!
//! The field offsets belong to one FMOD build, so the override is opt-in
//! (`RPS5E_FMOD_OUTPUT_RESET=1`). Otherwise the game's own libfmod export
//! runs (the HLE body is only used when no module exports the symbol).

use std::collections::HashMap;
use std::sync::Mutex;

use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};

const LIB: &str = "libfmod";
/// The symbol's name is not known; the catalog lists it under this one.
const NID: &str = "uPLTdl3psGk";
const NAME: &str = "FmodSystemSetOutput";

/// `System` object fields of the affected build.
const INITIALISED_OFFSET: u64 = 0x08;
const OUTPUT_TYPE_OFFSET: u64 = 0x116D0;
const OUTPUT_PENDING_OFFSET: u64 = 0x116D4;
/// Calls per object that still reset the premature initialisation.
const RESET_CALLS: u32 = 2;

static CALLS: Mutex<Option<HashMap<u64, u32>>> = Mutex::new(None);

/// FMOD::System::setOutput(System* this, FMOD_OUTPUTTYPE output) -> FMOD_OK
fn cb_system_set_output(ctx: &mut CpuContext) -> HleResult {
    let (system, output) = (ctx.rdi(), ctx.rsi() as u32);
    let call = {
        let mut g = CALLS.lock().unwrap_or_else(|p| p.into_inner());
        let n = g.get_or_insert_with(HashMap::new).entry(system).or_insert(0);
        *n += 1;
        *n
    };
    if call <= RESET_CALLS && system != 0 {
        ctx.write_bytes(system + INITIALISED_OFFSET, &[0]);
        ctx.write_u32(system + OUTPUT_TYPE_OFFSET, output);
        ctx.write_u32(system + OUTPUT_PENDING_OFFSET, 0);
    }
    log::debug!("[fmod] System::setOutput system={system:#x} output={output} call={call}");
    0
}

fn override_enabled() -> bool {
    std::env::var("RPS5E_FMOD_OUTPUT_RESET").is_ok_and(|v| v == "1")
}

pub fn register(reg: &mut HleRegistry) {
    if override_enabled() {
        reg.register_fn(NID, NAME, LIB, cb_system_set_output);
    } else {
        reg.register_fallback_fn(NID, NAME, LIB, cb_system_set_output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    #[test]
    fn nid_is_catalogued() {
        let db = ps5nid::builtin_exports();
        assert!(db.iter().any(|e| e.nid == NID && e.name == NAME && e.library == LIB));
    }

    #[test]
    fn set_output_resets_the_first_two_calls() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5DA6_0000_0000;
        vm.map(BASE, 0x12000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdi, BASE);
        for call in 1..=3u32 {
            ctx.write_bytes(BASE + INITIALISED_OFFSET, &[1]);
            ctx.write_u32(BASE + OUTPUT_PENDING_OFFSET, 7);
            ctx.set(Rsi, 10 + call as u64);
            assert_eq!(cb_system_set_output(&mut ctx), 0);
            let reset = call <= RESET_CALLS;
            assert_eq!(ctx.read_bytes(BASE + INITIALISED_OFFSET, 1).unwrap()[0], if reset { 0 } else { 1 });
            assert_eq!(ctx.read_u32(BASE + OUTPUT_PENDING_OFFSET), Some(if reset { 0 } else { 7 }));
        }
        assert_eq!(ctx.read_u32(BASE + OUTPUT_TYPE_OFFSET), Some(12));
    }
}
