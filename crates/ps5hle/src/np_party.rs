// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceNpParty HLE: the user is never in a party.
//!
//! State queries answer "not in a party"; member, session, voice and
//! message calls fail with NOT_IN_PARTY and write nothing (INVALID_ARGUMENT
//! for a null output). The plain and `A` variants behave alike. The library
//! keeps an initialised flag for Initialize / Terminate / CheckCallback and
//! the handler registration; handlers are accepted and never called, since
//! no party event can happen.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceNpParty";
const OK: HleResult = 0;

const fn err(code: u32) -> HleResult {
    code as i32 as HleResult
}
const ERR_ALREADY_INITIALIZED: HleResult = err(0x8055_2502);
const ERR_NOT_INITIALIZED: HleResult = err(0x8055_2503);
const ERR_INVALID_ARGUMENT: HleResult = err(0x8055_2504);
const ERR_NOT_IN_PARTY: HleResult = err(0x8055_2506);
const ERR_MEMORY_FAULT: HleResult = err(0x8002_000E);
const STATE_NOT_IN_PARTY: u16 = 2;

static INITIALIZED: AtomicBool = AtomicBool::new(false);

fn not_in_party(out: u64) -> HleResult {
    if out == 0 { ERR_INVALID_ARGUMENT } else { ERR_NOT_IN_PARTY }
}

fn write_state(ctx: &CpuContext, out: u64) -> HleResult {
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if ctx.write_bytes(out, &STATE_NOT_IN_PARTY.to_le_bytes()) { OK } else { ERR_MEMORY_FAULT }
}

/// sceNpPartyGetState(u16* state)
fn cb_get_state(ctx: &mut CpuContext) -> HleResult {
    write_state(ctx, ctx.rdi())
}

/// sceNpPartyGetStateAsUserA(userId, u16* state)
fn cb_get_state_as_user(ctx: &mut CpuContext) -> HleResult {
    write_state(ctx, ctx.rsi())
}

/// sceNpPartyGetId(id*), sceNpPartyGetMembers(A)(memberList*),
/// sceNpPartySendBinaryMessage(sendParams*)
fn cb_not_in_party_first(ctx: &mut CpuContext) -> HleResult {
    not_in_party(ctx.rdi())
}

/// sceNpPartyGetMemberInfo(A)(memberId, info*),
/// sceNpPartyGetMemberSessionInfo(memberId, info*)
fn cb_not_in_party_second(ctx: &mut CpuContext) -> HleResult {
    not_in_party(ctx.rsi())
}

/// sceNpPartyGetMemberVoiceInfo(memberId, voiceId, u8* state)
fn cb_not_in_party_third(ctx: &mut CpuContext) -> HleResult {
    not_in_party(ctx.rdx())
}

/// sceNpPartyInitialize(param*)
fn cb_initialize(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    match INITIALIZED.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => OK,
        Err(_) => ERR_ALREADY_INITIALIZED,
    }
}

/// sceNpPartyTerminate()
fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    if INITIALIZED.swap(false, Ordering::AcqRel) { OK } else { ERR_NOT_INITIALIZED }
}

/// sceNpPartyCheckCallback()
fn cb_check_callback(_ctx: &mut CpuContext) -> HleResult {
    if INITIALIZED.load(Ordering::Acquire) { OK } else { ERR_NOT_INITIALIZED }
}

/// sceNpPartyRegisterHandler(A)(handlers*, userdata)
fn cb_register_handler(ctx: &mut CpuContext) -> HleResult {
    if !INITIALIZED.load(Ordering::Acquire) {
        return ERR_NOT_INITIALIZED;
    }
    if ctx.rdi() == 0 { ERR_INVALID_ARGUMENT } else { OK }
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("aEzKdJzATZ0", "sceNpPartyGetState", cb_get_state),
    ("EjyAI+QNgFw", "sceNpPartyGetStateAsUserA", cb_get_state_as_user),
    ("DRA3ay-1DFQ", "sceNpPartyGetId", cb_not_in_party_first),
    ("T2UOKf00ZN0", "sceNpPartyGetMembers", cb_not_in_party_first),
    ("TaNw7W25QJw", "sceNpPartyGetMembersA", cb_not_in_party_first),
    ("U6VdUe-PNAY", "sceNpPartySendBinaryMessage", cb_not_in_party_first),
    ("F1P+-wpxQow", "sceNpPartyGetMemberInfo", cb_not_in_party_second),
    ("v2RYVGrJDkM", "sceNpPartyGetMemberInfoA", cb_not_in_party_second),
    ("4gOMfNYzllw", "sceNpPartyGetMemberSessionInfo", cb_not_in_party_second),
    ("EKi1jx59SP4", "sceNpPartyGetMemberVoiceInfo", cb_not_in_party_third),
    ("lhYCTQmBkds", "sceNpPartyInitialize", cb_initialize),
    ("oLYkibiHqRA", "sceNpPartyTerminate", cb_terminate),
    ("3e4k2mzLkmc", "sceNpPartyCheckCallback", cb_check_callback),
    ("kA88gbv71ao", "sceNpPartyRegisterHandler", cb_register_handler),
    ("+v4fVHMwFWc", "sceNpPartyRegisterHandlerA", cb_register_handler),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn never_in_a_party() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5A6C_5000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, &[0xFF; 4]);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_get_state(&mut ctx), OK);
        assert_eq!(ctx.read_bytes(BASE, 4).unwrap(), vec![2, 0, 0xFF, 0xFF]);
        ctx.set(Rdi, 0);
        assert_eq!(cb_get_state(&mut ctx), ERR_INVALID_ARGUMENT);
        ctx.write_bytes(BASE, &[0xFF; 4]);
        ctx.set(Rdi, 1);
        ctx.set(Rsi, BASE);
        assert_eq!(cb_get_state_as_user(&mut ctx), OK);
        assert_eq!(ctx.read_bytes(BASE, 4).unwrap(), vec![2, 0, 0xFF, 0xFF]);

        // Member queries: null outputs are refused, the rest is not in a
        // party; nothing is written.
        ctx.write_bytes(BASE + 0x100, &[0xA5; 64]);
        for (name, reg) in [
            ("sceNpPartyGetId", Rdi),
            ("sceNpPartyGetMembers", Rdi),
            ("sceNpPartyGetMembersA", Rdi),
            ("sceNpPartySendBinaryMessage", Rdi),
            ("sceNpPartyGetMemberInfo", Rsi),
            ("sceNpPartyGetMemberInfoA", Rsi),
            ("sceNpPartyGetMemberSessionInfo", Rsi),
            ("sceNpPartyGetMemberVoiceInfo", Rdx),
        ] {
            let cb = EXPORTS.iter().find(|e| e.1 == name).unwrap().2;
            for r in [Rdi, Rsi, Rdx] {
                ctx.set(r, 1);
            }
            ctx.set(reg, 0);
            assert_eq!(cb(&mut ctx), ERR_INVALID_ARGUMENT, "{name}");
            ctx.set(reg, BASE + 0x100);
            assert_eq!(cb(&mut ctx), ERR_NOT_IN_PARTY, "{name}");
        }
        assert_eq!(ctx.read_bytes(BASE + 0x100, 64).unwrap(), vec![0xA5; 64]);
    }

    /// The only test that touches the initialised flag.
    #[test]
    fn lifecycle_and_handlers() {
        let vm = ps5vm::VirtualMemory::new();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdi, 0x1000);
        assert_eq!(cb_terminate(&mut ctx), ERR_NOT_INITIALIZED);
        assert_eq!(cb_check_callback(&mut ctx), ERR_NOT_INITIALIZED);
        assert_eq!(cb_register_handler(&mut ctx), ERR_NOT_INITIALIZED);
        ctx.set(Rdi, 0);
        assert_eq!(cb_initialize(&mut ctx), ERR_INVALID_ARGUMENT);
        ctx.set(Rdi, 0x1000);
        assert_eq!(cb_initialize(&mut ctx), OK);
        assert_eq!(cb_initialize(&mut ctx), ERR_ALREADY_INITIALIZED);
        ctx.set(Rdi, 0);
        assert_eq!(cb_register_handler(&mut ctx), ERR_INVALID_ARGUMENT);
        ctx.set(Rdi, 0x1000);
        assert_eq!(cb_register_handler(&mut ctx), OK);
        assert_eq!(cb_check_callback(&mut ctx), OK);
        assert_eq!(cb_terminate(&mut ctx), OK);
        assert_eq!(cb_terminate(&mut ctx), ERR_NOT_INITIALIZED);
    }
}
