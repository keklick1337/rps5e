// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceGameLiveStreaming HLE: nothing is ever broadcast, so the status
//! reports "not on air" with no user, spectators or messages.

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceGameLiveStreaming";
const OK: HleResult = 0;

const ERR_INVALID_PARAM: HleResult = 0x80A0_0002u32 as i32 as i64;
const ERR_MEMORY_FAULT: HleResult = 0x8002_000Eu32 as i32 as i64;

/// SceGameLiveStreamingStatus2: userId i32 @0, isOnAir bool @4,
/// spectatorCount @8, textMessageCount @0xC, commandMessageCount @0x10,
/// reserved[52].
const STATUS2_SIZE: usize = 72;
const INVALID_USER_ID: i32 = -1;

fn cb_initialize(_ctx: &mut CpuContext) -> HleResult {
    OK
}

fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    OK
}

pub fn status2() -> [u8; STATUS2_SIZE] {
    let mut s = [0u8; STATUS2_SIZE];
    s[..4].copy_from_slice(&INVALID_USER_ID.to_le_bytes());
    s
}

/// sceGameLiveStreamingGetCurrentStatus2(SceGameLiveStreamingStatus2*)
fn cb_get_current_status2(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_INVALID_PARAM;
    }
    if ctx.write_bytes(out, &status2()) { OK } else { ERR_MEMORY_FAULT }
}

/// sceGameLiveStreamingGetProgramInfo(SceGameLiveStreamingProgramInfo*):
/// no program is running, which the info reports through an invalid
/// broadcasting user id (the leading field).
fn cb_get_program_info(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_INVALID_PARAM;
    }
    if ctx.write_bytes(out, &INVALID_USER_ID.to_le_bytes()) { OK } else { ERR_MEMORY_FAULT }
}

/// sceGameLiveStreamingGetSocialFeedbackMessagesCount(type, uint32_t* count):
/// types 1..3; nobody is watching, so there are no messages.
fn cb_get_social_feedback_messages_count(ctx: &mut CpuContext) -> HleResult {
    let (ty, out) = (ctx.rdi() as i32, ctx.rsi());
    if !(1..=3).contains(&ty) || out == 0 {
        return ERR_INVALID_PARAM;
    }
    if ctx.write_u32(out, 0) { OK } else { ERR_MEMORY_FAULT }
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("kvYEw2lBndk", "sceGameLiveStreamingInitialize", cb_initialize),
    ("9yK6Fk8mKOQ", "sceGameLiveStreamingTerminate", cb_terminate),
    ("lK8dLBNp9OE", "sceGameLiveStreamingGetCurrentStatus2", cb_get_current_status2),
    ("OIIm19xu+NM", "sceGameLiveStreamingGetProgramInfo", cb_get_program_info),
    ("yeQKjHETi40", "sceGameLiveStreamingGetSocialFeedbackMessagesCount", cb_get_social_feedback_messages_count),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn social_feedback_count() {
        use crate::context::CpuRegister::*;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D0C_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_u32(BASE, 7);
        ctx.set(Rdi, 2);
        ctx.set(Rsi, BASE);
        assert_eq!(cb_get_social_feedback_messages_count(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE), Some(0));
        ctx.set(Rdi, 4);
        assert_eq!(cb_get_social_feedback_messages_count(&mut ctx), ERR_INVALID_PARAM);
        ctx.set(Rdi, 1);
        ctx.set(Rsi, 0);
        assert_eq!(cb_get_social_feedback_messages_count(&mut ctx), ERR_INVALID_PARAM);
    }

    #[test]
    fn current_status2_is_off_air() {
        use crate::context::CpuRegister::*;
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D0C_0010_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, &[0xAA; STATUS2_SIZE + 8]);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_get_current_status2(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE), Some(u32::MAX), "invalid user id");
        assert!(ctx.read_bytes(BASE + 4, STATUS2_SIZE - 4).unwrap().iter().all(|&b| b == 0), "not on air");
        assert_eq!(ctx.read_bytes(BASE + STATUS2_SIZE as u64, 8).unwrap(), vec![0xAA; 8]);
        ctx.set(Rdi, 0);
        assert_eq!(cb_get_current_status2(&mut ctx), ERR_INVALID_PARAM);
    }

    #[test]
    fn status_layout() {
        let s = status2();
        assert_eq!(&s[..4], &(-1i32).to_le_bytes());
        assert!(s[4..].iter().all(|&b| b == 0));
    }
}
