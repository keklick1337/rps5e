// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceTextToSpeech2: one process-wide speech session without a speech
//! engine. The library is initialised, one voice is opened, text is
//! accepted for speaking and closed again; no sound is produced, so an
//! accepted utterance is finished at once and the speech status reads idle.
//!
//! The API has no handles: `Open(param)`, `Speak(param)`, `Cancel()`,
//! `GetSpeechStatus(int*)`, `Close()` and `Terminate()` act on the one
//! session. Parameter blocks are 64 bytes:
//! - Initialize: version u32 @0, thread priority i32 @4, CPU affinity u64 @8;
//! - Open: language u32 @0, voice type u32 @4;
//! - Speak: UTF-16 text pointer @0, length in units u32 @8, mode u32 @0xC,
//!   volume u32 @0x10, speed i32 @0x14.
//!
//! Misuse (missing parameter, a call in the wrong state) returns kernel
//! errno codes: EINVAL for a missing or empty parameter, EPERM for a call
//! before Initialize or without an open voice, EBUSY for a second Open.

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};
use std::sync::Mutex;

const LIB: &str = "libSceTextToSpeech2";

const OK: HleResult = 0;
const ERR_PERM: HleResult = 0x8002_0001u32 as i32 as i64;
const ERR_BUSY: HleResult = 0x8002_0010u32 as i32 as i64;
const ERR_INVAL: HleResult = 0x8002_0016u32 as i32 as i64;

/// GetSpeechStatus value: nothing is being spoken.
const SPEECH_STATUS_IDLE: u32 = 0;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct Session {
    initialized: bool,
    open: bool,
    /// Utterances accepted since the voice was opened (all finished).
    spoken: u64,
}

impl Session {
    fn initialize(&mut self) -> HleResult {
        self.initialized = true;
        OK
    }

    fn open(&mut self) -> HleResult {
        if !self.initialized {
            return ERR_PERM;
        }
        if self.open {
            return ERR_BUSY;
        }
        self.open = true;
        self.spoken = 0;
        OK
    }

    /// Accept one utterance; with no engine it finishes immediately.
    fn speak(&mut self, has_text: bool) -> HleResult {
        if !self.open {
            return ERR_PERM;
        }
        if !has_text {
            return ERR_INVAL;
        }
        self.spoken += 1;
        OK
    }

    /// Nothing is ever in progress, so there is nothing to stop.
    fn cancel(&self) -> HleResult {
        if self.open { OK } else { ERR_PERM }
    }

    fn status(&self) -> Result<u32, HleResult> {
        if self.open { Ok(SPEECH_STATUS_IDLE) } else { Err(ERR_PERM) }
    }

    fn close(&mut self) -> HleResult {
        if !self.open {
            return ERR_PERM;
        }
        self.open = false;
        OK
    }

    /// Terminate closes a voice left open.
    fn terminate(&mut self) -> HleResult {
        if !self.initialized {
            return ERR_PERM;
        }
        *self = Session::default();
        OK
    }
}

static SESSION: Mutex<Session> = Mutex::new(Session { initialized: false, open: false, spoken: 0 });

fn session() -> std::sync::MutexGuard<'static, Session> {
    SESSION.lock().unwrap_or_else(|e| e.into_inner())
}

/// sceTextToSpeech2Initialize(const InitParam*).
fn cb_initialize(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return ERR_INVAL;
    }
    session().initialize()
}

/// sceTextToSpeech2Open(const OpenParam*).
fn cb_open(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 {
        return ERR_INVAL;
    }
    session().open()
}

/// sceTextToSpeech2Speak(const SpeakParam*): needs a text pointer and a
/// nonzero length.
fn cb_speak(ctx: &mut CpuContext) -> HleResult {
    let param = ctx.rdi();
    let has_text = param != 0
        && ctx.read_u64(param).is_some_and(|text| text != 0)
        && ctx.read_u32(param + 8).is_some_and(|len| len != 0);
    session().speak(has_text)
}

/// sceTextToSpeech2Cancel().
fn cb_cancel(_ctx: &mut CpuContext) -> HleResult {
    session().cancel()
}

/// sceTextToSpeech2GetSpeechStatus(int* status).
fn cb_get_speech_status(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_INVAL;
    }
    match session().status() {
        Ok(status) => {
            if ctx.write_u32(out, status) { OK } else { ERR_INVAL }
        }
        Err(e) => e,
    }
}

/// sceTextToSpeech2Close().
fn cb_close(_ctx: &mut CpuContext) -> HleResult {
    session().close()
}

/// sceTextToSpeech2Terminate().
fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    session().terminate()
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("08JSg9p6bgQ", "sceTextToSpeech2GetSpeechStatus", cb_get_speech_status),
    ("2jiIxUmcsGo", "sceTextToSpeech2Cancel", cb_cancel),
    ("UOjiprYwVNw", "sceTextToSpeech2Initialize", cb_initialize),
    ("X0HZNbSiqyg", "sceTextToSpeech2Open", cb_open),
    ("8ntsRd07EQA", "sceTextToSpeech2Speak", cb_speak),
    ("t4e879M-cSw", "sceTextToSpeech2Close", cb_close),
    ("SoWHuVW0gpU", "sceTextToSpeech2Terminate", cb_terminate),
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

    /// Initialize -> Open -> Speak -> status idle -> Close -> Terminate, and
    /// the calls that are out of order.
    #[test]
    fn session_state_machine() {
        let mut s = Session::default();
        assert_eq!(s.open(), ERR_PERM, "Open before Initialize");
        assert_eq!(s.speak(true), ERR_PERM);
        assert_eq!(s.close(), ERR_PERM);
        assert_eq!(s.terminate(), ERR_PERM);

        assert_eq!(s.initialize(), OK);
        assert_eq!(s.status(), Err(ERR_PERM), "no voice open");
        assert_eq!(s.cancel(), ERR_PERM);
        assert_eq!(s.open(), OK);
        assert_eq!(s.open(), ERR_BUSY);
        assert_eq!(s.speak(false), ERR_INVAL);
        assert_eq!(s.speak(true), OK);
        assert_eq!(s.spoken, 1);
        assert_eq!(s.status(), Ok(SPEECH_STATUS_IDLE), "accepted speech has finished");
        assert_eq!(s.cancel(), OK);
        assert_eq!(s.close(), OK);
        assert_eq!(s.close(), ERR_PERM);
        assert_eq!(s.open(), OK, "reopen after Close");
        assert_eq!(s.spoken, 0);
        assert_eq!(s.terminate(), OK, "Terminate closes the open voice");
        assert_eq!(s, Session::default());
    }

    /// The callbacks read the guest parameter blocks.
    #[test]
    fn callbacks_validate_parameters() {
        use crate::context::CpuRegister::Rdi;
        const BASE: u64 = 0x5D71_7700_0000;
        let vm = ps5vm::VirtualMemory::new();
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        *session() = Session::default();

        ctx.set(Rdi, 0);
        assert_eq!(cb_initialize(&mut ctx), ERR_INVAL);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_initialize(&mut ctx), OK);
        assert_eq!(cb_open(&mut ctx), OK);

        let (param, text) = (BASE + 0x100, BASE + 0x200);
        ctx.write_bytes(text, &[b'h', 0, b'i', 0]);
        ctx.write_u64(param, text);
        ctx.write_u32(param + 8, 0);
        ctx.set(Rdi, param);
        assert_eq!(cb_speak(&mut ctx), ERR_INVAL, "empty text");
        ctx.write_u32(param + 8, 2);
        assert_eq!(cb_speak(&mut ctx), OK);
        ctx.set(Rdi, 0);
        assert_eq!(cb_speak(&mut ctx), ERR_INVAL);

        ctx.write_u32(BASE + 0x300, 0xFFFF_FFFF);
        ctx.set(Rdi, BASE + 0x300);
        assert_eq!(cb_get_speech_status(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE + 0x300), Some(SPEECH_STATUS_IDLE));
        ctx.set(Rdi, 0);
        assert_eq!(cb_get_speech_status(&mut ctx), ERR_INVAL);

        assert_eq!(cb_cancel(&mut ctx), OK);
        assert_eq!(cb_close(&mut ctx), OK);
        assert_eq!(cb_terminate(&mut ctx), OK);
        assert_eq!(cb_terminate(&mut ctx), ERR_PERM);
    }
}
