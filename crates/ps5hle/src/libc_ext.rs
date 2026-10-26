// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceLibcInternal process environment.
//!
//! The guest has its own environment, independent of the host process.
//! `setenv` is always served here; `getenv`/`unsetenv` are used when the
//! game's libc module does not export them itself. Strings handed out by
//! `getenv` live in guest memory and stay valid until the variable changes.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceLibcInternal";

const EINVAL: i32 = 22;
const ENOMEM: i32 = 12;

const MAX_ENV_TEXT: usize = 32 * 1024;

struct EnvVar {
    value: String,
    /// Guest copy of "value\0", allocated on the first getenv.
    guest: Option<u64>,
}

static ENV: Mutex<Option<HashMap<String, EnvVar>>> = Mutex::new(None);

fn with_env<R>(f: impl FnOnce(&mut HashMap<String, EnvVar>) -> R) -> R {
    let mut g = ENV.lock().unwrap_or_else(|p| p.into_inner());
    f(g.get_or_insert_with(HashMap::new))
}

fn fail(ctx: &CpuContext, errno: i32) -> HleResult {
    crate::kernel::set_errno(ctx, errno);
    -1
}

/// A valid variable name: non-empty, without '='.
fn read_name(ctx: &CpuContext, addr: u64) -> Option<String> {
    if addr == 0 {
        return None;
    }
    let name = ctx.read_cstring(addr, MAX_ENV_TEXT)?;
    (!name.is_empty() && !name.contains('=')).then_some(name)
}

/// Store a variable; false when it exists and `overwrite` is not set.
fn set_var(name: String, value: String, overwrite: bool) -> bool {
    with_env(|env| {
        if env.contains_key(&name) && !overwrite {
            return false;
        }
        env.insert(name, EnvVar { value, guest: None });
        true
    })
}

#[cfg(test)]
fn get_var(name: &str) -> Option<String> {
    with_env(|env| env.get(name).map(|v| v.value.clone()))
}

/// setenv(name, value, overwrite) -> 0 / -1
fn cb_setenv(ctx: &mut CpuContext) -> HleResult {
    let (name_addr, value_addr, overwrite) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as i32 != 0);
    let Some(name) = read_name(ctx, name_addr) else {
        return fail(ctx, EINVAL);
    };
    if value_addr == 0 {
        return fail(ctx, EINVAL);
    }
    let Some(value) = ctx.read_cstring(value_addr, MAX_ENV_TEXT) else {
        return fail(ctx, EINVAL);
    };
    log::debug!("[libc] setenv({name}={value}, overwrite={overwrite})");
    set_var(name, value, overwrite);
    0
}

/// unsetenv(name) -> 0 / -1
fn cb_unsetenv(ctx: &mut CpuContext) -> HleResult {
    let Some(name) = read_name(ctx, ctx.rdi()) else {
        return fail(ctx, EINVAL);
    };
    with_env(|env| env.remove(&name));
    0
}

/// getenv(name) -> char* / NULL
fn cb_getenv(ctx: &mut CpuContext) -> HleResult {
    let Some(name) = ctx.read_cstring(ctx.rdi(), MAX_ENV_TEXT).filter(|_| ctx.rdi() != 0) else {
        return 0;
    };
    let cached = with_env(|env| env.get(&name).map(|v| (v.value.clone(), v.guest)));
    let Some((value, guest)) = cached else { return 0 };
    if let Some(p) = guest {
        return p as HleResult;
    }
    let mut bytes = value.into_bytes();
    bytes.push(0);
    let Some(p) = crate::pthread::alloc_guest_block(ctx, bytes.len() as u64) else {
        crate::kernel::set_errno(ctx, ENOMEM);
        return 0;
    };
    if !ctx.write_bytes(p, &bytes) {
        return 0;
    }
    with_env(|env| {
        if let Some(v) = env.get_mut(&name) {
            v.guest = Some(p);
        }
    });
    p as HleResult
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[("M4YYbSFfJ8g", "setenv", cb_setenv)];

/// Used only when no loaded libc module exports them.
const FALLBACK_EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("smbQukfxYJM", "getenv", cb_getenv),
    ("CRJcH8CnPSI", "unsetenv", cb_unsetenv),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, LIB, cb);
    }
    for &(nid, name, cb) in FALLBACK_EXPORTS {
        reg.register_fallback_fn(nid, name, LIB, cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS.iter().chain(FALLBACK_EXPORTS) {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn setenv_and_unsetenv() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x7A40_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.write_bytes(BASE, b"RPS5E_TEST_VAR\0");
        ctx.write_bytes(BASE + 0x40, b"one\0");
        ctx.write_bytes(BASE + 0x80, b"two\0");
        ctx.write_bytes(BASE + 0xC0, b"A=B\0");
        ctx.set(Rdi, BASE);
        ctx.set(Rsi, BASE + 0x40);
        ctx.set(Rdx, 0);
        assert_eq!(cb_setenv(&mut ctx), 0);
        assert_eq!(get_var("RPS5E_TEST_VAR").as_deref(), Some("one"));
        ctx.set(Rsi, BASE + 0x80);
        assert_eq!(cb_setenv(&mut ctx), 0);
        assert_eq!(get_var("RPS5E_TEST_VAR").as_deref(), Some("one"));
        ctx.set(Rdx, 1);
        assert_eq!(cb_setenv(&mut ctx), 0);
        assert_eq!(get_var("RPS5E_TEST_VAR").as_deref(), Some("two"));
        assert_eq!(cb_unsetenv(&mut ctx), 0);
        assert_eq!(get_var("RPS5E_TEST_VAR"), None);
        // getenv of a missing variable never allocates
        assert_eq!(cb_getenv(&mut ctx), 0);
        ctx.set(Rdi, BASE + 0xC0);
        assert_eq!(cb_setenv(&mut ctx), -1);
        ctx.set(Rdi, 0);
        assert_eq!(cb_setenv(&mut ctx), -1);
        // A null value is rejected rather than stored as empty.
        ctx.set(Rdi, BASE);
        ctx.set(Rsi, 0);
        assert_eq!(cb_setenv(&mut ctx), -1);
        assert_eq!(get_var("RPS5E_TEST_VAR"), None);
    }
}
