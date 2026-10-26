// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceVoiceQoS — voice chat endpoints and connections, offline.
//!
//! Local and remote endpoints and the connections between them are real
//! objects with their attributes. Without a network peer and without a
//! captured microphone stream, no outgoing voice packet is ever produced
//! (ReadPacket yields 0 bytes); incoming packets are accepted and dropped,
//! since no remote participant can exist.

use std::collections::{BTreeMap, HashMap};
use std::sync::{LazyLock, Mutex, MutexGuard};

use crate::context::CpuContext;
use crate::registry::{HleCallback, HleRegistry, HleResult};

const LIB: &str = "libSceVoiceQoS";

const fn err(v: u32) -> HleResult {
    v as i32 as i64
}

// libSceVoiceQoS error codes.
const ERR_NOT_INITIALIZED: HleResult = err(0x8023_6001);
const ERR_INVALID_ARGUMENT: HleResult = err(0x8023_6002);
const ERR_ALREADY_INITIALIZED: HleResult = err(0x8023_6003);
const ERR_OUT_OF_RESOURCE: HleResult = err(0x8023_6004);
const ERR_INVALID_ID: HleResult = err(0x8023_6005);
const ERR_ALREADY_CONNECTED: HleResult = err(0x8023_6006);
const ERR_MEMORY_FAULT: HleResult = err(0x8002_000E);

const MAX_LOCAL_ENDPOINTS: usize = 4;
const MAX_REMOTE_ENDPOINTS: usize = 32;
const MAX_CONNECTIONS: usize = 64;

/// Endpoint attribute ids (mic/speaker volume and mute, bit rate, mic
/// usability, silence state, remote mute, speaker destination).
const ATTR_MAX: u32 = 8;
const ATTR_MIC_USABLE: u32 = 5;
const ATTR_SILENT_STATE: u32 = 6;
const MAX_ATTR_SIZE: usize = 16;

#[allow(dead_code)]
struct LocalEndpoint {
    user_id: i32,
    device_in_port: i32,
    device_out_port: i32,
    attributes: HashMap<u32, Vec<u8>>,
}

#[allow(dead_code)]
struct Connection {
    local: i32,
    remote: i32,
    bytes_received: u64,
}

#[derive(Default)]
struct VoiceQoSState {
    initialized: bool,
    app_type: i32,
    locals: BTreeMap<i32, LocalEndpoint>,
    /// Remote endpoints and their attributes.
    remotes: BTreeMap<i32, HashMap<u32, Vec<u8>>>,
    connections: BTreeMap<i32, Connection>,
    next_id: i32,
}

impl VoiceQoSState {
    fn next(&mut self) -> i32 {
        self.next_id = self.next_id.wrapping_add(1).max(1);
        self.next_id
    }
}

static STATE: LazyLock<Mutex<VoiceQoSState>> = LazyLock::new(|| Mutex::new(VoiceQoSState::default()));

fn state() -> MutexGuard<'static, VoiceQoSState> {
    STATE.lock().unwrap_or_else(|p| p.into_inner())
}

/// sceVoiceQoSInit(void* memBlock, uint32_t memSize, int32_t appType)
fn cb_init(ctx: &mut CpuContext) -> HleResult {
    let (mem, size, app_type) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx() as i32);
    if mem == 0 || size == 0 || app_type == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let mut s = state();
    if s.initialized {
        return ERR_ALREADY_INITIALIZED;
    }
    s.initialized = true;
    s.app_type = app_type;
    0
}

/// sceVoiceQoSEnd / sceVoiceQoSTerminate: releases every object.
fn cb_end(_ctx: &mut CpuContext) -> HleResult {
    let mut s = state();
    if !s.initialized {
        return ERR_NOT_INITIALIZED;
    }
    *s = VoiceQoSState::default();
    0
}

fn cb_set_mode(_ctx: &mut CpuContext) -> HleResult {
    if !state().initialized {
        return ERR_NOT_INITIALIZED;
    }
    0
}

/// sceVoiceQoSCreateLocalEndpoint(int* localId, userId, deviceInPort, deviceOutPort)
fn cb_create_local_endpoint(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    let (user_id, in_port, out_port) = (ctx.rsi() as i32, ctx.rdx() as i32, ctx.rcx() as i32);
    let mut s = state();
    if !s.initialized {
        return ERR_NOT_INITIALIZED;
    }
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if s.locals.len() >= MAX_LOCAL_ENDPOINTS {
        return ERR_OUT_OF_RESOURCE;
    }
    let id = s.next();
    if !ctx.write_u32(out, id as u32) {
        return ERR_MEMORY_FAULT;
    }
    s.locals.insert(
        id,
        LocalEndpoint { user_id, device_in_port: in_port, device_out_port: out_port, attributes: HashMap::new() },
    );
    0
}

/// sceVoiceQoSDeleteLocalEndpoint(localId): drops its connections.
fn cb_delete_local_endpoint(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let mut s = state();
    if !s.initialized {
        return ERR_NOT_INITIALIZED;
    }
    if s.locals.remove(&id).is_none() {
        return ERR_INVALID_ID;
    }
    s.connections.retain(|_, c| c.local != id);
    0
}

/// sceVoiceQoSCreateRemoteEndpoint(int* remoteId)
fn cb_create_remote_endpoint(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    let mut s = state();
    if !s.initialized {
        return ERR_NOT_INITIALIZED;
    }
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if s.remotes.len() >= MAX_REMOTE_ENDPOINTS {
        return ERR_OUT_OF_RESOURCE;
    }
    let id = s.next();
    if !ctx.write_u32(out, id as u32) {
        return ERR_MEMORY_FAULT;
    }
    s.remotes.insert(id, HashMap::new());
    0
}

/// sceVoiceQoSDeleteRemoteEndpoint(remoteId): drops its connections.
fn cb_delete_remote_endpoint(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let mut s = state();
    if !s.initialized {
        return ERR_NOT_INITIALIZED;
    }
    if s.remotes.remove(&id).is_none() {
        return ERR_INVALID_ID;
    }
    s.connections.retain(|_, c| c.remote != id);
    0
}

/// sceVoiceQoSConnect(int* connectionId, localId, remoteId)
fn cb_connect(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    let (local, remote) = (ctx.rsi() as i32, ctx.rdx() as i32);
    let mut s = state();
    if !s.initialized {
        return ERR_NOT_INITIALIZED;
    }
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if !s.locals.contains_key(&local) || !s.remotes.contains_key(&remote) {
        return ERR_INVALID_ID;
    }
    if s.connections.values().any(|c| c.local == local && c.remote == remote) {
        return ERR_ALREADY_CONNECTED;
    }
    if s.connections.len() >= MAX_CONNECTIONS {
        return ERR_OUT_OF_RESOURCE;
    }
    let id = s.next();
    if !ctx.write_u32(out, id as u32) {
        return ERR_MEMORY_FAULT;
    }
    s.connections.insert(id, Connection { local, remote, bytes_received: 0 });
    0
}

/// sceVoiceQoSDisconnect(connectionId)
fn cb_disconnect(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let mut s = state();
    if !s.initialized {
        return ERR_NOT_INITIALIZED;
    }
    match s.connections.remove(&id) {
        Some(_) => 0,
        None => ERR_INVALID_ID,
    }
}

/// Default value of an attribute never set: no microphone is usable and
/// the local side is silent; everything else starts at zero.
fn default_attribute(attr: u32, size: usize) -> Vec<u8> {
    let mut v = vec![0u8; size];
    if attr == ATTR_SILENT_STATE && size >= 4 {
        v[..4].copy_from_slice(&1u32.to_le_bytes());
    }
    v
}

/// sceVoiceQoSSetLocalEndpointAttribute(localId, attributeId, const void* value, int size)
fn cb_set_local_endpoint_attribute(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let attr = ctx.rsi() as u32;
    let (value, size) = (ctx.rdx(), ctx.rcx() as i32);
    let mut s = state();
    if !s.initialized {
        return ERR_NOT_INITIALIZED;
    }
    if attr > ATTR_MAX || attr == ATTR_MIC_USABLE || attr == ATTR_SILENT_STATE {
        return ERR_INVALID_ARGUMENT;
    }
    if value == 0 || size <= 0 || size as usize > MAX_ATTR_SIZE {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(bytes) = ctx.read_bytes(value, size as usize) else {
        return ERR_MEMORY_FAULT;
    };
    match s.locals.get_mut(&id) {
        Some(ep) => {
            ep.attributes.insert(attr, bytes);
            0
        }
        None => ERR_INVALID_ID,
    }
}

/// sceVoiceQoSSetRemoteEndpointAttribute(remoteId, attributeId, const void* value, int size):
/// stored with the remote endpoint; a size of 0 or less clears it.
fn cb_set_remote_endpoint_attribute(ctx: &mut CpuContext) -> HleResult {
    let (id, attr, value, size) = (ctx.rdi() as i32, ctx.rsi() as u32, ctx.rdx(), ctx.rcx() as i32);
    let mut s = state();
    if !s.initialized {
        return ERR_NOT_INITIALIZED;
    }
    if value == 0 || size as i64 > MAX_ATTR_SIZE as i64 {
        return ERR_INVALID_ARGUMENT;
    }
    let bytes = if size > 0 {
        match ctx.read_bytes(value, size as usize) {
            Some(b) => Some(b),
            None => return ERR_MEMORY_FAULT,
        }
    } else {
        None
    };
    let Some(attributes) = s.remotes.get_mut(&id) else { return ERR_INVALID_ID };
    match bytes {
        Some(b) => attributes.insert(attr, b),
        None => attributes.remove(&attr),
    };
    0
}

/// sceVoiceQoSGetLocalEndpointAttribute(localId, attributeId, void* value, int size)
fn cb_get_local_endpoint_attribute(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let attr = ctx.rsi() as u32;
    let (value, size) = (ctx.rdx(), ctx.rcx() as i32);
    let s = state();
    if !s.initialized {
        return ERR_NOT_INITIALIZED;
    }
    if attr > ATTR_MAX || value == 0 || size <= 0 || size as usize > MAX_ATTR_SIZE {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(ep) = s.locals.get(&id) else {
        return ERR_INVALID_ID;
    };
    let size = size as usize;
    let mut out = default_attribute(attr, size);
    if let Some(stored) = ep.attributes.get(&attr) {
        let n = stored.len().min(size);
        out[..n].copy_from_slice(&stored[..n]);
    }
    if ctx.write_bytes(value, &out) {
        0
    } else {
        ERR_MEMORY_FAULT
    }
}

/// sceVoiceQoSReadPacket(connectionId, void* data, uint32_t* size): no
/// voice is captured, so there is never an outgoing packet.
fn cb_read_packet(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let (data, size) = (ctx.rsi(), ctx.rdx());
    let s = state();
    if !s.initialized {
        return ERR_NOT_INITIALIZED;
    }
    if data == 0 || size == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if !s.connections.contains_key(&id) {
        return ERR_INVALID_ID;
    }
    if ctx.write_u32(size, 0) {
        0
    } else {
        ERR_MEMORY_FAULT
    }
}

/// sceVoiceQoSGetLocalEndpoint / GetRemoteEndpoint(connectionId, int* id):
/// an endpoint of the connection.
fn connection_endpoint(ctx: &CpuContext, remote: bool) -> HleResult {
    let (id, out) = (ctx.rdi() as i32, ctx.rsi());
    let s = state();
    if !s.initialized {
        return ERR_NOT_INITIALIZED;
    }
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(c) = s.connections.get(&id) else { return ERR_INVALID_ID };
    let endpoint = if remote { c.remote } else { c.local };
    if ctx.write_u32(out, endpoint as u32) { 0 } else { ERR_MEMORY_FAULT }
}

fn cb_get_local_endpoint(ctx: &mut CpuContext) -> HleResult {
    connection_endpoint(ctx, false)
}

fn cb_get_remote_endpoint(ctx: &mut CpuContext) -> HleResult {
    connection_endpoint(ctx, true)
}

/// sceVoiceQoSWritePacket(connectionId, const void* data, uint32_t* size):
/// the packet is consumed (all of it) and dropped.
fn cb_write_packet(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let (data, size_addr) = (ctx.rsi(), ctx.rdx());
    let mut s = state();
    if !s.initialized {
        return ERR_NOT_INITIALIZED;
    }
    if size_addr == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    let Some(size) = ctx.read_u32(size_addr) else {
        return ERR_MEMORY_FAULT;
    };
    if data == 0 && size != 0 {
        return ERR_INVALID_ARGUMENT;
    }
    match s.connections.get_mut(&id) {
        Some(c) => {
            c.bytes_received += size as u64;
            0
        }
        None => ERR_INVALID_ID,
    }
}

/// sceVoiceQoSGetStatus(connectionId, statusType, int* status): nothing is
/// ever ready to send or play.
fn cb_get_status(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as i32;
    let out = ctx.rdx();
    let s = state();
    if !s.initialized {
        return ERR_NOT_INITIALIZED;
    }
    if out == 0 {
        return ERR_INVALID_ARGUMENT;
    }
    if !s.connections.contains_key(&id) {
        return ERR_INVALID_ID;
    }
    if ctx.write_u32(out, 0) {
        0
    } else {
        ERR_MEMORY_FAULT
    }
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("U8IfNl6-Css", "sceVoiceQoSInit", cb_init),
    ("XPC8EyEuvyk", "sceVoiceQoSGetLocalEndpoint", cb_get_local_endpoint),
    ("9tfy4+aDxrM", "sceVoiceQoSGetRemoteEndpoint", cb_get_remote_endpoint),
    ("ATRGkmbolVM", "sceVoiceQoSEnd", cb_end),
    ("FuXenJLkk-c", "sceVoiceQoSTerminate", cb_end),
    ("+0lOiPZjnBI", "sceVoiceQoSSetMode", cb_set_mode),
    ("Trpt2QBZHCI", "sceVoiceQoSGetStatus", cb_get_status),
    ("lvNClhNHzxI", "sceVoiceQoSCreateLocalEndpoint", cb_create_local_endpoint),
    ("kE0kdvcHTiY", "sceVoiceQoSDeleteLocalEndpoint", cb_delete_local_endpoint),
    ("iqQQW2cBmWU", "sceVoiceQoSCreateRemoteEndpoint", cb_create_remote_endpoint),
    ("H4zqFaDhHW4", "sceVoiceQoSDeleteRemoteEndpoint", cb_delete_remote_endpoint),
    ("kLU6hhXsa2A", "sceVoiceQoSConnect", cb_connect),
    ("j9Xt85krooc", "sceVoiceQoSDisconnect", cb_disconnect),
    ("F7wS7FbfumQ", "sceVoiceQoSSetLocalEndpointAttribute", cb_set_local_endpoint_attribute),
    ("eZu2RP0Ma3w", "sceVoiceQoSGetLocalEndpointAttribute", cb_get_local_endpoint_attribute),
    ("cpC-zyHoMik", "sceVoiceQoSSetRemoteEndpointAttribute", cb_set_remote_endpoint_attribute),
    ("PWokFqab5q4", "sceVoiceQoSReadPacket", cb_read_packet),
    ("SpxLratrO1Q", "sceVoiceQoSWritePacket", cb_write_packet),
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

    fn set(ctx: &mut CpuContext, args: &[u64]) {
        for (r, v) in [Rdi, Rsi, Rdx, Rcx].iter().zip(args) {
            ctx.set(*r, *v);
        }
    }

    #[test]
    fn endpoints_and_connections() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5C10_0000_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        set(&mut ctx, &[BASE, 0, 0, 0]);
        assert_eq!(cb_create_local_endpoint(&mut ctx), ERR_NOT_INITIALIZED);
        set(&mut ctx, &[BASE + 0x800, 0x40000, 0x2000_0000]);
        assert_eq!(cb_init(&mut ctx), 0);
        assert_eq!(cb_init(&mut ctx), ERR_ALREADY_INITIALIZED);

        set(&mut ctx, &[BASE, 1, 0, 0]);
        assert_eq!(cb_create_local_endpoint(&mut ctx), 0);
        let local = ctx.read_u32(BASE).unwrap() as u64;
        set(&mut ctx, &[BASE + 4]);
        assert_eq!(cb_create_remote_endpoint(&mut ctx), 0);
        let remote = ctx.read_u32(BASE + 4).unwrap() as u64;
        set(&mut ctx, &[BASE + 8, local, remote]);
        assert_eq!(cb_connect(&mut ctx), 0);
        assert_eq!(cb_connect(&mut ctx), ERR_ALREADY_CONNECTED);
        let conn = ctx.read_u32(BASE + 8).unwrap() as u64;
        set(&mut ctx, &[conn, BASE + 0x40]);
        assert_eq!(cb_get_local_endpoint(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x40), Some(local as u32));
        assert_eq!(cb_get_remote_endpoint(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x40), Some(remote as u32));
        set(&mut ctx, &[conn, 0]);
        assert_eq!(cb_get_remote_endpoint(&mut ctx), ERR_INVALID_ARGUMENT);
        set(&mut ctx, &[0x7777, BASE + 0x40]);
        assert_eq!(cb_get_local_endpoint(&mut ctx), ERR_INVALID_ID);

        ctx.write_u32(BASE + 0x10, 80);
        set(&mut ctx, &[local, 0, BASE + 0x10, 4]);
        assert_eq!(cb_set_local_endpoint_attribute(&mut ctx), 0);
        set(&mut ctx, &[local, 0, BASE + 0x20, 4]);
        assert_eq!(cb_get_local_endpoint_attribute(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x20), Some(80));
        set(&mut ctx, &[local, ATTR_SILENT_STATE as u64, BASE + 0x20, 4]);
        assert_eq!(cb_get_local_endpoint_attribute(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x20), Some(1));
        set(&mut ctx, &[local, ATTR_MIC_USABLE as u64, BASE + 0x10, 4]);
        assert_eq!(cb_set_local_endpoint_attribute(&mut ctx), ERR_INVALID_ARGUMENT);

        ctx.write_u32(BASE + 0x30, 512);
        set(&mut ctx, &[conn, BASE + 0x100, BASE + 0x30]);
        assert_eq!(cb_read_packet(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x30), Some(0));
        ctx.write_u32(BASE + 0x30, 32);
        assert_eq!(cb_write_packet(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x30), Some(32));

        // Remote attributes: stored, cleared by a size of 0 or less.
        ctx.write_u32(BASE + 0x40, 0x1234);
        set(&mut ctx, &[remote, 3, BASE + 0x40, 4]);
        assert_eq!(cb_set_remote_endpoint_attribute(&mut ctx), 0);
        assert_eq!(state().remotes[&(remote as i32)].get(&3), Some(&vec![0x34, 0x12, 0, 0]));
        set(&mut ctx, &[remote, 3, 0, 4]);
        assert_eq!(cb_set_remote_endpoint_attribute(&mut ctx), ERR_INVALID_ARGUMENT);
        set(&mut ctx, &[remote, 3, BASE + 0x40, 0]);
        assert_eq!(cb_set_remote_endpoint_attribute(&mut ctx), 0);
        assert_eq!(state().remotes[&(remote as i32)].get(&3), None);
        set(&mut ctx, &[remote, 3, BASE + 0x40, u32::MAX as u64]);
        assert_eq!(cb_set_remote_endpoint_attribute(&mut ctx), 0, "a negative size clears too");
        set(&mut ctx, &[remote + 100, 3, BASE + 0x40, 4]);
        assert_eq!(cb_set_remote_endpoint_attribute(&mut ctx), ERR_INVALID_ID);

        // Deleting the remote endpoint drops the connection.
        set(&mut ctx, &[remote]);
        assert_eq!(cb_delete_remote_endpoint(&mut ctx), 0);
        set(&mut ctx, &[conn]);
        assert_eq!(cb_disconnect(&mut ctx), ERR_INVALID_ID);
        set(&mut ctx, &[local]);
        assert_eq!(cb_delete_local_endpoint(&mut ctx), 0);
        assert_eq!(cb_end(&mut ctx), 0);
        assert_eq!(cb_end(&mut ctx), ERR_NOT_INITIALIZED);
        set(&mut ctx, &[remote, 3, BASE + 0x40, 4]);
        assert_eq!(cb_set_remote_endpoint_attribute(&mut ctx), ERR_NOT_INITIALIZED);
    }
}
