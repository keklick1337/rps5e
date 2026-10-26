// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceVoice without voice transport.
//!
//! Ports are bookkeeping only: 254 port ids handed out round-robin, each
//! with its type, bit rate, volume, mute flag and connection state. Output ports
//! deliver no data and input ports accept and drop whatever is written.

use super::Export;
use crate::context::CpuContext;
use crate::registry::HleResult;
use std::sync::Mutex;

pub const VOICE_ERROR_ARGUMENT_INVALID: HleResult = 0x804E_0805u32 as i32 as i64;
pub const VOICE_ERROR_RESOURCE_INSUFFICIENT: HleResult = 0x804E_0808u32 as i32 as i64;
/// The console's answer is not known (kernel ENOENT, as an unimplemented import).
const ERROR_NOT_IMPLEMENTED: HleResult = 0x8002_0002u32 as i32 as i64;

/// SceVoiceResourceInfo: u16 maxVoiceInputPorts, maxVoiceOutputPorts,
/// maxDeviceInputPorts, maxDeviceOutputPorts, maxPorts.
pub const RESOURCE_INFO: [u16; 5] = [7, 7, 1, 1, 64];

const PORTS: usize = 256;
/// Not a port: the id space ends here.
const INVALID_PORT_ID: u32 = 0xFF;
const DEFAULT_BITRATE: u32 = 48_000;
const PORT_TYPE_OUT_VOICE: i32 = 2;
pub const PORT_ATTR_AUDIO_FLAG: i32 = 1001;
pub const PORT_ATTR_ENERGY_LEVEL: i32 = 1003;

/// SceVoicePortParam: i32 portType, u16 threshold, u16 bMute, float volume,
/// then a union (voice: i32 bitrate).
const PORT_PARAM_SIZE: usize = 0x18;
/// SceVoiceBasePortInfo: i32 portType, i32 state, u32 *pEdge, u32 numByte,
/// u32 frameSize, u16 numEdge, u16 reserved.
const PORT_INFO_SIZE: usize = 0x20;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PortParam {
    pub port_type: i32,
    pub muted: bool,
    pub volume: f32,
    pub bitrate: i32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Port {
    pub used: bool,
    pub port_type: i32,
    pub bitrate: u32,
    pub frame_size: u32,
    pub byte_count: u32,
    pub edge_count: u16,
    pub volume: f32,
    pub muted: bool,
}

impl Default for Port {
    fn default() -> Self {
        Self {
            used: false,
            port_type: -1,
            bitrate: DEFAULT_BITRATE,
            frame_size: 1,
            byte_count: 0,
            edge_count: 0,
            volume: 1.0,
            muted: false,
        }
    }
}

/// Fields of SceVoiceBasePortInfo besides the caller's edge pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortInfo {
    pub port_type: i32,
    pub state: i32,
    pub byte_count: u32,
    pub frame_size: u32,
    pub edge_count: u16,
}

pub struct Voice {
    ports: [Port; PORTS],
    next_id: u32,
    started: bool,
    /// Between sceVoiceInit and sceVoiceEnd.
    initialized: bool,
    /// sceVoiceEnableChat / sceVoiceDisableChat.
    chat_enabled: bool,
}

impl Default for Voice {
    fn default() -> Self {
        Self::new()
    }
}

impl Voice {
    pub const fn new() -> Self {
        const FREE: Port = Port {
            used: false,
            port_type: -1,
            bitrate: DEFAULT_BITRATE,
            frame_size: 1,
            byte_count: 0,
            edge_count: 0,
            volume: 1.0,
            muted: false,
        };
        Self { ports: [FREE; PORTS], next_id: 1, started: false, initialized: false, chat_enabled: false }
    }

    pub fn set_initialized(&mut self, initialized: bool) {
        self.initialized = initialized;
    }

    pub fn initialized(&self) -> bool {
        self.initialized
    }

    pub fn port_exists(&mut self, id: u32) -> bool {
        self.port(id).is_some()
    }

    /// Init and End: drop every port.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn set_started(&mut self, started: bool) {
        self.started = started;
    }

    fn port(&mut self, id: u32) -> Option<&mut Port> {
        if id as usize >= PORTS || id == INVALID_PORT_ID {
            return None;
        }
        let p = &mut self.ports[id as usize];
        p.used.then_some(p)
    }

    /// Next free id after the last one handed out; None when all are taken.
    pub fn create_port(&mut self, param: Option<&PortParam>) -> Option<u32> {
        for _ in 0..PORTS - 1 {
            let id = self.next_id;
            self.next_id += 1;
            if self.next_id == INVALID_PORT_ID {
                self.next_id = 1;
            }
            let p = &mut self.ports[id as usize];
            if p.used {
                continue;
            }
            *p = Port { used: true, port_type: param.map_or(-1, |p| p.port_type), ..Port::default() };
            if let Some(param) = param {
                if param.volume > 0.0 {
                    p.volume = param.volume;
                }
                p.muted = param.muted;
                if param.port_type == PORT_TYPE_OUT_VOICE && param.bitrate > 0 {
                    p.bitrate = param.bitrate as u32;
                }
            }
            return Some(id);
        }
        None
    }

    pub fn delete_port(&mut self, id: u32) {
        if let Some(p) = self.port(id) {
            *p = Port::default();
        }
    }

    /// Connecting marks both ends with one edge; disconnecting clears it.
    pub fn set_connected(&mut self, input: u32, output: u32, connected: bool) {
        let edges = connected as u16;
        for id in [input, output] {
            if let Some(p) = self.port(id) {
                p.edge_count = edges;
            }
        }
    }

    pub fn port_info(&mut self, id: u32) -> PortInfo {
        let started = self.started;
        match self.port(id) {
            Some(p) => PortInfo {
                port_type: p.port_type,
                state: started as i32,
                byte_count: p.byte_count,
                frame_size: if p.frame_size != 0 { p.frame_size } else { 1 },
                edge_count: p.edge_count,
            },
            None => PortInfo { port_type: -1, state: started as i32, byte_count: 0, frame_size: 1, edge_count: 0 },
        }
    }

    pub fn bitrate(&mut self, id: u32) -> u32 {
        self.port(id).map_or(DEFAULT_BITRATE, |p| if p.bitrate != 0 { p.bitrate } else { DEFAULT_BITRATE })
    }

    pub fn set_volume(&mut self, id: u32, volume: f32) {
        if let Some(p) = self.port(id) {
            p.volume = volume;
        }
    }

    pub fn volume(&mut self, id: u32) -> f32 {
        self.port(id).map_or(1.0, |p| p.volume)
    }

    pub fn set_muted(&mut self, id: u32, muted: bool) {
        if let Some(p) = self.port(id) {
            p.muted = muted;
        }
    }

    pub fn muted(&mut self, id: u32) -> bool {
        self.port(id).is_some_and(|p| p.muted)
    }
}

static VOICE: Mutex<Voice> = Mutex::new(Voice::new());

fn voice() -> std::sync::MutexGuard<'static, Voice> {
    VOICE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Zero `len` guest bytes at `addr`.
fn zero_guest(ctx: &CpuContext, addr: u64, len: u64) -> bool {
    static ZEROS: [u8; 4096] = [0; 4096];
    let mut done = 0;
    while done < len {
        let n = (len - done).min(ZEROS.len() as u64);
        if !ctx.write_bytes(addr + done, &ZEROS[..n as usize]) {
            return false;
        }
        done += n;
    }
    true
}

const VOICE_ERROR_NOT_INITIALIZED: HleResult = 0x804E_0801u32 as i32 as i64;
const VOICE_ERROR_ALREADY_INITIALIZED: HleResult = 0x804E_0802u32 as i32 as i64;

/// sceVoiceInit(SceVoiceInitParam *, SceVoiceVersion): once until
/// sceVoiceEnd (the module's ALREADY_INITIALIZED after that).
fn cb_init(_ctx: &mut CpuContext) -> HleResult {
    let mut v = voice();
    if v.initialized() {
        return VOICE_ERROR_ALREADY_INITIALIZED;
    }
    v.reset();
    v.set_initialized(true);
    0
}

/// sceVoicePausePortAll() / sceVoiceResumePortAll(): ports carry no audio,
/// so there is nothing to pause; the library must be initialized.
fn cb_pause_or_resume_all(_ctx: &mut CpuContext) -> HleResult {
    if voice().initialized() { 0 } else { VOICE_ERROR_NOT_INITIALIZED }
}

/// sceVoiceGetResourceInfo(SceVoiceResourceInfo *)
fn cb_get_resource_info(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    if !voice().initialized() {
        log::error!("[AUDIO] sceVoiceGetResourceInfo before sceVoiceInit is not implemented");
        return ERROR_NOT_IMPLEMENTED;
    }
    let bytes: Vec<u8> = RESOURCE_INFO.iter().flat_map(|v| v.to_le_bytes()).collect();
    if !ctx.write_bytes(out, &bytes) {
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    0
}

/// sceVoiceEnableChat() / sceVoiceDisableChat(): a flag once initialized;
/// before sceVoiceInit the console's answer is not known.
fn set_chat(on: bool, name: &str) -> HleResult {
    let mut v = voice();
    if !v.initialized() {
        log::error!("[AUDIO] {name} before sceVoiceInit is not implemented");
        return ERROR_NOT_IMPLEMENTED;
    }
    v.chat_enabled = on;
    0
}

fn cb_enable_chat(_ctx: &mut CpuContext) -> HleResult {
    set_chat(true, "sceVoiceEnableChat")
}

fn cb_disable_chat(_ctx: &mut CpuContext) -> HleResult {
    set_chat(false, "sceVoiceDisableChat")
}

/// sceVoiceResetPort(portId): ports hold no stream state, so a known port
/// is left as it is.
fn cb_reset_port(ctx: &mut CpuContext) -> HleResult {
    let id = ctx.rdi() as u32;
    if !voice().port_exists(id) {
        log::error!("[AUDIO] sceVoiceResetPort: unknown port {id}");
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    0
}

fn cb_ok(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// sceVoiceCreatePort(u32 *portId, const SceVoicePortParam *param)
fn cb_create_port(ctx: &mut CpuContext) -> HleResult {
    let (out, param_addr) = (ctx.rdi(), ctx.rsi());
    if !voice().initialized() {
        return VOICE_ERROR_NOT_INITIALIZED;
    }
    if out == 0 {
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    let param = if param_addr != 0 {
        let mut raw = [0u8; PORT_PARAM_SIZE];
        if ctx.vm.read(param_addr, &mut raw).is_err() {
            return VOICE_ERROR_ARGUMENT_INVALID;
        }
        Some(PortParam {
            port_type: i32::from_le_bytes(raw[0..4].try_into().unwrap()),
            muted: u16::from_le_bytes([raw[6], raw[7]]) != 0,
            volume: f32::from_le_bytes(raw[8..12].try_into().unwrap()),
            bitrate: i32::from_le_bytes(raw[12..16].try_into().unwrap()),
        })
    } else {
        None
    };
    let Some(id) = voice().create_port(param.as_ref()) else {
        return VOICE_ERROR_RESOURCE_INSUFFICIENT;
    };
    if !ctx.write_u32(out, id) {
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    0
}

fn cb_connect(ctx: &mut CpuContext) -> HleResult {
    voice().set_connected(ctx.rdi() as u32, ctx.rsi() as u32, true);
    0
}

fn cb_disconnect(ctx: &mut CpuContext) -> HleResult {
    voice().set_connected(ctx.rdi() as u32, ctx.rsi() as u32, false);
    0
}

fn cb_delete_port(ctx: &mut CpuContext) -> HleResult {
    voice().delete_port(ctx.rdi() as u32);
    0
}

fn cb_end(_ctx: &mut CpuContext) -> HleResult {
    voice().reset();
    0
}

fn cb_start(_ctx: &mut CpuContext) -> HleResult {
    let mut v = voice();
    if !v.initialized() {
        return VOICE_ERROR_NOT_INITIALIZED;
    }
    v.set_started(true);
    0
}

fn cb_stop(_ctx: &mut CpuContext) -> HleResult {
    voice().set_started(false);
    0
}

/// sceVoiceReadFromOPort(portId, void *data, u32 *size): no data, ever.
fn cb_read_from_oport(ctx: &mut CpuContext) -> HleResult {
    let (data, size_addr) = (ctx.rsi(), ctx.rdx());
    if size_addr == 0 {
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    let Some(size) = ctx.read_u32(size_addr) else {
        return VOICE_ERROR_ARGUMENT_INVALID;
    };
    if data != 0 && size != 0 && !zero_guest(ctx, data, size as u64) {
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    if !ctx.write_u32(size_addr, 0) {
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    0
}

/// sceVoiceWriteToIPort(portId, const void *data, u32 *size, i16 frameGaps)
fn cb_write_to_iport(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdx() == 0 {
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    0
}

/// sceVoiceGetPortInfo(portId, SceVoiceBasePortInfo *info): the caller's edge pointer stays.
fn cb_get_port_info(ctx: &mut CpuContext) -> HleResult {
    let (id, addr) = (ctx.rdi() as u32, ctx.rsi());
    if !voice().initialized() {
        return VOICE_ERROR_NOT_INITIALIZED;
    }
    if addr == 0 {
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    let Some(edge) = ctx.read_u64(addr + 8) else {
        return VOICE_ERROR_ARGUMENT_INVALID;
    };
    let info = voice().port_info(id);
    let mut raw = [0u8; PORT_INFO_SIZE];
    raw[0..4].copy_from_slice(&info.port_type.to_le_bytes());
    raw[4..8].copy_from_slice(&info.state.to_le_bytes());
    raw[8..16].copy_from_slice(&edge.to_le_bytes());
    raw[16..20].copy_from_slice(&info.byte_count.to_le_bytes());
    raw[20..24].copy_from_slice(&info.frame_size.to_le_bytes());
    raw[24..26].copy_from_slice(&info.edge_count.to_le_bytes());
    if !ctx.write_bytes(addr, &raw) {
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    0
}

/// sceVoiceGetPortAttr(portId, SceVoicePortAttr attr, void *value, i32 size)
fn cb_get_port_attr(ctx: &mut CpuContext) -> HleResult {
    let (attr, value, size) = (ctx.rsi() as i32, ctx.rdx(), ctx.rcx() as i32);
    if value == 0 || size <= 0 {
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    if !zero_guest(ctx, value, size as u64) {
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    // Audio is always flagged present; the energy level stays zero.
    if attr == PORT_ATTR_AUDIO_FLAG && !ctx.write_bytes(value, &[1]) {
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    0
}

/// sceVoiceGetBitRate(portId, u32 *bitrate)
fn cb_get_bit_rate(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    let bitrate = voice().bitrate(ctx.rdi() as u32);
    if !ctx.write_u32(out, bitrate) {
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    0
}

/// sceVoiceSetVolume(portId, float volume [xmm0])
fn cb_set_volume(ctx: &mut CpuContext) -> HleResult {
    let volume = ps5cpu::exec::trap_xmm_low(0).map_or(1.0, |v| f32::from_bits(v as u32));
    voice().set_volume(ctx.rdi() as u32, volume);
    0
}

/// sceVoiceGetVolume(portId, float *volume)
fn cb_get_volume(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    let volume = voice().volume(ctx.rdi() as u32);
    if !ctx.write_f32(out, volume) {
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    0
}

/// sceVoiceSetMuteFlag(portId, bool muted)
fn cb_set_mute_flag(ctx: &mut CpuContext) -> HleResult {
    voice().set_muted(ctx.rdi() as u32, ctx.rsi() as u8 != 0);
    0
}

/// sceVoiceGetMuteFlag(portId, bool *muted)
fn cb_get_mute_flag(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rsi();
    if out == 0 {
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    let muted = voice().muted(ctx.rdi() as u32);
    if !ctx.write_bytes(out, &[muted as u8]) {
        return VOICE_ERROR_ARGUMENT_INVALID;
    }
    0
}

const LIB: &str = "libSceVoice";

pub(crate) static EXPORTS: &[Export] = &[
    ("9TrhuGzberQ", "sceVoiceInit", LIB, cb_init),
    ("Dinob0yMRl8", "sceVoicePausePortAll", LIB, cb_pause_or_resume_all),
    ("jbkJFmOZ9U0", "sceVoiceResumePortAll", LIB, cb_pause_or_resume_all),
    ("Z6QV6j7igvE", "sceVoiceGetResourceInfo", LIB, cb_get_resource_info),
    ("lLkJVewQK68", "sceVoiceEnableChat", LIB, cb_enable_chat),
    ("xHSNCLrQq3A", "sceVoiceDisableChat", LIB, cb_disable_chat),
    ("udAxvCePkUs", "sceVoiceResetPort", LIB, cb_reset_port),
    ("clyKUyi3RYU", "sceVoiceSetThreadsParams", LIB, cb_ok),
    ("nXpje5yNpaE", "sceVoiceCreatePort", LIB, cb_create_port),
    ("oV9GAdJ23Gw", "sceVoiceConnectIPortToOPort", LIB, cb_connect),
    ("ajVj3QG2um4", "sceVoiceDisconnectIPortFromOPort", LIB, cb_disconnect),
    ("b7kJI+nx2hg", "sceVoiceDeletePort", LIB, cb_delete_port),
    ("Oo0S5PH7FIQ", "sceVoiceEnd", LIB, cb_end),
    ("54phPH2LZls", "sceVoiceStart", LIB, cb_start),
    ("Ao2YNSA7-Qo", "sceVoiceStop", LIB, cb_stop),
    ("cQ6DGsQEjV4", "sceVoiceReadFromOPort", LIB, cb_read_from_oport),
    ("YeJl6yDlhW0", "sceVoiceWriteToIPort", LIB, cb_write_to_iport),
    ("CrLqDwWLoXM", "sceVoiceGetPortInfo", LIB, cb_get_port_info),
    ("elcxZTEfHZM", "sceVoiceGetPortAttr", LIB, cb_get_port_attr),
    ("cJLufzou6bc", "sceVoiceGetBitRate", LIB, cb_get_bit_rate),
    ("QBFoAIjJoXQ", "sceVoiceSetVolume", LIB, cb_set_volume),
    ("jjkCjneOYSs", "sceVoiceGetVolume", LIB, cb_get_volume),
    ("gwUynkEgNFY", "sceVoiceSetMuteFlag", LIB, cb_set_mute_flag),
    ("Pc4z1QjForU", "sceVoiceGetMuteFlag", LIB, cb_get_mute_flag),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister;

    const BASE: u64 = 0x5CE5_0000_0000;

    #[test]
    fn port_ids_and_parameters() {
        let mut v = Voice::new();
        let out = PortParam { port_type: PORT_TYPE_OUT_VOICE, muted: false, volume: 0.5, bitrate: 16_000 };
        assert_eq!(v.create_port(Some(&out)), Some(1));
        assert_eq!(v.create_port(Some(&PortParam { port_type: 0, muted: true, volume: 0.0, bitrate: 9 })), Some(2));
        assert_eq!(v.create_port(None), Some(3));
        assert_eq!((v.bitrate(1), v.volume(1)), (16_000, 0.5));
        assert_eq!((v.bitrate(2), v.volume(2)), (DEFAULT_BITRATE, 1.0), "bit rate only for voice out ports");
        assert_eq!(v.port_info(3).port_type, -1);
        // Unknown ports report defaults.
        assert_eq!((v.bitrate(200), v.volume(200)), (DEFAULT_BITRATE, 1.0));
        assert_eq!(v.port_info(INVALID_PORT_ID), PortInfo { port_type: -1, state: 0, byte_count: 0, frame_size: 1, edge_count: 0 });

        v.set_connected(1, 2, true);
        assert_eq!((v.port_info(1).edge_count, v.port_info(2).edge_count), (1, 1));
        v.set_connected(1, 2, false);
        assert_eq!(v.port_info(1).edge_count, 0);
        v.set_started(true);
        assert_eq!(v.port_info(1).state, 1);
        v.set_volume(2, 0.25);
        assert_eq!(v.volume(2), 0.25);
        // The mute flag is kept per port, starting from bMute.
        assert!(!v.muted(1) && v.muted(2) && !v.muted(3));
        v.set_muted(1, true);
        v.set_muted(2, false);
        assert!(v.muted(1) && !v.muted(2));
        v.set_muted(200, true);
        assert!(!v.muted(200), "an unknown port kept a mute flag");

        // Deleted ids are not reused until the counter wraps.
        v.delete_port(1);
        assert_eq!(v.create_port(None), Some(4));
        for id in 5..INVALID_PORT_ID {
            assert_eq!(v.create_port(None), Some(id));
        }
        assert_eq!(v.create_port(None), Some(1), "id counter wraps past 0xFE");
        assert_eq!(v.create_port(None), None, "all 254 ids in use");
        v.reset();
        assert_eq!(v.create_port(None), Some(1));
        assert_eq!(v.port_info(1).state, 0);
    }

    #[test]
    fn exports_through_guest_memory() {
        let vm = crate::sync::test_util::vm_arena(BASE);
        let mut ctx = CpuContext::new(vm);
        // Before sceVoiceInit: the module's NOT_INITIALIZED (PS5).
        ctx.set(CpuRegister::Rdi, 0);
        for cb in [cb_create_port as fn(&mut CpuContext) -> HleResult, cb_start, cb_get_port_info, cb_pause_or_resume_all] {
            assert_eq!(cb(&mut ctx), VOICE_ERROR_NOT_INITIALIZED);
        }
        assert_eq!(cb_init(&mut ctx), 0);
        assert_eq!(cb_init(&mut ctx), VOICE_ERROR_ALREADY_INITIALIZED);
        assert_eq!(cb_pause_or_resume_all(&mut ctx), 0);
        ctx.set(CpuRegister::Rdi, 0);
        assert_eq!(cb_create_port(&mut ctx), VOICE_ERROR_ARGUMENT_INVALID);
        // Voice out port, volume 0.75, 32 kbps.
        ctx.write_u32(BASE + 0x100, PORT_TYPE_OUT_VOICE as u32);
        ctx.write_f32(BASE + 0x108, 0.75);
        ctx.write_u32(BASE + 0x10C, 32_000);
        ctx.set(CpuRegister::Rdi, BASE);
        ctx.set(CpuRegister::Rsi, BASE + 0x100);
        assert_eq!(cb_create_port(&mut ctx), 0);
        let id = ctx.read_u32(BASE).unwrap();
        assert_ne!(id, INVALID_PORT_ID);

        ctx.set(CpuRegister::Rdi, id as u64);
        ctx.set(CpuRegister::Rsi, BASE + 0x200);
        assert_eq!(cb_get_bit_rate(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x200), Some(32_000));
        assert_eq!(cb_get_volume(&mut ctx), 0);
        assert_eq!(ctx.read_f32(BASE + 0x200), Some(0.75));

        // Port info keeps the caller's edge pointer.
        ctx.write_bytes(BASE + 0x300, &[0xAA; PORT_INFO_SIZE]);
        ctx.write_u64(BASE + 0x308, 0x1234_5678);
        ctx.set(CpuRegister::Rsi, BASE + 0x300);
        assert_eq!(cb_get_port_info(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x300), Some(PORT_TYPE_OUT_VOICE as u32));
        assert_eq!(ctx.read_u64(BASE + 0x308), Some(0x1234_5678));
        assert_eq!(ctx.read_u32(BASE + 0x314), Some(1));
        assert_eq!(ctx.read_bytes(BASE + 0x318, 8).unwrap(), vec![0; 8]);

        // Output ports read nothing and clear the buffer.
        ctx.write_bytes(BASE + 0x400, &[0x55; 64]);
        ctx.write_u32(BASE + 0x500, 64);
        ctx.set(CpuRegister::Rsi, BASE + 0x400);
        ctx.set(CpuRegister::Rdx, BASE + 0x500);
        assert_eq!(cb_read_from_oport(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x500), Some(0));
        assert_eq!(ctx.read_bytes(BASE + 0x400, 64).unwrap(), vec![0; 64]);
        ctx.set(CpuRegister::Rdx, 0);
        assert_eq!(cb_read_from_oport(&mut ctx), VOICE_ERROR_ARGUMENT_INVALID);
        assert_eq!(cb_write_to_iport(&mut ctx), VOICE_ERROR_ARGUMENT_INVALID);
        ctx.set(CpuRegister::Rdx, BASE + 0x500);
        assert_eq!(cb_write_to_iport(&mut ctx), 0);

        // Port attributes: audio flag set, energy level zero.
        ctx.write_bytes(BASE + 0x600, &[0x77; 8]);
        ctx.set(CpuRegister::Rsi, PORT_ATTR_AUDIO_FLAG as u64);
        ctx.set(CpuRegister::Rdx, BASE + 0x600);
        ctx.set(CpuRegister::Rcx, 4);
        assert_eq!(cb_get_port_attr(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(BASE + 0x600, 8).unwrap(), vec![1, 0, 0, 0, 0x77, 0x77, 0x77, 0x77]);
        ctx.set(CpuRegister::Rsi, PORT_ATTR_ENERGY_LEVEL as u64);
        assert_eq!(cb_get_port_attr(&mut ctx), 0);
        assert_eq!(ctx.read_u32(BASE + 0x600), Some(0));
        ctx.set(CpuRegister::Rcx, 0);
        assert_eq!(cb_get_port_attr(&mut ctx), VOICE_ERROR_ARGUMENT_INVALID);

        // Mute flag through guest memory; a null output is invalid.
        ctx.set(CpuRegister::Rdi, id as u64);
        ctx.set(CpuRegister::Rsi, 1);
        assert_eq!(cb_set_mute_flag(&mut ctx), 0);
        ctx.write_bytes(BASE + 0x700, &[0x55]);
        ctx.set(CpuRegister::Rsi, BASE + 0x700);
        assert_eq!(cb_get_mute_flag(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(BASE + 0x700, 1).unwrap(), vec![1]);
        ctx.set(CpuRegister::Rsi, 0);
        assert_eq!(cb_set_mute_flag(&mut ctx), 0);
        ctx.set(CpuRegister::Rsi, BASE + 0x700);
        assert_eq!(cb_get_mute_flag(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(BASE + 0x700, 1).unwrap(), vec![0]);
        ctx.set(CpuRegister::Rsi, 0);
        assert_eq!(cb_get_mute_flag(&mut ctx), VOICE_ERROR_ARGUMENT_INVALID);

        // Resetting keeps a known port as it is; an unknown port is invalid.
        ctx.set(CpuRegister::Rdi, id as u64);
        assert_eq!(cb_reset_port(&mut ctx), 0);
        ctx.set(CpuRegister::Rsi, BASE + 0x200);
        assert_eq!(cb_get_volume(&mut ctx), 0);
        assert_eq!(ctx.read_f32(BASE + 0x200), Some(0.75));
        assert_eq!(cb_delete_port(&mut ctx), 0);
        assert_eq!(cb_reset_port(&mut ctx), VOICE_ERROR_ARGUMENT_INVALID, "deleted port");

        // Resource limits once initialized.
        assert_eq!(cb_end(&mut ctx), 0);
        ctx.set(CpuRegister::Rdi, BASE + 0x800);
        assert_eq!(cb_get_resource_info(&mut ctx), ERROR_NOT_IMPLEMENTED, "before init");
        assert_eq!(cb_enable_chat(&mut ctx), ERROR_NOT_IMPLEMENTED, "before init");
        assert_eq!(cb_init(&mut ctx), 0);
        assert_eq!(cb_enable_chat(&mut ctx), 0);
        assert!(voice().chat_enabled);
        assert_eq!(cb_disable_chat(&mut ctx), 0);
        assert!(!voice().chat_enabled);
        ctx.write_bytes(BASE + 0x800, &[0xEE; 12]);
        ctx.set(CpuRegister::Rdi, BASE + 0x800);
        assert_eq!(cb_get_resource_info(&mut ctx), 0);
        assert_eq!(ctx.read_bytes(BASE + 0x800, 12).unwrap(), vec![7, 0, 7, 0, 1, 0, 1, 0, 64, 0, 0xEE, 0xEE]);
        ctx.set(CpuRegister::Rdi, 0);
        assert_eq!(cb_get_resource_info(&mut ctx), VOICE_ERROR_ARGUMENT_INVALID);
        assert_eq!(cb_end(&mut ctx), 0);
        ctx.set(CpuRegister::Rdi, BASE + 0x800);
        assert_eq!(cb_get_resource_info(&mut ctx), ERROR_NOT_IMPLEMENTED, "after end");
    }
}
