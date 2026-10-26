// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Host-side NGS2 engine: systems, racks, voices and the render graph.
//!
//! Sampler voices (racks 0x1000 / 0x4001) consume queued waveform blocks
//! (PCM, VAG or ATRAC9), resample them to the system rate with a 32.32
//! fixed-point phase and optionally low-pass filter them. Voices are wired
//! together through ports (sampler -> submixer -> mastering); mastering
//! voices write the mix into the render buffers.
//!
//! Guest memory and guest calls go through [`Env`], so the engine runs
//! unchanged in unit tests against host memory.

use super::filter::{Filter, FilterParam, FILTER_PARAM_SIZE};
use super::reverb::{I3dl2Param, ReverbState, I3DL2_PARAM_SIZE};
use super::vag;
use super::waveform::{
    WAVEFORM_TYPE_ATRAC9, WAVEFORM_TYPE_PCM_F32L, WAVEFORM_TYPE_PCM_F64L, WAVEFORM_TYPE_PCM_I16L,
    WAVEFORM_TYPE_PCM_I24L, WAVEFORM_TYPE_PCM_I32L, WAVEFORM_TYPE_PCM_I8, WAVEFORM_TYPE_PCM_U8,
    WAVEFORM_TYPE_VAG,
};
use crate::audio::atrac9::Atrac9Decoder;
use std::collections::{BTreeMap, VecDeque};

// ===========================================================================
// Guest interface
// ===========================================================================

/// Guest memory and guest code as seen by the engine.
pub(crate) trait Env {
    fn read(&mut self, addr: u64, out: &mut [u8]) -> bool;
    fn write(&mut self, addr: u64, data: &[u8]) -> bool;
    /// Call a guest function (SysV ABI, integer arguments).
    fn call(&mut self, func: u64, args: [u64; 6]) -> u64;
    /// Run a voice callback. The callback may re-enter the engine through
    /// `eng` (e.g. sceNgs2VoiceControl from inside the callback).
    fn voice_callback(&mut self, eng: &mut Engine, func: u64, info: &CallbackInfo);
}

/// SceNgs2VoiceCallbackInfo (56 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CallbackInfo {
    pub data: u64,
    pub voice: u64,
    pub flag: u32,
    pub user: u64,
    pub block_data: u64,
    pub size: u32,
    pub repeats: u32,
    pub attributes: u32,
}

impl CallbackInfo {
    pub fn to_bytes(&self) -> [u8; 56] {
        let mut b = [0u8; 56];
        b[0..8].copy_from_slice(&self.data.to_le_bytes());
        b[8..16].copy_from_slice(&self.voice.to_le_bytes());
        b[16..20].copy_from_slice(&self.flag.to_le_bytes());
        b[24..32].copy_from_slice(&self.user.to_le_bytes());
        b[32..40].copy_from_slice(&self.block_data.to_le_bytes());
        b[40..44].copy_from_slice(&self.size.to_le_bytes());
        b[44..48].copy_from_slice(&self.repeats.to_le_bytes());
        b[48..52].copy_from_slice(&self.attributes.to_le_bytes());
        b
    }
}

fn read_bytes(env: &mut dyn Env, addr: u64, len: usize) -> Option<Vec<u8>> {
    let mut v = vec![0u8; len];
    (len == 0 || (addr != 0 && env.read(addr, &mut v))).then_some(v)
}

fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn le64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

fn put32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}

fn put64(b: &mut [u8], o: usize, v: u64) {
    b[o..o + 8].copy_from_slice(&v.to_le_bytes());
}

// ===========================================================================
// Error codes and layouts
// ===========================================================================

pub(crate) const OK: i32 = 0;
pub(crate) const ERR_INVALID_OUT_ADDRESS: i32 = 0x804A_8010u32 as i32;
pub(crate) const ERR_INVALID_OUT_SIZE: i32 = 0x804A_8011u32 as i32;
pub(crate) const ERR_INVALID_SYSTEM_HANDLE: i32 = 0x804A_8201u32 as i32;
pub(crate) const ERR_INVALID_RACK_HANDLE: i32 = 0x804A_8202u32 as i32;
/// Follows the system/rack handle numbering.
pub(crate) const ERR_INVALID_VOICE_HANDLE: i32 = 0x804A_8203u32 as i32;
pub(crate) const ERR_INVALID_VOICE_INDEX: i32 = 0x804A_8305u32 as i32;
pub(crate) const ERR_INVALID_WAVEFORM_DATA: i32 = 0x804A_8430u32 as i32;
pub(crate) const ERR_INVALID_WAVEFORM_FORMAT: i32 = 0x804A_8431u32 as i32;
pub(crate) const ERR_UNKNOWN_WAVEFORM_FORMAT: i32 = 0x804A_8432u32 as i32;
/// Arguments the library rejects without a dedicated code (kernel EINVAL).
pub(crate) const ERR_INVALID_ARGUMENT: i32 = 0x8002_0016u32 as i32;

pub(crate) const RACK_SAMPLER: u32 = 0x1000;
pub(crate) const RACK_SUBMIXER: u32 = 0x2000;
pub(crate) const RACK_REVERB: u32 = 0x2001;
pub(crate) const RACK_MASTERING: u32 = 0x3000;
/// Largest mastering gain factor (either sign).
const MAX_MASTERING_GAIN: f32 = 32.0;
/// The LFE channel of a 5.1 or 7.1 mastering voice.
const LFE_SOURCE_CHANNEL: usize = 3;
pub(crate) const RACK_CUSTOM_SAMPLER: u32 = 0x4001;
pub(crate) const RACK_CUSTOM_SUBMIXER: u32 = 0x4002;
pub(crate) const RACK_CUSTOM_MASTERING: u32 = 0x4003;

/// Working memory requested per system; the handle is its address.
pub(crate) const SYSTEM_BUFFER_SIZE: u64 = 0x400;
/// Rack working memory: header, then one slot per voice. Voice handles are
/// the slot addresses.
pub(crate) const RACK_HEADER_SIZE: u64 = 0x400;
pub(crate) const VOICE_STRIDE: u64 = 0x100;

/// sizeof(SceNgs2SystemOption).
pub(crate) const SYSTEM_OPTION_SIZE: usize = 144;
/// sizeof(SceNgs2ContextBufferInfo).
pub(crate) const CONTEXT_BUFFER_INFO_SIZE: usize = 64;
/// sizeof(SceNgs2SystemInfo).
pub(crate) const SYSTEM_INFO_SIZE: usize = 184;
/// sizeof(SceNgs2RackInfo).
pub(crate) const RACK_INFO_SIZE: usize = 216;
/// sizeof(SceNgs2VoicePortInfo).
pub(crate) const VOICE_PORT_INFO_SIZE: usize = 24;
/// sizeof(SceNgs2VoiceChannelsInfo).
pub(crate) const VOICE_CHANNELS_INFO_SIZE: usize = 8;
/// sceNgs2VoiceQueryInfo id of SceNgs2VoiceChannelsInfo.
pub(crate) const VOICE_INFO_CHANNELS: u32 = 0x4001;
/// Smallest grain a system or rack renders.
pub(crate) const MIN_GRAIN_SAMPLES: u32 = 64;
/// sizeof(SceNgs2WaveformBlock).
pub(crate) const WAVEFORM_BLOCK_SIZE: usize = 40;
/// Waveform-blocks flag: blocks without data are silence.
pub(crate) const WAVEFORM_BLOCKS_FLAG_SILENCE: u32 = 0x10;
/// sizeof(SceNgs2SamplerVoiceState).
pub(crate) const SAMPLER_VOICE_STATE_SIZE: usize = 56;
pub(crate) const CUSTOM_SAMPLER_VOICE_STATE_SIZE: usize = 48;

const RACK_OPTION_SIZE: usize = 176;
const SAMPLER_RACK_OPTION_SIZE: usize = 216;
const SUBMIXER_RACK_OPTION_SIZE: usize = 200;
const REVERB_RACK_OPTION_SIZE: usize = 184;
const MASTERING_RACK_OPTION_SIZE: usize = 184;
const CUSTOM_RACK_OPTION_SIZE: usize = 1280;
const CUSTOM_SUBMIXER_RACK_OPTION_SIZE: usize = 1288;
const CUSTOM_MASTERING_RACK_OPTION_SIZE: usize = 1288;
const CUSTOM_SAMPLER_RACK_OPTION_SIZE: usize = 1304;
pub(crate) const RACK_OPTION_MAX_SIZE: usize = CUSTOM_SAMPLER_RACK_OPTION_SIZE;

/// Custom module id of a user effect (SceNgs2UserFxOption).
const MODULE_USER_FX: u32 = 0x1f;
/// Custom module id of the biquad filter (driven by voice params 0x40001300 | index).
const MODULE_FILTER: u32 = 0x13;
/// Custom module id of the per-voice process handler (voice params 0x40001d00 | index).
const MODULE_USER_FX_HANDLER: u32 = 0x1d;
/// sizeof(SceNgs2UserFxProcessContext).
const USER_FX_PROCESS_CONTEXT_SIZE: usize = 48;
const MAX_MODULES: usize = 24;
/// Buffers a custom rack voice may keep (SceNgs2CustomRackOption.numBuffers).
const MAX_CUSTOM_BUFFERS: u32 = 4;
/// SceNgs2CustomRackPortInfo array after the module infos (8 bytes each:
/// sourceBufferId, reserved).
const OPT_PORTS: usize = OPT_MODULES + MAX_MODULES * MODULE_INFO_SIZE;

/// Channels of a voice's matrix rows and columns.
pub const MAX_CHANNELS: usize = 8;
/// Matrix levels are stored clamped to this magnitude.
pub const MAX_MATRIX_LEVEL: f32 = 4.0;

// ===========================================================================
// Default channel map
// ===========================================================================

/// Levels of one source channel into mono, stereo, 5.1 and 7.1.
struct DefaultRow {
    mono: f32,
    stereo: [f32; 2],
    surround51: [f32; 6],
    surround71: [f32; 8],
}

const HALF_POWER: f32 = std::f32::consts::FRAC_1_SQRT_2;
const QUARTER_POWER: f32 = std::f32::consts::SQRT_2 / 4.0;

const fn row(mono: f32, stereo: [f32; 2], surround51: [f32; 6], surround71: [f32; 8]) -> DefaultRow {
    DefaultRow { mono, stereo, surround51, surround71 }
}

const MONO_CENTER: DefaultRow = row(1.0, [HALF_POWER, HALF_POWER], [0., 0., 1., 0., 0., 0.], [0., 0., 1., 0., 0., 0., 0., 0.]);
const FRONT_LEFT: DefaultRow = row(HALF_POWER, [1., 0.], [1., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0., 0., 0.]);
const FRONT_RIGHT: DefaultRow = row(HALF_POWER, [0., 1.], [0., 1., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0., 0., 0.]);
const LFE: DefaultRow = row(0.0, [0., 0.], [0., 0., 0., 1., 0., 0.], [0., 0., 0., 1., 0., 0., 0., 0.]);
const QUAD_LEFT: DefaultRow = row(0.5, [HALF_POWER, 0.], [0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 1., 0., 0., 0.]);
const QUAD_RIGHT: DefaultRow = row(0.5, [0., HALF_POWER], [0., 0., 0., 0., 0., 1.], [0., 0., 0., 0., 0., 1., 0., 0.]);
const FIVE_SURROUND_LEFT: DefaultRow = row(0.5, [0., 0.], [0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 1., 0., 0., 0.]);
const FIVE_SURROUND_RIGHT: DefaultRow = row(0.5, [0., 0.], [0., 0., 0., 0., 0., 1.], [0., 0., 0., 0., 0., 1., 0., 0.]);
const SURROUND_LEFT: DefaultRow = row(0.5, [0.5, 0.], [0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 1., 0., 0., 0.]);
const SURROUND_RIGHT: DefaultRow = row(0.5, [0., 0.5], [0., 0., 0., 0., 0., 1.], [0., 0., 0., 0., 0., 1., 0., 0.]);
const BACK_CENTER: DefaultRow = row(
    QUARTER_POWER,
    [QUARTER_POWER, QUARTER_POWER],
    [0., 0., 0., 0., HALF_POWER, HALF_POWER],
    [0., 0., 0., 0., 0., 0., HALF_POWER, HALF_POWER],
);
const BACK_LEFT: DefaultRow = row(0.5, [0.5, 0.], [0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 0., 1., 0.]);
const BACK_RIGHT: DefaultRow = row(0.5, [0., 0.5], [0., 0., 0., 0., 0., 1.], [0., 0., 0., 0., 0., 0., 0., 1.]);

/// Source channels of every width, as the console's library maps them.
const DEFAULT_MAP: [&[DefaultRow]; MAX_CHANNELS] = [
    &[MONO_CENTER],
    &[FRONT_LEFT, FRONT_RIGHT],
    &[FRONT_LEFT, FRONT_RIGHT, LFE],
    &[FRONT_LEFT, FRONT_RIGHT, QUAD_LEFT, QUAD_RIGHT],
    &[FRONT_LEFT, FRONT_RIGHT, MONO_CENTER, FIVE_SURROUND_LEFT, FIVE_SURROUND_RIGHT],
    &[FRONT_LEFT, FRONT_RIGHT, MONO_CENTER, LFE, SURROUND_LEFT, SURROUND_RIGHT],
    &[FRONT_LEFT, FRONT_RIGHT, MONO_CENTER, LFE, SURROUND_LEFT, SURROUND_RIGHT, BACK_CENTER],
    &[FRONT_LEFT, FRONT_RIGHT, MONO_CENTER, LFE, SURROUND_LEFT, SURROUND_RIGHT, BACK_LEFT, BACK_RIGHT],
];

/// Level of source channel `src` (of `src_channels`) into destination
/// channel `dst` (of `dst_channels`) for a port without matrix levels:
/// mono reaches both stereo channels at -3 dB and the centre of 5.1/7.1,
/// stereo folds to mono at -3 dB per side, wider sources fold down with
/// 0.5 and -3 dB gains. None for widths the map does not cover (only 1,
/// 2, 6 and 8 destination channels are known).
pub fn default_level(src_channels: usize, src: usize, dst_channels: usize, dst: usize) -> Option<f32> {
    let r = DEFAULT_MAP.get(src_channels.checked_sub(1)?)?.get(src)?;
    match dst_channels {
        1 => (dst == 0).then_some(r.mono),
        2 => r.stereo.get(dst).copied(),
        6 => r.surround51.get(dst).copied(),
        8 => r.surround71.get(dst).copied(),
        _ => None,
    }
}

// ===========================================================================
// Options
// ===========================================================================

/// SceNgs2SystemOption.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SystemOption {
    pub size: u64,
    pub name: [u8; 64],
    pub flags: u32,
    pub max_grain_samples: u32,
    pub num_grain_samples: u32,
    pub sample_rate: u32,
    pub max_voice_channels: u32,
}

impl Default for SystemOption {
    /// sceNgs2SystemResetOption defaults.
    fn default() -> Self {
        Self {
            size: SYSTEM_OPTION_SIZE as u64,
            name: [0; 64],
            flags: 0,
            max_grain_samples: 512,
            num_grain_samples: 256,
            sample_rate: 48000,
            max_voice_channels: 0,
        }
    }
}

impl SystemOption {
    pub fn parse(b: &[u8; SYSTEM_OPTION_SIZE]) -> Self {
        Self {
            size: le64(b, 0),
            name: b[8..72].try_into().unwrap(),
            flags: le32(b, 104),
            max_grain_samples: le32(b, 108),
            num_grain_samples: le32(b, 112),
            sample_rate: le32(b, 116),
            max_voice_channels: le32(b, 120),
        }
    }

    pub fn to_bytes(&self) -> [u8; SYSTEM_OPTION_SIZE] {
        let mut b = [0u8; SYSTEM_OPTION_SIZE];
        put64(&mut b, 0, self.size);
        b[8..72].copy_from_slice(&self.name);
        put32(&mut b, 104, self.flags);
        put32(&mut b, 108, self.max_grain_samples);
        put32(&mut b, 112, self.num_grain_samples);
        put32(&mut b, 116, self.sample_rate);
        put32(&mut b, 120, self.max_voice_channels);
        b
    }
}

/// SceNgs2ContextBufferInfo: hostBuffer@0, hostBufferSize@8, reserved[5],
/// userData@56.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ContextBuffer {
    pub host_buffer: u64,
    pub size: u64,
    pub user_data: u64,
}

impl ContextBuffer {
    pub fn parse(b: &[u8]) -> Self {
        Self { host_buffer: le64(b, 0), size: le64(b, 8), user_data: le64(b, 56) }
    }

    pub fn to_bytes(&self) -> [u8; CONTEXT_BUFFER_INFO_SIZE] {
        let mut b = [0u8; CONTEXT_BUFFER_INFO_SIZE];
        put64(&mut b, 0, self.host_buffer);
        put64(&mut b, 8, self.size);
        put64(&mut b, 56, self.user_data);
        b
    }
}

/// SceNgs2BufferAllocator: allocHandler@0, freeHandler@8, userData@16.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Allocator {
    pub alloc: u64,
    pub free: u64,
    pub user_data: u64,
}

impl Allocator {
    pub fn parse(b: &[u8]) -> Self {
        Self { alloc: le64(b, 0), free: le64(b, 8), user_data: le64(b, 16) }
    }
}

/// Raw rack option image (the union of every rack option struct).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RackOption {
    pub bytes: Vec<u8>,
}

const OPT_MAX_GRAIN: usize = 76;
const OPT_MAX_VOICES: usize = 80;
const OPT_MAX_DELAY: usize = 84;
const OPT_MAX_MATRICES: usize = 88;
const OPT_MAX_PORTS: usize = 92;
const OPT_MAX_VOICE_CHANNELS: usize = 96;
const OPT_EXT: usize = RACK_OPTION_SIZE;
const OPT_CUSTOM_EXT: usize = CUSTOM_RACK_OPTION_SIZE;
const OPT_MODULES: usize = 192;
const MODULE_INFO_SIZE: usize = 40;

impl RackOption {
    pub fn from_bytes(src: &[u8]) -> Self {
        let mut bytes = vec![0u8; RACK_OPTION_MAX_SIZE];
        let n = src.len().min(RACK_OPTION_MAX_SIZE);
        bytes[..n].copy_from_slice(&src[..n]);
        Self { bytes }
    }

    fn u32_at(&self, o: usize) -> u32 {
        le32(&self.bytes, o)
    }

    fn set(&mut self, o: usize, v: u32) {
        put32(&mut self.bytes, o, v);
    }

    pub fn size(&self) -> u64 {
        le64(&self.bytes, 0)
    }
    pub fn max_voices(&self) -> u32 {
        self.u32_at(OPT_MAX_VOICES)
    }
    #[cfg(test)]
    pub fn set_max_voices(&mut self, v: u32) {
        self.set(OPT_MAX_VOICES, v)
    }
    pub fn max_matrices(&self) -> u32 {
        self.u32_at(OPT_MAX_MATRICES)
    }
    pub fn max_ports(&self) -> u32 {
        self.u32_at(OPT_MAX_PORTS)
    }
    #[cfg(test)]
    pub fn set_max_ports(&mut self, v: u32) {
        self.set(OPT_MAX_PORTS, v)
    }
    /// Common rack option: maxVoiceChannels.
    pub fn max_voice_channels(&self) -> u32 {
        self.u32_at(OPT_MAX_VOICE_CHANNELS)
    }
    #[cfg(test)]
    pub fn set_max_voice_channels(&mut self, v: u32) {
        self.set(OPT_MAX_VOICE_CHANNELS, v)
    }
    /// Sampler rack: maxFilters.
    pub fn sampler_max_filters(&self) -> u32 {
        self.u32_at(OPT_EXT + 16)
    }
    #[cfg(test)]
    pub fn set_sampler_max_filters(&mut self, v: u32) {
        self.set(OPT_EXT + 16, v)
    }
    /// Submixer / reverb / mastering rack: maxChannels.
    pub fn max_channels(&self) -> u32 {
        self.u32_at(OPT_EXT)
    }
    #[cfg(test)]
    pub fn set_max_channels(&mut self, v: u32) {
        self.set(OPT_EXT, v)
    }
    /// Custom racks: stateSize, numBuffers, numModules.
    #[cfg(test)]
    pub fn custom_state_size(&self) -> u32 {
        self.u32_at(OPT_EXT)
    }
    #[cfg(test)]
    pub fn set_custom_state_size(&mut self, v: u32) {
        self.set(OPT_EXT, v)
    }
    pub fn custom_num_buffers(&self) -> u32 {
        self.u32_at(OPT_EXT + 4)
    }
    pub fn custom_num_modules(&self) -> u32 {
        self.u32_at(OPT_EXT + 8).min(MAX_MODULES as u32)
    }
    /// Custom racks: the buffer output port `port` sends.
    pub fn custom_port_buffer(&self, port: usize) -> u32 {
        self.u32_at(OPT_PORTS + port * 8)
    }
    #[cfg(test)]
    pub fn set_custom_num_buffers(&mut self, v: u32) {
        self.set(OPT_EXT + 4, v)
    }
    fn module(&self, m: usize) -> ModuleInfo {
        let o = OPT_MODULES + m * MODULE_INFO_SIZE;
        ModuleInfo {
            option: le64(&self.bytes, o),
            module_id: self.u32_at(o + 8),
            source_buffer_id: self.u32_at(o + 12),
            extra_buffer_id: self.u32_at(o + 16),
            dest_buffer_id: self.u32_at(o + 20),
            state_size: self.u32_at(o + 28),
        }
    }

    /// Defaults the library uses when a rack is created without an option.
    pub fn default_for(rack_id: u32) -> Option<Self> {
        let mut o = Self::from_bytes(&[]);
        let common = |o: &mut Self, size: usize, voices: u32, delay: u32, matrices: u32, ports: u32| {
            put64(&mut o.bytes, 0, size as u64);
            o.set(OPT_MAX_GRAIN, 512);
            o.set(OPT_MAX_VOICES, voices);
            o.set(OPT_MAX_DELAY, delay);
            o.set(OPT_MAX_MATRICES, matrices);
            o.set(OPT_MAX_PORTS, ports);
        };
        match rack_id {
            RACK_SAMPLER => {
                common(&mut o, SAMPLER_RACK_OPTION_SIZE, 256, 0, 1, 8);
                // maxChannelWorks, maxCodecCaches, maxWaveformBlocks,
                // maxEnvelopePoints, maxFilters, maxAtrac9Decoders,
                // maxAtrac9ChannelWorks, maxAjmAtrac9Decoders, numPeakMeterBlocks.
                for (i, v) in [256, 32, 4, 4, 8, 256, 256, 0, 8].into_iter().enumerate() {
                    o.set(OPT_EXT + i * 4, v);
                }
            }
            RACK_SUBMIXER => {
                common(&mut o, SUBMIXER_RACK_OPTION_SIZE, 1, 1, 1, 8);
                for (i, v) in [8, 4, 8, 1, 8].into_iter().enumerate() {
                    o.set(OPT_EXT + i * 4, v);
                }
            }
            RACK_REVERB => {
                common(&mut o, REVERB_RACK_OPTION_SIZE, 1, 1, 1, 8);
                o.set(OPT_EXT, 8);
                o.set(OPT_EXT + 4, 1);
            }
            RACK_MASTERING => {
                common(&mut o, MASTERING_RACK_OPTION_SIZE, 1, 1, 0, 0);
                o.set(OPT_EXT, 8);
                o.set(OPT_EXT + 4, 8);
            }
            RACK_CUSTOM_SUBMIXER => {
                common(&mut o, CUSTOM_SUBMIXER_RACK_OPTION_SIZE, 1, 1, 1, 8);
                o.set(OPT_EXT + 4, 1);
                o.set(OPT_CUSTOM_EXT, 8);
                o.set(OPT_CUSTOM_EXT + 4, 1);
            }
            RACK_CUSTOM_MASTERING => {
                common(&mut o, CUSTOM_MASTERING_RACK_OPTION_SIZE, 1, 1, 0, 0);
                o.set(OPT_EXT + 4, 1);
                o.set(OPT_CUSTOM_EXT, 8);
                o.set(OPT_CUSTOM_EXT + 4, 1);
            }
            RACK_CUSTOM_SAMPLER => {
                common(&mut o, CUSTOM_SAMPLER_RACK_OPTION_SIZE, 256, 0, 1, 8);
                o.set(OPT_EXT + 4, 1);
                for (i, v) in [256, 4, 256, 256, 0, 32].into_iter().enumerate() {
                    o.set(OPT_CUSTOM_EXT + i * 4, v);
                }
            }
            _ => return None,
        }
        Some(o)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ModuleInfo {
    option: u64,
    module_id: u32,
    source_buffer_id: u32,
    /// The side-chain buffer: its channels follow the source's in a user
    /// effect's channel pointers.
    extra_buffer_id: u32,
    dest_buffer_id: u32,
    state_size: u32,
}

/// SceNgs2UserFxOption (72 bytes).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct UserFx {
    setup: u64,
    cleanup: u64,
    control: u64,
    process: u64,
    common_size: u64,
    param_size: u64,
    work_size: u64,
    user_data: u64,
}

impl UserFx {
    fn parse(b: &[u8]) -> Self {
        Self {
            setup: le64(b, 8),
            cleanup: le64(b, 16),
            control: le64(b, 24),
            process: le64(b, 32),
            common_size: le64(b, 40),
            param_size: le64(b, 48),
            work_size: le64(b, 56),
            user_data: le64(b, 64),
        }
    }
}

// ===========================================================================
// Waveform format / blocks
// ===========================================================================

/// SceNgs2WaveformFormat (24 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WaveformFormat {
    pub waveform_type: u32,
    pub num_channels: u32,
    pub sample_rate: u32,
    pub config_data: u32,
    pub frame_offset: u32,
    pub frame_margin: u32,
}

impl WaveformFormat {
    #[cfg(test)]
    pub const fn new(waveform_type: u32, num_channels: u32, sample_rate: u32, config_data: u32) -> Self {
        Self { waveform_type, num_channels, sample_rate, config_data, frame_offset: 0, frame_margin: 0 }
    }

    pub fn parse(b: &[u8]) -> Self {
        Self {
            waveform_type: le32(b, 0),
            num_channels: le32(b, 4),
            sample_rate: le32(b, 8),
            config_data: le32(b, 12),
            frame_offset: le32(b, 16),
            frame_margin: le32(b, 20),
        }
    }

    pub fn config_bytes(&self) -> [u8; 4] {
        self.config_data.to_be_bytes()
    }
}

/// SceNgs2WaveformBlock (40 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WaveformBlock {
    pub data_offset: u64,
    pub data_size: u64,
    pub num_repeats: u32,
    pub num_skip_samples: u32,
    pub num_samples: u32,
    pub reserved: u32,
    pub user_data: u64,
}

impl WaveformBlock {
    pub fn parse(b: &[u8]) -> Self {
        Self {
            data_offset: le64(b, 0),
            data_size: le64(b, 8),
            num_repeats: le32(b, 16),
            num_skip_samples: le32(b, 20),
            num_samples: le32(b, 24),
            reserved: le32(b, 28),
            user_data: le64(b, 32),
        }
    }

    pub fn to_bytes(&self) -> [u8; WAVEFORM_BLOCK_SIZE] {
        let mut b = [0u8; WAVEFORM_BLOCK_SIZE];
        put64(&mut b, 0, self.data_offset);
        put64(&mut b, 8, self.data_size);
        put32(&mut b, 16, self.num_repeats);
        put32(&mut b, 20, self.num_skip_samples);
        put32(&mut b, 24, self.num_samples);
        put32(&mut b, 28, self.reserved);
        put64(&mut b, 32, self.user_data);
        b
    }
}

/// Bytes of one interleaved PCM frame, or None for coded formats.
pub(crate) fn pcm_frame_bytes(waveform_type: u32, channels: u32) -> Option<u64> {
    let bps = match waveform_type {
        WAVEFORM_TYPE_PCM_I8 | WAVEFORM_TYPE_PCM_U8 => 1,
        WAVEFORM_TYPE_PCM_I16L => 2,
        WAVEFORM_TYPE_PCM_I24L => 3,
        WAVEFORM_TYPE_PCM_I32L | WAVEFORM_TYPE_PCM_F32L => 4,
        WAVEFORM_TYPE_PCM_F64L => 8,
        _ => return None,
    };
    Some(bps * channels as u64)
}

/// libstdc++ `std::lerp` for floats: exact at both ends and monotonic.
#[inline(always)]
pub(crate) fn lerp(a: f32, b: f32, t: f32) -> f32 {
    if (a <= 0.0 && b >= 0.0) || (a >= 0.0 && b <= 0.0) {
        return t * b + (1.0 - t) * a;
    }
    if t == 1.0 {
        return b;
    }
    let x = a + t * (b - a);
    if (t > 1.0) == (b > a) {
        if b < x { x } else { b }
    } else if b > x {
        x
    } else {
        b
    }
}

// ===========================================================================
// Voices
// ===========================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum PlayState {
    Empty = 0,
    Playing = 3,
    Paused = 5,
    Stopped = 0xb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RackType {
    Sampler,
    Submixer,
    Mastering,
    Reverb,
    CustomSubmixer,
    CustomMastering,
    CustomSampler,
}

impl RackType {
    pub fn from_id(rack_id: u32) -> Option<(Self, usize)> {
        Some(match rack_id {
            RACK_SAMPLER => (Self::Sampler, SAMPLER_RACK_OPTION_SIZE),
            RACK_SUBMIXER => (Self::Submixer, SUBMIXER_RACK_OPTION_SIZE),
            RACK_REVERB => (Self::Reverb, REVERB_RACK_OPTION_SIZE),
            RACK_MASTERING => (Self::Mastering, MASTERING_RACK_OPTION_SIZE),
            RACK_CUSTOM_SAMPLER => (Self::CustomSampler, CUSTOM_SAMPLER_RACK_OPTION_SIZE),
            RACK_CUSTOM_SUBMIXER => (Self::CustomSubmixer, CUSTOM_SUBMIXER_RACK_OPTION_SIZE),
            RACK_CUSTOM_MASTERING => (Self::CustomMastering, CUSTOM_MASTERING_RACK_OPTION_SIZE),
            _ => return None,
        })
    }

    pub fn id(self) -> u32 {
        match self {
            Self::Sampler => RACK_SAMPLER,
            Self::Submixer => RACK_SUBMIXER,
            Self::Reverb => RACK_REVERB,
            Self::Mastering => RACK_MASTERING,
            Self::CustomSampler => RACK_CUSTOM_SAMPLER,
            Self::CustomSubmixer => RACK_CUSTOM_SUBMIXER,
            Self::CustomMastering => RACK_CUSTOM_MASTERING,
        }
    }

    pub fn is_custom(self) -> bool {
        matches!(self, Self::CustomSubmixer | Self::CustomMastering | Self::CustomSampler)
    }

    pub fn is_sampler(self) -> bool {
        matches!(self, Self::Sampler | Self::CustomSampler)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Port {
    /// Destination voice handle (0 = unpatched).
    pub dest: u64,
    pub input: u32,
    pub volume: f32,
    pub matrix: i32,
}

impl Default for Port {
    fn default() -> Self {
        Self { dest: 0, input: 0, volume: 1.0, matrix: -1 }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Block {
    /// Guest address of the block data (waveform data + dataOffset).
    pub data: u64,
    pub info: WaveformBlock,
    pub cursor: u32,
    pub data_cursor: usize,
    pub repeated_count: u32,
    /// Decoded PCM16 of a VAG block (interleaved).
    pub decoded: Option<Box<[i16]>>,
}

impl Block {
    pub fn new(data: u64, info: WaveformBlock) -> Self {
        Self { data, info, cursor: 0, data_cursor: 0, repeated_count: 0, decoded: None }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Module {
    pub param: Vec<u8>,
    pub work: Vec<u8>,
    pub state: Vec<u8>,
    pub flags: u32,
    pub control_warned: bool,
    pub render_warned: bool,
    /// Module 0x1d: process handler and its three user data words.
    pub handler: u64,
    pub handler_data: [u64; 3],
}

#[derive(Debug)]
pub(crate) struct Voice {
    pub state: PlayState,
    pub state_flags: u32,
    pub callback: u64,
    pub callback_data: u64,
    pub callback_flags: u32,
    pub blocks: VecDeque<Block>,
    pub ports: Vec<Port>,
    pub matrices: Vec<Vec<f32>>,
    pub modules: Vec<Module>,
    /// Planar grain: `channels` runs of `grain` samples.
    pub samples: Vec<f32>,
    pub output_matrix: Vec<f32>,
    /// Mastering gain: the full-band and LFE factors on the output.
    pub fbw_level: f32,
    pub lfe_level: f32,
    pub channels: u32,
    pub output_id: u32,
    pub rendering: bool,
    pub rendered: bool,
    pub has_samples: bool,

    pub waveform_type: u32,
    pub decoder: Option<Atrac9Decoder>,
    pub decoder_config: [u8; 4],
    pub superframe_bytes: usize,
    /// ATRAC9 frame geometry and the decoded superframe being handed out.
    pub at9_frame_samples: u32,
    pub sf_frames: u32,
    pub sf_frame: u32,
    pub sf_frames_left: u32,
    pub sf_bytes_left: usize,
    pub planar: Vec<Vec<i16>>,
    pub decoded_frame: Vec<f32>,
    pub next_decoded_frame: Vec<f32>,
    pub compressed_input: Vec<u8>,
    pub frame_cursor: u32,
    pub frame_samples: u32,
    pub next_frame_samples: u32,
    pub accepts_blocks: bool,
    pub sample_rate: u32,
    pub sample_phase: u64,
    pub sample_step: u64,
    pub decoded_samples: u64,
    pub decoded_bytes: u64,
    pub duration_remaining: u32,
    pub skip_remaining: u32,
    pub waveform_end: u64,
    pub filters: Vec<Filter>,
    /// Reverb rack voices after setup.
    pub reverb: Option<Box<ReverbState>>,

    /// Reused render scratch: source frames and raw guest bytes.
    src_scratch: Vec<f32>,
    byte_scratch: Vec<u8>,
}

impl Default for Voice {
    fn default() -> Self {
        Self {
            state: PlayState::Empty,
            state_flags: 0,
            callback: 0,
            callback_data: 0,
            callback_flags: 0,
            blocks: VecDeque::new(),
            ports: Vec::new(),
            matrices: Vec::new(),
            modules: Vec::new(),
            samples: Vec::new(),
            output_matrix: Vec::new(),
            fbw_level: 1.0,
            lfe_level: 1.0,
            channels: 0,
            output_id: 0,
            rendering: false,
            rendered: false,
            has_samples: false,
            waveform_type: 0,
            decoder: None,
            decoder_config: [0; 4],
            superframe_bytes: 0,
            at9_frame_samples: 0,
            sf_frames: 0,
            sf_frame: 0,
            sf_frames_left: 0,
            sf_bytes_left: 0,
            planar: Vec::new(),
            decoded_frame: Vec::new(),
            next_decoded_frame: Vec::new(),
            compressed_input: Vec::new(),
            frame_cursor: 0,
            frame_samples: 0,
            next_frame_samples: 0,
            accepts_blocks: true,
            sample_rate: 0,
            sample_phase: 0,
            sample_step: 0,
            decoded_samples: 0,
            decoded_bytes: 0,
            duration_remaining: 0,
            skip_remaining: 0,
            waveform_end: 0,
            filters: Vec::new(),
            reverb: None,
            src_scratch: Vec::new(),
            byte_scratch: Vec::new(),
        }
    }
}

impl Voice {
    /// Bytes per source frame used for progress accounting and the read
    /// address (decoded PCM16 for VAG).
    fn frame_bytes(&self) -> u64 {
        pcm_frame_bytes(self.waveform_type, self.channels).unwrap_or(2 * self.channels as u64)
    }

    /// Guest address of the next waveform byte the sampler will read.
    pub fn waveform_data(&self) -> u64 {
        let Some(block) = self.blocks.front() else {
            return self.waveform_end;
        };
        let offset = if self.decoder.is_some() {
            block.data_cursor as u64
        } else if self.waveform_type == WAVEFORM_TYPE_VAG {
            let frame = block.info.num_skip_samples as u64 + block.cursor as u64;
            frame / vag::UNIT_SAMPLES * vag::UNIT_BYTES * self.channels as u64
        } else {
            (block.info.num_skip_samples as u64 + block.cursor as u64) * self.frame_bytes()
        };
        block.data.wrapping_add(offset)
    }

    pub fn reset_setup_state(&mut self) {
        self.set_event(4);
        // A sampler or mixer setup also drops the voice callback.
        self.callback = 0;
        self.callback_data = 0;
        self.callback_flags = 0;
        self.ports.fill(Port::default());
        for m in &mut self.matrices {
            m.clear();
        }
        self.filters.clear();
    }

    /// Sampler setup (param 0x10000000): a zeroed format defers the
    /// configuration — it only clears a voice that was never configured
    /// (no channels yet) and leaves a configured one alone.
    pub fn setup_sampler(&mut self, format: &WaveformFormat) {
        let deferred = *format == WaveformFormat::default();
        if deferred {
            if self.channels == 0 {
                self.reset_setup_state();
            }
            return;
        }
        if format.frame_margin != 0 || format.frame_offset != 0 {
            log::warn!("[ngs2] sampler frame offset/margin ignored: {format:?}");
        }
        self.reset_setup_state();
        self.channels = format.num_channels;
        self.sample_rate = format.sample_rate;
        self.waveform_type = format.waveform_type;
        self.sample_phase = 0;
        self.sample_step = (self.sample_rate as u64) << 32;
        self.decoded_samples = 0;
        self.decoded_bytes = 0;
        self.duration_remaining = 0;
        self.skip_remaining = 0;
        self.waveform_end = 0;
        self.blocks.clear();
        self.decoder = None;
        self.sf_frames_left = 0;
        self.sf_bytes_left = 0;
        self.decoded_frame.clear();
        self.next_decoded_frame.clear();
        self.compressed_input.clear();
        self.frame_cursor = 0;
        self.frame_samples = 0;
        self.next_frame_samples = 0;
        self.accepts_blocks = true;
        if deferred {
            return;
        }
        if format.sample_rate == 0 || format.num_channels == 0 {
            log::warn!("[ngs2] sampler setup without rate/channels: {format:?}");
            self.channels = 0;
            return;
        }
        if format.waveform_type == WAVEFORM_TYPE_ATRAC9 {
            let config = format.config_bytes();
            match Atrac9Decoder::new(&config) {
                Ok(decoder) => {
                    let c = decoder.config();
                    if c.channel_count as u32 != format.num_channels || c.sample_rate as u32 != format.sample_rate {
                        log::warn!("[ngs2] ATRAC9 config {config:02x?} disagrees with format {format:?}");
                        self.channels = 0;
                        return;
                    }
                    let samples = c.superframe_samples as usize;
                    self.superframe_bytes = c.superframe_bytes as usize;
                    self.at9_frame_samples = c.frame_samples as u32;
                    self.sf_frames = c.frames_per_superframe as u32;
                    self.planar = vec![vec![0i16; samples]; self.channels as usize];
                    self.decoded_frame.resize(c.frame_samples as usize * self.channels as usize, 0.0);
                    self.compressed_input.reserve(self.superframe_bytes);
                    self.decoder_config = config;
                    self.decoder = Some(decoder);
                }
                Err(e) => {
                    log::warn!("[ngs2] ATRAC9 config {config:02x?} rejected: {e}");
                    self.channels = 0;
                }
            }
        } else if pcm_frame_bytes(format.waveform_type, 1).is_none() && format.waveform_type != WAVEFORM_TYPE_VAG {
            log::warn!("[ngs2] unsupported sampler waveform type {:#x}", format.waveform_type);
            self.channels = 0;
        }
    }

    /// Matrix levels are stored clamped to -4..4 (NaN stays NaN); no
    /// levels, or more than 8x8, are rejected and leave the matrix as it is.
    pub fn set_matrix(&mut self, index: u32, levels: &[f32]) {
        if levels.is_empty() || levels.len() > MAX_CHANNELS * MAX_CHANNELS {
            log::warn!("[ngs2] matrix {index}: {} levels rejected", levels.len());
            return;
        }
        match self.matrices.get_mut(index as usize) {
            Some(m) => {
                m.clear();
                m.extend(levels.iter().map(|l| l.clamp(-MAX_MATRIX_LEVEL, MAX_MATRIX_LEVEL)));
            }
            None => log::warn!("[ngs2] matrix {index} out of range ({})", self.matrices.len()),
        }
    }

    pub fn set_volume(&mut self, index: u32, volume: f32) {
        match self.ports.get_mut(index as usize) {
            Some(p) => p.volume = volume,
            None => log::warn!("[ngs2] port {index} out of range ({})", self.ports.len()),
        }
    }

    pub fn set_port_matrix(&mut self, index: u32, matrix: i32) {
        match self.ports.get_mut(index as usize) {
            Some(p) => p.matrix = matrix,
            None => log::warn!("[ngs2] port {index} out of range ({})", self.ports.len()),
        }
    }

    pub fn set_event(&mut self, id: u32) {
        match id {
            1 => {
                if matches!(self.state, PlayState::Empty | PlayState::Stopped) {
                    self.state = PlayState::Playing;
                    // Reserve the voice now; playback flags publish after rendering.
                    self.state_flags |= 1;
                }
            }
            2 => {
                if matches!(self.state, PlayState::Playing | PlayState::Paused) {
                    self.state = PlayState::Stopped;
                }
            }
            4 | 8 => {
                self.state = PlayState::Empty;
                if let Some(r) = self.reverb.as_mut() {
                    r.clear();
                }
            }
            16 => {
                if self.state == PlayState::Playing {
                    self.state = PlayState::Paused;
                }
            }
            32 => {
                if self.state == PlayState::Paused {
                    self.state = PlayState::Playing;
                }
            }
            _ => log::warn!("[ngs2] unknown voice event {id:#x}"),
        }
    }

    fn playing(&self) -> bool {
        self.state == PlayState::Playing
    }
}

// ===========================================================================
// Systems and racks
// ===========================================================================

#[derive(Debug)]
pub(crate) struct System {
    pub option: SystemOption,
    pub buffer: ContextBuffer,
    pub allocator: Allocator,
    pub uid: u32,
    pub render_count: u64,
    pub user_data: u64,
    /// Renders of this system running on the calling thread (a voice
    /// callback may render again).
    pub render_depth: u32,
}

#[derive(Debug)]
pub(crate) struct Rack {
    pub handle: u64,
    pub system: u64,
    pub ty: RackType,
    pub option: RackOption,
    pub buffer: ContextBuffer,
    pub allocator: Allocator,
    pub voices: Vec<Voice>,
    pub common: Vec<Vec<u8>>,
    pub fx: Vec<UserFx>,
    pub uid: u32,
    /// sceNgs2RackSetUserData.
    pub user_data: u64,
}

/// Voice identity inside the engine: rack slot and voice index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Vid {
    pub slot: usize,
    pub idx: usize,
}

/// Routing edge: `src` port `port` feeds the voice whose handle is `dest`.
#[derive(Clone, Copy, Debug)]
struct Edge {
    dest: u64,
    seq: u32,
    src: Vid,
    port: u32,
}

/// Render buffer descriptor (SceNgs2RenderBufferInfo).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RenderBuffer {
    pub buffer: u64,
    pub size: u64,
    pub waveform_type: u32,
    pub num_channels: u32,
}

#[derive(Default)]
struct Scratch {
    voices: Vec<Vid>,
    edges: Vec<Edge>,
    out_mix: Vec<Vec<f32>>,
    out_used: Vec<bool>,
    bytes: Vec<u8>,
}

pub(crate) struct Engine {
    pub systems: BTreeMap<u64, System>,
    pub racks: Vec<Option<Rack>>,
    rack_by_handle: BTreeMap<u64, usize>,
    /// Rack slots, newest first.
    rack_order: Vec<usize>,
    next_uid: u32,
    next_rack_uid: u32,
    scratch: Scratch,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub const fn new() -> Self {
        Self {
            systems: BTreeMap::new(),
            racks: Vec::new(),
            rack_by_handle: BTreeMap::new(),
            rack_order: Vec::new(),
            next_uid: 1,
            next_rack_uid: 1,
            scratch: Scratch { voices: Vec::new(), edges: Vec::new(), out_mix: Vec::new(), out_used: Vec::new(), bytes: Vec::new() },
        }
    }

    // ---- lookup ----------------------------------------------------------

    pub fn voice(&self, v: Vid) -> Option<&Voice> {
        self.racks.get(v.slot)?.as_ref()?.voices.get(v.idx)
    }

    pub fn voice_mut(&mut self, v: Vid) -> Option<&mut Voice> {
        self.racks.get_mut(v.slot)?.as_mut()?.voices.get_mut(v.idx)
    }

    pub fn rack(&self, slot: usize) -> Option<&Rack> {
        self.racks.get(slot)?.as_ref()
    }

    pub fn rack_mut(&mut self, slot: usize) -> Option<&mut Rack> {
        self.racks.get_mut(slot)?.as_mut()
    }

    pub fn rack_slot(&self, handle: u64) -> Option<usize> {
        self.rack_by_handle.get(&handle).copied()
    }

    pub fn voice_handle(&self, v: Vid) -> u64 {
        self.rack(v.slot).map_or(0, |r| r.handle + RACK_HEADER_SIZE + v.idx as u64 * VOICE_STRIDE)
    }

    pub fn find_voice(&self, handle: u64) -> Option<Vid> {
        let (&base, &slot) = self.rack_by_handle.range(..=handle).next_back()?;
        let off = handle.checked_sub(base + RACK_HEADER_SIZE)?;
        if off % VOICE_STRIDE != 0 {
            return None;
        }
        let idx = (off / VOICE_STRIDE) as usize;
        (idx < self.rack(slot)?.voices.len()).then_some(Vid { slot, idx })
    }

    fn rack_type(&self, v: Vid) -> Option<RackType> {
        self.rack(v.slot).map(|r| r.ty)
    }

    /// System rate of the voice's rack, as the 32.32 phase of one output frame.
    fn output_rate(&self, v: Vid) -> u64 {
        let rate = self.rack(v.slot).and_then(|r| self.systems.get(&r.system)).map_or(48000, |s| s.option.sample_rate);
        (rate as u64) << 32
    }

    // ---- systems ---------------------------------------------------------

    pub fn system_create(&mut self, handle: u64, option: SystemOption, buffer: ContextBuffer, allocator: Allocator) -> u32 {
        let uid = self.next_uid;
        self.next_uid = self.next_uid.wrapping_add(1);
        self.systems.insert(handle, System { option, buffer, allocator, uid, render_count: 0, user_data: 0, render_depth: 0 });
        uid
    }

    /// Remove a system and every rack it owns (returned for cleanup).
    pub fn system_destroy(&mut self, env: &mut dyn Env, handle: u64) -> Option<(System, Vec<Rack>)> {
        let system = self.systems.remove(&handle)?;
        let owned: Vec<u64> =
            self.rack_order.iter().filter_map(|&s| self.rack(s)).filter(|r| r.system == handle).map(|r| r.handle).collect();
        let racks = owned.into_iter().filter_map(|h| self.rack_destroy(env, h)).collect();
        Some((system, racks))
    }

    pub fn rack_count(&self, system: u64) -> u32 {
        self.rack_order.iter().filter_map(|&s| self.rack(s)).filter(|r| r.system == system).count() as u32
    }

    // ---- racks -----------------------------------------------------------

    /// Create a rack at `handle` (its working buffer). Voices get
    /// `max_ports` ports and `max_matrices` matrices; user effects of custom
    /// racks are set up through `env`.
    pub fn rack_create(
        &mut self,
        env: &mut dyn Env,
        system: u64,
        rack_id: u32,
        option: RackOption,
        buffer: ContextBuffer,
        handle: u64,
    ) -> Result<usize, i32> {
        if !self.systems.contains_key(&system) {
            return Err(ERR_INVALID_SYSTEM_HANDLE);
        }
        let Some((ty, expected)) = RackType::from_id(rack_id) else {
            log::warn!("[ngs2] unknown rack id {rack_id:#x}");
            return Err(ERR_INVALID_ARGUMENT);
        };
        if option.size() != expected as u64 {
            log::debug!("[ngs2] rack {rack_id:#x} option size {} (expected {expected})", option.size());
        }
        if self.rack_by_handle.contains_key(&handle) {
            log::warn!("[ngs2] rack buffer {handle:#x} reused without destroy");
            self.rack_destroy(env, handle);
        }
        let max_voices = option.max_voices() as usize;
        let mut voices = Vec::with_capacity(max_voices);
        for _ in 0..max_voices {
            let mut v = Voice::default();
            v.ports.resize(option.max_ports() as usize, Port::default());
            v.matrices.resize(option.max_matrices() as usize, Vec::new());
            if ty.is_custom() {
                v.modules.resize(option.custom_num_modules() as usize, Module { flags: 1, ..Module::default() });
            }
            voices.push(v);
        }
        let rack = Rack {
            handle,
            system,
            ty,
            option,
            buffer,
            allocator: Allocator::default(),
            voices,
            common: vec![Vec::new(); MAX_MODULES],
            fx: vec![UserFx::default(); MAX_MODULES],
            uid: self.next_rack_uid,
            user_data: 0,
        };
        self.next_rack_uid = self.next_rack_uid.wrapping_add(1);
        let slot = match self.racks.iter().position(Option::is_none) {
            Some(s) => {
                self.racks[s] = Some(rack);
                s
            }
            None => {
                self.racks.push(Some(rack));
                self.racks.len() - 1
            }
        };
        self.rack_by_handle.insert(handle, slot);
        self.rack_order.insert(0, slot);
        self.init_modules(env, slot);
        Ok(slot)
    }

    /// Set up the user effects (module id 0x1f) of a custom rack.
    fn init_modules(&mut self, env: &mut dyn Env, slot: usize) {
        let Some(rack) = self.rack_mut(slot) else { return };
        if !rack.ty.is_custom() {
            return;
        }
        let buffers = rack.option.custom_num_buffers();
        if buffers == 0 || buffers > MAX_CUSTOM_BUFFERS {
            log::warn!("[ngs2] custom rack with {buffers} buffers");
        }
        for port in 0..(rack.option.max_ports() as usize).min(16) {
            if rack.option.custom_port_buffer(port) >= buffers.max(1) {
                log::warn!("[ngs2] custom rack port {port} sends buffer {} of {buffers}", rack.option.custom_port_buffer(port));
            }
        }
        for m in 0..rack.option.custom_num_modules() as usize {
            // Modules process their source buffer in place.
            let info = rack.option.module(m);
            if info.source_buffer_id != info.dest_buffer_id || info.source_buffer_id.max(info.extra_buffer_id) >= buffers.max(1) {
                log::warn!("[ngs2] custom module {m} buffers {}/{}/{} of {buffers}", info.source_buffer_id, info.extra_buffer_id, info.dest_buffer_id);
            }
        }
        let max_voices = rack.voices.len() as u64;
        for m in 0..rack.option.custom_num_modules() as usize {
            let info = rack.option.module(m);
            if info.module_id != MODULE_USER_FX {
                continue;
            }
            let Some(raw) = read_bytes(env, info.option, 72) else {
                log::warn!("[ngs2] unreadable user fx option {:#x}", info.option);
                continue;
            };
            let fx = UserFx::parse(&raw);
            rack.fx[m] = fx;
            rack.common[m] = vec![0; fx.common_size as usize];
            for i in 0..rack.voices.len() {
                let module = &mut rack.voices[i].modules[m];
                module.param = vec![0; fx.param_size as usize];
                module.work = vec![0; fx.work_size as usize];
                module.state = vec![0; info.state_size as usize];
                if fx.setup != 0 {
                    let ctx = fx_context(&mut rack.common[m], &mut rack.voices[i].modules[m], fx.user_data, max_voices, i as u64);
                    let r = env.call(fx.setup, [ctx.as_ptr() as u64, 0, 0, 0, 0, 0]) as i32;
                    if r != OK {
                        log::warn!("[ngs2] user fx setup returned {r:#x}");
                    }
                }
            }
        }
    }

    pub fn rack_destroy(&mut self, env: &mut dyn Env, handle: u64) -> Option<Rack> {
        let slot = self.rack_by_handle.remove(&handle)?;
        self.rack_order.retain(|&s| s != slot);
        let mut rack = self.racks.get_mut(slot)?.take()?;
        let max_voices = rack.voices.len() as u64;
        for i in 0..rack.voices.len() {
            for m in 0..rack.voices[i].modules.len() {
                let fx = rack.fx[m];
                if fx.cleanup != 0 {
                    let ctx = fx_context(&mut rack.common[m], &mut rack.voices[i].modules[m], fx.user_data, max_voices, i as u64);
                    let r = env.call(fx.cleanup, [ctx.as_ptr() as u64, 0, 0, 0, 0, 0]) as i32;
                    if r != OK {
                        log::warn!("[ngs2] user fx cleanup returned {r:#x}");
                    }
                }
            }
        }
        Some(rack)
    }

    /// sceNgs2RackGetVoiceHandle: ids past the rack size map to the last voice.
    pub fn rack_voice_handle(&self, rack: u64, voice_id: u32) -> Result<u64, i32> {
        let slot = self.rack_slot(rack).ok_or(ERR_INVALID_RACK_HANDLE)?;
        let count = self.rack(slot).map_or(0, |r| r.voices.len());
        let idx = voice_id as usize;
        if idx >= count {
            log::warn!("[ngs2] voice id {voice_id} >= max voices {count}");
            return Err(ERR_INVALID_VOICE_INDEX);
        }
        Ok(self.voice_handle(Vid { slot, idx }))
    }

    /// SceNgs2RackInfo image of a rack.
    pub fn rack_info(&self, handle: u64) -> Option<[u8; RACK_INFO_SIZE]> {
        let rack = self.rack(self.rack_slot(handle)?)?;
        let option = &rack.option;
        let rack_id = rack.ty.id();
        let max_channel_works = match rack.ty {
            RackType::Sampler => option.u32_at(OPT_EXT),
            RackType::CustomSampler => option.u32_at(OPT_CUSTOM_EXT),
            _ => 0,
        };
        let max_inputs = match rack.ty {
            RackType::Submixer => option.u32_at(OPT_EXT + 12),
            RackType::CustomSubmixer => option.u32_at(OPT_CUSTOM_EXT + 4),
            _ => 0,
        };
        let first = rack.voices.first();
        let render_count = self.systems.get(&rack.system).map_or(0, |s| s.render_count);
        let active = rack.voices.iter().filter(|v| v.state != PlayState::Empty).count() as u32;
        let mut b = [0u8; RACK_INFO_SIZE];
        b[0..63].copy_from_slice(&option.bytes[8..71]);
        put64(&mut b, 64, handle);
        b[72..72 + CONTEXT_BUFFER_INFO_SIZE].copy_from_slice(&rack.buffer.to_bytes());
        put64(&mut b, 136, rack.system);
        put32(&mut b, 144, rack_id >> 12);
        put32(&mut b, 148, rack_id);
        put32(&mut b, 152, rack.uid);
        put32(&mut b, 156, MIN_GRAIN_SAMPLES);
        put32(&mut b, 160, option.u32_at(OPT_MAX_GRAIN));
        put32(&mut b, 164, rack.voices.len() as u32);
        put32(&mut b, 168, max_channel_works);
        put32(&mut b, 172, max_inputs);
        put32(&mut b, 176, first.map_or(0, |v| v.matrices.len() as u32));
        put32(&mut b, 180, first.map_or(0, |v| v.ports.len() as u32));
        put32(&mut b, 184, 1);
        put64(&mut b, 200, render_count);
        put32(&mut b, 208, active);
        Some(b)
    }

    /// SceNgs2VoicePortInfo image of a voice port; None past the last port.
    pub fn voice_port_info(&self, vid: Vid, port: u32) -> Option<[u8; VOICE_PORT_INFO_SIZE]> {
        let p = self.voice(vid)?.ports.get(port as usize)?;
        let mut b = [0u8; VOICE_PORT_INFO_SIZE];
        put32(&mut b, 0, p.matrix as u32);
        put32(&mut b, 4, p.volume.to_bits());
        put64(&mut b, 16, p.dest);
        Some(b)
    }

    // ---- voice control ---------------------------------------------------

    /// sceNgs2VoiceControl: walk the parameter list at `params`.
    pub fn voice_control(&mut self, env: &mut dyn Env, vid: Vid, params: u64) -> i32 {
        let mut addr = params;
        for _ in 0..4096 {
            let mut head = [0u8; 8];
            if addr == 0 || !env.read(addr, &mut head) {
                log::warn!("[ngs2] unreadable voice param at {addr:#x}");
                return OK;
            }
            let size = u16::from_le_bytes([head[0], head[1]]) as usize;
            let next = i16::from_le_bytes([head[2], head[3]]);
            let id = le32(&head, 4);
            let mut body = [0u8; 0x200];
            let len = size.clamp(8, body.len());
            if !env.read(addr, &mut body[..len]) {
                return OK;
            }
            log::trace!("[ngs2] voice param id={id:#x} size={size} next={next}");
            // A play event (param 6) needs a configured voice: sending it
            // to one without channels is refused for the whole walk.
            const SCE_NGS2_VOICE_PARAM_EVENT: u32 = 6;
            const SCE_NGS2_VOICE_EVENT_PLAY: u32 = 1;
            const SCE_NGS2_ERROR_INVALID_VOICE_ACCESS: i32 = 0x804A_8351u32 as i32;
            if id == SCE_NGS2_VOICE_PARAM_EVENT && len >= 12 && le32(&body, 8) == SCE_NGS2_VOICE_EVENT_PLAY
                && self.voice(vid).is_some_and(|v| v.channels == 0)
            {
                return SCE_NGS2_ERROR_INVALID_VOICE_ACCESS;
            }
            self.apply_param(env, vid, id, &body[..len], size);
            if next == 0 {
                break;
            }
            addr = addr.wrapping_add(next as i64 as u64);
        }
        OK
    }

    fn apply_param(&mut self, env: &mut dyn Env, vid: Vid, id: u32, p: &[u8], size: usize) {
        let Some(ty) = self.rack_type(vid) else { return };
        let field = |o: usize| if o + 4 <= p.len() { le32(p, o) } else { 0 };
        let field64 = |o: usize| if o + 8 <= p.len() { le64(p, o) } else { 0 };
        if (id >> 15) & 1 != 0 {
            log::warn!("[ngs2] unsupported voice param id {id:#x}");
            return;
        }
        let rack_id = id >> 16;
        match rack_id {
            0x0000 => match id & 0x7fff {
                1 => {
                    let levels = self.read_levels(env, field64(16), field(12));
                    if let Some(v) = self.voice_mut(vid) {
                        v.set_matrix(field(8), &levels);
                    }
                }
                2 => {
                    if let Some(v) = self.voice_mut(vid) {
                        v.set_volume(field(8), f32::from_bits(field(12)));
                    }
                }
                3 => {
                    if let Some(v) = self.voice_mut(vid) {
                        v.set_port_matrix(field(8), field(12) as i32);
                    }
                }
                4 => {} // port delay
                5 => {
                    // A rack handle patches into the rack's first voice.
                    let mut dest = field64(16);
                    if dest != 0 && self.find_voice(dest).is_none() {
                        if let Some(slot) = self.rack_slot(dest).filter(|&s| self.rack(s).is_some_and(|r| !r.voices.is_empty())) {
                            dest = self.voice_handle(Vid { slot, idx: 0 });
                        }
                    }
                    let Some(v) = self.voice_mut(vid) else { return };
                    match v.ports.get_mut(field(8) as usize) {
                        Some(port) => {
                            port.dest = dest;
                            port.input = field(12);
                        }
                        None => log::warn!("[ngs2] patch port {} out of range", field(8)),
                    }
                }
                6 => {
                    if let Some(v) = self.voice_mut(vid) {
                        v.set_event(field(8));
                    }
                }
                7 => {
                    if let Some(v) = self.voice_mut(vid) {
                        v.callback = field64(8);
                        v.callback_data = field64(16);
                        v.callback_flags = field(24);
                    }
                }
                cid => log::warn!("[ngs2] unknown voice param {cid:#x}"),
            },
            0x2000 => {
                if ty != RackType::Submixer {
                    log::warn!("[ngs2] submixer param {id:#x} on {ty:?}");
                    return;
                }
                if id & 0xffff == 0 {
                    let max = self.rack(vid.slot).map_or(0, |r| r.option.max_channels());
                    let (channels, flags) = (field(8), field(12));
                    if channels == 0 || channels > max || flags != 0 {
                        log::warn!("[ngs2] submixer setup channels={channels} flags={flags:#x} (max {max})");
                        return;
                    }
                    if let Some(v) = self.voice_mut(vid) {
                        v.reset_setup_state();
                        v.channels = channels;
                    }
                }
            }
            0x2001 => {
                if ty != RackType::Reverb {
                    log::warn!("[ngs2] reverb param {id:#x} on {ty:?}");
                    return;
                }
                let Some(v) = self.voice_mut(vid) else { return };
                match id & 0xffff {
                    0 => {
                        let (inputs, outputs, flags) = (field(8), field(12), field(16));
                        if flags != 0 || inputs != outputs || !matches!(outputs, 1 | 2 | 6 | 8) {
                            log::warn!("[ngs2] reverb setup {inputs} -> {outputs} channels, flags {flags:#x} rejected");
                            return;
                        }
                        v.reset_setup_state();
                        v.channels = outputs;
                        v.reverb = Some(Box::new(ReverbState::new()));
                    }
                    1 => {
                        if size < 8 + I3DL2_PARAM_SIZE || p.len() < 8 + I3DL2_PARAM_SIZE {
                            log::warn!("[ngs2] reverb I3DL2 param of {size} bytes ignored");
                            return;
                        }
                        let params = I3dl2Param::parse(&p[8..8 + I3DL2_PARAM_SIZE]);
                        let reverb = v.reverb.get_or_insert_with(|| Box::new(ReverbState::new()));
                        if reverb.set_params(&params).is_err() {
                            log::warn!("[ngs2] reverb I3DL2 parameters with NaN rejected: {params:?}");
                        }
                    }
                    _ => log::warn!("[ngs2] unknown reverb param {id:#x}"),
                }
            }
            0x3000 => {
                if ty != RackType::Mastering {
                    log::warn!("[ngs2] mastering param {id:#x} on {ty:?}");
                    return;
                }
                match id & 0xffff {
                    // A setup also resets the mastering gain.
                    0 => {
                        if let Some(v) = self.voice_mut(vid) {
                            v.channels = field(8);
                            v.fbw_level = 1.0;
                            v.lfe_level = 1.0;
                        }
                    }
                    // SceNgs2MasteringVoiceGainParam {fbwLevel, lfeLevel}:
                    // linear factors, each clamped to [-32, 32] with a NaN
                    // kept (measured on a PS5, firmware 3.20 and 4.03).
                    4 => {
                        let level = |o: usize| f32::from_bits(field(o)).clamp(-MAX_MASTERING_GAIN, MAX_MASTERING_GAIN);
                        let (fbw, lfe) = (level(8), level(12));
                        if let Some(v) = self.voice_mut(vid) {
                            v.fbw_level = fbw;
                            v.lfe_level = lfe;
                        }
                    }
                    1 => {
                        if field(8) != 5 {
                            log::warn!("[ngs2] mastering matrix id {}", field(8));
                            return;
                        }
                        let levels = self.read_levels(env, field64(16), field(12));
                        if let Some(v) = self.voice_mut(vid) {
                            v.output_matrix.clear();
                            v.output_matrix.extend_from_slice(&levels);
                        }
                    }
                    5 => {
                        if let Some(v) = self.voice_mut(vid) {
                            v.output_id = field(8);
                        }
                    }
                    _ => log::debug!("[ngs2] mastering param {id:#x} ignored"),
                }
            }
            0x4000 => self.apply_module_param(env, vid, id, p),
            0x1000 | 0x4001 => {
                let expected = if rack_id == 0x1000 { RackType::Sampler } else { RackType::CustomSampler };
                if ty != expected {
                    log::warn!("[ngs2] sampler param {id:#x} on {ty:?}");
                    return;
                }
                self.apply_sampler_param(env, vid, id, p, size);
            }
            0x4002 => {
                if ty != RackType::CustomSubmixer || id & 0xffff != 0 {
                    log::warn!("[ngs2] custom submixer param {id:#x} on {ty:?}");
                    return;
                }
                if field(8) != field(12) {
                    log::warn!("[ngs2] custom submixer in/out channels {} != {}", field(8), field(12));
                }
                if let Some(v) = self.voice_mut(vid) {
                    v.channels = field(8);
                }
            }
            0x4003 => {
                // SceNgs2CustomMasteringVoiceSetupParam {numInputChannels,
                // flags} or SceNgs2CustomMasteringVoiceOutputParam
                // {outputId, reserved}.
                if ty != RackType::CustomMastering || size != 16 {
                    log::warn!("[ngs2] custom mastering param {id:#x} (size {size}) on {ty:?}");
                    return;
                }
                match id & 0xffff {
                    0 => {
                        let max = self.rack(vid.slot).map_or(0, |r| r.option.max_voice_channels());
                        let (channels, flags) = (field(8), field(12));
                        if channels == 0 || channels > max || flags != 0 {
                            log::warn!("[ngs2] custom mastering setup channels={channels} flags={flags:#x} (max {max})");
                            return;
                        }
                        if let Some(v) = self.voice_mut(vid) {
                            v.channels = channels;
                        }
                    }
                    1 => {
                        if let Some(v) = self.voice_mut(vid) {
                            v.output_id = field(8);
                        }
                    }
                    _ => log::warn!("[ngs2] unsupported custom mastering param {id:#x}"),
                }
            }
            _ => log::warn!("[ngs2] unknown voice param rack id {rack_id:#x}"),
        }
    }

    fn read_levels(&self, env: &mut dyn Env, ptr: u64, count: u32) -> Vec<f32> {
        let count = count.min(64 * 64) as usize;
        let Some(raw) = read_bytes(env, ptr, count * 4) else {
            log::warn!("[ngs2] unreadable matrix levels at {ptr:#x}");
            return Vec::new();
        };
        raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn apply_module_param(&mut self, env: &mut dyn Env, vid: Vid, id: u32, p: &[u8]) {
        let field = |o: usize| if o + 4 <= p.len() { le32(p, o) } else { 0 };
        let field64 = |o: usize| if o + 8 <= p.len() { le64(p, o) } else { 0 };
        let (data, size) = (field64(8), field64(16));
        let rate = (self.output_rate(vid) >> 32) as u32;
        let Some(rack) = self.rack_mut(vid.slot) else { return };
        if !rack.ty.is_custom() {
            log::warn!("[ngs2] custom module param {id:#x} on {:?}", rack.ty);
            return;
        }
        let index = (id & 0x1f) as usize;
        let num_modules = rack.option.custom_num_modules() as usize;
        let module_id = if index < num_modules { rack.option.module(index).module_id } else { u32::MAX };
        let ty = rack.ty;
        match id & 0xffff_ffe0 {
            0x4000_1300 => {
                // SceNgs2CustomVoiceFilterParam: type@8, channelMask u32@12,
                // frequency@16, q@20, level@24.
                if !matches!(ty, RackType::CustomSampler | RackType::CustomSubmixer) || module_id != MODULE_FILTER {
                    log::warn!("[ngs2] filter param {id:#x} on {ty:?} module {module_id:#x} rejected");
                    return;
                }
                let filter_ty = field(8);
                if filter_ty > 1 {
                    log::error!("[ngs2] custom filter type {filter_ty} is not implemented");
                    return;
                }
                let param = FilterParam {
                    index: index as u32,
                    location: 1,
                    ty: filter_ty,
                    channel_mask: field(12) as u64,
                    frequency: f32::from_bits(field(16)),
                    q: f32::from_bits(field(20)),
                    level: f32::from_bits(field(24)),
                };
                let Some(v) = rack.voices.get_mut(vid.idx) else { return };
                if v.filters.len() <= index {
                    v.filters.resize(index + 1, Filter::default());
                }
                let channels = v.channels;
                v.filters[index].configure(&param, rate, channels);
                return;
            }
            0x4000_1d00 => {
                // SceNgs2CustomVoiceUserFxParam: handler@8, userData0..2@16.
                if ty != RackType::CustomSubmixer || module_id != MODULE_USER_FX_HANDLER || p.len() < 40 {
                    log::warn!("[ngs2] user fx handler param {id:#x} ({} bytes) on {ty:?} module {module_id:#x} rejected", p.len());
                    return;
                }
                let Some(module) = rack.voices.get_mut(vid.idx).and_then(|v| v.modules.get_mut(index)) else { return };
                module.handler = field64(8);
                module.handler_data = [field64(16), field64(24), field64(32)];
                module.flags = 1;
                return;
            }
            _ => {}
        }
        let control = rack.fx.get(index).map_or(0, |f| f.control);
        let Some(module) = rack.voices.get_mut(vid.idx).and_then(|v| v.modules.get_mut(index)) else {
            log::warn!("[ngs2] custom module {index} out of range");
            return;
        };
        if id & 0xffff_ffe0 != 0x4000_1f00 || module_id != MODULE_USER_FX {
            if !module.control_warned {
                log::warn!("[ngs2] custom control {id:#x} is ignored");
                module.control_warned = true;
            }
            return;
        }
        if control != 0 || size != module.param.len() as u64 {
            log::warn!("[ngs2] user fx param size {size} (expected {})", module.param.len());
            return;
        }
        if !module.param.is_empty() && !env.read(data, &mut module.param) {
            log::warn!("[ngs2] unreadable user fx param at {data:#x}");
            return;
        }
        module.flags |= 2;
    }

    fn apply_sampler_param(&mut self, env: &mut dyn Env, vid: Vid, id: u32, p: &[u8], size: usize) {
        let rate = (self.output_rate(vid) >> 32) as u32;
        let (ty, max_filters) = match self.rack(vid.slot) {
            Some(r) => (r.ty, r.option.sampler_max_filters()),
            None => return,
        };
        let field = |o: usize| if o + 4 <= p.len() { le32(p, o) } else { 0 };
        match id & 0xffff {
            0 => {
                let mut raw = [0u8; 24];
                let n = p.len().saturating_sub(8).min(24);
                raw[..n].copy_from_slice(&p[8..8 + n]);
                let format = WaveformFormat::parse(&raw);
                if let Some(v) = self.voice_mut(vid) {
                    v.setup_sampler(&format);
                }
            }
            1 => self.queue_blocks(env, vid, p),
            4 => {
                if !ty.is_sampler() {
                    log::warn!("[ngs2] exit-loop on {ty:?}");
                    return;
                }
                if let Some(v) = self.voice_mut(vid) {
                    if let Some(b) = v.blocks.iter_mut().find(|b| b.info.num_repeats != 0) {
                        b.info.num_repeats = 0;
                    }
                }
            }
            5 => {
                let ratio = f32::from_bits(field(8));
                if !ratio.is_finite() {
                    log::warn!("[ngs2] non-finite pitch ratio");
                    return;
                }
                if let Some(v) = self.voice_mut(vid) {
                    let ratio = ratio.clamp(0.0, 4.0);
                    v.sample_step = (v.sample_rate as f64 * ratio as f64 * 4294967296.0).round() as u64;
                }
            }
            0xa => {
                if ty != RackType::Sampler || size != 8 + FILTER_PARAM_SIZE || p.len() < 8 + FILTER_PARAM_SIZE {
                    log::warn!("[ngs2] filter param size {size} on {ty:?}");
                    return;
                }
                let param = FilterParam::parse(&p[8..8 + FILTER_PARAM_SIZE]);
                if param.index >= max_filters {
                    log::warn!("[ngs2] filter index {} >= {max_filters}", param.index);
                    return;
                }
                if let Some(v) = self.voice_mut(vid) {
                    if v.filters.len() <= param.index as usize {
                        v.filters.resize(param.index as usize + 1, Filter::default());
                    }
                    let channels = v.channels;
                    v.filters[param.index as usize].configure(&param, rate, channels);
                }
            }
            _ => log::debug!("[ngs2] sampler param {id:#x} ignored"),
        }
    }

    /// Waveform blocks param: {data@8, flags@16, count@20, blocks@24}.
    fn queue_blocks(&mut self, env: &mut dyn Env, vid: Vid, p: &[u8]) {
        if p.len() < 32 {
            return;
        }
        let (data, flags, count, blocks_ptr) = (le64(p, 8), le32(p, 16), le32(p, 20), le64(p, 24));
        let count = count.min(4096) as usize;
        let Some(raw) = read_bytes(env, blocks_ptr, count * WAVEFORM_BLOCK_SIZE) else {
            log::warn!("[ngs2] unreadable waveform blocks at {blocks_ptr:#x}");
            return;
        };
        let Some(v) = self.voice_mut(vid) else { return };
        if !v.accepts_blocks || flags & !0x17 != 0 {
            log::debug!("[ngs2] blocks flags={flags:#x} accepts={}", v.accepts_blocks);
        }
        if flags & 4 != 0 {
            v.blocks.clear();
            v.sample_phase = 0;
            v.duration_remaining = 0;
            v.skip_remaining = 0;
            v.compressed_input.clear();
            v.sf_frames_left = 0;
            v.sf_bytes_left = 0;
            v.frame_cursor = 0;
            v.frame_samples = 0;
            v.next_frame_samples = 0;
            v.waveform_end = 0;
        }
        v.accepts_blocks = flags & 1 != 0;
        let only_data = flags & 2 != 0;
        // SILENCE (0x10): a block without data plays num_samples of silence.
        let allows_silence = flags & WAVEFORM_BLOCKS_FLAG_SILENCE != 0;
        for chunk in raw.chunks_exact(WAVEFORM_BLOCK_SIZE) {
            let mut block = WaveformBlock::parse(chunk);
            if block.num_samples == 0 && block.data_size == 0 {
                continue;
            }
            if !only_data && block.num_samples == 0 {
                log::warn!("[ngs2] block without samples ignored");
                continue;
            }
            if only_data {
                block.num_repeats = 0;
                block.num_skip_samples = 0;
                block.num_samples = 0;
            }
            let address = data.wrapping_add(block.data_offset);
            let mut queued = Block::new(address, block);
            let frames = block.num_skip_samples as u64 + block.num_samples as u64;
            if allows_silence && block.data_size == 0 {
                if v.decoder.is_some() {
                    log::warn!("[ngs2] silent block on a coded voice ignored");
                    continue;
                }
                // Decoded zeros: rendered and counted like PCM data.
                queued.decoded = Some(vec![0i16; (frames * v.channels as u64) as usize].into_boxed_slice());
                v.blocks.push_back(queued);
                continue;
            }
            if v.waveform_type == WAVEFORM_TYPE_VAG {
                match decode_vag_block(env, address, block.data_size, v.channels) {
                    Some(pcm) if pcm.len() as u64 >= frames * v.channels as u64 => {
                        if only_data {
                            // A data-only block plays the samples its data holds.
                            queued.info.num_samples = (pcm.len() / v.channels.max(1) as usize) as u32;
                        }
                        queued.decoded = Some(pcm);
                    }
                    _ => {
                        log::warn!("[ngs2] VAG block at {address:#x} too short for {frames} samples");
                        continue;
                    }
                }
            } else if v.decoder.is_none() {
                // A data-only block plays the frames its data holds, and a
                // streamed block may hold fewer frames than its duration:
                // it ends when its data does. A repeating block must hold
                // its whole duration.
                let in_data = block.data_size as u64 / v.frame_bytes();
                if only_data {
                    queued.info.num_samples = in_data as u32;
                } else if frames > in_data {
                    if queued.info.num_repeats != 0 {
                        log::warn!("[ngs2] repeating block at {address:#x} holds {in_data} of {frames} frames");
                        continue;
                    }
                    queued.info.num_samples = in_data.saturating_sub(block.num_skip_samples as u64) as u32;
                }
                // Nothing left to play: the block is not queued.
                if queued.info.num_samples == 0 {
                    continue;
                }
            }
            v.blocks.push_back(queued);
        }
    }

    // ---- state queries ---------------------------------------------------

    /// Fill a voice state image of `out.len()` bytes for the rack type.
    /// The state size sceNgs2VoiceGetState takes for the voice's rack type,
    /// when the console's is known.
    pub fn voice_state_size(&self, vid: Vid) -> Option<usize> {
        match self.rack_type(vid)? {
            RackType::Sampler => Some(SAMPLER_VOICE_STATE_SIZE),
            RackType::CustomSampler => Some(CUSTOM_SAMPLER_VOICE_STATE_SIZE),
            RackType::Submixer => Some(20),
            RackType::Reverb => Some(8),
            _ => None,
        }
    }

    pub fn voice_state(&self, vid: Vid, out: &mut [u8]) {
        out.fill(0);
        let (Some(v), Some(ty)) = (self.voice(vid), self.rack_type(vid)) else { return };
        if out.len() >= 4 {
            put32(out, 0, v.state_flags);
        }
        let user_data = v.blocks.front().map_or(0, |b| b.info.user_data);
        match ty {
            RackType::Sampler if out.len() >= SAMPLER_VOICE_STATE_SIZE => {
                out[8..12].copy_from_slice(&1.0f32.to_le_bytes());
                put64(out, 24, v.decoded_samples);
                put64(out, 32, v.decoded_bytes);
                put64(out, 40, user_data);
                put64(out, 48, v.waveform_data());
            }
            RackType::CustomSampler if out.len() >= CUSTOM_SAMPLER_VOICE_STATE_SIZE => {
                put64(out, 8, v.waveform_data());
                put64(out, 16, v.decoded_samples);
                put64(out, 24, v.decoded_bytes);
                put64(out, 32, user_data);
            }
            _ => {}
        }
    }

    // ---- rendering -------------------------------------------------------

    /// Render `target` pulling from `sources` (plus itself), as one grain.
    /// Callers reset `rendered` flags.
    #[cfg(test)]
    pub fn render_voice_from(&mut self, env: &mut dyn Env, target: Vid, sources: &[Vid], grain: u32) {
        let mut edges = std::mem::take(&mut self.scratch.edges);
        self.build_edges(sources, &mut edges);
        self.render_voice(env, target, &edges, grain);
        self.scratch.edges = edges;
    }

    fn build_edges(&self, voices: &[Vid], edges: &mut Vec<Edge>) {
        edges.clear();
        for &src in voices {
            let Some(v) = self.voice(src) else { continue };
            for (port_idx, port) in v.ports.iter().enumerate() {
                if port.dest != 0 && port.volume != 0.0 {
                    edges.push(Edge { dest: port.dest, seq: edges.len() as u32, src, port: port_idx as u32 });
                }
            }
        }
        edges.sort_unstable_by_key(|e| (e.dest, e.seq));
    }

    fn render_voice(&mut self, env: &mut dyn Env, vid: Vid, edges: &[Edge], grain: u32) {
        let handle = self.voice_handle(vid);
        let Some(ty) = self.rack_type(vid) else { return };
        let rate = self.output_rate(vid);
        let buffers = self.buffers(vid.slot);
        let Some(v) = self.voice_mut(vid) else { return };
        if v.rendered {
            return;
        }
        if v.channels == 0 {
            v.has_samples = false;
            v.rendered = true;
            return;
        }
        if v.rendering {
            log::warn!("[ngs2] routing cycle through voice {handle:#x}");
            return;
        }
        v.rendering = true;
        v.has_samples = false;
        let playing = v.playing();
        // Custom racks keep numBuffers consecutive channel-major buffers.
        let len = grain as usize * v.channels as usize * buffers;
        if playing || v.samples.len() != len {
            v.samples.clear();
            v.samples.resize(len, 0.0);
        }
        if playing {
            if ty.is_sampler() {
                self.consume_samples(env, vid, grain, rate);
                if let Some(v) = self.voice_mut(vid) {
                    let channels = v.channels;
                    // A playing stream without source data may still ring out a filter tail.
                    for f in v.filters.iter_mut() {
                        if v.has_samples || f.has_history() {
                            f.process(&mut v.samples, channels, grain);
                            v.has_samples = true;
                        }
                    }
                    if v.playing()
                        && v.blocks.is_empty()
                        && v.frame_cursor == v.frame_samples
                        && v.next_frame_samples == 0
                        && !v.accepts_blocks
                        && !v.filters.iter().any(Filter::has_history)
                    {
                        v.state = PlayState::Empty;
                    }
                }
            }
            let lo = edges.partition_point(|e| e.dest < handle);
            for e in edges[lo..].iter().take_while(|e| e.dest == handle) {
                if e.src == vid {
                    continue;
                }
                let port = match self.voice(e.src).and_then(|s| s.ports.get(e.port as usize)) {
                    Some(p) if p.dest == handle && p.volume != 0.0 => *p,
                    _ => continue,
                };
                self.render_voice(env, e.src, edges, grain);
                let source_buffer = self.port_buffer(e.src.slot, e.port as usize);
                self.mix_port(e.src, vid, &port, source_buffer, grain);
            }
            if ty == RackType::Reverb {
                self.process_reverb(vid, grain);
            }
            self.process_modules(env, vid, grain);
        }
        if let Some(v) = self.voice_mut(vid) {
            v.rendering = false;
            v.rendered = true;
        }
    }

    /// Add `src`'s grain into `dst` through a port (volume and matrix). A
    /// port without a matrix, or whose matrix levels were never set, mixes
    /// through the default channel map.
    fn mix_port(&mut self, src: Vid, dst: Vid, port: &Port, source_buffer: u32, grain: u32) {
        // Input N of a custom rack voice feeds its buffer N (other racks
        // have one input buffer).
        let custom = self.rack(dst.slot).is_some_and(|r| r.ty.is_custom());
        if custom && port.input as usize >= self.buffers(dst.slot) {
            log::warn!("[ngs2] port input {} past the destination's buffers", port.input);
            return;
        }
        let input = if custom { port.input as usize } else { 0 };
        let Some(d) = self.voice_mut(dst) else { return };
        let mut buffer = std::mem::take(&mut d.samples);
        let dst_channels = d.channels as usize;
        let start = (input * grain as usize * dst_channels).min(buffer.len());
        let out = &mut buffer[start..];
        let mut mixed = false;
        if let Some(s) = self.voice(src).filter(|s| s.has_samples) {
            let g = grain as usize;
            let src_channels = s.channels as usize;
            let matrix = if port.matrix < 0 { None } else { s.matrices.get(port.matrix as usize) };
            if port.matrix >= 0 && matrix.is_none() {
                log::warn!("[ngs2] port matrix {} out of range", port.matrix);
            } else {
                let matrix = matrix.filter(|m| !m.is_empty());
                let default_map = matrix.is_none() && default_level(src_channels, 0, dst_channels, 0).is_some();
                if matrix.is_none() && !default_map && src_channels != 0 {
                    static WARNED: std::sync::Once = std::sync::Once::new();
                    WARNED.call_once(|| {
                        log::warn!("[ngs2] no default channel map from {src_channels} into {dst_channels} channels: channel n goes to n")
                    });
                }
                let out_channels = match matrix {
                    None => dst_channels,
                    Some(m) => dst_channels.min(m.len() / src_channels.max(1)),
                };
                for d in 0..out_channels {
                    let Some(dst_run) = out.get_mut(d * g..(d + 1) * g) else { break };
                    for sc in 0..src_channels {
                        let level = port.volume
                            * match matrix {
                                None if default_map => default_level(src_channels, sc, dst_channels, d).unwrap_or(0.0),
                                None => (sc == d) as u32 as f32,
                                Some(m) => m[d * src_channels + sc],
                            };
                        if level == 0.0 {
                            continue;
                        }
                        let base = source_buffer as usize * g * src_channels;
                        let Some(src_run) = s.samples.get(base + sc * g..base + (sc + 1) * g) else { break };
                        for (o, &x) in dst_run.iter_mut().zip(src_run) {
                            *o += x * level;
                        }
                    }
                }
                mixed = true;
            }
        }
        if let Some(d) = self.voice_mut(dst) {
            d.samples = buffer;
            d.has_samples |= mixed;
        }
    }

    /// The buffers of a voice of rack `slot`: numBuffers for a custom rack,
    /// one otherwise.
    fn buffers(&self, slot: usize) -> usize {
        match self.rack(slot) {
            Some(r) if r.ty.is_custom() => r.option.custom_num_buffers().clamp(1, MAX_CUSTOM_BUFFERS) as usize,
            _ => 1,
        }
    }

    /// The buffer output port `port` of a voice of rack `slot` sends.
    fn port_buffer(&self, slot: usize, port: usize) -> u32 {
        match self.rack(slot) {
            Some(r) if r.ty.is_custom() && port < 16 => r.option.custom_port_buffer(port).min(MAX_CUSTOM_BUFFERS - 1),
            _ => 0,
        }
    }

    /// Run a reverb voice's reverb over its mixed input; the tail rings out
    /// without input.
    fn process_reverb(&mut self, vid: Vid, grain: u32) {
        let rate = (self.output_rate(vid) >> 32) as u32;
        let Some(v) = self.voice_mut(vid) else { return };
        let Some(reverb) = v.reverb.as_mut() else { return };
        match reverb.process(&mut v.samples, v.channels as usize, grain as usize, rate) {
            Ok(audible) => v.has_samples = audible,
            Err(e) => {
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| log::warn!("[ngs2] reverb not rendered: {e:?}"));
                v.samples.fill(0.0);
                v.has_samples = false;
            }
        }
    }

    /// Run the user effects of a custom rack voice over its grain.
    fn process_modules(&mut self, env: &mut dyn Env, vid: Vid, grain: u32) {
        let rate = (self.output_rate(vid) >> 32) as u32;
        let Some(rack) = self.rack_mut(vid.slot) else { return };
        if !rack.ty.is_custom() {
            return;
        }
        let buffers = rack.option.custom_num_buffers().clamp(1, MAX_CUSTOM_BUFFERS);
        let Rack { option, fx, common, voices, .. } = rack;
        let Some(v) = voices.get_mut(vid.idx) else { return };
        let channels = v.channels;
        // A user effect gets the source buffer's channels followed by the
        // extra (side-chain) buffer's: (source, extra) channel pointers.
        let channel_pointers = |v: &mut Voice, info: &ModuleInfo| -> Option<Vec<u64>> {
            if info.source_buffer_id != info.dest_buffer_id || info.source_buffer_id.max(info.extra_buffer_id) >= buffers {
                return None;
            }
            let (g, n) = (grain as usize, channels as usize);
            if v.samples.len() < buffers as usize * g * n {
                return None;
            }
            let base = v.samples.as_mut_ptr();
            let at = |buffer: u32, c: usize| {
                // SAFETY: buffer * g * n + c * g lies inside `samples`
                // (buffers * g * n long).
                unsafe { base.add(buffer as usize * g * n + c * g) as u64 }
            };
            Some((0..n).map(|c| at(info.source_buffer_id, c)).chain((0..n).map(|c| at(info.extra_buffer_id, c))).collect())
        };
        for m in 0..v.modules.len() {
            let info = option.module(m);
            if info.module_id == MODULE_FILTER {
                // Applied with the sampler's filters.
                continue;
            }
            if info.module_id == MODULE_USER_FX_HANDLER {
                if v.modules[m].handler == 0 {
                    continue;
                }
                let Some(pointers) = channel_pointers(v, &info) else { continue };
                let module = &mut v.modules[m];
                let mut ctx = [0u8; USER_FX_PROCESS_CONTEXT_SIZE];
                put64(&mut ctx, 0, pointers.as_ptr() as u64);
                put64(&mut ctx, 8, module.handler_data[0]);
                put64(&mut ctx, 16, module.handler_data[1]);
                put64(&mut ctx, 24, module.handler_data[2]);
                put32(&mut ctx, 32, module.flags);
                put32(&mut ctx, 36, channels);
                put32(&mut ctx, 40, grain);
                put32(&mut ctx, 44, rate);
                let r = env.call(module.handler, [ctx.as_ptr() as u64, 0, 0, 0, 0, 0]) as i32;
                if r != OK {
                    log::error!("[ngs2] user fx handler returned {r:#x}");
                }
                module.flags = 0;
                v.has_samples = v.samples.iter().any(|&x| x != 0.0);
                continue;
            }
            if info.module_id != MODULE_USER_FX {
                if !v.modules[m].render_warned {
                    log::warn!("[ngs2] custom module {:#04x} at index {m} is ignored during render", info.module_id);
                    v.modules[m].render_warned = true;
                }
                continue;
            }
            if !v.has_samples || fx[m].process == 0 {
                continue;
            }
            let Some(pointers) = channel_pointers(v, &info) else { continue };
            let module = &mut v.modules[m];
            let mut ctx = [0u8; 104];
            put64(&mut ctx, 0, pointers.as_ptr() as u64);
            put64(&mut ctx, 8, vec_ptr(&mut common[m]));
            put64(&mut ctx, 16, vec_ptr(&mut module.param));
            put64(&mut ctx, 24, vec_ptr(&mut module.work));
            put64(&mut ctx, 32, vec_ptr(&mut module.state));
            put64(&mut ctx, 40, fx[m].user_data);
            put32(&mut ctx, 48, module.flags);
            put32(&mut ctx, 52, channels);
            put32(&mut ctx, 56, channels);
            put32(&mut ctx, 60, grain);
            put32(&mut ctx, 64, rate);
            let r = env.call(fx[m].process, [ctx.as_ptr() as u64, 0, 0, 0, 0, 0]) as i32;
            if r != OK {
                log::warn!("[ngs2] user fx process returned {r:#x}");
            }
            module.flags = 0;
        }
    }

    /// Finish the front block: rewind a repeating block, otherwise pop it,
    /// then report it to the voice callback.
    fn finish_block(&mut self, env: &mut dyn Env, vid: Vid) {
        let handle = self.voice_handle(vid);
        let Some(v) = self.voice_mut(vid) else { return };
        let end = v.waveform_data();
        v.waveform_end = end;
        let has_decoder = v.decoder.is_some();
        let config = v.decoder_config;
        let Some(block) = v.blocks.front_mut() else { return };
        let repeat = block.info.num_repeats != 0;
        if repeat {
            if block.info.num_repeats != u32::MAX {
                block.info.num_repeats -= 1;
            }
            block.repeated_count += 1;
            block.cursor = 0;
        }
        let info = CallbackInfo {
            data: v.callback_data,
            voice: handle,
            flag: if repeat { 2 } else { 1 },
            user: block.info.user_data,
            block_data: block.data,
            size: block.info.data_size as u32,
            repeats: block.repeated_count,
            attributes: 0,
        };
        if repeat {
            if has_decoder {
                block.data_cursor = 0;
                v.sf_frames_left = 0;
                v.sf_bytes_left = 0;
                v.compressed_input.clear();
                v.decoder = Atrac9Decoder::new(&config).ok();
            }
        } else {
            v.blocks.pop_front();
        }
        if v.callback != 0 && v.callback_flags & info.flag != 0 {
            let func = v.callback;
            env.voice_callback(self, func, &info);
        }
    }

    fn advance_pcm(&mut self, env: &mut dyn Env, vid: Vid, rate: u64) {
        // Completion callbacks may interrupt advancement across multiple blocks.
        loop {
            let Some(v) = self.voice_mut(vid) else { return };
            if !(v.playing() && v.sample_phase >= rate) {
                break;
            }
            let frame_bytes = v.frame_bytes();
            let phase = v.sample_phase;
            let Some(block) = v.blocks.front_mut() else { break };
            let available = (block.info.num_samples - block.cursor) as u64;
            let advance = (phase / rate).min(available);
            block.cursor += advance as u32;
            let done = block.cursor == block.info.num_samples;
            v.decoded_samples += advance;
            v.decoded_bytes += advance * frame_bytes;
            v.sample_phase -= advance * rate;
            if done {
                self.finish_block(env, vid);
            }
        }
        if let Some(v) = self.voice_mut(vid) {
            if v.blocks.is_empty() && !v.accepts_blocks {
                v.sample_phase = 0;
            }
        }
    }

    /// Decode the next ATRAC9 frame of the front block into the current
    /// (`next == false`) or look-ahead frame. Superframes are decoded whole
    /// and handed out one frame at a time; input progress is charged per
    /// frame, and the rest of the superframe is drained when the block's
    /// duration ends. Returns whether samples came out.
    fn decode_frame(&mut self, env: &mut dyn Env, vid: Vid, next: bool) -> bool {
        loop {
            let Some(v) = self.voice_mut(vid) else { return false };
            if !v.playing() {
                return false;
            }
            let need = v.superframe_bytes;
            let Some(block) = v.blocks.front_mut() else { return false };
            if block.info.num_samples != 0 && block.data_cursor == 0 && v.sf_frames_left == 0 {
                v.duration_remaining = block.info.num_samples;
                v.skip_remaining = block.info.num_skip_samples;
            }
            if v.duration_remaining == 0 {
                self.finish_block(env, vid);
                continue;
            }
            let data_size = block.info.data_size as usize;
            if v.sf_frames_left == 0 {
                let left = data_size.saturating_sub(block.data_cursor);
                let mut input = std::mem::take(&mut v.byte_scratch);
                let pending = !v.compressed_input.is_empty() || left < need;
                if pending {
                    // Superframe spread across blocks: gather it in compressed_input.
                    let bytes = (need - v.compressed_input.len().min(need)).min(left);
                    input.resize(bytes, 0);
                    if bytes != 0 && !env.read(block.data.wrapping_add(block.data_cursor as u64), &mut input) {
                        input.fill(0);
                    }
                    v.compressed_input.extend_from_slice(&input);
                    block.data_cursor += bytes;
                    v.decoded_bytes += bytes as u64;
                    if v.compressed_input.len() < need {
                        v.byte_scratch = input;
                        self.finish_block(env, vid);
                        continue;
                    }
                    input.clear();
                    input.extend_from_slice(&v.compressed_input);
                    v.compressed_input.clear();
                } else {
                    input.resize(need, 0);
                    if !env.read(block.data.wrapping_add(block.data_cursor as u64), &mut input) {
                        input.fill(0);
                    }
                }
                let decoded = match v.decoder.as_mut() {
                    Some(d) => d.decode(&input, &mut v.planar),
                    None => Err(crate::audio::atrac9::Atrac9Error::BufferTooSmall),
                };
                v.byte_scratch = input;
                if let Err(e) = decoded {
                    log::warn!("[ngs2] ATRAC9 decode failed: {e}");
                    v.planar.iter_mut().for_each(|c| c.fill(0));
                }
                v.sf_frames_left = v.sf_frames;
                v.sf_frame = 0;
                v.sf_bytes_left = if pending { 0 } else { need };
            }
            // Serve one frame of the decoded superframe.
            let fs = v.at9_frame_samples as usize;
            let frame_index = v.sf_frame as usize;
            v.sf_frame += 1;
            v.sf_frames_left -= 1;
            let charge = if v.sf_frames_left == 0 {
                v.sf_bytes_left
            } else {
                (need / v.sf_frames.max(1) as usize).min(v.sf_bytes_left)
            };
            v.sf_bytes_left -= charge;
            let Some(block) = v.blocks.front_mut() else { return false };
            block.data_cursor += charge;
            v.decoded_bytes += charge as u64;
            let channels = v.channels as usize;
            let skip = v.skip_remaining.min(fs as u32) as usize;
            v.skip_remaining -= skip as u32;
            let count = (fs - skip).min(v.duration_remaining as usize);
            let Voice { planar, decoded_frame, next_decoded_frame, .. } = &mut *v;
            let frame = if next { next_decoded_frame } else { decoded_frame };
            if frame.len() < fs * channels {
                frame.resize(fs * channels, 0.0);
            }
            let first = frame_index * fs + skip;
            for (c, plane) in planar.iter().enumerate().take(channels) {
                for (i, &s) in plane[first..first + count].iter().enumerate() {
                    frame[i * channels + c] = s as f32 / 32768.0;
                }
            }
            if count != 0 {
                if next {
                    v.next_frame_samples = count as u32;
                } else {
                    v.frame_samples = count as u32;
                }
                v.duration_remaining -= count as u32;
            }
            if v.duration_remaining == 0 {
                // Drain the rest of the superframe.
                let rest = std::mem::take(&mut v.sf_bytes_left);
                if let Some(block) = v.blocks.front_mut() {
                    block.data_cursor += rest;
                }
                v.decoded_bytes += rest as u64;
                v.sf_frames_left = 0;
                v.compressed_input.clear();
            }
            let drained = v.blocks.front().is_none_or(|b| b.data_cursor >= data_size);
            if v.duration_remaining == 0 || (drained && v.sf_frames_left == 0 && v.compressed_input.is_empty()) {
                self.finish_block(env, vid);
            }
            if count != 0 {
                return true;
            }
        }
    }

    fn current_atrac_frame(&mut self, env: &mut dyn Env, vid: Vid) -> bool {
        let Some(v) = self.voice_mut(vid) else { return false };
        if v.frame_cursor == v.frame_samples {
            if v.next_frame_samples != 0 {
                std::mem::swap(&mut v.decoded_frame, &mut v.next_decoded_frame);
                v.frame_samples = v.next_frame_samples;
                v.next_frame_samples = 0;
            } else if !self.decode_frame(env, vid, false) {
                return false;
            }
            let Some(v) = self.voice_mut(vid) else { return false };
            v.frame_cursor = 0;
        }
        self.voice(vid).is_some_and(|v| v.playing() && v.frame_cursor < v.frame_samples)
    }

    fn advance_atrac9(&mut self, env: &mut dyn Env, vid: Vid, rate: u64) {
        while self.voice(vid).is_some_and(|v| v.playing() && v.sample_phase >= rate) {
            if !self.current_atrac_frame(env, vid) {
                break;
            }
            let Some(v) = self.voice_mut(vid) else { return };
            v.frame_cursor += 1;
            v.decoded_samples += 1;
            v.sample_phase -= rate;
        }
        if let Some(v) = self.voice_mut(vid) {
            if v.blocks.is_empty() && v.frame_cursor == v.frame_samples && v.next_frame_samples == 0 && !v.accepts_blocks {
                v.sample_phase = 0;
            }
        }
    }

    fn consume_samples(&mut self, env: &mut dyn Env, vid: Vid, grain: u32, rate: u64) {
        let mut output = 0u32;
        let coded = self.voice(vid).is_some_and(|v| v.decoder.is_some());
        loop {
            let Some(v) = self.voice(vid) else { return };
            if !(v.playing()
                && output < grain
                && (!v.blocks.is_empty() || v.frame_cursor < v.frame_samples || v.next_frame_samples != 0))
            {
                break;
            }
            if !coded {
                self.advance_pcm(env, vid, rate);
                let Some(v) = self.voice_mut(vid) else { return };
                if !v.playing() || v.blocks.is_empty() {
                    break;
                }
                let produced = pcm_run(v, env, grain, output, rate);
                if produced != 0 {
                    output += produced;
                    continue;
                }
                pcm_edge_sample(v, env, grain, output, rate);
                output += 1;
                continue;
            }
            self.advance_atrac9(env, vid, rate);
            if !self.current_atrac_frame(env, vid) {
                break;
            }
            let needs_next = self
                .voice(vid)
                .is_some_and(|v| v.frame_cursor + 1 == v.frame_samples && v.sample_phase != 0);
            if needs_next {
                let Some(v) = self.voice_mut(vid) else { return };
                if v.next_frame_samples == 0 && !v.blocks.is_empty() {
                    if v.next_decoded_frame.is_empty() {
                        let n = v.decoded_frame.len();
                        v.next_decoded_frame.resize(n, 0.0);
                    }
                    self.decode_frame(env, vid, true);
                }
                let Some(v) = self.voice(vid) else { return };
                if !v.playing() || v.frame_cursor >= v.frame_samples {
                    break;
                }
            }
            let Some(v) = self.voice_mut(vid) else { return };
            let channels = v.channels as usize;
            let cur = v.frame_cursor as usize * channels;
            let next_base = if v.frame_cursor + 1 < v.frame_samples {
                Some((false, cur + channels))
            } else if v.next_frame_samples != 0 {
                Some((true, 0))
            } else {
                None
            };
            let t = v.sample_phase as f32 / rate as f32;
            let g = grain as usize;
            for c in 0..channels {
                let a = v.decoded_frame[cur + c];
                let b = match next_base {
                    Some((false, n)) => v.decoded_frame[n + c],
                    Some((true, n)) => v.next_decoded_frame[n + c],
                    None => a,
                };
                v.samples[c * g + output as usize] = lerp(a, b, t);
            }
            v.sample_phase += v.sample_step;
            output += 1;
        }
        if coded {
            self.advance_atrac9(env, vid, rate);
        } else {
            self.advance_pcm(env, vid, rate);
        }
        if let Some(v) = self.voice_mut(vid) {
            v.has_samples = output != 0;
        }
    }

    /// sceNgs2SystemRender: render every voice of `system` and write the
    /// mastering voices into `outputs`.
    /// Write one grain of silence into each usable buffer (a render skipped
    /// while another thread holds the system lock); nothing is rendered.
    pub fn system_silence(&self, env: &mut dyn Env, system: u64, outputs: &[RenderBuffer]) -> i32 {
        let Some(sys) = self.systems.get(&system) else {
            return ERR_INVALID_SYSTEM_HANDLE;
        };
        let grain = sys.option.num_grain_samples as usize;
        for out in outputs {
            if let Some(bytes) = render_grain_bytes(out, grain) {
                zero_guest(env, out.buffer, bytes);
            }
        }
        OK
    }

    pub fn system_render(&mut self, env: &mut dyn Env, system: u64, outputs: &[RenderBuffer]) -> i32 {
        let Some(sys) = self.systems.get(&system) else {
            return ERR_INVALID_SYSTEM_HANDLE;
        };
        let grain = sys.option.num_grain_samples;
        let mut voices = std::mem::take(&mut self.scratch.voices);
        voices.clear();
        for &slot in &self.rack_order {
            let Some(rack) = self.racks[slot].as_ref() else { continue };
            if rack.system != system {
                continue;
            }
            voices.extend((0..rack.voices.len()).map(|idx| Vid { slot, idx }));
        }
        for &v in &voices {
            if let Some(v) = self.voice_mut(v) {
                v.rendered = false;
            }
        }
        let mut edges = std::mem::take(&mut self.scratch.edges);
        self.build_edges(&voices, &mut edges);

        let mut mix = std::mem::take(&mut self.scratch.out_mix);
        let mut used = std::mem::take(&mut self.scratch.out_used);
        if mix.len() < outputs.len() {
            mix.resize_with(outputs.len(), Vec::new);
        }
        used.clear();
        used.resize(outputs.len(), false);

        let g = grain as usize;
        for &vid in &voices {
            self.render_voice(env, vid, &edges, grain);
            if !matches!(self.rack_type(vid), Some(RackType::Mastering | RackType::CustomMastering)) {
                continue;
            }
            let Some(v) = self.voice(vid).filter(|v| v.has_samples) else { continue };
            let Some(out) = outputs.get(v.output_id as usize) else {
                log::warn!("[ngs2] mastering output {} >= {} buffers", v.output_id, outputs.len());
                continue;
            };
            if out.buffer == 0 || !matches!(out.waveform_type, 0x12 | 0x18) || out.num_channels == 0 {
                log::debug!("[ngs2] unsupported render buffer {out:?}");
                continue;
            }
            let oc = out.num_channels as usize;
            let id = v.output_id as usize;
            let m = &mut mix[id];
            if !used[id] {
                m.clear();
                m.resize(g * oc, 0.0);
                used[id] = true;
            }
            for dst in 0..oc {
                for src in 0..v.channels as usize {
                    let level = if v.output_matrix.is_empty() {
                        (src == dst) as u32 as f32
                    } else {
                        v.output_matrix.get(src * oc + dst).copied().unwrap_or(0.0)
                    };
                    if level == 0.0 {
                        continue;
                    }
                    // The LFE factor acts on channel 3 of a 5.1 or 7.1
                    // voice, the full-band one on every other channel.
                    let gain = if src == LFE_SOURCE_CHANNEL && v.channels >= 6 { v.lfe_level } else { v.fbw_level };
                    let level = level * gain;
                    let Some(run) = v.samples.get(src * g..(src + 1) * g) else { continue };
                    for (frame, &x) in m.chunks_exact_mut(oc).zip(run) {
                        frame[dst] += x * level;
                    }
                }
            }
        }

        // Each buffer receives exactly one grain: the mix, or silence. The
        // rest of a larger buffer (a ring the title fills grain by grain) is
        // left as it is.
        let mut bytes = std::mem::take(&mut self.scratch.bytes);
        for (i, out) in outputs.iter().enumerate() {
            let Some(grain_bytes) = render_grain_bytes(out, g) else { continue };
            bytes.clear();
            if used[i] {
                let oc = out.num_channels as usize;
                let samples = &mix[i][..g * oc];
                let bps = if out.waveform_type == 0x12 { 2 } else { 4 };
                if bps == 2 {
                    bytes.extend(samples.iter().flat_map(|&x| {
                        let s = (x.clamp(-1.0, 1.0) * 32768.0).round_ties_even() as i32;
                        (s.clamp(-32768, 32767) as i16).to_le_bytes()
                    }));
                } else {
                    bytes.extend(samples.iter().flat_map(|x| x.to_le_bytes()));
                }
            }
            let written = bytes.len() as u64;
            if written != 0 && !env.write(out.buffer, &bytes) {
                log::warn!("[ngs2] render buffer {:#x} not writable", out.buffer);
                continue;
            }
            zero_guest(env, out.buffer.wrapping_add(written), grain_bytes.saturating_sub(written));
        }

        for &vid in &voices {
            let streaming_sampler = self.rack_type(vid) == Some(RackType::Sampler);
            if let Some(v) = self.voice_mut(vid) {
                // No envelope is configured, so a release stop finishes in this render.
                if v.state == PlayState::Stopped {
                    v.state = PlayState::Empty;
                }
                v.state_flags = v.state as u32;
                // A playing stream that consumed every queued block and
                // waits for more reports EMPTY.
                if streaming_sampler && v.state == PlayState::Playing && v.blocks.is_empty() && v.accepts_blocks {
                    v.state_flags |= VOICE_STATE_FLAG_EMPTY;
                }
            }
        }
        if let Some(sys) = self.systems.get_mut(&system) {
            sys.render_count += 1;
        }
        self.scratch.voices = voices;
        self.scratch.edges = edges;
        self.scratch.out_mix = mix;
        self.scratch.out_used = used;
        self.scratch.bytes = bytes;
        OK
    }
}

/// SCE_NGS2_VOICE_STATE_FLAG_EMPTY: a stream played every queued block.
pub(crate) const VOICE_STATE_FLAG_EMPTY: u32 = 0x20;
/// Most channels of a render buffer.
const RENDER_CHANNELS_MAX: u32 = 8;

/// Bytes one grain occupies in `out`, or None when the buffer is absent or
/// cannot take a grain (it is then left untouched).
fn render_grain_bytes(out: &RenderBuffer, grain: usize) -> Option<u64> {
    if out.buffer == 0 {
        return None;
    }
    let bps = match out.waveform_type {
        0x12 => 2,
        0x18 => 4,
        other => {
            log::warn!("[ngs2] render buffer {:#x}: unsupported waveform type {other:#x}", out.buffer);
            return None;
        }
    };
    if out.num_channels == 0 || out.num_channels > RENDER_CHANNELS_MAX {
        log::warn!("[ngs2] render buffer {:#x}: {} channels", out.buffer, out.num_channels);
        return None;
    }
    let bytes = grain as u64 * u64::from(out.num_channels) * bps;
    if out.size < bytes {
        log::warn!("[ngs2] render buffer {:#x}: {} bytes < one grain ({bytes})", out.buffer, out.size);
        return None;
    }
    Some(bytes)
}

fn vec_ptr(v: &mut [u8]) -> u64 {
    if v.is_empty() { 0 } else { v.as_mut_ptr() as u64 }
}

/// SceNgs2UserFxContext (72 bytes) for module setup/cleanup.
fn fx_context(common: &mut [u8], module: &mut Module, user_data: u64, max_voices: u64, index: u64) -> [u8; 72] {
    let mut ctx = [0u8; 72];
    put64(&mut ctx, 0, vec_ptr(common));
    put64(&mut ctx, 8, vec_ptr(&mut module.param));
    put64(&mut ctx, 16, vec_ptr(&mut module.work));
    put64(&mut ctx, 24, user_data);
    put32(&mut ctx, 32, max_voices as u32);
    put32(&mut ctx, 36, index as u32);
    ctx
}

pub(crate) fn zero_guest(env: &mut dyn Env, addr: u64, len: u64) -> bool {
    const ZEROES: [u8; 4096] = [0; 4096];
    let mut off = 0;
    while off < len {
        let n = (len - off).min(ZEROES.len() as u64) as usize;
        if !env.write(addr.wrapping_add(off), &ZEROES[..n]) {
            return false;
        }
        off += n as u64;
    }
    true
}

/// Decode a (possibly multichannel, unit-interleaved) VAG block into
/// interleaved PCM16.
fn decode_vag_block(env: &mut dyn Env, addr: u64, size: u64, channels: u32) -> Option<Box<[i16]>> {
    let raw = read_bytes(env, addr, size.min(64 << 20) as usize)?;
    let channels = channels.max(1) as usize;
    let unit = vag::UNIT_BYTES as usize;
    let units = raw.len() / unit / channels;
    let mut planes = Vec::with_capacity(channels);
    for c in 0..channels {
        let mut frames = Vec::with_capacity(units * unit);
        for u in 0..units {
            let o = (u * channels + c) * unit;
            frames.extend_from_slice(&raw[o..o + unit]);
        }
        planes.push(vag::decode_units(&frames));
    }
    let len = planes.iter().map(Vec::len).min().unwrap_or(0);
    let mut out = vec![0i16; len * channels];
    for (c, plane) in planes.iter().enumerate() {
        for (i, &s) in plane[..len].iter().enumerate() {
            out[i * channels + c] = s;
        }
    }
    Some(out.into_boxed_slice())
}

/// Convert `frames` source frames starting at absolute frame `first` of
/// `block` to interleaved floats in `voice.src_scratch`.
fn load_frames(v: &mut Voice, env: &mut dyn Env, block_index: usize, first: u64, frames: usize) {
    let channels = v.channels as usize;
    let n = frames * channels;
    let mut out = std::mem::take(&mut v.src_scratch);
    out.clear();
    out.resize(n, 0.0);
    let block = &v.blocks[block_index];
    if let Some(pcm) = block.decoded.as_deref() {
        let start = first as usize * channels;
        if let Some(src) = pcm.get(start..start + n) {
            for (o, &s) in out.iter_mut().zip(src) {
                *o = s as f32 / 32768.0;
            }
        }
    } else if let Some(frame_bytes) = pcm_frame_bytes(v.waveform_type, v.channels) {
        let mut raw = std::mem::take(&mut v.byte_scratch);
        raw.clear();
        raw.resize(frames * frame_bytes as usize, 0);
        if env.read(block.data.wrapping_add(first * frame_bytes), &mut raw) {
            convert_pcm(v.waveform_type, &raw, &mut out);
        }
        v.byte_scratch = raw;
    }
    v.src_scratch = out;
}

/// Normalise raw little-endian PCM samples to floats.
fn convert_pcm(ty: u32, raw: &[u8], out: &mut [f32]) {
    match ty {
        WAVEFORM_TYPE_PCM_I16L => {
            for (o, c) in out.iter_mut().zip(raw.chunks_exact(2)) {
                *o = i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0;
            }
        }
        WAVEFORM_TYPE_PCM_F32L => {
            for (o, c) in out.iter_mut().zip(raw.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        WAVEFORM_TYPE_PCM_I8 => {
            for (o, &b) in out.iter_mut().zip(raw) {
                *o = b as i8 as f32 / 128.0;
            }
        }
        WAVEFORM_TYPE_PCM_U8 => {
            for (o, &b) in out.iter_mut().zip(raw) {
                *o = (b as f32 - 128.0) / 128.0;
            }
        }
        WAVEFORM_TYPE_PCM_I24L => {
            for (o, c) in out.iter_mut().zip(raw.chunks_exact(3)) {
                *o = (i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8) as f32 / 8388608.0;
            }
        }
        WAVEFORM_TYPE_PCM_I32L => {
            for (o, c) in out.iter_mut().zip(raw.chunks_exact(4)) {
                *o = i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / 2147483648.0;
            }
        }
        WAVEFORM_TYPE_PCM_F64L => {
            for (o, c) in out.iter_mut().zip(raw.chunks_exact(8)) {
                *o = f64::from_le_bytes(c.try_into().unwrap()) as f32;
            }
        }
        _ => {}
    }
}

/// Produce as many output samples as possible whose interpolation pair lies
/// inside the front block (one guest read for the whole run). Returns 0
/// when the next sample needs data past the block end.
fn pcm_run(v: &mut Voice, env: &mut dyn Env, grain: u32, output: u32, rate: u64) -> u32 {
    let Some(block) = v.blocks.front() else { return 0 };
    let num = block.info.num_samples as u64;
    let c = block.cursor as u64;
    if c + 1 >= num {
        return 0;
    }
    let skip = block.info.num_skip_samples as u64;
    let (p, s) = (v.sample_phase, v.sample_step);
    let remaining = (grain - output) as u64;
    // Outputs j with offset floor((p + j*s) / rate) <= num - 2 - c.
    let limit = (num - 1 - c) as u128 * rate as u128;
    let n = if s == 0 {
        remaining
    } else {
        ((limit - p as u128).div_ceil(s as u128)).min(remaining as u128) as u64
    };
    if n == 0 {
        return 0;
    }
    let last_pos = p as u128 + (n - 1) as u128 * s as u128;
    let last_off = (last_pos / rate as u128) as u64;
    load_frames(v, env, 0, skip + c, last_off as usize + 2);

    let channels = v.channels as usize;
    let g = grain as usize;
    let base = output as usize;
    let src = &v.src_scratch;
    let dst = &mut v.samples;
    if p == 0 && s == rate {
        // 1:1 rate: the interpolation weight is always zero.
        for ch in 0..channels {
            let run = &mut dst[ch * g + base..ch * g + base + n as usize];
            for (j, o) in run.iter_mut().enumerate() {
                *o = src[j * channels + ch];
            }
        }
    } else {
        let (s_int, s_frac) = ((s / rate) as usize, s % rate);
        let rf = rate as f32;
        let (mut off, mut ph) = (0usize, p);
        for j in 0..n as usize {
            let t = ph as f32 / rf;
            let a = &src[off * channels..off * channels + channels];
            let b = &src[(off + 1) * channels..(off + 1) * channels + channels];
            for ch in 0..channels {
                dst[ch * g + base + j] = lerp(a[ch], b[ch], t);
            }
            ph += s_frac;
            off += s_int;
            if ph >= rate {
                ph -= rate;
                off += 1;
            }
        }
    }
    let frame_bytes = v.frame_bytes();
    if let Some(block) = v.blocks.front_mut() {
        block.cursor += last_off as u32;
    }
    v.decoded_samples += last_off;
    v.decoded_bytes += last_off * frame_bytes;
    v.sample_phase = (last_pos - last_off as u128 * rate as u128) as u64 + s;
    n as u32
}

/// One output sample at the last frame of the front block: interpolate
/// towards the loop start, the next block, or hold the sample.
fn pcm_edge_sample(v: &mut Voice, env: &mut dyn Env, grain: u32, output: u32, rate: u64) {
    let channels = v.channels as usize;
    let block = &v.blocks[0];
    let cursor = block.info.num_skip_samples as u64 + block.cursor as u64;
    let next = if block.cursor + 1 < block.info.num_samples {
        Some((0, cursor + 1))
    } else if block.info.num_repeats != 0 {
        Some((0, block.info.num_skip_samples as u64))
    } else {
        v.blocks.get(1).map(|b| (1, b.info.num_skip_samples as u64))
    };
    load_frames(v, env, 0, cursor, 1);
    let mut pair = [0f32; 128];
    let ch = channels.min(64);
    pair[..ch].copy_from_slice(&v.src_scratch[..ch]);
    match next {
        Some((block_index, frame)) => {
            load_frames(v, env, block_index, frame, 1);
            pair[64..64 + ch].copy_from_slice(&v.src_scratch[..ch]);
        }
        None => pair.copy_within(0..ch, 64),
    }
    let t = v.sample_phase as f32 / rate as f32;
    let g = grain as usize;
    for c in 0..ch {
        v.samples[c * g + output as usize] = lerp(pair[c], pair[64 + c], t);
    }
    v.sample_phase += v.sample_step;
}
