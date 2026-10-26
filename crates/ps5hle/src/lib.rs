// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! High-Level Emulation of PS5 system libraries.
//!
//! Each export is identified by a NID string. We provide native Rust
//! callbacks for the subset needed to reach a game's main menu. The HLE
//! dispatcher receives guest register state through a context snapshot
//! and returns the value to write to RAX.

pub mod context;
pub mod host;
pub mod libc;
pub mod kernel;
pub mod memory;
pub mod time;
pub mod modules;
pub mod tls;
pub mod pthread;
pub mod signals;
pub mod videoout;
pub mod agc;
pub mod registry;
pub mod cxx;
pub mod math;
pub mod system_service;
pub mod system_gesture;
pub mod pad;
pub mod pad_audio;
pub mod pad_controller;
pub mod pause;
pub mod rtc;
pub mod printf;
pub mod app_info;
pub mod user_service;
pub mod app_content;
pub mod sysmodule;
pub mod common_dialog;
pub mod ime;
pub mod ime_common;
pub mod ime_dialog;
pub mod playgo;
pub mod playgo_dialog;
pub mod jpeg_dec;
pub mod jpeg_enc;
pub mod savedata;
pub mod savedata_dialog;
pub mod np_manager;
pub mod np_trophy2;
pub mod np_game_intent;
pub mod np_party;
pub mod np_utility;
pub mod np_entitlement;
pub mod np_auth;
pub mod np_uds;
pub mod np_webapi2;
pub mod np_commerce;
pub mod np_signaling;
pub mod filesystem;
pub mod sync;
pub mod stdio;
pub mod ampr;
pub mod net;
pub mod netctl;
pub mod posix_webkit;
pub mod http;
pub mod ssl;
pub mod json;
pub mod json_doc;
pub mod share;
pub mod text_to_speech2;
pub mod voice_qos;
pub mod misc_services;
pub mod firmware_answers;
pub mod move_controller;
pub mod vr_tracker2;
pub mod usbd;
pub mod ces;
pub mod hmd;
pub mod hmd2;
pub mod dbg_asan;
pub mod audio;
pub mod fiber;
pub mod ult;
pub mod kernel_ext;
pub mod libc_atomic;
pub mod libc_ext;
pub mod libc_string;
pub mod libc_thrd;
pub mod libc_random;
pub mod memops;
mod lock_bench;
pub mod disc_map;
pub mod videodec;
pub mod fmod;
pub mod scanf;
pub mod game_update;
pub mod game_live_streaming;
pub mod remoteplay;
pub mod razor_cpu;
pub mod job_manager;
pub mod player_invitation_dialog;
pub mod offline_dialogs;
pub mod g2p_vr_dialog;
pub mod signin_dialog;
pub mod login_dialog;
pub mod web_browser_dialog;
pub mod content_export;
pub mod content_search;
pub mod convert_keycode;
pub mod avplayer;
pub mod videodec2;
pub mod font;
pub mod png_dec;
pub mod png_enc;
pub mod rudp;
pub mod voice_chat;
pub mod psml;
pub mod zlib;

#[cfg(test)]
mod test_bases;

pub use context::{CpuContext, CpuRegister};
pub use registry::{HleRegistry, HleResult};
