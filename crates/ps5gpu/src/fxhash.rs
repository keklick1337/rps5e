// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Hash maps keyed by addresses, ids and small plain keys on the per-draw
//! paths: a fast non-cryptographic hasher instead of SipHash (keys come
//! from the emulator, not from an adversary).

pub type HashMap<K, V> = std::collections::HashMap<K, V, rustc_hash::FxBuildHasher>;
pub type HashSet<K> = std::collections::HashSet<K, rustc_hash::FxBuildHasher>;
