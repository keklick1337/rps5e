// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Audio libraries: libSceAudioOut / libSceAudioOut2 / libSceAudioIn,
//! libSceAudioPropagation, libSceAjm (MP3, ATRAC9, AAC and Opus decoding), libSceAudiodec, libSceOpusDec, libSceOpusCeltDec, libSceNgs2 and
//! libSceAcm, plus the host output backend they share.

pub mod acm;
pub mod ajm;
pub mod atrac9;
pub mod audio3d;
pub mod audiodec;
pub mod audio_in;
pub mod audio_out;
pub mod audio_out2;
pub mod host;
pub mod ngs2;
pub mod opus_dec;
pub mod pcm;
pub mod propagation;
pub mod sdl_host;
pub mod voice;

use crate::registry::{HleCallback, HleRegistry};

/// Whether the audio output system was initialised (sceAudioOutInit or
/// sceAudioOut2Initialize): the AudioOut2 queries refuse before it
/// (behaviour measured on hardware).
pub(crate) static AUDIO_OUT_INITIALIZED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// One export: (NID, name, library, callback).
pub(crate) type Export = (&'static str, &'static str, &'static str, HleCallback);

fn all_exports() -> impl Iterator<Item = &'static Export> {
    audio_out::EXPORTS
        .iter()
        .chain(audio_out2::EXPORTS)
        .chain(audio_in::EXPORTS)
        .chain(propagation::EXPORTS)
        .chain(acm::EXPORTS)
        .chain(ajm::EXPORTS)
        .chain(audiodec::EXPORTS)
        .chain(audio3d::EXPORTS)
        .chain(voice::EXPORTS)
        .chain(opus_dec::EXPORTS)
}

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, lib, cb) in all_exports() {
        reg.register_fn(nid, name, lib, cb);
    }
    ngs2::register(reg);
}

#[cfg(test)]
mod tests {
    #[test]
    fn registered_nids_match_names() {
        let mut seen = std::collections::HashSet::new();
        for &(nid, name, _, _) in super::all_exports() {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
            assert!(seen.insert(nid), "duplicate NID {nid} ({name})");
        }
        assert!(seen.len() > 90, "only {} exports", seen.len());
    }
}
