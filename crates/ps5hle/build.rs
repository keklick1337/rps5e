// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Builds the FFmpeg media shim used by AvPlayer/Videodec2 when the host
//! has FFmpeg development libraries. Without them (or with
//! `RPS5E_NO_FFMPEG` set) the crate builds with host decoding disabled.

fn main() {
    println!("cargo:rustc-check-cfg=cfg(ps5_ffmpeg)");
    println!("cargo:rerun-if-changed=csrc/media.c");
    println!("cargo:rerun-if-env-changed=RPS5E_NO_FFMPEG");
    if std::env::var_os("RPS5E_NO_FFMPEG").is_some() {
        println!("cargo:warning=RPS5E_NO_FFMPEG set: movie playback disabled");
        return;
    }
    let mut includes = Vec::new();
    for lib in ["libavformat", "libavcodec", "libavutil", "libswscale", "libswresample"] {
        match pkg_config::Config::new().probe(lib) {
            Ok(l) => includes.extend(l.include_paths),
            Err(e) => {
                // Windows builds normally have no FFmpeg (see
                // BUILD.md): not worth a warning there.
                if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
                    println!("cargo:warning=FFmpeg ({lib}) not found, movie playback disabled: {e}");
                }
                return;
            }
        }
    }
    let mut build = cc::Build::new();
    build.file("csrc/media.c").warnings(false).flag_if_supported("-std=c11");
    for inc in includes {
        build.include(inc);
    }
    build.compile("ps5media");
    println!("cargo:rustc-cfg=ps5_ffmpeg");
}
