#!/usr/bin/env bash
# RustPS5Emu (RPS5E)
# Copyright (C) 2026 Vladislav Tislenko aka keklick1337
# SPDX-License-Identifier: GPL-2.0-only
#
# Builds a small static FFmpeg (LGPL, FFmpeg's own decoders only, no
# external libraries) for release builds: the emulator links it in, so the
# packaged binary plays game movies without depending on the FFmpeg version
# of the user's distribution.
#
#   scripts/build-ffmpeg.sh <prefix>
#
# Then build with
#   PKG_CONFIG_PATH=<prefix>/lib/pkgconfig LIBAVFORMAT_STATIC=1 LIBAVCODEC_STATIC=1 \
#   LIBAVUTIL_STATIC=1 LIBSWSCALE_STATIC=1 LIBSWRESAMPLE_STATIC=1 cargo build --release
# (scripts/package.sh does this when RPS5E_FFMPEG_PREFIX is set).
# Needs a C compiler, make and nasm; FFMPEG_JOBS sets the build jobs.
set -euo pipefail
VERSION=8.0
SHA256=b2751fccb6cc4c77708113cd78b561059b6fa904b24162fa0be2d60273d27b8e
PREFIX="${1:?usage: $0 <prefix>}"
mkdir -p "$PREFIX"
PREFIX="$(cd "$PREFIX" && pwd)"
if [[ -f "$PREFIX/.built-$VERSION" ]]; then
    echo "FFmpeg $VERSION already in $PREFIX"
    exit 0
fi
WORK="$PREFIX/src"
mkdir -p "$WORK"
tarball="$WORK/ffmpeg-$VERSION.tar.xz"
[[ -f "$tarball" ]] || curl -sSfL -o "$tarball" "https://ffmpeg.org/releases/ffmpeg-$VERSION.tar.xz"
echo "$SHA256  $tarball" | sha256sum -c -
rm -rf "$WORK/ffmpeg-$VERSION"
tar -xJf "$tarball" -C "$WORK"
cd "$WORK/ffmpeg-$VERSION"
# What the AvPlayer / Videodec2 / Ajm shim (crates/ps5hle/csrc/media.c)
# opens: MP4 and WebM files with H.264, HEVC, VP8, VP9, AAC, Opus, Vorbis
# and MP3 streams, read through its own I/O callbacks.
./configure \
    --prefix="$PREFIX" \
    --enable-static --disable-shared --enable-pic \
    --disable-programs --disable-doc --disable-network --disable-autodetect \
    --disable-everything --disable-avdevice --disable-avfilter \
    --enable-avformat --enable-avcodec --enable-swscale --enable-swresample \
    --enable-demuxer=mov,matroska,mp3,aac,ogg,wav \
    --enable-decoder=h264,hevc,vp8,vp9,aac,aac_latm,opus,vorbis,mp3,mp3float,pcm_s16le,pcm_f32le \
    --enable-parser=h264,hevc,vp8,vp9,aac,aac_latm,opus,vorbis,mpegaudio \
    --enable-bsf=vp9_superframe_split,h264_mp4toannexb,hevc_mp4toannexb
make -j"${FFMPEG_JOBS:-$(nproc)}"
make install
# The archive carries FFmpeg's licence next to the binaries.
install -Dm644 COPYING.LGPLv2.1 "$PREFIX/share/ffmpeg/LICENSE.txt"
rm -rf "$WORK/ffmpeg-$VERSION"
touch "$PREFIX/.built-$VERSION"
echo "FFmpeg $VERSION installed in $PREFIX"
