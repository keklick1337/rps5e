#!/usr/bin/env bash
# RustPS5Emu (RPS5E)
# Copyright (C) 2026 Vladislav Tislenko aka keklick1337
# SPDX-License-Identifier: GPL-2.0-only
#
# Release archives: builds the `dist` profile (release without debug info)
# and packs the binaries with the docs, the desktop entry and the icons.
#
#   scripts/package.sh [linux|windows ...]     (default: linux)
#
#   linux    dist/rps5e-<version>-linux-x86_64.tar.gz
#   windows  dist/rps5e-<version>-windows-x86_64.zip (x86_64-pc-windows-gnu;
#            needs the MinGW-w64 cross toolchain, see BUILD.md), with
#            SDL3.dll (controllers, audio) from the official SDL release
#
# Environment: CARGO (default cargo; e.g. a wrapper that sets up the Windows
# cross toolchain), OUT (default <repo>/dist), BUILD=0 packs existing binaries,
# SDL3_ZIP (an already downloaded SDL3-<version>-win32-x64.zip; otherwise it is
# downloaded once into $OUT/cache).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARGO="${CARGO:-cargo}"
OUT="${OUT:-$ROOT/dist}"
VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$ROOT/Cargo.toml" | head -n1)"
[[ -n "$VERSION" ]] || { echo "package: no version in Cargo.toml" >&2; exit 1; }
TARGETS=("$@")
[[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=(linux)

# The files every archive carries next to the binaries.
common_files() {
    local dest="$1"
    install -m644 "$ROOT/README.md" "$ROOT/LICENSE" "$ROOT/CHANGELOG.md" "$ROOT/rps5e-input.example.toml" "$dest/"
    mkdir -p "$dest/assets"
    install -m644 "$ROOT/assets/rps5e-logo.svg" "$ROOT/assets/rps5e.ico" "$dest/assets/"
}

package_linux() {
    local name="rps5e-$VERSION-linux-x86_64" bin="$ROOT/target/dist"
    if [[ "${BUILD:-1}" != 0 ]]; then
        "$CARGO" build --profile dist --manifest-path "$ROOT/Cargo.toml" --bin rps5e --bin rps5e-cli
    fi
    local stage="$OUT/$name"
    rm -rf "$stage" && mkdir -p "$stage"
    install -m755 "$bin/rps5e" "$bin/rps5e-cli" "$stage/"
    common_files "$stage"
    # The layout of a prefix: copy share/ into ~/.local/share or /usr/share.
    install -Dm644 "$ROOT/packaging/rps5e.desktop" "$stage/share/applications/rps5e.desktop"
    for f in "$ROOT"/assets/icons/rps5e-*.png; do
        local s; s="$(basename "$f" .png)"; s="${s#rps5e-}"
        install -Dm644 "$f" "$stage/share/icons/hicolor/${s}x${s}/apps/rps5e.png"
    done
    install -Dm644 "$ROOT/assets/rps5e-logo.svg" "$stage/share/icons/hicolor/scalable/apps/rps5e.svg"
    tar -C "$OUT" -czf "$OUT/$name.tar.gz" "$name"
    rm -rf "$stage"
    echo "$OUT/$name.tar.gz"
}

# The SDL release the Windows archive carries (the emulator loads SDL3.dll at
# run time for controllers and audio; Windows has none of its own).
SDL3_VERSION=3.4.20
SDL3_SHA256=368bed8b24e44ad9ea372990814f2343fcfb9e468abb7f8f4a3bc146c7304752

# Put SDL3.dll and its licence into directory $1.
sdl3_windows() {
    local dest="$1" zip="${SDL3_ZIP:-}" name="SDL3-$SDL3_VERSION-win32-x64.zip"
    if [[ -z "$zip" ]]; then
        zip="$OUT/cache/$name"
        if [[ ! -f "$zip" ]]; then
            mkdir -p "$OUT/cache"
            curl -fL --retry 3 -o "$zip.part" "https://github.com/libsdl-org/SDL/releases/download/release-$SDL3_VERSION/$name"
            mv "$zip.part" "$zip"
        fi
    fi
    echo "$SDL3_SHA256  $zip" | sha256sum -c --quiet - || { echo "package: $zip is not the SDL $SDL3_VERSION release archive" >&2; exit 1; }
    python3 -I - "$zip" "$dest" <<'PY'
import os, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
for src, dst in (("SDL3.dll", "SDL3.dll"), ("LICENSE.txt", "SDL3-LICENSE.txt")):
    with open(os.path.join(sys.argv[2], dst), "wb") as f:
        f.write(z.read(src))
PY
}

package_windows() {
    local triple=x86_64-pc-windows-gnu
    local name="rps5e-$VERSION-windows-x86_64" bin="$ROOT/target/$triple/dist"
    if [[ "${BUILD:-1}" != 0 ]]; then
        "$CARGO" build --profile dist --target "$triple" --manifest-path "$ROOT/Cargo.toml" --bin rps5e --bin rps5e-cli
    fi
    local stage="$OUT/$name"
    rm -rf "$stage" && mkdir -p "$stage"
    install -m755 "$bin/rps5e.exe" "$bin/rps5e-cli.exe" "$stage/"
    common_files "$stage"
    sdl3_windows "$stage"
    rm -f "$OUT/$name.zip"
    python3 -I - "$OUT" "$name" <<'PY'
import os, sys, zipfile
out, name = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(os.path.join(out, name + ".zip"), "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for d, _, files in os.walk(os.path.join(out, name)):
        for f in sorted(files):
            p = os.path.join(d, f)
            z.write(p, os.path.relpath(p, out))
PY
    rm -rf "$stage"
    echo "$OUT/$name.zip"
}

mkdir -p "$OUT"
for t in "${TARGETS[@]}"; do
    case "$t" in
        linux) package_linux ;;
        windows) package_windows ;;
        *) echo "package: unknown target '$t' (linux, windows)" >&2; exit 2 ;;
    esac
done
