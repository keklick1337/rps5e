#!/usr/bin/env bash
# RustPS5Emu (RPS5E)
# Copyright (C) 2026 Vladislav Tislenko aka keklick1337
# SPDX-License-Identifier: GPL-2.0-only
#
# Launch a PS5 game with rps5e.
#
#   scripts/run-game.sh <game folder | eboot.bin> [rps5e-cli run options...]
#
# Environment (all optional):
#   DATA_DIR     game save data / settings      (default: the emulator's, ~/.local/share/rps5e)
#   CONFIG_DIR   emulator input settings        (default: the emulator's, ~/.config/rps5e)
#   RUST_LOG     log level                      (default: info)
#   BUILD=0      skip `cargo build --release`
#
# Any extra argument goes to `rps5e-cli run`, e.g.
#   --input-mode keyboard            keyboard only (keyboard_mouse, controller, auto)
#   --bind Cross=KEY:Space           rebind an action (repeatable)
#   --input-config my-input.toml     a whole mapping file
#   --fullscreen  --no-input  --pad-script 20:cross  --dump-frames DIR
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
GAME="$1"; shift
if [[ ! -e "$GAME" ]]; then
    echo "run-game: '$GAME' not found" >&2
    exit 1
fi
if [[ "${BUILD:-1}" != 0 ]]; then
    cargo build --release --manifest-path "$ROOT/Cargo.toml" --bin rps5e-cli
fi
export RUST_LOG="${RUST_LOG:-info}"
dirs=()
[[ -n "${DATA_DIR:-}" ]] && dirs+=(--data-dir "$DATA_DIR")
[[ -n "${CONFIG_DIR:-}" ]] && dirs+=(--config-dir "$CONFIG_DIR")
exec "$ROOT/target/release/rps5e-cli" run "$GAME" "${dirs[@]}" "$@"
