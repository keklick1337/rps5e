#!/usr/bin/env bash
# RustPS5Emu (RPS5E)
# Copyright (C) 2026 Vladislav Tislenko aka keklick1337
# SPDX-License-Identifier: GPL-2.0-only
#
# Record a play session's pad input, then replay it as a repeatable test with
# fps, frame and log output. See "Input recordings" in README.md.
#
#   scripts/input-test.sh record <game folder> <name>           play; Ctrl+C or close the window to stop
#   scripts/input-test.sh replay <game folder> <name> [opts]    replay, measure fps, dump frames
#   scripts/input-test.sh list
#
# A recording is named after the game's title ID (sce_sys/param.json).
# Recordings: local/recordings/<TITLE_ID>-<name>.txt (+ .data: the saves it starts from); results: local/tmp/input-test/<TITLE_ID>-<name>-<time>/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-$ROOT/target/release/rps5e-cli}"
REC_DIR="$ROOT/local/recordings"

# Title ID of a game folder (or of the folder holding the given eboot.bin).
title_id() {
    local dir="$1"
    [ -f "$dir" ] && dir="$(dirname "$dir")"
    python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1])).get("titleId", ""))' "$dir/sce_sys/param.json" 2>/dev/null || true
}

game_key() {
    local id; id="$(title_id "$1")"
    if [ -n "$id" ]; then echo "$id"; else basename "$1"; fi
}

wait_lock() {
    if pgrep -f "rps5e-cli run" >/dev/null; then echo "another emulator is running; waiting for it" >&2; fi
    flock "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/rps5e-run.lock" true
}

cmd="${1:-}"; shift || true
case "$cmd" in
record)
    [ $# -ge 2 ] || { echo "usage: $0 record <game> <name>" >&2; exit 2; }
    dir="$1"; name="$2"; shift 2
    [ -e "$dir" ] || { echo "no game folder '$dir'" >&2; exit 2; }
    key="$(game_key "$dir")"
    data="${DATA_DIR:-$ROOT/local/data/$key}"
    mkdir -p "$REC_DIR" "$data"
    rec="$REC_DIR/$key-$name.txt"
    # The saves the recording starts from; every replay starts from a copy.
    rm -rf "${REC_DIR:?}/$key-$name.data"
    cp -a "$data" "$REC_DIR/$key-$name.data"
    wait_lock
    echo "recording to $rec — play, then close the window or press Ctrl+C"
    # Recorded in the exact compile mode its replays use (draws wait for
    # their shaders), so the replay sees the same game timing.
    RPS5E_COMPILE_WAIT_MS="${RPS5E_COMPILE_WAIT_MS:-exact}" "$BIN" run "$dir" --record-input "$rec" --data-dir "$data" "$@" || true
    echo "saved $(grep -vc '^#' "$rec") pad changes, $(tail -1 "$rec" | cut -d' ' -f1) s → $rec"
    ;;
replay)
    [ $# -ge 2 ] || { echo "usage: $0 replay <game> <name> [extra run options]" >&2; exit 2; }
    dir="$1"; name="$2"; shift 2
    key="$(game_key "$dir")"
    rec="$REC_DIR/$key-$name.txt"
    [ -f "$rec" ] || { echo "no recording $rec" >&2; exit 2; }
    out="$ROOT/local/tmp/input-test/$key-$name-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$out"
    secs=$(awk '!/^#/ {t=$1} END {printf "%d", t + 0}' "$rec")
    secs=$((secs + ${TAIL_SECONDS:-15}))
    # Every replay starts from the saves the recording began with.
    data="$out/data"
    if [ -d "$REC_DIR/$key-$name.data" ]; then cp -a "$REC_DIR/$key-$name.data" "$data"; else mkdir -p "$data"; fi
    wait_lock
    # Shaders and pipelines the game made before are built first, outside
    # the timed run (the run would wait for them before the guest starts).
    echo "precompiling known shaders → $out/precompile.log"
    "$BIN" precompile "$dir" >"$out/precompile.log" 2>&1 || echo "precompile failed (see $out/precompile.log); the run builds them itself"
    wait_lock
    echo "replaying $rec for $secs s → $out"
    set +e
    # Exact mode: draws wait for their shaders (play skips them), so a
    # replay is repeatable.
    RPS5E_COMPILE_WAIT_MS="${RPS5E_COMPILE_WAIT_MS:-exact}" RPS5E_FRAME_TIMES="$out/frametimes.txt" RPS5E_DUMP_SCALE="${DUMP_SCALE:-4}" RPS5E_GPU_PROFILE=1 RUST_LOG="${RUST_LOG:-info}" timeout "$secs" "$BIN" run "$dir" --replay-input "$rec" \
        --data-dir "$data" --dump-frames "$out/frames" --dump-every "${DUMP_EVERY:-300}" "$@" >"$out/run.log" 2>&1
    code=$?
    set -e
    python3 - "$out/run.log" "$code" "$out/frametimes.txt" <<'PY' | tee "$out/summary.txt"
import re, sys, datetime
log, code = sys.argv[1], int(sys.argv[2])
pts = []
for line in open(log, errors="replace"):
    m = re.search(r"^\[(\S+Z) .*\[pm4\] .*\bflips=(\d+)", line)
    if m:
        t = datetime.datetime.fromisoformat(m.group(1).replace("Z", "+00:00")).timestamp()
        pts.append((t, int(m.group(2))))
fps = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(pts, pts[1:]) if b[0] > a[0]]
faults = sum(1 for l in open(log, errors="replace") if "GUEST FAULT" in l or "panicked" in l)
print(f"exit code {code} ({'timeout = ran to the end' if code == 124 else 'stopped early!'}), faults {faults}")
if fps:
    s = sorted(fps)
    print(f"fps per {len(fps)} intervals: avg {sum(fps)/len(fps):.1f}  min {s[0]:.1f}  p10 {s[len(s)//10]:.1f}  max {s[-1]:.1f}  flips {pts[-1][1]}")
    print("timeline (s fps):", " ".join(f"{int(b[0]-pts[0][0])}:{f:.0f}" for (a, b), f in zip(zip(pts, pts[1:]), fps)))
else:
    print("no [pm4] flip lines in the log")
# Per-frame times: freezes and frame pacing.
try:
    ft = [tuple(map(float, l.split())) for l in open(sys.argv[3]) if not l.startswith("#")]
except OSError:
    ft = []
if len(ft) > 2:
    d = sorted(x[1] for x in ft[1:])
    n = len(d)
    worst1 = d[int(n * 0.99):]
    print(f"frame time ms: median {d[n//2]:.1f}  p95 {d[int(n*0.95)]:.1f}  p99 {d[int(n*0.99)]:.1f}  max {d[-1]:.1f}  1%-low fps {1000/(sum(worst1)/len(worst1)):.1f}")
    print(f"frames over 17 ms: {sum(x > 17.5 for x in d)}  over 33 ms: {sum(x > 33.4 for x in d)}  over 50 ms: {sum(x > 50 for x in d)}  over 100 ms (freezes): {sum(x > 100 for x in d)}")
    fr = sorted(ft[1:], key=lambda x: -x[1])[:10]
    print("longest frames (at s: ms):", " ".join(f"{t/1000:.1f}:{dt:.0f}" for t, dt in sorted(fr)))
PY
    ;;
list)
    ls -1 "$REC_DIR" 2>/dev/null || echo "no recordings yet"
    ;;
*)
    sed -n '6,14p' "$0"; exit 2 ;;
esac
