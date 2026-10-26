#!/usr/bin/env python3
# RustPS5Emu (RPS5E)
# Copyright (C) 2026 Vladislav Tislenko aka keklick1337
# SPDX-License-Identifier: GPL-2.0-only

"""Flicker / stretch scan over consecutive dumped frames.

    RPS5E_DUMP_WINDOW=300-330 RPS5E_DUMP_SCALE=4 rps5e-cli run GAME \
        --dump-frames DIR --dump-every 1 ...
    scripts/glitch_scan.py DIR [--out PREFIX] [--top N]

For each frame i with both neighbours, a pixel is *transient* when it differs
from frame i-1 and from frame i+1 by more than THRESH (luma, 0-255) while
frames i-1 and i+1 agree on it (within THRESH/2): it popped for one frame
(flicker, a model stretched for one frame, a garbage triangle). Camera motion
moves edges continuously, so the neighbours disagree there and it does not
count. Prints one line per frame and a summary; with --out, writes the N
worst frames as strips (previous | frame | next | transient mask).
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

THRESH = 24


def load(path):
    img = Image.open(path).convert("RGB")
    a = np.asarray(img, dtype=np.float32)
    return a, a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", help="prefix of the worst-frame strips (PNG)")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--thresh", type=float, default=THRESH)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.dir) if f.endswith(".ppm") or f.endswith(".png"))
    files = [f for f in files if "_cpu" not in f] or files
    if len(files) < 3:
        sys.exit(f"{a.dir}: need at least 3 frames, have {len(files)}")
    rgb, luma = [], []
    for f in files:
        c, l = load(os.path.join(a.dir, f))
        rgb.append(c)
        luma.append(l)
    rows = []
    for i in range(1, len(files) - 1):
        p, c, n = luma[i - 1], luma[i], luma[i + 1]
        if p.shape != c.shape or c.shape != n.shape:
            continue
        dp, dn, pn = np.abs(c - p), np.abs(c - n), np.abs(p - n)
        mask = (dp > a.thresh) & (dn > a.thresh) & (pn < a.thresh / 2)
        frac = float(mask.mean())
        rows.append((i, frac, float(dp.mean()), mask))
        if not a.quiet:
            print(f"{files[i]} transient={frac * 100:.3f}% diff_prev={dp.mean():.2f}")
    fr = np.array([r[1] for r in rows])
    spikes = int((fr > 0.005).sum())
    print(
        f"frames={len(files)} transient mean={fr.mean() * 100:.3f}% p95={np.percentile(fr, 95) * 100:.3f}% "
        f"max={fr.max() * 100:.3f}% spikes(>0.5%)={spikes} ({spikes / len(fr) * 100:.1f}% of frames)"
    )
    if a.out:
        for i, frac, _, mask in sorted(rows, key=lambda r: -r[1])[: a.top]:
            m = np.zeros_like(rgb[i])
            m[mask] = (255, 0, 255)
            strip = np.concatenate([rgb[i - 1], rgb[i], rgb[i + 1], m], axis=1).clip(0, 255).astype(np.uint8)
            name = f"{a.out}_{os.path.splitext(files[i])[0]}_{frac * 100:.2f}.png"
            Image.fromarray(strip).save(name)
            print(f"wrote {name}")


if __name__ == "__main__":
    main()
