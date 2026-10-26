# RustPS5Emu (RPS5E)
# Copyright (C) 2026 Vladislav Tislenko aka keklick1337
# SPDX-License-Identifier: GPL-2.0-only

"""Print `<distinct colours> <mean brightness>` of a binary PPM (P6) frame."""
import sys

def main(path):
    with open(path, "rb") as f:
        data = f.read()
    parts = data.split(b"\n", 3)
    w, h = (int(v) for v in parts[1].split())
    px = parts[3]
    step = 3 * max(1, (w * h) // 20000)
    colours = set()
    total = n = 0
    for i in range(0, len(px) - 2, step):
        c = px[i:i + 3]
        colours.add(c)
        total += c[0] + c[1] + c[2]
        n += 1
    print(len(colours), round(total / (3 * max(n, 1)), 2))

main(sys.argv[1])
