#!/usr/bin/env python3
# RustPS5Emu (RPS5E)
# Copyright (C) 2026 Vladislav Tislenko aka keklick1337
# SPDX-License-Identifier: GPL-2.0-only

"""Extract constant arrays from a C++ test source into Rust constants.

Usage: extract_cpp_arrays.py <file.cpp> [Name ...]

Handles
  constexpr std::array<std::uint32_t, N> Name{ ... };   -> const NAME: [u32; N]
  constexpr Struct Name[] = { {a, b, c}, ... };          -> const NAME: [[u32; K]; M]
  constexpr std::uint32_t Name[M][K] = { {...}, ... };   -> const NAME: [[u32; K]; M]
(numbers keep their hex spelling; u/U/ull suffixes are dropped).
Used to port reference test vectors; the generated text is pasted into the
Rust test with the check logic written by hand.
"""
import re
import sys


def clean(tok):
    tok = tok.strip()
    tok = re.sub(r"(?i)(ull|ul|u|ll)$", "", tok)
    return tok


def to_rust_name(name):
    out = re.sub(r"(?<!^)([A-Z])", r"_\1", name).upper()
    return out


def main():
    src = open(sys.argv[1]).read()
    wanted = set(sys.argv[2:])
    for m in re.finditer(r"constexpr\s+std::array<std::uint(\d+)_t,\s*(\d+)>\s+(\w+)\s*\{(.*?)\};", src, re.S):
        bits, n, name, body = m.groups()
        if wanted and name not in wanted:
            continue
        vals = [clean(v) for v in body.split(",") if v.strip()]
        print(f"const {to_rust_name(name)}: [u{bits}; {len(vals)}] = [{', '.join(vals)}];")
    for m in re.finditer(r"constexpr\s+(\w+)\s+(\w+)\[\]\s*=\s*\{(.*?)\n\};", src, re.S):
        ty, name, body = m.groups()
        if wanted and name not in wanted:
            continue
        rows = re.findall(r"\{([^{}]*)\}", body)
        rows = [[clean(v) for v in r.split(",") if v.strip()] for r in rows]
        if not rows:
            continue
        k = len(rows[0])
        print(f"// {ty}")
        print(f"const {to_rust_name(name)}: [[u32; {k}]; {len(rows)}] = [")
        for r in rows:
            print("    [" + ", ".join(r) + "],")
        print("];")
    for m in re.finditer(r"constexpr\s+std::uint(\d+)_t\s+(\w+)\[(\d+)\]\[(\d+)\]\s*=\s*\{(.*?)\n\};", src, re.S):
        bits, name, n, k, body = m.groups()
        if wanted and name not in wanted:
            continue
        rows = re.findall(r"\{([^{}]*)\}", body)
        rows = [[clean(v) for v in r.split(",") if v.strip()] for r in rows]
        print(f"const {to_rust_name(name)}: [[u{bits}; {k}]; {n}] = [")
        for r in rows:
            print("    [" + ", ".join(r) + "],")
        print("];")


main()
