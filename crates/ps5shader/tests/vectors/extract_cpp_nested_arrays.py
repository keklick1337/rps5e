# RustPS5Emu (RPS5E)
# Copyright (C) 2026 Vladislav Tislenko aka keklick1337
# SPDX-License-Identifier: GPL-2.0-only

"""Extract named constant arrays of a reference C++ execution test (in
$REFERENCE_TESTS, a directory of hardware execution tests) into Rust consts.

usage: extract_cpp_nested_arrays.py <file.cpp> <Name[=RUST_NAME]> ...
Nested brace groups become nested arrays; a group whose children differ in
shape (a struct of a scalar and an array) is flattened. Strings and
identifiers are dropped.
"""
import glob
import os
import re
import sys

D = os.environ.get("REFERENCE_TESTS") or next(
    iter(sorted(glob.glob("local/other_emus/*/core/libs/prx/libSceAgcDriver/tests/execution"))), "")
D += "/"

TOKEN = re.compile(r'\s*(?:(\{)|(\})|("(?:[^"\\]|\\.)*")|(0[xX][0-9a-fA-F]+|\d+)([uUlL]*)|([A-Za-z_][\w:]*)|(,)|(.))', re.S)


def parse(text, i):
    """Parse a brace group starting at text[i] == '{'; returns (list, end)."""
    assert text[i] == "{"
    i += 1
    out = []
    while True:
        m = TOKEN.match(text, i)
        if not m:
            raise ValueError("eof")
        i = m.end()
        if m.group(1):
            sub, i = parse(text, i - 1)
            out.append(sub)
        elif m.group(2):
            return out, i
        elif m.group(4):
            out.append(int(m.group(4), 0))
        elif m.group(8) == "-":
            m2 = TOKEN.match(text, i)
            i = m2.end()
            out.append((-int(m2.group(4), 0)) & 0xFFFFFFFF)


def shape(v):
    if isinstance(v, int):
        return ()
    shapes = [shape(c) for c in v]
    if shapes and all(s == shapes[0] for s in shapes):
        return (len(v),) + shapes[0]
    return ("flat", len(flatten(v)))


def flatten(v):
    if isinstance(v, int):
        return [v]
    out = []
    for c in v:
        out += flatten(c)
    return out


def normalize(v):
    if isinstance(v, int):
        return v
    v = [normalize(c) for c in v]
    shapes = [shape(c) for c in v]
    if shapes and not all(s == shapes[0] for s in shapes):
        # Mixed children at this level: flatten each child consistently.
        return flatten(v)
    return v


def rust_type(v, ty):
    if isinstance(v, int):
        return ty
    return f"[{rust_type(v[0], ty)}; {len(v)}]"


def rust_value(v, indent=0):
    if isinstance(v, int):
        return f"{v:#x}"
    if all(isinstance(c, int) for c in v):
        return "[" + ", ".join(f"{c:#x}" for c in v) + "]"
    pad = "    " * (indent + 1)
    return "[\n" + "".join(f"{pad}{rust_value(c, indent + 1)},\n" for c in v) + "    " * indent + "]"


def rust_name(name):
    return re.sub(r"(?<!^)([A-Z])", r"_\1", name).upper()


def main():
    src = open(D + sys.argv[1]).read()
    for spec in sys.argv[2:]:
        name, _, rname = spec.partition("=")
        m = re.search(r"constexpr\s+([\w:<>, ]+?)\s+" + name + r"\b[^={;]*(=\s*)?\{", src)
        if not m:
            raise SystemExit(f"{name} not found")
        start = m.end() - 1
        value, _ = parse(src, start)
        # std::array<Struct, N> Name{{ ... }}: unwrap the extra braces.
        while len(value) == 1 and isinstance(value[0], list) and "std::array" in m.group(1) and not all(isinstance(c, int) for c in value[0]):
            value = value[0]
        value = normalize(value)
        ty = "u64" if any(x > 0xFFFFFFFF for x in flatten(value)) or "uint64" in m.group(1) else "u32"
        print(f"pub const {rname or rust_name(name)}: {rust_type(value, ty)} = {rust_value(value)};")


main()
