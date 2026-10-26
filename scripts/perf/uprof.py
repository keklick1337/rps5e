#!/usr/bin/env python3
# RustPS5Emu (RPS5E)
# Copyright (C) 2026 Vladislav Tislenko aka keklick1337
# SPDX-License-Identifier: GPL-2.0-only

"""Sampling profiler without root (perf_event_paranoid <= 2): user-space
samples with frame-pointer call chains, per thread, symbolised from the ELF
symbol tables.

    scripts/perf/uprof.py record -o prof.json.gz [-F 2000] -- <command> [args...]
    scripts/perf/uprof.py record -o prof.json.gz -p <pid> [-d <seconds>]
    scripts/perf/uprof.py report prof.json.gz [--thread RE] [--by-thread] [--tree] [--callers RE]
    scripts/perf/uprof.py annotate prof.json.gz 'SYMBOL_RE' [--thread RE]
    scripts/perf/uprof.py diff a.json.gz b.json.gz [--thread RE] [--scale-a N --scale-b M]
    scripts/perf/uprof.py run [record and report options] -- <command> [args...]   # record into local/tmp/uprof/, report

Call chains come from the kernel walking the user stack through frame
pointers: build with frame pointers for whole chains,

    RUSTFLAGS=-Cforce-frame-pointers=yes CARGO_TARGET_DIR=$PWD/tmp/target-fp cargo build --release

(the precompiled std and libc have none: chains through them are cut).

Launch mode opens one inherited sampling event per CPU on the child before
it execs, so every thread is sampled from its first instruction. Attach mode
opens one event per thread and rescans /proc/<pid>/task every --rescan
seconds. Mappings come from the kernel's MMAP2 records and from snapshots of
/proc/<pid>/maps (only non-empty ones are kept: a process that has exited
reads empty).

Never use it on a game run outside the one-emulator rule (AGENTS.md).
"""

import argparse
import bisect
import collections
import gzip
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perfev as pe  # noqa: E402

SAMPLE_TYPE = pe.SAMPLE_IP | pe.SAMPLE_TID | pe.SAMPLE_CALLCHAIN


def sample_attr(event, freq, extra_flags):
    flags = pe.F_DISABLED | pe.F_FREQ | pe.F_EXCLUDE_CALLCHAIN_KERNEL | pe.F_SAMPLE_ID_ALL | extra_flags
    if event == "cpu-clock":
        typ, cfg = pe.TYPE_SOFTWARE, pe.SW_CPU_CLOCK
    else:
        typ, cfg = pe.TYPE_HARDWARE, pe.HW_CPU_CYCLES
    return pe.Attr(typ, cfg, sample_period=freq, sample_type=SAMPLE_TYPE, flags=flags, max_stack=127)


class Recorder:
    def __init__(self, pid):
        self.pid = pid
        self.stacks = collections.Counter()  # (tid, ips tuple) -> samples
        self.names = {}
        self.maps = {}  # (start, end, pgoff, path) -> True, newest last
        self.lost = 0
        self.snapshots = 0
        self.last_maps = None

    def parse(self, typ, body):
        if typ == pe.REC_SAMPLE:
            ip, pid, tid, nr = struct.unpack_from("<QIIQ", body, 0)
            ips = struct.unpack_from(f"<{nr}Q", body, 24)
            chain = tuple(x for x in ips if x < pe.PERF_CONTEXT_MAX)
            if not chain:
                chain = (ip,)
            self.stacks[(tid, chain)] += 1
        elif typ == pe.REC_MMAP2:
            pid, tid, addr, length, pgoff = struct.unpack_from("<IIQQQ", body, 0)
            # maj, min, ino, ino_generation (or build id), prot, flags, filename
            prot, _flags = struct.unpack_from("<II", body, 56)
            name = body[64:].split(b"\0", 1)[0].decode(errors="replace")
            if prot & 4 and name.startswith("/"):
                self.maps[(addr, addr + length, pgoff, name)] = True
        elif typ == pe.REC_COMM:
            pid, tid = struct.unpack_from("<II", body, 0)
            self.names[tid] = body[8:].split(b"\0", 1)[0].decode(errors="replace")
        elif typ == pe.REC_LOST:
            _id, lost = struct.unpack_from("<QQ", body, 0)
            self.lost += lost

    def snapshot(self):
        for t in pe.tids(self.pid):
            n = pe.thread_name(self.pid, t)
            if n:
                self.names[t] = n
        try:
            with open(f"/proc/{self.pid}/maps") as f:
                text = f.read()
        except OSError:
            return
        if not text.strip() or text == self.last_maps:
            return
        self.last_maps = text
        self.snapshots += 1
        for line in text.splitlines():
            parts = line.split(None, 5)
            if len(parts) < 6 or "x" not in parts[1] or not parts[5].startswith("/"):
                continue
            a, b = (int(x, 16) for x in parts[0].split("-"))
            key = (a, b, int(parts[2], 16), parts[5].strip())
            self.maps.pop(key, None)
            self.maps[key] = True

    def save(self, path, meta):
        out = {
            "meta": meta,
            "names": {str(k): v for k, v in self.names.items()},
            "maps": [list(k) for k in self.maps],
            "stacks": [[tid, n, list(chain)] for (tid, chain), n in self.stacks.items()],
        }
        with gzip.open(path, "wt") as f:
            json.dump(out, f)


def record_launch(argv, freq, event, pages, poll):
    pid, release = pe.spawn_stopped(argv)
    rec = Recorder(pid)
    rings = []
    for c in range(os.cpu_count() or 1):
        attr = sample_attr(event, freq, pe.F_INHERIT | pe.F_ENABLE_ON_EXEC | pe.F_MMAP | pe.F_MMAP2 | pe.F_COMM | pe.F_TASK)
        try:
            fd = pe.open_event(attr, pid, c)
        except OSError as e:
            if event != "cpu-clock" and not rings:
                event = "cpu-clock"
                print(f"uprof: cycles unavailable ({e.strerror}), using cpu-clock", file=sys.stderr)
                fd = pe.open_event(sample_attr(event, freq, pe.F_INHERIT | pe.F_ENABLE_ON_EXEC | pe.F_MMAP | pe.F_MMAP2 | pe.F_COMM | pe.F_TASK), pid, c)
            else:
                raise
        rings.append(pe.Ring(fd, pages))
    t0 = time.monotonic()
    release()
    status = 0
    last_snap = 0.0
    while True:
        for r in rings:
            for typ, _m, body in r.records():
                rec.parse(typ, body)
        now = time.monotonic()
        if now - last_snap > 0.25:
            rec.snapshot()
            last_snap = now
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        time.sleep(poll)
    for r in rings:
        for typ, _m, body in r.records():
            rec.parse(typ, body)
        r.close()
        os.close(r.fd)
    return rec, {"command": argv, "event": event, "freq": freq, "wall_s": time.monotonic() - t0, "exit": os.waitstatus_to_exitcode(status)}


def record_attach(pid, freq, event, pages, poll, duration, rescan):
    rec = Recorder(pid)
    rings = {}
    t0 = time.monotonic()
    stop = [False]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    last_scan = -1e9
    while not stop[0]:
        now = time.monotonic()
        if now - last_scan >= rescan:
            last_scan = now
            rec.snapshot()
            for t in pe.tids(pid):
                if t in rings:
                    continue
                try:
                    fd = pe.open_event(sample_attr(event, freq, pe.F_COMM | pe.F_MMAP | pe.F_MMAP2), t)
                except OSError as e:
                    if e.errno == 3:
                        continue
                    if event != "cpu-clock":
                        event = "cpu-clock"
                        print(f"uprof: cycles unavailable ({e.strerror}), using cpu-clock", file=sys.stderr)
                        continue
                    raise
                pe.ioctl(fd, pe.IOC_ENABLE)
                rings[t] = pe.Ring(fd, pages)
        for r in rings.values():
            for typ, _m, body in r.records():
                rec.parse(typ, body)
        if not os.path.exists(f"/proc/{pid}") or (duration and now - t0 >= duration):
            break
        time.sleep(poll)
    for r in rings.values():
        for typ, _m, body in r.records():
            rec.parse(typ, body)
        r.close()
        os.close(r.fd)
    return rec, {"pid": pid, "event": event, "freq": freq, "wall_s": time.monotonic() - t0}


# ---------------------------------------------------------------- symbols

class Elf:
    """PT_LOAD segments and FUNC symbols of an ELF file."""

    def __init__(self, path):
        self.loads = []
        self.addrs = []
        self.syms = []
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            return
        if data[:4] != b"\x7fELF" or data[4] != 2:
            return
        e_phoff, e_shoff = struct.unpack_from("<QQ", data, 0x20)
        e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHHHH", data, 0x36)
        for i in range(e_phnum):
            p_type, _flags, p_offset, p_vaddr, _paddr, p_filesz, _memsz, _align = struct.unpack_from("<IIQQQQQQ", data, e_phoff + i * e_phentsize)
            if p_type == 1:
                self.loads.append((p_offset, p_vaddr, p_filesz))
        sections = [struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + i * e_shentsize) for i in range(e_shnum)] if e_shoff else []
        syms = {}
        for want in (2, 11):  # SHT_SYMTAB, SHT_DYNSYM
            for sh in sections:
                if sh[1] != want:
                    continue
                _name, _t, _fl, _addr, off, size, link, _info, _al, entsize = sh
                stroff = sections[link][4]
                for k in range(size // (entsize or 24)):
                    st_name, st_info, _other, shndx, st_value, st_size = struct.unpack_from("<IBBHQQ", data, off + k * 24)
                    if st_info & 0xF != 2 or shndx == 0 or st_value == 0:  # STT_FUNC, defined
                        continue
                    end = data.index(b"\0", stroff + st_name)
                    syms.setdefault(st_value, (st_size, data[stroff + st_name:end].decode(errors="replace")))
        for addr in sorted(syms):
            self.addrs.append(addr)
            self.syms.append((addr,) + syms[addr])

    def vaddr(self, file_off):
        for off, va, sz in self.loads:
            if off <= file_off < off + max(sz, 1):
                return file_off - off + va
        return None

    def lookup(self, va):
        i = bisect.bisect_right(self.addrs, va) - 1
        if i < 0:
            return None
        addr, size, name = self.syms[i]
        if size and va >= addr + size:
            # Past the symbol's end: still attribute nearby code (cold
            # parts, padding) to it if no other symbol follows soon.
            if va - (addr + size) > 4096:
                return None
        return name


HASH = re.compile(r"::h[0-9a-f]{16}$")
CRATE_HASH = re.compile(r"\[[0-9a-f]{16}\]")


def demangle_all(names):
    names = sorted(set(names))
    out = {n: n for n in names}
    if shutil.which("c++filt") and names:
        try:
            r = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True, timeout=120)
            lines = r.stdout.split("\n")
            if len(lines) >= len(names):
                out = dict(zip(names, lines))
        except (OSError, subprocess.SubprocessError):
            pass
    return {k: CRATE_HASH.sub("", HASH.sub("", v)) for k, v in out.items()}


class Symbolizer:
    def __init__(self, maps):
        self.maps = sorted((tuple(m) for m in maps), key=lambda m: m[0])
        self.starts = [m[0] for m in self.maps]
        self.elfs = {}
        self.cache = {}
        self.where = {}  # ip -> (path, ELF virtual address)

    def raw(self, ip):
        if ip in self.cache:
            return self.cache[ip]
        i = bisect.bisect_right(self.starts, ip) - 1
        res = ("[unknown]", "?")
        # The newest mapping that holds ip (snapshots may repeat ranges).
        while i >= 0:
            start, end, pgoff, path = self.maps[i]
            if start <= ip < end:
                elf = self.elfs.get(path)
                if elf is None:
                    elf = self.elfs[path] = Elf(path)
                mod = os.path.basename(path)
                va = elf.vaddr(ip - start + pgoff)
                name = elf.lookup(va) if va is not None else None
                res = (name, mod) if name else (f"[{mod}+{ip - start + pgoff:#x}]", mod)
                self.where[ip] = (path, va)
                break
            if start + (1 << 40) < ip:
                break
            i -= 1
        self.cache[ip] = res
        return res


# ---------------------------------------------------------------- report

def load(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def selected(path, args):
    """(meta, names, stacks [(tid, thread name, samples, demangled frames)])."""
    d = load(path)
    names = {int(k): v for k, v in d["names"].items()}
    sym = Symbolizer(d["maps"])
    thread_re = re.compile(args.thread) if args.thread else None
    stacks = []
    raw_names = set()
    for tid, n, chain in d["stacks"]:
        tname = names.get(tid, str(tid))
        if thread_re and not thread_re.search(tname):
            continue
        frames = [sym.raw(ip if k == 0 else ip - 1)[0] for k, ip in enumerate(chain)]
        raw_names.update(frames)
        stacks.append((tid, tname, n, frames))
    dem = demangle_all(raw_names)
    return d.get("meta", {}), [(t, tn, n, [dem.get(f, f) for f in fr]) for t, tn, n, fr in stacks]


def annotate(path, args):
    """The hottest sampled addresses inside symbols matching RE, with their
    source lines and inlined frames (addr2line)."""
    d = load(path)
    names = {int(k): v for k, v in d["names"].items()}
    sym = Symbolizer(d["maps"])
    thread_re = re.compile(args.thread) if args.thread else None
    raw = collections.Counter()
    for tid, n, chain in d["stacks"]:
        if thread_re and not thread_re.search(names.get(tid, str(tid))):
            continue
        raw[chain[0]] += n
    dem = demangle_all({sym.raw(ip)[0] for ip in raw})
    sre = re.compile(args.symbol)
    hits = collections.Counter()
    for ip, n in raw.items():
        name = dem.get(sym.raw(ip)[0], "")
        if sre.search(name) and ip in sym.where:
            hits[sym.where[ip]] += n
    total = sum(hits.values())
    print(f"uprof annotate /{args.symbol}/: {total} samples")
    by_file = collections.defaultdict(list)
    top = hits.most_common(args.top)
    for (f, va), n in top:
        by_file[f].append(va)
    lines = {}
    for f, vas in by_file.items():
        if not shutil.which("addr2line"):
            break
        r = subprocess.run(["addr2line", "-e", f, "-f", "-i", "-C", "-a"] + [hex(v) for v in vas], capture_output=True, text=True)
        cur = None
        for line in r.stdout.splitlines():
            if line.startswith("0x"):
                cur = (f, int(line, 16))
                lines[cur] = []
            elif cur:
                lines[cur].append(CRATE_HASH.sub("", HASH.sub("", line)))
    for key, n in top:
        print(f"{100.0 * n / total:6.2f}% {n:6d}  {key[1]:#x}")
        info = lines.get(key, [])
        # addr2line -i: function / file:line pairs, innermost first.
        for k in range(0, len(info) - 1, 2):
            print(f"          {info[k + 1].split('/')[-1]:<40} {info[k][:110]}")


def diff(a, b, args):
    """Per-symbol self (or inclusive) samples of two recordings of the same
    work: B - A, largest growth first."""
    out = []
    for side, path in enumerate((a, b)):
        sel = argparse.Namespace(thread=args.thread_b if side == 1 and args.thread_b else args.thread)
        _meta, stacks = selected(path, sel)
        c = collections.Counter()
        for _t, _tn, n, frames in stacks:
            if args.inclusive:
                for f in set(frames):
                    c[f] += n
            else:
                c[frames[0]] += n
        out.append((c, sum(s[2] for s in stacks)))
    (ca, ta), (cb, tb) = out
    sa = args.scale_a or 1.0
    sb = args.scale_b or 1.0
    print(f"uprof diff: A {a} ({ta} samples, /{sa:g}), B {b} ({tb} samples, /{sb:g}); {'inclusive' if args.inclusive else 'self'} samples per unit")
    print(f"{'A':>10}{'B':>10}{'B-A':>10}  symbol")
    keys = set(ca) | set(cb)
    rows = sorted(keys, key=lambda k: -(cb[k] / sb - ca[k] / sa))
    for k in rows[: args.top]:
        print(f"{ca[k] / sa:10.1f}{cb[k] / sb:10.1f}{cb[k] / sb - ca[k] / sa:10.1f}  {k}")
    print(f"{ta / sa:10.1f}{tb / sb:10.1f}{tb / sb - ta / sa:10.1f}  (total)")


def report(path, args):
    d = load(path)
    names = {int(k): v for k, v in d["names"].items()}
    sym = Symbolizer(d["maps"])
    thread_re = re.compile(args.thread) if args.thread else None
    stacks = []
    raw_names = set()
    for tid, n, chain in d["stacks"]:
        tname = names.get(tid, str(tid))
        if thread_re and not thread_re.search(tname):
            continue
        # Return addresses (callers) point after the call: look up ip - 1.
        frames = [sym.raw(ip if k == 0 else ip - 1)[0] for k, ip in enumerate(chain)]
        raw_names.update(frames)
        stacks.append((tid, tname, n, frames))
    dem = demangle_all(raw_names)
    total = sum(s[2] for s in stacks)
    meta = d.get("meta", {})
    print(f"uprof: {path}: {total} samples ({meta.get('event', '?')} @ {meta.get('freq', '?')} Hz, wall {meta.get('wall_s', 0):.2f} s)")
    if not total:
        return
    pct = lambda n: 100.0 * n / total  # noqa: E731

    by_thread = collections.Counter()
    for tid, tname, n, _f in stacks:
        by_thread[re.sub(r"\d+", "*", tname) if args.group else f"{tname} [{tid}]"] += n
    print("\nthreads:")
    for t, n in by_thread.most_common(args.threads):
        print(f"  {pct(n):6.2f}%  {n:8d}  {t}")

    def flat(sel, title):
        self_c = collections.Counter()
        incl = collections.Counter()
        tot = 0
        for _tid, _tn, n, frames in sel:
            fr = [dem.get(f, f) for f in frames]
            self_c[fr[0]] += n
            for f in set(fr):
                incl[f] += n
            tot += n
        print(f"\n{title} ({tot} samples): self% incl% symbol")
        key = incl if args.inclusive else self_c
        for f, _ in key.most_common(args.top):
            print(f"  {100.0 * self_c[f] / tot:6.2f}% {100.0 * incl[f] / tot:6.2f}%  {f}")

    if args.by_thread:
        groups = collections.defaultdict(list)
        for s in stacks:
            groups[re.sub(r"\d+", "*", s[1]) if args.group else f"{s[1]} [{s[0]}]"].append(s)
        for g, _ in by_thread.most_common(args.threads):
            flat(groups[g], f"thread {g}")
    else:
        flat(stacks, "all threads")

    if args.tree:
        root = {}

        def add(node, frames, n):
            for f in frames:
                c = node.setdefault(f, [0, {}])
                c[0] += n
                node = c[1]

        for _tid, _tn, n, frames in stacks:
            add(root, [dem.get(f, f) for f in reversed(frames)], n)
        print(f"\ntop-down tree (>= {args.min}%):")
        walk(root, total, args.min, 0, args.depth)

    if args.callers:
        cre = re.compile(args.callers)
        root = {}
        for _tid, _tn, n, frames in stacks:
            fr = [dem.get(f, f) for f in frames]
            for k, f in enumerate(fr):
                if cre.search(f):
                    node = root
                    for g in fr[k:]:
                        c = node.setdefault(g, [0, {}])
                        c[0] += n
                        node = c[1]
                    break
        print(f"\ncallers of /{args.callers}/ (bottom-up, >= {args.min}%):")
        walk(root, total, args.min, 0, args.depth)


def walk(node, total, min_pct, depth, max_depth):
    if depth >= max_depth:
        return
    for f, (n, kids) in sorted(node.items(), key=lambda kv: -kv[1][0]):
        p = 100.0 * n / total
        if p < min_pct:
            continue
        print(f"{'  ' * depth}{p:6.2f}%  {f}")
        walk(kids, total, min_pct, depth + 1, max_depth)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd_name", required=True)
    r = sub.add_parser("record", help="sample a command or a process into a file")
    r.add_argument("-o", "--output", required=True)
    rp = sub.add_parser("report", help="report a recorded file")
    rp.add_argument("file")
    an = sub.add_parser("annotate", help="hottest addresses of symbols matching RE, with source lines and inlined frames")
    an.add_argument("file")
    an.add_argument("symbol")
    an.add_argument("--thread")
    an.add_argument("--top", type=int, default=15)
    df = sub.add_parser("diff", help="per-symbol growth from recording A to B")
    df.add_argument("a")
    df.add_argument("b")
    df.add_argument("--scale-a", type=float, help="divide A's samples by this (work units, e.g. draws)")
    df.add_argument("--scale-b", type=float)
    df.add_argument("--thread", help="threads of A (and of B unless --thread-b)")
    df.add_argument("--thread-b", help="threads of B")
    df.add_argument("--inclusive", action="store_true")
    df.add_argument("--top", type=int, default=40)
    run = sub.add_parser("run", help="record into local/tmp/uprof/ and report")
    for p in (r, run):
        p.add_argument("-F", "--freq", type=int, default=2000, help="samples per second per thread")
        p.add_argument("--event", choices=("cycles", "cpu-clock"), default="cycles")
        p.add_argument("-p", "--pid", type=int)
        p.add_argument("-d", "--duration", type=float, default=0)
        p.add_argument("--rescan", type=float, default=0.1)
        p.add_argument("--pages", type=int, default=128, help="ring pages per event")
        p.add_argument("--poll", type=float, default=0.01)
    for p in (rp, run):
        p.add_argument("--thread", help="only threads whose name matches RE")
        p.add_argument("--by-thread", action="store_true", help="a flat profile per thread")
        p.add_argument("-g", "--group", action="store_true", help="merge threads whose names differ only in digits")
        p.add_argument("--inclusive", action="store_true", help="sort by inclusive samples")
        p.add_argument("--top", type=int, default=40)
        p.add_argument("--threads", type=int, default=12)
        p.add_argument("--tree", action="store_true", help="top-down call tree")
        p.add_argument("--callers", help="bottom-up tree of the callers of symbols matching RE")
        p.add_argument("--min", type=float, default=1.0, help="tree cutoff in percent")
        p.add_argument("--depth", type=int, default=30)
    for p in (r, run):
        p.add_argument("rest", nargs=argparse.REMAINDER, help="-- command [args...]")
    args = ap.parse_args()

    def do_record(a, out):
        rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
        if a.pid:
            rec, meta = record_attach(a.pid, a.freq, a.event, a.pages, a.poll, a.duration, a.rescan)
        elif rest:
            rec, meta = record_launch(rest, a.freq, a.event, a.pages, a.poll)
        else:
            ap.error("give -p PID or -- command")
        meta["lost"] = rec.lost
        meta["maps_snapshots"] = rec.snapshots
        rec.save(out, meta)
        print(f"uprof: {sum(rec.stacks.values())} samples, {len(rec.stacks)} stacks, {rec.lost} lost, {rec.snapshots} maps snapshots -> {out}", file=sys.stderr)

    if args.cmd_name == "record":
        do_record(args, args.output)
    elif args.cmd_name == "report":
        report(args.file, args)
    elif args.cmd_name == "annotate":
        annotate(args.file, args)
    elif args.cmd_name == "diff":
        diff(args.a, args.b, args)
    else:
        repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        out_dir = os.path.join(repo, "tmp", "uprof")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"prof-{int(time.time())}.json.gz")
        do_record(args, out)
        report(out, args)


if __name__ == "__main__":
    main()
