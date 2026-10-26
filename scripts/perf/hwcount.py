#!/usr/bin/env python3
# RustPS5Emu (RPS5E)
# Copyright (C) 2026 Vladislav Tislenko aka keklick1337
# SPDX-License-Identifier: GPL-2.0-only

"""Per-thread hardware counters without root (perf_event_paranoid <= 2).

    scripts/perf/hwcount.py [options] -- <command> [args...]   # launch and count
    scripts/perf/hwcount.py [options] -p <pid> [-d <seconds>]  # attach to a running process

Counts user-space instructions, cycles, cache references and misses, LLC and
dTLB read misses, branch misses, page faults and task clock per thread, and
reads minor faults and context switches per thread from /proc (the kernel's
own switch counters: a context-switch event at paranoid 2 sees only user
mode). Prints one row per thread (or per thread-name group with --group),
plus the system-wide TLB-shootdown interrupt delta (/proc/interrupts).

Launch mode opens one inherited event per CPU on the child before it execs
(enable_on_exec, inherit_stat): every thread's counts arrive as a READ record
when it exits, so short-lived threads are complete. Attach mode opens events
per thread and rescans /proc/<pid>/task every --rescan seconds (threads that
start and end between scans are missed).

Never use it on a game run outside the one-emulator rule (AGENTS.md).
"""

import argparse
import json
import os
import re
import select
import signal
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perfev as pe  # noqa: E402

# Five hardware events by default: more than the PMU's free counters are
# multiplexed and extrapolated, which is unreliable for short-lived threads
# spread over many CPUs (add others with -e/-r and check the mux column).
ALL_EVENTS = [
    ("cycles", pe.TYPE_HARDWARE, pe.HW_CPU_CYCLES),
    ("instructions", pe.TYPE_HARDWARE, pe.HW_INSTRUCTIONS),
    ("cache-misses", pe.TYPE_HARDWARE, pe.HW_CACHE_MISSES),
    ("dtlb-misses", pe.TYPE_HW_CACHE, pe.cache_config(pe.CACHE_DTLB)),
    ("branch-misses", pe.TYPE_HARDWARE, pe.HW_BRANCH_MISSES),
    ("page-faults", pe.TYPE_SOFTWARE, pe.SW_PAGE_FAULTS),
    ("task-clock", pe.TYPE_SOFTWARE, pe.SW_TASK_CLOCK),
    ("cache-refs", pe.TYPE_HARDWARE, pe.HW_CACHE_REFERENCES),
    ("llc-misses", pe.TYPE_HW_CACHE, pe.cache_config(pe.CACHE_LL)),
    ("l1d-misses", pe.TYPE_HW_CACHE, pe.cache_config(pe.CACHE_L1D)),
]
EVENTS = ALL_EVENTS[:7]

FMT = pe.FMT_TOTAL_TIME_ENABLED | pe.FMT_TOTAL_TIME_RUNNING | pe.FMT_ID


def ncpus():
    return os.cpu_count() or 1


class Threads:
    """Thread names and /proc stats of a process, polled while it runs."""

    def __init__(self, pid):
        self.pid = pid
        self.names = {}
        self.proc = {}

    def poll(self):
        for t in pe.tids(self.pid):
            n = pe.thread_name(self.pid, t)
            if n:
                self.names[t] = n
            s = pe.thread_proc_stats(self.pid, t)
            if s:
                self.proc[t] = s


def parse_read(body, nevents_unused=None):
    """pid, tid, value, enabled, running, id of a READ record."""
    import struct
    pid, tid, v, en, run, eid = struct.unpack_from("<IIQQQQ", body, 0)
    return pid, tid, v, en, run, eid


def parse_comm(body):
    import struct
    pid, tid = struct.unpack_from("<II", body, 0)
    name = body[8:].split(b"\0", 1)[0].decode(errors="replace")
    return pid, tid, name


def launch(argv, events, interval):
    pid, release = pe.spawn_stopped(argv)
    rings = []
    fds = []  # (name, fd)
    ids = {}
    opened = []
    # Hardware events form one group per CPU (scheduled together: their
    # ratios stay exact if the PMU multiplexes), software events stand
    # alone. Values are not scaled: an inherited per-CPU event's enabled
    # time does not follow its thread across CPUs, so the kernel's scaling
    # factors are wrong for threads that migrate; the mux column reports
    # the share of a thread's run time the group was counting.
    hw_types = (pe.TYPE_HARDWARE, pe.TYPE_HW_CACHE, pe.TYPE_RAW)
    events = [e for e in events if e[1] in hw_types] + [e for e in events if e[1] not in hw_types]
    leaders = None
    for name, typ, cfg in events:
        cpu_fds = []
        member = typ in hw_types and leaders is not None
        try:
            for c in range(ncpus()):
                flags = pe.F_INHERIT | pe.F_INHERIT_STAT | pe.F_SAMPLE_ID_ALL
                if not member:
                    flags |= pe.F_DISABLED | pe.F_ENABLE_ON_EXEC
                if not opened:
                    flags |= pe.F_COMM | pe.F_TASK
                # SAMPLE_READ with inherit (no samples are taken) turns off
                # the kernel's swapping of equivalent thread contexts, which
                # otherwise exchanges values between threads (and between
                # events of different order).
                attr = pe.Attr(typ, cfg, sample_type=pe.SAMPLE_TID | pe.SAMPLE_READ, read_format=FMT, flags=flags)
                cpu_fds.append(pe.open_event(attr, pid, c, leaders[c] if member else -1))
        except OSError as e:
            for f in cpu_fds:
                os.close(f)
            print(f"hwcount: {name} unavailable ({e.strerror})", file=sys.stderr)
            continue
        if typ in hw_types and leaders is None:
            leaders = cpu_fds
            lead_name = name
        if not opened:
            rings = [pe.Ring(f, 16) for f in cpu_fds]
        else:
            for c, f in enumerate(cpu_fds):
                pe.ioctl(f, pe.IOC_SET_OUTPUT, opened[0][1][c])
        for f in cpu_fds:
            ids[pe.event_id(f)] = name
        opened.append((name, cpu_fds))
    if not opened:
        os.kill(pid, signal.SIGKILL)
        sys.exit("hwcount: no event could be opened")
    threads = Threads(pid)
    tlb0 = pe.tlb_shootdowns()
    t0 = time.monotonic()
    release()
    # raw[tid][event][cpu] = [value, enabled, running]. Every per-CPU event
    # of a thread is enabled while the thread runs anywhere but counts only
    # on its CPU: a thread's value is the sum over CPUs, scaled by its
    # enabled time over the summed running time (counter multiplexing).
    raw = {}
    comms = {}

    def drain():
        for cpu, r in enumerate(rings):
            for typ, _misc, body in r.records():
                if typ == pe.REC_READ:
                    _p, tid, v, en, run, eid = parse_read(body)
                    ev = ids.get(eid)
                    if ev:
                        acc = raw.setdefault(tid, {}).setdefault(ev, {}).setdefault(cpu, [0, 0, 0])
                        acc[0] += v
                        acc[1] += en
                        acc[2] += run
                elif typ == pe.REC_COMM:
                    _p, tid, name = parse_comm(body)
                    comms[tid] = name
                elif typ in (pe.REC_EXIT, pe.REC_FORK):
                    # pid, ppid, tid, ptid, time
                    seen.add(struct.unpack_from("<IIII", body, 0)[2])

    seen = set()
    status = None
    while True:
        threads.poll()
        drain()
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        time.sleep(interval)
    wall = time.monotonic() - t0
    tlb1 = pe.tlb_shootdowns()
    drain()
    # The original task: its events' values include the children's (merged
    # at their exit), per CPU.
    main = {}
    for name, cpu_fds in opened:
        per_cpu = {}
        for cpu, f in enumerate(cpu_fds):
            v, en, run = pe.read_value(f)
            for d in raw.values():
                c = d.get(name, {}).get(cpu)
                if c:
                    v, en, run = v - c[0], en - c[1], run - c[2]
            per_cpu[cpu] = [max(0, v), max(0, en), max(0, run)]
        main[name] = per_cpu
    # The kernel may swap a thread's events with another's of the same
    # process (equivalent contexts; inherit_stat keeps the values with the
    # threads): the original events, which report no READ record, end up
    # on whichever thread holds them at exit. Their remainder belongs to
    # the one thread that exited without a READ record.
    missing = sorted(t for t in seen | {pid} if t not in raw)
    owner = missing[0] if len(missing) == 1 else -1
    raw[owner] = main

    per_tid = {}
    for t, d in raw.items():
        row = {ev: sum(c[0] for c in cpus.values()) for ev, cpus in d.items()}
        if leaders is not None and row.get("task-clock"):
            run = sum(c[2] for c in d.get(lead_name, {}).values())
            row["mux"] = min(1.0, run / row["task-clock"])
        per_tid[t] = row
    names = dict(threads.names)
    names.update(comms)
    if owner == -1:
        names[-1] = "<unattributed>"
    rc = os.waitstatus_to_exitcode(status) if status is not None else None
    for _name, cpu_fds in opened:
        for f in cpu_fds:
            os.close(f)
    return per_tid, names, threads.proc, wall, (tlb1 - tlb0) if tlb0 is not None and tlb1 is not None else None, rc


def attach(pid, events, duration, rescan):
    threads = Threads(pid)
    open_fds = {}  # tid -> [(name, fd)]
    final = {}
    unavailable = set()
    tlb0 = pe.tlb_shootdowns()
    t0 = time.monotonic()
    stop = [False]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    while not stop[0]:
        threads.poll()
        for t in pe.tids(pid):
            if t in open_fds:
                continue
            lst = []
            for name, typ, cfg in events:
                if name in unavailable:
                    continue
                try:
                    lst.append((name, pe.open_event(pe.Attr(typ, cfg, read_format=FMT), t)))
                except OSError as e:
                    if e.errno not in (3,):  # ESRCH: the thread ended
                        unavailable.add(name)
                        print(f"hwcount: {name} unavailable ({e.strerror})", file=sys.stderr)
            open_fds[t] = lst
        if not os.path.exists(f"/proc/{pid}"):
            break
        if duration and time.monotonic() - t0 >= duration:
            break
        time.sleep(rescan)
    wall = time.monotonic() - t0
    tlb1 = pe.tlb_shootdowns()
    for t, lst in open_fds.items():
        d = {}
        mux = 1.0
        for name, fd in lst:
            try:
                v, en, run = pe.read_value(fd)
                d[name] = pe.scaled(v, en, run)
                if en:
                    mux = min(mux, run / en)
            except OSError:
                pass
            os.close(fd)
        d["mux"] = mux
        final[t] = d
    return final, threads.names, threads.proc, wall, (tlb1 - tlb0) if tlb0 is not None and tlb1 is not None else None, None


def group_name(n):
    return re.sub(r"\d+", "*", n)


def report(per_tid, names, proc, wall, tlb, rc, args):
    rows = []
    for tid, d in per_tid.items():
        p = proc.get(tid)
        r = dict(d)
        r["tid"] = tid
        r["name"] = names.get(tid, "?")
        if p:
            r["minflt"], r["majflt"], r["vcs"], r["ivcs"] = p
        rows.append(r)
    if args.group:
        g = {}
        for r in rows:
            k = group_name(r["name"])
            a = g.setdefault(k, {"name": k, "tid": 0, "threads": 0})
            a["threads"] += 1
            for key, v in r.items():
                if key in ("name", "tid"):
                    continue
                a[key] = min(a.get(key, 1.0), v) if key == "mux" else a.get(key, 0) + v
        rows = list(g.values())
    rows.sort(key=lambda r: -r.get("cycles", r.get("task-clock", 0)))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"wall_s": wall, "tlb_shootdowns": tlb, "exit": rc, "threads": rows}, f, indent=1)
    cols = ["task-clock", "cycles", "instructions", "ipc"] + [e for e in ("cache-misses", "dtlb-misses", "branch-misses", "cache-refs", "llc-misses", "l1d-misses") if any(e in r for r in rows)] + ["page-faults", "minflt", "vcs", "ivcs", "mux"]
    cols += [s.split("=", 1)[0] for s in args.raw]
    hdr = f"{'thread':<24}{'tid':>8}" + "".join(f"{c:>14}" for c in cols)
    print(hdr)
    shown = 0
    for r in rows:
        if r.get("cycles", 0) < args.min_cycles and r.get("task-clock", 0) == 0:
            continue
        shown += 1
        if args.top and shown > args.top:
            break
        cells = []
        for c in cols:
            if c == "ipc":
                cy = r.get("cycles", 0)
                cells.append(f"{r.get('instructions', 0) / cy:14.2f}" if cy else f"{'-':>14}")
            elif c == "mux":
                cells.append(f"{r.get('mux', 1.0):14.2f}")
            elif c == "task-clock":
                v = r.get(c)
                cells.append(f"{v / 1e6:12.1f}ms" if v is not None else f"{'-':>14}")
            else:
                v = r.get(c)
                cells.append(f"{v:14d}" if v is not None else f"{'-':>14}")
        tid = r["threads"] if args.group else r["tid"]
        print(f"{r['name'][:23]:<24}{tid:>8}" + "".join(cells))
    if args.per_kinstr:
        print("\nper 1000 instructions:")
        for r in rows[: args.top or len(rows)]:
            ins = r.get("instructions", 0)
            if not ins:
                continue
            parts = [f"{c}={1000 * r.get(c, 0) / ins:.3f}" for c in ("cache-misses", "llc-misses", "l1d-misses", "dtlb-misses", "branch-misses") if c in r]
            print(f"  {r['name'][:23]:<24}" + " ".join(parts))
    print("(mux: share of the run time the hardware group counted; below 1 the counts are low, their ratios hold)")
    print(f"\nwall {wall:.3f} s; TLB shootdowns (system-wide) {tlb if tlb is not None else '?'}" + (f"; exit {rc}" if rc is not None else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--pid", type=int, help="attach to a running process")
    ap.add_argument("-d", "--duration", type=float, default=0, help="attach: seconds to count (0: until it exits or Ctrl-C)")
    ap.add_argument("--rescan", type=float, default=0.1, help="attach: thread rescan period (s)")
    ap.add_argument("--interval", type=float, default=0.05, help="launch: /proc poll period (s)")
    ap.add_argument("-e", "--events", help="comma-separated events of: " + ",".join(e[0] for e in ALL_EVENTS) + " (default: the first seven)")
    ap.add_argument("-r", "--raw", action="append", default=[], help="extra raw PMU event NAME=0xCONFIG (cpu PMU encoding, e.g. AMD event|umask<<8)")
    ap.add_argument("-g", "--group", action="store_true", help="sum threads whose names differ only in digits")
    ap.add_argument("--top", type=int, default=0, help="show only the first N rows")
    ap.add_argument("--min-cycles", type=int, default=1, help="hide threads below this many cycles")
    ap.add_argument("--per-kinstr", action="store_true", help="also print misses per 1000 instructions")
    ap.add_argument("--json", help="write the rows as JSON")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    events = EVENTS
    if args.events:
        want = args.events.split(",")
        events = [e for e in ALL_EVENTS if e[0] in want]
    for spec in args.raw:
        n, cfg = spec.split("=", 1)
        events = events + [(n, pe.TYPE_RAW, int(cfg, 0))]
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if args.pid:
        res = attach(args.pid, events, args.duration, args.rescan)
    elif cmd:
        res = launch(cmd, events, args.interval)
    else:
        ap.error("give -p PID or -- command")
    report(*res, args)


if __name__ == "__main__":
    main()
