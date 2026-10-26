# RustPS5Emu (RPS5E)
# Copyright (C) 2026 Vladislav Tislenko aka keklick1337
# SPDX-License-Identifier: GPL-2.0-only

"""perf_event_open(2) from Python (ctypes), for the unprivileged tools in
this directory (perf_event_paranoid=2: user-space events of our own
processes only, so every event sets exclude_kernel/exclude_hv).

Shared by uprof.py (sampling) and hwcount.py (counting). x86-64 Linux.
"""

import ctypes
import mmap
import os
import struct

SYS_PERF_EVENT_OPEN = 298  # x86-64

# perf_type_id
TYPE_HARDWARE, TYPE_SOFTWARE, TYPE_TRACEPOINT, TYPE_HW_CACHE, TYPE_RAW = 0, 1, 2, 3, 4

# PERF_COUNT_HW_*
HW_CPU_CYCLES, HW_INSTRUCTIONS, HW_CACHE_REFERENCES, HW_CACHE_MISSES = 0, 1, 2, 3
HW_BRANCH_INSTRUCTIONS, HW_BRANCH_MISSES = 4, 5
HW_STALLED_FRONTEND, HW_STALLED_BACKEND = 7, 8

# PERF_COUNT_SW_*
SW_CPU_CLOCK, SW_TASK_CLOCK, SW_PAGE_FAULTS, SW_CONTEXT_SWITCHES = 0, 1, 2, 3
SW_CPU_MIGRATIONS, SW_PAGE_FAULTS_MIN, SW_PAGE_FAULTS_MAJ = 4, 5, 6

# PERF_COUNT_HW_CACHE_* (config = id | op << 8 | result << 16)
CACHE_L1D, CACHE_L1I, CACHE_LL, CACHE_DTLB, CACHE_ITLB = 0, 1, 2, 3, 4
CACHE_OP_READ, CACHE_RESULT_ACCESS, CACHE_RESULT_MISS = 0, 0, 1


def cache_config(cache, op=CACHE_OP_READ, result=CACHE_RESULT_MISS):
    return cache | (op << 8) | (result << 16)


# sample_type
SAMPLE_IP, SAMPLE_TID, SAMPLE_TIME, SAMPLE_ADDR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
SAMPLE_READ, SAMPLE_CALLCHAIN, SAMPLE_ID, SAMPLE_CPU = 1 << 4, 1 << 5, 1 << 6, 1 << 7
SAMPLE_PERIOD, SAMPLE_STREAM_ID, SAMPLE_IDENTIFIER = 1 << 8, 1 << 9, 1 << 16

# read_format
FMT_TOTAL_TIME_ENABLED, FMT_TOTAL_TIME_RUNNING, FMT_ID, FMT_GROUP = 1, 2, 4, 8

# attr flag bits
F_DISABLED, F_INHERIT = 1 << 0, 1 << 1
F_EXCLUDE_KERNEL, F_EXCLUDE_HV = 1 << 5, 1 << 6
F_MMAP, F_COMM, F_FREQ, F_INHERIT_STAT, F_ENABLE_ON_EXEC, F_TASK = 1 << 8, 1 << 9, 1 << 10, 1 << 11, 1 << 12, 1 << 13
F_SAMPLE_ID_ALL = 1 << 18
F_EXCLUDE_CALLCHAIN_KERNEL = 1 << 21
F_MMAP2, F_COMM_EXEC = 1 << 23, 1 << 24
F_INHERIT_THREAD = 1 << 35

# record types
REC_MMAP, REC_LOST, REC_COMM, REC_EXIT, REC_THROTTLE, REC_UNTHROTTLE = 1, 2, 3, 4, 5, 6
REC_FORK, REC_READ, REC_SAMPLE, REC_MMAP2 = 7, 8, 9, 10

PERF_CONTEXT_MAX = (1 << 64) - 4095  # callchain context markers are >= this

# ioctls
IOC_ENABLE, IOC_DISABLE, IOC_RESET = 0x2400, 0x2401, 0x2403
IOC_SET_OUTPUT = 0x2405
IOC_ID = 0x80082407

ATTR_SIZE = 112  # PERF_ATTR_SIZE_VER5

_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long


class Attr:
    """A perf_event_attr (VER5 layout)."""

    def __init__(self, type_, config, sample_period=0, sample_type=0, read_format=0, flags=0, wakeup_events=0, max_stack=0):
        self.type = type_
        self.config = config
        self.sample_period = sample_period
        self.sample_type = sample_type
        self.read_format = read_format
        self.flags = flags | F_EXCLUDE_KERNEL | F_EXCLUDE_HV
        self.wakeup_events = wakeup_events
        self.max_stack = max_stack

    def pack(self):
        b = bytearray(ATTR_SIZE)
        struct.pack_into("<IIQQQQQII", b, 0, self.type, ATTR_SIZE, self.config, self.sample_period, self.sample_type, self.read_format, self.flags, self.wakeup_events, 0)
        # sample_max_stack (u16) at 108, after sample_regs_intr (u64 at 96)
        # and aux_watermark (u32 at 104).
        struct.pack_into("<H", b, 108, self.max_stack)
        return bytes(b)


def open_event(attr, pid, cpu=-1, group_fd=-1, flags=0):
    """perf_event_open; returns the fd or raises OSError."""
    buf = ctypes.create_string_buffer(attr.pack(), ATTR_SIZE)
    fd = _libc.syscall(SYS_PERF_EVENT_OPEN, buf, ctypes.c_int(pid), ctypes.c_int(cpu), ctypes.c_int(group_fd), ctypes.c_ulong(flags | 8))  # PERF_FLAG_FD_CLOEXEC
    if fd < 0:
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e))
    return fd


def ioctl(fd, req, arg=0):
    r = _libc.ioctl(ctypes.c_int(fd), ctypes.c_ulong(req), ctypes.c_ulong(arg))
    if r < 0:
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e))
    return r


def event_id(fd):
    v = ctypes.c_uint64(0)
    r = _libc.ioctl(ctypes.c_int(fd), ctypes.c_ulong(IOC_ID), ctypes.byref(v))
    if r < 0:
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e))
    return v.value


def read_value(fd):
    """(value, time_enabled, time_running) of a counting event opened with
    read_format TOTAL_TIME_ENABLED|TOTAL_TIME_RUNNING|ID."""
    data = os.read(fd, 32)
    v, en, run, _id = struct.unpack("<QQQQ", data)
    return v, en, run


def scaled(v, en, run):
    if run == 0:
        return 0
    return v if run >= en else int(v * en / run)


class Ring:
    """The mmap'ed ring buffer of an event: records between tail and head."""

    def __init__(self, fd, pages=64):
        self.page = mmap.PAGESIZE
        self.size = pages * self.page
        self.m = mmap.mmap(fd, (pages + 1) * self.page, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        self.fd = fd

    def records(self):
        """Yield (type, misc, body bytes) of the records written so far and
        release their space."""
        head = struct.unpack_from("<Q", self.m, 1024)[0]
        tail = struct.unpack_from("<Q", self.m, 1032)[0]
        base = self.page
        out = []
        while tail < head:
            off = tail % self.size
            hdr = self._read(base, off, 8)
            typ, misc, size = struct.unpack("<IHH", hdr)
            if size < 8:
                break
            body = self._read(base, (off + 8) % self.size, size - 8)
            out.append((typ, misc, body))
            tail += size
        struct.pack_into("<Q", self.m, 1032, tail)
        return out

    def _read(self, base, off, n):
        if off + n <= self.size:
            return bytes(self.m[base + off:base + off + n])
        first = self.size - off
        return bytes(self.m[base + off:base + self.size]) + bytes(self.m[base:base + n - first])

    def close(self):
        self.m.close()


def tids(pid):
    try:
        return [int(t) for t in os.listdir(f"/proc/{pid}/task")]
    except OSError:
        return []


def thread_name(pid, tid):
    try:
        with open(f"/proc/{pid}/task/{tid}/comm") as f:
            return f.read().strip()
    except OSError:
        return None


def thread_proc_stats(pid, tid):
    """(minflt, majflt, voluntary cs, involuntary cs) from /proc, or None."""
    try:
        with open(f"/proc/{pid}/task/{tid}/stat") as f:
            s = f.read()
        fields = s[s.rindex(")") + 2:].split()
        minflt, majflt = int(fields[7]), int(fields[9])
        vcs = ivcs = 0
        with open(f"/proc/{pid}/task/{tid}/status") as f:
            for line in f:
                if line.startswith("voluntary_ctxt_switches"):
                    vcs = int(line.split()[1])
                elif line.startswith("nonvoluntary_ctxt_switches"):
                    ivcs = int(line.split()[1])
        return minflt, majflt, vcs, ivcs
    except (OSError, ValueError, IndexError):
        return None


def tlb_shootdowns():
    """System-wide TLB shootdown interrupts (the TLB line of
    /proc/interrupts, summed over CPUs), or None."""
    try:
        with open("/proc/interrupts") as f:
            for line in f:
                if line.lstrip().startswith("TLB:"):
                    return sum(int(x) for x in line.split()[1:] if x.isdigit())
    except OSError:
        pass
    return None


def spawn_stopped(argv, env=None):
    """Fork a child that waits until released, then execs argv. Returns
    (pid, release): open events on pid with enable_on_exec, then call
    release()."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(w)
        os.read(r, 1)
        os.close(r)
        try:
            if env is not None:
                os.execvpe(argv[0], argv, env)
            os.execvp(argv[0], argv)
        finally:
            os._exit(127)
    os.close(r)

    def release():
        os.write(w, b"x")
        os.close(w)

    return pid, release
