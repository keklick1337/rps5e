# scripts/perf — profiling without root

`perf` is not installed and `sudo` is forbidden; `perf_event_paranoid` is 2, which still allows user-space events on our own processes. These tools call `perf_event_open(2)` from Python (`perfev.py`, ctypes, x86-64 Linux).

Never point them at a game outside the one-emulator rule (AGENTS.md): a game run under them is still a `rps5e-cli run`. Benchmarks and replays of captures are the intended targets. Output files go to `local/tmp/`.

## hwcount.py — per-thread hardware counters

```bash
scripts/perf/hwcount.py [-g] [--per-kinstr] [-e EVENTS] [-r NAME=0xCFG] -- <command> [args]
scripts/perf/hwcount.py -p <pid> [-d <seconds>]
```

- Columns: task-clock, cycles, instructions, IPC, cache misses, dTLB read misses, branch misses, page faults (counter), minor faults and voluntary / involuntary context switches (from `/proc`, sampled while the thread lives), `mux`; the last line has the system-wide TLB-shootdown delta (`/proc/interrupts`, includes other processes).
- `-g` sums threads whose names differ only in digits (`gpu-slice-*`).
- Launch mode: one inherited event per CPU on the child before it execs. Every thread's counts arrive when it exits, so short threads are complete. The hardware events form one group, so their ratios are exact. Values are not scaled: `mux` below 1 means the group was off the PMU for that share of the thread's run time (other PMU users, e.g. a KVM guest), so the counts are low while the ratios hold.
- Attach mode opens events per thread and rescans every 0.1 s.
- `-e` picks from `cycles,instructions,cache-misses,dtlb-misses,branch-misses,page-faults,task-clock,cache-refs,llc-misses,l1d-misses`. AMD has no generic LLC event; `-r` adds raw PMU events.

## uprof.py — sampling profiler with call chains

```bash
scripts/perf/uprof.py record -o local/tmp/x/prof.json.gz [-F 2000] -- <command> [args]
scripts/perf/uprof.py record -o local/tmp/x/prof.json.gz -p <pid> -d 10
scripts/perf/uprof.py report local/tmp/x/prof.json.gz [--thread RE] [--by-thread -g] [--inclusive] [--tree] [--callers RE]
scripts/perf/uprof.py annotate local/tmp/x/prof.json.gz 'SYMBOL_RE'      # hottest addresses with source lines and inlined frames
scripts/perf/uprof.py diff a.json.gz b.json.gz --thread RE [--thread-b RE] [--scale-a N --scale-b M]
scripts/perf/uprof.py run [record and report options] -- <command>  # into local/tmp/uprof/
```

- The kernel samples cycles (cpu-clock as a fallback) in user mode and walks the user stack through frame pointers. Whole chains need a frame-pointer build:

  ```bash
  RUSTFLAGS=-Cforce-frame-pointers=yes CARGO_TARGET_DIR=$PWD/local/tmp/target-fp cargo test --release -p ps5gpu --no-run
  ```

  The prebuilt std, libc and the GPU driver have no frame pointers, so chains through them stop there. Flat profiles and `annotate` need no rebuild.
- Mappings come from the kernel's MMAP2 records plus `/proc/<pid>/maps` snapshots taken every 0.25 s; empty snapshots of an exited process are dropped. Symbols come from each ELF's `.symtab`/`.dynsym`, demangled with `c++filt`, hashes removed. `annotate` uses `addr2line -i`; the release profile keeps debug info.
- `diff` compares self samples per symbol of two recordings of the same work (for example the slice threads at K=1 and K=4), largest growth first.

## Workloads that run without a game

- `RPS5E_SLICE_BENCH=1 cargo test -p ps5gpu --release -- --ignored slice_record_bench --nocapture`: K slice threads record synthetic UE4-shaped draws (`RPS5E_SLICE_BENCH_K`, `_MATERIALS`, `_PROGRAMS`, `_CONST`, `_WRITES`, `_AB`, `_SUMMARY`; see the file).
- `RPS5E_BINDLESS_BENCH_KEYS=8192 cargo test -p ps5gpu --release -- --ignored bindless_large_table_first_bind_time --nocapture`.
- `RPS5E_LOCK_BENCH=1 cargo test -p ps5hle --release -- --ignored host_lock_contention_bench --nocapture`.
- `RPS5E_DRAW_BENCH_FRAMES=20 cargo test -p ps5gpu --release -- --ignored draw_path_bench --nocapture`.

To run one under a tool, give the test binary as the command, for example:

```bash
RPS5E_SLICE_BENCH=1 RPS5E_SLICE_BENCH_K=4 scripts/perf/hwcount.py -g -- target/release/deps/ps5gpu-<hash> --ignored slice_record_bench --nocapture
```
