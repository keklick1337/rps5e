# AGENTS.md — RustPS5Emu (RPS5E)

**RustPS5Emu** is a PlayStation 5 emulator (compatibility layer, "Wine for PS5") in Rust. Main platform: Linux x86-64; Windows (x86_64-pc-windows-gnu) is supported too. Guest code runs natively, the system libraries are HLE in Rust, graphics go through Vulkan.

Goal: PS5 games run correctly and at native speed.

Read this file completely before you start. User documentation: `README.md`, `BUILD.md`, `CONTRIBUTING.md`.

## 1. Hard rules

- **One emulator at a time.** Only one `rps5e-cli run` (or `rps5e run`) may run on the machine; two at once can freeze the user's computer.
  - `rps5e-cli run` holds a flock on `$XDG_RUNTIME_DIR/rps5e-run.lock`; a second run waits.
  - Before `timeout N rps5e-cli run …`, wait for the lock first: `flock "$XDG_RUNTIME_DIR/rps5e-run.lock" true` (as `scripts/input-test.sh` does), then check that `pgrep -f "cli run"` is empty. Never wrap the run itself in `flock`: the binary takes the same lock and would deadlock.
  - Do not run copies of old binaries: they may not take the lock.
  - Builds and game runs never overlap: run every cargo command as `flock -s "$XDG_RUNTIME_DIR/rps5e-run.lock" nice -n 19 ionice -c3 cargo …` (shared lock; the emulator takes it exclusively). Parallel builds under a running game distort fps and expose timing bugs that are not there on a quiet machine.
  - Agents that only port or refactor code do not run games: `cargo build` and unit tests only.
- **No sudo.**
- **No hacks.** No per-game workarounds, no faked success codes. Fix the root cause; a system function behaves like the console's or returns the console's error.
- **No desktop screenshots** (spectacle, grim, import, …): they capture the user's private windows. Get frames from the emulator: `--dump-frames`, `RPS5E_DUMP_WINDOW_FRAMES`, and for the launcher the offscreen renderer (`RPS5E_GUI_RENDER_DIR`, see §5).
- **Temporary files only in `local/tmp/`** of the repository (git-ignored). An agent in a worktree uses `<main repo>/local/tmp/<task>/` and removes everything there except images its report refers to. The system `/tmp` is a RAM tmpfs: do not fill it. HLE trace: `PS5_HLE_TRACE=local/tmp/<task>/hle.txt`.
- **Do not claim a rendering bug is fixed without looking at frames** of the same scene before and after (crops in `local/tmp/…`, read them yourself). Counters do not replace frames.
- **Source headers.** Every source file (.rs, .py, .sh, Makefile, build.rs) starts with:
  ```
  // RustPS5Emu (RPS5E)
  // Copyright (C) 2026 Vladislav Tislenko aka keklick1337
  // SPDX-License-Identifier: GPL-2.0-only
  ```
  (`#` for scripts; years `2025-2026` for the core crates ps5elf, ps5vm, ps5cpu, ps5core, ps5nid), then a module doc comment saying what the file is for.
- **Public repository.** Do not name other emulator projects anywhere in tracked files (code, comments, scripts, file names, docs); reference material and notes stay in `local/`. Comments explain how the code works, not its history.
- **Builds:** `CARGO_BUILD_JOBS=16`. `cargo build --release` must have no warnings, also for the Windows target.
- **Video memory is shared.** A run whose log has `binding a dummy image: … ERROR_OUT_OF_DEVICE_MEMORY` is useless for checking pictures (GPU writes were lost). Running out of VRAM by itself is fine (memory spills to system memory).
- **Commits** carry no `Co-Authored-By` or other attribution lines.

## 2. Layout

### Tracked

| Path | What |
|---|---|
| `crates/ps5elf` | SELF/ELF: segments, PT_LOAD, dynamic section |
| `crates/ps5vm` | guest virtual memory, range registry (`ranges`, HLE checks access through it), `shutdown`, `appenv` (app folders) |
| `crates/ps5cpu` | native x86-64 execution: `exec.rs` (guest entry, direct import entry), signals, FS→GS patch, Windows exception handling |
| `crates/ps5nid` | NID database (SHA1 + salt) |
| `crates/ps5hle` | HLE libraries: kernel, libc, memory, pthread, signals, filesystem, audio, AvPlayer, network, services, `agc/` (libSceAgc + PM4 queue) |
| `crates/ps5core` | load → map → relocate → link imports (`linker.rs`), runtime |
| `crates/ps5shader` | shader recompiler: RDNA decoder, CFG/structurizer, SSA IR, optimisations, interpreter (reference for tests), resource planning; test vectors in `tests/vectors/` |
| `crates/ps5gpu` | Vulkan back-end: `shader/` (SPIR-V), `pipeline/`, `images/`, `buffers/`, `memory/` (write tracking), `scheduler/`, presenter, capture/replay |
| `crates/ps5input` | SDL3 controllers, keyboard and mouse → ScePad/SceMouse |
| `crates/ps5pkg` | fake PKG / PFS / PFSC / FFPKG |
| `crates/rps5e-cli` | binaries `rps5e-cli` and `rps5e` (no arguments: the launcher); Windows icon resource in `build.rs` |
| `crates/rps5e-gui` | launcher and in-game menu (egui); issue reports (`report.rs`) |
| `scripts/` | `run-game.sh`, `input-test.sh`, `package.sh`, `frame_stats.py`, `glitch_scan.py`, `perf/` (profiling without root), `games.conf.example` |
| `progress/` | `progress.py` (`make stats`), its data, `progress.svg` shown in README |
| `assets/` | logo, icons (PNG sizes, `.ico`, raw RGBA window icon) |
| `packaging/` | `rps5e.desktop` |

### Local (`local/`, git-ignored, this machine only)

Everything that is not public lives in `local/`:

- `local/games.conf` — `GAMES_DIR` (template `scripts/games.conf.example`), read by `make test` and `progress/progress.py`; the maintainer's copy may hold more per-game variables for local scripts. Game paths never go into tracked code or scripts: tests find a game by title ID under `GAMES_DIR` with `ps5vm::test_games::game_dir("PPSA…")` and skip without it.
- `local/docs/` — working notes (plans, GPU notes, port notes, history); `local/PLAN.md`, `local/PROGRESS.md`, `local/MEMORY.md` if present.
- `local/other_emus/` — reference projects and firmware dumps.
- `local/scripts/` — local tools: `smoke-test.sh` and the game-keyed `input-test.sh` (per-game scenarios), reference coverage, NID audit, ….
- `local/recordings/` — input recordings with save snapshots.
- `local/tmp/` — scratch files, traces, frame dumps; `local/logs/`, `local/data/`, captures, dumps.

Games live outside the repository (`~/Downloads/ps5games/`); never move or copy them.

## 3. Architecture (do not break)

- **Guest entry** only through `enter_guest`: sigsetjmp, return address `HOST_RETURN_SENTINEL`, exit through siglongjmp. Never keep the host rsp in a register across a guest call.
- **Imports**: stub `lea r11,[rip-7]; jmp [entry]` → `ps5cpu_import_entry` (host stack, gregs + FXSAVE frame) → HLE callback → RAX, plus RDX/XMM0 through `ctx.set_return_*`. ~40–75 ns. `RPS5E_IMPORT_TRAP=1` uses the old INT3 path.
- **Linking** (`ps5core/linker.rs`): kernel → always HLE; the libc allocator (when `libc.prx` exports all 7 functions), `sprintf`, `memmove`/`memcmp` → native `libc.prx`; `memset`/`memcpy`/`fputs` → HLE; an import without a real HLE implementation → the module's native export; `register_fallback_fn` → HLE only when no module exports it.
- **NIDs**: every `register_fn(nid, name)` must match SHA1(name + salt). Both `scePthread*` and `pthread_*` sets are registered.
- An unimplemented import returns `0x80020002`; kernel errors are `0x80020000 | errno`.
- **TLS**: every `fs:` access is patched to `gs:`; each host thread has `ARCH_SET_GS` = TCB (Windows: trampolines reading a TEB slot). Static TLS variant II. TCB: +0 self, +8 DTV, +0x10 handle, +0x28 canary, +0x40 errno, +0x60 self.
- **Threads 1:1**; a thread's identity is its pthread structure address.
- **Blocking waits in HLE** only through `signals::wait_cv` / `wait_while` / `sleep` (slices ≤ 10 ms). Never call guest code while holding a lock.
- **DT_INIT** of every PRX is called (DT_INIT_ARRAY is not called separately). eboot loads at 0x800000000, libc.prx at 0x100000000, other modules after the eboot image with a 2 MiB gap.
- **No NULL-page emulation.** A guest SIGSEGV prints diagnostics (module + offset, registers, rbp walk) and exits.
- **Memory** (`ps5hle/memory.rs`): memfd of 13 376 MiB direct + 448 MiB flexible, aliasing. Every mapping is registered in `ps5vm::ranges` and in `ps5gpu::memory` (`note_mapped`). PRT aperture: unmapped pages read zeros.
- **Process exit** (`ps5vm::shutdown`): never `std::process::exit` with live guest threads; use `shutdown::request` / `exit`; a new host thread with resources registers `shutdown::on_shutdown`.
- **Input** works on physical keys (winit `PhysicalKey`); the keyboard layout does not matter.
- **Graphics**: CPU write tracking per page (protection + fault); image cache with aliasing and page merging, write-back only into memory the CPU did not touch; shaders RDNA → SSA → SPIR-V, one wave = one host subgroup; every acceleration has its own `RPS5E_*` switch (`grep -rn RPS5E_ crates/ps5gpu`).
- **Names**: env variables `RPS5E_*`; app folders through `ps5vm::appenv::app_dir`.

## 4. How to work

1. Task → find the console's behaviour (tests on hardware, reference material in `local/other_emus/`, decrypted modules from game dumps) → implement it with tests. Never copy Sony code; port behaviour.
2. Graphics debugging: frame trace (`RPS5E_FRAME_TRACE`, `_MRT`, `_COUNT`, `_RAW`), profile (`RPS5E_GPU_PROFILE=1`: the `[gpu]` log line), shader dumps (`RPS5E_DUMP_SHADERS`, `ps5shader-dis`), the shader interpreter on captured inputs, frame capture (`RPS5E_CAPTURE_FRAMES`, `RPS5E_CAPTURE_AT`, `RPS5E_CAPTURE_COUNT`) and `rps5e-cli replay`. Method: find the **first** pass that produces a wrong result, check its inputs (formats, tiling, aliases, synchronisation), fix the cause, and add a test that fails before and passes after.
3. CPU/HLE debugging: `rps5e-cli imports`, `PS5_HLE_TRACE=…`, `PS5_HLE_TRACE_AFTER=<s>`, coredumpctl.
4. Speed measurements: A/B inside one run (`RPS5E_AB_PERIOD`, `RPS5E_AB_SCOPE`); separate runs vary with machine load.
5. After changes:
   - `cargo build --release` without warnings, and the Windows target: `cargo build --release --target x86_64-pc-windows-gnu` (BUILD.md);
   - tests: `cargo test --release -p ps5shader -p ps5gpu -p ps5hle -p ps5vm -p rps5e-gui` (device tests: `-- --ignored`). Under load a few tests can fail and pass alone: `ioctl_fionread_and_fionbio`, `fstatfs_reports_the_volume`, `kernel_socket_descriptors`, `mutex_hands_off_to_sleepers_in_fifo_order`;
   - run the affected games **one at a time** (input-test replays or smoke scenarios) and look at the frames;
   - commit.
6. Keep notes in `local/docs/` (what was wrong, the root cause, the fix, measurements), not in tracked files. User-visible changes go into `CHANGELOG.md`.

## 5. Commands

```bash
make build | test | test-gpu | gui | install | stats | package
rps5e                                                # launcher; in game F1 / PS opens the menu
RPS5E_GUI_RENDER_DIR=$PWD/local/tmp/<task>/ui cargo test --release -p rps5e-gui render_ -- --ignored  # launcher and menu widgets as PNG without a window
scripts/run-game.sh <game folder> --data-dir D --bind Cross=KEY:Space
OUT=$PWD/local/tmp/<task>/smoke SECONDS_PER_GAME=300 local/scripts/smoke-test.sh silent-hill-play   # maintainer's machine only
scripts/input-test.sh record|replay <game folder> <name>    # input recordings (local/recordings)
target/release/rps5e-cli run "<game folder>" --no-input --pad-script "25:cross,35:cross" \
    --data-dir local/tmp/<task>/data --config-dir local/tmp/<task>/cfg --dump-frames local/tmp/<task>/frames --dump-every 60
target/release/rps5e-cli imports "<eboot.bin>"       # HLE / LLE@addr / STUB
target/release/rps5e-cli info-pkg "<file.pkg>"
target/release/rps5e-cli precompile "<game folder>"|<title id> [--clear]   # build the game's shader list without running it
target/release/rps5e-cli shader-cache export|import|info …
target/release/rps5e-cli self2elf "<module.prx>" local/tmp/<task>/m.elf  # fake SELF → ELF for disassemblers
RPS5E_GPU_PROFILE=1 RUST_LOG=info …                  # GPU profile every 5 s
RPS5E_AUDIO=sdl|cpal|null, RPS5E_AUDIO_STATS=1       # audio
RPS5E_COMPILE_WAIT_MS=<ms>|exact                     # per-frame wait for compiling pipelines (run: 3 ms; tests and replays: exact)
RPS5E_DUMP_WINDOW_FRAMES=local/tmp/<task>/win RPS5E_DUMP_WINDOW_EVERY=<n> …  # window frames as shown (menus, overlays)
target/release/ps5shader-dis --bench <dump dir>      # shader compile time per phase
RPS5E_CAPTURE_FRAMES=local/tmp/<task>/cap RPS5E_CAPTURE_AT=<s> RPS5E_CAPTURE_COUNT=<n> …  # capture n frames of GPU work
target/release/rps5e-cli replay local/tmp/<task>/cap/capture-*.ps5cap --loops 5 --profile [--no-present] [--dump-frames DIR]
scripts/package.sh linux|windows                     # release archives in dist/
```

Smoke scenarios (`local/scripts/smoke-test.sh`, maintainer's machine only): `silent-hill`, `silent-hill-play`, `terminator-2d`, `astro-bot`, `astro-bot-play`, `demons-souls`, `spider-man-2`.

## License

GPL-2.0-only (`LICENSE`).
