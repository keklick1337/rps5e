# Contributing to RustPS5Emu

Thanks for helping! Bug reports, test results on other hardware and code are all welcome.

## Reporting bugs

Use the launcher's **Report issue** action (game menu, or the dialog after a crash): it opens a prefilled issue on
<https://github.com/keklick1337/rps5e/issues> with the game, versions, OS,
GPU and the last log lines. Add what you did, what you expected and what happened; screenshots and the full log (`~/.local/share/rps5e/logs/`) help. One issue per problem, please.

Never attach game files, firmware or keys.

## Building and testing

See [BUILD.md](BUILD.md) for the requirements. Then:

```sh
make build                                     # cargo build --release
cargo test --release -p ps5shader -p ps5gpu -p ps5hle -p ps5vm -p rps5e-gui
make test-gpu                                  # Vulkan device tests (validation layers)
```

- `cargo build --release` must finish **without warnings**.
- On a machine shared with other builds, limit the jobs: `CARGO_BUILD_JOBS=16`.
- A few socket and scheduling tests can fail on a heavily loaded machine; they pass when run alone.
- Tests that need real games find them by title ID under `GAMES_DIR` (set it in `local/games.conf`, copied from `scripts/games.conf.example`, and run `make test`); without it they are skipped.

## Repository layout

| Path | What |
|---|---|
| `crates/ps5elf` | SELF/ELF loading |
| `crates/ps5vm` | guest virtual memory, address range registry, shutdown |
| `crates/ps5cpu` | native execution of guest x86-64 code, signals/exceptions, TLS patching |
| `crates/ps5core` | loading, relocation and import linking |
| `crates/ps5nid` | NID database (SHA1 + salt) |
| `crates/ps5hle` | the system libraries (kernel, libc, AGC GPU driver, audio, pad, save data, network, ...) |
| `crates/ps5shader` | RDNA shader recompiler: decoder, SSA IR, optimisations, interpreter |
| `crates/ps5gpu` | Vulkan back-end: SPIR-V, pipelines, images, buffers, presentation |
| `crates/ps5input` | SDL3 controllers, keyboard and mouse |
| `crates/ps5pkg` | PKG / PFS / PFSC containers |
| `crates/rps5e-cli` | the `rps5e` and `rps5e-cli` binaries |
| `crates/rps5e-gui` | the launcher and the in-game menu (egui) |
| `scripts/` | run, input-record, packaging and profiling tools |
| `progress/` | implementation statistics generator and picture |
| `local/` | **not in git**: your game paths, notes, scratch files, logs |

Everything that belongs to your machine goes into `local/` (it is the only ignored folder besides `target/`): `local/games.conf`, `local/tmp/` for scratch files, `local/recordings/`, logs and notes.

## Code style

- Rust 2021, `cargo fmt` style (lines up to ~150 columns are fine), no warnings, no `unwrap` on guest-controlled data.
- Every source file starts with the project header:

  ```rust
  // RustPS5Emu (RPS5E)
  // Copyright (C) 2026 Vladislav Tislenko aka keklick1337
  // SPDX-License-Identifier: GPL-2.0-only
  ```

  followed by a module doc comment (`//!`) that says what the file is for. Scripts use the same header with `#`.
- Comments explain how the code works and why; no history ("was changed because...") and no notes to yourself.
- Fix root causes. No per-game workarounds, no faked success codes: a system function either behaves like the console's or returns the console's error.
- Every behaviour change comes with a test that fails before it and passes after it (interpreter-checked shader tests, HLE unit tests, device tests).
- Each GPU fast path or optimisation must be switchable off with its own `RPS5E_*` environment variable, so a regression can be bisected at run time.
- Environment variables use the `RPS5E_` prefix.

## Running games while developing

- **Only one emulator at a time.** `rps5e-cli run` holds a lock (`$XDG_RUNTIME_DIR/rps5e-run.lock`) and a second run waits; two games at once can freeze the machine. Scripts that use `timeout` wait for the lock first (`flock "$XDG_RUNTIME_DIR/rps5e-run.lock" true`) and never wrap the run itself in `flock`.
- Get frames from the emulator itself (`--dump-frames`, `RPS5E_DUMP_WINDOW_FRAMES`) rather than desktop screenshots.
- Before claiming a rendering fix, compare frames of the same scene before and after (input recordings make that repeatable; see README).
- Useful tools: `rps5e-cli imports <eboot.bin>`, `PS5_HLE_TRACE=<file>` (HLE call trace), `RPS5E_GPU_PROFILE=1` (GPU profile in the log), `RPS5E_CAPTURE_FRAMES` + `rps5e-cli replay` (GPU frame capture as a benchmark), `scripts/input-test.sh`.

## Branches and versions

The project is in early development. All work, including pull requests, goes straight into `master`, which is always the current development version and can be unstable. Versions are marked with tags (`v0.1.1-alpha`, ...); each tag gets prebuilt Linux and Windows archives on the Releases page. If you want something that was tested as a whole, use a tagged release; if you want the newest fixes, build `master`.

## Pull requests

1. Fork, create a branch from `master`, keep each pull request to one topic, and open the pull request against `master`.
2. Build without warnings, run the tests, and for game-visible changes test the affected games (one at a time).
3. Describe what changed and why, and how you tested it (games, scenes, before/after frames for rendering changes).
4. By contributing you agree that your code is licensed under GPL-2.0-only.

Do not add game files, firmware, keys or code copied from Sony's software.
