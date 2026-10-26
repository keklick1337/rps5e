# Changelog

All notable changes to RustPS5Emu (RPS5E) are listed here. Versions follow [Semantic Versioning](https://semver.org/).

## [0.1.0-alpha.2] - 2026-10-27

### Fixed

- ASTRO BOT: tile-shaped holes, black sky and ghost images left from earlier frames ([#1](https://github.com/keklick1337/rps5e/issues/1)).
- Torn frames in videos played through AvPlayer (ASTRO BOT intro).
- Demon's Souls: a crash with an endless memory fault after a CPU write raced the GPU image cache.

## [0.1.0-alpha.1] - 2026-10-26

First public alpha.

### Emulation

- SELF/ELF loading of decrypted games and modules, relocation and import linking; the game's own modules (libc.prx, middleware) run natively.
- Native execution of guest x86-64 code with patched `fs:` TLS access; AMD-only instructions emulated or patched on Intel CPUs.
- HLE system libraries: kernel (memory, threads, synchronisation, files, events, sockets), libc, AGC GPU driver and command processor, video out, audio (AudioOut, AJM, Ngs2, ATRAC9, MP3), AvPlayer and video decoding (FFmpeg), pad, user and system services, save data, trophies, fonts, IME, SSL, HTTP, PlayGo and more.
- GPU: PM4 command processing on Vulkan 1.3, RDNA shader recompiler to SPIR-V, image cache with aliasing and CPU write tracking, BC decompression, mesh-shader primitive paths, HDR10 output.
- Fake PKG, PFS and PFSC containers: info, install and direct run.
- Windows build (x86_64-pc-windows-gnu) besides Linux.

### Launcher and usability

- Launcher (`rps5e`): game library with folders, packages, covers, search, grid and list views; per-game emulator settings; run logs.
- Controls: keyboard, mouse and SDL3 controllers with press-to-bind, global bindings with per-game input profiles, up to 4 players, DualSense adaptive triggers, light bar, touch pad and motion.
- In-game menu (F1 / PS button): controls with the DualSense picture, graphics, audio and system settings.
- Shader precompilation before the game starts, with progress; portable per-game shader cache export and import (CLI and launcher).
- Input recording and replay (`--record-input`, `--replay-input`, `scripts/input-test.sh`).
- Report issue: a prefilled GitHub issue (game, versions, OS, GPU, log tail) from the game menu and after a crash.
- Performance overlay (F10), frame dumps, GPU frame capture and replay.
- Application icon in the windows and in the Windows executables; Linux desktop entry installed by `make install`.

### Project

- The project is named RustPS5Emu (RPS5E): binaries `rps5e` and `rps5e-cli`, environment variables `RPS5E_*`, data, settings and caches under `rps5e` folders.
- Release packages: `scripts/package.sh` (Linux tarball, Windows zip) with a `dist` build profile without debug info.

[0.1.0-alpha.2]: https://github.com/keklick1337/rps5e/releases/tag/v0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/keklick1337/rps5e/releases/tag/v0.1.0-alpha.1
