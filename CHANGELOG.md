# Changelog

All notable changes to RustPS5Emu (RPS5E) are listed here. Versions follow [Semantic Versioning](https://semver.org/).

## [0.1.1-alpha] - 2026-10-28

### Added

- Display resolution setting (720p, 1080p or 2160p) in the launcher, the "Run with..." dialog, `--display-resolution` and `RPS5E_DISPLAY_RESOLUTION`: games that ask the console for the TV's resolution are told the chosen one, as a PS5 on that TV would answer; the default stays 2160p.
- PlayGo: `scePlayGoRequestOptionalChunk`, `scePlayGoGetRequiredDisc`, `scePlayGoRequestNextChunk`, `scePlayGoSnapshot` and `scePlayGoSetInstallSpeed2` are implemented. The library now tracks its initialized and open-handle state (`NOT_INITIALIZED`, `ALREADY_INITIALIZED`, `BAD_HANDLE` after close), remembers the install speed that was set, and answers a chunk-id count query made without a list buffer.

### Performance

- GTA V in the city: the longest stalls are much shorter (about 2.5 s instead of 10 s) and driving stutters far less. The frame rate in the city is still low (about 7 fps now that the ray-traced reflections are drawn) and far from playable.
- Wave64 compute shaders on 32-wide GPUs (NVIDIA): lane stacks and register spills stay in registers, ballots share one exchange, and programs from 16 VGPRs run split; the ray traversal of GTA V takes about a third of its former GPU time.
- A CPU read of a descriptor the GPU wrote waits only for the copy that brought that word to memory, not for all queued GPU work.
- Texture uploads that do not fit the full staging ring read guest memory directly (the stall when GTA V enters the city: about 4 s → 1.5 s).
- Ray-tracing shaders read BVH nodes with vector loads: the ray traversal of GTA V is about 8 % faster.

### Fixed

- GTA V: car paint, mirrors and wheel rims no longer look pink or magenta, and the square patches in car reflections are gone. The pass that shades ray-traced reflections never ran, so its magenta placeholders stayed in the picture.
- GTA V: no more grey dots, dotted trails and smears on signs, cars and characters. A render target the game clears with a "single colour" DCC clear was never cleared, so old depth values piled up and broke depth of field.
- Windows: shader and pipeline caches are kept in `%LOCALAPPDATA%\rps5e`; before, they needed `HOME` and were missing without it.
- `PS5_HLE_TRACE` with a relative path writes to that path instead of a fixed file in `/tmp`.
- Default keyboard layout: the left and right halves of the touch pad are on Z and X (they had no key).
- `--refresh-rate` takes the same values the emulator uses (whole Hz, 30 to 1000), and `--language` accepts language 30 (Ukrainian) like `RPS5E_SYSTEM_LANGUAGE`.

### Removed

- `RPS5E_PLAYGO_HACK`: games without PlayGo data get the console's answer (PlayGo not supported) instead of a made-up chunk list.

### Documentation

- `docs/COMMANDS.md`: every program, command and option; `docs/ENVIRONMENT.md`: every environment variable and what it does.

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

[0.1.1-alpha]: https://github.com/keklick1337/rps5e/releases/tag/v0.1.1-alpha
[0.1.0-alpha.2]: https://github.com/keklick1337/rps5e/releases/tag/0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/keklick1337/rps5e/releases/tag/0.1.0-alpha.1
