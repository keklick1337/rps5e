# Changelog

All notable changes to RustPS5Emu (RPS5E) are listed here. Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- Silent Hill 2 (Remake) no longer hangs the GPU after START GAME. A small compute write (a 4-byte counter in reused transient memory) at an address where a render target once kept its fast-clear metadata was taken for a metadata clear, which overwrote the counter and left a GPU loop running for billions of iterations; only writes that cover a whole metadata slice count as clears now (`RPS5E_WHOLE_METADATA_WRITES`).
- Silent Hill 2 (Remake): the ray-tracing acceleration-structure build after START GAME no longer loops forever on the GPU. Compute shaders whose buffer record count comes from a value computed at run time (here an atomic counter read back with `v_readfirstlane`) were skipped, so the sort before the tree build never ran; they are translated now (`RPS5E_RUNTIME_BUFFER_RECORDS`). The game loads and reaches "PRESS ANY BUTTON TO PLAY".
- Silent Hill 2 (Remake) no longer freezes on the loading screen after START GAME (about one run in six). `sceAudioOut2ContextPush` in blocking mode returned with the context queue still full, so `sceAudioOut2ContextGetQueueLevel` right after it reported no free grain; Wwise renders only the free grains and pushes only rendered ones, so its audio thread stopped, and the garbage collector waited forever for an audio bank to unload while holding the loading screen's lock. A blocking push now returns once the queue has room for the next grain.
- Silent Hill 2 (Remake): the picture after START GAME is no longer black; the opening cutscene and the lookout are drawn. Unreal Engine 5 clears the compression keys of its render targets with a small compute kernel that loads the clear code from a constant buffer; the emulator runs that kernel as a plain memory fill and read the code as out of range (it counted the buffer's records as bytes instead of 16-byte units), so the separate-translucency target was cleared to transparent black instead of (0, 0, 0, 1) and the anti-aliasing pass multiplied the whole scene by 0.
- Silent Hill 2 (Remake) no longer crashes in the audio heap a few minutes into play. The `sceAcm` convolution-reverb shared-input job read its output array with a guessed layout and cleared the memory those words pointed at, which corrupted the Wwise heap; the job now only advances its batch cursor, like the other ACM job builders.

- Shader subroutine calls through a table of code addresses are found when the shader keeps the table's descriptor in vector register lanes between loading it and the call (Silent Hill 2 (Remake)), and calls through user data when the code address was copied before a branch. A callee that returns with `s_swappc_b64` into the null register is linked too, and every table call's dispatch code comes before all callee code. The Silent Hill 2 shader that needed this still does not compile: its tables link more callee code (about 30 000 words per table) than a 16-bit call reaches.
- `sceVideoOutIsOutputSupported` answers 0 for the 119.88 Hz output mode when the output has no 120 Hz, as the console does, instead of an error.

- Flips that a game reserves with `sceVideoOutSubmitEopFlip` and releases with `sceAgcCbReleaseMem` are performed; the release packet our `sceAgcCbReleaseMem` writes was not recognised, so the flip never happened.
- `COPY_DATA` packets that read memory on the prefetch engine read memory instead of GDS, and can load registers; the prefetch engine's atomic result and the clocks can be copied too. A `RELEASE_MEM` to an unsupported destination still raises its interrupt.
- GPU results are no longer lost when a game unmaps part of the memory a GPU buffer covers (`RPS5E_UNMAP_WHOLE_BUFFERS`), or maps the same memory again at the same address (`RPS5E_SAME_BACKING_REMAP`).
- Stencil add and subtract operations use the operation value from `DB_STENCILREFMASK` (a value of 0 leaves the stencil unchanged) (`RPS5E_STENCIL_ADD_OPVAL`).
- Primitive restart honours `MATCH_ALL_BITS`: a restart index wider than the index type never restarts.
- Fast clears of 5_6_5, 1_5_5_5, 5_5_5_1 and 4_4_4_4 render targets are decoded.
- Multisampled render targets and depth buffers take their real size in memory (`RPS5E_MSAA_BLOCKS`); they were taken as larger than they are and seen to overlap the memory after them.

- `sceKernelMunmap` rounds the range to whole pages like `munmap`; an unaligned length left part of a page mapped and charged to the flexible memory budget.
- A texture with a shorter mip chain than a cached image at the same address no longer samples that image's levels, which lie at other bytes (`RPS5E_ALIAS_SAME_MIPS`).
- Indirect indexed draws stop at the end of the index buffer (`INDEX_BUFFER_SIZE` counted from the index base); records with a first index read indices past it.
- Shader loops that wait for EXEC to empty end when the shader enabled lanes the host does not run (partial waves, wave64 programs on 32-lane GPUs) (`RPS5E_PRESENT_EXEC`).
- XOR wave reductions in graphics shaders are computed over the lanes the host subgroup has.
- Small render targets that the CPU reads back (a linear target, or a tiled one of at most 64x64 texels) are copied to guest memory after draws and clears, not only after shader stores (`RPS5E_READBACK_TARGETS`).

### Changed

- Faster image refreshes between views of one array surface that start at different layers (a texture or render target addressed at a layer of an array, as Unreal Engine 5 does with its 4-layer 8K atlases): the shared layers are copied image to image instead of going through the guest memory layout (`RPS5E_LAYER_COPY`): about a quarter more frames per second in Silent Hill 2's gameplay, measured A/B in one run.

## [0.1.2-alpha] - 2026-10-28

### Added

- Games run straight from disk images of game dumps, without extracting them: exFAT images (`.exfat`, also inside an MBR or GPT partition), `.ffpfsc` containers (a PS5 PFS image holding a compressed exFAT image) and UFS2 images (`.ffpkg` dumps made on the console). The game folder is found anywhere inside the image.
- The launcher lists `.exfat` and `.ffpfsc` files with their title, version and icon, and its file pickers accept them; `info-pkg` and `install-pkg` read them too.
- PS5 fake packages ("\x7FFIH" packages made in plain-text mode) run straight from the `.pkg`, without extracting them: the Kraken-compressed game image is decoded on demand, and the package's metadata (title, version, icon, `sce_sys/param.json`) is read from it. `info-pkg` and the launcher show them; PS5 retail packages are still reported as encrypted. Silent Hill 2 (Remake) starts from its 48 GB package and reaches the main menu and the New Game settings at about 1 fps; it can still crash right after the start screen (with `RPS5E_LABELS_AT_EOP=through` it does not), and loading a new game stalls.
- Kraken-compressed PFSC files (the PS5 per-file compression) are read; other PS5 compression formats are reported as unsupported instead of failing to read.
- `imports`, `info` and `load` on a package read its `eboot.bin` instead of loading the whole package file.
- libSceNpAuthAuthorizedAppDialog (open, status, result, close), libSceHttp2 WebSockets (offline: requests fail as without a network), `sceSaveDataPrepareForTransferring`, libSceTextToSpeech2 `Speak`/`Close`/`Terminate`, and the libc thread-destructor calls `_sceLibcInternalThreadDtors` and `_sceLibcInternalForceTlsDestructor`.

### Fixed

- Minecraft no longer crashes right at start: system libraries now provide their data objects (the IPv6 addresses `in6addr_any`, `in6addr_loopback` and the others of libSceNet) with real memory. Minecraft reaches the menus and gameplay.
- libSceNpAuth checks its arguments in the console's order and returns its error codes (invalid argument, invalid size, signed out) without writing the outputs on failure.
- Thread-local destructors run the way the console runs them: newest first, in up to four passes, and only while their module is still loaded.
- Minecraft: the sky and distant terrain are no longer pink or magenta and the screen is no longer split in two shades. A 16-bit colour export into an integer render target now keeps its bits, as the console's colour block stores them.
- Minecraft: the daytime sky is blue instead of black, and the large red and white glow bands are gone. Fragment depth is clamped to the viewport's depth range, as on the console, so a depth clear drawn with a polygon offset stores the far plane exactly.
- Unreal Engine 5 games (Silent Hill 2) no longer stop at start with "Command buffer overflow": a command buffer whose end cursor is 0 has room for any packet, as the console's library computes it.
- With `RPS5E_LABELS_AT_EOP=through`, end-of-pipe labels land in the order the queues wrote them; before, an older value could overwrite a newer one and leave both GPU queues waiting forever.
- Demon's Souls: foliage no longer vanishes for single frames, stray stretched triangles no longer flash across the screen and the tunnel exit no longer blows out. A CPU read of GPU results that the game makes after a command-buffer label now waits for those results to reach memory, as on the console; before, it could see an earlier frame's results.

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

[0.1.2-alpha]: https://github.com/keklick1337/rps5e/releases/tag/v0.1.2-alpha
[0.1.1-alpha]: https://github.com/keklick1337/rps5e/releases/tag/v0.1.1-alpha
[0.1.0-alpha.2]: https://github.com/keklick1337/rps5e/releases/tag/0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/keklick1337/rps5e/releases/tag/0.1.0-alpha.1
