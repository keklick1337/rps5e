# Changelog

All notable changes to RustPS5Emu (RPS5E) are listed here. Versions follow [Semantic Versioning](https://semver.org/).

## [0.1.3-alpha] - 2026-10-31

### Fixed

- Frame captures (`RPS5E_CAPTURE_FRAMES`) of GTA V replay again instead of hanging the GPU. While a capture recorded, GPU results copied straight into guest memory were later taken for CPU writes and recorded; the replay wrote them over its own GPU results, which differ where GPU work orders its atomics differently, and the mixed ray-tracing tree sent a shader into an endless loop. GPU results now reach guest memory through copies the capture sees while it records.
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
- Depth-compare samples of colour array textures that the shader compares itself (formats without a depth view) take the layer rounded half to even, as the GPU sampler does (a slice of 0.49999997 read layer 1); a not-equal compare passes against NaN.
- `sceAcmBatchJobPriority` is implemented and `sceAcmBatchJobNotification` adds its 16-byte job, clears the notification block and checks its arguments as the console's library does (Silent Hill 2 (Remake) adds priority jobs to its reverb batches).
- `sceNpTrophy2GetRewardIcon` reports that the icon does not exist, like the other trophy icon calls (Terminator 2D: NO FATE).

### Changed

- Less CPU time on the GPU command thread (GTA V's city frame is that thread's work): on Linux, the pages the GPU caches watch for game writes are write-protected with userfaultfd instead of memory protection changes (no split mappings, cheaper faults and protection changes; falls back to memory protection where the kernel lacks it; `RPS5E_UFFD_WATCH`); GPU objects are destroyed on a helper thread (`RPS5E_ASYNC_RETIRE`); GPU buffers copy their results into guest-memory imports that already exist instead of importing their own range each (`RPS5E_CHUNK_HOST_VIEWS`); the image cache no longer walks all images to sum their sizes every 256 draws, and its memory-pressure collection no longer sorts every idle image at each completed GPU submission. GTA V in the city: the command thread's kernel time per frame about 40 % lower; frame rate +5-6 % per change in in-run A/B measurements.
- Less GPU time and fewer GPU waits in GTA V's city: compute dispatches the game issues back to back with nothing between them are no longer forced to wait for each other (the console runs them at the same time; `RPS5E_GUEST_COMPUTE_ORDER`); the small kernels that turn the thread counts of the ray-tracing refit's indirect dispatches into group counts run in one batch ahead of the dispatches (`RPS5E_LAUNCH_BATCH`) or before the dispatch's own barrier (`RPS5E_EARLY_LAUNCH`), and GPU-written values copied into shader resource tables use narrow barriers (`RPS5E_NARROW_TABLE_COPY`). GPU busy time per frame in the alley scene dropped by about a third (about 90 ms to about 60 ms), the frame rate by 5-15 % in the same scene. Optional (`RPS5E_READ_THROUGHS=1`): GPU results larger than 16 KiB are copied back to guest memory only where the CPU reads them (another ~10 % less GPU time in GTA V; off by default because ASTRO BOT then stalls for seconds).
- On processors with a hybrid or asymmetric-cache layout (Ryzen X3D, Intel P/E cores) the GPU command thread gets one physical core of the fast ones to itself: it runs on one CPU there, and no other emulator thread uses that core (`RPS5E_RECORDER_CORE`). GTA V in the city: about 7 % more frames per second, measured A/B in one run.
- Small tiled render targets the game reads back on the CPU are copied to guest memory at the game's next submit after they are drawn, so the CPU read waits for that copy only, not for all GPU work queued by then (`RPS5E_EAGER_TARGET_READBACK`). GTA V reads four 1x1 targets every frame.
- Shorter stalls the first time a game meets new compute shaders: the compile lookahead now also reads past wait packets in the command stream, so more compute shaders start compiling before a dispatch needs them (`RPS5E_LOOKAHEAD_WAITS`). GTA V with empty shader caches: the total time the GPU thread waited for compiles went from 25 s to 15 s and the longest freeze from 18 s to 13 s.
- Graphics pipelines can be linked from separately compiled vertex-side and pixel-side parts where the GPU driver supports it (`VK_EXT_graphics_pipeline_library`, opt-in: `RPS5E_GPL=1`): the two parts compile in parallel and pipelines that differ only in blending, target formats, vertex layout or rasterizer state reuse them.
- The emulator's memcpy/memset into guest memory that the GPU caches watch marks the pages written in one step instead of one page fault per page (about 5-8 % fewer write faults in Silent Hill 2 and GTA V) (`RPS5E_PREMARK_WRITES`).
- Shader `max(0, x)` (a float maximum with a constant +0) is two instructions instead of the general NaN and signed-zero handling; results are the same bit for bit (`RPS5E_ZERO_MAX`).
- Shader f32 add, subtract, multiply and fused multiply-add test their result for NaN once instead of once per input, with the same results: 2-4 % less GPU time in f32-heavy compute shaders (Marvel's Spider-Man 2 capture) (`RPS5E_NAN_ONCE`).
- Less CPU work on the GPU command thread, the thread GTA V's city frame rate waits for: on Linux the emulator's own allocations can use mimalloc (`RPS5E_MIMALLOC=1`, opt-in: the allocator's share of the thread goes from about 10 % to 2 %, but Silent Hill 2 (Remake) stalls after START GAME with it); copies of game-written memory into the GPU upload buffer run on helper threads (`RPS5E_UPLOAD_THREADS`); the fill-kernel check of compute dispatches remembers its answer per shader (`RPS5E_FILL_DETECT_MEMO`); GPU-written ranges are marked as copied back in place (`RPS5E_WRITTEN_WORDS_IN_PLACE`). GTA V city, measured A/B in one run: 2-5 % less command-thread CPU per draw, within the scene-to-scene spread.
- GPU commands are recorded by two threads when that pays: the command interpreter decodes the game's command buffers and prepares the shader stages of the next draws while a recording thread (on the CPU kept for the recorder) records them (`RPS5E_GPU_RECORD_THREAD`, now `auto` by default: on while the recorder is the busy end, off where one thread is faster). Silent Hill's heavy room +10-20 % fps; light scenes stay at their one-thread speed. Fixes that made the two-thread mode pay: every vertex stage prepared ahead was thrown away (it was prepared with another wave size than the recorder uses); the interpreter's prepared stages are remembered like the recorder's (`RPS5E_PREFETCH_MEMO`); a remembered stage whose pages changed is reused when the bytes it was built from are unchanged (`RPS5E_CONTENT_CHECKS`, also for the one-thread memo); occlusion query dumps no longer make the interpreter wait for the recorder (`RPS5E_ASYNC_OCCLUSION`, 5 000 waits a second in Silent Hill); the recorder wakes the interpreter only while it waits for room.
- Fewer memory protection changes on the GPU command thread: re-watching scattered pages of one 4 MiB block takes one `mprotect` over them and the pages between that already have that protection (`RPS5E_PROTECT_BRIDGE`, Linux; GTA V city: 0.4-3 % less CPU per frame). Re-watching pages the game rewrote in a large buffer window was ~75 % of the time of those uploads.
- Fewer GPU waits on the GPU recording thread: a descriptor word whose last GPU write was already copied back to guest memory is read there, instead of copying its page out again and waiting for the GPU, also when the record of that page's copy-backs was dropped (`RPS5E_WORD_LANDED_ALONE`; GTA V's ray-tracing refit waited ~80 times per frame for such words). GTA V city, A/B in one run: slightly faster on average (within the scene-to-scene spread; the frame is bound by the recording thread's CPU time).
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

[0.1.3-alpha]: https://github.com/keklick1337/rps5e/releases/tag/v0.1.3-alpha
[0.1.2-alpha]: https://github.com/keklick1337/rps5e/releases/tag/v0.1.2-alpha
[0.1.1-alpha]: https://github.com/keklick1337/rps5e/releases/tag/v0.1.1-alpha
[0.1.0-alpha.2]: https://github.com/keklick1337/rps5e/releases/tag/0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/keklick1337/rps5e/releases/tag/0.1.0-alpha.1
