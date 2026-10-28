# Environment variables

RustPS5Emu reads its settings from the command line, from the launcher's settings and from environment variables. Every `rps5e-cli run` option and every launcher setting ends up as an `RPS5E_*` variable for the game process, so anything the options offer can also be set in the environment. A variable set in the environment applies to everything started from that shell, including games started from the launcher (they inherit it).

```sh
RPS5E_GPU_PROFILE=1 RUST_LOG=info rps5e-cli run ~/games/PPSA10112
RPS5E_DUMP_FRAMES=frames RPS5E_DUMP_EVERY=120 rps5e-cli run ~/games/PPSA10112   # same as --dump-frames frames --dump-every 120
```

## How to read the tables

- **For**: `user` marks settings a player may want to change; `dev` marks switches for developers and bug hunting. Changing a `dev` switch can break rendering, slow a game down or make it crash.
- **Switches that are on by default** are turned off with `0` (many also accept `off`; a few `false` or `no`, as listed). Other values leave them on.
- **Opt-in switches** are turned on with `1` (sometimes also `true`, `yes`, `on`, or "any value", as listed).
- **`RPS5E_SAFE=1`** turns most GPU fast paths off at once; rows that say "also off with `RPS5E_SAFE=1`" follow it. `RPS5E_FAST_PATHS=0` turns the per-draw fast paths off (also from the in-game menu).
- **A/B measurement** (`RPS5E_AB_PERIOD`, `RPS5E_AB_SCOPE`, `RPS5E_AB_ONLY`) switches groups of fast paths off and on in alternating periods of one run, so their speed can be compared on the same scene; rows that mention "A/B slow periods" belong to such a group.
- Variables are read once at start unless the description says otherwise.
- Switches described as "part of the shader cache key" change the translated shaders: changing them makes the emulator translate shaders again instead of using the cached ones.

## Command-line options and their variables

| `rps5e-cli run` option | Variable |
|---|---|
| `--data-dir DIR` | `RPS5E_DATA_DIR` |
| `--config-dir DIR` | `RPS5E_CONFIG_DIR` |
| `--input-config FILE` | `RPS5E_INPUT_CONFIG` |
| `--input-mode`, `--bind`, `--controller-color` | `RPS5E_INPUT_OVERRIDE` (TOML) |
| `--no-input` | `RPS5E_NO_HOST_INPUT=1` |
| `--pad-script SCRIPT` | `RPS5E_PAD_SCRIPT` |
| `--record-input FILE` | `RPS5E_INPUT_RECORD` |
| `--replay-input FILE` | `RPS5E_INPUT_REPLAY` (and `RPS5E_NO_HOST_INPUT=1`) |
| `--fullscreen` | `RPS5E_FULLSCREEN=1` |
| `--hdr` | `RPS5E_HDR=1` |
| `--display-resolution RES` | `RPS5E_DISPLAY_RESOLUTION` |
| `--perf-overlay` | `RPS5E_PERF_OVERLAY=1` |
| `--present-mode MODE` | `RPS5E_PRESENT_MODE` |
| `--gpu DEVICE` | `RPS5E_GPU_DEVICE` |
| `--headless` | `RPS5E_HEADLESS=1` |
| `--language ID` | `RPS5E_SYSTEM_LANGUAGE` |
| `--mute` | `RPS5E_AUDIO=null` |
| `--skip-notice-screen` | `RPS5E_SKIP_NOTICE_SCREEN=1` |
| `--refresh-rate HZ` | `RPS5E_GUEST_REFRESH_RATE` |
| `--game-patch FILE` | `RPS5E_GAME_PATCH` |
| `--no-prewarm-wait` | `RPS5E_PREWARM_WAIT=0` |
| `--compile-wait MS\|exact` | `RPS5E_COMPILE_WAIT_MS` (`run` sets 3 when neither is given) |
| `--dump-frames DIR` | `RPS5E_DUMP_FRAMES` |
| `--dump-every N` | `RPS5E_DUMP_EVERY` |

The launcher's settings map the same way (plus `RPS5E_FPS_COUNTER`, `RPS5E_SCALE_MODE`, `RPS5E_SCALE_FILTER`, `RPS5E_FAST_PATHS`, `RPS5E_VOLUME`, `RPS5E_MENU_PAUSE`, `RPS5E_SAVEDATA_DIR`, `RPS5E_TEMP0_DIR`, `RPS5E_DOWNLOAD0_DIR`, `RPS5E_SYSTEM_FONTS`, `RPS5E_NET_REDIRECT`, `RPS5E_PIPELINE_CACHE` and `RUST_LOG`); its "Run with..." dialog adds the debugging ones (GPU profile, HLE trace, frame dump, compile wait) and can override the display resolution for one run.

## General and run

Settings of a game run. Most of them also have a `rps5e-cli run` option or a launcher setting, named in the description.

| Variable | Values and default | What it does | For |
|---|---|---|---|
| `RPS5E_DISPLAY_RESOLUTION` | `720p`, `1080p`, `2160p` (or `4k`), any case; default `2160p` (an invalid value is reported in the log and gives `2160p`); also set by the CLI and launcher | The resolution of the emulated TV. A game whose `param.json` asks for resolution detection (`attribute3` bit 2) is told this resolution by `sceVideoOutGetOutputStatus` and `sceVideoOutGetResolutionStatus` and usually sizes its display buffers to it; other games are always told about a 2160p display, as on the console. Read once at start. Same as `--display-resolution`. | user |
| `RPS5E_FPS_COUNTER` | any non-empty value except `0`: on; default off; also set from the launcher and in-game menu | Shows a small presented-frames-per-second counter in the top-left corner of the game window. | user |
| `RPS5E_FULLSCREEN` | any non-empty value except `0`: on; default off; set by the CLI fullscreen option and the launcher | Opens the game window fullscreen. | user |
| `RPS5E_GAME_PATCH` | file path (mods JSON); unset: none | A cheat/patch file applied to the game's code at load. Same as `--game-patch`; the launcher sets it when cheats are on. | user |
| `RPS5E_GPU_DEVICE` | device index or case-insensitive name substring; default: best device found; also set by the CLI and launcher | Picks which Vulkan GPU to use (only Vulkan 1.3 devices are considered). | user |
| `RPS5E_GUEST_REFRESH_RATE` | whole number of Hz, 30 to 1000; default 60 (other values: 60) | The display refresh rate the game sees and the rate at which its flips are paced; 60 and 120 run at 59.94 / 119.88 Hz like the console. Same as `--refresh-rate`. | user |
| `RPS5E_HEADLESS` | any non-empty value except `0`: on; default off (automatic when no display or Vulkan device is available) | Runs with no game window; frames are processed but not shown. | user |
| `RPS5E_IGNORE_CPU_FEATURES` | `1`: on; default off | Starts the game even when the host CPU lacks a feature the PS5 CPU has (x86-64-v3: AVX2, FMA, BMI, ...); the game will likely crash when it uses one. | user |
| `RPS5E_MENU_PAUSE` | `1`: on; default off | Pauses the game while the in-game menu (F1) is open. A launcher setting. | user |
| `RPS5E_NET_REDIRECT` | IPv4/IPv6 address; unset: off | Network traffic is limited to loopback by default; with this set every outgoing connection is redirected to this address (for example a local server). Also a launcher setting. | user |
| `RPS5E_PERF_OVERLAY` | off by default; any non-empty value other than `0` turns it on; same as `--perf-overlay` | Shows the performance overlay (FPS, frame time graph, CPU/RAM/GPU use, emulator counters) from the start; F10 toggles it during play. | user |
| `RPS5E_PIPELINE_CACHE` | default `$XDG_CACHE_HOME/rps5e/pipelines/<title id>.bin` (`~/.cache/...`; Windows `%LOCALAPPDATA%\rps5e\pipelines`); a file path overrides it; `0` or `off` disables it | Location of the on-disk Vulkan pipeline cache for the game, loaded at start and saved periodically. Disabled: pipelines are built from scratch every run. Also used by `shader-cache export/import`. | user |
| `RPS5E_PRESENT_MODE` | `fifo`, `mailbox` or `immediate`; default `mailbox`; same as `--present-mode`; changeable in the in-game menu | How frames are presented to the window: `fifo` is vsync, `mailbox` shows the newest frame without tearing, `immediate` presents at once (may tear). | user |
| `RPS5E_PREWARM_WAIT` | on by default; `0` or `off` turns it off; same as `--no-prewarm-wait` | At start the game waits until every shader pipeline in its saved prewarm list is built (with a progress overlay). Off: the game starts at once and the list is built in the background (more stutter early on). | user |
| `RPS5E_SCALE_FILTER` | `linear` or `nearest`; default `linear`; changeable in the in-game menu | Filter used when scaling the game image into the window: smooth (`linear`) or sharp pixels (`nearest`). | user |
| `RPS5E_SCALE_MODE` | `fit`, `stretch` or `integer`; default `fit`; changeable in the in-game menu | How the game image fills the window: `fit` keeps the aspect ratio with black bars, `stretch` fills the window ignoring aspect, `integer` scales by whole multiples. | user |
| `RPS5E_SHADER_CACHE` | default `$XDG_CACHE_HOME/rps5e/` (`~/.cache/rps5e/`; Windows `%LOCALAPPDATA%\rps5e`); a directory path overrides it; `0` or `off` disables it; also an in-game menu switch | Location of the on-disk cache of compiled shaders, so shaders seen before are not translated again. Disabled: every shader is translated again each run (more stutter). | user |
| `RPS5E_SKIP_NOTICE_SCREEN` | `1`, `true`, `yes` or `on`; default off | Turns the console's "skip notice screen" system setting on; games that honour it skip their start-up notices. Same as `--skip-notice-screen`. | user |
| `RPS5E_SYSTEM_LANGUAGE` | console language number 0 to 30; default 1 (English, US) | The console language games see. Same as `--language` and the launcher's language setting. | user |

## Folders

Where the emulator keeps data and which host folders the game's mounts point to. `<data dir>` is the folder of `RPS5E_DATA_DIR`.

| Variable | Values and default | What it does | For |
|---|---|---|---|
| `RPS5E_CONFIG_DIR` | directory; default `$XDG_CONFIG_HOME/rps5e` (`~/.config/rps5e`), Windows `%APPDATA%\rps5e` | Emulator and launcher settings: `rps5e-input.toml`, `profiles/<TITLE_ID>.toml`, launcher settings. Same as `--config-dir`. | user |
| `RPS5E_DATA_DIR` | directory; default `$XDG_DATA_HOME/rps5e` (`~/.local/share/rps5e`), Windows `%LOCALAPPDATA%\rps5e` | Base folder for writable per-title data (one subfolder per title ID), save data and installed packages. Same as `--data-dir`. | user |
| `RPS5E_DOWNLOAD0_DIR` | directory; default `<data dir>/<TITLE_ID>/download0` | Host folder behind the game's `/download0` mount (downloaded content). Also a launcher setting. | user |
| `RPS5E_SAVEDATA_DIR` | directory; default `<data dir>/savedata` | Root folder of all save data (`<root>/<TITLE_ID>/<save name>`). Also a launcher setting. | user |
| `RPS5E_SYSTEM_FONTS` | directory; unset: host font folders only | Folder with the console's system fonts (or substitutes); searched before `/usr/share/fonts`, `/usr/local/share/fonts`, `~/.local/share/fonts` and `~/.fonts`. Also a launcher setting. | user |
| `RPS5E_TEMP0_DIR` | directory; default `<data dir>/<TITLE_ID>/temp0`, emptied at every start | Host folder behind the game's `/temp0` mount; a folder set here is not emptied. Also used for the free space the game is told about. Also a launcher setting. | user |
| `RPS5E_APP0_DIR` | host directory; default: the folder holding `eboot.bin` | The host folder the game sees as `/app0` (its own files and `sce_sys/param.json`). | dev |
| `RPS5E_DATA0_DIR` | directory; default `<data dir>/<TITLE_ID>/data` | Host folder behind the game's `/data` mount. | dev |
| `RPS5E_DEVLOG_APP_DIR` | directory; default `<data dir>/<TITLE_ID>/devlog/app` | Host folder behind the game's `/devlog/app` mount. | dev |
| `RPS5E_HOSTAPP_DIR` | directory; default `<data dir>/<TITLE_ID>/hostapp` | Host folder behind the game's `/hostapp` mount. | dev |
| `RPS5E_SYSTEM_DIR` | directory; unset: none | A decrypted dump of the console's `/system` folder, mounted as `/system` so games can load system modules from it. | dev |
| `RPS5E_SYSTEM_EX_DIR` | directory; unset: none | A decrypted dump of the console's `/system_ex` folder, mounted as `/system_ex`. | dev |
| `RPS5E_WRITABLE_APP0` | `1`: on; default off | Lets the game write into its own `/app0` folder (normally read-only, as on the console). | dev |

## Input

See also COMMANDS.md for the `--bind`, `--pad-script` and recording syntax, and `rps5e-input.example.toml` for the settings file.

| Variable | Values and default | What it does | For |
|---|---|---|---|
| `RPS5E_DEFAULT_PROFILE` | text; default `rps5e` | Text the emulator types into an empty on-screen keyboard (IME) dialog when the game asks for input, e.g. a profile or player name. | user |
| `RPS5E_INPUT_CONFIG` | file path; default: `./rps5e-input.toml`, then `<config dir>/rps5e-input.toml` | A specific input settings file to use (it must exist). Same as `--input-config`. | user |
| `RPS5E_INPUT_RECORD` | file path; unset: off | Records every change of player 1's pad (buttons, sticks, triggers) with its time to the file. Same as `--record-input`. | user |
| `RPS5E_INPUT_REPLAY` | file path; unset: off | Replays a recording made with `RPS5E_INPUT_RECORD` on player 1's pad at the recorded times. `--replay-input` also turns host input off. | user |
| `RPS5E_NO_HOST_INPUT` | any non-empty value except `0`; default off | Ignores host keyboard, mouse and controllers; the game sees a neutral pad (scripts and replays still work). Same as `--no-input`. | user |
| `RPS5E_PAD_SCRIPT` | `<seconds>:<input>[:<hold ms>]`, comma-separated | Scripted pad presses for unattended runs (see the `--pad-script` syntax in COMMANDS.md). Same as `--pad-script`. | user |
| `PS5_AUTO_CROSS` | comma-separated seconds, e.g. `40,52,64`; unset: off | Holds Cross for 0.4 s at each listed time (seconds since the game first read the pad). An older, simpler form of `RPS5E_PAD_SCRIPT`. | dev |

## Audio

| Variable | Values and default | What it does | For |
|---|---|---|---|
| `RPS5E_AUDIO` | `sdl` (or `sdl3`), `cpal` (or `alsa`), `null` (or `none`); default: SDL3, then cpal | Chooses the audio output back-end; `null` opens no device (silence; the game still runs at its normal pace). `--mute` and the launcher's mute option set `null`. | user |
| `RPS5E_VOLUME` | percent, 0 to 100; default 100 | Master volume of the game's sound at start (changeable in the in-game menu). A launcher setting. | user |
| `RPS5E_AUDIO_STATS` | any value except `0`; default off | Every 5 s each output stream logs the loudness (RMS) of what the game queued and what the device played, plus failed audio-port attribute calls. | dev |
| `RPS5E_FMOD_OUTPUT_RESET` | `1`: on; default off | Replaces one FMOD output-initialisation call so that some Unity games using FMOD Studio set up their audio output fully. Tied to one FMOD build, so it is opt-in. | dev |

## Logging and diagnostics

Log output goes through `RUST_LOG` (see [Standard variables](#standard-variables)); these add specific reports to it or write separate files.

| Variable | Values and default | What it does | For |
|---|---|---|---|
| `RPS5E_PERF_NVML` | on by default; only `0` turns it off | The performance overlay reads NVIDIA GPU usage through the NVML library. Off: NVML is not loaded; AMD sysfs or Vulkan memory budget data is used where available. | user |
| `PS5_HLE_TRACE` | a file path (relative to the current directory), or `1` for `hle_trace.txt` in the current directory; unset: off | Writes one line per system-library call (`nid rdi rsi rdx rcx rax caller tid`) to the file. Very large and slow. | dev |
| `PS5_HLE_TRACE_AFTER` | seconds; default 0 | With `PS5_HLE_TRACE`: start recording only this many seconds after the first call, to skip start-up. The launcher's "Run with..." dialog sets it. | dev |
| `RPS5E_BARRIER_SITES` | `1` turns it on; default off | With the GPU profile (`RPS5E_GPU_PROFILE=1`), counts full GPU barriers by source location in the periodic summary, and the GPU waits that blocked a thread by source location and by the shader being prepared (`gpu waits`). | dev |
| `RPS5E_DRAW_SKIP_LOG` | `1` turns it on; default off | Logs one line per presented frame in which draws were skipped, with counts per reason (still compiling, nothing to draw, shader unsupported, host error) and the first few shaders involved. | dev |
| `RPS5E_FRAME_TIMES` | file path; unset: off | Writes the time of every flip as `<ms since start> <ms since previous flip>` per line (rewritten periodically and at exit), so freezes and long frames show up exactly. `scripts/input-test.sh` uses it. | dev |
| `RPS5E_GPU_PROFILE` | `1`: on; `2`: also detailed sub-phase timings; default off; `--profile` sets `1` | Prints a `[gpu]` profile line every few seconds with GPU timings, back-end phases and top shaders. | dev |
| `RPS5E_GPU_TOP` | integer; default 10 | Number of shader programs listed in the GPU profile's top-shaders list (`RPS5E_GPU_PROFILE`). | dev |
| `RPS5E_GPU_TRACE` | integer N; default 0 | Logs details (colour/depth targets, formats, metadata, clears) of the first N draws the back-end executes. | dev |
| `RPS5E_GPU_TRACE_CS` | hex compute-shader address; default none | For every dispatch of that compute shader, logs its user data and the first 256 bytes of its resource table. | dev |
| `RPS5E_HLE_PROFILE` | seconds (> 0); unset: off | Counts calls and time per system-library function and logs the top 40 by time every this many seconds. | dev |
| `RPS5E_PIPELINE_STATS` | opt-in: `1` turns it on (needs VK_KHR_pipeline_executable_properties); default off | Logs the driver's statistics (registers, instruction counts) of every compute pipeline it creates, with its creation time. | dev |
| `RPS5E_PM4_DUMP` | number; default 0 | Logs the non-zero GPU register state of the first N draws. | dev |
| `RPS5E_PM4_PROFILE` | `1`: on; default off | Measures the time spent interpreting GPU command packets and the part spent in draw/dispatch calls (shown in the GPU profile line). | dev |
| `RPS5E_PM4_TABLES` | any value | Logs each GPU register table once when it is first applied. | dev |
| `RPS5E_PROFILE_LOCAL` | on by default; only `0` turns it off | GPU profiler time spans are gathered per thread and added to the shared counters in batches. Off: every span goes straight to the shared counters (more contention between threads). | dev |
| `RPS5E_RING_DEBUG` | off by default; set to any value to turn it on | Counts why buffer bindings did not become ring bindings (see `RPS5E_RING_BINDING`). Temporary diagnostic. | dev |
| `RPS5E_SHADER_INFO_VERBOSE` | `1` turns it on; off by default | `rps5e-cli shader-cache info` also prints the differing state values of the pipelines in a game's shader list, not only how many differ. | dev |
| `RPS5E_TRACE_DRAW` | hex shader address (optional `0x`); unset by default | Logs every draw whose vertex or pixel program is at that address, with its packet, count, index buffer and targets (`[trace-draw]` at info level). | dev |

## Graphics

How the GPU commands of the game are carried out. Unless marked `user`, these exist to narrow down rendering bugs: change one only to test whether a glitch depends on it.

| Variable | Values and default | What it does | For |
|---|---|---|---|
| `RPS5E_ACQUIRE_TIMEOUT_MS` | milliseconds; default 100 | How long the presenter waits for a window image before it drops the frame (prevents a hidden or minimized window from stalling the game). | user |
| `RPS5E_HDR` | any value except empty, `0`, `off`, `false`: on; default off; also set by the CLI and launcher | HDR10 output for games that present HDR (PQ) frames, when the display offers an HDR10 surface; SDR frames and overlays are shown converted. Read once at start. | user |
| `RPS5E_OCCLUSION_QUERIES` | opt-in: `1` turns it on; default off; also an in-game menu switch | Counts the game's occlusion queries on the GPU with real sample counts. Off (or without device support): every query reports the object as visible. | user |
| `RPS5E_AA_MASK` | on by default; `0` turns it off | Applies the game's per-quad anti-aliasing sample mask to multisampled draws; `0` keeps every sample. | dev |
| `RPS5E_ALIAS_MERGE` | on by default; `0`/`off` turns it off (also off with `RPS5E_SAFE=1`) | Lets an image that shares memory with others refill them from its GPU-modified contents even when the CPU wrote some of its pages; off, those aliases keep older contents. Also in the in-game menu. | dev |
| `RPS5E_ALPHA_TO_COVERAGE` | on by default; `0` turns it off | Emulates alpha-to-coverage (the first color output's alpha decides which samples are covered); `0` ignores it. | dev |
| `RPS5E_ARRAY_WINDOW` | on by default; `0` turns it off | A sampled view of a texture array covers only the slices the game's descriptor selects; `0` builds the view from the whole array starting at slice 0. | dev |
| `RPS5E_AUTO_INDEX_OFFSET` | on by default; `0` turns it off | Non-indexed draws start at the vertex offset register the game sets; `0` always starts at vertex 0. | dev |
| `RPS5E_CLOCK_RECAL` | on by default; `0` turns it off | Re-measures the offset between the GPU clock and the system clock once a second; `0` keeps the value measured at start, which drifts by about 26 ms per hour and can make frame limiters wait too long. | dev |
| `RPS5E_COLOR_VIEW_CLAMP` | on by default; `0` turns it off | Clamps a render target view's last array slice to the array's real depth; `0` takes the slice range as the game gives it. | dev |
| `RPS5E_DB_DEPTH_COPY` | on by default; `0` turns it off | Recognises a depth-to-color copy programmed through the depth block and performs it as a copy; `0` runs it as the draw it is programmed as. | dev |
| `RPS5E_DCC_EVERY_LAYER` | on by default; `0`/`off` turns it off (also off with `RPS5E_SAFE=1`) | For layered render targets whose compression metadata a compute shader overwrote, checks every bound layer for a fast-clear value; `0` checks only the first layer. | dev |
| `RPS5E_DEPTH_UNRESTRICTED` | on by default when the device supports VK_EXT_depth_range_unrestricted; `0` turns it off | Viewport depth ranges outside [0, 1] are used as the game sets them; `0` clamps them. | dev |
| `RPS5E_DISPATCH_SPLIT` | on by default; `0` turns it off | A compute dispatch larger than the device's group-count limit is split into several dispatches; `0` skips such dispatches. | dev |
| `RPS5E_DRAW_INDEX_REGISTER` | on by default; `0`/`off` turns it off | Multi-draw commands write the draw index into the register the game's shaders read; `0` leaves that register alone. | dev |
| `RPS5E_EARLY_FRAGMENT_TESTS` | on by default; `0` turns it off | Pixel shaders declare early depth/stencil testing where the game's state allows it; `0` leaves the choice to the driver (shaders that store to memory always test late). | dev |
| `RPS5E_EXT_DYNAMIC_STATE` | on by default when the device has VK_EXT_extended_dynamic_state; `0` turns it off | Enables the Vulkan extended-dynamic-state feature, which some drivers (RADV) need for draws to work even though its states are core in Vulkan 1.3; `0` leaves it disabled. | dev |
| `RPS5E_FILL_INVALIDATE` | `0`: off; default on | A pending metadata (fast-clear) fill is dropped when something else writes its memory before it is used. `0` keeps it until a bind takes it (older behaviour). | dev |
| `RPS5E_FLIP_EOP_EVENT` | on by default; `0` turns it off | A display flip also raises the graphics end-of-pipe event that games wait on; `0` raises none. | dev |
| `RPS5E_FLIP_EOP_LATCH` | on by default; `0` turns it off | A flip raises the end-of-pipe event only on queues that do not raise their own release interrupts; `0` raises it on every flip. | dev |
| `RPS5E_FORCE_DEGAMMA` | `0`: off; default on | Honours the sampler's force-degamma bit, so textures sampled with it are decoded as sRGB whatever their format. `0` ignores the bit. | dev |
| `RPS5E_FORCE_DST_ALPHA_1` | `0`: off; default on | Honours the render target's force-destination-alpha-1 bit in blending. `0` ignores the bit. | dev |
| `RPS5E_GPU` | `0`/`off`: off; default on | `0` disables the Vulkan device entirely: no GPU back-end, the presenter runs headless (nothing is drawn). | dev |
| `RPS5E_GPU_BACKEND` | `0`/`off`: off; default on | `0` disables the GPU back-end: GPU command buffers are only counted, never executed (no rendering). | dev |
| `RPS5E_GPU_HANG_TIMEOUT` | seconds; `0`: never; default 30 | How long the GPU may finish no work at all while the emulator waits for it before it is treated as hung. | dev |
| `RPS5E_GPU_IGNORE_MISSING` | `1`: on; default off | Starts even when the GPU lacks required Vulkan features (logs an error instead of refusing to run). Rendering may be wrong or crash. | dev |
| `RPS5E_GPU_LABEL_SYNC` | `1`: on; default off | Treats every satisfied GPU-memory wait as a CPU synchronisation point (also those on labels written by end-of-pipe work). Slower; for checking synchronisation bugs. | dev |
| `RPS5E_HTILE_FILL_DECODE` | `0`: off; default on | A memory fill over a depth buffer's HTILE metadata clears only the depth/stencil planes its values say. `0`: any such fill clears depth (older behaviour). | dev |
| `RPS5E_HTILE_KERNEL_DECODE` | `0`: off; default on | When a compute shader overwrites a depth buffer's HTILE metadata, the stored words decide the clear at next use. `0`: treated as a full depth and stencil clear (older behaviour). | dev |
| `RPS5E_HTILE_PER_SLICE` | `0`: off; default on | In a depth array, each slice takes the clear of its own HTILE metadata. `0`: one fill clears every slice and later slices' fills are ignored (older behaviour). | dev |
| `RPS5E_HTILE_STORE_BACK` | `0`: off; default on | After a depth fast clear is applied, the HTILE metadata in guest memory is written back as expanded, as the hardware does. `0`: left as cleared (older behaviour). | dev |
| `RPS5E_HTILE_WHOLE_CLEAR` | `0`/`off` turns off; default on (off with `RPS5E_SAFE=1`) | A depth clear coming from cleared HTILE metadata clears the whole surface. Off: it stays within the binding draw's scissor (older behaviour). | dev |
| `RPS5E_INDIRECT_GROUP_CLAMP` | `1`: on; default off | Indirect compute dispatches with group counts beyond the device's limits run as empty, instead of launching a near-endless dispatch that loses the GPU device. Costs an extra kernel per indirect dispatch. | dev |
| `RPS5E_LABELS_AT_EOP` | `1`, `0` or `through`; default `0` (`1` when `RPS5E_BDA=1`) | When end-of-pipe memory labels are written: `0` as soon as the work is recorded (fast), `1` once the GPU has finished it (exact, slower), `through` only waits for labels written while GPU results are still being copied back to guest memory. | dev |
| `RPS5E_LAYERED_VIEW_SLICES` | `0`: off; default on | A layered draw's colour view covers the slices the draw uses. `0`: clamped to the array's declared slices like any other (older behaviour). | dev |
| `RPS5E_LOAD_RACE_CHECK` | on by default; `0`/`off` turns it off; also off with `RPS5E_SAFE=1` | When a texture is loaded from guest memory while the CPU writes to it, it stays marked as CPU-written so it is reloaded later. Off: the load clears the marks anyway and such a write can be missed. | dev |
| `RPS5E_MAYBE_DIRTY` | on by default; only `0` turns it off | Edge pages of a texture shared with a neighbouring write are compared again at the next use, so CPU writes to the texture's bytes there are picked up. Off: such writes can be missed. | dev |
| `RPS5E_MESH` | on when the GPU supports VK_EXT_mesh_shader; only `0` turns it off | Uses mesh shaders for geometry-shader draws and for vertex programs that share data between lanes. Off: draws with a geometry stage are not drawn and such vertex programs run per vertex. | dev |
| `RPS5E_MESH_LINES` | on by default; only `0` turns it off | Geometry programs that output line strips are drawn. Off: those draws are skipped. | dev |
| `RPS5E_MESH_POINTS` | on by default; only `0` turns it off | Geometry programs that output point lists are drawn as mesh points. Off: those draws are refused. | dev |
| `RPS5E_MESH_STRIP_RESTART` | on by default; only `0` turns it off | Indexed strip draws run through mesh shaders honour primitive restart indices. Off: restart indices are fetched as ordinary vertices. | dev |
| `RPS5E_MESH_WAVE64_INFO` | opt-in: `1` turns it on; default off | Mesh-shader programs number lanes and waves as if every program were wave64 (the older layout) instead of telling wave32 programs their own waves. | dev |
| `RPS5E_METADATA_CONSUME_GPU` | on by default; `0`/`off` turns it off; also off with `RPS5E_SAFE=1` | When a fast clear is consumed and the GPU holds the surface's metadata, the result is written on the GPU too. Off: it is written to guest memory only. | dev |
| `RPS5E_MIXED_SAMPLES_DROP` | on by default; only `0` turns it off | A draw whose bound targets mix different multisample counts keeps only the targets matching the rasterizer's count. Off: all stay bound, which is an invalid Vulkan pass. | dev |
| `RPS5E_MSAA_CMASK` | on by default; `0` or `off` turns it off | Multisampled targets keep fast-clear state in CMASK, so their fast clears are applied. Off: CMASK clears of such targets are ignored. | dev |
| `RPS5E_NGG_MESH` | on by default (needs mesh shaders); only `0` turns it off | Vertex programs without a geometry stage that exchange data through LDS (culling, compaction) run as mesh workgroups with LDS shared per group. Off: they run per vertex with a private LDS copy. | dev |
| `RPS5E_NGG_MESH_STAGES` | on by default; only `0` turns it off | Only vertex programs whose stage setup is plain primitive generation run as mesh workgroups. Off: any merged program qualifies (older, looser rule). | dev |
| `RPS5E_NO_FLIP_PACING` | `1`: on; default off | Flips are shown as soon as they are ready instead of being paced to the guest refresh rate (`RPS5E_GUEST_REFRESH_RATE`); games may run faster than intended. | dev |
| `RPS5E_ORDERED_APPEND` | on by default; `0` or `off` turns it off | Ordered-append compute dispatches run group by group in order, as the console orders their appends. Off: all groups are dispatched at once (order not kept). | dev |
| `RPS5E_PACKED_CHANNELS` | on by default; only `0` turns it off | 5_6_5, 1_5_5_5 and 4_4_4_4 textures are read with the console's channel order. Off: the Vulkan channel order (colours can be swapped). | dev |
| `RPS5E_PER_FACE_BIAS` | on by default; only `0` turns it off | Draws with different depth bias for front and back faces are drawn in two passes, one per face. Off: one bias for both faces. | dev |
| `RPS5E_PIXEL_LDS_SIZE` | opt-in: `1` turns it on; default off | A pixel shader's LDS array is sized from the program's declared EXTRA_LDS_SIZE (up to 64 KiB) instead of only from the accesses it makes. | dev |
| `RPS5E_POLYGON_MODE_FACES` | on by default; only `0` turns it off | Wireframe/point fill mode is taken from the back-face setting when front faces are culled. Off: always the front-face setting. | dev |
| `RPS5E_PRT_HOLE_RANGES` | on by default; `0` or `off` turns it off | A buffer over partially mapped (sparse) memory keeps its mapped pages past an unmapped hole; the hole reads as zeros. Off: the range is cut at the first unmapped page. | dev |
| `RPS5E_PS_ITER_SAMPLE` | on by default; only `0` turns it off | Pixel shaders run once per sample when the game asks for per-sample shading. Off: always once per pixel. | dev |
| `RPS5E_PS_WAVE64` | on by default; only `0` turns it off | Pixel shaders run in wave64 unless the game selects wave32, as on the console. Off: every pixel shader runs as wave32. | dev |
| `RPS5E_QUAD_MASK_DATA` | on by default; only `0` turns it off | Per-pixel sample masks are passed to pixel shaders as data. Off: they are baked into the shader, one shader variant per mask (more compiles). | dev |
| `RPS5E_QUAD_PROVOKING_LAST` | on by default; only `0` turns it off | Quad primitives are split into triangles respecting the draw's provoking-vertex mode. Off: always split for the first provoking vertex (flat shading can use the wrong vertex). | dev |
| `RPS5E_READBACK_LINEAR_IMAGES` | default: at every collection; `access`: on the first CPU access; `0` or `off`: never; `RPS5E_SAFE=1` forces the default | Controls how GPU-written linear storage images are copied back to guest memory. `access` can leave readers that do not fault with stale data; `0` never copies them back. | dev |
| `RPS5E_SAMPLED_DCC` | on by default; only `0` turns it off | Textures that were fast-cleared through DCC compression keys and then only sampled read the clear colour, as on the console. Off: they read the raw bytes in memory. | dev |
| `RPS5E_SHADER_WRITE_INVALIDATE` | unset by default; the only value is `base` | By default a shader buffer write makes every cached image over the written range reload from memory at its next use. `base` refreshes only images that start at the buffer's address (an A/B check for render targets lost to oversized buffer ranges). | dev |
| `RPS5E_STAGING_RETAG` | on by default; `0` turns it off | Staging upload memory stays reserved until the GPU submission that actually reads it has finished, even when a mid-draw submit moved those commands later. With `0` it is freed with its original submission, which can overwrite data still in use. | dev |
| `RPS5E_STENCIL_EXPORT` | on by default; `0` turns it off | A stencil reference value written by the pixel shader is used (where the GPU supports it). With `0` it is ignored and the register's reference value is used. | dev |
| `RPS5E_STENCIL_INVERT` | on by default; `0` turns it off | Stencil replace operations whose value cannot share the host's single reference value are drawn as an equivalent bit inversion where the compare fixes the stencil value. With `0` the reference is kept and those writes can be wrong. | dev |
| `RPS5E_STENCIL_LOGIC` | on by default; `0` turns it off | Stencil logic operations (AND, OR, XOR, XNOR) are emulated on exactly the bits they change. With `0` AND, OR and XNOR keep the value and XOR inverts every writable bit. | dev |
| `RPS5E_UNCHECKED_EXTENT` | on by default; `0` turns it off | A zero-stride buffer descriptor without range checking is bound for the bytes its accesses actually read, even past its record count. With `0` only the record-count bytes are bound. | dev |
| `RPS5E_UNEXPORTED_INPUTS` | on by default; `0` turns it off | Pixel shader inputs whose value the vertex stage never exports read 0, as on the console. With `0` they get an undefined interpolated value. | dev |
| `RPS5E_UNKNOWN_HAZARDS` | on by default; `0` turns it off | Shader stages that read or write memory through addresses the emulator cannot bound wait for earlier unordered shader writes (and later work waits for them). With `0` they are not ordered, which can cause races. | dev |
| `RPS5E_UNNORMALIZED_BASE_LEVEL` | on by default; `0` turns it off | Textures sampled with unnormalized coordinates are bound as their base mip level only, which those reads use. With `0` the whole mip chain is bound. | dev |
| `RPS5E_UNRANGED_IMPORTS` | on by default; `0` or `off` turns it off | Shader reads through unbounded device addresses where no GPU copy of the memory exists read guest memory directly. With `0` they read zeros there. | dev |
| `RPS5E_UNRANGED_SYNC` | on by default; `0` or `off` turns it off | Before work that reads memory through unbounded device addresses, pages the CPU wrote are brought up to date on the GPU. With `0` such reads can see stale data, and zeros where no GPU copy exists. | dev |
| `RPS5E_UNVERSIONED_REG_DEFAULTS` | on by default; `0` turns it off | `sceAgcGetRegisterDefaults` returns the newest register-defaults table; `0` returns the version-0 compact table. | dev |
| `RPS5E_USER_CLIP_PLANES` | on by default; `0` turns it off | User clip planes the game enables are applied through clip distances. With `0` they are ignored. | dev |
| `RPS5E_WAIT64_NARROW` | on by default; `0` turns it off | A 64-bit GPU memory wait whose mask has no high bits reads only 4 bytes; `0` always reads 8 aligned bytes. | dev |
| `RPS5E_WRITEBACK_BEFORE_BUFFER_WRITE` | on by default; `0` turns it off | Before GPU work writes memory through a buffer, images there that the GPU changed are written back first, so both the image's results and the buffer's stores survive. With `0` a partial buffer store can drop the rest of the image. | dev |
| `RPS5E_ZERO_POSITION_CLIP` | on by default; `0` turns it off | Primitives with a vertex at an all-zero clip position are clipped away, as the console does. With `0` the host draws them, which shows as long stretched triangles. | dev |
| `RPS5E_ZERO_STRIDE_EXTENT` | on by default; `0` turns it off | A zero-stride buffer descriptor without range checking also covers the bytes its fetches read past its record count. With `0` those bytes are not copied to the GPU. | dev |

## Shaders and caches

| Variable | Values and default | What it does | For |
|---|---|---|---|
| `RPS5E_ASYNC_PIPELINES` | on by default; `0`/`off` turns it off (also off with `RPS5E_SAFE=1`) | Compiles shaders and pipelines on background worker threads so the game keeps running (draws whose pipeline is not ready may be skipped briefly); `0` compiles on the render thread when first needed, which stutters but never skips a draw. Also in the in-game menu. | user |
| `RPS5E_COMPILE_INDICATOR` | on by default; `0`/`off` turns it off | Shows the on-screen "compiling N shaders" indicator while shaders compile; `0` hides it. | user |
| `RPS5E_COMPILE_THREADS` | number of threads (> 0); default: logical CPUs minus 2 (at least 2) | Number of background shader/pipeline compile workers. | user |
| `RPS5E_COMPILE_WAIT_MS` | milliseconds, or `exact`/`wait`/`inf`; default when unset: `exact`, but `rps5e-cli run` and the launcher set 3 (also `--compile-wait`) | Per presented frame, how long draws wait for their shaders to finish compiling before being skipped (`0` skips at once); `exact` always waits, which avoids missing or fading effects but stutters. | user |
| `RPS5E_SHADER_PREWARM` | on by default; `0` turns it off | Keeps a per-game list of the shaders and pipelines the game used, written while playing and compiled at the next start before the game runs. `0` neither reads nor writes the list. | user |
| `RPS5E_BUFFER_ACCESS_ONCE` | `1` turns it on; default off | Shader code checks the buffer bounds once for a multi-dword buffer load instead of once per dword. | dev |
| `RPS5E_COHERENT_PAIRS` | on by default; `0` turns it off | Globally coherent two-dword buffer loads and stores are done as one 64-bit access; `0` keeps them as two separate dword accesses. | dev |
| `RPS5E_COMPILE_LOOKAHEAD` | number of dispatches; default 256; `0`/`off` turns it off | Scans ahead in the GPU command stream and starts compiling the compute shaders of upcoming dispatches early; the number sets how many dispatches ahead are looked at. | dev |
| `RPS5E_CONST_TEXEL_OFFSET` | on by default; `0` turns it off | Constant texture-sample offsets are compiled as native constant offsets; `0` passes them at run time instead. Part of the shader cache key. | dev |
| `RPS5E_DATA_CACHE_INVALIDATE` | on by default; `0`/`off` turns it off | Shader data-cache invalidate instructions become memory barriers that make earlier buffer writes visible; `0` ignores them. Part of the shader cache key. | dev |
| `RPS5E_DEVICE_ACCESS_ONCE` | on by default; `0`/`off` turns it off (also off with `RPS5E_SAFE=1`) | Shader code translates a multi-dword device-address access within one page once and loads the dwords as one vector; `0` translates each dword. | dev |
| `RPS5E_EMIT_CSE` | on by default; `0`/`off` turns it off (also off with `RPS5E_SAFE=1`) | The shader compiler shares common subexpressions across blocks along the dominator tree, not only within one block. | dev |
| `RPS5E_EMIT_FOLD` | `1` turns it on; default off | The shader compiler folds integer and boolean operations on constants (constant range checks, `x + 0`, selects on constants) instead of emitting them. | dev |
| `RPS5E_EXEC_RUN` | on by default; `0`/`off` turns it off (also off with `RPS5E_SAFE=1`) | In compute shaders, runs of vector writes under the same lane mask share one branch; `0` turns the grouping off. Also in the in-game menu. | dev |
| `RPS5E_EXEC_RUN_MIN` | integer >= 2; default 6 | With `RPS5E_EXEC_RUN` on: the shortest run of vector writes that is grouped under one branch. | dev |
| `RPS5E_FAST_TRANSCENDENTAL` | `1`: on; default off | Shader exp/log/reciprocal/rsqrt/sqrt use the host GPU's own precision and special values instead of the console's exact results. Faster but inexact; a warning is logged. | dev |
| `RPS5E_FLOAT_CLASS_COMPARE` | `1`: on; default off | Shader denormal-flush and NaN tests are compiled as float compares instead of integer bit tests. Opt-in: caused edge-of-screen garbage in testing. | dev |
| `RPS5E_FLOAT_RULES` | unset: all rules; `0`/`off`: none; otherwise a bit mask (decimal or `0x` hex) | Selects which hardware-measured float behaviours the shader translator reproduces (denormal flushes, division corner cases, f16 rounding, sin/cos series, …). Part of the shader disk-cache key. | dev |
| `RPS5E_FLUSH_ELISION` | `1`: on; default off | Skips emitting a denormal flush for values that provably cannot be denormal. Opt-in: caused edge-of-screen garbage in testing. | dev |
| `RPS5E_FULL_WAVE_EXEC` | `1`: on; default off | Compute shaders whose work-group is whole waves start with all lanes marked active, letting masked writes be simplified. Measured neutral on NVIDIA. | dev |
| `RPS5E_HALF_FMA_FAST` | `1`: on; default off | Mixed-precision f16 fused multiply-add rounds the f32 result to f16 directly, taking the exact slow path only at f16 rounding midpoints. Opt-in. | dev |
| `RPS5E_LANE_REDUCTIONS` | `0`/`off` turns off; default on (off with `RPS5E_SAFE=1`) | Wave reductions read back in graphics shaders are computed as host subgroup reductions over the lanes that exist, instead of reading lanes the host subgroup may lack. | dev |
| `RPS5E_LOG_SERIES` | on by default; `0` or `off` turns it off (not affected by `RPS5E_SAFE`) | Shader `v_log_f32` (and its f16 form) uses an exact series for inputs near 1.0 to match the console's precision. Off: the host GPU's plain log2 everywhere. Part of the shader cache key. | dev |
| `RPS5E_NATIVE_F16_RTE` | on by default where the device guarantees it; only `0` turns it off | f32-to-f16 conversions in shaders use the GPU's native round-to-nearest-even conversion. Off: an exact integer emulation everywhere (slower). | dev |
| `RPS5E_NGG_VERTEX_WAVE_INFO` | opt-in: `1` turns it on; default off | Merged vertex programs get one-wave subgroup inputs (active count, wave count, vertex/primitive counts). Off by default because it caused visible corruption and a slowdown in a tested game. Part of the shader cache key; also checked by some device tests. | dev |
| `RPS5E_PIPELINE_ROBUSTNESS` | opt-in: `1` turns it on (needs VK_EXT_pipeline_robustness); default off | Shaders guard their own buffer loads so compute pipelines can run without the driver's buffer robustness. Off by default because it measured slower on NVIDIA. Part of the shader cache key. | dev |
| `RPS5E_RELOCATABLE_CODE` | on by default; only `0` turns it off | Shader programs that read their own address are compiled once and work at any load address. Off: one compiled shader per address it is loaded at. | dev |
| `RPS5E_SHADER_COMPILER` | on by default; `0` or `off` turns it off | Turns off the shader compiler: no game shader is translated, so nothing is drawn (with `RPS5E_GPU_PLAN=1` the shader front end still runs for diagnostics). | dev |
| `RPS5E_SHADER_LINK` | on by default; `0` turns it off | Shader code that a program reaches through jumps or calls into other memory is linked into that program. With `0` such programs fail translation and their draws are skipped. | dev |
| `RPS5E_STORAGE_RESINFO_MIPS` | on by default; `0` turns it off | Size queries on storage images report the requested mip level's size and the real level count. With `0` they report the base level's size and one level. Part of the shader cache key. | dev |
| `RPS5E_STORE_QUANTIZE` | on by default; `0` turns it off | The shader itself converts float values to UNORM/SNORM when writing storage images, matching the console's rounding. With `0` the conversion is left to the host GPU. | dev |
| `RPS5E_STRICT_MODE_WRITES` | `1` turns it on; off by default | Shaders that change the float rounding mode in ways the translator does not model fail translation (their draws are skipped) instead of running with the declared mode. Clear the shader cache when switching it. | dev |
| `RPS5E_VERTEX_GUEST_WAVE` | on by default; `0` turns it off | Vertex shaders outside geometry subgroups are translated with the game's own wave size (wave32 or wave64). With `0` all are translated as wave32, and wave64 programs can read wrong lane masks. | dev |
| `RPS5E_WAVE64_BALLOT_RIDERS` | on by default; `0` or `off` turns it off | In wave64 compute shaders run split on a 32-wide GPU, later ballots reuse an earlier ballot's lane exchange; with `0` each ballot does its own. Part of the shader cache key. | dev |
| `RPS5E_WAVE64_LANE_SLOTS` | on by default; `0` or `off` turns it off | In split wave64 compute shaders, lane-indexed registers are kept in lane slots; with `0` their lane reads go through exchanges. Part of the shader cache key. | dev |
| `RPS5E_WAVE64_MAX_RIDERS` | integer; default 3 | Split wave64 compute shaders: the most ballots that may reuse one earlier ballot's exchange (see `RPS5E_WAVE64_BALLOT_RIDERS`). | dev |

## Performance switches

Each of these turns an optimisation off (or, for opt-in ones, on). They are for finding out whether a bug or a slowdown comes from that optimisation; the defaults are what games should run with. `RPS5E_SAFE=1` turns most GPU fast paths off at once, and `RPS5E_AB_PERIOD` alternates them during one run for speed comparisons.

| Variable | Values and default | What it does | For |
|---|---|---|---|
| `RPS5E_FAST_PATHS` | `0`/`off` turns off; default on (also off with `RPS5E_SAFE=1`); also toggled live from the in-game menu | Per-draw GPU fast paths (memoised draw state, draw recipes, upload prologues). Turning them off is slower but rules them out when a picture looks wrong. | user |
| `RPS5E_AB_ONLY` | comma-separated switch names; unset: all | With `RPS5E_AB_SCOPE=release`, only the named release-check switches (for example `RPS5E_DCC_IN_PLACE,RPS5E_DESC_WINDOW_TICKS`) toggle during A/B measurement. | dev |
| `RPS5E_AB_PERIOD` | seconds (> 0); unset: no A/B | A/B speed measurement within one run: the GPU fast paths are on during even periods of this length and off during odd ones, and the GPU profile summary counts draws and recording-thread time per mode (`ab=`). | dev |
| `RPS5E_AB_SCOPE` | `all` (default), `thread`, `paths`, `new`, `bda`, `release` | Chooses which group of fast paths `RPS5E_AB_PERIOD` toggles: all of them, only the recording thread, only the per-draw paths, only the newer paths (draw recipes, early submits, write-throughs, sub-allocation), only guest-memory reads, or only the release-check paths. | dev |
| `RPS5E_BDA` | `1` turns it on; default off | Experimental: read-only shader stages read guest memory directly through device addresses instead of copied buffers. Frame captures taken with it may not replay identically. Also in the in-game menu. | dev |
| `RPS5E_BDA_MIRROR_MIN` | bytes; default 65536 (64 KiB) | With `RPS5E_BDA=1`: ranges at least this large are mirrored in video memory while the CPU does not write them; smaller ranges are read from guest memory over the bus. | dev |
| `RPS5E_BDA_PLAN` | on by default; `0` turns it off; `buffers` or `all` widen it | With `RPS5E_BDA=1`: shaders read their scalar data from guest memory themselves; `buffers` also flattens data reads on the host side, `all` also leaves buffer descriptors to the shader (only correct when every guest mapping is visible to the GPU); `0` keeps the ordinary resource plan. | dev |
| `RPS5E_BDA_PRIVATE` | on by default; `0` turns it off | With `RPS5E_BDA=1`: private guest memory mappings are imported for direct GPU reads; `0` serves them from mirrored copies instead. | dev |
| `RPS5E_BDA_RESIDENT` | `1` turns it on; default off | Experimental, with `RPS5E_BDA=1`: pages that have a current mirror in video memory are read from it instead of over the bus (known to show stale reads in some scenes). | dev |
| `RPS5E_BINDLESS_TABLES` | on by default; `0` turns it off | Large bindless texture tables that a draw cannot bind directly are looked up through a host-side table map; `0` makes such draws fail instead (they are skipped). | dev |
| `RPS5E_BIND_MEMO` | on by default; `0` turns it off (also off with `RPS5E_SAFE=1` and during A/B slow periods) | Reuses a draw's resource bindings when nothing they depend on changed; `0` binds every resource again on every draw. Also in the in-game menu. | dev |
| `RPS5E_BUFFER_MEMO` | on by default; `0`/`off` turns it off (also off with `RPS5E_SAFE=1`) | Repeated read-only uses of the same buffer range skip the synchronisation work; `0` synchronises every use. Also in the in-game menu. | dev |
| `RPS5E_BUFFER_STREAM` | on by default; `0` turns it off | Small read-only buffer ranges (up to 16 KiB) whose guest bytes are newer than the GPU copy are bound from a fresh copy of those bytes; `0` always binds them from the GPU mirror. | dev |
| `RPS5E_BUFFER_SUBALLOC` | on by default; `0` turns it off (also off with `RPS5E_SAFE=1` and during A/B slow periods) | Places GPU buffers inside larger shared allocations; `0` makes one memory allocation per buffer. Also in the in-game menu. | dev |
| `RPS5E_COPY_KERNEL_HLE` | on by default; `0` turns it off | Recognises the SDK's image-copy compute kernel and performs it as a direct image copy when that gives the exact same result; `0` always runs the compute dispatch. | dev |
| `RPS5E_DCC_GPU` | on by default; `0` turns it off | Decides on the GPU whether a compute shader wrote a fast-clear code into a render target's compression metadata, avoiding a CPU wait for the GPU; `0` disables the GPU path. | dev |
| `RPS5E_DCC_GPU_ANY` | on by default; `0`/`off` turns it off (also off with `RPS5E_SAFE=1`) | Uses the GPU fast-clear decision for all target kinds; `0` limits it to plain 2D single-mip targets and reads the metadata back for others (a GPU wait per clear). | dev |
| `RPS5E_DCC_IN_PLACE` | on by default; `0`/`off` turns it off (also off with `RPS5E_SAFE=1`; toggled by A/B scope `release`) | Expands single-color compressed blocks of a sampled surface directly in guest memory; `0` expands them in a copy that is written back afterwards (same bytes). | dev |
| `RPS5E_DEPTH_READ_ONLY` | on by default; `0` turns it off (also off with `RPS5E_SAFE=1` and during A/B slow periods) | A draw that samples its own depth attachment keeps it in a read-only layout within the current render pass; `0` gives such a draw a render pass of its own. | dev |
| `RPS5E_DEPTH_STENCIL_SPLIT` | on by default; `0`/`off` turns it off (also off with `RPS5E_SAFE=1`) | A draw that writes only stencil while sampling the depth of the same attachment stays in the current render pass; `0` gives it a render pass of its own with full barriers around it. | dev |
| `RPS5E_DESC_WINDOW_KIB` | KiB (rounded up to a power of two, at most 1 GiB); default 1024; `0`: the single page only | When the CPU reads a GPU-written descriptor word, the GPU-written pages of this aligned window around it are downloaded together. | dev |
| `RPS5E_DESC_WINDOW_TICKS` | on by default; `0`/`off` turns it off (also off with `RPS5E_SAFE=1`; toggled by A/B scope `release`) | A descriptor read downloads only the window pages that do not make it wait longer; `0` downloads the whole window regardless. | dev |
| `RPS5E_DOWNLOAD_POOL` | on by default; `0`/`off` turns it off (also off with `RPS5E_SAFE=1`) | Reuses host-visible staging buffers for GPU-to-guest-memory downloads; `0` allocates and frees a buffer for each download. | dev |
| `RPS5E_DRAW_RECIPES` | on by default; `0` turns it off (also off with `RPS5E_SAFE=1` and during A/B slow periods) | Remembers each draw's shaders, views, layout and pipeline and reuses them for identical draws; `0` looks everything up on every draw. | dev |
| `RPS5E_EAGER_HOT_THROUGH` | on by default; `0`/`off` turns it off (also off with `RPS5E_SAFE=1` and during A/B slow periods) | GPU results that a CPU access waited for before are copied back to guest memory right after the dispatch that writes them; `0` waits for the next submit like other copies. | dev |
| `RPS5E_EAGER_THROUGH` | on by default; `0` turns it off (also off with `RPS5E_SAFE=1` and during A/B slow periods) | Copies of a compute dispatch's results back to guest memory are recorded right after the dispatch, so CPU reads wait less; `0` records them at the next submit. | dev |
| `RPS5E_EARLY_SUBMIT` | on by default; `0`/`off` turns it off (also off with `RPS5E_SAFE=1` and during A/B slow periods); or `<dispatches>,<draws>,<us>` thresholds (default 64,512,500) | Submits recorded GPU work early, after this many dispatches or draws or microseconds, so the GPU starts sooner on long command streams; `0` submits only at the end of each command-stream slice. | dev |
| `RPS5E_EMPTY_INDIRECT_SKIP` | on by default; `0`/`off` turns it off (also off with `RPS5E_SAFE=1`) | Skips indirect compute dispatches whose CPU-written arguments have a zero group count; `0` records them like any other. | dev |
| `RPS5E_FAST_MUTEX` | on by default; `0`, `off`, `false` or `no` turns it off | Guest mutexes keep their owner in an atomic lock word (uncontended lock/unlock without the host lock); off, every mutex operation takes the host lock. | dev |
| `RPS5E_FAST_PIPELINES` | `1`/`on`: on; `only`: unoptimised builds only (diagnostic); default off (forced off by `RPS5E_SAFE=1`) | A pipeline not yet in the pipeline cache is first built without driver optimisation so the draw can run at once, and an optimised build replaces it later. Gains nothing on drivers that compile equally fast either way. | dev |
| `RPS5E_FAST_WAIT` | on by default; `0`, `off`, `false` or `no` turns it off | Blocking waits register themselves without the global signal-table lock; off, each wait takes that lock twice. | dev |
| `RPS5E_FAULT_LANDING` | `0`/`off` turns off; default on (off with `RPS5E_SAFE=1`) | When the CPU touches memory the GPU wrote, a copy that already finished just unprotects the page without taking the GPU back-end lock. Off: every such access goes through the back-end and waits. | dev |
| `RPS5E_FAULT_WINDOW_GROW` | `0`/`off` turns off; default on (off with `RPS5E_SAFE=1` or in A/B slow periods) | When the CPU reads a large GPU-written range in sequence, each fault downloads a window that doubles up to 32 MiB, so fewer GPU waits. Off: every fault downloads 1 MiB. | dev |
| `RPS5E_FILL_HLE` | `0`: off; default on | Recognised SDK memory-fill compute kernels (plain, pattern and indexed fills) are done as direct buffer fills instead of running the shader. `0` runs them all as normal dispatches. | dev |
| `RPS5E_FILL_INDEX` | `0`/`off` turns off; default on (off with `RPS5E_SAFE=1`; toggled by A/B scope `release`) | A GPU write only checks nearby pending metadata fills instead of scanning all of them; avoids a large CPU cost in games with thousands of pending fills. | dev |
| `RPS5E_GPU_INDIRECT` | `0`/`off` turns off; default on (off with `RPS5E_SAFE=1`) | Indirect draws read their argument records on the GPU. Off: the CPU reads every record, waiting for the GPU work that wrote it. | dev |
| `RPS5E_GPU_INDIRECT_MESH` | `0`/`off` turns off; default on (off with `RPS5E_SAFE=1`) | Indirect mesh draws (geometry and primitive subgroups) read their record on the GPU. Off: read on the CPU. | dev |
| `RPS5E_GPU_INDIRECT_RECT` | `0`/`off` turns off; default on (off with `RPS5E_SAFE=1`) | Indirect rectangle-list draws read their records on the GPU. Off: read on the CPU. | dev |
| `RPS5E_GPU_INDIRECT_THREADS` | `0`/`off` turns off; default on (off with `RPS5E_SAFE=1`) | Indirect compute dispatches with thread dimensions read their record on the GPU. Off: the CPU reads it (waiting for the GPU) and runs a direct dispatch. | dev |
| `RPS5E_GPU_PREFETCH` | `0`: off; default on (off with `RPS5E_SAFE=1` or in A/B slow periods) | The command interpreter thread pre-prepares the shader stages of draws it hands to the recording thread. Off: the recorder prepares them itself. | dev |
| `RPS5E_GPU_RECORD_THREAD` | `1`: on; default off; also an in-game menu switch | Experimental: the command interpreter hands draws to a separate recording thread instead of recording them itself. Measured slower in tested games. | dev |
| `RPS5E_GPU_ZERO_COPY` | `0`/`off`: off; default on; also an in-game menu switch | GPU-to-CPU downloads are written straight into the imported guest memory pages. Off: they go through a staging buffer and a CPU copy. | dev |
| `RPS5E_HOT_PAGES` | `1`: on; default off (forced off by `RPS5E_SAFE=1`) | Guest pages the CPU keeps rewriting stop being write-watched and are compared with the last upload instead. Off by default because it handed the GPU stale data in testing. | dev |
| `RPS5E_IMAGE_DIRECT_COPY` | `0`/`off` turns off; default on (off with `RPS5E_SAFE=1`) | Format conversions between images sharing memory are done as one direct image copy. Off: they go through a buffer or the guest memory layout. | dev |
| `RPS5E_IMAGE_GENERATIONS` | `0`/`off` turns off; default on (off with `RPS5E_SAFE=1` or in A/B slow periods) | A change to one image invalidates only the memoised texture lookups of that image and its aliases. Off: every cached lookup is invalidated. | dev |
| `RPS5E_IMAGE_SUBALLOC` | `1`: on; default off | Small images share larger GPU memory blocks instead of one driver allocation each; reduces stalls when many new textures appear at once. | dev |
| `RPS5E_IMPORT_UPLOADS` | `0`/`off` turns off; default on (off with `RPS5E_SAFE=1`; toggled by A/B scope `release`) | Large texture uploads that do not fit the staging ring read texels straight from imported guest memory. Off: copied into a fresh upload buffer first. | dev |
| `RPS5E_INDEXED_FILL_HLE` | `0`: off; default on (also off with `RPS5E_FILL_HLE=0`) | The SDK indexed pattern-fill compute kernel is done as a direct buffer fill instead of running the shader. | dev |
| `RPS5E_INDEXED_INVALIDATE` | `0`/`off` turns off; default on (off with `RPS5E_SAFE=1` or in A/B slow periods) | A GPU buffer write checks only the images a page index names. Off: it checks every image's watch entry. | dev |
| `RPS5E_LAYOUT_INTERN` | `0`/`off` turns off; default on (off with `RPS5E_SAFE=1` or in A/B slow periods) | Shader binding layouts with equal contents are shared, so cached draw recipes and pipeline layouts keep hitting; keeps more templates and layouts per shader plan. | dev |
| `RPS5E_LAYOUT_MEMO` | on by default; `0`/`off` turns it off; also off with `RPS5E_SAFE=1`; toggled by A/B scope `release` | Remembers per thread which parts of a guest memory range are mapped, so draws that bind buffers inside the same partially mapped pools do not walk the memory map again each time. Off: the map is walked for every query. | dev |
| `RPS5E_LAZY_SELECT` | opt-in: `1` turns it on; default off | Shader translation: operations needed only by one side of a select run inside a branch instead of always. Exact; part of the shader cache key. | dev |
| `RPS5E_LEAN_FLUSH` | on by default; `0`/`off` turns it off; also off with `RPS5E_SAFE=1` | Shader translation: the emulated f32 denormal flush is one compare and one select instead of the longer older code; same results. Part of the shader cache key. | dev |
| `RPS5E_LOG_SERIES_BRANCH` | on by default; `0`/`off` turns it off; also off with `RPS5E_SAFE=1` | Shader translation: the near-1.0 log series is computed only in lanes whose input needs it (a branch) instead of always computed and selected; same results. Part of the shader cache key. | dev |
| `RPS5E_MUTEX_HANDOFF` | `1` (or any value except `0`/`off`/`false`/`no`): on; default off | A contended mutex unlock hands the mutex to the longest-waiting thread (strict FIFO) instead of letting waiters race for it. Fairer but slower. | dev |
| `RPS5E_MUTEX_SPIN_ALL` | `1` (as above): on; default off | Every mutex kind spins before sleeping, not only adaptive mutexes. | dev |
| `RPS5E_MUTEX_SPIN_US` | microseconds, 0 to 1000; default 4 | How long a thread spins on a contended adaptive mutex before it sleeps; `0` never spins. | dev |
| `RPS5E_NATIVE_FTZ` | on by default; `0`/`off` turns it off; also off with `RPS5E_SAFE=1` | Shaders that flush f32 denormals everywhere use the GPU's own flush-to-zero mode instead of emulated flush code. Part of the shader cache key. | dev |
| `RPS5E_PAGE_CLEAN_MEMO` | on by default; only `0` turns it off; also off with `RPS5E_SAFE=1` or an A/B slow period | A remembered read-only buffer use stays valid while its own pages are unchanged. Off: any CPU write anywhere invalidates it. | dev |
| `RPS5E_PAGE_GROUPS` | on by default; `0`/`off` turns it off; also off with `RPS5E_SAFE=1`; toggled by A/B scope `release` | Page change counters are bumped in groups instead of page by page (same result, less work). | dev |
| `RPS5E_PATCH_IN_PLACE` | on by default; `0` or `off` turns it off | Reuses snapshot buffers when cached shader-resource tables are patched for a draw; off, each patch allocates a new one. | dev |
| `RPS5E_PATTERN_FILL_HLE` | on by default; `0` turns it off (as does `RPS5E_FILL_HLE=0`) | The SDK's pattern-fill compute kernel with a single repeated value is done as a direct buffer fill instead of running the shader. Off: it runs as a normal dispatch. | dev |
| `RPS5E_PINNED_LOG` | on by default; `0`/`off` turns it off; also off with `RPS5E_SAFE=1`; toggled by A/B scope `release` | Bindless texture tables re-check only the images whose memory changed. Off: every image a table holds is re-checked whenever any pinned image changes. | dev |
| `RPS5E_PIN_RECORDER` | on by default; `0` or `off` turns it off | Pins the GPU command, recording and slice threads to the fast cores of a hybrid (P/E-core) or asymmetric-cache CPU. | dev |
| `RPS5E_PM4_WAKE` | on by default; `0` or `off` turns it off | GPU queues blocked on a memory wait are re-checked as soon as a label or flip is written; off, they are polled only every 100 µs. | dev |
| `RPS5E_PRECISE_BARRIERS` | on by default; only `0` turns it off; also off with `RPS5E_SAFE=1` or an A/B slow period | Compute dispatches get only the barriers they need. Off: a full barrier before and after every dispatch (slower). | dev |
| `RPS5E_PREFETCH_RANGES` | opt-in: `1` turns it on; default off; also off with `RPS5E_SAFE=1` or an A/B slow period | The command-interpreter thread also works out each draw's buffer ranges ahead of the recording thread. Off by default because it measured slower. | dev |
| `RPS5E_PREPARE_MEMO` | on by default; only `0` turns it off; also off with `RPS5E_SAFE=1` or an A/B slow period | Remembers each draw's prepared shader stages and reuses them. Off: every draw prepares its stages again. | dev |
| `RPS5E_PREPARE_PAGES` | on by default; `0`/`off` turns it off; also off with `RPS5E_SAFE=1` or an A/B slow period | Remembered shader stages stay valid while the memory pages they read are unchanged. Off: they are dropped whenever any guest memory may have changed. | dev |
| `RPS5E_PRESSURE_SETTLE` | on by default; `0`/`off` turns it off; also off with `RPS5E_SAFE=1` | Under memory pressure, GPU-written buffers are settled and released at a later cleanup without waiting. Off: each is downloaded first, one GPU wait each. | dev |
| `RPS5E_RANGE_BARRIERS` | on by default; `0`/`off` turns it off; also off with `RPS5E_SAFE=1` or an A/B slow period | Dispatches and copies of GPU results to guest memory are ordered only against work touching overlapping memory ranges. Off: full barriers are used instead (slower). | dev |
| `RPS5E_RANGE_CACHE4` | on by default; `0`, `off`, `false` or `no` turns it off | Each thread caches its last four guest memory range lookups; off, only one. | dev |
| `RPS5E_READBACK_ANY_SIZE` | on by default; `0`/`off` turns it off; also off with `RPS5E_SAFE=1` or an A/B slow period | Any GPU-written range the CPU has read back is afterwards written through to guest memory automatically. Off: only ranges up to 16 pages are, so large ranges the CPU reads every frame stall on the GPU each frame. | dev |
| `RPS5E_RECIPE_KEY` | on by default; `0`/`off` turns it off; also off with `RPS5E_SAFE=1` or an A/B slow period | Remembered draw setups use a compact key. Off: keys hold the whole bound targets, one remembered setup per hash. | dev |
| `RPS5E_RECORD_QUEUE` | a number of queued calls, at least 8; default 128 (invalid or below 8: 128) | How many calls the command interpreter may queue before waiting for the recording thread; larger lets the recorder lag further behind. | dev |
| `RPS5E_REGISTRY_CACHE` | `1`: on; default off | Lookups of semaphores, event flags, event queues and threads by handle use a per-thread cache instead of the shared registry lock. | dev |
| `RPS5E_RESOLVED_RANGES` | on by default; `0`/`off` turns it off; also off with `RPS5E_SAFE=1` or an A/B slow period | Draws reuse the buffer ranges and stream decisions worked out when they were reserved. Off: each binding works them out again. | dev |
| `RPS5E_RESOURCE_MEMO` | on by default; `0`/`off` turns it off; also off with `RPS5E_SAFE=1` or an A/B slow period | When a buffer binding changes, its unchanged buffers and ranges reuse the previous results. Off: every buffer of a changed binding is bound again. | dev |
| `RPS5E_RETIRED_BYTES_MIB` | MiB; default 512; `0` disables the limit | When GPU memory freed during recording reaches this amount, the open batch of GPU work is submitted early so the memory can be reclaimed. | dev |
| `RPS5E_RING_BINDING` | opt-in: `1` turns it on (needs `RPS5E_WIDE_OFFSETS=1`); default off; also off with `RPS5E_SAFE=1` or an A/B slow period | Read-only buffer bindings bind their whole containing buffer and pass the changing offset as data, so constant rings that move every draw keep reusing the binding. | dev |
| `RPS5E_SEMA_TARGETED_WAKE` | on by default; `0`, `off`, `false` or `no` turns it off | A semaphore signal wakes only the waiters it can satisfy; off, every signal wakes all waiters. | dev |
| `RPS5E_SETTLE_SKIP` | on by default; only `0` turns it off; also off with `RPS5E_SAFE=1` or an A/B slow period | Memory cleanups skip idle GPU-written buffers that have nothing to settle. Off: every cleanup walks all of them again. | dev |
| `RPS5E_SHADER_READONLY` | on by default; `0` or `off` turns it off; `RPS5E_SAFE=1` also turns it off | Buffers a shader never writes are marked read-only in the compiled shader so the driver can optimise them. | dev |
| `RPS5E_SHARED_SCOPES` | on by default; `0` or `off` turns it off; `RPS5E_SAFE=1` also turns it off; also in the in-game GPU menu | Consecutive draws share one Vulkan rendering scope. Off, every draw gets a scope of its own (slower). | dev |
| `RPS5E_SLICES` | integer 0-16 (larger values clamp to 16); default 0 (off) | Experimental slice mode: the GPU command thread only parses packets and logs register writes, and a separate slice thread rebuilds the state and records the draws in parallel with it. Any value above 0 turns it on in a game; in `rps5e-cli replay` it is the default of `--slices` (number of slice threads). | dev |
| `RPS5E_SLICE_AFFINITY` | `1` turns it on; off by default | Slice mode (`RPS5E_SLICES`): a wave's slices go to fixed slice threads instead of whichever is free, so the same draws hit the same thread's caches every frame. | dev |
| `RPS5E_SLICE_DRAWS` | integer ≥ 1; default 128 | Slice mode (`RPS5E_SLICES`): about how many draws go into one slice before it is cut at the next render-target change. | dev |
| `RPS5E_SLICE_QUEUE` | integer ≥ 1; default 4 | Slice mode (`RPS5E_SLICES`): the most slices handed to slice threads and not yet executed. | dev |
| `RPS5E_SLICE_UPLOADS` | on by default; `0` turns it off | Parallel slice recording: a draw that needs CPU-written buffer data is still recorded in parallel and its upload is made when the slice is committed. With `0` such draws are recorded serially. | dev |
| `RPS5E_SLICE_WAVE` | integer ≥ 1; default 4 | Parallel slice recording: slices per wave for each slice thread; threads take a wave's slices as they become free. | dev |
| `RPS5E_SLOT_LOOKUPS` | on by default; `0` or `off` turns it off; `RPS5E_SAFE=1`, `RPS5E_FAST_PATHS=0` and A/B slow periods also turn it off | The layout of each sampled texture is taken from what its binding already found instead of being worked out again on every draw. | dev |
| `RPS5E_SMEM_ONCE` | on by default; `0` or `off` turns it off; `RPS5E_SAFE=1` also turns it off | In wave64 shaders run as two guest lanes per host invocation, each scalar memory load is done once and shared by both halves instead of once per half. | dev |
| `RPS5E_BVH_VECTOR_LOADS` | on by default; `0` or `off` turns it off | Ray-tracing shaders read each BVH node with 16-byte vector loads under one residency check instead of one load per dword. | dev |
| `RPS5E_SOCKET_EVENTS` | on by default; `0` turns it off | Kernel sockets can be added to event queues as read/write events; `0` makes such registrations fail (`ENOENT`). | dev |
| `RPS5E_SPLIT_STAMPS` | on by default; `0` turns it off; `RPS5E_SAFE=1`, `RPS5E_FAST_PATHS=0` and A/B slow periods also turn it off | A buffer content or layout change drops only a stage's memoised buffer bindings; with `0` it drops all of them, textures and samplers included. | dev |
| `RPS5E_STAGE_MEMO_LRU` | on by default; `0` or `off` turns it off; `RPS5E_SAFE=1`, `RPS5E_FAST_PATHS=0` and A/B slow periods also turn it off | Each shader stage keeps up to 16 memoised binding sets (games switch materials and layouts on most draws); with `0` only one, so a draw with another layout drops the previous bindings. | dev |
| `RPS5E_STATE_FILTER` | on by default; `0` turns it off; `RPS5E_SAFE=1`, `RPS5E_FAST_PATHS=0` and A/B slow periods also turn it off; also in the in-game GPU menu | Draw state that did not change since the previous draw is not recorded again; with `0` every draw records its whole state. | dev |
| `RPS5E_STREAM_CARRY` | `1` turns it on; off by default; ignored with `RPS5E_SAFE=1` | Staging copies of buffer ranges whose guest pages did not change are reused by later GPU submissions instead of being copied again. | dev |
| `RPS5E_STREAM_MEMO` | on by default; `0` turns it off; `RPS5E_SAFE=1`, `RPS5E_FAST_PATHS=0` and A/B slow periods also turn it off | A streamed buffer binding reuses its earlier copy; with `0` every streamed binding copies its bytes again. | dev |
| `RPS5E_STREAM_PAGES` | on by default; `0` or `off` turns it off; `RPS5E_SAFE=1`, `RPS5E_FAST_PATHS=0` and A/B slow periods also turn it off | Stream copies stay valid while their own pages are unchanged; with `0` any change anywhere in the buffer cache invalidates them. | dev |
| `RPS5E_STREAM_TRIM` | on by default; `0` turns it off; `RPS5E_SAFE=1`, `RPS5E_FAST_PATHS=0` and A/B slow periods also turn it off | Stream copies cover only the bytes a shader can reach at constant offsets; with `0` they copy the whole bound range. | dev |
| `RPS5E_SUBGROUP_OPS` | on by default; `0` or `off` turns it off; `RPS5E_SAFE=1` also turns it off | Lane reads and lane shuffles with a fixed pattern are compiled to native subgroup broadcast and shuffle operations instead of generic exchanges. | dev |
| `RPS5E_SUBMIT_REQUEST` | on by default; `0` or `off` turns it off; `RPS5E_SAFE=1`, `RPS5E_FAST_PATHS=0` and A/B slow periods also turn it off | A CPU access waiting for a GPU copy that is still in the open command buffer asks the recording thread to submit right away. With `0` the wait lasts until the current batch is submitted anyway. | dev |
| `RPS5E_TABLE_CARRY` | `1` turns it on; off by default; inactive with `RPS5E_SAFE=1`, `RPS5E_FAST_PATHS=0` and in A/B slow periods | Flattened resource-table copies are reused across GPU submissions and uploaded again only when their words change or their staging memory is recycled. | dev |
| `RPS5E_TARGET_GENERATIONS` | on by default; `0` or `off` turns it off; `RPS5E_SAFE=1` also turns it off | The render-target memo checks each bound image for changes instead of being invalidated by any guest memory write anywhere. (Its doc comment speaks of `=1`, but the code treats it as on unless set to `0`.) | dev |
| `RPS5E_THROUGH_INDEX` | on by default; `0` or `off` turns it off; `RPS5E_SAFE=1` and slow periods of A/B scope `release` also turn it off | Lookups of buffer pages being copied back from the GPU use an address index instead of scanning every entry; results are the same. | dev |
| `RPS5E_UNLOCKED_COMPILE_WAIT` | on by default; `0` or `off` turns it off | A draw or dispatch waiting for a shader still compiling gives up the GPU back-end lock after 2 ms and retries, so CPU write faults, read-backs and flips are not stalled. With `0` it waits while holding the lock. | dev |
| `RPS5E_UPLOAD_PROLOGUE` | on by default; `0` turns it off; `RPS5E_SAFE=1`, `RPS5E_FAST_PATHS=0` and A/B slow periods also turn it off | Buffer uploads are moved ahead of the submission's work so rendering scopes stay open; with `0` every upload is recorded in order and ends the open rendering scope. | dev |
| `RPS5E_VIEW_MEMO` | on by default; `0` or `off` turns it off; `RPS5E_SAFE=1` also turns it off; also in the in-game GPU menu | Texture bindings reuse their earlier image lookup instead of looking the image up again on every draw. | dev |
| `RPS5E_WATCH_INDEX` | on by default; `0` or `off` turns it off; `RPS5E_SAFE=1` and slow periods of A/B scope `release` also turn it off | A CPU write fault or GPU buffer write finds the cached images it touches through a 1 MiB-granular index instead of scanning every image (which made write faults slow with thousands of images). | dev |
| `RPS5E_WAVE64_SPLIT` | on by default; `0` or `off` turns it off | On GPUs with 32-wide subgroups, wave64 compute shaders that qualify and use at least `RPS5E_WAVE64_SPLIT_VGPRS` registers run one guest lane per host invocation across two subgroups. With `0` all run paired (two guest lanes per invocation). Part of the shader cache key. | dev |
| `RPS5E_WAVE64_SPLIT_VGPRS` | integer; default 16; `0` splits every program that qualifies | The guest vector-register count from which wave64 compute shaders run split (see `RPS5E_WAVE64_SPLIT`). Part of the shader cache key. | dev |
| `RPS5E_WIDE_OFFSETS` | `1` turns it on; off by default; inactive with `RPS5E_SAFE=1`, `RPS5E_FAST_PATHS=0` and in A/B slow periods | Each buffer's byte offset gets a full 32-bit slot in the shader data instead of an 8-bit one. By itself it only widens the slot; it is needed by `RPS5E_RING_BINDING=1`. | dev |
| `RPS5E_WRITE_EXTENT` | on by default; `0` or `off` turns it off; `RPS5E_SAFE=1`, `RPS5E_FAST_PATHS=0` and A/B slow periods also turn it off | A compute dispatch marks as GPU-written only the buffer bytes its stores can reach (from its thread and group bounds), so the rest stays readable by the CPU without a download. With `0` the whole descriptor range is marked. | dev |
| `RPS5E_WRITTEN_WORDS` | on by default; `0` or `off` turns it off; `RPS5E_SAFE=1` and slow periods of A/B scope `release` also turn it off | A CPU read of a GPU-written descriptor word waits only for the copy-back that delivered that word; with `0` it waits for the newest copy-back of the whole page. | dev |

## Debugging and capture

Frame dumps, frame traces, GPU captures and switches that change emulation behaviour for debugging.

| Variable | Values and default | What it does | For |
|---|---|---|---|
| `RPS5E_SAFE` | off by default; `1` turns it on; also an in-game menu switch | Safe mode: turns off every GPU fast path that could hand the GPU stale data or skip work (memos, recipes, async pipelines, GPU indirect draws, on-access image read-back and more). Slow; used to check whether a glitch comes from an optimisation. | user |
| `PS5_AGC_SUSPEND_POINTS_IN_FLIGHT` | integer; default 1 | How many frame boundaries (`sceAgcSuspendPoint`) the game may queue ahead of the GPU thread before it is blocked; `0` makes every boundary wait for the GPU. | dev |
| `PS5_DISABLE_LLE_LIBC` | `1`: on; default off | Never binds the game's libc imports to the native `libc.prx`; every libc call goes to the emulator's own implementation. | dev |
| `PS5_LLE_LIBC_ALL` | `1`: on; default off | Binds every libc export the game imports to the native `libc.prx` instead of only the allocator, `sprintf`, `memmove` and `memcmp`. | dev |
| `PS5_LLE_LIBC_SAFE_ONLY` | `off`, `false` or `none`: same as `PS5_DISABLE_LLE_LIBC=1`; other values: no effect | Alternative spelling for turning native libc binding off. | dev |
| `RPS5E_ALLOW_PTRACE` | `1`: on; default off (Linux) | Lets any process of the same user attach to the emulator (`gdb -p`, `eu-stack`) under Yama `ptrace_scope=1`, for profiling and debugging. | dev |
| `RPS5E_BDA_PROBE` | comma-separated hex shader addresses; default none | With `RPS5E_BDA=1`: logs what the shader stages at these program addresses read through guest memory. | dev |
| `RPS5E_CAPTURE_AT` | seconds (decimal); default 0 | With `RPS5E_CAPTURE_FRAMES`: when (seconds since process start) the frame capture begins. | dev |
| `RPS5E_CAPTURE_COUNT` | number of frames (>= 1); default 1 | With `RPS5E_CAPTURE_FRAMES`: how many frames of GPU work to capture. | dev |
| `RPS5E_CAPTURE_FRAMES` | directory path; default off | Captures the GPU work of some frames into a `.ps5cap` file in this directory (once per run) for `rps5e-cli replay`. | dev |
| `RPS5E_DEBUG_DEPTH_EQUAL` | `ge` or `le`; default off | Diagnostic only: draws whose depth test is EQUAL test GREATER_OR_EQUAL (`ge`) or LESS_OR_EQUAL (`le`) instead, to find depth pre-pass mismatches. Changes the picture; not a fix. | dev |
| `RPS5E_DUMP_EVERY` | number of frames (>= 1); default 60 (also `--dump-every`) | With `RPS5E_DUMP_FRAMES`: dump every n-th presented frame. | dev |
| `RPS5E_DUMP_FRAMES` | directory path; default off (also `--dump-frames`) | Writes presented game frames as `frame_NNNNN_gpu.ppm` (GPU result) and `frame_NNNNN_cpu.ppm` (guest memory as the CPU sees it) into this directory, without menus or overlays. | dev |
| `RPS5E_DUMP_SCALE` | integer 1 to 16; default 1 | With `RPS5E_DUMP_FRAMES`: shrinks dumped frames by this factor (box average; a 4K frame at 4 becomes 960x540). | dev |
| `RPS5E_DUMP_SHADERS` | directory; unset: off | Writes every shader the game creates as `<hash>.bin` (code) plus `<hash>.json` (stage, registers, resource layout); fused shader pairs also get `fused-<a>-<b>.json`. Read them with `ps5shader-dis`. | dev |
| `RPS5E_DUMP_WINDOW` | `<from>-<to>` seconds; default: whole run | With `RPS5E_DUMP_FRAMES`: dumps only frames presented between these seconds of process time (for consecutive-frame glitch scans). | dev |
| `RPS5E_DUMP_WINDOW_EVERY` | number of presents (> 0); default 30 | With `RPS5E_DUMP_WINDOW_FRAMES`: dump every n-th present (every third while the shader compile panel is shown). | dev |
| `RPS5E_DUMP_WINDOW_FRAMES` | directory path; default off | Writes the window as shown (game frame plus menu, compile overlay and hints) at half size as `window_<n>.ppm` into this directory. | dev |
| `RPS5E_FRAME_TRACE` | comma-separated frame numbers, `a-b` ranges, or `@<seconds>` (first frame presented that long after start); unset: off | Logs every draw and dispatch of the chosen frames with outcome, shaders, targets and bound images. | dev |
| `RPS5E_FRAME_TRACE_ADDR` | comma-separated hex addresses; default none | With a frame trace, sampled images at these addresses are dumped too (as `tex`), whatever their size. | dev |
| `RPS5E_FRAME_TRACE_ARRAYS` | `1`: on; default off | With a frame trace, sampled array and cube images are dumped too (as `tex`). | dev |
| `RPS5E_FRAME_TRACE_BEFORE` | any value (set): on; default off | With a frame trace active, also dumps each pass's colour targets as they are before the pass runs (`pre<slot>`). | dev |
| `RPS5E_FRAME_TRACE_COUNT` | integer ≥ 1; default 1 | Number of consecutive frames traced from each `@<seconds>` entry of `RPS5E_FRAME_TRACE`. | dev |
| `RPS5E_FRAME_TRACE_DIR` | directory path; default none (log only) | With a frame trace, the images each pass writes are read back and saved as `<dir>/NNNN_<what>_<addr>.ppm` (layer 0, at most 960 px wide unless `RPS5E_FRAME_TRACE_WIDTH`, floats clamped to 0..1), with per-channel min/max/mean in the log. | dev |
| `RPS5E_FRAME_TRACE_FROM` | integer pass number; default 0 | With a frame trace, no image dumps before traced pass `n` (a whole frame of dumps takes minutes). | dev |
| `RPS5E_FRAME_TRACE_LAST` | `1`: on; default off | With a frame trace, an image used by consecutive passes is dumped once, after the last of them. | dev |
| `RPS5E_FRAME_TRACE_MRT` | integer; default 0 (all passes) | With a frame trace, only passes writing at least `n` colour targets dump images (e.g. 4 for G-buffer passes). | dev |
| `RPS5E_FRAME_TRACE_RAW` | `1`: on; default off | With a frame trace, also writes unclamped values of non-8-bit images as `.pfm` files (for checking HDR targets). | dev |
| `RPS5E_FRAME_TRACE_SKIP` | comma-separated dump kinds (e.g. `z,tex`); default none | With a frame trace, image dumps of these kinds are left out. | dev |
| `RPS5E_FRAME_TRACE_WIDTH` | pixels; default 960 | With `RPS5E_FRAME_TRACE_DIR`: image dumps are downscaled to at most this width (e.g. 3840 for full 4K dumps). | dev |
| `RPS5E_FULL_CORE` | `1`: on; default off (Linux) | Keeps the kernel's full core-dump filter; by default the ~14 GiB guest memory is left out of core dumps so a crash dump does not take minutes. | dev |
| `RPS5E_GPU_CHECKPOINTS` | `1`: on; default off | Enables NVIDIA diagnostic checkpoints: every dispatch leaves its shader address as a checkpoint, and a lost GPU device reports the last ones reached. | dev |
| `RPS5E_GPU_PLAN` | `1`: on; default off | Only when the shader compiler is disabled (`RPS5E_SHADER_COMPILER=0`): the GPU front-end still decodes and plans every draw for diagnostics, without compiling or drawing. | dev |
| `RPS5E_GPU_SKIP` | comma-separated hex shader code addresses; default none | Draws and dispatches using a shader at one of these addresses are not executed (for bisecting a misbehaving shader). | dev |
| `RPS5E_GPU_SUBMIT_EACH` | `1`: on; `2`: also waits for each and logs items over 20 ms; default 0 (off) | Submits every draw and dispatch to the GPU separately, so a GPU hang names exactly one item. Very slow. | dev |
| `RPS5E_IMPORT_TRAP` | `1`: on; default off | Routes system-library calls through INT3 traps and a signal handler instead of direct jumps. Much slower; for debugging the call path. | dev |
| `RPS5E_MEM_HISTORY` | opt-in: `1` turns it on; default off | Keeps a ring of recent GPU-side events on guest memory pages (uploads, GPU writes, copies, CPU faults, mapping changes) so a fatal guest crash report names the last writers of the memory involved. | dev |
| `RPS5E_NATIVE_MEMOPS` | `1`: on; default off | Binds `memset` and `memcpy` to the game's native libc instead of the emulator's versions (for A/B comparison). | dev |
| `RPS5E_REPLAY_DRAW_STATS` | off by default; set to any value to turn it on; read only by `rps5e-cli replay` | During a capture replay, collects and prints statistics of a frame's draws (groups of identical draws, repeated runs, user-data differences). Temporary diagnostic. | dev |
| `RPS5E_SERIALIZE` | off by default; `barrier` or `wait` | Diagnostic: `barrier` puts a full GPU barrier before every draw and dispatch; `wait` also submits and waits for the GPU after each one (very slow). Helps tell whether a glitch is a missing dependency between passes or between CPU and GPU. | dev |
| `RPS5E_SHADER_FAULTS` | `1` turns it on; off by default | Shaders count reads through device addresses of guest pages that have no page-table entry; the count appears as `unmapped-reads` in the `[gpu]` profile line. Costs an atomic and a branch per access. | dev |
| `RPS5E_SLICE_SHADOW` | any non-empty value other than `0` turns it on; off by default | Measurement only: classifies the GPU work of each frame and predicts frame times for several slice sizes and thread counts, without changing behaviour. | dev |
| `RPS5E_SLICE_VERIFY` | `1` turns it on; off by default | Slice mode: each slice's register snapshot and draw decisions are compared with what the serial path would produce; differences are logged. | dev |
| `RPS5E_SPIRV_DUMP` | directory path; unset by default | Writes every compiled game shader as `<dir>/<stage>_<address>_<key>.spv` (for spirv-val and disassembly). The menu and overlay shader tests also write their SPIR-V there. | dev |
| `RPS5E_WATCH_INDEX_VERIFY` | `1` turns it on; off by default | Every image-index lookup is also checked against a full scan, and lookups that disagree are counted. | dev |

## Set by the launcher and the command line

The launcher and `rps5e-cli run` set these for the game process; they are listed so that a copied command line makes sense. Setting them by hand is rarely useful.

| Variable | Values and default | What it does | For |
|---|---|---|---|
| `RPS5E_FROM_LAUNCHER` | any value | Set by the launcher for the games it starts: the in-game menu's exit button then just closes the game ("Exit to the launcher") instead of opening a new launcher. | dev |
| `RPS5E_GAME_KEY` | text | Set by the launcher: the game's key in its library, used by the in-game menu to save per-game settings for the right entry. Default: the game folder path. | dev |
| `RPS5E_GAME_TITLE` | text | Set by the launcher: the game's title shown in the in-game menu. Default: the title ID or folder name. | dev |
| `RPS5E_INPUT_OVERRIDE` | TOML text | Input settings applied on top of the settings file and game profile. `rps5e-cli run` builds it from `--input-mode`, `--bind` and `--controller-color`, the launcher from the device mode; it can also be set by hand (same format as `rps5e-input.toml`). | dev |

## Build time

| Variable | Values and default | What it does | For |
|---|---|---|---|
| `RPS5E_NO_FFMPEG` | set (any value) at build time | Builds without FFmpeg: in-game movies are skipped. Read by the build script, not at run time. | user |

## Testing and benchmarks

Read only by unit tests and ignored benchmark tests (`cargo test ... -- --ignored`); they have no effect on a game run.

| Variable | Values and default | What it does | For |
|---|---|---|---|
| `RPS5E_AT9_OUT` | file path | Output file of the ignored ATRAC9 reference-dump unit test (with `RPS5E_AT9_RAW`). Test only. | dev |
| `RPS5E_AT9_RAW` | file path | Input ATRAC9 stream of the ignored ATRAC9 reference-dump unit test. Test only. | dev |
| `RPS5E_AUDIO_TEST_DEVICE` | any value | Lets unit tests open a real audio device (they use the null back-end otherwise); needed for the ignored `device_tone` test. Test only. | dev |
| `RPS5E_BINDLESS_BENCH_KEYS` | number of textures | Turns on the ignored `bindless_large_table_first_bind_time` GPU benchmark with this many table entries. Test only. | dev |
| `RPS5E_BINDLESS_BENCH_STRIDE` | bytes; default 4096 (min 256) | Distance between textures in the bindless benchmark. Test only. | dev |
| `RPS5E_CAPTURE_BENCH_OUT` | file path | Turns on the ignored synthetic capture/replay benchmark test and names its capture file. Test only. | dev |
| `RPS5E_CAPTURE_SYNC_OUT` | file path | Turns on the ignored dispatch-synchronisation capture benchmark and names its capture file. Test only. | dev |
| `RPS5E_CAPTURE_TEST_OUT` | file path | Keeps the capture written by the capture/replay unit test at this path. Test only. | dev |
| `RPS5E_DRAW_BENCH_FRAMES` | number of frames | Turns on the ignored `draw_path_bench` GPU benchmark for this many frames. Test only. | dev |
| `RPS5E_DUMP_SPIRV` | directory path; default off | Read only by a unit test: writes the fixed blit shaders as SPIR-V files into this directory for external validation. Has no effect on a game run. | dev |
| `RPS5E_GUI_RENDER_DIR` | directory | Output folder of the ignored launcher `render_*` tests, which draw launcher pages, the in-game menu and the controls overlay to PNG without a window. Test only. | dev |
| `RPS5E_GUI_RENDER_SIZE` | `<w>x<h>` in points; default 1280x800 | Window size for the `render_library_scan` test. Test only. | dev |
| `RPS5E_GUI_SCAN_DIR` | directory | Games folder scanned by the `render_library_scan` and `print_scan` launcher tests (else `GAMES_DIR`). Test only. | dev |
| `RPS5E_LOCK_BENCH` | any value | Turns on the ignored `host_lock_contention_bench` test. Test only. | dev |
| `RPS5E_LOCK_BENCH_ITERS` | number; default 200000 | Iterations of the lock benchmark. Test only. | dev |
| `RPS5E_SLICE_BENCH` | any value | Turns on the ignored `slice_record_bench` GPU benchmark. Test only. | dev |
| `RPS5E_SLICE_BENCH_AB` | number of periods (>= 2) | A/B periods in the slice benchmark. Test only. | dev |
| `RPS5E_SLICE_BENCH_AFFINITY` | CPU list | CPU affinity of the slice benchmark's threads. Test only. | dev |
| `RPS5E_SLICE_BENCH_CONST` | bytes per draw, 256 to 65536; default 256 | Object constants per draw in the slice benchmark. Test only. | dev |
| `RPS5E_SLICE_BENCH_K` | comma-separated thread counts; default `1,2,4` | Slice thread counts the benchmark runs. Test only. | dev |
| `RPS5E_SLICE_BENCH_LOOPS` | number; default 30 | Loops of the slice benchmark. Test only. | dev |
| `RPS5E_SLICE_BENCH_MATERIALS` | number; default 256 | Materials in the slice benchmark. Test only. | dev |
| `RPS5E_SLICE_BENCH_PROGRAMS` | number; default 16 | Distinct pixel programs in the slice benchmark. Test only. | dev |
| `RPS5E_SLICE_BENCH_SUMMARY` | any value | Prints a summary line from the slice benchmark. Test only. | dev |
| `RPS5E_SLICE_BENCH_WRITES` | `1` | Adds guest memory writes to the slice benchmark. Test only. | dev |
| `RPS5E_SLICE_THREADS` | integer ≥ 1; default the `--slices` value | `rps5e-cli replay` with `--slices` 2 or more: the number of threads in the slice pool, when it should differ from the slice count (e.g. one thread to measure without contention). | dev |
| `RPS5E_SMOKE_SECS` | integer seconds; default 2 | Only read by the ignored `window_smoke_gradient` test: how long it shows its test gradient in a real window. | dev |
| `RPS5E_TEST_BISECT` | any value | Adds per-switch variants to an emitter comparison unit test. Test only. | dev |
| `RPS5E_TEST_VAR` | - | A variable name used only inside the `getenv`/`setenv` unit test; never read from the host environment. | dev |
| `RPS5E_TRANSLATE_BENCH` | path of a `.prewarm` list | Turns on the ignored `translate_list_bench` shader translation benchmark. `rps5e-cli shader-cache translate` does the same without a test build. Test only. | dev |
| `RPS5E_TRANSLATE_BENCH_KEYS` | comma-separated hex module keys | Limits the translation benchmark to these modules. Test only. | dev |
| `RPS5E_WAVE64_BENCH` | any value | Turns on the ignored wave64 split GPU benchmark. Test only. | dev |
## Standard variables

Variables of the system or of common tools that the emulator reads.

| Variable | What the emulator uses it for |
|---|---|
| `RUST_LOG` | Log filter (`error`, `warn`, `info`, `debug`, `trace`, or per module such as `info,ps5gpu=debug`). Without it only errors are printed. The launcher sets it from its log level setting (default `info`). |
| `HOME` | Base of the default folders when the `XDG_*` variables are unset (`~/.local/share`, `~/.config`, `~/.cache`); also searched for fonts (`~/.local/share/fonts`, `~/.fonts`) and used to expand `~` in launcher paths. |
| `XDG_DATA_HOME` | Default base of the data folder: `$XDG_DATA_HOME/rps5e`. |
| `XDG_CONFIG_HOME` | Default base of the settings folder: `$XDG_CONFIG_HOME/rps5e`. |
| `XDG_CACHE_HOME` | Base of the shader and pipeline caches: `$XDG_CACHE_HOME/rps5e` (not used on Windows, where caches go to `%LOCALAPPDATA%\rps5e`). |
| `XDG_RUNTIME_DIR` | Folder of the lock file `rps5e-run.lock` that lets only one emulator run at a time (a second `run`, `replay`, `precompile` or shader-cache import waits for the first). Falls back to the system temporary folder. |
| `DISPLAY`, `WAYLAND_DISPLAY` | When neither is set on Linux, the game runs headless (no window). |
| `APPDATA` (Windows) | Settings folder `%APPDATA%\rps5e`. |
| `LOCALAPPDATA` (Windows) | Data and cache folder `%LOCALAPPDATA%\rps5e`. |
| `USERPROFILE` (Windows) | Expands `~` in launcher paths when `HOME` is unset. |
| `USERNAME` (Windows) | The host user name reported to games that ask for the login name. |
| `GAMES_DIR` | A games folder the launcher shows as a library folder by default; also read by `make test`, `progress/progress.py` and tests that need a game (they look up a title ID under it and skip without it). |

Cargo's own variables (`CARGO_*`, `OUT_DIR`, `PROFILE`, `RUSTC`) and `WINDRES` / `RC` (the Windows resource compiler that embeds the icon) are read by build scripts only.

## Script variables

Variables of the scripts in `scripts/`.

| Script | Variable | Meaning |
|---|---|---|
| `run-game.sh` | `DATA_DIR`, `CONFIG_DIR` | Passed as `--data-dir` / `--config-dir` (default: the emulator's folders). |
| `run-game.sh` | `RUST_LOG` | Log filter (default `info`). |
| `run-game.sh`, `package.sh` | `BUILD` | `0` skips the `cargo build` step (default `1`). |
| `input-test.sh` | `BIN` | The `rps5e-cli` binary (default `target/release/rps5e-cli`). |
| `input-test.sh` | `DATA_DIR` | Data folder a recording starts from (default `local/data/<TITLE_ID>`); its saves are copied next to the recording, and every replay starts from that copy. |
| `input-test.sh` | `TAIL_SECONDS` | Seconds the replay keeps running after the last recorded input (default 15). |
| `input-test.sh` | `DUMP_SCALE`, `DUMP_EVERY` | Frame dump scale and interval of a replay (defaults 4 and 300). |
| `input-test.sh` | `RPS5E_COMPILE_WAIT_MS`, `RUST_LOG` | Default `exact` and `info` for recordings and replays. |
| `package.sh` | `CARGO` | The cargo command (default `cargo`; for example a wrapper that sets up the Windows cross toolchain). |
| `package.sh` | `OUT` | Output folder of the archives (default `dist/`). |
| `progress/progress.py` | `GAMES_DIR` | Games folder for the import statistics (else `local/games.conf`). |
| `progress/progress.py` | `PS5_FW_LIBS` | Folder of firmware library listings for the statistics (default: a copy under `local/`). |

## Test-only variables without the `RPS5E_` prefix

These are read only by unit tests and have no effect on the emulator: `PS5SHADER_CORPUS` (a shader dump folder for the corpus tests), `PS5SHADER_DUMP_SPV` (write the test modules' SPIR-V to a folder), `PS5SHADER_EMIT_BEFORE`, `PS5SHADER_IR_DUMP`, `PS5SHADER_IR_GREP`, `PS5SHADER_IR_OPS`, `PS5SHADER_NOVAL` (skip `spirv-val`), `PS5SHADER_SLOWEST`, `BDA_TEST_OFF`, `EXEC_REF_JOB`, `EXEC_REF_VERBOSE`, `REF_DEVICE_CASE`, `REF_DEVICE_EXPECTED`, `FIBER_TEST_TRACE`.
