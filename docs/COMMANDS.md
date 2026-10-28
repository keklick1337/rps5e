# Commands

The workspace builds three programs (into `target/release/`, or `target/x86_64-pc-windows-gnu/release/` with `.exe` for Windows):

| Binary | What it is |
|---|---|
| `rps5e` | The launcher (game library). With a command it does the same as `rps5e-cli`. |
| `rps5e-cli` | The command line. Without a command it opens the launcher too. |
| `ps5shader-dis` | Shader disassembler and translator test tool, for shader dumps made with `RPS5E_DUMP_SHADERS`. A developer tool. |

`rps5e` and `rps5e-cli` are the same program under two names; only the name in the help text differs. `rps5e-cli --help` lists the commands, `rps5e-cli <command> --help` the options of one, and `--version` prints the version. Most `run` options have an environment variable; see [ENVIRONMENT.md](ENVIRONMENT.md).

Only one emulator process runs at a time: `run`, `replay`, `precompile`, `shader-cache import`, `shader-cache clear`, `shader-cache translate` and `shader-cache relocation` take a lock (`$XDG_RUNTIME_DIR/rps5e-run.lock`) and wait, with a message, while another one holds it.

## Command overview

| Command | What it does |
|---|---|
| (none) | Open the launcher. |
| `gui [--settings]` | Open the launcher; `--settings` opens its settings at start. |
| `run <game>` | Run a game. |
| `info-pkg <pkg>` | Show what a package contains. |
| `install-pkg <pkg>` | Extract a package into a game folder. |
| `precompile <game>` | Build a game's shaders and pipelines without running it. |
| `shader-cache <action>` | Export, import, inspect or clear a game's shader cache. |
| `replay <capture>` | Replay a GPU frame capture as a benchmark. |
| `info <eboot>` | Print the SELF/ELF headers and segments of an executable. |
| `self2elf <in> <out>` | Extract the ELF from a decrypted (fake) SELF. |
| `load <eboot>` | Load and link an executable without running it. |
| `imports <eboot>` | List every import with how it is bound. |
| `hle-exports` | List every system function the emulator implements. |
| `isa-scan <paths>...` | Count instructions that depend on the host CPU. |

Paths named `<game>` accept a game folder (holding `eboot.bin` and `sce_sys/`), the `eboot.bin` in it, `sce_sys/param.json`, or a title ID such as `PPSA01342` (for the commands that only need the title's caches).

## run

```
rps5e-cli run <PATH> [options]
```

Runs a game. `PATH` is a game folder, its `eboot.bin`, or a package (`.pkg`, `.ffpkg`, raw PFS image) that is mounted read-only as `/app0`. Only game packages run; an update package must be installed over the game with `install-pkg` first.

Before the game starts, the shaders and pipelines it used on earlier runs are compiled, with a progress panel (see `--no-prewarm-wait`). Logs go to the terminal; set `RUST_LOG=info` for more than errors.

| Option | Meaning | Default |
|---|---|---|
| `--passcode PASSCODE` | Passcode of a fake package, 32 ASCII characters. | all zeros |
| `--data-dir DIR` | Writable game data: saves, `/temp0`, `/download0`, system settings, one subfolder per title ID. Created if missing. | `~/.local/share/rps5e` (`%LOCALAPPDATA%\rps5e`) |
| `--config-dir DIR` | Settings folder with `rps5e-input.toml` and `profiles/`. Created if missing. | `~/.config/rps5e` (`%APPDATA%\rps5e`) |
| `--input-config FILE` | An input settings file to use instead of the one in the settings folder; it must exist. | |
| `--input-mode MODE` | Player 1's devices: `auto` (controller, keyboard and mouse together), `keyboard_mouse`, `keyboard` (keyboard only, its own bindings) or `controller`. | from the settings, else `auto` |
| `--bind ACTION=SOURCE[,SOURCE]` | Rebind an action; repeatable. See [Binding syntax](#binding-syntax). | |
| `--controller-color #RRGGBB` | Fixed controller light bar colour, overriding the game's. | the game's colour |
| `--no-input` | Ignore host keyboard, mouse and controllers; the game sees a neutral pad. | |
| `--pad-script SCRIPT` | Scripted button presses, e.g. `20:cross,26:down`. See [Pad scripts](#pad-scripts). | |
| `--record-input FILE` | Record every change of player 1's pad to FILE. | |
| `--replay-input FILE` | Replay a recording made with `--record-input`; host input is ignored. | |
| `--fullscreen` | Open the game window fullscreen. | windowed |
| `--hdr` | HDR10 output for games that show HDR frames, when the display supports it. | off |
| `--display-resolution RES` | Resolution of the emulated TV: `720p`, `1080p` or `2160p` (`4k`). Games that detect the display's resolution pick their output size from it; other games are always told about a 2160p display, as on the console. The window scales the picture either way. | `2160p` |
| `--perf-overlay` | Show the performance overlay from the start (F10 toggles it). | off |
| `--present-mode MODE` | `fifo` (vsync), `mailbox` (newest frame, no tearing) or `immediate` (may tear). | `mailbox` |
| `--gpu DEVICE` | Vulkan device index or part of its name. | the best device |
| `--headless` | No window. | |
| `--language ID` | Console language number, 0 to 30 (1 = English US). | 1 |
| `--mute` | No audio output. | |
| `--skip-notice-screen` | Turn the console's skip-notice-screen setting on. | off |
| `--refresh-rate HZ` | Display refresh rate the game sees, a whole number of Hz from 30 to 1000; 60 and 120 run at 59.94 / 119.88 Hz like the console. | 60 |
| `--game-patch FILE` | A cheat/patch file (mods JSON) applied at load. | |
| `--no-prewarm-wait` | Start the game at once and compile its known shaders in the background (more stutter at first). | wait for them |
| `--compile-wait MS\|exact` | How long draws may wait per frame for shaders that are still compiling before they are skipped for that frame. `exact` always waits (no missing effects, but stutter); tests and replays use it. | 3 ms |
| `--dump-frames DIR` | Save presented frames as PPM images in DIR (created if missing). | |
| `--dump-every N` | With `--dump-frames`: save every Nth frame. | 60 |

Examples:

```sh
rps5e-cli run ~/games/PPSA10112
rps5e-cli run ~/games/PPSA10112 --input-mode keyboard --fullscreen
rps5e-cli run game.pkg --passcode 00000000000000000000000000000000
rps5e-cli run ~/games/PPSA10112 --bind Cross=KEY:Space --bind "R2=MOUSE:Right,KEY:E"
rps5e-cli run ~/games/PPSA10112 --no-input --pad-script "25:cross,35:cross" \
    --data-dir /tmp/test-data --dump-frames frames --dump-every 60
```

### Binding syntax

`--bind ACTION=SOURCE[,SOURCE...]` replaces all bindings of `ACTION` with the listed sources (the other actions keep theirs). With `--input-mode keyboard` it changes the keyboard-only binding set, otherwise the keyboard-and-mouse set. Invalid names are reported at start.

| Source | Meaning |
|---|---|
| `KEY:<name>` | A key by physical position (US layout names): SDL names such as `Return`, `Space`, `Left Shift`, `Keypad 4`, `F11`, or key codes such as `Enter`, `ShiftLeft`, `Numpad4`, `KeyA`. |
| `MOUSE:<button>` | `Left`, `Middle`, `Right`, `X1`, `X2`. |
| `WHEEL:Up`, `WHEEL:Down` | Mouse wheel; pad buttons only (held 80 ms per notch). |

Actions: `Cross Circle Triangle Square L1 R1 L2 R2 L3 R3 Options Up Right Down Left TouchPad`, the stick directions `LeftStickLeft LeftStickRight LeftStickUp LeftStickDown RightStickLeft RightStickRight RightStickUp RightStickDown`, the emulator keys `ToggleMouse ReleaseMouse ToggleFullscreen TogglePerfOverlay ToggleMenu ShowControls Create Mute`, the controller level keys `SpeakerVolume VibrationIntensity TriggerEffectIntensity`, motion `TiltForward TiltBack TiltLeft TiltRight TurnLeft TurnRight Shake MouseMotion`, and touch pad `TouchLeft TouchRight TouchpadMode TapLeft TapCenter TapRight SwipeUp SwipeDown SwipeLeft SwipeRight TwoFingerTouch`. `rps5e-input.example.toml` documents them with their defaults.

### Pad scripts

`--pad-script` (or `RPS5E_PAD_SCRIPT`) presses pad inputs at fixed times, for unattended runs. It works with `--no-input`.

```
<seconds>:<input>[:<hold ms>],<seconds>:<input>[:<hold ms>],...
```

- `<seconds>`: time since the input system started (decimals allowed). Entries may be in any order.
- `<input>`: a button — `cross` (or `x`), `circle` (or `o`), `square`, `triangle`, `options` (or `start`), `up`, `down`, `left`, `right`, `l1`, `r1`, `l2`, `r2`, `touch` — or several joined with `+` (`l1+r1`), or a full stick push: `lup`, `ldown`, `lleft`, `lright` for the left stick, `rup`, `rdown`, `rleft`, `rright` for the right one.
- `<hold ms>`: how long the input is held; default 150 ms.

Invalid entries are skipped with a warning. Example: `12:cross,20:down,21:cross:400,30:l1+r1`.

### Input recordings

`--record-input FILE` writes one line per change of player 1's pad: `seconds buttons lx ly rx ry l2 r2` (seconds since start, buttons as hex, sticks and triggers 0-255; motion is not recorded). `--replay-input FILE` plays it back at the recorded times and ignores host input. `scripts/input-test.sh` wraps both and keeps a copy of the save data, so a replay starts from the same state; see README.md.

## gui

```
rps5e-cli gui [--settings]
rps5e                      # the same
```

Opens the launcher. `--settings` opens the settings at start. Games started from the launcher run as `rps5e run <game>` in a child process.

## info-pkg

```
rps5e-cli info-pkg <PATH> [--passcode PASSCODE] [--no-files]
```

Shows what a package or PFS image contains: title, version, content ID, container details and every file with its size. `--no-files` leaves out the file list.

## install-pkg

```
rps5e-cli install-pkg <PATH> [--into DIR] [--data-dir DIR] [--passcode PASSCODE]
```

Extracts a game package into a folder that then runs like any dump. An update package is extracted over the installed game.

| Option | Meaning | Default |
|---|---|---|
| `--into DIR` | Destination folder. | `<data dir>/games/<TITLE_ID>` |
| `--data-dir DIR` | Data folder holding `games/`. | `RPS5E_DATA_DIR`, else `~/.local/share/rps5e` |
| `--passcode PASSCODE` | Passcode of a fake package (32 characters). | all zeros |

```sh
rps5e-cli install-pkg game.pkg
rps5e-cli install-pkg update.pkg --into ~/.local/share/rps5e/games/PPSA01342
```

## precompile

```
rps5e-cli precompile <GAME> [--threads N] [--clear]
```

Builds a game's shader list without running it: every shader the game created on earlier runs (translated again when the shader translator changed since) and every pipeline, into the shader and pipeline caches. The list itself is made by playing the game. Prints a progress line every second.

| Option | Meaning | Default |
|---|---|---|
| `--threads N` | Compile threads. | every core |
| `--clear` | Clear the game's pipeline cache and cached shaders first: everything is translated and built again. | |

```sh
rps5e-cli precompile ~/games/PPSA01342
rps5e-cli precompile PPSA01342 --clear
```

## shader-cache

A game's portable shader cache: guest shader code and pipeline state, without SPIR-V, driver data or local paths, so it works on any GPU. The driver's own pipeline cache is not shared.

| Action | Usage | What it does |
|---|---|---|
| `export` | `shader-cache export <GAME> <FILE>` | Writes the game's shader list to a portable file. |
| `import` | `shader-cache import <FILE> [GAME]` | Merges a portable file into the game's list (existing entries are kept); the new shaders are built on the next start or `precompile`. Without `GAME`, the title the file names is used; a file of another game is refused. |
| `info` | `shader-cache info <GAME>` | What the game's list holds: programs, variants and what they differ in. |
| `clear` | `shader-cache clear <GAME> [--list]`, `shader-cache clear --all [--list]` | Deletes the game's driver pipeline cache and marks its shader list for translation, so the next start or `precompile` builds everything again. `--list` also forgets the list itself (shaders are then compiled during play, as on a first run). `--all` does this for every game and also deletes all translated-shader caches. |
| `translate` | `shader-cache translate <GAME or .prewarm file> [--hashes FILE] [--keys K1,K2] [--dump DIR] [--threads N] [--top N]` | Developer tool: translates every shader of the list again, bypassing the caches (nothing is written to them), and prints the time per translation phase and the slowest shaders (`--top`, default 20). `--hashes` writes `key stage address words hash ms` per shader, to compare two builds; `--keys` limits it to these shader keys (hex); `--dump` writes each shader's SPIR-V and a one-entry list to DIR. |
| `relocation` | `shader-cache relocation <GAME>` | Developer tool: translates the shaders that read their own code address at two addresses and checks that the results agree. Writes nothing. |

```sh
rps5e-cli shader-cache export ~/games/PPSA10112 sh-shortmessage.shadercache
rps5e-cli shader-cache import sh-shortmessage.shadercache
rps5e-cli shader-cache clear --all
```

## replay

```
rps5e-cli replay <CAPTURE> [--loops N] [--no-present] [--profile] [--dump-frames DIR] [--slices N]
```

Replays a GPU frame capture (a `.ps5cap` file written with `RPS5E_CAPTURE_FRAMES`) through the Vulkan back-end without a window, as a benchmark: wall time, CPU time per draw and dispatch, GPU time. A developer tool.

| Option | Meaning | Default |
|---|---|---|
| `--loops N` | Timed loops over the captured frames (the first one compiles the shaders). | 3 |
| `--no-present` | Do not copy the displayed images (flips only count frames). | |
| `--profile` | GPU timings and back-end phase times (sets `RPS5E_GPU_PROFILE=1` unless it is set). | |
| `--dump-frames DIR` | After the timed loops, replay once more and write every displayed image as `frame_NNNN.ppm`. | |
| `--slices N` | Slice threads for the replay (0: serial). | `RPS5E_SLICES`, else 0; at most 16 |

```sh
RPS5E_CAPTURE_FRAMES=cap RPS5E_CAPTURE_AT=60 RPS5E_CAPTURE_COUNT=3 rps5e-cli run ~/games/PPSA21567
rps5e-cli replay cap/capture-*.ps5cap --loops 5 --profile
```

## Inspection commands

These read files only and never run game code.

| Command | What it prints |
|---|---|
| `info <PATH>` | SELF/ELF header, SELF segments and program headers of an `eboot.bin` or module. For a retail-signed (encrypted) SELF it prints the readable headers and then reports that the payload is encrypted. |
| `self2elf <PATH> <OUT>` | Writes the ELF inside a fake SELF (`eboot.bin`, `.prx`, `.sprx`) with headers and segments at their file offsets, for disassemblers. |
| `load <PATH>` | Loads, maps and links the executable without running it, and prints its base, size, entry point and dynamic section summary. |
| `imports <PATH>` | One line per imported function: `nid name library binding`, where binding is `HLE` (implemented by the emulator), `LLE@<address>` (bound to the game's own module) or `STUB` (missing: returns an error). Totals go to stderr. |
| `hle-exports` | Every system function the emulator knows, as TSV `nid name library kind`: `fn` (implemented), `fallback` (used only when no module exports it) or `stub` (known, not implemented). No game needed. |
| `isa-scan <PATHS>...` | For each file, or `eboot.bin` plus `sce_module/*.prx` of each game folder: instruction count, instructions that only AMD CPUs have, ones that some Intel generations lack, and other host-dependent ones (CPUID, RDTSC, ...), with sample addresses. |

```sh
rps5e-cli imports ~/games/PPSA10112/eboot.bin | grep STUB
rps5e-cli self2elf ~/games/PPSA10112/eboot.bin eboot.elf
rps5e-cli isa-scan ~/games/PPSA10112
```

## Keys

### In the game window

Defaults; every one can be rebound (Settings > Input and controls, a game's input profile, the in-game Controls page, or `--bind` with the action name in brackets).

| Key | Action |
|---|---|
| F1 (or the controller's PS button) | In-game menu: controls, graphics, audio, system settings, exit (`ToggleMenu`). |
| F2 (hold) | Show the current bindings over the game (`ShowControls`). |
| F8 | Game sound off / on (`Mute`). |
| F9 | Capture the mouse for mouse-look (`ToggleMouse`). |
| F10 | Performance overlay (`TogglePerfOverlay`). |
| F11 | Fullscreen (`ToggleFullscreen`). |
| F12 | The Create button; saves a screenshot of the game window (`Create`). |
| Escape | Release the captured mouse or leave touch pad mode (`ReleaseMouse`), and press Options. |

In `controller` mode the keyboard keeps F1, F2, F8, F10, F11 and F12. The default pad keys (WASD, mouse, Space, ...), the motion keys on the numpad and the touch pad keys are listed in README.md and `rps5e-input.example.toml`.

### In the launcher

| Key | Action |
|---|---|
| Double click | Play the game. |
| Right click | The game's menu. |
| F5 | Scan the library folders again. |

## ps5shader-dis

A developer tool for shaders dumped with `RPS5E_DUMP_SHADERS=<dir>` (`<hash>.bin` code blobs with `<hash>.json` descriptions, and `fused-<a>-<b>.json` for fused shader pairs).

```
ps5shader-dis [--analyses] [--wave32] [--cfg] [--tree] [--ir] [--ir-raw] <blob.bin | fused-*.json>...
ps5shader-dis --stats <dir>
ps5shader-dis --stats-mods <dir>
ps5shader-dis --bench <dir>
ps5shader-dis --list-opcodes
```

| Option | Meaning |
|---|---|
| (files only) | Disassemble each blob or fused pair. A blob is decoded at its original code address when its `.json` sits next to it. |
| `--analyses` | Also print the analyses the translator uses (full-EXEC regions, wave64 half-mask writes, MOVREL M0 values); keeps the disassembly. |
| `--wave32` | Analyse and translate as a 32-lane wave (default 64). |
| `--cfg` | Print the control-flow graph of the translation instead of the disassembly. |
| `--tree` | Print the structured region tree. |
| `--ir` | Print the optimised intermediate code. |
| `--ir-raw` | Print the intermediate code without optimisation. |
| `--stats <dir>` | Decode every `.bin` in the folder (default `.`) and print opcode coverage. |
| `--stats-mods <dir>` | Per opcode: the modifiers, constants and memory flags seen. |
| `--bench <dir>` | Translate every blob and print the time per phase and pass, and the slowest shaders. |
| `--list-opcodes` | Every opcode name the decoder knows, one per line (used by `make stats`). |

`--cfg`, `--tree`, `--ir` and `--ir-raw` also print a statistics line. The exit status is non-zero when a file fails to decode.

```sh
RPS5E_DUMP_SHADERS=shaders rps5e-cli run ~/games/PPSA10112
ps5shader-dis shaders/0123456789abcdef.bin
ps5shader-dis --ir shaders/0123456789abcdef.bin
ps5shader-dis --bench shaders
```
