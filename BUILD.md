# Building RustPS5Emu

RustPS5Emu builds with stable Rust on Linux x86-64 (the main platform) and for Windows x86-64 (`x86_64-pc-windows-gnu`, natively or cross-compiled from Linux).

## Host requirements

**CPU.** Game code runs natively, so the host needs what the PS5's Zen 2 CPU offers: x86-64-v3 (AVX2, FMA, F16C, BMI1/2, LZCNT, MOVBE, AES, PCLMULQDQ). That is Intel Haswell or newer, AMD Zen or newer. `rps5e-cli run` checks this at start; `RPS5E_IGNORE_CPU_FEATURES=1` runs anyway. AMD instructions Intel lacks (SSE4a `EXTRQ`/`INSERTQ`, `MONITORX`/`MWAITX`, ...) are emulated or patched at load.

**GPU.** A Vulkan 1.3 driver with `VK_KHR_push_descriptor` and `VK_KHR_fragment_shader_barycentric` (NVIDIA Turing and newer, AMD RDNA2 and newer with RADV, Intel Arc). Optional features (shader float64, mesh shaders, 64-bit image atomics, ...) are reported at start with what is lost without them.

**Memory.** Guest memory is a 13.5 GiB shared mapping; only touched pages use RAM. 16 GB of RAM is the practical minimum, 32 GB is comfortable.

## Linux

Packages (names differ per distribution):

- Rust (stable, via rustup or the distribution), a C compiler, `pkg-config`;
- Vulkan loader and driver (`libvulkan.so.1`);
- OpenSSL development files (`libssl-dev` / `openssl-devel`);
- ALSA development files (`libasound2-dev` / `alsa-lib-devel`);
- optional: FFmpeg development files (`libavformat`, `libavcodec`, `libavutil`, `libswscale`, `libswresample`) for in-game movies; without them movies are skipped;
- optional at run time: SDL3 (`libSDL3.so`) for controllers; keyboard and mouse work without it.

```sh
make build            # = cargo build --release
make test             # unit tests (make test-gpu: Vulkan device tests)
make install          # rps5e, rps5e-cli, desktop entry and icons into ~/.local (PREFIX=...)
```

Binaries: `target/release/rps5e` (launcher) and `target/release/rps5e-cli` (command line; the same program). The release profile keeps debug info for crash reports; `cargo build --profile dist` (what `scripts/package.sh` uses) builds stripped binaries into `target/dist/`.

## Windows

The Windows build runs natively on Windows 10 1803+ / Windows 11 and under Wine/Proton. Requirements to run:

- an x86-64-v3 CPU and a Vulkan 1.3 driver (`vulkan-1.dll` comes with it);
- optional: `SDL3.dll` next to the executables for controllers;
- enough commit space: guest memory is a 13.5 GiB pagefile-backed section; Windows charges it up front, so RAM plus page file must cover it (the default system-managed page file grows as needed).

Data go to `%LOCALAPPDATA%\rps5e` (saves, per-title data), settings to `%APPDATA%\rps5e`; `RPS5E_DATA_DIR` / `RPS5E_CONFIG_DIR` override them.

### Native build

1. Install Rust with rustup and the GNU toolchain: `rustup toolchain install stable-x86_64-pc-windows-gnu` and `rustup default stable-x86_64-pc-windows-gnu`.
2. Install MSYS2 and, in the MSYS2 UCRT64 shell, `pacman -S mingw-w64-ucrt-x86_64-gcc perl make` (OpenSSL is built from source and needs Perl and make; `windres`, which embeds the icon, comes with the MinGW binutils).
3. Put `C:\msys64\ucrt64\bin` and `C:\msys64\usr\bin` on `PATH`.
4. In the repository: `cargo build --release`. Binaries: `target\release\rps5e.exe` (launcher) and `target\release\rps5e-cli.exe`.

`.cargo/config.toml` adds `-C target-cpu=x86-64-v3` for the Windows targets.

### Cross build from Linux

1. A MinGW-w64 cross compiler (`x86_64-w64-mingw32-gcc`, `x86_64-w64-mingw32-windres`), Perl and make.
2. `rustup target add x86_64-pc-windows-gnu`.
3. `cargo build --release --target x86_64-pc-windows-gnu`. Binaries land in `target/x86_64-pc-windows-gnu/release/`.

Some distribution toolchains (Gentoo's `mingw64-toolchain`) ship the unwinder inside `libgcc.a` and have no `libgcc_eh.a`, which Rust asks for. An empty archive is enough:

```sh
mkdir -p ~/mingw-shim && x86_64-w64-mingw32-ar rcs ~/mingw-shim/libgcc_eh.a
printf '#!/bin/sh\nexec x86_64-w64-mingw32-gcc -L%s "$@"\n' ~/mingw-shim > ~/mingw-shim/gcc && chmod +x ~/mingw-shim/gcc
export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=~/mingw-shim/gcc
```

Use a linker wrapper rather than `RUSTFLAGS`: `RUSTFLAGS` would replace the `target-cpu` flag of `.cargo/config.toml`.

Unit tests run under Wine:

```sh
CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER=wine \
  cargo test --release --target x86_64-pc-windows-gnu -p ps5vm -p ps5cpu -p ps5hle
```

### Known Windows limitations

- Movie playback (FFmpeg) is off.
- `EVFILT_VNODE` file watches register but never fire.
- Exceptions are dispatched on the guest stack, and neither Windows nor Wine skips the 128-byte System V red zone. At load, every guest function that keeps data below RSP (found through the module's `.eh_frame`) is moved to a copy that runs with RSP lowered by 128 bytes, so a write-tracking fault or an emulated instruction there no longer overwrites that data; the `[redzone]` log line counts them. A function the rewrite cannot reproduce exactly (it addresses the thread pointer, or a branch lands inside its first 5 bytes) is left as it is and stays exposed; the same line counts those too.
- Guest code that addresses `fs:` with RIP-relative or 32-bit address forms is not rewritten (logged as `[tls] cannot rewrite`).

## Release packages

```sh
scripts/package.sh linux      # dist/rps5e-<version>-linux-x86_64.tar.gz
scripts/package.sh windows    # dist/rps5e-<version>-windows-x86_64.zip (cross build)
```

Both use the `dist` profile (release without debug info). `CARGO=<wrapper>` selects another cargo, for example one that sets up the Windows cross toolchain. The Linux tarball has a `share/` folder with the desktop entry and icons for `~/.local/share` or `/usr/share`.
