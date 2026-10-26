// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The emulator's command line, shared by the `rps5e` and `rps5e-cli`
//! binaries: with no arguments it opens the launcher (game library), with a
//! command it inspects or runs a PS5 SELF image (`run <game>`).

#[cfg(windows)]
#[allow(unused_imports)]
use ps5hle::host::c as libc;
use anyhow::{anyhow, Result};
use clap::{CommandFactory, FromArgMatches, Parser, Subcommand};
use ps5elf::{header, parse_self, ProgramType};

#[derive(Parser)]
#[command(name = "rps5e", version, about = "RustPS5Emu (RPS5E), a PS5 emulator (no command: open the launcher)")]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum ShaderCacheCmd {
    /// Write the game's shader cache to a portable file (guest shader code,
    /// translation inputs and pipeline state; no SPIR-V, driver data or
    /// paths of this machine).
    Export {
        /// The game folder (or its eboot.bin / param.json), or a title id.
        game: String,
        /// The file to write.
        file: String,
    },
    /// Merge a portable shader cache file into the game's list (records it
    /// already has are kept); the new shaders are translated and built on
    /// the next start or precompile.
    Import {
        /// The file to read.
        file: String,
        /// The game it is for [default: the title the file names; it must
        /// match].
        game: Option<String>,
    },
    /// What the game's shader list holds: programs, permutations and what
    /// they differ in.
    Info {
        /// The game folder (or its eboot.bin / param.json), or a title id.
        game: String,
    },
    /// Translate the game's programs that read their own address as
    /// relocatable modules and check each against a translation at
    /// another address (nothing is written).
    Relocation {
        /// The game folder (or its eboot.bin / param.json), or a title id.
        game: String,
    },
    /// Translate every shader of the game's list again, past every cache,
    /// timing each and hashing its SPIR-V (nothing of the game's caches is
    /// written): the translator's speed, its slowest programs, and with
    /// `--hashes` a file to compare two builds' output.
    Translate {
        /// The game folder (or its eboot.bin / param.json), a title id, or
        /// a `.prewarm` list file.
        game: String,
        /// Write `key stage address words hash ms` per module to FILE.
        #[arg(long, value_name = "FILE")]
        hashes: Option<String>,
        /// Only these module keys (hex, comma separated).
        #[arg(long, value_name = "KEYS", value_delimiter = ',')]
        keys: Vec<String>,
        /// Write each module's SPIR-V and a list holding only it to DIR.
        #[arg(long, value_name = "DIR")]
        dump: Option<String>,
        /// Translation threads [default: every core].
        #[arg(long, value_name = "N")]
        threads: Option<usize>,
        /// How many of the slowest modules to show.
        #[arg(long, value_name = "N", default_value_t = 20)]
        top: usize,
    },
    /// Reset compiled shaders: the game's driver pipeline cache is deleted
    /// and its shader list marked for translation, so the next start or
    /// precompile builds everything again. `--all` clears every game and
    /// the translated-module caches (also those of older builds).
    Clear {
        /// The game folder (or its eboot.bin / param.json), or a title id.
        #[arg(required_unless_present = "all")]
        game: Option<String>,
        /// Every game and every translated-module cache.
        #[arg(long, conflicts_with = "game")]
        all: bool,
        /// Also forget which shaders the game uses (its shader list): they
        /// are then compiled while playing again, as on a first run.
        #[arg(long)]
        list: bool,
    },
}

#[derive(Subcommand)]
enum Cmd {
    /// Print SELF/ELF header + segment table.
    Info {
        /// Path to eboot.bin / SELF.
        path: String,
    },
    /// Write the ELF inside a decrypted (fake) SELF, for disassemblers such
    /// as Ghidra: headers and segments at their file offsets.
    Self2elf {
        /// eboot.bin / .prx / .sprx (fake SELF or ELF).
        path: String,
        /// Output ELF file.
        out: String,
    },
    /// Count guest instructions whose presence or result depends on the
    /// host CPU (AMD-only extensions, extensions some Intel generations
    /// lack, CPUID/RDTSC/RSQRT...): executable segments of every given
    /// eboot.bin / .prx, or of eboot.bin and sce_module/*.prx in a game folder.
    IsaScan {
        /// Files or game folders.
        paths: Vec<String>,
    },
    /// Parse & map the image into a virtual address space (no execution).
    Load {
        /// Path to eboot.bin / SELF.
        path: String,
    },
    /// List every imported NID with its name and binding (HLE / LLE / stub).
    Imports {
        /// Path to eboot.bin / SELF.
        path: String,
    },
    /// List every HLE export (no game needed): NID, name, library and
    /// whether it is implemented (`fn`), a fallback for a missing module
    /// export (`fallback`) or a catalog stub (`stub`). Used by
    /// `progress/progress.py`.
    HleExports,
    /// Load the image, then attempt to execute it from its entry point.
    Run {
        /// Path to eboot.bin / SELF, the game folder that contains it, or a
        /// fake package / PFS image (.pkg, .ffpkg), mounted read-only as /app0.
        path: String,
        /// Passcode of a fake package (32 characters; default all zeros).
        #[arg(long, value_name = "PASSCODE")]
        passcode: Option<String>,
        #[command(flatten)]
        opts: RunOptions,
    },
    /// Open the launcher: game library, settings, key bindings.
    Gui {
        /// Open the settings window at start.
        #[arg(long)]
        settings: bool,
    },
    /// Show what a package or PFS image contains: title, version, content
    /// id, container details and every file with its size.
    InfoPkg {
        /// Path to a .pkg / .ffpkg / PFS image.
        path: String,
        /// Passcode of a fake package (32 characters; default all zeros).
        #[arg(long, value_name = "PASSCODE")]
        passcode: Option<String>,
        /// Do not list the files.
        #[arg(long)]
        no_files: bool,
    },
    /// Replay a frame capture (`RPS5E_CAPTURE_FRAMES`)
    /// through the GPU back-end, headless, as a benchmark: wall time,
    /// CPU time per draw and dispatch on the recording thread, GPU time.
    Replay {
        /// The `.ps5cap` file.
        capture: String,
        /// Timed loops over the captured frames (the first one compiles).
        #[arg(long, default_value_t = 3)]
        loops: u32,
        /// Do not copy flipped display images (flips only count frames).
        #[arg(long)]
        no_present: bool,
        /// GPU timings and back-end laps (sets RPS5E_GPU_PROFILE=1).
        #[arg(long)]
        profile: bool,
        /// After the timed loops, replay once more and write every flipped
        /// display buffer here as frame_NNNN.ppm.
        #[arg(long, value_name = "DIR")]
        dump_frames: Option<String>,
        /// Slice threads (0: serial; default RPS5E_SLICES): a sequencer
        /// thread cuts the records into slices that the replaying thread
        /// executes, as the command thread does in slice mode.
        #[arg(long)]
        slices: Option<u32>,
    },
    /// Build a game's shader prewarm list without running the game: every
    /// module the game created on earlier runs (translated again when the
    /// shader translator changed since) and every pipeline, into the module
    /// and pipeline caches. The list is made by running the game once.
    Precompile {
        /// The game folder (or its eboot.bin / param.json), or a title id
        /// such as PPSA01342.
        game: String,
        /// Compile threads [default: every core].
        #[arg(long, value_name = "N")]
        threads: Option<usize>,
        /// Clear the game's pipeline cache and cached modules first:
        /// everything is translated and built again.
        #[arg(long)]
        clear: bool,
    },
    /// A game's portable shader cache: export it for another player,
    /// import one (merged into the game's list; built on the next start or
    /// precompile), or show what the game's list holds. The driver's
    /// pipeline cache is specific to the GPU and driver and is not shared.
    ShaderCache {
        #[command(subcommand)]
        action: ShaderCacheCmd,
    },
    /// Extract a game package into a folder that then runs like any dump.
    /// An update package is extracted over the installed game.
    InstallPkg {
        /// Path to a .pkg / .ffpkg / PFS image.
        path: String,
        /// Destination folder [default: <data>/games/<TITLE_ID>].
        #[arg(long, value_name = "DIR")]
        into: Option<String>,
        /// Data folder holding games/ [RPS5E_DATA_DIR; default ~/.local/share/rps5e].
        #[arg(long, value_name = "DIR")]
        data_dir: Option<String>,
        /// Passcode of a fake package (32 characters; default all zeros).
        #[arg(long, value_name = "PASSCODE")]
        passcode: Option<String>,
    },
}

/// Launch settings. Each one maps onto the environment variable the
/// emulator reads, so they can also be set that way.
#[derive(clap::Args, Default)]
struct RunOptions {
    /// Writable game data (save data, /temp0, /download0, system settings),
    /// one subfolder per title id [RPS5E_DATA_DIR; default ~/.local/share/rps5e].
    #[arg(long, value_name = "DIR")]
    data_dir: Option<String>,
    /// Emulator settings folder with rps5e-input.toml and profiles/
    /// [RPS5E_CONFIG_DIR; default ~/.config/rps5e].
    #[arg(long, value_name = "DIR")]
    config_dir: Option<String>,
    /// Input mapping file to use instead of the one in the settings folder
    /// [RPS5E_INPUT_CONFIG].
    #[arg(long, value_name = "FILE")]
    input_config: Option<String>,
    /// Input devices: auto, keyboard_mouse, keyboard or controller.
    #[arg(long, value_name = "MODE")]
    input_mode: Option<String>,
    /// Bind an action, e.g. `--bind Cross=KEY:Space` or
    /// `--bind "R2=MOUSE:Right,KEY:E"` (repeatable; replaces that action's
    /// bindings; see rps5e-input.example.toml for names).
    #[arg(long, value_name = "ACTION=SOURCE[,SOURCE]")]
    bind: Vec<String>,
    /// Ignore host keyboard, mouse and controllers [RPS5E_NO_HOST_INPUT].
    #[arg(long)]
    no_input: bool,
    /// Scripted button taps, e.g. `20:cross,26:down` (seconds:button)
    /// [RPS5E_PAD_SCRIPT].
    #[arg(long, value_name = "SCRIPT")]
    pad_script: Option<String>,
    /// Record every pad change (buttons, sticks, triggers) to FILE
    /// [RPS5E_INPUT_RECORD].
    #[arg(long, value_name = "FILE")]
    record_input: Option<String>,
    /// Replay a recording made with --record-input; host input is ignored
    /// [RPS5E_INPUT_REPLAY].
    #[arg(long, value_name = "FILE")]
    replay_input: Option<String>,
    /// Open the game window fullscreen [RPS5E_FULLSCREEN].
    #[arg(long)]
    fullscreen: bool,
    /// HDR10 output for games that show HDR frames, when the display
    /// supports it (default off) [RPS5E_HDR].
    #[arg(long)]
    hdr: bool,
    /// Show the performance overlay from the start (F10 toggles)
    /// [RPS5E_PERF_OVERLAY].
    #[arg(long)]
    perf_overlay: bool,
    /// Window presentation: fifo, mailbox or immediate [RPS5E_PRESENT_MODE].
    #[arg(long, value_name = "MODE")]
    present_mode: Option<String>,
    /// Vulkan device index or name substring [RPS5E_GPU_DEVICE].
    #[arg(long, value_name = "DEVICE")]
    gpu: Option<String>,
    /// No window [RPS5E_HEADLESS].
    #[arg(long)]
    headless: bool,
    /// Console language id, 0..29 (1 = English US) [RPS5E_SYSTEM_LANGUAGE].
    #[arg(long, value_name = "ID")]
    language: Option<u32>,
    /// No audio output [RPS5E_AUDIO=null].
    #[arg(long)]
    mute: bool,
    /// Turn the console's skip-notice-screen setting on [RPS5E_SKIP_NOTICE_SCREEN].
    #[arg(long)]
    skip_notice_screen: bool,
    /// Guest display refresh rate in Hz [RPS5E_GUEST_REFRESH_RATE].
    #[arg(long, value_name = "HZ")]
    refresh_rate: Option<f64>,
    /// Game patch (mods JSON) applied at load [RPS5E_GAME_PATCH].
    #[arg(long, value_name = "FILE")]
    game_patch: Option<String>,
    /// Fixed controller light bar colour, `#RRGGBB` (overrides the game's).
    #[arg(long, value_name = "#RRGGBB")]
    controller_color: Option<String>,
    /// Start the game at once and build its shader prewarm list in the
    /// background (default: compile the whole list first)
    /// [RPS5E_PREWARM_WAIT=0].
    #[arg(long)]
    no_prewarm_wait: bool,
    /// How long draws may wait per frame for shaders still compiling
    /// before they are skipped: milliseconds, or `exact` (always wait;
    /// what tests, replays and smoke runs use) [RPS5E_COMPILE_WAIT_MS,
    /// default for play: 3].
    #[arg(long, value_name = "MS|exact")]
    compile_wait: Option<String>,
    /// Write every Nth presented frame as PPM into DIR [RPS5E_DUMP_FRAMES].
    #[arg(long, value_name = "DIR")]
    dump_frames: Option<String>,
    /// Frame interval for --dump-frames [RPS5E_DUMP_EVERY].
    #[arg(long, value_name = "N")]
    dump_every: Option<u64>,
}

/// The per-frame compile wait of interactive play (milliseconds): a module
/// or pipeline the disk or driver cache holds is ready well within it.
const DEFAULT_PLAY_COMPILE_WAIT_MS: &str = "3";

impl RunOptions {
    /// Translate the options into the environment before any emulator
    /// thread starts.
    fn apply(&self) -> Result<()> {
        let set = |k: &str, v: &str| std::env::set_var(k, v);
        let dir = |k: &str, v: &Option<String>| -> Result<()> {
            if let Some(d) = v {
                std::fs::create_dir_all(d).map_err(|e| anyhow!("{k} {d}: {e}"))?;
                let abs = std::fs::canonicalize(d).map_err(|e| anyhow!("{k} {d}: {e}"))?;
                set(k, &abs.to_string_lossy());
            }
            Ok(())
        };
        dir("RPS5E_DATA_DIR", &self.data_dir)?;
        dir("RPS5E_CONFIG_DIR", &self.config_dir)?;
        if let Some(f) = &self.input_config {
            if !std::path::Path::new(f).is_file() {
                return Err(anyhow!("--input-config {f}: no such file"));
            }
            set("RPS5E_INPUT_CONFIG", f);
        }
        if let Some(m) = &self.present_mode {
            let m = m.to_ascii_lowercase();
            if !["fifo", "mailbox", "immediate"].contains(&m.as_str()) {
                return Err(anyhow!("--present-mode {m}: expected fifo, mailbox or immediate"));
            }
            set("RPS5E_PRESENT_MODE", &m);
        }
        if let Some(g) = &self.gpu {
            set("RPS5E_GPU_DEVICE", g);
        }
        if self.headless {
            set("RPS5E_HEADLESS", "1");
        }
        if let Some(l) = self.language {
            if l > 29 {
                return Err(anyhow!("--language {l}: expected 0..29"));
            }
            set("RPS5E_SYSTEM_LANGUAGE", &l.to_string());
        }
        if self.mute {
            set("RPS5E_AUDIO", "null");
        }
        if self.skip_notice_screen {
            set("RPS5E_SKIP_NOTICE_SCREEN", "1");
        }
        if let Some(hz) = self.refresh_rate {
            if !(1.0..=1000.0).contains(&hz) {
                return Err(anyhow!("--refresh-rate {hz}: expected 1..1000"));
            }
            set("RPS5E_GUEST_REFRESH_RATE", &hz.to_string());
        }
        if let Some(f) = &self.game_patch {
            let abs = std::fs::canonicalize(f).map_err(|e| anyhow!("--game-patch {f}: {e}"))?;
            set(ps5core::game_patch::PATCH_ENV, &abs.to_string_lossy());
        }
        let mut toml = String::new();
        if let Some(c) = &self.controller_color {
            if ps5input::mapping::parse_color(c).is_none() {
                return Err(anyhow!("--controller-color {c}: expected #RRGGBB"));
            }
            toml.push_str(&format!("[gamepad]\nlightbar_color = {c:?}\n"));
        }
        if let Some(m) = &self.input_mode {
            if ps5input::mapping::DeviceMode::parse(m).is_none() {
                return Err(anyhow!("--input-mode {m}: expected auto, keyboard_mouse, keyboard or controller"));
            }
            toml.push_str(&format!("[device]\nmode = {m:?}\n"));
        }
        if !self.bind.is_empty() {
            // Keyboard-only mode reads its own binding set.
            let section = match self.input_mode.as_deref().and_then(ps5input::mapping::DeviceMode::parse) {
                Some(ps5input::mapping::DeviceMode::Keyboard) => "keyboard_bindings",
                _ => "bindings",
            };
            toml.push_str(&format!("[{section}]\n"));
            for b in &self.bind {
                let (action, sources) =
                    b.split_once('=').ok_or_else(|| anyhow!("--bind {b}: expected ACTION=SOURCE"))?;
                let list: Vec<String> =
                    sources.split(',').map(str::trim).filter(|s| !s.is_empty()).map(|s| format!("{s:?}")).collect();
                toml.push_str(&format!("{} = [{}]\n", action.trim(), list.join(", ")));
            }
        }
        if !toml.is_empty() {
            // Validate now so a typo fails at launch, not silently in game.
            ps5input::mapping::parse_config(&toml).map_err(|e| anyhow!("input options: {}", e.0))?;
            set(ps5input::OVERRIDE_ENV, &toml);
        }
        if self.no_input {
            set("RPS5E_NO_HOST_INPUT", "1");
        }
        if let Some(p) = &self.pad_script {
            set("RPS5E_PAD_SCRIPT", p);
        }
        if let Some(p) = &self.record_input {
            set("RPS5E_INPUT_RECORD", p);
        }
        if let Some(p) = &self.replay_input {
            set("RPS5E_INPUT_REPLAY", p);
            set("RPS5E_NO_HOST_INPUT", "1");
        }
        if self.fullscreen {
            set("RPS5E_FULLSCREEN", "1");
        }
        if self.hdr {
            set("RPS5E_HDR", "1");
        }
        if self.perf_overlay {
            set("RPS5E_PERF_OVERLAY", "1");
        }
        if self.no_prewarm_wait {
            set("RPS5E_PREWARM_WAIT", "0");
        }
        // Interactive play never waits for a shader compile: a draw whose
        // pipeline is not ready is skipped after a small per-frame budget
        //; tests ask for the exact mode explicitly.
        match &self.compile_wait {
            Some(v) => {
                if !v.eq_ignore_ascii_case("exact") && v.parse::<u64>().is_err() {
                    return Err(anyhow!("--compile-wait {v}: expected milliseconds or exact"));
                }
                set("RPS5E_COMPILE_WAIT_MS", v);
            }
            None if std::env::var_os("RPS5E_COMPILE_WAIT_MS").is_none() => set("RPS5E_COMPILE_WAIT_MS", DEFAULT_PLAY_COMPILE_WAIT_MS),
            None => {}
        }
        dir("RPS5E_DUMP_FRAMES", &self.dump_frames)?;
        if let Some(n) = self.dump_every {
            set("RPS5E_DUMP_EVERY", &n.to_string());
        }
        Ok(())
    }
}

/// Accept either the eboot itself or the game folder holding it.
fn resolve_eboot(path: &str) -> String {
    let p = std::path::Path::new(path);
    if p.is_dir() {
        let e = p.join("eboot.bin");
        if e.is_file() {
            return e.to_string_lossy().into_owned();
        }
    }
    path.to_string()
}

fn read_file(path: &str) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|e| anyhow!("read {path}: {e}"))
}

/// The plaintext headers of a signed (encrypted) SELF of the firmware.
fn print_signed_self_headers(path: &str, bytes: &[u8]) {
    let Ok(h) = ps5elf::parse_self_headers(bytes) else { return };
    println!("Image: {path}");
    println!("  format:       signed SELF (payload encrypted with console keys)");
    println!(
        "  type:         {:#x} ({})  phnum: {}",
        h.ehdr.e_type,
        if h.is_executable() { "executable" } else { "library" },
        h.ehdr.e_phnum
    );
    if let Some(ext) = h.ext {
        println!("  authority ID: {:#018x} ({:?})", ext.authority_id, ext.kind());
        println!("  program type: {}", ext.program_type);
        if let Some(fw) = ext.fw_version_string() {
            println!("  firmware:     {fw}");
        }
    }
    println!("  SELF blocks:  {} ({} encrypted)", h.segments.iter().filter(|s| s.is_blocked()).count(), h.encrypted_blocks());
    println!("\nProgram headers:");
    for p in &h.phdrs {
        println!(
            "  {:<22} vaddr={:#010x} filesz={:#x} memsz={:#x} flags={:#x}",
            format!("{:?}", ProgramType::from_raw(p.p_type)),
            p.p_vaddr,
            p.p_filesz,
            p.p_memsz,
            p.p_flags
        );
    }
    println!();
}

fn cmd_info(path: &str) -> Result<()> {
    let bytes = read_file(path)?;
    let image = match parse_self(&bytes) {
        Ok(image) => image,
        Err(e @ ps5elf::SelfLoadError::SignedSelf(_)) => {
            // The payload is encrypted; the headers are not.
            print_signed_self_headers(path, &bytes);
            return Err(anyhow!("parse: {e}"));
        }
        Err(e) => return Err(anyhow!("parse: {e}")),
    };

    let gen = if image.is_next_gen() { "PS5 (next-gen)" } else { "PS4" };
    let load_base = if image.is_next_gen() {
        header::PS5_MAIN_IMAGE_BASE
    } else {
        header::PS4_MAIN_IMAGE_BASE
    };

    println!("Image: {}", path);
    println!("  magic:        {:02x?}", &bytes[0..4]);
    println!("  format:       {}", if image.is_self { "SELF" } else { "ELF" });
    println!("  generation:   {}", gen);
    println!("  e_entry:      {:#x}", image.ehdr.e_entry);
    println!("  -> abs entry: {:#x}", load_base + image.ehdr.e_entry);
    println!("  load base:    {:#x}", load_base);
    println!(
        "  type/machine: {:#x}/{:#x}  phnum: {}",
        image.ehdr.e_type, image.ehdr.e_machine, image.ehdr.e_phnum
    );

    println!("\nSELF segments:");
    for s in &image.self_segments {
        println!(
            "  blocked={} phdr_id={:<3} enc={} comp={} off={:#x} csz={:#x} dsz={:#x}",
            s.is_blocked(),
            s.program_header_id(),
            s.is_encrypted(),
            s.is_compressed(),
            s.offset,
            s.compressed_size,
            s.decompressed_size
        );
    }

    println!("\nProgram headers:");
    for (i, s) in image.segments.iter().enumerate() {
        let kind = match s.p_type {
            ProgramType::Load => "LOAD",
            ProgramType::Dynamic => "DYN",
            ProgramType::Tls => "TLS",
            ProgramType::SceDynlibData => "SCE_DYNLIBDATA",
            ProgramType::SceProcParam => "SCE_PROCPARAM",
            ProgramType::SceRelRo => "SCE_RELRO",
            ProgramType::Other(v) => {
                println!(
                    "  ph[{:>2}] OTHER({:#x}) vaddr={:#014x} filesz={:#x}",
                    i, v, s.vaddr, s.filesz
                );
                continue;
            }
        };
        println!(
            "  ph[{:>2}] {:<14} vaddr={:#014x} (abs {:#x}) filesz={:#x} memsz={:#x} flags={:#x} data={}B",
            i,
            kind,
            s.vaddr,
            load_base + s.vaddr,
            s.filesz,
            s.memsz,
            s.p_flags,
            s.data.len()
        );
    }

    if let Some(idx) = image.dynamic_segment_index() {
        println!("\nPT_DYNAMIC at ph[{}]", idx);
    }
    Ok(())
}

fn cmd_load(path: &str) -> Result<()> {
    let bytes = read_file(path)?;
    match ps5core::load_eboot(&bytes, path) {
        Ok(game) => {
            println!("loaded OK");
            println!("  image_base = {:#x}", game.image_base);
            println!("  image_size = {:#x}", game.image_size);
            println!("  entry      = {:#x}", game.entry);
            println!("  hle exports = {}", game.hle.len());
            if let Some(d) = game.dynamic() {
                println!("\nDynamic:");
                println!("  entries    = {}", d.table.entries.len());
                println!("  strtab     = {}B", d.strtab.len());
                println!("  symtab     = {}B", d.symtab.len());
                println!("  relas      = {}", d.relas.len());
                println!("  jmp_rels   = {}", d.jmp_rels.len());
                println!("  init_array = {} entries", d.init_array.len());
                println!("  fini_array = {} entries", d.fini_array.len());
                println!("  init_func  = {:#x}", d.init_func);
                if !d.init_array.is_empty() {
                    let abs = d.init_array[0] + game.image_base;
                    println!("  init_array[0] abs = {:#x}", abs);
                }
                println!("\n  import stubs = {}", game.import_stubs.count());
            } else {
                println!("\nDynamic: (none / unreadable)");
            }
        }
        Err(ps5core::EmuError::Encrypted) => {
            eprintln!("eboot.bin is encrypted — provide a decrypted (fSELF) image.");
        }
        Err(e) => return Err(anyhow!("load: {e}")),
    }
    Ok(())
}

/// `hle-exports`: the HLE registry as TSV (nid, name, library, kind).
fn cmd_hle_exports() -> anyhow::Result<()> {
    let mut reg = ps5hle::HleRegistry::with_defaults();
    reg.register_known_stubs();
    let mut rows: Vec<_> = reg
        .exports()
        .map(|(e, implemented)| {
            let kind = if !implemented { "stub" } else if e.fallback { "fallback" } else { "fn" };
            (e.library, e.name, e.nid, kind)
        })
        .collect();
    rows.sort();
    println!("nid\tname\tlibrary\tkind");
    for (library, name, nid, kind) in rows {
        println!("{nid}\t{name}\t{library}\t{kind}");
    }
    Ok(())
}

fn cmd_isa_scan(paths: &[String]) -> Result<()> {
    use ps5cpu::isa_scan::{exec_segments, scan, IsaReport, AMD_ONLY, INTEL_GENERATION};
    let mut files = Vec::new();
    for p in paths {
        let path = std::path::Path::new(p);
        if path.is_dir() {
            files.push(path.join("eboot.bin"));
            let mut prx: Vec<_> = std::fs::read_dir(path.join("sce_module"))
                .map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|e| e == "prx" || e == "sprx")).collect())
                .unwrap_or_default();
            prx.sort();
            files.extend(prx);
        } else {
            files.push(path.to_path_buf());
        }
    }
    let mut total = IsaReport::default();
    for f in &files {
        let name = f.display().to_string();
        let elf = match ps5elf::self_to_elf(&read_file(&name)?) {
            Ok(e) => e,
            Err(e) => {
                println!("{name}: {e}");
                continue;
            }
        };
        let mut r = IsaReport::default();
        for (vaddr, bytes) in exec_segments(&elf) {
            scan(bytes, vaddr, &mut r);
        }
        println!("== {name}: {} instructions, AMD-only {}, Intel-generation {}", r.instructions, r.count_of(AMD_ONLY), r.count_of(INTEL_GENERATION));
        let line = |m: &std::collections::BTreeMap<String, u64>| m.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" ");
        println!("   features: {}", line(&r.features));
        println!("   host-dependent: {}", line(&r.host_dependent));
        for (k, v) in &r.samples {
            println!("   {k} at {}", v.iter().map(|a| format!("{a:#x}")).collect::<Vec<_>>().join(" "));
        }
        total.merge(&r);
    }
    if files.len() > 1 {
        println!("== total: {} instructions, AMD-only {}, Intel-generation {}", total.instructions, total.count_of(AMD_ONLY), total.count_of(INTEL_GENERATION));
        let line = |m: &std::collections::BTreeMap<String, u64>| m.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" ");
        println!("   features: {}", line(&total.features));
        println!("   host-dependent: {}", line(&total.host_dependent));
    }
    Ok(())
}

fn cmd_imports(path: &str) -> Result<()> {
    let bytes = read_file(path)?;
    let game = ps5core::load_eboot(&bytes, path).map_err(|e| anyhow!("load: {e}"))?;
    let mut rows: Vec<(String, String, String, String)> = game
        .import_stubs
        .nid_to_addr_iter()
        .map(|(nid, _)| {
            let e = game.hle.lookup(nid);
            let name = e.map(|e| e.name).unwrap_or("?").to_string();
            let lib = e.map(|e| e.library).unwrap_or("?").to_string();
            let status = if let Some(t) = game.lle_imports.get(nid) {
                format!("LLE@{t:#x}")
            } else if game.hle.is_implemented(nid) {
                "HLE".to_string()
            } else {
                "STUB".to_string()
            };
            (lib, name, nid.clone(), status)
        })
        .collect();
    rows.sort();
    for (lib, name, nid, status) in &rows {
        println!("{nid}\t{name}\t{lib}\t{status}");
    }
    let count = |p: &str| rows.iter().filter(|r| r.3.starts_with(p)).count();
    eprintln!(
        "total={} HLE={} LLE={} STUB={}",
        rows.len(),
        count("HLE"),
        count("LLE"),
        count("STUB")
    );
    Ok(())
}

/// One emulator process at a time on this machine: two games at once
/// saturate the CPU and share the GPU's memory. A later `run` waits for
/// the earlier one to exit; the kernel drops the lock with the process.
fn hold_run_lock() -> Result<()> {
    let dir = std::env::var_os("XDG_RUNTIME_DIR").map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir);
    hold_lock_file(&dir.join("rps5e-run.lock"))
}

fn hold_lock_file(path: &std::path::Path) -> Result<()> {
    let file = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(path).map_err(|e| anyhow!("open {}: {e}", path.display()))?;
    let fd = ps5hle::host::fs::as_raw_fd(&file);
    // SAFETY: flock on a descriptor we own.
    if unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        eprintln!("another emulator is running ({}); waiting for it to exit...", path.display());
        // SAFETY: as above.
        while unsafe { libc::flock(fd, libc::LOCK_EX) } != 0 {
            if std::io::Error::last_os_error().raw_os_error() != Some(libc::EINTR) {
                return Err(anyhow!("lock {}: {}", path.display(), std::io::Error::last_os_error()));
            }
        }
    }
    // Held until the process exits.
    std::mem::forget(file);
    Ok(())
}

/// Core dumps leave out shared anonymous memory (coredump_filter 0x31
/// instead of the kernel's 0x33): the guest's direct and flexible memory is
/// a ~14 GiB memfd mapping, and writing it out after a crash takes minutes,
/// with the machine under load and the emulator looking hung. Host code,
/// stacks and heap stay in the dump (coredumpctl). `RPS5E_FULL_CORE=1`
/// keeps the default filter.
fn limit_core_dumps() {
    // Linux core dump filter only.
    if cfg!(windows) {
        return;
    }
    if std::env::var("RPS5E_FULL_CORE").is_ok_and(|v| v == "1") {
        return;
    }
    if let Err(e) = std::fs::write("/proc/self/coredump_filter", "0x31") {
        log::warn!("cannot set /proc/self/coredump_filter: {e}");
    }
}

fn cmd_replay(capture: &str, loops: u32, present: bool, profile: bool, dump_frames: Option<String>, slices: Option<u32>) -> Result<()> {
    // Read once by the back-end: set before it starts. An explicit
    // RPS5E_GPU_PROFILE (=2: the detailed sub-phases) wins over --profile.
    if profile && std::env::var_os("RPS5E_GPU_PROFILE").is_none() {
        std::env::set_var("RPS5E_GPU_PROFILE", "1");
    }
    std::env::set_var("RPS5E_HEADLESS", "1");
    // The GPU is shared with a running game the same way.
    hold_run_lock()?;
    let opts = ps5gpu::capture::replay::Options {
        loops: loops.max(1),
        present,
        dump_dir: dump_frames.map(Into::into),
        slices: slices.unwrap_or_else(ps5gpu::slices::configured).min(16),
    };
    let report = ps5gpu::capture::replay::replay_file(std::path::Path::new(capture), &opts).map_err(|e| anyhow!("replay {capture}: {e}"))?;
    print!("{report}");
    // End like a game run: without the atexit chain. The GPU driver's
    // libraries are finalised there while its own pipeline-compile threads
    // still run, which crashed or aborted the process at exit ("NVVM
    // compilation failed", SIGSEGV in the driver compiler, double free).
    ps5vm::shutdown::exit(0)
}

/// The title id of `game`: a game folder, a file in it (eboot.bin,
/// param.json) or the id itself.
fn title_of(game: &str) -> Result<String> {
    let p = std::path::Path::new(game);
    if !p.exists() {
        return if ps5gpu::pipeline::cache::valid_title_id(game) { Ok(game.to_string()) } else { Err(anyhow!("{game}: no such game folder or title id")) };
    }
    let mut dir = if p.is_dir() { p.to_path_buf() } else { p.parent().map(std::path::Path::to_path_buf).unwrap_or_default() };
    if dir.file_name().is_some_and(|n| n == "sce_sys") {
        dir.pop();
    }
    let json = std::fs::read_to_string(dir.join("sce_sys").join("param.json")).map_err(|e| anyhow!("{}: {e}", dir.join("sce_sys/param.json").display()))?;
    let id = ps5hle::app_info::param_json_string(&json, "titleId")
        .or_else(|| ps5hle::app_info::param_json_string(&json, "contentId"))
        .map(|s| s.trim().to_string())
        .filter(|s| ps5gpu::pipeline::cache::valid_title_id(s))
        .ok_or_else(|| anyhow!("{}: no title id in param.json", dir.display()))?;
    Ok(id)
}

/// The pipeline cache file of `title` (its prewarm list sits next to it).
fn title_cache_path(title: &str) -> Result<std::path::PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))
        .ok_or_else(|| anyhow!("no cache directory (HOME unset)"))?;
    Ok(match std::env::var("RPS5E_PIPELINE_CACHE") {
        Ok(v) if !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("off") => std::path::PathBuf::from(v),
        _ => ps5gpu::pipeline::cache::title_path(&base, title),
    })
}

fn cmd_shader_cache(action: ShaderCacheCmd) -> Result<()> {
    match action {
        ShaderCacheCmd::Export { game, file } => {
            let title = title_of(&game)?;
            let list = ps5gpu::pipeline::prewarm::list_path(&title_cache_path(&title)?);
            let bytes = ps5gpu::pipeline::portable::export(&list, &title).map_err(|e| anyhow!("{title}: {e}"))?;
            std::fs::write(&file, &bytes).map_err(|e| anyhow!("{file}: {e}"))?;
            println!("{title}: shader cache written to {file} ({} bytes)", bytes.len());
        }
        ShaderCacheCmd::Import { file, game } => {
            let bytes = ps5gpu::pipeline::portable::read_file(std::path::Path::new(&file)).map_err(|e| anyhow!("{e}"))?;
            let named = ps5gpu::pipeline::portable::title_of(&bytes).ok_or_else(|| anyhow!("{file}: not a rps5e shader cache file"))?;
            let title = match game {
                Some(g) => title_of(&g)?,
                None if ps5gpu::pipeline::cache::valid_title_id(&named) => named,
                None => return Err(anyhow!("{file}: bad title id in the file")),
            };
            // A running game rewrites its list: wait for it.
            hold_run_lock()?;
            let list = ps5gpu::pipeline::prewarm::list_path(&title_cache_path(&title)?);
            let summary = ps5gpu::pipeline::portable::import(&list, &title, &bytes).map_err(|e| anyhow!("{file}: {e}"))?;
            println!("{title}: {summary}");
        }
        ShaderCacheCmd::Info { game } => {
            let title = title_of(&game)?;
            let list = ps5gpu::pipeline::prewarm::list_path(&title_cache_path(&title)?);
            print!("{title}: {}", ps5gpu::pipeline::portable::describe(&list).map_err(|e| anyhow!("{e}"))?);
        }
        ShaderCacheCmd::Relocation { game } => {
            let title = title_of(&game)?;
            std::env::set_var("RPS5E_HEADLESS", "1");
            hold_run_lock()?;
            let cache = title_cache_path(&title)?;
            let report = ps5gpu::pipeline::precompile::relocation_report(&cache, ps5gpu::pipeline::precompile::all_cores()).map_err(|e| anyhow!("{title}: {e}"))?;
            print!("{title}: {report}");
            ps5vm::shutdown::exit(0)
        }
        ShaderCacheCmd::Translate { game, hashes, keys, dump, threads, top } => {
            let list = if game.ends_with(".prewarm") { std::path::PathBuf::from(&game) } else { ps5gpu::pipeline::prewarm::list_path(&title_cache_path(&title_of(&game)?)?) };
            let keys: std::collections::HashSet<u64> = keys.iter().map(|k| u64::from_str_radix(k.trim().trim_start_matches("0x"), 16).map_err(|e| anyhow!("{k}: {e}"))).collect::<Result<_>>()?;
            std::env::set_var("RPS5E_HEADLESS", "1");
            hold_run_lock()?;
            let workers = threads.unwrap_or_else(ps5gpu::pipeline::precompile::all_cores).max(1);
            let start = std::time::Instant::now();
            let only = |k: u64| keys.is_empty() || keys.contains(&k);
            let modules = ps5gpu::pipeline::precompile::translate_list(&list, workers, &only, dump.as_deref().map(std::path::Path::new)).map_err(|e| anyhow!("{e}"))?;
            print!("{}", translation_summary(&modules, start.elapsed(), top));
            if let Some(file) = hashes {
                let mut out = String::new();
                for m in &modules {
                    out += &format!("{:016x} {} {:#x} {} {:016x} {:.1}\n", m.key, m.stage, m.address, m.words, m.hash, m.ns as f64 / 1e6);
                }
                std::fs::write(&file, out).map_err(|e| anyhow!("{file}: {e}"))?;
            }
            ps5vm::shutdown::exit(0)
        }
        ShaderCacheCmd::Clear { game, all, list } => {
            // A running game writes these files: wait for it.
            hold_run_lock()?;
            if all {
                clear_all_shader_caches(list)?;
            } else {
                let title = title_of(game.as_deref().unwrap_or_default())?;
                clear_title_shader_cache(&title, list)?;
            }
        }
    }
    Ok(())
}

/// `shader-cache translate`: totals, the time per part and the slowest
/// modules.
fn translation_summary(modules: &[ps5gpu::pipeline::precompile::TranslatedModule], wall: std::time::Duration, top: usize) -> String {
    use ps5gpu::shader::stats::LAST_NAMES;
    let ms = |ns: u64| ns as f64 / 1e6;
    let failed = modules.iter().filter(|m| m.error.is_some()).count();
    let total: u64 = modules.iter().map(|m| m.ns).sum();
    let mut parts = [0u64; LAST_NAMES.len()];
    for m in modules {
        for (a, b) in parts.iter_mut().zip(m.parts) {
            *a += b;
        }
    }
    let mut out = format!(
        "{} modules, {failed} not translated, {:.1} s of translation in {:.1} s\nparts ms: {}\n",
        modules.len(),
        total as f64 / 1e9,
        wall.as_secs_f64(),
        LAST_NAMES.iter().zip(parts).map(|(n, v)| format!("{n} {:.0}", ms(v))).collect::<Vec<_>>().join(" ")
    );
    let mut slow: Vec<&ps5gpu::pipeline::precompile::TranslatedModule> = modules.iter().collect();
    slow.sort_by_key(|m| std::cmp::Reverse(m.ns));
    for m in slow.iter().take(top) {
        let p: Vec<String> = LAST_NAMES.iter().zip(m.parts).filter(|(_, v)| *v >= 1_000_000).map(|(n, v)| format!("{n} {:.0}", ms(v))).collect();
        out += &format!("{:>9.1} ms  key {:016x} stage {} program {:#x} {} words  [{}]{}\n", ms(m.ns), m.key, m.stage, m.address, m.words, p.join(" "), m.error.as_ref().map_or(String::new(), |e| format!("  error: {e}")));
    }
    out
}

fn remove_if_present(path: &std::path::Path) -> Result<bool> {
    let r = if path.is_dir() { std::fs::remove_dir_all(path) } else { std::fs::remove_file(path) };
    match r {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(anyhow!("{}: {e}", path.display())),
    }
}

fn clear_title_shader_cache(title: &str, list: bool) -> Result<()> {
    let cache = title_cache_path(title)?;
    let shader_list = ps5gpu::pipeline::prewarm::list_path(&cache);
    remove_if_present(&cache)?;
    if list {
        remove_if_present(&shader_list)?;
        println!("{title}: pipeline cache and shader list removed");
    } else if shader_list.is_file() {
        let modules = ps5gpu::pipeline::prewarm::mark_for_recompile(&shader_list).map_err(|e| anyhow!("{e}"))?;
        println!("{title}: pipeline cache removed, {} shaders will be translated and built again on the next start or precompile", modules.len());
    } else {
        println!("{title}: pipeline cache removed (no shader list)");
    }
    Ok(())
}

fn clear_all_shader_caches(list: bool) -> Result<()> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))
        .ok_or_else(|| anyhow!("no cache directory (HOME unset)"))?;
    let app = ps5vm::appenv::app_dir(&base);
    let module_dir = match std::env::var("RPS5E_SHADER_CACHE") {
        Ok(v) if !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("off") => std::path::PathBuf::from(v),
        _ => app.clone(),
    };
    let mut removed = 0usize;
    for entry in std::fs::read_dir(&module_dir).into_iter().flatten().flatten() {
        if entry.file_name().to_string_lossy().starts_with("spirv-") && remove_if_present(&entry.path())? {
            removed += 1;
        }
    }
    let mut games = 0usize;
    for entry in std::fs::read_dir(app.join("pipelines")).into_iter().flatten().flatten() {
        let path = entry.path();
        let Some(title) = path.file_stem().map(|s| s.to_string_lossy().into_owned()) else { continue };
        if path.extension().is_some_and(|e| e == "bin") && ps5gpu::pipeline::cache::valid_title_id(&title) {
            clear_title_shader_cache(&title, list)?;
            games += 1;
        } else if list && path.extension().is_some_and(|e| e == "prewarm") {
            remove_if_present(&path)?;
        }
    }
    println!("cleared {removed} translated-module caches and the pipeline caches of {games} games");
    Ok(())
}

fn cmd_precompile(game: &str, threads: Option<usize>, clear: bool) -> Result<()> {
    let title = title_of(game)?;
    std::env::set_var("RPS5E_HEADLESS", "1");
    // The GPU is shared with a running game the same way.
    hold_run_lock()?;
    ps5gpu::pipeline::cache::set_title_id(&title);
    let cache = title_cache_path(&title)?;
    let threads = threads.unwrap_or_else(ps5gpu::pipeline::precompile::all_cores).max(1);
    println!("{title}: building {} on {threads} threads", ps5gpu::pipeline::prewarm::list_path(&cache).display());
    // One line a second (the launcher shows the last as a progress bar).
    let mut printed: Option<std::time::Instant> = None;
    let mut progress = |p: &ps5gpu::pipeline::PrewarmProgress| {
        if printed.is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(1)) || p.done == p.total {
            printed = Some(std::time::Instant::now());
            println!("  {}", ps5gpu::pipeline::precompile::progress_line(p));
        }
        true
    };
    let result = ps5gpu::pipeline::precompile::precompile(&cache, threads, clear, &mut progress);
    match result {
        Ok(summary) => {
            println!("{title}: {summary}");
            // As after a replay: without the atexit chain (the driver's
            // compile threads).
            ps5vm::shutdown::exit(0)
        }
        Err(e) => Err(anyhow!("{title}: {e}")),
    }
}

fn cmd_run(path: &str) -> Result<()> {
    hold_run_lock()?;
    limit_core_dumps();
    ps5cpu::host_cpu::check_guest_requirements().map_err(|e| anyhow!("{e}"))?;
    let bytes = read_file(path)?;
    let game = ps5core::load_eboot(&bytes, path).map_err(|e| anyhow!("load: {e}"))?;

    println!(
        "entry={:#x}  hle_exports={}  stubs={}",
        game.entry,
        game.hle.len(),
        game.import_stubs.count(),
    );

    let image = game.image.clone();
    let (image_base, entry) = (game.image_base, game.entry);
    let runtime = ps5core::runtime::Runtime::new(game);
    // Every shader and pipeline of the title's prewarm list, before any
    // guest code runs and before input starts (RPS5E_PREWARM_WAIT=0: in
    // the background instead). The wait is not the title's process time:
    // pad scripts and input recordings keep their timing.
    let waited = std::time::Instant::now();
    ps5gpu::pipeline::precompile::wait_before_guest();
    ps5hle::time::exclude_from_process_time(waited.elapsed());
    if ps5vm::shutdown::requested() {
        ps5vm::shutdown::exit(0);
    }
    // The in-game menu (F1 / PS button), before input starts so that its
    // hotkey is announced.
    if !headless() {
        rps5e_gui::ingame::install(std::path::Path::new(path));
    }
    ps5input::start();
    ps5cpu::exec::init();
    ps5cpu::exec::set_dispatcher(Box::new(runtime.dispatcher()));
    runtime.install_module_loader();
    ps5core::runtime::start_stats_reporter();
    ps5gpu::perf::set_hle_call_counter(ps5cpu::exec::total_imports);
    // The VM lives for the whole process.
    let vm_static: &'static ps5vm::VirtualMemory = unsafe { &*std::sync::Arc::into_raw(runtime.vm.clone()) };

    let stack_size = main_thread_stack_size(vm_static, &image, image_base);
    let (_tp, stack_top) = ps5hle::pthread::init_main_thread(vm_static, stack_size)
        .ok_or_else(|| anyhow!("cannot create the main guest thread"))?;

    // Module start: run DT_INIT of every preloaded PRX in load order with
    // all argument registers zero. DT_INIT is the module's `_init`, which
    // itself walks DT_PREINIT_ARRAY and the .ctors list. The main
    // executable's `_init` is called by its own `_start`.
    for (name, base, init) in runtime.module_inits() {
        if init == 0 {
            continue;
        }
        log::info!("[main] module_start {} DT_INIT at {:#x}", name, base + init);
        let exit = ps5cpu::exec::enter_guest(&ps5cpu::exec::call_regs(base + init, stack_top - 0x1000, [0; 6]));
        log::info!("[main] module {} DT_INIT done ({:?})", name, exit);
    }

    let params = build_entry_params(stack_top - 0x1000, "eboot.bin");
    println!("starting guest execution at entry={:#x}...", entry);
    let exit = ps5cpu::exec::enter_guest(&ps5cpu::exec::call_regs(
        entry,
        params,
        [params, runtime.exit_handler(), 0, 0, 0, 0],
    ));
    println!(
        "main thread finished: {:?} imports={}",
        exit,
        ps5cpu::exec::imports_dispatched()
    );
    // Other guest threads may still run: end through the shutdown path,
    // never through the atexit chain.
    ps5vm::shutdown::exit(0)
}

/// No game window (`RPS5E_HEADLESS=1`).
fn headless() -> bool {
    std::env::var("RPS5E_HEADLESS").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// Main thread stack size from ProcParam (`sceUserMainThreadStackSize`),
/// defaulting to 2 MiB.
fn main_thread_stack_size(vm: &ps5vm::VirtualMemory, image: &ps5elf::SelfImage, base: u64) -> u64 {
    const DEFAULT: u64 = 0x20_0000;
    let Some(pp) = image.phdrs.iter().find(|p| ProgramType::from_raw(p.p_type) == ProgramType::SceProcParam) else {
        return DEFAULT;
    };
    let mut ptr = [0u8; 8];
    if vm.read(base + pp.p_vaddr + 0x30, &mut ptr).is_err() {
        return DEFAULT;
    }
    let ptr = u64::from_le_bytes(ptr);
    let mut size = [0u8; 4];
    if ptr == 0 || vm.read(ptr, &mut size).is_err() {
        return DEFAULT;
    }
    let size = u32::from_le_bytes(size) as u64;
    if size == 0 { DEFAULT } else { (size + 0x3FFF) & !0x3FFF }
}

/// Write the process entry parameters below `top`:
/// `{ u32 argc; u32 pad; char* argv[3]; }` with argv[0] = `name`.
/// Returns the 16-byte aligned address of the parameter block.
fn build_entry_params(top: u64, name: &str) -> u64 {
    let mut sp = top & !0xF;
    let bytes = format!("{name}\0").into_bytes();
    sp = (sp - bytes.len() as u64) & !0xF;
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), sp as *mut u8, bytes.len()) };
    let argv0 = sp;
    sp = (sp - 0x20) & !0xF;
    unsafe {
        std::ptr::write(sp as *mut u32, 1);
        std::ptr::write((sp + 4) as *mut u32, 0);
        std::ptr::write((sp + 8) as *mut u64, argv0);
        std::ptr::write((sp + 0x10) as *mut u64, 0);
        std::ptr::write((sp + 0x18) as *mut u64, 0);
    }
    sp
}

/// The launcher; games it starts run as `<this binary> run <game>`.
fn gui(open_settings: bool) -> Result<()> {
    rps5e_gui::run_with(rps5e_gui::Options { open_settings }).map_err(|e| anyhow!("launcher: {e}"))
}

/// Run the command line of binary `bin` (its name in the usage text).
pub fn main_entry(bin: &'static str) -> Result<()> {
    env_logger::init();
    // RPS5E_ALLOW_PTRACE=1: let a non-parent sampler (eu-stack, gdb -p)
    // attach under Yama ptrace_scope=1, for profiling.
    #[cfg(unix)]
    if std::env::var("RPS5E_ALLOW_PTRACE").is_ok_and(|v| v == "1") {

        // SAFETY: plain prctl with integer arguments.
        unsafe { libc::prctl(libc::PR_SET_PTRACER, libc::PR_SET_PTRACER_ANY, 0, 0, 0) };
    }
    let cli = match Cli::from_arg_matches(&Cli::command().name(bin).bin_name(bin).get_matches()) {
        Ok(c) => c,
        Err(e) => e.exit(),
    };
    if matches!(cli.cmd, Some(Cmd::Run { .. })) {
        // Before any thread exists: SIGINT/SIGTERM/SIGHUP go to the
        // shutdown thread (window close and guest exit end there too).
        ps5vm::shutdown::install();
    }
    let Some(cmd) = cli.cmd else { return gui(false) };
    match cmd {
        Cmd::Gui { settings } => gui(settings),
        Cmd::Info { path } => cmd_info(&path),
        Cmd::Self2elf { path, out } => {
            let elf = ps5elf::self_to_elf(&read_file(&path)?).map_err(|e| anyhow!("{path}: {e}"))?;
            std::fs::write(&out, &elf).map_err(|e| anyhow!("{out}: {e}"))?;
            println!("{out}: {} bytes", elf.len());
            Ok(())
        }
        Cmd::Load { path } => cmd_load(&path),
        Cmd::IsaScan { paths } => cmd_isa_scan(&paths),
        Cmd::Imports { path } => cmd_imports(&path),
        Cmd::HleExports => cmd_hle_exports(),
        Cmd::Run { path, passcode, opts } => {
            opts.apply()?;
            let p = std::path::Path::new(&path);
            if p.is_file() && ps5pkg::is_package_file(p) {
                let eboot = mount_package(p, &pkg_options(passcode)?)?;
                return cmd_run(&eboot.to_string_lossy());
            }
            cmd_run(&resolve_eboot(&path))
        }
        Cmd::Replay { capture, loops, no_present, profile, dump_frames, slices } => cmd_replay(&capture, loops, !no_present, profile, dump_frames, slices),
        Cmd::Precompile { game, threads, clear } => cmd_precompile(&game, threads, clear),
        Cmd::ShaderCache { action } => cmd_shader_cache(action),
        Cmd::InfoPkg { path, passcode, no_files } => pkg_cli::info(&path, &pkg_options(passcode)?, !no_files),
        Cmd::InstallPkg { path, into, data_dir, passcode } => {
            let data = match data_dir {
                Some(d) => std::path::PathBuf::from(d),
                None => ps5hle::filesystem::user_data_dir(),
            };
            pkg_cli::install(&path, into.as_deref().map(std::path::Path::new), &data, &pkg_options(passcode)?)
        }
    }
}

fn pkg_options(passcode: Option<String>) -> Result<ps5pkg::OpenOptions> {
    let mut o = ps5pkg::OpenOptions::default();
    if let Some(p) = passcode {
        if p.len() != 32 || !p.is_ascii() {
            return Err(anyhow!("--passcode must be 32 ASCII characters"));
        }
        o.passcode = p;
    }
    Ok(o)
}

/// Mount a game package at /app0 and return the host path of its eboot.
fn mount_package(path: &std::path::Path, opts: &ps5pkg::OpenOptions) -> Result<std::path::PathBuf> {
    let pkg = pkg_cli::open_game(path, opts)?;
    if pkg.info.kind != ps5pkg::ContentKind::Game {
        return Err(anyhow!(
            "{}: this is {} content, not a game; only game packages can be run{}",
            path.display(),
            pkg.info.kind,
            if pkg.info.kind == ps5pkg::ContentKind::Patch {
                " (install the game with install-pkg, then install-pkg this update over it)"
            } else {
                ""
            }
        ));
    }
    if pkg.fs().lookup("eboot.bin").is_none() {
        return Err(anyhow!("{}: the package has no eboot.bin", path.display()));
    }
    let title = pkg.info.folder_name().unwrap_or_else(|| "UNKNOWN".into());
    let meta = std::fs::metadata(path)?;
    let mtime = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
    let abs = std::fs::canonicalize(path)?;
    let source_id = format!("{}\n{}\n{}\n", abs.display(), meta.len(), mtime);
    println!(
        "package: {} {} ({}) mounted read-only as /app0",
        title,
        pkg.info.name.as_deref().unwrap_or(""),
        pkg.info.version.as_deref().unwrap_or("?")
    );
    let cache = ps5hle::filesystem::mount_app0_image(pkg.fs(), &title, &source_id)
        .map_err(|e| anyhow!("mount {}: {e}", path.display()))?;
    Ok(cache.join("eboot.bin"))
}

mod pkg_cli;
