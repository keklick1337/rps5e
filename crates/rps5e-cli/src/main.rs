// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! `rps5e-cli`: the emulator's command line (no arguments: the launcher).

fn main() -> anyhow::Result<()> {
    rps5e_cli::main_entry("rps5e-cli")
}
