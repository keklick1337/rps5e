// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! `rps5e`: without arguments the launcher (game library); with
//! arguments the same commands as `rps5e-cli` (`rps5e run <game>`).

fn main() -> anyhow::Result<()> {
    rps5e_cli::main_entry("rps5e")
}
