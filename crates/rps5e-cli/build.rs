// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Windows builds: compiles a resource script with the application icon
//! and version info and links it into `rps5e.exe` and `rps5e-cli.exe`.
//! Other targets need nothing.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let ico = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/rps5e.ico");
    println!("cargo:rerun-if-changed={}", ico.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=WINDRES");
    println!("cargo:rerun-if-env-changed=RC");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ico = ico.canonicalize().unwrap_or(ico);
    let version = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();
    let nums: Vec<u32> = version.split(['.', '-']).take(3).map(|p| p.parse().unwrap_or(0)).collect();
    let (major, minor, patch) = (nums.first().copied().unwrap_or(0), nums.get(1).copied().unwrap_or(0), nums.get(2).copied().unwrap_or(0));
    // Forward slashes: both windres and rc accept them, and no escaping.
    let ico_path = ico.to_string_lossy().replace('\\', "/");
    let rc = format!(
        r#"1 ICON "{ico_path}"
1 VERSIONINFO
FILEVERSION {major},{minor},{patch},0
PRODUCTVERSION {major},{minor},{patch},0
BEGIN
  BLOCK "StringFileInfo"
  BEGIN
    BLOCK "040904B0"
    BEGIN
      VALUE "CompanyName", "Vladislav Tislenko (keklick1337)"
      VALUE "FileDescription", "RustPS5Emu (RPS5E)"
      VALUE "FileVersion", "{version}"
      VALUE "LegalCopyright", "Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337, GPL-2.0-only"
      VALUE "ProductName", "RustPS5Emu"
      VALUE "ProductVersion", "{version}"
    END
  END
  BLOCK "VarFileInfo"
  BEGIN
    VALUE "Translation", 0x409, 1200
  END
END
"#
    );
    let rc_file = out.join("rps5e.rc");
    std::fs::write(&rc_file, rc).expect("write the resource script");
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    let (status, obj) = if msvc {
        let res = out.join("rps5e.res");
        let tool = std::env::var("RC").unwrap_or_else(|_| "rc".into());
        (Command::new(tool).arg("/nologo").arg("/fo").arg(&res).arg(&rc_file).status(), res)
    } else {
        let obj = out.join("rps5e-res.o");
        let tool = std::env::var("WINDRES").unwrap_or_else(|_| default_windres());
        (Command::new(tool).arg("-O").arg("coff").arg("-i").arg(&rc_file).arg("-o").arg(&obj).status(), obj)
    };
    match status {
        Ok(s) if s.success() => println!("cargo:rustc-link-arg-bins={}", obj.display()),
        Ok(s) => println!("cargo:warning=resource compiler failed ({s}): the executables have no icon"),
        Err(e) => println!("cargo:warning=no resource compiler ({e}): the executables have no icon"),
    }
}

/// The MinGW resource compiler: the cross-prefixed one when building on
/// another host, plain `windres` on Windows (MSYS2).
fn default_windres() -> String {
    if cfg!(windows) {
        "windres".into()
    } else {
        "x86_64-w64-mingw32-windres".into()
    }
}
