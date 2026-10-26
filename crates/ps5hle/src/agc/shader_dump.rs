// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Shader capture for offline work on the shader front-end.
//!
//! With `RPS5E_DUMP_SHADERS=<dir>` set, every shader created through
//! `sceAgcCreateShader` is written as `<dir>/<hash>.bin` (the code blob,
//! `shader_size` bytes from the code address) plus `<dir>/<hash>.json` (the
//! shader object: stage, register tables, specials, user-data layout and
//! semantics). `<hash>` is the FNV-1a 64 hash of the code bytes, so a shader
//! created twice is written once. Fused GS/HS halves additionally produce
//! `<dir>/fused-<front>-<back>.json` naming the two code blobs.
//!
//! Shader object layout (0x60 bytes) as read here:
//!
//! | Offset | Field |
//! |---|---|
//! | +0x08 | user data `{u16* direct; u16* sharp[4]; u16 eud_dw, srt_dw, direct_count, sharp_count[4]}` |
//! | +0x10 | code |
//! | +0x18 / +0x20 | context / shader register tables `{u32 offset, u32 value}[]` |
//! | +0x28 | specials `{ge_cntl, vgt_shader_stages_en, u32 dispatch_modifier, u16 ud_start, u16 ud_end, u64 draw_modifier, vgt_gs_out_prim_type, ge_user_vgpr_en}` |
//! | +0x30 / +0x38 | input / output semantics (u32 each) |
//! | +0x40 | u32 header size, +0x44 u32 code size, +0x48 u32 embedded CB size (dqwords), +0x4C u32 target |
//! | +0x50 | u32 input count, +0x54 u16 scratch dwords/thread, +0x56 u16 output count, +0x58 u16 specials size |
//! | +0x5A | u8 type, +0x5B u8 context register count, +0x5C u8 shader register count |

use super::mem;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::OnceLock;

const MAX_CODE_BYTES: u32 = 1024 * 1024;
const MAX_SEMANTICS: u32 = 256;

fn dump_dir() -> Option<&'static PathBuf> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = PathBuf::from(std::env::var_os("RPS5E_DUMP_SHADERS").filter(|v| !v.is_empty())?);
        match std::fs::create_dir_all(&dir) {
            Ok(()) => {
                log::info!("[agc] dumping shaders to {}", dir.display());
                Some(dir)
            }
            Err(e) => {
                log::warn!("[agc] cannot create shader dump dir {}: {e}", dir.display());
                None
            }
        }
    })
    .as_ref()
}

/// FNV-1a 64-bit hash.
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3))
}

fn stage_name(ty: u8) -> &'static str {
    match ty {
        0 => "cs",
        1 => "ps",
        2 => "gs",
        3 => "hs",
        4 => "gs_front",
        5 => "hs_front",
        6 => "gs_back",
        7 => "hs_back",
        8 => "function",
        _ => "unknown",
    }
}

fn code_bytes(header: u64) -> Option<(u64, Vec<u8>)> {
    let code = mem::r64(header + 0x10)?;
    let size = mem::r32(header + 0x44)?;
    if code == 0 || size == 0 || size > MAX_CODE_BYTES {
        return None;
    }
    let mut buf = vec![0u8; size as usize];
    mem::read(code, &mut buf).then_some((code, buf))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn reg_list(out: &mut String, table: u64, count: u8) {
    out.push('[');
    if table != 0 {
        if let Some(raw) = mem::read_dwords(table, count as usize * 2) {
            for (i, c) in raw.as_chunks::<2>().0.iter().enumerate() {
                if i != 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "[{}, {}]", c[0], c[1]);
            }
        }
    }
    out.push(']');
}

fn u32_list(out: &mut String, addr: u64, count: u32) {
    out.push('[');
    if addr != 0 && count != 0 && count <= MAX_SEMANTICS {
        if let Some(raw) = mem::read_dwords(addr, count as usize) {
            for (i, v) in raw.iter().enumerate() {
                if i != 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "{v}");
            }
        }
    }
    out.push(']');
}

fn u16_list(addr: u64, count: u16) -> String {
    let mut out = String::from("[");
    for i in 0..count as u64 {
        if i != 0 {
            out.push_str(", ");
        }
        match mem::r16(addr + i * 2) {
            Some(v) => {
                let _ = write!(out, "{v}");
            }
            None => out.push_str("null"),
        }
    }
    out.push(']');
    out
}

fn register(addr: u64) -> String {
    match (mem::r32(addr), mem::r32(addr + 4)) {
        (Some(o), Some(v)) => format!("[{o}, {v}]"),
        _ => "null".into(),
    }
}

fn specials_json(addr: u64) -> String {
    if addr == 0 {
        return "null".into();
    }
    let dispatch = mem::r32(addr + 0x10).unwrap_or(0);
    let ud_start = mem::r16(addr + 0x14).unwrap_or(0);
    let ud_end = mem::r16(addr + 0x16).unwrap_or(0);
    let draw = mem::r64(addr + 0x18).unwrap_or(0);
    format!(
        "{{\"ge_cntl\": {}, \"vgt_shader_stages_en\": {}, \"dispatch_modifier\": {dispatch}, \
         \"user_data_range\": [{ud_start}, {ud_end}], \"draw_modifier\": {draw}, \
         \"vgt_gs_out_prim_type\": {}, \"ge_user_vgpr_en\": {}}}",
        register(addr),
        register(addr + 0x08),
        register(addr + 0x20),
        register(addr + 0x28)
    )
}

fn user_data_json(addr: u64) -> String {
    if addr == 0 {
        return "null".into();
    }
    let mut raw = [0u8; 0x36];
    if !mem::read(addr, &mut raw) {
        return "null".into();
    }
    let p = |o: usize| u64::from_le_bytes(raw[o..o + 8].try_into().unwrap());
    let h = |o: usize| u16::from_le_bytes([raw[o], raw[o + 1]]);
    let direct_count = h(0x2C);
    let direct = if p(0) != 0 && direct_count <= 64 { u16_list(p(0), direct_count) } else { "[]".into() };
    let mut sharps = String::from("[");
    let mut counts = String::from("[");
    for i in 0..4 {
        let n = h(0x2E + i * 2);
        let ptr = p(0x08 + i * 8);
        if i != 0 {
            sharps.push_str(", ");
            counts.push_str(", ");
        }
        let _ = write!(counts, "{n}");
        // Each entry: offset_dw in bits 14:0, size flag in bit 15.
        sharps.push_str(&if ptr != 0 && n <= 256 { u16_list(ptr, n) } else { "[]".into() });
    }
    sharps.push(']');
    counts.push(']');
    format!(
        "{{\"raw\": \"{}\", \"eud_size_dw\": {}, \"srt_size_dw\": {}, \"direct_resource_count\": {direct_count}, \
         \"direct_resource_offsets\": {direct}, \"sharp_resource_counts\": {counts}, \"sharp_resource_offsets\": {sharps}}}",
        hex(&raw),
        h(0x28),
        h(0x2A)
    )
}

/// Write `<hash>.bin` + `<hash>.json` for a successfully created shader.
pub(crate) fn on_create_shader(header: u64) {
    let Some(dir) = dump_dir() else { return };
    let Some((code, bytes)) = code_bytes(header) else {
        log::warn!("[agc] shader dump: unreadable code for header 0x{header:x}");
        return;
    };
    let hash = format!("{:016x}", fnv1a64(&bytes));
    let json_path = dir.join(format!("{hash}.json"));
    if json_path.exists() {
        return;
    }
    let mut hdr = [0u8; 0x60];
    if !mem::read(header, &mut hdr) {
        return;
    }
    let u32_at = |o: usize| u32::from_le_bytes(hdr[o..o + 4].try_into().unwrap());
    let u16_at = |o: usize| u16::from_le_bytes([hdr[o], hdr[o + 1]]);
    let u64_at = |o: usize| u64::from_le_bytes(hdr[o..o + 8].try_into().unwrap());
    let ty = hdr[0x5A];
    let mut j = String::new();
    let _ = writeln!(j, "{{");
    let _ = writeln!(j, "  \"hash\": \"{hash}\",");
    let _ = writeln!(j, "  \"type\": {ty},");
    let _ = writeln!(j, "  \"stage\": \"{}\",", stage_name(ty));
    let _ = writeln!(j, "  \"header_address\": {header},");
    let _ = writeln!(j, "  \"code_address\": {code},");
    let _ = writeln!(j, "  \"header_size\": {},", u32_at(0x40));
    let _ = writeln!(j, "  \"code_size\": {},", u32_at(0x44));
    let _ = writeln!(j, "  \"embedded_cb_size_dqw\": {},", u32_at(0x48));
    let _ = writeln!(j, "  \"target\": {},", u32_at(0x4C));
    let _ = writeln!(j, "  \"scratch_dw_per_thread\": {},", u16_at(0x54));
    let _ = writeln!(j, "  \"special_sizes_bytes\": {},", u16_at(0x58));
    let _ = writeln!(j, "  \"header_raw\": \"{}\",", hex(&hdr));
    j.push_str("  \"cx_registers\": ");
    reg_list(&mut j, u64_at(0x18), hdr[0x5B]);
    j.push_str(",\n  \"sh_registers\": ");
    reg_list(&mut j, u64_at(0x20), hdr[0x5C]);
    let _ = write!(j, ",\n  \"specials\": {},", specials_json(u64_at(0x28)));
    let _ = write!(j, "\n  \"user_data\": {},", user_data_json(u64_at(0x08)));
    j.push_str("\n  \"input_semantics\": ");
    u32_list(&mut j, u64_at(0x30), u32_at(0x50));
    j.push_str(",\n  \"output_semantics\": ");
    u32_list(&mut j, u64_at(0x38), u16_at(0x56) as u32);
    j.push_str("\n}\n");
    if let Err(e) = std::fs::write(dir.join(format!("{hash}.bin")), &bytes).and_then(|_| std::fs::write(&json_path, j)) {
        log::warn!("[agc] shader dump {hash}: {e}");
    }
}

/// Record a fused GS/HS pair: the front half's code ends in S_SETPC_B64 to
/// the back half.
pub(crate) fn on_fuse(front_header: u64, back_header: u64) {
    let Some(dir) = dump_dir() else { return };
    let (Some((fc, fb)), Some((bc, bb))) = (code_bytes(front_header), code_bytes(back_header)) else { return };
    let (fh, bh) = (fnv1a64(&fb), fnv1a64(&bb));
    let path = dir.join(format!("fused-{fh:016x}-{bh:016x}.json"));
    if path.exists() {
        return;
    }
    let j = format!(
        "{{\n  \"front\": \"{fh:016x}\",\n  \"back\": \"{bh:016x}\",\n  \"front_code_address\": {fc},\n  \"back_code_address\": {bc}\n}}\n"
    );
    if let Err(e) = std::fs::write(&path, j) {
        log::warn!("[agc] fused shader dump: {e}");
    }
}
