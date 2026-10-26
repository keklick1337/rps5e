// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libScePlayGo HLE: the title is always reported as fully installed.
//!
//! The number of chunks comes from the package metadata in the app root
//! (see [`set_app_root`]), tried in this order:
//! - `sce_sys/playgo-chunk.dat`: magic "plgo" (PS4 layout) or "plgx" (PS5
//!   packages), u16 chunk count at offset 10 (a zero count is no metadata);
//! - `playgo-chunkdefs.xml`: highest `<chunks default_chunk>` / `<chunk id>` + 1;
//! - `Data/chunkmanifest`: highest `\x08id\0` + u32 record + 1, ids dense from 0.
//! Chunk ids `0..count` are valid and installed (LOCAL_FAST). A package with
//! only `sce_sys/playgo-scenario.json` lists chunks 0 to 999. Without any
//! of them PlayGo is not supported (`scePlayGoOpen` fails), unless
//! `RPS5E_PLAYGO_HACK=1` asks for 1000 chunks at initialization.

use crate::context::CpuContext;
use crate::registry::{HleRegistry, HleResult};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const OK: HleResult = 0;

const fn err(code: u32) -> HleResult {
    code as i32 as i64
}

pub const ERR_INVALID_ARGUMENT: HleResult = err(0x80B2_0004);
pub const ERR_BAD_HANDLE: HleResult = err(0x80B2_0009);
pub const ERR_BAD_POINTER: HleResult = err(0x80B2_000A);
pub const ERR_BAD_SIZE: HleResult = err(0x80B2_000B);
pub const ERR_BAD_CHUNK_ID: HleResult = err(0x80B2_000C);
pub const ERR_BAD_SPEED: HleResult = err(0x80B2_000D);
pub const ERR_NOT_SUPPORT_PLAYGO: HleResult = err(0x80B2_000E);
pub const ERR_BAD_LOCUS: HleResult = err(0x80B2_0010);
pub const ERR_BAD_OPTIONAL_TYPE: HleResult = err(0x80B2_0024);

/// ScePlayGoInitParams: `const void* bufAddr; u32 bufSize; u32 reserved`.
const INIT_BUF_ADDR_OFFSET: u64 = 0;
const INIT_BUF_SIZE_OFFSET: u64 = 8;
const MIN_INIT_BUFFER_SIZE: u32 = 2 * 1024 * 1024;
pub const PLAYGO_HANDLE: i32 = 1;

const LOCUS_NOT_DOWNLOADED: i8 = 0;
const LOCUS_LOCAL_SLOW: i8 = 2;
const LOCUS_LOCAL_FAST: i8 = 3;

const INSTALL_SPEED_FULL: i32 = 2;

const OPTIONAL_TYPE_LANGUAGE: i32 = 0;
const OPTIONAL_TYPE_SCENARIO: i32 = 1;
const LANGUAGE_MASK_ALL: u64 = u64::MAX;
const SCENARIO_MASK_ALL: u64 = 0x1F;

/// Chunk count assumed by the PlayGo hack when the package has no metadata.
const DEFAULT_CHUNKS_NUM: u32 = 1000;
/// Highest chunk id + 1 accepted from chunk definitions and manifests.
const MAX_CHUNKS: u32 = 1000;
const PLAYGO_DAT_MAGIC: u32 = 0x6F67_6C70; // "plgo"
const PLAYGO_DAT_MAGIC_PS5: u32 = 0x7867_6C70; // "plgx"

/// ScePlayGoToDo: u16 chunkId; i8 locus; i8 reserved.
const TODO_SIZE: u64 = 4;
/// ScePlayGoProgress: u64 progressSize; u64 totalSize.
const PROGRESS_SIZE: usize = 16;

struct State {
    app_root: Option<PathBuf>,
    /// Number of chunks once known (0 = not loaded / no metadata).
    chunks_num: u32,
    /// scePlayGoSetLanguageMask's mask (every language until set).
    language_mask: u64,
}

static STATE: Mutex<State> = Mutex::new(State { app_root: None, chunks_num: 0, language_mask: LANGUAGE_MASK_ALL });

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Set the title's app0 root directory, whose package metadata gives the
/// chunk set.
pub fn set_app_root(path: impl Into<PathBuf>) {
    let mut st = state();
    st.app_root = Some(path.into());
    st.chunks_num = 0;
    st.language_mask = LANGUAGE_MASK_ALL;
}

fn playgo_hack_enabled() -> bool {
    std::env::var("RPS5E_PLAYGO_HACK").is_ok_and(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
}

// ---------------------------------------------------------------------------
// Package metadata
// ---------------------------------------------------------------------------

/// Chunk count from the package metadata below `root`.
pub fn load_chunks_num(root: &Path) -> Option<u32> {
    let sources: [(PathBuf, fn(&[u8]) -> Option<u32>); 3] = [
        (root.join("sce_sys").join("playgo-chunk.dat"), parse_playgo_dat),
        (root.join("playgo-chunkdefs.xml"), parse_chunkdefs),
        (root.join("Data").join("chunkmanifest"), parse_chunk_manifest),
    ];
    for (path, parse) in sources {
        let Ok(data) = std::fs::read(&path) else { continue };
        match parse(&data) {
            Some(n) => {
                log::info!("[PlayGo] chunks num = {n} (from {})", path.display());
                return Some(n);
            }
            None => log::warn!("[PlayGo] invalid chunk metadata: {}", path.display()),
        }
    }
    None
}

/// `playgo-chunk.dat`: u32 magic "plgo" or "plgx", u16 chunk count at
/// offset 10. A table without chunks defines nothing.
fn parse_playgo_dat(data: &[u8]) -> Option<u32> {
    let magic = u32::from_le_bytes(data.get(0..4)?.try_into().ok()?);
    if magic != PLAYGO_DAT_MAGIC && magic != PLAYGO_DAT_MAGIC_PS5 {
        return None;
    }
    let count = u16::from_le_bytes(data.get(10..12)?.try_into().ok()?) as u32;
    (count != 0).then_some(count)
}

/// Unsigned `attr = "N"` (or `'N'`) inside a tag; spaces/tabs around `=`.
fn parse_xml_uint_attr(tag: &str, attr: &str) -> Option<u32> {
    let at = tag.find(attr)?;
    let b = tag.as_bytes();
    let mut pos = at + attr.len();
    let skip = |pos: &mut usize| {
        while *pos < b.len() && (b[*pos] == b' ' || b[*pos] == b'\t') {
            *pos += 1;
        }
    };
    skip(&mut pos);
    if b.get(pos) != Some(&b'=') {
        return None;
    }
    pos += 1;
    skip(&mut pos);
    let quote = *b.get(pos)?;
    if quote != b'"' && quote != b'\'' {
        return None;
    }
    pos += 1;
    let start = pos;
    while pos < b.len() && b[pos].is_ascii_digit() {
        pos += 1;
    }
    if start == pos || b.get(pos) != Some(&quote) {
        return None;
    }
    tag[start..pos].parse().ok()
}

/// `playgo-chunkdefs.xml`: highest of `<chunks default_chunk="N">` and every
/// `<chunk ... id="N">`, plus one.
fn parse_chunkdefs(data: &[u8]) -> Option<u32> {
    if data.is_empty() {
        return None;
    }
    let xml = String::from_utf8_lossy(data);
    let mut max_id = 0u32;
    let mut found = false;
    if let Some(p) = xml.find("<chunks") {
        if let Some(end) = xml[p..].find('>') {
            if let Some(id) = parse_xml_uint_attr(&xml[p..p + end], "default_chunk") {
                max_id = max_id.max(id);
                found = true;
            }
        }
    }
    let mut pos = 0;
    while let Some(p) = xml[pos..].find("<chunk") {
        let start = pos + p;
        let name_end = start + 6;
        if !matches!(xml.as_bytes().get(name_end), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            pos = name_end;
            continue;
        }
        let Some(end) = xml[name_end..].find('>') else { break };
        let tag_end = name_end + end;
        if let Some(id) = parse_xml_uint_attr(&xml[start..tag_end], "id") {
            max_id = max_id.max(id);
            found = true;
        }
        pos = tag_end + 1;
    }
    (found && max_id < MAX_CHUNKS).then_some(max_id + 1)
}

/// `Data/chunkmanifest`: binary records `\x08 'i' 'd' \0` followed by a u32
/// chunk id. Ids must cover `0..=max` without gaps.
fn parse_chunk_manifest(data: &[u8]) -> Option<u32> {
    const ID_FIELD: [u8; 4] = [0x08, b'i', b'd', 0x00];
    let mut seen = [false; MAX_CHUNKS as usize];
    let mut max_id = 0u32;
    let mut found = false;
    let mut pos = 0;
    while pos + ID_FIELD.len() + 4 <= data.len() {
        if data[pos..pos + 4] == ID_FIELD {
            let v = &data[pos + 4..pos + 8];
            let id = u32::from_le_bytes([v[0], v[1], v[2], v[3]]);
            if id < MAX_CHUNKS {
                seen[id as usize] = true;
                max_id = max_id.max(id);
                found = true;
            }
        }
        pos += 1;
    }
    (found && seen[..=max_id as usize].iter().all(|&s| s)).then_some(max_id + 1)
}

/// The chunk count, loading the metadata on first use.
fn ensure_chunks_loaded() -> Option<u32> {
    let mut st = state();
    if st.chunks_num == 0 {
        if let Some(n) = st.app_root.as_deref().and_then(load_chunks_num) {
            st.chunks_num = n;
        } else if st.app_root.as_deref().is_some_and(|r| r.join("sce_sys").join("playgo-scenario.json").is_file()) {
            // Only a scenario file: chunks 0 to 999.
            st.chunks_num = DEFAULT_CHUNKS_NUM;
        }
    }
    (st.chunks_num != 0).then_some(st.chunks_num)
}

fn is_valid_chunk(id: u16) -> bool {
    ensure_chunks_loaded().is_some_and(|n| (id as u32) < n)
}

fn is_valid_locus(locus: i8) -> bool {
    matches!(locus, LOCUS_NOT_DOWNLOADED | LOCUS_LOCAL_SLOW | LOCUS_LOCAL_FAST)
}

fn validate_handle(handle: i32) -> Result<(), HleResult> {
    if handle == PLAYGO_HANDLE { Ok(()) } else { Err(ERR_BAD_HANDLE) }
}

fn validate_optional_type(kind: i32) -> Result<(), HleResult> {
    if matches!(kind, OPTIONAL_TYPE_LANGUAGE | OPTIONAL_TYPE_SCENARIO) {
        Ok(())
    } else {
        Err(ERR_BAD_OPTIONAL_TYPE)
    }
}

fn read_u16(ctx: &CpuContext, addr: u64) -> Result<u16, HleResult> {
    let b = ctx.read_bytes(addr, 2).ok_or(ERR_BAD_POINTER)?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}

/// Every id of the `n`-entry chunk list must be a valid chunk.
fn validate_chunk_list(ctx: &CpuContext, list: u64, n: u32) -> Result<(), HleResult> {
    for i in 0..n as u64 {
        if !is_valid_chunk(read_u16(ctx, list + i * 2)?) {
            return Err(ERR_BAD_CHUNK_ID);
        }
    }
    Ok(())
}

fn put(ok: bool) -> Result<(), HleResult> {
    if ok { Ok(()) } else { Err(ERR_BAD_POINTER) }
}

fn status(r: Result<(), HleResult>) -> HleResult {
    r.err().unwrap_or(OK)
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

/// scePlayGoInitialize(const ScePlayGoInitParams*)
fn cb_initialize(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        let params = ctx.rdi();
        if params == 0 {
            return Err(ERR_BAD_POINTER);
        }
        let buf_addr = ctx.read_u64(params + INIT_BUF_ADDR_OFFSET).ok_or(ERR_BAD_POINTER)?;
        let buf_size = ctx.read_u32(params + INIT_BUF_SIZE_OFFSET).ok_or(ERR_BAD_POINTER)?;
        if buf_addr == 0 {
            return Err(ERR_BAD_POINTER);
        }
        if buf_size < MIN_INIT_BUFFER_SIZE {
            return Err(ERR_BAD_SIZE);
        }
        if playgo_hack_enabled() && ensure_chunks_loaded().is_none() {
            state().chunks_num = DEFAULT_CHUNKS_NUM;
            log::info!("[PlayGo] hack: {DEFAULT_CHUNKS_NUM} chunks");
        }
        Ok(())
    })())
}

fn cb_terminate(_ctx: &mut CpuContext) -> HleResult {
    OK
}

/// scePlayGoOpen(ScePlayGoHandle* outHandle, const void* param)
fn cb_open(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        let (out_handle, param) = (ctx.rdi(), ctx.rsi());
        if out_handle == 0 {
            return Err(ERR_BAD_POINTER);
        }
        if param != 0 {
            return Err(ERR_INVALID_ARGUMENT);
        }
        if ensure_chunks_loaded().is_none() {
            return Err(ERR_NOT_SUPPORT_PLAYGO);
        }
        put(ctx.write_u32(out_handle, PLAYGO_HANDLE as u32))
    })())
}

fn cb_close(ctx: &mut CpuContext) -> HleResult {
    status(validate_handle(ctx.rdi() as i32))
}

/// scePlayGoGetLocus(handle, const u16* chunkIds, u32 n, i8* outLoci)
fn cb_get_locus(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        let (handle, list, n, out) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as u32, ctx.rcx());
        validate_handle(handle)?;
        if list == 0 || out == 0 {
            return Err(ERR_BAD_POINTER);
        }
        if n == 0 {
            return Err(ERR_BAD_SIZE);
        }
        for i in 0..n as u64 {
            if !is_valid_chunk(read_u16(ctx, list + i * 2)?) {
                return Err(ERR_BAD_CHUNK_ID);
            }
            put(ctx.write_bytes(out + i, &[LOCUS_LOCAL_FAST as u8]))?;
        }
        Ok(())
    })())
}

/// scePlayGoGetInstallSpeed(handle, i32* outSpeed): always full speed.
fn cb_get_install_speed(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        validate_handle(ctx.rdi() as i32)?;
        let out = ctx.rsi();
        if out == 0 {
            return Err(ERR_BAD_POINTER);
        }
        put(ctx.write_u32(out, INSTALL_SPEED_FULL as u32))
    })())
}

/// scePlayGoSetInstallSpeed(handle, i32 speed): 0..=2.
fn cb_set_install_speed(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        validate_handle(ctx.rdi() as i32)?;
        let speed = ctx.rsi() as i32;
        if !(0..=2).contains(&speed) {
            return Err(ERR_BAD_SPEED);
        }
        Ok(())
    })())
}

/// scePlayGoGetChunkId(handle, u16* outList, u32 n, u32* outEntries): ids
/// `0..min(n, chunks)`.
fn cb_get_chunk_id(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        let (handle, out_list, n, out_entries) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as u32, ctx.rcx());
        validate_handle(handle)?;
        if out_entries == 0 {
            return Err(ERR_BAD_POINTER);
        }
        if n != 0 && out_list == 0 {
            return Err(ERR_BAD_POINTER);
        }
        let entries = n.min(ensure_chunks_loaded().unwrap_or(0));
        let ids: Vec<u8> = (0..entries as u16).flat_map(u16::to_le_bytes).collect();
        if !ids.is_empty() {
            put(ctx.write_bytes(out_list, &ids))?;
        }
        put(ctx.write_u32(out_entries, entries))
    })())
}

/// scePlayGoGetToDoList(handle, ScePlayGoToDo* outList, u32 n, u32* outEntries):
/// nothing left to install.
fn cb_get_todo_list(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        let (handle, out_list, n, out_entries) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as u32, ctx.rcx());
        validate_handle(handle)?;
        if out_entries == 0 {
            return Err(ERR_BAD_POINTER);
        }
        if n != 0 && out_list == 0 {
            return Err(ERR_BAD_POINTER);
        }
        put(ctx.write_u32(out_entries, 0))
    })())
}

/// scePlayGoSetToDoList(handle, const ScePlayGoToDo* list, u32 n)
fn cb_set_todo_list(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        let (handle, list, n) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as u32);
        validate_handle(handle)?;
        if list == 0 {
            return Err(ERR_BAD_POINTER);
        }
        if n == 0 {
            return Err(ERR_BAD_SIZE);
        }
        for i in 0..n as u64 {
            let todo = ctx.read_bytes(list + i * TODO_SIZE, TODO_SIZE as usize).ok_or(ERR_BAD_POINTER)?;
            if !is_valid_chunk(u16::from_le_bytes([todo[0], todo[1]])) {
                return Err(ERR_BAD_CHUNK_ID);
            }
            if !is_valid_locus(todo[2] as i8) {
                return Err(ERR_BAD_LOCUS);
            }
        }
        Ok(())
    })())
}

/// scePlayGoPrefetch(handle, const u16* chunkIds, u32 n, i8 minimumLocus)
fn cb_prefetch(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        let (handle, list, n, locus) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as u32, ctx.rcx() as i8);
        validate_handle(handle)?;
        if list == 0 {
            return Err(ERR_BAD_POINTER);
        }
        if n == 0 {
            return Err(ERR_BAD_SIZE);
        }
        if !is_valid_locus(locus) {
            return Err(ERR_BAD_LOCUS);
        }
        validate_chunk_list(ctx, list, n)
    })())
}

/// scePlayGoGetEta(handle, const u16* chunkIds, u32 n, i64* outEta): 0.
fn cb_get_eta(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        let (handle, list, n, out) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as u32, ctx.rcx());
        validate_handle(handle)?;
        if list == 0 || out == 0 {
            return Err(ERR_BAD_POINTER);
        }
        if n == 0 {
            return Err(ERR_BAD_SIZE);
        }
        validate_chunk_list(ctx, list, n)?;
        put(ctx.write_u64(out, 0))
    })())
}

/// scePlayGoGetLanguageMask(handle, u64* outMask): the mask last set
/// (every language at first).
fn cb_get_language_mask(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        validate_handle(ctx.rdi() as i32)?;
        let out = ctx.rsi();
        if out == 0 {
            return Err(ERR_BAD_POINTER);
        }
        let mask = state().language_mask;
        put(ctx.write_u64(out, mask))
    })())
}

/// scePlayGoSetLanguageMask(handle, u64 mask): the mask is kept; every
/// language stays installed.
fn cb_set_language_mask(ctx: &mut CpuContext) -> HleResult {
    status(validate_handle(ctx.rdi() as i32).map(|()| state().language_mask = ctx.rsi()))
}

/// scePlayGoGetProgress(handle, const u16* chunkIds, u32 n, ScePlayGoProgress*):
/// progress = total = n.
fn cb_get_progress(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        let (handle, list, n, out) = (ctx.rdi() as i32, ctx.rsi(), ctx.rdx() as u32, ctx.rcx());
        validate_handle(handle)?;
        if list == 0 || out == 0 {
            return Err(ERR_BAD_POINTER);
        }
        if n == 0 {
            return Err(ERR_BAD_SIZE);
        }
        validate_chunk_list(ctx, list, n)?;
        let mut progress = [0u8; PROGRESS_SIZE];
        progress[..8].copy_from_slice(&(n as u64).to_le_bytes());
        progress[8..].copy_from_slice(&(n as u64).to_le_bytes());
        put(ctx.write_bytes(out, &progress))
    })())
}

/// scePlayGoGetOptionalChunk / GetSupportedOptionalChunk(handle, type,
/// ScePlayGoOptionalChunk* out): every language or scenario.
fn cb_get_optional_chunk(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        let (handle, kind, out) = (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx());
        validate_handle(handle)?;
        if out == 0 {
            return Err(ERR_BAD_POINTER);
        }
        validate_optional_type(kind)?;
        let mask = if kind == OPTIONAL_TYPE_LANGUAGE { LANGUAGE_MASK_ALL } else { SCENARIO_MASK_ALL };
        put(ctx.write_u64(out, mask))
    })())
}

/// scePlayGoPrefetchOptionalChunk(handle, type, const ScePlayGoOptionalChunk*)
fn cb_prefetch_optional_chunk(ctx: &mut CpuContext) -> HleResult {
    status((|| {
        let (handle, kind, option) = (ctx.rdi() as i32, ctx.rsi() as i32, ctx.rdx());
        validate_handle(handle)?;
        if option == 0 {
            return Err(ERR_BAD_POINTER);
        }
        validate_optional_type(kind)
    })())
}

const EXPORTS: &[(&str, &str, crate::registry::HleCallback)] = &[
    ("ts6GlZOKRrE", "scePlayGoInitialize", cb_initialize),
    ("MPe0EeBGM-E", "scePlayGoTerminate", cb_terminate),
    ("M1Gma1ocrGE", "scePlayGoOpen", cb_open),
    ("Uco1I0dlDi8", "scePlayGoClose", cb_close),
    ("uWIYLFkkwqk", "scePlayGoGetLocus", cb_get_locus),
    ("rvBSfTimejE", "scePlayGoGetInstallSpeed", cb_get_install_speed),
    ("4AAcTU9R3XM", "scePlayGoSetInstallSpeed", cb_set_install_speed),
    ("73fF1MFU8hA", "scePlayGoGetChunkId", cb_get_chunk_id),
    ("Nn7zKwnA5q0", "scePlayGoGetToDoList", cb_get_todo_list),
    ("gUPGiOQ1tmQ", "scePlayGoSetToDoList", cb_set_todo_list),
    ("-Q1-u1a7p0g", "scePlayGoPrefetch", cb_prefetch),
    ("v6EZ-YWRdMs", "scePlayGoGetEta", cb_get_eta),
    ("3OMbYZBaa50", "scePlayGoGetLanguageMask", cb_get_language_mask),
    ("LosLlHOpNqQ", "scePlayGoSetLanguageMask", cb_set_language_mask),
    ("-RJWNMK3fC8", "scePlayGoGetProgress", cb_get_progress),
    ("g4AZyxpSAlA", "scePlayGoGetOptionalChunk", cb_get_optional_chunk),
    ("HVAa744ecdw", "scePlayGoPrefetchOptionalChunk", cb_prefetch_optional_chunk),
    ("8-e7E989rCU", "scePlayGoGetInstallChunkId", cb_get_chunk_id),
    ("IfiN+-oeVWI", "scePlayGoGetSupportedOptionalChunk", cb_get_optional_chunk),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, "libScePlayGo", cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    #[test]
    fn playgo_dat() {
        let mut dat = b"plgo".to_vec();
        dat.extend_from_slice(&[0; 6]);
        dat.extend_from_slice(&7u16.to_le_bytes());
        assert_eq!(parse_playgo_dat(&dat), Some(7));
        // PS5 packages carry the same header under "plgx".
        dat[3] = b'x';
        assert_eq!(parse_playgo_dat(&dat), Some(7));
        dat[0] = b'x';
        assert_eq!(parse_playgo_dat(&dat), None);
        assert_eq!(parse_playgo_dat(b"plgo"), None);
        // No chunks: no chunk set.
        assert_eq!(parse_playgo_dat(b"plgx\0\0\0\0\0\0\0\0"), None);
    }

    #[test]
    fn chunkdefs_parsing() {
        let xml = r#"<?xml version="1.0"?>
<psproject><volume><chunk_info chunk_count="3" scenario_count="1">
  <chunks default_chunk = "0">
    <chunk id="0" label="base"/>
    <chunk label="extra" id = '5' />
    <chunk	layer_no="0" id="2"></chunk>
    <chunks_other id="77"/>
  </chunks>
</chunk_info></volume></psproject>"#;
        assert_eq!(parse_chunkdefs(xml.as_bytes()), Some(6));
        assert_eq!(parse_chunkdefs(b"<root/>"), None);
        assert_eq!(parse_chunkdefs(b""), None);
        assert_eq!(parse_chunkdefs(br#"<chunks default_chunk="1000">"#), None, "ids stay below 1000");
    }

    #[test]
    fn chunk_manifest_parsing() {
        let record = |id: u32| {
            let mut r = vec![0x08, b'i', b'd', 0];
            r.extend_from_slice(&id.to_le_bytes());
            r.extend_from_slice(b"xx");
            r
        };
        let dense: Vec<u8> = [1, 0, 2].into_iter().flat_map(record).collect();
        assert_eq!(parse_chunk_manifest(&dense), Some(3));
        let gap: Vec<u8> = [0, 2].into_iter().flat_map(record).collect();
        assert_eq!(parse_chunk_manifest(&gap), None, "chunk 1 missing");
        assert_eq!(parse_chunk_manifest(b"nothing"), None);
    }

    #[test]
    fn metadata_sources_in_order() {
        let dir = std::env::temp_dir().join(format!("ps5hle_playgo_test_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sce_sys")).unwrap();
        std::fs::create_dir_all(dir.join("Data")).unwrap();
        assert_eq!(load_chunks_num(&dir), None, "no metadata: PlayGo unsupported");
        let mut manifest = vec![0x08, b'i', b'd', 0];
        manifest.extend_from_slice(&0u32.to_le_bytes());
        std::fs::write(dir.join("Data").join("chunkmanifest"), &manifest).unwrap();
        assert_eq!(load_chunks_num(&dir), Some(1));
        std::fs::write(dir.join("playgo-chunkdefs.xml"), br#"<chunks default_chunk="0"><chunk id="3"/>"#).unwrap();
        assert_eq!(load_chunks_num(&dir), Some(4));
        std::fs::write(dir.join("sce_sys").join("playgo-chunk.dat"), b"plgo\0\0\0\0\0\0\x09\0").unwrap();
        assert_eq!(load_chunks_num(&dir), Some(9));
        // A corrupt first source falls through to the next one.
        std::fs::write(dir.join("sce_sys").join("playgo-chunk.dat"), b"junk").unwrap();
        assert_eq!(load_chunks_num(&dir), Some(4));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn chunk_queries() {
        let dir = std::env::temp_dir().join(format!("ps5hle_playgo_chunks_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sce_sys")).unwrap();
        // A PS5 package chunk table: "plgx", 3 chunks at 0x0A, 0x100 bytes.
        let mut header = [0u8; 0x100];
        header[..4].copy_from_slice(b"plgx");
        header[0x0A] = 3;
        std::fs::write(dir.join("sce_sys").join("playgo-chunk.dat"), header).unwrap();
        set_app_root(&dir);

        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D40_0500_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);
        ctx.set(Rdi, BASE);
        ctx.set(Rsi, 0);
        assert_eq!(cb_open(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE), Some(PLAYGO_HANDLE as u32));

        let ids = BASE + 0x100;
        let loci = BASE + 0x200;
        ctx.write_bytes(ids, &[0, 0, 2, 0, 3, 0]);
        ctx.write_bytes(loci, &[0xFF; 3]);
        ctx.set(Rdi, PLAYGO_HANDLE as u64);
        ctx.set(Rsi, ids);
        ctx.set(Rdx, 2);
        ctx.set(Rcx, loci);
        assert_eq!(cb_get_locus(&mut ctx), OK);
        assert_eq!(ctx.read_bytes(loci, 3).unwrap(), vec![3, 3, 0xFF]);
        ctx.set(Rdx, 3);
        assert_eq!(cb_get_locus(&mut ctx), ERR_BAD_CHUNK_ID, "chunk 3 does not exist");
        assert_eq!(cb_get_eta(&mut ctx), ERR_BAD_CHUNK_ID);
        ctx.set(Rdx, 2);
        assert_eq!(cb_get_progress(&mut ctx), OK);
        assert_eq!(ctx.read_u64(loci), Some(2));
        assert_eq!(ctx.read_u64(loci + 8), Some(2));
        ctx.set(Rcx, LOCUS_LOCAL_SLOW as u64);
        assert_eq!(cb_prefetch(&mut ctx), OK);
        ctx.set(Rcx, 1);
        assert_eq!(cb_prefetch(&mut ctx), ERR_BAD_LOCUS);

        // Chunk id list: min(requested, chunks).
        let out = BASE + 0x300;
        let entries = BASE + 0x400;
        ctx.set(Rsi, out);
        ctx.set(Rdx, 8);
        ctx.set(Rcx, entries);
        assert_eq!(cb_get_chunk_id(&mut ctx), OK);
        assert_eq!(ctx.read_u32(entries), Some(3));
        assert_eq!(ctx.read_bytes(out, 6).unwrap(), vec![0, 0, 1, 0, 2, 0]);

        // To-do list entries: chunk, then locus.
        let todo = BASE + 0x500;
        ctx.write_bytes(todo, &[1, 0, 3, 0, 2, 0, 9, 0]);
        ctx.set(Rsi, todo);
        ctx.set(Rdx, 1);
        assert_eq!(cb_set_todo_list(&mut ctx), OK);
        ctx.set(Rdx, 2);
        assert_eq!(cb_set_todo_list(&mut ctx), ERR_BAD_LOCUS);
        ctx.write_bytes(todo + 4, &[7, 0, 3, 0]);
        assert_eq!(cb_set_todo_list(&mut ctx), ERR_BAD_CHUNK_ID);

        // The language mask reads back what was set.
        ctx.set(Rsi, 0x5);
        assert_eq!(cb_set_language_mask(&mut ctx), OK);
        ctx.set(Rsi, out);
        assert_eq!(cb_get_language_mask(&mut ctx), OK);
        assert_eq!(ctx.read_u64(out), Some(0x5));
        ctx.set(Rdi, 7);
        assert_eq!(cb_set_language_mask(&mut ctx), ERR_BAD_HANDLE);
        ctx.set(Rdi, PLAYGO_HANDLE as u64);

        // Only a scenario file: chunks 0 to 999.
        std::fs::remove_file(dir.join("sce_sys").join("playgo-chunk.dat")).unwrap();
        std::fs::write(dir.join("sce_sys").join("playgo-scenario.json"), b"{}").unwrap();
        set_app_root(&dir);
        ctx.set(Rsi, out);
        assert_eq!(cb_get_language_mask(&mut ctx), OK);
        assert_eq!(ctx.read_u64(out), Some(LANGUAGE_MASK_ALL), "a new title starts with every language");
        ctx.set(Rdx, 8);
        ctx.set(Rcx, entries);
        assert_eq!(cb_get_chunk_id(&mut ctx), OK);
        assert_eq!(ctx.read_u32(entries), Some(8));
        assert!(is_valid_chunk(999) && !is_valid_chunk(1000));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Handle-level argument checks that do not depend on the chunk set.
    #[test]
    fn argument_validation() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D40_0400_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);

        // Initialize: pointer, then buffer size.
        ctx.set(Rdi, 0);
        assert_eq!(cb_initialize(&mut ctx), ERR_BAD_POINTER);
        ctx.write_u64(BASE, 0);
        ctx.write_u32(BASE + 8, 4 << 20);
        ctx.set(Rdi, BASE);
        assert_eq!(cb_initialize(&mut ctx), ERR_BAD_POINTER);
        ctx.write_u64(BASE, BASE + 0x100);
        ctx.write_u32(BASE + 8, (2 << 20) - 1);
        assert_eq!(cb_initialize(&mut ctx), ERR_BAD_SIZE);

        // Open: pointer, then param.
        ctx.set(Rdi, 0);
        assert_eq!(cb_open(&mut ctx), ERR_BAD_POINTER);
        ctx.set(Rdi, BASE);
        ctx.set(Rsi, 1);
        assert_eq!(cb_open(&mut ctx), ERR_INVALID_ARGUMENT);

        ctx.set(Rdi, 2);
        assert_eq!(cb_close(&mut ctx), ERR_BAD_HANDLE);
        ctx.set(Rdi, PLAYGO_HANDLE as u64);
        assert_eq!(cb_close(&mut ctx), OK);
        assert_eq!(cb_terminate(&mut ctx), OK);

        ctx.set(Rsi, BASE);
        assert_eq!(cb_get_install_speed(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE), Some(2));
        ctx.set(Rsi, 3);
        assert_eq!(cb_set_install_speed(&mut ctx), ERR_BAD_SPEED);
        ctx.set(Rsi, (-1i32) as u32 as u64);
        assert_eq!(cb_set_install_speed(&mut ctx), ERR_BAD_SPEED);
        ctx.set(Rsi, 0);
        assert_eq!(cb_set_install_speed(&mut ctx), OK);

        ctx.set(Rsi, BASE);
        assert_eq!(cb_get_language_mask(&mut ctx), OK);
        assert_eq!(ctx.read_u64(BASE), Some(u64::MAX));

        // Optional chunks: pointer before type.
        ctx.set(Rsi, OPTIONAL_TYPE_SCENARIO as u64);
        ctx.set(Rdx, 0);
        assert_eq!(cb_get_optional_chunk(&mut ctx), ERR_BAD_POINTER);
        ctx.set(Rdx, BASE);
        assert_eq!(cb_get_optional_chunk(&mut ctx), OK);
        assert_eq!(ctx.read_u64(BASE), Some(0x1F));
        ctx.set(Rsi, OPTIONAL_TYPE_LANGUAGE as u64);
        assert_eq!(cb_get_optional_chunk(&mut ctx), OK);
        assert_eq!(ctx.read_u64(BASE), Some(u64::MAX));
        ctx.set(Rsi, 2);
        assert_eq!(cb_get_optional_chunk(&mut ctx), ERR_BAD_OPTIONAL_TYPE);
        assert_eq!(cb_prefetch_optional_chunk(&mut ctx), ERR_BAD_OPTIONAL_TYPE);

        // Lists: pointers, then size; to-do list with no entries requested.
        ctx.set(Rsi, BASE);
        ctx.set(Rdx, 0);
        ctx.set(Rcx, BASE + 0x10);
        assert_eq!(cb_get_locus(&mut ctx), ERR_BAD_SIZE);
        assert_eq!(cb_get_eta(&mut ctx), ERR_BAD_SIZE);
        assert_eq!(cb_get_progress(&mut ctx), ERR_BAD_SIZE);
        assert_eq!(cb_prefetch(&mut ctx), ERR_BAD_SIZE);
        ctx.set(Rsi, 0);
        assert_eq!(cb_get_locus(&mut ctx), ERR_BAD_POINTER);
        ctx.write_u32(BASE + 0x10, 9);
        assert_eq!(cb_get_todo_list(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE + 0x10), Some(0));
        ctx.set(Rdx, 1);
        assert_eq!(cb_get_todo_list(&mut ctx), ERR_BAD_POINTER);
        assert_eq!(cb_get_chunk_id(&mut ctx), ERR_BAD_POINTER);
        ctx.set(Rcx, 0);
        assert_eq!(cb_get_chunk_id(&mut ctx), ERR_BAD_POINTER);
    }

    /// The handle is checked before any pointer, and a rejected call leaves
    /// the output untouched.
    #[test]
    fn speed_and_eta_check_the_handle_first() {
        let vm = ps5vm::VirtualMemory::new();
        const BASE: u64 = 0x5D4F_0700_0000;
        vm.map(BASE, 0x1000, ps5vm::MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(&vm);

        ctx.set(Rdi, (-1i32) as u32 as u64);
        ctx.set(Rsi, 0);
        ctx.set(Rdx, 0);
        ctx.set(Rcx, BASE);
        assert_eq!(cb_get_eta(&mut ctx), ERR_BAD_HANDLE);
        assert_eq!(cb_prefetch(&mut ctx), ERR_BAD_HANDLE);
        ctx.set(Rdi, PLAYGO_HANDLE as u64);
        ctx.set(Rcx, 0);
        assert_eq!(cb_get_eta(&mut ctx), ERR_BAD_POINTER);

        ctx.write_u32(BASE, u32::MAX);
        for handle in [(-1i32) as u32 as u64, 0x506C_476F] {
            ctx.set(Rdi, handle);
            ctx.set(Rsi, BASE);
            assert_eq!(cb_get_install_speed(&mut ctx), ERR_BAD_HANDLE);
            assert_eq!(ctx.read_u32(BASE), Some(u32::MAX));
        }
        ctx.set(Rdi, PLAYGO_HANDLE as u64);
        ctx.set(Rsi, 0);
        assert_eq!(cb_get_install_speed(&mut ctx), ERR_BAD_POINTER);
        ctx.set(Rsi, BASE);
        assert_eq!(cb_get_install_speed(&mut ctx), OK);
        assert_eq!(ctx.read_u32(BASE), Some(2));
    }
}
