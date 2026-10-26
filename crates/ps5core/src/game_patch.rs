// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Game patches ("cheats"): byte writes into a loaded program, read from a
//! mods JSON file (`rps5e-cli run --game-patch FILE` / `RPS5E_GAME_PATCH`).
//!
//! File format:
//!
//! ```json
//! {"id": "PPSA01234", "version": "01.000.000", "process": "eboot.bin",
//!  "mods": [{"name": "Infinite health", "enabled": true,
//!            "memory": [{"offset": "0x4A2F10", "off": "8B4710", "on": "909090"}]}]}
//! ```
//!
//! `id`, `version` and `process` must match the loaded game (title id,
//! application version from param.json, executable file name). Mods with
//! `enabled: false` are skipped. Offsets are addresses in the address space
//! the patch author used: the program is located by searching the first
//! write whose `off` bytes are not all zero in the program's file-backed
//! segments; that gives the author's image base, and every write is
//! translated by it and must find its `off` bytes in place. Writes outside
//! the matched program must have all-zero `off` bytes: they go to "code
//! cave" pages (16 KiB, read/write/execute) allocated at that address.
//! When no loaded program matches, the plan waits for the next module
//! loaded at run time.

use std::path::Path;
use std::sync::Mutex;

/// Code-cave page size.
pub const CAVE_PAGE_SIZE: u64 = 0x4000;

/// Environment variable naming the patch file.
pub const PATCH_ENV: &str = "RPS5E_GAME_PATCH";

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Write {
    pub source_address: u64,
    pub target_address: u64,
    pub off: Vec<u8>,
    pub on: Vec<u8>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    pub title_id: String,
    pub version: String,
    pub process: String,
    pub writes: Vec<Write>,
    pub mod_names: Vec<String>,
    pub cave_pages: Vec<u64>,
}

/// A program as the patcher sees it: identity-mapped memory at `base`.
#[derive(Clone, Debug)]
pub struct Image {
    pub name: String,
    pub base: u64,
    /// Mapped size.
    pub size: u64,
    /// File-backed segments (PT_LOAD / PT_SCE_RELRO): (address, file size).
    pub segments: Vec<(u64, u64)>,
}

impl Image {
    pub fn of(m: &crate::LoadedModule) -> Self {
        Self {
            name: m.name.clone(),
            base: m.base,
            size: m.size,
            segments: m.segments.iter().map(|s| (s.addr, s.file_size)).collect(),
        }
    }

    fn contains(&self, address: u64, size: usize) -> bool {
        size != 0 && address >= self.base && address.saturating_add(size as u64) <= self.base + self.size
    }
}

/// Allocates and frees code-cave pages.
pub trait CaveAllocator {
    /// Map `CAVE_PAGE_SIZE` bytes read/write/execute exactly at `page`.
    fn allocate(&mut self, page: u64) -> bool;
    fn free(&mut self, page: u64);
}

fn parse_bytes(text: &str) -> Option<Vec<u8>> {
    if text.is_empty() || text.len() % 2 != 0 || !text.is_ascii() {
        return None;
    }
    (0..text.len()).step_by(2).map(|i| u8::from_str_radix(&text[i..i + 2], 16).ok()).collect()
}

fn parse_hex_u64(text: &str) -> Option<u64> {
    let t = text.trim();
    let t = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")).unwrap_or(t);
    u64::from_str_radix(t, 16).ok()
}

/// Parse a mods file into a plan (disabled mods dropped).
pub fn parse_plan(text: &str) -> Result<Plan, String> {
    let root: serde_json::Value = serde_json::from_str(text).map_err(|_| "expected a mods JSON file".to_string())?;
    let obj = root.as_object().ok_or("expected a mods JSON file")?;
    let s = |k: &str| obj.get(k).and_then(|v| v.as_str()).map(str::to_string);
    let (Some(title_id), Some(version), Some(process), Some(mods)) =
        (s("id"), s("version"), s("process"), obj.get("mods").and_then(|m| m.as_array()))
    else {
        return Err("expected a mods JSON file with id, version, process and mods".into());
    };
    let mut plan = Plan { title_id, version, process, ..Default::default() };
    for m in mods {
        if !m.get("enabled").and_then(|e| e.as_bool()).unwrap_or(true) {
            continue;
        }
        let name = m.get("name").and_then(|n| n.as_str()).ok_or("mod without a name")?;
        plan.mod_names.push(name.to_string());
        for entry in m.get("memory").and_then(|x| x.as_array()).map(Vec::as_slice).unwrap_or(&[]) {
            let field = |k: &str| entry.get(k).and_then(|v| v.as_str()).unwrap_or("");
            let source_address = parse_hex_u64(field("offset")).ok_or("invalid mod memory offset")?;
            let (Some(off), Some(on)) = (parse_bytes(field("off")), parse_bytes(field("on"))) else {
                return Err("invalid mod memory bytes".into());
            };
            if off.len() != on.len() {
                return Err("invalid mod memory bytes".into());
            }
            plan.writes.push(Write { source_address, target_address: 0, off, on });
        }
    }
    Ok(plan)
}

pub fn load_plan(path: &Path) -> Result<Plan, String> {
    let text = std::fs::read_to_string(path).map_err(|_| "could not read cheat file".to_string())?;
    parse_plan(&text)
}

/// The plan must name the loaded game.
pub fn validate_game(plan: &Plan, title_id: &str, version: &str, process: &str) -> Result<(), String> {
    if !process.eq_ignore_ascii_case(&plan.process) || !title_id.eq_ignore_ascii_case(&plan.title_id) || version != plan.version {
        return Err("cheat file does not match the loaded game".into());
    }
    Ok(())
}

fn is_zero(b: &[u8]) -> bool {
    b.iter().all(|&x| x == 0)
}

/// # Safety
/// `address..address+len` must be readable.
unsafe fn memory(address: u64, len: usize) -> &'static [u8] {
    std::slice::from_raw_parts(address as *const u8, len)
}

fn translate(image: &Image, source_base: u64, source: u64) -> u64 {
    if source >= source_base {
        image.base.wrapping_add(source - source_base)
    } else {
        image.base.wrapping_sub(source_base - source)
    }
}

fn translate_and_validate(plan: &mut Plan, image: &Image, source_base: u64) -> bool {
    for w in &mut plan.writes {
        w.target_address = translate(image, source_base, w.source_address);
        if !is_zero(&w.off)
            && (!image.contains(w.target_address, w.off.len())
                || unsafe { memory(w.target_address, w.off.len()) } != w.off.as_slice())
        {
            return false;
        }
    }
    true
}

/// Find the author's image base for `image`; translates the writes.
pub fn matches(plan: &mut Plan, image: &Image) -> Option<u64> {
    let anchor = plan.writes.iter().find(|w| !is_zero(&w.off))?.clone();
    if image.size < anchor.off.len() as u64 {
        return None;
    }
    let n = anchor.off.len();
    for &(addr, file_size) in &image.segments {
        if (file_size as usize) < n || !image.contains(addr, file_size as usize) {
            continue;
        }
        let data = unsafe { memory(addr, file_size as usize) };
        let mut from = 0;
        while from + n <= data.len() {
            let Some(pos) = data[from..].windows(n).position(|w| w == anchor.off.as_slice()) else { break };
            let found = from + pos;
            let rva = addr - image.base + found as u64;
            if anchor.source_address >= rva {
                let source_base = anchor.source_address - rva;
                if translate_and_validate(plan, image, source_base) {
                    return Some(source_base);
                }
            }
            from = found + 1;
        }
    }
    None
}

fn release_caves(pages: &mut Vec<u64>, caves: &mut dyn CaveAllocator) {
    for &p in pages.iter().rev() {
        caves.free(p);
    }
    pages.clear();
}

fn allocate_caves(plan: &mut Plan, image: &Image, caves: &mut dyn CaveAllocator) -> Result<(), String> {
    let mut pages = Vec::new();
    for w in &plan.writes {
        if image.contains(w.target_address, w.off.len()) {
            continue;
        }
        if !is_zero(&w.off) {
            return Err("non-zero write is outside the matched module".into());
        }
        let end = w.target_address + w.off.len() as u64;
        let mut page = w.target_address & !(CAVE_PAGE_SIZE - 1);
        while page < end {
            if !pages.contains(&page) {
                pages.push(page);
            }
            page += CAVE_PAGE_SIZE;
        }
    }
    for page in pages {
        if !caves.allocate(page) {
            release_caves(&mut plan.cave_pages, caves);
            return Err("could not allocate code-cave page".into());
        }
        plan.cave_pages.push(page);
    }
    Ok(())
}

/// Write the patch bytes: code caves first, then the program.
/// `write_program` writes into the program whatever its protection.
pub fn apply_writes(
    plan: &mut Plan,
    image: &Image,
    caves: &mut dyn CaveAllocator,
    write_program: &mut dyn FnMut(u64, &[u8]) -> bool,
) -> Result<(), String> {
    allocate_caves(plan, image, caves)?;
    for external in [true, false] {
        for w in &plan.writes {
            if !image.contains(w.target_address, w.on.len()) != external {
                continue;
            }
            let ok = if external {
                unsafe { std::ptr::copy_nonoverlapping(w.on.as_ptr(), w.target_address as *mut u8, w.on.len()) };
                true
            } else {
                write_program(w.target_address, &w.on)
            };
            if !ok {
                return Err("could not finalize cheat write".into());
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Process-wide plan
// ---------------------------------------------------------------------------

struct State {
    pending: Option<Plan>,
    applied_caves: Vec<u64>,
}

static STATE: Mutex<State> = Mutex::new(State { pending: None, applied_caves: Vec::new() });

/// Code caves in the guest address space.
pub struct GuestCaves<'a>(pub &'a ps5vm::VirtualMemory);

impl CaveAllocator for GuestCaves<'_> {
    fn allocate(&mut self, page: u64) -> bool {
        // A fixed mapping would replace whatever is there.
        if (0..CAVE_PAGE_SIZE).step_by(0x1000).any(|o| ps5vm::ranges::containing(page + o).is_some()) {
            return false;
        }
        if self.0.map(page, CAVE_PAGE_SIZE, ps5vm::MemProtect::RWX).is_err() {
            return false;
        }
        ps5cpu::exec::add_guest_code_range(page, CAVE_PAGE_SIZE, "game_cheat_code_cave");
        true
    }

    fn free(&mut self, page: u64) {
        ps5cpu::exec::remove_guest_code_ranges(page, CAVE_PAGE_SIZE);
        let _ = self.0.unmap(page, CAVE_PAGE_SIZE);
    }
}

/// The running title's identity: (title id, application version).
fn running_game() -> (String, String) {
    let title = ps5hle::app_info::title_id().unwrap_or_default();
    let version = ps5hle::app_info::read_param_json()
        .and_then(|j| ps5hle::app_info::param_json_string(&j, "appVersion"))
        .or_else(ps5hle::app_info::content_version)
        .unwrap_or_default();
    (title, version)
}

/// Load the plan at `path`, check it against the game and apply it to the
/// first matching program (else keep it pending for modules loaded later).
pub fn apply(path: &Path, vm: &ps5vm::VirtualMemory, programs: &[crate::LoadedModule]) -> Result<(), String> {
    let Some(main) = programs.first().filter(|m| m.base != 0) else {
        return Err("main executable is not loaded".into());
    };
    let plan = load_plan(path)?;
    let (title, version) = running_game();
    validate_game(&plan, &title, &version, &main.name)?;
    if plan.writes.is_empty() {
        return Ok(());
    }
    STATE.lock().unwrap_or_else(|e| e.into_inner()).pending = Some(plan);
    for p in programs {
        apply_pending(vm, p)?;
        if STATE.lock().unwrap_or_else(|e| e.into_inner()).pending.is_none() {
            return Ok(());
        }
    }
    log::info!("Game cheat: waiting for the matching module");
    Ok(())
}

/// Apply a pending plan to `program` when it matches.
pub fn apply_pending(vm: &ps5vm::VirtualMemory, program: &crate::LoadedModule) -> Result<(), String> {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(plan) = st.pending.as_mut() else { return Ok(()) };
    let image = Image::of(program);
    let Some(source_base) = matches(plan, &image) else { return Ok(()) };
    log::info!("Game cheat: matched {} with source base {source_base:#x}", image.name);
    let mut caves = GuestCaves(vm);
    if let Err(e) = apply_writes(plan, &image, &mut caves, &mut |a, b| program.patch(a, b)) {
        release_caves(&mut plan.cave_pages, &mut caves);
        st.pending = None;
        return Err(e);
    }
    for name in &plan.mod_names {
        log::info!("Successfully applied cheat: {name}");
    }
    let plan = st.pending.take().unwrap_or_default();
    st.applied_caves = plan.cave_pages;
    Ok(())
}

/// Drop the plan and free the code caves.
pub fn clear(vm: &ps5vm::VirtualMemory) {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let mut caves = GuestCaves(vm);
    if let Some(mut p) = st.pending.take() {
        release_caves(&mut p.cave_pages, &mut caves);
    }
    let mut applied = std::mem::take(&mut st.applied_caves);
    release_caves(&mut applied, &mut caves);
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE: &str = r#"{
        "name": "Test", "id": "PPSA01234", "version": "01.000.000", "process": "eboot.bin",
        "credits": ["someone"],
        "mods": [
          {"name": "Infinite health", "type": "checkbox",
           "memory": [{"offset": "0x1004", "on": "9090", "off": "8B47"},
                      {"offset": "1010", "on": "C3", "off": "55"}]},
          {"name": "Disabled", "enabled": false, "memory": [{"offset": "0x0", "on": "00", "off": "11"}]},
          {"name": "Cave", "enabled": true, "memory": [{"offset": "0x9000", "on": "CCCC", "off": "0000"}]}
        ]}"#;

    #[test]
    fn parses_enabled_mods() {
        let p = parse_plan(FILE).unwrap();
        assert_eq!((p.title_id.as_str(), p.version.as_str(), p.process.as_str()), ("PPSA01234", "01.000.000", "eboot.bin"));
        assert_eq!(p.mod_names, ["Infinite health", "Cave"]);
        assert_eq!(p.writes.len(), 3);
        assert_eq!(p.writes[0], Write { source_address: 0x1004, target_address: 0, off: vec![0x8B, 0x47], on: vec![0x90, 0x90] });
        assert_eq!(p.writes[1].source_address, 0x1010);
        assert!(parse_plan("[]").is_err());
        assert!(parse_plan(r#"{"id":"A","version":"1","process":"e","mods":[{"name":"x","memory":[{"offset":"1","on":"9","off":"8"}]}]}"#).is_err());
        assert!(parse_plan(r#"{"id":"A","version":"1","process":"e","mods":[{"name":"x","memory":[{"offset":"1","on":"9090","off":"88"}]}]}"#).is_err());
        assert!(parse_plan(r#"{"id":"A","version":"1","process":"e","mods":[{"name":"x","memory":[{"offset":"zz","on":"90","off":"88"}]}]}"#).is_err());
    }

    #[test]
    fn validates_the_game() {
        let p = parse_plan(FILE).unwrap();
        assert!(validate_game(&p, "ppsa01234", "01.000.000", "EBOOT.BIN").is_ok());
        assert!(validate_game(&p, "PPSA01235", "01.000.000", "eboot.bin").is_err());
        assert!(validate_game(&p, "PPSA01234", "01.000.001", "eboot.bin").is_err());
        assert!(validate_game(&p, "PPSA01234", "01.000.000", "game.elf").is_err());
    }

    struct TestCaves {
        fail: bool,
        freed: Vec<u64>,
    }

    impl CaveAllocator for TestCaves {
        fn allocate(&mut self, _page: u64) -> bool {
            !self.fail
        }
        fn free(&mut self, page: u64) {
            self.freed.push(page);
        }
    }

    /// A 16 KiB-aligned buffer standing in for a mapped program.
    fn program(bytes: &[(usize, &[u8])]) -> (Vec<u8>, Image) {
        let mut buf = vec![0u8; 0x8000 + 0x4000];
        let base = (buf.as_ptr() as u64 + 0x3FFF) & !0x3FFF;
        let off = (base - buf.as_ptr() as u64) as usize;
        for (at, b) in bytes {
            buf[off + at..off + at + b.len()].copy_from_slice(b);
        }
        let image = Image { name: "eboot.bin".into(), base, size: 0x8000, segments: vec![(base, 0x2000), (base + 0x4000, 0x1000)] };
        (buf, image)
    }

    #[test]
    fn finds_the_source_base_and_writes() {
        // The file's offsets assume the program at 0x1000.
        let (buf, image) = program(&[(0x4, &[0x8B, 0x47]), (0x10, &[0x55])]);
        let mut plan = parse_plan(FILE).unwrap();
        plan.writes.pop(); // no cave
        let base = matches(&mut plan, &image).expect("match");
        assert_eq!(base, 0x1000);
        assert_eq!(plan.writes[0].target_address, image.base + 4);
        let mut caves = TestCaves { fail: false, freed: Vec::new() };
        let mut written = Vec::new();
        apply_writes(&mut plan, &image, &mut caves, &mut |a, b| {
            written.push((a - image.base, b.to_vec()));
            true
        })
        .unwrap();
        assert_eq!(written, [(4, vec![0x90, 0x90]), (0x10, vec![0xC3])]);
        assert!(plan.cave_pages.is_empty() && caves.freed.is_empty());
        drop(buf);
    }

    #[test]
    fn mismatching_bytes_do_not_match() {
        let (_buf, image) = program(&[(0x4, &[0x8B, 0x47]), (0x10, &[0x56])]);
        let mut plan = parse_plan(FILE).unwrap();
        assert!(matches(&mut plan, &image).is_none());
        // Only zero `off` bytes: nothing to anchor on.
        let mut zero = parse_plan(r#"{"id":"A","version":"1","process":"e","mods":[{"name":"x","memory":[{"offset":"1","on":"90","off":"00"}]}]}"#).unwrap();
        assert!(matches(&mut zero, &image).is_none());
    }

    #[test]
    fn writes_outside_need_zero_bytes_and_a_cave() {
        let (_buf, image) = program(&[(0x4, &[0x8B, 0x47]), (0x10, &[0x55])]);
        let mut plan = parse_plan(FILE).unwrap();
        // 0x9000 - 0x1000 = 0x8000: just past the program.
        matches(&mut plan, &image).unwrap();
        let mut caves = TestCaves { fail: true, freed: Vec::new() };
        let err = apply_writes(&mut plan, &image, &mut caves, &mut |_, _| true).unwrap_err();
        assert!(err.contains("code-cave"), "{err}");
        assert!(plan.cave_pages.is_empty());
        // A non-zero write outside the program is refused.
        let mut plan = parse_plan(FILE).unwrap();
        matches(&mut plan, &image).unwrap();
        plan.writes[2].off = vec![1, 0];
        let err = apply_writes(&mut plan, &image, &mut caves, &mut |_, _| true).unwrap_err();
        assert!(err.contains("outside"), "{err}");
    }
}
