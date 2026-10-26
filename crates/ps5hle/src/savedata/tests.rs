// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! SaveData tests: guest structures live in scratch guest memory, saves in
//! a private temporary root (`<tmp>/_SaveData`).

use super::*;
use std::cell::Cell;
#[cfg_attr(windows, allow(unused_imports))]
use std::time::Duration;

fn temp_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("ps5hle-savedata-{tag}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

const SCRATCH: u64 = 0x40_0000;

struct Harness<'a> {
    ctx: CpuContext<'a>,
    lib: SaveDataLib,
    base: u64,
    next: Cell<u64>,
    tmp: PathBuf,
}

impl<'a> Harness<'a> {
    fn new(vm: &'a ps5vm::VirtualMemory, base: u64, tmp: &Path, title: &str) -> Self {
        vm.map(base, SCRATCH, ps5vm::MemProtect::RW).unwrap();
        let mut lib = SaveDataLib::new(Some(tmp.join("_SaveData")), false);
        lib.title_id = Some(title.to_string());
        Self { ctx: CpuContext::new(vm), lib, base, next: Cell::new(base), tmp: tmp.to_path_buf() }
    }

    fn root(&self) -> PathBuf {
        self.tmp.join("_SaveData")
    }

    /// Allocate `len` zeroed guest bytes.
    fn alloc(&self, len: usize) -> u64 {
        let addr = self.next.get();
        let len = (len as u64).div_ceil(16) * 16;
        assert!(addr + len <= self.base + SCRATCH, "scratch exhausted");
        self.next.set(addr + len);
        self.ctx.write_bytes(addr, &vec![0u8; len as usize]);
        addr
    }

    fn bytes(&self, data: &[u8]) -> u64 {
        let addr = self.alloc(data.len().max(1));
        self.ctx.write_bytes(addr, data);
        addr
    }

    fn put(&self, addr: u64, off: u64, data: &[u8]) {
        assert!(self.ctx.write_bytes(addr + off, data));
    }

    fn put32(&self, addr: u64, off: u64, v: u32) {
        self.put(addr, off, &v.to_le_bytes());
    }

    fn put64(&self, addr: u64, off: u64, v: u64) {
        self.put(addr, off, &v.to_le_bytes());
    }

    fn read(&self, addr: u64, len: usize) -> Vec<u8> {
        self.ctx.read_bytes(addr, len).unwrap()
    }

    fn name(&self, text: &str, size: usize) -> u64 {
        let addr = self.alloc(size);
        let n = text.len().min(size - 1);
        self.put(addr, 0, &text.as_bytes()[..n]);
        addr
    }

    fn dir_name(&self, text: &str) -> u64 {
        self.name(text, DIR_NAME_SIZE)
    }

    fn title(&self, text: &str) -> u64 {
        self.name(text, TITLE_ID_SIZE)
    }

    fn reset(&mut self, title: &str) {
        assert_eq!(self.lib.terminate(), OK);
        self.lib.title_id = Some(title.to_string());
    }

    // -- SaveDataMemory -----------------------------------------------------

    fn setup_param(&self, user: i32, slot: u32, size: u64) -> u64 {
        let s = self.alloc(0x30);
        self.put32(s, 4, user as u32);
        self.put64(s, 8, size);
        self.put32(s, 0x28, slot);
        s
    }

    fn setup(&mut self, user: i32, slot: u32, size: u64, option: u32) -> u64 {
        let s = self.setup_param(user, slot, size);
        self.put32(s, 0, option);
        let r = self.alloc(MEMORY_SETUP_RESULT_SIZE);
        assert_eq!(self.lib.setup_memory2(&self.ctx, s, r), OK);
        self.ctx.read_u64(r).unwrap()
    }

    fn memory_data(&self, buf: u64, size: u64, offset: i64) -> u64 {
        let d = self.alloc(MEMORY_DATA_SIZE as usize);
        self.put64(d, 0, buf);
        self.put64(d, 8, size);
        self.put64(d, 0x10, offset as u64);
        d
    }

    fn set_struct(&self, user: i32, slot: u32, data: u64, data_num: u32) -> u64 {
        let s = self.alloc(0x40);
        self.put32(s, 0, user as u32);
        self.put64(s, 8, data);
        self.put32(s, 0x20, data_num);
        self.put32(s, 0x24, slot);
        s
    }

    fn set(&mut self, buf: u64, size: u64, offset: i64, user: i32, slot: u32) -> HleResult {
        let d = self.memory_data(buf, size, offset);
        let s = self.set_struct(user, slot, d, 1);
        self.lib.set_memory2(&self.ctx, s)
    }

    fn get_struct(&self, user: i32, slot: u32) -> u64 {
        let g = self.alloc(0x40);
        self.put32(g, 0, user as u32);
        self.put32(g, 0x20, slot);
        g
    }

    fn get(&mut self, size: usize, user: i32, slot: u32) -> Vec<u8> {
        let buf = self.alloc(size);
        let d = self.memory_data(buf, size as u64, 0);
        let g = self.get_struct(user, slot);
        self.put64(g, 8, d);
        assert_eq!(self.lib.get_memory2(&self.ctx, g), OK);
        self.read(buf, size)
    }

    fn sync_struct(&self, user: i32, slot: u32, option: u32) -> u64 {
        let s = self.alloc(0x28);
        self.put32(s, 0, user as u32);
        self.put32(s, 4, slot);
        self.put32(s, 8, option);
        s
    }

    // -- classic saves ------------------------------------------------------

    fn mount3_struct(&self, user: i32, dir: u64, mode: u32, blocks: u64) -> u64 {
        let m = self.alloc(0x50);
        self.put32(m, 0, user as u32);
        self.put64(m, 8, dir);
        self.put64(m, 0x10, blocks);
        self.put32(m, 0x20, mode);
        m
    }

    /// Mount3 → (status, result address).
    fn mount3(&mut self, user: i32, dir: &str, mode: u32, blocks: u64) -> (HleResult, u64) {
        let d = self.dir_name(dir);
        let m = self.mount3_struct(user, d, mode, blocks);
        let r = self.alloc(MOUNT_RESULT_SIZE);
        (self.lib.mount3(&self.ctx, m, r), r)
    }

    fn transfer_struct(&self, user: i32, title: u64, dir: u64) -> u64 {
        let m = self.alloc(0x40);
        self.put32(m, 0, user as u32);
        self.put64(m, 8, title);
        self.put64(m, 0x10, dir);
        m
    }

    fn transfer(&mut self, user: i32, title: &str, dir: &str) -> (HleResult, u64) {
        let t = self.title(title);
        let d = self.dir_name(dir);
        let m = self.transfer_struct(user, t, d);
        let r = self.alloc(MOUNT_RESULT_SIZE);
        (self.lib.transferring_mount(&self.ctx, m, r), r)
    }

    fn umount(&mut self, result: u64) -> HleResult {
        self.lib.umount(&self.ctx, 0, result)
    }

    fn delete(&mut self, user: i32, title: Option<&str>, dir: &str) -> HleResult {
        let t = title.map_or(0, |t| self.title(t));
        let d = self.dir_name(dir);
        let s = self.alloc(0x50);
        self.put32(s, 0, user as u32);
        self.put64(s, 8, t);
        self.put64(s, 0x10, d);
        self.lib.delete(&self.ctx, s)
    }

    fn mounted(&self) -> PathBuf {
        self.lib.last_mounted.clone().unwrap_or_default()
    }

    fn search(&mut self, user: i32, title: Option<&str>) -> Vec<String> {
        let t = title.map_or(0, |t| self.title(t));
        let cond = self.alloc(0x40);
        self.put32(cond, 0, user as u32);
        self.put64(cond, 8, t);
        let names = self.alloc(DIR_NAME_SIZE * 8);
        let result = self.alloc(SEARCH_RESULT_SIZE);
        self.put64(result, 8, names);
        self.put32(result, 0x10, 8);
        assert_eq!(self.lib.dir_name_search(&self.ctx, cond, result), OK);
        let hit = self.ctx.read_u32(result).unwrap();
        let set = self.ctx.read_u32(result + 0x14).unwrap();
        assert!(hit == set && set <= 8);
        (0..set as u64)
            .map(|i| {
                let raw = self.read(names + i * DIR_NAME_SIZE as u64, DIR_NAME_SIZE);
                String::from_utf8(c_str(&raw).unwrap().to_vec()).unwrap()
            })
            .collect()
    }
}

fn text(field: &[u8]) -> String {
    String::from_utf8_lossy(c_str(field).unwrap_or(field)).into_owned()
}

#[test]
fn validation_and_scatter() {
    const BASE: u64 = 0x5D10_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    let tmp = temp_dir("validate");
    let mut h = Harness::new(&vm, BASE, &tmp, "PPSA21564");
    h.reset("VALIDATE");
    let byte = h.bytes(&[0x55]);
    let get = h.get_struct(1, 0);
    let sync = h.sync_struct(1, 0, 0);
    assert_eq!(h.lib.get_memory2(&h.ctx, get), ERR_MEMORY_NOT_READY);
    assert_eq!(h.set(byte, 1, 0, 1, 0), ERR_MEMORY_NOT_READY);
    assert_eq!(h.lib.sync_memory(&h.ctx, sync), ERR_MEMORY_NOT_READY);
    assert_eq!(h.lib.setup_memory2(&h.ctx, 0, 0), ERR_PARAMETER);
    let invalid_user = h.setup_param(-1, 0, 16);
    assert_eq!(h.lib.setup_memory2(&h.ctx, invalid_user, 0), ERR_INVALID_LOGIN_USER);
    for (slot, size) in [(4, 16), (0, 0), (0, 32 * 1024 * 1024 + 1)] {
        let invalid = h.setup_param(1, slot, size);
        assert_eq!(h.lib.setup_memory2(&h.ctx, invalid, 0), ERR_PARAMETER);
    }
    assert_eq!(h.setup(1, 0, 16, 0), 0);
    let again = h.setup_param(1, 0, 16);
    assert_eq!(h.lib.setup_memory2(&h.ctx, again, 0), ERR_BUSY);
    let mut original: Vec<u8> = (0..16).collect();
    let orig = h.bytes(&original);
    assert_eq!(h.set(orig, 16, 0, 1, 0), OK);

    let data = h.alloc(2 * MEMORY_DATA_SIZE as usize);
    let put_data = |i: u64, offset: i64| {
        h.put64(data + i * MEMORY_DATA_SIZE, 0, byte);
        h.put64(data + i * MEMORY_DATA_SIZE, 8, 1);
        h.put64(data + i * MEMORY_DATA_SIZE, 0x10, offset as u64);
    };
    put_data(0, 2);
    put_data(1, 16);
    let set = h.set_struct(1, 0, data, 2);
    assert_eq!(h.lib.set_memory2(&h.ctx, set), ERR_PARAMETER);
    assert_eq!(h.get(16, 1, 0), original);
    h.put64(data + MEMORY_DATA_SIZE, 0x10, 12);
    assert_eq!(h.lib.set_memory2(&h.ctx, set), OK);
    original[2] = 0x55;
    original[12] = 0x55;
    assert_eq!(h.get(16, 1, 0), original);

    h.put(byte, 0, &[0xaa]);
    h.put32(set, 0x20, 0);
    assert_eq!(h.lib.set_memory2(&h.ctx, set), OK);
    original[2] = 0xaa;
    assert_eq!(h.get(16, 1, 0), original);
    h.put64(set, 8, 0);
    assert_eq!(h.lib.set_memory2(&h.ctx, set), OK);
    assert_eq!(h.get(16, 1, 0), original);
    h.put32(set, 0x20, 6);
    assert_eq!(h.lib.set_memory2(&h.ctx, set), ERR_PARAMETER);
    h.put32(set, 0x20, 1);
    h.put64(set, 8, 0);
    assert_eq!(h.lib.set_memory2(&h.ctx, set), ERR_PARAMETER);
    assert_eq!(h.set(byte, 1, -1, 1, 0), ERR_PARAMETER);
    assert_eq!(h.set(byte, 1, i64::MAX, 1, 0), ERR_PARAMETER);
    assert_eq!(h.set(byte, u64::MAX, 1, 1, 0), ERR_PARAMETER);
    let invalid = h.memory_data(byte, u64::MAX, 1);
    h.put64(get, 8, invalid);
    assert_eq!(h.lib.get_memory2(&h.ctx, get), ERR_PARAMETER);
    assert_eq!(h.read(byte, 1), [0xaa]);
    assert_eq!(h.get(16, 1, 0), original);
    assert_eq!(h.lib.terminate(), OK);
    assert_eq!(h.setup(1, 0, 16, 0), 16);
    assert_eq!(h.get(16, 1, 0), original);
    fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn isolation_and_sync() {
    const BASE: u64 = 0x5D11_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    let tmp = temp_dir("isolate");
    let mut h = Harness::new(&vm, BASE, &tmp, "PPSA21564");
    h.reset("ISOLATE");
    let byte = h.bytes(&[11]);
    assert_eq!(h.setup(1, 0, 16, 0), 0);
    assert_eq!(h.set(byte, 1, 0, 1, 0), OK);
    assert_eq!(h.setup(2, 0, 16, 0), 0);
    assert_eq!(h.setup(1, 1, 16, 0), 0);
    assert_eq!(h.get(16, 2, 0), vec![0u8; 16]);
    assert_eq!(h.get(16, 1, 1), vec![0u8; 16]);
    h.put(byte, 0, &[22]);
    assert_eq!(h.set(byte, 1, 0, 2, 0), OK);
    h.put(byte, 0, &[33]);
    assert_eq!(h.set(byte, 1, 0, 1, 1), OK);
    assert_eq!(h.get(16, 1, 0)[0], 11);
    assert_eq!(h.get(16, 2, 0)[0], 22);
    assert_eq!(h.get(16, 1, 1)[0], 33);

    let sync = h.sync_struct(1, 1, 0);
    assert_eq!(h.lib.sync_memory(&h.ctx, sync), OK);
    let event = h.alloc(EVENT_SIZE);
    assert_eq!(h.lib.get_event_result(&h.ctx, event), OK);
    let ev = h.read(event, EVENT_SIZE);
    assert_eq!(u32::from_le_bytes(ev[0..4].try_into().unwrap()), 3);
    assert_eq!(i32::from_le_bytes(ev[4..8].try_into().unwrap()), 0);
    assert_eq!(i32::from_le_bytes(ev[8..12].try_into().unwrap()), 1);
    assert_eq!(text(&ev[0x20..0x40]), "sce_sdmemory1");
    assert_eq!(text(&ev[0x10..0x20]), "ISOLATE");
    assert_eq!(h.lib.get_event_result(&h.ctx, event), ERR_NOT_FOUND);
    h.put32(sync, 8, 1);
    assert_eq!(h.lib.sync_memory(&h.ctx, sync), OK);
    assert_eq!(h.lib.get_event_result(&h.ctx, event), ERR_NOT_FOUND);
    h.put32(sync, 8, 2);
    assert_eq!(h.lib.sync_memory(&h.ctx, sync), ERR_PARAMETER);

    h.reset("OTHER");
    assert_eq!(h.setup(1, 0, 16, 0), 0);
    assert_eq!(h.get(16, 1, 0), vec![0u8; 16]);
    h.reset("ISOLATE");
    assert_eq!(h.setup(1, 0, 16, 0), 16);
    assert_eq!(h.setup(2, 0, 16, 0), 16);
    assert_eq!(h.setup(1, 1, 16, 0), 16);
    assert!(h.get(16, 1, 0)[0] == 11 && h.get(16, 2, 0)[0] == 22 && h.get(16, 1, 1)[0] == 33);
    h.reset("OTHER");
    let (status, result) = h.transfer(2, "ISOLATE", "sce_sdmemory");
    assert_eq!(status, OK);
    assert_eq!(h.mounted(), h.root().join("ISOLATE/sce_sdmemory/2"));
    assert_eq!(fs::read(h.mounted().join("memory.dat")).unwrap()[0], 22);
    assert_eq!(h.umount(result), OK);
    assert_eq!(h.delete(1, Some("ISOLATE"), "sce_sdmemory"), OK);
    h.reset("ISOLATE");
    assert_eq!(h.setup(1, 0, 16, 0), 0);
    assert_eq!(h.setup(2, 0, 16, 0), 16);
    assert_eq!(h.get(16, 2, 0)[0], 22);
    fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn failed_write_preserves_save() {
    const BASE: u64 = 0x5D12_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    let tmp = temp_dir("iofail");
    let mut h = Harness::new(&vm, BASE, &tmp, "PPSA21564");
    h.reset("IOFAIL");
    assert_eq!(h.setup(1, 0, 16, 0), 0);
    let byte = h.bytes(&[42]);
    assert_eq!(h.set(byte, 1, 0, 1, 0), OK);
    fs::rename(h.root(), tmp.join("save-backup")).unwrap();
    fs::write(h.root(), "not a directory").unwrap();
    h.put(byte, 0, &[99]);
    assert!(h.set(byte, 1, 0, 1, 0) < 0);
    assert_eq!(h.get(16, 1, 0)[0], 42);
    fs::remove_file(h.root()).unwrap();
    fs::rename(tmp.join("save-backup"), h.root()).unwrap();
    assert_eq!(h.lib.terminate(), OK);
    assert_eq!(h.setup(1, 0, 16, 0), 16);
    assert_eq!(h.get(16, 1, 0)[0], 42);
    fs::remove_dir_all(&tmp).unwrap();
}

/// Set must survive a restart on its own (no Sync, no Terminate): the first
/// library instance is dropped as if the process exited.
#[test]
fn restart_keeps_memory() {
    const BASE: u64 = 0x5D13_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    let tmp = temp_dir("restart");
    {
        let mut h = Harness::new(&vm, BASE, &tmp, "PPSA21564");
        let setup = h.setup_param(7, 3, 16);
        h.put32(setup, 0, 1);
        let icon_bytes = h.bytes(&[1, 2, 3, 4]);
        let icon = h.alloc(0x38);
        h.put64(icon, 0, icon_bytes);
        h.put64(icon, 8, 4);
        h.put64(icon, 0x10, 4);
        h.put64(setup, 0x20, icon);
        h.put64(setup, 0x10, 4);
        let mut param = make_param("Astro progress", "", "", 17);
        let param_addr = h.bytes(&param);
        h.put64(setup, 0x18, param_addr);
        assert_eq!(h.lib.setup_memory2(&h.ctx, setup, 0), OK);
        let progress = h.bytes(&[11, 22, 33, 44]);
        let data = h.memory_data(progress, 4, 4);
        let set = h.set_struct(7, 3, data, 0);
        h.put64(set, 0x18, icon);
        assert_eq!(h.lib.set_memory2(&h.ctx, set), OK);
        h.put64(set, 8, 0);
        param = make_param("Astro progress", "", "Coins, level state, rescued bots", 18);
        h.put(param_addr, 0, &param);
        h.put64(set, 0x10, param_addr);
        assert_eq!(h.lib.set_memory2(&h.ctx, set), OK);
    }
    let vm = ps5vm::VirtualMemory::new();
    let mut h = Harness::new(&vm, BASE, &tmp, "PPSA21564");
    assert_eq!(h.setup(7, 3, 24, 1), 16);
    let mut expected = vec![0u8; 24];
    expected[4..8].copy_from_slice(&[11, 22, 33, 44]);
    assert_eq!(h.get(24, 7, 3), expected);
    let param = h.alloc(PARAM_SIZE);
    let icon_bytes = h.bytes(&[9, 9, 9, 9]);
    let icon = h.alloc(0x38);
    h.put64(icon, 0, icon_bytes);
    h.put64(icon, 8, 4);
    h.put64(icon, 0x10, 4);
    let get = h.get_struct(7, 3);
    h.put64(get, 0x10, param);
    h.put64(get, 0x18, icon);
    assert_eq!(h.lib.get_memory2(&h.ctx, get), OK);
    assert_eq!(h.ctx.read_u64(icon + 0x10), Some(0));
    assert_eq!(h.read(icon_bytes, 4), [9, 9, 9, 9]);
    let p = h.read(param, PARAM_SIZE);
    assert_eq!(text(&p[PARAM_TITLE]), "Astro progress");
    assert_eq!(text(&p[PARAM_DETAIL]), "Coins, level state, rescued bots");
    assert_eq!(u32::from_le_bytes(p[0x500..0x504].try_into().unwrap()), 18);
    assert_eq!(h.lib.terminate(), OK);
    assert_eq!(h.setup(7, 3, 6, 0), 16);
    expected.truncate(6);
    assert_eq!(h.get(6, 7, 3), expected);
    fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn classic_save_paths() {
    const BASE: u64 = 0x5D14_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    let tmp = temp_dir("classic");
    let mut h = Harness::new(&vm, BASE, &tmp, "PPSA21564");
    h.reset("CLASSIC");
    let names = ["save.1", "save1", "slot@A", "slotA"];
    let (status, first) = h.mount3(1, "save.1", 4, 48);
    assert_eq!(status, OK);
    fs::write(h.mounted().join("progress"), "save.1").unwrap();
    assert_eq!(h.umount(first), OK);
    assert_eq!(h.search(1, None), ["save.1"]);
    for text in names {
        let (status, result) = h.mount3(1, text, if text == "save.1" { 1 } else { 4 }, 48);
        assert_eq!(status, OK);
        assert_eq!(h.mounted(), h.root().join("CLASSIC").join(text));
        if text != "save.1" {
            fs::write(h.mounted().join("progress"), text).unwrap();
        }
        assert_eq!(h.umount(result), OK);
    }
    assert_eq!(h.search(1, None), names);
    h.reset("OTHER");
    assert!(h.search(1, None).is_empty());
    assert_eq!(h.search(1, Some("CLASSIC")), names);
    for text in names {
        let (status, result) = h.transfer(1, "CLASSIC", text);
        assert_eq!(status, OK);
        assert_eq!(fs::read_to_string(h.mounted().join("progress")).unwrap(), text);
        assert_eq!(h.umount(result), OK);
    }
    h.reset("CLASSIC");
    for text in ["", ".", "..", "../save1", "save/1", "save\\1"] {
        h.lib.last_mounted = None;
        let (status, _) = h.mount3(1, text, 32, 0);
        assert_eq!(status, ERR_PARAMETER);
        assert!(h.lib.last_mounted.is_none());
        assert_eq!(h.delete(1, None, text), ERR_PARAMETER);
    }
    assert_eq!(h.search(1, None), names);
    fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn save_allocations() {
    const BASE: u64 = 0x5D15_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    let tmp = temp_dir("capacity");
    let mut h = Harness::new(&vm, BASE, &tmp, "PPSA21564");
    h.reset("CAPACITY");
    let names = ["Options", "Player"];
    let allocations = [48u64, 96];
    for blocks in [0, 47, 16385] {
        let (status, _) = h.mount3(1, names[0], 4, blocks);
        assert_eq!(status, ERR_PARAMETER);
        assert!(!h.root().join("CAPACITY/Options").exists());
    }
    let info = h.alloc(BLOCKS_INFO_SIZE);
    for (name, blocks) in names.iter().zip(allocations) {
        let (status, result) = h.mount3(1, name, 4, blocks);
        assert_eq!(status, OK);
        assert_eq!(h.ctx.read_u32(result + 0x1C), Some(1));
        fs::write(h.mounted().join("USR-DATA"), "saved payload").unwrap();
        assert_eq!(h.umount(result), OK);
        assert_eq!(h.lib.get_mount_info(&h.ctx, result, info), ERR_NOT_MOUNTED);
    }
    h.reset("CAPACITY");
    let found = h.alloc(DIR_NAME_SIZE * 2);
    let infos = h.alloc(BLOCKS_INFO_SIZE * 2);
    let cond = h.alloc(0x40);
    h.put32(cond, 0, 1);
    let search = h.alloc(SEARCH_RESULT_SIZE);
    h.put64(search, 8, found);
    h.put32(search, 0x10, 2);
    h.put64(search, 0x20, infos);
    assert_eq!(h.lib.dir_name_search(&h.ctx, cond, search), OK);
    assert_eq!(h.ctx.read_u32(search + 0x14), Some(2));
    let mut total = 0;
    for (i, (name, blocks)) in names.iter().zip(allocations).enumerate() {
        let i = i as u64;
        assert_eq!(text(&h.read(found + i * 32, 32)), *name);
        assert_eq!(h.ctx.read_u64(infos + i * 0x30), Some(blocks));
        assert_eq!(h.ctx.read_u64(infos + i * 0x30 + 8), Some(blocks - 1), "USR-DATA uses one block");
        total += h.ctx.read_u64(infos + i * 0x30).unwrap();
        let (status, result) = h.mount3(1, name, 34, 16384);
        assert_eq!(status, OK);
        assert_eq!(h.ctx.read_u32(result + 0x1C), Some(0));
        let mut f = fs::OpenOptions::new().append(true).open(h.mounted().join("USR-DATA")).unwrap();
        f.write_all(b"more data").unwrap();
        assert_eq!(h.lib.get_mount_info(&h.ctx, result, info), OK);
        assert_eq!(h.ctx.read_u64(info), Some(blocks));
        assert_eq!(h.ctx.read_u64(info + 8), Some(blocks - 1));
        assert_eq!(h.umount(result), OK);
    }
    assert_eq!(total, 144);
    let metadata = h.root().join("CAPACITY/Options/sce_sys/blocks.bin");
    fs::OpenOptions::new().write(true).open(&metadata).unwrap().set_len(7).unwrap();
    assert_eq!(h.lib.dir_name_search(&h.ctx, cond, search), ERR_BROKEN);
    assert_eq!(fs::metadata(&metadata).unwrap().len(), 7);
    fs::remove_file(&metadata).unwrap();
    assert_eq!(h.lib.dir_name_search(&h.ctx, cond, search), ERR_BROKEN);
    assert!(!metadata.exists());
    fs::remove_dir_all(&tmp).unwrap();
}

#[test]
#[cfg(unix)]
fn classic_save_params() {
    const BASE: u64 = 0x5D16_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    let tmp = temp_dir("params");
    let mut h = Harness::new(&vm, BASE, &tmp, "PPSA21564");
    h.reset("PARAMS");
    let name = h.dir_name("slot");
    let mount = h.mount3_struct(1, name, 4, 48);
    let mounted = h.alloc(MOUNT_RESULT_SIZE);
    assert_eq!(h.lib.mount3(&h.ctx, mount, mounted), OK);
    let param = h.bytes(&make_param("First save", "Chapter 2", "Progress", 42));
    assert_eq!(h.lib.set_param(&h.ctx, mounted, 0, 0, PARAM_SIZE as u64), ERR_PARAMETER);
    assert_eq!(h.lib.set_param(&h.ctx, mounted, 0, param, PARAM_SIZE as u64 - 1), ERR_PARAMETER);
    assert_eq!(h.lib.set_param(&h.ctx, mounted, 0, param, PARAM_SIZE as u64), OK);
    let saved = h.root().join("PARAMS/slot");
    assert!(saved.join("sce_sys/param.bin").exists());
    fs::create_dir(saved.join("nested")).unwrap();
    fs::write(saved.join("nested/progress"), "payload").unwrap();
    let loaded = h.alloc(PARAM_SIZE);
    let got = h.alloc(8);
    assert_eq!(h.lib.get_param(&h.ctx, mounted, 0, loaded, PARAM_SIZE as u64, got), OK);
    let p = h.read(loaded, PARAM_SIZE);
    assert_eq!(h.ctx.read_u64(got), Some(PARAM_SIZE as u64));
    assert_eq!(text(&p[PARAM_TITLE]), "First save");
    assert_eq!(text(&p[PARAM_SUB_TITLE]), "Chapter 2");
    assert_eq!(u32::from_le_bytes(p[0x500..0x504].try_into().unwrap()), 42);
    let mtime = |h: &Harness| i64::from_le_bytes(h.read(loaded + PARAM_MTIME as u64, 8).try_into().unwrap());
    assert!(mtime(&h) > 0);
    let param_time = fs::metadata(saved.join("sce_sys/param.bin")).unwrap().modified().unwrap();
    fs::File::options()
        .write(true)
        .open(saved.join("nested/progress"))
        .unwrap()
        .set_modified(param_time + Duration::from_secs(2 * 3600))
        .unwrap();
    assert_eq!(h.lib.get_param(&h.ctx, mounted, 0, loaded, PARAM_SIZE as u64, 0), OK);
    let nested_mtime = mtime(&h);
    assert!(nested_mtime > unix_secs(SystemTime::now()) + 3600);
    // A leftover "<file>.tmp" does not get in the way: the temporary file
    // is a new one of its own. A directory that takes no new file fails the
    // update and keeps the old parameters.
    fs::create_dir(saved.join("sce_sys/param.bin.tmp")).unwrap();
    h.put(param, 0, b"Failed update\0");
    use std::os::unix::fs::PermissionsExt;
    let writable = fs::metadata(saved.join("sce_sys")).unwrap().permissions();
    fs::set_permissions(saved.join("sce_sys"), fs::Permissions::from_mode(0o555)).unwrap();
    assert_eq!(h.lib.set_param(&h.ctx, mounted, 0, param, PARAM_SIZE as u64), ERR_INTERNAL);
    fs::set_permissions(saved.join("sce_sys"), writable).unwrap();
    fs::remove_dir(saved.join("sce_sys/param.bin.tmp")).unwrap();
    assert_eq!(h.lib.get_param(&h.ctx, mounted, 0, loaded, PARAM_SIZE as u64, 0), OK);
    assert_eq!(text(&h.read(loaded, 128)), "First save");
    h.put(param, 0, b"Updated save\0");
    assert_eq!(h.lib.set_param(&h.ctx, mounted, 0, param, PARAM_SIZE as u64), OK);
    assert_eq!(h.lib.get_param(&h.ctx, mounted, 0, loaded, PARAM_SIZE as u64, 0), OK);
    assert_eq!(text(&h.read(loaded, 128)), "Updated save");
    assert_eq!(h.umount(mounted), OK);
    assert_eq!(h.lib.get_param(&h.ctx, mounted, 0, loaded, PARAM_SIZE as u64, 0), ERR_NOT_MOUNTED);

    h.reset("OTHER");
    let source_title = h.title("PARAMS");
    let transfer = h.transfer_struct(1, source_title, name);
    let transferred = h.alloc(MOUNT_RESULT_SIZE);
    assert_eq!(h.lib.transferring_mount(&h.ctx, transfer, transferred), OK);
    assert_eq!(h.lib.get_param(&h.ctx, transferred, 0, loaded, PARAM_SIZE as u64, 0), OK);
    assert_eq!(text(&h.read(loaded, 128)), "Updated save");
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink("PARAMS", h.root().join("P_ALIAS")).unwrap();
        let alias = h.title("P_ALIAS");
        h.put64(transfer, 8, alias);
        let duplicate = h.alloc(MOUNT_RESULT_SIZE);
        assert_eq!(h.lib.transferring_mount(&h.ctx, transfer, duplicate), ERR_BUSY);
        h.put64(transfer, 8, source_title);
        fs::remove_file(h.root().join("P_ALIAS")).unwrap();
    }

    let other = h.alloc(MOUNT_RESULT_SIZE);
    assert_eq!(h.lib.mount3(&h.ctx, mount, other), OK);
    h.put(param, 0, b"Other title\0");
    assert_eq!(h.lib.set_param(&h.ctx, other, 0, param, PARAM_SIZE as u64), OK);
    let typed = h.bytes(b"Typed title\0");
    assert_eq!(h.lib.set_param(&h.ctx, other, 1, typed, 12), OK);
    // Text fields come back whole: a buffer under 128 bytes is refused.
    let title = h.alloc(128);
    assert_eq!(h.lib.get_param(&h.ctx, other, 1, title, 64, got), ERR_PARAMETER);
    assert_eq!(h.lib.get_param(&h.ctx, other, 1, title, 128, got), OK);
    assert_eq!(text(&h.read(title, 128)), "Typed title");
    assert_eq!(h.ctx.read_u64(got), Some(128));
    assert_eq!(h.lib.get_param(&h.ctx, other, 6, title, 128, got), ERR_PARAMETER);
    let user_param = h.bytes(&77u32.to_le_bytes());
    assert_eq!(h.lib.set_param(&h.ctx, other, 4, user_param, 4), OK);
    assert_eq!(h.lib.get_param(&h.ctx, other, 0, loaded, PARAM_SIZE as u64, 0), OK);
    let p = h.read(loaded, PARAM_SIZE);
    assert_eq!(text(&p[PARAM_TITLE]), "Typed title");
    assert_eq!(u32::from_le_bytes(p[0x500..0x504].try_into().unwrap()), 77);
    let mtime_addr = loaded + PARAM_MTIME as u64;
    assert_eq!(h.lib.get_param(&h.ctx, other, 5, mtime_addr, 8, 0), OK);
    assert!(mtime(&h) > 0);
    assert_eq!(h.lib.set_param(&h.ctx, other, 5, mtime_addr, 8), ERR_PARAMETER);
    assert_eq!(h.lib.get_param(&h.ctx, transferred, 0, loaded, PARAM_SIZE as u64, 0), OK);
    assert_eq!(text(&h.read(loaded, 128)), "Updated save");
    assert_eq!(h.umount(other), OK);
    assert_eq!(h.umount(transferred), OK);

    let names = h.alloc(DIR_NAME_SIZE);
    let params = h.alloc(PARAM_SIZE);
    let cond = h.alloc(0x40);
    h.put32(cond, 0, 1);
    h.put64(cond, 8, source_title);
    let result = h.alloc(SEARCH_RESULT_SIZE);
    h.put64(result, 8, names);
    h.put32(result, 0x10, 1);
    h.put64(result, 0x18, params);
    assert_eq!(h.lib.dir_name_search(&h.ctx, cond, result), OK);
    assert_eq!(h.ctx.read_u32(result + 0x14), Some(1));
    assert_eq!(text(&h.read(names, 32)), "slot");
    let p = h.read(params, PARAM_SIZE);
    assert_eq!(text(&p[PARAM_TITLE]), "Updated save");
    assert_eq!(i64::from_le_bytes(p[PARAM_MTIME..PARAM_MTIME + 8].try_into().unwrap()), nested_mtime);
    fs::remove_dir_all(&tmp).unwrap();
}

// ---------------------------------------------------------------------------
// Additional coverage
// ---------------------------------------------------------------------------

#[test]
fn nids_match_names() {
    let mut reg = HleRegistry::new();
    register(&mut reg);
    let text = include_str!("../savedata.rs");
    let mut count = 0;
    for line in text.lines().filter(|l| l.trim_start().starts_with("reg.register_fn(\"")) {
        let mut parts = line.split('"');
        let (nid, name) = (parts.nth(1).unwrap(), parts.nth(1).unwrap());
        assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        count += 1;
    }
    assert_eq!(count, 47);
}

#[test]
fn search_pattern_order_and_limits() {
    const BASE: u64 = 0x5D17_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    let tmp = temp_dir("search");
    let mut h = Harness::new(&vm, BASE, &tmp, "SEARCH");
    for name in ["slotB", "SLOTa", "other"] {
        let (status, r) = h.mount3(1, name, 4, 48);
        assert_eq!(status, OK);
        assert_eq!(h.umount(r), OK);
    }
    // System directories and directories without param.bin are not saves.
    fs::create_dir_all(h.root().join("SEARCH/sce_sdmemory/1")).unwrap();
    fs::create_dir_all(h.root().join("SEARCH/sce_sys")).unwrap();

    let pattern = h.dir_name("slot%");
    let names = h.alloc(DIR_NAME_SIZE * 4);
    let cond = h.alloc(0x40);
    h.put32(cond, 0, 1);
    h.put64(cond, 0x10, pattern);
    h.put32(cond, 0x1C, SORT_ORDER_DESCENT);
    let result = h.alloc(SEARCH_RESULT_SIZE);
    h.put(result, 0, &[0xFF; SEARCH_RESULT_SIZE]);
    h.put64(result, 8, names);
    h.put32(result, 0x10, 1);
    h.put64(result, 0x18, 0);
    h.put64(result, 0x20, 0);
    assert_eq!(h.lib.dir_name_search(&h.ctx, cond, result), OK);
    // Case-insensitive match, byte order, descending, truncated to dirNamesNum.
    assert_eq!(h.ctx.read_u32(result), Some(2));
    assert_eq!(h.ctx.read_u32(result + 4), Some(0));
    assert_eq!(h.ctx.read_u32(result + 0x14), Some(1));
    assert_eq!(h.read(result + 0x28, 0x10), vec![0u8; 0x10]);
    assert_eq!(text(&h.read(names, 32)), "slotB");

    h.put(pattern, 0, b"SLOT_\0");
    h.put32(cond, 0x1C, 0);
    h.put32(result, 0x10, 4);
    assert_eq!(h.lib.dir_name_search(&h.ctx, cond, result), OK);
    assert_eq!(h.ctx.read_u32(result + 0x14), Some(2));
    assert_eq!(text(&h.read(names, 32)), "SLOTa");

    h.put32(cond, 0x18, 6);
    assert_eq!(h.lib.dir_name_search(&h.ctx, cond, result), ERR_PARAMETER);
    h.put32(cond, 0x18, 0);
    h.put32(cond, 0x1C, 2);
    assert_eq!(h.lib.dir_name_search(&h.ctx, cond, result), ERR_PARAMETER);
    h.put32(cond, 0x1C, 0);
    h.put(pattern, 0, &[b'x'; 32]);
    assert_eq!(h.lib.dir_name_search(&h.ctx, cond, result), ERR_PARAMETER);
    h.put(pattern, 0, &[0; 32]);
    h.put64(result, 8, 0);
    assert_eq!(h.lib.dir_name_search(&h.ctx, cond, result), ERR_PARAMETER);
    h.put32(result, 0x10, 0);
    h.put32(cond, 0, u32::MAX);
    assert_eq!(h.lib.dir_name_search(&h.ctx, cond, result), ERR_INVALID_LOGIN_USER);
    h.put32(cond, 0, 1);
    assert_eq!(h.lib.dir_name_search(&h.ctx, cond, result), OK);
    assert_eq!(h.ctx.read_u32(result), Some(3));
    assert_eq!(h.ctx.read_u32(result + 0x14), Some(0));
    fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn mount_modes_slots_and_events() {
    const BASE: u64 = 0x5D18_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    let tmp = temp_dir("mount");
    let mut h = Harness::new(&vm, BASE, &tmp, "MOUNTS");
    assert_eq!(h.mount3(1, "A", 1, 0).0, ERR_NOT_FOUND);
    assert_eq!(h.mount3(-1, "A", 4, 48).0, ERR_INVALID_LOGIN_USER);
    let (status, a) = h.mount3(1, "A", 4, 48);
    assert_eq!(status, OK);
    assert_eq!(text(&h.read(a, 16)), "/savedata0");
    assert_eq!(h.mount3(1, "A", 1, 0).0, ERR_BUSY);
    assert_eq!(h.delete(1, None, "A"), ERR_BUSY);
    assert_eq!(h.lib.terminate(), ERR_BUSY);
    let mut results = vec![a];
    for i in 1..slots::COUNT {
        let (status, r) = h.mount3(1, &format!("S{i}"), 32, 48);
        assert_eq!(status, OK);
        assert_eq!(text(&h.read(r, 16)), format!("/savedata{i}"));
        results.push(r);
    }
    assert_eq!(h.mount3(1, "X", 32, 48).0, ERR_MOUNT_FULL);
    // Unmount with asynchronous backup reports UMOUNT_BACKUP_END.
    assert_eq!(h.lib.umount(&h.ctx, UMOUNT_MODE_BACKUP_ASYNC, a), OK);
    assert_eq!(h.umount(a), ERR_NOT_FOUND);
    let event = h.alloc(EVENT_SIZE);
    assert_eq!(h.lib.get_event_result(&h.ctx, 0), ERR_PARAMETER);
    assert_eq!(h.lib.get_event_result(&h.ctx, event), OK);
    assert_eq!(h.ctx.read_u32(event), Some(EVENT_TYPE_UMOUNT_BACKUP_END));
    // The freed slot is reused.
    let (status, again) = h.mount3(1, "A", 2, 0);
    assert_eq!(status, OK);
    assert_eq!(text(&h.read(again, 16)), "/savedata0");
    assert_eq!(h.mount3(1, "A2", 4, 48).0, ERR_MOUNT_FULL);
    for r in results.iter().skip(1) {
        assert_eq!(h.umount(*r), OK);
    }
    assert_eq!(h.umount(again), OK);
    assert_eq!(h.mount3(1, "A", 4, 48).0, ERR_EXISTS);
    assert_eq!(h.delete(1, None, "A"), OK);
    assert!(!h.root().join("MOUNTS/A").exists());
    assert_eq!(h.delete(1, None, "A"), OK);

    // Backup and commit events, queue capped at 20.
    let dir = h.dir_name("B");
    let title = h.title("MOUNTS");
    let backup = h.alloc(0x40);
    h.put32(backup, 0, 5);
    h.put64(backup, 8, title);
    h.put64(backup, 0x10, dir);
    assert_eq!(h.lib.backup(&h.ctx, backup), OK);
    assert_eq!(h.lib.get_event_result(&h.ctx, event), OK);
    let ev = h.read(event, EVENT_SIZE);
    assert_eq!(u32::from_le_bytes(ev[0..4].try_into().unwrap()), EVENT_TYPE_BACKUP_END);
    assert_eq!(i32::from_le_bytes(ev[8..12].try_into().unwrap()), 5);
    assert_eq!(text(&ev[0x10..0x20]), "MOUNTS");
    assert_eq!(text(&ev[0x20..0x40]), "B");
    let commit = h.alloc(0x28);
    h.put32(commit, 4, COMMIT_MODE_BACKUP_ASYNC);
    for _ in 0..25 {
        assert_eq!(h.lib.commit(&h.ctx, commit), OK);
    }
    assert_eq!(h.lib.events.len(), EVENT_QUEUE_MAX);
    assert_eq!(h.lib.commit(&h.ctx, 0), ERR_PARAMETER);
    assert_eq!(h.lib.create_transaction_resource(0), 1);
    assert_eq!(h.lib.create_transaction_resource(0), 2);
    assert_eq!(h.lib.terminate(), OK);
    assert!(h.lib.events.is_empty());
    fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn memory_limits_and_param_option() {
    const BASE: u64 = 0x5D19_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    let tmp = temp_dir("memlimit");
    let mut h = Harness::new(&vm, BASE, &tmp, "MEMLIMIT");
    let bad_option = h.setup_param(1, 0, 16);
    h.put32(bad_option, 0, 4);
    assert_eq!(h.lib.setup_memory2(&h.ctx, bad_option, 0), ERR_PARAMETER);
    assert_eq!(h.setup(1, 0, 16 * 1024 * 1024, MEMORY_DOUBLE_BUFFER), 0);
    let over = h.setup_param(1, 1, 1);
    assert_eq!(h.lib.setup_memory2(&h.ctx, over, 0), ERR_LIMITATION_OVER);
    assert_eq!(h.lib.terminate(), OK);
    for slot in 0..4 {
        assert_eq!(h.setup(slot as i32 + 1, slot, 16, 0), 0);
    }
    let fifth = h.setup_param(9, 0, 16);
    assert_eq!(h.lib.setup_memory2(&h.ctx, fifth, 0), ERR_LIMITATION_OVER);
    // A parameter needs the SET_PARAM option.
    let param = h.alloc(PARAM_SIZE);
    let get = h.get_struct(1, 0);
    h.put64(get, 0x10, param);
    assert_eq!(h.lib.get_memory2(&h.ctx, get), ERR_PARAMETER);
    let set = h.set_struct(1, 0, 0, 0);
    h.put64(set, 0x10, param);
    assert_eq!(h.lib.set_memory2(&h.ctx, set), ERR_PARAMETER);
    assert_eq!(h.lib.terminate(), OK);
    assert_eq!(h.setup(1, 0, 16, MEMORY_SET_PARAM), 0);
    assert_eq!(h.lib.get_memory2(&h.ctx, get), OK);
    assert_eq!(text(&h.read(param, 128)), MEMORY_DEFAULT_TITLE);
    // Synchronising while the memory directory is mounted is refused.
    let (status, r) = h.transfer(1, "MEMLIMIT", "sce_sdmemory");
    assert_eq!(status, ERR_NOT_FOUND);
    let _ = r;
    let sync = h.sync_struct(1, 0, 1);
    assert_eq!(h.lib.sync_memory(&h.ctx, sync), OK);
    let (status, r) = h.transfer(1, "MEMLIMIT", "sce_sdmemory");
    assert_eq!(status, OK);
    assert_eq!(h.lib.sync_memory(&h.ctx, sync), ERR_BUSY);
    assert_eq!(h.delete(1, None, "sce_sdmemory"), ERR_BUSY);
    assert_eq!(h.umount(r), OK);
    fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn icons_round_trip() {
    const BASE: u64 = 0x5D1B_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    let tmp = temp_dir("icon");
    let mut h = Harness::new(&vm, BASE, &tmp, "ICONS");
    let (status, r) = h.mount3(1, "slot", 4, 48);
    assert_eq!(status, OK);
    let png = h.bytes(&[0x89, b'P', b'N', b'G', 1, 2, 3]);
    let icon = h.alloc(0x38);
    h.put64(icon, 0, png);
    h.put64(icon, 8, 64);
    h.put64(icon, 0x10, 7);
    assert_eq!(h.lib.load_icon(&h.ctx, r, 0), ERR_PARAMETER);
    assert_eq!(h.lib.load_icon(&h.ctx, r, icon), OK);
    assert_eq!(h.ctx.read_u64(icon + 0x10), Some(0));
    h.put64(icon, 0x10, 7);
    assert_eq!(h.lib.save_icon(&h.ctx, r, icon), OK);
    assert_eq!(fs::read(h.root().join("ICONS/slot/sce_sys/icon0.png")).unwrap().len(), 7);
    let out = h.alloc(64);
    h.put64(icon, 0, out);
    assert_eq!(h.lib.load_icon(&h.ctx, r, icon), OK);
    assert_eq!(h.ctx.read_u64(icon + 0x10), Some(7));
    assert_eq!(h.read(out, 7), [0x89, b'P', b'N', b'G', 1, 2, 3]);
    assert_eq!(h.umount(r), OK);
    fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn storage_root_and_names() {
    assert!(storage_root(Some("rel/dir")).is_absolute());
    assert_eq!(storage_root(Some("/abs/x")), PathBuf::from("/abs/x"));
    for bad in ["", ".", "..", "../x", "a/b", "a\\b", "a b"] {
        assert!(!valid_path_component(bad), "{bad}");
    }
    assert!(valid_path_component("save.1@A-_"));
    assert!(dir_name_match(b"save0001", b"save%"));
    assert!(dir_name_match(b"save0001", b"save000_"));
    assert!(!dir_name_match(b"save0001", b"save00_"));
    assert!(dir_name_match(b"abc", b"%"));
    // A trailing '%' matches an empty rest.
    assert!(dir_name_match(b"trail", b"trail%"));
    assert!(dir_name_match(b"trail", b"trail%%"));
    assert!(dir_name_match(b"trail", b"trai_%"));
    assert!(!dir_name_match(b"trail", b"trail_%"));
    assert!(dir_name_match(b"trail1", b"trail_%"));
    assert!(!dir_name_match(b"abc", b"abd"));
    assert_eq!(memory_dir_name(0), "sce_sdmemory");
    assert_eq!(memory_dir_name(3), "sce_sdmemory3");
    assert!(is_memory_dir_name("sce_sdmemory2") && !is_memory_dir_name("sce_sdmemory4"));
}

/// Mount status (+0x1C): CREATED only when the mount made the save; a
/// CREATE2 mount of an existing save opens it.
#[test]
fn mount_status_reports_created_saves() {
    const BASE: u64 = 0x5E7B_0000_0000;
    const READ_WRITE: u32 = 2;
    let vm = ps5vm::VirtualMemory::new();
    let tmp = temp_dir("mount-status");
    let mut h = Harness::new(&vm, BASE, &tmp, "PPSA21564");
    h.reset("STATUS");
    for (dir, mode, expected, what) in [
        ("fresh", READ_WRITE | MOUNT_MODE_CREATE2, 1, "CREATE2 of a missing save"),
        ("fresh", READ_WRITE | MOUNT_MODE_CREATE2, 0, "CREATE2 of an existing save"),
        ("made", READ_WRITE | MOUNT_MODE_CREATE, 1, "CREATE of a missing save"),
        ("made", READ_WRITE, 0, "open of an existing save"),
    ] {
        let (status, result) = h.mount3(1, dir, mode, 48);
        assert_eq!(status, OK, "{what}: mounts");
        let got = u32::from_le_bytes(h.read(result + 0x1C, 4).try_into().unwrap());
        assert_eq!(got, expected, "{what}: status");
        assert_eq!(h.umount(result), OK, "{what}: unmounts");
    }
    fs::remove_dir_all(&tmp).unwrap();
}

/// Null structures and null pointers inside them are parameter errors for
/// every entry point; nothing is written to the result.
#[test]
fn null_arguments_are_parameter_errors() {
    const BASE: u64 = 0x5D1D_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    let tmp = temp_dir("null-args");
    let mut h = Harness::new(&vm, BASE, &tmp, "PPSA00002");
    let result = h.alloc(MOUNT_RESULT_SIZE);
    h.put(result, 0, &[0xAB; 4]);
    let empty = h.alloc(0x50);
    let named = h.mount3_struct(1, h.dir_name("kept"), 2, 48);
    assert_eq!(h.lib.mount3(&h.ctx, 0, result), ERR_PARAMETER);
    assert_eq!(h.lib.mount3(&h.ctx, named, 0), ERR_PARAMETER);
    assert_eq!(h.lib.mount3(&h.ctx, empty, result), ERR_PARAMETER);
    assert_eq!(h.lib.mount1(&h.ctx, named, 0), ERR_PARAMETER);
    assert_eq!(h.read(result, 4), vec![0xAB; 4]);
    assert_eq!(h.lib.delete(&h.ctx, 0), ERR_PARAMETER);
    assert_eq!(h.lib.delete(&h.ctx, empty), ERR_PARAMETER);
    let cond = h.alloc(0x40);
    h.put32(cond, 0, 1);
    let search = h.alloc(SEARCH_RESULT_SIZE);
    assert_eq!(h.lib.dir_name_search(&h.ctx, 0, search), ERR_PARAMETER);
    assert_eq!(h.lib.dir_name_search(&h.ctx, cond, 0), ERR_PARAMETER);
    // A null memory sync parameter is refused before any setup.
    assert_eq!(h.lib.sync_memory(&h.ctx, 0), ERR_PARAMETER);
    assert_eq!(ERR_PARAMETER, 0x809F_0000u32 as i32 as HleResult);
    fs::remove_dir_all(&tmp).unwrap();
}

/// Search results follow the sort key (by name first, then stably by the
/// key) and the order; infos report the blocks the files leave free; an
/// unknown key or order is refused; a failed mount leaves the result alone.
#[test]
fn search_sort_keys_and_free_blocks() {
    const BASE: u64 = 0x5D1C_0000_0000;
    let vm = ps5vm::VirtualMemory::new();
    let tmp = temp_dir("sort");
    let mut h = Harness::new(&vm, BASE, &tmp, "PPSA21565");
    h.reset("SORT");
    // (name, userParam, mtime, blocks, payload bytes)
    let saves = [("SORT_B", 1u32, 1000i64, 96u64, 40_000usize), ("SORT_A", 3, 2000, 200, 0), ("SORT_C", 2, 3000, 150, 1)];
    for (name, user_param, _, blocks, payload) in saves {
        let (status, result) = h.mount3(1, name, 4, blocks);
        assert_eq!(status, OK);
        let param = h.bytes(&make_param(name, "", "", user_param));
        assert_eq!(h.lib.set_param(&h.ctx, result, 0, param, PARAM_SIZE as u64), OK);
        if payload != 0 {
            fs::write(h.mounted().join("DATA"), vec![7u8; payload]).unwrap();
        }
        assert_eq!(h.umount(result), OK);
    }
    // The newest file of a save is its mtime.
    for (name, _, time, _, _) in saves {
        let dir = h.root().join("SORT").join(name);
        let mut pending = vec![dir];
        while let Some(d) = pending.pop() {
            for e in fs::read_dir(&d).unwrap().flatten() {
                if e.file_type().unwrap().is_dir() {
                    pending.push(e.path());
                } else {
                    let f = fs::File::options().write(true).open(e.path()).unwrap();
                    f.set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(time as u64)).unwrap();
                }
            }
        }
    }
    let search = |h: &mut Harness, key: u32, order: u32, capacity: u32| -> (HleResult, Vec<String>, Vec<(u64, u64)>, u32) {
        let cond = h.alloc(0x40);
        h.put32(cond, 0, 1);
        h.put32(cond, 0x18, key);
        h.put32(cond, 0x1C, order);
        let names = h.alloc(DIR_NAME_SIZE * 4);
        let infos = h.alloc(BLOCKS_INFO_SIZE * 4);
        let result = h.alloc(SEARCH_RESULT_SIZE);
        h.put64(result, 8, names);
        h.put32(result, 0x10, capacity);
        h.put64(result, 0x20, infos);
        let status = h.lib.dir_name_search(&h.ctx, cond, result);
        let set = h.ctx.read_u32(result + 0x14).unwrap();
        let found = (0..set as u64).map(|i| text(&h.read(names + i * 32, 32))).collect();
        let blocks = (0..set as u64).map(|i| (h.ctx.read_u64(infos + i * 0x30).unwrap(), h.ctx.read_u64(infos + i * 0x30 + 8).unwrap())).collect();
        (status, found, blocks, h.ctx.read_u32(result).unwrap())
    };
    let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let (status, found, blocks, hits) = search(&mut h, 0, 0, 4);
    assert_eq!((status, hits), (OK, 3));
    assert_eq!(found, names(&["SORT_A", "SORT_B", "SORT_C"]));
    // 40000 bytes take one 64 KiB block, one byte one block.
    assert_eq!(blocks, vec![(200, 200), (96, 95), (150, 149)]);
    assert_eq!(search(&mut h, 0, 1, 4).1, names(&["SORT_C", "SORT_B", "SORT_A"]));
    assert_eq!(search(&mut h, 1, 0, 4).1, names(&["SORT_B", "SORT_C", "SORT_A"]), "user param");
    assert_eq!(search(&mut h, 1, 1, 4).1, names(&["SORT_A", "SORT_C", "SORT_B"]));
    assert_eq!(search(&mut h, 2, 0, 4).1, names(&["SORT_B", "SORT_C", "SORT_A"]), "blocks");
    assert_eq!(search(&mut h, 3, 0, 4).1, names(&["SORT_B", "SORT_A", "SORT_C"]), "mtime");
    let (_, newest, _, hits) = search(&mut h, 3, 1, 1);
    assert_eq!((newest, hits), (names(&["SORT_C"]), 3), "the newest save first");
    assert_eq!(search(&mut h, 5, 0, 4).1, names(&["SORT_B", "SORT_C", "SORT_A"]), "free blocks");
    assert_eq!(search(&mut h, 4, 0, 4).0, ERR_PARAMETER);
    assert_eq!(search(&mut h, 6, 0, 4).0, ERR_PARAMETER);
    assert_eq!(search(&mut h, 0, 2, 4).0, ERR_PARAMETER);

    // A mount that fails leaves the caller's result as it was.
    let d = h.dir_name("MISSING");
    let m = h.mount3_struct(1, d, 1, 0);
    let r = h.alloc(MOUNT_RESULT_SIZE);
    h.put(r, 0, &[0x5A; MOUNT_RESULT_SIZE]);
    assert_eq!(h.lib.mount3(&h.ctx, m, r), ERR_NOT_FOUND);
    assert_eq!(h.read(r, MOUNT_RESULT_SIZE), vec![0x5A; MOUNT_RESULT_SIZE]);
    fs::remove_dir_all(&tmp).unwrap();
}
