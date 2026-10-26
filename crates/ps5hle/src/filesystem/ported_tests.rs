// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Kernel file-system behaviour: mount roots, unmapped host paths,
//! directory streams, Unicode names, save-file visibility and renames, and
//! AIO batches.

use super::*;
use crate::context::CpuRegister;
use ps5vm::{MemProtect, VirtualMemory};

/// Guest scratch memory plus a host temp directory mounted at `mount`.
struct Fixture {
    vm: &'static VirtualMemory,
    base: u64,
    root: PathBuf,
    mount: String,
}

impl Fixture {
    fn new(base: u64, tag: &str) -> Self {
        let vm: &'static VirtualMemory = Box::leak(Box::new(VirtualMemory::new()));
        vm.map(base, 0x40000, MemProtect::RW).unwrap();
        let root = std::env::temp_dir().join(format!("rps5e-fs-ported-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let mount = format!("/k4{tag}0");
        assert!(register_guest_path_mount(&mount, &root));
        Fixture { vm, base, root, mount }
    }

    fn ctx(&self) -> CpuContext<'static> {
        CpuContext::new(self.vm)
    }

    /// Write a guest string at slot `n` (0x400 bytes each) and return it.
    fn s(&self, ctx: &CpuContext, n: u64, text: &str) -> u64 {
        let addr = self.base + 0x2_0000 + n * 0x400;
        let mut b = text.as_bytes().to_vec();
        b.push(0);
        ctx.write_bytes(addr, &b);
        addr
    }

    fn guest(&self, rel: &str) -> String {
        format!("{}/{}", self.mount, rel)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        unregister_guest_path_mount(&self.mount);
        let _ = fs::remove_dir_all(&self.root);
        let _ = self.vm.unmap(self.base, 0x40000);
    }
}

fn call(ctx: &mut CpuContext, cb: HleCallback, args: &[u64]) -> HleResult {
    use CpuRegister::*;
    for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].iter().zip(args) {
        ctx.set(*r, *v);
    }
    cb(ctx)
}

fn open(ctx: &mut CpuContext, path: u64, flags: u64) -> HleResult {
    call(ctx, cb_sce_open, &[path, flags, 0o777])
}

fn stat_size(ctx: &CpuContext, buf: u64) -> i64 {
    ctx.read_u64(buf + 72).unwrap() as i64
}

/// Walk the records of a getdents buffer: (name, type).
fn records(bytes: &[u8]) -> Vec<(String, u8)> {
    let mut out = Vec::new();
    let mut off = 0;
    while off + 8 <= bytes.len() {
        let len = u16::from_le_bytes([bytes[off + 4], bytes[off + 5]]) as usize;
        let namlen = bytes[off + 7] as usize;
        assert!(len >= 8 + namlen + 1 && len <= bytes.len() - off, "record fits");
        out.push((String::from_utf8_lossy(&bytes[off + 8..off + 8 + namlen]).into_owned(), bytes[off + 6]));
        off += len;
    }
    out
}

#[test]
fn mount_root_lists_and_prefixes_end_at_components() {
    let f = Fixture::new(0x5CD0_0000_0000, "root");
    let mut ctx = f.ctx();
    fs::write(f.root.join("rpf.cache"), b"").unwrap();
    assert_eq!(resolve_guest_path(&f.guest("rpf.cache")), Some(f.root.join("rpf.cache")));
    assert_eq!(resolve_guest_path(&format!("{}1/rpf.cache", f.mount)), None, "prefix ends at a component");
    let buf = f.base;
    for path in [f.mount.clone(), format!("{}/", f.mount)] {
        for flags in [0, O_DIRECTORY] {
            let p = f.s(&ctx, 0, &path);
            let fd = open(&mut ctx, p, flags);
            assert!(fd >= 3, "open mounted root {path} flags {flags:#x}: {fd:#x}");
            let n = call(&mut ctx, cb_sce_getdents, &[fd as u64, buf, 512]);
            assert!(n > 0 && n <= 512);
            let recs = records(&ctx.read_bytes(buf, n as usize).unwrap());
            assert!(recs.iter().any(|(name, ty)| name == "rpf.cache" && *ty == DT_REG), "listing contains rpf.cache");
            assert_eq!(call(&mut ctx, cb_sce_close, &[fd as u64]), 0);
        }
    }
    assert!(unregister_guest_path_mount(&f.mount));
    assert_eq!(resolve_guest_path(&f.guest("rpf.cache")), None, "unmount by guest path");
}

#[test]
fn unmapped_host_paths_are_absent() {
    let f = Fixture::new(0x5CD1_0000_0000, "unmapped");
    let mut ctx = f.ctx();
    let host_file = f.root.join("host-only.dat");
    fs::write(&host_file, b"x").unwrap();
    let host_path = host_file.to_string_lossy().into_owned();
    let p = f.s(&ctx, 0, &host_path);
    let buf = f.base;
    assert_eq!(resolve_guest_path(&host_path), None);
    assert_eq!(open(&mut ctx, p, 0), sce_error(ENOENT));
    assert_eq!(call(&mut ctx, cb_sce_stat, &[p, buf]), sce_error(ENOENT));
    assert_eq!(call(&mut ctx, cb_sce_check_reachability, &[p]), sce_error(ENOENT));
    assert_eq!(call(&mut ctx, cb_sce_unlink, &[p]), sce_error(ENOENT));
    let dir = f.s(&ctx, 1, &f.root.to_string_lossy());
    assert_eq!(call(&mut ctx, cb_sce_rmdir, &[dir]), sce_error(ENOENT));
    assert!(host_file.exists(), "unmapped files cannot be removed");
    let missing = f.root.join("unmapped-create").to_string_lossy().into_owned();
    let m = f.s(&ctx, 2, &missing);
    assert_eq!(open(&mut ctx, m, 0x601), sce_error(ENOENT));
    assert_eq!(call(&mut ctx, cb_sce_mkdir, &[m, 0o777]), sce_error(ENOENT));
    assert!(!f.root.join("unmapped-create").exists(), "creation needs a mounted destination");
    let from = f.s(&ctx, 3, &f.guest("host-only.dat"));
    assert_eq!(call(&mut ctx, cb_sce_rename, &[from, m]), sce_error(ENOENT));
    assert!(host_file.exists() && !f.root.join("unmapped-create").exists(), "rename keeps the source");
}

#[test]
fn directory_stream_semantics() {
    let f = Fixture::new(0x5CD2_0000_0000, "dirstream");
    let mut ctx = f.ctx();
    for i in 0..48 {
        fs::write(f.root.join(format!("directory-entry-{i}")), b"").unwrap();
    }
    let p = f.s(&ctx, 0, &format!("{}/", f.mount));
    let fd = open(&mut ctx, p, 0) as u64;
    assert!(fd >= 3);
    let (stat, raw_buf, block, basep) = (f.base, f.base + 0x1000, f.base + 0x8000, f.base + 0x900);
    let end = call(&mut ctx, cb_sce_lseek, &[fd, 0, SEEK_END as u64]);
    assert!(end > 512 && end % 512 == 0, "SEEK_END is the padded stream size");
    assert_eq!(call(&mut ctx, cb_sce_fstat, &[fd, stat]), 0);
    assert_eq!(stat_size(&ctx, stat), end);
    assert_eq!(ctx.read_u32(stat + 88), Some(512), "st_blksize");
    assert_eq!(call(&mut ctx, cb_sce_stat, &[p, stat]), 0);
    assert_eq!(stat_size(&ctx, stat), end);
    assert_eq!(ctx.read_u64(stat + 80), Some(end as u64 / 512), "st_blocks");
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, 0, SEEK_SET as u64]), 0);
    assert_eq!(call(&mut ctx, cb_sce_read, &[fd, raw_buf, end as u64]), end, "raw read of the whole stream");
    assert_eq!(call(&mut ctx, cb_sce_read, &[fd, raw_buf + end as u64, 1]), 0, "then EOF");
    let raw = ctx.read_bytes(raw_buf, end as usize).unwrap();
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, (-end) as u64, SEEK_CUR as u64]), 0, "SEEK_CUR follows reads");

    let mut offset = 0i64;
    while offset < end {
        assert_eq!(call(&mut ctx, cb_sce_getdirentries, &[fd, block, 512, basep]), 512);
        assert_eq!(ctx.read_u64(basep), Some(offset as u64), "base reports each block position");
        let b = ctx.read_bytes(block, 512).unwrap();
        assert_eq!(&b[..], &raw[offset as usize..offset as usize + 512]);
        let mut pos = 0;
        while pos < 512 {
            let len = u16::from_le_bytes([b[pos + 4], b[pos + 5]]) as usize;
            let namlen = b[pos + 7] as usize;
            assert!(len >= 9 + namlen && len <= 512 - pos && b[pos + 8 + namlen] == 0, "records stay within blocks");
            pos += len;
        }
        offset += 512;
    }
    assert_eq!(call(&mut ctx, cb_sce_getdirentries, &[fd, block, 512, basep]), 0);
    assert_eq!(ctx.read_u64(basep), Some(end as u64), "EOF position");
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, 512, SEEK_SET as u64]), 512);
    assert_eq!(call(&mut ctx, cb_sce_getdirentries, &[fd, block, 512, basep]), 512);
    assert_eq!(ctx.read_u64(basep), Some(512));
    assert_eq!(&ctx.read_bytes(block, 512).unwrap()[..], &raw[512..1024], "restore and reread a position");

    let position = call(&mut ctx, cb_sce_lseek, &[fd, 0, SEEK_CUR as u64]);
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, 0, 9]), sce_error(EINVAL));
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, (-1i64) as u64, SEEK_SET as u64]), sce_error(EINVAL));
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, i64::MIN as u64, SEEK_CUR as u64]), sce_error(EINVAL));
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, i64::MAX as u64, SEEK_CUR as u64]), sce_error(EOVERFLOW));
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, 0, SEEK_CUR as u64]), position, "failed seeks keep the position");
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, (-19i64) as u64, SEEK_END as u64]), end - 19);
    assert_eq!(call(&mut ctx, cb_sce_read, &[fd, block, 512]), 19);
    assert_eq!(&ctx.read_bytes(block, 19).unwrap()[..], &raw[end as usize - 19..]);
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, 1, SEEK_SET as u64]), 1);
    assert_eq!(call(&mut ctx, cb_sce_getdents, &[fd, block, 512]), sce_error(EINVAL), "misaligned position");
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, 0, SEEK_CUR as u64]), 1);
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, 512, SEEK_END as u64]), end + 512);
    assert_eq!(call(&mut ctx, cb_sce_getdirentries, &[fd, block, 512, basep]), sce_error(EINVAL), "beyond EOF");
    assert_eq!(call(&mut ctx, cb_sce_read, &[fd, block, i32::MAX as u64 + 1]), sce_error(EINVAL), "oversized read");
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, 0, SEEK_CUR as u64]), end + 512);
    assert_eq!(call(&mut ctx, cb_sce_close, &[fd]), 0);
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, 0, SEEK_SET as u64]), sce_error(EBADF));
}

#[test]
fn unicode_paths_resolve() {
    let f = Fixture::new(0x5CD3_0000_0000, "unicode");
    let mut ctx = f.ctx();
    let nested = f.root.join("Test\u{e9}-\u{65e5}\u{672c}\u{8a9e}").join("nested");
    fs::create_dir_all(&nested).unwrap();
    let name = "asset-\u{e9}-\u{65e5}\u{672c}\u{8a9e}.bin";
    fs::write(nested.join(name), b"").unwrap();
    let guest = f.guest(&format!("Test\u{e9}-\u{65e5}\u{672c}\u{8a9e}/nested/{name}"));
    assert_eq!(resolve_guest_path(&guest), Some(nested.join(name)));
    let p = f.s(&ctx, 0, &guest);
    let fd = open(&mut ctx, p, 0);
    assert!(fd >= 3, "open Unicode guest path: {fd:#x}");
    assert_eq!(call(&mut ctx, cb_sce_close, &[fd as u64]), 0);
    // The listing returns UTF-8 names.
    let d = f.s(&ctx, 1, &f.guest(&format!("Test\u{e9}-\u{65e5}\u{672c}\u{8a9e}/nested")));
    let dfd = open(&mut ctx, d, O_DIRECTORY) as u64;
    let n = call(&mut ctx, cb_sce_getdents, &[dfd, f.base, 512]);
    let recs = records(&ctx.read_bytes(f.base, n as usize).unwrap());
    assert!(recs.iter().any(|(n, t)| n == name && *t == DT_REG));
    assert_eq!(call(&mut ctx, cb_sce_close, &[dfd]), 0);
}

#[test]
fn save_open_visibility() {
    let f = Fixture::new(0x5CD4_0000_0000, "savevis");
    let mut ctx = f.ctx();
    let p = f.s(&ctx, 0, &f.guest("visible-save.dat"));
    let (stat, data) = (f.base, f.base + 0x800);
    let fd = open(&mut ctx, p, 0xa01);
    assert!(fd >= 3, "exclusive create");
    assert_eq!(call(&mut ctx, cb_sce_stat, &[p, stat]), 0);
    assert_eq!(stat_size(&ctx, stat), 0, "created file is visible before close");
    assert_eq!(open(&mut ctx, p, 0xa01), sce_error(EEXIST), "exclusive creation sees an open file");
    ctx.write_bytes(data, b"saved progress");
    assert_eq!(call(&mut ctx, cb_sce_write, &[fd as u64, data, 14]), 14);
    assert_eq!(call(&mut ctx, cb_sce_close, &[fd as u64]), 0);
    assert_eq!(call(&mut ctx, cb_sce_stat, &[p, stat]), 0);
    assert_eq!(stat_size(&ctx, stat), 14);
    let t = open(&mut ctx, p, 0x401);
    assert!(t >= 3, "truncate existing save");
    assert_eq!(call(&mut ctx, cb_sce_stat, &[p, stat]), 0);
    assert_eq!(stat_size(&ctx, stat), 0, "truncation visible before close");
    assert_eq!(call(&mut ctx, cb_sce_close, &[t as u64]), 0);
    // O_TRUNC needs write access.
    assert_eq!(open(&mut ctx, p, 0x400), sce_error(EACCES));
}

#[test]
fn save_rename_keeps_open_descriptor() {
    let f = Fixture::new(0x5CD5_0000_0000, "saverename");
    let mut ctx = f.ctx();
    for payload in ["first-save", "replacement-save"] {
        let src = f.s(&ctx, 0, &f.guest("STEMP000.DAT"));
        let dst = f.s(&ctx, 1, &f.guest("SDATA000.DAT"));
        let data = f.base;
        let fd = open(&mut ctx, src, 0x601) as u64;
        assert!(fd >= 3);
        ctx.write_bytes(data, payload.as_bytes());
        assert_eq!(call(&mut ctx, cb_sce_write, &[fd, data, payload.len() as u64]), payload.len() as i64);
        assert_eq!(call(&mut ctx, cb_sce_rename, &[src, dst]), 0, "rename an open save file");
        ctx.write_bytes(data, b"-after-rename");
        assert_eq!(call(&mut ctx, cb_sce_write, &[fd, data, 13]), 13, "write through the renamed descriptor");
        assert_eq!(call(&mut ctx, cb_sce_close, &[fd]), 0);
        assert_eq!(fs::read(f.root.join("SDATA000.DAT")).unwrap(), format!("{payload}-after-rename").as_bytes());
    }
}

#[test]
fn aio_cancel_requests_and_set_param() {
    let f = Fixture::new(0x7E63_0000_0000, "aiocancel");
    let mut ctx = f.ctx();
    let path = f.s(&ctx, 0, &f.guest("c.dat"));
    let (req, result, ids, states, buffer, param) =
        (f.base + 0x100, f.base + 0x200, f.base + 0x300, f.base + 0x400, f.base + 0x600, f.base + 0x700);
    let fd = open(&mut ctx, path, 0x602);
    let mut r = Vec::new();
    for v in [0u64, 0, buffer, result] {
        r.extend_from_slice(&v.to_le_bytes());
    }
    r.extend_from_slice(&(fd as i32).to_le_bytes());
    r.extend_from_slice(&[0u8; 4]);
    ctx.write_bytes(req, &r);
    assert_eq!(call(&mut ctx, cb_aio_submit_read, &[req, 1, 2, ids]), 0);
    ctx.write_u32(ids + 4, 0);
    assert_eq!(call(&mut ctx, cb_aio_cancel_requests, &[ids, 2, states]), 0);
    assert_eq!(ctx.read_u32(states), Some(AIO_STATE_COMPLETED as u32), "a finished request stays finished");
    assert_eq!(ctx.read_u32(states + 4), Some(AIO_STATE_PROCESSING as u32), "id 0");
    ctx.write_u32(ids + 4, 9999);
    assert_eq!(call(&mut ctx, cb_aio_cancel_requests, &[ids, 2, states]), sce_error(EINVAL));
    assert_eq!(call(&mut ctx, cb_aio_cancel_requests, &[0, 1, states]), sce_error(EFAULT));
    assert_eq!(call(&mut ctx, cb_aio_cancel_requests, &[ids, u32::MAX as u64, states]), sce_error(EINVAL));
    assert_eq!(call(&mut ctx, cb_aio_cancel_requests, &[ids, 129, states]), sce_error(EINVAL));

    assert_eq!(call(&mut ctx, cb_aio_set_param, &[param, 8, 4, 1, 0x10000, 0x8000]), 0);
    let fields: Vec<u32> = (0..5).map(|i| ctx.read_u32(param + i * 4).unwrap()).collect();
    assert_eq!(fields, vec![8, 4, 1, 0x10000, 0x8000]);
    assert_eq!(call(&mut ctx, cb_aio_set_param, &[0, 8, 4, 1, 0, 0]), sce_error(EFAULT));
    assert_eq!(call(&mut ctx, cb_aio_set_param, &[param, 129, 4, 1, 0, 0]), sce_error(EINVAL));
    assert_eq!(call(&mut ctx, cb_aio_set_param, &[param, 8, 4, 2, 0, 0]), sce_error(EINVAL));
    assert_eq!(call(&mut ctx, cb_aio_set_param, &[param, 8, 4, 1, 0x100_0001, 0]), sce_error(EINVAL));
}

#[test]
fn aio_batches() {
    let f = Fixture::new(0x5CD6_0000_0000, "aio");
    let mut ctx = f.ctx();
    let path = f.s(&ctx, 0, &f.guest("aio.dat"));
    let payload = b"AIO payload\0";
    let (data, req, result, ids, states, timeout, buffer) =
        (f.base, f.base + 0x100, f.base + 0x200, f.base + 0x300, f.base + 0x400, f.base + 0x500, f.base + 0x600);
    let fd = open(&mut ctx, path, 0x602);
    assert!(fd >= 3);
    ctx.write_bytes(data, payload);
    assert_eq!(call(&mut ctx, cb_sce_write, &[fd as u64, data, payload.len() as u64]), payload.len() as i64);
    // Request { offset 2, size 4, buffer, result*, fd }
    let mut r = Vec::new();
    r.extend_from_slice(&2i64.to_le_bytes());
    r.extend_from_slice(&4u64.to_le_bytes());
    r.extend_from_slice(&buffer.to_le_bytes());
    r.extend_from_slice(&result.to_le_bytes());
    r.extend_from_slice(&(fd as i32).to_le_bytes());
    r.extend_from_slice(&[0u8; 4]);
    ctx.write_bytes(req, &r);
    for i in 0..2 {
        assert_eq!(call(&mut ctx, cb_aio_submit_read, &[req, 1, 2, ids + i * 4]), 0);
        assert_eq!(ctx.read_u64(result), Some(4));
        assert_eq!(ctx.read_u32(result + 8), Some(AIO_STATE_COMPLETED as u32));
        assert_eq!(&ctx.read_bytes(buffer, 4).unwrap()[..], &payload[2..6]);
    }
    let id = |ctx: &CpuContext, i: u64| ctx.read_u32(ids + i * 4).unwrap() as u64;
    ctx.write_u32(ids + 8, -1i32 as u32);
    let esrch = sce_error(ESRCH) as i32 as u32;
    let st = |ctx: &CpuContext, i: u64| ctx.read_u32(states + i * 4).unwrap();
    ctx.write_u32(timeout, 0);
    assert_eq!(call(&mut ctx, cb_aio_wait_requests, &[ids, 3, states, AIO_WAIT_OR as u64, timeout]), 0);
    assert_eq!((st(&ctx, 0), st(&ctx, 1), st(&ctx, 2)), (3, 3, esrch), "OR wait with a zero timeout");
    let notified = (AIO_STATE_COMPLETED | AIO_STATE_NOTIFIED) as u32;
    let id0 = id(&ctx, 0);
    assert_eq!(call(&mut ctx, cb_aio_poll_request, &[id0, states]), 0);
    assert_eq!(st(&ctx, 0), notified);
    assert_eq!(call(&mut ctx, cb_aio_poll_requests, &[ids, 2, states]), 0);
    assert_eq!((st(&ctx, 0), st(&ctx, 1)), (notified, notified));
    assert_eq!(call(&mut ctx, cb_aio_wait_request, &[id0, states, 0]), 0);
    assert_eq!(st(&ctx, 0), notified, "single and batch calls share notification state");
    ctx.write_u32(timeout, 1_000_000);
    assert_eq!(call(&mut ctx, cb_aio_wait_requests, &[ids, 3, states, AIO_WAIT_AND as u64, timeout]), 0);
    assert_eq!((st(&ctx, 0), st(&ctx, 1), st(&ctx, 2)), (notified, notified, esrch));
    assert!(ctx.read_u32(timeout).unwrap() <= 1_000_000);
    assert_eq!(call(&mut ctx, cb_aio_wait_requests, &[ids, 3, states, AIO_WAIT_OR as u64, 0]), 0);
    assert_eq!(call(&mut ctx, cb_aio_wait_requests, &[ids, 1, states, 0, 0]), 0, "one-id waits ignore the mode");
    assert_eq!(call(&mut ctx, cb_aio_delete_requests, &[ids, 3, states]), 0);
    assert_eq!((st(&ctx, 0), st(&ctx, 1), st(&ctx, 2)), (0, 0, esrch), "per-request delete results");
    assert_eq!(call(&mut ctx, cb_aio_poll_request, &[id0, states]), 0);
    assert_eq!(st(&ctx, 0), esrch);
    let id1 = id(&ctx, 1);
    assert_eq!(call(&mut ctx, cb_aio_delete_request, &[id1, states + 4]), 0);
    assert_eq!(st(&ctx, 1), esrch, "deleted ids are invalid");

    let efault = sce_error(EFAULT);
    let einval = sce_error(EINVAL);
    for batch in [cb_aio_poll_requests as HleCallback, cb_aio_delete_requests] {
        for i in 0..3 {
            ctx.write_u32(states + i * 4, 42);
        }
        assert_eq!(call(&mut ctx, batch, &[0, 1, states]), efault);
        assert_eq!(call(&mut ctx, batch, &[ids, 1, 0]), efault);
        assert_eq!(call(&mut ctx, batch, &[ids, 0, states]), einval);
        assert_eq!(call(&mut ctx, batch, &[ids, 129, states]), einval);
        assert_eq!((st(&ctx, 0), st(&ctx, 1), st(&ctx, 2)), (42, 42, 42), "invalid arguments leave outputs");
        let (zero_ids, errors) = (f.base + 0x1000, f.base + 0x2000);
        ctx.write_bytes(zero_ids, &[0u8; 512]);
        assert_eq!(call(&mut ctx, batch, &[zero_ids, 128, errors]), 0);
        assert!((0..128).all(|i| ctx.read_u32(errors + i * 4) == Some(esrch)), "128 invalid ids");
    }
    for i in 0..3 {
        ctx.write_u32(states + i * 4, 42);
    }
    let w = cb_aio_wait_requests as HleCallback;
    assert_eq!(call(&mut ctx, w, &[0, 1, states, 1, 0]), efault);
    assert_eq!(call(&mut ctx, w, &[ids, 1, 0, 1, 0]), efault);
    assert_eq!(call(&mut ctx, w, &[ids, 0, states, 1, 0]), einval);
    assert_eq!(call(&mut ctx, w, &[ids, 129, states, 1, 0]), einval);
    assert_eq!(call(&mut ctx, w, &[ids, 2, states, 0, 0]), einval);
    assert_eq!(call(&mut ctx, w, &[ids, 2, states, 3, 0]), einval);
    assert_eq!((st(&ctx, 0), st(&ctx, 1), st(&ctx, 2)), (42, 42, 42));
    let (zero_ids, errors) = (f.base + 0x1000, f.base + 0x2000);
    assert_eq!(call(&mut ctx, w, &[zero_ids, 128, errors, 1, 0]), 0);
    assert!((0..128).all(|i| ctx.read_u32(errors + i * 4) == Some(esrch)));
    assert_eq!(call(&mut ctx, cb_sce_close, &[fd as u64]), 0);
}

#[test]
fn preadv_pwritev_and_flock() {
    let f = Fixture::new(0x5CD7_0000_0000, "iov");
    let mut ctx = f.ctx();
    let path = f.s(&ctx, 0, &f.guest("iov.dat"));
    let fd = open(&mut ctx, path, 0x602) as u64;
    let (iov, a, b) = (f.base, f.base + 0x100, f.base + 0x200);
    ctx.write_bytes(a, b"hello ");
    ctx.write_bytes(b, b"world");
    ctx.write_u64(iov, a);
    ctx.write_u64(iov + 8, 6);
    ctx.write_u64(iov + 16, b);
    ctx.write_u64(iov + 24, 5);
    assert_eq!(call(&mut ctx, cb_sce_pwritev, &[fd, iov, 2, 3]), 11);
    assert_eq!(fs::read(f.root.join("iov.dat")).unwrap(), b"\0\0\0hello world");
    ctx.write_bytes(a, &[0u8; 6]);
    ctx.write_bytes(b, &[0u8; 5]);
    assert_eq!(call(&mut ctx, cb_sce_preadv, &[fd, iov, 2, 3]), 11);
    assert_eq!(&ctx.read_bytes(a, 6).unwrap()[..], b"hello ");
    assert_eq!(&ctx.read_bytes(b, 5).unwrap()[..], b"world");
    // The file position is unchanged.
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, 0, SEEK_CUR as u64]), 0);
    assert_eq!(call(&mut ctx, cb_sce_preadv, &[fd, iov, -1i64 as u64, 0]), sce_error(EINVAL));
    assert_eq!(call(&mut ctx, cb_sce_preadv, &[fd, iov, 2, -1i64 as u64]), sce_error(EINVAL));
    assert_eq!(call(&mut ctx, cb_sce_preadv, &[fd, 0, 2, 0]), sce_error(EFAULT));
    assert_eq!(call(&mut ctx, cb_sce_preadv, &[999, iov, 2, 0]), sce_error(EBADF));
    assert_eq!(call(&mut ctx, cb_sce_preadv, &[fd, iov, 0, 0]), 0);
    // The POSIX names: the same transfers, -1 (errno) on failure, and the
    // file position stays put.
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, 4, SEEK_SET as u64]), 4);
    ctx.write_bytes(a, &[0u8; 6]);
    assert_eq!(call(&mut ctx, cb_posix_preadv, &[fd, iov, 2, 1]), 11);
    assert_eq!(&ctx.read_bytes(a, 6).unwrap()[..], b"\0\0hell");
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, 0, SEEK_CUR as u64]), 4);
    assert_eq!(call(&mut ctx, cb_posix_preadv, &[fd, iov, 2, 14]), 0, "at the end");
    assert_eq!(call(&mut ctx, cb_posix_preadv, &[fd, iov, 2, -1i64 as u64]), -1);
    assert_eq!(call(&mut ctx, cb_posix_preadv, &[fd, iov, -1i64 as u64, 0]), -1);
    assert_eq!(call(&mut ctx, cb_posix_preadv, &[fd, 0, 1, 0]), -1);
    ctx.write_bytes(a, b"XY");
    ctx.write_u64(iov + 8, 2);
    assert_eq!(call(&mut ctx, cb_posix_pwritev, &[fd, iov, 1, 0]), 2);
    assert_eq!(&fs::read(f.root.join("iov.dat")).unwrap()[..4], b"XY\0h");
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, 0, SEEK_CUR as u64]), 4);
    assert_eq!(call(&mut ctx, cb_posix_pwritev, &[fd, iov, 1, -1i64 as u64]), -1);
    assert_eq!(call(&mut ctx, cb_posix_pwritev, &[fd, 0, 1, 0]), -1);
    // flock: POSIX, accepted for open files.
    assert_eq!(call(&mut ctx, cb_posix_flock, &[fd, 2]), 0);
    assert_eq!(call(&mut ctx, cb_posix_flock, &[fd, 8]), 0);
    assert_eq!(call(&mut ctx, cb_posix_flock, &[999, 2]), -1);
    let d = f.s(&ctx, 1, &f.mount);
    let dfd = open(&mut ctx, d, O_DIRECTORY) as u64;
    assert_eq!(call(&mut ctx, cb_sce_preadv, &[dfd, iov, 2, 0]), sce_error(EISDIR));
    assert_eq!(call(&mut ctx, cb_sce_close, &[dfd]), 0);
    assert_eq!(call(&mut ctx, cb_sce_close, &[fd]), 0);
}

/// mkdir / rmdir: a file used as a directory in the path is ENOTDIR, a
/// missing parent ENOENT, a trailing separator is accepted for a new or an
/// existing directory (reference GuestFilesystem cases).
#[test]
fn mkdir_rmdir_report_a_file_in_the_path() {
    let f = Fixture::new(0x7C56_0000_0000, "mkdirnd");
    let mut ctx = f.ctx();
    let errno = crate::kernel::bind_test_errno(&ctx);
    fs::write(f.root.join("present"), b"x").unwrap();
    let path = |ctx: &CpuContext, n: u64, rel: &str| f.s(ctx, n, &f.guest(rel));
    let posix = |ctx: &mut CpuContext, cb: HleCallback, p: u64| -> (HleResult, u32) {
        let r = call(ctx, cb, &[p, 0o700]);
        (r, ctx.read_u32(errno).unwrap())
    };
    let enotdir = sce_error(ENOTDIR);
    for rel in ["present/child", "present/child/leaf"] {
        let p = path(&ctx, 0, rel);
        assert_eq!(posix(&mut ctx, cb_posix_mkdir, p), (-1, 20), "mkdir {rel}");
        assert_eq!(call(&mut ctx, cb_sce_mkdir, &[p, 0o700]), enotdir, "sceKernelMkdir {rel}");
        assert_eq!(posix(&mut ctx, cb_posix_rmdir, p), (-1, 20), "rmdir {rel}");
        assert_eq!(call(&mut ctx, cb_sce_rmdir, &[p]), enotdir, "sceKernelRmdir {rel}");
    }
    let p = path(&ctx, 0, "present/");
    assert_eq!(posix(&mut ctx, cb_posix_rmdir, p), (-1, 20));
    assert_eq!(call(&mut ctx, cb_sce_rmdir, &[p]), enotdir);
    let p = path(&ctx, 0, "missing/");
    assert_eq!(posix(&mut ctx, cb_posix_rmdir, p), (-1, 2));
    assert_eq!(call(&mut ctx, cb_sce_rmdir, &[p]), sce_error(ENOENT));
    let p = path(&ctx, 0, "missing/leaf");
    assert_eq!(call(&mut ctx, cb_sce_mkdir, &[p, 0o700]), sce_error(ENOENT));
    assert!(f.root.join("present").is_file());
    for (cb_mk, cb_rm, rel) in [(cb_posix_mkdir as HleCallback, cb_posix_rmdir as HleCallback, "trailing_posix"), (cb_sce_mkdir, cb_sce_rmdir, "trailing_kernel")] {
        let p = path(&ctx, 0, &format!("{rel}/"));
        assert_eq!(call(&mut ctx, cb_mk, &[p, 0o700]), 0, "mkdir {rel}/");
        assert!(f.root.join(rel).is_dir());
        let p = path(&ctx, 1, rel);
        assert_eq!(call(&mut ctx, cb_rm, &[p]), 0, "rmdir {rel}");
        assert!(!f.root.join(rel).exists());
    }
}

/// rmdir of a path whose last component is "." is EINVAL for a
/// directory, ENOTDIR for a file and ENOENT for a missing path; the
/// directory stays (reference GuestFilesystem cases).
#[test]
fn rmdir_rejects_a_final_dot() {
    let f = Fixture::new(0x7C57_0000_0000, "rmdot");
    let mut ctx = f.ctx();
    let errno = crate::kernel::bind_test_errno(&ctx);
    fs::create_dir(f.root.join("dir")).unwrap();
    fs::write(f.root.join("present"), b"x").unwrap();
    for (rel, posix_errno) in [("dir/.", 22u32), ("dir/./", 22), ("present/.", 20), ("missing/.", 2)] {
        let p = f.s(&ctx, 0, &f.guest(rel));
        assert_eq!(call(&mut ctx, cb_posix_rmdir, &[p]), -1, "rmdir {rel}");
        assert_eq!(ctx.read_u32(errno), Some(posix_errno), "rmdir {rel}");
        assert_eq!(call(&mut ctx, cb_sce_rmdir, &[p]), sce_error(posix_errno as i32), "sceKernelRmdir {rel}");
    }
    assert!(f.root.join("dir").is_dir());
    let p = f.s(&ctx, 0, &f.guest("dir"));
    assert_eq!(call(&mut ctx, cb_posix_rmdir, &[p]), 0);
}

/// An empty path is ENOENT for every path call (reference GuestFilesystem
/// cases).
#[test]
fn empty_paths_are_enoent() {
    let f = Fixture::new(0x7C58_0000_0000, "emptyp");
    let mut ctx = f.ctx();
    let errno = crate::kernel::bind_test_errno(&ctx);
    fs::write(f.root.join("present"), b"x").unwrap();
    let empty = f.s(&ctx, 0, "");
    let present = f.s(&ctx, 1, &f.guest("present"));
    let stat = f.base;
    let enoent = sce_error(ENOENT);
    assert_eq!(call(&mut ctx, cb_sce_open, &[empty, 0, 0]), enoent, "sceKernelOpen");
    assert_eq!(call(&mut ctx, cb_sce_stat, &[empty, stat]), enoent, "sceKernelStat");
    assert_eq!(call(&mut ctx, cb_sce_check_reachability, &[empty]), enoent, "sceKernelCheckReachability");
    assert_eq!(call(&mut ctx, cb_sce_rename, &[empty, present]), enoent, "sceKernelRename from");
    assert_eq!(call(&mut ctx, cb_sce_rename, &[present, empty]), enoent, "sceKernelRename to");
    assert_eq!(call(&mut ctx, cb_sce_rmdir, &[empty]), enoent, "sceKernelRmdir");
    assert_eq!(call(&mut ctx, cb_sce_mkdir, &[empty, 0o700]), enoent, "sceKernelMkdir");
    assert_eq!(call(&mut ctx, cb_sce_chmod, &[empty, 0o700]), enoent, "sceKernelChmod");
    assert_eq!(call(&mut ctx, cb_sce_truncate, &[empty, 0]), enoent, "sceKernelTruncate");
    assert_eq!(call(&mut ctx, cb_sce_utimes, &[empty, 0]), enoent, "sceKernelUtimes");
    assert_eq!(call(&mut ctx, cb_sce_unlink, &[empty]), enoent, "sceKernelUnlink");
    assert_eq!(call(&mut ctx, cb_posix_chmod, &[empty, 0o600]), -1, "chmod");
    assert_eq!(ctx.read_u32(errno), Some(2));
    assert!(f.root.join("present").is_file());
}

/// truncate / sceKernelTruncate: shrink, grow with zeros, a negative
/// length EINVAL, a missing or empty path ENOENT, NULL EFAULT, a directory
/// EISDIR, a read-only file EACCES (reference GuestTruncate cases).
#[test]
fn truncate_sizes_and_errors() {
    let f = Fixture::new(0x7C59_0000_0000, "trunc");
    let mut ctx = f.ctx();
    let errno = crate::kernel::bind_test_errno(&ctx);
    let host = f.root.join("sized.txt");
    fs::write(&host, b"0123456789abcdef").unwrap();
    let p = f.s(&ctx, 0, &f.guest("sized.txt"));
    let fail = |ctx: &mut CpuContext, path: u64, len: i64| -> u32 {
        assert_eq!(call(ctx, cb_posix_truncate, &[path, len as u64]), -1);
        ctx.read_u32(errno).unwrap()
    };
    assert_eq!(call(&mut ctx, cb_posix_truncate, &[p, 6]), 0);
    assert_eq!(fs::read(&host).unwrap(), b"012345");
    assert_eq!(call(&mut ctx, cb_posix_truncate, &[p, 9]), 0);
    assert_eq!(fs::read(&host).unwrap(), b"012345\0\0\0");
    assert_eq!(call(&mut ctx, cb_posix_truncate, &[p, 0]), 0);
    assert_eq!(fs::metadata(&host).unwrap().len(), 0);
    assert_eq!(fail(&mut ctx, p, -1), 22);
    let missing = f.s(&ctx, 1, &f.guest("missing"));
    assert_eq!(fail(&mut ctx, missing, 0), 2);
    let empty = f.s(&ctx, 2, "");
    assert_eq!(fail(&mut ctx, empty, 0), 2);
    assert_eq!(fail(&mut ctx, 0, 0), 14);
    let dir = f.s(&ctx, 3, &f.mount);
    assert_eq!(fail(&mut ctx, dir, 0), 21);
    assert_eq!(call(&mut ctx, cb_sce_truncate, &[dir, 0]), sce_error(EISDIR));
    assert_eq!(call(&mut ctx, cb_sce_truncate, &[p, -1i64 as u64]), sce_error(EINVAL));
    fs::write(&host, b"locked").unwrap();
    let mut perms = fs::metadata(&host).unwrap().permissions();
    perms.set_readonly(true);
    fs::set_permissions(&host, perms.clone()).unwrap();
    let readonly_honoured = fs::OpenOptions::new().write(true).open(&host).is_err();
    if readonly_honoured {
        assert_eq!(fail(&mut ctx, p, 1), 13);
        assert_eq!(call(&mut ctx, cb_sce_truncate, &[p, 1]), sce_error(EACCES));
        assert_eq!(fs::read(&host).unwrap(), b"locked");
    }
    perms.set_readonly(false);
    fs::set_permissions(&host, perms).unwrap();
}

/// The scalar POSIX calls leave errno alone on success and fail with -1
/// and the errno: EBADF for a closed descriptor (also for a zero-length
/// transfer) or the wrong access mode, EINVAL for a bad whence, EFAULT for
/// a NULL buffer with a length; the kernel forms return the kernel code
/// (reference GuestPosixScalarIo cases).
#[test]
fn posix_scalar_io_errors() {
    let f = Fixture::new(0x7C5E_0000_0000, "scalario");
    let mut ctx = f.ctx();
    let errno = crate::kernel::bind_test_errno(&ctx);
    fs::write(f.root.join("scalar.txt"), b"posix").unwrap();
    let p = f.s(&ctx, 0, &f.guest("scalar.txt"));
    let fd = call(&mut ctx, cb_posix_open, &[p, 2, 0]) as u64;
    let buf = f.base;
    let fail = |ctx: &mut CpuContext, cb: HleCallback, args: &[u64]| -> u32 {
        assert_eq!(call(ctx, cb, args), -1);
        ctx.read_u32(errno).unwrap()
    };
    ctx.write_u32(errno, 13);
    assert_eq!(call(&mut ctx, cb_posix_lseek, &[fd, 0, SEEK_SET as u64]), 0);
    assert_eq!(call(&mut ctx, cb_posix_read, &[fd, buf, 1]), 1);
    assert_eq!(ctx.read_bytes(buf, 1).unwrap(), b"p");
    let sparse = (1u64 << 32) + 17;
    assert_eq!(call(&mut ctx, cb_posix_lseek, &[fd, sparse, SEEK_SET as u64]), sparse as i64);
    assert_eq!(call(&mut ctx, cb_posix_lseek, &[fd, 0, SEEK_SET as u64]), 0);
    ctx.write_bytes(buf, b"PS");
    assert_eq!(call(&mut ctx, cb_posix_write, &[fd, buf, 2]), 2);
    assert_eq!(ctx.read_u32(errno), Some(13), "errno untouched on success");
    let closed = 999u64;
    assert_eq!(fail(&mut ctx, cb_posix_read, &[closed, buf, 1]), 9);
    assert_eq!(fail(&mut ctx, cb_posix_write, &[closed, buf, 1]), 9);
    assert_eq!(fail(&mut ctx, cb_posix_lseek, &[closed, 0, 0]), 9);
    assert_eq!(fail(&mut ctx, cb_posix_lseek, &[fd, 0, -1i64 as u64]), 22);
    assert_eq!(fail(&mut ctx, cb_posix_lseek, &[fd, 0, 3]), 22);
    assert_eq!(fail(&mut ctx, cb_posix_read, &[fd, 0, 1]), 14);
    assert_eq!(fail(&mut ctx, cb_posix_write, &[fd, 0, 1]), 14);
    assert_eq!(call(&mut ctx, cb_posix_read, &[fd, 0, 0]), 0);
    assert_eq!(call(&mut ctx, cb_posix_write, &[fd, 0, 0]), 0);
    assert_eq!(fail(&mut ctx, cb_posix_read, &[closed, 0, 0]), 9);
    assert_eq!(fail(&mut ctx, cb_posix_write, &[closed, 0, 0]), 9);
    assert_eq!(call(&mut ctx, cb_sce_read, &[closed, buf, 1]), sce_error(EBADF));
    assert_eq!(call(&mut ctx, cb_sce_write, &[closed, buf, 1]), sce_error(EBADF));
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[closed, 0, 0]), sce_error(EBADF));
    assert_eq!(call(&mut ctx, cb_sce_close, &[fd]), 0);
    let ro = call(&mut ctx, cb_posix_open, &[p, 0, 0]) as u64;
    assert_eq!(fail(&mut ctx, cb_posix_write, &[ro, buf, 1]), 9, "write on a read-only descriptor");
    assert_eq!(call(&mut ctx, cb_sce_close, &[ro]), 0);
    let wo = call(&mut ctx, cb_posix_open, &[p, 1, 0]) as u64;
    assert_eq!(fail(&mut ctx, cb_posix_read, &[wo, buf, 1]), 9, "read on a write-only descriptor");
    assert_eq!(call(&mut ctx, cb_sce_close, &[wo]), 0);
}

/// ioctl / _ioctl: FIONREAD counts the queued bytes of a socket and the
/// unread bytes of a file, FIONBIO makes a socket non-blocking (a read
/// without data is then EAGAIN), other requests are ENOTTY, a closed
/// descriptor EBADF (reference GuestUdp cases).
#[test]
fn ioctl_fionread_and_fionbio() {
    let f = Fixture::new(0x7C5C_0000_0000, "ioctl");
    let mut ctx = f.ctx();
    let errno = crate::kernel::bind_test_errno(&ctx);
    let (sv, out, data) = (f.base, f.base + 0x40, f.base + 0x100);
    let (client, server) = crate::net::test_socketpair(&ctx, 2, sv);
    ctx.write_u32(out, 0xFFFF_FFFF);
    assert_eq!(call(&mut ctx, cb_posix_ioctl, &[server as u64, FIONREAD as u64, out]), 0);
    assert_eq!(ctx.read_u32(out), Some(0));
    ctx.write_bytes(data, b"datagram");
    assert_eq!(call(&mut ctx, cb_posix_write, &[client as u64, data, 8]), 8);
    assert_eq!(call(&mut ctx, cb_posix_ioctl, &[server as u64, FIONREAD as u64, out]), 0);
    assert_eq!(ctx.read_u32(out), Some(8));
    assert_eq!(call(&mut ctx, cb_posix_ioctl, &[server as u64, 123, out]), -1);
    assert_eq!(ctx.read_u32(errno), Some(25));
    assert_eq!(call(&mut ctx, cb_posix_read, &[server as u64, data, 8]), 8);
    ctx.write_u32(out, 1);
    assert_eq!(call(&mut ctx, cb_posix_ioctl, &[server as u64, FIONBIO as u64, out]), 0);
    assert_eq!(call(&mut ctx, cb_posix_read, &[server as u64, data, 8]), -1);
    assert_eq!(ctx.read_u32(errno), Some(35), "EAGAIN once non-blocking");
    assert_eq!(call(&mut ctx, cb_sce_close, &[server as u64]), 0);
    assert_eq!(call(&mut ctx, cb_posix_ioctl, &[server as u64, FIONREAD as u64, out]), -1);
    assert_eq!(ctx.read_u32(errno), Some(9));
    assert_eq!(call(&mut ctx, cb_sce_close, &[client as u64]), 0);
    // A file: the bytes after the position.
    fs::write(f.root.join("ten.bin"), b"0123456789").unwrap();
    let p = f.s(&ctx, 0, &f.guest("ten.bin"));
    let fd = open(&mut ctx, p, 0) as u64;
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, 3, SEEK_SET as u64]), 3);
    assert_eq!(call(&mut ctx, cb_posix_ioctl, &[fd, FIONREAD as u64, out]), 0);
    assert_eq!(ctx.read_u32(out), Some(7));
    assert_eq!(call(&mut ctx, cb_posix_ioctl, &[fd, FIONREAD as u64, 0]), -1);
    assert_eq!(ctx.read_u32(errno), Some(14));
    assert_eq!(call(&mut ctx, cb_sce_close, &[fd]), 0);
    let target = |name: &str| EXPORTS.iter().find(|e| e.1 == name).map(|e| e.3 as usize);
    assert_eq!(target("_ioctl"), target("ioctl"));
    for (alias, name) in [("_readv", "readv"), ("_writev", "writev"), ("_fcntl", "fcntl"), ("_fstat", "fstat")] {
        assert!(target(alias).is_some(), "{alias} is exported");
        assert_eq!(target(alias), target(name), "{alias}");
    }
}

/// fchown: owner 0 or -1 on an open descriptor; another owner is EPERM; a
/// closed descriptor EBADF.
#[test]
fn fchown_accepts_the_only_owner() {
    let f = Fixture::new(0x7C3A_0000_0000, "fchown");
    let mut ctx = f.ctx();
    let errno = crate::kernel::bind_test_errno(&ctx);
    fs::write(f.root.join("o.bin"), b"abc").unwrap();
    let p = f.s(&ctx, 0, &f.guest("o.bin"));
    let fd = open(&mut ctx, p, 0) as u64;
    assert_eq!(call(&mut ctx, cb_posix_fchown, &[fd, 0, 0]), 0);
    assert_eq!(call(&mut ctx, cb_posix_fchown, &[fd, 0xFFFF_FFFF, 0xFFFF_FFFF]), 0);
    assert_eq!(call(&mut ctx, cb_posix_fchown, &[fd, 0, 0xFFFF_FFFF]), 0);
    assert_eq!(call(&mut ctx, cb_posix_fchown, &[fd, 0xFFFF_FFFF, 0]), 0);
    assert_eq!(call(&mut ctx, cb_posix_fchown, &[fd, 1000, 0]), -1);
    assert_eq!(ctx.read_u32(errno), Some(1));
    assert_eq!(call(&mut ctx, cb_sce_close, &[fd]), 0);
    // The descriptor is checked before the owner.
    for (uid, gid) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        assert_eq!(call(&mut ctx, cb_posix_fchown, &[0x7fff_ffff, uid, gid]), -1);
        assert_eq!(ctx.read_u32(errno), Some(9));
    }
}

/// link makes a second name of the same file; its errors (reference
/// GuestKernelSys cases).
#[test]
fn link_makes_a_hard_link() {
    use std::os::unix::fs::MetadataExt;
    let f = Fixture::new(0x7C5A_0000_0000, "link");
    let mut ctx = f.ctx();
    let errno = crate::kernel::bind_test_errno(&ctx);
    fs::write(f.root.join("original.bin"), b"abc").unwrap();
    let original = f.s(&ctx, 0, &f.guest("original.bin"));
    let linked = f.s(&ctx, 1, &f.guest("linked.bin"));
    let fail = |ctx: &mut CpuContext, a: u64, b: u64| -> u32 {
        assert_eq!(call(ctx, cb_posix_link, &[a, b]), -1);
        ctx.read_u32(errno).unwrap()
    };
    assert_eq!(call(&mut ctx, cb_posix_link, &[original, linked]), 0);
    assert_eq!(fs::metadata(f.root.join("original.bin")).unwrap().nlink(), 2);
    assert_eq!(fs::read(f.root.join("linked.bin")).unwrap(), b"abc");
    fs::write(f.root.join("linked.bin"), b"").unwrap();
    assert_eq!(fs::metadata(f.root.join("original.bin")).unwrap().len(), 0, "one file");
    assert_eq!(fail(&mut ctx, original, linked), 17);
    let missing = f.s(&ctx, 2, &f.guest("missing"));
    let other = f.s(&ctx, 3, &f.guest("other"));
    assert_eq!(fail(&mut ctx, missing, other), 2);
    let deep = f.s(&ctx, 4, &f.guest("missing/linked.bin"));
    assert_eq!(fail(&mut ctx, original, deep), 2);
    let dir = f.s(&ctx, 5, &f.mount);
    let dir_link = f.s(&ctx, 6, &f.guest("directory"));
    assert_eq!(fail(&mut ctx, dir, dir_link), 1);
    assert_eq!(fail(&mut ctx, 0, linked), 14);
    let empty = f.s(&ctx, 7, "");
    assert_eq!(fail(&mut ctx, original, empty), 2);
    fs::remove_file(f.root.join("linked.bin")).unwrap();
    assert_eq!(fs::metadata(f.root.join("original.bin")).unwrap().nlink(), 1);
}

/// symlink stores a relative target as written (readlink gives it back;
/// a dangling link is allowed); its errors (reference GuestKernelSys
/// cases).
#[test]
fn symlink_stores_the_target() {
    let f = Fixture::new(0x7C5B_0000_0000, "symlink");
    let mut ctx = f.ctx();
    let errno = crate::kernel::bind_test_errno(&ctx);
    fs::write(f.root.join("original.bin"), b"abcd").unwrap();
    let target = f.s(&ctx, 0, "original.bin");
    let symbolic = f.s(&ctx, 1, &f.guest("symbolic"));
    let buf = f.base;
    assert_eq!(call(&mut ctx, cb_posix_symlink, &[target, symbolic]), 0);
    assert!(fs::symlink_metadata(f.root.join("symbolic")).unwrap().file_type().is_symlink());
    assert_eq!(call(&mut ctx, cb_posix_readlink, &[symbolic, buf, 64]), 12);
    assert_eq!(ctx.read_bytes(buf, 12).unwrap(), b"original.bin");
    assert_eq!(fs::metadata(f.root.join("symbolic")).unwrap().len(), 4);
    assert_eq!(call(&mut ctx, cb_posix_symlink, &[target, symbolic]), -1);
    assert_eq!(ctx.read_u32(errno), Some(17));
    let missing_target = f.s(&ctx, 2, "missing.bin");
    let dangling = f.s(&ctx, 3, &f.guest("dangling"));
    assert_eq!(call(&mut ctx, cb_posix_symlink, &[missing_target, dangling]), 0);
    assert_eq!(call(&mut ctx, cb_posix_readlink, &[dangling, buf, 64]), 11);
    assert!(!f.root.join("dangling").exists());
    let absolute = f.s(&ctx, 4, "/app0/eboot.bin");
    let abs_link = f.s(&ctx, 5, &f.guest("absolute"));
    assert_eq!(call(&mut ctx, cb_posix_symlink, &[absolute, abs_link]), -1);
    assert_eq!(ctx.read_u32(errno), Some(1));
    // A parent-directory component would leave the mount; `a..b` and `...`
    // are plain names.
    for (i, bad) in ["../x", "a/../b", "a/.."].into_iter().enumerate() {
        let t = f.s(&ctx, 7, bad);
        let l = f.s(&ctx, 8, &f.guest(&format!("up{i}")));
        assert_eq!(call(&mut ctx, cb_posix_symlink, &[t, l]), -1, "{bad}");
        assert_eq!(ctx.read_u32(errno), Some(1));
    }
    for (i, ok) in ["a..b", "..."].into_iter().enumerate() {
        let t = f.s(&ctx, 7, ok);
        let l = f.s(&ctx, 8, &f.guest(&format!("dots{i}")));
        assert_eq!(call(&mut ctx, cb_posix_symlink, &[t, l]), 0, "{ok}");
    }
    assert_eq!(call(&mut ctx, cb_posix_symlink, &[0, symbolic]), -1);
    assert_eq!(ctx.read_u32(errno), Some(14));
    let empty = f.s(&ctx, 6, "");
    assert_eq!(call(&mut ctx, cb_posix_symlink, &[empty, abs_link]), -1);
    assert_eq!(ctx.read_u32(errno), Some(2));
}

/// creat creates a file for writing, and truncates an existing one
/// (reference GuestFilesystem case); the descriptor is write-only.
#[test]
fn creat_creates_and_truncates() {
    let f = Fixture::new(0x7C54_0000_0000, "creat");
    let mut ctx = f.ctx();
    let path = f.s(&ctx, 0, &f.guest("creat_probe.bin"));
    let data = f.base;
    ctx.write_bytes(data, b"abc");
    let fd = call(&mut ctx, cb_posix_creat, &[path, 0o644]);
    assert!(fd >= 0);
    assert_eq!(call(&mut ctx, cb_posix_write, &[fd as u64, data, 3]), 3);
    assert_eq!(call(&mut ctx, cb_posix_read, &[fd as u64, data + 0x10, 3]), -1, "write-only");
    assert_eq!(call(&mut ctx, cb_sce_close, &[fd as u64]), 0);
    assert_eq!(fs::metadata(f.root.join("creat_probe.bin")).unwrap().len(), 3);
    let fd = call(&mut ctx, cb_posix_creat, &[path, 0o644]);
    assert!(fd >= 0);
    assert_eq!(call(&mut ctx, cb_sce_close, &[fd as u64]), 0);
    assert_eq!(fs::metadata(f.root.join("creat_probe.bin")).unwrap().len(), 0, "truncated");
    assert_eq!(call(&mut ctx, cb_posix_creat, &[0, 0o644]), -1, "NULL path");
}

/// fsync flushes a writable file without moving its position; a
/// descriptor that is not open is EBADF (sceKernelFsync and POSIX fsync).
#[test]
fn fsync_flushes_and_reports_bad_descriptors() {
    let f = Fixture::new(0x7C71_0000_0000, "fsync");
    let mut ctx = f.ctx();
    let path = f.s(&ctx, 0, &f.guest("fsync.dat"));
    let data = f.base;
    ctx.write_bytes(data, b"save");
    let fd = call(&mut ctx, cb_posix_creat, &[path, 0o644]);
    assert!(fd >= 3);
    assert_eq!(call(&mut ctx, cb_posix_write, &[fd as u64, data, 4]), 4);
    assert_eq!(call(&mut ctx, cb_sce_fsync, &[fd as u64]), 0);
    assert_eq!(call(&mut ctx, cb_posix_fsync, &[fd as u64]), 0);
    assert_eq!(lseek_core(fd as i32, 0, 1), Ok(4), "the position stays");
    assert_eq!(call(&mut ctx, cb_sce_close, &[fd as u64]), 0);
    // A descriptor nothing holds (a closed one may be reused at once by
    // tests running in parallel).
    let unused = 0x7FFF_0000u64;
    assert_eq!(fsync_core(unused as i32), Err(EBADF));
    assert_eq!(call(&mut ctx, cb_sce_fsync, &[unused]), sce_error(EBADF));
    assert_eq!(call(&mut ctx, cb_posix_fsync, &[unused]), -1);
}

/// readv / writev on a file: from the current position, buffers filled in
/// order, the position advances; argument errors as preadv (reference
/// GuestFileVector cases).
#[test]
fn readv_writev_on_files() {
    let f = Fixture::new(0x7C4E_0000_0000, "vec");
    let mut ctx = f.ctx();
    let path = f.s(&ctx, 0, &f.guest("vec.dat"));
    fs::write(f.root.join("vec.dat"), b"0123456789").unwrap();
    let fd = open(&mut ctx, path, 2) as u64;
    let (iov, a, b) = (f.base, f.base + 0x100, f.base + 0x200);
    let set_iov = |ctx: &CpuContext, list: &[(u64, u64)]| {
        for (i, &(base, len)) in list.iter().enumerate() {
            ctx.write_u64(iov + i as u64 * 16, base);
            ctx.write_u64(iov + i as u64 * 16 + 8, len);
        }
    };
    set_iov(&ctx, &[(a, 3), (b, 4)]);
    assert_eq!(call(&mut ctx, cb_posix_readv, &[fd, iov, 2]), 7);
    assert_eq!((ctx.read_bytes(a, 3).unwrap(), ctx.read_bytes(b, 4).unwrap()), (b"012".to_vec(), b"3456".to_vec()));
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, 0, SEEK_CUR as u64]), 7);
    assert_eq!(call(&mut ctx, cb_posix_readv, &[fd, iov, 2]), 3, "short read at the end");
    assert_eq!(ctx.read_bytes(a, 3).unwrap(), b"789");
    assert_eq!(call(&mut ctx, cb_sce_readv, &[fd, iov, 2]), 0, "at the end");
    assert_eq!(call(&mut ctx, cb_posix_readv, &[fd, iov, -1i64 as u64]), -1);
    assert_eq!(call(&mut ctx, cb_sce_readv, &[fd, iov, -1i64 as u64]), sce_error(EINVAL));
    assert_eq!(call(&mut ctx, cb_posix_writev, &[fd, 0, 1]), -1);
    assert_eq!(call(&mut ctx, cb_sce_writev, &[fd, 0, 1]), sce_error(EFAULT));
    assert_eq!(call(&mut ctx, cb_sce_readv, &[999, iov, 2]), sce_error(EBADF));
    assert_eq!(call(&mut ctx, cb_sce_writev, &[999, iov, 0]), sce_error(EBADF));
    assert_eq!(call(&mut ctx, cb_sce_writev, &[fd, iov, 0]), 0);
    // Gather write at the position: the buffers in iovec order.
    ctx.write_bytes(a, b"CDE");
    ctx.write_bytes(b, b"AB");
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, 2, SEEK_SET as u64]), 2);
    set_iov(&ctx, &[(b, 2), (a, 3)]);
    assert_eq!(call(&mut ctx, cb_posix_writev, &[fd, iov, 2]), 5);
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, 0, SEEK_CUR as u64]), 7);
    set_iov(&ctx, &[(a, 3), (b, 2)]);
    assert_eq!(call(&mut ctx, cb_sce_writev, &[fd, iov, 2]), 5);
    assert_eq!(fs::read(f.root.join("vec.dat")).unwrap(), b"01ABCDECDEAB");
    assert_eq!(call(&mut ctx, cb_sce_close, &[fd]), 0);
}

/// readv / writev on a kernel socket go through one recv / send: a stream
/// pair moves the gathered bytes and a datagram pair keeps one datagram
/// that the reader spreads over its buffers.
#[test]
fn readv_writev_on_kernel_sockets() {
    let f = Fixture::new(0x7C4F_0000_0000, "vsock");
    let mut ctx = f.ctx();
    let (iov, a, b, sv) = (f.base, f.base + 0x100, f.base + 0x200, f.base + 0x300);
    let set_iov = |ctx: &CpuContext, list: &[(u64, u64)]| {
        for (i, &(base, len)) in list.iter().enumerate() {
            ctx.write_u64(iov + i as u64 * 16, base);
            ctx.write_u64(iov + i as u64 * 16 + 8, len);
        }
    };
    for ty in [1, 2] {
        let (client, server) = crate::net::test_socketpair(&ctx, ty, sv);
        ctx.write_bytes(a, b"vector ");
        ctx.write_bytes(b, b"write\0");
        set_iov(&ctx, &[(a, 7), (b, 6)]);
        assert_eq!(call(&mut ctx, cb_sce_writev, &[client as u64, iov, 2]), 13, "type {ty}");
        ctx.write_bytes(a, &[0; 7]);
        ctx.write_bytes(b, &[0; 16]);
        set_iov(&ctx, &[(a, 7), (b, 16)]);
        assert_eq!(call(&mut ctx, cb_posix_readv, &[server as u64, iov, 2]), 13, "type {ty}");
        assert_eq!(ctx.read_bytes(a, 7).unwrap(), b"vector ");
        assert_eq!(ctx.read_cstr_bytes(b, 16).unwrap(), b"write");
        set_iov(&ctx, &[(a, 0)]);
        assert_eq!(call(&mut ctx, cb_sce_readv, &[server as u64, iov, 1]), 0, "empty request");
        for fd in [client, server] {
            assert_eq!(call(&mut ctx, cb_sce_close, &[fd as u64]), 0);
        }
    }
}

/// select: sockets are polled (readable once data arrives, writable with
/// space), the sets keep only ready descriptors and the result counts the
/// set bits; the random device is readable and writable but never
/// exceptional; descriptor 0 is never ready and 1, 2 are writable; files
/// are ready; a closed descriptor is EBADF, a negative count or a bad
/// timeval EINVAL; a timeout is waited out and an oversized one does not
/// overflow (reference GuestSceNet cases).
#[test]
fn select_polls_sockets_and_answers_the_rest() {
    let f = Fixture::new(0x7C50_0000_0000, "select");
    let mut ctx = f.ctx();
    let (rd, wr, ex, tv, sv, data) = (f.base, f.base + 0x80, f.base + 0x100, f.base + 0x180, f.base + 0x200, f.base + 0x300);
    let clear = |ctx: &CpuContext| {
        for s in [rd, wr, ex] {
            ctx.write_bytes(s, &[0; 128]);
        }
    };
    let set_bit = |ctx: &CpuContext, s: u64, fd: i32| {
        let w = s + (fd as u64 / 64) * 8;
        let v = ctx.read_u64(w).unwrap();
        ctx.write_u64(w, v | 1 << (fd % 64));
    };
    let bit = |ctx: &CpuContext, s: u64, fd: i32| ctx.read_u64(s + (fd as u64 / 64) * 8).unwrap() >> (fd % 64) & 1 != 0;
    let timeval = |ctx: &CpuContext, sec: i64, usec: i64| {
        ctx.write_u64(tv, sec as u64);
        ctx.write_u64(tv + 8, usec as u64);
        tv
    };
    let (client, server) = crate::net::test_socketpair(&ctx, 1, sv);
    let nfds = (client.max(server) + 1) as u64;
    // Nothing to read yet: a zero timeout returns 0 and clears the set.
    clear(&ctx);
    set_bit(&ctx, rd, server);
    let t = timeval(&ctx, 0, 0);
    assert_eq!(call(&mut ctx, cb_select, &[nfds, rd, 0, 0, t]), 0);
    assert!(!bit(&ctx, rd, server));
    // A 50 ms timeout is waited out.
    set_bit(&ctx, rd, server);
    let t = timeval(&ctx, 0, 50_000);
    let start = std::time::Instant::now();
    assert_eq!(call(&mut ctx, cb_select, &[nfds, rd, 0, 0, t]), 0);
    assert!(start.elapsed() >= std::time::Duration::from_millis(50));
    // Data on the server, space on the client.
    ctx.write_bytes(data, b"ping");
    assert_eq!(call(&mut ctx, cb_posix_write, &[client as u64, data, 4]), 4);
    clear(&ctx);
    set_bit(&ctx, rd, server);
    set_bit(&ctx, wr, client);
    set_bit(&ctx, ex, server);
    let t = timeval(&ctx, i64::MAX, 999_999);
    assert_eq!(call(&mut ctx, cb_select, &[nfds, rd, wr, ex, t]), 2, "an oversized timeout");
    assert!(bit(&ctx, rd, server) && bit(&ctx, wr, client) && !bit(&ctx, ex, server));
    // The random device: readable and writable, never exceptional; a
    // socket without data in the same call is cleared.
    let path = f.s(&ctx, 0, "/dev/urandom");
    let random = open(&mut ctx, path, 0) as i32;
    assert!(random >= 0);
    ctx.write_bytes(data, &[0; 4]);
    assert_eq!(call(&mut ctx, cb_posix_read, &[server as u64, data, 4]), 4);
    assert_eq!(ctx.read_bytes(data, 4).unwrap(), b"ping");
    clear(&ctx);
    set_bit(&ctx, rd, random);
    set_bit(&ctx, rd, server);
    set_bit(&ctx, ex, random);
    let n = (random.max(server) + 1) as u64;
    assert_eq!(call(&mut ctx, cb_select, &[n, rd, 0, ex, 0]), 1);
    assert!(bit(&ctx, rd, random) && !bit(&ctx, rd, server) && !bit(&ctx, ex, random));
    // The standard descriptors.
    clear(&ctx);
    ctx.write_u64(rd, 1);
    ctx.write_u64(wr, 6);
    let t = timeval(&ctx, 0, 0);
    assert_eq!(call(&mut ctx, cb_select, &[3, rd, wr, 0, t]), 2);
    assert_eq!((ctx.read_u64(rd), ctx.read_u64(wr)), (Some(0), Some(6)));
    ctx.write_u64(rd, 1);
    assert_eq!(call(&mut ctx, cb_select, &[1, rd, 0, 0, t]), 0);
    assert_eq!(ctx.read_u64(rd), Some(0));
    // Bits at or past nfds are not examined.
    clear(&ctx);
    ctx.write_u64(wr, 0b110);
    assert_eq!(call(&mut ctx, cb_select, &[2, 0, wr, 0, t]), 1);
    assert_eq!(ctx.read_u64(wr), Some(0b010));
    // Errors.
    clear(&ctx);
    set_bit(&ctx, rd, 900);
    assert_eq!(call(&mut ctx, cb_select, &[901, rd, 0, 0, t]), -1, "closed descriptor");
    assert_eq!(call(&mut ctx, cb_select, &[(-1i64) as u64, 0, 0, 0, t]), -1);
    let bad = timeval(&ctx, 0, 1_000_000);
    assert_eq!(call(&mut ctx, cb_select, &[0, 0, 0, 0, bad]), -1);
    let bad = timeval(&ctx, -1, 0);
    assert_eq!(call(&mut ctx, cb_select, &[0, 0, 0, 0, bad]), -1);
    let t = timeval(&ctx, 0, 0);
    assert_eq!(call(&mut ctx, cb_select, &[0, 0, 0, 0, t]), 0, "no descriptors");
    for fd in [client, server, random] {
        assert_eq!(call(&mut ctx, cb_sce_close, &[fd as u64]), 0);
    }
}

fn mode(ctx: &CpuContext, buf: u64) -> u16 {
    u16::from_le_bytes(ctx.read_bytes(buf + 8, 2).unwrap().try_into().unwrap())
}

/// stat: a trailing slash on a file is ENOTDIR, a directory with one is
/// fine, NULL pointers are EFAULT and unconvertible names are ENOENT.
/// lstat reports links themselves; readlink returns the target, truncated
/// to the buffer, without a terminator.
#[test]
fn stat_lstat_and_readlink() {
    let f = Fixture::new(0x5CD8_0000_0000, "lstat");
    let mut ctx = f.ctx();
    let (st, st2, buf) = (f.base, f.base + 0x100, f.base + 0x200);
    fs::write(f.root.join("present.txt"), b"posix").unwrap();
    let present = f.s(&ctx, 0, &f.guest("present.txt"));
    let slash = f.s(&ctx, 1, &f.guest("present.txt/"));
    let root_slash = f.s(&ctx, 2, &format!("{}/", f.mount));
    let root = f.s(&ctx, 3, &f.mount);
    let missing = f.s(&ctx, 4, &f.guest("missing.txt"));
    let empty = f.s(&ctx, 5, "");
    assert_eq!(call(&mut ctx, cb_sce_stat, &[present, st]), 0);
    assert_eq!(stat_size(&ctx, st), 5);
    assert_eq!(call(&mut ctx, cb_sce_stat, &[slash, st2]), sce_error(ENOTDIR), "a file with a trailing slash");
    assert_eq!(stat_core(&ctx, slash, st2), Err(ENOTDIR));
    assert_eq!(call(&mut ctx, cb_posix_stat, &[slash, st2]), -1);
    assert_eq!(call(&mut ctx, cb_sce_stat, &[root_slash, st2]), 0);
    assert_eq!(mode(&ctx, st2) & 0o170000, 0o040000);
    let bad = f.base + 0x300;
    ctx.write_bytes(bad, b"\xff\xfe\0");
    assert_eq!(call(&mut ctx, cb_sce_stat, &[bad, st2]), sce_error(ENOENT), "no file has such a name");
    assert_eq!(stat_core(&ctx, bad, st2), Err(ENOENT));
    assert_eq!(stat_core(&ctx, 0, st2), Err(EFAULT));
    assert_eq!(stat_core(&ctx, present, 0), Err(EFAULT));
    assert_eq!(stat_core(&ctx, empty, st2), Err(ENOENT));

    // lstat of entries that are not links is stat.
    assert_eq!(lstat_core(&ctx, present, st2), Ok(0));
    assert_eq!(stat_size(&ctx, st2), 5);
    assert_eq!(mode(&ctx, st2) & 0o170000, 0o100000);
    assert_eq!(ctx.read_u32(st2 + 4), ctx.read_u32(st + 4), "same inode as stat");
    assert_eq!(lstat_core(&ctx, root, st2), Ok(0));
    assert_eq!(mode(&ctx, st2) & 0o170000, 0o040000);
    assert_eq!(lstat_core(&ctx, missing, st2), Err(ENOENT));
    assert_eq!(lstat_core(&ctx, empty, st2), Err(ENOENT));
    assert_eq!(lstat_core(&ctx, 0, st2), Err(EFAULT));
    assert_eq!(lstat_core(&ctx, present, 0), Err(EFAULT));
    assert_eq!(call(&mut ctx, cb_posix_lstat, &[missing, st2]), -1);

    // Links: lstat sees the link, stat what it points to.
    std::os::unix::fs::symlink("present.txt", f.root.join("link")).unwrap();
    std::os::unix::fs::symlink("missing.txt", f.root.join("dangling")).unwrap();
    let link = f.s(&ctx, 6, &f.guest("link"));
    let dangling = f.s(&ctx, 7, &f.guest("dangling"));
    assert_eq!(call(&mut ctx, cb_posix_lstat, &[link, st2]), 0);
    assert_eq!(mode(&ctx, st2) & 0o170000, 0o120000);
    assert_eq!(stat_size(&ctx, st2), 11, "the length of the target");
    assert_eq!(call(&mut ctx, cb_sce_stat, &[link, st2]), 0);
    assert_eq!(mode(&ctx, st2) & 0o170000, 0o100000);
    assert_eq!(lstat_core(&ctx, dangling, st2), Ok(0));
    assert_eq!(mode(&ctx, st2) & 0o170000, 0o120000, "a dangling link");
    assert_eq!(stat_core(&ctx, dangling, st2), Err(ENOENT));
    let through = f.s(&ctx, 8, &f.guest("present.txt/child"));
    assert_eq!(lstat_core(&ctx, through, st2), Err(ENOTDIR));

    // readlink
    std::os::unix::fs::symlink("sub/target.txt", f.root.join("rl")).unwrap();
    let rl = f.s(&ctx, 9, &f.guest("rl"));
    ctx.write_bytes(buf, &[b'#'; 64]);
    assert_eq!(readlink_core(&ctx, rl, buf, 64), Ok(14));
    assert_eq!(&ctx.read_bytes(buf, 15).unwrap()[..], b"sub/target.txt#", "no terminator");
    ctx.write_bytes(buf, &[b'#'; 64]);
    assert_eq!(call(&mut ctx, cb_posix_readlink, &[rl, buf, 3]), 3);
    assert_eq!(&ctx.read_bytes(buf, 4).unwrap()[..], b"sub#", "truncated");
    assert_eq!(readlink_core(&ctx, rl, 0, 0), Ok(0));
    assert_eq!(readlink_core(&ctx, rl, 0, 64), Err(EFAULT));
    let upper = f.s(&ctx, 10, &f.guest("RL"));
    assert_eq!(readlink_core(&ctx, upper, buf, 64), Ok(14), "names match without regard to case");
    assert_eq!(readlink_core(&ctx, present, buf, 64), Err(EINVAL), "a file is no link");
    assert_eq!(readlink_core(&ctx, root, buf, 64), Err(EINVAL), "a directory is no link");
    assert_eq!(readlink_core(&ctx, missing, buf, 64), Err(ENOENT));
    assert_eq!(readlink_core(&ctx, empty, buf, 64), Err(ENOENT));
    assert_eq!(readlink_core(&ctx, 0, buf, 64), Err(EFAULT));
    assert_eq!(readlink_core(&ctx, 0, buf, u64::MAX), Err(EINVAL), "the size before the path");
    // ioctl requests FreeBSD cannot decode are ENOTTY before the descriptor.
    for request in [0u64, 0x0004_6601, 0x4000_6601, 0x2008_6601] {
        assert_eq!(ioctl_core(&ctx, -1, request, 0), Err(ENOTTY), "{request:#x}");
    }
    assert_eq!(ioctl_core(&ctx, -1, 0x2004_6601, 0), Err(EBADF));
    assert_eq!(call(&mut ctx, cb_posix_readlink, &[present, buf, 64]), -1);

    // unlink removes links, never their targets.
    assert_eq!(call(&mut ctx, cb_sce_unlink, &[link]), 0);
    assert_eq!(call(&mut ctx, cb_sce_unlink, &[dangling]), 0);
    assert!(f.root.join("present.txt").is_file());
    assert!(fs::symlink_metadata(f.root.join("link")).is_err() && fs::symlink_metadata(f.root.join("dangling")).is_err());
}

/// NULL paths are EFAULT and empty ones ENOENT; unlink of a directory is
/// EPERM and ignores the file's own permission bits; open of a directory
/// with O_CREAT|O_EXCL is EEXIST, with write access or O_CREAT EISDIR.
#[test]
fn path_argument_errors() {
    use std::os::unix::fs::PermissionsExt;
    let f = Fixture::new(0x5CD9_0000_0000, "patherr");
    let mut ctx = f.ctx();
    fs::create_dir(f.root.join("full")).unwrap();
    fs::write(f.root.join("full/entry"), b"x").unwrap();
    fs::write(f.root.join("file"), b"x").unwrap();
    let root = f.s(&ctx, 0, &f.mount);
    let full = f.s(&ctx, 1, &f.guest("full"));
    let file = f.s(&ctx, 2, &f.guest("file"));
    let empty = f.s(&ctx, 3, "");
    assert_eq!(call(&mut ctx, cb_sce_unlink, &[full]), sce_error(EPERM));
    assert_eq!(call(&mut ctx, cb_posix_unlink, &[full]), -1);
    assert_eq!(unlink_core(&ctx, full), Err(EPERM));
    assert!(f.root.join("full").is_dir());
    let cores: [fn(&CpuContext, u64) -> FsResult<i64>; 3] = [unlink_core, mkdir_core, rmdir_core];
    for core in cores {
        assert_eq!(core(&ctx, 0), Err(EFAULT));
        assert_eq!(core(&ctx, empty), Err(ENOENT));
    }
    assert_eq!(call(&mut ctx, cb_sce_rmdir, &[full]), sce_error(ENOTEMPTY));
    assert_eq!(rmdir_core(&ctx, file), Err(ENOTDIR));
    fs::set_permissions(f.root.join("file"), fs::Permissions::from_mode(0o400)).unwrap();
    assert_eq!(call(&mut ctx, cb_posix_unlink, &[file]), 0, "a read-only file is removable");
    assert!(!f.root.join("file").exists());
    for (flags, want) in [(0xa02, EEXIST), (0xa00, EEXIST), (0x200, EISDIR), (0x2, EISDIR), (0x1, EISDIR)] {
        assert_eq!(open_core(&ctx, root, flags), Err(want), "open of a directory with flags {flags:#x}");
    }
    let fd = open(&mut ctx, root, 0);
    assert!(fd >= 3);
    assert_eq!(call(&mut ctx, cb_sce_close, &[fd as u64]), 0);
    assert_eq!(open_core(&ctx, 0, 0), Err(EFAULT));
    assert_eq!(open_core(&ctx, empty, 0), Err(ENOENT));
    assert_eq!(check_reachability_core(&ctx, 0), Err(EINVAL));
}

/// rename, shared by libkernel and libc, keeps FreeBSD's errors.
#[test]
fn rename_errors() {
    let f = Fixture::new(0x5CDA_0000_0000, "rename");
    let mut ctx = f.ctx();
    let r = &f.root;
    fs::write(r.join("source.txt"), b"source").unwrap();
    fs::write(r.join("target.txt"), b"target").unwrap();
    fs::create_dir_all(r.join("source_directory/child")).unwrap();
    fs::create_dir(r.join("empty_directory")).unwrap();
    fs::create_dir(r.join("full_directory")).unwrap();
    fs::write(r.join("full_directory/entry"), b"entry").unwrap();
    let p = |ctx: &CpuContext, n: u64, rel: &str| f.s(ctx, n, &f.guest(rel));
    let source = p(&ctx, 0, "source.txt");
    let target = p(&ctx, 1, "target.txt");
    let source_dir = p(&ctx, 2, "source_directory");
    let empty_dir = p(&ctx, 3, "empty_directory");
    let full_dir = p(&ctx, 4, "full_directory");
    let missing = p(&ctx, 5, "missing");
    let absent_parent = p(&ctx, 6, "absent/target");
    let file_parent = p(&ctx, 7, "source.txt/target");
    let into_child = p(&ctx, 8, "source_directory/child/moved");
    let onto_child = p(&ctx, 9, "source_directory/child");
    assert_eq!(rename_core(&ctx, missing, target), Err(ENOENT));
    assert_eq!(rename_core(&ctx, source, absent_parent), Err(ENOENT));
    assert_eq!(rename_core(&ctx, source, file_parent), Err(ENOTDIR));
    assert_eq!(rename_core(&ctx, source, full_dir), Err(EISDIR));
    assert_eq!(rename_core(&ctx, source_dir, target), Err(ENOTDIR));
    assert_eq!(rename_core(&ctx, source_dir, full_dir), Err(ENOTEMPTY));
    assert_eq!(rename_core(&ctx, source_dir, into_child), Err(EINVAL));
    assert_eq!(rename_core(&ctx, source_dir, onto_child), Err(EINVAL));
    assert!(r.join("source.txt").is_file() && r.join("full_directory/entry").is_file());
    // A path ending in a dot component (reference GuestRename cases
    // measured on a PS5): EINVAL below a directory, ENOTDIR below a file,
    // the lookup error otherwise; the source is resolved first, and a
    // destination error needs a source that exists.
    let dot_dir = p(&ctx, 12, "full_directory/.");
    let dotdot_dir = p(&ctx, 13, "full_directory/..//");
    let dot_file = p(&ctx, 14, "source.txt/.");
    let dot_missing = p(&ctx, 15, "missing/.");
    let empty = f.s(&ctx, 16, "");
    assert_eq!(rename_core(&ctx, dot_dir, target), Err(EINVAL));
    assert_eq!(rename_core(&ctx, dotdot_dir, target), Err(EINVAL));
    assert_eq!(rename_core(&ctx, dot_file, target), Err(ENOTDIR));
    assert_eq!(rename_core(&ctx, dot_missing, target), Err(ENOENT));
    assert_eq!(rename_core(&ctx, source, dot_dir), Err(EINVAL));
    assert_eq!(rename_core(&ctx, source, 0), Err(EFAULT));
    assert_eq!(rename_core(&ctx, source, empty), Err(ENOENT));
    assert_eq!(rename_core(&ctx, missing, 0), Err(ENOENT), "the source first");
    assert_eq!(rename_core(&ctx, missing, dot_dir), Err(ENOENT));
    assert_eq!(rename_core(&ctx, 0, missing), Err(EFAULT));
    assert!(r.join("source.txt").is_file() && r.join("full_directory/entry").is_file());
    // A "." or ".." below a file is ENOTDIR everywhere; PATH_MAX and
    // NAME_MAX are ENAMETOOLONG (a long name below a missing directory is
    // just missing).
    let stat_buf = f.base + 0x3_0000;
    assert_eq!(stat_core(&ctx, dot_file, stat_buf), Err(ENOTDIR));
    assert_eq!(stat_core(&ctx, p(&ctx, 17, "source.txt/../target.txt"), stat_buf), Err(ENOTDIR));
    let long_name = "n".repeat(256);
    assert_eq!(stat_core(&ctx, p(&ctx, 18, &long_name), stat_buf), Err(ENAMETOOLONG));
    assert_eq!(stat_core(&ctx, p(&ctx, 19, &format!("missing/{long_name}")), stat_buf), Err(ENOENT));
    assert_eq!(stat_core(&ctx, p(&ctx, 20, &format!("{}/x", "d".repeat(1030))), stat_buf), Err(ENAMETOOLONG));
    assert_eq!(call(&mut ctx, cb_posix_rename, &[source, target]), 0);
    assert_eq!(fs::read(r.join("target.txt")).unwrap(), b"source");
    assert!(!r.join("source.txt").exists());
    assert_eq!(rename_core(&ctx, target, target), Ok(0));
    assert_eq!(posix_rename(&ctx, source_dir, empty_dir), 0, "a directory replaces an empty one");
    assert!(!r.join("source_directory").exists() && r.join("empty_directory/child").is_dir());
    // A file replaces a dangling link (the link, not its target).
    std::os::unix::fs::symlink(r.join("nonexistent"), r.join("dangling")).unwrap();
    fs::write(r.join("replacement"), b"replacement").unwrap();
    let replacement = p(&ctx, 10, "replacement");
    let dangling = p(&ctx, 11, "dangling");
    assert_eq!(rename_core(&ctx, replacement, dangling), Ok(0));
    assert!(fs::symlink_metadata(r.join("dangling")).unwrap().is_file());
    assert!(!r.join("nonexistent").exists());
    let empty = f.s(&ctx, 12, "");
    assert_eq!(rename_core(&ctx, 0, target), Err(EFAULT));
    assert_eq!(rename_core(&ctx, target, 0), Err(EFAULT));
    assert_eq!(rename_core(&ctx, empty, target), Err(ENOENT));
    assert_eq!(rename_core(&ctx, target, empty), Err(ENOENT));
    assert_eq!(call(&mut ctx, cb_posix_rename, &[0, target]), -1);
    assert_eq!(call(&mut ctx, cb_sce_rename, &[target, missing]), 0);
}

/// flock: FreeBSD decoding (LOCK_UN over LOCK_EX over LOCK_SH, EBADF for
/// none), BSD per-open-file locks with conversion, EWOULDBLOCK for a
/// contended LOCK_NB request, locks shared with duplicates and released
/// with the last descriptor, blocking requests that wait for the holder.
#[test]
fn flock_bsd_semantics() {
    let f = Fixture::new(0x5CDB_0000_0000, "flock");
    let mut ctx = f.ctx();
    fs::write(f.root.join("present.txt"), b"posix").unwrap();
    let p = f.s(&ctx, 0, &f.guest("present.txt"));
    let fl = |fd: i64, op: i32| flock_core(fd as i32, op);
    let close = |ctx: &mut CpuContext, fd: i64| assert_eq!(call(ctx, cb_sce_close, &[fd as u64]), 0);
    let locked = open(&mut ctx, p, 0);
    let other = open(&mut ctx, p, 0);
    assert!(locked >= 3 && other >= 3);
    let steps = [
        (locked, 8),
        (locked, 2 | 4),
        (locked, 2 | 4),
        (locked, 1 | 4),
        (other, 1 | 4),
        (other, 8),
        (locked, 1 | 4),
        (locked, 2 | 4),
        (locked, 8),
        (locked, 8),
        (other, 2 | 4),
        (other, 8),
        (locked, 2 | 4),
    ];
    for (i, (fd, op)) in steps.into_iter().enumerate() {
        assert_eq!(fl(fd, op), Ok(0), "step {i}: flock({fd}, {op:#x})");
    }
    close(&mut ctx, locked);
    let recycled = open(&mut ctx, p, 0);
    assert_eq!(fl(recycled, 8), Ok(0));
    assert_eq!(fl(recycled, 2 | 4), Ok(0), "closing the holder released its lock");
    close(&mut ctx, recycled);
    let kernel = open(&mut ctx, p, 0);
    assert_eq!(fl(kernel, 8), Ok(0));
    assert_eq!(fl(other, 0), Err(EBADF));
    assert_eq!(fl(other, 4 | 0x10), Err(EBADF));
    assert_eq!(call(&mut ctx, cb_posix_flock, &[other as u64, 0]), -1);
    assert_eq!(fl(kernel, 2 | 4 | 0x10), Ok(0), "unknown bits are ignored");
    assert_eq!(fl(kernel, 8 | 2), Ok(0), "LOCK_UN wins");
    assert_eq!(fl(other, 2 | 4), Ok(0));
    assert_eq!(fl(other, 8), Ok(0));
    close(&mut ctx, kernel);
    close(&mut ctx, other);

    let holder = open(&mut ctx, p, 0);
    let contender = open(&mut ctx, p, 0);
    assert_eq!(fl(holder, 2 | 4), Ok(0));
    assert_eq!(fl(contender, 2 | 4), Err(EWOULDBLOCK));
    assert_eq!(fl(contender, 1 | 4), Err(EWOULDBLOCK));
    assert_eq!(fl(holder, 1 | 4), Ok(0));
    assert_eq!(fl(contender, 1 | 4), Ok(0), "shared locks coexist");
    assert_eq!(fl(contender, 2 | 4), Err(EWOULDBLOCK));
    assert_eq!(fl(holder, 8), Ok(0));
    assert_eq!(fl(contender, 2 | 4), Ok(0));
    assert_eq!(fl(contender, 8), Ok(0));
    assert_eq!(fl(holder, 1 | 2 | 4 | 0x10), Ok(0), "a combined operation takes an exclusive lock");
    assert_eq!(fl(contender, 1 | 4), Err(EWOULDBLOCK));
    assert_eq!(fl(holder, 8), Ok(0));

    // A duplicate shares the lock; it goes with the last descriptor.
    assert_eq!(fl(holder, 2 | 4), Ok(0));
    let dup = dup_core(holder as i32).unwrap();
    close(&mut ctx, holder);
    assert_eq!(fl(contender, 2 | 4), Err(EWOULDBLOCK), "the duplicate still holds the lock");
    assert_eq!(fl(dup, 8), Ok(0), "the duplicate releases the shared lock");
    assert_eq!(fl(contender, 2 | 4), Ok(0));
    assert_eq!(fl(contender, 8), Ok(0));

    // A blocking request waits until the holder lets go.
    assert_eq!(fl(dup, 2), Ok(0));
    let release = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(50));
        close_core(dup as i32)
    });
    assert_eq!(fl(contender, 2), Ok(0));
    assert_eq!(release.join().unwrap(), Ok(0));
    assert_eq!(fl(contender, 8), Ok(0));
    close(&mut ctx, contender);
    assert_eq!(fl(contender, 2), Err(EBADF));
}

/// Kernel sockets through the file calls: read/write reach the socket as
/// recv/send (EWOULDBLOCK when empty, 0 at EOF, EPIPE after the peer went
/// away), file-only calls refuse them, closed descriptors are EBADF even
/// for zero-length reads, and dup2 onto a socket closes it.
#[test]
fn kernel_socket_descriptors() {
    let f = Fixture::new(0x5CDC_0000_0000, "sockfd");
    let mut ctx = f.ctx();
    let mut reg = crate::registry::HleRegistry::new();
    crate::net::register(&mut reg);
    let socketpair = reg.lookup("MZb0GKT3mo8").expect("socketpair export").callback;
    let sv = f.base;
    assert_eq!(call(&mut ctx, socketpair, &[1, 1, 0, sv]), 0);
    let (a, b) = (ctx.read_u32(sv).unwrap() as u64, ctx.read_u32(sv + 4).unwrap() as u64);
    let (out, inb, fsbuf) = (f.base + 0x100, f.base + 0x200, f.base + 0x1000);
    ctx.write_bytes(out, b"written to a socket\0");
    assert_eq!(call(&mut ctx, cb_posix_write, &[a, out, 20]), 20);
    assert_eq!(call(&mut ctx, cb_posix_read, &[b, inb, 32]), 20);
    assert_eq!(&ctx.read_bytes(inb, 20).unwrap()[..], b"written to a socket\0");
    assert_eq!(call(&mut ctx, cb_sce_write, &[b, out, 7]), 7);
    assert_eq!(call(&mut ctx, cb_sce_read, &[a, inb, 32]), 7);
    assert_eq!(&ctx.read_bytes(inb, 7).unwrap()[..], b"written");
    assert_eq!(fcntl_core(a as i32, 4, 4), Ok(0), "O_NONBLOCK");
    assert_eq!(call(&mut ctx, cb_sce_read, &[a, inb, 32]), sce_error(EWOULDBLOCK));
    assert_eq!(read_core(&ctx, a as i32, inb, 32), Err(EWOULDBLOCK));
    assert_eq!(call(&mut ctx, cb_sce_read, &[a, inb, 0]), 0, "zero-length read");
    assert_eq!(fdatasync_core(a as i32), Err(EINVAL));
    assert_eq!(call(&mut ctx, cb_sce_fdatasync, &[a]), sce_error(EINVAL));
    assert_eq!(call(&mut ctx, cb_posix_fdatasync, &[a]), -1);
    assert_eq!(fsync_core(a as i32), Err(EINVAL));
    assert_eq!(fchmod_core(a as i32, 0o600), Err(EINVAL));
    assert_eq!(futimes_core(&ctx, a as i32, 0), Err(EINVAL));
    assert_eq!(ftruncate_core(a as i32, 0), Err(EINVAL));
    assert_eq!(pread_core(&ctx, a as i32, inb, 1, 0), Err(ESPIPE));
    assert_eq!(pwrite_core(&ctx, a as i32, out, 1, 0), Err(ESPIPE));
    assert_eq!(fstatfs_core(&ctx, a as i32, fsbuf), Err(EINVAL));
    assert_eq!(flock_core(a as i32, LOCK_EX), Err(EOPNOTSUPP));
    assert_eq!(call(&mut ctx, cb_sce_close, &[b]), 0);
    assert_eq!(call(&mut ctx, cb_sce_read, &[a, inb, 32]), 0, "EOF once the peer closed");
    let mut written = 0;
    for _ in 0..100 {
        written = call(&mut ctx, cb_sce_write, &[a, out, 20]);
        if written < 0 {
            break;
        }
    }
    assert_eq!(written, sce_error(32), "EPIPE without a peer");
    assert_eq!(call(&mut ctx, cb_sce_close, &[a]), 0);
    // A closed number is free again: a test running alongside may have
    // opened something on it meanwhile.
    let closed = |r: FsResult<i64>, fd: i32| r == Err(EBADF) || (descriptor_open(fd) && !crate::net::is_kernel_socket(fd));
    for fd in [a as i32, b as i32] {
        assert!(closed(read_core(&ctx, fd, inb, 32), fd));
        assert!(closed(read_core(&ctx, fd, inb, 0), fd), "zero-length read of a closed descriptor");
        assert!(closed(write_core(&ctx, fd, out, 20), fd));
        assert!(closed(fdatasync_core(fd), fd));
        assert!(closed(fchmod_core(fd, 0o600), fd));
        assert!(closed(fstatfs_core(&ctx, fd, fsbuf), fd));
    }
    assert_eq!(call(&mut ctx, cb_sce_fdatasync, &[0x7fff_ffff]), sce_error(EBADF));
    assert_eq!(call(&mut ctx, cb_posix_read, &[0x7fff_ffff, inb, 0]), -1);

    // dup2 onto a socket closes the socket.
    assert_eq!(call(&mut ctx, socketpair, &[1, 1, 0, sv]), 0);
    let (c, d) = (ctx.read_u32(sv).unwrap() as i32, ctx.read_u32(sv + 4).unwrap() as u64);
    let path = f.s(&ctx, 0, &f.guest("dup2.dat"));
    let file = open(&mut ctx, path, 0x602) as i32;
    assert!(file >= 3);
    assert_eq!(dup2_core(file, c), Ok(c as i64));
    assert!(!crate::net::is_kernel_socket(c));
    assert_eq!(call(&mut ctx, cb_sce_write, &[c as u64, out, 7]), 7);
    assert_eq!(fs::read(f.root.join("dup2.dat")).unwrap(), b"written");
    for fd in [c as u64, d, file as u64] {
        assert_eq!(call(&mut ctx, cb_sce_close, &[fd]), 0);
    }
}

/// AIO requests that fail (a negative offset) end ABORTED with EINVAL;
/// a batch id ends ABORTED when any of its requests failed, per-request
/// ids carry their own state, and a missing result pointer fails the
/// submission before any request runs.
#[test]
fn aio_failed_requests() {
    let f = Fixture::new(0x5CDD_0000_0000, "aiofail");
    let mut ctx = f.ctx();
    let seed: Vec<u8> = (0..64u8).map(|i| b'a' + i % 26).collect();
    fs::write(f.root.join("aio.dat"), &seed).unwrap();
    let path = f.s(&ctx, 0, &f.guest("aio.dat"));
    let fd = open(&mut ctx, path, O_RDWR);
    assert!(fd >= 3);
    let (reqs, results, ids, states, buf) = (f.base, f.base + 0x400, f.base + 0x800, f.base + 0x900, f.base + 0x1000);
    let res = |i: u64| results + i * 16;
    let req = |ctx: &CpuContext, i: u64, offset: i64, nbyte: u64, data: u64, result: u64| {
        let mut r = Vec::new();
        r.extend_from_slice(&offset.to_le_bytes());
        r.extend_from_slice(&nbyte.to_le_bytes());
        r.extend_from_slice(&data.to_le_bytes());
        r.extend_from_slice(&result.to_le_bytes());
        r.extend_from_slice(&(fd as i32).to_le_bytes());
        r.extend_from_slice(&[0u8; 4]);
        ctx.write_bytes(reqs + i * AIO_RW_REQUEST_SIZE, &r);
    };
    let state = |ctx: &CpuContext, r: u64| (ctx.read_u32(r + 8).unwrap() as i32, ctx.read_u64(r).unwrap() as i64);
    let einval = sce_error(EINVAL);
    let wait_one = |ctx: &mut CpuContext| {
        let id = ctx.read_u32(ids).unwrap() as u64;
        assert_eq!(call(ctx, cb_aio_wait_request, &[id, states, 0]), 0);
        ctx.read_u32(states).unwrap() as i32
    };

    req(&ctx, 0, -1, 4, buf, res(0));
    assert_eq!(call(&mut ctx, cb_aio_submit_read, &[reqs, 1, 0, ids]), 0);
    assert_eq!(wait_one(&mut ctx), AIO_STATE_ABORTED);
    assert_eq!(state(&ctx, res(0)), (AIO_STATE_ABORTED, einval));

    req(&ctx, 0, 4, 4, buf, res(1));
    req(&ctx, 1, -1, 4, buf + 0x100, res(2));
    assert_eq!(call(&mut ctx, cb_aio_submit_read, &[reqs, 2, 0, ids]), 0);
    assert_eq!(wait_one(&mut ctx), AIO_STATE_ABORTED, "a batch with a failed request is aborted");
    assert_eq!(state(&ctx, res(1)), (AIO_STATE_COMPLETED, 4));
    assert_eq!(state(&ctx, res(2)), (AIO_STATE_ABORTED, einval));
    assert_eq!(&ctx.read_bytes(buf, 4).unwrap()[..], &seed[4..8]);

    assert_eq!(call(&mut ctx, cb_aio_submit_read_multiple, &[reqs, 2, 0, ids]), 0);
    let (id0, id1) = (ctx.read_u32(ids).unwrap(), ctx.read_u32(ids + 4).unwrap());
    assert!(id0 > 0 && id1 > 0 && id0 != id1);
    assert_eq!(call(&mut ctx, cb_aio_poll_requests, &[ids, 2, states]), 0);
    assert_eq!((ctx.read_u32(states), ctx.read_u32(states + 4)), (Some(3), Some(4)), "each id keeps its own state");

    ctx.write_bytes(buf + 0x200, b"GOOD");
    req(&ctx, 0, 8, 4, buf + 0x200, res(3));
    req(&ctx, 1, -1, 4, buf + 0x200, res(4));
    assert_eq!(call(&mut ctx, cb_aio_submit_write, &[reqs, 2, 0, ids]), 0);
    assert_eq!(wait_one(&mut ctx), AIO_STATE_ABORTED);
    assert_eq!(state(&ctx, res(3)), (AIO_STATE_COMPLETED, 4));
    assert_eq!(state(&ctx, res(4)), (AIO_STATE_ABORTED, einval));
    let contents = fs::read(f.root.join("aio.dat")).unwrap();
    assert_eq!(contents.len(), seed.len());
    assert_eq!(&contents[8..12], b"GOOD");
    assert_eq!(&contents[12..16], &seed[12..16]);

    assert_eq!(call(&mut ctx, cb_aio_submit_write_multiple, &[reqs, 2, 0, ids]), 0);
    assert_eq!(call(&mut ctx, cb_aio_wait_requests, &[ids, 2, states, AIO_WAIT_AND as u64, 0]), 0);
    assert_eq!((ctx.read_u32(states), ctx.read_u32(states + 4)), (Some(3), Some(4)));

    ctx.write_u64(res(5), 0xdead);
    ctx.write_u32(res(5) + 8, 0x77);
    req(&ctx, 0, 0, 4, buf, res(5));
    req(&ctx, 1, 0, 4, buf, 0);
    assert_eq!(call(&mut ctx, cb_aio_submit_read, &[reqs, 2, 0, ids]), sce_error(EFAULT));
    assert_eq!(state(&ctx, res(5)), (0x77, 0xdead), "nothing ran");
    assert_eq!(call(&mut ctx, cb_sce_close, &[fd as u64]), 0);
}

/// The POSIX getdents/getdirentries exports: the same stream as the
/// sceKernel forms, errno only on failure (EINVAL for a short buffer,
/// EFAULT for NULL, EBADF once closed).
#[test]
fn getdents_posix_exports() {
    let f = Fixture::new(0x5CDE_0000_0000, "dents");
    let mut ctx = f.ctx();
    for n in ["data.bin", "b", "a-much-longer-entry-name-0123456789"] {
        fs::write(f.root.join(n), b"").unwrap();
    }
    fs::create_dir(f.root.join("sub")).unwrap();
    let d = f.s(&ctx, 0, &f.mount);
    let fd = open(&mut ctx, d, O_DIRECTORY) as u64;
    assert!(fd >= 3);
    let (buf, other, basep) = (f.base, f.base + 0x2000, f.base + 0x4000);
    let whole = call(&mut ctx, cb_sce_getdents, &[fd, buf, 4096]);
    assert!(whole > 0);
    let names: Vec<String> = records(&ctx.read_bytes(buf, whole as usize).unwrap()).into_iter().map(|(n, _)| n).collect();
    assert_eq!(names.len(), 6);
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, 0, SEEK_SET as u64]), 0);
    assert_eq!(call(&mut ctx, cb_posix_getdents, &[fd, buf, 8]), -1);
    assert_eq!(getdirentries_core(&ctx, fd as i32, buf, 8, 0), Err(EINVAL));
    assert_eq!(call(&mut ctx, cb_posix_getdents, &[fd, 0, 64]), -1);
    assert_eq!(getdirentries_core(&ctx, fd as i32, 0, 64, 0), Err(EFAULT));
    assert_eq!(getdirentries_core(&ctx, fd as i32, buf, 0, 0), Err(EINVAL));
    ctx.write_u64(basep, u64::MAX);
    assert_eq!(call(&mut ctx, cb_posix_getdirentries, &[fd, other, 4096, basep]), whole);
    assert_eq!(ctx.read_u64(basep), Some(0));
    assert_eq!(ctx.read_bytes(other, whole as usize), ctx.read_bytes(buf, whole as usize));
    assert_eq!(call(&mut ctx, cb_posix_getdents, &[fd, other, 4096]), 0, "end of the stream");
    assert_eq!(call(&mut ctx, cb_sce_close, &[fd]), 0);
    assert_eq!(call(&mut ctx, cb_posix_getdents, &[fd, buf, 4096]), -1);
    assert_eq!(getdirentries_core(&ctx, fd as i32, buf, 4096, 0), Err(EBADF));
    assert_eq!(call(&mut ctx, cb_posix_getdirentries, &[fd, buf, 4096, 0]), -1);
}

/// _fstatfs: the host volume of files and directories in the FreeBSD
/// statfs layout; NULL is EFAULT, closed descriptors EBADF, and nothing is
/// written on failure.
#[test]
fn fstatfs_reports_the_volume() {
    let f = Fixture::new(0x5CDF_0000_0000, "statfs");
    let mut ctx = f.ctx();
    fs::write(f.root.join("file.txt"), b"fstatfs").unwrap();
    let p = f.s(&ctx, 0, &f.guest("file.txt"));
    let file = open(&mut ctx, p, 0) as u64;
    let d = f.s(&ctx, 1, &f.mount);
    let dir = open(&mut ctx, d, O_DIRECTORY) as u64;
    assert!(file >= 3 && dir >= 3);
    let buf = f.base;
    let vfs = host_statvfs_path(&f.root).unwrap();
    for fd in [file, dir] {
        ctx.write_bytes(buf, &[0x5a; KERNEL_STATFS_SIZE]);
        assert_eq!(call(&mut ctx, cb_posix_fstatfs, &[fd, buf]), 0);
        let rd = |o: u64| ctx.read_u64(buf + o).unwrap();
        assert_eq!(ctx.read_u32(buf), Some(STATFS_VERSION));
        let (flags, bsize, iosize, blocks, bfree, bavail) = (rd(8), rd(16), rd(24), rd(32), rd(40), rd(48) as i64);
        assert!(bsize > 0 && iosize > 0);
        assert!(blocks > 0 && bfree <= blocks);
        assert!(bavail >= 0 && bavail as u64 <= bfree);
        assert_eq!(blocks, vfs.f_blocks as u64);
        assert_eq!(flags & MNT_RDONLY, 0, "a writable mount");
        assert!(ctx.read_u32(buf + 184).unwrap() >= 255, "f_namemax");
        assert!(ctx.read_bytes(buf + 280, 192).unwrap().iter().all(|&b| b == 0), "no mount names");
    }
    assert_eq!(fstatfs_core(&ctx, file as i32, 0), Err(EFAULT));
    assert_eq!(call(&mut ctx, cb_sce_close, &[file]), 0);
    ctx.write_bytes(buf, &[0x5a; KERNEL_STATFS_SIZE]);
    assert_eq!(fstatfs_core(&ctx, file as i32, buf), Err(EBADF));
    assert_eq!(fstatfs_core(&ctx, -1, buf), Err(EBADF));
    assert_eq!(call(&mut ctx, cb_posix_fstatfs, &[file, buf]), -1);
    assert!(ctx.read_bytes(buf, KERNEL_STATFS_SIZE).unwrap().iter().all(|&b| b == 0x5a), "untouched on failure");
    assert_eq!(call(&mut ctx, cb_sce_close, &[dir]), 0);
    // fstatfs (libkernel_sys) is the same call as _fstatfs.
    let target = |name: &str| EXPORTS.iter().find(|e| e.1 == name).map(|e| e.3 as usize);
    assert_eq!(target("fstatfs"), Some(cb_posix_fstatfs as HleCallback as usize));
    assert_eq!(target("fstatfs"), target("_fstatfs"));
}

/// shm_open: anonymous objects (O_RDWR only) and named ones (O_CREAT,
/// O_EXCL, O_TRUNC, per-open access), with the accepted flag set enforced.
#[test]
fn shm_open_objects() {
    const BASE: u64 = 0x5CEE_0000_0000;
    let vm: &'static VirtualMemory = Box::leak(Box::new(VirtualMemory::new()));
    vm.map(BASE, 0x10000, MemProtect::RW).unwrap();
    let ctx = CpuContext::new(vm);
    let (text, back, last, name) = (BASE, BASE + 0x100, BASE + 0x200, BASE + 0x300);
    ctx.write_bytes(text, b"anonymous\0");
    let first = shm_open_core(&ctx, SHM_ANON, 0x10_0202).unwrap() as i32;
    let second = shm_open_core(&ctx, SHM_ANON, 0x2).unwrap() as i32;
    let third = shm_open_core(&ctx, SHM_ANON, 0x10_0e02).unwrap() as i32;
    assert!(first >= 3 && second >= 3 && third >= 3);
    assert!(first != second && second != third && first != third);
    assert_eq!(fcntl_core(first, F_GETFD, 0), Ok(1), "O_CLOEXEC");
    assert_eq!(fcntl_core(second, F_GETFD, 0), Ok(0));
    assert_eq!(ftruncate_core(first, 0x4000), Ok(0));
    assert_eq!(pwrite_core(&ctx, first, text, 10, 0x1000), Ok(10));
    assert_eq!(pread_core(&ctx, first, back, 10, 0x1000), Ok(10));
    assert_eq!(&ctx.read_bytes(back, 10).unwrap()[..], b"anonymous\0");
    ctx.write_bytes(last, &[1]);
    assert_eq!(pread_core(&ctx, first, last, 1, 0x3fff), Ok(1));
    assert_eq!(ctx.read_bytes(last, 1).unwrap(), vec![0]);
    assert_eq!(pread_core(&ctx, first, last, 1, 0x4000), Ok(0));
    assert_eq!(pread_core(&ctx, second, back, 10, 0), Ok(0), "anonymous objects are separate");
    assert_eq!(pwrite_core(&ctx, third, text, 10, 0), Ok(10));
    for fd in [first, second, third] {
        assert_eq!(close_core(fd), Ok(0));
    }
    for flags in [0x0, 0x200, 0x1, 0x3, 0x2 | 0x8, 0x2 | 0x4] {
        assert_eq!(shm_open_core(&ctx, SHM_ANON, flags), Err(EINVAL), "anonymous flags {flags:#x}");
    }
    let mut c = CpuContext::new(vm);
    c.set(CpuRegister::Rdi, SHM_ANON);
    c.set(CpuRegister::Rsi, 0);
    assert_eq!(cb_posix_shm_open(&mut c), -1);

    ctx.write_bytes(name, format!("/rps5e-shm-test-{}\0", std::process::id()).as_bytes());
    assert_eq!(shm_open_core(&ctx, name, 0x1), Err(EINVAL));
    assert_eq!(shm_open_core(&ctx, name, 0x202 | 0x8), Err(EINVAL));
    assert_eq!(shm_open_core(&ctx, name, 0x2), Err(ENOENT), "missing without O_CREAT");
    let a = shm_open_core(&ctx, name, 0x202).unwrap() as i32;
    ctx.write_bytes(text, b"named");
    assert_eq!(pwrite_core(&ctx, a, text, 5, 0), Ok(5));
    let b = shm_open_core(&ctx, name, 0x2).unwrap() as i32;
    assert_eq!(pread_core(&ctx, b, back, 5, 0), Ok(5));
    assert_eq!(&ctx.read_bytes(back, 5).unwrap()[..], b"named", "one object behind both");
    assert_eq!(shm_open_core(&ctx, name, 0xa02), Err(EEXIST));
    let ro = shm_open_core(&ctx, name, 0x0).unwrap() as i32;
    assert_eq!(pread_core(&ctx, ro, back, 5, 0), Ok(5));
    assert_eq!(pwrite_core(&ctx, ro, text, 5, 0), Err(EBADF), "opened read-only");
    let t = shm_open_core(&ctx, name, 0x402).unwrap() as i32;
    assert_eq!(pread_core(&ctx, a, back, 5, 0), Ok(0), "O_TRUNC emptied the object");
    for fd in [a, b, ro, t] {
        assert_eq!(close_core(fd), Ok(0));
    }
    let _ = vm.unmap(BASE, 0x10000);
}

/// sceKernelGetFsSandboxRandomWord: one alphanumeric word for every call
/// and thread; nothing is mounted under it.
#[test]
fn sandbox_random_word() {
    const BASE: u64 = 0x5CED_0000_0000;
    let vm: &'static VirtualMemory = Box::leak(Box::new(VirtualMemory::new()));
    vm.map(BASE, 0x1000, MemProtect::RW).unwrap();
    let mut ctx = CpuContext::new(vm);
    let word_addr = cb_get_fs_sandbox_random_word(&mut ctx) as u64;
    assert_ne!(word_addr, 0);
    let word = ctx.read_cstr_bytes(word_addr, 64).unwrap();
    assert!(!word.is_empty() && word.iter().all(u8::is_ascii_alphanumeric));
    assert_eq!(cb_get_fs_sandbox_random_word(&mut ctx) as u64, word_addr);
    let from_thread = std::thread::spawn(move || cb_get_fs_sandbox_random_word(&mut CpuContext::new(vm)) as u64)
        .join()
        .unwrap();
    assert_eq!(from_thread, word_addr);
    let path = format!("/{}/common/lib/libc.sprx\0", String::from_utf8(word).unwrap());
    ctx.write_bytes(BASE, path.as_bytes());
    assert_eq!(open_core(&ctx, BASE, 0), Err(ENOENT));
    let _ = vm.unmap(BASE, 0x1000);
}

/// A title whose param.json declares download data finds /download0 as a
/// directory from the start; an unreadable param.json binds /app0 anyway.
#[test]
fn download0_present_for_declared_download_data() {
    let _app0 = APP0_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let root = super::tests::temp_root("download0");
    std::env::set_var("RPS5E_DATA_DIR", root.join("user"));
    let game = root.join("declared");
    fs::create_dir_all(game.join("sce_sys")).unwrap();
    fs::write(game.join("sce_sys/param.json"), r#"{"titleId": "PPSA00000", "downloadDataSize": 1024}"#).unwrap();
    set_app0_root(&game);
    let dl = resolve_guest_path("/download0").unwrap();
    assert!(dl.is_dir(), "/download0 exists at start-up");
    assert!(dl.starts_with(root.join("user/PPSA00000")));
    let invalid = root.join("invalid");
    fs::create_dir_all(invalid.join("sce_sys")).unwrap();
    fs::write(invalid.join("sce_sys/param.json"), b"{ not json").unwrap();
    set_app0_root(&invalid);
    assert_eq!(resolve_guest_path("/app0/sce_sys"), Some(invalid.join("sce_sys")));
    let _ = fs::remove_dir_all(&root);
}

/// utime(path, utimbuf*): whole-second times, NULL times is "now", errors
/// are -1.
#[test]
fn utime_sets_whole_second_times() {
    use std::os::unix::fs::MetadataExt;
    let f = Fixture::new(0x5F83_0000_0000, "utime");
    let mut ctx = f.ctx();
    fs::write(f.root.join("data.bin"), b"timestamps").unwrap();
    let path = f.s(&ctx, 0, &f.guest("data.bin"));
    let times = f.base;
    ctx.write_u64(times, 1_000_000_000);
    ctx.write_u64(times + 8, 1_000_000_123);
    assert_eq!(call(&mut ctx, cb_posix_utime, &[path, times]), 0);
    let m = fs::metadata(f.root.join("data.bin")).unwrap();
    assert_eq!((m.atime(), m.atime_nsec(), m.mtime(), m.mtime_nsec()), (1_000_000_000, 0, 1_000_000_123, 0));
    let missing = f.s(&ctx, 1, &f.guest("missing.bin"));
    assert_eq!(call(&mut ctx, cb_posix_utime, &[missing, times]), -1);
    let empty = f.s(&ctx, 2, "");
    assert_eq!(call(&mut ctx, cb_posix_utime, &[empty, times]), -1);
    assert_eq!(call(&mut ctx, cb_posix_utime, &[0, times]), -1);
    assert_eq!(call(&mut ctx, cb_posix_utime, &[0, 0]), -1);
    let m = fs::metadata(f.root.join("data.bin")).unwrap();
    assert_eq!((m.atime(), m.mtime()), (1_000_000_000, 1_000_000_123), "failed calls change nothing");
    let now = || std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
    let before = now();
    assert_eq!(call(&mut ctx, cb_posix_utime, &[path, 0]), 0);
    let after = now();
    let m = fs::metadata(f.root.join("data.bin")).unwrap();
    assert!((before - 1..=after + 1).contains(&m.mtime()) && (before - 1..=after + 1).contains(&m.atime()));
}

/// sync() flushes and forgets every path written since the last sync
/// (a removed file is no longer recorded after sync).
#[test]
fn posix_sync_forgets_written_paths() {
    let f = Fixture::new(0x5F84_0000_0000, "sync");
    let mut ctx = f.ctx();
    let path = f.s(&ctx, 0, &f.guest("synced.bin"));
    let fd = open(&mut ctx, path, O_WRONLY | O_CREAT | 0x400);
    assert!(fd >= 3);
    assert_eq!(call(&mut ctx, cb_sce_close, &[fd as u64]), 0);
    assert_eq!(call(&mut ctx, cb_sce_unlink, &[path]), 0);
    assert_eq!(call(&mut ctx, cb_posix_sync, &[]), 0);
    let host = f.root.join("synced.bin");
    assert!(!WRITTEN_PATHS.lock().unwrap_or_else(|p| p.into_inner()).contains(&host));
}

/// sceKernelMkdir: an empty path is ENOENT, "/" and existing paths (with a
/// trailing slash or "/." too) EEXIST, a trailing slash creates the
/// directory, and a file used as a parent is ENOTDIR.
#[test]
fn mkdir_paths_and_errors() {
    let f = Fixture::new(0x6A15_0000_0000, "mkdir");
    let mut ctx = f.ctx();
    let empty = f.s(&ctx, 0, "");
    assert_eq!(call(&mut ctx, cb_sce_mkdir, &[empty, 0o777]), sce_error(ENOENT));
    let root = f.s(&ctx, 1, "/");
    assert_eq!(call(&mut ctx, cb_sce_mkdir, &[root, 0o777]), sce_error(EEXIST));
    let probe = f.s(&ctx, 2, &f.guest("probe"));
    assert_eq!(call(&mut ctx, cb_sce_mkdir, &[probe, 0o777]), 0);
    fs::write(f.root.join("probe/existing.bin"), b"existing").unwrap();
    let nested = f.s(&ctx, 3, &f.guest("probe/nested/"));
    assert_eq!(call(&mut ctx, cb_sce_mkdir, &[nested, 0o777]), 0);
    assert!(f.root.join("probe/nested").is_dir());
    assert_eq!(call(&mut ctx, cb_sce_mkdir, &[nested, 0o777]), sce_error(EEXIST));
    let dot = f.s(&ctx, 4, &f.guest("probe/./"));
    assert_eq!(call(&mut ctx, cb_sce_mkdir, &[dot, 0o777]), sce_error(EEXIST));
    let under_file = f.s(&ctx, 5, &f.guest("probe/existing.bin/child"));
    assert_eq!(call(&mut ctx, cb_sce_mkdir, &[under_file, 0o777]), sce_error(ENOTDIR));
    let orphan = f.s(&ctx, 6, &f.guest("missing/child"));
    assert_eq!(call(&mut ctx, cb_sce_mkdir, &[orphan, 0o777]), sce_error(ENOENT));
}

/// open with O_DIRECTORY (what opendir uses): a directory opens, a file is
/// ENOTDIR and stays untouched even with O_CREAT | O_TRUNC, a missing path
/// is ENOENT and a symbolic link loop ELOOP.
#[test]
fn open_directory_flag_errors() {
    let f = Fixture::new(0x6A16_0000_0000, "odir");
    let mut ctx = f.ctx();
    fs::write(f.root.join("present.txt"), b"posix").unwrap();
    std::os::unix::fs::symlink("symlink-loop", f.root.join("symlink-loop")).unwrap();
    let dir = f.s(&ctx, 0, &f.mount);
    let fd = open(&mut ctx, dir, O_DIRECTORY);
    assert!(fd >= 3);
    assert_eq!(call(&mut ctx, cb_sce_close, &[fd as u64]), 0);
    let present = f.s(&ctx, 1, &f.guest("present.txt"));
    assert_eq!(open(&mut ctx, present, O_DIRECTORY), sce_error(ENOTDIR));
    assert_eq!(open(&mut ctx, present, O_DIRECTORY | 0x402), sce_error(ENOTDIR));
    assert_eq!(fs::metadata(f.root.join("present.txt")).unwrap().len(), 5);
    let missing = f.s(&ctx, 2, &f.guest("missing.txt"));
    assert_eq!(open(&mut ctx, missing, O_DIRECTORY), sce_error(ENOENT));
    let lp = f.s(&ctx, 3, &f.guest("symlink-loop"));
    assert_eq!(open(&mut ctx, lp, O_DIRECTORY), sce_error(ELOOP));
}

/// realpath folds "." / ".." and both separators, keeps the given case,
/// leaves the path resolved up to a failure in the caller's buffer, and
/// bounds every step by PATH_MAX.
#[test]
fn realpath_resolves_lexically() {
    let f = Fixture::new(0x6A17_0000_0000, "rpath");
    let mut ctx = f.ctx();
    fs::create_dir_all(f.root.join("sub")).unwrap();
    fs::write(f.root.join("file.txt"), b"file").unwrap();
    fs::write(f.root.join("sub/inner.txt"), b"inner").unwrap();
    let guest = f.mount.clone();
    let name = guest.trim_start_matches('/').to_string();
    let out = f.base + 0x1000;
    let read_out = |ctx: &CpuContext| String::from_utf8(ctx.read_cstr_bytes(out, 2048).unwrap()).unwrap();
    let resolves = |ctx: &mut CpuContext, path: &str, expected: &str| {
        ctx.write_bytes(out, &[b'x'; 1024]);
        let p = f.s(ctx, 0, path);
        assert_eq!(call(ctx, cb_realpath, &[p, out]), out as HleResult, "{path}");
        assert_eq!(read_out(ctx), expected, "{path}");
    };
    resolves(&mut ctx, "/", "/");
    resolves(&mut ctx, &guest, &guest);
    resolves(&mut ctx, &format!("{guest}/"), &guest);
    resolves(&mut ctx, &format!("{guest}/sub/../file.txt"), &format!("{guest}/file.txt"));
    resolves(&mut ctx, &format!("{guest}/./sub//inner.txt"), &format!("{guest}/sub/inner.txt"));
    resolves(&mut ctx, &format!("/../..{guest}/sub/"), &format!("{guest}/sub"));
    resolves(&mut ctx, &format!("{guest}/sub/.."), &guest);
    resolves(&mut ctx, "/..", "/");
    resolves(&mut ctx, &format!("{guest}\\sub\\..\\file.txt"), &format!("{guest}/file.txt"));
    resolves(&mut ctx, &format!("\\{name}\\sub\\"), &format!("{guest}/sub"));
    resolves(&mut ctx, &format!("{name}/sub/inner.txt"), &format!("{guest}/sub/inner.txt"));
    let upper = guest.to_uppercase();
    resolves(&mut ctx, &format!("{upper}/FILE.TXT"), &format!("{upper}/FILE.TXT"));

    let fails = |ctx: &mut CpuContext, path: &str, errno: i32, trouble: Option<&str>| {
        assert_eq!(realpath_resolve(path).err().map(|e| e.0), Some(errno), "{path}");
        ctx.write_bytes(out, &[b'x'; 1024]);
        let p = f.s(ctx, 0, path);
        assert_eq!(call(ctx, cb_realpath, &[p, out]), 0, "{path}");
        if let Some(t) = trouble {
            assert_eq!(read_out(ctx), t, "{path}");
        }
    };
    fails(&mut ctx, "/realpath-unmounted/x", ENOENT, Some("/realpath-unmounted"));
    fails(&mut ctx, &format!("{guest}/missing"), ENOENT, Some(&format!("{guest}/missing")));
    fails(&mut ctx, &format!("{guest}/missing/../file.txt"), ENOENT, Some(&format!("{guest}/missing")));
    fails(&mut ctx, &format!("{guest}/file.txt/"), ENOTDIR, Some(&format!("{guest}/file.txt")));
    fails(&mut ctx, &format!("{guest}/file.txt/x"), ENOTDIR, Some(&format!("{guest}/file.txt")));
    fails(&mut ctx, &format!("{guest}/file.txt/."), ENOTDIR, None);
    fails(&mut ctx, &format!("{guest}/file.txt/.."), ENOTDIR, None);
    let overflowing = format!("/{}", "a".repeat(1023));
    fails(&mut ctx, &overflowing, ENAMETOOLONG, Some(&overflowing[..1023]));
    fails(&mut ctx, &format!("{guest}/sub/{}/inner.txt", ".".repeat(1024)), ENAMETOOLONG, None);
    let mut boundary = guest.clone();
    while boundary.len() + 8 <= 1023 {
        boundary += "/missing";
    }
    boundary += &format!("/{}", "m".repeat(1023 - boundary.len() - 1));
    assert_eq!(boundary.len(), 1023);
    fails(&mut ctx, &boundary, ENOENT, Some(&format!("{guest}/missing")));

    assert_eq!(call(&mut ctx, cb_realpath, &[0, out]), 0, "NULL path: EINVAL");
    let empty = f.s(&ctx, 1, "");
    assert_eq!(call(&mut ctx, cb_realpath, &[empty, out]), 0, "empty path: ENOENT");
}

/// fcntl record locks: one process never conflicts with itself, so F_SETLK
/// succeeds and F_GETLK reports F_UNLCK with the resolved start, after
/// FreeBSD's checks (access mode, whence, ranges, descriptor kind).
#[test]
fn fcntl_record_locks() {
    let f = Fixture::new(0x7C3B_0000_0000, "fcntl_locks");
    let mut ctx = f.ctx();
    let errno = crate::kernel::bind_test_errno(&ctx);
    fs::write(f.root.join("l.bin"), b"0123456789abcdef").unwrap();
    let p = f.s(&ctx, 0, &f.guest("l.bin"));
    let fd = open(&mut ctx, p, 0) as u64;
    let fl = f.base + 0x800;
    let put = |ctx: &CpuContext, ty: i16, whence: i16, start: i64, len: i64| {
        ctx.write_bytes(fl, &[0u8; 40]);
        ctx.write_bytes(fl, &start.to_le_bytes());
        ctx.write_bytes(fl + 8, &len.to_le_bytes());
        ctx.write_bytes(fl + 20, &ty.to_le_bytes());
        ctx.write_bytes(fl + 22, &whence.to_le_bytes());
    };
    let ty = |ctx: &CpuContext| i16::from_le_bytes(ctx.read_bytes(fl + 20, 2).unwrap().try_into().unwrap());
    put(&ctx, 1, 0, 0, 0);
    assert_eq!(call(&mut ctx, cb_posix_fcntl, &[fd, 11, fl]), 0);
    assert_eq!(ty(&ctx), 2, "F_UNLCK");
    put(&ctx, 3, 0, 0, 0);
    assert_eq!(call(&mut ctx, cb_posix_fcntl, &[fd, 12, fl]), -1, "write lock on a read-only fd");
    assert_eq!(ctx.read_u32(errno), Some(9));
    put(&ctx, 1, 0, 0, 0);
    assert_eq!(call(&mut ctx, cb_posix_fcntl, &[fd, 12, fl]), 0);
    assert_eq!(call(&mut ctx, cb_posix_fcntl, &[fd, 13, fl]), 0);
    put(&ctx, 1, 5, 0, 0);
    assert_eq!(call(&mut ctx, cb_posix_fcntl, &[fd, 11, fl]), -1);
    assert_eq!(ctx.read_u32(errno), Some(22));
    put(&ctx, 1, 0, -1, 0);
    assert_eq!(call(&mut ctx, cb_posix_fcntl, &[fd, 12, fl]), -1);
    assert_eq!(ctx.read_u32(errno), Some(22));
    put(&ctx, 7, 0, 0, 0);
    assert_eq!(call(&mut ctx, cb_posix_fcntl, &[fd, 12, fl]), -1);
    assert_eq!(ctx.read_u32(errno), Some(22));
    assert_eq!(call(&mut ctx, cb_sce_lseek, &[fd, 10, SEEK_SET as u64]), 10);
    put(&ctx, 1, 1, 5, 0);
    assert_eq!(call(&mut ctx, cb_posix_fcntl, &[fd, 11, fl]), 0);
    assert_eq!(ctx.read_u64(fl), Some(15), "SEEK_CUR start");
    put(&ctx, 1, 2, -6, 0);
    assert_eq!(call(&mut ctx, cb_posix_fcntl, &[fd, 11, fl]), 0);
    assert_eq!(ctx.read_u64(fl), Some(10), "SEEK_END start");
    // The old command writes the 24-byte struct only.
    put(&ctx, 1, 0, 0, 0);
    ctx.write_bytes(fl + 24, &[0xAB]);
    assert_eq!(call(&mut ctx, cb_posix_fcntl, &[fd, 7, fl]), 0);
    assert_eq!(ty(&ctx), 2);
    assert_eq!(ctx.read_bytes(fl + 24, 1).unwrap(), [0xAB]);
    assert_eq!(call(&mut ctx, cb_posix_fcntl, &[fd, 11, 0]), -1);
    assert_eq!(ctx.read_u32(errno), Some(14));
    assert_eq!(call(&mut ctx, cb_sce_close, &[fd]), 0);
    // A pipe has no record locks.
    let pfd = f.base + 0x900;
    assert_eq!(call(&mut ctx, cb_posix_pipe, &[pfd]), 0);
    let (r, w) = (ctx.read_u32(pfd).unwrap() as u64, ctx.read_u32(pfd + 4).unwrap() as u64);
    put(&ctx, 1, 0, 0, 0);
    assert_eq!(call(&mut ctx, cb_posix_fcntl, &[r, 11, fl]), -1);
    assert_eq!(ctx.read_u32(errno), Some(9));
    assert_eq!(call(&mut ctx, cb_sce_close, &[r]), 0);
    assert_eq!(call(&mut ctx, cb_sce_close, &[w]), 0);
}

/// O_FSYNC (0x80), from open or F_SETFL and shared by duplicates, syncs
/// every write, pwrite, writev and pwritev to a file once; clearing it on a
/// duplicate stops the syncs; a pipe accepts the flag and has nothing to
/// sync.
#[test]
fn fcntl_fsync_syncs_each_write() {
    const GETFL: u64 = 3;
    const SETFL: u64 = 4;
    const SYNC: u64 = 0x80;
    const RDWR_CREAT: u64 = 2 | 0x200;
    let synced = || SYNCHRONOUS_WRITES.load(std::sync::atomic::Ordering::Relaxed);
    let f = Fixture::new(0x7C3E_8000_0000, "fsync");
    let mut ctx = f.ctx();
    let path = f.s(&ctx, 0, &f.guest("sync.bin"));
    let file = call(&mut ctx, cb_posix_open, &[path, RDWR_CREAT, 0o600]) as u64;
    assert!((file as i64) >= 0);
    let text = f.base;
    ctx.write_bytes(text, b"abcdefxyz");
    assert_eq!(call(&mut ctx, cb_posix_fcntl, &[file, GETFL]) as u64 & SYNC, 0);
    let before = synced();
    assert_eq!(call(&mut ctx, cb_posix_write, &[file, text, 1]), 1);
    assert_eq!(synced(), before);

    assert_eq!(call(&mut ctx, cb_posix_fcntl, &[file, SETFL, SYNC]), 0);
    assert_ne!(call(&mut ctx, cb_posix_fcntl, &[file, GETFL]) as u64 & SYNC, 0);
    let before = synced();
    assert_eq!(call(&mut ctx, cb_posix_write, &[file, text + 1, 1]), 1);
    assert_eq!(synced(), before + 1);
    let iov = f.base + 0x100;
    for (i, (base, len)) in [(text + 2, 1u64), (text + 3, 1)].into_iter().enumerate() {
        ctx.write_u64(iov + i as u64 * 16, base);
        ctx.write_u64(iov + i as u64 * 16 + 8, len);
    }
    assert_eq!(call(&mut ctx, cb_posix_writev, &[file, iov, 2]), 2);
    assert_eq!(synced(), before + 2, "one sync per gathered write");
    assert_eq!(call(&mut ctx, cb_posix_pwrite, &[file, text + 6, 1, 10]), 1);
    assert_eq!(synced(), before + 3);
    let tail = f.base + 0x200;
    for (i, base) in [text + 7, text + 8].into_iter().enumerate() {
        ctx.write_u64(tail + i as u64 * 16, base);
        ctx.write_u64(tail + i as u64 * 16 + 8, 1);
    }
    assert_eq!(call(&mut ctx, cb_posix_pwritev, &[file, tail, 2, 11]), 2);
    assert_eq!(synced(), before + 4);

    let duplicate = call(&mut ctx, cb_posix_dup, &[file]) as u64;
    assert!((duplicate as i64) >= 0);
    assert_ne!(call(&mut ctx, cb_posix_fcntl, &[duplicate, GETFL]) as u64 & SYNC, 0);
    assert_eq!(call(&mut ctx, cb_posix_write, &[duplicate, text + 4, 1]), 1);
    assert_eq!(synced(), before + 5);
    assert_eq!(call(&mut ctx, cb_posix_fcntl, &[duplicate, SETFL, 0]), 0);
    assert_eq!(call(&mut ctx, cb_posix_fcntl, &[file, GETFL]) as u64 & SYNC, 0);
    let before = synced();
    assert_eq!(call(&mut ctx, cb_posix_write, &[file, text + 5, 1]), 1);
    assert_eq!(synced(), before);
    let contents = f.base + 0x300;
    assert_eq!(call(&mut ctx, cb_posix_pread, &[file, contents, 13, 0]), 13);
    let got = ctx.read_bytes(contents, 13).unwrap();
    assert_eq!((&got[..6], &got[10..]), (&b"abcdef"[..], &b"xyz"[..]));
    // The flag keeps the access mode and the other status flags.
    assert_eq!(call(&mut ctx, cb_posix_fcntl, &[file, SETFL, 8 | SYNC]), 0);
    assert_eq!(call(&mut ctx, cb_posix_fcntl, &[file, GETFL]) as u64, 2 | 8 | SYNC);
    assert_eq!(call(&mut ctx, cb_posix_close, &[duplicate]), 0);
    assert_eq!(call(&mut ctx, cb_posix_close, &[file]), 0);

    let opened = f.s(&ctx, 1, &f.guest("sync-open.bin"));
    let file = call(&mut ctx, cb_posix_open, &[opened, RDWR_CREAT | SYNC, 0o600]) as u64;
    assert_ne!(call(&mut ctx, cb_posix_fcntl, &[file, GETFL]) as u64 & SYNC, 0);
    let before = synced();
    assert_eq!(call(&mut ctx, cb_posix_write, &[file, text, 1]), 1);
    assert_eq!(synced(), before + 1);
    assert_eq!(call(&mut ctx, cb_posix_close, &[file]), 0);

    let pfd = f.base + 0x900;
    assert_eq!(call(&mut ctx, cb_posix_pipe, &[pfd]), 0);
    let (r, w) = (ctx.read_u32(pfd).unwrap() as u64, ctx.read_u32(pfd + 4).unwrap() as u64);
    assert_eq!(call(&mut ctx, cb_posix_fcntl, &[w, SETFL, SYNC]), 0);
    assert_ne!(call(&mut ctx, cb_posix_fcntl, &[w, GETFL]) as u64 & SYNC, 0);
    let before = synced();
    assert_eq!(call(&mut ctx, cb_posix_write, &[w, text, 1]), 1);
    assert_eq!(synced(), before);
    assert_eq!(call(&mut ctx, cb_posix_close, &[r]), 0);
    assert_eq!(call(&mut ctx, cb_posix_close, &[w]), 0);
}

/// Argument checks the console makes before the path or the operation
/// (reference GuestFileSystem / GuestFlock cases measured on a PS5): flock
/// on a closed descriptor EBADF and on a pipe EOPNOTSUPP whatever the
/// operation; access checks the mode first; utimes and lutimes the times;
/// open and openat refuse access mode 3 and O_CREAT|O_EXCL|O_DIRECTORY.
#[test]
fn checks_before_the_path() {
    let f = Fixture::new(0x7C3C_4000_0000, "argorder");
    let mut ctx = f.ctx();
    let errno = crate::kernel::bind_test_errno(&ctx);
    let pfd = f.base + 0x900;
    assert_eq!(call(&mut ctx, cb_posix_pipe, &[pfd]), 0);
    let (r, w) = (ctx.read_u32(pfd).unwrap() as u64, ctx.read_u32(pfd + 4).unwrap() as u64);
    for op in [0u64, 2, 0x40] {
        assert_eq!(call(&mut ctx, cb_posix_flock, &[r, op]), -1);
        assert_eq!(ctx.read_u32(errno), Some(EOPNOTSUPP as u32), "pipe, op {op}");
        assert_eq!(call(&mut ctx, cb_posix_flock, &[9999, op]), -1);
        assert_eq!(ctx.read_u32(errno), Some(EBADF as u32), "closed, op {op}");
    }
    assert_eq!(call(&mut ctx, cb_sce_close, &[r]), 0);
    assert_eq!(call(&mut ctx, cb_sce_close, &[w]), 0);
    assert_eq!(at::access_core(&ctx, 0, 8), Err(EINVAL));
    assert_eq!(at::access_core(&ctx, 0, 0), Err(EFAULT));
    let times = f.base + 0xA00;
    ctx.write_u64(times, 0);
    ctx.write_u64(times + 8, 1_000_000);
    ctx.write_u64(times + 16, 0);
    ctx.write_u64(times + 24, 0);
    assert_eq!(utimes_core(&ctx, 0, times), Err(EINVAL));
    assert_eq!(at::lutimes_core(&ctx, 0, times), Err(EINVAL));
    let path = f.s(&ctx, 0, &f.guest("never.txt"));
    for flags in [O_ACCMODE, O_CREAT | O_EXCL | O_DIRECTORY] {
        assert_eq!(open_core(&ctx, 0, flags), Err(EINVAL));
        assert_eq!(open_core(&ctx, path, flags), Err(EINVAL));
        assert_eq!(at::openat_core(&ctx, 9999, path, flags), Err(EINVAL));
    }
    assert!(!f.root.join("never.txt").exists());
}

/// A hung-up socket polls readable and hung up, never writable: a host
/// stream socket shut both ways reports POLLIN|POLLOUT|POLLHUP on Linux.
#[test]
fn poll_hangup_clears_pollout_on_sockets() {
    let mut fds = [0i32; 2];
    // SAFETY: a local socket pair, closed below.
    assert_eq!(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) }, 0);
    unsafe {
        libc::close(fds[1]);
        libc::shutdown(fds[0], libc::SHUT_WR);
    }
    let mut p = libc::pollfd { fd: fds[0], events: libc::POLLIN | libc::POLLOUT, revents: 0 };
    assert_eq!(unsafe { libc::poll(&mut p, 1, 0) }, 1);
    unsafe { libc::close(fds[0]) };
    assert_ne!(p.revents & libc::POLLHUP, 0, "host revents {:#x}", p.revents);
    let r = host_revents(POLLIN | POLLOUT, p.revents, false);
    assert_eq!(r & (POLLIN | POLLOUT | POLLHUP), POLLIN | POLLHUP);
    assert_eq!(host_revents(POLLOUT, POLLOUT_HOST_WITH_HUP, false) & POLLOUT, 0);
}

const POLLOUT_HOST_WITH_HUP: i16 = libc::POLLOUT | libc::POLLHUP;
