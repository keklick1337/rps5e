// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only
//
// Unit tests of the `ps5hle::filesystem::at` module.

use super::*;
use crate::context::CpuRegister;
use ps5vm::{MemProtect, VirtualMemory};

const BASE: u64 = 0x7E61_0000_0000;

fn call(ctx: &mut CpuContext, cb: HleCallback, args: &[u64]) -> HleResult {
    let regs = [CpuRegister::Rdi, CpuRegister::Rsi, CpuRegister::Rdx, CpuRegister::Rcx];
    for (r, v) in regs.iter().zip(args.iter().chain(std::iter::repeat(&0))) {
        ctx.set(*r, *v);
    }
    cb(ctx)
}

fn s(vm: &VirtualMemory, slot: u64, text: &str) -> u64 {
    let addr = BASE + 0x1000 + slot * 0x400;
    let mut b = text.as_bytes().to_vec();
    b.push(0);
    vm.write(addr, &b).unwrap();
    addr
}

fn open(ctx: &mut CpuContext, path: &str, flags: u64) -> i32 {
    let addr = s(ctx.vm, 0, path);
    open_core(ctx, addr, flags).unwrap() as i32
}

/// The `*at` calls, the working directory and the path checks, on a mount
/// of a scratch directory under the repository's local/tmp/.
#[test]
fn directory_relative_calls() {
    std::thread::spawn(|| {
        let vm: &'static VirtualMemory = Box::leak(Box::new(VirtualMemory::new()));
        vm.map(BASE, 0x10000, MemProtect::RW).unwrap();
        let mut ctx = CpuContext::new(vm);
        let errno = crate::kernel::bind_test_errno(&ctx);
        let err = |ctx: &CpuContext| ctx.read_u32(errno).unwrap() as i32;
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../local/tmp/at-tests-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("dir")).unwrap();
        fs::write(root.join("file.txt"), b"x").unwrap();
        assert!(register_guest_path_mount("/attest0", &root));
        let fdcwd = AT_FDCWD as u32 as u64;

        // Relative to a directory descriptor.
        let dir = open(&mut ctx, "/attest0/dir", O_DIRECTORY) as u64;
        let f = call(&mut ctx, cb_openat, &[dir, s(vm, 1, "made.bin"), O_WRONLY | O_CREAT, 0o644]);
        assert!(f > 2, "openat: {}", err(&ctx));
        assert!(root.join("dir/made.bin").exists());
        assert_eq!(close_core(f as i32), Ok(0));
        assert_eq!(call(&mut ctx, cb_mkdirat, &[dir, s(vm, 1, "sub"), 0o755]), 0);
        assert!(root.join("dir/sub").is_dir());
        assert_eq!(call(&mut ctx, cb_fchmodat, &[dir, s(vm, 1, "made.bin"), 0o600, 0]), 0);
        assert_eq!(call(&mut ctx, cb_fchmodat, &[dir, s(vm, 1, "made.bin"), 0o600, 1]), -1);
        assert_eq!(err(&ctx), EINVAL);
        assert_eq!(call(&mut ctx, cb_renameat, &[dir, s(vm, 1, "made.bin"), fdcwd, s(vm, 2, "/attest0/moved.bin")]), 0);
        assert!(root.join("moved.bin").exists() && !root.join("dir/made.bin").exists());
        assert_eq!(call(&mut ctx, cb_unlinkat, &[dir, s(vm, 1, "sub"), 0]), -1);
        assert_eq!(call(&mut ctx, cb_unlinkat, &[dir, s(vm, 1, "sub"), AT_REMOVEDIR]), 0);
        assert!(!root.join("dir/sub").exists());
        assert_eq!(call(&mut ctx, cb_unlinkat, &[dir, s(vm, 1, "x"), 0x1]), -1);
        assert_eq!(err(&ctx), EINVAL);

        // A file descriptor is ENOTDIR, a closed one EBADF; an absolute
        // path ignores the descriptor.
        let file = open(&mut ctx, "/attest0/file.txt", 0) as u64;
        assert_eq!(call(&mut ctx, cb_mkdirat, &[file, s(vm, 1, "nope"), 0o755]), -1);
        assert_eq!(err(&ctx), ENOTDIR);
        assert_eq!(call(&mut ctx, cb_mkdirat, &[0x0FFF_FFF0, s(vm, 1, "nope"), 0o755]), -1);
        assert_eq!(err(&ctx), EBADF);
        assert_eq!(call(&mut ctx, cb_mkdirat, &[file, s(vm, 1, "/attest0/abs"), 0o755]), 0);
        assert!(root.join("abs").is_dir());
        assert_eq!(call(&mut ctx, cb_mkdirat, &[dir, 0, 0o755]), -1);
        assert_eq!(err(&ctx), EFAULT);
        assert_eq!(call(&mut ctx, cb_mkdirat, &[dir, s(vm, 1, ""), 0o755]), -1);
        assert_eq!(err(&ctx), ENOENT);

        // access, chflags, pathconf, lutimes.
        assert_eq!(call(&mut ctx, cb_access, &[s(vm, 1, "/attest0/file.txt"), 4]), 0);
        assert_eq!(call(&mut ctx, cb_access, &[s(vm, 1, "/attest0/file.txt"), 8]), -1);
        assert_eq!(err(&ctx), EINVAL);
        assert_eq!(call(&mut ctx, cb_access, &[s(vm, 1, "/attest0/none"), 0]), -1);
        assert_eq!(err(&ctx), ENOENT);
        assert_eq!(call(&mut ctx, cb_chflags, &[s(vm, 1, "/attest0/file.txt"), 0]), 0);
        assert_eq!(call(&mut ctx, cb_chflags, &[s(vm, 1, "/attest0/file.txt"), SF_SNAPSHOT]), -1);
        assert_eq!(err(&ctx), EPERM);
        assert_eq!(call(&mut ctx, cb_chflags, &[s(vm, 1, "/attest0/file.txt"), 0x4000_0000]), -1);
        assert_eq!(err(&ctx), EOPNOTSUPP);
        assert_eq!(call(&mut ctx, cb_chflags, &[s(vm, 1, "/attest0/none"), 0]), -1);
        assert_eq!(err(&ctx), ENOENT);
        // Symbolic links: chflags and access follow them (a dangling link
        // is ENOENT, a loop ELOOP, a trailing slash on a file ENOTDIR);
        // O_CREAT|O_EXCL on a link is EEXIST even when its target is missing.
        std::os::unix::fs::symlink("missing.bin", root.join("dangling.link")).unwrap();
        std::os::unix::fs::symlink("file.txt", root.join("file.link")).unwrap();
        std::os::unix::fs::symlink("loop.link", root.join("loop.link")).unwrap();
        std::os::unix::fs::symlink("excl-target.txt", root.join("excl.link")).unwrap();
        assert_eq!(call(&mut ctx, cb_chflags, &[s(vm, 1, "/attest0/dangling.link"), 0]), -1);
        assert_eq!(err(&ctx), ENOENT);
        assert_eq!(call(&mut ctx, cb_chflags, &[s(vm, 1, "/attest0/file.link"), 0]), 0);
        assert_eq!(call(&mut ctx, cb_chflags, &[s(vm, 1, "/attest0/file.link/"), 0]), -1);
        assert_eq!(err(&ctx), ENOTDIR);
        assert_eq!(call(&mut ctx, cb_chflags, &[s(vm, 1, "/attest0/file.link"), 0x20]), -1);
        assert_eq!(err(&ctx), EOPNOTSUPP);
        assert_eq!(call(&mut ctx, cb_access, &[s(vm, 1, "/attest0/dangling.link"), 0]), -1);
        assert_eq!(err(&ctx), ENOENT);
        assert_eq!(call(&mut ctx, cb_access, &[s(vm, 1, "/attest0/dangling.link"), 4]), -1);
        assert_eq!(err(&ctx), ENOENT);
        assert_eq!(call(&mut ctx, cb_access, &[s(vm, 1, "/attest0/loop.link"), 0]), -1);
        assert_eq!(err(&ctx), ELOOP);
        assert_eq!(call(&mut ctx, cb_access, &[s(vm, 1, "/attest0/file.link"), 6]), 0);
        let excl = s(vm, 1, "/attest0/excl.link");
        assert_eq!(open_core(&ctx, excl, O_WRONLY | O_CREAT | O_EXCL), Err(EEXIST));
        assert!(!root.join("excl-target.txt").exists());
        let made = open_core(&ctx, excl, O_WRONLY | O_CREAT).unwrap() as i32;
        assert_eq!(close_core(made), Ok(0));
        assert!(root.join("excl-target.txt").is_file());
        assert_eq!(open_core(&ctx, excl, O_WRONLY | O_CREAT | O_EXCL), Err(EEXIST));
        // fchflags: the chflags rules on a descriptor; EBADF when closed.
        assert_eq!(call(&mut ctx, cb_fchflags, &[file, 0]), 0);
        assert_eq!(call(&mut ctx, cb_fchflags, &[dir, SF_SNAPSHOT]), -1);
        assert_eq!(err(&ctx), EPERM);
        assert_eq!(call(&mut ctx, cb_fchflags, &[file, 0x4000_0000]), -1);
        assert_eq!(err(&ctx), EOPNOTSUPP);
        assert_eq!(call(&mut ctx, cb_fchflags, &[0x0FFF_FFF0, 0]), -1);
        assert_eq!(err(&ctx), EBADF);
        // fstatat: relative to the descriptor, AT_SYMLINK_NOFOLLOW only.
        let st = BASE + 0x200;
        assert_eq!(call(&mut ctx, cb_fstatat, &[fdcwd, s(vm, 1, "/attest0/file.txt"), st, 0]), 0);
        assert_eq!(ctx.read_u64(st + 72), Some(1), "st_size");
        assert_eq!(call(&mut ctx, cb_fstatat, &[dir, s(vm, 1, "../file.txt"), st, AT_SYMLINK_NOFOLLOW]), 0);
        assert_eq!(call(&mut ctx, cb_fstatat, &[dir, s(vm, 1, "none"), st, 0]), -1);
        assert_eq!(err(&ctx), ENOENT);
        assert_eq!(call(&mut ctx, cb_fstatat, &[file, s(vm, 1, "x"), st, 0]), -1);
        assert_eq!(err(&ctx), ENOTDIR);
        assert_eq!(call(&mut ctx, cb_fstatat, &[dir, s(vm, 1, "x"), st, 0x100]), -1);
        assert_eq!(err(&ctx), EINVAL);
        assert_eq!(call(&mut ctx, cb_fstatat, &[dir, 0, st, 0]), -1);
        assert_eq!(err(&ctx), EFAULT);
        assert_eq!(call(&mut ctx, cb_pathconf, &[s(vm, 1, "/attest0/file.txt"), 4]), 255);
        assert_eq!(call(&mut ctx, cb_pathconf, &[s(vm, 1, "/attest0/file.txt"), 5]), 1024);
        assert_eq!(call(&mut ctx, cb_pathconf, &[s(vm, 1, "/attest0/file.txt"), 999]), -1);
        assert_eq!(err(&ctx), EINVAL);
        assert_eq!(call(&mut ctx, cb_lutimes, &[s(vm, 1, "/attest0/file.txt"), 0]), 0);
        ctx.write_u64(BASE + 0x100, 0);
        ctx.write_u64(BASE + 0x108, 2_000_000);
        assert_eq!(call(&mut ctx, cb_lutimes, &[s(vm, 1, "/attest0/file.txt"), BASE + 0x100]), -1);
        assert_eq!(err(&ctx), EINVAL);

        // chdir / fchdir move the start of relative lookups; chroot only
        // accepts "/". Tests with relative /app0 paths hold this lock.
        let _app0 = APP0_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        assert_eq!(call(&mut ctx, cb_chdir, &[s(vm, 1, "/attest0/file.txt")]), -1);
        assert_eq!(err(&ctx), ENOTDIR);
        assert_eq!(call(&mut ctx, cb_chdir, &[s(vm, 1, "/attest0")]), 0);
        assert_eq!(call(&mut ctx, cb_access, &[s(vm, 1, "moved.bin"), 0]), 0);
        assert_eq!(call(&mut ctx, cb_fchdir, &[dir]), 0);
        assert_eq!(call(&mut ctx, cb_access, &[s(vm, 1, "../moved.bin"), 0]), 0);
        assert_eq!(call(&mut ctx, cb_fchdir, &[file]), -1);
        assert_eq!(err(&ctx), ENOTDIR);
        reset_cwd();
        assert_eq!(call(&mut ctx, cb_chroot, &[s(vm, 1, "/")]), 0);
        assert_eq!(call(&mut ctx, cb_chroot, &[s(vm, 1, "/attest0/dir")]), -1);
        assert_eq!(err(&ctx), EPERM);

        assert_eq!(close_core(file as i32), Ok(0));
        assert_eq!(close_core(dir as i32), Ok(0));
        unregister_guest_path_mount("/attest0");
        let _ = fs::remove_dir_all(&root);
    })
    .join()
    .unwrap();
}

#[test]
fn nids_match_names() {
    for &(nid, name, _, _) in AT_EXPORTS {
        assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
    }
}
