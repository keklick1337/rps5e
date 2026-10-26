// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Linux mmap backend. Other platforms can be added in submodules.

#[cfg(unix)]
pub use sys::MmapBackend;

#[cfg(unix)]
pub fn page_size() -> u64 {
    // SAFETY: getpagesize-equivalent via sysconf.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as u64 }
}

#[cfg(windows)]
pub use win::MmapBackend;

#[cfg(windows)]
pub fn page_size() -> u64 {
    crate::os::page_size()
}

/// Windows: fixed mappings are views of the placeholder space
/// (`os::windows::space`), which replaces ranges like `MAP_FIXED`.
#[cfg(windows)]
mod win {
    use crate::os::space;
    use crate::{MemProtect, VmError};

    pub struct MmapBackend;

    impl MmapBackend {
        pub fn new() -> Self {
            Self
        }

        fn round(size: u64) -> u64 {
            (size + 0xFFF) & !0xFFF
        }

        pub fn map(&mut self, addr: u64, size: u64, prot: MemProtect) -> Result<(), VmError> {
            if addr % 0x1000 != 0 || size == 0 {
                return Err(VmError::Unaligned(addr));
            }
            let size = Self::round(size);
            if !space::commit(addr, size, prot) {
                return Err(VmError::MapFailed(addr, size, std::io::Error::last_os_error().to_string()));
            }
            crate::ranges::track(addr, size, prot);
            Ok(())
        }

        pub fn unmap(&mut self, addr: u64, size: u64) -> Result<(), VmError> {
            let size = Self::round(size);
            if !space::release(addr, size) {
                return Err(VmError::UnmapFailed(format!("{addr:#x}+{size:#x} is not mapped")));
            }
            crate::ranges::untrack(addr, size);
            Ok(())
        }

        pub fn protect(&mut self, addr: u64, size: u64, prot: MemProtect) -> Result<(), VmError> {
            let size = Self::round(size);
            if !crate::os::protect(addr, size, prot) {
                return Err(VmError::ProtectFailed(std::io::Error::last_os_error().to_string()));
            }
            crate::ranges::track(addr, size, prot);
            Ok(())
        }
    }
}

#[cfg(unix)]
mod sys {
    use super::page_size;
    use crate::{MemProtect, VmError};
    use std::collections::BTreeMap;

    fn to_native_prot(prot: MemProtect) -> i32 {
        let mut p = libc::PROT_NONE;
        if prot.read {
            p |= libc::PROT_READ;
        }
        if prot.write {
            p |= libc::PROT_WRITE;
        }
        if prot.exec {
            p |= libc::PROT_EXEC;
        }
        p
    }

    /// Tracks every mapping (addr -> (size, prot)) so we can panic-avoid on
    /// writes to unmapped regions and provide useful diagnostics.
    pub struct MmapBackend {
        maps: BTreeMap<u64, (u64, MemProtect)>,
    }

    impl MmapBackend {
        pub fn new() -> Self {
            Self {
                maps: BTreeMap::new(),
            }
        }

        fn round(size: u64) -> u64 {
            let ps = page_size();
            (size + ps - 1) & !(ps - 1)
        }

        pub fn map(&mut self, addr: u64, size: u64, prot: MemProtect) -> Result<(), VmError> {
            let ps = page_size();
            if addr % ps != 0 || size == 0 {
                return Err(VmError::Unaligned(addr));
            }
            let size = Self::round(size);
            let ptr = addr as *mut libc::c_void;
            // MAP_FIXED so the guest address space is identity-mapped. We
            // trust the loader/HLE layer not to hand out overlapping ranges.
            const MAP_FIXED: i32 = 0x10;
            let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | MAP_FIXED;
            let result = unsafe {
                libc::mmap(ptr, size as usize, to_native_prot(prot), flags, -1, 0)
            };
            if result == libc::MAP_FAILED {
                let err = std::io::Error::last_os_error();
                return Err(VmError::MapFailed(addr, size, err.to_string()));
            }
            self.maps.insert(addr, (size, prot));
            crate::ranges::track(addr, size, prot);
            Ok(())
        }

        pub fn unmap(&mut self, addr: u64, size: u64) -> Result<(), VmError> {
            let size = Self::round(size);
            let rc = unsafe { libc::munmap(addr as *mut libc::c_void, size as usize) };
            if rc != 0 {
                return Err(VmError::UnmapFailed(
                    std::io::Error::last_os_error().to_string(),
                ));
            }
            self.maps.remove(&addr);
            crate::ranges::untrack(addr, size);
            Ok(())
        }

        pub fn protect(&mut self, addr: u64, size: u64, prot: MemProtect) -> Result<(), VmError> {
            let size = Self::round(size);
            let rc = unsafe {
                libc::mprotect(addr as *mut libc::c_void, size as usize, to_native_prot(prot))
            };
            if rc != 0 {
                return Err(VmError::ProtectFailed(
                    std::io::Error::last_os_error().to_string(),
                ));
            }
            if let Some(e) = self.maps.get_mut(&addr) {
                e.1 = prot;
            }
            crate::ranges::track(addr, size, prot);
            Ok(())
        }
    }
}
