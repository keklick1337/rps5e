// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Guest CPU context passed to HLE callbacks.
//!
//! The context owns a reference to guest virtual memory so callbacks can
//! read/write guest addresses directly. Registers follow the x86-64 SysV
//! ordering used by the Orbis ABI.

use ps5vm::VirtualMemory;

/// x86-64 general-purpose register indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CpuRegister {
    Rax = 0,
    Rcx = 1,
    Rdx = 2,
    Rbx = 3,
    Rsp = 4,
    Rbp = 5,
    Rsi = 6,
    Rdi = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,
}

/// Snapshot of guest CPU state at an HLE call boundary.
pub struct CpuContext<'a> {
    pub regs: [u64; 16],
    pub rip: u64,
    pub vm: &'a VirtualMemory,
}

impl<'a> CpuContext<'a> {
    pub fn new(vm: &'a VirtualMemory) -> Self {
        Self {
            regs: [0; 16],
            rip: 0,
            vm,
        }
    }

    #[inline]
    pub fn get(&self, reg: CpuRegister) -> u64 {
        self.regs[reg as usize]
    }

    #[inline]
    pub fn set(&mut self, reg: CpuRegister, val: u64) {
        self.regs[reg as usize] = val;
    }

    #[inline]
    pub fn rdi(&self) -> u64 {
        self.get(CpuRegister::Rdi)
    }
    #[inline]
    pub fn rsi(&self) -> u64 {
        self.get(CpuRegister::Rsi)
    }
    #[inline]
    pub fn rdx(&self) -> u64 {
        self.get(CpuRegister::Rdx)
    }
    #[inline]
    pub fn rcx(&self) -> u64 {
        self.get(CpuRegister::Rcx)
    }
    #[inline]
    pub fn r8(&self) -> u64 {
        self.get(CpuRegister::R8)
    }
    #[inline]
    pub fn r9(&self) -> u64 {
        self.get(CpuRegister::R9)
    }

    /// Also return `value` in RDX: a 128-bit result is RAX (low) : RDX
    /// (high); the callback's own return value is RAX.
    pub fn set_return_rdx(&mut self, value: u64) {
        self.regs[CpuRegister::Rdx as usize] = value;
        ps5cpu::exec::set_return_rdx(value);
    }

    /// Return a `double` in XMM0.
    pub fn set_return_f64(&mut self, value: f64) {
        ps5cpu::exec::set_return_xmm0(value.to_bits() as u128);
    }

    /// Return a `float` in XMM0.
    pub fn set_return_f32(&mut self, value: f32) {
        ps5cpu::exec::set_return_xmm0(value.to_bits() as u128);
    }

    /// Return a `long double` (x87 80-bit extended) in ST0.
    pub fn set_return_st0(&mut self, value: [u8; 10]) {
        ps5cpu::exec::set_return_st0(value);
    }

    /// Read a u64 from guest memory.
    pub fn read_u64(&self, addr: u64) -> Option<u64> {
        let mut buf = [0u8; 8];
        self.vm.read(addr, &mut buf).ok()?;
        Some(u64::from_le_bytes(buf))
    }

    /// Write a u64 to guest memory.
    pub fn write_u64(&self, addr: u64, val: u64) -> bool {
        self.vm.write(addr, &val.to_le_bytes()).is_ok()
    }

    /// Read a u32 from guest memory.
    pub fn read_u32(&self, addr: u64) -> Option<u32> {
        let mut buf = [0u8; 4];
        self.vm.read(addr, &mut buf).ok()?;
        Some(u32::from_le_bytes(buf))
    }

    /// Write a u32 to guest memory.
    pub fn write_u32(&self, addr: u64, val: u32) -> bool {
        self.vm.write(addr, &val.to_le_bytes()).is_ok()
    }

    /// Read a null-terminated UTF-8 string from guest memory.
    pub fn read_cstring(&self, addr: u64, max_len: usize) -> Option<String> {
        String::from_utf8(self.read_cstr_bytes(addr, max_len)?).ok()
    }

    /// Read the bytes of a null-terminated string (without the terminator),
    /// at most `max_len` of them. Reads never cross a page boundary, so a string
    /// that ends just before an unmapped page (top of a stack, end of a
    /// heap block) is still read.
    pub fn read_cstr_bytes(&self, addr: u64, max_len: usize) -> Option<Vec<u8>> {
        if addr == 0 {
            return None;
        }
        let mut out = Vec::new();
        let mut cur = addr;
        let mut chunk = [0u8; 256];
        while out.len() < max_len {
            // Chunks never cross a page boundary.
            let page_left = (0x1000 - (cur & 0xFFF)) as usize;
            let n = page_left.min(chunk.len()).min(max_len - out.len());
            let buf = &mut chunk[..n];
            self.vm.read(cur, buf).ok()?;
            if let Some(p) = buf.iter().position(|&b| b == 0) {
                out.extend_from_slice(&buf[..p]);
                return Some(out);
            }
            out.extend_from_slice(buf);
            cur += n as u64;
        }
        Some(out)
    }

    /// Read `len` bytes from guest memory into a Vec.
    pub fn read_bytes(&self, addr: u64, len: usize) -> Option<Vec<u8>> {
        if addr == 0 || len == 0 {
            return Some(Vec::new());
        }
        let mut buf = vec![0u8; len];
        self.vm.read(addr, &mut buf).ok()?;
        Some(buf)
    }

    /// Write bytes to guest memory.
    pub fn write_bytes(&self, addr: u64, data: &[u8]) -> bool {
        if addr == 0 || data.is_empty() {
            return true;
        }
        self.vm.write(addr, data).is_ok()
    }

    /// Read a null-terminated wide (UTF-16LE) string from guest memory as
    /// a Vec of u16 units (without the terminator).
    pub fn read_wide_string(&self, addr: u64, max_units: usize) -> Option<Vec<u16>> {
        if addr == 0 {
            return None;
        }
        let mut units = Vec::new();
        let mut off = 0u64;
        while units.len() < max_units {
            let mut b = [0u8; 2];
            self.vm.read(addr + off, &mut b).ok()?;
            let u = u16::from_le_bytes(b);
            if u == 0 {
                break;
            }
            units.push(u);
            off += 2;
        }
        Some(units)
    }

    /// Write an f32 to guest memory.
    pub fn write_f32(&self, addr: u64, val: f32) -> bool {
        self.vm.write(addr, &val.to_le_bytes()).is_ok()
    }

    /// Read an f32 from guest memory.
    pub fn read_f32(&self, addr: u64) -> Option<f32> {
        let mut buf = [0u8; 4];
        self.vm.read(addr, &mut buf).ok()?;
        Some(f32::from_le_bytes(buf))
    }
}
