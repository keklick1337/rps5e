// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Import stub table.
//!
//! Every imported function NID gets one 16-byte stub slot in a dedicated
//! region at `STUB_BASE`. A stub is one of:
//!
//!   * HLE (default): `lea r11, [rip-7]; jmp [entry]` — a direct call into
//!     the host import entry (`ps5cpu::exec::import_entry_address`), which
//!     switches to the host stack and dispatches by stub address. No signal
//!     is involved; a call costs tens of nanoseconds.
//!   * HLE through a trap (`RPS5E_IMPORT_TRAP=1`): INT3 bytes; the SIGTRAP
//!     handler dispatches. Kept for comparison and bisecting.
//!   * native (LLE): `jmp [rip+0]; target` into a module's own export.
//!
//! Slot 0 of the region holds the 64-bit address of the import entry that
//! the HLE stubs jump through. All slots are INT3-padded, so reading a slot
//! as a pointer gives a non-zero value.

use ps5vm::{MemProtect, VirtualMemory};
use std::collections::{HashMap, HashSet};

pub const STUB_BASE: u64 = 0x0000_7000_0000_0000;
const STUB_SLOT_SIZE: u64 = 0x10;
const REGION_GROW: u64 = 0x4000;
/// Capacity of the stub region (slots), including the entry-pointer slot.
pub const MAX_SLOTS: usize = 0x1_0000;

/// Slot number of a stub address (None outside the stub region).
pub fn slot_index(addr: u64) -> Option<usize> {
    let off = addr.checked_sub(STUB_BASE)?;
    let i = (off / STUB_SLOT_SIZE) as usize;
    (i < MAX_SLOTS).then_some(i)
}

fn trap_stubs() -> bool {
    std::env::var("RPS5E_IMPORT_TRAP").is_ok_and(|v| v == "1")
}

/// Maps NID strings to stub guest addresses and manages the stub memory
/// region. Created empty during loading; stubs are assigned on demand.
pub struct StubTable {
    nid_to_addr: HashMap<String, u64>,
    next_base: u64,
    region_end: u64,
    trap: bool,
    /// NIDs with a real HLE implementation (weak imports bind only to these
    /// or to a module export).
    hle_functions: HashSet<String>,
}

impl Default for StubTable {
    fn default() -> Self {
        Self::new()
    }
}

impl StubTable {
    pub fn new() -> Self {
        Self {
            nid_to_addr: HashMap::new(),
            next_base: STUB_BASE + STUB_SLOT_SIZE,
            region_end: STUB_BASE,
            trap: trap_stubs(),
            hle_functions: HashSet::new(),
        }
    }

    /// Record the NIDs that have a real HLE implementation.
    pub fn set_hle_functions(&mut self, nids: impl IntoIterator<Item = String>) {
        self.hle_functions = nids.into_iter().collect();
    }

    /// Whether `nid` has a real HLE implementation.
    pub fn has_hle_function(&self, nid: &str) -> bool {
        self.hle_functions.contains(nid)
    }

    fn hle_stub(&self, addr: u64) -> [u8; 16] {
        if self.trap {
            [0xCC; 16]
        } else {
            ps5cpu::exec::import_stub_code(addr, STUB_BASE)
        }
    }

    /// Ensure a stub exists for `nid`, writing it into `vm` and returning its
    /// guest address. Subsequent calls for the same NID return the same addr.
    pub fn ensure_stub(&mut self, vm: &VirtualMemory, nid: &str) -> Result<u64, ps5vm::VmError> {
        if let Some(&addr) = self.nid_to_addr.get(nid) {
            return Ok(addr);
        }
        if self.next_base >= STUB_BASE + MAX_SLOTS as u64 * STUB_SLOT_SIZE {
            return Err(ps5vm::VmError::MapFailed(self.next_base, STUB_SLOT_SIZE, "import stub region full".into()));
        }
        // Grow the stub region one block at a time as needed.
        while self.next_base + STUB_SLOT_SIZE > self.region_end {
            vm.map(self.region_end, REGION_GROW, MemProtect::RWX)?;
            if self.region_end == STUB_BASE {
                let mut head = [0xCCu8; 16];
                head[..8].copy_from_slice(&ps5cpu::exec::import_entry_address().to_le_bytes());
                vm.write(STUB_BASE, &head)?;
            }
            self.region_end += REGION_GROW;
        }

        let addr = self.next_base;
        vm.write(addr, &self.hle_stub(addr))?;
        self.nid_to_addr.insert(nid.to_string(), addr);
        self.next_base += STUB_SLOT_SIZE;
        Ok(addr)
    }

    /// Turn the stub for `nid` into a direct bridge to native guest code:
    /// `jmp qword [rip+0]` followed by the 64-bit target (14 bytes, fits
    /// the 16-byte slot). Calls through the GOT/PLT then land in the
    /// module's own implementation without entering the host.
    pub fn redirect(&self, vm: &VirtualMemory, nid: &str, target: u64) -> Result<u64, ps5vm::VmError> {
        let Some(&addr) = self.nid_to_addr.get(nid) else {
            return Err(ps5vm::VmError::NotMapped(0));
        };
        let mut slot = [0xCCu8; STUB_SLOT_SIZE as usize];
        slot[..6].copy_from_slice(&[0xFF, 0x25, 0, 0, 0, 0]);
        slot[6..14].copy_from_slice(&target.to_le_bytes());
        vm.write(addr, &slot)?;
        Ok(addr)
    }

    /// Restore the HLE stub for `nid` (its native target was unloaded).
    pub fn unredirect(&self, vm: &VirtualMemory, nid: &str) -> Result<u64, ps5vm::VmError> {
        let Some(&addr) = self.nid_to_addr.get(nid) else {
            return Err(ps5vm::VmError::NotMapped(0));
        };
        vm.write(addr, &self.hle_stub(addr))?;
        Ok(addr)
    }

    /// Look up a stub address for an already-registered NID.
    pub fn lookup(&self, nid: &str) -> Option<u64> {
        self.nid_to_addr.get(nid).copied()
    }

    pub fn count(&self) -> usize {
        self.nid_to_addr.len()
    }

    /// Build a reverse map (stub address → NID string) for the dispatcher.
    pub fn addr_to_nid_map(&self) -> HashMap<u64, String> {
        self.nid_to_addr.iter().map(|(nid, &addr)| (addr, nid.clone())).collect()
    }

    /// Iterate over (NID, stub_address) pairs.
    pub fn nid_to_addr_iter(&self) -> impl Iterator<Item = (&String, &u64)> {
        self.nid_to_addr.iter()
    }
}
