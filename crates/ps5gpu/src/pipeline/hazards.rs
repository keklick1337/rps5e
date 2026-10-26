// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Memory hazards the scheduler cannot see as ranges.
//!
//! Shader work orders itself against earlier work by the guest ranges it
//! touches: a draw or dispatch reading a range an earlier dispatch wrote
//! without a barrier after it waits for that dispatch (`shader_write_overlaps`).
//! Accesses through device addresses that the plan could not bound, and
//! stages reading their buffers straight from guest memory, have no ranges:
//! such a stage may read anything earlier unordered shader work wrote, so it
//! waits for all of it (an unknown read). A stage storing through device
//! addresses may write anything (an unknown write): it is ordered against
//! everything before and after it.

use ps5shader::resources::{DeviceAddressRange, ShaderResourceInfo};

/// `RPS5E_UNKNOWN_HAZARDS=0`: stages reading or writing memory the
/// scheduler has no ranges for are not ordered against earlier unordered
/// shader writes (the behaviour before the hazards were tracked).
pub fn unknown_hazards_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_UNKNOWN_HAZARDS").is_ok_and(|v| v == "0"))
}

/// The stage reads memory outside the ranges it reserves: its buffers live
/// in guest memory (no buffer is bound), or it accesses device addresses
/// with a range whose base or extent was not evaluated for this draw.
pub fn reads_unranged_memory(info: &ShaderResourceInfo, ranges: &[DeviceAddressRange]) -> bool {
    if info.buffers_in_guest_memory && !info.buffers.is_empty() {
        return true;
    }
    info.uses_device_addresses && (ranges.is_empty() || ranges.iter().any(|r| !r.planned || r.size == 0))
}

/// The stage stores to memory outside the ranges it reserves (through a
/// device address whose range was not evaluated).
pub fn writes_unranged_memory(writes_device_memory: bool, ranges: &[DeviceAddressRange]) -> bool {
    writes_device_memory && (ranges.is_empty() || ranges.iter().any(|r| r.written && (!r.planned || r.size == 0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ps5shader::resources::BufferResource;

    fn range(planned: bool, size: u64, written: bool) -> DeviceAddressRange {
        DeviceAddressRange { handle: 0, base: if planned { 0x1000 } else { 0 }, size, planned, written }
    }

    #[test]
    fn ranged_reads_are_not_unknown() {
        let mut info = ShaderResourceInfo::default();
        assert!(!reads_unranged_memory(&info, &[]), "no device addresses, bound buffers");
        info.uses_device_addresses = true;
        assert!(!reads_unranged_memory(&info, &[range(true, 64, false)]), "every device range evaluated");
        assert!(reads_unranged_memory(&info, &[range(true, 64, false), range(false, 0, false)]), "a range without an evaluated base");
        assert!(reads_unranged_memory(&info, &[range(true, 0, false)]), "a range without an extent");
        assert!(reads_unranged_memory(&info, &[]), "device accesses with no range at all");
    }

    #[test]
    fn guest_memory_buffers_are_unknown_reads() {
        let mut info = ShaderResourceInfo { buffers_in_guest_memory: true, ..ShaderResourceInfo::default() };
        assert!(!reads_unranged_memory(&info, &[]), "no buffer to read");
        info.buffers.push(BufferResource::new(0, 0));
        assert!(reads_unranged_memory(&info, &[]), "buffers read through their V# addresses");
    }

    #[test]
    fn unranged_writes() {
        assert!(!writes_unranged_memory(false, &[range(false, 0, true)]), "the program stores through no device address");
        assert!(!writes_unranged_memory(true, &[range(true, 64, true), range(false, 0, false)]), "the written range is evaluated");
        assert!(writes_unranged_memory(true, &[range(false, 0, true)]), "a written range without a base");
        assert!(writes_unranged_memory(true, &[]), "stores with no range");
    }
}
