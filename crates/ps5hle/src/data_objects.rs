// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Data objects (STT_OBJECT exports) of the HLE system libraries.
//!
//! A game imports these by NID like functions, but reads them as memory:
//! the loader (`ps5core`) gives each one real storage with the initial bytes
//! listed here and binds the game's data relocations to it. An object a
//! loaded module also exports binds to that module instead (`fallback`).

/// One data object an HLE library exports.
#[derive(Clone, Copy, Debug)]
pub struct DataObject {
    /// Symbol name; its NID is SHA1(name + salt).
    pub name: &'static str,
    /// Exporting library (for diagnostics).
    pub library: &'static str,
    /// Initial contents; the storage is exactly this long.
    pub bytes: &'static [u8],
    /// Alignment of the storage in bytes.
    pub align: u64,
}

/// `struct in6_addr` with the last byte `last` and the first two bytes
/// `prefix` (network order); every other byte is zero.
const fn in6(prefix: [u8; 2], last: u8) -> [u8; 16] {
    let mut a = [0u8; 16];
    a[0] = prefix[0];
    a[1] = prefix[1];
    a[15] = last;
    a
}

const IN6_ANY: [u8; 16] = in6([0, 0], 0);
const IN6_LOOPBACK: [u8; 16] = in6([0, 0], 1);
const IN6_NODELOCAL_ALLNODES: [u8; 16] = in6([0xff, 0x01], 1);
const IN6_LINKLOCAL_ALLNODES: [u8; 16] = in6([0xff, 0x02], 1);
const IN6_LINKLOCAL_ALLROUTERS: [u8; 16] = in6([0xff, 0x02], 2);

/// libSceNet exports the BSD IPv6 well-known addresses under both their
/// POSIX names and the `sce_net_` names (SceNetIn6Addr, same layout).
static OBJECTS: &[DataObject] = &[
    DataObject { name: "in6addr_any", library: "libSceNet", bytes: &IN6_ANY, align: 16 },
    DataObject { name: "in6addr_loopback", library: "libSceNet", bytes: &IN6_LOOPBACK, align: 16 },
    DataObject { name: "in6addr_nodelocal_allnodes", library: "libSceNet", bytes: &IN6_NODELOCAL_ALLNODES, align: 16 },
    DataObject { name: "in6addr_linklocal_allnodes", library: "libSceNet", bytes: &IN6_LINKLOCAL_ALLNODES, align: 16 },
    DataObject { name: "sce_net_in6addr_any", library: "libSceNet", bytes: &IN6_ANY, align: 16 },
    DataObject { name: "sce_net_in6addr_loopback", library: "libSceNet", bytes: &IN6_LOOPBACK, align: 16 },
    DataObject { name: "sce_net_in6addr_nodelocal_allnodes", library: "libSceNet", bytes: &IN6_NODELOCAL_ALLNODES, align: 16 },
    DataObject { name: "sce_net_in6addr_linklocal_allnodes", library: "libSceNet", bytes: &IN6_LINKLOCAL_ALLNODES, align: 16 },
    DataObject { name: "sce_net_in6addr_linklocal_allrouters", library: "libSceNet", bytes: &IN6_LINKLOCAL_ALLROUTERS, align: 16 },
];

/// Every HLE data object.
pub fn objects() -> &'static [DataObject] {
    OBJECTS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find(name: &str) -> &'static DataObject {
        objects().iter().find(|o| o.name == name).unwrap()
    }

    #[test]
    fn ipv6_well_known_addresses() {
        assert_eq!(find("in6addr_any").bytes, &[0u8; 16]);
        let lo = find("in6addr_loopback").bytes;
        assert_eq!(lo[..15], [0u8; 15]);
        assert_eq!(lo[15], 1);
        assert_eq!(find("in6addr_linklocal_allnodes").bytes[..2], [0xff, 0x02]);
        assert_eq!(find("sce_net_in6addr_linklocal_allrouters").bytes[15], 2);
        assert_eq!(find("in6addr_nodelocal_allnodes").bytes[..2], [0xff, 0x01]);
    }

    #[test]
    fn names_are_unique_and_nids_known() {
        let mut nids: Vec<String> = objects().iter().map(|o| ps5nid::compute_nid(o.name)).collect();
        // NIDs as the game imports them.
        assert!(nids.iter().any(|n| n == "ZRAJo-A-ukc"));
        assert!(nids.iter().any(|n| n == "XCuA-GqjA-k"));
        let len = nids.len();
        nids.sort();
        nids.dedup();
        assert_eq!(nids.len(), len);
        for o in objects() {
            assert!(o.align.is_power_of_two() && !o.bytes.is_empty());
        }
    }
}
