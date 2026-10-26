// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Controller-to-player assignment (pure logic, re-run on every hot-plug).
//!
//! A player's selector is empty (any free controller), a serial number
//! (exact match) or a name (case-insensitive substring). Existing
//! assignments are kept while the device stays connected and still
//! matches, so plugging in another controller never steals a player's pad.

/// A connected controller as seen by the assignment.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeviceKey {
    pub id: u32,
    pub name: String,
    pub serial: String,
}

/// Whether `dev` satisfies `selector`.
pub fn matches(selector: &str, dev: &DeviceKey) -> bool {
    let sel = selector.trim();
    if sel.is_empty() {
        return true;
    }
    if !dev.serial.is_empty() && dev.serial.eq_ignore_ascii_case(sel) {
        return true;
    }
    dev.name.to_ascii_lowercase().contains(&sel.to_ascii_lowercase())
}

/// Assign devices (in connection order) to `players` = (player index,
/// selector). `previous[p]` is the device id player `p` had. Returns the
/// device id per player index (length `slots`).
pub fn assign(players: &[(usize, &str)], devices: &[DeviceKey], previous: &[Option<u32>], slots: usize) -> Vec<Option<u32>> {
    let mut out = vec![None; slots];
    let mut taken: Vec<u32> = Vec::new();
    // 1. Keep assignments that are still valid.
    for &(p, sel) in players {
        if let Some(Some(id)) = previous.get(p) {
            if let Some(d) = devices.iter().find(|d| d.id == *id) {
                if matches(sel, d) && !taken.contains(id) {
                    out[p] = Some(*id);
                    taken.push(*id);
                }
            }
        }
    }
    // 2. Specific selectors first, then "any controller".
    for specific in [true, false] {
        for &(p, sel) in players {
            if out[p].is_some() || sel.trim().is_empty() == specific {
                continue;
            }
            if let Some(d) = devices.iter().find(|d| !taken.contains(&d.id) && matches(sel, d)) {
                out[p] = Some(d.id);
                taken.push(d.id);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(id: u32, name: &str, serial: &str) -> DeviceKey {
        DeviceKey { id, name: name.into(), serial: serial.into() }
    }

    #[test]
    fn selector_matching() {
        let d = dev(1, "DualSense Wireless Controller", "aa:bb:cc");
        assert!(matches("", &d));
        assert!(matches("dualsense", &d));
        assert!(matches("AA:BB:CC", &d));
        assert!(!matches("Xbox", &d));
    }

    #[test]
    fn first_free_and_specific() {
        let devs = [dev(10, "Xbox Series X Controller", ""), dev(11, "DualSense Wireless Controller", "s1")];
        // Player 1 wants a DualSense, player 2 anything: the Xbox pad.
        let a = assign(&[(0, "DualSense"), (1, "")], &devs, &[], 4);
        assert_eq!(a, vec![Some(11), Some(10), None, None]);
        // Both "any": connection order.
        let a = assign(&[(0, ""), (1, "")], &devs, &[], 2);
        assert_eq!(a, vec![Some(10), Some(11)]);
        // More players than devices.
        let a = assign(&[(0, ""), (1, ""), (2, "")], &devs, &[], 3);
        assert_eq!(a[2], None);
    }

    #[test]
    fn hot_plug_keeps_existing_assignment() {
        let mut devs = vec![dev(11, "DualSense Wireless Controller", "s1")];
        let a = assign(&[(0, ""), (1, "")], &devs, &[], 2);
        assert_eq!(a, vec![Some(11), None]);
        // A new controller connected before the old one in the list.
        devs.insert(0, dev(12, "Xbox Controller", ""));
        let b = assign(&[(0, ""), (1, "")], &devs, &a, 2);
        assert_eq!(b, vec![Some(11), Some(12)]);
        // Player 1's controller unplugged: it takes the free one, player 2 has none.
        devs.retain(|d| d.id != 11);
        let c = assign(&[(0, ""), (1, "")], &devs, &b, 2);
        assert_eq!(c, vec![None, Some(12)]);
    }

    #[test]
    fn selected_controller_among_identical_ones() {
        let devs = [dev(1, "DualSense Wireless Controller", "s1"), dev(2, "DualSense Wireless Controller", "s2")];
        let a = assign(&[(0, "s2"), (1, "")], &devs, &[], 2);
        assert_eq!(a, vec![Some(2), Some(1)]);
    }
}
