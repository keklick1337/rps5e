// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Host CPU features against the console CPU (AMD Zen 2).
//!
//! Guest code runs natively and was compiled for Zen 2 without run-time
//! feature checks, so every extension it uses must either exist on the host
//! or be emulated:
//!
//!   * emulated (load-time patch or SIGILL, see `code_patch` / `x64emu`):
//!     SSE4a, MONITORX/MWAITX, SHA, RDPID, CLWB, ADX;
//!   * required: the x86-64-v3 set (AVX, AVX2, FMA, BMI1, BMI2, LZCNT,
//!     MOVBE, F16C) plus POPCNT, SSE4.2, AES and PCLMULQDQ. Without LZCNT
//!     and BMI1 the `lzcnt`/`tzcnt` encodings silently execute as BSR/BSF,
//!     so a host without them cannot run guest code correctly at all.
//!
//! Guest `cpuid` is not intercepted: it reports the host, which keeps the
//! guest's own feature checks truthful (a check for SSE4a on an Intel host
//! fails and the guest takes its fallback path).

use std::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};
use std::sync::OnceLock;

/// One CPU feature: name, present on the host, emulated when absent.
#[derive(Clone, Copy, Debug)]
pub struct Feature {
    pub name: &'static str,
    pub present: bool,
    pub emulated: bool,
}

#[derive(Clone, Debug)]
pub struct HostCpu {
    pub vendor: String,
    pub brand: String,
    pub features: Vec<Feature>,
}

impl HostCpu {
    pub fn has(&self, name: &str) -> bool {
        self.features.iter().any(|f| f.name == name && f.present)
    }

    pub fn is_intel(&self) -> bool {
        self.vendor == "GenuineIntel"
    }

    /// Features guest code needs that the host lacks and nothing emulates.
    pub fn missing_required(&self) -> Vec<&'static str> {
        self.features.iter().filter(|f| !f.present && !f.emulated).map(|f| f.name).collect()
    }

    /// Console features the host lacks that are emulated.
    pub fn emulated(&self) -> Vec<&'static str> {
        self.features.iter().filter(|f| !f.present && f.emulated).map(|f| f.name).collect()
    }
}

fn bit(v: u32, n: u32) -> bool {
    v >> n & 1 != 0
}

fn cpuid_string(regs: &[u32]) -> String {
    let bytes: Vec<u8> = regs.iter().flat_map(|r| r.to_le_bytes()).collect();
    String::from_utf8_lossy(&bytes).trim_matches(char::from(0)).trim().to_string()
}

fn detect() -> HostCpu {
    // SAFETY: CPUID exists on every x86-64 CPU; leaves are queried only up
    // to the reported maximum.
    unsafe {
        let l0 = __cpuid(0);
        let max = l0.eax;
        let vendor = cpuid_string(&[l0.ebx, l0.edx, l0.ecx]);
        let l1 = __cpuid(1);
        let l7 = if max >= 7 { __cpuid_count(7, 0) } else { std::mem::zeroed() };
        let max_ext = __cpuid(0x8000_0000).eax;
        let e1 = if max_ext >= 0x8000_0001 { __cpuid(0x8000_0001) } else { std::mem::zeroed() };
        let brand = if max_ext >= 0x8000_0004 {
            let r: Vec<u32> = (0x8000_0002u32..=0x8000_0004).flat_map(|l| {
                let c = __cpuid(l);
                [c.eax, c.ebx, c.ecx, c.edx]
            }).collect();
            cpuid_string(&r)
        } else {
            String::new()
        };
        // AVX state enabled by the OS (XCR0 SSE and AVX components).
        let os_avx = bit(l1.ecx, 27) && _xgetbv(0) & 6 == 6;
        let f = |name, present| Feature { name, present, emulated: false };
        let e = |name, present| Feature { name, present, emulated: true };
        let features = vec![
            f("SSSE3", bit(l1.ecx, 9)),
            f("SSE4.1", bit(l1.ecx, 19)),
            f("SSE4.2", bit(l1.ecx, 20)),
            f("POPCNT", bit(l1.ecx, 23)),
            f("AES", bit(l1.ecx, 25)),
            f("PCLMULQDQ", bit(l1.ecx, 1)),
            f("AVX", bit(l1.ecx, 28) && os_avx),
            f("F16C", bit(l1.ecx, 29) && os_avx),
            f("FMA", bit(l1.ecx, 12) && os_avx),
            f("MOVBE", bit(l1.ecx, 22)),
            f("AVX2", bit(l7.ebx, 5) && os_avx),
            f("BMI1", bit(l7.ebx, 3)),
            f("BMI2", bit(l7.ebx, 8)),
            f("LZCNT", bit(e1.ecx, 5)),
            e("ADX", bit(l7.ebx, 19)),
            e("SHA", bit(l7.ebx, 29)),
            e("CLWB", bit(l7.ebx, 24)),
            e("RDPID", bit(l7.ecx, 22)),
            e("SSE4A", bit(e1.ecx, 6)),
            e("MONITORX", bit(e1.ecx, 29)),
        ];
        HostCpu { vendor, brand, features }
    }
}

/// The host CPU (detected once).
pub fn host() -> &'static HostCpu {
    static HOST: OnceLock<HostCpu> = OnceLock::new();
    HOST.get_or_init(detect)
}

/// Check the host before guest code runs: an error names the missing
/// features (`RPS5E_IGNORE_CPU_FEATURES=1` runs anyway).
pub fn check_guest_requirements() -> Result<(), String> {
    let h = host();
    log::info!("[cpu] host: {} ({})", h.brand, h.vendor);
    let emulated = h.emulated();
    if !emulated.is_empty() {
        log::info!("[cpu] emulating console extensions the host lacks: {}", emulated.join(", "));
    }
    let missing = h.missing_required();
    if missing.is_empty() {
        return Ok(());
    }
    let msg = format!(
        "the host CPU lacks {}: guest code compiled for the console's Zen 2 uses them without checks \
         (an x86-64-v3 CPU is required: Intel Haswell / AMD Zen or newer)",
        missing.join(", ")
    );
    if std::env::var("RPS5E_IGNORE_CPU_FEATURES").is_ok_and(|v| v == "1") {
        log::error!("[cpu] {msg}; continuing (RPS5E_IGNORE_CPU_FEATURES=1)");
        return Ok(());
    }
    Err(msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_this_host() {
        let h = host();
        assert!(!h.vendor.is_empty());
        // Every x86-64 CPU that runs the test suite has SSSE3 at least.
        assert!(h.has("SSSE3"));
        // The emulated set never blocks a run.
        for name in h.emulated() {
            assert!(!h.missing_required().contains(&name));
        }
    }

    #[test]
    fn missing_and_emulated_split() {
        let h = HostCpu {
            vendor: "GenuineIntel".into(),
            brand: String::new(),
            features: vec![
                Feature { name: "AVX2", present: true, emulated: false },
                Feature { name: "LZCNT", present: false, emulated: false },
                Feature { name: "SSE4A", present: false, emulated: true },
                Feature { name: "SHA", present: true, emulated: true },
            ],
        };
        assert_eq!(h.missing_required(), vec!["LZCNT"]);
        assert_eq!(h.emulated(), vec!["SSE4A"]);
        assert!(h.is_intel());
    }
}
