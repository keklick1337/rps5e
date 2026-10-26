// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Static survey of guest code for instructions whose presence or result
//! depends on the host CPU (`rps5e-cli isa-scan`).
//!
//! The executable segments are decoded linearly. Compilers keep jump tables
//! and constants out of the text segment, so the sweep stays in step with
//! the real instruction stream; a stray hit inside padding shows up as a
//! sample address that can be checked in a disassembler.

use iced_x86::{CpuidFeature, Decoder, DecoderOptions, Instruction, Mnemonic};
use std::collections::BTreeMap;

/// Extensions of the console CPU that no Intel CPU implements.
pub const AMD_ONLY: &[CpuidFeature] = &[
    CpuidFeature::SSE4A,
    CpuidFeature::XOP,
    CpuidFeature::TBM,
    CpuidFeature::FMA4,
    CpuidFeature::CLZERO,
    CpuidFeature::RDPRU,
    CpuidFeature::MONITORX,
    CpuidFeature::D3NOW,
    CpuidFeature::SKINIT,
    CpuidFeature::SVM,
    CpuidFeature::INVLPGB,
    CpuidFeature::MCOMMIT,
];

/// Extensions of the console CPU that only some Intel generations have
/// (missing on Skylake-class desktop parts or on older cores).
pub const INTEL_GENERATION: &[CpuidFeature] = &[
    CpuidFeature::SHA,
    CpuidFeature::RDPID,
    CpuidFeature::CLWB,
    CpuidFeature::CLFLUSHOPT,
    CpuidFeature::ADX,
    CpuidFeature::RDSEED,
    CpuidFeature::AVX2,
    CpuidFeature::FMA,
    CpuidFeature::BMI1,
    CpuidFeature::BMI2,
    CpuidFeature::LZCNT,
    CpuidFeature::MOVBE,
    CpuidFeature::F16C,
    CpuidFeature::XSAVEC,
    CpuidFeature::XSAVES,
    CpuidFeature::PREFETCHW,
    CpuidFeature::WBNOINVD,
];

/// Instructions counted by name: those that run everywhere but return
/// host-specific values, and the console-only ones the emulator handles.
fn host_dependent(m: Mnemonic) -> bool {
    use Mnemonic::*;
    matches!(
        m,
        Cpuid | Rdtsc | Rdtscp | Rdpid | Xgetbv | Rsqrtps | Rsqrtss | Rcpps | Rcpss | Vrsqrtps | Vrsqrtss | Vrcpps | Vrcpss | Rdrand | Rdseed | Rdpmc
            | Extrq | Insertq | Movntss | Movntsd | Monitorx | Mwaitx | Adcx | Adox
    )
}

#[derive(Default, Debug)]
pub struct IsaReport {
    /// Decoded instructions.
    pub instructions: u64,
    /// Instructions per CPUID feature they need.
    pub features: BTreeMap<String, u64>,
    /// Host-dependent instructions by mnemonic.
    pub host_dependent: BTreeMap<String, u64>,
    /// Up to four addresses per AMD-only or Intel-generation feature and
    /// per host-dependent mnemonic.
    pub samples: BTreeMap<String, Vec<u64>>,
}

impl IsaReport {
    fn sample(&mut self, key: &str, ip: u64) {
        let v = self.samples.entry(key.to_string()).or_default();
        if v.len() < 4 {
            v.push(ip);
        }
    }

    pub fn merge(&mut self, other: &IsaReport) {
        self.instructions += other.instructions;
        for (k, v) in &other.features {
            *self.features.entry(k.clone()).or_default() += v;
        }
        for (k, v) in &other.host_dependent {
            *self.host_dependent.entry(k.clone()).or_default() += v;
        }
    }

    /// Instructions needing any of `features`.
    pub fn count_of(&self, features: &[CpuidFeature]) -> u64 {
        features.iter().map(|f| self.features.get(&format!("{f:?}")).copied().unwrap_or(0)).sum()
    }
}

/// Decode `code` (loaded at `ip`) and add its instructions to `report`.
pub fn scan(code: &[u8], ip: u64, report: &mut IsaReport) {
    let mut decoder = Decoder::with_ip(64, code, ip, DecoderOptions::NONE);
    let mut insn = Instruction::default();
    while decoder.can_decode() {
        decoder.decode_out(&mut insn);
        if insn.is_invalid() {
            continue;
        }
        report.instructions += 1;
        for &f in insn.cpuid_features() {
            if matches!(f, CpuidFeature::INTEL8086 | CpuidFeature::X64 | CpuidFeature::SSE | CpuidFeature::SSE2) {
                continue;
            }
            let key = format!("{f:?}");
            *report.features.entry(key.clone()).or_default() += 1;
            if AMD_ONLY.contains(&f) || INTEL_GENERATION.contains(&f) {
                report.sample(&key, insn.ip());
            }
        }
        if host_dependent(insn.mnemonic()) {
            let key = format!("{:?}", insn.mnemonic()).to_uppercase();
            *report.host_dependent.entry(key.clone()).or_default() += 1;
            report.sample(&key, insn.ip());
        }
    }
}

/// Executable PT_LOAD segments of a 64-bit little-endian ELF image as
/// (virtual address, file bytes).
pub fn exec_segments(elf: &[u8]) -> Vec<(u64, &[u8])> {
    let rd16 = |o: usize| elf.get(o..o + 2).map(|b| u16::from_le_bytes(b.try_into().unwrap())).unwrap_or(0) as usize;
    let rd32 = |o: usize| elf.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let rd64 = |o: usize| elf.get(o..o + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    if elf.len() < 64 || &elf[..4] != b"\x7fELF" {
        return Vec::new();
    }
    let (phoff, phentsize, phnum) = (rd64(0x20) as usize, rd16(0x36), rd16(0x38));
    let mut out = Vec::new();
    for i in 0..phnum {
        let ph = phoff + i * phentsize;
        let (ty, flags) = (rd32(ph), rd32(ph + 4));
        let (offset, vaddr, filesz) = (rd64(ph + 8) as usize, rd64(ph + 0x10), rd64(ph + 0x20) as usize);
        // PT_LOAD and PT_SCE_RELRO, executable.
        if (ty == 1 || ty == 0x6100_0010) && flags & 1 != 0 {
            if let Some(bytes) = elf.get(offset..offset.saturating_add(filesz)) {
                out.push((vaddr, bytes));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_amd_only_and_host_dependent() {
        #[rustfmt::skip]
        let code = [
            0x66, 0x0F, 0x78, 0xC1, 0x08, 0x04, // extrq xmm1, 8, 4
            0x0F, 0xA2,                         // cpuid
            0x0F, 0x01, 0xFC,                   // clzero
            0xF3, 0x0F, 0xBD, 0xC1,             // lzcnt eax, ecx
            0x0F, 0x53, 0xC1,                   // rcpps xmm0, xmm1
            0x90,
        ];
        let mut r = IsaReport::default();
        scan(&code, 0x1000, &mut r);
        assert_eq!(r.instructions, 6);
        assert_eq!(r.count_of(AMD_ONLY), 2);
        assert_eq!(r.features["SSE4A"], 1);
        assert_eq!(r.features["CLZERO"], 1);
        assert_eq!(r.features["LZCNT"], 1);
        assert_eq!(r.host_dependent["CPUID"], 1);
        assert_eq!(r.host_dependent["RCPPS"], 1);
        assert_eq!(r.samples["SSE4A"], vec![0x1000]);
    }
}
