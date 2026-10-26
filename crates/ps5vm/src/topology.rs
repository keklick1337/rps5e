// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Host CPU topology: the fast cores for the GPU recording threads.
//!
//! * Hybrid processors (Intel P-cores and E-cores): the recorder runs on
//!   the performance cores. Linux lists them in `/sys/devices/cpu_core/cpus`
//!   (E-cores in `/sys/devices/cpu_atom/cpus`); elsewhere, CPUs whose
//!   `cpu_capacity` differ mark the hybrid layout, the fast ones having the
//!   largest capacity.
//! * Processors whose core complexes have different L3 sizes (a stacked
//!   cache on one of them): the GPU command recorder runs measurably faster
//!   on the complex with the large cache, as its working set (resource
//!   caches, memos, guest descriptor lines) does not fit the small one.
//!
//! When both apply, the fast cores are the performance cores of the
//! large-cache complex. On symmetric processors nothing is pinned.

use std::path::Path;
use std::sync::OnceLock;

/// Parse a sysfs CPU list ("0-7,16-23").
pub fn parse_cpu_list(s: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in s.trim().split(',').filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                    out.extend(a..=b);
                }
            }
            None => {
                if let Ok(a) = part.trim().parse() {
                    out.push(a);
                }
            }
        }
    }
    out
}

/// Parse a sysfs cache size ("98304K").
fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim();
    let (num, mul) = match s.chars().last()? {
        'K' => (&s[..s.len() - 1], 1u64 << 10),
        'M' => (&s[..s.len() - 1], 1 << 20),
        'G' => (&s[..s.len() - 1], 1 << 30),
        _ => (s, 1),
    };
    num.parse::<u64>().ok().map(|n| n * mul)
}

/// The CPUs sharing the largest L3 when L3 sizes differ between complexes,
/// from `(size, shared cpu list)` per L3 instance.
pub fn largest_of(caches: &[(u64, Vec<usize>)]) -> Option<Vec<usize>> {
    let max = caches.iter().map(|c| c.0).max()?;
    let min = caches.iter().map(|c| c.0).min()?;
    if max == min {
        return None;
    }
    let mut cpus: Vec<usize> = caches.iter().filter(|c| c.0 == max).flat_map(|c| c.1.iter().copied()).collect();
    cpus.sort_unstable();
    cpus.dedup();
    (!cpus.is_empty()).then_some(cpus)
}

/// The CPUs with the largest `cpu_capacity` when capacities differ, from
/// `(cpu, capacity)` pairs.
pub fn highest_capacity(capacities: &[(usize, u64)]) -> Option<Vec<usize>> {
    let max = capacities.iter().map(|c| c.1).max()?;
    let min = capacities.iter().map(|c| c.1).min()?;
    if max == min {
        return None;
    }
    let mut cpus: Vec<usize> = capacities.iter().filter(|c| c.1 == max).map(|c| c.0).collect();
    cpus.sort_unstable();
    Some(cpus)
}

/// What the host topology says about fast cores.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Topology {
    /// P-cores of a hybrid processor (`cpu_core` PMU), when E-cores exist.
    pub hybrid_core: Option<Vec<usize>>,
    /// Per-CPU capacity (`cpu_capacity`), where the kernel reports it.
    pub capacities: Vec<(usize, u64)>,
    /// `(size, shared cpu list)` per L3 instance.
    pub l3: Vec<(u64, Vec<usize>)>,
}

impl Topology {
    /// The performance cores of a hybrid processor.
    pub fn performance_cores(&self) -> Option<Vec<usize>> {
        self.hybrid_core.clone().filter(|c| !c.is_empty()).or_else(|| highest_capacity(&self.capacities))
    }

    /// The cores the recorder is pinned to and why: the performance cores,
    /// narrowed to the largest-L3 complex when that leaves any.
    pub fn fast_cpus(&self) -> Option<(Vec<usize>, &'static str)> {
        match (self.performance_cores(), largest_of(&self.l3)) {
            (Some(p), Some(l3)) => {
                let both: Vec<usize> = p.iter().copied().filter(|c| l3.contains(c)).collect();
                Some(if both.is_empty() { (p, "performance") } else { (both, "performance large-cache") })
            }
            (Some(p), None) => Some((p, "performance")),
            (None, Some(l3)) => Some((l3, "large-cache")),
            (None, None) => None,
        }
    }

    /// Read the topology under a sysfs root (`/sys` on the host).
    pub fn read(sys: &Path) -> Topology {
        let read = |p: &Path| std::fs::read_to_string(p).ok();
        // The P-core list means a hybrid processor only when the E-core PMU
        // has CPUs too (a non-hybrid Intel CPU has neither directory).
        let atom = read(&sys.join("devices/cpu_atom/cpus")).map(|s| parse_cpu_list(&s)).unwrap_or_default();
        let hybrid_core = if atom.is_empty() { None } else { read(&sys.join("devices/cpu_core/cpus")).map(|s| parse_cpu_list(&s)) };
        let mut capacities = Vec::new();
        let mut l3: Vec<(u64, Vec<usize>)> = Vec::new();
        if let Ok(dir) = std::fs::read_dir(sys.join("devices/system/cpu")) {
            for e in dir.flatten() {
                let name = e.file_name();
                let Some(n) = name.to_str().and_then(|n| n.strip_prefix("cpu")).and_then(|n| n.parse::<usize>().ok()) else { continue };
                if let Some(c) = read(&e.path().join("cpu_capacity")).and_then(|s| s.trim().parse::<u64>().ok()) {
                    capacities.push((n, c));
                }
                let base = e.path().join("cache/index3");
                let (Some(size), Some(list)) = (read(&base.join("size")), read(&base.join("shared_cpu_list"))) else { continue };
                let (Some(size), cpus) = (parse_size(&size), parse_cpu_list(&list)) else { continue };
                if !l3.iter().any(|c| c.1 == cpus) {
                    l3.push((size, cpus));
                }
            }
        }
        capacities.sort_unstable();
        Topology { hybrid_core, capacities, l3 }
    }
}

/// `RPS5E_PIN_RECORDER=0` turns the pinning off.
fn pinning_enabled() -> bool {
    !std::env::var("RPS5E_PIN_RECORDER").is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off"))
}

/// The fast cores of an asymmetric processor and why they are fast.
pub fn fast_cpus() -> Option<(&'static [usize], &'static str)> {
    static CPUS: OnceLock<Option<(Vec<usize>, &'static str)>> = OnceLock::new();
    CPUS.get_or_init(|| Topology::read(Path::new("/sys")).fast_cpus()).as_ref().map(|(c, why)| (c.as_slice(), *why))
}

/// Pin the calling thread to the fast cores: the P-cores of a hybrid
/// processor, the largest-L3 complex of an asymmetric one
/// (`RPS5E_PIN_RECORDER=0` disables). True when pinned.
pub fn pin_current_thread_to_fast_cores(what: &str) -> bool {
    if !pinning_enabled() {
        return false;
    }
    let Some((cpus, why)) = fast_cpus() else { return false };
    #[cfg(windows)]
    // SAFETY: affinity of the calling thread.
    let ok = unsafe {
        let mask = cpus.iter().filter(|&&c| c < 64).fold(0usize, |m, &c| m | 1 << c);
        mask != 0 && windows_sys::Win32::System::Threading::SetThreadAffinityMask(windows_sys::Win32::System::Threading::GetCurrentThread(), mask) != 0
    };
    #[cfg(unix)]
    // SAFETY: plain cpu_set_t manipulation and a syscall on this thread.
    let ok = unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for &c in cpus {
            if c < libc::CPU_SETSIZE as usize {
                libc::CPU_SET(c, &mut set);
            }
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    };
    if ok {
        log::info!("{what}: pinned to the {why} CPUs {cpus:?}");
    }
    ok
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_lists_and_sizes_parse() {
        assert_eq!(parse_cpu_list("0-3,8,10-11\n"), vec![0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(parse_size("98304K\n"), Some(98304 << 10));
        assert_eq!(parse_size("32M"), Some(32 << 20));
    }

    #[test]
    fn only_asymmetric_caches_pick_a_complex() {
        let x3d = [(96 << 20, vec![0, 1, 16, 17]), (32 << 20, vec![2, 3, 18, 19])];
        assert_eq!(largest_of(&x3d), Some(vec![0, 1, 16, 17]));
        let even = [(32 << 20, vec![0, 1]), (32 << 20, vec![2, 3])];
        assert_eq!(largest_of(&even), None);
        assert_eq!(largest_of(&[]), None);
    }

    #[test]
    fn hybrid_capacities_pick_the_big_cores() {
        let caps = [(0, 1024), (1, 1024), (2, 600), (3, 600)];
        assert_eq!(highest_capacity(&caps), Some(vec![0, 1]));
        assert_eq!(highest_capacity(&[(0, 1024), (1, 1024)]), None);
    }

    /// A synthetic sysfs tree in the repository's local/tmp/, read and removed.
    fn sysfs(name: &str, files: &[(String, &str)]) -> Topology {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../local/tmp/topology-tests").join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for (rel, text) in files {
            let p = root.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, text).unwrap();
        }
        let t = Topology::read(&root);
        std::fs::remove_dir_all(&root).unwrap();
        t
    }

    fn l3(cpu: usize, size: &str, shared: &str) -> [(String, &'static str); 2] {
        let base = format!("devices/system/cpu/cpu{cpu}/cache/index3");
        [(format!("{base}/size"), size.to_string().leak()), (format!("{base}/shared_cpu_list"), shared.to_string().leak())]
    }

    #[test]
    fn intel_hybrid_pins_to_p_cores() {
        // An i7-12700-like layout: 8 P-cores with SMT (0-15), 4 E-cores
        // (16-19), one shared L3.
        let mut files = vec![("devices/cpu_core/cpus".to_string(), "0-15\n"), ("devices/cpu_atom/cpus".to_string(), "16-19\n")];
        for c in 0..20 {
            files.extend(l3(c, "25600K\n", "0-19\n"));
        }
        let t = sysfs("hybrid", &files);
        assert_eq!(t.hybrid_core, Some((0..16).collect()));
        assert_eq!(t.fast_cpus(), Some(((0..16).collect(), "performance")));
    }

    #[test]
    fn non_hybrid_intel_and_x3d() {
        // cpu_core alone (no E-cores) is not hybrid.
        let mut files = vec![("devices/cpu_core/cpus".to_string(), "0-7\n")];
        files.extend(l3(0, "16384K", "0-7"));
        let t = sysfs("plain", &files);
        assert_eq!(t.hybrid_core, None);
        assert_eq!(t.fast_cpus(), None);
        // Two CCDs, the first with stacked cache.
        let mut files = Vec::new();
        files.extend(l3(0, "98304K", "0-7,16-23"));
        files.extend(l3(8, "32768K", "8-15,24-31"));
        let t = sysfs("x3d", &files);
        let ccd0: Vec<usize> = (0..8).chain(16..24).collect();
        assert_eq!(t.fast_cpus(), Some((ccd0, "large-cache")));
    }

    #[test]
    fn capacity_and_cache_combine() {
        let t = Topology { hybrid_core: None, capacities: vec![(0, 1024), (1, 1024), (2, 512), (3, 512)], l3: vec![(32 << 20, vec![0, 2]), (16 << 20, vec![1, 3])] };
        assert_eq!(t.fast_cpus(), Some((vec![0], "performance large-cache")));
        let t = Topology { hybrid_core: Some(vec![0, 1]), capacities: vec![], l3: vec![(32 << 20, vec![2, 3]), (16 << 20, vec![0, 1])] };
        assert_eq!(t.fast_cpus(), Some((vec![0, 1], "performance")));
    }
}
