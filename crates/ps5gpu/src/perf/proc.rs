// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Parsers for the Linux `/proc` files the overlay samples. Pure functions
//! on the file text, so they are unit tested without a live system.

/// Aggregate CPU time of the machine (`/proc/stat`, first `cpu` line), in
/// clock ticks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuTimes {
    /// Every tick (user..steal; guest time is already part of user).
    pub total: u64,
    /// Ticks spent idle or waiting for I/O.
    pub idle: u64,
}

impl CpuTimes {
    /// Busy share between two samples, 0..=100.
    pub fn busy_percent(&self, earlier: &CpuTimes) -> f32 {
        let total = self.total.saturating_sub(earlier.total);
        let idle = self.idle.saturating_sub(earlier.idle);
        if total == 0 {
            return 0.0;
        }
        (total.saturating_sub(idle) as f64 * 100.0 / total as f64) as f32
    }
}

/// The aggregate `cpu` line of `/proc/stat`.
pub fn parse_proc_stat(text: &str) -> Option<CpuTimes> {
    let line = text.lines().find(|l| l.starts_with("cpu "))?;
    let v: Vec<u64> = line.split_ascii_whitespace().skip(1).take(8).map(|f| f.parse().unwrap_or(0)).collect();
    if v.len() < 4 {
        return None;
    }
    let idle = v[3] + v.get(4).copied().unwrap_or(0);
    Some(CpuTimes { total: v.iter().sum(), idle })
}

/// Number of `cpuN` lines of `/proc/stat` (online CPUs).
pub fn count_cpus(text: &str) -> usize {
    text.lines().filter(|l| l.starts_with("cpu") && l.as_bytes().get(3).is_some_and(u8::is_ascii_digit)).count()
}

/// The command name and CPU ticks (utime + stime) of a process or thread
/// `stat` file (`/proc/self/stat`, `/proc/self/task/<tid>/stat`).
pub fn parse_task_stat(text: &str) -> Option<(&str, u64)> {
    // The name is in parentheses and may itself hold spaces or ')'.
    let open = text.find('(')?;
    let close = text.rfind(')')?;
    if close < open {
        return None;
    }
    let name = &text[open + 1..close];
    // After ")": state(3) ppid(4) ... utime(14) stime(15).
    let mut f = text[close + 1..].split_ascii_whitespace();
    let utime: u64 = f.nth(11)?.parse().ok()?;
    let stime: u64 = f.next()?.parse().ok()?;
    Some((name, utime + stime))
}

/// Resident set size in pages (`/proc/self/statm`, second field).
pub fn parse_statm_resident(text: &str) -> Option<u64> {
    text.split_ascii_whitespace().nth(1)?.parse().ok()
}

/// System memory from `/proc/meminfo`: (total, used) in bytes, where used
/// is total minus available.
pub fn parse_meminfo(text: &str) -> Option<(u64, u64)> {
    let field = |name: &str| -> Option<u64> {
        let line = text.lines().find(|l| l.starts_with(name) && l.as_bytes().get(name.len()) == Some(&b':'))?;
        let kib: u64 = line[name.len() + 1..].split_ascii_whitespace().next()?.parse().ok()?;
        Some(kib * 1024)
    };
    let total = field("MemTotal")?;
    let available = field("MemAvailable").or_else(|| Some(field("MemFree")? + field("Cached").unwrap_or(0)))?;
    Some((total, total.saturating_sub(available)))
}

/// The active clock of an amdgpu `pp_dpm_sclk` file, in MHz: the line
/// marked with `*` (`1: 1850Mhz *`).
pub fn parse_dpm_clock(text: &str) -> Option<u32> {
    let line = text.lines().find(|l| l.trim_end().ends_with('*'))?;
    let value = line.split_ascii_whitespace().nth(1)?;
    let digits = value.trim_end_matches(|c: char| c.is_ascii_alphabetic());
    digits.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const STAT: &str = "cpu  100 5 50 800 20 3 2 0 7 0\ncpu0 50 2 25 400 10 1 1 0 3 0\ncpu1 50 3 25 400 10 2 1 0 4 0\nintr 1 2 3\nctxt 99\n";

    #[test]
    fn proc_stat_totals_and_busy_share() {
        let a = parse_proc_stat(STAT).unwrap();
        assert_eq!(a, CpuTimes { total: 980, idle: 820 });
        assert_eq!(count_cpus(STAT), 2);
        let b = CpuTimes { total: a.total + 200, idle: a.idle + 150 };
        assert!((b.busy_percent(&a) - 25.0).abs() < 1e-4);
        assert_eq!(a.busy_percent(&a), 0.0);
        assert!(parse_proc_stat("intr 1\n").is_none());
    }

    #[test]
    fn task_stat_with_odd_names() {
        let s = "4242 (rps5e-render) S 1 2 3 0 -1 4194304 125 0 0 0 1200 34 0 0 20 0 1 0 1 2 3";
        assert_eq!(parse_task_stat(s), Some(("rps5e-render", 1234)));
        let odd = "17 (a) b (c)) R 1 2 3 0 -1 0 0 0 0 0 10 5 0 0 20 0 1 0 1 2 3";
        assert_eq!(parse_task_stat(odd), Some(("a) b (c)", 15)));
        assert!(parse_task_stat("17 (short) R 1").is_none());
    }

    #[test]
    fn statm_and_meminfo() {
        assert_eq!(parse_statm_resident("57681 530 502 7 0 152 0\n"), Some(530));
        let mi = "MemTotal:       16000000 kB\nMemFree:         1000000 kB\nMemAvailable:    6000000 kB\nCached: 3 kB\n";
        assert_eq!(parse_meminfo(mi), Some((16_000_000 * 1024, 10_000_000 * 1024)));
        // Old kernels without MemAvailable: free + cached.
        let old = "MemTotal: 100 kB\nMemFree: 30 kB\nCached: 20 kB\n";
        assert_eq!(parse_meminfo(old), Some((100 * 1024, 50 * 1024)));
        assert!(parse_meminfo("Buffers: 1 kB\n").is_none());
    }

    #[test]
    fn amdgpu_dpm_clock() {
        assert_eq!(parse_dpm_clock("0: 500Mhz\n1: 1850Mhz *\n2: 2400Mhz\n"), Some(1850));
        assert_eq!(parse_dpm_clock("0: 500Mhz\n"), None);
    }
}
