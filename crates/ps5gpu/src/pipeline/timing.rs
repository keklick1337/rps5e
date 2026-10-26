// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! How long shader modules and pipelines take to build, and how long the
//! command thread waits for them (the `create=` part of the `[gpu]` line).
//!
//! Pipeline creation is where the driver compiles the SPIR-V to machine
//! code; a large program can take the driver seconds, and a draw or a
//! dispatch that needs it waits. Builds slower than [`SLOW_MS`] are logged
//! with their programs and SPIR-V size.

/// Builds at least this slow are logged one by one.
pub const SLOW_MS: u64 = 100;

/// Upper bounds (ms) of the histogram buckets; the last bucket is open.
const BUCKETS_MS: [u64; 5] = [1, 10, 100, 1000, 5000];

/// Count, total and longest time of one kind of work, with a histogram.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Times {
    pub count: u64,
    pub total_ns: u64,
    pub max_ns: u64,
    pub buckets: [u64; BUCKETS_MS.len() + 1],
}

impl Times {
    pub fn add(&mut self, ns: u64) {
        self.count += 1;
        self.total_ns += ns;
        self.max_ns = self.max_ns.max(ns);
        let ms = ns / 1_000_000;
        let i = BUCKETS_MS.iter().position(|&b| ms < b).unwrap_or(BUCKETS_MS.len());
        self.buckets[i] += 1;
    }

    /// `count/total ms/max ms[<1:n <10:n ... >=5000:n]` (empty buckets left
    /// out).
    pub fn format(&self) -> String {
        let mut s = format!("{}/{}ms/max{}ms", self.count, self.total_ns / 1_000_000, self.max_ns / 1_000_000);
        let mut parts = Vec::new();
        for (i, &n) in self.buckets.iter().enumerate() {
            if n == 0 {
                continue;
            }
            match BUCKETS_MS.get(i) {
                Some(b) => parts.push(format!("<{b}:{n}")),
                None => parts.push(format!(">={}:{n}", BUCKETS_MS[BUCKETS_MS.len() - 1])),
            }
        }
        if !parts.is_empty() {
            s += &format!("[{}]", parts.join(" "));
        }
        s
    }
}

/// The build and wait times of a provider.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompileTimes {
    /// Guest program to SPIR-V (our compiler, or the module cache).
    pub translate: Times,
    /// vkCreateShaderModule.
    pub module: Times,
    /// vkCreateGraphicsPipelines / vkCreateComputePipelines that were
    /// used as built: found in the pipeline cache, or built in full.
    pub graphics: Times,
    pub compute: Times,
    /// Pipelines built without driver optimisation (used until their
    /// optimised build replaces them), and those optimised builds.
    pub fast: Times,
    pub optimised: Times,
    /// Pipelines the cache probe found (no compile needed).
    pub probe_hits: u64,
    /// Fast pipelines replaced by their optimised build.
    pub swaps: u64,
    /// The command thread blocked on a module or pipeline.
    pub waits: Times,
}

impl CompileTimes {
    pub fn format(&self) -> String {
        format!(
            "create=[translate {} module {} gfx {} cs {} fast {} opt {} probe-hits {} swaps {} waits {}]",
            self.translate.format(),
            self.module.format(),
            self.graphics.format(),
            self.compute.format(),
            self.fast.format(),
            self.optimised.format(),
            self.probe_hits,
            self.swaps,
            self.waits.format()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_bucket_and_format() {
        let mut t = Times::default();
        t.add(500_000); // 0.5 ms
        t.add(5_000_000); // 5 ms
        t.add(150_000_000); // 150 ms
        t.add(7_000_000_000); // 7 s
        assert_eq!(t.count, 4);
        assert_eq!(t.max_ns, 7_000_000_000);
        assert_eq!(t.buckets, [1, 1, 0, 1, 0, 1]);
        assert_eq!(t.format(), "4/7155ms/max7000ms[<1:1 <10:1 <1000:1 >=5000:1]");
        assert_eq!(Times::default().format(), "0/0ms/max0ms");
    }
}
