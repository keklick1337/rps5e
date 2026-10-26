// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Host locks on hot guest paths under N guest-like threads.
//!
//! Each thread calls one HLE entry point in a loop on objects of its own
//! (its own semaphore, event queue, memory range), so what the threads
//! share is only the host-side lookup: a registry lock, the memory lock,
//! the thread table. The per-call cost at N threads against one thread
//! shows the lock's contention (cache-line transfers of its state word,
//! waiting).
//!
//! Run with `RPS5E_LOCK_BENCH=1 cargo test -p ps5hle --release -- --ignored
//! host_lock_contention_bench --nocapture`.

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};
    use std::time::Instant;

    /// Mean ns per call of `op` over `n` threads, `iters` calls each
    /// (`setup` makes each thread's object first).
    fn per_call<S: Send + 'static>(n: usize, iters: u64, setup: impl Fn(usize) -> S + Sync, op: impl Fn(&S) + Sync) -> f64 {
        let barrier = Arc::new(Barrier::new(n));
        let total = std::thread::scope(|s| {
            let handles: Vec<_> = (0..n)
                .map(|t| {
                    let b = barrier.clone();
                    let (setup, op) = (&setup, &op);
                    s.spawn(move || {
                        let obj = setup(t);
                        b.wait();
                        let t0 = Instant::now();
                        for _ in 0..iters {
                            op(&obj);
                        }
                        t0.elapsed().as_nanos() as f64
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum::<f64>()
        });
        total / (n as f64 * iters as f64)
    }

    #[test]
    #[ignore]
    fn host_lock_contention_bench() {
        if std::env::var_os("RPS5E_LOCK_BENCH").is_none() {
            return;
        }
        let iters: u64 = std::env::var("RPS5E_LOCK_BENCH_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(200_000);
        let threads = [1usize, 2, 4, 8, 16];
        let report = |name: &str, f: &dyn Fn(usize) -> f64| {
            let cells: Vec<String> = threads.iter().map(|&n| format!("{:8.1}", f(n))).collect();
            eprintln!("host_lock_contention_bench: {name:<34} ns/call at N={threads:?}: {}", cells.join(" "));
        };
        for memo in [false, true] {
            crate::sync::force_registry_cache(Some(memo));
            let tag = if memo { "memo" } else { "lock" };
            // Semaphores: the registry lookup (and the object's own lock).
            report(&format!("sema signal+poll ({tag})"), &|n| {
                per_call(n, iters, |_| crate::sync::sema::create("bench".into(), false, 0, i32::MAX), |&h| {
                    crate::sync::sema::signal(h, 1);
                    crate::sync::sema::poll(h, 1);
                })
            });
            // Event queues: the registry lookup.
            report(&format!("equeue is_valid ({tag})"), &|n| per_call(n, iters, |_| crate::sync::equeue::create(), |&h| assert!(crate::sync::equeue::is_valid_equeue(h))));
            // Event flags: the registry lookup.
            report(&format!("event flag poll ({tag})"), &|n| {
                per_call(n, iters, |_| crate::sync::event_flag::create("bench".into(), crate::sync::event_flag::ATTR_MULTI, 1), |&h| {
                    crate::sync::event_flag::poll(h, 1, crate::sync::event_flag::WAITMODE_OR);
                })
            });
            // The thread table (priority / affinity queries of a registered thread).
            report(&format!("pthread current_priority ({tag})"), &|n| per_call(n, iters, |_| crate::pthread::tests::fake_thread("bench", 1), |_| assert!(crate::pthread::current_priority() > 0)));
        }
        crate::sync::force_registry_cache(None);
        // The memory state lock (queries of the committed ranges).
        report("memory clamp_range_size (MEM)", &|n| per_call(n, iters / 4, |_| (), |_| assert_eq!(crate::memory::clamp_range_size(0x10_0000, 4096), 0)));
        // The range registry (per-thread cache on hits).
        report("ranges accessible (cached)", &|n| {
            per_call(n, iters, |t| {
                let a = 0x7100_0000_0000u64 + t as u64 * 0x10_0000;
                ps5vm::ranges::track(a, 0x10_0000, ps5vm::MemProtect::RW);
                a
            }, |&a| assert!(ps5vm::ranges::accessible(a + 64, 64, true)))
        });
    }
}
