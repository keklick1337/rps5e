// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Destruction of retired GPU objects on a helper thread.
//!
//! Objects retired with a tick (buffers, their host-memory imports, images,
//! views, pipelines) are freed once the tick completed. Freeing them costs
//! driver calls (an imported host allocation is unpinned, dedicated memory
//! is returned to the kernel driver) that took several percent of the
//! recording thread's time. With the reaper the scheduler hands the
//! completed objects over in one batch and a helper thread drops them.
//!
//! The objects are no longer used by anything once their tick completed:
//! the GPU is done with them and no owner holds them, so dropping them on
//! another thread only needs each object's own external synchronisation,
//! which the hand-over gives. Every object keeps its device alive (each
//! holds the device `Arc`), and the scheduler joins the reaper when it is
//! dropped.
//!
//! `RPS5E_ASYNC_RETIRE=0`: the recording thread drops them itself (A/B
//! release name `async-retire`).

use std::any::Any;
use std::sync::mpsc;

/// Objects handed over together.
pub type Batch = Vec<Box<dyn Any + Send>>;

/// `RPS5E_ASYNC_RETIRE` (default on).
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::backend::ab::switch_on("RPS5E_ASYNC_RETIRE")) && crate::backend::ab::release_fast("async-retire")
}

/// The helper thread and the channel to it.
pub struct Reaper {
    tx: Option<mpsc::Sender<Batch>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Reaper {
    /// Start the thread (None when it cannot be spawned: the caller drops
    /// the objects itself).
    pub fn start() -> Option<Reaper> {
        let (tx, rx) = mpsc::channel::<Batch>();
        let thread = std::thread::Builder::new()
            .name("gpu-reaper".into())
            .spawn(move || {
                // Spawned from the recording thread, whose core it would
                // inherit: the reaper runs beside it, not on it.
                ps5vm::topology::unpin_current_thread_from_recorder_core("gpu-reaper");
                while let Ok(batch) = rx.recv() {
                    drop(batch);
                }
            })
            .ok()?;
        Some(Reaper { tx: Some(tx), thread: Some(thread) })
    }

    /// Hand `batch` over; returns it when the thread is gone.
    pub fn send(&self, batch: Batch) -> Result<(), Batch> {
        match &self.tx {
            Some(tx) => tx.send(batch).map_err(|e| e.0),
            None => Err(batch),
        }
    }
}

impl Drop for Reaper {
    fn drop(&mut self) {
        // Closing the channel ends the thread after the batches sent.
        self.tx.take();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    struct Probe(Arc<AtomicU64>, std::sync::Mutex<Option<std::thread::ThreadId>>, Arc<std::sync::Mutex<Vec<std::thread::ThreadId>>>);
    impl Drop for Probe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
            self.2.lock().unwrap().push(std::thread::current().id());
            let _ = self.1.lock().unwrap().take();
        }
    }

    #[test]
    fn batches_are_dropped_on_the_helper_thread_and_drained_at_drop() {
        let count = Arc::new(AtomicU64::new(0));
        let threads = Arc::new(std::sync::Mutex::new(Vec::new()));
        let r = Reaper::start().expect("thread");
        for _ in 0..3 {
            let batch: Batch = (0..4).map(|_| Box::new(Probe(count.clone(), std::sync::Mutex::new(None), threads.clone())) as Box<dyn Any + Send>).collect();
            assert!(r.send(batch).is_ok());
        }
        drop(r);
        assert_eq!(count.load(Ordering::SeqCst), 12, "every object is dropped before the reaper's drop returns");
        let me = std::thread::current().id();
        assert!(threads.lock().unwrap().iter().all(|&t| t != me), "objects are dropped on the reaper thread");
    }
}
