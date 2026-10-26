// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Committed state shared with the slice threads under the frozen
//! discipline (R7b stage 2).
//!
//! While slice threads record a wave, the committing thread holds the
//! back-end guard and does nothing else: the committed state (the back-end,
//! the caches, the pipeline tables) is only read. Between waves only the
//! committing thread runs and writes it. [`Shared`] holds state that is
//! read through `&` on every thread and written through [`Shared::w`] only
//! by the committing thread outside a wave; [`Recording`] marks a wave.
//! Writes from a slice thread, or during a wave, panic.

use std::cell::{Cell, UnsafeCell};
use std::sync::atomic::{AtomicU32, Ordering};

/// Slice threads recording right now.
static RECORDING: AtomicU32 = AtomicU32::new(0);

thread_local! {
    static SLICE_THREAD: Cell<bool> = const { Cell::new(false) };
}

/// Mark the calling thread a slice thread (it never writes shared state).
pub fn mark_slice_thread() {
    SLICE_THREAD.with(|c| c.set(true));
}

/// True on a slice thread.
pub fn on_slice_thread() -> bool {
    SLICE_THREAD.with(|c| c.get())
}

/// A slice thread records while this lives.
pub struct Recording(());

impl Recording {
    pub fn begin() -> Recording {
        RECORDING.fetch_add(1, Ordering::AcqRel);
        Recording(())
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        RECORDING.fetch_sub(1, Ordering::AcqRel);
    }
}

/// True while any slice thread records.
pub fn recording() -> bool {
    RECORDING.load(Ordering::Acquire) != 0
}

/// State read on every thread, written by the committing thread between
/// waves only (see the module documentation).
pub struct Shared<T>(UnsafeCell<T>);

// SAFETY: written only by one thread while no other thread reads it
// (checked in `w`); reads are shared.
unsafe impl<T: Send + Sync> Sync for Shared<T> {}
// SAFETY: as above (the cell moves with its owner).
unsafe impl<T: Send> Send for Shared<T> {}

impl<T> Shared<T> {
    pub fn new(v: T) -> Shared<T> {
        Shared(UnsafeCell::new(v))
    }

    /// Write access: the committing thread, outside a wave. The reference
    /// must not outlive the statement that uses it (no other reference to
    /// the same value may be live meanwhile).
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub fn w(&self) -> &mut T {
        if on_slice_thread() || recording() {
            panic!("ps5gpu: shared state written during slice recording");
        }
        // SAFETY: no reader runs concurrently (frozen discipline, checked
        // above); callers keep the reference short-lived.
        unsafe { &mut *self.0.get() }
    }
}

impl<T> std::ops::Deref for Shared<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: writers run only when no reader does (see `w`).
        unsafe { &*self.0.get() }
    }
}

impl<T: Default> Default for Shared<T> {
    fn default() -> Self {
        Shared::new(T::default())
    }
}
