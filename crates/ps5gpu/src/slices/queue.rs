// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The ordered hand-over of slices from the sequencer to a slice thread:
//! a bounded FIFO (back-pressure on the sequencer) with completion
//! sequence numbers (drains wait for them).
//!
//! Wake-ups cost a system call and a context switch each, so they are
//! given only to threads that wait, and a sequencer waiting for room is
//! woken once the queue is half empty, not after every slice taken.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::Duration;

struct State<T> {
    items: VecDeque<(u64, T)>,
    /// Sequence number of the last slice pushed / executed.
    pushed: u64,
    done: u64,
    closed: bool,
    /// No more slices come: `pop` returns None once the queue is empty.
    ended: bool,
    /// Threads waiting for room / for a slice / for a completion.
    room_waiters: u32,
    item_waiters: u32,
    done_waiters: u32,
}

pub struct SliceQueue<T> {
    state: Mutex<State<T>>,
    room: Condvar,
    item: Condvar,
    done: Condvar,
    bound: usize,
}

/// Waits are cut into pieces this long to look at the process shutdown.
const POLL: Duration = Duration::from_millis(100);

fn stopping() -> bool {
    ps5vm::shutdown::requested()
}

impl<T> SliceQueue<T> {
    pub fn new(bound: usize) -> Self {
        SliceQueue {
            state: Mutex::new(State {
                items: VecDeque::new(),
                pushed: 0,
                done: 0,
                closed: false,
                ended: false,
                room_waiters: 0,
                item_waiters: 0,
                done_waiters: 0,
            }),
            room: Condvar::new(),
            item: Condvar::new(),
            done: Condvar::new(),
            bound: bound.max(1),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State<T>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn wait<'a>(cv: &Condvar, g: MutexGuard<'a, State<T>>) -> MutexGuard<'a, State<T>> {
        cv.wait_timeout(g, POLL).unwrap_or_else(|e| e.into_inner()).0
    }

    fn close_locked(&self, g: &mut State<T>) {
        g.closed = true;
        self.room.notify_all();
        self.item.notify_all();
        self.done.notify_all();
    }

    /// Queue slice `seq` (numbered from 1, consecutively) once the slices
    /// before it are queued and there is room. Returns the time spent
    /// waiting for room (ns), or None when the queue closed (the slice is
    /// dropped).
    pub fn push(&self, seq: u64, item: T) -> Option<u64> {
        let mut g = self.lock();
        let mut waited = None;
        while !g.closed && (g.pushed + 1 != seq || g.items.len() >= self.bound) {
            if stopping() {
                self.close_locked(&mut g);
                break;
            }
            waited.get_or_insert_with(std::time::Instant::now);
            g.room_waiters += 1;
            g = Self::wait(&self.room, g);
            g.room_waiters -= 1;
        }
        if g.closed {
            return None;
        }
        g.items.push_back((seq, item));
        g.pushed = seq;
        if g.item_waiters != 0 {
            self.item.notify_one();
        }
        if g.room_waiters != 0 {
            // Another producer may be next in sequence.
            self.room.notify_all();
        }
        Some(waited.map_or(0, |t| t.elapsed().as_nanos() as u64))
    }

    /// The next slice (blocks); None once the queue is closed, or ended
    /// and empty.
    pub fn pop(&self) -> Option<(u64, T)> {
        let mut g = self.lock();
        loop {
            if g.closed {
                return None;
            }
            if let Some(x) = g.items.pop_front() {
                // A sequencer waiting for room refills from half empty.
                if g.room_waiters != 0 && g.items.len() <= self.bound / 2 {
                    self.room.notify_all();
                }
                return Some(x);
            }
            if g.ended {
                return None;
            }
            if stopping() {
                self.close_locked(&mut g);
                return None;
            }
            g.item_waiters += 1;
            g = Self::wait(&self.item, g);
            g.item_waiters -= 1;
        }
    }

    /// Slice `seq` was executed.
    pub fn complete(&self, seq: u64) {
        let mut g = self.lock();
        g.done = g.done.max(seq);
        if g.done_waiters != 0 {
            self.done.notify_all();
        }
    }

    /// Wait until slice `seq` was executed. False when the queue closed
    /// first.
    pub fn wait_done(&self, seq: u64) -> bool {
        let mut g = self.lock();
        while g.done < seq {
            if g.closed {
                return false;
            }
            if stopping() {
                self.close_locked(&mut g);
                return false;
            }
            g.done_waiters += 1;
            g = Self::wait(&self.done, g);
            g.done_waiters -= 1;
        }
        true
    }

    /// The last slice executed.
    pub fn done(&self) -> u64 {
        self.lock().done
    }

    /// The last slice was pushed: `pop` drains the queue, then returns None.
    pub fn end(&self) {
        let mut g = self.lock();
        g.ended = true;
        self.item.notify_all();
    }

    /// Stop: queued slices are dropped, waiters return.
    pub fn close(&self) -> Vec<T> {
        let mut g = self.lock();
        self.close_locked(&mut g);
        g.items.drain(..).map(|(_, t)| t).collect()
    }

    pub fn closed(&self) -> bool {
        self.lock().closed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn slices_arrive_in_sequence_order_with_back_pressure() {
        let q = Arc::new(SliceQueue::new(2));
        // Pushed out of order from two threads: taken in sequence order.
        let q2 = q.clone();
        let late = std::thread::spawn(move || q2.push(2, "b").is_some());
        std::thread::sleep(Duration::from_millis(20));
        assert!(q.push(1, "a").is_some());
        assert!(late.join().unwrap());
        // Full: the third waits until one is taken.
        let q3 = q.clone();
        let third = std::thread::spawn(move || q3.push(3, "c"));
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(q.pop(), Some((1, "a")));
        assert!(third.join().unwrap().is_some());
        assert_eq!(q.pop(), Some((2, "b")));
        q.complete(2);
        assert!(q.wait_done(2));
        assert_eq!(q.done(), 2);
        // Closing wakes waiters and drops what is queued.
        let q4 = q.clone();
        let waiter = std::thread::spawn(move || q4.wait_done(3));
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(q.close(), vec!["c"]);
        assert!(!waiter.join().unwrap());
        assert_eq!(q.pop(), None);
        assert_eq!(q.push(4, "d"), None);
        // Ended: what is queued is still taken.
        let q = SliceQueue::new(4);
        q.push(1, 1);
        q.push(2, 2);
        q.end();
        assert_eq!((q.pop(), q.pop(), q.pop()), (Some((1, 1)), Some((2, 2)), None));
    }

    /// A producer and a consumer passing many slices through a small
    /// queue, with drains in between: nothing is lost or stuck.
    #[test]
    fn many_slices_through_a_small_queue() {
        let q = Arc::new(SliceQueue::new(4));
        let c = q.clone();
        let consumer = std::thread::spawn(move || {
            let mut next = 1;
            while let Some((n, v)) = c.pop() {
                assert_eq!((n, v), (next, next * 3));
                c.complete(n);
                next += 1;
            }
            next - 1
        });
        for n in 1..=5000u64 {
            assert!(q.push(n, n * 3).is_some());
            if n % 97 == 0 {
                assert!(q.wait_done(n));
            }
        }
        q.end();
        assert_eq!(consumer.join().unwrap(), 5000);
        assert!(q.wait_done(5000));
    }
}
