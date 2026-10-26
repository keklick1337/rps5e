// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Submission engines: APR (7 priorities) and AMM (3 priorities).
//!
//! A submission snapshots the commands recorded so far and is queued at
//! its priority; the submitting call returns at once. One host worker per
//! engine executes the queue heads command by command, always serving the
//! highest-priority queue that can progress. A wait command whose
//! condition does not hold blocks its queue (later submissions at that
//! priority stay behind it) while other priorities keep running; addresses
//! are re-polled every 100 µs because CPU and GPU fence writes do not
//! notify the engine. On completion (or the first failing command) the
//! `{result, errorOffset}` pair is written and waiters are released.
//!
//! AMM and APR share 256 32-bit counters; only the first 128 are
//! application-writable.

use super::amm;
use super::command_buffer::{self as cbuf, Command, CommandData};
use crate::agc::mem;
use crate::context::CpuContext;
use crate::filesystem;
use crate::registry::HleResult;
use crate::sync::equeue;
use crate::sync::{ERR_INVALID_ARGUMENT, ERR_MEMORY_FAULT, ERR_NOT_FOUND, OK};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io;
use crate::host::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

/// ESRCH: no submission with that id.
pub const ERR_NO_SUCH_SUBMISSION: HleResult = 0x8002_0003u32 as i32 as i64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Engine {
    Apr,
    Amm,
}

// ---------------------------------------------------------------------------
// Shared counters
// ---------------------------------------------------------------------------

pub const COUNTER_COUNT: usize = 256;
/// Counters below this index are writable by the application.
pub const APP_COUNTERS: u8 = 128;

static COUNTERS: [AtomicU32; COUNTER_COUNT] = [const { AtomicU32::new(0) }; COUNTER_COUNT];

pub fn counter(index: u8) -> u32 {
    COUNTERS[index as usize].load(Ordering::Acquire)
}

// ---------------------------------------------------------------------------
// Submission ids
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Submissions {
    next: u32,
    /// id -> complete
    pending: HashMap<u32, bool>,
}

fn submissions() -> &'static (Mutex<Submissions>, Condvar) {
    static S: OnceLock<(Mutex<Submissions>, Condvar)> = OnceLock::new();
    S.get_or_init(|| (Mutex::new(Submissions { next: 1, pending: HashMap::new() }), Condvar::new()))
}

fn lock_submissions() -> MutexGuard<'static, Submissions> {
    submissions().0.lock().unwrap_or_else(|p| p.into_inner())
}

fn allocate_submission_id() -> u32 {
    let mut s = lock_submissions();
    let mut id = s.next;
    s.next = s.next.wrapping_add(1);
    if id == 0 {
        id = s.next;
        s.next = s.next.wrapping_add(1);
    }
    s.pending.insert(id, false);
    id
}

fn set_submission_complete(id: u32) {
    if id == 0 {
        return;
    }
    let mut s = lock_submissions();
    if let Some(done) = s.pending.get_mut(&id) {
        *done = true;
    }
    submissions().1.notify_all();
}

/// Block until submission `id` completes and retire it; false when no such
/// submission is pending.
pub fn complete_submission(id: u32) -> bool {
    let (m, cv) = submissions();
    let mut s = lock_submissions();
    if !s.pending.contains_key(&id) {
        return false;
    }
    while s.pending.get(&id) == Some(&false) {
        s = crate::signals::wait_cv(m, cv, s, None).0;
    }
    s.pending.remove(&id).is_some()
}

// ---------------------------------------------------------------------------
// Queueing
// ---------------------------------------------------------------------------

struct Submission {
    commands: Vec<Command>,
    cursor: usize,
    /// Guest `{int32 result; uint32 errorOffset}` (0 = none).
    result: u64,
    id: u32,
}

struct CommandEngine {
    queues: Mutex<Vec<VecDeque<Submission>>>,
    cv: Condvar,
    worker: OnceLock<()>,
    name: &'static str,
}

/// Re-poll interval for blocked waits.
const WAIT_POLL: Duration = Duration::from_micros(100);

impl CommandEngine {
    fn new(priorities: usize, name: &'static str) -> Self {
        Self {
            queues: Mutex::new((0..priorities).map(|_| VecDeque::new()).collect()),
            cv: Condvar::new(),
            worker: OnceLock::new(),
            name,
        }
    }

    fn lock(&self) -> MutexGuard<'_, Vec<VecDeque<Submission>>> {
        self.queues.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn submit(&'static self, priority: usize, s: Submission) {
        self.worker.get_or_init(|| {
            std::thread::Builder::new()
                .name(self.name.into())
                .spawn(move || self.run())
                .expect("spawn AMPR engine thread");
        });
        self.lock()[priority].push_back(s);
        self.cv.notify_one();
    }

    fn run(&self) {
        let mut queues = self.lock();
        loop {
            let mut ready = None;
            let mut pending = false;
            for (p, q) in queues.iter().enumerate() {
                let Some(s) = q.front() else { continue };
                pending = true;
                if let Some(Command { data: CommandData::Wait { source, value, compare, is_counter }, .. }) =
                    s.commands.get(s.cursor)
                {
                    if !wait_satisfied(*source, *value, *compare, *is_counter) {
                        continue;
                    }
                }
                ready = Some(p);
                break;
            }
            let Some(p) = ready else {
                queues = if pending {
                    self.cv.wait_timeout(queues, WAIT_POLL).unwrap_or_else(|e| e.into_inner()).0
                } else {
                    self.cv.wait(queues).unwrap_or_else(|e| e.into_inner())
                };
                continue;
            };
            let (command, result_addr, id, last) = {
                let s = queues[p].front_mut().unwrap();
                let c = s.commands.get(s.cursor).copied();
                if c.is_some() {
                    s.cursor += 1;
                }
                (c, s.result, s.id, s.cursor == s.commands.len())
            };
            drop(queues);
            let (mut result, mut error_offset) = (OK, 0u32);
            if let Some(c) = command {
                result = execute(&c);
                if result != OK {
                    error_offset = c.record_offset as u32;
                }
            }
            let complete = result != OK || last;
            if complete {
                write_result(result_addr, result as i32, error_offset);
                set_submission_complete(id);
            }
            queues = self.lock();
            if complete {
                queues[p].pop_front();
            }
        }
    }
}

fn engine(e: Engine) -> &'static CommandEngine {
    static APR: OnceLock<CommandEngine> = OnceLock::new();
    static AMM: OnceLock<CommandEngine> = OnceLock::new();
    match e {
        Engine::Apr => APR.get_or_init(|| CommandEngine::new(7, "ampr-apr")),
        Engine::Amm => AMM.get_or_init(|| CommandEngine::new(3, "ampr-amm")),
    }
}

/// SceAmprResult: `{int32 result; uint32 errorOffset}`.
fn write_result(addr: u64, result: i32, error_offset: u32) -> bool {
    if addr == 0 {
        return true;
    }
    let mut b = [0u8; 8];
    b[..4].copy_from_slice(&result.to_le_bytes());
    b[4..].copy_from_slice(&error_offset.to_le_bytes());
    mem::write(addr, &b)
}

fn guest_writable(addr: u64, len: u64) -> bool {
    addr != 0 && ps5vm::ranges::accessible(addr, len, true)
}

/// Queue the commands of `cb` (APR: the object, everything written so
/// far; AMM: the command memory, the first `bytes` bytes).
pub fn queue_command_buffer(
    ctx: &CpuContext,
    e: Engine,
    cb: u64,
    bytes: u32,
    priority: u32,
    result: u64,
    out_id: u64,
) -> HleResult {
    let max_priority = if e == Engine::Amm { 2 } else { 6 };
    if cb == 0 || priority > max_priority {
        return ERR_INVALID_ARGUMENT;
    }
    if (result != 0 && !guest_writable(result, 8)) || (out_id != 0 && !guest_writable(out_id, 4)) {
        return ERR_MEMORY_FAULT;
    }
    let Some(state) = cbuf::try_get_state(ctx, cb) else { return ERR_MEMORY_FAULT };
    let end = if e == Engine::Amm { bytes as u64 } else { state.write_offset };
    if end == 0 || end > state.write_offset {
        return ERR_INVALID_ARGUMENT;
    }
    let commands: Vec<Command> = state.commands.into_iter().filter(|c| c.record_offset < end).collect();
    let id = if out_id != 0 { allocate_submission_id() } else { 0 };
    if out_id != 0 && !ctx.write_u32(out_id, id) {
        return ERR_MEMORY_FAULT;
    }
    log::debug!("[ampr] submit {:?} cb={:#x} prio={} commands={} id={:#x}", e, cb, priority, commands.len(), id);
    engine(e).submit(priority as usize, Submission { commands, cursor: 0, result, id });
    OK
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// Compare operations: 0 eq, 1 unsigned gt, 2 unsigned lt, 3 ne,
/// 4 wrapping ge, 5 signed gt, 6 signed lt. `sign` is the sign bit of the
/// operand width (bit 63 for addresses, bit 31 for counters).
pub fn compare_satisfied(value: u64, reference: u64, compare: u8, sign: u64) -> bool {
    match compare {
        0 => value == reference,
        1 => value > reference,
        2 => value < reference,
        3 => value != reference,
        4 => value.wrapping_sub(reference) & sign == 0,
        5 => (value ^ sign) > (reference ^ sign),
        _ => (value ^ sign) < (reference ^ sign),
    }
}

/// The u64 at a guest address, loaded with acquire ordering.
fn load_guest_u64(address: u64) -> Option<u64> {
    if address == 0 || address % 8 != 0 || !ps5vm::ranges::accessible(address, 8, false) {
        return None;
    }
    // SAFETY: the address is aligned and inside a tracked, readable guest mapping.
    Some(unsafe { AtomicU64::from_ptr(address as *mut u64) }.load(Ordering::Acquire))
}

fn wait_satisfied(source: u64, value: u64, compare: u8, is_counter: bool) -> bool {
    if is_counter {
        return compare_satisfied(counter(source as u8) as u64, value, compare, 1 << 31);
    }
    // An unreadable fence cannot block; executing the wait reports the fault.
    load_guest_u64(source).is_none_or(|v| compare_satisfied(v, value, compare, 1 << 63))
}

fn execute(c: &Command) -> HleResult {
    match c.data {
        CommandData::ReadFile { file_id, destination, size, file_offset } => {
            read_file(file_id, destination, size, file_offset)
        }
        CommandData::KernelEvent { equeue, ident, data } => {
            if equeue::trigger_ampr_event(equeue, ident as u32 as u64, data) {
                OK
            } else {
                log::warn!("[ampr] submit event failed: eq={equeue:#x} id={ident}");
                ERR_NOT_FOUND
            }
        }
        CommandData::WriteAddress { address, value } => {
            if address % 8 != 0 || !guest_writable(address, 8) {
                return ERR_MEMORY_FAULT;
            }
            // SAFETY: aligned address inside a tracked, writable guest mapping.
            unsafe { AtomicU64::from_ptr(address as *mut u64) }.store(value, Ordering::Release);
            OK
        }
        CommandData::WriteCounter { index, value } => {
            COUNTERS[index as usize].store(value, Ordering::Release);
            OK
        }
        CommandData::Wait { source, is_counter, .. } => {
            if !is_counter && load_guest_u64(source).is_none() {
                ERR_MEMORY_FAULT
            } else {
                OK
            }
        }
        CommandData::AmmMap(m) => {
            let r = amm::execute(&m);
            if r != OK {
                log::warn!(
                    "[ampr] AMM submit command failed: {:?} va={:#x} dmem={:#x} size={:#x} type={} prot={:#x} result={:#x}",
                    m.kind,
                    m.va,
                    m.dmem_offset,
                    m.size,
                    m.mtype,
                    m.prot,
                    r as u32
                );
            }
            r
        }
    }
}

// ---------------------------------------------------------------------------
// File reads
// ---------------------------------------------------------------------------

const CHUNK: u64 = 4 << 20;

/// Read up to `size` bytes at `offset` into guest memory; a short file
/// yields a short read, an offset at or past its end reads nothing.
fn read_file(file_id: u32, destination: u64, size: u64, file_offset: u64) -> HleResult {
    // Files of a package mounted at /app0 are read from the image.
    let image = filesystem::apr_image_file(file_id);
    let host = filesystem::apr_host_path(file_id).unwrap_or_default();
    if image.is_none() && host.as_os_str().is_empty() {
        log::warn!("[ampr] APR submit failed for unknown file id: {file_id:#010x}");
        return ERR_NOT_FOUND;
    }
    if size == 0 {
        return OK;
    }
    if destination.checked_add(size).is_none() || !ps5vm::ranges::accessible(destination, size, true) {
        return ERR_MEMORY_FAULT;
    }
    let file: Box<dyn Fn(&mut [u8], u64) -> io::Result<usize>> = match image {
        Some(data) => Box::new(move |b, at| data.read_at(b, at)),
        None => match cached_file(&host) {
            Ok(f) => Box::new(move |b, at| f.read_at(b, at)),
            Err(e) => {
                log::warn!("[ampr] APR read missing host file: {}", host.display());
                return e;
            }
        },
    };
    let mut buf = vec![0u8; CHUNK.min(size) as usize];
    let mut done = 0u64;
    let mut end_of_file = false;
    while done < size {
        let at = file_offset + done;
        let want = (buf.len() as u64).min(size - done) as usize;
        let n = match file(&mut buf[..want], at) {
            Ok(0) => {
                end_of_file = true;
                break;
            }
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => {
                log::warn!("[ampr] APR read of '{}' at {at:#x} failed: {e}", host.display());
                break;
            }
        };
        if !mem::write(destination + done, &buf[..n]) {
            return ERR_MEMORY_FAULT;
        }
        done += n as u64;
    }
    // The part of the destination past the end of the file reads as zeros.
    if end_of_file {
        buf.fill(0);
        while done < size {
            let n = (buf.len() as u64).min(size - done) as usize;
            if !mem::write(destination + done, &buf[..n]) {
                return ERR_MEMORY_FAULT;
            }
            done += n as u64;
        }
    }
    log::trace!(
        "[ampr] read id={:#010x} dst={:#x} size={:#x} off={:#x} read={:#x} '{}'",
        file_id,
        destination,
        size,
        file_offset,
        done,
        host.display()
    );
    OK
}

/// Bounded LRU of open host files, so asset storms do not exhaust the
/// process descriptor limit.
const MAX_CACHED_FILES: usize = 512;

#[derive(Default)]
struct FileCache {
    files: HashMap<PathBuf, (Arc<File>, u64)>, // file, last use
    tick: u64,
}

fn file_cache() -> MutexGuard<'static, FileCache> {
    static C: OnceLock<Mutex<FileCache>> = OnceLock::new();
    C.get_or_init(Default::default).lock().unwrap_or_else(|p| p.into_inner())
}

fn cached_file(host: &Path) -> Result<Arc<File>, HleResult> {
    {
        let mut c = file_cache();
        c.tick += 1;
        let tick = c.tick;
        if let Some(e) = c.files.get_mut(host) {
            e.1 = tick;
            return Ok(e.0.clone());
        }
    }
    let f = match File::open(host) {
        Ok(v) => v,
        Err(_) => {
            // Possibly out of descriptors: drop the cache and retry once.
            file_cache().files.clear();
            File::open(host).map_err(|_| ERR_NOT_FOUND)?
        }
    };
    let f = Arc::new(f);
    let mut c = file_cache();
    let tick = c.tick;
    if let Some(e) = c.files.get(host) {
        return Ok(e.0.clone());
    }
    while c.files.len() >= MAX_CACHED_FILES {
        let Some(oldest) = c.files.iter().min_by_key(|(_, v)| v.1).map(|(k, _)| k.clone()) else { break };
        c.files.remove(&oldest);
    }
    c.files.insert(host.to_path_buf(), (f.clone(), tick));
    Ok(f)
}
