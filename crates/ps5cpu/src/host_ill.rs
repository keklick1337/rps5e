// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! An illegal instruction in host code (not guest code, and nothing the
//! emulator emulates): a library aborted on purpose, typically a graphics
//! driver's shader compiler stopping in its fatal-error handler (`ud2`).
//! The thread cannot go on and the others would wait for it for ever, so
//! the process ends at once, after one report: the thread, the library
//! and offset of the instruction, the thread's note
//! (`ps5vm::thread_note`: the pipeline being compiled) and the text the
//! argument registers point at (a fatal-error handler is handed its
//! message).

/// The file mapped at `addr` and the offset of `addr` in that file.
#[cfg(unix)]
fn mapping_of(addr: u64) -> Option<(String, u64)> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    maps.lines().find_map(|line| mapping_line(line, addr))
}

#[cfg(windows)]
fn mapping_of(_addr: u64) -> Option<(String, u64)> {
    None
}

/// [`mapping_of`] for one line of `/proc/<pid>/maps`.
#[cfg_attr(windows, allow(dead_code))]
fn mapping_line(line: &str, addr: u64) -> Option<(String, u64)> {
    let mut fields = line.split_ascii_whitespace();
    let (start, end) = fields.next()?.split_once('-')?;
    let (start, end) = (u64::from_str_radix(start, 16).ok()?, u64::from_str_radix(end, 16).ok()?);
    if !(start..end).contains(&addr) {
        return None;
    }
    let offset = u64::from_str_radix(fields.nth(1)?, 16).ok()?;
    // Device and inode, then the path (absent for anonymous memory).
    let path = fields.nth(2)?;
    let name = path.rsplit('/').next().unwrap_or(path);
    Some((name.to_string(), addr - start + offset))
}

/// Up to `buf.len()` bytes at `addr`, as many as are readable (never
/// faults).
#[cfg(unix)]
fn read_memory(addr: u64, buf: &mut [u8]) -> usize {
    let mut got = 0;
    while got < buf.len() {
        // One page at a time: a read across an unmapped page returns only
        // what precedes it, or fails.
        let at = addr + got as u64;
        let len = ((0x1000 - (at & 0xFFF)) as usize).min(buf.len() - got);
        let local = libc::iovec { iov_base: buf[got..].as_mut_ptr().cast(), iov_len: len };
        let remote = libc::iovec { iov_base: at as *mut libc::c_void, iov_len: len };
        // SAFETY: both vectors describe `len` bytes; the kernel checks the
        // remote one.
        let n = unsafe { libc::process_vm_readv(libc::getpid(), &local, 1, &remote, 1, 0) };
        if n <= 0 {
            break;
        }
        got += n as usize;
        if (n as usize) < len {
            break;
        }
    }
    got
}

#[cfg(windows)]
fn read_memory(_addr: u64, _buf: &mut [u8]) -> usize {
    0
}

/// The message in `bytes`, if they start with one: at least eight
/// characters of printable text up to a NUL (or the end of `bytes`).
fn message(bytes: &[u8]) -> Option<String> {
    let text = &bytes[..bytes.iter().position(|&c| c == 0).unwrap_or(bytes.len())];
    let printable = |c: &u8| c.is_ascii_graphic() || matches!(c, b' ' | b'\n' | b'\t');
    (text.len() >= 8 && text.iter().all(printable)).then(|| String::from_utf8_lossy(text).into_owned())
}

/// The lines of the report for an illegal instruction at `rip` on thread
/// `thread` whose note is `note`; `text` reads the message a register
/// points at.
fn report_lines(thread: &str, rip: u64, place: Option<(String, u64)>, note: Option<String>, pointers: &[(&str, u64)], text: &dyn Fn(u64) -> Option<String>) -> Vec<String> {
    let place = match place {
        Some((name, offset)) => format!("{rip:#x} ({name}+{offset:#x})"),
        None => format!("{rip:#x}"),
    };
    let mut lines = vec![format!("[exec] HOST FAULT: illegal instruction on thread {thread} at {place}; a library aborted, the process ends now")];
    if let Some(note) = note {
        lines.push(format!("[exec]   the thread was compiling: {note}"));
    }
    let mut seen: Vec<u64> = Vec::new();
    for &(name, value) in pointers {
        if seen.contains(&value) {
            continue;
        }
        if let Some(t) = text(value) {
            seen.push(value);
            lines.push(format!("[exec]   {name} -> {t:?}"));
        }
    }
    lines
}

/// Report an illegal instruction of host code at `rip` (`rdi`, `rsi`, `rdx`:
/// the argument registers there) and end the process.
pub(crate) fn report_and_exit(signum: crate::ctx::c_int, rip: u64, rdi: u64, rsi: u64, rdx: u64) -> ! {
    let thread = std::thread::current();
    let text = |at: u64| {
        let mut buf = [0u8; 1024];
        let n = read_memory(at, &mut buf);
        message(&buf[..n])
    };
    for line in report_lines(thread.name().unwrap_or("unnamed"), rip, mapping_of(rip), ps5vm::thread_note::get(), &[("rsi", rsi), ("rdi", rdi), ("rdx", rdx)], &text) {
        log::error!("{line}");
    }
    log::logger().flush();
    ps5vm::os::exit_now(128 + signum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_maps_line_names_the_file_and_the_file_offset() {
        let line = "7f8c27998000-7f8c2c199000 r-xp 00398000 fc:00 553578                     /usr/lib64/libdriver-compiler.so.1";
        assert_eq!(mapping_line(line, 0x7f8c2799b07e), Some(("libdriver-compiler.so.1".to_string(), 0x39b07e)));
        assert_eq!(mapping_line(line, 0x7f8c2c199000), None);
        // Anonymous memory has no name.
        assert_eq!(mapping_line("7f9be04fa000-7f9be06fb000 rw-p 00000000 00:00 0", 0x7f9be04fa010), None);
    }

    #[test]
    fn messages_are_printable_text() {
        assert_eq!(message(b"cannot select: %8\0junk").as_deref(), Some("cannot select: %8"));
        assert_eq!(message(b"short\0"), None);
        assert_eq!(message(&[0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88]), None);
        assert_eq!(message(&[]), None);
    }

    #[cfg(unix)]
    #[test]
    fn unreadable_memory_reads_as_nothing() {
        let mut buf = [0u8; 64];
        assert_eq!(read_memory(8, &mut buf), 0);
        // The readable page before an unmapped one is read up to its end.
        let p = ps5vm::os::map_anon(0x2000, ps5vm::MemProtect::RW).expect("two pages");
        assert!(ps5vm::os::protect(p + 0x1000, 0x1000, ps5vm::MemProtect::NONE));
        // SAFETY: the first page is mapped writable.
        unsafe { std::ptr::copy_nonoverlapping(b"the reason\0".as_ptr(), (p + 0x1000 - 11) as *mut u8, 11) };
        let n = read_memory(p + 0x1000 - 11, &mut buf);
        assert_eq!(n, 11);
        assert_eq!(message(&buf[..n]).as_deref(), Some("the reason"));
    }

    #[test]
    fn the_report_names_thread_place_note_and_message() {
        let text = |at: u64| (at == 0x1000).then(|| "cannot select: gather".to_string());
        let lines = report_lines("rps5e-comp-12", 0x7f00_0000_207e, Some(("libdriver.so".into(), 0x39b07e)), Some("compute cs 2bda program 0x1068eb5a00".into()), &[("rsi", 0x1000), ("rdi", 0), ("rdx", 0x1000)], &text);
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert!(lines[0].contains("thread rps5e-comp-12") && lines[0].contains("(libdriver.so+0x39b07e)"), "{}", lines[0]);
        assert!(lines[1].ends_with("compute cs 2bda program 0x1068eb5a00"), "{}", lines[1]);
        assert_eq!(lines[2], "[exec]   rsi -> \"cannot select: gather\"");
        // Without a mapping or a note the first line stands alone.
        assert_eq!(report_lines("t", 0x10, None, None, &[], &text).len(), 1);
    }

    /// A child process whose host thread executes `ud2` ends at once with
    /// the exit status of the signal (not by the signal, which would write
    /// a core dump of the whole guest memory) and reports the thread, its
    /// note and the message its argument register points at.
    #[cfg(unix)]
    #[test]
    fn an_illegal_instruction_on_a_host_thread_ends_the_process_with_a_report() {
        const CHILD: &str = "RPS5E_TEST_HOST_ILL_CHILD";
        if std::env::var_os(CHILD).is_some() {
            struct Stderr;
            impl log::Log for Stderr {
                fn enabled(&self, _: &log::Metadata) -> bool {
                    true
                }
                fn log(&self, r: &log::Record) {
                    eprintln!("{}", r.args());
                }
                fn flush(&self) {}
            }
            static LOGGER: Stderr = Stderr;
            let _ = log::set_logger(&LOGGER);
            log::set_max_level(log::LevelFilter::Error);
            crate::exec::init();
            let worker = std::thread::Builder::new().name("rps5e-comp-12".into()).spawn(|| {
                let _note = ps5vm::thread_note::set("compute cs 00000000000000aa program 0x1000 7 words".into());
                let reason = c"cannot select: the test instruction";
                // SAFETY: the instruction faults; the handler ends the process.
                unsafe { std::arch::asm!("ud2", in("rsi") reason.as_ptr(), options(noreturn)) }
            });
            let _ = worker.unwrap().join();
            // Not reached: the fault ends the process.
            std::process::exit(0);
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "host_ill::tests::an_illegal_instruction_on_a_host_thread_ends_the_process_with_a_report", "--nocapture", "--test-threads", "1"])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let err = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(128 + crate::ctx::SIGILL), "status {:?}, stderr: {err}", out.status);
        assert!(err.contains("illegal instruction on thread rps5e-comp-12 at 0x"), "{err}");
        assert!(err.contains("the thread was compiling: compute cs 00000000000000aa program 0x1000 7 words"), "{err}");
        assert!(err.contains("rsi -> \"cannot select: the test instruction\""), "{err}");
    }
}
