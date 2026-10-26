// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Guest thread-pointer accesses on Windows.
//!
//! GS is the TEB on Windows x64 and the FS base cannot be set from user
//! mode, so the one-byte FS→GS prefix change of Linux is not possible.
//! Every FS-relative guest instruction is instead moved into a trampoline
//! that loads the guest thread pointer from a TEB TLS slot
//! (`mov S, gs:[slot]`, one instruction) and executes the instruction with
//! that pointer as an extra base register:
//!
//! ```text
//! mov rax, fs:[0x28]   ->  jmp T
//! T:  mov rax, gs:[slot]
//!     mov rax, [rax + 0x28]
//!     jmp back
//! ```
//!
//! When the instruction has no free destination register, a scratch
//! register is saved below the System V red zone. Instructions shorter
//! than the 5-byte jump get an INT3 instead; the exception handler sends
//! the thread to the trampoline (slower, but rare). Trampolines live in
//! chunks allocated within ±2 GiB of the code they serve.

use iced_x86::{
    BlockEncoder, BlockEncoderOptions, Code, Instruction, InstructionBlock, InstructionInfoFactory, MemoryOperand, Register,
};
use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

const CHUNK: u64 = 4 << 20;
const MAX_TRAMPOLINE: u64 = 128;
/// Furthest a trampoline chunk may be from the code it serves.
const REACH: u64 = 0x7000_0000;
/// The System V red zone plus the saved scratch register.
const STACK_SKIP: i64 = 128;

/// INT3 site -> trampoline.
static TRAPS: RwLock<Option<HashMap<u64, u64>>> = RwLock::new(None);

struct Arena {
    cur: u64,
    end: u64,
}

static ARENA: Mutex<Vec<Arena>> = Mutex::new(Vec::new());

/// What [`rewrite`] did.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlsCounts {
    pub jumps: usize,
    pub traps: usize,
    pub failed: usize,
}

/// Trampoline of an INT3-patched site at `addr`.
pub fn trap_target(addr: u64) -> Option<u64> {
    TRAPS.read().ok()?.as_ref()?.get(&addr).copied()
}

fn distance(a: u64, b: u64) -> u64 {
    a.abs_diff(b)
}

fn take_chunk(cand: u64) -> Option<Arena> {
    if ps5vm::os::space::is_free(cand, CHUNK) && ps5vm::os::space::commit(cand, CHUNK, ps5vm::MemProtect::RWX) {
        crate::exec::add_guest_code_range(cand, CHUNK, "code-trampolines");
        return Some(Arena { cur: cand, end: cand + CHUNK });
    }
    None
}

/// A new chunk of executable memory within reach of `near`: the closest
/// free space below it, else above it.
fn new_chunk(near: u64) -> Option<Arena> {
    use windows_sys::Win32::System::Memory::MEM_FREE;
    let g = ps5vm::os::ALLOCATION_GRANULARITY;
    let mut cand = near.saturating_sub(CHUNK) & !(g - 1);
    while cand >= 0x10_0000 && distance(cand, near) <= REACH {
        if let Some(a) = take_chunk(cand) {
            return Some(a);
        }
        let next = match ps5vm::os::query(cand) {
            Some(r) if r.state != MEM_FREE => r.start.saturating_sub(CHUNK) & !(g - 1),
            _ => cand - g,
        };
        cand = next.min(cand - g);
    }
    let mut cand = (near + g) & !(g - 1);
    while distance(cand + CHUNK, near) <= REACH {
        if let Some(a) = take_chunk(cand) {
            return Some(a);
        }
        let next = match ps5vm::os::query(cand) {
            Some(r) if r.state != MEM_FREE => (r.end + g - 1) & !(g - 1),
            _ => cand + g,
        };
        cand = next.max(cand + g);
    }
    log::error!("[tls] no free address space for trampolines near {near:#x}");
    None
}

/// Place up to `size` bytes of code within reach of `near` (a trampoline,
/// or a function copy of `redzone`): `code(at)` returns the bytes for
/// address `at`, which are copied there. The address, or None when there
/// is no room or `code` gives nothing (or more than `size`).
pub(crate) fn place(near: u64, size: u64, code: impl FnOnce(u64) -> Option<Vec<u8>>) -> Option<u64> {
    if size > CHUNK {
        return None;
    }
    let mut arenas = ARENA.lock().unwrap_or_else(|e| e.into_inner());
    let i = match arenas.iter().position(|a| a.cur + size <= a.end && distance(a.cur, near) < REACH) {
        Some(i) => i,
        None => {
            arenas.push(new_chunk(near)?);
            arenas.len() - 1
        }
    };
    let at = arenas[i].cur;
    let bytes = code(at).filter(|b| b.len() as u64 <= size)?;
    // SAFETY: free room of a committed RWX chunk of this arena.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), at as *mut u8, bytes.len()) };
    arenas[i].cur = (at + bytes.len() as u64 + 15) & !15;
    Some(at)
}

const SCRATCH: [Register; 15] = [
    Register::RAX,
    Register::RCX,
    Register::RDX,
    Register::RBX,
    Register::RBP,
    Register::RSI,
    Register::RDI,
    Register::R8,
    Register::R9,
    Register::R10,
    Register::R11,
    Register::R12,
    Register::R13,
    Register::R14,
    Register::R15,
];

fn is_gpr64(r: Register) -> bool {
    r == Register::None || r.is_gpr64()
}

fn displ_size(d: i64) -> u32 {
    if d == 0 {
        0
    } else if i8::try_from(d).is_ok() {
        1
    } else {
        8
    }
}

/// The instructions of the trampoline for the FS-relative `insn`, without
/// the jump back. None when the form is not supported.
pub(crate) fn build(insn: &Instruction, slot: u32) -> Option<Vec<Instruction>> {
    if insn.is_ip_rel_memory_operand() || insn.stack_pointer_increment() != 0 {
        return None;
    }
    let base = insn.memory_base();
    let index = insn.memory_index();
    if !is_gpr64(base) || !is_gpr64(index) {
        return None;
    }
    let tp = MemoryOperand::new(Register::None, Register::None, 1, slot as i64, 4, false, Register::GS);
    // A 64-bit destination register that the address does not use holds
    // the thread pointer itself.
    let fast = insn.code() == Code::Mov_r64_rm64 && insn.op0_register() != base && insn.op0_register() != index;
    let mut out = Vec::new();
    let (scratch, rsp_shift) = if fast {
        (insn.op0_register(), 0)
    } else {
        let mut info = InstructionInfoFactory::new();
        let used: Vec<Register> = info.info(insn).used_registers().iter().map(|u| u.register().full_register()).collect();
        let s = *SCRATCH.iter().find(|r| !used.contains(r) && **r != base && **r != index)?;
        let skip = MemoryOperand::with_base_displ(Register::RSP, -STACK_SKIP);
        out.push(Instruction::with2(Code::Lea_r64_m, Register::RSP, skip).ok()?);
        out.push(Instruction::with1(Code::Push_r64, s).ok()?);
        (s, STACK_SKIP + 8)
    };
    out.push(Instruction::with2(Code::Mov_r64_rm64, scratch, tp).ok()?);
    let mut disp = insn.memory_displacement64() as i64;
    if base == Register::RSP {
        disp += rsp_shift;
    }
    let mut new = *insn;
    new.set_segment_prefix(Register::None);
    match (base != Register::None, index != Register::None) {
        (false, false) => {
            new.set_memory_base(scratch);
            new.set_memory_index(Register::None);
        }
        (true, false) => {
            new.set_memory_base(base);
            new.set_memory_index(scratch);
            new.set_memory_index_scale(1);
        }
        (false, true) => {
            new.set_memory_base(scratch);
        }
        (true, true) => {
            let sum = if base == Register::RSP {
                MemoryOperand::with_base_index(Register::RSP, scratch)
            } else {
                MemoryOperand::with_base_index(scratch, base)
            };
            out.push(Instruction::with2(Code::Lea_r64_m, scratch, sum).ok()?);
            new.set_memory_base(scratch);
        }
    }
    new.set_memory_displacement64(disp as u64);
    new.set_memory_displ_size(displ_size(disp));
    out.push(new);
    if !fast {
        out.push(Instruction::with1(Code::Pop_r64, scratch).ok()?);
        let back = MemoryOperand::with_base_displ(Register::RSP, STACK_SKIP);
        out.push(Instruction::with2(Code::Lea_r64_m, Register::RSP, back).ok()?);
    }
    Some(out)
}

fn encode(list: &[Instruction], ip: u64) -> Option<Vec<u8>> {
    BlockEncoder::encode(64, InstructionBlock::new(list, ip), BlockEncoderOptions::NONE).ok().map(|r| r.code_buffer)
}

/// Move every FS-relative instruction of `fs` (decoded from `code`, which
/// starts at guest address `base`) into a trampoline.
///
/// # Safety
/// `code` must be writable guest code that no thread executes yet.
pub unsafe fn rewrite(base: u64, code: &mut [u8], fs: &[Instruction]) -> TlsCounts {
    let slot = crate::win::guest_tp_slot();
    let mut counts = TlsCounts::default();
    let mut traps = Vec::new();
    for insn in fs {
        let ip = insn.ip();
        let len = insn.len();
        let Some(mut list) = build(insn, slot) else {
            log::warn!("[tls] cannot rewrite {insn} at {ip:#x}");
            counts.failed += 1;
            continue;
        };
        let Ok(back) = Instruction::with_branch(Code::Jmp_rel32_64, ip + len as u64) else {
            counts.failed += 1;
            continue;
        };
        list.push(back);
        let Some(at) = place(ip, MAX_TRAMPOLINE, |at| encode(&list, at)) else {
            log::warn!("[tls] cannot place the trampoline of {insn} at {ip:#x}");
            counts.failed += 1;
            continue;
        };
        let off = (ip - base) as usize;
        if len >= 5 {
            let rel = at.wrapping_sub(ip + 5) as i64;
            code[off] = 0xE9;
            code[off + 1..off + 5].copy_from_slice(&(rel as i32).to_le_bytes());
            code[off + 5..off + len].fill(0xCC);
            counts.jumps += 1;
        } else {
            code[off] = 0xCC;
            traps.push((ip, at));
            counts.traps += 1;
        }
    }
    if !traps.is_empty() {
        let mut t = TRAPS.write().unwrap_or_else(|e| e.into_inner());
        t.get_or_insert_with(HashMap::new).extend(traps);
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced_x86::{Decoder, DecoderOptions, Formatter, IntelFormatter};

    fn text(bytes: &[u8]) -> Vec<String> {
        let insn = Decoder::with_ip(64, bytes, 0x1000, DecoderOptions::NONE).decode();
        let list = build(&insn, 0x1488).expect("supported");
        let mut f = IntelFormatter::new();
        list.iter()
            .map(|i| {
                let mut s = String::new();
                f.format(i, &mut s);
                s
            })
            .collect()
    }

    #[test]
    fn load_into_a_free_register_needs_no_scratch() {
        // mov rax, fs:[0x28]
        assert_eq!(text(&[0x64, 0x48, 0x8B, 0x04, 0x25, 0x28, 0, 0, 0]), ["mov rax,gs:[1488h]", "mov rax,[rax+28h]"]);
    }

    #[test]
    fn register_offset_uses_a_saved_scratch() {
        // mov eax, fs:[rax]
        assert_eq!(
            text(&[0x64, 0x8B, 0x00]),
            ["lea rsp,[rsp-80h]", "push rcx", "mov rcx,gs:[1488h]", "mov eax,[rax+rcx]", "pop rcx", "lea rsp,[rsp+80h]"]
        );
    }

    #[test]
    fn canary_compare_keeps_its_operands() {
        // sub rdx, fs:[0x28]
        let t = text(&[0x64, 0x48, 0x2B, 0x14, 0x25, 0x28, 0, 0, 0]);
        assert_eq!(t[2], "mov rax,gs:[1488h]");
        assert_eq!(t[3], "sub rdx,[rax+28h]");
    }

    /// Rewritten accesses run against the thread pointer of the guest
    /// thread, through a jump and through the INT3 path.
    #[test]
    fn rewritten_code_reads_the_guest_thread_block() {
        use crate::exec;
        exec::init();
        #[rustfmt::skip]
        let code: [u8; 13] = [
            0x64, 0x48, 0x8B, 0x04, 0x25, 0x08, 0, 0, 0, // mov rax, fs:[8]
            0x64, 0x03, 0x07,                            // add eax, fs:[rdi]
            0xC3,                                        // ret
        ];
        let page = ps5vm::os::map_anon(0x1000, ps5vm::MemProtect::RWX).unwrap();
        // SAFETY: the test's own page.
        let buf = unsafe { std::slice::from_raw_parts_mut(page as *mut u8, 0x1000) };
        buf[..code.len()].copy_from_slice(&code);
        let c = unsafe { crate::code_patch::patch_guest_code(page, code.len(), crate::code_patch::Replace { monitorx: false, sse4a: false }) };
        assert_eq!(c.fs_to_gs, 2);
        assert_eq!(buf[0], 0xE9);
        assert_eq!(buf[9], 0xCC);
        exec::add_guest_code_range(page, 0x1000, "tls-test");
        let block: Box<[u64; 4]> = Box::new([0, 0x1111_0000_0000_0005, 7, 0]);
        let top = 0x5E00_0000_0000u64;
        exec::map_guest_stack(top, 0x10000).unwrap();
        let r = std::thread::spawn(move || {
            exec::bind_thread_pointer(block.as_ptr() as u64);
            exec::enter_guest(&exec::call_regs(page, top, [16, 0, 0, 0, 0, 0]))
        })
        .join()
        .unwrap();
        assert_eq!(r, exec::GuestExit::Returned(12));
        exec::unmap_guest_stack(top, 0x10000);
    }
}
