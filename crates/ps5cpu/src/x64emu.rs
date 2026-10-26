// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Emulation of guest instructions the host CPU may not implement.
//!
//! PS5 code targets the console's AMD CPU. On a host without some of its
//! extensions the instruction raises SIGILL; the handler emulates it on the
//! saved context and resumes after it:
//!
//!   * SSE4a `EXTRQ` / `INSERTQ` (immediate and register forms);
//!   * `MONITORX` (no-op) / `MWAITX` (yield);
//!   * SHA-NI (`SHA1RNDS4`, `SHA1NEXTE`, `SHA1MSG1/2`, `SHA256RNDS2`,
//!     `SHA256MSG1/2`);
//!   * `RDPID` (TSC_AUX through RDTSCP) and `CLWB` (as CLFLUSH);
//!   * SSE4a `MOVNTSS` / `MOVNTSD` (plain stores; normally already
//!     replaced at load time, see `code_patch`);
//!   * ADX `ADCX` / `ADOX` (Intel before Broadwell);
//!   * VEX.128 `VRSQRTPS` carrying a cleared reserved VEX.vvvv bit — the
//!     mark of an instruction set aside for exact emulation. RSQRT results
//!     differ between vendors; the emulated one is the correctly rounded
//!     reciprocal square root, which meets the instruction's error bound.

/// Register state an instruction is emulated on.
pub trait EmuContext {
    fn rip(&self) -> u64;
    /// Instruction bytes at RIP (up to 15).
    fn code(&self) -> &[u8];
    fn advance(&mut self, len: usize);
    /// General register by hardware number (rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8..r15).
    fn gpr(&mut self, index: u8) -> &mut u64;
    fn xmm(&mut self, index: u8) -> Option<&mut [u8; 16]>;
    /// Zero bits 255:128 of YMM `index` (VEX.128 destination).
    fn clear_upper_ymm(&mut self, index: u8);
    /// RFLAGS.
    fn flags(&mut self) -> &mut u64;
}

/// Emulated instructions by family (see [`emulated_counts`]).
static EMULATED: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];
const FAMILIES: [&str; 5] = ["rsqrt", "monitorx", "sse4a", "sha", "other"];

/// Instructions emulated so far, by family (rsqrt, monitorx, sse4a, sha,
/// other: rdpid/clwb/movnt/adx), families that occurred only.
pub fn emulated_counts() -> Vec<(&'static str, u64)> {
    FAMILIES
        .iter()
        .zip(EMULATED.iter())
        .map(|(n, c)| (*n, c.load(std::sync::atomic::Ordering::Relaxed)))
        .filter(|c| c.1 != 0)
        .collect()
}

type Words = [u32; 4];

fn words(x: &[u8; 16]) -> Words {
    std::array::from_fn(|i| u32::from_le_bytes(x[i * 4..i * 4 + 4].try_into().unwrap()))
}

fn store_words(x: &mut [u8; 16], w: &Words) {
    for (i, v) in w.iter().enumerate() {
        x[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
}

fn qwords(x: &[u8; 16]) -> [u64; 2] {
    [u64::from_le_bytes(x[..8].try_into().unwrap()), u64::from_le_bytes(x[8..].try_into().unwrap())]
}

fn store_qwords(x: &mut [u8; 16], q: &[u64; 2]) {
    x[..8].copy_from_slice(&q[0].to_le_bytes());
    x[8..].copy_from_slice(&q[1].to_le_bytes());
}

/// Bits `[index, index + length)` of `value` (length 0 means 64).
pub fn extract_bit_field(value: u64, length: u32, index: u32) -> u64 {
    let mut length = length & 0x3F;
    let index = index & 0x3F;
    if length == 0 {
        length = 64;
    }
    length = length.min(64 - index);
    let mask = if length == 64 { u64::MAX } else { (1u64 << length) - 1 };
    (value >> index) & mask
}

/// `dst` with bits `[index, index + length)` replaced by the low bits of `src`.
pub fn insert_bit_field(dst: u64, src: u64, length: u32, index: u32) -> u64 {
    let mut length = length & 0x3F;
    let index = index & 0x3F;
    if length == 0 {
        length = 64;
    }
    length = length.min(64 - index);
    let mask = if length == 64 { u64::MAX } else { (1u64 << length) - 1 };
    (dst & !(mask << index)) | ((src & mask) << index)
}

// --- SHA-NI -----------------------------------------------------------------

fn sha1_msg1(d: &mut Words, s: &Words) {
    let (w0, w1, w2, w3, w4, w5) = (d[3], d[2], d[1], d[0], s[3], s[2]);
    *d = [w5 ^ w3, w4 ^ w2, w3 ^ w1, w2 ^ w0];
}

fn sha1_msg2(d: &mut Words, s: &Words) {
    let (w13, w14, w15) = (s[2], s[1], s[0]);
    let w16 = (d[3] ^ w13).rotate_left(1);
    let w17 = (d[2] ^ w14).rotate_left(1);
    let w18 = (d[1] ^ w15).rotate_left(1);
    let w19 = (d[0] ^ w16).rotate_left(1);
    *d = [w19, w18, w17, w16];
}

fn sha1_nexte(d: &mut Words, s: &Words) {
    let tmp = d[3].rotate_left(30);
    *d = [s[0], s[1], s[2], s[3].wrapping_add(tmp)];
}

fn sha1_rnds4(d: &mut Words, s: &Words, imm: u8) {
    let group = imm & 3;
    let k = [0x5a82_7999u32, 0x6ed9_eba1, 0x8f1b_bcdc, 0xca62_c1d6][group as usize];
    let f = |b: u32, c: u32, dd: u32| match group {
        0 => (b & c) ^ (!b & dd),
        2 => (b & c) ^ (b & dd) ^ (c & dd),
        _ => b ^ c ^ dd,
    };
    let w = [s[3], s[2], s[1], s[0]];
    let (mut a, mut b, mut c, mut dd, mut e) = (d[3], d[2], d[1], d[0], 0u32);
    for (round, wi) in w.iter().enumerate() {
        let mut term = f(b, c, dd).wrapping_add(a.rotate_left(5)).wrapping_add(*wi).wrapping_add(k);
        if round > 0 {
            term = term.wrapping_add(e);
        }
        e = dd;
        dd = c;
        c = b.rotate_left(30);
        b = a;
        a = term;
    }
    *d = [dd, c, b, a];
}

fn sigma0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}
fn sigma1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}
fn sum0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}
fn sum1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}

fn sha256_msg1(d: &mut Words, s: &Words) {
    let (w4, w3, w2, w1, w0) = (s[0], d[3], d[2], d[1], d[0]);
    *d = [w0.wrapping_add(sigma0(w1)), w1.wrapping_add(sigma0(w2)), w2.wrapping_add(sigma0(w3)), w3.wrapping_add(sigma0(w4))];
}

fn sha256_msg2(d: &mut Words, s: &Words) {
    let (w14, w15) = (s[2], s[3]);
    let w16 = d[0].wrapping_add(sigma1(w14));
    let w17 = d[1].wrapping_add(sigma1(w15));
    let w18 = d[2].wrapping_add(sigma1(w16));
    let w19 = d[3].wrapping_add(sigma1(w17));
    *d = [w16, w17, w18, w19];
}

fn sha256_rnds2(d: &mut Words, s: &Words, x0: &Words) {
    let (mut a, mut b, mut c, mut dd) = (s[3], s[2], d[3], d[2]);
    let (mut e, mut f, mut g, mut h) = (s[1], s[0], d[1], d[0]);
    for wk in &x0[..2] {
        let t1 = ((e & f) ^ (!e & g)).wrapping_add(sum1(e)).wrapping_add(*wk).wrapping_add(h);
        let t2 = ((a & b) ^ (a & c) ^ (b & c)).wrapping_add(sum0(a));
        h = g;
        g = f;
        f = e;
        e = t1.wrapping_add(dd);
        dd = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    *d = [f, e, b, a];
}

struct ShaInsn {
    escape: u8,
    opcode: u8,
    imm8: u8,
    rex: u8,
    modrm_offset: usize,
    length: usize,
}

fn decode_sha(code: &[u8]) -> Option<ShaInsn> {
    let at = |i: usize| code.get(i).copied();
    let (rex, mut off) = if at(0)? & 0xF0 == 0x40 { (at(0)?, 1) } else { (0, 0) };
    if at(off)? != 0x0F {
        return None;
    }
    let (escape, opcode) = match (at(off + 1)?, at(off + 2)?) {
        (0x38, op @ 0xC8..=0xCD) => (0x38, op),
        (0x3A, 0xCC) => (0x3A, 0xCC),
        _ => return None,
    };
    off += 3;
    let modrm_offset = off;
    let modrm = at(off)?;
    let (md, rm) = (modrm >> 6, modrm & 7);
    let mut end = off + 1;
    if md != 3 {
        let mut sib_base = 0xFF;
        if rm == 4 {
            sib_base = at(end)? & 7;
            end += 1;
        }
        if md == 0 && (rm == 5 || (rm == 4 && sib_base == 5)) {
            end += 4;
        } else if md == 1 {
            end += 1;
        } else if md == 2 {
            end += 4;
        }
    }
    let mut imm8 = 0;
    if escape == 0x3A {
        imm8 = at(end)?;
        end += 1;
    }
    Some(ShaInsn { escape, opcode, imm8, rex, modrm_offset, length: end })
}

fn reg_index(modrm: u8, rex: u8, reg_field: bool) -> u8 {
    if reg_field {
        ((modrm >> 3) & 7) | ((rex & 4) << 1)
    } else {
        (modrm & 7) | ((rex & 1) << 3)
    }
}

fn sha_memory_address(ctx: &mut dyn EmuContext, insn: &ShaInsn) -> Option<u64> {
    let code: Vec<u8> = ctx.code().to_vec();
    let at = |i: usize| code.get(i).copied();
    let disp32 = |i: usize| Some(i32::from_le_bytes([at(i)?, at(i + 1)?, at(i + 2)?, at(i + 3)?]) as i64 as u64);
    let modrm = at(insn.modrm_offset)?;
    let (md, rm) = (modrm >> 6, modrm & 7);
    if md == 3 {
        return None;
    }
    let mut off = insn.modrm_offset + 1;
    let mut result: u64;
    if rm == 4 {
        let sib = at(off)?;
        off += 1;
        let (scale, index_low, base_low) = (sib >> 6, (sib >> 3) & 7, sib & 7);
        let has_index = index_low != 4 || insn.rex & 2 != 0;
        let has_base = md != 0 || base_low != 5;
        result = 0;
        if has_base {
            result = result.wrapping_add(*ctx.gpr(base_low | ((insn.rex & 1) << 3)));
        }
        if has_index {
            result = result.wrapping_add(*ctx.gpr(index_low | ((insn.rex & 2) << 2)) << scale);
        }
        if !has_base {
            result = result.wrapping_add(disp32(off)?);
            off += 4;
        }
    } else if md == 0 && rm == 5 {
        result = (ctx.rip() + insn.length as u64).wrapping_add(disp32(off)?);
        off += 4;
    } else {
        result = *ctx.gpr(rm | ((insn.rex & 1) << 3));
    }
    if md == 1 {
        result = result.wrapping_add(at(off)? as i8 as i64 as u64);
    } else if md == 2 {
        result = result.wrapping_add(disp32(off)?);
    }
    Some(result)
}

fn try_sha(ctx: &mut dyn EmuContext) -> bool {
    let Some(insn) = decode_sha(ctx.code()) else { return false };
    let modrm = ctx.code()[insn.modrm_offset];
    let dest_index = reg_index(modrm, insn.rex, true);
    let (Some(mut dest), Some(x0)) = (ctx.xmm(dest_index).map(|x| words(x)), ctx.xmm(0).map(|x| words(x))) else { return false };
    let src = if modrm & 0xC0 == 0xC0 {
        let Some(s) = ctx.xmm(reg_index(modrm, insn.rex, false)).map(|x| words(x)) else { return false };
        s
    } else {
        let Some(addr) = sha_memory_address(ctx, &insn) else { return false };
        let mut b = [0u8; 16];
        unsafe { std::ptr::copy_nonoverlapping(addr as *const u8, b.as_mut_ptr(), 16) };
        words(&b)
    };
    match (insn.escape, insn.opcode) {
        (0x3A, 0xCC) => sha1_rnds4(&mut dest, &src, insn.imm8),
        (_, 0xC8) => sha1_nexte(&mut dest, &src),
        (_, 0xC9) => sha1_msg1(&mut dest, &src),
        (_, 0xCA) => sha1_msg2(&mut dest, &src),
        (_, 0xCB) => sha256_rnds2(&mut dest, &src, &x0),
        (_, 0xCC) => sha256_msg1(&mut dest, &src),
        (_, 0xCD) => sha256_msg2(&mut dest, &src),
        _ => return false,
    }
    store_words(ctx.xmm(dest_index).unwrap(), &dest);
    ctx.advance(insn.length);
    true
}

// --- SSE4a ------------------------------------------------------------------

fn try_sse4a(ctx: &mut dyn EmuContext) -> bool {
    let code = ctx.code();
    let at = |i: usize| code.get(i).copied().unwrap_or(0);
    let prefix = at(0);
    if prefix != 0x66 && prefix != 0xF2 {
        return false;
    }
    let mut off = 1;
    let mut rex = 0;
    if at(off) & 0xF0 == 0x40 {
        rex = at(off);
        off += 1;
    }
    if at(off) != 0x0F {
        return false;
    }
    let register_form = at(off + 1) == 0x79;
    if at(off + 1) != 0x78 && !register_form {
        return false;
    }
    let modrm = at(off + 2);
    if modrm & 0xC0 != 0xC0 {
        return false;
    }
    let reg = ((modrm >> 3) & 7) | ((rex & 4) << 1);
    let rm = (modrm & 7) | ((rex & 1) << 3);
    // Immediate EXTRQ encodes its destination in r/m; the other forms use reg.
    let dest_index = if prefix == 0x66 && !register_form { rm } else { reg };
    let (imm_len, imm_idx) = (at(off + 3), at(off + 4));
    let (Some(mut dest), Some(source)) = (ctx.xmm(dest_index).map(|x| qwords(x)), ctx.xmm(rm).map(|x| qwords(x))) else { return false };
    let mut len = off + 3;
    let (length, index) = if register_form {
        // EXTRQ controls are in the low quadword, INSERTQ's in the upper one.
        let controls = source[if prefix == 0x66 { 0 } else { 1 }];
        (controls as u8, (controls >> 8) as u8)
    } else {
        len += 2;
        (imm_len, imm_idx)
    };
    if prefix == 0x66 {
        dest = [extract_bit_field(dest[0], length as u32, index as u32), 0];
    } else {
        dest[0] = insert_bit_field(dest[0], source[0], length as u32, index as u32);
    }
    store_qwords(ctx.xmm(dest_index).unwrap(), &dest);
    ctx.advance(len);
    true
}

// --- MONITORX / MWAITX --------------------------------------------------------

fn try_monitorx_mwaitx(ctx: &mut dyn EmuContext) -> bool {
    let code = ctx.code();
    if code.len() < 3 || code[0] != 0x0F || code[1] != 0x01 || (code[2] != 0xFA && code[2] != 0xFB) {
        return false;
    }
    if code[2] == 0xFB {
        std::thread::yield_now();
    }
    ctx.advance(3);
    true
}

// --- RSQRT ---------------------------------------------------------------------

/// Reciprocal square root of one f32 lane (bit pattern in, bit pattern out).
pub fn reciprocal_square_root(bits: u32) -> u32 {
    let magnitude = bits & 0x7FFF_FFFF;
    if magnitude & 0x7F80_0000 == 0 {
        // Denormals count as signed zero regardless of MXCSR.DAZ.
        return (bits & 0x8000_0000) | 0x7F80_0000;
    }
    if magnitude > 0x7F80_0000 {
        return bits | 0x0040_0000; // quiet NaN
    }
    if bits & 0x8000_0000 != 0 {
        return 0xFFC0_0000;
    }
    if magnitude == 0x7F80_0000 {
        return 0;
    }
    let r = 1.0f64 / (f32::from_bits(bits) as f64).sqrt();
    (r as f32).to_bits()
}

fn try_rsqrt(ctx: &mut dyn EmuContext) -> bool {
    let code = ctx.code();
    let at = |i: usize| code.get(i).copied().unwrap_or(0);
    let (prefix_size, dest_ext, src_ext) = if at(0) == 0xC5 && at(1) & 0x7F == 0x70 {
        (2, (!at(1) & 0x80) >> 4, 0)
    } else if at(0) == 0xC4 && at(1) & 0x1F == 1 && at(2) & 0x7F == 0x70 {
        (3, (!at(1) & 0x80) >> 4, (!at(1) & 0x20) >> 2)
    } else {
        return false;
    };
    if at(prefix_size) != 0x52 || at(prefix_size + 1) & 0xC0 != 0xC0 {
        return false;
    }
    let modrm = at(prefix_size + 1);
    let dest = ((modrm >> 3) & 7) | dest_ext;
    let src = (modrm & 7) | src_ext;
    let Some(mut w) = ctx.xmm(src).map(|x| words(x)) else { return false };
    if ctx.xmm(dest).is_none() {
        return false;
    }
    for v in &mut w {
        *v = reciprocal_square_root(*v);
    }
    store_words(ctx.xmm(dest).unwrap(), &w);
    ctx.clear_upper_ymm(dest);
    ctx.advance(prefix_size + 2);
    true
}

// --- RDPID / CLWB ----------------------------------------------------------------

fn read_tsc_aux() -> Option<u32> {
    let ext = std::arch::x86_64::__cpuid(0x8000_0001);
    if ext.edx & (1 << 27) == 0 {
        return None;
    }
    let mut aux = 0u32;
    unsafe { std::arch::x86_64::__rdtscp(&mut aux) };
    Some(aux)
}

/// Hardware number and width (8 or 4 bytes) of a 64/32-bit general register.
fn gpr_number(reg: iced_x86::Register) -> Option<(usize, u8)> {
    use iced_x86::Register::*;
    const R64: [iced_x86::Register; 16] = [RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15];
    const R32: [iced_x86::Register; 16] = [EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI, R8D, R9D, R10D, R11D, R12D, R13D, R14D, R15D];
    R64.iter().position(|&r| r == reg).map(|i| (i, 8)).or_else(|| R32.iter().position(|&r| r == reg).map(|i| (i, 4)))
}

/// Linear address of the memory operand of `insn` (GS-relative operands
/// add the guest thread pointer).
fn memory_address(ctx: &mut dyn EmuContext, insn: &iced_x86::Instruction) -> Option<u64> {
    use iced_x86::Register;
    let mut regs = [0u64; 16];
    for (i, r) in regs.iter_mut().enumerate() {
        *r = *ctx.gpr(i as u8);
    }
    let operand = (0..insn.op_count()).find(|&i| insn.op_kind(i) == iced_x86::OpKind::Memory)?;
    let offset = insn.virtual_address(operand, 0, |reg, _, _| match reg {
        Register::FS | Register::GS | Register::ES | Register::CS | Register::SS | Register::DS => Some(0),
        _ => gpr_number(reg).map(|(n, w)| if w == 4 { regs[n] & 0xFFFF_FFFF } else { regs[n] }),
    })?;
    Some(match insn.memory_segment() {
        Register::GS => offset.wrapping_add(crate::tls_patch::thread_pointer()),
        _ => offset,
    })
}

const CF: u64 = 1;
const OF: u64 = 1 << 11;

/// ADCX / ADOX: `dst + src + carry` with the carry in and out of CF (ADCX)
/// or OF (ADOX); the other flags are unchanged.
fn add_with_flag(ctx: &mut dyn EmuContext, insn: &iced_x86::Instruction, flag: u64) -> bool {
    use iced_x86::OpKind;
    let Some((dst, width)) = gpr_number(insn.op0_register()) else { return false };
    let src = match insn.op1_kind() {
        OpKind::Register => {
            let Some((n, _)) = gpr_number(insn.op1_register()) else { return false };
            *ctx.gpr(n as u8)
        }
        OpKind::Memory => {
            let Some(addr) = memory_address(ctx, insn) else { return false };
            // SAFETY: the guest instruction itself reads this operand.
            unsafe {
                if width == 8 {
                    std::ptr::read_unaligned(addr as *const u64)
                } else {
                    std::ptr::read_unaligned(addr as *const u32) as u64
                }
            }
        }
        _ => return false,
    };
    let carry_in = (*ctx.flags() & flag != 0) as u64;
    let a = *ctx.gpr(dst as u8);
    let (value, carry) = if width == 8 {
        let (s1, c1) = a.overflowing_add(src);
        let (s2, c2) = s1.overflowing_add(carry_in);
        (s2, c1 || c2)
    } else {
        let wide = (a & 0xFFFF_FFFF) + (src & 0xFFFF_FFFF) + carry_in;
        (wide & 0xFFFF_FFFF, wide >> 32 != 0)
    };
    *ctx.gpr(dst as u8) = value;
    let f = ctx.flags();
    *f = if carry { *f | flag } else { *f & !flag };
    true
}

fn try_cpu_extensions(ctx: &mut dyn EmuContext) -> bool {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic, OpKind, Register};
    let code = ctx.code().to_vec();
    let rip = ctx.rip();
    let insn = Decoder::with_ip(64, &code, rip, DecoderOptions::NONE).decode();
    if insn.is_invalid() {
        return false;
    }
    match insn.mnemonic() {
        Mnemonic::Rdpid if insn.op0_kind() == OpKind::Register => {
            let Some((n, _)) = gpr_number(insn.op0_register()) else { return false };
            let Some(aux) = read_tsc_aux() else { return false };
            *ctx.gpr(n as u8) = aux as u64;
        }
        Mnemonic::Clwb if insn.op0_kind() == OpKind::Memory => {
            let mut regs = [0u64; 16];
            for (i, r) in regs.iter_mut().enumerate() {
                *r = *ctx.gpr(i as u8);
            }
            let Some(addr) = insn.virtual_address(0, 0, |reg, _, _| {
                match reg {
                    Register::FS | Register::GS | Register::ES | Register::CS | Register::SS | Register::DS => Some(0),
                    _ => gpr_number(reg).map(|(n, w)| if w == 4 { regs[n] & 0xFFFF_FFFF } else { regs[n] }),
                }
            }) else {
                return false;
            };
            // Keep FS/GS-relative addressing instead of treating the offset as a pointer.
            unsafe {
                match insn.memory_segment() {
                    Register::GS => std::arch::asm!("clflush gs:[{0}]", in(reg) addr, options(nostack)),
                    Register::FS => std::arch::asm!("clflush fs:[{0}]", in(reg) addr, options(nostack)),
                    _ => std::arch::x86_64::_mm_clflush(addr as *const u8),
                }
            }
        }
        Mnemonic::Movntss | Mnemonic::Movntsd if insn.op0_kind() == OpKind::Memory => {
            let reg = insn.op1_register();
            if !reg.is_xmm() {
                return false;
            }
            let Some(addr) = memory_address(ctx, &insn) else { return false };
            let Some(x) = ctx.xmm(reg.number() as u8).map(|x| *x) else { return false };
            let n = if insn.mnemonic() == Mnemonic::Movntss { 4 } else { 8 };
            // SAFETY: the guest instruction itself stores to this operand.
            unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), addr as *mut u8, n) };
        }
        Mnemonic::Adcx => {
            if !add_with_flag(ctx, &insn, CF) {
                return false;
            }
        }
        Mnemonic::Adox => {
            if !add_with_flag(ctx, &insn, OF) {
                return false;
            }
        }
        _ => return false,
    }
    ctx.advance(insn.len());
    true
}

/// Emulate the instruction at the context's RIP; false when it is not one
/// of the supported instructions.
pub fn try_emulate(ctx: &mut dyn EmuContext) -> bool {
    let family = if try_rsqrt(ctx) {
        0
    } else if try_monitorx_mwaitx(ctx) {
        1
    } else if try_sse4a(ctx) {
        2
    } else if try_sha(ctx) {
        3
    } else if try_cpu_extensions(ctx) {
        4
    } else {
        return false;
    };
    EMULATED[family].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    true
}

/// [`EmuContext`] over a Linux signal context.
pub struct SignalContext {
    uc: *mut crate::ctx::ucontext_t,
}

const GREG_OF: [crate::ctx::c_int; 16] = [
    crate::ctx::REG_RAX,
    crate::ctx::REG_RCX,
    crate::ctx::REG_RDX,
    crate::ctx::REG_RBX,
    crate::ctx::REG_RSP,
    crate::ctx::REG_RBP,
    crate::ctx::REG_RSI,
    crate::ctx::REG_RDI,
    crate::ctx::REG_R8,
    crate::ctx::REG_R9,
    crate::ctx::REG_R10,
    crate::ctx::REG_R11,
    crate::ctx::REG_R12,
    crate::ctx::REG_R13,
    crate::ctx::REG_R14,
    crate::ctx::REG_R15,
];

impl SignalContext {
    /// # Safety
    /// `uc` must be the context of a signal being handled on this thread.
    pub unsafe fn new(uc: *mut crate::ctx::ucontext_t) -> Option<Self> {
        (!uc.is_null()).then_some(Self { uc })
    }
}

impl EmuContext for SignalContext {
    fn rip(&self) -> u64 {
        unsafe { (*self.uc).uc_mcontext.gregs[crate::ctx::REG_RIP as usize] as u64 }
    }

    fn code(&self) -> &[u8] {
        let rip = self.rip();
        // Instructions are at most 15 bytes; stop at the end of a readable page run.
        let len = if ps5vm::ranges::accessible(rip, 15, false) || crate::exec::is_guest_code_addr(rip + 14) {
            15
        } else {
            (0x1000 - (rip & 0xFFF)).min(15)
        };
        unsafe { std::slice::from_raw_parts(rip as *const u8, len as usize) }
    }

    fn advance(&mut self, len: usize) {
        unsafe { (*self.uc).uc_mcontext.gregs[crate::ctx::REG_RIP as usize] += len as i64 };
    }

    fn gpr(&mut self, index: u8) -> &mut u64 {
        unsafe { &mut *((&mut (*self.uc).uc_mcontext.gregs[GREG_OF[index as usize] as usize]) as *mut i64 as *mut u64) }
    }

    fn xmm(&mut self, index: u8) -> Option<&mut [u8; 16]> {
        let fp = unsafe { (*self.uc).uc_mcontext.fpregs };
        if fp.is_null() || index >= 16 {
            return None;
        }
        Some(unsafe { &mut *((*fp)._xmm[index as usize].element.as_mut_ptr() as *mut [u8; 16]) })
    }

    fn flags(&mut self) -> &mut u64 {
        unsafe { &mut *((&mut (*self.uc).uc_mcontext.gregs[crate::ctx::REG_EFL as usize]) as *mut i64 as *mut u64) }
    }

    #[cfg(windows)]
    fn clear_upper_ymm(&mut self, index: u8) {
        if index < 16 {
            // SAFETY: the exception context the handler was given.
            unsafe { crate::win::clear_upper_ymm((*self.uc).native, index) };
        }
    }

    #[cfg(unix)]
    fn clear_upper_ymm(&mut self, index: u8) {
        // Linux signal frames use the XSAVE layout; an absent AVX component
        // already restores zeroes.
        let fp = unsafe { (*self.uc).uc_mcontext.fpregs } as *mut u8;
        if fp.is_null() || index >= 16 {
            return;
        }
        unsafe {
            let magic = std::ptr::read_unaligned(fp.add(464) as *const u32);
            let size = std::ptr::read_unaligned(fp.add(480) as *const u32);
            if magic != 0x4650_5853 || size < 832 {
                return;
            }
            let features = std::ptr::read_unaligned(fp.add(512) as *const u64);
            if features & 4 != 0 {
                std::ptr::write_bytes(fp.add(576 + index as usize * 16), 0, 16);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Context over a byte buffer and a register file.
    struct TestContext {
        code: Vec<u8>,
        rip: u64,
        gpr: [u64; 16],
        xmm: [[u8; 16]; 16],
        cleared: Vec<u8>,
        flags: u64,
    }

    impl TestContext {
        fn new(code: &[u8]) -> Self {
            Self { code: code.to_vec(), rip: 0, gpr: [0; 16], xmm: [[0; 16]; 16], cleared: Vec::new(), flags: 0x202 }
        }
        fn set_xmm(&mut self, i: usize, q: [u64; 2]) {
            store_qwords(&mut self.xmm[i], &q);
        }
        fn get_xmm(&self, i: usize) -> [u64; 2] {
            qwords(&self.xmm[i])
        }
    }

    impl EmuContext for TestContext {
        fn rip(&self) -> u64 {
            self.code.as_ptr() as u64 + self.rip
        }
        fn code(&self) -> &[u8] {
            &self.code[self.rip as usize..]
        }
        fn advance(&mut self, len: usize) {
            self.rip += len as u64;
        }
        fn gpr(&mut self, index: u8) -> &mut u64 {
            &mut self.gpr[index as usize]
        }
        fn xmm(&mut self, index: u8) -> Option<&mut [u8; 16]> {
            self.xmm.get_mut(index as usize)
        }
        fn clear_upper_ymm(&mut self, index: u8) {
            self.cleared.push(index);
        }
        fn flags(&mut self) -> &mut u64 {
            &mut self.flags
        }
    }

    const EXTRQ_XMM2_40_0: [u8; 6] = [0x66, 0x0F, 0x78, 0xC2, 0x28, 0x00];
    const EXTRQ_XMM2_64_0: [u8; 6] = [0x66, 0x0F, 0x78, 0xC2, 0x40, 0x00];
    const INSERTQ_XMM2_XMM0_32_0: [u8; 6] = [0xF2, 0x0F, 0x78, 0xD0, 0x20, 0x00];
    const EXTRQ_XMM10_40_0: [u8; 7] = [0x66, 0x41, 0x0F, 0x78, 0xC2, 0x28, 0x00];
    const INSERTQ_XMM10_XMM9_16_8: [u8; 7] = [0xF2, 0x45, 0x0F, 0x78, 0xD1, 0x10, 0x08];
    const INSERTQ_XMM10_XMM9: [u8; 5] = [0xF2, 0x45, 0x0F, 0x79, 0xD1];
    const INSERTQ_XMM9_XMM9: [u8; 5] = [0xF2, 0x45, 0x0F, 0x79, 0xC9];

    #[test]
    fn extrq_immediate() {
        let mut c = TestContext::new(&EXTRQ_XMM2_40_0);
        c.set_xmm(2, [0xaabb_ccdd_eeff_1122, 0x1234]);
        assert!(try_emulate(&mut c));
        assert_eq!(c.get_xmm(2), [0x0000_00dd_eeff_1122, 0], "keeps bits [39:0], zero-extends");
        assert_eq!(c.rip, 6);
    }

    #[test]
    fn extrq_full_width() {
        let mut c = TestContext::new(&EXTRQ_XMM2_64_0);
        c.set_xmm(2, [0x0123_4567_89ab_cdef, 0x55]);
        assert!(try_emulate(&mut c));
        assert_eq!(c.get_xmm(2), [0x0123_4567_89ab_cdef, 0]);
    }

    #[test]
    fn insertq_immediate() {
        let mut c = TestContext::new(&INSERTQ_XMM2_XMM0_32_0);
        c.set_xmm(0, [0x1122_3344_5566_7788, 0]);
        c.set_xmm(2, [0x0123_4567_89ab_cdef, 0x99]);
        assert!(try_emulate(&mut c));
        assert_eq!(c.get_xmm(2), [0x0123_4567_5566_7788, 0x99]);
    }

    #[test]
    fn rex_high_register_extrq() {
        let mut c = TestContext::new(&EXTRQ_XMM10_40_0);
        c.set_xmm(10, [0xaabb_ccdd_eeff_1122, 0]);
        assert!(try_emulate(&mut c));
        assert_eq!(c.get_xmm(10), [0x0000_00dd_eeff_1122, 0]);
        assert_eq!(c.rip, 7);
    }

    #[test]
    fn rex_high_register_insertq_with_index() {
        let mut c = TestContext::new(&INSERTQ_XMM10_XMM9_16_8);
        c.set_xmm(9, [0x8877_6655_4433_2211, 0xdead_beef_cafe_babe]);
        c.set_xmm(10, [0xaaaa_aaaa_aaaa_aaaa, 0xbbbb_bbbb_bbbb_bbbb]);
        assert!(try_emulate(&mut c));
        assert_eq!(c.get_xmm(10), [0xaaaa_aaaa_aa22_11aa, 0xbbbb_bbbb_bbbb_bbbb]);
    }

    #[test]
    fn insertq_register() {
        for (code, aliased) in [(&INSERTQ_XMM10_XMM9[..], false), (&INSERTQ_XMM9_XMM9[..], true)] {
            for controls in [0xc4c8u64, 0xc0c0, 0xffc1] {
                let source = [0x0123_4567_89ab_cdef, 0xdead_beef_0000_0000 | controls];
                let destination = [0x8877_6655_4433_2211, 0xfedc_ba98_7654_3210];
                let initial = if aliased { source } else { destination };
                let mut expected = initial[0];
                let length = if controls & 63 == 0 { 64 } else { controls & 63 };
                let index = (controls >> 8) & 63;
                let mut bit = 0;
                while bit < length && index + bit < 64 {
                    let mask = 1u64 << (index + bit);
                    expected = (expected & !mask) | (((source[0] >> bit) & 1) << (index + bit));
                    bit += 1;
                }
                let mut c = TestContext::new(code);
                c.set_xmm(9, source);
                if !aliased {
                    c.set_xmm(10, destination);
                }
                assert!(try_emulate(&mut c));
                let out = c.get_xmm(if aliased { 9 } else { 10 });
                assert_eq!(out, [expected, initial[1]], "controls {controls:#x} aliased={aliased}");
            }
        }
    }

    #[test]
    fn mwaitx_and_monitorx() {
        for code in [[0x0F, 0x01, 0xFB], [0x0F, 0x01, 0xFA]] {
            let mut c = TestContext::new(&code);
            assert!(try_emulate(&mut c));
            assert_eq!(c.rip, 3);
        }
    }

    #[test]
    fn unknown_instruction_refused() {
        let mut c = TestContext::new(&[0x0F, 0x0B, 0xC3]);
        assert!(!try_emulate(&mut c));
        assert_eq!(c.rip, 0);
    }

    #[test]
    fn trapped_vrsqrtps() {
        // vrsqrtps xmm1, xmm0 with the cleared vvvv bit (c5 f8 -> c5 f0).
        let mut c = TestContext::new(&[0xC5, 0xF0, 0x52, 0xC8]);
        let input = [4.0f32.to_bits(), 0, 0xbf80_0000, 0x7f80_0000];
        store_words(&mut c.xmm[0], &input);
        assert!(try_emulate(&mut c));
        let out = words(&c.xmm[1]);
        assert_eq!(f32::from_bits(out[0]), 0.5);
        assert_eq!(out[1], 0x7f80_0000, "zero -> +inf");
        assert_eq!(out[2], 0xffc0_0000, "negative -> default NaN");
        assert_eq!(out[3], 0, "inf -> 0");
        assert_eq!(c.cleared, [1], "VEX.128 zeroes the upper YMM half");
        assert_eq!(c.rip, 4);
    }

    /// Run `code` (instruction under test + `ret`-free tail) natively on
    /// this host to cross-check the emulation where the CPU supports it.
    fn has_sse4a() -> bool {
        std::arch::x86_64::__cpuid(0x8000_0001).ecx & (1 << 6) != 0
    }

    #[test]
    fn sse4a_matches_hardware() {
        if !has_sse4a() {
            return;
        }
        // extrq/insertq through inline asm on the host, against the emulator.
        let cases: [(u64, u64, u32, u32); 4] =
            [(0xaabb_ccdd_eeff_1122, 0, 40, 0), (0x0123_4567_89ab_cdef, 0x1122_3344_5566_7788, 16, 8), (u64::MAX, 0, 0, 0), (0x8000_0000_0000_0001, 0xF, 4, 60)];
        for (dst, src, len, idx) in cases {
            let hw_insert: u64;
            let hw_extract: u64;
            unsafe {
                let mut x: std::arch::x86_64::__m128i = std::mem::transmute([dst, 0x77u64]);
                let s: std::arch::x86_64::__m128i = std::mem::transmute([src, (idx as u64) << 8 | len as u64]);
                std::arch::asm!("insertq {x}, {s}", x = inout(xmm_reg) x, s = in(xmm_reg) s);
                hw_insert = std::mem::transmute::<_, [u64; 2]>(x)[0];
                let mut y: std::arch::x86_64::__m128i = std::mem::transmute([dst, 0x77u64]);
                let c: std::arch::x86_64::__m128i = std::mem::transmute([(idx as u64) << 8 | len as u64, 0u64]);
                std::arch::asm!("extrq {y}, {c}", y = inout(xmm_reg) y, c = in(xmm_reg) c);
                hw_extract = std::mem::transmute::<_, [u64; 2]>(y)[0];
            }
            assert_eq!(insert_bit_field(dst, src, len, idx), hw_insert, "insertq {dst:#x} {src:#x} {len} {idx}");
            assert_eq!(extract_bit_field(dst, len, idx), hw_extract, "extrq {dst:#x} {len} {idx}");
        }
    }

    fn has_sha() -> bool {
        std::arch::x86_64::__cpuid_count(7, 0).ebx & (1 << 29) != 0
    }

    #[test]
    fn sha_ni_matches_hardware() {
        if !has_sha() {
            return;
        }
        use std::arch::x86_64::*;
        let a: [u32; 4] = [0x0123_4567, 0x89ab_cdef, 0xfedc_ba98, 0x7654_3210];
        let b: [u32; 4] = [0x1111_2222, 0x3333_4444, 0x5555_6666, 0x7777_8888];
        let k: [u32; 4] = [0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5];
        let emu = |code: &[u8], a: [u32; 4], b: [u32; 4], x0: [u32; 4]| {
            let mut c = TestContext::new(code);
            store_words(&mut c.xmm[1], &a);
            store_words(&mut c.xmm[2], &b);
            store_words(&mut c.xmm[0], &x0);
            assert!(try_emulate(&mut c), "{code:02x?}");
            words(&c.xmm[1])
        };
        unsafe {
            let va: __m128i = std::mem::transmute(a);
            let vb: __m128i = std::mem::transmute(b);
            let vk: __m128i = std::mem::transmute(k);
            let hw = |v: __m128i| std::mem::transmute::<__m128i, [u32; 4]>(v);
            // Encodings: 0f 38 c8..cd /r (xmm1, xmm2) and 0f 3a cc /r ib.
            assert_eq!(emu(&[0x0F, 0x38, 0xC8, 0xCA], a, b, k), hw(_mm_sha1nexte_epu32(va, vb)));
            assert_eq!(emu(&[0x0F, 0x38, 0xC9, 0xCA], a, b, k), hw(_mm_sha1msg1_epu32(va, vb)));
            assert_eq!(emu(&[0x0F, 0x38, 0xCA, 0xCA], a, b, k), hw(_mm_sha1msg2_epu32(va, vb)));
            assert_eq!(emu(&[0x0F, 0x38, 0xCB, 0xCA], a, b, k), hw(_mm_sha256rnds2_epu32(va, vb, vk)));
            assert_eq!(emu(&[0x0F, 0x38, 0xCC, 0xCA], a, b, k), hw(_mm_sha256msg1_epu32(va, vb)));
            assert_eq!(emu(&[0x0F, 0x38, 0xCD, 0xCA], a, b, k), hw(_mm_sha256msg2_epu32(va, vb)));
            assert_eq!(emu(&[0x0F, 0x3A, 0xCC, 0xCA, 0x00], a, b, k), hw(_mm_sha1rnds4_epu32(va, vb, 0)));
            assert_eq!(emu(&[0x0F, 0x3A, 0xCC, 0xCA, 0x01], a, b, k), hw(_mm_sha1rnds4_epu32(va, vb, 1)));
            assert_eq!(emu(&[0x0F, 0x3A, 0xCC, 0xCA, 0x02], a, b, k), hw(_mm_sha1rnds4_epu32(va, vb, 2)));
            assert_eq!(emu(&[0x0F, 0x3A, 0xCC, 0xCA, 0x03], a, b, k), hw(_mm_sha1rnds4_epu32(va, vb, 3)));
        }
    }

    #[test]
    fn sha_memory_operand() {
        let src = [0x1111_2222u32, 0x3333_4444, 0x5555_6666, 0x7777_8888];
        let mut mem = [0u8; 16];
        store_words(&mut mem, &src);
        // sha1msg1 xmm1, [rax + 8]
        let mut c = TestContext::new(&[0x0F, 0x38, 0xC9, 0x48, 0x08]);
        c.gpr[0] = mem.as_ptr() as u64 - 8;
        let a = [1u32, 2, 3, 4];
        store_words(&mut c.xmm[1], &a);
        assert!(try_emulate(&mut c));
        let mut expect = a;
        sha1_msg1(&mut expect, &src);
        assert_eq!(words(&c.xmm[1]), expect);
        assert_eq!(c.rip, 5);
    }

    #[test]
    fn rdpid_reads_tsc_aux() {
        // rdpid rax: f3 0f c7 f8
        let mut c = TestContext::new(&[0xF3, 0x0F, 0xC7, 0xF8]);
        if read_tsc_aux().is_none() {
            assert!(!try_emulate(&mut c));
            return;
        }
        c.gpr[0] = u64::MAX;
        assert!(try_emulate(&mut c));
        assert!(c.gpr[0] <= u32::MAX as u64);
        assert_eq!(c.rip, 4);
    }

    #[test]
    fn clwb_flushes_the_line() {
        let line = [0u8; 64];
        // clwb [rdi]: 66 0f ae 37
        let mut c = TestContext::new(&[0x66, 0x0F, 0xAE, 0x37]);
        c.gpr[7] = line.as_ptr() as u64;
        assert!(try_emulate(&mut c));
        assert_eq!(c.rip, 4);
    }

    #[test]
    fn movnt_scalar_stores() {
        let mut mem = [0xEEu8; 16];
        // movntsd [rdi], xmm3: f2 0f 2b 1f
        let mut c = TestContext::new(&[0xF2, 0x0F, 0x2B, 0x1F]);
        c.gpr[7] = mem.as_mut_ptr() as u64;
        c.set_xmm(3, [0x0807_0605_0403_0201, u64::MAX]);
        assert!(try_emulate(&mut c));
        assert_eq!(c.rip, 4);
        assert_eq!(&mem[..8], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(&mem[8..], &[0xEE; 8]);
        // movntss [rsi + r8*4 + 4], xmm9: f3 46 0f 2b 4c 86 04
        let mut c = TestContext::new(&[0xF3, 0x46, 0x0F, 0x2B, 0x4C, 0x86, 0x04]);
        c.gpr[6] = mem.as_mut_ptr() as u64;
        c.gpr[8] = 1;
        c.set_xmm(9, [0xDDCC_BBAA, 0]);
        assert!(try_emulate(&mut c));
        assert_eq!(c.rip, 7);
        assert_eq!(&mem[8..12], &[0xAA, 0xBB, 0xCC, 0xDD]);
        assert_eq!(&mem[12..], &[0xEE; 4]);
    }

    fn has_adx() -> bool {
        std::arch::x86_64::__cpuid_count(7, 0).ebx & (1 << 19) != 0
    }

    /// ADCX/ADOX against the host where it has ADX (flags in, value and
    /// flags out), for 64- and 32-bit operands and a memory source.
    #[test]
    fn adx_matches_hardware() {
        let cases: [(u64, u64, bool); 6] = [
            (1, 2, false),
            (u64::MAX, 1, false),
            (u64::MAX, 0, true),
            (0x8000_0000_0000_0000, 0x8000_0000_0000_0000, true),
            (0xFFFF_FFFF, 1, false),
            (0x1234_5678_FFFF_FFFF, 0xFFFF_FFFF, true),
        ];
        for (a, b, carry) in cases {
            for (code, flag, wide) in [
                (&[0x66u8, 0x48, 0x0F, 0x38, 0xF6, 0xC1][..], CF, true), // adcx rax, rcx
                (&[0xF3, 0x48, 0x0F, 0x38, 0xF6, 0xC1][..], OF, true),   // adox rax, rcx
                (&[0x66, 0x0F, 0x38, 0xF6, 0xC1][..], CF, false),        // adcx eax, ecx
                (&[0xF3, 0x0F, 0x38, 0xF6, 0xC1][..], OF, false),        // adox eax, ecx
            ] {
                let mut c = TestContext::new(code);
                c.gpr[0] = a;
                c.gpr[1] = b;
                c.flags = 0x202 | if carry { flag } else { 0 } | if flag == CF { OF } else { CF };
                assert!(try_emulate(&mut c), "{code:02x?}");
                assert_eq!(c.rip as usize, code.len());
                // The other carry flag is untouched.
                assert_eq!(c.flags & (CF | OF) & !flag, (CF | OF) & !flag);
                let expect_sum = if wide { a as u128 + b as u128 + carry as u128 } else { (a & 0xFFFF_FFFF) as u128 + (b & 0xFFFF_FFFF) as u128 + carry as u128 };
                let bits = if wide { 64 } else { 32 };
                let mask = if wide { u64::MAX as u128 } else { 0xFFFF_FFFF };
                assert_eq!(c.gpr[0] as u128, expect_sum & mask);
                assert_eq!(c.flags & flag != 0, expect_sum >> bits != 0);
                if has_adx() && wide && flag == CF {
                    let (hw, hw_cf): (u64, u8);
                    unsafe {
                        std::arch::asm!(
                            "bt {c:r}, 0", "adcx {a}, {b}", "setc {f}",
                            a = inout(reg) a => hw, b = in(reg) b, c = in(reg) carry as u64, f = out(reg_byte) hw_cf,
                        );
                    }
                    assert_eq!((c.gpr[0], c.flags & CF != 0), (hw, hw_cf != 0));
                }
            }
        }
        // adcx rdx, [rsi + 8]: 66 48 0f 38 f6 56 08
        let mem = [0u64, 40];
        let mut c = TestContext::new(&[0x66, 0x48, 0x0F, 0x38, 0xF6, 0x56, 0x08]);
        c.gpr[2] = 2;
        c.gpr[6] = mem.as_ptr() as u64;
        c.flags |= CF;
        assert!(try_emulate(&mut c));
        assert_eq!(c.gpr[2], 43);
        assert_eq!(c.flags & CF, 0);
    }
}
