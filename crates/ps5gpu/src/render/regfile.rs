// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The register file of one hardware queue: the context, shader and
//! user-config banks, the one-slot context save of the context-state
//! packets, and the instance count. The PM4 interpreter writes it as it
//! parses; the slice recorder (`crate::slices`) replays the same writes
//! from a register log, so both use the same write semantics.

use super::{DirtyBanks, RegisterBanks};

/// Register bank sizes (dword registers).
pub const CONTEXT_REGISTER_COUNT: u32 = 0x400;
pub const SHADER_REGISTER_COUNT: u32 = 0x300;
pub const UCONFIG_REGISTER_COUNT: u32 = 0x4000;
/// Bank-local "nop" register offsets that writes may target harmlessly.
pub const CONTEXT_NOP: u32 = 0xDB;
pub const SHADER_NOP: u32 = 0x280;
pub const UCONFIG_NOP: u32 = 0xA2;
/// Selector bits some register offsets carry in their top nibble.
pub const REGISTER_SELECTOR_MASK: u32 = 0x7000_0000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Bank {
    Context,
    Shader,
    UserConfig,
}

impl Bank {
    /// (nop register, register count).
    pub fn limits(self) -> (u32, u32) {
        match self {
            Bank::Context => (CONTEXT_NOP, CONTEXT_REGISTER_COUNT),
            Bank::Shader => (SHADER_NOP, SHADER_REGISTER_COUNT),
            Bank::UserConfig => (UCONFIG_NOP, UCONFIG_REGISTER_COUNT),
        }
    }
}

/// Context-state operations (CLEAR_STATE and the private packet).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextState {
    Clear,
    Push,
    Pop,
    PushClear,
}

/// What a single register write did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Write {
    Written,
    /// The bank's nop register: nothing written.
    Nop,
    /// Outside the bank: nothing written.
    OutOfBank,
}

/// Context registers after a context clear: the hardware reset values the
/// drawing code relies on without writing them (VTE_CNTL with every
/// viewport transform enabled, CB_COLOR_CONTROL, blend controls, viewport
/// scissors / transforms, guard bands, ...); every other register is 0.
pub fn initial_context_registers() -> Vec<u32> {
    let mut r = vec![0u32; CONTEXT_REGISTER_COUNT as usize];
    let mut set = |off: usize, v: u32| {
        if let Some(x) = r.get_mut(off) {
            *x = v;
        }
    };
    for (off, v) in [
        // DB_COUNT_CONTROL: the initial and cleared context enables the
        // Z-pass counter zero on both slices (occlusion queries count
        // without the game writing the register).
        (0x001, 0x1100_0100),
        (0x202, 0x00cc_0010),
        (0x206, 0x43f),
        (0x009, 0x3f80_0000),
        (0x083, 0xffff),
        (0x08c, 0xaa99_aaaa),
        (0x00d, 0x4000_4000),
        (0x081, 0x8000_0000),
        (0x082, 0x4000_4000),
        (0x090, 0x8000_0000),
        (0x091, 0x4000_4000),
        // PA_SU_POINT_SIZE (1.0 wide and high) and PA_SU_POINT_MINMAX
        // (0..4095.9375): the SDK's defaults.
        (0x280, 0x0008_0008),
        (0x281, 0xffff_0000),
        (0x292, 2),
        (0x2dc, 0xaa00),
        (0x2f9, 0x2d),
        (0x30e, 0xffff_ffff),
        (0x30f, 0xffff_ffff),
        (0x313, 0x6000),
    ] {
        set(off, v);
    }
    for i in 0..8 {
        set(0x1e0 + i, 0x2001_0001);
    }
    for i in 0..16 {
        set(0x94 + 2 * i, 0x8000_0000);
        set(0x95 + 2 * i, 0x4000_4000);
        for j in 0..6 {
            set(0x10f + 6 * i + j, if j % 2 == 0 { 0x3f80_0000 } else { 0 });
        }
    }
    r
}

/// The context bank after a clear (computed once).
pub fn initial_context() -> &'static [u32] {
    static INITIAL: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    INITIAL.get_or_init(initial_context_registers)
}

const UCONFIG_WORDS: usize = UCONFIG_REGISTER_COUNT as usize / 64;

/// The registers of one queue.
#[derive(Clone, Debug)]
pub struct RegisterFile {
    pub context: Vec<u32>,
    pub shader: Vec<u32>,
    pub uconfig: Vec<u32>,
    /// Context bank saved by a context-state push (one slot).
    pub saved_context: Vec<u32>,
    pub context_pushed: bool,
    /// NUM_INSTANCES state (indirect draws set it from their records).
    pub num_instances: u32,
    /// Banks written since the last resolve.
    pub dirty: DirtyBanks,
    /// User-config registers written since the last reset (the bank is
    /// large and mostly zero: a snapshot copies only these).
    uconfig_written: Vec<u16>,
    uconfig_bits: Box<[u64; UCONFIG_WORDS]>,
}

impl Default for RegisterFile {
    fn default() -> Self {
        Self::new()
    }
}

impl RegisterFile {
    pub fn new() -> Self {
        RegisterFile {
            context: initial_context().to_vec(),
            shader: vec![0; SHADER_REGISTER_COUNT as usize],
            uconfig: vec![0; UCONFIG_REGISTER_COUNT as usize],
            saved_context: initial_context().to_vec(),
            context_pushed: false,
            num_instances: 1,
            dirty: DirtyBanks::ALL,
            uconfig_written: Vec::new(),
            uconfig_bits: Box::new([0; UCONFIG_WORDS]),
        }
    }

    #[inline]
    pub fn banks(&self) -> RegisterBanks<'_> {
        RegisterBanks { context: &self.context, shader: &self.shader, uconfig: &self.uconfig }
    }

    /// Processor reset: banks, saved context and instance count.
    pub fn reset(&mut self) {
        self.dirty = DirtyBanks::ALL;
        self.context.copy_from_slice(initial_context());
        self.shader.fill(0);
        for &o in &self.uconfig_written {
            self.uconfig[o as usize] = 0;
        }
        self.uconfig_written.clear();
        self.uconfig_bits.fill(0);
        self.saved_context.copy_from_slice(initial_context());
        self.context_pushed = false;
        self.num_instances = 1;
    }

    /// User-config registers written since the reset (offset, value).
    pub fn uconfig_written(&self) -> impl Iterator<Item = (u16, u32)> + '_ {
        self.uconfig_written.iter().map(|&o| (o, self.uconfig[o as usize]))
    }

    #[inline]
    fn note_uconfig(&mut self, offset: u32) {
        let (w, b) = (offset as usize / 64, offset % 64);
        if self.uconfig_bits[w] & (1 << b) == 0 {
            self.uconfig_bits[w] |= 1 << b;
            self.uconfig_written.push(offset as u16);
        }
    }

    #[inline]
    fn bank_mut(&mut self, bank: Bank) -> &mut Vec<u32> {
        match bank {
            Bank::Context => {
                self.dirty.context = true;
                &mut self.context
            }
            Bank::Shader => {
                self.dirty.shader = true;
                &mut self.shader
            }
            Bank::UserConfig => {
                self.dirty.uconfig = true;
                &mut self.uconfig
            }
        }
    }

    /// One register write (the nop register and offsets outside the bank
    /// are ignored).
    #[inline]
    pub fn write(&mut self, bank: Bank, offset: u32, value: u32) -> Write {
        let (nop_reg, count) = bank.limits();
        if offset == nop_reg {
            return Write::Nop;
        }
        if offset >= count {
            return Write::OutOfBank;
        }
        self.bank_mut(bank)[offset as usize] = value;
        if bank == Bank::UserConfig {
            self.note_uconfig(offset);
        }
        Write::Written
    }

    /// A run of consecutive registers from `offset` (SET_*_REG). Returns
    /// whether a write fell outside the bank.
    pub fn set_run(&mut self, bank: Bank, offset: u32, values: &[u32]) -> bool {
        let (nop_reg, count) = bank.limits();
        let end = offset as usize + values.len();
        if end <= count as usize && !(offset..end as u32).contains(&nop_reg) {
            // The whole run lands in the bank: one copy.
            self.bank_mut(bank)[offset as usize..end].copy_from_slice(values);
            if bank == Bank::UserConfig {
                for o in offset..end as u32 {
                    self.note_uconfig(o);
                }
            }
            false
        } else {
            let mut outside = false;
            for (i, v) in values.iter().enumerate() {
                outside |= self.write(bank, offset.wrapping_add(i as u32), *v) == Write::OutOfBank;
            }
            outside
        }
    }

    /// A guest register table of `{offset, value}` pairs in order (an
    /// all-ones offset is skipped; the selector bits are not part of the
    /// register). Returns whether a write fell outside the bank.
    pub fn table(&mut self, bank: Bank, entries: &[u32]) -> bool {
        let mut outside = false;
        for e in entries.chunks_exact(2) {
            let (raw, value) = (e[0], e[1]);
            if raw == u32::MAX {
                continue;
            }
            outside |= self.write(bank, raw & !REGISTER_SELECTOR_MASK, value) == Write::OutOfBank;
        }
        outside
    }

    /// A context-state operation. A clear restores the reset values; one
    /// context can be saved at a time (Err: the operation was ignored).
    pub fn context_state(&mut self, operation: ContextState) -> Result<(), &'static str> {
        self.dirty.context = true;
        match operation {
            ContextState::Clear => self.context.copy_from_slice(initial_context()),
            ContextState::Push | ContextState::PushClear => {
                if self.context_pushed {
                    return Err("push");
                }
                self.saved_context.copy_from_slice(&self.context);
                self.context_pushed = true;
                if operation == ContextState::PushClear {
                    self.context.copy_from_slice(initial_context());
                }
            }
            ContextState::Pop => {
                if !self.context_pushed {
                    return Err("pop");
                }
                self.context.copy_from_slice(&self.saved_context);
                self.context_pushed = false;
            }
        }
        Ok(())
    }

    /// Load the state a snapshot describes (`uconfig`: every register
    /// written since the reset; the others are zero).
    pub fn load(&mut self, context: &[u32], shader: &[u32], uconfig: &[(u16, u32)], saved_context: Option<&[u32]>, num_instances: u32) {
        self.context.copy_from_slice(context);
        self.shader.copy_from_slice(shader);
        for &o in &self.uconfig_written {
            self.uconfig[o as usize] = 0;
        }
        self.uconfig_written.clear();
        self.uconfig_bits.fill(0);
        for &(o, v) in uconfig {
            self.uconfig[o as usize] = v;
            self.note_uconfig(o as u32);
        }
        match saved_context {
            Some(s) => {
                self.saved_context.copy_from_slice(s);
                self.context_pushed = true;
            }
            None => {
                self.saved_context.copy_from_slice(initial_context());
                self.context_pushed = false;
            }
        }
        self.num_instances = num_instances;
        self.dirty = DirtyBanks::ALL;
    }

    /// Same register contents (and save slot, instance count) as `o`.
    pub fn same_state(&self, o: &RegisterFile) -> bool {
        self.context == o.context
            && self.shader == o.shader
            && self.uconfig == o.uconfig
            && self.context_pushed == o.context_pushed
            && (!self.context_pushed || self.saved_context == o.saved_context)
            && self.num_instances == o.num_instances
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_tables_and_nop_registers() {
        let mut r = RegisterFile::new();
        r.dirty = DirtyBanks::NONE;
        assert!(!r.set_run(Bank::Shader, 0x10, &[1, 2, 3]));
        assert_eq!(&r.shader[0x10..0x13], &[1, 2, 3]);
        assert!(r.dirty.shader && !r.dirty.context);
        // A run over the nop register skips it.
        r.set_run(Bank::Shader, SHADER_NOP - 1, &[7, 8, 9]);
        assert_eq!((r.shader[0x27F], r.shader[0x280], r.shader[0x281]), (7, 0, 9));
        // Past the bank: the part inside is written.
        assert!(r.set_run(Bank::Context, 0x3FF, &[5, 6]));
        assert_eq!(r.context[0x3FF], 5);
        assert!(!r.table(Bank::UserConfig, &[0x7000_0243, 3, u32::MAX, 9, 0x10, 4]));
        assert_eq!((r.uconfig[0x243], r.uconfig[0x10]), (3, 4));
        let written: Vec<(u16, u32)> = r.uconfig_written().collect();
        assert_eq!(written, vec![(0x243, 3), (0x10, 4)]);
        r.reset();
        assert!(r.uconfig.iter().all(|&v| v == 0));
        assert_eq!(r.uconfig_written().count(), 0);
    }

    #[test]
    fn context_push_pop_and_load() {
        let mut r = RegisterFile::new();
        r.write(Bank::Context, 0x202, 0x1234);
        assert!(r.context_state(ContextState::PushClear).is_ok());
        assert_eq!(r.context[0x202], initial_context()[0x202]);
        assert_eq!(r.context_state(ContextState::Push), Err("push"));
        assert!(r.context_state(ContextState::Pop).is_ok());
        assert_eq!(r.context[0x202], 0x1234);
        assert_eq!(r.context_state(ContextState::Pop), Err("pop"));
        r.write(Bank::UserConfig, 0x30, 1);
        r.num_instances = 4;
        let mut s = RegisterFile::new();
        s.write(Bank::UserConfig, 0x31, 2);
        let uc: Vec<(u16, u32)> = r.uconfig_written().collect();
        s.load(&r.context, &r.shader, &uc, None, 4);
        assert!(s.same_state(&r));
    }
}

#[cfg(test)]
mod initial_tests {
    /// Point size and limits start at the SDK defaults (reference QueueState).
    #[test]
    fn point_size_and_limits_start_at_sdk_defaults() {
        let r = super::initial_context();
        assert_eq!((r[0x280], r[0x281]), (0x0008_0008, 0xffff_0000));
    }
}
