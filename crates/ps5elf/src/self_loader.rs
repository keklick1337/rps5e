// RustPS5Emu (RPS5E)
// Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! SELF / ELF loader.
//!
//! A PS5 executable is an Orbis (FreeBSD-ABI) ELF64 wrapped in a SELF
//! container. The SELF header (`4F 15 3D 1D` or `54 14 F5 EE`) is followed by
//! a segment table and then the embedded ELF header and program headers.
//! Each SELF segment entry describes a payload block of one program header.
//!
//! Segment entry `bits` layout:
//!   bit 0x800  - "blocked": entry carries the payload of a program header
//!   bit 0x002  - encrypted payload (needs runtime decryption)
//!   bit 0x008  - compressed payload
//!   bits 20..31 - program-header index this block belongs to
//!
//! Reading `size` bytes at ELF file offset `off` (`read_at`) finds the
//! blocked entry whose program header covers `off` and reads from that
//! block; a block must be stored uncompressed (compressed size equals the
//! decompressed size equals `p_filesz`). An image whose remaining bytes
//! after the SELF body are exactly `size` long is read from there. A bare
//! ELF is read at `off` directly.
//!
//! Segment payloads extracted: PT_LOAD and PT_SCE_RELRO (mapped), PT_DYNAMIC
//! and PT_SCE_DYNLIBDATA (dynamic metadata, read from the file).

use crate::elf::{validate_orbis_ehdr, Elf64Ehdr, Elf64Phdr, ProgramType};
use crate::header::{SELF_MAGIC, SELF_MAGIC_ALT, SELF_SEG_FLAG_BLOCKED, SELF_SEG_FLAG_COMPRESSED, SELF_SEG_FLAG_ENCRYPTED};
use bytemuck::{Pod, Zeroable};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SelfLoadError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a SELF/ELF file (bad magic): {0:#x?}")]
    BadMagic([u8; 4]),
    #[error("bad SELF ident tail: got {0:#x?}")]
    BadIdentTail([u8; 12]),
    #[error("signed SELF of the console (ident {0:02x?}): its segments are encrypted with console keys; only decrypted dumps (fake SELF or ELF) load")]
    SignedSelf([u8; 12]),
    #[error("bad ELF header: {0}")]
    BadEhdr(&'static str),
    #[error("unexpected EOF at offset {0}")]
    Eof(usize),
    #[error("segment for program header {0} is encrypted; runtime decryption not implemented")]
    Encrypted(usize),
    #[error("compressed SELF block for program header {0} is not supported")]
    Compressed(usize),
    #[error("missing SELF segment for file offset {0:#x}")]
    Unresolved(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SelfHeaderC {
    pub ident: [u8; 12],
    pub size1: u16,
    pub size2: u16,
    pub file_size: u64,
    pub segments_num: u16,
    pub unknown: u16,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SelfSegmentC {
    pub bits: u64,
    pub offset: u64,
    pub compressed_size: u64,
    pub decompressed_size: u64,
}

/// Decoded SELF segment entry.
#[derive(Debug, Clone, Copy)]
pub struct SelfSegment {
    pub bits: u64,
    pub offset: u64,
    pub compressed_size: u64,
    pub decompressed_size: u64,
}

impl SelfSegment {
    pub fn is_blocked(&self) -> bool {
        self.bits & SELF_SEG_FLAG_BLOCKED != 0
    }
    pub fn program_header_id(&self) -> u32 {
        ((self.bits >> 20) & 0xFFF) as u32
    }
    pub fn is_encrypted(&self) -> bool {
        self.bits & SELF_SEG_FLAG_ENCRYPTED != 0
    }
    pub fn is_compressed(&self) -> bool {
        self.bits & SELF_SEG_FLAG_COMPRESSED != 0
    }
}

#[derive(Debug, Clone)]
pub struct LoadedProgramSegment {
    pub p_type: ProgramType,
    pub p_flags: u32,
    pub vaddr: u64,
    pub filesz: u64,
    pub memsz: u64,
    pub align: u64,
    /// File payload (PT_LOAD / PT_SCE_RELRO only; empty otherwise).
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct SelfImage {
    pub ehdr: Elf64Ehdr,
    pub entry: u64,
    pub segments: Vec<LoadedProgramSegment>,
    pub phdrs: Vec<Elf64Phdr>,
    pub self_segments: Vec<SelfSegment>,
    pub is_self: bool,
    pub elf_offset: usize,
    pub self_file_size: u64,
    /// PT_DYNAMIC contents, read from the file.
    pub dynamic: Vec<u8>,
    /// PT_SCE_DYNLIBDATA contents (tables addressed by the `DT_SCE_*` tags).
    pub dynlib_data: Vec<u8>,
}

impl SelfImage {
    pub fn image_base(&self) -> Option<u64> {
        self.segments.iter().find(|s| s.p_type == ProgramType::Load).map(|s| s.vaddr)
    }

    /// Segments that occupy the image (PT_LOAD and PT_SCE_RELRO).
    pub fn program_segments(&self) -> impl Iterator<Item = &LoadedProgramSegment> {
        self.segments.iter().filter(|s| s.p_type.is_mapped())
    }

    /// Index of PT_DYNAMIC, if present.
    pub fn dynamic_segment_index(&self) -> Option<usize> {
        self.phdrs.iter().position(|p| ProgramType::from_raw(p.p_type) == ProgramType::Dynamic)
    }

    /// True when `e_ident[EI_ABIVERSION] == 2` — next-generation (PS5) image.
    pub fn is_next_gen(&self) -> bool {
        self.ehdr.e_ident[8] == 2
    }

    /// Shared object (PRX) rather than the main executable.
    pub fn is_shared(&self) -> bool {
        self.ehdr.e_type == crate::elf::ElfType::ET_DYNAMIC
    }
}

fn read_struct<T: Pod>(buf: &[u8], off: usize) -> Result<T, SelfLoadError> {
    let sz = std::mem::size_of::<T>();
    if off.checked_add(sz).is_none_or(|e| e > buf.len()) {
        return Err(SelfLoadError::Eof(off + sz));
    }
    Ok(bytemuck::pod_read_unaligned(&buf[off..off + sz]))
}

/// Known SELF magic and ident tail.
fn is_self(hdr: &SelfHeaderC) -> bool {
    let id = &hdr.ident;
    let magic = [id[0], id[1], id[2], id[3]];
    if magic != SELF_MAGIC && magic != SELF_MAGIC_ALT {
        return false;
    }
    let tail = &id[4..12];
    (tail == [0x00, 0x01, 0x01, 0x12, 0x01, 0x01, 0x00, 0x00] && hdr.unknown == 0x22)
        || (tail == [0x10, 0x01, 0x01, 0x12, 0x01, 0x01, 0x00, 0x10] && hdr.unknown == 0x32)
}

/// The SELF of a system module or application as the firmware ships it:
/// segments flagged encrypted (and compressed), keys only on the console.
/// Ident byte 9 is 0x03 for libraries and executables, 0x07 for the
/// update-mode executable, 0x01 for the EAP kernel (FW 10.60).
fn is_signed_self(hdr: &SelfHeaderC) -> bool {
    hdr.ident[..4] == SELF_MAGIC_ALT && hdr.ident[4..8] == [0x10, 0x01, 0x01, 0x32]
}

struct Reader<'a> {
    buf: &'a [u8],
    self_hdr: Option<SelfHeaderC>,
    self_segments: &'a [SelfSegment],
    phdrs: &'a [Elf64Phdr],
}

impl Reader<'_> {
    fn slice(&self, off: u64, size: u64) -> Result<&[u8], SelfLoadError> {
        let end = off.checked_add(size).ok_or(SelfLoadError::Eof(usize::MAX))?;
        if end > self.buf.len() as u64 {
            return Err(SelfLoadError::Eof(end as usize));
        }
        Ok(&self.buf[off as usize..end as usize])
    }

    /// `size` bytes at ELF file offset `file_offset`.
    fn read_at(&self, file_offset: u64, size: u64) -> Result<&[u8], SelfLoadError> {
        let Some(hdr) = self.self_hdr else {
            return self.slice(file_offset, size);
        };
        for seg in self.self_segments.iter().filter(|s| s.is_blocked()) {
            let id = seg.program_header_id() as usize;
            let Some(ph) = self.phdrs.get(id) else { continue };
            if file_offset >= ph.p_offset && file_offset < ph.p_offset + ph.p_filesz {
                if seg.is_encrypted() {
                    return Err(SelfLoadError::Encrypted(id));
                }
                if seg.decompressed_size != ph.p_filesz || seg.compressed_size != seg.decompressed_size {
                    return Err(SelfLoadError::Compressed(id));
                }
                let offset = file_offset - ph.p_offset;
                if offset + size > seg.decompressed_size {
                    return Err(SelfLoadError::Unresolved(file_offset));
                }
                return self.slice(seg.offset + offset, size);
            }
        }
        if (self.buf.len() as u64).checked_sub(hdr.file_size) == Some(size) {
            return self.slice(hdr.file_size, size);
        }
        Err(SelfLoadError::Unresolved(file_offset))
    }
}

/// Parse a SELF (or bare Orbis ELF) image.
pub fn parse_self(buf: &[u8]) -> Result<SelfImage, SelfLoadError> {
    if buf.len() < 4 {
        return Err(SelfLoadError::Eof(buf.len()));
    }
    let magic = [buf[0], buf[1], buf[2], buf[3]];
    let self_hdr = read_struct::<SelfHeaderC>(buf, 0).ok().filter(is_self);
    if self_hdr.is_none() && (magic == SELF_MAGIC || magic == SELF_MAGIC_ALT) {
        let hdr: SelfHeaderC = read_struct(buf, 0)?;
        if is_signed_self(&hdr) {
            return Err(SelfLoadError::SignedSelf(hdr.ident));
        }
        return Err(SelfLoadError::BadIdentTail(hdr.ident));
    }

    let mut self_segments: Vec<SelfSegment> = Vec::new();
    let elf_offset = match &self_hdr {
        Some(hdr) => {
            let mut cursor = std::mem::size_of::<SelfHeaderC>();
            for _ in 0..hdr.segments_num {
                let s: SelfSegmentC = read_struct(buf, cursor)?;
                cursor += std::mem::size_of::<SelfSegmentC>();
                self_segments.push(SelfSegment {
                    bits: s.bits,
                    offset: s.offset,
                    compressed_size: s.compressed_size,
                    decompressed_size: s.decompressed_size,
                });
            }
            cursor
        }
        None if magic == [0x7f, 0x45, 0x4c, 0x46] => 0,
        None => return Err(SelfLoadError::BadMagic(magic)),
    };

    let ehdr: Elf64Ehdr = read_struct(buf, elf_offset)?;
    validate_orbis_ehdr(&ehdr).map_err(SelfLoadError::BadEhdr)?;

    let phdr_off = elf_offset + ehdr.e_phoff as usize;
    let mut phdrs: Vec<Elf64Phdr> = Vec::with_capacity(ehdr.e_phnum as usize);
    for i in 0..ehdr.e_phnum as usize {
        phdrs.push(read_struct(buf, phdr_off + i * std::mem::size_of::<Elf64Phdr>())?);
    }

    let reader = Reader { buf, self_hdr, self_segments: &self_segments, phdrs: &phdrs };
    let mut prog_segments = Vec::with_capacity(phdrs.len());
    let mut dynamic = Vec::new();
    let mut dynlib_data = Vec::new();
    for ph in &phdrs {
        let p_type = ProgramType::from_raw(ph.p_type);
        let data = if p_type.is_mapped() && ph.p_memsz != 0 && ph.p_filesz != 0 {
            reader.read_at(ph.p_offset, ph.p_filesz)?.to_vec()
        } else {
            Vec::new()
        };
        match p_type {
            ProgramType::Dynamic => dynamic = reader.read_at(ph.p_offset, ph.p_filesz)?.to_vec(),
            ProgramType::SceDynlibData => dynlib_data = reader.read_at(ph.p_offset, ph.p_filesz)?.to_vec(),
            _ => {}
        }
        prog_segments.push(LoadedProgramSegment {
            p_type,
            p_flags: ph.p_flags,
            vaddr: ph.p_vaddr,
            filesz: ph.p_filesz,
            memsz: ph.p_memsz,
            align: ph.p_align,
            data,
        });
    }

    Ok(SelfImage {
        ehdr,
        entry: ehdr.e_entry,
        segments: prog_segments,
        phdrs,
        self_segments,
        is_self: self_hdr.is_some(),
        elf_offset,
        self_file_size: self_hdr.map_or(0, |h| h.file_size),
        dynamic,
        dynlib_data,
    })
}

/// The extended info block of a SELF: it follows the program headers of the
/// embedded ELF, 16-byte aligned, and is stored in plaintext even in the
/// signed SELFs of the firmware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelfExtInfo {
    /// Program authority ID. Its top byte gives the class of the program
    /// (`ProgramKind`).
    pub authority_id: u64,
    /// Program type: 1 for the decrypted (fake) SELFs of game dumps, 8 for
    /// system executables, 9 for system libraries (FW 10.60 headers).
    pub program_type: u64,
    pub app_version: u64,
    pub fw_version: u64,
    pub digest: [u8; 32],
}

/// What kind of program a SELF holds, from its authority ID class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramKind {
    /// Class 0x31: a game executable or a module shipped with a game.
    Game,
    /// Class 0x48: a system executable (shell, daemons, system apps).
    SystemExecutable,
    /// Class 0x49: a system library (`/system/common/lib`, app `sce_module`).
    SystemLibrary,
    /// Class 0x38: the update-mode executable of the PUP.
    UpdateMode,
    Other(u8),
}

impl SelfExtInfo {
    pub fn kind(&self) -> ProgramKind {
        match (self.authority_id >> 56) as u8 {
            0x31 => ProgramKind::Game,
            0x48 => ProgramKind::SystemExecutable,
            0x49 => ProgramKind::SystemLibrary,
            0x38 => ProgramKind::UpdateMode,
            other => ProgramKind::Other(other),
        }
    }

    /// Firmware version as `major.minor` (BCD in the top two bytes), if set.
    pub fn fw_version_string(&self) -> Option<String> {
        (self.fw_version != 0).then(|| format!("{:x}.{:02x}", self.fw_version >> 56, (self.fw_version >> 48) & 0xff))
    }
}

/// The open part of any SELF, signed or fake: the SELF header, its segment
/// table, the embedded ELF header with the program headers, and the
/// extended info. Nothing past these is read, so it works on the encrypted
/// SELFs of the firmware, whose segment payloads cannot be read without
/// console keys.
#[derive(Debug, Clone)]
pub struct SelfHeaders {
    pub ident: [u8; 12],
    pub file_size: u64,
    pub segments: Vec<SelfSegment>,
    pub ehdr: Elf64Ehdr,
    pub phdrs: Vec<Elf64Phdr>,
    pub ext: Option<SelfExtInfo>,
    /// The firmware's signed form (as opposed to a decrypted dump).
    pub signed: bool,
}

impl SelfHeaders {
    /// Executable (ET_SCE_EXEC/ET_SCE_DYNEXEC) rather than a library.
    pub fn is_executable(&self) -> bool {
        matches!(self.ehdr.e_type, 0xfe00 | crate::elf::ElfType::ET_DYNEXEC)
    }

    /// Number of segment blocks whose payload is encrypted.
    pub fn encrypted_blocks(&self) -> usize {
        self.segments.iter().filter(|s| s.is_blocked() && s.is_encrypted()).count()
    }

    pub fn has_program_header(&self, p_type: u32) -> bool {
        self.phdrs.iter().any(|p| p.p_type == p_type)
    }
}

/// Parse the open headers of a SELF (see `SelfHeaders`).
pub fn parse_self_headers(buf: &[u8]) -> Result<SelfHeaders, SelfLoadError> {
    let hdr: SelfHeaderC = read_struct(buf, 0)?;
    let magic = [hdr.ident[0], hdr.ident[1], hdr.ident[2], hdr.ident[3]];
    if magic != SELF_MAGIC && magic != SELF_MAGIC_ALT {
        return Err(SelfLoadError::BadMagic(magic));
    }
    let signed = is_signed_self(&hdr);
    if !signed && !is_self(&hdr) {
        return Err(SelfLoadError::BadIdentTail(hdr.ident));
    }
    let mut segments = Vec::with_capacity(hdr.segments_num as usize);
    let mut cursor = std::mem::size_of::<SelfHeaderC>();
    for _ in 0..hdr.segments_num {
        let s: SelfSegmentC = read_struct(buf, cursor)?;
        cursor += std::mem::size_of::<SelfSegmentC>();
        segments.push(SelfSegment {
            bits: s.bits,
            offset: s.offset,
            compressed_size: s.compressed_size,
            decompressed_size: s.decompressed_size,
        });
    }
    let elf_offset = cursor;
    let ehdr: Elf64Ehdr = read_struct(buf, elf_offset)?;
    if &ehdr.e_ident[0..4] != b"\x7fELF" {
        return Err(SelfLoadError::BadEhdr("no ELF header after the SELF segment table"));
    }
    let phdr_off = elf_offset + ehdr.e_phoff as usize;
    let phdrs = (0..ehdr.e_phnum as usize)
        .map(|i| read_struct::<Elf64Phdr>(buf, phdr_off + i * std::mem::size_of::<Elf64Phdr>()))
        .collect::<Result<Vec<_>, _>>()?;
    let ext_off = (phdr_off + phdrs.len() * std::mem::size_of::<Elf64Phdr>() + 15) & !15;
    let ext = read_struct::<[u64; 4]>(buf, ext_off).ok().map(|v| {
        let mut digest = [0u8; 32];
        if let Some(d) = buf.get(ext_off + 32..ext_off + 64) {
            digest.copy_from_slice(d);
        }
        SelfExtInfo { authority_id: v[0], program_type: v[1], app_version: v[2], fw_version: v[3], digest }
    });
    Ok(SelfHeaders { ident: hdr.ident, file_size: hdr.file_size, segments, ehdr, phdrs, ext, signed })
}

/// The ELF inside a fake SELF (or an ELF as it is): the ELF header and
/// program headers, and every segment's file bytes at its `p_offset`, for
/// disassemblers. Section headers are dropped (a SELF carries none).
/// Segments the SELF has no block for (e.g. a signature) are left zero.
pub fn self_to_elf(buf: &[u8]) -> Result<Vec<u8>, SelfLoadError> {
    let img = parse_self(buf)?;
    let self_hdr = read_struct::<SelfHeaderC>(buf, 0).ok().filter(is_self);
    let reader = Reader { buf, self_hdr, self_segments: &img.self_segments, phdrs: &img.phdrs };
    let mut ehdr = img.ehdr;
    ehdr.e_shoff = 0;
    ehdr.e_shnum = 0;
    ehdr.e_shstrndx = 0;
    let headers = ehdr.e_phoff as usize + img.phdrs.len() * std::mem::size_of::<Elf64Phdr>();
    let end = img.phdrs.iter().map(|p| (p.p_offset + p.p_filesz) as usize).max().unwrap_or(0).max(headers);
    let mut out = vec![0u8; end];
    out[..std::mem::size_of::<Elf64Ehdr>()].copy_from_slice(bytemuck::bytes_of(&ehdr));
    for (i, ph) in img.phdrs.iter().enumerate() {
        let at = ehdr.e_phoff as usize + i * std::mem::size_of::<Elf64Phdr>();
        out[at..at + std::mem::size_of::<Elf64Phdr>()].copy_from_slice(bytemuck::bytes_of(ph));
    }
    for ph in &img.phdrs {
        if ph.p_filesz == 0 {
            continue;
        }
        match reader.read_at(ph.p_offset, ph.p_filesz) {
            Ok(data) => out[ph.p_offset as usize..(ph.p_offset + ph.p_filesz) as usize].copy_from_slice(data),
            Err(SelfLoadError::Unresolved(_)) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_self_unwraps_to_its_elf() {
        let elf = self_to_elf(&build_self(false)).unwrap();
        assert_eq!(&elf[..4], b"\x7fELF");
        assert_eq!(elf.len(), 0x2040);
        assert_eq!(elf[0x1000..0x1010], (0..0x10u8).map(|i| 0xC0 + i).collect::<Vec<_>>()[..]);
        assert_eq!(elf[0x2000..0x2040], (0..0x40u8).collect::<Vec<_>>()[..]);
        // The result loads like the SELF.
        let img = parse_self(&elf).unwrap();
        assert!(!img.is_self);
        assert_eq!(img.dynamic, (0x20..0x40u8).collect::<Vec<_>>());
    }

    /// A signed SELF as the firmware ships it: encrypted, compressed blocks,
    /// plaintext ELF and program headers, then the extended info. Values of
    /// the header of system_ex/app/NPXS40099/eboot.bin (FW 10.60).
    fn build_signed_self() -> Vec<u8> {
        let mut out = Vec::new();
        let hdr = SelfHeaderC {
            ident: [0x54, 0x14, 0xf5, 0xee, 0x10, 0x01, 0x01, 0x32, 0x01, 0x03, 0x00, 0x10],
            size1: 0x0530,
            size2: 0x0650,
            file_size: 0xb2be5,
            segments_num: 2,
            unknown: 0x52,
            pad: 0,
        };
        out.extend_from_slice(bytemuck::bytes_of(&hdr));
        let seg = |bits: u64, off: u64, c: u64, d: u64| SelfSegmentC { bits, offset: off, compressed_size: c, decompressed_size: d };
        out.extend_from_slice(bytemuck::bytes_of(&seg(0x13_0006, 0xb80, 0xed8, 0xed8)));
        out.extend_from_slice(bytemuck::bytes_of(&seg(0x2c0e, 0x1a60, 0x888d0, 0x1792dc)));
        out.extend_from_slice(bytemuck::bytes_of(&ehdr(0xfe10, 2)));
        out.extend_from_slice(bytemuck::bytes_of(&phdr(1, 0x4000, 0, 0x1792dc, 0x1792dc, 5)));
        out.extend_from_slice(bytemuck::bytes_of(&phdr(0x6100_0001, 0x2093a0, 0x2053a0, 0x60, 0x60, 4)));
        while out.len() % 16 != 0 {
            out.push(0);
        }
        for v in [0x4800_0000_0000_002cu64, 8, 0x1060_0000_0000_0000, 0x1060_0000_0000_0000] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend((0..32u8).map(|i| 0xA0 ^ i));
        out.resize(out.len() + 0x100, 0x5A); // signature, encrypted payload
        out
    }

    #[test]
    fn signed_self_headers_are_readable() {
        let buf = build_signed_self();
        assert!(matches!(parse_self(&buf), Err(SelfLoadError::SignedSelf(_))));
        let h = parse_self_headers(&buf).unwrap();
        assert!(h.signed && h.is_executable());
        assert_eq!(h.segments.len(), 2);
        assert_eq!(h.encrypted_blocks(), 1, "only the blocked entry counts");
        assert_eq!(h.segments[1].program_header_id(), 0);
        assert!(h.segments[1].is_compressed());
        assert_eq!(h.phdrs.len(), 2);
        assert!(h.has_program_header(0x6100_0001), "an executable carries PT_SCE_PROCPARAM");
        let ext = h.ext.unwrap();
        assert_eq!(ext.kind(), ProgramKind::SystemExecutable);
        assert_eq!(ext.program_type, 8);
        assert_eq!(ext.fw_version_string().as_deref(), Some("10.60"));
        assert_eq!(ext.digest[1], 0xA1);
    }

    #[test]
    fn program_kind_from_authority_id() {
        let ext = |id: u64| SelfExtInfo { authority_id: id, program_type: 0, app_version: 0, fw_version: 0, digest: [0; 32] };
        assert_eq!(ext(0x3100_0000_0000_0002).kind(), ProgramKind::Game);
        assert_eq!(ext(0x4900_0000_0000_0002).kind(), ProgramKind::SystemLibrary);
        assert_eq!(ext(0x3800_1000_0000_0001).kind(), ProgramKind::UpdateMode);
        assert_eq!(ext(0x2a00_0000_0000_0000).kind(), ProgramKind::Other(0x2a));
        assert_eq!(ext(0).fw_version_string(), None);
    }

    #[test]
    fn fake_self_headers_match_the_loader() {
        let buf = build_self(false);
        let h = parse_self_headers(&buf).unwrap();
        let img = parse_self(&buf).unwrap();
        assert!(!h.signed);
        assert_eq!(h.phdrs.len(), img.phdrs.len());
        assert_eq!(h.segments.len(), img.self_segments.len());
        assert_eq!(h.ehdr.e_type, 0xfe10);
        assert!(matches!(parse_self_headers(b"\x7fELF\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"), Err(SelfLoadError::BadMagic(_))));
    }

    #[test]
    fn signed_firmware_self_is_reported_as_such() {
        // The first 0x20 bytes of libSceAgc.sprx of firmware 10.60.
        let mut buf = vec![
            0x54, 0x14, 0xf5, 0xee, 0x10, 0x01, 0x01, 0x32, 0x01, 0x03, 0x00, 0x10, 0x30, 0x05, 0x30, 0x06, //
            0x6a, 0x72, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x52, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        buf.resize(0x200, 0);
        assert!(matches!(parse_self(&buf), Err(SelfLoadError::SignedSelf(_))));
    }

    fn ehdr(e_type: u16, phnum: u16) -> Elf64Ehdr {
        let mut e: Elf64Ehdr = Zeroable::zeroed();
        e.e_ident[..4].copy_from_slice(b"\x7fELF");
        e.e_ident[4] = 2;
        e.e_ident[5] = 1;
        e.e_ident[6] = 1;
        e.e_ident[7] = 9;
        e.e_ident[8] = 2;
        e.e_type = e_type;
        e.e_machine = 62;
        e.e_version = 1;
        e.e_phoff = 64;
        e.e_phentsize = 56;
        e.e_phnum = phnum;
        e
    }

    fn phdr(p_type: u32, offset: u64, vaddr: u64, filesz: u64, memsz: u64, flags: u32) -> Elf64Phdr {
        Elf64Phdr { p_type, p_flags: flags, p_offset: offset, p_vaddr: vaddr, p_paddr: 0, p_filesz: filesz, p_memsz: memsz, p_align: 0x4000 }
    }

    /// SELF with one code segment and a dynamic table inside a second
    /// (non-executable) segment: the dynamic table is read through the block
    /// of the segment that covers its file offset.
    fn build_self(compressed: bool) -> Vec<u8> {
        let phs = [
            phdr(1, 0x1000, 0, 0x10, 0x4000, 5),
            phdr(1, 0x2000, 0x4000, 0x40, 0x40, 0),
            phdr(2, 0x2020, 0x4020, 0x20, 0x20, 6),
        ];
        let mut elf = Vec::new();
        elf.extend_from_slice(bytemuck::bytes_of(&ehdr(0xfe10, 3)));
        for p in &phs {
            elf.extend_from_slice(bytemuck::bytes_of(p));
        }
        let segs = 2usize;
        let body_off = 32 + 32 * segs + elf.len();
        let code_off = body_off as u64;
        let data_off = code_off + 0x10;
        let mut out = Vec::new();
        let hdr = SelfHeaderC {
            ident: [0x4F, 0x15, 0x3D, 0x1D, 0x00, 0x01, 0x01, 0x12, 0x01, 0x01, 0x00, 0x00],
            size1: 0,
            size2: 0,
            file_size: data_off + 0x40,
            segments_num: segs as u16,
            unknown: 0x22,
            pad: 0,
        };
        out.extend_from_slice(bytemuck::bytes_of(&hdr));
        let blk = |id: u64, off: u64, size: u64, csize: u64| SelfSegmentC { bits: 0x800 | (id << 20), offset: off, compressed_size: csize, decompressed_size: size };
        out.extend_from_slice(bytemuck::bytes_of(&blk(0, code_off, 0x10, 0x10)));
        out.extend_from_slice(bytemuck::bytes_of(&blk(1, data_off, 0x40, if compressed { 0x30 } else { 0x40 })));
        out.extend_from_slice(&elf);
        out.extend((0..0x10u8).map(|i| 0xC0 + i));
        out.extend((0..0x40u8).map(|i| i));
        out
    }

    #[test]
    fn self_blocks_cover_file_offsets() {
        let img = parse_self(&build_self(false)).unwrap();
        assert!(img.is_self);
        assert_eq!(img.segments[0].data, (0..0x10u8).map(|i| 0xC0 + i).collect::<Vec<_>>());
        assert_eq!(img.segments[1].data.len(), 0x40);
        // PT_DYNAMIC at file offset 0x2020 lies inside segment 1's block.
        assert_eq!(img.dynamic, (0x20..0x40u8).collect::<Vec<_>>());
    }

    #[test]
    fn compressed_blocks_are_rejected() {
        assert!(matches!(parse_self(&build_self(true)), Err(SelfLoadError::Compressed(1))));
    }

    #[test]
    fn bare_elf_reads_file_offsets() {
        let mut elf = Vec::new();
        elf.extend_from_slice(bytemuck::bytes_of(&ehdr(0xfe18, 1)));
        elf.extend_from_slice(bytemuck::bytes_of(&phdr(1, 0x100, 0, 4, 0x10, 4)));
        elf.resize(0x104, 0);
        elf[0x100..0x104].copy_from_slice(&[1, 2, 3, 4]);
        let img = parse_self(&elf).unwrap();
        assert!(!img.is_self && img.is_shared());
        assert_eq!(img.segments[0].data, [1, 2, 3, 4]);
    }

    #[test]
    fn elf_header_validation() {
        let mut e = ehdr(0xfe10, 0);
        assert!(validate_orbis_ehdr(&e).is_ok());
        e.e_ident[7] = 0;
        assert!(validate_orbis_ehdr(&e).is_err(), "OS/ABI must be FreeBSD");
        e.e_ident[7] = 9;
        e.e_ident[8] = 1;
        assert!(validate_orbis_ehdr(&e).is_err(), "ABI version 0 or 2");
        e.e_ident[8] = 0;
        e.e_type = 2;
        assert!(validate_orbis_ehdr(&e).is_err(), "type must be dynexec/dynamic");
        e.e_type = 0xfe18;
        e.e_shentsize = 40;
        assert!(validate_orbis_ehdr(&e).is_err(), "section header size");
    }
}
