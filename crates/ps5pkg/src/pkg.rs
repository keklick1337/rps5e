// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The PKG container ("\x7FCNT", big-endian).
//!
//! ```text
//! 0x000 u32 magic 7F 43 4E 54    0x004 u32 flags
//! 0x010 u32 entry count          0x014 u16 sc entry count
//! 0x018 u32 entry table offset   0x01C u32 entry data size
//! 0x020 u64 body offset          0x028 u64 body size
//! 0x040 char content_id[36]      0x070 u32 DRM type
//! 0x074 u32 content type         0x078 u32 content flags
//! 0x080 u32 version date         0x084 u32 version hash
//! 0x404 u32 PFS image count      0x408 u64 PFS flags
//! 0x410 u64 PFS image offset     0x418 u64 PFS image size
//! 0x430 u64 package size
//! ```
//! Entry table: `count` x 32 bytes `{u32 id, u32 name offset, u32 flags1,
//! u32 flags2, u32 data offset, u32 data size, u64 pad}`; flags1 bit 31
//! marks an encrypted entry. Names of unnamed system entries follow from
//! their id; others come from the name table entry (id 0x200).

use crate::io::{be_u32, be_u64, read_vec, ReadAt};
use crate::Error;

pub const PKG_MAGIC: [u8; 4] = *b"\x7FCNT";
/// The PS5 retail container. It is always encrypted with console keys.
pub const PS5_RETAIL_MAGIC: [u8; 4] = *b"\x7FFIH";
pub const PKG_HEADER_SIZE: usize = 0x1000;
pub const ENTRY_SIZE: usize = 0x20;

pub const CONTENT_TYPE_GD: u32 = 0x1A;
pub const CONTENT_TYPE_AC: u32 = 0x1B;
pub const CONTENT_TYPE_AL: u32 = 0x1C;
pub const CONTENT_TYPE_DP: u32 = 0x1E;

pub const CONTENT_FLAG_FIRST_PATCH: u32 = 0x0010_0000;
pub const CONTENT_FLAG_SUBSEQUENT_PATCH: u32 = 0x4000_0000;
pub const CONTENT_FLAG_DELTA_PATCH: u32 = 0x4100_0000;
pub const CONTENT_FLAG_CUMULATIVE_PATCH: u32 = 0x6000_0000;
const PATCH_FLAGS: u32 = CONTENT_FLAG_FIRST_PATCH | CONTENT_FLAG_SUBSEQUENT_PATCH | CONTENT_FLAG_CUMULATIVE_PATCH;

/// PFS flag selecting the newer PFS key schedule.
pub const PFS_FLAG_NEW_CRYPT: u64 = 0x2000_0000_0000_0000;

pub const ENTRY_DIGESTS: u32 = 0x0001;
pub const ENTRY_KEYS: u32 = 0x0010;
pub const ENTRY_IMAGE_KEY: u32 = 0x0020;
pub const ENTRY_METAS: u32 = 0x0100;
pub const ENTRY_NAMES: u32 = 0x0200;
pub const ENTRY_LICENSE_DAT: u32 = 0x0400;
pub const ENTRY_PARAM_SFO: u32 = 0x1000;
pub const ENTRY_PARAM_JSON: u32 = 0x100B; // shareparam.json in PS4 packages
pub const ENTRY_ICON0_PNG: u32 = 0x1200;
pub const ENTRY_PIC0_PNG: u32 = 0x1220;
pub const ENTRY_PIC1_PNG: u32 = 0x1006;

/// File name of a well-known entry id.
pub fn entry_name(id: u32) -> Option<&'static str> {
    Some(match id {
        0x0001 => "digests",
        0x0010 => "entry_keys",
        0x0020 => "image_key",
        0x0080 => "general_digests",
        0x0100 => "metas",
        0x0200 => "entry_names",
        0x0400 => "license.dat",
        0x0401 => "license.info",
        0x0402 => "nptitle.dat",
        0x0403 => "npbind.dat",
        0x0404 => "selfinfo.dat",
        0x0406 => "imageinfo.dat",
        0x0407 => "target-deltainfo.dat",
        0x0408 => "origin-deltainfo.dat",
        0x0409 => "psreserved.dat",
        0x1000 => "param.sfo",
        0x1001 => "playgo-chunk.dat",
        0x1002 => "playgo-chunk.sha",
        0x1003 => "playgo-manifest.xml",
        0x1004 => "pronunciation.xml",
        0x1005 => "pronunciation.sig",
        0x1006 => "pic1.png",
        0x1007 => "pubtoolinfo.dat",
        0x1008 => "app/playgo-chunk.dat",
        0x1009 => "app/playgo-chunk.sha",
        0x100A => "app/playgo-manifest.xml",
        0x100B => "shareparam.json",
        0x100C => "shareoverlayimage.png",
        0x100D => "save_data.png",
        0x100E => "shareprivacyguardimage.png",
        0x1200 => "icon0.png",
        0x1220 => "pic0.png",
        0x1240 => "snd0.at9",
        0x1260 => "changeinfo/changeinfo.xml",
        0x1280 => "icon0.dds",
        0x12A0 => "pic0.dds",
        0x12C0 => "pic1.dds",
        _ => return None,
    })
}

/// One entry of the table.
#[derive(Clone, Debug)]
pub struct Entry {
    pub id: u32,
    pub name: String,
    pub flags1: u32,
    pub flags2: u32,
    pub offset: u32,
    pub size: u32,
}

impl Entry {
    pub fn is_encrypted(&self) -> bool {
        self.flags1 & 0x8000_0000 != 0
    }
}

/// The parsed container header and entry table.
#[derive(Clone, Debug)]
pub struct PkgHeader {
    pub flags: u32,
    pub entry_count: u32,
    pub entry_table_offset: u32,
    pub body_offset: u64,
    pub body_size: u64,
    pub content_id: String,
    pub drm_type: u32,
    pub content_type: u32,
    pub content_flags: u32,
    pub pfs_image_count: u32,
    pub pfs_flags: u64,
    pub pfs_image_offset: u64,
    pub pfs_image_size: u64,
    pub package_size: u64,
    pub entries: Vec<Entry>,
}

impl PkgHeader {
    pub fn parse(src: &dyn ReadAt) -> Result<Self, Error> {
        if src.len() < PKG_HEADER_SIZE as u64 {
            return Err(Error::Format("file shorter than a PKG header".into()));
        }
        let h = read_vec(src, 0, PKG_HEADER_SIZE)?;
        if h[..4] == PS5_RETAIL_MAGIC {
            return Err(Error::Encrypted("PS5 retail package: encrypted with console keys, cannot be used".into()));
        }
        if h[..4] != PKG_MAGIC {
            return Err(Error::Format("not a PKG file (bad magic)".into()));
        }
        let cid_raw = &h[0x40..0x40 + 0x30];
        let content_id = String::from_utf8_lossy(cid_raw.split(|&c| c == 0).next().unwrap_or(cid_raw)).into_owned();
        let mut hdr = PkgHeader {
            flags: be_u32(&h, 0x04),
            entry_count: be_u32(&h, 0x10),
            entry_table_offset: be_u32(&h, 0x18),
            body_offset: be_u64(&h, 0x20),
            body_size: be_u64(&h, 0x28),
            content_id,
            drm_type: be_u32(&h, 0x70),
            content_type: be_u32(&h, 0x74),
            content_flags: be_u32(&h, 0x78),
            pfs_image_count: be_u32(&h, 0x404),
            pfs_flags: be_u64(&h, 0x408),
            pfs_image_offset: be_u64(&h, 0x410),
            pfs_image_size: be_u64(&h, 0x418),
            package_size: be_u64(&h, 0x430),
            entries: Vec::new(),
        };
        let count = hdr.entry_count as usize;
        let table = hdr.entry_table_offset as u64;
        if count > 0x10000 || table.checked_add((count * ENTRY_SIZE) as u64).is_none_or(|e| e > src.len()) {
            return Err(Error::Format("PKG entry table lies outside the file".into()));
        }
        let raw = read_vec(src, table, count * ENTRY_SIZE)?;
        let mut entries: Vec<Entry> = raw
            .chunks_exact(ENTRY_SIZE)
            .map(|e| Entry {
                id: be_u32(e, 0),
                name: String::new(),
                flags1: be_u32(e, 8),
                flags2: be_u32(e, 12),
                offset: be_u32(e, 16),
                size: be_u32(e, 20),
            })
            .collect();
        let name_offsets: Vec<u32> = raw.chunks_exact(ENTRY_SIZE).map(|e| be_u32(e, 4)).collect();
        let names = entries
            .iter()
            .find(|e| e.id == ENTRY_NAMES && !e.is_encrypted())
            .and_then(|e| read_vec(src, e.offset as u64, e.size as usize).ok());
        for (e, &noff) in entries.iter_mut().zip(&name_offsets) {
            e.name = match (entry_name(e.id), &names) {
                (Some(n), _) => n.to_string(),
                (None, Some(t)) if noff != 0 && (noff as usize) < t.len() => {
                    let s = &t[noff as usize..];
                    String::from_utf8_lossy(s.split(|&c| c == 0).next().unwrap_or(s)).into_owned()
                }
                _ => format!("entry_{:04x}", e.id),
            };
        }
        hdr.entries = entries;
        Ok(hdr)
    }

    pub fn entry(&self, id: u32) -> Option<&Entry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Bytes of an unencrypted entry.
    pub fn read_entry(&self, src: &dyn ReadAt, id: u32) -> Option<Vec<u8>> {
        let e = self.entry(id).filter(|e| !e.is_encrypted())?;
        read_vec(src, e.offset as u64, e.size as usize).ok()
    }

    pub fn is_patch(&self) -> bool {
        self.content_type == CONTENT_TYPE_DP || self.content_flags & PATCH_FLAGS != 0
    }
    pub fn is_delta_patch(&self) -> bool {
        self.content_type == CONTENT_TYPE_DP || self.content_flags & CONTENT_FLAG_DELTA_PATCH == CONTENT_FLAG_DELTA_PATCH
    }
    pub fn is_additional_content(&self) -> bool {
        matches!(self.content_type, CONTENT_TYPE_AC | CONTENT_TYPE_AL)
    }

    /// Digest stored for derived key `index` (entry keys: 32-byte seed
    /// digest, then seven 32-byte key digests, then seven encrypted keys).
    pub fn key_digest(&self, src: &dyn ReadAt, index: usize) -> Option<[u8; 32]> {
        let e = self.entry(ENTRY_KEYS)?;
        if index >= 7 || (e.size as usize) < 32 + 7 * 32 {
            return None;
        }
        let v = read_vec(src, e.offset as u64 + 32 + index as u64 * 32, 32).ok()?;
        v.try_into().ok()
    }
}
