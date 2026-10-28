// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! exFAT: read-only access to exFAT volumes (game dumps distributed as
//! `.exfat` disk images, also found inside `.ffpfsc` containers).
//!
//! Main boot sector (little-endian, sector 0 of the volume):
//! ```text
//! 0x00 u8[3] jump (EB 76 90)      0x03 "EXFAT   "
//! 0x40 u64 partition offset       0x48 u64 volume length (sectors)
//! 0x50 u32 FAT offset (sectors)   0x54 u32 FAT length (sectors)
//! 0x58 u32 cluster heap offset    0x5C u32 cluster count
//! 0x60 u32 root directory cluster 0x64 u32 serial number
//! 0x68 u16 revision               0x6A u16 volume flags (bit 0: 2nd FAT active)
//! 0x6C u8 bytes-per-sector shift  0x6D u8 sectors-per-cluster shift
//! 0x6E u8 number of FATs          0x1FE 55 AA
//! ```
//! Sectors 0..11 form the boot region; sector 11 repeats a 32-bit checksum
//! of sectors 0..10 (bytes 0x6A, 0x6B and 0x70 of sector 0 excluded). A
//! backup boot region follows at sector 12.
//!
//! The FAT is an array of u32 next-cluster links (clusters start at 2;
//! 0xFFFFFFFF ends a chain). A stream with the NoFatChain flag occupies
//! consecutive clusters and its FAT entries are not used.
//!
//! Directories are arrays of 32-byte entries. Type 0x00 ends the directory;
//! types with bit 7 clear are unused. A file is an entry set: a File entry
//! (0x85: secondary count, set checksum, attributes, timestamps), a Stream
//! Extension (0xC0: flags, name length, valid data length, first cluster,
//! data length) and File Name entries (0xC1: 15 UTF-16 units each). The
//! root directory also holds the Allocation Bitmap (0x81) and the Up-case
//! Table (0x82) that defines case-insensitive name comparison.
//!
//! Everything is read lazily through [`ReadAt`]: opening a volume reads the
//! boot region and the up-case table, listing reads directory clusters, and
//! a file is mapped to runs of the image only when it is opened.

use crate::io::{invalid, le_u16, le_u32, le_u64, read_vec, ExtentList, ReadAt, Source};
use crate::pfs::{FileOpener, FsKind, Node, NodeId, Pfs};
use crate::Error;
use std::collections::{HashMap, HashSet};
use std::io;
use std::sync::{Arc, Mutex};

pub const EXFAT_SIGNATURE: &[u8; 8] = b"EXFAT   ";
pub const ENTRY_SIZE: usize = 32;
pub const ENTRY_BITMAP: u8 = 0x81;
pub const ENTRY_UPCASE: u8 = 0x82;
pub const ENTRY_FILE: u8 = 0x85;
pub const ENTRY_STREAM: u8 = 0xC0;
pub const ENTRY_NAME: u8 = 0xC1;
pub const ATTR_DIRECTORY: u16 = 0x10;
pub const FLAG_NO_FAT_CHAIN: u8 = 0x02;
const FAT_EOC: u32 = 0xFFFF_FFFF;
/// FAT bytes cached per page, and pages kept.
const FAT_PAGE: u64 = 0x1_0000;
const FAT_PAGES: usize = 64;
/// Upper bound on one directory's size (the format allows 256 MiB).
const MAX_DIR_BYTES: u64 = 256 << 20;
/// Upper bound on tree nodes (a corrupt image must not exhaust memory).
const MAX_NODES: usize = 8_000_000;

/// True when `sector0` (at least 11 bytes) starts like an exFAT volume.
pub fn is_exfat(sector0: &[u8]) -> bool {
    sector0.get(3..11) == Some(&EXFAT_SIGNATURE[..])
}

/// Boot-region checksum over `region` (11 sectors).
pub fn boot_checksum(region: &[u8]) -> u32 {
    let mut c = 0u32;
    for (i, &b) in region.iter().enumerate() {
        if i == 0x6A || i == 0x6B || i == 0x70 {
            continue;
        }
        c = c.rotate_right(1).wrapping_add(b as u32);
    }
    c
}

/// Entry-set checksum over `set` (bytes 2 and 3 of the first entry hold it).
pub fn set_checksum(set: &[u8]) -> u16 {
    let mut c = 0u16;
    for (i, &b) in set.iter().enumerate() {
        if i == 2 || i == 3 {
            continue;
        }
        c = c.rotate_right(1).wrapping_add(b as u16);
    }
    c
}

/// Up-case table checksum over its stored bytes.
pub fn table_checksum(data: &[u8]) -> u32 {
    data.iter().fold(0u32, |c, &b| c.rotate_right(1).wrapping_add(b as u32))
}

/// Seconds since 1970 of an exFAT timestamp (local time of the writer,
/// corrected by the UTC offset byte when its valid bit is set).
pub fn timestamp_to_unix(ts: u32, ms10: u8, utc: u8) -> i64 {
    let year = 1980 + (ts >> 25) as i64;
    let month = ((ts >> 21) & 0xF).clamp(1, 12) as i64;
    let day = ((ts >> 16) & 0x1F).max(1) as i64;
    let hour = ((ts >> 11) & 0x1F) as i64;
    let min = ((ts >> 5) & 0x3F) as i64;
    let sec = ((ts & 0x1F) * 2) as i64 + ms10 as i64 / 100;
    // Days from the civil date (proleptic Gregorian calendar).
    let (y, m) = if month <= 2 { (year - 1, month + 9) } else { (year, month - 3) };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * m + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    let mut t = days * 86_400 + hour * 3600 + min * 60 + sec;
    if utc & 0x80 != 0 {
        // Signed 7-bit count of 15-minute steps east of UTC.
        let steps = ((utc << 1) as i8 >> 1) as i64;
        t -= steps * 15 * 60;
    }
    t
}

/// A file or directory of an exFAT directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirEntry {
    pub name: String,
    pub is_dir: bool,
    /// Logical size (DataLength).
    pub size: u64,
    /// Bytes actually written (ValidDataLength); the rest reads as zeros.
    pub valid: u64,
    pub first_cluster: u32,
    /// NoFatChain: the data occupies consecutive clusters.
    pub contiguous: bool,
    pub mtime: i64,
}

/// An opened exFAT volume.
pub struct ExFat {
    src: Source,
    sector_shift: u32,
    cluster_shift: u32,
    fat_offset: u64,
    heap_offset: u64,
    cluster_count: u32,
    root_cluster: u32,
    /// Code unit -> upper case (identity where the table says nothing).
    upcase: Vec<u16>,
    fat_cache: Mutex<HashMap<u64, Arc<Vec<u8>>>>,
}

fn format(msg: impl Into<String>) -> Error {
    Error::Format(format!("exFAT: {}", msg.into()))
}

impl ExFat {
    /// Open the volume that starts at offset 0 of `src`.
    pub fn open(src: Source) -> Result<Self, Error> {
        let mut first = [0u8; 512];
        if crate::io::read_at(&*src, &mut first, 0)? < 512 || !is_exfat(&first) {
            return Err(format("missing boot sector signature"));
        }
        // The main boot region, else the backup one.
        let main = Self::boot_region(&src, 0);
        let boot = match main {
            Ok(b) => b,
            Err(e) => {
                let shift = first[0x6C] as u64;
                let backup = if (9..=12).contains(&shift) { Self::boot_region(&src, 12 << shift).ok() } else { None };
                backup.ok_or(e)?
            }
        };
        let sector_shift = boot[0x6C] as u32;
        let cluster_shift = sector_shift + boot[0x6D] as u32;
        let fats = boot[0x6E];
        let flags = le_u16(&boot, 0x6A);
        let fat_len = le_u32(&boot, 0x54) as u64;
        let mut fat_offset = (le_u32(&boot, 0x50) as u64) << sector_shift;
        if fats == 2 && flags & 1 != 0 {
            fat_offset += fat_len << sector_shift;
        }
        let heap_offset = (le_u32(&boot, 0x58) as u64) << sector_shift;
        let cluster_count = le_u32(&boot, 0x5C);
        let root_cluster = le_u32(&boot, 0x60);
        if !(1..=2).contains(&fats) || cluster_shift > 25 || fat_len == 0 {
            return Err(format("invalid boot sector geometry"));
        }
        if (fat_len << sector_shift) < (cluster_count as u64 + 2) * 4 {
            return Err(format("the FAT is smaller than the cluster count"));
        }
        let mut fs = ExFat {
            src,
            sector_shift,
            cluster_shift,
            fat_offset,
            heap_offset,
            cluster_count,
            root_cluster,
            upcase: Vec::new(),
            fat_cache: Mutex::new(HashMap::new()),
        };
        if !fs.valid_cluster(root_cluster) {
            return Err(format(format!("root directory cluster {root_cluster} is invalid")));
        }
        fs.upcase = fs.load_upcase().unwrap_or_else(|_| default_upcase());
        Ok(fs)
    }

    /// Read and verify the boot region at byte `base`; returns sector 0.
    fn boot_region(src: &Source, base: u64) -> Result<Vec<u8>, Error> {
        let mut s0 = [0u8; 512];
        src.read_exact_at(&mut s0, base)?;
        let shift = s0[0x6C] as u32;
        if !is_exfat(&s0) || !(9..=12).contains(&shift) || s0[0x1FE] != 0x55 || s0[0x1FF] != 0xAA {
            return Err(format("invalid boot sector"));
        }
        let sector = 1usize << shift;
        let region = read_vec(&**src, base, sector * 12)?;
        let sum = boot_checksum(&region[..sector * 11]);
        if region[sector * 11..].chunks_exact(4).any(|c| le_u32(c, 0) != sum) {
            return Err(format("boot region checksum mismatch"));
        }
        Ok(region[..sector].to_vec())
    }

    pub fn cluster_size(&self) -> u64 {
        1 << self.cluster_shift
    }

    pub fn sector_size(&self) -> u64 {
        1 << self.sector_shift
    }

    pub fn cluster_count(&self) -> u32 {
        self.cluster_count
    }

    fn valid_cluster(&self, c: u32) -> bool {
        c >= 2 && c - 2 < self.cluster_count
    }

    fn cluster_offset(&self, c: u32) -> u64 {
        self.heap_offset + ((c as u64 - 2) << self.cluster_shift)
    }

    /// The FAT entry of cluster `c`.
    fn fat_next(&self, c: u32) -> io::Result<u32> {
        let off = self.fat_offset + c as u64 * 4;
        let page = off / FAT_PAGE;
        let data = {
            let mut cache = self.fat_cache.lock().unwrap_or_else(|p| p.into_inner());
            match cache.get(&page) {
                Some(d) => d.clone(),
                None => {
                    let start = page * FAT_PAGE;
                    let len = FAT_PAGE.min(self.src.len().saturating_sub(start)) as usize;
                    let d = Arc::new(read_vec(&*self.src, start, len)?);
                    if cache.len() >= FAT_PAGES {
                        cache.clear();
                    }
                    cache.insert(page, d.clone());
                    d
                }
            }
        };
        let within = (off % FAT_PAGE) as usize;
        if within + 4 > data.len() {
            return Err(invalid("exFAT: FAT entry lies outside the image"));
        }
        Ok(le_u32(&data, within))
    }

    /// Map `len` bytes of a stream starting at `first` to image runs (zero
    /// `len` with a chain: follow the chain to its end, for the root
    /// directory whose size is not recorded).
    fn stream_runs(&self, first: u32, len: Option<u64>, contiguous: bool) -> Result<ExtentList, Error> {
        let cs = self.cluster_size();
        let mut out = ExtentList::new();
        if len == Some(0) {
            return Ok(out);
        }
        if !self.valid_cluster(first) {
            return Err(format(format!("stream starts at invalid cluster {first}")));
        }
        if contiguous {
            let len = len.unwrap_or(cs);
            let clusters = len.div_ceil(cs);
            if (first - 2) as u64 + clusters > self.cluster_count as u64 {
                return Err(format("contiguous stream runs past the cluster heap"));
            }
            out.push(len, Some(self.cluster_offset(first)));
            return Ok(out);
        }
        let mut c = first;
        let mut steps = 0u64;
        loop {
            let left = len.map(|l| l - out.len());
            let take = left.map_or(cs, |l| l.min(cs));
            out.push(take, Some(self.cluster_offset(c)));
            if left.is_some_and(|l| l <= cs) {
                break;
            }
            steps += 1;
            if steps > self.cluster_count as u64 {
                return Err(format("FAT chain loops"));
            }
            let next = self.fat_next(c)?;
            if next == FAT_EOC && len.is_none() {
                break;
            }
            if !self.valid_cluster(next) {
                return Err(format(format!("FAT chain ends early (cluster {c} -> {next:#x})")));
            }
            if len.is_none() && out.len() >= MAX_DIR_BYTES {
                return Err(format("directory too large"));
            }
            c = next;
        }
        Ok(out)
    }

    /// Readable contents of a file entry.
    pub fn open_entry(&self, e: &DirEntry) -> Result<Source, Error> {
        let valid = e.valid.min(e.size);
        let mut runs = if valid == 0 { ExtentList::new() } else { self.stream_runs(e.first_cluster, Some(valid), e.contiguous)? };
        runs.push(e.size - valid, None);
        Ok(runs.build(self.src.clone())?)
    }

    fn read_stream(&self, first: u32, len: Option<u64>, contiguous: bool) -> Result<Vec<u8>, Error> {
        if len.is_some_and(|l| l > MAX_DIR_BYTES) {
            return Err(format("directory too large"));
        }
        let src = self.stream_runs(first, len, contiguous)?.build(self.src.clone())?;
        Ok(read_vec(&*src, 0, src.len() as usize)?)
    }

    fn root_entries(&self) -> Result<Vec<u8>, Error> {
        self.read_stream(self.root_cluster, None, false)
    }

    fn load_upcase(&self) -> Result<Vec<u16>, Error> {
        let raw = self.root_entries()?;
        for e in raw.chunks_exact(ENTRY_SIZE) {
            if e[0] == 0 {
                break;
            }
            if e[0] != ENTRY_UPCASE {
                continue;
            }
            let (sum, first, len) = (le_u32(e, 4), le_u32(e, 20), le_u64(e, 24));
            if len == 0 || len > 0x2_0000 * 2 || len % 2 != 0 {
                return Err(format("up-case table has an invalid size"));
            }
            let data = self.read_stream(first, Some(len), false)?;
            if table_checksum(&data) != sum {
                return Err(format("up-case table checksum mismatch"));
            }
            return Ok(expand_upcase(&data));
        }
        Err(format("no up-case table"))
    }

    /// Upper-case form of a name, by the volume's up-case table.
    pub fn upcase_name(&self, name: &str) -> Vec<u16> {
        name.encode_utf16().map(|u| self.upcase.get(u as usize).copied().unwrap_or(u)).collect()
    }

    /// The entries of the root directory.
    pub fn root_dir(&self) -> Result<Vec<DirEntry>, Error> {
        parse_entries(&self.root_entries()?)
    }

    /// The entries of directory `dir`.
    pub fn read_dir(&self, dir: &DirEntry) -> Result<Vec<DirEntry>, Error> {
        if !dir.is_dir {
            return Err(format(format!("{} is not a directory", dir.name)));
        }
        if dir.size == 0 {
            return Ok(Vec::new());
        }
        parse_entries(&self.read_stream(dir.first_cluster, Some(dir.size), dir.contiguous)?)
    }

    /// Resolve a '/'-separated path from the root, comparing names through
    /// the up-case table as the format defines.
    pub fn lookup(&self, path: &str) -> Result<Option<DirEntry>, Error> {
        let mut entries = self.root_dir()?;
        let mut found: Option<DirEntry> = None;
        for seg in path.split(['/', '\\']).filter(|s| !s.is_empty() && *s != ".") {
            if let Some(d) = &found {
                if !d.is_dir {
                    return Ok(None);
                }
                entries = self.read_dir(d)?;
            }
            let want = self.upcase_name(seg);
            match entries.iter().find(|e| self.upcase_name(&e.name) == want) {
                Some(e) => found = Some(e.clone()),
                None => return Ok(None),
            }
        }
        Ok(found)
    }

    /// Index the whole directory tree (directories are read; file data is
    /// not touched).
    pub fn into_tree(self) -> Result<Pfs, Error> {
        let src = self.src.clone();
        let fs = Arc::new(self);
        let mut nodes = vec![Node {
            name: String::new(),
            parent: 0,
            ino: 0,
            is_dir: true,
            size: 0,
            mtime: 0,
            children: Vec::new(),
        }];
        let mut files: Vec<DirEntry> = Vec::new();
        let mut visited: HashSet<u32> = HashSet::from([fs.root_cluster]);
        let mut stack: Vec<(NodeId, Option<DirEntry>)> = vec![(0, None)];
        while let Some((dir, entry)) = stack.pop() {
            let list = match &entry {
                None => fs.root_dir()?,
                Some(e) => fs.read_dir(e)?,
            };
            let mut kids = Vec::with_capacity(list.len());
            for e in list {
                if e.name.is_empty() || e.name == "." || e.name == ".." || e.name.contains(['/', '\\']) {
                    continue;
                }
                if e.is_dir && e.size > 0 && !visited.insert(e.first_cluster) {
                    return Err(format("directory cycle"));
                }
                if nodes.len() >= MAX_NODES {
                    return Err(format("too many files"));
                }
                let id = nodes.len() as NodeId;
                let key = if e.is_dir { 0 } else { files.len() as u32 };
                nodes.push(Node {
                    name: e.name.clone(),
                    parent: dir,
                    ino: key,
                    is_dir: e.is_dir,
                    size: if e.is_dir { 0 } else { e.size },
                    mtime: e.mtime,
                    children: Vec::new(),
                });
                kids.push(id);
                if e.is_dir {
                    stack.push((id, Some(e)));
                } else {
                    files.push(e);
                }
            }
            nodes[dir as usize].children = kids;
        }
        Ok(Pfs::from_nodes(src, FsKind::ExFat, nodes, Box::new(ExFatFiles { fs, files })))
    }
}

struct ExFatFiles {
    fs: Arc<ExFat>,
    files: Vec<DirEntry>,
}

impl FileOpener for ExFatFiles {
    fn open(&self, key: u32) -> Result<Source, Error> {
        let e = self.files.get(key as usize).ok_or_else(|| format(format!("file {key} out of range")))?;
        self.fs.open_entry(e)
    }
}

/// The identity mapping with ASCII a-z upper-cased (used when a volume has
/// no readable up-case table).
fn default_upcase() -> Vec<u16> {
    (0..=0xFFFFu32).map(|u| if (0x61..=0x7A).contains(&u) { (u - 0x20) as u16 } else { u as u16 }).collect()
}

/// Expand a stored up-case table: a u16 per code unit from 0 upwards,
/// where 0xFFFF followed by N stands for N code units mapping to
/// themselves.
pub fn expand_upcase(data: &[u8]) -> Vec<u16> {
    let mut out: Vec<u16> = Vec::with_capacity(0x1_0000);
    let mut i = 0;
    while i + 1 < data.len() && out.len() < 0x1_0000 {
        let v = le_u16(data, i);
        i += 2;
        if v == 0xFFFF && i + 1 < data.len() {
            let n = le_u16(data, i) as usize;
            i += 2;
            for _ in 0..n {
                if out.len() >= 0x1_0000 {
                    break;
                }
                out.push(out.len() as u16);
            }
        } else {
            out.push(v);
        }
    }
    while out.len() < 0x1_0000 {
        out.push(out.len() as u16);
    }
    out
}

/// Decode the file entry sets of a directory's bytes. Sets that are
/// incomplete or fail their checksum are skipped.
pub fn parse_entries(raw: &[u8]) -> Result<Vec<DirEntry>, Error> {
    let count = raw.len() / ENTRY_SIZE;
    let entry = |i: usize| &raw[i * ENTRY_SIZE..(i + 1) * ENTRY_SIZE];
    let mut out = Vec::new();
    let mut i = 0;
    while i < count {
        let e = entry(i);
        let ty = e[0];
        if ty == 0 {
            break;
        }
        if ty != ENTRY_FILE {
            i += 1;
            continue;
        }
        let secondary = e[1] as usize;
        if !(2..=18).contains(&secondary) || i + secondary >= count {
            i += 1;
            continue;
        }
        let set = &raw[i * ENTRY_SIZE..(i + 1 + secondary) * ENTRY_SIZE];
        let stream = entry(i + 1);
        if stream[0] != ENTRY_STREAM || set_checksum(set) != le_u16(e, 2) {
            i += 1;
            continue;
        }
        let name_len = stream[3] as usize;
        let mut units: Vec<u16> = Vec::with_capacity(name_len);
        for k in 2..=secondary {
            let n = entry(i + k);
            if n[0] != ENTRY_NAME {
                continue;
            }
            for j in 0..15 {
                if units.len() < name_len {
                    units.push(le_u16(n, 2 + j * 2));
                }
            }
        }
        if units.len() == name_len && name_len > 0 {
            let attrs = le_u16(e, 4);
            out.push(DirEntry {
                name: String::from_utf16_lossy(&units),
                is_dir: attrs & ATTR_DIRECTORY != 0,
                size: le_u64(stream, 24),
                valid: le_u64(stream, 8),
                first_cluster: le_u32(stream, 20),
                contiguous: stream[1] & FLAG_NO_FAT_CHAIN != 0,
                mtime: timestamp_to_unix(le_u32(e, 12), e[21], e[23]),
            });
        }
        i += 1 + secondary;
    }
    Ok(out)
}

/// Byte offset of an exFAT volume in a disk image: the image itself, or a
/// partition of an MBR or GPT partition table.
pub fn find_volume(src: &dyn ReadAt) -> Option<u64> {
    let mut s0 = [0u8; 512];
    if crate::io::read_at(src, &mut s0, 0).ok()? < 512 {
        return None;
    }
    if is_exfat(&s0) {
        return Some(0);
    }
    if s0[510] != 0x55 || s0[511] != 0xAA {
        return None;
    }
    let is_volume = |off: u64| {
        let mut h = [0u8; 16];
        crate::io::read_at(src, &mut h, off).is_ok_and(|n| n >= 11) && is_exfat(&h)
    };
    let mut starts = Vec::new();
    for p in 0..4 {
        let e = &s0[446 + p * 16..462 + p * 16];
        let (kind, lba) = (e[4], le_u32(e, 8) as u64);
        if kind == 0xEE {
            // Protective MBR: read the GPT (512- or 4096-byte sectors).
            for sector in [512u64, 4096] {
                let Ok(h) = read_vec(src, sector, 92) else { continue };
                if &h[..8] != b"EFI PART" {
                    continue;
                }
                let (table, n, size) = (le_u64(&h, 72), le_u32(&h, 80).min(256) as u64, le_u32(&h, 84) as u64);
                if size < 48 {
                    continue;
                }
                for k in 0..n {
                    let Ok(pe) = read_vec(src, table * sector + k * size, 48) else { break };
                    let first = le_u64(&pe, 32);
                    if first != 0 {
                        starts.push(first * sector);
                    }
                }
            }
        } else if kind != 0 && lba != 0 {
            starts.push(lba * 512);
        }
    }
    starts.into_iter().find(|&o| is_volume(o))
}
