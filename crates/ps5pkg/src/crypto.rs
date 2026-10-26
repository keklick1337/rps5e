// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The cryptography of fake (self-made) packages.
//!
//! A fake package encrypts its outer PFS with a key derived from public
//! inputs only: the content id and the package passcode (32 characters,
//! all `'0'` unless the packager chose another one). No console key is
//! involved, so these images can be read here. Retail packages derive the
//! same key from a secret held by the console; they are detected (the key
//! digest stored in the package does not match) and refused.
//!
//! Derivation (index 1 is the PFS key, "EKPFS"):
//! `key(i) = SHA256( SHA256(be32(i)) || SHA256(content_id padded to 48) || passcode )`;
//! the package stores `SHA256(key(i)) XOR key(i)` per index so a passcode
//! can be checked without touching the image.
//!
//! PFS sectors (0x1000 bytes, starting at the first block after the
//! superblock) use AES-128-XTS with the sector number as tweak;
//! `(tweak key, data key) = HMAC-SHA256(EKPFS', le32(1) || seed)` split in
//! halves, where EKPFS' is EKPFS or, for the newer scheme flagged in the
//! package header, `HMAC-SHA256(EKPFS, seed)`.

use crate::io::{read_at, ReadAt, Source};
use aes::cipher::{generic_array::GenericArray, BlockDecrypt, BlockEncrypt, KeyInit};
use aes::Aes128;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use std::io;

/// Passcode used by package tools unless the packager sets another one.
pub const DEFAULT_PASSCODE: &str = "00000000000000000000000000000000";

/// XTS sector size of an encrypted PFS.
pub const XTS_SECTOR_SIZE: u64 = 0x1000;

fn sha256(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut m = <Hmac<Sha256> as Mac>::new_from_slice(key).expect("HMAC accepts any key length");
    m.update(data);
    m.finalize().into_bytes().into()
}

/// Key number `index` of a package derived from its content id and passcode.
pub fn derive_key(content_id: &str, passcode: &str, index: u32) -> [u8; 32] {
    let mut cid = [0u8; 48];
    let n = content_id.len().min(48);
    cid[..n].copy_from_slice(&content_id.as_bytes()[..n]);
    let mut data = [0u8; 96];
    data[..32].copy_from_slice(&sha256(&index.to_be_bytes()));
    data[32..64].copy_from_slice(&sha256(&cid));
    let mut pc = [0u8; 32];
    let n = passcode.len().min(32);
    pc[..n].copy_from_slice(&passcode.as_bytes()[..n]);
    data[64..].copy_from_slice(&pc);
    sha256(&data)
}

/// The digest a package stores for a derived key.
pub fn key_digest(key: &[u8; 32]) -> [u8; 32] {
    let mut d = sha256(key);
    for (a, b) in d.iter_mut().zip(key) {
        *a ^= b;
    }
    d
}

/// (tweak key, data key) of an encrypted PFS.
pub fn pfs_xts_keys(ekpfs: &[u8; 32], seed: &[u8; 16], new_scheme: bool) -> ([u8; 16], [u8; 16]) {
    let base = if new_scheme { hmac_sha256(ekpfs, seed) } else { *ekpfs };
    let mut msg = [0u8; 20];
    msg[..4].copy_from_slice(&1u32.to_le_bytes());
    msg[4..].copy_from_slice(seed);
    let k = hmac_sha256(&base, &msg);
    let (mut tweak, mut data) = ([0u8; 16], [0u8; 16]);
    tweak.copy_from_slice(&k[..16]);
    data.copy_from_slice(&k[16..]);
    (tweak, data)
}

/// Multiply the XTS tweak by x in GF(2^128) (little-endian convention).
fn gf_double(t: &mut [u8; 16]) {
    let mut carry = 0u8;
    for b in t.iter_mut() {
        let next = *b >> 7;
        *b = (*b << 1) | carry;
        carry = next;
    }
    if carry != 0 {
        t[0] ^= 0x87;
    }
}

/// AES-128-XTS over one sector, in place.
pub struct Xts {
    data: Aes128,
    tweak: Aes128,
}

impl Xts {
    pub fn new(tweak_key: &[u8; 16], data_key: &[u8; 16]) -> Self {
        Self {
            data: Aes128::new(GenericArray::from_slice(data_key)),
            tweak: Aes128::new(GenericArray::from_slice(tweak_key)),
        }
    }

    fn initial_tweak(&self, sector: u64) -> [u8; 16] {
        let mut t = [0u8; 16];
        t[..8].copy_from_slice(&sector.to_le_bytes());
        let mut block = GenericArray::from(t);
        self.tweak.encrypt_block(&mut block);
        block.into()
    }

    fn apply(&self, buf: &mut [u8], sector: u64, encrypt: bool) {
        let mut t = self.initial_tweak(sector);
        for chunk in buf.chunks_exact_mut(16) {
            for (c, k) in chunk.iter_mut().zip(&t) {
                *c ^= k;
            }
            let block = GenericArray::from_mut_slice(chunk);
            if encrypt {
                self.data.encrypt_block(block);
            } else {
                self.data.decrypt_block(block);
            }
            for (c, k) in chunk.iter_mut().zip(&t) {
                *c ^= k;
            }
            gf_double(&mut t);
        }
    }

    pub fn decrypt_sector(&self, buf: &mut [u8], sector: u64) {
        self.apply(buf, sector, false);
    }

    pub fn encrypt_sector(&self, buf: &mut [u8], sector: u64) {
        self.apply(buf, sector, true);
    }
}

/// An XTS-encrypted PFS image presented decrypted. Sectors before
/// `start_sector` (the superblock) are stored in the clear.
pub struct XtsReader {
    inner: Source,
    xts: Xts,
    start_sector: u64,
}

impl XtsReader {
    pub fn new(inner: Source, xts: Xts, start_sector: u64) -> Self {
        Self { inner, xts, start_sector }
    }
}

impl ReadAt for XtsReader {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        if offset.checked_add(buf.len() as u64).is_none_or(|e| e > self.len()) {
            return Err(crate::io::eof());
        }
        let mut sector_buf = [0u8; XTS_SECTOR_SIZE as usize];
        let mut done = 0usize;
        while done < buf.len() {
            let pos = offset + done as u64;
            let sector = pos / XTS_SECTOR_SIZE;
            let within = (pos % XTS_SECTOR_SIZE) as usize;
            let n = (XTS_SECTOR_SIZE as usize - within).min(buf.len() - done);
            if sector < self.start_sector {
                self.inner.read_exact_at(&mut buf[done..done + n], pos)?;
            } else {
                let got = read_at(&*self.inner, &mut sector_buf, sector * XTS_SECTOR_SIZE)?;
                // A trailing partial sector cannot be XTS data; zero-fill.
                sector_buf[got..].fill(0);
                self.xts.decrypt_sector(&mut sector_buf, sector);
                buf[done..done + n].copy_from_slice(&sector_buf[within..within + n]);
            }
            done += n;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xts_roundtrip_and_tweak_dependence() {
        let xts = Xts::new(&[1u8; 16], &[2u8; 16]);
        let plain: Vec<u8> = (0..4096u32).map(|i| (i * 7) as u8).collect();
        let mut a = plain.clone();
        xts.encrypt_sector(&mut a, 16);
        assert_ne!(a, plain);
        let mut b = plain.clone();
        xts.encrypt_sector(&mut b, 17);
        assert_ne!(a, b, "tweak must depend on the sector number");
        xts.decrypt_sector(&mut a, 16);
        assert_eq!(a, plain);
    }

    #[test]
    fn aes_xts_matches_ieee_1619_vector() {
        // IEEE P1619 XTS-AES-128 vector 2: key1 = 11..11, key2 = 22..22,
        // data unit 0x3333333333, plaintext 44..44 (32 bytes).
        let xts = Xts::new(&[0x22; 16], &[0x11; 16]);
        let mut buf = [0x44u8; 32];
        xts.encrypt_sector(&mut buf, 0x33_3333_3333);
        let want = [
            0xc4, 0x54, 0x18, 0x5e, 0x6a, 0x16, 0x93, 0x6e, 0x39, 0x33, 0x40, 0x38, 0xac, 0xef, 0x83, 0x8b, 0xfb, 0x18,
            0x6f, 0xff, 0x74, 0x80, 0xad, 0xc4, 0x28, 0x93, 0x82, 0xec, 0xd6, 0xd3, 0x94, 0xf0,
        ];
        assert_eq!(buf, want);
    }

    #[test]
    fn key_digest_is_self_consistent() {
        let cid = "UP0000-TEST00000_00-0000000000000000";
        let k1 = derive_key(cid, DEFAULT_PASSCODE, 1);
        let k1b = derive_key(cid, DEFAULT_PASSCODE, 1);
        assert_eq!(k1, k1b);
        assert_ne!(k1, derive_key(cid, DEFAULT_PASSCODE, 0));
        assert_ne!(k1, derive_key(cid, "11111111111111111111111111111111", 1));
        assert_ne!(key_digest(&k1), k1);
    }
}
