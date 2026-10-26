// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! ATRAC9 audio decoder.
//!
//! Port of LibAtrac9 (MIT, Alex Barney). The decoder is bit-exact with the
//! reference: identical tables, double-precision dequantisation/IMDCT and
//! `floor(x + 0.5)` rounding to 16-bit PCM. Malformed input never panics;
//! conditions the reference reports as exceptions become [`Atrac9Error`]s.

// Index loops deliberately mirror the reference implementation.
#![allow(clippy::needless_range_loop)]

mod band_extension;
mod bit_allocation;
mod bit_reader;
mod block;
mod config;
mod decoder;
mod huffman;
mod huffman_codebooks;
mod mdct;
mod quantization;
mod rng;
mod scale_factors;
mod tables;
mod unpack;

#[cfg(test)]
pub(crate) mod tests;

pub use config::{Atrac9Config, BlockType, ChannelConfig};
pub use decoder::{Atrac9Decoder, PcmEncoding};

/// Errors reported by the ATRAC9 decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Atrac9Error {
    /// The 4-byte configuration word is malformed.
    InvalidConfig(String),
    /// An input or output buffer is smaller than one superframe.
    BufferTooSmall,
    /// The bitstream is malformed.
    InvalidData(String),
}

impl std::fmt::Display for Atrac9Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Atrac9Error::InvalidConfig(msg) => write!(f, "invalid ATRAC9 config: {msg}"),
            Atrac9Error::BufferTooSmall => f.write_str("ATRAC9 buffer is too small"),
            Atrac9Error::InvalidData(msg) => write!(f, "invalid ATRAC9 data: {msg}"),
        }
    }
}

impl std::error::Error for Atrac9Error {}

#[inline]
pub(crate) fn invalid(msg: &str) -> Atrac9Error {
    Atrac9Error::InvalidData(msg.to_owned())
}
