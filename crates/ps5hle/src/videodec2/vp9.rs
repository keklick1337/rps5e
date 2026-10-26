// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! VP9 uncompressed frame headers (VP9 bitstream specification 6.2) and
//! superframe indexes (annex B), for sceVideodec2GetVp9PictureInfo.
//!
//! The parser keeps the decoder's eight reference slots and the colour
//! configuration of the last intra frame, so inter frames and "show
//! existing frame" packets report the size and colour of what they refer to.

/// SceVideodec2Vp9PictureInfo (88 bytes).
pub(super) const PICTURE_INFO_SIZE: u64 = 88;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Info {
    pub valid: bool,
    pub pts: u64,
    pub dts: u64,
    pub attached: u64,
    pub profile: u32,
    pub level: u32,
    pub key_frame: bool,
    pub intra_only: bool,
    pub bit_depth: u8,
    pub color_range: u8,
    pub color_space: u32,
    pub num_frames: u32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub render_width: u32,
    pub render_height: u32,
}

impl Info {
    /// The structure after `thisSize` (bytes 8..88).
    pub fn to_bytes(&self) -> [u8; PICTURE_INFO_SIZE as usize - 8] {
        let mut b = [0u8; PICTURE_INFO_SIZE as usize - 8];
        let mut put = |o: usize, v: &[u8]| b[o - 8..o - 8 + v.len()].copy_from_slice(v);
        put(8, &[self.valid as u8]);
        put(16, &self.pts.to_le_bytes());
        put(24, &self.dts.to_le_bytes());
        put(32, &self.attached.to_le_bytes());
        put(40, &self.profile.to_le_bytes());
        put(44, &self.level.to_le_bytes());
        put(48, &(self.key_frame as u32).to_le_bytes());
        put(52, &(self.intra_only as u32).to_le_bytes());
        put(56, &[self.bit_depth, self.color_range]);
        put(60, &self.color_space.to_le_bytes());
        put(64, &self.num_frames.to_le_bytes());
        put(68, &self.frame_width.to_le_bytes());
        put(72, &self.frame_height.to_le_bytes());
        put(76, &self.render_width.to_le_bytes());
        put(80, &self.render_height.to_le_bytes());
        b
    }
}

/// Reference slots and the colour configuration of the last intra frame.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct State {
    references: [Info; 8],
    colors: Info,
}

struct Bits<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Bits<'_> {
    fn read(&mut self, count: u32) -> Result<u32, &'static str> {
        let mut v = 0;
        for _ in 0..count {
            let byte = *self.bytes.get(self.position / 8).ok_or("truncated header")?;
            v = (v << 1) | ((byte >> (7 - self.position % 8)) & 1) as u32;
            self.position += 1;
        }
        Ok(v)
    }

    fn flag(&mut self) -> Result<bool, &'static str> {
        Ok(self.read(1)? != 0)
    }
}

/// One frame of a packet: its info and whether it is shown.
fn parse_frame(bytes: &[u8], state: &mut State, count: u32) -> Result<(Info, bool), &'static str> {
    let mut bits = Bits { bytes, position: 0 };
    if bits.read(2)? != 2 {
        return Err("invalid frame marker");
    }
    let low = bits.read(1)?;
    let profile = low | (bits.read(1)? << 1);
    if profile == 3 && bits.flag()? {
        return Err("invalid reserved bit");
    }
    if bits.flag()? {
        // show_existing_frame
        let mut info = state.references[bits.read(3)? as usize];
        if !info.valid {
            return Err("missing display reference");
        }
        info.key_frame = false;
        info.intra_only = false;
        info.profile = profile;
        info.num_frames = count;
        return Ok((info, true));
    }
    let key = !bits.flag()?;
    let shown = bits.flag()?;
    let resilient = bits.flag()?;
    let intra = !key && !shown && bits.flag()?;
    if !key && !resilient {
        bits.read(2)?; // reset_frame_context
    }
    let mut info = Info::default();
    let refresh;
    if key || intra {
        if bits.read(24)? != 0x49_8342 {
            return Err("invalid sync code");
        }
        info.bit_depth = 8;
        info.color_space = 1;
        if key || profile != 0 {
            if profile >= 2 {
                info.bit_depth = if bits.flag()? { 12 } else { 10 };
            }
            info.color_space = bits.read(3)?;
            if info.color_space != 7 {
                info.color_range = bits.read(1)? as u8;
                if profile == 1 || profile == 3 {
                    bits.read(2)?; // subsampling
                    if bits.flag()? {
                        return Err("invalid colour reserved bit");
                    }
                }
            } else {
                info.color_range = 1;
                if profile != 1 && profile != 3 {
                    return Err("RGB in profile 0 or 2");
                }
                if bits.flag()? {
                    return Err("invalid RGB reserved bit");
                }
            }
        }
        refresh = if key { 0xFF } else { bits.read(8)? };
        info.frame_width = bits.read(16)? + 1;
        info.frame_height = bits.read(16)? + 1;
    } else {
        refresh = bits.read(8)?;
        let mut indexes = [0usize; 3];
        for index in &mut indexes {
            *index = bits.read(3)? as usize;
            bits.read(1)?; // sign bias
        }
        info = state.colors;
        if !info.valid {
            return Err("missing colour configuration");
        }
        let mut inherited = false;
        for index in indexes {
            if bits.flag()? {
                let reference = state.references[index];
                if !reference.valid {
                    return Err("missing size reference");
                }
                info.frame_width = reference.frame_width;
                info.frame_height = reference.frame_height;
                inherited = true;
                break;
            }
        }
        if !inherited {
            info.frame_width = bits.read(16)? + 1;
            info.frame_height = bits.read(16)? + 1;
        }
    }
    let different = bits.flag()?;
    info.render_width = if different { bits.read(16)? + 1 } else { info.frame_width };
    info.render_height = if different { bits.read(16)? + 1 } else { info.frame_height };
    info.valid = true;
    info.profile = profile;
    info.key_frame = key;
    info.intra_only = intra;
    info.num_frames = count;
    if key || intra {
        state.colors = info;
    }
    for (slot, reference) in state.references.iter_mut().enumerate() {
        if refresh & (1 << slot) != 0 {
            *reference = info;
        }
    }
    Ok((info, shown))
}

/// The frames of a packet (a superframe holds several) with whether each
/// is shown. `state` changes only when the whole packet parses.
pub(super) fn parse_packet(bytes: &[u8], state: &mut State) -> Result<Vec<(Info, bool)>, &'static str> {
    let Some(&marker) = bytes.last() else { return Err("empty packet") };
    let count = (marker & 7) as usize + 1;
    let magnitude = ((marker >> 3) & 3) as usize + 1;
    let index_size = 2 + count * magnitude;
    let mut frames = Vec::new();
    if marker & 0xE0 == 0xC0 && bytes.len() >= index_size && bytes[bytes.len() - index_size] == marker {
        let end = bytes.len() - index_size;
        let mut offset = 0;
        for frame in 0..count {
            let size = (0..magnitude).fold(0usize, |s, b| s | (bytes[end + 1 + frame * magnitude + b] as usize) << (8 * b));
            if size == 0 || size > end - offset {
                return Err("invalid superframe size");
            }
            frames.push(&bytes[offset..offset + size]);
            offset += size;
        }
        if offset != end {
            return Err("superframe holds unindexed bytes");
        }
    } else {
        frames.push(bytes);
    }
    let mut next = *state;
    let parsed = frames.iter().map(|f| parse_frame(f, &mut next, frames.len() as u32)).collect::<Result<Vec<_>, _>>()?;
    *state = next;
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A key frame header of `width` x `height`, profile 0, BT.601 limited
    /// range, optionally with a render size.
    fn key_frame(width: u32, height: u32, render: Option<(u32, u32)>, shown: bool) -> Vec<u8> {
        let mut bits = Vec::new();
        let mut push = |v: u32, n: u32| (0..n).rev().for_each(|i| bits.push((v >> i) & 1 != 0));
        push(2, 2); // frame marker
        push(0, 2); // profile 0
        push(0, 1); // show_existing_frame
        push(0, 1); // key frame
        push(shown as u32, 1);
        push(0, 1); // error resilient
        push(0x49_8342, 24);
        push(2, 3); // colour space
        push(0, 1); // limited range
        push(width - 1, 16);
        push(height - 1, 16);
        match render {
            Some((w, h)) => {
                push(1, 1);
                push(w - 1, 16);
                push(h - 1, 16);
            }
            None => push(0, 1),
        }
        bits.resize(bits.len().div_ceil(8) * 8, false);
        bits.chunks(8).map(|c| c.iter().fold(0u8, |b, &x| b << 1 | x as u8)).collect()
    }

    #[test]
    fn key_frame_render_size_and_existing_frame() {
        let mut state = State::default();
        let frames = parse_packet(&key_frame(64, 40, Some((128, 80)), true), &mut state).unwrap();
        assert_eq!(frames.len(), 1);
        let (info, shown) = frames[0];
        assert!(shown && info.valid && info.key_frame && !info.intra_only);
        assert_eq!((info.frame_width, info.frame_height, info.render_width, info.render_height), (64, 40, 128, 80));
        assert_eq!((info.bit_depth, info.color_space, info.color_range, info.num_frames), (8, 2, 0, 1));
        // Show existing frame 3: the key frame refreshed every slot.
        let (existing, shown) = parse_packet(&[0x8B], &mut state).unwrap()[0];
        assert!(shown && existing.valid && !existing.key_frame);
        assert_eq!((existing.frame_width, existing.render_width), (64, 128));
    }

    #[test]
    fn superframes_and_errors() {
        let mut state = State::default();
        assert_eq!(parse_packet(&[0x88], &mut state), Err("missing display reference"));
        assert_eq!(parse_packet(&[0], &mut state), Err("invalid frame marker"));
        assert_eq!(parse_packet(&[0x82], &mut state), Err("truncated header"));
        assert_eq!(parse_packet(&[0xC1, 0, 0, 0xC1], &mut state), Err("invalid superframe size"));
        let (a, b) = (key_frame(64, 40, None, false), key_frame(32, 16, None, true));
        let mut sf = a.clone();
        sf.extend_from_slice(&b);
        sf.push(0xC9);
        for f in [&a, &b] {
            sf.extend_from_slice(&(f.len() as u16).to_le_bytes());
        }
        sf.push(0xC9);
        let frames = parse_packet(&sf, &mut state).unwrap();
        assert_eq!(frames.iter().map(|f| (f.1, f.0.frame_width, f.0.num_frames)).collect::<Vec<_>>(), [(false, 64, 2), (true, 32, 2)]);
        // A failed packet leaves the state alone.
        let before = state.references[0];
        let mut bad = sf.clone();
        bad[0] = 0;
        assert!(parse_packet(&bad, &mut state).is_err());
        assert_eq!(state.references[0], before);
    }

    #[test]
    fn info_layout() {
        let info = Info { valid: true, pts: 1, dts: 2, attached: 3, profile: 1, key_frame: true, bit_depth: 10, color_range: 1, color_space: 5, num_frames: 2, frame_width: 64, frame_height: 40, render_width: 32, render_height: 20, ..Info::default() };
        let b = info.to_bytes();
        let at = |o: usize| u32::from_le_bytes(b[o - 8..o - 4].try_into().unwrap());
        assert_eq!(b[0], 1);
        assert_eq!((at(16), at(24), at(32), at(40)), (1, 2, 3, 1));
        assert_eq!((at(48), at(52), b[56 - 8], b[57 - 8], at(60)), (1, 0, 10, 1, 5));
        assert_eq!((at(64), at(68), at(72), at(76), at(80)), (2, 64, 40, 32, 20));
    }
}
