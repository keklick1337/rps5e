// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! CPU fallbacks for index and primitive forms the host cannot draw
//! directly: 8-bit indices (widened to 16 bits), quad lists (split into
//! triangle lists), and a primitive-restart index other than the all-ones
//! value of the index type (rewritten).

use crate::render::{primitive, IndexType};
use ash::vk;

/// Host index data produced on the CPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostIndices {
    pub bytes: Vec<u8>,
    pub kind: vk::IndexType,
    pub count: u32,
}

impl HostIndices {
    fn from_u32(v: &[u32], wide: bool) -> HostIndices {
        if wide {
            HostIndices { bytes: v.iter().flat_map(|i| i.to_le_bytes()).collect(), kind: vk::IndexType::UINT32, count: v.len() as u32 }
        } else {
            HostIndices { bytes: v.iter().flat_map(|&i| (i as u16).to_le_bytes()).collect(), kind: vk::IndexType::UINT16, count: v.len() as u32 }
        }
    }
}

/// Decode guest indices of `kind`.
pub fn decode(kind: IndexType, bytes: &[u8]) -> Vec<u32> {
    match kind {
        IndexType::U8 => bytes.iter().map(|&b| b as u32).collect(),
        IndexType::U16 => bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).collect(),
        IndexType::U32 => bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
    }
}

fn all_ones(kind: IndexType) -> u32 {
    match kind {
        IndexType::U8 => 0xFF,
        IndexType::U16 => 0xFFFF,
        IndexType::U32 => u32::MAX,
    }
}

/// Whether a guest index buffer must go through the CPU: 8-bit indices,
/// quad lists, or a restart index the host cannot express.
pub fn needs_conversion(kind: IndexType, prim: u32, restart: Option<u32>) -> bool {
    kind == IndexType::U8 || prim == primitive::QUAD_LIST_LEGACY || restart.is_some_and(|r| r != all_ones(kind))
}

/// `RPS5E_QUAD_PROVOKING_LAST=0`: quads are split for the first provoking
/// vertex whatever the draw's mode (the earlier behaviour).
fn quad_provoking_last_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_QUAD_PROVOKING_LAST").is_ok_and(|v| v == "0"))
}

/// Split quads (a, b, c, d) into triangles whose provoking vertex is the
/// quad's: (a, b, c) (a, c, d) with the first provoking vertex,
/// (a, b, d) (b, c, d) with the last; a trailing partial quad is dropped.
pub fn quads_to_triangles(indices: &[u32], provoking_last: bool) -> Vec<u32> {
    let last = provoking_last && quad_provoking_last_enabled();
    let mut out = Vec::with_capacity(indices.len() / 4 * 6);
    for q in indices.chunks_exact(4) {
        if last {
            out.extend_from_slice(&[q[0], q[1], q[3], q[1], q[2], q[3]]);
        } else {
            out.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
        }
    }
    out
}

/// Index list of an auto (non-indexed) quad-list draw of `count` vertices.
pub fn auto_quads(count: u32, provoking_last: bool) -> Vec<u32> {
    let v: Vec<u32> = (0..count - count % 4).collect();
    quads_to_triangles(&v, provoking_last)
}

/// Convert guest indices for the host. `restart`: the guest restart index
/// when primitive restart is enabled. The output is 16-bit unless a value
/// needs 32 bits; restart positions become the host's all-ones value.
/// `provoking_last`: the draw takes flat attributes from the last vertex.
pub fn convert(kind: IndexType, bytes: &[u8], prim: u32, restart: Option<u32>, provoking_last: bool) -> HostIndices {
    let mut v = decode(kind, bytes);
    if prim == primitive::QUAD_LIST_LEGACY {
        v = quads_to_triangles(&v, provoking_last);
    }
    let wide = kind == IndexType::U32;
    let host_restart = if wide { u32::MAX } else { 0xFFFF };
    // Without restart the all-ones value is an ordinary index.
    if let Some(r) = restart {
        let r = r & all_ones(kind);
        for i in v.iter_mut() {
            if *i == r {
                *i = host_restart;
            } else if *i == host_restart {
                // A real index colliding with the host restart value: the
                // closest representable index (rare; clamped).
                *i = host_restart - 1;
            }
        }
    }
    HostIndices::from_u32(&v, wide)
}

/// Generated indices for an auto quad-list draw.
pub fn convert_auto(prim: u32, count: u32, provoking_last: bool) -> Option<HostIndices> {
    (prim == primitive::QUAD_LIST_LEGACY).then(|| {
        let v = auto_quads(count, provoking_last);
        HostIndices::from_u32(&v, count > 0xFFFF)
    })
}

/// Vertex count of a RECT_LIST draw drawn as a strip (one rectangle).
pub fn single_rectangle(count: u32) -> bool {
    matches!(count, 1 | 3 | 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widen_u8_with_restart() {
        let h = convert(IndexType::U8, &[0, 1, 2, 0xFF, 3, 4, 5], primitive::TRI_STRIP, Some(0xFF), false);
        assert_eq!(h.kind, vk::IndexType::UINT16);
        assert_eq!(decode(IndexType::U16, &h.bytes), vec![0, 1, 2, 0xFFFF, 3, 4, 5]);
        assert!(needs_conversion(IndexType::U8, primitive::TRI_LIST, None));
        assert!(!needs_conversion(IndexType::U16, primitive::TRI_STRIP, Some(0xFFFF)));
        assert!(needs_conversion(IndexType::U32, primitive::TRI_STRIP, Some(0xFFFF)));
    }

    #[test]
    fn quads() {
        let h = convert(IndexType::U16, &[0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0], primitive::QUAD_LIST_LEGACY, None, false);
        assert_eq!(decode(IndexType::U16, &h.bytes), vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]);
        assert_eq!(h.count, 12);
        let a = convert_auto(primitive::QUAD_LIST_LEGACY, 6, false).unwrap();
        assert_eq!(decode(IndexType::U16, &a.bytes), vec![0, 1, 2, 0, 2, 3]);
        assert!(convert_auto(primitive::TRI_LIST, 6, false).is_none());
    }

    /// With the last provoking vertex both triangles of a quad end at its
    /// fourth vertex, so its flat attributes come from that vertex
    /// (reference quad-list support).
    #[test]
    fn quads_last_provoking() {
        let h = convert(IndexType::U16, &[0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0], primitive::QUAD_LIST_LEGACY, None, true);
        assert_eq!(decode(IndexType::U16, &h.bytes), vec![0, 1, 3, 1, 2, 3, 4, 5, 7, 5, 6, 7]);
        let a = convert_auto(primitive::QUAD_LIST_LEGACY, 9, true).unwrap();
        assert_eq!(decode(IndexType::U16, &a.bytes), vec![0, 1, 3, 1, 2, 3, 4, 5, 7, 5, 6, 7]);
    }

    #[test]
    fn restart_rewrite_32() {
        let src: Vec<u8> = [5u32, 0xFFFF, 6, 0xFFFF_FFFF].iter().flat_map(|v| v.to_le_bytes()).collect();
        let h = convert(IndexType::U32, &src, primitive::TRI_STRIP, Some(0xFFFF), false);
        assert_eq!(decode(IndexType::U32, &h.bytes), vec![5, u32::MAX, 6, u32::MAX - 1]);
    }
}
