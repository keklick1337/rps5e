// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The guest BVH format the mirrors read, as plain functions over bytes
//! (the kernels in `kernels/` do the same on the GPU; tests check both).
//!
//! A BVH is one 256-byte-aligned block starting with a 256-byte header:
//! `+0` magic "PSR_BVHL", `+16` size in bytes, `+40` end of the leaf-entry
//! array, `+48` offset of the instance array (0: a bottom level), `+88`
//! first node index after the triangle nodes (64-byte units: the
//! leaf-entry array starts at `64 * [+88]`), `+104` triangle-node count,
//! `+112` instance count, `+124` first triangle-node index, `+252` version
//! (9). Node pointers carry the node type in bits 2:0 and the node's
//! 64-byte index in the rest (`base + (ptr & !7) * 8`); the root is the
//! BOX32 node at byte 256 (pointer `0x25`).
//!
//! Leaf entries are 8 bytes: a triangle-node pointer (type 0..3 picks the
//! node's triangle) and the primitive info (bit 31 ends a leaf list, bit 30
//! opaque, bits 29:0 the primitive index). A triangle node holds five
//! float3 vertices at 0, 12, 24, 36, 48 and the triangle-id word at 60.
//!
//! Instance records are 128 bytes: world-to-object rows at 0/16/32, the
//! BLAS address in 8-byte units (bits 53:0) and flags (63:54) at 48, the
//! BLAS root pointer at 56, custom index (23:0) and mask (31:24) at 60,
//! object-to-world rows at 64/80/96.

pub const MAGIC: [u32; 2] = [0x5F52_5350, 0x4C48_5642];
pub const VERSION: u32 = 9;

fn word(b: &[u8], at: usize) -> u32 {
    b.get(at..at + 4).map_or(0, |w| u32::from_le_bytes(w.try_into().unwrap()))
}

fn float(b: &[u8], at: usize) -> f32 {
    f32::from_bits(word(b, at))
}

/// The header fields the mirrors use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub size: u32,
    /// Byte range of the leaf-entry array.
    pub leaves: (u32, u32),
    pub instance_array: u32,
    pub instance_count: u32,
    pub triangle_nodes: (u32, u32),
}

impl Header {
    /// The header of a BVH block (None: not the format, or not version 9).
    pub fn parse(b: &[u8]) -> Option<Header> {
        if [word(b, 0), word(b, 4)] != MAGIC || word(b, 252) != VERSION {
            return None;
        }
        let first_after = word(b, 88);
        Some(Header {
            size: word(b, 16),
            leaves: (first_after.wrapping_mul(64), word(b, 40)),
            instance_array: word(b, 48),
            instance_count: word(b, 112),
            triangle_nodes: (word(b, 124), word(b, 104)),
        })
    }

    /// Leaf entries of a bottom level (as the instance kernel counts them).
    pub fn leaf_entries(&self) -> u32 {
        self.leaves.1.wrapping_sub(self.leaves.0) / 8
    }

    pub fn is_top_level(&self) -> bool {
        self.instance_array != 0
    }
}

/// The vertex slots of triangle type `ty` within a triangle node.
pub fn triangle_slots(ty: u32) -> Option<[usize; 3]> {
    match ty {
        0 => Some([0, 1, 2]),
        1 => Some([1, 3, 2]),
        2 => Some([2, 3, 4]),
        3 => Some([2, 4, 0]),
        _ => None,
    }
}

/// The triangle of leaf entry `index` of the BLAS block `b` (None: not a
/// triangle entry, or a vertex the guest marks unused).
pub fn leaf_triangle(b: &[u8], header: &Header, index: u32) -> Option<[[f32; 3]; 3]> {
    let at = header.leaves.0 as usize + index as usize * 8;
    let ptr = word(b, at);
    let slots = triangle_slots(ptr & 7)?;
    let node = (ptr & !7) as usize * 8;
    let v = slots.map(|s| [0, 1, 2].map(|k| float(b, node + s * 12 + k * 4)));
    v.iter().flatten().all(|x| x.abs() < 1e30).then_some(v)
}

/// One instance record.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    pub blas: u64,
    pub flags: u32,
    pub root: u32,
    pub custom: u32,
    pub mask: u32,
    /// Row-major 3x4 object-to-world transform (the host instance's).
    pub object_to_world: [f32; 12],
}

/// Instance record `i` of the TLAS block `b`.
pub fn instance(b: &[u8], header: &Header, i: u32) -> Instance {
    let at = header.instance_array as usize + i as usize * 128;
    let q = word(b, at + 48) as u64 | (word(b, at + 52) as u64) << 32;
    let id = word(b, at + 60);
    Instance {
        blas: (q & ((1 << 54) - 1)) << 3,
        flags: (q >> 54) as u32,
        root: word(b, at + 56),
        custom: id & 0xFF_FFFF,
        mask: id >> 24,
        object_to_world: std::array::from_fn(|k| float(b, at + 64 + k * 4)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A BLAS block: header, root BOX32 node with one leaf child, one
    /// triangle node (five vertices) and `entries` leaf entries using types
    /// 0..3 in turn.
    pub fn blas_bytes(vertices: [[f32; 3]; 5], entries: u32) -> Vec<u8> {
        let mut b = vec![0u8; 1024];
        let put = |b: &mut Vec<u8>, at: usize, w: u32| b[at..at + 4].copy_from_slice(&w.to_le_bytes());
        put(&mut b, 0, MAGIC[0]);
        put(&mut b, 4, MAGIC[1]);
        put(&mut b, 16, 1024);
        put(&mut b, 252, VERSION);
        // Triangle node at byte 512 (index 8); leaf entries from byte 576
        // (index 9).
        put(&mut b, 124, 8);
        put(&mut b, 104, 1);
        put(&mut b, 88, 9);
        put(&mut b, 40, 576 + entries * 8);
        for (s, v) in vertices.iter().enumerate() {
            for k in 0..3 {
                put(&mut b, 512 + s * 12 + k * 4, v[k].to_bits());
            }
        }
        for j in 0..entries {
            put(&mut b, 576 + j as usize * 8, (8 << 3) | (j % 4));
            put(&mut b, 580 + j as usize * 8, j | if j + 1 == entries { 1 << 31 } else { 0 });
        }
        // Root box: child 0 = the leaf list at byte 576 (pointer 576 | 0).
        put(&mut b, 256, 576);
        for c in 1..4 {
            put(&mut b, 256 + c * 4, u32::MAX);
        }
        b
    }

    #[test]
    fn header_and_leaf_triangles() {
        let v = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b = blas_bytes(v, 4);
        let h = Header::parse(&b).unwrap();
        assert_eq!(h.leaf_entries(), 4);
        assert!(!h.is_top_level());
        assert_eq!(h.triangle_nodes, (8, 1));
        assert_eq!(leaf_triangle(&b, &h, 0), Some([v[0], v[1], v[2]]));
        assert_eq!(leaf_triangle(&b, &h, 1), Some([v[1], v[3], v[2]]));
        assert_eq!(leaf_triangle(&b, &h, 2), Some([v[2], v[3], v[4]]));
        assert_eq!(leaf_triangle(&b, &h, 3), Some([v[2], v[4], v[0]]));
        let mut bad = b.clone();
        bad[252] = 8;
        assert!(Header::parse(&bad).is_none());
    }

    #[test]
    fn unused_vertices_make_no_triangle() {
        let v = [[0.0, 0.0, 0.0], [f32::MAX, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b = blas_bytes(v, 3);
        let h = Header::parse(&b).unwrap();
        assert_eq!(leaf_triangle(&b, &h, 0), None);
        assert_eq!(leaf_triangle(&b, &h, 1), None);
        assert_eq!(leaf_triangle(&b, &h, 2), Some([v[2], v[3], v[4]]));
    }

    #[test]
    fn instance_records() {
        let mut b = vec![0u8; 512];
        let put = |b: &mut Vec<u8>, at: usize, w: u32| b[at..at + 4].copy_from_slice(&w.to_le_bytes());
        put(&mut b, 0, MAGIC[0]);
        put(&mut b, 4, MAGIC[1]);
        put(&mut b, 252, VERSION);
        put(&mut b, 48, 256);
        put(&mut b, 112, 1);
        let blas = 0x2_f829_8800u64;
        let q = (blas >> 3) | 0x81 << 54;
        put(&mut b, 256 + 48, q as u32);
        put(&mut b, 256 + 52, (q >> 32) as u32);
        put(&mut b, 256 + 56, 0x25);
        put(&mut b, 256 + 60, 0x3F00_0007);
        let o2w = [1.0f32, 0.0, 0.0, -73.0, 0.0, 1.0, 0.0, 114.0, 0.0, 0.0, 1.0, 5.5];
        for (k, x) in o2w.iter().enumerate() {
            put(&mut b, 256 + 64 + k * 4, x.to_bits());
        }
        let h = Header::parse(&b).unwrap();
        assert!(h.is_top_level());
        let i = instance(&b, &h, 0);
        assert_eq!(i, Instance { blas, flags: 0x81, root: 0x25, custom: 7, mask: 0x3F, object_to_world: o2w });
    }
}
