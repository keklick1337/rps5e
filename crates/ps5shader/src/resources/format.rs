// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Descriptor format tables: the gfx10 unified FORMAT field of buffer
//! descriptors and MTBUF instructions, and the image-format facts the
//! resource planner needs.

/// The RDNA2 unified FORMAT field (V# word 3 bits 18:12, MTBUF FORMAT) as the
/// legacy (data-format, number-format) pair.
pub mod unified_format {
    /// `(data_format, number_format)` of a unified format, `None` for a
    /// reserved encoding. The table is the hardware's exact one: reserved
    /// spellings (10_11_11 integer/normalized, 10_10_10_2 SCALED, ...) are
    /// not derived from ranges.
    pub fn decode(unified: u32) -> Option<(u32, u32)> {
        let pair = match unified {
            0 => (0, 0),
            1 => (1, 0),
            2 => (1, 1),
            3 => (1, 2),
            4 => (1, 3),
            5 => (1, 4),
            6 => (1, 5),
            7 => (2, 0),
            8 => (2, 1),
            9 => (2, 2),
            10 => (2, 3),
            11 => (2, 4),
            12 => (2, 5),
            13 => (2, 7),
            14 => (3, 0),
            15 => (3, 1),
            16 => (3, 2),
            17 => (3, 3),
            18 => (3, 4),
            19 => (3, 5),
            20 => (4, 4),
            21 => (4, 5),
            22 => (4, 7),
            23 => (5, 0),
            24 => (5, 1),
            25 => (5, 2),
            26 => (5, 3),
            27 => (5, 4),
            28 => (5, 5),
            29 => (5, 7),
            36 => (6, 7),
            43 => (7, 7),
            44 => (8, 0),
            45 => (8, 1),
            48 => (8, 4),
            49 => (8, 5),
            50 => (9, 0),
            51 => (9, 1),
            52 => (9, 2),
            53 => (9, 3),
            54 => (9, 4),
            55 => (9, 5),
            56 => (10, 0),
            57 => (10, 1),
            58 => (10, 2),
            59 => (10, 3),
            60 => (10, 4),
            61 => (10, 5),
            62 => (11, 4),
            63 => (11, 5),
            64 => (11, 7),
            65 => (12, 0),
            66 => (12, 1),
            67 => (12, 2),
            68 => (12, 3),
            69 => (12, 4),
            70 => (12, 5),
            71 => (12, 7),
            72 => (13, 4),
            73 => (13, 5),
            74 => (13, 7),
            75 => (14, 4),
            76 => (14, 5),
            77 => (14, 7),
            128 => (1, 9),
            129 => (3, 9),
            130 => (10, 9),
            // Image-only encodings without a legacy data format keep the
            // unified identifier for exact downstream dispatch.
            131 => (131, 7),
            132 => (34, 7),
            133 => (16, 0),
            134 => (17, 0),
            135 => (18, 0),
            136 => (19, 0),
            137 => (137, 0),
            138 => (138, 0),
            139 => (139, 0),
            140 => (4, 7),
            141 => (20, 0),
            142 => (20, 4),
            143 => (21, 0),
            144 => (21, 4),
            145 => (22, 4),
            146 => (22, 7),
            147 => (32, 0),
            148 => (32, 1),
            149 => (32, 4),
            150 => (32, 9),
            151 => (33, 0),
            152 => (33, 1),
            153 => (33, 4),
            154 => (33, 9),
            169 => (169, 0),
            170 => (170, 9),
            171 => (171, 0),
            172 => (172, 9),
            173 => (173, 0),
            174 => (174, 9),
            175 => (175, 0),
            176 => (176, 1),
            177 => (177, 0),
            178 => (178, 1),
            179 => (179, 7),
            180 => (180, 7),
            181 => (181, 0),
            182 => (182, 9),
            _ => (0, 0),
        };
        (unified == 0 || pair.0 != 0).then_some(pair)
    }

    /// Components of a legacy data format; 0 for formats without a layout.
    pub fn component_count(data_format: u32) -> u32 {
        match data_format {
            1 | 2 | 4 => 1,
            3 | 5 | 11 => 2,
            6 | 7 | 13 => 3,
            8 | 9 | 10 | 12 | 14 => 4,
            _ => 0,
        }
    }

    /// One component of a legacy data format: byte offset of naturally
    /// aligned components, bit offset inside the first dword for packed
    /// fields.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ComponentLayout {
        pub component: u32,
        pub data_format: u32,
        pub byte_offset: u32,
        pub bit_offset: u32,
        pub bit_count: u32,
    }

    const fn l(component: u32, data_format: u32, byte_offset: u32, bit_offset: u32, bit_count: u32) -> ComponentLayout {
        ComponentLayout { component, data_format, byte_offset, bit_offset, bit_count }
    }

    /// Ordered by component, then format. Packed formats are named from the
    /// most significant field, so 2_10_10_10 (unified 50) has X in bits 0..9.
    pub const COMPONENT_LAYOUTS: &[ComponentLayout] = &[
        l(0, 1, 0, 0, 8),   // 8
        l(0, 2, 0, 0, 16),  // 16
        l(0, 3, 0, 0, 8),   // 8_8
        l(0, 4, 0, 0, 32),  // 32
        l(0, 5, 0, 0, 16),  // 16_16
        l(0, 6, 0, 0, 11),  // 10_11_11
        l(0, 7, 0, 0, 10),  // 11_11_10
        l(0, 8, 0, 0, 2),   // 10_10_10_2
        l(0, 9, 0, 0, 10),  // 2_10_10_10
        l(0, 10, 0, 0, 8),  // 8_8_8_8
        l(0, 11, 0, 0, 32), // 32_32
        l(0, 12, 0, 0, 16), // 16_16_16_16
        l(0, 13, 0, 0, 32), // 32_32_32
        l(0, 14, 0, 0, 32), // 32_32_32_32
        l(1, 3, 1, 0, 8),
        l(1, 5, 2, 0, 16),
        l(1, 6, 0, 11, 11),
        l(1, 7, 0, 10, 11),
        l(1, 8, 0, 2, 10),
        l(1, 9, 0, 10, 10),
        l(1, 10, 1, 0, 8),
        l(1, 11, 4, 0, 32),
        l(1, 12, 2, 0, 16),
        l(1, 13, 4, 0, 32),
        l(1, 14, 4, 0, 32),
        l(2, 6, 0, 22, 10),
        l(2, 7, 0, 21, 11),
        l(2, 8, 0, 12, 10),
        l(2, 9, 0, 20, 10),
        l(2, 10, 2, 0, 8),
        l(2, 12, 4, 0, 16),
        l(2, 13, 8, 0, 32),
        l(2, 14, 8, 0, 32),
        l(3, 8, 0, 22, 10),
        l(3, 9, 0, 30, 2),
        l(3, 10, 3, 0, 8),
        l(3, 12, 6, 0, 16),
        l(3, 14, 12, 0, 32),
    ];

    /// `(byte_offset, bit_offset, bit_count)` of one component.
    pub fn component_layout(data_format: u32, component: u32) -> Option<(u32, u32, u32)> {
        COMPONENT_LAYOUTS
            .iter()
            .find(|c| c.component == component && c.data_format == data_format)
            .map(|c| (c.byte_offset, c.bit_offset, c.bit_count))
    }

    /// Bytes an access of the first `component_count` components touches.
    pub fn access_byte_size(data_format: u32, component_count: u32) -> u32 {
        COMPONENT_LAYOUTS
            .iter()
            .filter(|c| c.data_format == data_format && c.component < component_count)
            .map(|c| c.byte_offset + (c.bit_offset + c.bit_count).div_ceil(8))
            .max()
            .unwrap_or(0)
    }

    /// Every component is one whole, aligned dword.
    pub fn has_whole_dword_components(data_format: u32) -> bool {
        let mut known = false;
        for c in COMPONENT_LAYOUTS.iter().filter(|c| c.data_format == data_format) {
            known = true;
            if c.bit_count != 32 || c.bit_offset != 0 {
                return false;
            }
        }
        known
    }
}

/// Numeric class of a sampled image format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ImageNumericClass {
    #[default]
    Unsupported,
    Float,
    Uint,
    Sint,
}

/// Facts about the 9-bit image FORMAT field (T# word 1 bits 28:20).
pub mod image_format {
    use super::ImageNumericClass;

    pub const INVALID: u32 = 0;
    pub const FORMAT_8_USCALED: u32 = 3;
    pub const FORMAT_8X2_USCALED: u32 = 16;
    pub const FORMAT_16_UNORM: u32 = 7;
    pub const FORMAT_32_UINT: u32 = 20;
    pub const FORMAT_32_SINT: u32 = 21;
    pub const FORMAT_32_FLOAT: u32 = 22;
    pub const FORMAT_32_32_UINT: u32 = 62;
    pub const FORMAT_32_32_SINT: u32 = 63;
    pub const FORMAT_32_32_FLOAT: u32 = 64;
    pub const FORMAT_32_32_32_32_SINT: u32 = 76;
    pub const FORMAT_11X2X10_UNORM: u32 = 30;
    pub const FORMAT_11X2X10_UINT: u32 = 34;
    pub const FORMAT_10X11X11_FLOAT: u32 = 43;
    pub const MAX_FORMAT: u32 = 182;

    pub const TYPE_1D: u32 = 8;
    pub const TYPE_2D: u32 = 9;
    pub const TYPE_3D: u32 = 10;
    pub const TYPE_CUBE: u32 = 11;
    pub const TYPE_1D_ARRAY: u32 = 12;
    pub const TYPE_2D_ARRAY: u32 = 13;
    pub const TYPE_2D_MSAA: u32 = 14;
    pub const TYPE_2D_MSAA_ARRAY: u32 = 15;

    /// Only these sample as a Vulkan depth format; comparison sampling of
    /// any other format is evaluated in the shader.
    pub fn has_depth_equivalent(format: u32) -> bool {
        matches!(format, FORMAT_16_UNORM | FORMAT_32_FLOAT)
    }

    use ImageNumericClass::{Float as F, Sint as S, Uint as U};
    const SAMPLED_FORMATS: &[(u32, ImageNumericClass)] = &[
        (1, F), (2, F), (FORMAT_8_USCALED, F), (5, U), (6, S), (7, F), (8, F), (11, U), (12, S), (13, F), (14, F), (15, F), (16, F),
        (18, U), (19, S), (20, U), (21, S), (22, F), (23, F), (24, F), (27, U), (28, S), (29, F), (30, F), (34, U),
        (36, F), (43, F), (50, F), (51, F), (54, U), (55, S), (56, F), (57, F), (60, U), (61, S), (62, U), (63, S), (64, F), (65, F), (66, F),
        (69, U), (70, S), (71, F), (72, U), (73, S), (74, F), (75, U), (76, S), (77, F), (128, F), (129, F), (130, F),
        (132, F), (133, F), (134, F), (136, F),
        // FMASK: the sample-to-fragment map, read as raw bits.
        (156, U), (157, U), (158, U), (159, U), (160, U), (161, U), (162, U), (163, U), (164, U), (165, U), (166, U), (167, U), (168, U),
        (169, F), (170, F), (171, F), (172, F), (173, F),
        (174, F), (175, F), (176, F), (177, F), (178, F), (179, F), (180, F), (181, F), (182, F),
    ];

    /// The numeric class a sampled image of `format` has; every format not
    /// in the table is unsupported for sampling.
    pub fn sampled_numeric_class(format: u32) -> ImageNumericClass {
        SAMPLED_FORMATS.iter().find(|(f, _)| *f == format).map_or(ImageNumericClass::Unsupported, |(_, c)| *c)
    }

    /// The format the emitter reads a converted format as (the 11_11_10
    /// UNORM and UINT formats are read as one 32-bit integer and unpacked
    /// in the shader).
    pub fn remap(format: u32) -> u32 {
        match format {
            FORMAT_8_USCALED => 1,
            FORMAT_8X2_USCALED => 14,
            FORMAT_11X2X10_UNORM | FORMAT_11X2X10_UINT => FORMAT_32_UINT,
            _ => format,
        }
    }

    /// The host image of `format` holds its X and Z in its B and R
    /// channels (the 10_11_11 float is stored as the host's 11_11_10 with
    /// the fields reversed): sampled views swap them back through their
    /// component mapping, storage views (identity mapping only) in the
    /// shader.
    pub fn host_reverses_channels(format: u32) -> bool {
        format == FORMAT_10X11X11_FLOAT
    }

    /// Signed integer formats a storage image can be read and written as
    /// through the unsigned view of the same layout (8, 16 and 32-bit
    /// components; the 96-bit 32_32_32 has no storage view).
    pub fn raw_sint_storage_format(format: u32) -> bool {
        matches!(format, 6 | 12 | 19 | FORMAT_32_SINT | 28 | 61 | FORMAT_32_32_SINT | 70 | FORMAT_32_32_32_32_SINT)
    }

    pub fn type_of(descriptor: &[u32]) -> u32 {
        (descriptor[3] >> 28) & 0xF
    }

    pub fn format_of(descriptor: &[u32]) -> u32 {
        (descriptor[1] >> 20) & 0x1FF
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unified_formats() {
        assert_eq!(unified_format::decode(0), Some((0, 0)));
        assert_eq!(unified_format::decode(22), Some((4, 7))); // 32 float
        assert_eq!(unified_format::decode(77), Some((14, 7))); // 32_32_32_32 float
        assert_eq!(unified_format::decode(56), Some((10, 0))); // 8_8_8_8 unorm
        assert_eq!(unified_format::decode(50), Some((9, 0))); // 2_10_10_10 unorm
        assert_eq!(unified_format::decode(132), Some((34, 7)));
        // Reserved encodings.
        for reserved in [30, 31, 35, 37, 46, 47, 78, 127, 155, 168, 183, 255] {
            assert_eq!(unified_format::decode(reserved), None, "{reserved}");
        }
        // Every valid buffer format below 78 decodes to a data format with a
        // component layout.
        for u in 1..78 {
            if let Some((d, _)) = unified_format::decode(u) {
                let n = unified_format::component_count(d);
                assert!((1..=4).contains(&n), "{u}");
                for c in 0..n {
                    assert!(unified_format::component_layout(d, c).is_some(), "{u} component {c}");
                }
            }
        }
    }

    #[test]
    fn component_layouts() {
        assert_eq!(unified_format::component_layout(9, 0), Some((0, 0, 10)));
        assert_eq!(unified_format::component_layout(9, 3), Some((0, 30, 2)));
        assert_eq!(unified_format::access_byte_size(14, 4), 16);
        assert_eq!(unified_format::access_byte_size(14, 2), 8);
        assert_eq!(unified_format::access_byte_size(10, 3), 3);
        assert_eq!(unified_format::access_byte_size(9, 1), 2);
        assert_eq!(unified_format::access_byte_size(99, 4), 0);
        assert!(unified_format::has_whole_dword_components(13));
        assert!(!unified_format::has_whole_dword_components(12));
        assert!(!unified_format::has_whole_dword_components(99));
        // The table is ordered by component, then format.
        assert!(unified_format::COMPONENT_LAYOUTS
            .windows(2)
            .all(|w| (w[0].component, w[0].data_format) < (w[1].component, w[1].data_format)));
    }

    #[test]
    fn image_formats() {
        assert_eq!(image_format::sampled_numeric_class(22), ImageNumericClass::Float);
        assert_eq!(image_format::sampled_numeric_class(20), ImageNumericClass::Uint);
        assert_eq!(image_format::sampled_numeric_class(21), ImageNumericClass::Sint);
        assert_eq!(image_format::sampled_numeric_class(2), ImageNumericClass::Float);
        assert_eq!(image_format::sampled_numeric_class(4), ImageNumericClass::Unsupported);
        assert_eq!(image_format::remap(34), 20);
        assert_eq!(image_format::remap(30), 20);
        assert_eq!(image_format::remap(56), 56);
        assert!(image_format::has_depth_equivalent(7));
        let t = [0, 22 << 20, 0, (image_format::TYPE_2D) << 28, 0, 0, 0, 0];
        assert_eq!(image_format::format_of(&t), 22);
        assert_eq!(image_format::type_of(&t), image_format::TYPE_2D);
    }
}
