// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

// Guest memory through the device-address page table (binding 0): one
// device address per 2^page_bits-byte guest page, 0 for pages the GPU has
// no copy of (they read as zeros). The table holds the page_low pages
// below 1 TiB first, then the pages of the window at GUEST_WINDOW_BASE,
// then one entry that is always 0, which every other address reads.

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(set = 0, binding = 0, std430) readonly buffer PageTable { uint64_t pages[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Words { uint w[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Floats { float f[]; };

const uint64_t GUEST_WINDOW_BASE = 0x80000000000ul;
const uint64_t GUEST_WINDOW_BYTES = 0x8000000000ul;

uint guest_entry(uint64_t a, uint page_bits, uint page_low) {
    uint64_t page = a >> page_bits;
    if (page < uint64_t(page_low)) return uint(page);
    uint64_t w = a - GUEST_WINDOW_BASE;
    if (w < GUEST_WINDOW_BYTES) return page_low + uint(w >> page_bits);
    return page_low + uint(GUEST_WINDOW_BYTES >> page_bits);
}

uint64_t guest_device(uint64_t a, uint page_bits, uint page_low) {
    uint64_t e = pages[guest_entry(a, page_bits, page_low)];
    if (e == 0ul) return 0ul;
    return e + (a & ((1ul << page_bits) - 1ul));
}

#define GUEST_WORD(a) guest_word_(a, pc.page_bits, pc.page_low)
uint guest_word_(uint64_t a, uint page_bits, uint page_low) {
    uint64_t d = guest_device(a, page_bits, page_low);
    if (d == 0ul) return 0u;
    return Words(d).w[0];
}
#define GUEST_FLOAT(a) uintBitsToFloat(GUEST_WORD(a))
