// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The frame capture file (`*.ps5cap`).
//!
//! ```text
//! magic    8 bytes  "PS5FCAP\0"
//! version  u32      VERSION
//! hlen     u32      header bytes
//! header   hlen     UTF-8 "key=value" lines (format, version, page_bytes,
//!                   frames, alias_bytes, gpu_present, start_seconds,
//!                   env.RPS5E_*=... of the capturing process)
//! records  ...      tag u8, len u32, payload (len bytes)
//! ```
//!
//! All integers are little endian, floats are stored as their bits. A
//! reader skips records with tags it does not know. The records, in the
//! order the back-end saw them:
//! - `MAP` / `UNMAP`: a guest mapping (the first ones describe the
//!   mappings when the capture started);
//! - `PATCH` / `ZERO`: guest memory as the back-end read it (whole 4 KiB
//!   pages where readable); `ZERO` for all-zero bytes. Written when the
//!   back-end first reads a page and whenever it reads it with different
//!   contents later, always before the item whose execution read it;
//! - `CPU_ACCESS`: a CPU access to a page the GPU caches watch (the fault
//!   the tracking resolved);
//! - the back-end entry points: draws and dispatches with their resolved
//!   state, completion callbacks, command-processor transfers, flushes,
//!   CPU synchronisation points and flips;
//! - `END`: totals; a file without it is incomplete.
//!
//! The item payloads are the fields of the resolved structs in declaration
//! order (see the `put_*` functions). Changing a struct changes [`VERSION`].

use crate::render::formats::{render_target_format, ComponentMap};
use crate::render::{
    BlendState, ColorTarget, DepthStencilState, DepthTarget, DispatchPacket, DrawKind, DrawPacket, GdsTransfer, GeometryRegs, GraphicsShaders, IndexBuffer, IndexType,
    IndirectRegisters, RasterState, Rect, ResolvedDispatch, ResolvedDraw, ShaderStage, Viewport,
};
use crate::SurfaceDesc;
use ash::vk;

pub const MAGIC: [u8; 8] = *b"PS5FCAP\0";
pub const VERSION: u32 = 6;
/// The oldest version this build reads (version 1 lacks the draw-index
/// register of indirect draws).
pub const OLDEST_VERSION: u32 = 1;

pub mod tag {
    pub const MAP: u8 = 1;
    pub const UNMAP: u8 = 2;
    pub const PATCH: u8 = 3;
    pub const ZERO: u8 = 4;
    pub const CPU_ACCESS: u8 = 5;
    pub const SYNC_POINT: u8 = 6;
    pub const DRAW: u8 = 7;
    pub const DISPATCH: u8 = 8;
    pub const END_OF_PIPE: u8 = 9;
    pub const AFTER_PIPE: u8 = 10;
    pub const MEMORY_FILLED: u8 = 11;
    pub const COPY_MEMORY: u8 = 12;
    pub const FILL_MEMORY: u8 = 13;
    pub const WRITE_MEMORY: u8 = 14;
    pub const GDS: u8 = 15;
    pub const FLUSH: u8 = 16;
    pub const SUBMISSION_COMPLETE: u8 = 17;
    pub const FLIP: u8 = 18;
    pub const END: u8 = 19;
}

/// One captured record other than guest memory contents.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// `[addr, addr+size)` maps the backing memfd at `backing` (None:
    /// private memory without a backing) with host protection `prot`.
    Map { addr: u64, size: u64, backing: Option<u64>, prot: i32 },
    Unmap { addr: u64, size: u64 },
    /// A CPU access to a watched page resolved by the GPU memory tracking.
    CpuAccess { addr: u64, write: bool },
    SyncPoint,
    Draw(Box<ResolvedDraw>),
    Dispatch(Box<ResolvedDispatch>),
    EndOfPipe,
    AfterPipe,
    MemoryFilled { addr: u64, size: u64, value: u32 },
    /// Command-processor transfers; `gpu`: the back-end executed it (else
    /// the CPU did).
    CopyMemory { dst: u64, src: u64, size: u64, gpu: bool },
    FillMemory { dst: u64, size: u64, value: u32, gpu: bool },
    WriteMemory { dst: u64, data: Vec<u8>, gpu: bool },
    Gds(GdsTransfer),
    Flush,
    SubmissionComplete,
    /// A display buffer was flipped; `gpu`: the presenter took the image
    /// from the back-end (a window), else it read guest memory.
    Flip { desc: SurfaceDesc, addr: u64, gpu: bool },
    End(Totals),
}

/// Totals of a finished capture (the `END` record).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Totals {
    pub flips: u64,
    pub draws: u64,
    pub dispatches: u64,
    pub items: u64,
    pub patches: u64,
    pub patch_bytes: u64,
    pub zero_bytes: u64,
    pub cpu_accesses: u64,
    pub maps: u64,
    /// CPU accesses the fault ring dropped (more between two items than it holds).
    pub dropped_accesses: u64,
}

/// A record as read back: an event or a piece of guest memory.
#[derive(Clone, Debug, PartialEq)]
pub enum Record {
    Event(Event),
    /// Guest bytes at `addr`: `data` is a byte range of the file.
    Patch { addr: u64, data: std::ops::Range<usize> },
    Zero { addr: u64, len: u64 },
}

#[derive(Debug)]
pub struct FormatError(pub String);

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FormatError {}

fn err<T>(msg: impl Into<String>) -> Result<T, FormatError> {
    Err(FormatError(msg.into()))
}

// ---------------------------------------------------------------- writing --

/// Payload builder.
#[derive(Default)]
pub struct Enc(pub Vec<u8>);

impl Enc {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn bool(&mut self, v: bool) {
        self.0.push(v as u8);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }
    fn u32s(&mut self, v: &[u32]) {
        for &x in v {
            self.u32(x);
        }
    }
    fn f32s(&mut self, v: &[f32]) {
        for &x in v {
            self.f32(x);
        }
    }
}

/// The file header: magic, version and the `key=value` lines.
pub fn header_bytes(lines: &[(String, String)]) -> Vec<u8> {
    let text: String = lines.iter().map(|(k, v)| format!("{k}={}\n", v.replace('\n', " "))).collect();
    let mut out = MAGIC.to_vec();
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
    out
}

/// Append one record (tag, length, payload) to `out`.
pub fn record(out: &mut Vec<u8>, tag: u8, payload: &[u8]) {
    out.push(tag);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
}

/// The record of guest bytes at `addr` (`ZERO` when they are all zero).
pub fn memory_record(out: &mut Vec<u8>, addr: u64, data: &[u8]) {
    if data.iter().all(|&b| b == 0) {
        let mut e = Enc::default();
        e.u64(addr);
        e.u64(data.len() as u64);
        record(out, tag::ZERO, &e.0);
    } else {
        out.push(tag::PATCH);
        out.extend_from_slice(&(8 + data.len() as u32).to_le_bytes());
        out.extend_from_slice(&addr.to_le_bytes());
        out.extend_from_slice(data);
    }
}

/// The record of `ev`.
pub fn event_record(out: &mut Vec<u8>, ev: &Event) {
    let mut e = Enc::default();
    let t = match ev {
        Event::Map { addr, size, backing, prot } => {
            e.u64(*addr);
            e.u64(*size);
            e.u64(backing.unwrap_or(u64::MAX));
            e.i32(*prot);
            tag::MAP
        }
        Event::Unmap { addr, size } => {
            e.u64(*addr);
            e.u64(*size);
            tag::UNMAP
        }
        Event::CpuAccess { addr, write } => {
            e.u64(*addr);
            e.bool(*write);
            tag::CPU_ACCESS
        }
        Event::SyncPoint => tag::SYNC_POINT,
        Event::Draw(d) => {
            put_draw(&mut e, d);
            tag::DRAW
        }
        Event::Dispatch(d) => {
            put_dispatch(&mut e, d);
            tag::DISPATCH
        }
        Event::EndOfPipe => tag::END_OF_PIPE,
        Event::AfterPipe => tag::AFTER_PIPE,
        Event::MemoryFilled { addr, size, value } => {
            e.u64(*addr);
            e.u64(*size);
            e.u32(*value);
            tag::MEMORY_FILLED
        }
        Event::CopyMemory { dst, src, size, gpu } => {
            e.u64(*dst);
            e.u64(*src);
            e.u64(*size);
            e.bool(*gpu);
            tag::COPY_MEMORY
        }
        Event::FillMemory { dst, size, value, gpu } => {
            e.u64(*dst);
            e.u64(*size);
            e.u32(*value);
            e.bool(*gpu);
            tag::FILL_MEMORY
        }
        Event::WriteMemory { dst, data, gpu } => {
            e.u64(*dst);
            e.bool(*gpu);
            e.0.extend_from_slice(data);
            tag::WRITE_MEMORY
        }
        Event::Gds(op) => {
            put_gds(&mut e, op);
            tag::GDS
        }
        Event::Flush => tag::FLUSH,
        Event::SubmissionComplete => tag::SUBMISSION_COMPLETE,
        Event::Flip { desc, addr, gpu } => {
            e.u32(desc.width);
            e.u32(desc.height);
            e.u32(desc.pitch_pixels);
            e.u64(desc.pixel_format);
            e.u32(desc.tiling);
            e.u64(*addr);
            e.bool(*gpu);
            tag::FLIP
        }
        Event::End(t) => {
            for v in [t.flips, t.draws, t.dispatches, t.items, t.patches, t.patch_bytes, t.zero_bytes, t.cpu_accesses, t.maps, t.dropped_accesses] {
                e.u64(v);
            }
            tag::END
        }
    };
    record(out, t, &e.0);
}

fn put_stage(e: &mut Enc, s: &ShaderStage) {
    let ShaderStage { address, rsrc1, rsrc2, user_data } = s;
    e.u64(*address);
    e.u32(*rsrc1);
    e.u32(*rsrc2);
    e.u32s(user_data);
}

fn draw_kind_code(k: DrawKind) -> u8 {
    match k {
        DrawKind::Auto => 0,
        DrawKind::Index => 1,
        DrawKind::IndexOffset => 2,
        DrawKind::Indirect => 3,
        DrawKind::IndexIndirect => 4,
        DrawKind::IndirectMulti => 5,
        DrawKind::IndexIndirectMulti => 6,
        DrawKind::IndexMultiInstanced => 7,
    }
}

fn put_packet(e: &mut Enc, p: &DrawPacket) {
    let DrawPacket {
        kind,
        queue,
        packet_address,
        count,
        index_address,
        index_base,
        index_buffer_size,
        index_type,
        instance_count,
        indirect_address,
        indirect_count,
        count_address,
        stride,
        draw_initiator,
        indirect_registers: IndirectRegisters { vertex_offset, instance_offset, first_index, draw_index },
    } = p;
    e.u8(draw_kind_code(*kind));
    e.u64(*queue as u64);
    e.u64(*packet_address);
    e.u32(*count);
    e.u64(*index_address);
    e.u64(*index_base);
    e.u32(*index_buffer_size);
    e.u32(*index_type);
    e.u32(*instance_count);
    e.u64(*indirect_address);
    e.u32(*indirect_count);
    e.u64(*count_address);
    e.u32(*stride);
    e.u32(*draw_initiator);
    e.u32s(&[*vertex_offset, *instance_offset, *first_index, *draw_index]);
}

fn put_blend(e: &mut Enc, b: &BlendState) {
    let BlendState { raw, enable, color_src, color_op, color_dst, alpha_src, alpha_op, alpha_dst, separate_alpha } = b;
    e.u32(*raw);
    e.bool(*enable);
    e.u32s(&[*color_src, *color_op, *color_dst, *alpha_src, *alpha_op, *alpha_dst]);
    e.bool(*separate_alpha);
}

fn put_color(e: &mut Enc, t: &ColorTarget) {
    let ColorTarget {
        slot,
        base,
        format,
        layout,
        number_type,
        order,
        width,
        height,
        pitch,
        swizzle,
        samples,
        mip_level,
        mip_levels,
        base_layer,
        layers,
        slice_size,
        write_mask,
        fast_clear,
        dcc_enabled,
        dcc_address,
        cmask_address,
        fmask_address,
        clear_word0,
        clear_word1,
        dcc_control,
        blend,
        info,
        attrib,
        attrib2,
        attrib3,
        view,
    } = t;
    e.u32(*slot);
    e.u64(*base);
    e.u32(format.guest as u32);
    e.i32(format.host.as_raw());
    e.u32(format.bytes_per_element);
    e.u8(format.export_mapping.0);
    e.u32s(&[*layout, *number_type, *order, *width, *height, *pitch, *swizzle, *samples, *mip_level, *mip_levels, *base_layer, *layers]);
    e.u64(*slice_size);
    e.u32(*write_mask);
    e.bool(*fast_clear);
    e.bool(*dcc_enabled);
    e.u64(*dcc_address);
    e.u64(*cmask_address);
    e.u64(*fmask_address);
    e.u32s(&[*clear_word0, *clear_word1, *dcc_control]);
    put_blend(e, blend);
    e.u32s(&[*info, *attrib, *attrib2, *attrib3, *view]);
}

fn put_depth(e: &mut Enc, t: &DepthTarget) {
    let DepthTarget {
        z_format,
        host_format,
        bytes_per_element,
        has_stencil,
        width,
        height,
        pitch,
        swizzle,
        samples,
        base_layer,
        layers,
        z_read_base,
        z_write_base,
        stencil_read_base,
        stencil_write_base,
        htile_base,
        htile_enabled,
        depth_slice_size,
        stencil_slice_size,
        z_info,
        stencil_info,
        depth_view,
    } = t;
    e.u32(*z_format);
    e.i32(host_format.as_raw());
    e.u32(*bytes_per_element);
    e.bool(*has_stencil);
    e.u32s(&[*width, *height, *pitch, *swizzle, *samples, *base_layer, *layers]);
    for v in [*z_read_base, *z_write_base, *stencil_read_base, *stencil_write_base, *htile_base] {
        e.u64(v);
    }
    e.bool(*htile_enabled);
    e.u64(*depth_slice_size);
    e.u64(*stencil_slice_size);
    e.u32s(&[*z_info, *stencil_info, *depth_view]);
}

fn put_depth_state(e: &mut Enc, s: &DepthStencilState) {
    let DepthStencilState {
        depth_test,
        depth_write,
        depth_compare,
        depth_bounds_enable,
        depth_bounds,
        stencil_test,
        stencil_compare,
        stencil_compare_back,
        backface,
        stencil_control,
        stencil_ref_mask,
        stencil_ref_mask_back,
        depth_clear_enable,
        stencil_clear_enable,
        depth_clear_value,
        stencil_clear_value,
        render_control,
        render_override,
        depth_control,
        depth_write_disable,
        stencil_write_disable,
    } = s;
    e.bool(*depth_test);
    e.bool(*depth_write);
    e.u32(*depth_compare);
    e.bool(*depth_bounds_enable);
    e.f32s(depth_bounds);
    e.bool(*stencil_test);
    e.u32(*stencil_compare);
    e.u32(*stencil_compare_back);
    e.bool(*backface);
    e.u32s(&[*stencil_control, *stencil_ref_mask, *stencil_ref_mask_back]);
    e.bool(*depth_clear_enable);
    e.bool(*stencil_clear_enable);
    e.f32(*depth_clear_value);
    e.u8(*stencil_clear_value);
    e.u32s(&[*render_control, *render_override, *depth_control]);
    e.bool(*depth_write_disable);
    e.bool(*stencil_write_disable);
}

fn put_viewport(e: &mut Enc, v: &Viewport) {
    let Viewport { x, y, width, height, min_depth, max_depth, scale, offset, z_min, z_max } = v;
    e.f32s(&[*x, *y, *width, *height, *min_depth, *max_depth]);
    e.f32s(scale);
    e.f32s(offset);
    e.f32(*z_min);
    e.f32(*z_max);
}

fn put_rect(e: &mut Enc, r: &Rect) {
    let Rect { x0, y0, x1, y1 } = r;
    for v in [*x0, *y0, *x1, *y1] {
        e.i32(v);
    }
}

fn put_raster(e: &mut Enc, r: &RasterState) {
    let RasterState {
        cull_front,
        cull_back,
        front_face_cw,
        polygon_mode,
        provoking_vertex_last,
        mode_cntl,
        clip_cntl,
        vte_cntl,
        vs_out_cntl,
        dx_clip_space,
        poly_offset_db_fmt,
        poly_offset,
        // Written at the end of the draw (version 6).
        user_clip_planes: _,
    } = r;
    e.bool(*cull_front);
    e.bool(*cull_back);
    e.bool(*front_face_cw);
    e.u32(*polygon_mode);
    e.bool(*provoking_vertex_last);
    e.u32s(&[*mode_cntl, *clip_cntl, *vte_cntl, *vs_out_cntl]);
    e.bool(*dx_clip_space);
    e.u32(*poly_offset_db_fmt);
    e.u32s(poly_offset);
}

fn put_shaders(e: &mut Enc, s: &GraphicsShaders) {
    let GraphicsShaders { ps, gs, es, vs, hs, ls, stages, geometry } = s;
    for st in [ps, gs, es, vs, hs, ls] {
        put_stage(e, st);
    }
    e.u32(*stages);
    let GeometryRegs { ge_cntl, max_output_per_subgroup, max_vert_out, esgs_ring_item_size, out_prim_type, user_data_address } = geometry;
    e.u32s(&[*ge_cntl, *max_output_per_subgroup, *max_vert_out, *esgs_ring_item_size, *out_prim_type]);
    e.u64(*user_data_address);
}

pub fn put_draw(e: &mut Enc, d: &ResolvedDraw) {
    let ResolvedDraw {
        packet,
        index,
        primitive_type,
        topology,
        primitive_restart,
        restart_index,
        index_offset,
        color_targets,
        resolve_targets,
        depth,
        depth_state,
        color_mode,
        rop3,
        target_mask,
        shader_mask,
        blend_constants,
        viewports,
        scissor,
        scissors,
        raster,
        shaders,
        ps_input_cntl,
        ps_input_ena,
        ps_input_addr,
        ps_in_control,
        shader_z_format,
        shader_col_format,
        shader_pos_format,
        aa_config,
        db_shader_control,
        alpha_to_mask,
        front_face_all_bits,
        msaa_enable,
        ps_iter_sample,
        aa_mask,
    } = d;
    put_packet(e, packet);
    match index {
        Some(IndexBuffer { address, kind, count }) => {
            e.u8(1);
            e.u64(*address);
            e.u8(match kind {
                IndexType::U16 => 0,
                IndexType::U32 => 1,
                IndexType::U8 => 2,
            });
            e.u32(*count);
        }
        None => e.u8(0),
    }
    e.u32(*primitive_type);
    e.i32(topology.map_or(-1, |t| t.as_raw()));
    e.bool(*primitive_restart);
    e.u32(*restart_index);
    e.u32(*index_offset);
    e.u32(color_targets.len() as u32);
    for t in color_targets {
        put_color(e, t);
    }
    match resolve_targets {
        Some((a, b)) => {
            e.u8(1);
            put_color(e, a);
            put_color(e, b);
        }
        None => e.u8(0),
    }
    match depth {
        Some(z) => {
            e.u8(1);
            put_depth(e, z);
        }
        None => e.u8(0),
    }
    put_depth_state(e, depth_state);
    e.u32s(&[*color_mode, *rop3, *target_mask, *shader_mask]);
    e.f32s(blend_constants);
    for v in viewports {
        put_viewport(e, v);
    }
    put_rect(e, scissor);
    for r in scissors {
        put_rect(e, r);
    }
    put_raster(e, raster);
    put_shaders(e, shaders);
    e.u32s(ps_input_cntl);
    e.u32s(&[*ps_input_ena, *ps_input_addr, *ps_in_control, *shader_z_format, *shader_col_format, *shader_pos_format, *aa_config, *db_shader_control, *alpha_to_mask]);
    e.u32(*front_face_all_bits as u32);
    e.u32(*msaa_enable as u32);
    e.u32(*ps_iter_sample as u32);
    e.u32s(aa_mask);
    e.u32s(&raster.user_clip_planes);
}

pub fn put_dispatch(e: &mut Enc, d: &ResolvedDispatch) {
    let ResolvedDispatch { packet: DispatchPacket { queue, packet_address, groups, indirect_address, dispatch_initiator, unsynced: _ }, shader, rsrc3, threads, start } = d;
    e.u64(*queue as u64);
    e.u64(*packet_address);
    e.u32s(groups);
    e.u64(*indirect_address);
    e.u32(*dispatch_initiator);
    put_stage(e, shader);
    e.u32(*rsrc3);
    e.u32s(threads);
    e.u32s(start);
}

fn put_gds(e: &mut Enc, op: &GdsTransfer) {
    match op {
        GdsTransfer::Write { offset, data } => {
            e.u8(0);
            e.u64(*offset);
            e.u32(data.len() as u32);
            e.0.extend_from_slice(data);
        }
        GdsTransfer::Fill { offset, size, value } => {
            e.u8(1);
            e.u64(*offset);
            e.u64(*size);
            e.u32(*value);
        }
        GdsTransfer::FromMemory { offset, src, size } => {
            e.u8(2);
            e.u64(*offset);
            e.u64(*src);
            e.u64(*size);
        }
        GdsTransfer::ToMemory { offset, dst, size } => {
            e.u8(3);
            e.u64(*offset);
            e.u64(*dst);
            e.u64(*size);
        }
    }
}

// ---------------------------------------------------------------- reading --

/// Payload reader.
pub struct Dec<'a> {
    b: &'a [u8],
    p: usize,
    /// Format version of the file the payload comes from.
    version: u32,
}

impl<'a> Dec<'a> {
    pub fn new(b: &'a [u8]) -> Dec<'a> {
        Dec { b, p: 0, version: VERSION }
    }
    /// A reader of a payload written by format `version`.
    pub fn with_version(b: &'a [u8], version: u32) -> Dec<'a> {
        Dec { b, p: 0, version }
    }

    /// Format version of the file the payload comes from.
    pub fn version(&self) -> u32 {
        self.version
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], FormatError> {
        if self.b.len() - self.p < n {
            return err(format!("record truncated at byte {} (needs {n} more)", self.p));
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn rest(&mut self) -> &'a [u8] {
        let s = &self.b[self.p..];
        self.p = self.b.len();
        s
    }
    fn u8(&mut self) -> Result<u8, FormatError> {
        Ok(self.take(1)?[0])
    }
    fn bool(&mut self) -> Result<bool, FormatError> {
        Ok(self.u8()? != 0)
    }
    fn u32(&mut self) -> Result<u32, FormatError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, FormatError> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, FormatError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, FormatError> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn u32s<const N: usize>(&mut self) -> Result<[u32; N], FormatError> {
        let mut v = [0u32; N];
        for x in &mut v {
            *x = self.u32()?;
        }
        Ok(v)
    }
    fn f32s<const N: usize>(&mut self) -> Result<[f32; N], FormatError> {
        let mut v = [0f32; N];
        for x in &mut v {
            *x = self.f32()?;
        }
        Ok(v)
    }
    fn done(&self, what: &str) -> Result<(), FormatError> {
        if self.p != self.b.len() {
            return err(format!("{what}: {} bytes left over", self.b.len() - self.p));
        }
        Ok(())
    }
}

fn get_stage(d: &mut Dec) -> Result<ShaderStage, FormatError> {
    Ok(ShaderStage { address: d.u64()?, rsrc1: d.u32()?, rsrc2: d.u32()?, user_data: d.u32s()? })
}

fn get_packet(d: &mut Dec) -> Result<DrawPacket, FormatError> {
    let kind = match d.u8()? {
        0 => DrawKind::Auto,
        1 => DrawKind::Index,
        2 => DrawKind::IndexOffset,
        3 => DrawKind::Indirect,
        4 => DrawKind::IndexIndirect,
        5 => DrawKind::IndirectMulti,
        6 => DrawKind::IndexIndirectMulti,
        7 => DrawKind::IndexMultiInstanced,
        k => return err(format!("draw kind {k}")),
    };
    Ok(DrawPacket {
        kind,
        queue: d.u64()? as usize,
        packet_address: d.u64()?,
        count: d.u32()?,
        index_address: d.u64()?,
        index_base: d.u64()?,
        index_buffer_size: d.u32()?,
        index_type: d.u32()?,
        instance_count: d.u32()?,
        indirect_address: d.u64()?,
        indirect_count: d.u32()?,
        count_address: d.u64()?,
        stride: d.u32()?,
        draw_initiator: d.u32()?,
        indirect_registers: {
            let [vertex_offset, instance_offset, first_index] = d.u32s()?;
            // Version 1 did not record the draw-index register.
            let draw_index = if d.version >= 2 { d.u32()? } else { crate::render::SH_NOP };
            IndirectRegisters { vertex_offset, instance_offset, first_index, draw_index }
        },
    })
}

fn get_blend(d: &mut Dec) -> Result<BlendState, FormatError> {
    let raw = d.u32()?;
    let enable = d.bool()?;
    let [color_src, color_op, color_dst, alpha_src, alpha_op, alpha_dst] = d.u32s()?;
    Ok(BlendState { raw, enable, color_src, color_op, color_dst, alpha_src, alpha_op, alpha_dst, separate_alpha: d.bool()? })
}

fn get_color(d: &mut Dec) -> Result<ColorTarget, FormatError> {
    let slot = d.u32()?;
    let base = d.u64()?;
    let guest = d.u32()?;
    let host = vk::Format::from_raw(d.i32()?);
    let bytes_per_element = d.u32()?;
    let export_mapping = ComponentMap(d.u8()?);
    let [layout, number_type, order, width, height, pitch, swizzle, samples, mip_level, mip_levels, base_layer, layers] = d.u32s()?;
    // The guest format is the one the CB_COLOR_INFO fields name.
    let Some(mut format) = render_target_format(layout, number_type, order).filter(|f| f.guest as u32 == guest) else {
        return err(format!("colour target format {guest} does not match layout {layout} / number type {number_type} / order {order}"));
    };
    format.host = host;
    format.bytes_per_element = bytes_per_element;
    format.export_mapping = export_mapping;
    Ok(ColorTarget {
        slot,
        base,
        format,
        layout,
        number_type,
        order,
        width,
        height,
        pitch,
        swizzle,
        samples,
        mip_level,
        mip_levels,
        base_layer,
        layers,
        slice_size: d.u64()?,
        write_mask: d.u32()?,
        fast_clear: d.bool()?,
        dcc_enabled: d.bool()?,
        dcc_address: d.u64()?,
        cmask_address: d.u64()?,
        fmask_address: d.u64()?,
        clear_word0: d.u32()?,
        clear_word1: d.u32()?,
        dcc_control: d.u32()?,
        blend: get_blend(d)?,
        info: d.u32()?,
        attrib: d.u32()?,
        attrib2: d.u32()?,
        attrib3: d.u32()?,
        view: d.u32()?,
    })
}

fn get_depth(d: &mut Dec) -> Result<DepthTarget, FormatError> {
    let z_format = d.u32()?;
    let host_format = vk::Format::from_raw(d.i32()?);
    let bytes_per_element = d.u32()?;
    let has_stencil = d.bool()?;
    let [width, height, pitch, swizzle, samples, base_layer, layers] = d.u32s()?;
    Ok(DepthTarget {
        z_format,
        host_format,
        bytes_per_element,
        has_stencil,
        width,
        height,
        pitch,
        swizzle,
        samples,
        base_layer,
        layers,
        z_read_base: d.u64()?,
        z_write_base: d.u64()?,
        stencil_read_base: d.u64()?,
        stencil_write_base: d.u64()?,
        htile_base: d.u64()?,
        htile_enabled: d.bool()?,
        depth_slice_size: d.u64()?,
        stencil_slice_size: d.u64()?,
        z_info: d.u32()?,
        stencil_info: d.u32()?,
        depth_view: d.u32()?,
    })
}

fn get_depth_state(d: &mut Dec) -> Result<DepthStencilState, FormatError> {
    Ok(DepthStencilState {
        depth_test: d.bool()?,
        depth_write: d.bool()?,
        depth_compare: d.u32()?,
        depth_bounds_enable: d.bool()?,
        depth_bounds: d.f32s()?,
        stencil_test: d.bool()?,
        stencil_compare: d.u32()?,
        stencil_compare_back: d.u32()?,
        backface: d.bool()?,
        stencil_control: d.u32()?,
        stencil_ref_mask: d.u32()?,
        stencil_ref_mask_back: d.u32()?,
        depth_clear_enable: d.bool()?,
        stencil_clear_enable: d.bool()?,
        depth_clear_value: d.f32()?,
        stencil_clear_value: d.u8()?,
        render_control: d.u32()?,
        render_override: d.u32()?,
        depth_control: d.u32()?,
        depth_write_disable: d.bool()?,
        stencil_write_disable: d.bool()?,
    })
}

fn get_viewport(d: &mut Dec) -> Result<Viewport, FormatError> {
    let [x, y, width, height, min_depth, max_depth] = d.f32s()?;
    Ok(Viewport { x, y, width, height, min_depth, max_depth, scale: d.f32s()?, offset: d.f32s()?, z_min: d.f32()?, z_max: d.f32()? })
}

fn get_rect(d: &mut Dec) -> Result<Rect, FormatError> {
    Ok(Rect { x0: d.i32()?, y0: d.i32()?, x1: d.i32()?, y1: d.i32()? })
}

fn get_raster(d: &mut Dec) -> Result<RasterState, FormatError> {
    let cull_front = d.bool()?;
    let cull_back = d.bool()?;
    let front_face_cw = d.bool()?;
    let polygon_mode = d.u32()?;
    let provoking_vertex_last = d.bool()?;
    let [mode_cntl, clip_cntl, vte_cntl, vs_out_cntl] = d.u32s()?;
    Ok(RasterState {
        cull_front,
        cull_back,
        front_face_cw,
        polygon_mode,
        provoking_vertex_last,
        mode_cntl,
        clip_cntl,
        vte_cntl,
        vs_out_cntl,
        dx_clip_space: d.bool()?,
        poly_offset_db_fmt: d.u32()?,
        poly_offset: d.u32s()?,
        user_clip_planes: [0; 24],
    })
}

fn get_shaders(d: &mut Dec) -> Result<GraphicsShaders, FormatError> {
    let ps = get_stage(d)?;
    let gs = get_stage(d)?;
    let es = get_stage(d)?;
    let vs = get_stage(d)?;
    let hs = get_stage(d)?;
    let ls = get_stage(d)?;
    let stages = d.u32()?;
    let [ge_cntl, max_output_per_subgroup, max_vert_out, esgs_ring_item_size, out_prim_type] = d.u32s()?;
    let geometry = GeometryRegs { ge_cntl, max_output_per_subgroup, max_vert_out, esgs_ring_item_size, out_prim_type, user_data_address: d.u64()? };
    Ok(GraphicsShaders { ps, gs, es, vs, hs, ls, stages, geometry })
}

pub fn get_draw(d: &mut Dec) -> Result<ResolvedDraw, FormatError> {
    let packet = get_packet(d)?;
    let index = match d.u8()? {
        0 => None,
        _ => {
            let address = d.u64()?;
            let kind = match d.u8()? {
                0 => IndexType::U16,
                1 => IndexType::U32,
                2 => IndexType::U8,
                k => return err(format!("index type {k}")),
            };
            Some(IndexBuffer { address, kind, count: d.u32()? })
        }
    };
    let primitive_type = d.u32()?;
    let topology = match d.i32()? {
        -1 => None,
        t => Some(vk::PrimitiveTopology::from_raw(t)),
    };
    let primitive_restart = d.bool()?;
    let restart_index = d.u32()?;
    let index_offset = d.u32()?;
    let n = d.u32()?;
    if n > 8 {
        return err(format!("{n} colour targets"));
    }
    let mut color_targets = smallvec::SmallVec::new();
    for _ in 0..n {
        color_targets.push(get_color(d)?);
    }
    let resolve_targets = match d.u8()? {
        0 => None,
        _ => Some((get_color(d)?, get_color(d)?)),
    };
    let depth = match d.u8()? {
        0 => None,
        _ => Some(get_depth(d)?),
    };
    let depth_state = get_depth_state(d)?;
    let [color_mode, rop3, target_mask, shader_mask] = d.u32s()?;
    let blend_constants = d.f32s()?;
    let mut viewports = [Viewport::default(); 16];
    for v in &mut viewports {
        *v = get_viewport(d)?;
    }
    let scissor = get_rect(d)?;
    let mut scissors = [Rect::default(); 16];
    for r in &mut scissors {
        *r = get_rect(d)?;
    }
    let raster = get_raster(d)?;
    let shaders = get_shaders(d)?;
    let ps_input_cntl = d.u32s()?;
    let [ps_input_ena, ps_input_addr, ps_in_control, shader_z_format, shader_col_format, shader_pos_format, aa_config, db_shader_control, alpha_to_mask] = d.u32s()?;
    // Version 3: SPI_BARYC_CNTL.FRONT_FACE_ALL_BITS of the draw.
    let front_face_all_bits = d.version() >= 3 && d.u32()? != 0;
    // Version 4: PA_SC_MODE_CNTL_0.MSAA_ENABLE. Older captures drew every
    // multisampled target at the standard sample locations, as with the bit.
    let msaa_enable = d.version() < 4 || d.u32()? != 0;
    // Version 5: PA_SC_MODE_CNTL_1.PS_ITER_SAMPLE and PA_SC_AA_MASK (older
    // captures: once per pixel, every sample).
    let (ps_iter_sample, aa_mask) = if d.version() >= 5 { (d.u32()? != 0, d.u32s()?) } else { (false, [u32::MAX; 2]) };
    // Version 6: PA_CL_UCP_n (older captures: no user clip planes).
    let mut raster = raster;
    if d.version() >= 6 {
        raster.user_clip_planes = d.u32s()?;
    }
    Ok(ResolvedDraw {
        packet,
        index,
        primitive_type,
        topology,
        primitive_restart,
        restart_index,
        index_offset,
        color_targets,
        resolve_targets,
        depth,
        depth_state,
        color_mode,
        rop3,
        target_mask,
        shader_mask,
        blend_constants,
        viewports,
        scissor,
        scissors,
        raster,
        shaders,
        ps_input_cntl,
        ps_input_ena,
        ps_input_addr,
        ps_in_control,
        shader_z_format,
        shader_col_format,
        shader_pos_format,
        aa_config,
        db_shader_control,
        alpha_to_mask,
        front_face_all_bits,
        msaa_enable,
        ps_iter_sample,
        aa_mask,
    })
}

pub fn get_dispatch(d: &mut Dec) -> Result<ResolvedDispatch, FormatError> {
    let packet = DispatchPacket { queue: d.u64()? as usize, packet_address: d.u64()?, groups: d.u32s()?, indirect_address: d.u64()?, dispatch_initiator: d.u32()?, unsynced: false };
    Ok(ResolvedDispatch { packet, shader: get_stage(d)?, rsrc3: d.u32()?, threads: d.u32s()?, start: d.u32s()? })
}

fn get_gds(d: &mut Dec) -> Result<GdsTransfer, FormatError> {
    Ok(match d.u8()? {
        0 => {
            let offset = d.u64()?;
            let n = d.u32()? as usize;
            GdsTransfer::Write { offset, data: d.take(n)?.to_vec() }
        }
        1 => GdsTransfer::Fill { offset: d.u64()?, size: d.u64()?, value: d.u32()? },
        2 => GdsTransfer::FromMemory { offset: d.u64()?, src: d.u64()?, size: d.u64()? },
        3 => GdsTransfer::ToMemory { offset: d.u64()?, dst: d.u64()?, size: d.u64()? },
        k => return err(format!("GDS transfer kind {k}")),
    })
}

/// Decode the record `tag` whose payload is `file[at]`. None for a tag
/// this reader does not know.
pub fn decode(t: u8, file: &[u8], at: std::ops::Range<usize>, version: u32) -> Result<Option<Record>, FormatError> {
    let payload = &file[at.clone()];
    let mut d = Dec::with_version(payload, version);
    let ev = match t {
        tag::PATCH => {
            let addr = d.u64()?;
            return Ok(Some(Record::Patch { addr, data: at.start + 8..at.end }));
        }
        tag::ZERO => {
            let r = Record::Zero { addr: d.u64()?, len: d.u64()? };
            d.done("zero")?;
            return Ok(Some(r));
        }
        tag::MAP => {
            let addr = d.u64()?;
            let size = d.u64()?;
            let backing = Some(d.u64()?).filter(|&b| b != u64::MAX);
            Event::Map { addr, size, backing, prot: d.i32()? }
        }
        tag::UNMAP => Event::Unmap { addr: d.u64()?, size: d.u64()? },
        tag::CPU_ACCESS => Event::CpuAccess { addr: d.u64()?, write: d.bool()? },
        tag::SYNC_POINT => Event::SyncPoint,
        tag::DRAW => Event::Draw(Box::new(get_draw(&mut d)?)),
        tag::DISPATCH => Event::Dispatch(Box::new(get_dispatch(&mut d)?)),
        tag::END_OF_PIPE => Event::EndOfPipe,
        tag::AFTER_PIPE => Event::AfterPipe,
        tag::MEMORY_FILLED => Event::MemoryFilled { addr: d.u64()?, size: d.u64()?, value: d.u32()? },
        tag::COPY_MEMORY => Event::CopyMemory { dst: d.u64()?, src: d.u64()?, size: d.u64()?, gpu: d.bool()? },
        tag::FILL_MEMORY => Event::FillMemory { dst: d.u64()?, size: d.u64()?, value: d.u32()?, gpu: d.bool()? },
        tag::WRITE_MEMORY => {
            let dst = d.u64()?;
            let gpu = d.bool()?;
            Event::WriteMemory { dst, gpu, data: d.rest().to_vec() }
        }
        tag::GDS => Event::Gds(get_gds(&mut d)?),
        tag::FLUSH => Event::Flush,
        tag::SUBMISSION_COMPLETE => Event::SubmissionComplete,
        tag::FLIP => Event::Flip {
            desc: SurfaceDesc { width: d.u32()?, height: d.u32()?, pitch_pixels: d.u32()?, pixel_format: d.u64()?, tiling: d.u32()? },
            addr: d.u64()?,
            gpu: d.bool()?,
        },
        tag::END => {
            let mut v = [0u64; 10];
            for x in &mut v {
                *x = d.u64()?;
            }
            let [flips, draws, dispatches, items, patches, patch_bytes, zero_bytes, cpu_accesses, maps, dropped_accesses] = v;
            Event::End(Totals { flips, draws, dispatches, items, patches, patch_bytes, zero_bytes, cpu_accesses, maps, dropped_accesses })
        }
        _ => return Ok(None),
    };
    d.done(&format!("record {t}"))?;
    Ok(Some(Record::Event(ev)))
}

/// A parsed capture: header lines and records. The file is mapped (or
/// held) whole; patch records point into it.
pub struct Capture {
    storage: Storage,
    pub header: Vec<(String, String)>,
    pub records: Vec<Record>,
    /// Records with tags this reader skipped.
    pub unknown: u64,
    /// The `END` totals (None: the capture is incomplete).
    pub totals: Option<Totals>,
}

enum Storage {
    Owned(Vec<u8>),
    #[cfg_attr(windows, allow(dead_code))]
    Mapped(*const u8, usize),
}

// SAFETY: the mapping is private and read-only for the life of the value.
unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

impl Drop for Storage {
    fn drop(&mut self) {
        if let Storage::Mapped(p, n) = *self {
            // SAFETY: our own mapping of `n` bytes.
            #[cfg(unix)]
            unsafe {
                libc::munmap(p as *mut libc::c_void, n)
            };
            #[cfg(windows)]
            let _ = (p, n);
        }
    }
}

impl Capture {
    /// Map and parse the file at `path`.
    #[cfg(windows)]
    pub fn open(path: &std::path::Path) -> Result<Capture, FormatError> {
        let bytes = std::fs::read(path).map_err(|e| FormatError(format!("{}: {e}", path.display())))?;
        if bytes.len() < 16 {
            return err(format!("{}: too short for a capture", path.display()));
        }
        Capture::parse(Storage::Owned(bytes))
    }

    /// Map and parse the file at `path`.
    #[cfg(unix)]
    pub fn open(path: &std::path::Path) -> Result<Capture, FormatError> {
        use std::os::fd::AsRawFd;
        let f = std::fs::File::open(path).map_err(|e| FormatError(format!("{}: {e}", path.display())))?;
        let n = f.metadata().map_err(|e| FormatError(format!("{}: {e}", path.display())))?.len() as usize;
        if n < 16 {
            return err(format!("{}: too short for a capture", path.display()));
        }
        // SAFETY: read-only private mapping of a file we opened.
        let p = unsafe { libc::mmap(std::ptr::null_mut(), n, libc::PROT_READ, libc::MAP_PRIVATE, f.as_raw_fd(), 0) };
        if p == libc::MAP_FAILED {
            return err(format!("{}: mmap failed", path.display()));
        }
        Capture::parse(Storage::Mapped(p as *const u8, n))
    }

    /// Parse a capture held in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Capture, FormatError> {
        Capture::parse(Storage::Owned(bytes))
    }

    pub fn bytes(&self) -> &[u8] {
        match &self.storage {
            Storage::Owned(v) => v,
            // SAFETY: the mapping lives as long as `self`.
            Storage::Mapped(p, n) => unsafe { std::slice::from_raw_parts(*p, *n) },
        }
    }

    fn parse(storage: Storage) -> Result<Capture, FormatError> {
        let mut c = Capture { storage, header: Vec::new(), records: Vec::new(), unknown: 0, totals: None };
        let b = c.bytes();
        if b.len() < 16 || b[..8] != MAGIC {
            return err("not a rps5e frame capture (bad magic)");
        }
        let version = u32::from_le_bytes(b[8..12].try_into().unwrap());
        if !(OLDEST_VERSION..=VERSION).contains(&version) {
            return err(format!("capture format version {version}, this build reads versions {OLDEST_VERSION} to {VERSION}"));
        }
        let hlen = u32::from_le_bytes(b[12..16].try_into().unwrap()) as usize;
        if b.len() < 16 + hlen {
            return err("header truncated");
        }
        let text = std::str::from_utf8(&b[16..16 + hlen]).map_err(|_| FormatError("header is not UTF-8".into()))?;
        let header: Vec<(String, String)> = text.lines().filter_map(|l| l.split_once('=')).map(|(k, v)| (k.to_string(), v.to_string())).collect();
        let mut records = Vec::new();
        let mut unknown = 0;
        let mut totals = None;
        let mut p = 16 + hlen;
        while p < b.len() {
            if b.len() - p < 5 {
                return err(format!("record header truncated at byte {p}"));
            }
            let t = b[p];
            let len = u32::from_le_bytes(b[p + 1..p + 5].try_into().unwrap()) as usize;
            let at = p + 5..p + 5 + len;
            if at.end > b.len() {
                // An interrupted capture: keep what is complete.
                log::warn!("ps5gpu: capture truncated at byte {p} (record {t} of {len} bytes)");
                break;
            }
            match decode(t, b, at.clone(), version).map_err(|e| FormatError(format!("record {t} at byte {p}: {e}")))? {
                Some(Record::Event(Event::End(t))) => totals = Some(t),
                Some(r) => records.push(r),
                None => unknown += 1,
            }
            p = at.end;
        }
        c.header = header;
        c.records = records;
        c.unknown = unknown;
        c.totals = totals;
        Ok(c)
    }

    /// The header value of `key`.
    pub fn header_value(&self, key: &str) -> Option<&str> {
        self.header.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    /// Guest bytes of a patch record.
    pub fn data(&self, r: &std::ops::Range<usize>) -> &[u8] {
        &self.bytes()[r.clone()]
    }
}
