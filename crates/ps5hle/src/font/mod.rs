// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceFont / libSceFontFt — font library, renderer and glyph rendering.
//!
//! Library, renderer, font and glyph handles are guest memory blocks
//! carrying the console's magic numbers (so guest-side sanity checks see
//! well-formed objects); their real state lives on the host keyed by the
//! guest address. Faces are parsed with a pure OpenType/TrueType reader and
//! rasterised with an anti-aliasing scanline rasteriser (8-bit coverage,
//! unhinted, horizontal shear for the slant effect).
//!
//! System font sets resolve to the console fonts when a directory named by
//! `RPS5E_SYSTEM_FONTS` provides them, otherwise to substitute host fonts
//! of the same style (see face.rs).

mod face;
mod objects;
mod style;
mod text;

use objects::write_glyph_metric_variants;

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, MutexGuard, RwLock};

use crate::context::{CpuContext, CpuRegister};
use crate::registry::{HleCallback, HleRegistry, HleResult};
use face::{q64, FaceData};

const LIB: &str = "libSceFont";
const LIB_FT: &str = "libSceFontFt";

const fn err(code: u32) -> HleResult {
    code as i32 as i64
}

pub(crate) const ERROR_INVALID_PARAMETER: HleResult = err(0x8046_0002);
pub(crate) const ERROR_INVALID_MEMORY: HleResult = err(0x8046_0003);
pub(crate) const ERROR_INVALID_LIBRARY: HleResult = err(0x8046_0004);
pub(crate) const ERROR_INVALID_FONT_HANDLE: HleResult = err(0x8046_0005);
pub(crate) const ERROR_INVALID_GLYPH: HleResult = err(0x8046_0006);
pub(crate) const ERROR_INVALID_RENDERER: HleResult = err(0x8046_0007);
pub(crate) const ERROR_INVALID_TEXT_SOURCE: HleResult = err(0x8046_0008);
pub(crate) const ERROR_INVALID_STRING: HleResult = err(0x8046_0009);
pub(crate) const ERROR_INVALID_WRITING: HleResult = err(0x8046_000A);
pub(crate) const ERROR_ALLOCATION_FAILED: HleResult = err(0x8046_0010);
const ERROR_FS_OPEN_FAILED: HleResult = err(0x8046_0011);
const ERROR_NO_SUPPORT_FORMAT: HleResult = err(0x8046_0019);
const ERROR_NO_SUPPORT_FUNCTION: HleResult = err(0x8046_0020);
const ERROR_ALREADY_SPECIFIED: HleResult = err(0x8046_0021);
const ERROR_ALREADY_ATTACHED: HleResult = err(0x8046_0022);
const ERROR_NOT_ATTACHED_CACHE_BUFFER: HleResult = err(0x8046_0025);
const ERROR_NO_SUPPORT_FONTSET: HleResult = err(0x8046_0031);
const ERROR_FONT_OPEN_FAILED: HleResult = err(0x8046_0036);
const ERROR_NO_SUPPORT_CODE: HleResult = err(0x8046_0041);
const ERROR_NO_SUPPORT_GLYPH: HleResult = err(0x8046_0042);
const ERROR_NO_SUPPORT_SURFACE: HleResult = err(0x8046_0050);
const ERROR_UNSET_PARAMETER: HleResult = err(0x8046_0058);
const ERROR_ALREADY_BOUND_RENDERER: HleResult = err(0x8046_0060);
const ERROR_NOT_BOUND_RENDERER: HleResult = err(0x8046_0061);

pub(crate) const MEMORY_MAGIC: u16 = 0x0F00;
const LIBRARY_MAGIC: u16 = 0x0F01;
pub(crate) const HANDLE_MAGIC: u16 = 0x0F02;
const GLYPH_MAGIC: u16 = 0x0F03;
const RENDERER_MAGIC: u16 = 0x0F07;
const STYLE_FRAME_MAGIC: u16 = 0x0F09;
const GENERATE_GLYPH_DETAIL_ID: u16 = 0x0FD3;

const STYLE_FRAME_FLAG_SCALE: u8 = 0x01;
const STYLE_FRAME_FLAG_SLANT: u8 = 0x02;
const STYLE_FRAME_FLAG_WEIGHT: u8 = 0x04;
const POINTS_PER_INCH: f32 = 72.0;
const DEFAULT_DPI: u32 = 72;

/// Largest font file accepted from guest memory.
const MAX_FONT_BYTES: u32 = 64 << 20;

// Guest object layouts.
const HANDLE_SIZE: u64 = 0x100;
const HANDLE_LIBRARY: u64 = 0x28;
const HANDLE_RENDERER: u64 = 0x30;
const LIBRARY_SIZE: u64 = 0x100;
const RENDERER_SIZE: u64 = 0x168;
const GLYPH_SIZE: u64 = 0x100;
const GLYPH_METRICS: u64 = 0x20;
const DRIVER_TABLE_SIZE: u64 = 0x120;
const RENDERER_SELECTION_SIZE: u64 = 0x20;

// ---------------------------------------------------------------------------
// Guest memory pool for library-owned objects
// ---------------------------------------------------------------------------

static POOL: Mutex<Option<HashMap<u64, Vec<u64>>>> = Mutex::new(None);
static POOL_SIZES: Mutex<Option<HashMap<u64, u64>>> = Mutex::new(None);

fn size_class(size: u64) -> u64 {
    size.max(0x100).next_power_of_two()
}

/// Allocate a zeroed guest block, reusing freed blocks of the same class.
pub(crate) fn galloc(ctx: &CpuContext, size: u64) -> Option<u64> {
    let class = size_class(size);
    let reused = POOL.lock().unwrap().get_or_insert_with(HashMap::new).get_mut(&class).and_then(|v| v.pop());
    let addr = match reused {
        Some(a) => {
            if !ctx.write_bytes(a, &vec![0u8; class as usize]) {
                return None;
            }
            a
        }
        None => crate::pthread::alloc_guest_block(ctx, class)?,
    };
    POOL_SIZES.lock().unwrap().get_or_insert_with(HashMap::new).insert(addr, class);
    Some(addr)
}

pub(crate) fn gfree(addr: u64) {
    let class = POOL_SIZES.lock().unwrap().get_or_insert_with(HashMap::new).remove(&addr);
    if let Some(class) = class {
        POOL.lock().unwrap().get_or_insert_with(HashMap::new).entry(class).or_default().push(addr);
    }
}

// ---------------------------------------------------------------------------
// Guest calls (text parsers) and floating-point arguments
// ---------------------------------------------------------------------------

pub(crate) type GuestCallFn = fn(u64, [u64; 6]) -> u64;

fn default_guest_call(func: u64, args: [u64; 6]) -> u64 {
    ps5cpu::exec::call_guest_nested(func, args)
}

static GUEST_CALL: RwLock<GuestCallFn> = RwLock::new(default_guest_call);

pub(crate) fn guest_call(func: u64, args: [u64; 6]) -> u64 {
    let f = *GUEST_CALL.read().unwrap_or_else(|p| p.into_inner());
    f(func, args)
}

#[cfg(test)]
pub(crate) fn set_guest_call(f: GuestCallFn) {
    *GUEST_CALL.write().unwrap() = f;
}

#[cfg(test)]
thread_local! {
    pub(crate) static TEST_XMM: std::cell::RefCell<[f32; 8]> = const { std::cell::RefCell::new([0.0; 8]) };
}

/// Single-precision floating-point argument `i` (xmm`i`).
fn float_arg(i: usize) -> f32 {
    #[cfg(test)]
    {
        TEST_XMM.with(|x| x.borrow()[i])
    }
    #[cfg(not(test))]
    {
        ps5cpu::exec::trap_xmm_low(i).map_or(0.0, |v| f32::from_bits(v as u32))
    }
}

// ---------------------------------------------------------------------------
// Host state
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Style {
    dpi_x: u32,
    dpi_y: u32,
    /// 0 = pixels, 1 = points.
    scale_unit: u32,
    scale_w: f32,
    scale_h: f32,
    weight_x: f32,
    weight_y: f32,
    slant: f32,
}

impl Style {
    fn new(scale: f32) -> Self {
        Style {
            dpi_x: DEFAULT_DPI,
            dpi_y: DEFAULT_DPI,
            scale_unit: 0,
            scale_w: scale,
            scale_h: scale,
            weight_x: 0.0,
            weight_y: 0.0,
            slant: 0.0,
        }
    }

    /// Effective pixel scale.
    fn pixels(&self) -> (f32, f32) {
        if self.scale_unit != 0 {
            let fx = if self.dpi_x != 0 { self.dpi_x as f32 / POINTS_PER_INCH } else { 1.0 };
            let fy = if self.dpi_y != 0 { self.dpi_y as f32 / POINTS_PER_INCH } else { 1.0 };
            (self.scale_w * fx, self.scale_h * fy)
        } else {
            (self.scale_w, self.scale_h)
        }
    }

    fn set_scale_pixel(&mut self, w: f32, h: f32) {
        self.scale_unit = 0;
        self.scale_w = w;
        self.scale_h = h;
    }

    fn set_slant(&mut self, s: f32) {
        if self.slant != s {
            self.slant = s.clamp(-1.0, 1.0);
        }
    }

    fn set_weight(&mut self, wx: f32, wy: f32) {
        self.weight_x = (wx - 1.0).clamp(-0.04, 0.04);
        self.weight_y = (wy - 1.0).clamp(-0.04, 0.04);
    }
}

#[derive(Default)]
struct Library {
    external_fonts: bool,
    system_fonts: bool,
    device_cache: u64,
    device_cache_owned: bool,
}

#[derive(Clone)]
pub(crate) struct Font {
    library: u64,
    /// The handle block was allocated by the library (not caller-provided).
    owned: bool,
    face: Arc<FaceData>,
    font_set: u32,
    style: Style,
    render_style: Style,
    renderer: u64,
    attribute: i32,
    /// sceFontSetScriptLanguage: language per script (unset reads 0).
    script_languages: BTreeMap<i32, i32>,
    /// sceFontSetTypographicDesign: feature per typographic item.
    typographic_features: BTreeMap<i32, i32>,
}

struct Glyph {
    attribute: i32,
    /// Font the glyph was generated from and its character code.
    owner: u64,
    code: u32,
    /// Guest block holding the outline points, contour ends and tags once
    /// sceFontGlyphRefersOutline built them (0 before).
    outline: u64,
}

/// A renderer's outline workspace (guest block and size).
struct Renderer {
    workspace: u64,
    workspace_size: u32,
}

#[derive(Default)]
struct State {
    libraries: HashMap<u64, Library>,
    renderers: HashMap<u64, Renderer>,
    fonts: HashMap<u64, Font>,
    glyphs: HashMap<u64, Glyph>,
    driver_table: u64,
    renderer_selection: u64,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

struct StateGuard(MutexGuard<'static, Option<State>>);

impl std::ops::Deref for StateGuard {
    type Target = State;
    fn deref(&self) -> &State {
        self.0.as_ref().unwrap()
    }
}

impl std::ops::DerefMut for StateGuard {
    fn deref_mut(&mut self) -> &mut State {
        self.0.as_mut().unwrap()
    }
}

fn state() -> StateGuard {
    let mut g = STATE.lock().unwrap_or_else(|p| p.into_inner());
    g.get_or_insert_with(State::default);
    StateGuard(g)
}

/// Snapshot of an open font.
pub(crate) fn font(handle: u64) -> Option<Font> {
    state().fonts.get(&handle).cloned()
}

fn read_u16(ctx: &CpuContext, addr: u64) -> Option<u16> {
    ctx.read_bytes(addr, 2).map(|b| u16::from_le_bytes([b[0], b[1]]))
}

fn write_u16(ctx: &CpuContext, addr: u64, v: u16) -> bool {
    ctx.write_bytes(addr, &v.to_le_bytes())
}

fn stack_arg(ctx: &CpuContext, i: u64) -> u64 {
    ctx.read_u64(ctx.get(CpuRegister::Rsp) + 8 + 8 * i).unwrap_or(0)
}

/// A FontMemory descriptor initialised by sceFontMemoryInit with an
/// interface providing alloc and free.
pub(crate) fn valid_memory(ctx: &CpuContext, memory: u64) -> bool {
    if read_u16(ctx, memory) != Some(MEMORY_MAGIC) {
        return false;
    }
    let iface = ctx.read_u64(memory + 0x18).unwrap_or(0);
    iface != 0 && ctx.read_u64(iface).unwrap_or(0) != 0 && ctx.read_u64(iface + 8).unwrap_or(0) != 0
}

// ---------------------------------------------------------------------------
// Memory, library and renderer
// ---------------------------------------------------------------------------

/// sceFontMemoryInit(FontMemory*, regionAddress, regionSize, iface,
/// mspaceObject, destroyCallback, destroyObject)
fn cb_memory_init(ctx: &mut CpuContext) -> HleResult {
    let (memory, region, size, iface, mspace, destroy) =
        (ctx.rdi(), ctx.rsi(), ctx.rdx() as u32, ctx.rcx(), ctx.r8(), ctx.r9());
    let destroy_obj = stack_arg(ctx, 0);
    if memory == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    if iface == 0 && (region == 0 || size == 0) {
        write_u16(ctx, memory, 0);
        return ERROR_INVALID_PARAMETER;
    }
    let mut b = [0u8; 0x40];
    b[0..2].copy_from_slice(&MEMORY_MAGIC.to_le_bytes());
    b[4..8].copy_from_slice(&size.to_le_bytes());
    b[8..16].copy_from_slice(&region.to_le_bytes());
    b[0x10..0x18].copy_from_slice(&mspace.to_le_bytes());
    b[0x18..0x20].copy_from_slice(&iface.to_le_bytes());
    b[0x20..0x28].copy_from_slice(&destroy.to_le_bytes());
    b[0x28..0x30].copy_from_slice(&destroy_obj.to_le_bytes());
    b[0x38..0x40].copy_from_slice(&mspace.to_le_bytes());
    if !ctx.write_bytes(memory, &b) {
        return ERROR_INVALID_PARAMETER;
    }
    0
}

/// sceFontMemoryTerm(FontMemory*)
fn cb_memory_term(ctx: &mut CpuContext) -> HleResult {
    let memory = ctx.rdi();
    if memory == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    if read_u16(ctx, memory) != Some(MEMORY_MAGIC) {
        return ERROR_INVALID_MEMORY;
    }
    write_u16(ctx, memory, 0);
    0
}

/// sceFontSelectLibraryFt(int value): the FreeType-class driver table.
fn cb_select_library_ft(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() as i32 != 0 {
        return 0;
    }
    let mut s = state();
    if s.driver_table == 0 {
        s.driver_table = galloc(ctx, DRIVER_TABLE_SIZE).unwrap_or(0);
    }
    s.driver_table as HleResult
}

/// sceFontSelectRendererFt(int value): the renderer selection record.
fn cb_select_renderer_ft(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() as i32 != 0 {
        return 0;
    }
    let mut s = state();
    if s.renderer_selection == 0 {
        if let Some(a) = galloc(ctx, RENDERER_SELECTION_SIZE) {
            ctx.write_u32(a + 4, RENDERER_SIZE as u32);
            s.renderer_selection = a;
        }
    }
    s.renderer_selection as HleResult
}

/// sceFontCreateLibraryWithEdition(memory, createParams, edition, FontLibrary*)
fn cb_create_library_with_edition(ctx: &mut CpuContext) -> HleResult {
    let (memory, params, edition, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx());
    create_library(ctx, memory, params, edition, out)
}

/// sceFontCreateLibrary(memory, createParams, FontLibrary*): edition 0.
fn cb_create_library(ctx: &mut CpuContext) -> HleResult {
    let (memory, params, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    create_library(ctx, memory, params, 0, out)
}

fn create_library(ctx: &CpuContext, memory: u64, params: u64, edition: u64, out: u64) -> HleResult {
    if out != 0 {
        ctx.write_u64(out, 0);
    }
    if memory == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    if !valid_memory(ctx, memory) {
        return ERROR_INVALID_MEMORY;
    }
    if params == 0 || out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let Some(lib) = galloc(ctx, LIBRARY_SIZE) else {
        return ERROR_ALLOCATION_FAILED;
    };
    write_u16(ctx, lib, LIBRARY_MAGIC);
    ctx.write_u64(lib + 0x80, params);
    state().libraries.insert(lib, Library::default());
    ctx.write_u64(out, lib);
    log::debug!("[font] library {lib:#x} created (edition {:#x})", edition >> 32);
    0
}

fn library_ok(ctx: &CpuContext, lib: u64) -> bool {
    lib != 0 && read_u16(ctx, lib) == Some(LIBRARY_MAGIC) && state().libraries.contains_key(&lib)
}

/// sceFontDestroyLibrary(FontLibrary*)
fn cb_destroy_library(ctx: &mut CpuContext) -> HleResult {
    let p = ctx.rdi();
    if p == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let lib = ctx.read_u64(p).unwrap_or(0);
    if !library_ok(ctx, lib) {
        return ERROR_INVALID_LIBRARY;
    }
    let Some(l) = state().libraries.remove(&lib) else {
        return ERROR_INVALID_LIBRARY;
    };
    if l.device_cache_owned {
        gfree(l.device_cache);
    }
    write_u16(ctx, lib, 0);
    gfree(lib);
    ctx.write_u64(p, 0);
    0
}

/// sceFontCreateRendererWithEdition(memory, createParams, edition, FontRenderer*)
fn cb_create_renderer_with_edition(ctx: &mut CpuContext) -> HleResult {
    let (memory, params, out) = (ctx.rdi(), ctx.rsi(), ctx.rcx());
    create_renderer(ctx, memory, params, out)
}

/// sceFontCreateRenderer(memory, createParams, FontRenderer*): edition 0.
fn cb_create_renderer(ctx: &mut CpuContext) -> HleResult {
    let (memory, params, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    create_renderer(ctx, memory, params, out)
}

/// Default size of a renderer's outline workspace.
const RENDERER_WORKSPACE_SIZE: u32 = 0x4000;

fn create_renderer(ctx: &CpuContext, memory: u64, params: u64, out: u64) -> HleResult {
    let fail = |ctx: &CpuContext, rc: HleResult| {
        if out != 0 {
            ctx.write_u64(out, 0);
        }
        rc
    };
    if memory == 0 {
        return fail(ctx, ERROR_INVALID_PARAMETER);
    }
    if !valid_memory(ctx, memory) {
        return fail(ctx, ERROR_INVALID_MEMORY);
    }
    if params == 0 || out == 0 {
        return fail(ctx, ERROR_INVALID_PARAMETER);
    }
    let Some(r) = galloc(ctx, RENDERER_SIZE) else {
        return fail(ctx, ERROR_ALLOCATION_FAILED);
    };
    let Some(workspace) = galloc(ctx, RENDERER_WORKSPACE_SIZE as u64) else {
        gfree(r);
        return fail(ctx, ERROR_ALLOCATION_FAILED);
    };
    write_u16(ctx, r, RENDERER_MAGIC);
    ctx.write_u64(r + 0x80, params);
    state().renderers.insert(r, Renderer { workspace, workspace_size: RENDERER_WORKSPACE_SIZE });
    ctx.write_u64(out, r);
    0
}

fn renderer_ok(ctx: &CpuContext, r: u64) -> bool {
    r != 0 && read_u16(ctx, r) == Some(RENDERER_MAGIC) && state().renderers.contains_key(&r)
}

/// sceFontDestroyRenderer(FontRenderer*)
fn cb_destroy_renderer(ctx: &mut CpuContext) -> HleResult {
    let p = ctx.rdi();
    if p == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let r = ctx.read_u64(p).unwrap_or(0);
    if r == 0 || read_u16(ctx, r) != Some(RENDERER_MAGIC) {
        return ERROR_INVALID_RENDERER;
    }
    let Some(renderer) = state().renderers.remove(&r) else {
        return ERROR_INVALID_RENDERER;
    };
    if renderer.workspace != 0 {
        gfree(renderer.workspace);
    }
    write_u16(ctx, r, 0);
    gfree(r);
    ctx.write_u64(p, 0);
    0
}

/// sceFontAttachDeviceCacheBuffer(library, buffer, size)
fn cb_attach_device_cache_buffer(ctx: &mut CpuContext) -> HleResult {
    let (lib, buffer, size) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as u32);
    if !library_ok(ctx, lib) {
        return ERROR_INVALID_LIBRARY;
    }
    let mut s = state();
    let l = s.libraries.get_mut(&lib).unwrap();
    if l.device_cache != 0 {
        return ERROR_ALREADY_ATTACHED;
    }
    if size < 0x1020 {
        return ERROR_INVALID_PARAMETER;
    }
    let (header, owned) = if buffer == 0 {
        match galloc(ctx, size as u64) {
            Some(a) => (a, true),
            None => return ERROR_ALLOCATION_FAILED,
        }
    } else {
        (buffer, false)
    };
    let pages = (size - 0x1000) >> 12;
    let mut b = [0u8; 24];
    b[0..4].copy_from_slice(&size.to_le_bytes());
    b[4..8].copy_from_slice(&pages.to_le_bytes());
    b[12..16].copy_from_slice(&pages.to_le_bytes());
    b[16..24].copy_from_slice(&0x0FF8_0000_1000u64.to_le_bytes());
    if !ctx.write_bytes(header, &b) {
        return ERROR_INVALID_PARAMETER;
    }
    l.device_cache = header;
    l.device_cache_owned = owned;
    0
}

fn support_fonts(ctx: &CpuContext, lib: u64, system: bool) -> HleResult {
    if !library_ok(ctx, lib) {
        return ERROR_INVALID_LIBRARY;
    }
    if ctx.read_u64(lib + 0x80).unwrap_or(0) == 0 {
        return ERROR_INVALID_LIBRARY;
    }
    let mut s = state();
    let l = s.libraries.get_mut(&lib).unwrap();
    let flag = if system { &mut l.system_fonts } else { &mut l.external_fonts };
    if *flag {
        return ERROR_ALREADY_SPECIFIED;
    }
    *flag = true;
    0
}

/// sceFontSupportExternalFonts(library, fontMax, formats)
fn cb_support_external_fonts(ctx: &mut CpuContext) -> HleResult {
    support_fonts(ctx, ctx.rdi(), false)
}

/// sceFontSupportSystemFonts(library)
fn cb_support_system_fonts(ctx: &mut CpuContext) -> HleResult {
    support_fonts(ctx, ctx.rdi(), true)
}

// ---------------------------------------------------------------------------
// Opening and closing fonts
// ---------------------------------------------------------------------------

/// Bind `face` to a handle block (`existing` when caller-provided).
fn install_font(ctx: &CpuContext, lib: u64, existing: u64, face: Arc<FaceData>, font_set: u32) -> Result<u64, HleResult> {
    let (handle, owned) = if existing != 0 {
        (existing, false)
    } else {
        (galloc(ctx, HANDLE_SIZE).ok_or(ERROR_ALLOCATION_FAILED)?, true)
    };
    let mut b = [0u8; 0x40];
    b[0..2].copy_from_slice(&HANDLE_MAGIC.to_le_bytes());
    let l = HANDLE_LIBRARY as usize;
    b[l..l + 8].copy_from_slice(&lib.to_le_bytes());
    if !ctx.write_bytes(handle, &b) {
        if owned {
            gfree(handle);
        }
        return Err(ERROR_INVALID_PARAMETER);
    }
    let style = Style::new(face.units_per_em as f32 / 64.0);
    state().fonts.insert(
        handle,
        Font {
            library: lib,
            owned,
            face,
            font_set,
            style,
            render_style: style,
            renderer: 0,
            attribute: 0,
            script_languages: BTreeMap::new(),
            typographic_features: BTreeMap::new(),
        },
    );
    Ok(handle)
}

/// sceFontOpenFontMemory(library, fontAddress, fontSize, const FontOpenDetail*, FontHandle*)
fn cb_open_font_memory(ctx: &mut CpuContext) -> HleResult {
    let (lib, addr, size, detail, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as u32, ctx.rcx(), ctx.r8());
    let clear = |ctx: &CpuContext| {
        if out != 0 {
            ctx.write_u64(out, 0);
        }
    };
    if !library_ok(ctx, lib) {
        clear(ctx);
        return ERROR_INVALID_LIBRARY;
    }
    if !state().libraries[&lib].external_fonts {
        clear(ctx);
        return ERROR_NO_SUPPORT_FUNCTION;
    }
    let (sub_font, unique_id) = if detail != 0 {
        (ctx.read_u32(detail + 8).unwrap_or(0), ctx.read_u32(detail + 12).unwrap_or(0) as i32)
    } else {
        (0, -1)
    };
    if addr == 0 || size == 0 || unique_id < -1 || out == 0 {
        clear(ctx);
        return ERROR_INVALID_PARAMETER;
    }
    if size > MAX_FONT_BYTES {
        clear(ctx);
        return ERROR_NO_SUPPORT_FORMAT;
    }
    let Some(bytes) = ctx.read_bytes(addr, size as usize) else {
        clear(ctx);
        return ERROR_INVALID_PARAMETER;
    };
    let Some(face) = FaceData::parse(bytes, sub_font) else {
        clear(ctx);
        return ERROR_NO_SUPPORT_FORMAT;
    };
    match install_font(ctx, lib, 0, face, 0) {
        Ok(h) => {
            ctx.write_u64(out, h);
            0
        }
        Err(e) => {
            clear(ctx);
            e
        }
    }
}

/// sceFontOpenFontSet(library, fontSetType, openMode, const FontOpenDetail*, FontHandle*)
fn cb_open_font_set(ctx: &mut CpuContext) -> HleResult {
    let (lib, set_type, mode, out) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx() as u32, ctx.r8());
    if !library_ok(ctx, lib) {
        if out != 0 {
            ctx.write_u64(out, 0);
        }
        return ERROR_INVALID_LIBRARY;
    }
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let rc = (|| {
        if !(1..=3).contains(&mode) {
            return Err(ERROR_INVALID_PARAMETER);
        }
        if !state().libraries[&lib].system_fonts {
            return Err(ERROR_NO_SUPPORT_FUNCTION);
        }
        if !face::is_system_font_set(set_type) {
            return Err(ERROR_NO_SUPPORT_FONTSET);
        }
        let (path, sub) = face::find_system_font_file(set_type).ok_or(ERROR_FONT_OPEN_FAILED)?;
        let face = face::load_font_file(&path, sub).ok_or(ERROR_FONT_OPEN_FAILED)?;
        log::info!("[font] font set {set_type:#010x} -> {}", path.display());
        install_font(ctx, lib, 0, face, set_type)
    })();
    match rc {
        Ok(h) => {
            ctx.write_u64(out, h);
            0
        }
        Err(e) => {
            if e == ERROR_FONT_OPEN_FAILED {
                log::warn!("[font] no font file for system font set {set_type:#010x}");
                static SEARCHED: std::sync::Once = std::sync::Once::new();
                SEARCHED.call_once(|| {
                    log::warn!(
                        "[font] system fonts are looked for in {}; put the console's fonts (or Noto substitutes) there or name a directory with RPS5E_SYSTEM_FONTS",
                        face::font_dirs().iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", ")
                    );
                });
            }
            ctx.write_u64(out, 0);
            e
        }
    }
}

/// sceFontOpenFontInstance(fontHandle, setupFont, FontHandle*)
fn cb_open_font_instance(ctx: &mut CpuContext) -> HleResult {
    let (src, setup, out) = (ctx.rdi(), ctx.rsi(), ctx.rdx());
    let clear = |ctx: &CpuContext| {
        if out != 0 {
            ctx.write_u64(out, 0);
        }
    };
    if src == 0 {
        clear(ctx);
        return ERROR_INVALID_FONT_HANDLE;
    }
    if setup == 0 && out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let Some(f) = font(src) else {
        clear(ctx);
        return ERROR_INVALID_FONT_HANDLE;
    };
    if setup != 0 {
        state().fonts.remove(&setup);
    }
    match install_font(ctx, f.library, setup, f.face.clone(), f.font_set) {
        Ok(h) => {
            if let Some(nf) = state().fonts.get_mut(&h) {
                nf.style = f.style;
                nf.render_style = f.render_style;
                nf.renderer = f.renderer;
            }
            if f.renderer != 0 {
                ctx.write_u64(h + HANDLE_RENDERER, f.renderer);
            }
            if out != 0 {
                ctx.write_u64(out, h);
            }
            0
        }
        Err(e) => {
            clear(ctx);
            e
        }
    }
}

/// sceFontCloseFont(fontHandle)
fn cb_close_font(ctx: &mut CpuContext) -> HleResult {
    let h = ctx.rdi();
    let Some(f) = state().fonts.remove(&h) else {
        return ERROR_INVALID_FONT_HANDLE;
    };
    ctx.write_bytes(h, &[0u8; 0x40]);
    if f.owned {
        gfree(h);
    }
    0
}

/// sceFontBindRenderer(fontHandle, renderer)
fn cb_bind_renderer(ctx: &mut CpuContext) -> HleResult {
    let (h, r) = (ctx.rdi(), ctx.rsi());
    let renderer_ok = renderer_ok(ctx, r);
    let mut s = state();
    let Some(f) = s.fonts.get_mut(&h) else {
        return ERROR_INVALID_FONT_HANDLE;
    };
    if f.renderer != 0 {
        return ERROR_ALREADY_BOUND_RENDERER;
    }
    if !renderer_ok {
        return ERROR_INVALID_RENDERER;
    }
    f.render_style = f.style;
    f.renderer = r;
    drop(s);
    ctx.write_u64(h + HANDLE_RENDERER, r);
    0
}

/// sceFontUnbindRenderer(fontHandle)
fn cb_unbind_renderer(ctx: &mut CpuContext) -> HleResult {
    let h = ctx.rdi();
    let mut s = state();
    let Some(f) = s.fonts.get_mut(&h) else {
        return ERROR_INVALID_FONT_HANDLE;
    };
    if f.renderer == 0 {
        return ERROR_NOT_BOUND_RENDERER;
    }
    f.renderer = 0;
    drop(s);
    ctx.write_u64(h + HANDLE_RENDERER, 0);
    0
}

/// sceFontRebindRenderer(fontHandle): re-applies the font's current style
/// to the bound renderer.
fn cb_rebind_renderer(ctx: &mut CpuContext) -> HleResult {
    let h = ctx.rdi();
    let mut s = state();
    let Some(f) = s.fonts.get_mut(&h) else {
        return ERROR_INVALID_FONT_HANDLE;
    };
    if f.renderer == 0 {
        return ERROR_NOT_BOUND_RENDERER;
    }
    f.render_style = f.style;
    0
}

/// sceFontGetAttribute(fontHandle, int attribute, int* nowAttribute): the
/// output first, then the handle.
fn cb_get_attribute(ctx: &mut CpuContext) -> HleResult {
    let (h, out) = (ctx.rdi(), ctx.rdx());
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let Some(now) = state().fonts.get(&h).map(|f| f.attribute) else {
        return ERROR_INVALID_FONT_HANDLE;
    };
    ctx.write_u32(out, now as u32);
    0
}

/// sceFontGlyphGetAttribute(FontGlyph, int attribute, int* nowAttribute)
fn cb_glyph_get_attribute(ctx: &mut CpuContext) -> HleResult {
    let (g, out) = (ctx.rdi(), ctx.rdx());
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let Some(now) = state().glyphs.get(&g).map(|glyph| glyph.attribute) else {
        return ERROR_INVALID_GLYPH;
    };
    ctx.write_u32(out, now as u32);
    0
}

/// sceFontDefineAttribute(fontHandle, int attribute, int* oldAttribute)
fn cb_define_attribute(ctx: &mut CpuContext) -> HleResult {
    let (h, attribute, old) = (ctx.rdi(), ctx.rsi() as i32, ctx.rdx());
    let prev = match state().fonts.get_mut(&h) {
        Some(f) => {
            let prev = f.attribute;
            f.attribute = attribute;
            Some(prev)
        }
        None => None,
    };
    if old != 0 {
        ctx.write_u32(old, prev.unwrap_or(0) as u32);
    }
    if prev.is_none() {
        return ERROR_INVALID_FONT_HANDLE;
    }
    0
}

/// Store `value` under `key` in one of a font's setting tables.
fn set_font_setting(h: u64, table: fn(&mut Font) -> &mut BTreeMap<i32, i32>, key: i32, value: i32) -> HleResult {
    match state().fonts.get_mut(&h) {
        Some(f) => {
            table(f).insert(key, value);
            0
        }
        None => ERROR_INVALID_FONT_HANDLE,
    }
}

/// Read `key` of a font's setting table into `out` (0 when never set).
fn get_font_setting(ctx: &CpuContext, h: u64, table: fn(&mut Font) -> &mut BTreeMap<i32, i32>, key: i32, out: u64) -> HleResult {
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let value = match state().fonts.get_mut(&h) {
        Some(f) => table(f).get(&key).copied().unwrap_or(0),
        None => return ERROR_INVALID_FONT_HANDLE,
    };
    if ctx.write_u32(out, value as u32) { 0 } else { ERROR_INVALID_PARAMETER }
}

fn script_languages(f: &mut Font) -> &mut BTreeMap<i32, i32> {
    &mut f.script_languages
}

fn typographic_features(f: &mut Font) -> &mut BTreeMap<i32, i32> {
    &mut f.typographic_features
}

/// sceFontSetScriptLanguage(fontHandle, int script, int language)
fn cb_set_script_language(ctx: &mut CpuContext) -> HleResult {
    set_font_setting(ctx.rdi(), script_languages, ctx.rsi() as i32, ctx.rdx() as i32)
}

/// sceFontGetScriptLanguage(fontHandle, int script, int* language)
fn cb_get_script_language(ctx: &mut CpuContext) -> HleResult {
    get_font_setting(ctx, ctx.rdi(), script_languages, ctx.rsi() as i32, ctx.rdx())
}

/// sceFontSetTypographicDesign(fontHandle, int typographic, int feature)
fn cb_set_typographic_design(ctx: &mut CpuContext) -> HleResult {
    set_font_setting(ctx.rdi(), typographic_features, ctx.rsi() as i32, ctx.rdx() as i32)
}

/// sceFontGetTypographicDesign(fontHandle, int typographic, int* feature)
fn cb_get_typographic_design(ctx: &mut CpuContext) -> HleResult {
    get_font_setting(ctx, ctx.rdi(), typographic_features, ctx.rsi() as i32, ctx.rdx())
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

fn update_style(h: u64, f: impl FnOnce(&mut Style)) -> HleResult {
    match state().fonts.get_mut(&h) {
        Some(font) => {
            f(&mut font.style);
            0
        }
        None => ERROR_INVALID_FONT_HANDLE,
    }
}

fn update_render_style(h: u64, f: impl FnOnce(&mut Style) -> HleResult) -> HleResult {
    match state().fonts.get_mut(&h) {
        Some(font) if font.renderer != 0 => f(&mut font.render_style),
        Some(_) => ERROR_NOT_BOUND_RENDERER,
        None => ERROR_INVALID_FONT_HANDLE,
    }
}

/// sceFontSetScalePixel(fontHandle, float w, float h)
fn cb_set_scale_pixel(ctx: &mut CpuContext) -> HleResult {
    let (w, h) = (float_arg(0), float_arg(1));
    update_style(ctx.rdi(), |s| s.set_scale_pixel(w, h))
}

/// sceFontSetEffectSlant(fontHandle, float slantRatio)
fn cb_set_effect_slant(ctx: &mut CpuContext) -> HleResult {
    let s = float_arg(0);
    update_style(ctx.rdi(), |st| st.set_slant(s))
}

/// sceFontSetEffectWeight(fontHandle, float wx, float wy, uint32 mode)
fn cb_set_effect_weight(ctx: &mut CpuContext) -> HleResult {
    if ctx.rsi() as u32 != 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let (wx, wy) = (float_arg(0), float_arg(1));
    update_style(ctx.rdi(), |s| s.set_weight(wx, wy))
}

/// sceFontSetupRenderScalePixel(fontHandle, float w, float h)
fn cb_setup_render_scale_pixel(ctx: &mut CpuContext) -> HleResult {
    let (w, h) = (float_arg(0), float_arg(1));
    update_render_style(ctx.rdi(), |s| {
        s.set_scale_pixel(w, h);
        0
    })
}

/// sceFontSetupRenderEffectSlant(fontHandle, float slantRatio)
fn cb_setup_render_effect_slant(ctx: &mut CpuContext) -> HleResult {
    let v = float_arg(0);
    update_render_style(ctx.rdi(), |s| {
        s.set_slant(v);
        0
    })
}

/// sceFontSetupRenderEffectWeight(fontHandle, float wx, float wy, uint32 mode)
fn cb_setup_render_effect_weight(ctx: &mut CpuContext) -> HleResult {
    let mode = ctx.rsi() as u32;
    let (wx, wy) = (float_arg(0), float_arg(1));
    update_render_style(ctx.rdi(), |s| {
        if mode != 0 {
            return ERROR_INVALID_PARAMETER;
        }
        s.set_weight(wx, wy);
        0
    })
}

// ---------------------------------------------------------------------------
// Metrics and layout
// ---------------------------------------------------------------------------

/// FontGlyphMetrics: width, height, horizontal {bearingX, bearingY,
/// advance}, vertical {bearingX, bearingY, advance}.
pub(crate) type GlyphMetrics = [f32; 8];

fn round_to_26_6(units: f32, px: f32, upem: f32) -> i64 {
    (units * px * 64.0 / upem).round() as i64
}

/// Unhinted glyph metrics of `code` at `style`.
///
/// Vertical metrics as the font engine lays a glyph out vertically: the
/// bearing x is half the horizontal advance left of the horizontal bearing;
/// a face with vertical tables gives the top bearing and advance height,
/// otherwise the advance height is the typographic line height and the glyph
/// is centred in it.
pub(crate) fn char_metrics(face: &FaceData, code: u32, style: &Style) -> Result<GlyphMetrics, HleResult> {
    let gid = face.glyph_index(code).ok_or(ERROR_NO_SUPPORT_GLYPH)?;
    let (w, h) = style.pixels();
    let upem = face.units_per_em as f32;
    let (sx, sy) = (w / upem, h / upem);
    let u = face.glyph_units(gid);
    // 26.6 fixed point.
    let fx = |v: f32| (v * 64.0).round() as i64;
    let from_fx = |v: i64| v as f32 / 64.0;
    let height = fx((u.y_max - u.y_min) * sy);
    let bearing_x = fx(u.x_min * sx);
    let advance = fx(u.advance * sx);
    let (vert_bearing_y, vert_advance) = match face.vertical_units(gid) {
        Some((top, adv)) => (fx(top * sy), fx(adv * sy)),
        None => {
            let adv = fx(face.synthesized_vertical_advance() * sy);
            ((adv - height) / 2, adv)
        }
    };
    let vert_bearing_x = bearing_x - advance / 2;
    Ok([
        q64((u.x_max - u.x_min) * sx),
        from_fx(height),
        from_fx(bearing_x),
        q64(u.y_max * sy),
        from_fx(advance),
        from_fx(vert_bearing_x),
        from_fx(vert_bearing_y),
        from_fx(vert_advance),
    ])
}

fn write_metrics(ctx: &CpuContext, addr: u64, m: &GlyphMetrics) -> bool {
    let mut b = [0u8; 32];
    for (i, v) in m.iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    ctx.write_bytes(addr, &b)
}

/// (baseline, line advance, decoration extent) for horizontal writing.
fn horizontal_layout(face: &FaceData, style: &Style) -> [f32; 3] {
    let (_, h) = style.pixels();
    let upem = face.units_per_em as f32;
    let mut y_min = round_to_26_6(face.bbox[1] as f32, h, upem);
    let mut y_max = round_to_26_6(face.bbox[3] as f32, h, upem);
    let mut effect = 0.0;
    if style.weight_y != 0.0 {
        let half = (style.weight_y * h * 64.0) as i64 / 2;
        effect = half as f32 / 64.0;
        y_min -= half;
        y_max += half;
    }
    let baseline = (y_max as f32 / 64.0).max(0.0);
    let descent = ((y_max - y_min) as f32 / 64.0 - y_max as f32 / 64.0).max(0.0);
    [baseline, baseline + descent, effect]
}

/// (baseline offset x, column advance, decoration span) for vertical writing.
fn vertical_layout(face: &FaceData, style: &Style) -> [f32; 3] {
    let (w, _) = style.pixels();
    let upem = face.units_per_em as f32;
    let x_min = round_to_26_6(face.bbox[0] as f32, w, upem);
    let x_max = round_to_26_6(face.bbox[2] as f32, w, upem);
    let half = (-x_min).max(x_max).max(0) as f32 / 64.0;
    let span = if style.weight_x != 0.0 { ((style.weight_x * w * 64.0) as i64 / 2) as f32 / 64.0 } else { 0.0 };
    [-half, half * 2.0, span]
}

fn write_floats(ctx: &CpuContext, addr: u64, v: &[f32]) -> bool {
    let b: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
    ctx.write_bytes(addr, &b)
}

/// sceFontGetHorizontalLayout(fontHandle, FontHorizontalLayout*)
/// { baselineOffset, lineAdvance, decorationExtent }
fn cb_get_horizontal_layout(ctx: &mut CpuContext) -> HleResult {
    let (h, out) = (ctx.rdi(), ctx.rsi());
    let Some(f) = font(h) else {
        if out != 0 {
            write_floats(ctx, out, &[0.0; 3]);
        }
        return ERROR_INVALID_FONT_HANDLE;
    };
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let l = horizontal_layout(&f.face, &f.style);
    write_floats(ctx, out, &l);
    0
}

/// sceFontGetVerticalLayout(fontHandle, FontVerticalLayout*)
/// { baselineOffsetX, columnAdvance, decorationSpan }
fn cb_get_vertical_layout(ctx: &mut CpuContext) -> HleResult {
    let (h, out) = (ctx.rdi(), ctx.rsi());
    let Some(f) = font(h) else {
        if out != 0 {
            write_floats(ctx, out, &[0.0; 3]);
        }
        return ERROR_INVALID_FONT_HANDLE;
    };
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    write_floats(ctx, out, &vertical_layout(&f.face, &f.style));
    0
}

/// sceFontGetRenderCharGlyphMetrics(fontHandle, code, FontGlyphMetrics*):
/// metrics at the bound renderer's style.
fn cb_get_render_char_glyph_metrics(ctx: &mut CpuContext) -> HleResult {
    get_char_glyph_metrics(ctx, true)
}

/// sceFontGetCharGlyphMetrics(fontHandle, code, FontGlyphMetrics*):
/// metrics at the font's own style.
fn cb_get_char_glyph_metrics(ctx: &mut CpuContext) -> HleResult {
    get_char_glyph_metrics(ctx, false)
}

fn get_char_glyph_metrics(ctx: &CpuContext, render: bool) -> HleResult {
    let (h, code, out) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx());
    let clear = |ctx: &CpuContext| {
        if out != 0 {
            write_metrics(ctx, out, &[0.0; 8]);
        }
    };
    let Some(f) = font(h) else {
        clear(ctx);
        return ERROR_INVALID_FONT_HANDLE;
    };
    if code == 0 {
        clear(ctx);
        return ERROR_NO_SUPPORT_CODE;
    }
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    if render && f.renderer == 0 {
        clear(ctx);
        return ERROR_NOT_BOUND_RENDERER;
    }
    let style = if render { &f.render_style } else { &f.style };
    match char_metrics(&f.face, code, style) {
        Ok(m) => {
            write_metrics(ctx, out, &m);
            0
        }
        Err(e) => {
            clear(ctx);
            e
        }
    }
}

/// Code argument of sceFontGetKerning: a Unicode code point, or a glyph
/// index when the top bit is set.
const KERNING_GLYPH_INDEX_FLAG: u32 = 0x8000_0000;

/// sceFontGetKerning(fontHandle, preCode, code, FontKerning*)
/// FontKerning { offsetX, offsetY, positionX, positionY }
fn cb_get_kerning(ctx: &mut CpuContext) -> HleResult {
    let (h, pre_code, code, out) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx() as u32, ctx.rcx());
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    write_floats(ctx, out, &[0.0; 4]);
    let Some(f) = font(h) else {
        return ERROR_INVALID_FONT_HANDLE;
    };
    if pre_code == 0 {
        return 0;
    }
    let glyph = |v: u32| -> Option<u16> {
        if v & KERNING_GLYPH_INDEX_FLAG != 0 {
            Some((v & 0xFFFF) as u16)
        } else {
            f.face.glyph_index(v)
        }
    };
    let n = f.face.glyph_count();
    let (Some(left), Some(right)) = (glyph(pre_code), glyph(code)) else {
        return ERROR_NO_SUPPORT_CODE;
    };
    if left == 0 || left >= n || right == 0 || right >= n {
        return ERROR_NO_SUPPORT_CODE;
    }
    let (w, _) = f.style.pixels();
    let units = f.face.kerning(left, right);
    write_floats(ctx, out, &[units as f32 * w / f.face.units_per_em as f32, 0.0, 0.0, 0.0]);
    0
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// FontRenderSurface (0x80): buffer, widthByte i32, pixelSizeByte i8,
/// pad, styleFlag u8, pad, width i32, height i32, scissor x0/y0/x1/y1,
/// system use (style frame pointer first).
struct Surface {
    buffer: u64,
    width_byte: i32,
    pixel_size: i32,
    style_flag: u8,
    width: i32,
    height: i32,
    scissor: [u32; 4],
    style_frame: u64,
}

fn read_surface(ctx: &CpuContext, addr: u64) -> Option<Surface> {
    let b = ctx.read_bytes(addr, 0x30)?;
    let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    Some(Surface {
        buffer: u64::from_le_bytes(b[0..8].try_into().unwrap()),
        width_byte: u32_at(8) as i32,
        pixel_size: b[0xC] as i8 as i32,
        style_flag: b[0xE],
        width: u32_at(0x10) as i32,
        height: u32_at(0x14) as i32,
        scissor: [u32_at(0x18), u32_at(0x1C), u32_at(0x20), u32_at(0x24)],
        style_frame: u64::from_le_bytes(b[0x28..0x30].try_into().unwrap()),
    })
}

/// Apply a surface's style frame (if attached) on top of the render style.
fn surface_style(ctx: &CpuContext, base: &Style, s: &Surface) -> Style {
    let mut st = *base;
    if s.style_flag & 1 == 0 || s.style_frame == 0 {
        return st;
    }
    let Some(f) = ctx.read_bytes(s.style_frame, 0x28) else {
        return st;
    };
    if u16::from_le_bytes([f[0], f[1]]) != STYLE_FRAME_MAGIC {
        return st;
    }
    let u32_at = |o: usize| u32::from_le_bytes(f[o..o + 4].try_into().unwrap());
    let f32_at = |o: usize| f32::from_bits(u32_at(o));
    let flags = f[2];
    let (hdpi, vdpi) = (u32_at(4), u32_at(8));
    if flags & STYLE_FRAME_FLAG_SCALE != 0 {
        st.scale_unit = u32_at(0xC);
        st.scale_w = f32_at(0x14);
        st.scale_h = f32_at(0x18);
        st.dpi_x = if hdpi != 0 { hdpi } else { DEFAULT_DPI };
        st.dpi_y = if vdpi != 0 { vdpi } else { DEFAULT_DPI };
    }
    if flags & STYLE_FRAME_FLAG_SLANT != 0 {
        st.slant = f32_at(0x24);
    }
    if flags & STYLE_FRAME_FLAG_WEIGHT != 0 {
        st.weight_x = f32_at(0x1C);
        st.weight_y = f32_at(0x20);
    }
    st
}

fn floor_i(v: f32) -> i32 {
    v.floor() as i32
}

fn ceil_i(v: f32) -> i32 {
    v.ceil() as i32
}

/// FontGlyphImageMetrics derived from the glyph metrics at pen (x, y).
fn image_metrics(m: &GlyphMetrics, x: f32, y: f32) -> ([f32; 4], [u32; 2]) {
    let left_f = x + m[2];
    let top_f = y + m[3];
    let right_f = left_f + m[0];
    let bottom_f = top_f - m[1];
    let (left, top, right, bottom) = (floor_i(left_f), ceil_i(top_f), ceil_i(right_f), floor_i(bottom_f));
    let advance_f = x + m[4];
    let advance = ceil_i(advance_f);
    let mut stride = advance;
    if advance_f != advance as f32 {
        let candidate = ((right as f32 - right_f) + advance_f) as i32;
        stride = right + 1;
        if advance_f <= right_f {
            stride = candidate;
        }
        if stride < candidate {
            stride = candidate;
        }
    }
    (
        [left as f32 - x, top as f32 - y, advance as f32 - x, stride as f32 - x],
        [(right - left).max(0) as u32, (top - bottom).max(0) as u32],
    )
}

/// Draw `code` at pen (x, y) (baseline origin, y down) into the surface.
/// Returns (metrics, 0x40-byte FontRenderOutput).
fn render_glyph(ctx: &CpuContext, f: &Font, code: u32, surface_addr: u64, x: f32, y: f32) -> Result<(GlyphMetrics, [u8; 0x40]), HleResult> {
    if f.renderer == 0 {
        return Err(ERROR_NOT_BOUND_RENDERER);
    }
    let s = read_surface(ctx, surface_addr).ok_or(ERROR_INVALID_PARAMETER)?;
    if s.buffer == 0 || s.width <= 0 || s.height <= 0 || s.width_byte <= 0 || s.pixel_size <= 0 {
        return Err(ERROR_NO_SUPPORT_SURFACE);
    }
    let bpp = s.pixel_size;
    if bpp != 1 && bpp != 4 {
        return Err(ERROR_NO_SUPPORT_SURFACE);
    }
    let style = surface_style(ctx, &f.render_style, &s);
    let m = char_metrics(&f.face, code, &style)?;
    let gid = f.face.glyph_index(code).ok_or(ERROR_NO_SUPPORT_GLYPH)?;
    let (pw, ph) = style.pixels();
    let upem = f.face.units_per_em as f32;
    let (ox, oy) = (x.floor(), y.floor());
    let bmp = face::rasterize(&f.face.outline(gid), pw / upem, ph / upem, style.slant, x - ox, y - oy);
    let dest_x = ox as i32 + bmp.left;
    let dest_y = oy as i32 + bmp.top;
    let clip = |v: u32, hi: i32| (v.min(i32::MAX as u32) as i32).clamp(0, hi);
    let (cx0, cy0) = (clip(s.scissor[0], s.width), clip(s.scissor[1], s.height));
    let (cx1, cy1) = (clip(s.scissor[2], s.width), clip(s.scissor[3], s.height));
    let (mut ux, mut uy, mut uw, mut uh) = (dest_x, dest_y, 0, 0);
    if bmp.width > 0 && bmp.height > 0 && cx1 > cx0 && cy1 > cy0 {
        let (r0, r1) = (dest_y.max(cy0), (dest_y + bmp.height as i32).min(cy1));
        let (c0, c1) = (dest_x.max(cx0), (dest_x + bmp.width as i32).min(cx1));
        ux = c0;
        uy = r0;
        uw = (c1 - c0).max(0);
        uh = (r1 - r0).max(0);
        if uw > 0 {
            for row in r0..r1 {
                let src = &bmp.coverage[(row - dest_y) as usize * bmp.width..][..bmp.width];
                let addr = s.buffer + row as u64 * s.width_byte as u64 + c0 as u64 * bpp as u64;
                let len = (c1 - c0) as usize * bpp as usize;
                let Some(mut line) = ctx.read_bytes(addr, len) else {
                    return Err(ERROR_NO_SUPPORT_SURFACE);
                };
                for col in c0..c1 {
                    let cov = src[(col - dest_x) as usize];
                    let o = (col - c0) as usize * bpp as usize;
                    for b in &mut line[o..o + bpp as usize] {
                        *b = (*b).max(cov);
                    }
                }
                if !ctx.write_bytes(addr, &line) {
                    return Err(ERROR_NO_SUPPORT_SURFACE);
                }
            }
        }
    }
    let (ux, uy) = (ux.max(0) as u32, uy.max(0) as u32);
    let mut out = [0u8; 0x40];
    let image = s.buffer + uy as u64 * s.width_byte as u64 + ux as u64 * bpp as u64;
    out[0x08..0x10].copy_from_slice(&image.to_le_bytes());
    out[0x10..0x14].copy_from_slice(&(s.width_byte as u32).to_le_bytes());
    out[0x14] = bpp as u8;
    for (i, v) in [ux, uy, uw.max(0) as u32, uh.max(0) as u32].iter().enumerate() {
        out[0x18 + i * 4..0x1C + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    let (fm, wh) = image_metrics(&m, x, y);
    for (i, v) in fm.iter().enumerate() {
        out[0x28 + i * 4..0x2C + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    out[0x38..0x3C].copy_from_slice(&wh[0].to_le_bytes());
    out[0x3C..0x40].copy_from_slice(&wh[1].to_le_bytes());
    Ok((m, out))
}

/// sceFontRenderCharGlyphImageHorizontal(fontHandle, code, FontRenderSurface*,
/// float x, float y, FontGlyphMetrics*, FontRenderOutput*)
fn cb_render_char_glyph_image_horizontal(ctx: &mut CpuContext) -> HleResult {
    let (h, code, surface, metrics, result) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx(), ctx.r8());
    let (x, y) = (float_arg(0), float_arg(1));
    let clear = |ctx: &CpuContext| {
        if metrics != 0 {
            write_metrics(ctx, metrics, &[0.0; 8]);
        }
        if result != 0 {
            ctx.write_bytes(result, &[0u8; 0x40]);
        }
    };
    let Some(f) = font(h) else {
        clear(ctx);
        return ERROR_INVALID_FONT_HANDLE;
    };
    let rc = if code == 0 {
        Err(ERROR_NO_SUPPORT_CODE)
    } else if surface == 0 || metrics == 0 || result == 0 {
        Err(ERROR_INVALID_PARAMETER)
    } else {
        render_glyph(ctx, &f, code, surface, x, y)
    };
    match rc {
        Ok((m, out)) => {
            write_metrics(ctx, metrics, &m);
            ctx.write_bytes(result, &out);
            0
        }
        Err(e) => {
            clear(ctx);
            e
        }
    }
}

/// void sceFontRenderSurfaceInit(FontRenderSurface*, void* buffer,
/// int bufWidthByte, int pixelSizeByte, int widthPixel, int heightPixel)
fn cb_render_surface_init(ctx: &mut CpuContext) -> HleResult {
    let s = ctx.rdi();
    if s == 0 {
        return 0;
    }
    let (buf, wb, ps) = (ctx.rsi(), ctx.rdx() as u32, ctx.rcx() as u8);
    let w = (ctx.r8() as i32).max(0) as u32;
    let h = (ctx.r9() as i32).max(0) as u32;
    let mut b = [0u8; 0x28];
    b[0..8].copy_from_slice(&buf.to_le_bytes());
    b[8..12].copy_from_slice(&wb.to_le_bytes());
    b[0xC] = ps;
    b[0x10..0x14].copy_from_slice(&w.to_le_bytes());
    b[0x14..0x18].copy_from_slice(&h.to_le_bytes());
    b[0x20..0x24].copy_from_slice(&w.to_le_bytes());
    b[0x24..0x28].copy_from_slice(&h.to_le_bytes());
    ctx.write_bytes(s, &b);
    0
}

/// void sceFontRenderSurfaceSetScissor(FontRenderSurface*, x0, y0, x1, y1):
/// stored as given; rendering clamps it to the surface.
fn cb_render_surface_set_scissor(ctx: &mut CpuContext) -> HleResult {
    let s = ctx.rdi();
    if s == 0 {
        return 0;
    }
    let v = [ctx.rsi() as u32, ctx.rdx() as u32, ctx.rcx() as u32, ctx.r8() as u32];
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    ctx.write_bytes(s + 0x18, &b);
    0
}

// ---------------------------------------------------------------------------
// Glyph objects
// ---------------------------------------------------------------------------

fn clamp_u16(v: f32) -> u16 {
    if v <= 0.0 {
        0
    } else {
        v.min(u16::MAX as f32).round() as u16
    }
}

/// sceFontGenerateCharGlyph(fontHandle, code, const FontGenerateGlyphDetail*, FontGlyph*)
fn cb_generate_char_glyph(ctx: &mut CpuContext) -> HleResult {
    let (h, code, detail, out) = (ctx.rdi(), ctx.rsi() as u32, ctx.rdx(), ctx.rcx());
    if out == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    ctx.write_u64(out, 0);
    if h == 0 {
        return ERROR_INVALID_FONT_HANDLE;
    }
    if code == 0 {
        return ERROR_NO_SUPPORT_CODE;
    }
    let Some(f) = font(h) else {
        return ERROR_INVALID_FONT_HANDLE;
    };
    let (mut id, mut options, mut glyph_form, mut metrics_form, mut mem) = (GENERATE_GLYPH_DETAIL_ID, 0u16, 0u8, 0u8, 0u64);
    if detail != 0 {
        let Some(d) = ctx.read_bytes(detail, 0x10) else {
            return ERROR_INVALID_PARAMETER;
        };
        id = u16::from_le_bytes([d[0], d[1]]);
        options = u16::from_le_bytes([d[4], d[5]]);
        glyph_form = d[6];
        metrics_form = d[7];
        mem = u64::from_le_bytes(d[8..16].try_into().unwrap());
    }
    if id != GENERATE_GLYPH_DETAIL_ID
        || options & !0x11 != 0
        || (glyph_form == 0 && metrics_form != 0)
        || (glyph_form != 0 && metrics_form == 0)
        || glyph_form > 1
        || metrics_form > 4
        || (mem != 0 && read_u16(ctx, mem) != Some(MEMORY_MAGIC))
    {
        return ERROR_INVALID_PARAMETER;
    }
    let m = match char_metrics(&f.face, code, &f.style) {
        Ok(m) => m,
        Err(e) => return e,
    };
    let Some(g) = galloc(ctx, GLYPH_SIZE) else {
        return ERROR_ALLOCATION_FAILED;
    };
    let (sw, sh) = f.style.pixels();
    let baseline = clamp_u16(m[3]);
    let mut b = [0u8; 0x20];
    b[0..2].copy_from_slice(&GLYPH_MAGIC.to_le_bytes());
    b[2..4].copy_from_slice(&options.to_le_bytes());
    b[4] = glyph_form;
    b[5] = metrics_form;
    b[6..8].copy_from_slice(&clamp_u16(sh).to_le_bytes());
    b[8..10].copy_from_slice(&baseline.to_le_bytes());
    b[10..12].copy_from_slice(&clamp_u16(m[1]).to_le_bytes());
    b[14..16].copy_from_slice(&baseline.to_le_bytes());
    b[16..20].copy_from_slice(&sw.to_le_bytes());
    b[20..24].copy_from_slice(&sh.to_le_bytes());
    b[24..32].copy_from_slice(&mem.to_le_bytes());
    ctx.write_bytes(g, &b);
    write_metrics(ctx, g + GLYPH_METRICS, &m);
    write_glyph_metric_variants(ctx, g, &m);
    state().glyphs.insert(g, Glyph { attribute: 0, owner: h, code, outline: 0 });
    ctx.write_u64(out, g);
    0
}

/// sceFontDeleteGlyph(const FontMemory*, FontGlyph*)
fn cb_delete_glyph(ctx: &mut CpuContext) -> HleResult {
    let p = ctx.rsi();
    if p == 0 {
        return ERROR_INVALID_PARAMETER;
    }
    let g = ctx.read_u64(p).unwrap_or(0);
    if g == 0 || read_u16(ctx, g) != Some(GLYPH_MAGIC) {
        return ERROR_INVALID_GLYPH;
    }
    let Some(glyph) = state().glyphs.remove(&g) else {
        return ERROR_INVALID_GLYPH;
    };
    if glyph.outline != 0 {
        gfree(glyph.outline);
    }
    write_u16(ctx, g, 0);
    gfree(g);
    ctx.write_u64(p, 0);
    0
}

/// sceFontGlyphDefineAttribute(FontGlyph, int attribute, int* oldAttribute)
fn cb_glyph_define_attribute(ctx: &mut CpuContext) -> HleResult {
    let (g, attribute, old) = (ctx.rdi(), ctx.rsi() as i32, ctx.rdx());
    let mut s = state();
    let Some(glyph) = s.glyphs.get_mut(&g) else {
        drop(s);
        if old != 0 {
            ctx.write_u32(old, 0);
        }
        return ERROR_INVALID_GLYPH;
    };
    let prev = glyph.attribute;
    glyph.attribute = attribute;
    drop(s);
    if old != 0 {
        ctx.write_u32(old, prev as u32);
    }
    0
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("CUKn5pX-NVY", "sceFontAttachDeviceCacheBuffer", cb_attach_device_cache_buffer),
    ("3OdRkSjOcog", "sceFontBindRenderer", cb_bind_renderer),
    ("vzHs3C8lWJk", "sceFontCloseFont", cb_close_font),
    ("nWrfPI4Okmg", "sceFontCreateLibrary", cb_create_library),
    ("8h-SOB-asgk", "sceFontDefineAttribute", cb_define_attribute),
    ("5kx49CAlO-M", "sceFontGetAttribute", cb_get_attribute),
    ("oO33Uex4Ui0", "sceFontGlyphGetAttribute", cb_glyph_get_attribute),
    ("FXP359ygujs", "sceFontDestroyLibrary", cb_destroy_library),
    ("L97d+3OgMlE", "sceFontGetCharGlyphMetrics", cb_get_char_glyph_metrics),
    ("sDuhHGNhHvE", "sceFontGetKerning", cb_get_kerning),
    ("h6hIgxXEiEc", "sceFontMemoryTerm", cb_memory_term),
    ("Z2cdsqJH+5k", "sceFontRebindRenderer", cb_rebind_renderer),
    ("3G4zhgKuxE8", "sceFontRenderCharGlyphImage", cb_render_char_glyph_image_horizontal),
    ("i6UNdSig1uE", "sceFontRenderCharGlyphImageVertical", cb_render_char_glyph_image_horizontal),
    ("vRxf4d0ulPs", "sceFontRenderSurfaceSetScissor", cb_render_surface_set_scissor),
    ("n590hj5Oe-k", "sceFontCreateLibraryWithEdition", cb_create_library_with_edition),
    ("WaSFJoRWXaI", "sceFontCreateRendererWithEdition", cb_create_renderer_with_edition),
    ("LHDoRWVFGqk", "sceFontDeleteGlyph", cb_delete_glyph),
    ("exAxkyVLt0s", "sceFontDestroyRenderer", cb_destroy_renderer),
    ("C-4Qw5Srlyw", "sceFontGenerateCharGlyph", cb_generate_char_glyph),
    ("imxVx8lm+KM", "sceFontGetHorizontalLayout", cb_get_horizontal_layout),
    ("IQtleGLL5pQ", "sceFontGetRenderCharGlyphMetrics", cb_get_render_char_glyph_metrics),
    ("3BrWWFU+4ts", "sceFontGetVerticalLayout", cb_get_vertical_layout),
    ("8-zmgsxkBek", "sceFontGlyphDefineAttribute", cb_glyph_define_attribute),
    ("whrS4oksXc4", "sceFontMemoryInit", cb_memory_init),
    ("JzCH3SCFnAU", "sceFontOpenFontInstance", cb_open_font_instance),
    ("KXUpebrFk1U", "sceFontOpenFontMemory", cb_open_font_memory),
    ("cKYtVmeSTcw", "sceFontOpenFontSet", cb_open_font_set),
    ("kAenWy1Zw5o", "sceFontRenderCharGlyphImageHorizontal", cb_render_char_glyph_image_horizontal),
    ("gdUCnU0gHdI", "sceFontRenderSurfaceInit", cb_render_surface_init),
    ("TMtqoFQjjbA", "sceFontSetEffectSlant", cb_set_effect_slant),
    ("v0phZwa4R5o", "sceFontSetEffectWeight", cb_set_effect_weight),
    ("N1EBMeGhf7E", "sceFontSetScalePixel", cb_set_scale_pixel),
    ("lz9y9UFO2UU", "sceFontSetupRenderEffectSlant", cb_setup_render_effect_slant),
    ("XIGorvLusDQ", "sceFontSetupRenderEffectWeight", cb_setup_render_effect_weight),
    ("6vGCkkQJOcI", "sceFontSetupRenderScalePixel", cb_setup_render_scale_pixel),
    ("mz2iTY0MK4A", "sceFontSupportExternalFonts", cb_support_external_fonts),
    ("SsRbbCiWoGw", "sceFontSupportSystemFonts", cb_support_system_fonts),
    ("1QjhKxrsOB8", "sceFontUnbindRenderer", cb_unbind_renderer),
    ("PxSR9UfJ+SQ", "sceFontSetScriptLanguage", cb_set_script_language),
    ("IrXeG0Lc6nA", "sceFontGetScriptLanguage", cb_get_script_language),
    ("SnsZua35ngs", "sceFontSetTypographicDesign", cb_set_typographic_design),
    ("7-miUT6pNQw", "sceFontGetTypographicDesign", cb_get_typographic_design),
];

const EXPORTS_FT: &[(&str, &str, HleCallback)] = &[
    ("oM+XCzVG3oM", "sceFontSelectLibraryFt", cb_select_library_ft),
    ("Xx974EW-QFY", "sceFontSelectRendererFt", cb_select_renderer_ft),
    // Font format modules of the FreeType library: each links its driver in
    // and returns 0.
    ("ZcQL0iSjvFw", "sceFontFtSupportBdf", cb_ft_support),
    ("LADHEyFTxRQ", "sceFontFtSupportCid", cb_ft_support),
    ("oakL15-mBtc", "sceFontFtSupportOpenType", cb_ft_support),
    ("dcQeaDr8UJc", "sceFontFtSupportOpenTypeOtf", cb_ft_support),
    ("2KXS-HkZT3c", "sceFontFtSupportOpenTypeTtf", cb_ft_support),
    ("H0mJnhKwV-s", "sceFontFtSupportPcf", cb_ft_support),
    ("S2mw3sYplAI", "sceFontFtSupportPfr", cb_ft_support),
    ("4BAhDLdrzUI", "sceFontFtSupportTrueType", cb_ft_support),
    ("nAfQ6qaL1fU", "sceFontFtSupportType1", cb_ft_support),
    ("X9+pzrGtBus", "sceFontFtSupportType42", cb_ft_support),
    ("w0hI3xsK-hc", "sceFontFtSupportWinFonts", cb_ft_support),
];

fn cb_ft_support(_ctx: &mut CpuContext) -> HleResult {
    0
}

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS.iter().chain(text::EXPORTS).chain(style::EXPORTS).chain(objects::EXPORTS) {
        reg.register_fn(nid, name, LIB, cb);
    }
    for &(nid, name, cb) in EXPORTS_FT {
        reg.register_fn(nid, name, LIB_FT, cb);
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod guest_tests;
