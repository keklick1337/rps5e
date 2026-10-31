// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Image operations: sample / gather / load / store / query / atomics on
//! the planned image and sampler bindings.

use super::emit::{Emitter, R, SCOPE_DEVICE};
use ps5shader::resources::{bindless, emulated_compare, image_binding, DescriptorBindingKind, ImageResource, ResourceUse, ImageDimension, ImageMipMode, ImageNumericClass, ImageResourceClass, ResourceBinding, ResourceOperand};
use ps5shader::ssa::*;
use rspirv::dr::Operand;
use rspirv::spirv::{Capability, Decoration, Dim, GlslStd450Op as GLOp, ImageFormat, ImageOperands, Op as S, StorageClass as SC, Word};

/// The SPIR-V view of one image binding.
#[derive(Clone, Copy)]
struct Binding {
    class: ImageResourceClass,
    numeric: ImageNumericClass,
    dimension: ImageDimension,
    cube: bool,
    /// Declared as a cube array (sampled cube bindings): coordinates become
    /// directions.
    cube_dim: bool,
    compare: bool,
    /// R64 unsigned view for 64-bit atomics.
    atomic64: bool,
    /// Image type and the binding variable.
    image_ty: Word,
    var: Word,
}

fn spatial(d: ImageDimension) -> u32 {
    match d {
        ImageDimension::Dim1D | ImageDimension::Dim1DArray => 1,
        ImageDimension::Dim3D => 3,
        _ => 2,
    }
}

fn arrayed(d: ImageDimension) -> bool {
    matches!(d, ImageDimension::Dim1DArray | ImageDimension::Dim2DArray | ImageDimension::Dim2DMsaaArray)
}

fn msaa(d: ImageDimension) -> bool {
    matches!(d, ImageDimension::Dim2DMsaa | ImageDimension::Dim2DMsaaArray)
}

impl Emitter<'_, '_> {
    fn image_binding(&mut self, kind: DescriptorBindingKind) -> R<Binding> {
        let number = kind.number();
        if let Some(&(image_ty, var)) = self.res.images.get(&number) {
            return Ok(Binding { image_ty, var, ..self.image_binding_shape(kind)? });
        }
        let count = self.binding_count(kind).ok_or("image binding missing from the layout")?;
        let b = self.image_binding_shape(kind)?;
        let n = self.cu(count);
        let arr = self.b.type_array(b.image_ty, n);
        let var = self.binding_var(arr, SC::UniformConstant, kind);
        self.res.images.insert(number, (b.image_ty, var));
        Ok(Binding { var, ..b })
    }

    /// The image array of the bindless set declared with the type of
    /// binding `kind` (every type aliases the one array binding; each slot
    /// holds a view of the type its table reads).
    fn bindless_binding(&mut self, kind: DescriptorBindingKind) -> R<Binding> {
        let number = kind.number();
        if let Some(&(image_ty, var)) = self.res.bindless_images.get(&number) {
            return Ok(Binding { image_ty, var, ..self.image_binding_shape(kind)? });
        }
        let b = self.image_binding_shape(kind)?;
        self.cap(Capability::RuntimeDescriptorArray);
        self.cap(Capability::ShaderNonUniform);
        self.cap(Capability::SampledImageArrayNonUniformIndexing);
        let arr = self.b.type_runtime_array(b.image_ty);
        let var = self.global(arr, SC::UniformConstant);
        self.decorate(var, Decoration::DescriptorSet, &[bindless::SET]);
        self.decorate(var, Decoration::Binding, &[bindless::IMAGE_BINDING]);
        self.res.bindless_images.insert(number, (b.image_ty, var));
        Ok(Binding { var, ..b })
    }

    /// The view of slot `slot` (a value that differs between waves) of a
    /// bindless image array.
    fn load_bindless_image(&mut self, b: &Binding, slot: Word) -> Word {
        let p = self.ptr(SC::UniformConstant, b.image_ty);
        let a = self.access(p, b.var, &[slot]);
        decorate_nonuniform(self, a);
        let image = self.load(b.image_ty, a);
        decorate_nonuniform(self, image);
        image
    }

    /// The shape and image type of binding `kind` (no variable: `var` is 0).
    fn image_binding_shape(&mut self, kind: DescriptorBindingKind) -> R<Binding> {
        let (class, numeric, dimension, atomic) = image_binding::describe(kind);
        let number = kind.number();
        let cube = image_binding::is_cube(kind);
        let cube_dim = cube && class == ImageResourceClass::Sampled;
        let compare = (43..=45).contains(&number);
        let atomic64 = image_binding::is_atomic64(kind);
        let sampled_ty = match numeric {
            _ if atomic64 && !self.feat.image_int64_atomics => return Err("64-bit image atomics without device support".into()),
            _ if atomic64 => {
                self.exts.insert("SPV_EXT_shader_image_int64");
                self.cap(Capability::Int64);
                self.cap(Capability::Int64Atomics);
                self.cap(Capability::Int64ImageEXT);
                self.ty(Type::U64)
            }
            ImageNumericClass::Float => self.t_f32(),
            ImageNumericClass::Uint => self.t_u32(),
            ImageNumericClass::Sint => self.t_i32(),
            _ => return Err(format!("image binding {kind:?} numeric class {numeric:?}")),
        };
        let dim = match dimension {
            ImageDimension::Dim1D | ImageDimension::Dim1DArray => {
                self.cap(if class == ImageResourceClass::Storage { Capability::Image1D } else { Capability::Sampled1D });
                Dim::Dim1D
            }
            ImageDimension::Dim3D => Dim::Dim3D,
            _ if cube_dim => {
                self.cap(Capability::SampledCubeArray);
                Dim::DimCube
            }
            _ => Dim::Dim2D,
        };
        let storage = class == ImageResourceClass::Storage;
        if storage {
            self.cap(Capability::StorageImageReadWithoutFormat);
            self.cap(Capability::StorageImageWriteWithoutFormat);
        }
        if msaa(dimension) && storage {
            self.cap(Capability::StorageImageMultisample);
        }
        let image_ty = self.b.type_image(
            sampled_ty,
            dim,
            compare as u32,
            arrayed(dimension) as u32,
            msaa(dimension) as u32,
            if storage { 2 } else { 1 },
            if atomic64 {
                ImageFormat::R64ui
            } else if atomic {
                ImageFormat::R32ui
            } else {
                ImageFormat::Unknown
            },
            None,
        );
        Ok(Binding { class, numeric, dimension, cube, cube_dim, compare, atomic64, image_ty, var: 0 })
    }

    fn sampler_var(&mut self) -> R<Word> {
        if let Some(v) = self.res.samplers {
            return Ok(v);
        }
        let count = self.binding_count(DescriptorBindingKind::Samplers).ok_or("sampler binding missing from the layout")?;
        let st = self.b.type_sampler();
        let n = self.cu(count);
        let arr = self.b.type_array(st, n);
        let var = self.binding_var(arr, SC::UniformConstant, DescriptorBindingKind::Samplers);
        self.res.samplers = Some(var);
        Ok(var)
    }

    fn load_image(&mut self, b: &Binding, element: u32) -> Word {
        let p = self.ptr(SC::UniformConstant, b.image_ty);
        let idx = self.cu(element);
        let a = self.access(p, b.var, &[idx]);
        self.load(b.image_ty, a)
    }

    /// Element `index` (a run-time value, wave-uniform) of an image binding.
    fn load_image_dynamic(&mut self, b: &Binding, index: Word) -> Word {
        let p = self.ptr(SC::UniformConstant, b.image_ty);
        let a = self.access(p, b.var, &[index]);
        self.load(b.image_ty, a)
    }

    fn load_sampler(&mut self, element: u32) -> R<Word> {
        let var = self.sampler_var()?;
        let st = self.b.type_sampler();
        let p = self.ptr(SC::UniformConstant, st);
        let idx = self.cu(element);
        let a = self.access(p, var, &[idx]);
        Ok(self.load(st, a))
    }
}

/// The bindless-set slot of an image selected through the host table map:
/// the key (the heap record the T# was read from: its dword-0 offset in the
/// heap V# minus the table's entry offset, over 32) indexes the map, whose
/// first word, key count and null slot the flattened table holds
/// ([`bindless::TABLE_WORDS`] at the image's mapping offset). A key past
/// the map reads the null slot, as a load past the heap V# reads a null T#.
fn bindless_slot(e: &mut Emitter, v: Value, image: u32, info: &ps5shader::resources::ImageResource) -> R<Word> {
    let request = e.input.request;
    let selector = request
        .plan
        .info
        .images
        .get(image as usize)
        .and_then(|i| request.plan.descriptor_sources.get(i.source as usize))
        .and_then(|s| s.indirect_image.as_ref())
        .ok_or("bindless image without a table selector")?;
    // A table, or a material key's heap record (`key << 5`).
    let entry_offset = match selector.table {
        Some(t) => t.entry_offset,
        None if !selector.dense && selector.wave_indexed.is_none() && selector.direct_candidates.is_none() => 0,
        None => return Err("bindless image without a table selector".into()),
    };
    let t_lo = e.f.args(v)[0];
    if e.f.op(t_lo) != Op::Construct {
        return Err("bindless image T# is not its table's record".into());
    }
    let load0 = e.f.args(t_lo)[0];
    if !matches!(e.f.op(load0), Op::ScalarBufferLoad { dword: 0, .. }) {
        return Err("bindless image T# is not its table's record".into());
    }
    let offset = e.arg(load0, 1)?;
    let tu = e.t_u32();
    let entry = e.cu(entry_offset);
    let rel = e.ins(S::ISub, tu, &[offset, entry]);
    let five = e.cu(5);
    let key = e.ins(S::ShiftRightLogical, tu, &[rel, five]);
    let word = |e: &mut Emitter, k: u32| {
        let index = e.cu(info.indirect_mapping_offset + k);
        e.word_buffer_load(DescriptorBindingKind::FlattenedResourceTable, index)
    };
    let first = word(e, 0)?;
    let count = word(e, 1)?;
    let null = word(e, 2)?;
    let tb = e.t_bool();
    let inside = e.ins(S::ULessThan, tb, &[key, count]);
    let zero = e.cu(0);
    let k = e.select(tu, inside, key, zero);
    let index = e.ins(S::IAdd, tu, &[first, k]);
    let slot = e.bindless_map_load(index);
    Ok(e.select(tu, inside, slot, null))
}

/// The binding element of a bindless-table image selected at run time: the
/// key (the heap record the T# was read from: its dword-0 offset in the heap
/// V# minus the table's entry offset, over 32) is looked up in the sorted
/// `[count, key, candidate, ...]` mapping of the flattened table, and the
/// candidate picks its element. None for an image with one candidate, or
/// when the candidates do not share the root's binding (the root is used).
fn table_element(e: &mut Emitter, v: Value, root: u32, kind: DescriptorBindingKind, root_element: u32) -> R<Option<Word>> {
    let request = e.input.request;
    let infos = &request.resources.info.images;
    let Some(info) = infos.get(root as usize) else { return Ok(None) };
    if info.indirect_search_iterations == 0 || info.indirect_root != root {
        return Ok(None);
    }
    let Some(plan_image) = request.plan.info.images.get(root as usize) else { return Ok(None) };
    let Some(table) = request.plan.descriptor_sources.get(plan_image.source as usize).and_then(|s| s.indirect_image.as_ref()).and_then(|s| s.table) else {
        return Ok(None);
    };
    // Candidate c: 0 is the root, c >= 1 the (c-1)-th appended image of it.
    let mut elements = vec![root_element];
    for (index, candidate) in infos.iter().enumerate().skip(request.plan.info.images.len()) {
        if candidate.indirect_root != root {
            continue;
        }
        if image_binding::for_image(candidate) != Some(kind) {
            return Ok(None);
        }
        let Some(el) = e.binding_element(kind, index as u32) else { return Ok(None) };
        elements.push(el);
    }
    if elements.len() < 2 {
        return Ok(None);
    }
    // The heap offset of T# dword 0: t_lo = construct(load0, ...).
    let t_lo = e.f.args(v)[0];
    if e.f.op(t_lo) != Op::Construct {
        return Ok(None);
    }
    let load0 = e.f.args(t_lo)[0];
    if !matches!(e.f.op(load0), Op::ScalarBufferLoad { dword: 0, .. }) {
        return Ok(None);
    }
    let offset = e.arg(load0, 1)?;
    let tu = e.t_u32();
    let entry = e.cu(table.entry_offset);
    let rel = e.ins(S::ISub, tu, &[offset, entry]);
    let five = e.cu(5);
    let key = e.ins(S::ShiftRightLogical, tu, &[rel, five]);
    // Binary search for the last mapped key <= key.
    let base = info.indirect_mapping_offset;
    let word = |e: &mut Emitter, index: Word| e.word_buffer_load(DescriptorBindingKind::FlattenedResourceTable, index);
    let count_at = e.cu(base);
    let count = word(e, count_at)?;
    let (mut lo, mut hi) = (e.cu(0), count);
    let one = e.cu(1);
    let two = e.cu(2);
    let key_base = e.cu(base + 1);
    let tb = e.t_bool();
    for _ in 0..info.indirect_search_iterations {
        let sum = e.ins(S::IAdd, tu, &[lo, hi]);
        let mid = e.ins(S::ShiftRightLogical, tu, &[sum, one]);
        let slot = e.ins(S::IMul, tu, &[mid, two]);
        let at = e.ins(S::IAdd, tu, &[key_base, slot]);
        let k = word(e, at)?;
        let le = e.ins(S::ULessThanEqual, tb, &[k, key]);
        lo = e.select(tu, le, mid, lo);
        hi = e.select(tu, le, hi, mid);
    }
    let slot = e.ins(S::IMul, tu, &[lo, two]);
    let at = e.ins(S::IAdd, tu, &[key_base, slot]);
    let found_key = word(e, at)?;
    let at1 = e.ins(S::IAdd, tu, &[at, one]);
    let candidate = word(e, at1)?;
    let found = e.ieq(found_key, key);
    // An unmapped key reads the root (its candidate 0).
    let zero = e.cu(0);
    let candidate = e.select(tu, found, candidate, zero);
    let mut element = e.cu(elements[0]);
    for (c, &el) in elements.iter().enumerate().skip(1) {
        let cc = e.cu(c as u32);
        let is = e.ieq(candidate, cc);
        let ew = e.cu(el);
        element = e.select(tu, is, ew, element);
    }
    Ok(Some(element))
}

/// Constant texel offsets become native ConstOffset operands unless
/// `RPS5E_CONST_TEXEL_OFFSET=0` (then they take the run-time path: an
/// Offset operand, or a coordinate shift on samples without one).
/// Part of the disk-cache key.
pub fn const_texel_offsets_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_CONST_TEXEL_OFFSET").is_ok_and(|v| v == "0"))
}

/// The constant `v` holds in the lanes `exec` keeps active: a constant, or
/// the active side of selects on `exec` (a vector write under EXEC).
fn active_constant(f: &Function, mut v: Value, exec: Value) -> Option<u32> {
    for _ in 0..8 {
        let d = f.value(v);
        match d.op {
            Op::Const(c) => return Some(c as u32),
            Op::Select if d.args[0] == exec => v = d.args[1],
            _ => return None,
        }
    }
    None
}

/// Address component layout of a MIMG instruction.
struct Address {
    words: Vec<Word>,
    a16: bool,
    /// 16-bit derivatives (`_g16` opcodes only).
    g16: bool,
    /// Components per derivative vector (0 without derivatives).
    deriv_dims: u32,
    /// Dwords of offset / bias / compare before the packable components.
    full: u32,
}

impl Address {
    /// The dword and the half (`None` for a 32-bit component) of packable
    /// component `k` (derivatives, coordinates, lod, clamp). Derivatives
    /// are 16-bit under G16 only (A16 leaves them 32-bit): each vector
    /// (d/dh, then d/dv) packs into its own dwords, an odd last component
    /// padded; the coordinates follow in the next dword, packed only under
    /// A16.
    fn position(&self, k: u32) -> (u32, Option<u32>) {
        let dd = self.deriv_dims;
        if self.g16 && k < 2 * dd {
            let per_vector = dd.div_ceil(2);
            let (vector, c) = (k / dd, k % dd);
            return (self.full + vector * per_vector + c / 2, Some(c % 2));
        }
        if k < 2 * dd {
            return (self.full + k, None);
        }
        let derivative_dwords = if self.g16 { 2 * dd.div_ceil(2) } else { 2 * dd };
        let (base, rest) = (self.full + derivative_dwords, k - 2 * dd);
        if self.a16 {
            (base + rest / 2, Some(rest % 2))
        } else {
            (base + rest, None)
        }
    }

    /// Packable component `k` as u32 bits (16-bit halves zero-extended).
    fn half(&self, e: &mut Emitter, k: u32) -> Word {
        let (dword, half) = self.position(k);
        let w = self.words.get(dword as usize).copied().unwrap_or_else(|| e.cu(0));
        match half {
            None => w,
            Some(h) => e.bits(w, h * 16, 16),
        }
    }

    fn full_word(&self, e: &mut Emitter, k: u32) -> Word {
        self.words.get(k as usize).copied().unwrap_or_else(|| e.cu(0))
    }

    /// Packable component `k` as f32 (from f16 when it is 16-bit).
    fn float(&self, e: &mut Emitter, k: u32) -> Word {
        let bits = self.half(e, k);
        let tf = e.t_f32();
        if self.position(k).1.is_some() {
            let v2 = e.t_fvec(2);
            let p = e.gl(v2, GLOp::UnpackHalf2x16, &[bits]);
            e.extract(tf, p, 0)
        } else {
            e.bitcast(tf, bits)
        }
    }
}

/// Coordinate components a guest MIMG DIM supplies.
fn guest_coords(dim: u8) -> u32 {
    match dim {
        0 => 1,
        1 => 2,
        2 | 3 => 3,
        4 => 2,
        5 | 6 => 3,
        _ => 4,
    }
}

fn guest_deriv_dims(dim: u8) -> u32 {
    match dim {
        0 | 4 => 1,
        2 => 3,
        _ => 2,
    }
}

/// The mip level a sample with a texel offset reads, for the offset
/// emulation without offset operands: an explicit LOD rounded to the
/// nearest level; in pixel shaders the implicit LOD of the quad (plus the
/// bias); level 0 elsewhere. Clamped to the image's levels. With linear
/// mip filtering the two blended levels share this one's offset.
#[allow(clippy::too_many_arguments)]
fn offset_level(e: &mut Emitter, image: Word, si: Word, coord: Word, sd: u32, n: u32, pixel: bool, ops: ImageOperands, operands: &[Word]) -> Word {
    let tf = e.t_f32();
    let ti = e.t_i32();
    let lod = if ops.contains(ImageOperands::LOD) {
        Some(operands[0])
    } else if pixel {
        // The query takes the spatial coordinates only (no array layer).
        let spatial = if n == sd {
            coord
        } else if sd == 1 {
            e.extract(tf, coord, 0)
        } else {
            let c: Vec<Word> = (0..sd).map(|k| e.extract(tf, coord, k)).collect();
            let t = e.t_fvec(sd);
            e.construct(t, &c)
        };
        let v2 = e.t_fvec(2);
        let q = e.ins(S::ImageQueryLod, v2, &[si, spatial]);
        let x = e.extract(tf, q, 0);
        Some(if ops.contains(ImageOperands::BIAS) { e.ins(S::FAdd, tf, &[x, operands[0]]) } else { x })
    } else {
        None
    };
    let Some(lod) = lod else { return e.ci(0) };
    let half = e.cf(0.5);
    let r = e.ins(S::FAdd, tf, &[lod, half]);
    let fl = e.gl(tf, GLOp::Floor, &[r]);
    let li = e.ins(S::ConvertFToS, ti, &[fl]);
    let levels = e.ins(S::ImageQueryLevels, ti, &[image]);
    let one = e.ci(1);
    let last = e.ins(S::ISub, ti, &[levels, one]);
    let zero = e.ci(0);
    e.gl(ti, GLOp::SClamp, &[li, zero, last])
}

pub fn emit(e: &mut Emitter, v: Value, d: &ValueData, attr: ImageAttr) -> R<Option<Word>> {
    if let ImageOp::BvhIntersect { .. } = attr.op {
        return super::bvh::emit(e, v, d, attr);
    }
    let uses = e.uses(v).cloned().unwrap_or_default();
    let image_res = uses.iter().find(|u| u.operand == ResourceOperand::Resource).map(|u| u.binding);
    let Some(ResourceBinding::Image(i)) = image_res else {
        return Err(format!("image operation without a planned image ({image_res:?})"));
    };
    let info = e.input.request.resources.info.images.get(i as usize).cloned().ok_or("image specialization missing")?;
    let kind = image_binding::for_image(&info).ok_or_else(|| format!("no binding class for image {i} ({:?} {:?} {:?})", info.resource_class, info.numeric_class, info.dimension))?;
    let (element, b) = if info.bindless {
        if !e.feat.bindless_tables {
            return Err("an image selected through the host table map needs descriptor indexing (runtime arrays, non-uniform indexing, partially bound update-after-bind sampled images)".into());
        }
        (0, e.bindless_binding(kind)?)
    } else {
        let element = e.binding_element(kind, i).ok_or("image absent from its binding")?;
        (element, e.image_binding(kind)?)
    };
    let n = d.args.len();
    let exec = e.arg(v, n - 1)?;
    let words: Vec<Word> = (3..3 + attr.address_dwords as usize).map(|k| e.arg(v, k)).collect::<R<_>>()?;
    let data: Vec<Word> = (3 + attr.address_dwords as usize..3 + attr.address_dwords as usize + attr.data_dwords as usize)
        .map(|k| e.arg(v, k))
        .collect::<R<_>>()?;
    let f = attr.flags;
    let sample_like = matches!(attr.op, ImageOp::Sample | ImageOp::Gather4 | ImageOp::GetLod);
    let full = if sample_like { f.offset as u32 + f.bias as u32 + f.compare as u32 } else { 0 };
    let deriv_dims = if sample_like && f.derivatives { guest_deriv_dims(attr.dim) } else { 0 };
    let addr = Address { words, a16: attr.a16, g16: f.g16, deriv_dims, full };
    if info.bindless {
        return emit_bindless(e, v, d, attr, &uses, i, &info, b, exec, &addr, &data);
    }
    let image = match table_element(e, v, i, kind, element)? {
        Some(dynamic) => e.load_image_dynamic(&b, dynamic),
        None => e.load_image(&b, element),
    };
    emit_op(e, v, d, attr, &uses, &info, element, None, b, image, exec, &addr, &data, None)
}

/// An image instruction on an image selected through the host table map:
/// the map word of its key holds the slot and the class of the view in it,
/// and the instruction runs on the view of that class (an integer T#
/// samples through the point copy of its sampler).
#[allow(clippy::too_many_arguments)]
fn emit_bindless(
    e: &mut Emitter,
    v: Value,
    d: &ValueData,
    attr: ImageAttr,
    uses: &[ResourceUse],
    image: u32,
    info: &ImageResource,
    b: Binding,
    exec: Word,
    addr: &Address,
    data: &[Word],
) -> R<Option<Word>> {
    let word = bindless_slot(e, v, image, info)?;
    let tu = e.t_u32();
    let mask = e.cu(bindless::SLOT_MASK);
    let slot = e.ins(S::BitwiseAnd, tu, &[word, mask]);
    decorate_nonuniform(e, slot);
    // The slot indexes the shared array directly: an atomic needs a
    // pointer into it (a load is not enough), so it travels to
    // [`emit_op`] beside the constant element the direct path uses.
    let dyn_slot = Some(slot);
    // A depth reference reads depth views only.
    if info.depth_compare {
        let view = e.load_bindless_image(&b, slot);
        return emit_op(e, v, d, attr, uses, info, 0, dyn_slot, b, view, exec, addr, data, None);
    }
    let shift = e.cu(bindless::MODE_SHIFT);
    let mode = e.ins(S::ShiftRightLogical, tu, &[word, shift]);
    let sampler = uses.iter().find_map(|u| match (u.operand, u.binding) {
        (ResourceOperand::Sampler, ResourceBinding::Sampler(s)) => Some(s),
        _ => None,
    });
    let point = sampler.and_then(|s| e.input.request.resources.info.sampled_pairs.iter().find(|p| p.image == image && p.sampler == s).and_then(|p| p.point_sampler));
    let ty = e.ty(d.ty);
    let mut failure = None;
    let r = e.switch_value(mode, bindless::MODES, ty, |e, k| {
        let numeric_class = match k {
            bindless::MODE_UINT => ImageNumericClass::Uint,
            bindless::MODE_SINT => ImageNumericClass::Sint,
            _ => ImageNumericClass::Float,
        };
        let class = ImageResource { numeric_class, ..info.clone() };
        let r = (|| {
            let kind = image_binding::for_image(&class).ok_or("no binding for a bindless view class")?;
            let b = e.bindless_binding(kind)?;
            let view = e.load_bindless_image(&b, slot);
            let point = if k == bindless::MODE_FLOAT { None } else { point };
            if k != bindless::MODE_FLOAT && sampler.is_some() && point.is_none() {
                return Err("integer bindless view without a point sampler".to_string());
            }
            emit_op(e, v, d, attr, uses, &class, 0, dyn_slot, b, view, exec, addr, data, point)?.ok_or_else(|| "bindless image instruction without a value".to_string())
        })();
        r.unwrap_or_else(|m| {
            failure = Some(m);
            e.undef(d.ty)
        })
    });
    match failure {
        Some(m) => Err(m),
        None => Ok(Some(r)),
    }
}

/// The operation of an image instruction on `image` (of binding `b`;
/// `point_sampler`: the sampler a sample uses instead of its own).
/// `dyn_slot`: the bindless array index of the selected view (an atomic
/// builds its texel pointer on it; `element` is the constant index of the
/// direct path).
#[allow(clippy::too_many_arguments)]
fn emit_op(
    e: &mut Emitter,
    v: Value,
    d: &ValueData,
    attr: ImageAttr,
    uses: &[ResourceUse],
    info: &ImageResource,
    element: u32,
    dyn_slot: Option<Word>,
    b: Binding,
    image: Word,
    exec: Word,
    addr: &Address,
    data: &[Word],
    point_sampler: Option<u32>,
) -> R<Option<Word>> {
    let n = d.args.len();
    let f = attr.flags;
    let tu = e.t_u32();
    let swizzle = info.shader_swizzle;
    match attr.op {
        ImageOp::Sample | ImageOp::Gather4 | ImageOp::GetLod => {
            // A converted integer image samples (point) and gathers its raw
            // words, unpacked after; a converted UNORM one has no filtered
            // host view.
            if attr.op == ImageOp::Gather4 && f.packed {
                return packed_horizontal_gather(e, addr, attr, info, &b, image, d.ty);
            }
            let conv = packed_conversion(info);
            // 11_11_10 UNORM samples only as a direct 2D image_sample_lz or
            // image_sample with 32-bit addresses and data, filtered in the
            // shader from level 0 (the implicit LOD is not modelled).
            let unorm_lz = attr.op == ImageOp::Sample
                && (f == SampleFlags { lod_zero: true, ..SampleFlags::default() } || f == SampleFlags::default())
                && attr.dim == 1
                && b.dimension == ImageDimension::Dim2D
                && !attr.a16
                && !attr.d16
                && !info.bindless
                && info.indirect_root == ps5shader::resources::NO_INDEX;
            if conv.as_ref().is_some_and(|c| attr.op == ImageOp::GetLod || (!matches!(c.kind, ps5shader::interp::buffer::FormatKind::Uint) && !unorm_lz)) {
                return Err("sampling, gathering or querying the LOD of a converted 11_11_10 UNORM image other than a direct 2D image_sample(_lz)".into());
            }
            let sampler_res = uses.iter().find(|u| u.operand == ResourceOperand::Sampler).map(|u| u.binding);
            let Some(ResourceBinding::Sampler(s)) = sampler_res else {
                return Err("sampled image operation without a planned sampler".into());
            };
            let se = e.binding_element(DescriptorBindingKind::Samplers, point_sampler.unwrap_or(s)).ok_or("sampler absent from its binding")?;
            let sampler = e.load_sampler(se)?;
            let sit = e.b.type_sampled_image(b.image_ty);
            let si = e.ins(S::SampledImage, sit, &[image, sampler]);
            if info.bindless {
                decorate_nonuniform(e, si);
            }
            // Packable component order: derivatives, coordinates, lod, clamp.
            let dd = guest_deriv_dims(attr.dim);
            let deriv = if f.derivatives { 2 * dd } else { 0 };
            let gc = guest_coords(attr.dim);
            let coord_first = deriv;
            let mut coord = if b.cube_dim { cube::coord(e, &addr, coord_first, attr.dim) } else { sample_coord(e, &addr, &b, coord_first, gc)? };
            if let Some(conv) = conv.as_ref().filter(|c| c.kind == ps5shader::interp::buffer::FormatKind::Unorm) {
                return Ok(Some(sample_packed_unorm_lz(e, conv, attr, si, image, coord, swizzle, d.ty)));
            }
            if attr.op == ImageOp::GetLod {
                // Implicit LODs need derivatives: pixel quads, or compute
                // quads; elsewhere the LOD reads as 0. GFX10 forms a compute
                // quad from four consecutive lanes: a 2-wide grid of whole
                // waves (two lanes per invocation, split waves) or, on the
                // group's own grid, linear derivative groups. Without those,
                // 2x2 quads of an even grid stand in for them.
                let compute = e.stage == ps5shader::resources::ShaderStage::Compute && e.mesh.is_none();
                let s = e.input.request.local_size;
                let total = s[0].max(1) * s[1].max(1) * s[2].max(1);
                let whole_waves = e.lanes == 2 || e.split.is_some();
                let compute_linear = compute && !whole_waves && e.feat.compute_derivatives_linear && total % 4 == 0;
                let compute_quads = compute && !compute_linear && e.feat.compute_derivatives && (whole_waves || (s[0] % 2 == 0 && s[1] % 2 == 0));
                if e.stage != ps5shader::resources::ShaderStage::Pixel && !compute_quads && !compute_linear {
                    let z = e.cu(0);
                    return Ok(Some(pack_result(e, &[z, z, z, z], attr, d.ty, false)));
                }
                if compute_linear {
                    e.exts.insert("SPV_NV_compute_shader_derivatives");
                    e.cap(Capability::ComputeDerivativeGroupLinearNV);
                    e.compute_derivatives_linear = true;
                }
                if compute_quads {
                    e.exts.insert("SPV_NV_compute_shader_derivatives");
                    e.cap(Capability::ComputeDerivativeGroupQuadsNV);
                    e.compute_derivatives = true;
                }
                e.cap(Capability::ImageQuery);
                let v2 = e.t_fvec(2);
                // A cube's LOD query takes the direction alone.
                let lod_coord = if b.cube_dim {
                    let v3 = e.t_fvec(3);
                    let tf = e.t_f32();
                    let c: Vec<Word> = (0..3).map(|k| e.extract(tf, coord, k)).collect();
                    e.construct(v3, &c)
                } else {
                    coord
                };
                let q = e.ins(S::ImageQueryLod, v2, &[si, lod_coord]);
                let tf = e.t_f32();
                let comps: Vec<Word> = (0..4).map(|k| {
                    if k < 2 {
                        let x = e.extract(tf, q, k);
                        e.bitcast(tu, x)
                    } else {
                        e.cu(0)
                    }
                }).collect();
                return Ok(Some(pack_result(e, &comps, attr, d.ty, false)));
            }
            if f.compare && !b.compare && info.emulated_compare & emulated_compare::ENABLED != 0 && attr.op == ImageOp::Sample {
                let k = f.offset as u32 + f.bias as u32;
                let w = addr.full_word(e, k);
                let tf = e.t_f32();
                let dref = e.bitcast(tf, w);
                let offset = f.offset.then(|| addr.full_word(e, 0));
                let r = compare_emulated(e, image, &b, coord, dref, offset, info.emulated_compare, f.lod_zero || f.lod || f.derivatives, f.lod || f.derivatives)?;
                let x = e.bitcast(tu, r);
                let z = e.cu(0);
                let texel = apply_swizzle(e, &[x, z, z, z], swizzle, b.numeric);
                return Ok(Some(pack_result(e, &texel, attr, d.ty, false)));
            }
            let mut ops = ImageOperands::NONE;
            let mut operands: Vec<Word> = Vec::new();
            let pixel = e.stage == ps5shader::resources::ShaderStage::Pixel;
            let explicit = f.lod || f.lod_zero || f.derivatives || !pixel;
            // Order of image operand words: Bias, Lod, Grad(dx, dy), Offset, MinLod.
            if f.bias && pixel && !f.lod && !f.lod_zero && !f.derivatives && attr.op == ImageOp::Sample {
                ops |= ImageOperands::BIAS;
                let k = f.offset as u32;
                let w = addr.full_word(e, k);
                let tf = e.t_f32();
                operands.push(if attr.a16 { let v2 = e.t_fvec(2); let p = e.gl(v2, GLOp::UnpackHalf2x16, &[w]); e.extract(tf, p, 0) } else { e.bitcast(tf, w) });
            }
            if explicit && attr.op == ImageOp::Sample {
                if f.derivatives && b.cube_dim {
                    // The LOD of the face-space gradients (the host projects
                    // direction gradients, which differ near face edges).
                    ops |= ImageOperands::LOD;
                    operands.push(cube::gradient_lod(e, &addr, image));
                } else if f.derivatives {
                    ops |= ImageOperands::GRAD;
                    let sd = spatial(b.dimension).max(1);
                    let mut dx = Vec::new();
                    let mut dy = Vec::new();
                    for c in 0..sd {
                        dx.push(if c < dd { addr.float(e, c) } else { e.cf(0.0) });
                        dy.push(if c < dd { addr.float(e, dd + c) } else { e.cf(0.0) });
                    }
                    let gt = if sd == 1 { e.t_f32() } else { e.t_fvec(sd) };
                    let gx = if sd == 1 { dx[0] } else { e.construct(gt, &dx) };
                    let gy = if sd == 1 { dy[0] } else { e.construct(gt, &dy) };
                    operands.push(gx);
                    operands.push(gy);
                } else {
                    ops |= ImageOperands::LOD;
                    operands.push(if f.lod { addr.float(e, deriv + gc) } else { e.cf(0.0) });
                }
            }
            // Cube sampling has no texel offsets.
            let constant_offset = if f.offset && !b.cube_dim && const_texel_offsets_enabled() { active_constant(e.f, d.args[3], d.args[n - 1]) } else { None };
            if let Some(bits) = constant_offset {
                // A constant offset word: native ConstOffset operands
                // (signed six-bit offsets in the low bits of each byte).
                let sd = spatial(b.dimension);
                let comps: Vec<Word> = (0..sd).map(|c| e.ci(((((bits >> (c * 8)) & 0x3F) ^ 0x20) as i32) - 0x20)).collect();
                let off = if sd == 1 {
                    comps[0]
                } else {
                    let ot = e.t_ivec(sd);
                    e.b.constant_composite(ot, comps)
                };
                ops |= ImageOperands::CONST_OFFSET;
                operands.push(off);
            } else if f.offset && !b.cube_dim {
                let w = addr.full_word(e, 0);
                let sd = spatial(b.dimension);
                let ti = e.t_i32();
                let comps: Vec<Word> = (0..sd).map(|c| {
                    let bits = e.cu(6);
                    let o = e.cu(c * 8);
                    let s = e.ins(S::BitFieldSExtract, tu, &[w, o, bits]);
                    e.bitcast(ti, s)
                }).collect();
                let ot = e.t_ivec(sd);
                let off = if sd == 1 { comps[0] } else { e.construct(ot, &comps) };
                if attr.op == ImageOp::Sample && !e.feat.sample_offset {
                    // Without offset operands on samples: shift the
                    // coordinates by offset / size of the level sampled
                    // (the offset counts texels of that level).
                    e.cap(Capability::ImageQuery);
                    let n = sd + arrayed(b.dimension) as u32;
                    let st = if sd == 1 { e.t_i32() } else { e.t_ivec(sd) };
                    let qt = if n == 1 { e.t_i32() } else { e.t_ivec(n) };
                    let level = offset_level(e, image, si, coord, sd, n, pixel, ops, &operands);
                    let size = e.ins(S::ImageQuerySizeLod, qt, &[image, level]);
                    let tf = e.t_f32();
                    let mut c = Vec::new();
                    for k in 0..n {
                        let x = if n == 1 { coord } else { e.extract(tf, coord, k) };
                        if k < sd {
                            let ti = e.t_i32();
                            let sz = if n == 1 { size } else { e.extract(ti, size, k) };
                            let szf = e.ins(S::ConvertSToF, tf, &[sz]);
                            let of = e.ins(S::ConvertSToF, tf, &[comps[k as usize]]);
                            let d = e.ins(S::FDiv, tf, &[of, szf]);
                            c.push(e.ins(S::FAdd, tf, &[x, d]));
                        } else {
                            c.push(x);
                        }
                    }
                    let _ = st;
                    coord = if n == 1 { c[0] } else { let t = e.t_fvec(n); e.construct(t, &c) };
                } else {
                    ops |= ImageOperands::OFFSET;
                    e.cap(Capability::ImageGatherExtended);
                    operands.push(off);
                }
            }
            // LOD clamp: a minimum LOD for implicit and gradient sampling, an
            // explicit LOD is raised to it.
            if f.clamp && attr.op == ImageOp::Sample {
                let k = deriv + gc + f.lod as u32;
                let clamp = addr.float(e, k);
                if ops.contains(ImageOperands::LOD) {
                    let tf = e.t_f32();
                    operands[0] = e.gl(tf, GLOp::FMax, &[operands[0], clamp]);
                } else {
                    e.cap(Capability::MinLod);
                    ops |= ImageOperands::MIN_LOD;
                    operands.push(clamp);
                }
            }
            let mut ins_ops = vec![Operand::IdRef(si), Operand::IdRef(coord)];
            let dref = if f.compare {
                let k = f.offset as u32 + f.bias as u32;
                let w = addr.full_word(e, k);
                let tf = e.t_f32();
                Some(e.bitcast(tf, w))
            } else {
                None
            };
            if let Some(dr) = dref {
                if !b.compare {
                    return Err("depth-compare sample on a non-compare binding".into());
                }
                ins_ops.push(Operand::IdRef(dr));
            }
            let component_class = if dref.is_some() { ImageNumericClass::Float } else { b.numeric };
            let st = match component_class {
                ImageNumericClass::Uint => e.t_u32(),
                ImageNumericClass::Sint => e.t_i32(),
                _ => e.t_f32(),
            };
            let vt = match component_class {
                ImageNumericClass::Uint => e.t_uvec(4),
                ImageNumericClass::Sint => e.t_ivec(4),
                _ => e.t_fvec(4),
            };
            if attr.op == ImageOp::Gather4 && matches!(b.dimension, ImageDimension::Dim1D | ImageDimension::Dim1DArray) {
                // Gathers need a 2D image: a 1D one is read as its two
                // neighbouring texels (one row), sampled at their centres.
                if dref.is_some() {
                    return Err("depth-compare gather of a 1D image".into());
                }
                if f.horizontal {
                    return Err("horizontal gather of a 1D image".into());
                }
                let comp = (attr.dmask as u32).trailing_zeros().min(3);
                let texels = gather_1d(e, si, image, coord, &b, &addr, f.offset, component_class)?;
                let comps: Vec<Word> = [0usize, 1, 1, 0]
                    .iter()
                    .map(|&t| {
                        let x = e.extract(st, texels[t], comp);
                        if component_class == ImageNumericClass::Uint { x } else { e.bitcast(tu, x) }
                    })
                    .collect();
                return Ok(Some(pack_gather(e, &comps, attr, d.ty, component_class != ImageNumericClass::Float)));
            }
            if attr.op == ImageOp::Gather4 {
                // A converted image gathers its words (component 0).
                let comp = if conv.is_some() { 0 } else { (attr.dmask as u32).trailing_zeros().min(3) };
                let op = if dref.is_some() { S::ImageDrefGather } else { S::ImageGather };
                if dref.is_none() {
                    let c = e.cu(comp);
                    ins_ops.push(Operand::IdRef(c));
                }
                if f.horizontal {
                    // Four texels of one row: the footprint's left texel
                    // shifted by -1, 0, 1 and 2 texels.
                    if !matches!(b.dimension, ImageDimension::Dim2D | ImageDimension::Dim2DArray) || b.cube_dim {
                        return Err(format!("horizontal gather of a {:?} image", b.dimension));
                    }
                    e.cap(Capability::ImageGatherExtended);
                    let iv2 = e.t_ivec(2);
                    let zero = e.ci(0);
                    let offsets: Vec<Word> = (-1..=2)
                        .map(|x| {
                            let x = e.ci(x);
                            e.b.constant_composite(iv2, [x, zero])
                        })
                        .collect();
                    let four = e.cu(4);
                    let array = e.b.type_array(iv2, four);
                    let offsets = e.b.constant_composite(array, offsets);
                    ins_ops.push(Operand::ImageOperands(ImageOperands::CONST_OFFSETS));
                    ins_ops.push(Operand::IdRef(offsets));
                } else if ops != ImageOperands::NONE {
                    ins_ops.push(Operand::ImageOperands(ops & !(ImageOperands::LOD | ImageOperands::BIAS | ImageOperands::GRAD | ImageOperands::MIN_LOD)));
                    // Only the offset survives for gathers.
                    if ops.intersects(ImageOperands::OFFSET | ImageOperands::CONST_OFFSET) {
                        ins_ops.push(Operand::IdRef(*operands.last().unwrap()));
                    } else {
                        ins_ops.pop();
                    }
                }
                // An explicit LOD selects the mip on the GPU: one view per
                // level, the gather reads level 0 of the selected view.
                let r = if info.mip_mode == ImageMipMode::Dynamic && info.mip_count > 1 && f.lod {
                    let mip = gather_mip(e, v, &addr, deriv + gc, info.mip_count)?;
                    e.switch_value(mip, info.mip_count, vt, |e, k| {
                        let img = e.load_image(&b, element + k);
                        let si = e.ins(S::SampledImage, sit, &[img, sampler]);
                        let mut ops = ins_ops.clone();
                        ops[0] = Operand::IdRef(si);
                        e.ins_ops(op, vt, ops)
                    })
                } else {
                    e.ins_ops(op, vt, ins_ops)
                };
                let comps: Vec<Word> = (0..4).map(|k| {
                    let x = e.extract(st, r, k);
                    if component_class == ImageNumericClass::Uint { x } else { e.bitcast(tu, x) }
                }).collect();
                if let Some(conv) = &conv {
                    // Each texel's field the gathered component selects
                    // (constants for 0 / 1 selectors).
                    let gathered = (attr.dmask as u32).trailing_zeros().min(3);
                    let selector = (swizzle >> (gathered * 3)) & 7;
                    let comps: Vec<Word> = comps
                        .iter()
                        .map(|&w| match selector {
                            4..=7 => super::memory::convert_load(e, conv, (selector - 4) % conv.count, w),
                            1 => e.cu(1),
                            _ => e.cu(0),
                        })
                        .collect();
                    return Ok(Some(pack_gather(e, &comps, attr, d.ty, true)));
                }
                return Ok(Some(pack_gather(e, &comps, attr, d.ty, component_class != ImageNumericClass::Float)));
            }
            let op = match (dref.is_some(), explicit) {
                (false, false) => S::ImageSampleImplicitLod,
                (false, true) => S::ImageSampleExplicitLod,
                (true, false) => S::ImageSampleDrefImplicitLod,
                (true, true) => S::ImageSampleDrefExplicitLod,
            };
            if ops != ImageOperands::NONE {
                ins_ops.push(Operand::ImageOperands(ops));
                ins_ops.extend(operands.iter().map(|&w| Operand::IdRef(w)));
            }
            let rt = if dref.is_some() { e.t_f32() } else { vt };
            let r = e.ins_ops(op, rt, ins_ops);
            let texel: Vec<Word> = if dref.is_some() {
                let x = e.bitcast(tu, r);
                let z = e.cu(0);
                vec![x, z, z, z]
            } else {
                (0..4).map(|k| {
                    let x = e.extract(st, r, k);
                    if component_class == ImageNumericClass::Uint { x } else { e.bitcast(tu, x) }
                }).collect()
            };
            if let Some(conv) = &conv {
                let fields: Vec<Word> = (0..4).map(|k| super::memory::convert_load(e, conv, k % conv.count, texel[0])).collect();
                let texel = apply_swizzle(e, &fields, swizzle, ImageNumericClass::Uint);
                return Ok(Some(pack_result(e, &texel, attr, d.ty, true)));
            }
            let texel = apply_swizzle(e, &texel, swizzle, b.numeric);
            Ok(Some(pack_result(e, &texel, attr, d.ty, component_class != ImageNumericClass::Float)))
        }
        ImageOp::Load | ImageOp::Store | ImageOp::Atomic(_) => {
            if b.cube_dim {
                return Err("texel access to a cube-sampled binding".into());
            }
            let coord = int_coord(e, &addr, &b, attr.dim)?;
            let gc = guest_coords(attr.dim);
            let storage = b.class == ImageResourceClass::Storage;
            match attr.op {
                ImageOp::Load => {
                    if attr.flags.elements != 0 {
                        return by_read(e, addr, attr, info, &b, image, swizzle, d.ty);
                    }
                    let st = match b.numeric {
                        ImageNumericClass::Uint => e.t_u32(),
                        ImageNumericClass::Sint => e.t_i32(),
                        _ => e.t_f32(),
                    };
                    let vt = match b.numeric {
                        ImageNumericClass::Uint => e.t_uvec(4),
                        ImageNumericClass::Sint => e.t_ivec(4),
                        _ => e.t_fvec(4),
                    };
                    let inside = texel_address_inside(e, addr, &b, attr, info, image, true);
                    let fetch = |e: &mut Emitter| if storage && attr.flags.mip && info.mip_mode == ImageMipMode::Dynamic && !msaa(b.dimension) {
                        // One view per level, as for stores: the load reads
                        // the selected one.
                        let mip = addr.half(e, gc);
                        e.switch_value(mip, info.mip_count, vt, |e, k| {
                            let image = e.load_image(&b, element + k);
                            e.ins_ops(S::ImageRead, vt, vec![Operand::IdRef(image), Operand::IdRef(coord)])
                        })
                    } else if storage {
                        let mut ops = vec![Operand::IdRef(image), Operand::IdRef(coord)];
                        if msaa(b.dimension) {
                            let s = addr.half(e, gc - 1);
                            ops.push(Operand::ImageOperands(ImageOperands::SAMPLE));
                            ops.push(Operand::IdRef(s));
                        }
                        e.ins_ops(S::ImageRead, vt, ops)
                    } else {
                        let mut ops = vec![Operand::IdRef(image), Operand::IdRef(coord)];
                        if msaa(b.dimension) {
                            let s = addr.half(e, gc - 1);
                            ops.push(Operand::ImageOperands(ImageOperands::SAMPLE));
                            ops.push(Operand::IdRef(s));
                        } else {
                            let lod = if attr.flags.mip { addr.half(e, gc) } else { e.cu(0) };
                            ops.push(Operand::ImageOperands(ImageOperands::LOD));
                            ops.push(Operand::IdRef(lod));
                        }
                        e.ins_ops(S::ImageFetch, vt, ops)
                    };
                    let r = match inside {
                        Some(c) => {
                            let zero = e.b.constant_null(vt);
                            e.if_value(c, vt, zero, fetch)
                        }
                        None => fetch(e),
                    };
                    let texel: Vec<Word> = (0..4).map(|k| {
                        let x = e.extract(st, r, k);
                        if b.numeric == ImageNumericClass::Uint { x } else { e.bitcast(tu, x) }
                    }).collect();
                    if attr.flags.packed {
                        // The texel's raw bits as dwords (no DST_SEL).
                        if swizzle != ps5shader::resources::IDENTITY_IMAGE_SWIZZLE {
                            return Err("packed texel loads need an identity swizzle".into());
                        }
                        let f = raw_texel_format(info)?;
                        let words = packed_texel(e, &f, &texel, attr.flags.packed_signed)?;
                        return Ok(Some(pack_result(e, &words, attr, d.ty, true)));
                    }
                    if let Some(conv) = sint_storage(info, &b) {
                        // Each signed field sign-extended from its width
                        // (the unsigned view zero-extends it).
                        let mut fields = texel.clone();
                        for c in 0..conv.count {
                            fields[c as usize] = super::memory::convert_load(e, &conv, c, texel[c as usize]);
                        }
                        let texel = apply_swizzle(e, &fields, swizzle, ImageNumericClass::Sint);
                        return Ok(Some(pack_result(e, &texel, attr, d.ty, true)));
                    }
                    if let Some(conv) = packed_conversion(info) {
                        // The fields of the packed word, converted; a
                        // component the format lacks repeats them.
                        if attr.d16 {
                            return Err("16-bit data from a converted image".into());
                        }
                        let fields: Vec<Word> = (0..4).map(|k| super::memory::convert_load(e, &conv, k % conv.count, texel[0])).collect();
                        let class = if matches!(conv.kind, ps5shader::interp::buffer::FormatKind::Uint | ps5shader::interp::buffer::FormatKind::Sint) { ImageNumericClass::Uint } else { ImageNumericClass::Float };
                        let texel = apply_swizzle(e, &fields, swizzle, class);
                        return Ok(Some(pack_result(e, &texel, attr, d.ty, class != ImageNumericClass::Float)));
                    }
                    let texel = apply_swizzle(e, &texel, swizzle, b.numeric);
                    Ok(Some(pack_result(e, &texel, attr, d.ty, b.numeric != ImageNumericClass::Float)))
                }
                ImageOp::Store => {
                    if !storage {
                        return Err("image store to a sampled binding".into());
                    }
                    // Data: dmask components in order.
                    let mut comps = Vec::new();
                    let mut k = 0;
                    for c in 0..4u32 {
                        if attr.dmask >> c & 1 != 0 {
                            let w = if attr.d16 {
                                let word = data.get((k / 2) as usize).copied().unwrap_or_else(|| e.cu(0));
                                if b.numeric == ImageNumericClass::Float {
                                    let v2 = e.t_fvec(2);
                                    let p = e.gl(v2, GLOp::UnpackHalf2x16, &[word]);
                                    let tf = e.t_f32();
                                    let x = e.extract(tf, p, k % 2);
                                    e.bitcast(tu, x)
                                } else {
                                    e.bits(word, (k % 2) * 16, 16)
                                }
                            } else {
                                data.get(k as usize).copied().unwrap_or_else(|| e.cu(0))
                            };
                            comps.push(w);
                            k += 1;
                        } else {
                            comps.push(e.cu(0));
                        }
                    }
                    if attr.flags.packed {
                        // Raw texel dwords (DMASK-selected, 0 otherwise), no
                        // DST_SEL: the view's components of those bits.
                        let f = raw_texel_format(info)?;
                        comps = packed_store_components(e, &f, &comps, b.numeric)?;
                    }
                    // The T# destination swizzle applies inversely: memory
                    // component c takes the data component whose selector
                    // is c (the lowest one on duplicates); unselected
                    // components store 0.
                    if swizzle != ps5shader::resources::IDENTITY_IMAGE_SWIZZLE && !attr.flags.packed {
                        let zero = e.cu(0);
                        let mut phys = [zero; 4];
                        for src in (0..4u32).rev() {
                            let s = (swizzle >> (src * 3)) & 7;
                            if s >= 4 {
                                phys[(s - 4) as usize] = comps[src as usize];
                            }
                        }
                        comps = phys.to_vec();
                    }
                    if let Some(conv) = sint_storage(info, &b).filter(|_| !attr.flags.packed) {
                        // Each field saturated to its signed width and
                        // masked (D16 halves sign-extended first).
                        for c in 0..conv.count {
                            let v = comps[c as usize];
                            comps[c as usize] = if attr.d16 {
                                super::convert::store_component_d16(e, &conv, c, v)
                            } else {
                                super::convert::store_component(e, &conv, c, v)
                            };
                        }
                    }
                    if let Some(conv) = packed_conversion(info) {
                        // Each field converted and packed into the word the
                        // view holds.
                        if attr.d16 {
                            return Err("16-bit data to a converted image".into());
                        }
                        let mut packed = e.cu(0);
                        for c in 0..conv.count {
                            let field = super::convert::store_component(e, &conv, c, comps[c as usize]);
                            let at = e.cu(conv.bit_offset[c as usize]);
                            let shifted = e.shl(field, at);
                            packed = e.bor(packed, shifted);
                        }
                        let zero = e.cu(0);
                        comps = vec![packed, zero, zero, zero];
                    }
                    let (st, vt) = match b.numeric {
                        ImageNumericClass::Uint => (e.t_u32(), e.t_uvec(4)),
                        ImageNumericClass::Sint => (e.t_i32(), e.t_ivec(4)),
                        _ => (e.t_f32(), e.t_fvec(4)),
                    };
                    let mut typed: Vec<Word> = comps.iter().map(|&c| if b.numeric == ImageNumericClass::Uint { c } else { e.bitcast(st, c) }).collect();
                    if b.numeric == ImageNumericClass::Float && info.store_quantize != 0 && !attr.flags.packed && packed_conversion(info).is_none() && store_quantize_enabled() {
                        for c in typed.iter_mut() {
                            *c = quantize_store_component(e, *c, info.store_quantize);
                        }
                    }
                    let texel = e.construct(vt, &typed);
                    let ms = msaa(b.dimension).then(|| addr.half(e, gc - 1));
                    let exec = match texel_address_inside(e, addr, &b, attr, info, image, false) {
                        Some(c) => e.land(exec, c),
                        None => exec,
                    };
                    let write = |e: &mut Emitter, image: Word| {
                        let mut ops = vec![Operand::IdRef(image), Operand::IdRef(coord), Operand::IdRef(texel)];
                        if let Some(s) = ms {
                            ops.push(Operand::ImageOperands(ImageOperands::SAMPLE));
                            ops.push(Operand::IdRef(s));
                        }
                        e.void(S::ImageWrite, ops);
                    };
                    if attr.flags.mip && info.mip_mode == ImageMipMode::Dynamic {
                        // One view per level: the store writes the selected
                        // one; a mip past the view writes nothing.
                        let mip = addr.half(e, gc);
                        e.if_then(exec, |e| {
                            e.switch_then(mip, info.mip_count, |e, k| {
                                let image = e.load_image(&b, element + k);
                                write(e, image);
                            })
                        });
                    } else {
                        e.if_then(exec, |e| write(e, image));
                    }
                    Ok(None)
                }
                ImageOp::Atomic(op) => {
                    if !storage || b.numeric != ImageNumericClass::Uint {
                        return Err("image atomic on a non-uint storage binding".into());
                    }
                    // Texel pointers need the image variable element itself:
                    // the bindless slot the map word selected (a value that
                    // differs between waves), else the constant element.
                    let idx = match dyn_slot {
                        Some(slot) => slot,
                        None => e.cu(element),
                    };
                    let ip = e.ptr(SC::UniformConstant, b.image_ty);
                    let img_ptr = e.access(ip, b.var, &[idx]);
                    if dyn_slot.is_some() {
                        decorate_nonuniform(e, img_ptr);
                    }
                    let zero = e.cu(0);
                    if b.atomic64 {
                        // Operands are dword pairs; the old value returns
                        // as one.
                        let t64 = e.ty(Type::U64);
                        let p = e.ptr(SC::Image, t64);
                        let texel = e.ins(S::ImageTexelPointer, p, &[img_ptr, coord, zero]);
                        let word = |k: usize| data.get(k).copied().unwrap_or(zero);
                        let (lo, hi) = (word(0), word(1));
                        let val = e.join64(lo, hi);
                        let cmp = if data.len() >= 4 {
                            let (lo, hi) = (word(2), word(3));
                            Some(e.join64(lo, hi))
                        } else {
                            None
                        };
                        let z64 = e.c64(0);
                        let r = e.if_value(exec, t64, z64, |e| super::memory::atomic64(e, texel, op, val, cmp, SCOPE_DEVICE));
                        let (lo, hi) = e.split64(r);
                        let t2 = e.t_uvec(2);
                        return Ok(Some(e.construct(t2, &[lo, hi])));
                    }
                    let p = e.ptr(SC::Image, tu);
                    let texel = e.ins(S::ImageTexelPointer, p, &[img_ptr, coord, zero]);
                    let val = data.first().copied().unwrap_or(zero);
                    let cmp = data.get(1).copied();
                    let r = e.if_value(exec, tu, zero, |e| super::memory::atomic(e, texel, op, val, cmp, SCOPE_DEVICE));
                    Ok(Some(r))
                }
                _ => unreachable!(),
            }
        }
        ImageOp::GetResinfo => {
            e.cap(Capability::ImageQuery);
            let sd = spatial(b.dimension) + arrayed(b.dimension) as u32;
            let st = if sd == 1 { e.t_u32() } else { e.t_uvec(sd) };
            let storage = b.class == ImageResourceClass::Storage;
            let size = if storage || msaa(b.dimension) {
                e.ins(S::ImageQuerySize, st, &[image])
            } else {
                let lod = addr.full_word(e, 0);
                e.ins(S::ImageQuerySizeLod, st, &[image, lod])
            };
            // A storage view holds the T#'s base level only (or one view per
            // level, the first being the base): the level the mip operand
            // names is that size shifted down, at least 1, and the levels
            // are the views'.
            let storage_mips = storage && !msaa(b.dimension) && super::storage_resinfo_mips_enabled();
            let levels = if storage_mips && info.mip_mode == ImageMipMode::Dynamic {
                e.cu(info.mip_count.max(1))
            } else if storage || msaa(b.dimension) {
                e.cu(1)
            } else {
                e.ins(S::ImageQueryLevels, tu, &[image])
            };
            let mut comps: Vec<Word> = (0..3).map(|k| if k < sd { if sd == 1 { size } else { e.extract(tu, size, k) } } else { e.cu(1) }).collect();
            if storage_mips {
                // Shifts of 32 or more are undefined in SPIR-V; 31 leaves 1.
                let lod = addr.full_word(e, 0);
                let max_shift = e.cu(31);
                let lod = e.umin(lod, max_shift);
                let one = e.cu(1);
                for c in comps.iter_mut().take(spatial(b.dimension) as usize) {
                    let shifted = e.shr(*c, lod);
                    *c = e.gl(tu, GLOp::UMax, &[shifted, one]);
                }
            }
            if b.cube && !b.cube_dim && sd == 3 {
                let six = e.cu(6);
                comps[2] = e.ins(S::UDiv, tu, &[comps[2], six]);
            }
            comps.push(levels);
            Ok(Some(pack_result(e, &comps, attr, d.ty, false)))
        }
        ImageOp::BvhIntersect { .. } => unreachable!(),
    }
}

/// Floating coordinates of a sample for the binding's dimension.
fn sample_coord(e: &mut Emitter, a: &Address, b: &Binding, first: u32, guest: u32) -> R<Word> {
    let sd = spatial(b.dimension);
    let n = sd + arrayed(b.dimension) as u32;
    let tf = e.t_f32();
    let mut c = Vec::new();
    for k in 0..n {
        let x = if k < guest { a.float(e, first + k) } else { e.cf(0.0) };
        c.push(x);
    }
    if b.cube && n == 3 {
        // Face coordinates arrive in [1, 2]; faces of cube arrays are
        // spaced by 8 in the slice coordinate.
        let one = e.cf(1.0);
        c[0] = e.ins(S::FSub, tf, &[c[0], one]);
        c[1] = e.ins(S::FSub, tf, &[c[1], one]);
        let tu = e.t_u32();
        let face = e.ins(S::ConvertFToU, tu, &[c[2]]);
        let three = e.cu(3);
        let one_u = e.cu(1);
        let hi = e.shr(face, three);
        let pad = e.shl(hi, one_u);
        let host = e.isub(face, pad);
        c[2] = e.ins(S::ConvertUToF, tf, &[host]);
    }
    let _ = tf;
    if n == 1 {
        Ok(c[0])
    } else {
        let t = e.t_fvec(n);
        Ok(e.construct(t, &c))
    }
}

/// A depth-compare sample of a colour image (no host depth view): fetch
/// the red channel of the footprint texels of level 0 and compare each
/// against the reference with the guest function, filtering the results
/// (percentage-closer) bilinearly when the sampler is linear. An explicit
/// LOD or gradients of a single-level view select that level (the sampler's
/// magnification and minification filters agree).
#[allow(clippy::too_many_arguments)]
fn compare_emulated(e: &mut Emitter, image: Word, b: &Binding, coord: Word, dref: Word, offset: Option<Word>, state: u32, explicit_level: bool, gradients: bool) -> R<Word> {
    use emulated_compare::*;
    if gradients && state & SINGLE_LEVEL == 0 {
        return Err("compare against a mipmapped colour image with gradients or an explicit LOD".into());
    }
    if !explicit_level && state & SINGLE_LEVEL == 0 {
        return Err("compare against a colour image across mip levels".into());
    }
    // A line (1D view) is addressed by u alone: v is ignored, the row is 0.
    let (layered, line) = match b.dimension {
        ImageDimension::Dim2D => (false, false),
        ImageDimension::Dim2DArray => (true, false),
        ImageDimension::Dim1D => (false, true),
        ImageDimension::Dim1DArray => (true, true),
        _ => return Err("compare against a colour image that is not a 1D, 2D or array view of them".into()),
    };
    if reference(state) == REFERENCE_ZERO {
        let _ = (image, coord, dref, offset, layered);
        return Ok(e.cf(0.0));
    }
    let (tf, ti, tu, tb) = (e.t_f32(), e.t_i32(), e.t_u32(), e.t_bool());
    e.cap(Capability::ImageQuery);
    let spatial_n = if line { 1 } else { 2 };
    let n = spatial_n + layered as u32;
    let qt = if n == 1 { ti } else { e.t_ivec(n) };
    let zero_i = e.ci(0);
    let size = e.ins(S::ImageQuerySizeLod, qt, &[image, zero_i]);
    let width = if n == 1 { size } else { e.extract(ti, size, 0) };
    let height = if line { e.ci(1) } else { e.extract(ti, size, 1) };
    let layer = if layered {
        let layers = e.extract(ti, size, spatial_n);
        // The layer is the slice coordinate rounded half to even, as the
        // host sampler selects it: floor(z + 0.5) picks layer 1 for
        // 0.49999997 (the sum rounds to 1.0) and rounds exact halves up.
        let z = e.extract(tf, coord, spatial_n);
        let zf = e.gl(tf, GLOp::RoundEven, &[z]);
        let zi = e.ins(S::ConvertFToS, ti, &[zf]);
        let one = e.ci(1);
        let last = e.ins(S::ISub, ti, &[layers, one]);
        Some(e.gl(ti, GLOp::SClamp, &[zi, zero_i, last]))
    } else {
        None
    };
    let dref = match reference(state) {
        REFERENCE_UNORM => {
            let (lo, hi) = (e.cf(0.0), e.cf(1.0));
            e.gl(tf, GLOp::FClamp, &[dref, lo, hi])
        }
        REFERENCE_SNORM => {
            let (lo, hi) = (e.cf(-1.0), e.cf(1.0));
            e.gl(tf, GLOp::FClamp, &[dref, lo, hi])
        }
        _ => dref,
    };
    let (one_f, zero_f) = (e.cf(1.0), e.cf(0.0));
    let border_red = if state & BORDER_WHITE != 0 { one_f } else { zero_f };
    let wrap = |e: &mut Emitter, i: Word, extent: Word, mode: u32| {
        if mode == ADDRESS_WRAP {
            e.ins(S::SMod, ti, &[i, extent])
        } else {
            let one = e.ci(1);
            let last = e.ins(S::ISub, ti, &[extent, one]);
            let z = e.ci(0);
            e.gl(ti, GLOp::SClamp, &[i, z, last])
        }
    };
    let outside = |e: &mut Emitter, i: Word, extent: Word| {
        let z = e.ci(0);
        let below = e.ins(S::SLessThan, tb, &[i, z]);
        let above = e.ins(S::SGreaterThanEqual, tb, &[i, extent]);
        e.ins(S::LogicalOr, tb, &[below, above])
    };
    let vt = e.t_fvec(4);
    let compare_texel = |e: &mut Emitter, x: Word, y: Word| -> Word {
        let (mx, my) = (address_x(state), address_y(state));
        let ux = wrap(e, x, width, mx);
        let mut c = vec![ux];
        if !line {
            c.push(wrap(e, y, height, my));
        }
        c.extend(layer);
        let cc = if c.len() == 1 {
            c[0]
        } else {
            let ct = e.t_ivec(c.len() as u32);
            e.construct(ct, &c)
        };
        let lod = e.ci(0);
        let texel = e.ins_ops(S::ImageFetch, vt, vec![Operand::IdRef(image), Operand::IdRef(cc), Operand::ImageOperands(ImageOperands::LOD), Operand::IdRef(lod)]);
        let mut red = e.extract(tf, texel, 0);
        let mut border = None;
        if mx == ADDRESS_BORDER {
            border = Some(outside(e, x, width));
        }
        if my == ADDRESS_BORDER && !line {
            let oy = outside(e, y, height);
            border = Some(match border {
                Some(bx) => e.ins(S::LogicalOr, tb, &[bx, oy]),
                None => oy,
            });
        }
        if let Some(bd) = border {
            red = e.select(tf, bd, border_red, red);
        }
        let op = match function(state) {
            0 => return zero_f,
            1 => S::FOrdLessThan,
            2 => S::FOrdEqual,
            3 => S::FOrdLessThanEqual,
            4 => S::FOrdGreaterThan,
            // NOT_EQUAL passes when either side is NaN (IEEE `!=`).
            5 => S::FUnordNotEqual,
            6 => S::FOrdGreaterThanEqual,
            _ => return one_f,
        };
        let pass = e.ins(op, tb, &[dref, red]);
        e.select(tf, pass, one_f, zero_f)
    };
    let u = if n == 1 { coord } else { e.extract(tf, coord, 0) };
    let v = if line { e.cf(0.0) } else { e.extract(tf, coord, 1) };
    let wf = e.ins(S::ConvertSToF, tf, &[width]);
    let hf = e.ins(S::ConvertSToF, tf, &[height]);
    let su = e.ins(S::FMul, tf, &[u, wf]);
    let sv = e.ins(S::FMul, tf, &[v, hf]);
    let (ox, oy) = match offset {
        Some(w) => {
            let bits = e.cu(6);
            let (o0, o8) = (e.cu(0), e.cu(8));
            let x = e.ins(S::BitFieldSExtract, tu, &[w, o0, bits]);
            let y = e.ins(S::BitFieldSExtract, tu, &[w, o8, bits]);
            (e.bitcast(ti, x), e.bitcast(ti, y))
        }
        None => (zero_i, zero_i),
    };
    if state & LINEAR == 0 {
        let fu = e.gl(tf, GLOp::Floor, &[su]);
        let fv = e.gl(tf, GLOp::Floor, &[sv]);
        let iu = e.ins(S::ConvertFToS, ti, &[fu]);
        let iv = e.ins(S::ConvertFToS, ti, &[fv]);
        let x = e.ins(S::IAdd, ti, &[iu, ox]);
        let y = e.ins(S::IAdd, ti, &[iv, oy]);
        return Ok(compare_texel(e, x, y));
    }
    let half = e.cf(0.5);
    let cu_ = e.ins(S::FSub, tf, &[su, half]);
    let cv = e.ins(S::FSub, tf, &[sv, half]);
    let fu = e.gl(tf, GLOp::Floor, &[cu_]);
    let fv = e.gl(tf, GLOp::Floor, &[cv]);
    let wu = e.ins(S::FSub, tf, &[cu_, fu]);
    let wv = e.ins(S::FSub, tf, &[cv, fv]);
    let iu = e.ins(S::ConvertFToS, ti, &[fu]);
    let iv = e.ins(S::ConvertFToS, ti, &[fv]);
    let x0 = e.ins(S::IAdd, ti, &[iu, ox]);
    let y0 = e.ins(S::IAdd, ti, &[iv, oy]);
    let one = e.ci(1);
    let x1 = e.ins(S::IAdd, ti, &[x0, one]);
    let y1 = e.ins(S::IAdd, ti, &[y0, one]);
    let c00 = compare_texel(e, x0, y0);
    let c10 = compare_texel(e, x1, y0);
    if line {
        return Ok(e.gl(tf, GLOp::FMix, &[c00, c10, wu]));
    }
    let c01 = compare_texel(e, x0, y1);
    let c11 = compare_texel(e, x1, y1);
    let top = e.gl(tf, GLOp::FMix, &[c00, c10, wu]);
    let bottom = e.gl(tf, GLOp::FMix, &[c01, c11, wu]);
    Ok(e.gl(tf, GLOp::FMix, &[top, bottom, wv]))
}

/// Integer coordinates of a load / store.
fn int_coord(e: &mut Emitter, a: &Address, b: &Binding, dim: u8) -> R<Word> {
    let sd = spatial(b.dimension);
    let n = sd + arrayed(b.dimension) as u32;
    let guest = guest_coords(dim) - msaa(b.dimension) as u32;
    let ti = e.t_i32();
    let mut c = Vec::new();
    for k in 0..n {
        let x = if k < guest { a.half(e, k) } else { e.cu(0) };
        c.push(e.bitcast(ti, x));
    }
    if n == 1 {
        Ok(c[0])
    } else {
        let t = e.t_ivec(n);
        Ok(e.construct(t, &c))
    }
}

/// Where a load (`load`) or store whose MIMG DIM addresses more than the
/// bound view reaches memory (None: everywhere), as measured: a 3D load of
/// a 1D or 2D T# reads 0 unless z is 0 (y of a 1D T# is ignored); a
/// 2D-array store through a 1D T# writes texel x only at slice 0 (y is
/// ignored); a 2D-array load or store of a 3D T# (the slice addresses the
/// depth) reads 0 and writes nothing past the depth of the level.
fn texel_address_inside(e: &mut Emitter, a: &Address, b: &Binding, attr: ImageAttr, info: &ImageResource, image: Word, load: bool) -> Option<Word> {
    const DIM_3D: u8 = 2;
    const DIM_2D_ARRAY: u8 = 5;
    if attr.dim == DIM_3D && load && matches!(b.dimension, ImageDimension::Dim1D | ImageDimension::Dim2D) {
        let z = a.half(e, 2);
        let zero = e.cu(0);
        return Some(e.ieq(z, zero));
    }
    if attr.dim == DIM_2D_ARRAY && !load && b.dimension == ImageDimension::Dim1D {
        let slice = a.half(e, 2);
        let zero = e.cu(0);
        return Some(e.ieq(slice, zero));
    }
    if attr.dim != DIM_2D_ARRAY || b.dimension != ImageDimension::Dim3D {
        return None;
    }
    e.cap(Capability::ImageQuery);
    let tu = e.t_u32();
    let v3 = e.t_uvec(3);
    let gc = guest_coords(attr.dim);
    let depth = if b.class == ImageResourceClass::Storage {
        let size = e.ins(S::ImageQuerySize, v3, &[image]);
        let depth = e.extract(tu, size, 2);
        if attr.flags.mip && info.mip_mode == ImageMipMode::Dynamic {
            // The view of the selected level: the base depth shifted down.
            let mip = a.half(e, gc);
            let max_shift = e.cu(31);
            let mip = e.umin(mip, max_shift);
            let shifted = e.shr(depth, mip);
            let one = e.cu(1);
            e.gl(tu, GLOp::UMax, &[shifted, one])
        } else {
            depth
        }
    } else {
        let lod = if attr.flags.mip { a.half(e, gc) } else { e.cu(0) };
        let size = e.ins(S::ImageQuerySizeLod, v3, &[image, lod]);
        e.extract(tu, size, 2)
    };
    let slice = a.half(e, 2);
    Some(e.ult(slice, depth))
}

/// The packed format of a converted image (11_11_10 UNORM / UINT): its
/// texels are one 32-bit word the shader unpacks on loads and packs on
/// stores.
/// `RPS5E_STORE_QUANTIZE=0` leaves the float-to-UNORM/SNORM conversion of
/// storage image writes to the host.
fn store_quantize_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_STORE_QUANTIZE").is_ok_and(|v| v == "0"))
}

/// The scale of a [`ImageResource::store_quantize`] code and the smallest
/// f32 reciprocal whose product with it is at least 1 (so the host's
/// conversion of `level * reciprocal` gives `level` back whether it rounds
/// or truncates).
fn quantize_scale(code: u32) -> (f32, f32) {
    let bits = code & 0xFF;
    let snorm = code & 0x100 != 0;
    let scale = ((1u32 << if snorm { bits - 1 } else { bits }) - 1) as f32;
    let mut reciprocal = 1.0 / scale;
    while (reciprocal as f64) * (scale as f64) < 1.0 {
        reciprocal = reciprocal.next_up();
    }
    (scale, reciprocal)
}

/// A float component stored to a UNORM / SNORM storage image, as hardware
/// converts it: NaN is 0, the value is clamped to [0, 1] ([-1, 1] for
/// SNORM) and takes the level nearest to the exact product with the scale,
/// ties to even. The f32 product is exact up to its rounding error, taken
/// from a Dekker two-product, which decides a product rounded onto a
/// midpoint. Returns `level / scale`, which the host converts back exactly.
fn quantize_store_component(e: &mut Emitter, v: Word, code: u32) -> Word {
    let (scale, reciprocal) = quantize_scale(code);
    let tf = e.t_f32();
    let tb = e.t_bool();
    let precise = |e: &mut Emitter, op: S, a: Word, b: Word| {
        let r = e.ins(op, tf, &[a, b]);
        e.decorate(r, Decoration::NoContraction, &[]);
        r
    };
    let zero = e.cf(0.0);
    let one = e.cf(1.0);
    let lower = e.cf(if code & 0x100 != 0 { -1.0 } else { 0.0 });
    let nan = e.ins(S::IsNan, tb, &[v]);
    let finite = e.select(tf, nan, zero, v);
    let clamped = e.gl(tf, GLOp::FClamp, &[finite, lower, one]);
    let s = e.cf(scale);
    let product = precise(e, S::FMul, clamped, s);
    let split = e.cf(4097.0);
    let high = |e: &mut Emitter, x: Word| {
        let spread = precise(e, S::FMul, x, split);
        let d = precise(e, S::FSub, spread, x);
        precise(e, S::FSub, spread, d)
    };
    let ch = high(e, clamped);
    let sh = high(e, s);
    let cl = precise(e, S::FSub, clamped, ch);
    let sl = precise(e, S::FSub, s, sh);
    let hh = precise(e, S::FMul, ch, sh);
    let mut error = precise(e, S::FSub, hh, product);
    for (a, b) in [(ch, sl), (cl, sh), (cl, sl)] {
        let p = precise(e, S::FMul, a, b);
        error = precise(e, S::FAdd, error, p);
    }
    let below = e.gl(tf, GLOp::Floor, &[product]);
    let fraction = precise(e, S::FSub, product, below);
    let half = e.cf(0.5);
    let tie = e.ins(S::FOrdEqual, tb, &[fraction, half]);
    let above = precise(e, S::FAdd, below, one);
    let even = e.gl(tf, GLOp::RoundEven, &[product]);
    let positive = e.ins(S::FOrdGreaterThan, tb, &[error, zero]);
    let negative = e.ins(S::FOrdLessThan, tb, &[error, zero]);
    let low_or_even = e.select(tf, negative, below, even);
    let broken = e.select(tf, positive, above, low_or_even);
    let level = e.select(tf, tie, broken, even);
    let r = e.cf(reciprocal);
    precise(e, S::FMul, level, r)
}

fn packed_conversion(info: &ImageResource) -> Option<ps5shader::interp::buffer::FormatInfo> {
    use ps5shader::interp::buffer::{FormatInfo, FormatKind};
    use ps5shader::resources::format::image_format;
    // X and Y 11 bits from bit 0, Z 10 bits from bit 22.
    let kind = match info.conversion_format {
        image_format::FORMAT_11X2X10_UNORM => FormatKind::Unorm,
        image_format::FORMAT_11X2X10_UINT => FormatKind::Uint,
        _ => return None,
    };
    Some(FormatInfo { kind, count: 3, bits: [11, 11, 10, 0], byte_offset: [0; 4], bit_offset: [0, 11, 22, 0], packed: true })
}

/// The layout of the T# format of a raw-texel access (MIMG PCK, BY forms):
/// components in order, X in the lowest bits.
fn raw_texel_format(info: &ImageResource) -> R<ps5shader::interp::buffer::FormatInfo> {
    ps5shader::interp::buffer::FormatInfo::new(info.packed_format)
        .filter(|f| !f.packed)
        .ok_or_else(|| format!("accesses raw texels of image format {} (no component layout)", info.packed_format))
}

fn texel_bits(f: &ps5shader::interp::buffer::FormatInfo) -> u32 {
    (0..f.count as usize).map(|c| f.bits[c]).sum()
}

/// The raw bits of a texel read through its typed view (`texel`: each
/// component's bits), X lowest, as four dwords; `signed` sign-extends them
/// from the texel's width. Exact for integer components, UNORM up to 16
/// bits (`round(v * max)`) and 32-bit floats.
fn packed_texel(e: &mut Emitter, f: &ps5shader::interp::buffer::FormatInfo, texel: &[Word], signed: bool) -> R<Vec<Word>> {
    use ps5shader::interp::buffer::FormatKind as K;
    let (tu, tf) = (e.t_u32(), e.t_f32());
    let zero = e.cu(0);
    let mut words = vec![zero; 4];
    let mut position = 0u32;
    for c in 0..f.count as usize {
        let bits = f.bits[c];
        let exact = matches!(f.kind, K::Uint | K::Sint) || (f.kind == K::Unorm && bits <= 16) || (f.kind == K::Float && bits == 32);
        if !exact {
            return Err("reads packed texels of a format whose bits are not recoverable from the view".into());
        }
        let mut value = if f.kind == K::Unorm {
            let x = e.bitcast(tf, texel[c]);
            let max = e.cf(((1u32 << bits) - 1) as f32);
            let scaled = e.ins(S::FMul, tf, &[x, max]);
            let half = e.cf(0.5);
            let rounded = e.ins(S::FAdd, tf, &[scaled, half]);
            e.ins(S::ConvertFToU, tu, &[rounded])
        } else {
            texel[c]
        };
        if bits < 32 {
            let mask = e.cu((1u32 << bits) - 1);
            value = e.band(value, mask);
        }
        if position % 32 != 0 {
            let at = e.cu(position % 32);
            value = e.shl(value, at);
        }
        let w = (position / 32) as usize;
        words[w] = e.bor(words[w], value);
        position += bits;
    }
    if signed && position < 32 {
        let (o, n) = (e.cu(0), e.cu(position));
        words[0] = e.ins(S::BitFieldSExtract, tu, &[words[0], o, n]);
    }
    Ok(words)
}

/// The view's component bits a raw-texel store of `words` (each dword of
/// the texel, X lowest) writes; exact only where the view reproduces the
/// bits (unsigned components through an unsigned view, 32-bit signed
/// through an unsigned one, 32-bit floats through a float one).
fn packed_store_components(e: &mut Emitter, f: &ps5shader::interp::buffer::FormatInfo, words: &[Word], view: ImageNumericClass) -> R<Vec<Word>> {
    use ps5shader::interp::buffer::FormatKind as K;
    if f.count == 3 && f.bits[0] == 32 {
        return Err("stores packed texels of a format the hardware does not write".into());
    }
    let zero = e.cu(0);
    let mut values = vec![zero; 4];
    let mut offset = 0u32;
    for c in 0..f.count as usize {
        let bits = f.bits[c];
        let exact = (f.kind == K::Uint && view == ImageNumericClass::Uint)
            || (bits == 32 && ((f.kind == K::Sint && view == ImageNumericClass::Uint) || (f.kind == K::Float && view == ImageNumericClass::Float)));
        if !exact || offset % 32 + bits > 32 {
            return Err("stores packed texels of a format whose bits are not reproducible through the view".into());
        }
        let mut value = words[(offset / 32) as usize];
        if offset % 32 != 0 {
            let at = e.cu(offset % 32);
            value = e.shr(value, at);
        }
        if bits < 32 {
            let mask = e.cu((1u32 << bits) - 1);
            value = e.band(value, mask);
        }
        values[c] = value;
        offset += bits;
    }
    Ok(values)
}

/// `image_load(_mip)_by2/_by4` and `_pck2/_pck4`: `elements` consecutive
/// texels of one row from `x & !(elements - 1)`, all zeros unless the
/// level exists, `x` is inside it with `elements` texels left, and `y` is
/// inside; BY returns each texel's components in turn, PCK2 / PCK4 their
/// raw bits packed into one dword.
#[allow(clippy::too_many_arguments)]
fn by_read(e: &mut Emitter, a: &Address, attr: ImageAttr, info: &ImageResource, b: &Binding, image: Word, swizzle: u32, ty: Type) -> R<Option<Word>> {
    let elements = attr.flags.elements as u32;
    if b.class != ImageResourceClass::Sampled || b.dimension != ImageDimension::Dim2D {
        return Err("MIMG BY2/BY4 or PCK2/PCK4 load outside the measured 2D sampled-image subset".into());
    }
    let f = raw_texel_format(info)?;
    let bits = texel_bits(&f);
    let dwords = (attr.dmask as u32 & 0xF).count_ones();
    let components = if attr.flags.packed { 1 } else { dwords / elements };
    let fits = if attr.flags.packed { bits * elements <= 32 } else { components == f.count && components * elements == dwords && bits * elements <= 32 };
    if !fits || swizzle != ps5shader::resources::IDENTITY_IMAGE_SWIZZLE {
        return Err("MIMG BY2/BY4 or PCK2/PCK4 load whose elements fill one dword (identity swizzle) as measured".into());
    }
    e.cap(Capability::ImageQuery);
    let (tu, tb) = (e.t_u32(), e.t_bool());
    let x = a.half(e, 0);
    let y = a.half(e, 1);
    let lod = if attr.flags.mip { a.half(e, 2) } else { e.cu(0) };
    let levels = e.ins(S::ImageQueryLevels, tu, &[image]);
    let lod_inside = e.ult(lod, levels);
    let zero = e.cu(0);
    let query_lod = e.select(tu, lod_inside, lod, zero);
    let v2 = e.t_uvec(2);
    let size = e.ins(S::ImageQuerySizeLod, v2, &[image, zero]);
    let one = e.cu(1);
    let extent = |e: &mut Emitter, axis: u32| {
        let base = e.extract(tu, size, axis);
        let shifted = e.shr(base, query_lod);
        e.gl(tu, GLOp::UMax, &[shifted, one])
    };
    let width = extent(e, 0);
    let height = extent(e, 1);
    let x_inside = e.ult(x, width);
    let left = e.isub(width, x);
    let n = e.cu(elements);
    let room = e.ule(n, left);
    let y_inside = e.ult(y, height);
    let mut inside = e.land(lod_inside, x_inside);
    inside = e.land(inside, room);
    inside = e.land(inside, y_inside);
    let align = e.cu(!(elements - 1));
    let aligned = e.band(x, align);
    let first = e.select(tu, inside, aligned, zero);
    let row = e.select(tu, inside, y, zero);
    let st = match b.numeric {
        ImageNumericClass::Uint => e.t_u32(),
        ImageNumericClass::Sint => e.t_i32(),
        _ => e.t_f32(),
    };
    let vt = match b.numeric {
        ImageNumericClass::Uint => e.t_uvec(4),
        ImageNumericClass::Sint => e.t_ivec(4),
        _ => e.t_fvec(4),
    };
    let mut words = vec![zero; 4];
    for element in 0..elements {
        let k = e.cu(element);
        let column = e.iadd(first, k);
        let coord = e.construct(v2, &[column, row]);
        let color = e.ins_ops(S::ImageFetch, vt, vec![Operand::IdRef(image), Operand::IdRef(coord), Operand::ImageOperands(ImageOperands::LOD), Operand::IdRef(query_lod)]);
        let texel: Vec<Word> = (0..4).map(|c| {
            let v = e.extract(st, color, c);
            if b.numeric == ImageNumericClass::Uint { v } else { e.bitcast(tu, v) }
        }).collect();
        if attr.flags.packed {
            let raw = packed_texel(e, &f, &texel, false)?[0];
            let mask = e.cu(if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 });
            let mut v = e.band(raw, mask);
            if element != 0 {
                let at = e.cu(element * bits);
                v = e.shl(v, at);
            }
            let v = e.select(tu, inside, v, zero);
            words[0] = e.bor(words[0], v);
        } else {
            for c in 0..components {
                words[(element * components + c) as usize] = e.select(tu, inside, texel[c as usize], zero);
            }
        }
    }
    let _ = tb;
    Ok(Some(pack_result(e, &words, attr, ty, true)))
}

/// `image_gather4h_pck`: the raw bits of four texels of one row (the
/// bilinear footprint's left texel shifted by -1 .. 2, clamped to the row)
/// as one bit stream, of which DMASK selects dwords; a row past the image
/// reads zeros, one above it the first row.
fn packed_horizontal_gather(e: &mut Emitter, a: &Address, attr: ImageAttr, info: &ImageResource, b: &Binding, image: Word, ty: Type) -> R<Option<Word>> {
    if !matches!(b.dimension, ImageDimension::Dim1D | ImageDimension::Dim2D) || b.cube_dim {
        return Err("packed horizontal gather of an image dimension whose row and slice rules are not measured".into());
    }
    if attr.flags.compare || attr.flags.clamp {
        return Err("packed horizontal gather with depth comparison or an LOD clamp".into());
    }
    let f = raw_texel_format(info)?;
    let bits = texel_bits(&f);
    let (tu, ti, tf, tb) = (e.t_u32(), e.t_i32(), e.t_f32(), e.t_bool());
    let zero = e.cu(0);
    let mut stream = vec![zero; 4];
    let mut row_inside = None;
    if bits <= 32 {
        e.cap(Capability::ImageQuery);
        let two_d = b.dimension == ImageDimension::Dim2D;
        let qt = if two_d { e.t_ivec(2) } else { ti };
        let zero_i = e.ci(0);
        let size = e.ins(S::ImageQuerySizeLod, qt, &[image, zero_i]);
        let extent = |e: &mut Emitter, axis: u32| if two_d { e.extract(ti, size, axis) } else { size };
        let floor_of = |e: &mut Emitter, k: u32, length: Word, centre: bool| {
            let v = a.float(e, k);
            let l = e.ins(S::ConvertSToF, tf, &[length]);
            let mut s = e.ins(S::FMul, tf, &[v, l]);
            e.b.decorate(s, Decoration::NoContraction, []);
            if centre {
                let half = e.cf(0.5);
                s = e.ins(S::FSub, tf, &[s, half]);
            }
            let fl = e.gl(tf, GLOp::Floor, &[s]);
            e.ins(S::ConvertFToS, ti, &[fl])
        };
        let clamp_into = |e: &mut Emitter, v: Word, length: Word| {
            let one = e.ci(1);
            let last = e.ins(S::ISub, ti, &[length, one]);
            let z = e.ci(0);
            e.gl(ti, GLOp::SClamp, &[v, z, last])
        };
        let width = extent(e, 0);
        let anchor = floor_of(e, 0, width, true);
        let row = if two_d {
            let height = extent(e, 1);
            let raw_row = floor_of(e, 1, height, false);
            row_inside = Some(e.ins(S::SLessThan, tb, &[raw_row, height]));
            Some(clamp_into(e, raw_row, height))
        } else {
            None
        };
        let st = match b.numeric {
            ImageNumericClass::Uint => e.t_u32(),
            ImageNumericClass::Sint => e.t_i32(),
            _ => e.t_f32(),
        };
        let vt = match b.numeric {
            ImageNumericClass::Uint => e.t_uvec(4),
            ImageNumericClass::Sint => e.t_ivec(4),
            _ => e.t_fvec(4),
        };
        for element in 0..4u32 {
            let delta = e.ci(element as i32 - 1);
            let shifted = e.ins(S::IAdd, ti, &[anchor, delta]);
            let column = clamp_into(e, shifted, width);
            let coord = match row {
                Some(r) => {
                    let v2 = e.t_ivec(2);
                    e.construct(v2, &[column, r])
                }
                None => column,
            };
            let color = e.ins_ops(S::ImageFetch, vt, vec![Operand::IdRef(image), Operand::IdRef(coord), Operand::ImageOperands(ImageOperands::LOD), Operand::IdRef(zero_i)]);
            let texel: Vec<Word> = (0..4).map(|c| {
                let v = e.extract(st, color, c);
                if b.numeric == ImageNumericClass::Uint { v } else { e.bitcast(tu, v) }
            }).collect();
            let raw = packed_texel(e, &f, &texel, false)?[0];
            let position = element * bits;
            let (word, shift) = ((position / 32) as usize, position % 32);
            let low = if shift == 0 {
                raw
            } else {
                let at = e.cu(shift);
                e.shl(raw, at)
            };
            stream[word] = e.bor(stream[word], low);
            if shift != 0 && shift + bits > 32 {
                let at = e.cu(32 - shift);
                let high = e.shr(raw, at);
                stream[word + 1] = e.bor(stream[word + 1], high);
            }
        }
    }
    let selected: Vec<Word> = (0..4)
        .map(|k| match row_inside {
            Some(c) if bits <= 32 => e.select(tu, c, stream[k], zero),
            _ if bits <= 32 => stream[k],
            _ => zero,
        })
        .collect();
    Ok(Some(pack_result(e, &selected, attr, ty, true)))
}

/// A bilinear level-0 sample of an 11_11_10 UNORM image (viewed as its
/// 32-bit words): the footprint's four words gathered, each field
/// weighted by the bilinear weights of the coordinate and normalized
/// (the sampler clamps to the edge texels; the gather order is
/// (x0, y1), (x1, y1), (x1, y0), (x0, y0)).
#[allow(clippy::too_many_arguments)]
fn sample_packed_unorm_lz(e: &mut Emitter, conv: &ps5shader::interp::buffer::FormatInfo, attr: ImageAttr, si: Word, image: Word, coord: Word, swizzle: u32, ty: Type) -> Word {
    let (tu, tf) = (e.t_u32(), e.t_f32());
    let (u4, f2, f4, u2) = (e.t_uvec(4), e.t_fvec(2), e.t_fvec(4), e.t_uvec(2));
    let zero = e.cu(0);
    let gathered = e.ins(S::ImageGather, u4, &[si, coord, zero]);
    e.cap(Capability::ImageQuery);
    let size = e.ins(S::ImageQuerySizeLod, u2, &[image, zero]);
    let size_f = e.ins(S::ConvertUToF, f2, &[size]);
    let scaled = e.ins(S::FMul, f2, &[coord, size_f]);
    let half = e.cf(0.5);
    let halves = e.construct(f2, &[half, half]);
    let centred = e.ins(S::FSub, f2, &[scaled, halves]);
    let fractions = e.gl(f2, GLOp::Fract, &[centred]);
    let one = e.cf(1.0);
    let (fx, fy) = (e.extract(tf, fractions, 0), e.extract(tf, fractions, 1));
    let (cx, cy) = (e.ins(S::FSub, tf, &[one, fx]), e.ins(S::FSub, tf, &[one, fy]));
    let weights = [(cx, fy), (fx, fy), (fx, cy), (cx, cy)].map(|(a, b)| e.ins(S::FMul, tf, &[a, b]));
    let weights = e.construct(f4, &weights);
    let fz = e.cf(0.0);
    let mut decoded: [Option<Word>; 4] = [None; 4];
    let selected: Vec<Word> = (0..4u32)
        .map(|k| {
            if (attr.dmask as u32 >> k) & 1 == 0 {
                return e.bitcast(tu, fz);
            }
            match (swizzle >> (k * 3)) & 7 {
                1 => e.bitcast(tu, one),
                s @ 4..=7 => {
                    let physical = ((s - 4) % conv.count) as usize;
                    let v = *decoded[physical].get_or_insert_with(|| {
                        let (o, n) = (e.cu(conv.bit_offset[physical]), e.cu(conv.bits[physical]));
                        let fields = e.ins(S::BitFieldUExtract, u4, &[gathered, o, n]);
                        let texels = e.ins(S::ConvertUToF, f4, &[fields]);
                        let dot = e.ins(S::Dot, tf, &[texels, weights]);
                        let scale = e.cf(1.0 / ((1u32 << conv.bits[physical]) - 1) as f32);
                        e.ins(S::FMul, tf, &[dot, scale])
                    });
                    e.bitcast(tu, v)
                }
                _ => e.bitcast(tu, fz),
            }
        })
        .collect();
    pack_result(e, &selected, attr, ty, false)
}

/// The signed integer format of a storage image viewed as unsigned (the
/// shader saturates and masks stored fields, sign-extends loaded ones).
fn sint_storage(info: &ImageResource, b: &Binding) -> Option<ps5shader::interp::buffer::FormatInfo> {
    use ps5shader::resources::format::image_format;
    if b.class != ImageResourceClass::Storage || !image_format::raw_sint_storage_format(info.conversion_format) {
        return None;
    }
    ps5shader::interp::buffer::FormatInfo::new(info.conversion_format).filter(|f| !f.packed)
}

/// The T# destination swizzle the shader applies (selector 0 = 0,
/// 1 = 1, 4..7 = x..w).
fn apply_swizzle(e: &mut Emitter, texel: &[Word], swizzle: u32, numeric: ImageNumericClass) -> Vec<Word> {
    if swizzle == ps5shader::resources::IDENTITY_IMAGE_SWIZZLE {
        return texel.to_vec();
    }
    (0..4)
        .map(|k| match (swizzle >> (k * 3)) & 7 {
            s @ 4..=7 => texel[(s - 4) as usize],
            1 => e.cu(if numeric == ImageNumericClass::Float { 0x3F80_0000 } else { 1 }),
            _ => e.cu(0),
        })
        .collect()
}

/// The dmask-selected components as the result type (packed f16 pairs
/// under D16).
fn pack_result(e: &mut Emitter, texel: &[Word], attr: ImageAttr, ty: Type, int: bool) -> Word {
    let picked: Vec<Word> = (0..4).filter(|c| attr.dmask >> c & 1 != 0).map(|c| texel[c as usize]).collect();
    let picked = if picked.is_empty() { vec![texel[0]] } else { picked };
    finish(e, picked, attr, ty, int)
}

/// The mip an explicit-LOD gather reads, relative to the view's base
/// level: the LOD (plus 0.5 with MIP_POINT_PRECLAMP) clamped to the S#
/// MIN/MAX_LOD, then to the T# MIN_LOD (in physical mip space), then to the
/// view's levels, rounded (after the clamp without preclamp); 0 with
/// MIP_FILTER none. A linear mip filter gathers from the lower (primary)
/// of the two levels: the LOD is truncated, not rounded. Descriptor words come from the instruction's T# (words
/// 1 and 3) and S# (words 1 and 2).
fn gather_mip(e: &mut Emitter, v: Value, addr: &Address, lod_index: u32, mip_count: u32) -> R<Word> {
    let tu = e.t_u32();
    let tf = e.t_f32();
    let t = e.arg(v, 0)?;
    let s = e.arg(v, 2)?;
    let t1 = e.extract(tu, t, 1);
    let t3 = e.extract(tu, t, 3);
    let s1 = e.extract(tu, s, 1);
    let s2 = e.extract(tu, s, 2);
    let lod_fixed = |e: &mut Emitter, w: Word, first: u32| {
        let f = e.bits(w, first, 12);
        let f = e.ins(S::ConvertUToF, tf, &[f]);
        let k = e.cf(1.0 / 256.0);
        e.ins(S::FMul, tf, &[f, k])
    };
    let zero_u = e.cu(0);
    let pre = e.bits(s2, 28, 1);
    let preclamp = e.ine(pre, zero_u);
    let zero = e.cf(0.0);
    let filter = e.bits(s2, 26, 2);
    let one_u = e.cu(1);
    let point = e.ieq(filter, one_u);
    let half = e.cf(0.5);
    let half = e.select(tf, point, half, zero);
    let base = e.bits(t3, 12, 4);
    let base = e.ins(S::ConvertUToF, tf, &[base]);
    let tmin = lod_fixed(e, t1, 8);
    let minimum = e.ins(S::FSub, tf, &[tmin, base]);
    let last = e.cf((mip_count - 1) as f32);
    let lod = addr.float(e, lod_index);
    let pre_add = e.select(tf, preclamp, half, zero);
    let mut lod = e.ins(S::FAdd, tf, &[lod, pre_add]);
    let (smin, smax) = (lod_fixed(e, s1, 0), lod_fixed(e, s1, 12));
    lod = e.gl(tf, GLOp::FClamp, &[lod, smin, smax]);
    lod = e.gl(tf, GLOp::FMax, &[lod, minimum]);
    lod = e.gl(tf, GLOp::FClamp, &[lod, zero, last]);
    let post_add = e.select(tf, preclamp, zero, half);
    lod = e.ins(S::FAdd, tf, &[lod, post_add]);
    // The clamped value is nonnegative: the conversion rounds to the mip.
    let mip = e.ins(S::ConvertFToU, tu, &[lod]);
    let top = e.cu(mip_count - 1);
    let mip = e.umin(mip, top);
    let none = e.ieq(filter, zero_u);
    Ok(e.select(tu, none, zero_u, mip))
}

/// The two texels of a 1D gather (left, right), each a 4-component vector
/// of the binding's numeric class: the footprint of the coordinate in
/// level 0 (plus the texel offset), each texel sampled at its centre with
/// LOD 0.
#[allow(clippy::too_many_arguments)]
fn gather_1d(
    e: &mut Emitter,
    si: Word,
    image: Word,
    coord: Word,
    b: &Binding,
    addr: &Address,
    offset: bool,
    class: ImageNumericClass,
) -> R<[Word; 2]> {
    e.cap(Capability::ImageQuery);
    let tf = e.t_f32();
    let ti = e.t_i32();
    let tu = e.t_u32();
    let array = arrayed(b.dimension);
    let zero_i = e.ci(0);
    let size = if array {
        let t2 = e.t_ivec(2);
        let s = e.ins(S::ImageQuerySizeLod, t2, &[image, zero_i]);
        e.extract(ti, s, 0)
    } else {
        e.ins(S::ImageQuerySizeLod, ti, &[image, zero_i])
    };
    let width = e.ins(S::ConvertSToF, tf, &[size]);
    let x = if array { e.extract(tf, coord, 0) } else { coord };
    let scaled = e.ins(S::FMul, tf, &[x, width]);
    let half = e.cf(0.5);
    let u = e.ins(S::FSub, tf, &[scaled, half]);
    let mut i0 = e.gl(tf, GLOp::Floor, &[u]);
    if offset {
        let w = addr.full_word(e, 0);
        let (o, n) = (e.cu(0), e.cu(6));
        let off = e.ins(S::BitFieldSExtract, tu, &[w, o, n]);
        let off = e.bitcast(ti, off);
        let off = e.ins(S::ConvertSToF, tf, &[off]);
        i0 = e.ins(S::FAdd, tf, &[i0, off]);
    }
    let vt = match class {
        ImageNumericClass::Uint => e.t_uvec(4),
        ImageNumericClass::Sint => e.t_ivec(4),
        _ => e.t_fvec(4),
    };
    let lod = e.cf(0.0);
    let mut out = [0; 2];
    for (k, centre) in [0.5f32, 1.5].into_iter().enumerate() {
        let c = e.cf(centre);
        let t = e.ins(S::FAdd, tf, &[i0, c]);
        let t = e.ins(S::FDiv, tf, &[t, width]);
        let at = if array {
            let layer = e.extract(tf, coord, 1);
            let v2 = e.t_fvec(2);
            e.construct(v2, &[t, layer])
        } else {
            t
        };
        out[k] = e.ins_ops(
            S::ImageSampleExplicitLod,
            vt,
            vec![Operand::IdRef(si), Operand::IdRef(at), Operand::ImageOperands(ImageOperands::LOD), Operand::IdRef(lod)],
        );
    }
    Ok(out)
}

fn pack_gather(e: &mut Emitter, texel: &[Word], attr: ImageAttr, ty: Type, int: bool) -> Word {
    finish(e, texel.to_vec(), attr, ty, int)
}

/// `int`: the components are integer texels (D16 keeps their low 16 bits)
/// rather than floats (D16 rounds them to f16, to nearest even).
fn finish(e: &mut Emitter, comps: Vec<Word>, attr: ImageAttr, ty: Type, int: bool) -> Word {
    let tu = e.t_u32();
    let comps = if attr.d16 {
        let tf = e.t_f32();
        comps
            .chunks(2)
            .map(|p| {
                let half = |e: &mut Emitter, x: Word| {
                    if int {
                        let m = e.cu(0xFFFF);
                        e.ins(S::BitwiseAnd, tu, &[x, m])
                    } else {
                        let f = e.bitcast(tf, x);
                        e.f32_to_f16_bits_rte(f)
                    }
                };
                let lo = half(e, p[0]);
                match p.get(1) {
                    Some(&h) => {
                        let hi = half(e, h);
                        let sh = e.cu(16);
                        let hi = e.ins(S::ShiftLeftLogical, tu, &[hi, sh]);
                        e.ins(S::BitwiseOr, tu, &[lo, hi])
                    }
                    None => lo,
                }
            })
            .collect()
    } else {
        comps
    };
    let n = ty.components() as usize;
    let mut out: Vec<Word> = comps.into_iter().take(n).collect();
    while out.len() < n {
        out.push(e.cu(0));
    }
    if n == 1 {
        out[0]
    } else {
        let t = e.ty(ty);
        e.construct(t, &out)
    }
}

fn decorate_nonuniform(e: &mut Emitter, id: Word) {
    e.decorate(id, Decoration::NonUniform, &[]);
}

/// Cube sampling: the guest addresses a cube by face coordinates (s, t)
/// and a face slice (cube instructions: s, t in [1, 2] from
/// `v_cubesc`/`v_cubetc` / `v_cubema` + 1.5, slice = face + 8 * cube;
/// 2D-array instructions: s, t in [0, 1], slice = face + 6 * cube). The
/// host samples a cube array through a direction, so filtering crosses
/// face edges.
mod cube {
    use super::*;

    /// Face-relative (u, v) in [-1, 1], the face and the cube index.
    fn face_coords(e: &mut Emitter, a: &Address, first: u32, dim: u8) -> (Word, Word, Word, Word) {
        let tf = e.t_f32();
        let tu = e.t_u32();
        let s = a.float(e, first);
        let t = if super::guest_coords(dim) > 1 { a.float(e, first + 1) } else { e.cf(0.0) };
        let slice = if super::guest_coords(dim) > 2 { a.float(e, first + 2) } else { e.cf(0.0) };
        let two = e.cf(2.0);
        // Cube instructions: [1, 2] -> 2s - 3; arrays of faces: [0, 1] -> 2s - 1.
        let bias = e.cf(if dim == 3 { 3.0 } else { 1.0 });
        let s2 = e.ins(S::FMul, tf, &[s, two]);
        let u = e.ins(S::FSub, tf, &[s2, bias]);
        let t2 = e.ins(S::FMul, tf, &[t, two]);
        let v = e.ins(S::FSub, tf, &[t2, bias]);
        let zero = e.cf(0.0);
        let clamped = e.gl(tf, GLOp::FMax, &[slice, zero]);
        let rounded = e.gl(tf, GLOp::RoundEven, &[clamped]);
        let index = e.ins(S::ConvertFToU, tu, &[rounded]);
        let (face, cube) = if dim == 3 {
            let seven = e.cu(7);
            let three = e.cu(3);
            (e.ins(S::BitwiseAnd, tu, &[index, seven]), e.shr(index, three))
        } else {
            let six = e.cu(6);
            (e.ins(S::UMod, tu, &[index, six]), e.ins(S::UDiv, tu, &[index, six]))
        };
        (u, v, face, cube)
    }

    /// `[x, y, z]` of face `face` at (u, v), `major` on the face's axis
    /// (the direction for `major = 1`; its linear part, for derivatives,
    /// with `major = 0`).
    fn direction(e: &mut Emitter, u: Word, v: Word, face: Word, major: f32) -> [Word; 3] {
        let tf = e.t_f32();
        let nu = e.ins(S::FNegate, tf, &[u]);
        let nv = e.ins(S::FNegate, tf, &[v]);
        let p = e.cf(major);
        let n = e.cf(-major);
        // Faces +X, -X, +Y, -Y, +Z, -Z.
        let table: [[Word; 6]; 3] = [[p, n, u, u, u, nu], [nv, nv, p, n, nv, nv], [nu, u, v, nv, p, n]];
        let mut out = [0; 3];
        for (c, faces) in table.iter().enumerate() {
            let mut acc = faces[5];
            for f in (0..5).rev() {
                let k = e.cu(f as u32);
                let is = e.ieq(face, k);
                acc = e.select(tf, is, faces[f], acc);
            }
            out[c] = acc;
        }
        out
    }

    /// The cube-array coordinate (direction, cube index).
    pub(super) fn coord(e: &mut Emitter, a: &Address, first: u32, dim: u8) -> Word {
        let (u, v, face, cube) = face_coords(e, a, first, dim);
        let d = direction(e, u, v, face, 1.0);
        let tf = e.t_f32();
        let layer = e.ins(S::ConvertUToF, tf, &[cube]);
        let v4 = e.t_fvec(4);
        e.construct(v4, &[d[0], d[1], d[2], layer])
    }

    /// The LOD of a gradient sample from its face-coordinate derivatives
    /// (ds/dx, dt/dx, ds/dy, dt/dy, a whole face per unit), as for a 2D
    /// image of the face: log2 of the longer gradient in texels.
    pub(super) fn gradient_lod(e: &mut Emitter, a: &Address, image: Word) -> Word {
        let (tf, ti) = (e.t_f32(), e.t_i32());
        e.cap(Capability::ImageQuery);
        let v3 = e.t_ivec(3);
        let zero = e.ci(0);
        let size = e.ins(S::ImageQuerySizeLod, v3, &[image, zero]);
        let width = e.extract(ti, size, 0);
        let width = e.ins(S::ConvertSToF, tf, &[width]);
        let v2 = e.t_fvec(2);
        let mut lengths = [0; 2];
        for (i, l) in lengths.iter_mut().enumerate() {
            let ds = a.float(e, 2 * i as u32);
            let dt = a.float(e, 2 * i as u32 + 1);
            let g = e.construct(v2, &[ds, dt]);
            *l = e.gl(tf, GLOp::Length, &[g]);
        }
        let longest = e.gl(tf, GLOp::FMax, &[lengths[0], lengths[1]]);
        let texels = e.ins(S::FMul, tf, &[longest, width]);
        e.gl(tf, GLOp::Log2, &[texels])
    }
}
