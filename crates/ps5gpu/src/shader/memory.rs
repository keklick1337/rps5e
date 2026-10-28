// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Memory: push / shader data, descriptor bindings, buffers (bound
//! storage buffers with guest range checks and format conversion),
//! device-address (page table) accesses, LDS, scratch and atomics.

use super::emit::{Emitter, R, SCOPE_DEVICE, SCOPE_WORKGROUP};
use ps5shader::interp::buffer::{format_source, FormatInfo, FormatKind, FormatSource};
use ps5shader::resources::{bindless, native_binding_index, DescriptorBindingKind, ResourceBinding, ResourceOperand, ShaderStage};
use ps5shader::ssa::*;
use rspirv::dr::Operand;
use rspirv::spirv::{Capability, Decoration, GlslStd450Op as GLOp, Op as S, StorageClass as SC, Word};
use std::collections::{BTreeSet, HashMap};

/// Descriptor variables declared so far.
#[derive(Default)]
pub struct Resources {
    pub push: Option<Word>,
    pub shader_data: Option<Word>,
    pub buffers: Option<Word>,
    pub flat_table: Option<Word>,
    pub page_table: Option<Word>,
    pub fault: Option<Word>,
    pub gds: Option<Word>,
    pub samplers: Option<Word>,
    pub images: HashMap<u32, (Word, Word)>,
    /// The bindless set: its map words and its image array per image
    /// binding kind (one variable per declared type, all on one binding).
    pub bindless_map: Option<Word>,
    pub bindless_images: HashMap<u32, (Word, Word)>,
    pub lds: Option<Word>,
    pub lds_dwords: u32,
    /// Lane-private LDS layout of a graphics stage (see [`lane_private_lds`]),
    /// decided with the array.
    pub lds_slots: Option<LaneSlots>,
    /// The LDS is a Block-decorated struct (explicit workgroup layout) with
    /// this u64 alias for 64-bit atomics.
    pub lds64: Option<Word>,
    /// u64 views of the bound buffers and the GDS (64-bit atomics).
    pub buffers64: Option<Word>,
    pub gds64: Option<Word>,
    pub lens64: HashMap<Word, Word>,
    pub scratch: [Option<Word>; 2],
    pub scratch_dwords: u32,
    pub u32_block: Option<Word>,
    pub u64_block: Option<Word>,
    /// Runtime arrays of the blocks above (shared by their read-only views).
    pub u32_rta: Option<Word>,
    pub u64_rta: Option<Word>,
    /// Read-only (NonWritable) views of the blocks.
    pub u32_block_ro: Option<Word>,
    pub u64_block_ro: Option<Word>,
    /// The bound buffers through read-only blocks (loads of buffers the
    /// module never writes).
    pub buffers_ro: Option<Word>,
    /// The bound buffers through a Coherent view (GLC accesses).
    pub buffers_coherent: Option<Word>,
    pub uses_push: bool,
    /// Byte adjustment of each bound buffer element (hoisted).
    pub adjust: HashMap<u32, Word>,
    /// Words in each bound buffer element (hoisted).
    pub lens: HashMap<Word, Word>,
}

/// Addresses the fault buffer keeps after its count word.
pub const FAULT_SLOTS: u32 = 64;

/// Where a buffer access reads.
#[derive(Clone, Copy)]
pub enum Target {
    /// Bound storage buffer element (its id and index in the binding) and
    /// its byte adjustment.
    Bound { element: Word, adjust: Word, el: u32 },
    /// Through the device-address page table at this 64-bit base.
    Device { base: Word },
}

impl Emitter<'_, '_> {
    fn u32_block(&mut self) -> Word {
        if let Some(b) = self.res.u32_block {
            return b;
        }
        let tu = self.t_u32();
        let rta = self.b.type_runtime_array(tu);
        self.decorate(rta, Decoration::ArrayStride, &[4]);
        let id = self.b.id();
        let s = self.b.type_struct_id(Some(id), [rta]);
        self.decorate(s, Decoration::Block, &[]);
        self.b.member_decorate(s, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
        self.res.u32_block = Some(s);
        self.res.u32_rta = Some(rta);
        s
    }

    /// A NonWritable block over the runtime array `rta`.
    fn read_only_block(&mut self, rta: Word) -> Word {
        let id = self.b.id();
        let s = self.b.type_struct_id(Some(id), [rta]);
        self.decorate(s, Decoration::Block, &[]);
        self.b.member_decorate(s, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
        self.b.member_decorate(s, 0, Decoration::NonWritable, []);
        s
    }

    /// The u32 block, read-only with [`EmitOptions::readonly`](super::EmitOptions).
    fn u32_block_for_reads(&mut self) -> Word {
        let block = self.u32_block();
        if !self.feat.emit.readonly {
            return block;
        }
        if let Some(b) = self.res.u32_block_ro {
            return b;
        }
        let rta = self.res.u32_rta.expect("u32 block");
        let b = self.read_only_block(rta);
        self.res.u32_block_ro = Some(b);
        b
    }

    /// The u64 block, read-only with [`EmitOptions::readonly`](super::EmitOptions).
    fn u64_block_for_reads(&mut self) -> Word {
        let block = self.u64_block();
        if !self.feat.emit.readonly {
            return block;
        }
        if let Some(b) = self.res.u64_block_ro {
            return b;
        }
        let rta = self.res.u64_rta.expect("u64 block");
        let b = self.read_only_block(rta);
        self.res.u64_block_ro = Some(b);
        b
    }

    fn u64_block(&mut self) -> Word {
        if let Some(b) = self.res.u64_block {
            return b;
        }
        let t = self.ty(Type::U64);
        let rta = self.b.type_runtime_array(t);
        self.decorate(rta, Decoration::ArrayStride, &[8]);
        let id = self.b.id();
        let s = self.b.type_struct_id(Some(id), [rta]);
        self.decorate(s, Decoration::Block, &[]);
        self.b.member_decorate(s, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
        self.res.u64_block = Some(s);
        self.res.u64_rta = Some(rta);
        s
    }

    pub fn binding_var(&mut self, ty: Word, class: SC, kind: DescriptorBindingKind) -> Word {
        let v = self.global(ty, class);
        let binding = native_binding_index(self.stage, kind);
        self.decorate(v, Decoration::DescriptorSet, &[0]);
        self.decorate(v, Decoration::Binding, &[binding]);
        v
    }

    /// Element count of a binding kind in the layout (at least 1).
    pub fn binding_count(&self, kind: DescriptorBindingKind) -> Option<u32> {
        self.input.request.bindings.find(kind).map(|b| (b.resources.len() as u32).max(1))
    }

    /// Element index of dense resource `resource` in binding `kind`.
    pub fn binding_element(&self, kind: DescriptorBindingKind, resource: u32) -> Option<u32> {
        let b = self.input.request.bindings.find(kind)?;
        b.resources.iter().position(|&r| r == resource).map(|p| p as u32)
    }

    /// A single-buffer storage binding of u32 words.
    fn word_buffer(&mut self, kind: DescriptorBindingKind) -> R<Word> {
        let cached = match kind {
            DescriptorBindingKind::ShaderData => self.res.shader_data,
            DescriptorBindingKind::FlattenedResourceTable => self.res.flat_table,
            DescriptorBindingKind::GlobalDataShare => self.res.gds,
            _ => None,
        };
        if let Some(v) = cached {
            return Ok(v);
        }
        if self.input.request.bindings.find(kind).is_none() {
            return Err(format!("the layout has no {kind:?} binding"));
        }
        // Shader data and the resource table are host data the module
        // only reads.
        let read_only = matches!(kind, DescriptorBindingKind::ShaderData | DescriptorBindingKind::FlattenedResourceTable);
        let block = if read_only { self.u32_block_for_reads() } else { self.u32_block() };
        let v = self.binding_var(block, SC::StorageBuffer, kind);
        if read_only && self.feat.emit.readonly {
            self.decorate(v, Decoration::Restrict, &[]);
        }
        if kind == DescriptorBindingKind::GlobalDataShare {
            // GDS words are shared by every work-group (a wave spinning on a
            // flag another one stores must see the store), and the 32- and
            // 64-bit views alias one binding.
            self.decorate(v, Decoration::Coherent, &[]);
            self.decorate(v, Decoration::Aliased, &[]);
        }
        match kind {
            DescriptorBindingKind::ShaderData => self.res.shader_data = Some(v),
            DescriptorBindingKind::FlattenedResourceTable => self.res.flat_table = Some(v),
            DescriptorBindingKind::GlobalDataShare => self.res.gds = Some(v),
            _ => {}
        }
        Ok(v)
    }

    /// Word `index` of a single-buffer binding.
    pub fn word_buffer_load(&mut self, kind: DescriptorBindingKind, index: Word) -> R<Word> {
        let var = self.word_buffer(kind)?;
        let tu = self.t_u32();
        let p = self.ptr(SC::StorageBuffer, tu);
        let zero = self.cu(0);
        let a = self.access(p, var, &[zero, index]);
        if !self.is_constant(index) {
            // A word the module finds at run time (a table search): only
            // the driver keeps it in range.
            self.unchecked_buffers = true;
        }
        if self.feat.emit.readonly && matches!(kind, DescriptorBindingKind::ShaderData | DescriptorBindingKind::FlattenedResourceTable) {
            self.invariant_ptrs.insert(a);
        }
        Ok(self.load(tu, a))
    }

    /// Word `index` of the bindless table maps (the storage buffer of the
    /// bindless set).
    pub fn bindless_map_load(&mut self, index: Word) -> Word {
        let var = match self.res.bindless_map {
            Some(v) => v,
            None => {
                let block = self.u32_block_for_reads();
                let v = self.global(block, SC::StorageBuffer);
                self.decorate(v, Decoration::DescriptorSet, &[bindless::SET]);
                self.decorate(v, Decoration::Binding, &[bindless::MAP_BINDING]);
                self.res.bindless_map = Some(v);
                v
            }
        };
        let tu = self.t_u32();
        let p = self.ptr(SC::StorageBuffer, tu);
        let zero = self.cu(0);
        let a = self.access(p, var, &[zero, index]);
        // The host keeps the index inside the map.
        self.unchecked_buffers = true;
        self.load(tu, a)
    }

    pub(crate) fn push_block(&mut self) -> Word {
        if let Some(p) = self.res.push {
            return p;
        }
        let tu = self.t_u32();
        let n = self.cu(32);
        let arr = self.b.type_array(tu, n);
        self.decorate(arr, Decoration::ArrayStride, &[4]);
        let id = self.b.id();
        let s = self.b.type_struct_id(Some(id), [arr]);
        self.decorate(s, Decoration::Block, &[]);
        self.b.member_decorate(s, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
        let v = self.global(s, SC::PushConstant);
        self.res.push = Some(v);
        self.res.uses_push = true;
        v
    }

    /// Shader-data dword `i` (push constants or the shader-data buffer),
    /// loaded once in the prologue.
    pub fn shader_data(&mut self, i: u32) -> R<Word> {
        let layout = &self.input.request.bindings;
        if layout.uses_push_data() {
            let dword = layout.push_data_start_dword + i;
            if dword >= 32 {
                return Err(format!("push data dword {dword} past the block"));
            }
            let block = self.push_block();
            let tu = self.t_u32();
            let p = self.ptr(SC::PushConstant, tu);
            let (zero, idx) = (self.cu(0), self.cu(dword));
            let a = self.hoist(S::AccessChain, p, &[block, zero, idx]);
            Ok(self.hoist(S::Load, tu, &[a]))
        } else {
            let idx = self.cu(i);
            self.in_prologue(|e| e.word_buffer_load(DescriptorBindingKind::ShaderData, idx))
        }
    }

    /// Byte adjustment of bound buffer element `element` (< 256 narrow;
    /// the full dword wide).
    fn buffer_adjust(&mut self, element: u32) -> R<Word> {
        if let Some(&w) = self.res.adjust.get(&element) {
            return Ok(w);
        }
        let layout = &self.input.request.bindings;
        let w = if element < layout.memory_offset_count {
            let (word, shift, bits) = layout.memory_offset_slot(element);
            let packed = self.shader_data(word)?;
            let tu = self.t_u32();
            if bits == 32 {
                packed
            } else {
                let (o, c) = (self.cu(shift as u32), self.cu(bits as u32));
                self.hoist(S::BitFieldUExtract, tu, &[packed, o, c])
            }
        } else {
            self.cu(0)
        };
        self.res.adjust.insert(element, w);
        Ok(w)
    }

    fn buffers_var(&mut self) -> R<Word> {
        if let Some(v) = self.res.buffers {
            return Ok(v);
        }
        let n = self.binding_count(DescriptorBindingKind::Buffers).ok_or("the layout has no buffer binding")?;
        let block = self.u32_block();
        let nc = self.cu(n);
        let arr = self.b.type_array(block, nc);
        let v = self.binding_var(arr, SC::StorageBuffer, DescriptorBindingKind::Buffers);
        if self.coherent_buffers {
            self.decorate(v, Decoration::Coherent, &[]);
        }
        self.res.buffers = Some(v);
        Ok(v)
    }

    /// The bound buffers through read-only blocks (Restrict when the module
    /// writes no buffer memory at all).
    fn buffers_ro_var(&mut self) -> R<Word> {
        if let Some(v) = self.res.buffers_ro {
            return Ok(v);
        }
        let n = self.binding_count(DescriptorBindingKind::Buffers).ok_or("the layout has no buffer binding")?;
        let block = self.u32_block_for_reads();
        let nc = self.cu(n);
        let arr = self.b.type_array(block, nc);
        let v = self.binding_var(arr, SC::StorageBuffer, DescriptorBindingKind::Buffers);
        if !self.writes_buffers {
            self.decorate(v, Decoration::Restrict, &[]);
        }
        if self.coherent_buffers {
            self.decorate(v, Decoration::Coherent, &[]);
        }
        self.res.buffers_ro = Some(v);
        Ok(v)
    }

    /// The module never writes bound buffer element `el`.
    fn element_read_only(&self, el: u32) -> bool {
        let Some(b) = self.input.request.bindings.find(DescriptorBindingKind::Buffers) else { return false };
        let Some(&r) = b.resources.get(el as usize) else { return false };
        self.input.request.resources.info.buffers.get(r as usize).is_some_and(|b| !b.written && !b.atomic)
    }

    /// Pointer for a load of word `index` of bound buffer `element` (index
    /// `el`): through the read-only view when the module never writes the
    /// element and the load is not coherent.
    fn bound_load_ptr(&mut self, element: Word, el: u32, index: Word) -> R<Word> {
        if !self.feat.emit.readonly || self.volatile || !self.element_read_only(el) {
            return self.bound_ptr(element, index);
        }
        let var = self.buffers_ro_var()?;
        let tu = self.t_u32();
        let p = self.ptr(SC::StorageBuffer, tu);
        let zero = self.cu(0);
        let a = self.access(p, var, &[element, zero, index]);
        if !self.writes_buffers {
            self.invariant_ptrs.insert(a);
        }
        Ok(a)
    }

    /// Load a word through `p` (volatile for coherent accesses).
    fn load_word(&mut self, p: Word) -> Word {
        let tu = self.t_u32();
        if self.volatile {
            self.ins_ops(S::Load, tu, vec![Operand::IdRef(p), Operand::MemoryAccess(rspirv::spirv::MemoryAccess::VOLATILE)])
        } else {
            self.load(tu, p)
        }
    }

    /// Word `index` of bound buffer `element`, 0 when `index` is past the
    /// bound range: with [`EmitOptions::guarded_loads`](super::EmitOptions)
    /// only an in-range word is read (the module keeps its own indices in
    /// range), else the load relies on the driver's robustness and the
    /// result is selected.
    fn bound_word(&mut self, element: Word, el: u32, index: Word) -> R<Word> {
        let len = self.bound_len(element)?;
        let ok = self.ult(index, len);
        let tu = self.t_u32();
        let zero = self.cu(0);
        if self.feat.emit.guarded_loads {
            // Word 0 stands in for an index past the range: every bound
            // range holds at least one word (the host binds 4 bytes or
            // more, or its null buffer), and the select drops it. A load
            // without a branch keeps the loads of the program overlapping.
            let safe = self.select(tu, ok, index, zero);
            let p = self.bound_load_ptr(element, el, safe)?;
            let w = self.load_word(p);
            return Ok(self.select(tu, ok, w, zero));
        }
        let p = self.bound_load_ptr(element, el, index)?;
        // Robust access returns 0 for out-of-bounds words anyway; the
        // select keeps the result defined without it.
        self.unchecked_buffers = true;
        let w = self.load_word(p);
        Ok(self.select(tu, ok, w, zero))
    }

    /// `n` consecutive words of bound buffer `element` (index `el`) from
    /// word `first`; word `k` is what [`Self::bound_word`] reads at
    /// `indices[k]` (`indices[k]` is `first + k` unless it wraps). When all
    /// `n` lie in the bound range (then nothing wraps: the range holds at
    /// most 2^30 words) they are read with one check, otherwise one by
    /// one.
    fn bound_words(&mut self, element: Word, el: u32, first: Word, indices: &[Word]) -> R<Vec<Word>> {
        let n = indices.len() as u32;
        let len = self.bound_len(element)?;
        let tu = self.t_u32();
        let inside = self.ult(first, len);
        let room = self.isub(len, first);
        let count = self.cu(n);
        let fits = self.ule(count, room);
        let all = self.land(inside, fits);
        let err = std::cell::RefCell::new(None::<String>);
        let words = self.if_else_values(
            all,
            tu,
            n as usize,
            |e| {
                (0..n)
                    .map(|k| {
                        let index = if k == 0 {
                            first
                        } else {
                            let c = e.cu(k);
                            e.iadd(first, c)
                        };
                        match e.bound_load_ptr(element, el, index) {
                            Ok(p) => e.load_word(p),
                            Err(m) => {
                                err.borrow_mut().get_or_insert(m);
                                e.cu(0)
                            }
                        }
                    })
                    .collect()
            },
            |e| {
                indices
                    .iter()
                    .map(|&index| {
                        e.bound_word(element, el, index).unwrap_or_else(|m| {
                            err.borrow_mut().get_or_insert(m);
                            e.cu(0)
                        })
                    })
                    .collect()
            },
        );
        err.into_inner().map_or(Ok(words), Err)
    }

    /// The bound buffers through a Coherent view (an Aliased variable of the
    /// same binding): GLC loads and stores go to the device-coherent level
    /// of the caches. A non-coherent host cache (one per multiprocessor)
    /// may keep serving a word another work-group has since changed, so a
    /// work-group spinning on a flag its predecessor publishes (a decoupled
    /// look-back scan) would never see it.
    fn buffers_coherent_var(&mut self) -> R<Word> {
        if let Some(v) = self.res.buffers_coherent {
            return Ok(v);
        }
        let n = self.binding_count(DescriptorBindingKind::Buffers).ok_or("the layout has no buffer binding")?;
        let block = self.u32_block();
        let nc = self.cu(n);
        let arr = self.b.type_array(block, nc);
        let v = self.binding_var(arr, SC::StorageBuffer, DescriptorBindingKind::Buffers);
        self.res.buffers_coherent = Some(v);
        self.decorate(v, Decoration::Coherent, &[]);
        self.decorate(v, Decoration::Aliased, &[]);
        let plain = self.buffers_var()?;
        self.decorate(plain, Decoration::Aliased, &[]);
        Ok(v)
    }

    /// Pointer to word `index` of bound buffer `element` (through the
    /// Coherent view for a GLC access).
    fn bound_ptr(&mut self, element: Word, index: Word) -> R<Word> {
        let var = if self.volatile { self.buffers_coherent_var()? } else { self.buffers_var()? };
        let tu = self.t_u32();
        let p = self.ptr(SC::StorageBuffer, tu);
        let zero = self.cu(0);
        Ok(self.access(p, var, &[element, zero, index]))
    }

    /// The bound buffers as arrays of u64 (aliases of the same binding).
    fn buffers64_var(&mut self) -> R<Word> {
        if let Some(v) = self.res.buffers64 {
            return Ok(v);
        }
        let n = self.binding_count(DescriptorBindingKind::Buffers).ok_or("the layout has no buffer binding")?;
        let block = self.u64_block();
        let nc = self.cu(n);
        let arr = self.b.type_array(block, nc);
        let v = self.binding_var(arr, SC::StorageBuffer, DescriptorBindingKind::Buffers);
        self.res.buffers64 = Some(v);
        if self.coherent_buffers {
            self.decorate(v, Decoration::Coherent, &[]);
        }
        // The 64-bit view aliases the 32-bit one of the same binding: a
        // dword store and a qword access to the same bytes stay ordered.
        self.decorate(v, Decoration::Aliased, &[]);
        let v32 = self.buffers_var()?;
        self.decorate(v32, Decoration::Aliased, &[]);
        Ok(v)
    }

    /// Run `f(pointer)` on the 8-byte aligned qword at `offset` (u64
    /// pointer) when it is in range; returns its result (0 out of range).
    fn target_atomic64(&mut self, t: Target, offset: Word, f: impl FnOnce(&mut Self, Word) -> Word) -> R<Word> {
        let t64 = self.ty(Type::U64);
        let zero = self.c64(0);
        match t {
            Target::Bound { element, adjust, .. } => {
                let byte = self.iadd(offset, adjust);
                let three = self.cu(3);
                let index = self.shr(byte, three);
                let var = self.buffers64_var()?;
                let len = match self.res.lens64.get(&element) {
                    Some(&w) => w,
                    None => {
                        let block = self.u64_block();
                        let p = self.ptr(SC::StorageBuffer, block);
                        let tu = self.t_u32();
                        let w = self.in_prologue(|e| {
                            let a = e.access(p, var, &[element]);
                            e.ins_ops(S::ArrayLength, tu, vec![Operand::IdRef(a), Operand::LiteralBit32(0)])
                        });
                        self.res.lens64.insert(element, w);
                        w
                    }
                };
                let ok = self.ult(index, len);
                let p = self.ptr(SC::StorageBuffer, t64);
                let z = self.cu(0);
                let p = self.access(p, var, &[element, z, index]);
                Ok(self.if_value(ok, t64, zero, |e| f(e, p)))
            }
            Target::Device { base } => {
                let o64 = self.ins(S::UConvert, t64, &[offset]);
                let a = self.ins(S::IAdd, t64, &[base, o64]);
                let m = self.c64(!7);
                let a = self.ins(S::BitwiseAnd, t64, &[a, m]);
                let (dev, resident) = self.translate_address(a)?;
                Ok(self.if_value(resident, t64, zero, |e| {
                    e.physical_caps();
                    let p = e.ptr(SC::PhysicalStorageBuffer, t64);
                    let p = e.ins(S::ConvertUToPtr, p, &[dev]);
                    f(e, p)
                }))
            }
        }
    }

    /// Words in bound buffer `element`.
    fn bound_len(&mut self, element: Word) -> R<Word> {
        if let Some(&w) = self.res.lens.get(&element) {
            return Ok(w);
        }
        let var = self.buffers_var()?;
        let block = self.u32_block();
        let p = self.ptr(SC::StorageBuffer, block);
        let tu = self.t_u32();
        // The length is invariant: computed once in the prologue.
        let w = self.in_prologue(|e| {
            let a = e.access(p, var, &[element]);
            e.ins_ops(S::ArrayLength, tu, vec![Operand::IdRef(a), Operand::LiteralBit32(0)])
        });
        self.res.lens.insert(element, w);
        Ok(w)
    }

    // ------------------------------------------------------------------
    // Device addresses
    // ------------------------------------------------------------------

    fn page_table(&mut self) -> R<Word> {
        if let Some(v) = self.res.page_table {
            return Ok(v);
        }
        if self.input.request.bindings.find(DescriptorBindingKind::DeviceAddressPageTable).is_none() {
            return Err("an address-based access without a device-address page table binding".into());
        }
        let block = self.u64_block_for_reads();
        let v = self.binding_var(block, SC::StorageBuffer, DescriptorBindingKind::DeviceAddressPageTable);
        if self.feat.emit.readonly {
            // Only the host writes the table, between dispatches.
            self.decorate(v, Decoration::Restrict, &[]);
        }
        self.res.page_table = Some(v);
        Ok(v)
    }

    /// Device address of guest `addr` (u64), and whether its page is
    /// resident.
    fn translate_address(&mut self, addr: Word) -> R<(Word, Word)> {
        let table = self.page_table()?;
        let t64 = self.ty(Type::U64);
        let shift = self.cu(ps5shader::resources::paging::PAGE_BITS);
        let page64 = self.ins(S::ShiftRightLogical, t64, &[addr, shift]);
        let limit = self.c64(crate::buffers::BDA_PAGE_COUNT - 1);
        let page64 = self.ins(S::BitwiseAnd, t64, &[page64, limit]);
        let tu = self.t_u32();
        let page = self.ins(S::UConvert, tu, &[page64]);
        let p = self.ptr(SC::StorageBuffer, t64);
        let zero = self.cu(0);
        let a = self.access(p, table, &[zero, page]);
        if self.feat.emit.readonly {
            self.invariant_ptrs.insert(a);
        }
        let entry = self.load(t64, a);
        let mask = self.c64(ps5shader::resources::paging::PAGE_OFFSET_MASK);
        let off = self.ins(S::BitwiseAnd, t64, &[addr, mask]);
        let dev = self.ins(S::IAdd, t64, &[entry, off]);
        let z64 = self.c64(0);
        let tb = self.t_bool();
        let resident = self.ins(S::INotEqual, tb, &[entry, z64]);
        // Addresses without an entry are noted in the fault buffer (count,
        // then up to FAULT_SLOTS addresses) for the host's diagnostics
        // (`RPS5E_SHADER_FAULTS=1`: an atomic and a branch per access).
        let fault = if self.feat.emit.fault_log { self.fault_buffer() } else { None };
        if let Some(fault) = fault {
            let missing = self.lnot(resident);
            self.if_then(missing, |e| {
                let tu = e.t_u32();
                let p = e.ptr(SC::StorageBuffer, tu);
                let zero = e.cu(0);
                let slot0 = e.access(p, fault, &[zero, zero]);
                let (sc, sem, one) = (e.cu(SCOPE_DEVICE), e.cu(0), e.cu(1));
                let n = e.ins(S::AtomicIAdd, tu, &[slot0, sc, sem, one]);
                let slots = e.cu(FAULT_SLOTS);
                let room = e.ult(n, slots);
                e.if_then(room, |e| {
                    let (lo, hi) = e.split64(addr);
                    let two = e.cu(2);
                    let i = e.imul(n, two);
                    let i = e.iadd(i, two);
                    let a = e.access(p, fault, &[zero, i]);
                    e.store(a, lo);
                    let i1 = e.iadd(i, one);
                    let a = e.access(p, fault, &[zero, i1]);
                    e.store(a, hi);
                });
            });
        }
        Ok((dev, resident))
    }

    /// The fault buffer (bound with the page table), if the layout has it.
    fn fault_buffer(&mut self) -> Option<Word> {
        if let Some(v) = self.res.fault {
            return Some(v);
        }
        self.input.request.bindings.find(DescriptorBindingKind::FaultBuffer)?;
        let block = self.u32_block();
        let v = self.binding_var(block, SC::StorageBuffer, DescriptorBindingKind::FaultBuffer);
        self.res.fault = Some(v);
        Some(v)
    }

    fn physical_caps(&mut self) {
        self.cap(Capability::PhysicalStorageBufferAddresses);
    }

    fn phys_ptr(&mut self, dev: Word) -> Word {
        self.physical_caps();
        let tu = self.t_u32();
        let p = self.ptr(SC::PhysicalStorageBuffer, tu);
        self.ins(S::ConvertUToPtr, p, &[dev])
    }

    /// Load the dword at guest `addr` (u64, dword aligned); 0 when the page
    /// is not resident. A coherent (GLC) load is a relaxed device-scope
    /// atomic load: a volatile load may still hit a multiprocessor's
    /// non-coherent cache and miss another work-group's store.
    pub fn device_load(&mut self, addr: Word) -> R<Word> {
        let (dev, resident) = self.translate_address(addr)?;
        let tu = self.t_u32();
        let zero = self.cu(0);
        Ok(self.if_value(resident, tu, zero, |e| {
            let p = e.phys_ptr(dev);
            if e.volatile {
                let (scope, semantics) = (e.cu(SCOPE_DEVICE), e.cu(0));
                return e.ins(S::AtomicLoad, tu, &[p, scope, semantics]);
            }
            e.ins_ops(S::Load, tu, vec![Operand::IdRef(p), Operand::MemoryAccess(rspirv::spirv::MemoryAccess::ALIGNED), Operand::LiteralBit32(4)])
        }))
    }

    /// Load `count` (a multiple of 4) consecutive dwords at guest `addr`
    /// (u64, 16-byte aligned, within one page); zeros when the page is not
    /// resident.
    pub fn device_load_dwords(&mut self, addr: Word, count: u32) -> R<Vec<Word>> {
        if self.feat.emit.bvh_vector_loads {
            return self.device_load_quads(addr, count);
        }
        let (dev, resident) = self.translate_address(addr)?;
        let tu = self.t_u32();
        let t4 = self.t_uvec(4);
        let t64 = self.ty(Type::U64);
        let z = self.cu(0);
        let zero4 = self.const_vec(Type::U32X4, &[z, z, z, z]);
        let mut out = Vec::with_capacity(count as usize);
        for first in (0..count).step_by(4) {
            let quad = self.if_value(resident, t4, zero4, |e| {
                let parts: Vec<Word> = (0..4)
                    .map(|i| {
                        let off = e.c64(((first + i) * 4) as u64);
                        let a = e.ins(S::IAdd, t64, &[dev, off]);
                        let p = e.phys_ptr(a);
                        e.ins_ops(S::Load, tu, vec![Operand::IdRef(p), Operand::MemoryAccess(rspirv::spirv::MemoryAccess::ALIGNED), Operand::LiteralBit32(4)])
                    })
                    .collect();
                e.construct(t4, &parts)
            });
            for i in 0..4 {
                out.push(self.extract(tu, quad, i));
            }
        }
        Ok(out)
    }

    /// [`Self::device_load_dwords`]: one residency branch around the
    /// 16-byte vector loads of the whole run.
    fn device_load_quads(&mut self, addr: Word, count: u32) -> R<Vec<Word>> {
        let (dev, resident) = self.translate_address(addr)?;
        let tu = self.t_u32();
        let t64 = self.ty(Type::U64);
        let zeros = vec![self.cu(0); count as usize];
        Ok(self.if_values(resident, tu, &zeros, |e| {
            e.physical_caps();
            let t4 = e.t_uvec(4);
            let p4 = e.ptr(SC::PhysicalStorageBuffer, t4);
            let mut out = Vec::with_capacity(count as usize);
            for first in (0..count).step_by(4) {
                let a = if first == 0 {
                    dev
                } else {
                    let off = e.c64(first as u64 * 4);
                    e.ins(S::IAdd, t64, &[dev, off])
                };
                let p = e.ins(S::ConvertUToPtr, p4, &[a]);
                let v = e.ins_ops(S::Load, t4, vec![Operand::IdRef(p), Operand::MemoryAccess(rspirv::spirv::MemoryAccess::ALIGNED), Operand::LiteralBit32(16)]);
                for i in 0..4 {
                    out.push(e.extract(tu, v, i));
                }
            }
            out
        }))
    }

    /// `n` dwords from guest address `a0` (u64, dword aligned); dword `k` is
    /// what [`Self::device_load`] reads at `a0 + 4k`. When all `n` lie in
    /// one page the address is translated once and the dwords are loaded
    /// as vectors; otherwise one by one.
    pub fn device_load_span(&mut self, a0: Word, n: u32) -> R<Vec<Word>> {
        let page = ps5shader::resources::paging::PAGE_OFFSET_MASK as u32 + 1;
        let tu = self.t_u32();
        let t64 = self.ty(Type::U64);
        let (lo, _) = self.split64(a0);
        let mask = self.cu(page - 1);
        let within = self.band(lo, mask);
        let last = self.cu(page - 4 * n);
        let one_page = self.ule(within, last);
        let err = std::cell::RefCell::new(None::<String>);
        let words = self.if_else_values(
            one_page,
            tu,
            n as usize,
            |e| match e.translate_address(a0) {
                Ok((dev, resident)) => {
                    let zeros = vec![e.cu(0); n as usize];
                    e.if_values(resident, tu, &zeros, |e| {
                        let mut out = Vec::with_capacity(n as usize);
                        let mut first = 0;
                        while first < n {
                            let count = (n - first).min(4);
                            let a = if first == 0 {
                                dev
                            } else {
                                let off = e.c64(first as u64 * 4);
                                e.ins(S::IAdd, t64, &[dev, off])
                            };
                            e.physical_caps();
                            let vt = if count == 1 { tu } else { e.t_uvec(count) };
                            let p = e.ptr(SC::PhysicalStorageBuffer, vt);
                            let p = e.ins(S::ConvertUToPtr, p, &[a]);
                            let access = if e.volatile { rspirv::spirv::MemoryAccess::ALIGNED | rspirv::spirv::MemoryAccess::VOLATILE } else { rspirv::spirv::MemoryAccess::ALIGNED };
                            let v = e.ins_ops(S::Load, vt, vec![Operand::IdRef(p), Operand::MemoryAccess(access), Operand::LiteralBit32(4)]);
                            if count == 1 {
                                out.push(v);
                            } else {
                                for i in 0..count {
                                    out.push(e.extract(tu, v, i));
                                }
                            }
                            first += count;
                        }
                        out
                    })
                }
                Err(m) => {
                    err.borrow_mut().get_or_insert(m);
                    vec![e.cu(0); n as usize]
                }
            },
            |e| {
                (0..n)
                    .map(|k| {
                        let a = if k == 0 {
                            a0
                        } else {
                            let off = e.c64(k as u64 * 4);
                            e.ins(S::IAdd, t64, &[a0, off])
                        };
                        e.device_load(a).unwrap_or_else(|m| {
                            err.borrow_mut().get_or_insert(m);
                            e.cu(0)
                        })
                    })
                    .collect()
            },
        );
        err.into_inner().map_or(Ok(words), Err)
    }

    /// Store `values` as consecutive dwords from guest address `a0` (u64,
    /// dword aligned), each as [`Self::device_store`] does; within one
    /// page the address is translated once.
    pub fn device_store_span(&mut self, a0: Word, values: &[Word]) -> R<()> {
        let n = values.len() as u32;
        let page = ps5shader::resources::paging::PAGE_OFFSET_MASK as u32 + 1;
        let t64 = self.ty(Type::U64);
        let (lo, _) = self.split64(a0);
        let mask = self.cu(page - 1);
        let within = self.band(lo, mask);
        let last = self.cu(page - 4 * n);
        let one_page = self.ule(within, last);
        let mut err = None;
        let (then_l, else_l, merge) = (self.b.id(), self.b.id(), self.b.id());
        self.b.selection_merge(merge, rspirv::spirv::SelectionControl::NONE).unwrap();
        self.b.branch_conditional(one_page, then_l, else_l, []).unwrap();
        self.begin(then_l);
        self.open_scope();
        match self.translate_address(a0) {
            Ok((dev, resident)) => self.if_then(resident, |e| {
                for (k, &v) in values.iter().enumerate() {
                    let a = if k == 0 {
                        dev
                    } else {
                        let off = e.c64(k as u64 * 4);
                        e.ins(S::IAdd, t64, &[dev, off])
                    };
                    let p = e.phys_ptr(a);
                    e.void(S::Store, vec![Operand::IdRef(p), Operand::IdRef(v), Operand::MemoryAccess(rspirv::spirv::MemoryAccess::ALIGNED), Operand::LiteralBit32(4)]);
                }
            }),
            Err(m) => err = Some(m),
        }
        self.close_scope();
        self.b.branch(merge).unwrap();
        self.begin(else_l);
        self.open_scope();
        for (k, &v) in values.iter().enumerate() {
            let a = if k == 0 {
                a0
            } else {
                let off = self.c64(k as u64 * 4);
                self.ins(S::IAdd, t64, &[a0, off])
            };
            if let Err(m) = self.device_store(a, v) {
                err.get_or_insert(m);
            }
        }
        self.close_scope();
        self.b.branch(merge).unwrap();
        self.begin(merge);
        err.map_or(Ok(()), Err)
    }

    pub fn device_store(&mut self, addr: Word, value: Word) -> R<()> {
        let (dev, resident) = self.translate_address(addr)?;
        self.if_then(resident, |e| {
            let p = e.phys_ptr(dev);
            let access = if e.volatile { rspirv::spirv::MemoryAccess::ALIGNED | rspirv::spirv::MemoryAccess::VOLATILE } else { rspirv::spirv::MemoryAccess::ALIGNED };
            e.void(S::Store, vec![Operand::IdRef(p), Operand::IdRef(value), Operand::MemoryAccess(access), Operand::LiteralBit32(4)]);
        });
        Ok(())
    }

    // ------------------------------------------------------------------
    // Word access over a target
    // ------------------------------------------------------------------

    /// Load the dword at byte `offset` of the target (offset relative to
    /// the descriptor base; dword aligned down).
    fn target_load(&mut self, t: Target, offset: Word) -> R<Word> {
        match t {
            Target::Bound { element, adjust, el } => {
                let byte = self.iadd(offset, adjust);
                let two = self.cu(2);
                let index = self.shr(byte, two);
                self.bound_target_word(element, el, index)
            }
            Target::Device { base } => {
                let t64 = self.ty(Type::U64);
                let o64 = self.ins(S::UConvert, t64, &[offset]);
                let a = self.ins(S::IAdd, t64, &[base, o64]);
                let m = self.c64(!3);
                let a = self.ins(S::BitwiseAnd, t64, &[a, m]);
                self.device_load(a)
            }
        }
    }

    /// Word `index` of bound buffer `element` (index `el`) as
    /// [`Self::target_load`] reads it.
    fn bound_target_word(&mut self, element: Word, el: u32, index: Word) -> R<Word> {
        if self.feat.emit.readonly || self.feat.emit.guarded_loads {
            return self.bound_word(element, el, index);
        }
        let len = self.bound_len(element)?;
        let ok = self.ult(index, len);
        let tu = self.t_u32();
        let zero = self.cu(0);
        // Robust access returns 0 for out-of-bounds words anyway;
        // the select keeps the result defined without it.
        self.unchecked_buffers = true;
        let p = self.bound_ptr(element, index)?;
        let w = if self.volatile {
            self.ins_ops(S::Load, tu, vec![Operand::IdRef(p), Operand::MemoryAccess(rspirv::spirv::MemoryAccess::VOLATILE)])
        } else {
            self.load(tu, p)
        };
        Ok(self.select(tu, ok, w, zero))
    }

    fn target_store(&mut self, t: Target, offset: Word, value: Word) -> R<()> {
        match t {
            Target::Bound { element, adjust, .. } => {
                let byte = self.iadd(offset, adjust);
                let two = self.cu(2);
                let index = self.shr(byte, two);
                let len = self.bound_len(element)?;
                let ok = self.ult(index, len);
                let p = self.bound_ptr(element, index)?;
                self.if_then(ok, |e| e.store(p, value));
                Ok(())
            }
            Target::Device { base } => {
                let t64 = self.ty(Type::U64);
                let o64 = self.ins(S::UConvert, t64, &[offset]);
                let a = self.ins(S::IAdd, t64, &[base, o64]);
                let m = self.c64(!3);
                let a = self.ins(S::BitwiseAnd, t64, &[a, m]);
                self.device_store(a, value)
            }
        }
    }

    /// Store the low `bytes` (1 or 2) bytes of `value` at byte `offset`
    /// (two atomics keep neighbouring bytes intact).
    fn target_store_sub(&mut self, t: Target, offset: Word, value: Word, bytes: u32) -> R<()> {
        let three = self.cu(3);
        let low = match t {
            Target::Bound { adjust, .. } => {
                let b = self.iadd(offset, adjust);
                self.band(b, three)
            }
            Target::Device { .. } => self.band(offset, three),
        };
        let eight = self.cu(8);
        let shift = self.imul(low, eight);
        let field = self.cu(if bytes == 1 { 0xFF } else { 0xFFFF });
        let v = self.band(value, field);
        let v = self.shl(v, shift);
        let m = self.shl(field, shift);
        let tu = self.t_u32();
        let nm = self.ins(S::Not, tu, &[m]);
        self.target_atomic(t, offset, |e, p| {
            let (scope, sem) = (e.cu(SCOPE_DEVICE), e.cu(0));
            e.ins(S::AtomicAnd, tu, &[p, scope, sem, nm]);
            e.ins(S::AtomicOr, tu, &[p, scope, sem, v])
        })?;
        if bytes == 2 {
            // A short at the last byte of a dword: its high byte is the low
            // byte of the next dword.
            let crosses = self.ieq(low, three);
            self.if_then(crosses, |e| {
                let one = e.cu(1);
                let next = e.iadd(offset, one);
                let eight = e.cu(8);
                let high = e.shr(value, eight);
                let byte = e.cu(0xFF);
                let high = e.band(high, byte);
                let keep = e.cu(!0xFFu32);
                let r = e.target_atomic(t, next, |e, p| {
                    let (scope, sem) = (e.cu(SCOPE_DEVICE), e.cu(0));
                    e.ins(S::AtomicAnd, tu, &[p, scope, sem, keep]);
                    e.ins(S::AtomicOr, tu, &[p, scope, sem, high])
                });
                if let Err(m) = r {
                    e.res_error(m);
                }
            });
            self.take_error()?;
        }
        Ok(())
    }

    /// Run `f(pointer)` on the word at `offset` when it is in range;
    /// returns its result (0 out of range).
    fn target_atomic(&mut self, t: Target, offset: Word, f: impl FnOnce(&mut Self, Word) -> Word) -> R<Word> {
        let tu = self.t_u32();
        let zero = self.cu(0);
        match t {
            Target::Bound { element, adjust, .. } => {
                let byte = self.iadd(offset, adjust);
                let two = self.cu(2);
                let index = self.shr(byte, two);
                let len = self.bound_len(element)?;
                let ok = self.ult(index, len);
                let p = self.bound_ptr(element, index)?;
                Ok(self.if_value(ok, tu, zero, |e| f(e, p)))
            }
            Target::Device { base } => {
                let t64 = self.ty(Type::U64);
                let o64 = self.ins(S::UConvert, t64, &[offset]);
                let a = self.ins(S::IAdd, t64, &[base, o64]);
                let m = self.c64(!3);
                let a = self.ins(S::BitwiseAnd, t64, &[a, m]);
                let (dev, resident) = self.translate_address(a)?;
                Ok(self.if_value(resident, tu, zero, |e| {
                    let p = e.phys_ptr(dev);
                    f(e, p)
                }))
            }
        }
    }

    /// The target of a planned buffer access, or the device-address
    /// fallback through the descriptor's base.
    fn buffer_target(&mut self, v: Value, desc: Word) -> R<Target> {
        let binding = self.uses(v).and_then(|u| u.iter().find(|u| u.operand == ResourceOperand::Resource)).map(|u| u.binding);
        if let Some(ResourceBinding::Buffer(i)) = binding {
            if let Some(el) = self.binding_element(DescriptorBindingKind::Buffers, i) {
                let adjust = self.buffer_adjust(el)?;
                let element = self.cu(el);
                return Ok(Target::Bound { element, adjust, el });
            }
        }
        let tu = self.t_u32();
        let w0 = self.extract(tu, desc, 0);
        let w1 = self.extract(tu, desc, 1);
        let ffff = self.cu(0xFFFF);
        let hi = self.band(w1, ffff);
        let base = self.join64(w0, hi);
        Ok(Target::Device { base })
    }

    fn buffer_resource(&self, v: Value) -> Option<u32> {
        let binding = self.uses(v).and_then(|u| u.iter().find(|u| u.operand == ResourceOperand::Resource)).map(|u| u.binding);
        match binding {
            Some(ResourceBinding::Buffer(i)) => Some(i),
            _ => None,
        }
    }
}

/// Value of user-data SGPR `register`.
pub fn user_data(e: &mut Emitter, register: u32) -> R<Word> {
    let layout = &e.input.request.bindings;
    match layout.user_data_registers.iter().position(|&r| r == register) {
        Some(i) => e.shader_data(i as u32),
        None => Ok(e.cu(0)),
    }
}

/// Range check of one buffer access unit (see `Op::BufferLoad`).
struct BufferAddress {
    index: Word,
    /// `voffset + inst offset` (wrapping), without `extra`.
    offset: Word,
    soffset: Word,
    stride: Word,
    records: Word,
    oob: Word,
    /// Byte offset of the access from the descriptor base, without `extra`
    /// (swizzling applied).
    byte: Word,
    /// SWIZZLE_EN with a nonzero stride: a constant when the descriptor
    /// fields are specialized, otherwise a runtime bool.
    swizzle: Swizzle,
    /// Specialized stride and OOB_SELECT (bound buffers).
    known: Option<(u32, u32)>,
}

#[derive(Clone, Copy)]
enum Swizzle {
    Off,
    On,
    Runtime(Word),
}

impl Emitter<'_, '_> {
    /// The specialized V# stride fields of a bound buffer (stride |
    /// swizzle << 14 | index stride << 16 | add-tid << 20).
    fn packed_stride(&self, v: Value) -> Option<u32> {
        let i = self.buffer_resource(v)?;
        self.input.request.resources.info.buffers.get(i as usize).map(|b| b.packed_stride)
    }

    fn buffer_address(&mut self, v: Value, attr: &BufferAttr, desc: Word, vindex: Word, voffset: Word, soffset: Word) -> BufferAddress {
        let tu = self.t_u32();
        let zero = self.cu(0);
        let index = if attr.idxen { vindex } else { zero };
        let base = if attr.offen { voffset } else { zero };
        let offset = if attr.offset == 0 {
            base
        } else {
            let imm = self.cu(attr.offset);
            self.iadd(base, imm)
        };
        let records = self.extract(tu, desc, 2);
        let w3 = self.extract(tu, desc, 3);
        let oob = self.bits(w3, 28, 2);
        // Bound buffers are compiled for their descriptor's stride fields;
        // device-address accesses decode them at run time.
        let packed = self.packed_stride(v);
        let known = packed.map(|p| (p & 0x3FFF, (p >> 24) & 3));
        let (stride, swizzle, index_stride, add_tid) = match packed {
            Some(p) => {
                let stride = p & 0x3FFF;
                let swz = if stride != 0 && (p >> 14) & 1 != 0 { Swizzle::On } else { Swizzle::Off };
                let is = self.cu((p >> 16) & 3);
                let tid = if (p >> 20) & 1 != 0 { Some(None) } else { None };
                (self.cu(stride), swz, is, tid)
            }
            None => {
                let w1 = self.extract(tu, desc, 1);
                let stride = self.bits(w1, 16, 14);
                let en = self.bits(w1, 31, 1);
                let en = self.ine(en, zero);
                let nz = self.ine(stride, zero);
                let swz = self.land(en, nz);
                let is = self.bits(w3, 21, 2);
                let tid = self.bits(w3, 23, 1);
                let tid = self.ine(tid, zero);
                (stride, Swizzle::Runtime(swz), is, Some(Some(tid)))
            }
        };
        // ADD_TID adds the lane to the index (constant or run-time flag).
        let index = match add_tid {
            Some(tid) => {
                let lane = self.lane_id();
                let m = self.cu(63);
                let lane = self.band(lane, m);
                let with = self.iadd(index, lane);
                match tid {
                    Some(flag) => self.select(tu, flag, with, index),
                    None => with,
                }
            }
            None => index,
        };
        let linear = {
            let is = self.imul(index, stride);
            self.iadd(is, offset)
        };
        let rel = match swizzle {
            Swizzle::Off => linear,
            _ => {
                // Records interleave in groups of 2^(INDEX_STRIDE + 3) with
                // dword elements.
                let three = self.cu(3);
                let shift = self.iadd(index_stride, three);
                let one = self.cu(1);
                let n = self.shl(one, shift);
                let msb_i = self.shr(index, shift);
                let nm1 = self.isub(n, one);
                let lsb_i = self.band(index, nm1);
                let not3 = self.cu(!3);
                let off_msb = self.band(offset, not3);
                let off_lsb = self.band(offset, three);
                let a = self.imul(msb_i, stride);
                let a = self.iadd(a, off_msb);
                let msb = self.imul(a, n);
                let two = self.cu(2);
                let l = self.shl(lsb_i, two);
                let lsb = self.iadd(l, off_lsb);
                let sw = self.iadd(msb, lsb);
                match swizzle {
                    Swizzle::Runtime(c) => self.select(tu, c, sw, linear),
                    _ => sw,
                }
            }
        };
        let byte = self.iadd(rel, soffset);
        BufferAddress { index, offset, soffset, stride, records, oob, byte, swizzle, known }
    }

    /// Byte offset (relative to the descriptor base) of byte `extra`, and
    /// whether `bytes` bytes there are in range (the rules of
    /// `ps5shader::interp::buffer::buffer_in_bounds`).
    fn buffer_unit(&mut self, a: &BufferAddress, extra: u32, bytes: u32) -> (Word, Word) {
        let byte = if extra == 0 {
            a.byte
        } else {
            let ex = self.cu(extra);
            self.iadd(a.byte, ex)
        };
        (byte, self.buffer_unit_ok(a, extra, bytes))
    }

    /// Whether `bytes` bytes at byte `extra` of the access are in range
    /// (the second half of [`Self::buffer_unit`]).
    fn buffer_unit_ok(&mut self, a: &BufferAddress, extra: u32, bytes: u32) -> Word {
        let t64 = self.ty(Type::U64);
        let tb = self.t_bool();
        let off = if extra == 0 {
            a.offset
        } else {
            let ex = self.cu(extra);
            self.iadd(a.offset, ex)
        };
        if let Some((stride, oob)) = a.known {
            // The rule of the specialized OOB_SELECT only, in 32-bit
            // arithmetic where it cannot overflow.
            let ok = match oob {
                1 => self.ult(a.index, a.records),
                2 => {
                    let zero = self.cu(0);
                    self.ine(a.records, zero)
                }
                0 => {
                    // Structured with offset: index < records and the
                    // bytes lie inside the record (none with stride 0).
                    let in_records = self.ult(a.index, a.records);
                    if bytes > stride {
                        self.cb(false)
                    } else {
                        let last = self.cu(stride - bytes);
                        let fits = self.ule(off, last);
                        self.land(in_records, fits)
                    }
                }
                _ => {
                    // raw: soffset <= records and off + bytes <= records - soffset.
                    let c1 = self.ule(a.soffset, a.records);
                    let room = self.isub(a.records, a.soffset);
                    let nb = self.cu(bytes);
                    let payload = |e: &mut Self, size: Word| {
                        let big = e.ule(nb, size);
                        let last = e.isub(size, nb);
                        let fits = e.ule(off, last);
                        e.land(big, fits)
                    };
                    let tail = match a.swizzle {
                        Swizzle::On if oob == 3 => {
                            let in_room = self.ult(a.index, room);
                            let st = self.cu(stride);
                            let p = payload(self, st);
                            self.land(in_room, p)
                        }
                        _ => payload(self, room),
                    };
                    self.land(c1, tail)
                }
            };
            return ok;
        }
        // Range rules in 64-bit arithmetic.
        let w = |e: &mut Self, x: Word| e.ins(S::UConvert, t64, &[x]);
        let (off64, so64, rec64, idx64, str64) = (w(self, off), w(self, a.soffset), w(self, a.records), w(self, a.index), w(self, a.stride));
        let nb = self.c64(bytes as u64);
        let end = self.ins(S::IAdd, t64, &[off64, nb]);
        // raw: soffset <= records && off + bytes <= records - soffset
        let c1 = self.ins(S::ULessThanEqual, tb, &[so64, rec64]);
        let room = self.ins(S::ISub, t64, &[rec64, so64]);
        let c2 = self.ins(S::ULessThanEqual, tb, &[end, room]);
        let raw_payload = c2;
        let idx_ok = self.ult(a.index, a.records);
        // Swizzled raw buffers check the index against the room and the
        // payload against the stride.
        let raw_tail = match a.swizzle {
            Swizzle::Off => raw_payload,
            sw => {
                let in_room = self.ins(S::ULessThan, tb, &[idx64, room]);
                let in_stride = self.ins(S::ULessThanEqual, tb, &[end, str64]);
                let swizzled = self.land(in_room, in_stride);
                match sw {
                    Swizzle::Runtime(c) => self.select(tb, c, swizzled, raw_payload),
                    _ => swizzled,
                }
            }
        };
        let raw = self.land(c1, raw_tail);
        // structured with offset: index < records && off + bytes <= stride
        let in_record = self.ins(S::ULessThanEqual, tb, &[end, str64]);
        let sel0 = self.land(idx_ok, in_record);
        let zero = self.cu(0);
        let nonzero = self.ine(a.records, zero);
        let (one, two) = (self.cu(1), self.cu(2));
        let is1 = self.ieq(a.oob, one);
        let is2 = self.ieq(a.oob, two);
        let is0 = self.ieq(a.oob, zero);
        let r = self.select(tb, is2, nonzero, raw);
        let r = self.select(tb, is1, idx_ok, r);
        self.select(tb, is0, sel0, r)
    }
}

/// `RPS5E_COHERENT_PAIRS=0`: GLC dwordx2 buffer loads and stores stay two
/// dword accesses.
fn coherent_pairs_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("RPS5E_COHERENT_PAIRS").is_ok_and(|v| v == "0"))
}

/// Emit a memory operation. Globally coherent buffer and global/FLAT
/// accesses (GLC) are volatile: another wave or work-group may change the
/// memory between two reads (a spin on a flag another work-group publishes
/// must not be hoisted out of its loop).
pub fn emit(e: &mut Emitter, v: Value, d: &ValueData) -> R<Option<Word>> {
    let coherent = matches!(d.op, Op::BufferLoad { attr, .. } | Op::BufferStore { attr, .. } if attr.glc)
        || matches!(d.op, Op::GlobalLoad { glc: true, .. } | Op::GlobalStore { glc: true, .. });
    e.volatile = coherent;
    let r = emit_access(e, v, d);
    e.volatile = false;
    r
}

fn emit_access(e: &mut Emitter, v: Value, d: &ValueData) -> R<Option<Word>> {
    let n = d.args.len();
    let exec = |e: &mut Emitter| e.arg(v, n - 1);
    match d.op {
        Op::ScalarLoad { .. } => {
            let binding = e.uses(v).and_then(|u| u.iter().find(|u| u.operand == ResourceOperand::Resource)).map(|u| u.binding);
            if binding == Some(ResourceBinding::Suppressed) {
                return Ok(Some(e.cu(0)));
            }
            let addr = e.arg(v, 0)?;
            if let Some(ResourceBinding::FlattenedSlot(s)) = binding {
                let idx = e.cu(s);
                return Ok(Some(e.word_buffer_load(DescriptorBindingKind::FlattenedResourceTable, idx)?));
            }
            if e.feat.emit.device_once {
                if let Some(w) = group_load(e, v)? {
                    return Ok(Some(w));
                }
            }
            Ok(Some(e.device_load(addr)?))
        }
        Op::ScalarBufferLoad { .. } => {
            let binding = e.uses(v).and_then(|u| u.iter().find(|u| u.operand == ResourceOperand::Resource)).map(|u| u.binding);
            if binding == Some(ResourceBinding::Suppressed) {
                // A T# dword of a run-time selected image: only that image's
                // handle reads it, and the image binding stands for it.
                return Ok(Some(e.cu(0)));
            }
            let (desc, offset) = (e.arg(v, 0)?, e.arg(v, 1)?);
            if let Some(ResourceBinding::FlattenedSlot(s)) = binding {
                let idx = e.cu(s);
                return Ok(Some(e.word_buffer_load(DescriptorBindingKind::FlattenedResourceTable, idx)?));
            }
            let t = e.buffer_target(v, desc)?;
            let grouped = match t {
                Target::Bound { .. } => e.feat.emit.bound_once,
                Target::Device { .. } => e.feat.emit.device_once,
            };
            if grouped {
                if let Some(w) = group_load(e, v)? {
                    return Ok(Some(w));
                }
            }
            // In range when the dword lies below the buffer size
            // (stride * records, or records bytes without a stride).
            let tu = e.t_u32();
            let w1 = e.extract(tu, desc, 1);
            let stride = e.bits(w1, 16, 14);
            let records = e.extract(tu, desc, 2);
            let zero = e.cu(0);
            let s0 = e.ieq(stride, zero);
            let t64 = e.ty(Type::U64);
            let (s64, r64) = (e.ins(S::UConvert, t64, &[stride]), e.ins(S::UConvert, t64, &[records]));
            let prod = e.ins(S::IMul, t64, &[s64, r64]);
            let size = e.select(t64, s0, r64, prod);
            let o64 = e.ins(S::UConvert, t64, &[offset]);
            let four = e.c64(4);
            let end = e.ins(S::IAdd, t64, &[o64, four]);
            let tb = e.t_bool();
            let ok = e.ins(S::ULessThanEqual, tb, &[end, size]);
            let three = e.cu(3);
            let low = e.band(offset, three);
            let aligned = e.ieq(low, zero);
            let ok = e.land(ok, aligned);
            let w = e.target_load(t, offset)?;
            Ok(Some(e.select(tu, ok, w, zero)))
        }
        Op::BufferLoad { attr, data } => {
            let (desc, vi, vo, so) = (e.arg(v, 0)?, e.arg(v, 1)?, e.arg(v, 2)?, e.arg(v, 3)?);
            let t = e.buffer_target(v, desc)?;
            let a = e.buffer_address(v, &attr, desc, vi, vo, so);
            match data {
                BufferData::Raw(access) => Ok(Some(raw_load(e, t, &a, access, d.ty)?)),
                BufferData::Format { components, typed_format, d16 } => {
                    Ok(Some(format_load(e, v, t, &a, desc, components as u32, typed_format, d16, d.ty)?))
                }
            }
        }
        Op::BufferStore { attr, data } => {
            let (desc, vi, vo, so, value) = (e.arg(v, 0)?, e.arg(v, 1)?, e.arg(v, 2)?, e.arg(v, 3)?, e.arg(v, 4)?);
            let vty = e.arg_ty(v, 4);
            let ex = exec(e)?;
            let t = e.buffer_target(v, desc)?;
            let a = e.buffer_address(v, &attr, desc, vi, vo, so);
            e.if_then(ex, |e| {
                let r = match data {
                    BufferData::Raw(access) => raw_store(e, t, &a, access, value, vty),
                    BufferData::Format { components, typed_format, d16 } => {
                        format_store(e, v, t, &a, desc, components as u32, typed_format, d16, value, vty)
                    }
                };
                if let Err(m) = r {
                    e.res_error(m);
                }
            });
            e.take_error()?;
            Ok(None)
        }
        Op::BufferAtomic { attr, op } => {
            let (desc, vi, vo, so) = (e.arg(v, 0)?, e.arg(v, 1)?, e.arg(v, 2)?, e.arg(v, 3)?);
            let data = e.arg(v, 4)?;
            let dty = e.arg_ty(v, 4);
            let cmp = if n > 6 { Some(e.arg(v, 5)?) } else { None };
            let ex = exec(e)?;
            let t = e.buffer_target(v, desc)?;
            let a = e.buffer_address(v, &attr, desc, vi, vo, so);
            if dty == Type::U64 {
                let (byte, ok) = e.buffer_unit(&a, 0, 8);
                let cond = e.land(ex, ok);
                let t64 = e.ty(Type::U64);
                let zero = e.c64(0);
                let r = e.if_value(cond, t64, zero, |e| {
                    e.target_atomic64(t, byte, |e, p| atomic64(e, p, op, data, cmp, SCOPE_DEVICE)).unwrap_or(zero)
                });
                return Ok(Some(r));
            }
            let (byte, ok) = e.buffer_unit(&a, 0, 4);
            let cond = e.land(ex, ok);
            let tu = e.t_u32();
            let zero = e.cu(0);
            let r = e.if_value(cond, tu, zero, |e| {
                e.target_atomic(t, byte, |e, p| atomic(e, p, op, data, cmp, SCOPE_DEVICE)).unwrap_or(zero)
            });
            Ok(Some(r))
        }
        Op::GlobalLoad { access, flat, .. } => {
            let addr = e.arg(v, 0)?;
            if !e.routes_apertures(flat) {
                return Ok(Some(global_load(e, d, addr, access)?));
            }
            let ex = exec(e)?;
            let ty = e.ty(d.ty);
            let zero = zero_of(e, d.ty);
            let r = e.if_value(ex, ty, zero, |e| {
                let r = route_flat(e, addr, ty, |e, a, low| aperture_load_access(e, a, low, access, d.ty), |e| global_load(e, d, addr, access));
                guarded(e, r)
            });
            e.take_error()?;
            Ok(Some(r))
        }
        Op::GlobalStore { access, flat, .. } => {
            let (addr, value) = (e.arg(v, 0)?, e.arg(v, 1)?);
            let vty = e.arg_ty(v, 1);
            let ex = exec(e)?;
            let routed = e.routes_apertures(flat);
            e.if_then(ex, |e| {
                let r = if routed {
                    route_flat_store(e, addr, |e, a, low| aperture_store_access(e, a, low, access, value, vty), |e| global_store(e, addr, access, value, vty))
                } else {
                    global_store(e, addr, access, value, vty)
                };
                if let Err(m) = r {
                    e.res_error(m);
                }
            });
            e.take_error()?;
            Ok(None)
        }
        Op::GlobalAtomic { op, flat, .. } => {
            let (addr, data) = (e.arg(v, 0)?, e.arg(v, 1)?);
            let cmp = if n > 3 { Some(e.arg(v, 2)?) } else { None };
            let ex = exec(e)?;
            let wide = e.arg_ty(v, 1) == Type::U64;
            let ty = e.ty(if wide { Type::U64 } else { Type::U32 });
            let zero = if wide { e.c64(0) } else { e.cu(0) };
            let routed = e.routes_apertures(flat);
            let r = e.if_value(ex, ty, zero, |e| {
                let r = if routed {
                    route_flat(e, addr, ty, |e, a, low| e.aperture_atomic(a, low, op, data, cmp, wide), |e| global_atomic(e, addr, op, data, cmp, wide))
                } else {
                    global_atomic(e, addr, op, data, cmp, wide)
                };
                guarded(e, r)
            });
            e.take_error()?;
            Ok(Some(r))
        }
        Op::LdsLoad { access } => {
            let addr = e.arg(v, 0)?;
            let tu = e.t_u32();
            let mut out = Vec::new();
            for k in 0..access.dwords() {
                let p = match e.lds_slot_ptr(v, k)? {
                    Some(p) => p,
                    None => {
                        let off = e.cu(4 * k);
                        let a = e.iadd(addr, off);
                        e.lds_ptr(a)?
                    }
                };
                out.push(e.load(tu, p));
            }
            let r = match access {
                Access::Dwords(1) => out[0],
                Access::Dwords(_) => {
                    let ty = e.ty(d.ty);
                    e.construct(ty, &out)
                }
                _ => {
                    let three = e.cu(3);
                    let low = e.band(addr, three);
                    let eight = e.cu(8);
                    let sh = e.imul(low, eight);
                    sub_dword(e, out[0], sh, access)
                }
            };
            Ok(Some(r))
        }
        Op::LdsStore { access } => {
            let (addr, value) = (e.arg(v, 0)?, e.arg(v, 1)?);
            let vty = e.arg_ty(v, 1);
            let ex = exec(e)?;
            e.if_then(ex, |e| {
                let r = (|| -> R<()> {
                    match access {
                        Access::Dwords(k) => {
                            for i in 0..k as u32 {
                                let w = component(e, value, vty, i);
                                let p = match e.lds_slot_ptr(v, i)? {
                                    Some(p) => p,
                                    None => {
                                        let off = e.cu(4 * i);
                                        let a = e.iadd(addr, off);
                                        e.lds_ptr(a)?
                                    }
                                };
                                e.store(p, w);
                            }
                        }
                        _ => {
                            let three = e.cu(3);
                            let low = e.band(addr, three);
                            let eight = e.cu(8);
                            let sh = e.imul(low, eight);
                            let field = e.cu(if access.bytes() == 1 { 0xFF } else { 0xFFFF });
                            let val = e.band(value, field);
                            let val = e.shl(val, sh);
                            let m = e.shl(field, sh);
                            let tu = e.t_u32();
                            let nm = e.ins(S::Not, tu, &[m]);
                            let p = match e.lds_slot_ptr(v, 0)? {
                                Some(p) => p,
                                None => e.lds_ptr(addr)?,
                            };
                            if e.lds_class() == SC::Workgroup {
                                let (scope, sem) = (e.cu(SCOPE_WORKGROUP), e.cu(0));
                                e.ins(S::AtomicAnd, tu, &[p, scope, sem, nm]);
                                e.ins(S::AtomicOr, tu, &[p, scope, sem, val]);
                            } else {
                                let old = e.load(tu, p);
                                let keep = e.band(old, nm);
                                let new = e.bor(keep, val);
                                e.store(p, new);
                            }
                        }
                    }
                    Ok(())
                })();
                if let Err(m) = r {
                    e.res_error(m);
                }
            });
            e.take_error()?;
            Ok(None)
        }
        Op::LdsAtomic { op } => {
            let (addr, data) = (e.arg(v, 0)?, e.arg(v, 1)?);
            let cmp = if n > 3 { Some(e.arg(v, 2)?) } else { None };
            if e.lds_class() != SC::Workgroup {
                if e.arg_ty(v, 1) == Type::U64 {
                    return Err("64-bit LDS atomics in a graphics stage".into());
                }
                // Each invocation has an LDS of its own: a plain
                // read-modify-write of it.
                let ex = exec(e)?;
                let tu = e.t_u32();
                let zero = e.cu(0);
                let p = e.lds_ptr(addr)?;
                let r = e.if_value(ex, tu, zero, |e| {
                    let old = e.load(tu, p);
                    let new = atomic_next(e, op, old, data, cmp);
                    e.store(p, new);
                    old
                });
                return Ok(Some(r));
            }
            if e.arg_ty(v, 1) == Type::U64 {
                let ex = exec(e)?;
                let t64 = e.ty(Type::U64);
                let zero = e.c64(0);
                let p = e.lds_ptr64(addr)?;
                // An operand outside the allocated LDS is skipped.
                let (_, inside) = e.lds_qword(addr);
                let ex = e.land(ex, inside);
                let r = e.if_value(ex, t64, zero, |e| atomic64(e, p, op, data, cmp, SCOPE_WORKGROUP));
                return Ok(Some(r));
            }
            let ex = exec(e)?;
            let tu = e.t_u32();
            let zero = e.cu(0);
            let p = e.lds_ptr(addr)?;
            // An operand outside the allocated LDS is skipped (the address
            // does not wrap).
            let inside = e.lds_dword_inside(addr);
            let ex = e.land(ex, inside);
            let r = e.if_value(ex, tu, zero, |e| atomic(e, p, op, data, cmp, SCOPE_WORKGROUP));
            Ok(Some(r))
        }
        Op::LdsAppend { consume } => {
            let (addr, count) = (e.arg(v, 0)?, e.arg(v, 1)?);
            let tu = e.t_u32();
            if e.lds_class() != SC::Workgroup {
                // An LDS of each invocation's own: it counts the wave's
                // entries in its copy.
                let p = e.lds_ptr(addr)?;
                let old = e.load(tu, p);
                let new = if consume { e.isub(old, count) } else { e.iadd(old, count) };
                e.store(p, new);
                return Ok(Some(old));
            }
            let leader = AppendLeader::new(e);
            let zero = e.cu(0);
            let p = e.lds_ptr(addr)?;
            let old = e.if_value(leader.first, tu, zero, |e| {
                let (sc, sem) = (e.cu(SCOPE_WORKGROUP), e.cu(0));
                e.ins(if consume { S::AtomicISub } else { S::AtomicIAdd }, tu, &[p, sc, sem, count])
            });
            Ok(Some(leader.broadcast(e, old)))
        }
        Op::ScratchLoad { access, .. } => {
            let off = e.arg(v, 0)?;
            let tu = e.t_u32();
            let mut out = Vec::new();
            for k in 0..access.dwords() {
                let o = e.cu(4 * k);
                let a = e.iadd(off, o);
                let p = e.scratch_ptr(a)?;
                out.push(e.load(tu, p));
            }
            let r = match access {
                Access::Dwords(1) => out[0],
                Access::Dwords(_) => {
                    let ty = e.ty(d.ty);
                    e.construct(ty, &out)
                }
                _ => {
                    let three = e.cu(3);
                    let low = e.band(off, three);
                    let eight = e.cu(8);
                    let sh = e.imul(low, eight);
                    sub_dword(e, out[0], sh, access)
                }
            };
            Ok(Some(r))
        }
        Op::ScratchStore { access, .. } => {
            let (off, value) = (e.arg(v, 0)?, e.arg(v, 1)?);
            let vty = e.arg_ty(v, 1);
            let ex = exec(e)?;
            e.if_then(ex, |e| {
                let r = (|| -> R<()> {
                    match access {
                        Access::Dwords(k) => {
                            for i in 0..k as u32 {
                                let w = component(e, value, vty, i);
                                let o = e.cu(4 * i);
                                let a = e.iadd(off, o);
                                let p = e.scratch_ptr(a)?;
                                e.store(p, w);
                            }
                        }
                        _ => {
                            let tu = e.t_u32();
                            let p = e.scratch_ptr(off)?;
                            let old = e.load(tu, p);
                            let three = e.cu(3);
                            let low = e.band(off, three);
                            let eight = e.cu(8);
                            let sh = e.imul(low, eight);
                            let bits = e.cu(access.bytes() * 8);
                            let r = e.ins(S::BitFieldInsert, tu, &[old, value, sh, bits]);
                            e.store(p, r);
                        }
                    }
                    Ok(())
                })();
                if let Err(m) = r {
                    e.res_error(m);
                }
            });
            e.take_error()?;
            Ok(None)
        }
        Op::GdsLoad { access } => {
            let addr = e.arg(v, 0)?;
            let tu = e.t_u32();
            let mut out = Vec::new();
            for k in 0..access.dwords() {
                let p = e.gds_ptr(addr, k)?;
                out.push(e.load(tu, p));
            }
            let r = match access {
                Access::Dwords(1) => out[0],
                Access::Dwords(_) => {
                    let ty = e.ty(d.ty);
                    e.construct(ty, &out)
                }
                _ => {
                    let three = e.cu(3);
                    let low = e.band(addr, three);
                    let eight = e.cu(8);
                    let sh = e.imul(low, eight);
                    sub_dword(e, out[0], sh, access)
                }
            };
            Ok(Some(r))
        }
        Op::GdsStore { access } => {
            let (addr, value) = (e.arg(v, 0)?, e.arg(v, 1)?);
            let vty = e.arg_ty(v, 1);
            let ex = exec(e)?;
            e.if_then(ex, |e| {
                let r = (|| -> R<()> {
                    match access {
                        Access::Dwords(k) => {
                            for i in 0..k as u32 {
                                let w = component(e, value, vty, i);
                                let p = e.gds_ptr(addr, i)?;
                                e.store(p, w);
                            }
                        }
                        _ => {
                            let three = e.cu(3);
                            let low = e.band(addr, three);
                            let eight = e.cu(8);
                            let sh = e.imul(low, eight);
                            let field = e.cu(if access.bytes() == 1 { 0xFF } else { 0xFFFF });
                            let val = e.band(value, field);
                            let val = e.shl(val, sh);
                            let m = e.shl(field, sh);
                            let tu = e.t_u32();
                            let nm = e.ins(S::Not, tu, &[m]);
                            let p = e.gds_ptr(addr, 0)?;
                            let (scope, sem) = (e.cu(SCOPE_DEVICE), e.cu(0));
                            e.ins(S::AtomicAnd, tu, &[p, scope, sem, nm]);
                            e.ins(S::AtomicOr, tu, &[p, scope, sem, val]);
                        }
                    }
                    Ok(())
                })();
                if let Err(m) = r {
                    e.res_error(m);
                }
            });
            e.take_error()?;
            Ok(None)
        }
        Op::GdsAtomic { op } => {
            let (addr, data) = (e.arg(v, 0)?, e.arg(v, 1)?);
            let cmp = if n > 3 { Some(e.arg(v, 2)?) } else { None };
            let ex = exec(e)?;
            if e.arg_ty(v, 1) == Type::U64 {
                let t64 = e.ty(Type::U64);
                let zero = e.c64(0);
                let p = e.gds_ptr64(addr)?;
                let r = e.if_value(ex, t64, zero, |e| atomic64(e, p, op, data, cmp, SCOPE_DEVICE));
                return Ok(Some(r));
            }
            let tu = e.t_u32();
            let zero = e.cu(0);
            let p = e.gds_ptr(addr, 0)?;
            let r = e.if_value(ex, tu, zero, |e| atomic(e, p, op, data, cmp, SCOPE_DEVICE));
            Ok(Some(r))
        }
        Op::GdsAppend { consume } => {
            let (addr, count) = (e.arg(v, 0)?, e.arg(v, 1)?);
            let tu = e.t_u32();
            let leader = AppendLeader::new(e);
            let zero = e.cu(0);
            // The counter is addressed without the 64 KiB wrap: an address
            // past the 64 KiB GDS (an invalid M0 region) leaves it alone.
            let limit = e.cu(GDS_BYTES);
            let inside = e.ult(addr, limit);
            let go = e.land(leader.first, inside);
            let p = e.gds_ptr(addr, 0)?;
            let old = e.if_value(go, tu, zero, |e| {
                let (sc, sem) = (e.cu(SCOPE_DEVICE), e.cu(0));
                e.ins(if consume { S::AtomicISub } else { S::AtomicIAdd }, tu, &[p, sc, sem, count])
            });
            Ok(Some(leader.broadcast(e, old)))
        }
        other => Err(format!("{} is not a memory operation", other.name())),
    }
}

/// The invocation of the host subgroup that performs a wave-wide append
/// or consume: the counter changes once per guest wave and every lane
/// receives the old value.
///
/// In a pixel shader the subgroup also holds helper invocations, whose
/// atomics have no effect and return undefined values: the leader is the
/// lowest invocation that is not a helper (none in a subgroup of helpers,
/// whose guest exec is empty), and its result is read back from it.
struct AppendLeader {
    first: Word,
    /// The leader's subgroup invocation (pixel shaders).
    lane: Option<Word>,
}

impl AppendLeader {
    fn new(e: &mut Emitter) -> AppendLeader {
        e.cap(Capability::GroupNonUniform);
        e.cap(Capability::GroupNonUniformBallot);
        let scope = e.cu(super::emit::SCOPE_SUBGROUP);
        let tb = e.t_bool();
        if e.stage != ShaderStage::Pixel {
            let first = e.ins(S::GroupNonUniformElect, tb, &[scope]);
            return AppendLeader { first, lane: None };
        }
        // Evaluated here, not in the prologue: a lane demoted earlier is a
        // helper by now.
        let helper = if e.feat.demote {
            e.cap(Capability::DemoteToHelperInvocation);
            e.exts.insert("SPV_EXT_demote_to_helper_invocation");
            e.ins(S::IsHelperInvocationEXT, tb, &[])
        } else {
            e.builtin_value(rspirv::spirv::BuiltIn::HelperInvocation, tb)
        };
        let live = e.lnot(helper);
        let t4 = e.t_uvec(4);
        let ballot = e.ins(S::GroupNonUniformBallot, t4, &[scope, live]);
        let tu = e.t_u32();
        let lane = e.ins(S::GroupNonUniformBallotFindLSB, tu, &[scope, ballot]);
        let me = e.host_lane();
        let first = e.ieq(me, lane);
        AppendLeader { first, lane: Some(lane) }
    }

    /// The leader's `old` in every invocation.
    fn broadcast(&self, e: &mut Emitter, old: Word) -> Word {
        match self.lane {
            Some(lane) => e.shuffle(Type::Scalar(Scalar::U32), old, lane),
            None => {
                let tu = e.t_u32();
                let scope = e.cu(super::emit::SCOPE_SUBGROUP);
                e.ins(S::GroupNonUniformBroadcastFirst, tu, &[scope, old])
            }
        }
    }
}

/// Scalar loads (`s_load_dword*`, `s_buffer_load_dword*`) of one
/// instruction: their dwords lie at constant offsets from one base, so
/// they are emitted together with one range check or address translation
/// (see [`group_load`]).
#[derive(Default)]
pub struct LoadGroups {
    of: HashMap<Value, usize>,
    groups: Vec<LoadGroup>,
}

#[derive(Clone)]
struct LoadGroup {
    /// Members and their byte offsets from `base`.
    members: Vec<(Value, u32)>,
    /// The base address (u64, `s_load`) or offset (u32, `s_buffer_load`);
    /// None: the offsets are absolute.
    base: Option<Value>,
    /// Byte offsets of the first dword and past the last.
    lo: u32,
    hi: u32,
    /// `s_buffer_load` (else `s_load`).
    buffer: bool,
}

/// Largest span of a load group in bytes (an `s_load_dwordx16`).
const GROUP_SPAN: u32 = 64;

/// `v` as `base + constant` (constant offsets of IAdd chains; None: a
/// constant).
fn split_offset(f: &Function, mut v: Value, wide: bool) -> (Option<Value>, u64) {
    let mut c = 0u64;
    let mask = if wide { u64::MAX } else { 0xFFFF_FFFF };
    for _ in 0..8 {
        let d = f.value(v);
        match d.op {
            Op::Const(k) => return (None, c.wrapping_add(k) & mask),
            Op::IAdd => {
                let (a, b) = (d.args[0], d.args[1]);
                match (f.op(a), f.op(b)) {
                    (_, Op::Const(k)) => {
                        c = c.wrapping_add(k) & mask;
                        v = a;
                    }
                    (Op::Const(k), _) => {
                        c = c.wrapping_add(k) & mask;
                        v = b;
                    }
                    _ => break,
                }
            }
            _ => break,
        }
    }
    (Some(v), c)
}

impl LoadGroups {
    fn build(e: &Emitter) -> LoadGroups {
        let f = e.f;
        let mut out = LoadGroups::default();
        for &b in &f.order {
            let insts = &f.block(b).insts;
            // (pc, buffer, descriptor, base) -> (member, offset, position)
            let mut by: HashMap<(u32, bool, Option<Value>, Option<Value>), Vec<(Value, u64, usize)>> = HashMap::new();
            for (pos, &v) in insts.iter().enumerate() {
                let d = f.value(v);
                let (pc, buffer) = match d.op {
                    Op::ScalarLoad { pc, .. } => (pc, false),
                    Op::ScalarBufferLoad { pc, .. } => (pc, true),
                    _ => continue,
                };
                let binding = e.uses(v).and_then(|u| u.iter().find(|u| u.operand == ResourceOperand::Resource)).map(|u| u.binding);
                if matches!(binding, Some(ResourceBinding::Suppressed | ResourceBinding::FlattenedSlot(_))) {
                    continue;
                }
                let (desc, at) = if buffer { (Some(d.args[0]), d.args[1]) } else { (None, d.args[0]) };
                let (base, c) = split_offset(f, at, !buffer);
                by.entry((pc, buffer, desc, base)).or_default().push((v, c, pos));
            }
            for ((_, buffer, _, base), list) in by {
                if list.len() < 2 || list.iter().any(|&(_, c, _)| c % 4 != 0 || c > u32::MAX as u64) {
                    continue;
                }
                let lo = list.iter().map(|m| m.1).min().unwrap() as u32;
                let hi = list.iter().map(|m| m.1).max().unwrap() as u32 + 4;
                if hi - lo > GROUP_SPAN {
                    continue;
                }
                // Every member reads through the same resource.
                let first = list[0].0;
                if buffer && list.iter().any(|m| e.buffer_resource(m.0) != e.buffer_resource(first)) {
                    continue;
                }
                // Nothing between the members writes memory (they are the
                // dwords of one instruction).
                let (p0, p1) = (list.iter().map(|m| m.2).min().unwrap(), list.iter().map(|m| m.2).max().unwrap());
                if insts[p0..=p1].iter().any(|&x| f.op(x).has_side_effects()) {
                    continue;
                }
                let index = out.groups.len();
                for m in &list {
                    out.of.insert(m.0, index);
                }
                out.groups.push(LoadGroup { members: list.iter().map(|m| (m.0, m.1 as u32)).collect(), base, lo, hi, buffer });
            }
        }
        out
    }
}

/// The value of scalar load `v` when it belongs to a load group: the
/// first member reached emits every member's load (one range check or
/// address translation for the group, the per-dword guest rules kept);
/// the later ones take their results.
fn group_load(e: &mut Emitter, v: Value) -> R<Option<Word>> {
    if let Some(&w) = e.group_results.get(&(v, e.half)) {
        return Ok(Some(w));
    }
    if e.load_groups.is_none() {
        e.load_groups = Some(LoadGroups::build(e));
    }
    let groups = e.load_groups.as_ref().unwrap();
    let Some(&index) = groups.of.get(&v) else { return Ok(None) };
    let g = groups.groups[index].clone();
    let n = (g.hi - g.lo) / 4;
    let words = if g.buffer { buffer_group(e, v, &g, n)? } else { scalar_group(e, &g, n)? };
    let half = e.half;
    for (&(m, _), w) in g.members.iter().zip(words) {
        e.group_results.insert((m, half), w);
    }
    Ok(e.group_results.get(&(v, half)).copied())
}

/// The members of an `s_load` group: device-address dwords from
/// `base + lo` (each as [`Emitter::device_load`] reads it).
fn scalar_group(e: &mut Emitter, g: &LoadGroup, n: u32) -> R<Vec<Word>> {
    let t64 = e.ty(Type::U64);
    let base = match g.base {
        Some(b) => e.val(b)?,
        None => e.c64(0),
    };
    let lo = e.c64(g.lo as u64);
    let a0 = e.ins(S::IAdd, t64, &[base, lo]);
    let words = e.device_load_span(a0, n)?;
    Ok(g.members.iter().map(|&(_, c)| words[((c - g.lo) / 4) as usize]).collect())
}

/// The members of an `s_buffer_load` group (`first` is the member being
/// emitted): each in range as its own dword (the rule of the single
/// `ScalarBufferLoad`), the host words read with one check.
fn buffer_group(e: &mut Emitter, first: Value, g: &LoadGroup, n: u32) -> R<Vec<Word>> {
    let desc = e.arg(first, 0)?;
    let t = e.buffer_target(first, desc)?;
    let tu = e.t_u32();
    let t64 = e.ty(Type::U64);
    let base = match g.base {
        Some(b) => e.val(b)?,
        None => e.cu(0),
    };
    let w1 = e.extract(tu, desc, 1);
    let stride = e.bits(w1, 16, 14);
    let records = e.extract(tu, desc, 2);
    let zero = e.cu(0);
    let s0 = e.ieq(stride, zero);
    let (s64, r64) = (e.ins(S::UConvert, t64, &[stride]), e.ins(S::UConvert, t64, &[records]));
    let prod = e.ins(S::IMul, t64, &[s64, r64]);
    let size = e.select(t64, s0, r64, prod);
    let tb = e.t_bool();
    let three = e.cu(3);
    let four = e.c64(4);
    let offset = |e: &mut Emitter, c: u32| {
        if c == 0 {
            base
        } else {
            let k = e.cu(c);
            e.iadd(base, k)
        }
    };
    // Per member: in range when the dword lies below the buffer size and
    // the offset is aligned.
    let mut ok = Vec::with_capacity(g.members.len());
    for &(_, c) in &g.members {
        let o = offset(e, c);
        let o64 = e.ins(S::UConvert, t64, &[o]);
        let end = e.ins(S::IAdd, t64, &[o64, four]);
        let inside = e.ins(S::ULessThanEqual, tb, &[end, size]);
        let low = e.band(o, three);
        let aligned = e.ieq(low, zero);
        ok.push(e.land(inside, aligned));
    }
    let words = match t {
        Target::Bound { element, adjust, el } => {
            let two = e.cu(2);
            let indices: Vec<Word> = (0..n)
                .map(|j| {
                    let o = offset(e, g.lo + 4 * j);
                    let b = e.iadd(o, adjust);
                    e.shr(b, two)
                })
                .collect();
            e.bound_words(element, el, indices[0], &indices)?
        }
        Target::Device { base: address } => {
            // Contiguous while the 32-bit offsets do not wrap.
            let o_lo = offset(e, g.lo);
            let limit = e.cu(u32::MAX - 4 * (n - 1));
            let no_wrap = e.ule(o_lo, limit);
            let err = std::cell::RefCell::new(None::<String>);
            let words = e.if_else_values(
                no_wrap,
                tu,
                n as usize,
                |e| {
                    let o64 = e.ins(S::UConvert, t64, &[o_lo]);
                    let a = e.ins(S::IAdd, t64, &[address, o64]);
                    let m = e.c64(!3);
                    let a0 = e.ins(S::BitwiseAnd, t64, &[a, m]);
                    e.device_load_span(a0, n).unwrap_or_else(|m| {
                        err.borrow_mut().get_or_insert(m);
                        vec![e.cu(0); n as usize]
                    })
                },
                |e| {
                    (0..n)
                        .map(|j| {
                            let o = offset(e, g.lo + 4 * j);
                            e.target_load(t, o).unwrap_or_else(|m| {
                                err.borrow_mut().get_or_insert(m);
                                e.cu(0)
                            })
                        })
                        .collect()
                },
            );
            if let Some(m) = err.into_inner() {
                return Err(m);
            }
            words
        }
    };
    Ok(g.members.iter().zip(ok).map(|(&(_, c), ok)| e.select(tu, ok, words[((c - g.lo) / 4) as usize], zero)).collect())
}

/// Component `i` of a u32 / u32xN value.
/// Bytes of the GDS buffer every pipeline binds.
pub const GDS_BYTES: u32 = 64 << 10;

fn component(e: &mut Emitter, value: Word, ty: Type, i: u32) -> Word {
    if ty.is_vector() {
        let tu = e.t_u32();
        e.extract(tu, value, i)
    } else {
        value
    }
}

/// The id of a result that stands in after an error raised inside a
/// structured-helper closure (the module is discarded).
fn guarded(e: &mut Emitter, r: R<Word>) -> Word {
    match r {
        Ok(w) => w,
        Err(m) => {
            e.res_error(m);
            e.cu(0)
        }
    }
}

/// The zero value of a u32 or u32xN result.
fn zero_of(e: &mut Emitter, ty: Type) -> Word {
    let z = e.cu(0);
    if ty.is_vector() {
        let parts = vec![z; ty.components() as usize];
        e.const_vec(ty, &parts)
    } else {
        z
    }
}

/// A device-address load of `access` at `addr` (u64).
fn global_load(e: &mut Emitter, d: &ValueData, addr: Word, access: Access) -> R<Word> {
    let t = Target::Device { base: addr };
    let mut out = Vec::new();
    if e.feat.emit.device_once && !e.volatile && matches!(access, Access::Dwords(k) if k >= 2) {
        // Dword k is at (addr + 4k) & ~3 = (addr & ~3) + 4k.
        let t64 = e.ty(Type::U64);
        let m = e.c64(!3);
        let a0 = e.ins(S::BitwiseAnd, t64, &[addr, m]);
        out = e.device_load_span(a0, access.dwords())?;
    } else {
        for k in 0..access.dwords() {
            let off = e.cu(4 * k);
            out.push(e.target_load(t, off)?);
        }
    }
    Ok(match access {
        Access::Dwords(_) => {
            if out.len() == 1 {
                out[0]
            } else {
                let ty = e.ty(d.ty);
                e.construct(ty, &out)
            }
        }
        _ => {
            let (lo, _) = e.split64(addr);
            let three = e.cu(3);
            let low = e.band(lo, three);
            let eight = e.cu(8);
            let sh = e.imul(low, eight);
            sub_dword(e, out[0], sh, access)
        }
    })
}

/// A device-address store of `access` at `addr` (u64).
fn global_store(e: &mut Emitter, addr: Word, access: Access, value: Word, vty: Type) -> R<()> {
    let t = Target::Device { base: addr };
    match access {
        Access::Dwords(k) if k >= 2 && e.feat.emit.device_once => {
            let values: Vec<Word> = (0..k as u32).map(|i| component(e, value, vty, i)).collect();
            let t64 = e.ty(Type::U64);
            let m = e.c64(!3);
            let a0 = e.ins(S::BitwiseAnd, t64, &[addr, m]);
            e.device_store_span(a0, &values)?;
        }
        Access::Dwords(k) => {
            for i in 0..k as u32 {
                let w = component(e, value, vty, i);
                let off = e.cu(4 * i);
                e.target_store(t, off, w)?;
            }
        }
        _ => {
            let zero = e.cu(0);
            e.target_store_sub(t, zero, value, access.bytes())?;
        }
    }
    Ok(())
}

/// A device-address atomic at `addr` (u64); returns the old value.
fn global_atomic(e: &mut Emitter, addr: Word, op: AtomicOp, data: Word, cmp: Option<Word>, wide: bool) -> R<Word> {
    let t = Target::Device { base: addr };
    let z = e.cu(0);
    if wide {
        let zero = e.c64(0);
        return Ok(e.target_atomic64(t, z, |e, p| atomic64(e, p, op, data, cmp, SCOPE_DEVICE)).unwrap_or(zero));
    }
    Ok(e.target_atomic(t, z, |e, p| atomic(e, p, op, data, cmp, SCOPE_DEVICE)).unwrap_or(z))
}

/// The value of a FLAT access of a compute program: through the shared
/// aperture (LDS) or the private one (scratch) at the address's low dword,
/// otherwise `global` at the device address.
fn route_flat(
    e: &mut Emitter,
    addr: Word,
    ty: Word,
    aperture: impl Fn(&mut Emitter, Aperture, Word) -> R<Word>,
    global: impl FnOnce(&mut Emitter) -> R<Word>,
) -> R<Word> {
    let (low, shared, private) = e.split_aperture(addr);
    let r = e.if_else_value(
        shared,
        ty,
        |e| {
            let r = aperture(e, Aperture::Lds, low);
            guarded(e, r)
        },
        |e| {
            e.if_else_value(
                private,
                ty,
                |e| {
                    let r = aperture(e, Aperture::Scratch, low);
                    guarded(e, r)
                },
                |e| {
                    let r = global(e);
                    guarded(e, r)
                },
            )
        },
    );
    e.take_error()?;
    Ok(r)
}

/// A FLAT store of a compute program, routed like [`route_flat`].
fn route_flat_store(
    e: &mut Emitter,
    addr: Word,
    aperture: impl Fn(&mut Emitter, Aperture, Word) -> R<()>,
    global: impl FnOnce(&mut Emitter) -> R<()>,
) -> R<()> {
    let (low, shared, private) = e.split_aperture(addr);
    e.if_then(shared, |e| {
        if let Err(m) = aperture(e, Aperture::Lds, low) {
            e.res_error(m);
        }
    });
    e.if_then(private, |e| {
        if let Err(m) = aperture(e, Aperture::Scratch, low) {
            e.res_error(m);
        }
    });
    let local = e.lor(shared, private);
    let device = e.lnot(local);
    e.if_then(device, |e| {
        if let Err(m) = global(e) {
            e.res_error(m);
        }
    });
    e.take_error()
}

/// A FLAT load of `access` through an aperture at byte `offset`: each dword
/// of a wider access is its own element.
fn aperture_load_access(e: &mut Emitter, a: Aperture, offset: Word, access: Access, ty: Type) -> R<Word> {
    match access {
        Access::Dwords(k) => {
            let mut out = Vec::with_capacity(k as usize);
            for i in 0..k as u32 {
                let o = if i == 0 {
                    offset
                } else {
                    let c = e.cu(4 * i);
                    e.iadd(offset, c)
                };
                out.push(e.aperture_load(a, o, 4)?);
            }
            if out.len() == 1 {
                Ok(out[0])
            } else {
                let t = e.ty(ty);
                Ok(e.construct(t, &out))
            }
        }
        _ => {
            let raw = e.aperture_load(a, offset, access.bytes())?;
            let zero = e.cu(0);
            Ok(sub_dword(e, raw, zero, access))
        }
    }
}

/// A FLAT store of `access` through an aperture at byte `offset`.
fn aperture_store_access(e: &mut Emitter, a: Aperture, offset: Word, access: Access, value: Word, vty: Type) -> R<()> {
    match access {
        Access::Dwords(k) => {
            for i in 0..k as u32 {
                let w = component(e, value, vty, i);
                let o = if i == 0 {
                    offset
                } else {
                    let c = e.cu(4 * i);
                    e.iadd(offset, c)
                };
                e.aperture_store(a, o, 4, w)?;
            }
            Ok(())
        }
        _ => e.aperture_store(a, offset, access.bytes(), value),
    }
}

/// A byte or short of `word` at bit `shift`, extended to u32.
fn sub_dword(e: &mut Emitter, word: Word, shift: Word, access: Access) -> Word {
    let tu = e.t_u32();
    let bits = e.cu(access.bytes() * 8);
    match access {
        Access::I8 | Access::I16 => e.ins(S::BitFieldSExtract, tu, &[word, shift, bits]),
        _ => e.ins(S::BitFieldUExtract, tu, &[word, shift, bits]),
    }
}

/// Whether the 8-byte pair at dword unit 0 of `a` is in range and 8-byte
/// aligned in its target; returns (that, the byte offset of the pair).
fn whole_pair(e: &mut Emitter, t: Target, a: &BufferAddress) -> (Word, Word) {
    let (byte, ok0) = e.buffer_unit(a, 0, 4);
    let ok1 = e.buffer_unit_ok(a, 4, 4);
    let in_range = e.land(ok0, ok1);
    let low = match t {
        Target::Bound { adjust, .. } => e.iadd(byte, adjust),
        Target::Device { base } => {
            let (lo, _) = e.split64(base);
            e.iadd(lo, byte)
        }
    };
    let seven = e.cu(7);
    let rem = e.band(low, seven);
    let zero = e.cu(0);
    let aligned = e.ieq(rem, zero);
    (e.land(in_range, aligned), byte)
}

/// A GLC `buffer_store_dwordx2` is one 64-bit write on the hardware, so a
/// work-group polling a {flag, value} pair never sees the flag without its
/// value: an 8-byte aligned, in-range pair is written with one 64-bit
/// atomic store; any other pair is two dword stores.
fn coherent_pair_store(e: &mut Emitter, t: Target, a: &BufferAddress, value: Word, vty: Type) -> R<()> {
    let (whole, byte) = whole_pair(e, t, a);
    let lo = component(e, value, vty, 0);
    let hi = component(e, value, vty, 1);
    let mut err = None;
    let split = e.lnot(whole);
    e.if_then(whole, |e| {
        let q = e.join64(lo, hi);
        let r = e.target_atomic64(t, byte, |e, p| {
            e.cap(Capability::Int64Atomics);
            let (sc, sem) = (e.cu(SCOPE_DEVICE), e.cu(0));
            e.void(S::AtomicStore, vec![Operand::IdRef(p), Operand::IdRef(sc), Operand::IdRef(sem), Operand::IdRef(q)]);
            e.c64(0)
        });
        if let Err(m) = r {
            err = Some(m);
        }
    });
    e.if_then(split, |e| {
        if let Err(m) = raw_store_dwords(e, t, a, 2, value, vty) {
            err = Some(m);
        }
    });
    err.map_or(Ok(()), Err)
}

/// `k` dwords of a raw store, one target store each.
fn raw_store_dwords(e: &mut Emitter, t: Target, a: &BufferAddress, k: u32, value: Word, vty: Type) -> R<()> {
    for i in 0..k {
        let (byte, ok) = e.buffer_unit(a, 4 * i, 4);
        let w = component(e, value, vty, i);
        let mut err = None;
        e.if_then(ok, |e| {
            if let Err(m) = e.target_store(t, byte, w) {
                err = Some(m);
            }
        });
        if let Some(m) = err {
            return Err(m);
        }
    }
    Ok(())
}

/// A GLC `buffer_load_dwordx2` is one 64-bit access on the hardware, so it
/// never sees half of a `buffer_atomic_*_x2` or of a coherent pair store:
/// an 8-byte aligned, in-range pair is read with one 64-bit atomic load;
/// any other pair is two dword loads.
fn coherent_pair_load(e: &mut Emitter, t: Target, a: &BufferAddress, ty: Type) -> R<Word> {
    let (whole, byte) = whole_pair(e, t, a);
    let tv = e.ty(ty);
    let r = e.if_else_value(
        whole,
        tv,
        |e| {
            let t64 = e.ty(Type::U64);
            let q = e.target_atomic64(t, byte, |e, p| {
                e.cap(Capability::Int64Atomics);
                let (sc, sem) = (e.cu(SCOPE_DEVICE), e.cu(0));
                e.ins(S::AtomicLoad, t64, &[p, sc, sem])
            });
            let q = guarded(e, q);
            let (lo, hi) = e.split64(q);
            e.construct(tv, &[lo, hi])
        },
        |e| {
            let r = raw_load_dwords(e, t, a, 2, ty);
            guarded(e, r)
        },
    );
    e.take_error()?;
    Ok(r)
}

/// `k` dwords of a raw load, one target load each.
fn raw_load_dwords(e: &mut Emitter, t: Target, a: &BufferAddress, k: u32, ty: Type) -> R<Word> {
    let tu = e.t_u32();
    let zero = e.cu(0);
    let mut out = Vec::new();
    for i in 0..k {
        let (byte, ok) = e.buffer_unit(a, 4 * i, 4);
        let w = e.target_load(t, byte)?;
        out.push(e.select(tu, ok, w, zero));
    }
    if out.len() == 1 {
        Ok(out[0])
    } else {
        let t = e.ty(ty);
        Ok(e.construct(t, &out))
    }
}

fn raw_load(e: &mut Emitter, t: Target, a: &BufferAddress, access: Access, ty: Type) -> R<Word> {
    let tu = e.t_u32();
    let zero = e.cu(0);
    if access == Access::Dwords(2) && e.volatile && coherent_pairs_enabled() {
        return coherent_pair_load(e, t, a, ty);
    }
    match access {
        Access::Dwords(k) if k >= 2 && matches!(t, Target::Bound { .. }) && e.feat.emit.bound_once => {
            // One host range check for the dwords; the guest rules stay per
            // dword.
            let Target::Bound { element, adjust, el } = t else { unreachable!() };
            let units: Vec<(Word, Word)> = (0..k as u32).map(|i| e.buffer_unit(a, 4 * i, 4)).collect();
            let two = e.cu(2);
            let indices: Vec<Word> = units
                .iter()
                .map(|&(byte, _)| {
                    let b = e.iadd(byte, adjust);
                    e.shr(b, two)
                })
                .collect();
            let words = e.bound_words(element, el, indices[0], &indices)?;
            let out: Vec<Word> = words.iter().zip(&units).map(|(&w, &(_, ok))| e.select(tu, ok, w, zero)).collect();
            let t = e.ty(ty);
            Ok(e.construct(t, &out))
        }
        Access::Dwords(k) if k >= 2 && matches!(t, Target::Bound { .. }) && e.feat.emit.const_fold => {
            // Word i is word 0's index plus i in 30 bits: the byte offset
            // of dword i wraps at 2^32 like the guest's (b + 4i, 32-bit),
            // so ((b + 4i) mod 2^32) >> 2 = ((b >> 2) + i) mod 2^30. The
            // wrapped dword can be in the guest range (a structured access
            // checks only its index; a raw one at an offset just below
            // 2^32 of a buffer of 2^32 - 1 bytes), so the wrap is kept.
            let Target::Bound { element, adjust, el } = t else { unreachable!() };
            let b = e.iadd(a.byte, adjust);
            let two = e.cu(2);
            let first = e.shr(b, two);
            let words = e.cu(0x3FFF_FFFF);
            let mut out = Vec::new();
            for i in 0..k as u32 {
                let ok = e.buffer_unit_ok(a, 4 * i, 4);
                let index = if i == 0 {
                    first
                } else {
                    let c = e.cu(i);
                    let next = e.iadd(first, c);
                    e.band(next, words)
                };
                let w = e.bound_target_word(element, el, index)?;
                out.push(e.select(tu, ok, w, zero));
            }
            let t = e.ty(ty);
            Ok(e.construct(t, &out))
        }
        Access::Dwords(k) => {
            let mut out = Vec::new();
            for i in 0..k as u32 {
                let (byte, ok) = e.buffer_unit(a, 4 * i, 4);
                let w = e.target_load(t, byte)?;
                out.push(e.select(tu, ok, w, zero));
            }
            if out.len() == 1 {
                Ok(out[0])
            } else {
                let t = e.ty(ty);
                Ok(e.construct(t, &out))
            }
        }
        _ => {
            let (byte, ok) = e.buffer_unit(a, 0, access.bytes());
            let w = e.target_load(t, byte)?;
            let three = e.cu(3);
            let low = match t {
                Target::Bound { adjust, .. } => {
                    let b = e.iadd(byte, adjust);
                    e.band(b, three)
                }
                Target::Device { base } => {
                    let (lo, _) = e.split64(base);
                    let s = e.iadd(lo, byte);
                    e.band(s, three)
                }
            };
            let eight = e.cu(8);
            let sh = e.imul(low, eight);
            let r = sub_dword(e, w, sh, access);
            Ok(e.select(tu, ok, r, zero))
        }
    }
}

fn raw_store(e: &mut Emitter, t: Target, a: &BufferAddress, access: Access, value: Word, vty: Type) -> R<()> {
    if access == Access::Dwords(2) && e.volatile && coherent_pairs_enabled() {
        return coherent_pair_store(e, t, a, value, vty);
    }
    match access {
        Access::Dwords(k) if matches!(t, Target::Bound { .. }) && e.feat.emit.bound_once => {
            // One branch per dword on the guest rule and the host range
            // together (not three nested ones).
            let Target::Bound { element, adjust, .. } = t else { unreachable!() };
            let len = e.bound_len(element)?;
            let two = e.cu(2);
            for i in 0..k as u32 {
                let (byte, ok) = e.buffer_unit(a, 4 * i, 4);
                let w = component(e, value, vty, i);
                let b = e.iadd(byte, adjust);
                let index = e.shr(b, two);
                let inside = e.ult(index, len);
                let go = e.land(ok, inside);
                let p = e.bound_ptr(element, index)?;
                e.if_then(go, |e| e.store(p, w));
            }
            Ok(())
        }
        Access::Dwords(k) => raw_store_dwords(e, t, a, k as u32, value, vty),
        _ => {
            let (byte, ok) = e.buffer_unit(a, 0, access.bytes());
            let mut err = None;
            e.if_then(ok, |e| {
                if let Err(m) = e.target_store_sub(t, byte, value, access.bytes()) {
                    err = Some(m);
                }
            });
            err.map_or(Ok(()), Err)
        }
    }
}

/// The unified format a formatted access converts with (the MTBUF
/// format, else the specialised descriptor format).
fn access_format(e: &Emitter, v: Value, typed: Option<u32>) -> R<(u32, [u32; 4])> {
    if let Some(f) = typed {
        return Ok((f, [4, 5, 6, 7]));
    }
    let i = e.buffer_resource(v).ok_or("a formatted access through a device address")?;
    let info = e.input.request.resources.info.buffers.get(i as usize).ok_or("buffer specialization is missing")?;
    let s = info.descriptor_swizzle;
    Ok((info.descriptor_format, [s & 7, (s >> 3) & 7, (s >> 6) & 7, (s >> 9) & 7]))
}

#[allow(clippy::too_many_arguments)]
fn format_load(
    e: &mut Emitter,
    v: Value,
    t: Target,
    a: &BufferAddress,
    desc: Word,
    components: u32,
    typed: Option<u32>,
    d16: bool,
    ty: Type,
) -> R<Word> {
    if typed.is_none() && e.buffer_resource(v).is_none() && (components == 1 || !d16) {
        // Each component through its own destination select of the V#.
        // The element's dwords are read once for every format case: read
        // in each of the 128 cases (and each component case inside), a
        // load of five components through a device address was 2.4
        // million SPIR-V words, which the driver took 150 s to compile
        //.
        let pre = match t {
            Target::Device { base } => Some(Preloaded::read(e, base, a)?),
            Target::Bound { .. } => None,
        };
        let words = format_load_runtime(e, t, a, desc, d16, components, pre.as_ref())?;
        return Ok(if words.len() == 1 {
            words[0]
        } else {
            let t = e.ty(ty);
            e.construct(t, &words)
        });
    }
    let (format, dst_sel) = access_format(e, v, typed)?;
    let Some(info) = FormatInfo::new(format) else {
        // An invalid format (0 among them) reads raw dwords.
        return raw_load(e, t, a, Access::Dwords(components as u8), ty);
    };
    let sources: Vec<FormatSource> = (0..components).map(|c| format_source(&info, typed.is_some(), dst_sel, c)).collect();
    format_load_sources(e, t, a, &info, &sources, d16, ty, None)
}

/// The dwords of a formatted element read through a device address before
/// a per-format switch: the dword holding byte `k` of the element is what
/// [`Emitter::target_load`] reads at the element's byte `k` (dword aligned
/// down), for `k < 16`.
struct Preloaded {
    /// The dwords at the element's address aligned down, and four after.
    words: Vec<Word>,
    /// The element address's byte in its dword.
    low: Word,
    /// Whether the bytes (start, count) of the element are in range, for
    /// every component span of every format ([`Emitter::buffer_unit_ok`]).
    ok: HashMap<(u32, u32), Word>,
}

impl Preloaded {
    fn read(e: &mut Emitter, base: Word, a: &BufferAddress) -> R<Preloaded> {
        let byte = a.byte;
        let mut spans: Vec<(u32, u32)> = (0..128).filter_map(FormatInfo::new).flat_map(|info| (0..info.count).map(move |c| info.component_span(c))).collect();
        spans.push((0, 4));
        spans.sort_unstable();
        spans.dedup();
        let ok = spans.into_iter().map(|(b, n)| ((b, n), e.buffer_unit_ok(a, b, n))).collect();
        let t64 = e.ty(Type::U64);
        let o64 = e.ins(S::UConvert, t64, &[byte]);
        let addr = e.ins(S::IAdd, t64, &[base, o64]);
        let m = e.c64(!3);
        let aligned = e.ins(S::BitwiseAnd, t64, &[addr, m]);
        let words = e.device_load_span(aligned, 5)?;
        let (lo, _) = e.split64(addr);
        let three = e.cu(3);
        let low = e.band(lo, three);
        Ok(Preloaded { words, low, ok })
    }

    /// [`Emitter::buffer_unit`] of the access, its range check computed
    /// before the switch.
    fn unit(&self, e: &mut Emitter, a: &BufferAddress, extra: u32, bytes: u32) -> (Word, Word) {
        match self.ok.get(&(extra, bytes)) {
            Some(&ok) => {
                let byte = if extra == 0 {
                    a.byte
                } else {
                    let ex = e.cu(extra);
                    e.iadd(a.byte, ex)
                };
                (byte, ok)
            }
            None => e.buffer_unit(a, extra, bytes),
        }
    }

    /// The dword [`Emitter::target_load`] reads at the element's byte `k`.
    fn word_at(&self, e: &mut Emitter, k: u32) -> Word {
        let first = (k >> 2) as usize;
        if first + 1 >= self.words.len() {
            return self.words[self.words.len() - 1];
        }
        // low + (k & 3) reaches the next dword or not.
        let tu = e.t_u32();
        let rem = e.cu(k & 3);
        let s = e.iadd(self.low, rem);
        let four = e.cu(4);
        let next = e.ule(four, s);
        e.select(tu, next, self.words[first + 1], self.words[first])
    }
}

/// The output components of a formatted load through a descriptor the
/// shader selects at run time (read through its device address): the
/// format and each component's destination select come from the V# word
/// 3. One switch over the 128 formats converts every component of the
/// element once (masked by the element's range check) and gives the
/// format's component count and its constant one; each output component
/// then picks its destination select from those with selects. (A switch
/// per output component with a case per component inside it, every case
/// reading memory, made GTA V's compute program 0x803dd3c00 2.4 million
/// SPIR-V words, 150 s in the driver: GPU_PLAN §76.)
fn format_load_runtime(e: &mut Emitter, t: Target, a: &BufferAddress, desc: Word, d16: bool, components: u32, pre: Option<&Preloaded>) -> R<Vec<Word>> {
    let tu = e.t_u32();
    let t4 = e.ty(Type::U32X4);
    let w3 = e.extract(tu, desc, 3);
    let format = e.bits(w3, 12, 7);
    let mut err = None;
    // The converted components (zero past the format's count); an invalid
    // format: the raw dword 0 of the element in x.
    let values = e.switch_value(format, 128, t4, |e, f| {
        let r = (|| -> R<Word> {
            let zero = e.cu(0);
            let Some(info) = FormatInfo::new(f) else {
                let raw = match pre {
                    Some(p) => {
                        let (_, ok) = p.unit(e, a, 0, 4);
                        let w = p.word_at(e, 0);
                        e.select(tu, ok, w, zero)
                    }
                    None => raw_load(e, t, a, Access::Dwords(1), Type::U32)?,
                };
                return Ok(e.construct(t4, &[raw, zero, zero, zero]));
            };
            let sources: Vec<FormatSource> = (0..4).map(|c| if c < info.count { FormatSource::Memory(c) } else { FormatSource::Zero }).collect();
            format_load_sources(e, t, a, &info, &sources, d16, Type::U32X4, pre)
        })();
        r.unwrap_or_else(|m| {
            err.get_or_insert(m);
            let z = e.cu(0);
            e.construct(t4, &[z, z, z, z])
        })
    });
    if let Some(m) = err {
        return Err(m);
    }
    // (component count, 0 for an invalid format; the constant one).
    let t2 = e.t_uvec(2);
    let meta = e.switch_value(format, 128, t2, |e, f| {
        let (count, one) = match FormatInfo::new(f) {
            None => (0, 0),
            Some(info) => (
                info.count,
                match info.kind {
                    FormatKind::Uint | FormatKind::Sint => 1,
                    _ if d16 => 0x3C00,
                    _ => 0x3F80_0000,
                },
            ),
        };
        let (c, o) = (e.cu(count), e.cu(one));
        e.construct(t2, &[c, o])
    });
    let count = e.extract(tu, meta, 0);
    let one = e.extract(tu, meta, 1);
    let comps: Vec<Word> = (0..4).map(|i| e.extract(tu, values, i)).collect();
    let zero = e.cu(0);
    let invalid = e.ieq(count, zero);
    let c1 = e.cu(1);
    // A zero count would divide by zero below; its result is not used.
    let divisor = e.select(tu, invalid, c1, count);
    let mut out = Vec::with_capacity(components as usize);
    for o in 0..components {
        let sel = e.bits(w3, 3 * o, 3);
        let is_one = e.ieq(sel, c1);
        let constant = e.select(tu, is_one, one, zero);
        let c4 = e.cu(4);
        let memory = e.ule(c4, sel);
        let s = e.isub(sel, c4);
        let comp = e.ins(S::UMod, tu, &[s, divisor]);
        let mut v = comps[0];
        for (i, &c) in comps.iter().enumerate().skip(1) {
            let k = e.cu(i as u32);
            let here = e.ieq(comp, k);
            v = e.select(tu, here, c, v);
        }
        let v = e.select(tu, memory, v, constant);
        out.push(e.select(tu, invalid, comps[0], v));
    }
    Ok(out)
}

/// A formatted load of `info` whose output components come from `sources`
/// (the element's dwords from `pre` when given, else loaded here).
#[allow(clippy::too_many_arguments)]
fn format_load_sources(e: &mut Emitter, t: Target, a: &BufferAddress, info: &FormatInfo, sources: &[FormatSource], d16: bool, ty: Type, pre: Option<&Preloaded>) -> R<Word> {
    let info = *info;
    let tu = e.t_u32();
    let zero = e.cu(0);
    // The element is in range only when every component of it is.
    let mut ok: Option<Word> = None;
    let mut memory: HashMap<u32, Word> = HashMap::new();
    for s in sources {
        if let FormatSource::Memory(c) = *s {
            if memory.contains_key(&c) {
                continue;
            }
            let (byte, bytes) = info.component_span(c);
            let (off, in_range) = match pre {
                Some(p) => p.unit(e, a, byte, bytes),
                None => e.buffer_unit(a, byte, bytes),
            };
            ok = Some(match ok {
                Some(o) => e.land(o, in_range),
                None => in_range,
            });
            let word = match pre {
                Some(p) => p.word_at(e, byte),
                None => e.target_load(t, off)?,
            };
            // Sub-dword components: shift by the byte position.
            let raw = if info.packed || bytes == 4 {
                word
            } else {
                let three = e.cu(3);
                let low = match t {
                    Target::Bound { adjust, .. } => {
                        let b = e.iadd(off, adjust);
                        e.band(b, three)
                    }
                    Target::Device { base } => {
                        let (lo, _) = e.split64(base);
                        let s = e.iadd(lo, off);
                        e.band(s, three)
                    }
                };
                let eight = e.cu(8);
                let sh = e.imul(low, eight);
                e.shr(word, sh)
            };
            let value = convert_load(e, &info, c, raw);
            let value = if d16 {
                let field = load_field(e, &info, c, raw);
                super::convert::load_component_d16(e, &info, c, field, value)
            } else {
                value
            };
            memory.insert(c, value);
        }
    }
    // Format accesses are range-checked all or nothing over the whole
    // element, also the components the access does not return.
    if let Some(o) = ok {
        let mut all = o;
        for c in 0..info.count {
            if !memory.contains_key(&c) {
                let (byte, bytes) = info.component_span(c);
                let (_, in_range) = match pre {
                    Some(p) => p.unit(e, a, byte, bytes),
                    None => e.buffer_unit(a, byte, bytes),
                };
                all = e.land(all, in_range);
            }
        }
        ok = Some(all);
    }
    let mut out = Vec::new();
    for s in sources {
        out.push(match *s {
            FormatSource::Memory(c) => {
                let m = memory[&c];
                let ok = ok.unwrap();
                e.select(tu, ok, m, zero)
            }
            FormatSource::One if matches!(info.kind, FormatKind::Uint | FormatKind::Sint) => e.cu(1),
            FormatSource::One if d16 => e.cu(0x3C00),
            FormatSource::One => e.cu(0x3F80_0000),
            FormatSource::Zero => zero,
        });
    }
    Ok(if out.len() == 1 {
        out[0]
    } else {
        let t = e.ty(ty);
        e.construct(t, &out)
    })
}

/// The field of component `c` whose bits start at bit 0 of `raw` (packed
/// fields at their bit offset), sign-extended for signed formats.
fn load_field(e: &mut Emitter, info: &FormatInfo, c: u32, raw: Word) -> Word {
    let tu = e.t_u32();
    let bits = info.bits[c as usize];
    let shift = if info.packed { info.bit_offset[c as usize] } else { 0 };
    let signed = matches!(info.kind, FormatKind::Snorm | FormatKind::Sscaled | FormatKind::Sint);
    if bits >= 32 {
        raw
    } else {
        let (o, n) = (e.cu(shift), e.cu(bits));
        e.ins(if signed { S::BitFieldSExtract } else { S::BitFieldUExtract }, tu, &[raw, o, n])
    }
}

/// Load conversion of component `c` whose field starts at bit 0 of `raw`
/// (packed fields at their bit offset).
pub(super) fn convert_load(e: &mut Emitter, info: &FormatInfo, c: u32, raw: Word) -> Word {
    let tu = e.t_u32();
    let tf = e.t_f32();
    let bits = info.bits[c as usize];
    let field = load_field(e, info, c, raw);
    match info.kind {
        FormatKind::Uint | FormatKind::Sint => field,
        FormatKind::Uscaled => {
            let f = e.ins(S::ConvertUToF, tf, &[field]);
            e.bitcast(tu, f)
        }
        FormatKind::Sscaled => {
            let f = e.ins(S::ConvertSToF, tf, &[field]);
            e.bitcast(tu, f)
        }
        FormatKind::Unorm => {
            let max = if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 };
            if bits >= 32 && e.feat.no_float64 {
                unorm32_to_f32_bits(e, field)
            } else if bits >= 32 {
                // Exact quotient through f64.
                let t64 = e.ty(Type::F64);
                let f = e.ins(S::ConvertUToF, t64, &[field]);
                let m = e.konst(Type::F64, (max as f64).to_bits());
                let q = e.ins(S::FDiv, t64, &[f, m]);
                let n = e.ins(S::FConvert, tf, &[q]);
                e.bitcast(tu, n)
            } else {
                unorm_to_f32_bits(e, field, bits)
            }
        }
        FormatKind::Snorm => {
            // |x| (the most negative value clamped) over 2^(bits-1) - 1,
            // exactly, with the sign put back.
            let max = ((1u64 << (bits - 1)) - 1) as u32;
            let ti = e.t_i32();
            let tb = e.t_bool();
            let fi = e.bitcast(ti, field);
            let zero = e.ci(0);
            let negative = e.ins(S::SLessThan, tb, &[fi, zero]);
            let neg = e.ins(S::SNegate, tu, &[field]);
            let magnitude = e.select(tu, negative, neg, field);
            let m = e.cu(max);
            let clamped = e.gl(tu, GLOp::UMin, &[magnitude, m]);
            let sign_bit = e.cu(0x8000_0000);
            let plus = e.cu(0);
            let sign = e.select(tu, negative, sign_bit, plus);
            let mag = unorm_to_f32_bits(e, clamped, bits - 1);
            e.bor(sign, mag)
        }
        FormatKind::Float => match bits {
            32 => field,
            16 => e.f16_bits_to_f32(field),
            _ => ufloat_to_f32(e, field, bits),
        },
    }
}

/// An unsigned normalized `bits`-bit value (1..=31) as f32 bits, exactly:
/// x / (2^bits - 1) is the repeating binary fraction of x (normalised by
/// its leading one), rounded to nearest on bit 7 of the pattern.
/// The f32 bits of a 32-bit unorm field `x / (2^32 - 1)`, correctly rounded,
/// with integer code (devices without shaderFloat64). The quotient is
/// `x / 2^32` plus `x / 2^64 + ...`, a positive excess below one unit of
/// `x`: it changes the nearest f32 of `x` only where `x` lies exactly halfway
/// between two (the excess then rounds up instead of to even), and
/// `0xFFFFFFFF` is 1.0.
fn unorm32_to_f32_bits(e: &mut Emitter, x: Word) -> Word {
    let tu = e.t_u32();
    let tf = e.t_f32();
    let tb = e.t_bool();
    let nearest = e.ins(S::ConvertUToF, tf, &[x]);
    let bits = e.bitcast(tu, nearest);
    // Dropped low bits: msb - 23 when x needs more than 24 bits (1..8).
    let msb = e.gl(tu, GLOp::FindUMsb, &[x]);
    let c23 = e.cu(23);
    let excess = e.isub(msb, c23);
    let limit = e.cu(1 << 24);
    let wide = e.ule(limit, x);
    let one = e.cu(1);
    let shift = e.select(tu, wide, excess, one);
    let all = e.shl(one, shift);
    let mask = e.isub(all, one);
    let low = e.band(x, mask);
    let half = e.shr(all, one);
    let tie = e.ins(S::IEqual, tb, &[low, half]);
    let tie = e.ins(S::LogicalAnd, tb, &[tie, wide]);
    let kept = e.shr(x, shift);
    let kept_low = e.band(kept, one);
    let zero = e.cu(0);
    let even = e.ins(S::IEqual, tb, &[kept_low, zero]);
    let bump = e.ins(S::LogicalAnd, tb, &[tie, even]);
    let up = e.iadd(bits, one);
    let rounded = e.select(tu, bump, up, bits);
    // Scale by 2^-32 (exact: x >= 1 stays normal).
    let f = e.bitcast(tf, rounded);
    let scale = e.konst(Type::F32, (2f32.powi(-32)).to_bits() as u64);
    let scaled = e.ins(S::FMul, tf, &[f, scale]);
    let scaled = e.bitcast(tu, scaled);
    let max = e.cu(u32::MAX);
    let is_max = e.ins(S::IEqual, tb, &[x, max]);
    let unit = e.cu(1f32.to_bits());
    e.select(tu, is_max, unit, scaled)
}

/// CPU model of [`unorm32_to_f32_bits`].
#[cfg(test)]
pub(crate) fn unorm32_model(x: u32) -> u32 {
    if x == u32::MAX {
        return 1f32.to_bits();
    }
    let mut bits = (x as f32).to_bits();
    if x >= 1 << 24 {
        let shift = 31 - x.leading_zeros() - 23;
        let low = x & ((1 << shift) - 1);
        if low == 1 << (shift - 1) && (x >> shift) & 1 == 0 {
            bits += 1;
        }
    }
    (f32::from_bits(bits) * 2f32.powi(-32)).to_bits()
}

fn unorm_to_f32_bits(e: &mut Emitter, value: Word, bits: u32) -> Word {
    let tu = e.t_u32();
    let tb = e.t_bool();
    let maximum = (1u32 << bits) - 1;
    let zero = e.cu(0);
    let is_zero = e.ins(S::IEqual, tb, &[value, zero]);
    let one = e.cu(1);
    let safe = e.select(tu, is_zero, one, value);
    let msb = e.gl(tu, GLOp::FindUMsb, &[safe]);
    let top = e.cu(bits - 1);
    let leading = e.ins(S::ISub, tu, &[top, msb]);
    let high = e.ins(S::ShiftLeftLogical, tu, &[safe, leading]);
    let width = e.cu(bits);
    let rest = e.ins(S::ISub, tu, &[width, leading]);
    let low = e.ins(S::ShiftRightLogical, tu, &[safe, rest]);
    let rotated = e.bor(high, low);
    let mask = e.cu(maximum);
    let rotated = e.ins(S::BitwiseAnd, tu, &[rotated, mask]);
    let up = e.cu(32 - bits);
    let mut pattern = e.ins(S::ShiftLeftLogical, tu, &[rotated, up]);
    let mut shift = bits;
    while shift < 32 {
        let s = e.cu(shift);
        let r = e.ins(S::ShiftRightLogical, tu, &[pattern, s]);
        pattern = e.bor(pattern, r);
        shift *= 2;
    }
    let c7 = e.cu(7);
    let r7 = e.ins(S::ShiftRightLogical, tu, &[pattern, c7]);
    let round = e.ins(S::BitwiseAnd, tu, &[r7, one]);
    let c8 = e.cu(8);
    let r8 = e.ins(S::ShiftRightLogical, tu, &[pattern, c8]);
    let rounded = e.ins(S::IAdd, tu, &[r8, round]);
    let bias = e.cu(126 - bits);
    let ex = e.ins(S::IAdd, tu, &[msb, bias]);
    let c23 = e.cu(23);
    let exponent = e.ins(S::ShiftLeftLogical, tu, &[ex, c23]);
    let r = e.ins(S::IAdd, tu, &[exponent, rounded]);
    e.select(tu, is_zero, zero, r)
}

/// Unsigned 11/10-bit float to f32 bits.
fn ufloat_to_f32(e: &mut Emitter, raw: Word, bits: u32) -> Word {
    let tu = e.t_u32();
    let tf = e.t_f32();
    let mb: u32 = if bits == 11 { 6 } else { 5 };
    let mantissa = e.bits(raw, 0, mb);
    let exponent = e.bits(raw, mb, 5);
    let sh = e.cu(23 - mb);
    let wide = e.shl(mantissa, sh);
    // Denormal: mantissa * 2^(-14 - mb).
    let mf = e.ins(S::ConvertUToF, tf, &[mantissa]);
    let scale = e.cf((2.0f32).powi(-14 - mb as i32));
    let den = e.ins(S::FMul, tf, &[mf, scale]);
    let den = e.bitcast(tu, den);
    let inf = e.cu(0x7F80_0000);
    let infv = e.bor(inf, wide);
    let bias = e.cu(127 - 15);
    let eb = e.iadd(exponent, bias);
    let c23 = e.cu(23);
    let es = e.shl(eb, c23);
    let norm = e.bor(es, wide);
    let zero = e.cu(0);
    let c31 = e.cu(31);
    let is0 = e.ieq(exponent, zero);
    let is31 = e.ieq(exponent, c31);
    let r = e.select(tu, is31, infv, norm);
    e.select(tu, is0, den, r)
}

#[allow(clippy::too_many_arguments)]
fn format_store(
    e: &mut Emitter,
    v: Value,
    t: Target,
    a: &BufferAddress,
    _desc: Word,
    components: u32,
    typed: Option<u32>,
    d16: bool,
    value: Word,
    vty: Type,
) -> R<()> {
    let (format, _) = access_format(e, v, typed)?;
    let Some(info) = FormatInfo::new(format) else {
        // An invalid format (0 among them) stores raw dwords.
        return raw_store(e, t, a, Access::Dwords(components as u8), value, vty);
    };
    let tu = e.t_u32();
    let mut ok: Option<Word> = None;
    for c in 0..info.count {
        let (byte, bytes) = info.component_span(c);
        let (_, r) = e.buffer_unit(a, byte, bytes);
        ok = Some(match ok {
            Some(o) => e.land(o, r),
            None => r,
        });
    }
    let ok = ok.unwrap();
    let mut err = None;
    e.if_then(ok, |e| {
        let r = (|| -> R<()> {
            let zero = e.cu(0);
            let enc: Vec<Word> = (0..info.count)
                .map(|c| {
                    if c >= components {
                        return zero;
                    }
                    let d = component(e, value, vty, c);
                    if d16 {
                        super::convert::store_component_d16(e, &info, c, d)
                    } else {
                        super::convert::store_component(e, &info, c, d)
                    }
                })
                .collect();
            if info.packed {
                let mut word = zero;
                for c in 0..info.count as usize {
                    let (o, n) = (e.cu(info.bit_offset[c]), e.cu(info.bits[c]));
                    word = e.ins(S::BitFieldInsert, tu, &[word, enc[c], o, n]);
                }
                let (off, _) = e.buffer_unit(a, 0, 4);
                e.target_store(t, off, word)?;
            } else {
                for c in 0..info.count {
                    let (byte, bytes) = info.component_span(c);
                    let (off, _) = e.buffer_unit(a, byte, bytes);
                    if bytes == 4 {
                        e.target_store(t, off, enc[c as usize])?;
                    } else {
                        e.target_store_sub(t, off, enc[c as usize], bytes)?;
                    }
                }
            }
            Ok(())
        })();
        if let Err(m) = r {
            err = Some(m);
        }
    });
    err.map_or(Ok(()), Err)
}

/// DS_MIN_F32 / DS_MAX_F32 replacement value (see `AtomicOp::DsFMin`).
fn ds_float_min_max(e: &mut Emitter, old: Word, src: Word, max: bool) -> Word {
    let tu = e.t_u32();
    let classify = |e: &mut Emitter, bits: Word| {
        let m = e.cu(0x7FFF_FFFF);
        let abs = e.band(bits, m);
        let inf = e.cu(0x7F80_0000);
        let nan = e.ult(inf, abs);
        let q = e.cu(0x0040_0000);
        let qb = e.band(bits, q);
        let zero = e.cu(0);
        let unquiet = e.ieq(qb, zero);
        let signaling = e.land(nan, unquiet);
        let tiny = e.cu(0x0080_0000);
        let small = e.ult(abs, tiny);
        let sm = e.cu(0x8000_0000);
        let sign = e.band(bits, sm);
        let positive = e.ieq(sign, zero);
        let nonzero = e.ult(zero, abs);
        let positive_denorm = e.land(positive, nonzero);
        let one = e.cu(1);
        let small_bits = e.select(tu, positive_denorm, one, sign);
        let key_bits = e.select(tu, small, small_bits, bits);
        let kneg = e.band(key_bits, sm);
        let negative = e.ult(zero, kneg);
        let inverted = e.ins(S::Not, tu, &[key_bits]);
        let flagged = e.bor(key_bits, sm);
        let key = e.select(tu, negative, inverted, flagged);
        (nan, signaling, key)
    };
    let (lnan, lsig, lkey) = classify(e, old);
    let (rnan, rsig, rkey) = classify(e, src);
    let choose = if max { e.ult(lkey, rkey) } else { e.ule(rkey, lkey) };
    let mut r = e.select(tu, choose, src, old);
    r = e.select(tu, lnan, src, r);
    r = e.select(tu, rnan, old, r);
    let q = e.cu(0x0040_0000);
    let sq = e.bor(src, q);
    r = e.select(tu, rsig, sq, r);
    let oq = e.bor(old, q);
    e.select(tu, lsig, oq, r)
}

/// One atomic on `ptr` (u32); returns the old value.
pub fn atomic(e: &mut Emitter, ptr: Word, op: AtomicOp, data: Word, cmp: Option<Word>, scope: u32) -> Word {
    let tu = e.t_u32();
    let (sc, sem) = (e.cu(scope), e.cu(0));
    let simple = |e: &mut Emitter, o: S| e.ins(o, tu, &[ptr, sc, sem, data]);
    match op {
        AtomicOp::Swap => simple(e, S::AtomicExchange),
        AtomicOp::Add => simple(e, S::AtomicIAdd),
        AtomicOp::Sub => simple(e, S::AtomicISub),
        AtomicOp::SMin => simple(e, S::AtomicSMin),
        AtomicOp::UMin => simple(e, S::AtomicUMin),
        AtomicOp::SMax => simple(e, S::AtomicSMax),
        AtomicOp::UMax => simple(e, S::AtomicUMax),
        AtomicOp::And => simple(e, S::AtomicAnd),
        AtomicOp::Or => simple(e, S::AtomicOr),
        AtomicOp::Xor => simple(e, S::AtomicXor),
        AtomicOp::CmpSwap => {
            let c = cmp.unwrap_or(data);
            e.ins(S::AtomicCompareExchange, tu, &[ptr, sc, sem, sem, data, c])
        }
        other => cas_loop(e, ptr, scope, |e, old| atomic_next(e, other, old, data, cmp)),
    }
}

/// DS_ADD_F32's stored value (see `AtomicOp::DsFAdd`): flushes by the f32
/// denormal field, the host's nearest-even add between them, NaN operands
/// stored quieted (data first) and an invalid sum as 0xffc00000.
fn ds_add_f32(e: &mut Emitter, old: Word, data: Word, flush_inputs: bool, flush_result: bool) -> Word {
    let tu = e.t_u32();
    let tf = e.t_f32();
    let flush = |e: &mut Emitter, x: Word, on: bool| -> Word {
        if !on {
            return x;
        }
        let em = e.cu(0x7F80_0000);
        let exp = e.band(x, em);
        let zero = e.cu(0);
        let tiny = e.ieq(exp, zero);
        let sm = e.cu(0x8000_0000);
        let sign = e.band(x, sm);
        e.select(tu, tiny, sign, x)
    };
    let nan = |e: &mut Emitter, x: Word| -> Word {
        let mm = e.cu(0x7FFF_FFFF);
        let mag = e.band(x, mm);
        let inf = e.cu(0x7F80_0000);
        e.ult(inf, mag)
    };
    let (fo, fd) = (flush(e, old, flush_inputs), flush(e, data, flush_inputs));
    let (fo, fd) = (e.bitcast(tf, fo), e.bitcast(tf, fd));
    let sum = e.ins(S::FAdd, tf, &[fo, fd]);
    e.decorate(sum, rspirv::spirv::Decoration::NoContraction, &[]);
    let bits = e.bitcast(tu, sum);
    let r = flush(e, bits, flush_result);
    let invalid = nan(e, r);
    let default = e.cu(0xFFC0_0000);
    let r = e.select(tu, invalid, default, r);
    let quiet = e.cu(0x0040_0000);
    let old_nan = nan(e, old);
    let quiet_old = e.bor(old, quiet);
    let r = e.select(tu, old_nan, quiet_old, r);
    let data_nan = nan(e, data);
    let quiet_data = e.bor(data, quiet);
    e.select(tu, data_nan, quiet_data, r)
}

/// The value a 32-bit atomic `op` leaves in memory that held `old`.
pub fn atomic_next(e: &mut Emitter, op: AtomicOp, old: Word, data: Word, cmp: Option<Word>) -> Word {
    let tu = e.t_u32();
    let tf = e.t_f32();
    let tb = e.t_bool();
    let pick = |e: &mut Emitter, o: S| {
        let c = e.ins(o, tb, &[old, data]);
        e.select(tu, c, old, data)
    };
    match op {
        AtomicOp::Swap => data,
        AtomicOp::Add => e.iadd(old, data),
        AtomicOp::Sub => e.isub(old, data),
        AtomicOp::SMin => pick(e, S::SLessThan),
        AtomicOp::UMin => pick(e, S::ULessThan),
        AtomicOp::SMax => pick(e, S::SGreaterThan),
        AtomicOp::UMax => pick(e, S::UGreaterThan),
        AtomicOp::And => e.band(old, data),
        AtomicOp::Or => e.bor(old, data),
        AtomicOp::Xor => e.ins(S::BitwiseXor, tu, &[old, data]),
        AtomicOp::CmpSwap => {
            let c = cmp.unwrap_or(data);
            let eq = e.ieq(old, c);
            e.select(tu, eq, data, old)
        }
        AtomicOp::Inc => {
            let one = e.cu(1);
            let ge = e.ule(data, old);
            let inc = e.iadd(old, one);
            let zero = e.cu(0);
            e.select(tu, ge, zero, inc)
        }
        AtomicOp::Dec => {
            let one = e.cu(1);
            let zero = e.cu(0);
            let z = e.ieq(old, zero);
            let gt = e.ult(data, old);
            let wrap = e.lor(z, gt);
            let dec = e.isub(old, one);
            e.select(tu, wrap, data, dec)
        }
        AtomicOp::RSub => e.isub(data, old),
        AtomicOp::CSub => {
            let ge = e.ule(data, old);
            let s = e.isub(old, data);
            let zero = e.cu(0);
            e.select(tu, ge, s, zero)
        }
        AtomicOp::FAdd => {
            let (fo, fd) = (e.bitcast(tf, old), e.bitcast(tf, data));
            let r = e.ins(S::FAdd, tf, &[fo, fd]);
            e.bitcast(tu, r)
        }
        AtomicOp::DsFAdd { flush_inputs, flush_result } => ds_add_f32(e, old, data, flush_inputs, flush_result),
        AtomicOp::FMin | AtomicOp::FMax => float32_min_max(e, old, data, op == AtomicOp::FMax),
        AtomicOp::DsFMin | AtomicOp::DsFMax => ds_float_min_max(e, old, data, op == AtomicOp::DsFMax),
        AtomicOp::MaskOr => {
            // data = mask, cmp = value.
            let value = cmp.unwrap_or(data);
            let nm = e.ins(S::Not, tu, &[data]);
            let keep = e.band(old, nm);
            e.bor(keep, value)
        }
        AtomicOp::Wrap => {
            let add = cmp.unwrap_or(data);
            let ge = e.ule(data, old);
            let s = e.isub(old, data);
            let a = e.iadd(old, add);
            e.select(tu, ge, s, a)
        }
        AtomicOp::FCmpSwap => {
            // IEEE equality on the bits (denormals compare as themselves
            // whatever the module's float controls; +0 == -0; NaN never).
            let _ = tf;
            let c = cmp.unwrap_or(data);
            let same = e.ieq(old, c);
            let magnitude = e.cu(0x7FFF_FFFF);
            let mag = e.band(old, magnitude);
            let inf = e.cu(0x7F80_0000);
            let nan = e.ult(inf, mag);
            let not_nan = e.ins(S::LogicalNot, tb, &[nan]);
            let equal_bits = e.land(same, not_nan);
            let either = e.bor(old, c);
            let both_mag = e.band(either, magnitude);
            let zero = e.cu(0);
            let zeros = e.ieq(both_mag, zero);
            let eq = e.lor(equal_bits, zeros);
            e.select(tu, eq, data, old)
        }
    }
}

/// One atomic on `ptr` (u64) with u64 operands; returns the old value.
pub fn atomic64(e: &mut Emitter, ptr: Word, op: AtomicOp, data: Word, cmp: Option<Word>, scope: u32) -> Word {
    e.cap(Capability::Int64Atomics);
    let t = e.ty(Type::U64);
    let (sc, sem) = (e.cu(scope), e.cu(0));
    let simple = |e: &mut Emitter, o: S| e.ins(o, t, &[ptr, sc, sem, data]);
    match op {
        AtomicOp::Swap => simple(e, S::AtomicExchange),
        AtomicOp::Add => simple(e, S::AtomicIAdd),
        AtomicOp::Sub => simple(e, S::AtomicISub),
        AtomicOp::SMin => simple(e, S::AtomicSMin),
        AtomicOp::UMin => simple(e, S::AtomicUMin),
        AtomicOp::SMax => simple(e, S::AtomicSMax),
        AtomicOp::UMax => simple(e, S::AtomicUMax),
        AtomicOp::And => simple(e, S::AtomicAnd),
        AtomicOp::Or => simple(e, S::AtomicOr),
        AtomicOp::Xor => simple(e, S::AtomicXor),
        AtomicOp::CmpSwap => {
            let c = cmp.unwrap_or(data);
            e.ins(S::AtomicCompareExchange, t, &[ptr, sc, sem, sem, data, c])
        }
        other => cas_loop_typed(e, t, ptr, scope, |e, old| atomic_next64(e, other, old, data, cmp)),
    }
}

/// The value a 64-bit atomic `op` leaves in memory that held `old`.
pub fn atomic_next64(e: &mut Emitter, op: AtomicOp, old: Word, data: Word, cmp: Option<Word>) -> Word {
    let t = e.ty(Type::U64);
    let tb = e.t_bool();
    let one = e.c64(1);
    let zero = e.c64(0);
    let pick = |e: &mut Emitter, o: S| {
        let c = e.ins(o, tb, &[old, data]);
        e.select(t, c, old, data)
    };
    match op {
        AtomicOp::Swap => data,
        AtomicOp::Add => e.ins(S::IAdd, t, &[old, data]),
        AtomicOp::Sub => e.ins(S::ISub, t, &[old, data]),
        AtomicOp::SMin => pick(e, S::SLessThan),
        AtomicOp::UMin => pick(e, S::ULessThan),
        AtomicOp::SMax => pick(e, S::SGreaterThan),
        AtomicOp::UMax => pick(e, S::UGreaterThan),
        AtomicOp::And => e.ins(S::BitwiseAnd, t, &[old, data]),
        AtomicOp::Or => e.ins(S::BitwiseOr, t, &[old, data]),
        AtomicOp::Xor => e.ins(S::BitwiseXor, t, &[old, data]),
        AtomicOp::CmpSwap => {
            let c = cmp.unwrap_or(data);
            let eq = e.ins(S::IEqual, tb, &[old, c]);
            e.select(t, eq, data, old)
        }
        AtomicOp::Inc => {
            let ge = e.ins(S::ULessThanEqual, tb, &[data, old]);
            let inc = e.ins(S::IAdd, t, &[old, one]);
            e.select(t, ge, zero, inc)
        }
        AtomicOp::Dec => {
            let z = e.ins(S::IEqual, tb, &[old, zero]);
            let gt = e.ins(S::ULessThan, tb, &[data, old]);
            let wrap = e.lor(z, gt);
            let dec = e.ins(S::ISub, t, &[old, one]);
            e.select(t, wrap, data, dec)
        }
        AtomicOp::RSub => e.ins(S::ISub, t, &[data, old]),
        AtomicOp::CSub => {
            let ge = e.ins(S::ULessThanEqual, tb, &[data, old]);
            let d = e.ins(S::ISub, t, &[old, data]);
            e.select(t, ge, d, zero)
        }
        AtomicOp::MaskOr => {
            // data = mask, cmp = value.
            let value = cmp.unwrap_or(zero);
            let nm = e.ins(S::Not, t, &[data]);
            let keep = e.ins(S::BitwiseAnd, t, &[old, nm]);
            e.ins(S::BitwiseOr, t, &[keep, value])
        }
        AtomicOp::Wrap => {
            let add = cmp.unwrap_or(zero);
            let ge = e.ins(S::ULessThanEqual, tb, &[data, old]);
            let s = e.ins(S::ISub, t, &[old, data]);
            let a = e.ins(S::IAdd, t, &[old, add]);
            e.select(t, ge, s, a)
        }
        AtomicOp::FMin | AtomicOp::FMax => float64_min_max(e, old, data, op == AtomicOp::FMax),
        AtomicOp::FAdd => {
            e.cap(Capability::Float64);
            let tf = e.ty(Type::F64);
            let (fo, fd) = (e.bitcast(tf, old), e.bitcast(tf, data));
            let r = e.ins(S::FAdd, tf, &[fo, fd]);
            e.bitcast(t, r)
        }
        AtomicOp::FCmpSwap => {
            e.cap(Capability::Float64);
            let tf = e.ty(Type::F64);
            let c = cmp.unwrap_or(zero);
            let (fo, fc) = (e.bitcast(tf, old), e.bitcast(tf, c));
            let eq = e.ins(S::FOrdEqual, tb, &[fo, fc]);
            e.select(t, eq, data, old)
        }
        // The DS float forms have no 64-bit encoding.
        AtomicOp::DsFMin | AtomicOp::DsFMax | AtomicOp::DsFAdd { .. } => old,
    }
}

/// 64-bit memory float min/max on raw bits (the interpreter's
/// `atomic_float_min_max`): a signalling NaN (source first, then memory)
/// is returned quieted; with two NaNs the source wins; a single quiet NaN
/// yields the other operand; otherwise the total order with -0 < +0
/// decides and ties keep the memory value.
/// Memory f32 min/max on raw bits (the rule of
/// `ps5shader::interp::atomic_float_min_max`, measured on RDNA2): a
/// signaling NaN (source first, then memory) is returned quieted; with two
/// NaNs the source wins; a single quiet NaN yields the other operand;
/// otherwise the total order with -0 < +0 decides (denormals kept) and
/// ties keep the memory value.
fn float32_min_max(e: &mut Emitter, old: Word, src: Word, max: bool) -> Word {
    let tu = e.t_u32();
    let classify = |e: &mut Emitter, v: Word| {
        let m = e.cu(0x7FFF_FFFF);
        let abs = e.band(v, m);
        let inf = e.cu(0x7F80_0000);
        let nan = e.ult(inf, abs);
        let q = e.cu(0x0040_0000);
        let qb = e.band(v, q);
        let zero = e.cu(0);
        let unquiet = e.ieq(qb, zero);
        let signaling = e.land(nan, unquiet);
        let sm = e.cu(0x8000_0000);
        let s = e.band(v, sm);
        let negative = e.ine(s, zero);
        let inv = e.ins(S::Not, tu, &[v]);
        let flagged = e.bor(v, sm);
        let key = e.select(tu, negative, inv, flagged);
        (nan, signaling, key)
    };
    let (onan, osig, okey) = classify(e, old);
    let (snan, ssig, skey) = classify(e, src);
    let better = if max { e.ult(okey, skey) } else { e.ult(skey, okey) };
    let mut r = e.select(tu, better, src, old);
    r = e.select(tu, onan, src, r);
    let both = e.select(tu, onan, src, old);
    r = e.select(tu, snan, both, r);
    let q = e.cu(0x0040_0000);
    let oq = e.bor(old, q);
    r = e.select(tu, osig, oq, r);
    let sq = e.bor(src, q);
    e.select(tu, ssig, sq, r)
}

fn float64_min_max(e: &mut Emitter, old: Word, src: Word, max: bool) -> Word {
    let t = e.ty(Type::U64);
    let tb = e.t_bool();
    let sign = e.c64(1 << 63);
    let abs_mask = e.c64((1 << 63) - 1);
    let infinity = e.c64(0x7FF0_0000_0000_0000);
    let quiet = e.c64(0x0008_0000_0000_0000);
    let zero = e.c64(0);
    let classify = |e: &mut Emitter, v: Word| {
        let abs = e.ins(S::BitwiseAnd, t, &[v, abs_mask]);
        let nan = e.ins(S::ULessThan, tb, &[infinity, abs]);
        let q = e.ins(S::BitwiseAnd, t, &[v, quiet]);
        let unquiet = e.ins(S::IEqual, tb, &[q, zero]);
        let signaling = e.land(nan, unquiet);
        let s = e.ins(S::BitwiseAnd, t, &[v, sign]);
        let negative = e.ins(S::INotEqual, tb, &[s, zero]);
        let inv = e.ins(S::Not, t, &[v]);
        let flagged = e.ins(S::BitwiseOr, t, &[v, sign]);
        let key = e.select(t, negative, inv, flagged);
        (nan, signaling, key)
    };
    let (onan, osig, okey) = classify(e, old);
    let (snan, ssig, skey) = classify(e, src);
    let better = if max { e.ins(S::ULessThan, tb, &[okey, skey]) } else { e.ins(S::ULessThan, tb, &[skey, okey]) };
    let mut r = e.select(t, better, src, old);
    r = e.select(t, onan, src, r);
    let both = e.select(t, onan, src, old);
    r = e.select(t, snan, both, r);
    let oq = e.ins(S::BitwiseOr, t, &[old, quiet]);
    r = e.select(t, osig, oq, r);
    let sq = e.ins(S::BitwiseOr, t, &[src, quiet]);
    e.select(t, ssig, sq, r)
}

/// `do { old = *p; new = f(old) } while (cas(p, new, old) != old)`.
fn cas_loop(e: &mut Emitter, ptr: Word, scope: u32, f: impl FnOnce(&mut Emitter, Word) -> Word) -> Word {
    let tu = e.t_u32();
    cas_loop_typed(e, tu, ptr, scope, f)
}

/// [`cas_loop`] on a pointer to `tu` (u32 or u64).
fn cas_loop_typed(e: &mut Emitter, tu: Word, ptr: Word, scope: u32, f: impl FnOnce(&mut Emitter, Word) -> Word) -> Word {
    let (header, body, cont, merge) = (e.b.id(), e.b.id(), e.b.id(), e.b.id());
    e.b.branch(header).unwrap();
    e.begin(header);
    e.open_scope();
    e.b.loop_merge(merge, cont, rspirv::spirv::LoopControl::NONE, []).unwrap();
    e.b.branch(body).unwrap();
    e.begin(body);
    let (sc, sem) = (e.cu(scope), e.cu(0));
    let old = e.ins(S::AtomicLoad, tu, &[ptr, sc, sem]);
    let new = f(e, old);
    let got = e.ins(S::AtomicCompareExchange, tu, &[ptr, sc, sem, sem, new, old]);
    let tb = e.t_bool();
    let done = e.ins(S::IEqual, tb, &[got, old]);
    e.b.branch_conditional(done, merge, cont, []).unwrap();
    e.begin(cont);
    e.b.branch(header).unwrap();
    e.close_scope();
    e.begin(merge);
    old
}

/// The function has a 64-bit atomic that can reach LDS.
fn uses_lds_atomics64(f: &Function) -> bool {
    f.order.iter().any(|&b| {
        f.block(b).insts.iter().any(|&v| {
            let d = f.value(v);
            // A compute program's 64-bit FLAT atomics may reach LDS too.
            let lds = matches!(d.op, Op::LdsAtomic { .. }) || (matches!(d.op, Op::GlobalAtomic { flat: true, .. }) && f.stage == Stage::Compute);
            lds && d.args.get(1).is_some_and(|&a| f.ty(a) == Type::U64)
        })
    })
}

/// Upper bound (exclusive, bytes) of every LDS address the function's loads
/// and stores can form, or None when some address is not bounded by the
/// analysis (or the function uses LDS atomics / append).
fn lds_extent(f: &Function) -> Option<u64> {
    let mut end = 0u64;
    for &b in &f.order {
        for &v in &f.block(b).insts {
            let bytes = match f.op(v) {
                Op::LdsLoad { access } | Op::LdsStore { access } => access.bytes() as u64,
                Op::LdsAtomic { .. } if f.ty(f.value(v).args[1]) == Type::U64 => 8,
                Op::LdsAtomic { .. } | Op::LdsAppend { .. } => 4,
                _ => continue,
            };
            let addr = f.value(v).args[0];
            end = end.max(max_value(f, addr, 0)? + bytes);
        }
    }
    Some(end)
}

/// Largest value a 32-bit integer expression can take (None when unknown).
fn max_value(f: &Function, v: Value, depth: u32) -> Option<u64> {
    if depth > 32 {
        return None;
    }
    let d = f.value(v);
    let arg = |i: usize| max_value(f, d.args[i], depth + 1);
    let konst = |i: usize| match f.op(d.args[i]) {
        Op::Const(c) => Some(c & 0xFFFF_FFFF),
        _ => None,
    };
    let r = match d.op {
        Op::Const(c) => c & 0xFFFF_FFFF,
        Op::LaneId => 63,
        Op::IAdd => arg(0)? + arg(1)?,
        Op::IMul => arg(0)? * arg(1)?,
        Op::Shl => arg(0)? << konst(1)?.min(32),
        Op::LShr => arg(0)? >> konst(1)?.min(32),
        Op::And => match (arg(0), arg(1)) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => return None,
        },
        Op::Or | Op::UMax => {
            let m = arg(0)?.max(arg(1)?);
            if d.op == Op::Or {
                (m + 1).next_power_of_two() - 1
            } else {
                m
            }
        }
        Op::UMin => match (arg(0), arg(1)) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => return None,
        },
        Op::BitFieldUExtract => {
            let w = konst(2)?.min(32);
            (1u64 << w) - 1
        }
        Op::Select => arg(1)?.max(arg(2)?),
        _ => return None,
    };
    // Past 32 bits the address wraps: no useful bound.
    (r <= 0xFFFF_FFFF).then_some(r)
}

/// Lane-private LDS layout: the slot (array index) of every LDS dword the
/// program touches, keyed by the lane-independent part of its address.
pub struct LaneSlots {
    /// LDS access instruction → the lane-independent byte address `c`.
    constant: HashMap<Value, u64>,
    /// Touched dwords of lane 0 (`c / 4 ..`), in slot order.
    dwords: Vec<u64>,
}

impl LaneSlots {
    fn slot(&self, dword: u64) -> Option<u32> {
        self.dwords.binary_search(&dword).ok().map(|i| i as u32)
    }
}

/// `lane * stride + constant` form of a 32-bit address (bytes).
fn lane_affine(f: &Function, v: Value, depth: u32) -> Option<(u64, u64)> {
    if depth > 64 {
        return None;
    }
    let d = f.value(v);
    let arg = |i: usize| lane_affine(f, d.args[i], depth + 1);
    let r = match d.op {
        Op::Const(c) => (0, c & 0xFFFF_FFFF),
        Op::LaneId => (1, 0),
        Op::IAdd => {
            let (a, b) = (arg(0)?, arg(1)?);
            (a.0 + b.0, a.1 + b.1)
        }
        Op::IMul => {
            let (a, b) = (arg(0)?, arg(1)?);
            match (a.0, b.0) {
                (0, _) => (b.0 * a.1, b.1 * a.1),
                (_, 0) => (a.0 * b.1, a.1 * b.1),
                _ => return None,
            }
        }
        Op::Shl => {
            let (a, b) = (arg(0)?, arg(1)?);
            if b.0 != 0 || b.1 >= 32 {
                return None;
            }
            (a.0 << b.1, a.1 << b.1)
        }
        _ => return None,
    };
    (r.0 <= 0xFFFF_FFFF && r.1 <= 0xFFFF_FFFF).then_some(r)
}

/// Graphics stages have no workgroup memory: their LDS is a per-invocation
/// array. That is exact only when no lane reads what another lane wrote.
/// The common case is the compiler's ADDTID spill space
/// (`ds_write/read_addtid_b32`: `M0 + lane * 4 + offset`): every address is
/// `lane * stride + c` with one stride and constants, and when no two lanes
/// can ever form the same dword, each lane owns its dwords. Then the array
/// only needs one element per dword lane 0 touches (a few dozen), instead
/// of the whole wave's span — the driver reserves the array for every
/// resident invocation.
fn lane_private_lds(f: &Function) -> Option<LaneSlots> {
    let wave = f.wave_size.max(1) as u64;
    let mut stride = None;
    let mut constant = HashMap::new();
    let mut touched = BTreeSet::new();
    for &b in &f.order {
        for &v in &f.block(b).insts {
            let bytes = match f.op(v) {
                Op::LdsLoad { access } | Op::LdsStore { access } => access.bytes() as u64,
                Op::LdsAtomic { .. } | Op::LdsAppend { .. } => return None,
                _ => continue,
            };
            let (s, c) = lane_affine(f, f.value(v).args[0], 0)?;
            if s == 0 || s % 4 != 0 || stride.is_some_and(|t| t != s) || c + s * (wave - 1) + bytes > 0x1_0000_0000 {
                return None;
            }
            stride = Some(s);
            for d in c / 4..(c + bytes).div_ceil(4) {
                touched.insert(d);
            }
            constant.insert(v, c);
            if touched.len() > 4096 {
                return None;
            }
        }
    }
    // Lane l touches dword d + l * t: lanes l1 != l2 meet when two touched
    // dwords differ by a multiple (below the wave size) of t.
    let t = stride? / 4;
    let dwords: Vec<u64> = touched.into_iter().collect();
    for (i, &a) in dwords.iter().enumerate() {
        for &b in &dwords[i + 1..] {
            let diff = b - a;
            if diff % t == 0 && diff / t < wave {
                return None;
            }
        }
    }
    Some(LaneSlots { constant, dwords })
}

impl Emitter<'_, '_> {
    /// The LDS array, declared on first use.
    fn lds_var(&mut self) -> Word {
        if let Some(v) = self.res.lds {
            return v;
        }
        let mut dwords = match self.input.request.local_data_share_dwords {
            0 => 16384,
            n => n.min(16384),
        };
        if self.lds_class() == SC::Private {
            // Graphics stages keep LDS per invocation, which costs the
            // driver `dwords * 4` bytes of local memory for every resident
            // invocation (64 KiB each would reserve gigabytes and stall the
            // GPU). Lane-private layouts get one element per touched dword;
            // otherwise the array covers the bytes the program can address
            // when that is bounded.
            if let Some(slots) = lane_private_lds(self.f) {
                dwords = (slots.dwords.len() as u32).max(1);
                self.res.lds_slots = Some(slots);
            } else if let Some(end) = lds_extent(self.f) {
                let needed = (end.div_ceil(4) as u32).max(64).next_power_of_two();
                dwords = dwords.min(needed);
            }
        }
        let tu = self.t_u32();
        let n = self.cu(dwords);
        if self.lds_class() == SC::Workgroup && self.feat.shared_int64_atomics && uses_lds_atomics64(self.f) && self.mesh.is_none() {
            // 64-bit atomics need a u64 view of the same memory: Block
            // structs in workgroup memory alias each other.
            self.exts.insert("SPV_KHR_workgroup_memory_explicit_layout");
            self.cap(Capability::WorkgroupMemoryExplicitLayoutKHR);
            let arr = self.unique_array(tu, n);
            self.decorate(arr, Decoration::ArrayStride, &[4]);
            let id = self.b.id();
            let s = self.b.type_struct_id(Some(id), [arr]);
            self.decorate(s, Decoration::Block, &[]);
            self.b.member_decorate(s, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
            let v = self.global(s, SC::Workgroup);
            let t64 = self.ty(Type::U64);
            let n64 = self.cu(dwords / 2);
            let arr64 = self.unique_array(t64, n64);
            self.decorate(arr64, Decoration::ArrayStride, &[8]);
            let id = self.b.id();
            let s64 = self.b.type_struct_id(Some(id), [arr64]);
            self.decorate(s64, Decoration::Block, &[]);
            self.b.member_decorate(s64, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
            let v64 = self.global(s64, SC::Workgroup);
            self.decorate(v, Decoration::Aliased, &[]);
            self.decorate(v64, Decoration::Aliased, &[]);
            self.res.lds = Some(v);
            self.res.lds64 = Some(v64);
            self.res.lds_dwords = dwords;
            return v;
        }
        // Workgroup and private arrays carry no explicit layout: the type
        // is not shared with the ArrayStride-decorated block arrays.
        let arr = self.unique_array(tu, n);
        let v = self.global(arr, self.lds_class());
        self.res.lds = Some(v);
        self.res.lds_dwords = dwords;
        v
    }

    /// An array type of its own (never shared with a decorated one).
    fn unique_array(&mut self, element: Word, length: Word) -> Word {
        let id = self.b.id();
        self.b.type_array_id(Some(id), element, length)
    }

    /// Pointer to the 8-byte aligned qword holding byte `byte` of the LDS
    /// (the address wraps like dword accesses).
    pub fn lds_ptr64(&mut self, byte: Word) -> R<Word> {
        self.lds_var();
        let Some(var) = self.res.lds64 else {
            return Err(if !self.feat.shared_int64_atomics {
                "64-bit LDS atomics without device support".into()
            } else if self.mesh.is_some() {
                "64-bit LDS atomics in a mesh stage".into()
            } else {
                "64-bit LDS atomics in a graphics stage".into()
            });
        };
        let (idx, _) = self.lds_qword(byte);
        let t64 = self.ty(Type::U64);
        let p = self.ptr(SC::Workgroup, t64);
        let zero = self.cu(0);
        Ok(self.access(p, var, &[zero, idx]))
    }

    /// Whether the dword holding LDS byte `byte` lies inside the allocated
    /// LDS.
    pub fn lds_dword_inside(&mut self, byte: Word) -> Word {
        let two = self.cu(2);
        let idx = self.shr(byte, two);
        let dwords = self.cu(self.res.lds_dwords);
        self.ult(idx, dwords)
    }

    /// Qword index of LDS byte `byte` (8-byte aligned; the address does
    /// not wrap) and whether it lies inside the allocated LDS.
    pub fn lds_qword(&mut self, byte: Word) -> (Word, Word) {
        let three = self.cu(3);
        let idx = self.shr(byte, three);
        let qwords = self.cu(self.res.lds_dwords / 2);
        let inside = self.ult(idx, qwords);
        let last = self.cu((self.res.lds_dwords / 2).max(1) - 1);
        (self.umin(idx, last), inside)
    }

    /// Pointer to dword `k` of LDS access `v` in a lane-private layout
    /// (None when the layout is not lane-private).
    pub fn lds_slot_ptr(&mut self, v: Value, k: u32) -> R<Option<Word>> {
        let var = self.lds_var();
        let Some(slots) = &self.res.lds_slots else { return Ok(None) };
        let c = *slots.constant.get(&v).ok_or("LDS access outside the lane-private layout")?;
        let slot = slots.slot(c / 4 + k as u64).ok_or("LDS dword outside the lane-private layout")?;
        let tu = self.t_u32();
        let idx = self.cu(slot);
        let p = self.ptr(SC::Private, tu);
        Ok(Some(self.access(p, var, &[idx])))
    }

    pub fn lds_ptr(&mut self, byte: Word) -> R<Word> {
        let var = self.lds_var();
        if self.res.lds_slots.is_some() {
            return Err("LDS address outside the lane-private layout".into());
        }
        let tu = self.t_u32();
        let two = self.cu(2);
        let idx = self.shr(byte, two);
        let m = self.cu(self.res.lds_dwords - 1);
        let idx = if self.res.lds_dwords.is_power_of_two() { self.band(idx, m) } else { self.umin(idx, m) };
        let class = self.lds_class();
        let p = self.ptr(class, tu);
        if self.res.lds64.is_some() {
            let zero = self.cu(0);
            return Ok(self.access(p, var, &[zero, idx]));
        }
        Ok(self.access(p, var, &[idx]))
    }

    /// Dword `k` after byte address `byte` of the GDS. The address wraps at
    /// 64 KiB, the size of the bound GDS buffer, so every index is in range.
    pub fn gds_ptr(&mut self, byte: Word, k: u32) -> R<Word> {
        let var = self.word_buffer(DescriptorBindingKind::GlobalDataShare)?;
        let tu = self.t_u32();
        let two = self.cu(2);
        let idx = self.shr(byte, two);
        let idx = if k == 0 {
            idx
        } else {
            let kk = self.cu(k);
            self.iadd(idx, kk)
        };
        let m = self.cu(GDS_BYTES / 4 - 1);
        let idx = self.band(idx, m);
        let p = self.ptr(SC::StorageBuffer, tu);
        let zero = self.cu(0);
        Ok(self.access(p, var, &[zero, idx]))
    }

    /// The qword holding GDS byte `byte` (8-byte aligned; the address wraps
    /// at 64 KiB like dword accesses), through a u64 view of the GDS buffer.
    pub fn gds_ptr64(&mut self, byte: Word) -> R<Word> {
        let var = match self.res.gds64 {
            Some(v) => v,
            None => {
                if self.input.request.bindings.find(DescriptorBindingKind::GlobalDataShare).is_none() {
                    return Err("the layout has no GlobalDataShare binding".into());
                }
                let block = self.u64_block();
                let v = self.binding_var(block, SC::StorageBuffer, DescriptorBindingKind::GlobalDataShare);
                self.decorate(v, Decoration::Coherent, &[]);
                self.decorate(v, Decoration::Aliased, &[]);
                self.res.gds64 = Some(v);
                v
            }
        };
        let t64 = self.ty(Type::U64);
        let three = self.cu(3);
        let idx = self.shr(byte, three);
        let m = self.cu(GDS_BYTES / 8 - 1);
        let idx = self.band(idx, m);
        let p = self.ptr(SC::StorageBuffer, t64);
        let zero = self.cu(0);
        Ok(self.access(p, var, &[zero, idx]))
    }

    pub fn lds_class(&self) -> SC {
        if self.stage == ps5shader::resources::ShaderStage::Compute || self.mesh.is_some() {
            SC::Workgroup
        } else {
            SC::Private
        }
    }

    /// This lane half's scratch array, declared on first use.
    fn scratch_var(&mut self) -> Word {
        let half = self.half;
        match self.res.scratch[half] {
            Some(v) => v,
            None => {
                let dwords = self.input.request.scratch_dwords.clamp(64, 16384).next_power_of_two();
                let tu = self.t_u32();
                let n = self.cu(dwords);
                let arr = self.unique_array(tu, n);
                let v = self.global(arr, SC::Private);
                self.res.scratch[half] = Some(v);
                self.res.scratch_dwords = dwords;
                v
            }
        }
    }

    pub fn scratch_ptr(&mut self, byte: Word) -> R<Word> {
        let var = self.scratch_var();
        let tu = self.t_u32();
        let two = self.cu(2);
        let idx = self.shr(byte, two);
        let m = self.cu(self.res.scratch_dwords - 1);
        let idx = self.band(idx, m);
        let p = self.ptr(SC::Private, tu);
        Ok(self.access(p, var, &[idx]))
    }
}

/// The local memory a FLAT aperture reaches.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Aperture {
    /// SHARED_BASE: the work-group's LDS.
    Lds,
    /// PRIVATE_BASE: the invocation's scratch.
    Scratch,
}

impl Emitter<'_, '_> {
    /// FLAT-segment accesses of a compute program route the shared and
    /// private apertures by address.
    fn routes_apertures(&self, flat: bool) -> bool {
        flat && self.stage == ShaderStage::Compute && self.mesh.is_none()
    }

    /// The low dword of a FLAT address and whether its high dword names the
    /// shared or the private aperture (each a 4 GiB window).
    fn split_aperture(&mut self, addr: Word) -> (Word, Word, Word) {
        let (low, high) = self.split64(addr);
        let s = self.cu(ps5shader::inline_constants::SHARED_APERTURE_HIGH);
        let shared = self.ieq(high, s);
        let p = self.cu(ps5shader::inline_constants::PRIVATE_APERTURE_HIGH);
        let private = self.ieq(high, p);
        (low, shared, private)
    }

    /// Dwords of an aperture's memory: the LDS the dispatch allocates; the
    /// scratch the host sizes, or else the scratch array the program's
    /// scratch accesses use.
    fn aperture_dwords(&mut self, a: Aperture) -> u32 {
        match a {
            Aperture::Lds => {
                if self.input.request.local_data_share_dwords == 0 {
                    return 0;
                }
                self.lds_var();
                self.res.lds_dwords
            }
            Aperture::Scratch => {
                self.scratch_var();
                match self.input.request.scratch_dwords {
                    0 => self.res.scratch_dwords,
                    n => n.min(self.res.scratch_dwords),
                }
            }
        }
    }

    /// Pointer to dword `index` of an aperture's memory (inside it).
    fn aperture_word_ptr(&mut self, a: Aperture, index: Word) -> R<Word> {
        let tu = self.t_u32();
        match a {
            Aperture::Lds => {
                let var = self.lds_var();
                if self.res.lds_slots.is_some() {
                    return Err("a FLAT access through the shared aperture in a lane-private LDS layout".into());
                }
                let class = self.lds_class();
                let p = self.ptr(class, tu);
                if self.res.lds64.is_some() {
                    let zero = self.cu(0);
                    return Ok(self.access(p, var, &[zero, index]));
                }
                Ok(self.access(p, var, &[index]))
            }
            Aperture::Scratch => {
                let var = self.scratch_var();
                let p = self.ptr(SC::Private, tu);
                Ok(self.access(p, var, &[index]))
            }
        }
    }

    /// `bytes` bytes from byte `offset` lie inside the aperture's memory
    /// (`dwords` long), and the offset of the last one does not wrap.
    fn aperture_fits(&mut self, dwords: u32, offset: Word, bytes: u32) -> Word {
        let last = bytes - 1;
        let limit = self.cu(u32::MAX - last);
        let no_wrap = self.ule(offset, limit);
        let end = if last == 0 {
            offset
        } else {
            let l = self.cu(last);
            self.iadd(offset, l)
        };
        let two = self.cu(2);
        let index = self.shr(end, two);
        let n = self.cu(dwords);
        let inside = self.ult(index, n);
        self.land(no_wrap, inside)
    }

    /// One element of `bytes` (1, 2 or 4) at byte `offset` of an aperture,
    /// at any alignment, zero-extended; 0 unless it lies whole inside.
    fn aperture_load(&mut self, a: Aperture, offset: Word, bytes: u32) -> R<Word> {
        let zero = self.cu(0);
        let dwords = self.aperture_dwords(a);
        if dwords == 0 {
            return Ok(zero);
        }
        let ok = self.aperture_fits(dwords, offset, bytes);
        let tu = self.t_u32();
        let r = self.if_value(ok, tu, zero, |e| {
            let r = e.aperture_load_inside(a, offset, bytes);
            guarded(e, r)
        });
        self.take_error()?;
        Ok(r)
    }

    fn aperture_load_inside(&mut self, a: Aperture, offset: Word, bytes: u32) -> R<Word> {
        let tu = self.t_u32();
        let two = self.cu(2);
        let first = self.shr(offset, two);
        let p = self.aperture_word_ptr(a, first)?;
        let w0 = self.load(tu, p);
        let three = self.cu(3);
        let low = self.band(offset, three);
        let eight = self.cu(8);
        let shift = self.imul(low, eight);
        let value = if bytes == 1 {
            self.shr(w0, shift)
        } else {
            // The element may continue in the next dword.
            let l = self.cu(bytes - 1);
            let end = self.iadd(offset, l);
            let last = self.shr(end, two);
            let p = self.aperture_word_ptr(a, last)?;
            let w1 = self.load(tu, p);
            let down = self.shr(w0, shift);
            let thirty_two = self.cu(32);
            let up_shift = self.isub(thirty_two, shift);
            let up = self.shl(w1, up_shift);
            let joined = self.bor(down, up);
            let zero = self.cu(0);
            let aligned = self.ieq(shift, zero);
            self.select(tu, aligned, w0, joined)
        };
        if bytes == 4 {
            return Ok(value);
        }
        let field = self.cu((1u32 << (8 * bytes)) - 1);
        Ok(self.band(value, field))
    }

    /// Store the low `bytes` (1, 2 or 4) of `value` at byte `offset` of an
    /// aperture, at any alignment; dropped unless the element lies whole
    /// inside.
    fn aperture_store(&mut self, a: Aperture, offset: Word, bytes: u32, value: Word) -> R<()> {
        let dwords = self.aperture_dwords(a);
        if dwords == 0 {
            return Ok(());
        }
        let ok = self.aperture_fits(dwords, offset, bytes);
        self.if_then(ok, |e| {
            if let Err(m) = e.aperture_store_inside(a, offset, bytes, value) {
                e.res_error(m);
            }
        });
        self.take_error()
    }

    fn aperture_store_inside(&mut self, a: Aperture, offset: Word, bytes: u32, value: Word) -> R<()> {
        let two = self.cu(2);
        let first = self.shr(offset, two);
        let three = self.cu(3);
        let low = self.band(offset, three);
        let eight = self.cu(8);
        let shift = self.imul(low, eight);
        let field = self.cu(if bytes == 4 { u32::MAX } else { (1u32 << (8 * bytes)) - 1 });
        let v = self.band(value, field);
        if bytes == 4 {
            // An aligned dword is one store.
            let zero = self.cu(0);
            let aligned = self.ieq(shift, zero);
            let p = self.aperture_word_ptr(a, first)?;
            self.if_then(aligned, |e| e.store(p, v));
            let unaligned = self.lnot(aligned);
            self.if_then(unaligned, |e| {
                if let Err(m) = e.aperture_store_split(a, offset, first, shift, field, v, bytes) {
                    e.res_error(m);
                }
            });
            return self.take_error();
        }
        self.aperture_store_split(a, offset, first, shift, field, v, bytes)
    }

    /// The bytes of an element in dword `first` from bit `shift` and, when
    /// it continues there, the next dword.
    #[allow(clippy::too_many_arguments)]
    fn aperture_store_split(&mut self, a: Aperture, offset: Word, first: Word, shift: Word, field: Word, v: Word, bytes: u32) -> R<()> {
        let m0 = self.shl(field, shift);
        let v0 = self.shl(v, shift);
        self.aperture_merge(a, first, m0, v0)?;
        if bytes == 1 {
            return Ok(());
        }
        let two = self.cu(2);
        let l = self.cu(bytes - 1);
        let end = self.iadd(offset, l);
        let last = self.shr(end, two);
        let spans = self.ine(last, first);
        self.if_then(spans, |e| {
            let thirty_two = e.cu(32);
            let down = e.isub(thirty_two, shift);
            let m1 = e.shr(field, down);
            let v1 = e.shr(v, down);
            if let Err(m) = e.aperture_merge(a, last, m1, v1) {
                e.res_error(m);
            }
        });
        self.take_error()
    }

    /// Replace the `mask` bits of dword `index` with `value` (masked); in
    /// the work-group's LDS the neighbouring bytes stay intact under
    /// concurrent writers.
    fn aperture_merge(&mut self, a: Aperture, index: Word, mask: Word, value: Word) -> R<()> {
        let p = self.aperture_word_ptr(a, index)?;
        let tu = self.t_u32();
        let keep = self.ins(S::Not, tu, &[mask]);
        if a == Aperture::Lds && self.lds_class() == SC::Workgroup {
            let (scope, sem) = (self.cu(SCOPE_WORKGROUP), self.cu(0));
            self.ins(S::AtomicAnd, tu, &[p, scope, sem, keep]);
            self.ins(S::AtomicOr, tu, &[p, scope, sem, value]);
        } else {
            let old = self.load(tu, p);
            let kept = self.band(old, keep);
            let new = self.bor(kept, value);
            self.store(p, new);
        }
        Ok(())
    }

    /// An atomic through an aperture at byte `offset`: the operand (4 or,
    /// `wide`, 8 bytes) must be aligned to its size and lie inside the
    /// memory, or nothing happens and 0 is returned. Scratch is private to
    /// the invocation, so its update is a plain read-modify-write.
    fn aperture_atomic(&mut self, a: Aperture, offset: Word, op: AtomicOp, data: Word, cmp: Option<Word>, wide: bool) -> R<Word> {
        let bytes = if wide { 8 } else { 4 };
        let (ty, none) = if wide { (self.ty(Type::U64), self.c64(0)) } else { (self.t_u32(), self.cu(0)) };
        let dwords = self.aperture_dwords(a);
        if dwords == 0 {
            return Ok(none);
        }
        let m = self.cu(bytes - 1);
        let low = self.band(offset, m);
        let zero = self.cu(0);
        let aligned = self.ieq(low, zero);
        let two = self.cu(2);
        let first = self.shr(offset, two);
        let last = if wide {
            let one = self.cu(1);
            self.iadd(first, one)
        } else {
            first
        };
        let n = self.cu(dwords);
        let inside = self.ult(last, n);
        let ok = self.land(aligned, inside);
        let r = self.if_value(ok, ty, none, |e| {
            let r = e.aperture_atomic_inside(a, offset, first, op, data, cmp, wide);
            guarded(e, r)
        });
        self.take_error()?;
        Ok(r)
    }

    #[allow(clippy::too_many_arguments)]
    fn aperture_atomic_inside(&mut self, a: Aperture, offset: Word, first: Word, op: AtomicOp, data: Word, cmp: Option<Word>, wide: bool) -> R<Word> {
        let tu = self.t_u32();
        match (a, wide) {
            (Aperture::Lds, false) => {
                let p = self.aperture_word_ptr(a, first)?;
                Ok(atomic(self, p, op, data, cmp, SCOPE_WORKGROUP))
            }
            (Aperture::Lds, true) => {
                let p = self.lds_ptr64(offset)?;
                Ok(atomic64(self, p, op, data, cmp, SCOPE_WORKGROUP))
            }
            (Aperture::Scratch, false) => {
                let p = self.aperture_word_ptr(a, first)?;
                let old = self.load(tu, p);
                let new = atomic_next(self, op, old, data, cmp);
                self.store(p, new);
                Ok(old)
            }
            (Aperture::Scratch, true) => {
                let p0 = self.aperture_word_ptr(a, first)?;
                let one = self.cu(1);
                let second = self.iadd(first, one);
                let p1 = self.aperture_word_ptr(a, second)?;
                let lo = self.load(tu, p0);
                let hi = self.load(tu, p1);
                let old = self.join64(lo, hi);
                let new = atomic_next64(self, op, old, data, cmp);
                let (nlo, nhi) = self.split64(new);
                self.store(p0, nlo);
                self.store(p1, nhi);
                Ok(old)
            }
        }
    }
}
