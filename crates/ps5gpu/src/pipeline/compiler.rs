// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! The shader compiler interface: one permutation of a guest program in,
//! SPIR-V plus reflection out.
//!
//! The pipeline provider builds a [`ShaderCompileRequest`] per permutation
//! (program + resource plan, the specialised resource tables, the binding
//! layout and the stage options such as pixel outputs, vertex inputs and
//! workgroup size) and hands it to the installed [`ShaderCompiler`]. The
//! result is cached by [`ShaderCompileInput::key`], so a compiler may be
//! stateless.
//!
//! # Binding contract
//!
//! Every stage uses descriptor set 0. A resource of kind `k` of stage `s`
//! lives at binding [`native_binding_index`]`(s, k)` with the descriptor
//! type of [`descriptor_type`] and `max(1, resources.len())` elements (in
//! the order of `DescriptorBinding::resources`). The push data (see
//! `BindingLayout`: live user-data registers, shader base, packed buffer
//! offset adjustments, dispatch limits) occupies dwords
//! `push_data_start_dword ..` of one 128-byte push-constant block shared by
//! all stages of the pipeline; when it does not fit it is read from the
//! `ShaderData` storage buffer instead. Buffer bindings point at the guest
//! address aligned down to the device's storage-buffer offset alignment; the
//! byte adjustment (< 256) is packed into the memory-offset dwords (8 bits
//! per buffer) and must be added by the shader.
//!
//! Pixel shaders write color output `host_location` of each
//! `request.pixel_outputs` entry (location = guest slot), with the
//! component type of its `kind`. Vertex shaders write `gl_Position` (as a
//! member of a `gl_PerVertex` block) and their parameters as `vec4` outputs
//! at the locations reported in [`ShaderReflection::outputs`].

use ash::vk;
use ps5shader::resources::{
    image_binding, native_binding_index, DescriptorBindingKind, ImageResourceClass, ResourceSpecialization, ShaderCompileRequest,
    ShaderStage,
};
use std::fmt;

/// Everything a compiler receives for one permutation.
pub struct ShaderCompileInput<'a, 'p> {
    pub stage: ShaderStage,
    /// A vertex-stage program run as a geometry subgroup (mesh shader).
    pub mesh: Option<MeshConfig>,
    pub request: &'a ShaderCompileRequest<'p>,
    /// The module-affecting resource state the request was specialised for.
    pub specialization: &'a ResourceSpecialization,
    /// Guest address of the program entry.
    pub guest_address: u64,
    /// Identity of this permutation (program hash, stage, specialization,
    /// layout and stage options); equal keys must produce equal modules.
    pub key: u64,
    /// The code words the translation may read (jump tables) when they
    /// come from a record instead of guest memory (a module translated
    /// again from the prewarm list).
    pub code: Option<&'a (dyn ps5shader::CodeSource + Sync)>,
}

/// Scalar class of an interface variable.
/// A geometry (merged ES+GS) subgroup run as one mesh workgroup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshConfig {
    /// VGT_PRIMITIVE_TYPE of the input primitives (1 points, 2 lines,
    /// 4 triangles, 5 fan, 6 strip).
    pub input_primitive: u32,
    pub primitives_per_group: u32,
    pub vertices_per_group: u32,
    /// Vertices and primitives a subgroup may emit.
    pub max_vertices: u32,
    pub max_primitives: u32,
    /// Guest lanes per workgroup (a multiple of 64).
    pub threads: u32,
    /// VGT_ESGS_RING_ITEMSIZE: LDS stride of one ES vertex.
    pub esgs_item_size: u32,
    /// First shader-data dword of the draw parameters (see
    /// `shader::mesh`).
    pub draw_dword: u32,
    /// The geometry stage emits line strips (VGT_GS_OUT_PRIM_TYPE 1): the
    /// workgroup outputs lines of two vertex indices.
    pub lines: bool,
    /// The geometry stage emits point lists (VGT_GS_OUT_PRIM_TYPE 0): each
    /// primitive is one vertex index (PrimitivePointIndicesEXT).
    pub points: bool,
}

impl MeshConfig {
    /// Vertices of one input primitive.
    pub fn input_size(&self) -> u32 {
        match self.input_primitive {
            1 => 1,
            2 => 2,
            _ => 3,
        }
    }

    /// Vertices between consecutive input primitives.
    pub fn input_step(&self) -> u32 {
        if matches!(self.input_primitive, 5 | 6) {
            1
        } else {
            self.input_size()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComponentKind {
    Float,
    Uint,
    Sint,
}

/// One descriptor binding a module declares (set 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReflectedBinding {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub count: u32,
}

/// One stage input or output (`location`, components, type).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReflectedVariable {
    pub location: u32,
    pub components: u32,
    pub kind: ComponentKind,
    /// Flat (not interpolated) parameter.
    pub flat: bool,
}

/// What the provider needs to know about a compiled module.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderReflection {
    /// Entry point name (`main` when empty).
    pub entry_point: String,
    pub bindings: Vec<ReflectedBinding>,
    /// Push-constant bytes the module reads (`offset..offset+size`), if any.
    pub push_constants: Option<(u32, u32)>,
    /// Vertex attributes a vertex shader consumes (fixed-function fetch).
    pub vertex_inputs: Vec<ReflectedVariable>,
    /// Vertex parameters / pixel color outputs.
    pub outputs: Vec<ReflectedVariable>,
    /// Vertex shader writes `gl_PointSize` / the clip distance count.
    pub writes_point_size: bool,
    pub clip_distances: u32,
    /// Vertex shader selects the viewport (`gl_ViewportIndex`).
    pub writes_viewport_index: bool,
    /// Pixel shader writes depth / sample mask.
    pub writes_depth: bool,
    /// Compute workgroup size.
    pub local_size: [u32; 3],
    /// Bytes of LDS the module keeps per invocation (graphics stages
    /// without workgroup memory); 0 when LDS is shared or unused.
    pub private_lds_bytes: u32,
    /// Subgroup size a compute pipeline must require (0: the default).
    pub required_subgroup_size: u32,
    /// Every storage-buffer access of the module stays in its bound range
    /// by the module's own checks: a compute pipeline may turn the
    /// driver's buffer robustness off (VK_EXT_pipeline_robustness).
    pub own_buffer_checks: bool,
}

impl ShaderReflection {
    pub fn entry(&self) -> &str {
        if self.entry_point.is_empty() {
            "main"
        } else {
            &self.entry_point
        }
    }
}

/// A compiled permutation.
#[derive(Clone, Debug)]
pub struct CompiledShader {
    pub spirv: Vec<u32>,
    pub reflection: ShaderReflection,
    /// The guest code words the translation read besides the decoded
    /// program (address, word): kept to translate the module again.
    pub code_reads: Vec<(u64, u32)>,
}

/// Why a permutation was not compiled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompileError {
    /// The compiler does not handle this program yet (the draw is reported
    /// as needing a shader).
    Unsupported(String),
    /// The program is invalid or translation failed.
    Failed(String),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompileError::Unsupported(m) => write!(f, "unsupported: {m}"),
            CompileError::Failed(m) => write!(f, "failed: {m}"),
        }
    }
}

impl std::error::Error for CompileError {}

/// Translates guest programs to SPIR-V. Called from the GPU back-end with
/// its lock held; results are cached by the provider per
/// [`ShaderCompileInput::key`].
pub trait ShaderCompiler: Send + Sync {
    fn compile(&self, input: &ShaderCompileInput<'_, '_>) -> Result<CompiledShader, CompileError>;

    /// [`ShaderCompiler::compile`] past any cache of modules by key (a
    /// trial translation that is not the module of `input.key`).
    fn compile_uncached(&self, input: &ShaderCompileInput<'_, '_>) -> Result<CompiledShader, CompileError> {
        self.compile(input)
    }

    /// A short name for diagnostics.
    fn name(&self) -> &str {
        "shader-compiler"
    }

    /// Identity of the translator (its sources, the device features and
    /// the switches that change what it emits): modules of another stamp
    /// are translated again. None: no stamp (modules are kept as made).
    fn translator_stamp(&self) -> Option<u64> {
        None
    }

    /// Drop the cached module of `key` (a full recompile translates it
    /// again instead of reading it back).
    fn forget_cached(&self, _key: u64) {}
}

/// The Vulkan descriptor type of a binding kind.
pub fn descriptor_type(kind: DescriptorBindingKind) -> vk::DescriptorType {
    match kind {
        DescriptorBindingKind::Image(_) => match image_binding::resource_class(kind) {
            ImageResourceClass::Storage => vk::DescriptorType::STORAGE_IMAGE,
            _ => vk::DescriptorType::SAMPLED_IMAGE,
        },
        DescriptorBindingKind::Samplers => vk::DescriptorType::SAMPLER,
        _ => vk::DescriptorType::STORAGE_BUFFER,
    }
}

/// Shader stage flag of a guest stage.
pub fn stage_flags(stage: ShaderStage) -> vk::ShaderStageFlags {
    match stage {
        ShaderStage::Pixel => vk::ShaderStageFlags::FRAGMENT,
        ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
        _ => vk::ShaderStageFlags::VERTEX,
    }
}

/// The set-0 bindings a request's layout declares (what a conforming module
/// must use).
pub fn layout_bindings(stage: ShaderStage, request: &ShaderCompileRequest<'_>) -> Vec<ReflectedBinding> {
    request
        .bindings
        .descriptors
        .iter()
        .map(|b| ReflectedBinding {
            binding: native_binding_index(stage, b.kind),
            descriptor_type: descriptor_type(b.kind),
            count: (b.resources.len() as u32).max(1),
        })
        .collect()
}
