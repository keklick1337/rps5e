// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Graphics pipelines linked from stage libraries
//! (VK_EXT_graphics_pipeline_library with fast linking).
//!
//! A graphics pipeline is four parts: the vertex input interface, the
//! pre-rasterization shaders (vertex / mesh and tessellation stages with
//! viewport and rasterization state), the fragment shader (with its depth,
//! stencil and multisample state) and the fragment output interface
//! (attachment formats, blending). When a draw needs a pipeline nobody
//! built yet, the provider builds each missing part as a library — the two
//! shader parts on compile workers, in parallel; the interface parts, which
//! hold no shader code, on the recording thread — and links them without
//! link-time optimisation, which takes well under a millisecond. The draw
//! runs with the linked pipeline at once. The pipeline goes into the
//! prewarm list, so the next start builds it whole (into the pipeline
//! cache) before the game runs; with `RPS5E_GPL=rebuild` the whole pipeline
//! is also built during this run, behind every other compile job, and
//! swapped in when ready.
//!
//! A part is keyed by the shader it holds and only the state of its own
//! subset ([`LibKey::of`]), so pipelines that differ in, say, blending or
//! target formats share their shader libraries: a new combination of known
//! shaders costs a link, not a compile.
//!
//! Off by default (`RPS5E_GPL=1` or `rebuild` turns it on). On NVIDIA (615)
//! a library costs the driver as much as a whole pipeline (Silent Hill: 561
//! shader libraries 165 s against 394 whole pipelines 151 s; only a quarter
//! of the libraries were shared), so the split only shortens the wait for
//! one new pipeline, and a fast-linked pipeline lacks the optimisations
//! across stages (unused vertex outputs stay): Silent Hill's first room ran
//! at 21 fps with linked pipelines against 25-26 with whole ones. Drivers
//! without the extension or without fast linking always take the
//! monolithic path.

use super::state::{FixedFunctionKey, RectMode, VertexInputKey};
use ash::vk;

/// How missing graphics pipelines are made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Built whole on a compile worker.
    Off,
    /// Linked from libraries.
    Link,
    /// Linked from libraries at once; the whole build replaces it.
    Rebuild,
}

/// `RPS5E_GPL`: unset / `0` / `off`: [`Mode::Off`]; `1` / `on` / `link`:
/// [`Mode::Link`]; `rebuild`: [`Mode::Rebuild`]. `RPS5E_SAFE=1` turns it
/// off.
pub fn mode() -> Mode {
    static MODE: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| parse_mode(std::env::var("RPS5E_GPL").ok().as_deref(), crate::backend::ab::safe()))
}

fn parse_mode(v: Option<&str>, safe: bool) -> Mode {
    match v.map(str::trim) {
        _ if safe => Mode::Off,
        Some(v) if v.eq_ignore_ascii_case("rebuild") => Mode::Rebuild,
        Some(v) if v == "1" || v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("link") => Mode::Link,
        _ => Mode::Off,
    }
}

/// Whether the device should enable the extension (the mode is not off).
pub fn requested() -> bool {
    mode() != Mode::Off
}

/// One of the four parts of a graphics pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    VertexInput,
    PreRaster,
    Fragment,
    Output,
}

impl Part {
    pub fn flag(self) -> vk::GraphicsPipelineLibraryFlagsEXT {
        match self {
            Part::VertexInput => vk::GraphicsPipelineLibraryFlagsEXT::VERTEX_INPUT_INTERFACE,
            Part::PreRaster => vk::GraphicsPipelineLibraryFlagsEXT::PRE_RASTERIZATION_SHADERS,
            Part::Fragment => vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_SHADER,
            Part::Output => vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_OUTPUT_INTERFACE,
        }
    }

    /// The parts holding shader code (built on compile workers).
    pub fn has_code(self) -> bool {
        matches!(self, Part::PreRaster | Part::Fragment)
    }

    /// Whether `stage` belongs to this part's shader stages.
    pub fn owns_stage(self, stage: vk::ShaderStageFlags) -> bool {
        match self {
            Part::PreRaster => stage != vk::ShaderStageFlags::FRAGMENT,
            Part::Fragment => stage == vk::ShaderStageFlags::FRAGMENT,
            Part::VertexInput | Part::Output => false,
        }
    }

    /// Whether dynamic state `s` belongs to this part.
    pub fn owns_dynamic(self, s: vk::DynamicState) -> bool {
        use vk::DynamicState as D;
        match self {
            Part::VertexInput => false,
            Part::PreRaster => matches!(s, D::VIEWPORT | D::SCISSOR | D::LINE_WIDTH | D::DEPTH_BIAS | D::CULL_MODE | D::FRONT_FACE | D::DEPTH_BIAS_ENABLE),
            Part::Fragment => matches!(
                s,
                D::STENCIL_COMPARE_MASK
                    | D::STENCIL_WRITE_MASK
                    | D::STENCIL_REFERENCE
                    | D::DEPTH_TEST_ENABLE
                    | D::DEPTH_WRITE_ENABLE
                    | D::DEPTH_COMPARE_OP
                    | D::STENCIL_TEST_ENABLE
                    | D::STENCIL_OP
                    | D::DEPTH_BOUNDS
                    | D::DEPTH_BOUNDS_TEST_ENABLE
            ),
            Part::Output => s == D::BLEND_CONSTANTS,
        }
    }
}

/// A library: its part, the shader it holds (0: none), the layout, and the
/// pipeline state its part reads (the rest set to fixed values).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LibKey {
    pub part: Part,
    pub shader: u64,
    pub layout: vk::PipelineLayout,
    /// The generated tessellation stages (pre-rasterization only).
    pub tess: Option<(vk::ShaderModule, vk::ShaderModule)>,
    pub mesh: bool,
    pub ff: FixedFunctionKey,
}

impl LibKey {
    /// The key of `part` of the pipeline of shaders `vs` / `ps` (0: none),
    /// layout `layout` and state `ff`.
    pub fn of(part: Part, vs: u64, ps: u64, layout: vk::PipelineLayout, tess: Option<(vk::ShaderModule, vk::ShaderModule)>, mesh: bool, ff: &FixedFunctionKey) -> LibKey {
        let mut m = FixedFunctionKey {
            topology: 0,
            rect: RectMode::None,
            primitive_restart: false,
            vertex_input: VertexInputKey::default(),
            polygon_mode: 0,
            depth_clip: false,
            depth_clamp: false,
            negative_one_to_one: false,
            color_formats: [0; super::state::MAX_COLOR_ATTACHMENTS],
            color_count: 0,
            blend: Default::default(),
            depth_format: 0,
            stencil_format: 0,
            samples: 0,
            centered_samples: false,
            sample_shading: false,
            sample_mask: 0,
            logic_op: None,
            viewport_count: 0,
            provoking_last: false,
        };
        let multisample = |m: &mut FixedFunctionKey| {
            m.samples = ff.samples;
            m.centered_samples = ff.centered_samples;
            m.sample_shading = ff.sample_shading;
            m.sample_mask = ff.sample_mask;
        };
        let (shader, layout, tess, mesh) = match part {
            Part::VertexInput => {
                m.topology = ff.topology;
                m.primitive_restart = ff.primitive_restart;
                m.vertex_input = ff.vertex_input.clone();
                (0, vk::PipelineLayout::null(), None, false)
            }
            Part::PreRaster => {
                // The topology class decides how primitives are assembled
                // for the stages (points, lines, triangles, patches).
                m.topology = ff.topology;
                m.polygon_mode = ff.polygon_mode;
                m.depth_clip = ff.depth_clip;
                m.depth_clamp = ff.depth_clamp;
                m.negative_one_to_one = ff.negative_one_to_one;
                m.viewport_count = ff.viewport_count;
                m.provoking_last = ff.provoking_last;
                (vs, layout, tess, mesh)
            }
            Part::Fragment => {
                multisample(&mut m);
                m.depth_format = ff.depth_format;
                m.stencil_format = ff.stencil_format;
                (ps, layout, None, false)
            }
            Part::Output => {
                multisample(&mut m);
                m.color_count = ff.color_count;
                m.color_formats = ff.color_formats;
                m.blend = ff.blend;
                m.logic_op = ff.logic_op;
                m.depth_format = ff.depth_format;
                m.stencil_format = ff.stencil_format;
                (0, vk::PipelineLayout::null(), None, false)
            }
        };
        LibKey { part, shader, layout, tess, mesh, ff: m }
    }
}

/// The dynamic states of every graphics pipeline (`depth_bounds`: the
/// device has depth bounds tests).
pub fn dynamic_states(depth_bounds: bool) -> Vec<vk::DynamicState> {
    let mut dynamic = vec![
        vk::DynamicState::VIEWPORT,
        vk::DynamicState::SCISSOR,
        vk::DynamicState::LINE_WIDTH,
        vk::DynamicState::DEPTH_BIAS,
        vk::DynamicState::BLEND_CONSTANTS,
        vk::DynamicState::STENCIL_COMPARE_MASK,
        vk::DynamicState::STENCIL_WRITE_MASK,
        vk::DynamicState::STENCIL_REFERENCE,
        vk::DynamicState::CULL_MODE,
        vk::DynamicState::FRONT_FACE,
        vk::DynamicState::DEPTH_TEST_ENABLE,
        vk::DynamicState::DEPTH_WRITE_ENABLE,
        vk::DynamicState::DEPTH_COMPARE_OP,
        vk::DynamicState::STENCIL_TEST_ENABLE,
        vk::DynamicState::STENCIL_OP,
        vk::DynamicState::DEPTH_BIAS_ENABLE,
    ];
    if depth_bounds {
        dynamic.push(vk::DynamicState::DEPTH_BOUNDS);
        dynamic.push(vk::DynamicState::DEPTH_BOUNDS_TEST_ENABLE);
    }
    dynamic
}

/// The parts a pipeline is linked from (mesh pipelines have no vertex
/// input).
pub fn parts(mesh: bool) -> &'static [Part] {
    if mesh {
        &[Part::PreRaster, Part::Fragment, Part::Output]
    } else {
        &[Part::VertexInput, Part::PreRaster, Part::Fragment, Part::Output]
    }
}

/// Link `libraries` into a graphics pipeline without link-time
/// optimisation (any thread).
pub fn link(device: &ash::Device, layout: vk::PipelineLayout, libraries: &[vk::Pipeline]) -> Result<vk::Pipeline, vk::Result> {
    let mut libs = vk::PipelineLibraryCreateInfoKHR::default().libraries(libraries);
    let info = vk::GraphicsPipelineCreateInfo::default().layout(layout).push_next(&mut libs);
    // SAFETY: the libraries are live pipeline libraries of `device` made
    // with the same layout; the structures live through the call.
    let pipelines = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) }.map_err(|(_, e)| e)?;
    Ok(pipelines[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::state::BlendKey;

    fn ff() -> FixedFunctionKey {
        FixedFunctionKey {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST.as_raw(),
            rect: RectMode::None,
            primitive_restart: false,
            vertex_input: VertexInputKey::default(),
            polygon_mode: 0,
            depth_clip: true,
            depth_clamp: false,
            negative_one_to_one: false,
            color_formats: [vk::Format::R8G8B8A8_UNORM.as_raw(); 8],
            color_count: 1,
            blend: [BlendKey::default(); 8],
            depth_format: vk::Format::D32_SFLOAT.as_raw(),
            stencil_format: 0,
            samples: 1,
            centered_samples: false,
            sample_shading: false,
            sample_mask: u32::MAX,
            logic_op: None,
            viewport_count: 1,
            provoking_last: false,
        }
    }

    #[test]
    fn modes_parse() {
        assert_eq!(parse_mode(None, false), Mode::Off);
        assert_eq!(parse_mode(Some("0"), false), Mode::Off);
        assert_eq!(parse_mode(Some("off"), false), Mode::Off);
        assert_eq!(parse_mode(Some("link"), false), Mode::Link);
        assert_eq!(parse_mode(Some("rebuild"), false), Mode::Rebuild);
        assert_eq!(parse_mode(Some("1"), false), Mode::Link);
        assert_eq!(parse_mode(Some("1"), true), Mode::Off);
        assert_eq!(parse_mode(Some("bogus"), false), Mode::Off);
    }

    /// Pipelines that differ only in output state share their shader
    /// libraries, and the other way round.
    #[test]
    fn parts_key_only_their_own_state() {
        let a = ff();
        let mut b = ff();
        b.blend[0].enable = true;
        b.color_formats[0] = vk::Format::R16G16B16A16_SFLOAT.as_raw();
        let l = vk::PipelineLayout::null();
        for part in [Part::VertexInput, Part::PreRaster, Part::Fragment] {
            assert_eq!(LibKey::of(part, 1, 2, l, None, false, &a), LibKey::of(part, 1, 2, l, None, false, &b), "{part:?}");
        }
        assert_ne!(LibKey::of(Part::Output, 1, 2, l, None, false, &a), LibKey::of(Part::Output, 1, 2, l, None, false, &b));
        // Another pixel shader: only the fragment part changes.
        assert_eq!(LibKey::of(Part::PreRaster, 1, 2, l, None, false, &a), LibKey::of(Part::PreRaster, 1, 3, l, None, false, &a));
        assert_ne!(LibKey::of(Part::Fragment, 1, 2, l, None, false, &a), LibKey::of(Part::Fragment, 1, 3, l, None, false, &a));
        // Rasterization state belongs to the pre-rasterization part.
        let mut c = ff();
        c.depth_clip = false;
        assert_ne!(LibKey::of(Part::PreRaster, 1, 2, l, None, false, &a), LibKey::of(Part::PreRaster, 1, 2, l, None, false, &c));
        assert_eq!(LibKey::of(Part::Fragment, 1, 2, l, None, false, &a), LibKey::of(Part::Fragment, 1, 2, l, None, false, &c));
        // Multisample state is in both fragment parts (they must agree).
        let mut d = ff();
        d.samples = 4;
        assert_ne!(LibKey::of(Part::Fragment, 1, 2, l, None, false, &a), LibKey::of(Part::Fragment, 1, 2, l, None, false, &d));
        assert_ne!(LibKey::of(Part::Output, 1, 2, l, None, false, &a), LibKey::of(Part::Output, 1, 2, l, None, false, &d));
    }

    #[test]
    fn dynamic_states_belong_to_one_part() {
        for s in dynamic_states(true) {
            let n = [Part::VertexInput, Part::PreRaster, Part::Fragment, Part::Output].iter().filter(|p| p.owns_dynamic(s)).count();
            assert_eq!(n, 1, "{s:?}");
        }
    }
}
