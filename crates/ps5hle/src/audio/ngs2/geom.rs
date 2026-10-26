// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Panning, geometry and waveform-block helpers (pure functions over the
//! guest struct images).

use super::engine::{
    default_level, WaveformBlock, WaveformFormat, ERR_INVALID_WAVEFORM_FORMAT, ERR_UNKNOWN_WAVEFORM_FORMAT, OK,
};
use super::vag;
use super::waveform::{WAVEFORM_TYPE_ATRAC9, WAVEFORM_TYPE_VAG};
use crate::audio::atrac9::Atrac9Config;

/// sizeof(SceNgs2PanWork): speakerAngle[8], unitAngle, numSpeakers.
pub(crate) const PAN_WORK_SIZE: usize = 40;
/// sizeof(SceNgs2GeomListenerParam).
pub(crate) const GEOM_LISTENER_PARAM_SIZE: usize = 60;
/// sizeof(SceNgs2GeomListenerWork).
pub(crate) const GEOM_LISTENER_WORK_SIZE: usize = 96;
/// sizeof(SceNgs2GeomSourceParam).
pub(crate) const GEOM_SOURCE_PARAM_SIZE: usize = 108;
/// sizeof(SceNgs2GeomAttribute).
pub(crate) const GEOM_ATTRIBUTE_SIZE: usize = 308;

fn putf(b: &mut [u8], o: usize, v: f32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}

fn put32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}

fn getf(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn get32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

/// One unit angle in degrees and in radians: the only units accepted.
const DEGREES: f32 = 360.0;
const RADIANS: f32 = 2.0 * std::f32::consts::PI;
/// Parameters one volume matrix takes at most.
pub(crate) const PAN_MAX_PARAMS: usize = 8;

/// Speaker layout of a pan work area: the order the caller's speaker
/// angles are taken in, the 5.1/7.1 channel each speaker feeds, the
/// default azimuths (degrees, counter-clockwise from the front) and the
/// width of the mix the speakers feed.
struct PanLayout {
    input_order: &'static [usize],
    channels: &'static [usize],
    default_degrees: &'static [f32],
    mix_channels: usize,
}

const STEREO_LAYOUT: PanLayout =
    PanLayout { input_order: &[1, 0], channels: &[1, 0], default_degrees: &[90.0, 270.0], mix_channels: 2 };
const QUAD_LAYOUT: PanLayout = PanLayout {
    input_order: &[1, 3, 2, 0],
    channels: &[1, 5, 4, 0],
    default_degrees: &[40.0, 110.0, 250.0, 320.0],
    mix_channels: 6,
};
const FIVE_LAYOUT: PanLayout = PanLayout {
    input_order: &[2, 1, 4, 3, 0],
    channels: &[2, 1, 5, 4, 0],
    default_degrees: &[0.0, 40.0, 110.0, 250.0, 320.0],
    mix_channels: 6,
};
const SEVEN_LAYOUT: PanLayout = PanLayout {
    input_order: &[2, 1, 4, 6, 5, 3, 0],
    channels: &[2, 1, 5, 7, 6, 4, 0],
    default_degrees: &[0.0, 40.0, 110.0, 150.0, 210.0, 250.0, 320.0],
    mix_channels: 8,
};

fn pan_layout(num_speakers: u32) -> Option<&'static PanLayout> {
    match num_speakers {
        2 => Some(&STEREO_LAYOUT),
        4 => Some(&QUAD_LAYOUT),
        5 => Some(&FIVE_LAYOUT),
        7 => Some(&SEVEN_LAYOUT),
        _ => None,
    }
}

fn is_unit_angle(unit: f32) -> bool {
    unit == DEGREES || unit == RADIANS
}

/// Why a pan call fails (each maps to its own SCE_NGS2_ERROR_*).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PanError {
    InvalidOutAddress,
    UnitAngle,
    Speaker,
    MatrixFormat,
    Work,
    Param,
    NumChannels,
    /// An input on which the console's library did not return within 60 s
    /// (an infinite angle, one too large for a turn to change it, a speaker
    /// angle wrapping to exactly the unit angle): not reproduced.
    NoReturn,
}

fn default_speakers(work: &mut [u8; PAN_WORK_SIZE], layout: &PanLayout, num_speakers: u32, unit_angle: f32) {
    let n = num_speakers as usize;
    *work = [0u8; PAN_WORK_SIZE];
    let mut angles = [0f32; 8];
    angles[..n].copy_from_slice(layout.default_degrees);
    angles[n] = layout.default_degrees[0] + DEGREES;
    let mut unit = DEGREES;
    if unit_angle == RADIANS {
        for a in &mut angles[..n] {
            *a = *a / 180.0 * std::f32::consts::PI;
        }
        unit = RADIANS;
    }
    for (i, a) in angles.iter().enumerate() {
        putf(work, i * 4, *a);
    }
    putf(work, 32, unit);
    put32(work, 36, num_speakers);
}

/// sceNgs2PanInit over the work image `work`: the speakers' azimuths (the
/// caller's angles, taken in the layout's input order and wrapped into one
/// turn), then the first again one turn later, the unit angle and the
/// speaker count; the slots past those keep what they held. Without angles
/// the default layout is used, in degrees unless the unit is radians, and
/// the rest of the image is cleared; without angles and with no speakers
/// the image is the 7-speaker default in degrees with the unit angle
/// radians or degrees. A NaN speaker angle is stored as it is.
pub(crate) fn pan_init(work: &mut [u8; PAN_WORK_SIZE], speaker_angles: Option<&[f32]>, unit_angle: f32, num_speakers: u32) -> Result<(), PanError> {
    if speaker_angles.is_some() && !is_unit_angle(unit_angle) {
        return Err(PanError::UnitAngle);
    }
    if speaker_angles.is_none() && num_speakers == 0 {
        default_speakers(work, &SEVEN_LAYOUT, 7, DEGREES);
        putf(work, 32, if unit_angle == RADIANS { RADIANS } else { DEGREES });
        return Ok(());
    }
    let layout = pan_layout(num_speakers).ok_or(PanError::Speaker)?;
    let Some(input) = speaker_angles else {
        default_speakers(work, layout, num_speakers, unit_angle);
        return Ok(());
    };
    let n = num_speakers as usize;
    let mut angles = [0f32; 8];
    for i in 0..n {
        let mut angle = *input.get(layout.input_order[i]).ok_or(PanError::NoReturn)?;
        while angle < 0.0 {
            if angle + unit_angle == angle {
                return Err(PanError::NoReturn);
            }
            angle += unit_angle;
        }
        while angle >= unit_angle {
            if !(unit_angle < angle) || angle - unit_angle == angle {
                return Err(PanError::NoReturn);
            }
            angle -= unit_angle;
        }
        angles[i] = angle;
    }
    angles[n] = angles[0] + unit_angle;
    for (i, a) in angles[..=n].iter().enumerate() {
        putf(work, i * 4, *a);
    }
    putf(work, 32, unit_angle);
    put32(work, 36, num_speakers);
    Ok(())
}

/// Output channels of a volume matrix format (0 means stereo).
pub(crate) fn matrix_channels(matrix_format: u32) -> u32 {
    if matrix_format == 0 { 2 } else { matrix_format.min(8) }
}

/// sizeof(SceNgs2PanParam): angle, distance, fbwLevel, lfeLevel.
pub(crate) const PAN_PARAM_SIZE: usize = 16;

/// One SceNgs2PanParam.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct PanParam {
    pub angle: f32,
    pub distance: f32,
    pub fbw_level: f32,
    pub lfe_level: f32,
}

impl PanParam {
    pub(crate) fn parse(b: &[u8]) -> Self {
        Self { angle: getf(b, 0), distance: getf(b, 4), fbw_level: getf(b, 8), lfe_level: getf(b, 12) }
    }
}

/// Index of the LFE channel in the 5.1 / 7.1 orders (L R C LFE Ls Rs [Lb Rb]).
const LFE_CHANNEL: usize = 3;

fn equal_power(position: f32) -> f32 {
    (position * (std::f32::consts::PI / 2.0)).sin()
}

/// Split `weight` between speakers `speaker - 1` and `speaker` when `angle`
/// lies between them.
fn add_between_speakers(angles: &[f32; 8], speaker: usize, angle: f32, weight: f32, gains: &mut [f32; 8]) {
    let (from, to) = (angles[speaker - 1], angles[speaker]);
    if !(angle >= from) || !(to > angle) {
        return;
    }
    let position = (angle - from) / (to - from);
    gains[speaker - 1] += equal_power(1.0 - position) * weight;
    gains[speaker] += equal_power(position) * weight;
}

/// The work-image checks of sceNgs2PanGetVolumeMatrix, in the console's
/// order: matrix format, speaker count, unit angle, parameter count.
pub(crate) fn pan_check(work: &[u8], num_params: usize, matrix_format: u32) -> Result<(), PanError> {
    if !matches!(matrix_format, 1 | 2 | 6 | 8) {
        return Err(PanError::MatrixFormat);
    }
    pan_layout(get32(work, 36)).ok_or(PanError::Speaker)?;
    if !is_unit_angle(getf(work, 32)) {
        return Err(PanError::UnitAngle);
    }
    if num_params > PAN_MAX_PARAMS {
        return Err(PanError::NumChannels);
    }
    Ok(())
}

/// sceNgs2PanGetVolumeMatrix: a `matrix_format` x `params` matrix
/// (destination-major, one column per parameter).
///
/// Each parameter's angle is placed between its two neighbouring speakers
/// with equal-power gains; the opposite direction gets the rest, weighted
/// by the distance (-1 opposite .. 1 toward, clamped). The speakers feed
/// their 5.1/7.1 channels (scaled by `fbw_level`, the LFE channel takes
/// `lfe_level`), and that mix reaches the matrix's width through the
/// default channel map. NaN angles and distances are not rejected: a NaN
/// angle (of the parameter or of the first speaker) reaches no speaker, a
/// NaN distance makes the levels it weights NaN (measured on a PS5).
pub(crate) fn pan_volume_matrix(work: &[u8], params: &[PanParam], matrix_format: u32) -> Result<Vec<f32>, PanError> {
    pan_check(work, params.len(), matrix_format)?;
    let num_speakers = get32(work, 36);
    let unit = getf(work, 32);
    let mut angles = [0f32; 8];
    for (i, a) in angles.iter_mut().enumerate() {
        *a = getf(work, i * 4);
    }
    let layout = pan_layout(num_speakers).ok_or(PanError::Speaker)?;
    let n = num_speakers as usize;
    let format = matrix_format as usize;
    let first = angles[0];
    let mut out = vec![0f32; params.len() * format];
    for (pi, p) in params.iter().enumerate() {
        let distance = if p.distance > 1.0 {
            1.0
        } else if p.distance < -1.0 {
            -1.0
        } else {
            p.distance
        };
        let toward_weight = ((distance + 1.0) * 0.5 * std::f32::consts::PI * 0.5).sin();
        let opposite_weight = ((1.0 - distance) * 0.5 * std::f32::consts::PI * 0.5).sin();
        let mut angle = p.angle;
        while angle < first || angle >= first + unit {
            let before = angle;
            if angle >= first + unit {
                angle -= unit;
            }
            if angle < first {
                angle += unit;
            }
            if angle == before {
                // Infinite, or too large to wrap into one turn.
                return Err(PanError::NoReturn);
            }
        }
        let mut opposite = angle + unit * 0.5;
        if !(opposite < first + unit) {
            opposite -= unit;
        }
        let mut gains = [0f32; 8];
        for speaker in 1..=n {
            add_between_speakers(&angles, speaker, angle, toward_weight, &mut gains);
            add_between_speakers(&angles, speaker, opposite, opposite_weight, &mut gains);
        }
        let mut mix = [0f32; 8];
        mix[layout.channels[0]] = gains[0] + gains[n];
        for speaker in 1..n {
            mix[layout.channels[speaker]] = gains[speaker];
        }
        for (channel, m) in mix.iter_mut().enumerate() {
            *m = if channel == LFE_CHANNEL { p.lfe_level } else { *m * p.fbw_level };
        }
        for dest in 0..format {
            let level = &mut out[dest * params.len() + pi];
            for (channel, &m) in mix.iter().enumerate().take(layout.mix_channels) {
                *level += default_level(layout.mix_channels, channel, format, dest).unwrap_or(0.0) * m;
            }
        }
    }
    Ok(out)
}

pub(crate) fn geom_reset_listener_param() -> [u8; GEOM_LISTENER_PARAM_SIZE] {
    let mut p = [0u8; GEOM_LISTENER_PARAM_SIZE];
    putf(&mut p, 12 + 8, 1.0); // orientFront.z
    putf(&mut p, 24 + 4, 1.0); // orientUp.y
    putf(&mut p, 48, 343.0); // soundSpeed
    p
}

pub(crate) fn geom_reset_source_param() -> [u8; GEOM_SOURCE_PARAM_SIZE] {
    let mut p = [0u8; GEOM_SOURCE_PARAM_SIZE];
    putf(&mut p, 24 + 8, 1.0); // direction.z
    putf(&mut p, 36, 1.0); // cone.innerLevel
    putf(&mut p, 40, 360.0); // cone.innerAngle
    putf(&mut p, 44, 1.0); // cone.outerLevel
    putf(&mut p, 48, 360.0); // cone.outerAngle
    put32(&mut p, 52, 0); // rolloff.model
    putf(&mut p, 56, 1_000_000.0); // rolloff.maxDistance
    putf(&mut p, 60, 1.0); // rolloff.rolloffFactor
    putf(&mut p, 64, 1.0); // rolloff.referenceDistance
    putf(&mut p, 68, 1.0); // dopplerFactor
    putf(&mut p, 72, 1.0); // fbwLevel
    putf(&mut p, 76, 1.0); // lfeLevel
    putf(&mut p, 80, 1.0); // maxLevel
    putf(&mut p, 84, 0.0); // minLevel
    put32(&mut p, 92, 2); // numSpeakers
    put32(&mut p, 96, 2); // matrixFormat
    p
}

/// sceNgs2GeomCalcListener: identity transform, listener velocity and
/// sound speed (default 343 m/s).
pub(crate) fn geom_calc_listener(param: &[u8; GEOM_LISTENER_PARAM_SIZE], flags: u32) -> [u8; GEOM_LISTENER_WORK_SIZE] {
    let mut w = [0u8; GEOM_LISTENER_WORK_SIZE];
    for i in 0..4 {
        putf(&mut w, (i * 4 + i) * 4, 1.0);
    }
    w[64..76].copy_from_slice(&param[36..48]);
    let speed = getf(param, 48);
    putf(&mut w, 76, if speed > 0.0 { speed } else { 343.0 });
    put32(&mut w, 80, flags & 1);
    w
}

/// sceNgs2GeomApply: unity pitch, the source position for 3D audio and a
/// diagonal level matrix at the source's maximum level.
pub(crate) fn geom_apply(source: &[u8; GEOM_SOURCE_PARAM_SIZE]) -> [u8; GEOM_ATTRIBUTE_SIZE] {
    let mut a = [0u8; GEOM_ATTRIBUTE_SIZE];
    putf(&mut a, 0, 1.0);
    a[260..272].copy_from_slice(&source[0..12]);
    let (max_level, min_level) = (getf(source, 80), getf(source, 84));
    putf(&mut a, 272, min_level.max(max_level));
    let channels = matrix_channels(get32(source, 96)) as usize;
    let level = if max_level > 0.0 { max_level } else { 1.0 };
    for ch in 0..channels {
        putf(&mut a, 4 + (ch * 8 + ch) * 4, level);
    }
    a
}

/// sceNgs2CalcWaveformBlock: the byte range holding `num_samples` samples
/// from `sample_pos`.
pub(crate) fn calc_waveform_block(format: Option<&WaveformFormat>, sample_pos: u32, num_samples: u32) -> (i32, WaveformBlock) {
    let mut block = WaveformBlock::default();
    let Some(format) = format.filter(|f| (1..=8).contains(&f.num_channels) && (1..=192000).contains(&f.sample_rate))
    else {
        return (ERR_INVALID_WAVEFORM_FORMAT, block);
    };
    let (frame_samples, frame_bytes) = match format.waveform_type {
        WAVEFORM_TYPE_ATRAC9 => {
            let Ok(c) = Atrac9Config::new(&format.config_bytes()) else {
                return (ERR_INVALID_WAVEFORM_FORMAT, block);
            };
            if c.channel_count as u32 != format.num_channels || c.sample_rate as u32 != format.sample_rate {
                return (ERR_INVALID_WAVEFORM_FORMAT, block);
            }
            (c.superframe_samples as u64, c.superframe_bytes as u64)
        }
        WAVEFORM_TYPE_VAG => (vag::UNIT_SAMPLES, vag::UNIT_BYTES * format.num_channels as u64),
        ty => match super::engine::pcm_frame_bytes(ty, format.num_channels) {
            Some(bytes) => (1, bytes),
            None => return (ERR_UNKNOWN_WAVEFORM_FORMAT, block),
        },
    };
    let skip = sample_pos as u64 % frame_samples;
    block.data_offset = sample_pos as u64 / frame_samples * frame_bytes;
    if num_samples != 0 {
        block.data_size = (skip + num_samples as u64).div_ceil(frame_samples) * frame_bytes;
        block.num_skip_samples = skip as u32;
    }
    block.num_samples = num_samples;
    (OK, block)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(angle: f32, distance: f32, fbw_level: f32, lfe_level: f32) -> PanParam {
        PanParam { angle, distance, fbw_level, lfe_level }
    }

    fn angles(w: &[u8]) -> Vec<f32> {
        (0..8).map(|i| getf(w, i * 4)).collect()
    }

    /// sceNgs2PanInit over a zeroed work image.
    fn init(speaker_angles: Option<&[f32]>, unit_angle: f32, num_speakers: u32) -> Result<[u8; PAN_WORK_SIZE], PanError> {
        let mut w = [0u8; PAN_WORK_SIZE];
        pan_init(&mut w, speaker_angles, unit_angle, num_speakers).map(|()| w)
    }

    /// Work images of sceNgs2PanInit, as the console's library builds them.
    #[test]
    fn pan_init_layouts() {
        let pi = std::f32::consts::PI;
        let w = init(None, 360.0, 2).unwrap();
        assert_eq!((get32(&w, 36), getf(&w, 32)), (2, 360.0));
        assert_eq!(&angles(&w)[..3], [90.0, 270.0, 450.0]);
        let w = init(None, 2.0 * pi, 2).unwrap();
        assert_eq!(getf(&w, 32), 2.0 * pi);
        assert_eq!(&angles(&w)[..3], [90.0 / 180.0 * pi, 270.0 / 180.0 * pi, 450.0]);
        // Without angles any unit gives degrees.
        let w = init(None, 100.0, 4).unwrap();
        assert_eq!((getf(&w, 32), angles(&w)[4]), (360.0, 400.0));
        // Caller angles in the layout's input order, wrapped into one turn.
        let w = init(Some(&[-30.0, 30.0, 0.0, -110.0, 470.0]), 360.0, 5).unwrap();
        assert_eq!(&angles(&w)[..6], [0.0, 30.0, 110.0, 250.0, 330.0, 360.0]);

        for count in [1, 3, 6, 8] {
            assert_eq!(init(None, 360.0, count), Err(PanError::Speaker), "{count} speakers");
            assert_eq!(init(Some(&[0.0; 8]), 360.0, count), Err(PanError::Speaker), "{count} speakers");
        }
        assert_eq!(init(Some(&[0.0; 8]), 360.0, 0), Err(PanError::Speaker));
        assert_eq!(init(Some(&[-30.0, 30.0]), 180.0, 2), Err(PanError::UnitAngle));
        assert!(init(Some(&[-30.0, 30.0]), 2.0 * pi, 2).is_ok());
        assert_eq!(init(Some(&[1e30, 30.0]), 360.0, 2), Err(PanError::NoReturn));
        assert_eq!(init(Some(&[-1e30, 30.0]), 360.0, 2), Err(PanError::NoReturn));
        assert_eq!(init(Some(&[f32::INFINITY, 30.0]), 360.0, 2), Err(PanError::NoReturn));
        assert_eq!(init(Some(&[720.0, 30.0]), 360.0, 2), Err(PanError::NoReturn), "an angle wrapping onto one turn");
        assert!(init(Some(&[-360.0, 30.0]), 360.0, 2).is_ok());
    }

    /// PS5 work images: no speakers and no angles is the 7-speaker default
    /// in degrees (unit radians or degrees); a NaN speaker angle is kept;
    /// caller angles leave the slots past the wrap-around entry alone.
    #[test]
    fn pan_init_ps5_images() {
        let pi = std::f32::consts::PI;
        let seven = [0.0, 40.0, 110.0, 150.0, 210.0, 250.0, 320.0, 360.0];
        for (unit, stored) in [(360.0, 360.0), (2.0 * pi, 2.0 * pi), (180.0, 360.0), (f32::NAN, 360.0)] {
            let mut w = [0x5Au8; PAN_WORK_SIZE];
            assert_eq!(pan_init(&mut w, None, unit, 0), Ok(()));
            assert_eq!((get32(&w, 36), getf(&w, 32)), (7, stored), "unit {unit}");
            assert_eq!(angles(&w), seven);
        }
        let w = init(Some(&[0.0, f32::NAN]), 360.0, 2).unwrap();
        assert!(angles(&w)[0].is_nan() && angles(&w)[1] == 0.0 && angles(&w)[2].is_nan());
        assert_eq!((get32(&w, 36), getf(&w, 32)), (2, 360.0));
        let mut w = [0x5Au8; PAN_WORK_SIZE];
        assert_eq!(pan_init(&mut w, Some(&[-30.0, 30.0]), 360.0, 2), Ok(()));
        assert_eq!(&angles(&w)[..3], [30.0, 330.0, 390.0]);
        assert_eq!(&w[12..32], &[0x5Au8; 20], "slots 3..8 untouched");
        assert_eq!(pan_init(&mut w, None, 360.0, 2), Ok(()));
        assert_eq!(&angles(&w)[3..], [0.0; 5], "the default layout clears them");
    }

    /// Volume matrices measured on the console's library (destination-major).
    #[test]
    fn pan_volume_matrices() {
        let pi = std::f32::consts::PI;
        type Case = (Option<Vec<f32>>, f32, u32, Vec<PanParam>, u32, Vec<f32>);
        let cases: Vec<Case> = vec![
            (None, 360.0, 2, (0..8).map(|i| p(45.0 * i as f32, 1.0, 1.0, 0.0)).collect(), 2,
             vec![0.707106769, 0.382683456, 0.0, 0.382683456, 0.707106769, 0.923879504, 1.0, 0.923879504,
                  0.707106769, 0.923879504, 1.0, 0.923879504, 0.707106769, 0.382683456, 0.0, 0.382683456]),
            (None, 360.0, 2,
             vec![p(30.0, 0.0, 1.0, 0.0), p(30.0, -1.0, 1.0, 0.0), p(30.0, 0.5, 0.5, 0.0), p(30.0, 3.0, 1.0, 0.0), p(-30.0, -0.25, 1.0, 0.0)], 2,
             vec![0.965925813, 0.866025448, 0.39667666, 0.49999997, 0.896872759, 0.965925813, 0.49999997, 0.495722443, 0.866025448, 0.997858942]),
            (None, 360.0, 2, vec![p(60.0, 1.0, 1.0, 0.5)], 1, vec![0.866025388]),
            (None, 360.0, 2, vec![p(60.0, 1.0, 1.0, 0.5)], 6, vec![0.258819073, 0.965925813, 0.0, 0.0, 0.0, 0.0]),
            (None, 2.0 * pi, 2, vec![p(0.25 * pi, 1.0, 1.0, 0.0), p(1.5 * pi, 1.0, 1.0, 0.0)], 2, vec![0.999965429, 1.0, 0.00831161533, 0.0]),
            (None, 360.0, 4, vec![p(100.0, 1.0, 1.0, 0.25)], 6, vec![0.0, 0.222520918, 0.0, 0.25, 0.0, 0.974927902]),
            (None, 360.0, 4, vec![p(100.0, 1.0, 1.0, 0.25)], 2, vec![0.0, 0.709984899]),
            (None, 360.0, 5, vec![p(70.0, 1.0, 0.8, 0.5), p(200.0, 0.5, 1.0, 0.0)], 6,
             vec![0.0, 0.0, 0.625465155, 0.270598054, 0.0, 0.270598054, 0.5, 0.0, 0.0, 0.782271147, 0.498791903, 0.491533577]),
            (None, 360.0, 5, vec![p(70.0, 1.0, 0.8, 0.5)], 1, vec![0.691666603]),
            (None, 360.0, 7, vec![p(180.0, 1.0, 1.0, 0.25)], 8, vec![0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.707106769, 0.707106769]),
            (None, 360.0, 7, vec![p(180.0, 1.0, 1.0, 0.25)], 6, vec![0.0, 0.0, 0.0, 0.25, 0.707106769, 0.707106769]),
            (Some(vec![-pi / 6.0, pi / 6.0, 0.0, -110.0 * pi / 180.0, 110.0 * pi / 180.0]), 2.0 * pi, 5,
             vec![p(1.0, 1.0, 1.0, 0.0), p(-2.5, 0.2, 1.0, 0.1)], 6,
             vec![0.0, 0.0, 0.859783232, 0.582614243, 0.0, 0.0, 0.0, 0.100000001, 0.0, 0.753403723, 0.510659218, 0.372568965]),
            (Some(vec![-30.0, 30.0, 0.0, -90.0, 90.0, -150.0, 150.0]), 360.0, 7, vec![p(-120.0, 1.0, 1.0, 0.0)], 1, vec![0.707106769]),
            (Some(vec![-45.0, 45.0]), 360.0, 2, vec![p(10.0, 1.0, 1.0, 0.0), p(720.0 + 170.0, 1.0, 1.0, 0.0)], 2,
             vec![0.57357651, 0.664795876, 0.819152057, 0.747025073]),
            (Some(vec![-40.0, 40.0, -120.0, 120.0]), 360.0, 4, vec![p(-80.0, 0.75, 1.0, 0.0)], 8,
             vec![0.69351995, 0.0746578425, 0.0, 0.0, 0.69351995, 0.180239946, 0.0, 0.0]),
        ];
        for (i, (speakers, unit, count, params, format, expected)) in cases.iter().enumerate() {
            let w = init(speakers.as_deref(), *unit, *count).unwrap();
            let m = pan_volume_matrix(&w, params, *format).unwrap();
            assert_eq!(m.len(), expected.len(), "case {i}");
            for (k, (a, b)) in m.iter().zip(expected).enumerate() {
                assert!((a - b).abs() < 1e-6, "case {i} level {k}: {a} != {b} ({m:?})");
            }
        }
    }

    #[test]
    fn pan_volume_matrix_rejects() {
        let w = init(None, 360.0, 2).unwrap();
        let front = p(0.0, 1.0, 1.0, 0.0);
        assert_eq!(pan_volume_matrix(&w, &[], 2), Ok(Vec::new()));
        for format in [0, 3, 4] {
            assert_eq!(pan_volume_matrix(&w, &[front], format), Err(PanError::MatrixFormat), "format {format}");
        }
        assert_eq!(pan_volume_matrix(&w, &[PanParam::default(); 9], 3), Err(PanError::MatrixFormat));
        assert!(pan_volume_matrix(&w, &[front], 8).is_ok());
        assert_eq!(pan_volume_matrix(&w, &[PanParam::default(); 9], 2), Err(PanError::NumChannels));
        assert!(pan_volume_matrix(&w, &[PanParam::default(); 8], 2).is_ok());
        assert_eq!(pan_volume_matrix(&w, &[p(1e30, 1.0, 1.0, 0.0)], 2), Err(PanError::NoReturn), "an angle too large to wrap");
        assert_eq!(pan_volume_matrix(&w, &[p(f32::INFINITY, 1.0, 1.0, 0.0)], 2), Err(PanError::NoReturn));
        let mut bad = w;
        put32(&mut bad, 36, 3);
        assert_eq!(pan_volume_matrix(&bad, &[front], 2), Err(PanError::Speaker));
        assert_eq!(pan_volume_matrix(&bad, &[front], 3), Err(PanError::MatrixFormat));
        assert_eq!(pan_volume_matrix(&bad, &[PanParam::default(); 9], 2), Err(PanError::Speaker));
        putf(&mut bad, 32, 100.0);
        assert_eq!(pan_volume_matrix(&bad, &[front], 2), Err(PanError::Speaker));
        put32(&mut bad, 36, 2);
        assert_eq!(pan_volume_matrix(&bad, &[front], 2), Err(PanError::UnitAngle));
        assert_eq!(pan_volume_matrix(&bad, &[PanParam::default(); 9], 2), Err(PanError::UnitAngle));
    }

    /// NaN inputs measured on a PS5: a NaN parameter angle reaches no
    /// speaker (LFE still takes its level), a NaN distance makes the
    /// levels it weights NaN, a NaN first speaker gives silence.
    #[test]
    fn pan_volume_matrix_nan_inputs() {
        let w = init(None, 360.0, 2).unwrap();
        let m = pan_volume_matrix(&w, &[p(f32::NAN, 1.0, 1.0, 0.0)], 2).unwrap();
        assert_eq!(m, [0.0, 0.0]);
        let m = pan_volume_matrix(&w, &[p(0.0, 1.0, 1.0, 0.0), p(f32::NAN, 1.0, 1.0, 0.0)], 2).unwrap();
        assert!((m[0] - 0.707106769).abs() < 1e-6 && m[1] == 0.0 && (m[2] - 0.707106769).abs() < 1e-6 && m[3] == 0.0, "{m:?}");
        let seven = init(None, 360.0, 0).unwrap();
        let nan_each = [p(f32::NAN, 1.0, 1.0, 0.5), p(30.0, f32::NAN, 1.0, 0.5), p(30.0, 1.0, f32::NAN, 0.5), p(30.0, 1.0, 1.0, f32::NAN)];
        let m = pan_volume_matrix(&seven, &nan_each, 8).unwrap();
        for channel in 0..8 {
            assert_eq!(m[channel * 4], if channel == 3 { 0.5 } else { 0.0 }, "channel {channel}");
            for k in 1..4 {
                assert!(m[channel * 4 + k].is_nan(), "channel {channel} param {k}: {m:?}");
            }
        }
        let pair = init(Some(&[-30.0, 30.0]), 360.0, 2).unwrap();
        let m = pan_volume_matrix(&pair, &[p(0.0, 1.0, 1.0, 0.0), p(30.0, f32::NAN, 1.0, 0.0)], 2).unwrap();
        assert!((m[0] - 0.707106769).abs() < 1e-6 && m[1].is_nan() && (m[2] - 0.707106769).abs() < 1e-6 && m[3].is_nan(), "{m:?}");
        let mut silent = w;
        putf(&mut silent, 0, f32::NAN);
        assert_eq!(pan_volume_matrix(&silent, &[p(30.0, 1.0, 1.0, 0.0)], 2), Ok(vec![0.0, 0.0]));
    }
}
