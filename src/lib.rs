// Copyright (c) 2022 solarliner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#![feature(portable_simd)]
#![feature(array_from_fn)]
#![feature(const_for)]

use std::f32::consts::TAU;
use std::{
    simd::{f32x2, Simd},
    sync::Arc,
};

use biquad::{Biquad, BiquadParams};
use editor::DelayEditor;
use nih_plug::prelude::*;

use early::Early;
use nih_plug_vizia::ViziaState;
use pitch::PitchShifter;
use simdmath::simd_f32tanh;

use crate::delay::Delay;

pub mod biquad;
pub mod delay;
mod diffusion;
mod early;
mod editor;
mod hadamard;
mod householder;
pub mod pitch;
mod simdmath;

#[derive(Params)]
struct DelayParams {
    #[id = "ersize"]
    size: FloatParam,
    #[id = "fbck"]
    feedback: FloatParam,
    #[id = "delay"]
    delay: FloatParam,
    #[id = "mddpt"]
    mod_depth: FloatParam,
    #[id = "mdspd"]
    mod_speed: FloatParam,
    #[id = "dlow"]
    damp_low: FloatParam,
    #[id = "dhigh"]
    damp_high: FloatParam,
    // #[id = "shimr"]
    pitch_amt: FloatParam,
}

impl Default for DelayParams {
    fn default() -> Self {
        Self {
            size: FloatParam::new("Size", 0.5, FloatRange::Linear { min: 0., max: 1. })
                .with_unit("%")
                .with_string_to_value(formatters::s2v_f32_percentage())
                .with_value_to_string(formatters::v2s_f32_percentage(2))
                .with_smoother(SmoothingStyle::Linear(20.)),
            feedback: FloatParam::new("Feedback", 0.7, FloatRange::Linear { min: 0., max: 1.25 })
                .with_unit("%")
                .with_string_to_value(formatters::s2v_f32_percentage())
                .with_value_to_string(formatters::v2s_f32_percentage(2)),
            delay: FloatParam::new("Delay", 0.2, FloatRange::Linear { min: 1e-3, max: 2. })
                .with_unit("s")
                .with_smoother(SmoothingStyle::Linear(200.)),
            mod_depth: FloatParam::new(
                "Mod Depth",
                0.1,
                FloatRange::Skewed {
                    min: 0.,
                    max: 1.,
                    factor: FloatRange::skew_factor(-2.),
                },
            )
            .with_unit("%")
            .with_string_to_value(formatters::s2v_f32_percentage())
            .with_value_to_string(formatters::v2s_f32_percentage(2))
            .with_smoother(SmoothingStyle::Linear(200.)),
            mod_speed: FloatParam::new(
                "Mod Speed",
                0.3,
                FloatRange::Skewed {
                    min: 1e-3,
                    max: 3.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_smoother(SmoothingStyle::Exponential(150.0))
            .with_string_to_value(formatters::s2v_f32_hz_then_khz())
            .with_value_to_string(formatters::v2s_f32_hz_then_khz(2)),
            damp_low: FloatParam::new(
                "Low Damping",
                100.,
                FloatRange::Skewed {
                    min: 20.,
                    max: 20e3,
                    factor: FloatRange::skew_factor(-2.5),
                },
            )
            .with_smoother(SmoothingStyle::Logarithmic(100.))
            .with_string_to_value(formatters::s2v_f32_hz_then_khz())
            .with_value_to_string(formatters::v2s_f32_hz_then_khz(2)),
            damp_high: FloatParam::new(
                "High Damping",
                3000.,
                FloatRange::Skewed {
                    min: 20.,
                    max: 20e3,
                    factor: FloatRange::skew_factor(-2.5),
                },
            )
            .with_smoother(SmoothingStyle::Logarithmic(100.))
            .with_string_to_value(formatters::s2v_f32_hz_then_khz())
            .with_value_to_string(formatters::v2s_f32_hz_then_khz(2)),
            pitch_amt: FloatParam::new("Shimmer", 0.0, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_smoother(SmoothingStyle::Linear(100.0))
                .with_string_to_value(formatters::s2v_f32_percentage())
                .with_value_to_string(formatters::v2s_f32_percentage(2)),
        }
    }
}

struct Reverb {
    params: Arc<DelayParams>,
    editor_state: Arc<ViziaState>,
    diffusion: Early<4>,
    delay: Delay<f32x2>,
    damp_low: Biquad<2>,
    damp_high: Biquad<2>,
    pitch: PitchShifter<2>,
    phase: f32,
}

impl Reverb {
    fn new_with_params(params: Arc<DelayParams>, samplerate: f32) -> Self {
        Self {
            params,
            editor_state: DelayEditor::default_state(),
            diffusion: Early::new(samplerate),
            delay: Delay::new(samplerate as usize * 2),
            damp_low: Biquad::default(),
            damp_high: Biquad::default(),
            pitch: PitchShifter::new(f32::ceil(300.0 * samplerate) as _),
            phase: 0.,
        }
    }

    fn new(samplerate: f32) -> Self {
        Self::new_with_params(Arc::default(), samplerate)
    }

    fn next_sample(
        &mut self,
        samplerate: f32,
        size: f32,
        feedback: f32,
        delay: f32,
        mod_depth: f32,
        pitch_amt: f32,
        sample: Simd<f32, 2>,
    ) -> Simd<f32, 2> {
        let delayed = sample
            + self
                .delay
                .tap((delay * samplerate).max(1.).min(samplerate - 1.))
                * Simd::splat(feedback);
        let delayed = self.damp_low.next_sample(delayed);
        let delayed = self.damp_high.next_sample(delayed);
        let diffuse_input =
            Simd::gather_or_default(delayed.as_array(), Simd::from_array([0, 1, 0, 1]));
        let diffused = self.diffusion.next_sample(size, mod_depth, diffuse_input);
        let diffused = f32x2::gather_or_default(diffused.as_array(), Simd::from_array([0, 1]));
        let shifted = self.pitch.next_sample(samplerate, 2., diffused);
        let diffused = diffused * Simd::splat(1.0 - pitch_amt) + shifted * Simd::splat(pitch_amt);
        let diffused = simd_f32tanh(diffused);
        self.delay.push_next(diffused);
        diffused
    }

    fn tick_phase(&mut self, samplerate: f32, mod_speed: f32) {
        self.phase += mod_speed / samplerate;
        if self.phase > 1. {
            self.phase -= 1.;
        }
    }
}

impl Default for Reverb {
    fn default() -> Self {
        Self::new(44100.)
    }
}

impl Plugin for Reverb {
    const NAME: &'static str = "NIH Reverb";
    const VENDOR: &'static str = env!("CARGO_PKG_AUTHORS");
    const URL: &'static str = env!("CARGO_PKG_HOMEPAGE");
    const EMAIL: &'static str = "N/A";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&self) -> Option<Box<dyn Editor>> {
        DelayEditor::create(self.params.clone(), self.editor_state.clone())
    }

    fn accepts_bus_config(&self, config: &BusConfig) -> bool {
        config.num_input_channels == config.num_output_channels && config.num_input_channels == 2
    }

    fn initialize(
        &mut self,
        _bus_config: &BusConfig,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext,
    ) -> bool {
        *self = Self::new_with_params(self.params.clone(), buffer_config.sample_rate);
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext,
    ) -> ProcessStatus {
        let samplerate = context.transport().sample_rate;
        for mut channels in buffer.iter_samples() {
            let feedback = self.params.feedback.smoothed.next();
            let size = self.params.size.smoothed.next();
            let mod_depth = self.params.mod_depth.smoothed.next();
            let mod_speed = self.params.mod_speed.smoothed.next();
            let pitch_amt = self.params.pitch_amt.smoothed.next();
            let delay =
                self.params.delay.smoothed.next() + 15e-3 * mod_depth * f32::sin(TAU * self.phase);

            self.damp_low.params = BiquadParams::highpass_1p(
                Simd::splat(self.params.damp_low.smoothed.next() / samplerate),
                Simd::splat(1.),
            );
            self.damp_high.params = BiquadParams::lowpass_1p(
                Simd::splat(self.params.damp_high.smoothed.next() / samplerate),
                Simd::splat(1.),
            );

            self.tick_phase(samplerate, mod_speed);

            channels.from_simd(self.next_sample(
                samplerate,
                size,
                feedback,
                delay,
                mod_depth,
                pitch_amt,
                channels.to_simd::<2>(),
            ));
        }
        ProcessStatus::Normal
    }
}

#[cfg(feature = "vst3")]
impl Vst3Plugin for Reverb {
    const VST3_CLASS_ID: [u8; 16] = *b"SolarLinerNihPlg";
    const VST3_CATEGORIES: &'static str = "Fx|Delay|Reverb";
}

#[cfg(feature = "vst3")]
nih_export_vst3!(Reverb);

#[cfg(test)]
mod tests {}
