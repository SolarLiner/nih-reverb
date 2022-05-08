#![feature(portable_simd)]
#![feature(array_from_fn)]
#![feature(const_for)]

use std::f32::consts::TAU;
use std::{
    simd::{f32x2, Simd},
    sync::Arc,
};

use nih_plug::prelude::*;

use early::Early;

use crate::delay::Delay;

mod biquad;
mod delay;
mod diffusion;
mod early;
mod hadamard;
mod householder;
mod simdmath;

#[derive(Params)]
struct DelayParams {
    #[id = "ersize"]
    size: FloatParam,
    #[id = "fbck"]
    feedback: FloatParam,
    #[id = "delay"]
    delay: FloatParam,
}

impl Default for DelayParams {
    fn default() -> Self {
        Self {
            size: FloatParam::new("Size", 0.5, FloatRange::Linear { min: 0., max: 1. })
                .with_unit("%")
                .with_string_to_value(formatters::s2v_f32_percentage())
                .with_value_to_string(formatters::v2s_f32_percentage(2))
                .with_smoother(SmoothingStyle::Linear(20.)),
            feedback: FloatParam::new("Feedback", 0.5, FloatRange::Linear { min: 0., max: 1.25 })
                .with_unit("%")
                .with_string_to_value(formatters::s2v_f32_percentage())
                .with_value_to_string(formatters::v2s_f32_percentage(2)),
            delay: FloatParam::new("Delay", 1., FloatRange::Linear { min: 1e-3, max: 2. })
                .with_unit("s")
                .with_smoother(SmoothingStyle::Linear(200.)),
        }
    }
}

struct Reverb {
    params: Arc<DelayParams>,
    diffusion: Early<4>,
    delay: Delay<f32x2>,
    phase: f32,
}

impl Default for Reverb {
    fn default() -> Self {
        Self {
            params: Arc::default(),
            diffusion: Early::new(44100.),
            // bandpass: Biquad::new(BiquadParams::bandpass(
            //     f32x8::from_array(std::array::from_fn(|_| {
            //         0.1 + rand::thread_rng().gen_range(-0.06..0.1)
            //     })),
            //     f32x8::splat(1.),
            // )),
            delay: Delay::new(44100 * 2),
            phase: 0.,
        }
    }
}

impl Plugin for Reverb {
    const NAME: &'static str = "Delay";
    const VENDOR: &'static str = "SolarLiner";
    const URL: &'static str = "N/A";
    const EMAIL: &'static str = "N/A";
    const VERSION: &'static str = "0.0.2";

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn accepts_bus_config(&self, config: &BusConfig) -> bool {
        config.num_input_channels == config.num_output_channels && config.num_input_channels == 2
    }

    fn initialize(
        &mut self,
        _bus_config: &BusConfig,
        _buffer_config: &BufferConfig,
        context: &mut impl ProcessContext,
    ) -> bool {
        let samplerate = context.transport().sample_rate;
        self.diffusion = Early::new(samplerate);
        self.delay = Delay::new((samplerate * 2.) as _);
        true
    }

    // fn initialize(
    //     &mut self,
    //     _bus_config: &BusConfig,
    //     _buffer_config: &BufferConfig,
    //     context: &mut impl ProcessContext,
    // ) -> bool {
    //     let new_sr = context.transport().sample_rate;
    //     if self.samplerate != new_sr {
    //         self.samplerate = new_sr;
    //         self.taps = [0; 4].map(|_| Tap::new(self.samplerate));
    //         eprintln!("Initialize: samplerate: {}", self.samplerate);
    //     }
    //     true
    // }

    // fn reset(&mut self) {
    //     eprintln!("Reset");
    //     self.early = Early::new(self.samplerate);
    //     for tap in self.taps.iter_mut() {
    //         *tap = Tap::new(self.samplerate);
    //     }
    // }

    fn process(&mut self, buffer: &mut Buffer, context: &mut impl ProcessContext) -> ProcessStatus {
        let samplerate = context.transport().sample_rate;
        for mut channels in buffer.iter_samples() {
            self.phase += 0.3 / samplerate;
            if self.phase > 1. {
                self.phase -= 1.;
            }
            let feedback = self.params.feedback.smoothed.next();
            let size = self.params.size.smoothed.next();
            let delay = self.params.delay.smoothed.next() + 5e-3 * f32::sin(TAU * self.phase);

            let sample = channels.to_simd::<2>();
            let delayed = sample + self.delay.tap(delay * samplerate) * Simd::splat(feedback);
            let diffuse_input =
                Simd::gather_or_default(delayed.as_array(), Simd::from_array([0, 1, 1, 0]));
            let diffused = self.diffusion.next_sample(size, diffuse_input);
            let diffused = f32x2::gather_or_default(diffused.as_array(), Simd::from_array([1, 2]));
            let diffused = simdmath::simd_f32tanh(diffused);
            self.delay.push_next(diffused);

            channels.from_simd(diffused);
        }
        ProcessStatus::Normal
    }
}

impl Vst3Plugin for Reverb {
    const VST3_CLASS_ID: [u8; 16] = *b"SolarLinerNihPlg";
    const VST3_CATEGORIES: &'static str = "Fx|Delay|Reverb";
}

nih_export_vst3!(Reverb);
