#![feature(portable_simd)]
#![feature(array_from_fn)]
#![feature(const_for)]

use std::f32::consts::TAU;
use std::simd::{f32x2, f32x8};
use std::sync::Arc;

use nih_plug::prelude::*;
use rand::Rng;

use early::Early;

use crate::biquad::{Biquad, BiquadParams};
use crate::delay::Delay;

mod allpass;
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
}

impl Default for DelayParams {
    fn default() -> Self {
        Self {
            size: FloatParam::new("Size", 0.5, FloatRange::Linear { min: 0., max: 1. })
                .with_unit("%")
                .with_string_to_value(formatters::s2v_f32_percentage())
                .with_value_to_string(formatters::v2s_f32_percentage(2))
                .with_smoother(SmoothingStyle::Linear(20.)),
            feedback: FloatParam::new("Feedback", 0.5, FloatRange::Linear { min: 0., max: 1. })
                .with_unit("%")
                .with_string_to_value(formatters::s2v_f32_percentage())
                .with_value_to_string(formatters::v2s_f32_percentage(2)),
        }
    }
}

struct Reverb {
    params: Arc<DelayParams>,
    early: Early<8>,
    //bandpass: Biquad<8>,
    delay: Delay<f32x8>,
    phase: f32,
}

impl Default for Reverb {
    fn default() -> Self {
        Self {
            params: Arc::default(),
            early: Early::new(44100.),
            // bandpass: Biquad::new(BiquadParams::bandpass(
            //     f32x8::from_array(std::array::from_fn(|_| {
            //         0.1 + rand::thread_rng().gen_range(-0.06..0.1)
            //     })),
            //     f32x8::splat(1.),
            // )),
            delay: Delay::new(44100),
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
        bus_config: &BusConfig,
        buffer_config: &BufferConfig,
        context: &mut impl ProcessContext,
    ) -> bool {
        let samplerate = context.transport().sample_rate;
        self.early = Early::new(samplerate);
        self.delay = Delay::new(samplerate as _);
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
            let sample = channels.to_simd::<2>();
            let sample = f32x8::from_array([
                sample[0], sample[1], sample[0], sample[1], sample[0], sample[1], sample[0],
                sample[1],
            ]);
            let early = householder::transform(self.early.next_sample(size, sample));
            let early = simdmath::simd_f32tanh(early);
            let sr = samplerate;
            let delays = f32x8::from_array(
                [6., 9., 13., 22., 48., 121., 251., 557.].map(|v| size * v * 1e-3 * sr),
            );
            let res = self.delay.get(delays);
            let res = simdmath::simd_f32tanh(res);
            // let res = self.bandpass.next_sample(res);
            self.delay
                .push_next(householder::transform(res * f32x8::splat(feedback)) + early);

            channels.from_simd(f32x2::from_array([res[1], res[3]]));
        }
        ProcessStatus::Normal
    }
}

impl Vst3Plugin for Reverb {
    const VST3_CLASS_ID: [u8; 16] = *b"SolarLinerNihPlg";
    const VST3_CATEGORIES: &'static str = "Fx|Delay|Reverb";
}

nih_export_vst3!(Reverb);
