#![feature(portable_simd)]

use std::{
    simd::{LaneCount, Simd, SupportedLaneCount},
    sync::{atomic::AtomicBool, Arc},
};

use nih_plug::prelude::*;
use nih_reverb::biquad::{Biquad, BiquadParams};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum BiquadMode {
    #[id = "lp"]
    #[name = "LP (24)"]
    Lowpass,
    #[id = "hp"]
    #[name = "HP (24)"]
    Highpass,
    #[id = "bp"]
    #[name = "BP (24)"]
    Bandpass,
}

#[derive(Params)]
struct PluginParams {
    #[id = "mode"]
    mode: EnumParam<BiquadMode>,
    #[id = "frq"]
    frequency: FloatParam,
    #[id = "q"]
    q: FloatParam,
}

#[derive(Debug, Default, Clone)]
struct Tick {
    repr: Arc<AtomicBool>,
}

impl Tick {
    fn tick(&self) {
        self.repr.store(true, std::sync::atomic::Ordering::Release)
    }

    fn has_tick(&self) -> bool {
        self.repr
            .fetch_and(false, std::sync::atomic::Ordering::Acquire)
    }
}

impl PluginParams {
    fn new(filter_update_tick: Tick) -> Self {
        Self {
            mode: EnumParam::new("Mode", BiquadMode::Lowpass)
                .non_automatable()
                .with_callback(Arc::new(move |_| {
                    filter_update_tick.tick();
                })),
            frequency: FloatParam::new(
                "Frequency",
                3000.0,
                FloatRange::Skewed {
                    min: 1.,
                    max: 40000.0,
                    factor: FloatRange::skew_factor(-2.5),
                },
            ),
            q: FloatParam::new(
                "Q",
                std::f32::consts::FRAC_1_SQRT_2,
                FloatRange::Skewed {
                    min: 0.001,
                    max: 10.0,
                    factor: FloatRange::skew_factor(-2.5),
                },
            ),
        }
    }

    fn next_biquad_params<const N: usize>(&self, sr: f32) -> BiquadParams<N>
    where
        LaneCount<N>: SupportedLaneCount,
    {
        let fc = Simd::splat(self.frequency.smoothed.next() / sr / 2.0);
        let q = Simd::splat(self.q.smoothed.next());

        match self.mode.value() {
            BiquadMode::Lowpass => BiquadParams::lowpass_1p(fc, q),
            BiquadMode::Bandpass => BiquadParams::bandpass(fc, q),
            BiquadMode::Highpass => BiquadParams::highpass_1p(fc, q),
        }
    }
}

#[derive(Clone)]
struct BiquadPlugin<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    filter_update_tick: Tick,
    params: Arc<PluginParams>,
    biquad: Biquad<N>,
}

impl<const N: usize> Default for BiquadPlugin<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn default() -> Self {
        let filter_update_tick = Tick::default();

        let params = PluginParams::new(filter_update_tick.clone());
        Self {
            filter_update_tick,
            params: Arc::new(params),
            biquad: Biquad::default(),
        }
    }
}

impl<const N: usize> BiquadPlugin<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn next_sample(&mut self, sr: f32, input: Simd<f32, N>) -> Simd<f32, N> {
        if self.filter_update_tick.has_tick() {
            self.biquad.reset();
        }

        self.biquad.params = self.params.next_biquad_params(sr);

        self.biquad.next_sample(input)
    }
}

impl Plugin for BiquadPlugin<2> {
    const NAME: &'static str = "Biquad";

    const VENDOR: &'static str = "SolarLiner";

    const URL: &'static str = "N/A";

    const EMAIL: &'static str = "N/A";

    const VERSION: &'static str = "0.0.1";

    const DEFAULT_NUM_INPUTS: u32 = 2;

    const DEFAULT_NUM_OUTPUTS: u32 = 2;

    const DEFAULT_AUX_INPUTS: Option<AuxiliaryIOConfig> = None;

    const DEFAULT_AUX_OUTPUTS: Option<AuxiliaryIOConfig> = None;

    const PORT_NAMES: PortNames = PortNames {
        main_input: None,
        main_output: None,
        aux_inputs: None,
        aux_outputs: None,
    };

    const MIDI_INPUT: MidiConfig = MidiConfig::None;

    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = false;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _bus_config: &BusConfig,
        _buffer_config: &BufferConfig,
        context: &mut impl InitContext,
    ) -> bool {
        self.biquad.reset();
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
            channels.from_simd(self.next_sample(samplerate, channels.to_simd()));
        }
        ProcessStatus::Normal
    }
}

impl Vst3Plugin for BiquadPlugin<2> {
    const VST3_CLASS_ID: [u8; 16] = *b"SolarLinerNihBiq";

    const VST3_CATEGORIES: &'static str = "Fx|Filter";
}

nih_export_vst3!(BiquadPlugin::<2>);
