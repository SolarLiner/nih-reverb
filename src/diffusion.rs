use std::simd::{LaneCount, Simd, SupportedLaneCount};

use rand::prelude::*;

use crate::delay::Delay;
use crate::{hadamard, householder};

pub struct Diffusion<const L: usize>
where
    LaneCount<L>: SupportedLaneCount,
{
    delay: Delay<Simd<f32, L>>,
    shuffle: Simd<f32, L>,
    offsets: [f32; L],
    samplerate: f32,
}

impl<const L: usize> Diffusion<L>
where
    LaneCount<L>: SupportedLaneCount,
{
    pub fn new(samplerate: f32) -> Self {
        Self {
            delay: Delay::new(samplerate as usize),
            shuffle: {
                let zeros = Simd::splat(0.);
                let ones = Simd::splat(1.);
                let (_, res) = zeros.interleave(ones);
                res
            },
            offsets: {
                let mut rng = thread_rng();
                std::array::from_fn(|_| rng.gen_range(-1e-2..1e-2))
            },
            samplerate,
        }
    }

    pub fn next_sample(&mut self, size: f32, input: Simd<f32, L>) -> Simd<f32, L> {
        let delays = std::array::from_fn(|i| {
            self.samplerate * 330. / 1e3 * size * (i as f32 / L as f32) + self.offsets[i]
        });
        let taps = self.delay.get(Simd::from_array(delays));
        self.delay.push_next(input);

        householder::transform(self.shuffle * taps)
    }
}
