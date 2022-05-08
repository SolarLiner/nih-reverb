use std::f32::consts::TAU;
use std::simd::{LaneCount, Simd, SupportedLaneCount};

use rand::prelude::*;

use crate::delay::Delay;
use crate::householder;

pub struct Diffusion<const L: usize>
where
    LaneCount<L>: SupportedLaneCount,
{
    delay: Delay<Simd<f32, L>>,
    polarity: Simd<f32, L>,
    offsets: [f32; L],
    phases: [f32; L],
    samplerate: f32,
}

impl<const L: usize> Diffusion<L>
where
    LaneCount<L>: SupportedLaneCount,
{
    pub fn new(samplerate: f32) -> Self {
        Self {
            delay: Delay::new(samplerate as usize),
            polarity: {
                let zeros = Simd::splat(-1.);
                let ones = Simd::splat(1.);
                let (_, res) = zeros.interleave(ones);
                dbg!(res)
            },
            offsets: {
                let mut rng = thread_rng();
                std::array::from_fn(|_| rng.gen_range(-1e-2..1e-2))
            },
            phases: std::array::from_fn(|_| rand::random()),
            samplerate,
        }
    }

    pub fn next_sample(&mut self, size: f32, input: Simd<f32, L>) -> Simd<f32, L> {
        let delays = std::array::from_fn(|i| {
            let t = i as f32 / L as f32;
            self.samplerate
                * (300e-3 * t * size + self.offsets[i] + 1e-3 * f32::sin(TAU * self.phases[i]))
        });
        for p in &mut self.phases {
            *p += 0.3 / self.samplerate;
            if *p > 1. {
                *p -= 1.;
            }
        }
        let taps = self.delay.get(Simd::from_array(delays));
        let taps = shuffle(taps);
        self.delay.push_next(input);

        householder::transform(self.polarity * taps)
        // taps
    }
}

fn shuffle<const N: usize>(inp: Simd<f32, N>) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let in_arr = inp.as_array();
    let out_arr = std::array::from_fn(|n| {
        let i = (n * 187 + 288) % N;
        let k = if (n % 2) == 0 { 1. } else { -1. };
        in_arr[i] * k
    });
    Simd::from_array(out_arr)
}
