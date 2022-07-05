// Copyright (c) 2022 solarliner
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use std::simd::{LaneCount, Simd, SupportedLaneCount};

use crate::diffusion::Diffusion;

pub struct Early<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    ap: [Diffusion<LANES>; LANES],
}

impl<const LANES: usize> Early<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub fn new(samplerate: f32) -> Self {
        Self {
            ap: std::array::from_fn(|i| {
                Diffusion::new(400e-3 * samplerate * (1. + (i as f32 / LANES as f32).powi(2)))
            }),
        }
    }
}

impl<const LANES: usize> Early<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub fn next_sample(
        &mut self,
        size: f32,
        mod_depth: f32,
        input: Simd<f32, LANES>,
    ) -> Simd<f32, LANES> {
        self.ap
            .iter_mut()
            .fold(input, |s, ap| ap.next_sample(size, mod_depth, s))
    }

    pub fn next_block(&mut self, size: &[f32], mod_depth: &[f32], buffer: &mut [Simd<f32, LANES>]) {
        for diffuse in self.ap.iter_mut() {
            diffuse.next_block(size, mod_depth, buffer);
        }
    }
}
