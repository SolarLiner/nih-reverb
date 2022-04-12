use std::collections::VecDeque;
use std::simd::{LaneCount, Simd, StdFloat, SupportedLaneCount};

use crate::delay::Delay;

#[derive(Debug, Clone)]
pub struct Allpass<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    delay: Delay<Simd<f32, LANES>>,
}

impl<const LANES: usize> Allpass<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub fn reset(&mut self) {
        self.delay.reset();
    }
}

impl<const LANES: usize> Allpass<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub fn new(max_samples: usize) -> Self {
        Self {
            delay: Delay::new(max_samples),
        }
    }

    pub fn next_sample(
        &mut self,
        gain: Simd<f32, LANES>,
        pos: f32,
        input: Simd<f32, LANES>,
    ) -> Simd<f32, LANES> {
        let fb = self.delay.tap(pos);
        self.delay.push_next(input + fb * gain);
        fb + input * gain
    }
}

#[derive(Debug, Clone)]
pub struct AllpassLine<const N: usize, const L: usize>
where
    LaneCount<L>: SupportedLaneCount,
{
    ap: [Allpass<L>; N],
    delays: [f32; N],
}

impl<const N: usize, const L: usize> AllpassLine<N, L>
where
    LaneCount<L>: SupportedLaneCount,
{
    pub fn new(delays: [f32; N]) -> Self {
        let ap = delays.map(|v| Allpass::new(v.ceil() as _));
        Self { ap, delays }
    }

    pub fn reset(&mut self) {
        for ap in self.ap.iter_mut() {
            ap.reset();
        }
    }

    pub fn next_sample(
        &mut self,
        size: f32,
        offset: f32,
        gain: Simd<f32, L>,
        input: Simd<f32, L>,
    ) -> Simd<f32, L> {
        let k = size.clamp(0., 1.);
        self.ap
            .iter_mut()
            .zip(self.delays)
            .fold(input, |s, (ap, del)| {
                ap.next_sample(gain, del * k + offset * (1. - k), s)
            })
    }
}
