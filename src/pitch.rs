use std::simd::{LaneCount, Simd, SupportedLaneCount};

use crate::delay::Delay;

pub struct PitchShifter<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    buffer: Delay<Simd<f32, N>>,
    pos: f32,
}

impl<const N: usize> PitchShifter<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub fn new(max_delay: usize) -> Self {
        Self {
            buffer: Delay::new(max_delay),
            pos: 0.,
        }
    }

    pub fn next_sample(
        &mut self,
        samplerate: f32,
        pitch: f32,
        input: Simd<f32, N>,
    ) -> Simd<f32, N> {
        let out = self.buffer.tap(self.pos);
        self.pos += pitch;
        if self.pos > self.buffer.len() as _ {
            self.pos -= self.buffer.len() as f32;
        }
        self.buffer.push_next(input);
        out
    }
}
