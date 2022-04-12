use std::{
    collections::VecDeque,
    simd::{LaneCount, Simd, SupportedLaneCount}
};

#[derive(Debug, Clone)]
pub struct Delay<T> {
    buffer: VecDeque<T>,
}

impl<T> Delay<T> {
    pub fn push_next(&mut self, next: T) {
        self.buffer.pop_back();
        self.buffer.push_front(next);
    }
}

impl<T: Default> Delay<T> {
    pub fn new(max_delay: usize) -> Self {
        Self {
            buffer: VecDeque::from_iter(std::iter::repeat_with(T::default).take(max_delay)),
        }
    }

    pub fn reset(&mut self) {
        for s in self.buffer.iter_mut() {
            *s = T::default();
        }
    }
}

impl<const L: usize> Delay<Simd<f32, L>> where LaneCount<L>: SupportedLaneCount {
    pub fn get(&self, pos: Simd<f32, L>) -> Simd<f32, L> {
        let mut res = Simd::splat(0.);
        for i in 0..L {
            res[i] = self.tap(pos[i])[i];
        }
        res
    }

    pub fn tap(&self, pos: f32) -> Simd<f32, L> {
        let ix = pos.floor();
        let ixf = pos.fract();
        let a = self.buffer[ix as usize % self.buffer.len()];
        let b = self.buffer[(ix as usize + 1)  % self.buffer.len()];
        a + (b - a) * Simd::splat(ixf)
    }
}
