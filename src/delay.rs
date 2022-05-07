use std::{
    collections::VecDeque,
    simd::{LaneCount, Simd, SupportedLaneCount},
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
}

impl<const L: usize> Delay<Simd<f32, L>>
where
    LaneCount<L>: SupportedLaneCount,
{
    pub fn get(&mut self, pos: Simd<f32, L>) -> Simd<f32, L> {
        let mut res = Simd::splat(0.);
        for i in 0..L {
            res[i] = self.tap(pos[i])[i];
        }
        res
    }

    // Cubic interpolation
    pub fn tap(&mut self, pos: f32) -> Simd<f32, L> {
        let ix = pos.floor() as _;
        let f = pos.fract();

        let a0 = self.sample(ix - 2);
        let a1 = self.sample(ix - 1);
        let b0 = self.sample(ix);
        let b1 = self.sample(ix + 1);

        cubic(f, [a0, a1, b0, b1])
    }

    // Nearest-neighbor interpolation
    #[cfg(never)]
    pub fn tap(&mut self, pos: f32) -> Simd<f32, L> {
        let ix = pos.round() as _;
        let s = self.sample(ix);
        return s;
    }

    fn sample(&self, i: isize) -> Simd<f32, L> {
        let index = (i + self.buffer.len() as isize) % self.buffer.len() as isize;
        self.buffer[index as usize]
    }
}

#[inline(always)]
fn cubic<const L: usize>(t: f32, p: [Simd<f32, L>; 4]) -> Simd<f32, L>
where
    LaneCount<L>: SupportedLaneCount,
{
    let t = Simd::splat(t);
    let half = Simd::splat(0.5);
    let two = Simd::splat(2.);
    let three = Simd::splat(3.);
    let four = Simd::splat(4.);
    let five = Simd::splat(5.);

    return p[1]
        + half
            * t
            * (p[2] - p[0]
                + t * (two * p[0] - five * p[1] + four * p[2] - p[3]
                    + t * (three * (p[1] - p[2]) + p[3] - p[0])));
}
