use std::f32::consts::TAU;
use std::simd::{LaneCount, Simd, SupportedLaneCount};

use crate::simdmath::*;

#[derive(Debug, Copy, Clone)]
pub struct BiquadParams<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    a: [Simd<f32, LANES>; 2],
    b: [Simd<f32, LANES>; 3],
}

impl<const LANES: usize> Default for BiquadParams<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn default() -> Self {
        Self {
            a: [Simd::splat(0.); 2],
            b: [Simd::splat(1.), Simd::splat(0.), Simd::splat(0.)],
        }
    }
}

impl<const LANES: usize> BiquadParams<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub fn bandpass(fc: Simd<f32, LANES>, q: Simd<f32, LANES>) -> Self {
        let w0 = Simd::splat(TAU) * fc;
        let cw0 = simd_f32cos(w0);
        let a = simd_f32sin(w0) / (Simd::splat(2.) * q);

        let b0 = a;
        let b1 = Simd::splat(0.);
        let b2 = -a;
        let a0 = Simd::splat(1.) + a;
        let a1 = Simd::splat(-2.) * cw0;
        let a2 = Simd::splat(1.) - a;

        Self {
            a: [a1 / a0, a2 / a0],
            b: [b0, b1, b2],
        }
    }

    pub fn allpass(fc: Simd<f32, LANES>, q: Simd<f32, LANES>) -> Self {
        let w0 = Simd::splat(TAU) * fc;
        let cw0 = simd_f32cos(w0);
        let a = simd_f32sin(w0) / (Simd::splat(2.) * q);

        let a0 = Simd::splat(1.) + a;
        let b0 = (Simd::splat(1.) - a) / a0;
        let b1 = (Simd::splat(-2.) * cw0) / a0;
        let b2 = Simd::splat(1.);
        let a1 = (Simd::splat(-2.) * cw0) / a0;
        let a2 = b0;
        Self {
            a: [a1, a2],
            b: [b0, b1, b2],
        }
    }

    pub fn lowpass_1p(fc: Simd<f32, LANES>, q: Simd<f32, LANES>) -> Self {
        let k = simd_f32tan(fc / Simd::splat(2.));
        let a = Simd::splat(1.) + k;

        let a1 = -(Simd::splat(1.) - fc) / a;
        let b0 = k / a;
        let b1 = k / a;

        Self {
            a: [a1, Simd::splat(0.)],
            b: [b0, b1, Simd::splat(0.)],
        }
    }

    pub fn highpass_1p(fc: Simd<f32, LANES>, q: Simd<f32, LANES>) -> Self {
        let k = simd_f32tan(fc / Simd::splat(2.));
        let a = Simd::splat(1.) + k;

        let a1 = -(Simd::splat(1.) - fc) / a;
        let b0 = Simd::splat(1.) / a;
        let b1 = Simd::splat(-1.) / a;

        Self {
            a: [a1, Simd::splat(0.)],
            b: [b0, b1, Simd::splat(0.)],
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Biquad<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub params: BiquadParams<LANES>,
    state: [Simd<f32, LANES>; 2],
}

impl<const LANES: usize> Default for Biquad<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn default() -> Self {
        Self {
            params: BiquadParams::default(),
            state: [Simd::splat(0.); 2],
        }
    }
}

impl<const LANES: usize> Biquad<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub fn new(params: BiquadParams<LANES>) -> Self {
        Self {
            params,
            ..Default::default()
        }
    }

    pub fn next_sample(&mut self, input: Simd<f32, LANES>) -> Simd<f32, LANES> {
        let out = self.state[0] + self.params.b[0] * input;
        self.state[0] = self.state[1] + self.params.b[1] * input - self.params.a[0] * out;
        self.state[1] = self.params.b[2] * input - self.params.a[1] * out;
        out
    }

    pub fn reset(&mut self) {
        self.state = [Simd::splat(0.); 2];
    }
}

#[cfg(test)]
mod tests {
    use std::{iter::repeat, simd::Simd};

    use approx::assert_abs_diff_eq;

    use super::{Biquad, BiquadParams};

    fn test_unit(params: BiquadParams<1>, steady: f32) {
        let mut biquad = Biquad::new(params);
        let steady_actual = repeat(0.)
            .take(10)
            .chain(repeat(1.))
            .map(|v| biquad.next_sample(Simd::from_array([v]))[0])
            .take(5000)
            .last()
            .unwrap();
        assert_abs_diff_eq!(steady, steady_actual);
    }

    #[test]
    fn step_lowpass_1p() {
        test_unit(
            BiquadParams::lowpass_1p(Simd::splat(0.3), Simd::splat(1.)),
            1.,
        );
    }

    #[test]
    fn step_highpass_1p() {
        test_unit(
            BiquadParams::highpass_1p(Simd::splat(0.3), Simd::splat(1.)),
            0.,
        );
    }
}
