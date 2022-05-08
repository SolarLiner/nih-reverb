use std::simd::*;

#[inline(always)]
pub fn simd_f32func<T: SimdElement, const LANES: usize>(
    f: impl Fn(T) -> T,
    mut x: Simd<T, LANES>,
) -> Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    for elem in x.as_mut_array() {
        *elem = f(*elem);
    }
    x
}

#[inline(always)]
pub fn simd_f32tanh<const LANES: usize>(x: Simd<f32, LANES>) -> Simd<f32, { LANES }>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    simd_f32func(f32::tanh, x)
}

#[inline(always)]
pub fn simd_f32cos<const LANES: usize>(x: Simd<f32, LANES>) -> Simd<f32, { LANES }>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    simd_f32func(f32::cos, x)
}

#[inline(always)]
pub fn simd_f32sin<const LANES: usize>(x: Simd<f32, LANES>) -> Simd<f32, { LANES }>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    simd_f32func(f32::sin, x)
}

#[inline(always)]
pub fn simd_f32tan<const LANES: usize>(x: Simd<f32, LANES>) -> Simd<f32, { LANES }>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    simd_f32func(f32::tan, x)
}
