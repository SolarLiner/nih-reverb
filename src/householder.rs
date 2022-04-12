use std::simd::{LaneCount, Simd, SupportedLaneCount};

pub fn transform<const L: usize>(mut v: Simd<f32, L>) -> Simd<f32, L>
where
    LaneCount<L>: SupportedLaneCount,
{
    let s = v.to_array().into_iter().sum::<f32>() * (-2.0 / L as f32);
    for i in 0..L {
        v[i] += s;
    }
    v
}
