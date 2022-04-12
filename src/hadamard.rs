use std::simd::{LaneCount, Simd, SupportedLaneCount};

#[inline]
pub fn fwht<const L: usize>(mut a: Simd<f32, L>) -> Simd<f32, L>
where
    LaneCount<L>: SupportedLaneCount,
{
    let mut h = 1;
    while h < L {
        for i in (0..L).step_by(h * 2) {
            for j in i..i + h {
                let x = a[j];
                let y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
        }
        h *= 2;
    }

    a
}
