use std::simd::Simd;

pub trait VectorDirecitons {
    const DOWN: Self;
}

pub type Vector2D = Simd<f32, 2>;

impl VectorDirecitons for Vector2D {
    const DOWN: Self = Self::from_array([0., 1.0]);
}
