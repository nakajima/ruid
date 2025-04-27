use std::simd::Simd;

pub trait VectorDirecitons {
    const DOWN: Self;
}

pub type Vector2D = Simd<f32, 2>;
pub type Vector3D = Simd<f32, 3>;

impl VectorDirecitons for Vector2D {
    const DOWN: Self = Self::from_array([0., 1.0]);
}

impl VectorDirecitons for Vector3D {
    const DOWN: Self = Self::from_array([0., 1.0, 0.]);
}
