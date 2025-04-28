#![feature(random)]
#![feature(portable_simd)]
#![feature(array_repeat)]
#![feature(future_join)]

pub mod renderer;
pub mod simulation;
pub mod vectors;

#[cfg(target_arch = "wasm32")]
pub mod web;
