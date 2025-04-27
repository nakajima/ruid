#![feature(portable_simd)]

use crate::vectors::VectorDirecitons;
use app::App;
use vectors::Vector2D;
use winit::event_loop::{ControlFlow, EventLoop};

mod app;
mod vectors;

pub struct Simulation {
    dot_radius: f32,
    dot_positions: Vec<Vector2D>,
    dot_velocities: Vec<Vector2D>,
    width: f32,
    height: f32,
    gravity: f32,
}

impl Default for Simulation {
    fn default() -> Self {
        Self {
            dot_positions: vec![],
            dot_velocities: vec![],
            dot_radius: 60.,
            width: 0.,
            height: 0.,
            gravity: 1200.,
        }
    }
}

impl Simulation {
    fn add_dot(&mut self, position: Vector2D, velocity: Vector2D) {
        self.dot_positions.push(position);
        self.dot_velocities.push(velocity);
    }

    fn update(&mut self, time_delta: f32) {
        for i in 0..self.dot_positions.len() {
            let pos = self.dot_positions[i];
            if pos[1] > self.height - self.dot_radius {
                self.dot_velocities[i] =
                    Vector2D::from_array([0., -self.dot_velocities[i][1]]) * Vector2D::splat(0.7);
                self.dot_positions[i] =
                    Vector2D::from_array([self.dot_positions[i][0], self.height - self.dot_radius]);
            } else {
                self.dot_velocities[i] +=
                    Vector2D::DOWN * Vector2D::splat(self.gravity) * Vector2D::splat(time_delta);
                self.dot_positions[i] += self.dot_velocities[i] * Vector2D::splat(time_delta);
            }
        }
    }

    fn draw(&self, frame: &mut [u8]) {
        let width = self.width as usize;

        // Clear frame with black background
        for pixel in frame.chunks_exact_mut(4) {
            pixel.copy_from_slice(&[0, 0, 0, 255]);
        }

        // Draw each dot
        for &pos in &self.dot_positions {
            let center_x = pos[0] as usize;
            let center_y = pos[1] as usize;
            let radius = self.dot_radius as usize;

            // Calculate bounding box with bounds checking
            let min_x = center_x.saturating_sub(radius);
            let min_y = center_y.saturating_sub(radius);
            let max_x = (center_x + radius).min(width.saturating_sub(1));
            let max_y = (center_y + radius)
                .min(self.height as usize)
                .saturating_sub(1);

            // Only check pixels in the bounding box
            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    let dx = x as f32 - pos[0];
                    let dy = y as f32 - pos[1];
                    let distance_squared = dx * dx + dy * dy;

                    if distance_squared <= self.dot_radius * self.dot_radius {
                        let idx = (y * width + x) * 4;
                        if idx + 3 < frame.len() {
                            // Set pixel to white
                            frame[idx] = 0; // R
                            frame[idx + 1] = 170; // G
                            frame[idx + 2] = 255; // B
                            frame[idx + 3] = 255; // A
                        }
                    }
                }
            }
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // dispatched any events. This is ideal for games and similar applications.
    event_loop.set_control_flow(ControlFlow::Poll);

    // ControlFlow::Wait pauses the event loop if no events are available to process.
    // This is ideal for non-game applications that only update in response to user
    // input, and uses significantly less power/CPU time than ControlFlow::Poll.
    event_loop.set_control_flow(ControlFlow::Wait);
    event_loop.run_app(&mut App::default()).unwrap();
}
