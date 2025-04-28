use std::{
    array,
    sync::{Arc, LazyLock},
};

use log::error;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use pixels::{Pixels, SurfaceTexture, wgpu};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

use crate::{
    simulation::{INITIAL_DOT_COUNT, Simulation},
    vectors::Vector2D,
};

// Parse it into the font type.
static FONT_BYTES: &[u8] = include_bytes!("../JetBrainsMonoNL-Light.ttf");
static FONT: LazyLock<fontdue::Font> = LazyLock::new(|| {
    fontdue::Font::from_bytes(FONT_BYTES, fontdue::FontSettings::default()).unwrap()
});

pub struct Renderer<'a> {
    pixels: Option<Pixels<'a>>,
    simulation: Simulation,
    last_update_instant: Instant,
    fps: f32,
    frame_times: Vec<f32>,
    rng: rand::rngs::SmallRng,
}

impl<'a> Default for Renderer<'a> {
    fn default() -> Self {
        Self {
            pixels: None,
            simulation: Simulation::default(),
            last_update_instant: Instant::now(),
            fps: 0.0,
            frame_times: Vec::with_capacity(100),
            rng: SmallRng::from_seed(array::repeat(1)),
        }
    }
}

impl<'a> Renderer<'a> {
    pub fn init<W: wgpu::WindowHandle + 'a>(
        &mut self,
        width: u32,
        height: u32,
        window: Arc<W>,
        mut pixels: Pixels<'a>,
    ) {
        pixels.enable_vsync(false);

        self.pixels = Some(pixels);
        self.simulation.width = width as f32;
        self.simulation.height = height as f32;

        for _ in 0..INITIAL_DOT_COUNT {
            self.simulation.add_dot(
                self.rng.random::<Vector2D>() * Vector2D::from_array([width as f32, height as f32]),
                Vector2D::from_array([0., 0.]),
            );
        }

        self.last_update_instant = Instant::now();
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let pixels = match self.pixels.as_mut() {
            Some(pixels) => pixels,
            None => return,
        };

        match pixels.resize_surface(width, height) {
            Ok(()) => {}
            Err(e) => error!("Error resizing surface: {:?}", e),
        }

        match pixels.resize_buffer(width, height) {
            Ok(()) => {}
            Err(e) => error!("Error resizing buffer: {:?}", e),
        }

        self.simulation.width = width as f32;
        self.simulation.height = height as f32;
    }

    pub fn update_and_draw(&mut self) {
        let now = Instant::now();
        let time_delta = now.duration_since(self.last_update_instant).as_secs_f32();
        self.simulation.update(time_delta);
        self.update_fps(time_delta);

        let pixels = match self.pixels.as_mut() {
            Some(pixels) => pixels,
            None => return,
        };

        let frame = pixels.frame_mut();
        self.simulation.draw(frame);

        // Draw FPS counter
        let fps_text = format!("FPS: {:.1}", self.fps);

        Self::draw_text(
            frame,
            &fps_text,
            self.simulation.width as usize,
            self.simulation.height as usize,
            10,
            20,
        );

        // Render the frame to the window
        pixels.render().unwrap();
        self.last_update_instant = now;
    }
}

impl<'a> Renderer<'a> {
    fn update_fps(&mut self, frame_time: f32) {
        self.frame_times.push(frame_time);

        if self.frame_times.len() > 100 {
            self.frame_times.remove(0);
        }

        let avg_frame_time = self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32;
        self.fps = if avg_frame_time > 0.0 {
            1.0 / avg_frame_time
        } else {
            0.0
        };
    }

    fn draw_text(frame: &mut [u8], text: &str, width: usize, height: usize, x: usize, y: usize) {
        let color = [255, 255, 255, 255]; // White text

        let mut cursor_x = x;

        for c in text.chars() {
            let (metrics, bitmap) = FONT.rasterize(c, 17.0);

            // Iterate through the bitmap and draw pixels
            for row in 0..metrics.height {
                for col in 0..metrics.width {
                    let bitmap_idx = row * metrics.width + col;
                    let alpha = bitmap[bitmap_idx] as f32 / 255.0;

                    if alpha > 0.0 {
                        let frame_x = cursor_x + col;
                        let frame_y = y + row;

                        // Make sure we're within frame boundaries
                        if frame_x < width as usize && frame_y < height as usize {
                            let frame_idx = (frame_y * width + frame_x) * 4;

                            if frame_idx + 3 < frame.len() {
                                // Blend text color with existing color using alpha
                                let curr_r = frame[frame_idx];
                                let curr_g = frame[frame_idx + 1];
                                let curr_b = frame[frame_idx + 2];

                                frame[frame_idx] =
                                    (curr_r as f32 * (1.0 - alpha) + color[0] as f32 * alpha) as u8;
                                frame[frame_idx + 1] =
                                    (curr_g as f32 * (1.0 - alpha) + color[1] as f32 * alpha) as u8;
                                frame[frame_idx + 2] =
                                    (curr_b as f32 * (1.0 - alpha) + color[2] as f32 * alpha) as u8;
                                frame[frame_idx + 3] = 255;
                            }
                        }
                    }
                }
            }

            // Move cursor to the right for the next character
            cursor_x += metrics.width + 1; // Add spacing between characters
        }
    }
}
