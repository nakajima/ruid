use std::{
    sync::{Arc, LazyLock},
    time::Instant,
};

use pixels::{Pixels, SurfaceTexture};
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
    window: Option<Arc<Window>>,
    pixels: Option<Pixels<'a>>,
    simulation: Simulation,
    last_update_instant: Instant,
    fps: f32,
    frame_times: Vec<f32>,
}

impl<'a> Default for Renderer<'a> {
    fn default() -> Self {
        Self {
            window: None,
            pixels: None,
            simulation: Simulation::default(),
            last_update_instant: Instant::now(),
            fps: 0.0,
            frame_times: Vec::with_capacity(100),
        }
    }
}

impl<'a> ApplicationHandler for Renderer<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        let window_size = window.inner_size();
        let surface_texture =
            SurfaceTexture::new(window_size.width, window_size.height, Arc::clone(&window));

        // // Initialize pixels rendering
        let mut pixels =
            Pixels::new(window_size.width, window_size.height, surface_texture).unwrap();
        pixels.enable_vsync(false);

        self.pixels = Some(pixels);
        self.window = Some(window);
        self.simulation.width = window_size.width as f32;
        self.simulation.height = window_size.height as f32;

        for _ in 0..INITIAL_DOT_COUNT {
            self.simulation.add_dot(
                rand::random::<Vector2D>()
                    * Vector2D::from_array([window_size.width as f32, window_size.height as f32]),
                Vector2D::from_array([0., 0.]),
            );
        }

        self.last_update_instant = Instant::now();
        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::Resized(size) => {
                self.pixels
                    .as_mut()
                    .unwrap()
                    .resize_surface(size.width, size.height)
                    .unwrap();
                self.pixels
                    .as_mut()
                    .unwrap()
                    .resize_buffer(size.width, size.height)
                    .unwrap();
                self.simulation.width = size.width as f32;
                self.simulation.height = size.height as f32;
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // Notify that you're about to draw.
                self.window.as_ref().unwrap().pre_present_notify();

                let now = Instant::now();
                let time_delta = now.duration_since(self.last_update_instant).as_secs_f32();
                self.simulation.update(time_delta);
                self.update_fps(time_delta);

                let pixels = self.pixels.as_mut().unwrap();
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
                self.window.as_ref().unwrap().request_redraw();
                self.last_update_instant = now;
            }
            _ => (),
        }
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
