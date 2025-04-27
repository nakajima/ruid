use std::{sync::Arc, time::Instant};

use pixels::{Pixels, SurfaceTexture};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

use crate::{Simulation, Vector2D};

pub struct App<'a> {
    window: Option<Arc<Window>>,
    pixels: Option<Pixels<'a>>,
    simulation: Simulation,
    last_update_instant: Instant,
}

impl<'a> Default for App<'a> {
    fn default() -> Self {
        Self {
            window: None,
            pixels: None,
            simulation: Simulation::default(),
            last_update_instant: Instant::now(),
        }
    }
}

impl<'a> ApplicationHandler for App<'a> {
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
        let pixels = Pixels::new(window_size.width, window_size.height, surface_texture).unwrap();

        self.pixels = Some(pixels);
        self.window = Some(window);
        self.simulation.width = window_size.width as f32;
        self.simulation.height = window_size.height as f32;

        self.simulation.add_dot(
            Vector2D::from_array([100., 100.]),
            Vector2D::from_array([0., 0.]),
        );

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
                self.last_update_instant = now;

                let pixels = self.pixels.as_mut().unwrap();
                let frame = pixels.frame_mut();
                self.simulation.draw(frame);

                // Render the frame to the window
                pixels.render().unwrap();
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}
