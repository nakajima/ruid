use std::sync::Arc;

use pixels::{Pixels, SurfaceTexture};
use ruid::renderer::Renderer;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

fn main() {
    let event_loop = EventLoop::new().unwrap();

    // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // dispatched any events. This is ideal for games and similar applications.
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut NativeRenderer::default()).unwrap();
}

#[derive(Default)]
pub struct NativeRenderer<'a> {
    window: Option<Arc<Window>>,
    renderer: Renderer<'a>,
}

impl<'a> ApplicationHandler for NativeRenderer<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        let window_size = window.inner_size();
        let surface_texture =
            SurfaceTexture::new(window_size.width, window_size.height, Arc::clone(&window));

        let pixels = Pixels::new(window_size.width, window_size.height, surface_texture).unwrap();

        self.renderer.init(
            window_size.width,
            window_size.height,
            window.clone(),
            pixels,
        );

        self.window = Some(window);
        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::Resized(size) => {
                self.renderer.resize(size.width, size.height);
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // Notify that you're about to draw.
                self.window.as_ref().unwrap().pre_present_notify();
                self.renderer.update_and_draw();
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}
