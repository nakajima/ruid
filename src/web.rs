use std::sync::Arc;

use crate::renderer::Renderer;
use ::js_sys::*;
use log::{error, warn};
use pixels::wgpu::{DeviceDescriptor, Features, Limits, PowerPreference, RequestAdapterOptions};
use pixels::{PixelsBuilder, SurfaceTexture, wgpu::Backends};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use winit::platform::web::WindowExtWebSys;
use winit::{application::ApplicationHandler, event::WindowEvent, window::Window};
use winit::{
    dpi::LogicalSize,
    event_loop::{ControlFlow, EventLoop},
};

#[wasm_bindgen(start)]
pub fn start() {
    warn!("spawning run");
    spawn_local(run());
}

#[wasm_bindgen]
pub async fn run() {
    use log::info;

    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Trace).expect("error initializing logger");

    warn!("hi main");

    info!("setting up event loop");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);

    let window = Arc::new(
        event_loop
            .create_window(Window::default_attributes())
            .unwrap(),
    );

    // Attach winit canvas to body element
    web_sys::window()
        .and_then(|win| win.document())
        .and_then(|doc| doc.body())
        .and_then(|body| {
            body.append_child(&web_sys::Element::from(window.canvas().unwrap()))
                .ok()
        })
        .expect("couldn't append canvas to document body");

    let surface_texture = SurfaceTexture::new(1024, 768 as u32, Arc::clone(&window));

    let texture_format = pixels::wgpu::TextureFormat::Rgba8Unorm;

    let mut renderer = WebRenderer::default();

    let mut adapter_options = RequestAdapterOptions::default();
    // adapter_options.force_fallback_adapter = false;
    adapter_options.power_preference = PowerPreference::HighPerformance;

    let mut device_descriptor = DeviceDescriptor::default();
    device_descriptor.required_features = Features::empty();

    device_descriptor.required_limits = Limits::downlevel_webgl2_defaults();
    // device_descriptor.required_limits = Limits::default();
    device_descriptor.required_limits.max_texture_dimension_2d = 2048;

    let pixels = match PixelsBuilder::new(1024, 768, surface_texture)
        .request_adapter_options(adapter_options)
        .device_descriptor(device_descriptor)
        .texture_format(texture_format)
        .surface_texture_format(texture_format)
        // .wgpu_backend(Backends::BROWSER_WEBGPU)
        .wgpu_backend(Backends::GL)
        .build_async()
        .await
    {
        Ok(pixels) => pixels,
        Err(e) => {
            error!("Error building pixels: {:?}", e);
            panic!("Error building pixels: {:?}", e)
        }
    };

    renderer.window = Some(window.clone());

    renderer
        .renderer
        .init(1024, 768 as u32, window.clone(), pixels);

    event_loop.run_app(&mut renderer).unwrap();
}

#[derive(Default)]
pub struct WebRenderer<'a> {
    window: Option<Arc<Window>>,
    renderer: Renderer<'a>,
}

impl<'a> ApplicationHandler for WebRenderer<'a> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::Resized(size) => {
                // self.renderer.resize(size.width, size.height);
                // self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // Notify that you're about to draw.
                // self.window.as_ref().unwrap().pre_present_notify();
                self.renderer.update_and_draw();
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}
