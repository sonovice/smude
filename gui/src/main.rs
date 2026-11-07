#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod app;
mod export;
mod import;
mod preview;
mod theme;
mod widgets;
mod worker;

fn main() -> eframe::Result {
    let icon = image::load_from_memory(include_bytes!("../assets/icon.png"))
        .expect("bundled icon")
        .into_rgba8();
    let icon = eframe::egui::IconData {
        width: icon.width(),
        height: icon.height(),
        rgba: icon.into_raw(),
    };
    #[cfg(target_os = "macos")]
    let renderer = eframe::Renderer::Wgpu;
    #[cfg(not(target_os = "macos"))]
    let renderer = eframe::Renderer::Glow;
    let options = eframe::NativeOptions {
        renderer,
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Smude — Notenseiten entzerren")
            .with_icon(icon)
            .with_inner_size([1360., 880.])
            .with_min_inner_size([1000., 650.])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    let inputs = std::env::args_os()
        .skip(1)
        .map(std::path::PathBuf::from)
        .collect();
    eframe::run_native(
        "Smude",
        options,
        Box::new(move |cc| {
            app::SmudeApp::new(cc, inputs)
                .map(|app| Box::new(app) as Box<dyn eframe::App>)
                .map_err(Into::into)
        }),
    )
}
