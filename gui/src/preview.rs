//! One preview decoder, only the selected images; no texture per imported file.
use eframe::egui::{self, ColorImage};
use std::{
    path::PathBuf,
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

pub struct Request {
    pub generation: u64,
    pub left: Option<PathBuf>,
    pub right: Option<PathBuf>,
}
pub struct Preview {
    pub generation: u64,
    pub left: bool,
    pub result: Result<(ColorImage, [u32; 2]), String>,
}
pub struct Loader {
    tx: Sender<Request>,
    pub rx: Receiver<Preview>,
}
impl Loader {
    pub fn new(ctx: egui::Context) -> Self {
        let (tx, requests) = mpsc::channel::<Request>();
        let (images, rx) = mpsc::sync_channel(2);
        thread::Builder::new()
            .name("smude-preview".into())
            .spawn(move || {
                while let Ok(mut request) = requests.recv() {
                    // Rapid navigation replaces obsolete requests, instead of spawning decoders.
                    while let Ok(latest) = requests.try_recv() {
                        request = latest;
                    }
                    for (left, path) in [(true, request.left), (false, request.right)] {
                        let Some(path) = path else {
                            continue;
                        };
                        let result = decode(&path);
                        if images
                            .send(Preview {
                                generation: request.generation,
                                left,
                                result,
                            })
                            .is_err()
                        {
                            return;
                        }
                        ctx.request_repaint();
                    }
                }
            })
            .expect("preview worker");
        Self { tx, rx }
    }
    pub fn request(&self, request: Request) {
        let _ = self.tx.send(request);
    }
}
fn decode(path: &std::path::Path) -> Result<(ColorImage, [u32; 2]), String> {
    let mut reader = image::ImageReader::open(path).map_err(|e| e.to_string())?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(512 * 1024 * 1024);
    reader.limits(limits);
    let image = reader.decode().map_err(|e| e.to_string())?;
    let dimensions = [image.width(), image.height()];
    let thumb = image.thumbnail(4096, 4096).into_rgba8();
    Ok((
        ColorImage::from_rgba_unmultiplied(
            [thumb.width() as usize, thumb.height() as usize],
            thumb.as_raw(),
        ),
        dimensions,
    ))
}
