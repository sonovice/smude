mod canvas;
mod centered;
mod dialogs;
mod navigation;
mod panels;
mod shading;

use crate::{
    export::{self, ExportItem},
    import, preview, theme, worker,
};
use eframe::egui::{self, Color32, RichText, Vec2};
use smude_geometry::application::{
    DetectionProgress, DetectionStage, OutputKind, ProcessingResult, Settings,
};
use std::{
    collections::HashSet,
    path::PathBuf,
    sync::{Arc, mpsc},
    time::Instant,
};

#[derive(Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct Preferences {
    export_directory: Option<PathBuf>,
}

enum State {
    Queued,
    Running(Instant),
    Ready {
        result: ProcessingResult,
        seconds: f64,
    },
    Failed(String),
}
struct Job {
    id: u64,
    path: PathBuf,
    state: State,
    overlays: Vec<Overlay>,
}
#[derive(Clone)]
struct Overlay {
    data: Arc<DetectionProgress>,
    arrived: Instant,
}
struct Texture {
    handle: egui::TextureHandle,
    dimensions: [u32; 2],
}
#[derive(Default)]
struct Canvas {
    scale: Option<f32>,
    last_scale: f32,
    pan: Vec2,
    viewport: Option<egui::Rect>,
    anchor: Option<navigation::Navigation>,
}
struct ExportDone {
    result: Result<Vec<PathBuf>, String>,
    directory: PathBuf,
    sources: Vec<PathBuf>,
}

pub struct SmudeApp {
    prefs: Preferences,
    jobs: Vec<Job>,
    selected: Option<u64>,
    selected_page: usize,
    next_id: u64,
    paused: bool,
    importing: bool,
    importing_rx: mpsc::Receiver<import::Discovery>,
    importing_tx: mpsc::Sender<import::Discovery>,
    processor: worker::Processor,
    preview_loader: preview::Loader,
    preview_generation: u64,
    preview_key: Option<(u64, usize, OutputKind, bool)>,
    left: Option<Texture>,
    right: Option<Texture>,
    left_error: Option<String>,
    right_error: Option<String>,
    left_canvas: Canvas,
    right_canvas: Canvas,
    view: View,
    pending_navigation: Option<(u64, navigation::Side, navigation::Navigation)>,
    notice: Option<String>,
    export_rx: Option<mpsc::Receiver<ExportDone>>,
    last_export: Option<PathBuf>,
    exported_sources: HashSet<PathBuf>,
    cache: Arc<tempfile::TempDir>,
    show_detections: bool,
    show_about: bool,
    license_view: Option<bool>,
    confirm_quit: bool,
    allow_quit: bool,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    Compare,
    Result,
    Original,
}

impl SmudeApp {
    pub fn new(cc: &eframe::CreationContext<'_>, inputs: Vec<PathBuf>) -> Result<Self, String> {
        let prefs = cc
            .storage
            .and_then(|s| eframe::get_value(s, "preferences"))
            .unwrap_or_default();
        Self::from_context(&cc.egui_ctx, prefs, inputs)
    }
    fn from_context(
        ctx: &egui::Context,
        prefs: Preferences,
        inputs: Vec<PathBuf>,
    ) -> Result<Self, String> {
        theme::install(ctx);
        let cache = Arc::new(
            tempfile::Builder::new()
                .prefix("smude-session-")
                .tempdir()
                .map_err(|e| e.to_string())?,
        );
        let (importing_tx, importing_rx) = mpsc::channel();
        let mut app = Self {
            prefs,
            jobs: vec![],
            selected: None,
            selected_page: 0,
            next_id: 1,
            paused: false,
            importing: false,
            importing_rx,
            importing_tx,
            processor: worker::Processor::new(cache.clone(), ctx.clone()),
            preview_loader: preview::Loader::new(ctx.clone()),
            preview_generation: 0,
            preview_key: None,
            left: None,
            right: None,
            left_error: None,
            right_error: None,
            left_canvas: Canvas::default(),
            right_canvas: Canvas::default(),
            view: View::Compare,
            pending_navigation: None,
            notice: None,
            export_rx: None,
            last_export: None,
            exported_sources: HashSet::new(),
            cache,
            show_detections: true,
            show_about: false,
            license_view: None,
            confirm_quit: false,
            allow_quit: false,
        };
        if !inputs.is_empty() {
            app.import(inputs, ctx);
        }
        Ok(app)
    }
    fn import(&mut self, paths: Vec<PathBuf>, ctx: &egui::Context) {
        if paths.is_empty() {
            return;
        }
        if self.importing {
            self.notice = Some("Bitte warten, bis der Ordner eingelesen ist.".into());
            return;
        }
        self.importing = true;
        let tx = self.importing_tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let discovery = import::discover(&paths);
            let _ = tx.send(discovery);
            ctx.request_repaint();
        });
    }
    fn choose_files(&mut self, ctx: &egui::Context) {
        if let Some(paths) = rfd::FileDialog::new()
            .set_title("Notenseiten hinzufügen")
            .add_filter("Bilder", &["jpg", "jpeg", "png"])
            .pick_files()
        {
            self.import(paths, ctx);
        }
    }
    fn choose_folder(&mut self, ctx: &egui::Context) {
        if let Some(path) = rfd::FileDialog::new()
            .set_title("Ordner mit Notenseiten auswählen")
            .pick_folder()
        {
            self.import(vec![path], ctx);
        }
    }
    fn has_unsaved_work(&self) -> bool {
        self.export_rx.is_some()
            || self.jobs.iter().any(|job| match &job.state {
                State::Queued | State::Running(_) => true,
                State::Failed(_) => false,
                State::Ready { result, .. } => result.pages.iter().any(|page| {
                    ![
                        OutputKind::Clean,
                        OutputKind::Gray,
                        OutputKind::Binary,
                        OutputKind::Color,
                    ]
                    .iter()
                    .any(|kind| self.exported_sources.contains(&page.image(*kind)))
                }),
            })
    }
    fn busy(&self) -> bool {
        self.jobs
            .iter()
            .any(|j| matches!(j.state, State::Running(_)))
    }
    fn selected_job(&self) -> Option<&Job> {
        self.jobs.iter().find(|j| Some(j.id) == self.selected)
    }
    fn result(&self) -> Option<&ProcessingResult> {
        match &self.selected_job()?.state {
            State::Ready { result, .. } => Some(result),
            _ => None,
        }
    }
    fn select(&mut self, id: u64) {
        self.selected = Some(id);
        self.selected_page = 0;
        self.preview_key = None;
    }
    fn poll(&mut self, ctx: &egui::Context) {
        while let Ok(discovery) = self.importing_rx.try_recv() {
            self.importing = false;
            let mut added = 0;
            for path in discovery.files {
                if self.jobs.iter().any(|j| j.path == path) {
                    continue;
                }
                let id = self.next_id;
                self.next_id += 1;
                self.jobs.push(Job {
                    id,
                    path,
                    state: State::Queued,
                    overlays: vec![],
                });
                added += 1;
                if self.selected.is_none() {
                    self.select(id);
                }
            }
            if !discovery.warnings.is_empty() {
                self.notice = Some(discovery.warnings.join("\n"));
            } else if added == 0 {
                self.notice = Some("Keine neuen JPEG- oder PNG-Dateien gefunden.".into());
            }
        }
        while let Ok(progress) = self.processor.progress.try_recv() {
            if let Some(job) = self.jobs.iter_mut().find(|j| j.id == progress.id) {
                let event = progress.event;
                if event.stage == DetectionStage::ResetMusic {
                    job.overlays.retain(|old| {
                        old.data.page_id != event.page_id || old.data.stage == DetectionStage::Page
                    });
                } else {
                    job.overlays.retain(|old| {
                        old.data.page_id != event.page_id || old.data.stage != event.stage
                    });
                    job.overlays.push(Overlay {
                        data: event,
                        arrived: Instant::now(),
                    });
                }
            }
        }
        while let Ok(done) = self.processor.rx.try_recv() {
            if let Some(job) = self.jobs.iter_mut().find(|j| j.id == done.id) {
                job.state = match done.result {
                    Ok(result) => State::Ready {
                        result,
                        seconds: done.seconds,
                    },
                    Err(e) => State::Failed(e),
                };
            }
        }
        while let Ok(preview) = self.preview_loader.rx.try_recv() {
            if preview.generation != self.preview_generation {
                continue;
            }
            let (slot, error) = if preview.left {
                (&mut self.left, &mut self.left_error)
            } else {
                (&mut self.right, &mut self.right_error)
            };
            match preview.result {
                Ok((image, dimensions)) => {
                    *slot = Some(Texture {
                        handle: ctx.load_texture(
                            if preview.left { "original" } else { "result" },
                            image,
                            egui::TextureOptions::LINEAR,
                        ),
                        dimensions,
                    })
                }
                Err(e) => *error = Some(e),
            }
        }
        if let Some(rx) = &self.export_rx {
            match rx.try_recv() {
                Ok(done) => {
                    match done.result {
                        Ok(paths) => {
                            self.exported_sources.extend(done.sources);
                            self.last_export = Some(done.directory);
                            self.notice = Some(format!(
                                "{} Seite{} exportiert. Vorhandene Dateien wurden beibehalten.",
                                paths.len(),
                                if paths.len() == 1 { "" } else { "n" }
                            ));
                        }
                        Err(e) => self.notice = Some(format!("Export fehlgeschlagen: {e}")),
                    }
                    self.export_rx = None;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.export_rx = None;
                    self.notice = Some("Export wurde unerwartet beendet.".into());
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        if !self.paused
            && !self.busy()
            && let Some(job) = self
                .jobs
                .iter_mut()
                .find(|j| matches!(j.state, State::Queued))
        {
            let work = worker::Work {
                id: job.id,
                input: job.path.clone(),
                settings: Settings::default(),
            };
            job.state = match self.processor.submit(work) {
                Ok(()) => State::Running(Instant::now()),
                Err(e) => State::Failed(e),
            };
        }
        self.request_previews();
    }
    fn request_previews(&mut self) {
        let Some(job) = self.selected_job() else {
            return;
        };
        let ready = matches!(job.state, State::Ready { .. });
        let key = (job.id, self.selected_page, OutputKind::Clean, ready);
        if self.preview_key == Some(key) {
            return;
        }
        let preserve_left = self
            .preview_key
            .is_some_and(|previous| previous.0 == job.id)
            && self.left.is_some();
        let left = (!preserve_left).then(|| job.path.clone());
        let right = self
            .result()
            .and_then(|r| r.pages.get(self.selected_page))
            .map(|p| p.image(OutputKind::Clean));
        self.preview_generation += 1;
        self.preview_loader.request(preview::Request {
            generation: self.preview_generation,
            left,
            right,
        });
        self.preview_key = Some(key);
        if !preserve_left {
            self.left = None;
            self.left_canvas = Canvas::default();
        }
        self.right = None;
        self.left_error = None;
        self.right_error = None;
        self.right_canvas = Canvas::default();
    }
    fn export_items(&self, all: bool) -> Vec<ExportItem> {
        self.jobs
            .iter()
            .filter(|j| all || Some(j.id) == self.selected)
            .flat_map(|job| {
                let stem = job
                    .path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned();
                let State::Ready { result, .. } = &job.state else {
                    return vec![];
                };
                result
                    .pages
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| all || *i == self.selected_page)
                    .map(|(_, page)| ExportItem {
                        source: page.image(OutputKind::Clean),
                        stem: stem.clone(),
                        page: page.id.clone(),
                    })
                    .collect()
            })
            .collect()
    }
    fn start_export(&mut self, all: bool, ctx: &egui::Context) {
        let items = self.export_items(all);
        if items.is_empty() {
            return;
        }
        let mut picker = rfd::FileDialog::new().set_title("Exportordner auswählen");
        if let Some(directory) = &self.prefs.export_directory {
            picker = picker.set_directory(directory);
        }
        let Some(directory) = picker.pick_folder() else {
            return;
        };
        self.prefs.export_directory = Some(directory.clone());
        let (tx, rx) = mpsc::channel();
        self.export_rx = Some(rx);
        let ctx = ctx.clone();
        let cache = self.cache.clone();
        std::thread::spawn(move || {
            let _keep_cache_alive = cache;
            let sources = items.iter().map(|item| item.source.clone()).collect();
            let result = export::export(&items, &directory).map_err(|e| e.to_string());
            let _ = tx.send(ExportDone {
                result,
                directory,
                sources,
            });
            ctx.request_repaint();
        });
    }
}

impl eframe::App for SmudeApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll(ctx);
        let dropped = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_owned())
                .collect::<Vec<_>>()
        });
        if !dropped.is_empty() {
            self.import(dropped, ctx);
        }
        if ctx.input(|i| i.viewport().close_requested())
            && !self.allow_quit
            && self.has_unsaved_work()
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.confirm_quit = true;
        }
        if self.busy() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.interface(ui);
    }
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, "preferences", &self.prefs);
    }
}

impl SmudeApp {
    fn interface(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::Panel::top("header")
            .frame(egui::Frame::NONE)
            .show(ui, |ui| self.header(ui));
        egui::Panel::bottom("footer")
            .exact_size(28.)
            .show_separator_line(false)
            .frame(egui::Frame::NONE)
            .show(ui, |ui| self.footer(ui));
        egui::Panel::left("files-compact")
            .default_size(180.)
            .size_range(156. ..=260.)
            .resizable(true)
            .frame(egui::Frame::new().fill(theme::PANEL).inner_margin(8))
            .show(ui, |ui| self.sidebar(ui));
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| {
                theme::backdrop::paint_page(ui.painter(), ui.max_rect());
                egui::Frame::new().inner_margin(20).show(ui, |ui| {
                    if let Some(message) = self.notice.clone() {
                        crate::widgets::notice_frame(crate::widgets::Tone::Neutral).show(
                            ui,
                            |ui| {
                                crate::widgets::centered_row(ui, |ui| {
                                    ui.add(egui::Label::new(message).wrap());
                                    if ui.add(crate::widgets::IconButton::close()).clicked() {
                                        self.notice = None;
                                    }
                                });
                            },
                        );
                        ui.add_space(8.);
                    }
                    self.workspace(ui);
                });
            });
        if ctx.input(|i| !i.raw.hovered_files.is_empty()) {
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("drop-overlay"),
            ));
            let rect = ctx.content_rect().shrink(14.);
            painter.rect_filled(rect, 14., Color32::from_black_alpha(225));
            painter.rect_stroke(
                rect,
                14.,
                egui::Stroke::new(2., theme::GOLD),
                egui::StrokeKind::Inside,
            );
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Dateien und Ordner hier ablegen",
                egui::FontId::proportional(26.),
                theme::GOLD,
            );
        }
        self.dialogs(&ctx);
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    use smude_geometry::application::RenderedPage;
    #[test]
    fn real_app_keeps_comparison_and_bottom_controls_inside_their_regions() {
        let ctx = egui::Context::default();
        let mut app = SmudeApp::from_context(&ctx, Preferences::default(), vec![]).unwrap();
        app.jobs.push(Job {
            id: 1,
            path: "score.png".into(),
            state: State::Running(Instant::now()),
            overlays: vec![],
        });
        app.selected = Some(1);
        let texture = ctx.load_texture(
            "source",
            egui::ColorImage::filled([2, 2], Color32::WHITE),
            egui::TextureOptions::LINEAR,
        );
        app.left = Some(Texture {
            handle: texture.clone(),
            dimensions: [1200, 1600],
        });
        let mut snapshots = vec![];
        for ready in [false, false, true, true] {
            if ready {
                app.jobs[0].state = State::Ready {
                    result: ProcessingResult {
                        pages: vec![
                            RenderedPage {
                                quality: Default::default(),
                                id: "1".into(),
                                mapping: None,
                                directory: "unused".into(),
                            },
                            RenderedPage {
                                quality: Default::default(),
                                id: "2".into(),
                                mapping: None,
                                directory: "unused".into(),
                            },
                        ],
                        warnings: vec![],
                    },
                    seconds: 1.,
                };
                app.right = Some(Texture {
                    handle: texture.clone(),
                    dimensions: [1100, 1500],
                });
            }
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1000., 650.),
                    )),
                    ..Default::default()
                },
                |ui| app.interface(ui),
            );
            output.textures_delta.clear();
            let canvases: Vec<_> = output
                .shapes
                .iter()
                .filter_map(|shape| match &shape.shape {
                    egui::Shape::Rect(r)
                        if r.stroke.color == theme::BORDER
                            && r.rect.width() > 200.
                            && r.rect.height() > 100. =>
                    {
                        Some(r.rect)
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(canvases.len(), 2);
            assert_eq!(canvases[0].size(), canvases[1].size());
            assert_eq!(canvases[0].top(), canvases[1].top());
            for shape in &output.shapes {
                if let egui::Shape::Text(text) = &shape.shape {
                    assert!(
                        !text.galley.job.text.contains("use of widget ID"),
                        "duplicate widget ID in the actual app"
                    );
                }
                match &shape.shape {
                    egui::Shape::LineSegment { points, stroke }
                        if stroke.color == theme::BORDER
                            && points[0].x == points[1].x
                            && (points[0].y - points[1].y).abs() > 200. =>
                    {
                        assert!(
                            points[0].y.max(points[1].y) <= 623.,
                            "sidebar separator leaks into status bar: {points:?}"
                        );
                    }
                    egui::Shape::Rect(r)
                        if r.rect.width() < 200. && r.rect.top() > 560. && r.rect.left() < 200. =>
                    {
                        assert!(
                            r.rect.bottom() <= 622.,
                            "sidebar action leaks into status bar: {:?}",
                            r.rect
                        );
                    }
                    _ => {}
                }
            }
            snapshots.push(canvases);
        }
        assert_eq!(snapshots[1], snapshots[3]);
    }
}
