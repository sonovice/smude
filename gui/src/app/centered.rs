//! Centered paragraphs and measured rows, ported from Wing Companion.
#![allow(dead_code)]
use std::sync::Arc;

use eframe::egui;
use eframe::egui::text::LayoutJob;
use eframe::egui::{
    Align, Color32, Context, Galley, Layout, Sense, TextFormat, Ui, UiBuilder, vec2,
};

use crate::theme::{Icon, Type};

/// Centered text in runs of different looks; `\n` breaks a line, as `<br />`
/// did.
pub(crate) struct Paragraph {
    job: LayoutJob,
}

impl Paragraph {
    pub fn new() -> Self {
        let mut job = LayoutJob {
            halign: Align::Center,
            ..LayoutJob::default()
        };
        job.wrap.max_width = f32::INFINITY;
        Self { job }
    }

    /// Add `text` in `role` and `color`.
    pub fn run(mut self, text: &str, role: Type, color: Color32) -> Self {
        let format = TextFormat {
            line_height: Some(role.line_height),
            ..role.format(color)
        };
        self.job.append(text, 0.0, format);
        self
    }

    /// Lay the text out wrapped at `width`.
    fn layout(&self, ctx: &Context, width: f32) -> Arc<Galley> {
        let mut job = self.job.clone();
        job.wrap.max_width = width;
        ctx.fonts_mut(|fonts| fonts.layout_job(job))
    }

    /// The width of the longest line, unwrapped.
    pub fn width(&self, ctx: &Context) -> f32 {
        self.layout(ctx, f32::INFINITY).size().x
    }

    pub fn show(&self, ui: &mut Ui) {
        let galley = self.layout(ui.ctx(), ui.available_width());
        let (rect, _) =
            ui.allocate_exact_size(vec2(ui.available_width(), galley.size().y), Sense::hover());
        // Centered rows lie around the galley's origin.
        ui.painter()
            .galley(rect.center_top(), galley, Color32::PLACEHOLDER);
    }
}

/// The modal's width for its longest paragraph, as CSS shrinks a box with
/// a `max-w-*` to fit its contents.
pub(crate) fn fit_width(
    ctx: &Context,
    paragraphs: &[&Paragraph],
    padding: f32,
    max_width: f32,
) -> f32 {
    let widest = paragraphs.iter().map(|p| p.width(ctx)).fold(0.0, f32::max);
    (widest.ceil() + 2.0 * padding).min(max_width)
}

/// A large icon, centered.
pub(crate) fn icon(ui: &mut Ui, icon: Icon, size: f32, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), size), Sense::hover());
    icon.paint(ui.painter(), rect.center(), size, color);
}

/// A row of widgets centered as a whole. Its width is known only once laid
/// out, so it comes from the last frame, with one more pass when it changes.
pub(crate) fn centered_row<R>(ui: &mut Ui, add_contents: impl FnOnce(&mut Ui) -> R) -> R {
    let id = ui.next_auto_id().with("centered_row");
    let known = ui.data(|data| data.get_temp::<f32>(id));
    let width = known.unwrap_or(0.0);
    let left = ui.max_rect().center().x - width / 2.0;
    let rect = egui::Rect::from_min_size(egui::pos2(left, ui.cursor().top()), vec2(width, 0.0));
    let builder = UiBuilder::new()
        .max_rect(rect)
        .layout(Layout::left_to_right(Align::Center));
    let mut child = ui.new_child(builder);
    let inner = crate::widgets::centered_row(&mut child, add_contents).inner;
    let used = child.min_rect();
    if known != Some(used.width()) {
        ui.data_mut(|data| data.insert_temp(id, used.width()));
        ui.ctx().request_discard("a centered row's width changed");
    }
    ui.allocate_rect(
        egui::Rect::from_min_size(
            egui::pos2(ui.max_rect().left(), used.top()),
            vec2(ui.available_width(), used.height()),
        ),
        Sense::hover(),
    );
    inner
}
