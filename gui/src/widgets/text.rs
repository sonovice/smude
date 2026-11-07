use eframe::egui;
use eframe::egui::{Color32, Response, Sense, Ui, Widget, vec2};

use crate::theme::{Type, color};

/// One line of text in a typographic role, in a box as tall as the role's
/// CSS line height with the text centered in it, so rows of text line up
/// as they did in the web UI. Cut short with "…" when
/// [`truncate`](Self::truncate)d.
pub struct Text {
    text: String,
    role: Type,
    color: Color32,
    truncate: bool,
}

impl Text {
    pub fn new(text: impl Into<String>, role: Type) -> Self {
        Self {
            text: text.into(),
            role,
            color: color::TEXT,
            truncate: false,
        }
    }

    pub fn color(mut self, color: Color32) -> Self {
        self.color = color;
        self
    }

    /// Fit the text into the available width.
    pub fn truncate(mut self) -> Self {
        self.truncate = true;
        self
    }
}

impl Widget for Text {
    fn ui(self, ui: &mut Ui) -> Response {
        let max_width = if self.truncate {
            ui.available_width()
        } else {
            f32::INFINITY
        };
        let galley = self
            .role
            .layout(ui, self.text.as_str(), self.color, max_width);
        let size = vec2(galley.size().x, self.role.line_height.max(galley.size().y));
        let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
        response.widget_info(|| {
            egui::WidgetInfo::labeled(egui::WidgetType::Label, ui.is_enabled(), &self.text)
        });
        if ui.is_rect_visible(rect) {
            let pos = rect.left_center() - vec2(0.0, galley.size().y / 2.0);
            ui.painter().galley(pos, galley, self.color);
        }
        response
    }
}
