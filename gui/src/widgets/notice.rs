//! Messages about the app's state: inline notices, banners across the
//! window and the status bar.

use eframe::egui;
use eframe::egui::{
    Align, Color32, Frame, Layout, Margin, Rect, Response, Sense, Ui, UiBuilder, Widget, vec2,
};

use crate::theme::{Type, border, color, radius, role};

/// How good or bad a message is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    Neutral,
    Success,
    Warning,
    Error,
}

/// A bordered box for a message among the content, like a failed
/// connection under the console list. Its text takes [`notice_text`].
pub fn notice_frame(tone: Tone) -> Frame {
    let (fill, stroke) = match tone {
        Tone::Neutral => (color::SURFACE, color::BORDER),
        Tone::Success => (color::GREEN_DIM, color::GREEN.gamma_multiply(0.3)),
        Tone::Warning => (color::ACCENT_GLOW, color::AMBER.gamma_multiply(0.3)),
        Tone::Error => (color::RED_DIM, color::RED.gamma_multiply(0.4)),
    };
    Frame::new()
        .fill(fill)
        .stroke(border(stroke))
        .corner_radius(radius::CARD)
        .inner_margin(Margin::symmetric(12, 10))
}

/// The text color in a [`notice_frame`].
pub fn notice_text(tone: Tone) -> Color32 {
    match tone {
        Tone::Neutral => color::TEXT_MID,
        Tone::Success => color::GREEN,
        Tone::Warning => color::AMBER,
        Tone::Error => color::RED,
    }
}

/// A strip across the window with a short, loud message, like DISK SPACE LOW.
pub struct Banner {
    tone: Tone,
    message: String,
}

impl Banner {
    /// The banner's text role.
    pub const ROLE: Type = role::BUTTON.leading(16.0);

    pub fn new(tone: Tone, message: impl Into<String>) -> Self {
        Self {
            tone,
            message: message.into(),
        }
    }

    /// The banner's background and text colors.
    pub fn colors(tone: Tone) -> (Color32, Color32) {
        match tone {
            Tone::Neutral => (color::SURFACE_ALT, color::TEXT),
            Tone::Success => (color::GREEN.gamma_multiply(0.9), Color32::BLACK),
            Tone::Warning => (color::AMBER.gamma_multiply(0.9), Color32::BLACK),
            Tone::Error => (color::RED.gamma_multiply(0.9), Color32::WHITE),
        }
    }
}

impl Widget for Banner {
    fn ui(self, ui: &mut Ui) -> Response {
        let (fill, text) = Self::colors(self.tone);
        let (rect, response) = ui.allocate_exact_size(
            vec2(ui.available_width(), 6.0 + Self::ROLE.line_height + 6.0),
            Sense::hover(),
        );
        if ui.is_rect_visible(rect) {
            ui.painter().rect_filled(rect, 0, fill);
            let galley = Self::ROLE.layout(ui, self.message, text, rect.width() - 24.0);
            ui.painter()
                .galley(rect.center() - galley.size() / 2.0, galley, text);
        }
        response
    }
}

/// What the status bar shows besides its contents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusTone {
    Idle,
    /// The network connection to the WING is lost.
    Disconnected,
    Recording,
}

/// The line at the bottom of the main screen: contents at the left, in the
/// middle and at the right, in 12 point text. Plain labels in it take the
/// bar's text color.
pub struct StatusBar {
    tone: StatusTone,
}

impl StatusBar {
    pub fn new(tone: StatusTone) -> Self {
        Self { tone }
    }

    /// The bar's background and text colors.
    pub fn colors(tone: StatusTone) -> (Color32, Color32) {
        match tone {
            StatusTone::Idle => (color::SURFACE, color::TEXT_DIM),
            StatusTone::Disconnected => (color::RED, color::TEXT),
            StatusTone::Recording => (color::RED_BRIGHT, color::TEXT),
        }
    }

    pub fn show(
        self,
        ui: &mut Ui,
        left: impl FnOnce(&mut Ui),
        center: impl FnOnce(&mut Ui),
        right: impl FnOnce(&mut Ui),
    ) -> Response {
        let (fill, text) = Self::colors(self.tone);
        let edge = if self.tone == StatusTone::Idle {
            color::BORDER
        } else {
            fill
        };
        let (rect, response) =
            ui.allocate_exact_size(vec2(ui.available_width(), 28.0), Sense::hover());
        ui.painter().rect_filled(rect, 0, fill);
        ui.painter()
            .hline(rect.x_range(), rect.top() + 0.5, border(edge));

        let inner = rect.shrink2(vec2(12.0, 0.0));
        let part = |ui: &mut Ui, rect: Rect, layout: Layout, add: Box<dyn FnOnce(&mut Ui) + '_>| {
            let mut child = ui.new_child(UiBuilder::new().max_rect(rect).layout(layout));
            child.set_clip_rect(rect.intersect(ui.clip_rect()));
            child.visuals_mut().override_text_color = Some(text);
            child.style_mut().override_font_id = Some(role::SMALL.font_id());
            child.spacing_mut().item_spacing.x = 12.0;
            add(&mut child);
        };
        part(
            ui,
            inner,
            Layout::left_to_right(Align::Center),
            Box::new(left),
        );
        part(
            ui,
            inner,
            Layout::left_to_right(Align::Center).with_main_align(Align::Center),
            Box::new(center),
        );
        part(
            ui,
            inner,
            Layout::right_to_left(Align::Center),
            Box::new(right),
        );
        response
    }
}
