use eframe::egui;
use eframe::egui::{
    Color32, CornerRadius, Rect, Response, Sense, StrokeKind, Ui, Widget, pos2, vec2,
};

use super::is_hot;
use crate::theme::{border, color, radius, role};

/// One choice of a [`Segmented`] control.
pub struct Segment<T> {
    value: T,
    label: String,
    caption: Option<String>,
    enabled: bool,
}

impl<T> Segment<T> {
    pub fn new(value: T, label: impl Into<String>) -> Self {
        Self {
            value,
            label: label.into(),
            caption: None,
            enabled: true,
        }
    }

    /// A second, smaller line, like "48 channels".
    pub fn caption(mut self, caption: impl Into<String>) -> Self {
        self.caption = Some(caption.into());
        self
    }

    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

/// A row of joined buttons choosing one of a few values, like the stream
/// (USB, WSG, WDANTE) or what to name track files by. The segments share
/// the width equally. The response is changed when the choice is.
pub struct Segmented<'a, T> {
    current: &'a mut T,
    segments: Vec<Segment<T>>,
}

impl<'a, T: PartialEq> Segmented<'a, T> {
    pub fn new(current: &'a mut T, segments: impl IntoIterator<Item = Segment<T>>) -> Self {
        Self {
            current,
            segments: segments.into_iter().collect(),
        }
    }
}

impl<T: PartialEq> Widget for Segmented<'_, T> {
    fn ui(self, ui: &mut Ui) -> Response {
        let captioned = self
            .segments
            .iter()
            .any(|segment| segment.caption.is_some());
        let caption_role = role::TINY.tracking(0.025);
        let content = role::SEGMENT.line_height
            + if captioned {
                2.0 + caption_role.line_height
            } else {
                0.0
            };
        let size = vec2(ui.available_width(), 1.0 + 10.0 + content + 10.0 + 1.0);
        let (rect, mut response) = ui.allocate_exact_size(size, Sense::hover());

        let count = self.segments.len();
        if count == 0 {
            return response;
        }
        // Neighbours share a border, as `-ml-px` did.
        let width = (rect.width() + (count - 1) as f32) / count as f32;
        let rects: Vec<Rect> = (0..count)
            .map(|i| {
                Rect::from_min_size(
                    pos2(rect.left() + i as f32 * (width - 1.0), rect.top()),
                    vec2(width, rect.height()),
                )
            })
            .collect();
        let responses: Vec<Response> = self
            .segments
            .iter()
            .zip(&rects)
            .enumerate()
            .map(|(i, (segment, rect))| {
                let sense = if segment.enabled {
                    Sense::click()
                } else {
                    Sense::hover()
                };
                ui.interact(*rect, response.id.with(i), sense)
            })
            .collect();
        let clicked = responses.iter().position(Response::clicked);

        let mut painted = Vec::with_capacity(count);
        for (i, segment) in self.segments.into_iter().enumerate() {
            let active = match clicked {
                Some(clicked) => clicked == i,
                None => *self.current == segment.value,
            };
            let corners = CornerRadius {
                nw: if i == 0 { radius::LG } else { 0 },
                sw: if i == 0 { radius::LG } else { 0 },
                ne: if i == count - 1 { radius::LG } else { 0 },
                se: if i == count - 1 { radius::LG } else { 0 },
            };
            let hot = segment.enabled && is_hot(&responses[i]);
            responses[i].widget_info(|| {
                egui::WidgetInfo::selected(
                    egui::WidgetType::Button,
                    ui.is_enabled() && segment.enabled,
                    active,
                    &segment.label,
                )
            });
            painted.push((
                rects[i],
                corners,
                segment.label,
                segment.caption,
                segment.enabled,
                active,
                hot,
            ));
            if clicked == Some(i) && *self.current != segment.value {
                *self.current = segment.value;
                response.mark_changed();
            }
        }

        if ui.is_rect_visible(rect) {
            // The chosen segment last, so its gold border lies on top.
            painted.sort_by_key(|&(.., active, hot)| (active, hot));
            for (segment_rect, corners, label, caption_text, enabled, active, hot) in painted {
                let (fill, stroke, text, caption) = match (enabled, active, hot) {
                    (false, _, _) => (
                        Color32::TRANSPARENT,
                        color::BORDER,
                        color::TEXT_DIM,
                        color::TEXT_DIM,
                    ),
                    (true, true, _) => (
                        color::ACCENT_GLOW_STRONG,
                        color::ACCENT,
                        color::ACCENT,
                        color::ACCENT.gamma_multiply(0.5),
                    ),
                    (true, false, true) => (
                        Color32::TRANSPARENT,
                        color::BORDER_HI,
                        color::TEXT,
                        color::TEXT_DIM,
                    ),
                    (true, false, false) => (
                        Color32::TRANSPARENT,
                        color::BORDER,
                        color::TEXT_MID,
                        color::TEXT_DIM,
                    ),
                };
                let mut painter = ui.painter().clone();
                if !enabled {
                    painter.multiply_opacity(0.3);
                }
                painter.rect(
                    segment_rect,
                    corners,
                    fill,
                    border(stroke),
                    StrokeKind::Inside,
                );

                let label = role::SEGMENT.layout(ui, label, text, segment_rect.width() - 8.0);
                let top = segment_rect.center().y - content / 2.0;
                let label_pos = pos2(
                    segment_rect.center().x - label.size().x / 2.0,
                    top + (role::SEGMENT.line_height - label.size().y) / 2.0,
                );
                painter.galley(label_pos, label, text);
                if let Some(caption_text) = caption_text {
                    let galley =
                        caption_role.layout(ui, caption_text, caption, segment_rect.width() - 8.0);
                    let caption_top = top + role::SEGMENT.line_height + 2.0;
                    let pos = pos2(
                        segment_rect.center().x - galley.size().x / 2.0,
                        caption_top + (caption_role.line_height - galley.size().y) / 2.0,
                    );
                    painter.galley(pos, galley, caption);
                }
            }
        }
        response
    }
}
