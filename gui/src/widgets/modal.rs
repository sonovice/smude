use eframe::egui;
use eframe::egui::{
    Align, Context, Frame, Id, Layout, Margin, ScrollArea, Shape, StrokeKind, Ui, vec2,
};

use super::{IconButton, Text, centered_row};
use crate::theme::{backdrop, border, color, radius, role};

/// A dialog over the dimmed app, centered: alerts, the license, About.
///
/// Without a title the body gets the generous padding of the alerts; with
/// one, a header with the title (and the close button) sits above a body
/// padded like the license's.
///
/// ```ignore
/// let modal = Modal::new("eula").title("License agreement").closable(true).show(ctx, |ui| { … });
/// if modal.close_requested { … }
/// ```
pub struct Modal {
    id: Id,
    title: Option<String>,
    closable: bool,
    dismissible: bool,
    width: f32,
    padding: Option<Margin>,
    scrollable: bool,
}

/// What became of a [`Modal`] this frame.
pub struct ModalResponse<R> {
    pub inner: R,
    /// The user asked to close it: the close button, a click on the
    /// backdrop or Escape. Only set when closing or dismissal is enabled.
    pub close_requested: bool,
}

impl Modal {
    /// A modal 512 points wide (`max-w-lg`), that only its own buttons close.
    pub fn new(id_salt: impl std::hash::Hash + std::fmt::Debug) -> Self {
        Self {
            id: Id::new(id_salt),
            title: None,
            closable: false,
            dismissible: false,
            width: 512.0,
            padding: None,
            scrollable: false,
        }
    }

    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Offer a close button, and close on a click beside the modal or Escape.
    pub fn closable(mut self, closable: bool) -> Self {
        self.closable = closable;
        self
    }

    /// Allow backdrop clicks or Escape to dismiss without adding a close button.
    pub fn dismissible(mut self, dismissible: bool) -> Self {
        self.dismissible = dismissible;
        self
    }

    pub fn width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    /// The body's padding, instead of the default for a modal with or
    /// without a title.
    pub fn padding(mut self, padding: impl Into<Margin>) -> Self {
        self.padding = Some(padding.into());
        self
    }

    /// Scroll the body when it outgrows the window, as offline Activation can.
    pub fn scrollable(mut self, scrollable: bool) -> Self {
        self.scrollable = scrollable;
        self
    }

    pub fn show<R>(
        self,
        ctx: &Context,
        add_contents: impl FnOnce(&mut Ui) -> R,
    ) -> ModalResponse<R> {
        let mut close_clicked = false;
        // At most 85% of the window tall, and a margin around it on small ones.
        let screen = ctx.content_rect();
        let width = self.width.min(screen.width() - 48.0);
        let max_height = screen.height() * 0.85;

        let modal = egui::Modal::new(self.id)
            .backdrop_color(color::BACKDROP)
            .frame(Frame::NONE)
            .show(ctx, |ui| {
                let background = ui.painter().add(Shape::Noop);
                let response = ui.vertical(|ui| {
                    ui.set_width(width);
                    ui.set_max_height(max_height);
                    let spacing = ui.spacing().item_spacing;
                    ui.spacing_mut().item_spacing = vec2(0.0, 0.0);

                    if let Some(title) = &self.title {
                        let header =
                            Frame::new()
                                .inner_margin(Margin::symmetric(24, 16))
                                .show(ui, |ui| {
                                    ui.set_width(width - 48.0);
                                    centered_row(ui, |ui| {
                                        ui.add(
                                            Text::new(title.to_uppercase(), role::MODAL_TITLE)
                                                .color(color::ACCENT),
                                        );
                                        ui.with_layout(
                                            Layout::right_to_left(Align::Center),
                                            |ui| {
                                                if self.closable {
                                                    close_clicked =
                                                        ui.add(IconButton::close()).clicked();
                                                }
                                            },
                                        );
                                    });
                                });
                        let rect = header.response.rect;
                        ui.painter().hline(
                            rect.x_range(),
                            rect.bottom() - 0.5,
                            border(color::BORDER),
                        );
                    }

                    let padding = self.padding.unwrap_or(if self.title.is_some() {
                        Margin::symmetric(24, 20)
                    } else {
                        Margin::same(40)
                    });
                    Frame::new()
                        .inner_margin(padding)
                        .show(ui, |ui| {
                            ui.set_width(width - padding.sum().x);
                            ui.spacing_mut().item_spacing = spacing;
                            if self.scrollable {
                                ScrollArea::vertical()
                                    .id_salt(self.id.with("body"))
                                    .max_height(ui.available_height().max(0.0))
                                    .show(ui, add_contents)
                                    .inner
                            } else {
                                add_contents(ui)
                            }
                        })
                        .inner
                });

                let rect = response.response.rect;
                if self.title.is_none() && self.closable {
                    let corner = egui::Rect::from_min_size(
                        rect.right_top() + vec2(-40.0, 12.0),
                        vec2(28.0, 28.0),
                    );
                    close_clicked = ui.put(corner, IconButton::close()).clicked();
                }
                let painter = ui.painter();
                painter.set(
                    background,
                    backdrop::diagonal_gradient(
                        rect,
                        radius::CARD,
                        color::SURFACE_ALT,
                        color::SURFACE,
                    ),
                );
                painter.rect_stroke(
                    rect,
                    radius::CARD,
                    border(color::BORDER),
                    StrokeKind::Inside,
                );
                response.inner
            });

        let close_requested =
            (self.closable || self.dismissible) && (close_clicked || modal.should_close());
        ModalResponse {
            inner: modal.inner,
            close_requested,
        }
    }
}
