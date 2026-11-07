use eframe::egui;
use eframe::egui::{Color32, Frame, InnerResponse, Margin, Sense, Shape, Ui, UiBuilder, vec2};

use crate::theme::{border, color};

/// A full-width row that acts as one button, like a console in the list of
/// discovered consoles. The row lights up under the pointer; its contents
/// get the [`RowState`] to light up with it.
///
/// ```ignore
/// let row = ListRow::new().show(ui, |ui, state| {
///     let name = if state.hovered { color::ACCENT } else { color::TEXT_MID };
///     ui.add(Text::new(&console.model, role::LIST).color(name));
/// });
/// if row.response.clicked() { … }
/// ```
pub struct ListRow {
    selected: bool,
    divider: bool,
    padding: Margin,
    radius: u8,
    hover_fill: Color32,
    bleed: f32,
}

/// What a row's contents need to know to match the row.
#[derive(Clone, Copy, Debug)]
pub struct RowState {
    pub hovered: bool,
    pub selected: bool,
}

impl Default for ListRow {
    fn default() -> Self {
        Self::new()
    }
}

impl ListRow {
    /// A row of a list in a card: `px-3 py-2.5`, a divider below, a gold
    /// wash under the pointer.
    pub fn new() -> Self {
        Self {
            selected: false,
            divider: true,
            padding: Margin::symmetric(12, 10),
            radius: 0,
            hover_fill: color::ACCENT_GLOW,
            bleed: 0.0,
        }
    }

    /// Mark the row as the chosen one, washed in gold.
    pub fn selected(mut self, selected: bool) -> Self {
        self.selected = selected;
        self
    }

    /// Draw a divider below the row; off for the last row of a card.
    pub fn divider(mut self, divider: bool) -> Self {
        self.divider = divider;
        self
    }

    pub fn padding(mut self, padding: impl Into<Margin>) -> Self {
        self.padding = padding.into();
        self
    }

    /// Round the row's background, for rows that are not part of a card's
    /// edge.
    pub fn radius(mut self, radius: u8) -> Self {
        self.radius = radius;
        self
    }

    /// The background under the pointer.
    pub fn hover_fill(mut self, fill: Color32) -> Self {
        self.hover_fill = fill;
        self
    }

    /// Let the background reach this far past the row at the left and
    /// right, as Tailwind's `-mx-3 px-3`.
    pub fn bleed(mut self, bleed: f32) -> Self {
        self.bleed = bleed;
        self
    }

    pub fn show<R>(
        self,
        ui: &mut Ui,
        add_contents: impl FnOnce(&mut Ui, RowState) -> R,
    ) -> InnerResponse<R> {
        let background = ui.painter().add(Shape::Noop);
        let width = ui.available_width();
        // The divider is a CSS border, below the padding.
        let mut padding = self.padding;
        if self.divider {
            padding.bottom += 1;
        }
        let builder = UiBuilder::new().sense(Sense::click());
        let InnerResponse { inner, response } = ui.scope_builder(builder, |ui| {
            let state = RowState {
                hovered: super::is_hot(&ui.response()),
                selected: self.selected,
            };
            Frame::new()
                .inner_margin(padding)
                .show(ui, |ui| {
                    ui.set_width(width - padding.sum().x);
                    super::centered_row(ui, |ui| add_contents(ui, state)).inner
                })
                .inner
        });

        let rect = response.rect;
        let fill = if self.selected {
            color::ACCENT_GLOW_STRONG
        } else if super::is_hot(&response) {
            self.hover_fill
        } else {
            Color32::TRANSPARENT
        };
        let painter = ui.painter();
        painter.set(
            background,
            Shape::rect_filled(rect.expand2(vec2(self.bleed, 0.0)), self.radius, fill),
        );
        if self.divider {
            painter.hline(rect.x_range(), rect.bottom() - 0.5, border(color::BORDER));
        }
        InnerResponse { inner, response }
    }
}
