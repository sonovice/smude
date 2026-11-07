use eframe::egui;
use eframe::egui::{Color32, CornerRadius, Response, Sense, StrokeKind, Ui, Vec2, Widget, vec2};

use crate::theme::{Icon, Type, border, color, radius, role};

/// How a button is drawn at rest and under the pointer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ButtonStyle {
    pub idle: Look,
    pub hover: Look,
}

/// The colors of a button in one state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Look {
    pub fill: Color32,
    pub stroke: Color32,
    pub text: Color32,
}

const fn look(fill: Color32, stroke: Color32, text: Color32) -> Look {
    Look { fill, stroke, text }
}

const CLEAR: Color32 = Color32::TRANSPARENT;

impl ButtonStyle {
    /// The everyday button: grey outline, gold under the pointer (RESCAN,
    /// CONNECT, Save PDF).
    pub const OUTLINE: Self = Self {
        idle: look(CLEAR, color::BORDER, color::TEXT_MID),
        hover: look(color::ACCENT_GLOW, color::ACCENT, color::ACCENT),
    };
    /// [`OUTLINE`](Self::OUTLINE) for minor actions, in dim text (REFRESH).
    pub const SUBTLE: Self = Self {
        idle: look(CLEAR, color::BORDER, color::TEXT_DIM),
        ..Self::OUTLINE
    };
    /// A gold outline that fills under the pointer (ENTER KEY).
    pub const ACCENT: Self = Self {
        idle: look(CLEAR, color::ACCENT, color::ACCENT),
        hover: look(color::ACCENT, color::ACCENT, color::BG),
    };
    /// The one action a modal is for, filled gold (ACCEPT).
    pub const PRIMARY: Self = Self {
        idle: look(color::ACCENT, color::ACCENT, color::BG),
        hover: look(color::ACCENT_BRIGHT, color::ACCENT, color::BG),
    };
    /// Neutral modal actions, white under the pointer (DISMISS).
    pub const NEUTRAL: Self = Self {
        idle: look(CLEAR, color::BORDER_HI, color::TEXT_MID),
        hover: look(CLEAR, color::TEXT, color::TEXT),
    };
    /// The REC button, ready to record.
    pub const RECORD: Self = Self {
        idle: look(rgba(color::RED, 0.2), color::RED, color::RED),
        hover: look(rgba(color::RED, 0.3), color::RED, color::RED),
    };
    /// The STOP button while recording.
    pub const RECORDING: Self = Self {
        idle: look(color::RED_BRIGHT, color::RED_BRIGHT, color::TEXT),
        hover: look(color::RED_BRIGHT, color::RED_BRIGHT, color::TEXT),
    };
    /// No chrome until the pointer comes, like the close button of a modal.
    pub const GHOST: Self = Self {
        idle: look(CLEAR, CLEAR, color::TEXT_DIM),
        hover: look(rgba(Color32::WHITE, 0.1), CLEAR, color::TEXT),
    };

    fn look(&self, hot: bool) -> Look {
        if hot { self.hover } else { self.idle }
    }
}

/// `color` at `alpha`, as Tailwind's `bg-red/20`.
const fn rgba(color: Color32, alpha: f32) -> Color32 {
    Color32::from_rgba_unmultiplied_const(
        color.r(),
        color.g(),
        color.b(),
        (alpha * 255.0 + 0.5) as u8,
    )
}

/// The button sizes of the web UI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ButtonSize {
    /// Modal actions (`px-6 py-2.5 text-sm`).
    Large,
    /// The default (`px-4 py-2 text-xs`).
    Regular,
    /// Header and toolbar buttons, 32 points tall (`h-8 px-2.5`).
    Toolbar,
    /// Buttons in a card header (`px-3 py-1`, `rounded-md`).
    Compact,
    /// Buttons beside a small label (`px-2 py-1 text-[11px]`, `rounded-md`).
    Small,
}

impl ButtonSize {
    /// The label's role, the padding around it and the corner radius.
    fn metrics(self) -> (Type, Vec2, u8) {
        match self {
            ButtonSize::Large => (role::BUTTON_LARGE, vec2(24.0, 10.0), radius::LG),
            ButtonSize::Regular => (role::BUTTON.leading(16.0), vec2(16.0, 8.0), radius::LG),
            ButtonSize::Toolbar => (role::BUTTON.leading(16.0), vec2(10.0, 8.0), radius::LG),
            ButtonSize::Compact => (role::BUTTON, vec2(12.0, 4.0), radius::MD),
            ButtonSize::Small => (role::BUTTON.size(11.0), vec2(8.0, 4.0), radius::MD),
        }
    }
}

/// A button with a text label and maybe an icon before it.
///
/// ```ignore
/// if ui.add(Button::new("CONNECT")).clicked() { … }
/// ui.add(Button::new("Save PDF").icon(Icon::FileText).size(ButtonSize::Toolbar));
/// ```
/// Disable it with `ui.add_enabled`; it then fades as in the web UI.
pub struct Button {
    label: String,
    icon: Option<Icon>,
    style: ButtonStyle,
    size: ButtonSize,
    selected: bool,
    min_width: f32,
}

impl Button {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            icon: None,
            style: ButtonStyle::OUTLINE,
            size: ButtonSize::Regular,
            selected: false,
            min_width: 0.0,
        }
    }

    pub fn icon(mut self, icon: Icon) -> Self {
        self.icon = Some(icon);
        self
    }

    pub fn style(mut self, style: ButtonStyle) -> Self {
        self.style = style;
        self
    }

    pub fn size(mut self, size: ButtonSize) -> Self {
        self.size = size;
        self
    }

    /// Keep the hover look, as for the button of an open menu.
    pub fn selected(mut self, selected: bool) -> Self {
        self.selected = selected;
        self
    }

    /// Stretch to at least this width, the label centered.
    pub fn min_width(mut self, min_width: f32) -> Self {
        self.min_width = min_width;
        self
    }
}

impl Widget for Button {
    fn ui(self, ui: &mut Ui) -> Response {
        let (role, padding, radius) = self.size.metrics();
        let icon_size = role.size + 2.0;
        let gap = 6.0;

        let galley = role.layout(ui, self.label.clone(), Color32::PLACEHOLDER, f32::INFINITY);
        let icon_width = if self.icon.is_some() {
            icon_size + gap
        } else {
            0.0
        };
        let content = vec2(icon_width + galley.size().x, role.line_height);
        // CSS borders add to the box.
        let mut size = content + 2.0 * padding + Vec2::splat(2.0);
        size.x = size.x.max(self.min_width);
        if self.size == ButtonSize::Toolbar {
            size.y = 32.0;
        }

        let (rect, response) = ui.allocate_exact_size(size, Sense::click());
        response.widget_info(|| {
            egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), &self.label)
        });
        if ui.is_rect_visible(rect) {
            let look = self.style.look(self.selected || is_hot(&response));
            let painter = ui.painter();
            painter.rect(
                rect,
                radius,
                look.fill,
                border(look.stroke),
                StrokeKind::Inside,
            );

            let mut x = rect.center().x - content.x / 2.0;
            if let Some(icon) = self.icon {
                icon.paint(
                    painter,
                    egui::pos2(x + icon_size / 2.0, rect.center().y),
                    icon_size,
                    look.text,
                );
                x += icon_width;
            }
            let pos = egui::pos2(x, rect.center().y - galley.size().y / 2.0);
            painter.galley(pos, galley, look.text);
        }
        response
    }
}

/// A square button showing just an icon, like the Help button.
pub struct IconButton {
    icon: Icon,
    accessible_label: Option<String>,
    style: ButtonStyle,
    side: f32,
    icon_size: f32,
    radius: u8,
    selected: bool,
}

impl IconButton {
    /// A 32 point bordered button with a 16 point icon.
    pub fn new(icon: Icon) -> Self {
        Self {
            icon,
            accessible_label: None,
            style: ButtonStyle::OUTLINE,
            side: 32.0,
            icon_size: 16.0,
            radius: radius::LG,
            selected: false,
        }
    }

    /// The close button of modals and panels: 28 points, a 12 point ×, no
    /// chrome until the pointer comes.
    pub fn close() -> Self {
        Self {
            icon: Icon::X,
            accessible_label: Some("Schließen".into()),
            style: ButtonStyle::GHOST,
            side: 28.0,
            icon_size: 14.0,
            radius: 6,
            selected: false,
        }
    }

    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.accessible_label = Some(label.into());
        self
    }

    pub fn style(mut self, style: ButtonStyle) -> Self {
        self.style = style;
        self
    }

    /// The button's side and the icon's size, in points.
    pub fn size(mut self, side: f32, icon_size: f32) -> Self {
        self.side = side;
        self.icon_size = icon_size;
        self
    }

    pub fn radius(mut self, radius: u8) -> Self {
        self.radius = radius;
        self
    }

    /// Keep the hover look, as for the button of an open menu.
    pub fn selected(mut self, selected: bool) -> Self {
        self.selected = selected;
        self
    }
}

impl Widget for IconButton {
    fn ui(self, ui: &mut Ui) -> Response {
        let (rect, response) = ui.allocate_exact_size(Vec2::splat(self.side), Sense::click());
        if let Some(label) = &self.accessible_label {
            response.widget_info(|| {
                egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), label)
            });
        }
        if ui.is_rect_visible(rect) {
            let look = self.style.look(self.selected || is_hot(&response));
            let painter = ui.painter();
            painter.rect(
                rect,
                CornerRadius::same(self.radius),
                look.fill,
                border(look.stroke),
                StrokeKind::Inside,
            );
            self.icon
                .paint(painter, rect.center(), self.icon_size, look.text);
        }
        response
    }
}

/// Under the pointer and able to act on it.
pub(crate) fn is_hot(response: &Response) -> bool {
    response.enabled() && response.hovered()
}

/// Two buttons joined in one outline: an action and a smaller one beside
/// it, like REC and the clock that schedules a start.
///
/// ```ignore
/// let (rec, clock) = SplitButton::new("● REC", Icon::Clock).style(ButtonStyle::RECORD).show(ui);
/// ```
pub struct SplitButton {
    label: String,
    side: Icon,
    style: ButtonStyle,
    side_selected: bool,
}

impl SplitButton {
    pub fn new(label: impl Into<String>, side: Icon) -> Self {
        Self {
            label: label.into(),
            side,
            style: ButtonStyle::OUTLINE,
            side_selected: false,
        }
    }

    pub fn style(mut self, style: ButtonStyle) -> Self {
        self.style = style;
        self
    }

    /// Keep the side button's hover look, as while its popover is open.
    pub fn side_selected(mut self, selected: bool) -> Self {
        self.side_selected = selected;
        self
    }

    /// The responses of the action and of the side button.
    pub fn show(self, ui: &mut Ui) -> (Response, Response) {
        let role = role::BUTTON.leading(16.0);
        let height = 32.0;
        let icon_size = 14.0;
        let galley = role.layout(ui, self.label, Color32::PLACEHOLDER, f32::INFINITY);
        let main_width = galley.size().x + 24.0 + 1.0;
        let side_width = icon_size + 16.0 + 1.0;
        let (rect, _) =
            ui.allocate_exact_size(vec2(main_width + side_width, height), Sense::hover());
        let main_rect = egui::Rect::from_min_size(rect.min, vec2(main_width, height));
        let side_rect =
            egui::Rect::from_min_max(egui::pos2(main_rect.right(), rect.top()), rect.max);
        let main = ui.interact(main_rect, ui.next_auto_id().with("main"), Sense::click());
        let side = ui.interact(side_rect, ui.next_auto_id().with("side"), Sense::click());

        if ui.is_rect_visible(rect) {
            let painter = ui.painter();
            // The fill follows each half's state; the outline is the idle one.
            painter.rect_filled(rect, radius::LG, self.style.idle.fill);
            let main_look = self.style.look(is_hot(&main));
            let side_look = self.style.look(self.side_selected || is_hot(&side));
            let left = CornerRadius {
                nw: radius::LG,
                sw: radius::LG,
                ne: 0,
                se: 0,
            };
            let right = CornerRadius {
                nw: 0,
                sw: 0,
                ne: radius::LG,
                se: radius::LG,
            };
            if main_look != self.style.idle {
                painter.rect_filled(main_rect, left, main_look.fill);
            }
            if side_look != self.style.idle {
                painter.rect_filled(side_rect, right, side_look.fill);
            }
            painter.rect_stroke(
                rect,
                radius::LG,
                border(self.style.idle.stroke),
                StrokeKind::Inside,
            );
            painter.vline(
                side_rect.left(),
                rect.y_range(),
                border(self.style.idle.stroke),
            );
            let pos = main_rect.center() - galley.size() / 2.0;
            painter.galley(pos, galley, main_look.text);
            self.side
                .paint(painter, side_rect.center(), icon_size, side_look.text);
        }
        (main, side)
    }
}
