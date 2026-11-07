//! Visual foundation ported from Wing Companion.
#![allow(dead_code, unused_imports)]
//! The look of the app: colors, fonts, icons, radii, shadows and the
//! backgrounds, ported from the Tailwind theme of the web UI it replaces.
//!
//! [`install`] sets them up once, when the app starts. Screens then draw
//! with the [`widgets`](crate::widgets), which already wear the theme, and
//! reach for the tokens here only for what no widget covers:
//!
//! - Colors: [`color`], named for their role (`color::TEXT_MID`, not
//!   "grey"). A Tailwind opacity modifier like `text-accent/60` is
//!   `color::ACCENT.gamma_multiply(0.6)`.
//! - Text: a [`Type`] is a typographic role (face, size, letter spacing
//!   and CSS line height); [`role`] has the ones the UI uses, like
//!   `role::SECTION_TITLE`. Draw text with [`widgets::Text`](crate::widgets::Text),
//!   which sizes its line box as CSS did, or turn a role into egui text with
//!   [`Type::rich`] and [`Type::format`]. Plain egui labels get [`role::BODY`].
//! - Icons: [`Icon`]; `Icon::Clock.rich(14.0)` in any text, or
//!   [`Icon::paint`] at a point.
//! - Spacing: Tailwind's 4 px grid, ported as plain numbers: `px-3` is
//!   12.0, `py-2.5` is 10.0. A CSS border adds to the box, so a bordered
//!   box with `py-2` around 16 points of text is 34 tall; the widgets
//!   count it. [`radius`] and [`shadow`] hold the rest.
//! - Layout: a flex row with `items-center` is
//!   [`widgets::centered_row`](crate::widgets::centered_row), not
//!   `ui.horizontal`, which misaligns items of different heights; inside
//!   it, `ui.with_layout(Layout::right_to_left(Align::Center), …)` is
//!   `justify-between`. Outside such a row a right-to-left layout takes
//!   all the height there is.
//! - Backgrounds: [`backdrop`] paints the page, its glows and the gradients
//!   of cards and modals.
//!
//! `examples/gallery.rs` shows every widget and role; run it with
//! `cargo run -p wing-companion-ui --example gallery` after changing them.

pub mod backdrop;
pub mod color;
mod font;
mod icon;

use eframe::egui;
use eframe::egui::{
    Color32, Context, CornerRadius, Frame, Margin, Stroke, Style, TextStyle, Visuals,
};

pub use font::{Face, Type, role};
pub use icon::Icon;

/// Corner radii, as the Tailwind theme set them.
pub mod radius {
    /// `rounded`: small badges.
    pub const XS: u8 = 4;
    /// `rounded-sm`: tooltips.
    pub const SM: u8 = 8;
    /// `rounded-md`: compact buttons.
    pub const MD: u8 = 10;
    /// `rounded-lg`: buttons, inputs, popovers.
    pub const LG: u8 = 12;
    /// `rounded-[10px]`: cards and modals.
    pub const CARD: u8 = 10;
    /// `rounded-full`: egui caps a radius at half the shape's height.
    pub const PILL: u8 = u8::MAX;
}

/// Drop shadows.
pub mod shadow {
    use eframe::egui::{Color32, Shadow};

    /// `shadow-lg`: popovers and menus.
    pub const POPOVER: Shadow = Shadow {
        offset: [0, 10],
        blur: 15,
        spread: 0,
        color: Color32::from_black_alpha(26),
    };
    /// Tooltips, which float over anything.
    pub const TOOLTIP: Shadow = Shadow {
        offset: [0, 4],
        blur: 12,
        spread: 0,
        color: Color32::from_black_alpha(102),
    };
}

/// The hairline used for every border.
pub fn border(color: Color32) -> Stroke {
    Stroke::new(1.0, color)
}

/// The frame of popovers and menus.
pub fn popover_frame() -> Frame {
    Frame::new()
        .fill(color::SURFACE_ALT)
        .stroke(border(color::BORDER_HI))
        .corner_radius(radius::LG)
        .shadow(shadow::POPOVER)
}

/// Set up fonts, colors and spacing. Call once, before the first frame.
pub fn install(ctx: &Context) {
    ctx.set_fonts(font::definitions());
    ctx.set_theme(egui::Theme::Dark);
    let mut s = Style::default();
    style(&mut s);
    ctx.set_global_style(s);
}

fn style(style: &mut Style) {
    style.text_styles = [
        (TextStyle::Small, role::TINY.font_id()),
        (TextStyle::Body, role::BODY.font_id()),
        (TextStyle::Button, role::BUTTON.font_id()),
        (TextStyle::Heading, role::BRAND.font_id()),
        (TextStyle::Monospace, role::MONO.font_id()),
    ]
    .into();

    let spacing = &mut style.spacing;
    spacing.item_spacing = egui::vec2(8.0, 8.0);
    spacing.button_padding = egui::vec2(16.0, 8.0);
    spacing.menu_margin = Margin::symmetric(12, 8);
    spacing.window_margin = Margin::same(24);
    // Rows start this tall; the widgets size themselves.
    spacing.interact_size.y = 16.0;
    spacing.tooltip_width = 256.0;
    // The web UI hid scrollbars; these show only while scrolling.
    spacing.scroll = egui::style::ScrollStyle::floating();
    spacing.scroll.dormant_handle_opacity = 0.0;
    spacing.scroll.dormant_background_opacity = 0.0;

    style.interaction.tooltip_delay = 0.5;
    style.interaction.selectable_labels = false;
    style.visuals = visuals();
}

fn visuals() -> Visuals {
    let mut visuals = Visuals::dark();
    visuals.panel_fill = color::BG;
    visuals.window_fill = color::SURFACE_ALT;
    visuals.window_stroke = border(color::BORDER_HI);
    visuals.window_corner_radius = CornerRadius::same(radius::CARD);
    visuals.window_shadow = shadow::POPOVER;
    // Tooltips and popups, as the web UI's tooltips.
    visuals.menu_corner_radius = CornerRadius::same(radius::SM);
    visuals.popup_shadow = shadow::TOOLTIP;
    visuals.extreme_bg_color = color::BG;
    visuals.faint_bg_color = color::SURFACE;
    visuals.code_bg_color = color::SURFACE_ALT;
    visuals.hyperlink_color = color::ACCENT;
    visuals.warn_fg_color = color::AMBER;
    visuals.error_fg_color = color::RED;
    visuals.selection.bg_fill = color::ACCENT.gamma_multiply(0.3);
    visuals.selection.stroke = Stroke::new(1.0, color::TEXT);
    visuals.text_cursor.stroke = Stroke::new(1.5, color::ACCENT);
    visuals.disabled_alpha = 0.4;

    // egui's own widgets as the app's secondary buttons.
    let widgets = &mut visuals.widgets;
    widgets.noninteractive.bg_fill = color::SURFACE;
    widgets.noninteractive.weak_bg_fill = color::SURFACE;
    widgets.noninteractive.bg_stroke = border(color::BORDER);
    widgets.noninteractive.fg_stroke = Stroke::new(1.0, color::TEXT);
    for (state, fill, stroke, text) in [
        (
            &mut widgets.inactive,
            Color32::TRANSPARENT,
            color::BORDER,
            color::TEXT_MID,
        ),
        (
            &mut widgets.hovered,
            color::ACCENT_GLOW,
            color::ACCENT,
            color::ACCENT,
        ),
        (
            &mut widgets.active,
            color::ACCENT_GLOW_STRONG,
            color::ACCENT,
            color::ACCENT,
        ),
        (
            &mut widgets.open,
            color::ACCENT_GLOW_STRONG,
            color::ACCENT,
            color::ACCENT,
        ),
    ] {
        state.bg_fill = fill;
        state.weak_bg_fill = fill;
        state.bg_stroke = border(stroke);
        state.fg_stroke = Stroke::new(1.0, text);
        state.corner_radius = CornerRadius::same(radius::LG);
        state.expansion = 0.0;
    }
    visuals
}

// Compatibility names for the preview workspace.
pub use color::{
    ACCENT as GOLD, BG, BORDER, GREEN, RED, SURFACE as PANEL, SURFACE_ALT as RAISED,
    TEXT_DIM as MUTED,
};
pub fn primary(text: impl Into<String>) -> crate::widgets::Button {
    crate::widgets::Button::new(text).style(crate::widgets::ButtonStyle::PRIMARY)
}
pub fn section(ui: &mut egui::Ui, text: &str) {
    ui.add(
        crate::widgets::Text::new(text.to_uppercase(), role::SECTION_TITLE).color(color::ACCENT),
    );
}
