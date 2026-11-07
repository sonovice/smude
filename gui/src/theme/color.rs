//! The palette: a neutral-warm dark UI with a soft gold accent.
//!
//! Translucent tokens are meant to be painted over the dark surfaces, as
//! in the Tailwind theme they come from. For a Tailwind opacity modifier
//! (`text-accent/60`) use `ACCENT.gamma_multiply(0.6)`.

use eframe::egui;
use eframe::egui::Color32;

/// `rgba()` with the alpha as a fraction, as the Tailwind theme writes it.
const fn rgba(r: u8, g: u8, b: u8, alpha: f32) -> Color32 {
    Color32::from_rgba_unmultiplied_const(r, g, b, (alpha * 255.0 + 0.5) as u8)
}

// Surfaces, from the window background up to raised controls.

/// The window background.
pub const BG: Color32 = Color32::from_rgb(0x15, 0x15, 0x15);
/// Panels: the header, the sidebar, the status bar.
pub const SURFACE: Color32 = Color32::from_rgb(0x19, 0x1a, 0x1b);
/// Popovers, tooltips, table headers, and where card gradients start.
pub const SURFACE_ALT: Color32 = Color32::from_rgb(0x22, 0x23, 0x25);
/// Hovered rows.
pub const SURFACE_HI: Color32 = Color32::from_rgb(0x2a, 0x2b, 0x2d);
/// The off track of a toggle.
pub const SURFACE_RAISED: Color32 = Color32::from_rgb(0x36, 0x36, 0x36);
/// Card, divider and button borders.
pub const BORDER: Color32 = Color32::from_rgb(0x36, 0x36, 0x36);
/// Borders that must stand out: inputs, popovers, modal buttons.
pub const BORDER_HI: Color32 = Color32::from_rgb(0x48, 0x48, 0x48);

// The accent, a soft gold.

/// Headings, the selection, hovered controls.
pub const ACCENT: Color32 = Color32::from_rgb(0xff, 0xcc, 0x66);
/// A hovered filled accent button.
pub const ACCENT_BRIGHT: Color32 = Color32::from_rgb(0xff, 0xd8, 0x8a);
/// Accent fills that should not shout, like the disk usage bar.
pub const ACCENT_DIM: Color32 = Color32::from_rgb(0xcc, 0x99, 0x33);
/// The background of an armed scheduled start.
pub const ACCENT_MUTED: Color32 = Color32::from_rgb(0x2a, 0x22, 0x08);
/// The faint wash behind a hovered control.
pub const ACCENT_GLOW: Color32 = rgba(0xff, 0xcc, 0x66, 0.06);
/// The wash behind a selected control.
pub const ACCENT_GLOW_STRONG: Color32 = rgba(0xff, 0xcc, 0x66, 0.14);

/// Warnings; the accent's hue, kept apart so warnings can change alone.
pub const AMBER: Color32 = ACCENT;

// Status.

/// Connected, done, safe.
pub const GREEN: Color32 = Color32::from_rgb(0x00, 0xd4, 0x7a);
/// The background of a success notice.
pub const GREEN_DIM: Color32 = rgba(0x00, 0xd4, 0x7a, 0.10);
/// Errors and the REC button.
pub const RED: Color32 = Color32::from_rgb(0xe0, 0x40, 0x40);
/// Recording in progress, and errors that must be seen at once.
pub const RED_BRIGHT: Color32 = Color32::from_rgb(0xff, 0x52, 0x52);
/// The background of an error notice.
pub const RED_DIM: Color32 = rgba(0xe0, 0x40, 0x40, 0.08);

// Text.

/// Body text.
pub const TEXT: Color32 = Color32::WHITE;
/// Secondary text: labels, idle buttons.
pub const TEXT_MID: Color32 = Color32::from_rgb(0xaa, 0xaa, 0xaa);
/// Hints, placeholders, captions.
pub const TEXT_DIM: Color32 = rgba(0xff, 0xff, 0xff, 0.5);

/// Dims the app behind a modal.
pub const BACKDROP: Color32 = rgba(0, 0, 0, 0.7);

// Level meters.

/// Below -10 dBFS.
pub const METER_GREEN: Color32 = Color32::from_rgb(0x22, 0xc5, 0x5e);
/// From -10 dBFS.
pub const METER_YELLOW: Color32 = Color32::from_rgb(0xea, 0xb3, 0x08);
/// From -3 dBFS, and clipping.
pub const METER_RED: Color32 = Color32::from_rgb(0xef, 0x44, 0x44);
