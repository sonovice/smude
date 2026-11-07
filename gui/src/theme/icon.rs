//! The lucide icons the UI uses, drawn from lucide's icon font so they
//! stay crisp at any size and take any color.
//!
//! Adding one: add its name to `ICONS` in `assets/fonts/subset.sh`, rerun
//! the script, and add a variant here with its code point from lucide's
//! `codepoints.json`.

use eframe::egui;
use eframe::egui::{Align2, Color32, FontFamily, FontId, Painter, Pos2, Rect, RichText};

use super::font::ICONS;

/// A lucide icon. The variants the web UI drew as hand-made SVGs map to
/// their closest lucide icon: [`Icon::FileText`] for Save PDF,
/// [`Icon::Folder`] for choosing a folder, [`Icon::ExternalLink`] for
/// opening one, [`Icon::Sun`] for the light toggle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Icon {
    Bug,
    Check,
    CircleAlert,
    CircleCheck,
    CircleQuestionMark,
    Clock,
    ExternalLink,
    FileText,
    Folder,
    FolderOpen,
    HardDriveDownload,
    KeyRound,
    Sun,
    TriangleAlert,
    Unplug,
    X,
}

impl Icon {
    /// The icon's character in the icon font, which every text family falls
    /// back to: an icon can be part of any text.
    pub const fn char(self) -> char {
        match self {
            Icon::Bug => '\u{e20c}',
            Icon::Check => '\u{e06c}',
            Icon::CircleAlert => '\u{e077}',
            Icon::CircleCheck => '\u{e226}',
            Icon::CircleQuestionMark => '\u{e082}',
            Icon::Clock => '\u{e087}',
            Icon::ExternalLink => '\u{e0b9}',
            Icon::FileText => '\u{e0cc}',
            Icon::Folder => '\u{e0d7}',
            Icon::FolderOpen => '\u{e247}',
            Icon::HardDriveDownload => '\u{e4e5}',
            Icon::KeyRound => '\u{e4a3}',
            Icon::Sun => '\u{e178}',
            Icon::TriangleAlert => '\u{e193}',
            Icon::Unplug => '\u{e45d}',
            Icon::X => '\u{e1b2}',
        }
    }

    /// The icon at `size` points, as lucide's `size` prop, for egui's widgets.
    pub fn rich(self, size: f32) -> RichText {
        RichText::new(self.char().to_string()).font(Self::font_id(size))
    }

    /// Paint the icon `size` points square, centered on `center`.
    pub fn paint(self, painter: &Painter, center: Pos2, size: f32, color: Color32) -> Rect {
        painter.text(
            center,
            Align2::CENTER_CENTER,
            self.char(),
            Self::font_id(size),
            color,
        )
    }

    fn font_id(size: f32) -> FontId {
        FontId::new(size, FontFamily::Name(ICONS.into()))
    }
}
