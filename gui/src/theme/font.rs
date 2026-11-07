//! Typefaces and the typographic roles of the UI.
//!
//! The UI sets text in Inter, headings in Parkinsans and numbers that must
//! line up in JetBrains Mono. The fonts are variable; each weight the UI
//! uses is its own egui font family, since a [`FontId`] carries no weight.
//! Every family falls back to the lucide icons and then to egui's own fonts,
//! so icons can sit in a line of text and any script still shows.

use std::sync::Arc;

use eframe::egui;
use eframe::egui::epaint::text::{FontTweak, VariationCoords};
use eframe::egui::text::LayoutJob;
use eframe::egui::{
    Color32, FontData, FontDefinitions, FontFamily, FontId, Galley, RichText, TextFormat, Ui,
};

const INTER: &[u8] = include_bytes!("../../assets/fonts/Inter.ttf");
const PARKINSANS: &[u8] = include_bytes!("../../assets/fonts/Parkinsans.ttf");
const JETBRAINS_MONO: &[u8] = include_bytes!("../../assets/fonts/JetBrainsMono.ttf");
const LUCIDE: &[u8] = include_bytes!("../../assets/fonts/lucide.ttf");

/// The font key and family of the lucide icons alone, for drawing an icon
/// outside of text.
pub(super) const ICONS: &str = "lucide";

/// A typeface in one weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Face {
    /// Inter, regular: body text.
    Sans,
    /// Inter, semibold: names, segment labels.
    SansSemiBold,
    /// Inter, bold: buttons and small labels.
    SansBold,
    /// Parkinsans, semibold: card titles.
    HeadingSemiBold,
    /// Parkinsans, bold: the brand, table headers, modal titles.
    HeadingBold,
    /// JetBrains Mono, regular: IP addresses, versions, times.
    Mono,
    /// JetBrains Mono, bold: the recording time.
    MonoBold,
}

impl Face {
    const ALL: [Face; 7] = [
        Face::Sans,
        Face::SansSemiBold,
        Face::SansBold,
        Face::HeadingSemiBold,
        Face::HeadingBold,
        Face::Mono,
        Face::MonoBold,
    ];

    /// The font key, which is also the name of the family for the weights
    /// egui has no built-in family for.
    fn key(self) -> &'static str {
        match self {
            Face::Sans => "Inter",
            Face::SansSemiBold => "Inter SemiBold",
            Face::SansBold => "Inter Bold",
            Face::HeadingSemiBold => "Parkinsans SemiBold",
            Face::HeadingBold => "Parkinsans Bold",
            Face::Mono => "JetBrains Mono",
            Face::MonoBold => "JetBrains Mono Bold",
        }
    }

    fn source(self) -> (&'static [u8], f32) {
        match self {
            Face::Sans => (INTER, 400.0),
            Face::SansSemiBold => (INTER, 600.0),
            Face::SansBold => (INTER, 700.0),
            Face::HeadingSemiBold => (PARKINSANS, 600.0),
            Face::HeadingBold => (PARKINSANS, 700.0),
            Face::Mono => (JETBRAINS_MONO, 400.0),
            Face::MonoBold => (JETBRAINS_MONO, 700.0),
        }
    }

    /// The egui family. The regular weights are egui's `Proportional` and
    /// `Monospace`, so egui's own widgets use them too.
    pub fn family(self) -> FontFamily {
        match self {
            Face::Sans => FontFamily::Proportional,
            Face::Mono => FontFamily::Monospace,
            _ => FontFamily::Name(self.key().into()),
        }
    }

    pub fn font_id(self, size: f32) -> FontId {
        FontId::new(size, self.family())
    }
}

/// Every face as its own family, each falling back to the icons and to
/// egui's default fonts.
pub(super) fn definitions() -> FontDefinitions {
    let mut defs = FontDefinitions::empty();
    // Match egui's proportional fallbacks and scale tweaks. All app
    // families use these; Hack was loaded but never part of any family.
    let mut fallbacks = Vec::new();
    for (name, bytes, scale) in [
        ("Ubuntu-Light", epaint_default_fonts::UBUNTU_LIGHT, 1.0),
        (
            "NotoEmoji-Regular",
            epaint_default_fonts::NOTO_EMOJI_REGULAR,
            0.81,
        ),
        ("emoji-icon-font", epaint_default_fonts::EMOJI_ICON, 0.90),
    ] {
        defs.font_data.insert(
            name.into(),
            Arc::new(FontData::from_static(bytes).tweak(FontTweak {
                scale,
                ..Default::default()
            })),
        );
        fallbacks.push(name.to_owned());
    }

    defs.font_data
        .insert(ICONS.into(), Arc::new(FontData::from_static(LUCIDE)));
    defs.families
        .insert(FontFamily::Name(ICONS.into()), vec![ICONS.into()]);

    for face in Face::ALL {
        let (font, weight) = face.source();
        let tweak = FontTweak {
            coords: VariationCoords::new([(b"wght", weight)]),
            ..FontTweak::default()
        };
        defs.font_data.insert(
            face.key().into(),
            Arc::new(FontData::from_static(font).tweak(tweak)),
        );
        let family = [face.key().into(), ICONS.into()]
            .into_iter()
            .chain(fallbacks.iter().cloned())
            .collect();
        defs.families.insert(face.family(), family);
    }
    defs
}

/// A typographic role: a face, a size and the CSS spacing around it.
///
/// Widgets size their boxes by [`line_height`](Self::line_height) and center
/// the text in it, as CSS does, so paddings ported from Tailwind give the
/// same heights as in the web UI.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Type {
    pub face: Face,
    /// The font size in points (CSS pixels).
    pub size: f32,
    /// The letter spacing in em, as Tailwind's `tracking-*`.
    pub tracking: f32,
    /// The height of a line in points, as CSS `line-height`.
    pub line_height: f32,
}

impl Type {
    /// A role in `face` at `size`, with the app's body line height of
    /// 1.45, which every `text-[Npx]` inherits.
    pub const fn new(face: Face, size: f32) -> Self {
        Self {
            face,
            size,
            tracking: 0.0,
            line_height: size * 1.45,
        }
    }

    /// Letter spacing in em: Tailwind's `tracking-wide` is 0.025,
    /// `tracking-wider` 0.05, `tracking-widest` 0.1.
    pub const fn tracking(self, em: f32) -> Self {
        Self {
            tracking: em,
            ..self
        }
    }

    /// The line height in points.
    pub const fn leading(self, line_height: f32) -> Self {
        Self {
            line_height,
            ..self
        }
    }

    pub const fn face(self, face: Face) -> Self {
        Self { face, ..self }
    }

    pub const fn size(self, size: f32) -> Self {
        Self { size, ..self }
    }

    pub fn font_id(self) -> FontId {
        self.face.font_id(self.size)
    }

    /// The letter spacing in points.
    pub fn letter_spacing(self) -> f32 {
        self.tracking * self.size
    }

    /// `text` in this role, for egui's own widgets.
    pub fn rich(self, text: impl Into<String>) -> RichText {
        RichText::new(text)
            .font(self.font_id())
            .extra_letter_spacing(self.letter_spacing())
    }

    /// The format of a section of a [`LayoutJob`] in this role.
    pub fn format(self, color: Color32) -> TextFormat {
        TextFormat {
            font_id: self.font_id(),
            extra_letter_spacing: self.letter_spacing(),
            color,
            ..TextFormat::default()
        }
    }

    /// Lay out `text` on one line, cut short with "…" beyond `max_width`.
    pub fn layout(
        self,
        ui: &Ui,
        text: impl Into<String>,
        color: Color32,
        max_width: f32,
    ) -> Arc<Galley> {
        let mut job = LayoutJob::single_section(text.into(), self.format(color));
        job.wrap.max_width = max_width;
        job.wrap.max_rows = 1;
        job.wrap.break_anywhere = true;
        ui.painter().layout_job(job)
    }

    /// Lay out `text` wrapped at `width`, with this role's line height.
    pub fn layout_wrapped(
        self,
        ui: &Ui,
        text: impl Into<String>,
        color: Color32,
        width: f32,
    ) -> Arc<Galley> {
        let format = TextFormat {
            line_height: Some(self.line_height),
            ..self.format(color)
        };
        let mut job = LayoutJob::simple_format(text.into(), format);
        job.wrap.max_width = width;
        ui.painter().layout_job(job)
    }
}

/// The roles the web UI used, named for where they appear. Derive others
/// with [`Type::tracking`], [`Type::size`] or [`Type::face`].
pub mod role {
    use super::{Face, Type};

    /// Running text (the page's 14 px).
    pub const BODY: Type = Type::new(Face::Sans, 14.0);
    /// List rows, names, values (`text-[13px]`).
    pub const LIST: Type = Type::new(Face::Sans, 13.0);
    /// Captions, hints, the status bar (`text-[12px]`).
    pub const SMALL: Type = Type::new(Face::Sans, 12.0);
    /// Fine print under labels (`text-[11px]`).
    pub const TINY: Type = Type::new(Face::Sans, 11.0);
    /// Tooltip text.
    pub const TOOLTIP: Type = Type::new(Face::Sans, 12.0).leading(19.2);
    /// IP addresses and other figures that must line up.
    pub const MONO: Type = Type::new(Face::Mono, 13.0);
    /// Firmware versions, paths, times.
    pub const MONO_SMALL: Type = Type::new(Face::Mono, 12.0);

    /// Button labels (`text-[12px] font-bold tracking-wider`).
    pub const BUTTON: Type = Type::new(Face::SansBold, 12.0).tracking(0.05);
    /// Labels of the large buttons in modals (`text-sm font-bold tracking-wider`).
    pub const BUTTON_LARGE: Type = Type::new(Face::SansBold, 14.0).tracking(0.05).leading(20.0);
    /// Segment labels (`text-xs font-semibold tracking-wider`).
    pub const SEGMENT: Type = Type::new(Face::SansSemiBold, 12.0)
        .tracking(0.05)
        .leading(16.0);
    /// The small uppercase labels inside cards, like STREAM and DRIVER.
    pub const LABEL: Type = Type::new(Face::SansBold, 11.0).tracking(0.12);

    /// Card titles, like SELECT CONSOLE.
    pub const SECTION_TITLE: Type = Type::new(Face::HeadingSemiBold, 12.0).tracking(0.10);
    /// Table column headers.
    pub const COLUMN_HEADER: Type = Type::new(Face::HeadingBold, 12.0).tracking(0.12);
    /// Panel titles, like the sidebar's SETTINGS.
    pub const PANEL_TITLE: Type = Type::new(Face::HeadingSemiBold, 13.0).tracking(0.08);
    /// Modal titles.
    pub const MODAL_TITLE: Type = Type::new(Face::HeadingBold, 13.0).tracking(0.12);
    /// The headline of an alert, like AUDIO DEVICE DISCONNECTED (`text-2xl`).
    pub const DISPLAY: Type = Type::new(Face::HeadingBold, 24.0)
        .tracking(0.05)
        .leading(32.0);

    /// WING COMPANION on the start screens.
    pub const BRAND: Type = Type::new(Face::HeadingBold, 16.0)
        .tracking(0.2)
        .leading(20.0);
    /// WING COMPANION in the header.
    pub const BRAND_COMPACT: Type = Type::new(Face::HeadingBold, 12.0)
        .tracking(0.18)
        .leading(15.0);
    /// The line under the brand, like LIVE RECORDING TOOLKIT.
    pub const TAGLINE: Type = Type::new(Face::Sans, 12.0).tracking(0.1).leading(15.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_faces_keep_latin_greek_cyrillic_and_symbol_fallbacks() {
        let ctx = egui::Context::default();
        ctx.set_fonts(definitions());
        ctx.begin_pass(egui::RawInput::default());
        for face in Face::ALL {
            ctx.fonts_mut(|fonts| {
                for c in "ÄÖÜéŁΩЖ√".chars() {
                    assert!(fonts.has_glyph(&face.font_id(14.0), c), "{face:?}: {c}");
                }
            });
        }
        ctx.end_pass().textures_delta.clear();
    }
}
