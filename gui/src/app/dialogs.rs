//! Dialog layout follows Wing Companion's About and quit dialogs.
use super::*;
use crate::theme::{Face, Type, color, role};
use crate::widgets::{Button, ButtonSize, ButtonStyle, Modal, Text};

pub(super) fn logo(ui: &mut egui::Ui, size: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::Vec2::splat(size), egui::Sense::hover());
    for k in 0..5 {
        let y = rect.top() + size * (0.24 + k as f32 * 0.13);
        ui.painter().line_segment(
            [
                egui::pos2(rect.left(), y),
                egui::pos2(rect.right(), y - size * 0.1),
            ],
            egui::Stroke::new(size * 0.035, color::ACCENT),
        );
    }
}

impl SmudeApp {
    pub(super) fn dialogs(&mut self, ctx: &egui::Context) {
        if self.show_about {
            let modal = Modal::new("about-smude").closable(true).width(448.).padding(0).show(ctx, |ui| {
                let width = ui.available_width();
                egui::ScrollArea::vertical().max_height((ctx.content_rect().height() * 0.85 - 70.).max(100.)).show(ui, |ui| {
                    egui::Frame::new().inner_margin(egui::Margin { left: 40, right: 40, top: 40, bottom: 32 }).show(ui, |ui| {
                        ui.set_width(width - 80.);
                        ui.spacing_mut().item_spacing.y = 20.;
                        ui.vertical_centered(|ui| {
                            logo(ui, 64.);
                            ui.vertical_centered(|ui| {
                                ui.spacing_mut().item_spacing.y = 4.;
                                ui.add(Text::new("SMUDE", Type::new(Face::HeadingBold, 18.).tracking(0.18).leading(28.)).color(color::ACCENT));
                                ui.add(Text::new("NOTENSEITEN ENTZERREN", role::SMALL.tracking(0.16)).color(color::TEXT_DIM));
                            });
                            ui.add(Text::new(concat!("v", env!("CARGO_PKG_VERSION")), role::SMALL.tracking(0.05)).color(color::TEXT_DIM));
                            super::centered::Paragraph::new().run("Fotografierte Notenseiten automatisch entzerren und als saubere, papierweiße Bilder exportieren.", role::SMALL.leading(19.5), color::TEXT_MID).show(ui);
                            super::centered::Paragraph::new().run("Die Verarbeitung bleibt auf deinem Computer.", role::SMALL.leading(19.5), color::TEXT_DIM).show(ui);
                            ui.add(Text::new("Simon Waloschek", role::SMALL).color(color::TEXT_DIM));
                            if ui.add(Button::new("Projektlizenz").size(ButtonSize::Compact).style(ButtonStyle::NEUTRAL)).clicked() { self.license_view = Some(false); }
                            if ui.add(Button::new("Schriften und Icons").size(ButtonSize::Compact).style(ButtonStyle::NEUTRAL)).clicked() { self.license_view = Some(true); }
                        });
                    });
                });
                egui::Frame::new().inner_margin(egui::Margin { left: 0, right: 0, top: 8, bottom: 20 }).show(ui, |ui| {
                    ui.set_width(width);
                    super::centered::centered_row(ui, |ui| {
                        if ui.add(Button::new("SCHLIESSEN").style(ButtonStyle::NEUTRAL).size(ButtonSize::Large)).clicked() { self.show_about = false; }
                    });
                });
            });
            if modal.close_requested && self.license_view.is_none() {
                self.show_about = false;
            }
        }
        if let Some(fonts) = self.license_view {
            let modal = Modal::new("licenses-smude")
                .title(if fonts {
                    "Schriften und Icons"
                } else {
                    "Projektlizenz"
                })
                .closable(true)
                .width(640.)
                .scrollable(true)
                .show(ctx, |ui| {
                    let text = if fonts {
                        concat!(
                            "Inter\n",
                            include_str!("../../assets/fonts/Inter-OFL.txt"),
                            "\n\nParkinsans\n",
                            include_str!("../../assets/fonts/Parkinsans-OFL.txt"),
                            "\n\nJetBrains Mono\n",
                            include_str!("../../assets/fonts/JetBrainsMono-OFL.txt"),
                            "\n\nLucide\n",
                            include_str!("../../assets/fonts/lucide-LICENSE.txt")
                        )
                    } else {
                        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../LICENSE.md"))
                    };
                    ui.label(role::SMALL.rich(text).color(color::TEXT_MID));
                });
            if modal.close_requested {
                self.license_view = None;
            }
        }
        if self.confirm_quit {
            let (quit, stay) = Modal::new("confirm-quit")
                .width(480.)
                .show(ctx, |ui| {
                    ui.spacing_mut().item_spacing.y = 24.;
                    super::centered::icon(ui, theme::Icon::CircleAlert, 56., color::ACCENT);
                    super::centered::Paragraph::new()
                        .run(
                            "Smude schließen?",
                            Type::new(Face::HeadingBold, 20.)
                                .tracking(0.025)
                                .leading(28.),
                            color::ACCENT,
                        )
                        .show(ui);
                    super::centered::Paragraph::new()
                        .run(
                            "Nicht exportierte Ergebnisse und die Warteschlange werden verworfen.",
                            role::BODY.leading(20.),
                            color::TEXT_MID,
                        )
                        .show(ui);
                    ui.add_space(8.);
                    super::centered::centered_row(ui, |ui| {
                        let quit = ui
                            .add(
                                Button::new("SCHLIESSEN")
                                    .style(ButtonStyle::NEUTRAL)
                                    .size(ButtonSize::Large),
                            )
                            .clicked();
                        let stay = ui
                            .add(
                                Button::new("ABBRECHEN")
                                    .style(ButtonStyle::PRIMARY)
                                    .size(ButtonSize::Large),
                            )
                            .clicked();
                        (quit, stay)
                    })
                })
                .inner;
            if quit {
                self.allow_quit = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            } else if stay
                || ctx
                    .input(|i| i.key_pressed(egui::Key::Escape) || i.key_pressed(egui::Key::Enter))
            {
                self.confirm_quit = false;
            }
        }
    }
}
