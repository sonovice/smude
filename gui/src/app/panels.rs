use super::canvas::image_panel;
use super::*;
use crate::widgets::{Button, ButtonSize, Segment, Segmented, Text};

impl SmudeApp {
    pub(super) fn header(&mut self, ui: &mut egui::Ui) {
        let (rect, _) =
            ui.allocate_exact_size(egui::vec2(ui.available_width(), 64.), egui::Sense::hover());
        ui.painter().rect_filled(rect, 0, theme::PANEL);
        theme::backdrop::Glow::HEADER.paint(ui.painter(), rect);
        ui.painter().hline(
            rect.x_range(),
            rect.bottom() - 0.5,
            theme::border(theme::BORDER),
        );
        let mut ui =
            ui.new_child(egui::UiBuilder::new().max_rect(rect.shrink2(egui::vec2(16., 12.))));
        crate::widgets::centered_row(&mut ui, |ui| {
            ui.spacing_mut().item_spacing.x = 12.;
            super::dialogs::logo(ui, 40.);
            ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                ui.spacing_mut().item_spacing.y = 0.;
                ui.add(
                    Text::new(
                        "Notenseiten entzerren",
                        theme::role::SMALL.tracking(0.05).leading(15.),
                    )
                    .color(theme::MUTED),
                );
                ui.add(Text::new("SMUDE", theme::role::BRAND_COMPACT).color(theme::GOLD));
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let count = self.export_items(true).len();
                if ui
                    .add_enabled(
                        count > 0 && self.export_rx.is_none(),
                        Button::new(if self.export_rx.is_some() {
                            "Export läuft …"
                        } else {
                            "Alle exportieren"
                        })
                        .style(crate::widgets::ButtonStyle::ACCENT)
                        .size(ButtonSize::Toolbar)
                        .icon(theme::Icon::HardDriveDownload),
                    )
                    .clicked()
                {
                    self.start_export(true, ui.ctx());
                }
            });
        });
    }
    pub(super) fn sidebar(&mut self, ui: &mut egui::Ui) {
        crate::widgets::centered_row(ui, |ui| {
            theme::section(ui, "DATEIEN");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 4.;
                if ui
                    .add_enabled(
                        !self.importing,
                        crate::widgets::IconButton::new(theme::Icon::Folder)
                            .label("Ordner hinzufügen")
                            .size(28., 14.),
                    )
                    .on_hover_text("Ordner hinzufügen")
                    .clicked()
                {
                    self.choose_folder(ui.ctx());
                }
                if ui
                    .add_enabled(
                        !self.importing,
                        crate::widgets::IconButton::new(theme::Icon::FileText)
                            .label("Dateien hinzufügen")
                            .size(28., 14.),
                    )
                    .on_hover_text("Dateien hinzufügen")
                    .clicked()
                {
                    self.choose_files(ui.ctx());
                }
            });
        });
        ui.add_space(8.);
        let body = ui.available_rect_before_wrap();
        let controls_rect =
            egui::Rect::from_min_max(egui::pos2(body.left(), body.bottom() - 32.), body.max);
        let list_rect =
            egui::Rect::from_min_max(body.min, egui::pos2(body.right(), controls_rect.top() - 8.));
        ui.allocate_rect(body, egui::Sense::hover());
        let mut list = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(list_rect)
                .layout(egui::Layout::top_down(egui::Align::Min)),
        );
        list.set_clip_rect(list_rect.intersect(ui.clip_rect()));
        let mut select = None;
        egui::ScrollArea::vertical()
            .max_height(list_rect.height())
            .auto_shrink([false, false])
            .show(&mut list, |ui| {
                for job in &self.jobs {
                    let selected = Some(job.id) == self.selected;
                    let (status, color) = match &job.state {
                        State::Queued => ("Wartet".to_string(), theme::MUTED),
                        State::Running(start) => (
                            format!("In Arbeit · {:.0}s", start.elapsed().as_secs_f64()),
                            theme::GOLD,
                        ),
                        State::Ready { result, seconds } => (
                            format!(
                                "{} Seite{} · {seconds:.1} s",
                                result.pages.len(),
                                if result.pages.len() == 1 { "" } else { "n" }
                            ),
                            theme::GREEN,
                        ),
                        State::Failed(_) => ("Nicht verarbeitet".into(), theme::RED),
                    };
                    let row = crate::widgets::ListRow::new()
                        .selected(selected)
                        .divider(false)
                        .radius(8)
                        .padding(egui::Margin::symmetric(8, 5))
                        .show(ui, |ui, state| {
                            let (rect, _) = ui.allocate_exact_size(
                                egui::vec2(ui.available_width(), 18.),
                                egui::Sense::hover(),
                            );
                            let name = job.path.file_name().unwrap_or_default().to_string_lossy();
                            let text_color = if selected || state.hovered {
                                theme::GOLD
                            } else {
                                theme::color::TEXT_MID
                            };
                            let galley = theme::role::LIST.layout(
                                ui,
                                name.to_string(),
                                text_color,
                                (rect.width() - 22.).max(1.),
                            );
                            ui.painter().galley(
                                egui::pos2(rect.left(), rect.center().y - galley.size().y / 2.),
                                galley,
                                text_color,
                            );
                            let center = egui::pos2(rect.right() - 7., rect.center().y);
                            match &job.state {
                                State::Ready { .. } => {
                                    theme::Icon::Check.paint(ui.painter(), center, 14., color);
                                }
                                State::Failed(_) => {
                                    theme::Icon::TriangleAlert.paint(
                                        ui.painter(),
                                        center,
                                        14.,
                                        color,
                                    );
                                }
                                State::Running(_) => {
                                    ui.put(
                                        egui::Rect::from_center_size(
                                            center,
                                            egui::Vec2::splat(14.),
                                        ),
                                        egui::Spinner::new().size(14.).color(color),
                                    );
                                }
                                State::Queued => {}
                            }
                        });
                    let response = row
                        .response
                        .on_hover_text(format!("{}\n{status}", job.path.display()));
                    if response.clicked() {
                        select = Some(job.id);
                    }
                }
            });
        if let Some(id) = select {
            self.select(id);
        }
        let mut controls = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(controls_rect)
                .layout(egui::Layout::left_to_right(egui::Align::Center)),
        );
        controls.set_clip_rect(controls_rect.intersect(ui.clip_rect()));
        let ui = &mut controls;
        {
            let removable = !self.busy() && self.export_rx.is_none();

            if ui
                .add_enabled(
                    removable && self.selected.is_some(),
                    crate::widgets::IconButton::new(theme::Icon::X).size(24., 14.),
                )
                .on_hover_text("Ausgewählte Datei entfernen")
                .clicked()
            {
                self.jobs.retain(|j| Some(j.id) != self.selected);
                self.selected = self.jobs.first().map(|j| j.id);
                self.selected_page = 0;
                self.preview_key = None;
                self.left = None;
                self.right = None;
            }
            if ui
                .add_enabled(
                    removable && !self.jobs.is_empty(),
                    Button::new("Leeren").size(ButtonSize::Small),
                )
                .clicked()
            {
                self.jobs.clear();
                self.selected = None;
                self.preview_key = None;
                self.left = None;
                self.right = None;
            }
        }
    }
    pub(super) fn workspace(&mut self, ui: &mut egui::Ui) {
        if self.jobs.is_empty() {
            let outer = ui.available_rect_before_wrap();
            theme::backdrop::Glow::SPOTLIGHT.paint(ui.painter(), outer);
            let id = ui.id().with("empty-group-height");
            let height = ui.data(|d| d.get_temp::<f32>(id)).unwrap_or(240.);
            let rect = egui::Rect::from_center_size(
                outer.center(),
                egui::vec2(outer.width().min(540.), height),
            );
            let mut group = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(rect)
                    .layout(egui::Layout::top_down(egui::Align::Center)),
            );
            group.spacing_mut().item_spacing.y = 12.;
            group.add(Text::new(
                "Notenseiten hinzufügen",
                theme::role::DISPLAY.size(22.).tracking(0.01),
            ));
            super::centered::Paragraph::new().run("Dateien oder Ordner hierher ziehen.\nJPEG und PNG werden automatisch entzerrt.", theme::role::BODY.leading(21.), theme::color::TEXT_MID).show(&mut group);
            group.add_space(12.);
            if group
                .add(
                    Button::new("Dateien auswählen")
                        .style(crate::widgets::ButtonStyle::PRIMARY)
                        .icon(theme::Icon::FileText),
                )
                .clicked()
            {
                self.choose_files(ui.ctx());
            }
            if group
                .add(Button::new("Ordner auswählen").icon(theme::Icon::Folder))
                .clicked()
            {
                self.choose_folder(ui.ctx());
            }
            let measured = group.min_rect().height();
            if (measured - height).abs() > 0.5 {
                ui.data_mut(|d| d.insert_temp(id, measured));
                ui.ctx().request_discard("center the complete start group");
            }
            ui.allocate_rect(outer, egui::Sense::hover());
            return;
        }
        ui.horizontal_wrapped(|ui| {
            ui.scope(|ui| {
                ui.set_width(300.);
                ui.add(Segmented::new(
                    &mut self.view,
                    [
                        Segment::new(View::Compare, "Vergleich"),
                        Segment::new(View::Result, "Ergebnis"),
                        Segment::new(View::Original, "Original"),
                    ],
                ));
            });
            if ui.add(Button::new("Erkennungen").size(ButtonSize::Compact).selected(self.show_detections))
                .on_hover_text("Original-Overlays: Gold = Seite und Notenlinien, dunkles Gold = Taktstriche, neutrale Grautöne = Systemgrenzen und Text. Erscheinen nach tatsächlichem Erkennungsfortschritt.")
                .clicked() { self.show_detections = !self.show_detections; }
            let count = self.result().map_or(0, |r| r.pages.len());
            ui.add_enabled_ui(count > 1, |ui| {
                egui::ComboBox::from_id_salt("selected-page")
                    .width(76.)
                    .selected_text(format!("Seite {}", self.selected_page + 1))
                    .show_ui(ui, |ui| {
                        for i in 0..count { ui.selectable_value(&mut self.selected_page, i, format!("Seite {}", i + 1)); }
                    });
            });
            ui.separator();
            if ui
                .add(Button::new("Einpassen").size(ButtonSize::Compact))
                .clicked()
            {
                self.left_canvas = Canvas::default();
                self.right_canvas = Canvas::default();
                self.pending_navigation = None;
            }
            if ui
                .add(Button::new("100 %").size(ButtonSize::Compact))
                .clicked()
            {
                self.zoom_control(None,ui.ctx());
            }
            for (text, factor) in [("−", 0.8), ("+", 1.25)] {
                if ui
                    .add(Button::new(text).size(ButtonSize::Compact))
                    .clicked()
                {
                    self.zoom_control(Some(factor),ui.ctx());
                }
            }
        });
        ui.add_space(8.);
        let height = ui.available_height().max(100.);
        let no_result = match self.selected_job().map(|job| &job.state) {
            Some(State::Failed(error)) => Some(friendly_error(error)),
            Some(State::Queued) => Some("Ergebnis erscheint nach der Verarbeitung."),
            Some(State::Running(_)) => Some("Notenseite wird entzerrt …"),
            _ => None,
        }
        .map(str::to_owned);
        let overlays = if self.show_detections {
            self.selected_job()
                .map(|j| j.overlays.clone())
                .unwrap_or_default()
        } else {
            vec![]
        };
        let scan = self
            .selected_job()
            .and_then(|job| match &job.state {
                State::Running(start)
                    if !job
                        .overlays
                        .iter()
                        .any(|o| o.data.stage == DetectionStage::Staves) =>
                {
                    Some(start.elapsed().as_secs_f32())
                }
                _ => None,
            })
            .filter(|_| self.show_detections);
        let can_export = !self.export_items(false).is_empty() && self.export_rx.is_none();
        let mut export_clicked = false;
        let navigation_ctx = ui.ctx().clone();
        let mut handle =
            |app: &mut Self, side, output: super::canvas::PanelResponse, ctx: &egui::Context| {
                export_clicked |= output.export_clicked;
                if output.reset {
                    app.left_canvas = Canvas::default();
                    app.right_canvas = Canvas::default();
                    app.pending_navigation = None;
                    ctx.request_repaint();
                } else if let Some(nav) = output.navigation {
                    app.navigate(side, nav, ctx);
                }
            };
        match self.view {
            View::Compare => {
                ui.columns(2, |cols| {
                    let output = image_panel(
                        &mut cols[0],
                        "Original",
                        self.left.as_ref(),
                        &mut self.left_canvas,
                        self.left_error.as_deref(),
                        height,
                        super::canvas::Decorations {
                            overlays: &overlays,
                            export: false,
                            scan,
                        },
                    );
                    handle(
                        self,
                        super::navigation::Side::Original,
                        output,
                        &navigation_ctx,
                    );
                    let output = image_panel(
                        &mut cols[1],
                        "Ergebnis",
                        self.right.as_ref(),
                        &mut self.right_canvas,
                        self.right_error.as_deref().or(no_result.as_deref()),
                        height,
                        super::canvas::Decorations {
                            overlays: &[],
                            export: can_export,
                            scan: None,
                        },
                    );
                    handle(
                        self,
                        super::navigation::Side::Result,
                        output,
                        &navigation_ctx,
                    );
                });
            }
            View::Result => {
                let output = image_panel(
                    ui,
                    "Ergebnis",
                    self.right.as_ref(),
                    &mut self.right_canvas,
                    self.right_error.as_deref().or(no_result.as_deref()),
                    height,
                    super::canvas::Decorations {
                        overlays: &[],
                        export: can_export,
                        scan: None,
                    },
                );
                handle(
                    self,
                    super::navigation::Side::Result,
                    output,
                    &navigation_ctx,
                );
            }
            View::Original => {
                let output = image_panel(
                    ui,
                    "Original",
                    self.left.as_ref(),
                    &mut self.left_canvas,
                    self.left_error.as_deref(),
                    height,
                    super::canvas::Decorations {
                        overlays: &overlays,
                        export: false,
                        scan,
                    },
                );
                handle(
                    self,
                    super::navigation::Side::Original,
                    output,
                    &navigation_ctx,
                );
            }
        }
        self.flush_navigation(ui.ctx());
        if export_clicked {
            self.start_export(false, ui.ctx());
        }
    }
    pub(super) fn footer(&mut self, ui: &mut egui::Ui) {
        let done = self
            .jobs
            .iter()
            .filter(|j| matches!(j.state, State::Ready { .. } | State::Failed(_)))
            .count();
        let label = if self.importing {
            "Ordner wird eingelesen …".into()
        } else if self.jobs.is_empty() {
            "JPEG / PNG · Lokal auf deinem Computer".into()
        } else {
            format!("{done} von {} Dateien abgeschlossen", self.jobs.len())
        };
        let mut pause = false;
        let mut about = false;
        let mut folder = false;
        crate::widgets::StatusBar::new(crate::widgets::StatusTone::Idle).show(
            ui,
            |ui| {
                ui.add(Text::new(label, theme::role::SMALL).color(theme::MUTED));
            },
            |_| {},
            |ui| {
                about = ui
                    .add(
                        crate::widgets::IconButton::new(theme::Icon::CircleQuestionMark)
                            .label("Über Smude")
                            .style(crate::widgets::ButtonStyle::GHOST)
                            .size(24., 14.),
                    )
                    .on_hover_text("Über Smude")
                    .clicked();
                if self.last_export.is_some() {
                    folder = ui
                        .add(
                            crate::widgets::IconButton::new(theme::Icon::FolderOpen)
                                .style(crate::widgets::ButtonStyle::GHOST)
                                .size(24., 14.),
                        )
                        .on_hover_text("Exportordner öffnen")
                        .clicked();
                }
                if self
                    .jobs
                    .iter()
                    .any(|j| matches!(j.state, State::Queued | State::Running(_)))
                {
                    pause = ui
                        .add(
                            Button::new(if self.paused {
                                "Fortsetzen"
                            } else {
                                "Pausieren"
                            })
                            .size(ButtonSize::Small),
                        )
                        .clicked();
                }
            },
        );
        if about {
            self.show_about = true;
        }
        if pause {
            self.paused = !self.paused;
        }
        if folder
            && let Some(path) = &self.last_export
            && let Err(e) = open::that(path)
        {
            self.notice = Some(e.to_string());
        }
    }
}

fn friendly_error(error: &str) -> &str {
    if error.contains("no page produced") || error.contains("no_confident_regions") {
        "Keine vollständige Notenseite erkannt. Bitte eine andere Aufnahme verwenden."
    } else if error.contains("no reliable five-line") {
        "Keine zuverlässigen Notenzeilen gefunden. Bitte Bildschärfe und Ausrichtung prüfen."
    } else {
        error
    }
}
