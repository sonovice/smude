use super::*;

pub(super) struct Decorations<'a> {
    pub overlays: &'a [Overlay],
    pub export: bool,
    pub scan: Option<f32>,
}

#[derive(Default)]
pub(super) struct PanelResponse {
    pub export_clicked: bool,
    pub navigation: Option<super::navigation::Navigation>,
    pub reset: bool,
}

pub(super) fn image_panel(
    ui: &mut egui::Ui,
    title: &str,
    texture: Option<&Texture>,
    canvas: &mut Canvas,
    error: Option<&str>,
    height: f32,
    decorations: Decorations<'_>,
) -> PanelResponse {
    let mut output = PanelResponse::default();
    let (outer, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), height),
        egui::Sense::hover(),
    );
    let header_rect = egui::Rect::from_min_size(outer.min, egui::vec2(outer.width(), 32.));
    let rect = egui::Rect::from_min_max(egui::pos2(outer.left(), outer.top() + 40.), outer.max);
    let mut header = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(header_rect)
            .id_salt(("image-header", title))
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    );
    header.set_clip_rect(header_rect.intersect(ui.clip_rect()));
    crate::widgets::centered_row(&mut header, |ui| {
        theme::section(ui, title);
        if let Some(t) = texture {
            ui.label(
                RichText::new(format!("{} × {}", t.dimensions[0], t.dimensions[1]))
                    .small()
                    .color(theme::MUTED),
            );
        }
        if decorations.export {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                output.export_clicked = ui
                    .add(
                        crate::widgets::IconButton::new(theme::Icon::HardDriveDownload)
                            .label("Diese Seite exportieren")
                            .size(24., 14.),
                    )
                    .on_hover_text("Diese Seite exportieren")
                    .clicked();
            });
        }
    });
    let response = ui.interact(
        rect,
        ui.id().with(("image-canvas", title)),
        egui::Sense::drag(),
    );
    canvas.viewport = Some(rect);
    let painter = ui.painter_at(rect);
    painter.add(theme::backdrop::diagonal_gradient(
        rect,
        10,
        theme::RAISED,
        theme::PANEL,
    ));
    painter.rect_stroke(
        rect,
        10.,
        egui::Stroke::new(1., theme::BORDER),
        egui::StrokeKind::Inside,
    );
    let Some(texture) = texture else {
        let galley = theme::role::BODY.layout_wrapped(
            ui,
            error.unwrap_or("Vorschau wird bereitgestellt …"),
            theme::MUTED,
            (rect.width() - 48.).max(1.),
        );
        painter.galley(rect.center() - galley.size() / 2., galley, theme::MUTED);
        return output;
    };
    let dimensions = egui::vec2(texture.dimensions[0] as f32, texture.dimensions[1] as f32);
    let fit = ((rect.width() - 24.) / dimensions.x)
        .min((rect.height() - 24.) / dimensions.y)
        .max(0.01);
    if let Some(anchor) = canvas.anchor {
        canvas.place(texture.dimensions, anchor);
    }
    let mut scale = canvas.scale.unwrap_or(fit);
    let mut interacted = false;
    if response.hovered() {
        let zoom = ui.input(|i| i.zoom_delta());
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        let factor = if zoom != 1. {
            zoom
        } else {
            (scroll * 0.002).exp()
        };
        if factor != 1. {
            interacted = true;
            let new_scale = (scale * factor).clamp(0.04, 8.);
            if let Some(pointer) = response.hover_pos() {
                let relative = pointer - rect.center();
                canvas.pan = relative - (relative - canvas.pan) * (new_scale / scale);
            }
            scale = new_scale;
            canvas.scale = Some(scale);
        }
    }
    if response.dragged() {
        interacted = true;
        canvas.pan += response.drag_delta();
    }
    if response.double_clicked() {
        output.reset = true;
        canvas.scale = None;
        canvas.pan = Vec2::ZERO;
        scale = fit;
    }
    canvas.last_scale = scale;
    if interacted {
        canvas.anchor = None;
        if let Some(pointer) = response
            .interact_pointer_pos()
            .or_else(|| response.hover_pos())
        {
            output.navigation =
                canvas.navigation_at(texture.dimensions, (pointer - rect.min) / rect.size());
        }
    }
    if output.reset {
        canvas.anchor = None;
        output.navigation = None;
    }
    let image_rect = egui::Rect::from_center_size(rect.center() + canvas.pan, dimensions * scale);
    painter.image(
        texture.handle.id(),
        image_rect,
        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1., 1.)),
        Color32::WHITE,
    );
    let overlay_painter = painter.with_clip_rect(rect.intersect(image_rect));
    super::shading::paint(
        ui,
        &overlay_painter,
        image_rect,
        texture.dimensions,
        decorations.overlays,
    );
    if let Some(seconds) = decorations.scan {
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(16));
        let phase = (seconds / 1.8).fract();
        let y = image_rect.top() + phase * image_rect.height();
        for (offset, alpha) in [(0., 150), (-6., 55), (-12., 20)] {
            overlay_painter.hline(
                image_rect.x_range(),
                y + offset,
                egui::Stroke::new(1., theme::GOLD.gamma_multiply(alpha as f32 / 255.)),
            );
        }
    }
    for overlay in decorations.overlays {
        let event = &overlay.data;
        let elapsed = overlay.arrived.elapsed().as_secs_f32();
        if elapsed < 0.85 {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(16));
        }
        let color = match event.stage {
            DetectionStage::Page => theme::GOLD,
            DetectionStage::Staves => theme::GOLD.gamma_multiply(0.65),
            DetectionStage::Barlines => theme::color::ACCENT_DIM,
            DetectionStage::StaffEdges => theme::color::TEXT_MID,
            DetectionStage::Text => theme::color::TEXT_MID.gamma_multiply(0.7),
            DetectionStage::ResetMusic => continue,
        };
        for (index, path) in event.paths.iter().enumerate() {
            let fraction = ((elapsed - (index as f32 * 0.012).min(0.18)) / 0.6).clamp(0., 1.);
            let fraction = fraction * fraction * (3. - 2. * fraction);
            let points = revealed_path(path, fraction);
            let points: Vec<_> = points
                .iter()
                .map(|p| image_rect.min + egui::vec2(p.x as f32, p.y as f32) * scale)
                .collect();
            if points.len() > 1 {
                overlay_painter.add(egui::Shape::line(points, egui::Stroke::new(1.4, color)));
            }
        }
    }
    output
}

/// Reveal by arc length so sparse and dense detector paths animate at equal speed.
fn revealed_path(path: &[smude_geometry::Point], fraction: f32) -> Vec<smude_geometry::Point> {
    use smude_geometry::Point;
    if path.len() < 2 || fraction <= 0. || path.iter().any(|p| !p.x.is_finite() || !p.y.is_finite())
    {
        return vec![];
    }
    if fraction >= 1. {
        return path.to_vec();
    }
    let length = |a: Point, b: Point| (a.x - b.x).hypot(a.y - b.y);
    let mut remaining =
        path.windows(2).map(|p| length(p[0], p[1])).sum::<f64>() * f64::from(fraction);
    let mut result = vec![path[0]];
    for pair in path.windows(2) {
        let segment = length(pair[0], pair[1]);
        if segment > remaining {
            let t = remaining / segment;
            result.push(Point {
                x: pair[0].x + t * (pair[1].x - pair[0].x),
                y: pair[0].y + t * (pair[1].y - pair[0].y),
            });
            break;
        }
        result.push(pair[1]);
        remaining -= segment;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use smude_geometry::Point;
    #[test]
    fn comparison_canvases_stay_identical_before_and_after_result_arrives() {
        let ctx = egui::Context::default();
        theme::install(&ctx);
        let texture = Texture {
            handle: ctx.load_texture(
                "test",
                egui::ColorImage::filled([2, 2], Color32::WHITE),
                egui::TextureOptions::LINEAR,
            ),
            dimensions: [1200, 1600],
        };
        let mut left_canvas = Canvas::default();
        let mut right_canvas = Canvas::default();
        let mut layouts = vec![];
        for ready in [false, false, true, true] {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1000., 700.),
                    )),
                    ..Default::default()
                },
                |ui| {
                    egui::CentralPanel::default().show(ui, |ui| {
                        ui.columns(2, |cols| {
                            image_panel(
                                &mut cols[0],
                                "Original",
                                Some(&texture),
                                &mut left_canvas,
                                None,
                                550.,
                                Decorations {
                                    overlays: &[],
                                    export: false,
                                    scan: None,
                                },
                            );
                            image_panel(
                                &mut cols[1],
                                "Ergebnis",
                                ready.then_some(&texture),
                                &mut right_canvas,
                                None,
                                550.,
                                Decorations {
                                    overlays: &[],
                                    export: ready,
                                    scan: None,
                                },
                            );
                        });
                    });
                },
            );
            output.textures_delta.clear();
            let rects: Vec<_> = output
                .shapes
                .iter()
                .filter_map(|shape| match &shape.shape {
                    egui::Shape::Rect(r)
                        if r.stroke.color == theme::BORDER
                            && r.rect.width() > 100.
                            && r.rect.height() > 100. =>
                    {
                        Some(r.rect)
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(rects.len(), 2);
            assert_eq!(rects[0].size(), rects[1].size());
            assert_eq!(rects[0].top(), rects[1].top());
            layouts.push(rects);
        }
        assert_eq!(layouts[1], layouts[3]);
    }

    #[test]
    fn reveal_interpolates_by_length_and_retains_complete_curve() {
        let path = [
            Point { x: 0., y: 0. },
            Point { x: 10., y: 0. },
            Point { x: 10., y: 30. },
        ];
        assert!(revealed_path(&path, 0.).is_empty());
        let half = revealed_path(&path, 0.5);
        assert_eq!(half.len(), 3);
        assert_eq!((half[2].x, half[2].y), (10., 10.));
        let full = revealed_path(&path, 1.);
        assert_eq!(full.len(), 3);
        assert_eq!((full[2].x, full[2].y), (10., 30.));
    }
}
