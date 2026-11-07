//! Corresponding music coordinates, rather than equal pixel offsets.
use super::*;
use smude_geometry::{Point, application::PreviewMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Side {
    Original,
    Result,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Navigation {
    pub point: Point,
    pub position: Vec2,
    pub scale: f32,
}
impl Canvas {
    pub(super) fn navigation_at(&self, dimensions: [u32; 2], position: Vec2) -> Option<Navigation> {
        let rect = self.viewport?;
        let scale = self.scale.unwrap_or(self.last_scale);
        if !scale.is_finite() || scale <= 0. {
            return None;
        }
        let relative = (position - Vec2::splat(0.5)) * rect.size();
        let point = egui::vec2(dimensions[0] as f32, dimensions[1] as f32) / 2.
            + (relative - self.pan) / scale
            - Vec2::splat(0.5);
        Some(Navigation {
            point: Point {
                x: f64::from(point.x),
                y: f64::from(point.y),
            },
            position,
            scale,
        })
    }
    pub(super) fn place(&mut self, dimensions: [u32; 2], anchor: Navigation) {
        self.anchor = Some(anchor);
        self.scale = Some(anchor.scale);
        self.last_scale = anchor.scale;
        if let Some(rect) = self.viewport {
            self.pan = (anchor.position - Vec2::splat(0.5)) * rect.size()
                - (egui::vec2(anchor.point.x as f32, anchor.point.y as f32) + Vec2::splat(0.5)
                    - egui::vec2(dimensions[0] as f32, dimensions[1] as f32) / 2.)
                    * anchor.scale;
        }
    }
}
fn corresponding(map: &PreviewMap, side: Side, navigation: Navigation) -> Option<Navigation> {
    let source = match side {
        Side::Original => navigation.point,
        Side::Result => map.result_to_source(navigation.point)?,
    };
    if !map.contains_source(source) {
        return None;
    }
    let ratio = map.local_scale_at_source(source)? as f32;
    let (point, scale) = match side {
        Side::Original => (map.source_to_result(source)?, navigation.scale / ratio),
        Side::Result => (source, navigation.scale * ratio),
    };
    (scale.is_finite() && (0.0001..10000.).contains(&scale)).then_some(Navigation {
        point,
        scale,
        position: navigation.position,
    })
}
impl SmudeApp {
    pub(super) fn navigate(&mut self, side: Side, navigation: Navigation, ctx: &egui::Context) {
        if let Some(id) = self.selected {
            self.pending_navigation = Some((id, side, navigation));
        }
        self.flush_navigation(ctx);
    }
    pub(super) fn flush_navigation(&mut self, ctx: &egui::Context) {
        let Some((id, side, navigation)) = self.pending_navigation else {
            return;
        };
        if self.selected != Some(id) {
            self.pending_navigation = None;
            return;
        }
        let Some(map) = self
            .result()
            .and_then(|r| r.pages.get(self.selected_page))
            .and_then(|p| p.mapping.clone())
        else {
            return;
        };
        let Some(dimensions) = (match side {
            Side::Original => self.right.as_ref(),
            Side::Result => self.left.as_ref(),
        })
        .map(|t| t.dimensions) else {
            return;
        };
        if let Some(anchor) = corresponding(&map, side, navigation) {
            match side {
                Side::Original => self.right_canvas.place(dimensions, anchor),
                Side::Result => self.left_canvas.place(dimensions, anchor),
            }
            ctx.request_repaint();
        }
        self.pending_navigation = None;
    }
    pub(super) fn zoom_control(&mut self, factor: Option<f32>, ctx: &egui::Context) {
        let side = if self.view == View::Result {
            Side::Result
        } else {
            Side::Original
        };
        let Some(dimensions) = (match side {
            Side::Original => self.left.as_ref(),
            Side::Result => self.right.as_ref(),
        })
        .map(|t| t.dimensions) else {
            return;
        };
        let canvas = match side {
            Side::Original => &mut self.left_canvas,
            Side::Result => &mut self.right_canvas,
        };
        let Some(mut anchor) = canvas.navigation_at(dimensions, Vec2::splat(0.5)) else {
            return;
        };
        anchor.scale = factor.map_or(1., |factor| (anchor.scale * factor).clamp(0.04, 8.));
        canvas.place(dimensions, anchor);
        self.navigate(side, anchor, ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smude_geometry::application::{OutputKind, Settings, process_file};

    fn frame(ctx: &egui::Context, app: &mut SmudeApp, time: f64, events: Vec<egui::Event>) {
        let mut output = ctx.run_ui(
            egui::RawInput {
                time: Some(time),
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1100., 800.),
                )),
                events,
                ..Default::default()
            },
            |ui| app.interface(ui),
        );
        output.textures_delta.clear();
    }
    #[test]
    fn wheel_zoom_in_either_pane_keeps_the_same_music_under_the_pointer() {
        let temporary = tempfile::tempdir().unwrap();
        let mut image = image::RgbImage::from_pixel(600, 800, image::Rgb([255; 3]));
        for center in [220_i32, 500] {
            for k in -2..=2 {
                for x in 60..560 {
                    let bend = (0.00012 * (x as f64 - 300.).powi(2)).round() as i32;
                    image.put_pixel(x, (center + k * 12 + bend) as u32, image::Rgb([0; 3]));
                }
            }
            for x in [80, 220, 400, 540] {
                let bend = (0.00012 * (x as f64 - 300.).powi(2)).round() as i32;
                for y in center - 24 + bend..=center + 24 + bend {
                    image.put_pixel(x, y as u32, image::Rgb([0; 3]));
                }
            }
        }
        let input = temporary.path().join("score.png");
        image.save(&input).unwrap();
        let result = process_file(
            &input,
            &temporary.path().join("out"),
            &Settings {
                detect_pages: false,
                text_correction: false,
                ..Default::default()
            },
        )
        .unwrap();
        let map = result.pages[0].mapping.clone().unwrap();
        let (width, height) =
            image::image_dimensions(result.pages[0].image(OutputKind::Clean)).unwrap();
        let ctx = egui::Context::default();
        let mut app = SmudeApp::from_context(&ctx, Preferences::default(), vec![]).unwrap();
        let texture = ctx.load_texture(
            "navigation-test",
            egui::ColorImage::filled([2, 2], Color32::WHITE),
            egui::TextureOptions::LINEAR,
        );
        app.left = Some(Texture {
            handle: texture.clone(),
            dimensions: [600, 800],
        });
        app.right = Some(Texture {
            handle: texture,
            dimensions: [width, height],
        });
        app.jobs.push(Job {
            id: 1,
            path: input,
            state: State::Ready {
                result,
                seconds: 1.,
            },
            overlays: vec![],
        });
        app.selected = Some(1);
        frame(&ctx, &mut app, 0.1, vec![]);
        frame(&ctx, &mut app, 0.2, vec![]);
        for (step, side, point, position, scale) in [
            (
                1.,
                Side::Original,
                Point { x: 300., y: 220. },
                egui::vec2(0.35, 0.65),
                2.,
            ),
            (
                2.,
                Side::Result,
                Point { x: 400., y: 500. },
                egui::vec2(0.73, 0.27),
                1.3,
            ),
        ] {
            frame(&ctx, &mut app, step - 0.1, vec![egui::Event::PointerGone]);
            let origin_point = if side == Side::Original {
                point
            } else {
                map.source_to_result(point).unwrap()
            };
            let anchor = Navigation {
                point: origin_point,
                position,
                scale,
            };
            let canvas = match side {
                Side::Original => &mut app.left_canvas,
                Side::Result => &mut app.right_canvas,
            };
            let dimensions = if side == Side::Original {
                [600, 800]
            } else {
                [width, height]
            };
            canvas.place(dimensions, anchor);
            app.navigate(side, anchor, &ctx);
            frame(&ctx, &mut app, step, vec![]);
            let canvas = match side {
                Side::Original => &app.left_canvas,
                Side::Result => &app.right_canvas,
            };
            let rect = canvas.viewport.unwrap();
            let cursor = rect.min + position * rect.size();
            frame(
                &ctx,
                &mut app,
                step + 0.1,
                vec![
                    egui::Event::PointerMoved(cursor),
                    egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Point,
                        phase: egui::TouchPhase::Move,
                        delta: egui::vec2(0., 90.),
                        modifiers: Default::default(),
                    },
                ],
            );
            let left = app.left_canvas.navigation_at([600, 800], position).unwrap();
            let right = app
                .right_canvas
                .navigation_at([width, height], position)
                .unwrap();
            let mapped = map.source_to_result(left.point).unwrap();
            assert!(
                (left.point.x - point.x).hypot(left.point.y - point.y) < 0.002,
                "{side:?}: expected {point:?}, got source {left:?}, result {right:?}"
            );
            assert!((right.point.x - mapped.x).hypot(right.point.y - mapped.y) < 0.002);
            assert!(
                (f64::from(left.scale / right.scale)
                    - map.local_scale_at_source(left.point).unwrap())
                .abs()
                    < 0.001
            );
            let zoomed = if side == Side::Original {
                left.scale
            } else {
                right.scale
            };
            assert!(zoomed > scale, "wheel did not zoom the hovered pane");
            frame(
                &ctx,
                &mut app,
                step + 0.15,
                vec![egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0., -180.),
                    phase: egui::TouchPhase::Move,
                    modifiers: Default::default(),
                }],
            );
            let left_out = app.left_canvas.navigation_at([600, 800], position).unwrap();
            let right_out = app
                .right_canvas
                .navigation_at([width, height], position)
                .unwrap();
            let mapped_out = map.source_to_result(left_out.point).unwrap();
            assert!((left_out.point.x - point.x).hypot(left_out.point.y - point.y) < 0.002);
            assert!(
                (right_out.point.x - mapped_out.x).hypot(right_out.point.y - mapped_out.y) < 0.002
            );
            assert!(
                (if side == Side::Original {
                    left_out.scale
                } else {
                    right_out.scale
                }) < zoomed,
                "wheel did not zoom out"
            );
            frame(
                &ctx,
                &mut app,
                step + 0.2,
                vec![egui::Event::PointerButton {
                    pos: cursor,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: Default::default(),
                }],
            );
            let moved = cursor + egui::vec2(30., 18.);
            frame(
                &ctx,
                &mut app,
                step + 0.3,
                vec![egui::Event::PointerMoved(moved)],
            );
            let rect = match side {
                Side::Original => app.left_canvas.viewport.unwrap(),
                Side::Result => app.right_canvas.viewport.unwrap(),
            };
            let position = (moved - rect.min) / rect.size();
            let left = app.left_canvas.navigation_at([600, 800], position).unwrap();
            let right = app
                .right_canvas
                .navigation_at([width, height], position)
                .unwrap();
            let mapped = map.source_to_result(left.point).unwrap();
            assert!(
                (right.point.x - mapped.x).hypot(right.point.y - mapped.y) < 0.003,
                "pan lost its correspondence"
            );
            frame(
                &ctx,
                &mut app,
                step + 0.4,
                vec![egui::Event::PointerButton {
                    pos: moved,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Default::default(),
                }],
            );
        }
    }
}
