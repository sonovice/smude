//! Low-resolution page cutouts fly into a bounded, tappable session stack.
use super::{
    aspect_fill,
    snapshot::{self, SessionFiles, Snapshot},
};
use crate::state::State;
use nsh_core::prelude::*;
use std::{sync::Arc, time::Duration};

#[derive(Clone)]
struct Face {
    order: u64,
    page: usize,
    shot: Arc<Snapshot>,
}
impl PartialEq for Face {
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order && self.page == other.page
    }
}
#[derive(Clone, PartialEq)]
struct Flight {
    face: Face,
    source: [f32; 4],
    target: [f32; 4],
}
#[derive(Clone, Copy)]
pub struct CaptureStack {
    faces: Signal<Vec<Face>>,
    flights: Signal<Vec<Flight>>,
    sequence: Signal<u64>,
    files: Signal<Option<Arc<SessionFiles>>>,
    pub scene: ElementRef,
    pub preview: ElementRef,
    target: ElementRef,
}
impl CaptureStack {
    pub fn new() -> Self {
        Self {
            faces: signal(vec![]),
            flights: signal(vec![]),
            sequence: signal(0),
            files: signal(None),
            scene: element_ref(),
            preview: element_ref(),
            target: element_ref(),
        }
    }
    pub fn capture(self, s: State, token: u64) {
        let Some(frame) = snapshot::get(token) else {
            return;
        };
        let Some(repo) = s.repo() else { return };
        let session = self.files.get_untracked().unwrap_or_else(|| {
            let files = Arc::new(SessionFiles::new(&repo.root));
            self.files.set(Some(files.clone()));
            files
        });
        self.sequence.update(|n| *n += 1);
        let order = self.sequence.get_untracked();
        spawn(async move {
            // Freeze native geometry at acceptance, before encoding or later detections.
            let scene = self.scene.measure().await;
            let preview = self.preview.measure().await;
            let target = self.target.measure().await;
            let Ok(Ok(shot)) = blocking(move || snapshot::write(frame, session, order)).await
            else {
                return;
            };
            for page in 0..shot.frame.polygons.len() {
                let face = Face {
                    order,
                    page,
                    shot: shot.clone(),
                };
                let geometry =
                    scene
                        .zip(preview)
                        .zip(target)
                        .and_then(|((scene, preview), target)| {
                            let b = snapshot::bounds(&shot.frame.polygons[page])?;
                            let dims = [
                                shot.frame.image.width() as f32,
                                shot.frame.image.height() as f32,
                            ];
                            let view = [preview.frame.size.width.0, preview.frame.size.height.0];
                            if view.iter().any(|v| *v <= 0.) {
                                return None;
                            }
                            let a = aspect_fill([b[0], b[1]], dims, view);
                            let z = aspect_fill([b[0] + b[2], b[1] + b[3]], dims, view);
                            Some((
                                [
                                    preview.page.x.0 - scene.page.x.0 + a[0],
                                    preview.page.y.0 - scene.page.y.0 + a[1],
                                    z[0] - a[0],
                                    z[1] - a[1],
                                ],
                                [
                                    target.page.x.0 - scene.page.x.0 + 15.,
                                    target.page.y.0 - scene.page.y.0 + 7.,
                                    42.,
                                    58.,
                                ],
                            ))
                        });
                if let Some((source, target)) = geometry.filter(|_| !reduce_motion()) {
                    self.flights.update(|v| {
                        v.push(Flight {
                            face: face.clone(),
                            source,
                            target,
                        });
                        if v.len() > 6 {
                            v.remove(0);
                        }
                    });
                    timer(Duration::from_millis(430)).await;
                }
                self.faces.update(|v| insert_face(v, face.clone()));
                self.flights.update(|v| v.retain(|f| f.face != face));
                if page == 0 && shot.frame.polygons.len() > 1 {
                    timer(Duration::from_millis(45)).await;
                }
            }
        });
    }
    pub fn button(self, s: State, id: u64) -> View {
        view()
            .test_id("capture-stack")
            .element_ref(&self.target)
            .w(px(76.))
            .h(px(72.))
            .relative()
            .role(AccessibilityRole::Button)
            .label(move || crate::i18n::review_pages(s.project(id).pages.len()))
            .disabled(move || s.project(id).pages.is_empty())
            .on_press(move || s.review_camera(id))
            .active(|v| v.opacity(0.6))
            .child(each(
                move || {
                    let faces = self.faces.get();
                    let count = s.project(id).pages.len().clamp(1, 3);
                    (0..count)
                        .map(|i| (i, faces.iter().rev().nth(count - 1 - i).cloned()))
                        .collect::<Vec<_>>()
                },
                |row| row.0,
                move |row, _| {
                    let rotation = animated_value(0.);
                    effect(move || {
                        rotation.set(
                            (row.get().0 as f32 + 1.
                                - s.project(id).pages.len().clamp(1, 3) as f32)
                                * 5.,
                        )
                    });
                    canvas(move || {
                        let (index, face) = row.get();
                        if let Some(face) = face {
                            cutout(&face, [42., 58.])
                        } else {
                            paper([42., 58.], if index == 0 { 0x888888 } else { 0xcccccc })
                        }
                    })
                    .absolute()
                    .left(move || {
                        px(17.
                            - (s.project(id).pages.len().clamp(1, 3) as f32
                                - 1.
                                - row.get().0 as f32)
                                * 3.)
                    })
                    .top(move || {
                        px(7.
                            - (s.project(id).pages.len().clamp(1, 3) as f32
                                - 1.
                                - row.get().0 as f32)
                                * 2.)
                    })
                    .w(px(42.))
                    .h(px(58.))
                    .bind_rotate(&rotation)
                    .accessible(false)
                    .pointer_events(PointerEvents::None)
                },
            ))
            .child(when(
                move || !s.project(id).pages.is_empty(),
                move || {
                    view()
                        .absolute()
                        .bottom(px(0.))
                        .right(px(1.))
                        .min_w(px(23.))
                        .h(px(23.))
                        .px_1()
                        .rounded_full()
                        .bg(rgb(0xffcc66))
                        .items_center()
                        .justify_center()
                        .child(
                            text(move || s.project(id).pages.len().to_string())
                                .text_size(px(12.))
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(rgb(0x151515)),
                        )
                },
            ))
    }
    pub fn overlay(self) -> View {
        view()
            .test_id("capture-flights")
            .absolute()
            .inset_0()
            .pointer_events(PointerEvents::None)
            .accessible(false)
            .z_index(40)
            .child(each(
                move || self.flights.get(),
                |f| (f.face.order, f.face.page),
                move |item, _| {
                    let f = item.get_untracked();
                    let progress = animated_value(0.);
                    let scale = (f.target[2] / f.source[2]).min(f.target[3] / f.source[3]);
                    let x = progress.interpolate(
                        &[0., 1.],
                        &[
                            0.,
                            f.target[0] + f.target[2] * 0.5 - f.source[0] - f.source[2] * 0.5,
                        ],
                    );
                    let y = progress.interpolate(
                        &[0., 1.],
                        &[
                            0.,
                            f.target[1] + f.target[3] * 0.5 - f.source[1] - f.source[3] * 0.5,
                        ],
                    );
                    let scaling = progress.interpolate(&[0., 0.12, 1.], &[1., 1.025, scale]);

                    let alpha = progress.interpolate(&[0., 0.07, 1.], &[0., 1., 1.]);
                    cleanup(move || progress.stop());
                    progress.timing(1., 400., Easing::EaseInOut);
                    canvas(move || cutout(&f.face, [f.source[2], f.source[3]]))
                        .test_id("capture-flying-page")
                        .absolute()
                        .left(px(f.source[0]))
                        .top(px(f.source[1]))
                        .w(px(f.source[2]))
                        .h(px(f.source[3]))
                        .bind_translate_x(&x)
                        .bind_translate_y(&y)
                        .bind_scale(&scaling)
                        .bind_opacity(&alpha)
                },
            ))
    }
}
fn insert_face(faces: &mut Vec<Face>, face: Face) {
    faces.push(face);
    faces.sort_by_key(|f| (f.order, f.page));
    let remove = faces.len().saturating_sub(3);
    faces.drain(..remove);
}
fn paper(size: [f32; 2], color: u32) -> Painter {
    Painter::new()
        .begin_path()
        .move_to(0., 0.)
        .line_to(size[0], 0.)
        .line_to(size[0], size[1])
        .line_to(0., size[1])
        .close_path()
        .fill_color(rgb(color))
        .fill()
}
fn cutout(face: &Face, size: [f32; 2]) -> Painter {
    let polygon = &face.shot.frame.polygons[face.page];
    let Some([x, y, w, h]) = snapshot::bounds(polygon) else {
        return Painter::new();
    };
    // Keep the page's aspect ratio inside the fixed stack slot.
    let iw = face.shot.frame.image.width() as f32 * w;
    let ih = face.shot.frame.image.height() as f32 * h;
    let k = (size[0] / iw).min(size[1] / ih);
    let actual = [iw * k, ih * k];
    let offset = [(size[0] - actual[0]) * 0.5, (size[1] - actual[1]) * 0.5];
    let mut p = Painter::new().save().begin_path();
    for (i, point) in polygon.iter().enumerate() {
        let px = offset[0] + (point[0] - x) / w * actual[0];
        let py = offset[1] + (point[1] - y) / h * actual[1];
        p = if i == 0 {
            p.move_to(px, py)
        } else {
            p.line_to(px, py)
        };
    }
    p.close_path()
        .clip()
        .draw_image(
            &face.shot.path.to_string_lossy(),
            Bounds {
                origin: point(
                    px(offset[0] - x / w * actual[0]),
                    px(offset[1] - y / h * actual[1]),
                ),
                size: nsh_core::geometry::size(px(actual[0] / w), px(actual[1] / h)),
            },
        )
        .restore()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsh_engine::{EngineConfig, Root};
    use nsh_testing::Headless;
    #[component]
    fn Harness() -> impl IntoElement {
        let s = State::new();
        let stack = CaptureStack::new();
        let root = std::env::temp_dir().join("smude-stack-ui-test");
        s.root.set(Some(root));
        let mut lib = crate::model::Library::examples();
        lib.projects[0].pages.truncate(2);
        let id = lib.projects[0].id;
        s.library.set(lib);
        s.routes
            .set(vec![crate::Route::Projects, crate::Route::Camera(id, None)]);
        let token = snapshot::remember(
            &image::RgbImage::from_pixel(120, 80, image::Rgb([90, 120, 60])),
            vec![
                vec![[0.05, 0.1], [0.45, 0.1], [0.48, 0.9], [0.05, 0.9]],
                vec![[0.52, 0.1], [0.95, 0.1], [0.95, 0.9], [0.52, 0.9]],
            ],
        )
        .unwrap();
        view()
            .relative()
            .w(px(390.))
            .h(px(760.))
            .element_ref(&stack.scene)
            .child(view().element_ref(&stack.preview).w(px(390.)).h(px(450.)))
            .child(stack.button(s, id))
            .child(stack.overlay())
            .child(
                view()
                    .test_id("take")
                    .w(px(44.))
                    .h(px(44.))
                    .role(AccessibilityRole::Button)
                    .on_press(move || stack.capture(s, token)),
            )
            .child(text(move || format!("landed:{}", stack.faces.get().len())))
            .child(text(move || {
                format!("flights:{}", stack.flights.get().len())
            }))
            .child(text(move || {
                format!("route:{:?}", s.routes.get().last().unwrap())
            }))
    }
    #[test]
    fn spread_lands_two_pages_quick_shots_stay_bounded_and_stack_opens_collection() {
        let mut app = Headless::start(EngineConfig::new(Root::of::<Harness>())).unwrap();
        app.first_render().unwrap();
        app.press("id:take").unwrap();
        app.wait_for("id:capture-flying-page").unwrap();
        // Acceptance of the next shot is independent of the first flight.
        app.press("id:take").unwrap();
        app.wait_for("text:landed:3").unwrap();
        app.wait_for("text:flights:0").unwrap();
        app.press("id:capture-stack").unwrap();
        app.wait_for("text:route:Project(1)").unwrap();
    }
    #[test]
    fn reduced_motion_lands_both_pages_without_flight() {
        let mut config = EngineConfig::new(Root::of::<Harness>());
        config.environment.reduce_motion = true;
        let mut app = Headless::start(config).unwrap();
        app.first_render().unwrap();
        app.press("id:take").unwrap();
        app.wait_for("text:landed:2").unwrap();
        assert!(!app.screen().matches("id:capture-flying-page").unwrap());
    }
}
