//! Pointer-owned drag previews with edge scrolling and an independent second finger.
//! The library is saved only when the held pointer is released.
use crate::state::State;
use nsh_core::{TouchEvent, TouchPhase, prelude::*};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Visual {
    pub x: AnimatedValue,
    pub y: AnimatedValue,
    pub step: Signal<[f32; 2]>,
}
#[derive(Clone, Debug, PartialEq)]
struct Drag {
    token: u64,
    page: u64,
    from: usize,
    order: Vec<u64>,
    columns: usize,
    pointer: Option<u32>,
    origin: [f32; 2],
    start_scroll: f32,
    visual: Visual,
}
#[derive(Clone, Copy, Debug, PartialEq)]
struct Finger {
    id: u32,
    at: [f32; 2],
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Metrics {
    offset: f32,
    top: f32,
    height: f32,
    content: f32,
}
#[derive(Clone, Copy)]
pub struct Reordering {
    drag: Signal<Option<Drag>>,
    preview: Signal<Vec<u64>>,
    columns: Signal<usize>,
    fingers: Signal<Vec<Finger>>,
    candidates: Signal<Vec<(u64, u32)>>,
    secondary: Signal<Option<Finger>>,
    metrics: Signal<Metrics>,
    requested: Signal<f32>,
    suppress: Signal<Option<(u64, u64)>>,
    scroll: ElementRef,
    pub scroll_value: AnimatedValue,
    start_scroll_value: AnimatedValue,
}
impl Reordering {
    pub fn new(columns: Signal<usize>, scroll: ElementRef) -> Self {
        Self {
            drag: signal(None),
            preview: signal(vec![]),
            columns,
            fingers: signal(vec![]),
            candidates: signal(vec![]),
            secondary: signal(None),
            metrics: signal(Metrics::default()),
            requested: signal(0.),
            suppress: signal(None),
            scroll,
            scroll_value: animated_value(0.),
            start_scroll_value: animated_value(0.),
        }
    }
    pub fn active(self) -> bool {
        self.drag.get().is_some()
    }
    pub fn dragging(self, page: u64) -> bool {
        self.drag
            .with(|d| d.as_ref().is_some_and(|d| d.page == page))
    }
    pub fn owns_scrolling(self) -> bool {
        self.active() || self.secondary.get().is_some()
    }
    pub fn display_y(self, page: u64, y: AnimatedValue) -> AnimatedNode {
        let active = animated_value(0.);
        effect(move || active.set(if self.dragging(page) { 1. } else { 0. }));
        // Native scroll updates this graph on the UI thread, so the held page
        // stays under the pointer even before on_scroll reaches Rust.
        y.add(&self.scroll_value.sub(&self.start_scroll_value).mul(&active))
    }
    pub fn viewport(self, height: f32) {
        self.metrics.update(|m| m.height = height);
    }
    pub fn content(self, height: f32) {
        self.metrics.update(|m| m.content = height);
    }
    pub fn scrolled(self, event: &ScrollEvent) {
        self.metrics.update(|m| {
            m.offset = event.offset.y.0.max(0.);
            m.height = event.size.height.0;
            m.content = event.content_size.height.0;
        });
        if !self.owns_scrolling() {
            self.requested.set(event.offset.y.0.max(0.));
        }
        self.refresh_pointer();
    }
    pub fn arm(self, page: u64, reference: ElementRef, event: &TouchEvent) {
        if event.phase != TouchPhase::Start || self.active() {
            return;
        }
        let Some(node) = reference.node() else {
            return;
        };
        for p in &event.points {
            if p.changed && p.path.contains(&node) {
                self.fingers.update(|v| {
                    v.retain(|f| f.id != p.id);
                    v.push(Finger {
                        id: p.id,
                        at: [p.page.x.0, p.page.y.0],
                    });
                });
                self.candidates.update(|v| {
                    v.retain(|p| p.0 != page);
                    v.push((page, p.id));
                });
            }
        }
    }
    pub fn hold(self, s: State, project: u64, page: u64, visual: Visual) {
        if self.active() || self.secondary.get_untracked().is_some() {
            return;
        }
        let finger = self
            .candidates
            .with_untracked(|v| v.iter().find(|p| p.0 == page).map(|p| p.1))
            .and_then(|id| {
                self.fingers
                    .with_untracked(|v| v.iter().find(|f| f.id == id).copied())
            });
        if let Some(f) = finger {
            self.begin(s, project, page, visual, Some(f));
        }
    }
    fn begin(self, s: State, project: u64, page: u64, visual: Visual, finger: Option<Finger>) {
        let order = s
            .project(project)
            .pages
            .iter()
            .map(|p| p.id)
            .collect::<Vec<_>>();
        let Some(from) = order.iter().position(|p| *p == page) else {
            return;
        };
        let offset = self.metrics.get_untracked().offset;
        self.requested.set(offset);
        self.start_scroll_value.set(offset);
        let token = crate::model::id();
        self.preview.set(order.clone());
        self.drag.set(Some(Drag {
            token,
            page,
            from,
            order,
            columns: self.columns.get_untracked(),
            pointer: finger.map(|f| f.id),
            origin: finger.map_or([0., 0.], |f| f.at),
            start_scroll: offset,
            visual,
        }));
        spawn(async move {
            if let Some(bounds) = self.scroll.measure().await
                && self
                    .drag
                    .with_untracked(|d| d.as_ref().is_some_and(|d| d.token == token))
            {
                self.metrics.update(|m| {
                    m.top = bounds.page.y.0;
                    m.height = bounds.frame.size.height.0;
                });
            }
            let mut previous = nsh_core::time::now();
            loop {
                timer(Duration::from_millis(16)).await;
                let Some(d) = self.drag.get_untracked().filter(|d| d.token == token) else {
                    break;
                };
                if self.columns.get_untracked() != d.columns {
                    self.cancel();
                    break;
                }
                let dt = nsh_core::time::now()
                    .duration_since(previous)
                    .as_secs_f32()
                    .min(0.05);
                previous = nsh_core::time::now();
                if d.pointer.is_none() {
                    let x = d.visual.x.value().await;
                    let y = d.visual.y.value().await;
                    self.update([x, y], d.visual.step.get_untracked());
                }
                if self.secondary.get_untracked().is_none()
                    && let Some(id) = d.pointer
                    && let Some(f) = self
                        .fingers
                        .with_untracked(|v| v.iter().find(|f| f.id == id).copied())
                {
                    let m = self.metrics.get_untracked();
                    self.scroll_by(edge_velocity(f.at[1] - m.top, m.height) * dt);
                }
            }
        });
    }
    pub fn touch(self, s: State, project: u64, event: &TouchEvent) {
        if event.phase == TouchPhase::Cancel {
            self.cancel();
            return;
        }
        let drag = self.drag.get_untracked();
        for p in &event.points {
            if event.phase == TouchPhase::End && p.changed {
                continue;
            }
            self.fingers.update(|v| {
                v.retain(|f| f.id != p.id);
                v.push(Finger {
                    id: p.id,
                    at: [p.page.x.0, p.page.y.0],
                });
            });
        }
        if let Some(d) = &drag
            && event.phase == TouchPhase::Start
            && self.secondary.get_untracked().is_none()
            && let Some(p) = event
                .points
                .iter()
                .find(|p| p.changed && Some(p.id) != d.pointer)
        {
            self.secondary.set(Some(Finger {
                id: p.id,
                at: [p.page.x.0, p.page.y.0],
            }));
        }
        if let Some(old) = self.secondary.get_untracked()
            && let Some(p) = event.points.iter().find(|p| p.id == old.id && p.changed)
        {
            if event.phase == TouchPhase::End {
                self.secondary.set(None);
            } else if event.phase == TouchPhase::Move {
                self.scroll_by(old.at[1] - p.page.y.0);
                self.secondary.set(Some(Finger {
                    id: p.id,
                    at: [p.page.x.0, p.page.y.0],
                }));
            }
        }
        if let Some(d) = drag
            && let Some(p) = event.points.iter().find(|p| Some(p.id) == d.pointer)
        {
            let delta = [p.page.x.0 - d.origin[0], p.page.y.0 - d.origin[1]];
            d.visual.x.set(delta[0]);
            d.visual.y.set(delta[1]);
            self.update(delta, d.visual.step.get_untracked());
            if event.phase == TouchPhase::End && p.changed {
                if delta[0].hypot(delta[1]) > 6.
                    || (self.metrics.get_untracked().offset - d.start_scroll).abs() > 6.
                {
                    self.suppress_click(d.page);
                }
                self.commit(s, project, false, delta);
            }
        }
        if event.phase == TouchPhase::End {
            let lifted = event
                .points
                .iter()
                .filter(|p| p.changed)
                .map(|p| p.id)
                .collect::<Vec<_>>();
            self.fingers
                .update(|v| v.retain(|f| !lifted.contains(&f.id)));
            self.candidates
                .update(|v| v.retain(|p| !lifted.contains(&p.1)));
        }
    }
    fn refresh_pointer(self) {
        if let Some(d) = self.drag.get_untracked()
            && let Some(id) = d.pointer
            && let Some(f) = self
                .fingers
                .with_untracked(|v| v.iter().find(|f| f.id == id).copied())
        {
            self.update(
                [f.at[0] - d.origin[0], f.at[1] - d.origin[1]],
                d.visual.step.get_untracked(),
            );
        }
    }
    fn scroll_by(self, delta: f32) {
        let m = self.metrics.get_untracked();
        let to = (self.requested.get_untracked() + delta).clamp(0., (m.content - m.height).max(0.));
        if (to - self.requested.get_untracked()).abs() > 0.01 {
            self.requested.set(to);
            self.scroll.scroll_to(point(px(0.), px(to)), false);
        }
    }
    fn suppress_click(self, page: u64) {
        let token = crate::model::id();
        self.suppress.set(Some((page, token)));
        spawn(async move {
            timer(Duration::from_millis(180)).await;
            if self.suppress.get_untracked() == Some((page, token)) {
                self.suppress.set(None);
            }
        });
    }
    pub fn ignore_click(self, page: u64) -> bool {
        self.suppress
            .with_untracked(|p| p.is_some_and(|p| p.0 == page))
    }
    pub fn offset(self, page: u64, base: usize, step: [f32; 2]) -> Option<[f32; 2]> {
        let drag = self.drag.get();
        if drag.as_ref().is_some_and(|d| d.page == page) {
            return None;
        }
        let to = self
            .preview
            .with(|v| v.iter().position(|p| *p == page))
            .unwrap_or(base);
        let cols = drag
            .as_ref()
            .map_or_else(|| self.columns.get(), |d| d.columns)
            .max(1);
        Some([
            ((to % cols) as f32 - (base % cols) as f32) * step[0],
            ((to / cols) as f32 - (base / cols) as f32) * step[1],
        ])
    }
    fn update(self, delta: [f32; 2], step: [f32; 2]) {
        let Some(d) = self.drag.get_untracked() else {
            return;
        };
        let mut order = d.order;
        let scroll_delta = self.metrics.get_untracked().offset - d.start_scroll;
        let to = destination(
            d.from,
            order.len(),
            [delta[0], delta[1] + scroll_delta],
            step,
            d.columns,
        );
        order.remove(d.from);
        order.insert(to, d.page);
        if self.preview.get_untracked() != order {
            self.preview.set(order);
        }
    }
    fn commit(self, s: State, project: u64, cancelled: bool, delta: [f32; 2]) {
        let Some(d) = self.drag.get_untracked() else {
            return;
        };
        let to = destination(
            d.from,
            d.order.len(),
            [
                delta[0],
                delta[1] + self.metrics.get_untracked().offset - d.start_scroll,
            ],
            d.visual.step.get_untracked(),
            d.columns,
        );
        self.reset_drag();
        if !cancelled && self.columns.get_untracked() == d.columns && d.from != to {
            s.update(project, |p| p.move_page_to(d.page, to));
        }
    }
    pub fn cancel(self) {
        self.reset_drag();
        self.fingers.set(vec![]);
        self.candidates.set(vec![]);
        self.secondary.set(None);
    }
    fn reset_drag(self) {
        if let Some(d) = self.drag.get_untracked() {
            d.visual.x.set(0.);
            d.visual.y.set(0.);
        }
        self.drag.set(None);
        self.preview.set(vec![]);
    }
    /// Mouse/headless fallback. Phones use touch identities instead of a
    /// centroid-based pan, which would jump when a second finger arrives.
    pub fn attach_pan_fallback(
        self,
        card: View,
        s: State,
        project: u64,
        page: u64,
        visual: Visual,
        selected: Signal<Vec<u64>>,
    ) -> View {
        if cfg!(any(target_os = "ios", target_os = "android")) {
            return card;
        }
        card.pan_values(Some(&visual.x), Some(&visual.y))
            .pan_enabled(move || selected.with(|ids| ids.as_slice() == [page]))
            .on_pan_start(move |_: &PanEvent| self.begin(s, project, page, visual, None))
            .on_pan_end(move |e: &PanEvent| {
                self.commit(
                    s,
                    project,
                    e.cancelled,
                    [e.translation.x.0, e.translation.y.0],
                )
            })
    }
}
fn destination(
    from: usize,
    count: usize,
    delta: [f32; 2],
    step: [f32; 2],
    columns: usize,
) -> usize {
    let cols = columns.max(1);
    let col = ((from % cols) as f32 + delta[0] / step[0].max(1.))
        .round()
        .clamp(0., (cols - 1) as f32) as usize;
    let row = ((from / cols) as f32 + delta[1] / step[1].max(1.))
        .round()
        .max(0.) as usize;
    (row * cols + col).min(count.saturating_sub(1))
}
fn edge_velocity(y: f32, height: f32) -> f32 {
    if height <= 0. {
        return 0.;
    }
    let band = 64f32.min(height / 4.);
    if y < band {
        -700. * ((band - y) / band).clamp(0., 1.).powi(2)
    } else if y > height - band {
        700. * ((y - height + band) / band).clamp(0., 1.).powi(2)
    } else {
        0.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn destination_supports_phone_and_tablet_grids_and_clamps_edges() {
        assert_eq!(destination(0, 30, [0., 450.], [100., 150.], 3), 9);
        assert_eq!(destination(0, 30, [0., 450.], [100., 150.], 5), 15);
        assert_eq!(destination(4, 7, [1000., 1000.], [100., 150.], 3), 6);
        assert_eq!(destination(4, 7, [-1000., -1000.], [100., 150.], 3), 0);
    }
    #[test]
    fn edge_scroll_accelerates_smoothly_and_only_near_the_edges() {
        assert_eq!(edge_velocity(100., 600.), 0.);
        assert_eq!(edge_velocity(536., 600.), 0.);
        assert_eq!(edge_velocity(0., 600.), -700.);
        assert_eq!(edge_velocity(600., 600.), 700.);
        assert_eq!(edge_velocity(32., 600.), -175.);
        assert_eq!(edge_velocity(568., 600.), 175.);
        assert_eq!(edge_velocity(-100., 600.), -700.);
        assert_eq!(edge_velocity(800., 600.), 700.);
        assert_eq!(edge_velocity(0., 0.), 0.);
    }
}
