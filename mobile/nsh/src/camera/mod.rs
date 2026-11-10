//! Shared live-camera geometry. Camera frames stay native; only small contours
//! cross into the Rust UI. The painter is shared between iOS and Android.
use nsh_core::prelude::*;
use serde::{Deserialize, Serialize};
pub mod capture_stack;
mod overlay;
mod snapshot;
pub use overlay::update_overlay;
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    pub width: f32,
    pub height: f32,
    pub polygons: Vec<Vec<[f32; 2]>>,
    #[serde(default)]
    pub alignment: Option<[f32; 2]>,
}
/// Coordinates are normalized within the physically oriented analysis buffer.
/// Both native previews use aspect-fill, with the same orientation and no mirror.
pub fn aspect_fill(point: [f32; 2], frame: [f32; 2], view: [f32; 2]) -> [f32; 2] {
    let scale = (view[0] / frame[0]).max(view[1] / frame[1]);
    [
        point[0] * frame[0] * scale + (view[0] - frame[0] * scale) * 0.5,
        point[1] * frame[1] * scale + (view[1] - frame[1] * scale) * 0.5,
    ]
}
pub fn drawing(frame: &Frame, size: [f32; 2]) -> Painter {
    let mut p = Painter::new();
    if frame.width <= 0. || frame.height <= 0. {
        return p;
    }
    for polygon in &frame.polygons {
        if polygon.len() < 3 {
            continue;
        }
        p = p.begin_path();
        for (i, point) in polygon.iter().enumerate() {
            let [x, y] = aspect_fill(*point, [frame.width, frame.height], size);
            p = if i == 0 {
                p.move_to(x, y)
            } else {
                p.line_to(x, y)
            };
        }
        p = p
            .close_path()
            .fill_color(rgb(0xffcc66).opacity(0.14))
            .fill()
            .stroke_color(rgb(0xffcc66))
            .line_width(2.)
            .stroke();
    }
    p
}
#[cfg(target_os = "android")]
mod android;
#[cfg(any(target_os = "ios", target_os = "android"))]
mod inference;
#[cfg(target_os = "ios")]
pub mod ios;

/// Passive page-relative guide; has no effect on shutter eligibility.
pub fn alignment_drawing(offset: [f32; 2]) -> Painter {
    let circle = |mut painter: Painter, cx: f32, cy: f32, r: f32| {
        painter = painter.begin_path();
        for i in 0..32 {
            let a = i as f32 * std::f32::consts::TAU / 32.;
            let (x, y) = (cx + r * a.cos(), cy + r * a.sin());
            painter = if i == 0 {
                painter.move_to(x, y)
            } else {
                painter.line_to(x, y)
            };
        }
        painter.close_path()
    };
    let p = circle(Painter::new(), 35., 21., 15.)
        .stroke_color(rgb(0xffcc66).opacity(0.6))
        .line_width(1.)
        .stroke()
        .begin_path()
        .move_to(15., 21.)
        .line_to(55., 21.)
        .move_to(35., 1.)
        .line_to(35., 41.)
        .stroke_color(rgb(0xffffff).opacity(0.3))
        .line_width(1.)
        .stroke();
    circle(
        p,
        35. + (offset[0] / 0.2).clamp(-1., 1.) * 14.,
        21. + (offset[1] / 0.2).clamp(-1., 1.) * 14.,
        3.5,
    )
    .fill_color(rgb(0xffcc66))
    .fill()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn centered_points_stay_centered() {
        assert_eq!(
            aspect_fill([0.5, 0.5], [480., 640.], [400., 700.]),
            [200., 350.]
        );
    }
    #[test]
    fn fill_crops_horizontally_without_stretch() {
        let p = aspect_fill([0., 0.], [480., 640.], [400., 700.]);
        assert_eq!(p, [-62.5, 0.]);
    }
    #[test]
    fn rotated_frame_maps_in_its_own_orientation() {
        assert_eq!(
            aspect_fill([1., 1.], [640., 480.], [800., 600.]),
            [800., 600.]
        );
    }
}

/// A short dark shutter cue. It never participates in readiness or hit testing.
#[derive(Clone, Copy)]
pub struct CaptureFeedback {
    opacity: AnimatedValue,
}
impl CaptureFeedback {
    pub fn new() -> Self {
        let opacity = animated_value(0.);
        cleanup(move || opacity.stop());
        Self { opacity }
    }
    pub fn capture(self) {
        self.opacity.stop();
        self.opacity.set(if reduce_motion() { 0.25 } else { 0.65 });
        self.opacity.timing(
            0.,
            if reduce_motion() { 80. } else { 180. },
            Easing::EaseOut,
        );
    }
    pub fn view(self) -> View {
        view()
            .test_id("capture-flash")
            .absolute()
            .inset_0()
            .bg(rgb(0x000000))
            .bind_opacity(&self.opacity)
            .pointer_events(PointerEvents::None)
            .accessible(false)
    }
}
