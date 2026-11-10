//! Relative horizontal metric from orthogonal print directions and camera rays.
//! A common vertical vanishing direction is evidence-gated; no cylindrical
//! polynomial or per-system shear is assumed. Focal length is an explicit prior.
use crate::{
    geometry::{Observations, Point},
    math::median,
};
use serde::Serialize;

#[derive(Clone, Debug)]
pub(crate) struct ScaleSample {
    pub point: Point,
    pub scale: f64,
    pub confidence: f64,
}
#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct SurfaceMetricReport {
    pub accepted: bool,
    pub before_relative_scale_rms: f64,
    pub after_relative_scale_rms: f64,
    pub before_direction_rms_degrees: f64,
    pub after_direction_rms_degrees: f64,
    pub samples: usize,
    pub supporting_staves: usize,
    pub vertical_direction: Option<[f64; 3]>,
    pub median_direction_error_degrees: f64,
    pub focal_prior_pixels: f64,
    pub maximum_requested_expansion: f64,
    pub rejection: Option<&'static str>,
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: [f64; 3]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

pub(crate) fn measure(
    data: &Observations,
    map: &impl Fn(Point) -> Point,
    principal: Point,
    focal: f64,
) -> (Vec<ScaleSample>, SurfaceMetricReport) {
    let mut report = SurfaceMetricReport {
        focal_prior_pixels: focal,
        ..Default::default()
    };
    let reject = |mut r: SurfaceMetricReport, reason| {
        r.rejection = Some(reason);
        (vec![], r)
    };
    if data.staves.len() < 3 || !focal.is_finite() || focal <= 0. {
        return reject(report, "insufficient_camera_or_staff_support");
    }
    let mut lines = vec![];
    let mut positions = vec![];
    for b in &data.barlines {
        if b.x.len() < 3 {
            continue;
        }
        let n = b.x.len() as f64;
        let x = b.x.iter().sum::<f64>() / n;
        let y = b.y.iter().sum::<f64>() / n;
        let den = b.y.iter().map(|p| (p - y).powi(2)).sum::<f64>();
        if den < 1. {
            continue;
        }
        let m =
            b.x.iter()
                .zip(&b.y)
                .map(|(a, b)| (a - x) * (b - y))
                .sum::<f64>()
                / den;
        let p = data.staves[b.staff].sample(0.5);
        let gap = (p[4].y - p[0].y) / 4.;
        let rms =
            (b.x.iter()
                .zip(&b.y)
                .map(|(a, b)| (a - x - m * (b - y)).powi(2))
                .sum::<f64>()
                / n)
                .sqrt();
        if rms > 0.12 * gap || m.abs() > 0.6 {
            continue;
        }
        let z = -((x - principal.x) - m * (y - principal.y)) / focal;
        let len = (1. + m * m).sqrt();
        lines.push([1. / len, -m / len, z / len]);
        positions.push(Point { x, y });
    }
    if lines.len() < 8 {
        return reject(report, "insufficient_straight_bar_support");
    }
    let xmin = positions.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
    let xmax = positions
        .iter()
        .map(|p| p.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let ymin = positions.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
    let ymax = positions
        .iter()
        .map(|p| p.y)
        .fold(f64::NEG_INFINITY, f64::max);
    if xmax - xmin < 0.35 * f64::from(data.width) || ymax - ymin < 0.25 * f64::from(data.height) {
        return reject(report, "bar_directions_have_insufficient_spatial_support");
    }
    // Homogeneous VP = (vx, 1, vz); infinity (vz=0) remains representable.
    let (mut vx, mut vz) = (0., 0.);
    for _ in 0..5 {
        let (mut a, mut b, mut c, mut d, mut e) = (0., 0., 0., 0., 0.);
        for l in &lines {
            let residual = (l[0] * vx + l[1] + l[2] * vz).abs();
            let w = 0.008 / residual.max(0.008);
            a += w * l[0] * l[0];
            b += w * l[0] * l[2];
            c += w * l[2] * l[2];
            d -= w * l[0] * l[1];
            e -= w * l[2] * l[1];
        }
        let det = a * c - b * b;
        if det <= 1e-8 * a * c {
            return reject(report, "ill_conditioned_vertical_direction");
        }
        vx = (d * c - e * b) / det;
        vz = (e * a - d * b) / det;
    }
    let v = [vx, 1., vz];
    let len = norm(v);
    let v = v.map(|x| x / len);
    let mut errors: Vec<_> = lines
        .iter()
        .zip(&positions)
        .map(|(l, p)| {
            let px = (p.x - principal.x) / focal;
            let py = (p.y - principal.y) / focal;
            let predicted = (v[0] - px * v[2]) / (v[1] - py * v[2]);
            (predicted.atan() - (-l[1] / l[0]).atan())
                .abs()
                .to_degrees()
        })
        .collect();
    errors.sort_by(f64::total_cmp);
    report.median_direction_error_degrees = median(errors.clone());
    report.vertical_direction = Some(v);
    if report.median_direction_error_degrees > 0.6 || errors[errors.len() * 9 / 10] > 2. {
        return reject(report, "vertical_rulings_are_not_consistent");
    }
    let mut result = vec![];
    for staff in &data.staves {
        let mut samples = vec![];
        let (left, right) = (staff.x[0], *staff.x.last().unwrap());
        for i in 0..41 {
            let t = i as f64 / 40.;
            let p = staff.sample(t);
            let gap = (p[4].y - p[0].y) / 4.;
            let radius = 3. * gap;
            let a = staff.sample(((p[2].x - radius - left) / (right - left)).clamp(0., 1.))[2];
            let b = staff.sample(((p[2].x + radius - left) / (right - left)).clamp(0., 1.))[2];
            let slope = (b.y - a.y) / (b.x - a.x);
            let ray = [
                (p[2].x - principal.x) / focal,
                (p[2].y - principal.y) / focal,
                1.,
            ];
            // The image tangent back-projects to a plane through the camera.
            // Its intersection with the plane perpendicular to the vertical
            // ruling gives the local 3-D horizontal direction (up to sign).
            let plane = cross(ray, [1., slope, 0.]);
            let h = cross(v, plane);
            let condition = norm(h) / norm(plane);
            if condition < 0.07 {
                continue;
            }
            let h = h.map(|x| x / norm(h));
            let hp = [h[0] - ray[0] * h[2], h[1] - ray[1] * h[2]];
            let vp = [v[0] - ray[0] * v[2], v[1] - ray[1] * v[2]];
            // Projected normal spacing / projected horizontal length. Depth
            // and printed staff size cancel in this ratio. One sqrt(1+m²)
            // converts vertical gaps to normal gaps, another converts dx to
            // tangent arc length when forming the x-derivative constraint.
            let physical = (hp[0] * vp[1] - hp[1] * vp[0]).abs() / (hp[0] * hp[0] + hp[1] * hp[1]);
            let center = map(p[2]);
            let top = map(p[0]);
            let bottom = map(p[4]);
            let a = map(Point {
                x: p[2].x - 0.5,
                y: p[2].y - slope * 0.5,
            });
            let b = map(Point {
                x: p[2].x + 0.5,
                y: p[2].y + slope * 0.5,
            });
            let scale =
                (bottom.y - top.y) / 4. / gap * (1. + slope * slope) * physical / (b.x - a.x);
            if scale.is_finite() && (0.3..3.).contains(&scale) {
                samples.push((
                    t,
                    ScaleSample {
                        point: center,
                        scale,
                        confidence: (condition / 0.2).min(1.),
                    },
                ));
            }
        }
        let reference: Vec<_> = samples
            .iter()
            .filter(|(t, _)| (0.3..=0.85).contains(t))
            .map(|(_, s)| s.scale)
            .collect();
        if reference.len() < 8 {
            continue;
        }
        let reference = median(reference);
        let count = samples.len() as f64;
        for (_, mut s) in samples {
            // Recover relative foreshortening, not an invented global font size
            // or page aspect. A bounded prior limits uncertain camera depth.
            s.scale = (s.scale / reference).clamp(0.95, 1.45);
            s.confidence /= count;
            report.maximum_requested_expansion =
                report.maximum_requested_expansion.max(s.scale - 1.);
            result.push(s);
        }
        report.supporting_staves += 1;
    }
    report.samples = result.len();
    if report.supporting_staves < 3 || report.maximum_requested_expansion < 0.08 {
        return reject(report, "no_supported_horizontal_foreshortening");
    }
    (result, report)
}
