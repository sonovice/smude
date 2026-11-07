use crate::geometry::{Config, Observations, Point, StaffEdge};
use crate::math::median;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Projective {
    pub parameters: [f64; 4],
    origin: Point,
    scale: f64,
}

impl Projective {
    pub fn jacobian_det(&self, point: Point) -> f64 {
        let [b, c, p, q] = self.parameters;
        let z = 1.
            + p * (point.x - self.origin.x) / self.scale
            + q * (point.y - self.origin.y) / self.scale;
        (1. - b * c) / (z * z * z)
    }
    pub fn identity(data: &Observations) -> Self {
        Self {
            parameters: [0.; 4],
            origin: Point {
                x: f64::from(data.width - 1) / 2.,
                y: f64::from(data.height - 1) / 2.,
            },
            scale: f64::from(data.width.max(data.height)),
        }
    }
    pub fn forward(&self, point: Point) -> Point {
        let [b, c, p, q] = self.parameters;
        let x = (point.x - self.origin.x) / self.scale;
        let y = (point.y - self.origin.y) / self.scale;
        let z = 1. + p * x + q * y;
        Point {
            x: self.origin.x + self.scale * (x + b * y) / z,
            y: self.origin.y + self.scale * (c * x + y) / z,
        }
    }
    pub fn inverse(&self, point: Point) -> Point {
        let [b, c, p, q] = self.parameters;
        let u = (point.x - self.origin.x) / self.scale;
        let v = (point.y - self.origin.y) / self.scale;
        // Solve the two linear equations after multiplying the denominator.
        let (a, bb, cc, d) = (1. - u * p, b - u * q, c - v * p, 1. - v * q);
        let det = a * d - bb * cc;
        Point {
            x: self.origin.x + self.scale * (u * d - bb * v) / det,
            y: self.origin.y + self.scale * (a * v - cc * u) / det,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PerspectiveReport {
    pub source: &'static str,
    pub page_frame_rejection: Option<&'static str>,
    pub enabled: bool,
    pub parameters: [f64; 4],
    pub converged: bool,
    pub iterations: usize,
    pub at_parameter_bound: bool,
    pub heldout_x_fraction: [f64; 2],
    pub before_heldout_spacing_rms: f64,
    pub after_heldout_spacing_rms: f64,
    pub metric_assumption: &'static str,
}

fn sample(data: &Observations, model: &Projective, train: bool) -> Vec<Vec<[f64; 5]>> {
    data.staves
        .iter()
        .map(|s| {
            let point = |k: usize, i: usize| {
                model.forward(Point {
                    x: s.x[i],
                    y: s.y[k][i],
                })
            };
            let lo = (0..5)
                .map(|k| point(k, 0).x)
                .fold(f64::NEG_INFINITY, f64::max);
            let hi = (0..5)
                .map(|k| point(k, s.x.len() - 1).x)
                .fold(f64::INFINITY, f64::min);
            (0..40)
                .filter_map(|j| {
                    let t = 0.02 + 0.96 * j as f64 / 39.;
                    if ((0.38..=0.58).contains(&t)) == train {
                        return None;
                    }
                    let x = lo + (hi - lo) * t;
                    Some(std::array::from_fn(|k| {
                        // Compare the five lines at common transformed x.
                        // A homography preserves each measured line segment.
                        let (mut a, mut b) = (0, s.x.len() - 1);
                        while b - a > 1 {
                            let m = (a + b) / 2;
                            if point(k, m).x < x {
                                a = m;
                            } else {
                                b = m;
                            }
                        }
                        let (a, b) = (point(k, a), point(k, b));
                        a.y + (b.y - a.y) * (x - a.x) / (b.x - a.x)
                    }))
                })
                .collect()
        })
        .collect()
}

/// Eliminate h(x) and one constant d over the ENTIRE staff width. The common
/// h(x) is free to bend; separate d(x) is intentionally forbidden in this fit.
fn staff_regression(samples: &[[f64; 5]]) -> (Vec<f64>, f64) {
    let k = [-2., -1., 0., 1., 2.];
    let mut d = median(samples.iter().map(|s| (s[4] - s[0]) / 4.).collect());
    let mut centers: Vec<f64> = samples.iter().map(|s| s[2]).collect();
    for _ in 0..8 {
        let (mut num, mut den) = (0., 0.);
        for (s, h) in samples.iter().zip(&mut centers) {
            let weights: [f64; 5] = std::array::from_fn(|i| {
                (0.08 * d / (s[i] - *h - k[i] * d).abs().max(1e-12)).min(1.)
            });
            *h = (0..5).map(|i| weights[i] * (s[i] - k[i] * d)).sum::<f64>()
                / weights.iter().sum::<f64>();
            num += (0..5).map(|i| weights[i] * k[i] * (s[i] - *h)).sum::<f64>();
            den += (0..5).map(|i| weights[i] * k[i] * k[i]).sum::<f64>();
        }
        d = num / den;
    }
    (centers, d)
}

pub(crate) fn joint_staff(samples: &[[f64; 5]]) -> (f64, f64) {
    let (centers, d) = staff_regression(samples);
    (median(centers), d)
}

fn residual(
    data: &Observations,
    model: &Projective,
    config: &Config,
    edges: &[StaffEdge],
) -> Vec<f64> {
    let samples = sample(data, model, true);
    let mut out = vec![];
    let mut spacings = vec![];
    let k = [-2., -1., 0., 1., 2.];
    for (staff, values) in data.staves.iter().zip(&samples) {
        let space = median(
            (0..staff.x.len())
                .map(|j| (staff.y[4][j] - staff.y[0][j]) / 4.)
                .collect(),
        );
        let (centers, d) = staff_regression(values);
        spacings.push(d);
        for (s, h) in values.iter().zip(centers) {
            out.extend(
                (0..5).map(|i| (s[i] - h - k[i] * d) / space / (values.len() as f64).sqrt()),
            );
        }
        // A weak global staff-direction prior fixes the remaining vertical
        // shear gauge. It never requires the intervening curve to be straight.
        let ends = [staff.sample(0.02)[2], staff.sample(0.98)[2]].map(|p| model.forward(p));
        out.push(0.03 * (ends[1].y - ends[0].y) / space);
    }
    for group in &config.staff_size_groups {
        if group.len() > 1 {
            let target = median(group.iter().map(|i| spacings[*i]).collect());
            out.extend(
                group
                    .iter()
                    .map(|i| config.size_prior * (spacings[*i] / target - 1.)),
            );
        }
    }
    for bar in &data.barlines {
        let points: Vec<Point> = (0..12)
            .map(|j| {
                let i = j * (bar.x.len() - 1) / 11;
                model.forward(Point {
                    x: bar.x[i],
                    y: bar.y[i],
                })
            })
            .collect();
        let mean = points.iter().map(|p| p.x).sum::<f64>() / 12.;
        let s = &data.staves[bar.staff];
        let space = median(
            (0..s.x.len())
                .map(|j| (s.y[4][j] - s.y[0][j]) / 4.)
                .collect(),
        );
        out.extend(points.iter().map(|p| (p.x - mean) / space / 12_f64.sqrt()));
    }
    // Staff endpoints are geometric observations, never printed bar strokes.
    // Their uncertainty gets one quarter of the real-bar data weight.
    let mut claimed = vec![];
    for edge in edges {
        if edge.confidence < 0.7 || claimed.contains(&(edge.side, edge.staff)) {
            continue;
        }
        let mut members = vec![edge.staff];
        let mut cursor = 0;
        while cursor < members.len() {
            if let Some(other) = edges
                .iter()
                .find(|e| e.staff == members[cursor] && e.side == edge.side && e.confidence >= 0.7)
            {
                for staff in &other.aligned_staffs {
                    if !members.contains(staff)
                        && edges.iter().any(|e| {
                            e.staff == *staff && e.side == edge.side && e.confidence >= 0.7
                        })
                    {
                        members.push(*staff);
                    }
                }
            }
            cursor += 1;
        }
        claimed.extend(members.iter().map(|i| (edge.side, *i)));
        let points: Vec<_> = edges
            .iter()
            .filter(|e| e.side == edge.side && members.contains(&e.staff) && e.confidence >= 0.7)
            .flat_map(|e| e.points.iter().map(|p| model.forward(*p)))
            .collect();
        if points.len() < 3 {
            continue;
        }
        let space = median(
            (0..data.staves[edge.staff].x.len())
                .map(|j| (data.staves[edge.staff].y[4][j] - data.staves[edge.staff].y[0][j]) / 4.)
                .collect(),
        );
        let mean = points.iter().map(|p| p.x).sum::<f64>() / points.len() as f64;
        let weight = (0.25 * edge.confidence / points.len() as f64).sqrt();
        out.extend(points.iter().map(|p| weight * (p.x - mean) / space));
    }
    out.extend(model.parameters.map(|v| 0.03 * v));
    out
}

fn loss(r: &[f64]) -> f64 {
    r.iter()
        .map(|v| {
            let a = v.abs();
            if a < 0.08 {
                a * a / 2.
            } else {
                0.08 * (a - 0.04)
            }
        })
        .sum()
}

fn solve<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for j in 0..N {
        let i = (j..N).max_by(|i, k| a[*i][j].abs().total_cmp(&a[*k][j].abs()))?;
        a.swap(i, j);
        b.swap(i, j);
        if a[j][j].abs() < 1e-16 {
            return None;
        }
        let pivot = a[j][j];
        for value in a[j].iter_mut().skip(j) {
            *value /= pivot;
        }
        b[j] /= pivot;
        for i in 0..N {
            if i == j {
                continue;
            }
            let factor = a[i][j];
            let pivot_row = a[j];
            for (value, pivot) in a[i].iter_mut().zip(pivot_row).skip(j) {
                *value -= factor * pivot;
            }
            b[i] -= factor * b[j];
        }
    }
    Some(b)
}

fn heldout_rms(data: &Observations, model: &Projective) -> f64 {
    let mut squared = vec![];
    for (s, values) in data.staves.iter().zip(sample(data, model, false)) {
        let space = median(
            (0..s.x.len())
                .map(|j| (s.y[4][j] - s.y[0][j]) / 4.)
                .collect(),
        );
        let (centers, d) = staff_regression(&values);
        for (v, h) in values.iter().zip(centers) {
            squared.extend((0..5).map(|i| ((v[i] - h - (i as f64 - 2.) * d) / space).powi(2)));
        }
    }
    (squared.iter().sum::<f64>() / squared.len() as f64).sqrt()
}

pub(crate) fn fit(
    data: &Observations,
    config: &Config,
    edges: &[StaffEdge],
) -> (Projective, PerspectiveReport) {
    let mut model = Projective::identity(data);
    let before = heldout_rms(data, &model);
    let mut converged = !config.perspective;
    let mut iterations = 0;
    if config.perspective {
        let mut damping = 1e-3;
        for _ in 0..100 {
            iterations += 1;
            let r = residual(data, &model, config, edges);
            let cost = loss(&r);
            let mut columns = vec![];
            for j in 0..4 {
                let mut next = model.clone();
                next.parameters[j] += 1e-5;
                columns.push(
                    residual(data, &next, config, edges)
                        .iter()
                        .zip(&r)
                        .map(|(a, b)| (a - b) / 1e-5)
                        .collect::<Vec<_>>(),
                );
            }
            let mut a = [[0.; 4]; 4];
            let mut b = [0.; 4];
            for i in 0..r.len() {
                let w = (0.08 / r[i].abs().max(1e-12)).min(1.);
                for j in 0..4 {
                    b[j] -= w * columns[j][i] * r[i];
                    for k in 0..4 {
                        a[j][k] += w * columns[j][i] * columns[k][i];
                    }
                }
            }
            for (j, row) in a.iter_mut().enumerate() {
                row[j] += damping;
            }
            let Some(delta) = solve(a, b) else {
                break;
            };
            let mut next = model.clone();
            for (j, change) in delta.iter().enumerate() {
                let limit = if j < 2 { 0.25 } else { 0.6 };
                next.parameters[j] = (next.parameters[j] + change).clamp(-limit, limit);
            }
            let next_cost = loss(&residual(data, &next, config, edges));
            if next_cost < cost {
                model = next;
                damping = (damping * 0.3).max(1e-10);
                if cost - next_cost < 1e-10 {
                    converged = true;
                    break;
                }
            } else {
                damping *= 10.;
                if delta.iter().map(|v| v * v).sum::<f64>() < 1e-14 {
                    converged = true;
                    break;
                }
            }
        }
    }
    let report = PerspectiveReport {
        source: if config.perspective {
            "staff_spacing"
        } else {
            "disabled"
        },
        page_frame_rejection: None,
        enabled: config.perspective,
        parameters: model.parameters,
        converged,
        iterations,
        at_parameter_bound: model
            .parameters
            .iter()
            .enumerate()
            .any(|(j, v)| v.abs() >= if j < 2 { 0.249 } else { 0.599 }),
        heldout_x_fraction: [0.38, 0.58],
        before_heldout_spacing_rms: before,
        after_heldout_spacing_rms: heldout_rms(data, &model),
        metric_assumption: "Central aspect is fixed by gauge; weak mean staff-direction prior fixes vertical shear. Absolute aspect and arbitrary horizontal surface stretch are unidentifiable.",
    };
    (model, report)
}

/// Four ordered paper corners determine a broad projective page frame, up to
/// independent output axis scales/translations (fixed later by raster aspect).
fn page_frame(data: &Observations, corners: [Point; 4]) -> Option<Projective> {
    if corners.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
        return None;
    }
    for i in 0..4 {
        let (a, b, c) = (corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]);
        if (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x) <= 1e-6 {
            return None;
        }
    }
    let mut model = Projective::identity(data);
    let mut a = [[0.; 8]; 8];
    let mut b = [0.; 8];
    for (i, (point, (u, v))) in corners
        .iter()
        .zip([(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
        .enumerate()
    {
        let x = (point.x - model.origin.x) / model.scale;
        let y = (point.y - model.origin.y) / model.scale;
        a[2 * i] = [x, y, 1., 0., 0., 0., -u * x, -u * y];
        b[2 * i] = u;
        a[2 * i + 1] = [0., 0., 0., x, y, 1., -v * x, -v * y];
        b[2 * i + 1] = v;
    }
    let h = solve(a, b)?;
    let ax = h[0] - h[2] * h[6];
    let ay = h[4] - h[5] * h[7];
    if ax <= 1e-6 || ay <= 1e-6 {
        return None;
    }
    model.parameters = [
        (h[1] - h[2] * h[7]) / ax,
        (h[3] - h[5] * h[6]) / ay,
        h[6],
        h[7],
    ];
    if model.parameters.iter().any(|p| !p.is_finite()) {
        return None;
    }
    // Staff-only optimization needs tight parameter bounds to avoid ambiguous
    // fits. A measured quadrilateral supplies that missing constraint; strong
    // valid perspective must instead be checked for poles and orientation.
    if 1. - model.parameters[0] * model.parameters[1] <= 1e-4 {
        return None;
    }
    for x in [0., f64::from(data.width - 1)] {
        for y in [0., f64::from(data.height - 1)] {
            let z = 1.
                + model.parameters[2] * (x - model.origin.x) / model.scale
                + model.parameters[3] * (y - model.origin.y) / model.scale;
            if z < 0.25 {
                return None;
            }
        }
    }
    Some(model)
}

pub(crate) fn fit_with_page_frame(
    data: &Observations,
    config: &Config,
    corners: Option<[Point; 4]>,
    edges: &[StaffEdge],
) -> (Projective, PerspectiveReport) {
    if !config.perspective || corners.is_none() {
        return fit(data, config, edges);
    }
    let fallback = |reason| {
        let (model, mut report) = fit(data, config, edges);
        report.page_frame_rejection = Some(reason);
        (model, report)
    };
    if !config.staff_size_groups.is_empty() {
        return fallback("explicit_staff_size_groups_preferred");
    }
    let Some(model) = page_frame(data, corners.unwrap()) else {
        return fallback("invalid_or_excessive_page_frame");
    };
    let angles: Vec<f64> = data
        .barlines
        .iter()
        .filter_map(|bar| {
            let a = model.forward(Point {
                x: bar.x[0],
                y: bar.y[0],
            });
            let n = bar.x.len() - 1;
            let b = model.forward(Point {
                x: bar.x[n],
                y: bar.y[n],
            });
            ((b.y - a.y).abs() > 1.)
                .then(|| (b.x - a.x).abs().atan2((b.y - a.y).abs()).to_degrees())
        })
        .collect();
    if !angles.is_empty() && median(angles) > 2. {
        return fallback("page_frame_conflicts_with_bar_directions");
    }
    let report = PerspectiveReport {
        source: "page_boundary",
        page_frame_rejection: None,
        enabled: true,
        parameters: model.parameters,
        converged: true,
        iterations: 0,
        at_parameter_bound: false,
        heldout_x_fraction: [0.38, 0.58],
        before_heldout_spacing_rms: heldout_rms(data, &Projective::identity(data)),
        after_heldout_spacing_rms: heldout_rms(data, &model),
        metric_assumption: "Broad rectangular page-frame hypothesis from near-straight side boundaries; interior staff curvature remains free. Output aspect comes from a separate prior. This is not a recovered3D surface.",
    };
    (model, report)
}

#[cfg(test)]
mod page_frame_tests {
    use super::*;
    fn empty_data() -> Observations {
        Observations {
            width: 1000,
            height: 1400,
            staves: vec![],
            barlines: vec![],
        }
    }
    #[test]
    fn known_camera_frame_is_recovered_without_a_second_image_warp() {
        let data = empty_data();
        let mut expected = Projective::identity(&data);
        expected.parameters = [-0.035, 0.012, 0.16, 0.24];
        let rect = [
            Point { x: 160., y: 180. },
            Point { x: 840., y: 180. },
            Point { x: 840., y: 1200. },
            Point { x: 160., y: 1200. },
        ];
        let corners = rect.map(|p| expected.inverse(p));
        let found = page_frame(&data, corners).unwrap();
        for (a, b) in found.parameters.iter().zip(expected.parameters) {
            assert!((a - b).abs() < 1e-10);
        }
        let p = corners.map(|p| found.forward(p));
        assert!((p[0].x - p[3].x).abs() < 1e-8);
        assert!((p[1].x - p[2].x).abs() < 1e-8);
    }
    #[test]
    fn measured_strong_page_frame_is_not_limited_by_staff_optimizer_bounds() {
        let data = empty_data();
        let mut expected = Projective::identity(&data);
        expected.parameters = [0.38, -0.285, 0.045, 0.374];
        let corners = [
            Point { x: 180., y: 250. },
            Point { x: 820., y: 250. },
            Point { x: 820., y: 1150. },
            Point { x: 180., y: 1150. },
        ]
        .map(|p| expected.inverse(p));
        let found = page_frame(&data, corners).unwrap();
        for (a, b) in found.parameters.iter().zip(expected.parameters) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn spacing_samples_share_transformed_x_on_slanted_lines() {
        let mut data = empty_data();
        let mut model = Projective::identity(&data);
        model.parameters = [0.1, 0.08, 0.2, 0.25];
        let endpoints: [[Point; 2]; 5] = std::array::from_fn(|k| {
            [200., 800.].map(|x| {
                model.inverse(Point {
                    x,
                    y: 600. + 0.18 * (x - 500.) + (k as f64 - 2.) * 15.,
                })
            })
        });
        let lo = endpoints
            .iter()
            .map(|p| p[0].x)
            .fold(f64::NEG_INFINITY, f64::max);
        let hi = endpoints
            .iter()
            .map(|p| p[1].x)
            .fold(f64::INFINITY, f64::min);
        let x = vec![lo, hi];
        let y = std::array::from_fn(|k| {
            x.iter()
                .map(|x| {
                    let [a, b] = endpoints[k];
                    a.y + (b.y - a.y) * (x - a.x) / (b.x - a.x)
                })
                .collect()
        });
        data.staves.push(crate::geometry::Staff { x, y });
        for row in sample(&data, &model, true).into_iter().flatten() {
            for pair in row.windows(2) {
                assert!((pair[1] - pair[0] - 15.).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn crossed_or_nonfinite_corners_are_rejected() {
        let data = empty_data();
        let mut p = [
            Point { x: 0., y: 0. },
            Point { x: 900., y: 0. },
            Point { x: 900., y: 1300. },
            Point { x: 0., y: 1300. },
        ];
        p.swap(1, 2);
        assert!(page_frame(&data, p).is_none());
        p[0].x = f64::NAN;
        assert!(page_frame(&data, p).is_none());
    }
}
