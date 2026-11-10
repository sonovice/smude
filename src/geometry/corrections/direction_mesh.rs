//! A common, coarse cubic B-spline x field driven by vertical ink directions.
//! Staff flattening supplies y. No cylinder, independent per-symbol shifts or
//! local y residuals are fitted here. Control differences certify monotonic x.
use crate::geometry::{Observations, Point, StaffEdge, VerticalPath};
use serde::Serialize;

const NX: usize = 10;
const NY: usize = 16;
const N: usize = NX * NY;

#[derive(Clone, Debug, Serialize)]
pub struct DirectionMeshReport {
    pub model: &'static str,
    pub grid: [usize; 2],
    pub bounds: [f64; 4],
    pub bars: usize,
    pub stems: usize,
    pub boundary_columns: usize,
    pub staff_edge_paths: usize,
    pub spacing_samples: usize,
    pub before_rms_direction_degrees: f64,
    pub after_rms_direction_degrees: f64,
    pub minimum_x_derivative_bound: f64,
    pub maximum_x_derivative_bound: f64,
    pub maximum_y_shear_bound: f64,
    pub accepted_scale: f64,
    pub note: &'static str,
}

#[derive(Clone, Debug)]
pub struct DirectionMesh {
    bounds: [f64; 4],
    controls: [f64; N],
    report: DirectionMeshReport,
}

impl Default for DirectionMesh {
    fn default() -> Self {
        Self {
            bounds: [0., 0., 1., 1.],
            controls: [0.; N],
            report: DirectionMeshReport {
                model: "coarse_cubic_bspline_x_field",
                grid: [NX, NY],
                bounds: [0., 0., 1., 1.],
                bars: 0,
                stems: 0,
                boundary_columns: 0,
                staff_edge_paths: 0,
                spacing_samples: 0,
                before_rms_direction_degrees: 0.,
                after_rms_direction_degrees: 0.,
                minimum_x_derivative_bound: 1.,
                maximum_x_derivative_bound: 1.,
                maximum_y_shear_bound: 0.,
                accepted_scale: 0.,
                note: "No directional support; identity. This model does not recover unique page metric geometry.",
            },
        }
    }
}

/// Clamped cardinal cubic weights and physical derivatives. Repeated end
/// controls extend the field constantly outside the supported rectangle.
fn basis<const K: usize>(p: f64, lo: f64, hi: f64) -> ([f64; K], [f64; K]) {
    let h = (hi - lo) / (K - 1) as f64;
    let t = ((p - lo) / h).clamp(0., (K - 1) as f64);
    let i = t.floor() as isize;
    let f = t - i as f64;
    let w = [
        (1. - f).powi(3) / 6.,
        (3. * f.powi(3) - 6. * f * f + 4.) / 6.,
        (-3. * f.powi(3) + 3. * f * f + 3. * f + 1.) / 6.,
        f.powi(3) / 6.,
    ];
    let d = [
        -0.5 * (1. - f).powi(2),
        1.5 * f * f - 2. * f,
        -1.5 * f * f + f + 0.5,
        0.5 * f * f,
    ];
    let mut a = [0.; K];
    let mut b = [0.; K];
    for k in 0..4 {
        let j = (i + k as isize - 1).clamp(0, (K - 1) as isize) as usize;
        a[j] += w[k];
        if p >= lo && p <= hi {
            b[j] += d[k] / h;
        }
    }
    (a, b)
}

/// Inclusive index range holding every non-zero weight of [`basis`].
fn support<const K: usize>(p: f64, lo: f64, hi: f64) -> (usize, usize) {
    let h = (hi - lo) / (K - 1) as f64;
    let i = ((p - lo) / h).clamp(0., (K - 1) as f64).floor() as usize;
    (i.saturating_sub(1), (i + 2).min(K - 1))
}

#[derive(Clone)]
pub(crate) struct MeshRow {
    polynomials: [[f64; 4]; NX - 1],
    lo: f64,
    hi: f64,
    inverse_cell_width: f64,
}
impl MeshRow {
    fn displacement(&self, x: f64) -> (f64, f64) {
        let q = ((x - self.lo) * self.inverse_cell_width).clamp(0., (NX - 1) as f64);
        let i = (q.floor() as usize).min(NX - 2);
        let t = q - i as f64;
        let [a, b, c, d] = self.polynomials[i];
        let value = ((a * t + b) * t + c) * t + d;
        let derivative = if x < self.lo || x > self.hi {
            0.
        } else {
            (3. * a * t * t + 2. * b * t + c) * self.inverse_cell_width
        };
        (value, derivative)
    }
    pub fn inverse_x(&self, x: f64) -> f64 {
        let mut x = [x];
        self.inverse_xs(&mut x);
        x[0]
    }
    /// Replace every x by its inverse. One sample's Newton steps form a
    /// single dependency chain, so a few samples are stepped side by side;
    /// each takes exactly the steps it would take alone.
    pub fn inverse_xs(&self, xs: &mut [f64]) {
        const LANES: usize = 4;
        for xs in xs.chunks_mut(LANES) {
            let mut q = [0.; LANES];
            let mut active = [false; LANES];
            for ((q, active), x) in q.iter_mut().zip(&mut active).zip(xs.iter()) {
                *q = x - self.displacement(*x).0;
                *active = true;
            }
            // The certified derivative bound is positive over the entire
            // field, including the constant extension; Newton has no
            // alternative root.
            for _ in 0..10 {
                for ((q, active), x) in q.iter_mut().zip(&mut active).zip(xs.iter()) {
                    let (u, dx) = self.displacement(*q);
                    let step = (*q + u - x) / (1. + dx);
                    if *active {
                        *q -= step;
                        *active = step.abs() >= 1e-9 || step.is_nan();
                    }
                }
                if !active.contains(&true) {
                    break;
                }
            }
            xs.copy_from_slice(&q[..xs.len()]);
        }
    }
}
impl DirectionMesh {
    fn weights(&self, p: Point, derivative: bool) -> [f64; N] {
        let (x, dx) = basis::<NX>(p.x, self.bounds[0], self.bounds[2]);
        let (y, _) = basis::<NY>(p.y, self.bounds[1], self.bounds[3]);
        std::array::from_fn(|i| if derivative { dx[i % NX] } else { x[i % NX] } * y[i / NX])
    }
    /// `weights(p, derivative) · controls` over the 4x4 support only. Terms are
    /// visited in dense order, so the sum equals the dense product exactly.
    fn dot(&self, p: Point, derivative: bool) -> f64 {
        let (value, slope) = basis::<NX>(p.x, self.bounds[0], self.bounds[2]);
        let x = if derivative { slope } else { value };
        let (y, _) = basis::<NY>(p.y, self.bounds[1], self.bounds[3]);
        let (x0, x1) = support::<NX>(p.x, self.bounds[0], self.bounds[2]);
        let (y0, y1) = support::<NY>(p.y, self.bounds[1], self.bounds[3]);
        let mut sum = 0.;
        for (j, y) in y.iter().enumerate().take(y1 + 1).skip(y0) {
            for (i, x) in x.iter().enumerate().take(x1 + 1).skip(x0) {
                sum += x * y * self.controls[j * NX + i];
            }
        }
        sum
    }
    pub fn report(&self) -> &DirectionMeshReport {
        &self.report
    }
    pub fn forward(&self, p: Point) -> Point {
        if self.report.accepted_scale == 0. {
            return p;
        }
        Point {
            x: p.x + self.dot(p, false),
            y: p.y,
        }
    }
    pub(crate) fn inverse_row(&self, y: f64) -> Option<MeshRow> {
        if self.report.accepted_scale == 0. {
            return None;
        }
        let (w, _) = basis::<NY>(y, self.bounds[1], self.bounds[3]);
        let controls: [f64; NX] =
            std::array::from_fn(|x| (0..NY).map(|y| w[y] * self.controls[y * NX + x]).sum());
        let polynomials = std::array::from_fn(|i| {
            let p0 = controls[i.saturating_sub(1)];
            let p1 = controls[i];
            let p2 = controls[i + 1];
            let p3 = controls[(i + 2).min(NX - 1)];
            [
                (-p0 + 3. * p1 - 3. * p2 + p3) / 6.,
                (p0 - 2. * p1 + p2) / 2.,
                (-p0 + p2) / 2.,
                (p0 + 4. * p1 + p2) / 6.,
            ]
        });
        Some(MeshRow {
            polynomials,
            lo: self.bounds[0],
            hi: self.bounds[2],
            inverse_cell_width: (NX - 1) as f64 / (self.bounds[2] - self.bounds[0]),
        })
    }
    pub fn inverse(&self, p: Point) -> Point {
        Point {
            x: self.inverse_row(p.y).map_or(p.x, |r| r.inverse_x(p.x)),
            y: p.y,
        }
    }
    pub fn jacobian_det(&self, p: Point) -> f64 {
        1. + self.dot(p, true)
    }
    fn derivative_bounds(&self) -> (f64, f64) {
        let hx = (self.bounds[2] - self.bounds[0]) / (NX - 1) as f64;
        let hy = (self.bounds[3] - self.bounds[1]) / (NY - 1) as f64;
        let mut x = 0_f64;
        let mut y = 0_f64;
        for j in 0..NY {
            for i in 0..NX {
                if i + 1 < NX {
                    x = x.max(
                        (self.controls[j * NX + i + 1] - self.controls[j * NX + i]).abs() / hx,
                    );
                }
                if j + 1 < NY {
                    y = y.max(
                        (self.controls[(j + 1) * NX + i] - self.controls[j * NX + i]).abs() / hy,
                    );
                }
            }
        }
        (x, y)
    }

    /// Fit in already-flattened world coordinates, while observations retain
    /// original source coordinates. Separate families get bounded influence.
    pub fn fit(
        data: &Observations,
        map: &impl Fn(Point) -> Point,
        edges: &[StaffEdge],
        verticals: &[VerticalPath],
        bounds: [f64; 4],
    ) -> Self {
        Self::fit_with_metric(data, map, edges, verticals, bounds, &[], None)
    }
    pub(crate) fn fit_with_metric(
        data: &Observations,
        map: &impl Fn(Point) -> Point,
        edges: &[StaffEdge],
        verticals: &[VerticalPath],
        bounds: [f64; 4],
        metric: &[super::surface_metric::ScaleSample],
        prior: Option<&Self>,
    ) -> Self {
        let mut mesh = Self {
            bounds,
            ..Self::default()
        };
        mesh.report.bounds = bounds;
        if data.validate().is_err()
            || bounds.iter().any(|v| !v.is_finite())
            || bounds[2] <= bounds[0]
            || bounds[3] <= bounds[1]
        {
            return Self::default();
        }
        let spaces: Vec<_> = data
            .staves
            .iter()
            .map(|s| {
                let p = s.sample(0.5);
                (map(p[4]).y - map(p[0]).y) / 4.
            })
            .collect();
        let space = crate::math::median(spaces).max(1.);
        let mut paths: Vec<(Vec<Point>, f64, u8)> = vec![];
        for b in &data.barlines {
            let p: Vec<_> =
                b.x.iter()
                    .zip(&b.y)
                    .map(|(x, y)| map(Point { x: *x, y: *y }))
                    .collect();
            if valid(&p, space) {
                paths.push((resample(&p), 1., 0));
            }
        }
        mesh.report.bars = paths.len();
        for v in verticals {
            let p: Vec<_> = v.points.iter().copied().map(map).collect();
            if v.weight.is_finite() && v.weight > 0. && v.weight <= 1. && valid(&p, space) {
                paths.push((resample(&p), 0.3 * v.weight, 1));
            }
        }
        mesh.report.stems = paths.len() - mesh.report.bars;
        // Aligned starts/ends are geometry, not invented printed bars.
        for c in super::system_boundaries::detect(data, edges) {
            let p: Vec<_> = c.points.into_iter().map(map).collect();
            if valid(&p, space) {
                paths.push((resample(&p), 0.5, 2));
            }
        }
        mesh.report.boundary_columns = paths.len() - mesh.report.bars - mesh.report.stems;
        // Five independently measured endpoints identify a local printed edge
        // direction without claiming that different systems share an x column.
        let mut edge_staffs = std::collections::BTreeSet::new();
        let mut edge_sides = vec![];
        for edge in edges {
            let side = (
                edge.staff,
                edge.side == crate::geometry::StaffEdgeSide::Start,
            );
            if edge.staff >= data.staves.len()
                || edge_sides.contains(&side)
                || !edge.confidence.is_finite()
                || edge.confidence < 0.9
                || edge.points.len() != 5
                || edge.line_indices != [0, 1, 2, 3, 4]
                || !edge.points.windows(2).all(|p| p[1].y > p[0].y)
                || edge.points.iter().any(|p| {
                    !p.x.is_finite()
                        || !p.y.is_finite()
                        || p.x < 1.
                        || p.y < 1.
                        || p.x >= f64::from(data.width - 1)
                        || p.y >= f64::from(data.height - 1)
                })
            {
                continue;
            }
            let p = &edge.points;
            let source_space = (p[4].y - p[0].y) / 4.;
            let m = direction(p);
            let center_x = p.iter().map(|p| p.x).sum::<f64>() / 5.;
            let center_y = p.iter().map(|p| p.y).sum::<f64>() / 5.;
            let rms = (p
                .iter()
                .map(|p| (p.x - center_x - m * (p.y - center_y)).powi(2))
                .sum::<f64>()
                / 5.)
                .sqrt();
            if rms > (0.12 * source_space).max(0.5) {
                continue;
            }
            let mapped: Vec<_> = p.iter().copied().map(map).collect();
            if valid(&mapped, space) {
                paths.push((resample(&mapped), 1., 3));
                mesh.report.staff_edge_paths += 1;
                edge_staffs.insert(edge.staff);
                edge_sides.push(side);
            }
        }
        if mesh.report.bars + mesh.report.stems < 3 && edge_staffs.len() < 3 {
            return mesh;
        }
        let mut rows = vec![];
        // A common ruling supplies a common width correction. Preserve the
        // already fitted inter-row direction field, rather than fitting a
        // different enlargement independently to each staff's sample budget.
        if let Some(prior) = prior {
            for y in 0..NY - 1 {
                for x in 0..NX {
                    let k = y * NX + x;
                    let w = 0.5;
                    let mut a = [0.; N];
                    a[k] = -w;
                    a[k + NX] = w;
                    rows.push(Row {
                        a,
                        b: w * (prior.controls[k + NX] - prior.controls[k]) / space,
                        data: false,
                        normalizer: 1.,
                    });
                }
            }
        }
        for sample in metric {
            let w = 6. * sample.confidence.sqrt();
            rows.push(Row {
                a: mesh.weights(sample.point, true).map(|v| v * space * w),
                b: (sample.scale - 1.) * w,
                data: false,
                normalizer: 1.,
            });
        }
        // Each spatial cell has a bounded budget, so dense music does not
        // overwhelm a sparsely notated neighboring region.
        let cell = |p: Point| -> usize {
            let x = (((p.x - bounds[0]) / (bounds[2] - bounds[0]) * 3.).floor() as i32).clamp(0, 2);
            let y = (((p.y - bounds[1]) / (bounds[3] - bounds[1]) * 5.).floor() as i32).clamp(0, 4);
            (y * 3 + x) as usize
        };
        let mut counts = [[0usize; 15]; 4];
        for (p, _, family) in &paths {
            counts[*family as usize][cell(p[p.len() / 2])] += 1;
        }
        for (p, confidence, family) in &paths {
            let bs: Vec<_> = p.iter().map(|p| mesh.weights(*p, false)).collect();
            let mean: [f64; N] =
                std::array::from_fn(|k| bs.iter().map(|b| b[k]).sum::<f64>() / bs.len() as f64);
            let x = p.iter().map(|p| p.x).sum::<f64>() / p.len() as f64;
            let len = p.last().unwrap().y - p[0].y;
            let weight = if metric.is_empty() { 1. } else { 4. }
                * confidence
                * (len / space).min(16.).sqrt()
                / (p.len() as f64 * counts[*family as usize][cell(p[p.len() / 2])] as f64).sqrt();
            for (p, b) in p.iter().zip(bs) {
                rows.push(Row {
                    a: std::array::from_fn(|k| weight * (b[k] - mean[k])),
                    b: -weight * (p.x - x) / space,
                    data: metric.is_empty() || *family != 0,
                    normalizer: weight,
                });
            }
        }
        // Relative spacing supplies a weak local scale prior along each staff.
        // This is an approximate isotropy regularizer, not recovered metric
        // truth: perspective is fitted separately using all five line indices.
        for s in data.staves.iter().filter(|_| metric.is_empty()) {
            let samples: Vec<_> = (0..17)
                .filter_map(|i| {
                    let p = s.sample(0.05 + 0.9 * i as f64 / 16.);
                    let top = map(p[0]);
                    let bot = map(p[4]);
                    let center = map(p[2]);
                    let a = map(Point {
                        x: p[2].x - 0.1,
                        y: p[2].y,
                    });
                    let b = map(Point {
                        x: p[2].x + 0.1,
                        y: p[2].y,
                    });
                    let scale = (bot.y - top.y) / (p[4].y - p[0].y) / ((b.x - a.x) / 0.2);
                    (scale.is_finite() && scale > 0.).then_some((center, scale))
                })
                .collect();
            if samples.is_empty() {
                continue;
            }
            let mid = crate::math::median(samples.iter().map(|(_, s)| *s).collect());
            for (p, s) in samples {
                let desired = (s / mid - 1.).clamp(-0.12, 0.12);
                let w = 0.5;
                rows.push(Row {
                    a: mesh.weights(p, true).map(|v| v * space * w),
                    b: desired * w,
                    data: false,
                    normalizer: 1.,
                });
                mesh.report.spacing_samples += 1;
            }
        }
        // Thin-plate-like regularization on a coarse shared lattice. Controls
        // are in staff-space units; dense resampling cannot alter the balance.
        for y in 0..NY {
            for x in 0..NX {
                let k = y * NX + x;
                if x + 2 < NX {
                    add_regularizer(
                        &mut rows,
                        &[(k, 1.), (k + 1, -2.), (k + 2, 1.)],
                        if metric.is_empty() { 0.04 } else { 0.015 },
                    );
                }
                if y + 2 < NY {
                    add_regularizer(&mut rows, &[(k, 1.), (k + NX, -2.), (k + 2 * NX, 1.)], 0.04);
                }
                if x + 1 < NX && y + 1 < NY {
                    add_regularizer(
                        &mut rows,
                        &[(k, 1.), (k + 1, -1.), (k + NX, -1.), (k + NX + 1, 1.)],
                        0.04,
                    );
                }
                add_regularizer(&mut rows, &[(k, 1.)], 0.003);
            }
        }
        // Fix only the translation gauge, without pinning individual systems.
        rows.push(Row {
            a: [1. / N as f64; N],
            b: 0.,
            data: false,
            normalizer: 1.,
        });
        // Direction evidence alone permits horizontal scale collapse. Page
        // ratio fixes the mean width gauge, while local width may still vary.
        let left = Point {
            x: bounds[0],
            y: (bounds[1] + bounds[3]) / 2.,
        };
        let right = Point {
            x: bounds[2],
            y: left.y,
        };
        let mut width_gauge = [0.; N];
        for j in 0..9 {
            let y = bounds[1] + (bounds[3] - bounds[1]) * j as f64 / 8.;
            let a = mesh.weights(Point { y, ..left }, false);
            let b = mesh.weights(Point { y, ..right }, false);
            for k in 0..N {
                width_gauge[k] += 4. * (b[k] - a[k]) / 9.;
            }
        }
        rows.push(Row {
            a: width_gauge,
            b: 0.,
            data: false,
            normalizer: 1.,
        });
        let mut coefficients = [0.; N];
        for _ in 0..5 {
            let mut a = [[0.; N]; N];
            let mut b = [0.; N];
            for row in &rows {
                let residual = row
                    .a
                    .iter()
                    .zip(coefficients)
                    .map(|(a, c)| a * c)
                    .sum::<f64>()
                    - row.b;
                let residual = residual / row.normalizer.max(1e-12);
                let weight = if row.data {
                    0.12 / residual.abs().max(0.12)
                } else {
                    1.
                };
                // Rows touch a few neighbouring controls; zero terms add nothing.
                let mut active = [0; N];
                let mut count = 0;
                for (i, value) in row.a.iter().enumerate() {
                    if *value != 0. {
                        active[count] = i;
                        count += 1;
                    }
                }
                let active = &active[..count];
                for &i in active {
                    b[i] += weight * row.a[i] * row.b;
                    for &j in active {
                        a[i][j] += weight * row.a[i] * row.a[j];
                    }
                }
            }
            let Some(c) = solve(a, b) else {
                return mesh;
            };
            coefficients = c;
        }
        mesh.controls = coefficients.map(|v| v * space);
        mesh.report.accepted_scale = 1.;
        let (dx, dy) = mesh.derivative_bounds();
        let amplitude = mesh.controls.iter().map(|c| c.abs()).fold(0_f64, f64::max);
        let max_dx: f64 = if metric.is_empty() { 0.12 } else { 0.45 };
        let limit = (max_dx / dx.max(max_dx))
            .min(0.18 / dy.max(0.18))
            .min((0.06 * (bounds[2] - bounds[0])) / amplitude.max(1e-9))
            .min(1.);
        mesh.controls.iter_mut().for_each(|c| *c *= limit);
        for (p, w, family) in &mut paths {
            *w /= (counts[*family as usize][cell(p[p.len() / 2])] as f64).sqrt();
        }
        let before = rms(&paths, |p| p);
        let mut scale = 1.;
        // Never use a direction-only gain to sacrifice the long bar evidence.
        while scale > 1. / 64. {
            let bar_worse = paths.iter().filter(|(_, _, f)| *f == 0).any(|(p, _, _)| {
                let q: Vec<_> = p.iter().map(|p| mesh.forward(*p)).collect();
                direction(&q).abs() > direction(p).abs() + 0.00175
            });
            let after = rms(&paths, |p| mesh.forward(p));
            let metric_gain = !metric.is_empty() && {
                let error = |corrected: bool| {
                    metric
                        .iter()
                        .map(|s| {
                            let scale = if corrected {
                                mesh.jacobian_det(s.point)
                            } else {
                                1.
                            };
                            s.confidence * (scale - s.scale).powi(2)
                        })
                        .sum::<f64>()
                };
                error(true) < error(false) * 0.9
            };
            // Already-vertical bars cannot get straighter. A metric-only gain
            // is valid when their direction remains within 0.05 degrees.
            if !bar_worse
                && (after < before || (metric_gain && after <= before + 0.05_f64.to_radians()))
            {
                break;
            }
            mesh.controls.iter_mut().for_each(|c| *c *= 0.5);
            scale *= 0.5;
        }
        if scale <= 1. / 64. {
            mesh.controls = [0.; N];
            scale = 0.;
        }
        let (dx, dy) = mesh.derivative_bounds();
        mesh.report.before_rms_direction_degrees = before.to_degrees();
        mesh.report.after_rms_direction_degrees = rms(&paths, |p| mesh.forward(p)).to_degrees();
        mesh.report.minimum_x_derivative_bound = 1. - dx;
        mesh.report.maximum_x_derivative_bound = 1. + dx;
        mesh.report.maximum_y_shear_bound = dy;
        mesh.report.accepted_scale = limit * scale;
        mesh.report.note = if metric.is_empty() {
            "Common x-only field; y is exactly fixed. Relative staff spacing is a weak local scale prior, not a unique metric reconstruction. Residuals use fit observations, not independent quality ground truth."
        } else {
            "Common x-only field; y is exactly fixed. Bounded relative width evidence from orthogonal print directions and a common camera-space vertical ruling. Focal length is a prior. Residuals use fit observations, not independent quality ground truth."
        };
        mesh
    }
}

struct Row {
    a: [f64; N],
    b: f64,
    data: bool,
    normalizer: f64,
}
fn add_regularizer(rows: &mut Vec<Row>, entries: &[(usize, f64)], weight: f64) {
    let mut a = [0.; N];
    for (i, v) in entries {
        a[*i] = v * weight;
    }
    rows.push(Row {
        a,
        b: 0.,
        data: false,
        normalizer: 1.,
    });
}
fn valid(p: &[Point], space: f64) -> bool {
    p.len() >= 2
        && p.iter().all(|p| p.x.is_finite() && p.y.is_finite())
        && p.windows(2).all(|p| p[1].y > p[0].y)
        && p.last().unwrap().y - p[0].y >= 1.5 * space
}
fn resample(p: &[Point]) -> Vec<Point> {
    (0..9)
        .map(|j| {
            let y = p[0].y + (p.last().unwrap().y - p[0].y) * j as f64 / 8.;
            let i = p.partition_point(|p| p.y < y).clamp(1, p.len() - 1);
            let t = (y - p[i - 1].y) / (p[i].y - p[i - 1].y);
            Point {
                x: p[i - 1].x + t * (p[i].x - p[i - 1].x),
                y,
            }
        })
        .collect()
}
fn direction(p: &[Point]) -> f64 {
    let n = p.len() as f64;
    let x = p.iter().map(|p| p.x).sum::<f64>() / n;
    let y = p.iter().map(|p| p.y).sum::<f64>() / n;
    p.iter().map(|p| (p.x - x) * (p.y - y)).sum::<f64>()
        / p.iter().map(|p| (p.y - y).powi(2)).sum::<f64>().max(1e-12)
}
fn rms(paths: &[(Vec<Point>, f64, u8)], map: impl Fn(Point) -> Point) -> f64 {
    let sw = paths.iter().map(|(_, w, _)| w * w).sum::<f64>().max(1e-12);
    (paths
        .iter()
        .map(|(p, w, _)| {
            let q: Vec<_> = p.iter().copied().map(&map).collect();
            w * w * direction(&q).powi(2)
        })
        .sum::<f64>()
        / sw)
        .sqrt()
}
fn solve(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for k in 0..N {
        let p = (k..N).max_by(|i, j| a[*i][k].abs().total_cmp(&a[*j][k].abs()))?;
        if a[p][k].abs() < 1e-14 {
            return None;
        }
        a.swap(k, p);
        b.swap(k, p);
        for i in k + 1..N {
            let t = a[i][k] / a[k][k];
            let row = a[k];
            for j in k..N {
                a[i][j] -= t * row[j];
            }
            b[i] -= t * b[k];
        }
    }
    let mut x = [0.; N];
    for i in (0..N).rev() {
        x[i] = (b[i] - (i + 1..N).map(|j| a[i][j] * x[j]).sum::<f64>()) / a[i][i];
    }
    Some(x)
}
