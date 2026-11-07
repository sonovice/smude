//! Evidence-gated x-only correction shared by whole connected systems.
//! All supported models are affine in x at each y: exact inverse and y identity.
use crate::geometry::{Observations, Point};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Band {
    pub staves: Vec<usize>,
    pub model: &'static str,
    pub reason: &'static str,
    pub bounds: [f64; 2],
    pub support: [f64; 2],
    pub center: Point,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub paths: usize,
    pub before_rms_slope: f64,
    pub after_rms_slope: f64,
}

#[derive(Clone, Default, Debug, Serialize)]
pub struct VerticalCorrection {
    pub bands: Vec<Band>,
}

fn slope(p: &[Point]) -> f64 {
    let cy = p.iter().map(|p| p.y).sum::<f64>() / p.len() as f64;
    let cx = p.iter().map(|p| p.x).sum::<f64>() / p.len() as f64;
    p.iter().map(|p| (p.x - cx) * (p.y - cy)).sum::<f64>()
        / p.iter().map(|p| (p.y - cy).powi(2)).sum::<f64>().max(1e-12)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    (v[(v.len() - 1) / 2] + v[v.len() / 2]) / 2.
}

fn root(parent: &[usize], mut i: usize) -> usize {
    while parent[i] != i {
        i = parent[i];
    }
    i
}

fn bend(points: &[Point]) -> f64 {
    let m = slope(points);
    let center = points.iter().map(|p| p.x - m * p.y).sum::<f64>() / points.len() as f64;
    (points
        .iter()
        .map(|p| (p.x - m * p.y - center).powi(2))
        .sum::<f64>()
        / points.len() as f64)
        .sqrt()
}

fn solve_four(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> Option<[f64; 4]> {
    for k in 0..4 {
        let pivot = (k..4).max_by(|i, j| a[*i][k].abs().total_cmp(&a[*j][k].abs()))?;
        if a[pivot][k].abs() < 1e-12 {
            return None;
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        for i in k + 1..4 {
            let t = a[i][k] / a[k][k];
            let pivot_row = a[k];
            for (value, pivot_value) in a[i][k..].iter_mut().zip(&pivot_row[k..]) {
                *value -= t * pivot_value;
            }
            b[i] -= t * b[k];
        }
    }
    let mut x = [0.; 4];
    for i in (0..4).rev() {
        x[i] = (b[i] - (i + 1..4).map(|j| a[i][j] * x[j]).sum::<f64>()) / a[i][i];
    }
    Some(x)
}

/// A common affine-x field with quadratic height variation. The latent x of
/// each whole bar is Schur-eliminated in damped Gauss-Newton steps. Four smooth
/// coefficients govern the entire system, never one shift per bar.
fn fit_height_dependent(
    paths: &[Vec<Point>],
    center: Point,
    xscale: f64,
    yscale: f64,
    space: f64,
) -> Option<[f64; 4]> {
    if paths.len() < 3 || xscale <= 0. || yscale < 4. * space {
        return None;
    }
    let middle: Vec<_> = paths
        .iter()
        .map(|p| (p[0].y + p.last().unwrap().y) / 2.)
        .collect();
    let spread = middle.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - middle.iter().copied().fold(f64::INFINITY, f64::min);
    let full: Vec<_> = paths
        .iter()
        .filter(|p| p.last().unwrap().y - p[0].y >= 1.6 * yscale)
        .collect();
    let full_x: Vec<_> = full
        .iter()
        .map(|p| p.iter().map(|p| p.x).sum::<f64>() / p.len() as f64)
        .collect();
    let full_spread = full_x.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - full_x.iter().copied().fold(f64::INFINITY, f64::min);
    let broad_full_support = full.len() >= 3 && full_spread >= 8. * space;
    if !broad_full_support && (paths.len() < 5 || spread < 0.3 * yscale) {
        return None;
    }
    let samples: Vec<Vec<Point>> = paths
        .iter()
        .map(|p| (0..16).map(|i| p[i * (p.len() - 1) / 15]).collect())
        .collect();
    let mut latent: Vec<_> = samples
        .iter()
        .map(|p| p.iter().map(|p| (p.x - center.x) / xscale).sum::<f64>() / p.len() as f64)
        .collect();
    let cost = |coeff: &[f64; 4], positions: &[f64]| -> f64 {
        samples
            .iter()
            .enumerate()
            .map(|(j, points)| {
                points
                    .iter()
                    .map(|p| {
                        let t = (p.y - center.y) / yscale;
                        let prediction = positions[j] * (1. + coeff[2] * t + coeff[3] * t * t)
                            + coeff[0] * t
                            + coeff[1] * t * t;
                        ((p.x - center.x) / xscale - prediction).powi(2) / 16.
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
    };
    let mut coefficients = [0.; 4];
    for _ in 0..20 {
        let mut normal = [[0.; 4]; 4];
        let mut rhs = [0.; 4];
        let mut blocks = vec![([0.; 4], 0., 0.); samples.len()];
        for (j, points) in samples.iter().enumerate() {
            for p in points {
                let t = (p.y - center.y) / yscale;
                let row = [t, t * t, latent[j] * t, latent[j] * t * t];
                let derivative = 1. + coefficients[2] * t + coefficients[3] * t * t;
                let residual = (p.x - center.x) / xscale
                    - latent[j] * derivative
                    - coefficients[0] * t
                    - coefficients[1] * t * t;
                for (r, v) in row.iter().enumerate() {
                    rhs[r] += v * residual / 16.;
                    blocks[j].0[r] += v * derivative / 16.;
                    for (c, w) in row.iter().enumerate() {
                        normal[r][c] += v * w / 16.;
                    }
                }
                blocks[j].1 += derivative * derivative / 16.;
                blocks[j].2 += derivative * residual / 16.;
            }
        }
        // Eliminate latent constant bar positions: only four shared shape
        // parameters remain, never a separate local warp for every bar.
        for (coupling, diagonal, gradient) in &blocks {
            if *diagonal < 1e-12 {
                return None;
            }
            for (r, v) in coupling.iter().enumerate() {
                rhs[r] -= v * gradient / diagonal;
                for (c, w) in coupling.iter().enumerate() {
                    normal[r][c] -= v * w / diagonal;
                }
            }
        }
        for (i, row) in normal.iter_mut().enumerate() {
            row[i] += 1e-8;
            rhs[i] -= 1e-8 * coefficients[i];
        }
        let delta = solve_four(normal, rhs)?;
        let latent_delta: Vec<_> = blocks
            .iter()
            .map(|(coupling, diagonal, gradient)| {
                (gradient - coupling.iter().zip(delta).map(|(a, b)| a * b).sum::<f64>()) / diagonal
            })
            .collect();
        let before = cost(&coefficients, &latent);
        let mut accepted = false;
        let mut scale = 1.;
        for _ in 0..10 {
            let candidate = std::array::from_fn(|i| coefficients[i] + scale * delta[i]);
            let positions: Vec<_> = latent
                .iter()
                .zip(&latent_delta)
                .map(|(u, d)| u + scale * d)
                .collect();
            if cost(&candidate, &positions) <= before + 1e-18 {
                coefficients = candidate;
                latent = positions;
                accepted = true;
                break;
            }
            scale *= 0.5;
        }
        if !accepted || delta.iter().all(|v| v.abs() < 1e-12) {
            break;
        }
    }
    coefficients.iter().all(|x| x.is_finite()).then_some([
        coefficients[0] * xscale / yscale,
        coefficients[2] / yscale,
        coefficients[1] * xscale / (yscale * yscale),
        coefficients[3] / (yscale * yscale),
    ])
}

impl Band {
    fn displacement(&self, y: f64) -> (f64, f64) {
        if y <= self.support[0] || y >= self.support[1] || self.model == "identity" {
            return (0., 0.);
        }
        let u = if y < self.bounds[0] {
            (y - self.support[0]) / (self.bounds[0] - self.support[0])
        } else if y > self.bounds[1] {
            (self.support[1] - y) / (self.support[1] - self.bounds[1])
        } else {
            1.
        };
        let weight = u * u * (3. - 2. * u);
        let t = y - self.center.y;
        (
            (self.a * t + self.c * t * t) * weight,
            (self.b * t + self.d * t * t) * weight,
        )
    }
    fn forward(&self, p: Point) -> Point {
        let (a, b) = self.displacement(p.y);
        Point {
            x: self.center.x + (p.x - self.center.x - a) / (1. + b),
            y: p.y,
        }
    }
}

impl VerticalCorrection {
    pub fn forward(&self, p: Point) -> Point {
        self.bands
            .iter()
            .find(|b| p.y > b.support[0] && p.y < b.support[1])
            .map_or(p, |b| b.forward(p))
    }
    pub fn x_scale(&self, p: Point) -> f64 {
        self.bands
            .iter()
            .find(|b| p.y > b.support[0] && p.y < b.support[1])
            .map_or(1., |b| 1. / (1. + b.displacement(p.y).1))
    }
    pub fn jacobian_det(&self, p: Point) -> f64 {
        self.x_scale(p)
    }
    /// All x coefficients depend only on output y, so rendering can reuse them
    /// for the complete scanline without approximating the inverse map.
    #[cfg(feature = "render")]
    pub(crate) fn inverse_row(&self, y: f64) -> Option<(f64, f64, f64)> {
        self.bands
            .iter()
            .find(|b| y > b.support[0] && y < b.support[1])
            .map(|b| {
                let (a, scale_delta) = b.displacement(y);
                (b.center.x, 1. + scale_delta, a)
            })
    }
    pub fn inverse(&self, p: Point) -> Point {
        self.bands
            .iter()
            .find(|b| p.y > b.support[0] && p.y < b.support[1])
            .map_or(p, |b| {
                let (a, scale_delta) = b.displacement(p.y);
                Point {
                    x: b.center.x + (p.x - b.center.x) * (1. + scale_delta) + a,
                    y: p.y,
                }
            })
    }
    /// Fit on validated source observations mapped through the already-fitted
    /// page/staff geometry. `input_width` is a world-coordinate span, not a
    /// coordinate origin. The optional stage returns identity when observations
    /// cannot support a bounded common correction.
    pub fn fit(
        data: &Observations,
        map: &impl Fn(Point) -> Point,
        target_staff_spacings: &[f64],
        input_width: f64,
    ) -> Self {
        if data.validate().is_err()
            || target_staff_spacings.is_empty()
            || !input_width.is_finite()
            || input_width <= 0.
            || target_staff_spacings
                .iter()
                .any(|s| !s.is_finite() || *s <= 0.)
        {
            return Self::default();
        }
        let mut parent: Vec<_> = (0..data.staves.len()).collect();
        // These memberships come from a measured continuous cross-staff ink
        // path, not coincident x positions of disconnected short strokes.
        for bar in &data.barlines {
            for other in &bar.aligned_staffs {
                let a = root(&parent, bar.staff);
                let b = root(&parent, *other);
                parent[b] = a;
            }
        }
        let mut groups: Vec<Vec<usize>> = vec![];
        for i in 0..parent.len() {
            let r = root(&parent, i);
            if let Some(g) = groups.iter_mut().find(|g| root(&parent, g[0]) == r) {
                g.push(i);
            } else {
                groups.push(vec![i]);
            }
        }
        let mut output = Self::default();
        for group in groups {
            let space = if target_staff_spacings.len() == data.staves.len() {
                median(group.iter().map(|i| target_staff_spacings[*i]).collect())
            } else {
                median(target_staff_spacings.to_vec())
            };
            let staff_points: Vec<_> = group
                .iter()
                .flat_map(|i| {
                    let s = &data.staves[*i];
                    s.y.iter()
                        .flat_map(|y| s.x.iter().zip(y).map(|(x, y)| map(Point { x: *x, y: *y })))
                        .collect::<Vec<_>>()
                })
                .collect();
            let lo = staff_points
                .iter()
                .map(|p| p.y)
                .fold(f64::INFINITY, f64::min);
            let hi = staff_points
                .iter()
                .map(|p| p.y)
                .fold(f64::NEG_INFINITY, f64::max);
            let paths: Vec<Vec<Point>> = data
                .barlines
                .iter()
                .filter(|b| group.contains(&b.staff))
                .map(|b| {
                    b.x.iter()
                        .zip(&b.y)
                        .map(|(x, y)| map(Point { x: *x, y: *y }))
                        .collect::<Vec<_>>()
                })
                .filter(|p| p.last().unwrap().y - p[0].y >= 2. * space)
                .collect();
            let center = Point {
                x: (staff_points
                    .iter()
                    .map(|p| p.x)
                    .fold(f64::INFINITY, f64::min)
                    + staff_points
                        .iter()
                        .map(|p| p.x)
                        .fold(f64::NEG_INFINITY, f64::max))
                    / 2.,
                y: (lo + hi) / 2.,
            };
            let mut band = Band {
                staves: group,
                model: "identity",
                reason: "insufficient independent bars",
                bounds: [lo, hi],
                support: [lo, hi],
                center,
                a: 0.,
                b: 0.,
                c: 0.,
                d: 0.,
                paths: paths.len(),
                before_rms_slope: 0.,
                after_rms_slope: 0.,
            };
            if paths.len() < 2 {
                output.bands.push(band);
                continue;
            }
            let records: Vec<_> = paths
                .iter()
                .map(|p| {
                    (
                        p.iter().map(|p| p.x).sum::<f64>() / p.len() as f64 - center.x,
                        slope(p),
                        ((p.last().unwrap().y - p[0].y) / space).min(16.).powi(2),
                    )
                })
                .collect();
            let before = (records.iter().map(|(_, m, w)| w * m * m).sum::<f64>()
                / records.iter().map(|r| r.2).sum::<f64>())
            .sqrt();
            band.before_rms_slope = before;
            band.after_rms_slope = before;
            let long: Vec<_> = records.iter().copied().filter(|r| r.2 >= 36.).collect();
            let fitting = if long.len() >= 3 { &long } else { &records };
            let a = median(fitting.iter().map(|r| r.1).collect());
            let mut candidates = vec![(a, 0., 0., 0., "shared_shear")];
            let xmin = fitting.iter().map(|r| r.0).fold(f64::INFINITY, f64::min);
            let xmax = fitting
                .iter()
                .map(|r| r.0)
                .fold(f64::NEG_INFINITY, f64::max);
            if fitting.len() >= 3 && xmax - xmin >= 8. * space {
                let fit = |omit: Option<usize>| {
                    let v: Vec<_> = fitting
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| Some(*i) != omit)
                        .map(|(_, r)| *r)
                        .collect();
                    let sw = v.iter().map(|r| r.2).sum::<f64>();
                    let x = v.iter().map(|r| r.0 * r.2).sum::<f64>() / sw;
                    let y = v.iter().map(|r| r.1 * r.2).sum::<f64>() / sw;
                    let b = v.iter().map(|r| r.2 * (r.0 - x) * (r.1 - y)).sum::<f64>()
                        / v.iter()
                            .map(|r| r.2 * (r.0 - x).powi(2))
                            .sum::<f64>()
                            .max(1e-12);
                    (y - b * x, b)
                };
                let mut linear = 0.;
                let mut constant = 0.;
                for (i, r) in fitting.iter().enumerate() {
                    let (a, b) = fit(Some(i));
                    linear += (r.1 - a - b * r.0).powi(2);
                    let med = median(
                        fitting
                            .iter()
                            .enumerate()
                            .filter(|(j, _)| *j != i)
                            .map(|(_, r)| r.1)
                            .collect(),
                    );
                    constant += (r.1 - med).powi(2);
                }
                if linear < 0.7 * constant {
                    let (a, b) = fit(None);
                    candidates.push((a, b, 0., 0., "affine_across_width"));
                }
            }
            if let Some([a, b, c, d]) =
                fit_height_dependent(&paths, center, input_width / 2., (hi - lo) / 2., space)
            {
                candidates.push((a, b, c, d, "affine_across_width_and_height"));
            }
            let mut best = before;
            for (a, b, c, d, model) in candidates {
                // Bounded width change and slope over the full input width,
                // including transition support, not only at fitted bars.
                let radius = (hi - lo) / 2. + 3. * space;
                if b.abs() * radius + d.abs() * radius * radius > 0.019
                    || a.abs()
                        + 2. * c.abs() * radius
                        + (b.abs() + 2. * d.abs() * radius) * input_width / 2.
                        > 0.08
                {
                    continue;
                }
                let mut candidate = band.clone();
                candidate.model = model;
                candidate.a = a;
                candidate.b = b;
                candidate.c = c;
                candidate.d = d;
                candidate.support = [lo - 3. * space, hi + 3. * space];
                let after = (paths
                    .iter()
                    .zip(&records)
                    .map(|(p, r)| {
                        let q: Vec<_> = p.iter().map(|p| candidate.forward(*p)).collect();
                        r.2 * slope(&q).powi(2)
                    })
                    .sum::<f64>()
                    / records.iter().map(|r| r.2).sum::<f64>())
                .sqrt();
                let worsened = paths.iter().any(|p| {
                    let q: Vec<_> = p.iter().map(|p| candidate.forward(*p)).collect();
                    slope(&q).abs() > slope(p).abs() + 0.00524 || bend(&q) > bend(p) + 0.25
                });
                if !worsened && after < before * 0.8 && after < best && before > 0.00087 {
                    best = after;
                    candidate.after_rms_slope = after;
                    candidate.reason = "multiple bars agree; residual reduced";
                    band = candidate;
                }
            }
            if band.model == "identity" {
                band.reason = "inconsistent directions or no bounded improvement";
            }
            output.bands.push(band);
        }
        output
            .bands
            .sort_by(|a, b| a.bounds[0].total_cmp(&b.bounds[0]));
        // Independent columns / overlapping vertical domains cannot share this
        // x-affine per-y map. Abstain instead of forcing an arbitrary ordering.
        let mut conflict = vec![false; output.bands.len()];
        for i in 0..output.bands.len() {
            for j in i + 1..output.bands.len() {
                if output.bands[i].bounds[1] >= output.bands[j].bounds[0] {
                    conflict[i] = true;
                    conflict[j] = true;
                }
            }
        }
        for (i, b) in output.bands.iter_mut().enumerate() {
            if conflict[i] {
                b.model = "identity";
                b.reason = "overlapping system domains";
                b.a = 0.;
                b.b = 0.;
                b.c = 0.;
                b.d = 0.;
                b.support = b.bounds;
            }
        }
        for i in 1..output.bands.len() {
            let mid = (output.bands[i - 1].bounds[1] + output.bands[i].bounds[0]) / 2.;
            output.bands[i - 1].support[1] = output.bands[i - 1].support[1].min(mid);
            output.bands[i].support[0] = output.bands[i].support[0].max(mid);
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Config, Geometry};
    fn correction() -> VerticalCorrection {
        VerticalCorrection {
            bands: vec![Band {
                staves: vec![0],
                model: "affine_across_width",
                reason: "test",
                bounds: [100., 300.],
                support: [60., 340.],
                center: Point { x: 500., y: 200. },
                a: 0.02,
                b: 0.00005,
                c: 0.,
                d: 0.,
                paths: 4,
                before_rms_slope: 0.02,
                after_rms_slope: 0.,
            }],
        }
    }
    #[test]
    fn exact_inverse_and_y_identity_including_transitions() {
        let c = correction();
        for y in 0..400 {
            for x in [0., 200., 1000.] {
                let p = Point { x, y: y as f64 };
                let q = c.forward(p);
                let r = c.inverse(q);
                assert_eq!(p.y, q.y);
                assert!((p.x - r.x).abs() < 1e-10);
            }
        }
    }
    #[test]
    fn linear_family_becomes_vertical_without_local_wobble() {
        let c = correction();
        for u in [-400., 0., 400.] {
            let mapped: Vec<_> = (100..301)
                .map(|y| {
                    let t = y as f64 - 200.;
                    c.forward(Point {
                        x: 500. + u + (0.02 + 0.00005 * u) * t,
                        y: y as f64,
                    })
                })
                .collect();
            for p in mapped {
                assert!((p.x - 500. - u).abs() < 1e-10);
            }
        }
    }
    fn observations(a: f64, b: f64) -> Observations {
        use crate::geometry::{Barline, Staff};
        let staves = (0..2)
            .map(|i| Staff {
                x: vec![50., 950.],
                y: std::array::from_fn(|k| vec![100. + i as f64 * 140. + k as f64 * 10.; 2]),
            })
            .collect();
        let barlines = [100., 500., 900.]
            .iter()
            .map(|x| {
                let y: Vec<_> = (100..281).step_by(5).map(|y| y as f64).collect();
                let m = a + b * (x - 500.);
                Barline {
                    staff: 0,
                    aligned_staffs: vec![1],
                    x: y.iter().map(|y| x + m * (y - 190.)).collect(),
                    y,
                }
            })
            .collect();
        Observations {
            width: 1000,
            height: 500,
            staves,
            barlines,
        }
    }
    #[test]
    fn coherent_width_dependent_family_is_fitted_from_multiple_full_bars() {
        let data = observations(0.02, 0.00003);
        let geometry = Geometry::fit(
            &data,
            &Config {
                perspective: false,
                ..Default::default()
            },
        )
        .unwrap();
        let c = VerticalCorrection::fit(
            &data,
            &|p| geometry.forward(p),
            &geometry.report().target_staff_spacings,
            data.width as f64,
        );
        assert_eq!(c.bands.len(), 1);
        assert_eq!(c.bands[0].model, "affine_across_width");
        for bar in &data.barlines {
            let p: Vec<_> = bar
                .x
                .iter()
                .zip(&bar.y)
                .map(|(x, y)| c.forward(geometry.forward(Point { x: *x, y: *y })))
                .collect();
            assert!(slope(&p).abs() < 1e-8);
        }
    }
    #[test]
    fn one_bar_does_not_enable_a_system_correction() {
        let mut data = observations(0.04, 0.);
        data.barlines.truncate(1);
        let geometry = Geometry::fit(
            &data,
            &Config {
                perspective: false,
                ..Default::default()
            },
        )
        .unwrap();
        let c = VerticalCorrection::fit(
            &data,
            &|p| geometry.forward(p),
            &geometry.report().target_staff_spacings,
            data.width as f64,
        );
        assert!(c.bands.iter().all(|b| b.model == "identity"));
    }
    #[test]
    fn contradictory_directions_at_same_x_are_not_forced_into_a_shear() {
        let mut data = observations(0., 0.);
        let original = data.barlines[1].clone();
        data.barlines.clear();
        for m in [-0.04, 0.04] {
            let mut bar = original.clone();
            bar.x = bar.y.iter().map(|y| 500. + m * (y - 190.)).collect();
            data.barlines.push(bar);
        }
        let geometry = Geometry::fit(
            &data,
            &Config {
                perspective: false,
                ..Default::default()
            },
        )
        .unwrap();
        let c = VerticalCorrection::fit(
            &data,
            &|p| geometry.forward(p),
            &geometry.report().target_staff_spacings,
            data.width as f64,
        );
        assert!(c.bands.iter().all(|b| b.model == "identity"));
    }
}
