//! Local text curvature residual with hard identity on protected music.
use crate::{Point, TextLine};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct TextWarpReport {
    pub detected_lines: usize,
    pub accepted_blocks: usize,
    pub fitted_lines: usize,
    pub max_displacement_pixels: f64,
    pub min_vertical_derivative: f64,
    pub music_identity: bool,
}
#[derive(Clone, Debug)]
pub(crate) struct TextWarp {
    bounds: [f64; 4],
    w: usize,
    h: usize,
    values: Vec<f64>,
}
fn smooth(t: f64) -> f64 {
    let t = t.clamp(0., 1.);
    t * t * (3. - 2. * t)
}
fn value(points: &[Point], x: f64) -> f64 {
    let i = points
        .partition_point(|p| p.x < x)
        .clamp(1, points.len() - 1);
    let (a, b) = (points[i - 1], points[i]);
    a.y + (b.y - a.y) * (x - a.x) / (b.x - a.x)
}
struct Constraint {
    residual: Vec<Point>,
    bounds: [f64; 4],
    pad: f64,
    weight: f64,
    block_id: usize,
}
impl TextWarp {
    pub fn fit(
        lines: &[TextLine],
        protected: &[[f64; 4]],
        bounds: [f64; 4],
    ) -> (Option<Self>, TextWarpReport) {
        let mut constraints = vec![];
        for line in lines {
            let p = &line.points;
            if p.len() < 3
                || !line.confidence.is_finite()
                || line.confidence <= 0.
                || !line.uncertainty_pixels.is_finite()
                || line.uncertainty_pixels < 0.
                || line.bounds.iter().any(|v| !v.is_finite())
                || p.iter().any(|p| !p.x.is_finite() || !p.y.is_finite())
                || p.windows(2).any(|q| q[1].x <= q[0].x)
            {
                continue;
            }
            let height = line.bounds[3] - line.bounds[1];
            let span = p.last().unwrap().x - p[0].x;
            if height <= 0. || span <= 1. {
                continue;
            }
            // Each line has its own unknown vertical offset. Its departures
            // from horizontal constrain only its observed span. In particular,
            // a short word contributes local tilt, never an extrapolated page curve.
            let mean_y = p.iter().map(|p| p.y).sum::<f64>() / p.len() as f64;
            let variance = p.iter().map(|p| (p.y - mean_y).powi(2)).sum::<f64>() / p.len() as f64;
            let noise = line.uncertainty_pixels.max(0.5);
            // Ridge towards identity: small departures relative to measurement
            // uncertainty have little influence, without a minimum word length.
            let weight = line.confidence.clamp(0., 1.) / (line.confidence + 0.1) * variance
                / (variance + noise * noise);
            constraints.push(Constraint {
                residual: p
                    .iter()
                    .map(|p| Point {
                        x: p.x,
                        y: mean_y - p.y,
                    })
                    .collect(),
                bounds: [p[0].x, line.bounds[1], p.last().unwrap().x, line.bounds[3]],
                pad: (height * 0.75).max(12.),
                weight,
                block_id: line.block_id,
            });
        }
        let mut ids: Vec<_> = constraints.iter().map(|c| c.block_id).collect();
        ids.sort_unstable();
        ids.dedup();
        let mut report = TextWarpReport {
            detected_lines: lines.len(),
            fitted_lines: constraints.len(),
            accepted_blocks: ids.len(),
            max_displacement_pixels: 0.,
            min_vertical_derivative: 1.,
            music_identity: true,
        };
        if constraints.is_empty() {
            return (None, report);
        }
        // Resolve small words without allocating a full-resolution displacement
        // field. Grid density is bounded and independent of output resolution.
        let min_span = constraints
            .iter()
            .map(|c| c.bounds[2] - c.bounds[0])
            .fold(f64::INFINITY, f64::min);
        let step = (min_span / 6.).max(1.);
        let w = (((bounds[2] - bounds[0]) / step).ceil() as usize + 1).clamp(97, 257);
        let h = (((bounds[3] - bounds[1]) / step).ceil() as usize + 1).clamp(145, 385);
        let dx = (bounds[2] - bounds[0]) / (w - 1) as f64;
        let dy = (bounds[3] - bounds[1]) / (h - 1) as f64;
        let mut values = vec![0.; w * h];
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let p = Point {
                    x: bounds[0] + x as f64 * dx,
                    y: bounds[1] + y as f64 * dy,
                };
                // One grid cell of protection makes interpolated displacement
                // exactly zero throughout the complete music guard.
                if protected.iter().any(|r| {
                    p.x >= r[0] - dx && p.x <= r[2] + dx && p.y >= r[1] - dy && p.y <= r[3] + dy
                }) {
                    continue;
                }
                let (mut sum, mut weight) = (0., 0.25); // identity prior
                for constraint in &constraints {
                    let b = constraint.bounds;
                    if p.x < b[0] || p.x > b[2] {
                        continue;
                    }
                    let pad = constraint.pad;
                    let wy = smooth((p.y - b[1] + pad) / pad) * smooth((b[3] + pad - p.y) / pad);
                    let x_pad = pad.min((b[2] - b[0]) * 0.2);
                    let wx = smooth((p.x - b[0]) / x_pad) * smooth((b[2] - p.x) / x_pad);
                    let local_weight = constraint.weight * wx * wy;
                    sum += local_weight * value(&constraint.residual, p.x);
                    weight += local_weight;
                }
                values[y * w + x] = sum / weight;
            }
        }
        // A global derivative bound includes transitions into music guards. It
        // prevents folds without relying on the text detector's confidence score.
        let mut derivative = 0_f64;
        for y in 0..h - 1 {
            for x in 0..w {
                derivative =
                    derivative.max((values[(y + 1) * w + x] - values[y * w + x]).abs() / dy);
            }
        }
        let mut horizontal = 0_f64;
        for y in 0..h {
            for x in 0..w - 1 {
                horizontal = horizontal.max((values[y * w + x + 1] - values[y * w + x]).abs() / dx);
            }
        }
        let amplitude = values.iter().map(|v| v.abs()).fold(0_f64, f64::max);
        let scale = (0.25 / derivative.max(1e-12))
            .min(0.15 / horizontal.max(1e-12))
            .min(0.02 * (bounds[2] - bounds[0]).min(bounds[3] - bounds[1]) / amplitude.max(1e-12))
            .min(1.);
        for v in &mut values {
            *v *= scale;
            report.max_displacement_pixels = report.max_displacement_pixels.max(v.abs());
        }
        report.min_vertical_derivative = 1. - derivative * scale;
        if report.max_displacement_pixels < 0.15 {
            report.accepted_blocks = 0;
            report.max_displacement_pixels = 0.;
            report.min_vertical_derivative = 1.;
            return (None, report);
        }
        (
            Some(Self {
                bounds,
                w,
                h,
                values,
            }),
            report,
        )
    }
    fn row(&self, x: f64, y: usize) -> f64 {
        let u = ((x - self.bounds[0]) / (self.bounds[2] - self.bounds[0]) * (self.w - 1) as f64)
            .clamp(0., (self.w - 1) as f64);
        let i = (u.floor() as usize).min(self.w - 2);
        let t = u - i as f64;
        self.values[y * self.w + i] * (1. - t) + self.values[y * self.w + i + 1] * t
    }
    pub fn forward(&self, p: Point) -> Point {
        if p.x < self.bounds[0]
            || p.x > self.bounds[2]
            || p.y < self.bounds[1]
            || p.y > self.bounds[3]
        {
            return p;
        }
        let v = (p.y - self.bounds[1]) / (self.bounds[3] - self.bounds[1]) * (self.h - 1) as f64;
        let j = (v.floor() as usize).min(self.h - 2);
        let t = v - j as f64;
        Point {
            x: p.x,
            y: p.y + self.row(p.x, j) * (1. - t) + self.row(p.x, j + 1) * t,
        }
    }
    pub(crate) fn affects_row(&self, y: f64) -> bool {
        if y < self.bounds[1] || y > self.bounds[3] {
            return false;
        }
        let v = (y - self.bounds[1]) / (self.bounds[3] - self.bounds[1]) * (self.h - 1) as f64;
        let j = (v.floor() as usize).min(self.h - 2);
        self.values[j * self.w..(j + 2) * self.w]
            .iter()
            .any(|v| *v != 0.)
    }
    pub fn inverse(&self, p: Point) -> Point {
        if self.forward(p).y == p.y {
            return p;
        }
        if p.x < self.bounds[0]
            || p.x > self.bounds[2]
            || p.y < self.bounds[1]
            || p.y > self.bounds[3]
        {
            return p;
        }
        let dy = (self.bounds[3] - self.bounds[1]) / (self.h - 1) as f64;
        let at = |j| self.bounds[1] + j as f64 * dy + self.row(p.x, j);
        let (mut lo, mut hi) = (0, self.h - 1);
        while hi - lo > 1 {
            let m = (lo + hi) / 2;
            if at(m) <= p.y {
                lo = m;
            } else {
                hi = m;
            }
        }
        Point {
            x: p.x,
            y: self.bounds[1] + lo as f64 * dy + (p.y - at(lo)) / (at(hi) - at(lo)) * dy,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn lines() -> Vec<TextLine> {
        (0..4)
            .map(|k| TextLine {
                points: (0..81)
                    .map(|j| {
                        let x = 100. + j as f64 * 10.;
                        Point {
                            x,
                            y: 200.
                                + k as f64 * 30.
                                + 15. * (std::f64::consts::PI * (x - 100.) / 800.).sin(),
                        }
                    })
                    .collect(),
                bounds: [100., 190. + k as f64 * 30., 900., 220. + k as f64 * 30.],
                block_id: 0,
                confidence: 0.8,
                uncertainty_pixels: 1.,
            })
            .collect()
    }
    #[test]
    fn a_single_short_word_contributes_with_uncertainty_and_local_support() {
        let short = TextLine {
            points: (0..9)
                .map(|i| Point {
                    x: 300. + i as f64 * 8.,
                    y: 200. + i as f64,
                })
                .collect(),
            bounds: [300., 194., 365., 218.],
            block_id: 0,
            confidence: 0.2,
            uncertainty_pixels: 0.5,
        };
        let guard = [[0., 300., 1000., 550.]];
        let (warp, report) =
            TextWarp::fit(std::slice::from_ref(&short), &guard, [0., 0., 1000., 700.]);
        let warp = warp.unwrap();
        assert_eq!(report.fitted_lines, 1);
        let a = short.points[2];
        let b = short.points[6];
        assert!(warp.forward(b).y - warp.forward(a).y < (b.y - a.y) * 0.8);
        let mut uncertain = short.clone();
        uncertain.uncertainty_pixels = 8.;
        let (_, weak) = TextWarp::fit(&[uncertain], &guard, [0., 0., 1000., 700.]);
        assert!(weak.max_displacement_pixels < report.max_displacement_pixels);
        for point in [
            Point { x: 100., y: 204. },
            Point { x: 500., y: 204. },
            Point { x: 330., y: 400. },
        ] {
            let q = warp.forward(point);
            assert_eq!((q.x, q.y), (point.x, point.y));
        }
    }

    #[test]
    fn text_curvature_improves_and_music_is_exact_identity() {
        let l = lines();
        let guards = [[0., 350., 1000., 550.]];
        let (w, r) = TextWarp::fit(&l, &guards, [0., 0., 1000., 700.]);
        let w = w.unwrap();
        assert!(r.accepted_blocks > 0);
        assert!(r.min_vertical_derivative >= 0.75);
        let raw = l[1].points.iter().map(|p| p.y).collect::<Vec<_>>();
        let corrected = l[1]
            .points
            .iter()
            .map(|p| w.forward(*p).y)
            .collect::<Vec<_>>();
        let span = |a: &[f64]| {
            a.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - a.iter().copied().fold(f64::INFINITY, f64::min)
        };
        assert!(span(&corrected) < span(&raw) * 0.85);
        for x in (0..1000).step_by(11) {
            for y in (350..550).step_by(7) {
                let p = Point {
                    x: x as f64,
                    y: y as f64,
                };
                let q = w.forward(p);
                assert_eq!(p.x, q.x);
                assert_eq!(p.y, q.y);
            }
        }
        for x in (0..1000).step_by(13) {
            for y in (0..700).step_by(9) {
                let p = Point {
                    x: x as f64,
                    y: y as f64,
                };
                let q = w.inverse(w.forward(p));
                assert!((p.y - q.y).abs() < 1e-9);
            }
        }
    }
    #[test]
    fn one_or_two_supported_lines_can_constrain_local_geometry() {
        let mut l = lines();
        l.truncate(2);
        assert!(TextWarp::fit(&l, &[], [0., 0., 1000., 700.]).0.is_some());
    }
}
