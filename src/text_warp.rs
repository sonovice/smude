//! Local text curvature residual with hard identity on protected music.
use crate::{Point, TextLine, median};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct TextWarpReport {
    pub detected_lines: usize,
    pub accepted_blocks: usize,
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
struct Block {
    lines: Vec<Vec<Point>>,
    bounds: [f64; 4],
    pad: f64,
}
impl TextWarp {
    pub fn fit(
        lines: &[TextLine],
        protected: &[[f64; 4]],
        bounds: [f64; 4],
    ) -> (Option<Self>, TextWarpReport) {
        let mut blocks = vec![];
        let mut ids: Vec<_> = lines.iter().map(|l| l.block_id).collect();
        ids.sort_unstable();
        ids.dedup();
        for id in ids {
            let group: Vec<_> = lines
                .iter()
                .filter(|l| l.block_id == id && l.confidence >= 0.25 && l.points.len() >= 5)
                .collect();
            if group.len() < 3 {
                continue;
            }
            let mut b = [
                f64::INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ];
            let mut curves = vec![];
            for line in group {
                let p = &line.points;
                let mx = p.iter().map(|p| p.x).sum::<f64>() / p.len() as f64;
                let my = p.iter().map(|p| p.y).sum::<f64>() / p.len() as f64;
                let den = p.iter().map(|p| (p.x - mx).powi(2)).sum::<f64>();
                if den <= 1. {
                    continue;
                }
                let slope = p.iter().map(|p| (p.x - mx) * (p.y - my)).sum::<f64>() / den;
                curves.push(
                    p.iter()
                        .map(|p| Point {
                            x: p.x,
                            y: my + slope * (p.x - mx) - p.y,
                        })
                        .collect::<Vec<_>>(),
                );
                for k in 0..2 {
                    b[k] = b[k].min(line.bounds[k]);
                    b[k + 2] = b[k + 2].max(line.bounds[k + 2]);
                }
            }
            if curves.len() < 3 {
                continue;
            }
            let pad = ((b[3] - b[1]) * 0.5).max(12.);
            blocks.push(Block {
                lines: curves,
                bounds: b,
                pad,
            });
        }
        let mut report = TextWarpReport {
            detected_lines: lines.len(),
            accepted_blocks: blocks.len(),
            max_displacement_pixels: 0.,
            min_vertical_derivative: 1.,
            music_identity: true,
        };
        if blocks.is_empty() {
            return (None, report);
        }
        let w = 97;
        let h = 145;
        let dx = (bounds[2] - bounds[0]) / (w - 1) as f64;
        let dy = (bounds[3] - bounds[1]) / (h - 1) as f64;
        let mut values = vec![0.; w * h];
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let p = Point {
                    x: bounds[0] + x as f64 * dx,
                    y: bounds[1] + y as f64 * dy,
                };
                // Expanding by one grid cell makes bilinear interpolation exactly zero
                // throughout every protected rectangle, including its boundary.
                if protected.iter().any(|r| {
                    p.x >= r[0] - dx && p.x <= r[2] + dx && p.y >= r[1] - dy && p.y <= r[3] + dy
                }) {
                    continue;
                }
                let mut candidates = vec![];
                for block in &blocks {
                    let b = block.bounds;
                    let wy = smooth((p.y - b[1] + block.pad) / block.pad)
                        * smooth((b[3] + block.pad - p.y) / block.pad);
                    if wy == 0. || p.x < b[0] || p.x > b[2] {
                        continue;
                    }
                    let samples: Vec<_> = block
                        .lines
                        .iter()
                        .filter(|l| p.x >= l[0].x && p.x <= l.last().unwrap().x)
                        .map(|l| value(l, p.x))
                        .collect();
                    if samples.len() < 2 {
                        continue;
                    }
                    let wx = smooth((p.x - b[0]) / block.pad) * smooth((b[2] - p.x) / block.pad);
                    candidates.push(median(samples) * wx * wy);
                }
                if !candidates.is_empty() {
                    values[y * w + x] = median(candidates);
                }
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
    fn isolated_or_two_lines_cannot_enable_warp() {
        let mut l = lines();
        l.truncate(2);
        assert!(TextWarp::fit(&l, &[], [0., 0., 1000., 700.]).0.is_none());
    }
}
