//! Reversible regular page grid: rectified unit coordinates -> source pixels.
use crate::geometry::Point;
use crate::{Error, Result};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PageGrid {
    side: usize,
    points: Vec<Point>,
    source_bounds: [f64; 4],
}

impl PageGrid {
    pub fn new(points: Vec<Point>) -> Result<Self> {
        let side = (points.len() as f64).sqrt() as usize;
        if side < 2
            || side * side != points.len()
            || points.iter().any(|p| !p.x.is_finite() || !p.y.is_finite())
        {
            return Err(Error::Invalid("invalid learned page grid"));
        }
        let bounds = [
            points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min),
            points.iter().map(|p| p.y).fold(f64::INFINITY, f64::min),
            points.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max),
            points.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max),
        ];
        let grid = Self {
            side,
            points,
            source_bounds: bounds,
        };
        for j in 0..side - 1 {
            for i in 0..side - 1 {
                for (u, v) in [(0., 0.), (1., 0.), (0., 1.), (1., 1.)] {
                    let (_, dx, dy) = grid.cell(i, j, u, v);
                    if dx.x * dy.y - dx.y * dy.x <= 1e-8 {
                        return Err(Error::Invalid("learned page grid folds or collapses"));
                    }
                }
            }
        }
        Ok(grid)
    }

    fn corners(&self, i: usize, j: usize) -> [Point; 4] {
        let top = &self.points[j * self.side + i..][..2];
        let bottom = &self.points[(j + 1) * self.side + i..][..2];
        [top[0], top[1], bottom[0], bottom[1]]
    }

    fn position([a, b, c, d]: [Point; 4], u: f64, v: f64) -> Point {
        Point {
            x: a.x * (1. - u) * (1. - v) + b.x * u * (1. - v) + c.x * (1. - u) * v + d.x * u * v,
            y: a.y * (1. - u) * (1. - v) + b.y * u * (1. - v) + c.y * (1. - u) * v + d.y * u * v,
        }
    }

    fn cell(&self, i: usize, j: usize, u: f64, v: f64) -> (Point, Point, Point) {
        let [a, b, c, d] = self.corners(i, j);
        (
            Self::position([a, b, c, d], u, v),
            Point {
                x: (b.x - a.x) * (1. - v) + (d.x - c.x) * v,
                y: (b.y - a.y) * (1. - v) + (d.y - c.y) * v,
            },
            Point {
                x: (c.x - a.x) * (1. - u) + (d.x - b.x) * u,
                y: (c.y - a.y) * (1. - u) + (d.y - b.y) * u,
            },
        )
    }

    /// Cell index and in-cell coordinate of a unit coordinate along one axis.
    fn axis(&self, unit: f64) -> (usize, f64) {
        let t = unit * (self.side - 1) as f64;
        // Linear extension of edge cells preserves surrounding crop coordinates.
        let i = (t.floor().max(0.) as usize).min(self.side - 2);
        (i, t - i as f64)
    }

    /// Cell indices and in-cell coordinates of a unit point.
    fn locate(&self, p: Point) -> (usize, usize, f64, f64) {
        let (i, u) = self.axis(p.x);
        let (j, v) = self.axis(p.y);
        (i, j, u, v)
    }

    fn sample(&self, p: Point) -> (Point, Point, Point) {
        let n = (self.side - 1) as f64;
        let (i, j, u, v) = self.locate(p);
        let (p, dx, dy) = self.cell(i, j, u, v);
        (
            p,
            Point {
                x: dx.x * n,
                y: dx.y * n,
            },
            Point {
                x: dy.x * n,
                y: dy.y * n,
            },
        )
    }

    pub fn inverse(&self, unit: Point) -> Point {
        let (i, j, u, v) = self.locate(unit);
        Self::position(self.corners(i, j), u, v)
    }

    /// Cells of raster columns: located once, they serve every `inverse_row`.
    pub fn columns(&self, unit_x: impl Iterator<Item = f64>) -> Vec<(usize, f64)> {
        unit_x.map(|x| self.axis(x)).collect()
    }

    /// `inverse` at each of `columns` along the raster row `unit_y`, bit for bit.
    pub fn inverse_row<'a>(
        &'a self,
        columns: &'a [(usize, f64)],
        unit_y: f64,
    ) -> impl Iterator<Item = Point> + 'a {
        let (j, v) = self.axis(unit_y);
        let (top, bottom) = self.points[j * self.side..][..2 * self.side].split_at(self.side);
        columns.iter().map(move |&(i, u)| {
            Self::position([top[i], top[i + 1], bottom[i], bottom[i + 1]], u, v)
        })
    }

    pub fn forward(&self, source: Point) -> Option<Point> {
        let b = self.source_bounds;
        let mut p = Point {
            x: (source.x - b[0]) / (b[2] - b[0]),
            y: (source.y - b[1]) / (b[3] - b[1]),
        };
        for _ in 0..24 {
            let (q, dx, dy) = self.sample(p);
            let ex = q.x - source.x;
            let ey = q.y - source.y;
            if ex.abs().max(ey.abs()) < 1e-7 {
                return Some(p);
            }
            let det = dx.x * dy.y - dx.y * dy.x;
            if !det.is_finite() || det <= 1e-10 {
                return None;
            }
            p.x -= ((ex * dy.y - ey * dy.x) / det).clamp(-0.5, 0.5);
            p.y -= ((dx.x * ey - dx.y * ex) / det).clamp(-0.5, 0.5);
            if !p.x.is_finite() || !p.y.is_finite() || p.x.abs() > 4. || p.y.abs() > 4. {
                return None;
            }
        }
        None
    }

    pub fn inverse_jacobian(&self, p: Point) -> f64 {
        let (_, dx, dy) = self.sample(p);
        dx.x * dy.y - dx.y * dy.x
    }

    pub fn mean_width(&self) -> f64 {
        self.points
            .chunks(self.side)
            .map(|row| {
                row.windows(2)
                    .map(|p| (p[1].x - p[0].x).hypot(p[1].y - p[0].y))
                    .sum::<f64>()
            })
            .sum::<f64>()
            / self.side as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn curved_grid_roundtrips_and_extends_the_crop_without_fake_music() {
        let grid = PageGrid::new(
            (0..25)
                .flat_map(|j| {
                    (0..25).map(move |i| {
                        let u = i as f64 / 24.;
                        let v = j as f64 / 24.;
                        Point {
                            x: 80. + 900. * u + 50. * (v * std::f64::consts::PI).sin(),
                            y: 40. + 1400. * v + 30. * u * (1. - u),
                        }
                    })
                })
                .collect(),
        )
        .unwrap();
        for j in -1..=21 {
            for i in -1..=21 {
                let p = Point {
                    x: i as f64 / 20.,
                    y: j as f64 / 20.,
                };
                let q = grid.forward(grid.inverse(p)).unwrap();
                assert!((p.x - q.x).abs().max((p.y - q.y).abs()) < 1e-8);
            }
        }
    }
    #[test]
    fn row_inverse_equals_point_inverse_including_the_extension() {
        let grid = PageGrid::new(
            (0..25)
                .flat_map(|j| {
                    (0..25).map(move |i| Point {
                        x: 80.3 + 37.7 * i as f64 + 9. * (j as f64 * 0.21).sin(),
                        y: 40.9 + 58.1 * j as f64 + 0.31 * (i * i) as f64,
                    })
                })
                .collect(),
        )
        .unwrap();
        let unit = |k: i32| f64::from(k) / 617.;
        let columns = grid.columns((-40..=660).map(unit));
        for y in (-40..=660).step_by(7).map(unit) {
            for (x, p) in (-40..=660).map(unit).zip(grid.inverse_row(&columns, y)) {
                let q = grid.inverse(Point { x, y });
                assert_eq!(
                    (p.x.to_bits(), p.y.to_bits()),
                    (q.x.to_bits(), q.y.to_bits())
                );
            }
        }
    }
    #[test]
    fn rejects_folded_and_nonfinite_predictions() {
        assert!(
            PageGrid::new(vec![
                Point { x: 0., y: 0. },
                Point { x: 1., y: 0. },
                Point { x: 1., y: 1. },
                Point { x: 0., y: 1. }
            ])
            .is_err()
        );
        assert!(PageGrid::new(vec![Point { x: f64::NAN, y: 0. }; 625]).is_err());
    }
}
