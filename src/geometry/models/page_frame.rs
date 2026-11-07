//! Four-corner rectification for detector input only. Measurements return to
//! the untouched source crop before fitting/rendering the final image.
use crate::geometry::{Barline, Observations, Point, Staff};
use image::{Rgb, RgbImage};
use rayon::prelude::*;
use serde::Serialize;

use crate::{Error, Result};

type Matrix = [[f64; 3]; 3];

#[derive(Clone, Debug, Serialize)]
pub struct PageAnalysisMap {
    source_width: u32,
    source_height: u32,
    width: u32,
    height: u32,
    to_unit: Matrix,
    from_unit: Matrix,
}

fn project(h: Matrix, p: Point) -> Option<Point> {
    let z = h[2][0] * p.x + h[2][1] * p.y + h[2][2];
    if !z.is_finite() || z.abs() < 1e-12 {
        return None;
    }
    let q = Point {
        x: (h[0][0] * p.x + h[0][1] * p.y + h[0][2]) / z,
        y: (h[1][0] * p.x + h[1][1] * p.y + h[1][2]) / z,
    };
    (q.x.is_finite() && q.y.is_finite()).then_some(q)
}

fn invert(m: Matrix) -> Option<Matrix> {
    let [a, b, c] = m;
    let adj = [
        [
            b[1] * c[2] - b[2] * c[1],
            a[2] * c[1] - a[1] * c[2],
            a[1] * b[2] - a[2] * b[1],
        ],
        [
            b[2] * c[0] - b[0] * c[2],
            a[0] * c[2] - a[2] * c[0],
            a[2] * b[0] - a[0] * b[2],
        ],
        [
            b[0] * c[1] - b[1] * c[0],
            a[1] * c[0] - a[0] * c[1],
            a[0] * b[1] - a[1] * b[0],
        ],
    ];
    let det = a[0] * adj[0][0] + a[1] * adj[1][0] + a[2] * adj[2][0];
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    Some(adj.map(|row| row.map(|v| v / det)))
}

impl PageAnalysisMap {
    /// Corners are TL, TR, BR, BL in source-crop pixel coordinates. The ratio
    /// is physical page height / width. Analysis never enlarges the mean page
    /// width; the final output still samples the original crop independently.
    pub fn new(
        source_width: u32,
        source_height: u32,
        corners: [Point; 4],
        height_width_ratio: f64,
        max_analysis_width: u32,
    ) -> Result<Self> {
        if source_width < 2
            || source_height < 2
            || max_analysis_width < 2
            || !height_width_ratio.is_finite()
            || !(0.1..=10.).contains(&height_width_ratio)
        {
            return Err(Error::Invalid("invalid page-analysis dimensions or ratio"));
        }
        if corners.iter().any(|p| {
            !p.x.is_finite()
                || !p.y.is_finite()
                || p.x < 0.
                || p.y < 0.
                || p.x > f64::from(source_width)
                || p.y > f64::from(source_height)
        }) {
            return Err(Error::Invalid("page-analysis corners outside source crop"));
        }
        let q = corners.map(|p| Point {
            x: p.x / f64::from(source_width),
            y: p.y / f64::from(source_height),
        });
        for i in 0..4 {
            let (a, b, c) = (q[i], q[(i + 1) % 4], q[(i + 2) % 4]);
            let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
            if cross <= 1e-8 {
                return Err(Error::Invalid(
                    "page-analysis quadrilateral must be convex and ordered",
                ));
            }
        }
        // Exact square-to-quadrilateral homography in normalized coordinates.
        let dx1 = q[1].x - q[2].x;
        let dx2 = q[3].x - q[2].x;
        let dy1 = q[1].y - q[2].y;
        let dy2 = q[3].y - q[2].y;
        let dx3 = q[0].x - q[1].x + q[2].x - q[3].x;
        let dy3 = q[0].y - q[1].y + q[2].y - q[3].y;
        let det = dx1 * dy2 - dx2 * dy1;
        if det.abs() < 1e-12 {
            return Err(Error::Invalid("degenerate page-analysis quadrilateral"));
        }
        let g = (dx3 * dy2 - dx2 * dy3) / det;
        let h = (dx1 * dy3 - dx3 * dy1) / det;
        let from_unit = [
            [
                q[1].x - q[0].x + g * q[1].x,
                q[3].x - q[0].x + h * q[3].x,
                q[0].x,
            ],
            [
                q[1].y - q[0].y + g * q[1].y,
                q[3].y - q[0].y + h * q[3].y,
                q[0].y,
            ],
            [g, h, 1.],
        ];
        // A linear denominator attains its extrema at the corners: this
        // excludes poles throughout the page, rather than clipping parameters.
        let denominators = [1., 1. + g, 1. + g + h, 1. + h];
        let min = denominators.iter().copied().fold(f64::INFINITY, f64::min);
        let max = denominators.iter().copied().fold(0., f64::max);
        if min <= 1e-7 * max {
            return Err(Error::Invalid(
                "page-analysis homography has an unsafe pole",
            ));
        }
        let to_unit =
            invert(from_unit).ok_or(Error::Invalid("singular page-analysis homography"))?;
        let norm = |m: Matrix| {
            m.iter()
                .map(|r| r.iter().map(|x| x.abs()).sum::<f64>())
                .fold(0., f64::max)
        };
        if norm(from_unit) * norm(to_unit) > 1e8 {
            return Err(Error::Invalid("ill-conditioned page-analysis homography"));
        }
        let edge = |a: Point, b: Point| (a.x - b.x).hypot(a.y - b.y);
        let width = ((edge(corners[0], corners[1]) + edge(corners[3], corners[2])) / 2.)
            .round()
            .max(2.)
            .min(f64::from(max_analysis_width)) as u32;
        if width > 32000 {
            return Err(Error::Invalid(
                "page-analysis raster exceeds resource limit",
            ));
        }
        let height = (f64::from(width - 1) * height_width_ratio).round().max(1.) as u32 + 1;
        if height > 32000 || u64::from(width) * u64::from(height) > 50_000_000 {
            return Err(Error::Invalid(
                "page-analysis raster exceeds resource limit",
            ));
        }
        Ok(Self {
            source_width,
            source_height,
            width,
            height,
            to_unit,
            from_unit,
        })
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn forward(&self, p: Point) -> Option<Point> {
        project(
            self.to_unit,
            Point {
                x: p.x / f64::from(self.source_width),
                y: p.y / f64::from(self.source_height),
            },
        )
        .map(|q| Point {
            x: q.x * f64::from(self.width - 1),
            y: q.y * f64::from(self.height - 1),
        })
    }

    /// Exact determinant of the projective pixel-to-pixel map.
    pub fn jacobian_det(&self, p: Point) -> f64 {
        let h = self.to_unit;
        let det = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
            - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
            + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
        let z = h[2][0] * p.x / f64::from(self.source_width)
            + h[2][1] * p.y / f64::from(self.source_height)
            + h[2][2];
        det / z.powi(3) * f64::from(self.width - 1) * f64::from(self.height - 1)
            / (f64::from(self.source_width) * f64::from(self.source_height))
    }

    pub fn inverse(&self, p: Point) -> Option<Point> {
        project(
            self.from_unit,
            Point {
                x: p.x / f64::from(self.width - 1),
                y: p.y / f64::from(self.height - 1),
            },
        )
        .map(|q| Point {
            x: q.x * f64::from(self.source_width),
            y: q.y * f64::from(self.source_height),
        })
    }

    /// Bilinear detector raster. Supply the already contour-masked crop so
    /// that curved boundaries and adjacent pages remain white. No output image
    /// is ever rendered from this temporary raster.
    pub fn render_analysis(&self, source: &RgbImage) -> Result<RgbImage> {
        if source.dimensions() != (self.source_width, self.source_height) {
            return Err(Error::Invalid("page-analysis source dimensions differ"));
        }
        let mut out = RgbImage::from_pixel(self.width, self.height, Rgb([255; 3]));
        out.as_mut()
            .par_chunks_mut(self.width as usize * 3)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, pixel) in row.chunks_mut(3).enumerate() {
                    let Some(p) = self.inverse(Point {
                        x: x as f64,
                        y: y as f64,
                    }) else {
                        continue;
                    };
                    if p.x < 0.
                        || p.y < 0.
                        || p.x > f64::from(source.width() - 1)
                        || p.y > f64::from(source.height() - 1)
                    {
                        continue;
                    }
                    let x0 = p.x.floor() as u32;
                    let y0 = p.y.floor() as u32;
                    let tx = p.x - f64::from(x0);
                    let ty = p.y - f64::from(y0);
                    let mut sum = [0.; 3];
                    for (xx, wx) in [(x0, 1. - tx), ((x0 + 1).min(source.width() - 1), tx)] {
                        for (yy, wy) in [(y0, 1. - ty), ((y0 + 1).min(source.height() - 1), ty)] {
                            for (v, c) in sum.iter_mut().zip(source.get_pixel(xx, yy).0) {
                                *v += f64::from(c) * wx * wy;
                            }
                        }
                    }
                    for (v, s) in pixel.iter_mut().zip(sum) {
                        *v = s.round().clamp(0., 255.) as u8;
                    }
                }
            });
        Ok(out)
    }

    /// Restore full source-crop coordinates. Each of the five staff curves is
    /// transformed independently, then interpolated over their common source-x
    /// interval. Staff identities and line indices are never sorted together.
    pub fn observations_to_source(&self, data: &Observations) -> Result<Observations> {
        data.validate()?;
        if (data.width, data.height) != self.dimensions() {
            return Err(Error::Invalid(
                "page-analysis observation dimensions differ",
            ));
        }
        let restore = |x: f64, y: f64| -> Result<Point> {
            let mut p = self
                .inverse(Point { x, y })
                .ok_or(Error::Invalid("invalid inverse page-analysis point"))?;
            if p.x < -1e-6
                || p.y < -1e-6
                || p.x > f64::from(self.source_width) + 1e-6
                || p.y > f64::from(self.source_height) + 1e-6
            {
                return Err(Error::Invalid("restored measurement outside source crop"));
            }
            // Model contours may denote the outer pixel edge (width/height).
            p.x = p.x.clamp(0., f64::from(self.source_width - 1));
            p.y = p.y.clamp(0., f64::from(self.source_height - 1));
            Ok(p)
        };
        let mut staves = vec![];
        for staff in &data.staves {
            let mut lines = vec![];
            for y in &staff.y {
                let mut line = staff
                    .x
                    .iter()
                    .zip(y)
                    .map(|(x, y)| restore(*x, *y))
                    .collect::<Result<Vec<_>>>()?;
                if line.windows(2).all(|w| w[0].x > w[1].x) {
                    line.reverse();
                }
                if line.windows(2).any(|w| w[0].x >= w[1].x) {
                    return Err(Error::Invalid("restored staff is not a source-x graph"));
                }
                lines.push(line);
            }
            let left = lines
                .iter()
                .map(|l| l[0].x)
                .fold(f64::NEG_INFINITY, f64::max);
            let right = lines
                .iter()
                .map(|l| l.last().unwrap().x)
                .fold(f64::INFINITY, f64::min);
            if right - left < 1. {
                return Err(Error::Invalid(
                    "restored staff lines have no common source-x interval",
                ));
            }
            let n = (right - left).ceil() as usize;
            let x: Vec<_> = (0..=n)
                .map(|j| left + (right - left) * j as f64 / n as f64)
                .collect();
            let y = std::array::from_fn(|k| {
                x.iter()
                    .map(|at| {
                        let line = &lines[k];
                        let hi = line.partition_point(|p| p.x < *at).clamp(1, line.len() - 1);
                        let (a, b) = (line[hi - 1], line[hi]);
                        let t = (*at - a.x) / (b.x - a.x);
                        a.y + t * (b.y - a.y)
                    })
                    .collect()
            });
            staves.push(Staff { x, y });
        }
        let mut barlines = vec![];
        for bar in &data.barlines {
            let mut points = bar
                .x
                .iter()
                .zip(&bar.y)
                .map(|(x, y)| restore(*x, *y))
                .collect::<Result<Vec<_>>>()?;
            if points.windows(2).all(|w| w[0].y > w[1].y) {
                points.reverse();
            }
            if points.windows(2).any(|w| w[0].y >= w[1].y) {
                return Err(Error::Invalid("restored barline is not a source-y graph"));
            }
            barlines.push(Barline {
                staff: bar.staff,
                x: points.iter().map(|p| p.x).collect(),
                y: points.iter().map(|p| p.y).collect(),
                aligned_staffs: bar.aligned_staffs.clone(),
            });
        }
        let restored = Observations {
            width: self.source_width,
            height: self.source_height,
            staves,
            barlines,
        };
        restored.validate()?;
        Ok(restored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn corners() -> [Point; 4] {
        [
            Point { x: 170., y: 70. },
            Point { x: 820., y: 140. },
            Point { x: 970., y: 1350. },
            Point { x: 40., y: 1420. },
        ]
    }
    fn map() -> PageAnalysisMap {
        PageAnalysisMap::new(1000, 1500, corners(), 1.5, 700).unwrap()
    }
    #[test]
    fn trapezoid_corners_ratio_and_roundtrip() {
        let m = map();
        let (w, h) = m.dimensions();
        assert!(w <= 700);
        assert!((f64::from(h - 1) / f64::from(w - 1) - 1.5).abs() < 0.002);
        let targets = [
            Point { x: 0., y: 0. },
            Point {
                x: f64::from(w - 1),
                y: 0.,
            },
            Point {
                x: f64::from(w - 1),
                y: f64::from(h - 1),
            },
            Point {
                x: 0.,
                y: f64::from(h - 1),
            },
        ];
        for (source, target) in corners().into_iter().zip(targets) {
            let q = m.forward(source).unwrap();
            assert!((q.x - target.x).hypot(q.y - target.y) < 1e-8);
        }
        for x in [0., 100., 400., 699.] {
            for y in [0., 200., 800.] {
                let p = Point { x, y };
                let q = m.forward(m.inverse(p).unwrap()).unwrap();
                assert!((p.x - q.x).hypot(p.y - q.y) < 1e-8);
            }
        }
    }
    #[test]
    fn observations_restore_original_coordinates_and_line_identity() {
        let m = map();
        let (width, height) = m.dimensions();
        let x: Vec<_> = (50..650).step_by(3).map(f64::from).collect();
        let staff = Staff {
            x: x.clone(),
            y: std::array::from_fn(|k| vec![400. + k as f64 * 10.; x.len()]),
        };
        let data = Observations {
            width,
            height,
            staves: vec![staff],
            barlines: vec![Barline {
                staff: 0,
                x: vec![200., 200.],
                y: vec![400., 440.],
                aligned_staffs: vec![],
            }],
        };
        let restored = m.observations_to_source(&data).unwrap();
        assert_eq!((restored.width, restored.height), (1000, 1500));
        for k in 0..5 {
            for (x, y) in restored.staves[0].x.iter().zip(&restored.staves[0].y[k]) {
                let p = m.forward(Point { x: *x, y: *y }).unwrap();
                assert!((p.y - (400. + k as f64 * 10.)).abs() < 0.01);
            }
        }
        assert!(restored.barlines[0].y[0] < restored.barlines[0].y[1]);
    }
    #[test]
    fn invalid_quads_are_rejected() {
        let mut c = corners();
        c.swap(1, 2);
        assert!(PageAnalysisMap::new(1000, 1500, c, 1.5, 700).is_err());
        let mut c = corners();
        c[1] = c[0];
        assert!(PageAnalysisMap::new(1000, 1500, c, 1.5, 700).is_err());
        let mut c = corners();
        c[0].x = f64::NAN;
        assert!(PageAnalysisMap::new(1000, 1500, c, 1.5, 700).is_err());
        assert!(PageAnalysisMap::new(1000, 1500, corners(), 0., 700).is_err());
    }
    #[test]
    fn analysis_raster_preserves_white_mask_and_source_dimensions() {
        let m = map();
        let source = RgbImage::from_pixel(1000, 1500, Rgb([255; 3]));
        let out = m.render_analysis(&source).unwrap();
        assert_eq!(out.dimensions(), m.dimensions());
        assert!(out.pixels().all(|p| p.0 == [255; 3]));
        assert!(m.render_analysis(&RgbImage::new(20, 20)).is_err());
    }
}
