use image::{Rgb, RgbImage};
use rayon::prelude::*;

use crate::{Error, Geometry, Point, Result};

fn weight(x: f64) -> f64 {
    let x = x.abs();
    if x < 1. {
        1.5 * x * x * x - 2.5 * x * x + 1.
    } else if x < 2. {
        -0.5 * x * x * x + 2.5 * x * x - 4. * x + 2.
    } else {
        0.
    }
}

fn sample(image: &RgbImage, p: Point) -> [u8; 3] {
    if !p.x.is_finite()
        || !p.y.is_finite()
        || p.x < -2.
        || p.y < -2.
        || p.x > f64::from(image.width()) + 1.
        || p.y > f64::from(image.height()) + 1.
    {
        return [255; 3];
    }
    let (x, y) = (p.x.floor() as i64, p.y.floor() as i64);
    let mut sum = [0.; 3];
    // Separable cubic weights: evaluate each of the eight taps once, rather
    // than recomputing both polynomials for every one of the 16 source pixels.
    let wx = [-1, 0, 1, 2].map(|i| weight(p.x - (x + i) as f64));
    let wy = [-1, 0, 1, 2].map(|j| weight(p.y - (y + j) as f64));
    for j in -1..=2 {
        for i in -1..=2 {
            let w = wx[(i + 1) as usize] * wy[(j + 1) as usize];
            let (xx, yy) = (x + i, y + j);
            let pixel = if xx < 0
                || yy < 0
                || xx >= i64::from(image.width())
                || yy >= i64::from(image.height())
            {
                Rgb([255; 3])
            } else {
                *image.get_pixel(xx as u32, yy as u32)
            };
            for c in 0..3 {
                sum[c] += w * f64::from(pixel[c]);
            }
        }
    }
    sum.map(|v| v.round().clamp(0., 255.) as u8)
}

impl Geometry {
    /// Catmull-Rom bicubic reconstruction directly from the original RGB image.
    /// Parallel rows share immutable geometry; no intermediate raster stages.
    pub fn render(&self, image: &RgbImage) -> Result<RgbImage> {
        self.render_with_aspect(image, None)
    }

    /// Raster gauge only: keep the vertical scale, adjust horizontal scale.
    /// The fitted geometric map is unchanged; sample the source once.
    /// Match a physical-page aspect prior to the mapped paper extent rather
    /// than to blank canvas outside a projectively rectified quadrilateral.
    pub fn canvas_ratio_for_page(&self, ratio: f64, contour: &[Point]) -> Result<f64> {
        if contour.len() < 3 || !ratio.is_finite() || !(0.1..=10.).contains(&ratio) {
            return Err(Error::Invalid("invalid page-aspect reference"));
        }
        let points: Vec<_> = contour.iter().map(|p| self.forward(*p)).collect();
        let left = points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let right = points.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        let top = points.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let bottom = points.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
        let [x0, y0, x1, y1] = self.report().bounds;
        if right <= left || bottom <= top {
            return Err(Error::Invalid("empty mapped page extent"));
        }
        let canvas = ratio * (right - left) / (bottom - top) * (y1 - y0) / (x1 - x0);
        if !canvas.is_finite() || !(0.1..=10.).contains(&canvas) {
            return Err(Error::Invalid("implausible page-aspect canvas"));
        }
        Ok(canvas)
    }

    pub fn raster_size(&self, height_width_ratio: Option<f64>) -> Result<(u32, u32)> {
        let [x0, y0, x1, y1] = self.report().bounds;
        let height = (y1 - y0).ceil() as u32 + 1;
        let width = if let Some(ratio) = height_width_ratio {
            if !ratio.is_finite() || !(0.1..=10.).contains(&ratio) {
                return Err(Error::Invalid("invalid output height/width ratio"));
            }
            (f64::from(height) / ratio).round().max(2.) as u32
        } else {
            (x1 - x0).ceil() as u32 + 1
        };
        Ok((width, height))
    }

    pub fn render_with_aspect(
        &self,
        image: &RgbImage,
        height_width_ratio: Option<f64>,
    ) -> Result<RgbImage> {
        let (width, height) = self.raster_size(height_width_ratio)?;
        self.render_at_dimensions(image, width, height)
    }

    /// Bounded detector preview; final rendering independently samples source.
    pub fn render_at_dimensions(
        &self,
        image: &RgbImage,
        width: u32,
        height: u32,
    ) -> Result<RgbImage> {
        let r = self.report();
        if width < 2 || height < 2 {
            return Err(Error::Invalid("invalid preview dimensions"));
        }
        if image.dimensions() != (r.input_width, r.input_height) {
            return Err(Error::Invalid(
                "image dimensions differ from fitted observations",
            ));
        }
        let [x0, y0, x1, y1] = r.bounds;
        let pixels = u64::from(width) * u64::from(height);
        if pixels > u64::from(image.width()) * u64::from(image.height()) * 4 {
            return Err(Error::Invalid(
                "output canvas exceeds four times the input area",
            ));
        }
        let mut bytes = vec![255; pixels as usize * 3];
        bytes
            .par_chunks_mut(width as usize * 3)
            .enumerate()
            .for_each(|(j, row)| {
                let y = y0 + (y1 - y0) * j as f64 / f64::from(height - 1);
                let inverse = self.inverse_scanline(y);
                for (i, pixel) in row.chunks_exact_mut(3).enumerate() {
                    let x = x0 + (x1 - x0) * i as f64 / f64::from(width - 1);
                    pixel.copy_from_slice(&sample(image, inverse(x)));
                }
            });
        RgbImage::from_raw(width, height, bytes)
            .ok_or(Error::Invalid("invalid rendered image buffer"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Barline, Config, Observations, Staff, TextLine};
    #[test]
    fn scanline_inverse_matches_general_composition_with_text_gcs_and_x_correction() {
        let data = Observations {
            width: 1000,
            height: 700,
            staves: [250., 400., 550.]
                .into_iter()
                .map(|center| Staff {
                    x: vec![100., 300., 500., 700., 900.],
                    y: std::array::from_fn(|k| {
                        (0..5)
                            .map(|j| center + (k as f64 - 2.) * 10. + 0.4 * (j as f64).sin())
                            .collect()
                    }),
                })
                .collect(),
            barlines: (0..3)
                .flat_map(|staff| {
                    [150., 450., 850.].map(move |x| Barline {
                        staff,
                        x: vec![x, x + 2.4],
                        y: vec![230. + 150. * staff as f64, 270. + 150. * staff as f64],
                        aligned_staffs: vec![],
                    })
                })
                .collect(),
        };
        for gcs in [false, true] {
            let mut map = if gcs {
                Geometry::fit_profiled_gcs(&data, &Config::default(), None, None, &[], true)
                    .0
                    .unwrap()
            } else {
                Geometry::fit(&data, &Config::default()).unwrap()
            };
            map.add_vertical_correction(&data).unwrap();
            map.add_text_curves(
                &[TextLine {
                    points: (0..51)
                        .map(|i| Point {
                            x: 100. + i as f64 * 16.,
                            y: 110. + 0.04 * i as f64 + 4. * (i as f64 / 12.).sin(),
                        })
                        .collect(),
                    bounds: [100., 100., 900., 122.],
                    block_id: 0,
                    confidence: 0.9,
                    uncertainty_pixels: 0.5,
                }],
                &[[80., 210., 920., 610.]],
            );
            for j in 0..241 {
                let y = j as f64 * 3. - 12.5;
                let inverse = map.inverse_scanline(y);
                for i in 0..141 {
                    let x = i as f64 * 7.2 - 7.5;
                    let a = inverse(x);
                    let b = map.inverse(Point { x, y });
                    assert!(
                        a.x.to_bits() == b.x.to_bits() || (a.x.is_nan() && b.x.is_nan()),
                        "gcs={gcs}, {x}, {y}: {a:?} != {b:?}"
                    );
                    assert!(a.y.to_bits() == b.y.to_bits() || (a.y.is_nan() && b.y.is_nan()));
                }
            }
        }
    }
}
