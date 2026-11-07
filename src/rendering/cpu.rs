//! CPU reference sampler: full-precision inverse geometry and bicubic reconstruction.
use crate::geometry::{Geometry, Point};
use crate::{Error, Result};
use image::{Rgb, RgbImage};
use rayon::prelude::*;

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
    pub(super) fn render_cpu_at_dimensions(
        &self,
        image: &RgbImage,
        width: u32,
        height: u32,
    ) -> Result<RgbImage> {
        self.validate_raster(image, width, height)?;
        let [x0, y0, x1, y1] = self.report().bounds;
        let pixels = u64::from(width) * u64::from(height);
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
