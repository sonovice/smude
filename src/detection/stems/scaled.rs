//! Bounded-resolution stem analysis with original pixel-centre coordinates.
use super::{StemDetection, detect_stems};
use crate::detection::resample::{luma, resize_triangle};
use crate::geometry::Observations;
use crate::{Error, Result};
use image::RgbImage;

pub fn detect_stems_scaled(
    image: &RgbImage,
    data: &Observations,
    max_width: u32,
) -> Result<StemDetection> {
    data.validate()?;
    if image.dimensions() != (data.width, data.height) || max_width < 2 {
        return Err(Error::Invalid("invalid scaled stem image or width"));
    }
    let w = image.width().min(max_width);
    let h = ((image.height() as f64 * w as f64 / image.width() as f64).round() as u32).max(2);
    let small = resize_triangle(image, w, h);
    let gray = luma(&small);
    let binary = crate::rendering::output::binarize_sauvola(&gray);
    let (sx, sy) = (w as f64 / data.width as f64, h as f64 / data.height as f64);
    let mut scaled = data.clone();
    scaled.width = w;
    scaled.height = h;
    let scale = |v: &mut Vec<f64>, s: f64, max: u32| {
        v.iter_mut()
            .for_each(|x| *x = ((*x + 0.5) * s - 0.5).clamp(0., (max - 1) as f64))
    };
    for staff in &mut scaled.staves {
        scale(&mut staff.x, sx, w);
        for y in &mut staff.y {
            scale(y, sy, h);
        }
    }
    for bar in &mut scaled.barlines {
        scale(&mut bar.x, sx, w);
        scale(&mut bar.y, sy, h);
    }
    let mut found = detect_stems(&binary, &scaled)?;
    for stem in &mut found.stems {
        for p in &mut stem.points {
            p.x = (p.x + 0.5) / sx - 0.5;
            p.y = (p.y + 0.5) / sy - 0.5;
        }
        for head in &mut stem.heads {
            head.center.x = (head.center.x + 0.5) / sx - 0.5;
            head.center.y = (head.center.y + 0.5) / sy - 0.5;
            for k in 0..4 {
                let s = if k % 2 == 0 { sx } else { sy };
                head.bounds[k] = (head.bounds[k] + 0.5) / s - 0.5;
            }
        }
        stem.slope *= sy / sx;
        stem.line_residual_pixels /= sx;
    }
    Ok(found)
}
