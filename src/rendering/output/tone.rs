//! Local contrast, binarization and optical antialiasing.
use image::GrayImage;
use rayon::prelude::*;

/// Fixed Sauvola threshold: 51×51 window, k=.25, R=128, replicated borders.
/// Prefix sums give constant work per pixel without another geometric sample.
pub fn binarize_sauvola(gray: &GrayImage) -> GrayImage {
    adaptive_tone(gray, false)
}

/// Local paper-to-white normalization with a continuous foreground transition.
/// Unlike the binary result, partial pixel coverage is retained as gray.
pub fn normalize_paper(gray: &GrayImage) -> GrayImage {
    adaptive_tone(gray, true)
}

fn adaptive_tone(gray: &GrayImage, soft: bool) -> GrayImage {
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    if w == 0 || h == 0 {
        return gray.clone();
    }
    const RADIUS: usize = 25;
    const WINDOW: usize = 2 * RADIUS + 1;
    let (pw, ph) = (w + 2 * RADIUS, h + 2 * RADIUS);
    let stride = pw + 1;
    let mut sum = vec![0_u64; stride * (ph + 1)];
    let mut squares = vec![0_u64; sum.len()];
    for y in 0..ph {
        let source_y = y.saturating_sub(RADIUS).min(h - 1);
        let (mut row, mut row_squared) = (0_u64, 0_u64);
        for x in 0..pw {
            let source_x = x.saturating_sub(RADIUS).min(w - 1);
            let value = u64::from(gray.as_raw()[source_y * w + source_x]);
            row += value;
            row_squared += value * value;
            let i = (y + 1) * stride + x + 1;
            sum[i] = sum[i - stride] + row;
            squares[i] = squares[i - stride] + row_squared;
        }
    }
    let area = (WINDOW * WINDOW) as f64;
    let mut output = vec![255; w * h];
    output.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, pixel) in row.iter_mut().enumerate() {
            let (a, b, c, d) = (
                y * stride + x,
                y * stride + x + WINDOW,
                (y + WINDOW) * stride + x,
                (y + WINDOW) * stride + x + WINDOW,
            );
            let mean = (sum[d] + sum[a] - sum[b] - sum[c]) as f64 / area;
            let mean_squared = (squares[d] + squares[a] - squares[b] - squares[c]) as f64 / area;
            let stddev = (mean_squared - mean * mean).max(0.).sqrt();
            let threshold = mean * (1. + 0.25 * (stddev / 128. - 1.));
            let value = f64::from(gray.as_raw()[y * w + x]);
            *pixel = if soft {
                if value == 0. {
                    0
                } else {
                    let half = (mean * 0.1).max(stddev * 0.5).min(mean * 0.18).max(2.);
                    let black = threshold - half;
                    let white = (threshold + half).min(mean * 0.98);
                    (255. * ((value - black) / (white - black).max(1.)).clamp(0., 1.)).round() as u8
                }
            } else if value <= threshold {
                0
            } else {
                255
            };
        }
    });
    GrayImage::from_raw(gray.width(), gray.height(), output).unwrap()
}

/// Very light, symmetric 3-tap optical antialiasing (Gaussian sigma .5px).
/// White stays exactly white away from ink; no geometric map or resize changes.
pub fn antialias_print(gray: &GrayImage) -> GrayImage {
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    if w == 0 || h == 0 {
        return gray.clone();
    }
    let a = (-2_f64).exp() / (1. + 2. * (-2_f64).exp());
    let b = 1. - 2. * a;
    let mut horizontal = vec![0_f64; w * h];
    horizontal
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, value) in row.iter_mut().enumerate() {
                *value = a * f64::from(gray.as_raw()[y * w + x.saturating_sub(1)])
                    + b * f64::from(gray.as_raw()[y * w + x])
                    + a * f64::from(gray.as_raw()[y * w + (x + 1).min(w - 1)]);
            }
        });
    let mut out = vec![255; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, value) in row.iter_mut().enumerate() {
            let v = (a * horizontal[y.saturating_sub(1) * w + x]
                + b * horizontal[y * w + x]
                + a * horizontal[(y + 1).min(h - 1) * w + x])
                .round()
                .clamp(0., 255.) as u8;
            *value = if v >= 254 { 255 } else { v };
        }
    });
    GrayImage::from_raw(gray.width(), gray.height(), out).unwrap()
}
