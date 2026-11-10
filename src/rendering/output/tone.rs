//! Local contrast, binarization and optical antialiasing.
use image::{GrayImage, RgbImage};
use rayon::prelude::*;
use std::ops::{Add, Sub};

/// The pixels of `image::imageops::grayscale`, one row per task: the integer
/// luma the GPU renderer also computes.
pub(crate) fn grayscale(image: &RgbImage) -> GrayImage {
    let width = image.width() as usize;
    let mut gray = vec![0; width * image.height() as usize];
    gray.par_chunks_mut(width.max(1))
        .zip(image.as_raw().par_chunks(3 * width.max(1)))
        .for_each(|(gray, rgb)| {
            for (gray, rgb) in gray.iter_mut().zip(rgb.chunks_exact(3)) {
                let [r, g, b] = [rgb[0], rgb[1], rgb[2]].map(u32::from);
                *gray = ((2126 * r + 7152 * g + 722 * b) / 10000) as u8;
            }
        });
    GrayImage::from_raw(image.width(), image.height(), gray).unwrap()
}

/// Fixed Sauvola threshold: 51×51 window, k=.25, R=128, replicated borders.
/// Sliding sums give constant work per pixel without another geometric sample.
pub fn binarize_sauvola(gray: &GrayImage) -> GrayImage {
    adaptive_tone(gray, false)
}

/// Legacy contrast enhancement for the text detector only. Keep its calibrated
/// input unchanged; final grayscale uses the independent paper field estimator.
#[cfg(feature = "application")]
pub(crate) fn normalize_text_analysis(gray: &GrayImage) -> GrayImage {
    adaptive_tone(gray, true)
}

const RADIUS: usize = 25;
const WINDOW: usize = 2 * RADIUS + 1;

/// The sum of the values and of their squares over the window around every
/// pixel of one row, borders replicated. Rows are visited in order: each
/// costs one horizontal pass over the row entering the window and two updates
/// of the column sums. All sums are exact integers that fit `u32`.
struct WindowSums<'a> {
    source: &'a [u8],
    width: usize,
    height: usize,
    /// Horizontal window sums of the source rows inside the vertical window,
    /// stored by row index modulo `WINDOW`.
    rows: Vec<[u32; 2]>,
    /// `rows` summed over the vertical window of the current row.
    columns: Vec<[u32; 2]>,
}

impl<'a> WindowSums<'a> {
    /// The sums of row `y`.
    fn new(gray: &'a GrayImage, y: usize) -> Self {
        let (width, height) = (gray.width() as usize, gray.height() as usize);
        let mut sums = Self {
            source: gray.as_raw(),
            width,
            height,
            rows: vec![[0; 2]; WINDOW * width],
            columns: vec![[0; 2]; width],
        };
        let mut loaded = None;
        for offset in 0..WINDOW {
            let row = (y + offset).saturating_sub(RADIUS).min(height - 1);
            if loaded != Some(row) {
                sums.load(row);
                loaded = Some(row);
            }
            sums.accumulate(row, u32::add);
        }
        sums
    }

    /// Move from the sums of row `y` to those of row `y + 1`.
    fn advance(&mut self, y: usize) {
        // The row that leaves is subtracted before the entering row reuses
        // its storage.
        self.accumulate(y.saturating_sub(RADIUS), u32::sub);
        let entering = y + 1 + RADIUS;
        if entering < self.height {
            self.load(entering);
        }
        self.accumulate(entering.min(self.height - 1), u32::add);
    }

    fn accumulate(&mut self, row: usize, combine: impl Fn(u32, u32) -> u32) {
        let row = &self.rows[(row % WINDOW) * self.width..][..self.width];
        for (column, row) in self.columns.iter_mut().zip(row) {
            *column = [combine(column[0], row[0]), combine(column[1], row[1])];
        }
    }

    /// Compute the horizontal window sums of one source row.
    fn load(&mut self, row: usize) {
        let width = self.width;
        let values = &self.source[row * width..][..width];
        let value = |x: usize| {
            let value = u32::from(values[x.min(width - 1)]);
            [value, value * value]
        };
        let mut sum = [0; 2];
        for offset in 0..WINDOW {
            let [value, square] = value(offset.saturating_sub(RADIUS));
            sum = [sum[0] + value, sum[1] + square];
        }
        let sums = &mut self.rows[(row % WINDOW) * width..][..width];
        for (x, sums) in sums.iter_mut().enumerate() {
            if x > 0 {
                let entering = value(x + RADIUS);
                let leaving = value((x - 1).saturating_sub(RADIUS));
                sum = [
                    sum[0] + entering[0] - leaving[0],
                    sum[1] + entering[1] - leaving[1],
                ];
            }
            *sums = sum;
        }
    }
}

fn adaptive_tone(gray: &GrayImage, soft: bool) -> GrayImage {
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    if w == 0 || h == 0 {
        return gray.clone();
    }
    // Each task first sums a whole window, so bands are much taller than one.
    const BAND: usize = 256;
    let area = (WINDOW * WINDOW) as f64;
    let mut output = vec![255; w * h];
    output
        .par_chunks_mut(w * BAND)
        .enumerate()
        .for_each(|(band, rows)| {
            let mut sums = WindowSums::new(gray, band * BAND);
            for (offset, row) in rows.chunks_mut(w).enumerate() {
                let y = band * BAND + offset;
                if offset > 0 {
                    sums.advance(y - 1);
                }
                let values = &gray.as_raw()[y * w..][..w];
                for ((pixel, value), [sum, squares]) in
                    row.iter_mut().zip(values).zip(&sums.columns)
                {
                    let mean = f64::from(*sum) / area;
                    let mean_squared = f64::from(*squares) / area;
                    let stddev = (mean_squared - mean * mean).max(0.).sqrt();
                    let threshold = mean * (1. + 0.25 * (stddev / 128. - 1.));
                    let value = f64::from(*value);
                    *pixel = if soft {
                        if value == 0. {
                            0
                        } else {
                            let half = (mean * 0.1).max(stddev * 0.5).min(mean * 0.18).max(2.);
                            let black = threshold - half;
                            let white = (threshold + half).min(mean * 0.98);
                            (255. * ((value - black) / (white - black).max(1.)).clamp(0., 1.))
                                .round() as u8
                        }
                    } else if value <= threshold {
                        0
                    } else {
                        255
                    };
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(len: usize) -> Vec<u8> {
        let mut state = 0x2545_f491_u32;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn window_sums_equal_direct_sums_over_replicated_borders() {
        for (width, height) in [(1, 1), (7, 3), (60, 80), (30, 300)] {
            let gray =
                GrayImage::from_raw(width, height, noise((width * height) as usize)).unwrap();
            let (w, h) = (width as usize, height as usize);
            let mut sums = WindowSums::new(&gray, 0);
            for y in 0..h {
                if y > 0 {
                    sums.advance(y - 1);
                }
                assert_eq!(sums.columns, WindowSums::new(&gray, y).columns);
                for x in 0..w {
                    let mut expected = [0; 2];
                    for dy in 0..WINDOW {
                        for dx in 0..WINDOW {
                            let sx = (x + dx).saturating_sub(RADIUS).min(w - 1);
                            let sy = (y + dy).saturating_sub(RADIUS).min(h - 1);
                            let value = u32::from(gray.as_raw()[sy * w + sx]);
                            expected = [expected[0] + value, expected[1] + value * value];
                        }
                    }
                    assert_eq!(sums.columns[x], expected, "{width}x{height} at {x},{y}");
                }
            }
        }
    }

    #[test]
    fn grayscale_equals_the_image_crate_conversion() {
        for (width, height) in [(0, 0), (1, 1), (37, 19)] {
            let mut raw = noise((3 * width * height) as usize);
            if let Some(pixel) = raw.first_chunk_mut::<3>() {
                *pixel = [255; 3];
            }
            let image = RgbImage::from_raw(width, height, raw).unwrap();
            assert_eq!(grayscale(&image), image::imageops::grayscale(&image));
        }
    }
}
