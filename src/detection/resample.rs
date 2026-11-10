//! Detector input preparation that reproduces the `image` crate sample for
//! sample at a fraction of its cost: rows are processed whole and in parallel.
use image::{DynamicImage, GrayImage, ImageBuffer, Pixel, RgbImage};
use rayon::prelude::*;

/// `DynamicImage::into_luma8` of the image. The conversion is the crate's
/// own, applied to bands of rows side by side.
pub(crate) fn luma(image: &RgbImage) -> GrayImage {
    const BAND: usize = 64;
    let width = image.width() as usize;
    let mut gray = vec![0; width * image.height() as usize];
    gray.par_chunks_mut((width * BAND).max(1))
        .zip(image.as_raw().par_chunks((3 * width * BAND).max(1)))
        .for_each(|(gray, rgb)| {
            let rows = (gray.len() / width) as u32;
            let band = RgbImage::from_raw(image.width(), rows, rgb.to_vec())
                .expect("whole rows of the source");
            gray.copy_from_slice(&DynamicImage::ImageRgb8(band).into_luma8());
        });
    GrayImage::from_raw(image.width(), image.height(), gray).expect("one sample per pixel")
}

/// The source samples one output sample blends: the first one and the
/// normalized weight of each.
type Taps = (usize, Vec<f32>);

fn taps(source: usize, target: usize) -> Vec<Taps> {
    let ratio = source as f32 / target as f32;
    let scale = ratio.max(1.);
    (0..target)
        .map(|sample| {
            let centre = (sample as f32 + 0.5) * ratio;
            let first = ((centre - scale).floor() as i64).clamp(0, source as i64 - 1);
            let end = ((centre + scale).ceil() as i64).clamp(first + 1, source as i64);
            let centre = centre - 0.5;
            let mut weights: Vec<f32> = (first..end)
                .map(|i| (1. - ((i as f32 - centre) / scale).abs()).max(0.))
                .collect();
            let sum = weights.iter().fold(0., |sum, weight| sum + weight);
            weights.iter_mut().for_each(|weight| *weight /= sum);
            (first as usize, weights)
        })
        .collect()
}

/// Rows first, then columns, summing in `f32` in tap order: the order that
/// fixes every rounding of the reference.
fn resample<const C: usize>(
    source: &[u8],
    width: usize,
    columns: &[Taps],
    rows: &[Taps],
) -> Vec<u8> {
    let mut target = vec![0; columns.len() * rows.len() * C];
    target
        .par_chunks_exact_mut(columns.len() * C)
        .zip(rows)
        .for_each_init(
            || vec![0_f32; width * C],
            |blend, (target, (first, weights))| {
                blend.fill(0.);
                let rows = source[first * width * C..].chunks_exact(width * C);
                for (row, weight) in rows.zip(weights) {
                    for (sum, sample) in blend.iter_mut().zip(row) {
                        *sum += f32::from(*sample) * weight;
                    }
                }
                let (blend, _) = blend.as_chunks::<C>();
                let (target, _) = target.as_chunks_mut::<C>();
                for (target, (first, weights)) in target.iter_mut().zip(columns) {
                    let mut sum = [0_f32; C];
                    for (pixel, weight) in blend[*first..].iter().zip(weights) {
                        for (sum, sample) in sum.iter_mut().zip(pixel) {
                            *sum += sample * weight;
                        }
                    }
                    *target = sum.map(|sum| sum.clamp(0., 255.).round() as u8);
                }
            },
        );
    target
}

/// `image::imageops::resize` with the triangle filter.
pub(crate) fn resize_triangle<P: Pixel<Subpixel = u8>>(
    image: &ImageBuffer<P, Vec<u8>>,
    width: u32,
    height: u32,
) -> ImageBuffer<P, Vec<u8>> {
    if (width, height) == image.dimensions() {
        return image.clone();
    }
    if image.is_empty() || width == 0 || height == 0 {
        return ImageBuffer::new(width, height);
    }
    let source = image.as_raw();
    let from = image.width() as usize;
    let columns = taps(from, width as usize);
    let rows = taps(image.height() as usize, height as usize);
    let samples = match P::CHANNEL_COUNT {
        1 => resample::<1>(source, from, &columns, &rows),
        2 => resample::<2>(source, from, &columns, &rows),
        3 => resample::<3>(source, from, &columns, &rows),
        4 => resample::<4>(source, from, &columns, &rows),
        count => unreachable!("{count} channels in an image pixel"),
    };
    ImageBuffer::from_raw(width, height, samples).expect("one sample per target channel")
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::imageops::{FilterType, resize};

    fn noise(len: usize) -> Vec<u8> {
        let mut state = 0x9e37_79b9_u32;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn luma_reproduces_the_image_crate_conversion() {
        // Zero, one and several bands, the last of them partial.
        for (width, height) in [(0, 0), (5, 1), (37, 64), (23, 150)] {
            let color = RgbImage::from_raw(width, height, noise((width * height * 3) as usize));
            let color = color.unwrap();
            assert_eq!(
                luma(&color),
                DynamicImage::ImageRgb8(color).into_luma8(),
                "{width}x{height}"
            );
        }
    }

    #[test]
    fn resize_triangle_reproduces_the_image_crate_sample_for_sample() {
        let gray = GrayImage::from_raw(211, 97, noise(211 * 97)).unwrap();
        let color = RgbImage::from_raw(211, 97, noise(211 * 97 * 3)).unwrap();
        for (width, height) in [(83, 38), (64, 97), (211, 40), (300, 150), (1, 1), (211, 97)] {
            assert_eq!(
                resize_triangle(&gray, width, height),
                resize(&gray, width, height, FilterType::Triangle),
                "gray {width}x{height}"
            );
            assert_eq!(
                resize_triangle(&color, width, height),
                resize(&color, width, height, FilterType::Triangle),
                "color {width}x{height}"
            );
        }
    }
}
