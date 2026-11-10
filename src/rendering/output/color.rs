//! RGB paper and print tone correction, with conservative handling of full-bleed covers.
//! A bounded sample grid estimates paper; original foreground pixels are never resized.
use image::RgbImage;
use rayon::prelude::*;

type Rgb = [f32; 3];
#[derive(Debug)]
struct Estimate {
    coverage: f32,
    black: f32,
    enabled: bool,
}

/// Correct sampled paper illumination and apply a restrained neutral black point.
/// Foreground pixels stay at their original resolution. Without enough neutral
/// paper evidence (for example a colored cover), preserve the input exactly.
pub fn enhance_color(input: &RgbImage) -> RgbImage {
    correct(input).0
}

fn correct(input: &RgbImage) -> (RgbImage, Estimate) {
    let (w, h) = input.dimensions();
    let mut stats = Estimate {
        coverage: 0.,
        black: 0.,
        enabled: false,
    };
    if w == 0 || h == 0 {
        return (input.clone(), stats);
    }
    let scale = (768. / w.max(h) as f32).min(1.);
    let (sw, sh) = (
        (w as f32 * scale).round().max(1.) as usize,
        (h as f32 * scale).round().max(1.) as usize,
    );
    let sample: Vec<[u8; 3]> = (0..sh)
        .flat_map(|y| {
            (0..sw).map(move |x| {
                let px = (((x * 2 + 1) as u64 * u64::from(w)) / (2 * sw) as u64) as u32;
                let py = (((y * 2 + 1) as u64 * u64::from(h)) / (2 * sh) as u64) as u32;
                input.get_pixel(px.min(w - 1), py.min(h - 1)).0
            })
        })
        .collect();
    let neutral = |p: [u8; 3]| {
        let lo = *p.iter().min().unwrap();
        let hi = *p.iter().max().unwrap();
        lo >= 40 && u16::from(hi - lo) * 4 <= u16::from(hi)
    };
    let mut hist = [[0usize; 256]; 3];
    let mut count = 0;
    for p in sample.iter().copied().filter(|p| neutral(*p)) {
        count += 1;
        for c in 0..3 {
            hist[c][p[c] as usize] += 1;
        }
    }
    if count * 2 < sample.len() {
        return (input.clone(), stats);
    }
    let anchor: Rgb = std::array::from_fn(|c| quantile(&hist[c], count * 95 / 100).max(1) as f32);
    let paper = |p: [u8; 3]| {
        if !neutral(p) {
            return false;
        }
        let balanced: Rgb = std::array::from_fn(|c| f32::from(p[c]) / anchor[c]);
        let lo = balanced.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = balanced.iter().copied().fold(0., f32::max);
        hi - lo < hi * 0.12
    };
    // Equal-size cells avoid partial last-cell interpolation bias.
    let (gw, gh) = (sw.div_ceil(16), sh.div_ceil(16));
    let mut field = vec![[0.; 3]; gw * gh];
    let mut valid = vec![false; gw * gh];
    for y in 0..gh {
        for x in 0..gw {
            let (x0, x1, y0, y1) = (
                x * sw / gw,
                (x + 1) * sw / gw,
                y * sh / gh,
                (y + 1) * sh / gh,
            );
            let mut hist = [[0usize; 256]; 3];
            let mut n = 0;
            for yy in y0..y1 {
                for &p in &sample[yy * sw + x0..yy * sw + x1] {
                    if paper(p) {
                        n += 1;
                        for c in 0..3 {
                            hist[c][p[c] as usize] += 1;
                        }
                    }
                }
            }
            if n * 5 >= (x1 - x0) * (y1 - y0) * 2 {
                valid[y * gw + x] = true;
                field[y * gw + x] = std::array::from_fn(|c| quantile(&hist[c], n * 4 / 5) as f32);
            }
        }
    }
    stats.coverage = valid.iter().filter(|&&v| v).count() as f32 / valid.len() as f32;
    // Full-bleed colored covers do not supply a trustworthy paper reference.
    if stats.coverage < 0.60 {
        return (input.clone(), stats);
    }
    stats.enabled = true;
    for (p, &valid) in field.iter_mut().zip(&valid) {
        if !valid {
            *p = anchor;
        }
    }
    // Extend measured paper into illustrations using only neighboring paper.
    // All iterations operate on at most 48x48 cells, never on the original.
    let mut next = field.clone();
    for _ in 0..160 {
        for y in 0..gh {
            for x in 0..gw {
                let i = y * gw + x;
                if valid[i] {
                    continue;
                }
                let mut sum = [0.; 3];
                let mut n = 0.;
                for yy in y.saturating_sub(1)..=(y + 1).min(gh - 1) {
                    for xx in x.saturating_sub(1)..=(x + 1).min(gw - 1) {
                        for (s, v) in sum.iter_mut().zip(field[yy * gw + xx]) {
                            *s += v;
                        }
                        n += 1.;
                    }
                }
                next[i] = sum.map(|v| v / n);
            }
        }
        std::mem::swap(&mut next, &mut field);
    }
    // A small neutral dark population supplies a restrained black point after
    // white balance. No individual color channel is stretched independently.
    let mut dark = [0usize; 256];
    let mut n = 0;
    for (i, p) in sample.iter().enumerate() {
        let white = interpolate(&field, gw, gh, i % sw, i / sw, sw, sh);
        let rgb: Rgb = std::array::from_fn(|c| f32::from(p[c]) / white[c].max(32.));
        let hi = rgb.iter().copied().fold(0., f32::max);
        let lo = rgb.iter().copied().fold(f32::INFINITY, f32::min);
        if hi < 0.4 && hi - lo <= hi * 0.25 {
            let value = (rgb.iter().sum::<f32>() / 3. * 255.).round() as usize;
            dark[value.min(255)] += 1;
            n += 1;
        }
    }
    if n * 500 >= sample.len() {
        stats.black = (quantile(&dark, n / 10) as f32 / 255. * 0.5).min(0.08);
    }
    // Two percent highlight headroom removes paper texture. It is deliberate
    // display clipping, not a claim to recover an unknown digital original.
    let black = stats.black;
    let slope = 255. / (1. - black);
    let mut output = input.clone();
    let xs: Vec<_> = (0..w as usize).map(|x| axis(x, w as usize, gw)).collect();
    output
        .as_mut()
        .par_chunks_mut(w as usize * 3)
        .enumerate()
        .for_each_init(
            || vec![[0.; 3]; w as usize],
            |gains, (y, row)| {
                let (y0, y1, ty) = axis(y, h as usize, gh);
                for (gain, &(x0, x1, tx)) in gains.iter_mut().zip(&xs) {
                    for c in 0..3 {
                        let top = field[y0 * gw + x0][c] * (1. - tx) + field[y0 * gw + x1][c] * tx;
                        let bottom =
                            field[y1 * gw + x0][c] * (1. - tx) + field[y1 * gw + x1][c] * tx;
                        gain[c] = slope / ((top * (1. - ty) + bottom * ty) * 0.98).max(32.);
                    }
                }
                for (p, gain) in row.as_chunks_mut::<3>().0.iter_mut().zip(gains.iter()) {
                    for c in 0..3 {
                        p[c] = (f32::from(p[c]) * gain[c] - slope * black)
                            .round()
                            .clamp(0., 255.) as u8;
                    }
                }
            },
        );
    (output, stats)
}
fn quantile(hist: &[usize; 256], rank: usize) -> u8 {
    let mut sum = 0;
    for (v, &n) in hist.iter().enumerate() {
        sum += n;
        if sum >= rank.max(1) {
            return v as u8;
        }
    }
    255
}
fn axis(i: usize, size: usize, count: usize) -> (usize, usize, f32) {
    let x = ((i as f32 + 0.5) * count as f32 / size as f32 - 0.5).clamp(0., (count - 1) as f32);
    let a = x.floor() as usize;
    (a, (a + 1).min(count - 1), x - a as f32)
}
fn interpolate(field: &[Rgb], gw: usize, gh: usize, x: usize, y: usize, w: usize, h: usize) -> Rgb {
    let (x0, x1, tx) = axis(x, w, gw);
    let (y0, y1, ty) = axis(y, h, gh);
    std::array::from_fn(|c| {
        (field[y0 * gw + x0][c] * (1. - tx) + field[y0 * gw + x1][c] * tx) * (1. - ty)
            + (field[y1 * gw + x0][c] * (1. - tx) + field[y1 * gw + x1][c] * tx) * ty
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;
    #[test]
    fn tinted_shaded_paper_clears_but_colored_fill_and_pixel_ink_survive() {
        let input = RgbImage::from_fn(800, 1000, |x, y| {
            let paper = [
                210. - x as f32 * 0.05,
                195. - x as f32 * 0.045,
                180. - x as f32 * 0.04,
            ];
            let ink = if (500..720).contains(&x) && (120..350).contains(&y) {
                [0.25, 0.65, 0.85]
            } else if x == 300 || y == 650 {
                [0.25; 3]
            } else {
                [1.; 3]
            };
            Rgb(std::array::from_fn(|c| (paper[c] * ink[c]).round() as u8))
        });
        let (out, estimate) = correct(&input);
        assert!(estimate.enabled);
        for (x, y) in [(100, 100), (400, 400), (740, 900)] {
            assert!(out.get_pixel(x, y).0.iter().all(|&v| v >= 250));
        }
        let color = out.get_pixel(600, 230).0;
        assert!(
            color[0] < 80 && color[1] > 130 && color[1] < 185 && color[2] > 200,
            "{color:?}"
        );
        assert!(out.get_pixel(300, 500).0.iter().all(|&v| v < 80));
        assert!(out.get_pixel(400, 650).0.iter().all(|&v| v < 80));
    }
    #[test]
    fn full_bleed_cover_and_empty_images_are_unchanged() {
        let cover = RgbImage::from_fn(800, 1000, |x, y| {
            Rgb(if (50..750).contains(&x) && (120..135).contains(&y) {
                [230; 3]
            } else {
                [20, 70, 130]
            })
        });
        let (out, stats) = correct(&cover);
        assert!(!stats.enabled);
        assert_eq!(out, cover);
        assert_eq!(enhance_color(&RgbImage::new(0, 0)).dimensions(), (0, 0));
    }
}
