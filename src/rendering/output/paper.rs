//! Estimate illumination on a bounded grid; correct original gray samples.
use image::GrayImage;
use rayon::prelude::*;

/// Continuous paper correction, independent of foreground/binary decisions.
/// Closing fills dark strokes in the illumination estimate only. Max pooling
/// keeps thin ink from lowering the reduced background samples. We never resize
/// the foreground, stretch its contrast, or snap near-white coverage to white.
/// Dense ink wider than the closing window and hard shadows remain ambiguous;
/// this conservative baseline does not attempt to reconstruct missing details.
pub fn normalize_paper(gray: &GrayImage) -> GrayImage {
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    if w == 0 || h == 0 {
        return gray.clone();
    }
    let step = w.max(h).div_ceil(1024);
    let (bw, bh) = (w.div_ceil(step), h.div_ceil(step));
    let mut background = vec![0_u8; bw * bh];
    background
        .par_chunks_mut(bw)
        .zip(gray.par_chunks(w * step))
        .for_each_init(
            || vec![0_u8; w],
            |columns, (cells, band)| {
                columns.fill(0);
                for row in band.chunks(w) {
                    for (max, value) in columns.iter_mut().zip(row) {
                        *max = (*max).max(*value);
                    }
                }
                for (cell, values) in cells.iter_mut().zip(columns.chunks(step)) {
                    *cell = values.iter().copied().max().unwrap_or(0);
                }
            },
        );
    // Follow broad paper blotches while still bridging thin print. A larger
    // window flattens these illumination changes and leaves gray clouds after
    // normalization. Keep a minimum support for small images; morphology buffers
    // remain bounded by the analysis grid.
    let radius = (w.min(h) / 128).max(8).div_ceil(step).max(2);
    let broad_radius = (w.min(h) / 64).max(8).div_ceil(step).max(2);
    let broad = if broad_radius > radius {
        let dilated = extrema_2d(&background, bw, bh, broad_radius, true);
        Some(extrema_2d(&dilated, bw, bh, broad_radius, false))
    } else {
        None
    };
    let dilated = extrema_2d(&background, bw, bh, radius, true);
    background = extrema_2d(&dilated, bw, bh, radius, false);
    if let Some(broad) = broad {
        // A small window can fit inside a filled head or thick beam. Let it
        // follow mild paper variation, but retain the broad estimate wherever
        // it would otherwise classify substantial ink contrast as illumination.
        // Blend continuously to avoid a threshold seam in the illumination field.
        background
            .par_iter_mut()
            .zip(broad)
            .for_each(|(fine, broad)| {
                let contrast = f32::from(broad.saturating_sub(*fine)) / f32::from(broad.max(1));
                let t = ((contrast - 0.12) / 0.04).clamp(0., 1.);
                let t = t * t * (3. - 2. * t);
                *fine = (f32::from(*fine) * (1. - t) + f32::from(broad) * t).round() as u8;
            });
    }
    let paper = Grid::new(&background, [bw, bh], step, [w, h]);
    let mut out = vec![255; w * h];
    out.par_chunks_mut(w)
        .zip(gray.par_chunks(w))
        .enumerate()
        .for_each_init(Default::default, |rows, (y, (row, gray))| {
            let (top, bottom, ty) = paper.rows(y, rows);
            for (((value, gray), top), bottom) in row.iter_mut().zip(gray).zip(top).zip(bottom) {
                let paper = (top * (1. - ty) + bottom * ty).max(32.);
                // Limit gain in nearly black regions; quantize only after correction.
                *value = (255. * f32::from(*gray) / paper).round().clamp(0., 255.) as u8;
            }
        });
    GrayImage::from_raw(gray.width(), gray.height(), out).unwrap()
}

/// A pooled grid, interpolated bilinearly at full resolution row by row.
struct Grid<'a> {
    cells: &'a [u8],
    columns: usize,
    rows: usize,
    step: usize,
    height: usize,
    xs: Vec<(usize, usize, f32)>,
}

/// A horizontally interpolated grid row, kept while image rows still need it.
#[derive(Default)]
struct GridRow {
    index: Option<usize>,
    values: Vec<f32>,
}

impl<'a> Grid<'a> {
    fn new(cells: &'a [u8], [columns, rows]: [usize; 2], step: usize, [w, h]: [usize; 2]) -> Self {
        Self {
            cells,
            columns,
            rows,
            step,
            height: h,
            xs: (0..w).map(|x| grid_sample(x, w, step, columns)).collect(),
        }
    }

    /// The grid rows above and below image row `y`, interpolated along x, and
    /// the weight of the lower one.
    fn rows<'c>(&self, y: usize, cache: &'c mut [GridRow; 2]) -> (&'c [f32], &'c [f32], f32) {
        let (y0, y1, ty) = grid_sample(y, self.height, self.step, self.rows);
        if cache[1].index == Some(y0) {
            cache.swap(0, 1);
        }
        for (row, index) in cache.iter_mut().zip([y0, y1]) {
            if row.index != Some(index) {
                let cells = &self.cells[index * self.columns..][..self.columns];
                row.values.clear();
                row.values.extend(self.xs.iter().map(|&(x0, x1, tx)| {
                    f32::from(cells[x0]) * (1. - tx) + f32::from(cells[x1]) * tx
                }));
                row.index = Some(index);
            }
        }
        (&cache[0].values, &cache[1].values, ty)
    }
}

/// Interpolate at pooled pixel centres, including the last partial block.
fn grid_sample(pixel: usize, size: usize, step: usize, count: usize) -> (usize, usize, f32) {
    let center = |i: usize| (i * step + ((i + 1) * step).min(size) - 1) as f32 * 0.5;
    let x = pixel as f32;
    let i = (((x - center(0)) / step as f32).floor().max(0.) as usize).min(count - 1);
    let j = (i + 1).min(count - 1);
    let t = if i == j {
        0.
    } else {
        ((x - center(i)) / (center(j) - center(i))).clamp(0., 1.)
    };
    (i, j, t)
}

/// Moving extremum over a square window of `2 * radius + 1` pixels, clipped
/// to the image.
fn extrema_2d(input: &[u8], w: usize, h: usize, radius: usize, maximum: bool) -> Vec<u8> {
    let vertical = extrema_rows(input, w, h, radius, maximum);
    // Transpose the small analysis image so that both passes merge whole rows.
    let columns = transpose(&vertical, w, h);
    transpose(&extrema_rows(&columns, h, w, radius, maximum), h, w)
}

fn transpose(input: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut columns = vec![0; input.len()];
    columns
        .par_chunks_mut(h)
        .enumerate()
        .for_each(|(x, column)| {
            for (y, value) in column.iter_mut().enumerate() {
                *value = input[y * w + x];
            }
        });
    columns
}

/// Moving extremum along y (van Herk/Gil-Werman). Rows are grouped in blocks
/// of one window; every window is then a suffix of one block joined with a
/// prefix of the next, so each pass only merges adjacent rows.
fn extrema_rows(input: &[u8], w: usize, h: usize, radius: usize, maximum: bool) -> Vec<u8> {
    let merge = |row: &mut [u8], other: &[u8]| {
        let values = row.iter_mut().zip(other);
        if maximum {
            values.for_each(|(value, other)| *value = (*value).max(*other));
        } else {
            values.for_each(|(value, other)| *value = (*value).min(*other));
        }
    };
    // Neutral rows around the image keep every window complete.
    let (window, rows) = (2 * radius + 1, h + 2 * radius);
    let mut prefix = vec![if maximum { 0 } else { 255 }; rows * w];
    prefix[radius * w..][..input.len()].copy_from_slice(input);
    let mut suffix = prefix.clone();
    for y in (1..rows).filter(|y| y % window != 0) {
        let (above, row) = prefix.split_at_mut(y * w);
        merge(&mut row[..w], &above[(y - 1) * w..]);
    }
    for y in (0..rows - 1).rev().filter(|y| y % window != window - 1) {
        let (row, below) = suffix.split_at_mut((y + 1) * w);
        merge(&mut row[y * w..], &below[..w]);
    }
    suffix.truncate(input.len());
    for (row, prefix) in suffix.chunks_mut(w).zip(prefix[2 * radius * w..].chunks(w)) {
        merge(row, prefix);
    }
    suffix
}

/// Adaptive print tones, applied to the rectified and cropped grayscale master.
/// Local bright-population tails set paper white; dark print sets black.
/// Unlike `normalize_paper`, this display transform clips highlights.
/// Preserve the master when adjusting tones: weak ink overlapping paper texture
/// cannot be guaranteed to survive. No binary mask or foreground synthesis is used.
pub fn enhance_print(gray: &GrayImage) -> GrayImage {
    let mut normalized = normalize_paper(gray);
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    if w == 0 || h == 0 {
        return normalized;
    }
    // Estimate only low-frequency tone fields; never resize foreground pixels.
    let step = w.max(h).div_ceil(16).clamp(64, 256);
    let (bw, bh) = (w.div_ceil(step), h.div_ceil(step));
    let mut whites = vec![255_u8; bw * bh];
    let global = whites
        .par_chunks_mut(bw)
        .zip(normalized.par_chunks(w * step))
        .map(|(whites, band)| {
            let mut global = [0_usize; 256];
            for (bx, white) in whites.iter_mut().enumerate() {
                let columns = bx * step..((bx + 1) * step).min(w);
                let histogram = histogram(band.chunks(w).map(|row| &row[columns.clone()]));
                for (total, count) in global.iter_mut().zip(histogram) {
                    *total += count;
                }
                *white = paper_white(&histogram);
            }
            global
        })
        .reduce(
            || [0; 256],
            |a, b| std::array::from_fn(|value| a[value] + b[value]),
        );
    // Sparse ink or blank pages do not provide a reliable black reference.
    let dark: usize = global[..181].iter().sum();
    let black = if dark >= (w * h).div_ceil(100) {
        quantile(&global, 0, 180, (w * h).div_ceil(200)).min(160)
    } else {
        0
    };
    // The darkest page percentile may describe just a border or a few strong
    // strokes. Use the median of the dark population for the printed cores,
    // independently of how much blank paper surrounds the music.
    let ink_black = if dark >= (w * h).div_ceil(100) {
        quantile(&global, 0, 180, dark.div_ceil(2))
            .min(160)
            .max(black)
    } else {
        black
    };
    let white = Grid::new(&whites, [bw, bh], step, [w, h]);
    normalized
        .par_chunks_mut(w)
        .enumerate()
        .for_each_init(Default::default, |rows, (y, row)| {
            let (top, bottom, ty) = white.rows(y, rows);
            for ((value, top), bottom) in row.iter_mut().zip(top).zip(bottom) {
                *value = print_tone(*value, black, ink_black, top * (1. - ty) + bottom * ty);
            }
        });
    normalized
}

/// Value counts of a tile of at most `u32::MAX` pixels. Separate tables keep
/// runs of equal values from waiting on a single counter.
fn histogram<'a>(rows: impl Iterator<Item = &'a [u8]>) -> [usize; 256] {
    let mut tables = [[0_u32; 256]; 4];
    for row in rows {
        let quads = row.chunks_exact(4);
        for &value in quads.remainder() {
            tables[0][value as usize] += 1;
        }
        for quad in quads {
            for (table, &value) in tables.iter_mut().zip(quad) {
                table[value as usize] += 1;
            }
        }
    }
    std::array::from_fn(|value| tables.iter().map(|table| table[value] as usize).sum())
}

fn print_tone(value: u8, black: u8, ink_black: u8, white: f32) -> u8 {
    let range = white - f32::from(black);
    let base = 255. * (f32::from(value) - f32::from(black)) / range;
    // Anchor the stronger black at dark print, keeping existing bright tones
    // and white clipping exactly as before. This retains faint edge coverage
    // and avoids increasing the contrast of residual paper/show-through clouds.
    // Both curves are evaluated unconditionally so that rows vectorize.
    const SHOULDER: f32 = 192.;
    let dark_point = 255. * f32::from(ink_black - black) / range;
    let dark = SHOULDER * (base - dark_point) / (SHOULDER - dark_point).max(1.);
    let tone = if base < SHOULDER { dark } else { base };
    tone.round().clamp(0., 255.) as u8
}

fn paper_white(histogram: &[usize; 256]) -> u8 {
    // Textured paper can remain well below 220 after maximum-pooled lighting
    // correction. Estimate its actual bright mode instead of treating those
    // tiles as dense ink and leaving their paper gray.
    // A 3-bin sum reduces sensitivity to quantization; ties prefer brighter modes.
    let mode = (160_usize..=255)
        .max_by_key(|&v| {
            histogram[v.saturating_sub(1)..=(v + 1).min(255)]
                .iter()
                .sum::<usize>()
        })
        .unwrap();
    let low = mode.saturating_sub(20).max(160);
    let count: usize = histogram[low..].iter().sum();
    let total: usize = histogram.iter().sum();
    // Dense ink / very dark areas: retain the conservative illumination result.
    if count * 2 < total || count == 0 {
        return 255;
    }
    // Lower tail of the bright population maps most paper texture to white.
    // This intentionally clips uncertain highlights; it is not a foreground mask.
    quantile(histogram, low, 255, count.div_ceil(50))
}

fn quantile(histogram: &[usize; 256], low: usize, high: usize, rank: usize) -> u8 {
    let mut cumulative = 0;
    for (value, count) in histogram.iter().enumerate().take(high + 1).skip(low) {
        cumulative += count;
        if cumulative >= rank {
            return value as u8;
        }
    }
    high as u8
}

#[cfg(test)]
mod adaptive_tests {
    use super::*;
    use image::Luma;

    #[test]
    fn moving_extrema_match_the_clipped_window_definition() {
        for (w, h, radius) in [(1, 1, 3), (7, 5, 1), (23, 31, 2), (40, 17, 9), (19, 64, 40)] {
            let input: Vec<u8> = (0..w * h).map(|i| (i * 7919 % 251) as u8).collect();
            for maximum in [true, false] {
                let actual = extrema_2d(&input, w, h, radius, maximum);
                for y in 0..h {
                    for x in 0..w {
                        let window = (y.saturating_sub(radius)..(y + radius + 1).min(h))
                            .flat_map(|yy| {
                                (x.saturating_sub(radius)..(x + radius + 1).min(w))
                                    .map(move |xx| yy * w + xx)
                            })
                            .map(|i| input[i]);
                        let expected = if maximum { window.max() } else { window.min() };
                        assert_eq!(Some(actual[y * w + x]), expected, "{w}x{h} r{radius}");
                    }
                }
            }
        }
    }

    #[test]
    fn tile_histogram_counts_every_pixel_once() {
        let rows: Vec<Vec<u8>> = (0..9_usize)
            .map(|y| (0..y + 3).map(|x| (x * y % 5) as u8).collect())
            .collect();
        let actual = histogram(rows.iter().map(Vec::as_slice));
        let mut expected = [0; 256];
        for value in rows.iter().flatten() {
            expected[*value as usize] += 1;
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn textured_normalized_paper_below_220_still_sets_a_white_point() {
        // Same failure pattern as the photographed textured pages: an upper
        // paper population around 205 plus dark notation, not dense solid ink.
        let mut histogram = [0; 256];
        for (value, count) in histogram.iter_mut().enumerate().take(221).skip(185) {
            *count = 200 - value.abs_diff(205) * 5;
        }
        histogram[30] = 1000;
        let white = paper_white(&histogram);
        assert!((185..=205).contains(&white), "paper left gray: {white}");
        assert_eq!(print_tone(210, 0, 30, f32::from(white)), 255);
        assert!(print_tone(160, 0, 30, f32::from(white)) < 230);
    }

    #[test]
    fn dark_outliers_do_not_leave_the_main_print_washed_out() {
        let input = GrayImage::from_fn(1024, 512, |x, y| {
            Luma([if (20..80).contains(&x) && (20..80).contains(&y) {
                0
            } else if (80..900).contains(&x)
                && [150, 180, 210, 240, 270]
                    .iter()
                    .any(|&v| (v..v + 4).contains(&y))
            {
                88 + (x % 25) as u8
            } else if (100..900).contains(&x) && y == 320 {
                230
            } else {
                240
            }])
        });
        let output = enhance_print(&input);
        let dark = (80..900)
            .filter(|&x| output.get_pixel(x, 151)[0] < 20)
            .count();
        assert!(
            dark > 500,
            "main print stays gray: {dark}/820 near-black pixels"
        );
        assert_eq!(output.get_pixel(400, 100)[0], 255);
        assert_eq!(
            output.get_pixel(400, 320)[0],
            244,
            "changed faint highlight coverage"
        );
        let levels: std::collections::BTreeSet<_> =
            (80..900).map(|x| output.get_pixel(x, 151)[0]).collect();
        assert!(levels.len() > 5, "forced print to binary");
    }

    #[test]
    fn stronger_black_curve_is_monotone_and_keeps_bright_coverage() {
        for white in [220., 237.5, 255.] {
            for black in [0, 32, 100, 160] {
                for ink_black in [black, 160] {
                    let mut previous = 0;
                    for value in 0..=255 {
                        let output = print_tone(value, black, ink_black, white);
                        let baseline = print_tone(value, black, black, white);
                        assert!(output >= previous, "tone reversal at {value}");
                        assert!(output <= baseline, "brightened ink at {value}");
                        if baseline >= 193 {
                            assert_eq!(output, baseline, "changed bright coverage at {value}");
                        }
                        previous = output;
                    }
                    assert_eq!(previous, 255);
                }
            }
        }
    }

    #[test]
    fn broad_paper_blotches_are_removed_while_thin_ink_and_gray_heads_survive() {
        let input = GrayImage::from_fn(2048, 2048, |x, y| {
            // Illumination changes continuously, including where faint ink
            // crosses the edge of the cloud.
            let edge = x
                .saturating_sub(700)
                .min(880_u32.saturating_sub(x))
                .min(y.saturating_sub(700))
                .min(760_u32.saturating_sub(y));
            let paper = 240 - (25 * edge.min(12) / 12) as u8;
            let gray_head = (x as i32 - 850).pow(2) * 25 + (y as i32 - 734).pow(2) * 100 < 2500;
            Luma([if gray_head {
                paper - 40
            } else if (600..1000).contains(&x) && [704, 716, 728, 740, 752].contains(&y) {
                70
            } else if ((620..1050).contains(&x) && y == 720)
                || ((826..829).contains(&x) && (733..736).contains(&y))
            {
                paper - 12
            } else {
                paper
            }])
        });
        let output = enhance_print(&input);
        let white = (731..737)
            .flat_map(|y| (730..780).map(move |x| (x, y)))
            .filter(|&(x, y)| output.get_pixel(x, y)[0] == 255)
            .count();
        assert!(
            white >= 295,
            "paper cloud remains: {white}/300 white pixels"
        );
        for x in 620..1050 {
            assert!(output.get_pixel(x, 720)[0] < 250, "lost weak line at {x}");
        }
        assert!(output.get_pixel(827, 734)[0] < 250, "lost faint dot");
        assert!(
            output.get_pixel(850, 734)[0] < 245,
            "erased gray notehead core"
        );
    }

    #[test]
    fn background_window_does_not_follow_the_inside_of_large_noteheads() {
        let input = GrayImage::from_fn(2048, 2048, |x, y| {
            let inside = (i64::from(x) - 1000).pow(2) * 35 * 35
                + (i64::from(y) - 1000).pow(2) * 28 * 28
                < 28 * 28 * 35 * 35;
            Luma([if inside { 180 } else { 240 }])
        });
        let output = enhance_print(&input);
        assert!(
            output.get_pixel(1000, 1000)[0] < 210,
            "lifted notehead core"
        );
    }

    #[test]
    fn textured_paper_becomes_white_without_erasing_thin_ink() {
        for base in [150, 210] {
            let input = GrayImage::from_fn(320, 240, |x, y| {
                let noise = ((x * 17 + y * 31) % 9) as u8;
                let paper = base + noise;
                Luma([if y == 100 || x == 159 {
                    base - 30
                } else {
                    paper
                }])
            });
            let output = enhance_print(&input);
            let white = (10..90)
                .flat_map(|y| (10..150).map(move |x| (x, y)))
                .filter(|&(x, y)| output.get_pixel(x, y)[0] == 255)
                .count();
            assert!(
                white > 80 * 140 * 95 / 100,
                "paper stays gray at {base}: {white}"
            );
            for x in 10..310 {
                assert!(output.get_pixel(x, 100)[0] < 240);
            }
            for y in 10..230 {
                assert!(output.get_pixel(159, y)[0] < 240);
            }
        }
    }

    #[test]
    fn local_noise_and_shading_are_corrected_without_tile_seams_in_lines() {
        let input = GrayImage::from_fn(1027, 300, |x, y| {
            let paper = 140 + x * 80 / 1026;
            let noise = (x * 17 + y * 31) % (3 + x / 128);
            Luma([(if y == 149 || (x == 513 && (90..210).contains(&y)) {
                paper - 30
            } else {
                paper + noise
            }) as u8])
        });
        let output = enhance_print(&input);
        for x in 20..1007 {
            assert!(output.get_pixel(x, 149)[0] < 240, "lost line at {x}");
            assert!(
                output.get_pixel(x, 149)[0].abs_diff(output.get_pixel(x - 1, 149)[0]) < 5,
                "tone seam at {x}"
            );
        }
        for start in [20, 380, 750] {
            let white = (20..80)
                .flat_map(|y| (start..start + 200).map(move |x| (x, y)))
                .filter(|&(x, y)| output.get_pixel(x, y)[0] == 255)
                .count();
            assert!(white > 60 * 200 * 95 / 100, "paper at {start}: {white}");
        }
    }

    #[test]
    fn black_reference_is_separate_and_blank_images_are_stable() {
        for value in [0, 80, 180, 255] {
            let output = enhance_print(&GrayImage::from_pixel(128, 128, Luma([value])));
            assert!(
                output
                    .as_raw()
                    .iter()
                    .all(|&p| p == if value == 0 { 0 } else { 255 })
            );
        }
        assert_eq!(enhance_print(&GrayImage::new(0, 0)).dimensions(), (0, 0));
        let input = GrayImage::from_fn(256, 256, |x, y| {
            Luma([if (80..90).contains(&y) {
                90
            } else if x == 100 {
                180
            } else {
                240
            }])
        });
        let output = enhance_print(&input);
        assert_eq!(output.get_pixel(50, 85)[0], 0);
        assert!((1..255).contains(&output.get_pixel(100, 50)[0]));
    }

    #[test]
    fn clean_paper_keeps_one_level_coverage_and_is_not_forced_binary() {
        let mut input = GrayImage::from_pixel(180, 160, Luma([240]));
        for x in 20..160 {
            input.put_pixel(x, 80, Luma([239]));
        }
        input.put_pixel(90, 50, Luma([220]));
        let output = enhance_print(&input);
        assert_eq!(output.get_pixel(30, 30)[0], 255);
        assert!((1..255).contains(&output.get_pixel(90, 50)[0]));
        assert!((1..255).contains(&output.get_pixel(50, 80)[0]));
    }
}
