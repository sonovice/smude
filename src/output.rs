//! Output processing after geometric reconstruction; never detection input.
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

/// Transfer confirmed removed ink to gray output without thresholding valid
/// antialiased glyph edges. Halo pixels are removed only when closer to a
/// removed core than to retained ink; ties protect the retained foreground.
pub fn apply_binary_cleanup(gray: &mut GrayImage, raw: &GrayImage, cleaned: &GrayImage) -> usize {
    apply_binary_cleanup_impl(gray, raw, cleaned, None)
}

/// Reconstruct complete soft-gray rim halos from confirmed removed cores.
/// Propagation follows gray connectivity only near the true page perimeter;
/// retained ink and its AA fringe are barriers, and isolated weak glyphs remain.
pub fn apply_binary_cleanup_with_contour(
    gray: &mut GrayImage,
    raw: &GrayImage,
    cleaned: &GrayImage,
    contour: &[crate::Point],
) -> usize {
    apply_binary_cleanup_impl(gray, raw, cleaned, Some(contour))
}

fn apply_binary_cleanup_impl(
    gray: &mut GrayImage,
    raw: &GrayImage,
    cleaned: &GrayImage,
    contour: Option<&[crate::Point]>,
) -> usize {
    assert_eq!(gray.dimensions(), raw.dimensions());
    assert_eq!(raw.dimensions(), cleaned.dimensions());
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    // The soft transition can reveal a wider gray paper-edge halo than the
    // binary core. Extend only from confirmed removed ink; retained foreground
    // wins the distance competition, including its antialiased fringe.
    let cap: u8 = ((w.min(h) / 80).clamp(4, 64) + 1) as u8;
    let mut removed = vec![cap; w * h];
    let mut kept = vec![cap; w * h];
    let mut any = false;
    for i in 0..w * h {
        if raw.as_raw()[i] == 0 {
            if cleaned.as_raw()[i] == 255 {
                removed[i] = 0;
                any = true;
            } else {
                kept[i] = 0;
            }
        }
    }
    if !any || w == 0 || h == 0 {
        return 0;
    }
    let distance = |d: &mut [u8]| {
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let mut v = d[i];
                if x > 0 {
                    v = v.min(d[i - 1] + 1);
                }
                if y > 0 {
                    v = v.min(d[i - w] + 1);
                    if x > 0 {
                        v = v.min(d[i - w - 1] + 1);
                    }
                    if x + 1 < w {
                        v = v.min(d[i - w + 1] + 1);
                    }
                }
                d[i] = v.min(cap);
            }
        }
        for y in (0..h).rev() {
            for x in (0..w).rev() {
                let i = y * w + x;
                let mut v = d[i];
                if x + 1 < w {
                    v = v.min(d[i + 1] + 1);
                }
                if y + 1 < h {
                    v = v.min(d[i + w] + 1);
                    if x > 0 {
                        v = v.min(d[i + w - 1] + 1);
                    }
                    if x + 1 < w {
                        v = v.min(d[i + w + 1] + 1);
                    }
                }
                d[i] = v.min(cap);
            }
        }
    };
    rayon::join(|| distance(&mut removed), || distance(&mut kept));
    let mut count = 0;
    if let Some(contour) =
        contour.filter(|p| p.len() >= 3 && p.iter().all(|p| p.x.is_finite() && p.y.is_finite()))
    {
        let band = ((w.min(h) as f64 * 0.04).ceil() as u16).max(2);
        let boundary = distance_to_contour(contour, w, h, band + 1);
        let mut seen = vec![false; w * h];
        let mut queue = vec![];
        // This is seeded reconstruction, not a wider global eroded mask. A
        // distant gray halo is removed only if connected to a proved rim core.
        for i in 0..w * h {
            if removed[i] == 0 && gray.as_raw()[i] < 250 && boundary[i] <= band && kept[i] > 3 {
                seen[i] = true;
                queue.push(i);
            }
        }
        let mut cursor = 0;
        while cursor < queue.len() {
            let i = queue[cursor];
            cursor += 1;
            if gray.as_raw()[i] != 255 {
                count += 1;
                gray.as_mut()[i] = 255;
            }
            let (x, y) = (i % w, i / w);
            for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    let j = yy * w + xx;
                    if !seen[j] && gray.as_raw()[j] < 250 && boundary[j] <= band && kept[j] > 3 {
                        seen[j] = true;
                        queue.push(j);
                    }
                }
            }
        }
    }
    for i in 0..w * h {
        if removed[i] < cap && removed[i] < kept[i] {
            if gray.as_raw()[i] != 255 {
                count += 1;
            }
            gray.as_mut()[i] = 255;
        }
    }
    count
}

/// Remove faint perimeter strokes which never produced a Sauvola black core.
/// Shape classification uses normalized gray, and protected music rectangles
/// are restored exactly. No global inset or content crop is introduced.
pub fn whiten_gray_border(
    gray: &mut GrayImage,
    contour: &[crate::Point],
    margin: u32,
    protected: &[[f64; 4]],
) -> usize {
    let before = gray.clone();
    let raw = GrayImage::from_fn(gray.width(), gray.height(), |x, y| {
        let protected_pixel = protected.iter().any(|r| {
            f64::from(x) >= r[0]
                && f64::from(x) <= r[2]
                && f64::from(y) >= r[1]
                && f64::from(y) <= r[3]
        });
        image::Luma([if !protected_pixel && gray.get_pixel(x, y)[0] < 240 {
            0
        } else {
            255
        }])
    });
    let mut cleaned = raw.clone();
    whiten_page_border_impl(&mut cleaned, contour, margin, true, Some(gray.as_raw()));
    apply_binary_cleanup_with_contour(gray, &raw, &cleaned, contour);
    let mut count = 0;
    for (x, y, pixel) in gray.enumerate_pixels_mut() {
        if protected.iter().any(|r| {
            f64::from(x) >= r[0]
                && f64::from(x) <= r[2]
                && f64::from(y) >= r[1]
                && f64::from(y) <= r[3]
        }) {
            *pixel = *before.get_pixel(x, y);
        } else if *pixel != *before.get_pixel(x, y) {
            count += 1;
        }
    }
    count
}

/// Capped Chebyshev distance seeded from the actual mapped perimeter. The
/// image canvas is deliberately not treated as a physical page boundary.
fn distance_to_contour(contour: &[crate::Point], w: usize, h: usize, cap: u16) -> Vec<u16> {
    let mut d = vec![cap; w * h];
    for (a, b) in contour
        .iter()
        .zip(contour.iter().cycle().skip(1))
        .take(contour.len())
    {
        let n = (a.x - b.x).abs().max((a.y - b.y).abs()).ceil().max(1.) as usize;
        for j in 0..=n {
            let t = j as f64 / n as f64;
            let x = (a.x + (b.x - a.x) * t).round() as i64;
            let y = (a.y + (b.y - a.y) * t).round() as i64;
            if x >= 0 && y >= 0 && x < w as i64 && y < h as i64 {
                d[y as usize * w + x as usize] = 0;
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let mut v = d[i];
            if x > 0 {
                v = v.min(d[i - 1] + 1);
            }
            if y > 0 {
                v = v.min(d[i - w] + 1);
                if x > 0 {
                    v = v.min(d[i - w - 1] + 1);
                }
                if x + 1 < w {
                    v = v.min(d[i - w + 1] + 1);
                }
            }
            d[i] = v.min(cap);
        }
    }
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = y * w + x;
            let mut v = d[i];
            if x + 1 < w {
                v = v.min(d[i + 1] + 1);
            }
            if y + 1 < h {
                v = v.min(d[i + w] + 1);
                if x > 0 {
                    v = v.min(d[i + w - 1] + 1);
                }
                if x + 1 < w {
                    v = v.min(d[i + w + 1] + 1);
                }
            }
            d[i] = v.min(cap);
        }
    }
    d
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

/// Whiten outside a mapped page polygon and a narrow inward border band.
/// Scanline filling plus two separable minimum filters avoids inverse-map calls
/// per output pixel. Coordinates are final output raster pixels.
pub fn whiten_page_border(binary: &mut GrayImage, contour: &[crate::Point], margin: u32) -> usize {
    whiten_page_border_impl(binary, contour, margin, false, None)
}

fn whiten_page_border_impl(
    binary: &mut GrayImage,
    contour: &[crate::Point],
    margin: u32,
    gray_frames: bool,
    gray_tone: Option<&[u8]>,
) -> usize {
    use std::collections::VecDeque;
    let (w, h) = (binary.width() as usize, binary.height() as usize);
    if contour.len() < 3
        || w == 0
        || h == 0
        || contour.iter().any(|p| !p.x.is_finite() || !p.y.is_finite())
    {
        return 0;
    }
    let mut mask = vec![0_u8; w * h];
    for y in 0..h {
        let yy = y as f64;
        let mut cuts = vec![];
        for (a, b) in contour
            .iter()
            .zip(contour.iter().cycle().skip(1))
            .take(contour.len())
        {
            if (a.y > yy) != (b.y > yy) {
                cuts.push(a.x + (yy - a.y) * (b.x - a.x) / (b.y - a.y));
            }
        }
        cuts.sort_by(f64::total_cmp);
        for pair in cuts.chunks_exact(2) {
            let start = pair[0].ceil().max(0.) as usize;
            let end = (pair[1].floor() + 1.).max(0.) as usize;
            if start < end.min(w) {
                mask[y * w + start..y * w + end.min(w)].fill(1);
            }
        }
    }
    // Classify the original connected ink before erosion can split a frame
    // into detached, character-sized fragments.
    let mut removed = if margin > 0 {
        remove_rim_components(binary, &mask, contour, margin, gray_frames, gray_tone)
    } else {
        0
    };
    let radius = margin as usize;
    // Erode in output pixels; outside the raster is outside the page.
    fn erode_line(input: &[u8], radius: usize) -> Vec<u8> {
        let n = input.len();
        let mut out = vec![0; n];
        let mut zeros = VecDeque::new();
        let mut right = 0;
        for (i, value) in out.iter_mut().enumerate() {
            let end = i.saturating_add(radius).saturating_add(1).min(n);
            while right < end {
                if input[right] == 0 {
                    zeros.push_back(right);
                }
                right += 1;
            }
            let left = i.saturating_sub(radius);
            while zeros.front().is_some_and(|j| *j < left) {
                zeros.pop_front();
            }
            if i >= radius && i.saturating_add(radius) < n && zeros.is_empty() {
                *value = 1;
            }
        }
        out
    }
    if radius > 0 {
        let mut horizontal = vec![0; w * h];
        horizontal
            .par_chunks_mut(w)
            .zip(mask.par_chunks(w))
            .for_each(|(out, row)| out.copy_from_slice(&erode_line(row, radius)));
        for x in 0..w {
            let column: Vec<u8> = (0..h).map(|y| horizontal[y * w + x]).collect();
            let result = erode_line(&column, radius);
            for y in 0..h {
                mask[y * w + x] = result[y];
            }
        }
    }
    for (value, inside) in binary.as_mut().iter_mut().zip(&mask) {
        if *inside == 0 {
            if *value == 0 {
                removed += 1;
            }
            *value = 255;
        }
    }
    removed
}

/// Entire boundary-following strokes can lie well inside an estimated outline.
/// Distance to the actual (possibly curved) mask, not a rectangular crop, limits
/// component selection. Compact glyphs and components entering the interior stay.
fn remove_rim_components(
    binary: &mut GrayImage,
    mask: &[u8],
    contour: &[crate::Point],
    margin: u32,
    gray_frames: bool,
    gray_tone: Option<&[u8]>,
) -> usize {
    let (w, h) = (binary.width() as usize, binary.height() as usize);
    let band = ((w.min(h) as f64 * 0.04).ceil() as u16).max(2);
    let cap = band + 1;
    let close_band = (margin.saturating_mul(3).min(u32::from(band))) as u16;
    let mut distance: Vec<u16> = mask.iter().map(|&v| if v == 0 { 0 } else { cap }).collect();
    // A crop edge can cut through page interior. Seed only the actual mapped
    // perimeter (and confirmed outside mask), never arbitrary canvas edges.
    for (a, b) in contour
        .iter()
        .zip(contour.iter().cycle().skip(1))
        .take(contour.len())
    {
        let n = ((a.x - b.x).abs().max((a.y - b.y).abs()).ceil() as usize).max(1);
        for j in 0..=n {
            let t = j as f64 / n as f64;
            let x = (a.x + (b.x - a.x) * t).round() as i64;
            let y = (a.y + (b.y - a.y) * t).round() as i64;
            if x >= 0 && y >= 0 && x < w as i64 && y < h as i64 {
                distance[y as usize * w + x as usize] = 0;
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let mut d = distance[i];
            if x > 0 {
                d = d.min(distance[i - 1] + 1);
            }
            if y > 0 {
                d = d.min(distance[i - w] + 1);
                if x > 0 {
                    d = d.min(distance[i - w - 1] + 1);
                }
                if x + 1 < w {
                    d = d.min(distance[i - w + 1] + 1);
                }
            }
            distance[i] = d;
        }
    }
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = y * w + x;
            let mut d = distance[i];
            if x + 1 < w {
                d = d.min(distance[i + 1] + 1);
            }
            if y + 1 < h {
                d = d.min(distance[i + w] + 1);
                if x > 0 {
                    d = d.min(distance[i + w - 1] + 1);
                }
                if x + 1 < w {
                    d = d.min(distance[i + w + 1] + 1);
                }
            }
            distance[i] = d;
        }
    }
    let mut seen = vec![false; w * h];
    let mut component = vec![];
    let mut confirmed = vec![];
    let mut pending: Vec<Vec<usize>> = vec![];
    let mut corner_fragments = vec![];
    let mut removed = 0;
    let short = w.min(h);
    for seed in 0..w * h {
        if seen[seed] || binary.as_raw()[seed] != 0 || (gray_frames && distance[seed] > band) {
            continue;
        }
        component.clear();
        component.push(seed);
        seen[seed] = true;
        let (mut min_x, mut max_x, mut min_y, mut max_y) = (w, 0, h, 0);
        let mut entirely_near = true;
        let mut max_distance = 0;
        let mut cursor = 0;
        while cursor < component.len() {
            let i = component[cursor];
            cursor += 1;
            let (x, y) = (i % w, i / w);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            entirely_near &= distance[i] <= band;
            max_distance = max_distance.max(distance[i]);
            for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    let j = yy * w + xx;
                    if !seen[j] && binary.as_raw()[j] == 0 && (!gray_frames || distance[j] <= band)
                    {
                        seen[j] = true;
                        component.push(j);
                    }
                }
            }
        }
        let (bw, bh) = (max_x - min_x + 1, max_y - min_y + 1);
        let span = bw.max(bh);
        let thickness = bw.min(bh);
        let elongated = span >= 3 * thickness;
        let curved_frame = span >= (short / 16).max(8)
            && thickness >= (short / 25).max(4)
            && component.len() <= (bw + bh) * (margin as usize * 2).max(3);
        // Tiny fragments immediately behind the shaved contour belong to the
        // same uncertain paper edge. Further in, require a long boundary stroke;
        // a compact page number (including an isolated "1") is not a rim.
        let compact_glyph = bw >= 5 && bh >= 8 && span <= (short / 20).max(8);
        let tiny_edge_fleck = max_distance <= close_band && component.len() <= 8;
        // A multi-sided frame has a large bounding box in both dimensions.
        // Only gray cleanup enables this extra hypothesis: every component
        // pixel must still be near the actual paper perimeter, and occupancy
        // must be sparse relative to its page-sized box and perimeter band.
        let multi_side_frame = gray_frames
            && span >= short / 2
            && thickness >= short / 3
            && component.len() * 8 <= bw * bh
            && component.len() <= (bw + bh) * usize::from(band) * 3 / 5;
        let boundary_stroke = !compact_glyph
            && span >= (short / 40).max(8)
            && (elongated || curved_frame || multi_side_frame);
        let corner_wedge = gray_frames
            && entirely_near
            && is_corner_wedge(&component, w, [min_x, min_y, max_x, max_y], contour, band);
        if corner_wedge {
            corner_fragments.push([min_x, min_y, max_x, max_y]);
        }
        if entirely_near && (tiny_edge_fleck || boundary_stroke || corner_wedge) {
            if corner_wedge || (boundary_stroke && span >= short / 10) {
                confirmed.extend(component.iter().copied());
            }
            for &i in &component {
                binary.as_mut()[i] = 255;
            }
            removed += component.len();
        } else if entirely_near {
            pending.push(component.clone());
        }
    }
    // Several character-sized faint fragments can collectively establish a
    // broken rim. Keep that evidence separate: it must not authorize removal
    // of a dark folio or lyric merely because it lies in the same normal strip.
    let chain_seed = gray_tone
        .map(|tone| gray_fragment_chains(&pending, tone, contour, w, short, margin, &distance));
    let mut chain_confirmed = vec![];
    if let Some(seeds) = chain_seed {
        let mut index = 0;
        pending.retain(|points| {
            let accepted = seeds[index];
            index += 1;
            if accepted {
                for &i in points {
                    binary.as_mut()[i] = 255;
                }
                removed += points.len();
                chain_confirmed.extend(points.iter().copied());
            }
            !accepted
        });
    }
    // Recover short strokes interrupted by white gaps, but only when they
    // continue a confirmed long rim in the same narrow normal-distance strip.
    // Compact isolated numerals have no supporting rim and remain untouched.
    let gap = (short / 12).max(8);
    loop {
        let mut changed = false;
        pending.retain(|points| {
            // Thin detached paper fibers immediately beside an accepted solid
            // corner inherit that evidence, but compact nearby glyphs do not.
            let min_x = points.iter().map(|i| i % w).min().unwrap();
            let max_x = points.iter().map(|i| i % w).max().unwrap();
            let min_y = points.iter().map(|i| i / w).min().unwrap();
            let max_y = points.iter().map(|i| i / w).max().unwrap();
            let (bw, bh) = (max_x - min_x + 1, max_y - min_y + 1);
            let pad = margin.max(2) as usize;
            let corner_fiber = bw.min(bh) <= pad
                && (bw.max(bh) >= 3 * bw.min(bh) || points.len() <= 8)
                && corner_fragments.iter().any(|[x0, y0, x1, y1]| {
                    min_x >= x0.saturating_sub(pad)
                        && min_y >= y0.saturating_sub(pad)
                        && max_x <= x1.saturating_add(pad)
                        && max_y <= y1.saturating_add(pad)
                });
            if corner_fiber {
                for &i in points {
                    binary.as_mut()[i] = 255;
                }
                removed += points.len();
                changed = true;
                return false;
            }
            let n = points.len() as f64;
            let cx = points.iter().map(|i| (i % w) as f64).sum::<f64>() / n;
            let cy = points.iter().map(|i| (i / w) as f64).sum::<f64>() / n;
            let mut tangent = (0., 0.);
            let mut closest = f64::INFINITY;
            for (a, b) in contour
                .iter()
                .zip(contour.iter().cycle().skip(1))
                .take(contour.len())
            {
                let dx = b.x - a.x;
                let dy = b.y - a.y;
                let norm = dx * dx + dy * dy;
                if norm < 1e-9 {
                    continue;
                }
                let t = (((cx - a.x) * dx + (cy - a.y) * dy) / norm).clamp(0., 1.);
                let d = (cx - a.x - t * dx).powi(2) + (cy - a.y - t * dy).powi(2);
                if d < closest {
                    closest = d;
                    let length = norm.sqrt();
                    tangent = (dx / length, dy / length);
                }
            }
            let (tx, ty) = tangent;
            let (mut along, mut across) = (0., 0.);
            for &i in points {
                let dx = (i % w) as f64 - cx;
                let dy = (i / w) as f64 - cy;
                along += (dx * tx + dy * ty).powi(2);
                across += (-dx * ty + dy * tx).powi(2);
            }
            if points.len() < 3 || along < 2. * across || along / n < 2. {
                return true;
            }
            let normal = points.iter().map(|i| f64::from(distance[*i])).sum::<f64>() / n;
            let minx = points.iter().map(|i| i % w).min().unwrap();
            let maxx = points.iter().map(|i| i % w).max().unwrap();
            let miny = points.iter().map(|i| i / w).min().unwrap();
            let maxy = points.iter().map(|i| i / w).max().unwrap();
            let supported_by = |support: &[usize]| {
                support.iter().any(|&i| {
                    let (x, y) = (i % w, i / w);
                    x.saturating_add(gap) >= minx
                        && x <= maxx.saturating_add(gap)
                        && y.saturating_add(gap) >= miny
                        && y <= maxy.saturating_add(gap)
                        && (f64::from(distance[i]) - normal).abs() <= f64::from(margin.max(2))
                })
            };
            let supported = supported_by(&confirmed);
            let chain_supported = gray_tone
                .is_some_and(|tone| faint_fragment_tone(points, tone).is_some())
                && supported_by(&chain_confirmed);
            if supported || chain_supported {
                for &i in points {
                    binary.as_mut()[i] = 255;
                }
                removed += points.len();
                if supported {
                    confirmed.extend(points.iter().copied());
                } else {
                    chain_confirmed.extend(points.iter().copied());
                }
                changed = true;
                false
            } else {
                true
            }
        });
        if !changed {
            break;
        }
    }
    removed
}

fn faint_fragment_tone(points: &[usize], tone: &[u8]) -> Option<f64> {
    if points.len() < 3 || points.iter().any(|i| tone[*i] < 80) {
        return None;
    }
    let mean = points.iter().map(|i| f64::from(tone[*i])).sum::<f64>() / points.len() as f64;
    (100.0..=225.0).contains(&mean).then_some(mean)
}

/// Bootstrap a rim only from a coherent chain, not from the fact that a small
/// isolated mark happens to be near paper. All candidate pixels are outside
/// protected music because the gray caller excluded those before components.
fn gray_fragment_chains(
    pending: &[Vec<usize>],
    tone: &[u8],
    contour: &[crate::Point],
    w: usize,
    short: usize,
    margin: u32,
    distance: &[u16],
) -> Vec<bool> {
    struct Fragment {
        index: usize,
        center: (f64, f64),
        tangent: (f64, f64),
        span: f64,
        normal: f64,
        tone: f64,
    }
    let mut fragments = vec![];
    for (index, points) in pending.iter().enumerate() {
        let Some(mean_tone) = faint_fragment_tone(points, tone) else {
            continue;
        };
        let n = points.len() as f64;
        let center = (
            points.iter().map(|i| (i % w) as f64).sum::<f64>() / n,
            points.iter().map(|i| (i / w) as f64).sum::<f64>() / n,
        );
        let mut tangent = (0., 0.);
        let mut closest = f64::INFINITY;
        for (a, b) in contour
            .iter()
            .zip(contour.iter().cycle().skip(1))
            .take(contour.len())
        {
            let (dx, dy) = (b.x - a.x, b.y - a.y);
            let norm = dx * dx + dy * dy;
            if norm < 1e-9 {
                continue;
            }
            let t = (((center.0 - a.x) * dx + (center.1 - a.y) * dy) / norm).clamp(0., 1.);
            let d = (center.0 - a.x - t * dx).powi(2) + (center.1 - a.y - t * dy).powi(2);
            if d < closest {
                closest = d;
                let length = norm.sqrt();
                tangent = (dx / length, dy / length);
            }
        }
        let (mut lo, mut hi, mut nl, mut nh) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for &i in points {
            let (dx, dy) = ((i % w) as f64 - center.0, (i / w) as f64 - center.1);
            let t = dx * tangent.0 + dy * tangent.1;
            let r = -dx * tangent.1 + dy * tangent.0;
            lo = lo.min(t);
            hi = hi.max(t);
            nl = nl.min(r);
            nh = nh.max(r);
        }
        let span = hi - lo + 1.;
        let thickness = nh - nl + 1.;
        if span < 6.
            || span < 2. * thickness
            || thickness > f64::from(margin.saturating_add(1).max(3))
        {
            continue;
        }
        let normal = points.iter().map(|i| f64::from(distance[*i])).sum::<f64>() / n;
        // Outside texture and the already-eroded immediate edge do not need
        // chain inference and can contain many irrelevant small components.
        if normal <= f64::from(margin.max(1)) {
            continue;
        }
        fragments.push(Fragment {
            index,
            center,
            tangent,
            span,
            normal,
            tone: mean_tone,
        });
    }
    let mut parent: Vec<_> = (0..fragments.len()).collect();
    fn root(parent: &[usize], mut i: usize) -> usize {
        while parent[i] != i {
            i = parent[i];
        }
        i
    }
    let tolerance = f64::from(margin.max(2));
    for i in 0..fragments.len() {
        for j in i + 1..fragments.len() {
            let (a, b) = (&fragments[i], &fragments[j]);
            let (dx, dy) = (b.center.0 - a.center.0, b.center.1 - a.center.1);
            let along = (dx * a.tangent.0 + dy * a.tangent.1).abs();
            let across = (-dx * a.tangent.1 + dy * a.tangent.0).abs();
            let parallel = (a.tangent.0 * b.tangent.0 + a.tangent.1 * b.tangent.1).abs();
            let gap = along - (a.span + b.span) / 2.;
            if parallel > 0.995
                && across <= tolerance
                && (a.normal - b.normal).abs() <= tolerance
                && (a.tone - b.tone).abs() <= 25.
                && gap <= (short / 25).max(8) as f64
            {
                let (a, b) = (root(&parent, i), root(&parent, j));
                parent[b] = a;
            }
        }
    }
    let mut accepted = vec![false; pending.len()];
    for i in 0..fragments.len() {
        if root(&parent, i) != i {
            continue;
        }
        let chain: Vec<_> = (0..fragments.len())
            .filter(|j| root(&parent, *j) == i)
            .map(|j| &fragments[j])
            .collect();
        if chain.len() < 3 {
            continue;
        }
        let t = chain[0].tangent;
        let origin = chain[0].center;
        let (mut lo, mut hi, mut nl, mut nh) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for f in &chain {
            let (dx, dy) = (f.center.0 - origin.0, f.center.1 - origin.1);
            let p = dx * t.0 + dy * t.1;
            let r = -dx * t.1 + dy * t.0;
            lo = lo.min(p - f.span / 2.);
            hi = hi.max(p + f.span / 2.);
            nl = nl.min(r);
            nh = nh.max(r);
        }
        let span = hi - lo;
        let coverage = chain.iter().map(|f| f.span).sum::<f64>() / span;
        let tone_range = chain
            .iter()
            .map(|f| f.tone)
            .fold(f64::NEG_INFINITY, f64::max)
            - chain.iter().map(|f| f.tone).fold(f64::INFINITY, f64::min);
        if span >= (short / 15).max(24) as f64
            && coverage >= 0.4
            && nh - nl <= tolerance
            && tone_range <= 25.
        {
            for f in chain {
                accepted[f.index] = true;
            }
        }
    }
    accepted
}

/// Solid tapered corner fragments differ from compact printed glyphs: they
/// have two long outer legs facing a real contour corner and almost no holes.
fn is_corner_wedge(
    pixels: &[usize],
    width: usize,
    bounds: [usize; 4],
    contour: &[crate::Point],
    band: u16,
) -> bool {
    let [x0, y0, x1, y1] = bounds;
    let (bw, bh) = (x1 - x0 + 1, y1 - y0 + 1);
    let band = usize::from(band);
    if bw.min(bh) < 5 || bw.max(bh) < 10 || bw.max(bh) > 2 * band || pixels.len() * 10 > bw * bh * 7
    {
        return false;
    }
    let near_corner = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)]
        .iter()
        .any(|&(sx, sy)| {
            let corner = contour
                .iter()
                .max_by(|a, b| (sx * a.x + sy * a.y).total_cmp(&(sx * b.x + sy * b.y)))
                .unwrap();
            let nx = if sx < 0. { x0 } else { x1 };
            let ny = if sy < 0. { y0 } else { y1 };
            (nx as f64 - corner.x).abs() <= (2 * band) as f64
                && (ny as f64 - corner.y).abs() <= (2 * band) as f64
        });
    if near_corner && solid_triangle(pixels, width) {
        return true;
    }
    let mut left = vec![usize::MAX; bh];
    let mut right = vec![0; bh];
    for &i in pixels {
        let (x, y) = (i % width, i / width - y0);
        left[y] = left[y].min(x);
        right[y] = right[y].max(x);
    }
    let filled_span: usize = left
        .iter()
        .zip(&right)
        .filter(|(l, _)| **l != usize::MAX)
        .map(|(l, r)| r - l + 1)
        .sum();
    if pixels.len() * 10 < filled_span * 9 {
        return false;
    }
    let tolerance = (band / 20).max(2);
    for (sx, sy) in [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)] {
        let Some(corner) = contour
            .iter()
            .max_by(|a, b| (sx * a.x + sy * a.y).total_cmp(&(sx * b.x + sy * b.y)))
        else {
            continue;
        };
        let near_x = if sx < 0. { x0 } else { x1 };
        let near_y = if sy < 0. { y0 } else { y1 };
        if (near_x as f64 - corner.x).abs() > band as f64
            || (near_y as f64 - corner.y).abs() > band as f64
        {
            continue;
        }
        let mut rows = vec![false; bh];
        let mut columns = vec![false; bw];
        for &i in pixels {
            let (x, y) = (i % width, i / width);
            if x.abs_diff(near_x) <= tolerance {
                rows[y - y0] = true;
            }
            if y.abs_diff(near_y) <= tolerance {
                columns[x - x0] = true;
            }
        }
        if rows.iter().filter(|&&v| v).count() * 10 >= bh * 8
            && columns.iter().filter(|&&v| v).count() * 10 >= bw * 8
        {
            return true;
        }
    }
    false
}

/// Convex-hull solidity and triangularity handle rotated corner wedges without
/// turning a filled rectangle, hollow numeral, or thin bent glyph into a wedge.
fn solid_triangle(pixels: &[usize], width: usize) -> bool {
    if pixels.len() < 30 {
        return false;
    }
    let mut p: Vec<(f64, f64)> = pixels
        .iter()
        .map(|i| ((i % width) as f64, (i / width) as f64))
        .collect();
    p.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    p.dedup();
    fn cross(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
        (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
    }
    let mut hull = vec![];
    for &q in &p {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], q) <= 0. {
            hull.pop();
        }
        hull.push(q);
    }
    let n = hull.len();
    for &q in p.iter().rev().skip(1) {
        while hull.len() > n && cross(hull[hull.len() - 2], hull[hull.len() - 1], q) <= 0. {
            hull.pop();
        }
        hull.push(q);
    }
    hull.pop();
    if hull.len() < 3 {
        return false;
    }
    let area = hull
        .iter()
        .zip(hull.iter().cycle().skip(1))
        .take(hull.len())
        .map(|(a, b)| a.0 * b.1 - a.1 * b.0)
        .sum::<f64>()
        .abs()
        / 2.;
    if area < 20. || (pixels.len() as f64) < area * 0.78 {
        return false;
    }
    let mut triangle: f64 = 0.;
    for i in 0..hull.len() {
        for j in i + 1..hull.len() {
            for k in j + 1..hull.len() {
                triangle = triangle.max(cross(hull[i], hull[j], hull[k]).abs() / 2.);
            }
        }
    }
    triangle >= area * 0.72
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn shaded_ink_matches_direct_replicated_window_reference() {
        let gray = GrayImage::from_fn(57, 39, |x, y| {
            Luma([
                if (8..=48).contains(&x) && [6, 12, 18, 24, 30].contains(&y) {
                    18
                } else {
                    (90 + x * 2 + y / 2) as u8
                },
            ])
        });
        let binary = binarize_sauvola(&gray);
        for y in 0..gray.height() {
            for x in 0..gray.width() {
                let (mut total, mut squared) = (0., 0.);
                for dy in -25_i64..=25 {
                    for dx in -25_i64..=25 {
                        let xx = (i64::from(x) + dx).clamp(0, i64::from(gray.width()) - 1) as u32;
                        let yy = (i64::from(y) + dy).clamp(0, i64::from(gray.height()) - 1) as u32;
                        let v = f64::from(gray.get_pixel(xx, yy)[0]);
                        total += v;
                        squared += v * v;
                    }
                }
                let m = total / 2601.;
                let sd = (squared / 2601. - m * m).max(0.).sqrt();
                let expected =
                    if f64::from(gray.get_pixel(x, y)[0]) <= m * (1. + 0.25 * (sd / 128. - 1.)) {
                        0
                    } else {
                        255
                    };
                assert_eq!(binary.get_pixel(x, y)[0], expected, "{x},{y}");
            }
        }
        assert_eq!(binary.dimensions(), gray.dimensions());
        assert_eq!(binary.get_pixel(20, 18)[0], 0);
        assert_eq!(binary.get_pixel(20, 19)[0], 255);
    }

    #[test]
    fn tiny_constant_images_have_no_border_artifacts() {
        for tone in [0, 60, 128, 255] {
            let binary = binarize_sauvola(&GrayImage::from_pixel(2, 3, Luma([tone])));
            assert!(
                binary
                    .as_raw()
                    .iter()
                    .all(|p| *p == if tone == 0 { 0 } else { 255 })
            );
        }
    }
}

#[cfg(test)]
mod border_tests {
    use super::*;
    use crate::Point;
    use image::Luma;

    #[test]
    fn masks_outside_and_thin_border_without_flooding_attached_content() {
        let contour = vec![
            Point { x: 3., y: 3. },
            Point { x: 36., y: 3. },
            Point { x: 36., y: 36. },
            Point { x: 3., y: 36. },
        ];
        let mut binary = GrayImage::from_pixel(40, 40, Luma([255]));
        // A border stroke is connected to an interior symbol. Only the portion
        // in the excluded band must disappear, not the entire component.
        for x in 0..20 {
            binary.put_pixel(x, 20, Luma([0]));
        }
        binary.put_pixel(20, 22, Luma([0]));
        let count = whiten_page_border(&mut binary, &contour, 2);
        assert!(count > 0);
        for x in 0..5 {
            assert_eq!(binary.get_pixel(x, 20)[0], 255);
        }
        for x in 5..20 {
            assert_eq!(binary.get_pixel(x, 20)[0], 0);
        }
        assert_eq!(binary.get_pixel(20, 22)[0], 0);
        assert_eq!(binary.dimensions(), (40, 40));
    }

    #[test]
    fn curved_polygon_interior_and_zero_margin_are_respected() {
        let contour = vec![
            Point { x: 15., y: 2. },
            Point { x: 28., y: 15. },
            Point { x: 15., y: 28. },
            Point { x: 2., y: 15. },
        ];
        let mut binary = GrayImage::from_pixel(30, 30, Luma([0]));
        whiten_page_border(&mut binary, &contour, 0);
        assert_eq!(binary.get_pixel(15, 15)[0], 0);
        assert_eq!(binary.get_pixel(3, 3)[0], 255);
        assert_eq!(binary.get_pixel(26, 26)[0], 255);
    }
}

#[cfg(test)]
mod rim_regression {
    use super::*;
    use crate::Point;
    use image::Luma;

    #[test]
    fn short_curved_corner_frame_is_removed_as_one_object() {
        let contour = vec![
            Point { x: 0., y: 0. },
            Point { x: 599., y: 0. },
            Point { x: 599., y: 799. },
            Point { x: 0., y: 799. },
        ];
        let mut binary = GrayImage::from_pixel(600, 800, Luma([255]));
        for y in 700..783 {
            for x in 10..13 {
                binary.put_pixel(x, y, Luma([0]));
            }
        }
        for x in 10..90 {
            for y in 780..783 {
                binary.put_pixel(x, y, Luma([0]));
            }
        }
        whiten_page_border(&mut binary, &contour, 2);
        assert!(binary.as_raw().iter().all(|v| *v == 255));
    }

    #[test]
    fn short_fragment_is_removed_only_with_confirmed_aligned_rim_support() {
        let contour = vec![
            Point { x: 0., y: 0. },
            Point { x: 599., y: 0. },
            Point { x: 599., y: 799. },
            Point { x: 0., y: 799. },
        ];
        let mut binary = GrayImage::from_pixel(600, 800, Luma([255]));
        for y in 100..700 {
            binary.put_pixel(590, y, Luma([0]));
        }
        for y in 730..744 {
            binary.put_pixel(590, y, Luma([0]));
        }
        for y in 300..314 {
            binary.put_pixel(7, y, Luma([0]));
        }
        whiten_page_border(&mut binary, &contour, 4);
        for y in 100..700 {
            assert_eq!(binary.get_pixel(590, y)[0], 255);
        }
        for y in 730..744 {
            assert_eq!(binary.get_pixel(590, y)[0], 255);
        }
        for y in 300..314 {
            assert_eq!(binary.get_pixel(7, y)[0], 0);
        }
    }

    #[test]
    fn compact_near_edge_numeral_inside_uncertainty_band_is_preserved() {
        let contour = vec![
            Point { x: 0., y: 0. },
            Point { x: 599., y: 0. },
            Point { x: 599., y: 799. },
            Point { x: 0., y: 799. },
        ];
        let mut binary = GrayImage::from_pixel(600, 800, Luma([255]));
        for y in 200..208 {
            for x in 5..10 {
                if x == 5 || x == 9 || y == 200 || y == 207 {
                    binary.put_pixel(x, y, Luma([0]));
                }
            }
        }
        for y in 300..314 {
            binary.put_pixel(7, y, Luma([0]));
        }
        let saved = binary.clone();
        whiten_page_border(&mut binary, &contour, 4);
        for y in 200..208 {
            for x in 5..10 {
                assert_eq!(binary.get_pixel(x, y), saved.get_pixel(x, y));
            }
        }
        for y in 300..314 {
            assert_eq!(binary.get_pixel(7, y)[0], 0);
        }
    }

    #[test]
    fn raster_crop_edge_is_not_mistaken_for_a_paper_boundary() {
        let contour = vec![
            Point { x: -100., y: -100. },
            Point { x: 699., y: -100. },
            Point { x: 699., y: 899. },
            Point { x: -100., y: 899. },
        ];
        let mut binary = GrayImage::from_pixel(600, 800, Luma([255]));
        for y in 100..300 {
            binary.put_pixel(10, y, Luma([0]));
        }
        whiten_page_border(&mut binary, &contour, 2);
        for y in 100..300 {
            assert_eq!(binary.get_pixel(10, y)[0], 0);
        }
    }

    #[test]
    fn detached_wavy_rim_is_removed_whole_while_nearby_glyph_and_inner_ink_survive() {
        let contour = vec![
            Point { x: 0., y: 0. },
            Point { x: 599., y: 0. },
            Point { x: 599., y: 799. },
            Point { x: 0., y: 799. },
        ];
        let mut binary = GrayImage::from_pixel(600, 800, Luma([255]));
        let mut rim = vec![];
        for x in 40..560 {
            let y = 780 - (4. * (x as f64 / 70.).sin()).round() as u32;
            for yy in y..y + 3 {
                binary.put_pixel(x, yy, Luma([0]));
                rim.push((x, yy));
            }
        }
        // Compact page numeral in the same broad boundary band.
        for y in 760..774 {
            for x in 12..19 {
                if x == 12 || x == 18 || y == 760 || y == 773 {
                    binary.put_pixel(x, y, Luma([0]));
                }
            }
        }
        // A long edge-adjacent stroke enters actual page content and must stay.
        for x in 8..200 {
            binary.put_pixel(x, 400, Luma([0]));
        }
        // A thin isolated page numeral must not be mistaken for a short rim.
        for y in 730..744 {
            binary.put_pixel(15, y, Luma([0]));
        }
        let saved = binary.clone();
        whiten_page_border(&mut binary, &contour, 2);
        assert!(
            rim.into_iter()
                .all(|(x, y)| binary.get_pixel(x, y)[0] == 255)
        );
        for y in 760..774 {
            for x in 12..19 {
                assert_eq!(binary.get_pixel(x, y), saved.get_pixel(x, y));
            }
        }
        for x in 8..200 {
            assert_eq!(binary.get_pixel(x, 400)[0], 0);
        }
        for y in 730..744 {
            assert_eq!(binary.get_pixel(15, y)[0], 0);
        }
    }
}

/// Inclusive ink bounds plus padding, expressed as x/y/width/height.
/// Ignore isolated components smaller than six pixels when choosing the crop;
/// do not erase them from the output. A padding band retains tiny punctuation
/// adjacent to real content. Blank images retain their original canvas.
pub fn content_crop(binary: &GrayImage, padding: u32) -> [u32; 4] {
    let (w, h) = (binary.width() as usize, binary.height() as usize);
    let mut seen = vec![false; w * h];
    let mut queue = vec![];
    let (mut left, mut top, mut right, mut bottom) = (w, h, 0, 0);
    let mut found = false;
    for seed in 0..w * h {
        if seen[seed] || binary.as_raw()[seed] != 0 {
            continue;
        }
        queue.clear();
        queue.push(seed);
        seen[seed] = true;
        let mut cursor = 0;
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
        while cursor < queue.len() {
            let i = queue[cursor];
            cursor += 1;
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
            for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    let j = yy * w + xx;
                    if !seen[j] && binary.as_raw()[j] == 0 {
                        seen[j] = true;
                        queue.push(j);
                    }
                }
            }
        }
        if queue.len() >= 6 {
            found = true;
            left = left.min(x0);
            top = top.min(y0);
            right = right.max(x1);
            bottom = bottom.max(y1);
        }
    }
    if !found {
        return [0, 0, w as u32, h as u32];
    }
    let pad = padding as usize;
    let x = left.saturating_sub(pad);
    let y = top.saturating_sub(pad);
    let end_x = (right + 1).saturating_add(pad).min(w);
    let end_y = (bottom + 1).saturating_add(pad).min(h);
    [x as u32, y as u32, (end_x - x) as u32, (end_y - y) as u32]
}

#[cfg(test)]
mod crop_tests {
    use super::*;
    use image::Luma;
    #[test]
    fn crop_keeps_separate_page_number_and_ignores_corner_specks() {
        let mut image = GrayImage::from_pixel(100, 120, Luma([255]));
        for y in 30..90 {
            for x in 20..80 {
                image.put_pixel(x, y, Luma([0]));
            }
        }
        for y in 10..16 {
            image.put_pixel(70, y, Luma([0]));
        }
        image.put_pixel(0, 119, Luma([0]));
        assert_eq!(content_crop(&image, 5), [15, 5, 70, 90]);
        assert_eq!(image.get_pixel(0, 119)[0], 0);
    }
    #[test]
    fn blank_and_border_content_keep_valid_bounds() {
        let mut image = GrayImage::from_pixel(12, 9, Luma([255]));
        assert_eq!(content_crop(&image, 2), [0, 0, 12, 9]);
        for x in 0..6 {
            image.put_pixel(x, 0, Luma([0]));
        }
        assert_eq!(content_crop(&image, 2), [0, 0, 8, 3]);
    }
}

/// Conservative page framing from the transformed paper boundary, independent
/// of the amount and position of printed content. Keep all boundary extrema.
pub fn page_crop(contour: &[crate::Point], width: u32, height: u32, padding: u32) -> [u32; 4] {
    if contour.len() < 3 || contour.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
        return [0, 0, width, height];
    }
    let pad = f64::from(padding);
    let x = (contour
        .iter()
        .map(|p| p.x)
        .fold(f64::INFINITY, f64::min)
        .floor()
        - pad)
        .max(0.)
        .min(f64::from(width)) as u32;
    let y = (contour
        .iter()
        .map(|p| p.y)
        .fold(f64::INFINITY, f64::min)
        .floor()
        - pad)
        .max(0.)
        .min(f64::from(height)) as u32;
    let ex = (contour
        .iter()
        .map(|p| p.x)
        .fold(f64::NEG_INFINITY, f64::max)
        .ceil()
        + 1.
        + pad)
        .max(0.)
        .min(f64::from(width)) as u32;
    let ey = (contour
        .iter()
        .map(|p| p.y)
        .fold(f64::NEG_INFINITY, f64::max)
        .ceil()
        + 1.
        + pad)
        .max(0.)
        .min(f64::from(height)) as u32;
    if ex <= x || ey <= y {
        return [0, 0, width, height];
    }
    [x, y, ex - x, ey - y]
}

#[cfg(test)]
mod page_crop_tests {
    use super::*;
    use crate::Point;
    #[test]
    fn preserves_blank_paper_and_curved_boundary_extrema() {
        let contour = [
            Point { x: 10., y: 8. },
            Point { x: 90., y: 8. },
            Point { x: 98., y: 50. },
            Point { x: 90., y: 110. },
            Point { x: 10., y: 110. },
        ];
        assert_eq!(page_crop(&contour, 120, 130, 2), [8, 6, 93, 107]);
        // Framing has no dependency on a tight central music block.
        let mut binary = GrayImage::from_pixel(120, 130, image::Luma([255]));
        for x in 40..70 {
            binary.put_pixel(x, 50, image::Luma([0]));
        }
        assert_ne!(page_crop(&contour, 120, 130, 2), content_crop(&binary, 2));
    }
    #[test]
    fn missing_boundary_retains_canvas_instead_of_guessing_content_crop() {
        assert_eq!(page_crop(&[], 120, 130, 2), [0, 0, 120, 130]);
    }
}

#[cfg(test)]
mod print_tests {
    use super::*;
    use image::Luma;
    #[test]
    fn varying_paper_is_white_and_thin_ink_stays_high_contrast() {
        let input = GrayImage::from_fn(160, 90, |x, y| {
            Luma([if y == 45 && (20..140).contains(&x) {
                20
            } else {
                (90 + x / 2) as u8
            }])
        });
        let normalized = normalize_paper(&input);
        assert_eq!(normalized.get_pixel(30, 10)[0], 255);
        assert_eq!(normalized.get_pixel(130, 10)[0], 255);
        let print = antialias_print(&normalized);
        assert_eq!(print.dimensions(), input.dimensions());
        assert_eq!(print.get_pixel(80, 10)[0], 255);
        assert!(print.get_pixel(80, 45)[0] < 80);
        assert!((1..255).contains(&print.get_pixel(80, 44)[0]));
    }
    #[test]
    fn constant_paper_levels_are_white_and_black_is_not_inverted() {
        for tone in [40, 80, 120, 180, 240, 255] {
            let input = GrayImage::from_pixel(8, 9, Luma([tone]));
            assert!(normalize_paper(&input).as_raw().iter().all(|p| *p == 255));
        }
        assert!(
            normalize_paper(&GrayImage::from_pixel(8, 9, Luma([0])))
                .as_raw()
                .iter()
                .all(|p| *p == 0)
        );
    }
}

#[cfg(test)]
mod print_mask_tests {
    use super::*;
    use image::Luma;
    #[test]
    fn removes_rim_halo_but_protects_retained_antialiasing() {
        let mut gray = GrayImage::from_pixel(30, 10, Luma([255]));
        let mut raw = gray.clone();
        let mut clean = gray.clone();
        for y in 1..9 {
            raw.put_pixel(5, y, Luma([0]));
            gray.put_pixel(5, y, Luma([0]));
            gray.put_pixel(6, y, Luma([150]));
            raw.put_pixel(20, y, Luma([0]));
            clean.put_pixel(20, y, Luma([0]));
            gray.put_pixel(20, y, Luma([0]));
            gray.put_pixel(19, y, Luma([130]));
        }
        apply_binary_cleanup(&mut gray, &raw, &clean);
        assert_eq!(gray.get_pixel(5, 5)[0], 255);
        assert_eq!(gray.get_pixel(6, 5)[0], 255);
        assert_eq!(gray.get_pixel(20, 5)[0], 0);
        assert_eq!(gray.get_pixel(19, 5)[0], 130);
    }
}

#[cfg(test)]
mod gray_rim_reconstruction_tests {
    use super::*;
    use crate::Point;
    use image::Luma;
    #[test]
    fn faint_fragment_chain_is_removed_without_authorizing_dark_folio_or_music() {
        let mut gray = GrayImage::from_pixel(600, 800, Luma([255]));
        let contour = [
            Point { x: 10., y: 10. },
            Point { x: 590., y: 10. },
            Point { x: 590., y: 790. },
            Point { x: 10., y: 790. },
        ];
        let fragments = [(200, 214), (225, 239), (250, 264), (275, 286)];
        for (a, b) in fragments {
            for y in a..b {
                for x in 574..579 {
                    gray.put_pixel(x, y, Luma([175]));
                }
            }
        }
        // A black isolated folio in the same strip and within continuation
        // distance must not inherit the new faint-chain evidence.
        for y in 312..329 {
            for x in 574..579 {
                gray.put_pixel(x, y, Luma([40]));
            }
        }
        // A faint isolated glyph on another edge is not a chain.
        for y in 310..324 {
            for x in 23..28 {
                gray.put_pixel(x, y, Luma([175]));
            }
        }
        // Printed inner frame and protected music remain exact.
        for y in 180..350 {
            gray.put_pixel(550, y, Luma([30]));
        }
        for y in [190, 200, 210, 220, 230] {
            for x in 80..240 {
                gray.put_pixel(x, y, Luma([35]));
            }
        }
        let protected = [[75., 170., 245., 260.]];
        let before = gray.clone();
        whiten_gray_border(&mut gray, &contour, 4, &protected);
        for (a, b) in fragments {
            for y in a..b {
                for x in 574..579 {
                    assert_eq!(gray.get_pixel(x, y)[0], 255);
                }
            }
        }
        for y in 312..329 {
            for x in 574..579 {
                assert_eq!(gray.get_pixel(x, y), before.get_pixel(x, y));
            }
        }
        for y in 310..324 {
            for x in 23..28 {
                assert_eq!(gray.get_pixel(x, y), before.get_pixel(x, y));
            }
        }
        for y in 180..350 {
            assert_eq!(gray.get_pixel(550, y), before.get_pixel(550, y));
        }
        for y in 170..=260 {
            for x in 75..=245 {
                assert_eq!(gray.get_pixel(x, y), before.get_pixel(x, y));
            }
        }
    }
    #[test]
    fn short_faint_marks_without_three_collinear_components_stay() {
        let contour = [
            Point { x: 10., y: 10. },
            Point { x: 590., y: 10. },
            Point { x: 590., y: 790. },
            Point { x: 10., y: 790. },
        ];
        for columns in [vec![574, 574], vec![574, 565, 582]] {
            let mut gray = GrayImage::from_pixel(600, 800, Luma([255]));
            for (i, x) in columns.iter().enumerate() {
                for y in 200 + i as u32 * 25..214 + i as u32 * 25 {
                    for xx in *x..*x + 5 {
                        gray.put_pixel(xx, y, Luma([175]));
                    }
                }
            }
            let before = gray.clone();
            whiten_gray_border(&mut gray, &contour, 4, &[]);
            assert_eq!(
                gray, before,
                "small near-edge glyphs do not establish a boundary chain"
            );
        }
    }
    #[test]
    fn boundary_stroke_with_interior_contact_keeps_every_interior_pixel() {
        let mut gray = GrayImage::from_pixel(300, 400, Luma([255]));
        for y in 40..350 {
            gray.put_pixel(15, y, Luma([190]));
        }
        for x in 15..180 {
            gray.put_pixel(x, 100, Luma([120]));
        }
        let before = gray.clone();
        let contour = [
            Point { x: 10., y: 10. },
            Point { x: 290., y: 10. },
            Point { x: 290., y: 390. },
            Point { x: 10., y: 390. },
        ];
        whiten_gray_border(&mut gray, &contour, 1, &[]);
        assert_eq!(gray.get_pixel(15, 200)[0], 255);
        for y in 23..378 {
            for x in 23..278 {
                assert_eq!(gray.get_pixel(x, y), before.get_pixel(x, y));
            }
        }
    }
    #[test]
    fn rotated_solid_wedge_is_distinct_from_compact_bent_ink() {
        let mut triangle = vec![];
        for y in 10..40 {
            for x in 10..50 {
                // Rotated right triangle, rasterized by three half-plane tests.
                let p = (x as f64, y as f64);
                let v = [(12., 18.), (45., 12.), (23., 36.)];
                let cross = |a: (f64, f64), b: (f64, f64)| {
                    (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)
                };
                if cross(v[0], v[1]) >= 0. && cross(v[1], v[2]) >= 0. && cross(v[2], v[0]) >= 0. {
                    triangle.push(y * 100 + x);
                }
            }
        }
        assert!(solid_triangle(&triangle, 100));
        let mut bent = vec![];
        for y in 10..35 {
            for x in 20..23 {
                bent.push(y * 100 + x);
            }
        }
        for x in 12..20 {
            for y in 10..13 {
                bent.push(y * 100 + x);
            }
        }
        assert!(!solid_triangle(&bent, 100));
    }
    #[test]
    fn corner_wedge_is_removed_but_nearby_folio_survives() {
        let mut gray = GrayImage::from_pixel(1000, 1200, Luma([255]));
        for y in 0..24 {
            for x in 0..(18 - y * 18 / 24) {
                gray.put_pixel(25 + x, 35 + y, Luma([60]));
            }
        }
        for y in 50..61 {
            gray.put_pixel(42, y, Luma([190]));
        }
        // Near the same real corner: compact "1" shape, not a solid wedge.
        for y in 35..59 {
            for x in 50..53 {
                gray.put_pixel(x, y, Luma([90]));
            }
        }
        for x in 45..53 {
            gray.put_pixel(x, 35, Luma([90]));
        }
        let before = gray.clone();
        let contour = [
            Point { x: 20., y: 30. },
            Point { x: 980., y: 30. },
            Point { x: 980., y: 1180. },
            Point { x: 20., y: 1180. },
        ];
        whiten_gray_border(&mut gray, &contour, 2, &[]);
        assert_eq!(gray.get_pixel(26, 38)[0], 255);
        assert_eq!(gray.get_pixel(42, 55)[0], 255);
        for y in 35..59 {
            for x in 45..54 {
                assert_eq!(gray.get_pixel(x, y), before.get_pixel(x, y));
            }
        }
    }
    #[test]
    fn thick_three_sided_gray_frame_is_removed_without_changing_binary_default() {
        let mut gray = GrayImage::from_pixel(300, 400, Luma([255]));
        // Connected U shape: large in both dimensions, thicker than the old
        // perimeter*erosion-margin area limit, but wholly follows the page.
        for y in 20..380 {
            for x in 15..19 {
                gray.put_pixel(x, y, Luma([70]));
            }
            for x in 281..285 {
                gray.put_pixel(x, y, Luma([70]));
            }
        }
        for y in 378..382 {
            for x in 15..285 {
                gray.put_pixel(x, y, Luma([70]));
            }
        }
        // A weak compact folio and a boundary-adjacent protected music stroke.
        for y in 22..32 {
            for x in 80..87 {
                gray.put_pixel(x, y, Luma([180]));
            }
        }
        for x in 16..150 {
            gray.put_pixel(x, 100, Luma([120]));
        }
        let contour = [
            Point { x: 10., y: 10. },
            Point { x: 290., y: 10. },
            Point { x: 290., y: 390. },
            Point { x: 10., y: 390. },
        ];
        let before = gray.clone();
        let mut binary = GrayImage::from_fn(300, 400, |x, y| {
            Luma([if gray.get_pixel(x, y)[0] < 240 {
                0
            } else {
                255
            }])
        });
        whiten_page_border(&mut binary, &contour, 1);
        assert_eq!(
            binary.get_pixel(18, 200)[0],
            0,
            "binary default stays conservative"
        );
        whiten_gray_border(&mut gray, &contour, 1, &[[0., 90., 299., 110.]]);
        assert_eq!(gray.get_pixel(18, 200)[0], 255);
        assert_eq!(gray.get_pixel(150, 380)[0], 255);
        assert_eq!(gray.get_pixel(281, 200)[0], 255);
        assert_eq!(gray.get_pixel(83, 26)[0], 180);
        for y in 90..=110 {
            for x in 0..300 {
                assert_eq!(gray.get_pixel(x, y), before.get_pixel(x, y));
            }
        }
    }
    #[test]
    fn gray_only_rim_disappears_but_weak_folio_and_music_are_unchanged() {
        let mut gray = GrayImage::from_pixel(300, 400, Luma([255]));
        for y in 40..350 {
            gray.put_pixel(15, y, Luma([190]));
        }
        // A compact, faint page number and a real boundary-adjacent music stroke.
        for y in 18..28 {
            for x in 18..24 {
                gray.put_pixel(x, y, Luma([180]));
            }
        }
        for x in 20..270 {
            gray.put_pixel(x, 100, Luma([120]));
        }
        let contour = [
            crate::Point { x: 10., y: 10. },
            crate::Point { x: 290., y: 10. },
            crate::Point { x: 290., y: 390. },
            crate::Point { x: 10., y: 390. },
        ];
        let old = gray.clone();
        let empty = GrayImage::from_pixel(300, 400, Luma([255]));
        apply_binary_cleanup_with_contour(&mut gray, &empty, &empty, &contour);
        assert_eq!(
            gray.get_pixel(15, 200)[0],
            190,
            "old black-core path leaves this rim"
        );
        whiten_gray_border(&mut gray, &contour, 1, &[[0., 90., 299., 110.]]);
        assert_eq!(gray.get_pixel(15, 200)[0], 255);
        assert_eq!(gray.get_pixel(20, 22)[0], 180);
        for y in 90..=110 {
            for x in 0..300 {
                assert_eq!(gray.get_pixel(x, y), old.get_pixel(x, y));
            }
        }
    }
    #[test]
    fn long_soft_rim_is_removed_beyond_local_seed_radius_without_losing_nearby_ink() {
        let mut gray = GrayImage::from_pixel(200, 180, Luma([255]));
        let mut raw = gray.clone();
        let mut cleaned = gray.clone();
        for y in 30..160 {
            for x in 12..17 {
                gray.put_pixel(x, y, Luma([190]));
            }
        }
        for y in 140..150 {
            raw.put_pixel(13, y, Luma([0]));
            gray.put_pixel(13, y, Luma([0]));
        }
        // A retained glyph and its gray AA fringe are close to the halo.
        for y in 60..68 {
            for x in 20..25 {
                gray.put_pixel(x, y, Luma([0]));
                raw.put_pixel(x, y, Luma([0]));
                cleaned.put_pixel(x, y, Luma([0]));
            }
        }
        for y in 60..68 {
            gray.put_pixel(19, y, Luma([160]));
        }
        // An isolated weak glyph has no strong binary core at all.
        for y in 15..23 {
            for x in 12..17 {
                gray.put_pixel(x, y, Luma([185]));
            }
        }
        let contour = [
            Point { x: 10., y: 10. },
            Point { x: 190., y: 10. },
            Point { x: 190., y: 170. },
            Point { x: 10., y: 170. },
        ];
        apply_binary_cleanup_with_contour(&mut gray, &raw, &cleaned, &contour);
        assert_eq!(gray.get_pixel(13, 40)[0], 255);
        assert_eq!(gray.get_pixel(14, 158)[0], 255);
        assert_eq!(gray.get_pixel(20, 64)[0], 0);
        assert_eq!(gray.get_pixel(19, 64)[0], 160);
        assert_eq!(gray.get_pixel(14, 18)[0], 185);
    }
    #[test]
    fn reconstruction_stops_at_true_page_band_and_does_not_use_canvas_edges() {
        let mut gray = GrayImage::from_pixel(200, 180, Luma([255]));
        let mut raw = gray.clone();
        let cleaned = gray.clone();
        // Connected gray stroke extends from the paper rim into page content.
        for x in 42..90 {
            gray.put_pixel(x, 90, Luma([190]));
        }
        raw.put_pixel(43, 90, Luma([0]));
        // A separate canvas-edge component is not a page-boundary cue.
        for y in 20..160 {
            gray.put_pixel(1, y, Luma([190]));
        }
        raw.put_pixel(1, 140, Luma([0]));
        let contour = [
            Point { x: 40., y: 10. },
            Point { x: 190., y: 10. },
            Point { x: 190., y: 170. },
            Point { x: 40., y: 170. },
        ];
        apply_binary_cleanup_with_contour(&mut gray, &raw, &cleaned, &contour);
        assert_eq!(gray.get_pixel(44, 90)[0], 255);
        assert_eq!(gray.get_pixel(60, 90)[0], 190);
        assert_eq!(gray.get_pixel(1, 40)[0], 190);
    }
}
