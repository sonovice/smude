//! Conservative classical text-curve observations. No OCR or learned model.
//! Music masks are hard exclusions; this module never changes image pixels.
use std::collections::VecDeque;

use image::{GrayImage, imageops};
use serde::Serialize;

use crate::{Point, binarize_sauvola, median};

#[derive(Clone, Debug, Serialize)]
pub struct TextLine {
    /// Smoothed body centre, not a typographically exact baseline.
    pub points: Vec<Point>,
    /// Valid support only, in input image pixels: [left, top, right, bottom].
    pub bounds: [f64; 4],
    pub block_id: usize,
    /// Heuristic support score, not a calibrated probability.
    pub confidence: f64,
    pub uncertainty_pixels: f64,
}

struct Component {
    bounds: [usize; 4],
    pixels: Vec<usize>,
}

fn components(mask: &[bool], width: usize, height: usize) -> Vec<Component> {
    let mut visited = vec![false; mask.len()];
    let mut out = vec![];
    for start in 0..mask.len() {
        if !mask[start] || visited[start] {
            continue;
        }
        visited[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut pixels = vec![];
        let mut bounds = [start % width, start / width, start % width, start / width];
        while let Some(i) = queue.pop_front() {
            let (x, y) = (i % width, i / width);
            pixels.push(i);
            bounds[0] = bounds[0].min(x);
            bounds[1] = bounds[1].min(y);
            bounds[2] = bounds[2].max(x);
            bounds[3] = bounds[3].max(y);
            for yy in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let j = yy * width + xx;
                    if mask[j] && !visited[j] {
                        visited[j] = true;
                        queue.push_back(j);
                    }
                }
            }
        }
        bounds[2] += 1;
        bounds[3] += 1;
        out.push(Component { bounds, pixels });
    }
    out
}

/// Detect prose curves outside caller-supplied complete music regions. Bounds
/// are XYXY in input pixels; callers should include lyrics, dynamics and slurs.
/// Short isolated words and uncertain components intentionally yield no curve.
pub fn detect_text_lines(gray: &GrayImage, protected_music: &[[f64; 4]]) -> Vec<TextLine> {
    if gray.width() < 32 || gray.height() < 32 {
        return vec![];
    }
    let scale = (1200. / f64::from(gray.width().max(gray.height()))).min(1.);
    let (w, h) = (
        (f64::from(gray.width()) * scale).round() as u32,
        (f64::from(gray.height()) * scale).round() as u32,
    );
    let image = if scale < 1. {
        imageops::resize(gray, w, h, imageops::FilterType::Triangle)
    } else {
        gray.clone()
    };
    let binary = binarize_sauvola(&image);
    let (w, h) = (w as usize, h as usize);
    let mut allowed = vec![true; w * h];
    for bounds in protected_music {
        if bounds.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let a = ((bounds[0] * scale).floor() as i64 - 3).clamp(0, w as i64) as usize;
        let b = ((bounds[1] * scale).floor() as i64 - 3).clamp(0, h as i64) as usize;
        let c = ((bounds[2] * scale).ceil() as i64 + 3).clamp(0, w as i64) as usize;
        let d = ((bounds[3] * scale).ceil() as i64 + 3).clamp(0, h as i64) as usize;
        if a >= c || b >= d {
            continue;
        }
        for y in b..d {
            allowed[y * w + a..y * w + c].fill(false);
        }
    }
    let ink: Vec<bool> = binary
        .as_raw()
        .iter()
        .zip(&allowed)
        .map(|(v, a)| *v < 128 && *a)
        .collect();
    let glyphs: Vec<_> = components(&ink, w, h)
        .into_iter()
        .filter(|c| {
            let [a, b, z, d] = c.bounds;
            let (cw, ch) = (z - a, d - b);
            (4..=60).contains(&ch) && cw <= 4 * ch && c.pixels.len() >= 4
        })
        .collect();
    if glyphs.len() < 3 {
        return vec![];
    }
    let character_height = median(
        glyphs
            .iter()
            .map(|c| (c.bounds[3] - c.bounds[1]) as f64)
            .collect(),
    )
    .max(4.);
    // Link glyph components rather than their dilated pixel envelopes. A
    // descender touching the horizontal smear of the following line must not
    // fuse the two prose lines into one tall component.
    let mut order: Vec<usize> = (0..glyphs.len()).collect();
    order.sort_by_key(|i| glyphs[*i].bounds[0]);
    let mut edges = vec![];
    for (position, i) in order.iter().copied().enumerate() {
        let a = glyphs[i].bounds;
        let height_a = (a[3] - a[1]) as f64;
        let center_a = (a[1] + a[3]) as f64 / 2.;
        for j in order.iter().copied().skip(position + 1) {
            let b = glyphs[j].bounds;
            let gap = b[0] as f64 - a[2] as f64;
            if gap > 5.1 * height_a.max(character_height) {
                break;
            }
            if gap < -0.15 * character_height {
                continue;
            }
            let height_b = (b[3] - b[1]) as f64;
            if height_a.max(height_b) > 2.4 * height_a.min(height_b) {
                continue;
            }
            let pair_height = (height_a + height_b) / 2.;
            if gap > 3. * pair_height {
                continue;
            }
            let dy = (b[1] + b[3]) as f64 / 2. - center_a;
            let dx = (b[0] + b[2]) as f64 / 2. - (a[0] + a[2]) as f64 / 2.;
            if dy.abs() > 0.65 * pair_height || dy.abs() > 0.35 * dx + 0.3 * pair_height {
                continue;
            }
            let y = (center_a + (b[1] + b[3]) as f64 / 2.) / 2.;
            let start = a[2].min(w - 1);
            let end = b[0].min(w - 1);
            if start <= end
                && (start..=end).any(|x| !allowed[(y.round() as usize).min(h - 1) * w + x])
            {
                continue;
            }
            let cost = gap.max(0.) / pair_height
                + 3. * (dy / pair_height).powi(2)
                + 0.15 * (height_a / height_b).ln().abs();
            edges.push((cost, i, j));
        }
    }
    edges.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut next = vec![None; glyphs.len()];
    let mut previous = vec![None; glyphs.len()];
    for (_, i, j) in edges {
        if next[i].is_none() && previous[j].is_none() {
            next[i] = Some(j);
            previous[j] = Some(i);
        }
    }
    let mut chains = vec![];
    for (start, pred) in previous.iter().enumerate() {
        if pred.is_some() {
            continue;
        }
        let mut ids = vec![start];
        let mut current = start;
        while let Some(j) = next[current] {
            ids.push(j);
            current = j;
        }
        chains.push(ids);
    }
    // Recover short breaks left by a locally ambiguous glyph attachment. Use
    // several endpoint glyphs to estimate text flow instead of matching a
    // single ascender/descender centre.
    let endpoint = |ids: &[usize], end: bool| {
        let samples: Vec<_> = if end {
            ids.iter().rev().take(4).copied().collect()
        } else {
            ids.iter().take(4).copied().collect()
        };
        let coords: Vec<_> = samples
            .iter()
            .map(|i| {
                let b = glyphs[*i].bounds;
                (
                    (b[0] + b[2]) as f64 / 2.,
                    (b[1] + b[3]) as f64 / 2.,
                    (b[3] - b[1]) as f64,
                )
            })
            .collect();
        let mx = coords.iter().map(|p| p.0).sum::<f64>() / coords.len() as f64;
        let my = coords.iter().map(|p| p.1).sum::<f64>() / coords.len() as f64;
        let den = coords.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>();
        let slope = if den > 1. {
            coords.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>() / den
        } else {
            0.
        };
        let x = if end {
            ids.iter().map(|i| glyphs[*i].bounds[2]).max().unwrap()
        } else {
            ids.iter().map(|i| glyphs[*i].bounds[0]).min().unwrap()
        } as f64;
        (
            x,
            my + slope.clamp(-0.35, 0.35) * (x - mx),
            median(coords.iter().map(|p| p.2).collect()),
        )
    };
    let mut edges = vec![];
    for (i, a) in chains.iter().enumerate() {
        for (j, b) in chains.iter().enumerate() {
            if i == j {
                continue;
            }
            // Short chains may have slightly overlapping italic envelopes.
            // A strict left-to-right order still makes the chain graph acyclic.
            let left_a = a.iter().map(|id| glyphs[*id].bounds[0]).min().unwrap();
            let left_b = b.iter().map(|id| glyphs[*id].bounds[0]).min().unwrap();
            if (left_b, j) <= (left_a, i) {
                continue;
            }
            let (ax, ay, ah) = endpoint(a, true);
            let (bx, by, bh) = endpoint(b, false);
            let gap = bx - ax;
            let height = (ah + bh) / 2.;
            if gap < -0.35 * height
                || gap > 3. * height
                || ah.max(bh) > 2.4 * ah.min(bh)
                || (by - ay).abs() > 0.55 * height + 0.15 * gap.max(0.)
            {
                continue;
            }
            let n = gap.ceil().max(1.) as usize;
            if (0..=n).any(|k| {
                let t = k as f64 / n as f64;
                let x = (ax + t * gap).round() as usize;
                let y = (ay + t * (by - ay)).round() as usize;
                x >= w || y >= h || !allowed[y * w + x]
            }) {
                continue;
            }
            edges.push((
                gap.max(0.) / height + 3. * ((by - ay) / height).powi(2),
                i,
                j,
            ));
        }
    }
    edges.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut next = vec![None; chains.len()];
    let mut previous = vec![None; chains.len()];
    for (_, i, j) in edges {
        if next[i].is_none() && previous[j].is_none() {
            next[i] = Some(j);
            previous[j] = Some(i);
        }
    }
    let mut lines = vec![];
    for (start, pred) in previous.iter().enumerate() {
        if pred.is_some() {
            continue;
        }
        let mut ids = chains[start].clone();
        let mut current = start;
        while let Some(j) = next[current] {
            ids.extend_from_slice(&chains[j]);
            current = j;
        }
        let estimated_characters = ids
            .iter()
            .map(|i| {
                let b = glyphs[*i].bounds;
                ((b[2] - b[0]) as f64 / (0.7 * (b[3] - b[1]) as f64)).max(1.)
            })
            .sum::<f64>();
        if ids.len() < 3 || estimated_characters < 6. {
            continue;
        }
        let members: Vec<_> = ids.iter().map(|i| &glyphs[*i]).collect();
        let a = members.iter().map(|g| g.bounds[0]).min().unwrap();
        let b = members.iter().map(|g| g.bounds[1]).min().unwrap();
        let z = members.iter().map(|g| g.bounds[2]).max().unwrap();
        let d = members.iter().map(|g| g.bounds[3]).max().unwrap();
        let local_height = median(
            members
                .iter()
                .map(|g| (g.bounds[3] - g.bounds[1]) as f64)
                .collect(),
        )
        .max(4.);
        if ((z - a) as f64) < 5. * local_height || ((d - b) as f64) > 5. * local_height {
            continue;
        }
        let mut span_ink = vec![false; (z - a) * (d - b)];
        for g in &members {
            for i in &g.pixels {
                let (x, y) = (i % w, i / w);
                span_ink[(y - b) * (z - a) + x - a] = true;
            }
        }
        let mut samples = vec![];
        let step = (local_height * 0.75).round().max(2.) as usize;
        for x in (a..z).step_by(step) {
            let lo = x.saturating_sub(step).max(a);
            let hi = (x + step + 1).min(z);
            let mut rows = vec![];
            for y in b..d {
                for xx in lo..hi {
                    if span_ink[(y - b) * (z - a) + xx - a] {
                        rows.push(y as f64);
                    }
                }
            }
            if rows.len() < 4 {
                continue;
            }
            rows.sort_by(f64::total_cmp);
            let center = (rows[rows.len() / 5] + rows[rows.len() * 4 / 5]) / 2.;
            samples.push(Point {
                x: x as f64,
                y: center,
            });
        }
        if samples.len() < 6 {
            continue;
        }
        // Local affine smoothing suppresses character-shape jitter without
        // imposing a straight line on the entire curved text span.
        let radius = 6. * local_height;
        let points: Vec<Point> = samples
            .iter()
            .map(|p| {
                let nearby: Vec<_> = samples
                    .iter()
                    .filter(|q| (q.x - p.x).abs() <= radius)
                    .collect();
                let mean_x = nearby.iter().map(|q| q.x).sum::<f64>() / nearby.len() as f64;
                let mean_y = nearby.iter().map(|q| q.y).sum::<f64>() / nearby.len() as f64;
                let denom = nearby.iter().map(|q| (q.x - mean_x).powi(2)).sum::<f64>();
                let slope = if denom > 0. {
                    nearby
                        .iter()
                        .map(|q| (q.x - mean_x) * (q.y - mean_y))
                        .sum::<f64>()
                        / denom
                } else {
                    0.
                };
                Point {
                    x: p.x / scale,
                    y: (mean_y + slope * (p.x - mean_x)) / scale,
                }
            })
            .collect();
        let rms = (samples
            .iter()
            .zip(&points)
            .map(|(p, q)| (p.y - q.y * scale).powi(2))
            .sum::<f64>()
            / samples.len() as f64)
            .sqrt();
        if rms > 0.35 * local_height {
            continue;
        }
        // A curved text span's rectangular bounding box may overlap a music
        // box at its opposite corner even though every glyph and the curve
        // itself remain outside. Test the actual supported path, not that box.
        if points.windows(2).any(|pair| {
            let (a, b) = (pair[0], pair[1]);
            let n = ((b.x - a.x) * scale).ceil().max(1.) as usize;
            (0..=n).any(|i| {
                let t = i as f64 / n as f64;
                let x = ((a.x + t * (b.x - a.x)) * scale).round() as usize;
                let y = ((a.y + t * (b.y - a.y)) * scale).round() as i64;
                (-1..=1).any(|dy| {
                    x >= w
                        || y + dy < 0
                        || y + dy >= h as i64
                        || !allowed[(y + dy) as usize * w + x]
                })
            })
        }) {
            continue;
        }
        lines.push(TextLine {
            points,
            bounds: [
                a as f64 / scale,
                b as f64 / scale,
                z as f64 / scale,
                d as f64 / scale,
            ],
            block_id: 0,
            confidence: (estimated_characters / 20.).min(1.) * (1. - rms / local_height).max(0.),
            uncertainty_pixels: (rms + 0.5) / scale,
        });
    }
    lines.sort_by(|a, b| {
        a.bounds[1]
            .total_cmp(&b.bounds[1])
            .then(a.bounds[0].total_cmp(&b.bounds[0]))
    });
    for i in 0..lines.len() {
        let a = lines[i].bounds;
        lines[i].block_id = i;
        for j in (0..i).rev() {
            let b = lines[j].bounds;
            let overlap = (a[2].min(b[2]) - a[0].max(b[0])).max(0.);
            if overlap > 0.5 * (a[2] - a[0]).min(b[2] - b[0])
                && a[1] - b[3] < 3. * character_height / scale
            {
                lines[i].block_id = lines[j].block_id;
                break;
            }
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;
    fn paragraph(columns: &[u32]) -> GrayImage {
        let mut im = GrayImage::from_pixel(650, 220, Luma([255]));
        for left in columns {
            for line in 0..3 {
                for letter in 0..20 {
                    let x = left + letter * 10;
                    let bend = (8. * ((x - *left) as f64 / 190. * std::f64::consts::PI).sin())
                        .round() as u32;
                    for yy in 0..8 {
                        for xx in 0..5 {
                            if xx == 0 || xx == 4 || yy == 0 || yy == 4 {
                                im.put_pixel(x + xx, 30 + line * 40 + bend + yy, Luma([0]));
                            }
                        }
                    }
                }
            }
        }
        im
    }
    #[test]
    fn curved_paragraph_has_smooth_local_text_curves() {
        let lines = detect_text_lines(&paragraph(&[20]), &[]);
        assert_eq!(lines.len(), 3);
        let p = &lines[0].points;
        assert!(p[p.len() / 2].y - p[0].y > 3.);
        assert!(p.windows(2).all(|w| w[0].x < w[1].x));
    }
    #[test]
    fn disconnected_columns_stay_separate() {
        let lines = detect_text_lines(&paragraph(&[20, 370]), &[]);
        assert_eq!(lines.len(), 6);
        assert!(lines.iter().all(|l| l.bounds[2] - l.bounds[0] < 230.));
        assert_ne!(lines[0].block_id, lines[1].block_id);
    }
    #[test]
    fn full_music_bands_are_hard_exclusions() {
        let mut im = paragraph(&[20]);
        for y in [30, 40, 50, 60, 70] {
            for x in 340..630 {
                im.put_pixel(x, y, Luma([0]));
            }
        }
        for x in (350..620).step_by(15) {
            for y in 20..84 {
                im.put_pixel(x, y, Luma([0]));
            }
        }
        let lines = detect_text_lines(&im, &[[330., 10., 640., 95.]]);
        assert_eq!(lines.len(), 3);
        assert!(lines.iter().all(|l| l.bounds[2] < 330.));
        assert!(detect_text_lines(&im, &[[0., 0., 650., 220.]]).is_empty());
    }

    #[test]
    fn curved_text_path_survives_mask_at_opposite_bounding_box_corner() {
        let mut im = GrayImage::from_pixel(300, 140, Luma([255]));
        for letter in 0..22_u32 {
            let x = 20 + letter * 10;
            let y = (70. - 1.2 * letter as f64).round() as u32;
            for yy in 0..8 {
                for xx in 0..5 {
                    if xx == 0 || xx == 4 || yy == 0 || yy == 4 {
                        im.put_pixel(x + xx, y + yy, Luma([0]));
                    }
                }
            }
        }
        let lines = detect_text_lines(&im, &[[140., 68., 290., 120.]]);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].bounds[1] < 68. && lines[0].bounds[3] > 68.);
        assert!(lines[0].points.iter().all(|p| p.x < 137. || p.y < 64.));
    }

    #[test]
    fn close_prose_lines_with_ascenders_descenders_and_word_gaps_stay_separate() {
        let mut im = GrayImage::from_pixel(500, 160, Luma([255]));
        for line in 0..3_u32 {
            for letter in 0..30_u32 {
                let x = 20 + letter * 10 + (letter / 7) * 5;
                let bend = (3. * (letter as f64 / 29. * std::f64::consts::PI).sin()).round() as u32;
                let y = 30 + line * 16 + bend;
                let asc = if letter % 5 == 1 { 3 } else { 0 };
                let desc = if letter % 7 == 3 { 3 } else { 0 };
                for yy in 0..8 + asc + desc {
                    for xx in 0..5 {
                        if xx == 0 || xx == 4 || yy == asc || yy == asc + 4 {
                            im.put_pixel(x + xx, y + yy - asc, Luma([0]));
                        }
                    }
                }
            }
        }
        let lines = detect_text_lines(&im, &[]);
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert!(lines.iter().all(|l| l.bounds[2] - l.bounds[0] > 300.));
    }

    #[test]
    fn different_font_scales_do_not_use_one_global_link_distance() {
        let mut im = GrayImage::from_pixel(600, 230, Luma([255]));
        for (line, height, step) in [(30_u32, 6_u32, 9_u32), (120, 14, 20)] {
            for letter in 0..22_u32 {
                let x = 20 + letter * step + (letter / 7) * height;
                for yy in 0..height {
                    for xx in 0..height / 2 {
                        if xx == 0 || xx == height / 2 - 1 || yy == 0 || yy == height / 2 {
                            im.put_pixel(x + xx, line + yy, Luma([0]));
                        }
                    }
                }
            }
        }
        let lines = detect_text_lines(&im, &[]);
        assert_eq!(lines.len(), 2, "{lines:?}");
    }

    #[test]
    fn touching_letters_can_form_a_short_label_with_only_three_components() {
        let mut im = GrayImage::from_pixel(140, 90, Luma([255]));
        for (x, y, w, h) in [(20, 30, 14, 10), (36, 33, 6, 7), (47, 30, 32, 10)] {
            for yy in y..y + h {
                for xx in x..x + w {
                    im.put_pixel(xx, yy, Luma([0]));
                }
            }
        }
        let lines = detect_text_lines(&im, &[]);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].confidence >= 0.25);
    }
}
