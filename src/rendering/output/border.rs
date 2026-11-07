//! Remove page rims and shadows while preserving nearby notation.
use super::PageBorder;
use super::distance::*;
use image::GrayImage;
use rayon::prelude::*;
use std::borrow::Cow;

/// Transfer confirmed removed ink to gray output without thresholding valid
/// antialiased glyph edges. Halo pixels are removed only when closer to a
/// removed core than to retained ink; ties protect the retained foreground.
pub fn apply_binary_cleanup(gray: &mut GrayImage, raw: &GrayImage, cleaned: &GrayImage) -> usize {
    apply_binary_cleanup_impl(gray, raw, cleaned, None, None)
}

/// Reconstruct complete soft-gray rim halos from confirmed removed cores.
/// Propagation follows gray connectivity only near the true page perimeter;
/// retained ink and its AA fringe are barriers, and isolated weak glyphs remain.
pub fn apply_binary_cleanup_with_contour(
    gray: &mut GrayImage,
    raw: &GrayImage,
    cleaned: &GrayImage,
    contour: &[crate::geometry::Point],
) -> usize {
    apply_binary_cleanup_impl(gray, raw, cleaned, Some(contour), None)
}

pub(super) fn apply_binary_cleanup_impl(
    gray: &mut GrayImage,
    raw: &GrayImage,
    cleaned: &GrayImage,
    contour: Option<&[crate::geometry::Point]>,
    prepared: Option<&PageBorder<'_>>,
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
    let (removed, kept) =
        gpu_pair_distance(prepared, &removed, &kept, w, h, cap).unwrap_or_else(|| {
            rayon::join(
                || capped_distance(&removed, w, h, cap, usize::from(cap)),
                || capped_distance(&kept, w, h, cap, usize::from(cap)),
            )
        });
    let mut count = 0;
    if let Some(contour) =
        contour.filter(|p| p.len() >= 3 && p.iter().all(|p| p.x.is_finite() && p.y.is_finite()))
    {
        let band = ((w.min(h) as f64 * 0.04).ceil() as u16).max(2);
        let boundary = if let Some(prepared) = prepared {
            Cow::Borrowed(
                prepared
                    .boundary
                    .get_or_init(|| distance_to_contour(contour, w, h, band + 1, Some(prepared)))
                    .as_slice(),
            )
        } else {
            Cow::Owned(distance_to_contour(contour, w, h, band + 1, prepared))
        };
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
    contour: &[crate::geometry::Point],
    margin: u32,
    protected: &[[f64; 4]],
) -> usize {
    whiten_gray_border_impl(gray, contour, margin, protected, None)
}

pub(super) fn whiten_gray_border_impl(
    gray: &mut GrayImage,
    contour: &[crate::geometry::Point],
    margin: u32,
    protected: &[[f64; 4]],
    prepared: Option<&PageBorder<'_>>,
) -> usize {
    let before = gray.clone();
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let protected_mask = protected_pixel_mask(w, h, protected);
    let mut ink = vec![255; w * h];
    ink.par_iter_mut()
        .zip(gray.as_raw())
        .zip(&protected_mask)
        .for_each(|((out, tone), protected)| {
            if *protected == 0 && *tone < 240 {
                *out = 0;
            }
        });
    let raw = GrayImage::from_raw(gray.width(), gray.height(), ink).unwrap();
    let mut cleaned = raw.clone();
    whiten_page_border_impl(
        &mut cleaned,
        contour,
        margin,
        true,
        Some(gray.as_raw()),
        prepared,
    );
    apply_binary_cleanup_impl(gray, &raw, &cleaned, Some(contour), prepared);
    gray.as_mut()
        .par_iter_mut()
        .zip(before.as_raw())
        .zip(&protected_mask)
        .map(|((pixel, before), protected)| {
            if *protected != 0 {
                *pixel = *before;
            }
            usize::from(*pixel != *before)
        })
        .sum()
}

/// Rasterize the union once instead of testing every pixel against every music
/// rectangle twice. Inclusive float bounds match the original pixel predicate.
pub(super) fn protected_pixel_mask(w: usize, h: usize, protected: &[[f64; 4]]) -> Vec<u8> {
    let mut mask = vec![0; w * h];
    for r in protected {
        if r.iter().any(|v| v.is_nan()) || r[0] > r[2] || r[1] > r[3] {
            continue;
        }
        let x0 = r[0].ceil().clamp(0., w as f64) as usize;
        let y0 = r[1].ceil().clamp(0., h as f64) as usize;
        let x1 = (r[2].floor() + 1.).clamp(0., w as f64) as usize;
        let y1 = (r[3].floor() + 1.).clamp(0., h as f64) as usize;
        if x0 >= x1 || y0 >= y1 {
            continue;
        }
        for y in y0..y1 {
            mask[y * w + x0..y * w + x1].fill(1);
        }
    }
    mask
}

/// Whiten outside a mapped page polygon and a narrow inward border band.
/// Scanline filling plus two separable minimum filters avoids inverse-map calls
/// per output pixel. Coordinates are final output raster pixels.
pub fn whiten_page_border(
    binary: &mut GrayImage,
    contour: &[crate::geometry::Point],
    margin: u32,
) -> usize {
    whiten_page_border_impl(binary, contour, margin, false, None, None)
}

pub(super) fn whiten_page_border_impl(
    binary: &mut GrayImage,
    contour: &[crate::geometry::Point],
    margin: u32,
    gray_frames: bool,
    gray_tone: Option<&[u8]>,
    prepared: Option<&PageBorder<'_>>,
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
    let mut mask = if let Some(prepared) = prepared {
        prepared
            .mask
            .get_or_init(|| page_mask(contour, w, h))
            .clone()
    } else {
        page_mask(contour, w, h)
    };
    // Classify the original connected ink before erosion can split a frame
    // into detached, character-sized fragments.
    let mut removed = if margin > 0 {
        remove_rim_components(
            binary,
            &mask,
            contour,
            margin,
            gray_frames,
            gray_tone,
            prepared,
        )
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

fn page_mask(contour: &[crate::geometry::Point], w: usize, h: usize) -> Vec<u8> {
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
    mask
}

/// Entire boundary-following strokes can lie well inside an estimated outline.
/// Distance to the actual (possibly curved) mask, not a rectangular crop, limits
/// component selection. Compact glyphs and components entering the interior stay.
fn remove_rim_components(
    binary: &mut GrayImage,
    mask: &[u8],
    contour: &[crate::geometry::Point],
    margin: u32,
    gray_frames: bool,
    gray_tone: Option<&[u8]>,
    prepared: Option<&PageBorder<'_>>,
) -> usize {
    let (w, h) = (binary.width() as usize, binary.height() as usize);
    let band = ((w.min(h) as f64 * 0.04).ceil() as u16).max(2);
    let cap = band + 1;
    let close_band = (margin.saturating_mul(3).min(u32::from(band))) as u16;
    let distance = if let Some(prepared) = prepared {
        Cow::Borrowed(
            prepared
                .rim
                .get_or_init(|| rim_distance(mask, contour, w, h, cap, Some(prepared)))
                .as_slice(),
        )
    } else {
        Cow::Owned(rim_distance(mask, contour, w, h, cap, prepared))
    };
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
    contour: &[crate::geometry::Point],
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
    contour: &[crate::geometry::Point],
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
pub(super) fn solid_triangle(pixels: &[usize], width: usize) -> bool {
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
