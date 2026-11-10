//! Remove page rims and shadows while preserving nearby notation.
use super::PageBorder;
use super::distance::*;
use image::GrayImage;
use rayon::prelude::*;
use std::borrow::Cow;

/// Whiten only confidently exterior pixels. Preserve the entire page and an
/// outward uncertainty band, regardless of tone, ink components or music boxes.
pub(super) fn whiten_exterior_impl(
    gray: &mut GrayImage,
    contour: &[crate::geometry::Point],
    margin: u32,
    prepared: &PageBorder<'_>,
) {
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    if w == 0
        || h == 0
        || contour.len() < 3
        || contour.iter().any(|p| !p.x.is_finite() || !p.y.is_finite())
    {
        return;
    }
    let mask = prepared.mask.get_or_init(|| page_mask(contour, w, h));
    let margin = margin.min(u32::from(u16::MAX - 1)) as u16;
    // The band field shared with the gray cleanup answers the same question
    // whenever it reaches at least as far as the margin.
    let band = rim_band(w, h);
    let boundary = if margin <= band {
        Cow::Borrowed(
            prepared
                .boundary
                .get_or_init(|| distance_to_contour(contour, w, h, band + 1))
                .as_slice(),
        )
    } else {
        Cow::Owned(distance_to_contour(contour, w, h, margin + 1))
    };
    // Only tiles reaching beyond the page hold such pixels.
    for run in Tiles::below(mask, w, h, 1).spans().runs() {
        let fields = mask[run.clone()].iter().zip(&boundary[run.clone()]);
        for (pixel, (inside, distance)) in gray.as_mut()[run].iter_mut().zip(fields) {
            if *inside == 0 && *distance > margin {
                *pixel = 255;
            }
        }
    }
}

/// Width of the strip along the page perimeter in which rims are sought.
fn rim_band(w: usize, h: usize) -> u16 {
    ((w.min(h) as f64 * 0.04).ceil() as u16).max(2)
}

/// Transfer confirmed removed ink to gray output without thresholding valid
/// antialiased glyph edges. Halo pixels are removed only when closer to a
/// removed core than to retained ink; ties protect the retained foreground.
pub fn apply_binary_cleanup(gray: &mut GrayImage, raw: &GrayImage, cleaned: &GrayImage) -> usize {
    apply_binary_cleanup_impl(gray, raw, cleaned, None, None, None)
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
    apply_binary_cleanup_impl(gray, raw, cleaned, Some(contour), None, None)
}

pub(super) fn apply_binary_cleanup_impl(
    gray: &mut GrayImage,
    raw: &GrayImage,
    cleaned: &GrayImage,
    contour: Option<&[crate::geometry::Point]>,
    protected: Option<&[u8]>,
    prepared: Option<&PageBorder<'_>>,
) -> usize {
    assert_eq!(gray.dimensions(), raw.dimensions());
    assert_eq!(raw.dimensions(), cleaned.dimensions());
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    // The soft transition can reveal a wider gray paper-edge halo than the
    // binary core. Extend only from confirmed removed ink; retained foreground
    // wins the distance competition, including its antialiased fringe.
    let cap: u8 = ((w.min(h) / 80).clamp(4, 64) + 1) as u8;
    let reach = usize::from(cap) - 1;
    // One vectorized pass fills fresh memory faster than threads sharing it;
    // `&&` would leave a branch in it that ink-heavy pages mispredict.
    let seeds = |removed: bool| -> Vec<u8> {
        (raw.as_raw().iter())
            .zip(cleaned.as_raw())
            .map(|(raw, cleaned)| {
                if (*raw == 0) & ((*cleaned == 255) == removed) {
                    0
                } else {
                    cap
                }
            })
            .collect()
    };
    let mut removed = seeds(true);
    if !removed.contains(&0) {
        return 0;
    }
    let seeded = Tiles::below(&removed, w, h, cap);
    let reached = seeded.grown(reach);
    let contour =
        contour.filter(|p| p.len() >= 3 && p.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
    let band = rim_band(w, h);
    // Retained ink is consulted only where removed ink reaches and, for the
    // reconstruction, along the perimeter; its distance is exact there once
    // every tile within reach spreads its seeds.
    let mut consulted = contour.map_or_else(
        || Tiles::empty(w, h),
        |contour| contour_tiles(contour, w, h).grown(band.into()),
    );
    consulted.include(&reached);
    let retained = consulted.grown(reach);
    let mut kept = seeds(false);
    // Protected pixels take part in every decision but keep their tone.
    let frozen = |i: usize| protected.is_some_and(|mask| mask[i] != 0);
    rayon::join(
        || capped_distance(&mut removed, cap, &seeded),
        || capped_distance(&mut kept, cap, &retained),
    );
    let mut count = 0;
    if let Some(contour) = contour {
        let boundary = if let Some(prepared) = prepared {
            Cow::Borrowed(
                prepared
                    .boundary
                    .get_or_init(|| distance_to_contour(contour, w, h, band + 1))
                    .as_slice(),
            )
        } else {
            Cow::Owned(distance_to_contour(contour, w, h, band + 1))
        };
        let mut seen = vec![false; w * h];
        // This is seeded reconstruction, not a wider global eroded mask. A
        // distant gray halo is removed only if connected to a proved rim core.
        let mut queue = vec![];
        for run in seeded.spans().runs() {
            let first = run.start;
            let cores = removed[run]
                .iter()
                .enumerate()
                .filter(|(_, removed)| **removed == 0);
            for i in cores.map(|(offset, _)| first + offset) {
                if gray.as_raw()[i] < 250 && boundary[i] <= band && kept[i] > 3 {
                    seen[i] = true;
                    queue.push(i);
                }
            }
        }
        let mut cursor = 0;
        while cursor < queue.len() {
            let i = queue[cursor];
            cursor += 1;
            if gray.as_raw()[i] != 255 && !frozen(i) {
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
    for run in reached.spans().runs() {
        let first = run.start;
        let distances = removed[run.clone()].iter().zip(&kept[run.clone()]);
        let pixels = gray.as_mut()[run].iter_mut().zip(distances);
        for (offset, (pixel, (removed, kept))) in pixels.enumerate() {
            if *removed < cap && removed < kept && *pixel != 255 && !frozen(first + offset) {
                *pixel = 255;
                count += 1;
            }
        }
    }
    count
}

/// Remove faint perimeter strokes which never produced a Sauvola black core.
/// Shape classification uses normalized gray, and protected music rectangles
/// are never altered. No global inset or content crop is introduced.
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
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    if w == 0
        || h == 0
        || margin == 0
        || contour.len() < 3
        || contour.iter().any(|p| !p.x.is_finite() || !p.y.is_finite())
    {
        return 0;
    }
    let protected_mask = protected_pixel_mask(w, h, protected);
    let ink = (gray.as_raw().iter())
        .zip(&protected_mask)
        .map(|(tone, protected)| {
            if *protected == 0 && *tone < 255 {
                0
            } else {
                255
            }
        })
        .collect();
    let raw = GrayImage::from_raw(gray.width(), gray.height(), ink).unwrap();
    let mut cleaned = raw.clone();
    let mask = if let Some(prepared) = prepared {
        Cow::Borrowed(
            prepared
                .mask
                .get_or_init(|| page_mask(contour, w, h))
                .as_slice(),
        )
    } else {
        Cow::Owned(page_mask(contour, w, h))
    };
    // Use only confirmed rim shapes, not the binary path's eroded page mask:
    // an uncertain contour must not shave faint print off the gray master.
    remove_rim_components(
        &mut cleaned,
        &mask,
        contour,
        margin,
        true,
        Some(gray.as_raw()),
        prepared,
    );
    apply_binary_cleanup_impl(
        gray,
        &raw,
        &cleaned,
        Some(contour),
        Some(&protected_mask),
        prepared,
    )
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
    if margin > 0 {
        erode_mask(&mut mask, w, h, margin as usize);
    }
    // Only tiles reaching beyond the eroded page hold such pixels.
    for run in Tiles::below(&mask, w, h, 1).spans().runs() {
        let (pixels, mask) = (&mut binary.as_mut()[run.clone()], &mask[run]);
        // Two plain loops are vectorized where one branching loop is not.
        removed += (pixels.iter().zip(mask))
            .filter(|(value, inside)| **inside == 0 && **value == 0)
            .count();
        for (value, inside) in pixels.iter_mut().zip(mask) {
            *value = if *inside == 0 { 255 } else { *value };
        }
    }
    removed
}

/// Square erosion in output pixels of a mask holding 1 inside the page and 0
/// outside it; outside the raster is outside the page. A pixel survives when
/// its distance to the nearest zero exceeds `radius`, and that field is swept
/// only near the zeros.
pub(super) fn erode_mask(mask: &mut [u8], w: usize, h: usize, radius: usize) {
    // Erosions add up, which keeps any radius within the range of the field.
    let mut rest = radius;
    while rest > 0 {
        let step = rest.min(usize::from(u8::MAX) - 2) as u8;
        rest -= usize::from(step);
        let cap = step + 1;
        let zeros = Tiles::below(mask, w, h, 1);
        let swept = zeros.swept(cap.into()).spans();
        for run in swept.runs() {
            for value in &mut mask[run] {
                *value = if *value == 0 { 0 } else { cap };
            }
        }
        capped_distance(mask, cap, &zeros);
        for run in swept.runs() {
            for value in &mut mask[run] {
                *value = u8::from(*value == cap);
            }
        }
    }
    let (rows, columns) = (radius.min(h), radius.min(w));
    mask[..rows * w].fill(0);
    mask[(h - rows) * w..].fill(0);
    for row in mask.chunks_mut(w.max(1)) {
        row[..columns].fill(0);
        row[w - columns..].fill(0);
    }
}

pub(super) fn page_mask(contour: &[crate::geometry::Point], w: usize, h: usize) -> Vec<u8> {
    // An end point lies above exactly the rows before `below`; an edge cuts
    // the rows where only one of its end points does.
    let below = |p: &crate::geometry::Point| (p.y.ceil().max(0.) as usize).min(h);
    let mut cuts = vec![vec![]; h];
    for (a, b) in contour
        .iter()
        .zip(contour.iter().cycle().skip(1))
        .take(contour.len())
    {
        let (first, last) = (below(a).min(below(b)), below(a).max(below(b)));
        for (y, cuts) in cuts.iter_mut().enumerate().take(last).skip(first) {
            let yy = y as f64;
            cuts.push(a.x + (yy - a.y) * (b.x - a.x) / (b.y - a.y));
        }
    }
    let mut mask = vec![0_u8; w * h];
    mask.par_chunks_mut(w.max(1))
        .zip(cuts)
        .for_each(|(row, mut cuts)| {
            cuts.sort_by(f64::total_cmp);
            for pair in cuts.chunks_exact(2) {
                let start = pair[0].ceil().max(0.) as usize;
                let end = (pair[1].floor() + 1.).max(0.) as usize;
                if start < end.min(w) {
                    row[start..end.min(w)].fill(1);
                }
            }
        });
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
    let band = rim_band(w, h);
    let cap = band + 1;
    let close_band = (margin.saturating_mul(3).min(u32::from(band))) as u16;
    let seeded = rim_tiles(mask, contour, w, h);
    let distance = if let Some(prepared) = prepared {
        Cow::Borrowed(
            prepared
                .rim
                .get_or_init(|| rim_distance(mask, contour, w, cap, &seeded))
                .as_slice(),
        )
    } else {
        Cow::Owned(rim_distance(mask, contour, w, cap, &seeded))
    };
    // 1: unvisited ink within the band. 2: ink beyond it, which disqualifies
    // any component it touches; gray cleanup instead cuts components there.
    // Only the band and the pixels beside it are ever looked at.
    let zone = seeded.grown(usize::from(band) + 1).spans();
    let mut state = vec![0_u8; w * h];
    for run in zone.runs() {
        let inks = binary.as_raw()[run.clone()]
            .iter()
            .zip(&distance[run.clone()]);
        for (state, (ink, distance)) in state[run].iter_mut().zip(inks) {
            *state = match (*ink, *distance <= band) {
                (0, true) => 1,
                (0, false) if !gray_frames => 2,
                _ => 0,
            };
        }
    }
    let corners = CORNERS.map(|(sx, sy)| {
        contour
            .iter()
            .max_by(|a, b| (sx * a.x + sy * a.y).total_cmp(&(sx * b.x + sy * b.y)))
            .unwrap()
    });
    let mut component = vec![];
    let short = w.min(h);
    let gap = (short / 12).max(8);
    let mut confirmed = RimSupport::new(w, h, gap, cap);
    let mut pending: Vec<Vec<usize>> = vec![];
    let mut corner_fragments = vec![];
    let mut removed = 0;
    let mut runs = zone.runs();
    let mut run = 0..0;
    loop {
        let Some(seed) = find_from(&state[..run.end], 1, run.start) else {
            match runs.next() {
                Some(next) => run = next,
                None => break,
            }
            continue;
        };
        run.start = seed + 1;
        component.clear();
        component.push(seed);
        state[seed] = 0;
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
            max_distance = max_distance.max(distance[i]);
            for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    let j = yy * w + xx;
                    match state[j] {
                        1 => {
                            state[j] = 0;
                            component.push(j);
                        }
                        2 => entirely_near = false,
                        _ => {}
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
            && is_corner_wedge(&component, w, [min_x, min_y, max_x, max_y], corners, band);
        if corner_wedge {
            corner_fragments.push([min_x, min_y, max_x, max_y]);
        }
        if entirely_near && (tiny_edge_fleck || boundary_stroke || corner_wedge) {
            if corner_wedge || (boundary_stroke && span >= short / 10) {
                confirmed.extend(&component, &distance);
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
    let tangents = ContourTangents::new(contour);
    let chain_seed = gray_tone
        .map(|tone| gray_fragment_chains(&pending, tone, &tangents, w, short, margin, &distance));
    let mut chain_confirmed = RimSupport::new(w, h, gap, cap);
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
                chain_confirmed.extend(points, &distance);
            }
            !accepted
        });
    }
    // Recover short strokes interrupted by white gaps, but only when they
    // continue a confirmed long rim in the same narrow normal-distance strip.
    // Compact isolated numerals have no supporting rim and remain untouched.
    // Shape and position never change between passes; only the support grows.
    struct Candidate {
        points: Vec<usize>,
        reach: [usize; 4],
        levels: Vec<u64>,
        faint: bool,
    }
    let pad = margin.max(2) as usize;
    let mut candidates = vec![];
    for points in pending {
        // Thin detached paper fibers immediately beside an accepted solid
        // corner inherit that evidence, but compact nearby glyphs do not.
        let min_x = points.iter().map(|i| i % w).min().unwrap();
        let max_x = points.iter().map(|i| i % w).max().unwrap();
        let min_y = points.iter().map(|i| i / w).min().unwrap();
        let max_y = points.iter().map(|i| i / w).max().unwrap();
        let (bw, bh) = (max_x - min_x + 1, max_y - min_y + 1);
        let corner_fiber = bw.min(bh) <= pad
            && (bw.max(bh) >= 3 * bw.min(bh) || points.len() <= 8)
            && corner_fragments.iter().any(|[x0, y0, x1, y1]| {
                min_x >= x0.saturating_sub(pad)
                    && min_y >= y0.saturating_sub(pad)
                    && max_x <= x1.saturating_add(pad)
                    && max_y <= y1.saturating_add(pad)
            });
        if corner_fiber {
            for &i in &points {
                binary.as_mut()[i] = 255;
            }
            removed += points.len();
            continue;
        }
        if points.len() < 3 {
            continue;
        }
        let n = points.len() as f64;
        let cx = points.iter().map(|i| (i % w) as f64).sum::<f64>() / n;
        let cy = points.iter().map(|i| (i / w) as f64).sum::<f64>() / n;
        let (tx, ty) = tangents.nearest(cx, cy);
        let (mut along, mut across) = (0., 0.);
        for &i in &points {
            let dx = (i % w) as f64 - cx;
            let dy = (i / w) as f64 - cy;
            along += (dx * tx + dy * ty).powi(2);
            across += (-dx * ty + dy * tx).powi(2);
        }
        if along < 2. * across || along / n < 2. {
            continue;
        }
        let normal = points.iter().map(|i| f64::from(distance[*i])).sum::<f64>() / n;
        let mut levels = vec![0_u64; confirmed.words];
        for level in 0..=cap {
            if (f64::from(level) - normal).abs() <= f64::from(margin.max(2)) {
                levels[usize::from(level) / 64] |= 1 << (level % 64);
            }
        }
        candidates.push(Candidate {
            reach: [
                min_x.saturating_sub(gap),
                min_y.saturating_sub(gap),
                max_x.saturating_add(gap).min(w - 1),
                max_y.saturating_add(gap).min(h - 1),
            ],
            levels,
            faint: gray_tone.is_some_and(|tone| faint_fragment_tone(&points, tone).is_some()),
            points,
        });
    }
    loop {
        let mut changed = false;
        candidates.retain(|candidate| {
            let Candidate {
                points,
                reach,
                levels,
                faint,
            } = candidate;
            let supported = confirmed.supports(*reach, levels, &distance);
            let chain_supported = *faint && chain_confirmed.supports(*reach, levels, &distance);
            if supported || chain_supported {
                for &i in points {
                    binary.as_mut()[i] = 255;
                }
                removed += points.len();
                if supported {
                    confirmed.extend(points, &distance);
                } else {
                    chain_confirmed.extend(points, &distance);
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

/// Confirmed rim pixels in a coarse grid that records which rim distances each
/// cell holds. A box query then reads pixels only in cells that straddle the
/// box edge and contain a matching distance.
pub(super) struct RimSupport {
    cell: usize,
    columns: usize,
    width: usize,
    words: usize,
    levels: Vec<u64>,
    pixels: Vec<Vec<usize>>,
}

impl RimSupport {
    pub(super) fn new(w: usize, h: usize, gap: usize, cap: u16) -> Self {
        let cell = (gap / 4).max(1);
        let columns = w.div_ceil(cell);
        let cells = columns * h.div_ceil(cell);
        let words = usize::from(cap) / 64 + 1;
        Self {
            cell,
            columns,
            width: w,
            words,
            levels: vec![0; cells * words],
            pixels: vec![vec![]; cells],
        }
    }

    pub(super) fn extend(&mut self, points: &[usize], distance: &[u16]) {
        for &i in points {
            let cell = (i / self.width / self.cell) * self.columns + i % self.width / self.cell;
            let level = usize::from(distance[i]);
            self.levels[cell * self.words + level / 64] |= 1 << (level % 64);
            self.pixels[cell].push(i);
        }
    }

    /// Whether any pixel inside the inclusive box lies at one of `levels`.
    pub(super) fn supports(
        &self,
        [x0, y0, x1, y1]: [usize; 4],
        levels: &[u64],
        distance: &[u16],
    ) -> bool {
        let cell = self.cell;
        (y0 / cell..=y1 / cell).any(|row| {
            (x0 / cell..=x1 / cell).any(|column| {
                let index = row * self.columns + column;
                let held = &self.levels[index * self.words..][..self.words];
                if held.iter().zip(levels).all(|(a, b)| a & b == 0) {
                    return false;
                }
                let inside = column * cell >= x0
                    && row * cell >= y0
                    && (column + 1) * cell <= x1 + 1
                    && (row + 1) * cell <= y1 + 1;
                inside
                    || self.pixels[index].iter().any(|&i| {
                        let (x, y) = (i % self.width, i / self.width);
                        let level = usize::from(distance[i]);
                        (x0..=x1).contains(&x)
                            && (y0..=y1).contains(&y)
                            && levels[level / 64] >> (level % 64) & 1 != 0
                    })
            })
        })
    }
}

/// Nearest-segment tangents of a closed contour. Segments are grouped into
/// runs with bounding boxes; a run that cannot beat the best distance is
/// skipped, and the rest are scanned in contour order so ties resolve exactly
/// as in a full scan.
pub(super) struct ContourTangents<'a> {
    contour: &'a [crate::geometry::Point],
    runs: Vec<[f64; 4]>,
}

impl<'a> ContourTangents<'a> {
    const RUN: usize = 32;

    pub(super) fn new(contour: &'a [crate::geometry::Point]) -> Self {
        let runs = (0..contour.len())
            .step_by(Self::RUN)
            .map(|start| {
                let end = (start + Self::RUN).min(contour.len());
                (start..=end).map(|i| &contour[i % contour.len()]).fold(
                    [
                        f64::INFINITY,
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                        f64::NEG_INFINITY,
                    ],
                    |[x0, y0, x1, y1], p| [x0.min(p.x), y0.min(p.y), x1.max(p.x), y1.max(p.y)],
                )
            })
            .collect();
        Self { contour, runs }
    }

    /// Unit tangent of the first segment at minimal distance from the point.
    pub(super) fn nearest(&self, cx: f64, cy: f64) -> (f64, f64) {
        let n = self.contour.len();
        let scan = |run: usize, closest: &mut f64, tangent: &mut (f64, f64)| {
            for i in run * Self::RUN..((run + 1) * Self::RUN).min(n) {
                let (a, b) = (&self.contour[i], &self.contour[(i + 1) % n]);
                let dx = b.x - a.x;
                let dy = b.y - a.y;
                let norm = dx * dx + dy * dy;
                if norm < 1e-9 {
                    continue;
                }
                let t = (((cx - a.x) * dx + (cy - a.y) * dy) / norm).clamp(0., 1.);
                let d = (cx - a.x - t * dx).powi(2) + (cy - a.y - t * dy).powi(2);
                if d < *closest {
                    *closest = d;
                    let length = norm.sqrt();
                    *tangent = (dx / length, dy / length);
                }
            }
        };
        let floor = |[x0, y0, x1, y1]: &[f64; 4]| {
            let dx = (x0 - cx).max(cx - x1).max(0.);
            let dy = (y0 - cy).max(cy - y1).max(0.);
            dx * dx + dy * dy
        };
        let mut limit = f64::INFINITY;
        if let Some(first) = (0..self.runs.len())
            .min_by(|a, b| floor(&self.runs[*a]).total_cmp(&floor(&self.runs[*b])))
        {
            scan(first, &mut limit, &mut (0., 0.));
        }
        // The slack dwarfs rounding in the segment distance, so a skipped run
        // could never have supplied the minimum.
        let limit = limit * (1. + 1e-6) + 1e-6;
        let mut closest = f64::INFINITY;
        let mut tangent = (0., 0.);
        for (run, bounds) in self.runs.iter().enumerate() {
            if floor(bounds) <= limit {
                scan(run, &mut closest, &mut tangent);
            }
        }
        tangent
    }
}

/// Index of the first `value` at or after `from`. Blocks are tested as a
/// whole first because matches are sparse in page-sized rasters.
fn find_from(values: &[u8], value: u8, from: usize) -> Option<usize> {
    let mut start = from;
    for block in values[from..].chunks(64) {
        if block.contains(&value) {
            return block.iter().position(|v| *v == value).map(|i| start + i);
        }
        start += block.len();
    }
    None
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
    tangents: &ContourTangents<'_>,
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
        let tangent = tangents.nearest(center.0, center.1);
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

/// Diagonal directions whose extreme contour points are the page corners.
const CORNERS: [(f64, f64); 4] = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)];

/// Solid tapered corner fragments differ from compact printed glyphs: they
/// have two long outer legs facing a real contour corner and almost no holes.
fn is_corner_wedge(
    pixels: &[usize],
    width: usize,
    bounds: [usize; 4],
    corners: [&crate::geometry::Point; 4],
    band: u16,
) -> bool {
    let [x0, y0, x1, y1] = bounds;
    let (bw, bh) = (x1 - x0 + 1, y1 - y0 + 1);
    let band = usize::from(band);
    if bw.min(bh) < 5 || bw.max(bh) < 10 || bw.max(bh) > 2 * band || pixels.len() * 10 > bw * bh * 7
    {
        return false;
    }
    let near_corner = CORNERS.iter().zip(corners).any(|(&(sx, sy), corner)| {
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
    for ((sx, sy), corner) in CORNERS.into_iter().zip(corners) {
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
