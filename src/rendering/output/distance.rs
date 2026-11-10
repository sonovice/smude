//! Exact capped Chebyshev distance fields.
use rayon::prelude::*;
use std::ops::Range;

/// Side of the square blocks in which a field is classified as worth sweeping.
const TILE: usize = 64;

/// Exact capped distance transform, in place. `field` holds 0 at seeds and
/// `cap` elsewhere; no value may exceed `cap`.
pub(super) fn capped_distance<T>(field: &mut [T], w: usize, h: usize, cap: T)
where
    T: Copy + Ord + From<u8> + Into<usize> + std::ops::Add<Output = T> + Send + Sync,
{
    capped_distance_in_bands(field, w, h, cap, rayon::current_num_threads());
}

/// [`capped_distance`] with the rows split into at most `bands` bands that are
/// swept side by side. A value depends only on the field within `cap` rows of
/// it, so a band swept together with that many rows around it is exact.
pub(super) fn capped_distance_in_bands<T>(field: &mut [T], w: usize, h: usize, cap: T, bands: usize)
where
    T: Copy + Ord + From<u8> + Into<usize> + std::ops::Add<Output = T> + Send + Sync,
{
    if w == 0 || h == 0 {
        return;
    }
    let halo: usize = cap.into();
    // The rows around a band are swept with both neighbours as well; bands
    // several times taller keep that a small share of the work.
    let bands = bands.min(h / (4 * halo.max(1))).max(1);
    if bands == 1 {
        return sweep(field, w, h, cap);
    }
    let rows = h.div_ceil(bands);
    let source = &*field;
    let swept: Vec<_> = (0..h.div_ceil(rows))
        .into_par_iter()
        .map(|band| {
            let first = (band * rows).saturating_sub(halo);
            let last = ((band + 1) * rows + halo).min(h);
            let mut local = source[first * w..last * w].to_vec();
            sweep(&mut local, w, last - first, cap);
            (band * rows - first, local)
        })
        .collect();
    field
        .par_chunks_mut(rows * w)
        .zip(swept)
        .for_each(|(band, (skipped, local))| {
            band.copy_from_slice(&local[skipped * w..][..band.len()]);
        });
}

/// Two chamfer sweeps with unit weights, which are exact for the Chebyshev
/// metric. They visit only tiles a seed can reach: every other pixel already
/// holds its final value, which keeps the restricted sweeps exact.
fn sweep<T>(field: &mut [T], w: usize, h: usize, cap: T)
where
    T: Copy + Ord + From<u8> + Into<usize> + std::ops::Add<Output = T> + Send + Sync,
{
    let spans = reachable_spans(field, w, h, cap);
    let one = T::from(1);
    for y in 0..h {
        let (above, below) = field.split_at_mut(y * w);
        let row = &mut below[..w];
        for span in &spans[y / TILE] {
            if y > 0 {
                relax(row, &above[(y - 1) * w..], span, one);
            }
            let mut reach = if span.start > 0 {
                row[span.start - 1] + one
            } else {
                cap
            };
            for value in &mut row[span.clone()] {
                *value = (*value).min(reach);
                reach = *value + one;
            }
        }
    }
    for y in (0..h).rev() {
        let (above, below) = field.split_at_mut((y + 1) * w);
        let row = &mut above[y * w..];
        for span in &spans[y / TILE] {
            if y + 1 < h {
                relax(row, &below[..w], span, one);
            }
            let mut reach = if span.end < w {
                row[span.end] + one
            } else {
                cap
            };
            for value in row[span.clone()].iter_mut().rev() {
                *value = (*value).min(reach);
                reach = *value + one;
            }
        }
    }
}

/// Lower `row[span]` to one more than the three vertical neighbours in `other`.
fn relax<T>(row: &mut [T], other: &[T], span: &Range<usize>, one: T)
where
    T: Copy + Ord + std::ops::Add<Output = T>,
{
    let w = row.len();
    if span.start == 0 {
        row[0] = row[0].min(other[0].min(other[1.min(w - 1)]) + one);
    }
    if span.end == w && w > 1 {
        row[w - 1] = row[w - 1].min(other[w - 2].min(other[w - 1]) + one);
    }
    let (start, end) = (span.start.max(1), span.end.min(w - 1));
    if start < end {
        let left = &other[start - 1..end - 1];
        let right = &other[start + 1..end + 1];
        for (((value, a), b), c) in row[start..end]
            .iter_mut()
            .zip(left)
            .zip(&other[start..end])
            .zip(right)
        {
            *value = (*value).min((*a).min(*b).min(*c) + one);
        }
    }
}

/// Pixel column ranges, per tile row, whose values a seed can still lower:
/// tiles within `cap - 1` pixels of a tile holding a value below `cap`, except
/// tiles which are zero throughout.
fn reachable_spans<T>(field: &[T], w: usize, h: usize, cap: T) -> Vec<Vec<Range<usize>>>
where
    T: Copy + Ord + From<u8> + Into<usize> + Send + Sync,
{
    let (columns, rows) = (w.div_ceil(TILE), h.div_ceil(TILE));
    let zero = T::from(0);
    // Per tile: (holds a value below cap, holds a value above zero).
    let mut tiles = vec![(false, false); columns * rows];
    tiles
        .par_chunks_mut(columns)
        .zip(field.par_chunks(w * TILE))
        .for_each(|(tiles, band)| {
            for row in band.chunks(w) {
                for (tile, values) in tiles.iter_mut().zip(row.chunks(TILE)) {
                    tile.0 |= values.iter().copied().min().is_some_and(|v| v < cap);
                    tile.1 |= values.iter().copied().max().is_some_and(|v| v > zero);
                }
            }
        });
    let reach = cap.into().saturating_sub(1).div_ceil(TILE);
    // Separable dilation of the seeded tiles by `reach` tiles.
    let mut near = vec![false; columns * rows];
    for (near, tiles) in near.chunks_mut(columns).zip(tiles.chunks(columns)) {
        let mut last = None;
        for c in 0..columns {
            last = if tiles[c].0 { Some(c) } else { last };
            near[c] = last.is_some_and(|seeded| c - seeded <= reach);
        }
        let mut next = None;
        for c in (0..columns).rev() {
            next = if tiles[c].0 { Some(c) } else { next };
            near[c] |= next.is_some_and(|seeded| seeded - c <= reach);
        }
    }
    let mut reachable = vec![false; columns * rows];
    for c in 0..columns {
        let mut last = None;
        for r in 0..rows {
            last = if near[r * columns + c] { Some(r) } else { last };
            reachable[r * columns + c] = last.is_some_and(|seeded| r - seeded <= reach);
        }
        let mut next = None;
        for r in (0..rows).rev() {
            next = if near[r * columns + c] { Some(r) } else { next };
            reachable[r * columns + c] |= next.is_some_and(|seeded| seeded - r <= reach);
        }
    }
    reachable
        .chunks(columns)
        .zip(tiles.chunks(columns))
        .map(|(reachable, tiles)| {
            let mut spans: Vec<Range<usize>> = vec![];
            for c in (0..columns).filter(|c| reachable[*c] && tiles[*c].1) {
                let (start, end) = (c * TILE, ((c + 1) * TILE).min(w));
                match spans.last_mut() {
                    Some(span) if span.end == start => span.end = end,
                    _ => spans.push(start..end),
                }
            }
            spans
        })
        .collect()
}

/// Zero the pixels of the mapped perimeter. The image canvas is deliberately
/// not treated as a physical page boundary: a crop edge can cut through page
/// interior.
fn seed_contour(field: &mut [u16], contour: &[crate::geometry::Point], w: usize, h: usize) {
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
                field[y as usize * w + x as usize] = 0;
            }
        }
    }
}

/// Capped Chebyshev distance to the mapped perimeter.
pub(super) fn distance_to_contour(
    contour: &[crate::geometry::Point],
    w: usize,
    h: usize,
    cap: u16,
) -> Vec<u16> {
    let mut distance = vec![cap; w * h];
    seed_contour(&mut distance, contour, w, h);
    capped_distance(&mut distance, w, h, cap);
    distance
}

/// Capped Chebyshev distance to the mapped perimeter or anything outside the
/// page mask.
pub(super) fn rim_distance(
    mask: &[u8],
    contour: &[crate::geometry::Point],
    w: usize,
    h: usize,
    cap: u16,
) -> Vec<u16> {
    let mut distance: Vec<u16> = mask
        .par_iter()
        .map(|&v| if v == 0 { 0 } else { cap })
        .collect();
    seed_contour(&mut distance, contour, w, h);
    capped_distance(&mut distance, w, h, cap);
    distance
}
