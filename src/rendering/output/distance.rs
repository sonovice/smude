//! Exact capped Chebyshev distance fields.
use rayon::prelude::*;
use std::ops::Range;

/// Side of the square blocks in which a raster is described coarsely.
pub(super) const TILE: usize = 64;

/// Pixels of a row that a sweep tests at once for being settled.
const BLOCK: usize = 32;

/// A set of tiles: the part of a raster that some work concerns.
pub(super) struct Tiles {
    w: usize,
    h: usize,
    columns: usize,
    held: Vec<bool>,
}

impl Tiles {
    pub(super) fn empty(w: usize, h: usize) -> Self {
        let columns = w.div_ceil(TILE);
        Self {
            w,
            h,
            columns,
            held: vec![false; columns * h.div_ceil(TILE)],
        }
    }

    /// The tiles holding a value below `limit`.
    pub(super) fn below<T>(field: &[T], w: usize, h: usize, limit: T) -> Self
    where
        T: Copy + Ord + Send + Sync,
    {
        let mut tiles = Self::empty(w, h);
        tiles
            .held
            .par_chunks_mut(tiles.columns.max(1))
            .zip(field.par_chunks((w * TILE).max(1)))
            .for_each(|(held, band)| {
                for row in band.chunks(w) {
                    for (held, values) in held.iter_mut().zip(row.chunks(TILE)) {
                        // Folding, unlike `any`, tests a whole tile row at once.
                        *held = *held || values.iter().fold(false, |any, v| any | (*v < limit));
                    }
                }
            });
        tiles
    }

    /// Add the tile of a pixel.
    pub(super) fn insert(&mut self, x: usize, y: usize) {
        self.held[y / TILE * self.columns + x / TILE] = true;
    }

    /// Every tile within `pixels` of a tile in the set.
    pub(super) fn grown(&self, pixels: usize) -> Self {
        let (columns, rows, reach) = (
            self.columns,
            self.held.len() / self.columns.max(1),
            pixels.div_ceil(TILE),
        );
        // Separable dilation by `reach` tiles.
        let mut near = vec![false; self.held.len()];
        for (near, held) in near
            .chunks_mut(columns.max(1))
            .zip(self.held.chunks(columns.max(1)))
        {
            let mut last = None;
            for c in 0..columns {
                last = if held[c] { Some(c) } else { last };
                near[c] = last.is_some_and(|held| c - held <= reach);
            }
            let mut next = None;
            for c in (0..columns).rev() {
                next = if held[c] { Some(c) } else { next };
                near[c] |= next.is_some_and(|held| held - c <= reach);
            }
        }
        let mut grown = Self::empty(self.w, self.h);
        for c in 0..columns {
            let mut last = None;
            for r in 0..rows {
                last = if near[r * columns + c] { Some(r) } else { last };
                grown.held[r * columns + c] = last.is_some_and(|held| r - held <= reach);
            }
            let mut next = None;
            for r in (0..rows).rev() {
                next = if near[r * columns + c] { Some(r) } else { next };
                grown.held[r * columns + c] |= next.is_some_and(|held| held - r <= reach);
            }
        }
        grown
    }

    /// The tiles a distance transform capped at `cap` reads or writes when
    /// the seeds of these tiles spread: those they reach and their neighbours.
    pub(super) fn swept(&self, cap: usize) -> Self {
        self.grown(cap.saturating_sub(1)).grown(1)
    }

    /// Add the tiles of another set over the same raster.
    pub(super) fn include(&mut self, other: &Self) {
        for (held, other) in self.held.iter_mut().zip(&other.held) {
            *held |= other;
        }
    }

    pub(super) fn spans(&self) -> Spans {
        let rows = self.held.chunks(self.columns.max(1)).map(|held| {
            let mut spans: Vec<Range<usize>> = vec![];
            for c in (0..self.columns).filter(|c| held[*c]) {
                let (start, end) = (c * TILE, ((c + 1) * TILE).min(self.w));
                match spans.last_mut() {
                    Some(span) if span.end == start => span.end = end,
                    _ => spans.push(start..end),
                }
            }
            spans
        });
        Spans {
            w: self.w,
            h: self.h,
            rows: rows.collect(),
        }
    }
}

/// The pixel columns of a set of tiles, as ranges per row of tiles.
pub(super) struct Spans {
    w: usize,
    h: usize,
    rows: Vec<Vec<Range<usize>>>,
}

impl Spans {
    /// The column ranges of pixel row `y`.
    pub(super) fn row(&self, y: usize) -> &[Range<usize>] {
        &self.rows[y / TILE]
    }

    /// The pixel index ranges of all rows, in raster order.
    pub(super) fn runs(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        (0..self.h).flat_map(move |y| {
            let row = y * self.w;
            (self.row(y).iter()).map(move |span| row + span.start..row + span.end)
        })
    }
}

/// Capped distance transform, in place, of the seeds in `seeded` tiles.
/// Within their [`Tiles::swept`] `field` holds 0 at seeds and `cap`, which is
/// not the largest value of `T`, elsewhere; other tiles hold what they like
/// and keep it. A swept value is never too small, and exact when `seeded`
/// covers every seed within `cap - 1` pixels of it.
pub(super) fn capped_distance<T>(field: &mut [T], cap: T, seeded: &Tiles)
where
    T: Copy + Ord + From<u8> + Into<usize> + std::ops::Add<Output = T> + Send + Sync,
{
    capped_distance_in_bands(field, cap, seeded, rayon::current_num_threads());
}

/// [`capped_distance`] with the rows split into at most `bands` bands that are
/// swept side by side. A value depends only on the field within `cap` rows of
/// it, so a band is exact once each sweep reaches it from that many rows away.
pub(super) fn capped_distance_in_bands<T>(field: &mut [T], cap: T, seeded: &Tiles, bands: usize)
where
    T: Copy + Ord + From<u8> + Into<usize> + std::ops::Add<Output = T> + Send + Sync,
{
    let (w, h) = (seeded.w, seeded.h);
    if w == 0 || h == 0 {
        return;
    }
    let halo: usize = cap.into();
    // The rows before a band are swept by its neighbour as well; bands
    // several times taller keep that a small share of the work.
    let bands = bands.min(h / (4 * halo.max(1))).max(1);
    let spans = reachable_spans(field, cap, seeded);
    if bands == 1 {
        sweep_down(field, w, 0, None, &spans, cap);
        return sweep_up(field, w, 0, None, &spans, cap);
    }
    let rows = h.div_ceil(bands);
    // Each sweep starts on a copy of the rows it passes before its band,
    // taken while the neighbouring band has not yet changed them.
    let copies = |field: &[T], before: &dyn Fn(usize) -> Range<usize>| -> Vec<Vec<T>> {
        (0..h.div_ceil(rows))
            .map(|band| {
                let rows = before(band * rows);
                field[rows.start * w..rows.end * w].to_vec()
            })
            .collect()
    };
    let above = copies(field, &|first| first.saturating_sub(halo)..first);
    field
        .par_chunks_mut(rows * w)
        .zip(above)
        .enumerate()
        .for_each(|(band, (own, mut above))| {
            let (first, lead) = (band * rows, above.len() / w);
            sweep_down(&mut above, w, first - lead, None, &spans, cap);
            sweep_down(own, w, first, above.chunks(w).last(), &spans, cap);
        });
    let below = copies(field, &|first| {
        let end = (first + rows).min(h);
        end..(end + halo).min(h)
    });
    field
        .par_chunks_mut(rows * w)
        .zip(below)
        .enumerate()
        .for_each(|(band, (own, mut below))| {
            let first = band * rows;
            sweep_up(&mut below, w, first + own.len() / w, None, &spans, cap);
            sweep_up(own, w, first, below.chunks(w).next(), &spans, cap);
        });
}

/// First of two chamfer sweeps with unit weights, which together are exact
/// for the Chebyshev metric. `rows` starts at image row `first` and follows
/// the swept row `before`. Only `spans`, the tiles a seed can reach, are
/// visited, and of those only the blocks a scan along the row would change.
fn sweep_down<T>(
    rows: &mut [T],
    w: usize,
    first: usize,
    before: Option<&[T]>,
    spans: &Spans,
    cap: T,
) where
    T: Copy + Ord + From<u8> + std::ops::Add<Output = T>,
{
    let one = T::from(1);
    for y in 0..rows.len() / w {
        let (above, below) = rows.split_at_mut(y * w);
        let row = &mut below[..w];
        let before = above.chunks(w).last().or(before);
        for span in spans.row(first + y) {
            if let Some(before) = before {
                relax(row, before, span, one);
            }
            let mut start = span.start;
            while start < span.end {
                let end = (start + BLOCK).min(span.end);
                let mut reach = if start > 0 { row[start - 1] + one } else { cap };
                if row[start] > reach
                    || !rises_gently(&row[start..end - 1], &row[start + 1..end], one)
                {
                    for value in &mut row[start..end] {
                        *value = (*value).min(reach);
                        reach = *value + one;
                    }
                }
                start = end;
            }
        }
    }
}

/// Second chamfer sweep, from the last of `rows` to the first; `before` is
/// the swept row below them.
fn sweep_up<T>(rows: &mut [T], w: usize, first: usize, before: Option<&[T]>, spans: &Spans, cap: T)
where
    T: Copy + Ord + From<u8> + std::ops::Add<Output = T>,
{
    let one = T::from(1);
    for y in (0..rows.len() / w).rev() {
        let (above, below) = rows.split_at_mut((y + 1) * w);
        let row = &mut above[y * w..];
        let before = below.chunks(w).next().or(before);
        for span in spans.row(first + y) {
            if let Some(before) = before {
                relax(row, before, span, one);
            }
            let mut end = span.end;
            while end > span.start {
                let start = end.saturating_sub(BLOCK).max(span.start);
                let mut reach = if end < w { row[end] + one } else { cap };
                if row[end - 1] > reach
                    || !rises_gently(&row[start + 1..end], &row[start..end - 1], one)
                {
                    for value in row[start..end].iter_mut().rev() {
                        *value = (*value).min(reach);
                        reach = *value + one;
                    }
                }
                end = start;
            }
        }
    }
}

/// Whether no value of `next` exceeds its counterpart in `from` by more than
/// one. A scan across such pixels changes nothing, and unlike the scan, which
/// hands each value on to the next pixel, this test is vectorized.
fn rises_gently<T>(from: &[T], next: &[T], one: T) -> bool
where
    T: Copy + Ord + std::ops::Add<Output = T>,
{
    (from.iter().zip(next)).fold(true, |gentle, (from, next)| gentle & (*next <= *from + one))
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

/// Where a seed can still lower a value: the tiles within `cap - 1` pixels of
/// a seeded one, except tiles which are zero throughout.
fn reachable_spans<T>(field: &[T], cap: T, seeded: &Tiles) -> Spans
where
    T: Copy + Ord + From<u8> + Into<usize> + Send + Sync,
{
    let mut reachable = seeded.grown(cap.into().saturating_sub(1));
    let (w, zero) = (reachable.w, T::from(0));
    reachable
        .held
        .par_chunks_mut(reachable.columns)
        .zip(field.par_chunks(w * TILE))
        .for_each(|(held, band)| {
            for (c, held) in held.iter_mut().enumerate().filter(|(_, held)| **held) {
                let columns = c * TILE..((c + 1) * TILE).min(w);
                *held = (band.chunks(w)).any(|row| {
                    row[columns.clone()]
                        .iter()
                        .fold(false, |any, v| any | (*v > zero))
                });
            }
        });
    reachable.spans()
}

/// Visit the pixels of the mapped perimeter. The image canvas is deliberately
/// not treated as a physical page boundary: a crop edge can cut through page
/// interior.
fn contour_pixels(
    contour: &[crate::geometry::Point],
    w: usize,
    h: usize,
    mut visit: impl FnMut(usize, usize),
) {
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
                visit(x as usize, y as usize);
            }
        }
    }
}

/// The tiles the mapped perimeter crosses.
pub(super) fn contour_tiles(contour: &[crate::geometry::Point], w: usize, h: usize) -> Tiles {
    let mut tiles = Tiles::empty(w, h);
    contour_pixels(contour, w, h, |x, y| tiles.insert(x, y));
    tiles
}

/// The tiles holding a pixel of the mapped perimeter or outside the page mask.
pub(super) fn rim_tiles(
    mask: &[u8],
    contour: &[crate::geometry::Point],
    w: usize,
    h: usize,
) -> Tiles {
    let mut tiles = Tiles::below(mask, w, h, 1);
    contour_pixels(contour, w, h, |x, y| tiles.insert(x, y));
    tiles
}

/// Capped Chebyshev distance to the mapped perimeter.
pub(super) fn distance_to_contour(
    contour: &[crate::geometry::Point],
    w: usize,
    h: usize,
    cap: u16,
) -> Vec<u16> {
    let mut distance = vec![cap; w * h];
    contour_pixels(contour, w, h, |x, y| distance[y * w + x] = 0);
    capped_distance(&mut distance, cap, &contour_tiles(contour, w, h));
    distance
}

/// Capped Chebyshev distance to the mapped perimeter or anything outside the
/// page mask; `seeded` is their [`rim_tiles`].
pub(super) fn rim_distance(
    mask: &[u8],
    contour: &[crate::geometry::Point],
    w: usize,
    cap: u16,
    seeded: &Tiles,
) -> Vec<u16> {
    // One vectorized pass fills fresh memory faster than threads sharing it.
    let mut distance: Vec<u16> = mask.iter().map(|&v| if v == 0 { 0 } else { cap }).collect();
    contour_pixels(contour, w, seeded.h, |x, y| distance[y * w + x] = 0);
    capped_distance(&mut distance, cap, seeded);
    distance
}
