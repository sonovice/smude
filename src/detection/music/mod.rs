//! Classical five-line probes and ink-centred bar tracking. No learned model.
//! A native baseline derived from Smude's tested Python experiment, not OMR
//! assembly. Input is near-upright; steep curves remain a detection limitation.
use crate::geometry::{Barline, Observations, Point, Staff};
use crate::math::{median, slope};
use image::{GrayImage, RgbImage, imageops};
use rayon::prelude::*;

use crate::{Error, Result};

mod bars;
mod staff_edges;
mod staff_smoothing;
mod trace;
use bars::bars;

struct Ink {
    w: usize,
    h: usize,
    pixels: Vec<f64>,
    seed_paper: Vec<bool>,
    allowed: Option<Vec<bool>>,
}
impl Ink {
    fn permits(&self, x: f64, y: f64) -> bool {
        if !x.is_finite() || !y.is_finite() {
            return false;
        }
        let (x, y) = (x.round() as i64, y.round() as i64);
        x >= 0
            && y >= 0
            && x < self.w as i64
            && y < self.h as i64
            && self
                .allowed
                .as_ref()
                .is_none_or(|mask| mask[y as usize * self.w + x as usize])
    }
    fn at(&self, x: f64, y: f64) -> f64 {
        if !self.permits(x, y) {
            return 0.;
        }
        let (x, y) = (x.round() as i64, y.round() as i64);
        if x < 0 || y < 0 || x >= self.w as i64 || y >= self.h as i64 {
            0.
        } else {
            self.pixels[y as usize * self.w + x as usize]
        }
    }
    fn nearby(&self, x: f64, y: f64) -> f64 {
        self.at(x - 1., y)
            .max(self.at(x, y))
            .max(self.at(x + 1., y))
    }
}

/// Ink is more than 12 levels darker than the mean of the 51-pixel box around
/// it, clipped to the image; seed paper is at least 0.65 of the
/// 90th-percentile tone.
fn threshold(gray: &GrayImage) -> Ink {
    const REACH: usize = 25;
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let tones = gray.as_raw();
    let widths: Vec<u32> = (0..w)
        .map(|x| ((x + REACH + 1).min(w) - x.saturating_sub(REACH)) as u32)
        .collect();
    // Tone sums over the box rows of each column, and their running total
    // from `REACH` columns before the image to as many past it: the box of
    // column `x` then spans entries `x` and `x + 2 * REACH + 1`.
    let mut columns = vec![0_u32; w];
    let mut total = vec![0_u32; w + 2 * REACH + 1];
    let mut rows = 0..0;
    let mut pixels = vec![0.; w * h];
    for (y, (pixels, row)) in pixels
        .chunks_exact_mut(w)
        .zip(tones.chunks_exact(w))
        .enumerate()
    {
        let next = y.saturating_sub(REACH)..(y + REACH + 1).min(h);
        for entering in tones[rows.end * w..next.end * w].chunks_exact(w) {
            for (sum, tone) in columns.iter_mut().zip(entering) {
                *sum += u32::from(*tone);
            }
        }
        for leaving in tones[rows.start * w..next.start * w].chunks_exact(w) {
            for (sum, tone) in columns.iter_mut().zip(leaving) {
                *sum -= u32::from(*tone);
            }
        }
        rows = next;
        let mut running = 0;
        for (total, sum) in total[REACH + 1..].iter_mut().zip(&columns) {
            running += sum;
            *total = running;
        }
        total[REACH + w..].fill(running);
        let height = rows.len() as u32;
        let boxes = total.iter().zip(&total[2 * REACH + 1..]);
        for (((pixel, tone), width), (left, right)) in
            pixels.iter_mut().zip(row).zip(&widths).zip(boxes)
        {
            // `tone < sum / area - 12.` in floating point decides the same: the
            // quotient is never within rounding of a whole level it is not on.
            let darker = (u32::from(*tone) + 12) * width * height < right - left;
            *pixel = f64::from(u8::from(darker));
        }
    }
    let mut histogram = [0_usize; 256];
    for tone in tones {
        histogram[usize::from(*tone)] += 1;
    }
    let mut count = 0;
    let reference = histogram.iter().position(|tones| {
        count += tones;
        count > w * h * 9 / 10
    });
    let reference = reference.map_or(0., |tone| tone as f64);
    let seed_paper = tones
        .iter()
        .map(|v| f64::from(*v) >= reference * 0.65)
        .collect();
    Ink {
        w,
        h,
        pixels,
        seed_paper,
        allowed: None,
    }
}

/// Row bits, 64 to a word, as seen `shift` rows further up; rows past the end
/// are unset.
fn rows_below(bits: &[u64], shift: usize) -> impl Iterator<Item = u64> + '_ {
    let (words, offset) = (shift / 64, shift % 64);
    let word = move |index: usize| bits.get(index).copied().unwrap_or(0);
    (words..words + bits.len()).map(move |index| {
        if offset == 0 {
            word(index)
        } else {
            word(index) >> offset | word(index + 1) << (64 - offset)
        }
    })
}

#[derive(Clone)]
struct Hit {
    y: [f64; 5],
    score: f64,
}
fn probes(profile: &[f64]) -> Vec<Hit> {
    let mut candidates = vec![];
    // Row maxima over each pooling radius in turn, with a bit per row for
    // whether they reach the response every line of a comb needs. Each radius
    // widens the previous one.
    let mut pools: Vec<(Vec<f64>, Vec<u64>)> = vec![];
    for space in 3_usize..=24 {
        if 4 * space >= profile.len() {
            continue;
        }
        let radius = (space as f64 * 0.12).round().max(1.) as usize;
        while pools.len() < radius {
            let narrower = pools.last().map_or(profile, |(pooled, _)| pooled);
            let pooled: Vec<f64> = (0..profile.len())
                .map(|y| {
                    narrower[y.saturating_sub(1)..(y + 2).min(profile.len())]
                        .iter()
                        .copied()
                        .fold(0., f64::max)
                })
                .collect();
            let present = pooled
                .chunks(64)
                .map(|rows| {
                    rows.iter()
                        .enumerate()
                        .fold(0, |bits, (row, v)| bits | u64::from(*v >= 0.12) << row)
                })
                .collect();
            pools.push((pooled, present));
        }
        let (pooled, present) = &pools[radius - 1];
        let n = profile.len() - 4 * space;
        let score = |y: usize| {
            let lines: [f64; 5] = std::array::from_fn(|k| pooled[y + k * space]);
            if lines.iter().any(|v| *v < 0.12) {
                0.
            } else {
                lines.iter().sum::<f64>() / 5.
            }
        };
        // A missing line zeroes the score, and nearly every row lacks one.
        let mut complete = present.clone();
        for k in 1..5 {
            for (rows, line) in complete.iter_mut().zip(rows_below(present, k * space)) {
                *rows &= line;
            }
        }
        let complete = complete.iter().enumerate().flat_map(|(word, rows)| {
            std::iter::successors(Some(*rows), |rows| Some(rows & rows.wrapping_sub(1)))
                .take_while(|rows| *rows != 0)
                .map(move |rows| word * 64 + rows.trailing_zeros() as usize)
        });
        for y in complete.filter(|y| (1..n.saturating_sub(1)).contains(y)) {
            let peak = score(y);
            if peak < 0.26 || peak < score(y - 1) || peak <= score(y + 1) {
                continue;
            }
            let yy = std::array::from_fn(|k| {
                let center = y + k * space;
                let lo = center.saturating_sub(radius);
                let hi = (center + radius + 1).min(profile.len());
                (lo..hi)
                    .max_by(|a, b| profile[*a].total_cmp(&profile[*b]))
                    .unwrap() as f64
            });
            if yy
                .windows(2)
                .all(|v| v[1] - v[0] >= 0.65 * space as f64 && v[1] - v[0] <= 1.35 * space as f64)
            {
                candidates.push(Hit { y: yy, score: peak });
            }
        }
    }
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut selected: Vec<Hit> = vec![];
    for hit in candidates {
        if selected.iter().all(|old| {
            (hit.y[2] - old.y[2]).abs()
                > 3. * ((hit.y[4] - hit.y[0]) / 4.).min((old.y[4] - old.y[0]) / 4.)
        }) {
            selected.push(hit);
        }
    }
    selected
}

fn smooth(values: &[f64], sigma: f64) -> Vec<f64> {
    let radius = (sigma * 3.).ceil() as i64;
    let weights: Vec<f64> = (-radius..=radius)
        .map(|j| (-0.5 * (j as f64 / sigma).powi(2)).exp())
        .collect();
    let total = weights.iter().sum::<f64>();
    (0..values.len())
        .map(|i| {
            weights
                .iter()
                .enumerate()
                .map(|(j, w)| {
                    let index =
                        (i as i64 + j as i64 - radius).clamp(0, values.len() as i64 - 1) as usize;
                    w * values[index]
                })
                .sum::<f64>()
                / total
        })
        .collect()
}

// Join only interrupted observations of the same five ruled lines. Geometric
// agreement alone cannot distinguish a staff from separate adjacent examples.
fn can_stitch(ink: &Ink, left: &Staff, right: &Staff, sx: f64, sy: f64) -> bool {
    let a = left.x[left.x.len() - 1];
    let b = right.x[0];
    let gap = (b - a) * sx;
    if gap <= 0. {
        return false;
    }
    let last = left.x.len() - 1;
    let dl = (left.y[4][last] - left.y[0][last]) * sy / 4.;
    let dr = (right.y[4][0] - right.y[0][0]) * sy / 4.;
    let d = dl.min(dr);
    if gap > 8. * d || (dl - dr).abs() > 0.15 * d {
        return false;
    }
    let at = |staff: &Staff, k: usize, x: f64| {
        crate::geometry::observations::interpolate(&staff.x, &staff.y[k], x)
    };
    let window = 2. * d / sx;
    let al = (a - window).max(left.x[0]);
    let br = (b + window).min(*right.x.last().unwrap());
    if a - al < d / sx || br - b < d / sx {
        return false;
    }
    for k in 0..5 {
        let ya = left.y[k][last];
        let yb = right.y[k][0];
        let ml = (ya - at(left, k, al)) / (a - al);
        let mr = (at(right, k, br) - yb) / (br - b);
        // Agreement in both directions prevents changing five-line phase or
        // bridging curves that only happen to meet at one endpoint.
        if ((ya + ml * (b - a)) - yb).abs() * sy > 0.35 * d
            || ((yb - mr * (b - a)) - ya).abs() * sy > 0.35 * d
        {
            return false;
        }
    }
    let mut supported = [0_usize; 5];
    let mut samples = 0;
    for xx in (a * sx).floor() as usize + 1..(b * sx).ceil() as usize {
        let x = xx as f64 / sx;
        let t = (x - a) / (b - a);
        samples += 1;
        for (k, count) in supported.iter_mut().enumerate() {
            let y = (left.y[k][last] * (1. - t) + right.y[k][0] * t) * sy;
            // Never bridge a confirmed boundary, even when a dark fold would
            // otherwise supply abundant ink. Check every original line.
            if !ink.permits(xx as f64, y) {
                return false;
            }
            if [-1., 0., 1.]
                .iter()
                .any(|dy| ink.at(xx as f64, y + dy) > 0.)
            {
                *count += 1;
            }
        }
    }
    samples > 0
        && supported
            .iter()
            .filter(|count| **count as f64 / samples as f64 >= 0.85)
            .count()
            >= 3
}

fn stitch_staves(ink: &Ink, mut staves: Vec<Staff>, sx: f64, sy: f64) -> Vec<Staff> {
    loop {
        let mut best: Option<(f64, usize, usize)> = None;
        for (i, left) in staves.iter().enumerate() {
            for (j, right) in staves.iter().enumerate() {
                if i == j {
                    continue;
                }
                let gap = right.x[0] - left.x[left.x.len() - 1];
                if gap > 0.
                    && best.is_none_or(|old| gap < old.0)
                    && can_stitch(ink, left, right, sx, sy)
                {
                    best = Some((gap, i, j));
                }
            }
        }
        let Some((_, i, j)) = best else {
            return staves;
        };
        let (left, right) = (&staves[i], &staves[j]);
        let a = *left.x.last().unwrap();
        let b = right.x[0];
        let bridge: Vec<_> = ((a * sx).floor() as usize + 1..(b * sx).ceil() as usize)
            .map(|x| x as f64 / sx)
            .collect();
        let mut x = left.x.clone();
        x.extend(&bridge);
        x.extend(&right.x);
        let y = std::array::from_fn(|k| {
            let mut values = left.y[k].clone();
            let ya = *values.last().unwrap();
            let yb = right.y[k][0];
            values.extend(bridge.iter().map(|x| ya + (yb - ya) * (*x - a) / (b - a)));
            values.extend(&right.y[k]);
            values
        });
        let merged = Staff { x, y };
        staves.remove(i.max(j));
        staves.remove(i.min(j));
        staves.push(merged);
    }
}

fn staff_ink_quality(ink: &Ink, staff: &Staff, sx: f64, sy: f64) -> f64 {
    let mut quality = 0.;
    let mut count = 0;
    for j in (0..staff.x.len()).step_by((staff.x.len() / 128).max(1)) {
        let x = staff.x[j] * sx;
        let y = std::array::from_fn::<_, 5, _>(|k| staff.y[k][j] * sy);
        let d = (y[4] - y[0]) / 4.;
        let sample = |y: f64| {
            [-0.75, 0., 0.75]
                .iter()
                .map(|dy| ink.at(x, y + dy))
                .fold(0., f64::max)
        };
        let foreground = y.iter().map(|y| sample(*y)).sum::<f64>() / 5.;
        let outside = sample(y[0] - d).max(sample(y[4] + d));
        quality += foreground - 0.2 * outside;
        count += 1;
    }
    quality / count.max(1) as f64
}

fn staffs(ink: &Ink, sx: f64, sy: f64) -> Vec<Staff> {
    let mut tracks: Vec<Vec<(usize, [f64; 5])>> = vec![];
    let stride = 6;
    let columns: Vec<usize> = (4..ink.w.saturating_sub(4)).step_by(stride).collect();
    let column_hits: Vec<Vec<Hit>> = columns
        .par_iter()
        .map(|x| {
            let profile: Vec<f64> = (0..ink.h)
                .map(|y| {
                    ink.pixels[y * ink.w + x - 4..y * ink.w + x + 5]
                        .iter()
                        .sum::<f64>()
                        / 9.
                })
                .collect();
            probes(&smooth(&profile, 0.55))
        })
        .collect();
    for (x, hits) in columns.into_iter().zip(column_hits) {
        let mut costs = vec![];
        for (i, track) in tracks.iter().enumerate() {
            let (last_x, last) = track.last().unwrap();
            if x - last_x > stride * 16 {
                continue;
            }
            let space = (last[4] - last[0]) / 4.;
            let old = track[track.len().saturating_sub(5)];
            let expected: [f64; 5] = std::array::from_fn(|k| {
                let slope = if *last_x > old.0 {
                    ((last[k] - old.1[k]) / (*last_x - old.0) as f64).clamp(-0.3, 0.3)
                } else {
                    0.
                };
                last[k] + slope * (x - last_x) as f64
            });
            for (j, hit) in hits.iter().enumerate() {
                let offset = (0..5)
                    .map(|k| (expected[k] - hit.y[k]).abs())
                    .fold(0., f64::max);
                let difference = ((hit.y[4] - hit.y[0]) / 4. - space).abs();
                if offset < 3_f64.max(space * 0.65).max((x - last_x) as f64 * 0.12)
                    && difference < space * 0.25
                    && (ink.allowed.is_none()
                        || (*last_x + 1..=x).all(|xx| {
                            let t = (xx - last_x) as f64 / (x - last_x) as f64;
                            (0..5)
                                .all(|k| ink.permits(xx as f64, last[k] * (1. - t) + hit.y[k] * t))
                        }))
                {
                    costs.push((offset + 2. * difference, i, j));
                }
            }
        }
        costs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut used_tracks = vec![false; tracks.len()];
        let mut used_hits = vec![false; hits.len()];
        for (_, i, j) in costs {
            if !used_tracks[i] && !used_hits[j] {
                tracks[i].push((x, hits[j].y));
                used_tracks[i] = true;
                used_hits[j] = true;
            }
        }
        for (j, hit) in hits.iter().enumerate() {
            if !used_hits[j] {
                tracks.push(vec![(x, hit.y)]);
            }
        }
    }
    let planes = trace::Planes::new(ink);
    let out: Vec<Staff> = tracks
        .into_par_iter()
        .filter_map(|track| {
            let (first, last) = (track[0].0, track[track.len() - 1].0);
            if ((last - first) as f64) < (ink.w as f64 * 0.05) || track.len() < 4 {
                return None;
            }
            trace::staff(&planes, &track, sx, sy)
        })
        .collect();
    // Dense lyrics can complete a false, widely spaced five-line comb across
    // TWO real staves. Prefer long tracks measured in their own staff spaces,
    // then reject overlapping bands in the shared x interval. Distinct printed
    // staff bands cannot occupy the same pixels; do not pass these harmonics to
    // geometry and let a size prior legitimize them.
    // Several seeds can trace the same staff. A minute spacing/length
    // difference must not let a slipped five-line phase beat actual ink support.
    // Evaluate after smoothing, against the unchanged input image, once per path.
    let mut scored: Vec<_> = out
        .into_iter()
        .map(|staff| {
            let span = staff.x[staff.x.len() - 1] - staff.x[0];
            let height = median(staff.y[4].clone()) - median(staff.y[0].clone());
            let support = staff_ink_quality(ink, &staff, sx, sy).max(0.);
            let score = span / height * support * support;
            (staff, score)
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut selected: Vec<Staff> = vec![];
    for (staff, _) in scored {
        let overlaps = selected.iter().any(|other| {
            let lo = staff.x[0].max(other.x[0]);
            let hi = staff.x[staff.x.len() - 1].min(other.x[other.x.len() - 1]);
            if lo >= hi {
                return false;
            }
            [lo, (lo + hi) / 2., hi].iter().any(|x| {
                let at = |s: &Staff, k: usize| {
                    crate::geometry::observations::interpolate(&s.x, &s.y[k], *x)
                };
                at(&staff, 0) < at(other, 4) && at(other, 0) < at(&staff, 4)
            })
        });
        if !overlaps {
            selected.push(staff);
        }
    }
    let mut selected = stitch_staves(ink, selected, sx, sy);
    selected.sort_by(|a, b| median(a.y[2].clone()).total_cmp(&median(b.y[2].clone())));
    selected
}

/// Re-measure vertical ink in a rectified preview while retaining staff identities.
pub(crate) fn detect_bars_on_staves(image: &RgbImage, data: &Observations) -> Result<Vec<Barline>> {
    data.validate()?;
    if image.dimensions() != (data.width, data.height) {
        return Err(Error::Invalid(
            "bar preview dimensions differ from staff observations",
        ));
    }
    let ink = threshold(&crate::rendering::output::grayscale(image));
    Ok(bars(&ink, &data.staves, 1., 1.))
}

/// Native image-to-observations baseline. No external Python/OMR dependency.
pub fn detect(image: &RgbImage, max_width: u32) -> Result<Observations> {
    detect_masked(image, max_width, None, None)
}

// Rasterize scanline intersections once. Testing a several-hundred-vertex page
// contour independently for every RGB pixel would dominate the actual warp.
fn region_mask(
    region: &crate::detection::PageRegion,
    width: usize,
    height: usize,
    sx: f64,
    sy: f64,
) -> Vec<bool> {
    let mut mask = vec![false; width * height];
    for row in 0..height {
        let y = row as f64 / sy;
        let mut intersections = vec![];
        let mut spans = vec![];
        for (a, b) in region
            .contour
            .iter()
            .zip(region.contour.iter().cycle().skip(1))
            .take(region.contour.len())
        {
            if (a.y > y) != (b.y > y) {
                intersections.push(a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y));
            }
            if (a.y - y).abs() < 1e-7 && (b.y - y).abs() < 1e-7 {
                spans.push((a.x.min(b.x), a.x.max(b.x)));
            }
        }
        intersections.sort_by(f64::total_cmp);
        spans.extend(intersections.chunks_exact(2).map(|pair| (pair[0], pair[1])));
        for (left, right) in spans {
            let start = (left * sx - 1e-7).ceil().max(0.) as usize;
            let end = ((right * sx + 1e-7).floor().max(-1.) as i64 + 1).max(0) as usize;
            if start < end.min(width) {
                mask[row * width + start..row * width + end.min(width)].fill(true);
            }
        }
    }
    mask
}

/// Integer crop in original pixels. Only the eventual geometric render resamples.
/// Pixels outside the page contour are white, never filled from a neighbouring page.
pub fn crop_page(image: &RgbImage, region: &crate::detection::PageRegion) -> Result<RgbImage> {
    crate::detection::PageRegions {
        width: image.width(),
        height: image.height(),
        regions: vec![region.clone()],
    }
    .validate()?;
    let [x0, y0, x1, y1] = region.bounds();
    let (width, height) = (x1 - x0, y1 - y0);
    // Validated bounds lie inside the image, so whole rows copy without per-pixel access.
    let row = width as usize * 3;
    let mut pixels = Vec::with_capacity(row * height as usize);
    for source in image
        .as_raw()
        .chunks_exact(image.width() as usize * 3)
        .skip(y0 as usize)
        .take(height as usize)
    {
        pixels.extend_from_slice(&source[x0 as usize * 3..][..row]);
    }
    let mut crop =
        RgbImage::from_raw(width, height, pixels).expect("crop buffer matches its dimensions");
    let local = local_region(region, x0, y0);
    let mask = region_mask(
        &local,
        crop.width() as usize,
        crop.height() as usize,
        1.,
        1.,
    );
    crop.par_chunks_mut(row.max(1))
        .zip(mask.par_chunks((width as usize).max(1)))
        .for_each(|(pixels, mask)| {
            let (pixels, _) = pixels.as_chunks_mut::<3>();
            for (pixel, _) in pixels.iter_mut().zip(mask).filter(|(_, inside)| !**inside) {
                *pixel = [255; 3];
            }
        });
    Ok(crop)
}

pub(crate) fn local_region(
    region: &crate::detection::PageRegion,
    x0: u32,
    y0: u32,
) -> crate::detection::PageRegion {
    let mut local = region.clone();
    for p in &mut local.contour {
        p.x -= f64::from(x0);
        p.y -= f64::from(y0);
    }
    local
}

/// Region-bound detection. Coordinates and dimensions refer to the integer page
/// crop; its source offset remains available through `region.bounds()`.
pub fn detect_page(
    image: &RgbImage,
    region: &crate::detection::PageRegion,
    max_width: u32,
) -> Result<Observations> {
    let crop = crop_page(image, region)?;
    detect_cropped_page(&crop, region, max_width)
}

/// Detect in an already masked, full-resolution integer crop, avoiding a second
/// copy. The supplied original-image region fixes its offset and dimensions.
pub fn detect_cropped_page(
    crop: &RgbImage,
    region: &crate::detection::PageRegion,
    max_width: u32,
) -> Result<Observations> {
    let [x0, y0, x1, y1] = region.bounds();
    if crop.dimensions() != (x1 - x0, y1 - y0) {
        return Err(Error::Invalid(
            "page crop dimensions differ from region bounds",
        ));
    }
    let local = local_region(region, x0, y0);
    detect_masked(crop, max_width, Some(&local), None)
}

/// Detect with a mask already expressed in image pixels, without cropping or
/// changing the coordinate origin. Useful for projectively prepared analysis.
pub fn detect_with_region_mask(
    image: &RgbImage,
    region: &crate::detection::PageRegion,
    max_width: u32,
) -> Result<Observations> {
    detect_masked(image, max_width, Some(region), None)
}

type StaffObserver<'a> = dyn Fn(&[Staff]) + 'a;

pub(crate) fn detect_masked(
    image: &RgbImage,
    max_width: u32,
    region: Option<&crate::detection::PageRegion>,
    on_staves: Option<&StaffObserver<'_>>,
) -> Result<Observations> {
    if max_width < 64 || image.width() < 64 || image.height() < 64 {
        return Err(Error::Invalid(
            "detection needs an image and analysis width at least 64 pixels",
        ));
    }
    let width = image.width().min(max_width);
    let height =
        (f64::from(image.height()) * f64::from(width) / f64::from(image.width())).round() as u32;
    let gray = crate::detection::resample::luma(image);
    let analysis = imageops::resize(&gray, width, height, imageops::FilterType::Triangle);
    let mut ink = threshold(&analysis);
    let color = imageops::resize(image, width, height, imageops::FilterType::Triangle);
    for (paper, pixel) in ink.seed_paper.iter_mut().zip(color.pixels()) {
        let high = *pixel.0.iter().max().unwrap() as f64;
        let low = *pixel.0.iter().min().unwrap() as f64;
        *paper &= high - low <= high * 0.35;
    }
    let sx = f64::from(width) / f64::from(image.width());
    let sy = f64::from(height) / f64::from(image.height());
    if let Some(region) = region {
        let allowed = region_mask(region, ink.w, ink.h, sx, sy);
        let pixels = ink.pixels.iter_mut().zip(&mut ink.seed_paper);
        for ((pixel, paper), inside) in pixels.zip(&allowed) {
            if !inside {
                *pixel = 0.;
                *paper = false;
            }
        }
        ink.allowed = Some(allowed);
    }
    let staves = staffs(&ink, sx, sy);
    if staves.is_empty() {
        return Err(Error::Invalid("no reliable five-line staves found"));
    }
    if let Some(observer) = on_staves {
        observer(&staves);
    }
    let barlines = bars(&ink, &staves, sx, sy);
    let data = Observations {
        width: image.width(),
        height: image.height(),
        staves,
        barlines,
    };
    data.validate()?;
    Ok(data)
}

/// Measure staff starts/ends from individual ink ridges, not shared curve x.
/// Coordinates refer to the supplied original crop. Contour-masked source
/// pixels should already be white. Supply the contour-aware variant to
/// distinguish masked-off paper from genuine blank margin at a staff edge.
pub fn detect_staff_edges(
    image: &RgbImage,
    data: &Observations,
    max_width: u32,
) -> Result<Vec<crate::geometry::observations::StaffEdge>> {
    detect_staff_edges_with_region_mask(image, data, max_width, None)
}

/// As `detect_staff_edges`, with a confirmed page contour in this crop's
/// coordinate system. Both inside and outside transition probes must remain
/// inside the contour; a masked-off line ending is not a measured staff edge.
pub fn detect_staff_edges_with_region_mask(
    image: &RgbImage,
    data: &Observations,
    max_width: u32,
    region: Option<&crate::detection::PageRegion>,
) -> Result<Vec<crate::geometry::observations::StaffEdge>> {
    data.validate()?;
    if image.dimensions() != (data.width, data.height)
        || max_width < 64
        || image.width() < 64
        || image.height() < 64
    {
        return Err(Error::Invalid(
            "staff-edge image dimensions differ or analysis width is too small",
        ));
    }
    let width = image.width().min(max_width);
    let height = (f64::from(image.height()) * f64::from(width) / f64::from(image.width()))
        .round()
        .max(1.) as u32;
    let gray = crate::detection::resample::luma(image);
    let analysis = crate::detection::resample::resize_triangle(&gray, width, height);
    let mut ink = threshold(&analysis);
    if let Some(region) = region {
        // Validate topology after a whole-polygon affine normalization. The
        // actual contour stays unchanged: curved mapped rims may lie outside
        // the detector canvas and must be rasterized with natural clipping.
        if region
            .contour
            .iter()
            .any(|p| !p.x.is_finite() || !p.y.is_finite())
        {
            return Err(Error::Invalid("nonfinite staff-edge page mask"));
        }
        let min_x = region
            .contour
            .iter()
            .map(|p| p.x)
            .fold(f64::INFINITY, f64::min);
        let max_x = region
            .contour
            .iter()
            .map(|p| p.x)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_y = region
            .contour
            .iter()
            .map(|p| p.y)
            .fold(f64::INFINITY, f64::min);
        let max_y = region
            .contour
            .iter()
            .map(|p| p.y)
            .fold(f64::NEG_INFINITY, f64::max);
        if max_x <= min_x || max_y <= min_y {
            return Err(Error::Invalid("degenerate staff-edge page mask"));
        }
        let mut normalized = region.clone();
        for p in &mut normalized.contour {
            p.x = (p.x - min_x) / (max_x - min_x) * 1000.;
            p.y = (p.y - min_y) / (max_y - min_y) * 1000.;
        }
        crate::detection::PageRegions {
            width: 1001,
            height: 1001,
            regions: vec![normalized],
        }
        .validate()?;
        ink.allowed = Some(region_mask(
            region,
            ink.w,
            ink.h,
            f64::from(width) / f64::from(image.width()),
            f64::from(height) / f64::from(image.height()),
        ));
    }
    Ok(staff_edges::detect(
        &ink,
        data,
        f64::from(width) / f64::from(image.width()),
        f64::from(height) / f64::from(image.height()),
    ))
}

#[cfg(test)]
mod raster_tests {
    use super::*;

    #[test]
    fn threshold_equals_its_floating_point_definition() {
        // Wider and taller than one box, with flat, noisy and edged areas.
        let (w, h) = (83, 67);
        let gray = GrayImage::from_fn(w, h, |x, y| {
            let noise = (x * 31 + y * 17 + x * y) % 23;
            let stroke = if x % 19 < 2 || y % 29 == 5 { 60 } else { 0 };
            image::Luma([(120 + 4 * (x % 30) + noise - stroke) as u8])
        });
        let ink = threshold(&gray);
        let tone = |x: u32, y: u32| f64::from(gray.get_pixel(x, y)[0]);
        let mut tones = gray.as_raw().clone();
        tones.sort_unstable();
        let reference = f64::from(tones[tones.len() * 9 / 10]);
        let mut inked = 0;
        for y in 0..h {
            for x in 0..w {
                let (x0, x1) = (x.saturating_sub(25), (x + 26).min(w));
                let (y0, y1) = (y.saturating_sub(25), (y + 26).min(h));
                let sum = (y0..y1)
                    .flat_map(|yy| (x0..x1).map(move |xx| (xx, yy)))
                    .map(|(xx, yy)| u64::from(gray.get_pixel(xx, yy)[0]))
                    .sum::<u64>();
                let mean = sum as f64 / ((x1 - x0) * (y1 - y0)) as f64;
                let expected = f64::from(u8::from(tone(x, y) < mean - 12.));
                let index = (y * w + x) as usize;
                assert_eq!(ink.pixels[index], expected, "x={x}, y={y}");
                assert_eq!(ink.seed_paper[index], tone(x, y) >= reference * 0.65);
                inked += expected as usize;
            }
        }
        assert!(inked > 100 && inked < (w * h) as usize / 2);
    }

    #[test]
    fn rows_below_shift_bits_across_words() {
        let bits = [
            0x8000_0000_0000_0001,
            0x0000_0000_0000_0002,
            0x4000_0000_0000_0000,
        ];
        let row =
            |bits: &[u64], row: usize| bits.get(row / 64).is_some_and(|w| w >> (row % 64) & 1 == 1);
        for shift in [0, 1, 63, 64, 65, 126, 127, 128, 191, 192, 500] {
            let shifted: Vec<u64> = rows_below(&bits, shift).collect();
            assert_eq!(shifted.len(), bits.len());
            for y in 0..192 {
                assert_eq!(
                    row(&shifted, y),
                    row(&bits, y + shift),
                    "shift={shift}, y={y}"
                );
            }
        }
    }

    #[test]
    fn probes_find_combs_at_any_row_and_need_all_five_lines() {
        let comb = |top: usize, space: usize, lines: usize| {
            let mut profile = vec![0.; 400];
            for k in 0..lines {
                profile[top + k * space] = 1.;
            }
            probes(&profile)
        };
        for (top, space) in [(5, 9), (58, 12), (120, 20), (300, 24)] {
            let hits = comb(top, space, 5);
            assert_eq!(hits.len(), 1, "top={top}, space={space}");
            let expected: [f64; 5] = std::array::from_fn(|k| (top + k * space) as f64);
            assert_eq!(hits[0].y, expected);
            assert_eq!(hits[0].score, 1.);
            assert!(comb(top, space, 4).is_empty());
        }
    }
}

#[cfg(test)]
mod stitch_tests {
    use super::*;
    fn fixture(occluded: bool) -> (Ink, Vec<Staff>) {
        let mut ink = Ink {
            w: 800,
            h: 300,
            pixels: vec![0.; 800 * 300],
            seed_paper: vec![true; 800 * 300],
            allowed: None,
        };
        for x in 100..=600 {
            if (261..310).contains(&x) && !occluded {
                continue;
            }
            for y in [100, 110, 120, 130, 140] {
                ink.pixels[y * 800 + x] = 1.;
            }
            if occluded && (261..310).contains(&x) {
                for y in 97..144 {
                    ink.pixels[y * 800 + x] = 1.;
                }
            }
        }
        let make = |lo, hi| {
            let x: Vec<_> = (lo..=hi).map(f64::from).collect();
            Staff {
                y: std::array::from_fn(|k| vec![100. + k as f64 * 10.; x.len()]),
                x,
            }
        };
        (ink, vec![make(100, 260), make(310, 600)])
    }
    #[test]
    fn beam_occlusion_joins_without_changing_measured_samples() {
        let (ink, staffs) = fixture(true);
        let original = staffs.clone();
        let stitched = stitch_staves(&ink, staffs, 1., 1.);
        assert_eq!(stitched.len(), 1);
        for staff in &original {
            for (j, x) in staff.x.iter().enumerate() {
                let target = stitched[0].x.binary_search_by(|v| v.total_cmp(x)).unwrap();
                for k in 0..5 {
                    assert_eq!(stitched[0].y[k][target], staff.y[k][j]);
                }
            }
        }
        assert_eq!(stitched[0].x.len(), 501);
    }
    #[test]
    fn aligned_independent_examples_across_white_gap_remain_separate() {
        let (ink, staffs) = fixture(false);
        assert_eq!(stitch_staves(&ink, staffs, 1., 1.).len(), 2);
    }
    #[test]
    fn confirmed_page_barrier_blocks_stitch_even_with_ink() {
        let (mut ink, staffs) = fixture(true);
        let mut allowed = vec![true; 800 * 300];
        for y in 0..300 {
            allowed[y * 800 + 285] = false;
        }
        ink.allowed = Some(allowed);
        assert_eq!(stitch_staves(&ink, staffs, 1., 1.).len(), 2);
    }
}
