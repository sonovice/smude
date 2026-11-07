//! Classical five-line probes and ink-centred bar tracking. No learned model.
//! A native baseline derived from Smude's tested Python experiment, not OMR
//! assembly. Input is near-upright; steep curves remain a detection limitation.
use image::{GrayImage, RgbImage, imageops};

use crate::{Barline, Error, Observations, Point, Result, Staff, median, slope};

mod bars;
mod staff_edges;
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

fn threshold(gray: &GrayImage) -> Ink {
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let mut sum = vec![0_u64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0_u64;
        for x in 0..w {
            row += u64::from(gray.get_pixel(x as u32, y as u32)[0]);
            sum[(y + 1) * (w + 1) + x + 1] = sum[y * (w + 1) + x + 1] + row;
        }
    }
    let mut pixels = vec![0.; w * h];
    for y in 0..h {
        for x in 0..w {
            let (x0, x1, y0, y1) = (
                x.saturating_sub(25),
                (x + 26).min(w),
                y.saturating_sub(25),
                (y + 26).min(h),
            );
            let s = sum[y1 * (w + 1) + x1] + sum[y0 * (w + 1) + x0]
                - sum[y1 * (w + 1) + x0]
                - sum[y0 * (w + 1) + x1];
            let mean = s as f64 / ((x1 - x0) * (y1 - y0)) as f64;
            pixels[y * w + x] = if f64::from(gray.get_pixel(x as u32, y as u32)[0]) < mean - 12. {
                1.
            } else {
                0.
            };
        }
    }
    let mut tones = gray.as_raw().clone();
    tones.sort_unstable();
    let reference = f64::from(tones[tones.len() * 9 / 10]);
    let seed_paper = gray
        .as_raw()
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

#[derive(Clone)]
struct Hit {
    y: [f64; 5],
    score: f64,
}
fn probes(profile: &[f64]) -> Vec<Hit> {
    let mut candidates = vec![];
    for space in 3_usize..=24 {
        if 4 * space >= profile.len() {
            continue;
        }
        let radius = (space as f64 * 0.12).round().max(1.) as usize;
        let pooled: Vec<f64> = (0..profile.len())
            .map(|y| {
                profile[y.saturating_sub(radius)..(y + radius + 1).min(profile.len())]
                    .iter()
                    .copied()
                    .fold(0., f64::max)
            })
            .collect();
        let n = profile.len() - 4 * space;
        let score: Vec<f64> = (0..n)
            .map(|y| {
                let lines: [f64; 5] = std::array::from_fn(|k| pooled[y + k * space]);
                if lines.iter().any(|v| *v < 0.12) {
                    0.
                } else {
                    lines.iter().sum::<f64>() / 5.
                }
            })
            .collect();
        for y in 1..n.saturating_sub(1) {
            if score[y] < 0.26 || score[y] < score[y - 1] || score[y] <= score[y + 1] {
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
                candidates.push(Hit {
                    y: yy,
                    score: score[y],
                });
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
        crate::observations::interpolate(&staff.x, &staff.y[k], x)
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

fn staffs(ink: &Ink, sx: f64, sy: f64) -> Vec<Staff> {
    let mut tracks: Vec<Vec<(usize, [f64; 5])>> = vec![];
    let stride = 6;
    for x in (4..ink.w.saturating_sub(4)).step_by(stride) {
        let profile: Vec<f64> = (0..ink.h)
            .map(|y| {
                ink.pixels[y * ink.w + x - 4..y * ink.w + x + 5]
                    .iter()
                    .sum::<f64>()
                    / 9.
            })
            .collect();
        let hits = probes(&smooth(&profile, 0.55));
        let mut costs = vec![];
        for (i, track) in tracks.iter().enumerate() {
            let (last_x, last) = track.last().unwrap();
            if x - last_x > stride * 16 {
                continue;
            }
            let space = (last[4] - last[0]) / 4.;
            let old = track[track.len().saturating_sub(5)];
            for (j, hit) in hits.iter().enumerate() {
                let offset = (0..5)
                    .map(|k| {
                        let slope = if *last_x > old.0 {
                            ((last[k] - old.1[k]) / (*last_x - old.0) as f64).clamp(-0.3, 0.3)
                        } else {
                            0.
                        };
                        (last[k] + slope * (x - last_x) as f64 - hit.y[k]).abs()
                    })
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
    let mut out = vec![];
    for track in tracks {
        let (first, last) = (track[0].0, track[track.len() - 1].0);
        if ((last - first) as f64) < (ink.w as f64 * 0.05) || track.len() < 4 {
            continue;
        }
        if let Some(staff) = trace::staff(ink, &track, sx, sy) {
            out.push(staff);
        }
    }
    // Dense lyrics can complete a false, widely spaced five-line comb across
    // TWO real staves. Prefer long tracks measured in their own staff spaces,
    // then reject overlapping bands in the shared x interval. Distinct printed
    // staff bands cannot occupy the same pixels; do not pass these harmonics to
    // geometry and let a size prior legitimize them.
    out.sort_by(|a, b| {
        let score = |s: &Staff| {
            (s.x[s.x.len() - 1] - s.x[0]) / (median(s.y[4].clone()) - median(s.y[0].clone()))
        };
        score(b).total_cmp(&score(a))
    });
    let mut selected: Vec<Staff> = vec![];
    for staff in out {
        let overlaps = selected.iter().any(|other| {
            let lo = staff.x[0].max(other.x[0]);
            let hi = staff.x[staff.x.len() - 1].min(other.x[other.x.len() - 1]);
            if lo >= hi {
                return false;
            }
            [lo, (lo + hi) / 2., hi].iter().any(|x| {
                let at = |s: &Staff, k: usize| crate::observations::interpolate(&s.x, &s.y[k], *x);
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

/// Native image-to-observations baseline. No external Python/OMR dependency.
pub fn detect(image: &RgbImage, max_width: u32) -> Result<Observations> {
    detect_masked(image, max_width, None, None)
}

// Rasterize scanline intersections once. Testing a several-hundred-vertex page
// contour independently for every RGB pixel would dominate the actual warp.
fn region_mask(
    region: &crate::PageRegion,
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
pub fn crop_page(image: &RgbImage, region: &crate::PageRegion) -> Result<RgbImage> {
    crate::PageRegions {
        width: image.width(),
        height: image.height(),
        regions: vec![region.clone()],
    }
    .validate()?;
    let [x0, y0, x1, y1] = region.bounds();
    let mut crop = imageops::crop_imm(image, x0, y0, x1 - x0, y1 - y0).to_image();
    let local = local_region(region, x0, y0);
    let mask = region_mask(
        &local,
        crop.width() as usize,
        crop.height() as usize,
        1.,
        1.,
    );
    for (inside, pixel) in mask.iter().zip(crop.pixels_mut()) {
        if !inside {
            *pixel = image::Rgb([255; 3]);
        }
    }
    Ok(crop)
}

pub(crate) fn local_region(region: &crate::PageRegion, x0: u32, y0: u32) -> crate::PageRegion {
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
    region: &crate::PageRegion,
    max_width: u32,
) -> Result<Observations> {
    let crop = crop_page(image, region)?;
    detect_cropped_page(&crop, region, max_width)
}

/// Detect in an already masked, full-resolution integer crop, avoiding a second
/// copy. The supplied original-image region fixes its offset and dimensions.
pub fn detect_cropped_page(
    crop: &RgbImage,
    region: &crate::PageRegion,
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
    region: &crate::PageRegion,
    max_width: u32,
) -> Result<Observations> {
    detect_masked(image, max_width, Some(region), None)
}

type StaffObserver<'a> = dyn Fn(&[Staff]) + 'a;

pub(crate) fn detect_masked(
    image: &RgbImage,
    max_width: u32,
    region: Option<&crate::PageRegion>,
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
    let gray = image::DynamicImage::ImageRgb8(image.clone()).into_luma8();
    let analysis = imageops::resize(&gray, width, height, imageops::FilterType::Triangle);
    let mut ink = threshold(&analysis);
    let color = imageops::resize(image, width, height, imageops::FilterType::Triangle);
    for (i, pixel) in color.pixels().enumerate() {
        let high = *pixel.0.iter().max().unwrap() as f64;
        let low = *pixel.0.iter().min().unwrap() as f64;
        ink.seed_paper[i] &= high - low <= high * 0.35;
    }
    let sx = f64::from(width) / f64::from(image.width());
    let sy = f64::from(height) / f64::from(image.height());
    if let Some(region) = region {
        let allowed = region_mask(region, ink.w, ink.h, sx, sy);
        for y in 0..ink.h {
            for x in 0..ink.w {
                let index = y * ink.w + x;
                if !allowed[index] {
                    ink.pixels[index] = 0.;
                    ink.seed_paper[index] = false;
                }
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
) -> Result<Vec<crate::observations::StaffEdge>> {
    detect_staff_edges_with_region_mask(image, data, max_width, None)
}

/// As `detect_staff_edges`, with a confirmed page contour in this crop's
/// coordinate system. Both inside and outside transition probes must remain
/// inside the contour; a masked-off line ending is not a measured staff edge.
pub fn detect_staff_edges_with_region_mask(
    image: &RgbImage,
    data: &Observations,
    max_width: u32,
    region: Option<&crate::PageRegion>,
) -> Result<Vec<crate::observations::StaffEdge>> {
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
    let gray = image::DynamicImage::ImageRgb8(image.clone()).into_luma8();
    let analysis = imageops::resize(&gray, width, height, imageops::FilterType::Triangle);
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
        crate::PageRegions {
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
