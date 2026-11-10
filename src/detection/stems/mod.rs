//! Experimental, standalone head-supported stem geometry.
//!
//! Input is a binary music image (black below 128) and staff/bar observations
//! in that exact raster. Output preserves measured source positions. It is not
//! a complete musical classification; the default mesh uses its conservative
//! geometry subset.
mod ported;
use ported::edge;
mod heads;
mod pixels;
mod probe;
mod scaled;
pub use scaled::detect_stems_scaled;

use crate::geometry::{Observations, Point, Staff};
use crate::{Error, Result};
use image::GrayImage;
use pixels::{Pixels, View};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HeadShape {
    Filled,
    Open,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeadEvidence {
    pub center: Point,
    pub bounds: [f64; 4],
    pub shape: HeadShape,
    /// Template support, not a calibrated classification probability.
    pub score: f64,
    /// Share of template samples outside known staff-rule bands.
    pub known_support: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StemObservation {
    /// Staff search context, not an inferred voice or cross-staff assignment.
    pub staff: usize,
    /// Measured ink centres in the supplied binary raster, ordered by y.
    /// Gaps across staff rules, heads and beams are retained as gaps.
    pub points: Vec<Point>,
    pub heads: Vec<HeadEvidence>,
    /// Strong geometry requires more support than symbol detection.
    pub geometry_usable: bool,
    /// Measured dx/dy, not a prescribed upright direction.
    pub slope: f64,
    pub line_residual_pixels: f64,
    /// Uncalibrated quality of narrow-row, free-edge and head support.
    pub score: f64,
}
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct StemDetection {
    pub stems: Vec<StemObservation>,
    /// Probe hypotheses across direction trials, before deduplication.
    pub probed_lines: usize,
    /// Rejected hypotheses, not unique physical objects.
    pub rejected_as_barline: usize,
    /// Hypotheses with a shaft but insufficient head evidence.
    pub rejected_without_head: usize,
}

impl StemDetection {
    /// Conservative geometry subset. Local agreement is checked on measured
    /// source slopes, never by replacing a shaft with a vertical segment.
    pub fn vertical_evidence(&self, data: &Observations) -> Vec<crate::geometry::VerticalPath> {
        if data.validate().is_err() {
            return vec![];
        }
        self.stems
            .iter()
            .filter_map(|stem| {
                let staff = data.staves.get(stem.staff)?;
                let middle = staff.sample(0.5);
                let space = (middle[4].y - middle[0].y) / 4.;
                if !stem.geometry_usable
                    || stem.heads.iter().all(|h| h.score < 0.78)
                    || stem.points.len() < 3
                    || stem.line_residual_pixels > 0.05 * space
                {
                    return None;
                }
                let x = stem.points[0].x;
                let neighbors: Vec<_> = self
                    .stems
                    .iter()
                    .filter(|s| {
                        s.staff == stem.staff
                            && s.geometry_usable
                            && s.points.len() >= 3
                            && (s.points[0].x - x).abs() < 20. * space
                            && (s.points[0].x - x).abs() > 0.5 * space
                    })
                    .map(|s| s.slope)
                    .collect();
                if neighbors.len() < 2 {
                    return None;
                }
                let direction = crate::math::median(neighbors);
                if (stem.slope - direction).abs() > 0.08 {
                    return None;
                }
                Some(crate::geometry::VerticalPath {
                    points: stem.points.clone(),
                    weight: 0.8,
                })
            })
            .collect()
    }
}

fn measure(view: &View<'_>, line: &probe::Line, space: f64) -> Vec<Point> {
    let mut points = vec![];
    let radius = (space * 0.22).ceil() as i32;
    for y in line.top..line.bottom {
        let p = view.point(f64::from(line.centre), f64::from(view.height() - 1 - y));
        let x = p.x.round() as i32;
        let y = p.y.round() as i32;
        let seed = (-radius..=radius)
            .filter(|dx| view.pixels.own(view.component.id, x + dx, y))
            .min_by_key(|dx| dx.abs());
        let Some(dx) = seed else {
            continue;
        };
        let mut left = x + dx;
        let mut right = left;
        while view.pixels.own(view.component.id, left - 1, y) && f64::from(right - left) < space {
            left -= 1;
        }
        while view.pixels.own(view.component.id, right + 1, y) && f64::from(right - left) < space {
            right += 1;
        }
        if f64::from(right - left + 1) <= space * 0.4 + 1. {
            points.push(Point {
                x: f64::from(left + right) / 2.,
                y: f64::from(y),
            });
        }
    }
    points
}
fn fit_line(points: &[Point]) -> (f64, f64) {
    let n = points.len() as f64;
    let mx = points.iter().map(|p| p.x).sum::<f64>() / n;
    let my = points.iter().map(|p| p.y).sum::<f64>() / n;
    let var = points.iter().map(|p| (p.y - my).powi(2)).sum::<f64>();
    let slope = points.iter().map(|p| (p.x - mx) * (p.y - my)).sum::<f64>() / var.max(1e-12);
    let rms = (points
        .iter()
        .map(|p| (p.x - mx - slope * (p.y - my)).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    (slope, rms)
}
fn overlaps_bar(points: &[Point], staff: usize, data: &Observations, space: f64) -> bool {
    data.barlines
        .iter()
        .filter(|b| b.staff == staff || b.aligned_staffs.contains(&staff))
        .any(|b| {
            let near = points
                .iter()
                .filter(|p| {
                    if p.y < b.y[0] || p.y > *b.y.last().unwrap() {
                        return false;
                    }
                    let i = b.y.partition_point(|y| *y < p.y).clamp(1, b.y.len() - 1);
                    let t = (p.y - b.y[i - 1]) / (b.y[i] - b.y[i - 1]).max(1e-9);
                    (p.x - (b.x[i - 1] + t * (b.x[i] - b.x[i - 1]))).abs() < 0.25 * space
                })
                .count();
            near as f64 > points.len() as f64 * 0.65
        })
}
/// Verify the shared head cluster and the free shaft on its other side.
/// Uses the complete measured path before excluding heads/rules, so a clef
/// cannot become a stem by dropping all of its inconvenient wide rows.
fn retain_attached_heads(heads: &mut Vec<HeadEvidence>, points: &[Point], slope: f64, space: f64) {
    let first = points[0];
    let last = *points.last().unwrap();
    let all = heads.clone();
    let anchored: Vec<_> = all
        .iter()
        .filter(|h| {
            let shaft_x = first.x + slope * (h.center.y - first.y);
            let left = h.center.x < shaft_x;
            let end_distance = if left {
                last.y - h.center.y
            } else {
                h.center.y - first.y
            };
            // Small glyph fragments on sharps, rests and clefs often match a
            // template inside the shaft rather than at its physical attachment.
            end_distance >= -0.8 * space
                && end_distance <= 0.8 * space
                && (h.center.x - shaft_x).abs() >= 0.18 * space
        })
        .collect();
    heads.retain(|h| {
        anchored.iter().any(|anchor| {
            let shaft_x = first.x + slope * (anchor.center.y - first.y);
            let up = anchor.center.x < shaft_x;
            let free = if up {
                h.center.y - first.y
            } else {
                last.y - h.center.y
            };
            let same_cluster = (h.center.y - anchor.center.y).abs() <= 2.5 * space;
            free >= 1.8 * space && same_cluster
        })
    });
    // Expand a validated end into a chord, but never invent a free shaft by
    // treating alternating clef lobes as a cluster of small heads.
}

fn staff_stems(
    image: &GrayImage,
    data: &Observations,
    index: usize,
    staff: &Staff,
) -> StemDetection {
    let middle = staff.sample(0.5);
    let space = (middle[4].y - middle[0].y) / 4.;
    let mut result = StemDetection::default();
    if space < 3. {
        return result;
    }
    let pixels = Pixels::new(image, staff, space);
    let mut matcher = heads::HeadMatcher::new(&pixels, space);
    let metrics = ported::metrics::NoteMetrics::new(space.round() as i32);
    for component in &pixels.components {
        for shear in [-0.3_f64, -0.2, -0.1, 0., 0.1, 0.2, 0.3] {
            let padding = (shear.abs() * f64::from(component.bounds[3] - component.bounds[1]) / 2.)
                .ceil() as i32
                + 2;
            let view = View::new(&pixels, component, shear, padding);
            let lines = probe::probe(
                &view,
                space.round() as i32,
                (1.6 * space).ceil() as i32,
                (0.42 * space).ceil() as i32 + 1,
                None,
                150,
            );
            for line in lines.lines {
                result.probed_lines += 1;
                let edge_line = ported::types::StemLine {
                    x: line.centre,
                    left: line.left,
                    right: line.right - 1,
                    y: view.height() - line.bottom,
                    len: line.bottom - line.top,
                };
                let narrow = edge::narrow_rows(
                    &view,
                    &metrics,
                    &edge_line,
                    (space * 0.48).floor() as i32 + 3,
                );
                if narrow.total < metrics.space.space {
                    continue;
                }
                let mut points = measure(&view, &line, space);
                if points.len() < 3 || points.last().unwrap().y - points[0].y < 1.6 * space {
                    continue;
                }
                if overlaps_bar(&points, index, data, space) {
                    result.rejected_as_barline += 1;
                    continue;
                }
                let (_, rms) = fit_line(&points);

                if rms > space * 0.16 {
                    continue;
                }
                let (slope, _) = fit_line(&points);
                let mut heads =
                    matcher.find_all(points[0], *points.last().unwrap(), slope, space, staff);
                // A head-shaped patch somewhere beside a line is not a note.
                // Keep only attachments on the head end of a real shaft. Chord
                // heads may occur along that end, including displaced seconds.
                retain_attached_heads(&mut heads, &points, slope, space);
                if heads.is_empty() {
                    result.rejected_without_head += 1;
                    continue;
                }
                // Geometry consumes narrow source-ink rows across gaps, never
                // head/beam intersections or assumed straight vertical pixels.
                points.retain(|p| {
                    !view.pixels.on_rule(p.x.round() as i32, p.y.round() as i32)
                        && heads.iter().all(|h| p.y < h.bounds[1] || p.y > h.bounds[3])
                });
                if points.len() < 3 {
                    continue;
                }
                let (slope, rms) = fit_line(&points);
                // Recheck the actual shaft after removing head/rule samples.
                // A clef/rest can look straight only while its lobes participate
                // in the first fit; its remaining ink is sparse or bent.
                let span = points.last().unwrap().y - points[0].y;
                if rms > 0.06 * space || points.len() as f64 / span.max(1.) < 0.4 {
                    result.rejected_without_head += 1;
                    continue;
                }
                let geometry_usable = span >= 1.5 * space
                    && points.len() as f64 >= 1.2 * space
                    && points.len() as f64 / span.max(1.) >= 0.55
                    && rms <= 0.05 * space
                    && heads.iter().any(|h| h.score >= 0.78);
                // The inherited edge test assumed erased rules. On raw ink,
                // staff/beam interruptions lower confidence instead of hiding
                // a head-supported candidate before head matching can run.
                let edge_supported = edge::stem_edge(
                    &view,
                    &metrics,
                    &edge_line,
                    edge::StemEdgeContext::default(),
                )
                .is_clear();
                let score = heads.iter().map(|h| h.score).fold(0., f64::max)
                    * (f64::from(narrow.total) / f64::from(edge_line.len)).min(1.)
                    / (1. + rms)
                    * if edge_supported { 1. } else { 0.8 };
                result.stems.push(StemObservation {
                    staff: index,
                    points,
                    heads,
                    geometry_usable,
                    slope,
                    line_residual_pixels: rms,
                    score,
                });
            }
        }
    }
    result.stems.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut unique: Vec<StemObservation> = vec![];
    for stem in result.stems.drain(..) {
        let y = stem.points[0].y;
        if let Some(old) = unique.iter_mut().find(|old| {
            let overlap = old
                .points
                .last()
                .unwrap()
                .y
                .min(stem.points.last().unwrap().y)
                - old.points[0].y.max(stem.points[0].y);
            overlap > 0.
                && old.heads.iter().any(|a| {
                    stem.heads.iter().any(|b| {
                        (a.center.x - b.center.x).abs() < 0.5 * space
                            && (a.center.y - b.center.y).abs() < 0.75 * space
                    })
                })
                && ((old.points[0].x + old.slope * (y - old.points[0].y)) - stem.points[0].x).abs()
                    < space * 0.25
        }) {
            for head in stem.heads {
                if old.heads.iter().all(|h| {
                    (h.center.x - head.center.x).abs() >= space * 0.5
                        || (h.center.y - head.center.y).abs() >= space * 0.75
                }) {
                    old.heads.push(head);
                }
            }
        } else {
            unique.push(stem);
        }
    }
    unique.sort_by(|a, b| a.points[0].x.total_cmp(&b.points[0].x));
    result.stems = unique;
    result
}

/// Detect head-supported stem shafts without parsing music semantics.
///
/// Call after page rectification, using matching staff/bar observations. The
/// fixed direction hypotheses cover dx/dy in [-0.3, 0.3]; stronger distortion
/// can cause abstention. Filled/open ordinary heads are supported, specialized
/// notation and ambiguous glyphs remain outside this experimental detector.
/// Head matching observes untouched pixels and treats known staff rules as
/// neutral evidence. `geometry_usable` separates sufficient shaft support
/// from merely recognized head-supported candidates. Missing heads, unusual
/// glyphs and false positives are still possible; this is not full OMR.
/// No pixels, observations or dewarping settings are modified.
pub fn detect_stems(binary: &GrayImage, data: &Observations) -> Result<StemDetection> {
    data.validate()?;
    if binary.dimensions() != (data.width, data.height) {
        return Err(Error::Invalid(
            "stem image dimensions differ from observations",
        ));
    }
    let parts: Vec<_> = data
        .staves
        .par_iter()
        .enumerate()
        .map(|(i, staff)| staff_stems(binary, data, i, staff))
        .collect();
    let mut result = StemDetection::default();
    for part in parts {
        result.probed_lines += part.probed_lines;
        result.rejected_as_barline += part.rejected_as_barline;
        result.rejected_without_head += part.rejected_without_head;
        result.stems.extend(part.stems);
    }
    Ok(result)
}
