//! Staff rules end at independently observed ink transitions. These are weak
//! geometric boundaries, never musical barlines or synthesized edge strokes.
use super::Ink;
use crate::geometry::observations::{StaffEdge, StaffEdgeSide};
use crate::geometry::{Observations, Point, Staff};
use crate::math::median;
use rayon::prelude::*;

fn course(staff: &Staff, k: usize, x: f64, sx: f64, sy: f64, d: f64) -> f64 {
    let lo = staff.x[0];
    let hi = *staff.x.last().unwrap();
    let at = |x| crate::geometry::observations::interpolate(&staff.x, &staff.y[k], x);
    let y = if x / sx < lo {
        let b = (lo + 3. * d / sx).min(hi);
        at(lo) + (x / sx - lo) * (at(b) - at(lo)) / (b - lo)
    } else if x / sx > hi {
        let a = (hi - 3. * d / sx).max(lo);
        at(hi) + (x / sx - hi) * (at(hi) - at(a)) / (hi - a)
    } else {
        at(x / sx)
    };
    y * sy
}

fn ridge(ink: &Ink, x: f64, y: f64, d: f64) -> (bool, bool) {
    let Some(dy) = (-1_i32..=1)
        .filter(|dy| ink.at(x, y + f64::from(*dy)) > 0.)
        .min_by_key(|dy| dy.abs())
    else {
        return (false, false);
    };
    let row = y.round() as i64 + i64::from(dy);
    let mut top = row;
    let mut bottom = row;
    let limit = d.ceil() as i64;
    while row - top < limit && ink.at(x, (top - 1) as f64) > 0. {
        top -= 1;
    }
    while bottom - row < limit && ink.at(x, (bottom + 1) as f64) > 0. {
        bottom += 1;
    }
    (true, (bottom - top + 1) as f64 <= (0.35 * d).ceil().max(2.))
}

fn line_end(
    ink: &Ink,
    staff: &Staff,
    k: usize,
    side: StaffEdgeSide,
    sx: f64,
    sy: f64,
    d: f64,
) -> Option<(Point, f64)> {
    let inward = if side == StaffEdgeSide::Start {
        1.
    } else {
        -1.
    };
    let reference = if inward > 0. {
        staff.x[0] * sx
    } else {
        *staff.x.last().unwrap() * sx
    };
    let radius = 8. * d;
    let inside = (6. * d).ceil().max(16.) as usize;
    let outside = (1.5 * d).ceil().max(4.) as usize;
    let mut best: Option<(f64, Point, f64)> = None;
    for xx in (reference - radius).floor().max(0.) as usize
        ..=(reference + radius).ceil().min((ink.w - 1) as f64) as usize
    {
        let x = xx as f64;
        let y = course(staff, k, x, sx, sy, d);
        // Endpoints require actual paper on both sides. An image crop is not
        // an ink-to-paper transition and cannot provide a geometric boundary.
        if x - (outside as f64) < 2. || x + (outside as f64) > (ink.w - 3) as f64 {
            continue;
        }
        let mut present = 0;
        let mut entry_present = 0;
        let entry = (d.ceil() as usize).max(4);
        let mut longest_clean = 0_usize;
        let mut clean_run = 0_usize;
        let mut clean_gap = 0_usize;
        let mut clear = 0;
        let mut permitted = true;
        for j in 0..inside {
            let at = x + inward * j as f64;
            let yy = course(staff, k, at, sx, sy, d);
            if !ink.permits(at, yy) {
                permitted = false;
                break;
            }
            let (has, _) = ridge(ink, at, yy, d);
            present += usize::from(has);
            if j < entry {
                entry_present += usize::from(has);
            }
        }
        // Clefs and key signatures attach broad ink to real staff rules.
        // Look farther inward for a sustained clean rule instead of requiring
        // most of the immediate entry window to be narrow.
        for j in 0..(12. * d).ceil() as usize {
            let at = x + inward * j as f64;
            let yy = course(staff, k, at, sx, sy, d);
            if !ink.permits(at, yy) {
                break;
            }
            if ridge(ink, at, yy, d).1 {
                clean_run += if clean_run > 0 { clean_gap + 1 } else { 1 };
                clean_gap = 0;
                longest_clean = longest_clean.max(clean_run);
            } else {
                clean_gap += 1;
                if clean_gap > 1 {
                    clean_run = 0;
                    clean_gap = 0;
                }
            }
        }
        for j in 1..=outside {
            let at = x - inward * j as f64;
            let yy = course(staff, k, at, sx, sy, d);
            if !ink.permits(at, yy) {
                permitted = false;
                break;
            }
            clear += usize::from(!ridge(ink, at, yy, d).0);
        }
        if !permitted || !ridge(ink, x, y, d).0 {
            continue;
        }
        let coverage = present as f64 / inside as f64;
        let clean = (longest_clean as f64 / (3. * d)).min(1.);
        let blank = clear as f64 / outside as f64;
        // A brace can supply an isolated leading pixel beside the real rule.
        // The first full staff space must already be contiguous rule ink.
        if coverage < 0.9
            || (longest_clean as f64) < 2. * d
            || blank < 0.7
            || entry_present + 1 < entry
        {
            continue;
        }
        // Select the actual transition pixel, not a location one or two pixels
        // inside an antialiased stroke that also passes the aggregate checks.
        if ridge(ink, x - inward, course(staff, k, x - inward, sx, sy, d), d).0 {
            continue;
        }
        let confidence = 0.4 * coverage + 0.3 * clean + 0.3 * blank;
        let score = confidence - 0.03 * (x - reference).abs() / radius;
        if best.as_ref().is_none_or(|b| score > b.0) {
            best = Some((
                score,
                Point {
                    x: x / sx,
                    y: y / sy,
                },
                confidence,
            ));
        }
    }
    best.map(|(_, p, c)| (p, c))
}

fn coherent(points: &[(usize, Point, f64)], spacing: f64) -> Vec<(usize, Point, f64)> {
    let mut best = vec![];
    for a in points {
        for b in points {
            if a.0 >= b.0 || (b.1.y - a.1.y).abs() < 1e-6 {
                continue;
            }
            let slope = (b.1.x - a.1.x) / (b.1.y - a.1.y);
            let inliers: Vec<_> = points
                .iter()
                .copied()
                .filter(|p| {
                    (p.1.x - a.1.x - slope * (p.1.y - a.1.y)).abs() <= 1_f64.max(0.18 * spacing)
                })
                .collect();
            if inliers.len() > best.len() {
                best = inliers;
            }
        }
    }
    best
}

fn edge_line(edge: &StaffEdge) -> (Point, f64) {
    let y = median(edge.points.iter().map(|p| p.y).collect());
    let x = median(edge.points.iter().map(|p| p.x).collect());
    let slopes: Vec<_> = edge
        .points
        .iter()
        .enumerate()
        .flat_map(|(i, a)| {
            edge.points.iter().skip(i + 1).filter_map(move |b| {
                ((b.y - a.y).abs() > 1e-6).then_some((b.x - a.x) / (b.y - a.y))
            })
        })
        .collect();
    (Point { x, y }, median(slopes))
}

pub(super) fn detect(ink: &Ink, data: &Observations, sx: f64, sy: f64) -> Vec<StaffEdge> {
    let per_staff: Vec<Vec<StaffEdge>> = data
        .staves
        .par_iter()
        .enumerate()
        .map(|(i, staff)| {
            let mut edges = vec![];
            for side in [StaffEdgeSide::Start, StaffEdgeSide::End] {
                let endpoint = if side == StaffEdgeSide::Start {
                    0
                } else {
                    staff.x.len() - 1
                };
                let d = (staff.y[4][endpoint] - staff.y[0][endpoint]) * sy / 4.;
                if d < 2. {
                    continue;
                }
                let points: Vec<_> = (0..5)
                    .filter_map(|k| {
                        line_end(ink, staff, k, side, sx, sy, d).map(|(p, c)| (k, p, c))
                    })
                    .collect();
                let points = coherent(&points, d / sy);
                // Three adjacent lines provide only half the staff's lever arm:
                // a brace foot can mimic their endpoints. Require observations
                // spanning most of the original five-line band.
                if points.len() < 3 || points.last().unwrap().0 - points[0].0 < 3 {
                    continue;
                }
                let confidence = points.iter().map(|p| p.2).sum::<f64>() / points.len() as f64
                    * (points.len() as f64 / 5.).sqrt();
                edges.push(StaffEdge {
                    staff: i,
                    side,
                    points: points.iter().map(|p| p.1).collect(),
                    line_indices: points.iter().map(|p| p.0).collect(),
                    confidence,
                    aligned_staffs: vec![],
                });
            }
            edges
        })
        .collect();
    let mut edges: Vec<_> = per_staff.into_iter().flatten().collect();
    // Alignment is a geometric relation requiring BOTH observed boundaries.
    // Shared end x alone would join indented examples or unrelated columns.
    for i in 0..data.staves.len() {
        for j in i + 1..data.staves.len() {
            let a = &data.staves[i];
            let b = &data.staves[j];
            let overlap = (*a.x.last().unwrap()).min(*b.x.last().unwrap()) - a.x[0].max(b.x[0]);
            let short = (a.x.last().unwrap() - a.x[0]).min(b.x.last().unwrap() - b.x[0]);
            if overlap < 0.7 * short {
                continue;
            }
            let spacing = ((median(a.y[4].clone()) - median(a.y[0].clone()))
                .min(median(b.y[4].clone()) - median(b.y[0].clone())))
                / 4.;
            let mut pairs = vec![];
            for side in [StaffEdgeSide::Start, StaffEdgeSide::End] {
                let aa = edges.iter().position(|e| e.staff == i && e.side == side);
                let bb = edges.iter().position(|e| e.staff == j && e.side == side);
                let (Some(aa), Some(bb)) = (aa, bb) else {
                    continue;
                };
                let (p, m) = edge_line(&edges[aa]);
                let (q, n) = edge_line(&edges[bb]);
                if (p.y - q.y).abs() > 20. * spacing
                    || (m - n).abs() > 0.2
                    || (q.x - p.x - m * (q.y - p.y)).abs() > 0.75 * spacing
                    || (p.x - q.x - n * (p.y - q.y)).abs() > 0.75 * spacing
                {
                    continue;
                }
                pairs.push((aa, bb));
            }
            let printed_connection = data.barlines.iter().any(|bar| {
                let contains = |staff| bar.staff == staff || bar.aligned_staffs.contains(&staff);
                contains(i) && contains(j)
            });
            if pairs.len() == 2 || printed_connection {
                for (aa, bb) in pairs {
                    edges[aa].aligned_staffs.push(j);
                    edges[bb].aligned_staffs.push(i);
                }
            }
        }
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(start: [usize; 5], end: [usize; 5]) -> (Ink, Observations) {
        let mut ink = Ink {
            w: 500,
            h: 280,
            pixels: vec![0.; 500 * 280],
            seed_paper: vec![true; 500 * 280],
            allowed: None,
        };
        for k in 0..5 {
            for x in start[k]..=end[k] {
                ink.pixels[(100 + 10 * k) * 500 + x] = 1.;
            }
        }
        let x: Vec<_> = (130..=350).map(f64::from).collect();
        let staff = Staff {
            x: x.clone(),
            y: std::array::from_fn(|k| vec![100. + 10. * k as f64; x.len()]),
        };
        (
            ink,
            Observations {
                width: 500,
                height: 280,
                staves: vec![staff],
                barlines: vec![],
            },
        )
    }
    #[test]
    fn independently_slanted_endpoints_do_not_repeat_shared_curve_x() {
        let start = [100, 103, 106, 109, 112];
        let end = [390, 388, 386, 384, 382];
        let (ink, data) = fixture(start, end);
        let edges = detect(&ink, &data, 1., 1.);
        assert_eq!(edges.len(), 2);
        for edge in edges {
            assert_eq!(edge.line_indices, vec![0, 1, 2, 3, 4]);
            for (k, p) in edge.line_indices.iter().zip(&edge.points) {
                let expected = if edge.side == StaffEdgeSide::Start {
                    start[*k]
                } else {
                    end[*k]
                };
                assert!((p.x - expected as f64).abs() < 0.1);
            }
        }
    }
    #[test]
    fn attached_clef_ink_can_be_broad_before_clean_rule_resumes() {
        let (mut ink, data) = fixture([100; 5], [390; 5]);
        for x in 105..=145 {
            for y in 97..=143 {
                ink.pixels[y * 500 + x] = 1.;
            }
        }
        let edges = detect(&ink, &data, 1., 1.);
        let start = edges
            .iter()
            .find(|e| e.side == StaffEdgeSide::Start)
            .expect("clean inward rule survives a broad attached clef");
        assert_eq!(start.points.len(), 5);
        assert!(start.points.iter().all(|p| p.x == 100.));
    }
    #[test]
    fn isolated_brace_ink_before_the_rule_is_not_an_endpoint() {
        let (mut ink, data) = fixture([100; 5], [390; 5]);
        for k in 0..5 {
            ink.pixels[(100 + 10 * k) * 500 + 90] = 1.;
        }
        let edges = detect(&ink, &data, 1., 1.);
        let start = edges
            .iter()
            .find(|e| e.side == StaffEdgeSide::Start)
            .unwrap();
        assert!(start.points.iter().all(|p| p.x == 100.));
    }
    #[test]
    fn page_mask_can_extend_outside_the_analysis_canvas_without_clamping() {
        let (ink, data) = fixture([100; 5], [390; 5]);
        let mut image = image::RgbImage::from_pixel(500, 280, image::Rgb([255; 3]));
        for y in 0..280 {
            for x in 0..500 {
                if ink.pixels[y * 500 + x] > 0. {
                    image.put_pixel(x as u32, y as u32, image::Rgb([0; 3]));
                }
            }
        }
        let region = crate::detection::PageRegion {
            id: "test".into(),
            confidence: 1.,
            evidence: vec![],
            contour: vec![
                Point { x: -20., y: -10. },
                Point { x: 520., y: -10. },
                Point { x: 520., y: 300. },
                Point { x: -20., y: 300. },
            ],
        };
        let edges =
            super::super::detect_staff_edges_with_region_mask(&image, &data, 500, Some(&region))
                .unwrap();
        assert_eq!(edges.len(), 2);
        assert_eq!(region.contour[0].x, -20.);
    }
    #[test]
    fn raster_crop_does_not_become_an_edge() {
        let (ink, mut data) = fixture([0; 5], [499; 5]);
        let x: Vec<_> = (30..=470).map(f64::from).collect();
        data.staves[0] = Staff {
            x: x.clone(),
            y: std::array::from_fn(|k| vec![100. + 10. * k as f64; x.len()]),
        };
        assert!(detect(&ink, &data, 1., 1.).is_empty());
    }
    #[test]
    fn three_adjacent_line_ends_do_not_establish_a_whole_staff_edge() {
        let (mut ink, data) = fixture([100; 5], [390; 5]);
        for k in [3, 4] {
            for x in 0..500 {
                ink.pixels[(100 + 10 * k) * 500 + x] = 1.;
            }
        }
        assert!(
            detect(&ink, &data, 1., 1.)
                .iter()
                .all(|e| e.side != StaffEdgeSide::Start)
        );
    }
    #[test]
    fn confirmed_mask_edge_is_not_an_ink_transition() {
        let (mut ink, data) = fixture([100; 5], [390; 5]);
        let mut mask = vec![true; ink.w * ink.h];
        for y in 0..ink.h {
            for x in 0..100 {
                mask[y * ink.w + x] = false;
            }
        }
        ink.allowed = Some(mask);
        assert!(
            detect(&ink, &data, 1., 1.)
                .iter()
                .all(|e| e.side != StaffEdgeSide::Start)
        );
    }
    #[test]
    fn confirmed_printed_connection_can_support_a_single_visible_side() {
        let (mut ink, mut data) = fixture([100; 5], [390; 5]);
        let mut lower = data.staves[0].clone();
        for line in &mut lower.y {
            for y in line {
                *y += 80.;
            }
        }
        data.staves.push(lower);
        for k in 0..5 {
            for x in 100..500 {
                ink.pixels[(180 + 10 * k) * 500 + x] = 1.;
            }
        }
        let unlinked = detect(&ink, &data, 1., 1.);
        assert!(unlinked.iter().all(|e| e.aligned_staffs.is_empty()));
        for y in 100..=220 {
            ink.pixels[y * 500 + 200] = 1.;
        }
        data.barlines.push(crate::geometry::Barline {
            staff: 0,
            x: vec![200., 200.],
            y: vec![100., 220.],
            aligned_staffs: vec![1],
        });
        let linked = detect(&ink, &data, 1., 1.);
        let starts: Vec<_> = linked
            .iter()
            .filter(|e| e.side == StaffEdgeSide::Start)
            .collect();
        assert_eq!(starts.len(), 2);
        assert!(starts.iter().all(|e| e.aligned_staffs == vec![1 - e.staff]));
        assert!(
            linked
                .iter()
                .filter(|e| e.side == StaffEdgeSide::End)
                .all(|e| e.aligned_staffs.is_empty())
        );
    }

    #[test]
    fn two_measured_consistent_boundaries_can_align() {
        let (mut ink, mut data) = fixture([100; 5], [390; 5]);
        let mut staff = data.staves[0].clone();
        for line in &mut staff.y {
            for y in line {
                *y += 80.;
            }
        }
        for k in 0..5 {
            for x in 100..=390 {
                ink.pixels[(180 + 10 * k) * 500 + x] = 1.;
            }
        }
        data.staves.push(staff);
        let edges = detect(&ink, &data, 1., 1.);
        assert_eq!(edges.len(), 4);
        assert!(edges.iter().all(|e| e.aligned_staffs == vec![1 - e.staff]));
    }
    #[test]
    fn uncertain_line_indices_are_omitted_without_renumbering() {
        let (mut ink, data) = fixture([100, 103, 106, 109, 112], [390; 5]);
        for k in [1, 3] {
            for x in 0..500 {
                ink.pixels[(100 + 10 * k) * 500 + x] = 1.;
            }
        }
        let edges = detect(&ink, &data, 1., 1.);
        let edge = edges
            .iter()
            .find(|e| e.side == StaffEdgeSide::Start)
            .unwrap();
        assert_eq!(edge.line_indices, vec![0, 2, 4]);
        assert!(edge.confidence < 0.9);
    }
    #[test]
    fn independent_columns_and_indentation_do_not_align() {
        let (mut ink, mut data) = fixture([100; 5], [390; 5]);
        for (start, end, center) in [(125, 390, 180), (270, 450, 230)] {
            let x: Vec<_> = (start + 10..end - 10).map(|v| v as f64).collect();
            let staff = Staff {
                x: x.clone(),
                y: std::array::from_fn(|k| vec![(center + 10 * k) as f64; x.len()]),
            };
            for k in 0..5 {
                for x in start..=end {
                    ink.pixels[(center + 10 * k) * 500 + x] = 1.;
                }
            }
            data.staves.push(staff);
        }
        let edges = detect(&ink, &data, 1., 1.);
        assert!(edges.len() >= 4);
        assert!(edges.iter().all(|e| e.aligned_staffs.is_empty()));
    }
}
