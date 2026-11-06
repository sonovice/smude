//! Classical vertical strokes and cross-staff connectivity.
use super::*;

struct Candidate {
    bar: Barline,
    width: f64,
    clean: f64,
    extension: f64,
}
fn stroke(
    ink: &Ink,
    staff: usize,
    top: f64,
    bottom: f64,
    center: f64,
    proposal: f64,
) -> Option<Candidate> {
    let space = (bottom - top) / 4.;
    let mid = (top + bottom) / 2.;
    let reach = (space * 0.65).round().max(3.) as i64;
    let mut points = vec![];
    let mut widths = vec![];
    let mut eligible = 0;
    for yy in top.round() as i64..=bottom.round() as i64 {
        let y = yy as f64;
        let phase = (y - top) / space;
        if
        // The antialiased horizontal attachment can extend almost two
        // pixels from its detected centre at an eight-pixel staff spacing.
        (phase - phase.round()).abs() <= 0.22 {
            continue;
        }
        eligible += 1;
        let predicted = (center + proposal * (y - mid)).round();
        let mut best: Option<(f64, f64)> = None;
        let mut left = -reach;
        while left <= reach {
            if ink.at(predicted + left as f64, y) == 0. {
                left += 1;
                continue;
            }
            let mut right = left;
            while right < reach && ink.at(predicted + (right + 1) as f64, y) > 0. {
                right += 1;
            }
            let width = (right - left + 1) as f64;
            let c = (left + right) as f64 / 2.;
            if left > -reach
                && right < reach
                && width <= 3_f64.max(space * 0.5)
                && best.is_none_or(|(old, _)| c.abs() < old.abs())
            {
                best = Some((c, width));
            }
            left = right + 1;
        }
        if let Some((c, width)) = best {
            points.push(Point {
                x: predicted + c,
                y,
            });
            widths.push(width);
        }
    }
    if points.len() < 8 || points.len() as f64 / (eligible as f64) < 0.9 {
        return None;
    }
    let mut fit_slope = slope(&points);
    let mut intercept = points
        .iter()
        .map(|p| p.x - fit_slope * (p.y - mid))
        .sum::<f64>()
        / points.len() as f64;
    for _ in 0..3 {
        let residuals: Vec<f64> = points
            .iter()
            .map(|p| (p.x - intercept - fit_slope * (p.y - mid)).abs())
            .collect();
        let limit = 1_f64.max(3. * median(residuals));
        let kept: Vec<Point> = points
            .iter()
            .copied()
            .filter(|p| (p.x - intercept - fit_slope * (p.y - mid)).abs() <= limit)
            .collect();
        if kept.len() < 8 {
            return None;
        }
        fit_slope = slope(&kept);
        intercept = kept
            .iter()
            .map(|p| p.x - fit_slope * (p.y - mid))
            .sum::<f64>()
            / kept.len() as f64;
    }
    if fit_slope.abs() > 0.3 {
        return None;
    }
    // Count broad attachments separately; narrow-run sampling must not hide heads.
    let mut clean = 0;
    for p in &points {
        let x = intercept + fit_slope * (p.y - mid);
        let mut a = x.round() as i64;
        let mut b = a;
        if ink.at(a as f64, p.y) == 0. {
            continue;
        }
        while a > 0 && ink.at((a - 1) as f64, p.y) > 0. && x - (a as f64) < space {
            a -= 1;
        }
        while b + 1 < ink.w as i64 && ink.at((b + 1) as f64, p.y) > 0. && b as f64 - x < space {
            b += 1;
        }
        if (b - a + 1) as f64 <= 3_f64.max(space * 0.5) {
            clean += 1;
        }
    }
    let clean = clean as f64 / eligible as f64;
    if clean < 0.82 {
        return None;
    }
    let mut extension = 0_f64;
    for (end, dir) in [(top, -1.), (bottom, 1.)] {
        let count = (0..12)
            .filter(|j| {
                let y = end + dir * (0.3 + 0.55 * (*j as f64) / 11.) * space;
                ink.nearby(intercept + fit_slope * (y - mid), y) > 0.
            })
            .count();
        extension = extension.max(count as f64 / 12.);
    }
    let y: Vec<f64> = (top.round() as i64..=bottom.round() as i64)
        .map(|v| v as f64)
        .collect();
    Some(Candidate {
        bar: Barline {
            staff,
            x: y.iter()
                .map(|v| intercept + fit_slope * (v - mid))
                .collect(),
            y,
            aligned_staffs: vec![],
        },
        width: median(widths),
        clean,
        extension,
    })
}

/// Follow a proposed continuation through the gap. Missing ink is tolerated
/// only over short runs; a page barrier is never an admissible occlusion.
fn bridge(ink: &Ink, upper: &Candidate, lower: &Candidate) -> Option<(f64, Vec<Point>)> {
    let a = &upper.bar;
    let b = &lower.bar;
    let space = ((a.y.last()? - a.y[0]) / 4.).min((b.y.last()? - b.y[0]) / 4.);
    let ay = *a.y.last()?;
    let by = b.y[0];
    let gap = by - ay;
    if gap <= 0. || gap > 12. * space {
        return None;
    }
    let ax = *a.x.last()?;
    let bx = b.x[0];
    let sa = (ax - a.x[0]) / (ay - a.y[0]);
    let sb = (*b.x.last()? - bx) / (b.y.last()? - by);
    let error = ((ax + sa * gap - bx).abs() + (bx - sb * gap - ax).abs()) / 2.;
    if error > 0.9 * space || ((bx - ax) / gap).abs() > 0.35 {
        return None;
    }
    let reach = (space * 0.35).ceil().max(2.) as i64;
    let mut found = Vec::new();
    let mut missing = 0;
    let mut longest_missing = 0;
    let mut support = 0;
    for yy in ay as i64 + 1..by as i64 {
        let y = yy as f64;
        let predicted = ax + (bx - ax) * (y - ay) / gap;
        if !ink.permits(predicted, y) {
            return None;
        }
        let mut best: Option<f64> = None;
        let center = predicted.round() as i64;
        let mut left = center - reach;
        while left <= center + reach {
            if ink.at(left as f64, y) == 0. {
                left += 1;
                continue;
            }
            let mut right = left;
            while right < ink.w as i64 - 1 && ink.at((right + 1) as f64, y) > 0. {
                right += 1;
                if right - left > (space * 0.6).ceil().max(3.) as i64 {
                    break;
                }
            }
            let x = (left + right) as f64 / 2.;
            if right - left < (space * 0.6).ceil().max(3.) as i64
                && (x - predicted).abs() <= reach as f64
                && best.is_none_or(|old| (x - predicted).abs() < (old - predicted).abs())
            {
                best = Some(x);
            }
            left = right + 1;
        }
        if let Some(x) = best {
            support += 1;
            missing = 0;
            found.push(Point { x, y });
        } else {
            missing += 1;
            longest_missing = longest_missing.max(missing);
            found.push(Point { x: f64::NAN, y });
        }
    }
    if found.is_empty()
        || support as f64 / (found.len() as f64) < 0.86
        || longest_missing as f64 > (space * 0.6).max(2.)
    {
        return None;
    }
    // Preserve measured curvature across the whitespace. Interpolate only
    // short unsupported runs between their nearest measured neighbours.
    let measured = found.clone();
    for (i, p) in found.iter_mut().enumerate() {
        if p.x.is_nan() {
            let left = measured[..i]
                .iter()
                .rev()
                .find(|p| p.x.is_finite())
                .copied()
                .unwrap_or(Point { x: ax, y: ay });
            let right = measured[i + 1..]
                .iter()
                .find(|p| p.x.is_finite())
                .copied()
                .unwrap_or(Point { x: bx, y: by });
            p.x = left.x + (right.x - left.x) * (p.y - left.y) / (right.y - left.y);
        }
    }
    let interpolated = found.clone();
    for (i, p) in found.iter_mut().enumerate() {
        let start = i.saturating_sub(1);
        let end = (i + 2).min(interpolated.len());
        p.x = median(interpolated[start..end].iter().map(|p| p.x).collect());
        if !ink.permits(p.x, p.y) {
            return None;
        }
    }
    Some((error, found))
}

pub(super) fn bars(ink: &Ink, staves: &[Staff], sx: f64, sy: f64) -> Vec<Barline> {
    let mut candidates: Vec<Candidate> = vec![];
    for (owner, staff) in staves.iter().enumerate() {
        let mut scores = vec![];
        // Endpoints can stop short of a terminal stroke when a clef or thick bar
        // occludes horizontal evidence. Search only a small staff-scaled margin.
        let space = (staff.y[4][0] - staff.y[0][0]) * sy / 4.;
        let margin = 1.75 * space;
        let left = (staff.x[0] * sx - margin).round().max(0.) as usize;
        let right = (staff.x[staff.x.len() - 1] * sx + margin)
            .round()
            .min((ink.w - 1) as f64) as usize;
        for x in left..=right {
            let lines: [f64; 5] = std::array::from_fn(|k| {
                crate::observations::interpolate(&staff.x, &staff.y[k], x as f64 / sx) * sy
            });
            let mid = (lines[0] + lines[4]) / 2.;
            let mut best = (0_f64, 0_f64);
            for j in -8..=8 {
                let s = j as f64 * 0.025;
                let mut all = 0_f64;
                let mut minimum = 1_f64;
                for k in 0..4 {
                    let sum = (0..12)
                        .map(|i| {
                            let t = 0.2 + 0.6 * i as f64 / 11.;
                            let y = lines[k] * (1. - t) + lines[k + 1] * t;
                            ink.nearby(x as f64 + s * (y - mid), y)
                        })
                        .sum::<f64>()
                        / 12.;
                    minimum = minimum.min(sum);
                    all += sum / 4.;
                }
                if minimum >= 0.8 && (all > best.0 || (all == best.0 && s.abs() < best.1.abs())) {
                    best = (all, s);
                }
            }
            scores.push((best, lines[0], lines[4]));
        }
        for i in 0..scores.len() {
            if scores[i].0.0 < 0.92
                || (i > 0 && scores[i].0.0 < scores[i - 1].0.0)
                // Every tied peak must reach the stroke fit: choosing only
                // the right edge of a broad plateau loses thin double bars
                // and may pick a proposal whose pixel support is worse.
                || (i + 1 < scores.len() && scores[i].0.0 < scores[i + 1].0.0)
            {
                continue;
            }
            let ((_, proposal), top, bottom) = scores[i];
            if let Some(candidate) = stroke(ink, owner, top, bottom, (left + i) as f64, proposal) {
                let center = median(candidate.bar.x.clone());
                if candidates
                    .iter()
                    .filter(|c| c.bar.staff == owner)
                    .all(|old| {
                        (center - median(old.bar.x.clone())).abs()
                            > candidate.width.max(old.width) * 0.5 + 0.5
                    })
                {
                    candidates.push(candidate);
                }
            }
        }
    }
    // Alignment alone is not connectivity: stems on adjacent staves can have
    // the same x. Confirm narrow ink in the entire intervening whitespace.
    let mut links = vec![];
    for (i, upper) in candidates.iter().enumerate() {
        for (j, lower) in candidates.iter().enumerate() {
            if lower.bar.staff != upper.bar.staff + 1 {
                continue;
            }
            if let Some((cost, bridge)) = bridge(ink, upper, lower) {
                links.push((cost, i, j, bridge));
            }
        }
    }
    links.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut next = vec![None; candidates.len()];
    let mut previous = vec![None; candidates.len()];
    let mut bridges = vec![Vec::new(); candidates.len()];
    for (_, i, j, bridge) in links {
        // One-to-one continuation preserves the two physical strokes of a
        // double bar, rather than joining all nearby detections into one.
        if next[i].is_none() && previous[j].is_none() {
            next[i] = Some(j);
            previous[j] = Some(i);
            bridges[i] = bridge;
        }
    }
    let mut out = vec![];
    for (i, c) in candidates.iter().enumerate() {
        if previous[i].is_some() {
            continue;
        }
        if next[i].is_none() && (c.extension > 0.7 || c.clean < 0.95) {
            continue;
        }
        let mut b = c.bar.clone();
        let mut current = i;
        while let Some(j) = next[current] {
            for p in &bridges[current] {
                b.x.push(p.x);
                b.y.push(p.y);
            }
            b.x.extend_from_slice(&candidates[j].bar.x);
            b.y.extend_from_slice(&candidates[j].bar.y);
            b.aligned_staffs.push(candidates[j].bar.staff);
            current = j;
        }
        if b.x.iter().zip(&b.y).any(|(x, y)| !ink.permits(*x, *y)) {
            continue;
        }
        b.x.iter_mut().for_each(|v| *v /= sx);
        b.y.iter_mut().for_each(|v| *v /= sy);
        out.push(b);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(centers: &[f64]) -> (Ink, Vec<Staff>) {
        let mut ink = Ink {
            w: 260,
            h: 300,
            pixels: vec![0.; 260 * 300],
            seed_paper: vec![true; 260 * 300],
            allowed: None,
        };
        let staves = centers
            .iter()
            .map(|center| Staff {
                x: vec![32., 215.],
                y: std::array::from_fn(|k| vec![center + (k as f64 - 2.) * 8.; 2]),
            })
            .collect::<Vec<_>>();
        for s in &staves {
            for line in &s.y {
                for x in 32..=228 {
                    ink.pixels[line[0] as usize * ink.w + x] = 1.;
                }
            }
        }
        (ink, staves)
    }
    fn vertical(ink: &mut Ink, x: usize, top: usize, bottom: usize, width: usize) {
        for y in top..=bottom {
            for xx in x..x + width {
                ink.pixels[y * ink.w + xx] = 1.;
            }
        }
    }
    #[test]
    fn full_three_staff_stroke_retains_gap_ink_without_duplicate_segments() {
        let (mut ink, staves) = fixture(&[60., 130., 200.]);
        vertical(&mut ink, 90, 44, 216, 2);
        let bars = bars(&ink, &staves, 1., 1.);
        let full = bars
            .iter()
            .filter(|b| b.y[0] <= 44. && *b.y.last().unwrap() >= 216.)
            .collect::<Vec<_>>();
        assert_eq!(full.len(), 1);
        assert_eq!(full[0].aligned_staffs, vec![1, 2]);
        assert_eq!(
            bars.iter()
                .filter(|b| (median(b.x.clone()) - 90.5).abs() < 2.)
                .count(),
            1
        );
    }
    #[test]
    fn aligned_disconnected_extended_stems_are_not_system_barlines() {
        let (mut ink, staves) = fixture(&[60., 130.]);
        vertical(&mut ink, 100, 35, 85, 2);
        vertical(&mut ink, 100, 105, 155, 2);
        assert!(bars(&ink, &staves, 1., 1.).is_empty());
    }
    #[test]
    fn system_initial_and_terminal_double_bars_are_found_at_staff_edges() {
        let (mut ink, staves) = fixture(&[60., 130.]);
        vertical(&mut ink, 32, 44, 146, 2);
        vertical(&mut ink, 222, 44, 146, 1);
        vertical(&mut ink, 226, 44, 146, 3);
        let bars = bars(&ink, &staves, 1., 1.);
        for x in [32.5, 222., 227.] {
            assert!(
                bars.iter().any(|b| (median(b.x.clone()) - x).abs() < 1.5
                    && b.y[0] <= 44.
                    && *b.y.last().unwrap() >= 146.),
                "missing edgebar{x}: {bars:?}"
            );
        }
    }
    #[test]
    fn curved_gap_keeps_its_observed_shape_and_tolerates_short_occlusion() {
        let (mut ink, staves) = fixture(&[60., 130.]);
        vertical(&mut ink, 90, 44, 76, 2);
        vertical(&mut ink, 90, 114, 146, 2);
        for y in 77..114 {
            if (94..=95).contains(&y) {
                continue;
            }
            let x =
                90 + (2. * (std::f64::consts::PI * (y - 76) as f64 / 38.).sin()).round() as usize;
            vertical(&mut ink, x, y, y, 2);
        }
        let result = bars(&ink, &staves, 1., 1.);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].aligned_staffs, vec![1]);
        let middle = result[0].y.iter().position(|y| *y == 94.).unwrap();
        assert!(
            result[0].x[middle] > 91.5,
            "measured gap curvature was flattened"
        );
    }

    #[test]
    fn tied_score_plateau_retains_both_close_parallel_strokes() {
        let (mut ink, staves) = fixture(&[60., 130.]);
        vertical(&mut ink, 90, 44, 146, 2);
        vertical(&mut ink, 94, 44, 146, 2);
        let result = bars(&ink, &staves, 1., 1.);
        for target in [90.5, 94.5] {
            assert!(
                result
                    .iter()
                    .any(|b| (median(b.x.clone()) - target).abs() < 0.75
                        && b.aligned_staffs == vec![1]),
                "missing stroke {target}"
            );
        }
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn staff_intersection_band_does_not_count_horizontal_attachments_as_missing_bar() {
        let (mut ink, _) = fixture(&[]);
        vertical(&mut ink, 90, 40, 80, 2);
        // Simulate the outer edge of a printed staff attachment two pixels
        // away from its detected centre, as seen in source-resolution crops.
        for y in [38, 48, 58, 68, 78] {
            for x in 32..=215 {
                ink.pixels[y * ink.w + x] = 1.;
            }
        }
        assert!(stroke(&ink, 0, 40., 80., 90.5, 0.).is_some());
    }

    #[test]
    fn confirmed_page_barrier_cannot_be_crossed_in_the_gap() {
        let (mut ink, staves) = fixture(&[60., 130.]);
        vertical(&mut ink, 90, 44, 146, 2);
        let mut allowed = vec![true; ink.w * ink.h];
        for y in 93..98 {
            allowed[y * ink.w..(y + 1) * ink.w].fill(false);
        }
        ink.allowed = Some(allowed);
        assert!(
            !bars(&ink, &staves, 1., 1.)
                .iter()
                .any(|b| b.y[0] < 93. && *b.y.last().unwrap() > 98.)
        );
    }
}
