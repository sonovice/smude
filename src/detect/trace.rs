//! Bidirectional local evidence tracking from coarse five-line seeds.
//! Short tangent-aligned probes survive curved gutter regions; white gaps and
//! uniform dark page edges do not provide five separate ink ridges.
use super::{Ink, smooth};
use crate::{Staff, median};

#[derive(Clone, Copy)]
struct State {
    x: f64,
    h: f64,
    d: f64,
    m: f64,
}

fn permitted(ink: &Ink, state: State) -> bool {
    (-2..=2).all(|k| ink.permits(state.x, state.h + f64::from(k) * state.d))
}

fn evidence(ink: &Ink, state: State) -> f64 {
    let mut foreground = 0.;
    let mut strong = 0;
    for k in -2..=2 {
        let sum = [-4., -2., 0., 2., 4.]
            .iter()
            .map(|dx| {
                let y = state.h + f64::from(k) * state.d + state.m * dx;
                [-1., 0., 1.]
                    .iter()
                    .map(|dy| ink.at(state.x + dx, y + dy))
                    .fold(0., f64::max)
            })
            .sum::<f64>()
            / 5.;
        foreground += sum / 5.;
        if sum >= 0.6 {
            strong += 1;
        }
    }
    if strong < 3 || foreground < 0.58 {
        return -1.;
    }
    let mut background = 0_f64;
    for fraction in [1. / 3., 0.5, 2. / 3.] {
        let between = (-2..2)
            .map(|k| {
                [-4., -2., 0., 2., 4.]
                    .iter()
                    .map(|dx| {
                        ink.at(
                            state.x + dx,
                            state.h + (f64::from(k) + fraction) * state.d + state.m * dx,
                        )
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
            / 20.;
        background = background.max(between);
    }
    foreground - 0.8 * background
}

fn correct_phase(ink: &Ink, data: &mut [(f64, [f64; 5])]) -> i32 {
    let score = |phase: i32| {
        let mut foreground = 0.;
        let mut above = 0.;
        let mut below = 0.;
        let mut count = 0.;
        for (x, y) in data.iter().step_by((data.len() / 128).max(1)) {
            let d = (y[4] - y[0]) / 4.;
            let sample = |k: f64| {
                [-1., 0., 1.]
                    .iter()
                    .map(|dy| ink.at(*x, y[2] + (k + f64::from(phase)) * d + dy))
                    .fold(0., f64::max)
            };
            foreground += (-2..=2).map(|k| sample(f64::from(k))).sum::<f64>() / 5.;
            above += sample(-3.);
            below += sample(3.);
            count += 1.;
        }
        (foreground - 0.35 * above.max(below)) / count
    };
    let original = score(0);
    let mut best = 0;
    let mut maximum = original;
    for phase in [-2, -1, 1, 2] {
        let candidate = score(phase);
        if candidate > maximum + 0.05 {
            maximum = candidate;
            best = phase;
        }
    }
    if best != 0 {
        for (_, rows) in data {
            let old = *rows;
            let d = (old[4] - old[0]) / 4.;
            *rows = std::array::from_fn(|k| {
                let index = k as i32 + best;
                if index < 0 {
                    old[0] + f64::from(index) * d
                } else if index > 4 {
                    old[4] + f64::from(index - 4) * d
                } else {
                    old[index as usize]
                }
            });
        }
    }
    best
}

fn fit(ink: &Ink, predicted: State) -> Option<(State, [f64; 5], usize)> {
    let radius = (predicted.d * 0.4).ceil().max(2.) as i32;
    let mut best = predicted;
    let mut score = -1.;
    for dy in -radius..=radius {
        for ratio in [0.96, 1., 1.04] {
            for dm in [-0.2, 0., 0.2] {
                let candidate = State {
                    h: predicted.h + f64::from(dy),
                    d: predicted.d * ratio,
                    m: (predicted.m + dm).clamp(-1.3, 1.3),
                    ..predicted
                };
                if !permitted(ink, candidate) {
                    continue;
                }
                if candidate.h - 2. * candidate.d < 0.
                    || candidate.h + 2. * candidate.d > (ink.h - 1) as f64
                {
                    continue;
                }
                let q = evidence(ink, candidate)
                    - 0.02 * f64::from(dy).abs() / predicted.d
                    - 0.03 * (ratio - 1.).abs()
                    - 0.005 * dm.abs();
                if q > score {
                    score = q;
                    best = candidate;
                }
            }
        }
    }
    if score < 0.4 {
        return None;
    }
    // Refine WHOLE vertical ink-run centres rather than retaining any path
    // that happens to lie inside the stroke. Broad notehead/stem runs abstain.
    let reach = (best.d * 0.35).ceil().max(2.) as i32;
    let mut observations = vec![];
    let mut measured: [Option<f64>; 5] = [None; 5];
    for k in -2..=2 {
        let mut centers = vec![];
        // Span several staff spaces so one notehead cannot dominate all
        // run-centre measurements. Tangent alignment follows real page bends.
        for fraction in [-1.5, -1., -0.5, 0., 0.5, 1., 1.5] {
            let dx = fraction * best.d;
            let expected = best.h + f64::from(k) * best.d + best.m * dx;
            let y = expected.round() as i32;
            let near = (-reach..=reach)
                .filter(|dy| ink.at(best.x + dx, f64::from(y + dy)) > 0.)
                .min_by_key(|dy| dy.abs());
            let Some(offset) = near else {
                continue;
            };
            let mut top = y + offset;
            let mut bottom = top;
            while top > 0
                && expected - f64::from(top) < best.d * 0.6
                && ink.at(best.x + dx, f64::from(top - 1)) > 0.
            {
                top -= 1;
            }
            while bottom + 1 < ink.h as i32
                && f64::from(bottom) - expected < best.d * 0.6
                && ink.at(best.x + dx, f64::from(bottom + 1)) > 0.
            {
                bottom += 1;
            }
            // Ink-run length is an integer pixel count. Round the fractional
            // staff-relative budget upward: a spacing of 8.49 px must not reject
            // ordinary 3 px strokes solely because 0.35*d falls just below 3.
            if f64::from(bottom - top + 1) > (best.d * 0.35).ceil().max(2.) {
                continue;
            }
            let center = f64::from(top + bottom) / 2.;
            centers.push(center - best.m * dx);
        }
        if centers.len() >= 3 {
            let center = median(centers);
            measured[(k + 2) as usize] = Some(center);
            observations.push((f64::from(k), center));
        }
    }
    if observations.len() < 3 {
        return None;
    }
    for _ in 0..3 {
        let mean_k = observations.iter().map(|(k, _)| k).sum::<f64>() / observations.len() as f64;
        let mean_y = observations.iter().map(|(_, y)| y).sum::<f64>() / observations.len() as f64;
        let numerator = observations
            .iter()
            .map(|(k, y)| (k - mean_k) * (y - mean_y))
            .sum::<f64>();
        let denominator = observations
            .iter()
            .map(|(k, _)| (k - mean_k).powi(2))
            .sum::<f64>();
        best.d = numerator / denominator;
        best.h = mean_y - best.d * mean_k;
        observations.retain(|(k, y)| (y - best.h - k * best.d).abs() <= 1.0_f64.max(best.d * 0.2));
        if observations.len() < 3 {
            return None;
        }
    }
    if best.d < predicted.d * 0.85 || best.d > predicted.d * 1.15 {
        return None;
    }
    if best.h - 2. * best.d < 0. || best.h + 2. * best.d > (ink.h - 1) as f64 {
        return None;
    }
    for (k, value) in measured.iter_mut().enumerate() {
        if !observations.iter().any(|(j, _)| *j == k as f64 - 2.) {
            *value = None;
        }
    }
    let lines = std::array::from_fn(|k| measured[k].unwrap_or(best.h + (k as f64 - 2.) * best.d));
    if lines.iter().any(|y| !ink.permits(best.x, *y)) {
        return None;
    }
    if lines.windows(2).any(|w| w[1] - w[0] < best.d * 0.55) {
        return None;
    }
    Some((best, lines, observations.len()))
}

pub(super) fn staff(ink: &Ink, track: &[(usize, [f64; 5])], sx: f64, sy: f64) -> Option<Staff> {
    let mut seed = None;
    let mut best_score = 0.4;
    let stride = (track.len() / 16).max(1);
    for j in (0..track.len()).step_by(stride) {
        let (x, lines) = track[j];
        let a = track[j.saturating_sub(2)];
        let b = track[(j + 2).min(track.len() - 1)];
        let m = if b.0 > a.0 {
            (b.1[2] - a.1[2]) / (b.0 - a.0) as f64
        } else {
            0.
        };
        let state = State {
            x: x as f64,
            h: lines[2],
            d: (lines[4] - lines[0]) / 4.,
            m,
        };
        if let Some((fitted, ys, count)) = fit(ink, state) {
            if count < 4 {
                continue;
            }
            let paper = [-1.5, -0.5, 0.5, 1.5]
                .iter()
                .flat_map(|k| {
                    [-8., 0., 8.].iter().map(move |dx| {
                        let x = (fitted.x + dx).round() as i64;
                        let y = (fitted.h + k * fitted.d + fitted.m * dx).round() as i64;
                        x >= 0
                            && y >= 0
                            && x < ink.w as i64
                            && y < ink.h as i64
                            && ink.seed_paper[y as usize * ink.w + x as usize]
                    })
                })
                .filter(|v| *v)
                .count();
            if paper < 7 {
                continue;
            }
            let score =
                evidence(ink, fitted) - 0.03 * ((j as f64 / track.len() as f64) - 0.5).abs();
            if score > best_score {
                best_score = score;
                seed = Some((fitted, ys));
            }
        }
    }
    let (seed, ys) = seed?;
    let walk = |direction: f64| {
        let mut data = vec![];
        let mut state = seed;
        let mut blank = 0.;
        let mut pending = vec![];
        while state.x >= 4. && state.x < (ink.w - 4) as f64 {
            let x = state.x + direction * 3.;
            if x < 4. || x >= (ink.w - 4) as f64 {
                break;
            }
            let predicted = State {
                x,
                h: state.h + state.m * direction * 3.,
                ..state
            };
            // A confirmed page boundary is a barrier, including during dropout
            // bridging. Check intermediate pixels so a thin slit cannot be
            // crossed by the three-pixel tracking stride.
            if (1..=3).any(|step| {
                !permitted(
                    ink,
                    State {
                        x: state.x + direction * f64::from(step),
                        h: state.h + state.m * direction * f64::from(step),
                        ..state
                    },
                )
            }) {
                break;
            }
            if predicted.h - 2. * predicted.d < 0.
                || predicted.h + 2. * predicted.d > (ink.h - 1) as f64
            {
                break;
            }
            if let Some((mut next, lines, _)) = fit(ink, predicted) {
                if next.d < seed.d * 0.7 || next.d > seed.d * 1.3 {
                    break;
                }
                // A longer history estimates tangent without differentiating
                // one-pixel quantization. Refined local m remains the fallback.
                let previous = data.iter().rev().nth(4).copied().unwrap_or((seed.x, ys));
                if (next.x - previous.0).abs() > 6. {
                    let previous_center = previous.1.iter().sum::<f64>() / 5.;
                    next.m = ((next.h - previous_center) / (next.x - previous.0)).clamp(-1.3, 1.3);
                }
                let last_good = data.last().copied().unwrap_or((seed.x, ys));
                for (x, rows) in &mut pending {
                    let fraction = (*x - last_good.0) / (next.x - last_good.0);
                    *rows = std::array::from_fn(|k| {
                        last_good.1[k] * (1. - fraction) + lines[k] * fraction
                    });
                }
                data.append(&mut pending);
                data.push((next.x, lines));
                state = next;
                blank = 0.;
            } else {
                blank += 3.;
                // Dense notation can hide narrow ridges while still supporting
                // the comb. Blank gutters receive only the short dropout budget.
                let allowance = if evidence(ink, predicted) >= 0. {
                    6.
                } else {
                    3.
                };
                if blank > seed.d * allowance {
                    break;
                }
                pending.push((
                    predicted.x,
                    std::array::from_fn(|k| predicted.h + (k as f64 - 2.) * predicted.d),
                ));
                state = predicted;
            }
        }
        data
    };
    let mut left = walk(-1.);
    left.reverse();
    left.push((seed.x, ys));
    left.extend(walk(1.));
    if left.len() < 8 {
        return None;
    }
    // Four true lines plus a beam can look like another five-line comb. Use
    // evidence over the ENTIRE track to select the actual five-line phase.
    correct_phase(ink, &mut left);
    super::staff_smoothing::regularize(&mut left);
    let first = left[0].0;
    let last = left[left.len() - 1].0;
    if last - first < (ink.w as f64 * 0.05).max(12. * seed.d) {
        return None;
    }
    // A short, self-contained musical example is valid; an equally short
    // fragment cut off by the raster edge is not a complete staff observation.
    if last - first < ink.w as f64 * 0.25
        && (first < 2. * seed.d || last > (ink.w - 1) as f64 - 2. * seed.d)
    {
        return None;
    }
    let xx: Vec<f64> = left.iter().map(|p| p.0).collect();
    let x: Vec<f64> = (first.ceil() as usize..=last.floor() as usize)
        .map(|x| x as f64)
        .collect();
    let y = std::array::from_fn(|k| {
        let yy: Vec<f64> = left.iter().map(|p| p.1[k]).collect();
        let full: Vec<f64> = x
            .iter()
            .map(|x| crate::observations::interpolate(&xx, &yy, *x))
            .collect();
        smooth(&full, 2.).iter().map(|v| v / sy).collect()
    });
    let staff = Staff {
        x: x.iter().map(|v| v / sx).collect(),
        y,
    };
    if staff.x.iter().enumerate().any(|(j, x)| {
        staff
            .y
            .iter()
            .any(|line| !ink.permits(*x * sx, line[j] * sy))
    }) {
        return None;
    }
    // Letter fragments can satisfy the nine-pixel local comb at every step.
    // Real ruled lines also have sustained horizontal support. Check each
    // original line identity independently; this needs no common page-wide
    // staff size and tolerates one-pixel sampling cracks and symbol occlusion.
    let spacing = median(
        staff.y[4]
            .iter()
            .zip(&staff.y[0])
            .map(|(a, b)| (a - b) * sy / 4.)
            .collect(),
    );
    let continuous = staff
        .y
        .iter()
        .filter(|line| {
            let mut length = 0.;
            let mut longest = 0_f64;
            let mut gap = 0;
            for (x, y) in staff.x.iter().zip(line.iter()) {
                let present = [-1., 0., 1.]
                    .iter()
                    .any(|dy| ink.at(*x * sx, *y * sy + dy) > 0.);
                if present {
                    length += if length > 0. { f64::from(gap + 1) } else { 1. };
                    longest = longest.max(length);
                    gap = 0;
                } else {
                    gap += 1;
                    if gap > 1 {
                        length = 0.;
                        gap = 0;
                    }
                }
            }
            longest >= 4. * spacing
        })
        .count();
    if continuous < 3 {
        return None;
    }
    Some(staff)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn decorated_staff(curved: bool) -> (Ink, Vec<(usize, [f64; 5])>) {
        let mut ink = Ink {
            w: 700,
            h: 280,
            pixels: vec![0.; 700 * 280],
            seed_paper: vec![true; 700 * 280],
            allowed: None,
        };
        let center = |x: usize| {
            130. + if curved {
                7. * ((x as f64 - 50.) / 600. * std::f64::consts::PI).sin()
            } else {
                0.
            }
        };
        for x in 30..670 {
            for k in -2..=2 {
                let y = (center(x) + k as f64 * 12.).round() as usize;
                ink.pixels[y * ink.w + x] = 1.;
            }
        }
        // Thin edge runs of a notehead can pass the broad-run rejection while
        // pulling one line off the staff. The surrounding staff is unambiguous.
        for (cx, k) in [(230, 1), (258, 2), (420, -1)] {
            for x in cx - 9..=cx + 9 {
                let y = (center(x) + k as f64 * 12.).round() as usize;
                for yy in y..=y + 2 {
                    ink.pixels[yy * ink.w + x] = 1.;
                }
            }
        }
        let track = (50..650)
            .step_by(6)
            .map(|x| {
                (
                    x,
                    std::array::from_fn(|k| center(x) + (k as f64 - 2.) * 12.),
                )
            })
            .collect();
        (ink, track)
    }

    #[test]
    fn notehead_edges_do_not_bend_staff_traces() {
        for curved in [false, true] {
            let (ink, track) = decorated_staff(curved);
            let found = staff(&ink, &track, 1., 1.).unwrap();
            let max_error = found
                .x
                .iter()
                .enumerate()
                .filter(|(_, x)| **x > 100. && **x < 600.)
                .flat_map(|(j, x)| (0..5).map(move |k| (j, *x, k)))
                .map(|(j, x, k)| {
                    let truth = 130.
                        + (k as f64 - 2.) * 12.
                        + if curved {
                            7. * ((x - 50.) / 600. * std::f64::consts::PI).sin()
                        } else {
                            0.
                        };
                    (found.y[k][j] - truth).abs()
                })
                .fold(0_f64, f64::max);
            assert!(max_error < 0.65, "curved={curved}, max error={max_error}");
        }
    }

    #[test]
    fn repeated_text_fragments_do_not_form_a_staff() {
        let mut ink = Ink {
            w: 900,
            h: 400,
            pixels: vec![0.; 900 * 400],
            seed_paper: vec![true; 900 * 400],
            allowed: None,
        };
        // Five paragraph baselines: locally convincing thin glyph fragments,
        // but no continuous rules even though line spacing is perfectly equal.
        for x in 80..820 {
            if x % 16 >= 12 {
                continue;
            }
            for center in [100, 120, 140, 160, 180] {
                for y in center - 1..=center + 1 {
                    ink.pixels[y * 900 + x] = 1.;
                }
            }
        }
        let track: Vec<_> = (90..810)
            .step_by(6)
            .map(|x| (x, [100., 120., 140., 160., 180.]))
            .collect();
        assert!(staff(&ink, &track, 1., 1.).is_none());
    }

    #[test]
    fn short_staff_is_measured_in_staff_spaces_not_page_fraction() {
        let mut ink = Ink {
            w: 900,
            h: 400,
            pixels: vec![0.; 900 * 400],
            seed_paper: vec![true; 900 * 400],
            allowed: None,
        };
        for x in 350..550 {
            for center in [200, 209, 218, 227, 236] {
                ink.pixels[center * 900 + x] = 1.;
            }
        }
        let track: Vec<_> = (360..540)
            .step_by(6)
            .map(|x| (x, [200., 209., 218., 227., 236.]))
            .collect();
        let found =
            staff(&ink, &track, 1., 1.).expect("a 22-space staff remains valid on a wide page");
        assert!(found.x[0] < 360. && found.x.last().unwrap() > &540.);
    }

    #[test]
    fn broad_connected_runs_still_abstain() {
        let mut ink = Ink {
            w: 160,
            h: 200,
            pixels: vec![0.; 160 * 200],
            seed_paper: vec![true; 160 * 200],
            allowed: None,
        };
        for x in 20..140 {
            for center in [100, 109, 117, 126, 134] {
                for y in center - 3..=center + 3 {
                    ink.pixels[y * 160 + x] = 1.;
                }
            }
        }
        assert!(
            fit(
                &ink,
                State {
                    x: 80.,
                    h: 117.,
                    d: 8.5,
                    m: 0.
                }
            )
            .is_none()
        );
    }
    #[test]
    fn three_pixel_staff_strokes_survive_fractional_spacing_run_budget() {
        let mut ink = Ink {
            w: 160,
            h: 200,
            pixels: vec![0.; 160 * 200],
            seed_paper: vec![true; 160 * 200],
            allowed: None,
        };
        for x in 20..140 {
            for center in [100, 109, 117, 126, 134] {
                for y in center - 1..=center + 1 {
                    ink.pixels[y * 160 + x] = 1.;
                }
            }
        }
        let predicted = State {
            x: 80.,
            h: 117.,
            d: 8.5,
            m: 0.,
        };
        assert_eq!(evidence(&ink, predicted), 1.);
        let (_, lines, count) = fit(&ink, predicted)
            .expect("ordinary 3 px staff ink must not be rejected by a 2.975 px run threshold");
        assert_eq!(count, 5);
        for (actual, expected) in lines.iter().zip([100., 109., 117., 126., 134.]) {
            assert!((actual - expected).abs() < 0.1);
        }
    }
    #[test]
    fn long_sixth_line_selects_staff_identity_over_a_short_beam() {
        let mut ink = Ink {
            w: 500,
            h: 220,
            pixels: vec![0.; 500 * 220],
            seed_paper: vec![true; 500 * 220],
            allowed: None,
        };
        for x in 20..480 {
            for y in [100, 110, 120, 130, 140] {
                ink.pixels[y * 500 + x] = 1.;
            }
        }
        for x in 100..350 {
            ink.pixels[90 * 500 + x] = 1.;
            ink.pixels[150 * 500 + x] = 1.;
        }
        for initial in [-1, 1] {
            let mut track: Vec<_> = (30..470)
                .step_by(3)
                .map(|x| {
                    (
                        x as f64,
                        std::array::from_fn(|k| 100. + 10. * (k as i32 + initial) as f64),
                    )
                })
                .collect();
            assert_eq!(correct_phase(&ink, &mut track), -initial);
            for (_, lines) in track {
                assert_eq!(lines, [100., 110., 120., 130., 140.]);
            }
        }
    }
}
