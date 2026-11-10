//! Bidirectional local evidence tracking from coarse five-line seeds.
//! Short tangent-aligned probes survive curved gutter regions; white gaps and
//! uniform dark page edges do not provide five separate ink ridges.
use super::{Ink, smooth};
use crate::geometry::Staff;
use crate::math::median;

/// Masked ink and its vertical three-pixel maximum as byte planes: tracing
/// probes these two neighbourhoods millions of times per page.
pub(super) struct Planes<'a> {
    ink: &'a Ink,
    own: Vec<bool>,
    tall: Vec<bool>,
}

impl<'a> Planes<'a> {
    pub(super) fn new(ink: &'a Ink) -> Self {
        let own: Vec<bool> = match &ink.allowed {
            Some(mask) => ink
                .pixels
                .iter()
                .zip(mask)
                .map(|(value, inside)| *inside && *value > 0.)
                .collect(),
            None => ink.pixels.iter().map(|value| *value > 0.).collect(),
        };
        let mut tall = own.clone();
        for (y, row) in tall.chunks_exact_mut(ink.w).enumerate() {
            for neighbour in [y.wrapping_sub(1), y + 1] {
                if neighbour < ink.h {
                    let source = &own[neighbour * ink.w..(neighbour + 1) * ink.w];
                    for (value, other) in row.iter_mut().zip(source) {
                        *value |= *other;
                    }
                }
            }
        }
        Self { ink, own, tall }
    }

    fn column(&self, x: f64) -> Option<usize> {
        let column = x.round();
        (column >= 0. && column < self.ink.w as f64).then_some(column as usize)
    }

    fn row(&self, y: f64) -> Option<usize> {
        let row = y.round();
        (row >= 0. && row < self.ink.h as f64).then_some(row as usize)
    }

    fn inked(&self, column: usize, y: f64) -> bool {
        self.row(y)
            .is_some_and(|row| self.own[row * self.ink.w + column])
    }

    /// Whether any of `y - 1.`, `y` and `y + 1.` is inked.
    fn ridged(&self, column: usize, y: f64) -> bool {
        let row = y.round();
        // The three rows round separately and are not always adjacent: around
        // y = ±0.5 they skip row 0, which the precomputed maximum includes.
        if (y - 1.).round() == row - 1.
            && (y + 1.).round() == row + 1.
            && let Some(row) = self.row(y)
        {
            self.tall[row * self.ink.w + column]
        } else {
            [-1., 0., 1.].iter().any(|dy| self.inked(column, y + dy))
        }
    }

    /// `Ink::at`, for ink that is exactly 0 or 1.
    fn at(&self, x: f64, y: f64) -> f64 {
        let inked = self.column(x).is_some_and(|column| self.inked(column, y));
        f64::from(u8::from(inked))
    }

    /// The largest of `at(x, y - 1.)`, `at(x, y)` and `at(x, y + 1.)`.
    fn ridge(&self, x: f64, y: f64) -> f64 {
        let ridged = self.column(x).is_some_and(|column| self.ridged(column, y));
        f64::from(u8::from(ridged))
    }
}

/// The five columns, two pixels apart, on which every state at one `x` shows
/// its comb.
struct Comb<'a> {
    planes: &'a Planes<'a>,
    columns: [(f64, Option<usize>); 5],
}

impl<'a> Comb<'a> {
    fn new(planes: &'a Planes<'a>, x: f64) -> Self {
        let columns = [-4., -2., 0., 2., 4.].map(|dx| (dx, planes.column(x + dx)));
        Self { planes, columns }
    }

    fn count(&self, y: f64, m: f64, inked: impl Fn(usize, f64) -> bool) -> f64 {
        let hits = self
            .columns
            .iter()
            .filter(|(dx, column)| column.is_some_and(|column| inked(column, y + m * dx)));
        hits.count() as f64
    }

    /// Columns with a ridge on the line of slope `m` through `y` at `x`.
    fn ridges(&self, y: f64, m: f64) -> f64 {
        self.count(y, m, |column, y| self.planes.ridged(column, y))
    }

    /// Columns with ink on the line of slope `m` through `y` at `x`.
    fn ink(&self, y: f64, m: f64) -> f64 {
        self.count(y, m, |column, y| self.planes.inked(column, y))
    }
}

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

/// Five-line evidence on the comb at `state.x`: mean ridge support less the
/// densest interline ink. `None` unless three lines are strong, the mean reaches 0.58 and the
/// result is `viable`. That predicate must not accept a value after rejecting a
/// larger one: reading stops once even full support on the unread lines, or
/// the interline ink already seen, rules the result out.
fn viable_evidence(comb: &Comb, state: State, viable: impl Fn(f64) -> bool) -> Option<f64> {
    let supported = |strong: i32, mean: f64| strong >= 3 && mean >= 0.58 && viable(mean);
    let mut foreground = 0.;
    let mut strong = 0;
    for k in -2..=2 {
        let unread = 3 - k;
        let reachable = (0..unread).fold(foreground, |mean, _| mean + 1. / 5.);
        if !supported(strong + unread, reachable) {
            return None;
        }
        let sum = comb.ridges(state.h + f64::from(k) * state.d, state.m) / 5.;
        foreground += sum / 5.;
        if sum >= 0.6 {
            strong += 1;
        }
    }
    if !supported(strong, foreground) {
        return None;
    }
    let mut background = 0_f64;
    for fraction in [1. / 3., 0.5, 2. / 3.] {
        let between = (-2..2)
            .map(|k| comb.ink(state.h + (f64::from(k) + fraction) * state.d, state.m))
            .sum::<f64>()
            / 20.;
        background = background.max(between);
        if !viable(foreground - 0.8 * background) {
            return None;
        }
    }
    Some(foreground - 0.8 * background)
}

fn evidence(planes: &Planes, state: State) -> f64 {
    viable_evidence(&Comb::new(planes, state.x), state, |_| true).unwrap_or(-1.)
}

fn exterior_evidence(comb: &Comb, state: State) -> f64 {
    // Four rules plus a tie can resemble the adjacent five-line packet. The
    // omitted fifth rule then remains visible just outside it. Include that
    // evidence while walking, rather than correcting one phase for the whole
    // staff after a local identity slip has already propagated to its end.
    [-3., 3.]
        .iter()
        .map(|k| comb.ridges(state.h + k * state.d, state.m) / 5.)
        .fold(0., f64::max)
}

fn correct_phase(planes: &Planes, data: &mut [(f64, [f64; 5])]) -> i32 {
    let score = |phase: i32| {
        let mut foreground = 0.;
        let mut above = 0.;
        let mut below = 0.;
        let mut count = 0.;
        for (x, y) in data.iter().step_by((data.len() / 128).max(1)) {
            let d = (y[4] - y[0]) / 4.;
            let sample = |k: f64| planes.ridge(*x, y[2] + (k + f64::from(phase)) * d);
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

/// The coarse probes cover both sides of an occlusion. Their five-line identity
/// is independent of the fine walk's accumulated slope and prevents ties from
/// pulling that walk onto an adjacent or compressed packet. Unsupported gaps
/// and irregular probe groups provide no prior.
fn coarse_guide(track: &[(usize, [f64; 5])], x: f64) -> Option<(f64, f64)> {
    let i = track.partition_point(|(xx, _)| (*xx as f64) < x);
    if i == 0 || i == track.len() {
        return None;
    }
    let packet = |rows: [f64; 5]| {
        let h = rows.iter().sum::<f64>() / 5.;
        let d = rows
            .iter()
            .enumerate()
            .map(|(k, y)| (k as f64 - 2.) * y)
            .sum::<f64>()
            / 10.;
        (d > 0.
            && rows
                .iter()
                .enumerate()
                .all(|(k, y)| (y - h - (k as f64 - 2.) * d).abs() <= (0.18 * d).max(1.)))
        .then_some((h, d))
    };
    let (a, b) = (track[i - 1], track[i]);
    let (ha, da) = packet(a.1)?;
    let (hb, db) = packet(b.1)?;
    if b.0 - a.0 == 0 || (b.0 - a.0) as f64 > 8. * da.min(db) || (da - db).abs() > 0.2 * da.min(db)
    {
        return None;
    }
    let t = (x - a.0 as f64) / (b.0 - a.0) as f64;
    Some((ha * (1. - t) + hb * t, da * (1. - t) + db * t))
}

fn fits_guide(state: State, guide: Option<(f64, f64)>) -> bool {
    guide.is_none_or(|(h, d)| {
        (state.h - h).abs() <= 0.25 * d + 0.5 && (state.d - d).abs() <= 0.12 * d + 0.25
    })
}

fn fit(planes: &Planes, predicted: State) -> Option<(State, [f64; 5], usize)> {
    fit_guided(planes, predicted, None)
}

fn fit_guided(
    planes: &Planes,
    predicted: State,
    guide: Option<(f64, f64)>,
) -> Option<(State, [f64; 5], usize)> {
    let ink = planes.ink;
    let radius = (predicted.d * 0.4).ceil().max(2.) as i32;
    let comb = Comb::new(planes, predicted.x);
    let mut best = predicted;
    let mut score = -1.;
    // The best of all candidates is the first maximum of an ascending scan
    // over offset, spacing ratio and slope change. Visiting small corrections
    // first finds it early; `scan` then still settles ties as that scan does.
    let mut scan = usize::MAX;
    for dy in std::iter::once(0).chain((1..=radius).flat_map(|distance| [-distance, distance])) {
        for (ratio_scan, ratio) in [(1, 1.), (0, 0.96), (2, 1.04)] {
            let upright = State {
                h: predicted.h + f64::from(dy),
                d: predicted.d * ratio,
                ..predicted
            };
            if !fits_guide(upright, guide) || !permitted(ink, upright) {
                continue;
            }
            if upright.h - 2. * upright.d < 0. || upright.h + 2. * upright.d > (ink.h - 1) as f64 {
                continue;
            }
            for (slope_scan, dm) in [(1, 0.), (0, -0.2), (2, 0.2)] {
                let candidate = State {
                    m: (predicted.m + dm).clamp(-1.3, 1.3),
                    ..upright
                };
                let candidate_scan = ((dy + radius) as usize * 3 + ratio_scan) * 3 + slope_scan;
                let rank = |support: f64| {
                    support
                        - 0.02 * f64::from(dy).abs() / predicted.d
                        - 0.03 * (ratio - 1.).abs()
                        - 0.005 * dm.abs()
                };
                let wins = |support: f64| {
                    let q = rank(support);
                    q > score || (q == score && candidate_scan < scan)
                };
                // Interline and exterior ink only lower a rank, so a candidate
                // whose remaining support cannot win is settled without them,
                // usually without reading any pixel.
                let Some(support) =
                    viable_evidence(&comb, candidate, |support| support >= 0.4 && wins(support))
                else {
                    continue;
                };
                let support = support - 0.35 * exterior_evidence(&comb, candidate);
                if wins(support) {
                    score = rank(support);
                    scan = candidate_scan;
                    best = candidate;
                }
            }
        }
    }
    // Exterior ink ranks competing phases; it must not reject a real staff
    // under a beam or shorten the existing dropout-bridging allowance.
    if score == -1. {
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
            // Nearest ink; of two equally near rows the upper one.
            let near = (0..=reach)
                .flat_map(|distance| [-distance, distance])
                .find(|dy| planes.at(best.x + dx, f64::from(y + dy)) > 0.);
            let Some(offset) = near else {
                continue;
            };
            let mut top = y + offset;
            let mut bottom = top;
            while top > 0
                && expected - f64::from(top) < best.d * 0.6
                && planes.at(best.x + dx, f64::from(top - 1)) > 0.
            {
                top -= 1;
            }
            while bottom + 1 < ink.h as i32
                && f64::from(bottom) - expected < best.d * 0.6
                && planes.at(best.x + dx, f64::from(bottom + 1)) > 0.
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
    if !fits_guide(best, guide) || best.d < predicted.d * 0.85 || best.d > predicted.d * 1.15 {
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

pub(super) fn staff(
    planes: &Planes,
    track: &[(usize, [f64; 5])],
    sx: f64,
    sy: f64,
) -> Option<Staff> {
    let ink = planes.ink;
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
        if let Some((fitted, ys, count)) = fit(planes, state) {
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
                evidence(planes, fitted) - 0.03 * ((j as f64 / track.len() as f64) - 0.5).abs();
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
            let guide = coarse_guide(track, x);
            let mut predicted = State {
                x,
                h: state.h + state.m * direction * 3.,
                ..state
            };
            if let Some((h, d)) = guide {
                // This bounds only the search prior. Missing ridge samples
                // remain pending and are bridged between measured endpoints.
                predicted.h = predicted.h.clamp(h - 0.25 * d, h + 0.25 * d);
                predicted.d = predicted.d.clamp(0.88 * d, 1.12 * d);
            }
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
            if let Some((mut next, lines, _)) = fit_guided(planes, predicted, guide) {
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
                // A strong five-line comb can persist beneath broad notation
                // even when narrow-ridge fitting abstains. Keep those samples
                // pending until ridges are reacquired; uniform dark bands and
                // blank gutters retain the short dropout budgets.
                let support = evidence(planes, predicted);
                let allowance = if support >= 0.25 {
                    24.
                } else if support >= 0. {
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
    // Four true lines plus a beam can look like another five-line comb. Use
    // evidence over the ENTIRE track to select the actual five-line phase.
    correct_phase(planes, &mut left);
    super::staff_smoothing::regularize(&mut left);
    let xx: Vec<f64> = left.iter().map(|p| p.0).collect();
    let x: Vec<f64> = (first.ceil() as usize..=last.floor() as usize)
        .map(|x| x as f64)
        .collect();
    let y = std::array::from_fn(|k| {
        let yy: Vec<f64> = left.iter().map(|p| p.1[k]).collect();
        let full: Vec<f64> = x
            .iter()
            .map(|x| crate::geometry::observations::interpolate(&xx, &yy, *x))
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
                if planes.ridge(*x * sx, *y * sy) > 0. {
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
    #[test]
    fn planes_and_combs_equal_separately_rounded_masked_lookups() {
        let (w, h) = (7, 6);
        let pattern = |i: usize, modulus: usize| (i * 7 + i / 3).is_multiple_of(modulus);
        let mut ink = Ink {
            w,
            h,
            pixels: (0..w * h)
                .map(|i| f64::from(u8::from(pattern(i, 3))))
                .collect(),
            seed_paper: vec![true; w * h],
            allowed: None,
        };
        for masked in [false, true] {
            if masked {
                ink.allowed = Some((0..w * h).map(|i| !pattern(i, 5)).collect());
            }
            let planes = Planes::new(&ink);
            let coordinates = |limit: usize| {
                (-12..=4 * limit as i32 + 12)
                    .map(|i| f64::from(i) / 4.)
                    .chain([
                        f64::NAN,
                        f64::INFINITY,
                        -0.5 - f64::EPSILON,
                        0.5 - f64::EPSILON,
                    ])
            };
            for x in coordinates(w) {
                for y in coordinates(h) {
                    let expected = [-1., 0., 1.]
                        .iter()
                        .map(|dy| ink.at(x, y + dy))
                        .fold(0., f64::max);
                    assert_eq!(
                        planes.at(x, y),
                        ink.at(x, y),
                        "x={x}, y={y}, masked={masked}"
                    );
                    assert_eq!(
                        planes.ridge(x, y),
                        expected,
                        "x={x}, y={y}, masked={masked}"
                    );
                }
            }
            for x in [-3., 0., 2.5, 3., 6.] {
                let comb = Comb::new(&planes, x);
                for (y, m) in [(-0.5, 0.), (0.5, 1.3), (2., -0.2), (3.25, 0.4), (5., -1.3)] {
                    let along = |at: &dyn Fn(f64, f64) -> f64| {
                        [-4., -2., 0., 2., 4.]
                            .iter()
                            .map(|dx| at(x + dx, y + m * dx))
                            .sum::<f64>()
                    };
                    assert_eq!(comb.ridges(y, m), along(&|x, y| planes.ridge(x, y)));
                    assert_eq!(comb.ink(y, m), along(&|x, y| planes.at(x, y)));
                }
            }
        }
    }

    #[test]
    fn long_dense_notation_rejoins_measured_staff_but_blank_gap_does_not() {
        for mode in [0, 1, 2] {
            let (mut ink, track) = decorated_staff(false);
            for x in 190..430 {
                for y in 100..160 {
                    ink.pixels[y * ink.w + x] = 0.;
                }
                if mode == 2 {
                    for y in 100..160 {
                        ink.pixels[y * ink.w + x] = 1.;
                    }
                }
                if mode == 1 {
                    for k in -2..=2 {
                        let center = 130 + k * 12;
                        for y in center - 3..=center + 3 {
                            ink.pixels[y as usize * ink.w + x] = 1.;
                        }
                    }
                }
            }
            // A left-side seed must cross dense notation using real comb ink,
            // then reacquire narrow ridges; blank examples remain separate.
            let left: Vec<_> = track.into_iter().filter(|(x, _)| *x < 170).collect();
            let found = staff(&Planes::new(&ink), &left, 1., 1.).unwrap();
            if mode == 1 {
                assert!(*found.x.last().unwrap() > 600., "dense run truncated");
                for k in 0..5 {
                    for y in &found.y[k] {
                        assert!((*y - (106. + 12. * k as f64)).abs() < 1.5);
                    }
                }
            } else {
                assert!(*found.x.last().unwrap() < 220.);
            }
        }
    }

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
            let found = staff(&Planes::new(&ink), &track, 1., 1.).unwrap();
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
        assert!(staff(&Planes::new(&ink), &track, 1., 1.).is_none());
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
        let found = staff(&Planes::new(&ink), &track, 1., 1.)
            .expect("a 22-space staff remains valid on a wide page");
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
                &Planes::new(&ink),
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
        let planes = Planes::new(&ink);
        assert_eq!(evidence(&planes, predicted), 1.);
        let (_, lines, count) = fit(&planes, predicted)
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
            assert_eq!(correct_phase(&Planes::new(&ink), &mut track), -initial);
            for (_, lines) in track {
                assert_eq!(lines, [100., 110., 120., 130., 140.]);
            }
        }
    }
}
