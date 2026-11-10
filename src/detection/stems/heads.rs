//! Head matching on untouched ink using the retained discrete templates.
//! Staff-rule pixels are unobserved evidence, not erased or counted as heads.
use super::{
    HeadEvidence, HeadShape,
    pixels::Pixels,
    ported::{
        metrics::NoteMetrics,
        templates::{HalfHeadTemplate, HeadMask, OpenHeadTemplates, TemplatePoint},
    },
};
use crate::geometry::{Point, Staff, observations::interpolate};
use std::collections::HashMap;

/// The share of an outline's visible points that must be ink.
const OUTLINE: f64 = 0.65;

#[derive(Clone, Copy)]
struct Match {
    shape: HeadShape,
    score: f64,
    known: f64,
}

/// Template points of one row: bit `i` of `mask` is the point at column `x + i`.
struct Run {
    x: i32,
    up: i32,
    mask: u64,
    /// The columns of the mask at any of the adjacent positions one read serves.
    span: u64,
}

/// A template point list as row masks, so a row is observed in one step. A
/// mask leaves `slack` of its 64 columns free: one read of a row then serves
/// the list at `slack + 1` adjacent positions.
struct Points {
    runs: Vec<Run>,
    len: usize,
    slack: usize,
}
impl Points {
    fn new(points: &[TemplatePoint], slack: usize) -> Self {
        let mut sorted = points.to_vec();
        sorted.sort_unstable_by_key(|p| (p.up, p.x));
        let mut runs: Vec<Run> = vec![];
        for p in sorted {
            // A repeated point votes once per repetition: it needs a run of its own.
            let free = runs.iter_mut().filter(|run| run.up == p.up).find(|run| {
                let bit = (p.x - run.x) as usize;
                bit < 64 - slack && run.mask >> bit & 1 == 0
            });
            match free {
                Some(run) => run.mask |= 1 << (p.x - run.x),
                None => runs.push(Run {
                    x: p.x,
                    up: p.up,
                    mask: 1,
                    span: 0,
                }),
            }
        }
        for run in &mut runs {
            run.span = (0..=slack).fold(0, |span, shift| span | run.mask << shift);
        }
        Self {
            runs,
            len: points.len(),
            slack,
        }
    }
}

/// A template mask with the paper ring that must surround a matched body.
struct Stencil {
    set: Points,
    clear: Points,
    /// The wing with the ink points it tolerates.
    wing: Option<(Points, i32)>,
    guard: Points,
}
impl Stencil {
    /// The stencil of `mask`, searched up to `radius` columns off an origin.
    fn new(mask: &HeadMask, radius: i32) -> Self {
        let points = |points: &[TemplatePoint]| Points::new(points, (2 * radius as usize).min(32));
        let body = || mask.set.iter().chain(&mask.clear);
        let mut guard = vec![];
        if let (Some(low), Some(high)) = (body().map(|p| p.up).min(), body().map(|p| p.up).max()) {
            for up in low - 2..=high + 2 {
                let near = || body().filter(|p| (p.up - up).abs() <= 2).map(|p| p.x);
                let (Some(left), Some(right)) = (near().min(), near().max()) else {
                    continue;
                };
                guard.extend(
                    [left - 2, left - 1, right + 1, right + 2]
                        .into_iter()
                        .filter(|x| x.abs() > 2)
                        .map(|x| TemplatePoint { x, up }),
                );
            }
        }
        Self {
            set: points(&mask.set),
            clear: points(&mask.clear),
            wing: mask
                .wing
                .as_ref()
                .map(|wing| (points(&wing.points), wing.threshold)),
            guard: points(&guard),
        }
    }
}

/// `[ink, observed]` counts of the template points that no staff rule hides.
fn observe(pixels: &Pixels, points: &Points, x: i32, y: i32) -> [usize; 2] {
    let mut count = [0; 2];
    for run in &points.runs {
        let observed = pixels.observed(x + run.x, y - run.up);
        for (count, bits) in count.iter_mut().zip(observed) {
            *count += (bits & run.mask).count_ones() as usize;
        }
    }
    count
}

/// The counts of [`observe`] for an outline at `(x, y)` and at each further
/// column of `counts`, `None` where it cannot hold enough ink to match.
fn outlines(
    pixels: &Pixels,
    points: &Points,
    x: i32,
    y: i32,
    counts: &mut [Option<[usize; 2]>],
    rows: &mut Vec<[u64; 2]>,
) {
    let step = points.slack + 1;
    for (group, counts) in counts.chunks_mut(step).enumerate() {
        let first = x + (group * step) as i32;
        rows.clear();
        rows.extend(
            points
                .runs
                .iter()
                .map(|run| pixels.observed(first + run.x, y - run.up)),
        );
        // No position of the group sees more ink, or has more points hidden,
        // than all of them together: most groups end here.
        let (mut ink, mut hidden) = (0, 0);
        for (run, [seen, open]) in points.runs.iter().zip(&*rows) {
            ink += (seen & run.span).count_ones() as usize;
            hidden += (!open & run.span).count_ones() as usize;
        }
        if hidden < points.len && (ink as f64 / (points.len - hidden) as f64) < OUTLINE {
            counts.fill(None);
            continue;
        }
        for (shift, count) in counts.iter_mut().enumerate() {
            let mut sum = [0; 2];
            for (run, row) in points.runs.iter().zip(&*rows) {
                for (sum, bits) in sum.iter_mut().zip(row) {
                    *sum += (bits >> shift & run.mask).count_ones() as usize;
                }
            }
            *count = Some(sum);
        }
    }
}

/// The match of the stencil at `(x, y)`, given the counts of its `outline` there.
fn score(pixels: &Pixels, stencil: &Stencil, x: i32, y: i32, outline: [usize; 2]) -> Option<Match> {
    if outline[1] < 4 {
        return None;
    }
    let outer = outline[0] as f64 / outline[1] as f64;
    if outer < OUTLINE {
        return None;
    }
    let hollow = observe(pixels, &stencil.clear, x, y);
    // Neutralized ink must not turn an almost-unobserved shape into a match.
    let known =
        (outline[1] + hollow[1]) as f64 / (stencil.set.len + stencil.clear.len).max(1) as f64;
    if hollow[1] < 3 || known < 0.45 {
        return None;
    }
    let inner = hollow[0] as f64 / hollow[1] as f64;
    let (shape, interior) = if inner >= 0.7 {
        (HeadShape::Filled, inner)
    } else if inner <= 0.4 {
        (HeadShape::Open, 1. - inner)
    } else {
        return None;
    };
    if let Some((wing, threshold)) = &stencil.wing {
        let [ink, count] = observe(pixels, wing, x, y);
        if count > 0 && (ink as f64 / count as f64) > *threshold as f64 / wing.len.max(1) as f64 {
            return None;
        }
    }
    // Require observable paper around the discrete body, not just enough ink
    // inside it. Otherwise a template can match a subpart of a sharp or rest.
    let guard = observe(pixels, &stencil.guard, x, y);
    if guard[1] < 6 || guard[0] as f64 / guard[1] as f64 > 0.2 {
        return None;
    }
    // Require agreement of the complete visible template, not just the
    // separate permissive outline/interior prechecks above.
    let quality = (outer + interior * 4. / 3.) / (1. + 4. / 3.);
    Some(Match {
        shape,
        score: quality,
        known,
    })
}

/// A match with its offset from the origin it was searched around.
type Placed = (Match, i32, i32);

/// One half-head template in the two sizes it is searched at.
struct Half {
    right: bool,
    space: i32,
    half: i32,
    /// How far off an origin the stencils are searched, in both axes.
    radius: i32,
    stencils: [Stencil; 2],
}
impl Half {
    /// The best match of either size around the origin `(x, y)`, with its offset.
    fn search(&self, pixels: &Pixels, x: i32, y: i32) -> Option<Placed> {
        let radius = self.radius;
        let mut counts = vec![None; 2 * radius as usize + 1];
        let mut rows = vec![];
        let mut best: Option<Placed> = None;
        for stencil in &self.stencils {
            for dy in -radius..=radius {
                // Nearly every position fails on its outline, so the outlines
                // of a whole row of positions are counted from one read.
                let (left, y) = (x - radius, y + dy);
                outlines(pixels, &stencil.set, left, y, &mut counts, &mut rows);
                for (dx, outline) in (-radius..=radius).zip(&counts) {
                    if let Some(outline) = *outline
                        && let Some(hit) = score(pixels, stencil, x + dx, y, outline)
                        && best
                            .as_ref()
                            .is_none_or(|(old, _, _)| hit.score > old.score)
                    {
                        best = Some((hit, dx, dy));
                    }
                }
            }
        }
        best
    }
}

/// The head templates of one staff, prepared once for all its stems. The
/// direction trials of a component probe nearly the same shafts, so the
/// searches already made are kept by half and origin.
pub(super) struct HeadMatcher<'a> {
    pixels: &'a Pixels,
    halves: [Half; 4],
    searched: HashMap<(usize, i32, i32), Option<Placed>>,
}
impl<'a> HeadMatcher<'a> {
    pub fn new(pixels: &'a Pixels, space: f64) -> Self {
        let metrics = NoteMetrics::new(space.round() as i32);
        let m = metrics.space;
        let templates = OpenHeadTemplates::new(m.space);
        let half = |template: HalfHeadTemplate, right: bool, small: bool| {
            let radius = (template.space >> 3).max(1);
            Half {
                right,
                space: template.space,
                half: if small {
                    metrics.grace.half + metrics.grace.eighth
                } else {
                    m.half + m.eighth
                },
                radius,
                stencils: [&template.narrow, &template.full].map(|mask| Stencil::new(mask, radius)),
            }
        };
        Self {
            pixels,
            searched: HashMap::new(),
            halves: [
                half(templates.left, false, false),
                half(templates.small_left, false, true),
                half(templates.right, true, false),
                half(templates.small_right, true, true),
            ],
        }
    }

    pub fn find_all(
        &mut self,
        from: Point,
        to: Point,
        slope: f64,
        space: f64,
        staff: &Staff,
    ) -> Vec<HeadEvidence> {
        let middle = interpolate(&staff.x, &staff.y[2], from.x);
        let low = ((from.y - space - middle) * 2. / space).floor() as i32;
        let high = ((to.y + space - middle) * 2. / space).ceil() as i32;
        let mut found: Vec<HeadEvidence> = vec![];
        for (index, template) in self.halves.iter().enumerate() {
            let half = template.half;
            for rung in low..=high {
                let cy = (middle + f64::from(rung) * space / 2.).round() as i32;
                if (f64::from(cy) - middle).abs() > 4. * space {
                    continue;
                }
                let x = (from.x + slope * (f64::from(cy) - from.y)).round() as i32;
                let origin_y = cy + template.space / 2;
                let Some((hit, dx, dy)) = *self
                    .searched
                    .entry((index, x, origin_y))
                    .or_insert_with(|| template.search(self.pixels, x, origin_y))
                else {
                    continue;
                };
                let center = Point {
                    x: f64::from(x + dx + if template.right { half } else { -half }),
                    y: f64::from(cy + dy),
                };
                let height = f64::from(template.space) / 2.;
                let head = HeadEvidence {
                    center,
                    bounds: [
                        center.x - f64::from(half),
                        center.y - height,
                        center.x + f64::from(half),
                        center.y + height,
                    ],
                    shape: hit.shape,
                    score: hit.score,
                    known_support: hit.known,
                };
                if let Some(old) = found.iter_mut().find(|h| {
                    (h.center.x - center.x).abs() < space * 0.5
                        && (h.center.y - center.y).abs() < space * 0.75
                }) {
                    if head.score > old.score {
                        *old = head;
                    }
                } else {
                    found.push(head);
                }
            }
        }
        found.sort_by(|a, b| a.center.y.total_cmp(&b.center.y));
        found
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detection::stems::pixels::Pixels;
    use image::{GrayImage, Luma};
    #[test]
    fn staff_rule_ink_does_not_vote_for_or_against_a_template() {
        let staff = Staff {
            x: vec![0., 79.],
            y: std::array::from_fn(|k| vec![20. + k as f64 * 10.; 2]),
        };
        let mask = OpenHeadTemplates::new(10).left.full;
        let stencil = Stencil::new(&mask, 1);
        let score = |pixels: &Pixels| {
            let outline = observe(pixels, &stencil.set, 50, 50);
            score(pixels, &stencil, 50, 50, outline)
        };
        let mut a = GrayImage::from_pixel(80, 80, Luma([255]));
        for y in [20, 30, 40, 50, 60] {
            for x in 0..80 {
                a.put_pixel(x, y, Luma([0]));
            }
        }
        for p in &mask.set {
            a.put_pixel((50 + p.x) as u32, (50 - p.up) as u32, Luma([0]));
        }
        let mut b = a.clone();
        for y in [40, 50] {
            for x in 25..57 {
                b.put_pixel(x, y, Luma([255]));
            }
        }
        let pa = Pixels::new(&a, &staff, 10.);
        let pb = Pixels::new(&b, &staff, 10.);
        let sa = score(&pa).expect("masked open template");
        let sb = score(&pb).expect("same template without rule ink");
        assert_eq!(sa.shape, HeadShape::Open);
        assert_eq!(sb.shape, HeadShape::Open);
        assert_eq!(sa.score, sb.score);
        assert_eq!(sa.known, sb.known);
    }
}
