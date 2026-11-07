//! Optional smooth horizontal shear shared within musical systems.
use super::Flatten;
use crate::geometry::models::perspective::Projective;
use crate::geometry::observations::interpolate;
use crate::geometry::{Observations, Point, StaffEdge, SystemReport};
use crate::math::{median, slope};
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub(super) struct Shear {
    pub(super) systems: Vec<SystemReport>,
    knots: Vec<f64>,
    values: Vec<f64>,
    derivatives: Vec<f64>,
}

impl Shear {
    pub(super) fn at(&self, y: f64) -> f64 {
        if y <= self.knots[0] || y >= self.knots[self.knots.len() - 1] {
            return 0.;
        }
        let i = self.knots.partition_point(|v| *v <= y) - 1;
        let length = self.knots[i + 1] - self.knots[i];
        let t = (y - self.knots[i]) / length;
        (2. * t * t * t - 3. * t * t + 1.) * self.values[i]
            + (t * t * t - 2. * t * t + t) * length * self.derivatives[i]
            + (-2. * t * t * t + 3. * t * t) * self.values[i + 1]
            + (t * t * t - t * t) * length * self.derivatives[i + 1]
    }
    pub(super) fn forward(&self, p: Point) -> Point {
        Point {
            x: p.x + self.at(p.y),
            y: p.y,
        }
    }
    pub(super) fn inverse(&self, p: Point) -> Point {
        Point {
            x: p.x - self.at(p.y),
            y: p.y,
        }
    }
}

pub(crate) fn systems(data: &Observations) -> Vec<Vec<usize>> {
    let n = data.staves.len();
    let mut counts = vec![0_usize; n];
    let mut matched = vec![vec![0_usize; n]; n];
    let mut full_spans = vec![0_usize; n];
    for b in &data.barlines {
        counts[b.staff] += 1;
        for other in &b.aligned_staffs {
            matched[b.staff][*other] += 1;
        }
        // Native detection emits one physical path for a connected bar.
        // Older observations instead contain reciprocal staff-height paths;
        // retain their voting rule without counting those twice.
        let mut members = b.aligned_staffs.clone();
        members.push(b.staff);
        members.sort_unstable();
        for pair in members.windows(2) {
            let (a, z) = (pair[0], pair[1]);
            if z != a + 1 {
                continue;
            }
            let upper = interpolate(&data.staves[a].x, &data.staves[a].y[2], b.x[0]);
            let lower = interpolate(
                &data.staves[z].x,
                &data.staves[z].y[2],
                *b.x.last().unwrap(),
            );
            if b.y[0] <= upper && *b.y.last().unwrap() >= lower {
                full_spans[z] += 1;
            }
        }
    }
    let mut out = vec![vec![0]];
    for i in 1..n {
        let threshold = 2_f64.max(0.4 * counts[i - 1].min(counts[i]) as f64);
        if full_spans[i] >= 2 || matched[i][i - 1].min(matched[i - 1][i]) as f64 >= threshold {
            out.last_mut().unwrap().push(i);
        } else {
            out.push(vec![i]);
        }
    }
    out
}

pub(super) fn shear(
    data: &Observations,
    flat: &Flatten,
    projective: &Projective,
    space: f64,
    enabled: bool,
    edges: &[StaffEdge],
) -> Result<Shear> {
    let paths: Vec<Vec<Point>> = data
        .barlines
        .iter()
        .map(|b| {
            b.x.iter()
                .zip(&b.y)
                .map(|(x, y)| flat.forward(projective.forward(Point { x: *x, y: *y })))
                .collect()
        })
        .collect();
    let mut reports = vec![];
    for group in systems(data) {
        let top = group
            .iter()
            .map(|i| flat.targets[1 + 5 * i])
            .fold(f64::INFINITY, f64::min);
        let bottom = group
            .iter()
            .map(|i| flat.targets[5 + 5 * i])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut lo = top;
        let mut hi = bottom;
        let mut slopes = vec![];
        for (b, p) in data.barlines.iter().zip(&paths) {
            if group.contains(&b.staff) {
                lo = lo.min(p.first().unwrap().y);
                hi = hi.max(p.last().unwrap().y);
                if p.last().unwrap().y - p.first().unwrap().y >= space {
                    slopes.extend(std::iter::repeat_n(slope(p), 4));
                }
            }
        }
        // Weaker, explicitly geometric staff edges remain separate from bars.
        for edge in edges
            .iter()
            .filter(|e| group.contains(&e.staff) && e.confidence >= 0.7)
        {
            let points: Vec<_> = edge
                .points
                .iter()
                .map(|p| flat.forward(projective.forward(*p)))
                .collect();
            if points.last().unwrap().y - points[0].y >= space {
                slopes.push(slope(&points));
            }
        }
        let candidate_shear = if slopes.is_empty() {
            0.
        } else {
            (-median(slopes)).clamp(-0.15, 0.15)
        };
        reports.push(SystemReport {
            staves: group,
            region: [lo - 0.5 * space, hi + 0.5 * space],
            center: (top + bottom) / 2.,
            shear: if enabled { candidate_shear } else { 0. },
            candidate_shear,
        });
    }
    if !enabled {
        return Ok(Shear {
            systems: reports,
            knots: vec![flat.min_y, flat.max_y],
            values: vec![0., 0.],
            derivatives: vec![0., 0.],
        });
    }
    for i in 1..reports.len() {
        if reports[i - 1].region[1] + space >= reports[i].region[0] {
            let left = flat.targets[5 + 5 * reports[i - 1].staves[reports[i - 1].staves.len() - 1]];
            let right = flat.targets[1 + 5 * reports[i].staves[0]];
            let edge = (left + right) / 2.;
            reports[i - 1].region[1] = reports[i - 1].region[1].min(edge - 0.5 * space);
            reports[i].region[0] = reports[i].region[0].max(edge + 0.5 * space);
        }
    }
    let mut result = Shear {
        knots: vec![reports[0].region[0] - 3. * space],
        values: vec![0.],
        derivatives: vec![0.],
        systems: reports,
    };
    for system in &result.systems {
        for y in system.region {
            result.knots.push(y);
            result.values.push(system.shear * (y - system.center));
            result.derivatives.push(system.shear);
        }
    }
    result
        .knots
        .push(result.systems.last().unwrap().region[1] + 3. * space);
    result.values.push(0.);
    result.derivatives.push(0.);
    if result.knots.windows(2).any(|w| w[0] >= w[1]) {
        return Err(Error::Invalid(
            "overlapping systems require explicit geometry repair",
        ));
    }
    Ok(result)
}
