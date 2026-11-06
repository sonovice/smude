//! Locally supported monotone staff flattening. Horizontal coordinates never
//! change. Only staffs whose measured x domains overlap constrain their order.
use crate::{Error, Point, Result};

#[derive(Clone, Debug)]
struct StaffCurves {
    lines: [Vec<Point>; 5],
    left: f64,
    right: f64,
    support: [f64; 2],
    center: f64,
    space: f64,
}

fn line_at(line: &[Point], x: f64) -> f64 {
    if x <= line[0].x {
        return line[0].y;
    }
    if x >= line[line.len() - 1].x {
        return line[line.len() - 1].y;
    }
    let hi = line.partition_point(|p| p.x <= x);
    let (a, b) = (line[hi - 1], line[hi]);
    a.y + (b.y - a.y) * (x - a.x) / (b.x - a.x)
}
impl StaffCurves {
    fn band(&self, x: f64) -> [f64; 2] {
        [line_at(&self.lines[0], x), line_at(&self.lines[4], x)]
    }
    fn target_band(&self) -> [f64; 2] {
        [self.center - 2. * self.space, self.center + 2. * self.space]
    }
    fn measured(&self, x: f64) -> bool {
        x >= self.left && x <= self.right
    }
    fn weight(&self, x: f64) -> f64 {
        let t = if x < self.left {
            (x - self.support[0]) / (self.left - self.support[0]).max(1e-12)
        } else if x > self.right {
            (self.support[1] - x) / (self.support[1] - self.right).max(1e-12)
        } else {
            1.
        };
        let t = t.clamp(0., 1.);
        t * t * (3. - 2. * t)
    }
}
fn overlap(a: [f64; 2], b: [f64; 2]) -> bool {
    a[0] <= b[1] && b[0] <= a[1]
}

#[derive(Clone, Debug)]
struct Column {
    source: Vec<f64>,
    target: Vec<f64>,
}
impl Column {
    fn at(&self, y: f64) -> f64 {
        let i = self
            .source
            .partition_point(|v| *v <= y)
            .saturating_sub(1)
            .min(self.source.len() - 2);
        self.target[i]
            + (y - self.source[i]) / (self.source[i + 1] - self.source[i])
                * (self.target[i + 1] - self.target[i])
    }
}
#[derive(Clone, Debug)]
struct Interval {
    x: [f64; 2],
    /// Each anchor is [source_left, source_right, target_left, target_right].
    anchors: Vec<[f64; 4]>,
}
impl Interval {
    fn segment(&self, x: f64, y: f64, inverse: bool) -> [f64; 4] {
        let t = ((x - self.x[0]) / (self.x[1] - self.x[0])).clamp(0., 1.);
        let coordinate = |a: &[f64; 4], target: bool| {
            let k = if target { 2 } else { 0 };
            a[k] + t * (a[k + 1] - a[k])
        };
        let mut i = self
            .anchors
            .partition_point(|a| coordinate(a, inverse) <= y)
            .saturating_sub(1)
            .min(self.anchors.len() - 2);
        // An anchor may be born at an existing anchor. The zero-width band
        // exists only at the exact interval endpoint and has zero target width.
        while i + 1 < self.anchors.len() - 1
            && coordinate(&self.anchors[i + 1], inverse) - coordinate(&self.anchors[i], inverse)
                == 0.
        {
            i += 1;
        }
        while i > 0
            && coordinate(&self.anchors[i + 1], inverse) - coordinate(&self.anchors[i], inverse)
                == 0.
        {
            i -= 1;
        }
        let (a, b) = (
            coordinate(&self.anchors[i], inverse),
            coordinate(&self.anchors[i + 1], inverse),
        );
        let (c, d) = (
            coordinate(&self.anchors[i], !inverse),
            coordinate(&self.anchors[i + 1], !inverse),
        );
        [a, b, c, d]
    }
    fn at(&self, x: f64, y: f64, inverse: bool) -> f64 {
        let [a, b, c, d] = self.segment(x, y, inverse);
        c + (y - a) / (b - a) * (d - c)
    }
    fn vertical_scale(&self, x: f64, y: f64) -> f64 {
        let [a, b, c, d] = self.segment(x, y, false);
        (d - c) / (b - a)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LocalFlatten {
    intervals: Vec<Interval>,
    centers: Vec<f64>,
    pub(crate) min_source_gap_pixels: f64,
    pub(crate) max_vertical_scale: f64,
    pub(crate) max_boundary_jump_pixels: f64,
}

fn column(staffs: &[StaffCurves], x: f64, bounds: [f64; 4]) -> Result<Column> {
    let mut hard = vec![(bounds[1], bounds[1]), (bounds[3], bounds[3])];
    for s in staffs.iter().filter(|s| s.measured(x)) {
        for k in 0..5 {
            hard.push((
                line_at(&s.lines[k], x),
                s.center + (k as f64 - 2.) * s.space,
            ));
        }
    }
    hard.sort_by(|a, b| a.0.total_cmp(&b.0));
    if hard
        .windows(2)
        .any(|w| w[1].0 - w[0].0 <= 1e-10 || w[1].1 - w[0].1 <= 1e-10)
    {
        return Err(Error::Invalid("local measured staff anchors cross"));
    }
    let base = Column {
        source: hard.iter().map(|p| p.0).collect(),
        target: hard.iter().map(|p| p.1).collect(),
    };
    let mut anchors: Vec<(f64, f64, bool)> = hard.iter().map(|p| (p.0, p.1, true)).collect();
    for s in staffs
        .iter()
        .filter(|s| !s.measured(x) && x >= s.support[0] && x <= s.support[1])
    {
        let weight = s.weight(x);
        // A newly appearing zero-weight band must be inserted into the map
        // already induced by every other band, not force its hard-only base.
        // Omitting it here makes both adjacent interval representations agree.
        if weight == 0. {
            continue;
        }
        for k in 0..5 {
            let source = line_at(&s.lines[k], x);
            if source <= bounds[1] || source >= bounds[3] {
                continue;
            }
            let target = base.at(source);
            anchors.push((
                source,
                target + weight * (s.center + (k as f64 - 2.) * s.space - target),
                false,
            ));
        }
    }
    anchors.sort_by(|a, b| a.0.total_cmp(&b.0).then(b.2.cmp(&a.2)));
    // Zero-support endpoints may coincide with another band's boundary.
    anchors.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-10);
    let fixed: Vec<_> = anchors
        .iter()
        .enumerate()
        .filter_map(|(i, a)| a.2.then_some(i))
        .collect();
    for pair in fixed.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let alpha =
            0.05_f64.min(0.25 * (anchors[b].1 - anchors[a].1) / (anchors[b].0 - anchors[a].0));
        let lower = anchors[a].1 - alpha * anchors[a].0;
        let upper = anchors[b].1 - alpha * anchors[b].0;
        let mut blocks: Vec<(usize, usize, f64, usize)> = vec![];
        for (i, anchor) in anchors.iter().enumerate().take(b).skip(a + 1) {
            blocks.push((i, i, (anchor.1 - alpha * anchor.0).clamp(lower, upper), 1));
            while blocks.len() > 1 {
                let z = blocks[blocks.len() - 1];
                let p = blocks[blocks.len() - 2];
                if p.2 / p.3 as f64 <= z.2 / z.3 as f64 {
                    break;
                }
                blocks.pop();
                blocks.pop();
                blocks.push((p.0, z.1, p.2 + z.2, p.3 + z.3));
            }
        }
        for (lo, hi, sum, n) in blocks {
            for anchor in anchors.iter_mut().take(hi + 1).skip(lo) {
                anchor.1 = sum / n as f64 + alpha * anchor.0;
            }
        }
    }
    Ok(Column {
        source: anchors.iter().map(|a| a.0).collect(),
        target: anchors.iter().map(|a| a.1).collect(),
    })
}

impl LocalFlatten {
    pub(crate) fn fit(
        lines: &[[Vec<Point>; 5]],
        centers: &[f64],
        spacings: &[f64],
        bounds: [f64; 4],
    ) -> Result<Self> {
        if lines.is_empty()
            || lines.len() != centers.len()
            || lines.len() != spacings.len()
            || bounds.iter().any(|v| !v.is_finite())
            || bounds[0] >= bounds[2]
            || bounds[1] >= bounds[3]
        {
            return Err(Error::Invalid("invalid local staff flattening inputs"));
        }
        let mut staffs = vec![];
        for ((lines, center), space) in lines.iter().zip(centers).zip(spacings) {
            if !center.is_finite()
                || !space.is_finite()
                || *space <= 0.
                || lines.iter().any(|l| {
                    l.len() < 2
                        || l.iter().any(|p| !p.x.is_finite() || !p.y.is_finite())
                        || l.windows(2).any(|w| w[0].x >= w[1].x)
                })
            {
                return Err(Error::Invalid("invalid local staff curves"));
            }
            let left = lines.iter().map(|l| l[0].x).fold(f64::INFINITY, f64::min);
            let right = lines
                .iter()
                .map(|l| l[l.len() - 1].x)
                .fold(f64::NEG_INFINITY, f64::max);
            staffs.push(StaffCurves {
                lines: lines.clone(),
                left,
                right,
                support: [
                    (left - 8. * space).max(bounds[0]),
                    (right + 8. * space).min(bounds[2]),
                ],
                center: *center,
                space: *space,
            });
        }
        // Directed constraints exist only over actual common horizontal support.
        let mut edges = vec![];
        for i in 0..staffs.len() {
            for j in i + 1..staffs.len() {
                let (a, b) = (
                    staffs[i].left.max(staffs[j].left),
                    staffs[i].right.min(staffs[j].right),
                );
                if a > b {
                    continue;
                }
                let mut xx = vec![a, b];
                xx.extend(
                    staffs[i]
                        .lines
                        .iter()
                        .chain(&staffs[j].lines)
                        .flat_map(|l| l.iter())
                        .filter_map(|p| (p.x >= a && p.x <= b).then_some(p.x)),
                );
                let above = xx
                    .iter()
                    .all(|x| staffs[i].band(*x)[1] < staffs[j].band(*x)[0] - 1e-9);
                let below = xx
                    .iter()
                    .all(|x| staffs[j].band(*x)[1] < staffs[i].band(*x)[0] - 1e-9);
                if !above && !below {
                    return Err(Error::Invalid("overlapping measured local staff bands"));
                }
                let (upper, lower) = if above { (i, j) } else { (j, i) };
                edges.push((
                    upper,
                    lower,
                    2. * (staffs[i].space + staffs[j].space)
                        + 0.05 * staffs[i].space.min(staffs[j].space),
                ));
            }
        }
        // Project only violated partial-order inequalities; disconnected columns
        // retain their independently fitted centres, including equal heights.
        let mut converged = false;
        for _ in 0..10000 {
            let mut change = 0_f64;
            for (a, b, gap) in &edges {
                let violation = (staffs[*a].center + gap - staffs[*b].center).max(0.);
                if violation > 0. {
                    staffs[*a].center -= violation / 2.;
                    staffs[*b].center += violation / 2.;
                    change = change.max(violation);
                }
            }
            for s in &mut staffs {
                let lo = bounds[1] + 2. * s.space + 1e-6;
                let hi = bounds[3] - 2. * s.space - 1e-6;
                if lo >= hi {
                    return Err(Error::Invalid("local staff exceeds vertical canvas"));
                }
                let next = s.center.clamp(lo, hi);
                change = change.max((next - s.center).abs());
                s.center = next;
            }
            if change < 1e-10 {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(Error::Invalid("inconsistent local staff partial order"));
        }
        // A soft extension stops before a different observed band. Curves are
        // endpoint-clamped only within this bounded padding, never page-wide.
        for i in 0..staffs.len() {
            for j in 0..staffs.len() {
                if i == j {
                    continue;
                }
                for side in 0..2 {
                    let edge = if side == 0 {
                        staffs[i].left
                    } else {
                        staffs[i].right
                    };
                    let own = staffs[i].band(edge);
                    let a = staffs[i].support[0].max(staffs[j].left);
                    let b = staffs[i].support[1].min(staffs[j].right);
                    if a > b {
                        continue;
                    }
                    let mut xx = vec![a, b];
                    xx.extend(
                        staffs[j]
                            .lines
                            .iter()
                            .flat_map(|l| l.iter())
                            .filter_map(|p| (p.x > a && p.x < b).then_some(p.x)),
                    );
                    xx.sort_by(f64::total_cmp);
                    xx.dedup();
                    // Include exact crossings of the other top/bottom and this
                    // endpoint band, so a collision cannot hide between samples.
                    let mut roots = vec![];
                    for pair in xx.windows(2) {
                        for k in [0, 4] {
                            for y in own {
                                let (ya, yb) = (
                                    line_at(&staffs[j].lines[k], pair[0]) - y,
                                    line_at(&staffs[j].lines[k], pair[1]) - y,
                                );
                                if ya * yb < 0. {
                                    roots.push(pair[0] + (pair[1] - pair[0]) * ya / (ya - yb));
                                }
                            }
                        }
                    }
                    xx.extend(roots);
                    xx.sort_by(f64::total_cmp);
                    xx.dedup();
                    let target_conflict = overlap(staffs[i].target_band(), staffs[j].target_band());
                    for pair in xx.windows(2) {
                        let mid = (pair[0] + pair[1]) / 2.;
                        if (side == 0 && mid >= edge) || (side == 1 && mid <= edge) {
                            continue;
                        }
                        if target_conflict || overlap(own, staffs[j].band(mid)) {
                            if side == 0 {
                                staffs[i].support[0] = staffs[i].support[0].max(pair[1].min(edge));
                            } else {
                                staffs[i].support[1] = staffs[i].support[1].min(pair[0].max(edge));
                            }
                        }
                    }
                }
            }
        }
        // Conflicting soft bands must not overlap one another either. This
        // leaves a shared identity corridor between side-by-side fragments.
        for i in 0..staffs.len() {
            for j in i + 1..staffs.len() {
                for si in 0..2 {
                    for sj in 0..2 {
                        let ei = if si == 0 {
                            staffs[i].left
                        } else {
                            staffs[i].right
                        };
                        let ej = if sj == 0 {
                            staffs[j].left
                        } else {
                            staffs[j].right
                        };
                        if !overlap(staffs[i].band(ei), staffs[j].band(ej))
                            && !overlap(staffs[i].target_band(), staffs[j].target_band())
                        {
                            continue;
                        }
                        let ai = if si == 0 {
                            [staffs[i].support[0], ei]
                        } else {
                            [ei, staffs[i].support[1]]
                        };
                        let aj = if sj == 0 {
                            [staffs[j].support[0], ej]
                        } else {
                            [ej, staffs[j].support[1]]
                        };
                        if ai[0].max(aj[0]) >= ai[1].min(aj[1]) {
                            continue;
                        }
                        if si == 0 && sj == 0 {
                            if ei < ej {
                                staffs[j].support[0] = staffs[j].support[0].max(ei);
                            } else {
                                staffs[i].support[0] = staffs[i].support[0].max(ej);
                            }
                        } else if si == 1 && sj == 1 {
                            if ei < ej {
                                staffs[i].support[1] = staffs[i].support[1].min(ej);
                            } else {
                                staffs[j].support[1] = staffs[j].support[1].min(ei);
                            }
                        } else {
                            let mid = (ei + ej) / 2.;
                            if si == 1 {
                                staffs[i].support[1] = staffs[i].support[1].min(mid);
                                staffs[j].support[0] = staffs[j].support[0].max(mid);
                            } else {
                                staffs[i].support[0] = staffs[i].support[0].max(mid);
                                staffs[j].support[1] = staffs[j].support[1].min(mid);
                            }
                        }
                    }
                }
            }
        }
        let mut xx = vec![bounds[0], bounds[2]];
        for s in &staffs {
            xx.extend([s.left, s.right, s.support[0], s.support[1]]);
            xx.extend(s.lines.iter().flat_map(|l| l.iter()).map(|p| p.x));
            // Resolve the smooth taper; observed line interpolation remains
            // exact because every measured knot is also retained.
            for (a, b) in [(s.support[0], s.left), (s.right, s.support[1])] {
                let n = ((b - a) / 2.).ceil().max(1.) as usize;
                xx.extend((1..n).map(|i| a + (b - a) * i as f64 / n as f64));
            }
        }
        xx.sort_by(f64::total_cmp);
        xx.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        let columns = xx
            .iter()
            .map(|x| column(&staffs, *x, bounds))
            .collect::<Result<Vec<_>>>()?;
        let mut intervals = vec![];
        let mut min_source_gap = f64::INFINITY;
        let mut max_scale = 0_f64;
        for (k, pair) in xx.windows(2).enumerate() {
            let middle = (pair[0] + pair[1]) / 2.;
            let mut ids = vec![];
            for (id, s) in staffs.iter().enumerate() {
                if middle > s.support[0] && middle < s.support[1] {
                    for line in 0..5 {
                        ids.push((id, line));
                    }
                }
            }
            ids.sort_by(|a, b| {
                line_at(&staffs[a.0].lines[a.1], middle)
                    .total_cmp(&line_at(&staffs[b.0].lines[b.1], middle))
            });
            let mut anchors = vec![[bounds[1]; 4]];
            for (id, line) in ids {
                let a = line_at(&staffs[id].lines[line], pair[0]);
                let b = line_at(&staffs[id].lines[line], pair[1]);
                anchors.push([a, b, columns[k].at(a), columns[k + 1].at(b)]);
            }
            anchors.push([bounds[3]; 4]);
            // Canonicalize roundoff at a deliberately shared band boundary;
            // no tiny negative source/target interval is admissible.
            for i in 1..anchors.len() {
                for endpoint in 0..2 {
                    let source = anchors[i][endpoint] - anchors[i - 1][endpoint];
                    let target = anchors[i][endpoint + 2] - anchors[i - 1][endpoint + 2];
                    if source.abs() < 1e-10 && target.abs() < 1e-8 {
                        anchors[i][endpoint] = anchors[i - 1][endpoint];
                        anchors[i][endpoint + 2] = anchors[i - 1][endpoint + 2];
                    }
                }
            }
            for a in anchors.windows(2) {
                for endpoint in 0..2 {
                    let source = a[1][endpoint] - a[0][endpoint];
                    let target = a[1][endpoint + 2] - a[0][endpoint + 2];
                    if source < 0.
                        || target < 0.
                        || (source == 0. && target != 0.)
                        || (source > 0. && target == 0.)
                    {
                        return Err(Error::Invalid("local support mesh folds"));
                    }
                    if source > 1e-10 {
                        min_source_gap = min_source_gap.min(source);
                        max_scale = max_scale.max(target / source);
                    }
                }
            }
            intervals.push(Interval {
                x: [pair[0], pair[1]],
                anchors,
            });
        }
        let mut model = Self {
            intervals,
            centers: staffs.iter().map(|s| s.center).collect(),
            min_source_gap_pixels: min_source_gap,
            max_vertical_scale: max_scale,
            max_boundary_jump_pixels: 0.,
        };
        model.max_boundary_jump_pixels = model.boundary_jump_max_pixels();
        if !model.max_boundary_jump_pixels.is_finite() || model.max_boundary_jump_pixels > 1e-7 {
            return Err(Error::Invalid(
                "local staff map has an x-boundary discontinuity",
            ));
        }
        Ok(model)
    }
    pub(crate) fn target_centers(&self) -> &[f64] {
        &self.centers
    }
    /// Exact one-sided vertical derivative using the same source bracket as
    /// `forward`. At a y-knot, the upper-index (right-hand) band is selected.
    pub(crate) fn vertical_scale(&self, p: Point) -> f64 {
        if !p.x.is_finite() || !p.y.is_finite() {
            return f64::NAN;
        }
        let i = self
            .intervals
            .partition_point(|i| i.x[1] <= p.x)
            .min(self.intervals.len() - 1);
        self.intervals[i].vertical_scale(p.x, p.y)
    }
    /// Audit both representations of every shared x boundary. Each boundary
    /// map is piecewise affine in y; its maximum discrepancy on the page occurs
    /// at a source/target breakpoint from either adjacent interval. Inverse
    /// jumps are checked as well. No epsilon x-offset or finite triangle hides
    /// a topology seam here.
    pub(crate) fn boundary_jump_max_pixels(&self) -> f64 {
        let mut maximum = 0_f64;
        for pair in self.intervals.windows(2) {
            let x = pair[0].x[1];
            for inverse in [false, true] {
                let offset = if inverse { 2 } else { 0 };
                for y in pair[0]
                    .anchors
                    .iter()
                    .map(|a| a[offset + 1])
                    .chain(pair[1].anchors.iter().map(|a| a[offset]))
                {
                    let a = pair[0].at(x, y, inverse);
                    let b = pair[1].at(x, y, inverse);
                    if !a.is_finite() || !b.is_finite() {
                        return f64::INFINITY;
                    }
                    maximum = maximum.max((a - b).abs());
                }
            }
        }
        maximum
    }
    pub(crate) fn forward(&self, p: Point) -> Point {
        if !p.x.is_finite() || !p.y.is_finite() {
            return p;
        }
        let i = self
            .intervals
            .partition_point(|i| i.x[1] <= p.x)
            .min(self.intervals.len() - 1);
        Point {
            x: p.x,
            y: self.intervals[i].at(p.x, p.y, false),
        }
    }
    pub(crate) fn inverse(&self, p: Point) -> Point {
        if !p.x.is_finite() || !p.y.is_finite() {
            return p;
        }
        let i = self
            .intervals
            .partition_point(|i| i.x[1] <= p.x)
            .min(self.intervals.len() - 1);
        Point {
            x: p.x,
            y: self.intervals[i].at(p.x, p.y, true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn staff(left: f64, right: f64, center: f64, bend: f64) -> [Vec<Point>; 5] {
        std::array::from_fn(|k| {
            (0..21)
                .map(|i| {
                    let x = left + (right - left) * i as f64 / 20.;
                    Point {
                        x,
                        y: center
                            + (k as f64 - 2.) * 10.
                            + bend * ((x - left) / (right - left) * std::f64::consts::PI).sin(),
                    }
                })
                .collect()
        })
    }
    fn check(model: &LocalFlatten, staffs: &[[Vec<Point>; 5]], spaces: &[f64]) {
        assert!(
            model.boundary_jump_max_pixels() < 1e-7,
            "boundary jump {}",
            model.boundary_jump_max_pixels()
        );
        for (i, s) in staffs.iter().enumerate() {
            for (k, line) in s.iter().enumerate() {
                for p in line {
                    assert!(
                        (model.forward(*p).y - model.centers[i] - (k as f64 - 2.) * spaces[i])
                            .abs()
                            < 1e-8
                    );
                }
            }
        }
        for x in (0..800).step_by(3) {
            for y in (0..600).step_by(7) {
                let p = Point {
                    x: x as f64 + 0.123,
                    y: y as f64 + 0.321,
                };
                let q = model.inverse(model.forward(p));
                assert!((q.y - p.y).abs() < 1e-7, "{p:?} {q:?}");
                assert!(
                    model
                        .forward(Point {
                            x: p.x,
                            y: p.y + 0.001
                        })
                        .y
                        > model.forward(p).y
                );
                assert!(model.vertical_scale(p) > 0.);
                let delta = 1e-7;
                let numerical = (model
                    .forward(Point {
                        x: p.x,
                        y: p.y + delta,
                    })
                    .y
                    - model.forward(p).y)
                    / delta;
                assert!((numerical - model.vertical_scale(p)).abs() < 1e-5);
            }
        }
    }
    #[test]
    fn independent_side_by_side_grand_staffs_keep_equal_target_heights() {
        let s = vec![
            staff(50., 320., 170., 14.),
            staff(50., 320., 290., -10.),
            staff(420., 750., 175., -12.),
            staff(420., 750., 295., 8.),
        ];
        let m = LocalFlatten::fit(
            &s,
            &[170., 290., 170., 290.],
            &[10.; 4],
            [0., 0., 800., 600.],
        )
        .unwrap();
        assert_eq!(m.target_centers(), &[170., 290., 170., 290.]);
        check(&m, &s, &[10.; 4]);
    }
    #[test]
    fn fragments_and_staggered_columns_do_not_need_global_staff_order() {
        let s = vec![
            staff(50., 250., 200., 8.),
            staff(300., 500., 202., -8.),
            staff(550., 750., 198., 9.),
            staff(210., 430., 380., 10.),
        ];
        let m = LocalFlatten::fit(
            &s,
            &[200., 200., 200., 380.],
            &[10.; 4],
            [0., 0., 800., 600.],
        )
        .unwrap();
        check(&m, &s, &[10.; 4]);
    }
    #[test]
    fn full_width_curves_flatten_exactly_and_map_is_continuous_at_support_edges() {
        let s = vec![staff(200., 600., 170., 12.), staff(200., 600., 300., -8.)];
        let m = LocalFlatten::fit(&s, &[180., 290.], &[10.; 2], [0., 0., 800., 600.]).unwrap();
        check(&m, &s, &[10.; 2]);
        for x in [120., 200., 600., 680.] {
            for y in [130., 170., 250., 320.] {
                let a = m.forward(Point { x: x - 1e-7, y });
                let b = m.forward(Point { x: x + 1e-7, y });
                assert!((a.y - b.y).abs() < 1e-5);
            }
        }
        assert_eq!(m.forward(Point { x: 20., y: 170. }).y, 170.);
        assert_eq!(m.forward(Point { x: 780., y: 170. }).y, 170.);
    }

    #[test]
    fn padding_stops_before_a_bending_neighbour_instead_of_crossing_it() {
        let a = staff(400., 750., 160., 0.);
        let b = std::array::from_fn(|k| {
            (0..61)
                .map(|i| {
                    let x = 150. + 10. * i as f64;
                    Point {
                        x,
                        y: 160. + (k as f64 - 2.) * 10. + 0.2 * (x - 150.),
                    }
                })
                .collect()
        });
        let curves = vec![a, b];
        let model =
            LocalFlatten::fit(&curves, &[160., 230.], &[10.; 2], [0., 0., 800., 600.]).unwrap();
        check(&model, &curves, &[10.; 2]);
        for x in [300., 350., 400., 450., 750.] {
            for y in [140., 180., 200., 240.] {
                let a = model.forward(Point { x: x - 1e-7, y });
                let b = model.forward(Point { x: x + 1e-7, y });
                assert!((a.y - b.y).abs() < 1e-5, "discontinuous at{x},{y}");
            }
        }
        assert!(model.min_source_gap_pixels > 0.);
        assert!(model.max_vertical_scale.is_finite());
    }
}
