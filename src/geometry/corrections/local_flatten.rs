//! Locally supported monotone staff flattening. Horizontal coordinates never
//! change. Only staffs whose measured x domains overlap constrain their order.
use crate::geometry::Point;
use crate::{Error, Result};

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
/// `line_at` for non-decreasing queries: the knot bracket only ever advances.
struct LineCursor<'a> {
    line: &'a [Point],
    hi: usize,
}
impl<'a> LineCursor<'a> {
    fn new(line: &'a [Point]) -> Self {
        Self { line, hi: 1 }
    }
    fn at(&mut self, x: f64) -> f64 {
        let line = self.line;
        if x <= line[0].x {
            return line[0].y;
        }
        if x >= line[line.len() - 1].x {
            return line[line.len() - 1].y;
        }
        while line[self.hi].x <= x {
            self.hi += 1;
        }
        let (a, b) = (line[self.hi - 1], line[self.hi]);
        a.y + (b.y - a.y) * (x - a.x) / (b.x - a.x)
    }
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

#[derive(Clone, Debug, Default)]
struct Column {
    source: Vec<f64>,
    target: Vec<f64>,
}
impl Column {
    /// `hint` is the bracket of an earlier query; ascending queries then scan
    /// forward instead of searching.
    fn at(&self, y: f64, hint: &mut usize) -> f64 {
        let last = self.source.len() - 2;
        let mut i = (*hint).min(last);
        if self.source[i] <= y {
            while i < last && self.source[i + 1] <= y {
                i += 1;
            }
        } else {
            i = self
                .source
                .partition_point(|v| *v <= y)
                .saturating_sub(1)
                .min(last);
        }
        *hint = i;
        self.target[i]
            + (y - self.source[i]) / (self.source[i + 1] - self.source[i])
                * (self.target[i + 1] - self.target[i])
    }
}

/// Band of an earlier query in a nearby interval. Neighbouring pixels almost
/// always share it, so it is tried before the search; any value is valid.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Hint(usize);

/// The band of `len` non-decreasing coordinates that holds `y`; band `i` spans
/// `coordinate(i)..coordinate(i + 1)`.
fn band(len: usize, y: f64, hint: &mut Hint, coordinate: impl Fn(usize) -> f64) -> usize {
    let last = len - 2;
    // A band that strictly contains y has positive width and is exactly
    // the one the search below selects.
    let contains = |i: usize| i <= last && coordinate(i) <= y && y < coordinate(i + 1);
    let i = if contains(hint.0) {
        hint.0
    } else if contains(hint.0 + 1) {
        hint.0 + 1
    } else {
        let (mut i, mut end) = (0, len);
        while i < end {
            let middle = i + (end - i) / 2;
            if coordinate(middle) <= y {
                i = middle + 1;
            } else {
                end = middle;
            }
        }
        let mut i = i.saturating_sub(1).min(last);
        // An anchor may be born at an existing anchor. The zero-width band
        // exists only at the exact interval endpoint and has zero target width.
        while i < last && coordinate(i + 1) - coordinate(i) == 0. {
            i += 1;
        }
        while i > 0 && coordinate(i + 1) - coordinate(i) == 0. {
            i -= 1;
        }
        i
    };
    hint.0 = i;
    i
}

/// An interval at one x: [source, target] of every anchor there.
#[derive(Default)]
struct Cut(Vec<[f64; 2]>);
impl Cut {
    /// `Interval::at` at this x.
    fn at(&self, y: f64, inverse: bool, hint: &mut Hint) -> f64 {
        let (from, to) = (usize::from(inverse), usize::from(!inverse));
        let i = band(self.0.len(), y, hint, |i| self.0[i][from]);
        let (a, b) = (self.0[i], self.0[i + 1]);
        a[to] + (y - a[from]) / (b[from] - a[from]) * (b[to] - a[to])
    }
}

#[derive(Clone, Copy)]
struct Interval<'a> {
    x: [f64; 2],
    /// This interval's anchors are `len` elements `stride` apart. Each is
    /// [source_left, source_right, target_left, target_right].
    anchors: &'a [[f64; 4]],
    stride: usize,
    len: usize,
}
impl<'a> Interval<'a> {
    fn anchors(&self) -> impl Iterator<Item = &'a [f64; 4]> + use<'a> {
        self.anchors.iter().step_by(self.stride).take(self.len)
    }
    /// Where `x` lies between this interval's ends.
    fn fraction(&self, x: f64) -> f64 {
        ((x - self.x[0]) / (self.x[1] - self.x[0])).clamp(0., 1.)
    }
    fn segment(&self, x: f64, y: f64, inverse: bool, hint: &mut Hint) -> [f64; 4] {
        let t = self.fraction(x);
        let coordinate = |i: usize, target: bool| {
            let a = &self.anchors[i * self.stride];
            let k = if target { 2 } else { 0 };
            a[k] + t * (a[k + 1] - a[k])
        };
        let i = band(self.len, y, hint, |i| coordinate(i, inverse));
        [
            coordinate(i, inverse),
            coordinate(i + 1, inverse),
            coordinate(i, !inverse),
            coordinate(i + 1, !inverse),
        ]
    }
    /// Every anchor at `x`, formed exactly as `segment` forms them.
    fn cut(&self, x: f64, cut: &mut Cut) {
        let t = self.fraction(x);
        cut.0.clear();
        cut.0.extend(
            self.anchors()
                .map(|a| [a[0] + t * (a[1] - a[0]), a[2] + t * (a[3] - a[2])]),
        );
    }
    fn at(&self, x: f64, y: f64, inverse: bool, hint: &mut Hint) -> f64 {
        let [a, b, c, d] = self.segment(x, y, inverse, hint);
        c + (y - a) / (b - a) * (d - c)
    }
    fn vertical_scale(&self, x: f64, y: f64) -> f64 {
        let [a, b, c, d] = self.segment(x, y, false, &mut Hint::default());
        (d - c) / (b - a)
    }
}

/// Interval `i` spans `knots[i]..knots[i + 1]`. Its anchor `h` of `counts[i]`
/// is `anchors[h * counts.len() + i]`: the same band of neighbouring intervals
/// is contiguous, so nearby pixels read a few short runs of this large table.
#[derive(Clone, Debug)]
pub(crate) struct LocalFlatten {
    /// Padded with infinities to a whole search `window` past every interval.
    knots: Vec<f64>,
    counts: Vec<u32>,
    anchors: Vec<[f64; 4]>,
    /// Per uniform x bucket, the number of intervals ending in an earlier one.
    buckets: Vec<u32>,
    buckets_per_pixel: f64,
    /// A power of two above the number of intervals ending in any one bucket.
    window: usize,
    centers: Vec<f64>,
    pub(crate) min_source_gap_pixels: f64,
    pub(crate) max_vertical_scale: f64,
    pub(crate) max_boundary_jump_pixels: f64,
}

/// Reused buffers for fitting one column per x-knot.
#[derive(Default)]
struct ColumnFit {
    hard: Vec<(f64, f64)>,
    base: Column,
    anchors: Vec<(f64, f64, bool)>,
    fixed: Vec<usize>,
    blocks: Vec<(usize, usize, f64, usize)>,
}
impl ColumnFit {
    /// `heights` holds every staff line at `x`, five per staff in staff order.
    fn fit(
        &mut self,
        staffs: &[StaffCurves],
        heights: &[f64],
        x: f64,
        bounds: [f64; 4],
        column: &mut Column,
    ) -> Result<()> {
        let Self {
            hard,
            base,
            anchors,
            fixed,
            blocks,
        } = self;
        hard.clear();
        hard.extend([(bounds[1], bounds[1]), (bounds[3], bounds[3])]);
        for (s, heights) in staffs.iter().zip(heights.chunks_exact(5)) {
            if s.measured(x) {
                for (k, source) in heights.iter().enumerate() {
                    hard.push((*source, s.center + (k as f64 - 2.) * s.space));
                }
            }
        }
        hard.sort_by(|a, b| a.0.total_cmp(&b.0));
        if hard
            .windows(2)
            .any(|w| w[1].0 - w[0].0 <= 1e-10 || w[1].1 - w[0].1 <= 1e-10)
        {
            return Err(Error::Invalid("local measured staff anchors cross"));
        }
        base.source.clear();
        base.source.extend(hard.iter().map(|p| p.0));
        base.target.clear();
        base.target.extend(hard.iter().map(|p| p.1));
        anchors.clear();
        anchors.extend(hard.iter().map(|p| (p.0, p.1, true)));
        for (s, heights) in staffs.iter().zip(heights.chunks_exact(5)) {
            if s.measured(x) || x < s.support[0] || x > s.support[1] {
                continue;
            }
            let weight = s.weight(x);
            // A newly appearing zero-weight band must be inserted into the map
            // already induced by every other band, not force its hard-only base.
            // Omitting it here makes both adjacent interval representations agree.
            if weight == 0. {
                continue;
            }
            let mut hint = 0;
            for (k, source) in heights.iter().enumerate() {
                if *source <= bounds[1] || *source >= bounds[3] {
                    continue;
                }
                let target = base.at(*source, &mut hint);
                anchors.push((
                    *source,
                    target + weight * (s.center + (k as f64 - 2.) * s.space - target),
                    false,
                ));
            }
        }
        anchors.sort_by(|a, b| a.0.total_cmp(&b.0).then(b.2.cmp(&a.2)));
        // Zero-support endpoints may coincide with another band's boundary.
        anchors.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-10);
        fixed.clear();
        fixed.extend(
            anchors
                .iter()
                .enumerate()
                .filter_map(|(i, a)| a.2.then_some(i)),
        );
        for pair in fixed.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let alpha =
                0.05_f64.min(0.25 * (anchors[b].1 - anchors[a].1) / (anchors[b].0 - anchors[a].0));
            let lower = anchors[a].1 - alpha * anchors[a].0;
            let upper = anchors[b].1 - alpha * anchors[b].0;
            blocks.clear();
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
            for &(lo, hi, sum, n) in blocks.iter() {
                for anchor in anchors.iter_mut().take(hi + 1).skip(lo) {
                    anchor.1 = sum / n as f64 + alpha * anchor.0;
                }
            }
        }
        column.source.clear();
        column.source.extend(anchors.iter().map(|a| a.0));
        column.target.clear();
        column.target.extend(anchors.iter().map(|a| a.1));
        Ok(())
    }
}

/// Whether `upper` stays above `lower` at `a`, `b` and every knot of `lines`
/// between them.
fn ordered(
    upper: &StaffCurves,
    lower: &StaffCurves,
    lines: &[&[Vec<Point>; 5]; 2],
    [a, b]: [f64; 2],
) -> bool {
    lines.iter().flat_map(|lines| lines.iter()).all(|line| {
        let mut bottom = LineCursor::new(&upper.lines[4]);
        let mut top = LineCursor::new(&lower.lines[0]);
        let knots = line.iter().map(|p| p.x).filter(|x| *x >= a && *x <= b);
        std::iter::once(a)
            .chain(knots)
            .chain(std::iter::once(b))
            .all(|x| bottom.at(x) < top.at(x) - 1e-9)
    })
}

impl LocalFlatten {
    fn new(knots: Vec<f64>, counts: Vec<u32>, anchors: Vec<[f64; 4]>, centers: Vec<f64>) -> Self {
        let intervals = counts.len();
        let mut model = Self {
            buckets: vec![0; 2 * intervals + 1],
            buckets_per_pixel: (2 * intervals) as f64 / (knots[intervals] - knots[0]),
            window: 1,
            knots,
            counts,
            anchors,
            centers,
            min_source_gap_pixels: f64::INFINITY,
            max_vertical_scale: 0.,
            max_boundary_jump_pixels: 0.,
        };
        // Intervals ending in an earlier bucket lie left of every x in a
        // bucket, those ending in a later one right of it.
        for k in 1..=intervals {
            let bucket = model.bucket(model.knots[k]);
            model.buckets[bucket + 1] += 1;
        }
        let fullest = model.buckets.iter().fold(0, |a, b| a.max(*b));
        for b in 1..model.buckets.len() {
            model.buckets[b] += model.buckets[b - 1];
        }
        model.window = (fullest as usize + 1).next_power_of_two();
        model.knots.resize(intervals + model.window, f64::INFINITY);
        model
    }
    pub(crate) fn identity(bounds: [f64; 4]) -> Self {
        Self {
            min_source_gap_pixels: bounds[3] - bounds[1],
            max_vertical_scale: 1.,
            ..Self::new(
                vec![bounds[0], bounds[2]],
                vec![2],
                vec![[bounds[1]; 4], [bounds[3]; 4]],
                vec![],
            )
        }
    }
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
                let lines = [&staffs[i].lines, &staffs[j].lines];
                let above = ordered(&staffs[i], &staffs[j], &lines, [a, b]);
                if !above && !ordered(&staffs[j], &staffs[i], &lines, [a, b]) {
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
                    // Only windows on the soft side of `edge` can trim. Those
                    // end at the first sample at or beyond it; one more window
                    // is kept on the right side, where a crossing may round up
                    // onto the last sample at or before it.
                    let mut xx = vec![a, b];
                    for line in &staffs[j].lines {
                        let (lo, hi) = if side == 0 {
                            (0, line.partition_point(|p| p.x < edge) + 1)
                        } else {
                            (
                                line.partition_point(|p| p.x <= edge).saturating_sub(2),
                                line.len(),
                            )
                        };
                        xx.extend(
                            line.iter()
                                .take(hi)
                                .skip(lo)
                                .filter_map(|p| (p.x > a && p.x < b).then_some(p.x)),
                        );
                    }
                    xx.sort_by(f64::total_cmp);
                    xx.dedup();
                    if side == 0 {
                        xx.truncate(xx.partition_point(|x| *x < edge) + 1);
                    } else {
                        xx.drain(..xx.partition_point(|x| *x <= edge).saturating_sub(2));
                    }
                    // Include exact crossings of the other top/bottom and this
                    // endpoint band, so a collision cannot hide between samples.
                    let band: Vec<[f64; 2]> = {
                        let mut top = LineCursor::new(&staffs[j].lines[0]);
                        let mut bottom = LineCursor::new(&staffs[j].lines[4]);
                        xx.iter().map(|x| [top.at(*x), bottom.at(*x)]).collect()
                    };
                    let mut roots = vec![];
                    for (pair, band) in xx.windows(2).zip(band.windows(2)) {
                        for (a, b) in band[0].into_iter().zip(band[1]) {
                            for y in own {
                                let (ya, yb) = (a - y, b - y);
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
        // The knots are swept once, left to right: every staff line is sampled
        // at each knot and interval midpoint, and only the two columns bounding
        // the current interval are kept.
        let mut cursors: Vec<_> = staffs
            .iter()
            .flat_map(|s| s.lines.iter().map(|line| LineCursor::new(line)))
            .collect();
        let mut sample = |x: f64, heights: &mut Vec<f64>| {
            heights.clear();
            heights.extend(cursors.iter_mut().map(|cursor| cursor.at(x)));
        };
        let mut fit = ColumnFit::default();
        let (mut left, mut middle, mut right) = (vec![], vec![], vec![]);
        let (mut left_column, mut right_column) = (Column::default(), Column::default());
        sample(xx[0], &mut left);
        fit.fit(&staffs, &left, xx[0], bounds, &mut left_column)?;
        // Sized for every staff in every interval; rows no interval reaches
        // are dropped below without ever having been written.
        let intervals = xx.len() - 1;
        let mut table = vec![[0.; 4]; intervals * (5 * staffs.len() + 2)];
        let mut counts = Vec::with_capacity(intervals);
        let mut anchors = vec![];
        let mut order = vec![];
        let mut folds = false;
        let mut min_source_gap = f64::INFINITY;
        let mut max_scale = 0_f64;
        for (k, pair) in xx.windows(2).enumerate() {
            let x = (pair[0] + pair[1]) / 2.;
            sample(x, &mut middle);
            sample(pair[1], &mut right);
            fit.fit(&staffs, &right, pair[1], bounds, &mut right_column)?;
            order.clear();
            for (id, s) in staffs.iter().enumerate() {
                if x > s.support[0] && x < s.support[1] {
                    order.extend(5 * id..5 * id + 5);
                }
            }
            order.sort_by(|a, b| middle[*a].total_cmp(&middle[*b]));
            anchors.clear();
            anchors.push([bounds[1]; 4]);
            let mut hints = [0; 2];
            for line in &order {
                let (a, b) = (left[*line], right[*line]);
                anchors.push([
                    a,
                    b,
                    left_column.at(a, &mut hints[0]),
                    right_column.at(b, &mut hints[1]),
                ]);
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
                    // Reported only once every column has been fitted.
                    folds |= source < 0.
                        || target < 0.
                        || (source == 0. && target != 0.)
                        || (source > 0. && target == 0.);
                    if source > 1e-10 {
                        min_source_gap = min_source_gap.min(source);
                        max_scale = max_scale.max(target / source);
                    }
                }
            }
            for (h, anchor) in anchors.iter().enumerate() {
                table[h * intervals + k] = *anchor;
            }
            counts.push(anchors.len() as u32);
            std::mem::swap(&mut left, &mut right);
            std::mem::swap(&mut left_column, &mut right_column);
        }
        if folds {
            return Err(Error::Invalid("local support mesh folds"));
        }
        table.truncate(intervals * counts.iter().fold(0, |a, b| a.max(*b)) as usize);
        table.shrink_to_fit();
        let mut model = Self {
            min_source_gap_pixels: min_source_gap,
            max_vertical_scale: max_scale,
            ..Self::new(xx, counts, table, staffs.iter().map(|s| s.center).collect())
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
    /// Saturating and monotone in `x`, so equal buckets bracket the knots.
    fn bucket(&self, x: f64) -> usize {
        (((x - self.knots[0]) * self.buckets_per_pixel) as usize).min(self.buckets.len() - 2)
    }
    fn interval(&self, i: usize) -> Interval<'_> {
        Interval {
            x: [self.knots[i], self.knots[i + 1]],
            anchors: &self.anchors[i..],
            stride: self.counts.len(),
            len: self.counts[i] as usize,
        }
    }
    /// The first interval ending right of `x`, or the last one.
    fn interval_at(&self, x: f64) -> Interval<'_> {
        // How many intervals a bucket holds is unpredictable from one pixel to
        // the next, so its whole window is bisected without a branch.
        let mut i = self.buckets[self.bucket(x)] as usize;
        let mut step = self.window / 2;
        while step > 0 {
            i += step * usize::from(self.knots[i + step] <= x);
            step /= 2;
        }
        self.interval(i.min(self.counts.len() - 1))
    }
    /// Exact one-sided vertical derivative using the same source bracket as
    /// `forward`. At a y-knot, the upper-index (right-hand) band is selected.
    pub(crate) fn vertical_scale(&self, p: Point) -> f64 {
        if !p.x.is_finite() || !p.y.is_finite() {
            return f64::NAN;
        }
        self.interval_at(p.x).vertical_scale(p.x, p.y)
    }
    /// Audit both representations of every shared x boundary. Each boundary
    /// map is piecewise affine in y; its maximum discrepancy on the page occurs
    /// at a source/target breakpoint from either adjacent interval. Inverse
    /// jumps are checked as well. No epsilon x-offset or finite triangle hides
    /// a topology seam here.
    pub(crate) fn boundary_jump_max_pixels(&self) -> f64 {
        // Boundaries are audited independently, and a maximum does not depend
        // on the order of its operands.
        const CHUNK: usize = 64;
        let boundaries: Vec<_> = (1..self.counts.len())
            .step_by(CHUNK)
            .map(|start| start..(start + CHUNK).min(self.counts.len()))
            .collect();
        crate::parallel::map(&boundaries, |boundaries| {
            self.boundary_jump_over(boundaries.clone())
        })
        .into_iter()
        .fold(0., f64::max)
    }
    /// `boundary_jump_max_pixels` over the boundaries at these knots.
    fn boundary_jump_over(&self, boundaries: std::ops::Range<usize>) -> f64 {
        let mut maximum = 0_f64;
        // Each boundary is queried at every anchor of both intervals, so
        // their anchors there are formed once.
        let mut cuts = [Cut::default(), Cut::default()];
        for k in boundaries {
            let pair = [self.interval(k - 1), self.interval(k)];
            let x = self.knots[k];
            for (interval, cut) in pair.iter().zip(&mut cuts) {
                interval.cut(x, cut);
            }
            for inverse in [false, true] {
                let offset = if inverse { 2 } else { 0 };
                let mut hints = [Hint::default(); 2];
                for y in pair[0]
                    .anchors()
                    .map(|a| a[offset + 1])
                    .chain(pair[1].anchors().map(|a| a[offset]))
                {
                    let a = cuts[0].at(y, inverse, &mut hints[0]);
                    let b = cuts[1].at(y, inverse, &mut hints[1]);
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
        Point {
            x: p.x,
            y: self
                .interval_at(p.x)
                .at(p.x, p.y, false, &mut Hint::default()),
        }
    }
    pub(crate) fn inverse(&self, p: Point) -> Point {
        self.inverse_near(p, &mut Hint::default())
    }
    /// `inverse`, starting the band search from `hint` and updating it.
    pub(crate) fn inverse_near(&self, p: Point, hint: &mut Hint) -> Point {
        if !p.x.is_finite() || !p.y.is_finite() {
            return p;
        }
        Point {
            x: p.x,
            y: self.interval_at(p.x).at(p.x, p.y, true, hint),
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
