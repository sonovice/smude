//! Candidate-scoped column probing. Kept separate from semantic acceptance.
//! Coordinates are local, y-down for output and bottom-up for pixel probes.
use super::pixels::View;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Line {
    pub left: i32,
    pub right: i32,
    pub top: i32,
    pub bottom: i32,
    pub centre: i32,
}
#[derive(Default)]
pub(super) struct Probe {
    pub lines: Vec<Line>,
    pub tall: bool,
}

/// The first row from `from` whose bit is `set`, `limit` if none precedes it.
fn seek(column: &[u64], from: i32, limit: i32, set: bool) -> i32 {
    let mut row = from;
    while row < limit {
        let word = column[row as usize / 64];
        let rest = if set { word } else { !word } >> (row % 64);
        if rest != 0 {
            return (row + rest.trailing_zeros() as i32).min(limit);
        }
        row = (row / 64 + 1) * 64;
    }
    limit
}

pub(super) fn probe(
    px: &View<'_>,
    space: i32,
    min_length: i32,
    max_width: i32,
    height_limit: Option<i32>,
    capacity: usize,
) -> Probe {
    let (d2, d3) = (
        (space + 1) >> 2,
        if space < 5 { 1 } else { (space + 3) >> 3 },
    );
    let (dx, dy) = (px.width(), px.height());
    // No row above the view holds ink of the component.
    let limit = height_limit.map_or(dy, |limit| limit.min(dy));
    let mut probe = Probe::default();
    let (mut first, mut left, mut lo, mut len, mut best_lo) = (true, 0, 0, 0, 0);
    let line = |lo: i32, len: i32, centre: i32, left: i32, right: i32| Line {
        left,
        right: right + 1,
        top: px.height() - lo - len,
        bottom: px.height() - lo,
        centre,
    };
    // Whether a new line at `centre` replaces (or yields to) the previous one: `Some(true)`
    // replace it, `Some(false)` keep the previous one and drop the new.
    let previous_decision =
        |lines: &[Line], centre: i32, lo: i32, end: i32, len: i32, overlap_test: bool| {
            let p = lines.last()?;
            let (p_lo, p_len) = (px.height() - p.bottom, p.bottom - p.top);
            if centre - p.centre >= d2 || (overlap_test && !(end > p_lo && p_lo + p_len > lo)) {
                return None;
            }
            Some(len > p_len)
        };
    // Per column: the rows with ink on it or beside it, and the rows a run
    // continues through, which also bridges a gap below ink on the column.
    let words = px.column(0).len();
    let (mut near, mut joined) = (vec![0; words], vec![0; words]);
    let mut x = 0;
    while x < dx {
        let next = x + 1;
        let right_probe = if x == dx - 1 { x } else { x + 1 };
        let left_probe = if x == 0 { 0 } else { x - 1 };
        let (left_ink, ink, right_ink) =
            (px.column(left_probe), px.column(x), px.column(right_probe));
        for i in 0..words {
            near[i] = left_ink[i] | ink[i] | right_ink[i];
            let above = ink[i] >> 1 | ink.get(i + 1).map_or(0, |word| word << 63);
            joined[i] = near[i] | above;
        }
        let mut best = 0;
        let mut y = 0;
        loop {
            y = seek(&near, y, limit, true);
            let e = seek(&joined, y, limit, false);
            let run = e - y;
            if run > best && run >= min_length {
                best = run;
                best_lo = y;
            }
            if e >= limit {
                break;
            }
            y = e;
        }
        let end = lo + len;
        let gap = if len == 0 {
            0
        } else if lo > best_lo {
            lo - (best + best_lo)
        } else {
            best_lo - end
        };
        if best > len >> 1 && gap < d2 + d3 && (first || x - left < max_width) {
            let run_end = best_lo + best;
            let (new_lo, new_end) = if len != 0 {
                (lo.min(best_lo), end.max(run_end))
            } else {
                (best_lo, run_end)
            };
            if first {
                left = x;
            }
            first = false;
            (lo, len) = (new_lo, new_end - new_lo);
            x = next;
            continue;
        }
        x = next;
        if len == 0 {
            continue;
        }
        let centre = (left + x - 1) >> 1;
        match previous_decision(&probe.lines, centre, lo, end, len, true) {
            Some(false) => continue,
            Some(true) => {
                probe.lines.pop();
            }
            None => {}
        }
        first = true;
        if x - 1 - left > max_width {
            (lo, len) = (0, 0);
            continue;
        }
        probe.lines.push(line(lo, len, centre, left, x - 1));
        if probe.lines.len() == capacity {
            probe.lines.clear();
            return probe;
        }
        probe.tall |= len > dy / 2;
        (lo, len) = (0, 0);
    }
    if len == 0 {
        return probe;
    }
    let centre = (left + x) >> 1;
    match previous_decision(&probe.lines, centre, lo, lo + len, len, false) {
        Some(false) => return probe,
        Some(true) => {
            probe.lines.pop();
        }
        None => {}
    }
    if dx - left > max_width {
        return probe;
    }
    probe.lines.push(line(lo, len, centre, left, dx));
    if probe.lines.len() == capacity {
        probe.lines.clear();
        return probe;
    }
    probe
}
