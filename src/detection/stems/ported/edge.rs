//! Whether a vertical line stands free like a stem: clear rows along it.
//!
//! A stem is a thin vertical stroke: along most of its rows the candidate's ink around it is
//! no wider than a stroke, and stretches of it have nothing on either side. A vertical line
//! through a solid shape (a head, a beam, a clef) has neither. The measurements here count
//! such rows ([`narrow_rows`], [`clear_run`]) and decide from them whether a line is a stem
//! ([`stem_edge`]).

use super::space::percent_of;
use super::types::SpacePercent;

use super::super::pixels::View as CandidateView;
use super::metrics::NoteMetrics;
use super::types::StemLine;

/// Below this height (in staff spaces) a candidate is short, and its narrow rows are only
/// counted when the ink beside them is not a wide shape.
const SHORT_CANDIDATE_SPACES: i32 = 4;
/// The widest ink run across a stem row that still counts as narrow, by default: this share of
/// the wide space plus one, plus two at low resolution.
const NARROW_ROW_WIDTH: SpacePercent = SpacePercent(48);
/// Extra width of a narrow row at low resolution. Unit: pixels.
const LOW_RESOLUTION_EXTRA_WIDTH: i32 = 2;
/// Columns left of the line a narrow-row probe never goes past (the probe stops at the
/// candidate's left edge, or this far left of the line, whichever is further left).
const PROBE_LEFT_MARGIN: i32 = 2;

/// Rows a split candidate gets for free (it lost ink at the cut).
const CHOPPED_ALLOWANCE: SpacePercent = SpacePercent(15);
/// The least narrow rows of a stem.
const MIN_NARROW_ROWS: SpacePercent = SpacePercent(160);
/// Candidates lower than this need fewer narrow rows.
const SHORT_STEM_CANDIDATE: SpacePercent = SpacePercent(308);
/// The least narrow rows on a short candidate (and on a short grace note at low resolution).
const MIN_NARROW_ROWS_SHORT: SpacePercent = SpacePercent(145);
const MIN_NARROW_ROWS_SHORT_GRACE_LOW: SpacePercent = SpacePercent(125);
/// Narrow rows (share of the line) above which a long-run line is a stem, by resolution:
/// normal, low, lower.
const NARROW_SHARE_LONG: [SpacePercent; 3] = [SpacePercent(40), SpacePercent(39), SpacePercent(38)];
/// Narrow rows (share of the line) above which a line with a long run is a stem.
const NARROW_SHARE: [SpacePercent; 3] = [SpacePercent(31), SpacePercent(30), SpacePercent(29)];
/// The longest narrow run that makes a line a stem with enough narrow rows.
const LONG_NARROW_RUN: SpacePercent = SpacePercent(75);
/// A second count with a wider row: narrow rows and longest run that make a line a stem.
const WIDER_NARROW_ROWS: SpacePercent = SpacePercent(240);
const WIDER_LONG_RUN: SpacePercent = SpacePercent(85);
/// The share of the line its narrow rows must exceed in the last test.
const LAST_NARROW_SHARE: SpacePercent = SpacePercent(30);
/// A grace note's longest run and narrow rows are scaled up by these.
const GRACE_RUN_SCALE: SpacePercent = SpacePercent(230);
const GRACE_ROWS_SCALE: SpacePercent = SpacePercent(145);
/// A tailed stem gets one fifth of the bold space on its longest run and one quarter on its
/// narrow rows.
const TAILED_RUN_DIVISOR: i32 = 5;
const TAILED_ROWS_DIVISOR: i32 = 4;

/// The top rows of a candidate an untailed spike's ink must keep narrow: at least this many.
const SPIKE_MIN_ROWS: i32 = 13;
/// The widest an untailed spike's top may be: at least this many columns.
const SPIKE_MIN_WIDTH: i32 = 4;

/// A whole as a percentage.
const FULL_PERCENT: i32 = 100;

/// The longest run of rows along a line that is clear on both sides.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClearRun {
    /// Its length in rows (0: none).
    pub length: i32,
    /// Its first row (the line's first row if there is none).
    pub start: i32,
}

/// The rows of `line` where the candidate's ink across the line is narrow.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NarrowRows {
    /// How many rows are narrow.
    pub total: i32,
    /// The length of the first run of narrow rows (0 if the line starts with them and they
    /// never end, or if there are none).
    pub first_run: i32,
    /// The length of the longest run.
    pub longest_run: i32,
}

/// Whether a line is a free-standing stem, and why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StemEdge {
    /// The line is part of a solid shape.
    Solid,
    /// The line is a stem.
    Clear,
    /// The line is a stem only by its narrow rows and a long run, the weakest acceptance
    /// (the candidate is marked so later tests can be stricter).
    ClearByLongRun,
}

impl StemEdge {
    /// Whether the line is a stem.
    pub fn is_clear(self) -> bool {
        self != StemEdge::Solid
    }
}

/// The longest run of `line`'s rows where columns `x1` and `x2` both hold none of the
/// candidate's ink (with `any_symbol`: no symbol or staff-line ink at all). The row that ends a
/// run is not the start of the next.
pub fn clear_run(
    view: &CandidateView<'_>,
    line: &StemLine,
    x1: i32,
    x2: i32,
    any_symbol: bool,
) -> ClearRun {
    let clear = |x: i32, y: i32| {
        if any_symbol {
            !view.any_symbol_ink(x, y)
        } else {
            !view.own(x, y)
        }
    };
    let end = line.end();
    let mut best = ClearRun {
        length: 0,
        start: line.y,
    };
    let mut y = line.y;
    while y < end {
        let mut n = 0;
        while y + n < end && clear(x1, y + n) && clear(x2, y + n) {
            n += 1;
        }
        if best.length < n {
            best = ClearRun {
                length: n,
                start: y,
            };
        }
        y += n + 1;
    }
    best
}

/// One row of a narrow-row probe: whether the candidate has ink there, the width of its run
/// through the line (two more than the pixels beside the start), and the columns where the run
/// stopped on either side.
struct RowRun {
    found: bool,
    width: i32,
    stop_left: i32,
    stop_right: i32,
}

/// The candidate's run through `line` on row `y`: from the line's column (or the first own
/// pixel between its left and right columns) outwards until a pixel that is neither own ink nor
/// a barrier, never left of `left_limit` and never right of the candidate.
fn row_run(view: &CandidateView<'_>, line: &StemLine, left_limit: i32, y: i32) -> RowRun {
    let mut start = if view.own(line.x, y) {
        line.x
    } else {
        line.left
    };
    while !view.own(start, y) && start < line.right {
        start += 1;
    }
    if !view.own(start, y) {
        start = line.x;
    }
    let mut found = view.own(start, y);
    let mut width = 2;
    let mut left = start - 1;
    while left > left_limit && !view.clear_of_barrier(left, y) {
        found = true;
        width += 1;
        left -= 1;
    }
    let mut right = start + 1;
    while right < view.width() && !view.clear_of_barrier(right, y) {
        found = true;
        width += 1;
        right += 1;
    }
    RowRun {
        found,
        width,
        stop_left: left,
        stop_right: right,
    }
}

/// The default narrow-row width of `metrics`' staff.
pub fn default_narrow_width(metrics: &NoteMetrics) -> i32 {
    let low = if metrics.resolution.low() {
        LOW_RESOLUTION_EXTRA_WIDTH
    } else {
        0
    };
    percent_of(metrics.space.wide, NARROW_ROW_WIDTH) + low + 1
}

/// Counts the rows of `line` where the candidate's ink across it is narrower than `width`
/// (0: [`default_narrow_width`]). On a short candidate a row counts only if the ink beside the
/// run is not part of a wide shape: when every counted row has ink a quarter space beyond the
/// run's left (or right) end, nothing counts.
pub fn narrow_rows(
    view: &CandidateView<'_>,
    metrics: &NoteMetrics,
    line: &StemLine,
    width: i32,
) -> NarrowRows {
    let short = view.height() < metrics.space.space * SHORT_CANDIDATE_SPACES;
    let width = if width == 0 {
        default_narrow_width(metrics)
    } else {
        width
    };
    let left_limit = (line.x - PROBE_LEFT_MARGIN).min(0);
    let offset = metrics.space.quarter;
    let mut rows = NarrowRows::default();
    let (mut run, mut wide_left, mut wide_right) = (0, 0, 0);
    for y in line.y..line.end() {
        let r = row_run(view, line, left_limit, y);
        if r.found && r.width < width {
            rows.total += 1;
            run += 1;
            rows.longest_run = rows.longest_run.max(run);
            if short {
                wide_left += view.own(r.stop_left - offset, y) as i32;
                wide_right += view.own(r.stop_right + offset, y) as i32;
            }
        } else {
            if rows.first_run == 0 {
                rows.first_run = run;
            }
            run = 0;
        }
    }
    if short && (wide_left == rows.total || wide_right == rows.total) {
        rows.total = 0;
    }
    rows
}

/// What [`stem_edge`] knows about the line besides its pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StemEdgeContext {
    /// The candidate was split off a larger one.
    pub chopped: bool,
    /// The stem carries tails.
    pub tailed: bool,
    /// The note is a grace note.
    pub grace: bool,
    /// The stem carries tremolo strokes.
    pub tremolo: bool,
}

/// Decides whether `line` is a free-standing stem from its narrow rows: enough of them in
/// total and in one run, with allowances for split candidates, tails, tremolo strokes, short
/// candidates and grace notes; a second count with a wider row decides borderline lines.
pub fn stem_edge(
    view: &CandidateView<'_>,
    metrics: &NoteMetrics,
    line: &StemLine,
    context: StemEdgeContext,
) -> StemEdge {
    let space = &metrics.space;
    let res = metrics.resolution;
    let width =
        percent_of(space.wide, NARROW_ROW_WIDTH) + (!res.high()) as i32 + res.lower() as i32 + 1;
    let first = narrow_rows(view, metrics, line, width);
    let mut longest = first.longest_run;
    let mut total = first.total;
    if context.chopped {
        total += metrics.percent(CHOPPED_ALLOWANCE);
    }
    let rows = if context.tailed {
        longest += space.bold / TAILED_RUN_DIVISOR;
        total + space.bold / TAILED_ROWS_DIVISOR
    } else if context.tremolo {
        total + space.eighth
    } else {
        total - space.eighth
    };
    let short = view.height() < metrics.percent(SHORT_STEM_CANDIDATE);
    let least = metrics.percent(match (short, context.grace && res.low()) {
        (false, _) => MIN_NARROW_ROWS,
        (true, true) => MIN_NARROW_ROWS_SHORT_GRACE_LOW,
        (true, false) => MIN_NARROW_ROWS_SHORT,
    });
    let band = if !res.low() {
        0
    } else if res.lower() {
        2
    } else {
        1
    };
    let (rows, scaled_longest) = if context.grace {
        (
            percent_of(rows, GRACE_ROWS_SCALE),
            percent_of(longest, GRACE_RUN_SCALE),
        )
    } else {
        (rows, longest)
    };
    longest = scaled_longest;
    if rows < least && longest < least - space.eighth {
        return StemEdge::Solid;
    }
    if rows > percent_of(line.len, NARROW_SHARE_LONG[band])
        && longest >= metrics.percent(LONG_NARROW_RUN)
    {
        return StemEdge::Clear;
    }
    if rows > percent_of(line.len, NARROW_SHARE[band]) {
        if longest > metrics.parse.min_clear_stem_run
            || first.first_run > metrics.parse.min_clear_stem_run / 2
        {
            return StemEdge::Clear;
        }
        let wider = narrow_rows(view, metrics, line, width + 1);
        if wider.total > metrics.percent(WIDER_NARROW_ROWS)
            && wider.longest_run > metrics.percent(WIDER_LONG_RUN)
        {
            return StemEdge::Clear;
        }
        longest = wider.longest_run;
    }
    if rows <= percent_of(line.len, LAST_NARROW_SHARE) || longest < metrics.percent(LONG_NARROW_RUN)
    {
        return StemEdge::Solid;
    }
    StemEdge::ClearByLongRun
}

/// Whether the top rows of the candidate (at least 12, a space less one) hold its ink only
/// within a narrow band of columns: a spike without a tail. Without ink there, it is one.
pub fn untailed_spike(view: &CandidateView<'_>, metrics: &NoteMetrics) -> bool {
    let rows = metrics.space.space.max(SPIKE_MIN_ROWS) - 1;
    let inked = |x: i32| (0..rows).any(|y| view.own(x, y));
    let Some(left) = (0..view.width()).find(|&x| inked(x)) else {
        return true;
    };
    let right = (left + 1..=view.width()).rev().find(|&x| inked(x));
    match right {
        Some(right) => right - left <= metrics.space.quarter.max(SPIKE_MIN_WIDTH),
        None => false,
    }
}

/// How far `value` has moved into the band from `threshold` to `limit`, in percent: 0 before
/// the threshold, 100 at or past the limit (the band may run either way).
pub fn band_encroachment(value: i32, threshold: i32, limit: i32) -> i32 {
    let (num, den) = if threshold < limit {
        if value < threshold {
            return 0;
        }
        if limit <= value {
            return FULL_PERCENT;
        }
        (value - threshold, limit - threshold)
    } else {
        if threshold < value {
            return 0;
        }
        if value <= limit {
            return FULL_PERCENT;
        }
        (threshold - value, threshold - limit)
    };
    num * FULL_PERCENT / den
}
