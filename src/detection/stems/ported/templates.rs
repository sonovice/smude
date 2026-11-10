//! Pixel templates of open (hollow) note heads, one pair of halves per staff size.
//!
//! An open head is matched half by half. Each half is a [`HeadMask`]: the points that must be
//! ink (its two slanted outline strokes) and the points that must be paper (the hollow
//! inside); the right half also has a *wing*, a small box beyond the head that must be paper
//! too. Each half comes in two sizes: [`HalfHeadTemplate::narrow`], for a head drawn at
//! nine tenths of the staff space, and [`HalfHeadTemplate::full`].
//!
//! The strokes are walked row by row with fixed-point slopes (sixteenths of a pixel), so the
//! points depend only on the staff space. Points are offsets from the template origin:
//! `x` columns to the right and `up` rows upwards (the matching measures heights, see
//! the bottom-up raster frame).

use super::types::Percent;
use super::types::{Px, SpacePercent, StaffSpace};

/// Small staves (cue or ossia staves) use heads of this share of the staff space. Unit:
/// percent of the staff space.
const SMALL_HEAD: SpacePercent = SpacePercent(79);
/// The narrow template's head size. Unit: percent of the staff space.
const NARROW_HEAD: SpacePercent = SpacePercent(90);
/// Staff spaces below this are low resolution: a match needs fewer outline points, and the
/// wing allows one more ink point. Unit: pixels.
const LOW_RESOLUTION_SPACE: i32 = 15;
/// The share of the outline points a match needs (at low resolution).
const PASS_LOW_RESOLUTION: Percent = Percent::constant(68);
/// The share of the outline points a match needs.
const PASS: Percent = Percent::constant(70);
/// The share of the hollow points a match needs (as paper).
const CLEAR_PASS: Percent = Percent::constant(37);
/// The slopes are stepped in sixteenths of a pixel: a shift.
const SLOPE_SHIFT: u32 = 4;
/// The rounding added before a size is quartered.
const QUARTER_ROUNDING: i32 = 1;
/// … and before it is cut into eighths.
const EIGHTH_ROUNDING: i32 = 3;
/// A size of at least this many pixels has a quarter of its own (smaller ones use 1).
const MIN_QUARTERED: i32 = 4;
/// Sizes below this have a one-pixel rim. Unit: pixels.
const MIN_RIMMED: i32 = 5;
/// A quarter as a shift.
const QUARTER_SHIFT: u32 = 2;
/// An eighth as a shift.
const EIGHTH_SHIFT: u32 = 3;

/// A point of a template: `x` columns right of and `up` rows above the origin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TemplatePoint {
    /// Columns to the right.
    pub x: i32,
    /// Rows upwards.
    pub up: i32,
}

/// The points of one half of an open head at one size.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HeadMask {
    /// The outline: points that must be ink.
    pub set: Vec<TemplatePoint>,
    /// The hollow: points that must be paper.
    pub clear: Vec<TemplatePoint>,
    /// The right half's wing: a box beyond the head that must be paper.
    pub wing: Option<Wing>,
}

/// The paper box right of an open head.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Wing {
    /// Its points.
    pub points: Vec<TemplatePoint>,
    /// The ink points in it that a match tolerates.
    pub threshold: i32,
}

/// Which half of a head a template covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadSide {
    /// The left half.
    Left,
    /// The right half, with the wing.
    Right,
}

/// One half of an open head at both sizes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HalfHeadTemplate {
    /// The half.
    pub side: HeadSide,
    /// The staff space the template is drawn for.
    pub space: i32,
    /// The template for a head of nine tenths of the space.
    pub narrow: HeadMask,
    /// The template for a head of the full space.
    pub full: HeadMask,
}

/// The open-head templates of a staff.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenHeadTemplates {
    /// The left half for the staff's heads.
    pub left: HalfHeadTemplate,
    /// The right half for the staff's heads.
    pub right: HalfHeadTemplate,
    /// The left half for small heads.
    pub small_left: HalfHeadTemplate,
    /// The right half for small heads.
    pub small_right: HalfHeadTemplate,
    /// The share of the outline points a match needs.
    pub pass: Percent,
    /// The share of the hollow points a match needs.
    pub clear_pass: Percent,
}

impl OpenHeadTemplates {
    /// The templates of a staff with staff space `space` (pixels).
    pub fn new(space: i32) -> Self {
        let small = StaffSpace(Px(space)).scale(SMALL_HEAD).get();
        OpenHeadTemplates {
            left: HalfHeadTemplate::new(HeadSide::Left, space),
            right: HalfHeadTemplate::new(HeadSide::Right, space),
            small_left: HalfHeadTemplate::new(HeadSide::Left, small),
            small_right: HalfHeadTemplate::new(HeadSide::Right, small),
            pass: if space < LOW_RESOLUTION_SPACE {
                PASS_LOW_RESOLUTION
            } else {
                PASS
            },
            clear_pass: CLEAR_PASS,
        }
    }
}

impl HalfHeadTemplate {
    /// The `side` half for staff space `space`.
    pub fn new(side: HeadSide, space: i32) -> Self {
        let mask = |size: i32| match side {
            HeadSide::Left => left_half(size),
            HeadSide::Right => right_half(size, space),
        };
        HalfHeadTemplate {
            side,
            space,
            narrow: mask(StaffSpace(Px(space)).scale(NARROW_HEAD).get()),
            full: mask(space),
        }
    }

    /// The bounds of all points of both sizes, inclusive: `(left, bottom, right, top)`;
    /// `None` without points.
    pub fn extent(&self) -> Option<(i32, i32, i32, i32)> {
        let wing = |m: &HeadMask| {
            m.wing
                .iter()
                .flat_map(|w| w.points.clone())
                .collect::<Vec<_>>()
        };
        let points: Vec<TemplatePoint> = [&self.narrow, &self.full]
            .into_iter()
            .flat_map(|m| m.set.iter().chain(&m.clear).copied().chain(wing(m)))
            .collect();
        let first = points.first()?;
        Some(
            points
                .iter()
                .fold((first.x, first.up, first.x, first.up), |(l, b, r, t), p| {
                    (l.min(p.x), b.min(p.up), r.max(p.x), t.max(p.up))
                }),
        )
    }
}

/// The size `size` cut into a quarter (1 for tiny sizes) and the matching half.
fn half_and_quarter(size: i32) -> (i32, i32) {
    let half = size >> 1;
    if half == 0 {
        return (1, 1);
    }
    let quarter = (size + QUARTER_ROUNDING) >> QUARTER_SHIFT;
    (
        half,
        if size + QUARTER_ROUNDING < MIN_QUARTERED {
            1
        } else {
            quarter
        },
    )
}

/// The left half of a head `size` pixels high: an outer stroke rising leftwards from the
/// bottom and an inner one falling from the top, with the hollow between them.
fn left_half(size: i32) -> HeadMask {
    let (half, quarter) = half_and_quarter(size);
    let span = if size >> 1 == 0 {
        1
    } else {
        half + ((size + QUARTER_ROUNDING) >> QUARTER_SHIFT)
    };
    let mut mask = HeadMask::default();
    let mut outer = (quarter + size) << SLOPE_SHIFT;
    let outer_step = outer / span;
    let mut inner = half << SLOPE_SHIFT;
    let inner_step = inner / span;
    let far_left = -(half + size);
    let mut row = 0;
    while row < quarter + half {
        let top_row = size - row;
        let inner_x = inner >> SLOPE_SHIFT;
        let mut x = -inner_x;
        for i in inner_x..(outer >> SLOPE_SHIFT) {
            mask.set.push(TemplatePoint { x, up: row });
            mask.set.push(TemplatePoint {
                x: far_left + i,
                up: top_row,
            });
            x -= 1;
        }
        let hollow_end = row + half;
        row += 1;
        mask.clear
            .extend((row..hollow_end).map(|up| TemplatePoint { x, up }));
        mask.clear
            .extend((far_left..inner_x - (half + size)).map(|x| TemplatePoint { x, up: top_row }));
        inner -= inner_step;
        outer -= outer_step;
    }
    mask
}

/// The right half of a head `size` pixels high on a staff of space `space`: two strokes
/// closing the head on the right, a rim of an eighth of the size thick, and the wing.
fn right_half(size: i32, space: i32) -> HeadMask {
    let half = size >> 1;
    let eighth = ((size + EIGHTH_ROUNDING) as u32 >> EIGHTH_SHIFT) as i32;
    let (rows_half, span, quarter, reach, rim) = if half == 0 {
        (1, 1, 1, size + QUARTER_ROUNDING, 1)
    } else {
        let quarter = (size + QUARTER_ROUNDING) >> QUARTER_SHIFT;
        let span = quarter + half;
        if size + QUARTER_ROUNDING < MIN_QUARTERED {
            (half, span, 1, size + QUARTER_ROUNDING, 1)
        } else {
            (
                half,
                span,
                quarter,
                quarter + size,
                if size < MIN_RIMMED { 1 } else { eighth.max(1) },
            )
        }
    };
    let mut mask = HeadMask::default();
    let mut width = reach << SLOPE_SHIFT;
    let width_step = width / span;
    let rim_width = (rim + rows_half) << SLOPE_SHIFT;
    let rim_step = rim_width / span;
    let rows = quarter + rows_half;
    let lift = rim / 2;
    let (mut start, mut end) = (rim_width, width);
    let mut row = 0;
    while row < rows {
        let mut x = (rim + reach) - (start >> SLOPE_SHIFT);
        for _ in (start >> SLOPE_SHIFT)..=(end >> SLOPE_SHIFT) {
            mask.set.push(TemplatePoint { x, up: row + lift });
            x -= 1;
        }
        let hollow_end = row + rows_half;
        row += 1;
        mask.clear
            .extend((row..hollow_end).map(|up| TemplatePoint { x, up: lift + up }));
        start -= rim_step;
        end -= width_step;
    }
    let lower_step = rim_width / rows_half;
    let mut lower = rim_width;
    for k in 0..rows {
        let first = lower >> SLOPE_SHIFT;
        let left = (reach + rim * 2) - rows_half - size + first;
        for (j, x) in (first..(width >> SLOPE_SHIFT)).zip(left..) {
            if rows_half + size - rim <= rim + reach + j {
                mask.set.push(TemplatePoint {
                    x,
                    up: (rim + size + lift) - k,
                });
            }
        }
        lower -= lower_step;
        width -= width_step;
    }
    mask.wing = Some(wing(size, space, half));
    mask
}

/// The wing of a right half of a head `size` pixels high (half `half`) on a staff of space
/// `space`: an eighth of the size high, half the size wide, a quarter beyond the head.
fn wing(size: i32, space: i32, half: i32) -> Wing {
    let height = size >> EIGHTH_SHIFT;
    let left = size + (size >> QUARTER_SHIFT);
    let bottom = (size >> QUARTER_SHIFT) + height;
    let tolerated = ((half + 1) * height) / 2;
    let points = (left..=left + half)
        .flat_map(|x| (bottom..bottom + height).map(move |up| TemplatePoint { x, up }))
        .collect();
    Wing {
        points,
        threshold: if space < LOW_RESOLUTION_SPACE {
            tolerated + 1
        } else {
            tolerated
        },
    }
}

#[cfg(test)]
#[path = "tests/templates.rs"]
mod tests;
