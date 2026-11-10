//! The lengths a staff's notes are measured with.
//!
//! Every note measurement scales with the staff space of the note's staff: the fractions of
//! [`SpaceMetrics`], the smaller fractions of grace and cue notes ([`GraceMetrics`]), the
//! staff's size limits ([`ParseMetrics`]), and the resolution bands some thresholds depend on
//! ([`NoteMetrics::resolution`]).

use super::parse_metrics::ParseMetrics;
use super::space::{SpaceMetrics, percent_of};
use super::types::{Px, SpacePercent, StaffSpace};

/// The size of a grace (or cue) note relative to a full-size note.
const GRACE_SCALE: SpacePercent = SpacePercent(60);
/// Staff spaces below this have no grace eighth (it is 0). Unit: pixels.
const GRACE_EIGHTH_MIN_SPACE: i32 = 5;
/// Rounding added before the eighth of the space is taken. Unit: pixels.
const EIGHTH_ROUNDING: i32 = 3;
/// The eighth of a staff space as a shift.
const EIGHTH_SHIFT: u32 = 3;
/// The quarter of a staff space as a shift (after adding one).
const QUARTER_SHIFT: u32 = 2;

/// Staff spaces below this are the lowest resolution band. Unit: pixels.
const LOWEST_SPACE: i32 = 9;
/// Staff spaces below this are the lower resolution band. Unit: pixels.
const LOWER_SPACE: i32 = 12;
/// Staff spaces below this are the lower band or just above it. Unit: pixels.
const NEAR_LOWER_SPACE: i32 = 13;
/// Staff spaces below this are low resolution. Unit: pixels.
const LOW_SPACE: i32 = 15;
/// Staff spaces below this are low resolution or just above it. Unit: pixels.
const NEAR_LOW_SPACE: i32 = 18;
/// Staff spaces above this are high resolution. Unit: pixels.
const HIGH_SPACE: i32 = 19;

/// The fractions of a grace note's staff space (60 % of the full ones).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraceMetrics {
    /// The grace space.
    pub space: i32,
    /// The grace half space.
    pub half: i32,
    /// The grace quarter space.
    pub quarter: i32,
    /// The grace eighth space (0 on very small staves).
    pub eighth: i32,
}

impl GraceMetrics {
    /// The grace fractions of staff space `space`.
    pub fn new(space: i32) -> Self {
        let eighth = if space < GRACE_EIGHTH_MIN_SPACE {
            0
        } else {
            percent_of((space + EIGHTH_ROUNDING) >> EIGHTH_SHIFT, GRACE_SCALE)
        };
        GraceMetrics {
            space: percent_of(space, GRACE_SCALE),
            half: percent_of(space >> 1, GRACE_SCALE),
            quarter: percent_of((space + 1) >> QUARTER_SHIFT, GRACE_SCALE),
            eighth,
        }
    }
}

/// How fine a staff is printed, by its staff space; some thresholds change between bands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Resolution {
    space: i32,
}

impl Resolution {
    /// Spaces below 9 px.
    pub fn lowest(self) -> bool {
        self.space < LOWEST_SPACE
    }

    /// Spaces below 12 px.
    pub fn lower(self) -> bool {
        self.space < LOWER_SPACE
    }

    /// Spaces below 13 px.
    pub fn near_lower(self) -> bool {
        self.space < NEAR_LOWER_SPACE
    }

    /// Spaces below 15 px: low resolution.
    pub fn low(self) -> bool {
        self.space < LOW_SPACE
    }

    /// Spaces below 18 px.
    pub fn near_low(self) -> bool {
        self.space < NEAR_LOW_SPACE
    }

    /// Spaces above 19 px: high resolution.
    pub fn high(self) -> bool {
        self.space > HIGH_SPACE
    }
}

/// The lengths the notes of one staff are measured with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoteMetrics {
    /// The fractions of the staff space.
    pub space: SpaceMetrics,
    /// The fractions of a grace note's space.
    pub grace: GraceMetrics,
    /// The resolution band.
    pub resolution: Resolution,
    /// The staff's size limits.
    pub parse: ParseMetrics,
}

impl NoteMetrics {
    /// The note metrics of staff space `space`.
    pub fn new(space: i32) -> Self {
        let metrics = SpaceMetrics::new(space);
        NoteMetrics {
            space: metrics,
            grace: GraceMetrics::new(space),
            resolution: Resolution { space },
            parse: ParseMetrics::of(StaffSpace(Px(space))),
        }
    }

    /// `percent` of the staff space, truncated towards zero.
    pub fn percent(&self, percent: SpacePercent) -> i32 {
        percent_of(self.space.space, percent)
    }
}
