//! The lengths derived from a staff space that the staff and bar steps use.

use super::types::{Px, SpacePercent, StaffSpace};

/// Staff spaces below this use a fixed small fraction for [`SpaceMetrics::eighth`]. Unit: pixels.
const SMALL_SPACE: i32 = 5;
/// Staff spaces below this get the padded [`SpaceMetrics::bold`] and [`SpaceMetrics::wide`]. Unit: pixels.
const LOW_RESOLUTION_SPACE: i32 = 20;
/// Padding of [`SpaceMetrics::bold`] at low resolution. Unit: pixels.
const BOLD_PADDING: i32 = 5;
/// Padding of [`SpaceMetrics::wide`] at low resolution. Unit: pixels.
const WIDE_PADDING: i32 = 10;
/// The smallest stroke width a staff's symbols are measured with. Unit: pixels.
const MIN_STROKE: i32 = 2;
/// Rounding added before the eighth of the space is taken. Unit: pixels.
const EIGHTH_ROUNDING: i32 = 3;
/// The eighth of a staff space as a shift.
const EIGHTH_SHIFT: u32 = 3;
/// The quarter of a staff space as a shift.
const QUARTER_SHIFT: u32 = 2;

/// A staff space and the fractions of it that the layout steps measure with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpaceMetrics {
    /// The staff space d0.
    pub space: i32,
    /// Half the space, rounded down.
    pub half: i32,
    /// A quarter of the space, rounded to nearest.
    pub quarter: i32,
    /// An eighth of the space, rounded up; 1 for very small spaces.
    pub eighth: i32,
    /// Half plus a quarter of the space.
    pub three_quarters: i32,
    /// The space, padded at low resolution: the size of bold strokes and dots.
    pub bold: i32,
    /// The space, padded more at low resolution.
    pub wide: i32,
    /// The thinnest stroke: the eighth, but at least two pixels.
    pub stroke: i32,
}

impl SpaceMetrics {
    /// The metrics of a staff space of `space` pixels.
    pub fn new(space: i32) -> Self {
        let half = space >> 1;
        let quarter = (space + 1) >> QUARTER_SHIFT;
        let eighth = if space < SMALL_SPACE {
            1
        } else {
            ((space as u32 + EIGHTH_ROUNDING as u32) >> EIGHTH_SHIFT) as i32
        };
        let three_quarters = quarter + half;
        let low = space < LOW_RESOLUTION_SPACE;
        SpaceMetrics {
            space,
            half,
            quarter,
            eighth,
            three_quarters,
            bold: if low {
                three_quarters + BOLD_PADDING
            } else {
                space
            },
            wide: if low { half + WIDE_PADDING } else { space },
            stroke: eighth.max(MIN_STROKE),
        }
    }

    /// `percent` of the staff space, truncated (`d0 * p / 100`).
    pub fn percent(self, percent: SpacePercent) -> i32 {
        StaffSpace(Px(self.space)).scale(percent).get()
    }
}

/// `percent` of `value`, truncated towards zero (`v * p / 100`).
pub fn percent_of(value: i32, percent: SpacePercent) -> i32 {
    StaffSpace(Px(value)).scale(percent).get()
}
