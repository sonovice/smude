//! Only the parse limit consumed by the retained stem-edge measurements.
use super::types::{SpacePercent, StaffSpace};
const MIN_CLEAR_STEM_RUN: SpacePercent = SpacePercent(150);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParseMetrics {
    pub min_clear_stem_run: i32,
}
impl ParseMetrics {
    pub fn of(space: StaffSpace) -> Self {
        Self {
            min_clear_stem_run: space.scale(MIN_CLEAR_STEM_RUN).get(),
        }
    }
}
