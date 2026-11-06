use serde::{Deserialize, Serialize};

use crate::{Error, Point, Result};

/// Immutable detector measurements in input-raster pixels. In page mode the
/// input is an integer page crop; its original-image offset is separate metadata.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Observations {
    pub width: u32,
    pub height: u32,
    pub staves: Vec<Staff>,
    pub barlines: Vec<Barline>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Staff {
    pub x: Vec<f64>,
    pub y: [Vec<f64>; 5],
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Barline {
    pub staff: usize,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    #[serde(default)]
    pub aligned_staffs: Vec<usize>,
}

/// Geometric staff boundary evidence, independent of printed bar strokes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum StaffEdgeSide {
    Start,
    End,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StaffEdge {
    pub staff: usize,
    pub side: StaffEdgeSide,
    /// Independently measured line endpoints, in original raster coordinates.
    pub points: Vec<Point>,
    /// Original indices 0..4 corresponding one-to-one with `points`.
    pub line_indices: Vec<usize>,
    /// Uncalibrated support strength from ridge continuity, blank margin and
    /// number of independently measured rules; not a model probability.
    pub confidence: f64,
    pub aligned_staffs: Vec<usize>,
}

impl Staff {
    pub(crate) fn sample(&self, fraction: f64) -> [Point; 5] {
        let x = self.x[0] + fraction * (self.x[self.x.len() - 1] - self.x[0]);
        std::array::from_fn(|k| Point {
            x,
            y: interpolate(&self.x, &self.y[k], x),
        })
    }
}

pub(crate) fn interpolate(x: &[f64], y: &[f64], at: f64) -> f64 {
    if at <= x[0] {
        return y[0];
    }
    if at >= x[x.len() - 1] {
        return y[y.len() - 1];
    }
    let hi = x.partition_point(|v| *v < at);
    let t = (at - x[hi - 1]) / (x[hi] - x[hi - 1]);
    y[hi - 1] * (1. - t) + y[hi] * t
}

impl Observations {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.width < 2 || self.height < 2 || self.staves.is_empty() {
            return Err(Error::Invalid(
                "need image dimensions and at least one staff",
            ));
        }
        for s in &self.staves {
            if s.x.len() < 2
                || s.x
                    .iter()
                    .any(|x| !x.is_finite() || *x < 0. || *x > f64::from(self.width - 1))
                || s.x.windows(2).any(|w| w[0] >= w[1])
                || s.y.iter().any(|y| {
                    y.len() != s.x.len()
                        || y.iter()
                            .any(|v| !v.is_finite() || *v < 0. || *v > f64::from(self.height - 1))
                })
                || (0..s.x.len()).any(|j| (0..4).any(|k| s.y[k][j] >= s.y[k + 1][j]))
            {
                return Err(Error::Invalid(
                    "staff curves must be ordered and inside the image",
                ));
            }
        }
        for b in &self.barlines {
            if b.staff >= self.staves.len()
                || b.aligned_staffs.iter().any(|s| *s >= self.staves.len())
                || b.aligned_staffs.contains(&b.staff)
                || b.aligned_staffs
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    != b.aligned_staffs.len()
                || b.x.len() < 2
                || b.x.len() != b.y.len()
                || b.x.iter().chain(&b.y).any(|v| !v.is_finite())
                || b.x
                    .iter()
                    .any(|x| *x < 0. || *x > f64::from(self.width - 1))
                || b.y
                    .iter()
                    .any(|y| *y < 0. || *y > f64::from(self.height - 1))
                || b.y.windows(2).any(|w| w[0] >= w[1])
            {
                return Err(Error::Invalid(
                    "barline paths must have ordered y and valid staff indices",
                ));
            }
        }
        Ok(())
    }
}
