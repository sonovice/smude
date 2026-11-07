//! Shared pixel-coordinate primitives and geometry fit configuration.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Config {
    /// Fit a broad projective component using staff spacing and bar directions.
    /// Local residual bends remain independent; no 3-D surface is assumed.
    pub perspective: bool,
    /// Optional regularization for systems with a common residual direction.
    /// Disabled by default: different local directions need not share a shear.
    pub system_shear: bool,
    /// Explicit zero-based equal-size groups. Undeclared staves stay independent.
    pub staff_size_groups: Vec<Vec<usize>>,
    /// Soft relative weight of declared equal-size priors.
    pub size_prior: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            perspective: true,
            system_shear: false,
            staff_size_groups: vec![],
            size_prior: 1.,
        }
    }
}
