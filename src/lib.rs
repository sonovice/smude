//! Native GCS and staff-driven rectification with flexible local corrections.
//! Core fitting uses observations; optional ONNX inference supplies page contours.
//! Fit once, transform observations, sample the original image once.

#[cfg(feature = "application")]
pub mod application;

use serde::{Deserialize, Serialize};

mod gcs;
mod local_flatten;
mod observations;
mod perspective;
mod staff_conditioning;
mod staff_extensions;
mod system_boundaries;
mod vertical_correction;
mod warp;

#[cfg(feature = "render")]
mod detect;
#[cfg(feature = "render")]
mod output;
#[cfg(feature = "render")]
mod page_analysis;
#[cfg(feature = "render")]
mod pages;
#[cfg(feature = "render")]
mod pipeline;
#[cfg(feature = "render")]
mod render;
#[cfg(feature = "render")]
mod text_lines;
#[cfg(feature = "render")]
mod text_warp;

#[cfg(feature = "onnx-pages")]
mod page_model;

pub use gcs::{GcsMap, GcsParameters, GcsReport};
pub use observations::{Barline, Observations, Staff, StaffEdge, StaffEdgeSide};
pub use perspective::PerspectiveReport;
pub use staff_conditioning::{
    ConditioningReport, StaffConditioningReport, condition as condition_staff_observations,
};
pub use staff_extensions::{StaffEndpointRejection, StaffExtensionReport};
pub use system_boundaries::BoundaryColumn;
pub use vertical_correction::VerticalCorrection;
pub use warp::{ExtrapolationReport, FitProfile, Geometry, Report, SystemReport, Validation};

#[cfg(feature = "render")]
pub use detect::{
    crop_page, detect, detect_cropped_page, detect_page, detect_staff_edges,
    detect_staff_edges_with_region_mask, detect_with_region_mask,
};
#[cfg(feature = "render")]
pub use output::{
    antialias_print, apply_binary_cleanup, apply_binary_cleanup_with_contour, binarize_sauvola,
    content_crop, normalize_paper, page_crop, whiten_gray_border, whiten_page_border,
};
#[cfg(feature = "render")]
pub use page_analysis::PageAnalysisMap;
#[cfg(feature = "render")]
pub use pages::{PageRegion, PageRegions};
#[cfg(feature = "render")]
pub use pipeline::GcsPipelineReport;
#[cfg(feature = "render")]
pub use text_lines::{TextLine, detect_text_lines};
#[cfg(feature = "render")]
pub use text_warp::TextWarpReport;

#[cfg(feature = "onnx-pages")]
pub use page_model::{PageDetectionReport, PageDetector, PageLayoutMode, PageSpreadLayout};

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

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(&'static str),
    #[cfg(feature = "onnx-pages")]
    #[error("ONNX page detection: {0}")]
    PageModel(String),
}
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n.is_multiple_of(2) {
        (values[n / 2 - 1] + values[n / 2]) / 2.
    } else {
        values[n / 2]
    }
}

pub(crate) fn slope(points: &[Point]) -> f64 {
    let n = points.len() as f64;
    let x = points.iter().map(|p| p.x).sum::<f64>() / n;
    let y = points.iter().map(|p| p.y).sum::<f64>() / n;
    let den = points.iter().map(|p| (p.y - y).powi(2)).sum::<f64>();
    if den < 1e-12 {
        return 0.;
    }
    points.iter().map(|p| (p.x - x) * (p.y - y)).sum::<f64>() / den
}
