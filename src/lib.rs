//! Sheet-music page detection, reversible geometry and image reconstruction.
//!
//! - `detection`: page, music and text observations (`render` feature).
//! - [`geometry`]: fit and compose pixel-coordinate transformations.
//! - `rendering`: CPU/GPU rasterization and output preparation (`render`).
//! - `application`: shared file-processing workflow (`application`).
//!
//! Root re-exports preserve the existing CLI, desktop and mobile interface.
#[cfg(feature = "application")]
pub mod application;
#[cfg(feature = "render")]
pub mod detection;
mod error;
pub mod geometry;
mod math;
mod parallel;
#[cfg(feature = "render")]
pub mod rendering;

#[cfg(feature = "onnx-pages")]
pub use detection::{PageDetectionReport, PageDetector, PageLayoutMode, PageSpreadLayout};
#[cfg(feature = "render")]
pub use detection::{
    PageRegion, PageRegions, TextLine, crop_page, detect, detect_cropped_page, detect_page,
    detect_staff_edges, detect_staff_edges_with_region_mask, detect_text_lines,
    detect_with_region_mask,
};
pub use error::{Error, Result};
pub use geometry::{
    Barline, BoundaryColumn, ConditioningReport, Config, ExtrapolationReport, FitProfile, GcsMap,
    GcsParameters, GcsReport, Geometry, Observations, PerspectiveReport, Point, Report, Staff,
    StaffConditioningReport, StaffEdge, StaffEdgeSide, StaffEndpointRejection,
    StaffExtensionReport, SystemReport, Validation, VerticalCorrection,
    condition_staff_observations,
};
#[cfg(feature = "render")]
pub use geometry::{GcsPipelineReport, PageAnalysisMap, TextWarpReport};
#[cfg(feature = "render")]
pub use rendering::output::{
    antialias_print, apply_binary_cleanup, apply_binary_cleanup_with_contour, binarize_sauvola,
    content_crop, normalize_paper, page_crop, whiten_gray_border, whiten_page_border,
};
#[cfg(feature = "render")]
pub use rendering::{RenderBackend, RenderProfile};
