//! Reversible source-pixel geometry and its observation model.
//!
//! Broad page models live in models/; evidence-driven corrections in
//! corrections/. fit/ composes these transformations without retaining images.
#[cfg(feature = "render")]
pub(crate) mod composition;
pub(crate) mod corrections;
pub(crate) mod fit;
pub(crate) mod models;
pub(crate) mod observations;
mod report;
mod types;

#[cfg(feature = "render")]
pub use composition::GcsPipelineReport;
#[cfg(feature = "render")]
pub use corrections::TextWarpReport;
pub use corrections::{
    BoundaryColumn, ConditioningReport, StaffConditioningReport, StaffEndpointRejection,
    StaffExtensionReport, VerticalCorrection, condition_staff_observations,
};
pub use fit::Geometry;
#[cfg(feature = "render")]
pub use models::PageAnalysisMap;
pub use models::{GcsMap, GcsParameters, GcsReport, PerspectiveReport};
pub use observations::{Barline, Observations, Staff, StaffEdge, StaffEdgeSide};
pub use report::{ExtrapolationReport, FitProfile, Report, SystemReport, Validation};
pub use types::{Config, Point};
