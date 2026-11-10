//! Residual corrections and regularization from musical/text observations.
mod direction_mesh;
pub(crate) mod local_flatten;
pub(crate) use direction_mesh::MeshRow;
pub use direction_mesh::{DirectionMesh, DirectionMeshReport};
mod staff_conditioning;
pub(crate) mod staff_extensions;
pub(crate) mod surface_metric;
pub(crate) mod system_boundaries;
#[cfg(feature = "render")]
pub(crate) mod text_warp;
mod vertical_correction;

pub use staff_conditioning::{
    ConditioningReport, StaffConditioningReport, condition as condition_staff_observations,
};
pub use staff_extensions::{StaffEndpointRejection, StaffExtensionReport};
pub use system_boundaries::BoundaryColumn;
#[cfg(feature = "render")]
pub use text_warp::TextWarpReport;
pub use vertical_correction::VerticalCorrection;
