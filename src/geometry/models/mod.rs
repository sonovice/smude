//! Broad page models: projective frame, perspective and GCS surface.
mod gcs;
#[cfg(feature = "render")]
mod page_frame;
pub(crate) mod perspective;
pub use gcs::{GcsMap, GcsParameters, GcsReport};
#[cfg(feature = "render")]
pub use page_frame::PageAnalysisMap;
pub use perspective::PerspectiveReport;
