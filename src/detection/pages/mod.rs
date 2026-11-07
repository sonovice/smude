//! Page regions, spread layout and optional learned contour/ratio inference.
#[cfg(feature = "onnx-pages")]
mod filter;
#[cfg(feature = "onnx-pages")]
mod model;
mod regions;

#[cfg(feature = "onnx-pages")]
pub use model::{PageDetectionReport, PageDetector, PageLayoutMode, PageSpreadLayout};
pub use regions::{PageRegion, PageRegions};
