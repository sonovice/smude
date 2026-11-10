//! Image observations: classical music/text detection and optional ONNX pages.
pub(crate) mod music;
pub mod pages;
pub mod quality;
mod resample;
pub mod stems;
mod text;

pub use music::{
    crop_page, detect, detect_cropped_page, detect_page, detect_staff_edges,
    detect_staff_edges_with_region_mask, detect_with_region_mask,
};
#[cfg(feature = "onnx-pages")]
pub use pages::{PageDetectionReport, PageDetector, PageLayoutMode, PageSpreadLayout};
pub use pages::{PageRegion, PageRegions};
pub use text::{TextLine, detect_text_lines};
