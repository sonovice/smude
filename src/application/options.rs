//! Shared processing configuration; no GUI or command-line dependencies.
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum PageLayout {
    #[default]
    Auto,
    Single,
    Double,
}
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum SpreadLayout {
    #[default]
    SideBySide,
    Stacked,
}
#[derive(Clone, Copy, Debug, Default)]
pub enum CropMode {
    #[default]
    Page,
    Content,
}

pub struct Options {
    /// Optional live detection observer; events use original-image coordinates.
    pub progress: Option<super::ProgressObserver>,
    /// Existing Smude barlines.json in original-image pixels.
    pub geometry: Option<PathBuf>,
    /// Optional oracle/manual TL,TR,BR,BL corners (four Point objects as JSON).
    pub geometry_corners: Option<PathBuf>,
    /// Dense source-paper contour for supplied geometry and final pixel masking.
    pub geometry_contour: Option<PathBuf>,
    /// Physical page height/width for supplied geometry (page mode uses its model).
    pub page_aspect: Option<f64>,
    /// Explicitly enable page detection (already the default; compatibility flag).
    pub pages: bool,
    /// Process an already cropped image without page detection.
    pub no_pages: bool,
    /// page-boundary model page layout; automatic classification is the default.
    pub page_layout: PageLayout,
    /// Arrangement of a double-page capture; independent of image aspect ratio.
    pub spread_layout: SpreadLayout,
    /// Override the bundled page-layout and contour/aspect ONNX weights.
    pub page_models: Option<PathBuf>,
    /// Explicit PageRegions JSON in original-image pixel coordinates.
    pub page_regions: Option<PathBuf>,
    pub image: PathBuf,
    pub out: PathBuf,
    /// Enable projective correction (the default; retained for explicit commands).
    pub perspective: bool,
    /// Disable the default projective stage and only flatten staff curves.
    pub no_perspective: bool,
    /// Disable native GCS and retain the projective/local geometry comparison.
    pub no_gcs: bool,
    /// Disable the default bounded, evidence-gated x-only system correction.
    pub no_vertical_correction: bool,
    /// Prefer a guarded projective page frame from detected near-straight sides.
    pub page_perspective: bool,
    /// Use only the staff-derived projective component, without page framing.
    pub no_page_perspective: bool,
    /// Ablation: detect in the original crop instead of the page-rectified analysis image.
    pub no_analysis_perspective: bool,
    /// Disable local text curvature correction (music geometry is always protected).
    pub no_text_dewarp: bool,
    /// Disable independently measured staff-start/end geometry observations.
    pub no_staff_edges: bool,
    /// Opt into one shared final shear per system (not valid for all deformations).
    pub system_shear: bool,
    /// Explicit same-size groups: '0,1,2,3;4,5'. Ossias stay separate.
    pub same_size: Option<String>,
    pub size_prior: f64,
    pub no_render: bool,
    /// Export native classical detections without fitting or rendering.
    pub detect_only: bool,
    /// Optional analysis long-edge cap. Default policy remains max width 1800.
    pub analysis_long_edge: Option<u32>,
    /// Legacy analysis width override, for reproducing earlier runs.
    pub analysis_width: Option<u32>,
    /// Inward binary page-border margin as a fraction of the output short edge.
    /// Zero disables cleanup. Applies only when a page contour is available.
    pub binary_page_margin: f64,
    /// Retain the full reconstructed canvas instead of final page/content framing.
    pub no_final_crop: bool,
    /// Frame the detected paper (default), or explicitly crop tightly to ink.
    pub crop_mode: CropMode,
    /// Padding around final page/content bounds, as a fraction of the short edge.
    pub crop_padding: f64,
    /// Optional input-pixel probes to transform for independent external QA.
    pub probe_points: Option<PathBuf>,
}

impl Options {
    pub fn uses_page_mode(&self) -> bool {
        self.page_regions.is_some()
            || self.pages
            || (cfg!(feature = "onnx-pages") && !self.no_pages && self.geometry.is_none())
    }

    pub fn analysis_width_for(&self, image: &image::RgbImage) -> u32 {
        self.analysis_width.unwrap_or_else(|| {
            if self.analysis_long_edge.is_none() {
                return 1800;
            }
            let longest = image.width().max(image.height());
            (u64::from(image.width())
                * u64::from(
                    self.analysis_long_edge
                        .map(|limit| limit.min(longest))
                        .unwrap_or(longest),
                )
                / u64::from(longest)) as u32
        })
    }
}

impl Default for Options {
    fn default() -> Self {
        Self {
            progress: None,
            geometry: None,
            geometry_corners: None,
            geometry_contour: None,
            page_aspect: None,
            pages: false,
            no_pages: false,
            page_layout: PageLayout::default(),
            spread_layout: SpreadLayout::default(),
            page_models: None,
            page_regions: None,
            image: PathBuf::default(),
            out: PathBuf::default(),
            perspective: false,
            no_perspective: false,
            no_gcs: false,
            no_vertical_correction: false,
            page_perspective: false,
            no_page_perspective: false,
            no_analysis_perspective: false,
            no_text_dewarp: false,
            no_staff_edges: false,
            system_shear: false,
            same_size: None,
            size_prior: 1.,
            no_render: false,
            detect_only: false,
            analysis_long_edge: None,
            analysis_width: None,
            binary_page_margin: 0.004,
            no_final_crop: false,
            crop_mode: CropMode::default(),
            crop_padding: 0.01,
            probe_points: None,
        }
    }
}
