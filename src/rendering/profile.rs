//! Renderer selection and per-stage diagnostics.
use image::RgbImage;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderBackend {
    #[default]
    Auto,
    Cpu,
    Gpu,
}

/// Wall-clock preparation/readback and optional hardware timestamp timings.
/// The sampled map error is a probe measurement, not an analytic error bound.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct RenderProfile {
    pub backend: String,
    pub adapter: Option<String>,
    pub fallback_reason: Option<String>,
    pub initialization_seconds: f64,
    pub map_seconds: f64,
    pub upload_seconds: f64,
    pub gpu_wait_readback_seconds: f64,
    pub unpack_seconds: f64,
    pub gpu_render_seconds: Option<f64>,
    pub gpu_horizontal_seconds: Option<f64>,
    pub gpu_paper_seconds: Option<f64>,
    pub gpu_border_distance_calls: usize,
    pub gpu_border_distance_seconds: f64,
    pub gpu_border_fallback_reason: Option<String>,
    pub map_evaluations: usize,
    pub map_nodes: usize,
    pub exact_map_pixels: usize,
    pub max_sampled_map_error_pixels: f64,
}

pub(crate) struct PreparedOutput {
    pub color: Option<RgbImage>,
    pub tones: Option<(image::GrayImage, image::GrayImage, image::GrayImage)>,
    pub profile: RenderProfile,
}
