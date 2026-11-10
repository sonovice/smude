//! Serializable fit diagnostics and validation; no rendering or detector state.
use crate::geometry::{Config, PerspectiveReport, StaffEdge};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct SystemReport {
    pub staves: Vec<usize>,
    pub region: [f64; 2],
    pub center: f64,
    pub shear: f64,
    pub candidate_shear: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Validation {
    pub sampled_min_jacobian: f64,
    pub roundtrip_max_pixels: f64,
    pub final_stage_y_displacement: f64,
    pub max_staff_line_span_pixels: f64,
    pub bar_median_angle_degrees: f64,
    pub bar_p90_angle_degrees: f64,
    pub bar_max_bend_pixels: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    #[cfg(feature = "render")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_model: Option<crate::geometry::PageAnalysisMap>,
    #[cfg(feature = "render")]
    pub direction_mesh: Option<crate::geometry::MeshPipelineReport>,
    pub schema: &'static str,
    pub input_width: u32,
    pub input_height: u32,
    pub perspective: PerspectiveReport,
    pub config: Config,
    pub measured_staff_spacings: Vec<f64>,
    pub target_staff_spacings: Vec<f64>,
    pub systems: Vec<SystemReport>,
    pub bounds: [f64; 4],
    pub validation: Validation,
    pub validation_note: &'static str,
    pub extrapolation: ExtrapolationReport,
    pub staff_edges: Vec<StaffEdge>,
    pub staff_extensions: crate::geometry::StaffExtensionReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vertical_correction: Option<crate::geometry::VerticalCorrection>,
    #[cfg(feature = "render")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gcs: Option<crate::geometry::GcsPipelineReport>,
    #[cfg(feature = "render")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_warp: Option<crate::geometry::TextWarpReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExtrapolationReport {
    pub model: &'static str,
    pub support_limit_staff_spaces: f64,
    pub max_boundary_jump_pixels: f64,
    pub adjusted_anchor_samples: usize,
    pub max_adjustment_pixels: f64,
    pub changed_x_bounds: Option<[f64; 2]>,
    pub min_source_gap_pixels: f64,
    pub max_vertical_scale: f64,
}

/// Wall-clock stage timings for one geometry fit, including unsuccessful fits.
#[derive(Default, Clone, Debug, Serialize)]
pub struct FitProfile {
    pub input_validation_seconds: f64,
    pub perspective_seconds: f64,
    pub staff_warp_seconds: f64,
    pub system_shear_seconds: f64,
    pub gcs_seconds: f64,
    pub direction_mesh_seconds: f64,
    pub vertical_correction_seconds: f64,
    pub staff_conditioning_seconds: f64,
    pub map_validation_seconds: f64,
    pub total_seconds: f64,
    pub unattributed_seconds: f64,
    pub failure_stage: Option<&'static str>,
}
