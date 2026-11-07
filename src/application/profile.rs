//! Per-stage timing and explicit failure reports.
use std::{fs, path::Path, time::Instant};

use serde::Serialize;

#[derive(Clone, Default, Serialize)]
pub(super) struct StepProfile {
    pub(super) decode_seconds: f64,
    pub(super) source_crop_seconds: f64,
    pub(super) analysis_perspective_seconds: f64,
    pub(super) analysis_png_write_seconds: f64,
    pub(super) analysis_retry_seconds: f64,
    pub(super) source_png_write_seconds: f64,
    pub(super) native_detection_seconds: f64,
    pub(super) staff_edge_detection_seconds: f64,
    pub(super) text_analysis_render_seconds: f64,
    pub(super) text_detection_seconds: f64,
    pub(super) text_fit_seconds: f64,
    pub(super) observations_load_seconds: f64,
    pub(super) observations_write_seconds: f64,
    pub(super) input_validation_seconds: f64,
    pub(super) perspective_seconds: f64,
    pub(super) gcs_seconds: f64,
    pub(super) vertical_correction_seconds: f64,
    pub(super) staff_conditioning_seconds: f64,
    pub(super) staff_warp_seconds: f64,
    pub(super) system_shear_seconds: f64,
    pub(super) map_validation_seconds: f64,
    pub(super) failed_fit_stage_seconds: f64,
    pub(super) geometry_export_seconds: f64,
    pub(super) render_seconds: f64,
    pub(super) grayscale_seconds: f64,
    pub(super) binarization_seconds: f64,
    pub(super) print_normalization_seconds: f64,
    pub(super) print_border_seconds: f64,
    pub(super) print_antialias_seconds: f64,
    pub(super) binary_border_seconds: f64,
    pub(super) final_crop_seconds: f64,
    pub(super) png_write_seconds: f64,
    pub(super) report_write_seconds: f64,
    pub(super) total_seconds: f64,
    pub(super) unattributed_seconds: f64,
    pub(super) failure_stage: Option<&'static str>,
    #[serde(skip)]
    pub(super) started: Option<Instant>,
}

impl StepProfile {
    pub(super) fn restart_attempt(&mut self) {
        let elapsed = self.started.unwrap().elapsed().as_secs_f64();
        *self = Self {
            started: self.started,
            decode_seconds: self.decode_seconds,
            source_crop_seconds: self.source_crop_seconds,
            source_png_write_seconds: self.source_png_write_seconds,
            analysis_perspective_seconds: self.analysis_perspective_seconds,
            analysis_png_write_seconds: self.analysis_png_write_seconds,
            ..Self::default()
        };
        self.analysis_retry_seconds = (elapsed - self.measured_seconds()).max(0.);
    }
    pub(super) fn start() -> Self {
        Self {
            started: Some(Instant::now()),
            ..Self::default()
        }
    }
    fn measured_seconds(&self) -> f64 {
        self.decode_seconds
            + self.source_crop_seconds
            + self.analysis_perspective_seconds
            + self.analysis_png_write_seconds
            + self.analysis_retry_seconds
            + self.source_png_write_seconds
            + self.native_detection_seconds
            + self.staff_edge_detection_seconds
            + self.text_analysis_render_seconds
            + self.text_detection_seconds
            + self.text_fit_seconds
            + self.observations_load_seconds
            + self.observations_write_seconds
            + self.input_validation_seconds
            + self.perspective_seconds
            + self.gcs_seconds
            + self.vertical_correction_seconds
            + self.staff_conditioning_seconds
            + self.staff_warp_seconds
            + self.system_shear_seconds
            + self.map_validation_seconds
            + self.failed_fit_stage_seconds
            + self.geometry_export_seconds
            + self.render_seconds
            + self.grayscale_seconds
            + self.binarization_seconds
            + self.print_normalization_seconds
            + self.print_border_seconds
            + self.print_antialias_seconds
            + self.binary_border_seconds
            + self.final_crop_seconds
            + self.png_write_seconds
            + self.report_write_seconds
    }
}

pub(super) fn finish_profile<T>(
    out: &Path,
    profile: &mut StepProfile,
    result: &Result<T, Box<dyn std::error::Error>>,
) -> Result<(), Box<dyn std::error::Error>> {
    profile.total_seconds = profile.started.unwrap().elapsed().as_secs_f64();
    profile.unattributed_seconds = (profile.total_seconds - profile.measured_seconds()).max(0.);
    if result.is_ok() {
        profile.failure_stage = None;
    }
    // The final profile JSON writes are excluded to avoid self-measurement.
    fs::write(
        out.join("profile.json"),
        serde_json::to_vec_pretty(profile)?,
    )?;
    let report_path = out.join("report.json");
    let mut report: serde_json::Value = if result.is_ok() {
        if report_path.exists() {
            serde_json::from_slice(&fs::read(&report_path)?)?
        } else {
            serde_json::json!({"status":"success"})
        }
    } else {
        serde_json::json!({"status":"failed", "error":result.as_ref().err().map(ToString::to_string)})
    };
    report["profile"] = serde_json::to_value(profile)?;
    fs::write(report_path, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
