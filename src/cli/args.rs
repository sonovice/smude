//! Command-line contract and analysis-resolution policy.
use std::path::PathBuf;

use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(super) enum PageLayout {
    Auto,
    Single,
    Double,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(super) enum SpreadLayout {
    SideBySide,
    Stacked,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(super) enum CropMode {
    Page,
    Content,
}

#[derive(Parser)]
#[command(
    name = "smude-v2",
    version,
    about = "Native GCS and staff-driven rectification with default ONNX page detection"
)]
pub(super) struct Args {
    /// Existing Smude barlines.json in original-image pixels.
    #[arg(long)]
    pub(super) geometry: Option<PathBuf>,
    /// Optional oracle/manual TL,TR,BR,BL corners (four Point objects as JSON).
    #[arg(long, requires = "geometry")]
    pub(super) geometry_corners: Option<PathBuf>,
    /// Dense source-paper contour for supplied geometry and final pixel masking.
    #[arg(long, requires = "geometry")]
    pub(super) geometry_contour: Option<PathBuf>,
    /// Physical page height/width for supplied geometry (page mode uses its model).
    #[arg(long, requires = "geometry")]
    pub(super) page_aspect: Option<f64>,
    /// Explicitly enable page detection (already the default; compatibility flag).
    #[arg(long, hide = true, conflicts_with_all = ["no_pages", "geometry", "same_size", "probe_points"])]
    pub(super) pages: bool,
    /// Process an already cropped image without page detection.
    #[arg(long, conflicts_with_all = ["pages", "page_regions"])]
    pub(super) no_pages: bool,
    /// page-boundary model page layout; automatic classification is the default.
    #[arg(long, value_enum, default_value = "auto")]
    pub(super) page_layout: PageLayout,
    /// Arrangement of a double-page capture; independent of image aspect ratio.
    #[arg(long, value_enum, default_value = "side-by-side")]
    pub(super) spread_layout: SpreadLayout,
    /// Prepared ONNX weights: page layout and contour/aspect head.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/models/page-boundary"))]
    pub(super) page_models: PathBuf,
    /// Explicit PageRegions JSON in original-image pixel coordinates.
    #[arg(long, conflicts_with_all = ["geometry", "same_size", "probe_points"])]
    pub(super) page_regions: Option<PathBuf>,
    #[arg(long)]
    pub(super) image: PathBuf,
    #[arg(long)]
    pub(super) out: PathBuf,
    /// Enable projective correction (the default; retained for explicit commands).
    #[arg(long, overrides_with = "no_perspective")]
    pub(super) perspective: bool,
    /// Disable the default projective stage and only flatten staff curves.
    #[arg(long, overrides_with = "perspective")]
    pub(super) no_perspective: bool,
    /// Disable native GCS and retain the projective/local geometry comparison.
    #[arg(long)]
    pub(super) no_gcs: bool,
    /// Disable the default bounded, evidence-gated x-only system correction.
    #[arg(long)]
    pub(super) no_vertical_correction: bool,
    /// Prefer a guarded projective page frame from detected near-straight sides.
    #[arg(long, overrides_with = "no_page_perspective")]
    pub(super) page_perspective: bool,
    /// Use only the staff-derived projective component, without page framing.
    #[arg(long, overrides_with = "page_perspective")]
    pub(super) no_page_perspective: bool,
    /// Ablation: detect in the original crop instead of the page-rectified analysis image.
    #[arg(long)]
    pub(super) no_analysis_perspective: bool,
    /// Disable local text curvature correction (music geometry is always protected).
    #[arg(long)]
    pub(super) no_text_dewarp: bool,
    /// Disable independently measured staff-start/end geometry observations.
    #[arg(long)]
    pub(super) no_staff_edges: bool,
    /// Opt into one shared final shear per system (not valid for all deformations).
    #[arg(long)]
    pub(super) system_shear: bool,
    /// Explicit same-size groups: '0,1,2,3;4,5'. Ossias stay separate.
    #[arg(long)]
    pub(super) same_size: Option<String>,
    #[arg(long, default_value_t = 1.)]
    pub(super) size_prior: f64,
    #[arg(long)]
    pub(super) no_render: bool,
    /// Export native classical detections without fitting or rendering.
    #[arg(long)]
    pub(super) detect_only: bool,
    /// Optional analysis long-edge cap. Default policy remains max width 1800.
    #[arg(long, value_parser = clap::value_parser!(u32).range(64..), conflicts_with = "analysis_width")]
    pub(super) analysis_long_edge: Option<u32>,
    /// Legacy analysis width override, for reproducing earlier runs.
    #[arg(long, value_parser = clap::value_parser!(u32).range(64..))]
    pub(super) analysis_width: Option<u32>,
    /// Inward binary page-border margin as a fraction of the output short edge.
    /// Zero disables cleanup. Applies only when a page contour is available.
    #[arg(long, default_value_t = 0.004)]
    pub(super) binary_page_margin: f64,
    /// Retain the full reconstructed canvas instead of final page/content framing.
    #[arg(long)]
    pub(super) no_final_crop: bool,
    /// Frame the detected paper (default), or explicitly crop tightly to ink.
    #[arg(long, value_enum, default_value = "page")]
    pub(super) crop_mode: CropMode,
    /// Padding around final page/content bounds, as a fraction of the short edge.
    #[arg(long, default_value_t = 0.01)]
    pub(super) crop_padding: f64,
    /// Optional input-pixel probes to transform for independent external QA.
    #[arg(long)]
    pub(super) probe_points: Option<PathBuf>,
}

impl Args {
    pub(super) fn uses_page_mode(&self) -> bool {
        self.page_regions.is_some()
            || self.pages
            || (cfg!(feature = "onnx-pages") && !self.no_pages && self.geometry.is_none())
    }

    pub(super) fn analysis_width_for(&self, image: &image::RgbImage) -> u32 {
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

#[cfg(test)]
mod output_defaults_tests {
    use super::*;
    use smude_geometry::Config;

    #[test]
    fn analysis_limit_uses_long_edge_without_upscaling_and_keeps_legacy_override() {
        use clap::Parser;
        let args = super::Args::parse_from([
            "smude-v2",
            "--image",
            "unused",
            "--out",
            "unused",
            "--analysis-long-edge",
            "1800",
        ]);
        assert_eq!(
            args.analysis_width_for(&image::RgbImage::new(3000, 4000)),
            1350
        );
        assert_eq!(
            args.analysis_width_for(&image::RgbImage::new(4000, 3000)),
            1800
        );
        assert_eq!(
            args.analysis_width_for(&image::RgbImage::new(600, 800)),
            600
        );
        let old = super::Args::parse_from([
            "smude-v2",
            "--image",
            "unused",
            "--out",
            "unused",
            "--analysis-width",
            "1800",
        ]);
        assert_eq!(
            old.analysis_width_for(&image::RgbImage::new(3000, 4000)),
            1800
        );
    }

    #[test]
    fn page_detection_is_default_with_explicit_bypass_and_manual_inputs() {
        let parse = |flags: &[&str]| {
            let mut command = vec!["smude-v2", "--image", "source.png", "--out", "result"];
            command.extend_from_slice(flags);
            Args::try_parse_from(command)
        };
        assert_eq!(
            parse(&[]).unwrap().uses_page_mode(),
            cfg!(feature = "onnx-pages")
        );
        assert!(!parse(&["--no-pages"]).unwrap().uses_page_mode());
        assert!(
            !parse(&["--geometry", "saved.json"])
                .unwrap()
                .uses_page_mode()
        );
        assert!(
            parse(&["--page-regions", "regions.json"])
                .unwrap()
                .uses_page_mode()
        );
        assert!(parse(&["--pages"]).unwrap().uses_page_mode());
        assert!(parse(&["--pages", "--no-pages"]).is_err());
        assert!(parse(&["--no-pages", "--page-regions", "regions.json"]).is_err());
    }

    #[test]
    fn perspective_is_default_and_explicit_flags_override_each_other() {
        let parse = |flags: &[&str]| {
            let mut command = vec!["smude-v2", "--image", "source.png", "--out", "result"];
            command.extend_from_slice(flags);
            Args::try_parse_from(command).unwrap()
        };
        for flags in [
            vec![],
            vec!["--perspective"],
            vec!["--no-perspective", "--perspective"],
        ] {
            let args = parse(&flags);
            assert!(args.perspective || !args.no_perspective);
            assert!(!args.system_shear);
            assert_eq!(args.uses_page_mode(), cfg!(feature = "onnx-pages"));
        }
        for flags in [
            vec!["--no-perspective"],
            vec!["--perspective", "--no-perspective"],
        ] {
            let args = parse(&flags);
            assert!(!args.perspective && args.no_perspective);
        }
        assert!(Config::default().perspective);
        assert!(!Config::default().system_shear);
    }
}
