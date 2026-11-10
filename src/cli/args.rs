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

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(super) enum RenderBackend {
    Auto,
    Cpu,
    Gpu,
}

#[derive(Parser)]
#[command(
    name = "smude-v2",
    version,
    about = "Staff-driven mesh rectification with default ONNX page detection"
)]
pub(super) struct Args {
    /// Also write source/analysis images and color, grayscale and binary variants.
    #[arg(long)]
    pub(super) diagnostics: bool,
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
    /// Override the bundled page-layout and contour/aspect ONNX weights.
    #[arg(long)]
    pub(super) page_models: Option<PathBuf>,
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
    /// Explicitly select the default page/staff/direction mesh.
    #[arg(long, conflicts_with_all = ["no_gcs", "gcs", "no_perspective"])]
    pub(super) direction_mesh: bool,
    /// Use the native GCS model instead of the default mesh.
    #[arg(long, conflicts_with_all = ["no_gcs", "direction_mesh", "no_perspective"])]
    pub(super) gcs: bool,
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
    /// Offscreen image renderer; auto uses GPU for large output rasters.
    #[arg(long, value_enum, default_value = "auto")]
    pub(super) render_backend: RenderBackend,
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
    /// Disable automatic A4 framing for DIN-like page estimates.
    #[arg(long)]
    pub(super) no_a4: bool,
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

impl From<Args> for smude_geometry::application::Options {
    fn from(args: Args) -> Self {
        use smude_geometry::application as engine;
        Self {
            processing_progress: None,
            progress: None,
            diagnostics: args.diagnostics,
            geometry: args.geometry,
            geometry_corners: args.geometry_corners,
            geometry_contour: args.geometry_contour,
            page_aspect: args.page_aspect,
            pages: args.pages,
            no_pages: args.no_pages,
            page_layout: match args.page_layout {
                PageLayout::Auto => engine::PageLayout::Auto,
                PageLayout::Single => engine::PageLayout::Single,
                PageLayout::Double => engine::PageLayout::Double,
            },
            spread_layout: match args.spread_layout {
                SpreadLayout::SideBySide => engine::SpreadLayout::SideBySide,
                SpreadLayout::Stacked => engine::SpreadLayout::Stacked,
            },
            page_models: args.page_models,
            page_regions: args.page_regions,
            image: args.image,
            out: args.out,
            perspective: args.perspective,
            no_perspective: args.no_perspective,
            no_gcs: args.no_gcs,
            direction_mesh: args.direction_mesh
                || !(args.gcs || args.no_gcs || args.no_perspective),
            no_vertical_correction: args.no_vertical_correction,
            page_perspective: args.page_perspective,
            no_page_perspective: args.no_page_perspective,
            no_analysis_perspective: args.no_analysis_perspective,
            no_text_dewarp: args.no_text_dewarp,
            no_staff_edges: args.no_staff_edges,
            system_shear: args.system_shear,
            same_size: args.same_size,
            size_prior: args.size_prior,
            no_render: args.no_render,
            render_backend: match args.render_backend {
                RenderBackend::Auto => smude_geometry::RenderBackend::Auto,
                RenderBackend::Cpu => smude_geometry::RenderBackend::Cpu,
                RenderBackend::Gpu => smude_geometry::RenderBackend::Gpu,
            },
            detect_only: args.detect_only,
            analysis_long_edge: args.analysis_long_edge,
            analysis_width: args.analysis_width,
            binary_page_margin: args.binary_page_margin,
            no_final_crop: args.no_final_crop,
            no_a4: args.no_a4,
            crop_mode: match args.crop_mode {
                CropMode::Page => engine::CropMode::Page,
                CropMode::Content => engine::CropMode::Content,
            },
            crop_padding: args.crop_padding,
            probe_points: args.probe_points,
        }
    }
}

#[cfg(test)]
mod output_defaults_tests {
    use super::*;
    use smude_geometry::{Config, application::Options};

    #[test]
    fn analysis_limit_uses_long_edge_without_upscaling_and_keeps_legacy_override() {
        use clap::Parser;
        let args: Options = super::Args::parse_from([
            "smude-v2",
            "--image",
            "unused",
            "--out",
            "unused",
            "--analysis-long-edge",
            "1800",
        ])
        .into();
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
        let old: Options = super::Args::parse_from([
            "smude-v2",
            "--image",
            "unused",
            "--out",
            "unused",
            "--analysis-width",
            "1800",
        ])
        .into();
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
            Args::try_parse_from(command).map(Options::from)
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
    fn mesh_is_default_with_explicit_gcs_and_legacy_ablations() {
        let parse = |flags: &[&str]| {
            let mut command = vec!["smude-v2", "--image", "source.png", "--out", "result"];
            command.extend_from_slice(flags);
            Args::try_parse_from(command).map(Options::from)
        };
        assert!(Options::default().direction_mesh);
        assert!(parse(&[]).unwrap().direction_mesh);
        assert!(parse(&["--direction-mesh"]).unwrap().direction_mesh);
        assert!(!parse(&["--gcs"]).unwrap().direction_mesh);
        assert!(!parse(&["--no-gcs"]).unwrap().direction_mesh);
        assert!(!parse(&["--no-perspective"]).unwrap().direction_mesh);
        assert!(parse(&["--gcs", "--direction-mesh"]).is_err());
        assert!(parse(&["--gcs", "--no-gcs"]).is_err());
    }

    #[test]
    fn perspective_is_default_and_explicit_flags_override_each_other() {
        let parse = |flags: &[&str]| {
            let mut command = vec!["smude-v2", "--image", "source.png", "--out", "result"];
            command.extend_from_slice(flags);
            Options::from(Args::try_parse_from(command).unwrap())
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
