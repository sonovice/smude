//! One processing engine for the CLI and desktop app.
mod options;
mod pages;
mod preview_map;
mod process;
mod profile;
mod progress;
pub use preview_map::PreviewMap;
pub use progress::{DetectionProgress, DetectionStage, ProgressObserver};

use crate::Observations;
pub use options::{CropMode, Options, PageLayout, SpreadLayout};
use pages::process_pages;
use process::process;
use profile::{StepProfile, finish_profile};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OutputKind {
    #[default]
    Clean,
    Gray,
    Binary,
    Color,
}
impl OutputKind {
    pub fn filename(self) -> &'static str {
        match self {
            Self::Clean => "rectified-clean.png",
            Self::Gray => "rectified-gray.png",
            Self::Binary => "rectified-binary.png",
            Self::Color => "rectified.png",
        }
    }
}
#[derive(Clone, Debug)]
pub struct RenderedPage {
    pub id: String,
    pub directory: PathBuf,
    pub mapping: Option<std::sync::Arc<PreviewMap>>,
}
impl RenderedPage {
    pub fn image(&self, kind: OutputKind) -> PathBuf {
        self.directory.join(kind.filename())
    }
}
#[derive(Clone, Debug)]
pub struct ProcessingResult {
    pub pages: Vec<RenderedPage>,
    pub warnings: Vec<String>,
}

/// Product settings. Geometry and output processing keep their validated defaults.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Settings {
    pub detect_pages: bool,
    pub page_layout: PageLayout,
    pub spread_layout: SpreadLayout,
    pub text_correction: bool,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            detect_pages: true,
            page_layout: PageLayout::Auto,
            spread_layout: SpreadLayout::SideBySide,
            text_correction: true,
        }
    }
}
/// Process one input into a caller-owned working directory. Outputs sample the
/// original at full resolution. Partial page success is returned with warnings.
pub fn process_file(
    input: &Path,
    output: &Path,
    settings: &Settings,
) -> Result<ProcessingResult, Box<dyn std::error::Error>> {
    process_file_with_progress(input, output, settings, None)
}
/// Process with optional live measurements; image results are identical to `process_file`.
pub fn process_file_with_progress(
    input: &Path,
    output: &Path,
    settings: &Settings,
    progress: Option<ProgressObserver>,
) -> Result<ProcessingResult, Box<dyn std::error::Error>> {
    run(&Options {
        progress,
        image: input.to_owned(),
        out: output.to_owned(),
        no_pages: !settings.detect_pages,
        page_layout: settings.page_layout,
        spread_layout: settings.spread_layout,
        no_text_dewarp: !settings.text_correction,
        ..Options::default()
    })
}

/// Advanced processing entry point shared with the CLI. No argument parsing.
pub fn run(args: &Options) -> Result<ProcessingResult, Box<dyn std::error::Error>> {
    if !args.binary_page_margin.is_finite() || !(0. ..=0.02).contains(&args.binary_page_margin) {
        return Err("binary page margin must be a finite fraction in 0..0.02".into());
    }
    if !args.crop_padding.is_finite() || !(0. ..=0.1).contains(&args.crop_padding) {
        return Err("crop padding must be a finite fraction in 0..0.1".into());
    }
    if args
        .page_aspect
        .is_some_and(|r| !r.is_finite() || !(0.1..=10.).contains(&r))
    {
        return Err("page aspect must be finite in 0.1..10".into());
    }
    if args.uses_page_mode() {
        if args.same_size.is_some() || args.probe_points.is_some() {
            return Err("--same-size and --probe-points require --no-pages or --geometry".into());
        }
        return process_pages(args);
    }
    fs::create_dir_all(&args.out)?;
    let mut profile = StepProfile::start();
    let result = (|| {
        let mut source = None;
        let detection_start = Instant::now();
        let data: Observations = if let Some(path) = &args.geometry {
            profile.failure_stage = Some("observations_load");
            let start = Instant::now();
            let loaded = fs::read(path)
                .map_err(|e| -> Box<dyn std::error::Error> { e.into() })
                .and_then(|b| Ok(serde_json::from_slice(&b)?));
            profile.observations_load_seconds = start.elapsed().as_secs_f64();
            loaded?
        } else {
            profile.failure_stage = Some("decode");
            let start = Instant::now();
            let decoded = image::open(&args.image).map(|image| image.to_rgb8());
            profile.decode_seconds = start.elapsed().as_secs_f64();
            source = Some(decoded?);
            profile.failure_stage = Some("native_detection");
            let start = Instant::now();
            let on_staves = |staves: &[crate::Staff]| args.emit_staves("1", staves, |p| p);
            let detected = crate::detect::detect_masked(
                source.as_ref().unwrap(),
                args.analysis_width_for(source.as_ref().unwrap()),
                None,
                Some(&on_staves),
            );
            profile.native_detection_seconds = start.elapsed().as_secs_f64();
            detected?
        };
        let detection_seconds = detection_start.elapsed().as_secs_f64();
        process(
            args,
            &args.out,
            &data,
            source,
            detection_seconds,
            None,
            &mut profile,
        )
    })();
    finish_profile(&args.out, &mut profile, &result)?;
    result.map(|summary| ProcessingResult {
        pages: if args.no_render || args.detect_only {
            vec![]
        } else {
            vec![RenderedPage {
                id: "1".into(),
                directory: args.out.clone(),
                mapping: summary.mapping,
            }]
        },
        warnings: vec![],
    })
}
