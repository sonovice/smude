//! ONNX page proposals and optional full interior page geometry.
use super::filter;
use crate::{Error, Result};
pub use filter::RejectedPage;

use crate::detection::{PageRegion, PageRegions};

use crate::geometry::Point;
use image::RgbImage;
use onnx_runtime::{
    environment::Environment,
    session::{
        Session,
        builder::{AutoDevicePolicy, SessionBuilder},
    },
    value::TensorRef,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
    time::Instant,
};

pub const PAGE_CLASSES: [&str; 5] = [
    "no_page",
    "one_page",
    "one_page_bound",
    "two_page_landscape",
    "two_page_portrait",
];

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PageLayoutMode {
    #[default]
    Auto,
    Single,
    Double,
}

/// Arrangement of two pages, independent of the canvas aspect ratio.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PageSpreadLayout {
    #[default]
    SideBySide,
    Stacked,
}

#[derive(Clone, Debug, Serialize)]
pub struct PageDetectionReport {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub model_grids_source: Vec<Vec<Point>>,
    pub full_geometry_model_load_seconds: f64,
    pub class_scores: Option<[f32; 5]>,
    pub class_names: [&'static str; 5],
    pub no_page_score: Option<f32>,
    pub selected_layout: PageLayoutMode,
    pub spread_layout: PageSpreadLayout,
    pub selected_class: Option<&'static str>,
    pub warnings: Vec<String>,
    pub rejected_pages: Vec<RejectedPage>,
    pub source_view_order: Vec<&'static str>,
    pub score_note: &'static str,
    pub classification_seconds: f64,
    pub preprocessing_seconds: f64,
    pub boundary_inference_seconds: f64,
    pub contour_seconds: f64,
    pub page_filter_seconds: f64,
    pub total_seconds: f64,
    pub source_views: Vec<[u32; 4]>,
    pub padded_sizes: Vec<[u32; 2]>,
    /// Unclipped perimeter points after unpadding and restoring source-view offset.
    pub raw_boundaries_source: Vec<Vec<Point>>,
    /// Height / width, in source-view order; invalid estimates are omitted.
    pub height_width_ratios: Vec<Option<f64>>,
}

fn model_error(error: impl std::fmt::Display) -> Error {
    Error::PageModel(error.to_string())
}

fn onnx_session_builder(accelerated: bool, profile: Option<&Path>) -> Result<SessionBuilder> {
    static ENVIRONMENT: OnceLock<std::result::Result<Environment, String>> = OnceLock::new();
    let environment = ENVIRONMENT
        .get_or_init(|| {
            onnx_runtime::init()
                .with_telemetry(false)
                .build()
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(model_error)?;
    let builder = Session::builder(environment).map_err(model_error)?;
    let builder = if accelerated {
        #[cfg(target_os = "ios")]
        {
            builder
                .with_execution_providers([onnx_runtime::ep::CoreML::default()
                    .with_model_format(onnx_runtime::ep::coreml::ModelFormat::NeuralNetwork)
                    .with_compute_units(match std::env::var("SMUDE_COREML_UNITS").as_deref() {
                        Ok("cpu") => onnx_runtime::ep::coreml::ComputeUnits::CPUOnly,
                        Ok("gpu") => onnx_runtime::ep::coreml::ComputeUnits::CPUAndGPU,
                        _ => onnx_runtime::ep::coreml::ComputeUnits::All,
                    })
                    .with_low_precision_accumulation_on_gpu(false)
                    .with_static_input_shapes(true)
                    .build()
                    .error_on_failure()])
                .map_err(model_error)?
        }
        #[cfg(target_os = "android")]
        {
            builder
                .with_execution_providers([onnx_runtime::ep::NNAPI::default()
                    .with_disable_cpu(true)
                    .build()
                    .error_on_failure()])
                .map_err(model_error)?
        }
        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        {
            return Err(model_error(
                "mobile acceleration benchmark requires iOS or Android",
            ));
        }
    } else {
        builder
    };
    let builder = builder
        .with_auto_device(AutoDevicePolicy::PreferCPU)
        .map_err(model_error)?
        .with_intra_threads(4)
        .map_err(model_error)?
        .with_inter_threads(1)
        .map_err(model_error)?;
    if let Some(profile) = profile {
        builder.with_profiling(profile).map_err(model_error)
    } else {
        Ok(builder)
    }
}

/// NHWC model input: portrait 3:4, RGB plus a constant alpha channel.
const INPUT_WIDTH: u32 = 144;
const INPUT_HEIGHT: u32 = 192;
const INPUT_CHANNELS: usize = 4;

struct ImageModel {
    session: Mutex<Session>,
    output_len: usize,
}
impl ImageModel {
    fn load(directory: &Path, name: &str, output_len: usize) -> Result<Self> {
        let path = directory.join(format!("{name}.onnx"));
        let session = onnx_session_builder(false, None)?
            .commit_from_file(&path)
            .map_err(|e| model_error(format!("loading {}: {e}", path.display())))?;
        Ok(Self {
            session: Mutex::new(session),
            output_len,
        })
    }
    fn embedded(
        name: &str,
        bytes: &[u8],
        output_len: usize,
        accelerated: bool,
        directory: Option<&Path>,
    ) -> Result<Self> {
        let profile = directory.map(|p| p.join(name));
        let session = onnx_session_builder(accelerated, profile.as_deref())?
            .commit_from_memory(bytes)
            .map_err(|e| model_error(format!("loading embedded {name}: {e}")))?;
        Ok(Self {
            session: Mutex::new(session),
            output_len,
        })
    }
    fn run(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != (INPUT_WIDTH * INPUT_HEIGHT) as usize * INPUT_CHANNELS {
            return Err(model_error("unexpected NHWC input length"));
        }
        let mut session = self
            .session
            .lock()
            .map_err(|_| model_error("ONNX session lock poisoned"))?;
        let shape = [
            1,
            INPUT_HEIGHT as usize,
            INPUT_WIDTH as usize,
            INPUT_CHANNELS,
        ];
        let tensor = TensorRef::from_array_view((shape, input)).map_err(model_error)?;
        let outputs = session
            .run(onnx_runtime::inputs![tensor])
            .map_err(model_error)?;
        let values = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(model_error)?
            .1
            .to_vec();
        if values.len() != self.output_len || values.iter().any(|v| !v.is_finite()) {
            return Err(model_error(format!(
                "invalid model output: {} values, expected {} finite float32 values",
                values.len(),
                self.output_len
            )));
        }
        Ok(values)
    }
}

/// Loads only the classifier and compact boundary model, using ONNX Runtime CPU.
pub struct PageDetector {
    classifier: ImageModel,
    boundary: ImageModel,
    geometry: OnceLock<std::result::Result<ImageModel, String>>,
    model_dir: Option<PathBuf>,
}
impl PageDetector {
    /// Load the bundled weights directly from executable memory, without files.
    pub fn embedded() -> Result<Self> {
        Self::embedded_options(false, None)
    }

    /// The bundled detector, loaded once per process: later scans skip session setup.
    pub fn embedded_shared() -> Result<&'static Self> {
        static SHARED: OnceLock<std::result::Result<PageDetector, String>> = OnceLock::new();
        SHARED
            .get_or_init(|| Self::embedded().map_err(|e| e.to_string()))
            .as_ref()
            .map_err(model_error)
    }

    /// Device QA only: profile node placement, latency and coordinate parity.
    /// Registration failures are errors, never silently reported as acceleration.
    /// Default application processing continues to use the measured CPU backend.
    pub fn embedded_profiled(accelerated: bool, directory: &Path) -> Result<Self> {
        Self::embedded_options(accelerated, Some(directory))
    }

    pub fn finish_profiling(&self) -> Result<Vec<String>> {
        [&self.classifier, &self.boundary]
            .into_iter()
            .map(|model| {
                model
                    .session
                    .lock()
                    .map_err(|_| model_error("ONNX session lock poisoned"))?
                    .end_profiling()
                    .map_err(model_error)
            })
            .collect()
    }

    fn embedded_options(accelerated: bool, directory: Option<&Path>) -> Result<Self> {
        const LAYOUT: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/models/page-boundary/page-layout.onnx"
        ));
        const BOUNDARY: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/models/page-boundary/page-boundary.onnx"
        ));
        Ok(Self {
            classifier: ImageModel::embedded("page-layout", LAYOUT, 5, accelerated, directory)?,
            boundary: ImageModel::embedded("page-boundary", BOUNDARY, 193, accelerated, directory)?,
            geometry: OnceLock::new(),
            model_dir: None,
        })
    }

    /// Explicitly override bundled weights with externally supplied models.
    pub fn load(model_dir: &Path) -> Result<Self> {
        Ok(Self {
            classifier: ImageModel::load(model_dir, "page-layout", 5)?,
            boundary: ImageModel::load(model_dir, "page-boundary", 193)?,
            geometry: OnceLock::new(),
            model_dir: Some(model_dir.into()),
        })
    }
    pub fn detect(
        &self,
        image: &RgbImage,
        layout: PageLayoutMode,
    ) -> Result<(PageRegions, PageDetectionReport)> {
        self.detect_with_spread_layout(image, layout, PageSpreadLayout::default())
    }
    pub fn detect_with_spread_layout(
        &self,
        image: &RgbImage,
        layout: PageLayoutMode,
        spread_layout: PageSpreadLayout,
    ) -> Result<(PageRegions, PageDetectionReport)> {
        self.detect_impl(image, layout, spread_layout, false)
    }

    /// Batch processing uses all predicted page points. Live camera proposals
    /// can keep the compact boundary head without loading the interior head.
    pub fn detect_full_with_spread_layout(
        &self,
        image: &RgbImage,
        layout: PageLayoutMode,
        spread_layout: PageSpreadLayout,
    ) -> Result<(PageRegions, PageDetectionReport)> {
        self.detect_impl(image, layout, spread_layout, true)
    }

    fn detect_impl(
        &self,
        image: &RgbImage,
        layout: PageLayoutMode,
        spread_layout: PageSpreadLayout,
        full_geometry: bool,
    ) -> Result<(PageRegions, PageDetectionReport)> {
        let started = Instant::now();
        let load_started = Instant::now();
        let geometry_model = if full_geometry {
            Some(
                self.geometry
                    .get_or_init(|| {
                        if let Some(dir) = &self.model_dir {
                            ImageModel::load(dir, "page-geometry", 1251)
                        } else {
                            ImageModel::embedded(
                                "page-geometry",
                                include_bytes!(concat!(
                                    env!("CARGO_MANIFEST_DIR"),
                                    "/models/page-boundary/page-geometry.onnx"
                                )),
                                1251,
                                false,
                                None,
                            )
                        }
                        .map_err(|e| e.to_string())
                    })
                    .as_ref()
                    .map_err(model_error)?,
            )
        } else {
            None
        };
        let full_geometry_model_load_seconds = if full_geometry {
            load_started.elapsed().as_secs_f64()
        } else {
            0.
        };
        let (w, h) = image.dimensions();
        if w < 2 || h < 2 || w.max(h) > 12000 {
            return Err(model_error("input dimensions must be 2..12000 pixels"));
        }
        let mut preprocessing_seconds = 0.;
        let mut classification_seconds = 0.;
        let full_view = [0, 0, w, h];
        // The classifier input also serves a single full-image source view.
        let mut full_input = None;
        let scores = if layout == PageLayoutMode::Auto {
            let t = Instant::now();
            let input = full_input.insert(model_input(image, full_view, INPUT_WIDTH, INPUT_HEIGHT));
            preprocessing_seconds += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let values = self
                .classifier
                .run(&input.tensor)
                .map_err(|e| model_error(format!("page classification failed: {e}")))?;
            classification_seconds = t.elapsed().as_secs_f64();
            Some(
                <[f32; 5]>::try_from(values)
                    .map_err(|_| model_error("classifier output length"))?,
            )
        } else {
            None
        };
        let selected = selected_layout(layout, scores.as_ref());
        let source_views = source_views(w, h, selected, spread_layout);
        let mut padded_sizes = vec![];
        let mut raw_boundaries_source = vec![];
        let mut height_width_ratios = vec![];
        let mut regions = vec![];
        let mut boundary_inference_seconds = 0.;
        let mut contour_seconds = 0.;
        let mut model_grids_source = vec![];
        for (index, view) in source_views.iter().copied().enumerate() {
            let t = Instant::now();
            let ModelInput {
                tensor: input,
                padded: size,
                offset,
            } = full_input
                .take()
                .filter(|_| view == full_view)
                .unwrap_or_else(|| model_input(image, view, INPUT_WIDTH, INPUT_HEIGHT));
            preprocessing_seconds += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let mut raw = geometry_model
                .unwrap_or(&self.boundary)
                .run(&input)
                .map_err(|e| {
                    model_error(format!(
                        "page-boundary source view {} failed: {e}",
                        index + 1
                    ))
                })?;
            boundary_inference_seconds += t.elapsed().as_secs_f64();
            let t = Instant::now();
            if full_geometry {
                // The numeric index array is also a Rust array expression,
                // checked at compile time without a runtime JSON dependency.
                const ORDER: &[usize; 1251] = &include!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/models/page-boundary/page-geometry-order.json"
                ));
                if raw.len() != 1251 {
                    return Err(model_error("full page geometry shape"));
                }
                let mut decoded = vec![0_f32; 1251];
                for (value, &index) in raw.iter().zip(ORDER) {
                    decoded[index] = *value;
                }
                model_grids_source.push(source_points(&decoded[..1250], size, offset, view)?);
                let indices = (0..25)
                    .chain((1..25).map(|j| j * 25 + 24))
                    .chain((0..24).rev().map(|i| 24 * 25 + i))
                    .chain((1..24).rev().map(|j| j * 25));
                raw = indices
                    .flat_map(|i| [decoded[2 * i], decoded[2 * i + 1]])
                    .collect();
                raw.push(decoded[1250]);
            }
            height_width_ratios.push(height_width_ratio(&raw));
            let boundary = boundary_source(&raw, size, offset, view)?;
            let contour = clip_polygon(&boundary, w, h);
            regions.push(PageRegion {
                id: (index + 1).to_string(),
                contour,
                confidence: 0.,
                evidence: vec![
                    if full_geometry {"ONNX full page grid: 625 interior/perimeter points and aspect"}
                        else {"ONNX contour-only head: 96 perimeter points and aspect; no interior outputs"}.into(),
                    "uncalibrated model boundary/confidence unavailable".into(),
                ],
            });
            raw_boundaries_source.push(boundary);
            padded_sizes.push(size);
            contour_seconds += t.elapsed().as_secs_f64();
        }
        let filter_started = Instant::now();
        let rejected_pages = filter::filter(&mut regions, &raw_boundaries_source, w, h);
        let page_filter_seconds = filter_started.elapsed().as_secs_f64();
        let pages = PageRegions {
            width: w,
            height: h,
            regions,
        };
        let t = Instant::now();
        if !pages.regions.is_empty() {
            pages
                .validate()
                .map_err(|e| model_error(format!("invalid page boundary: {e}")))?;
        }
        contour_seconds += t.elapsed().as_secs_f64();
        let class_index = scores
            .as_ref()
            .and_then(|s| s.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)))
            .map(|(i, _)| i);
        let mut warnings = if class_index == Some(0) {
            vec!["Classifier selected no_page; original prototype policy still runs a single freeform boundary. This is not validated page presence.".into()]
        } else {
            vec![]
        };
        for (index, ratio) in height_width_ratios.iter().enumerate() {
            if ratio.is_none() {
                warnings.push(format!(
                    "Source view {} has invalid height/width ratio; preserve geometry aspect",
                    index + 1
                ));
            }
        }
        let source_view_order = if selected == PageLayoutMode::Single {
            vec!["full_image"]
        } else if spread_layout == PageSpreadLayout::SideBySide {
            vec!["left_view", "right_view"]
        } else {
            vec!["upper_view", "lower_view"]
        };
        let report = PageDetectionReport {
            model_grids_source,
            full_geometry_model_load_seconds,
            rejected_pages,
            spread_layout,
            selected_class: class_index.map(|i| PAGE_CLASSES[i]),
            warnings,
            source_view_order,
            class_scores: scores,
            class_names: PAGE_CLASSES,
            no_page_score: scores.map(|s| s[0]),
            selected_layout: selected,
            score_note: "Classifier outputs (including no_page) are model scores, not calibrated confidence or validated page presence. Forced layouts skip classification.",
            classification_seconds,
            preprocessing_seconds,
            boundary_inference_seconds,
            contour_seconds,
            page_filter_seconds,
            total_seconds: started.elapsed().as_secs_f64(),
            source_views,
            padded_sizes,
            raw_boundaries_source,
            height_width_ratios,
        };
        Ok((pages, report))
    }
}

fn selected_layout(requested: PageLayoutMode, scores: Option<&[f32; 5]>) -> PageLayoutMode {
    match requested {
        PageLayoutMode::Auto => {
            if scores
                .and_then(|v| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)))
                .is_some_and(|(i, _)| i >= 3)
            {
                PageLayoutMode::Double
            } else {
                PageLayoutMode::Single
            }
        }
        other => other,
    }
}
fn source_views(
    w: u32,
    h: u32,
    layout: PageLayoutMode,
    spread_layout: PageSpreadLayout,
) -> Vec<[u32; 4]> {
    if layout != PageLayoutMode::Double {
        vec![[0, 0, w, h]]
    } else if spread_layout == PageSpreadLayout::SideBySide {
        let span = w * 9 / 16;
        vec![[0, 0, span, h], [w - span, 0, span, h]]
    } else {
        let span = h * 9 / 16;
        vec![[0, 0, w, span], [0, h - span, w, span]]
    }
}
/// Size of the edge-replicated 3:4 portrait canvas around a `w`×`h` view and
/// the view's offset inside it.
fn portrait_padding(w: u32, h: u32) -> ([u32; 2], [u32; 2]) {
    let pw = w.max((h as f32 * 0.75) as u32);
    let ph = h.max((w as f32 / 0.75) as u32);
    ([pw, ph], [(pw - w) / 2, (ph - h) / 2])
}

/// First source sample and normalized weights of one output sample.
struct Taps {
    first: usize,
    weights: Vec<f32>,
}

/// Triangle-filter taps for resampling `size` samples to `new_size`, in the
/// exact f32 arithmetic of `image::imageops::resize`, which defined the
/// preprocessing of the model training data.
fn triangle_taps(size: u32, new_size: u32) -> Vec<Taps> {
    let ratio = size as f32 / new_size as f32;
    let scale = ratio.max(1.);
    (0..new_size)
        .map(|out| {
            let center = (out as f32 + 0.5) * ratio;
            let first = ((center - scale).floor() as i64).clamp(0, i64::from(size) - 1);
            let end = ((center + scale).ceil() as i64).clamp(first + 1, i64::from(size));
            let center = center - 0.5;
            let mut weights: Vec<f32> = (first..end)
                .map(|i| (1. - ((i as f32 - center) / scale).abs()).max(0.))
                .collect();
            let mut sum = 0.;
            for weight in &weights {
                sum += weight;
            }
            for weight in &mut weights {
                *weight /= sum;
            }
            Taps {
                first: first as usize,
                weights,
            }
        })
        .collect()
}

struct ModelInput {
    /// NHWC values in 0..=1 with constant alpha 1.
    tensor: Vec<f32>,
    padded: [u32; 2],
    offset: [u32; 2],
}

/// Model input of one source view `[x, y, width, height]`: portrait padding by
/// edge replication, then a separable Triangle resize, rows first.
///
/// Values equal an 8-bit `image::imageops::resize` of the materialized padded
/// crop; neither the crop nor the padding is copied at source resolution.
fn model_input(image: &RgbImage, view: [u32; 4], width: u32, height: u32) -> ModelInput {
    let [x0, y0, w, h] = view.map(|v| v as usize);
    let (padded, offset) = portrait_padding(view[2], view[3]);
    let [dx, dy] = offset.map(|v| v as usize);
    let stride = image.width() as usize * 3;
    let source_row = |padded_y: usize| {
        let y = y0 + padded_y.saturating_sub(dy).min(h - 1);
        &image.as_raw()[y * stride + x0 * 3..][..w * 3]
    };
    let columns = triangle_taps(padded[0], width);
    let rows = triangle_taps(padded[1], height);
    let mut tensor = vec![1_f32; (width * height) as usize * INPUT_CHANNELS];
    tensor
        .par_chunks_mut(width as usize * INPUT_CHANNELS)
        .zip(&rows)
        .for_each_init(
            || vec![0_f32; w * 3],
            |line, (output, taps)| {
                // Every sum accumulates its taps in source order, as the
                // reference does; only then are the results bit-identical.
                line.fill(0.);
                for (i, &weight) in taps.weights.iter().enumerate() {
                    for (sum, &value) in line.iter_mut().zip(source_row(taps.first + i)) {
                        *sum += f32::from(value) * weight;
                    }
                }
                for (pixel, taps) in output.chunks_exact_mut(INPUT_CHANNELS).zip(&columns) {
                    let mut sums = [0_f32; 3];
                    for (i, &weight) in taps.weights.iter().enumerate() {
                        let x = (taps.first + i).saturating_sub(dx).min(w - 1) * 3;
                        for (sum, value) in sums.iter_mut().zip(&line[x..x + 3]) {
                            *sum += value * weight;
                        }
                    }
                    for (value, sum) in pixel.iter_mut().zip(sums) {
                        *value = sum.clamp(0., 255.).round() / 255.;
                    }
                }
            },
        );
    ModelInput {
        tensor,
        padded,
        offset,
    }
}
fn height_width_ratio(raw: &[f32]) -> Option<f64> {
    if raw.len() != 193 {
        return None;
    }
    let ratio = f64::from(raw[192]);
    (ratio.is_finite() && (0.1..=10.).contains(&ratio)).then_some(ratio)
}

fn boundary_source(
    raw: &[f32],
    padded: [u32; 2],
    offset: [u32; 2],
    view: [u32; 4],
) -> Result<Vec<Point>> {
    if raw.len() != 193 {
        return Err(model_error(
            "expected 193 contour/aspect outputs; prepare the compact page model",
        ));
    }
    source_points(&raw[..192], padded, offset, view)
}

fn source_points(
    raw: &[f32],
    padded: [u32; 2],
    offset: [u32; 2],
    view: [u32; 4],
) -> Result<Vec<Point>> {
    raw.chunks_exact(2)
        .map(|pair| {
            let (x, y) = (pair[0], pair[1]);
            if !x.is_finite() || !y.is_finite() {
                return Err(model_error("non-finite boundary point"));
            }
            Ok(Point {
                x: ((x + 0.5) * padded[0] as f32) as f64 - offset[0] as f64 + view[0] as f64,
                y: ((y * 0.75 + 0.5) * padded[1] as f32) as f64 - offset[1] as f64 + view[1] as f64,
            })
        })
        .collect()
}
fn clip_polygon(points: &[Point], width: u32, height: u32) -> Vec<Point> {
    let mut polygon = points.to_vec();
    for (axis, bound, lower) in [
        (0, 0., true),
        (0, width as f64, false),
        (1, 0., true),
        (1, height as f64, false),
    ] {
        let coord = |p: Point| if axis == 0 { p.x } else { p.y };
        let inside = |p: Point| {
            if lower {
                coord(p) >= bound
            } else {
                coord(p) <= bound
            }
        };
        let intersect = |a: Point, b: Point| {
            let t = (bound - coord(a)) / (coord(b) - coord(a));
            let mut p = Point {
                x: a.x + t * (b.x - a.x),
                y: a.y + t * (b.y - a.y),
            };
            if axis == 0 {
                p.x = bound
            } else {
                p.y = bound
            };
            p
        };
        let previous = std::mem::take(&mut polygon);
        if previous.is_empty() {
            break;
        }
        let mut a = *previous.last().unwrap();
        for b in previous {
            match (inside(a), inside(b)) {
                (true, true) => polygon.push(b),
                (true, false) => polygon.push(intersect(a, b)),
                (false, true) => {
                    polygon.push(intersect(a, b));
                    polygon.push(b)
                }
                _ => {}
            }
            a = b;
        }
    }
    let mut unique: Vec<Point> = vec![];
    for p in polygon {
        if !unique.iter().any(|q| (p.x - q.x).hypot(p.y - q.y) < 1e-7) {
            unique.push(p)
        }
    }
    unique
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, imageops};

    /// The original preprocessing: copy the view, pad it and resize in 8 bit.
    fn reference_input(image: &RgbImage, view: [u32; 4], width: u32, height: u32) -> Vec<f32> {
        let crop = imageops::crop_imm(image, view[0], view[1], view[2], view[3]).to_image();
        let (w, h) = crop.dimensions();
        let ([pw, ph], [dx, dy]) = portrait_padding(w, h);
        let padded = RgbImage::from_fn(pw, ph, |x, y| {
            *crop.get_pixel(
                x.saturating_sub(dx).min(w - 1),
                y.saturating_sub(dy).min(h - 1),
            )
        });
        imageops::resize(&padded, width, height, imageops::FilterType::Triangle)
            .pixels()
            .flat_map(|p| {
                let [r, g, b] = p.0.map(|v| f32::from(v) / 255.);
                [r, g, b, 1.]
            })
            .collect()
    }
    fn rectangle() -> [f32; 193] {
        let mut d = [0.; 193];
        for i in 0..96 {
            let (x, y) = match i {
                0..=24 => (i as f32 / 24., 0.),
                25..=48 => (1., (i - 24) as f32 / 24.),
                49..=72 => ((72 - i) as f32 / 24., 1.),
                _ => (0., (96 - i) as f32 / 24.),
            };
            d[2 * i] = x - 0.5;
            d[2 * i + 1] = (y - 0.5) / 0.75;
        }
        d[192] = 1.4;
        d
    }
    #[test]
    fn tensor_is_nhwc_rgba_normalized() {
        let image = RgbImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                Rgb([255, 0, 128])
            } else {
                Rgb([0, 255, 64])
            }
        });
        let t = model_input(&image, [0, 0, 2, 1], 2, 2).tensor;
        assert_eq!(t.len(), 16);
        assert_eq!(t[..8], t[8..]);
        assert_eq!(t[0], 1.);
        assert_eq!(t[1], 0.);
        assert_eq!(t[2], 128. / 255.);
        assert_eq!(t[3], 1.);
        assert_eq!(t[4], 0.);
        assert_eq!(t[5], 1.);
        assert_eq!(t[7], 1.);
    }
    #[test]
    fn unpadding_restores_original_view_offset() {
        let (padded, offset) = portrait_padding(8, 4);
        assert_eq!(padded, [8, 10]);
        assert_eq!(offset, [0, 3]);
        let p = boundary_source(&rectangle(), [8, 10], offset, [100, 200, 8, 4]).unwrap();
        assert_eq!((p[0].x, p[0].y), (100., 197.));
        assert_eq!((p[24].x, p[24].y), (108., 197.));
        assert!((p[48].y - 207.).abs() < 1e-5);
    }
    #[test]
    fn model_input_matches_padded_image_resize() {
        let image = RgbImage::from_fn(97, 61, |x, y| {
            let v = x
                .wrapping_mul(2_654_435_761)
                .wrapping_add(y.wrapping_mul(40_503))
                >> 7;
            Rgb([v as u8, (v >> 8) as u8, (v >> 16) as u8])
        });
        // Padded rows, padded columns, offset views, enlargement and identity.
        for (view, width, height) in [
            ([0, 0, 97, 61], 12, 16),
            ([0, 0, 30, 61], 12, 16),
            ([43, 5, 54, 50], 12, 16),
            ([10, 20, 7, 9], 12, 16),
            ([3, 2, 12, 16], 12, 16),
            ([0, 0, 97, 61], 144, 192),
        ] {
            let input = model_input(&image, view, width, height);
            assert_eq!(
                (input.padded, input.offset),
                portrait_padding(view[2], view[3])
            );
            assert!(
                input.tensor == reference_input(&image, view, width, height),
                "view {view:?} to {width}x{height}"
            );
        }
    }
    #[test]
    fn aspect_does_not_affect_boundary() {
        let d = rectangle();
        let first = boundary_source(&d, [300, 400], [0, 0], [0, 0, 300, 400]).unwrap();
        let mut changed = d;
        changed[192] = -999.;
        let second = boundary_source(&changed, [300, 400], [0, 0], [0, 0, 300, 400]).unwrap();
        assert_eq!(first.len(), 96);
        for (a, b) in first.iter().zip(second) {
            assert_eq!((a.x, a.y), (b.x, b.y));
        }
        assert!(boundary_source(&vec![0.; 1251], [300, 400], [0, 0], [0, 0, 300, 400]).is_err());
    }
    #[test]
    fn compact_aspect_is_validated() {
        let mut decoded = rectangle();
        for ratio in [0.5, 1.414, 2.0] {
            decoded[192] = ratio;
            assert_eq!(height_width_ratio(&decoded), Some(f64::from(ratio)));
        }
        for ratio in [f32::NAN, f32::INFINITY, -1., 0., 20.] {
            decoded[192] = ratio;
            assert_eq!(height_width_ratio(&decoded), None);
        }
    }
    #[test]
    fn clip_keeps_polygon_in_image_edges() {
        let p = vec![
            Point { x: -5., y: -2. },
            Point { x: 12., y: -2. },
            Point { x: 12., y: 9. },
            Point { x: -5., y: 9. },
        ];
        let c = clip_polygon(&p, 10, 8);
        assert_eq!(c.len(), 4);
        let pages = PageRegions {
            width: 10,
            height: 8,
            regions: vec![PageRegion {
                id: "page-1".into(),
                contour: c,
                confidence: 0.,
                evidence: vec![],
            }],
        };
        pages.validate().unwrap();
        assert!(pages.regions[0].contains(10., 8.));
        assert!(
            clip_polygon(
                &[
                    Point { x: -3., y: 0. },
                    Point { x: -2., y: 2. },
                    Point { x: -4., y: 2. }
                ],
                10,
                8
            )
            .is_empty()
        );
    }
    #[test]
    fn spread_direction_is_independent_of_canvas_aspect() {
        for (w, h) in [(160, 159), (160, 160), (160, 161), (80, 160)] {
            let span = w * 9 / 16;
            assert_eq!(
                source_views(w, h, PageLayoutMode::Double, PageSpreadLayout::default()),
                vec![[0, 0, span, h], [w - span, 0, span, h]]
            );
            let span = h * 9 / 16;
            assert_eq!(
                source_views(w, h, PageLayoutMode::Double, PageSpreadLayout::Stacked),
                vec![[0, 0, w, span], [0, h - span, w, span]]
            );
        }
    }
    #[test]
    fn forced_layout_has_one_or_two_overlapping_views() {
        assert_eq!(
            source_views(
                160,
                80,
                PageLayoutMode::Single,
                PageSpreadLayout::SideBySide
            ),
            vec![[0, 0, 160, 80]]
        );
        assert_eq!(
            source_views(
                160,
                80,
                PageLayoutMode::Double,
                PageSpreadLayout::SideBySide
            ),
            vec![[0, 0, 90, 80], [70, 0, 90, 80]]
        );
        assert_eq!(
            source_views(80, 160, PageLayoutMode::Double, PageSpreadLayout::Stacked),
            vec![[0, 0, 80, 90], [0, 70, 80, 90]]
        );
        assert_eq!(
            selected_layout(PageLayoutMode::Auto, Some(&[0., 0., 0., 1., 0.])),
            PageLayoutMode::Double
        );
        assert_eq!(
            selected_layout(PageLayoutMode::Auto, Some(&[1., 0., 0., 0., 0.])),
            PageLayoutMode::Single
        );
    }
}
