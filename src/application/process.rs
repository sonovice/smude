//! Single-page fit, composed rendering and output diagnostics.
use crate::geometry::{Config, Geometry, Observations, Point};
use image::{
    ImageEncoder,
    codecs::png::{CompressionType, FilterType, PngEncoder},
};
use rayon::prelude::*;
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use super::{
    options::{CropMode, Options},
    profile::StepProfile,
};

#[derive(Serialize)]
pub(super) struct ProcessSummary {
    pub(super) quality: crate::detection::quality::CaptureQuality,
    #[serde(skip)]
    pub(super) mapping: Option<std::sync::Arc<super::PreviewMap>>,
    pub(super) staves: usize,
    pub(super) barlines: usize,
    pub(super) staff_edges: usize,
    pub(super) analysis_dimensions: Option<[u32; 2]>,
    pub(super) detection_seconds: f64,
    pub(super) fit_seconds: Option<f64>,
    pub(super) render_seconds: Option<f64>,
}

#[derive(Clone, Copy)]
pub(super) struct PageOutput<'a> {
    pub(super) page_model: Option<&'a crate::geometry::PageAnalysisMap>,
    pub(super) ratio: Option<f64>,
    pub(super) region: &'a crate::detection::PageRegion,
    pub(super) corners: Option<[Point; 4]>,
    pub(super) analysis_dimensions: Option<[u32; 2]>,
    pub(super) camera_center: Point,
}

/// Clear geometric page-edge artifacts before they can become protected text.
fn clear_text_analysis_rims(
    analysis: &mut image::GrayImage,
    contour: &[Point],
    music: &[[f64; 4]],
) {
    let margin = (f64::from(analysis.width().min(analysis.height())) * 0.004)
        .ceil()
        .max(1.) as u32;
    crate::rendering::output::PageBorder::new(contour).clean_gray(analysis, margin, music);
}

/// Staff, staff-edge and bar measurements in world coordinates. Tracing every
/// sample through the fitted map is costly, so each consumer shares one pass.
struct MusicTrace {
    staves: Vec<[Vec<Point>; 5]>,
    edges: Vec<Vec<Point>>,
    bars: Vec<Vec<Point>>,
}

impl MusicTrace {
    fn new(data: &Observations, geometry: &Geometry) -> Self {
        let trace = |x: &[f64], y: &[f64]| -> Vec<Point> {
            x.iter()
                .zip(y)
                .map(|(x, y)| geometry.forward(Point { x: *x, y: *y }))
                .collect()
        };
        Self {
            staves: data
                .staves
                .par_iter()
                .map(|staff| std::array::from_fn(|k| trace(&staff.x, &staff.y[k])))
                .collect(),
            edges: geometry
                .report()
                .staff_edges
                .iter()
                .map(|edge| edge.points.iter().map(|p| geometry.forward(*p)).collect())
                .collect(),
            bars: data
                .barlines
                .iter()
                .map(|bar| trace(&bar.x, &bar.y))
                .collect(),
        }
    }

    /// The same measurements after the text residual fitted since `new`.
    fn with_text(&self, geometry: &Geometry) -> Self {
        let text = |points: &Vec<Point>| -> Vec<Point> {
            points.iter().map(|p| geometry.forward_text(*p)).collect()
        };
        Self {
            staves: self
                .staves
                .iter()
                .map(|lines| std::array::from_fn(|k| text(&lines[k])))
                .collect(),
            edges: self.edges.iter().map(text).collect(),
            bars: self.bars.iter().map(text).collect(),
        }
    }

    fn regions(
        &self,
        data: &Observations,
        geometry: &Geometry,
        padding: [f64; 2],
        horizontal: f64,
    ) -> Vec<[f64; 4]> {
        let bounds = |points: &mut dyn Iterator<Item = &Point>, px: f64, py: f64| {
            let mut r = [
                f64::INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ];
            for p in points {
                r[0] = r[0].min(p.x);
                r[1] = r[1].min(p.y);
                r[2] = r[2].max(p.x);
                r[3] = r[3].max(p.y);
            }
            [r[0] - px, r[1] - py, r[2] + px, r[3] + py]
        };
        let report = geometry.report();
        let mut regions = vec![];
        for (i, lines) in self.staves.iter().enumerate() {
            let space = report.target_staff_spacings[i];
            let edges = report
                .staff_edges
                .iter()
                .zip(&self.edges)
                .filter(|(e, _)| e.staff == i && e.confidence >= 0.7)
                .flat_map(|(_, points)| points);
            let mut region = bounds(
                &mut lines.iter().flatten().chain(edges),
                horizontal * space,
                0.,
            );
            region[1] -= padding[0] * space;
            region[3] += padding[1] * space;
            regions.push(region);
        }
        for (bar, points) in data.barlines.iter().zip(&self.bars) {
            let space = report.target_staff_spacings[bar.staff];
            regions.push(bounds(&mut points.iter(), horizontal * space, 3. * space));
        }
        regions
    }
}

// Broad guards include dynamics and other nearby musical annotations. Where
// text has actually been measured, release only its correction neighbourhood;
// the tighter music guards, including the tone-filter margin, remain absolute.
fn text_music_guards(
    lines: &[crate::detection::TextLine],
    broad: Vec<[f64; 4]>,
    music: &[[f64; 4]],
) -> Vec<[f64; 4]> {
    let mut guards = broad;
    for line in lines {
        let b = line.bounds;
        if music
            .iter()
            .any(|r| b[0] < r[2] && b[2] > r[0] && b[1] < r[3] && b[3] > r[1])
        {
            continue;
        }
        let height = b[3] - b[1];
        let mean = line.points.iter().map(|p| p.y).sum::<f64>() / line.points.len().max(1) as f64;
        let amplitude = line
            .points
            .iter()
            .map(|p| (p.y - mean).abs())
            .fold(0., f64::max);
        let pad = (height * 0.75).max(12.).max(amplitude * 8.);
        let x_pad = (height * 0.75).max(12.).max(amplitude * 16.);
        let c = [b[0] - x_pad, b[1] - pad, b[2] + x_pad, b[3] + pad];
        guards = guards
            .into_iter()
            .flat_map(|r| {
                let a = [
                    r[0].max(c[0]),
                    r[1].max(c[1]),
                    r[2].min(c[2]),
                    r[3].min(c[3]),
                ];
                if a[0] >= a[2] || a[1] >= a[3] {
                    return vec![r];
                }
                [
                    [r[0], r[1], r[2], a[1]],
                    [r[0], a[3], r[2], r[3]],
                    [r[0], a[1], a[0], a[3]],
                    [a[2], a[1], r[2], a[3]],
                ]
                .into_iter()
                .filter(|r| r[0] < r[2] && r[1] < r[3])
                .collect()
            })
            .collect();
    }
    guards.extend_from_slice(music);
    guards
}

#[cfg(test)]
mod text_guard_tests {
    use super::*;
    use crate::{detection::TextLine, geometry::corrections::text_warp::TextWarp};
    #[derive(serde::Deserialize)]
    struct Header {
        lines: Vec<TextLine>,
        broad: Vec<[f64; 4]>,
        music: Vec<[f64; 4]>,
        bounds: [f64; 4],
    }
    #[test]
    fn measured_header_is_corrected_without_releasing_music_or_unobserved_annotations() {
        let header: Header = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/text/header-guard-overlap.json"
        )))
        .unwrap();
        let guards = text_music_guards(&header.lines, header.broad, &header.music);
        let (warp, _) = TextWarp::fit(&header.lines, &guards, header.bounds);
        let warp = warp.unwrap();
        for line in &header.lines {
            let span = |points: &[Point]| {
                points.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max)
                    - points.iter().map(|p| p.y).fold(f64::INFINITY, f64::min)
            };
            let corrected: Vec<_> = line.points.iter().map(|p| warp.forward(*p)).collect();
            assert!(
                span(&corrected) < span(&line.points) * 0.4,
                "header remains curved"
            );
        }
        for r in header.music {
            for t in [0., 0.5, 1.] {
                let p = Point {
                    x: r[0] + t * (r[2] - r[0]),
                    y: (r[1] + r[3]) * 0.5,
                };
                assert_eq!(warp.forward(p).y, p.y);
            }
        }
        let annotation = Point { x: 500., y: 600. };
        assert_eq!(warp.forward(annotation).y, annotation.y);
    }
}

pub(super) fn process(
    args: &Options,
    out: &Path,
    data: &Observations,
    source: Option<image::RgbImage>,
    detection_seconds: f64,
    page_output: Option<PageOutput<'_>>,
    profile: &mut StepProfile,
) -> Result<ProcessSummary, Box<dyn std::error::Error>> {
    let quality_start = Instant::now();
    let capture_quality = source
        .as_ref()
        .map(|image| crate::detection::quality::assess_capture(image, data))
        .unwrap_or_default();
    profile.capture_quality_seconds = quality_start.elapsed().as_secs_f64();
    let page_id = page_output.map_or("1", |p| p.region.id.as_str());
    let to_original = |p: Point| {
        let [x, y, _, _] = page_output.map_or([0; 4], |p| p.region.bounds());
        Point {
            x: p.x + f64::from(x),
            y: p.y + f64::from(y),
        }
    };
    if args.geometry.is_some() {
        args.emit_staves(page_id, &data.staves, to_original);
    }
    args.emit(page_id, super::DetectionStage::Barlines, || {
        data.barlines
            .iter()
            .map(|b| {
                b.x.iter()
                    .zip(&b.y)
                    .map(|(x, y)| to_original(Point { x: *x, y: *y }))
                    .collect()
            })
            .collect()
    });
    let height_width_ratio = page_output.and_then(|p| p.ratio).or(args.page_aspect);
    let oracle_corners: Option<[Point; 4]> = args
        .geometry_corners
        .as_ref()
        .map(|path| -> Result<_, Box<dyn std::error::Error>> {
            Ok(serde_json::from_slice(&fs::read(path)?)?)
        })
        .transpose()?;
    let oracle_contour: Option<Vec<Point>> = args
        .geometry_contour
        .as_ref()
        .map(|path| -> Result<_, Box<dyn std::error::Error>> {
            let points: Vec<Point> = serde_json::from_slice(&fs::read(path)?)?;
            if points.len() < 4 || points.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
                return Err("invalid geometry contour".into());
            }
            Ok(points)
        })
        .transpose()?;
    let analysis_dimensions =
        if let Some(dimensions) = page_output.and_then(|p| p.analysis_dimensions) {
            Some(dimensions)
        } else if args.geometry.is_some() {
            None
        } else {
            let longest = data.width.max(data.height);
            let limit = args.analysis_width.unwrap_or_else(|| {
                if args.analysis_long_edge.is_none() {
                    return 1800;
                }
                (u64::from(data.width)
                    * u64::from(
                        args.analysis_long_edge
                            .map(|limit| limit.min(longest))
                            .unwrap_or(longest),
                    )
                    / u64::from(longest)) as u32
            });
            let width = data.width.min(limit);
            Some([
                width,
                (f64::from(data.height) * f64::from(width) / f64::from(data.width)).round() as u32,
            ])
        };
    fs::create_dir_all(out)?;
    profile.failure_stage = Some("observations_write");
    let start = Instant::now();
    fs::write(out.join("observations.json"), serde_json::to_vec(data)?)?;
    profile.observations_write_seconds = start.elapsed().as_secs_f64();
    profile.failure_stage = Some("staff_edge_detection");
    let started = Instant::now();
    let staff_edges = if !args.no_staff_edges
        && !data.staves.is_empty()
        && let Some(image) = source.as_ref()
    {
        let local_region = page_output.map(|p| {
            let mut region = p.region.clone();
            let [x, y, _, _] = region.bounds();
            for point in &mut region.contour {
                point.x -= f64::from(x);
                point.y -= f64::from(y);
            }
            region
        });
        crate::detection::detect_staff_edges_with_region_mask(
            image,
            data,
            args.analysis_width_for(image),
            local_region.as_ref(),
        )?
    } else {
        vec![]
    };
    profile.staff_edge_detection_seconds += started.elapsed().as_secs_f64();
    args.emit(page_id, super::DetectionStage::StaffEdges, || {
        staff_edges
            .iter()
            .map(|e| e.points.iter().copied().map(to_original).collect())
            .collect()
    });
    let started = Instant::now();
    fs::write(
        out.join("staff-edges.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"coordinate_space":"input_crop_pixels","source":"independent per-line ink endpoints; geometric evidence, not printed bar strokes","edges":staff_edges}),
        )?,
    )?;
    profile.observations_write_seconds += started.elapsed().as_secs_f64();
    if args.detect_only {
        println!(
            "{} staves, {} barline strokes; {detection_seconds:.3}s",
            data.staves.len(),
            data.barlines.len()
        );
        return Ok(ProcessSummary {
            quality: capture_quality.clone(),
            mapping: None,
            staves: data.staves.len(),
            barlines: data.barlines.len(),
            staff_edges: staff_edges.len(),
            analysis_dimensions,
            detection_seconds,
            fit_seconds: None,
            render_seconds: None,
        });
    }
    let groups = args
        .same_size
        .as_ref()
        .map(|s| {
            s.split(';')
                .map(|g| {
                    g.split(',')
                        .map(|i| i.trim().parse::<usize>())
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();
    let config = Config {
        perspective: args.perspective || !args.no_perspective,
        system_shear: args.system_shear,
        staff_size_groups: groups,
        size_prior: args.size_prior,
    };
    let use_mesh = args.direction_mesh && !args.no_gcs && !args.no_perspective;
    let verticals = if use_mesh
        && !data.staves.is_empty()
        && let Some(image) = source.as_ref()
    {
        let start = Instant::now();
        let stems = crate::detection::stems::detect_stems_scaled(
            image,
            data,
            args.analysis_width_for(image),
        )?;
        let paths = stems.vertical_evidence(data);
        profile.stem_detection_seconds = start.elapsed().as_secs_f64();
        if args.diagnostics {
            fs::write(out.join("stems.json"), serde_json::to_vec(&stems)?)?;
        }
        paths
    } else {
        vec![]
    };
    args.processing(super::ProcessingEvent::Stage(
        super::ProcessingStage::Dewarping,
    ));
    let start = Instant::now();
    let corners = page_output.and_then(|p| p.corners).or(oracle_corners);
    let (geometry, fit_profile) = if let Some(map) = page_output.and_then(|p| p.page_model) {
        use crate::geometry::fit::PageResidualModel;
        let mode = if use_mesh {
            PageResidualModel::Mesh
        } else if !args.no_gcs && !args.no_perspective {
            PageResidualModel::Gcs {
                vertical: !args.no_vertical_correction,
            }
        } else {
            PageResidualModel::Staff {
                vertical: !args.no_vertical_correction,
            }
        };
        Geometry::fit_profiled_on_page(data, &config, map, &staff_edges, &verticals, mode)
    } else if use_mesh {
        Geometry::fit_profiled_direction_mesh(
            data,
            &config,
            if args.no_page_perspective {
                None
            } else {
                corners
            },
            height_width_ratio,
            &staff_edges,
            &verticals,
        )
    } else if !args.no_gcs && !args.no_perspective {
        Geometry::fit_profiled_gcs_with_camera(
            data,
            &config,
            if args.no_page_perspective {
                None
            } else {
                corners
            },
            height_width_ratio,
            &staff_edges,
            !args.no_vertical_correction,
            page_output.map(|p| p.camera_center),
        )
    } else {
        let (result, mut profile) =
            Geometry::fit_profiled_with_staff_edges(data, &config, corners, &staff_edges);
        let result = result.and_then(|mut geometry| {
            if !args.no_vertical_correction {
                let start = Instant::now();
                geometry.add_vertical_correction(data)?;
                profile.vertical_correction_seconds = start.elapsed().as_secs_f64();
            }
            Ok(geometry)
        });
        (result, profile)
    };
    let fit_seconds = start.elapsed().as_secs_f64();
    profile.input_validation_seconds = fit_profile.input_validation_seconds;
    profile.perspective_seconds = fit_profile.perspective_seconds;
    profile.gcs_seconds = fit_profile.gcs_seconds;
    profile.direction_mesh_seconds = fit_profile.direction_mesh_seconds;
    profile.vertical_correction_seconds = fit_profile.vertical_correction_seconds;
    profile.staff_conditioning_seconds = fit_profile.staff_conditioning_seconds;
    profile.staff_warp_seconds = fit_profile.staff_warp_seconds;
    profile.system_shear_seconds = fit_profile.system_shear_seconds;
    profile.map_validation_seconds = fit_profile.map_validation_seconds;
    profile.failure_stage = fit_profile.failure_stage;
    if geometry.is_err() {
        profile.failed_fit_stage_seconds = fit_profile.unattributed_seconds;
    }
    let mut geometry = geometry?;
    let mut refined_data = None;
    if use_mesh
        && data.barlines.len() < 2 * data.staves.len()
        && let Some(source) = source.as_ref()
    {
        let started = Instant::now();
        let mut report = serde_json::json!({"accepted":false,"original_bars":data.barlines.len()});
        match super::bar_refinement::discover(&geometry, data, source) {
            Ok((candidate, preview, detected)) => {
                report["detected_bars"] = detected.into();
                report["merged_bars"] = candidate.barlines.len().into();
                report["analysis_size"] = serde_json::json!([preview.width(), preview.height()]);
                if args.diagnostics {
                    preview.save(out.join("bar-refinement-preview.png"))?;
                }
                if candidate.barlines.len() > data.barlines.len() {
                    let mut fitted = geometry.clone();
                    let refinement =
                        fitted.refine_direction_mesh(&candidate, &staff_edges, &verticals);
                    let fitted = refinement.map(|_| fitted);
                    match fitted {
                        Ok(fitted) => {
                            let before =
                                super::bar_refinement::errors(&geometry, &candidate.barlines);
                            let after = super::bar_refinement::errors(&fitted, &candidate.barlines);
                            let accepted = super::bar_refinement::improves(
                                &before,
                                &after,
                                data.barlines.len(),
                            );
                            report["before_normalized_deviation"] = serde_json::json!(before);
                            report["after_normalized_deviation"] = serde_json::json!(after);
                            report["accepted"] = accepted.into();
                            if accepted {
                                geometry = fitted;
                                refined_data = Some(candidate);
                            }
                        }
                        Err(error) => report["rejection"] = error.to_string().into(),
                    }
                }
            }
            Err(error) => report["rejection"] = error.to_string().into(),
        }
        profile.bar_refinement_seconds = started.elapsed().as_secs_f64();
        fs::write(
            out.join("bar-refinement.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
    }
    let data = refined_data.as_ref().unwrap_or(data);
    // The original image's camera centre survives page cropping. For a score
    // excerpt with no page assignment that information is unknown: its crop
    // centre is not legitimate camera evidence for surface foreshortening.
    if use_mesh && page_output.is_some() {
        let started = Instant::now();
        let principal = page_output.map_or(
            Point {
                x: f64::from(data.width - 1) * 0.5,
                y: f64::from(data.height - 1) * 0.5,
            },
            |p| p.camera_center,
        );
        let [ox, oy, _, _] = page_output.map_or([0; 4], |p| p.region.bounds());
        let focal =
            (2. * (principal.x + f64::from(ox)) + 1.).max(2. * (principal.y + f64::from(oy)) + 1.);
        let metric =
            geometry.refine_surface_metric(data, &staff_edges, &verticals, principal, focal);
        profile.direction_mesh_seconds += started.elapsed().as_secs_f64();
        fs::write(
            out.join("surface-metric.json"),
            serde_json::to_vec_pretty(&metric)?,
        )?;
    }
    if refined_data.is_some() {
        fs::write(
            out.join("observations-refined.json"),
            serde_json::to_vec(data)?,
        )?;
    }
    profile.failure_stage = Some("geometry_export");
    let export_start = Instant::now();
    fs::create_dir_all(out)?;
    // Trace input measurements into exact output RASTER coordinates. These
    // traces allow an independent reader to audit the rendered ink positions.
    #[derive(Serialize)]
    struct MappedStaff {
        lines: [Vec<Point>; 5],
    }
    #[derive(Serialize)]
    struct MappedBar {
        staff: usize,
        points: Vec<Point>,
    }
    #[derive(Serialize)]
    struct MappedEdge {
        staff: usize,
        side: crate::geometry::StaffEdgeSide,
        points: Vec<Point>,
        line_indices: Vec<usize>,
        confidence: f64,
        aligned_staffs: Vec<usize>,
    }
    #[derive(Serialize)]
    struct Mapped {
        coordinate_space: &'static str,
        page_contour: Option<Vec<Point>>,
        staves: Vec<MappedStaff>,
        barlines: Vec<MappedBar>,
        text_curves: Vec<Vec<Point>>,
        text_curves_before: Vec<Vec<Point>>,
        protected_music: Vec<[f64; 4]>,
        staff_edges: Vec<MappedEdge>,
    }
    let [x0, y0, x1, y1] = geometry.report().bounds;
    let source_page_contour = page_output
        .map(|p| {
            let region = p.region;
            let [ox, oy, _, _] = region.bounds();
            // Subdivide the sparse contour before the nonlinear y-warp.
            let mut contour = vec![];
            for (a, b) in region
                .contour
                .iter()
                .zip(region.contour.iter().cycle().skip(1))
                .take(region.contour.len())
            {
                let n = ((a.x - b.x).hypot(a.y - b.y) / 8.).ceil().max(1.) as usize;
                for j in 0..n {
                    let t = j as f64 / n as f64;
                    contour.push(Point {
                        x: a.x + (b.x - a.x) * t - f64::from(ox),
                        y: a.y + (b.y - a.y) * t - f64::from(oy),
                    });
                }
            }
            contour
        })
        .or(oracle_contour)
        .or_else(|| {
            oracle_corners.map(|corners| {
                let mut points = vec![];
                for i in 0..4 {
                    for j in 0..=32 {
                        let t = j as f64 / 32.;
                        let (a, b) = (corners[i], corners[(i + 1) % 4]);
                        points.push(Point {
                            x: a.x + (b.x - a.x) * t,
                            y: a.y + (b.y - a.y) * t,
                        });
                    }
                }
                points
            })
        });
    let canvas_height_width_ratio = if geometry.report().page_model.is_some()
        || geometry.report().perspective.source == "page_boundary"
        || geometry.report().gcs.is_some()
        || geometry.report().direction_mesh.is_some()
    {
        match (height_width_ratio, source_page_contour.as_deref()) {
            (Some(ratio), Some(contour)) => Some(geometry.canvas_ratio_for_page(ratio, contour)?),
            _ => height_width_ratio,
        }
    } else {
        height_width_ratio
    };
    let (raster_width, raster_height) = geometry.raster_size(canvas_height_width_ratio)?;
    let music = MusicTrace::new(data, &geometry);
    let protected_music = music.regions(data, &geometry, [8., 8.], 3.);
    let detection_music = music.regions(data, &geometry, [2., 3.], 3.);
    let mut text_lines = vec![];
    let mut text_sources = vec![];
    if !args.no_text_dewarp
        && let Some(source) = source.as_ref()
    {
        let started = Instant::now();
        // Small titles and footers need more than the staff analysis raster.
        // This remains bounded independently of the full-resolution output.
        let scale = (1800. / f64::from(raster_width.max(raster_height))).min(1.);
        let (aw, ah) = (
            (f64::from(raster_width) * scale).round().max(2.) as u32,
            (f64::from(raster_height) * scale).round().max(2.) as u32,
        );
        let analysis = geometry.render_at_dimensions(source, aw, ah)?;
        let mut analysis = crate::rendering::output::normalize_text_analysis(
            &crate::rendering::output::grayscale(&analysis),
        );
        profile.text_analysis_render_seconds = started.elapsed().as_secs_f64();
        let to_analysis = |p: Point| Point {
            x: (p.x - x0) * f64::from(aw - 1) / (x1 - x0),
            y: (p.y - y0) * f64::from(ah - 1) / (y1 - y0),
        };
        let from_analysis = |p: Point| Point {
            x: x0 + p.x * (x1 - x0) / f64::from(aw - 1),
            y: y0 + p.y * (y1 - y0) / f64::from(ah - 1),
        };
        let guards: Vec<_> = detection_music
            .iter()
            .map(|r| {
                let a = to_analysis(Point { x: r[0], y: r[1] });
                let b = to_analysis(Point { x: r[2], y: r[3] });
                [a.x, a.y, b.x, b.y]
            })
            .collect();
        let started = Instant::now();
        if let Some(contour) = &source_page_contour {
            let contour: Vec<_> = contour
                .iter()
                .map(|p| to_analysis(geometry.forward(*p)))
                .collect();
            clear_text_analysis_rims(&mut analysis, &contour, &guards);
        }
        text_lines = crate::detection::detect_text_lines(&analysis, &guards);
        profile.text_detection_seconds = started.elapsed().as_secs_f64();
        for line in &mut text_lines {
            line.points.iter_mut().for_each(|p| *p = from_analysis(*p));
            let a = from_analysis(Point {
                x: line.bounds[0],
                y: line.bounds[1],
            });
            let b = from_analysis(Point {
                x: line.bounds[2],
                y: line.bounds[3],
            });
            line.bounds = [a.x, a.y, b.x, b.y];
            line.uncertainty_pixels *= (y1 - y0) / f64::from(ah - 1);
            text_sources.push(
                line.points
                    .iter()
                    .map(|p| geometry.inverse(*p))
                    .collect::<Vec<_>>(),
            );
        }
        args.emit(page_id, super::DetectionStage::Text, || {
            text_sources
                .iter()
                .map(|line| line.iter().copied().map(to_original).collect())
                .collect()
        });
        let started = Instant::now();
        // Keep the tone window and optical-AA neighbourhood around music too.
        let px = 31. * (x1 - x0) / f64::from(raster_width - 1);
        let py = 31. * (y1 - y0) / f64::from(raster_height - 1);
        let expand = |r: &[f64; 4]| [r[0] - px, r[1] - py, r[2] + px, r[3] + py];
        let tone_guards = text_music_guards(
            &text_lines,
            protected_music.iter().map(expand).collect(),
            &detection_music.iter().map(expand).collect::<Vec<_>>(),
        );
        geometry.add_text_curves(&text_lines, &tone_guards);
        profile.text_fit_seconds = started.elapsed().as_secs_f64();
        fs::write(
            out.join("text-lines.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"coordinate_space":"base_geometry_world_pixels","lines":text_lines,"detection_music":detection_music,"protected_music":protected_music,"text_music_guards":tone_guards,"warp":geometry.report().text_warp}),
            )?,
        )?;
    }
    let music = music.with_text(&geometry);
    let world_to_pixel = |q: Point| Point {
        x: (q.x - x0) * f64::from(raster_width - 1) / (x1 - x0),
        y: (q.y - y0) * f64::from(raster_height - 1) / (y1 - y0),
    };
    let to_pixel = |p: Point| world_to_pixel(geometry.forward(p));
    let pixels = |points: &Vec<Point>| -> Vec<Point> {
        points.iter().copied().map(world_to_pixel).collect()
    };
    let mapped_page_contour = source_page_contour
        .as_ref()
        .map(|points| points.iter().map(|p| to_pixel(*p)).collect::<Vec<_>>());
    let mut mapped = Mapped {
        coordinate_space: "output_raster_pixels",
        staff_edges: staff_edges
            .iter()
            .map(|edge| MappedEdge {
                staff: edge.staff,
                side: edge.side,
                points: edge.points.iter().map(|p| to_pixel(*p)).collect(),
                line_indices: edge.line_indices.clone(),
                confidence: edge.confidence,
                aligned_staffs: edge.aligned_staffs.clone(),
            })
            .collect(),
        text_curves: text_sources
            .iter()
            .map(|l| l.iter().map(|p| to_pixel(*p)).collect())
            .collect(),
        text_curves_before: text_lines
            .iter()
            .map(|l| {
                l.points
                    .iter()
                    .map(|p| Point {
                        x: (p.x - x0) * f64::from(raster_width - 1) / (x1 - x0),
                        y: (p.y - y0) * f64::from(raster_height - 1) / (y1 - y0),
                    })
                    .collect()
            })
            .collect(),
        protected_music: protected_music
            .iter()
            .map(|r| {
                [
                    (r[0] - x0) * f64::from(raster_width - 1) / (x1 - x0),
                    (r[1] - y0) * f64::from(raster_height - 1) / (y1 - y0),
                    (r[2] - x0) * f64::from(raster_width - 1) / (x1 - x0),
                    (r[3] - y0) * f64::from(raster_height - 1) / (y1 - y0),
                ]
            })
            .collect(),
        page_contour: mapped_page_contour.clone(),
        staves: music
            .staves
            .iter()
            .map(|lines| MappedStaff {
                lines: std::array::from_fn(|k| pixels(&lines[k])),
            })
            .collect(),
        barlines: data
            .barlines
            .iter()
            .zip(&music.bars)
            .map(|(bar, points)| MappedBar {
                staff: bar.staff,
                points: pixels(points),
            })
            .collect(),
    };
    let mut mapped_probes = if let Some(path) = &args.probe_points {
        let input: Vec<Point> = serde_json::from_slice(&fs::read(path)?)?;
        if input.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
            return Err("probe points must be finite".into());
        }
        Some(input.into_iter().map(to_pixel).collect::<Vec<_>>())
    } else {
        None
    };
    profile.geometry_export_seconds = (export_start.elapsed().as_secs_f64()
        - profile.text_analysis_render_seconds
        - profile.text_detection_seconds
        - profile.text_fit_seconds)
        .max(0.);
    let mut final_crop = [0, 0, raster_width, raster_height];
    let mut final_frame = None;
    let (render_seconds, output_processing) = if args.no_render {
        (None, None)
    } else {
        let source = match source {
            Some(image) => image,
            None => {
                profile.failure_stage = Some("decode");
                let start = Instant::now();
                let decoded = super::decode::open_rgb(&args.image);
                profile.decode_seconds += start.elapsed().as_secs_f64();
                decoded?
            }
        };
        profile.failure_stage = Some("render");
        args.processing(super::ProcessingEvent::Stage(
            super::ProcessingStage::Rendering,
        ));
        let start = Instant::now();
        let color_page = crate::rendering::output::preserve_color(&source, !data.staves.is_empty());
        let rendered = geometry.render_prepared(
            &source,
            raster_width,
            raster_height,
            args.render_backend,
            true,
            args.diagnostics || color_page,
        );
        let seconds = start.elapsed().as_secs_f64();
        profile.render_seconds = seconds;
        let rendered = rendered?;
        profile.renderer = Some(rendered.profile);
        args.processing(super::ProcessingEvent::Stage(
            super::ProcessingStage::Enhancement,
        ));
        let mut result = rendered.color;
        let output_start = Instant::now();
        let (mut gray, mut binary) = if let Some(tones) = rendered.tones {
            tones
        } else {
            profile.failure_stage = Some("grayscale");
            let gray_start = Instant::now();
            let gray = crate::rendering::output::grayscale(result.as_ref().unwrap());
            profile.grayscale_seconds = gray_start.elapsed().as_secs_f64();
            profile.failure_stage = Some("binarization");
            let binary_start = Instant::now();
            let binary = crate::rendering::output::binarize_sauvola(&gray);
            profile.binarization_seconds = binary_start.elapsed().as_secs_f64();
            (gray, binary)
        };
        use crate::rendering::output::{PageBorder, cropped};
        let border_contour = mapped_page_contour
            .as_ref()
            .filter(|_| args.binary_page_margin > 0.);
        let border_margin_pixels = border_contour.map_or(0, |_| {
            (f64::from(binary.width().min(binary.height())) * args.binary_page_margin)
                .round()
                .max(1.) as u32
        });
        let whiten = |binary: &mut image::GrayImage| {
            border_contour.map_or(0, |contour| {
                PageBorder::new(contour).whiten_binary(binary, border_margin_pixels)
            })
        };
        // Only a content crop reads the binary image. A page crop follows the
        // contour, and the binary rim is then removed beside the print path.
        let mut border_removed_pixels = None;
        if !args.no_final_crop && matches!(args.crop_mode, CropMode::Content) {
            let started = Instant::now();
            border_removed_pixels = Some(whiten(&mut binary));
            profile.binary_border_seconds = started.elapsed().as_secs_f64();
        }
        profile.failure_stage = Some("final_crop");
        let crop_started = Instant::now();
        if !args.no_final_crop {
            let padding =
                (f64::from(binary.width().min(binary.height())) * args.crop_padding).round() as u32;
            final_crop = match args.crop_mode {
                CropMode::Content => crate::rendering::output::content_crop(&binary, padding),
                CropMode::Page => crate::rendering::output::page_crop(
                    mapped_page_contour.as_deref().unwrap_or(&[]),
                    raster_width,
                    raster_height,
                    padding,
                ),
            };
        }
        profile.final_crop_seconds = crop_started.elapsed().as_secs_f64();
        let crop = (final_crop != [0, 0, raster_width, raster_height]).then_some(final_crop);
        let every_image = args.diagnostics || color_page;
        let joined = Instant::now();
        let (border_removed_pixels, (mut clean, print_elapsed)) = rayon::join(
            || {
                let removed = border_removed_pixels.unwrap_or_else(|| whiten(&mut binary));
                if let Some(crop) = crop.filter(|_| every_image) {
                    binary = cropped(&binary, crop);
                }
                removed
            },
            || {
                let print_started = Instant::now();
                if let Some(crop) = crop {
                    if every_image {
                        result = Some(cropped(result.as_ref().unwrap(), crop));
                    }
                    gray = cropped(&gray, crop);
                }
                profile.final_crop_seconds += print_started.elapsed().as_secs_f64();
                // Tone estimates must see the final page, not discarded framing pixels.
                // Diagnostic gray stays the unmodified (cropped) sampler output.
                profile.failure_stage = Some("print_normalization");
                let print_start = Instant::now();
                let mut clean = if color_page {
                    image::DynamicImage::ImageRgb8(crate::rendering::output::enhance_color(
                        result.as_ref().unwrap(),
                    ))
                } else {
                    image::DynamicImage::ImageLuma8(crate::rendering::output::enhance_print(&gray))
                };
                profile.print_normalization_seconds = print_start.elapsed().as_secs_f64();
                if let Some(contour) = border_contour {
                    let started = Instant::now();
                    let cropped_contour: Vec<_> = contour
                        .iter()
                        .map(|p| Point {
                            x: p.x - f64::from(final_crop[0]),
                            y: p.y - f64::from(final_crop[1]),
                        })
                        .collect();
                    // Text-warp guards extend eight spaces beyond a staff. They must
                    // not shelter a page rim above the first or below the last staff.
                    // Three spaces retain stems/ledger notes; compact detached print
                    // remains protected by the border classifier's glyph safeguards.
                    let mut protected: Vec<_> = music
                        .regions(data, &geometry, [3., 3.], 0.75)
                        .iter()
                        .map(|r| {
                            [
                                (r[0] - x0) * f64::from(raster_width - 1) / (x1 - x0),
                                (r[1] - y0) * f64::from(raster_height - 1) / (y1 - y0),
                                (r[2] - x0) * f64::from(raster_width - 1) / (x1 - x0),
                                (r[3] - y0) * f64::from(raster_height - 1) / (y1 - y0),
                            ]
                        })
                        .collect();
                    for (curve, text) in mapped.text_curves.iter().zip(&text_lines) {
                        let padding = (text.bounds[3] - text.bounds[1])
                            * f64::from(raster_height - 1)
                            / (y1 - y0)
                            + 8.;
                        if let Some(first) = curve.first() {
                            let mut bounds = [first.x, first.y, first.x, first.y];
                            for p in curve {
                                bounds[0] = bounds[0].min(p.x);
                                bounds[1] = bounds[1].min(p.y);
                                bounds[2] = bounds[2].max(p.x);
                                bounds[3] = bounds[3].max(p.y);
                            }
                            protected.push([
                                bounds[0] - padding,
                                bounds[1] - padding,
                                bounds[2] + padding,
                                bounds[3] + padding,
                            ]);
                        }
                    }
                    for bounds in &mut protected {
                        bounds[0] -= f64::from(final_crop[0]);
                        bounds[2] -= f64::from(final_crop[0]);
                        bounds[1] -= f64::from(final_crop[1]);
                        bounds[3] -= f64::from(final_crop[1]);
                    }
                    let border = PageBorder::new(&cropped_contour);
                    if let Some(gray) = clean.as_mut_luma8() {
                        border.clean_gray(gray, border_margin_pixels, &protected);
                    } else if let Some(rgb) = clean.as_mut_rgb8() {
                        border.clean_color(rgb, border_margin_pixels);
                    }
                    profile.print_border_seconds = started.elapsed().as_secs_f64();
                }
                (clean, print_started.elapsed())
            },
        );
        // Stage times add up to the elapsed time, so the binary rim is charged
        // with what it adds to the print path.
        profile.binary_border_seconds +=
            joined.elapsed().saturating_sub(print_elapsed).as_secs_f64();
        profile.failure_stage = Some("final_framing");
        let framing_started = Instant::now();
        if !args.no_final_crop
            && !args.no_a4
            && matches!(args.crop_mode, CropMode::Page)
            && let Some(paper) = clean.as_luma8()
        {
            let mut protected = Vec::new();
            let mut protect = |points: Vec<Point>, padding: f64| {
                if points.is_empty() {
                    return;
                }
                let mut r = [
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ];
                for p in points {
                    r[0] = r[0].min(p.x);
                    r[1] = r[1].min(p.y);
                    r[2] = r[2].max(p.x);
                    r[3] = r[3].max(p.y);
                }
                protected.push([
                    r[0] - padding - f64::from(final_crop[0]),
                    r[1] - padding - f64::from(final_crop[1]),
                    r[2] + padding - f64::from(final_crop[0]),
                    r[3] + padding - f64::from(final_crop[1]),
                ]);
            };
            for staff in &mapped.staves {
                protect(staff.lines.iter().flatten().copied().collect(), 2.);
            }
            for bar in &mapped.barlines {
                protect(bar.points.clone(), 2.);
            }
            for (curve, text) in mapped.text_curves.iter().zip(&text_lines) {
                let height =
                    (text.bounds[3] - text.bounds[1]) * f64::from(raster_height - 1) / (y1 - y0);
                protect(curve.clone(), height.max(2.));
            }
            if let Some(frame) =
                crate::rendering::output::A4Frame::plan(paper, height_width_ratio, &protected)
            {
                clean = image::DynamicImage::ImageLuma8(frame.apply_gray(paper));
                if args.diagnostics {
                    result = result.as_ref().map(|rgb| frame.apply_rgb(rgb));
                    gray = frame.apply_gray(&gray);
                    binary = frame.apply_gray(&binary);
                }
                final_frame = Some(frame);
            }
        }
        profile.final_framing_seconds = framing_started.elapsed().as_secs_f64();
        profile.failure_stage = Some("png_write");
        args.processing(super::ProcessingEvent::Stage(
            super::ProcessingStage::Saving,
        ));
        let png_start = Instant::now();
        if args.diagnostics {
            let result = result.as_ref().unwrap();
            let file = fs::File::create(out.join("rectified.png"))?;
            PngEncoder::new_with_quality(file, CompressionType::Default, FilterType::Adaptive)
                .write_image(
                    result.as_raw(),
                    result.width(),
                    result.height(),
                    image::ExtendedColorType::Rgb8,
                )?;
            gray.save(out.join("rectified-gray.png"))?;
            binary.save(out.join("rectified-binary.png"))?;
        }
        clean.save(out.join("rectified-clean.png"))?;
        profile.png_write_seconds += png_start.elapsed().as_secs_f64();
        (
            Some(seconds),
            Some(OutputProcessing {
                binary_border_method: "contour-relative rim components before final interior mask",
                binary_border_margin_pixels: border_margin_pixels,
                binary_border_removed_pixels: border_removed_pixels,
                color_preserved: color_page,
                clean_color: "post-crop RGB paper white and restrained neutral black; preserve covers without reliable paper evidence",
                clean_grayscale: "post-crop illumination division; adaptive paper white and print black; contour-relative rim reconstruction with music/text protection; no Sauvola mask or post-blur",
                grayscale: "imagecrate luminance from rectified RGB",
                binarization: "Sauvola",
                window_size: 51,
                k: 0.25,
                dynamic_range: 128.,
                border: "replicate",
                seconds: output_start.elapsed().as_secs_f64(),
            }),
        )
    };
    profile.failure_stage = Some("geometry_export");
    let export_started = Instant::now();
    let translation = final_frame
        .as_ref()
        .map_or(Point { x: 0., y: 0. }, |frame| frame.translation());
    let shift = |p: &mut Point| {
        p.x += translation.x - f64::from(final_crop[0]);
        p.y += translation.y - f64::from(final_crop[1]);
    };
    if let Some(contour) = &mut mapped.page_contour {
        for point in contour {
            shift(point);
        }
    }
    for line in mapped
        .text_curves
        .iter_mut()
        .chain(&mut mapped.text_curves_before)
    {
        for point in line {
            shift(point);
        }
    }
    for edge in &mut mapped.staff_edges {
        for point in &mut edge.points {
            shift(point);
        }
    }
    for r in &mut mapped.protected_music {
        r[0] += translation.x - f64::from(final_crop[0]);
        r[2] += translation.x - f64::from(final_crop[0]);
        r[1] += translation.y - f64::from(final_crop[1]);
        r[3] += translation.y - f64::from(final_crop[1]);
    }
    for staff in &mut mapped.staves {
        for line in &mut staff.lines {
            for point in line {
                shift(point);
            }
        }
    }
    for bar in &mut mapped.barlines {
        for point in &mut bar.points {
            shift(point);
        }
    }
    fs::write(
        out.join("mapped_geometry.json"),
        serde_json::to_vec(&mapped)?,
    )?;
    if let Some(points) = &mut mapped_probes {
        for point in points.iter_mut() {
            shift(point);
        }
        fs::write(out.join("mapped_probes.json"), serde_json::to_vec(points)?)?;
    }
    profile.geometry_export_seconds += export_started.elapsed().as_secs_f64();
    #[derive(Serialize)]
    struct Run<'a> {
        capture_quality: &'a crate::detection::quality::CaptureQuality,
        geometry: &'a crate::geometry::Report,
        output_height_width_ratio: Option<f64>,
        render_canvas_height_width_ratio: Option<f64>,
        output_size: [u32; 2],
        uncropped_output_size: [u32; 2],
        final_crop: [u32; 4],
        final_crop_mode: &'static str,
        final_frame: Option<&'a crate::rendering::output::A4Frame>,
        fit_seconds: f64,
        render_seconds: Option<f64>,
        input_geometry: Option<PathBuf>,
        detection_seconds: f64,
        input_image: PathBuf,
        output_processing: Option<OutputProcessing>,
    }
    let report = Run {
        capture_quality: &capture_quality,
        geometry: geometry.report(),
        output_height_width_ratio: height_width_ratio,
        render_canvas_height_width_ratio: canvas_height_width_ratio,
        output_size: final_frame
            .as_ref()
            .map_or([final_crop[2], final_crop[3]], |frame| frame.output_size),
        final_frame: final_frame.as_ref(),
        uncropped_output_size: [raster_width, raster_height],
        final_crop,
        final_crop_mode: if args.no_final_crop {
            "none"
        } else {
            match args.crop_mode {
                CropMode::Page => "page",
                CropMode::Content => "content",
            }
        },
        fit_seconds,
        render_seconds,
        input_geometry: args.geometry.clone(),
        detection_seconds,
        input_image: args.image.clone(),
        output_processing,
    };
    profile.failure_stage = Some("report_write");
    let start = Instant::now();
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    profile.report_write_seconds = start.elapsed().as_secs_f64();
    println!(
        "fit {fit_seconds:0.3}s; render {render_seconds:?}s; {}",
        out.display()
    );
    let [ox, oy, _, _] = page_output.map_or([0; 4], |p| p.region.bounds());
    let mapping = render_seconds.map(|_| {
        std::sync::Arc::new(
            super::PreviewMap::new(
                geometry,
                Point {
                    x: f64::from(ox),
                    y: f64::from(oy),
                },
                [data.width, data.height],
                page_output.map(|p| p.region.clone()),
                [raster_width, raster_height],
                final_crop,
            )
            .with_output_translation(translation),
        )
    });
    Ok(ProcessSummary {
        quality: capture_quality,
        mapping,
        staves: data.staves.len(),
        barlines: data.barlines.len(),
        staff_edges: staff_edges.len(),
        analysis_dimensions,
        detection_seconds,
        fit_seconds: Some(fit_seconds),
        render_seconds,
    })
}

#[derive(Serialize)]
struct OutputProcessing {
    color_preserved: bool,
    clean_color: &'static str,
    binary_border_method: &'static str,
    binary_border_margin_pixels: u32,
    binary_border_removed_pixels: usize,
    grayscale: &'static str,
    clean_grayscale: &'static str,
    binarization: &'static str,
    window_size: usize,
    k: f64,
    dynamic_range: f64,
    border: &'static str,
    seconds: f64,
}

#[cfg(test)]
mod rim_analysis_regression {
    use super::*;
    use image::{GrayImage, Luma};
    #[test]
    fn perimeter_is_removed_before_it_can_be_promoted_to_protected_text() {
        let mut analysis = GrayImage::from_pixel(400, 600, Luma([255]));
        let contour = vec![
            Point { x: 10., y: 10. },
            Point { x: 390., y: 10. },
            Point { x: 390., y: 590. },
            Point { x: 10., y: 590. },
        ];
        for x in 20..350 {
            analysis.put_pixel(x, 12, Luma([140]));
            analysis.put_pixel(x, 13, Luma([245]));
        }
        for x in [150, 162, 174] {
            for y in 34..44 {
                for xx in x..x + 6 {
                    if xx == x || xx == x + 5 || y == 34 || y == 43 {
                        analysis.put_pixel(xx, y, Luma([140]));
                    }
                }
            }
        }
        for x in 15..380 {
            analysis.put_pixel(x, 100, Luma([170]));
        }
        clear_text_analysis_rims(&mut analysis, &contour, &[[12., 95., 382., 105.]]);
        assert_eq!(
            analysis.get_pixel(100, 12)[0],
            255,
            "A page rim must not seed text observations"
        );
        assert_eq!(
            analysis.get_pixel(100, 13)[0],
            255,
            "Remove the edge's AA fringe too"
        );
        assert_eq!(
            analysis.get_pixel(150, 38)[0],
            140,
            "Keep a genuine compact character"
        );
        assert_eq!(
            analysis.get_pixel(18, 100)[0],
            170,
            "Music guards preserve near-edge staff ink"
        );
    }
}
