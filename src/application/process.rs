//! Single-page fit, composed rendering and output diagnostics.
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use crate::{Config, Geometry, Observations, Point};
use image::{
    ImageEncoder,
    codecs::png::{CompressionType, FilterType, PngEncoder},
};
use serde::Serialize;

use super::{
    options::{CropMode, Options},
    profile::StepProfile,
};

#[derive(Serialize)]
pub(super) struct ProcessSummary {
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
    pub(super) ratio: Option<f64>,
    pub(super) region: &'a crate::PageRegion,
    pub(super) corners: Option<[Point; 4]>,
    pub(super) analysis_dimensions: Option<[u32; 2]>,
    pub(super) camera_center: Point,
}

fn music_regions(data: &Observations, geometry: &Geometry, padding: [f64; 2]) -> Vec<[f64; 4]> {
    let bounds = |points: Vec<Point>, px: f64, py: f64| {
        [
            points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min) - px,
            points.iter().map(|p| p.y).fold(f64::INFINITY, f64::min) - py,
            points.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max) + px,
            points.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max) + py,
        ]
    };
    let mut regions = vec![];
    for (i, staff) in data.staves.iter().enumerate() {
        let space = geometry.report().target_staff_spacings[i];
        let mut points: Vec<Point> = staff
            .y
            .iter()
            .flat_map(|line| {
                staff
                    .x
                    .iter()
                    .zip(line)
                    .map(|(x, y)| geometry.forward(Point { x: *x, y: *y }))
            })
            .collect();
        points.extend(
            geometry
                .report()
                .staff_edges
                .iter()
                .filter(|e| e.staff == i && e.confidence >= 0.7)
                .flat_map(|e| e.points.iter().map(|p| geometry.forward(*p))),
        );
        let mut region = bounds(points, 3. * space, 0.);
        region[1] -= padding[0] * space;
        region[3] += padding[1] * space;
        regions.push(region);
    }
    for bar in &data.barlines {
        let space = geometry.report().target_staff_spacings[bar.staff];
        let points = bar
            .x
            .iter()
            .zip(&bar.y)
            .map(|(x, y)| geometry.forward(Point { x: *x, y: *y }))
            .collect();
        regions.push(bounds(points, 3. * space, 3. * space));
    }
    regions
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
        crate::detect_staff_edges_with_region_mask(
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
    let start = Instant::now();
    let corners = page_output.and_then(|p| p.corners).or(oracle_corners);
    let (geometry, fit_profile) = if !args.no_gcs && !args.no_perspective {
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
        side: crate::StaffEdgeSide,
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
    let canvas_height_width_ratio = if geometry.report().perspective.source == "page_boundary"
        || geometry.report().gcs.is_some()
    {
        match (height_width_ratio, source_page_contour.as_deref()) {
            (Some(ratio), Some(contour)) => Some(geometry.canvas_ratio_for_page(ratio, contour)?),
            _ => height_width_ratio,
        }
    } else {
        height_width_ratio
    };
    let (raster_width, raster_height) = geometry.raster_size(canvas_height_width_ratio)?;
    let protected_music = music_regions(data, &geometry, [8., 8.]);
    let detection_music = music_regions(data, &geometry, [2., 3.]);
    let mut text_lines = vec![];
    let mut text_sources = vec![];
    if !args.no_text_dewarp
        && let Some(source) = source.as_ref()
    {
        let started = Instant::now();
        let scale = (1000. / f64::from(raster_width.max(raster_height))).min(1.);
        let (aw, ah) = (
            (f64::from(raster_width) * scale).round().max(2.) as u32,
            (f64::from(raster_height) * scale).round().max(2.) as u32,
        );
        let analysis = geometry.render_at_dimensions(source, aw, ah)?;
        let analysis = crate::normalize_paper(&image::imageops::grayscale(&analysis));
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
        text_lines = crate::detect_text_lines(&analysis, &guards);
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
        // Keep the 51px tone window and optical-AA neighbourhood unchanged too.
        let px = 31. * (x1 - x0) / f64::from(raster_width - 1);
        let py = 31. * (y1 - y0) / f64::from(raster_height - 1);
        let tone_guards: Vec<_> = protected_music
            .iter()
            .map(|r| [r[0] - px, r[1] - py, r[2] + px, r[3] + py])
            .collect();
        geometry.add_text_curves(&text_lines, &tone_guards);
        profile.text_fit_seconds = started.elapsed().as_secs_f64();
        fs::write(
            out.join("text-lines.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"coordinate_space":"base_geometry_world_pixels","lines":text_lines,"detection_music":detection_music,"protected_music":protected_music,"warp":geometry.report().text_warp}),
            )?,
        )?;
    }
    let to_pixel = |p: Point| {
        let q = geometry.forward(p);
        Point {
            x: (q.x - x0) * f64::from(raster_width - 1) / (x1 - x0),
            y: (q.y - y0) * f64::from(raster_height - 1) / (y1 - y0),
        }
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
        staves: data
            .staves
            .iter()
            .map(|s| MappedStaff {
                lines: std::array::from_fn(|k| {
                    s.x.iter()
                        .zip(&s.y[k])
                        .map(|(x, y)| to_pixel(Point { x: *x, y: *y }))
                        .collect()
                }),
            })
            .collect(),
        barlines: data
            .barlines
            .iter()
            .map(|b| MappedBar {
                staff: b.staff,
                points: b
                    .x
                    .iter()
                    .zip(&b.y)
                    .map(|(x, y)| to_pixel(Point { x: *x, y: *y }))
                    .collect(),
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
    let (render_seconds, output_processing) = if args.no_render {
        (None, None)
    } else {
        let source = match source {
            Some(image) => image,
            None => {
                profile.failure_stage = Some("decode");
                let start = Instant::now();
                let decoded = image::open(&args.image).map(|image| image.to_rgb8());
                profile.decode_seconds += start.elapsed().as_secs_f64();
                decoded?
            }
        };
        profile.failure_stage = Some("render");
        let start = Instant::now();
        let rendered = geometry.render_with_aspect(&source, canvas_height_width_ratio);
        let seconds = start.elapsed().as_secs_f64();
        profile.render_seconds = seconds;
        let mut result = rendered?;
        let output_start = Instant::now();
        profile.failure_stage = Some("grayscale");
        let gray_start = Instant::now();
        let mut gray = image::imageops::grayscale(&result);
        profile.grayscale_seconds = gray_start.elapsed().as_secs_f64();
        profile.failure_stage = Some("binarization");
        let binary_start = Instant::now();
        let mut binary = crate::binarize_sauvola(&gray);
        let raw_binary = binary.clone();
        profile.binarization_seconds = binary_start.elapsed().as_secs_f64();
        profile.failure_stage = Some("print_normalization");
        let print_start = Instant::now();
        let mut clean = crate::normalize_paper(&gray);
        let normalized_before_cleanup = clean.clone();
        profile.print_normalization_seconds = print_start.elapsed().as_secs_f64();
        let mut border_removed_pixels = 0;
        let mut border_margin_pixels = 0;
        if let Some(contour) = mapped_page_contour
            .as_ref()
            .filter(|_| args.binary_page_margin > 0.)
        {
            let border_started = Instant::now();
            border_margin_pixels = (f64::from(binary.width().min(binary.height()))
                * args.binary_page_margin)
                .round()
                .max(1.) as u32;
            border_removed_pixels =
                crate::whiten_page_border(&mut binary, contour, border_margin_pixels);
            profile.binary_border_seconds = border_started.elapsed().as_secs_f64();
            let print_border_start = Instant::now();
            crate::apply_binary_cleanup_with_contour(&mut clean, &raw_binary, &binary, contour);
            crate::whiten_gray_border(
                &mut clean,
                contour,
                border_margin_pixels,
                &mapped.protected_music,
            );
            // Share confirmed cleanup with the non-normalized grayscale too;
            // tone differences are retained everywhere else.
            for ((pixel, before), after) in gray
                .as_mut()
                .iter_mut()
                .zip(normalized_before_cleanup.as_raw())
                .zip(clean.as_raw())
            {
                if *after == 255 && *before < 255 {
                    *pixel = 255;
                }
            }
            profile.print_border_seconds = print_border_start.elapsed().as_secs_f64();
        }
        profile.failure_stage = Some("print_antialias");
        let antialias_start = Instant::now();
        clean = crate::antialias_print(&clean);
        profile.print_antialias_seconds = antialias_start.elapsed().as_secs_f64();
        profile.failure_stage = Some("final_crop");
        let crop_started = Instant::now();
        if !args.no_final_crop {
            let padding =
                (f64::from(binary.width().min(binary.height())) * args.crop_padding).round() as u32;
            final_crop = match args.crop_mode {
                CropMode::Content => crate::content_crop(&binary, padding),
                CropMode::Page => crate::page_crop(
                    mapped_page_contour.as_deref().unwrap_or(&[]),
                    raster_width,
                    raster_height,
                    padding,
                ),
            };
            let [x, y, w, h] = final_crop;
            if final_crop != [0, 0, raster_width, raster_height] {
                result = image::imageops::crop_imm(&result, x, y, w, h).to_image();
                gray = image::imageops::crop_imm(&gray, x, y, w, h).to_image();
                binary = image::imageops::crop_imm(&binary, x, y, w, h).to_image();
                clean = image::imageops::crop_imm(&clean, x, y, w, h).to_image();
            }
        }
        profile.final_crop_seconds = crop_started.elapsed().as_secs_f64();
        profile.failure_stage = Some("png_write");
        let png_start = Instant::now();
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
        clean.save(out.join("rectified-clean.png"))?;
        profile.png_write_seconds += png_start.elapsed().as_secs_f64();
        (
            Some(seconds),
            Some(OutputProcessing {
                binary_border_method: "contour-relative rim components before final interior mask",
                binary_border_margin_pixels: border_margin_pixels,
                binary_border_removed_pixels: border_removed_pixels,
                clean_grayscale: "adaptive paper-white normalization; continuous tone, sigma .5px optical antialiasing",
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
    let shift = |p: &mut Point| {
        p.x -= f64::from(final_crop[0]);
        p.y -= f64::from(final_crop[1]);
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
        r[0] -= f64::from(final_crop[0]);
        r[2] -= f64::from(final_crop[0]);
        r[1] -= f64::from(final_crop[1]);
        r[3] -= f64::from(final_crop[1]);
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
        geometry: &'a crate::Report,
        output_height_width_ratio: Option<f64>,
        render_canvas_height_width_ratio: Option<f64>,
        output_size: [u32; 2],
        uncropped_output_size: [u32; 2],
        final_crop: [u32; 4],
        final_crop_mode: &'static str,
        fit_seconds: f64,
        render_seconds: Option<f64>,
        input_geometry: Option<PathBuf>,
        detection_seconds: f64,
        input_image: PathBuf,
        output_processing: Option<OutputProcessing>,
    }
    let report = Run {
        geometry: geometry.report(),
        output_height_width_ratio: height_width_ratio,
        render_canvas_height_width_ratio: canvas_height_width_ratio,
        output_size: [final_crop[2], final_crop[3]],
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
        std::sync::Arc::new(super::PreviewMap::new(
            geometry,
            Point {
                x: f64::from(ox),
                y: f64::from(oy),
            },
            [data.width, data.height],
            page_output.map(|p| p.region.clone()),
            [raster_width, raster_height],
            final_crop,
        ))
    });
    Ok(ProcessSummary {
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
