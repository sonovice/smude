//! Page detection and independent per-page processing.
use crate::detection::PageRegions;
use std::{fs, path::PathBuf, time::Instant};

use crate::geometry::Point;
use serde::Serialize;

#[cfg(feature = "onnx-pages")]
use super::options::{PageLayout, SpreadLayout};
use super::{
    options::Options,
    process::{PageOutput, ProcessSummary, process},
    profile::{StepProfile, finish_profile},
};

#[derive(Serialize)]
struct PageRun {
    id: String,
    source_bounds: [u32; 4],
    coordinate_space: &'static str,
    assignment_status: &'static str,
    dir: PathBuf,
    status: &'static str,
    error: Option<String>,
    warning: Option<String>,
    seconds: f64,
    detected_staves: Option<usize>,
    detected_barlines: Option<usize>,
    measurements: Option<ProcessSummary>,
}

#[derive(Serialize)]
struct PagesRun {
    schema: &'static str,
    input_image: PathBuf,
    input_regions: Option<PathBuf>,
    page_method: &'static str,
    coordinate_space: &'static str,
    status: &'static str,
    error: Option<&'static str>,
    region_seconds: f64,
    decode_seconds: f64,
    total_seconds: f64,
    pages: Vec<PageRun>,
}

fn automatic_regions(
    args: &Options,
    image: &image::RgbImage,
) -> Result<PageRegions, Box<dyn std::error::Error>> {
    #[cfg(feature = "onnx-pages")]
    {
        let started = Instant::now();
        let supplied;
        let detector = if let Some(directory) = &args.page_models {
            supplied = crate::detection::PageDetector::load(directory)?;
            &supplied
        } else {
            crate::detection::PageDetector::embedded_shared()?
        };
        let model_load_seconds = started.elapsed().as_secs_f64();
        let layout = match args.page_layout {
            PageLayout::Auto => crate::detection::PageLayoutMode::Auto,
            PageLayout::Single => crate::detection::PageLayoutMode::Single,
            PageLayout::Double => crate::detection::PageLayoutMode::Double,
        };
        let spread_layout = match args.spread_layout {
            SpreadLayout::SideBySide => crate::detection::PageSpreadLayout::SideBySide,
            SpreadLayout::Stacked => crate::detection::PageSpreadLayout::Stacked,
        };
        let (regions, detection) =
            detector.detect_full_with_spread_layout(image, layout, spread_layout)?;
        fs::write(
            args.out.join("page_detection.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "backend":"onnxruntime", "model_directory":args.page_models,
                "model_source": if args.page_models.is_some() { "directory" } else { "embedded" },
                "model_load_seconds":model_load_seconds, "detection":detection,
                "neural_mesh_warp_requested":true
            }))?,
        )?;
        Ok(regions)
    }
    #[cfg(not(feature = "onnx-pages"))]
    {
        let _ = (args, image);
        Err("page-boundary model page detection requires --features onnx-pages".into())
    }
}

pub(super) fn process_pages(
    args: &Options,
) -> Result<super::ProcessingResult, Box<dyn std::error::Error>> {
    let run_started = Instant::now();
    fs::create_dir_all(&args.out)?;
    let decode_started = Instant::now();
    // The bundled page models initialise while the image decodes. Detection
    // then finds them ready, or reports why they are not.
    let image = std::thread::scope(|scope| {
        #[cfg(feature = "onnx-pages")]
        if args.page_regions.is_none() && args.page_models.is_none() {
            scope.spawn(crate::detection::PageDetector::embedded_shared);
        }
        #[cfg(not(feature = "onnx-pages"))]
        let _ = scope;
        super::decode::open_rgb(&args.image)
    })?;
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    args.processing(super::ProcessingEvent::InputPixels(
        u64::from(image.width()) * u64::from(image.height()),
    ));
    args.processing(super::ProcessingEvent::Stage(
        super::ProcessingStage::PageDetection,
    ));
    let start = Instant::now();
    let regions: PageRegions = if let Some(path) = &args.page_regions {
        serde_json::from_slice(&fs::read(path)?)?
    } else {
        match automatic_regions(args, &image) {
            Ok(regions) => regions,
            Err(error) => {
                fs::write(
                    args.out.join("pages_report.json"),
                    serde_json::to_vec_pretty(&serde_json::json!({
                        "schema": "smude.pages_run.v1",
                        "input_image": args.image,
                        "status": "failed",
                        "error": error.to_string(),
                        "pages": []
                    }))?,
                )?;
                return Err(error);
            }
        }
    };
    if regions.regions.is_empty()
        && args.page_regions.is_none()
        && args.page_layout != super::PageLayout::Double
    {
        let detection: serde_json::Value =
            serde_json::from_slice(&fs::read(args.out.join("page_detection.json"))?)?;
        // Do not merge a failed spread or override explicitly supplied regions.
        if detection["detection"]["selected_layout"] != "double" {
            fs::write(
                args.out.join("pages.json"),
                serde_json::to_vec_pretty(&regions)?,
            )?;
            if let Some(result) = super::cropped_score::try_process(
                args,
                &image,
                run_started,
                decode_seconds,
                start.elapsed().as_secs_f64(),
            )? {
                return Ok(result);
            }
        }
    }
    if !regions.regions.is_empty() || args.page_regions.is_some() {
        regions.validate()?;
    }
    if (regions.width, regions.height) != image.dimensions() {
        return Err("page-region dimensions differ from input image".into());
    }
    for region in &regions.regions {
        args.emit(&region.id, super::DetectionStage::Page, || {
            let mut contour = region.contour.clone();
            if let Some(first) = contour.first().copied() {
                contour.push(first);
            }
            vec![contour]
        });
    }
    fs::write(
        args.out.join("pages.json"),
        serde_json::to_vec_pretty(&regions)?,
    )?;
    let mut report = PagesRun {
        schema: "smude.pages_run.v1",
        input_image: args.image.clone(),
        input_regions: args.page_regions.clone(),
        page_method: if args.page_regions.is_some() {
            "supplied"
        } else {
            "onnxruntime-full-page-grid"
        },
        coordinate_space: "original_image_pixels",
        status: "running",
        error: None,
        region_seconds: start.elapsed().as_secs_f64(),
        decode_seconds,
        total_seconds: 0.,
        pages: vec![],
    };
    let ratio_report: serde_json::Value = if args.page_regions.is_none() {
        fs::read(args.out.join("page_detection.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default()
    } else {
        serde_json::Value::Null
    };
    for (page_index, region) in regions.regions.iter().enumerate() {
        args.processing(super::ProcessingEvent::PageStarted {
            index: page_index,
            count: regions.regions.len(),
        });
        args.processing(super::ProcessingEvent::Stage(
            super::ProcessingStage::Analysis,
        ));
        let ratio = region
            .id
            .parse::<usize>()
            .ok()
            .and_then(|id| id.checked_sub(1))
            .and_then(|i| ratio_report["detection"]["height_width_ratios"][i].as_f64());
        let corners = if (args.perspective || !args.no_perspective)
            && (args.page_perspective || !args.no_page_perspective)
        {
            region
                .id
                .parse::<usize>()
                .ok()
                .and_then(|id| id.checked_sub(1))
                .and_then(|i| {
                    let points: Vec<Point> = serde_json::from_value(
                        ratio_report["detection"]["raw_boundaries_source"][i].clone(),
                    )
                    .ok()?;
                    if points.len() != 96 {
                        return None;
                    }
                    let c = [points[0], points[24], points[48], points[72]];
                    let [ox, oy, x1, y1] = region.bounds();
                    let limit = f64::from((x1 - ox).min(y1 - oy)) * 0.02;
                    let deviation = |p: Point, a: Point, b: Point| {
                        ((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)).abs()
                            / (b.x - a.x).hypot(b.y - a.y).max(1.)
                    };
                    if (24..=48).any(|j| deviation(points[j], c[1], c[2]) > limit)
                        || (72..96).any(|j| deviation(points[j], c[3], c[0]) > limit)
                    {
                        return None;
                    }
                    Some(c.map(|p| Point {
                        x: p.x - f64::from(ox),
                        y: p.y - f64::from(oy),
                    }))
                })
        } else {
            None
        };
        let start = Instant::now();
        let dir = args.out.join(format!("page-{}", region.id));
        fs::create_dir_all(&dir)?;
        let mut entry = PageRun {
            id: region.id.clone(),
            source_bounds: region.bounds(),
            coordinate_space: "page_crop_pixels",
            assignment_status: if args.page_regions.is_some() {
                "supplied_region"
            } else if regions.regions.len() == 1 {
                "unresolved_page_identity"
            } else {
                "proposed_page_partition"
            },
            dir: dir.clone(),
            status: "failed",
            error: None,
            warning: None,
            seconds: 0.,
            detected_staves: None,
            detected_barlines: None,
            measurements: None,
        };
        // A page failure is local: siblings still get observations and a fit.
        let mut profile = StepProfile::start();
        let result = (|| -> Result<ProcessSummary, Box<dyn std::error::Error>> {
            let crop_started = Instant::now();
            let crop = crate::detection::crop_page(&image, region)?;
            profile.source_crop_seconds = crop_started.elapsed().as_secs_f64();
            let page_model = region
                .id
                .parse::<usize>()
                .ok()
                .and_then(|id| id.checked_sub(1))
                .and_then(|i| {
                    serde_json::from_value::<Vec<Point>>(
                        ratio_report["detection"]["model_grids_source"][i].clone(),
                    )
                    .ok()
                })
                .zip(ratio)
                .map(|(mut points, ratio)| {
                    let [x, y, _, _] = region.bounds();
                    for p in &mut points {
                        p.x -= f64::from(x);
                        p.y -= f64::from(y);
                    }
                    crate::geometry::PageAnalysisMap::from_grid(
                        crop.width(),
                        crop.height(),
                        points,
                        ratio,
                    )
                })
                .transpose()?;
            let write_started = Instant::now();
            if args.diagnostics {
                crop.save(dir.join("source.png"))?;
            }
            profile.source_png_write_seconds = write_started.elapsed().as_secs_f64();
            profile.failure_stage = Some("analysis_perspective");
            let prep_started = Instant::now();
            let mut analysis_rejection = if page_model.is_some() {
                None
            } else if args.no_analysis_perspective {
                Some("disabled".to_string())
            } else if corners.is_none() || ratio.is_none() {
                Some("no_supported_corners_or_ratio".to_string())
            } else {
                None
            };
            let analysis_width = args.analysis_width.unwrap_or_else(|| {
                args.analysis_long_edge.map_or(1800, |limit| {
                    (f64::from(limit) / ratio.unwrap_or(1.).max(1.))
                        .floor()
                        .max(64.) as u32
                })
            });
            let analysis_map = page_model
                .as_ref()
                .map(|m| m.bounded(analysis_width))
                .or_else(|| {
                    corners
                        .zip(ratio)
                        .filter(|_| !args.no_analysis_perspective)
                        .and_then(|(corners, ratio)| {
                            crate::geometry::PageAnalysisMap::new(
                                crop.width(),
                                crop.height(),
                                corners,
                                ratio,
                                args.analysis_width.unwrap_or_else(|| {
                                    args.analysis_long_edge.map_or(1800, |limit| {
                                        (f64::from(limit) / ratio.max(1.)).floor().max(64.) as u32
                                    })
                                }),
                            )
                            .inspect_err(|error| {
                                analysis_rejection = Some(error.to_string());
                            })
                            .ok()
                        })
                });
            let prepared = analysis_map
                .as_ref()
                .map(|map| map.render_analysis(&crop))
                .transpose()?;
            let analysis_region = analysis_map.as_ref().map(|map| {
                let [ox, oy, _, _] = region.bounds();
                crate::detection::PageRegion {
                    id: region.id.clone(),
                    contour: region
                        .contour
                        .iter()
                        .filter_map(|p| {
                            map.forward(Point {
                                x: p.x - f64::from(ox),
                                y: p.y - f64::from(oy),
                            })
                        })
                        .collect(),
                    confidence: region.confidence,
                    evidence: vec!["original boundary in projective analysis coordinates".into()],
                }
            });
            profile.analysis_perspective_seconds = prep_started.elapsed().as_secs_f64();
            fs::write(
                dir.join("analysis-preparation.json"),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "enabled": analysis_map.is_some(), "rejection": analysis_rejection,
                    "height_width_ratio": ratio, "coordinate_space": "original_page_crop_pixels",
                    "analysis_dimensions": analysis_map.as_ref().map(|m| m.dimensions()),
                    "final_source": "original full-resolution page crop; analysis pixels are never used for final output"
                }))?,
            )?;
            if let (Some(map), Some(prepared)) = (&analysis_map, &prepared) {
                let write_started = Instant::now();
                if args.diagnostics {
                    prepared.save(dir.join("analysis.png"))?;
                }
                fs::write(
                    dir.join("analysis-map.json"),
                    serde_json::to_vec_pretty(map)?,
                )?;
                profile.analysis_png_write_seconds = write_started.elapsed().as_secs_f64();
            }
            let mut attempt = |use_prepared: bool,
                               crop: image::RgbImage,
                               profile: &mut StepProfile|
             -> Result<ProcessSummary, Box<dyn std::error::Error>> {
                args.processing(super::ProcessingEvent::Stage(
                    super::ProcessingStage::Analysis,
                ));
                profile.failure_stage = Some("native_detection");
                let detection_start = Instant::now();
                args.emit(&region.id, super::DetectionStage::ResetMusic, Vec::new);
                let [ox, oy, _, _] = region.bounds();
                let to_original = |p: Point| Point {
                    x: p.x + f64::from(ox),
                    y: p.y + f64::from(oy),
                };
                let on_staves = |staves: &[crate::geometry::Staff]| {
                    args.emit_staves(&region.id, staves, |p| {
                        let p = if use_prepared {
                            analysis_map.as_ref().unwrap().inverse(p).unwrap_or(Point {
                                x: f64::NAN,
                                y: f64::NAN,
                            })
                        } else {
                            p
                        };
                        to_original(p)
                    });
                };
                let detected = if use_prepared {
                    let prepared = prepared.as_ref().unwrap();
                    crate::detection::music::detect_masked(
                        prepared,
                        prepared.width(),
                        analysis_region.as_ref(),
                        Some(&on_staves),
                    )
                    .and_then(|data| analysis_map.as_ref().unwrap().observations_to_source(&data))
                } else {
                    let local = crate::detection::music::local_region(region, ox, oy);
                    crate::detection::music::detect_masked(
                        &crop,
                        args.analysis_width_for(&crop),
                        Some(&local),
                        Some(&on_staves),
                    )
                };
                let detection_seconds = detection_start.elapsed().as_secs_f64();
                profile.native_detection_seconds = detection_seconds;
                let data = match detected {
                    Ok(data) => data,
                    Err(crate::Error::Invalid("no reliable five-line staves found"))
                        if page_model.is_some() =>
                    {
                        crate::geometry::Observations {
                            width: crop.width(),
                            height: crop.height(),
                            staves: vec![],
                            barlines: vec![],
                        }
                    }
                    Err(error) => return Err(error.into()),
                };
                entry.detected_staves = Some(data.staves.len());
                entry.detected_barlines = Some(data.barlines.len());
                let page_output = Some(PageOutput {
                    page_model: page_model.as_ref(),
                    ratio,
                    region,
                    corners,
                    camera_center: Point {
                        x: f64::from(image.width() - 1) / 2. - f64::from(region.bounds()[0]),
                        y: f64::from(image.height() - 1) / 2. - f64::from(region.bounds()[1]),
                    },
                    analysis_dimensions: if use_prepared {
                        analysis_map.as_ref().map(|map| {
                            let (w, h) = map.dimensions();
                            [w, h]
                        })
                    } else {
                        None
                    },
                });
                // A colorful cover can produce short false five-line combs in
                // lettering. A rejected music perspective is not evidence that
                // the independently validated learned page grid is unusable.
                let color_without_bars = page_model.is_some()
                    && !data.staves.is_empty()
                    && data.barlines.is_empty()
                    && crate::rendering::output::preserve_color(&crop, false);
                let result = process(
                    args,
                    &dir,
                    &data,
                    Some(crop),
                    detection_seconds,
                    page_output,
                    profile,
                );
                let rejected_music_perspective = result.as_ref().err().is_some_and(|error| {
                    matches!(error.downcast_ref::<crate::Error>(), Some(crate::Error::Invalid(
                        "perspective fit is underconstrained or reached a parameter bound; inspect size groups or disable perspective"
                    )))
                });
                if !color_without_bars || !rejected_music_perspective {
                    return result;
                }
                fs::write(
                    dir.join("rejected-music-observations.json"),
                    serde_json::to_vec(&data)?,
                )?;
                fs::write(
                    dir.join("rejected-music-profile.json"),
                    serde_json::to_vec_pretty(profile)?,
                )?;
                let reason = result.as_ref().err().unwrap().to_string();
                entry.warning = Some(format!(
                    "Color page uses learned page geometry without rejected music residuals: {reason}"
                ));
                args.emit(&region.id, super::DetectionStage::ResetMusic, Vec::new);
                profile.restart_attempt();
                let crop = crate::detection::crop_page(&image, region)?;
                let no_music = crate::geometry::Observations {
                    width: crop.width(),
                    height: crop.height(),
                    staves: vec![],
                    barlines: vec![],
                };
                process(
                    args,
                    &dir,
                    &no_music,
                    Some(crop),
                    detection_seconds,
                    page_output,
                    profile,
                )
            };
            let first = attempt(analysis_map.is_some(), crop, &mut profile);
            let first_error = first.as_ref().err().map(ToString::to_string);
            let retry = analysis_map.is_some()
                && first.is_err()
                && matches!(
                    profile.failure_stage,
                    Some(
                        "native_detection"
                            | "input_validation"
                            | "perspective"
                            | "staff_warp"
                            | "system_shear"
                            | "map_validation"
                    )
                );
            let result = if retry {
                fs::write(
                    dir.join("analysis-attempt-profile.json"),
                    serde_json::to_vec_pretty(&profile)?,
                )?;
                if dir.join("observations.json").exists() {
                    fs::copy(
                        dir.join("observations.json"),
                        dir.join("analysis-attempt-observations.json"),
                    )?;
                }
                // The first attempt consumed the crop: only this rare retry pays for
                // a second one instead of every page holding a full-resolution copy.
                let crop = crate::detection::crop_page(&image, region)?;
                profile.restart_attempt();
                attempt(false, crop, &mut profile)
            } else {
                first
            };
            fs::write(
                dir.join("analysis-attempts.json"),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "prepared_analysis": analysis_map.is_some(), "retried_original_crop": retry,
                    "first_error": first_error, "final_success": result.is_ok()
                }))?,
            )?;
            result
        })();
        finish_profile(&dir, &mut profile, &result)?;
        match result {
            Ok(summary) => {
                entry.status = "success";
                entry.measurements = Some(summary);
            }
            Err(error) => {
                let message = error.to_string();
                fs::write(dir.join("error.log"), &message)?;
                entry.error = Some(message);
            }
        }
        entry.seconds = start.elapsed().as_secs_f64();
        args.processing(super::ProcessingEvent::PageFinished { index: page_index });
        report.pages.push(entry);
        fs::write(
            args.out.join("pages_report.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
    }
    report.total_seconds = run_started.elapsed().as_secs_f64();
    let successes = report
        .pages
        .iter()
        .filter(|page| page.status == "success")
        .count();
    report.status = if successes == report.pages.len() && successes > 0 {
        "success"
    } else if successes > 0 {
        "partial"
    } else {
        "failed"
    };
    if successes == 0 {
        report.error = Some(if report.pages.is_empty() {
            "no_confident_regions"
        } else {
            "no_successful_pages"
        });
    }
    fs::write(
        args.out.join("pages_report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "{successes}/{} pages succeeded; {}",
        report.pages.len(),
        args.out.display()
    );
    if successes == 0 {
        return Err("no page produced a successful result; inspect pages_report.json".into());
    }
    Ok(super::ProcessingResult {
        pages: if args.detect_only || args.no_render {
            vec![]
        } else {
            report
                .pages
                .iter()
                .filter(|p| p.status == "success")
                .map(|p| super::RenderedPage {
                    quality: p
                        .measurements
                        .as_ref()
                        .map(|m| m.quality.clone())
                        .unwrap_or_default(),
                    id: p.id.clone(),
                    directory: p.dir.clone(),
                    mapping: p.measurements.as_ref().and_then(|m| m.mapping.clone()),
                })
                .collect()
        },
        warnings: report
            .pages
            .iter()
            .flat_map(|p| {
                p.error
                    .iter()
                    .chain(p.warning.iter())
                    .map(|e| format!("Page {}: {e}", p.id))
            })
            .collect(),
    })
}
