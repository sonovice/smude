#![cfg(feature = "application")]

use image::{Rgb, RgbImage};
use smude_geometry::application::{
    DetectionStage, OutputKind, Settings, process_file, process_file_with_progress,
};
use std::sync::{Arc, Mutex};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);

#[test]
#[cfg(feature = "onnx-pages")]
fn color_cover_with_false_staffs_keeps_its_page_geometry_and_color() {
    let fixture = Fixture::new();
    let input =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pages/color-cover.jpg");
    // Match the mobile repository's EXIF-normalization boundary exactly.
    use image::ImageDecoder;
    let mut decoder = image::ImageReader::open(&input)
        .unwrap()
        .into_decoder()
        .unwrap();
    let orientation = decoder.orientation().unwrap();
    let mut original = image::DynamicImage::from_decoder(decoder).unwrap();
    original.apply_orientation(orientation);
    let input = fixture.0.join("oriented.png");
    original.save(&input).unwrap();
    let result = process_file(
        &input,
        &fixture.0.join("color-cover"),
        &Settings {
            page_layout: smude_geometry::application::PageLayout::Single,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.pages.len(), 1);
    let page = &result.pages[0];
    let rejected: serde_json::Value = serde_json::from_slice(
        &fs::read(page.directory.join("rejected-music-observations.json")).unwrap(),
    )
    .unwrap();
    assert!(!rejected["staves"].as_array().unwrap().is_empty());
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(page.directory.join("report.json")).unwrap()).unwrap();
    assert!(report["geometry"]["page_model"]["grid"].is_object());
    assert!(report["profile"]["failure_stage"].is_null());
    let image = image::open(page.image(OutputKind::Clean)).unwrap();
    assert_eq!(image.color(), image::ColorType::Rgb8);
    let rgb = image.to_rgb8();
    let blue = rgb
        .pixels()
        .filter(|p| u16::from(p[2]) > u16::from(p[0]) + 30)
        .count();
    assert!(blue > rgb.width() as usize * rgb.height() as usize / 2);
    assert!(result.warnings.iter().any(|w| w.contains("music")));
}

#[test]
#[cfg(feature = "onnx-pages")]
fn text_page_with_small_color_panel_uses_rgb_paper_tones() {
    let fixture = Fixture::new();
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pages/text-only.jpg");
    let mut photo = image::open(source).unwrap().to_rgb8();
    // A small illustration on otherwise neutral text paper. The page geometry
    // and text remain the same as the existing no-staff integration fixture.
    for y in photo.height() * 62 / 100..photo.height() * 80 / 100 {
        for x in photo.width() * 55 / 100..photo.width() * 80 / 100 {
            photo.put_pixel(x, y, Rgb([70, 140, 170]));
        }
    }
    let input = fixture.0.join("color-text.png");
    photo.save(&input).unwrap();
    let result = process_file(
        &input,
        &fixture.0.join("result"),
        &Settings {
            page_layout: smude_geometry::application::PageLayout::Single,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.pages.len(), 1);
    let output = image::open(result.pages[0].image(OutputKind::Clean)).unwrap();
    assert_eq!(output.color(), image::ColorType::Rgb8);
    let rgb = output.to_rgb8();
    let paper = rgb
        .pixels()
        .filter(|p| p.0.iter().all(|&v| v == 255))
        .count();
    let blue = rgb
        .pixels()
        .filter(|p| u16::from(p[2]) > u16::from(p[0]) + 50)
        .count();
    assert!(
        paper > rgb.width() as usize * rgb.height() as usize / 2,
        "RGB paper was left uncorrected"
    );
    assert!(blue > 5000, "colored illustration was lost");
}

#[test]
#[cfg(feature = "onnx-pages")]
fn learned_page_model_renders_a_document_without_any_staffs() {
    let fixture = Fixture::new();
    let input =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pages/text-only.jpg");
    let output = fixture.0.join("text-only");
    let result = process_file(
        &input,
        &output,
        &Settings {
            page_layout: smude_geometry::application::PageLayout::Single,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.pages.len(), 1);
    let page = &result.pages[0];
    let observations: serde_json::Value =
        serde_json::from_slice(&fs::read(page.directory.join("observations.json")).unwrap())
            .unwrap();
    assert!(observations["staves"].as_array().unwrap().is_empty());
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(page.directory.join("report.json")).unwrap()).unwrap();
    assert!(report["geometry"]["page_model"]["grid"].is_object());
    assert!(report["profile"]["failure_stage"].is_null());
    assert!(
        report["geometry"]["text_warp"]["detected_lines"]
            .as_u64()
            .unwrap()
            > 0
    );
    let image = image::open(page.image(OutputKind::Clean))
        .unwrap()
        .to_luma8();
    assert!(image.width() > 500 && image.height() > 800);
    assert!(image.as_raw().iter().any(|&p| p < 32));
    assert!(image.as_raw().iter().filter(|&&p| p == 255).count() > image.as_raw().len() * 3 / 4);
}

#[test]
#[cfg(feature = "onnx-pages")]
fn full_geometry_keeps_the_camera_boundary_and_ratio() {
    use smude_geometry::detection::{PageDetector, PageLayoutMode, PageSpreadLayout};
    let image = image::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pages/text-only.jpg"),
    )
    .unwrap()
    .to_rgb8();
    let detector = PageDetector::embedded().unwrap();
    let (_, compact) = detector.detect(&image, PageLayoutMode::Single).unwrap();
    let (_, full) = detector
        .detect_full_with_spread_layout(
            &image,
            PageLayoutMode::Single,
            PageSpreadLayout::SideBySide,
        )
        .unwrap();
    assert_eq!(full.model_grids_source[0].len(), 625);
    assert!(
        (compact.height_width_ratios[0].unwrap() - full.height_width_ratios[0].unwrap()).abs()
            < 1e-5
    );
    for (a, b) in compact.raw_boundaries_source[0]
        .iter()
        .zip(&full.raw_boundaries_source[0])
    {
        assert!(
            (a.x - b.x).hypot(a.y - b.y) < 0.03,
            "changed camera boundary: {a:?} {b:?}"
        );
    }
}

#[test]
#[cfg(feature = "onnx-pages")]
fn learned_text_spread_outputs_two_independent_reversible_pages() {
    let fixture = Fixture::new();
    let page = image::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pages/text-only.jpg"),
    )
    .unwrap()
    .to_rgb8();
    let mut source = RgbImage::new(page.width() * 2, page.height());
    image::imageops::replace(&mut source, &page, 0, 0);
    image::imageops::replace(&mut source, &page, i64::from(page.width()), 0);
    let input = fixture.0.join("spread.png");
    source.save(&input).unwrap();
    let result = process_file(
        &input,
        &fixture.0.join("spread"),
        &Settings {
            page_layout: smude_geometry::application::PageLayout::Double,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.pages.len(), 2);
    for (i, page) in result.pages.iter().enumerate() {
        assert!(page.image(OutputKind::Clean).is_file());
        let map = page.mapping.as_ref().unwrap();
        let p = smude_geometry::Point {
            x: 600. + 1200. * i as f64,
            y: 850.,
        };
        assert!(map.contains_source(p));
        let q = map
            .result_to_source(map.source_to_result(p).unwrap())
            .unwrap();
        assert!((p.x - q.x).hypot(p.y - q.y) < 1e-4);
    }
}
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "smude-application-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn score_image() -> RgbImage {
    let mut image = RgbImage::from_pixel(1000, 800, Rgb([255; 3]));
    for center in [180_i32, 380, 580] {
        for k in -2..=2 {
            for x in 100..900 {
                for dy in -1..=1 {
                    image.put_pixel(x, (center + k * 12 + dy) as u32, Rgb([0; 3]));
                }
            }
        }
        for x in [110, 350, 600, 880] {
            for y in center - 24..=center + 24 {
                for dx in -1..=1 {
                    image.put_pixel((x + dx) as u32, y as u32, Rgb([0; 3]));
                }
            }
        }
    }
    image
}

#[test]
fn shared_engine_keeps_default_clean_equal_to_diagnostic_output() {
    let fixture = Fixture::new();
    let mut image = score_image();
    // Weak isolated notation must survive the complete output path, including
    // normalization and any cleanup, not just a tone-function unit test.
    for x in 200..800 {
        image.put_pixel(x, 280, Rgb([220; 3]));
        image.put_pixel(x, 281, Rgb([239; 3]));
    }
    image.put_pixel(450, 260, Rgb([230; 3]));
    let input = fixture.0.join("scan.png");
    image.save(&input).unwrap();
    let progress = Arc::new(Mutex::new(vec![]));
    let receiver = progress.clone();
    let result = process_file_with_progress(
        &input,
        &fixture.0.join("out"),
        &Settings {
            detect_pages: false,
            ..Default::default()
        },
        Some(Arc::new(move |event| receiver.lock().unwrap().push(event))),
    )
    .unwrap();
    let events = progress.lock().unwrap();
    assert_eq!(
        events.iter().map(|e| e.stage).collect::<Vec<_>>(),
        vec![
            DetectionStage::Staves,
            DetectionStage::Barlines,
            DetectionStage::StaffEdges,
            DetectionStage::Text
        ]
    );
    assert_eq!(events[0].paths.len(), 15);
    assert!(
        events[0]
            .paths
            .iter()
            .flatten()
            .all(|p| p.x >= 90. && p.x <= 910. && p.y > 100. && p.y < 650.)
    );
    let plain = process_file(
        &input,
        &fixture.0.join("plain"),
        &Settings {
            detect_pages: false,
            ..Default::default()
        },
    )
    .unwrap();
    let map = result.pages[0].mapping.as_ref().unwrap();
    let measured: smude_geometry::Observations = serde_json::from_slice(
        &fs::read(result.pages[0].directory.join("observations.json")).unwrap(),
    )
    .unwrap();
    let mapped: serde_json::Value = serde_json::from_slice(
        &fs::read(result.pages[0].directory.join("mapped_geometry.json")).unwrap(),
    )
    .unwrap();
    for (i, staff) in measured.staves.iter().enumerate() {
        for k in 0..5 {
            for j in 0..staff.x.len() {
                let source = smude_geometry::Point {
                    x: staff.x[j],
                    y: staff.y[k][j],
                };
                let point = map.source_to_result(source).unwrap();
                let expected = &mapped["staves"][i]["lines"][k][j];
                assert!((point.x - expected["x"].as_f64().unwrap()).abs() < 1e-8);
                assert!((point.y - expected["y"].as_f64().unwrap()).abs() < 1e-8);
                let restored = map.result_to_source(point).unwrap();
                assert!((restored.x - source.x).hypot(restored.y - source.y) < 0.01);
            }
        }
    }
    assert_eq!(result.pages.len(), 1);
    assert!(result.warnings.is_empty());
    let clean = result.pages[0].image(OutputKind::Clean);
    assert_eq!(
        fs::read(&clean).unwrap(),
        fs::read(plain.pages[0].image(OutputKind::Clean)).unwrap()
    );
    for kind in [OutputKind::Gray, OutputKind::Binary, OutputKind::Color] {
        assert!(!result.pages[0].image(kind).exists());
    }
    let diagnostics = smude_geometry::application::run(&smude_geometry::application::Options {
        image: input,
        out: fixture.0.join("diagnostics"),
        no_pages: true,
        diagnostics: true,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(
        fs::read(&clean).unwrap(),
        fs::read(diagnostics.pages[0].image(OutputKind::Clean)).unwrap()
    );
    let raw = image::open(diagnostics.pages[0].image(OutputKind::Gray))
        .unwrap()
        .to_luma8();
    let corrected = image::open(&clean).unwrap().to_luma8();
    let mut weak = 0;
    for (before, after) in raw.as_raw().iter().zip(corrected.as_raw()) {
        if (210..=240).contains(before) {
            weak += 1;
            assert!(
                *after < 255,
                "weak source sample {before} was erased by output processing"
            );
        }
    }
    assert!(
        weak > 500,
        "fixture must exercise weak ink after geometric sampling"
    );
    let mut dimensions = None;
    for kind in [
        OutputKind::Clean,
        OutputKind::Gray,
        OutputKind::Binary,
        OutputKind::Color,
    ] {
        let size = image::image_dimensions(diagnostics.pages[0].image(kind)).unwrap();
        assert_eq!(*dimensions.get_or_insert(size), size);
        assert!(size.0 >= 900 && size.1 >= 700);
    }
}

#[test]
fn page_progress_keeps_source_offsets_and_independent_page_identity() {
    use smude_geometry::{
        PageRegion, PageRegions, Point,
        application::{Options, run},
    };
    let fixture = Fixture::new();
    let score = score_image();
    let mut image = RgbImage::from_pixel(2400, 1000, Rgb([255; 3]));
    for offset in [100, 1300] {
        image::imageops::replace(&mut image, &score, offset, 100);
    }
    let input = fixture.0.join("spread.png");
    image.save(&input).unwrap();
    let regions = PageRegions {
        width: 2400,
        height: 1000,
        regions: [100., 1300.]
            .into_iter()
            .enumerate()
            .map(|(i, x)| PageRegion {
                id: (i + 1).to_string(),
                confidence: 1.,
                evidence: vec![],
                contour: vec![
                    Point { x, y: 100. },
                    Point {
                        x: x + 1000.,
                        y: 100.,
                    },
                    Point {
                        x: x + 1000.,
                        y: 900.,
                    },
                    Point { x, y: 900. },
                ],
            })
            .collect(),
    };
    let regions_file = fixture.0.join("regions.json");
    fs::write(&regions_file, serde_json::to_vec(&regions).unwrap()).unwrap();
    let progress = Arc::new(Mutex::new(vec![]));
    let receiver = progress.clone();
    let processing = Arc::new(Mutex::new(vec![]));
    let processing_receiver = processing.clone();
    let result = run(&Options {
        image: input,
        out: fixture.0.join("out"),
        page_regions: Some(regions_file),
        no_gcs: true,
        no_text_dewarp: true,
        progress: Some(Arc::new(move |event| receiver.lock().unwrap().push(event))),
        processing_progress: Some(Arc::new(move |event| {
            processing_receiver.lock().unwrap().push(event)
        })),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(result.pages.len(), 2);
    {
        use smude_geometry::application::{ProcessingEvent as E, ProcessingStage as S};
        let events = processing.lock().unwrap();
        assert_eq!(events.first(), Some(&E::Stage(S::Preparing)));
        assert_eq!(events.last(), Some(&E::Finished));
        for index in 0..2 {
            let start = events
                .iter()
                .position(|e| *e == E::PageStarted { index, count: 2 })
                .unwrap();
            let end = events
                .iter()
                .position(|e| *e == E::PageFinished { index })
                .unwrap();
            let stages: Vec<_> = events[start..end]
                .iter()
                .filter_map(|e| if let E::Stage(s) = e { Some(*s) } else { None })
                .collect();
            for s in [
                S::Analysis,
                S::Dewarping,
                S::Rendering,
                S::Enhancement,
                S::Saving,
            ] {
                assert!(stages.contains(&s), "missing {s:?} for page {index}");
            }
            assert!(
                stages
                    .windows(2)
                    .all(|s| (s[0] as usize) <= (s[1] as usize))
            );
        }
    }
    for (page, offset) in result.pages.iter().zip([100., 1300.]) {
        let map = page.mapping.as_ref().unwrap();
        let source = smude_geometry::Point {
            x: offset + 350.,
            y: 280.,
        };
        assert!(map.contains_source(source));
        assert!(!map.contains_source(smude_geometry::Point {
            x: offset - 10.,
            y: 280.
        }));
        let restored = map
            .result_to_source(map.source_to_result(source).unwrap())
            .unwrap();
        assert!((source.x - restored.x).hypot(source.y - restored.y) < 1e-7);
    }
    let events = progress.lock().unwrap();
    assert_eq!(
        events.iter().take(2).map(|e| e.stage).collect::<Vec<_>>(),
        vec![DetectionStage::Page; 2]
    );
    for (i, offset) in [100., 1300.].into_iter().enumerate() {
        let page = (i + 1).to_string();
        let events: Vec<_> = events.iter().filter(|e| e.page_id == page).collect();
        assert_eq!(
            events.iter().map(|e| e.stage).collect::<Vec<_>>(),
            vec![
                DetectionStage::Page,
                DetectionStage::ResetMusic,
                DetectionStage::Staves,
                DetectionStage::Barlines,
                DetectionStage::StaffEdges
            ]
        );
        let staves = events
            .iter()
            .find(|e| e.stage == DetectionStage::Staves)
            .unwrap();
        assert_eq!(staves.paths.len(), 15);
        assert!(
            staves
                .paths
                .iter()
                .flatten()
                .all(|p| p.x >= offset + 90. && p.x <= offset + 910. && p.y > 200. && p.y < 750.)
        );
    }
}

#[test]
fn direction_mesh_anchors_ratio_to_paper_rather_than_surrounding_canvas() {
    use smude_geometry::application::{Options, run};
    use smude_geometry::{Barline, Observations, Point, Staff};
    let fixture = Fixture::new();
    let data = Observations {
        width: 1000,
        height: 800,
        staves: [180., 380., 580.]
            .into_iter()
            .map(|y| Staff {
                x: vec![100., 900.],
                y: std::array::from_fn(|k| vec![y + k as f64 * 12.; 2]),
            })
            .collect(),
        barlines: (0..3)
            .flat_map(|staff| {
                [120., 350., 600., 880.].into_iter().map(move |x| Barline {
                    staff,
                    x: vec![x; 2],
                    y: vec![180. + staff as f64 * 200., 228. + staff as f64 * 200.],
                    aligned_staffs: vec![],
                })
            })
            .collect(),
    };
    let corners = [
        Point { x: 120., y: 50. },
        Point { x: 880., y: 50. },
        Point { x: 950., y: 750. },
        Point { x: 50., y: 750. },
    ];
    let observations = fixture.0.join("geometry.json");
    let boundary = fixture.0.join("corners.json");
    fs::write(&observations, serde_json::to_vec(&data).unwrap()).unwrap();
    fs::write(&boundary, serde_json::to_vec(&corners).unwrap()).unwrap();
    let out = fixture.0.join("mesh");
    run(&Options {
        geometry: Some(observations),
        geometry_corners: Some(boundary),
        page_aspect: Some(1.4),
        out: out.clone(),
        direction_mesh: true,
        no_render: true,
        no_text_dewarp: true,
        no_final_crop: true,
        ..Default::default()
    })
    .unwrap();
    let mapped: serde_json::Value =
        serde_json::from_slice(&fs::read(out.join("mapped_geometry.json")).unwrap()).unwrap();
    let points = mapped["page_contour"].as_array().unwrap();
    let extent = |axis: &str| {
        let min = points
            .iter()
            .map(|p| p[axis].as_f64().unwrap())
            .fold(f64::INFINITY, f64::min);
        let max = points
            .iter()
            .map(|p| p[axis].as_f64().unwrap())
            .fold(f64::NEG_INFINITY, f64::max);
        max - min
    };
    assert!(
        (extent("y") / extent("x") - 1.4).abs() < 0.002,
        "paper aspect was replaced by canvas aspect: {}",
        extent("y") / extent("x")
    );
}

#[test]
fn shared_settings_select_mesh_and_keep_older_settings_compatible() {
    use smude_geometry::application::DewarpModel;
    let previous: Settings = serde_json::from_str(r#"{"detect_pages":false}"#).unwrap();
    assert_eq!(previous.dewarp_model, DewarpModel::Mesh);
    let explicit: Settings = serde_json::from_str(r#"{"dewarp_model":"gcs"}"#).unwrap();
    assert_eq!(explicit.dewarp_model, DewarpModel::Gcs);
    let fixture = Fixture::new();
    let input = fixture.0.join("score.png");
    score_image().save(&input).unwrap();
    let settings: Settings = serde_json::from_str(
        r#"{"detect_pages":false,"text_correction":false,"dewarp_model":"mesh"}"#,
    )
    .unwrap();
    let result = process_file(&input, &fixture.0.join("mesh"), &settings).unwrap();
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(result.pages[0].directory.join("report.json")).unwrap())
            .unwrap();
    assert!(report["geometry"]["gcs"].is_null());
    assert_eq!(
        report["geometry"]["direction_mesh"]["model"],
        "page_frame_staff_spacing_direction_mesh_no_gcs"
    );
    let profile: serde_json::Value =
        serde_json::from_slice(&fs::read(result.pages[0].directory.join("profile.json")).unwrap())
            .unwrap();
    assert_eq!(profile["gcs_seconds"], 0.0);
}

#[test]
#[cfg(feature = "onnx-pages")]
fn cropped_score_is_processed_without_inventing_a_paper_boundary() {
    let fixture = Fixture::new();
    let input =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pages/cropped-score.jpg");
    let source = image::open(&input).unwrap().to_rgb8();
    let (regions, _) = smude_geometry::PageDetector::embedded()
        .unwrap()
        .detect_full_with_spread_layout(
            &source,
            smude_geometry::PageLayoutMode::Single,
            smude_geometry::PageSpreadLayout::SideBySide,
        )
        .unwrap();
    assert!(
        regions.regions.is_empty(),
        "fixture must reproduce missing paper boundaries"
    );
    let out = fixture.0.join("result");
    let result = process_file(
        &input,
        &out,
        &Settings {
            page_layout: smude_geometry::application::PageLayout::Single,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.pages.len(), 1);
    let page = &result.pages[0];
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(out.join("pages_report.json")).unwrap()).unwrap();
    assert_eq!(report["page_method"], "cropped-score");
    let mapped: serde_json::Value =
        serde_json::from_slice(&fs::read(page.directory.join("mapped_geometry.json")).unwrap())
            .unwrap();
    assert!(mapped["page_contour"].is_null());
    let image = image::open(page.image(OutputKind::Clean))
        .unwrap()
        .to_luma8();
    let map = page.mapping.as_ref().unwrap();
    let q = map
        .source_to_result(smude_geometry::Point { x: 1047., y: 700. })
        .unwrap();
    assert_eq!(
        image.get_pixel(q.x.round() as u32, q.y.round() as u32)[0],
        255,
        "detached desk strip remains"
    );
    let observations: smude_geometry::Observations =
        serde_json::from_slice(&fs::read(page.directory.join("observations.json")).unwrap())
            .unwrap();
    assert!(observations.staves.len() >= 6);
    let staff = &observations.staves[0];
    let i = staff.x.len() / 2;
    let p = smude_geometry::Point {
        x: staff.x[i],
        y: staff.y[2][i],
    };
    let q = map.source_to_result(p).unwrap();
    // Mapping a subpixel stroke centre to the nearest pixel can land on its
    // antialiased shoulder. The actual dark rule must survive within one pixel.
    let (x, y) = (q.x.round() as u32, q.y.round() as u32);
    assert!(
        (y.saturating_sub(1)..=(y + 1).min(image.height() - 1))
            .any(|y| image.get_pixel(x, y)[0] < 100),
        "staff ink was lost"
    );
    let back = map.result_to_source(q).unwrap();
    assert!((back.x - p.x).hypot(back.y - p.y) < 1e-4);
}
