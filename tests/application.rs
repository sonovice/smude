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
    let image = score_image();
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
    let result = run(&Options {
        image: input,
        out: fixture.0.join("out"),
        page_regions: Some(regions_file),
        no_gcs: true,
        no_text_dewarp: true,
        progress: Some(Arc::new(move |event| receiver.lock().unwrap().push(event))),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(result.pages.len(), 2);
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
