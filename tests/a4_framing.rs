#![cfg(feature = "application")]
use image::{Rgb, RgbImage};
use serde_json::{Value, json};
use smude_geometry::{
    Point, RenderBackend,
    application::{Options, run},
};
use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn final_a4_frame_preserves_pixels_and_updates_overlays_probes_and_linked_preview() {
    let root = std::env::temp_dir().join(format!(
        "smude-a4-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&root).unwrap();
    let mut image = RgbImage::from_pixel(300, 420, Rgb([240; 3]));
    let xs = vec![30., 90., 150., 210., 270.];
    let mut staves = vec![];
    let mut bars = vec![];
    for (staff, center) in [120, 280].into_iter().enumerate() {
        let lines: Vec<Vec<f64>> = (-2..=2)
            .map(|k| vec![(center + 8 * k) as f64; xs.len()])
            .collect();
        staves.push(json!({"x":xs,"y":lines}));
        for k in -2..=2 {
            for x in 30..271 {
                image.put_pixel(x, (center + 8 * k) as u32, Rgb([0; 3]));
            }
        }
        for x in [40, 150, 260] {
            for y in center - 16..=center + 16 {
                image.put_pixel(x, y as u32, Rgb([0; 3]));
            }
            bars.push(
                json!({"staff":staff,"x":[x,x],"y":[center-16,center+16],"aligned_staffs":[]}),
            );
        }
    }
    // Header/footer outside the musical systems must contribute to framing.
    for y in 25..36 {
        for x in 190..245 {
            image.put_pixel(x, y, Rgb([175; 3]));
        }
    }
    for y in 392..402 {
        for x in 145..153 {
            image.put_pixel(x, y, Rgb([160; 3]));
        }
    }
    let input = root.join("input.png");
    image.save(&input).unwrap();
    let geometry = root.join("geometry.json");
    fs::write(
        &geometry,
        json!({"width":300,"height":420,"staves":staves,"barlines":bars}).to_string(),
    )
    .unwrap();
    let probes = root.join("probes.json");
    fs::write(&probes, json!([{"x":150.,"y":120.}]).to_string()).unwrap();
    let process = |name: &str, no_a4: bool| {
        run(&Options {
            image: input.clone(),
            geometry: Some(geometry.clone()),
            page_aspect: Some(1.4),
            out: root.join(name),
            no_a4,
            no_text_dewarp: true,
            diagnostics: true,
            render_backend: RenderBackend::Cpu,
            probe_points: Some(probes.clone()),
            ..Default::default()
        })
        .unwrap()
    };
    let before = process("before", true);
    let after = process("after", false);
    let report: Value =
        serde_json::from_slice(&fs::read(root.join("after/report.json")).unwrap()).unwrap();
    let frame = &report["final_frame"];
    assert_eq!(frame["paper_size_mm"], json!([210, 297]));
    let array = |key: &str| {
        frame[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>()
    };
    let crop = array("source_crop");
    let offset = array("content_offset");
    let size = array("output_size");
    assert!((f64::from(size[1]) - f64::from(size[0]) * 297. / 210.).abs() <= 0.5);
    assert_eq!(report["output_size"], frame["output_size"]);
    let shift = [
        offset[0] as f64 - crop[0] as f64,
        offset[1] as f64 - crop[1] as f64,
    ];
    for name in [
        "rectified-clean.png",
        "rectified-gray.png",
        "rectified-binary.png",
        "rectified.png",
    ] {
        let a = image::open(root.join("before").join(name))
            .unwrap()
            .to_rgb8();
        let b = image::open(root.join("after").join(name))
            .unwrap()
            .to_rgb8();
        assert_eq!(b.dimensions(), (size[0], size[1]));
        for y in 0..crop[3] {
            for x in 0..crop[2] {
                assert_eq!(
                    a.get_pixel(crop[0] + x, crop[1] + y),
                    b.get_pixel(offset[0] + x, offset[1] + y),
                    "{name} pixel changed"
                );
            }
        }
        assert_eq!(b.get_pixel(0, 0).0, [255; 3]);
    }
    let p = Point { x: 150., y: 120. };
    let a = before.pages[0]
        .mapping
        .as_ref()
        .unwrap()
        .source_to_result(p)
        .unwrap();
    let map = after.pages[0].mapping.as_ref().unwrap();
    let b = map.source_to_result(p).unwrap();
    assert!((b.x - a.x - shift[0]).abs() < 1e-7 && (b.y - a.y - shift[1]).abs() < 1e-7);
    let back = map.result_to_source(b).unwrap();
    assert!((back.x - p.x).hypot(back.y - p.y) < 1e-5);
    let mapped: Value =
        serde_json::from_slice(&fs::read(root.join("after/mapped_geometry.json")).unwrap())
            .unwrap();
    let point = &mapped["staves"][0]["lines"][2][2];
    assert!((point["x"].as_f64().unwrap() - b.x).abs() < 1e-7);
    assert!((point["y"].as_f64().unwrap() - b.y).abs() < 1e-7);
    let probes: Value =
        serde_json::from_slice(&fs::read(root.join("after/mapped_probes.json")).unwrap()).unwrap();
    assert!((probes[0]["x"].as_f64().unwrap() - b.x).abs() < 1e-7);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn top_rim_is_not_protected_by_the_first_staff_annotation_envelope() {
    let root = std::env::temp_dir().join(format!("smude-top-rim-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let mut image = RgbImage::from_pixel(300, 420, Rgb([240; 3]));
    for x in 30..270 {
        image.put_pixel(x, 4, Rgb([0; 3]));
    }
    for y in [64, 72, 80, 88, 96] {
        for x in 30..270 {
            image.put_pixel(x, y, Rgb([0; 3]));
        }
    }
    // A real stem extending above the staff remains part of the image.
    for y in 46..65 {
        image.put_pixel(150, y, Rgb([0; 3]));
    }
    let input = root.join("input.png");
    image.save(&input).unwrap();
    let geometry = root.join("geometry.json");
    fs::write(&geometry, json!({"width":300,"height":420,"staves":[{"x":[30.,90.,150.,210.,269.],"y":[vec![64.;5],vec![72.;5],vec![80.;5],vec![88.;5],vec![96.;5]]}],"barlines":[]}).to_string()).unwrap();
    let contour = root.join("contour.json");
    fs::write(
        &contour,
        json!([{"x":0.,"y":0.},{"x":299.,"y":0.},{"x":299.,"y":419.},{"x":0.,"y":419.}])
            .to_string(),
    )
    .unwrap();
    run(&Options {
        image: input,
        geometry: Some(geometry),
        geometry_contour: Some(contour),
        out: root.join("out"),
        no_a4: true,
        no_text_dewarp: true,
        diagnostics: true,
        render_backend: RenderBackend::Cpu,
        ..Default::default()
    })
    .unwrap();
    let result = image::open(root.join("out/rectified-clean.png"))
        .unwrap()
        .to_luma8();
    assert!(
        (0..10).all(|y| (35..260).all(|x| result.get_pixel(x, y)[0] == 255)),
        "The broad text-warp guard must not shelter the top rim"
    );
    assert!(
        result.get_pixel(150, 50)[0] < 128,
        "Keep the genuine music stem"
    );
    fs::remove_dir_all(root).unwrap();
}
