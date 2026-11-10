//! The actual native detector adapter is also tested on the host. No mobile
//! UI or synthetic boundary predictions stand in for the shared model.
#[path = "../src/camera/inference.rs"]
mod inference;

#[test]
#[ignore = "requires local photo corpus; set SMUDE_CAMERA_PHOTOS to colon-separated paths"]
fn real_camera_photos() {
    let paths = std::env::var("SMUDE_CAMERA_PHOTOS").expect("photo paths");
    for path in paths.split(':') {
        let input = image::open(path)
            .expect("open photo")
            .resize(640, 640, image::imageops::FilterType::Triangle)
            .to_rgba8();
        let result = inference::detect(
            input.as_raw(),
            input.width() as usize,
            input.height() as usize,
            input.width() as usize * 4,
            false,
        );
        assert!(result.get("error").is_none(), "{path}: {result}");
        assert_eq!(result["scene"].as_array().unwrap().len(), 64);
        println!(
            "{}",
            serde_json::json!({"photo":path,"class":result["selected_class"],"scores":result["class_scores"],"pages":result["pages"],"drawn":result["polygons"].as_array().unwrap().len(),"accepted":result["accepted_pages"],"ready_candidate":result["valid"],"rejected":result["rejected"],"milliseconds":result["milliseconds"]})
        );
    }
}

#[test]
#[ignore = "set SMUDE_CAMERA_SPREAD_PHOTO to an independently identified complete two-page fixture"]
fn auto_detected_spread_keeps_both_contours_and_capture_eligibility() {
    let path = std::env::var("SMUDE_CAMERA_SPREAD_PHOTO").expect("complete spread fixture");
    let input = image::open(path)
        .expect("open spread")
        .resize(640, 640, image::imageops::FilterType::Triangle)
        .to_rgba8();
    let result = inference::detect(
        input.as_raw(),
        input.width() as usize,
        input.height() as usize,
        input.width() as usize * 4,
        false,
    );
    assert_eq!(result["pages"], 2, "{result}");
    assert_eq!(result["accepted_pages"], 2, "{result}");
    assert_eq!(result["polygons"].as_array().unwrap().len(), 2);
    assert_eq!(result["valid"], true, "{result}");
    println!(
        "spread class={} raw scores={}",
        result["selected_class"], result["class_scores"]
    );
}

#[test]
#[ignore = "set SMUDE_CAMERA_ORIENTATION_PHOTO to a real upright photo"]
fn rotation_locked_input_matches_upright_detection_and_preview_coordinates() {
    let path = std::env::var("SMUDE_CAMERA_ORIENTATION_PHOTO").expect("upright fixture");
    let upright = image::open(path)
        .unwrap()
        .resize(640, 640, image::imageops::FilterType::Triangle)
        .to_rgba8();
    let run = |im: &image::RgbaImage, turns| {
        inference::detect_oriented(
            im.as_raw(),
            im.width() as usize,
            im.height() as usize,
            im.width() as usize * 4,
            false,
            turns,
        )
    };
    let reference = run(&upright, 0);
    assert_eq!(
        reference["valid"], true,
        "fixture must contain complete detected pages: {reference}"
    );
    for encoded_turns in 1..=3 {
        let sideways = match encoded_turns {
            1 => image::imageops::rotate90(&upright),
            2 => image::imageops::rotate180(&upright),
            _ => image::imageops::rotate270(&upright),
        };
        let corrected = run(&sideways, 4 - encoded_turns);
        assert_eq!(corrected["selected_class"], reference["selected_class"]);
        assert_eq!(corrected["pages"], reference["pages"]);
        assert_eq!(corrected["valid"], true);
        assert_eq!(corrected["width"], sideways.width());
        assert_eq!(corrected["height"], sideways.height());
        for (a, b) in reference["polygons"]
            .as_array()
            .unwrap()
            .iter()
            .zip(corrected["polygons"].as_array().unwrap())
        {
            for (p, q) in a.as_array().unwrap().iter().zip(b.as_array().unwrap()) {
                let (x, y) = (
                    p[0].as_f64().unwrap() * upright.width() as f64,
                    p[1].as_f64().unwrap() * upright.height() as f64,
                );
                let (xx, yy) = match encoded_turns {
                    1 => (upright.height() as f64 - 1. - y, x),
                    2 => (
                        upright.width() as f64 - 1. - x,
                        upright.height() as f64 - 1. - y,
                    ),
                    _ => (y, upright.width() as f64 - 1. - x),
                };
                assert!((q[0].as_f64().unwrap() * sideways.width() as f64 - xx).abs() < 1e-6);
                assert!((q[1].as_f64().unwrap() * sideways.height() as f64 - yy).abs() < 1e-6);
            }
        }
        println!(
            "rotation {encoded_turns}: class={}, pages={}, exact preview mapping",
            corrected["selected_class"], corrected["pages"]
        );
    }
}

#[test]
#[ignore = "set SMUDE_CAMERA_FLAT_SPREAD to the failing rotation-locked camera screenshot crop"]
fn flat_rotation_locked_spread_recovers_from_overlapping_sideways_predictions() {
    let path = std::env::var("SMUDE_CAMERA_FLAT_SPREAD").expect("sideways failing spread");
    let image = image::open(path)
        .unwrap()
        .resize(640, 640, image::imageops::FilterType::Triangle)
        .to_rgba8();
    let result = inference::detect_oriented(
        image.as_raw(),
        image.width() as usize,
        image.height() as usize,
        image.width() as usize * 4,
        false,
        0,
    );
    assert_eq!(result["pages"], 2, "{result}");
    assert_eq!(result["accepted_pages"], 2, "{result}");
    assert_eq!(result["valid"], true, "{result}");
    assert_eq!(result["polygons"].as_array().unwrap().len(), 2);
    assert!([1, 3].contains(&result["analysis_turns"].as_u64().unwrap()));
}
