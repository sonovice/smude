#![cfg(feature = "cli")]

use image::{Rgb, RgbImage};
use serde_json::{Value, json};
use std::{
    fs,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn cli_content_crop_keeps_output_pixels_and_geometry_consistent() {
    let root = std::env::temp_dir().join(format!(
        "smude-output-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&root).unwrap();
    let mut source = RgbImage::from_pixel(240, 200, Rgb([240, 235, 220]));
    let mut staves = vec![];
    let mut bars = vec![];
    for (s, center) in [50, 130].iter().enumerate() {
        let rows: Vec<Vec<f64>> = (-2..=2)
            .map(|k| vec![f64::from(center + 8 * k); 5])
            .collect();
        staves.push(json!({"x":[20.,50.,100.,150.,220.],"y":rows}));
        for k in -2..=2 {
            for x in 20..=220 {
                source.put_pixel(x, (center + 8 * k) as u32, Rgb([0; 3]));
            }
        }
        for x in [60, 120, 180] {
            bars.push(json!({"staff":s,"x":[x as f64,x as f64],"y":[center-16,center+16],"aligned_staffs":[]}));
        }
    }
    let image = root.join("source.png");
    let observations = root.join("geometry.json");
    source.save(&image).unwrap();
    fs::write(
        &observations,
        serde_json::to_vec(&json!({"width":240,"height":200,"staves":staves,"barlines":bars}))
            .unwrap(),
    )
    .unwrap();
    let out = root.join("default");
    let call = |destination: &std::path::Path, flags: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_smude-v2"))
            .arg("--image")
            .arg(&image)
            .arg("--out")
            .arg(destination)
            .arg("--geometry")
            .arg(&observations)
            .args(flags)
            .output()
            .unwrap()
    };
    let run = call(&out, &["--crop-mode", "content"]);
    let default = root.join("page-default");
    assert!(call(&default, &[]).status.success());
    let default_report: Value =
        serde_json::from_slice(&fs::read(default.join("report.json")).unwrap()).unwrap();
    assert_eq!(default_report["final_crop_mode"], "page");
    assert_eq!(
        default_report["output_size"],
        default_report["uncropped_output_size"]
    );
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let rgb = image::open(out.join("rectified.png")).unwrap().to_rgb8();
    let gray = image::open(out.join("rectified-gray.png"))
        .unwrap()
        .to_luma8();
    let binary = image::open(out.join("rectified-binary.png"))
        .unwrap()
        .to_luma8();
    let clean = image::open(out.join("rectified-clean.png"))
        .unwrap()
        .to_luma8();
    assert_eq!(clean.dimensions(), rgb.dimensions());
    assert!(clean.as_raw().iter().any(|v| *v > 0 && *v < 255));
    assert_eq!(gray, image::imageops::grayscale(&rgb));
    assert_eq!(binary.dimensions(), rgb.dimensions());
    assert!(binary.as_raw().iter().all(|p| [0, 255].contains(p)));
    let report: Value =
        serde_json::from_slice(&fs::read(out.join("report.json")).unwrap()).unwrap();
    let cx = report["final_crop"][0].as_u64().unwrap() as u32;
    let cy = report["final_crop"][1].as_u64().unwrap() as u32;
    assert!(cx > 0 && cy > 0);
    assert_eq!(binary.get_pixel(100 - cx, 50 - cy)[0], 0);
    assert_eq!(binary.get_pixel(0, 0)[0], 255);
    assert_eq!(report["output_size"][0], rgb.width());
    assert_eq!(report["output_size"][1], rgb.height());
    let uncropped = root.join("uncropped");
    assert!(call(&uncropped, &["--no-final-crop"]).status.success());
    let full_rgb = image::open(uncropped.join("rectified.png"))
        .unwrap()
        .to_rgb8();
    assert_eq!(
        rgb,
        image::imageops::crop_imm(&full_rgb, cx, cy, rgb.width(), rgb.height()).to_image()
    );
    let full_mapped: Value =
        serde_json::from_slice(&fs::read(uncropped.join("mapped_geometry.json")).unwrap()).unwrap();
    let cropped_mapped: Value =
        serde_json::from_slice(&fs::read(out.join("mapped_geometry.json")).unwrap()).unwrap();
    for (a, b) in cropped_mapped["staves"]
        .as_array()
        .unwrap()
        .iter()
        .zip(full_mapped["staves"].as_array().unwrap())
    {
        for (la, lb) in a["lines"]
            .as_array()
            .unwrap()
            .iter()
            .zip(b["lines"].as_array().unwrap())
        {
            for (pa, pb) in la.as_array().unwrap().iter().zip(lb.as_array().unwrap()) {
                assert!(
                    (pa["x"].as_f64().unwrap() + f64::from(cx) - pb["x"].as_f64().unwrap()).abs()
                        < 1e-8
                );
                assert!(
                    (pa["y"].as_f64().unwrap() + f64::from(cy) - pb["y"].as_f64().unwrap()).abs()
                        < 1e-8
                );
            }
        }
    }
    assert_eq!(report["geometry"]["config"]["perspective"], true);
    assert_eq!(report["geometry"]["config"]["system_shear"], false);
    assert_eq!(report["output_processing"]["binarization"], "Sauvola");
    assert!(report["profile"]["failure_stage"].is_null());
    for stage in [
        "grayscale_seconds",
        "binarization_seconds",
        "print_normalization_seconds",
        "print_antialias_seconds",
        "png_write_seconds",
        "perspective_seconds",
    ] {
        assert!(report["profile"][stage].as_f64().unwrap() > 0.);
    }
    assert_eq!(report["profile"]["native_detection_seconds"], 0.);
    let components: f64 = report["profile"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, _)| {
            k.ends_with("_seconds")
                && !["total_seconds", "unattributed_seconds"].contains(&k.as_str())
        })
        .map(|(_, v)| v.as_f64().unwrap())
        .sum();
    assert!(components <= report["profile"]["total_seconds"].as_f64().unwrap() + 1e-9);
    let disabled = root.join("disabled");
    assert!(
        call(&disabled, &["--no-perspective", "--no-render"])
            .status
            .success()
    );
    let report: Value =
        serde_json::from_slice(&fs::read(disabled.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["geometry"]["config"]["perspective"], false);
    assert!(report["output_processing"].is_null());
    assert_eq!(report["profile"]["render_seconds"], 0.);
    assert_eq!(report["profile"]["binarization_seconds"], 0.);
    assert!(!disabled.join("rectified-gray.png").exists());
    let detected = root.join("detected");
    assert!(call(&detected, &["--detect-only"]).status.success());
    let report: Value =
        serde_json::from_slice(&fs::read(detected.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "success");
    assert_eq!(report["profile"]["render_seconds"], 0.);
    let blank = root.join("blank.png");
    RgbImage::from_pixel(240, 200, Rgb([255; 3]))
        .save(&blank)
        .unwrap();
    let failed_out = root.join("failed");
    let failed = Command::new(env!("CARGO_BIN_EXE_smude-v2"))
        .arg("--image")
        .arg(blank)
        .arg("--out")
        .arg(&failed_out)
        .arg("--no-pages")
        .output()
        .unwrap();
    assert!(!failed.status.success());
    let report: Value =
        serde_json::from_slice(&fs::read(failed_out.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "failed");
    assert_eq!(report["profile"]["failure_stage"], "native_detection");
    assert!(
        report["profile"]["native_detection_seconds"]
            .as_f64()
            .unwrap()
            > 0.
    );
    fs::remove_dir_all(root).unwrap();
}
