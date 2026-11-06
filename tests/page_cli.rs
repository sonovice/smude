#![cfg(feature = "cli")]

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use image::{Rgb, RgbImage};
use serde_json::{Value, json};

static NEXT: AtomicUsize = AtomicUsize::new(0);

#[test]
fn retired_page_methods_are_not_accepted() {
    let output = Command::new(env!("CARGO_BIN_EXE_smude-v2"))
        .args([
            "--image",
            "unused.png",
            "--out",
            "unused",
            "--page-method",
            "legacy",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("--page-method"), "{error}");
    assert!(error.contains("unexpected argument"), "{error}");
}

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "smude-page-cli-{}-{nonce}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn packet(&self) -> PathBuf {
        let path = self.0.join("regions.json");
        fs::write(&path, serde_json::to_vec(&json!({
            "width": 1400, "height": 500,
            "regions": [
                {"id":"left", "contour":[{"x":0.,"y":0.},{"x":700.,"y":0.},{"x":700.,"y":500.},{"x":0.,"y":500.}], "confidence":1., "evidence":["test annotation"]},
                {"id":"right", "contour":[{"x":700.,"y":0.},{"x":1400.,"y":0.},{"x":1400.,"y":500.},{"x":700.,"y":500.}], "confidence":1., "evidence":["test annotation"]}
            ]
        })).unwrap()).unwrap();
        path
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn manual_pages_keep_successful_sibling_and_explicit_failed_page() {
    let fixture = Fixture::new();
    let mut image = RgbImage::from_pixel(1400, 500, Rgb([255; 3]));
    for center in [140_i32, 330] {
        for k in -2..=2 {
            for x in 60..640 {
                for dy in -1..=1 {
                    image.put_pixel(x, (center + 12 * k + dy) as u32, Rgb([0; 3]));
                }
            }
        }
    }
    let input = fixture.0.join("source.png");
    image.save(&input).unwrap();
    let regions = fixture.packet();
    let out = fixture.0.join("result");
    let status = Command::new(env!("CARGO_BIN_EXE_smude-v2"))
        .args([
            "--image",
            input.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--page-regions",
            regions.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let report: Value =
        serde_json::from_slice(&fs::read(out.join("pages_report.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "partial");
    assert_eq!(report["pages"][0]["status"], "success");
    assert_eq!(report["pages"][1]["status"], "failed");
    assert_eq!(
        report["pages"][1]["source_bounds"],
        json!([700, 0, 1400, 500])
    );
    assert!(out.join("page-left/rectified.png").exists());
    let page_report: Value =
        serde_json::from_slice(&fs::read(out.join("page-left/report.json")).unwrap()).unwrap();
    assert_eq!(page_report["final_crop_mode"], "page");
    assert_eq!(
        page_report["output_size"],
        page_report["uncropped_output_size"]
    );
    assert!(out.join("page-right/source.png").exists());
    assert!(out.join("page-right/error.log").exists());
    assert!(!out.join("rectified.png").exists());
    assert!(!out.join("page-right/rectified.png").exists());
    let observations: Value =
        serde_json::from_slice(&fs::read(out.join("page-left/observations.json")).unwrap())
            .unwrap();
    assert_eq!(observations["width"], 700);
    assert_eq!(observations["height"], 500);
    assert_eq!(observations["staves"].as_array().unwrap().len(), 2);
}

#[test]
fn all_failed_pages_leave_diagnostics_and_fail_the_command() {
    let fixture = Fixture::new();
    let input = fixture.0.join("blank.png");
    RgbImage::from_pixel(1400, 500, Rgb([255; 3]))
        .save(&input)
        .unwrap();
    let regions = fixture.packet();
    let out = fixture.0.join("result");
    let status = Command::new(env!("CARGO_BIN_EXE_smude-v2"))
        .args([
            "--image",
            input.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--page-regions",
            regions.to_str().unwrap(),
            "--detect-only",
        ])
        .output()
        .unwrap();
    assert!(!status.status.success());
    let report: Value =
        serde_json::from_slice(&fs::read(out.join("pages_report.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "failed");
    assert_eq!(report["pages"].as_array().unwrap().len(), 2);
    assert!(out.join("page-left/source.png").exists());
    assert!(out.join("page-right/source.png").exists());
}

#[cfg(feature = "onnx-pages")]
#[test]
fn default_uses_page_detection_and_no_pages_bypasses_models() {
    let fixture = Fixture::new();
    let image = fixture.0.join("blank.png");
    RgbImage::from_pixel(600, 800, Rgb([255; 3]))
        .save(&image)
        .unwrap();
    let missing_models = fixture.0.join("missing-models");
    let run = |name: &str, bypass: bool| {
        let output = fixture.0.join(name);
        let mut command = Command::new(env!("CARGO_BIN_EXE_smude-v2"));
        command
            .arg("--image")
            .arg(&image)
            .arg("--out")
            .arg(&output)
            .arg("--page-models")
            .arg(&missing_models);
        if bypass {
            command.arg("--no-pages");
        }
        let status = command.output().unwrap();
        assert!(!status.status.success());
        output
    };
    let default = run("default", false);
    let pages: Value =
        serde_json::from_slice(&fs::read(default.join("pages_report.json")).unwrap()).unwrap();
    assert_eq!(pages["status"], "failed");
    assert!(pages["error"].as_str().unwrap().contains("missing-models"));
    assert!(!default.join("report.json").exists());
    let bypass = run("bypass", true);
    let report: Value =
        serde_json::from_slice(&fs::read(bypass.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["profile"]["failure_stage"], "native_detection");
    assert!(!bypass.join("pages_report.json").exists());
}
