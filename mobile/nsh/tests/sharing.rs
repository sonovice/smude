//! Prepared-profile sizes describe the bytes actually shared, not estimates.
use image::{GrayImage, Luma};
use smude_mobile_ui::exporting::{ExportInput, sharing};
use std::{
    fs,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!("smude-sharing-{}", smude_mobile_ui::model::id()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn page(&self) -> ExportInput {
        let path = self.0.join("page.png");
        GrayImage::from_fn(2100, 2970, |x, y| {
            Luma([
                if (y % 90 < 2 && x > 70 && x < 2030) || (x / 3 + y / 5) % 73 == 0 {
                    0
                } else {
                    255
                },
            ])
        })
        .save(&path)
        .unwrap();
        fs::write(
            self.0.join("report.json"),
            br#"{"output_size":[2100,2970],"final_frame":{"paper_size_mm":[210,297]}}"#,
        )
        .unwrap();
        ExportInput { path, rotation: 90 }
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
#[test]
fn sizes_are_exact_jpeg_profiles_preserve_a4_and_publish_reuses_identical_bytes() {
    let temp = Scratch::new();
    let input = temp.page();
    let original = fs::read(&input.path).unwrap();
    let directory = temp.0.join("exports");
    let mut profiles = vec![];
    sharing::prepare(
        std::slice::from_ref(&input),
        "Étude",
        &directory,
        &Arc::new(AtomicBool::new(false)),
        |i, p| {
            assert_eq!(i, profiles.len());
            profiles.push(p);
        },
    )
    .unwrap();
    assert_eq!(profiles.len(), 3);
    let mut sizes = vec![];
    for profile in &profiles {
        let bytes = fs::read(&profile.path).unwrap();
        assert_eq!(profile.bytes, bytes.len() as u64);
        let pdf = String::from_utf8_lossy(&bytes);
        assert!(pdf.contains("/DCTDecode"));
        assert!(pdf.contains("/DeviceGray"));
        assert!(pdf.contains("/BitsPerComponent 8"));
        let bounds = pdf
            .split("/MediaBox [")
            .nth(1)
            .unwrap()
            .split(']')
            .next()
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        assert!((bounds[2] - 297. * 72. / 25.4).abs() < 0.001);
        assert!((bounds[3] - 210. * 72. / 25.4).abs() < 0.001);
        sizes.push(profile.bytes);
    }
    assert!(sizes[0] < sizes[1] && sizes[1] < sizes[2], "{sizes:?}");
    let published = sharing::publish(&profiles[1], "../New title.PDF", &directory).unwrap();
    assert_eq!(published.file_name().unwrap(), "New title.pdf");
    assert_eq!(
        fs::read(&published).unwrap(),
        fs::read(&profiles[1].path).unwrap()
    );
    let cache = profiles[0].path.parent().unwrap().to_owned();
    drop(profiles);
    assert!(!cache.exists());
    assert!(published.is_file());
    assert_eq!(fs::read(input.path).unwrap(), original);
}
#[test]
fn cancellation_keeps_ready_profiles_alive_until_released_and_cleans_the_rest() {
    let temp = Scratch::new();
    let cancel = Arc::new(AtomicBool::new(false));
    let mut ready = vec![];
    let directory = temp.0.join("exports");
    sharing::prepare(&[temp.page()], "Scans", &directory, &cancel, |_, p| {
        ready.push(p);
        cancel.store(true, Ordering::Relaxed);
    })
    .unwrap();
    assert_eq!(ready.len(), 1);
    let cache = ready[0].path.parent().unwrap().to_owned();
    assert!(ready[0].path.is_file());
    drop(ready);
    assert!(!cache.exists());
    assert_eq!(fs::read_dir(directory).unwrap().count(), 0);
}
#[test]
fn missing_page_does_not_publish_partial_pdf_profiles() {
    let temp = Scratch::new();
    let directory = temp.0.join("exports");
    let result = sharing::prepare(
        &[ExportInput {
            path: temp.0.join("missing.png"),
            rotation: 0,
        }],
        "Missing",
        &directory,
        &Arc::new(AtomicBool::new(false)),
        |_, _| panic!("No profile may be ready"),
    );
    assert!(result.is_err());
    assert_eq!(fs::read_dir(directory).unwrap().count(), 0);
}

#[test]
#[ignore = "independent Poppler decoder; requires pdfimages"]
fn poppler_decodes_prepared_gray_and_rgb_pages_at_each_profile_resolution() {
    use image::{ImageDecoder, Rgb, RgbImage};
    use std::process::Command;
    let temp = Scratch::new();
    let gray = temp.page();
    let path = temp.0.join("cover.png");
    RgbImage::from_pixel(120, 80, Rgb([200, 35, 70]))
        .save(&path)
        .unwrap();
    let mut profiles = vec![];
    sharing::prepare(
        &[gray, ExportInput { path, rotation: 0 }],
        "Mixed",
        &temp.0.join("exports"),
        &Arc::new(AtomicBool::new(false)),
        |_, p| profiles.push(p),
    )
    .unwrap();
    for (i, profile) in profiles.into_iter().enumerate() {
        let prefix = temp.0.join(format!("decoded-{i}"));
        let output = Command::new("pdfimages")
            .arg("-png")
            .arg(&profile.path)
            .arg(&prefix)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let mut outputs = fs::read_dir(&temp.0)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with(&format!("decoded-{i}-"))
            })
            .collect::<Vec<_>>();
        outputs.sort();
        assert_eq!(outputs.len(), 2);
        let decoder = image::ImageReader::open(&outputs[0])
            .unwrap()
            .into_decoder()
            .unwrap();
        let (w, h) = decoder.dimensions();
        assert_eq!(w, [1600, 2600, 2970][i]);
        assert_eq!(decoder.color_type(), image::ColorType::L8);
        assert!(w > h);
        let image = image::open(&outputs[1]).unwrap().into_rgb8();
        assert_eq!(image.dimensions(), (120, 80));
        assert!(image.get_pixel(60, 40)[0] > 190);
        assert!(image.get_pixel(60, 40)[1] < 50);
    }
}
