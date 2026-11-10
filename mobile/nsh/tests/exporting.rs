//! Export regressions exercise actual files. Poppler independently decodes the
//! PDF test images and renders the real score fixture in the ignored QA test.
use image::{GrayImage, ImageDecoder, Luma};
use smude_mobile_ui::{
    exporting::{ExportInput, export, prepare_image},
    model::{Color, ExportOptions, ExportSize, Format},
};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "smude-export-test-{}-{}",
            std::process::id(),
            smude_mobile_ui::model::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn image(&self, name: &str, w: u32, h: u32) -> PathBuf {
        let path = self.0.join(name);
        GrayImage::from_fn(w, h, |x, y| {
            Luma([if (x / 7 + y / 11) % 5 == 0 {
                0
            } else if x < w / 2 {
                180
            } else {
                255
            }])
        })
        .save(&path)
        .unwrap();
        path
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn input(path: PathBuf, rotation: u16) -> ExportInput {
    ExportInput { path, rotation }
}
fn options(format: Format, color: Color, size: ExportSize) -> ExportOptions {
    ExportOptions {
        format,
        color,
        size,
    }
}

#[test]
fn grayscale_jpeg_is_single_channel_with_rotation_resolution_and_safe_name() {
    let temp = Scratch::new();
    let source = temp.image("input.png", 3000, 1000);
    let before = fs::read(&source).unwrap();
    let paths = export(
        &[input(source.clone(), 90)],
        options(Format::Jpeg, Color::Gray, ExportSize::Small),
        "../Probe: Walzer.pdf",
        &temp.0.join("exports"),
    )
    .unwrap();
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].file_name().unwrap(), "Probe Walzer.jpg");
    let reader = image::ImageReader::open(&paths[0])
        .unwrap()
        .into_decoder()
        .unwrap();
    assert_eq!(reader.color_type(), image::ColorType::L8);
    assert_eq!(reader.dimensions(), (533, 1600));
    assert_eq!(
        fs::read(source).unwrap(),
        before,
        "Export must not modify the source"
    );
}

#[test]
fn smaller_images_are_not_upscaled_and_batch_filenames_follow_input_order() {
    let temp = Scratch::new();
    let a = temp.image("a.png", 81, 123);
    let b = temp.image("b.png", 47, 29);
    let paths = export(
        &[input(b, 0), input(a, 270)],
        options(Format::Jpeg, Color::Gray, ExportSize::Standard),
        "Noten",
        &temp.0.join("exports"),
    )
    .unwrap();
    assert_eq!(
        paths
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect::<Vec<_>>(),
        ["Noten-001.jpg", "Noten-002.jpg"]
    );
    assert_eq!(image::image_dimensions(&paths[0]).unwrap(), (47, 29));
    assert_eq!(image::image_dimensions(&paths[1]).unwrap(), (123, 81));
    assert_eq!(paths[0].parent(), paths[1].parent());
}

#[test]
fn failed_batch_never_publishes_a_partial_export_or_overwrites_an_earlier_one() {
    let temp = Scratch::new();
    let source = temp.image("a.png", 83, 117);
    let dir = temp.0.join("exports");
    let opt = options(Format::Jpeg, Color::Gray, ExportSize::Original);
    let first = export(&[input(source.clone(), 0)], opt, "Probe", &dir).unwrap();
    let bytes = fs::read(&first[0]).unwrap();
    assert!(
        export(
            &[input(source, 0), input(temp.0.join("missing.png"), 0)],
            opt,
            "Probe",
            &dir
        )
        .is_err()
    );
    assert_eq!(
        fs::read_dir(&dir).unwrap().count(),
        1,
        "No partial directory may remain"
    );
    assert_eq!(fs::read(&first[0]).unwrap(), bytes);
}

#[test]
fn binary_pdf_is_true_one_bit_and_gray_pdf_is_single_channel_lossless() {
    let temp = Scratch::new();
    let source = temp.image("a.png", 81, 123);
    for color in [Color::Gray, Color::Binary] {
        let paths = export(
            &[input(source.clone(), 0)],
            options(Format::Pdf, color, ExportSize::Original),
            "Étude",
            &temp.0.join("exports"),
        )
        .unwrap();
        let bytes = fs::read(&paths[0]).unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(bytes.starts_with(b"%PDF-"));
        assert!(text.contains("/DeviceGray"));
        assert!(text.contains("/FlateDecode"));
        assert!(text.contains(if color == Color::Binary {
            "/BitsPerComponent 1"
        } else {
            "/BitsPerComponent 8"
        }));
        assert!(!text.contains("/DeviceRGB"));
        assert!(!text.contains("/JBIG2Decode"));
        assert!(!text.contains("/DCTDecode"));
    }
}

#[test]
fn export_rejects_empty_invalid_rotation_and_binary_jpeg() {
    let temp = Scratch::new();
    let path = temp.image("a.png", 13, 9);
    let dir = temp.0.join("exports");
    let opt = ExportOptions::default();
    assert!(export(&[], opt, "name", &dir).is_err());
    assert!(export(&[input(path.clone(), 45)], opt, "name", &dir).is_err());
    assert!(
        export(
            &[input(path, 0)],
            options(Format::Jpeg, Color::Binary, ExportSize::Small),
            "name",
            &dir
        )
        .is_err()
    );
    assert_eq!(fs::read_dir(dir).unwrap().count(), 0);
}

#[test]
#[ignore = "independent PDF decoder and rendering QA; requires Poppler"]
fn poppler_decodes_lossless_samples_page_order_rotation_and_odd_width_bit_rows() {
    let temp = Scratch::new();
    let a = temp.image("a.png", 83, 117);
    let b = temp.image("b.png", 49, 31);
    let pages = [input(a, 0), input(b, 90)];
    for color in [Color::Gray, Color::Binary] {
        let opt = options(Format::Pdf, color, ExportSize::Original);
        let pdf = export(&pages, opt, "Prüfung", &temp.0.join("exports"))
            .unwrap()
            .remove(0);
        let prefix = pdf.parent().unwrap().join("decoded");
        command("pdfimages", &[Path::new("-png"), &pdf, &prefix]);
        for (i, page) in pages.iter().enumerate() {
            let decoded = image::open(prefix.with_file_name(format!("decoded-{i:03}.png")))
                .unwrap()
                .into_luma8();
            assert_eq!(
                decoded,
                prepare_image(page, opt).unwrap(),
                "Poppler sample roundtrip page {i}, {color:?}"
            );
        }
    }
}

#[test]
#[ignore = "writes real score PDF and PNG QA artifacts under /tmp; requires Poppler"]
fn render_score_exports_for_visual_qa() {
    let directory = PathBuf::from("/tmp/smude-export-qa");
    fs::create_dir_all(&directory).unwrap();
    let source = std::env::var_os("SMUDE_EXPORT_QA_IMAGE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/score.jpg"));
    let pages = [input(source.clone(), 0), input(source, 90)];
    for color in [Color::Gray, Color::Binary] {
        let name = if color == Color::Binary {
            "score-binary"
        } else {
            "score-gray"
        };
        let pdf = export(
            &pages,
            options(Format::Pdf, color, ExportSize::Original),
            name,
            &directory,
        )
        .unwrap()
        .remove(0);
        let saved = directory.join(format!("{name}.pdf"));
        fs::copy(&pdf, &saved).unwrap();
        command(
            "pdftoppm",
            &[
                Path::new("-scale-to"),
                Path::new("1400"),
                Path::new("-png"),
                &saved,
                &directory.join(name),
            ],
        );
        command("pdfinfo", &[&saved]);
    }
}
fn command(program: &str, arguments: &[&Path]) {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .expect("Poppler command must be installed for this explicitly requested QA test");
    assert!(
        output.status.success(),
        "{}: {}",
        program,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn binary_export_preserves_faint_staff_pixels_in_paper_white_output() {
    let temp = Scratch::new();
    let path = temp.0.join("staff.png");
    let mut image = GrayImage::from_pixel(101, 59, Luma([255]));
    for y in [11, 19, 27, 35, 43] {
        for x in 7..94 {
            image.put_pixel(x, y, Luma([210]));
        }
    }
    image.save(&path).unwrap();
    let result = prepare_image(
        &input(path, 0),
        options(Format::Pdf, Color::Binary, ExportSize::Original),
    )
    .unwrap();
    for y in [11, 19, 27, 35, 43] {
        for x in 7..94 {
            assert_eq!(result.get_pixel(x, y)[0], 0);
        }
    }
    assert_eq!(result.get_pixel(0, 0)[0], 255);
    assert!(result.as_raw().iter().all(|v| *v == 0 || *v == 255));
}

#[test]
fn automatic_export_keeps_color_covers_and_gray_music_in_one_pdf() {
    let scratch = Scratch::new();
    let cover = scratch.0.join("cover.png");
    image::RgbImage::from_pixel(64, 96, image::Rgb([45, 90, 160]))
        .save(&cover)
        .unwrap();
    let music = scratch.image("music.png", 65, 97);
    let pages = vec![input(cover.clone(), 90), input(music, 0)];
    let automatic = options(Format::Pdf, Color::Automatic, ExportSize::Original);
    let raster = smude_mobile_ui::exporting::prepare_raster(&pages[0], automatic).unwrap();
    assert_eq!((raster.width(), raster.height()), (96, 64));
    assert_eq!(raster.to_rgb8().get_pixel(10, 10).0, [45, 90, 160]);
    let pdf = export(&pages, automatic, "mixed", &scratch.0).unwrap();
    let bytes = fs::read(&pdf[0]).unwrap();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("/DeviceRGB"));
    assert!(text.contains("/DeviceGray"));
    assert!(text.contains("/Colors 3"));
    let jpeg = export(
        &pages[..1],
        options(Format::Jpeg, Color::Automatic, ExportSize::Original),
        "cover",
        &scratch.0,
    )
    .unwrap();
    let decoder = image::ImageReader::open(&jpeg[0])
        .unwrap()
        .into_decoder()
        .unwrap();
    assert_eq!(decoder.color_type(), image::ColorType::Rgb8);
    let p = image::open(&jpeg[0]).unwrap().to_rgb8().get_pixel(10, 10).0;
    assert!(p[2] > p[0] + 80);
    let gray = export(
        &pages[..1],
        options(Format::Jpeg, Color::Gray, ExportSize::Original),
        "gray",
        &scratch.0,
    )
    .unwrap();
    assert_eq!(
        image::ImageReader::open(&gray[0])
            .unwrap()
            .into_decoder()
            .unwrap()
            .color_type(),
        image::ColorType::L8
    );
    if Command::new("pdfimages").arg("-v").output().is_ok() {
        let prefix = scratch.0.join("decoded");
        assert!(
            Command::new("pdfimages")
                .arg("-png")
                .arg(&pdf[0])
                .arg(&prefix)
                .status()
                .unwrap()
                .success()
        );
        let decoded = image::open(scratch.0.join("decoded-000.png"))
            .unwrap()
            .to_rgb8();
        assert_eq!(decoded.get_pixel(10, 10).0, [45, 90, 160]);
    }
}

#[test]
fn pipeline_a4_metadata_sets_physical_pdf_size_across_rotation_and_export_resolution() {
    let temp = Scratch::new();
    let path = temp.image("rectified-clean.png", 2100, 2970);
    fs::write(
        temp.0.join("report.json"),
        serde_json::json!({
            "output_size":[2100,2970],
            "final_frame":{"paper_size_mm":[210,297]}
        })
        .to_string(),
    )
    .unwrap();
    for rotation in [0, 90] {
        for size in [ExportSize::Original, ExportSize::Small] {
            let files = export(
                &[input(path.clone(), rotation)],
                options(Format::Pdf, Color::Gray, size),
                "A4",
                &temp.0.join(format!("{rotation}-{size:?}")),
            )
            .unwrap();
            let bytes = fs::read(&files[0]).unwrap();
            let text = String::from_utf8_lossy(&bytes);
            let bounds: Vec<f32> = text
                .split("/MediaBox [")
                .nth(1)
                .unwrap()
                .split(']')
                .next()
                .unwrap()
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            let expected = if rotation == 0 {
                [210., 297.]
            } else {
                [297., 210.]
            };
            assert!((bounds[2] - expected[0] * 72. / 25.4).abs() < 0.001);
            assert!((bounds[3] - expected[1] * 72. / 25.4).abs() < 0.001);
        }
    }
    // Shape alone does not authorize declaring an arbitrary old image A4.
    fs::write(temp.0.join("report.json"), "{}").unwrap();
    let files = export(
        &[input(path, 0)],
        options(Format::Pdf, Color::Gray, ExportSize::Small),
        "original-format",
        &temp.0.join("no-declaration"),
    )
    .unwrap();
    let bytes = fs::read(&files[0]).unwrap();
    let text = String::from_utf8_lossy(&bytes);
    let bounds: Vec<f32> = text
        .split("/MediaBox [")
        .nth(1)
        .unwrap()
        .split(']')
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(bounds[3], 842.);
}

#[test]
fn pixel_rounded_small_a4_keeps_physical_paper_size_in_pdf() {
    let temp = Scratch::new();
    let source = temp.image("tiny-a4.png", 61, 86);
    fs::write(
        temp.0.join("report.json"),
        br#"{"output_size":[61,86],"final_frame":{"paper_size_mm":[210,297]}}"#,
    )
    .unwrap();
    let files = export(
        &[input(source, 0)],
        options(Format::Pdf, Color::Gray, ExportSize::Original),
        "Tiny A4",
        &temp.0.join("exports"),
    )
    .unwrap();
    let bytes = fs::read(&files[0]).unwrap();
    let pdf = String::from_utf8_lossy(&bytes);
    let bounds = pdf
        .split("/MediaBox [")
        .nth(1)
        .unwrap()
        .split(']')
        .next()
        .unwrap()
        .split_whitespace()
        .map(|n| n.parse::<f32>().unwrap())
        .collect::<Vec<_>>();
    assert!((bounds[2] - 210. * 72. / 25.4).abs() < 0.001);
    assert!((bounds[3] - 297. * 72. / 25.4).abs() < 0.001);
}
