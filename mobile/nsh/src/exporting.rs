//! Real document export from completed Smude output images.
//!
//! Image samples are prepared once per page and shared by the preview and export
//! paths. Automatic exports retain RGB covers and gray music; explicit gray or
//! 1-bit PDF options remain available. PDF samples use lossless Flate streams.
//! A batch becomes visible only
//! after every file has been written successfully.
pub mod sharing;
use crate::model::{Color, ExportOptions, ExportSize, Format};
use flate2::{Compression, write::ZlibEncoder};
use image::{GrayImage, ImageDecoder, ImageReader, imageops};
use pdf_writer::{Content, Filter, Finish, Name, Pdf, Rect, Ref, TextStr, types::Predictor};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug)]
pub struct ExportInput {
    /// A completed output image, never a placeholder or an in-flight job path.
    pub path: PathBuf,
    /// Clockwise display rotation, applied without geometric resampling.
    pub rotation: u16,
}

/// Export in the caller's order. The returned paths belong to one newly
/// published directory; existing exports and source files are never overwritten.
/// Keep this directory alive while a native share sheet can still read it.
pub fn export(
    pages: &[ExportInput],
    options: ExportOptions,
    filename: &str,
    directory: &Path,
) -> Result<Vec<PathBuf>, String> {
    if pages.is_empty() {
        return Err("Keine Seiten zum Exportieren ausgewählt.".into());
    }
    if pages.len() > 10_000 {
        return Err("Zu viele Seiten für einen Export.".into());
    }
    if options.format == Format::Jpeg && options.color == Color::Binary {
        return Err("Echtes Schwarz-Weiß ist als PDF verfügbar; JPEG verwendet Graustufen.".into());
    }
    fs::create_dir_all(directory).map_err(io_error)?;
    let batch = Batch::new(directory)?;
    let name = safe_filename(filename);
    let filenames = match options.format {
        Format::Pdf => {
            let filename = format!("{name}.pdf");
            let bytes = pdf(pages, options, &name)?;
            write_new(&batch.staging.join(&filename), &bytes)?;
            vec![filename]
        }
        Format::Jpeg => {
            let digits = pages.len().to_string().len().max(3);
            let mut filenames = Vec::with_capacity(pages.len());
            for (index, page) in pages.iter().enumerate() {
                let gray = prepare_raster(page, options)?;
                let filename = if pages.len() == 1 {
                    format!("{name}.jpg")
                } else {
                    format!("{name}-{:0digits$}.jpg", index + 1)
                };
                let mut file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(batch.staging.join(&filename))
                    .map_err(io_error)?;
                let quality = match options.size {
                    ExportSize::Small => 85,
                    ExportSize::Standard => 92,
                    ExportSize::Original => 97,
                };
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut file, quality)
                    .encode(
                        gray.as_bytes(),
                        gray.width(),
                        gray.height(),
                        gray.color().into(),
                    )
                    .map_err(|e| format!("JPEG konnte nicht erstellt werden: {e}"))?;
                file.sync_all().map_err(io_error)?;
                filenames.push(filename);
            }
            filenames
        }
    };
    batch.publish(&filenames)
}

/// The exact raster used by the export, also usable for an export preview.
/// Rotation preserves pixels; optional downscaling never enlarges the image.
/// Binarization happens after resizing, so a 1-bit PDF is actually binary.
fn load_and_frame(
    input: &ExportInput,
    options: ExportOptions,
) -> Result<image::DynamicImage, String> {
    if !input.rotation.is_multiple_of(90) {
        return Err("Die Seitendrehung muss ein Vielfaches von 90° sein.".into());
    }
    let reader = ImageReader::open(&input.path)
        .map_err(|e| format!("Ergebnisbild konnte nicht geöffnet werden: {e}"))?
        .with_guessed_format()
        .map_err(|e| format!("Bildformat konnte nicht gelesen werden: {e}"))?;
    let mut decoder = reader
        .into_decoder()
        .map_err(|e| format!("Ergebnisbild konnte nicht gelesen werden: {e}"))?;
    let orientation = decoder.orientation().map_err(|e| e.to_string())?;
    let mut image = image::DynamicImage::from_decoder(decoder)
        .map_err(|e| format!("Ergebnisbild konnte nicht gelesen werden: {e}"))?;
    image.apply_orientation(orientation);
    // Flatten alpha on white before either color preservation or grayscale.
    let image = if image.color().has_alpha() {
        let rgba = image.into_rgba8();
        image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(
            rgba.width(),
            rgba.height(),
            |x, y| {
                let p = rgba.get_pixel(x, y).0;
                image::Rgb(std::array::from_fn(|k| {
                    ((u32::from(p[k]) * u32::from(p[3]) + 255 * (255 - u32::from(p[3])) + 127)
                        / 255) as u8
                }))
            },
        ))
    } else if image.color().has_color() {
        image::DynamicImage::ImageRgb8(image.into_rgb8())
    } else {
        image::DynamicImage::ImageLuma8(image.into_luma8())
    };
    let gray = match input.rotation % 360 {
        90 => image.rotate90(),
        180 => image.rotate180(),
        270 => image.rotate270(),
        _ => image,
    };
    if gray.width() == 0 || gray.height() == 0 {
        return Err("Das Ergebnisbild ist leer.".into());
    }
    let limit = match options.size {
        ExportSize::Small => 1600,
        ExportSize::Standard => 2600,
        ExportSize::Original => gray.width().max(gray.height()),
    };
    let longest = gray.width().max(gray.height());
    let gray = if longest > limit {
        let width = ((u64::from(gray.width()) * u64::from(limit) + u64::from(longest) / 2)
            / u64::from(longest))
        .max(1) as u32;
        let height = ((u64::from(gray.height()) * u64::from(limit) + u64::from(longest) / 2)
            / u64::from(longest))
        .max(1) as u32;
        gray.resize_exact(width, height, imageops::FilterType::Lanczos3)
    } else {
        gray
    };
    Ok(gray)
}

/// Automatic exports preserve RGB covers and the single-channel music raster.
pub fn prepare_raster(
    input: &ExportInput,
    options: ExportOptions,
) -> Result<image::DynamicImage, String> {
    if options.color == Color::Automatic {
        load_and_frame(input, options)
    } else {
        prepare_image(input, options).map(image::DynamicImage::ImageLuma8)
    }
}

pub fn prepare_image(input: &ExportInput, options: ExportOptions) -> Result<GrayImage, String> {
    let gray = load_and_frame(input, options)?.into_luma8();
    Ok(match options.color {
        Color::Automatic | Color::Gray => gray,
        Color::Binary => {
            // These are already paper-white outputs, not raw camera frames.
            // A second local contrast threshold erases thin, partially covered
            // staff pixels. Keep those strokes while leaving the white paper
            // white; the loss of gray coverage is explicit in the BW option.
            GrayImage::from_fn(gray.width(), gray.height(), |x, y| {
                image::Luma([if gray.get_pixel(x, y)[0] <= 224 {
                    0
                } else {
                    255
                }])
            })
        }
    })
}

/// Only a pipeline-declared A4 frame assigns physical paper dimensions. A
/// coincidentally DIN-shaped image or color cover retains its observed canvas.
fn pdf_page_size(input: &ExportInput, width: u32, height: u32) -> [f32; 2] {
    let declared = (|| {
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(input.path.parent()?.join("report.json")).ok()?)
                .ok()?;
        let mm = report["final_frame"]["paper_size_mm"].as_array()?;
        let mut size = [mm.first()?.as_u64()? as u32, mm.get(1)?.as_u64()? as u32];
        if size != [210, 297] && size != [297, 210] {
            return None;
        }
        let original = image::image_dimensions(&input.path).ok()?;
        if report["output_size"][0].as_u64()? != u64::from(original.0)
            || report["output_size"][1].as_u64()? != u64::from(original.1)
        {
            return None;
        }
        if input.rotation % 180 == 90 {
            size.swap(0, 1);
        }
        let expected_height = f64::from(width) * f64::from(size[1]) / f64::from(size[0]);
        // A declared A4 raster rounds its aspect to individual pixels. The
        // relative-only guard incorrectly rejected valid small A4 images.
        if (f64::from(height) - expected_height).abs() > 1.5
            && (f64::from(height) / expected_height - 1.).abs() > 0.002
        {
            return None;
        }
        Some([size[0] as f32 * 72. / 25.4, size[1] as f32 * 72. / 25.4])
    })();
    declared.unwrap_or_else(|| {
        let scale = 842. / width.max(height) as f32;
        [width as f32 * scale, height as f32 * scale]
    })
}

fn pdf(pages: &[ExportInput], options: ExportOptions, title: &str) -> Result<Vec<u8>, String> {
    pdf_with_jpeg(pages, options, title, None, &|| false)
}
fn pdf_with_jpeg(
    pages: &[ExportInput],
    options: ExportOptions,
    title: &str,
    jpeg_quality: Option<u8>,
    cancelled: &impl Fn() -> bool,
) -> Result<Vec<u8>, String> {
    let mut pdf = Pdf::new();
    let catalog = Ref::new(1);
    let tree = Ref::new(2);
    pdf.catalog(catalog).pages(tree);
    pdf.document_info(Ref::new(3))
        .title(TextStr(title))
        .creator(TextStr("Smude"));
    let ids = (0..pages.len())
        .map(|i| Ref::new(4 + i as i32 * 3))
        .collect::<Vec<_>>();
    pdf.pages(tree)
        .kids(ids.iter().copied())
        .count(pages.len() as i32);
    for (input, page_id) in pages.iter().zip(ids) {
        if cancelled() {
            return Err("cancelled".into());
        }
        let gray = prepare_raster(input, options)?;
        let image_id = Ref::new(page_id.get() + 1);
        let content_id = Ref::new(page_id.get() + 2);
        let [width, height] = pdf_page_size(input, gray.width(), gray.height());
        let mut page = pdf.page(page_id);
        page.parent(tree)
            .media_box(Rect::new(0., 0., width, height))
            .contents(content_id);
        page.resources().x_objects().pair(Name(b"Page"), image_id);
        page.finish();
        let binary = options.color == Color::Binary;
        let bytes = if jpeg_quality.is_some() && !binary {
            Vec::new()
        } else if binary {
            pack_bits(gray.as_luma8().unwrap())
        } else {
            png_up(
                gray.as_bytes(),
                gray.width() as usize * gray.color().channel_count() as usize,
                gray.height() as usize,
            )
        };
        let jpeg = jpeg_quality.filter(|_| !binary);
        let compressed = if let Some(quality) = jpeg {
            let mut bytes = Vec::new();
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, quality)
                .encode(
                    gray.as_bytes(),
                    gray.width(),
                    gray.height(),
                    gray.color().into(),
                )
                .map_err(|e| e.to_string())?;
            bytes
        } else {
            compress(&bytes)?
        };
        let mut image = pdf.image_xobject(image_id, &compressed);
        image.filter(if jpeg.is_some() {
            Filter::DctDecode
        } else {
            Filter::FlateDecode
        });
        image
            .width(gray.width() as i32)
            .height(gray.height() as i32)
            .bits_per_component(if binary { 1 } else { 8 })
            .interpolate(!binary);
        let channels = if gray.color().has_color() { 3 } else { 1 };
        if channels == 3 {
            image.color_space().device_rgb();
        } else {
            image.color_space().device_gray();
        }
        if !binary && jpeg.is_none() {
            image
                .decode_parms()
                .predictor(Predictor::PngUp)
                .columns(gray.width() as i32)
                .colors(channels)
                .bits_per_component(8);
        }
        image.finish();
        let mut content = Content::new();
        content.save_state();
        content.transform([width, 0., 0., height, 0., 0.]);
        content.x_object(Name(b"Page"));
        content.restore_state();
        pdf.stream(content_id, &content.finish());
    }
    Ok(pdf.finish())
}

fn compress(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(bytes).map_err(io_error)?;
    encoder.finish().map_err(io_error)
}

/// PDF samples are MSB-first, padded independently on each row; 1 means white.
fn pack_bits(gray: &GrayImage) -> Vec<u8> {
    let stride = (gray.width() as usize).div_ceil(8);
    let mut packed = vec![255; stride * gray.height() as usize];
    for (y, row) in gray
        .as_raw()
        .chunks_exact(gray.width() as usize)
        .enumerate()
    {
        for (x, &sample) in row.iter().enumerate() {
            if sample == 0 {
                packed[y * stride + x / 8] &= !(0x80 >> (x % 8));
            }
        }
    }
    packed
}

/// PNG-Up prediction with one filter byte per row, as required by PDF /Predictor 12.
fn png_up(samples: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() + height);
    for (y, row) in samples.chunks_exact(width).enumerate() {
        bytes.push(2);
        for (x, &sample) in row.iter().enumerate() {
            let above = if y == 0 {
                0
            } else {
                samples[(y - 1) * width + x]
            };
            bytes.push(sample.wrapping_sub(above));
        }
    }
    bytes
}

fn safe_filename(value: &str) -> String {
    let mut name = value.trim().to_owned();
    for extension in [".pdf", ".jpeg", ".jpg"] {
        if name.to_ascii_lowercase().ends_with(extension) {
            name.truncate(name.len() - extension.len());
            break;
        }
    }
    let name = name
        .chars()
        .map(|c| {
            if c.is_control() || "\\/:*?\"<>|".contains(c) {
                ' '
            } else {
                c
            }
        })
        .collect::<String>();
    let name = name.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut shortened = String::new();
    for character in name.trim_matches([' ', '.']).chars() {
        if shortened.len() + character.len_utf8() > 180 {
            break;
        }
        shortened.push(character);
    }
    let name = shortened;
    if name.is_empty() {
        "Noten".into()
    } else {
        name
    }
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(io_error)?;
    file.write_all(bytes).map_err(io_error)?;
    file.sync_all().map_err(io_error)
}

fn io_error(error: std::io::Error) -> String {
    format!("Export konnte nicht gespeichert werden: {error}")
}

struct Batch {
    staging: PathBuf,
    published: PathBuf,
    complete: bool,
}
impl Batch {
    fn new(parent: &Path) -> Result<Self, String> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let key = format!(
            "{now:x}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let staging = parent.join(format!(".export-{key}.partial"));
        let published = parent.join(format!("export-{key}"));
        fs::create_dir(&staging).map_err(io_error)?;
        Ok(Self {
            staging,
            published,
            complete: false,
        })
    }
    fn publish(mut self, filenames: &[String]) -> Result<Vec<PathBuf>, String> {
        fs::rename(&self.staging, &self.published).map_err(io_error)?;
        self.complete = true;
        Ok(filenames
            .iter()
            .map(|name| self.published.join(name))
            .collect())
    }
}
impl Drop for Batch {
    fn drop(&mut self) {
        if !self.complete {
            let _ = fs::remove_dir_all(&self.staging);
        }
    }
}
