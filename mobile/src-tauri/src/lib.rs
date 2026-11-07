//! Thin mobile adapter. All detection, geometry and full-resolution output live
//! in the same smude-geometry crate used by the CLI and desktop application.
use image::{ImageDecoder, ImageReader};
use serde::Serialize;
use smude_geometry::application::{DetectionStage, OutputKind, Settings};
#[cfg(not(any(target_os = "android", target_os = "ios")))]
use std::io::Write;
use std::{
    fs,
    io::Cursor,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Instant,
};
use tauri::{Emitter, Manager};
#[cfg(not(target_os = "ios"))]
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_fs::{FilePath, FsExt};

struct Jobs {
    busy: Arc<AtomicBool>,
    next: AtomicU64,
}
impl Default for Jobs {
    fn default() -> Self {
        // URLs must remain unique across launches, including browser asset caches.
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        Self {
            busy: Arc::default(),
            next: AtomicU64::new(seed),
        }
    }
}
struct Busy(Arc<AtomicBool>);
impl Drop for Busy {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

#[derive(Serialize)]
struct Imported {
    path: String,
    name: String,
    width: u32,
    height: u32,
}
#[derive(Serialize)]
struct Page {
    id: String,
    path: String,
    width: u32,
    height: u32,
    profile: serde_json::Value,
}
#[derive(Serialize)]
struct ResultPages {
    seconds: f64,
    pages: Vec<Page>,
    warnings: Vec<String>,
}
#[derive(Clone, Serialize)]
struct Progress {
    label: &'static str,
    page: String,
}
#[derive(Serialize)]
struct Environment {
    platform: &'static str,
    architecture: &'static str,
    demo: Option<String>,
    auto_run: bool,
    benchmark_models: bool,
}
fn cache(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let root = app
        .path()
        .app_cache_dir()
        .map_err(|e| e.to_string())?
        .join("processing");
    fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    Ok(root)
}
fn owned_path(app: &tauri::AppHandle, path: &str) -> Result<PathBuf, String> {
    let path = fs::canonicalize(path).map_err(|e| e.to_string())?;
    if !path.starts_with(fs::canonicalize(cache(app)?).map_err(|e| e.to_string())?) {
        return Err("Bild gehört nicht zu dieser Sitzung.".into());
    }
    Ok(path)
}
#[tauri::command]
async fn environment(app: tauri::AppHandle) -> Result<Environment, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let demo = [app.path().app_data_dir(), app.path().document_dir()]
            .into_iter()
            .filter_map(Result::ok)
            .map(|p| p.join("test-input.jpg"))
            .find(|p| p.is_file())
            .map(|p| p.to_string_lossy().into_owned());
        let marker = app
            .path()
            .app_data_dir()
            .ok()
            .map(|p| p.join("run-test-once"));
        let auto_run = std::env::var("SMUDE_AUTORUN").as_deref() == Ok("1")
            || marker.as_ref().is_some_and(|p| p.is_file());
        if let Some(marker) = marker {
            let _ = fs::remove_file(marker);
        }
        let benchmark_marker = app
            .path()
            .app_data_dir()
            .ok()
            .map(|p| p.join("benchmark-models"));
        let benchmark_models = std::env::var("SMUDE_BENCH_MODELS").as_deref() == Ok("1")
            || benchmark_marker.as_ref().is_some_and(|p| p.is_file());
        if let Some(marker) = benchmark_marker {
            let _ = fs::remove_file(marker);
        }
        Environment {
            benchmark_models,
            auto_run,
            platform: std::env::consts::OS,
            architecture: std::env::consts::ARCH,
            demo,
        }
    })
    .await
    .map_err(|e| e.to_string())
}
#[tauri::command]
async fn import_image(app: tauri::AppHandle, path: String) -> Result<Imported, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let selected: FilePath = path
            .parse()
            .map_err(|e: std::convert::Infallible| e.to_string())?;
        let bytes = app.fs().read(selected).map_err(|e| e.to_string())?;
        let id = app.state::<Jobs>().next.fetch_add(1, Ordering::Relaxed);
        let directory = cache(&app)?;
        if is_heif(&bytes) {
            let source = directory.join(format!("heif-{id}.heic"));
            let destination = directory.join(format!("input-{id}.png"));
            fs::write(&source, bytes).map_err(|e| e.to_string())?;
            let decoded = tauri_plugin_smude_device::call(
                &app,
                "decodeImage",
                serde_json::json!({ "source": source, "destination": destination }),
            );
            let _ = fs::remove_file(source);
            decoded?;
            imported_file(destination)
        } else {
            prepare_import(&bytes, &directory, id)
        }
    })
    .await
    .map_err(|e| e.to_string())?
}
fn is_heif(bytes: &[u8]) -> bool {
    bytes.len() >= 12
        && &bytes[4..8] == b"ftyp"
        && bytes[8..bytes.len().min(64)].chunks_exact(4).any(|brand| {
            matches!(
                brand,
                b"heic" | b"heix" | b"hevc" | b"hevx" | b"mif1" | b"msf1"
            )
        })
}
fn imported_file(path: PathBuf) -> Result<Imported, String> {
    let (width, height) = image::image_dimensions(&path).map_err(|e| e.to_string())?;
    Ok(Imported {
        path: path.to_string_lossy().into_owned(),
        name: "Notenbild".into(),
        width,
        height,
    })
}
#[tauri::command]
async fn capture_photo(app: tauri::AppHandle) -> Result<Option<String>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let id = app.state::<Jobs>().next.fetch_add(1, Ordering::Relaxed);
        let extension = if cfg!(target_os = "android") {
            "jpg"
        } else {
            "png"
        };
        let destination = cache(&app)?.join(format!("camera-{id}.{extension}"));
        let result = tauri_plugin_smude_device::call(
            &app,
            "capturePhoto",
            serde_json::json!({"destination":destination}),
        )?;
        Ok(result
            .get("path")
            .and_then(|p| p.as_str())
            .map(str::to_owned))
    })
    .await
    .map_err(|e| e.to_string())?
}
fn prepare_import(bytes: &[u8], directory: &std::path::Path, id: u64) -> Result<Imported, String> {
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| e.to_string())?;
    let format = reader.format();
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let mut decoder = reader
        .into_decoder()
        .map_err(|_| "Bitte ein JPEG-, PNG- oder HEIC-Bild auswählen.".to_string())?;
    let orientation = decoder.orientation().map_err(|e| e.to_string())?;
    let (path, width, height) = if orientation == image::metadata::Orientation::NoTransforms {
        let (width, height) = decoder.dimensions();
        let ext = if format == Some(image::ImageFormat::Jpeg) {
            "jpg"
        } else {
            "png"
        };
        let target = directory.join(format!("input-{id}.{ext}"));
        // Ordinary imports need only a copy, not a full decode/re-encode.
        fs::write(&target, bytes).map_err(|e| e.to_string())?;
        (target, width, height)
    } else {
        let mut image = image::DynamicImage::from_decoder(decoder).map_err(|e| e.to_string())?;
        image.apply_orientation(orientation);
        let target = directory.join(format!("input-{id}.png"));
        image.save(&target).map_err(|e| e.to_string())?;
        (target, image.width(), image.height())
    };
    Ok(Imported {
        path: path.to_string_lossy().into_owned(),
        name: "Notenbild".into(),
        width,
        height,
    })
}
#[tauri::command]
async fn benchmark_models(app: tauri::AppHandle, path: String) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        let image = image::open(owned_path(&app, &path)?).map_err(|e| e.to_string())?.to_rgb8();
        let directory = cache(&app)?.join("model-benchmark");
        fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
        let mut reports = vec![];
        let mut reference: Option<smude_geometry::PageDetectionReport> = None;
        for accelerated in [false, true] {
            let started = Instant::now();
            let run_directory = directory.join(if accelerated { "accelerated" } else { "cpu" });
            fs::create_dir_all(&run_directory).map_err(|e| e.to_string())?;
            let detector = match smude_geometry::PageDetector::embedded_profiled(accelerated, &run_directory) {
                Ok(detector) => detector,
                Err(error) => { reports.push(serde_json::json!({"accelerated":accelerated,"error":error.to_string()})); continue; }
            };
            let load_seconds = started.elapsed().as_secs_f64();
            let mut runs = vec![];
            let mut max_boundary_difference = 0_f64;
            let mut max_ratio_difference = 0_f64;
            let mut same_class = true;
            let mut run_error = None;
            for _ in 0..4 {
                match detector.detect(&image, smude_geometry::PageLayoutMode::Auto) {
                    Ok((_, report)) => {
                        if let Some(reference) = &reference {
                            same_class &= report.selected_class == reference.selected_class;
                            if report.raw_boundaries_source.len() != reference.raw_boundaries_source.len() { max_boundary_difference = f64::INFINITY; }
                            for (actual, expected) in report.raw_boundaries_source.iter().zip(&reference.raw_boundaries_source) {
                                for (a,b) in actual.iter().zip(expected) {
                                    max_boundary_difference = max_boundary_difference.max((a.x-b.x).hypot(a.y-b.y));
                                }
                            }
                            for (a,b) in report.height_width_ratios.iter().zip(&reference.height_width_ratios) {
                                if let (Some(a),Some(b)) = (a,b) { max_ratio_difference = max_ratio_difference.max((a-b).abs()); }
                            }
                        } else { reference = Some(report.clone()); }
                        runs.push(serde_json::json!({"classification_seconds":report.classification_seconds,"boundary_seconds":report.boundary_inference_seconds}));
                    }
                    Err(error) => { run_error = Some(error.to_string()); break; }
                }
            }
            let profiles = detector.finish_profiling().map_err(|e| e.to_string())?;
            let mut placement = std::collections::BTreeMap::<String, usize>::new();
            for profile in &profiles {
                let events: serde_json::Value = serde_json::from_slice(&fs::read(profile).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
                if let Some(events) = events.as_array() {
                    for event in events {
                        if let Some(provider) = event.pointer("/args/provider").and_then(|v|v.as_str()) {
                            *placement.entry(provider.into()).or_default() += 1;
                        }
                    }
                }
            }
            reports.push(serde_json::json!({"accelerated":accelerated,"load_seconds":load_seconds,"runs":runs,
                "node_execution_counts":placement,"max_boundary_difference_pixels":max_boundary_difference,
                "max_ratio_difference":max_ratio_difference,"same_class":same_class,"error":run_error,
                "coordinate_parity_passed": run_error.is_none() && same_class && max_boundary_difference <= 0.5 && max_ratio_difference <= 0.001,
                "coreml_units": std::env::var("SMUDE_COREML_UNITS").unwrap_or_else(|_| "all".into())}));
        }
        fs::write(directory.join("report.json"), serde_json::to_vec_pretty(&reports).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        Ok(())
    }).await.map_err(|e|e.to_string())?
}
#[tauri::command]
async fn process_image(app: tauri::AppHandle, path: String) -> Result<ResultPages, String> {
    let jobs = app.state::<Jobs>();
    if jobs.busy.swap(true, Ordering::AcqRel) {
        return Err("Eine Seite wird bereits verarbeitet.".into());
    }
    let busy = Busy(jobs.busy.clone());
    let id = jobs.next.fetch_add(1, Ordering::Relaxed);
    tauri::async_runtime::spawn_blocking(move || {
        let _busy = busy;
        let started = Instant::now();
        let input = owned_path(&app, &path)?;
        let output = cache(&app)?.join(format!("job-{id}"));
        let observer_app = app.clone();
        let observer = Arc::new(
            move |event: smude_geometry::application::DetectionProgress| {
                let label = match event.stage {
                    DetectionStage::Page => "Seite erkannt",
                    DetectionStage::ResetMusic => "Notenlinien suchen",
                    DetectionStage::Staves => "Notenlinien erkannt",
                    DetectionStage::Barlines => "Taktstriche erkannt",
                    DetectionStage::StaffEdges => "Geometrie berechnen",
                    DetectionStage::Text => "Bild aufbereiten",
                };
                let _ = observer_app.emit(
                    "processing-stage",
                    Progress {
                        label,
                        page: event.page_id,
                    },
                );
            },
        );
        let threads = std::thread::available_parallelism()
            .map_or(2, |n| n.get().saturating_sub(2).clamp(1, 4));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| e.to_string())?;
        // Device QA can select a backend without adding product UI controls.
        let backend = std::env::var("SMUDE_RENDER_BACKEND").ok().or_else(|| {
            app.path()
                .app_data_dir()
                .ok()
                .and_then(|p| fs::read_to_string(p.join("render-backend")).ok())
        });
        let settings = Settings {
            render_backend: match backend.as_deref().map(str::trim) {
                Some("cpu") => smude_geometry::RenderBackend::Cpu,
                Some("gpu") => smude_geometry::RenderBackend::Gpu,
                Some("auto") | None => smude_geometry::RenderBackend::Auto,
                Some(value) => return Err(format!("Unbekannter Renderer: {value}")),
            },
            ..Settings::default()
        };
        let result = pool.install(|| {
            smude_geometry::application::process_file_with_progress(
                &input,
                &output,
                &settings,
                Some(observer),
            )
            .map_err(|e| e.to_string())
        })?;
        let mut pages = vec![];
        for page in result.pages {
            let image = page.image(OutputKind::Clean);
            let (width, height) = image::image_dimensions(&image).map_err(|e| e.to_string())?;
            let profile = fs::read(page.directory.join("profile.json"))
                .ok()
                .and_then(|b| serde_json::from_slice(&b).ok())
                .unwrap_or(serde_json::Value::Null);
            pages.push(Page {
                id: page.id,
                path: image.to_string_lossy().into_owned(),
                width,
                height,
                profile,
            });
        }
        let result = ResultPages {
            seconds: started.elapsed().as_secs_f64(),
            pages,
            warnings: result.warnings,
        };
        fs::write(
            output.join("mobile-report.json"),
            serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        Ok(result)
    })
    .await
    .map_err(|e| e.to_string())?
}
#[tauri::command]
async fn export_page(app: tauri::AppHandle, path: String) -> Result<bool, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let input = owned_path(&app, &path)?;
        #[cfg(target_os = "ios")]
        {
            let id = app.state::<Jobs>().next.fetch_add(1, Ordering::Relaxed);
            let staging = cache(&app)?.join(format!("export-{id}"));
            fs::create_dir_all(&staging).map_err(|e| e.to_string())?;
            let source = staging.join("Smude.png");
            fs::copy(&input, &source).map_err(|e| e.to_string())?;
            let result = tauri_plugin_smude_device::call(&app, "exportImage", serde_json::json!({
                "source":source, "directory":app.path().document_dir().map_err(|e| e.to_string())?,
                "verify":std::env::var("SMUDE_VERIFY_EXPORT").as_deref() == Ok("1")
            }));
            let _ = fs::remove_dir_all(staging);
            let value = result?;
            let saved = value.get("saved").and_then(|v| v.as_bool()).unwrap_or(false);
            if std::env::var("SMUDE_VERIFY_EXPORT").as_deref() == Ok("1") {
                fs::write(cache(&app)?.join("last-export.json"), serde_json::to_vec_pretty(&value).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
                if saved && value.get("matches_source").and_then(|v|v.as_bool()) != Some(true) {
                    return Err(format!("Exportprüfung fehlgeschlagen: {value}"));
                }
            }
            return Ok(saved);
        }
        #[cfg(not(target_os = "ios"))]
        {
        let Some(destination) = app
            .dialog()
            .file()
            .set_file_name("Smude.png")
            .add_filter("PNG", &["png"])
            .blocking_save_file()
        else {
            return Ok(false);
        };
        #[cfg(target_os = "android")]
        tauri_plugin_smude_device::call(&app,
                "writeExport",
                serde_json::json!({
                    "source":input, "destination":destination.to_string()
                }),
            )
            .map_err(|e| e.to_string())?;
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        {
            let mut options = tauri_plugin_fs::OpenOptions::new();
            options.write(true).create(true).truncate(true);
            let mut file = app
                .fs()
                .open(destination, options)
                .map_err(|e| e.to_string())?;
            file.write_all(&fs::read(input).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        }
        Ok(true)
        }
    })
    .await
    .map_err(|e| e.to_string())?
}
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(Jobs::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_smude_device::init())
        .invoke_handler(tauri::generate_handler![
            environment,
            import_image,
            capture_photo,
            benchmark_models,
            process_image,
            export_page
        ])
        .run(tauri::generate_context!())
        .expect("Smude mobile runtime");
}

#[cfg(test)]
mod tests {
    use super::*;
    fn directory() -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "smude-mobile-import-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
    fn jpeg() -> Vec<u8> {
        let image = image::RgbImage::from_fn(96, 64, |x, y| image::Rgb([x as u8, y as u8, 120]));
        let mut bytes = vec![];
        image::codecs::jpeg::JpegEncoder::new(&mut bytes)
            .encode_image(&image)
            .unwrap();
        bytes
    }
    #[test]
    fn heif_container_is_detected_by_brands_instead_of_filename() {
        assert!(is_heif(b"\0\0\0\x18ftypheic\0\0\0\0mif1heic"));
        assert!(is_heif(b"\0\0\0\x18ftypmif1\0\0\0\0heic"));
        assert!(!is_heif(b"\0\0\0\x10ftypmp42\0\0\0\0"));
        assert!(!is_heif(&jpeg()));
        assert!(!is_heif(b"ftypheic"));
    }
    #[test]
    fn ordinary_import_preserves_original_bytes_and_resolution() {
        let dir = directory();
        let bytes = jpeg();
        let result = prepare_import(&bytes, &dir, 1).unwrap();
        assert_eq!((result.width, result.height), (96, 64));
        assert_eq!(fs::read(&result.path).unwrap(), bytes);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn phone_exif_orientation_is_applied_before_geometry() {
        let dir = directory();
        let jpeg = jpeg();
        let exif = b"Exif\0\0II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0";
        let mut bytes = vec![0xff, 0xd8, 0xff, 0xe1];
        bytes.extend_from_slice(&((exif.len() + 2) as u16).to_be_bytes());
        bytes.extend_from_slice(exif);
        bytes.extend_from_slice(&jpeg[2..]);
        let result = prepare_import(&bytes, &dir, 2).unwrap();
        assert_eq!((result.width, result.height), (64, 96));
        assert_eq!(image::image_dimensions(result.path).unwrap(), (64, 96));
        fs::remove_dir_all(dir).unwrap();
    }
}
