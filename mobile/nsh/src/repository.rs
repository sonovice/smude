//! Durable project metadata and immutable media. A manifest is published only
//! after its files exist; completed processing caches are independently atomic.
use crate::model::*;
use image::{ImageDecoder, ImageReader};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::{Component, Path, PathBuf},
};
#[derive(Clone)]
pub struct Repository {
    pub root: PathBuf,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rendered {
    #[serde(default)]
    pub quality: smude_geometry::detection::quality::CaptureQuality,
    pub side: usize,
    pub path: String,
    pub dimensions: [u32; 2],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderBatch {
    pub pages: Vec<Rendered>,
    pub warnings: Vec<String>,
}
#[derive(Clone, Debug)]
pub struct Job {
    pub project: u64,
    pub page: Page,
}
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
impl Repository {
    pub fn open(root: PathBuf) -> Result<Self, String> {
        fs::create_dir_all(&root).map_err(err)?;
        for d in ["sources", "renders", "exports", "captures"] {
            fs::create_dir_all(root.join(d)).map_err(err)?;
        }
        Ok(Self { root })
    }
    pub fn path(&self, path: &str) -> Result<PathBuf, String> {
        let p = Path::new(path);
        if p.is_absolute() || p.components().any(|c| !matches!(c, Component::Normal(_))) {
            return Err("Ungültiger Projektpfad".into());
        }
        Ok(self.root.join(p))
    }
    pub fn load(&self) -> Result<Library, String> {
        let p = self.root.join("library.json");
        if !p.exists() {
            return Ok(Library::default());
        }
        let parse = |p: &Path| -> Result<Library, String> {
            serde_json::from_slice(&fs::read(p).map_err(err)?).map_err(err)
        };
        let mut l = parse(&p)
            .or_else(|_| parse(&self.root.join("library.previous.json")))
            .map_err(|e| format!("Projekte konnten nicht geladen werden: {e}"))?;
        l.initialize_project_dates();
        // An atomic native JPEG may have completed just before the UI closed.
        // Unfinished requests cannot survive process termination; expose a retake.
        for import in &mut l.imports {
            if import.awaiting_capture && std::path::Path::new(&import.path).is_file() {
                import.awaiting_capture = false;
            }
        }
        let unfinished: Vec<_> = l
            .imports
            .iter()
            .filter(|i| i.awaiting_capture && i.capture_expired())
            .flat_map(|i| i.pages.clone())
            .collect();
        l.imports
            .retain(|i| !i.awaiting_capture || !i.capture_expired());
        for p in l
            .projects
            .iter_mut()
            .flat_map(|p| &mut p.pages)
            .filter(|p| unfinished.contains(&p.id))
        {
            if let Some(old) = p.replacement.take() {
                *p = *old;
            } else {
                p.status = PageStatus::Failed;
                p.error = Some("Capture interrupted".into());
            }
        }
        l.projects.retain(|p| !p.draft || !p.pages.is_empty());
        for p in l.projects.iter_mut().flat_map(|p| &mut p.pages) {
            if p.model != Model::Mesh {
                p.model = Model::Mesh;
                if p.status != PageStatus::Importing
                    && p.source
                        .as_deref()
                        .is_some_and(|s| self.path(s).is_ok_and(|s| s.is_file()))
                {
                    p.invalidate();
                }
            }
            if let Some(old) = p.replacement.as_mut() {
                old.model = Model::Mesh;
            }
            // Retain the previous preview while refreshing an old render.
            if p.status == PageStatus::Ready
                && p.output.as_deref().is_some_and(|path| {
                    path.starts_with("renders/page-v") && !path.starts_with("renders/page-v13-")
                })
                && p.source
                    .as_deref()
                    .is_some_and(|path| self.path(path).is_ok_and(|path| path.is_file()))
            {
                p.invalidate();
            }
            if p.status == PageStatus::Processing {
                p.status = PageStatus::Pending;
            }
            if let Some(path) = &p.output
                && !self.path(path)?.is_file()
            {
                p.output = None;
                p.status = PageStatus::Pending;
            }
        }
        Ok(l)
    }
    pub fn save(&self, l: &Library) -> Result<(), String> {
        let bytes = serde_json::to_vec(l).map_err(err)?;
        let path = self.root.join("library.json");
        if path.exists() {
            let old = fs::read(&path).map_err(err)?;
            if serde_json::from_slice::<Library>(&old).is_ok() {
                atomic(&self.root.join("library.previous.json"), &old)?;
            }
        }
        atomic(&path, &bytes)
    }
    pub fn import(&self, source: &Path) -> Result<String, String> {
        let reader = ImageReader::open(source)
            .map_err(err)?
            .with_guessed_format()
            .map_err(err)?;
        let format = reader.format().ok_or("Bildformat nicht erkannt")?;
        if !matches!(format, image::ImageFormat::Jpeg | image::ImageFormat::Png) {
            return Err("Bitte ein JPEG-, PNG- oder HEIC-Bild auswählen.".into());
        }
        let dims = reader.into_dimensions().map_err(err)?;
        if dims.0 < 2 || dims.1 < 2 || u64::from(dims.0) * u64::from(dims.1) > 120_000_000 {
            return Err("Dieses Bild ist zu groß oder ungültig.".into());
        }
        let relative = format!(
            "sources/{}.{}",
            id(),
            if format == image::ImageFormat::Jpeg {
                "jpg"
            } else {
                "png"
            }
        );
        let finalpath = self.path(&relative)?;
        let temp = finalpath.with_extension("tmp");
        fs::copy(source, &temp).map_err(err)?;
        fs::File::open(&temp)
            .map_err(err)?
            .sync_all()
            .map_err(err)?;
        fs::rename(temp, finalpath).map_err(err)?;
        Ok(relative)
    }
    pub fn process(&self, job: &Job) -> Result<RenderBatch, String> {
        self.process_with_progress(job, None)
    }
    pub fn process_with_progress(
        &self,
        job: &Job,
        progress: Option<smude_geometry::application::ProcessingObserver>,
    ) -> Result<RenderBatch, String> {
        use smude_geometry::application::{ProcessingEvent, ProcessingStage};
        let notify = |event| {
            if let Some(observer) = &progress {
                observer(event);
            }
        };
        notify(ProcessingEvent::Stage(ProcessingStage::Preparing));
        let page = &job.page;
        let source = self.path(page.source.as_deref().ok_or("Original fehlt")?)?;
        let folder = format!(
            "renders/page-v13-{}-{:?}-{}-{:?}",
            page.capture, page.model, page.input_rotation, page.layout
        );
        let directory = self.path(&folder)?;
        let manifest = directory.join("completed.json");
        if let Ok(bytes) = fs::read(&manifest)
            && let Ok(result) = serde_json::from_slice::<RenderBatch>(&bytes)
            && result
                .pages
                .iter()
                .all(|p| self.path(&p.path).is_ok_and(|p| p.is_file()))
        {
            notify(ProcessingEvent::Finished);
            return Ok(result);
        }
        fs::create_dir_all(&directory).map_err(err)?;
        let mut decoder = ImageReader::open(&source)
            .map_err(err)?
            .with_guessed_format()
            .map_err(err)?
            .into_decoder()
            .map_err(err)?;
        let orientation = decoder.orientation().map_err(err)?;
        let (width, height) = decoder.dimensions();
        notify(ProcessingEvent::InputPixels(
            u64::from(width) * u64::from(height),
        ));
        let input = if orientation == image::metadata::Orientation::NoTransforms
            && page.input_rotation == 0
        {
            source
        } else {
            let mut img = image::DynamicImage::from_decoder(decoder).map_err(err)?;
            img.apply_orientation(orientation);
            img = match page.input_rotation % 360 {
                90 => img.rotate90(),
                180 => img.rotate180(),
                270 => img.rotate270(),
                _ => img,
            };
            let input = directory.join("oriented.png");
            img.save(&input).map_err(err)?;
            input
        };
        let observer: Option<smude_geometry::application::ProcessingObserver> =
            progress.clone().map(|observer| {
                std::sync::Arc::new(move |event| {
                    if event != ProcessingEvent::Finished {
                        observer(event);
                    }
                }) as smude_geometry::application::ProcessingObserver
            });
        let result = smude_geometry::application::process_file_with_observers(
            &input,
            &directory,
            &smude_geometry::application::Settings {
                page_layout: page.layout,
                dewarp_model: match page.model {
                    Model::Gcs => smude_geometry::application::DewarpModel::Gcs,
                    Model::Mesh => smude_geometry::application::DewarpModel::Mesh,
                },
                ..Default::default()
            },
            None,
            observer,
        )
        .map_err(err)?;
        let mut pages = vec![];
        for (i, p) in result.pages.into_iter().enumerate() {
            let path = p.image(smude_geometry::application::OutputKind::Clean);
            let dimensions = image::image_dimensions(&path).map_err(err)?;
            pages.push(Rendered {
                quality: p.quality,
                side: p
                    .id
                    .parse::<usize>()
                    .ok()
                    .and_then(|v| v.checked_sub(1))
                    .unwrap_or(i),
                path: path
                    .strip_prefix(&self.root)
                    .map_err(err)?
                    .to_string_lossy()
                    .into_owned(),
                dimensions: [dimensions.0, dimensions.1],
            });
        }
        if pages.is_empty() {
            return Err("Keine vollständige Notenseite erkannt. Bitte das Original prüfen oder neu aufnehmen.".into());
        }
        pages.sort_by_key(|p| p.side);
        let batch = RenderBatch {
            pages,
            warnings: result.warnings,
        };
        atomic(&manifest, &serde_json::to_vec(&batch).map_err(err)?)?;
        notify(ProcessingEvent::Finished);
        Ok(batch)
    }
}
pub fn atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let temporary = path.with_extension(format!("{}.tmp", id()));
    let result = (|| {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(err)?;
        f.write_all(bytes).map_err(err)?;
        f.sync_all().map_err(err)?;
        fs::rename(&temporary, path).map_err(err)?;
        if let Some(parent) = path.parent()
            && let Ok(dir) = fs::File::open(parent)
        {
            let _ = dir.sync_all();
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}
/// EXIF fallback when a capture completed across an activity restart.
pub(crate) fn source_rotation(path: &Path) -> u16 {
    (|| {
        let reader = ImageReader::open(path).ok()?.with_guessed_format().ok()?;
        let mut decoder = reader.into_decoder().ok()?;
        use image::metadata::Orientation;
        Some(match decoder.orientation().ok()? {
            Orientation::Rotate90 | Orientation::Rotate90FlipH => 90,
            Orientation::Rotate180 => 180,
            Orientation::Rotate270 | Orientation::Rotate270FlipH => 270,
            _ => 0,
        })
    })()
    .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn durable_recovery_and_owned_media() {
        let root = std::env::temp_dir().join(format!("smude-store-{}", id()));
        let r = Repository::open(root.clone()).unwrap();
        let mut l = Library::default();
        l.projects.push(Project::new("Test".into()));
        r.save(&l).unwrap();
        l.projects[0].title = "New".into();
        r.save(&l).unwrap();
        fs::write(root.join("library.json"), "broken").unwrap();
        assert_eq!(r.load().unwrap().projects[0].title, "Test");
        assert!(r.path("../escape").is_err());
        let _ = fs::remove_dir_all(root);
    }
}
