//! Prepared PDF profiles: exact byte counts and reuse of the selected file.
use super::*;
use std::sync::{Arc, atomic::AtomicBool};

#[derive(Clone, Debug)]
pub struct PreparedPdf {
    pub path: PathBuf,
    pub bytes: u64,
    _cache: Arc<Cache>,
}
impl PartialEq for PreparedPdf {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path && self.bytes == other.bytes
    }
}
#[derive(Debug)]
struct Cache {
    path: PathBuf,
}
impl Drop for Cache {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
pub const PROFILES: [ExportSize; 3] = [
    ExportSize::Small,
    ExportSize::Standard,
    ExportSize::Original,
];
pub fn index(size: ExportSize) -> usize {
    match size {
        ExportSize::Small => 0,
        ExportSize::Standard => 1,
        ExportSize::Original => 2,
    }
}
pub fn prepare(
    pages: &[ExportInput],
    title: &str,
    directory: &Path,
    cancel: &Arc<AtomicBool>,
    mut ready: impl FnMut(usize, PreparedPdf),
) -> Result<(), String> {
    if pages.is_empty() {
        return Err("No pages selected".into());
    }
    fs::create_dir_all(directory).map_err(io_error)?;
    let cancelled = || cancel.load(Ordering::Relaxed);
    let cache = directory.join(format!("pdf-profiles-{}", crate::model::id()));
    fs::create_dir(&cache).map_err(io_error)?;
    let cache = Arc::new(Cache { path: cache });
    for (index, size) in PROFILES.into_iter().enumerate() {
        if cancelled() {
            return Ok(());
        }
        let quality = [75, 88, 95][index];
        let bytes = match pdf_with_jpeg(
            pages,
            ExportOptions {
                format: Format::Pdf,
                color: Color::Automatic,
                size,
            },
            title,
            Some(quality),
            &cancelled,
        ) {
            Err(_) if cancelled() => return Ok(()),
            result => result?,
        };
        if cancelled() {
            return Ok(());
        }
        let path = cache.path.join(format!("{index}.pdf"));
        write_new(&path, &bytes)?;
        ready(
            index,
            PreparedPdf {
                path,
                bytes: bytes.len() as u64,
                _cache: cache.clone(),
            },
        );
    }
    Ok(())
}
pub fn publish(
    prepared: &PreparedPdf,
    filename: &str,
    directory: &Path,
) -> Result<PathBuf, String> {
    let batch = Batch::new(directory)?;
    let filename = format!("{}.pdf", safe_filename(filename));
    let path = batch.staging.join(&filename);
    fs::copy(&prepared.path, &path).map_err(io_error)?;
    fs::File::open(&path)
        .map_err(io_error)?
        .sync_all()
        .map_err(io_error)?;
    Ok(batch.publish(&[filename])?.remove(0))
}
