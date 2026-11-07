//! Export cached, full-resolution outputs without overwriting existing files.
use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
};

#[derive(Clone)]
pub struct ExportItem {
    pub source: PathBuf,
    pub stem: String,
    pub page: String,
}

pub fn export(items: &[ExportItem], directory: &Path) -> io::Result<Vec<PathBuf>> {
    fs::create_dir_all(directory)?;
    let mut written = Vec::new();
    for item in items {
        let stem: String = item
            .stem
            .chars()
            .map(|c| {
                if c.is_control() || "<>:\"/\\|?*".contains(c) {
                    '_'
                } else {
                    c
                }
            })
            .collect();
        let base = format!("{stem}-page-{}", item.page);
        let mut input = fs::File::open(&item.source)?;
        let mut destination = None;
        for suffix in 0..10_000 {
            let name = if suffix == 0 {
                format!("{base}.png")
            } else {
                format!("{base}-{suffix}.png")
            };
            let path = directory.join(name);
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => {
                    destination = Some((path, file));
                    break;
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(e),
            }
        }
        let (path, mut file) = destination.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::AlreadyExists,
                "Zu viele Dateien mit demselben Namen",
            )
        })?;
        if let Err(e) = io::copy(&mut input, &mut file).and_then(|_| file.flush()) {
            drop(file);
            let _ = fs::remove_file(path);
            return Err(e);
        }
        written.push(path);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exports_full_bytes_and_preserves_existing_files_and_duplicate_names() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source.png");
        fs::write(&source, b"full-resolution-bytes").unwrap();
        let out = tmp.path().join("export");
        fs::create_dir(&out).unwrap();
        fs::write(out.join("score-page-1.png"), b"existing").unwrap();
        let item = ExportItem {
            source,
            stem: "score".into(),
            page: "1".into(),
        };
        let paths = export(&[item.clone(), item], &out).unwrap();
        assert_eq!(fs::read(out.join("score-page-1.png")).unwrap(), b"existing");
        assert_ne!(paths[0], paths[1]);
        assert_eq!(fs::read(&paths[0]).unwrap(), b"full-resolution-bytes");
    }
}
