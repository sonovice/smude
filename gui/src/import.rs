use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub struct Discovery {
    pub files: Vec<PathBuf>,
    pub warnings: Vec<String>,
}

pub fn supported(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
        ["jpg", "jpeg", "png"]
            .iter()
            .any(|s| e.eq_ignore_ascii_case(s))
    })
}

/// Recurse into folders, skip directory symlinks and deduplicate canonical paths.
pub fn discover(paths: &[PathBuf]) -> Discovery {
    let mut files = Vec::new();
    let mut warnings = Vec::new();
    for path in paths {
        if path.is_file() {
            if supported(path) {
                match path.canonicalize() {
                    Ok(path) => files.push(path),
                    Err(e) => warnings.push(format!("{}: {e}", path.display())),
                }
            } else {
                warnings.push(format!(
                    "{}: Nur JPEG und PNG werden unterstützt.",
                    path.display()
                ));
            }
        } else if path.is_dir() {
            for entry in WalkDir::new(path).follow_links(false).sort_by_file_name() {
                match entry {
                    Ok(entry) if entry.file_type().is_file() && supported(entry.path()) => {
                        match entry.path().canonicalize() {
                            Ok(path) => files.push(path),
                            Err(e) => warnings.push(e.to_string()),
                        }
                    }
                    Ok(_) => {}
                    Err(e) => warnings.push(e.to_string()),
                }
            }
        } else {
            warnings.push(format!("{} ist nicht verfügbar.", path.display()));
        }
    }
    files.sort();
    files.dedup();
    Discovery { files, warnings }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn folders_are_recursive_sorted_and_duplicates_are_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("inner")).unwrap();
        for p in ["b.JPG", "inner/a.png", "notes.txt"] {
            std::fs::write(tmp.path().join(p), b"x").unwrap();
        }
        let found = discover(&[tmp.path().to_owned(), tmp.path().join("b.JPG")]);
        assert_eq!(found.files.len(), 2);
        assert!(found.warnings.is_empty());
        assert!(found.files.iter().all(|p| supported(p)));
    }
}
