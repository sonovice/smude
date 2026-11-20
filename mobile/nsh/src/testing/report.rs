//! A page's original, its result and everything measured about both, as one
//! `reports` record.
use super::upload::Field;
use crate::{model::Page, repository::Repository};
use serde_json::{Value, json};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Context {
    pub install: String,
    pub app: Value,
    pub device: Value,
    pub language: String,
}

/// The record's fields without its files. `capture` is the sidecar the camera
/// wrote at the shutter (pose, orientation, detection); imports have none.
pub fn fields(
    page: &Page,
    position: (usize, usize),
    capture: Option<Value>,
    context: &Context,
) -> Vec<(&'static str, Field)> {
    let mut described = serde_json::to_value(page).unwrap_or(Value::Null);
    if let Some(page) = described.as_object_mut() {
        // The kept previous version is restorable state, not this page.
        page.remove("replacement");
        page.insert("position".into(), json!(position.0));
        page.insert("pages".into(), json!(position.1));
        page.insert("language".into(), json!(context.language));
    }
    vec![
        ("install", Field::Text(context.install.clone())),
        ("status", Field::Text(format!("{:?}", page.status).to_lowercase())),
        ("error", Field::Text(page.error.clone().unwrap_or_default())),
        ("app", Field::Json(context.app.clone())),
        ("device", Field::Json(context.device.clone())),
        ("page", Field::Json(described)),
        ("capture", Field::Json(capture.unwrap_or(Value::Null))),
    ]
}

/// Reads the files off the UI thread. The original is required; a missing
/// result (failed pages) or sidecar (imports, older captures) is not.
pub fn prepare(
    repo: &Repository,
    page: &Page,
    position: (usize, usize),
    context: &Context,
) -> Result<Vec<(&'static str, Field)>, String> {
    let source = repo.path(page.source.as_deref().ok_or("This page has no original.")?)?;
    let capture = std::fs::read(crate::repository::sidecar(&source))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok());
    let mut fields = fields(page, position, capture, context);
    fields.push(("original", file(&source)?));
    if let Some(output) = page.output.as_deref()
        && let Ok(output) = repo.path(output)
        && output.is_file()
    {
        fields.push(("result", file(&output)?));
    }
    Ok(fields)
}

fn file(path: &Path) -> Result<Field, String> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("image")
        .to_owned();
    let mime = match path.extension().and_then(|e| e.to_str()) {
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("png") => "image/png",
        _ => "application/octet-stream",
    };
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    Ok(Field::File { name, mime, bytes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::PageStatus;

    fn context() -> Context {
        Context {
            install: "abc".into(),
            app: json!({"version": "0.1.0", "build": "42"}),
            device: json!({"model": "iPhone16,1"}),
            language: "de".into(),
        }
    }
    fn page() -> Page {
        serde_json::from_value(json!({
            "id": 7, "capture": 3, "side": 0, "rotation": 90, "model": "Mesh",
            "review": false, "source": "sources/1.jpg", "output": null,
            "status": "Failed", "error": "Page not detected", "input_rotation": 270,
            "replacement": {"id": 7, "capture": 2, "side": 0, "rotation": 0,
                "model": "Mesh", "review": false, "source": "sources/0.jpg", "output": null}
        }))
        .unwrap()
    }
    fn json_field<'a>(fields: &'a [(&str, Field)], name: &str) -> &'a Value {
        match fields.iter().find(|(n, _)| *n == name) {
            Some((_, Field::Json(v))) => v,
            _ => panic!("missing json field {name}"),
        }
    }
    fn text_field<'a>(fields: &'a [(&str, Field)], name: &str) -> &'a str {
        match fields.iter().find(|(n, _)| *n == name) {
            Some((_, Field::Text(v))) => v,
            _ => panic!("missing text field {name}"),
        }
    }

    #[test]
    fn a_failed_page_reports_its_error_geometry_and_position_without_the_kept_version() {
        let page = page();
        assert_eq!(page.status, PageStatus::Failed);
        let fields = fields(&page, (2, 5), Some(json!({"gravity": [0, -1, 0]})), &context());
        assert_eq!(text_field(&fields, "status"), "failed");
        assert_eq!(text_field(&fields, "error"), "Page not detected");
        assert_eq!(text_field(&fields, "install"), "abc");
        let described = json_field(&fields, "page");
        assert_eq!(described["input_rotation"], 270);
        assert_eq!(described["rotation"], 90);
        assert_eq!(described["position"], 2);
        assert_eq!(described["pages"], 5);
        assert!(described.get("replacement").is_none());
        assert_eq!(json_field(&fields, "capture")["gravity"][1], -1);
        assert_eq!(json_field(&fields, "device")["model"], "iPhone16,1");
    }

    #[test]
    fn preparing_attaches_the_original_and_its_capture_sidecar() {
        let root = std::env::temp_dir().join(format!("smude-report-{}", crate::model::id()));
        let repo = Repository::open(root.clone()).unwrap();
        std::fs::write(root.join("sources/1.jpg"), b"jpeg").unwrap();
        std::fs::write(root.join("sources/1.jpg.json"), br#"{"device_orientation":1}"#).unwrap();
        let fields = prepare(&repo, &page(), (1, 1), &context()).unwrap();
        assert_eq!(json_field(&fields, "capture")["device_orientation"], 1);
        assert!(fields.iter().any(|(n, f)| *n == "original"
            && matches!(f, Field::File { mime: "image/jpeg", bytes, .. } if bytes == b"jpeg")));
        assert!(!fields.iter().any(|(n, _)| *n == "result"));
        std::fs::remove_dir_all(root).unwrap();
    }
}
