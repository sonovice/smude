use serde::{Deserialize, Serialize};
mod deletion;
mod preview;
mod projects;
pub use deletion::Deletion;
pub use preview::{PagePreview, PreviewKind};
pub use smude_geometry::application::PageLayout;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Model {
    Gcs,
    #[default]
    Mesh,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Page {
    #[serde(default)]
    pub source_rotation: u16,
    #[serde(default)]
    pub quality: smude_geometry::detection::quality::CaptureQuality,
    #[serde(default)]
    pub status: PageStatus,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub discover: bool,
    #[serde(default)]
    pub dimensions: [u32; 2],
    #[serde(default)]
    pub replacement: Option<Box<Page>>,
    pub id: u64,
    pub capture: u64,
    pub side: usize,
    pub rotation: u16,
    #[serde(default)]
    pub input_rotation: u16,
    #[serde(default)]
    pub layout: PageLayout,
    pub model: Model,
    pub review: bool,
    pub source: Option<String>,
    pub output: Option<String>,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Project {
    /// A persistent camera batch that has not been assigned to a collection.
    #[serde(default)]
    pub draft: bool,
    pub id: u64,
    pub title: String,
    pub pages: Vec<Page>,
    /// Last explicit user edit, Unix milliseconds. Background rendering does
    /// not advance this. Zero denotes a library created before this field.
    #[serde(default)]
    pub modified_at: u64,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Library {
    #[serde(default)]
    pub imports: Vec<PendingImport>,
    pub projects: Vec<Project>,
}
pub fn id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(100);
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;
    allocate_id(&NEXT, time)
}
fn allocate_id(next: &AtomicU64, time: u64) -> u64 {
    let previous = next
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |previous| {
            Some(time.max(previous + 1))
        })
        .unwrap();
    time.max(previous + 1)
}
impl Project {
    pub fn new(title: String) -> Self {
        Self {
            draft: false,
            id: id(),
            title,
            pages: vec![],
            modified_at: projects::now_millis(),
        }
    }
    pub fn capture(&mut self, sides: usize, source: Option<String>, replace: Option<u64>) {
        let capture = id();
        let pages = (0..sides)
            .map(|side| Page {
                source_rotation: 0,
                quality: Default::default(),
                status: PageStatus::Pending,
                error: None,
                revision: 0,
                discover: false,
                dimensions: [0, 0],
                replacement: None,
                id: id(),
                capture,
                side,
                rotation: 0,
                input_rotation: 0,
                layout: PageLayout::Auto,
                model: Model::default(),
                review: false,
                source: source.clone(),
                output: None,
            })
            .collect::<Vec<_>>();
        if let Some(replace) = replace {
            if let (Some(old), Some(new)) = (
                self.pages.iter_mut().find(|p| p.id == replace),
                pages.first(),
            ) {
                *old = Page {
                    id: old.id,
                    model: old.model,
                    ..new.clone()
                };
            }
        } else {
            self.pages.extend(pages);
        }
    }
    pub fn ordered_selection(&self, ids: &[u64]) -> Vec<Page> {
        self.pages
            .iter()
            .filter(|p| ids.contains(&p.id))
            .cloned()
            .collect()
    }
    pub fn move_page_to(&mut self, id: u64, to: usize) {
        if let Some(from) = self.pages.iter().position(|p| p.id == id) {
            let to = to.min(self.pages.len() - 1);
            let page = self.pages.remove(from);
            self.pages.insert(to, page);
        }
    }
    pub fn move_page(&mut self, id: u64, delta: isize) {
        if let Some(from) = self.pages.iter().position(|p| p.id == id) {
            let to = from as isize + delta;
            if to >= 0 && to < self.pages.len() as isize {
                self.pages.swap(from, to as usize);
            }
        }
    }
}
impl Library {
    pub fn examples() -> Self {
        let mut p = Project::new("Klavier · Allegro moderato".into());
        p.id = 1;
        p.capture(4, None, None);
        for (i, page) in p.pages.iter_mut().enumerate() {
            page.id = i as u64 + 1;
            page.status = PageStatus::Ready;
            page.source = Some("fixture-original.jpg".into());
            page.output = Some("fixture-result.png".into());
            page.review = i == 3;
        }
        Self {
            projects: vec![p],
            imports: vec![],
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Format {
    #[default]
    Pdf,
    Jpeg,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Color {
    #[default]
    Automatic,
    Gray,
    Binary,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportSize {
    Small,
    Standard,
    #[default]
    Original,
}
impl ExportSize {
    pub fn label(self) -> &'static str {
        match self {
            Self::Small => "Klein",
            Self::Standard => "Standard",
            Self::Original => "Original",
        }
    }
    pub fn detail(self) -> &'static str {
        match self {
            Self::Small => "Lange Kante: maximal 1.600 px",
            Self::Standard => "Lange Kante: maximal 2.600 px",
            Self::Original => "Volle Ergebnisauflösung",
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportOptions {
    pub format: Format,
    pub color: Color,
    pub size: ExportSize,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn new_captures_default_to_mesh_and_preserve_saved_gcs_choices() {
        assert_eq!(Model::default(), Model::Mesh);
        let mut project = Project::new("Defaults".into());
        project.capture(1, None, None);
        assert_eq!(project.pages[0].model, Model::Mesh);
        project.pages[0].model = Model::Gcs;
        let json = serde_json::to_string(&project).unwrap();
        let mut restored: Project = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.pages[0].model, Model::Gcs);
        let id = restored.pages[0].id;
        restored.capture(1, None, Some(id));
        assert_eq!(restored.pages[0].model, Model::Gcs);
    }

    #[test]
    fn retake_preserves_identity_and_sibling() {
        let mut p = Project::new("Test".into());
        p.capture(2, None, None);
        let first = p.pages[0].id;
        let second = p.pages[1].clone();
        p.capture(1, Some("replacement".into()), Some(first));
        assert_eq!(p.pages.len(), 2);
        assert_eq!(p.pages[0].id, first);
        assert_eq!(p.pages[1], second);
    }
    #[test]
    fn selection_uses_current_document_order() {
        let mut p = Project::new("Test".into());
        p.capture(3, None, None);
        let a = p.pages[0].id;
        let b = p.pages[2].id;
        assert_eq!(p.ordered_selection(&[b, a])[0].id, a);
        p.move_page(a, 1);
        assert_eq!(p.pages[1].id, a);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PageStatus {
    Importing,
    #[default]
    Pending,
    Processing,
    Ready,
    Failed,
}
impl Page {
    /// UIKit/ImageIO applies EXIF in its image loader; Android BitmapFactory does not.
    pub fn preview_rotation(&self, original: bool) -> f32 {
        let raw = original || self.output.is_none();
        (self.rotation
            + if raw {
                self.input_rotation
                    + if cfg!(target_os = "android") {
                        self.source_rotation
                    } else {
                        0
                    }
            } else {
                0
            }) as f32
    }
    pub fn invalidate(&mut self) {
        self.revision += 1;
        self.status = PageStatus::Pending;
        self.error = None;
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PendingImport {
    #[serde(default)]
    pub pages: Vec<u64>,
    #[serde(default)]
    pub awaiting_capture: bool,
    pub id: u64,
    pub project: u64,
    pub path: String,
    pub replace: Option<u64>,
    #[serde(default)]
    pub layout: PageLayout,
    #[serde(default)]
    pub input_rotation: u16,
}

#[cfg(test)]
mod id_tests {
    use super::*;
    #[test]
    fn delayed_threads_and_backward_clock_values_cannot_reuse_an_id() {
        let next = AtomicU64::new(100);
        assert_eq!(allocate_id(&next, 1000), 1000);
        assert_eq!(allocate_id(&next, 998), 1001);
        assert_eq!(allocate_id(&next, 999), 1002);
        let values = std::thread::scope(|s| {
            let workers: Vec<_> = (0..8)
                .map(|_| {
                    let next = &next;
                    s.spawn(move || {
                        (0..100)
                            .map(|_| allocate_id(next, 1000))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            workers
                .into_iter()
                .flat_map(|w| w.join().unwrap())
                .collect::<Vec<_>>()
        });
        let unique: std::collections::HashSet<_> = values.iter().collect();
        assert_eq!(unique.len(), values.len());
    }
}

impl PendingImport {
    pub(crate) fn capture_expired(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros();
        now.saturating_sub(u128::from(self.id)) > 120_000_000
    }
}
