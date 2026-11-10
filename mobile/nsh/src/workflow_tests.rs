//! State transitions run inside a real NSH owner. The opt-in pipeline test uses
//! owned copies of the photograph and keeps inspectable artifacts under /tmp.
use crate::{
    exporting::{self, ExportInput},
    model::*,
    repository::{Job, RenderBatch, Rendered, Repository},
    state::State,
};
use nsh_core::{prelude::*, storage::Storage};
use nsh_engine::{EngineConfig, Root};
use nsh_testing::{Headless, MemoryStorage};
use std::{fs, path::PathBuf, time::Instant};

const CASE: &str = "workflow-test-case";
const RESULT: &str = "workflow-test-result";

#[component]
fn ProcessingHarness() -> impl IntoElement {
    use crate::processing::ProcessingDisplay;
    use smude_geometry::application::ProcessingStage;
    let s = State::new();
    let mut project = Project::new("Fortschritt".into());
    project.capture(1, None, None);
    let page = signal(Some(project.pages.remove(0)));
    view()
        .child(crate::ui::processing_bar(s, page, false))
        .child(
            crate::ui::action("Fortschritt setzen", false, move || {
                let p = page.get_untracked().unwrap();
                s.processing.set(Some(ProcessingDisplay {
                    page_id: p.id,
                    capture: p.capture,
                    revision: p.revision,
                    percent: 42.,
                    stage: ProcessingStage::Rendering,
                    page_index: 1,
                    page_count: 2,
                }));
            })
            .test_id("advance"),
        )
        .child(
            crate::ui::action("Seite ersetzen", false, move || {
                page.update(|p| p.as_mut().unwrap().revision += 1);
            })
            .test_id("replace"),
        )
}

#[test]
fn progress_uses_nsh_range_semantics_and_ignores_replaced_page_events() {
    use nsh_core::{AccessibilityRole, MountItem};
    use std::sync::{Arc, Mutex};
    let mut config = EngineConfig::new(Root::of::<ProcessingHarness>());
    config.environment.locale = nsh_core::platform::Locale::parse("de-DE").unwrap();
    config.storage = Some(MemoryStorage::new([]));
    let mut app = Headless::start(config).unwrap();
    app.wait_for("text:Vorbereiten").unwrap();
    let values = Arc::new(Mutex::new(vec![]));
    let seen = values.clone();
    app.observe(move |_, batch| {
        for item in &batch.items {
            if let MountItem::ViewProps { props, .. } = item
                && props.accessibility.role == AccessibilityRole::Progressbar
            {
                seen.lock().unwrap().push(props.accessibility.value.clone());
            }
        }
    });
    app.press("id:advance").unwrap();
    app.wait_for("text:Seite 2 von 2 · Bild erstellen").unwrap();
    app.wait_for("text:42 %").unwrap();
    let range = values.lock().unwrap().last().cloned().flatten().unwrap();
    assert_eq!(
        (range.min, range.max, range.now),
        (Some(0.), Some(100.), Some(42.))
    );
    app.press("id:replace").unwrap();
    app.wait_for("text:Vorbereiten").unwrap();
    assert!(!app.screen().matches("text:42 %").unwrap());
    assert_eq!(
        values
            .lock()
            .unwrap()
            .last()
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap()
            .now,
        None
    );
}

struct Fixture {
    state: State,
    repo: Repository,
    project: u64,
}
impl Fixture {
    fn new() -> Self {
        let repo = Repository::open(PathBuf::from("/tmp").join(format!("smude-workflow-{}", id())))
            .unwrap();
        let state = State::new();
        state.root.set(Some(repo.root.clone()));
        let mut project = Project::new("Quartett".into());
        project.capture(3, Some("sources/owned.jpg".into()), None);
        project.pages[1].status = PageStatus::Processing;
        let project_id = project.id;
        assert!(state.save(Library {
            projects: vec![project],
            imports: vec![]
        }));
        Self {
            state,
            repo,
            project: project_id,
        }
    }
    fn pages(&self) -> Vec<Page> {
        self.state.project(self.project).pages
    }
    fn job(&self) -> Job {
        Job {
            project: self.project,
            page: self.pages()[1].clone(),
        }
    }
    fn batch(&self) -> RenderBatch {
        let pages = (0..2)
            .map(|side| {
                let relative = format!("renders/side-{side}.png");
                image::GrayImage::new(31 + side as u32, 47)
                    .save(self.repo.path(&relative).unwrap())
                    .unwrap();
                Rendered {
                    quality: Default::default(),
                    side,
                    path: relative,
                    dimensions: [31 + side as u32, 47],
                }
            })
            .collect();
        RenderBatch {
            pages,
            warnings: vec![],
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.repo.root);
    }
}

fn run_scenario(name: &str) {
    let f = Fixture::new();
    match name {
        "capture-completes-after-leaving" => {
            let initial = f.pages().len();
            f.state.push(crate::state::Route::Camera(f.project, None));
            let a = f.state.begin_capture(f.project, None, 1, 0).unwrap();
            let b = f.state.begin_capture(f.project, None, 2, 0).unwrap();
            assert_eq!(f.pages().len(), initial + 3);
            let ids: Vec<_> = f.pages().iter().map(|p| p.id).collect();
            f.state.back();
            for request in [&b, &a] {
                let id = request["request_id"].as_u64().unwrap();
                f.state.capture_event(
                    &serde_json::json!({"event":"capture_started","request_id":id,"rotation":90}),
                );
                image::RgbImage::new(11, 17)
                    .save(request["path"].as_str().unwrap())
                    .unwrap();
                f.state
                    .capture_event(&serde_json::json!({"event":"capture_saved","request_id":id,"source_rotation":90}));
                let import = f
                    .state
                    .library
                    .get_untracked()
                    .imports
                    .into_iter()
                    .find(|i| i.id == id)
                    .unwrap();
                assert!(!import.awaiting_capture);
                assert!(
                    f.pages()
                        .iter()
                        .filter(|p| import.pages.contains(&p.id))
                        .all(|p| p
                            .source
                            .as_ref()
                            .is_some_and(|s| s.starts_with("captures/"))
                            && p.source_rotation == 90)
                );
                let result = f.repo.import(std::path::Path::new(&import.path));
                f.state.finish_import(import, result);
            }
            assert!(f.state.library.get_untracked().imports.is_empty());
            let pages = f.pages();
            assert_eq!(pages.iter().map(|p| p.id).collect::<Vec<_>>(), ids);
            assert!(pages[initial..].iter().all(|p| p.source.is_some()
                && p.status == PageStatus::Pending
                && p.input_rotation == 90));
            assert_eq!(pages[initial + 1].source, pages[initial + 2].source);
            f.state.capture_event(
                &serde_json::json!({"event":"capture_saved","request_id":a["request_id"]}),
            );
            assert_eq!(f.pages(), pages);
        }
        "pending-delete-undo" => {
            let request = f.state.begin_capture(f.project, None, 1, 0).unwrap();
            let page = f.pages().last().unwrap().id;
            f.state.delete(f.project, &[page]);
            assert!(f.state.library.get_untracked().imports.is_empty());
            image::RgbImage::new(11, 17)
                .save(request["path"].as_str().unwrap())
                .unwrap();
            f.state.capture_event(
                &serde_json::json!({"event":"capture_saved","request_id":request["request_id"]}),
            );
            f.state.restore();
            assert!(f.pages().iter().any(|p| p.id == page));
            let recovered = f.repo.load().unwrap();
            assert_eq!(recovered.imports.len(), 1);
            assert!(!recovered.imports[0].awaiting_capture);
            f.state.library.set(recovered.clone());
            let import = recovered.imports[0].clone();
            let result = f.repo.import(std::path::Path::new(&import.path));
            f.state.finish_import(import, result);
            assert!(
                f.pages()
                    .iter()
                    .find(|p| p.id == page)
                    .unwrap()
                    .source
                    .is_some()
            );
        }
        "capture-failure-retake" => {
            let old = f.pages()[0].clone();
            let request = f
                .state
                .begin_capture(f.project, Some(old.id), 1, 0)
                .unwrap();
            assert_eq!(f.pages()[0].status, PageStatus::Importing);
            f.state.capture_event(
                &serde_json::json!({"event":"capture_failed","request_id":request["request_id"]}),
            );
            assert_eq!(f.pages()[0], old);
            assert!(f.state.library.get_untracked().imports.is_empty());
        }
        "capture-count-snapshot" => {
            let n = f.pages().len();
            let request = f.state.begin_capture(f.project, None, 1, 90).unwrap();
            assert_eq!(f.pages().last().unwrap().input_rotation, 90);
            f.state.capture_event(&serde_json::json!({"event":"capture_started","request_id":request["request_id"],"pages":2,"rotation":270}));
            assert_eq!(f.pages().len(), n + 2);
            assert!(
                f.pages()[n..]
                    .iter()
                    .all(|p| p.layout == PageLayout::Double && p.input_rotation == 270)
            );
            assert_eq!(f.state.library.get_untracked().imports[0].pages.len(), 2);
        }
        "rotate-reserved" => {
            f.state.reserve_file(
                f.project,
                "/tmp/photo.jpg".into(),
                None,
                PageLayout::Single,
                90,
            );
            let p = f.pages().last().unwrap().id;
            f.state.page(f.project, p, |p| p.rotation = 90);
            let import = f.state.library.get_untracked().imports[0].clone();
            f.state
                .finish_import(import, Ok("sources/photo.jpg".into()));
            let page = f.pages().into_iter().find(|v| v.id == p).unwrap();
            assert_eq!(page.input_rotation, 180);
            assert_eq!(page.rotation, 0);
            assert_eq!(page.status, PageStatus::Pending);
        }
        "raw-exif" => {
            let path = f.repo.path("captures/raw.jpg").unwrap();
            image::RgbImage::new(11, 17).save(&path).unwrap();
            let jpeg = fs::read(&path).unwrap();
            let mut payload =
                b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0".to_vec();
            let mut bytes = jpeg[..2].to_vec();
            bytes.extend([255, 225]);
            bytes.extend(((payload.len() + 2) as u16).to_be_bytes());
            bytes.append(&mut payload);
            bytes.extend_from_slice(&jpeg[2..]);
            fs::write(&path, &bytes).unwrap();
            assert_eq!(crate::repository::source_rotation(&path), 90);
            let source = f.repo.import(&path).unwrap();
            assert_eq!(
                fs::read(f.repo.path(&source).unwrap()).unwrap(),
                bytes,
                "camera JPEG remains byte-identical including EXIF"
            );
            assert_eq!(
                crate::repository::source_rotation(&f.repo.path(&source).unwrap()),
                90
            );
        }
        "immediate-pages" => {
            let before = f.pages().len();
            f.state.queue_oriented_capture(
                f.project,
                vec!["/tmp/new-spread.jpg".into()],
                None,
                PageLayout::Double,
                90,
            );
            assert_eq!(
                f.pages().len(),
                before + 2,
                "both photographed sides must be visible before import or rendering"
            );
            let pages = f.pages();
            assert_eq!(pages[before].capture, pages[before + 1].capture);
            assert_eq!((pages[before].side, pages[before + 1].side), (0, 1));
            assert_eq!(pages[before].input_rotation, 90);
        }
        "quick-capture-merge" => {
            use crate::state::Route;
            f.state.quick_capture();
            let Route::Camera(draft, None) = f.state.routes.get_untracked().last().unwrap().clone()
            else {
                panic!("Camera must open directly");
            };
            assert!(f.state.project(draft).draft);
            f.state.queue_capture(
                draft,
                vec!["/tmp/importing.jpg".into()],
                None,
                PageLayout::Single,
            );
            let import = f.state.library.get_untracked().imports[0].clone();
            let imported_page = import.pages[0];
            f.state.update(draft, |p| {
                p.capture(1, Some("sources/owned.jpg".into()), None)
            });
            let rendering = f.state.project(draft).pages[1].clone();
            let job = Job {
                project: draft,
                page: rendering.clone(),
            };
            let previous = f.pages().len();
            f.state.finish_camera(draft);
            assert_eq!(
                f.state.routes.get_untracked().last(),
                Some(&Route::Project(draft))
            );
            f.state.assign_draft(draft, f.project);
            assert_eq!(
                f.state.routes.get_untracked().last(),
                Some(&Route::Project(f.project))
            );
            assert!(
                !f.state
                    .library
                    .get_untracked()
                    .projects
                    .iter()
                    .any(|p| p.id == draft)
            );
            assert_eq!(
                f.state.library.get_untracked().imports[0].project,
                f.project
            );
            f.state.finish_import(import, Ok("sources/new.jpg".into()));
            assert!(f.state.library.get_untracked().imports.is_empty());
            assert_eq!(
                f.pages()
                    .iter()
                    .find(|p| p.id == imported_page)
                    .unwrap()
                    .source
                    .as_deref(),
                Some("sources/new.jpg")
            );
            f.state.finish_job(job, Ok(f.batch()));
            let rendered = f
                .pages()
                .into_iter()
                .find(|p| p.id == rendering.id)
                .unwrap();
            assert_eq!(rendered.status, PageStatus::Ready);
            assert_eq!(f.pages().len(), previous + 2);
            assert_eq!(f.repo.load().unwrap().projects[0].pages.len(), previous + 2);
            f.state.routes.set(vec![Route::Projects]);
            f.state.quick_capture();
            let Route::Camera(empty, None) = f.state.routes.get_untracked().last().unwrap().clone()
            else {
                panic!()
            };
            f.state.finish_camera(empty);
            assert_eq!(f.state.routes.get_untracked(), vec![Route::Projects]);
            assert!(
                !f.state
                    .library
                    .get_untracked()
                    .projects
                    .iter()
                    .any(|p| p.id == empty)
            );
        }
        "mesh-migration" => {
            let source = "sources/legacy.jpg";
            fs::write(f.repo.path(source).unwrap(), b"immutable").unwrap();
            let mut library = f.state.library.get_untracked();
            let page = &mut library.projects[0].pages[0];
            page.model = Model::Gcs;
            page.source = Some(source.into());
            page.status = PageStatus::Ready;
            page.output = Some("renders/page-v8-old/rectified-clean.png".into());
            let preview = f.repo.path(page.output.as_ref().unwrap()).unwrap();
            fs::create_dir_all(preview.parent().unwrap()).unwrap();
            fs::write(preview, b"retained preview").unwrap();
            page.rotation = 90;
            let before = page.clone();
            let modified = library.projects[0].last_modified();
            f.repo.save(&library).unwrap();
            let loaded = f.repo.load().unwrap();
            let page = &loaded.projects[0].pages[0];
            assert_eq!(page.model, Model::Mesh);
            assert_eq!(page.status, PageStatus::Pending);
            assert_eq!(page.source, before.source);
            assert_eq!(page.output, before.output);
            assert_eq!(page.rotation, 90);
            assert_eq!(page.revision, before.revision + 1);
            assert_eq!(loaded.projects[0].last_modified(), modified);
            f.repo.save(&loaded).unwrap();
            assert_eq!(f.repo.load().unwrap(), loaded);
        }
        "render-migration" => {
            let source = "sources/original.jpg";
            fs::write(f.repo.path(source).unwrap(), b"immutable source").unwrap();
            let mut l = f.state.library.get_untracked();
            let project = &mut l.projects[0];
            let edited = project.last_modified();
            for (i, page) in project.pages.iter_mut().enumerate() {
                let version = if i == 0 { "v12" } else { "v13" };
                let path = format!("renders/page-{version}-fixture-{i}/rectified-clean.png");
                let full = f.repo.path(&path).unwrap();
                fs::create_dir_all(full.parent().unwrap()).unwrap();
                fs::write(full, b"retained preview").unwrap();
                page.status = PageStatus::Ready;
                page.output = Some(path);
                page.source = Some(source.into());
                page.rotation = 90;
                page.model = Model::Mesh;
            }
            f.repo.save(&l).unwrap();
            let loaded = f.repo.load().unwrap();
            let p = &loaded.projects[0];
            assert_eq!(p.last_modified(), edited);
            assert_eq!(p.pages[0].status, PageStatus::Pending);
            assert_eq!(p.pages[0].revision, l.projects[0].pages[0].revision + 1);
            assert_eq!(p.pages[0].output, l.projects[0].pages[0].output);
            assert_eq!(p.pages[0].rotation, 90);
            assert_eq!(p.pages[0].model, Model::Mesh);
            assert_eq!(p.pages[1], l.projects[0].pages[1]);
            f.repo.save(&loaded).unwrap();
            assert_eq!(f.repo.load().unwrap(), loaded);
        }
        "capture-orientation" => {
            f.state.queue_oriented_capture(
                f.project,
                vec!["/tmp/photo.jpg".into()],
                None,
                smude_geometry::application::PageLayout::Double,
                90,
            );
            let durable = f.repo.load().unwrap();
            assert_eq!(durable.imports[0].input_rotation, 90);
            assert_eq!(
                durable.imports[0].layout,
                smude_geometry::application::PageLayout::Double
            );
            let legacy: PendingImport = serde_json::from_str(
                r#"{"id":1,"project":1,"path":"x","replace":null,"layout":"Single"}"#,
            )
            .unwrap();
            assert_eq!(legacy.input_rotation, 0);
        }
        "stale" => {
            let job = f.job();
            f.state.page(f.project, job.page.id, |p| p.rotation = 90);
            let expected = f.state.library.get_untracked();
            assert_eq!(f.pages()[1].revision, job.page.revision + 1);
            f.state.finish_job(job.clone(), Ok(f.batch()));
            assert_eq!(f.state.library.get_untracked(), expected);
            f.state.finish_job(job, Err("late failure".into()));
            assert_eq!(f.state.library.get_untracked(), expected);
            assert_eq!(f.repo.load().unwrap(), expected);
        }
        "deleted" => {
            let job = f.job();
            f.state.delete(f.project, &[job.page.id]);
            let expected = f.state.library.get_untracked();
            f.state.finish_job(job.clone(), Ok(f.batch()));
            assert_eq!(f.state.library.get_untracked(), expected);
            f.state.delete_project(f.project);
            f.state.finish_job(job, Ok(f.batch()));
            assert!(f.state.library.get_untracked().projects.is_empty());
            assert!(f.repo.load().unwrap().projects.is_empty());
        }
        "discovery" => {
            f.state.page(f.project, f.job().page.id, |p| {
                p.discover = true;
                p.layout = PageLayout::Double;
            });
            let before = f.pages();
            let edited_at = f.state.project(f.project).modified_at;
            let batch = f.batch();
            f.state.finish_job(f.job(), Ok(batch.clone()));
            let after = f.pages();
            assert_eq!(after.len(), 4);
            assert_eq!(after[0], before[0]);
            assert_eq!(after[3], before[2]);
            assert_eq!(after[1].id, before[1].id);
            assert_ne!(after[2].id, after[1].id);
            for (i, page) in after[1..3].iter().enumerate() {
                assert_eq!(page.capture, before[1].capture);
                assert_eq!(page.side, i);
                assert_eq!(page.status, PageStatus::Ready);
                assert_eq!(page.layout, PageLayout::Double);
                assert!(!page.discover);
                assert_eq!(page.output.as_ref(), Some(&batch.pages[i].path));
                assert_eq!(page.dimensions, batch.pages[i].dimensions);
            }
            assert_eq!(f.repo.load().unwrap().projects[0].pages, after);
            assert_eq!(
                f.state.project(f.project).modified_at,
                edited_at,
                "background discovery must not change project recency"
            );
        }
        "reprocess-child" => {
            f.state
                .page(f.project, f.job().page.id, |p| p.discover = true);
            f.state.finish_job(f.job(), Ok(f.batch()));
            let before = f.pages();
            let child = before[2].id;
            f.state.page(f.project, child, |p| p.model = Model::Mesh);
            let job = Job {
                project: f.project,
                page: f.pages()[2].clone(),
            };
            f.state.finish_job(job.clone(), Ok(f.batch()));
            let after = f.pages();
            assert_eq!(after.len(), 4);
            assert_eq!(after[0], before[0]);
            assert_eq!(after[1], before[1]);
            assert_eq!(after[3], before[3]);
            assert_eq!(after[2].id, child);
            assert_eq!(after[2].side, 1);
            assert_eq!(after[2].model, Model::Mesh);
            assert_eq!(after[2].output.as_deref(), Some("renders/side-1.png"));
            // A repeated completion must never append another copy of its sibling.
            f.state.finish_job(job, Ok(f.batch()));
            assert_eq!(f.pages(), after);
        }
        "retake-failure" | "retake-missing-side" => {
            f.state.finish_job(f.job(), Ok(f.batch()));
            let original = f.pages()[1].clone();
            f.state.page(f.project, original.id, |p| {
                p.replacement = Some(Box::new(original.clone()));
                p.source = Some("sources/replacement.jpg".into());
                p.capture = id();
                p.output = None;
                p.invalidate();
                p.status = PageStatus::Processing;
            });
            let job = f.job();
            let before = f.pages();
            let result = if name == "retake-missing-side" {
                let mut batch = f.batch();
                batch.pages.retain(|p| p.side != job.page.side);
                Ok(batch)
            } else {
                Err("Keine Seite erkannt".into())
            };
            f.state.finish_job(job, result);
            let after = f.pages();
            assert_eq!(after[1], original);
            assert_eq!(after[0], before[0]);
            assert_eq!(after[2], before[2]);
            assert!(
                f.state
                    .notice
                    .get_untracked()
                    .contains("bisherige Seite bleibt erhalten")
            );
            assert_eq!(f.repo.load().unwrap().projects[0].pages[1], original);
        }
        "rotate-unprocessed" => {
            let before = f.pages();
            f.state.page(f.project, before[1].id, |p| p.rotation = 90);
            let after = f.pages();
            assert_eq!(after[1].input_rotation, 90);
            assert_eq!(after[1].rotation, 0);
            assert_eq!(after[1].status, PageStatus::Pending);
            assert_eq!(after[1].revision, before[1].revision + 1);
            assert_eq!(after[1].source, before[1].source);
            assert_eq!(after[1].capture, before[1].capture);
            assert_eq!(after[0], before[0]);
            assert_eq!(after[2], before[2]);
            assert_eq!(f.repo.load().unwrap().projects[0].pages, after);
            // Repeated turns accumulate on the source, without an extra output turn.
            f.state.page(f.project, before[1].id, |p| p.rotation = 90);
            assert_eq!(f.pages()[1].input_rotation, 180);
            assert_eq!(f.pages()[1].rotation, 0);
        }
        "rotate-completed" => {
            f.state.finish_job(f.job(), Ok(f.batch()));
            let before = f.pages();
            f.state.page(f.project, before[1].id, |p| p.rotation = 90);
            let after = f.pages();
            assert_eq!(after[1].rotation, 90);
            assert_eq!(after[1].input_rotation, before[1].input_rotation);
            assert_eq!(after[1].revision, before[1].revision);
            assert_eq!(after[1].status, PageStatus::Ready);
            assert_eq!(after[1].output, before[1].output);
            assert_eq!(after[1].source, before[1].source);
            assert_eq!(after[1].capture, before[1].capture);
            assert_eq!(after[1].side, before[1].side);
            assert_eq!(after[0], before[0]);
            assert_eq!(after[2], before[2]);
            assert_eq!(f.repo.load().unwrap().projects[0].pages[1], after[1]);
        }
        "capture-layout" => {
            let paths = vec![
                "/tmp/camera-first.jpg".to_owned(),
                "/tmp/camera-second.jpg".to_owned(),
            ];
            f.state
                .queue_capture(f.project, paths.clone(), None, PageLayout::Double);
            let replacement = f.pages()[1].id;
            f.state
                .queue_import(f.project, vec!["/tmp/import.jpg".into()], Some(replacement));
            let queued = f.state.library.get_untracked().imports;
            assert_eq!(queued.len(), 3);
            assert_ne!(queued[0].id, queued[1].id);
            for (i, item) in queued[..2].iter().enumerate() {
                assert_eq!(item.layout, PageLayout::Double);
                assert_eq!(item.path, paths[i]);
                assert_eq!(item.project, f.project);
                assert_eq!(item.replace, None);
            }
            assert_eq!(queued[2].layout, PageLayout::Auto);
            assert_eq!(queued[2].replace, Some(replacement));
            assert_eq!(queued[2].path, "/tmp/import.jpg");
            assert_eq!(f.repo.load().unwrap().imports, queued);
        }
        "recovery" => {
            let before = f.pages();
            assert_eq!(before[1].status, PageStatus::Processing);
            let reopened = Repository::open(f.repo.root.clone()).unwrap();
            let recovered = reopened.load().unwrap();
            let after = &recovered.projects[0].pages;
            assert_eq!(after[1].status, PageStatus::Pending);
            assert_eq!(after[1].id, before[1].id);
            assert_eq!(after[1].source, before[1].source);
            assert_eq!(after[1].revision, before[1].revision);
            assert_eq!(after[0], before[0]);
            assert_eq!(after[2], before[2]);
            reopened.save(&recovered).unwrap();
            assert_eq!(reopened.load().unwrap(), recovered);
        }
        _ => panic!("Unknown workflow test: {name}"),
    }
}

#[test]
fn captured_spreads_reserve_both_pages_before_import_or_rendering() {
    check("immediate-pages");
}

#[component]
fn WorkflowHarness() -> impl IntoElement {
    let case = Storage::get_item(CASE).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_scenario(&case)));
    let report = match result {
        Ok(()) => "ok".to_owned(),
        Err(error) => error
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| error.downcast_ref::<&str>().map(|v| (*v).to_owned()))
            .unwrap_or_else(|| "workflow panicked".into()),
    };
    Storage::set_item(RESULT, report).unwrap();
    view().child(text("workflow complete"))
}

fn check(case: &str) {
    let storage = MemoryStorage::new([(CASE.into(), case.into())]);
    let mut config = EngineConfig::new(Root::of::<WorkflowHarness>());
    config.environment.locale = nsh_core::platform::Locale::parse("de-DE").unwrap();
    config.storage = Some(storage.clone());
    let mut app = Headless::start(config).unwrap();
    app.wait_for("text:workflow complete").unwrap();
    assert_eq!(storage.entries()[RESULT], "ok", "{case}");
}

#[test]
fn stale_revision_completion_is_ignored() {
    check("stale");
}
#[test]
fn deleted_pages_and_projects_are_not_resurrected() {
    check("deleted");
}
#[test]
fn discovery_inserts_two_ordered_pages_at_the_original_position() {
    check("discovery");
}
#[test]
fn reprocessing_a_spread_child_preserves_its_sibling() {
    check("reprocess-child");
}
#[test]
fn failed_retake_restores_the_previous_completed_page() {
    check("retake-failure");
}
#[test]
fn retake_without_the_selected_side_restores_the_previous_page() {
    check("retake-missing-side");
}
#[test]
fn interrupted_processing_recovers_durably_as_pending() {
    check("recovery");
}

#[test]
fn rotating_unprocessed_pages_changes_source_orientation_and_invalidates_the_job() {
    check("rotate-unprocessed");
}
#[test]
fn rotating_completed_pages_preserves_the_render_and_its_capture_sibling() {
    check("rotate-completed");
}
#[test]
fn camera_capture_preserves_double_layout_while_file_import_uses_auto() {
    check("capture-layout");
}

#[test]
#[ignore = "full photograph, native inference and rendering; run explicitly with --ignored --nocapture"]
fn real_photograph_gcs_mesh_cache_and_exports() {
    let source = PathBuf::from("/Users/simon/Downloads/input_fullsize.jpg");
    let original = fs::read(&source).expect("the user's test photograph must be available");
    let root = PathBuf::from("/tmp").join(format!("smude-real-workflow-{}", id()));
    let repo = Repository::open(root.clone()).unwrap();
    let owned = repo.import(&source).unwrap();
    assert_eq!(fs::read(repo.path(&owned).unwrap()).unwrap(), original);
    let mut project = Project::new("Workflow test".into());
    project.capture(1, Some(owned), None);
    project.pages[0].discover = true;
    let mut inputs = vec![];
    for model in [Model::Gcs, Model::Mesh] {
        let mut page = project.pages[0].clone();
        page.model = model;
        let job = Job {
            project: project.id,
            page,
        };
        let begin = Instant::now();
        let rendered = repo.process(&job).unwrap();
        let elapsed = begin.elapsed();
        assert_eq!(rendered.pages.len(), 1, "test photograph is a single page");
        let output = &rendered.pages[0];
        let path = repo.path(&output.path).unwrap();
        let dimensions = image::image_dimensions(&path).unwrap();
        assert_eq!([dimensions.0, dimensions.1], output.dimensions);
        assert!(dimensions.0 > 1000 && dimensions.1 > 1000);
        let completed = path
            .ancestors()
            .find(|p| p.join("completed.json").is_file())
            .expect("completed processing manifest")
            .join("completed.json");
        let before_image = fs::metadata(&path).unwrap().modified().unwrap();
        let before_manifest = fs::metadata(&completed).unwrap().modified().unwrap();
        let cache_start = Instant::now();
        let cached = repo.process(&job).unwrap();
        let cache_elapsed = cache_start.elapsed();
        assert_eq!(
            serde_json::to_value(&rendered).unwrap(),
            serde_json::to_value(&cached).unwrap()
        );
        assert_eq!(
            fs::metadata(&path).unwrap().modified().unwrap(),
            before_image
        );
        assert_eq!(
            fs::metadata(&completed).unwrap().modified().unwrap(),
            before_manifest
        );
        println!(
            "{model:?}: process={elapsed:?}, cache={cache_elapsed:?}, dimensions={dimensions:?}, warnings={:?}",
            rendered.warnings
        );
        inputs.push(ExportInput { path, rotation: 0 });
    }
    for color in [Color::Gray, Color::Binary] {
        let paths = exporting::export(
            &inputs,
            ExportOptions {
                format: Format::Pdf,
                color,
                size: ExportSize::Standard,
            },
            &format!("comparison-{color:?}"),
            &root.join("exports"),
        )
        .unwrap();
        assert_eq!(paths.len(), 1);
        let bytes = fs::read(&paths[0]).unwrap();
        assert!(bytes.starts_with(b"%PDF-"));
        let pdf = String::from_utf8_lossy(&bytes);
        assert!(pdf.contains("/Count 2"));
        assert!(pdf.contains("/DeviceGray"));
        assert!(pdf.contains(if color == Color::Gray {
            "/BitsPerComponent 8"
        } else {
            "/BitsPerComponent 1"
        }));
        println!(
            "PDF {color:?}: {} bytes, {}",
            bytes.len(),
            paths[0].display()
        );
    }
    let jpegs = exporting::export(
        &inputs,
        ExportOptions {
            format: Format::Jpeg,
            color: Color::Gray,
            size: ExportSize::Small,
        },
        "comparison",
        &root.join("exports"),
    )
    .unwrap();
    assert_eq!(jpegs.len(), 2);
    for path in jpegs {
        let image = image::open(&path).unwrap();
        assert_eq!(image.color(), image::ColorType::L8);
        assert_eq!(image.width().max(image.height()), 1600);
    }
    assert_eq!(
        fs::read(&source).unwrap(),
        original,
        "source photograph must remain unchanged"
    );
    println!("Inspect pipeline and export artifacts: {}", root.display());
}

#[test]
fn accepted_capture_rotation_survives_the_durable_import_queue() {
    check("capture-orientation");
}

#[test]
fn old_pipeline_results_refresh_without_losing_preview_original_or_user_settings() {
    check("render-migration");
}

#[test]
fn captured_files_finish_out_of_order_after_the_camera_screen_is_closed() {
    check("capture-completes-after-leaving");
}
#[test]
fn pending_capture_deletion_and_undo_restore_the_original_queue_membership() {
    check("pending-delete-undo");
}
#[test]
fn a_failed_native_retake_restores_the_previous_page() {
    check("capture-failure-retake");
}

#[test]
fn accepted_native_capture_count_and_rotation_replace_the_preview_snapshot() {
    check("capture-count-snapshot");
}

#[test]
fn rotation_before_capture_import_finishes_is_not_lost() {
    check("rotate-reserved");
}

#[test]
fn raw_camera_jpeg_and_its_exif_orientation_survive_import_without_reencoding() {
    check("raw-exif");
}

#[component]
fn IndependentWorkersHarness() -> impl IntoElement {
    use std::sync::{Arc, Condvar, Mutex};
    let s = State::new();
    let root = PathBuf::from(Storage::get_item("worker-test-root").unwrap());
    let repo = Repository::open(root.clone()).unwrap();
    let source = repo.path("sources/source.jpg").unwrap();
    image::RgbImage::new(11, 17).save(&source).unwrap();
    let output = "renders/ready.png";
    image::GrayImage::new(11, 17)
        .save(repo.path(output).unwrap())
        .unwrap();
    let mut collection = Project::new("Workers".into());
    collection.capture(2, Some("sources/source.jpg".into()), None);
    let id = collection.id;
    let first = collection.pages[0].id;
    let mut ready = collection.pages[1].clone();
    ready.status = PageStatus::Ready;
    ready.output = Some(output.into());
    collection.pages[1] = ready.clone();
    s.root.set(Some(root));
    s.save(Library {
        projects: vec![collection],
        imports: vec![],
    });
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let waiting = gate.clone();
    s.start_worker_with(move |_, _, _| {
        let (lock, wake) = &*waiting;
        let (_guard, timeout) = wake
            .wait_timeout_while(
                lock.lock().unwrap(),
                std::time::Duration::from_secs(10),
                |released| !*released,
            )
            .unwrap();
        if timeout.timed_out() {
            return Err("test render was never released".into());
        }
        Ok(RenderBatch {
            warnings: vec![],
            pages: vec![Rendered {
                side: 0,
                path: "renders/ready.png".into(),
                dimensions: [11, 17],
                quality: Default::default(),
            }],
        })
    });
    view()
        .child(text(move || {
            let p = s.project(id);
            let held = p
                .pages
                .iter()
                .any(|p| p.id == first && p.status == PageStatus::Processing);
            let imported = p.pages.len() == 3
                && p.pages[2].status == PageStatus::Pending
                && s.library.with(|l| l.imports.is_empty());
            if held && imported && s.export_preview.get().is_some() {
                "import and export finished while rendering is blocked"
            } else if held {
                "rendering is blocked"
            } else {
                "waiting"
            }
        }))
        .child(crate::ui::action("enqueue", false, move || {
            s.reserve_file(
                id,
                source.to_string_lossy().into_owned(),
                None,
                PageLayout::Single,
                0,
            );
            s.export_request.set(Some(crate::state::ExportRequest {
                project: id,
                pages: vec![ready.clone()],
                options: s.export.get_untracked(),
                filename: "worker-test".into(),
                preview: true,
            }));
        }))
        .child(crate::ui::action("release", false, move || {
            let (lock, wake) = &*gate;
            *lock.lock().unwrap() = true;
            wake.notify_all();
        }))
}

#[test]
fn imports_and_export_preview_finish_while_a_render_worker_is_blocked() {
    let root = std::env::temp_dir().join(format!("smude-worker-lanes-{}", id()));
    let mut config = EngineConfig::new(Root::of::<IndependentWorkersHarness>());
    config.storage = Some(MemoryStorage::new([(
        "worker-test-root".into(),
        root.to_string_lossy().into_owned(),
    )]));
    let mut app = Headless::start(config).unwrap();
    app.wait_for("text:rendering is blocked").unwrap();
    app.press("text:enqueue").unwrap();
    let completed = app.wait_for("text:import and export finished while rendering is blocked");
    app.press("text:release").unwrap();
    completed.unwrap();
    drop(app);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn unassigned_capture_batches_merge_without_losing_in_flight_imports_or_renders() {
    check("quick-capture-merge");
}
#[test]
fn legacy_gcs_pages_migrate_once_to_mesh_preserving_original_and_preview() {
    check("mesh-migration");
}

#[component]
fn ShareRetry27Harness() -> impl IntoElement {
    let state = State::new();
    state
        .root
        .set(Some(PathBuf::from(Storage::get_item("ux27-root").unwrap())));
    state.start_worker();
    view()
        .flex_1()
        .child(
            nsh_base::nav_stack()
                .flex_1()
                .child(crate::screens::export(state, 1, vec![1])),
        )
}
#[test]
fn pdf_error_has_a_working_retry_without_changing_the_selection() {
    let root = std::env::temp_dir().join(format!("smude-share-retry27-{}", id()));
    fs::create_dir_all(&root).unwrap();
    let mut library = Library::examples();
    library.projects[0].pages[0].output = Some("missing.png".into());
    let storage = MemoryStorage::new([
        (
            "smude-rust-ui-v1".into(),
            serde_json::to_string(&library).unwrap(),
        ),
        ("ux27-root".into(), root.to_string_lossy().into_owned()),
    ]);
    let mut config = EngineConfig::new(Root::of::<ShareRetry27Harness>());
    config.storage = Some(storage);
    let mut app = Headless::start(config).unwrap();
    app.wait_for("id:share-error").unwrap();
    assert!(!app.screen().enabled("id:share-pdf").unwrap());
    image::GrayImage::from_pixel(80, 120, image::Luma([255]))
        .save(root.join("missing.png"))
        .unwrap();
    app.press("id:share-retry").unwrap();
    app.wait_until(
        "prepared PDF after retry",
        std::time::Duration::from_secs(10),
        |screen| screen.enabled("id:share-pdf").unwrap_or(false),
    )
    .unwrap();
    assert!(!app.screen().matches("id:share-error").unwrap());
    assert!(app.shows("Share 1 page"));
    drop(app);
    fs::remove_dir_all(root).unwrap();
}
