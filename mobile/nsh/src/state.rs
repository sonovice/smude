mod captures;
mod collections;
mod sharing;
mod workers;
use crate::i18n::{Key, t};
use crate::{
    model::*,
    repository::{Job, RenderBatch, Repository},
};
use nsh_core::{prelude::*, storage::Storage};
use std::{path::PathBuf, time::Duration};
#[derive(Clone, Debug, PartialEq)]
pub enum Route {
    Projects,
    Name(Option<u64>),
    Project(u64),
    Viewer(u64, u64),
    Camera(u64, Option<u64>),
    Export(u64, Vec<u64>),
}
#[derive(Clone, Debug, PartialEq)]
pub struct ExportRequest {
    pub project: u64,
    pub pages: Vec<Page>,
    pub options: ExportOptions,
    pub filename: String,
    pub preview: bool,
}
#[derive(Clone, Copy)]
pub struct State {
    pub localization: crate::i18n::Localization,
    pub library: Signal<Library>,
    pub project_grid: Signal<bool>,
    pub routes: Signal<Vec<Route>>,
    pub notice: Signal<String>,
    pub undo: Signal<Option<Deletion>>,
    pub export: Signal<ExportOptions>,
    pub root: Signal<Option<PathBuf>>,
    pub ready: Signal<bool>,
    pub working: Signal<Option<u64>>,
    pub busy: Signal<bool>,
    pub progress: Signal<String>,
    pub processing: Signal<Option<crate::processing::ProcessingDisplay>>,
    pub service: ElementRef,
    pub import_request: Signal<Option<(u64, u64)>>,
    pub export_request: Signal<Option<ExportRequest>>,
    pub export_preview: Signal<Option<String>>,
    pub share_request: Signal<Option<sharing::Request>>,
    pub share_ready: Signal<Option<sharing::Prepared>>,
    pub testing: crate::testing::Testing,
}
impl State {
    pub fn new() -> Self {
        let mut library: Library = Storage::get_item("smude-rust-ui-v1")
            .and_then(|v| serde_json::from_str(&v).ok())
            .unwrap_or_default();
        library.initialize_project_dates();
        Self {
            localization: crate::i18n::Localization::new(),
            library: signal(library),
            project_grid: signal(
                Storage::get_item("smude-project-view").as_deref() == Some("grid"),
            ),
            routes: signal(vec![Route::Projects]),
            notice: signal(String::new()),
            undo: signal(None),
            export: signal(ExportOptions::default()),
            root: signal(None),
            ready: signal(!cfg!(any(target_os = "ios", target_os = "android"))),
            working: signal(None),
            busy: signal(false),
            progress: signal(String::new()),
            processing: signal(None),
            service: element_ref(),
            import_request: signal(None),
            export_request: signal(None),
            export_preview: signal(None),
            share_request: signal(None),
            share_ready: signal(None),
            testing: crate::testing::Testing::new(),
        }
    }
    pub fn push(self, r: Route) {
        self.routes.update(|v| v.push(r));
    }
    pub fn back(self) {
        self.routes.update(|v| {
            if v.len() > 1 {
                v.pop();
            }
        });
    }
    pub fn project(self, id: u64) -> Project {
        self.library.with(|l| {
            l.projects
                .iter()
                .find(|p| p.id == id)
                .cloned()
                .unwrap_or_default()
        })
    }
    pub fn repo(self) -> Option<Repository> {
        self.root.get_untracked().map(|root| Repository { root })
    }
    pub fn path(self, path: &str) -> String {
        self.repo()
            .and_then(|r| r.path(path).ok())
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_owned())
    }
    pub fn persist(self, next: Library, clear_undo: bool) -> bool {
        let result = if let Some(repo) = self.repo() {
            repo.save(&next)
        } else {
            serde_json::to_string(&next)
                .map_err(|e| e.to_string())
                .and_then(|v| Storage::set_item("smude-rust-ui-v1", v).map_err(|e| e.to_string()))
        };
        match result {
            Ok(()) => {
                if clear_undo {
                    self.undo.set(None);
                }
                self.library.set(next);
                true
            }
            Err(e) => {
                self.notice.set(crate::i18n::failure(
                    crate::i18n::Failure::Save,
                    &e.to_string(),
                ));
                false
            }
        }
    }
    pub fn save(self, mut next: Library) -> bool {
        next.record_user_edits(&self.library.get_untracked());
        self.persist(next, true)
    }
    pub fn set_project_grid(self, grid: bool) {
        self.project_grid.set(grid);
        if let Err(error) =
            Storage::set_item("smude-project-view", if grid { "grid" } else { "list" })
        {
            self.notice.set(crate::i18n::failure(
                crate::i18n::Failure::Preference,
                &error.to_string(),
            ));
        }
    }
    pub fn update(self, id: u64, f: impl FnOnce(&mut Project)) {
        let mut l = self.library.get_untracked();
        if let Some(p) = l.projects.iter_mut().find(|p| p.id == id) {
            f(p);
            self.save(l);
        }
    }
    pub fn page(self, id: u64, page: u64, f: impl FnOnce(&mut Page)) {
        self.update(id, |p| {
            if let Some(v) = p.pages.iter_mut().find(|p| p.id == page) {
                let old = (v.rotation, v.model);
                f(v);
                if v.status == PageStatus::Importing {
                    return;
                }
                if old.0 != v.rotation && v.output.is_none() && v.source.is_some() {
                    v.input_rotation = (v.input_rotation + v.rotation + 360 - old.0) % 360;
                    v.rotation = 0;
                    v.invalidate();
                } else if old.1 != v.model && v.source.is_some() {
                    v.invalidate();
                }
            }
        });
    }
    pub fn delete(self, id: u64, ids: &[u64]) {
        let old = self.library.get_untracked();
        let mut next = old.clone();
        if let Some(p) = next.projects.iter_mut().find(|p| p.id == id) {
            p.pages.retain(|p| !ids.contains(&p.id));
        }
        next.imports.retain_mut(|i| {
            if i.pages.is_empty() {
                return true;
            }
            i.pages.retain(|p| !ids.contains(p));
            !i.pages.is_empty()
        });
        if self.save(next) {
            let deletion = Deletion::pages(&old, id, ids);
            self.notice
                .set(deletion.localized_message(self.localization.language()));
            self.undo.set(Some(deletion));
        }
    }
    pub fn delete_project(self, id: u64) {
        self.delete_projects(&[id]);
    }
    pub fn delete_projects(self, ids: &[u64]) {
        if ids.is_empty() {
            return;
        }
        let old = self.library.get_untracked();
        let mut next = old.clone();
        next.projects.retain(|p| !ids.contains(&p.id));
        next.imports.retain(|i| !ids.contains(&i.project));
        if self.save(next) {
            let deletion = Deletion::projects(&old, ids);
            self.notice
                .set(deletion.localized_message(self.localization.language()));
            self.undo.set(Some(deletion));
            self.routes.set(vec![Route::Projects]);
        }
    }
    pub fn restore(self) {
        if let Some(deletion) = self.undo.get_untracked() {
            let restored = deletion.restore(&self.library.get_untracked());
            if self.save(restored) {
                self.notice.set(String::new());
            } else {
                let message = self.notice.get_untracked();
                self.undo.update(|d| {
                    if let Some(d) = d {
                        d.set_restore_error(message);
                    }
                });
            }
        }
    }
    pub fn queue_import(self, project: u64, paths: Vec<String>, replace: Option<u64>) {
        self.queue_capture(project, paths, replace, PageLayout::Auto);
    }
    pub fn queue_capture(
        self,
        project: u64,
        paths: Vec<String>,
        replace: Option<u64>,
        layout: PageLayout,
    ) {
        self.queue_oriented_capture(project, paths, replace, layout, 0);
    }
    pub fn queue_oriented_capture(
        self,
        project: u64,
        paths: Vec<String>,
        replace: Option<u64>,
        layout: PageLayout,
        input_rotation: u16,
    ) {
        for path in paths {
            self.reserve_file(project, path, replace, layout, input_rotation);
        }
    }
    pub fn request_import(self, project: u64, files: bool) {
        if self.busy.get_untracked() {
            return;
        }
        let request = id();
        self.import_request.set(Some((request, project)));
        self.busy.set(true);
        self.progress.set(t(Key::ChoosingImages).into());
        self.service.command(
            if files { "import_files" } else { "import" },
            serde_json::json!({"request_id":request}),
        );
    }
    pub fn service_event(self, e: &NativeEvent) {
        if self.testing.event(e) {
            return;
        }
        match &*e.name {
            "localization" => {
                if let Some(tag) = e.data["language"].as_str() {
                    self.localization
                        .preferred
                        .set(Some(crate::i18n::Language::from_tag(tag)));
                }
                if let Ok(dates) = serde_json::from_value(e.data["dates"].clone()) {
                    self.localization.dates.set(dates);
                }
            }
            "capture_started" | "capture_saved" | "capture_failed" => self.capture_event(&e.data),
            "ready" => {
                if self.root.get_untracked().is_some() {
                    return;
                }
                if let Some(error) = e.data["error"].as_str() {
                    self.notice.set(error.to_owned());
                    return;
                }
                if let Some(root) = e.data["root"].as_str() {
                    match Repository::open(root.into()).and_then(|r| {
                        let fresh = !r.root.join("library.json").exists();
                        let mut l = r.load()?;
                        if fresh {
                            for mut project in self.library.get_untracked().projects {
                                if project.id == 1
                                    && project.pages.iter().all(|p| p.source.is_none())
                                {
                                    continue;
                                }
                                let mut seen = std::collections::HashSet::new();
                                for p in &project.pages {
                                    if let Some(path) = &p.source
                                        && seen.insert(path.clone())
                                    {
                                        l.imports.push(PendingImport {
                                            pages: vec![],
                                            awaiting_capture: false,
                                            id: id(),
                                            project: project.id,
                                            path: path.clone(),
                                            replace: None,
                                            layout: PageLayout::Auto,
                                            input_rotation: p.input_rotation,
                                        });
                                    }
                                }
                                project.pages.clear();
                                l.projects.push(project);
                            }
                            r.save(&l)?;
                        }
                        Ok((r, l))
                    }) {
                        Ok((r, l)) => {
                            self.root.set(Some(r.root));
                            self.library.set(l);
                            self.ready.set(true);
                        }
                        Err(e) => self.notice.set(e),
                    }
                }
            }
            "imported" => {
                let req = e.data["request_id"].as_u64();
                if let Some((expected, project)) = self.import_request.get_untracked()
                    && req == Some(expected)
                {
                    self.import_request.set(None);
                    self.busy.set(false);
                    self.progress.set(String::new());
                    if let Some(err) = e.data["error"].as_str() {
                        self.notice.set(err.into());
                    } else if let Some(paths) = e.data["paths"].as_array() {
                        self.queue_import(
                            project,
                            paths
                                .iter()
                                .filter_map(|p| p.as_str().map(str::to_owned))
                                .collect(),
                            None,
                        );
                    }
                }
            }
            "shared" => {
                self.busy.set(false);
                self.progress.set(String::new());
                if let Some(err) = e.data["error"].as_str() {
                    self.notice.set(err.into());
                }
            }
            _ => {}
        }
    }
    pub fn finish_job(self, job: Job, result: Result<RenderBatch, String>) {
        let mut l = self.library.get_untracked();
        let Some(project) = l.projects.iter_mut().find(|p| {
            p.pages.iter().any(|v| {
                v.id == job.page.id
                    && v.capture == job.page.capture
                    && v.revision == job.page.revision
            })
        }) else {
            return;
        };
        let Some(index) = project.pages.iter().position(|p| {
            p.id == job.page.id && p.revision == job.page.revision && p.capture == job.page.capture
        }) else {
            return;
        };
        match result {
            Ok(batch) => {
                if !batch.warnings.is_empty() {
                    self.notice.set(batch.warnings.join(" · "));
                }
                let old = project.pages[index].clone();
                let selected = if old.discover {
                    batch.pages
                } else {
                    batch
                        .pages
                        .into_iter()
                        .filter(|p| p.side == old.side)
                        .collect()
                };
                if selected.is_empty() {
                    let page = &mut project.pages[index];
                    if let Some(old) = page.replacement.take() {
                        *page = *old;
                        self.notice.set(
                            t(Key::TheReplacementCouldNotBeMatchedYourPreviousPageIsPreserved)
                                .into(),
                        );
                    } else {
                        page.status = PageStatus::Failed;
                        page.error = Some(t(Key::ThisPageWasNotFullyDetected).into());
                    }
                } else {
                    let pages = selected
                        .into_iter()
                        .enumerate()
                        .map(|(i, r)| {
                            let mut p = old.clone();
                            if i > 0 {
                                p.id = id();
                            }
                            p.quality = r.quality;
                            p.review = p.quality.retake_suggested;
                            p.side = r.side;
                            p.output = Some(r.path);
                            p.dimensions = r.dimensions;
                            p.discover = false;
                            p.status = PageStatus::Ready;
                            p.error = None;
                            p.replacement = None;
                            p
                        })
                        .collect::<Vec<_>>();
                    project.pages.splice(index..index + 1, pages);
                }
            }
            Err(e) => {
                let p = &mut project.pages[index];
                if let Some(old) = p.replacement.take() {
                    *p = *old;
                    self.notice.set(crate::i18n::failure(
                        crate::i18n::Failure::Retake,
                        &e.to_string(),
                    ));
                } else {
                    p.status = PageStatus::Failed;
                    p.error = Some(e);
                }
            }
        }
        if !self.persist(l, false) {
            // Keep the last durable manifest authoritative. Expose a retryable
            // failure in memory instead of leaving a completed job "Processing".
            self.library.update(|library| {
                if let Some(page) = library
                    .projects
                    .iter_mut()
                    .find(|p| p.id == job.project)
                    .and_then(|p| p.pages.iter_mut().find(|p| p.id == job.page.id))
                {
                    page.status = PageStatus::Failed;
                    page.error =
                        Some(t(Key::CouldNotSaveTheResultCheckAvailableStorageAndTryAgain).into());
                }
            });
        }
    }
}
