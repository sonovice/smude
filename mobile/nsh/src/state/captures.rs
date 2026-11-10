//! Durable capture reservations outlive the camera screen and keep page IDs stable.
use super::*;
impl State {
    fn reserve(
        self,
        project: u64,
        path: String,
        replace: Option<u64>,
        layout: PageLayout,
        rotation: u16,
        awaiting: bool,
    ) -> Option<u64> {
        let mut l = self.library.get_untracked();
        let request = id();
        let p = l.projects.iter_mut().find(|p| p.id == project)?;
        let mut pages = vec![];
        if let Some(target) = replace {
            let old = p.pages.iter_mut().find(|p| p.id == target)?;
            let original = old
                .replacement
                .take()
                .map(|p| *p)
                .unwrap_or_else(|| old.clone());
            old.replacement = Some(Box::new(original));
            old.capture = request;
            old.side = 0;
            old.rotation = 0;
            old.input_rotation = rotation;
            old.layout = PageLayout::Single;
            old.discover = false;
            old.revision += 1;
            old.status = PageStatus::Importing;
            old.error = None;
            pages.push(old.id);
        } else {
            let count = if layout == PageLayout::Double { 2 } else { 1 };
            let start = p.pages.len();
            p.capture(count, None, None);
            for page in &mut p.pages[start..] {
                page.capture = request;
                page.layout = layout;
                page.input_rotation = rotation;
                page.discover = layout == PageLayout::Auto;
                page.status = PageStatus::Importing;
                pages.push(page.id);
            }
        }
        l.imports.push(PendingImport {
            id: request,
            project,
            path,
            replace,
            layout,
            input_rotation: rotation,
            pages,
            awaiting_capture: awaiting,
        });
        self.save(l).then_some(request)
    }
    pub fn reserve_file(
        self,
        project: u64,
        path: String,
        replace: Option<u64>,
        layout: PageLayout,
        rotation: u16,
    ) -> Option<u64> {
        self.reserve(project, path, replace, layout, rotation % 360, false)
    }
    pub fn begin_capture(
        self,
        project: u64,
        replace: Option<u64>,
        count: usize,
        rotation: u16,
    ) -> Option<serde_json::Value> {
        let root = self.repo()?.root;
        let file = root.join("captures").join(format!("{}.jpg", id()));
        let layout = if count == 2 && replace.is_none() {
            PageLayout::Double
        } else {
            PageLayout::Single
        };
        let path = file.to_string_lossy().into_owned();
        let request = self.reserve(project, path.clone(), replace, layout, rotation % 360, true)?;
        Some(serde_json::json!({"request_id":request,"path":path,"single_only":replace.is_some()}))
    }
    pub fn capture_event(self, data: &serde_json::Value) {
        let Some(request) = data["request_id"].as_u64() else {
            return;
        };
        let mut l = self.library.get_untracked();
        let Some(index) = l.imports.iter().position(|i| i.id == request) else {
            return;
        };
        match data["event"].as_str() {
            Some("capture_started") => {
                let rotation = (data["rotation"]
                    .as_u64()
                    .unwrap_or(u64::from(l.imports[index].input_rotation))
                    % 360) as u16;
                let count = data["pages"]
                    .as_u64()
                    .unwrap_or(l.imports[index].pages.len() as u64)
                    .clamp(1, 2) as usize;
                let project = l.imports[index].project;
                let old_ids = l.imports[index].pages.clone();
                let mut ids = old_ids.clone();
                if let Some(p) = l.projects.iter_mut().find(|p| p.id == project) {
                    if count < ids.len() {
                        let removed = ids.split_off(count);
                        p.pages.retain(|p| !removed.contains(&p.id));
                    }
                    if count > ids.len()
                        && let Some(template) =
                            p.pages.iter().find(|p| ids.contains(&p.id)).cloned()
                    {
                        for side in ids.len()..count {
                            let mut page = template.clone();
                            page.id = id();
                            page.side = side;
                            page.replacement = None;
                            let at = p
                                .pages
                                .iter()
                                .rposition(|p| ids.contains(&p.id))
                                .map_or(p.pages.len(), |i| i + 1);
                            ids.push(page.id);
                            p.pages.insert(at, page);
                        }
                    }
                    for p in p.pages.iter_mut().filter(|p| ids.contains(&p.id)) {
                        p.input_rotation = rotation;
                        p.layout = if count == 2 {
                            PageLayout::Double
                        } else {
                            PageLayout::Single
                        };
                    }
                }
                l.imports[index].pages = ids;
                l.imports[index].layout = if count == 2 {
                    PageLayout::Double
                } else {
                    PageLayout::Single
                };
                l.imports[index].input_rotation = rotation;
            }
            Some("capture_saved") => {
                l.imports[index].awaiting_capture = false;
                if let Some(repo) = self.repo()
                    && let Ok(path) =
                        std::path::Path::new(&l.imports[index].path).strip_prefix(&repo.root)
                {
                    let ids = l.imports[index].pages.clone();
                    let source = path.to_string_lossy().into_owned();
                    let exif = data["source_rotation"]
                        .as_u64()
                        .map(|v| (v % 360) as u16)
                        .unwrap_or_else(|| {
                            if cfg!(target_os = "android") {
                                crate::repository::source_rotation(std::path::Path::new(
                                    &l.imports[index].path,
                                ))
                            } else {
                                0
                            }
                        });
                    for p in l
                        .projects
                        .iter_mut()
                        .flat_map(|p| &mut p.pages)
                        .filter(|p| ids.contains(&p.id))
                    {
                        p.source = Some(source.clone());
                        p.source_rotation = exif;
                    }
                }
            }
            Some("capture_failed") => {
                let import = l.imports.remove(index);
                let rejected = data["reason"].as_str() == Some("not_ready");
                if rejected {
                    for project in &mut l.projects {
                        project
                            .pages
                            .retain(|p| !import.pages.contains(&p.id) || p.replacement.is_some());
                    }
                }
                for p in l
                    .projects
                    .iter_mut()
                    .flat_map(|p| &mut p.pages)
                    .filter(|p| import.pages.contains(&p.id))
                {
                    if let Some(old) = p.replacement.take() {
                        *p = *old;
                    } else {
                        p.status = PageStatus::Failed;
                        p.error = Some(t(Key::CouldNotSaveThePhoto).into());
                    }
                }
                self.notice.set(t(Key::CouldNotSaveThePhoto).into());
            }
            _ => return,
        }
        self.persist(l, false);
    }
    pub(crate) fn finish_import(self, import: PendingImport, result: Result<String, String>) {
        let mut l = self.library.get_untracked();
        let Some(import) = l.imports.iter().find(|i| i.id == import.id).cloned() else {
            return;
        };
        l.imports.retain(|i| i.id != import.id);
        if let Some(project) = l.projects.iter_mut().find(|p| p.id == import.project) {
            match result {
                Ok(source) => {
                    let exif = if cfg!(target_os = "android") {
                        self.repo()
                            .and_then(|r| r.path(&source).ok())
                            .map(|p| crate::repository::source_rotation(&p))
                    } else {
                        None
                    };
                    if import.pages.is_empty() {
                        // Records queued by older app versions have no reservations.
                        if let Some(target) = import.replace {
                            if let Some(p) = project.pages.iter_mut().find(|p| p.id == target) {
                                let original = p.clone();
                                p.replacement = Some(Box::new(original));
                                p.source = Some(source);
                                p.source_rotation = exif.unwrap_or(0);
                                p.capture = id();
                                p.side = 0;
                                p.rotation = 0;
                                p.input_rotation = import.input_rotation;
                                p.layout = PageLayout::Single;
                                p.discover = false;
                                p.invalidate();
                            }
                        } else {
                            project.capture(1, Some(source), None);
                            let p = project.pages.last_mut().unwrap();
                            p.discover = true;
                            p.source_rotation = exif.unwrap_or(0);
                            p.layout = import.layout;
                            p.input_rotation = import.input_rotation;
                        }
                    } else {
                        for p in project
                            .pages
                            .iter_mut()
                            .filter(|p| import.pages.contains(&p.id))
                        {
                            p.source = Some(source.clone());
                            if let Some(exif) = exif {
                                p.source_rotation = exif;
                            }
                            p.status = PageStatus::Pending;
                            p.input_rotation = import.input_rotation;
                            if p.output.is_none() {
                                p.input_rotation = (p.input_rotation + p.rotation) % 360;
                                p.rotation = 0;
                            }
                            p.error = None;
                        }
                    }
                }
                Err(error) => {
                    for p in project
                        .pages
                        .iter_mut()
                        .filter(|p| import.pages.contains(&p.id))
                    {
                        if let Some(old) = p.replacement.take() {
                            *p = *old;
                        } else {
                            p.status = PageStatus::Failed;
                            p.error = Some(error.clone());
                        }
                    }
                    self.notice
                        .set(crate::i18n::failure(crate::i18n::Failure::Import, &error));
                }
            }
        }
        self.persist(l, false);
    }
}
