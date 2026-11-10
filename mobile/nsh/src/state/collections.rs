//! Camera batches remain durable and unassigned until explicitly named or merged.
use super::*;
impl State {
    pub fn quick_capture(self) {
        let mut library = self.library.get_untracked();
        let mut draft = Project::new(t(Key::Scans).into());
        draft.draft = true;
        let id = draft.id;
        library.projects.insert(0, draft);
        if self.save(library) {
            self.push(Route::Camera(id, None));
        }
    }
    pub fn finish_camera(self, id: u64) {
        let project = self.project(id);
        if project.draft && project.pages.is_empty() {
            let mut library = self.library.get_untracked();
            library.projects.retain(|p| p.id != id);
            self.save(library);
            self.back();
        } else if project.draft && self.routes.with_untracked(|r| r.len() == 2) {
            self.routes.update(|r| {
                if let Some(last) = r.last_mut() {
                    *last = Route::Project(id);
                }
            });
        } else {
            self.back();
        }
    }
    /// Review never waits for capture saves or processing workers.
    pub fn review_camera(self, id: u64) {
        self.routes.update(|routes| {
            if let Some(index) = routes.iter().position(|r| *r == Route::Project(id)) {
                routes.truncate(index + 1);
            } else if let Some(last) = routes.last_mut() {
                *last = Route::Project(id);
            }
        });
    }
    pub fn assign_draft(self, source: u64, target: u64) {
        if source == target {
            return;
        }
        let mut library = self.library.get_untracked();
        let Some(from) = library
            .projects
            .iter()
            .position(|p| p.id == source && p.draft)
        else {
            return;
        };
        if !library.projects.iter().any(|p| p.id == target && !p.draft) {
            return;
        }
        let draft = library.projects.remove(from);
        library
            .projects
            .iter_mut()
            .find(|p| p.id == target)
            .unwrap()
            .pages
            .extend(draft.pages);
        for request in &mut library.imports {
            if request.project == source {
                request.project = target;
            }
        }
        if self.save(library) {
            self.routes.update(|r| {
                for route in r {
                    match route {
                        Route::Project(id)
                        | Route::Viewer(id, _)
                        | Route::Camera(id, _)
                        | Route::Export(id, _)
                            if *id == source =>
                        {
                            *id = target
                        }
                        _ => {}
                    }
                }
            });
        }
    }
}
