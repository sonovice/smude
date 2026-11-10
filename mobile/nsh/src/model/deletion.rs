//! One reversible deletion. Restoration merges only removed items into current
//! state, so background processing results can never be rolled back by Undo.
use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Deletion {
    pub id: u64,
    pub message: String,
    restore_error: Option<String>,
    projects: Vec<(usize, Project)>,
    pages: Vec<(u64, usize, Page)>,
    imports: Vec<PendingImport>,
}
impl Deletion {
    pub fn pages(library: &Library, project: u64, ids: &[u64]) -> Self {
        let pages = library
            .projects
            .iter()
            .filter(|p| p.id == project)
            .flat_map(|p| {
                p.pages
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| ids.contains(&p.id))
                    .map(|(i, p)| (project, i, p.clone()))
            })
            .collect::<Vec<_>>();
        Self {
            id: id(),
            restore_error: None,
            message: if pages.len() == 1 {
                "Seite gelöscht"
            } else {
                "Seiten gelöscht"
            }
            .into(),
            projects: vec![],
            pages,
            imports: library
                .imports
                .iter()
                .filter(|i| i.pages.iter().any(|p| ids.contains(p)))
                .map(|i| {
                    let mut i = i.clone();
                    i.pages.retain(|p| ids.contains(p));
                    i
                })
                .collect(),
        }
    }
    pub fn projects(library: &Library, ids: &[u64]) -> Self {
        let projects = library
            .projects
            .iter()
            .enumerate()
            .filter(|(_, p)| ids.contains(&p.id))
            .map(|(i, p)| (i, p.clone()))
            .collect::<Vec<_>>();
        Self {
            id: id(),
            restore_error: None,
            message: if projects.len() == 1 {
                "Projekt gelöscht"
            } else {
                "Projekte gelöscht"
            }
            .into(),
            projects,
            pages: vec![],
            imports: library
                .imports
                .iter()
                .filter(|i| ids.contains(&i.project))
                .cloned()
                .collect(),
        }
    }
    pub fn localized_message(&self, language: crate::i18n::Language) -> String {
        if let Some(error) = &self.restore_error {
            return error.clone();
        }
        use crate::i18n::Key;
        let key = if self.projects.is_empty() {
            if self.pages.len() == 1 {
                Key::PageDeleted
            } else {
                Key::PagesDeleted
            }
        } else if self.projects.len() == 1 {
            Key::CollectionDeleted
        } else {
            Key::CollectionsDeleted
        };
        language.text(key).into()
    }
    pub(crate) fn set_restore_error(&mut self, error: String) {
        self.restore_error = Some(error);
    }
    pub fn restore(&self, current: &Library) -> Library {
        let mut result = current.clone();
        let mut restored_projects = vec![];
        let prepare = |mut p: Page| {
            if p.status == PageStatus::Processing {
                p.status = PageStatus::Pending;
            }
            p
        };
        for (i, project) in &self.projects {
            if result.projects.iter().any(|p| p.id == project.id) {
                continue;
            }
            let mut project = project.clone();
            project.pages = project.pages.into_iter().map(prepare).collect();
            restored_projects.push(project.id);
            result
                .projects
                .insert((*i).min(result.projects.len()), project);
        }
        for (project, index, page) in &self.pages {
            if let Some(p) = result.projects.iter_mut().find(|p| p.id == *project)
                && !p.pages.iter().any(|v| v.id == page.id)
            {
                p.pages
                    .insert((*index).min(p.pages.len()), prepare(page.clone()));
            }
        }
        for entry in &self.imports {
            let restored = restored_projects.contains(&entry.project)
                || self
                    .pages
                    .iter()
                    .any(|(_, _, p)| entry.pages.contains(&p.id));
            if !restored {
                continue;
            }
            if let Some(current) = result.imports.iter_mut().find(|i| i.id == entry.id) {
                for page in &entry.pages {
                    if !current.pages.contains(page) {
                        current.pages.push(*page);
                    }
                }
            } else {
                result.imports.push(entry.clone());
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn undo_keeps_completed_background_work_and_is_idempotent() {
        let mut project = Project::new("Test".into());
        project.capture(2, Some("original.jpg".into()), None);
        let old = Library {
            projects: vec![project.clone()],
            imports: vec![],
        };
        let removed = project.pages[0].id;
        let deletion = Deletion::pages(&old, project.id, &[removed]);
        let mut current = old.clone();
        current.projects[0].pages.remove(0);
        current.projects[0].pages[0].status = PageStatus::Ready;
        current.projects[0].pages[0].output = Some("new-render.png".into());
        let restored = deletion.restore(&current);
        assert_eq!(restored.projects[0].pages[0].id, removed);
        assert_eq!(restored.projects[0].pages[1], current.projects[0].pages[0]);
        assert_eq!(deletion.restore(&restored), restored);
    }
    #[test]
    fn project_undo_preserves_new_projects_and_current_edits() {
        let a = Project::new("A".into());
        let b = Project::new("B".into());
        let old = Library {
            projects: vec![a.clone(), b.clone()],
            imports: vec![],
        };
        let deletion = Deletion::projects(&old, &[a.id]);
        let mut current = Library {
            projects: vec![b.clone(), Project::new("C".into())],
            imports: vec![],
        };
        current.projects[0].title = "B updated".into();
        let restored = deletion.restore(&current);
        assert_eq!(restored.projects[0], a);
        assert_eq!(&restored.projects[1..], current.projects.as_slice());
    }
}
