//! Project recency is user-edit time, independent of asynchronous render jobs.
use super::{Library, Project};

pub(super) fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
impl Project {
    /// Legacy IDs encode capture/creation time in microseconds. This preserves
    /// useful historical ordering without inventing a migration-time edit.
    pub fn last_modified(&self) -> u64 {
        if self.modified_at != 0 {
            return self.modified_at;
        }
        std::iter::once(self.id)
            .chain(self.pages.iter().flat_map(|p| [p.id, p.capture]))
            .filter(|id| *id >= 946_684_800_000_000)
            .max()
            .unwrap_or(0)
            / 1000
    }
}
impl Library {
    pub(crate) fn initialize_project_dates(&mut self) {
        for project in &mut self.projects {
            if project.modified_at == 0 {
                project.modified_at = project.last_modified();
            }
        }
    }
    pub fn recent_projects(&self) -> Vec<Project> {
        let mut projects = self.projects.clone();
        projects.sort_by(|a, b| {
            b.last_modified()
                .cmp(&a.last_modified())
                .then(b.id.cmp(&a.id))
        });
        projects
    }
    pub(crate) fn record_user_edits(&mut self, previous: &Self) {
        let now = now_millis();
        for project in &mut self.projects {
            let old = previous.projects.iter().find(|p| p.id == project.id);
            let imports_changed = self
                .imports
                .iter()
                .filter(|i| i.project == project.id)
                .ne(previous.imports.iter().filter(|i| i.project == project.id));
            if old != Some(project) || imports_changed {
                project.modified_at =
                    now.max(old.map_or(0, |p| p.last_modified().saturating_add(1)));
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_libraries_load_and_keep_known_capture_dates() {
        let old = r#"{"id":1794220037117065,"title":"Old","pages":[]}"#;
        let project: Project = serde_json::from_str(old).unwrap();
        assert_eq!(project.modified_at, 0);
        assert_eq!(project.last_modified(), 1794220037117);
    }
    #[test]
    fn edits_reorder_only_changed_projects_and_noops_keep_their_dates() {
        let mut a = Project::new("A".into());
        a.modified_at = 100;
        let mut b = Project::new("B".into());
        b.modified_at = 200;
        let original = Library {
            projects: vec![a.clone(), b.clone()],
            imports: vec![],
        };
        assert_eq!(original.recent_projects()[0].id, b.id);
        let mut edited = original.clone();
        edited.projects[0].title = "A edited".into();
        edited.record_user_edits(&original);
        assert_eq!(edited.recent_projects()[0].id, a.id);
        assert_eq!(edited.projects[1].modified_at, 200);
        let mut noop = edited.clone();
        noop.record_user_edits(&edited);
        assert_eq!(noop, edited);
    }
    #[test]
    fn legacy_dates_are_frozen_before_background_pages_can_be_discovered() {
        let project: Project =
            serde_json::from_str(r#"{"id":1794220037117065,"title":"Old","pages":[]}"#).unwrap();
        let mut library = Library {
            projects: vec![project],
            imports: vec![],
        };
        library.initialize_project_dates();
        let known = library.projects[0].last_modified();
        library.projects[0].capture(2, None, None);
        assert_eq!(library.projects[0].last_modified(), known);
    }
}
