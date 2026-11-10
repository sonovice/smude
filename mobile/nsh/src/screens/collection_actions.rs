//! Shared collection commands; native controls present the anchored overflow menu.
use super::*;
use crate::ui::native_controls::Choice;
pub fn choices(s: State, id: u64, selection: Option<Vec<u64>>) -> Vec<Choice> {
    let collection = s.project(id);
    let selecting = selection.is_some();
    let ids: Vec<_> = collection
        .pages
        .iter()
        .filter(|p| selection.as_ref().is_none_or(|ids| ids.contains(&p.id)))
        .map(|p| p.id)
        .collect();
    let mut choices = vec![];
    if !selecting {
        choices.push(Choice::new(
            "photos",
            t(Key::ImportPhotos),
            "photo.on.rectangle",
        ));
        choices.push(Choice::new("files", t(Key::Files), "folder"));
        if !collection.draft {
            choices.push(Choice::new("rename", t(Key::Rename), "pencil"));
        }
    }
    if !ids.is_empty() {
        choices.push(
            Choice::new(
                "share",
                crate::i18n::share_pages(ids.len()),
                "square.and.arrow.up",
            )
            .payload(serde_json::json!(ids)),
        );
    }
    if !selecting && collection.draft {
        choices.push(Choice::new(
            "collect",
            t(Key::AddToCollection),
            "folder.badge.plus",
        ));
        choices.push(Choice::new("delete", t(Key::DeleteScans), "trash").destructive());
    } else if !selecting {
        choices.push(Choice::new("delete", t(Key::DeleteCollection), "trash").destructive());
    }
    choices
}
pub fn select(s: State, id: u64, action: &str, payload: &serde_json::Value) {
    if !s
        .library
        .with_untracked(|l| l.projects.iter().any(|p| p.id == id))
    {
        return;
    }
    match action {
        "photos" => s.request_import(id, false),
        "files" => s.request_import(id, true),
        "rename" => s.push(Route::Name(Some(id))),
        "share" => {
            let ids: Vec<_> = payload
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|p| p.as_u64())
                .filter(|page| s.project(id).pages.iter().any(|p| p.id == *page))
                .collect();
            if !ids.is_empty() {
                s.push(Route::Export(id, ids));
            }
        }
        "collect" => assign(s, id),
        "delete" => s.delete_project(id),
        _ => {}
    }
}
/// Long-pressing the title also exposes the same commands.
pub fn more(s: State, id: u64, selection: Option<Vec<u64>>) {
    let entries = choices(s, id, selection);
    let mut dialog = nsh_base::alert_dialog().title(t(Key::MoreActions));
    for entry in &entries {
        dialog = if entry.destructive {
            dialog.destructive(entry.id, entry.title.clone())
        } else {
            dialog.action(entry.id, entry.title.clone())
        };
    }
    let answer = dialog
        .cancel("cancel", t(Key::Cancel))
        .action_sheet()
        .show();
    spawn(async move {
        if let Ok(result) = answer.await
            && let Some(entry) = entries
                .iter()
                .find(|e| Some(e.id) == result.action.as_deref())
        {
            select(s, id, entry.id, &entry.payload);
        }
    });
}
pub fn assign(s: State, id: u64) {
    let mut dialog = nsh_base::alert_dialog()
        .title(t(Key::AddToCollection))
        .action("new", t(Key::CreateNewCollection));
    for p in s
        .library
        .get_untracked()
        .recent_projects()
        .into_iter()
        .filter(|p| !p.draft)
    {
        dialog = dialog.action(p.id.to_string(), p.title);
    }
    let answer = dialog
        .cancel("cancel", t(Key::Cancel))
        .action_sheet()
        .show();
    spawn(async move {
        if let Ok(result) = answer.await {
            match result.action.as_deref() {
                Some("new") => s.push(Route::Name(Some(id))),
                Some(action) => {
                    if let Ok(target) = action.parse() {
                        s.assign_draft(id, target);
                    }
                }
                _ => {}
            }
        }
    });
}
