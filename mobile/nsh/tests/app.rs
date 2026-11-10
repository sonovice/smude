//! Engine-level behavior tests for the Rust UI. These do not simulate native
//! gesture progress/cancellation, camera hardware, or actual PDF/JPEG export.
use std::sync::Arc;

use nsh_engine::{EngineConfig, Root};
use nsh_testing::{Headless, MemoryStorage};
use smude_mobile_ui::{
    App,
    model::{Library, Model},
};

const STORAGE: &str = "smude-rust-ui-v1";

fn share_collection(app: &mut Headless) {
    app.on_dialog(|dialog| {
        nsh_testing::DialogAnswer::Press(
            dialog
                .buttons
                .iter()
                .find(|b| b.text.ends_with(" teilen"))
                .expect("contextual share action")
                .text
                .clone(),
        )
    });
    app.press("id:collection-more").unwrap();
}

#[test]
fn project_save_tracks_keyboard_overlap_and_restores_its_position() {
    use nsh_core::{MountItem, prelude::*};
    use std::{collections::HashMap, sync::Mutex};
    let (mut app, _) = start(Library::default());
    let frames = Arc::new(Mutex::new(HashMap::new()));
    let footer = Arc::new(Mutex::new(None));
    let seen_frames = frames.clone();
    let seen_footer = footer.clone();
    app.observe(move |_, batch| {
        for item in &batch.items {
            match item {
                MountItem::Frame { id, frame } => {
                    seen_frames.lock().unwrap().insert(*id, *frame);
                }
                MountItem::ViewProps { id, props }
                    if props.test_id.as_deref() == Some("project-name-footer") =>
                {
                    *seen_footer.lock().unwrap() = Some(*id);
                }
                _ => {}
            }
        }
    });
    app.set_viewport(Size {
        width: px(390.),
        height: px(844.),
    })
    .unwrap();
    app.set_environment("safe area", |e| e.safe_area.bottom = px(34.))
        .unwrap();
    app.press("id:new-project").unwrap();
    let bounds = || frames.lock().unwrap()[&footer.lock().unwrap().unwrap()];
    let before = bounds();
    app.set_environment("keyboard shown", |e| e.keyboard_height = px(336.))
        .unwrap();
    let above = bounds();
    assert!((before.origin.y.0 - above.origin.y.0 - 302.).abs() < 1.);
    assert_eq!(before.size.height, above.size.height);
    // Resized Android surfaces report no overlap; do not apply it again.
    app.set_environment("keyboard hidden or surface resized", |e| {
        e.keyboard_height = px(0.)
    })
    .unwrap();
    assert!((bounds().origin.y.0 - before.origin.y.0).abs() < 1.);
}

fn start(library: Library) -> (Headless, Arc<MemoryStorage>) {
    let storage = MemoryStorage::new([(STORAGE.into(), serde_json::to_string(&library).unwrap())]);
    let mut config = EngineConfig::new(Root::of::<App>());
    config.environment.locale = nsh_core::platform::Locale::parse("de-DE").unwrap();
    config.storage = Some(storage.clone());
    {
        let mut app = Headless::start(config).unwrap();
        app.wait_for("text:Sammlungen").unwrap();
        (app, storage)
    }
}
fn library(storage: &MemoryStorage) -> Library {
    serde_json::from_str(&storage.entries()[STORAGE]).unwrap()
}
fn assert_same_project_content(actual: Library, mut expected: Library) {
    for p in &mut expected.projects {
        let restored = actual.projects.iter().find(|v| v.id == p.id).unwrap();
        assert!(restored.last_modified() >= p.last_modified());
        p.modified_at = restored.modified_at;
    }
    assert_eq!(actual, expected);
}
fn project(app: &mut Headless) {
    app.press("id:project-1").unwrap();
    app.wait_for("id:project-title").unwrap();
}

#[test]
fn creating_and_renaming_a_project_persists_across_restart() {
    let (mut app, storage) = start(Library::default());
    app.press("id:new-project").unwrap();
    app.clear("id:project-name").unwrap();
    assert!(!app.screen().enabled("text:Speichern").unwrap());
    app.type_text("id:project-name", "Quartett").unwrap();
    app.press("text:Speichern").unwrap();
    app.wait_for("text:Noch keine Seiten").unwrap();
    assert_eq!(library(&storage).projects[0].title, "Quartett");
    app.on_dialog(|_| nsh_testing::DialogAnswer::Press("Umbenennen".into()));
    app.press("id:collection-more").unwrap();
    app.clear("id:project-name").unwrap();
    app.type_text("id:project-name", "Quartett · Probe")
        .unwrap();
    app.press("text:Speichern").unwrap();
    app.back().unwrap();
    app.wait_for("text:Sammlungen").unwrap();
    assert!(
        app.back().is_err(),
        "root must leave back handling to the platform"
    );
    let persisted = library(&storage);
    assert_eq!(persisted.projects[0].title, "Quartett · Probe");
    drop(app);
    let (mut restarted, _) = start(persisted);
    restarted.wait_for("text:Quartett · Probe").unwrap();
}

#[test]
fn selection_and_viewer_position_survive_native_route_back() {
    let (mut app, _) = start(Library::examples());
    project(&mut app);
    app.long_press("id:page-4").unwrap();
    app.press("id:page-1").unwrap();
    share_collection(&mut app);
    app.wait_for("id:share-filename").unwrap();
    app.back().unwrap();
    app.wait_for("id:page-selection-4").unwrap();
    assert!(
        app.screen()
            .accessibility_state("id:page-4")
            .unwrap()
            .selected
    );
    assert!(!app.shows("Zum Verschieben ziehen"));
    assert!(!app.shows("Alle auswählen"));
    app.press("id:selection-done").unwrap();
    app.press("id:page-3").unwrap();
    app.wait_for("text:3 / 4").unwrap();
    app.press("id:viewer-share").unwrap();
    app.wait_for("id:share-filename").unwrap();
    app.back().unwrap();
    app.wait_for("text:3 / 4").unwrap();
    app.back().unwrap();
    app.wait_for("id:project-title").unwrap();
    app.back().unwrap();
    app.wait_for("text:Sammlungen").unwrap();
}

#[test]
fn deleting_a_selection_can_be_undone_without_changing_page_identity() {
    let original = Library::examples();
    let (mut app, storage) = start(original.clone());
    project(&mut app);
    app.long_press("id:page-2").unwrap();
    app.press("id:page-4").unwrap();
    app.press("id:selection-delete").unwrap();
    app.wait_for("id:project-title").unwrap();
    assert_eq!(
        library(&storage).projects[0]
            .pages
            .iter()
            .map(|p| p.id)
            .collect::<Vec<_>>(),
        [1, 3]
    );
    app.press("text:Rückgängig").unwrap();
    app.wait_for("id:project-title").unwrap();
    assert_same_project_content(library(&storage), original);
}

#[test]
fn tapping_a_toast_dismisses_it_without_undoing_the_deletion() {
    let (mut app, storage) = start(Library::examples());
    project(&mut app);
    app.long_press("id:page-2").unwrap();
    app.press("id:selection-delete").unwrap();
    app.wait_for("id:toast").unwrap();
    app.press("id:toast").unwrap();
    app.wait_for_gone("id:toast", std::time::Duration::from_secs(2))
        .unwrap();
    assert!(!app.screen().matches("id:toast").unwrap());
    assert_eq!(library(&storage).projects[0].pages.len(), 3);
}

#[test]
fn undo_timeout_pauses_for_screen_readers_and_resumes_afterwards() {
    let storage = MemoryStorage::new([(
        STORAGE.into(),
        serde_json::to_string(&Library::examples()).unwrap(),
    )]);
    let mut config = EngineConfig::new(Root::of::<App>());
    config.environment.locale = nsh_core::platform::Locale::parse("de-DE").unwrap();
    config.storage = Some(storage.clone());
    config.clock = nsh_core::time::Clock::manual(std::time::SystemTime::UNIX_EPOCH);
    let mut app = Headless::start(config).unwrap();
    app.wait_for("text:Sammlungen").unwrap();
    project(&mut app);
    app.set_environment("screen reader", |e| {
        e.screen_reader = true;
        e.reduce_motion = true;
    })
    .unwrap();
    app.long_press("id:page-2").unwrap();
    app.press("id:selection-delete").unwrap();
    app.wait_for("id:toast").unwrap();
    app.run_for(std::time::Duration::from_secs(10)).unwrap();
    assert!(app.screen().matches("id:toast").unwrap());
    app.set_environment("screen reader off", |e| e.screen_reader = false)
        .unwrap();
    app.run_for(std::time::Duration::from_secs(8)).unwrap();
    assert!(!app.screen().matches("id:toast").unwrap());
    assert_eq!(library(&storage).projects[0].pages.len(), 3);
}

#[test]
fn rotation_keeps_the_original_and_more_does_not_offer_gcs() {
    let mut original = Library::examples();
    original.projects[0].pages[0].source = Some("/private/original.jpg".into());
    original.projects[0].pages[0].output = Some("/private/processed.png".into());
    let (mut app, storage) = start(original);
    project(&mut app);
    app.long_press("id:page-4").unwrap();
    app.press("id:selection-delete").unwrap();
    app.press("id:page-1").unwrap();
    app.press("id:viewer-rotate").unwrap();
    let persisted = library(&storage);
    assert_eq!(persisted.projects[0].pages[0].rotation, 90);
    assert_eq!(
        persisted.projects[0].pages[0].source.as_deref(),
        Some("/private/original.jpg")
    );
    assert!(
        !app.shows("Rückgängig"),
        "undo must not erase a subsequent edit"
    );
    app.on_dialog(move |dialog| {
        let description = format!("{dialog:?}");
        assert!(!description.contains("GCS"));
        assert!(!description.contains("Entzerrung"));
        nsh_testing::DialogAnswer::Press("Abbrechen".into())
    });
    app.press("id:viewer-more").unwrap();
    app.wait_for("id:viewer-rotate").unwrap();
    assert_eq!(library(&storage).projects[0].pages[0].model, Model::Mesh);
    for _ in 0..3 {
        app.press("id:viewer-rotate").unwrap();
    }
    assert_eq!(library(&storage).projects[0].pages[0].rotation, 0);
    assert_eq!(library(&storage).projects[0].pages.len(), 3);
}

#[test]
fn page_order_changes_document_order_and_export_selection() {
    let (mut app, storage) = start(Library::examples());
    project(&mut app);
    app.long_press("id:page-1").unwrap();
    app.drag(
        "id:page-drag-1",
        nsh_core::geometry::point(nsh_core::Points(120.), nsh_core::Points(0.)),
    )
    .unwrap();
    assert_eq!(
        library(&storage).projects[0]
            .pages
            .iter()
            .map(|p| p.id)
            .collect::<Vec<_>>(),
        [2, 1, 3, 4]
    );
    app.press("id:selection-done").unwrap();
    app.long_press("id:page-2").unwrap();
    share_collection(&mut app);
    app.wait_for("id:share-filename").unwrap();
}

#[test]
fn sharing_has_only_a_filename_fixed_pdf_suffix_and_three_size_profiles() {
    let (mut app, _) = start(Library::examples());
    project(&mut app);
    share_collection(&mut app);
    app.wait_for("id:share-filename").unwrap();
    app.wait_for("id:share-extension").unwrap();
    for removed in [
        "Exportieren",
        "Auswahl:",
        "JPEG",
        "Darstellung",
        "Format",
        "Exportvorschau",
        "Volle Ergebnisauflösung",
    ] {
        assert!(!app.shows(removed), "Removed control: {removed}");
    }
    for i in 0..3 {
        let selector = format!("id:share-size-{i}");
        app.press(&selector).unwrap();
        assert!(
            app.screen()
                .accessibility_state(&selector)
                .unwrap()
                .selected
        );
    }
    app.clear("id:share-filename").unwrap();
    app.type_text("id:share-filename", "Probe.PDF").unwrap();
    assert_eq!(
        app.screen().input_text("id:share-filename").unwrap(),
        Some("Probe")
    );
    assert!(!app.screen().enabled("id:share-pdf").unwrap());
}

#[test]
fn camera_cannot_capture_without_a_valid_native_detection_and_back_keeps_project() {
    let (mut app, storage) = start(Library::examples());
    project(&mut app);
    app.press("id:collection-camera").unwrap();
    app.wait_for("text:Kamera starten …").unwrap();
    assert!(!app.screen().enabled("id:shutter").unwrap());
    assert!(!app.screen().enabled("label:Licht").unwrap());
    assert_eq!(
        app.screen()
            .accessibility_state("label:Licht")
            .unwrap()
            .checked,
        Some(nsh_core::Checked::False),
    );
    app.back().unwrap();
    app.wait_for("id:project-title").unwrap();
    assert_eq!(library(&storage).projects[0].pages.len(), 4);
}

#[test]
fn batch_spreads_keep_capture_groups_and_retake_only_the_selected_page() {
    use smude_mobile_ui::model::Project;
    let mut project = Project::new("Sonate".into());
    project.capture(2, Some("/private/spread-a.jpg".into()), None);
    project.capture(1, Some("/private/single-b.jpg".into()), None);
    project.capture(2, Some("/private/spread-c.jpg".into()), None);
    assert_eq!(project.pages.len(), 5);
    assert_eq!(project.pages[0].capture, project.pages[1].capture);
    assert_ne!(project.pages[1].capture, project.pages[2].capture);
    assert_eq!(project.pages[3].capture, project.pages[4].capture);
    assert_eq!(project.pages[0].side, 0);
    assert_eq!(project.pages[1].side, 1);
    project.pages[1].model = Model::Mesh;
    let before = project.pages.clone();
    project.capture(1, Some("/private/retake.jpg".into()), Some(before[1].id));
    assert_eq!(project.pages[1].id, before[1].id);
    assert_eq!(project.pages[1].model, Model::Mesh);
    assert_eq!(
        project.pages[1].source.as_deref(),
        Some("/private/retake.jpg")
    );
    assert_ne!(project.pages[1].capture, before[1].capture);
    for i in [0, 2, 3, 4] {
        assert_eq!(project.pages[i], before[i]);
    }
}

#[test]
fn projects_use_long_press_multi_selection_and_undo() {
    let mut original = Library::examples();
    let mut second = smude_mobile_ui::model::Project::new("Zweites Projekt".into());
    second.id = 2;
    original.projects.push(second);
    let (mut app, storage) = start(original.clone());
    assert!(!app.screen().matches("id:delete-projects").unwrap());
    app.long_press("id:project-1").unwrap();
    app.wait_for("id:delete-projects").unwrap();
    assert!(
        app.screen()
            .accessibility_state("id:project-1")
            .unwrap()
            .selected
    );
    app.press("id:project-2").unwrap();
    app.press("id:delete-projects").unwrap();
    app.wait_for("text:Noch keine Sammlungen").unwrap();
    assert!(library(&storage).projects.is_empty());
    app.press("text:Rückgängig").unwrap();
    app.wait_for("id:project-1").unwrap();
    assert_same_project_content(library(&storage), original);
    app.long_press("id:project-1").unwrap();
    app.press("id:delete-projects").unwrap();
    app.wait_for("id:toast").unwrap();
    assert_eq!(library(&storage).projects.len(), 1);
}

#[test]
fn local_blur_warning_can_be_kept_or_replaced_without_losing_the_original() {
    let mut original = Library::examples();
    let page = &mut original.projects[0].pages[1];
    page.quality = smude_geometry::detection::quality::CaptureQuality {
        retake_suggested: true,
        bands: vec![smude_geometry::detection::quality::FocusBand {
            band: 0,
            edge_spread_staff_spaces: 0.15,
            supports: 150,
            blurred: true,
        }],
    };
    page.review = true;
    page.source = Some("sources/original.jpg".into());
    let (mut app, storage) = start(original.clone());
    project(&mut app);
    app.press("id:page-2").unwrap();
    app.wait_for("id:focus-warning").unwrap();
    app.press("id:retake-soft-page").unwrap();
    app.wait_for("text:Seite 2 ersetzen").unwrap();
    assert_eq!(
        library(&storage).projects[0].pages[1],
        original.projects[0].pages[1]
    );
    app.back().unwrap();
    app.press("text:Behalten").unwrap();
    app.wait_for_gone("id:focus-warning", std::time::Duration::from_secs(1))
        .unwrap();
    let kept = &library(&storage).projects[0].pages[1];
    assert!(!kept.review);
    assert_eq!(kept.source, original.projects[0].pages[1].source);
    assert_eq!(kept.quality, original.projects[0].pages[1].quality);
}

#[test]
fn project_browser_centers_chevrons_sorts_edits_and_remembers_grid() {
    use nsh_core::{MountItem, prelude::*};
    use smude_mobile_ui::model::Project;
    use std::{collections::HashMap, sync::Mutex};
    let mut a = Project::new("Älteres Projekt".into());
    a.id = 101;
    a.modified_at = 100;
    let mut b = Project::new("Neueres Projekt".into());
    b.id = 102;
    b.modified_at = 200;
    let mut c = Project::new("Dritte Sammlung".into());
    c.id = 103;
    c.modified_at = 50;
    let mut d = Project::new("Vierte Sammlung".into());
    d.id = 104;
    d.modified_at = 25;
    let storage = MemoryStorage::new([(
        STORAGE.into(),
        serde_json::to_string(&Library {
            projects: vec![a, b, c, d],
            imports: vec![],
        })
        .unwrap(),
    )]);
    let mut config = EngineConfig::new(Root::of::<App>());
    config.environment.locale = nsh_core::platform::Locale::parse("de-DE").unwrap();
    config.storage = Some(storage.clone());
    let mut app = Headless::start(config).unwrap();
    let frames = Arc::new(Mutex::new(HashMap::new()));
    let ids = Arc::new(Mutex::new(HashMap::new()));
    let seen = frames.clone();
    let named = ids.clone();
    app.observe(move |_, batch| {
        for item in &batch.items {
            match item {
                MountItem::Frame { id, frame } => {
                    seen.lock().unwrap().insert(*id, *frame);
                }
                MountItem::ViewProps { id, props } => {
                    if let Some(name) = &props.test_id {
                        named.lock().unwrap().insert(name.to_string(), *id);
                    }
                }
                _ => {}
            }
        }
    });
    app.set_viewport(Size {
        width: px(390.),
        height: px(844.),
    })
    .unwrap();
    app.wait_for("id:project-list").unwrap();
    let bounds = |name: &str| frames.lock().unwrap()[&ids.lock().unwrap()[name]];
    let newer = bounds("project-102");
    let older = bounds("project-101");
    assert!(newer.origin.y < older.origin.y);
    assert!(
        newer.size.height.0 <= 90.,
        "list rows should be compact: {newer:?}"
    );
    let chevron = bounds("project-chevron-102");
    assert!(
        (chevron.origin.y.0 + chevron.size.height.0 * 0.5 - newer.size.height.0 * 0.5).abs() < 1.
    );
    assert!(app.screen().matches("id:project-placeholder-101").unwrap());
    app.press("id:project-101").unwrap();
    app.wait_for("id:project-title").unwrap();
    app.on_dialog(|_| nsh_testing::DialogAnswer::Press("Umbenennen".into()));
    app.press("id:collection-more").unwrap();
    app.clear("id:project-name").unwrap();
    app.type_text("id:project-name", "Jetzt bearbeitet")
        .unwrap();
    app.press("text:Speichern").unwrap();
    app.back().unwrap();
    app.wait_for("id:project-list").unwrap();
    assert!(bounds("project-101").origin.y < bounds("project-102").origin.y);
    let before = library(&storage);
    app.press("id:project-view-grid").unwrap();
    app.wait_for("id:project-grid").unwrap();
    let first = bounds("project-101");
    let second = bounds("project-102");
    assert!((first.origin.y.0 - second.origin.y.0).abs() < 1.);
    assert!(second.origin.x > first.origin.x);
    let third = bounds("project-103");
    let fourth = bounds("project-104");
    assert!((third.origin.y.0 - first.origin.y.0).abs() < 1. && third.origin.x > second.origin.x);
    assert!(
        fourth.origin.y > first.origin.y,
        "grid must wrap after three cards"
    );
    let cover = bounds("project-cover-101");
    assert!(cover.size.height.0 > 110., "larger previews: {cover:?}");
    assert!(
        first.size.height.0 - cover.size.height.0 < 65.,
        "no oversized metadata footer: {first:?}, {cover:?}"
    );
    assert!(!app.screen().matches("id:project-chevron-101").unwrap());
    assert_eq!(
        library(&storage),
        before,
        "switching layout must not edit projects"
    );
    assert_eq!(storage.entries()["smude-project-view"], "grid");
    drop(app);
    let mut config = EngineConfig::new(Root::of::<App>());
    config.environment.locale = nsh_core::platform::Locale::parse("de-DE").unwrap();
    config.storage = Some(storage);
    let mut restarted = Headless::start(config).unwrap();
    restarted.wait_for("id:project-grid").unwrap();
}

#[test]
fn language_follows_the_os_and_keeps_collection_contents_on_change() {
    use nsh_core::platform::Locale;
    for (tag, heading, new_name) in [
        ("en-GB", "Collections", "New collection"),
        ("de-DE", "Sammlungen", "Neue Sammlung"),
        ("pt-BR", "Coleções", "Nova coleção"),
        ("pt-PT", "Coleções", "Nova coleção"),
        ("fr-FR", "Collections", "New collection"),
    ] {
        let original = Library::examples();
        let storage =
            MemoryStorage::new([(STORAGE.into(), serde_json::to_string(&original).unwrap())]);
        let mut config = EngineConfig::new(Root::of::<App>());
        config.storage = Some(storage.clone());
        config.environment.locale = Locale::parse(tag).unwrap();
        let mut app = Headless::start(config).unwrap();
        app.wait_for(&format!("text:{heading}")).unwrap();
        app.press("id:new-project").unwrap();
        app.wait_for(&format!("text:{new_name}")).unwrap();
        app.back().unwrap();
        app.set_environment("language changed", |env| {
            env.locale = Locale::parse("pt-BR").unwrap()
        })
        .unwrap();
        app.wait_for("text:Coleções").unwrap();
        assert_same_project_content(library(&storage), original);
    }
}

#[test]
fn camera_done_and_back_remain_available_during_an_outstanding_photo() {
    use nsh_core::{EventKind, EventPayload, MountItem};
    use std::sync::Mutex;
    let (mut app, _) = start(Library::examples());
    project(&mut app);
    let camera = Arc::new(Mutex::new(None));
    let found = camera.clone();
    app.observe(move |_, batch| {
        for item in &batch.items {
            if let MountItem::Component { id, name, .. } = item
                && name.as_ref() == "smude.camera"
            {
                *found.lock().unwrap() = Some(*id);
            }
        }
    });
    app.press("id:collection-camera").unwrap();
    app.wait_for("id:camera-done").unwrap();
    let id = camera.lock().unwrap().unwrap();
    app.engine().control().dispatch_event(id,EventKind::NativeEvent,EventPayload::Native{name:"status".into(),data:serde_json::json!({"ready":false,"capturing":true,"width":1,"height":1,"polygons":[]})});
    app.set_environment("deliver acquisition event", |_| {})
        .unwrap();
    assert!(app.screen().enabled("id:camera-done").unwrap());
    app.press("id:camera-done").unwrap();
    app.wait_for("id:project-title").unwrap();
}

#[test]
fn viewer_toggle_is_exclusive_and_pagination_controls_share_one_center_line() {
    use nsh_core::MountItem;
    use std::{collections::HashMap, sync::Mutex};
    let (mut app, _) = start(Library::examples());
    project(&mut app);
    let frames = Arc::new(Mutex::new(HashMap::new()));
    let names = Arc::new(Mutex::new(HashMap::new()));
    let f = frames.clone();
    let n = names.clone();
    app.observe(move |_, batch| {
        for item in &batch.items {
            match item {
                MountItem::Frame { id, frame } => {
                    f.lock().unwrap().insert(*id, *frame);
                }
                MountItem::ViewProps { id, props } => {
                    if let Some(name) = &props.test_id {
                        n.lock().unwrap().insert(name.to_string(), *id);
                    }
                }
                _ => {}
            }
        }
    });
    app.press("id:page-2").unwrap();
    app.wait_for("id:page-navigation").unwrap();
    app.press("label:Original").unwrap();
    assert_eq!(
        app.screen()
            .accessibility_state("label:Original")
            .unwrap()
            .pressed,
        Some(nsh_core::Checked::True)
    );
    assert_eq!(
        app.screen()
            .accessibility_state("label:Bearbeitet")
            .unwrap()
            .pressed,
        Some(nsh_core::Checked::False)
    );
    app.press("label:Bearbeitet").unwrap();
    assert_eq!(
        app.screen()
            .accessibility_state("label:Bearbeitet")
            .unwrap()
            .pressed,
        Some(nsh_core::Checked::True)
    );
    let middle = |name: &str| {
        let frame = frames.lock().unwrap()[&names.lock().unwrap()[name]];
        frame.origin.y.0 + frame.size.height.0 * 0.5
    };
    assert!((middle("page-previous") - middle("page-position")).abs() < 0.5);
    assert!((middle("page-next") - middle("page-position")).abs() < 0.5);
    assert!(!app.shows("auf diesem Gerät"));
}

#[test]
fn home_camera_opens_without_naming_a_collection_and_empty_done_leaves_no_draft() {
    let (mut app, storage) = start(Library::default());
    app.press("id:home-camera").unwrap();
    app.wait_for("id:camera-done").unwrap();
    assert_eq!(library(&storage).projects.len(), 1);
    assert!(library(&storage).projects[0].draft);
    app.press("id:camera-done").unwrap();
    app.wait_for("text:Sammlungen").unwrap();
    assert!(library(&storage).projects.is_empty());
}
#[test]
fn collection_grid_grows_from_three_phone_columns_to_more_tablet_columns() {
    use nsh_core::{MountItem, prelude::*};
    use std::{collections::HashMap, sync::Mutex};
    let (mut app, _) = start(Library::examples());
    let frames = Arc::new(Mutex::new(HashMap::new()));
    let names = Arc::new(Mutex::new(HashMap::new()));
    let f = frames.clone();
    let n = names.clone();
    app.observe(move |_, batch| {
        for item in &batch.items {
            match item {
                MountItem::Frame { id, frame } => {
                    f.lock().unwrap().insert(*id, *frame);
                }
                MountItem::ViewProps { id, props } => {
                    if let Some(name) = &props.test_id {
                        n.lock().unwrap().insert(name.to_string(), *id);
                    }
                }
                _ => {}
            }
        }
    });
    app.set_viewport(Size {
        width: px(390.),
        height: px(844.),
    })
    .unwrap();
    project(&mut app);
    let frame = |name: &str| frames.lock().unwrap()[&names.lock().unwrap()[name]];
    assert_eq!(frame("page-drag-1").origin.y, frame("page-drag-3").origin.y);
    assert!(frame("page-drag-4").origin.y > frame("page-drag-1").origin.y);
    let card = frame("page-drag-1");
    assert!(
        card.size.height.0 - card.size.width.0 * 1.414 < 60.,
        "No fixed-height blank space under the thumbnail: {card:?}"
    );
    app.set_viewport(Size {
        width: px(1024.),
        height: px(1366.),
    })
    .unwrap();
    assert_eq!(frame("page-drag-1").origin.y, frame("page-drag-4").origin.y);
    assert!(!app.shows("Seiten hinzufügen"));
    assert!(!app.shows("Fotos importieren"));
    app.wait_for("id:collection-camera").unwrap();
    app.wait_for("id:collection-more").unwrap();
}

// Real component touch routing, including two independent pointers. This does
// not pretend the host's one-pointer pan helper is a multitouch gesture.
struct ReorderTouches {
    app: Headless,
    storage: Arc<MemoryStorage>,
    nodes: std::collections::HashMap<String, nsh_core::NodeId>,
}
impl ReorderTouches {
    fn new() -> Self {
        use nsh_core::{MountItem, prelude::*};
        use std::sync::Mutex;
        let mut lib = Library::examples();
        let template = lib.projects[0].pages[0].clone();
        lib.projects[0].pages = (1..=30)
            .map(|id| {
                let mut p = template.clone();
                p.id = id;
                p.capture = id;
                p
            })
            .collect();
        let storage = MemoryStorage::new([(STORAGE.into(), serde_json::to_string(&lib).unwrap())]);
        let mut config = EngineConfig::new(Root::of::<App>());
        config.environment.locale = nsh_core::platform::Locale::parse("de-DE").unwrap();
        config.storage = Some(storage.clone());
        config.clock = nsh_core::time::Clock::manual(std::time::SystemTime::UNIX_EPOCH);
        let mut app = Headless::start(config).unwrap();
        app.wait_for("text:Sammlungen").unwrap();
        let nodes = Arc::new(Mutex::new(std::collections::HashMap::new()));
        let seen = nodes.clone();
        app.observe(move |_, batch| {
            for item in &batch.items {
                if let MountItem::ViewProps { id, props } = item
                    && let Some(name) = &props.test_id
                {
                    seen.lock().unwrap().insert(name.to_string(), *id);
                }
            }
        });
        app.set_viewport(size(px(390.), px(844.))).unwrap();
        project(&mut app);
        app.run_for(std::time::Duration::from_millis(50)).unwrap();
        let nodes = nodes.lock().unwrap().clone();
        Self {
            app,
            storage,
            nodes,
        }
    }
    fn event(
        &mut self,
        phase: nsh_core::TouchPhase,
        points: &[(u32, f32, f32, bool)],
        primary_card: bool,
    ) {
        use nsh_core::{
            EventKind, EventPayload, Points, ROOT_ID, TouchEvent, TouchPoint, geometry::point,
        };
        let scroll = self.nodes["page-scroll"];
        let card = self.nodes["page-drag-1"];
        let payload = TouchEvent {
            phase,
            timestamp: 0.,
            activated: None,
            points: points
                .iter()
                .map(|&(id, x, y, changed)| TouchPoint {
                    id,
                    target: if id == 7 { card } else { scroll },
                    path: if phase == nsh_core::TouchPhase::Start && changed {
                        if id == 7 && primary_card {
                            vec![card, scroll]
                        } else {
                            vec![scroll]
                        }
                    } else {
                        vec![]
                    },
                    page: point(Points(x), Points(y)),
                    location: point(Points(x), Points(y)),
                    changed,
                })
                .collect(),
        };
        self.app.engine().control().dispatch_event(
            ROOT_ID,
            EventKind::Touch,
            EventPayload::Touch(Box::new(payload)),
        );
        self.app
            .run_for(std::time::Duration::from_millis(20))
            .unwrap();
    }
    fn hold(&mut self) {
        self.event(nsh_core::TouchPhase::Start, &[(7, 70., 250., true)], true);
        let target = self
            .app
            .screen()
            .press_target("id:page-1")
            .unwrap()
            .unwrap();
        self.app.engine().control().dispatch_event(
            target,
            nsh_core::EventKind::LongPress,
            nsh_core::EventPayload::None,
        );
        self.app
            .run_for(std::time::Duration::from_millis(20))
            .unwrap();
    }
    fn last_scroll(&self) -> f32 {
        self.app
            .screen()
            .commands()
            .iter()
            .filter_map(|(_, c)| match c {
                nsh_core::Command::ScrollTo { y, .. } => Some(y.0),
                _ => None,
            })
            .next_back()
            .unwrap_or(0.)
    }
    fn order(&self) -> Vec<u64> {
        library(&self.storage).projects[0]
            .pages
            .iter()
            .map(|p| p.id)
            .collect()
    }
}
#[test]
fn second_finger_scrolls_while_primary_holds_and_drop_uses_scrolled_position() {
    use nsh_core::TouchPhase::*;
    let mut t = ReorderTouches::new();
    t.hold();
    let before = t.order();
    t.event(
        Start,
        &[(7, 70., 250., false), (19, 260., 550., true)],
        false,
    );
    t.event(
        Move,
        &[(7, 70., 250., false), (19, 260., 230., true)],
        false,
    );
    assert!(
        (t.last_scroll() - 320.).abs() < 1.,
        "second finger must scroll the content, not take over dragging: {}",
        t.last_scroll()
    );
    assert_eq!(
        t.order(),
        before,
        "preview must not persist before primary release"
    );
    t.event(End, &[(7, 70., 250., true), (19, 260., 230., false)], false);
    let after = t.order();
    assert_ne!(
        after, before,
        "stationary primary drops in its new content row"
    );
    assert_eq!(after.iter().filter(|&&id| id == 1).count(), 1);
    assert!(after.iter().position(|&id| id == 1).unwrap() >= 3);
    t.event(Move, &[(19, 260., 130., true)], false);
    assert!((t.last_scroll() - 420.).abs() < 1.);
    t.event(End, &[(19, 260., 130., true)], false);
    assert_eq!(
        t.order(),
        after,
        "secondary release must not commit another move"
    );
    t.app.press("id:selection-done").unwrap();
    t.app
        .run_for(std::time::Duration::from_millis(200))
        .unwrap();
    t.app.press("id:page-1").unwrap();
    t.app.wait_for("id:page-pager").unwrap();
}
#[test]
fn held_page_autoscrolls_at_both_edges_and_cancel_keeps_original_order() {
    use nsh_core::TouchPhase::*;
    let mut t = ReorderTouches::new();
    t.hold();
    let before = t.order();
    t.event(Move, &[(7, 70., 810., true)], false);
    t.app
        .run_for(std::time::Duration::from_millis(600))
        .unwrap();
    let down = t.last_scroll();
    assert!(
        down > 100.,
        "holding near bottom should scroll without another move: {down}"
    );
    t.event(Move, &[(7, 70., 2., true)], false);
    t.app
        .run_for(std::time::Duration::from_millis(800))
        .unwrap();
    assert!(t.last_scroll() < down, "top edge should scroll back up");
    t.event(Cancel, &[(7, 70., 2., true)], false);
    assert_eq!(t.order(), before);
    let stopped = t.last_scroll();
    t.app
        .run_for(std::time::Duration::from_millis(400))
        .unwrap();
    assert_eq!(t.last_scroll(), stopped, "cancel stops the autoscroll loop");
}

#[test]
fn empty_home_and_collection_offer_actions_and_search_has_a_clearable_no_results_state() {
    let (mut app, _) = start(Library::default());
    assert!(app.screen().matches("id:empty-home-camera").unwrap());
    assert!(!app.screen().matches("id:search").unwrap());
    assert!(!app.screen().matches("id:project-view-grid").unwrap());
    app.press("id:empty-home-create").unwrap();
    app.type_text("id:project-name", "Morning").unwrap();
    app.press("text:Speichern").unwrap();
    assert!(app.screen().enabled("id:empty-collection-camera").unwrap());
    assert!(app.screen().enabled("id:empty-collection-import").unwrap());
    app.back().unwrap();
    app.type_text("id:search", "does-not-exist").unwrap();
    app.wait_for("id:search-empty").unwrap();
    assert!(!app.screen().matches("text:Unzugeordnete Scans").unwrap());
    app.press("id:clear-search").unwrap();
    assert!(!app.screen().matches("id:search-empty").unwrap());
    assert!(app.shows("Morning"));
}
#[test]
fn selected_share_has_exact_scope_and_deselecting_all_never_shares_the_collection() {
    let (mut app, _) = start(Library::examples());
    project(&mut app);
    app.long_press("id:page-1").unwrap();
    app.press("id:page-2").unwrap();
    app.on_dialog(|d| {
        assert!(d.buttons.iter().any(|b| b.text == "2 Seiten teilen"));
        assert!(!d.buttons.iter().any(|b| b.text == "Umbenennen"));
        nsh_testing::DialogAnswer::Press("2 Seiten teilen".into())
    });
    app.press("id:collection-more").unwrap();
    app.wait_for("text:2 Seiten teilen").unwrap();
    assert!(app.screen().matches("id:share-preparing").unwrap());
    app.back().unwrap();
    app.press("id:page-1").unwrap();
    app.press("id:page-2").unwrap();
    assert!(!app.screen().enabled("id:selection-delete").unwrap());
    assert!(!app.screen().enabled("id:collection-more").unwrap());
    assert!(!app.screen().matches("id:share-filename").unwrap());
}
#[test]
fn viewer_disables_missing_results_and_never_invents_demo_music() {
    use smude_mobile_ui::model::PageStatus;
    let mut lib = Library::examples();
    lib.projects[0].pages[0].output = None;
    lib.projects[0].pages[0].status = PageStatus::Failed;
    lib.projects[0].pages[1].output = None;
    lib.projects[0].pages[1].source = None;
    let (mut app, _) = start(lib);
    project(&mut app);
    app.press("id:page-1").unwrap();
    assert_eq!(
        app.screen()
            .accessibility_state("label:Original")
            .unwrap()
            .pressed,
        Some(nsh_core::Checked::True)
    );
    assert!(!app.screen().enabled("text:Bearbeitet").unwrap());
    assert!(!app.screen().enabled("id:viewer-share").unwrap());
    app.press("id:page-next").unwrap();
    app.wait_for("id:image-placeholder").unwrap();
    assert!(!app.screen().enabled("text:Original").unwrap());
    assert!(!app.screen().enabled("text:Bearbeitet").unwrap());
}
#[test]
fn draft_creation_has_a_distinct_action_title_and_direct_assignment() {
    let mut lib = Library::examples();
    lib.projects[0].draft = true;
    let (mut app, storage) = start(lib);
    app.wait_for("text:Unzugeordnete Scans").unwrap();
    project(&mut app);
    app.on_dialog(|d| {
        assert!(
            d.buttons
                .iter()
                .any(|b| b.text == "＋ Neue Sammlung erstellen …")
        );
        nsh_testing::DialogAnswer::Press("＋ Neue Sammlung erstellen …".into())
    });
    app.press("id:draft-assign").unwrap();
    app.wait_for("text:Neue Sammlung").unwrap();
    assert!(!app.shows("Umbenennen"));
    app.type_text("id:project-name", "Collected").unwrap();
    app.press("text:Speichern").unwrap();
    assert!(!library(&storage).projects[0].draft);
    assert_eq!(library(&storage).projects[0].pages.len(), 4);
}
#[test]
fn camera_flash_follows_native_capture_without_blocking_controls_or_retake_counters() {
    use nsh_core::{EventKind, EventPayload, MountItem, animation::AnimationOp};
    use std::sync::Mutex;
    let (mut app, _) = start(Library::examples());
    project(&mut app);
    let camera = Arc::new(Mutex::new(None));
    let ops = Arc::new(Mutex::new(Vec::new()));
    let c = camera.clone();
    let a = ops.clone();
    app.observe(move |_, batch| {
        for item in &batch.items {
            match item {
                MountItem::Component { id, name, .. } if name.as_ref() == "smude.camera" => {
                    *c.lock().unwrap() = Some(*id)
                }
                MountItem::Animation(op) => a.lock().unwrap().push(op.clone()),
                _ => {}
            }
        }
    });
    app.press("id:collection-camera").unwrap();
    assert!(!app.screen().matches("id:camera-review").unwrap());
    assert!(
        !ops.lock()
            .unwrap()
            .iter()
            .any(|op| matches!(op,AnimationOp::Set{value,..}if *value==0.65))
    );
    let node = camera.lock().unwrap().unwrap();
    app.engine().control().dispatch_event(
        node,
        EventKind::NativeEvent,
        EventPayload::Native {
            name: "capture_started".into(),
            data: serde_json::json!({"pages":2}),
        },
    );
    app.set_environment("native capture accepted", |_| {})
        .unwrap();
    assert!(
        ops.lock()
            .unwrap()
            .iter()
            .any(|op| matches!(op,AnimationOp::Set{value,..}if *value==0.65))
    );
    assert!(app.screen().enabled("id:camera-done").unwrap());
    assert!(app.screen().enabled("id:capture-stack").unwrap());
    assert!(app.screen().matches("label:4 Seiten ansehen").unwrap());
    app.engine().control().dispatch_event(node,EventKind::NativeEvent,EventPayload::Native{name:"status".into(),data:serde_json::json!({"reason":"permission","message":"Permission required","ready":false})});
    app.set_environment("permission missing", |_| {}).unwrap();
    app.wait_for("id:camera-settings").unwrap();
    app.wait_for("id:camera-import-fallback").unwrap();
    app.press("id:camera-done").unwrap();
    app.press("id:page-2").unwrap();
    app.press("id:viewer-retake").unwrap();
    app.wait_for("text:Seite 2 ersetzen").unwrap();
    assert!(!app.screen().matches("id:camera-review").unwrap());
    assert!(!app.shows("0 Seiten"));
}

#[test]
fn processing_overlay_keeps_collection_tiles_and_grid_positions_unchanged() {
    use nsh_core::{MountItem, prelude::*};
    use smude_mobile_ui::model::PageStatus;
    use std::{collections::HashMap, sync::Mutex};
    let mut reference = None;
    for status in [
        PageStatus::Ready,
        PageStatus::Importing,
        PageStatus::Pending,
        PageStatus::Processing,
        PageStatus::Failed,
    ] {
        let mut lib = Library::examples();
        lib.projects[0].pages[0].status = status;
        let (mut app, _) = start(lib);
        let frames = Arc::new(Mutex::new(HashMap::new()));
        let names = Arc::new(Mutex::new(HashMap::new()));
        let f = frames.clone();
        let n = names.clone();
        app.observe(move |_, batch| {
            for item in &batch.items {
                match item {
                    MountItem::Frame { id, frame } => {
                        f.lock().unwrap().insert(*id, *frame);
                    }
                    MountItem::ViewProps { id, props } => {
                        if let Some(name) = &props.test_id {
                            n.lock().unwrap().insert(name.to_string(), *id);
                        }
                    }
                    _ => {}
                }
            }
        });
        app.set_viewport(size(px(390.), px(844.))).unwrap();
        assert_eq!(
            app.screen().matches("id:project-dim-1").unwrap(),
            status != PageStatus::Ready,
            "Home shade for {status:?}"
        );
        project(&mut app);
        assert_eq!(
            app.screen().matches("id:page-dim-1").unwrap(),
            status != PageStatus::Ready
        );
        let bounds = |name: &str| frames.lock().unwrap()[&names.lock().unwrap()[name]];
        let tiles = [bounds("page-1"), bounds("page-2")];
        if let Some(expected) = reference {
            assert_eq!(
                tiles, expected,
                "Progress changed the collection layout for {status:?}"
            );
        } else {
            reference = Some(tiles);
        }
        assert_eq!(
            app.screen()
                .matches("id:page-processing-indicator")
                .unwrap(),
            matches!(
                status,
                PageStatus::Importing | PageStatus::Pending | PageStatus::Processing
            )
        );
        assert!(!app.screen().matches("id:processing-progress").unwrap());
    }
}
