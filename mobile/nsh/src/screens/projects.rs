//! Compact project browser; presentation preference is separate from project edits.
use super::page_source;
use crate::i18n::{Key, t};
use crate::{
    model::Project,
    state::{Route, State},
    ui::*,
};
use nsh_core::prelude::*;

pub fn projects(s: State) -> Screen {
    let query = signal(String::new());
    let selecting = signal(false);
    let selected = signal(Vec::<u64>::new());
    let has_matches = move || {
        s.library
            .with(|l| l.projects.iter().any(|p| visible(p, &query.get())))
    };
    home_screen(s, "Smude")
        .header_right(
            when(
                move || selecting.get(),
                move || {
                    trash_button(
                        move || crate::i18n::delete_collections(selected.with(|ids| ids.len())),
                        move || {
                            s.delete_projects(&selected.get_untracked());
                            selected.set(vec![]);
                            selecting.set(false);
                        },
                    )
                    .test_id("delete-projects")
                    .accessibility_state(move || AccessibilityState {
                        disabled: selected.with(|ids| ids.is_empty()),
                        ..Default::default()
                    })
                },
            )
            .otherwise(move || {
                view()
                    .flex_row()
                    .gap_1()
                    .child(
                        header_icon(t(Key::ScanPages), ActionIcon::Scan, move || {
                            s.quick_capture()
                        })
                        .test_id("home-camera")
                        .disabled(move || !s.ready.get()),
                    )
                    .child(
                        plus_button(t(Key::NewCollection), move || s.push(Route::Name(None)))
                            .test_id("new-project")
                            .disabled(move || !s.ready.get()),
                    )
            }),
        )
        .child(
            scroll()
                .keyboard_taps(nsh_core::KeyboardTaps::Handled)
                .flex_1()
                .child(
                    view()
                        .p_4()
                        .gap_3()
                        .child(
                            view()
                                .flex_row()
                                .items_center()
                                .justify_between()
                                .child(
                                    text(t(Key::Collections))
                                        .text_size(px(24.))
                                        .font_weight(FontWeight::SEMIBOLD),
                                )
                                .child(when(
                                    move || s.library.with(|l| !l.projects.is_empty()),
                                    move || {
                                        native_controls::layout_segments(
                                            move || usize::from(s.project_grid.get()),
                                            move |index| s.set_project_grid(index == 1),
                                        )
                                    },
                                )),
                        )
                        .child(when(
                            move || s.library.with(|l| !l.projects.is_empty()),
                            move || {
                                nsh_base::input()
                                    .test_id("search")
                                    .value(query)
                                    .on_change(move |v: &str| query.set(v.to_owned()))
                                    .placeholder(t(Key::SearchCollections))
                                    .h(px(touch_target()))
                                    .px_3()
                                    .rounded(px(10.))
                                    .bg(rgb(RAISED))
                                    .text_color(rgb(0xffffff))
                            },
                        ))
                        .child(when(
                            move || {
                                s.library.with(|l| {
                                    l.projects
                                        .iter()
                                        .any(|p| p.draft && visible(p, &query.get()))
                                })
                            },
                            move || {
                                view().gap_2().child(small(t(Key::UnfiledScans))).child(
                                    when(
                                        move || s.project_grid.get(),
                                        move || {
                                            collection(s, query, selecting, selected, true, true)
                                        },
                                    )
                                    .otherwise(move || {
                                        collection(s, query, selecting, selected, false, true)
                                    }),
                                )
                            },
                        ))
                        .child(
                            when(
                                move || s.project_grid.get(),
                                move || collection(s, query, selecting, selected, true, false),
                            )
                            .otherwise(move || {
                                collection(s, query, selecting, selected, false, false)
                            }),
                        )
                        .child(when(
                            move || s.library.with(|l| l.projects.is_empty()),
                            move || {
                                view()
                                    .test_id("home-empty")
                                    .py_5()
                                    .gap_3()
                                    .child(small(t(Key::NoCollectionsYet)))
                                    .child(
                                        action(t(Key::ScanPages), true, move || s.quick_capture())
                                            .test_id("empty-home-camera")
                                            .disabled(move || !s.ready.get()),
                                    )
                                    .child(
                                        action(t(Key::NewCollection), false, move || {
                                            s.push(Route::Name(None))
                                        })
                                        .test_id("empty-home-create")
                                        .disabled(move || !s.ready.get()),
                                    )
                            },
                        ))
                        .child(when(
                            move || !query.get().trim().is_empty() && !has_matches(),
                            move || {
                                view()
                                    .test_id("search-empty")
                                    .py_5()
                                    .gap_2()
                                    .child(small(t(Key::NoCollectionsFound)))
                                    .child(
                                        link(t(Key::ClearSearch), move || query.set(String::new()))
                                            .test_id("clear-search"),
                                    )
                            },
                        )),
                ),
        )
        .child(when(
            move || selecting.get(),
            move || {
                bottom(
                    action(t(Key::Done), false, move || {
                        selected.set(vec![]);
                        selecting.set(false);
                    })
                    .test_id("project-selection-done"),
                )
            },
        ))
}
fn visible(p: &Project, query: &str) -> bool {
    (!p.draft || !p.pages.is_empty())
        && p.title
            .to_lowercase()
            .contains(&query.trim().to_lowercase())
}
fn collection(
    s: State,
    query: Signal<String>,
    selecting: Signal<bool>,
    selected: Signal<Vec<u64>>,
    grid: bool,
    drafts: bool,
) -> View {
    let width = signal(320f32);
    view()
        .test_id(if drafts {
            "scan-grid"
        } else if grid {
            "project-grid"
        } else {
            "project-list"
        })
        .w_full()
        .flex_row()
        .flex_wrap()
        .gap_2()
        .on_layout(move |b: &Bounds<Points>| width.set(b.size.width.0))
        .child(each(
            move || {
                let q = query.get();
                s.library.with(|l| {
                    l.recent_projects()
                        .into_iter()
                        .filter(|p| p.draft == drafts && visible(p, &q))
                        .collect::<Vec<_>>()
                })
            },
            |p: &Project| p.id,
            move |p, _| {
                let columns = move || {
                    if grid {
                        grid_columns(width.get()) as f32
                    } else {
                        1.
                    }
                };
                card(s, p, selecting, selected, grid).w(move || {
                    px(((width.get() - 8. * (columns() - 1.)) / columns())
                        .floor()
                        .max(0.))
                })
            },
        ))
}
fn card(
    s: State,
    p: ReadSignal<Project>,
    selecting: Signal<bool>,
    selected: Signal<Vec<u64>>,
    grid: bool,
) -> View {
    let id = p.get_untracked().id;
    let base = view()
        .test_id(&format!("project-{id}"))
        .role(AccessibilityRole::Button)
        .label(move || {
            let p = p.get();
            format!(
                "{}, {}, {}",
                p.title,
                page_count(p.pages.len()),
                modified_label(&p)
            )
        })
        .accessibility_state(move || AccessibilityState {
            selected: selected.with(|ids| ids.contains(&id)),
            ..Default::default()
        })
        .on_long_press(move || {
            selected.set(vec![id]);
            selecting.set(true);
        })
        .on_context_menu(move |_: &ContextMenuEvent| {
            selected.set(vec![id]);
            selecting.set(true);
        })
        .on_press(move || {
            if selecting.get() {
                selected.update(|ids| {
                    if ids.contains(&id) {
                        ids.retain(|v| *v != id);
                    } else {
                        ids.push(id);
                    }
                });
            } else {
                s.push(Route::Project(id));
            }
        })
        .relative()
        .p_3()
        .gap_3()
        .rounded(px(12.))
        .border_1()
        .border_color(move || {
            rgb(if selected.with(|ids| ids.contains(&id)) {
                GOLD
            } else {
                EDGE
            })
        })
        .bg_linear(deg(135.))
        .gradient_from(rgb(RAISED))
        .gradient_to(rgb(PANEL));
    let base = if grid {
        base.flex_col().p_2().gap_2()
    } else {
        base.flex_row().items_center()
    };
    let info = view()
        .gap_1()
        .child(
            text(move || p.get().title)
                .text_size(px(if grid { 13. } else { 16. }))
                .font_weight(FontWeight::SEMIBOLD)
                .number_of_lines(1),
        )
        .child(
            text(move || {
                let p = p.get();
                if grid {
                    page_count(p.pages.len())
                } else {
                    format!("{} · {}", page_count(p.pages.len()), modified_label(&p))
                }
            })
            .text_size(px(12.))
            .text_color(rgb(MUTED))
            .number_of_lines(1),
        );
    let info = if grid {
        info.w_full()
    } else {
        info.flex_1().min_w(px(0.))
    };
    base.child(cover(s, p, grid))
        .child(info)
        .child(when(
            move || !grid,
            move || {
                view()
                    .test_id(&format!("project-chevron-{id}"))
                    .w(px(20.))
                    .h(px(20.))
                    .flex_shrink_0()
                    .opacity(move || if selecting.get() { 0f32 } else { 1f32 })
                    .items_center()
                    .justify_center()
                    .pointer_events(PointerEvents::None)
                    .child(
                        canvas(|| {
                            Painter::new()
                                .begin_path()
                                .move_to(7., 4.)
                                .line_to(13., 10.)
                                .line_to(7., 16.)
                                .stroke_color(rgb(GOLD))
                                .line_width(1.8)
                                .stroke()
                        })
                        .w(px(20.))
                        .h(px(20.)),
                    )
            },
        ))
        .child(when(
            move || selecting.get(),
            move || {
                selection_circle(move || selected.with(|ids| ids.contains(&id)))
                    .test_id(&format!("project-selection-{id}"))
            },
        ))
}
fn cover(s: State, p: ReadSignal<Project>, grid: bool) -> View {
    let size = signal([44f32, 60f32]);
    let cover = view()
        .relative()
        .test_id(&format!("project-cover-{}", p.get_untracked().id))
        .items_center()
        .justify_center()
        .flex_shrink_0()
        .overflow_hidden()
        .rounded(px(4.))
        .on_layout(move |b: &Bounds<Points>| size.set([b.size.width.0, b.size.height.0]));
    let cover = if grid {
        cover.w_full().h(move || px(size.get()[0] * 1.367))
    } else {
        cover.w(px(44.)).h(px(60.))
    };
    cover
        .child(
            when(
                move || p.with(|p| p.pages.first().is_some_and(|p| p.preview(false).is_some())),
                move || {
                    fitted_image(
                        move || {
                            p.with(|p| {
                                p.pages
                                    .first()
                                    .map(|p| page_source(s, p, false))
                                    .unwrap_or_default()
                            })
                        },
                        move || {
                            p.with(|p| {
                                p.pages
                                    .first()
                                    .map_or(0., |p| p.preview(false).map_or(0., |p| p.rotation))
                            })
                        },
                        move || size.get()[0],
                        move || size.get()[1],
                    )
                    .accessible(false)
                },
            )
            .otherwise(move || {
                let id = p.get_untracked().id;
                canvas(move || {
                    let k = if grid { 1.5 } else { 1. };
                    Painter::new()
                        .begin_path()
                        .move_to(3. * k, 2. * k)
                        .line_to(21. * k, 2. * k)
                        .line_to(30. * k, 11. * k)
                        .line_to(30. * k, 42. * k)
                        .line_to(3. * k, 42. * k)
                        .close_path()
                        .fill_color(rgb(0x2b2d2f))
                        .fill()
                        .stroke_color(rgb(MUTED).opacity(0.65))
                        .line_width(1.3)
                        .stroke()
                        .begin_path()
                        .move_to(21. * k, 2. * k)
                        .line_to(21. * k, 11. * k)
                        .line_to(30. * k, 11. * k)
                        .move_to(9. * k, 22. * k)
                        .line_to(24. * k, 22. * k)
                        .move_to(9. * k, 29. * k)
                        .line_to(24. * k, 29. * k)
                        .stroke_color(rgb(MUTED).opacity(0.45))
                        .line_width(1.3)
                        .stroke()
                })
                .test_id(&format!("project-placeholder-{id}"))
                .w(px(if grid { 48. } else { 32. }))
                .h(px(if grid { 66. } else { 44. }))
                .accessible(false)
            }),
        )
        .child(when(
            move || {
                p.with(|p| {
                    p.pages
                        .first()
                        .is_some_and(|p| p.status != crate::model::PageStatus::Ready)
                })
            },
            move || unprocessed_shade(&format!("project-dim-{}", p.get_untracked().id)),
        ))
}
fn page_count(n: usize) -> String {
    crate::i18n::language().pages(n)
}
fn modified_label(p: &Project) -> String {
    if let Some(l) = use_context::<crate::i18n::Localization>()
        && let Some(date) = l
            .dates
            .with(|dates| dates.get(&p.last_modified().to_string()).cloned())
    {
        return date;
    }
    i64::try_from(p.last_modified())
        .ok()
        .and_then(chrono::DateTime::from_timestamp_millis)
        .filter(|_| p.last_modified() != 0)
        .map(|d| {
            d.with_timezone(&chrono::Local)
                .format(match crate::i18n::language() {
                    crate::i18n::Language::En => "%Y-%m-%d %H:%M",
                    crate::i18n::Language::De => "%d.%m.%Y, %H:%M",
                    crate::i18n::Language::Pt => "%d/%m/%Y %H:%M",
                })
                .to_string()
        })
        .unwrap_or_else(|| t(Key::DateUnavailable).into())
}
