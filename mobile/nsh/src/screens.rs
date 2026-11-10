use crate::i18n::{Key, t};
use crate::{
    model::*,
    state::{Route, State},
    ui::*,
};
use nsh_core::prelude::*;

mod collection_actions;
mod projects;
mod sharing;
mod viewer_actions;
pub use projects::projects;
pub use sharing::export;

pub fn name(s: State, id: Option<u64>) -> Screen {
    let name = signal(
        id.filter(|id| !s.project(*id).draft)
            .map(|id| s.project(id).title)
            .unwrap_or_default(),
    );
    let input = element_ref();
    let save = move || {
        let title = name.get_untracked().trim().to_owned();
        if title.is_empty() {
            return;
        }
        let mut l = s.library.get_untracked();
        let target = if let Some(id) = id {
            if let Some(p) = l.projects.iter_mut().find(|p| p.id == id) {
                p.title = title;
                p.draft = false;
            }
            id
        } else {
            let p = Project::new(title);
            let id = p.id;
            l.projects.insert(0, p);
            id
        };
        if s.save(l) {
            input.blur();
            if id.is_some() {
                s.back();
            } else {
                s.routes.update(|r| {
                    r.pop();
                    r.push(Route::Project(target));
                });
            }
        }
    };
    screen(
        s,
        if id.is_some_and(|id| !s.project(id).draft) {
            t(Key::Rename)
        } else {
            t(Key::NewCollection)
        },
    )
    .child(
        view()
            .flex_1()
            .p_5()
            .gap_3()
            .child(small(t(Key::CollectionName)))
            .child(
                nsh_base::input()
                    .element_ref(&input)
                    .test_id("project-name")
                    .value(name)
                    .placeholder(t(Key::CollectionName))
                    .auto_focus(true)
                    .selected_range((0, name.get_untracked().encode_utf16().count() as u32))
                    .return_key_type(ReturnKeyType::Done)
                    .on_submit(move |_: &str| save())
                    .on_change(move |v: &str| name.set(v.to_owned()))
                    .h(px(56.))
                    .px_4()
                    .rounded(px(12.))
                    .bg(rgb(RAISED)),
            ),
    )
    .child(
        bottom(action(t(Key::Save), true, save).disabled(move || name.get().trim().is_empty()))
            .test_id("project-name-footer")
            .flex_shrink(0.)
            // NSH reports only the keyboard overlapping this surface (zero
            // after an Android resize). The root already owns the safe inset.
            // Reading the environment here also follows the native animation.
            .mb(move || px((keyboard_height().0 - safe_area().bottom.0).max(0.))),
    )
}
pub fn project(s: State, id: u64) -> Screen {
    let selecting = signal(false);
    let selected = signal(Vec::<u64>::new());
    let grid_width = signal(360f32);
    let columns = signal(3usize);
    let page_scroll = element_ref();
    let reorder = crate::reordering::Reordering::new(columns, page_scroll);
    cleanup(move || reorder.cancel());
    screen(s, move || {
        if s.project(id).draft {
            t(Key::UnfiledScans).into()
        } else {
            s.project(id).title
        }
    })
    .header_title(
        view()
            .test_id("project-title")
            .on_long_press(move || collection_actions::more(s, id, None))
            .on_context_menu(move |_: &ContextMenuEvent| collection_actions::more(s, id, None))
            .child(
                text(move || {
                    if s.project(id).draft {
                        t(Key::UnfiledScans).into()
                    } else {
                        s.project(id).title
                    }
                })
                .text_size(px(18.))
                .font_weight(FontWeight::SEMIBOLD)
                .number_of_lines(1),
            ),
    )
    .header_right(
        view()
            .flex_row()
            .gap_1()
            .child(
                crate::ui::native_controls::bar_button(
                    "collection-camera",
                    t(Key::ScanPages),
                    "camera",
                    move || s.ready.get(),
                    Vec::new,
                    move |_, _| s.push(Route::Camera(id, None)),
                )
                .disabled(move || !s.ready.get()),
            )
            .child(
                crate::ui::native_controls::bar_button(
                    "collection-more",
                    t(Key::MoreActions),
                    "ellipsis",
                    move || !selecting.get() || selected.with(|ids| !ids.is_empty()),
                    move || {
                        collection_actions::choices(s, id, selecting.get().then(|| selected.get()))
                    },
                    move |action, payload| collection_actions::select(s, id, action, payload),
                )
                .disabled(move || selecting.get() && selected.with(|ids| ids.is_empty())),
            ),
    )
    .child(
        scroll()
            .test_id("page-scroll")
            .element_ref(&page_scroll)
            .flex_1()
            .scroll_value(&reorder.scroll_value)
            .scrollable(move || !reorder.owns_scrolling())
            .on_scroll(move |e: &ScrollEvent| reorder.scrolled(e))
            .on_layout(move |b: &Bounds<Points>| reorder.viewport(b.size.height.0))
            .on_touch(move |e: &nsh_core::TouchEvent| reorder.touch(s, id, e))
            .child(
                view()
                    .on_layout(move |b: &Bounds<Points>| reorder.content(b.size.height.0))
                    .p_5()
                    .gap_4()
                    .child(when(
                        move || s.project(id).draft && !s.project(id).pages.is_empty(),
                        move || {
                            link(t(Key::AddToCollection), move || {
                                collection_actions::assign(s, id)
                            })
                            .test_id("draft-assign")
                        },
                    ))
                    .child(
                        view()
                            .flex_row()
                            .flex_wrap()
                            .gap_3()
                            .on_layout(move |b: &Bounds<Points>| {
                                grid_width.set(b.size.width.0);
                                columns.set(grid_columns(b.size.width.0));
                            })
                            .child(each(
                                move || s.project(id).pages,
                                |p: &Page| p.id,
                                move |p, index| {
                                    let page_id = p.get_untracked().id;
                                    let thumb_width = signal(160f32);
                                    let cell = signal([180f32, 234f32]);
                                    let dx = animated_value(0.);
                                    let dy = animated_value(0.);
                                    let visual = crate::reordering::Visual {
                                        x: dx,
                                        y: dy,
                                        step: cell,
                                    };
                                    let drag_ref = element_ref();
                                    let display_y = reorder.display_y(page_id, dy);
                                    effect(move || {
                                        if let Some([x, y]) =
                                            reorder.offset(page_id, index.get(), cell.get())
                                        {
                                            if reorder.active() && !reduce_motion() {
                                                dx.timing(x, 160., Easing::EaseOut);
                                                dy.timing(y, 160., Easing::EaseOut);
                                            } else {
                                                dx.set(x);
                                                dy.set(y);
                                            }
                                        }
                                    });
                                    let card = view()
                                        .test_id(&format!("page-drag-{page_id}"))
                                        .element_ref(&drag_ref)
                                        .w(move || {
                                            // Native frames are point-rounded. Keep each
                                            // cell within its share so a fractional pixel
                                            // cannot wrap the third column onto a new row.
                                            px(((grid_width.get()
                                                - 12. * (columns.get() - 1) as f32)
                                                / columns.get() as f32)
                                                .floor()
                                                .max(0.))
                                        })
                                        .gap_2()
                                        .on_layout(move |b: &Bounds<Points>| {
                                            cell.set([b.size.width.0 + 12., b.size.height.0 + 12.])
                                        })
                                        .z_index(move || {
                                            if selected.with(|ids| ids.as_slice() == [page_id]) {
                                                10
                                            } else {
                                                0
                                            }
                                        })
                                        .bind_translate_x(&dx)
                                        .bind_translate_y(&display_y);
                                    reorder
                                        .attach_pan_fallback(card, s, id, page_id, visual, selected)
                                        .on_touch(move |e: &nsh_core::TouchEvent| {
                                            reorder.arm(page_id, drag_ref, e);
                                            if selecting.get_untracked()
                                                && selected.with_untracked(|ids| {
                                                    ids.as_slice() == [page_id]
                                                })
                                            {
                                                reorder.hold(s, id, page_id, visual);
                                            }
                                        })
                                        .child(
                                            view()
                                                .test_id(&format!("page-{page_id}"))
                                                .role(AccessibilityRole::Button)
                                                .accessibility_state(move || AccessibilityState {
                                                    selected: selected
                                                        .with(|ids| ids.contains(&page_id)),
                                                    ..Default::default()
                                                })
                                                .accessibility_value(move || {
                                                    processing_accessibility(s, &p.get())
                                                })
                                                .on_long_press(move || {
                                                    selected.set(vec![page_id]);
                                                    selecting.set(true);
                                                    reorder.hold(s, id, page_id, visual);
                                                })
                                                .on_context_menu(move |_: &ContextMenuEvent| {
                                                    selected.set(vec![page_id]);
                                                    selecting.set(true);
                                                })
                                                .on_press(move || {
                                                    if reorder.ignore_click(page_id) {
                                                        return;
                                                    }
                                                    if selecting.get() {
                                                        selected.update(|ids| {
                                                            if ids.contains(&page_id) {
                                                                ids.retain(|id| *id != page_id)
                                                            } else {
                                                                ids.push(page_id)
                                                            }
                                                        });
                                                    } else {
                                                        s.push(Route::Viewer(id, page_id));
                                                    }
                                                })
                                                .relative()
                                                .bg(rgb(RAISED))
                                                .p_2()
                                                .rounded(px(12.))
                                                .border_2()
                                                .border_color(move || {
                                                    rgb(
                                                        if selected.with(|v| v.contains(&page_id)) {
                                                            GOLD
                                                        } else {
                                                            EDGE
                                                        },
                                                    )
                                                })
                                                .child(
                                                    view()
                                                        .w_full()
                                                        .relative()
                                                        .h(move || px(thumb_width.get() * 1.414))
                                                        .on_layout(move |b: &Bounds<Points>| {
                                                            thumb_width.set(b.size.width.0)
                                                        })
                                                        .child(page_image(
                                                            s,
                                                            move || p.get(),
                                                            || false,
                                                            move || thumb_width.get(),
                                                            move || thumb_width.get() * 1.414,
                                                        ))
                                                        .child(when(
                                                            move || {
                                                                p.get().status != PageStatus::Ready
                                                            },
                                                            move || {
                                                                unprocessed_shade(&format!(
                                                                    "page-dim-{page_id}"
                                                                ))
                                                            },
                                                        ))
                                                        .child(when(
                                                            move || {
                                                                matches!(
                                                                    p.get().status,
                                                                    PageStatus::Importing
                                                                        | PageStatus::Pending
                                                                        | PageStatus::Processing
                                                                )
                                                            },
                                                            move || {
                                                                processing_indicator(s, move || {
                                                                    p.get()
                                                                })
                                                            },
                                                        )),
                                                )
                                                .child(
                                                    view()
                                                        .pt_3()
                                                        .pb_1()
                                                        .flex_row()
                                                        .justify_between()
                                                        .child(text(move || {
                                                            crate::i18n::page_title(index.get() + 1)
                                                        }))
                                                        .child(
                                                            text(move || match p.get().status {
                                                                PageStatus::Importing
                                                                | PageStatus::Pending => "",
                                                                PageStatus::Processing => "",
                                                                PageStatus::Failed => "!",
                                                                PageStatus::Ready => {
                                                                    if p.get().review {
                                                                        "!"
                                                                    } else {
                                                                        ""
                                                                    }
                                                                }
                                                            })
                                                            .text_color(rgb(GOLD)),
                                                        ),
                                                )
                                                .child(when(
                                                    move || selecting.get(),
                                                    move || {
                                                        selection_circle(move || {
                                                            selected.with(|v| v.contains(&page_id))
                                                        })
                                                        .test_id(&format!(
                                                            "page-selection-{page_id}"
                                                        ))
                                                    },
                                                )),
                                        )
                                },
                            )),
                    )
                    .child(when(
                        move || s.project(id).pages.is_empty(),
                        move || {
                            view()
                                .gap_2()
                                .child(title(t(Key::NoPagesYet)))
                                .child(small(t(Key::TakeAPhotoOrImportImages)))
                                .child(
                                    action(t(Key::ScanPages), true, move || {
                                        s.push(Route::Camera(id, None))
                                    })
                                    .test_id("empty-collection-camera")
                                    .disabled(move || !s.ready.get()),
                                )
                                .child(
                                    action(t(Key::ImportPhotos), false, move || {
                                        s.request_import(id, false)
                                    })
                                    .test_id("empty-collection-import")
                                    .disabled(move || !s.ready.get()),
                                )
                        },
                    )),
            ),
    )
    .child(when(
        move || selecting.get(),
        move || {
            bottom(
                view()
                    .flex_row()
                    .justify_between()
                    .child(
                        destructive_link(
                            move || crate::i18n::delete_pages(selected.with(|ids| ids.len())),
                            move || {
                                s.delete(id, &selected.get_untracked());
                                selected.set(vec![]);
                                selecting.set(false);
                            },
                        )
                        .test_id("selection-delete")
                        .disabled(move || selected.with(|ids| ids.is_empty())),
                    )
                    .child(
                        link(t(Key::Done), move || {
                            selected.set(vec![]);
                            selecting.set(false);
                        })
                        .test_id("selection-done"),
                    ),
            )
        },
    ))
}

pub fn viewer(s: State, id: u64, page_id: u64) -> Screen {
    let index = signal(
        s.project(id)
            .pages
            .iter()
            .position(|p| p.id == page_id)
            .unwrap_or(0),
    );
    let width = signal(360_f32);
    let height = signal(600_f32);
    let original = signal(false);
    let pager = element_ref();
    let zoomed = signal(false);
    let current = move || s.project(id).pages.get(index.get()).cloned();
    screen(s, move || {
        crate::i18n::language().page_of(index.get() + 1, s.project(id).pages.len())
    })
    .child(viewer_gestures(s, id))
    .header_right_system_item(nsh_core::HeaderSystemItem::Share)
    .header_right(
        link(t(Key::Share), move || {
            if let Some(p) = current() {
                s.push(Route::Export(id, vec![p.id]));
            }
        })
        .test_id("viewer-share")
        .disabled(move || !current().is_some_and(|p| p.can_share())),
    )
    .child(viewer_actions::actions(s, id, current, original))
    .child(when(
        move || current().is_some_and(|p| p.status != PageStatus::Ready),
        move || {
            view()
                .px_5()
                .gap_2()
                .child(
                    when(
                        move || current().is_some_and(|p| p.status == PageStatus::Processing),
                        move || processing_bar(s, current, false),
                    )
                    .otherwise(move || {
                        small(move || {
                            current()
                                .map(|p| match p.status {
                                    PageStatus::Failed => {
                                        t(Key::CouldNotProcessThisPageTryAgainOrRetakeThePhoto)
                                            .into()
                                    }
                                    PageStatus::Importing => t(Key::Preparing).into(),
                                    PageStatus::Pending => t(Key::WaitingToProcess).into(),
                                    _ => String::new(),
                                })
                                .unwrap_or_default()
                        })
                    }),
                )
                .child(when(
                    move || current().is_some_and(|p| p.status == PageStatus::Failed),
                    move || {
                        link(t(Key::TryAgain), move || {
                            if let Some(p) = current() {
                                s.page(id, p.id, Page::invalidate);
                            }
                        })
                    },
                ))
        },
    ))
    .child(when(
        move || {
            current().is_some_and(|p| {
                p.status == PageStatus::Ready && p.review && p.quality.retake_suggested
            })
        },
        move || {
            view()
                .test_id("focus-warning")
                .px_5()
                .py_2()
                .gap_1()
                .child(
                    text(move || {
                        current()
                            .and_then(|p| crate::i18n::quality(&p.quality))
                            .unwrap_or_default()
                    })
                    .text_color(rgb(GOLD))
                    .text_size(px(14.)),
                )
                .child(
                    view()
                        .flex_row()
                        .justify_between()
                        .child(
                            link(t(Key::Retake), move || {
                                if let Some(p) = current() {
                                    s.push(Route::Camera(id, Some(p.id)));
                                }
                            })
                            .test_id("retake-soft-page"),
                        )
                        .child(link(t(Key::Keep), move || {
                            if let Some(p) = current() {
                                s.page(id, p.id, |p| p.review = false);
                            }
                        })),
                )
        },
    ))
    .child(
        view()
            .flex_1()
            .on_layout(move |b: &Bounds<Points>| {
                width.set(b.size.width.0);
                height.set((b.size.height.0 - 32.).max(64.));
                pager.scroll_to(
                    Point {
                        x: px(index.get() as f32 * b.size.width.0),
                        y: px(0.),
                    },
                    false,
                );
            })
            .child(
                scroll()
                    .element_ref(&pager)
                    .test_id("page-pager")
                    .scrollable(move || !zoomed.get())
                    .horizontal()
                    .paging(true)
                    .flex_1()
                    .on_scroll_end(move |p: &Point<Points>| {
                        index.set((p.x.0 / width.get()).round() as usize)
                    })
                    .child(view().flex_row().child(each(
                        move || s.project(id).pages,
                        |p: &Page| p.id,
                        move |page, position| {
                            view()
                                .w(move || px(width.get()))
                                .px_4()
                                .py_4()
                                .justify_center()
                                .child(when(
                                    move || position.get().abs_diff(index.get()) <= 1,
                                    move || {
                                        scroll()
                                            .on_scroll(move |e: &ScrollEvent| {
                                                if position.get() == index.get() {
                                                    zoomed.set(e.zoom_scale > 1.01);
                                                }
                                            })
                                            .zoom_range(1., 4.)
                                            .center_content(true)
                                            .child(page_image(
                                                s,
                                                move || page.get(),
                                                move || original.get(),
                                                move || width.get() - 32.,
                                                move || height.get(),
                                            ))
                                    },
                                ))
                        },
                    ))),
            ),
    )
    .child(
        view()
            .test_id("page-navigation")
            .h(px(52.))
            .flex_row()
            .items_center()
            .justify_center()
            .gap_2()
            .child(
                pager_action(t(Key::PreviousPage), false, move || {
                    if index.get() > 0 {
                        index.update(|v| *v -= 1);
                        pager.scroll_to(
                            Point {
                                x: px(index.get() as f32 * width.get()),
                                y: px(0.),
                            },
                            true,
                        );
                    }
                })
                .label(t(Key::PreviousPage))
                .test_id("page-previous")
                .disabled(move || index.get() == 0),
            )
            .child(
                view()
                    .test_id("page-position")
                    .h(px(44.))
                    .min_w(px(64.))
                    .items_center()
                    .justify_center()
                    .child(small(move || {
                        format!("{} / {}", index.get() + 1, s.project(id).pages.len())
                    })),
            )
            .child(
                pager_action(t(Key::NextPage), true, move || {
                    if index.get() + 1 < s.project(id).pages.len() {
                        index.update(|v| *v += 1);
                        pager.scroll_to(
                            Point {
                                x: px(index.get() as f32 * width.get()),
                                y: px(0.),
                            },
                            true,
                        );
                    }
                })
                .label(t(Key::NextPage))
                .test_id("page-next")
                .disabled(move || index.get() + 1 >= s.project(id).pages.len()),
            ),
    )
}
pub fn camera(s: State, id: u64, replace: Option<u64>) -> Screen {
    let light = signal(false);
    let flash = crate::camera::CaptureFeedback::new();
    let camera_problem = signal(String::new());
    let light_available = signal(false);
    let contours = signal(crate::camera::Frame::default());
    let contour_revision = std::rc::Rc::new(std::cell::Cell::new(0u64));
    let cancel_contours = contour_revision.clone();
    cleanup(move || cancel_contours.set(cancel_contours.get().wrapping_add(1)));
    let level = signal(false);
    effect(move || {
        if let Some(v) = contours.get().alignment {
            let error = v[0].abs().max(v[1].abs());
            if error < 0.055 {
                level.set(true);
            } else if error > 0.085 {
                level.set(false);
            }
        } else {
            level.set(false);
        }
    });
    let size = signal([1_f32, 1_f32]);
    let ready = signal(false);
    let detected_pages = signal(1usize);
    let detected_turns = signal(0u16);
    let capturing = signal(false);
    let message = signal(t(Key::StartingCamera).to_owned());
    let session_start = crate::model::id();
    let latest_soft_page = move || {
        s.project(id)
            .pages
            .into_iter()
            .enumerate()
            .rev()
            .find(|(_, p)| p.id > session_start)
            .filter(|(_, p)| {
                p.status == PageStatus::Ready && p.review && p.quality.retake_suggested
            })
    };
    let stack = crate::camera::capture_stack::CaptureStack::new();
    let camera = element_ref();
    screen(
        s,
        move || if let Some(page) = replace {
            let index = s.project(id).pages.iter().position(|p| p.id == page).unwrap_or(0);
            crate::i18n::replace_page(index + 1)
        } else { t(Key::ScanPages).into() },
    )
    .header_right_system_item(nsh_core::HeaderSystemItem::Done)
    .header_right(link(t(Key::Done),move ||s.finish_camera(id)).test_id("camera-done"))
    .child(view().relative().flex_1().element_ref(&stack.scene).child(
        view()
            .relative().element_ref(&stack.preview)
            .flex_1()
            .overflow_hidden()
            .on_layout(move |b: &Bounds<Points>| size.set([b.size.width.0, b.size.height.0]))
            .child(
                native("smude.camera")
                    .test_id("camera-preview")
                    .prop("active", move || {
                        serde_json::json!(s.routes.with(
                            |r| matches!(r.last(),Some(Route::Camera(route_id,_)) if *route_id==id)
                        ))
                    })
                    .element_ref(&camera)
                    .flex_1()
                    .on_native_event(move |e: &NativeEvent| {
                        if &*e.name == "torch" {
                            light.set(e.data["on"].as_bool().unwrap_or(false));
                            light_available.set(e.data["available"].as_bool().unwrap_or(false));
                        }
                        if &*e.name == "status" {
                            camera_problem.set(e.data["reason"].as_str().unwrap_or_default().to_owned());
                            crate::camera::update_overlay(
                                contours,
                                contour_revision.clone(),
                                serde_json::from_value(e.data.clone()).unwrap_or_default(),
                            );
                            ready.set(e.data["ready"].as_bool().unwrap_or(false));
                            detected_pages.set(e.data["pages"].as_u64().unwrap_or(1) as usize);
                            if let Some(turns)=e.data["analysis_turns"].as_u64(){detected_turns.set((turns%4)as u16*90);}
                            capturing.set(e.data["capturing"].as_bool().unwrap_or(false));
                            message.set(
                                crate::i18n::camera_message(e.data["message"].as_str().unwrap_or("page_in_frame")),
                            );
                        }
                        if &*e.name == "capture_started" {
                            flash.capture();
                            stack.capture(s,e.data["preview_frame"].as_u64().unwrap_or(0));
                            if replace.is_some(){
                                spawn(async move {
                                    timer(std::time::Duration::from_millis(560)).await;
                                    if s.routes.with_untracked(|r| r.last() == Some(&Route::Camera(id, replace))) { s.back(); }
                                });
                            }
                        }
                        if &*e.name == "capture_failed" {capturing.set(false);}

                    }),
            )
            .child(flash.view())
            .child(
                canvas(move || crate::camera::drawing(&contours.get(), size.get()))
                    .absolute()
                    .inset_0()
                    .pointer_events(PointerEvents::None),
            )
            .child(
                nsh_base::switch()
                    .accessibility_label(t(Key::Light))
                    .checked(light)
                    .disabled(move || !light_available.get() || capturing.get())
                    .on_change(move |on: &bool| {
                        camera.command(if *on { "torch_on" } else { "torch_off" }, serde_json::Value::Null);
                    })
                    .absolute()
                    .top(px(14.))
                    .right(px(14.))
                    .min_h(px(44.))
                    .px_3()
                    .flex_row()
                    .items_center()
                    .gap_2()
                    .rounded_full()
                    .bg(rgb(BG).opacity(0.85))
                    .disabled_style(|s| s.opacity(0.4))
                    .track_class("w-10 h-6 rounded-full bg-[#484848] checked:bg-[#ffcc66]")
                    .thumb_class("absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white checked:bg-[#151515] checked:translate-x-4")
                    .child(text(t(Key::Light)).text_size(px(14.)).text_color(rgb(0xffffff))),
            )
            .child(when(
                move || contours.get().alignment.is_some(),
                move || {
                    view()
                        .absolute()
                        .top(px(14.))
                        .left(px(0.))
                        .right(px(0.))
                        .items_center()
                        .pointer_events(PointerEvents::None)
                        .child(
                            view()
                                .rounded(px(16.))
                                .bg(rgb(BG).opacity(0.85))
                                .px_3()
                                .py_2()
                                .items_center()
                                .child(
                                    canvas(move || {
                                        crate::camera::alignment_drawing(
                                            contours.get().alignment.unwrap_or([0., 0.]),
                                        )
                                    })
                                    .w(px(70.))
                                    .h(px(42.)),
                                )
                                .child(
                                    text(move || {
                                        if level.get() {
                                            t(Key::WellAligned)
                                        } else {
                                            t(Key::KeepTheCameraParallelToThePage)
                                        }
                                    })
                                    .text_size(px(12.))
                                    .text_color(rgb(GOLD)),
                                ),
                        )
                },
            )),
    )
    .child(when(
        move || !camera_problem.get().is_empty(),
        move || view().test_id("camera-problem").p_4().gap_2().bg(rgb(RAISED))
            .child(text(move || message.get()).text_color(rgb(0xffffff)).font_weight(FontWeight::SEMIBOLD))
            .child(when(move || camera_problem.get() == "permission", move || link(t(Key::OpenSettings), move || s.service.command("open_settings", serde_json::Value::Null)).test_id("camera-settings")))
            .child(when(move || replace.is_none(), move || link(t(Key::ImportPhotos), move || s.request_import(id, false)).test_id("camera-import-fallback"))
                .otherwise(move || link(t(Key::Cancel), move || s.back()))),
    ))
    .child(when(
        move || latest_soft_page().is_some() && !capturing.get(),
        move || {
            view()
                .test_id("capture-focus-warning")
                .px_4()
                .py_2()
                .gap_1()
                .child(
                    text(move || {
                        latest_soft_page()
                            .map(|(i, p)| {
                                format!(
                                    "{}: {}",
                                    crate::i18n::page_title(i+1),
                                    crate::i18n::quality(&p.quality).unwrap_or_default()
                                )
                            })
                            .unwrap_or_default()
                    })
                    .text_size(px(14.))
                    .text_color(rgb(GOLD)),
                )
                .child(link(t(Key::Retake), move || {
                    if let Some((_, p)) = latest_soft_page() {
                        s.routes.update(|r| {
                            if let Some(last) = r.last_mut() {
                                *last = Route::Camera(id, Some(p.id));
                            }
                        });
                    }
                }))
        },
    ))
    .child(when(move || camera_problem.get().is_empty(), move || view().p_3().items_center().child(small(move || {
        if replace.is_some() && detected_pages.get() == 2 {
            t(Key::CaptureOnlyThePageYouWantToReplace).into()
        } else {
            message.get()
        }
    }))))
    .child(bottom(
        view()
            .flex_row()
            .justify_between()
            .items_center()
            .child(stack.button(s,id))
            .child(
                action(t(Key::Capture), true, move || {
                    if let Some(arguments)=s.begin_capture(id,replace,detected_pages.get_untracked(),detected_turns.get_untracked()) {
                        capturing.set(true);camera.command("capture",arguments);
                    }
                })
                .disabled(move || !ready.get() || capturing.get() || !s.ready.get() || (replace.is_some() && detected_pages.get() != 1))
                .test_id("shutter"),
            )
            .child(view().w(px(76.)).h(px(72.)).items_center().justify_center().child(
                when(move||replace.is_none() && camera_problem.get().is_empty(),move||
                    link(t(Key::Photos),move||s.request_import(id,false)).disabled(move||capturing.get())
                )
            )),
    )).child(stack.overlay()))
}

fn viewer_gestures(s: State, id: u64) -> impl IntoElement {
    #[cfg(target_os = "ios")]
    {
        native("smude.viewer-gestures")
            .w(px(0.))
            .h(px(0.))
            .prop("active", move || {
                serde_json::json!(
                    s.routes.with(
                        |r| matches!(r.last(),Some(Route::Viewer(route_id,_)) if *route_id==id)
                    )
                )
            })
            .into_element()
    }
    #[cfg(not(target_os = "ios"))]
    {
        let _ = (s, id);
        view().h(px(0.)).into_element()
    }
}

fn page_source(s: State, page: &Page, original: bool) -> String {
    page.preview(original)
        .map(|p| s.path(p.path))
        .unwrap_or_default()
}
fn page_image(
    s: State,
    page: impl Fn() -> Page + Copy + 'static,
    original: impl Fn() -> bool + Copy + 'static,
    width: impl Fn() -> f32 + Copy + 'static,
    height: impl Fn() -> f32 + Copy + 'static,
) -> View {
    view().w(move || px(width())).h(move || px(height())).child(
        when(
            move || page().preview(original()).is_some(),
            move || {
                fitted_image(
                    move || page_source(s, &page(), original()),
                    move || page().preview(original()).map_or(0., |p| p.rotation),
                    width,
                    height,
                )
            },
        )
        .otherwise(move || {
            view()
                .test_id("image-placeholder")
                .w_full()
                .h_full()
                .p_2()
                .items_center()
                .justify_center()
                .child(
                    small(t(Key::ImageNotAvailable))
                        .text_size(px(12.))
                        .text_center(),
                )
        }),
    )
}
