//! Native page controls. Menu targets are captured when the menu is built.
use crate::{
    i18n::{Key, t},
    model::{Page, PreviewKind},
    state::{Route, State},
    ui::native_controls::{self, Choice},
};
use nsh_core::prelude::*;

pub fn actions(
    s: State,
    id: u64,
    current: impl Fn() -> Option<Page> + Copy + 'static,
    original: Signal<bool>,
) -> View {
    view()
        .flex_row()
        .items_center()
        .justify_between()
        .px_4()
        .pt_3()
        .pb_2()
        .gap_1()
        .flex_wrap()
        .child(native_controls::segments(
            move || {
                [
                    t(Key::Original).into(),
                    t(
                        if current().is_some_and(|p| {
                            p.preview(false)
                                .is_some_and(|p| p.kind == PreviewKind::PreviousResult)
                        }) {
                            Key::PreviousResult
                        } else {
                            Key::Result
                        },
                    )
                    .into(),
                ]
            },
            move || {
                usize::from(
                    current()
                        .and_then(|p| p.preview(original.get()).map(|p| p.kind))
                        .is_some_and(|kind| kind != PreviewKind::Original),
                )
            },
            move || current().map_or([false, false], |p| [p.source.is_some(), p.has_result()]),
            move |index| original.set(index == 0),
        ))
        .child(
            view()
                .flex_row()
                .gap_1()
                .child(native_controls::button(
                    "viewer-rotate",
                    t(Key::RotateClockwise),
                    "rotate.right",
                    move || current().is_some(),
                    Vec::new,
                    move |_, _| {
                        if let Some(p) = current() {
                            s.page(id, p.id, |p| p.rotation = (p.rotation + 90) % 360);
                        }
                    },
                ))
                .child(native_controls::button(
                    "viewer-retake",
                    t(Key::Retake),
                    "camera",
                    move || current().is_some(),
                    Vec::new,
                    move |_, _| {
                        if let Some(p) = current() {
                            s.push(Route::Camera(id, Some(p.id)));
                        }
                    },
                ))
                .child(native_controls::button(
                    "viewer-more",
                    t(Key::MoreActions),
                    "ellipsis",
                    move || current().is_some(),
                    move || {
                        current()
                            .map(|p| {
                                let mut choices = vec![];
                                if s.testing.is_testing() && p.source.is_some() {
                                    choices.push(
                                        Choice::new(
                                            "report",
                                            t(Key::SendOriginalToDeveloper),
                                            "paperplane",
                                        )
                                        .payload(serde_json::json!(p.id)),
                                    );
                                }
                                choices.push(
                                    Choice::new("delete", t(Key::DeletePage), "trash")
                                        .destructive()
                                        .payload(serde_json::json!(p.id)),
                                );
                                choices
                            })
                            .unwrap_or_default()
                    },
                    move |action, payload| {
                        if action == "report"
                            && let Some(page) = payload.as_u64()
                        {
                            s.testing.send_report(s, id, page);
                        }
                        if action == "delete"
                            && let Some(page) = payload.as_u64()
                            && s.project(id).pages.iter().any(|p| p.id == page)
                        {
                            s.delete(id, &[page]);
                            s.back();
                        }
                    },
                )),
        )
}
