//! PDF-only sharing, with exact sizes from prepared profiles and a fixed suffix.
use super::*;
pub fn export(s: State, id: u64, page_ids: Vec<u64>) -> Screen {
    let ids = signal(page_ids);
    let filename = signal(filename_stem(&s.project(id).title));
    s.export.update(|o| {
        o.format = Format::Pdf;
        o.color = Color::Automatic;
    });
    effect(move || {
        let _ = s.library.get();
        let _ = s.root.get();
        s.prepare_share(id, &ids.get());
    });
    cleanup(move || s.cancel_share());
    let selected_pages = move || s.project(id).ordered_selection(&ids.get());
    screen(s, move || crate::i18n::share_pages(selected_pages().len()))
        .child(
            scroll()
                .keyboard_taps(nsh_core::KeyboardTaps::Handled)
                .flex_1()
                .child(
                    view()
                        .p_5()
                        .gap_5()
                        .child(
                            view().gap_2().child(small(t(Key::FileName))).child(
                                view()
                                    .flex_row()
                                    .items_center()
                                    .rounded(px(12.))
                                    .bg(rgb(RAISED))
                                    .px_4()
                                    .child(
                                        nsh_base::input()
                                            .test_id("share-filename")
                                            .value(filename)
                                            .on_change(move |v: &str| {
                                                filename.set(filename_stem(v))
                                            })
                                            .return_key_type(ReturnKeyType::Done)
                                            .h(px(50.))
                                            .flex_1(),
                                    )
                                    .child(
                                        text(".pdf")
                                            .test_id("share-extension")
                                            .text_color(rgb(MUTED))
                                            .pointer_events(PointerEvents::None),
                                    ),
                            ),
                        )
                        .child(view().gap_2().child(small(t(Key::Size))).children(
                            crate::exporting::sharing::PROFILES.map(move |size| {
                                view()
                                    .test_id(&format!(
                                        "share-size-{}",
                                        crate::exporting::sharing::index(size)
                                    ))
                                    .role(AccessibilityRole::Button)
                                    .label(move || {
                                        format!(
                                            "{}, {}",
                                            profile_label(size),
                                            prepared_size(s, size)
                                        )
                                    })
                                    .accessibility_state(move || AccessibilityState {
                                        selected: s.export.get().size == size,
                                        ..Default::default()
                                    })
                                    .min_h(px(58.))
                                    .px_4()
                                    .py_3()
                                    .rounded(px(12.))
                                    .border_1()
                                    .border_color(move || {
                                        rgb(if s.export.get().size == size {
                                            GOLD
                                        } else {
                                            EDGE
                                        })
                                    })
                                    .bg(rgb(PANEL))
                                    .on_press(move || s.export.update(|o| o.size = size))
                                    .child(
                                        view()
                                            .flex_row()
                                            .items_center()
                                            .justify_between()
                                            .child(
                                                text(move || profile_label(size))
                                                    .text_size(px(16.)),
                                            )
                                            .child(small(move || prepared_size(s, size))),
                                    )
                            }),
                        ))
                        .child(when(
                            move || {
                                !selected_pages().is_empty()
                                    && selected_pages().iter().all(Page::can_share)
                                    && s.share_ready.get().is_none_or(|p| {
                                        p.error.is_none() && p.profiles.iter().any(Option::is_none)
                                    })
                            },
                            move || activity(t(Key::PreparingPdf)).test_id("share-preparing"),
                        ))
                        .child(when(
                            move || {
                                selected_pages().is_empty()
                                    || selected_pages().iter().any(|p| !p.can_share())
                            },
                            move || {
                                small(move || {
                                    if selected_pages().is_empty() {
                                        t(Key::NoPagesToShare)
                                    } else {
                                        t(Key::WaitUntilAllSelectedPagesAreReady)
                                    }
                                })
                                .test_id("share-waiting")
                            },
                        ))
                        .child(when(
                            move || s.share_ready.get().and_then(|p| p.error).is_some(),
                            move || {
                                view()
                                    .test_id("share-error")
                                    .gap_2()
                                    .child(small(t(Key::CouldNotPreparePdf)))
                                    .child(
                                        link(t(Key::TryAgain), move || {
                                            s.cancel_share();
                                            s.prepare_share(id, &ids.get_untracked());
                                        })
                                        .test_id("share-retry"),
                                    )
                            },
                        )),
                ),
        )
        .child(
            bottom(
                action(
                    move || {
                        if s.busy.get() {
                            t(Key::PleaseWait)
                        } else {
                            t(Key::Share)
                        }
                    },
                    true,
                    move || s.share_pdf(filename.get_untracked()),
                )
                .test_id("share-pdf")
                .disabled(move || {
                    s.busy.get()
                        || filename.get().trim().is_empty()
                        || s.share_ready
                            .get()
                            .and_then(|p| {
                                p.profiles[crate::exporting::sharing::index(s.export.get().size)]
                                    .clone()
                            })
                            .is_none()
                }),
            )
            .mb(move || px((keyboard_height().0 - safe_area().bottom.0).max(0.))),
        )
}
fn profile_label(size: ExportSize) -> &'static str {
    if size != ExportSize::Original {
        return crate::i18n::size_label(size);
    }
    match crate::i18n::language() {
        crate::i18n::Language::En => "Large",
        crate::i18n::Language::De => "Groß",
        crate::i18n::Language::Pt => "Grande",
    }
}
fn file_size(bytes: u64) -> String {
    let text = if bytes < 1_000_000 {
        format!("{} kB", bytes.div_ceil(1000))
    } else {
        format!("{:.1} MB", bytes as f64 / 1_000_000.)
    };
    if crate::i18n::language() == crate::i18n::Language::En {
        text
    } else {
        text.replace('.', ",")
    }
}

fn filename_stem(name: &str) -> String {
    if name
        .get(name.len().saturating_sub(4)..)
        .is_some_and(|s| s.eq_ignore_ascii_case(".pdf"))
    {
        name[..name.len() - 4].into()
    } else {
        name.into()
    }
}

fn prepared_size(s: State, size: ExportSize) -> String {
    s.share_ready
        .get()
        .and_then(|p| p.profiles[crate::exporting::sharing::index(size)].clone())
        .map_or_else(|| "…".into(), |p| file_size(p.bytes))
}
