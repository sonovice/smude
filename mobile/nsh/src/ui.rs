#[cfg(target_os = "ios")]
pub mod ios_header;
pub mod native_controls;
use crate::i18n::{Key, t};
use crate::state::State;
use nsh_base::nav_page;
use nsh_core::prelude::*;

pub const BG: u32 = 0x151515;
pub const PANEL: u32 = 0x191a1b;
pub const RAISED: u32 = 0x222325;
pub const GOLD: u32 = 0xffcc66;
pub const EDGE: u32 = 0x363636;
pub const DESTRUCTIVE: u32 = 0xff6b6b;
pub const MUTED: u32 = 0xaaaaaa;
pub fn action(
    label: impl IntoReactive<SharedString>,
    primary: bool,
    f: impl Fn() + 'static,
) -> View {
    view()
        .role(AccessibilityRole::Button)
        .min_h(px(48.))
        .px_4()
        .py_3()
        .rounded(px(12.))
        .items_center()
        .justify_center()
        .bg(rgb(if primary { GOLD } else { RAISED }))
        .active(|s| s.opacity(0.7))
        .disabled_style(|s| s.opacity(0.35))
        .on_press(f)
        .child(
            text(label)
                .text_size(px(16.))
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(rgb(if primary { BG } else { 0xffffff })),
        )
}
pub fn link(label: impl IntoReactive<SharedString>, f: impl Fn() + 'static) -> View {
    colored_link(label, GOLD, f)
}
pub fn destructive_link(label: impl IntoReactive<SharedString>, f: impl Fn() + 'static) -> View {
    colored_link(label, DESTRUCTIVE, f)
}
fn colored_link(
    label: impl IntoReactive<SharedString>,
    color: u32,
    f: impl Fn() + 'static,
) -> View {
    view()
        .role(AccessibilityRole::Button)
        .min_h(px(touch_target()))
        .px_2()
        .justify_center()
        .active(|s| s.opacity(0.6))
        .on_press(f)
        .child(text(label).text_size(px(16.)).text_color(rgb(color)))
}
pub fn small(label: impl IntoReactive<SharedString>) -> Text {
    text(label).text_size(px(14.)).text_color(rgb(MUTED))
}
pub fn title(label: impl IntoReactive<SharedString>) -> Text {
    text(label)
        .text_size(px(28.))
        .font_weight(FontWeight::SEMIBOLD)
        .text_color(rgb(0xffffff))
}
pub fn bottom(content: impl IntoElement) -> View {
    view()
        .p_4()
        .gap_3()
        .border_t_1()
        .border_color(rgb(EDGE))
        .bg(rgb(PANEL))
        .child(content)
}
pub fn screen(state: State, title_value: impl IntoReactive<SharedString> + Clone) -> Screen {
    base_screen(state, title_value, true)
}
pub fn home_screen(state: State, title_value: impl IntoReactive<SharedString> + Clone) -> Screen {
    base_screen(state, title_value, false).header_back_visible(false)
}
fn base_screen(
    state: State,
    title_value: impl IntoReactive<SharedString> + Clone,
    back: bool,
) -> Screen {
    nav_page()
        .title(title_value.clone())
        .header_back_title(t(Key::Back))
        .bg(rgb(BG))
        .text_color(rgb(0xffffff))
        .header_style_with(|s| {
            s.bg(rgb(BG))
                .text_color(rgb(0xffffff))
                .tint_color(rgb(GOLD))
        })
        .header(
            view()
                .min_h(px(56.))
                .px_4()
                .flex_row()
                .items_center()
                .gap_3()
                .child(when(
                    move || back && state.routes.with(|r| r.len() > 1),
                    move || link(t(Key::Back), move || state.back()),
                ))
                .child(
                    text(title_value)
                        .text_size(px(18.))
                        .font_weight(FontWeight::SEMIBOLD),
                ),
        )
}
/// Keep the visible rectangle stable for quarter-turns; rotating an ordinary
/// layout box alone would spill the wide result outside the pager.
pub fn fitted_image(
    source: impl IntoReactive<nsh_core::element::ImageSrc>,
    rotation: impl IntoReactive<f32>,
    width: impl IntoReactive<f32>,
    height: impl IntoReactive<f32>,
) -> View {
    let rotation = rotation.into_reactive();
    let r = computed(move || rotation.get());
    let width = width.into_reactive();
    let w = computed(move || width.get());
    let height = height.into_reactive();
    let h = computed(move || height.get());
    let iw = computed(move || {
        if (r.get() as u32 / 90) % 2 == 1 {
            h.get()
        } else {
            w.get()
        }
    });
    let ih = computed(move || {
        if (r.get() as u32 / 90) % 2 == 1 {
            w.get()
        } else {
            h.get()
        }
    });
    view()
        .relative()
        .w(move || px(w.get()))
        .h(move || px(h.get()))
        .child(
            image(source)
                .absolute()
                .left(move || px((w.get() - iw.get()) / 2.))
                .top(move || px((h.get() - ih.get()) / 2.))
                .w(move || px(iw.get()))
                .h(move || px(ih.get()))
                .object_fit(nsh_core::ResizeMode::Contain)
                .rotate(move || deg(r.get())),
        )
}

/// The entire card is the hit target; the selection badge never captures taps.
pub fn selection_circle(selected: impl IntoReactive<bool>) -> View {
    let selected = selected.into_reactive();
    let checked = computed(move || selected.get());
    view()
        .absolute()
        .top(px(8.))
        .right(px(8.))
        .w(px(26.))
        .h(px(26.))
        .rounded_full()
        .border_2()
        .border_color(move || rgb(if checked.get() { GOLD } else { 0xffffff }))
        .bg(move || rgb(if checked.get() { GOLD } else { 0x333333 }))
        .items_center()
        .justify_center()
        .pointer_events(PointerEvents::None)
        .accessible(false)
        .accessibility_elements_hidden(true)
        .child(
            text(move || if checked.get() { "✓" } else { "" })
                .text_size(px(18.))
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(rgb(BG)),
        )
}

pub fn trash_button(label: impl IntoReactive<SharedString>, f: impl Fn() + 'static) -> View {
    view()
        .role(AccessibilityRole::Button)
        .label(label)
        .w(px(touch_target()))
        .h(px(touch_target()))
        .items_center()
        .justify_center()
        .on_press(f)
        .child(bar_symbol("trash"))
        .child(
            canvas(|| {
                Painter::new()
                    .begin_path()
                    .move_to(3., 6.)
                    .line_to(21., 6.)
                    .move_to(8., 6.)
                    .line_to(8., 3.)
                    .line_to(16., 3.)
                    .line_to(16., 6.)
                    .move_to(5., 6.)
                    .line_to(6., 21.)
                    .line_to(18., 21.)
                    .line_to(19., 6.)
                    .move_to(10., 10.)
                    .line_to(10., 17.)
                    .move_to(14., 10.)
                    .line_to(14., 17.)
                    .stroke_color(rgb(DESTRUCTIVE))
                    .line_width(1.8)
                    .stroke()
            })
            .w(px(24.))
            .h(px(24.))
            .pointer_events(PointerEvents::None),
        )
}

pub fn plus_button(label: &'static str, f: impl Fn() + 'static) -> View {
    view()
        .role(AccessibilityRole::Button)
        .label(label)
        .w(px(touch_target()))
        .h(px(touch_target()))
        .items_center()
        .justify_center()
        .on_press(f)
        .child(bar_symbol("plus"))
        .child(
            canvas(|| {
                Painter::new()
                    .begin_path()
                    .move_to(12., 3.)
                    .line_to(12., 21.)
                    .move_to(3., 12.)
                    .line_to(21., 12.)
                    .stroke_color(rgb(GOLD))
                    .line_width(1.8)
                    .stroke()
            })
            .w(px(24.))
            .h(px(24.))
            .pointer_events(PointerEvents::None),
        )
}

/// Dims only the image, leaving page labels, selection and progress legible.
pub fn unprocessed_shade(test: &str) -> View {
    view()
        .test_id(test)
        .absolute()
        .inset_0()
        .bg(rgb(0x000000).opacity(0.48))
        .pointer_events(PointerEvents::None)
        .accessible(false)
}

/// Overlay remains in the thumbnail's fixed bounds for pending, active and
/// completed pages alike. Progress belongs to the current capture revision.
pub fn processing_indicator(
    s: State,
    page: impl Fn() -> crate::model::Page + Copy + 'static,
) -> View {
    view()
        .absolute()
        .inset_0()
        .items_center()
        .justify_center()
        .pointer_events(PointerEvents::None)
        .child(native_controls::progress_pie(move || {
            let page = page();
            s.processing
                .get()
                .filter(|p| {
                    p.page_id == page.id && p.capture == page.capture && p.revision == page.revision
                })
                .map(|p| p.percent)
        }))
}

/// Keep progress available to screen readers without adding visible status text.
pub fn processing_accessibility(
    s: State,
    page: &crate::model::Page,
) -> Option<nsh_core::AccessibilityValue> {
    use crate::model::PageStatus;
    if !matches!(
        page.status,
        PageStatus::Importing | PageStatus::Pending | PageStatus::Processing
    ) {
        return None;
    }
    let active = s.processing.get().filter(|p| {
        p.page_id == page.id && p.capture == page.capture && p.revision == page.revision
    });
    Some(nsh_core::AccessibilityValue {
        min: Some(0.),
        max: Some(100.),
        now: active.map(|p| p.percent),
        text: Some(
            active
                .map_or_else(
                    || t(Key::Preparing).to_owned(),
                    |p| format!("{} · {:.0}%", p.label(), p.percent),
                )
                .into(),
        ),
    })
}

/// The NSH progress component owns range/assistive semantics; the worker owns
/// its value. Late events from a replaced page never attach to its new revision.
pub fn processing_bar(
    s: State,
    page: impl IntoReactive<Option<crate::model::Page>>,
    compact: bool,
) -> View {
    let page = page.into_reactive();
    let current = computed(move || {
        let page = page.get()?;
        s.processing.get().filter(|p| {
            p.page_id == page.id && p.capture == page.capture && p.revision == page.revision
        })
    });
    view()
        .test_id("processing-progress")
        .w_full()
        .gap_2()
        .child(
            nsh_base::progress()
                .accessibility_label(t(Key::PageProcessing))
                .value(move || current.get().map_or(0., |p| p.percent))
                .indeterminate(move || current.get().is_none())
                .w_full()
                .track_class("h-1.5 w-full rounded-full overflow-hidden bg-[#363636]")
                .indicator_class("h-full rounded-full bg-[#ffcc66]"),
        )
        .child(when(
            move || !compact,
            move || {
                view()
                    .flex_row()
                    .justify_between()
                    .gap_2()
                    .child(small(move || {
                        current
                            .get()
                            .map_or_else(|| t(Key::Preparing).into(), |p| p.label())
                    }))
                    .child(small(move || {
                        current
                            .get()
                            .map_or_else(String::new, |p| format!("{:.0} %", p.percent))
                    }))
            },
        ))
}

#[derive(Clone, Copy)]
pub enum ActionIcon {
    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    More,
    Scan,
    Previous,
    Next,
}
pub fn action_icon(icon: ActionIcon, color: u32) -> impl IntoElement {
    canvas(move || {
        let mut p = Painter::new().begin_path();
        match icon {
            #[cfg(not(any(target_os = "ios", target_os = "android")))]
            ActionIcon::More => {
                for x in [5., 12., 19.] {
                    p = p.move_to(x, 11.5).line_to(x, 12.5);
                }
            }
            ActionIcon::Scan => {
                for (x, y, sx, sy) in [
                    (3., 3., 1., 1.),
                    (21., 3., -1., 1.),
                    (3., 21., 1., -1.),
                    (21., 21., -1., -1.),
                ] {
                    p = p
                        .move_to(x, y + 5. * sy)
                        .line_to(x, y)
                        .line_to(x + 5. * sx, y);
                }
            }
            ActionIcon::Previous => {
                p = p.move_to(15., 6.).line_to(9., 12.).line_to(15., 18.);
            }
            ActionIcon::Next => {
                p = p.move_to(9., 6.).line_to(15., 12.).line_to(9., 18.);
            }
        }
        p.stroke_color(rgb(color))
            .line_width(1.8)
            .line_cap(nsh_core::canvas::LineCap::Round)
            .line_join(nsh_core::canvas::LineJoin::Round)
            .stroke()
    })
    .w(px(22.))
    .h(px(22.))
    .pointer_events(PointerEvents::None)
}
pub fn pager_action(label: &'static str, next: bool, press: impl Fn() + 'static) -> View {
    view()
        .role(AccessibilityRole::Button)
        .label(label)
        .w(px(touch_target()))
        .h(px(touch_target()))
        .items_center()
        .justify_center()
        .disabled_style(|s| s.opacity(0.25))
        .active(|s| s.opacity(0.6))
        .on_press(press)
        .child(action_icon(
            if next {
                ActionIcon::Next
            } else {
                ActionIcon::Previous
            },
            GOLD,
        ))
}

pub fn header_icon(label: &'static str, icon: ActionIcon, press: impl Fn() + 'static) -> View {
    view()
        .role(AccessibilityRole::Button)
        .label(label)
        .w(px(touch_target()))
        .h(px(touch_target()))
        .items_center()
        .justify_center()
        .active(|s| s.opacity(0.6))
        .disabled_style(|s| s.opacity(0.35))
        .on_press(press)
        .child(bar_symbol(match icon {
            ActionIcon::Scan => "camera",
            #[cfg(not(any(target_os = "ios", target_os = "android")))]
            ActionIcon::More => "ellipsis",
            ActionIcon::Previous => "chevron.backward",
            ActionIcon::Next => "chevron.forward",
        }))
        .child(action_icon(icon, GOLD))
}
/// Three columns on phones, adding columns as tablet/window width permits.
pub fn grid_columns(width: f32) -> usize {
    ((width + 12.) / 160.).floor().max(3.) as usize
}

fn bar_symbol(symbol: &'static str) -> View {
    let container = view()
        .w(px(0.))
        .h(px(0.))
        .pointer_events(PointerEvents::None)
        .accessibility_elements_hidden(true);
    #[cfg(target_os = "ios")]
    {
        container.child(native("smude.header-symbol").prop("symbol", serde_json::json!(symbol)))
    }
    #[cfg(not(target_os = "ios"))]
    {
        let _ = symbol;
        container
    }
}

pub fn touch_target() -> f32 {
    if cfg!(target_os = "android") {
        48.
    } else {
        44.
    }
}

/// An activity indicator driven by the native animation graph, not fake percentages.
pub fn activity(label: impl IntoReactive<SharedString>) -> View {
    let angle = animated_value(0.);
    let motion = Animation::timing(
        &angle,
        std::f32::consts::TAU,
        Timing::new(950., Easing::Linear),
    )
    .looped(None);
    let animation = motion.clone();
    effect(move || {
        if reduce_motion() {
            animation.stop();
        } else {
            animation.start();
        }
    });
    cleanup(move || motion.stop());
    view()
        .accessible(true)
        .role(AccessibilityRole::Progressbar)
        .flex_row()
        .items_center()
        .gap_2()
        .child(
            canvas(|| {
                Painter::new()
                    .begin_path()
                    .move_to(17., 10.)
                    .arc(10., 10., 7., 0., 4.8)
                    .stroke_color(rgb(GOLD))
                    .line_width(2.)
                    .stroke()
            })
            .w(px(20.))
            .h(px(20.))
            .bind_rotate(&angle)
            .accessible(false),
        )
        .child(small(label))
}
