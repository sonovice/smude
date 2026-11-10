//! Platform controls own presentation; Rust owns choices and their immutable targets.
use super::touch_target;
use nsh_core::prelude::*;
use serde::Serialize;

#[derive(Clone, Serialize)]
pub struct Choice {
    pub id: &'static str,
    pub title: String,
    pub symbol: &'static str,
    pub destructive: bool,
    pub payload: serde_json::Value,
}
impl Choice {
    pub fn new(id: &'static str, title: impl Into<String>, symbol: &'static str) -> Self {
        Self {
            id,
            title: title.into(),
            symbol,
            destructive: false,
            payload: serde_json::Value::Null,
        }
    }
    pub fn destructive(mut self) -> Self {
        self.destructive = true;
        self
    }
    pub fn payload(mut self, payload: serde_json::Value) -> Self {
        self.payload = payload;
        self
    }
}
/// A real UIKit/Android button, optionally with a menu attached to that button.
pub fn button(
    test: &'static str,
    label: &'static str,
    symbol: &'static str,
    enabled: impl Fn() -> bool + Copy + 'static,
    choices: impl Fn() -> Vec<Choice> + Copy + 'static,
    selected: impl Fn(&str, &serde_json::Value) + Copy + 'static,
) -> View {
    control(test, label, symbol, enabled, choices, selected, false)
}
/// Mounted directly as a native navigation item, without a custom header view.
pub fn bar_button(
    test: &'static str,
    label: &'static str,
    symbol: &'static str,
    enabled: impl Fn() -> bool + Copy + 'static,
    choices: impl Fn() -> Vec<Choice> + Copy + 'static,
    selected: impl Fn(&str, &serde_json::Value) + Copy + 'static,
) -> View {
    control(test, label, symbol, enabled, choices, selected, true)
}
fn control(
    test: &'static str,
    label: &'static str,
    symbol: &'static str,
    enabled: impl Fn() -> bool + Copy + 'static,
    choices: impl Fn() -> Vec<Choice> + Copy + 'static,
    selected: impl Fn(&str, &serde_json::Value) + Copy + 'static,
    bar: bool,
) -> View {
    #[cfg(any(target_os = "ios", target_os = "android"))]
    {
        view()
            .w(px(touch_target()))
            .h(px(touch_target()))
            .disabled(move || !enabled())
            .child(
                native("smude.control")
                    .test_id(test)
                    .w_full()
                    .h_full()
                    .prop("config", move || {
                        serde_json::json!({
                            "kind": "button", "test_id": test, "label": label,
                            "symbol": symbol, "choices": choices(), "enabled": enabled(), "bar_button": bar
                        })
                    })
                    .on_native_event(move |event: &NativeEvent| {
                        if &*event.name == "select" {
                            selected(
                                event.data["id"].as_str().unwrap_or_default(),
                                &event.data["payload"],
                            );
                        }
                    }),
            )
    }
    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    {
        let _ = bar;
        super::header_icon(
            label,
            if symbol == "camera" {
                super::ActionIcon::Scan
            } else {
                super::ActionIcon::More
            },
            move || {
                let entries = choices();
                if entries.is_empty() {
                    selected("press", &serde_json::Value::Null);
                    return;
                }
                let mut dialog = nsh_base::alert_dialog().title(label);
                for item in &entries {
                    dialog = if item.destructive {
                        dialog.destructive(item.id, item.title.clone())
                    } else {
                        dialog.action(item.id, item.title.clone())
                    };
                }
                let answer = dialog
                    .cancel("cancel", crate::i18n::t(crate::i18n::Key::Cancel))
                    .action_sheet()
                    .show();
                spawn(async move {
                    if let Ok(result) = answer.await
                        && let Some(entry) = entries
                            .iter()
                            .find(|i| Some(i.id) == result.action.as_deref())
                    {
                        selected(entry.id, &entry.payload);
                    }
                });
            },
        )
        .test_id(test)
        .disabled(move || !enabled())
    }
}
/// Platform segmented control with system list/grid symbols.
pub fn layout_segments(
    selected: impl Fn() -> usize + Copy + 'static,
    changed: impl Fn(usize) + Copy + 'static,
) -> View {
    #[cfg(any(target_os = "ios", target_os = "android"))]
    {
        view().w(px(112.)).h(px(touch_target())).child(
            native("smude.control")
                .test_id("project-layout")
                .w_full().h_full()
                .prop("config", move || serde_json::json!({
                    "kind":"segments", "test_id":"project-layout",
                    "labels":[crate::i18n::t(crate::i18n::Key::ListView),crate::i18n::t(crate::i18n::Key::GridView)],
                    "symbols":["list.bullet","square.grid.3x3"],
                    "selected":selected(),"enabled":[true,true]
                }))
                .on_native_event(move |event: &NativeEvent| {
                    if &*event.name == "change"
                        && let Some(value) = event.data["value"].as_u64().filter(|value| *value < 2)
                    { changed(value as usize); }
                }),
        )
    }
    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    {
        let item = move |index, test: &'static str, label: &'static str| {
            view()
                .test_id(test)
                .role(AccessibilityRole::Button)
                .label(label)
                .accessibility_state(move || AccessibilityState {
                    selected: selected() == index,
                    ..Default::default()
                })
                .min_h(px(touch_target()))
                .px_2()
                .items_center()
                .justify_center()
                .on_press(move || changed(index))
                .child(text(label))
        };
        view()
            .flex_row()
            .child(item(
                0,
                "project-view-list",
                crate::i18n::t(crate::i18n::Key::ListView),
            ))
            .child(item(
                1,
                "project-view-grid",
                crate::i18n::t(crate::i18n::Key::GridView),
            ))
    }
}
/// Native platform progress, hosted inside the image without affecting layout.
pub fn progress_pie(value: impl Fn() -> Option<f32> + Copy + 'static) -> View {
    #[cfg(any(target_os = "ios", target_os = "android"))]
    {
        view().w(px(48.)).h(px(48.)).child(
            native("smude.control").test_id("page-processing-indicator").w_full().h_full()
                .prop("config", move || serde_json::json!({
                    "kind":"progress", "test_id":"page-processing-indicator",
                    "label":crate::i18n::t(crate::i18n::Key::PageProcessing),
                    "value":value().unwrap_or(0.).clamp(0.,100.), "indeterminate":value().is_none()
                }))
        )
    }
    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    {
        view()
            .test_id("page-processing-indicator")
            .w(px(48.))
            .h(px(48.))
            .child(
                nsh_base::progress()
                    .value(move || value().unwrap_or(0.))
                    .indeterminate(move || value().is_none())
                    .w_full()
                    .track_class("h-1.5 w-full rounded-full bg-[#363636]")
                    .indicator_class("h-full rounded-full bg-[#ffcc66]"),
            )
    }
}
/// Disabled segments stay visible and cannot select an unavailable image.
pub fn segments(
    labels: impl Fn() -> [String; 2] + Copy + 'static,
    selected: impl Fn() -> usize + Copy + 'static,
    enabled: impl Fn() -> [bool; 2] + Copy + 'static,
    changed: impl Fn(usize) + Copy + 'static,
) -> View {
    #[cfg(any(target_os = "ios", target_os = "android"))]
    {
        view().w(px(200.)).h(px(touch_target())).child(
            native("smude.control")
                .test_id("viewer-mode")
                .w_full()
                .h_full()
                .prop("config", move || {
                    serde_json::json!({
                        "kind": "segments", "test_id": "viewer-mode",
                        "labels": labels(), "selected": selected(), "enabled": enabled()
                    })
                })
                .on_native_event(move |event: &NativeEvent| {
                    if &*event.name == "change"
                        && let Some(value) = event.data["value"].as_u64().filter(|value| *value < 2)
                        && enabled()[value as usize]
                    {
                        changed(value as usize);
                    }
                }),
        )
    }
    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    {
        view()
            .flex_row()
            .p_1()
            .rounded(px(12.))
            .bg(rgb(super::RAISED))
            .child(segment_fallback(0, labels, selected, enabled, changed))
            .child(segment_fallback(1, labels, selected, enabled, changed))
    }
}
#[cfg(not(any(target_os = "ios", target_os = "android")))]
fn segment_fallback(
    index: usize,
    labels: impl Fn() -> [String; 2] + Copy + 'static,
    selected: impl Fn() -> usize + Copy + 'static,
    enabled: impl Fn() -> [bool; 2] + Copy + 'static,
    changed: impl Fn(usize) + Copy + 'static,
) -> impl IntoElement {
    nsh_base::toggle()
        .pressed(move || selected() == index)
        .disabled(move || !enabled()[index])
        .disabled_style(|s| s.opacity(0.35))
        .accessibility_label(move || Some(labels()[index].clone().into()))
        .on_change(move |_: &bool| changed(index))
        .min_h(px(touch_target()))
        .px_2()
        .items_center()
        .justify_center()
        .rounded(px(8.))
        .bg(move || {
            rgb(if selected() == index {
                0x393326
            } else {
                super::RAISED
            })
        })
        .child(
            text(move || labels()[index].clone())
                .text_size(px(13.))
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(move || {
                    rgb(if selected() == index {
                        super::GOLD
                    } else {
                        super::MUTED
                    })
                }),
        )
}

#[cfg(target_os = "ios")]
pub mod ios {
    use nsh_core::JsonValue;
    use nsh_platform_ios::{ComponentContext, Components, NativeComponent};
    use objc2::rc::Retained;
    use objc2_ui_kit::{UIBarButtonItem, UIView};
    use std::{
        ffi::{CStr, CString},
        rc::Rc,
    };
    unsafe extern "C" {
        fn smude_control_create(
            context: *mut std::ffi::c_void,
            event: extern "C" fn(*mut std::ffi::c_void, *const std::ffi::c_char),
        ) -> *mut UIView;
        fn smude_control_config(view: *mut UIView, json: *const std::ffi::c_char);
        fn smude_control_shutdown(view: *mut UIView);
        fn smude_control_bar_item(view: *mut UIView) -> *mut UIBarButtonItem;
    }
    extern "C" fn event(raw: *mut std::ffi::c_void, json: *const std::ffi::c_char) {
        if raw.is_null() || json.is_null() {
            return;
        }
        let context = unsafe { &*raw.cast::<ComponentContext>() };
        let data: JsonValue =
            serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap_or("{}"))
                .unwrap_or_default();
        let name = data["event"].as_str().unwrap_or("select").to_owned();
        context.emit(&name, data);
    }
    struct Control {
        view: Retained<UIView>,
        _context: Box<ComponentContext>,
        bar: Option<Retained<UIBarButtonItem>>,
    }
    impl NativeComponent for Control {
        fn view(&self) -> &UIView {
            &self.view
        }
        fn bar_button_item(&self) -> Option<&UIBarButtonItem> {
            self.bar.as_deref()
        }
        fn set_prop(&mut self, name: &str, value: Option<&JsonValue>) {
            if name == "config"
                && let Ok(json) = CString::new(value.unwrap_or(&JsonValue::Null).to_string())
            {
                unsafe {
                    smude_control_config(
                        Retained::as_ptr(&self.view) as *mut UIView,
                        json.as_ptr(),
                    );
                    self.bar = Retained::from_raw(smude_control_bar_item(Retained::as_ptr(
                        &self.view,
                    )
                        as *mut UIView));
                }
            }
        }
    }
    impl Drop for Control {
        fn drop(&mut self) {
            unsafe { smude_control_shutdown(Retained::as_ptr(&self.view) as *mut UIView) }
        }
    }
    pub fn register(components: &Components) {
        components.register(
            "smude.control",
            Rc::new(|context| {
                let mut context = Box::new(context);
                let view = unsafe {
                    Retained::from_raw(smude_control_create(
                        (&mut *context as *mut ComponentContext).cast(),
                        event,
                    ))
                }
                .expect("native control allocation");
                Box::new(Control {
                    view,
                    _context: context,
                    bar: None,
                })
            }),
        );
    }
}
