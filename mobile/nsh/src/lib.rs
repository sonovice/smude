//! Smude's native mobile UI, written directly in Rust. No TSX or JS runtime.
mod camera;
pub mod exporting;
mod feedback;
pub mod i18n;
pub mod model;
mod processing;
mod reordering;
pub mod repository;
mod screens;
mod services;
mod state;
mod ui;
use nsh_base::nav_stack;
use nsh_core::prelude::*;
use state::{Route, State};
#[native_module("smude/native")]
fn register_ui(registry: &mut Registry) {
    registry.register::<App>();
    registry.assets(assets!());
}
#[component]
pub fn App() -> impl IntoElement {
    let state = State::new();
    effect(move || {
        let dates: Vec<_> = state
            .library
            .with(|l| l.projects.iter().map(|p| p.last_modified()).collect());
        if state.ready.get() {
            state
                .service
                .command("localization", serde_json::json!({"dates":dates}));
        }
    });
    state.start_worker();
    let feedback = feedback::Feedback::new(state);
    view()
        .relative()
        .pb_safe_area()
        .px_safe_area()
        .flex_1()
        .bg(rgb(ui::BG))
        .child(
            native("smude.services")
                .element_ref(&state.service)
                .w(px(0.))
                .h(px(0.))
                .on_native_event(move |e: &NativeEvent| {
                    feedback.event(e);
                    state.service_event(e);
                }),
        )
        .child(nav_stack().flex_1().child(each(
            move || {
                let _ = state.localization.language();
                state.routes.get()
            },
            move |r: &Route| format!("{r:?}-{:?}", state.localization.language()),
            move |route, index| {
                let page = match route.get_untracked() {
                    Route::Projects => screens::projects(state),
                    Route::Name(id) => screens::name(state, id),
                    Route::Project(id) => screens::project(state, id),
                    Route::Viewer(id, page) => screens::viewer(state, id, page),
                    Route::Camera(id, replace) => screens::camera(state, id, replace),
                    Route::Export(id, pages) => screens::export(state, id, pages),
                };
                page.on_dismiss(move || state.routes.update(|r| r.truncate(index.get_untracked())))
            },
        )))
        .child(feedback.view())
}

#[native_components(ios)]
fn ios_components(components: &nsh_platform_ios::Components) {
    camera::ios::register(components);
    services::ios::register(components);
    ui::ios_header::register(components);
    ui::native_controls::ios::register(components);
}

#[cfg(test)]
mod workflow_tests;
