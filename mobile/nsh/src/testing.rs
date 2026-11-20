//! Feedback that only test builds collect: originals with their capture and
//! processing context, and a one-time price question.
//!
//! The native host reports TestFlight at runtime (`environment` event). Android
//! cannot tell its test tracks apart, so `SMUDE_TESTING=1` at build time forces
//! test mode there and in development builds. App Store builds never show,
//! measure or send anything from this module.
mod report;
mod survey;
mod upload;

use crate::i18n::{Key, t};
use crate::state::State;
use nsh_base::alert_dialog;
use nsh_core::{prelude::*, storage::Storage};
use serde_json::{Value, json};

/// Build-time override; read once, so a stale value cannot flip mid-session.
fn forced() -> bool {
    matches!(option_env!("SMUDE_TESTING"), Some("1" | "true" | "yes"))
}

#[derive(Clone, Copy)]
pub struct Testing {
    host: Signal<bool>,
    /// What the native host knows about itself: `app` and `device` objects.
    environment: Signal<Value>,
    survey: survey::Survey,
}
impl Testing {
    pub fn new() -> Self {
        Self {
            host: signal(false),
            environment: signal(json!({})),
            survey: survey::Survey::load(),
        }
    }
    /// True in TestFlight and in builds made with `SMUDE_TESTING=1`.
    pub fn is_testing(self) -> bool {
        forced() || self.host.get()
    }
    /// Consumes the host's `environment` event; other events pass through.
    pub fn event(self, e: &NativeEvent) -> bool {
        if &*e.name != "environment" {
            return false;
        }
        self.host.set(e.data["testing"].as_bool().unwrap_or(false));
        self.environment
            .set(json!({"app": e.data["app"], "device": e.data["device"]}));
        true
    }
    /// Counts active use and asks the price question once it is due.
    pub fn start(self, state: State) {
        self.survey.start(state, self);
    }
    /// Asks for consent, then sends the page's original and its context.
    pub fn send_report(self, state: State, project: u64, page: u64) {
        let Some(repo) = state.repo() else { return };
        let project = state.project(project);
        let Some(index) = project.pages.iter().position(|p| p.id == page) else {
            return;
        };
        let page = project.pages[index].clone();
        let position = (index + 1, project.pages.len());
        let context = self.context();
        spawn(async move {
            let answer = alert_dialog()
                .title(t(Key::SendOriginalToDeveloper))
                .description(t(Key::SendOriginalDescription))
                .cancel("cancel", t(Key::Cancel))
                .action("send", t(Key::Send))
                .show()
                .await;
            if !answer.is_ok_and(|a| a.action.as_deref() == Some("send")) {
                return;
            }
            state.notice.set(t(Key::SendingReport).into());
            let fields = blocking(move || report::prepare(&repo, &page, position, &context))
                .await
                .map_err(|e| e.to_string())
                .and_then(|r| r);
            let result = match fields {
                Ok(fields) => upload::create_multipart("reports", &fields).await,
                Err(error) => Err(error),
            };
            state.notice.set(match result {
                Ok(()) => t(Key::ReportSent).into(),
                Err(error) => crate::i18n::failure(crate::i18n::Failure::Report, &error),
            });
        });
    }
    /// Fields every record carries: who (an anonymous installation), which
    /// build and which device.
    fn context(self) -> report::Context {
        let environment = self.environment.get_untracked();
        let mut device = environment["device"].clone();
        if !device.is_object() {
            device = json!({});
        }
        device["os"] = json!(format!("{:?}", os()));
        device["os_version"] = json!(os_version());
        device["idiom"] = json!(format!("{:?}", idiom()));
        device["locale"] = json!(locale().as_str());
        let mut app = environment["app"].clone();
        if !app.is_object() {
            app = json!({});
        }
        app["core"] = json!(env!("CARGO_PKG_VERSION"));
        app["forced_testing"] = json!(forced());
        report::Context {
            install: install_id(),
            app,
            device,
            language: format!("{:?}", crate::i18n::language()).to_lowercase(),
        }
    }
}

/// A random, app-local identifier that counts each tester once. It is not
/// derived from the device and is lost when the app is deleted.
fn install_id() -> String {
    const KEY: &str = "smude-install-id";
    if let Some(id) = Storage::get_item(KEY).filter(|id| !id.is_empty()) {
        return id;
    }
    use std::hash::{BuildHasher, Hasher};
    let random = std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish();
    let id = format!("{random:016x}{:x}", crate::model::id());
    let _ = Storage::set_item(KEY, id.clone());
    id
}
