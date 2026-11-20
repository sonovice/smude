//! The one-time price question: after five minutes of active use, at the next
//! calm moment, never during capture, processing dialogs or sharing.
use super::Testing;
use crate::i18n::{Key, t};
use crate::state::{Route, State};
use nsh_base::alert_dialog;
use nsh_core::{platform::Lifecycle, prelude::*, storage::Storage};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Duration;

/// One-time purchase prices offered, in euros.
pub const PRICES: [u32; 6] = [0, 5, 10, 15, 20, 25];
const DUE_AFTER_SECONDS: u64 = 5 * 60;
const TICK_SECONDS: u64 = 10;
const KEY: &str = "smude-testing-survey";

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Progress {
    /// Foreground seconds in test mode, summed across launches.
    #[serde(default)]
    pub active_seconds: u64,
    /// Set once the question was answered or declined; it never returns.
    #[serde(default)]
    pub asked: bool,
    /// An answer the backend has not confirmed yet; retried on launch.
    #[serde(default)]
    pub unsent: Option<Answer>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Answer {
    /// `None`: declined to answer.
    pub price: Option<u32>,
    pub active_seconds: u64,
}
impl Progress {
    pub fn due(&self) -> bool {
        !self.asked && self.active_seconds >= DUE_AFTER_SECONDS
    }
    /// The chosen action of the dialog: `p<euros>`, anything else declines.
    pub fn answer(&mut self, action: Option<&str>) -> Answer {
        let price = action
            .and_then(|a| a.strip_prefix('p'))
            .and_then(|p| p.parse().ok())
            .filter(|p| PRICES.contains(p));
        let answer = Answer {
            price,
            active_seconds: self.active_seconds,
        };
        self.asked = true;
        self.unsent = Some(answer.clone());
        answer
    }
}

/// Only screens where nothing is in progress: the collection list or one
/// collection, no picker, export or share sheet open.
pub fn calm(routes: &[Route], busy: bool) -> bool {
    !busy && matches!(routes.last(), Some(Route::Projects | Route::Project(_)))
}

pub fn price_label(price: u32, language: crate::i18n::Language) -> String {
    match language {
        crate::i18n::Language::En => format!("€{price}"),
        _ => format!("{price} €"),
    }
}

#[derive(Clone, Copy)]
pub struct Survey {
    progress: Stored<Progress>,
    showing: Stored<bool>,
}
impl Survey {
    pub fn load() -> Self {
        Self {
            progress: stored(
                Storage::get_item(KEY)
                    .and_then(|v| serde_json::from_str(&v).ok())
                    .unwrap_or_default(),
            ),
            showing: stored(false),
        }
    }
    fn save(self) {
        let value = self.progress.with(|p| serde_json::to_string(p));
        if let Ok(value) = value {
            let _ = Storage::set_item(KEY, value);
        }
    }
    pub fn start(self, state: State, testing: Testing) {
        let retried = stored(false);
        let _ = set_interval(Duration::from_secs(TICK_SECONDS), move || {
            if !state.ready.get_untracked() || !testing.is_testing() {
                return;
            }
            if !retried.with(|r| std::mem::replace(r, true)) {
                self.send(testing);
            }
            if lifecycle() != Lifecycle::Active || self.progress.with(|p| p.asked) {
                return;
            }
            self.progress.with(|p| p.active_seconds += TICK_SECONDS);
            self.save();
            let calm = state.routes.with_untracked(|r| calm(r, state.busy.get_untracked()));
            if calm && self.progress.with(|p| p.due()) && !self.showing.with(|s| *s) {
                self.ask(testing);
            }
        });
    }
    fn ask(self, testing: Testing) {
        self.showing.with(|s| *s = true);
        let language = crate::i18n::language();
        let mut dialog = alert_dialog()
            .title(t(Key::PriceQuestion))
            .description(t(Key::PriceQuestionDescription))
            .action_sheet();
        for price in PRICES {
            dialog = dialog.action(format!("p{price}"), price_label(price, language));
        }
        let shown = dialog.cancel("skip", t(Key::NoAnswer)).show();
        spawn(async move {
            let answer = shown.await;
            self.showing.with(|s| *s = false);
            // A dialog the host could not show is asked again later.
            let Ok(answer) = answer else { return };
            self.progress.with(|p| p.answer(answer.action.as_deref()));
            self.save();
            self.send(testing);
        });
    }
    /// Sends the stored answer; it stays stored until the backend accepts it.
    fn send(self, testing: Testing) {
        let Some(answer) = self.progress.with(|p| p.unsent.clone()) else {
            return;
        };
        let context = testing.context();
        let record = json!({
            "install": context.install,
            "price": answer.price,
            "answered": answer.price.is_some(),
            "active_seconds": answer.active_seconds,
            "language": context.language,
            "app": context.app,
            "device": context.device,
        });
        spawn(async move {
            if super::upload::create_json("price_votes", &record).await.is_ok() {
                self.progress.with(|p| {
                    if p.unsent.as_ref() == Some(&answer) {
                        p.unsent = None;
                    }
                });
                self.save();
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_question_is_due_once_after_five_active_minutes() {
        let mut progress = Progress {
            active_seconds: 290,
            ..Default::default()
        };
        assert!(!progress.due());
        progress.active_seconds += TICK_SECONDS;
        assert!(progress.due());
        let answer = progress.answer(Some("p15"));
        assert_eq!(answer.price, Some(15));
        assert_eq!(answer.active_seconds, 300);
        assert!(!progress.due());
        assert_eq!(progress.unsent, Some(answer));
    }

    #[test]
    fn declining_or_an_unknown_action_records_no_price_but_still_ends_the_question() {
        for action in [None, Some("skip"), Some("p7"), Some("15")] {
            let mut progress = Progress {
                active_seconds: 400,
                ..Default::default()
            };
            assert_eq!(progress.answer(action).price, None);
            assert!(progress.asked);
        }
    }

    #[test]
    fn progress_written_by_this_build_survives_a_reload_and_old_values_default() {
        let progress: Progress = serde_json::from_str("{}").unwrap();
        assert_eq!(progress, Progress::default());
        let mut asked = Progress {
            active_seconds: 310,
            ..Default::default()
        };
        asked.answer(Some("p0"));
        let restored: Progress =
            serde_json::from_str(&serde_json::to_string(&asked).unwrap()).unwrap();
        assert_eq!(restored, asked);
        assert_eq!(restored.unsent.unwrap().price, Some(0));
    }

    #[test]
    fn only_idle_collection_screens_are_calm() {
        assert!(calm(&[Route::Projects], false));
        assert!(calm(&[Route::Projects, Route::Project(1)], false));
        assert!(!calm(&[Route::Projects, Route::Project(1)], true));
        assert!(!calm(&[Route::Projects, Route::Camera(1, None)], false));
        assert!(!calm(&[Route::Projects, Route::Viewer(1, 2)], false));
        assert!(!calm(&[Route::Projects, Route::Export(1, vec![])], false));
    }

    #[test]
    fn fifteen_euros_is_offered_and_labels_follow_the_language() {
        assert!(PRICES.contains(&15));
        assert_eq!(price_label(15, crate::i18n::Language::De), "15 €");
        assert_eq!(price_label(15, crate::i18n::Language::En), "€15");
    }
}
