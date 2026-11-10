//! Styled application feedback on NSH's toast lifecycle, gesture and a11y roots.
use crate::i18n::{Key, t};
use crate::state::{Route, State};
use crate::ui;
use nsh_base::{
    Anchor, SwipeDirection, ToastManager, ToastMotion, ToastOptions, ToastStackState,
    ToastTransitionStatus, toast, toast_stack,
};
use nsh_core::{prelude::*, time::now};
use std::time::Duration;

#[derive(Clone, Debug, PartialEq)]
struct Message {
    id: u64,
    text: String,
    deletion: Option<u64>,
    duration: u64,
}
#[derive(Clone, Copy, Debug, PartialEq)]
struct Preferences {
    info: u64,
    action: u64,
    persistent_action: bool,
    foreground: bool,
}
impl Default for Preferences {
    fn default() -> Self {
        Self {
            info: 3000,
            action: 6000,
            persistent_action: false,
            foreground: true,
        }
    }
}
#[derive(Clone, Copy)]
pub(crate) struct Feedback {
    state: State,
    current: Signal<Option<Message>>,
    manager: Stored<ToastManager<u64, Message>>,
    shown: Signal<Vec<(u64, Message, ToastTransitionStatus)>>,
    stack: ToastStackState,
    preferences: Signal<Preferences>,
    motion: ToastMotion,
}
impl Feedback {
    pub fn new(state: State) -> Self {
        let mut motion = ToastMotion::sonner();
        motion.duration = Duration::from_millis(180);
        motion.exit_duration = Duration::from_millis(120);
        motion.collapsed_visible = 1;
        let this = Self {
            state,
            current: signal(None),
            manager: stored(ToastManager::new(motion)),
            shown: signal(vec![]),
            stack: ToastStackState::new(),
            preferences: signal(Preferences::default()),
            motion,
        };
        effect(move || {
            if state.ready.get() {
                state
                    .service
                    .command("feedback_preferences", serde_json::json!({}));
            }
        });
        effect(move || {
            let ready = state.ready.get();
            let notice = state.notice.get();
            let undo = state.undo.get();
            let p = this.preferences.get();
            if !ready {
                return;
            }
            // Preserve Undo ahead of background warnings; those remain queued
            // in notice until the active deletion message closes.
            let (text, deletion) = undo.map_or((notice, None), |d| {
                (d.localized_message(crate::i18n::language()), Some(d.id))
            });
            let old = this.current.get_untracked();
            if text.is_empty() {
                if let Some(old) = old {
                    this.dismiss(old.id);
                    this.current.set(None);
                }
                return;
            }
            let duration = if deletion.is_some() {
                p.action
            } else if screen_reader() || text.chars().count() > 80 {
                p.info.max(8000)
            } else {
                p.info
            };
            let same = old
                .as_ref()
                .is_some_and(|m| m.text == text && m.deletion == deletion);
            if same && old.as_ref().unwrap().duration == duration {
                return;
            }
            let id = if same {
                old.as_ref().unwrap().id
            } else {
                crate::model::id()
            };
            if !same {
                if let Some(old) = old {
                    this.dismiss(old.id);
                }
                state
                    .service
                    .command("feedback_preferences", serde_json::json!({}));
            }
            let message = Message {
                id,
                text,
                deletion,
                duration,
            };
            this.current.set(Some(message.clone()));
            this.manager.with(|m| {
                m.push(
                    id,
                    message,
                    ToastOptions::timeout(Some(Duration::from_millis(duration))),
                    now(),
                )
            });
            this.sync();
        });
        let active = computed(move || !this.shown.get().is_empty());
        effect(move || {
            if !active.get() {
                return;
            }
            let tick = set_interval(Duration::from_millis(32), move || {
                let p = this.preferences.get_untracked();
                let action = this
                    .current
                    .get_untracked()
                    .is_some_and(|m| m.deletion.is_some());
                let paused = this.stack.is_expanded()
                    || !p.foreground
                    || (action && (screen_reader() || p.persistent_action));
                let advanced = this.manager.with(|m| m.advance(now(), paused));
                if advanced.changed {
                    this.sync();
                }
                for (id, _) in advanced.removed {
                    this.complete(id);
                }
            });
            cleanup(move || tick.cancel());
        });
        this
    }
    fn sync(self) {
        self.shown.set(
            self.manager
                .with(|m| m.visible(1).map(|(id, m, s)| (*id, m.clone(), s)).collect()),
        );
    }
    fn dismiss(self, id: u64) {
        self.manager.with(|m| m.dismiss(&id, now()));
        self.sync();
    }
    fn complete(self, id: u64) {
        let Some(m) = self.current.get_untracked().filter(|m| m.id == id) else {
            return;
        };
        batch(|| {
            self.current.set(None);
            if self.state.notice.get_untracked() == m.text {
                self.state.notice.set(String::new());
            }
            if m.deletion
                .is_some_and(|id| self.state.undo.get_untracked().is_some_and(|d| d.id == id))
            {
                self.state.undo.set(None);
            }
        });
    }
    pub fn event(self, event: &NativeEvent) {
        if &*event.name != "feedback_preferences" {
            return;
        }
        let read = |key: &str, fallback| event.data[key].as_u64().unwrap_or(fallback).max(1000);
        self.preferences.set(Preferences {
            info: read("info_ms", 3000),
            action: read("action_ms", 6000),
            persistent_action: event.data["persistent_action"].as_bool().unwrap_or(false),
            foreground: event.data["foreground"].as_bool().unwrap_or(true),
        });
    }
    pub fn view(self) -> View {
        view()
            .absolute()
            .left(px(0.))
            .right(px(0.))
            .z_index(20)
            .bottom(move || {
                let clearance = if matches!(self.state.routes.get().last(), Some(Route::Camera(..)))
                {
                    104.
                } else {
                    0.
                };
                px(safe_area().bottom.0.max(keyboard_height().0) + 12. + clearance)
            })
            .pl(move || px(safe_area().left.0 + 16.))
            .pr(move || px(safe_area().right.0 + 16.))
            .items_center()
            .child(
                toast_stack()
                    .placement(Anchor::BottomCenter)
                    .state(self.stack)
                    .close_on_escape(false)
                    .motion(self.motion)
                    .w(relative(1.))
                    .max_w(px(560.))
                    .child(each(
                        self.shown,
                        |(id, _, _)| *id,
                        move |item, _| {
                            let id = item.get_untracked().0;
                            toast()
                                .transition_status(move || item.get().2)
                                .accessibility_label(move || Some(item.get().1.text.into()))
                                .swipe_direction(Some(if cfg!(target_os = "android") {
                                    SwipeDirection::Right
                                } else {
                                    SwipeDirection::Down
                                }))
                                .on_dismiss_request(move || self.dismiss(id))
                                .rounded(px(14.))
                                .bg(rgb(ui::RAISED))
                                .px_3()
                                .py_2()
                                .transition_all()
                                .duration(move || {
                                    if item.get().2 == ToastTransitionStatus::Ending {
                                        120_f32
                                    } else {
                                        180_f32
                                    }
                                })
                                .starting_class(if reduce_motion() {
                                    "opacity-0"
                                } else {
                                    "opacity-0 translate-y-2"
                                })
                                .ending_class(if reduce_motion() {
                                    "opacity-0"
                                } else {
                                    "opacity-0 translate-y-1"
                                })
                                .child(
                                    view()
                                        .test_id("toast")
                                        // Keep the dismiss area and Undo as separate
                                        // VoiceOver/TalkBack targets, even though
                                        // tapping empty card space also dismisses it.
                                        .accessible(false)
                                        .on_press(move || self.dismiss(id))
                                        .flex_row()
                                        .items_center()
                                        .gap_2()
                                        .child(
                                            view()
                                                .flex_1()
                                                .min_h(px(if cfg!(target_os = "android") {
                                                    48.
                                                } else {
                                                    44.
                                                }))
                                                .justify_center()
                                                .role(AccessibilityRole::Button)
                                                .on_press(move || self.dismiss(id))
                                                .child(
                                                    text(move || item.get().1.text)
                                                        .text_size(px(15.))
                                                        .text_color(rgb(0xf2f2f2)),
                                                ),
                                        )
                                        .child(when(
                                            move || item.get().1.deletion.is_some(),
                                            move || {
                                                ui::link(t(Key::Undo), move || {
                                                    if self
                                                        .current
                                                        .get_untracked()
                                                        .is_some_and(|m| m.id == id)
                                                    {
                                                        self.state.restore();
                                                    }
                                                })
                                                .min_h(px(if cfg!(target_os = "android") {
                                                    48.
                                                } else {
                                                    44.
                                                }))
                                            },
                                        )),
                                )
                        },
                    )),
            )
    }
}
