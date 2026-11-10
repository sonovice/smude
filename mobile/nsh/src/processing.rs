//! Time-weighted estimates anchored to real worker boundaries, not a fake timer.
use crate::i18n::{Key, t};
use crate::model::Model;
use serde::{Deserialize, Serialize};
use smude_geometry::application::{ProcessingEvent as Event, ProcessingStage as Stage};
use std::collections::BTreeSet;

const KEY: &str = "smude-processing-timings-v1";
const PIXELS: f64 = 49_000_000.;
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TimingProfiles {
    mesh_small: [f64; 7],
    mesh_large: [f64; 7],
    gcs_small: [f64; 7],
    gcs_large: [f64; 7],
}
impl Default for TimingProfiles {
    fn default() -> Self {
        // Initial per-stage seconds from measured iPhone photo profiles. Pixel
        // work is normalized to49MP; bounded analysis/geometry remain separate.
        let mesh = [0.20, 0.35, 1.80, 0.65, 0.55, 1.25, 0.08];
        let gcs = [0.20, 0.35, 1.00, 1.20, 0.55, 1.25, 0.08];
        Self {
            mesh_small: mesh,
            mesh_large: mesh,
            gcs_small: gcs,
            gcs_large: gcs,
        }
    }
}
impl TimingProfiles {
    pub fn load() -> Self {
        nsh_core::storage::Storage::get_item(KEY)
            .and_then(|s| serde_json::from_str(&s).ok())
            .filter(|v: &Self| {
                [v.mesh_small, v.mesh_large, v.gcs_small, v.gcs_large]
                    .iter()
                    .flatten()
                    .all(|v| v.is_finite() && *v > 0.)
            })
            .unwrap_or_default()
    }
    fn profile(&self, model: Model, small: bool) -> [f64; 7] {
        match (model, small) {
            (Model::Mesh, true) => self.mesh_small,
            (Model::Mesh, false) => self.mesh_large,
            (Model::Gcs, true) => self.gcs_small,
            (Model::Gcs, false) => self.gcs_large,
        }
    }
    pub fn learn(&mut self, run: &ProcessingRun) {
        if !self.update(run) {
            return;
        }
        if let Ok(json) = serde_json::to_string(self) {
            let _ = nsh_core::storage::Storage::set_item(KEY, json);
        }
    }
    fn update(&mut self, run: &ProcessingRun) -> bool {
        // Cached results and failed/cancelled runs must not train durations.
        if !run.finished || run.counts[Stage::Rendering as usize] == 0 {
            return false;
        }
        let profile = match (run.model, run.pixels < 8_000_000) {
            (Model::Mesh, true) => &mut self.mesh_small,
            (Model::Mesh, false) => &mut self.mesh_large,
            (Model::Gcs, true) => &mut self.gcs_small,
            (Model::Gcs, false) => &mut self.gcs_large,
        };
        for (i, value) in profile.iter_mut().enumerate() {
            if run.counts[i] > 0 {
                let measured = (run.durations[i] / f64::from(run.counts[i])).clamp(0.005, 120.);
                *value = (*value * 0.75 + measured * 0.25).clamp(0.005, 120.);
            }
        }
        true
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProcessingDisplay {
    pub page_id: u64,
    pub capture: u64,
    pub revision: u64,
    pub percent: f32,
    pub stage: Stage,
    pub page_index: usize,
    pub page_count: usize,
}
impl ProcessingDisplay {
    pub fn label(self) -> String {
        let label = match self.stage {
            Stage::Preparing => t(Key::Preparing),
            Stage::PageDetection => t(Key::DetectingPage),
            Stage::Analysis => t(Key::AnalyzingPage),
            Stage::Dewarping => t(Key::CorrectingGeometry),
            Stage::Rendering => t(Key::RenderingImage),
            Stage::Enhancement => t(Key::EnhancingImage),
            Stage::Saving => match crate::i18n::language() {
                crate::i18n::Language::En => "Saving",
                crate::i18n::Language::De => "Speichern",
                crate::i18n::Language::Pt => "Salvando",
            },
        };
        if self.page_count > 1 {
            format!(
                "{} · {label}",
                crate::i18n::language().page_of(self.page_index + 1, self.page_count)
            )
        } else {
            label.into()
        }
    }
}
pub struct ProcessingRun {
    model: Model,
    profiles: TimingProfiles,
    pixels: u64,
    page: Option<usize>,
    pages: usize,
    completed: usize,
    stage: Stage,
    started: f64,
    active: bool,
    finished: bool,
    durations: [f64; 7],
    counts: [u32; 7],
    seen: BTreeSet<(usize, usize)>,
    shown: f32,
}
impl ProcessingRun {
    pub fn new(model: Model, profiles: TimingProfiles) -> Self {
        Self {
            model,
            profiles,
            pixels: 49_000_000,
            page: None,
            pages: 1,
            completed: 0,
            stage: Stage::Preparing,
            started: 0.,
            active: true,
            finished: false,
            durations: [0.; 7],
            counts: [0; 7],
            seen: BTreeSet::new(),
            shown: 0.,
        }
    }
    fn pixel_scale(&self, stage: Stage) -> f64 {
        match stage {
            Stage::Preparing => (self.pixels as f64 / PIXELS).clamp(0.03, 4.),
            Stage::Rendering | Stage::Enhancement | Stage::Saving => {
                (self.pixels as f64 / PIXELS / self.pages as f64).clamp(0.03, 4.)
            }
            _ => 1.,
        }
    }
    fn close(&mut self, now: f64) {
        if !self.active {
            return;
        }
        let i = self.stage as usize;
        self.durations[i] += (now - self.started).max(0.) / self.pixel_scale(self.stage);
        let page = if i < 2 {
            usize::MAX
        } else {
            self.page.unwrap_or(0)
        };
        if self.seen.insert((page, i)) {
            self.counts[i] += 1;
        }
        self.active = false;
    }
    pub fn event(&mut self, event: Event, now: f64) {
        match event {
            Event::InputPixels(pixels) => self.pixels = pixels.max(1),
            Event::Stage(stage) => {
                if !self.active || stage != self.stage {
                    self.close(now);
                    self.stage = stage;
                    self.started = now;
                    self.active = true;
                }
            }
            Event::PageStarted { index, count } => {
                self.close(now);
                self.page = Some(index);
                self.pages = count.max(1);
                self.stage = Stage::Analysis;
                self.started = now;
                self.active = true;
            }
            Event::PageFinished { index } => {
                self.close(now);
                self.completed = self.completed.max(index + 1);
            }
            Event::Finished => {
                self.close(now);
                self.finished = true;
            }
        }
    }
    pub fn display(&mut self, now: f64, key: (u64, u64, u64)) -> ProcessingDisplay {
        let base = self.profiles.profile(self.model, self.pixels < 8_000_000);
        let stages = [
            Stage::Preparing,
            Stage::PageDetection,
            Stage::Analysis,
            Stage::Dewarping,
            Stage::Rendering,
            Stage::Enhancement,
            Stage::Saving,
        ];
        let times: [f64; 7] = std::array::from_fn(|i| base[i] * self.pixel_scale(stages[i]));
        let root = times[0] + times[1];
        let page_total = times[2..].iter().sum::<f64>();
        let total = root + page_total * self.pages as f64;
        let i = self.stage as usize;
        let done = if self.finished {
            total
        } else if !self.active {
            root + page_total * self.completed as f64
        } else if i < 2 {
            times[..i].iter().sum::<f64>() + (now - self.started).max(0.).min(times[i] * 0.9)
        } else {
            root + page_total * self.page.unwrap_or(0) as f64
                + times[2..i].iter().sum::<f64>()
                + (now - self.started).max(0.).min(times[i] * 0.9)
        };
        // Reanalysis and discovering a second page never move the bar backwards.
        // 100% belongs to publication of Ready, not to a timing prediction.
        self.shown = self.shown.max((100. * done / total).clamp(0., 99.) as f32);
        ProcessingDisplay {
            page_id: key.0,
            capture: key.1,
            revision: key.2,
            percent: self.shown,
            stage: self.stage,
            page_index: self.page.unwrap_or(0),
            page_count: self.pages,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_slow_stage_stops_before_its_boundary_and_finishing_never_claims_ready() {
        let mut p = ProcessingRun::new(Model::Mesh, TimingProfiles::default());
        p.event(Event::PageStarted { index: 0, count: 1 }, 0.);
        p.event(Event::Stage(Stage::Rendering), 1.);
        let stalled = p.display(100., (1, 2, 3)).percent;
        assert!(stalled < 90.);
        assert_eq!(p.display(1000., (1, 2, 3)).percent, stalled);
        p.event(Event::Stage(Stage::Saving), 1001.);
        assert!(p.display(2000., (1, 2, 3)).percent < 100.);
        p.event(Event::Finished, 2001.);
        assert_eq!(p.display(2001., (1, 2, 3)).percent, 99.);
    }
    #[test]
    fn spreads_and_retries_are_monotone_and_keep_the_second_page_in_the_budget() {
        let mut p = ProcessingRun::new(Model::Mesh, TimingProfiles::default());
        p.event(Event::Stage(Stage::PageDetection), 0.2);
        let a = p.display(0.5, (1, 2, 3)).percent;
        p.event(Event::PageStarted { index: 0, count: 2 }, 0.6);
        p.event(Event::Stage(Stage::Saving), 1.);
        let b = p.display(2., (1, 2, 3)).percent;
        assert!(b >= a && b < 65.);
        p.event(Event::PageFinished { index: 0 }, 2.1);
        p.event(Event::PageStarted { index: 1, count: 2 }, 2.2);
        let c = p.display(2.3, (1, 2, 3)).percent;
        assert!(c >= b);
        p.event(Event::Stage(Stage::Dewarping), 3.);
        let d = p.display(3.5, (1, 2, 3)).percent;
        p.event(Event::Stage(Stage::Analysis), 3.6);
        assert!(p.display(3.7, (1, 2, 3)).percent >= d);
    }
    #[test]
    fn cache_hits_do_not_train_processing_times() {
        let mut p = ProcessingRun::new(Model::Mesh, TimingProfiles::default());
        p.event(Event::Finished, 0.001);
        let mut profiles = TimingProfiles::default();
        let before = profiles.mesh_large;
        profiles.learn(&p);
        assert_eq!(before, profiles.mesh_large);
    }
    #[test]
    fn real_completed_stages_update_only_the_matching_model_and_size_profile() {
        let mut run = ProcessingRun::new(Model::Mesh, TimingProfiles::default());
        run.event(Event::Stage(Stage::Rendering), 1.);
        run.event(Event::Stage(Stage::Enhancement), 2.1);
        run.event(Event::Stage(Stage::Saving), 3.35);
        run.event(Event::PageFinished { index: 0 }, 3.45);
        let mut profiles = TimingProfiles::default();
        assert!(
            !profiles.update(&run),
            "unfinished jobs are not calibration data"
        );
        run.event(Event::Finished, 3.46);
        assert!(profiles.update(&run));
        assert!((profiles.mesh_large[4] - 0.6875).abs() < 1e-8);
        assert_eq!(profiles.mesh_small[4], 0.55);
        assert_eq!(profiles.gcs_large[4], 0.55);
    }
}
