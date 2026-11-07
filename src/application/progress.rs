//! Detection progress in untouched, full-resolution input-image coordinates.
use crate::geometry::{Point, Staff};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectionStage {
    Page,
    /// Discard music observations from a failed analysis attempt for this page.
    ResetMusic,
    Staves,
    Barlines,
    StaffEdges,
    Text,
}
#[derive(Clone, Debug)]
pub struct DetectionProgress {
    pub page_id: String,
    pub stage: DetectionStage,
    pub paths: Vec<Vec<Point>>,
}
/// Called synchronously when measurements become available, on the processing
/// thread. Receivers should enqueue the event rather than block the pipeline.
pub type ProgressObserver = Arc<dyn Fn(DetectionProgress) + Send + Sync>;

impl super::Options {
    pub(super) fn emit(
        &self,
        page_id: &str,
        stage: DetectionStage,
        paths: impl FnOnce() -> Vec<Vec<Point>>,
    ) {
        if let Some(observer) = &self.progress {
            observer(DetectionProgress {
                page_id: page_id.into(),
                stage,
                paths: paths(),
            });
        }
    }
    pub(super) fn emit_staves(
        &self,
        page_id: &str,
        staves: &[Staff],
        map: impl Fn(Point) -> Point,
    ) {
        self.emit(page_id, DetectionStage::Staves, || {
            staves
                .iter()
                .flat_map(|s| {
                    s.y.iter().map(|line| {
                        s.x.iter()
                            .zip(line)
                            .map(|(x, y)| map(Point { x: *x, y: *y }))
                            .collect()
                    })
                })
                .collect()
        });
    }
}
