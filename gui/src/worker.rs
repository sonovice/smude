use eframe::egui;
use smude_geometry::application::{ProcessingResult, Settings};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        mpsc::{self, Receiver, Sender},
    },
    thread,
    time::Instant,
};

pub struct Work {
    pub id: u64,
    pub input: PathBuf,
    pub settings: Settings,
}
pub struct Finished {
    pub id: u64,
    pub seconds: f64,
    pub result: Result<ProcessingResult, String>,
}
pub struct LiveProgress {
    pub id: u64,
    pub event: Arc<smude_geometry::application::DetectionProgress>,
}
pub struct Processor {
    pub progress: Receiver<LiveProgress>,
    tx: Sender<Work>,
    pub rx: Receiver<Finished>,
}
impl Processor {
    pub fn new(cache: Arc<tempfile::TempDir>, ctx: egui::Context) -> Self {
        let (tx, jobs) = mpsc::channel::<Work>();
        let (completed, rx) = mpsc::channel();
        let (progress_tx, progress) = mpsc::channel();
        thread::Builder::new().name("smude-processing".into()).spawn(move || {
            // Leave room for the UI, but do not cap large machines at four workers.
            let available = std::thread::available_parallelism().map_or(3, |n| n.get());
            let reserved = if available >= 8 { 2 } else { 1 };
            let threads = available.saturating_sub(reserved).clamp(1, 8);
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("processing pool");
                while let Ok(job) = jobs.recv() {
                let start = Instant::now();
                let output = cache.path().join(format!("job-{}", job.id));
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if output.exists() { std::fs::remove_dir_all(&output).map_err(|e| e.to_string())?; }
                    let tx = progress_tx.clone(); let ctx = ctx.clone(); let id = job.id;
                    let observer = Arc::new(move |event| {
                        let _ = tx.send(LiveProgress { id, event: Arc::new(event) });
                        ctx.request_repaint();
                    });
                    pool.install(|| smude_geometry::application::process_file_with_progress(&job.input, &output, &job.settings, Some(observer)).map_err(|e| e.to_string()))
                })).unwrap_or_else(|_| Err("Die Verarbeitung wurde unerwartet abgebrochen. Erneut versuchen oder eine andere Aufnahme verwenden.".into()));
                if completed.send(Finished { id: job.id, seconds: start.elapsed().as_secs_f64(), result }).is_err() { break; }
                ctx.request_repaint();
            }
        }).expect("processing worker");
        Self { tx, rx, progress }
    }
    pub fn submit(&self, work: Work) -> Result<(), String> {
        self.tx
            .send(work)
            .map_err(|_| "Der Verarbeitungsdienst ist nicht verfügbar.".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_inputs_do_not_stop_the_serial_worker() {
        let cache = Arc::new(tempfile::tempdir().unwrap());
        let worker = Processor::new(cache.clone(), egui::Context::default());
        for id in [41, 42] {
            worker
                .submit(Work {
                    id,
                    input: cache.path().join("missing.png"),
                    settings: Settings::default(),
                })
                .unwrap();
        }
        for id in [41, 42] {
            let finished = worker
                .rx
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap();
            assert_eq!(finished.id, id);
            assert!(finished.result.is_err());
        }
    }
}
