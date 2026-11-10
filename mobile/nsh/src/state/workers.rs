//! Independent bounded lanes keep file imports and sharing out of the render queue.
use super::*;
impl State {
    pub fn start_worker(self) {
        self.start_worker_with(|repo, job, observer| {
            repo.process_with_progress(&job, Some(observer))
        });
    }
    pub(crate) fn start_worker_with(
        self,
        process: impl Fn(
            Repository,
            Job,
            smude_geometry::application::ProcessingObserver,
        ) -> Result<RenderBatch, String>
        + Clone
        + Send
        + 'static,
    ) {
        self.start_import_worker();
        self.start_export_worker();
        spawn(async move {
            let mut profiles = crate::processing::TimingProfiles::load();
            loop {
                if let Some(repo) = self.repo() {
                    if self
                        .routes
                        .with(|r| matches!(r.last(), Some(Route::Camera(..))))
                    {
                        timer(Duration::from_millis(100)).await;
                        continue;
                    }
                    let job = self.library.with(|l| {
                        l.projects.iter().find_map(|p| {
                            p.pages
                                .iter()
                                .find(|v| v.status == PageStatus::Pending && v.source.is_some())
                                .cloned()
                                .map(|page| Job {
                                    project: p.id,
                                    page,
                                })
                        })
                    });
                    if let Some(job) = job {
                        self.working.set(Some(job.page.id));
                        let mut l = self.library.get_untracked();
                        if let Some(p) = l
                            .projects
                            .iter_mut()
                            .find(|p| p.id == job.project)
                            .and_then(|p| p.pages.iter_mut().find(|p| p.id == job.page.id))
                        {
                            p.status = PageStatus::Processing;
                        }
                        if !self.persist(l, false) {
                            self.working.set(None);
                            timer(Duration::from_secs(1)).await;
                            continue;
                        }
                        let copy = job.clone();
                        let key = (job.page.id, job.page.capture, job.page.revision);
                        let started = std::time::Instant::now();
                        let run = std::rc::Rc::new(std::cell::RefCell::new(
                            crate::processing::ProcessingRun::new(job.page.model, profiles.clone()),
                        ));
                        let (sender, receiver) = std::sync::mpsc::channel();
                        let observer: smude_geometry::application::ProcessingObserver =
                            std::sync::Arc::new(move |event| {
                                let _ = sender.send((std::time::Instant::now(), event));
                            });
                        let current = run.clone();
                        let poll = std::rc::Rc::new(move || {
                            let mut run = current.borrow_mut();
                            for (at, event) in receiver.try_iter() {
                                run.event(
                                    event,
                                    at.saturating_duration_since(started).as_secs_f64(),
                                );
                            }
                            self.processing
                                .set(Some(run.display(started.elapsed().as_secs_f64(), key)));
                        });
                        poll();
                        let tick = poll.clone();
                        let polling = set_interval(Duration::from_millis(100), move || tick());
                        let process = process.clone();
                        let result = blocking(move || process(repo, copy, observer))
                            .await
                            .map_err(|e| e.to_string())
                            .and_then(|r| r);
                        polling.cancel();
                        poll();
                        if result.is_ok() {
                            profiles.learn(&run.borrow());
                        }
                        self.finish_job(job, result);
                        self.processing.set(None);
                        self.working.set(None);
                        continue;
                    }
                }
                timer(Duration::from_millis(100)).await;
            }
        });
    }
    pub(super) fn start_import_worker(self) {
        spawn(async move {
            loop {
                if let Some(repo) = self.repo() {
                    let completed = self.library.with(|l| {
                        l.imports
                            .iter()
                            .find(|i| {
                                i.awaiting_capture
                                    && (std::path::Path::new(&i.path).is_file()
                                        || i.capture_expired())
                            })
                            .cloned()
                    });
                    if let Some(import) = completed {
                        self.capture_event(&serde_json::json!({"event":if std::path::Path::new(&import.path).is_file(){"capture_saved"}else{"capture_failed"},"request_id":import.id}));
                        continue;
                    }
                    if let Some(import) = self
                        .library
                        .with(|l| l.imports.iter().find(|i| !i.awaiting_capture).cloned())
                    {
                        let source = PathBuf::from(&import.path);
                        let owned = repo.clone();
                        let result = blocking(move || owned.import(&source))
                            .await
                            .map_err(|e| e.to_string())
                            .and_then(|r| r);
                        self.finish_import(import, result);
                        continue;
                    }
                }
                timer(Duration::from_millis(100)).await;
            }
        });
    }
    pub(super) fn start_export_worker(self) {
        spawn(async move {
            loop {
                if let Some(repo) = self.repo()
                    && let Some(request) = self.share_request.get_untracked()
                {
                    self.share_request.set(None);
                    self.prepare_share_request(repo, request).await;
                    continue;
                }
                if let Some(repo) = self.repo()
                    && let Some(request) = self.export_request.get_untracked()
                {
                    self.export_request.set(None);
                    let root = repo.root.clone();
                    let request_copy = request.clone();
                    let result = blocking(move || -> Result<Vec<String>, String> {
                        let inputs = request_copy
                            .pages
                            .iter()
                            .map(|p| crate::exporting::ExportInput {
                                path: root.join(p.output.as_ref().unwrap()),
                                rotation: p.rotation,
                            })
                            .collect::<Vec<_>>();
                        if request_copy.preview {
                            let img =
                                crate::exporting::prepare_raster(&inputs[0], request_copy.options)?;
                            let path = root.join("exports").join(format!("preview-{}.png", id()));
                            img.save(&path).map_err(|e| e.to_string())?;
                            Ok(vec![path.to_string_lossy().into_owned()])
                        } else {
                            crate::exporting::export(
                                &inputs,
                                request_copy.options,
                                &request_copy.filename,
                                &root.join("exports"),
                            )
                            .map(|v| {
                                v.into_iter()
                                    .map(|p| p.to_string_lossy().into_owned())
                                    .collect()
                            })
                        }
                    })
                    .await
                    .map_err(|e| e.to_string())
                    .and_then(|r| r);
                    match result {
                        Ok(paths) => {
                            if request.preview {
                                if self.export.get_untracked() == request.options {
                                    self.export_preview.set(paths.first().cloned());
                                }
                                self.busy.set(false);
                                self.progress.set(String::new());
                            } else {
                                self.progress.set(t(Key::Sharing).into());
                                self.service.command("share",serde_json::json!({"request_id":id(),"paths":paths,"mime":if request.options.format==Format::Pdf{"application/pdf"}else{"image/jpeg"}}));
                            }
                        }
                        Err(e) => {
                            self.busy.set(false);
                            self.progress.set(String::new());
                            self.notice.set(crate::i18n::failure(
                                crate::i18n::Failure::Export,
                                &e.to_string(),
                            ));
                        }
                    }
                    continue;
                }
                timer(Duration::from_millis(100)).await;
            }
        });
    }
}
