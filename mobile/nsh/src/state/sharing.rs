//! A share screen owns one cancelable preparation; late results never replace newer pages.
use super::*;
use crate::exporting::{ExportInput, sharing::PreparedPdf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone)]
pub struct Request {
    pub key: String,
    pub pages: Vec<Page>,
    pub title: String,
    pub cancel: Arc<AtomicBool>,
}
#[derive(Clone)]
pub struct Prepared {
    pub key: String,
    pub profiles: [Option<PreparedPdf>; 3],
    pub error: Option<String>,
    pub cancel: Arc<AtomicBool>,
}
impl PartialEq for Request {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl PartialEq for Prepared {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.profiles == other.profiles && self.error == other.error
    }
}
impl State {
    pub fn prepare_share(self, project: u64, ids: &[u64]) {
        let collection = self.project(project);
        let pages = collection.ordered_selection(ids);
        if pages.is_empty()
            || pages
                .iter()
                .any(|p| p.status != PageStatus::Ready || p.output.is_none())
        {
            self.cancel_share();
            return;
        }
        let key = format!(
            "{}:{:?}",
            collection.title,
            pages
                .iter()
                .map(|p| (p.id, p.capture, p.revision, p.rotation, p.output.clone()))
                .collect::<Vec<_>>()
        );
        if self
            .share_ready
            .get_untracked()
            .is_some_and(|p| p.key == key)
        {
            return;
        }
        self.cancel_share();
        let cancel = Arc::new(AtomicBool::new(false));
        self.share_ready.set(Some(Prepared {
            key: key.clone(),
            profiles: [None, None, None],
            error: None,
            cancel: cancel.clone(),
        }));
        self.share_request.set(Some(Request {
            key,
            pages,
            title: collection.title,
            cancel,
        }));
    }
    pub fn cancel_share(self) {
        if let Some(ready) = self.share_ready.get_untracked() {
            ready.cancel.store(true, Ordering::Relaxed);
        }
        self.share_request.set(None);
        self.share_ready.set(None);
    }
    pub fn share_pdf(self, filename: String) {
        if self.busy.get_untracked() {
            return;
        }
        let Some(repo) = self.repo() else {
            return;
        };
        let Some(profile) = self.share_ready.get_untracked().and_then(|p| {
            p.profiles[crate::exporting::sharing::index(self.export.get_untracked().size)].clone()
        }) else {
            return;
        };
        self.busy.set(true);
        self.progress.set(t(Key::PreparingExport).into());
        spawn(async move {
            let result = blocking(move || {
                crate::exporting::sharing::publish(&profile, &filename, &repo.root.join("exports"))
            })
            .await
            .map_err(|e| e.to_string())
            .and_then(|r| r);
            match result {
                Ok(path)=>self.service.command("share",serde_json::json!({"request_id":id(),"paths":[path.to_string_lossy()],"mime":"application/pdf"})),
                Err(error)=>{self.busy.set(false); self.progress.set(String::new()); self.notice.set(crate::i18n::failure(crate::i18n::Failure::Export,&error));}
            }
        });
    }
    pub(super) async fn prepare_share_request(self, repo: Repository, request: Request) {
        let (sender, receiver) = std::sync::mpsc::channel();
        let key = request.key.clone();
        let poll = std::rc::Rc::new(move || {
            for (index, profile) in receiver.try_iter() {
                self.share_ready.update(|p| {
                    if let Some(p) = p
                        && p.key == key
                    {
                        p.profiles[index] = Some(profile);
                    }
                });
            }
        });
        let tick = poll.clone();
        let timer = set_interval(Duration::from_millis(100), move || tick());
        let key = request.key.clone();
        let result = blocking(move || {
            let inputs = request
                .pages
                .iter()
                .map(|p| ExportInput {
                    path: repo.root.join(p.output.as_ref().unwrap()),
                    rotation: p.rotation,
                })
                .collect::<Vec<_>>();
            crate::exporting::sharing::prepare(
                &inputs,
                &request.title,
                &repo.root.join("exports"),
                &request.cancel,
                move |i, p| {
                    let _ = sender.send((i, p));
                },
            )
        })
        .await
        .map_err(|e| e.to_string())
        .and_then(|r| r);
        timer.cancel();
        poll();
        if let Err(error) = result {
            self.share_ready.update(|p| {
                if let Some(p) = p
                    && p.key == key
                {
                    p.error = Some(error);
                }
            });
        }
    }
}
