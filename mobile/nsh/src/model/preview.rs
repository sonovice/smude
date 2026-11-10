//! One source of truth for the image and comparison label, including old previews.
use super::{Page, PageStatus};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreviewKind {
    Original,
    Result,
    PreviousResult,
}
#[derive(Clone, Copy, Debug)]
pub struct PagePreview<'a> {
    pub path: &'a str,
    pub kind: PreviewKind,
    pub rotation: f32,
}
impl Page {
    pub fn has_result(&self) -> bool {
        self.output.is_some() || self.replacement.as_ref().is_some_and(|p| p.has_result())
    }
    pub fn can_share(&self) -> bool {
        self.status == PageStatus::Ready && self.output.is_some()
    }
    pub fn preview(&self, prefer_original: bool) -> Option<PagePreview<'_>> {
        let original = || {
            self.source.as_deref().map(|path| PagePreview {
                path,
                kind: PreviewKind::Original,
                rotation: self.preview_rotation(true),
            })
        };
        if prefer_original && let Some(image) = original() {
            return Some(image);
        }
        if let Some(path) = self.output.as_deref() {
            return Some(PagePreview {
                path,
                kind: if self.status == PageStatus::Ready {
                    PreviewKind::Result
                } else {
                    PreviewKind::PreviousResult
                },
                rotation: self.preview_rotation(false),
            });
        }
        if let Some(previous) = self.replacement.as_ref().and_then(|p| p.preview(false))
            && previous.kind != PreviewKind::Original
        {
            return Some(PagePreview {
                kind: PreviewKind::PreviousResult,
                ..previous
            });
        }
        original()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Library;
    #[test]
    fn missing_result_never_claims_to_be_processed_and_missing_images_stay_empty() {
        let mut page = Library::examples().projects.remove(0).pages.remove(0);
        page.output = None;
        for state in [
            PageStatus::Importing,
            PageStatus::Pending,
            PageStatus::Processing,
            PageStatus::Failed,
        ] {
            page.status = state;
            assert_eq!(page.preview(false).unwrap().kind, PreviewKind::Original);
            assert!(!page.has_result());
            assert!(!page.can_share());
        }
        page.source = None;
        assert!(page.preview(false).is_none());
        assert!(page.preview(true).is_none());
    }
    #[test]
    fn refresh_and_retake_preserve_labelled_previous_results_with_their_rotation() {
        let mut page = Library::examples().projects.remove(0).pages.remove(0);
        page.rotation = 90;
        page.invalidate();
        assert_eq!(
            page.preview(false).unwrap().kind,
            PreviewKind::PreviousResult
        );
        assert_eq!(page.preview(true).unwrap().kind, PreviewKind::Original);
        let backup = page.clone();
        page.replacement = Some(Box::new(backup));
        page.output = None;
        page.rotation = 0;
        let previous = page.preview(false).unwrap();
        assert_eq!(previous.kind, PreviewKind::PreviousResult);
        assert_eq!(previous.rotation, 90.);
        assert!(page.has_result());
        assert!(!page.can_share());
    }
}
