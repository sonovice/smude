//! Final output preparation; these images never become detection input.
mod border;
mod crop;
mod distance;
#[cfg(test)]
mod tests;
mod tone;

pub use border::{
    apply_binary_cleanup, apply_binary_cleanup_with_contour, whiten_gray_border, whiten_page_border,
};
pub use crop::{content_crop, page_crop};
pub use tone::{antialias_print, binarize_sauvola, normalize_paper};

use border::{apply_binary_cleanup_impl, whiten_gray_border_impl, whiten_page_border_impl};
use image::GrayImage;
use std::sync::{
    OnceLock,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

/// Shared page geometry for binary and gray cleanup. Distance fields depend
/// only on the perimeter, not on tone, so compute each at most once per page.
pub(crate) struct PageBorder<'a> {
    contour: &'a [crate::geometry::Point],
    mask: OnceLock<Vec<u8>>,
    rim: OnceLock<Vec<u16>>,
    boundary: OnceLock<Vec<u16>>,
    use_gpu: bool,
    gpu_calls: AtomicUsize,
    gpu_error: OnceLock<String>,
    gpu_nanoseconds: AtomicU64,
}

impl<'a> PageBorder<'a> {
    pub(crate) fn new(contour: &'a [crate::geometry::Point]) -> Self {
        Self {
            contour,
            mask: OnceLock::new(),
            rim: OnceLock::new(),
            boundary: OnceLock::new(),
            use_gpu: false,
            gpu_calls: AtomicUsize::new(0),
            gpu_error: OnceLock::new(),
            gpu_nanoseconds: AtomicU64::new(0),
        }
    }
    pub(crate) fn with_gpu(mut self, enabled: bool) -> Self {
        // Full-page distance readbacks regress on the tested Android Vulkan
        // device. Keep its exact parallel CPU transform while still using
        // GPU bicubic sampling and paper normalization.
        self.use_gpu = enabled && !cfg!(target_os = "android");
        if enabled && cfg!(target_os = "android") {
            let _ = self
                .gpu_error
                .set("CPU distance fields selected on Android after device profiling".into());
        }
        self
    }
    pub(crate) fn gpu_stats(&self) -> (usize, f64, Option<String>) {
        (
            self.gpu_calls.load(Ordering::Relaxed),
            self.gpu_nanoseconds.load(Ordering::Relaxed) as f64 / 1e9,
            self.gpu_error.get().cloned(),
        )
    }
    #[cfg(feature = "gpu-render")]
    fn record_gpu(&self, start: std::time::Instant) {
        self.gpu_calls.fetch_add(1, Ordering::Relaxed);
        self.gpu_nanoseconds
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    pub(crate) fn whiten_binary(&self, image: &mut GrayImage, margin: u32) -> usize {
        whiten_page_border_impl(image, self.contour, margin, false, None, Some(self))
    }
    pub(crate) fn clean_gray(
        &self,
        gray: &mut GrayImage,
        raw: &GrayImage,
        binary: &GrayImage,
        margin: u32,
        protected: &[[f64; 4]],
    ) {
        apply_binary_cleanup_impl(gray, raw, binary, Some(self.contour), Some(self));
        whiten_gray_border_impl(gray, self.contour, margin, protected, Some(self));
    }
}
