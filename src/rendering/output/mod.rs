//! Final output preparation; these images never become detection input.
mod a4;
mod appearance;
mod border;
mod color;
mod crop;
mod distance;
mod paper;
#[cfg(test)]
mod tests;
mod tone;

pub use a4::A4Frame;
pub(crate) use appearance::preserve_color;
pub use border::{
    apply_binary_cleanup, apply_binary_cleanup_with_contour, whiten_gray_border, whiten_page_border,
};
#[cfg(feature = "application")]
pub(crate) use crop::cropped;
pub use crop::{content_crop, page_crop};
pub use color::enhance_color;
pub use paper::{enhance_print, normalize_paper};
pub(crate) use tone::grayscale;
#[cfg(feature = "application")]
pub(crate) use tone::normalize_text_analysis;
pub use tone::{antialias_print, binarize_sauvola};

use border::{whiten_exterior_impl, whiten_gray_border_impl, whiten_page_border_impl};
use image::GrayImage;
use std::sync::OnceLock;

/// Shared page geometry for binary and gray cleanup. Distance fields depend
/// only on the perimeter, not on tone, so compute each at most once per page.
pub(crate) struct PageBorder<'a> {
    contour: &'a [crate::geometry::Point],
    mask: OnceLock<Vec<u8>>,
    rim: OnceLock<Vec<u16>>,
    boundary: OnceLock<Vec<u16>>,
}

impl<'a> PageBorder<'a> {
    pub(crate) fn new(contour: &'a [crate::geometry::Point]) -> Self {
        Self {
            contour,
            mask: OnceLock::new(),
            rim: OnceLock::new(),
            boundary: OnceLock::new(),
        }
    }
    pub(crate) fn whiten_binary(&self, image: &mut GrayImage, margin: u32) -> usize {
        whiten_page_border_impl(image, self.contour, margin, false, None, Some(self))
    }
    /// Remove shape-confirmed page rims while preserving measured content and
    /// the uncertain contour band. No Sauvola mask decides which ink survives.
    pub(crate) fn clean_color(&self, rgb: &mut image::RgbImage, margin: u32) {
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let mask = self
            .mask
            .get_or_init(|| border::page_mask(self.contour, w, h));
        let margin = margin.min(u32::from(u16::MAX - 1)) as u16;
        let distance = distance::distance_to_contour(self.contour, w, h, margin + 1);
        use rayon::prelude::*;
        rgb.as_mut()
            .par_chunks_mut(3)
            .zip(mask)
            .zip(distance)
            .for_each(|((p, inside), d)| {
                if *inside == 0 && d > margin {
                    p.fill(255);
                }
            });
    }
    pub(crate) fn clean_gray(&self, gray: &mut GrayImage, margin: u32, protected: &[[f64; 4]]) {
        whiten_gray_border_impl(gray, self.contour, margin, protected, Some(self));
        whiten_exterior_impl(gray, self.contour, margin, self);
    }
}
