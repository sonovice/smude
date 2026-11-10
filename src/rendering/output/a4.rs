//! Final DIN-A framing: preserve sampled ink pixels and replace only exterior whitespace.
use crate::geometry::Point;
use image::{GrayImage, Luma, Rgb, RgbImage};
use serde::Serialize;

const RATIO_TOLERANCE: f64 = 0.04;
const MARGIN_MM: u32 = 10;

#[derive(Clone, Debug, Serialize)]
pub struct A4Frame {
    pub paper_size_mm: [u32; 2],
    pub source_crop: [u32; 4],
    pub output_size: [u32; 2],
    pub content_offset: [u32; 2],
    pub minimum_margin_pixels: u32,
    /// Actual whitespace at left, right, top and bottom, including aspect-fit padding.
    pub margins_mm: [f64; 4],
    pub estimated_page_height_width_ratio: f64,
}
impl A4Frame {
    /// Classification uses the physical-page estimate, never the ink rectangle.
    pub fn plan(image: &GrayImage, ratio: Option<f64>, protected: &[[f64; 4]]) -> Option<Self> {
        let ratio = ratio.filter(|r| r.is_finite() && *r > 0.)?;
        let paper = if (ratio / (297. / 210.) - 1.).abs() <= RATIO_TOLERANCE {
            [210, 297]
        } else if (ratio / (210. / 297.) - 1.).abs() <= RATIO_TOLERANCE {
            [297, 210]
        } else {
            return None;
        };
        let crop = ink_bounds(image, protected)?;
        // Physical PDFs are exactly A4; a raster can match its ratio to the
        // nearest pixel. Requiring multiples of 70x99 unnecessarily padded
        // the page by up to 69x98 pixels, especially visible at low resolution.
        // Only enlarge a white canvas: notation samples remain unchanged.
        let required_width = (u64::from(crop[2]) * u64::from(paper[0]))
            .div_ceil(u64::from(paper[0] - 2 * MARGIN_MM));
        let required_by_height = (u64::from(crop[3]) * u64::from(paper[0]))
            .div_ceil(u64::from(paper[1] - 2 * MARGIN_MM));
        let mut width = required_width.max(required_by_height).max(1) as u32;
        let (size, margin) = loop {
            let height = ((u64::from(width) * u64::from(paper[1]) + u64::from(paper[0]) / 2)
                / u64::from(paper[0]))
            .max(1) as u32;
            let margin = (u64::from(width) * u64::from(MARGIN_MM))
                .div_ceil(u64::from(paper[0]))
                .max((u64::from(height) * u64::from(MARGIN_MM)).div_ceil(u64::from(paper[1])))
                as u32;
            if width >= crop[2] + 2 * margin && height >= crop[3] + 2 * margin {
                break ([width, height], margin);
            }
            width += 1;
        };
        let offset = [(size[0] - crop[2]) / 2, (size[1] - crop[3]) / 2];
        let margins_mm = [
            f64::from(offset[0]) * f64::from(paper[0]) / f64::from(size[0]),
            f64::from(size[0] - offset[0] - crop[2]) * f64::from(paper[0]) / f64::from(size[0]),
            f64::from(offset[1]) * f64::from(paper[1]) / f64::from(size[1]),
            f64::from(size[1] - offset[1] - crop[3]) * f64::from(paper[1]) / f64::from(size[1]),
        ];
        Some(Self {
            paper_size_mm: paper,
            source_crop: crop,
            output_size: size,
            content_offset: offset,
            minimum_margin_pixels: margin,
            margins_mm,
            estimated_page_height_width_ratio: ratio,
        })
    }
    pub fn translation(&self) -> Point {
        Point {
            x: f64::from(self.content_offset[0]) - f64::from(self.source_crop[0]),
            y: f64::from(self.content_offset[1]) - f64::from(self.source_crop[1]),
        }
    }
    pub fn apply_gray(&self, source: &GrayImage) -> GrayImage {
        let mut out = GrayImage::from_pixel(self.output_size[0], self.output_size[1], Luma([255]));
        self.copy_rows(source.as_raw(), source.width(), out.as_mut(), 1);
        out
    }
    pub fn apply_rgb(&self, source: &RgbImage) -> RgbImage {
        let mut out = RgbImage::from_pixel(self.output_size[0], self.output_size[1], Rgb([255; 3]));
        self.copy_rows(source.as_raw(), source.width(), out.as_mut(), 3);
        out
    }
    fn copy_rows(&self, source: &[u8], source_width: u32, out: &mut [u8], channels: usize) {
        let [x, y, w, h] = self.source_crop;
        for row in 0..h {
            let src = ((y + row) as usize * source_width as usize + x as usize) * channels;
            let dst = ((self.content_offset[1] + row) as usize * self.output_size[0] as usize
                + self.content_offset[0] as usize)
                * channels;
            let len = w as usize * channels;
            out[dst..dst + len].copy_from_slice(&source[src..src + len]);
        }
    }
}

/// A bounded grid pools actual dark/gray source samples, rather than resizing
/// them away. Components select bounds only: retained pixels are never erased.
fn ink_bounds(image: &GrayImage, protected: &[[f64; 4]]) -> Option<[u32; 4]> {
    let (w, h) = image.dimensions();
    if w < 2 || h < 2 {
        return None;
    }
    let step = w.max(h).div_ceil(1600);
    let (gw, gh) = (w.div_ceil(step) as usize, h.div_ceil(step) as usize);
    let mut counts = vec![0_u32; gw * gh];
    let mut dark = vec![0_u32; gw * gh];
    for (y, row) in image.as_raw().chunks_exact(w as usize).enumerate() {
        let offset = y / step as usize * gw;
        for (x, &value) in row.iter().enumerate() {
            if value < 255 {
                let index = offset + x / step as usize;
                counts[index] += 1;
                dark[index] += u32::from(value < 220);
            }
        }
    }
    let mut bounds = [w, h, 0, 0];
    let mut found = false;
    let mut queue = Vec::new();
    for seed in 0..counts.len() {
        if counts[seed] == 0 {
            continue;
        }
        let mut mass = counts[seed];
        let mut core = dark[seed];
        counts[seed] = 0;
        queue.clear();
        queue.push(seed);
        let mut component = [gw, gh, 0, 0];
        let mut cursor = 0;
        while cursor < queue.len() {
            let i = queue[cursor];
            cursor += 1;
            let (x, y) = (i % gw, i / gw);
            component[0] = component[0].min(x);
            component[1] = component[1].min(y);
            component[2] = component[2].max(x);
            component[3] = component[3].max(y);
            for yy in y.saturating_sub(1)..=(y + 1).min(gh - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(gw - 1) {
                    let j = yy * gw + xx;
                    if counts[j] > 0 {
                        mass += counts[j];
                        core += dark[j];
                        counts[j] = 0;
                        queue.push(j);
                    }
                }
            }
        }
        if mass >= 6 {
            let rectangle = [
                component[0] as u32 * step,
                component[1] as u32 * step,
                ((component[2] + 1) as u32 * step).min(w),
                ((component[3] + 1) as u32 * step).min(h),
            ];
            // A relative dark fraction alone promotes tiny gray paper grains to
            // page content. Require scale-aware absolute stroke support too;
            // weak components still get the pixel-level faint-print check.
            let selected = if core >= 4 * step * step && core.saturating_mul(20) >= mass {
                Some(rectangle)
            } else {
                faint_ink_bounds(image, rectangle, step)
            };
            if let Some(rectangle) = selected {
                found = true;
                bounds[0] = bounds[0].min(rectangle[0]);
                bounds[1] = bounds[1].min(rectangle[1]);
                bounds[2] = bounds[2].max(rectangle[2]);
                bounds[3] = bounds[3].max(rectangle[3]);
            }
        }
    }
    if !found {
        return None;
    }
    for r in protected {
        if r.iter().all(|v| v.is_finite()) && r[2] >= r[0] && r[3] >= r[1] {
            bounds[0] = bounds[0].min(r[0].floor().clamp(0., f64::from(w)) as u32);
            bounds[1] = bounds[1].min(r[1].floor().clamp(0., f64::from(h)) as u32);
            bounds[2] = bounds[2].max((r[2].ceil() + 1.).clamp(0., f64::from(w)) as u32);
            bounds[3] = bounds[3].max((r[3].ceil() + 1.).clamp(0., f64::from(h)) as u32);
        }
    }
    let x = bounds[0].saturating_sub(2);
    let y = bounds[1].saturating_sub(2);
    Some([x, y, (bounds[2] + 2).min(w) - x, (bounds[3] + 2).min(h) - y])
}

/// Refine weak-only pooled components at original pixel resolution. Pooling can
/// join diffuse paper grains into a page-sized false foreground. A faint glyph
/// instead has coherent thin strokes. This selects the crop, never erases ink.
fn faint_ink_bounds(image: &GrayImage, region: [u32; 4], step: u32) -> Option<[u32; 4]> {
    let [left, top, right, bottom] = region;
    let (w, h) = ((right - left) as usize, (bottom - top) as usize);
    if w == 0 || h == 0 {
        return None;
    }
    let mut seen = vec![false; w * h];
    let mut queue = Vec::new();
    let radius = step.div_ceil(2).clamp(1, 2);
    let mut result = [right, bottom, left, top];
    let mut found = false;
    for y in top..bottom {
        for x in left..right {
            let index = (y - top) as usize * w + (x - left) as usize;
            if seen[index] || image.get_pixel(x, y)[0] == 255 {
                continue;
            }
            queue.clear();
            queue.push((x, y));
            seen[index] = true;
            let mut cursor = 0;
            let mut thick = false;
            let mut deep_ink = false;
            let mut bbox = [x, y, x + 1, y + 1];
            while cursor < queue.len() {
                let (x, y) = queue[cursor];
                cursor += 1;
                deep_ink |= image.get_pixel(x, y)[0] < 128;
                bbox[0] = bbox[0].min(x);
                bbox[1] = bbox[1].min(y);
                bbox[2] = bbox[2].max(x + 1);
                bbox[3] = bbox[3].max(y + 1);
                if !thick
                    && x > left + radius
                    && x + radius + 1 < right
                    && y > top + radius
                    && y + radius + 1 < bottom
                {
                    thick = (1..=radius + 1).all(|d| {
                        image.get_pixel(x - d, y)[0] < 255
                            && image.get_pixel(x + d, y)[0] < 255
                            && image.get_pixel(x, y - d)[0] < 255
                            && image.get_pixel(x, y + d)[0] < 255
                    });
                }
                for yy in y.saturating_sub(1).max(top)..=(y + 1).min(bottom - 1) {
                    for xx in x.saturating_sub(1).max(left)..=(x + 1).min(right - 1) {
                        let j = (yy - top) as usize * w + (xx - left) as usize;
                        if !seen[j] && image.get_pixel(xx, yy)[0] < 255 {
                            seen[j] = true;
                            queue.push((xx, yy));
                        }
                    }
                }
            }
            let span = (bbox[2] - bbox[0]).max(bbox[3] - bbox[1]);
            // Antialiased faint print still has a coherent darkest stroke core.
            // A paper fiber fades gradually across its whole footprint instead.
            let faint_stroke = !thick && span >= 6 * step && {
                let darkest = queue
                    .iter()
                    .map(|&(x, y)| image.get_pixel(x, y)[0])
                    .min()
                    .unwrap_or(255);
                let core = queue
                    .iter()
                    .filter(|&&(x, y)| image.get_pixel(x, y)[0] <= darkest.saturating_add(2))
                    .count();
                core * 5 >= queue.len()
            };
            if queue.len() >= 6 && (deep_ink || faint_stroke) {
                found = true;
                result[0] = result[0].min(bbox[0]);
                result[1] = result[1].min(bbox[1]);
                result[2] = result[2].max(bbox[2]);
                result[3] = result[3].max(bbox[3]);
            }
        }
    }
    found.then_some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diffuse_gray_paper_does_not_frame_a4_but_unrecognized_faint_glyphs_do() {
        let mut image = GrayImage::from_pixel(900, 1273, Luma([255]));
        for y in 300..1000 {
            for x in 200..700 {
                image.put_pixel(x, y, Luma([40]));
            }
        }
        for y in 400..850 {
            for x in 20..140 {
                image.put_pixel(x, y, Luma([248]));
            }
        }
        for x in 300..320 {
            image.put_pixel(x, 130, Luma([254]));
        }
        for y in 180..210 {
            for x in 600..620 {
                if !(602..618).contains(&x) || !(182..208).contains(&y) {
                    image.put_pixel(x, y, Luma([254]));
                }
            }
        }
        let frame = A4Frame::plan(&image, Some(297. / 210.), &[]).unwrap();
        assert!(
            frame.source_crop[0] >= 195,
            "Diffuse paper must not be treated as ink: {frame:?}"
        );
        assert!(
            frame.source_crop[1] <= 130,
            "Retain an unrecognized faint header"
        );
        assert!(
            frame.source_crop[0] + frame.source_crop[2] > 619,
            "Retain the faint hollow glyph"
        );
    }
    #[test]
    fn small_gray_paper_spots_do_not_enlarge_the_content_rectangle() {
        let mut image = GrayImage::from_pixel(1200, 1800, Luma([255]));
        for y in 250..1500 {
            for x in 200..1000 {
                image.put_pixel(x, y, Luma([30]));
            }
        }
        // A little dark center makes a paper grain pass a relative ink test,
        // although its absolute support is too small at this image scale.
        for (x, y) in [(40, 700), (1120, 1000), (600, 70)] {
            for yy in y..y + 9 {
                for xx in x..x + 9 {
                    image.put_pixel(xx, yy, Luma([248]));
                }
            }
            for yy in y + 3..y + 6 {
                for xx in x + 3..x + 6 {
                    image.put_pixel(xx, yy, Luma([180]));
                }
            }
        }
        // Unrecognized, genuinely faint print must still contribute.
        for x in 500..530 {
            image.put_pixel(x, 150, Luma([254]));
        }
        let frame = A4Frame::plan(&image, Some(297. / 210.), &[]).unwrap();
        assert!(
            frame.source_crop[0] >= 195,
            "Paper spots widened crop: {frame:?}"
        );
        assert!(frame.source_crop[0] + frame.source_crop[2] <= 1005);
        assert!(
            (145..=150).contains(&frame.source_crop[1]),
            "Keep faint print: {frame:?}"
        );
    }
    #[test]
    fn near_din_pages_get_pixel_rounded_a4_and_symmetric_margins_without_resampling() {
        for (ratio, paper) in [(1.39, [210, 297]), (0.72, [297, 210])] {
            let mut image = GrayImage::from_pixel(420, 594, Luma([255]));
            for y in 67..480 {
                for x in 39..370 {
                    image.put_pixel(x, y, Luma([((x + y) % 250) as u8]));
                }
            }
            let frame = A4Frame::plan(&image, Some(ratio), &[]).unwrap();
            assert_eq!(frame.paper_size_mm, paper);
            assert_eq!(
                frame.minimum_margin_pixels,
                (frame.output_size[0] * 10)
                    .div_ceil(paper[0])
                    .max((frame.output_size[1] * 10).div_ceil(paper[1])),
                "A4 output targets a 10 mm minimum margin"
            );
            assert!(
                (f64::from(frame.output_size[1])
                    - f64::from(frame.output_size[0]) * f64::from(paper[1]) / f64::from(paper[0]))
                .abs()
                    <= 0.5
            );
            let horizontal = (frame.output_size[0] - frame.source_crop[2]) / 2;
            let vertical = (frame.output_size[1] - frame.source_crop[3]) / 2;
            assert!(
                horizontal.min(vertical) <= frame.minimum_margin_pixels + 1,
                "No extra coarse-grid padding: {frame:?}"
            );
            assert!(frame.margins_mm.iter().all(|m| *m >= 10. - 1e-9));
            let result = frame.apply_gray(&image);
            let [cx, cy, cw, ch] = frame.source_crop;
            let [ox, oy] = frame.content_offset;
            assert!(ox >= frame.minimum_margin_pixels && oy >= frame.minimum_margin_pixels);
            assert!(result.width() - ox - cw >= frame.minimum_margin_pixels);
            assert!(result.height() - oy - ch >= frame.minimum_margin_pixels);
            assert!(ox.abs_diff(result.width() - ox - cw) <= 1);
            assert!(oy.abs_diff(result.height() - oy - ch) <= 1);
            for y in 0..ch {
                for x in 0..cw {
                    assert_eq!(
                        result.get_pixel(x + ox, y + oy),
                        image.get_pixel(x + cx, y + cy)
                    );
                }
            }
            assert_eq!(result.get_pixel(0, 0)[0], 255);
        }
    }
    #[test]
    fn faint_unrecognized_headers_footer_and_supported_tiny_marks_survive() {
        let mut image = GrayImage::from_pixel(2100, 2970, Luma([255]));
        for x in 300..1700 {
            image.put_pixel(x, 900, Luma([0]));
        }
        for x in 1800..1820 {
            image.put_pixel(x, 120, Luma([254]));
        }
        for y in 2750..2770 {
            image.put_pixel(1000, y, Luma([250]));
        }
        image.put_pixel(150, 700, Luma([252]));
        image.put_pixel(0, 0, Luma([20])); // Isolated sensor/dust pixel must not frame the page.
        let frame = A4Frame::plan(&image, Some(297. / 210.), &[[149., 699., 151., 701.]]).unwrap();
        let [x, y, w, h] = frame.source_crop;
        assert!(x > 0 && x <= 149 && y <= 120 && x + w > 1819 && y + h > 2769);
        let result = frame.apply_gray(&image);
        let t = frame.translation();
        assert_eq!(
            result.get_pixel((1805. + t.x) as u32, (120. + t.y) as u32)[0],
            254
        );
        assert_eq!(
            result.get_pixel((150. + t.x) as u32, (700. + t.y) as u32)[0],
            252
        );
    }
    #[test]
    fn unknown_letter_square_blank_and_invalid_ratios_do_not_trigger_a4() {
        let image = GrayImage::from_pixel(200, 280, Luma([0]));
        for ratio in [None, Some(11. / 8.5), Some(1.), Some(f64::NAN), Some(-1.)] {
            assert!(A4Frame::plan(&image, ratio, &[]).is_none());
        }
        assert!(
            A4Frame::plan(
                &GrayImage::from_pixel(200, 280, Luma([255])),
                Some(297. / 210.),
                &[]
            )
            .is_none()
        );
    }
}
