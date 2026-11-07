//! Final content and physical-page crop bounds.
use image::GrayImage;

/// Inclusive ink bounds plus padding, expressed as x/y/width/height.
/// Ignore isolated components smaller than six pixels when choosing the crop;
/// do not erase them from the output. A padding band retains tiny punctuation
/// adjacent to real content. Blank images retain their original canvas.
pub fn content_crop(binary: &GrayImage, padding: u32) -> [u32; 4] {
    let (w, h) = (binary.width() as usize, binary.height() as usize);
    let mut seen = vec![false; w * h];
    let mut queue = vec![];
    let (mut left, mut top, mut right, mut bottom) = (w, h, 0, 0);
    let mut found = false;
    for seed in 0..w * h {
        if seen[seed] || binary.as_raw()[seed] != 0 {
            continue;
        }
        queue.clear();
        queue.push(seed);
        seen[seed] = true;
        let mut cursor = 0;
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
        while cursor < queue.len() {
            let i = queue[cursor];
            cursor += 1;
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
            for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    let j = yy * w + xx;
                    if !seen[j] && binary.as_raw()[j] == 0 {
                        seen[j] = true;
                        queue.push(j);
                    }
                }
            }
        }
        if queue.len() >= 6 {
            found = true;
            left = left.min(x0);
            top = top.min(y0);
            right = right.max(x1);
            bottom = bottom.max(y1);
        }
    }
    if !found {
        return [0, 0, w as u32, h as u32];
    }
    let pad = padding as usize;
    let x = left.saturating_sub(pad);
    let y = top.saturating_sub(pad);
    let end_x = (right + 1).saturating_add(pad).min(w);
    let end_y = (bottom + 1).saturating_add(pad).min(h);
    [x as u32, y as u32, (end_x - x) as u32, (end_y - y) as u32]
}

/// Conservative page framing from the transformed paper boundary, independent
/// of the amount and position of printed content. Keep all boundary extrema.
pub fn page_crop(
    contour: &[crate::geometry::Point],
    width: u32,
    height: u32,
    padding: u32,
) -> [u32; 4] {
    if contour.len() < 3 || contour.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
        return [0, 0, width, height];
    }
    let pad = f64::from(padding);
    let x = (contour
        .iter()
        .map(|p| p.x)
        .fold(f64::INFINITY, f64::min)
        .floor()
        - pad)
        .max(0.)
        .min(f64::from(width)) as u32;
    let y = (contour
        .iter()
        .map(|p| p.y)
        .fold(f64::INFINITY, f64::min)
        .floor()
        - pad)
        .max(0.)
        .min(f64::from(height)) as u32;
    let ex = (contour
        .iter()
        .map(|p| p.x)
        .fold(f64::NEG_INFINITY, f64::max)
        .ceil()
        + 1.
        + pad)
        .max(0.)
        .min(f64::from(width)) as u32;
    let ey = (contour
        .iter()
        .map(|p| p.y)
        .fold(f64::NEG_INFINITY, f64::max)
        .ceil()
        + 1.
        + pad)
        .max(0.)
        .min(f64::from(height)) as u32;
    if ex <= x || ey <= y {
        return [0, 0, width, height];
    }
    [x, y, ex - x, ey - y]
}
