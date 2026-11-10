//! Cheap capture diagnostics from original pixels, independent of output tones.
//! Staff rules supply a known physical scale; blank margins are not blur samples.
use crate::geometry::observations::interpolate;
use crate::geometry::{Observations, Point};
use image::RgbImage;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CaptureQuality {
    pub bands: Vec<FocusBand>,
    pub retake_suggested: bool,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FocusBand {
    /// 0 = upper third, 1 = middle, 2 = lower, in reading orientation.
    pub band: usize,
    pub edge_spread_staff_spaces: f64,
    pub supports: usize,
    pub blurred: bool,
}
impl CaptureQuality {
    pub fn message(&self) -> Option<String> {
        let weak: Vec<_> = self
            .bands
            .iter()
            .filter(|b| b.blurred)
            .map(|b| b.band)
            .collect();
        if weak.is_empty() {
            return None;
        }
        Some(
            match weak.as_slice() {
                [0] => "Oben möglicherweise unscharf",
                [1] => "Mitte möglicherweise unscharf",
                [2] => "Unten möglicherweise unscharf",
                _ => "Teile der Seite möglicherweise unscharf",
            }
            .into(),
        )
    }
}
fn gray(image: &RgbImage, x: f64, y: f64) -> Option<f64> {
    if x < 0. || y < 0. || x >= f64::from(image.width() - 1) || y >= f64::from(image.height() - 1) {
        return None;
    }
    let (ix, iy) = (x as u32, y as u32);
    let (tx, ty) = (x - f64::from(ix), y - f64::from(iy));
    let at = |x, y| {
        let p = image.get_pixel(x, y).0;
        (2126. * f64::from(p[0]) + 7152. * f64::from(p[1]) + 722. * f64::from(p[2])) / 10000.
    };
    Some(
        (at(ix, iy) * (1. - tx) + at(ix + 1, iy) * tx) * (1. - ty)
            + (at(ix, iy + 1) * (1. - tx) + at(ix + 1, iy + 1) * tx) * ty,
    )
}
fn quantile(values: &mut [f64], fraction: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * fraction).round() as usize]
}

pub fn assess_capture(image: &RgbImage, data: &Observations) -> CaptureQuality {
    if image.dimensions() != (data.width, data.height) || image.width() < 2 || image.height() < 2 {
        return CaptureQuality::default();
    }
    let mut bands: [Vec<(f64, usize)>; 3] = std::array::from_fn(|_| vec![]);
    for staff in &data.staves {
        if staff.x.len() < 3 || staff.y.iter().any(|r| r.len() != staff.x.len()) {
            continue;
        }
        let center = interpolate(
            &staff.x,
            &staff.y[2],
            (staff.x[0] + staff.x[staff.x.len() - 1]) * 0.5,
        );
        let band = ((center / f64::from(data.height) * 3.) as usize).min(2);
        let mut scores = vec![];
        for j in 0..70 {
            let x = staff.x[1] + (staff.x[staff.x.len() - 2] - staff.x[1]) * j as f64 / 69.;
            let p: [f64; 5] = std::array::from_fn(|k| interpolate(&staff.x, &staff.y[k], x));
            let space = (p[4] - p[0]) / 4.;
            if !space.is_finite() || space < 4. {
                continue;
            }
            for k in [0, 2, 4] {
                let slope = (interpolate(&staff.x, &staff.y[k], x + 1.)
                    - interpolate(&staff.x, &staff.y[k], x - 1.))
                    / 2.;
                let norm = (1. + slope * slope).sqrt();
                let gap = space / norm;
                let step = gap * 0.84 / 40.;
                let samples: Option<Vec<_>> = (0..41)
                    .map(|i| {
                        let d = (i as f64 - 20.) * step;
                        gray(image, x - d * slope / norm, p[k] + d / norm)
                    })
                    .collect();
                let Some(samples) = samples else {
                    continue;
                };
                let mut left = samples[..5].to_vec();
                let mut right = samples[36..].to_vec();
                let paper = quantile(&mut left, 0.5).min(quantile(&mut right, 0.5));
                let ink = samples[12..29]
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min);
                let contrast = paper - ink;
                if contrast < 25. {
                    continue;
                }
                let gradient = samples
                    .windows(2)
                    .map(|p| (p[1] - p[0]).abs() / step)
                    .fold(0_f64, f64::max);
                if gradient > 0. {
                    scores.push(contrast / (gradient * gap));
                }
            }
        }
        if scores.len() >= 40 {
            let n = scores.len();
            bands[band].push((quantile(&mut scores, 0.35), n));
        }
    }
    let mut result = CaptureQuality::default();
    for (band, values) in bands.iter().enumerate() {
        if values.is_empty() {
            continue;
        }
        let mut scores: Vec<_> = values.iter().map(|v| v.0).collect();
        result.bands.push(FocusBand {
            band,
            edge_spread_staff_spaces: quantile(&mut scores, 0.75),
            supports: values.iter().map(|v| v.1).sum(),
            blurred: false,
        });
    }
    let best = result
        .bands
        .iter()
        .map(|b| b.edge_spread_staff_spaces)
        .fold(f64::INFINITY, f64::min);
    // Heuristic warning, never a capture veto. Detect local depth-of-field loss
    // without letting a sharp lower page hide a soft upper page.
    for band in &mut result.bands {
        let spread = band.edge_spread_staff_spaces;
        band.blurred = spread > 0.10 && (spread > best * 1.35 || best > 0.10);
    }
    result.retake_suggested = result.bands.iter().any(|b| b.blurred);
    result
}

/// Opposing edge scales give a visual frontality guide, not a calibrated angle
/// or an estimate of gravity. Curved book pages can also produce an imbalance.
pub fn page_alignment(polygon: &[Point]) -> Option<[f64; 2]> {
    if polygon.len() != 96 || polygon.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
        return None;
    }
    let length =
        |a: usize, b: usize| (polygon[a].x - polygon[b].x).hypot(polygon[a].y - polygon[b].y);
    let (top, bottom, left, right) = (length(0, 24), length(72, 48), length(0, 72), length(24, 48));
    if [top, bottom, left, right].iter().any(|v| *v < 1.) {
        return None;
    }
    Some([
        (left - right) / (left + right),
        (top - bottom) / (top + bottom),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Staff;
    use image::Rgb;
    #[test]
    fn local_defocus_is_detected_without_penalizing_blank_margins() {
        let staves: Vec<_> = [160., 380., 600., 820., 1040., 1260.]
            .iter()
            .map(|&c| Staff {
                x: vec![60., 200., 400., 600., 740.],
                y: std::array::from_fn(|k| vec![c + (k as f64 - 2.) * 20.; 5]),
            })
            .collect();
        let build = |blur_top: bool| {
            RgbImage::from_fn(800, 1440, |x, y| {
                let mut ink = 0_f64;
                if (60..=740).contains(&x) {
                    for s in &staves {
                        for k in 0..5 {
                            let d = (f64::from(y) - s.y[k][0]).abs();
                            let sigma: f64 = if blur_top && y < 480 { 2.6 } else { 0.65 };
                            ink = ink.max(170. * (-0.5 * (d / sigma).powi(2)).exp());
                        }
                    }
                }
                Rgb([(240. - ink).round() as u8; 3])
            })
        };
        let data = Observations {
            width: 800,
            height: 1440,
            staves: staves.clone(),
            barlines: vec![],
        };
        assert!(!assess_capture(&build(false), &data).retake_suggested);
        let bad = assess_capture(&build(true), &data);
        assert_eq!(
            bad.message().as_deref(),
            Some("Oben möglicherweise unscharf")
        );
        assert!(bad.bands.iter().filter(|b| b.blurred).all(|b| b.band == 0));
        assert_eq!(
            assess_capture(&RgbImage::from_pixel(800, 1440, Rgb([240; 3])), &data),
            CaptureQuality::default()
        );
    }
    #[test]
    fn frontality_uses_page_edges_and_rejects_degenerate_input() {
        let contour = |top: f64| {
            (0..96)
                .map(|i| {
                    let (side, t) = (i / 24, (i % 24) as f64 / 24.);
                    let c = [
                        Point { x: top, y: 0. },
                        Point {
                            x: 300. - top,
                            y: 0.,
                        },
                        Point { x: 300., y: 400. },
                        Point { x: 0., y: 400. },
                    ];
                    let (a, b) = (c[side], c[(side + 1) % 4]);
                    Point {
                        x: a.x + (b.x - a.x) * t,
                        y: a.y + (b.y - a.y) * t,
                    }
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(page_alignment(&contour(0.)), Some([0., 0.]));
        assert!(page_alignment(&contour(60.)).unwrap()[1] < -0.2);
        assert!(page_alignment(&vec![Point { x: 0., y: 0. }; 96]).is_none());
    }
}
