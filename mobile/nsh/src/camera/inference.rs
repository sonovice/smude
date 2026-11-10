use smude_geometry::detection::{PageDetector, PageLayoutMode};
use std::{
    borrow::Cow,
    ffi::{CString, c_char},
    sync::OnceLock,
};

#[cfg(test)]
pub fn detect(
    bytes: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    bgra: bool,
) -> serde_json::Value {
    detect_oriented(bytes, width, height, stride, bgra, 0)
}

/// Rotate only the model input to physical upright, then map its contours back
/// to the unchanged preview buffer. Works even when the UI rotation is locked.
pub fn detect_oriented(
    bytes: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    bgra: bool,
    clockwise_turns: u32,
) -> serde_json::Value {
    if width < 2
        || height < 2
        || width > 2048
        || height > 2048
        || stride < width * 4
        || stride.checked_mul(height).is_none_or(|n| bytes.len() < n)
    {
        return serde_json::json!({"error":"Invalid camera image"});
    }
    let mut rgb = image::RgbImage::new(width as u32, height as u32);
    for (y, row) in bytes.chunks(stride).take(height).enumerate() {
        for x in 0..width {
            let p = &row[x * 4..x * 4 + 4];
            rgb.put_pixel(
                x as u32,
                y as u32,
                image::Rgb(if bgra {
                    [p[2], p[1], p[0]]
                } else {
                    [p[0], p[1], p[2]]
                }),
            );
        }
    }
    let scene = scene_signature(&rgb);
    let oriented = rotated(&rgb, clockwise_turns);
    static DETECTOR: OnceLock<Result<PageDetector, String>> = OnceLock::new();
    let detector =
        match DETECTOR.get_or_init(|| PageDetector::embedded().map_err(|e| e.to_string())) {
            Ok(d) => d,
            Err(e) => return serde_json::json!({"error":e}),
        };
    match detector.detect(&oriented, PageLayoutMode::Auto) {
        Err(e) => serde_json::json!({"error":e.to_string()}),
        Ok((mut pages, mut report)) => {
            let mut actual_turns = clockwise_turns % 4;
            let mut elapsed = report.total_seconds;
            // A flat phone has no reliable cardinal gravity orientation. The
            // classifier can still identify a sideways spread, while the fixed
            // left/right splitter creates overlapping page predictions. Only
            // this explicit failure gets bounded orthogonal retries.
            if report.selected_class == Some("two_page_landscape")
                && (pages.regions.len() != 2 || !report.rejected_pages.is_empty())
            {
                let mut best_score = -1_f32;
                for turns in [(actual_turns + 1) % 4, (actual_turns + 3) % 4] {
                    if let Ok((candidate, alternative)) =
                        detector.detect(&rotated(&rgb, turns), PageLayoutMode::Auto)
                    {
                        elapsed += alternative.total_seconds;
                        let score = alternative.class_scores.map(|s| s[4]).unwrap_or(0.);
                        if alternative.selected_class == Some("two_page_portrait")
                            && score >= 0.5
                            && alternative.selected_layout == PageLayoutMode::Double
                            && candidate.regions.len() == 2
                            && alternative.raw_boundaries_source.len() == 2
                            && alternative.rejected_pages.is_empty()
                            && score > best_score
                        {
                            best_score = score;
                            pages = candidate;
                            report = alternative;
                            actual_turns = turns;
                        }
                    }
                }
            }
            let none = report.selected_class == Some("no_page");
            let expected = if none {
                0
            } else if report.selected_layout == PageLayoutMode::Double {
                2
            } else {
                1
            };
            // Show what the model found, including both halves of a spread.
            // The downstream completeness filter remains authoritative for the
            // shutter: displaying a contour is not permission to capture it.
            let polygons = if none {
                vec![]
            } else {
                let raw = report
                    .raw_boundaries_source
                    .iter()
                    .map(|polygon| {
                        polygon
                            .iter()
                            .map(|&p| preview_point(p, width, height, actual_turns))
                            .collect()
                    })
                    .collect::<Vec<_>>();
                display_polygons(&raw, width, height)
            };
            let valid = !none
                && pages.regions.len() == expected
                && polygons.len() == expected
                && report.rejected_pages.is_empty();
            let guides: Vec<_> = if valid {
                report
                    .raw_boundaries_source
                    .iter()
                    .filter_map(|p| smude_geometry::detection::quality::page_alignment(p))
                    .collect()
            } else {
                vec![]
            };
            let alignment = if guides.is_empty() {
                None
            } else {
                let v = [
                    guides.iter().map(|v| v[0]).sum::<f64>() / guides.len() as f64,
                    guides.iter().map(|v| v[1]).sum::<f64>() / guides.len() as f64,
                ];
                Some(match actual_turns {
                    1 => [v[1], -v[0]],
                    2 => [-v[0], -v[1]],
                    3 => [-v[1], v[0]],
                    _ => v,
                })
            };
            let message = if none {
                "page_in_frame"
            } else if !valid && expected == 2 {
                "include_spread"
            } else if !valid {
                "include_page"
            } else if expected == 2 {
                "spread_detected"
            } else {
                "page_detected"
            };
            #[cfg(any(target_os = "ios", target_os = "android"))]
            let preview_frame = if valid {
                super::snapshot::remember(
                    &rgb,
                    polygons
                        .iter()
                        .map(|p| p.iter().map(|p| [p[0] as f32, p[1] as f32]).collect())
                        .collect(),
                )
            } else {
                None
            };
            #[cfg(not(any(target_os = "ios", target_os = "android")))]
            let preview_frame: Option<u64> = None;
            serde_json::json!({
                "width":width,"height":height,"polygons":polygons,"preview_frame":preview_frame,
                "valid":valid,"pages":expected,"accepted_pages":pages.regions.len(),
                "selected_class":report.selected_class,"class_scores":report.class_scores,
                "rejected":report.rejected_pages,"message":message,"scene":scene,
                "analysis_turns":actual_turns,"alignment":alignment,
                "milliseconds":elapsed*1000.
            })
        }
    }
}

fn rotated(image: &image::RgbImage, turns: u32) -> Cow<'_, image::RgbImage> {
    match turns % 4 {
        1 => Cow::Owned(image::imageops::rotate90(image)),
        2 => Cow::Owned(image::imageops::rotate180(image)),
        3 => Cow::Owned(image::imageops::rotate270(image)),
        _ => Cow::Borrowed(image),
    }
}

fn preview_point(
    p: smude_geometry::geometry::Point,
    width: usize,
    height: usize,
    turns: u32,
) -> smude_geometry::geometry::Point {
    use smude_geometry::geometry::Point;
    let (w, h) = ((width - 1) as f64, (height - 1) as f64);
    match turns % 4 {
        1 => Point { x: p.y, y: h - p.x },
        2 => Point {
            x: w - p.x,
            y: h - p.y,
        },
        3 => Point { x: w - p.y, y: p.x },
        _ => p,
    }
}
/// An 8×8 spatial luminance fingerprint, in row-major order and normalized
/// to 0..1. Native latches subtract the mean brightness shift before comparing
/// frames, so auto-exposure alone does not look like a changed scene.
fn scene_signature(image: &image::RgbImage) -> Vec<f64> {
    let mut signature = Vec::with_capacity(64);
    let (width, height) = image.dimensions();
    for by in 0..8 {
        let y0 = by * height / 8;
        let y1 = ((by + 1) * height / 8).max(y0 + 1).min(height);
        for bx in 0..8 {
            let x0 = bx * width / 8;
            let x1 = ((bx + 1) * width / 8).max(x0 + 1).min(width);
            let mut luminance = 0_u64;
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = image.get_pixel(x, y).0;
                    luminance +=
                        77 * u64::from(p[0]) + 150 * u64::from(p[1]) + 29 * u64::from(p[2]);
                }
            }
            let count = u64::from(x1 - x0) * u64::from(y1 - y0);
            signature.push(luminance as f64 / (count as f64 * 256. * 255.));
        }
    }
    signature
}

fn display_polygons(
    raw: &[Vec<smude_geometry::geometry::Point>],
    width: usize,
    height: usize,
) -> Vec<Vec<[f64; 2]>> {
    raw.iter()
        .filter(|points| {
            points.len() >= 3 && points.iter().all(|p| p.x.is_finite() && p.y.is_finite())
        })
        .map(|points| {
            points
                .iter()
                .map(|p| {
                    [
                        (p.x / width as f64).clamp(0., 1.),
                        (p.y / height as f64).clamp(0., 1.),
                    ]
                })
                .collect()
        })
        .collect()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn smude_detect_bgra(
    data: *const u8,
    len: usize,
    width: usize,
    height: usize,
    stride: usize,
    clockwise_turns: u32,
) -> *mut c_char {
    let result = std::panic::catch_unwind(|| {
        if data.is_null() || len > 32 * 1024 * 1024 {
            return serde_json::json!({"error":"Invalid camera buffer"});
        }
        let bytes = unsafe { std::slice::from_raw_parts(data, len) };
        detect_oriented(bytes, width, height, stride, true, clockwise_turns)
    })
    .unwrap_or_else(|_| serde_json::json!({"error":"Page detection failed"}));
    CString::new(result.to_string()).unwrap().into_raw()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smude_detection_free(p: *mut c_char) {
    if !p.is_null() {
        drop(unsafe { CString::from_raw(p) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotated_model_points_return_to_exact_preview_pixels_for_both_pages() {
        use smude_geometry::geometry::Point;
        let (w, h) = (480, 640);
        for original in [
            Point { x: 12., y: 31. },
            Point { x: 210., y: 600. },
            Point { x: 255., y: 27. },
            Point { x: 470., y: 599. },
        ] {
            let rotated = [
                original,
                Point {
                    x: (h - 1) as f64 - original.y,
                    y: original.x,
                },
                Point {
                    x: (w - 1) as f64 - original.x,
                    y: (h - 1) as f64 - original.y,
                },
                Point {
                    x: original.y,
                    y: (w - 1) as f64 - original.x,
                },
            ];
            for (turns, p) in rotated.into_iter().enumerate() {
                let restored = preview_point(p, w, h, turns as u32);
                assert_eq!(restored.x, original.x);
                assert_eq!(restored.y, original.y);
            }
        }
    }
    #[test]
    fn scene_signature_preserves_spatial_changes_and_uniform_exposure() {
        let dark = image::RgbImage::from_pixel(16, 16, image::Rgb([32, 32, 32]));
        let light = image::RgbImage::from_pixel(16, 16, image::Rgb([96, 96, 96]));
        let before = scene_signature(&dark);
        let after = scene_signature(&light);
        assert_eq!(before.len(), 64);
        for (a, b) in before.iter().zip(&after) {
            assert!(((b - a) - 64. / 255.).abs() < 1e-10);
        }
        let mut spatial = dark;
        for y in 0..8 {
            for x in 0..16 {
                spatial.put_pixel(x, y, image::Rgb([255, 255, 255]));
            }
        }
        let changed = scene_signature(&spatial);
        assert!(changed[..32].iter().all(|x| (*x - 1.).abs() < 1e-10));
        assert!(
            changed[32..]
                .iter()
                .all(|x| (*x - 32. / 255.).abs() < 1e-10)
        );
    }
    #[test]
    fn raw_spread_contours_remain_distinct_and_bounded() {
        use smude_geometry::geometry::Point;
        let first = vec![
            Point { x: -5., y: 0. },
            Point { x: 50., y: 0. },
            Point { x: 50., y: 100. },
        ];
        let second = vec![
            Point { x: 50., y: 0. },
            Point { x: 105., y: 0. },
            Point { x: 100., y: 100. },
        ];
        let display = display_polygons(&[first, second], 100, 100);
        assert_eq!(display.len(), 2);
        assert_eq!(display[0][0], [0., 0.]);
        assert_eq!(display[1][1], [1., 0.]);
    }
}
