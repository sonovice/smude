//! Visible score excerpts when no complete physical page can be assigned.
//! Missing paper is neither reconstructed nor used as a metric/aspect prior.
use super::{Options, ProcessingResult, RenderedPage, StepProfile, finish_profile};
use crate::geometry::Observations;
use image::{Rgb, RgbImage};
use std::{fs, time::Instant};

fn supported(data: &Observations) -> bool {
    if data.staves.len() < 2 || data.barlines.len() < 2 {
        return false;
    }
    // Separate columns/spread halves must never share one excerpt warp.
    for (i, a) in data.staves.iter().enumerate() {
        for b in &data.staves[i + 1..] {
            let disjoint = a.x.last().unwrap() < &b.x[0] || b.x.last().unwrap() < &a.x[0];
            let range = |s: &crate::geometry::Staff| {
                (
                    s.y[0].iter().copied().fold(f64::INFINITY, f64::min),
                    s.y[4].iter().copied().fold(f64::NEG_INFINITY, f64::max),
                )
            };
            let (a0, a1) = range(a);
            let (b0, b1) = range(b);
            if disjoint && a1.min(b1) - a0.max(b0) > 0.4 * (a1 - a0).min(b1 - b0) {
                return false;
            }
        }
    }
    let long = data
        .staves
        .iter()
        .filter(|s| s.x.last().unwrap() - s.x[0] > f64::from(data.width) * 0.45)
        .count();
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for bar in &data.barlines {
        let x = bar.x.iter().sum::<f64>() / bar.x.len() as f64;
        lo = lo.min(x);
        hi = hi.max(x);
    }
    long >= 2 && hi - lo > f64::from(data.width) * 0.25
}

/// Remove only a narrow, sustained chromatic exterior next to neutral paper.
/// A full colored sheet, a small annotation or a gray shadow is insufficient.
fn clear_visible_exterior(image: &RgbImage, data: &Observations) -> (RgbImage, usize) {
    let (w, h) = (image.width(), image.height());
    let mut out = image.clone();
    let mut removed = 0;
    if w < 40 || h < 40 {
        return (out, removed);
    }
    let mut left = f64::INFINITY;
    let mut right = f64::NEG_INFINITY;
    for staff in &data.staves {
        let space = (staff.y[4][0] - staff.y[0][0]) / 4.;
        left = left.min(staff.x[0] - 3. * space);
        right = right.max(staff.x.last().unwrap() + 3. * space);
    }
    let chromatic = |p: Rgb<u8>| {
        let hi = *p.0.iter().max().unwrap();
        let lo = *p.0.iter().min().unwrap();
        hi - lo >= 24 && u16::from(lo) * 100 < u16::from(hi) * 70
    };
    let paper = |p: Rgb<u8>| {
        let hi = *p.0.iter().max().unwrap();
        let lo = *p.0.iter().min().unwrap();
        hi - lo < 20 && lo > 80
    };
    for from_right in [false, true] {
        let limit = w / 10;
        let x_at = |d| if from_right { w - 1 - d } else { d };
        let mut boundaries = vec![None; h as usize];
        for y in 0..h {
            if (0..3)
                .filter(|&d| chromatic(*image.get_pixel(x_at(d), y)))
                .count()
                < 2
            {
                continue;
            }
            let mut neutral = 0;
            for d in 3..limit {
                neutral = if paper(*image.get_pixel(x_at(d), y)) {
                    neutral + 1
                } else {
                    0
                };
                if neutral == 5 {
                    let edge = x_at(d - 4);
                    if (from_right && f64::from(edge) > right)
                        || (!from_right && f64::from(edge) < left)
                    {
                        boundaries[y as usize] = Some(edge);
                    }
                    break;
                }
            }
        }
        if boundaries.iter().filter(|b| b.is_some()).count() * 4 < (h as usize) * 3 {
            continue;
        }
        for y in 0..h {
            // Leave unsupported rows alone; never invent a hidden paper edge.
            let Some(edge) = boundaries[y as usize] else {
                continue;
            };
            let range = if from_right { edge + 1..w } else { 0..edge };
            for x in range {
                out.put_pixel(x, y, Rgb([255; 3]));
                removed += 1;
            }
        }
    }
    (out, removed)
}

pub(super) fn try_process(
    args: &Options,
    image: &RgbImage,
    run_started: Instant,
    decode_seconds: f64,
    page_detection_seconds: f64,
) -> Result<Option<ProcessingResult>, Box<dyn std::error::Error>> {
    let mut profile = StepProfile::start();
    profile.started = Some(run_started);
    profile.decode_seconds = decode_seconds;
    profile.page_detection_seconds = page_detection_seconds;
    args.processing(super::ProcessingEvent::PageStarted { index: 0, count: 1 });
    args.processing(super::ProcessingEvent::Stage(
        super::ProcessingStage::Analysis,
    ));
    let started = Instant::now();
    args.emit("1", super::DetectionStage::ResetMusic, Vec::new);
    let on_staves = |staves: &[crate::geometry::Staff]| args.emit_staves("1", staves, |p| p);
    let data = match crate::detection::music::detect_masked(
        image,
        args.analysis_width_for(image),
        None,
        Some(&on_staves),
    ) {
        Ok(data) => data,
        Err(crate::Error::Invalid("no reliable five-line staves found")) => {
            args.emit("1", super::DetectionStage::ResetMusic, Vec::new);
            return Ok(None);
        }
        Err(e) => return Err(e.into()),
    };
    profile.native_detection_seconds = started.elapsed().as_secs_f64();
    let accepted = supported(&data);
    fs::write(
        args.out.join("cropped-score-attempt.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "accepted":accepted,"staves":data.staves.len(),"barlines":data.barlines.len(),
            "reason":"complete physical page unavailable; require multiple long five-line staffs and separated crossing bars",
            "physical_page_boundary":null,"physical_page_ratio":null,
        }))?,
    )?;
    if !accepted {
        args.emit("1", super::DetectionStage::ResetMusic, Vec::new);
        return Ok(None);
    }
    let started = Instant::now();
    let (source, exterior_pixels) = clear_visible_exterior(image, &data);
    profile.visible_exterior_seconds = started.elapsed().as_secs_f64();
    let directory = args.out.join("page-1");
    fs::create_dir_all(&directory)?;
    let result = super::process(
        args,
        &directory,
        &data,
        Some(source),
        profile.native_detection_seconds,
        None,
        &mut profile,
    );
    finish_profile(&directory, &mut profile, &result)?;
    let summary = result?;
    fs::write(
        args.out.join("pages_report.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema":"smude.pages_run.v1","input_image":args.image,"page_method":"cropped-score",
            "coordinate_space":"original_image_pixels","status":"success","error":null,
            "decode_seconds":decode_seconds,"region_seconds":page_detection_seconds,
            "total_seconds":run_started.elapsed().as_secs_f64(),"visible_exterior_pixels":exterior_pixels,
            "pages":[{"id":"1","dir":directory,"source_bounds":[0,0,image.width(),image.height()],
                "assignment_status":"visible_score_excerpt","status":"success","measurements":summary}],
        }))?,
    )?;
    args.processing(super::ProcessingEvent::PageFinished { index: 0 });
    Ok(Some(ProcessingResult {
        pages: if args.no_render || args.detect_only {
            vec![]
        } else {
            vec![RenderedPage {
                id: "1".into(),
                directory,
                mapping: summary.mapping,
                quality: summary.quality,
            }]
        },
        warnings: vec![],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unsupported_content_and_parallel_columns_are_not_a_single_score_crop() {
        use crate::geometry::{Barline, Staff};
        let mut data = Observations {
            width: 1000,
            height: 600,
            staves: vec![],
            barlines: vec![],
        };
        assert!(!supported(&data));
        data.staves = vec![
            Staff {
                x: vec![10., 490.],
                y: std::array::from_fn(|k| vec![100. + k as f64 * 10.; 2]),
            },
            Staff {
                x: vec![510., 990.],
                y: std::array::from_fn(|k| vec![100. + k as f64 * 10.; 2]),
            },
        ];
        data.barlines = vec![
            Barline {
                staff: 0,
                x: vec![100., 100.],
                y: vec![100., 140.],
                aligned_staffs: vec![],
            },
            Barline {
                staff: 1,
                x: vec![800., 800.],
                y: vec![100., 140.],
                aligned_staffs: vec![],
            },
        ];
        assert!(
            !supported(&data),
            "separate columns were merged into one warp"
        );
    }
    #[test]
    fn a_colored_sheet_or_small_mark_is_not_an_exterior_strip() {
        let data = Observations {
            width: 100,
            height: 200,
            staves: vec![],
            barlines: vec![],
        };
        for image in [
            RgbImage::from_pixel(100, 200, Rgb([150, 90, 35])),
            RgbImage::from_fn(100, 200, |x, y| {
                if x > 95 && y < 20 {
                    Rgb([150, 90, 35])
                } else {
                    Rgb([235; 3])
                }
            }),
        ] {
            let (result, count) = clear_visible_exterior(&image, &data);
            assert_eq!(count, 0);
            assert_eq!(result, image);
        }
    }
}
