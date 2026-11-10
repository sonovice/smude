//! One bounded detection pass on the provisional rectification. All accepted
//! observations return to source coordinates before fitting the final geometry.
use crate::Result;
use crate::geometry::{Barline, Geometry, Observations, Point};
use image::RgbImage;

pub(super) fn discover(
    base: &Geometry,
    data: &Observations,
    source: &RgbImage,
) -> Result<(Observations, RgbImage, usize)> {
    let [x0, y0, x1, y1] = base.report().bounds;
    let scale = (1799. / (x1 - x0).max(y1 - y0)).min(1.);
    let (w, h) = (
        ((x1 - x0) * scale).ceil() as u32 + 1,
        ((y1 - y0) * scale).ceil() as u32 + 1,
    );
    let preview = base.render_at_dimensions(source, w, h)?;
    let to_pixel = |p: Point| {
        let q = base.forward(p);
        Point {
            x: (q.x - x0) * f64::from(w - 1) / (x1 - x0),
            y: (q.y - y0) * f64::from(h - 1) / (y1 - y0),
        }
    };
    let (mut mapped, _, shift) =
        crate::geometry::composition::map_observations(data, &[], to_pixel)?;
    // map_observations chooses a shifted canvas for general composition. Undo
    // that shift to match the explicit preview raster, preserving each line.
    mapped.width = w;
    mapped.height = h;
    mapped.barlines.clear();
    for staff in &mut mapped.staves {
        for x in &mut staff.x {
            *x += shift.x;
        }
        for line in &mut staff.y {
            for y in line {
                *y += shift.y;
            }
        }
    }
    let detected = crate::detection::music::detect_bars_on_staves(&preview, &mapped)?;
    let count = detected.len();
    let mut result = data.clone();
    for mut bar in detected {
        let points: Vec<_> = bar
            .x
            .iter()
            .zip(&bar.y)
            .map(|(x, y)| {
                base.inverse(Point {
                    x: x0 + x * (x1 - x0) / f64::from(w - 1),
                    y: y0 + y * (y1 - y0) / f64::from(h - 1),
                })
            })
            .collect();
        if points.iter().any(|p| {
            !p.x.is_finite()
                || !p.y.is_finite()
                || p.x < 0.
                || p.y < 0.
                || p.x > f64::from(data.width - 1)
                || p.y > f64::from(data.height - 1)
        }) || points.windows(2).any(|p| p[1].y <= p[0].y)
        {
            continue;
        }
        bar.x = points.iter().map(|p| p.x).collect();
        bar.y = points.iter().map(|p| p.y).collect();
        let staff = &data.staves[bar.staff];
        let space = (staff.y[4][0] - staff.y[0][0]) / 4.;
        if result
            .barlines
            .iter()
            .any(|old| same_stroke(old, &bar, space))
        {
            continue;
        }
        result.barlines.push(bar);
    }
    result.validate()?;
    Ok((result, preview, count))
}

fn same_stroke(a: &Barline, b: &Barline, space: f64) -> bool {
    let low = a.y[0].max(b.y[0]);
    let high = a.y[a.y.len() - 1].min(b.y[b.y.len() - 1]);
    if high - low < space {
        return false;
    }
    [low, (low + high) / 2., high].iter().all(|y| {
        (crate::geometry::observations::interpolate(&a.y, &a.x, *y)
            - crate::geometry::observations::interpolate(&b.y, &b.x, *y))
        .abs()
            < 0.5 * space
    })
}

/// Deviation from a constant x, normalized by each path's vertical span.
/// Captures both tilt and curvature. Fit evidence, not independent ground truth.
pub(super) fn errors(geometry: &Geometry, bars: &[Barline]) -> Vec<f64> {
    bars.iter()
        .map(|bar| {
            let p: Vec<_> = bar
                .x
                .iter()
                .zip(&bar.y)
                .map(|(x, y)| geometry.forward(Point { x: *x, y: *y }))
                .collect();
            let mean = p.iter().map(|p| p.x).sum::<f64>() / p.len() as f64;
            let span = (p.last().unwrap().y - p[0].y).abs().max(1.);
            (p.iter().map(|p| (p.x - mean).powi(2)).sum::<f64>() / p.len() as f64).sqrt() / span
        })
        .collect()
}

pub(super) fn improves(before: &[f64], after: &[f64], original_count: usize) -> bool {
    !before.is_empty()
        && before.len() == after.len()
        && after.iter().all(|x| x.is_finite())
        && after
            .iter()
            .zip(before)
            .take(original_count)
            .all(|(a, b)| *a <= b + 0.002)
        && after.iter().sum::<f64>() < before.iter().sum::<f64>() * 0.99
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bar(x: f64, y: [f64; 2]) -> Barline {
        Barline {
            staff: 0,
            x: vec![x; 2],
            y: y.to_vec(),
            aligned_staffs: vec![],
        }
    }
    #[test]
    fn overlap_deduplicates_segments_but_preserves_parallel_bars() {
        let full = bar(40., [10., 100.]);
        assert!(same_stroke(&full, &bar(40.4, [30., 60.]), 4.));
        assert!(!same_stroke(&full, &bar(44., [10., 100.]), 4.));
        assert!(!same_stroke(&full, &bar(40., [110., 150.]), 4.));
    }
    #[test]
    fn refinement_requires_improvement_and_protects_original_evidence() {
        assert!(improves(&[0.01, 0.05], &[0.01, 0.02], 1));
        assert!(!improves(&[0.01, 0.05], &[0.02, 0.01], 1));
        assert!(!improves(&[0.01, 0.05], &[0.01, 0.05], 1));
        assert!(!improves(&[0.01], &[f64::NAN], 0));
        assert!(!improves(&[], &[], 0));
    }
    #[test]
    fn second_pass_recovers_source_paths_without_changing_staff_y_warp() {
        use crate::geometry::{Config, Staff};
        let data = Observations {
            width: 240,
            height: 260,
            staves: [40., 110., 180.]
                .into_iter()
                .map(|y| Staff {
                    x: vec![20., 220.],
                    y: std::array::from_fn(|k| vec![y + k as f64 * 8.; 2]),
                })
                .collect(),
            barlines: vec![],
        };
        let mut source = RgbImage::from_pixel(240, 260, image::Rgb([255; 3]));
        for staff in &data.staves {
            for line in &staff.y {
                for x in 20..=220 {
                    source.put_pixel(x, line[0] as u32, image::Rgb([0; 3]));
                }
            }
            for y in staff.y[0][0] as u32..=staff.y[4][0] as u32 {
                for x in [80, 81, 160, 161] {
                    source.put_pixel(x, y, image::Rgb([0; 3]));
                }
            }
        }
        let config = Config {
            perspective: false,
            ..Config::default()
        };
        let (base, _) = Geometry::fit_profiled_direction_mesh(&data, &config, None, None, &[], &[]);
        let base = base.unwrap();
        let (merged, preview, _) = discover(&base, &data, &source).unwrap();
        assert_eq!(merged.barlines.len(), 6);
        assert!(preview.width().max(preview.height()) <= 1800);
        for bar in &merged.barlines {
            assert!(
                bar.x
                    .iter()
                    .all(|x| (x - 80.5).abs() < 1. || (x - 160.5).abs() < 1.)
            );
        }
        let mut refined = base.clone();
        refined.refine_direction_mesh(&merged, &[], &[]).unwrap();
        for y in [0., 60., 130., 240.] {
            for x in [0., 60., 170., 239.] {
                let p = Point { x, y };
                assert_eq!(base.forward(p).y, refined.forward(p).y);
                let back = refined.inverse(refined.forward(p));
                assert!((back.x - x).hypot(back.y - y) < 0.01);
            }
        }
        assert!(data.barlines.is_empty());
    }
}
