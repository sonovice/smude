use serde::Deserialize;
use smude_geometry::{Barline, Observations, Point, Staff, VerticalCorrection};

fn slope(points: &[Point]) -> f64 {
    let cy = points.iter().map(|p| p.y).sum::<f64>() / points.len() as f64;
    let cx = points.iter().map(|p| p.x).sum::<f64>() / points.len() as f64;
    points.iter().map(|p| (p.x - cx) * (p.y - cy)).sum::<f64>()
        / points.iter().map(|p| (p.y - cy).powi(2)).sum::<f64>()
}
fn points(bar: &Barline) -> Vec<Point> {
    bar.x
        .iter()
        .zip(&bar.y)
        .map(|(x, y)| Point { x: *x, y: *y })
        .collect()
}

#[test]
fn complete_system_with_different_upper_and_lower_directions_uses_a_common_height_field() {
    let staves = (0..3)
        .map(|s| Staff {
            x: vec![50., 950.],
            y: std::array::from_fn(|k| vec![110. + s as f64 * 110. + k as f64 * 10.; 2]),
        })
        .collect();
    let barlines = [
        (100., 110., 370., 0, vec![1, 2]),
        (500., 220., 370., 1, vec![2]),
        (900., 220., 370., 1, vec![2]),
        (500., 110., 150., 0, vec![]),
        (900., 110., 150., 0, vec![]),
    ]
    .into_iter()
    .map(|(x, lo, hi, staff, aligned_staffs)| {
        let y: Vec<_> = (0..41).map(|i| lo + (hi - lo) * i as f64 / 40.).collect();
        let x = y
            .iter()
            .map(|y| {
                let t = y - 240.;
                let a = 0.02 * t + 0.00003 * t * t;
                let b = 0.00001 * t + 0.0000002 * t * t;
                500. + (x - 500.) * (1. + b) + a
            })
            .collect();
        Barline {
            staff,
            aligned_staffs,
            x,
            y,
        }
    })
    .collect();
    let data = Observations {
        width: 1000,
        height: 500,
        staves,
        barlines,
    };
    let correction = VerticalCorrection::fit(&data, &|p| p, &[10.; 3], 1000.);
    assert_eq!(correction.bands.len(), 1);
    let band = &correction.bands[0];
    assert_eq!(band.model, "affine_across_width_and_height");
    assert!(band.after_rms_slope < band.before_rms_slope * 0.2);
    for b in &data.barlines {
        let after: Vec<_> = points(b)
            .into_iter()
            .map(|p| correction.forward(p))
            .collect();
        assert!(slope(&after).abs().atan().to_degrees() < 0.15);
    }
    for y in 0..500 {
        for x in [0., 500., 1000.] {
            let p = Point { x, y: y as f64 };
            let q = correction.forward(p);
            let back = correction.inverse(q);
            assert_eq!(q.y, p.y);
            assert!((back.x - p.x).abs() < 1e-10);
            assert!((0.98..1.02).contains(&correction.x_scale(p)));
        }
    }
}

#[derive(Deserialize)]
struct Frozen {
    observations: Observations,
    spacings: Vec<f64>,
}
#[test]
fn photographed_0011_middle_systems_are_not_rejected_as_one_shared_shear() {
    let frozen: Frozen = serde_json::from_str(include_str!("fixtures/vertical_0011.json")).unwrap();
    let data = &frozen.observations;
    let correction = VerticalCorrection::fit(data, &|p| p, &frozen.spacings, data.width as f64);
    assert_eq!(correction.bands.len(), 4);
    for i in [1, 2] {
        let band = &correction.bands[i];
        assert_eq!(band.model, "affine_across_width_and_height");
        assert!(band.after_rms_slope < band.before_rms_slope * 0.4);
        let initial = &data
            .barlines
            .iter()
            .find(|b| b.staff == band.staves[0] && !b.aligned_staffs.is_empty())
            .unwrap();
        let q: Vec<_> = points(initial)
            .into_iter()
            .map(|p| correction.forward(p))
            .collect();
        assert!(slope(&q).abs().atan().to_degrees() < 0.15);
    }
    let shift = Point {
        x: 5000.,
        y: -1000.,
    };
    let shifted = VerticalCorrection::fit(
        data,
        &|p| Point {
            x: p.x + shift.x,
            y: p.y + shift.y,
        },
        &frozen.spacings,
        data.width as f64,
    );
    for bar in &data.barlines {
        for p in points(bar) {
            let q = correction.forward(p);
            let shifted_q = shifted.forward(Point {
                x: p.x + shift.x,
                y: p.y + shift.y,
            });
            assert!((q.x + shift.x - shifted_q.x).abs() < 1e-8);
            assert_eq!(q.y + shift.y, shifted_q.y);
        }
    }
}

#[test]
fn independent_columns_at_same_height_do_not_get_a_shared_x_field() {
    let staves = [vec![20., 450.], vec![550., 980.]]
        .into_iter()
        .map(|x| Staff {
            x,
            y: std::array::from_fn(|k| vec![100. + 10. * k as f64; 2]),
        })
        .collect();
    let mut barlines = vec![];
    for staff in 0..2 {
        for x in [100., 300.] {
            let y: Vec<_> = (100..141).map(|y| y as f64).collect();
            barlines.push(Barline {
                staff,
                aligned_staffs: vec![],
                x: y.iter()
                    .map(|y| x + 500. * staff as f64 + 0.03 * (y - 120.))
                    .collect(),
                y,
            });
        }
    }
    let data = Observations {
        width: 1000,
        height: 500,
        staves,
        barlines,
    };
    let correction = VerticalCorrection::fit(&data, &|p| p, &[10.; 2], 1000.);
    assert_eq!(correction.bands.len(), 2);
    assert!(correction.bands.iter().all(|b| b.model == "identity"));
}

#[test]
fn incompatible_strokes_at_same_x_and_y_cannot_be_explained_by_height_variation() {
    let staves = (0..3)
        .map(|s| Staff {
            x: vec![50., 950.],
            y: std::array::from_fn(|k| vec![110. + s as f64 * 110. + k as f64 * 10.; 2]),
        })
        .collect();
    let barlines = [
        (100., 110., 370., 0.),
        (500., 110., 150., 0.04),
        (500., 110., 150., -0.04),
        (900., 220., 370., 0.04),
        (900., 220., 370., -0.04),
    ]
    .into_iter()
    .map(|(x, lo, hi, m)| {
        let y: Vec<_> = (0..41).map(|i| lo + (hi - lo) * i as f64 / 40.).collect();
        Barline {
            staff: 0,
            aligned_staffs: vec![1, 2],
            x: y.iter().map(|y| x + m * (y - (lo + hi) / 2.)).collect(),
            y,
        }
    })
    .collect();
    let data = Observations {
        width: 1000,
        height: 500,
        staves,
        barlines,
    };
    let correction = VerticalCorrection::fit(&data, &|p| p, &[10.; 3], 1000.);
    assert!(correction.bands.iter().all(|b| b.model == "identity"));
}

#[test]
fn three_complete_independent_bars_identify_height_variation() {
    let staves = (0..2)
        .map(|s| Staff {
            x: vec![50., 950.],
            y: std::array::from_fn(|k| vec![110. + s as f64 * 140. + k as f64 * 10.; 2]),
        })
        .collect();
    let barlines = [100., 500., 900.]
        .into_iter()
        .map(|x| {
            let y: Vec<_> = (0..41).map(|i| 110. + 180. * i as f64 / 40.).collect();
            let x = y
                .iter()
                .map(|y| {
                    let t = y - 200.;
                    let a = 0.02 * t + 0.00003 * t * t;
                    let b = 0.00001 * t + 0.0000002 * t * t;
                    500. + (x - 500.) * (1. + b) + a
                })
                .collect();
            Barline {
                staff: 0,
                aligned_staffs: vec![1],
                x,
                y,
            }
        })
        .collect();
    let data = Observations {
        width: 1000,
        height: 500,
        staves,
        barlines,
    };
    let correction = VerticalCorrection::fit(&data, &|p| p, &[10.; 2], 1000.);
    assert_eq!(correction.bands[0].model, "affine_across_width_and_height");
    for bar in &data.barlines {
        let q: Vec<_> = points(bar)
            .into_iter()
            .map(|p| correction.forward(p))
            .collect();
        assert!(slope(&q).abs() < 1e-7);
    }
}
