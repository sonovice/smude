//! Independent analytic GCS checks. These do not generate observations through
//! the implementation under test, nor through the earlier Python adapter.
use smude_geometry::{Barline, GcsMap, GcsParameters, Observations, Point, Staff};

fn camera(width: u32, height: u32) -> GcsParameters {
    GcsParameters {
        a: [0., 0.28, 0., 0.],
        rvec: [0.045, -0.14, 0.012],
        f: 1500.,
        cx: (width - 1) as f64 / 2.,
        cy: (height - 1) as f64 / 2.,
        scale: height as f64,
    }
}

// Closed-form arc length for z=c*x², unlike the implementation's monotone
// composite integration table. Rodrigues below uses quaternion rotation.
fn arc(x: f64, c: f64) -> f64 {
    if c == 0. {
        x
    } else {
        0.5 * (x * (1. + (2. * c * x).powi(2)).sqrt() + (2. * c * x).asinh() / (2. * c))
    }
}
fn rotate(p: [f64; 3], r: [f64; 3]) -> [f64; 3] {
    let theta = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    if theta < 1e-15 {
        return p;
    }
    let q: [f64; 3] = std::array::from_fn(|i| r[i] / theta * (theta / 2.).sin());
    let w = (theta / 2.).cos();
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let t = cross(q, p);
    let u = cross(q, t);
    std::array::from_fn(|i| p[i] + 2. * w * t[i] + 2. * u[i])
}
fn oracle(surface: Point, params: &GcsParameters) -> (Point, Point) {
    let c = params.a[1] / params.scale;
    let p = rotate(
        [surface.x, surface.y, c * surface.x * surface.x],
        params.rvec,
    );
    (
        Point {
            x: params.cx + params.f * p[0] / (params.f + p[2]),
            y: params.cy + params.f * p[1] / (params.f + p[2]),
        },
        Point {
            x: params.cx + arc(surface.x, c),
            y: params.cy + surface.y,
        },
    )
}

#[test]
fn independent_quadratic_projection_agrees_with_both_map_directions() {
    let params = camera(1000, 1400);
    let map = GcsMap::from_parameters(params.clone(), 1000, 1400).unwrap();
    for x in [-350., -200., 0., 180., 330.] {
        for y in [-500., -200., 0., 230., 460.] {
            let (image, flat) = oracle(Point { x, y }, &params);
            let out = map.forward(image);
            let back = map.inverse(flat);
            assert!(
                (out.x - flat.x).abs() < 0.004,
                "arc mismatch {}",
                out.x - flat.x
            );
            assert!((out.y - flat.y).abs() < 1e-7);
            assert!((back.x - image.x).hypot(back.y - image.y) < 0.004);
            assert!(map.jacobian_det(image) > 0.);
        }
    }
}

#[test]
fn entire_raster_roundtrips_at_boundaries_and_between_features() {
    let map = GcsMap::from_parameters(camera(1000, 1400), 1000, 1400).unwrap();
    for iy in 0..31 {
        for ix in 0..29 {
            let p = Point {
                x: ix as f64 / 28. * 999.,
                y: iy as f64 / 30. * 1399.,
            };
            let q = map.inverse(map.forward(p));
            assert!((q.x - p.x).hypot(q.y - p.y) < 1e-6);
        }
    }
    assert!(map.report().max_ray_equation_derivative < 0.);
}

fn observations() -> Observations {
    let params = camera(1000, 1400);
    observations_for(&params)
}

fn observations_for(params: &GcsParameters) -> Observations {
    let mut staves = vec![];
    let mut barlines = vec![];
    for i in 0..5 {
        let y0 = -450. + i as f64 * 190.;
        let projected: [Vec<_>; 5] = std::array::from_fn(|k| {
            (0..81)
                .map(|j| {
                    oracle(
                        Point {
                            x: -340. + j as f64 / 80. * 680.,
                            y: y0 + k as f64 * 11.,
                        },
                        params,
                    )
                    .0
                })
                .collect()
        });
        let lo = projected
            .iter()
            .map(|p| p[0].x)
            .fold(f64::NEG_INFINITY, f64::max);
        let hi = projected
            .iter()
            .map(|p| p.last().unwrap().x)
            .fold(f64::INFINITY, f64::min);
        let x: Vec<_> = (0..41).map(|j| lo + (hi - lo) * j as f64 / 40.).collect();
        let y = std::array::from_fn(|k| {
            x.iter()
                .map(|x| {
                    let p = &projected[k];
                    let j = p
                        .partition_point(|p| p.x <= *x)
                        .saturating_sub(1)
                        .min(p.len() - 2);
                    p[j].y + (p[j + 1].y - p[j].y) * (*x - p[j].x) / (p[j + 1].x - p[j].x)
                })
                .collect()
        });
        staves.push(Staff { x, y });
        for x0 in [-330., -110., 110., 330.] {
            let p: Vec<_> = (0..17)
                .map(|j| {
                    oracle(
                        Point {
                            x: x0,
                            y: y0 + j as f64 / 16. * 44.,
                        },
                        params,
                    )
                    .0
                })
                .collect();
            barlines.push(Barline {
                staff: i,
                aligned_staffs: vec![],
                x: p.iter().map(|p| p.x).collect(),
                y: p.iter().map(|p| p.y).collect(),
            });
        }
    }
    Observations {
        width: (2. * params.cx + 1.).round() as u32,
        height: (2. * params.cy + 1.).round() as u32,
        staves,
        barlines,
    }
}

fn horizontal_error(data: &Observations, map: Option<&GcsMap>) -> f64 {
    let mut e = 0.;
    let mut n = 0;
    for staff in &data.staves {
        for line in &staff.y {
            let y: Vec<_> = staff
                .x
                .iter()
                .zip(line)
                .map(|(x, y)| map.map_or(*y, |m| m.forward(Point { x: *x, y: *y }).y))
                .collect();
            let mean = y.iter().sum::<f64>() / y.len() as f64;
            e += y.iter().map(|y| (y - mean).powi(2)).sum::<f64>();
            n += y.len();
        }
    }
    (e / n as f64).sqrt()
}

#[test]
fn native_fit_reduces_curvature_and_preserves_every_staff() {
    let data = observations();
    let before = horizontal_error(&data, None);
    let fitted = GcsMap::fit(&data, &[], None).unwrap();
    let after = horizontal_error(&data, Some(&fitted));
    eprintln!(
        "GCS before={before} after={after} cost {} -> {} params {:?}",
        fitted.report().initial_cost,
        fitted.report().final_cost,
        fitted.report().parameters
    );
    assert!(
        after < before * 0.55,
        "native fitting did not improve curvature enough: {before} -> {after}"
    );
    assert!(fitted.report().final_cost < fitted.report().initial_cost * 0.65);
    assert_eq!(fitted.report().staff_groups, 5);
    assert_eq!(fitted.report().full_bar_paths, 20);
    assert!(fitted.report().maximum_roundtrip_pixels < 1e-5);
}

#[test]
fn physical_camera_roll_above_fifteen_degrees_levels_staves_and_bars() {
    let mut params = camera(1000, 1400);
    params.rvec[2] = -0.34;
    let data = observations_for(&params);
    let before = horizontal_error(&data, None);
    let model = GcsMap::fit(&data, &[], None).unwrap();
    let after = horizontal_error(&data, Some(&model));
    assert!(model.report().roll_initialization_radians.abs() > 0.26);
    assert!(model.report().parameters.rvec[2].abs() > 0.26);
    assert!(
        after < 0.08 && after < before * 0.02,
        "physical roll was not removed: {before}->{after}"
    );
    for bar in &data.barlines {
        let points: Vec<_> = bar
            .x
            .iter()
            .zip(&bar.y)
            .map(|(x, y)| model.forward(Point { x: *x, y: *y }))
            .collect();
        let mean = points.iter().map(|p| p.x).sum::<f64>() / points.len() as f64;
        let rms =
            (points.iter().map(|p| (p.x - mean).powi(2)).sum::<f64>() / points.len() as f64).sqrt();
        assert!(
            rms < 0.04,
            "full bar not vertical after roll correction: {rms}"
        );
    }
    assert!(model.report().minimum_sampled_jacobian > 0.);
    assert!(model.report().maximum_roundtrip_pixels < 1e-6);
}

#[test]
fn metric_paper_pose_seeds_strong_combined_tilt_and_camera_roll() {
    let mut params = camera(1200, 1600);
    params.f = 1800.;
    params.rvec = [-0.40, -0.18, -0.32];
    params.a = [0., 0.05, 0., 0.];
    let data = observations_for(&params);
    let paper = [
        Point { x: -350., y: -520. },
        Point { x: 350., y: -520. },
        Point { x: 350., y: 520. },
        Point { x: -350., y: 520. },
    ]
    .map(|p| oracle(p, &params).0);
    let ratio = 1040. / (2. * arc(350., params.a[1] / params.scale));
    let map = GcsMap::fit_with_camera(
        &data,
        &[],
        Some(ratio),
        Some(paper),
        Some(Point {
            x: params.cx,
            y: params.cy,
        }),
    )
    .unwrap();
    assert!(map.report().paper_pose_initialization.is_some());
    assert!(horizontal_error(&data, Some(&map)) < 0.10);
    assert!(map.report().parameters.rvec[2].abs() > 0.26);
    for bar in &data.barlines {
        let p: Vec<_> = bar
            .x
            .iter()
            .zip(&bar.y)
            .map(|(x, y)| map.forward(Point { x: *x, y: *y }))
            .collect();
        let mean = p.iter().map(|p| p.x).sum::<f64>() / p.len() as f64;
        let rms = (p.iter().map(|p| (p.x - mean).powi(2)).sum::<f64>() / p.len() as f64).sqrt();
        let first = p.first().unwrap();
        let last = p.last().unwrap();
        let angle = ((last.x - first.x) / (last.y - first.y))
            .abs()
            .atan()
            .to_degrees();
        // Use the independent suite's one-degree verticality requirement;
        // parameter priors do not promise an exact camera-parameter recovery.
        assert!(
            angle < 1.0 && rms < 0.2,
            "strong-pose full bar angle={angle}°, x residual={rms}px"
        );
    }
}

#[test]
fn invalid_extreme_surface_is_rejected_before_ray_root_branch_switches() {
    let mut p = camera(1000, 1400);
    p.a = [0., 0., 0., 5.];
    p.rvec = [0.5, -0.8, 0.1];
    assert!(GcsMap::from_parameters(p, 1000, 1400).is_err());
}

#[test]
fn flat_observations_remain_a_valid_fitted_model_not_an_exception() {
    let data = Observations {
        width: 1000,
        height: 1400,
        staves: vec![Staff {
            x: vec![50., 950.],
            y: std::array::from_fn(|k| vec![400. + k as f64 * 12.; 2]),
        }],
        barlines: vec![],
    };
    let map = GcsMap::fit(&data, &[], Some(1399. / 999.)).unwrap();
    assert!(map.report().final_cost < 1e-12);
    for p in [Point { x: 0., y: 0. }, Point { x: 999., y: 1399. }] {
        let q = map.forward(p);
        assert!((q.x - p.x).hypot(q.y - p.y) < 1e-9);
    }
}

#[test]
fn camera_space_paper_aspect_uses_actual_corners_not_padded_background() {
    let params = camera(1000, 1400);
    let data = observations();
    // The real paper has a different aspect from the surrounding raster.
    let corners = [
        Point { x: -350., y: -520. },
        Point { x: 350., y: -520. },
        Point { x: 350., y: 520. },
        Point { x: -350., y: 520. },
    ]
    .map(|p| oracle(p, &params).0);
    let ratio = 1040. / (2. * arc(350., params.a[1] / params.scale));
    assert!((ratio - 1399. / 999.).abs() > 0.05);
    let map = GcsMap::fit_with_camera(
        &data,
        &[],
        Some(ratio),
        Some(corners),
        Some(Point {
            x: params.cx,
            y: params.cy,
        }),
    )
    .unwrap();
    assert_eq!(
        map.report().aspect_constraint,
        "actual measured paper corners"
    );
    assert_eq!(
        map.report().camera_principal_point_source,
        "original sensor center adjusted for source crop"
    );
    // Calibrate only a global similarity on points disjoint from all probes.
    let mut calibration = vec![];
    for x in [-310., -130., 110., 310.] {
        for y in [-470., 470.] {
            let (image, material) = oracle(Point { x, y }, &params);
            calibration.push((map.forward(image), material));
        }
    }
    let qm = Point {
        x: calibration.iter().map(|p| p.0.x).sum::<f64>() / 8.,
        y: calibration.iter().map(|p| p.0.y).sum::<f64>() / 8.,
    };
    let pm = Point {
        x: calibration.iter().map(|p| p.1.x).sum::<f64>() / 8.,
        y: calibration.iter().map(|p| p.1.y).sum::<f64>() / 8.,
    };
    let denominator = calibration
        .iter()
        .map(|p| (p.0.x - qm.x).powi(2) + (p.0.y - qm.y).powi(2))
        .sum::<f64>();
    let a = calibration
        .iter()
        .map(|p| (p.0.x - qm.x) * (p.1.x - pm.x) + (p.0.y - qm.y) * (p.1.y - pm.y))
        .sum::<f64>()
        / denominator;
    let b = calibration
        .iter()
        .map(|p| (p.0.x - qm.x) * (p.1.y - pm.y) - (p.0.y - qm.y) * (p.1.x - pm.x))
        .sum::<f64>()
        / denominator;
    let align = |q: Point| Point {
        x: pm.x + a * (q.x - qm.x) - b * (q.y - qm.y),
        y: pm.y + b * (q.x - qm.x) + a * (q.y - qm.y),
    };
    let mut squared = 0.;
    let mut count = 0.;
    for x in [-270., -60., 170., 290.] {
        for y in [-420., -120., 160., 330.] {
            let (image, material) = oracle(Point { x, y }, &params);
            let q = align(map.forward(image));
            squared += (q.x - material.x).powi(2) + (q.y - material.y).powi(2);
            count += 1.;
            let u0 = arc(x, params.a[1] / params.scale);
            let glyph: Vec<_> = (0..33)
                .map(|j| {
                    let theta = j as f64 * std::f64::consts::TAU / 32.;
                    let u = u0 + 5.5 * theta.cos();
                    let v = y + 3.5 * theta.sin();
                    let mut sx = u;
                    for _ in 0..10 {
                        sx -= (arc(sx, params.a[1] / params.scale) - u)
                            / (1. + (2. * params.a[1] / params.scale * sx).powi(2)).sqrt();
                    }
                    align(map.forward(oracle(Point { x: sx, y: v }, &params).0))
                })
                .collect();
            let w = glyph.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max)
                - glyph.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
            let h = glyph.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max)
                - glyph.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
            assert!(
                ((w / h) / (11. / 7.) - 1.).abs() < 0.02,
                "glyph aspect changed despite actual paper prior"
            );
        }
    }
    assert!(
        (squared / count).sqrt() < 0.8,
        "independent material coordinates changed"
    );
}

#[test]
fn original_sensor_principal_point_may_be_outside_crop() {
    let data = Observations {
        width: 400,
        height: 600,
        staves: vec![Staff {
            x: vec![30., 370.],
            y: std::array::from_fn(|k| vec![200. + k as f64 * 10.; 2]),
        }],
        barlines: vec![],
    };
    let map =
        GcsMap::fit_with_camera(&data, &[], None, None, Some(Point { x: -150., y: 320. })).unwrap();
    assert_eq!(map.report().parameters.cx, -150.);
    for p in [Point { x: 0., y: 0. }, Point { x: 399., y: 599. }] {
        let q = map.forward(p);
        assert!((p.x - q.x).hypot(p.y - q.y) < 1e-8);
    }
}

#[test]
fn fractional_paper_edges_expand_certificate_without_clamping_camera_or_points() {
    let data = Observations {
        width: 400,
        height: 600,
        staves: vec![Staff {
            x: vec![30., 370.],
            y: std::array::from_fn(|k| vec![200. + k as f64 * 10.; 2]),
        }],
        barlines: vec![],
    };
    let corners = [
        Point { x: 0.25, y: 0.15 },
        Point { x: 399.8, y: 0.15 },
        Point { x: 399.8, y: 599.7 },
        Point { x: 0.25, y: 599.7 },
    ];
    let principal = Point { x: 123.4, y: 211.2 };
    let map = GcsMap::fit_with_camera(
        &data,
        &[],
        Some((599.7 - 0.15) / (399.8 - 0.25)),
        Some(corners),
        Some(principal),
    )
    .unwrap();
    assert_eq!(map.report().parameters.cx, principal.x);
    assert_eq!(map.report().parameters.cy, principal.y);
    for p in corners {
        let q = map.forward(p);
        let r = map.inverse(q);
        assert!((q.x - p.x).hypot(q.y - p.y) < 1e-8);
        assert!((r.x - p.x).hypot(r.y - p.y) < 1e-8);
        assert!(map.jacobian_det(p) > 0.);
    }
    let mut invalid = corners;
    invalid[1].x = 400.01;
    assert!(GcsMap::fit_with_paper(&data, &[], None, Some(invalid)).is_err());
}

#[test]
fn auxiliary_shafts_are_separate_from_bars_and_preserve_a_known_surface() {
    use smude_geometry::geometry::VerticalPath;
    let params = camera(1000, 1400);
    let mut data = observations_for(&params);
    data.barlines.clear();
    let paths: Vec<_> = (0..5)
        .flat_map(|i| {
            let params = &params;
            [-270., -90., 90., 270.]
                .into_iter()
                .map(move |x| VerticalPath {
                    points: (0..9)
                        .map(|j| {
                            oracle(
                                Point {
                                    x,
                                    y: -450. + i as f64 * 190. + j as f64 * 4.,
                                },
                                params,
                            )
                            .0
                        })
                        .collect(),
                    weight: 0.8,
                })
        })
        .collect();
    let baseline = GcsMap::fit(&data, &[], None).unwrap();
    let map = GcsMap::fit_with_verticals(&data, &[], None, None, None, &paths).unwrap();
    let held_out_error = |map: &GcsMap| {
        [-180., 0., 180.]
            .iter()
            .map(|x| {
                let a = map.forward(oracle(Point { x: *x, y: -390. }, &params).0);
                let b = map.forward(oracle(Point { x: *x, y: 390. }, &params).0);
                ((b.x - a.x) / (b.y - a.y)).atan().to_degrees().abs()
            })
            .sum::<f64>()
            / 3.
    };
    println!(
        "independent synthetic held-out vertical angle: baseline={}, stems={}",
        held_out_error(&baseline),
        held_out_error(&map)
    );
    assert_eq!(map.report().full_bar_paths, 0);
    assert_eq!(map.report().auxiliary_vertical_paths, 20);
    assert!(horizontal_error(&data, Some(&map)) < 0.2);
    // Independent, unseen shafts between the fitting shafts.
    for x in [-180., 0., 180.] {
        let p: Vec<_> = [-390., -200., 0., 190., 390.]
            .map(|y| map.forward(oracle(Point { x, y }, &params).0))
            .into();
        let lo = p.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let hi = p.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        assert!(hi - lo < 2., "held-out vertical deviates by {}", hi - lo);
    }
}

#[test]
fn invalid_auxiliary_verticals_are_rejected() {
    use smude_geometry::geometry::VerticalPath;
    let data = observations();
    for path in [
        VerticalPath {
            points: vec![Point { x: 10., y: 10. }; 3],
            weight: 0.8,
        },
        VerticalPath {
            points: vec![
                Point { x: 10., y: 10. },
                Point { x: 10., y: 20. },
                Point {
                    x: f64::NAN,
                    y: 30.,
                },
            ],
            weight: 0.8,
        },
        VerticalPath {
            points: vec![
                Point { x: 10., y: 10. },
                Point { x: 10., y: 20. },
                Point { x: 10., y: 30. },
            ],
            weight: 2.,
        },
    ] {
        assert!(GcsMap::fit_with_verticals(&data, &[], None, None, None, &[path]).is_err());
    }
}
