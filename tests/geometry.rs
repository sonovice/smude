use smude_geometry::{Barline, Config, Geometry, Observations, Point, Staff};

fn camera_inverse(p: Point) -> Point {
    let (u, v) = ((p.x - 499.5) / 1400., (p.y - 699.5) / 1400.);
    let (b, c, pp, q) = (-0.035, 0.012, 0.16, 0.24);
    let (a, bb, cc, d) = (1. - u * pp, b - u * q, c - v * pp, 1. - v * q);
    let det = a * d - bb * cc;
    Point {
        x: 499.5 + 1400. * (u * d - bb * v) / det,
        y: 699.5 + 1400. * (a * v - cc * u) / det,
    }
}

fn interp(points: &[Point], x: f64) -> f64 {
    let hi = points
        .partition_point(|p| p.x < x)
        .clamp(1, points.len() - 1);
    let a = points[hi - 1];
    let b = points[hi];
    a.y + (x - a.x) / (b.x - a.x) * (b.y - a.y)
}

fn fixture(perspective: bool, bent: bool) -> Observations {
    let transform = |p| if perspective { camera_inverse(p) } else { p };
    let mut staves = vec![];
    let mut barlines = vec![];
    for (s, center) in [260., 440., 840., 1020.].iter().enumerate() {
        let lines: [Vec<Point>; 5] = std::array::from_fn(|k| {
            (0..161)
                .map(|j| {
                    let x = 160. + 680. * j as f64 / 160.;
                    let bend = if bent {
                        8. * (x / 90. + s as f64).sin()
                    } else {
                        0.
                    };
                    transform(Point {
                        x,
                        y: center + (k as f64 - 2.) * 14. + bend,
                    })
                })
                .collect()
        });
        let lo = lines
            .iter()
            .map(|l| l[0].x)
            .fold(f64::NEG_INFINITY, f64::max);
        let hi = lines
            .iter()
            .map(|l| l[l.len() - 1].x)
            .fold(f64::INFINITY, f64::min);
        let x: Vec<f64> = (0..161).map(|j| lo + (hi - lo) * j as f64 / 160.).collect();
        let y = std::array::from_fn(|k| x.iter().map(|x| interp(&lines[k], *x)).collect());
        staves.push(Staff { x, y });
        for bx in [200., 350., 500., 650., 800.] {
            let offset = if bent {
                8. * (bx / 90. + s as f64).sin()
            } else {
                0.
            };
            let points: Vec<Point> = (0..81)
                .map(|j| {
                    transform(Point {
                        x: bx,
                        y: center - 28. + offset + 56. * j as f64 / 80.,
                    })
                })
                .collect();
            barlines.push(Barline {
                staff: s,
                x: points.iter().map(|p| p.x).collect(),
                y: points.iter().map(|p| p.y).collect(),
                aligned_staffs: vec![s ^ 1],
            });
        }
    }
    Observations {
        width: 1000,
        height: 1400,
        staves,
        barlines,
    }
}

fn assert_close(a: Point, b: Point, tolerance: f64) {
    assert!((a.x - b.x).hypot(a.y - b.y) < tolerance, "{a:?} vs {b:?}");
}

#[test]
fn known_perspective_improves_heldout_geometry_and_metric_grid() {
    let data = fixture(true, false);
    let config = Config {
        perspective: true,
        staff_size_groups: vec![vec![0, 1, 2, 3]],
        ..Config::default()
    };
    let geometry = Geometry::fit(&data, &config).unwrap();
    let p = &geometry.report().perspective;
    assert!(p.converged && !p.at_parameter_bound);
    assert!(
        p.after_heldout_spacing_rms < p.before_heldout_spacing_rms * 0.1,
        "{p:?}"
    );
    let mut expected = vec![];
    let mut actual = vec![];
    for i in 0..9 {
        for j in 0..13 {
            let p = Point {
                x: 200. + 600. * i as f64 / 8.,
                y: 240. + 800. * j as f64 / 12.,
            };
            let source = camera_inverse(p);
            expected.push(p);
            actual.push(geometry.forward(source));
            assert_close(geometry.inverse(geometry.forward(source)), source, 1e-7);
        }
    }
    // Align only by a global similarity, preserving the ability to catch
    // spurious local shear/aspect changes in the known synthetic geometry.
    let mean = |v: &[Point]| Point {
        x: v.iter().map(|p| p.x).sum::<f64>() / v.len() as f64,
        y: v.iter().map(|p| p.y).sum::<f64>() / v.len() as f64,
    };
    let a = mean(&actual);
    let e = mean(&expected);
    let mut dot = 0.;
    let mut cross = 0.;
    let mut norm = 0.;
    for (aa, ee) in actual.iter().zip(&expected) {
        let (ax, ay, ex, ey) = (aa.x - a.x, aa.y - a.y, ee.x - e.x, ee.y - e.y);
        dot += ax * ex + ay * ey;
        cross += ax * ey - ay * ex;
        norm += ax * ax + ay * ay;
    }
    let mut error = 0.;
    for (aa, ee) in actual.iter().zip(&expected) {
        let ax = aa.x - a.x;
        let ay = aa.y - a.y;
        error += (e.x + (dot * ax - cross * ay) / norm - ee.x).powi(2)
            + (e.y + (cross * ax + dot * ay) / norm - ee.y).powi(2);
    }
    let rms = (error / actual.len() as f64).sqrt();
    assert!(
        rms < 1.,
        "known metric grid RMS {rms}, parameters {:?}",
        p.parameters
    );
    assert!(geometry.report().validation.sampled_min_jacobian > 0.);
}

#[test]
fn independent_non_cylindrical_bends_and_small_staff_are_preserved() {
    let mut data = fixture(false, true);
    data.barlines.clear();
    let s = &mut data.staves[3];
    for j in 0..s.x.len() {
        let center = s.y[2][j];
        for k in 0..5 {
            s.y[k][j] = center + (s.y[k][j] - center) * 0.7;
        }
    }
    let config = Config {
        // Isolate staff-size preservation from the global projective fit.
        perspective: false,
        staff_size_groups: vec![vec![0, 1, 2]],
        ..Config::default()
    };
    let geometry = Geometry::fit(&data, &config).unwrap();
    let spaces = &geometry.report().target_staff_spacings;
    assert!((spaces[3] / spaces[0] - 0.7).abs() < 1e-9);
    assert!(geometry.report().validation.max_staff_line_span_pixels < 1e-8);
    for i in 0..100 {
        let p = Point {
            x: 50. + i as f64 * 9.,
            y: 100. + i as f64 * 11.,
        };
        assert_close(geometry.inverse(geometry.forward(p)), p, 1e-8);
    }
    let projected = Geometry::fit(&data, &Config::default()).unwrap();
    assert!(projected.report().config.perspective);
    // The projective stage staggers the knots of the five lines. Flattening
    // merges knots closer than its knot spacing, so these strongly bent
    // lines keep a chord error of a few hundredths of a pixel.
    assert!(projected.report().validation.max_staff_line_span_pixels < 0.05);
}

#[test]
fn shared_shear_preserves_y_width_and_straight_strokes() {
    let mut data = fixture(false, false);
    for b in &mut data.barlines {
        let center = (b.y[0] + b.y[b.y.len() - 1]) / 2.;
        for (x, y) in b.x.iter_mut().zip(&b.y) {
            *x += 0.08 * (y - center);
        }
    }
    let geometry = Geometry::fit(
        &data,
        &Config {
            // This test measures the x-only stage in isolation.
            perspective: false,
            system_shear: true,
            ..Config::default()
        },
    )
    .unwrap();
    for b in &data.barlines {
        let points: Vec<Point> =
            b.x.iter()
                .zip(&b.y)
                .map(|(x, y)| geometry.forward(Point { x: *x, y: *y }))
                .collect();
        let min = points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let max = points.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        assert!(max - min < 1e-8);
        for (p, y) in points.iter().zip(&b.y) {
            assert!((p.y - y).abs() < 1e-8);
        }
    }
    for y in [220., 300., 400., 500., 700., 900., 1100.] {
        let a = geometry.forward(Point { x: 200., y });
        let b = geometry.forward(Point { x: 700., y });
        assert!((b.x - a.x - 500.).abs() < 1e-8);
    }
}

#[test]
fn invalid_observations_and_size_groups_fail_before_fitting() {
    let data = fixture(false, false);
    for groups in [vec![vec![0, 1], vec![1, 2]], vec![vec![10]], vec![vec![]]] {
        assert!(
            Geometry::fit(
                &data,
                &Config {
                    staff_size_groups: groups,
                    ..Config::default()
                }
            )
            .is_err()
        );
    }
    let mut bad = data.clone();
    bad.staves[0].y.swap(1, 2);
    assert!(Geometry::fit(&bad, &Config::default()).is_err());
    let mut bad = data.clone();
    bad.barlines[0].y[2] = f64::NAN;
    assert!(Geometry::fit(&bad, &Config::default()).is_err());
}

#[test]
fn deduplicated_full_bars_group_all_crossed_staves() {
    let mut data = fixture(false, false);
    data.barlines.clear();
    for x in [250., 650.] {
        data.barlines.push(Barline {
            staff: 0,
            x: vec![x; 3],
            y: vec![232., 350., 468.],
            aligned_staffs: vec![1],
        });
    }
    let geometry = Geometry::fit(
        &data,
        &Config {
            perspective: false,
            ..Config::default()
        },
    )
    .unwrap();
    assert_eq!(geometry.report().systems[0].staves, vec![0, 1]);
    assert_eq!(geometry.report().systems.len(), 3);
}

#[test]
fn one_accidental_bar_alignment_cannot_merge_systems() {
    let mut data = fixture(false, false);
    data.barlines
        .iter_mut()
        .find(|b| b.staff == 1)
        .unwrap()
        .aligned_staffs
        .push(2);
    data.barlines
        .iter_mut()
        .find(|b| b.staff == 2)
        .unwrap()
        .aligned_staffs
        .push(1);
    let geometry = Geometry::fit(&data, &Config::default()).unwrap();
    let groups: Vec<Vec<usize>> = geometry
        .report()
        .systems
        .iter()
        .map(|s| s.staves.clone())
        .collect();
    assert_eq!(groups, vec![vec![0, 1], vec![2, 3]]);
}

#[test]
fn unsupported_staff_extensions_are_completed_without_moving_measurements() {
    let x0: Vec<f64> = (400..=800).step_by(10).map(f64::from).collect();
    let x1: Vec<f64> = (200..=800).step_by(10).map(f64::from).collect();
    let first = Staff {
        x: x0.clone(),
        y: std::array::from_fn(|k| {
            x0.iter()
                .map(|x| 140. + 10. * k as f64 - 0.05 * (x - 400.))
                .collect()
        }),
    };
    let second = Staff {
        x: x1.clone(),
        y: std::array::from_fn(|k| {
            x1.iter()
                .map(|x| 175. + 10. * k as f64 + 0.075 * (x - 200.))
                .collect()
        }),
    };
    // Constant left extrapolations overlap by five pixels, but every measured
    // anchor is ordered wherever source support actually exists.
    let data = Observations {
        width: 1000,
        height: 500,
        staves: vec![first, second],
        barlines: vec![],
    };
    let geometry = Geometry::fit(
        &data,
        &Config {
            perspective: false,
            ..Config::default()
        },
    )
    .unwrap();
    let r = geometry.report();
    assert_eq!(r.extrapolation.model, "local_support_monotone_columns");
    assert_eq!(r.extrapolation.support_limit_staff_spaces, 8.);
    assert!(r.extrapolation.min_source_gap_pixels > 0.);
    assert!(r.extrapolation.max_vertical_scale <= 4. + 1e-8);
    assert!(r.validation.max_staff_line_span_pixels < 1e-8);
    assert!(r.validation.sampled_min_jacobian > 0.);
    for x in [0., 100., 300., 400., 600., 999.] {
        for y in [1., 160., 185., 220., 499.] {
            let p = Point { x, y };
            assert_close(geometry.inverse(geometry.forward(p)), p, 1e-7);
        }
    }
}

#[test]
fn shared_shear_can_harm_a_system_with_different_local_directions() {
    let mut data = fixture(false, false);
    // A smooth deformation may produce different horizontal motion over the
    // page width. These sampled bar sections have two opposite directions.
    for (i, b) in data.barlines.iter_mut().enumerate() {
        let local_slope = if i % 5 < 2 { -0.08 } else { 0.08 };
        let center = (b.y[0] + b.y[b.y.len() - 1]) / 2.;
        for (x, y) in b.x.iter_mut().zip(&b.y) {
            *x += local_slope * (y - center);
        }
    }
    let unsheared_config = Config {
        perspective: false,
        ..Config::default()
    };
    let default = Geometry::fit(&data, &unsheared_config).unwrap();
    let sheared = Geometry::fit(
        &data,
        &Config {
            system_shear: true,
            ..unsheared_config
        },
    )
    .unwrap();
    assert!(default.report().systems.iter().all(|s| s.shear == 0.));
    assert!(
        default
            .report()
            .systems
            .iter()
            .any(|s| s.candidate_shear.abs() > 0.07)
    );
    for b in data.barlines.iter().step_by(5) {
        let lo = Point {
            x: b.x[0],
            y: b.y[0],
        };
        let hi = Point {
            x: *b.x.last().unwrap(),
            y: *b.y.last().unwrap(),
        };
        let slope = |g: &Geometry| {
            let a = g.forward(lo);
            let z = g.forward(hi);
            (z.x - a.x) / (z.y - a.y)
        };
        assert!((slope(&default) + 0.08).abs() < 1e-8);
        assert!(slope(&sheared).abs() > slope(&default).abs() * 1.9);
    }
    // With no perspective or final shear requested, x is preserved exactly,
    // even when the staff-derived vertical correction is active.
    for p in [Point { x: 200., y: 250. }, Point { x: 750., y: 980. }] {
        assert_eq!(default.forward(p).x, p.x);
    }
}

#[cfg(feature = "render")]
#[test]
fn identity_render_samples_correct_pixels_and_checks_dimensions() {
    let data = fixture(false, false);
    let geometry = Geometry::fit(&data, &Config::default()).unwrap();
    let image = image::RgbImage::from_fn(1000, 1400, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, 91])
    });
    let out = geometry.render(&image).unwrap();
    assert_eq!(out, image);
    assert!(geometry.render(&image::RgbImage::new(10, 10)).is_err());
}

#[cfg(feature = "render")]
#[test]
fn aspect_prior_changes_horizontal_raster_scale_with_single_reconstruction() {
    let data = fixture(false, false);
    let geometry = Geometry::fit(&data, &Config::default()).unwrap();
    let image = image::RgbImage::from_fn(1000, 1400, |x, y| {
        image::Rgb([(x / 5) as u8, (y % 251) as u8, 91])
    });
    let before = geometry.forward(Point { x: 500., y: 700. });
    let out = geometry.render_with_aspect(&image, Some(2.)).unwrap();
    assert_eq!(out.dimensions(), (700, 1400));
    for y in [0, 150, 700, 1399] {
        assert_eq!(out.get_pixel(0, y), image.get_pixel(0, y));
        assert_eq!(out.get_pixel(699, y), image.get_pixel(999, y));
    }
    let after = geometry.forward(Point { x: 500., y: 700. });
    assert_eq!((before.x, before.y), (after.x, after.y));
    for ratio in [f64::NAN, f64::INFINITY, 0., -1., 20.] {
        assert!(geometry.raster_size(Some(ratio)).is_err());
    }
}

#[test]
fn paper_frame_stays_vertical_after_staff_flattening_without_system_shear() {
    let data = fixture(true, true);
    let corners = [
        Point { x: 160., y: 180. },
        Point { x: 840., y: 180. },
        Point { x: 840., y: 1200. },
        Point { x: 160., y: 1200. },
    ]
    .map(camera_inverse);
    let (geometry, _) =
        Geometry::fit_profiled_with_page_frame(&data, &Config::default(), Some(corners));
    let geometry = geometry.unwrap();
    assert_eq!(geometry.report().perspective.source, "page_boundary");
    for x in [160., 840.] {
        let mapped: Vec<_> = (0..40)
            .map(|i| {
                geometry.forward(camera_inverse(Point {
                    x,
                    y: 180. + 1020. * i as f64 / 39.,
                }))
            })
            .collect();
        let left = mapped.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let right = mapped.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        assert!(right - left < 1e-6);
    }
    assert!(geometry.report().validation.sampled_min_jacobian > 0.);
}

#[cfg(feature = "render")]
#[test]
fn page_aspect_prior_targets_paper_extent_not_extra_canvas() {
    let data = fixture(false, false);
    let geometry = Geometry::fit(&data, &Config::default()).unwrap();
    let contour = [
        Point { x: 100., y: 100. },
        Point { x: 900., y: 100. },
        Point { x: 900., y: 1300. },
        Point { x: 100., y: 1300. },
    ];
    let canvas = geometry.canvas_ratio_for_page(2., &contour).unwrap();
    let (w, h) = geometry.raster_size(Some(canvas)).unwrap();
    let bounds = geometry.report().bounds;
    let pixel_ratio = (1200. / 800.) * f64::from(h - 1) / f64::from(w - 1)
        * (bounds[2] - bounds[0])
        / (bounds[3] - bounds[1]);
    assert!((pixel_ratio - 2.).abs() < 0.005);
    assert!((f64::from(h) / f64::from(w) - 2.).abs() > 0.05);
}

#[test]
fn independent_staff_edges_constrain_camera_without_becoming_bars() {
    use smude_geometry::{StaffEdge, StaffEdgeSide};
    let mut data = fixture(true, false);
    data.barlines.clear();
    let mut edges = vec![];
    for (i, center) in [260., 440., 840., 1020.].iter().enumerate() {
        for (side, x) in [(StaffEdgeSide::Start, 160.), (StaffEdgeSide::End, 840.)] {
            edges.push(StaffEdge {
                staff: i,
                side,
                points: (0..5)
                    .map(|k| {
                        camera_inverse(Point {
                            x,
                            y: center + (k as f64 - 2.) * 14.,
                        })
                    })
                    .collect(),
                line_indices: (0..5).collect(),
                confidence: 1.,
                aligned_staffs: (0..4).filter(|j| *j != i).collect(),
            });
        }
    }
    let config = Config::default();
    let plain = Geometry::fit(&data, &config).unwrap();
    let constrained = Geometry::fit_profiled_with_staff_edges(&data, &config, None, &edges)
        .0
        .unwrap();
    let spread = |g: &Geometry| {
        [StaffEdgeSide::Start, StaffEdgeSide::End]
            .iter()
            .map(|side| {
                let xx: Vec<_> = edges
                    .iter()
                    .filter(|e| e.side == *side)
                    .flat_map(|e| e.points.iter().map(|p| g.forward(*p).x))
                    .collect();
                let mean = xx.iter().sum::<f64>() / xx.len() as f64;
                xx.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / xx.len() as f64
            })
            .sum::<f64>()
            .sqrt()
    };
    assert!(
        spread(&constrained) < spread(&plain) * 0.25,
        "{} vs {}",
        spread(&constrained),
        spread(&plain)
    );
    assert!(data.barlines.is_empty());
    assert_eq!(constrained.report().staff_edges.len(), 8);
    assert!(constrained.report().validation.sampled_min_jacobian > 0.);
}

#[test]
fn measured_staff_endpoints_extend_missing_tip_without_inventing_bars() {
    use smude_geometry::{StaffEdge, StaffEdgeSide};
    let data = Observations {
        width: 1000,
        height: 700,
        staves: vec![Staff {
            x: vec![200., 500., 800.],
            y: std::array::from_fn(|k| {
                [200., 500., 800.]
                    .map(|x| 200. + k as f64 * 10. + 0.05 * (x - 200.))
                    .to_vec()
            }),
        }],
        barlines: vec![],
    };
    let config = Config {
        perspective: false,
        ..Config::default()
    };
    let baseline = Geometry::fit(&data, &config).unwrap();
    let edge = StaffEdge {
        staff: 0,
        side: StaffEdgeSide::Start,
        points: [0, 2, 4]
            .map(|k| Point {
                x: 140.,
                y: 197. + k as f64 * 10.,
            })
            .to_vec(),
        line_indices: vec![0, 2, 4],
        confidence: 1.,
        aligned_staffs: vec![],
    };
    let g =
        Geometry::fit_profiled_with_staff_edges(&data, &config, None, std::slice::from_ref(&edge))
            .0
            .unwrap();
    for (k, p) in edge.line_indices.iter().zip(&edge.points) {
        let target = g
            .forward(Point {
                x: 500.,
                y: data.staves[0].y[*k][1],
            })
            .y;
        assert!((g.forward(*p).y - target).abs() < 1e-8);
    }
    let p = edge.points[0];
    let before = (baseline.forward(p).y
        - baseline
            .forward(Point {
                x: 500.,
                y: data.staves[0].y[0][1],
            })
            .y)
        .abs();
    assert!(before > 0.5);
    assert!(g.report().validation.max_staff_line_span_pixels < 1e-8);
    assert!(data.barlines.is_empty());
    let mut uncertain = edge;
    uncertain.confidence = 0.4;
    let unchanged = Geometry::fit_profiled_with_staff_edges(&data, &config, None, &[uncertain])
        .0
        .unwrap();
    assert_close(baseline.forward(p), unchanged.forward(p), 1e-9);
}
