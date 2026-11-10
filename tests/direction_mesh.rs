use smude_geometry::geometry::{DirectionMesh, VerticalPath};
use smude_geometry::{Barline, Config, Geometry, Observations, Point, Staff};

// Independent non-cylindrical deformation. x has height-dependent translation
// AND width scale; y bends with both coordinates. No fitted map generates data.
fn warp(x: f64, y: f64, curved_y: bool) -> Point {
    let a = (std::f64::consts::PI * y / 1000.).sin();
    Point {
        x: x + 20. * a * (1. + 0.25 * (x - 400.) / 400.),
        y: y + if curved_y {
            9. * (std::f64::consts::TAU * x / 800.).sin() * a
        } else {
            0.
        },
    }
}
fn scene(curved_y: bool) -> Observations {
    let mut staves = vec![];
    let mut bars = vec![];
    for staff in 0..6 {
        let y = 100. + staff as f64 * 140.;
        let curves: [Vec<Point>; 5] = std::array::from_fn(|k| {
            (0..257)
                .map(|j| warp(80. + j as f64 * 640. / 256., y + k as f64 * 10., curved_y))
                .collect()
        });
        let lo = curves
            .iter()
            .map(|c| c[0].x)
            .fold(f64::NEG_INFINITY, f64::max);
        let hi = curves
            .iter()
            .map(|c| c.last().unwrap().x)
            .fold(f64::INFINITY, f64::min);
        let x: Vec<_> = (0..129).map(|j| lo + (hi - lo) * j as f64 / 128.).collect();
        let yy = std::array::from_fn(|k| {
            x.iter()
                .map(|x| {
                    let i = curves[k]
                        .partition_point(|p| p.x < *x)
                        .clamp(1, curves[k].len() - 1);
                    let a = curves[k][i - 1];
                    let b = curves[k][i];
                    a.y + (b.y - a.y) * (*x - a.x) / (b.x - a.x)
                })
                .collect()
        });
        staves.push(Staff { x, y: yy });
        for x in [120., 280., 440., 600., 700.] {
            let p: Vec<_> = (0..17)
                .map(|j| warp(x, y + j as f64 * 2.5, curved_y))
                .collect();
            bars.push(Barline {
                staff,
                x: p.iter().map(|p| p.x).collect(),
                y: p.iter().map(|p| p.y).collect(),
                aligned_staffs: vec![],
            });
        }
    }
    Observations {
        width: 800,
        height: 1000,
        staves,
        barlines: bars,
    }
}
fn angle(p: &[Point]) -> f64 {
    let a = p[0];
    let b = *p.last().unwrap();
    ((b.x - a.x) / (b.y - a.y)).atan().to_degrees().abs()
}
#[test]
fn shared_field_corrects_unseen_non_cylindrical_directions_and_is_invertible() {
    let data = scene(false);
    let mesh = DirectionMesh::fit(&data, &|p| p, &[], &[], [0., 0., 799., 999.]);
    println!("{:?}", mesh.report());
    assert!(mesh.report().accepted_scale > 0.);
    assert!(
        mesh.report().after_rms_direction_degrees
            < mesh.report().before_rms_direction_degrees * 0.4
    );
    for x in [190., 360., 530.] {
        for y in [170., 310., 450., 590., 730.] {
            let p: [Point; 2] = [y, y + 35.].map(|y| warp(x, y, false));
            let q = p.map(|p| mesh.forward(p));
            assert!(
                angle(&q) < 0.5,
                "unseen direction {} -> {}",
                angle(&p),
                angle(&q)
            );
        }
    }
    for y in [-20., 0., 137., 500., 999., 1030.] {
        for x in [-30., 0., 110., 390., 799., 830.] {
            let p = Point { x, y };
            let q = mesh.forward(p);
            let r = mesh.inverse(q);
            assert_eq!(q.y, p.y);
            assert_eq!(r.y, p.y);
            assert!((r.x - p.x).abs() < 1e-7, "roundtrip {p:?} {r:?}");
            assert!(mesh.jacobian_det(p) >= mesh.report().minimum_x_derivative_bound - 1e-12);
            assert!(mesh.jacobian_det(p) <= mesh.report().maximum_x_derivative_bound + 1e-12);
        }
    }
}
#[test]
fn one_vertical_does_not_enable_a_mesh() {
    let mut data = scene(false);
    data.barlines.truncate(1);
    let mesh = DirectionMesh::fit(&data, &|p| p, &[], &[], [0., 0., 799., 999.]);
    let p = Point { x: 430., y: 345. };
    let q = mesh.forward(p);
    assert_eq!((p.x, p.y), (q.x, q.y));
    assert_eq!(mesh.report().accepted_scale, 0.);
}
#[test]
fn short_unsupported_paths_cannot_override_precise_bars() {
    let data = scene(false);
    let verticals: Vec<_> = (0..100)
        .map(|i| VerticalPath {
            points: (0..9)
                .map(|j| Point {
                    x: 300. + i as f64 * 0.01 + j as f64 * 0.75,
                    y: 400. + j as f64 * 0.1,
                })
                .collect(),
            weight: 1.,
        })
        .collect();
    let a = DirectionMesh::fit(&data, &|p| p, &[], &[], [0., 0., 799., 999.]);
    let b = DirectionMesh::fit(&data, &|p| p, &[], &verticals, [0., 0., 799., 999.]);
    assert_eq!(b.report().stems, 0);
    for p in [Point { x: 150., y: 180. }, Point { x: 680., y: 820. }] {
        assert!((a.forward(p).x - b.forward(p).x).abs() < 1e-10);
    }
}
#[cfg(feature = "render")]
#[test]
fn complete_mesh_pipeline_uses_spacing_and_never_fits_gcs() {
    let data = scene(true);
    let config = Config {
        perspective: false,
        ..Config::default()
    };
    let (geometry, profile) =
        Geometry::fit_profiled_direction_mesh(&data, &config, None, None, &[], &[]);
    let geometry = geometry.unwrap();
    println!("{:?}", geometry.report().direction_mesh);
    assert_eq!(profile.gcs_seconds, 0.);
    assert!(geometry.report().gcs.is_none());
    assert!(
        geometry
            .report()
            .direction_mesh
            .as_ref()
            .unwrap()
            .field
            .spacing_samples
            >= 100
    );
    assert!(geometry.report().validation.max_staff_line_span_pixels < 0.5);
    assert!(geometry.report().validation.sampled_min_jacobian > 0.);
    for y in [50., 200., 500., 800., 950.] {
        for x in [20., 200., 450., 750.] {
            let p = Point { x, y };
            let r = geometry.inverse(geometry.forward(p));
            assert!((r.x - p.x).hypot(r.y - p.y) < 1e-6);
        }
    }
}

#[test]
fn repeating_short_measurements_does_not_increase_their_influence() {
    let data = scene(false);
    let paths: Vec<_> = [190., 350., 510.]
        .into_iter()
        .map(|x| VerticalPath {
            points: (0..9)
                .map(|j| warp(x, 300. + j as f64 * 4., false))
                .collect(),
            weight: 0.8,
        })
        .collect();
    let repeated: Vec<_> = paths
        .iter()
        .flat_map(|p| std::iter::repeat_n(p.clone(), 8))
        .collect();
    let a = DirectionMesh::fit(&data, &|p| p, &[], &paths, [0., 0., 799., 999.]);
    let b = DirectionMesh::fit(&data, &|p| p, &[], &repeated, [0., 0., 799., 999.]);
    for p in [
        Point { x: 200., y: 200. },
        Point { x: 500., y: 500. },
        Point { x: 700., y: 800. },
    ] {
        assert!((a.forward(p).x - b.forward(p).x).abs() < 1e-7);
    }
}

#[test]
fn independent_staff_ends_support_direction_without_requiring_shared_columns() {
    use smude_geometry::{StaffEdge, StaffEdgeSide};
    let mut data = scene(false);
    data.barlines.clear();
    let edges: Vec<_> = (0..6)
        .flat_map(|staff| {
            // Deliberately staggered widths: no global alignment is being assumed.
            [StaffEdgeSide::Start, StaffEdgeSide::End].map(move |side| {
                let x = if side == StaffEdgeSide::Start {
                    90. + staff as f64 * 9.
                } else {
                    710. - staff as f64 * 7.
                };
                StaffEdge {
                    staff,
                    side,
                    confidence: 1.,
                    aligned_staffs: vec![],
                    line_indices: vec![0, 1, 2, 3, 4],
                    points: (0..5)
                        .map(|k| warp(x, 100. + staff as f64 * 140. + k as f64 * 10., false))
                        .collect(),
                }
            })
        })
        .collect();
    let mesh = DirectionMesh::fit(&data, &|p| p, &edges, &[], [0., 0., 799., 999.]);
    assert!(mesh.report().accepted_scale > 0., "{:?}", mesh.report());
    assert!(
        mesh.report().after_rms_direction_degrees
            < mesh.report().before_rms_direction_degrees * 0.5
    );
    // Incomplete / incoherent endpoints must not authorize the same correction.
    let sparse = DirectionMesh::fit(&data, &|p| p, &edges[..4], &[], [0., 0., 799., 999.]);
    assert_eq!(
        sparse.report().accepted_scale,
        0.,
        "two staves are not enough edge-only support"
    );
    let mut bad = edges;
    for e in &mut bad {
        e.points[2].x += 15.;
    }
    let rejected = DirectionMesh::fit(&data, &|p| p, &bad, &[], [0., 0., 799., 999.]);
    assert_eq!(rejected.report().accepted_scale, 0.);
}

#[test]
fn alternating_system_directions_fit_without_sacrificing_already_upright_bars() {
    let bend = |x: f64, y: f64| x + 9. * (std::f64::consts::TAU * y / 400.).sin();
    let staves = (0..8)
        .map(|i| {
            let y = 60. + i as f64 * 110.;
            Staff {
                x: vec![40., 760.],
                y: std::array::from_fn(|k| vec![y + k as f64 * 8.; 2]),
            }
        })
        .collect::<Vec<_>>();
    let barlines = staves
        .iter()
        .enumerate()
        .flat_map(|(staff, s)| {
            [160., 400., 640.].map(|x| {
                let y = (0..33).map(|j| s.y[0][0] + j as f64).collect::<Vec<_>>();
                Barline {
                    staff,
                    x: y.iter().map(|y| bend(x, *y)).collect(),
                    y,
                    aligned_staffs: vec![],
                }
            })
        })
        .collect();
    let data = Observations {
        width: 800,
        height: 1000,
        staves,
        barlines,
    };
    let mesh = DirectionMesh::fit(&data, &|p| p, &[], &[], [0., 0., 799., 999.]);
    assert!(mesh.report().accepted_scale > 0.25, "{:?}", mesh.report());
    for b in &data.barlines {
        let p =
            b.x.iter()
                .zip(&b.y)
                .map(|(x, y)| Point { x: *x, y: *y })
                .collect::<Vec<_>>();
        let q = p.iter().map(|p| mesh.forward(*p)).collect::<Vec<_>>();
        assert!(angle(&q) <= angle(&p) + 0.11);
    }
    assert!(
        mesh.report().after_rms_direction_degrees
            < mesh.report().before_rms_direction_degrees * 0.55
    );
}
