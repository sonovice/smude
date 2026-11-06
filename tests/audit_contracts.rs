//! Public-interface robustness checks independent of detector heuristics.
use smude_geometry::{Barline, Config, Geometry, Observations, Point, Staff};

fn fixture() -> Observations {
    let staves = [80.0, 180.0]
        .iter()
        .map(|center| Staff {
            x: vec![30.0, 160.0, 290.0],
            y: std::array::from_fn(|k| vec![center + (k as f64 - 2.0) * 8.0; 3]),
        })
        .collect();
    let barlines = [80.0, 180.0]
        .iter()
        .enumerate()
        .flat_map(|(staff, center)| {
            [80.0, 160.0, 240.0].iter().map(move |x| Barline {
                staff,
                x: vec![*x; 3],
                y: vec![center - 16.0, *center, center + 16.0],
                aligned_staffs: vec![],
            })
        })
        .collect();
    Observations {
        width: 320,
        height: 260,
        staves,
        barlines,
    }
}

#[test]
fn finite_negative_and_beyond_page_points_roundtrip_without_horizontal_motion() {
    let geometry = Geometry::fit(
        &fixture(),
        &Config {
            perspective: false,
            ..Config::default()
        },
    )
    .unwrap();
    for p in [
        Point { x: -40.0, y: -30.0 },
        Point { x: -40.0, y: 150.0 },
        Point { x: 400.0, y: 300.0 },
        Point { x: 80.0, y: 120.0 },
    ] {
        let mapped = geometry.forward(p);
        assert_eq!(mapped.x, p.x);
        let restored = geometry.inverse(mapped);
        assert!((restored.x - p.x).hypot(restored.y - p.y) < 1e-9);
    }
    assert!(!geometry.report().config.perspective);
    assert!(!geometry.report().config.system_shear);
}

#[test]
fn malformed_measurements_are_errors_not_panics() {
    let mut cases = vec![];
    let mut bad = fixture();
    bad.staves[0].x[0] = -1.0;
    cases.push(bad);
    let mut bad = fixture();
    bad.staves[0].x[1] = f64::NAN;
    cases.push(bad);
    let mut bad = fixture();
    bad.staves[0].y[0][1] = f64::INFINITY;
    cases.push(bad);
    let mut bad = fixture();
    bad.staves[0].y[0].clear();
    cases.push(bad);
    let mut bad = fixture();
    bad.staves[0].x[1] = bad.staves[0].x[0];
    cases.push(bad);
    let mut bad = fixture();
    bad.barlines[0].x[0] = -1.0;
    cases.push(bad);
    let mut bad = fixture();
    bad.barlines[0].y[1] = bad.barlines[0].y[0];
    cases.push(bad);
    let mut bad = fixture();
    bad.barlines[0].staff = 10;
    cases.push(bad);
    for bad in cases {
        let result = std::panic::catch_unwind(|| Geometry::fit(&bad, &Config::default()));
        assert!(result.is_ok(), "invalid observations panicked");
        assert!(
            result.unwrap().is_err(),
            "invalid observations were accepted"
        );
    }
}

#[test]
fn finite_extreme_size_priors_are_rejected_if_they_collapse_target_geometry() {
    for prior in [f64::NAN, f64::INFINITY, -1.0, f64::MAX] {
        let config = Config {
            staff_size_groups: vec![vec![0, 1]],
            size_prior: prior,
            ..Config::default()
        };
        let result = std::panic::catch_unwind(|| Geometry::fit(&fixture(), &config));
        assert!(result.is_ok(), "bad size prior panicked");
        assert!(result.unwrap().is_err(), "bad size prior accepted: {prior}");
    }
}

#[test]
fn crossing_staff_bands_are_rejected_even_when_each_staff_is_internally_ordered() {
    let mut data = fixture();
    for k in 0..5 {
        data.staves[1].y[k][1] = 80.0 + (k as f64 - 2.0) * 8.0;
    }
    assert!(Geometry::fit(&data, &Config::default()).is_err());
}

#[test]
fn nonfinite_point_mapping_must_not_panic() {
    let geometry = Geometry::fit(&fixture(), &Config::default()).unwrap();
    for p in [
        Point {
            x: f64::NAN,
            y: 100.0,
        },
        Point {
            x: 100.0,
            y: f64::NAN,
        },
        Point {
            x: f64::INFINITY,
            y: 100.0,
        },
        Point {
            x: 100.0,
            y: f64::NEG_INFINITY,
        },
    ] {
        for method in [Geometry::forward, Geometry::inverse] {
            let mapped = std::panic::catch_unwind(|| method(&geometry, p));
            assert!(
                mapped.is_ok(),
                "nonfinite point caused public point-map panic"
            );
            let q = mapped.unwrap();
            assert!(
                !q.x.is_finite() || !q.y.is_finite(),
                "nonfinite input became valid-looking coordinates"
            );
        }
    }
}

#[test]
fn duplicate_or_self_alignment_is_not_independent_system_evidence() {
    let mut duplicate = fixture();
    // One physical matched stroke must not count as two reciprocal witnesses.
    duplicate.barlines[0].aligned_staffs = vec![1, 1];
    duplicate.barlines[3].aligned_staffs = vec![0, 0];
    assert!(Geometry::fit(&duplicate, &Config::default()).is_err());
    let mut self_link = fixture();
    self_link.barlines[0].aligned_staffs = vec![0];
    assert!(Geometry::fit(&self_link, &Config::default()).is_err());
}
