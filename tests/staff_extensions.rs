use smude_geometry::{Config, Geometry, Observations, Point, Staff, StaffEdge, StaffEdgeSide};

fn data() -> Observations {
    Observations {
        width: 400,
        height: 350,
        staves: vec![Staff {
            x: vec![100., 150., 200.],
            y: std::array::from_fn(|k| {
                vec![
                    150. + 10. * k as f64,
                    125. + 10. * k as f64,
                    100. + 10. * k as f64,
                ]
            }),
        }],
        barlines: vec![],
    }
}
fn edge(indices: &[usize], staggered: bool, bend: f64) -> StaffEdge {
    StaffEdge {
        staff: 0,
        side: StaffEdgeSide::Start,
        line_indices: indices.to_vec(),
        confidence: 1.,
        aligned_staffs: vec![],
        points: indices
            .iter()
            .map(|k| {
                let x = 60. + if staggered { 2. * *k as f64 } else { 0. };
                Point {
                    x,
                    y: 200. - 0.5 * x + 10. * *k as f64 + bend,
                }
            })
            .collect(),
    }
}
fn fit(data: &Observations, edge: &StaffEdge) -> Geometry {
    Geometry::fit_profiled_with_staff_edges(
        data,
        &Config {
            perspective: false,
            ..Default::default()
        },
        None,
        std::slice::from_ref(edge),
    )
    .0
    .unwrap()
}
fn assert_measurements_retained(geometry: &Geometry, original: &Observations, edge: &StaffEdge) {
    for k in 0..5 {
        let target = geometry
            .forward(Point {
                x: 100.,
                y: 150. + 10. * k as f64,
            })
            .y;
        for (x, y) in original.staves[0].x.iter().zip(&original.staves[0].y[k]) {
            assert!((geometry.forward(Point { x: *x, y: *y }).y - target).abs() < 1e-8);
        }
    }
    for (k, p) in edge.line_indices.iter().zip(&edge.points) {
        let target = geometry
            .forward(Point {
                x: 100.,
                y: 150. + 10. * *k as f64,
            })
            .y;
        assert!((geometry.forward(*p).y - target).abs() < 1e-8);
    }
}

#[test]
fn partial_measured_boundary_does_not_create_constant_y_ghost_lines() {
    let data = data();
    let edge = edge(&[0, 1, 3, 4], false, 0.);
    let geometry = fit(&data, &edge);
    assert_measurements_retained(&geometry, &data, &edge);
    let target = geometry.forward(Point { x: 100., y: 170. }).y;
    assert!((geometry.forward(Point { x: 60., y: 190. }).y - target).abs() < 1e-8);
    assert_eq!(
        geometry.report().staff_extensions.accepted_endpoint_samples,
        4
    );
    assert_eq!(geometry.report().staff_extensions.inferred_tail_samples, 1);
    assert!(
        geometry
            .report()
            .staff_extensions
            .rejected_endpoints
            .is_empty()
    );
}

#[test]
fn staggered_valid_endpoints_preserve_all_measured_samples_and_complete_sister_tails() {
    let data = data();
    let edge = edge(&[0, 1, 3, 4], true, 0.);
    let geometry = fit(&data, &edge);
    assert_measurements_retained(&geometry, &data, &edge);
    assert_eq!(
        geometry.report().staff_extensions.accepted_endpoint_samples,
        4
    );
    assert_eq!(geometry.report().staff_extensions.inferred_tail_samples, 4);
}

#[test]
fn one_wrong_line_phase_is_reported_without_dropping_valid_tips() {
    let data = data();
    let mut edge = edge(&[0, 1, 2, 3, 4], false, 0.);
    edge.points[2].y += 10.;
    let geometry = fit(&data, &edge);
    assert_eq!(
        geometry.report().staff_extensions.rejected_endpoints.len(),
        1
    );
    assert_eq!(
        geometry.report().staff_extensions.rejected_endpoints[0].line_index,
        2
    );
    assert_eq!(
        geometry.report().staff_extensions.accepted_endpoint_samples,
        4
    );
    let mut valid = edge.clone();
    valid.points.remove(2);
    valid.line_indices.remove(2);
    assert_measurements_retained(&geometry, &data, &valid);
}

#[test]
fn coherent_curved_tip_is_not_rejected_by_a_straight_tangent_phase_prior() {
    let data = data();
    let edge = edge(&[0, 1, 2, 3, 4], false, 10.);
    let geometry = fit(&data, &edge);
    assert_measurements_retained(&geometry, &data, &edge);
    assert_eq!(
        geometry.report().staff_extensions.accepted_endpoint_samples,
        5
    );
    assert!(
        geometry
            .report()
            .staff_extensions
            .rejected_endpoints
            .is_empty()
    );
}

#[test]
fn observed_tip_is_not_pruned_by_a_source_pixel_budget_in_rectified_units() {
    let data = data();
    let mut edge = edge(&[0, 1, 3, 4], false, 0.);
    for (k, p) in edge.line_indices.iter().zip(&mut edge.points) {
        p.x = 10.;
        p.y = 195. + 10. * *k as f64;
    }
    let geometry = fit(&data, &edge);
    assert_measurements_retained(&geometry, &data, &edge);
    assert_eq!(
        geometry.report().staff_extensions.accepted_endpoint_samples,
        4
    );
    assert!(
        geometry
            .report()
            .staff_extensions
            .rejected_endpoints
            .is_empty()
    );
}
