#![cfg(feature = "render")]
use serde::Deserialize;
use smude_geometry::{Config, Geometry, Observations, Point, StaffEdge};
#[derive(Deserialize)]
struct Fixture {
    data: Observations,
    edges: Vec<StaffEdge>,
    corners: [Point; 4],
    ratio: f64,
    camera_center: Point,
}
fn span(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - v.iter().copied().fold(f64::INFINITY, f64::min)
}
#[test]
fn common_music_columns_are_aligned_by_one_camera_fit_and_first_system_stays_indented() {
    let f: Fixture = serde_json::from_str(include_str!("fixtures/system_alignment.json")).unwrap();
    let geometry = Geometry::fit_profiled_gcs_with_camera(
        &f.data,
        &Config::default(),
        Some(f.corners),
        Some(f.ratio),
        &f.edges,
        true,
        Some(f.camera_center),
    )
    .0
    .unwrap();
    let space =
        geometry.report().target_staff_spacings.iter().sum::<f64>() / f.data.staves.len() as f64;
    let mut left = vec![];
    let mut right = vec![];
    for i in (0..f.data.staves.len()).step_by(2) {
        let bar = f
            .data
            .barlines
            .iter()
            .filter(|b| b.staff == i && !b.aligned_staffs.is_empty())
            .min_by(|a, b| {
                (a.x.iter().sum::<f64>() / a.x.len() as f64)
                    .total_cmp(&(b.x.iter().sum::<f64>() / b.x.len() as f64))
            })
            .unwrap();
        left.push(
            bar.x
                .iter()
                .zip(&bar.y)
                .map(|(x, y)| geometry.forward(Point { x: *x, y: *y }).x)
                .sum::<f64>()
                / bar.x.len() as f64,
        );
        let points: Vec<_> = f
            .edges
            .iter()
            .filter(|e| {
                [i, i + 1].contains(&e.staff) && e.side == smude_geometry::StaffEdgeSide::End
            })
            .flat_map(|e| e.points.iter().map(|p| geometry.forward(*p).x))
            .collect();
        right.push(points.iter().sum::<f64>() / points.len() as f64);
    }
    assert!(
        span(&left[1..]) < 0.35 * space,
        "common left boundary still drifts: {left:?}"
    );
    assert!(
        span(&right) < 0.35 * space,
        "common right boundary still drifts: {right:?}"
    );
    assert!(
        left[0] - left[1] > 5. * space,
        "intentional first-system indentation was flattened"
    );
    assert!(geometry.report().validation.bar_p90_angle_degrees < 1.);
    assert!(geometry.report().validation.max_staff_line_span_pixels < 0.3 * space);
    assert!(geometry.report().validation.sampled_min_jacobian > 0.5);
    assert!(geometry.report().validation.roundtrip_max_pixels < 0.001);
    assert_eq!(
        geometry
            .report()
            .gcs
            .as_ref()
            .unwrap()
            .model
            .boundary_columns
            .len(),
        2
    );
}
