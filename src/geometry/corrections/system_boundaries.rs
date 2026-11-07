//! Conservative cross-system columns from independent staff ends or full bars.
//! Membership is decided in the source image, before fitting any camera.
use crate::geometry::{Observations, Point, StaffEdge, StaffEdgeSide};
use crate::math::median;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct BoundaryColumn {
    pub side: StaffEdgeSide,
    pub systems: Vec<Vec<usize>>,
    pub sources: Vec<&'static str>,
    pub source_slope: f64,
    pub source_max_center_error_pixels: f64,
    pub points: Vec<Point>,
}
struct Candidate {
    system: Vec<usize>,
    points: Vec<Point>,
    source: &'static str,
}
fn center(points: &[Point]) -> Point {
    Point {
        x: points.iter().map(|p| p.x).sum::<f64>() / points.len() as f64,
        y: points.iter().map(|p| p.y).sum::<f64>() / points.len() as f64,
    }
}
fn slope(points: &[Point]) -> f64 {
    let c = center(points);
    points
        .iter()
        .map(|p| (p.x - c.x) * (p.y - c.y))
        .sum::<f64>()
        / points
            .iter()
            .map(|p| (p.y - c.y).powi(2))
            .sum::<f64>()
            .max(1e-12)
}
fn sample(points: &[Point]) -> Vec<Point> {
    (0..8).map(|i| points[i * (points.len() - 1) / 7]).collect()
}
fn consensus(candidates: &[Candidate], space: f64) -> Option<Vec<usize>> {
    if candidates.len() < 3 {
        return None;
    }
    let centers: Vec<_> = candidates.iter().map(|c| center(&c.points)).collect();
    let tolerance = (0.3 * space).max(1.5);
    let inliers = |m: f64, b: f64| -> Vec<usize> {
        candidates
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let lo = c.points.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
                let hi = c
                    .points
                    .iter()
                    .map(|p| p.y)
                    .fold(f64::NEG_INFINITY, f64::max);
                let rms = (c
                    .points
                    .iter()
                    .map(|p| (p.x - m * p.y - b).powi(2))
                    .sum::<f64>()
                    / c.points.len() as f64)
                    .sqrt();
                (rms < tolerance && (slope(&c.points) - m).abs() < 0.02 + 1.5 / (hi - lo).max(1.))
                    .then_some(i)
            })
            .collect()
    };
    let mut best = vec![];
    let mut error = f64::INFINITY;
    for (i, a) in centers.iter().enumerate() {
        for b in centers.iter().skip(i + 1) {
            if (a.y - b.y).abs() < 8. * space {
                continue;
            }
            let m = (b.x - a.x) / (b.y - a.y);
            let intercept = a.x - m * a.y;
            let indices = inliers(m, intercept);
            let residual = indices
                .iter()
                .map(|i| (centers[*i].x - m * centers[*i].y - intercept).powi(2))
                .sum::<f64>();
            if indices.len() > best.len() || (indices.len() == best.len() && residual < error) {
                best = indices;
                error = residual;
            }
        }
    }
    if best.len() < 3 || best.len() * 5 < candidates.len() * 3 {
        return None;
    }
    let points: Vec<_> = best.iter().map(|i| centers[*i]).collect();
    let c = center(&points);
    let m = slope(&points);
    let refined = inliers(m, c.x - m * c.y);
    (refined.len() >= 3 && refined.len() * 5 >= candidates.len() * 3).then_some(refined)
}

pub(crate) fn detect(data: &Observations, edges: &[StaffEdge]) -> Vec<BoundaryColumn> {
    let space = median(
        data.staves
            .iter()
            .map(|s| (median(s.y[4].clone()) - median(s.y[0].clone())) / 4.)
            .collect(),
    );
    let groups = crate::geometry::fit::systems(data);
    let mut output = vec![];
    for side in [StaffEdgeSide::Start, StaffEdgeSide::End] {
        let mut candidates = vec![];
        for group in &groups {
            let points: Vec<_> = edges
                .iter()
                .filter(|e| group.contains(&e.staff) && e.side == side && e.confidence >= 0.7)
                .flat_map(|e| e.points.iter().copied())
                .collect();
            if points.len() >= 3 {
                candidates.push(Candidate {
                    system: group.clone(),
                    points: sample(&points),
                    source: "independent_staff_ends",
                });
                continue;
            }
            // A continuous bar at the observed rule boundary supplies genuine
            // endpoint evidence even when the connected stroke defeats the
            // ink-to-blank staff-end detector. Ordinary inner bars are excluded.
            let bars: Vec<_> = data
                .barlines
                .iter()
                .filter(|bar| group.contains(&bar.staff))
                .filter_map(|bar| {
                    let mut members = bar.aligned_staffs.clone();
                    members.push(bar.staff);
                    members.sort_unstable();
                    members.dedup();
                    if members.len() < 2 || !group.iter().all(|i| members.contains(i)) {
                        return None;
                    }
                    let points: Vec<_> = bar
                        .x
                        .iter()
                        .zip(&bar.y)
                        .map(|(x, y)| Point { x: *x, y: *y })
                        .collect();
                    let x = center(&points).x;
                    let near = group.iter().all(|i| {
                        let staff = &data.staves[*i];
                        let boundary = if side == StaffEdgeSide::Start {
                            staff.x[0]
                        } else {
                            *staff.x.last().unwrap()
                        };
                        (x - boundary).abs() < 1.25 * space
                    });
                    near.then_some(points)
                })
                .collect();
            if bars.len() == 1 {
                candidates.push(Candidate {
                    system: group.clone(),
                    points: sample(&bars[0]),
                    source: "continuous_boundary_bar",
                });
            }
        }
        let Some(indices) = consensus(&candidates, space) else {
            continue;
        };
        let centers: Vec<_> = indices
            .iter()
            .map(|i| center(&candidates[*i].points))
            .collect();
        let c = center(&centers);
        let m = slope(&centers);
        output.push(BoundaryColumn {
            side,
            systems: indices
                .iter()
                .map(|i| candidates[*i].system.clone())
                .collect(),
            sources: indices.iter().map(|i| candidates[*i].source).collect(),
            source_slope: m,
            source_max_center_error_pixels: centers
                .iter()
                .map(|p| (p.x - c.x - m * (p.y - c.y)).abs())
                .fold(0., f64::max),
            points: indices
                .iter()
                .flat_map(|i| candidates[*i].points.iter().copied())
                .collect(),
        });
    }
    // A lone column cannot separate shear, width perspective and surface shape
    // reliably. Require both sides to be independently supported by at least
    // three of the same systems before adding a new global camera constraint.
    if output.len() != 2
        || output[0]
            .systems
            .iter()
            .filter(|group| output[1].systems.contains(group))
            .count()
            < 3
    {
        return vec![];
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Barline, Staff};
    fn data(left: &[f64], right: &[f64]) -> (Observations, Vec<StaffEdge>) {
        let mut staves = vec![];
        let mut barlines = vec![];
        let mut edges = vec![];
        for (i, (left, right)) in left.iter().zip(right).enumerate() {
            let start = 100. + left;
            let end = 900. + right;
            let y = 200. + i as f64 * 300.;
            for j in 0..2 {
                let cy = y + j as f64 * 70.;
                let id = staves.len();
                staves.push(Staff {
                    x: vec![start + 0.025 * (cy + 20.), end + 0.025 * (cy - 20.)],
                    y: std::array::from_fn(|k| vec![cy + (k as f64 - 2.) * 10.; 2]),
                });
                edges.push(StaffEdge {
                    staff: id,
                    side: StaffEdgeSide::End,
                    points: (0..5)
                        .map(|k| {
                            let y = cy + (k as f64 - 2.) * 10.;
                            Point {
                                x: end + 0.025 * y,
                                y,
                            }
                        })
                        .collect(),
                    line_indices: (0..5).collect(),
                    confidence: 1.,
                    aligned_staffs: vec![],
                });
            }
            for x in [start, 500., end] {
                let yy: Vec<_> = (0..17).map(|j| y - 20. + 110. * j as f64 / 16.).collect();
                barlines.push(Barline {
                    staff: i * 2,
                    aligned_staffs: vec![i * 2 + 1],
                    x: yy.iter().map(|y| x + 0.025 * y).collect(),
                    y: yy,
                });
            }
        }
        (
            Observations {
                width: 1100,
                height: 1800,
                staves,
                barlines,
            },
            edges,
        )
    }
    #[test]
    fn continuous_start_bars_supply_shared_column_without_losing_real_indents_or_short_ends() {
        let (data, edges) = data(&[80., 0., 0., 0., 0.], &[0., 0., 0., 0., -120.]);
        let columns = detect(&data, &edges);
        assert_eq!(columns.len(), 2);
        let left = columns
            .iter()
            .find(|c| c.side == StaffEdgeSide::Start)
            .unwrap();
        assert_eq!(
            left.systems,
            vec![vec![2, 3], vec![4, 5], vec![6, 7], vec![8, 9]]
        );
        assert!(left.sources.iter().all(|s| *s == "continuous_boundary_bar"));
        let right = columns
            .iter()
            .find(|c| c.side == StaffEdgeSide::End)
            .unwrap();
        assert_eq!(
            right.systems,
            vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]]
        );
        assert!((left.source_slope - 0.025).abs() < 1e-9);
    }
    #[test]
    fn collinear_centers_do_not_override_individually_incompatible_boundary_directions() {
        let (data, edges) = data(&[0., 15., 30., 45., 60.], &[0.; 5]);
        assert!(detect(&data, &edges).is_empty());
    }
    #[test]
    fn no_shared_column_is_invented_from_inner_bars_or_two_systems() {
        let (mut data, edges) = data(&[0.; 5], &[0.; 5]);
        data.barlines.retain(|b| {
            center(
                &b.x.iter()
                    .zip(&b.y)
                    .map(|(x, y)| Point { x: *x, y: *y })
                    .collect::<Vec<_>>(),
            )
            .x > 300.
        });
        assert!(detect(&data, &edges).is_empty());
        data.staves.truncate(4);
        data.barlines.retain(|b| b.staff < 4);
        let edges: Vec<_> = edges.into_iter().filter(|e| e.staff < 4).collect();
        assert!(detect(&data, &edges).is_empty());
    }
}
