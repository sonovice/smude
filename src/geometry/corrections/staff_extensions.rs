//! Complete a partially measured five-line boundary without creating constant-y
//! phantom tails. Original curve samples and admitted ink endpoints are retained.
use crate::geometry::{Point, StaffEdge, StaffEdgeSide};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct StaffEndpointRejection {
    pub staff: usize,
    pub line_index: usize,
    pub side: StaffEdgeSide,
    pub mapped_point: Point,
    pub reason: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct StaffExtensionReport {
    pub coordinate_space: &'static str,
    pub accepted_endpoint_samples: usize,
    pub inferred_tail_samples: usize,
    pub rejected_endpoints: Vec<StaffEndpointRejection>,
}

impl Default for StaffExtensionReport {
    fn default() -> Self {
        Self {
            coordinate_space: "post_camera_pre_staff_warp_pixels",
            accepted_endpoint_samples: 0,
            inferred_tail_samples: 0,
            rejected_endpoints: vec![],
        }
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    (values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.
}

fn tangent(line: &[Point], start: bool) -> f64 {
    let count = line.len().min(4);
    let points = if start {
        &line[..count]
    } else {
        &line[line.len() - count..]
    };
    let cx = points.iter().map(|p| p.x).sum::<f64>() / count as f64;
    let cy = points.iter().map(|p| p.y).sum::<f64>() / count as f64;
    points.iter().map(|p| (p.x - cx) * (p.y - cy)).sum::<f64>()
        / points
            .iter()
            .map(|p| (p.x - cx).powi(2))
            .sum::<f64>()
            .max(1e-12)
}

pub(crate) fn extend_measured_ends(
    staff: usize,
    lines: &mut [Vec<Point>; 5],
    edges: &[StaffEdge],
    report: &mut StaffExtensionReport,
) {
    let core = lines.clone();
    for side in [StaffEdgeSide::Start, StaffEdgeSide::End] {
        let start = matches!(side, StaffEdgeSide::Start);
        let endpoints: [Point; 5] = std::array::from_fn(|k| {
            if start {
                core[k][0]
            } else {
                *core[k].last().unwrap()
            }
        });
        let slope = median(core.iter().map(|line| tangent(line, start)).collect());
        let centers: Vec<_> = endpoints.iter().map(|p| p.y - slope * p.x).collect();
        let space = median(centers.windows(2).map(|p| p[1] - p[0]).collect());
        if !space.is_finite() || space <= 0. {
            continue;
        }
        let predict = |k: usize, x: f64| endpoints[k].y + slope * (x - endpoints[k].x);
        let mut candidates = vec![];
        for edge in edges
            .iter()
            .filter(|e| e.staff == staff && e.confidence >= 0.7 && e.side == side)
        {
            for (k, p) in edge.line_indices.iter().zip(&edge.points) {
                let distance = if start {
                    endpoints[*k].x - p.x
                } else {
                    p.x - endpoints[*k].x
                };
                if distance <= 1e-10 {
                    continue;
                }
                candidates.push((*k, *p, p.y - predict(*k, p.x)));
            }
        }
        if candidates.is_empty() {
            continue;
        }
        let common = median(candidates.iter().map(|p| p.2).collect());
        candidates.retain(|(k, p, residual)| {
            if (residual - common).abs() <= 0.45 * space {
                true
            } else {
                report.rejected_endpoints.push(StaffEndpointRejection {
                    staff,
                    line_index: *k,
                    side,
                    mapped_point: *p,
                    reason: "endpoint phase conflicts with measured five-line boundary",
                });
                false
            }
        });
        if candidates.is_empty() {
            continue;
        }
        let common = median(candidates.iter().map(|p| p.2).collect());
        let extent = if start {
            candidates
                .iter()
                .map(|p| p.1.x)
                .chain(endpoints.iter().map(|p| p.x))
                .fold(f64::INFINITY, f64::min)
        } else {
            candidates
                .iter()
                .map(|p| p.1.x)
                .chain(endpoints.iter().map(|p| p.x))
                .fold(f64::NEG_INFINITY, f64::max)
        };
        for (k, line) in lines.iter_mut().enumerate() {
            if (endpoints[k].x - extent).abs() <= 1e-10 {
                continue;
            }
            let mut measured: Vec<_> = candidates
                .iter()
                .filter(|p| p.0 == k)
                .map(|p| p.1)
                .collect();
            measured.sort_by(|a, b| a.x.total_cmp(&b.x));
            measured.dedup_by(|a, b| (a.x - b.x).abs() < 1e-10);
            report.accepted_endpoint_samples += measured.len();
            let residual = if measured.is_empty() {
                common
            } else {
                median(measured.iter().map(|p| p.y - predict(k, p.x)).collect())
            };
            let actual_edge = if start {
                measured.first()
            } else {
                measured.last()
            };
            let needs_virtual = actual_edge.is_none_or(|p| (p.x - extent).abs() > 1e-10);
            if needs_virtual {
                let p = Point {
                    x: extent,
                    y: predict(k, extent) + residual,
                };
                if start {
                    measured.insert(0, p);
                } else {
                    measured.push(p);
                }
                report.inferred_tail_samples += 1;
            }
            if start {
                measured.append(line);
                *line = measured;
            } else {
                line.extend(measured);
            }
        }
    }
}
