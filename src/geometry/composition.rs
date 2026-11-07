//! Coordinate composition for the native page-frame → GCS → local pipeline.
use crate::geometry::{GcsMap, GcsReport, Observations, PageAnalysisMap, Point, Staff, StaffEdge};
use crate::{Error, Result};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct GcsPipelineReport {
    pub four_point: Option<PageAnalysisMap>,
    pub four_point_rejection: Option<String>,
    pub camera_coordinates: &'static str,
    pub gcs_shift: Point,
    pub model: GcsReport,
    pub selected_model: &'static str,
    pub plane_evidence: Option<PlaneEvidence>,
    pub staff_conditioning: crate::geometry::ConditioningReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct PlaneEvidence {
    pub accepted: bool,
    pub staff_horizontal_rms_p90: f64,
    pub bar_horizontal_displacement_rms_p90: f64,
    pub relative_gap_variation_p90: f64,
    pub noise_tolerance_pixels: f64,
    pub reason: &'static str,
}

#[derive(Clone, Debug)]
pub(crate) struct PreGcs {
    pub frame: Option<PageAnalysisMap>,
    pub four_point_rejection: Option<String>,
    pub gcs: GcsMap,
    pub gcs_shift: Point,
    pub plane_evidence: Option<PlaneEvidence>,
    pub staff_conditioning: crate::geometry::ConditioningReport,
}

impl PreGcs {
    pub fn forward(&self, p: Point) -> Point {
        // Detection takes place after H. Its physical camera rays are H^-1(q),
        // not rays through an invented new square-pixel camera. In the composed
        // source map H and this known inverse cancel analytically.
        let q = if self.is_plane() {
            self.frame.as_ref().unwrap().forward(p).unwrap_or(Point {
                x: f64::NAN,
                y: f64::NAN,
            })
        } else {
            self.gcs.forward(p)
        };
        Point {
            x: q.x - self.gcs_shift.x,
            y: q.y - self.gcs_shift.y,
        }
    }
    pub fn inverse(&self, p: Point) -> Point {
        let p = Point {
            x: p.x + self.gcs_shift.x,
            y: p.y + self.gcs_shift.y,
        };
        if self.is_plane() {
            self.frame.as_ref().unwrap().inverse(p).unwrap_or(Point {
                x: f64::NAN,
                y: f64::NAN,
            })
        } else {
            self.gcs.inverse(p)
        }
    }
    pub fn jacobian_det(&self, p: Point) -> f64 {
        if self.is_plane() {
            self.frame.as_ref().unwrap().jacobian_det(p)
        } else {
            self.gcs.jacobian_det(p)
        }
    }
    pub fn is_plane(&self) -> bool {
        self.plane_evidence.as_ref().is_some_and(|e| e.accepted)
    }
    pub fn report(&self) -> GcsPipelineReport {
        GcsPipelineReport {
            four_point: self.frame.clone(),
            four_point_rejection: self.four_point_rejection.clone(),
            camera_coordinates: "source-camera rays; known four-point homography compensated analytically",
            gcs_shift: self.gcs_shift,
            model: self.gcs.report().clone(),
            selected_model: if self.is_plane() {
                "page_plane"
            } else {
                "physical_gcs"
            },
            plane_evidence: self.plane_evidence.clone(),
            staff_conditioning: self.staff_conditioning.clone(),
        }
    }
}

/// A plane is a separate geometric hypothesis, selected from measured staff
/// and full-bar consistency after the known page homography, never on GCS error.
pub(crate) fn plane_evidence(data: &Observations, frame: &PageAnalysisMap) -> PlaneEvidence {
    let mut line_errors = vec![];
    let mut gaps = vec![];
    let mut variation = vec![];
    let mut bar_errors = vec![];
    for staff in &data.staves {
        let curves: [Vec<Point>; 5] = std::array::from_fn(|k| {
            staff
                .x
                .iter()
                .zip(&staff.y[k])
                .filter_map(|(x, y)| frame.forward(Point { x: *x, y: *y }))
                .collect()
        });
        if curves
            .iter()
            .any(|p| p.len() < 2 || p.windows(2).any(|w| w[1].x <= w[0].x))
        {
            continue;
        }
        for p in &curves {
            let mean = p.iter().map(|p| p.y).sum::<f64>() / p.len() as f64;
            line_errors.push(
                (p.iter().map(|p| (p.y - mean).powi(2)).sum::<f64>() / p.len() as f64).sqrt(),
            );
        }
        let lo = curves
            .iter()
            .map(|p| p[0].x)
            .fold(f64::NEG_INFINITY, f64::max);
        let hi = curves
            .iter()
            .map(|p| p.last().unwrap().x)
            .fold(f64::INFINITY, f64::min);
        if hi <= lo {
            continue;
        }
        let gg: Vec<_> = (0..33)
            .flat_map(|j| {
                let x = lo + (hi - lo) * j as f64 / 32.;
                (0..4)
                    .map(|k| interp(&curves[k + 1], x) - interp(&curves[k], x))
                    .collect::<Vec<_>>()
            })
            .collect();
        let mean = gg.iter().sum::<f64>() / gg.len() as f64;
        if mean > 0. {
            gaps.push(mean);
            variation.push(
                (gg.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / gg.len() as f64).sqrt()
                    / mean,
            );
        }
    }
    for bar in &data.barlines {
        let p: Vec<_> = bar
            .x
            .iter()
            .zip(&bar.y)
            .filter_map(|(x, y)| frame.forward(Point { x: *x, y: *y }))
            .collect();
        if p.len() < 2 {
            continue;
        }
        let mean = p.iter().map(|p| p.x).sum::<f64>() / p.len() as f64;
        bar_errors
            .push((p.iter().map(|p| (p.x - mean).powi(2)).sum::<f64>() / p.len() as f64).sqrt());
    }
    fn q90(v: &mut [f64]) -> f64 {
        if v.is_empty() {
            return f64::MAX;
        }
        v.sort_by(f64::total_cmp);
        v[((v.len() - 1) as f64 * 0.9).ceil() as usize]
    }
    let spacing = if gaps.is_empty() {
        0.
    } else {
        crate::math::median(gaps)
    };
    let tolerance = 0.25 + 0.01 * spacing;
    let line = q90(&mut line_errors);
    let bars = q90(&mut bar_errors);
    let gap = q90(&mut variation);
    let accepted = data.staves.len() >= 2
        && data.barlines.len() >= 2
        && line < tolerance
        && bars < tolerance
        && gap < 0.03;
    PlaneEvidence {
        accepted,
        staff_horizontal_rms_p90: line,
        bar_horizontal_displacement_rms_p90: bars,
        relative_gap_variation_p90: gap,
        noise_tolerance_pixels: tolerance,
        reason: if accepted {
            "known page homography explains staffs, spacing and bars within measurement noise"
        } else {
            "curvature, gap variation or insufficient plane evidence"
        },
    }
}

pub(crate) fn border(width: u32, height: u32) -> Vec<Point> {
    let mut out = Vec::with_capacity(404);
    for i in 0..=100 {
        let x = f64::from(width - 1) * i as f64 / 100.;
        let y = f64::from(height - 1) * i as f64 / 100.;
        out.extend([
            Point { x, y: 0. },
            Point {
                x,
                y: f64::from(height - 1),
            },
            Point { x: 0., y },
            Point {
                x: f64::from(width - 1),
                y,
            },
        ]);
    }
    out
}

fn interp(points: &[Point], x: f64) -> f64 {
    let i = points
        .partition_point(|p| p.x < x)
        .clamp(1, points.len() - 1);
    let (a, b) = (points[i - 1], points[i]);
    let t = ((x - a.x) / (b.x - a.x)).clamp(0., 1.);
    a.y * (1. - t) + b.y * t
}

/// Transform each of the five curves independently, then measure/regrid at
/// common output x. Preserve all original staff and line IDs and bar topology.
pub(crate) fn map_observations(
    data: &Observations,
    edges: &[StaffEdge],
    map: impl Fn(Point) -> Point,
) -> Result<(Observations, Vec<StaffEdge>, Point)> {
    let curves: Vec<[Vec<Point>; 5]> = data
        .staves
        .iter()
        .map(|s| {
            std::array::from_fn(|k| {
                s.x.iter()
                    .zip(&s.y[k])
                    .map(|(x, y)| map(Point { x: *x, y: *y }))
                    .collect()
            })
        })
        .collect();
    let bars: Vec<Vec<Point>> = data
        .barlines
        .iter()
        .map(|b| {
            b.x.iter()
                .zip(&b.y)
                .map(|(x, y)| map(Point { x: *x, y: *y }))
                .collect()
        })
        .collect();
    let mut mapped_edges = edges.to_vec();
    for e in &mut mapped_edges {
        e.points = e.points.iter().map(|p| map(*p)).collect();
    }
    let mut all = border(data.width, data.height)
        .into_iter()
        .map(&map)
        .collect::<Vec<_>>();
    all.extend(
        curves
            .iter()
            .flat_map(|s| s.iter().flat_map(|p| p.iter().copied())),
    );
    all.extend(bars.iter().flatten().copied());
    all.extend(mapped_edges.iter().flat_map(|e| e.points.iter().copied()));
    if all.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
        return Err(Error::Invalid("nonfinite pre-staff coordinate map"));
    }
    let shift = Point {
        x: all.iter().map(|p| p.x).fold(f64::INFINITY, f64::min) - 2.,
        y: all.iter().map(|p| p.y).fold(f64::INFINITY, f64::min) - 2.,
    };
    let width = (all.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max) - shift.x + 3.).ceil();
    let height = (all.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max) - shift.y + 3.).ceil();
    if width > 50000. || height > 50000. || width * height > 200_000_000. {
        return Err(Error::Invalid("pre-staff map expands page excessively"));
    }
    let mut staves = vec![];
    for s in curves {
        if s.iter()
            .any(|p| p.len() < 2 || p.windows(2).any(|w| w[1].x <= w[0].x))
        {
            return Err(Error::Invalid("pre-staff curves are not monotone in x"));
        }
        let lo = s.iter().map(|p| p[0].x).fold(f64::NEG_INFINITY, f64::max);
        let hi = s
            .iter()
            .map(|p| p.last().unwrap().x)
            .fold(f64::INFINITY, f64::min);
        if hi <= lo {
            return Err(Error::Invalid(
                "staff curves have no common transformed x domain",
            ));
        }
        let n = s[0].len().clamp(32, 256);
        let x: Vec<_> = (0..n)
            .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
            .collect();
        let y = std::array::from_fn(|k| x.iter().map(|x| interp(&s[k], *x) - shift.y).collect());
        staves.push(Staff {
            x: x.iter().map(|x| x - shift.x).collect(),
            y,
        });
    }
    let mut barlines = data.barlines.clone();
    for (b, p) in barlines.iter_mut().zip(bars) {
        b.x = p.iter().map(|p| p.x - shift.x).collect();
        b.y = p.iter().map(|p| p.y - shift.y).collect();
    }
    for e in &mut mapped_edges {
        for p in &mut e.points {
            p.x -= shift.x;
            p.y -= shift.y;
        }
    }
    let mapped = Observations {
        width: width as u32,
        height: height as u32,
        staves,
        barlines,
    };
    mapped.validate()?;
    Ok((mapped, mapped_edges, shift))
}

#[cfg(test)]
mod tests {
    use crate::geometry::{Barline, Config, Geometry, Observations, Point, Staff};
    #[test]
    fn unsupported_optional_corner_hypothesis_does_not_disable_valid_source_camera_fit() {
        let data = Observations {
            width: 300,
            height: 400,
            staves: [120., 240.]
                .iter()
                .map(|y| Staff {
                    x: vec![30., 270.],
                    y: std::array::from_fn(|k| vec![y + (k as f64 - 2.) * 8.; 2]),
                })
                .collect(),
            barlines: [50., 150., 250.]
                .iter()
                .map(|x| Barline {
                    staff: 0,
                    aligned_staffs: vec![1],
                    x: vec![*x; 2],
                    y: vec![104., 256.],
                })
                .collect(),
        };
        let corners = [
            Point { x: -0.5, y: 0. },
            Point { x: 299., y: 0. },
            Point { x: 299., y: 399. },
            Point { x: 0., y: 399. },
        ];
        let geometry = Geometry::fit_profiled_gcs(
            &data,
            &Config::default(),
            Some(corners),
            Some(399. / 299.),
            &[],
            true,
        )
        .0
        .unwrap();
        let report = geometry.report().gcs.as_ref().unwrap();
        assert!(report.four_point.is_none());
        assert!(
            report
                .four_point_rejection
                .as_ref()
                .unwrap()
                .contains("outside source crop")
        );
        assert_eq!(report.selected_model, "physical_gcs");
        assert!(geometry.report().validation.sampled_min_jacobian > 0.);
    }
}
