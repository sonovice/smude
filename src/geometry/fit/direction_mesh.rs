//! GCS-free page frame -> five-line flattening -> common direction mesh.
use super::Geometry;
use crate::geometry::{
    Config, DirectionMesh, DirectionMeshReport, FitProfile, Observations, PageAnalysisMap, Point,
    StaffEdge, VerticalPath,
};
use crate::{Error, Result};
use serde::Serialize;
use std::time::Instant;

#[derive(Clone, Debug, Serialize)]
pub struct MeshPipelineReport {
    pub model: &'static str,
    pub four_point: Option<PageAnalysisMap>,
    pub frame_shift: Point,
    pub staff_conditioning: crate::geometry::ConditioningReport,
    pub field: DirectionMeshReport,
    pub metric_assumption: &'static str,
}
impl Geometry {
    pub(crate) fn refine_surface_metric(
        &mut self,
        data: &Observations,
        edges: &[StaffEdge],
        verticals: &[VerticalPath],
        principal: Point,
        focal: f64,
    ) -> crate::geometry::corrections::surface_metric::SurfaceMetricReport {
        use crate::geometry::corrections::surface_metric;
        let old = self.direction_mesh.clone();
        let old_bounds = self.report.bounds;
        self.direction_mesh = DirectionMesh::default();
        let (samples, mut report) =
            surface_metric::measure(data, &|p| self.forward(p), principal, focal);
        if samples.is_empty() {
            self.direction_mesh = old;
            return report;
        }
        let candidate = DirectionMesh::fit_with_metric(
            data,
            &|p| self.forward(p),
            edges,
            verticals,
            old.report().bounds,
            &samples,
            Some(&old),
        );
        let error = |mesh: &DirectionMesh| {
            (samples
                .iter()
                .map(|s| s.confidence * (mesh.jacobian_det(s.point) / s.scale).ln().powi(2))
                .sum::<f64>()
                / samples.iter().map(|s| s.confidence).sum::<f64>())
            .sqrt()
        };
        report.before_relative_scale_rms = error(&old);
        report.after_relative_scale_rms = error(&candidate);
        report.before_direction_rms_degrees = old.report().after_rms_direction_degrees;
        report.after_direction_rms_degrees = candidate.report().after_rms_direction_degrees;
        if report.after_relative_scale_rms < report.before_relative_scale_rms * 0.95
            && candidate.report().after_rms_direction_degrees
                <= old.report().after_rms_direction_degrees + 0.15
        {
            self.direction_mesh = candidate;
            let border = crate::geometry::composition::border(data.width, data.height);
            let points: Vec<_> = border.iter().map(|p| self.forward(*p)).collect();
            self.report.bounds = [
                points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min),
                points.iter().map(|p| p.y).fold(f64::INFINITY, f64::min),
                points.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max),
                points.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max),
            ];
            if self.validate(data).is_ok() {
                report.accepted = true;
                if let Some(r) = &mut self.report.direction_mesh {
                    r.field = self.direction_mesh.report().clone();
                    r.metric_assumption = "Relative width uses orthogonal staff/bar directions and an evidence-gated common vertical ruling with an approximate camera focal prior. This is bounded metric evidence, not unique reconstruction of arbitrary 3-D paper.";
                }
                return report;
            }
        }
        self.direction_mesh = old;
        self.report.bounds = old_bounds;
        report.rejection = Some("metric_gain_or_direction_safety_check_failed");
        report
    }
    /// Replace only the common x-direction field using additional source-space
    /// measurements. Keep the first pass page frame and staff y warp fixed.
    pub(crate) fn refine_direction_mesh(
        &mut self,
        data: &Observations,
        edges: &[StaffEdge],
        verticals: &[VerticalPath],
    ) -> Result<()> {
        self.direction_mesh = DirectionMesh::default();
        let border = crate::geometry::composition::border(data.width, data.height);
        let bounds = |geometry: &Self| {
            let p: Vec<_> = border.iter().map(|p| geometry.forward(*p)).collect();
            [
                p.iter().map(|p| p.x).fold(f64::INFINITY, f64::min),
                p.iter().map(|p| p.y).fold(f64::INFINITY, f64::min),
                p.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max),
                p.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max),
            ]
        };
        self.direction_mesh =
            DirectionMesh::fit(data, &|p| self.forward(p), edges, verticals, bounds(self));
        if let Some(report) = &mut self.report.direction_mesh {
            report.field = self.direction_mesh.report().clone();
        }
        self.report.bounds = bounds(self);
        self.validate(data)
    }

    /// Experimental alternative to GCS. Curves are compared at common mapped
    /// x; five-line spacing informs projective fit and vertical flattening.
    /// The last stage fits only x, shared over the entire page.
    pub fn fit_profiled_direction_mesh(
        data: &Observations,
        config: &Config,
        corners: Option<[Point; 4]>,
        ratio: Option<f64>,
        edges: &[StaffEdge],
        verticals: &[VerticalPath],
    ) -> (Result<Self>, FitProfile) {
        let started = Instant::now();
        let mut profile = FitProfile::default();
        let result = (|| {
            profile.failure_stage = Some("mesh_input_validation");
            let stage = Instant::now();
            data.validate()?;
            if verticals.iter().any(|v| {
                v.points.len() < 3
                    || !v.weight.is_finite()
                    || !(0.0..=1.0).contains(&v.weight)
                    || v.points.windows(2).any(|p| p[1].y <= p[0].y)
                    || v.points.iter().any(|p| {
                        !p.x.is_finite()
                            || !p.y.is_finite()
                            || p.x < 0.
                            || p.y < 0.
                            || p.x > (data.width - 1) as f64
                            || p.y > (data.height - 1) as f64
                    })
            }) {
                return Err(Error::Invalid("invalid mesh vertical observations"));
            }
            profile.input_validation_seconds = stage.elapsed().as_secs_f64();
            profile.failure_stage = Some("mesh_page_frame");
            let stage = Instant::now();
            let frame = if config.perspective {
                corners
                    .zip(ratio)
                    .map(|(c, r)| PageAnalysisMap::new(data.width, data.height, c, r, data.width))
                    .transpose()?
            } else {
                None
            };
            let (mapped, mapped_edges, shift) = if let Some(frame) = &frame {
                crate::geometry::composition::map_observations(data, edges, |p| {
                    frame.forward(p).unwrap_or(Point {
                        x: f64::NAN,
                        y: f64::NAN,
                    })
                })?
            } else {
                (data.clone(), edges.to_vec(), Point { x: 0., y: 0. })
            };
            profile.perspective_seconds = stage.elapsed().as_secs_f64();
            let stage = Instant::now();
            let (conditioned, conditioning) =
                crate::geometry::condition_staff_observations(&mapped);
            profile.staff_conditioning_seconds = stage.elapsed().as_secs_f64();
            // After the page frame, remaining gap changes still constrain both
            // projective denominators. No 3-D cylindrical surface is fitted.
            let (base, local) =
                Self::fit_profiled_with_staff_edges(&conditioned, config, None, &mapped_edges);
            profile.perspective_seconds += local.perspective_seconds;
            profile.staff_warp_seconds = local.staff_warp_seconds;
            profile.system_shear_seconds = local.system_shear_seconds;
            profile.map_validation_seconds = local.map_validation_seconds;
            let mut geometry = base?;
            geometry.pre_frame = frame.clone().map(|f| (f, shift));
            geometry.report.input_width = data.width;
            geometry.report.input_height = data.height;
            geometry.report.staff_edges = edges.to_vec();
            let border = crate::geometry::composition::border(data.width, data.height);
            let bounds = |geometry: &Self| {
                let p: Vec<_> = border.iter().map(|p| geometry.forward(*p)).collect();
                [
                    p.iter().map(|p| p.x).fold(f64::INFINITY, f64::min),
                    p.iter().map(|p| p.y).fold(f64::INFINITY, f64::min),
                    p.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max),
                    p.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max),
                ]
            };
            geometry.report.bounds = bounds(&geometry);
            profile.failure_stage = Some("direction_mesh");
            let stage = Instant::now();
            geometry.direction_mesh = DirectionMesh::fit(
                data,
                &|p| geometry.forward(p),
                edges,
                verticals,
                geometry.report.bounds,
            );
            profile.direction_mesh_seconds = stage.elapsed().as_secs_f64();
            geometry.report.direction_mesh = Some(MeshPipelineReport {
                model: "page_frame_staff_spacing_direction_mesh_no_gcs",
                four_point: frame,
                frame_shift: shift,
                staff_conditioning: conditioning,
                field: geometry.direction_mesh.report().clone(),
                metric_assumption: "Five-line groups retain constant spacing, independently per staff unless sizes are explicitly grouped. Perspective uses spacing across x and y; residual mesh uses only a weak relative scale prior. Ratio anchors page aspect; arbitrary horizontal metric is not uniquely recovered.",
            });
            geometry.report.bounds = bounds(&geometry);
            geometry.report.validation_note = "GCS-free frame/staff/direction mesh. Final mesh fixes y exactly; control differences bound positive x derivative globally. Staff/bar statistics are observation diagnostics, not independent ground truth.";
            profile.failure_stage = Some("mesh_composed_validation");
            let stage = Instant::now();
            geometry.validate(data)?;
            profile.map_validation_seconds += stage.elapsed().as_secs_f64();
            Ok(geometry)
        })();
        profile.total_seconds = started.elapsed().as_secs_f64();
        profile.unattributed_seconds = (profile.total_seconds
            - profile.input_validation_seconds
            - profile.perspective_seconds
            - profile.staff_conditioning_seconds
            - profile.staff_warp_seconds
            - profile.system_shear_seconds
            - profile.map_validation_seconds
            - profile.direction_mesh_seconds)
            .max(0.);
        if result.is_ok() {
            profile.failure_stage = None;
        }
        (result, profile)
    }
}

#[cfg(test)]
mod metric_tests {
    use super::*;
    use crate::geometry::{Barline, Staff};
    fn score(
        curved: bool,
    ) -> (
        Observations,
        impl Fn(f64, f64) -> Point,
        impl Fn(f64) -> f64,
    ) {
        let depth = move |u: f64| {
            if curved {
                140. * (-(u + 450.) / 170.).exp()
            } else {
                0.
            }
        };
        let project = move |u: f64, v: f64| Point {
            x: 600. + 1200. * u / (1200. + depth(u)),
            y: 1000. + 1200. * v / (1200. + depth(u)),
        };
        let staves: Vec<_> = [-650., -400., -150., 200., 450., 700.]
            .into_iter()
            .map(|v| {
                let p: Vec<_> = (0..241)
                    .map(|i| project(-450. + 900. * i as f64 / 240., v))
                    .collect();
                Staff {
                    x: p.iter().map(|p| p.x).collect(),
                    y: std::array::from_fn(|k| {
                        (0..241)
                            .map(|i| {
                                project(-450. + 900. * i as f64 / 240., v + (k as f64 - 2.) * 12.).y
                            })
                            .collect()
                    }),
                }
            })
            .collect();
        let mut barlines = vec![];
        for (staff, v) in [-650., -400., -150., 200., 450., 700.]
            .into_iter()
            .enumerate()
        {
            for j in 0..13 {
                let u = -450. + 75. * j as f64;
                let p: Vec<_> = (0..9)
                    .map(|k| project(u, v - 24. + 6. * k as f64))
                    .collect();
                barlines.push(Barline {
                    staff,
                    x: p.iter().map(|p| p.x).collect(),
                    y: p.iter().map(|p| p.y).collect(),
                    aligned_staffs: vec![],
                });
            }
        }
        (
            Observations {
                width: 1200,
                height: 2000,
                staves,
                barlines,
            },
            project,
            move |u| {
                if curved {
                    (-140. / 170.) * (-(u + 450.) / 170.).exp()
                } else {
                    0.
                }
            },
        )
    }
    #[test]
    fn folded_print_recovers_relative_width_without_changing_staff_heights() {
        let (data, project, slope) = score(true);
        let (base, _) = Geometry::fit_profiled_direction_mesh(
            &data,
            &Config {
                perspective: false,
                ..Default::default()
            },
            None,
            None,
            &[],
            &[],
        );
        let base = base.unwrap();
        let mut fixed = base.clone();
        let width_error = |g: &Geometry| {
            let v = 450.;
            let gap = g.forward(project(0., v + 12.)).y - g.forward(project(0., v)).y;
            [-400., -350., -300.]
                .into_iter()
                .map(|u| {
                    let width =
                        (g.forward(project(u + 1., v)).x - g.forward(project(u - 1., v)).x) / 2.;
                    (width / gap * 12. / (1. + slope(u).powi(2)).sqrt() - 1.).abs()
                })
                .sum::<f64>()
                / 3.
        };
        let before = width_error(&base);
        let report =
            fixed.refine_surface_metric(&data, &[], &[], Point { x: 600., y: 1000. }, 1200.);
        assert!(report.accepted, "{report:?}");
        let after = width_error(&fixed);
        assert!(
            after < before * 0.65,
            "metric width error {before} -> {after}; {report:?}"
        );
        for staff in &data.staves {
            for p in staff.sample(0.2) {
                assert_eq!(base.forward(p).y, fixed.forward(p).y);
            }
        }
        assert!(fixed.report().validation.roundtrip_max_pixels < 1e-4);
        assert!(fixed.report().validation.bar_p90_angle_degrees < 0.1);
        assert!(fixed.report().validation.bar_max_bend_pixels < 0.1);
    }
    #[test]
    fn frontal_print_does_not_receive_a_metric_warp() {
        let (data, _, _) = score(false);
        let (mut base, _) = Geometry::fit_profiled_direction_mesh(
            &data,
            &Config {
                perspective: false,
                ..Default::default()
            },
            None,
            None,
            &[],
            &[],
        );
        let base = base.as_mut().unwrap();
        let report =
            base.refine_surface_metric(&data, &[], &[], Point { x: 600., y: 1000. }, 1200.);
        assert!(!report.accepted, "{report:?}");
    }
    #[test]
    fn contradictory_vertical_rulings_do_not_define_a_surface_metric() {
        let (mut data, _, _) = score(true);
        for (i, bar) in data.barlines.iter_mut().enumerate() {
            let y = bar.y.iter().sum::<f64>() / bar.y.len() as f64;
            let slope = if i % 2 == 0 { 0.2 } else { -0.2 };
            for (x, py) in bar.x.iter_mut().zip(&bar.y) {
                *x += slope * (py - y);
            }
        }
        let (samples, report) = crate::geometry::corrections::surface_metric::measure(
            &data,
            &|p| p,
            Point { x: 600., y: 1000. },
            1200.,
        );
        assert!(samples.is_empty());
        assert_eq!(
            report.rejection,
            Some("vertical_rulings_are_not_consistent")
        );
    }
}
