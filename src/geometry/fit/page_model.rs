//! Learned page geometry followed by optional music residuals.
use super::*;
use crate::geometry::{PageAnalysisMap, PerspectiveReport, VerticalPath};

pub(crate) enum PageResidualModel {
    Mesh,
    Gcs { vertical: bool },
    Staff { vertical: bool },
}

impl Geometry {
    fn identity(width: u32, height: u32, config: &Config) -> Self {
        let data = Observations {
            width,
            height,
            staves: vec![],
            barlines: vec![],
        };
        let bounds = [0., 0., f64::from(width - 1), f64::from(height - 1)];
        Self {
            pre_page: None,
            pre_frame: None,
            pre_gcs: None,
            text: None,
            projective: Projective::identity(&data),
            direction_mesh: crate::geometry::DirectionMesh::default(),
            flat: Flatten {
                local: crate::geometry::corrections::local_flatten::LocalFlatten::identity(bounds),
                targets: vec![0., bounds[3]],
                min_y: 0.,
                max_y: bounds[3],
            },
            horizontal: system_shear::Shear::identity(0., bounds[3]),
            vertical: crate::geometry::VerticalCorrection::default(),
            report: Report {
                schema: "smude.geometry.v1",
                input_width: width,
                input_height: height,
                page_model: None,
                direction_mesh: None,
                gcs: None,
                text_warp: None,
                config: config.clone(),
                bounds,
                staff_edges: vec![],
                staff_extensions: Default::default(),
                measured_staff_spacings: vec![],
                target_staff_spacings: vec![],
                systems: vec![],
                vertical_correction: None,
                validation: Validation {
                    sampled_min_jacobian: 1.,
                    roundtrip_max_pixels: 0.,
                    final_stage_y_displacement: 0.,
                    max_staff_line_span_pixels: 0.,
                    bar_median_angle_degrees: 0.,
                    bar_p90_angle_degrees: 0.,
                    bar_max_bend_pixels: 0.,
                },
                perspective: PerspectiveReport {
                    source: "learned_page_grid",
                    enabled: true,
                    parameters: [0.; 4],
                    page_frame_rejection: None,
                    converged: true,
                    iterations: 0,
                    at_parameter_bound: false,
                    heldout_x_fraction: [0.; 2],
                    before_heldout_spacing_rms: 0.,
                    after_heldout_spacing_rms: 0.,
                    metric_assumption: "Predicted interior page grid and physical page aspect; no music observations required.",
                },
                validation_note: "Learned page geometry; no invented music measurements.",
                extrapolation: ExtrapolationReport {
                    model: "identity_residual",
                    support_limit_staff_spaces: 0.,
                    max_boundary_jump_pixels: 0.,
                    adjusted_anchor_samples: 0,
                    max_adjustment_pixels: 0.,
                    changed_x_bounds: None,
                    min_source_gap_pixels: 0.,
                    max_vertical_scale: 1.,
                },
            },
        }
    }

    pub(crate) fn fit_profiled_on_page(
        data: &Observations,
        config: &Config,
        page: &PageAnalysisMap,
        edges: &[StaffEdge],
        verticals: &[VerticalPath],
        model: PageResidualModel,
    ) -> (Result<Self>, FitProfile) {
        let started = Instant::now();
        let mut profile = FitProfile::default();
        let result = (|| {
            if page.source_dimensions() != (data.width, data.height) {
                return Err(Error::Invalid("page model source dimensions differ"));
            }
            let (mut geometry, shift) = if data.staves.is_empty() {
                let (w, h) = page.dimensions();
                (Self::identity(w, h, config), Point { x: 0., y: 0. })
            } else {
                let stage = Instant::now();
                let (mapped, mapped_edges, shift) =
                    crate::geometry::composition::map_observations(data, edges, |p| {
                        page.forward(p).unwrap_or(Point {
                            x: f64::NAN,
                            y: f64::NAN,
                        })
                    })?;
                let mut paths = verticals.to_vec();
                for path in &mut paths {
                    for p in &mut path.points {
                        let q = page
                            .forward(*p)
                            .ok_or(Error::Invalid("invalid learned vertical map"))?;
                        *p = Point {
                            x: q.x - shift.x,
                            y: q.y - shift.y,
                        };
                    }
                }
                let mapping_seconds = stage.elapsed().as_secs_f64();
                let (result, fit) = match model {
                    PageResidualModel::Mesh => Self::fit_profiled_direction_mesh(
                        &mapped,
                        config,
                        None,
                        None,
                        &mapped_edges,
                        &paths,
                    ),
                    PageResidualModel::Gcs { vertical } => Self::fit_profiled_gcs_with_evidence(
                        &mapped,
                        config,
                        &GcsFitEvidence {
                            edges: &mapped_edges,
                            vertical_paths: &paths,
                            vertical_correction: vertical,
                            ..Default::default()
                        },
                    ),
                    PageResidualModel::Staff { vertical } => {
                        let (r, mut p) = Self::fit_profiled_with_staff_edges(
                            &mapped,
                            config,
                            None,
                            &mapped_edges,
                        );
                        let r = r.and_then(|mut g| {
                            if vertical {
                                let t = Instant::now();
                                g.add_vertical_correction(&mapped)?;
                                p.vertical_correction_seconds += t.elapsed().as_secs_f64();
                            }
                            Ok(g)
                        });
                        (r, p)
                    }
                };
                profile = fit;
                profile.perspective_seconds += mapping_seconds;
                (result?, shift)
            };
            geometry.pre_page = Some((page.clone(), shift));
            geometry.report.page_model = Some(page.clone());
            geometry.report.input_width = data.width;
            geometry.report.input_height = data.height;
            geometry.report.staff_edges = edges.to_vec();
            geometry.validate(data)?;
            Ok(geometry)
        })();
        profile.total_seconds = started.elapsed().as_secs_f64();
        if result.is_ok() {
            profile.failure_stage = None;
        }
        (result, profile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn page_without_staves_keeps_reversible_geometry_and_no_music_anchors() {
        let points = (0..25)
            .flat_map(|j| {
                (0..25).map(move |i| Point {
                    x: 40. + i as f64 * 20. + (j as f64 * 0.1).sin() * 10.,
                    y: 30. + j as f64 * 30.,
                })
            })
            .collect();
        let page = PageAnalysisMap::from_grid(600, 800, points, 1.4).unwrap();
        let data = Observations {
            width: 600,
            height: 800,
            staves: vec![],
            barlines: vec![],
        };
        let (g, _) = Geometry::fit_profiled_on_page(
            &data,
            &Config::default(),
            &page,
            &[],
            &[],
            PageResidualModel::Mesh,
        );
        let g = g.unwrap();
        assert!(g.report().target_staff_spacings.is_empty());
        for j in 1..20 {
            for i in 1..20 {
                let p = page
                    .inverse(Point {
                        x: i as f64 * 20.,
                        y: j as f64 * 25.,
                    })
                    .unwrap();
                let q = g.inverse(g.forward(p));
                assert!((q.x - p.x).hypot(q.y - p.y) < 1e-5);
                let f = g.forward(p);
                let back = g.inverse_scanline(f.y).at(f.x, &mut Nearby::default());
                assert!((back.x - p.x).hypot(back.y - p.y) < 1e-5);
            }
        }
    }
}
