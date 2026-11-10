//! Fit and compose the reversible geometry; rasterization lives in rendering/.
#[cfg(feature = "render")]
mod direction_mesh;
#[cfg(feature = "render")]
mod page_model;
#[cfg(feature = "render")]
pub(crate) use page_model::PageResidualModel;
mod system_shear;
use super::report::{ExtrapolationReport, FitProfile, Report, Validation};
#[cfg(feature = "render")]
pub use direction_mesh::MeshPipelineReport;
pub(crate) use system_shear::systems;
use system_shear::{Shear, shear};

use crate::math::{median, slope};
use std::time::Instant;

use crate::geometry::models::perspective::{self, Projective, joint_staff};
use crate::geometry::observations::interpolate;
use crate::geometry::{Config, Observations, Point, StaffEdge};
use crate::{Error, Result};

#[derive(Clone, Debug)]
struct Curve {
    x: std::sync::Arc<[f64]>,
    y: Vec<f64>,
}

#[derive(Clone, Debug)]
struct Flatten {
    local: crate::geometry::corrections::local_flatten::LocalFlatten,
    targets: Vec<f64>,
    min_y: f64,
    max_y: f64,
}
impl Flatten {
    fn forward(&self, p: Point) -> Point {
        self.local.forward(p)
    }
    fn forward_near(&self, p: Point, nearby: &mut Nearby) -> Point {
        self.local.forward_near(p, &mut nearby.staff_band)
    }
    fn inverse(&self, p: Point) -> Point {
        self.local.inverse(p)
    }
    #[cfg(feature = "render")]
    fn inverse_near(&self, p: Point, nearby: &mut Nearby) -> Point {
        self.local.inverse_near(p, &mut nearby.staff_band)
    }
}

/// What a previous sample of the map found, to shorten the searches of the
/// next sample close to it. Any state yields the same coordinates.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Nearby {
    staff_band: crate::geometry::corrections::local_flatten::Hint,
}

/// One output row of [`Geometry::inverse`] with its exactly y-dependent
/// stages hoisted out of the pixel loop.
#[cfg(feature = "render")]
pub(crate) struct Scanline<'a> {
    geometry: &'a Geometry,
    y: f64,
    /// Only nonzero text spans need the general inverse, since text can
    /// change y locally.
    text: Vec<[f64; 2]>,
    vertical: Option<(f64, f64, f64)>,
    mesh: Option<crate::geometry::corrections::MeshRow>,
    shear: f64,
}
#[cfg(feature = "render")]
impl Scanline<'_> {
    #[cfg(test)]
    pub(crate) fn at(&self, x: f64, nearby: &mut Nearby) -> Point {
        let mut p = [Point { x, y: self.y }];
        self.fill(&[x], &mut p, nearby);
        p[0]
    }
    /// The source point of every x in this row. A sample is one long
    /// dependency chain through all stages, so a batch of samples passes
    /// through one stage at a time and their chains overlap.
    pub(crate) fn fill(&self, xs: &[f64], out: &mut [Point], nearby: &mut Nearby) {
        const BATCH: usize = 32;
        let (geometry, y) = (self.geometry, self.y);
        let in_text = |x: &f64| self.text.iter().any(|span| *x >= span[0] && *x <= span[1]);
        for (xs, out) in xs.chunks(BATCH).zip(out.chunks_mut(BATCH)) {
            if !xs.iter().all(in_text) {
                let mut flat = [0.; BATCH];
                let flat = &mut flat[..xs.len()];
                flat.copy_from_slice(xs);
                if let Some(mesh) = &self.mesh {
                    mesh.inverse_xs(flat);
                }
                for (x, out) in flat.iter().zip(out.iter_mut()) {
                    let x = self
                        .vertical
                        .map_or(*x, |(center, scale, a)| center + (*x - center) * scale + a);
                    let flat = Point {
                        x: x - self.shear,
                        y,
                    };
                    *out = geometry.flat.inverse_near(flat, nearby);
                }
                for p in out.iter_mut() {
                    *p = geometry.projective.inverse(*p);
                }
                for p in out.iter_mut() {
                    *p = geometry.inverse_page(*p);
                }
            }
            for (x, out) in xs.iter().zip(out.iter_mut()).filter(|(x, _)| in_text(x)) {
                *out = geometry.inverse(Point { x: *x, y });
            }
        }
    }
}

/// Fitted, reversible geometry. All maps use pixel coordinates and (x,y) order.
/// No detector/model objects or intermediate raster images are retained.
#[derive(Clone, Debug)]
pub struct Geometry {
    #[cfg(feature = "render")]
    pre_page: Option<(crate::geometry::PageAnalysisMap, Point)>,
    projective: Projective,
    direction_mesh: crate::geometry::DirectionMesh,
    #[cfg(feature = "render")]
    pre_frame: Option<(crate::geometry::PageAnalysisMap, Point)>,
    flat: Flatten,
    horizontal: Shear,
    vertical: crate::geometry::VerticalCorrection,
    #[cfg(feature = "render")]
    pre_gcs: Option<crate::geometry::composition::PreGcs>,
    report: Report,
    #[cfg(feature = "render")]
    text: Option<crate::geometry::corrections::text_warp::TextWarp>,
}

/// Optional source-coordinate evidence for the native GCS pipeline.
#[cfg(feature = "render")]
#[derive(Default)]
pub struct GcsFitEvidence<'a> {
    pub corners: Option<[Point; 4]>,
    pub ratio: Option<f64>,
    pub edges: &'a [StaffEdge],
    pub vertical_correction: bool,
    pub camera_center: Option<Point>,
    pub vertical_paths: &'a [crate::geometry::VerticalPath],
}

impl Geometry {
    /// Native full pipeline. The four-point and GCS maps are composed into the
    /// same source-pixel inverse used by text analysis and final reconstruction.
    #[cfg(feature = "render")]
    pub fn fit_profiled_gcs(
        data: &Observations,
        config: &Config,
        corners: Option<[Point; 4]>,
        ratio: Option<f64>,
        edges: &[StaffEdge],
        vertical_correction: bool,
    ) -> (Result<Self>, FitProfile) {
        Self::fit_profiled_gcs_with_camera(
            data,
            config,
            corners,
            ratio,
            edges,
            vertical_correction,
            None,
        )
    }

    #[cfg(feature = "render")]
    pub fn fit_profiled_gcs_with_camera(
        data: &Observations,
        config: &Config,
        corners: Option<[Point; 4]>,
        ratio: Option<f64>,
        edges: &[StaffEdge],
        vertical_correction: bool,
        camera_center: Option<Point>,
    ) -> (Result<Self>, FitProfile) {
        Self::fit_profiled_gcs_with_evidence(
            data,
            config,
            &GcsFitEvidence {
                corners,
                ratio,
                edges,
                vertical_correction,
                camera_center,
                vertical_paths: &[],
            },
        )
    }

    /// Experimental additional shafts share the global GCS surface; they do
    /// not create a staff-local correction or prescribe a source direction.
    #[cfg(feature = "render")]
    pub fn fit_profiled_gcs_with_evidence(
        data: &Observations,
        config: &Config,
        evidence: &GcsFitEvidence<'_>,
    ) -> (Result<Self>, FitProfile) {
        let GcsFitEvidence {
            corners,
            ratio,
            edges,
            vertical_correction,
            camera_center,
            vertical_paths,
        } = *evidence;
        let started = Instant::now();
        let mut profile = FitProfile::default();
        let result = (|| {
            profile.failure_stage = Some("input_validation");
            let stage = Instant::now();
            data.validate()?;
            profile.input_validation_seconds = stage.elapsed().as_secs_f64();
            profile.failure_stage = Some("perspective");
            let stage = Instant::now();
            // A measured quadrilateral is an optional camera hypothesis. An
            // unsupported contour must not prevent fitting valid source rays.
            // Keep the rejection explicit and do not invent/clamp its corners.
            let (frame, four_point_rejection) = match corners.zip(ratio) {
                Some((c, r)) => {
                    match crate::geometry::PageAnalysisMap::new(
                        data.width,
                        data.height,
                        c,
                        r,
                        data.width,
                    ) {
                        Ok(map) => (Some(map), None),
                        Err(error) => (None, Some(error.to_string())),
                    }
                }
                None => (None, None),
            };
            profile.perspective_seconds = stage.elapsed().as_secs_f64();
            profile.failure_stage = Some("gcs");
            let stage = Instant::now();
            let gcs = crate::geometry::GcsMap::fit_with_verticals(
                data,
                edges,
                ratio,
                frame.as_ref().and(corners),
                camera_center,
                vertical_paths,
            )?;
            let plane_evidence = frame
                .as_ref()
                .map(|m| crate::geometry::composition::plane_evidence(data, m));
            let use_plane = plane_evidence.as_ref().is_some_and(|e| e.accepted);
            let (mapped, mapped_edges, gcs_shift) =
                crate::geometry::composition::map_observations(data, edges, |p| {
                    if use_plane {
                        frame.as_ref().unwrap().forward(p).unwrap_or(Point {
                            x: f64::NAN,
                            y: f64::NAN,
                        })
                    } else {
                        gcs.forward(p)
                    }
                })?;
            profile.gcs_seconds = stage.elapsed().as_secs_f64();
            profile.failure_stage = Some("staff_conditioning");
            let stage = Instant::now();
            let (conditioned, staff_conditioning) =
                crate::geometry::condition_staff_observations(&mapped);
            profile.staff_conditioning_seconds = stage.elapsed().as_secs_f64();
            let local_config = Config {
                perspective: false,
                system_shear: config.system_shear,
                staff_size_groups: config.staff_size_groups.clone(),
                size_prior: config.size_prior,
            };
            let (geometry, local) = Self::fit_profiled_with_staff_edges(
                &conditioned,
                &local_config,
                None,
                &mapped_edges,
            );
            profile.input_validation_seconds += local.input_validation_seconds;
            profile.staff_warp_seconds = local.staff_warp_seconds;
            profile.system_shear_seconds = local.system_shear_seconds;
            profile.map_validation_seconds = local.map_validation_seconds;
            if geometry.is_err() {
                profile.failure_stage = local.failure_stage;
            }
            let mut geometry = geometry?;
            let pre = crate::geometry::composition::PreGcs {
                frame,
                four_point_rejection,
                gcs,
                gcs_shift,
                plane_evidence,
                staff_conditioning,
            };
            geometry.report.gcs = Some(pre.report());
            geometry.pre_gcs = Some(pre);
            geometry.report.input_width = data.width;
            geometry.report.input_height = data.height;
            geometry.report.config = config.clone();
            geometry.report.staff_edges = edges.to_vec();
            geometry.report.validation_note = "Native source-camera GCS/page-plane with noise-conditioned local staff anchors and bounded x correction. Validation uses untouched detector observations, not independent ground truth.";
            geometry.report.perspective.enabled = true;
            geometry.report.perspective.source = if use_plane {
                "page_boundary"
            } else {
                "native_gcs"
            };
            let border: Vec<_> = crate::geometry::composition::border(data.width, data.height)
                .into_iter()
                .map(|p| geometry.forward(p))
                .collect();
            geometry.report.bounds = [
                border.iter().map(|p| p.x).fold(f64::INFINITY, f64::min),
                border.iter().map(|p| p.y).fold(f64::INFINITY, f64::min),
                border.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max),
                border.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max),
            ];
            profile.failure_stage = Some("composed_map_validation");
            let stage = Instant::now();
            geometry.validate(data)?;
            profile.map_validation_seconds += stage.elapsed().as_secs_f64();
            if vertical_correction {
                profile.failure_stage = Some("vertical_correction");
                let stage = Instant::now();
                geometry.add_vertical_correction(data)?;
                profile.vertical_correction_seconds = stage.elapsed().as_secs_f64();
            }
            Ok(geometry)
        })();
        profile.total_seconds = started.elapsed().as_secs_f64();
        profile.unattributed_seconds = (profile.total_seconds
            - profile.input_validation_seconds
            - profile.perspective_seconds
            - profile.gcs_seconds
            - profile.staff_conditioning_seconds
            - profile.staff_warp_seconds
            - profile.system_shear_seconds
            - profile.map_validation_seconds
            - profile.vertical_correction_seconds)
            .max(0.);
        if result.is_ok() {
            profile.failure_stage = None;
        }
        (result, profile)
    }
    pub fn fit(data: &Observations, config: &Config) -> Result<Self> {
        Self::fit_profiled(data, config).0
    }

    /// Fit with the same numerical operations as `fit`, recording stage timings.
    pub fn fit_profiled(data: &Observations, config: &Config) -> (Result<Self>, FitProfile) {
        Self::fit_profiled_with_page_frame(data, config, None)
    }

    /// Optional broad paper-frame hypothesis, composed into the same inverse map.
    pub fn fit_profiled_with_page_frame(
        data: &Observations,
        config: &Config,
        corners: Option<[Point; 4]>,
    ) -> (Result<Self>, FitProfile) {
        Self::fit_profiled_with_staff_edges(data, config, corners, &[])
    }

    pub fn fit_profiled_with_staff_edges(
        data: &Observations,
        config: &Config,
        corners: Option<[Point; 4]>,
        edges: &[StaffEdge],
    ) -> (Result<Self>, FitProfile) {
        let started = Instant::now();
        let mut profile = FitProfile::default();
        let result = Self::fit_recorded(data, config, corners, edges, &mut profile);
        profile.total_seconds = started.elapsed().as_secs_f64();
        let completed = profile.input_validation_seconds
            + profile.perspective_seconds
            + profile.staff_warp_seconds
            + profile.system_shear_seconds
            + profile.gcs_seconds
            + profile.vertical_correction_seconds
            + profile.staff_conditioning_seconds
            + profile.map_validation_seconds;
        profile.unattributed_seconds = (profile.total_seconds - completed).max(0.);
        if result.is_ok() {
            profile.failure_stage = None;
        }
        (result, profile)
    }

    fn fit_recorded(
        data: &Observations,
        config: &Config,
        corners: Option<[Point; 4]>,
        edges: &[StaffEdge],
        profile: &mut FitProfile,
    ) -> Result<Self> {
        profile.failure_stage = Some("input_validation");
        let stage = Instant::now();
        data.validate()?;
        for edge in edges {
            if edge.staff >= data.staves.len()
                || edge.points.len() < 3
                || edge.points.len() != edge.line_indices.len()
                || !edge.confidence.is_finite()
                || !(0.0..=1.0).contains(&edge.confidence)
                || edge.line_indices.iter().any(|i| *i >= 5)
                || edge.line_indices.windows(2).any(|p| p[0] >= p[1])
                || edge
                    .aligned_staffs
                    .iter()
                    .any(|i| *i >= data.staves.len() || *i == edge.staff)
                || edge.points.iter().any(|p| {
                    !p.x.is_finite()
                        || !p.y.is_finite()
                        || p.x < 0.
                        || p.y < 0.
                        || p.x > f64::from(data.width - 1)
                        || p.y > f64::from(data.height - 1)
                })
            {
                return Err(Error::Invalid("invalid staff-edge observations"));
            }
        }
        if !config.size_prior.is_finite() || config.size_prior < 0. {
            return Err(Error::Invalid("size prior must be finite and nonnegative"));
        }
        let mut seen = vec![false; data.staves.len()];
        for group in &config.staff_size_groups {
            if group.is_empty() {
                return Err(Error::Invalid("empty staff size group"));
            }
            for i in group {
                if *i >= seen.len() || seen[*i] {
                    return Err(Error::Invalid(
                        "size groups need unique valid zero-based staff indices",
                    ));
                }
                seen[*i] = true;
            }
        }
        profile.input_validation_seconds = stage.elapsed().as_secs_f64();
        profile.failure_stage = Some("perspective");
        let stage = Instant::now();
        let (projective, perspective) =
            perspective::fit_with_page_frame(data, config, corners, edges);
        profile.perspective_seconds = stage.elapsed().as_secs_f64();
        if perspective.at_parameter_bound || !perspective.converged {
            return Err(Error::Invalid(
                "perspective fit is underconstrained or reached a parameter bound; inspect size groups or disable perspective",
            ));
        }
        profile.failure_stage = Some("staff_warp");
        let stage = Instant::now();
        let mut border = vec![];
        for i in 0..101 {
            let x = f64::from(data.width - 1) * i as f64 / 100.;
            let y = f64::from(data.height - 1) * i as f64 / 100.;
            border.extend([
                Point { x, y: 0. },
                Point {
                    x,
                    y: f64::from(data.height - 1),
                },
                Point { x: 0., y },
                Point {
                    x: f64::from(data.width - 1),
                    y,
                },
            ]);
        }
        let pb: Vec<Point> = border.iter().map(|p| projective.forward(*p)).collect();
        let min_y = pb.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let max_y = pb.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
        let mut staff_curves = vec![];
        let mut staff_extensions = crate::geometry::StaffExtensionReport::default();
        let projected_edges: Vec<_> = edges
            .iter()
            .map(|edge| {
                let mut edge = edge.clone();
                edge.points = edge.points.iter().map(|p| projective.forward(*p)).collect();
                edge
            })
            .collect();
        let mut centers = vec![];
        let mut spacings = vec![];
        for (staff_index, staff) in data.staves.iter().enumerate() {
            let mut lines: [Vec<Point>; 5] = std::array::from_fn(|k| {
                staff
                    .x
                    .iter()
                    .zip(&staff.y[k])
                    .map(|(x, y)| projective.forward(Point { x: *x, y: *y }))
                    .collect()
            });
            // Preserve measured tips, completing unmeasured sister-line tails
            // coherently instead of clamping them to a fictitious constant y.
            crate::geometry::corrections::staff_extensions::extend_measured_ends(
                staff_index,
                &mut lines,
                &projected_edges,
                &mut staff_extensions,
            );
            if lines.iter().any(|l| l.windows(2).any(|w| w[0].x >= w[1].x)) {
                return Err(Error::Invalid("projected staff curves turn back in x"));
            }
            let local: Vec<Curve> = lines
                .iter()
                .map(|l| Curve {
                    x: l.iter().map(|p| p.x).collect(),
                    y: l.iter().map(|p| p.y).collect(),
                })
                .collect();
            let left = local
                .iter()
                .map(|c| c.x[0])
                .fold(f64::NEG_INFINITY, f64::max);
            let right = local
                .iter()
                .map(|c| c.x[c.x.len() - 1])
                .fold(f64::INFINITY, f64::min);
            if left >= right {
                return Err(Error::Invalid("staff lines have no common x interval"));
            }
            let samples: Vec<[f64; 5]> = (0..128)
                .map(|i| {
                    let x = left + (right - left) * i as f64 / 127.;
                    std::array::from_fn(|k| interpolate(&local[k].x, &local[k].y, x))
                })
                .collect();
            let (center, space) = joint_staff(&samples);
            centers.push(center);
            spacings.push(space);
            staff_curves.push(lines);
        }
        let mut target_spacings = spacings.clone();
        for group in &config.staff_size_groups {
            let common = median(group.iter().map(|i| spacings[*i].ln()).collect());
            for i in group {
                target_spacings[*i] = ((spacings[*i].ln() + config.size_prior * common)
                    / (1. + config.size_prior))
                    .exp();
            }
        }
        let min_x = pb.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let max_x = pb.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        let local = crate::geometry::corrections::local_flatten::LocalFlatten::fit(
            &staff_curves,
            &centers,
            &target_spacings,
            [min_x, min_y, max_x, max_y],
        )?;
        let mut targets = vec![min_y];
        for (center, space) in local.target_centers().iter().zip(&target_spacings) {
            targets.extend((0..5).map(|k| center + (k as f64 - 2.) * space));
        }
        targets.push(max_y);
        let boundary_jump = local.max_boundary_jump_pixels;
        if !boundary_jump.is_finite() || boundary_jump > 1e-7 {
            return Err(Error::Invalid("local staff supports are discontinuous"));
        }
        let extrapolation = ExtrapolationReport {
            model: "local_support_monotone_columns",
            support_limit_staff_spaces: 8.,
            max_boundary_jump_pixels: boundary_jump,
            adjusted_anchor_samples: 0,
            max_adjustment_pixels: 0.,
            changed_x_bounds: None,
            min_source_gap_pixels: local.min_source_gap_pixels,
            max_vertical_scale: local.max_vertical_scale,
        };
        let flat = Flatten {
            local,
            targets,
            min_y,
            max_y,
        };
        profile.staff_warp_seconds = stage.elapsed().as_secs_f64();
        profile.failure_stage = Some("system_shear");
        let stage = Instant::now();
        let horizontal = shear(
            data,
            &flat,
            &projective,
            median(target_spacings.clone()),
            config.system_shear,
            edges,
        )?;
        profile.system_shear_seconds = stage.elapsed().as_secs_f64();
        profile.failure_stage = Some("map_validation");
        let stage = Instant::now();
        let report = Report {
            #[cfg(feature = "render")]
            page_model: None,
            #[cfg(feature = "render")]
            direction_mesh: None,
            schema: "smude.geometry.v1",
            input_width: data.width,
            input_height: data.height,
            perspective,
            config: config.clone(),
            measured_staff_spacings: spacings,
            target_staff_spacings: target_spacings,
            systems: horizontal.systems.clone(),
            bounds: [0.; 4],
            validation: Validation {
                sampled_min_jacobian: 0.,
                roundtrip_max_pixels: 0.,
                final_stage_y_displacement: 0.,
                max_staff_line_span_pixels: 0.,
                bar_median_angle_degrees: 0.,
                bar_p90_angle_degrees: 0.,
                bar_max_bend_pixels: 0.,
            },
            validation_note: "Spacing/angles use detector measurements, not independent ink or metric ground truth. Jacobian sampled over page; no local x mesh.",
            extrapolation,
            staff_edges: edges.to_vec(),
            staff_extensions,
            vertical_correction: None,
            #[cfg(feature = "render")]
            gcs: None,
            #[cfg(feature = "render")]
            text_warp: None,
        };
        let mut geometry = Self {
            #[cfg(feature = "render")]
            pre_page: None,
            projective,
            direction_mesh: crate::geometry::DirectionMesh::default(),
            #[cfg(feature = "render")]
            pre_frame: None,
            flat,
            horizontal,
            vertical: crate::geometry::VerticalCorrection::default(),
            #[cfg(feature = "render")]
            pre_gcs: None,
            report,
            #[cfg(feature = "render")]
            text: None,
        };
        geometry.validate(data)?;
        let fb: Vec<Point> = border.iter().map(|p| geometry.forward(*p)).collect();
        geometry.report.bounds = [
            fb.iter().map(|p| p.x).fold(f64::INFINITY, f64::min),
            fb.iter().map(|p| p.y).fold(f64::INFINITY, f64::min),
            fb.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max),
            fb.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max),
        ];
        profile.map_validation_seconds = stage.elapsed().as_secs_f64();
        Ok(geometry)
    }

    pub fn forward(&self, p: Point) -> Point {
        self.forward_near(p, &mut Nearby::default())
    }
    /// `forward`, shortening its searches by those of an earlier point.
    pub(crate) fn forward_near(&self, p: Point, nearby: &mut Nearby) -> Point {
        let Some(flat) = self.flattened(p, nearby) else {
            return Point {
                x: f64::NAN,
                y: f64::NAN,
            };
        };
        let result = self
            .direction_mesh
            .forward(self.vertical.forward(self.horizontal.forward(flat)));
        #[cfg(feature = "render")]
        if let Some(text) = &self.text {
            return text.forward(result);
        }
        result
    }
    /// `forward(p).y`: past the staff flattening only a text residual moves y.
    fn forward_y(&self, p: Point, nearby: &mut Nearby) -> f64 {
        #[cfg(feature = "render")]
        if self.text.is_some() {
            return self.forward_near(p, nearby).y;
        }
        self.flattened(p, nearby).map_or(f64::NAN, |flat| flat.y)
    }
    /// The stages of `forward` up to the staff flattening, unless the point
    /// leaves the finite plane on the way.
    fn flattened(&self, p: Point, nearby: &mut Nearby) -> Option<Point> {
        let finite = |p: Point| p.x.is_finite() && p.y.is_finite();
        if !finite(p) {
            return None;
        }
        #[cfg(feature = "render")]
        let p = self.pre_page.as_ref().map_or(p, |(map, shift)| {
            let q = map.forward(p).unwrap_or(Point {
                x: f64::NAN,
                y: f64::NAN,
            });
            Point {
                x: q.x - shift.x,
                y: q.y - shift.y,
            }
        });
        #[cfg(feature = "render")]
        let p = self.pre_gcs.as_ref().map_or(p, |m| m.forward(p));
        #[cfg(feature = "render")]
        let p = self.pre_frame.as_ref().map_or(p, |(frame, shift)| {
            let q = frame.forward(p).unwrap_or(Point {
                x: f64::NAN,
                y: f64::NAN,
            });
            Point {
                x: q.x - shift.x,
                y: q.y - shift.y,
            }
        });
        let projected = self.projective.forward(p);
        if !finite(projected) {
            return None;
        }
        let flat = self.flat.forward_near(projected, nearby);
        finite(flat).then_some(flat)
    }
    /// The protected text residual alone. `forward` is this applied to the base
    /// map, so points traced before `add_text_curves` remain reusable after it.
    #[cfg(feature = "render")]
    pub fn forward_text(&self, base: Point) -> Point {
        self.text.as_ref().map_or(base, |text| text.forward(base))
    }
    pub fn inverse(&self, p: Point) -> Point {
        if !p.x.is_finite() || !p.y.is_finite() {
            return Point {
                x: f64::NAN,
                y: f64::NAN,
            };
        }
        #[cfg(feature = "render")]
        let p = self.text.as_ref().map_or(p, |text| text.inverse(p));
        let p = self.direction_mesh.inverse(p);
        let p = self.projective.inverse(
            self.flat
                .inverse(self.horizontal.inverse(self.vertical.inverse(p))),
        );
        #[cfg(feature = "render")]
        let p = self.inverse_page(p);
        p
    }
    /// The page stages of `inverse`, from projected coordinates to the source.
    #[cfg(feature = "render")]
    fn inverse_page(&self, p: Point) -> Point {
        let p = self.pre_gcs.as_ref().map_or(p, |m| m.inverse(p));
        let p = self.pre_frame.as_ref().map_or(p, |(frame, shift)| {
            frame
                .inverse(Point {
                    x: p.x + shift.x,
                    y: p.y + shift.y,
                })
                .unwrap_or(Point {
                    x: f64::NAN,
                    y: f64::NAN,
                })
        });
        self.pre_page.as_ref().map_or(p, |(map, shift)| {
            map.inverse(Point {
                x: p.x + shift.x,
                y: p.y + shift.y,
            })
            .unwrap_or(Point {
                x: f64::NAN,
                y: f64::NAN,
            })
        })
    }
    #[cfg(feature = "render")]
    pub(crate) fn inverse_scanline(&self, y: f64) -> Scanline<'_> {
        Scanline {
            geometry: self,
            y,
            text: self
                .text
                .as_ref()
                .map_or_else(Vec::new, |text| text.active_row_spans(y)),
            vertical: self.vertical.inverse_row(y),
            mesh: self.direction_mesh.inverse_row(y),
            shear: self.horizontal.at(y),
        }
    }

    /// Fit the bounded common x field after horizontal flattening. This never
    /// changes output y and is composed before any protected text residual.
    pub fn add_vertical_correction(&mut self, data: &Observations) -> Result<()> {
        let width = self.report.bounds[2].max(self.report.bounds[2] - self.report.bounds[0]);
        let correction = crate::geometry::VerticalCorrection::fit(
            data,
            &|p| self.forward(p),
            &self.report.target_staff_spacings,
            width,
        );
        self.report.vertical_correction = Some(correction.clone());
        self.vertical = correction;
        self.validate(data)?;
        let mut border = Vec::with_capacity(404);
        for i in 0..=100 {
            let x = f64::from(data.width - 1) * i as f64 / 100.;
            let y = f64::from(data.height - 1) * i as f64 / 100.;
            border.extend([
                Point { x, y: 0. },
                Point {
                    x,
                    y: f64::from(data.height - 1),
                },
                Point { x: 0., y },
                Point {
                    x: f64::from(data.width - 1),
                    y,
                },
            ]);
        }
        let border: Vec<_> = border.into_iter().map(|p| self.forward(p)).collect();
        self.report.bounds = [
            border.iter().map(|p| p.x).fold(f64::INFINITY, f64::min),
            border.iter().map(|p| p.y).fold(f64::INFINITY, f64::min),
            border.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max),
            border.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max),
        ];
        Ok(())
    }
    /// Local text residual is composed into the final original-image inverse.
    /// Staff and bar guards are exact identity regions, not soft penalties.
    #[cfg(feature = "render")]
    pub fn add_text_curves(
        &mut self,
        lines: &[crate::detection::TextLine],
        protected: &[[f64; 4]],
    ) {
        let (warp, report) = crate::geometry::corrections::text_warp::TextWarp::fit(
            lines,
            protected,
            self.report.bounds,
        );
        self.report.validation.final_stage_y_displacement = report.max_displacement_pixels;
        self.report.validation_note = "Jacobian/roundtrip samples describe the base staff map. The local text residual has its own exact vertical-derivative bound and is identity on protected music. Staff/bar statistics use detector observations, not independent ground truth.";
        self.text = warp;
        self.report.text_warp = Some(report);
    }
    pub fn report(&self) -> &Report {
        &self.report
    }

    fn validate(&mut self, data: &Observations) -> Result<()> {
        let mut min_det = f64::INFINITY;
        let mut roundtrip = 0_f64;
        for j in 0..41 {
            for i in 0..31 {
                let p = Point {
                    x: f64::from(data.width - 1) * i as f64 / 30.,
                    y: f64::from(data.height - 1) * j as f64 / 40.,
                };
                let out = self.forward(p);
                let back = self.inverse(out);
                roundtrip = roundtrip.max((p.x - back.x).hypot(p.y - back.y));
                // Finite triangles can cross C0 mesh knots and report the wrong
                // orientation. Use the exact composition: projective map,
                // monotone vertical map, then a determinant-one x shear.
                #[cfg(feature = "render")]
                let (p, pre_det) = self.pre_page.as_ref().map_or((p, 1.), |(map, shift)| {
                    let q = map.forward(p).unwrap_or(Point {
                        x: f64::NAN,
                        y: f64::NAN,
                    });
                    (
                        Point {
                            x: q.x - shift.x,
                            y: q.y - shift.y,
                        },
                        map.jacobian_det(p),
                    )
                });
                #[cfg(feature = "render")]
                let (p, pre_det) = self.pre_gcs.as_ref().map_or((p, pre_det), |m| {
                    (m.forward(p), pre_det * m.jacobian_det(p))
                });
                #[cfg(feature = "render")]
                let (p, pre_det) =
                    self.pre_frame
                        .as_ref()
                        .map_or((p, pre_det), |(frame, shift)| {
                            let q = frame.forward(p).unwrap_or(Point {
                                x: f64::NAN,
                                y: f64::NAN,
                            });
                            (
                                Point {
                                    x: q.x - shift.x,
                                    y: q.y - shift.y,
                                },
                                pre_det * frame.jacobian_det(p),
                            )
                        });
                #[cfg(not(feature = "render"))]
                let pre_det = 1.;
                let projected = self.projective.forward(p);
                let flattened = self.horizontal.forward(self.flat.forward(projected));
                let det = pre_det
                    * self.projective.jacobian_det(p)
                    * self.flat.local.vertical_scale(projected)
                    * self.vertical.jacobian_det(flattened)
                    * self
                        .direction_mesh
                        .jacobian_det(self.vertical.forward(flattened));
                if !det.is_finite() || det <= 0. {
                    return Err(Error::Invalid("fitted geometry folds the page"));
                }
                min_det = min_det.min(det);
            }
        }
        // Every staff sample is traced through the whole map: the costly part.
        let line_spans = crate::parallel::map(&data.staves, |s| {
            let mut span = 0_f64;
            let mut nearby = Nearby::default();
            for k in 0..5 {
                let values: Vec<f64> =
                    s.x.iter()
                        .zip(&s.y[k])
                        .map(|(x, y)| self.forward_y(Point { x: *x, y: *y }, &mut nearby))
                        .collect();
                span = span.max(
                    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                        - values.iter().copied().fold(f64::INFINITY, f64::min),
                );
            }
            span
        });
        let line_span = line_spans.into_iter().fold(0_f64, f64::max);
        let mut angles = vec![];
        let mut max_bend = 0_f64;
        for bar in &data.barlines {
            let p: Vec<Point> = bar
                .x
                .iter()
                .zip(&bar.y)
                .map(|(x, y)| self.forward(Point { x: *x, y: *y }))
                .collect();
            let s = slope(&p);
            angles.push(s.abs().atan().to_degrees());
            let mean = p.iter().map(|p| p.x - s * p.y).sum::<f64>() / p.len() as f64;
            let rms = (p
                .iter()
                .map(|p| (p.x - s * p.y - mean).powi(2))
                .sum::<f64>()
                / p.len() as f64)
                .sqrt();
            max_bend = max_bend.max(rms);
        }
        angles.sort_by(f64::total_cmp);
        let (med, p90) = if angles.is_empty() {
            (0., 0.)
        } else {
            (
                median(angles.clone()),
                angles[((angles.len() - 1) as f64 * 0.9).ceil() as usize],
            )
        };
        self.report.validation = Validation {
            sampled_min_jacobian: min_det,
            roundtrip_max_pixels: roundtrip,
            final_stage_y_displacement: 0.,
            max_staff_line_span_pixels: line_span,
            bar_median_angle_degrees: med,
            bar_p90_angle_degrees: p90,
            bar_max_bend_pixels: max_bend,
        };
        Ok(())
    }
}

#[cfg(test)]
mod profile_tests {
    use super::*;
    use crate::geometry::Staff;

    fn observations(centers: &[f64]) -> Observations {
        Observations {
            width: 200,
            height: 160,
            staves: centers
                .iter()
                .map(|center| Staff {
                    x: vec![20., 100., 180.],
                    y: std::array::from_fn(|k| vec![center + (k as f64 - 2.) * 5.; 3]),
                })
                .collect(),
            barlines: vec![],
        }
    }

    #[test]
    fn profiled_fit_preserves_geometry_and_report() {
        let data = observations(&[40., 110.]);
        let config = Config {
            perspective: false,
            ..Config::default()
        };
        let regular = Geometry::fit(&data, &config).unwrap();
        let (result, profile) = Geometry::fit_profiled(&data, &config);
        let profiled = result.unwrap();
        assert_eq!(profile.failure_stage, None);
        assert!(profile.total_seconds.is_finite() && profile.total_seconds >= 0.);
        assert!(profile.unattributed_seconds >= 0.);
        assert_eq!(
            serde_json::to_value(regular.report()).unwrap(),
            serde_json::to_value(profiled.report()).unwrap()
        );
        for y in [0., 30., 75., 120., 159.] {
            for x in [0., 20., 100., 180., 199.] {
                let point = Point { x, y };
                let a = regular.forward(point);
                let b = profiled.forward(point);
                assert_eq!((a.x, a.y), (b.x, b.y));
                let a = regular.inverse(point);
                let b = profiled.inverse(point);
                assert_eq!((a.x, a.y), (b.x, b.y));
            }
        }
        let serialized = serde_json::to_value(profile).unwrap();
        assert!(serialized["map_validation_seconds"].is_number());
    }

    #[test]
    fn profiled_target_overlap_error_identifies_unfinished_stage() {
        let data = observations(&[50., 55.]);
        let config = Config {
            perspective: false,
            ..Config::default()
        };
        let (result, profile) = Geometry::fit_profiled(&data, &config);
        assert!(matches!(
            result,
            Err(Error::Invalid("overlapping measured local staff bands"))
        ));
        assert_eq!(profile.failure_stage, Some("staff_warp"));
        assert_eq!(profile.staff_warp_seconds, 0.);
        assert_eq!(profile.system_shear_seconds, 0.);
        assert_eq!(profile.map_validation_seconds, 0.);
        assert!(profile.total_seconds >= profile.input_validation_seconds);
        assert!(profile.unattributed_seconds >= 0.);
    }

    #[test]
    fn profiled_input_error_returns_profile_alongside_error() {
        let (result, profile) = Geometry::fit_profiled(&observations(&[]), &Config::default());
        assert!(result.is_err());
        assert_eq!(profile.failure_stage, Some("input_validation"));
        assert_eq!(profile.input_validation_seconds, 0.);
        assert_eq!(profile.perspective_seconds, 0.);
        assert_eq!(profile.total_seconds, profile.unattributed_seconds);
    }
}
