use serde::Serialize;
use std::time::Instant;

use crate::observations::interpolate;
use crate::perspective::{self, PerspectiveReport, Projective, joint_staff};
use crate::{Config, Error, Observations, Point, Result, StaffEdge, median, slope};

#[derive(Clone, Debug)]
struct Curve {
    x: std::sync::Arc<[f64]>,
    y: Vec<f64>,
}

#[derive(Clone, Debug)]
struct Flatten {
    local: crate::local_flatten::LocalFlatten,
    targets: Vec<f64>,
    min_y: f64,
    max_y: f64,
}
impl Flatten {
    fn forward(&self, p: Point) -> Point {
        self.local.forward(p)
    }
    fn inverse(&self, p: Point) -> Point {
        self.local.inverse(p)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SystemReport {
    pub staves: Vec<usize>,
    pub region: [f64; 2],
    pub center: f64,
    pub shear: f64,
    pub candidate_shear: f64,
}

#[derive(Clone, Debug)]
struct Shear {
    systems: Vec<SystemReport>,
    knots: Vec<f64>,
    values: Vec<f64>,
    derivatives: Vec<f64>,
}

impl Shear {
    fn at(&self, y: f64) -> f64 {
        if y <= self.knots[0] || y >= self.knots[self.knots.len() - 1] {
            return 0.;
        }
        let i = self.knots.partition_point(|v| *v <= y) - 1;
        let length = self.knots[i + 1] - self.knots[i];
        let t = (y - self.knots[i]) / length;
        (2. * t * t * t - 3. * t * t + 1.) * self.values[i]
            + (t * t * t - 2. * t * t + t) * length * self.derivatives[i]
            + (-2. * t * t * t + 3. * t * t) * self.values[i + 1]
            + (t * t * t - t * t) * length * self.derivatives[i + 1]
    }
    fn forward(&self, p: Point) -> Point {
        Point {
            x: p.x + self.at(p.y),
            y: p.y,
        }
    }
    fn inverse(&self, p: Point) -> Point {
        Point {
            x: p.x - self.at(p.y),
            y: p.y,
        }
    }
}

fn systems(data: &Observations) -> Vec<Vec<usize>> {
    let n = data.staves.len();
    let mut counts = vec![0_usize; n];
    let mut matched = vec![vec![0_usize; n]; n];
    let mut full_spans = vec![0_usize; n];
    for b in &data.barlines {
        counts[b.staff] += 1;
        for other in &b.aligned_staffs {
            matched[b.staff][*other] += 1;
        }
        // Native detection emits one physical path for a connected bar.
        // Older observations instead contain reciprocal staff-height paths;
        // retain their voting rule without counting those twice.
        let mut members = b.aligned_staffs.clone();
        members.push(b.staff);
        members.sort_unstable();
        for pair in members.windows(2) {
            let (a, z) = (pair[0], pair[1]);
            if z != a + 1 {
                continue;
            }
            let upper = interpolate(&data.staves[a].x, &data.staves[a].y[2], b.x[0]);
            let lower = interpolate(
                &data.staves[z].x,
                &data.staves[z].y[2],
                *b.x.last().unwrap(),
            );
            if b.y[0] <= upper && *b.y.last().unwrap() >= lower {
                full_spans[z] += 1;
            }
        }
    }
    let mut out = vec![vec![0]];
    for i in 1..n {
        let threshold = 2_f64.max(0.4 * counts[i - 1].min(counts[i]) as f64);
        if full_spans[i] >= 2 || matched[i][i - 1].min(matched[i - 1][i]) as f64 >= threshold {
            out.last_mut().unwrap().push(i);
        } else {
            out.push(vec![i]);
        }
    }
    out
}

fn shear(
    data: &Observations,
    flat: &Flatten,
    projective: &Projective,
    space: f64,
    enabled: bool,
    edges: &[StaffEdge],
) -> Result<Shear> {
    let paths: Vec<Vec<Point>> = data
        .barlines
        .iter()
        .map(|b| {
            b.x.iter()
                .zip(&b.y)
                .map(|(x, y)| flat.forward(projective.forward(Point { x: *x, y: *y })))
                .collect()
        })
        .collect();
    let mut reports = vec![];
    for group in systems(data) {
        let top = group
            .iter()
            .map(|i| flat.targets[1 + 5 * i])
            .fold(f64::INFINITY, f64::min);
        let bottom = group
            .iter()
            .map(|i| flat.targets[5 + 5 * i])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut lo = top;
        let mut hi = bottom;
        let mut slopes = vec![];
        for (b, p) in data.barlines.iter().zip(&paths) {
            if group.contains(&b.staff) {
                lo = lo.min(p.first().unwrap().y);
                hi = hi.max(p.last().unwrap().y);
                if p.last().unwrap().y - p.first().unwrap().y >= space {
                    slopes.extend(std::iter::repeat_n(slope(p), 4));
                }
            }
        }
        // Weaker, explicitly geometric staff edges remain separate from bars.
        for edge in edges
            .iter()
            .filter(|e| group.contains(&e.staff) && e.confidence >= 0.7)
        {
            let points: Vec<_> = edge
                .points
                .iter()
                .map(|p| flat.forward(projective.forward(*p)))
                .collect();
            if points.last().unwrap().y - points[0].y >= space {
                slopes.push(slope(&points));
            }
        }
        let candidate_shear = if slopes.is_empty() {
            0.
        } else {
            (-median(slopes)).clamp(-0.15, 0.15)
        };
        reports.push(SystemReport {
            staves: group,
            region: [lo - 0.5 * space, hi + 0.5 * space],
            center: (top + bottom) / 2.,
            shear: if enabled { candidate_shear } else { 0. },
            candidate_shear,
        });
    }
    if !enabled {
        return Ok(Shear {
            systems: reports,
            knots: vec![flat.min_y, flat.max_y],
            values: vec![0., 0.],
            derivatives: vec![0., 0.],
        });
    }
    for i in 1..reports.len() {
        if reports[i - 1].region[1] + space >= reports[i].region[0] {
            let left = flat.targets[5 + 5 * reports[i - 1].staves[reports[i - 1].staves.len() - 1]];
            let right = flat.targets[1 + 5 * reports[i].staves[0]];
            let edge = (left + right) / 2.;
            reports[i - 1].region[1] = reports[i - 1].region[1].min(edge - 0.5 * space);
            reports[i].region[0] = reports[i].region[0].max(edge + 0.5 * space);
        }
    }
    let mut result = Shear {
        knots: vec![reports[0].region[0] - 3. * space],
        values: vec![0.],
        derivatives: vec![0.],
        systems: reports,
    };
    for system in &result.systems {
        for y in system.region {
            result.knots.push(y);
            result.values.push(system.shear * (y - system.center));
            result.derivatives.push(system.shear);
        }
    }
    result
        .knots
        .push(result.systems.last().unwrap().region[1] + 3. * space);
    result.values.push(0.);
    result.derivatives.push(0.);
    if result.knots.windows(2).any(|w| w[0] >= w[1]) {
        return Err(Error::Invalid(
            "overlapping systems require explicit geometry repair",
        ));
    }
    Ok(result)
}

#[derive(Clone, Debug, Serialize)]
pub struct Validation {
    pub sampled_min_jacobian: f64,
    pub roundtrip_max_pixels: f64,
    pub final_stage_y_displacement: f64,
    pub max_staff_line_span_pixels: f64,
    pub bar_median_angle_degrees: f64,
    pub bar_p90_angle_degrees: f64,
    pub bar_max_bend_pixels: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub schema: &'static str,
    pub input_width: u32,
    pub input_height: u32,
    pub perspective: PerspectiveReport,
    pub config: Config,
    pub measured_staff_spacings: Vec<f64>,
    pub target_staff_spacings: Vec<f64>,
    pub systems: Vec<SystemReport>,
    pub bounds: [f64; 4],
    pub validation: Validation,
    pub validation_note: &'static str,
    pub extrapolation: ExtrapolationReport,
    pub staff_edges: Vec<StaffEdge>,
    pub staff_extensions: crate::StaffExtensionReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vertical_correction: Option<crate::VerticalCorrection>,
    #[cfg(feature = "render")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gcs: Option<crate::GcsPipelineReport>,
    #[cfg(feature = "render")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_warp: Option<crate::TextWarpReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExtrapolationReport {
    pub model: &'static str,
    pub support_limit_staff_spaces: f64,
    pub max_boundary_jump_pixels: f64,
    pub adjusted_anchor_samples: usize,
    pub max_adjustment_pixels: f64,
    pub changed_x_bounds: Option<[f64; 2]>,
    pub min_source_gap_pixels: f64,
    pub max_vertical_scale: f64,
}

/// Wall-clock stage timings for one geometry fit, including unsuccessful fits.
#[derive(Default, Clone, Debug, Serialize)]
pub struct FitProfile {
    pub input_validation_seconds: f64,
    pub perspective_seconds: f64,
    pub staff_warp_seconds: f64,
    pub system_shear_seconds: f64,
    pub gcs_seconds: f64,
    pub vertical_correction_seconds: f64,
    pub staff_conditioning_seconds: f64,
    pub map_validation_seconds: f64,
    pub total_seconds: f64,
    pub unattributed_seconds: f64,
    pub failure_stage: Option<&'static str>,
}

/// Fitted, reversible geometry. All maps use pixel coordinates and (x,y) order.
/// No detector/model objects or intermediate raster images are retained.
#[derive(Clone, Debug)]
pub struct Geometry {
    projective: Projective,
    flat: Flatten,
    horizontal: Shear,
    vertical: crate::VerticalCorrection,
    #[cfg(feature = "render")]
    pre_gcs: Option<crate::pipeline::PreGcs>,
    report: Report,
    #[cfg(feature = "render")]
    text: Option<crate::text_warp::TextWarp>,
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
                    match crate::PageAnalysisMap::new(data.width, data.height, c, r, data.width) {
                        Ok(map) => (Some(map), None),
                        Err(error) => (None, Some(error.to_string())),
                    }
                }
                None => (None, None),
            };
            profile.perspective_seconds = stage.elapsed().as_secs_f64();
            profile.failure_stage = Some("gcs");
            let stage = Instant::now();
            let gcs = crate::GcsMap::fit_with_camera(
                data,
                edges,
                ratio,
                frame.as_ref().and(corners),
                camera_center,
            )?;
            let plane_evidence = frame
                .as_ref()
                .map(|m| crate::pipeline::plane_evidence(data, m));
            let use_plane = plane_evidence.as_ref().is_some_and(|e| e.accepted);
            let (mapped, mapped_edges, gcs_shift) =
                crate::pipeline::map_observations(data, edges, |p| {
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
            let (conditioned, staff_conditioning) = crate::condition_staff_observations(&mapped);
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
            let pre = crate::pipeline::PreGcs {
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
            let border: Vec<_> = crate::pipeline::border(data.width, data.height)
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
        let mut staff_extensions = crate::StaffExtensionReport::default();
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
            crate::staff_extensions::extend_measured_ends(
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
        let local = crate::local_flatten::LocalFlatten::fit(
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
            projective,
            flat,
            horizontal,
            vertical: crate::VerticalCorrection::default(),
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
        if !p.x.is_finite() || !p.y.is_finite() {
            return Point {
                x: f64::NAN,
                y: f64::NAN,
            };
        }
        #[cfg(feature = "render")]
        let p = self.pre_gcs.as_ref().map_or(p, |m| m.forward(p));
        let projected = self.projective.forward(p);
        if !projected.x.is_finite() || !projected.y.is_finite() {
            return Point {
                x: f64::NAN,
                y: f64::NAN,
            };
        }
        let flat = self.flat.forward(projected);
        if !flat.x.is_finite() || !flat.y.is_finite() {
            return Point {
                x: f64::NAN,
                y: f64::NAN,
            };
        }
        let result = self.vertical.forward(self.horizontal.forward(flat));
        #[cfg(feature = "render")]
        if let Some(text) = &self.text {
            return text.forward(result);
        }
        result
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
        let p = self.projective.inverse(
            self.flat
                .inverse(self.horizontal.inverse(self.vertical.inverse(p))),
        );
        #[cfg(feature = "render")]
        let p = self.pre_gcs.as_ref().map_or(p, |m| m.inverse(p));
        p
    }
    /// Fit the bounded common x field after horizontal flattening. This never
    /// changes output y and is composed before any protected text residual.
    pub fn add_vertical_correction(&mut self, data: &Observations) -> Result<()> {
        let width = self.report.bounds[2].max(self.report.bounds[2] - self.report.bounds[0]);
        let correction = crate::VerticalCorrection::fit(
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
    pub fn add_text_curves(&mut self, lines: &[crate::TextLine], protected: &[[f64; 4]]) {
        let (warp, report) = crate::text_warp::TextWarp::fit(lines, protected, self.report.bounds);
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
                let (p, pre_det) = self
                    .pre_gcs
                    .as_ref()
                    .map_or((p, 1.), |m| (m.forward(p), m.jacobian_det(p)));
                #[cfg(not(feature = "render"))]
                let pre_det = 1.;
                let projected = self.projective.forward(p);
                let flattened = self.horizontal.forward(self.flat.forward(projected));
                let det = pre_det
                    * self.projective.jacobian_det(p)
                    * self.flat.local.vertical_scale(projected)
                    * self.vertical.jacobian_det(flattened);
                if !det.is_finite() || det <= 0. {
                    return Err(Error::Invalid("fitted geometry folds the page"));
                }
                min_det = min_det.min(det);
            }
        }
        let mut line_span = 0_f64;
        for s in &data.staves {
            for k in 0..5 {
                let values: Vec<f64> =
                    s.x.iter()
                        .zip(&s.y[k])
                        .map(|(x, y)| self.forward(Point { x: *x, y: *y }).y)
                        .collect();
                line_span = line_span.max(
                    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                        - values.iter().copied().fold(f64::INFINITY, f64::min),
                );
            }
        }
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
    use crate::Staff;

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
