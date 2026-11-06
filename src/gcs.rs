//! Native generalized-cylinder geometry and a bounded music-specific fit.
//!
//! Camera/surface conventions follow the Kil2017 GCS formulation. In contrast
//! to the earlier experiment, ray intersections have a full-domain uniqueness
//! certificate and both directions use the same monotone arc-length table.
use serde::{Deserialize, Serialize};

use crate::{Error, Observations, Point, Result, StaffEdge};

const N: usize = 8;
const ARC_STEPS: usize = 2048;
const K_LO: f64 = 0.4;
const K_HI: f64 = 2.0;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GcsParameters {
    pub a: [f64; 4],
    pub rvec: [f64; 3],
    pub f: f64,
    pub cx: f64,
    pub cy: f64,
    pub scale: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct GcsReport {
    pub model: &'static str,
    pub parameters: GcsParameters,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub accepted_iterations: usize,
    pub rejected_steps: usize,
    pub staff_groups: usize,
    pub full_bar_paths: usize,
    pub edge_paths: usize,
    pub height_width_ratio: Option<f64>,
    pub aspect_constraint: &'static str,
    pub camera_principal_point_source: &'static str,
    pub roll_initialization_radians: f64,
    pub rotation_z_bounds: [f64; 2],
    pub paper_pose_initialization: Option<[f64; 3]>,
    pub residual_units: &'static str,
    pub depth_bounds: [f64; 2],
    pub max_ray_equation_derivative: f64,
    pub minimum_sampled_jacobian: f64,
    pub maximum_roundtrip_pixels: f64,
    pub validation: &'static str,
}

#[derive(Clone, Debug)]
pub struct GcsMap {
    parameters: GcsParameters,
    rotation: [[f64; 3]; 3],
    arc_x: Vec<f64>,
    arc_u: Vec<f64>,
    report: GcsReport,
}

#[derive(Clone, Copy)]
struct Interval(f64, f64);
impl Interval {
    fn add(self, a: f64) -> Self {
        Self(self.0 + a, self.1 + a)
    }
    fn mul(self, other: Self) -> Self {
        let v = [
            self.0 * other.0,
            self.0 * other.1,
            self.1 * other.0,
            self.1 * other.1,
        ];
        Self(
            v.iter().copied().fold(f64::INFINITY, f64::min),
            v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    }
    fn scale(self, a: f64) -> Self {
        self.mul(Self(a, a))
    }
}

impl GcsParameters {
    pub fn identity(width: u32, height: u32) -> Self {
        Self {
            a: [0.; 4],
            rvec: [0.; 3],
            f: width.max(height) as f64,
            cx: (width - 1) as f64 / 2.,
            cy: (height - 1) as f64 / 2.,
            scale: width.max(height) as f64,
        }
    }
    fn g(&self, x: f64) -> f64 {
        let t = x / self.scale;
        self.scale * t * (self.a[0] + t * (self.a[1] + t * (self.a[2] + t * self.a[3])))
    }
    fn gp(&self, x: f64) -> f64 {
        let t = x / self.scale;
        self.a[0] + t * (2. * self.a[1] + t * (3. * self.a[2] + t * 4. * self.a[3]))
    }
    fn range(&self, x: Interval, derivative: bool) -> Interval {
        let t = x.scale(1. / self.scale);
        if derivative {
            t.scale(4. * self.a[3])
                .add(3. * self.a[2])
                .mul(t)
                .add(2. * self.a[1])
                .mul(t)
                .add(self.a[0])
        } else {
            t.scale(self.a[3])
                .add(self.a[2])
                .mul(t)
                .add(self.a[1])
                .mul(t)
                .add(self.a[0])
                .mul(t)
                .scale(self.scale)
        }
    }
    fn theta(&self) -> [f64; N] {
        [
            self.a[0],
            self.a[1],
            self.a[2],
            self.a[3],
            self.rvec[0],
            self.rvec[1],
            self.rvec[2],
            (self.f / self.scale).ln(),
        ]
    }
    fn with_theta(&self, t: [f64; N]) -> Self {
        Self {
            a: [t[0], t[1], t[2], t[3]],
            rvec: [t[4], t[5], t[6]],
            f: self.scale * t[7].exp(),
            ..self.clone()
        }
    }
}

fn rodrigues(v: [f64; 3]) -> [[f64; 3]; 3] {
    let theta = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let (a, b) = if theta < 1e-8 {
        (1. - theta * theta / 6., 0.5 - theta * theta / 24.)
    } else {
        (theta.sin() / theta, (1. - theta.cos()) / (theta * theta))
    };
    let k = [[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]];
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (if i == j { 1. } else { 0. })
                + a * k[i][j]
                + b * (0..3).map(|m| k[i][m] * k[m][j]).sum::<f64>()
        })
    })
}

fn interpolate(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let i = xs
        .partition_point(|v| *v <= x)
        .saturating_sub(1)
        .min(xs.len() - 2);
    ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])
}

fn nan() -> Point {
    Point {
        x: f64::NAN,
        y: f64::NAN,
    }
}

impl GcsMap {
    /// Build a certified camera/surface map, independently of the fitter.
    /// Rectified coordinates retain the principal-point offset, so identity
    /// parameters map pixels to themselves.
    pub fn from_parameters(parameters: GcsParameters, width: u32, height: u32) -> Result<Self> {
        if width < 2
            || height < 2
            || parameters.scale <= 0.
            || parameters.f <= 0.
            || parameters
                .a
                .iter()
                .chain(parameters.rvec.iter())
                .chain([parameters.f, parameters.scale, parameters.cx, parameters.cy].iter())
                .any(|v| !v.is_finite())
        {
            return Err(Error::Invalid("invalid GCS camera or dimensions"));
        }
        let rotation = rodrigues(parameters.rvec);
        let mut b1 = Interval(f64::INFINITY, f64::NEG_INFINITY);
        let mut b3 = Interval(f64::INFINITY, f64::NEG_INFINITY);
        for x in [0., (width - 1) as f64] {
            for y in [0., (height - 1) as f64] {
                let p = [x - parameters.cx, y - parameters.cy, parameters.f];
                let r1 = (0..3).map(|i| p[i] * rotation[i][0]).sum::<f64>();
                let r3 = (0..3).map(|i| p[i] * rotation[i][2]).sum::<f64>();
                b1.0 = b1.0.min(r1);
                b1.1 = b1.1.max(r1);
                b3.0 = b3.0.min(r3);
                b3.1 = b3.1.max(r3);
            }
        }
        let c1 = parameters.f * rotation[2][0];
        let c3 = parameters.f * rotation[2][2];
        let sx = b1.mul(Interval(K_LO, K_HI)).add(-c1);
        let derivative = parameters.range(sx, true).mul(b1).add(-b3.0).1;
        let at_low = parameters
            .range(b1.scale(K_LO).add(-c1), false)
            .add(c3 - K_LO * b3.1)
            .0;
        let at_high = parameters
            .range(b1.scale(K_HI).add(-c1), false)
            .add(c3 - K_HI * b3.0)
            .1;
        // Interval arithmetic covers all image rays and the full positive-depth
        // interval, including points between the sampled validation lattice.
        if derivative >= -0.015 * parameters.f || at_low <= 0. || at_high >= 0. {
            return Err(Error::Invalid(
                "GCS surface lacks a unique full-page positive-depth intersection",
            ));
        }
        // Cropping can move the camera principal point outside the page raster.
        // Keep the complete certified ray interval in the same arc table rather
        // than assuming the optical axis lies inside the crop.
        let span =
            (3. * parameters.scale).max(sx.0.abs()).max(sx.1.abs()) + parameters.scale * 1e-9;
        let arc_x: Vec<_> = (0..=ARC_STEPS)
            .map(|i| -span + 2. * span * i as f64 / ARC_STEPS as f64)
            .collect();
        let mut arc_u = vec![0.; arc_x.len()];
        for i in 1..arc_x.len() {
            let (a, b) = (arc_x[i - 1], arc_x[i]);
            let speed = |x: f64| (1. + parameters.gp(x).powi(2)).sqrt();
            arc_u[i] =
                arc_u[i - 1] + (b - a) / 6. * (speed(a) + 4. * speed((a + b) / 2.) + speed(b));
        }
        let origin = arc_u[ARC_STEPS / 2];
        for u in &mut arc_u {
            *u -= origin;
        }
        let report = GcsReport {
            model: "native_quartic_generalized_cylinder",
            parameters: parameters.clone(),
            initial_cost: 0.,
            final_cost: 0.,
            accepted_iterations: 0,
            rejected_steps: 0,
            staff_groups: 0,
            full_bar_paths: 0,
            edge_paths: 0,
            height_width_ratio: None,
            aspect_constraint: "not supplied",
            camera_principal_point_source: "explicit model parameters",
            roll_initialization_radians: 0.,
            rotation_z_bounds: [-1.2, 1.2],
            paper_pose_initialization: None,
            residual_units: "global median rectified staff spacing; independent constant spacing per staff",
            depth_bounds: [K_LO, K_HI],
            max_ray_equation_derivative: derivative,
            minimum_sampled_jacobian: f64::INFINITY,
            maximum_roundtrip_pixels: 0.,
            validation: "Unique full-domain positive-depth intersection certified by interval bounds; monotone shared arc table. Reconstruction accuracy remains observation-dependent.",
        };
        let mut map = Self {
            parameters,
            rotation,
            arc_x,
            arc_u,
            report,
        };
        for iy in 0..=12 {
            for ix in 0..=12 {
                let p = Point {
                    x: ix as f64 / 12. * (width - 1) as f64,
                    y: iy as f64 / 12. * (height - 1) as f64,
                };
                let q = map.forward(p);
                let back = map.inverse(q);
                let det = map.jacobian_det(p);
                let error = (back.x - p.x).hypot(back.y - p.y);
                if !det.is_finite() || det <= 0. || !error.is_finite() || error > 1e-5 {
                    return Err(Error::Invalid(
                        "GCS map fails orientation or inverse validation",
                    ));
                }
                map.report.minimum_sampled_jacobian = map.report.minimum_sampled_jacobian.min(det);
                map.report.maximum_roundtrip_pixels =
                    map.report.maximum_roundtrip_pixels.max(error);
            }
        }
        Ok(map)
    }
    fn intersect(&self, p: Point) -> Option<[f64; 3]> {
        if !p.x.is_finite() || !p.y.is_finite() {
            return None;
        }
        let params = &self.parameters;
        let ray = [p.x - params.cx, p.y - params.cy, params.f];
        let b: [f64; 3] =
            std::array::from_fn(|j| (0..3).map(|i| ray[i] * self.rotation[i][j]).sum());
        let c: [f64; 3] = std::array::from_fn(|j| params.f * self.rotation[2][j]);
        let phi = |k: f64| params.g(k * b[0] - c[0]) - k * b[2] + c[2];
        let (mut lo, mut hi) = (K_LO, K_HI);
        if phi(lo) < 0. || phi(hi) > 0. {
            return None;
        }
        let mut k = 1.;
        for _ in 0..40 {
            let value = phi(k);
            if value.abs() < params.scale * 1e-12 {
                break;
            }
            if value > 0. {
                lo = k;
            } else {
                hi = k;
            }
            let derivative = params.gp(k * b[0] - c[0]) * b[0] - b[2];
            let next = k - value / derivative;
            k = if next > lo && next < hi {
                next
            } else {
                (lo + hi) / 2.
            };
        }
        let sx = k * b[0] - c[0];
        if sx < self.arc_x[0] || sx > self.arc_x[ARC_STEPS] {
            return None;
        }
        Some([sx, k * b[1] - c[1], k])
    }
    pub fn forward(&self, p: Point) -> Point {
        self.intersect(p).map_or_else(nan, |s| Point {
            x: interpolate(&self.arc_x, &self.arc_u, s[0]) + self.parameters.cx,
            y: s[1] + self.parameters.cy,
        })
    }
    pub fn inverse(&self, p: Point) -> Point {
        if !p.x.is_finite() || !p.y.is_finite() {
            return nan();
        }
        let u = p.x - self.parameters.cx;
        if u < self.arc_u[0] || u > self.arc_u[ARC_STEPS] {
            return nan();
        }
        let x = interpolate(&self.arc_u, &self.arc_x, u);
        let y = p.y - self.parameters.cy;
        let s = [x, y, self.parameters.g(x)];
        let w: [f64; 3] = std::array::from_fn(|i| {
            (0..3).map(|j| self.rotation[i][j] * s[j]).sum::<f64>()
                + if i == 2 { self.parameters.f } else { 0. }
        });
        if w[2] <= 0. {
            return nan();
        }
        Point {
            x: self.parameters.cx + self.parameters.f * w[0] / w[2],
            y: self.parameters.cy + self.parameters.f * w[1] / w[2],
        }
    }
    pub fn jacobian_det(&self, p: Point) -> f64 {
        let Some(s) = self.intersect(p) else {
            return f64::NAN;
        };
        let x = s[0];
        let y = s[1];
        let params = &self.parameters;
        let world = [x, y, params.g(x)];
        let w: [f64; 3] = std::array::from_fn(|i| {
            (0..3).map(|j| self.rotation[i][j] * world[j]).sum::<f64>()
                + if i == 2 { params.f } else { 0. }
        });
        let dx: [f64; 3] =
            std::array::from_fn(|i| self.rotation[i][0] + self.rotation[i][2] * params.gp(x));
        let dy: [f64; 3] = std::array::from_fn(|i| self.rotation[i][1]);
        let px = params.f * (dx[0] * w[2] - w[0] * dx[2]) / w[2].powi(2);
        let py = params.f * (dy[0] * w[2] - w[0] * dy[2]) / w[2].powi(2);
        let qx = params.f * (dx[1] * w[2] - w[1] * dx[2]) / w[2].powi(2);
        let qy = params.f * (dy[1] * w[2] - w[1] * dy[2]) / w[2].powi(2);
        let index = self
            .arc_x
            .partition_point(|v| *v <= x)
            .saturating_sub(1)
            .min(ARC_STEPS - 1);
        let speed = (self.arc_u[index + 1] - self.arc_u[index])
            / (self.arc_x[index + 1] - self.arc_x[index]);
        speed / (px * qy - py * qx)
    }
    pub fn report(&self) -> &GcsReport {
        &self.report
    }

    pub fn fit(
        data: &Observations,
        edges: &[StaffEdge],
        height_width_ratio: Option<f64>,
    ) -> Result<Self> {
        Self::fit_with_paper(data, edges, height_width_ratio, None)
    }

    /// The aspect prior belongs to the measured paper, not the raster/background.
    /// Corners are supplied in the same camera coordinates as observations.
    pub fn fit_with_paper(
        data: &Observations,
        edges: &[StaffEdge],
        height_width_ratio: Option<f64>,
        paper: Option<[Point; 4]>,
    ) -> Result<Self> {
        Self::fit_with_camera(data, edges, height_width_ratio, paper, None)
    }

    /// Physical camera rays use the original sensor principal point shifted by
    /// the source-crop offset. A principal point outside the crop is valid.
    pub fn fit_with_camera(
        data: &Observations,
        edges: &[StaffEdge],
        height_width_ratio: Option<f64>,
        paper: Option<[Point; 4]>,
        principal_point: Option<Point>,
    ) -> Result<Self> {
        data.validate()?;
        if height_width_ratio.is_some_and(|r| !r.is_finite() || r <= 0.) {
            return Err(Error::Invalid("invalid GCS page ratio"));
        }
        if principal_point.is_some_and(|p| !p.x.is_finite() || !p.y.is_finite()) {
            return Err(Error::Invalid("nonfinite GCS camera principal point"));
        }
        if paper.is_some_and(|p| {
            p.iter().any(|p| {
                !p.x.is_finite()
                    || !p.y.is_finite()
                    || p.x < 0.
                    || p.y < 0.
                    || p.x > data.width as f64
                    || p.y > data.height as f64
            })
        }) {
            return Err(Error::Invalid(
                "GCS paper corners must be inside its camera domain",
            ));
        }
        if edges.iter().any(|e| {
            e.staff >= data.staves.len()
                || !e.confidence.is_finite()
                || e.points.iter().any(|p| {
                    !p.x.is_finite()
                        || !p.y.is_finite()
                        || p.x < 0.
                        || p.y < 0.
                        || p.x > (data.width - 1) as f64
                        || p.y > (data.height - 1) as f64
                })
        }) {
            return Err(Error::Invalid("invalid GCS staff edge evidence"));
        }
        let observations = FitData::new(data, edges, height_width_ratio, paper);
        // Model contours describe the continuous pixel-cell edge, which may
        // lie past the final integer pixel centre. Expand the certified camera
        // rectangle to include those points without moving/clamping them or
        // changing the sensor principal point and scale.
        let extra_x =
            u32::from(paper.is_some_and(|p| p.iter().any(|p| p.x > (data.width - 1) as f64)));
        let extra_y =
            u32::from(paper.is_some_and(|p| p.iter().any(|p| p.y > (data.height - 1) as f64)));
        let camera_width = data
            .width
            .checked_add(extra_x)
            .ok_or(Error::Invalid("GCS camera extent overflow"))?;
        let camera_height = data
            .height
            .checked_add(extra_y)
            .ok_or(Error::Invalid("GCS camera extent overflow"))?;
        let mut parameters = GcsParameters::identity(data.width, data.height);
        if let Some(point) = principal_point {
            parameters.cx = point.x;
            parameters.cy = point.y;
        }
        let identity = Self::from_parameters(parameters.clone(), camera_width, camera_height)?;
        let initial = observations.residuals(&identity);
        let mut best = identity;
        let initial_cost = norm(&initial);
        let mut best_cost = initial_cost;
        let mut accepted = 0;
        let mut rejected = 0;
        let roll = staff_roll(data).clamp(-1.2, 1.2);
        let mut seeds = vec![(0., 0.), (0.16, 0.), (-0.16, 0.)];
        if roll.abs() > 1e-3 {
            seeds.extend([(0., roll), (0.16, roll), (-0.16, roll)]);
        }
        let mut starts: Vec<_> = seeds
            .into_iter()
            .map(|(seed, roll_seed)| {
                let mut t = parameters.theta();
                t[1] = seed;
                t[4] = seed * 0.6;
                t[6] = roll_seed;
                t
            })
            .collect();
        let paper_pose = paper
            .zip(height_width_ratio)
            .and_then(|(paper, ratio)| paper_camera_seed(&parameters, paper, ratio));
        if let Some(seed) = &paper_pose {
            starts.push(seed.theta());
        }
        for mut t in starts {
            let Ok(mut map) =
                Self::from_parameters(parameters.with_theta(t), camera_width, camera_height)
            else {
                continue;
            };
            let mut residuals = observations.residuals(&map);
            let mut cost = norm(&residuals);
            let mut lambda = 0.01;
            for _ in 0..55 {
                let mut jac = vec![[0.; N]; residuals.len()];
                for j in 0..N {
                    let h = 2e-4;
                    let mut probe = t;
                    probe[j] += h;
                    let derivative = Self::from_parameters(
                        parameters.with_theta(probe),
                        camera_width,
                        camera_height,
                    )
                    .ok()
                    .map(|p| (observations.residuals(&p), h))
                    .or_else(|| {
                        probe = t;
                        probe[j] -= h;
                        Self::from_parameters(
                            parameters.with_theta(probe),
                            camera_width,
                            camera_height,
                        )
                        .ok()
                        .map(|p| (observations.residuals(&p), -h))
                    });
                    if let Some((r, h)) = derivative {
                        for i in 0..jac.len() {
                            jac[i][j] = (r[i] - residuals[i]) / h;
                        }
                    }
                }
                let mut normal = [[0.; N]; N];
                let mut rhs = [0.; N];
                for (row, r) in jac.iter().zip(&residuals) {
                    for j in 0..N {
                        rhs[j] -= row[j] * r;
                        for k in 0..N {
                            normal[j][k] += row[j] * row[k];
                        }
                    }
                }
                for (j, row) in normal.iter_mut().enumerate() {
                    row[j] += lambda * (row[j] + 1e-5);
                }
                let Some(mut step) = solve(normal, rhs) else {
                    break;
                };
                let length = step.iter().map(|s| s * s).sum::<f64>().sqrt();
                if length > 0.3 {
                    for s in &mut step {
                        *s *= 0.3 / length;
                    }
                }
                if length < 2e-7 {
                    break;
                }
                // These are physical source-camera rotations. The former
                // post-homography ±0.15 roll assumption is invalid here.
                let limits = [0.65, 1.5, 1.5, 1.5, 0.5, 0.65, 1.2, 0.8];
                let candidate =
                    std::array::from_fn(|j| (t[j] + step[j]).clamp(-limits[j], limits[j]));
                let trial = Self::from_parameters(
                    parameters.with_theta(candidate),
                    camera_width,
                    camera_height,
                )
                .ok()
                .map(|m| {
                    let r = observations.residuals(&m);
                    let c = norm(&r);
                    (m, r, c)
                });
                if let Some((m, r, c)) = trial.filter(|(_, _, c)| c.is_finite() && *c < cost - 1e-9)
                {
                    let change = cost - c;
                    t = candidate;
                    map = m;
                    residuals = r;
                    cost = c;
                    accepted += 1;
                    lambda = (lambda * 0.35).max(1e-8);
                    if change < 1e-7 * (1. + cost) {
                        break;
                    }
                } else {
                    rejected += 1;
                    lambda *= 8.;
                    if lambda > 1e10 {
                        break;
                    }
                }
            }
            if cost < best_cost {
                best = map;
                best_cost = cost;
            }
        }
        best.report.initial_cost = initial_cost;
        best.report.final_cost = best_cost;
        best.report.accepted_iterations = accepted;
        best.report.rejected_steps = rejected;
        best.report.staff_groups = data.staves.len();
        best.report.full_bar_paths = data.barlines.len();
        best.report.edge_paths = observations.edges.len();
        best.report.height_width_ratio = height_width_ratio;
        best.report.roll_initialization_radians = roll;
        best.report.paper_pose_initialization = paper_pose.map(|p| p.rvec);
        best.report.aspect_constraint = if height_width_ratio.is_some() && paper.is_some() {
            "actual measured paper corners"
        } else {
            "not applied without measured paper corners"
        };
        best.report.camera_principal_point_source = if principal_point.is_some() {
            "original sensor center adjusted for source crop"
        } else {
            "source raster center assumed"
        };
        Ok(best)
    }
}

/// Orientation is an initialization, never a forced rotation. Weight each
/// staff's five line fits by its observed horizontal span so short fragments
/// cannot outvote long measured staves. Existing zero-roll starts stay in the
/// search and all candidate cameras retain the full-domain safety certificate.
fn staff_roll(data: &Observations) -> f64 {
    let mut angles = vec![];
    for staff in &data.staves {
        let n = staff.x.len() as f64;
        let mx = staff.x.iter().sum::<f64>() / n;
        let xx = staff.x.iter().map(|x| (x - mx).powi(2)).sum::<f64>();
        if xx <= 1e-12 {
            continue;
        }
        for line in &staff.y {
            let my = line.iter().sum::<f64>() / n;
            let slope = staff
                .x
                .iter()
                .zip(line)
                .map(|(x, y)| (x - mx) * (y - my))
                .sum::<f64>()
                / xx;
            angles.push((slope.atan(), staff.x[staff.x.len() - 1] - staff.x[0]));
        }
    }
    angles.sort_by(|a, b| a.0.total_cmp(&b.0));
    let midpoint = angles.iter().map(|p| p.1).sum::<f64>() / 2.;
    let mut weight = 0.;
    for (angle, w) in angles {
        weight += w;
        if weight >= midpoint {
            return angle;
        }
    }
    0.
}

/// A page quadrilateral plus its metric aspect supplies a legitimate camera
/// pose initialization. The homography is used only to estimate the camera
/// axes: observations and final rays stay in the original sensor coordinates.
/// Curved pages need not obey this plane hypothesis, so it is an additional
/// start and never overrides the fitted surface or the safety certificate.
fn paper_camera_seed(
    parameters: &GcsParameters,
    paper: [Point; 4],
    ratio: f64,
) -> Option<GcsParameters> {
    let uv = [
        [-0.5, -ratio / 2.],
        [0.5, -ratio / 2.],
        [0.5, ratio / 2.],
        [-0.5, ratio / 2.],
    ];
    let mut matrix = [[0.; N]; N];
    let mut target = [0.; N];
    for i in 0..4 {
        let (x, y) = (uv[i][0], uv[i][1]);
        let (a, b) = (paper[i].x, paper[i].y);
        matrix[2 * i] = [x, y, 1., 0., 0., 0., -a * x, -a * y];
        target[2 * i] = a;
        matrix[2 * i + 1] = [0., 0., 0., x, y, 1., -b * x, -b * y];
        target[2 * i + 1] = b;
    }
    let h = solve(matrix, target)?;
    let a = [h[0] - parameters.cx * h[6], h[1] - parameters.cx * h[7]];
    let b = [h[3] - parameters.cy * h[6], h[4] - parameters.cy * h[7]];
    let c = [h[6], h[7]];
    let mut focal_squared = vec![];
    for (numerator, denominator) in [
        (-(a[0] * a[1] + b[0] * b[1]), c[0] * c[1]),
        (
            -(a[0] * a[0] + b[0] * b[0] - a[1] * a[1] - b[1] * b[1]),
            c[0] * c[0] - c[1] * c[1],
        ),
    ] {
        if denominator.abs() > 1e-12 {
            let f2 = numerator / denominator;
            if f2.is_finite() && f2 > 0. {
                focal_squared.push(f2);
            }
        }
    }
    let f = if focal_squared.is_empty() {
        parameters.f
    } else {
        (focal_squared.iter().map(|v| v.ln()).sum::<f64>() / focal_squared.len() as f64 / 2.).exp()
    }
    .clamp(
        parameters.scale * (-0.8_f64).exp(),
        parameters.scale * 0.8_f64.exp(),
    );
    let normalize = |v: [f64; 3]| {
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n < 1e-12 {
            None
        } else {
            Some(v.map(|x| x / n))
        }
    };
    let x = normalize([a[0] / f, b[0] / f, c[0]])?;
    let mut y = [a[1] / f, b[1] / f, c[1]];
    let dot = (0..3).map(|i| x[i] * y[i]).sum::<f64>();
    for i in 0..3 {
        y[i] -= dot * x[i];
    }
    let y = normalize(y)?;
    let z = [
        x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2],
        x[0] * y[1] - x[1] * y[0],
    ];
    let r = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]];
    let angle = ((r[0][0] + r[1][1] + r[2][2] - 1.) / 2.)
        .clamp(-1., 1.)
        .acos();
    let factor = if angle < 1e-8 {
        0.5
    } else {
        angle / (2. * angle.sin())
    };
    let pose = [
        (r[2][1] - r[1][2]) * factor,
        (r[0][2] - r[2][0]) * factor,
        (r[1][0] - r[0][1]) * factor,
    ];
    if pose.iter().any(|v| !v.is_finite())
        || pose[0].abs() > 0.5
        || pose[1].abs() > 0.65
        || pose[2].abs() > 1.2
    {
        return None;
    }
    Some(GcsParameters {
        f,
        rvec: pose,
        ..parameters.clone()
    })
}

struct FitData {
    staffs: Vec<[Vec<Point>; 5]>,
    bars: Vec<Vec<Point>>,
    edges: Vec<(Vec<Point>, f64)>,
    space: f64,
    original_y_scale: f64,
    ratio: Option<f64>,
    paper: Option<[Point; 4]>,
}
impl FitData {
    fn new(
        data: &Observations,
        edges: &[StaffEdge],
        ratio: Option<f64>,
        paper: Option<[Point; 4]>,
    ) -> Self {
        let staffs: Vec<[Vec<Point>; 5]> = data
            .staves
            .iter()
            .map(|s| {
                std::array::from_fn(|k| {
                    (0..24)
                        .map(|i| {
                            let x = s.x[0] + (s.x[s.x.len() - 1] - s.x[0]) * i as f64 / 23.;
                            Point {
                                x,
                                y: crate::observations::interpolate(&s.x, &s.y[k], x),
                            }
                        })
                        .collect()
                })
            })
            .collect();
        let all_y: Vec<_> = staffs
            .iter()
            .flat_map(|s| s.iter().flat_map(|l| l.iter().map(|p| p.y)))
            .collect();
        let mut spaces: Vec<_> = staffs
            .iter()
            .flat_map(|s| (0..4).map(|k| s[k + 1][12].y - s[k][12].y))
            .collect();
        spaces.sort_by(f64::total_cmp);
        let space = spaces[spaces.len() / 2];
        let bars = data
            .barlines
            .iter()
            .map(|b| {
                let raw: Vec<_> =
                    b.x.iter()
                        .zip(&b.y)
                        .map(|(x, y)| Point { x: *x, y: *y })
                        .collect();
                sample_path(&raw, 16)
            })
            .collect();
        let edges = edges
            .iter()
            .filter(|e| e.confidence >= 0.7 && e.points.len() >= 3)
            .map(|e| (e.points.clone(), 0.2 * e.confidence))
            .collect();
        Self {
            staffs,
            bars,
            edges,
            space,
            original_y_scale: stddev(&all_y).max(space),
            ratio,
            paper,
        }
    }
    fn residuals(&self, map: &GcsMap) -> Vec<f64> {
        let mut out = vec![];
        let mut all_y = vec![];
        let mut output_spaces = vec![];
        let mut staff_residuals = vec![];
        for staff in &self.staffs {
            let points: [Vec<_>; 5] =
                std::array::from_fn(|k| staff[k].iter().map(|p| map.forward(*p)).collect());
            let mean = points
                .iter()
                .flat_map(|p| p.iter().map(|p| p.y))
                .sum::<f64>()
                / 120.;
            let spacing = points
                .iter()
                .enumerate()
                .map(|(k, p)| p.iter().map(|p| (k as f64 - 2.) * p.y).sum::<f64>())
                .sum::<f64>()
                / 240.;
            let unit = spacing.abs().max(self.space * 0.05);
            output_spaces.push(unit);
            // Five-line order is part of the observation semantics.
            out.push(20. * (spacing / self.space).min(0.));
            for (k, line) in points.iter().enumerate() {
                for p in line {
                    staff_residuals.push(p.y - mean - (k as f64 - 2.) * spacing);
                    all_y.push(p.y);
                }
            }
        }
        let weight =
            (self.staffs.len() as f64 * 120. / (self.bars.len().max(1) * 16) as f64).sqrt();
        output_spaces.sort_by(f64::total_cmp);
        let output_space = output_spaces[output_spaces.len() / 2];
        out.extend(
            staff_residuals
                .into_iter()
                .map(|r| robust(r / output_space)),
        );
        for path in &self.bars {
            let p: Vec<_> = path.iter().map(|p| map.forward(*p)).collect();
            let mean = p.iter().map(|p| p.x).sum::<f64>() / p.len() as f64;
            out.extend(
                p.iter()
                    .map(|p| weight * robust((p.x - mean) / output_space)),
            );
        }
        for (path, w) in &self.edges {
            let p: Vec<_> = path.iter().map(|p| map.forward(*p)).collect();
            let mean = p.iter().map(|p| p.x).sum::<f64>() / p.len() as f64;
            out.extend(p.iter().map(|p| w * robust((p.x - mean) / output_space)));
        }
        // Residuals are dimensionless in rectified staff spaces, preventing a
        // uniform-scale collapse from reducing their cost. Source-camera y
        // variance includes roll and perspective; forcing the rectified page
        // to retain it biases legitimate camera poses. Keep only a weak gauge
        // conditioner, not a physical equality between these two extents.
        out.push(0.2 * (stddev(&all_y) / self.original_y_scale).max(1e-12).ln());
        // Mild camera/surface priors select a bounded shape among weakly
        // identified metric reconstructions, not a preferred page format.
        let theta = map.parameters.theta();
        for (i, t) in theta.iter().enumerate() {
            out.push(
                t * if i == 7 {
                    2.
                } else if i == 0 {
                    0.8
                } else {
                    0.15
                },
            );
        }
        if let (Some(ratio), Some(paper)) = (self.ratio, self.paper) {
            let corners = paper.map(|p| map.forward(p));
            let distance = |a: Point, b: Point| (a.x - b.x).hypot(a.y - b.y);
            let w = (distance(corners[1], corners[0]) + distance(corners[2], corners[3])) / 2.;
            let h = (distance(corners[3], corners[0]) + distance(corners[2], corners[1])) / 2.;
            out.push(3. * ((h / w) / ratio).max(1e-12).ln());
        }
        out
    }
}

fn sample_path(path: &[Point], n: usize) -> Vec<Point> {
    let mut arc = vec![0.];
    for pair in path.windows(2) {
        arc.push(arc.last().unwrap() + (pair[1].x - pair[0].x).hypot(pair[1].y - pair[0].y));
    }
    let length = *arc.last().unwrap();
    if length < 1e-9 {
        return vec![path[0]; n];
    }
    (0..n)
        .map(|i| {
            let at = length * i as f64 / (n - 1) as f64;
            let j = arc
                .partition_point(|v| *v <= at)
                .saturating_sub(1)
                .min(path.len() - 2);
            let t = (at - arc[j]) / (arc[j + 1] - arc[j]).max(1e-12);
            Point {
                x: path[j].x + (path[j + 1].x - path[j].x) * t,
                y: path[j].y + (path[j + 1].y - path[j].y) * t,
            }
        })
        .collect()
}
fn robust(x: f64) -> f64 {
    if x.abs() <= 1.5 {
        x
    } else {
        x.signum() * (3. * x.abs() - 2.25).sqrt()
    }
}
fn stddev(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|v| v * v).sum()
}
fn solve(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let pivot = (col..N).max_by(|i, j| a[*i][col].abs().total_cmp(&a[*j][col].abs()))?;
        if a[pivot][col].abs() < 1e-15 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let d = a[col][col];
        for value in &mut a[col][col..] {
            *value /= d;
        }
        b[col] /= d;
        for i in 0..N {
            if i == col {
                continue;
            }
            let f = a[i][col];
            let pivot_row = a[col];
            for (j, value) in a[i].iter_mut().enumerate().skip(col) {
                *value -= f * pivot_row[j];
            }
            b[i] -= f * b[col];
        }
    }
    Some(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Barline, Staff};
    #[test]
    fn complete_objective_is_invariant_under_uniform_source_scale() {
        let data = Observations {
            width: 1000,
            height: 1400,
            staves: [300., 800.]
                .into_iter()
                .map(|center| {
                    let x = vec![100., 300., 500., 700., 900.];
                    let y = std::array::from_fn(|k| {
                        x.iter()
                            .map(|x| center + 12. * k as f64 + 0.0001_f64 * (x - 500_f64).powi(2))
                            .collect()
                    });
                    Staff { x, y }
                })
                .collect(),
            barlines: vec![Barline {
                staff: 0,
                aligned_staffs: vec![1],
                x: vec![110., 115.],
                y: vec![300., 848.],
            }],
        };
        let mut parameters = GcsParameters::identity(1000, 1400);
        parameters.a[1] = 0.12;
        parameters.rvec = [0.03, -0.08, 0.01];
        let map = GcsMap::from_parameters(parameters.clone(), 1000, 1400).unwrap();
        let a = FitData::new(&data, &[], None, None).residuals(&map);
        let mut twice = data.clone();
        twice.width *= 2;
        twice.height *= 2;
        for staff in &mut twice.staves {
            for x in &mut staff.x {
                *x *= 2.;
            }
            for line in &mut staff.y {
                for y in line {
                    *y *= 2.;
                }
            }
        }
        for bar in &mut twice.barlines {
            for x in &mut bar.x {
                *x *= 2.;
            }
            for y in &mut bar.y {
                *y *= 2.;
            }
        }
        parameters.f *= 2.;
        parameters.scale *= 2.;
        parameters.cx *= 2.;
        parameters.cy *= 2.;
        let map = GcsMap::from_parameters(parameters, 2000, 2800).unwrap();
        let b = FitData::new(&twice, &[], None, None).residuals(&map);
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert!(
                (a - b).abs() < 1e-9,
                "objective depends on arbitrary pixel scale"
            );
        }
    }
}
