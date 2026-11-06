//! Independent homography/material-coordinate contract for the full native map.
//! The camera below does not call the implementation's camera or GCS model.
#![cfg(feature = "render")]

use smude_geometry::{Barline, Config, Geometry, Observations, Point, Staff};

#[derive(Clone, Copy)]
enum Camera {
    Identity,
    Physical,
    GeneralHomography,
    Cylinder,
}

fn camera(p: Point, mode: Camera) -> Point {
    let u = p.x - 380.;
    let v = p.y - 550.;
    if matches!(mode, Camera::GeneralHomography) {
        let (u, v) = (u / 1100., v / 1100.);
        let z = 1. + 0.13 * u + 0.18 * v;
        Point {
            x: 500. + 1100. * (0.97 * u + 0.025 * v) / z,
            y: 700. + 1100. * (-0.015 * u + 1.02 * v) / z,
        }
    } else if matches!(mode, Camera::Physical | Camera::Cylinder) {
        // Explicit independent Ry -> Rx -> Rz projection. An arbitrary
        // eight-parameter homography need not be realizable by a physical
        // square-pixel camera with the known image-centre principal point.
        let (sx, cx) = 0.10_f64.sin_cos();
        let (sy, cy) = (-0.20_f64).sin_cos();
        let (sz, cz) = 0.025_f64.sin_cos();
        let (u, zs) = if matches!(mode, Camera::Cylinder) {
            let radius = 1350.;
            (
                radius * (u / radius).sin(),
                radius * (1. - (u / radius).cos()),
            )
        } else {
            (u, 0.)
        };
        let x = cy * u + sy * zs;
        let zr = -sy * u + cy * zs;
        let y = cx * v - sx * zr;
        let z = sx * v + cx * zr + 1550.;
        Point {
            x: 500. + 1550. * (cz * x - sz * y) / z,
            y: 700. + 1550. * (sz * x + cz * y) / z,
        }
    } else {
        Point {
            x: p.x + 120.,
            y: p.y + 150.,
        }
    }
}

fn interpolate(points: &[Point], at: f64) -> f64 {
    let i = points
        .partition_point(|p| p.x < at)
        .clamp(1, points.len() - 1);
    let a = points[i - 1];
    let b = points[i];
    a.y + (b.y - a.y) * (at - a.x) / (b.x - a.x)
}

fn observations(mode: Camera) -> Observations {
    let mut staves = vec![];
    let mut bars = vec![];
    let centers = [220., 310., 510., 600., 800., 890.];
    for (i, center) in centers.iter().enumerate() {
        let curves: [Vec<Point>; 5] = std::array::from_fn(|k| {
            (0..257)
                .map(|j| {
                    camera(
                        Point {
                            x: 60. + 640. * j as f64 / 256.,
                            y: center + (k as f64 - 2.) * 10.,
                        },
                        mode,
                    )
                })
                .collect()
        });
        let lo = curves
            .iter()
            .map(|p| p[0].x)
            .fold(f64::NEG_INFINITY, f64::max);
        let hi = curves
            .iter()
            .map(|p| p.last().unwrap().x)
            .fold(f64::INFINITY, f64::min);
        let x: Vec<_> = (0..97)
            .filter(|j| *j < 37 || *j > 56)
            .map(|j| lo + (hi - lo) * j as f64 / 96.)
            .collect();
        let y = std::array::from_fn(|k| x.iter().map(|x| interpolate(&curves[k], *x)).collect());
        staves.push(Staff { x, y });
        if i % 2 == 0 {
            for x in [70., 300., 690.] {
                let p: Vec<_> = (0..65)
                    .map(|j| {
                        camera(
                            Point {
                                x,
                                y: center - 20. + 130. * j as f64 / 64.,
                            },
                            mode,
                        )
                    })
                    .collect();
                bars.push(Barline {
                    staff: i,
                    aligned_staffs: vec![i + 1],
                    x: p.iter().map(|p| p.x).collect(),
                    y: p.iter().map(|p| p.y).collect(),
                });
            }
        }
    }
    Observations {
        width: 1000,
        height: 1400,
        staves,
        barlines: bars,
    }
}

struct Similarity {
    a: f64,
    b: f64,
    source: Point,
    target: Point,
}
impl Similarity {
    fn fit(actual: &[Point], expected: &[Point]) -> Self {
        let mean = |p: &[Point]| Point {
            x: p.iter().map(|p| p.x).sum::<f64>() / p.len() as f64,
            y: p.iter().map(|p| p.y).sum::<f64>() / p.len() as f64,
        };
        let source = mean(actual);
        let target = mean(expected);
        let mut dot = 0.;
        let mut cross = 0.;
        let mut norm = 0.;
        for (p, q) in actual.iter().zip(expected) {
            let (x, y, u, v) = (
                p.x - source.x,
                p.y - source.y,
                q.x - target.x,
                q.y - target.y,
            );
            dot += x * u + y * v;
            cross += x * v - y * u;
            norm += x * x + y * y;
        }
        Self {
            a: dot / norm,
            b: cross / norm,
            source,
            target,
        }
    }
    fn apply(&self, p: Point) -> Point {
        let (x, y) = (p.x - self.source.x, p.y - self.source.y);
        Point {
            x: self.target.x + self.a * x - self.b * y,
            y: self.target.y + self.b * x + self.a * y,
        }
    }
}

fn check_material_geometry(mode: Camera) {
    check_with_crop(mode, false);
}

fn check_with_crop(mode: Camera, cropped: bool) {
    let mut data = observations(mode);
    let offset = if cropped {
        Point { x: 60., y: 80. }
    } else {
        Point { x: 0., y: 0. }
    };
    let project = |p| {
        let q = camera(p, mode);
        Point {
            x: q.x - offset.x,
            y: q.y - offset.y,
        }
    };
    if cropped {
        data.width = 920;
        data.height = 1300;
        for s in &mut data.staves {
            for x in &mut s.x {
                *x -= offset.x;
            }
            for line in &mut s.y {
                for y in line {
                    *y -= offset.y;
                }
            }
        }
        for b in &mut data.barlines {
            for x in &mut b.x {
                *x -= offset.x;
            }
            for y in &mut b.y {
                *y -= offset.y;
            }
        }
    }
    let corners = [
        Point { x: 0., y: 0. },
        Point { x: 760., y: 0. },
        Point { x: 760., y: 1100. },
        Point { x: 0., y: 1100. },
    ]
    .map(project);
    let (geometry, _) = Geometry::fit_profiled_gcs_with_camera(
        &data,
        &Config::default(),
        Some(corners),
        Some(1100. / 760.),
        &[],
        true,
        Some(Point {
            x: 500. - offset.x,
            y: 700. - offset.y,
        }),
    );
    let geometry = geometry.unwrap();
    // Similarity calibration points are not evaluation or glyph samples.
    let calibration: Vec<_> = (0..6)
        .flat_map(|i| {
            (0..7).map(move |j| Point {
                x: 103. + 554. * i as f64 / 5.,
                y: 211. + 698. * j as f64 / 6.,
            })
        })
        .collect();
    let actual: Vec<_> = calibration
        .iter()
        .map(|p| geometry.forward(project(*p)))
        .collect();
    let similarity = Similarity::fit(&actual, &calibration);
    let mut squared = 0.;
    let mut n = 0.;
    for i in 0..33 {
        for j in 0..47 {
            let target = Point {
                x: 12. + 736. * i as f64 / 32.,
                y: 12. + 1076. * j as f64 / 46.,
            };
            let source = project(target);
            let rectified = geometry.forward(source);
            let back = geometry.inverse(rectified);
            assert!(
                (back.x - source.x).hypot(back.y - source.y) < 1e-5,
                "full-map inverse: {source:?} {back:?}"
            );
            let aligned = similarity.apply(rectified);
            squared += (aligned.x - target.x).powi(2) + (aligned.y - target.y).powi(2);
            n += 1.;
            let h = 0.05;
            let dx = geometry.forward(Point {
                x: source.x + h,
                y: source.y,
            });
            let dy = geometry.forward(Point {
                x: source.x,
                y: source.y + h,
            });
            let jac = ((dx.x - rectified.x) * (dy.y - rectified.y)
                - (dx.y - rectified.y) * (dy.x - rectified.x))
                / (h * h);
            assert!(jac > 0., "nonpositive full-page sampled determinant {jac}");
        }
    }
    assert!(
        (squared / n).sqrt()
            < if matches!(mode, Camera::Cylinder) {
                1.5
            } else {
                0.3
            },
        "heldout material RMS {}",
        (squared / n).sqrt()
    );
    for x in [120., 370., 635.] {
        for y in [220., 510., 890.] {
            let points: Vec<_> = (0..65)
                .map(|j| {
                    let angle = std::f64::consts::TAU * j as f64 / 64.;
                    similarity.apply(geometry.forward(project(Point {
                        x: x + 5.5 * angle.cos(),
                        y: y + 3.5 * angle.sin(),
                    })))
                })
                .collect();
            let width = points.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max)
                - points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
            let height = points.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max)
                - points.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
            assert!(
                ((width / height) / (11. / 7.) - 1.).abs()
                    < if matches!(mode, Camera::Cylinder) {
                        0.02
                    } else {
                        0.01
                    },
                "glyph aspect changed: {width}/{height}"
            );
        }
    }
}

#[test]
fn native_full_pipeline_preserves_independent_identity_material_coordinates() {
    check_material_geometry(Camera::Identity);
}

#[test]
fn native_full_pipeline_recovers_independent_perspective_material_coordinates() {
    check_material_geometry(Camera::Physical);
}

#[test]
fn planar_hypothesis_recovers_general_non_square_pixel_homography() {
    check_material_geometry(Camera::GeneralHomography);
}

#[test]
fn isometric_cylinder_preserves_independent_material_and_glyph_metrics() {
    check_with_crop(Camera::Cylinder, false);
}

#[test]
fn asymmetric_page_crop_preserves_camera_calibration_and_glyph_metrics() {
    check_with_crop(Camera::Cylinder, true);
}
