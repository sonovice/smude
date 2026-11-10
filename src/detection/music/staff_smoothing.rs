//! Staff-scale robust local regression of traced ink centres, before geometry
//! or live overlays see them. Short symbol edges are not page curvature.
use crate::math::{median, median_in_place};

fn quadratic(x: &[f64], y: &[f64], weights: &[f64]) -> Option<[f64; 3]> {
    let mut s = [0.; 5];
    let mut t = [0.; 3];
    for ((x, y), w) in x.iter().zip(y).zip(weights) {
        let mut power = *w;
        for j in 0..5 {
            s[j] += power;
            if j < 3 {
                t[j] += power * y;
            }
            power *= x;
        }
    }
    if s[0] <= 1e-12 {
        return None;
    }
    let a = s[2] - s[1] * s[1] / s[0];
    let b = s[3] - s[1] * s[2] / s[0];
    let c = s[4] - s[2] * s[2] / s[0];
    let u = t[1] - s[1] * t[0] / s[0];
    let v = t[2] - s[2] * t[0] / s[0];
    let det = a * c - b * b;
    if det.abs() <= 1e-12 {
        return None;
    }
    let linear = (u * c - v * b) / det;
    let square = (v * a - u * b) / det;
    Some([
        (t[0] - linear * s[1] - square * s[2]) / s[0],
        linear,
        square,
    ])
}

pub(super) fn regularize(track: &mut [(f64, [f64; 5])]) {
    if track.len() < 9 {
        return;
    }
    let spacing = median(track.iter().map(|(_, y)| (y[4] - y[0]) / 4.).collect());
    if spacing <= 0. {
        return;
    }
    // Physical support in staff spaces, independent of analysis resolution,
    // sampling density, page size and the number of systems on the page.
    let radius = (6. * spacing).max(9.);
    let mut source = track.to_vec();
    // A single displaced ink centre is contradicted by the other four rules.
    // Use their equal-spacing geometry as an outlier check, retaining ordinary
    // raster quantization and never flattening a bend shared by the packet.
    for (_, rows) in &mut source {
        let mut gaps = Vec::with_capacity(10);
        for a in 0..5 {
            for b in a + 1..5 {
                gaps.push((rows[b] - rows[a]) / (b - a) as f64);
            }
        }
        let gap = median(gaps);
        let center = median(
            rows.iter()
                .enumerate()
                .map(|(k, y)| y - (k as f64 - 2.) * gap)
                .collect(),
        );
        let tolerance = (0.05 * gap).max(0.5);
        let reliable = rows
            .iter()
            .enumerate()
            .filter(|(k, y)| (**y - center - (*k as f64 - 2.) * gap).abs() <= tolerance)
            .count();
        if reliable >= 4 {
            for (k, y) in rows.iter_mut().enumerate() {
                let expected = center + (k as f64 - 2.) * gap;
                let error = (*y - expected).abs();
                if error > tolerance && error < 0.3 * gap {
                    *y = expected;
                }
            }
        }
    }
    smooth_curves(&source, track, radius, 0.25 * spacing);
}

/// Robust, scale-aware smoothing shared by horizontal rules and complete
/// vertical strokes. Coordinates and measured domains never change.
pub(super) fn smooth_curves<const N: usize>(
    source: &[(f64, [f64; N])],
    target: &mut [(f64, [f64; N])],
    radius: f64,
    max_delta: f64,
) {
    const FITS: usize = 3;
    let (mut x, mut spatial, mut y, mut weights) = (vec![], vec![], vec![], vec![]);
    let (mut residual, mut ranked) = (vec![], vec![]);
    for (origin, output) in target {
        let start = source.partition_point(|(x, _)| *x < *origin - radius);
        let end = source.partition_point(|(x, _)| *x <= *origin + radius);
        let local = &source[start..end];
        if local.len() < 5 {
            continue;
        }
        x.clear();
        x.extend(local.iter().map(|(x, _)| (x - *origin) / radius));
        spatial.clear();
        spatial.extend(
            x.iter()
                .map(|x: &f64| (1. - x.abs().powi(3)).max(0.).powi(3)),
        );
        for (k, value) in output.iter_mut().enumerate() {
            y.clear();
            y.extend(local.iter().map(|(_, y)| y[k]));
            weights.clone_from(&spatial);
            let mut prediction = *value;
            for fit in 1..=FITS {
                let Some(coeff) = quadratic(&x, &y, &weights) else {
                    break;
                };
                prediction = coeff[0];
                // The last fit is final; its residuals would reweight nothing.
                if fit == FITS {
                    break;
                }
                residual.clear();
                residual.extend(
                    x.iter()
                        .zip(&y)
                        .map(|(x, y)| y - coeff[0] - coeff[1] * x - coeff[2] * x * x),
                );
                ranked.clone_from(&residual);
                let center = median_in_place(&mut ranked);
                for (deviation, residual) in ranked.iter_mut().zip(&residual) {
                    *deviation = (residual - center).abs();
                }
                let sigma = (1.4826 * median_in_place(&mut ranked)).max(0.15);
                for ((weight, spatial), residual) in weights.iter_mut().zip(&spatial).zip(&residual)
                {
                    let u = (residual - center) / (4.685 * sigma);
                    *weight = spatial * (1. - u * u).max(0.).powi(2);
                }
            }
            // A questionable local fit cannot relocate a staff by a whole gap.
            *value += (prediction - *value).clamp(-max_delta, max_delta);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn removes_short_independent_wobbles_but_keeps_shared_broad_curvature() {
        let truth =
            |x: f64, k: usize| 100. + k as f64 * 10. + 6. * (x / 90.).sin() + 0.0001 * x * x;
        let mut track: Vec<_> = (0..201)
            .map(|i| {
                let x = i as f64 * 3.;
                (
                    x,
                    std::array::from_fn(|k| {
                        truth(x, k)
                            + 0.15 * (x * 1.1 + k as f64).sin()
                            + if k == 3 {
                                1.7 * (-((x - 300.) / 4.).powi(2)).exp()
                            } else {
                                0.
                            }
                    }),
                )
            })
            .collect();
        regularize(&mut track);
        let error = track
            .iter()
            .flat_map(|(x, y)| {
                y.iter()
                    .enumerate()
                    .map(move |(k, y)| (y - truth(*x, k)).abs())
            })
            .fold(0_f64, f64::max);
        let worst = track
            .iter()
            .flat_map(|(x, y)| {
                y.iter()
                    .enumerate()
                    .map(move |(k, y)| (*x, k, y - truth(*x, k)))
            })
            .max_by(|a, b| a.2.abs().total_cmp(&b.2.abs()))
            .unwrap();
        assert!(error < 0.3, "maximum error={error}, at {worst:?}");
    }
    #[test]
    fn genuine_quadratic_geometry_and_domains_are_preserved_including_ends() {
        let mut track: Vec<_> = (0..101)
            .map(|i| {
                let x = i as f64 * 3.;
                (
                    x,
                    std::array::from_fn(|k| {
                        70. + 0.02 * x + 0.0004 * x * x + k as f64 * (10. + 0.001 * x)
                    }),
                )
            })
            .collect();
        let before = track.clone();
        regularize(&mut track);
        for ((x, y), (bx, by)) in track.iter().zip(before) {
            assert_eq!(*x, bx);
            for (y, by) in y.iter().zip(by) {
                assert!((y - by).abs() < 1e-8);
            }
        }
    }
}
