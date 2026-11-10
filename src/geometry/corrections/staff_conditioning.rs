//! Condition uncertain geometric measurements, never raster pixels.
//!
//! A five-line regression separates common center/spacing curvature from
//! independent line noise. Robust second differences estimate the latter.
//! Adaptive polynomial fits follow a noise discrepancy principle; coherent
//! bends remain evidence rather than becoming mandatory horizontal targets.
use crate::geometry::{Observations, Staff};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct StaffConditioningReport {
    pub staff: usize,
    pub status: &'static str,
    pub estimated_noise_pixels: f64,
    pub spacing_pixels: f64,
    pub changed_samples: usize,
    pub max_deviation_pixels: f64,
    pub rms_deviation_pixels: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ConditioningReport {
    pub model: &'static str,
    pub staves: Vec<StaffConditioningReport>,
    pub changed_samples: usize,
    pub max_deviation_pixels: f64,
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    (values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.
}

fn solve(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for k in 0..3 {
        let pivot = (k..3).max_by(|i, j| a[*i][k].abs().total_cmp(&a[*j][k].abs()))?;
        if a[pivot][k].abs() < 1e-12 {
            return None;
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        for i in k + 1..3 {
            let t = a[i][k] / a[k][k];
            let row = a[k];
            for (value, reference) in a[i][k..].iter_mut().zip(&row[k..]) {
                *value -= t * reference;
            }
            b[i] -= t * b[k];
        }
    }
    let mut x = [0.; 3];
    for i in (0..3).rev() {
        x[i] = (b[i] - (i + 1..3).map(|j| a[i][j] * x[j]).sum::<f64>()) / a[i][i];
    }
    Some(x)
}

fn smooth_profile(x: &[f64], values: &[f64], noise: f64) -> Vec<f64> {
    if noise <= 1e-12 {
        return values.to_vec();
    }
    let n = x.len();
    let mut sizes = vec![n];
    sizes.extend([65, 33, 17, 9, 5].into_iter().filter(|size| *size < n));
    let mut result = values.to_vec();
    for i in 0..n {
        for size in &sizes {
            if *size < 5 {
                continue;
            }
            let start = i.saturating_sub(size / 2).min(n - size);
            let end = start + size;
            let scale = (x[end - 1] - x[start]).max(1e-12);
            let origin = x[i];
            let mut normal = [[0.; 3]; 3];
            let mut rhs = [0.; 3];
            for j in start..end {
                let t = (x[j] - origin) / scale;
                let row = [1., t, t * t];
                for (r, a) in row.iter().enumerate() {
                    rhs[r] += a * values[j];
                    for (c, b) in row.iter().enumerate() {
                        normal[r][c] += a * b;
                    }
                }
            }
            let Some(coeff) = solve(normal, rhs) else {
                continue;
            };
            let sse = (start..end)
                .map(|j| {
                    let t = (x[j] - origin) / scale;
                    (values[j] - coeff[0] - coeff[1] * t - coeff[2] * t * t).powi(2)
                })
                .sum::<f64>();
            let freedom = (size - 3) as f64;
            // Approximate upper 95% chi-square noise budget. A real bend that
            // cannot be represented by this window forces a smaller support.
            if sse <= noise * noise * (freedom + 2. * (2. * freedom).sqrt()) {
                result[i] = coeff[0];
                break;
            }
        }
    }
    result
}

/// One staff's report and, where its measurements are conditioned, its lines.
fn condition_staff(
    staff: usize,
    source: &Staff,
) -> (StaffConditioningReport, Option<[Vec<f64>; 5]>) {
    let n = source.x.len();
    let mut item = StaffConditioningReport {
        staff,
        status: "insufficient_samples",
        estimated_noise_pixels: 0.,
        spacing_pixels: 0.,
        changed_samples: 0,
        max_deviation_pixels: 0.,
        rms_deviation_pixels: 0.,
    };
    let center: Vec<_> = (0..n)
        .map(|j| source.y.iter().map(|y| y[j]).sum::<f64>() / 5.)
        .collect();
    let spacing: Vec<_> = (0..n)
        .map(|j| {
            source
                .y
                .iter()
                .enumerate()
                .map(|(k, y)| (k as f64 - 2.) * y[j])
                .sum::<f64>()
                / 10.
        })
        .collect();
    item.spacing_pixels = median(spacing.clone());
    if n < 9 || item.spacing_pixels <= 0. {
        return (item, None);
    }
    let residual: [Vec<f64>; 5] = std::array::from_fn(|k| {
        (0..n)
            .map(|j| source.y[k][j] - center[j] - (k as f64 - 2.) * spacing[j])
            .collect()
    });
    let mut differences = vec![];
    for (k, line) in residual.iter().enumerate() {
        let leverage = 0.2 + (k as f64 - 2.).powi(2) / 10.;
        for j in 1..n - 1 {
            let t = (source.x[j] - source.x[j - 1]) / (source.x[j + 1] - source.x[j - 1]);
            let prediction = (1. - t) * line[j - 1] + t * line[j + 1];
            let variance = (1. + (1. - t).powi(2) + t * t) * (1. - leverage);
            differences.push((line[j] - prediction) / variance.sqrt());
        }
    }
    let baseline = median(differences.clone());
    let sigma = median(
        differences
            .into_iter()
            .map(|v| (v - baseline).abs())
            .collect(),
    ) / 0.6744897501960817;
    item.estimated_noise_pixels = sigma;
    // Avoid perturbing coherent noiseless geometry for numerical residue.
    if sigma <= 1e-4 * item.spacing_pixels {
        item.status = "coherent_geometry";
        return (item, None);
    }
    // A very inconsistent five-line packet is not evidence for denoising.
    if sigma > 0.1 * item.spacing_pixels {
        item.status = "uncertainty_exceeds_conditioning_scope";
        return (item, None);
    }
    let centers = smooth_profile(&source.x, &center, sigma / 5_f64.sqrt());
    let spacings = smooth_profile(&source.x, &spacing, sigma / 10_f64.sqrt());
    let residuals: [Vec<f64>; 5] = std::array::from_fn(|k| {
        smooth_profile(
            &source.x,
            &residual[k],
            sigma * (1. - 0.2 - (k as f64 - 2.).powi(2) / 10.).sqrt(),
        )
    });
    let limit = (3. * sigma).min(0.05 * item.spacing_pixels);
    let mut lines = source.y.clone();
    let mut squared = 0.;
    for j in 0..n {
        let candidate: [f64; 5] = std::array::from_fn(|k| {
            let estimate = centers[j] + (k as f64 - 2.) * spacings[j] + residuals[k][j];
            source.y[k][j] + (estimate - source.y[k][j]).clamp(-limit, limit)
        });
        if candidate.windows(2).any(|w| w[1] <= w[0]) {
            continue;
        }
        for (k, value) in candidate.iter().enumerate() {
            let difference = value - source.y[k][j];
            lines[k][j] = *value;
            squared += difference * difference;
            item.max_deviation_pixels = item.max_deviation_pixels.max(difference.abs());
            if difference.abs() > 1e-10 {
                item.changed_samples += 1;
            }
        }
    }
    item.status = "conditioned_measurement_noise";
    item.rms_deviation_pixels = (squared / (5 * n) as f64).sqrt();
    (item, Some(lines))
}

/// New-GCS-pipeline measurement conditioner. Domains, IDs, bars and image
/// dimensions are untouched. Legacy exact `Geometry::fit` does not call this.
pub fn condition(data: &Observations) -> (Observations, ConditioningReport) {
    let mut output = data.clone();
    let mut report = ConditioningReport {
        model: "noise_aware_five_line_profiles",
        staves: vec![],
        changed_samples: 0,
        max_deviation_pixels: 0.,
    };
    if data.validate().is_err() {
        return (output, report);
    }
    // Staves are conditioned independently of each other.
    let staves: Vec<_> = data.staves.iter().enumerate().collect();
    let conditioned =
        crate::parallel::map(&staves, |(staff, source)| condition_staff(*staff, source));
    for ((item, lines), target) in conditioned.into_iter().zip(&mut output.staves) {
        if let Some(lines) = lines {
            target.y = lines;
            report.changed_samples += item.changed_samples;
            report.max_deviation_pixels =
                report.max_deviation_pixels.max(item.max_deviation_pixels);
        }
        report.staves.push(item);
    }
    (output, report)
}
