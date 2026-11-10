//! Internal numerical helpers shared by geometry and detection.
use crate::geometry::Point;

pub(crate) fn median(mut values: Vec<f64>) -> f64 {
    median_in_place(&mut values)
}

/// The median of a sorted copy, by selection: `values` is left partitioned.
pub(crate) fn median_in_place(values: &mut [f64]) -> f64 {
    let n = values.len();
    let (below, middle, _) = values.select_nth_unstable_by(n / 2, f64::total_cmp);
    if n.is_multiple_of(2) {
        let lower = below.iter().copied().max_by(f64::total_cmp);
        (lower.expect("median of an empty slice") + *middle) / 2.
    } else {
        *middle
    }
}

pub(crate) fn slope(points: &[Point]) -> f64 {
    let n = points.len() as f64;
    let x = points.iter().map(|p| p.x).sum::<f64>() / n;
    let y = points.iter().map(|p| p.y).sum::<f64>() / n;
    let den = points.iter().map(|p| (p.y - y).powi(2)).sum::<f64>();
    if den < 1e-12 {
        return 0.;
    }
    points.iter().map(|p| (p.x - x) * (p.y - y)).sum::<f64>() / den
}
