//! Internal numerical helpers shared by geometry and detection.
use crate::geometry::Point;

pub(crate) fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n.is_multiple_of(2) {
        (values[n / 2 - 1] + values[n / 2]) / 2.
    } else {
        values[n / 2]
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
