//! Exact capped distance fields, with optional GPU acceleration.
use super::PageBorder;
use rayon::prelude::*;

pub(super) fn gpu_pair_distance(
    prepared: Option<&PageBorder<'_>>,
    a: &[u8],
    b: &[u8],
    w: usize,
    h: usize,
    cap: u8,
) -> Option<(Vec<u8>, Vec<u8>)> {
    let prepared = prepared.filter(|p| p.use_gpu && p.gpu_error.get().is_none())?;
    #[cfg(feature = "gpu-render")]
    {
        let start = std::time::Instant::now();
        let result = match crate::rendering::gpu::distance_pair(a, b, w, h, cap) {
            Ok(result) => result,
            Err(error) => {
                let _ = prepared.gpu_error.set(error);
                return None;
            }
        };
        prepared.record_gpu(start);
        Some(result)
    }
    #[cfg(not(feature = "gpu-render"))]
    {
        let _ = (prepared, a, b, w, h, cap);
        None
    }
}

pub(super) fn gpu_single_distance(
    prepared: Option<&PageBorder<'_>>,
    seeds: &[u16],
    w: usize,
    h: usize,
    cap: u16,
) -> Option<Vec<u16>> {
    let prepared = prepared.filter(|p| p.use_gpu && p.gpu_error.get().is_none())?;
    #[cfg(feature = "gpu-render")]
    {
        let start = std::time::Instant::now();
        let result = match crate::rendering::gpu::distance_single(seeds, w, h, cap) {
            Ok(result) => result,
            Err(error) => {
                let _ = prepared.gpu_error.set(error);
                return None;
            }
        };
        prepared.record_gpu(start);
        Some(result)
    }
    #[cfg(not(feature = "gpu-render"))]
    {
        let _ = (prepared, seeds, w, h, cap);
        None
    }
}

/// Exact capped distance transform in independent row strips. A halo at least
/// the distance cap excludes any influence from seeds beyond a strip, while
/// allowing the serial recurrences to run in parallel without approximations.
pub(super) fn capped_distance<T>(seeds: &[T], w: usize, h: usize, cap: T, halo: usize) -> Vec<T>
where
    T: Copy + Ord + From<u8> + std::ops::Add<Output = T> + Send + Sync,
{
    if w == 0 || h == 0 {
        return seeds.to_vec();
    }
    const ROWS: usize = 256;
    let one = T::from(1);
    let mut output = vec![cap; seeds.len()];
    output
        .par_chunks_mut(w * ROWS)
        .enumerate()
        .for_each(|(strip, out)| {
            let y0 = strip * ROWS;
            let y1 = (y0 + ROWS).min(h);
            let start = y0.saturating_sub(halo);
            let end = (y1 + halo).min(h);
            let height = end - start;
            let mut d = seeds[start * w..end * w].to_vec();
            for y in 0..height {
                for x in 0..w {
                    let i = y * w + x;
                    let mut v = d[i];
                    if x > 0 {
                        v = v.min(d[i - 1] + one);
                    }
                    if y > 0 {
                        v = v.min(d[i - w] + one);
                        if x > 0 {
                            v = v.min(d[i - w - 1] + one);
                        }
                        if x + 1 < w {
                            v = v.min(d[i - w + 1] + one);
                        }
                    }
                    d[i] = v.min(cap);
                }
            }
            for y in (0..height).rev() {
                for x in (0..w).rev() {
                    let i = y * w + x;
                    let mut v = d[i];
                    if x + 1 < w {
                        v = v.min(d[i + 1] + one);
                    }
                    if y + 1 < height {
                        v = v.min(d[i + w] + one);
                        if x > 0 {
                            v = v.min(d[i + w - 1] + one);
                        }
                        if x + 1 < w {
                            v = v.min(d[i + w + 1] + one);
                        }
                    }
                    d[i] = v.min(cap);
                }
            }
            out.copy_from_slice(&d[(y0 - start) * w..(y1 - start) * w]);
        });
    output
}

/// Capped Chebyshev distance seeded from the actual mapped perimeter. The
/// image canvas is deliberately not treated as a physical page boundary.
pub(super) fn distance_to_contour(
    contour: &[crate::geometry::Point],
    w: usize,
    h: usize,
    cap: u16,
    prepared: Option<&PageBorder<'_>>,
) -> Vec<u16> {
    let mut d = vec![cap; w * h];
    for (a, b) in contour
        .iter()
        .zip(contour.iter().cycle().skip(1))
        .take(contour.len())
    {
        let n = (a.x - b.x).abs().max((a.y - b.y).abs()).ceil().max(1.) as usize;
        for j in 0..=n {
            let t = j as f64 / n as f64;
            let x = (a.x + (b.x - a.x) * t).round() as i64;
            let y = (a.y + (b.y - a.y) * t).round() as i64;
            if x >= 0 && y >= 0 && x < w as i64 && y < h as i64 {
                d[y as usize * w + x as usize] = 0;
            }
        }
    }
    gpu_single_distance(prepared, &d, w, h, cap)
        .unwrap_or_else(|| capped_distance(&d, w, h, cap, usize::from(cap)))
}

pub(super) fn rim_distance(
    mask: &[u8],
    contour: &[crate::geometry::Point],
    w: usize,
    h: usize,
    cap: u16,
    prepared: Option<&PageBorder<'_>>,
) -> Vec<u16> {
    let mut distance: Vec<u16> = mask.iter().map(|&v| if v == 0 { 0 } else { cap }).collect();
    // A crop edge can cut through page interior. Seed only the actual mapped
    // perimeter (and confirmed outside mask), never arbitrary canvas edges.
    for (a, b) in contour
        .iter()
        .zip(contour.iter().cycle().skip(1))
        .take(contour.len())
    {
        let n = ((a.x - b.x).abs().max((a.y - b.y).abs()).ceil() as usize).max(1);
        for j in 0..=n {
            let t = j as f64 / n as f64;
            let x = (a.x + (b.x - a.x) * t).round() as i64;
            let y = (a.y + (b.y - a.y) * t).round() as i64;
            if x >= 0 && y >= 0 && x < w as i64 && y < h as i64 {
                distance[y as usize * w + x as usize] = 0;
            }
        }
    }
    gpu_single_distance(prepared, &distance, w, h, cap)
        .unwrap_or_else(|| capped_distance(&distance, w, h, cap, usize::from(cap)))
}
