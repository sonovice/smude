//! Visual interpolation only. Native detection and capture eligibility stay raw.
use super::Frame;
use nsh_core::prelude::*;
use std::time::{Duration, Instant};
use std::{cell::Cell, rc::Rc};

pub fn update_overlay(shown: Signal<Frame>, revision: Rc<Cell<u64>>, next: Frame) {
    revision.set(revision.get().wrapping_add(1));
    let token = revision.get();
    let from = shown.get_untracked();
    let Some(target) = aligned_target(&from, next.clone()) else {
        // Never animate stale geometry across orientation, page-count or scene changes.
        shown.set(next);
        return;
    };
    spawn(async move {
        let start = Instant::now();
        loop {
            if revision.get() != token {
                return;
            }
            let t = (start.elapsed().as_secs_f32() / 0.22).min(1.);
            shown.set(interpolate(&from, &target, 1. - (1. - t).powi(3)));
            if t >= 1. {
                return;
            }
            timer(Duration::from_millis(16)).await;
        }
    });
}

fn aligned_target(previous: &Frame, mut next: Frame) -> Option<Frame> {
    if previous.width != next.width
        || previous.height != next.height
        || previous.polygons.is_empty()
        || previous.polygons.len() != next.polygons.len()
    {
        return None;
    }
    // Keep two page identities stable if the detector changes their list order.
    if next.polygons.len() == 2 {
        let center = |p: &Vec<[f32; 2]>| {
            let n = p.len().max(1) as f32;
            [
                p.iter().map(|p| p[0]).sum::<f32>() / n,
                p.iter().map(|p| p[1]).sum::<f32>() / n,
            ]
        };
        let cost =
            |a: usize, b: usize| distance(center(&previous.polygons[a]), center(&next.polygons[b]));
        if cost(0, 1) + cost(1, 0) < cost(0, 0) + cost(1, 1) {
            next.polygons.swap(0, 1);
        }
    }
    for (before, after) in previous.polygons.iter().zip(&mut next.polygons) {
        if before.len() < 3 || before.len() != after.len() {
            return None;
        }
        // A contour's starting vertex/winding must not make its corners cross.
        let n = before.len();
        let mut best = (f32::INFINITY, 0, false);
        for reverse in [false, true] {
            for shift in 0..n {
                let cost = (0..n)
                    .map(|i| distance(before[i], after[index(i, shift, n, reverse)]))
                    .sum::<f32>()
                    / n as f32;
                if cost < best.0 {
                    best = (cost, shift, reverse);
                }
            }
        }
        if !best.0.is_finite() || best.0 > 0.08 {
            return None;
        }
        *after = (0..n).map(|i| after[index(i, best.1, n, best.2)]).collect();
    }
    Some(next)
}
fn index(i: usize, shift: usize, n: usize, reverse: bool) -> usize {
    if reverse {
        (shift + n - i) % n
    } else {
        (shift + i) % n
    }
}
fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
fn interpolate(from: &Frame, to: &Frame, t: f32) -> Frame {
    Frame {
        alignment: match (from.alignment, to.alignment) {
            (Some(a), Some(b)) => Some([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]),
            (_, b) => b,
        },
        width: to.width,
        height: to.height,
        polygons: from
            .polygons
            .iter()
            .zip(&to.polygons)
            .map(|(a, b)| {
                a.iter()
                    .zip(b)
                    .map(|(p, q)| [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t])
                    .collect()
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frame(dx: f32) -> Frame {
        Frame {
            alignment: None,
            width: 480.,
            height: 640.,
            polygons: vec![vec![
                [0.1 + dx, 0.1],
                [0.9 + dx, 0.1],
                [0.9 + dx, 0.9],
                [0.1 + dx, 0.9],
            ]],
        }
    }
    #[test]
    fn updates_move_continuously_and_retarget_from_the_visible_position() {
        let from = frame(0.);
        let to = aligned_target(&from, frame(0.04)).unwrap();
        let middle = interpolate(&from, &to, 0.5);
        assert!((middle.polygons[0][0][0] - 0.12).abs() < 1e-6);
        let newer = aligned_target(&middle, frame(0.06)).unwrap();
        assert_eq!(interpolate(&middle, &newer, 0.), middle);
        assert_eq!(interpolate(&middle, &newer, 1.), newer);
    }
    #[test]
    fn contour_vertex_order_does_not_spin_or_cross_corners() {
        let from = frame(0.);
        let mut to = frame(0.02);
        to.polygons[0].reverse();
        to.polygons[0].rotate_left(2);
        assert_eq!(aligned_target(&from, to).unwrap(), frame(0.02));
    }
    #[test]
    fn lost_pages_rotation_and_large_moves_clear_instead_of_lagging() {
        let from = frame(0.);
        assert!(aligned_target(&from, Frame::default()).is_none());
        let mut rotated = frame(0.);
        rotated.width = 640.;
        rotated.height = 480.;
        assert!(aligned_target(&from, rotated).is_none());
        assert!(aligned_target(&from, frame(0.3)).is_none());
        let mut spread = frame(0.);
        spread.polygons.push(spread.polygons[0].clone());
        assert!(aligned_target(&from, spread).is_none());
    }
}
