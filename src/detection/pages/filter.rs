//! Conservative proposal filtering in original-image coordinates.
use crate::detection::PageRegion;
use crate::geometry::Point;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct RejectedPage {
    pub id: String,
    pub reason: &'static str,
    pub duplicate_of: Option<String>,
    pub overlap_fraction_of_smaller: Option<f64>,
}

// A single corner touching the image is insufficient evidence. Require half a
// model edge (12 of 25 samples) against the same original-image boundary.
fn clipped(boundary: &[Point], width: u32, height: u32) -> bool {
    let margin = (f64::from(width.min(height)) * 0.002).max(2.);
    (0..4).any(|side| {
        let on_edge = |p: &Point| match side {
            0 => p.x <= margin,
            1 => p.x >= f64::from(width) - margin,
            2 => p.y <= margin,
            _ => p.y >= f64::from(height) - margin,
        };
        // Cyclic perimeter, so the left edge can cross the array boundary.
        let mut run = 0;
        for p in boundary.iter().cycle().take(boundary.len() + 11) {
            run = if on_edge(p) { run + 1 } else { 0 };
            if run >= 12 {
                return true;
            }
        }
        false
    })
}

pub(super) fn filter(
    regions: &mut Vec<PageRegion>,
    raw: &[Vec<Point>],
    width: u32,
    height: u32,
) -> Vec<RejectedPage> {
    let mut rejected = vec![];
    regions.retain(|region| {
        let index = region
            .id
            .parse::<usize>()
            .expect("generated numeric page ID")
            - 1;
        if clipped(&raw[index], width, height) {
            rejected.push(RejectedPage {
                id: region.id.clone(),
                reason: "clipped_at_image_boundary",
                duplicate_of: None,
                overlap_fraction_of_smaller: None,
            });
            false
        } else {
            true
        }
    });
    // Compare actual polygons, not bounding boxes: adjacent curved pages can
    // have overlapping boxes. Fixed original-coordinate sampling is independent
    // of proposal order, crop dimensions and the eventual warp.
    const GRID: usize = 160;
    let masks: Vec<Vec<bool>> = regions
        .iter()
        .map(|region| {
            (0..GRID * GRID)
                .map(|i| {
                    region.contains(
                        (i % GRID) as f64 / GRID as f64 * f64::from(width)
                            + f64::from(width) / (2. * GRID as f64),
                        (i / GRID) as f64 / GRID as f64 * f64::from(height)
                            + f64::from(height) / (2. * GRID as f64),
                    )
                })
                .collect()
        })
        .collect();
    let areas: Vec<usize> = masks
        .iter()
        .map(|m| m.iter().filter(|v| **v).count())
        .collect();
    let mut order: Vec<usize> = (0..regions.len()).collect();
    order.sort_by_key(|i| std::cmp::Reverse(areas[*i]));
    let mut keep: Vec<usize> = vec![];
    let mut removed = vec![false; regions.len()];
    for index in order {
        for &other in &keep {
            let overlap = masks[index]
                .iter()
                .zip(&masks[other])
                .filter(|(a, b)| **a && **b)
                .count();
            let fraction = overlap as f64 / areas[index].max(1) as f64;
            // Real pages may share a crease, not nearly half their visible area.
            if fraction >= 0.45 {
                removed[index] = true;
                rejected.push(RejectedPage {
                    id: regions[index].id.clone(),
                    reason: "duplicate_page_overlap",
                    duplicate_of: Some(regions[other].id.clone()),
                    overlap_fraction_of_smaller: Some(fraction),
                });
                break;
            }
        }
        if !removed[index] {
            keep.push(index);
        }
    }
    let mut index = 0;
    regions.retain(|_| {
        let accepted = !removed[index];
        index += 1;
        accepted
    });
    rejected
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rectangle(id: &str, x: f64, y: f64, w: f64, h: f64) -> PageRegion {
        let corners = [
            Point { x, y },
            Point { x: x + w, y },
            Point { x: x + w, y: y + h },
            Point { x, y: y + h },
        ];
        let contour = (0..4)
            .flat_map(|side| {
                (0..24).map(move |k| {
                    let a = corners[side];
                    let b = corners[(side + 1) % 4];
                    let t = k as f64 / 24.;
                    Point {
                        x: a.x + (b.x - a.x) * t,
                        y: a.y + (b.y - a.y) * t,
                    }
                })
            })
            .collect();
        PageRegion {
            id: id.into(),
            contour,
            confidence: 0.,
            evidence: vec![],
        }
    }
    #[test]
    fn keeps_distinct_pages_including_shared_crease() {
        let mut pages = vec![
            rectangle("1", 10., 10., 90., 180.),
            rectangle("2", 100., 10., 90., 180.),
        ];
        let raw = pages.iter().map(|p| p.contour.clone()).collect::<Vec<_>>();
        assert!(filter(&mut pages, &raw, 200, 200).is_empty());
        assert_eq!(pages.len(), 2);
    }
    #[test]
    fn discards_duplicate_preserving_larger_and_stable_id() {
        let mut pages = vec![
            rectangle("1", 20., 20., 100., 160.),
            rectangle("2", 65., 20., 110., 160.),
        ];
        let raw = pages.iter().map(|p| p.contour.clone()).collect::<Vec<_>>();
        let rejected = filter(&mut pages, &raw, 200, 200);
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].id, "2");
        assert_eq!(rejected[0].duplicate_of.as_deref(), Some("2"));
    }
    #[test]
    fn filters_sustained_original_edge_contact_but_not_corner_touch() {
        assert!(clipped(
            &rectangle("1", -5., 20., 100., 160.).contour,
            200,
            200
        ));
        assert!(!clipped(
            &[
                Point { x: 0., y: 50. },
                Point { x: 80., y: 0. },
                Point { x: 150., y: 100. },
                Point { x: 70., y: 160. }
            ],
            200,
            200
        ));
    }
    #[test]
    fn all_clipped_proposals_are_rejected_without_fallback() {
        let mut pages = vec![rectangle("1", -5., 20., 100., 160.)];
        let raw = vec![pages[0].contour.clone()];
        assert_eq!(filter(&mut pages, &raw, 200, 200).len(), 1);
        assert!(pages.is_empty());
    }
}
