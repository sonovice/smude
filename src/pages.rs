//! Source-image page polygons and validation, shared by ONNX and manual regions.
use serde::{Deserialize, Serialize};

use crate::{Error, Point, Result};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PageRegion {
    pub id: String,
    pub contour: Vec<Point>,
    /// Uncalibrated proposal strength, not a probability of correctness.
    pub confidence: f64,
    pub evidence: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PageRegions {
    pub width: u32,
    pub height: u32,
    pub regions: Vec<PageRegion>,
}

fn cross(a: Point, b: Point, p: Point) -> f64 {
    (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
}

impl PageRegion {
    /// Boundaries belong to the region; neighbouring pages may share a crease.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        if !x.is_finite() || !y.is_finite() || self.contour.len() < 3 {
            return false;
        }
        let p = Point { x, y };
        let mut inside = false;
        for i in 0..self.contour.len() {
            let a = self.contour[i];
            let b = self.contour[(i + 1) % self.contour.len()];
            if cross(a, b, p).abs() <= 1e-7
                && x >= a.x.min(b.x) - 1e-7
                && x <= a.x.max(b.x) + 1e-7
                && y >= a.y.min(b.y) - 1e-7
                && y <= a.y.max(b.y) + 1e-7
            {
                return true;
            }
            if (a.y > y) != (b.y > y) && x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x {
                inside = !inside;
            }
        }
        inside
    }

    /// Integer crop bounds; right and bottom are exclusive.
    pub fn bounds(&self) -> [u32; 4] {
        let min_x = self
            .contour
            .iter()
            .map(|p| p.x)
            .fold(f64::INFINITY, f64::min);
        let min_y = self
            .contour
            .iter()
            .map(|p| p.y)
            .fold(f64::INFINITY, f64::min);
        let max_x = self.contour.iter().map(|p| p.x).fold(0., f64::max);
        let max_y = self.contour.iter().map(|p| p.y).fold(0., f64::max);
        [
            min_x.floor().max(0.) as u32,
            min_y.floor().max(0.) as u32,
            max_x.ceil().max(0.) as u32,
            max_y.ceil().max(0.) as u32,
        ]
    }
}

impl PageRegions {
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 || self.regions.is_empty() {
            return Err(Error::Invalid(
                "page regions require an image and at least one region",
            ));
        }
        for (j, region) in self.regions.iter().enumerate() {
            if region.id.is_empty()
                || !region
                    .id
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
                || self.regions[..j].iter().any(|p| p.id == region.id)
                || !region.confidence.is_finite()
                || !(0. ..=1.).contains(&region.confidence)
                || region.contour.len() < 3
                || region.contour.iter().any(|p| {
                    !p.x.is_finite()
                        || !p.y.is_finite()
                        || p.x < 0.
                        || p.y < 0.
                        || p.x > f64::from(self.width)
                        || p.y > f64::from(self.height)
                })
            {
                return Err(Error::Invalid(
                    "invalid page ID, confidence or contour coordinates",
                ));
            }
            let n = region.contour.len();
            let mut twice_area = 0.;
            for i in 0..n {
                let a = region.contour[i];
                let b = region.contour[(i + 1) % n];
                if (a.x - b.x).hypot(a.y - b.y) < 1e-7 {
                    return Err(Error::Invalid("page contour has a repeated vertex"));
                }
                twice_area += a.x * b.y - a.y * b.x;
                for k in i + 2..n {
                    if i == 0 && k == n - 1 {
                        continue;
                    }
                    let c = region.contour[k];
                    let d = region.contour[(k + 1) % n];
                    if cross(a, b, c) * cross(a, b, d) < -1e-10
                        && cross(c, d, a) * cross(c, d, b) < -1e-10
                    {
                        return Err(Error::Invalid("page contour self-intersects"));
                    }
                }
            }
            if twice_area.abs() < 2. {
                return Err(Error::Invalid("page contour has no area"));
            }
        }
        Ok(())
    }
}
