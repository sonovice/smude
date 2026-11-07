//! Exact final-raster ↔ original-photo coordinates for linked previews.
use crate::detection::PageRegion;
use crate::geometry::{Geometry, Point};

#[derive(Clone, Debug)]
pub struct PreviewMap {
    geometry: Geometry,
    source_offset: Point,
    source_bounds: [f64; 4],
    page: Option<PageRegion>,
    world_bounds: [f64; 4],
    raster_size: [u32; 2],
    final_crop: [u32; 4],
}
impl PreviewMap {
    pub(super) fn new(
        geometry: Geometry,
        source_offset: Point,
        source_size: [u32; 2],
        page: Option<PageRegion>,
        raster_size: [u32; 2],
        final_crop: [u32; 4],
    ) -> Self {
        let world_bounds = geometry.report().bounds;
        Self {
            geometry,
            source_offset,
            source_bounds: [
                source_offset.x,
                source_offset.y,
                source_offset.x + f64::from(source_size[0]),
                source_offset.y + f64::from(source_size[1]),
            ],
            page,
            world_bounds,
            raster_size,
            final_crop,
        }
    }
    pub fn contains_source(&self, p: Point) -> bool {
        p.x.is_finite()
            && p.y.is_finite()
            && self.page.as_ref().map_or_else(
                || {
                    p.x >= self.source_bounds[0]
                        && p.x <= self.source_bounds[2]
                        && p.y >= self.source_bounds[1]
                        && p.y <= self.source_bounds[3]
                },
                |page| page.contains(p.x, p.y),
            )
    }
    pub fn source_to_result(&self, p: Point) -> Option<Point> {
        let q = self.geometry.forward(Point {
            x: p.x - self.source_offset.x,
            y: p.y - self.source_offset.y,
        });
        let [x0, y0, x1, y1] = self.world_bounds;
        finite(Point {
            x: (q.x - x0) * f64::from(self.raster_size[0] - 1) / (x1 - x0)
                - f64::from(self.final_crop[0]),
            y: (q.y - y0) * f64::from(self.raster_size[1] - 1) / (y1 - y0)
                - f64::from(self.final_crop[1]),
        })
    }
    pub fn result_to_source(&self, p: Point) -> Option<Point> {
        let [x0, y0, x1, y1] = self.world_bounds;
        let q = self.geometry.inverse(Point {
            x: x0
                + (p.x + f64::from(self.final_crop[0])) * (x1 - x0)
                    / f64::from(self.raster_size[0] - 1),
            y: y0
                + (p.y + f64::from(self.final_crop[1])) * (y1 - y0)
                    / f64::from(self.raster_size[1] - 1),
        });
        finite(Point {
            x: q.x + self.source_offset.x,
            y: q.y + self.source_offset.y,
        })
    }
    /// Area-equivalent local pixel scale. This accounts for perspective and
    /// nonlinear deformation without pretending both axes have one exact scale.
    pub fn local_scale_at_source(&self, p: Point) -> Option<f64> {
        let x0 = self.source_to_result(Point { x: p.x - 0.5, ..p })?;
        let x1 = self.source_to_result(Point { x: p.x + 0.5, ..p })?;
        let y0 = self.source_to_result(Point { y: p.y - 0.5, ..p })?;
        let y1 = self.source_to_result(Point { y: p.y + 0.5, ..p })?;
        let det = (x1.x - x0.x) * (y1.y - y0.y) - (x1.y - x0.y) * (y1.x - y0.x);
        (det.is_finite() && det > 1e-12).then(|| det.sqrt())
    }
}
fn finite(p: Point) -> Option<Point> {
    (p.x.is_finite() && p.y.is_finite()).then_some(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Config, Observations, Staff};
    #[test]
    fn nonlinear_music_mapping_roundtrips_with_source_offset_aspect_and_crop() {
        let data = Observations {
            width: 1000,
            height: 700,
            barlines: vec![],
            staves: [180., 440.]
                .into_iter()
                .map(|center| Staff {
                    x: vec![100., 300., 500., 700., 900.],
                    y: std::array::from_fn(|k| {
                        [100., 300., 500., 700., 900.]
                            .into_iter()
                            .map(|x: f64| {
                                center + (k as f64 - 2.) * 12. + 0.0001 * (x - 500.).powi(2)
                            })
                            .collect()
                    }),
                })
                .collect(),
        };
        let geometry = Geometry::fit(
            &data,
            &Config {
                perspective: false,
                ..Default::default()
            },
        )
        .unwrap();
        let bounds = geometry.report().bounds;
        let source = Point {
            x: 340. + 300.,
            y: 170. + 180. + 4.,
        };
        let expected = geometry.forward(Point {
            x: source.x - 340.,
            y: source.y - 170.,
        });
        let map = PreviewMap::new(
            geometry,
            Point { x: 340., y: 170. },
            [1000, 700],
            None,
            [1600, 900],
            [47, 31, 1500, 800],
        );
        let output = map.source_to_result(source).unwrap();
        assert!(
            (output.x - ((expected.x - bounds[0]) * 1599. / (bounds[2] - bounds[0]) - 47.)).abs()
                < 1e-9
        );
        assert!(
            (output.y - ((expected.y - bounds[1]) * 899. / (bounds[3] - bounds[1]) - 31.)).abs()
                < 1e-9
        );
        for y in (80..600).step_by(47) {
            for x in (100..900).step_by(73) {
                let p = Point {
                    x: x as f64 + 340.,
                    y: y as f64 + 170.,
                };
                let q = map
                    .result_to_source(map.source_to_result(p).unwrap())
                    .unwrap();
                assert!((p.x - q.x).hypot(p.y - q.y) < 1e-7);
                assert!(map.local_scale_at_source(p).unwrap() > 0.);
            }
        }
        let a = map.source_to_result(Point { x: 440., y: 366. }).unwrap();
        let b = map.source_to_result(Point { x: 840., y: 350. }).unwrap();
        assert!((a.y - b.y).abs() < 1e-7); // Same bent staff rule becomes horizontal.
        assert!(!map.contains_source(Point { x: 100., y: 100. }));
    }
}
