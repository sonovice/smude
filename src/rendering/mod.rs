//! Rasterize fitted geometry and prepare the final paper-white image.
//! CPU and GPU sampling share the same inverse geometry; output processing
//! never feeds back into detection.
mod cpu;
#[cfg(feature = "gpu-render")]
pub(crate) mod gpu;
pub mod output;
mod profile;
use profile::PreparedOutput;
pub use profile::{RenderBackend, RenderProfile};

use crate::{Error, Result};
use image::RgbImage;

use crate::geometry::{Geometry, Point};

impl Geometry {
    /// Catmull-Rom bicubic reconstruction directly from the original RGB image.
    /// Parallel rows share immutable geometry; no intermediate raster stages.
    pub fn render(&self, image: &RgbImage) -> Result<RgbImage> {
        self.render_with_aspect(image, None)
    }

    /// Raster gauge only: keep the vertical scale, adjust horizontal scale.
    /// The fitted geometric map is unchanged; sample the source once.
    /// Match a physical-page aspect prior to the mapped paper extent rather
    /// than to blank canvas outside a projectively rectified quadrilateral.
    pub fn canvas_ratio_for_page(&self, ratio: f64, contour: &[Point]) -> Result<f64> {
        if contour.len() < 3 || !ratio.is_finite() || !(0.1..=10.).contains(&ratio) {
            return Err(Error::Invalid("invalid page-aspect reference"));
        }
        let points: Vec<_> = contour.iter().map(|p| self.forward(*p)).collect();
        let left = points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let right = points.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        let top = points.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let bottom = points.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
        let [x0, y0, x1, y1] = self.report().bounds;
        if right <= left || bottom <= top {
            return Err(Error::Invalid("empty mapped page extent"));
        }
        let canvas = ratio * (right - left) / (bottom - top) * (y1 - y0) / (x1 - x0);
        if !canvas.is_finite() || !(0.1..=10.).contains(&canvas) {
            return Err(Error::Invalid("implausible page-aspect canvas"));
        }
        Ok(canvas)
    }

    pub fn raster_size(&self, height_width_ratio: Option<f64>) -> Result<(u32, u32)> {
        let [x0, y0, x1, y1] = self.report().bounds;
        let height = (y1 - y0).ceil() as u32 + 1;
        let width = if let Some(ratio) = height_width_ratio {
            if !ratio.is_finite() || !(0.1..=10.).contains(&ratio) {
                return Err(Error::Invalid("invalid output height/width ratio"));
            }
            (f64::from(height) / ratio).round().max(2.) as u32
        } else {
            (x1 - x0).ceil() as u32 + 1
        };
        Ok((width, height))
    }

    pub fn render_with_aspect(
        &self,
        image: &RgbImage,
        height_width_ratio: Option<f64>,
    ) -> Result<RgbImage> {
        let (width, height) = self.raster_size(height_width_ratio)?;
        self.render_profiled_at_dimensions(image, width, height, RenderBackend::Auto)
            .map(|(image, _)| image)
    }

    /// Bounded detector preview; final rendering independently samples source.
    pub fn render_at_dimensions(
        &self,
        image: &RgbImage,
        width: u32,
        height: u32,
    ) -> Result<RgbImage> {
        self.render_profiled_at_dimensions(image, width, height, RenderBackend::Auto)
            .map(|(image, _)| image)
    }

    pub fn render_profiled_at_dimensions(
        &self,
        image: &RgbImage,
        width: u32,
        height: u32,
        backend: RenderBackend,
    ) -> Result<(RgbImage, RenderProfile)> {
        let result = self.render_prepared(image, width, height, backend, false, true)?;
        Ok((result.color.unwrap(), result.profile))
    }

    pub(crate) fn render_prepared(
        &self,
        image: &RgbImage,
        width: u32,
        height: u32,
        backend: RenderBackend,
        tone: bool,
        color: bool,
    ) -> Result<PreparedOutput> {
        self.validate_raster(image, width, height)?;
        let mut profile = RenderProfile {
            backend: "cpu".into(),
            ..RenderProfile::default()
        };
        if backend == RenderBackend::Gpu
            || (backend == RenderBackend::Auto && u64::from(width) * u64::from(height) >= 1_000_000)
        {
            #[cfg(feature = "gpu-render")]
            match crate::rendering::gpu::render(self, image, width, height, tone, color) {
                Ok(output) => {
                    return Ok(PreparedOutput {
                        color: output.color,
                        tones: output.tones,
                        profile: output.profile,
                    });
                }
                Err(error) => {
                    if backend == RenderBackend::Gpu {
                        return Err(Error::Rendering(error));
                    }
                    profile.fallback_reason = Some(error);
                }
            }
            #[cfg(not(feature = "gpu-render"))]
            {
                let _ = (tone, color);
                if backend == RenderBackend::Gpu {
                    return Err(Error::Invalid("GPU renderer was not built"));
                }
                profile.fallback_reason = Some("GPU renderer was not built".into());
            }
        }
        Ok(PreparedOutput {
            color: Some(self.render_cpu_at_dimensions(image, width, height)?),
            tones: None,
            profile,
        })
    }

    fn validate_raster(&self, image: &RgbImage, width: u32, height: u32) -> Result<()> {
        let r = self.report();
        if width < 2 || height < 2 {
            return Err(Error::Invalid("invalid preview dimensions"));
        }
        if image.dimensions() != (r.input_width, r.input_height) {
            return Err(Error::Invalid(
                "image dimensions differ from fitted observations",
            ));
        }
        let pixels = u64::from(width) * u64::from(height);
        if pixels > u64::from(image.width()) * u64::from(image.height()) * 4 {
            return Err(Error::Invalid(
                "output canvas exceeds four times the input area",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detection::TextLine;
    use crate::geometry::fit::Nearby;
    use crate::geometry::{Barline, Config, Observations, Staff};
    #[test]
    fn scanline_inverse_matches_general_composition_with_text_gcs_and_x_correction() {
        let data = Observations {
            width: 1000,
            height: 700,
            staves: [250., 400., 550.]
                .into_iter()
                .map(|center| Staff {
                    x: vec![100., 300., 500., 700., 900.],
                    y: std::array::from_fn(|k| {
                        (0..5)
                            .map(|j| center + (k as f64 - 2.) * 10. + 0.4 * (j as f64).sin())
                            .collect()
                    }),
                })
                .collect(),
            barlines: (0..3)
                .flat_map(|staff| {
                    [150., 450., 850.].map(move |x| Barline {
                        staff,
                        x: vec![x, x + 2.4],
                        y: vec![230. + 150. * staff as f64, 270. + 150. * staff as f64],
                        aligned_staffs: vec![],
                    })
                })
                .collect(),
        };
        for gcs in [false, true] {
            let mut map = if gcs {
                Geometry::fit_profiled_gcs(&data, &Config::default(), None, None, &[], true)
                    .0
                    .unwrap()
            } else {
                Geometry::fit(&data, &Config::default()).unwrap()
            };
            map.add_vertical_correction(&data).unwrap();
            map.add_text_curves(
                &[TextLine {
                    points: (0..51)
                        .map(|i| Point {
                            x: 100. + i as f64 * 16.,
                            y: 110. + 0.04 * i as f64 + 4. * (i as f64 / 12.).sin(),
                        })
                        .collect(),
                    bounds: [100., 100., 900., 122.],
                    block_id: 0,
                    confidence: 0.9,
                    uncertainty_pixels: 0.5,
                }],
                &[[80., 210., 920., 610.]],
            );
            // One state carried across every sample and row, as tiles and
            // rendered rows do, must agree with a fresh one at each sample,
            // and so must a row mapped as a whole.
            let mut nearby = Nearby::default();
            let xs: Vec<_> = (0..141).map(|i| i as f64 * 7.2 - 7.5).collect();
            let mut row = vec![Point { x: 0., y: 0. }; xs.len()];
            for j in 0..241 {
                let y = j as f64 * 3. - 12.5;
                let scanline = map.inverse_scanline(y);
                scanline.fill(&xs, &mut row, &mut nearby);
                for (&x, &filled) in xs.iter().zip(&row) {
                    let b = map.inverse(Point { x, y });
                    for a in [
                        scanline.at(x, &mut Nearby::default()),
                        scanline.at(x, &mut nearby),
                        filled,
                    ] {
                        assert!(
                            a.x.to_bits() == b.x.to_bits() || (a.x.is_nan() && b.x.is_nan()),
                            "gcs={gcs}, {x}, {y}: {a:?} != {b:?}"
                        );
                        assert!(a.y.to_bits() == b.y.to_bits() || (a.y.is_nan() && b.y.is_nan()));
                    }
                }
            }
        }
    }
}
