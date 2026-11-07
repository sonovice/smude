//! Cached source-coordinate page masks; only the selected preview owns a texture.
use super::*;

#[derive(Clone)]
struct Cache {
    pages: Vec<Arc<DetectionProgress>>,
    dimensions: [u32; 2],
    texture: egui::TextureHandle,
}

pub(super) fn paint(
    ui: &egui::Ui,
    painter: &egui::Painter,
    image_rect: egui::Rect,
    dimensions: [u32; 2],
    overlays: &[Overlay],
) {
    let pages: Vec<_> = overlays
        .iter()
        .filter(|o| o.data.stage == DetectionStage::Page)
        .collect();
    if pages.is_empty() {
        return;
    }
    let id = ui.id().with("page-shading");
    let previous = ui.data(|d| d.get_temp::<Cache>(id));
    let cache = match previous {
        Some(cache)
            if cache.dimensions == dimensions
                && cache.pages.len() == pages.len()
                && cache
                    .pages
                    .iter()
                    .zip(&pages)
                    .all(|(a, b)| Arc::ptr_eq(a, &b.data)) =>
        {
            cache
        }
        _ => {
            let data: Vec<_> = pages.iter().map(|o| o.data.clone()).collect();
            let mask = page_mask(dimensions, &data);
            let cache = Cache {
                pages: data,
                dimensions,
                texture: ui
                    .ctx()
                    .load_texture("page-shading", mask, egui::TextureOptions::LINEAR),
            };
            ui.data_mut(|d| d.insert_temp(id, cache.clone()));
            cache
        }
    };
    let fade = pages
        .iter()
        .map(|o| (o.arrived.elapsed().as_secs_f32() / 0.25).min(1.))
        .fold(1., f32::min);
    painter.image(
        cache.texture.id(),
        image_rect,
        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1., 1.)),
        Color32::from_white_alpha((fade * 255.) as u8),
    );
}

fn page_mask(dimensions: [u32; 2], pages: &[Arc<DetectionProgress>]) -> egui::ColorImage {
    let scale = (1536. / dimensions[0].max(dimensions[1]) as f64).min(1.);
    let [width, height] = dimensions.map(|d| (f64::from(d) * scale).round().max(1.) as usize);
    let mut mask = egui::ColorImage::filled([width, height], Color32::from_black_alpha(180));
    // Fill the union of curved polygons scanline by scanline. Overlapping pages
    // receive the same alpha, rather than accumulating several dark veils.
    for row in 0..height {
        let y = (row as f64 + 0.5) * f64::from(dimensions[1]) / height as f64;
        for path in pages.iter().flat_map(|p| &p.paths).filter(|p| p.len() >= 3) {
            let mut xs = vec![];
            for (a, b) in path
                .iter()
                .zip(path.iter().cycle().skip(1))
                .take(path.len())
            {
                if (a.y > y) != (b.y > y) {
                    let x = a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y);
                    if x.is_finite() {
                        xs.push(x);
                    }
                }
            }
            xs.sort_by(f64::total_cmp);
            for pair in xs.chunks_exact(2) {
                let index = |x: f64| {
                    ((x * width as f64 / f64::from(dimensions[0]) - 0.5)
                        .ceil()
                        .max(0.) as usize)
                        .min(width)
                };
                let (left, right) = (index(pair[0]), index(pair[1]));
                mask.pixels[row * width + left..row * width + right]
                    .fill(Color32::from_black_alpha(56));
            }
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use smude_geometry::Point;
    fn page(points: &[[f64; 2]]) -> Arc<DetectionProgress> {
        Arc::new(DetectionProgress {
            page_id: "1".into(),
            stage: DetectionStage::Page,
            paths: vec![points.iter().map(|p| Point { x: p[0], y: p[1] }).collect()],
        })
    }
    #[test]
    fn shade_respects_concave_contours_and_union_of_pages() {
        let a = page(&[
            [10., 10.],
            [50., 10.],
            [50., 30.],
            [30., 30.],
            [30., 70.],
            [10., 70.],
        ]);
        let b = page(&[
            [20., 10.],
            [90., 10.],
            [90., 60.],
            [60., 60.],
            [60., 30.],
            [20., 30.],
        ]);
        let mask = page_mask([100, 80], &[a, b]);
        let alpha = |x: usize, y: usize| mask.pixels[y * 100 + x].a();
        assert_eq!(alpha(0, 0), 180);
        assert_eq!(alpha(20, 50), 56);
        assert_eq!(alpha(75, 50), 56);
        assert_eq!(alpha(40, 50), 180);
        assert_eq!(alpha(25, 20), 56); // Overlap is not darker.
    }
}
