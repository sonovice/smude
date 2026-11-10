use super::*;
use crate::geometry::{Config, Observations, Staff};
use crate::rendering::RenderBackend;

fn scene() -> (Geometry, RgbImage) {
    let (width, height) = (1280, 960);
    let staves: Vec<_> = [240., 480., 720.]
        .into_iter()
        .map(|center| Staff {
            x: (0..65).map(|i| 40. + 1200. * i as f64 / 64.).collect(),
            y: std::array::from_fn(|k| {
                (0..65)
                    .map(|i| {
                        let x = 40. + 1200. * i as f64 / 64.;
                        center + (k as f64 - 2.) * 17. + 8. * (x / 250.).sin()
                    })
                    .collect()
            }),
        })
        .collect();
    let observations = Observations {
        width,
        height,
        staves,
        barlines: vec![],
    };
    let geometry = Geometry::fit(
        &observations,
        &Config {
            perspective: false,
            ..Config::default()
        },
    )
    .unwrap();
    let source = RgbImage::from_fn(width, height, |x, y| {
        let mut black = false;
        for center in [240., 480., 720.] {
            let local = f64::from(y) - center - 8. * (f64::from(x) / 250.).sin();
            for k in -2..=2 {
                black |= (local - f64::from(k) * 17.).abs() < 0.8;
            }
            let nx = f64::from(x % 120) - 60.;
            black |= nx * nx / 64. + (local - 8.).powi(2) / 25. < 1.;
            black |= x % 120 == 69 && (-42. ..8.).contains(&local);
        }
        let tone = if black {
            12
        } else {
            235 + ((x + y) % 15) as u8
        };
        image::Rgb([tone, tone.saturating_sub(2), tone.saturating_sub(5)])
    });
    (geometry, source)
}

#[test]
fn adaptive_map_preserves_inverse_geometry_on_an_independent_pixel_lattice() {
    let (geometry, _) = scene();
    let (width, height) = geometry.raster_size(None).unwrap();
    let map = map::Map::build(&geometry, width, height).unwrap();
    let [x0, y0, x1, y1] = geometry.report().bounds;
    let mut worst = 0_f64;
    for y in (1..height).step_by(7) {
        for x in (1..width).step_by(11) {
            let p = geometry.inverse(crate::geometry::Point {
                x: x0 + (x1 - x0) * f64::from(x) / f64::from(width - 1),
                y: y0 + (y1 - y0) * f64::from(y) / f64::from(height - 1),
            });
            let mut index = map.roots[(y / 32 * map.columns + x / 32) as usize] as usize;
            let (mut ox, mut oy, mut size) = (x / 32 * 32, y / 32 * 32, 32);
            let actual = loop {
                let n = map.nodes[index];
                if n.mode == 2 {
                    break map.dense[(n.offset + (y - oy) * size + x - ox) as usize];
                }
                if n.mode == 0 {
                    let t = [(x - ox) as f32 / size as f32, (y - oy) as f32 / size as f32];
                    let mix = |a: [f32; 2], b: [f32; 2], t: f32| {
                        std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t)
                    };
                    break mix(
                        mix(n.corners[0], n.corners[1], t[0]),
                        mix(n.corners[2], n.corners[3], t[0]),
                        t[1],
                    );
                }
                size /= 2;
                let dx = u32::from(x >= ox + size);
                let dy = u32::from(y >= oy + size);
                ox += dx * size;
                oy += dy * size;
                index = (n.offset + dx + dy * 2) as usize;
            };
            worst = worst.max((f64::from(actual[0]) - p.x).hypot(f64::from(actual[1]) - p.y));
        }
    }
    assert!(worst <= 0.04, "independent source-pixel error: {worst}");
}

#[test]
#[ignore = "requires a hardware GPU; run explicitly during device validation"]
fn gpu_bicubic_and_binary_formulas_match_cpu_reference() {
    let (geometry, source) = scene();
    let (width, height) = geometry.raster_size(None).unwrap();
    let (reference, _) = geometry
        .render_profiled_at_dimensions(&source, width, height, RenderBackend::Cpu)
        .unwrap();
    let output = super::render(&geometry, &source, width, height, true, true).unwrap();
    eprintln!("{}", serde_json::to_string_pretty(&output.profile).unwrap());
    let color = output.color.unwrap();
    let differences: Vec<_> = reference
        .as_raw()
        .iter()
        .zip(color.as_raw())
        .map(|(a, b)| a.abs_diff(*b))
        .collect();
    let mean = differences.iter().map(|&v| f64::from(v)).sum::<f64>() / differences.len() as f64;
    let maximum = *differences.iter().max().unwrap();
    assert!(mean < 0.3, "mean bicubic difference {mean}, max {maximum}");
    assert!(maximum <= 16, "max bicubic difference {maximum}");
    let (gray, binary) = output.tones.unwrap();
    let expected_gray = image::imageops::grayscale(&color);
    assert_eq!(gray, expected_gray);
    let expected_binary = crate::rendering::output::binarize_sauvola(&gray);
    let mismatches = binary
        .as_raw()
        .iter()
        .zip(expected_binary.as_raw())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        mismatches * 10000 < binary.as_raw().len(),
        "binary threshold mismatches: {mismatches}"
    );
}
