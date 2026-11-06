#![cfg(feature = "render")]
use image::{Rgb, RgbImage};
use smude_geometry::{
    Config, Geometry, Observations, Point, Staff, TextLine, antialias_print, normalize_paper,
};
#[test]
fn composed_text_warp_keeps_rendered_and_normalized_music_identical() {
    check_text_music_safety(false);
}
#[test]
fn native_gcs_and_x_correction_keep_text_guards_through_render_and_grayscale() {
    check_text_music_safety(true);
}
fn check_text_music_safety(native_gcs: bool) {
    let data = Observations {
        width: 1000,
        height: 700,
        staves: vec![Staff {
            x: vec![100., 900.],
            y: std::array::from_fn(|k| vec![400. + k as f64 * 10.; 2]),
        }],
        barlines: vec![],
    };
    let base = if native_gcs {
        Geometry::fit_profiled_gcs(&data, &Config::default(), None, None, &[], true)
            .0
            .unwrap()
    } else {
        Geometry::fit(
            &data,
            &Config {
                perspective: false,
                ..Config::default()
            },
        )
        .unwrap()
    };
    let mut text = base.clone();
    let lines: Vec<_> = (0..4)
        .map(|k| TextLine {
            points: (0..81)
                .map(|i| {
                    let x = 100. + i as f64 * 10.;
                    Point {
                        x,
                        y: 140.
                            + k as f64 * 32.
                            + 15. * (std::f64::consts::PI * (x - 100.) / 800.).sin(),
                    }
                })
                .collect(),
            bounds: [100., 132. + k as f64 * 32., 900., 160. + k as f64 * 32.],
            block_id: 0,
            confidence: 0.9,
            uncertainty_pixels: 1.,
        })
        .collect();
    // A tone window reaches25px beyond its pixel; leave an additional AA margin.
    text.add_text_curves(&lines, &[[19., 309., 981., 631.]]);
    assert!(
        text.report()
            .text_warp
            .as_ref()
            .unwrap()
            .max_displacement_pixels
            > 0.
    );
    let image = RgbImage::from_fn(1000, 700, |x, y| {
        let bend = (15. * (std::f64::consts::PI * (x as f64 - 100.) / 800.).sin()).round() as i32;
        let glyph = (100..=900).contains(&x)
            && x % 10 < 5
            && (0..4).any(|k| (y as i32 - (140 + k * 32 + bend)).abs() < 4);
        let music = (100..=900).contains(&x) && [400, 410, 420, 430, 440].contains(&y);
        Rgb([if glyph || music { 20 } else { 255 }; 3])
    });
    let a = base.render(&image).unwrap();
    let b = text.render(&image).unwrap();
    assert_ne!(a, b);
    let ag = antialias_print(&normalize_paper(&image::imageops::grayscale(&a)));
    let bg = antialias_print(&normalize_paper(&image::imageops::grayscale(&b)));
    for y in 340..=600 {
        for x in 50..=950 {
            assert_eq!(a.get_pixel(x, y), b.get_pixel(x, y));
            assert_eq!(ag.get_pixel(x, y), bg.get_pixel(x, y));
        }
    }
}
