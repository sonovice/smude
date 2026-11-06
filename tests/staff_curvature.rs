#![cfg(feature = "render")]

use image::{Rgb, RgbImage};
use smude_geometry::detect;

#[test]
fn steep_edge_curvature_is_traced_to_the_actual_printed_end() {
    let mut image = RgbImage::from_pixel(1280, 600, Rgb([255; 3]));
    for (center, amplitude) in [(150., 80.), (390., -70.)] {
        for x in 180..=1100 {
            let t = (x - 180) as f64 / 920.;
            let h = center + amplitude * t.powi(8);
            for k in -2..=2 {
                let y = (h + f64::from(k) * 10.).round() as u32;
                for dy in 0..2 {
                    image.put_pixel(x, y + dy, Rgb([0; 3]));
                }
            }
        }
    }
    let data = detect(&image, 1800).unwrap();
    assert_eq!(data.staves.len(), 2);
    for (i, s) in data.staves.iter().enumerate() {
        assert!(
            s.x[0] <= 188. && s.x[s.x.len() - 1] >= 1092.,
            "staff {i}: {}..{}",
            s.x[0],
            s.x[s.x.len() - 1]
        );
        let (center, amplitude) = if i == 0 { (150., 80.) } else { (390., -70.) };
        let errors: Vec<f64> =
            s.x.iter()
                .zip(&s.y[2])
                .map(|(x, y)| (y - (center + amplitude * ((x - 180.) / 920.).powi(8) + 0.5)).abs())
                .collect();
        assert!(errors.iter().copied().fold(0., f64::max) < 1.5);
    }
}

#[test]
fn white_gutter_and_black_page_edge_do_not_extend_music_support() {
    let mut image = RgbImage::from_pixel(1280, 420, Rgb([255; 3]));
    for y in 0..420 {
        for x in 20..45 {
            image.put_pixel(x, y, Rgb([0; 3]));
        }
    }
    for center in [120_i32, 290] {
        for k in -2..=2 {
            for x in (180..=1000).chain(1100..=1279) {
                image.put_pixel(x, (center + k * 10) as u32, Rgb([0; 3]));
            }
        }
    }
    let data = detect(&image, 1800).unwrap();
    assert_eq!(data.staves.len(), 2);
    for staff in data.staves {
        assert!(staff.x[0] >= 172.);
        assert!(*staff.x.last().unwrap() <= 1008., "crossed a white gutter");
    }
}

#[test]
fn parallel_wood_like_background_ridges_are_not_paper_staff_seeds() {
    let mut image = RgbImage::from_pixel(1280, 720, Rgb([145, 100, 60]));
    for y in 50..500 {
        for x in 120..1160 {
            image.put_pixel(x, y, Rgb([245, 244, 235]));
        }
    }
    for center in [150_i32, 350] {
        for k in -2..=2 {
            for x in 180..=1100 {
                image.put_pixel(x, (center + k * 10) as u32, Rgb([0; 3]));
            }
        }
    }
    for k in -2..=2 {
        for x in 10..1260 {
            image.put_pixel(x, (620 + k * 12) as u32, Rgb([35, 25, 15]));
        }
    }
    let data = detect(&image, 1800).unwrap();
    assert_eq!(data.staves.len(), 2);
    assert!(data.staves.iter().all(|s| s.y[2].iter().all(|y| *y < 500.)));
}
