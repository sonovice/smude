#![cfg(feature = "render")]

use image::{Rgb, RgbImage};
use smude_geometry::{Config, Geometry, detect};

#[test]
fn native_detection_tracks_five_lines_and_centres_tilted_bar_ink() {
    let mut image = RgbImage::from_pixel(800, 400, Rgb([255; 3]));
    for center in [100_i32, 260] {
        for k in -2..=2 {
            for x in 50..750 {
                for dy in -1..=1 {
                    image.put_pixel(x, (center + k * 12 + dy) as u32, Rgb([0; 3]));
                }
            }
        }
        for (bx, slope) in [(160., 0.), (400., 0.08), (650., -0.08)] {
            for y in center - 24..=center + 24 {
                let x = (bx + slope * (y - center) as f64).round() as u32;
                for dx in 0..3 {
                    image.put_pixel(x + dx, y as u32, Rgb([0; 3]));
                }
            }
        }
    }
    let found = detect(&image, 1800).unwrap();
    assert_eq!(found.staves.len(), 2);
    assert_eq!(found.barlines.len(), 6);
    for bar in &found.barlines {
        let x = bar.x.iter().sum::<f64>() / bar.x.len() as f64;
        let slope = (bar.x.last().unwrap() - bar.x[0]) / (bar.y.last().unwrap() - bar.y[0]);
        let expected = if x < 250. {
            0.
        } else if x < 500. {
            0.08
        } else {
            -0.08
        };
        assert!((slope - expected).abs() < 0.02, "x={x}, slope={slope}");
    }
    assert!(Geometry::fit(&found, &Config::default()).is_ok());
}

#[test]
fn blank_input_fails_explicitly_instead_of_inventing_staves() {
    let image = RgbImage::from_pixel(800, 400, Rgb([255; 3]));
    assert!(detect(&image, 1800).is_err());
}

#[test]
fn cross_staff_harmonics_from_lyric_like_ink_are_not_extra_staves() {
    let mut image = RgbImage::from_pixel(800, 400, Rgb([255; 3]));
    for center in [100_i32, 160, 240, 300] {
        for k in -2..=2 {
            for x in 50..750 {
                image.put_pixel(x, (center + k * 6) as u32, Rgb([0; 3]));
            }
        }
    }
    for y in [88, 107, 126, 145, 164] {
        for x in 250..650 {
            image.put_pixel(x, y, Rgb([0; 3]));
        }
    }
    let found = detect(&image, 1800).unwrap();
    assert_eq!(found.staves.len(), 4);
    assert!(Geometry::fit(&found, &Config::default()).is_ok());
}
