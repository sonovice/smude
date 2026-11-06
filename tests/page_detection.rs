#![cfg(feature = "render")]
use image::{Rgb, RgbImage};
use smude_geometry::{PageRegion, Point, crop_page, detect_page};

fn polygon(points: &[(f64, f64)]) -> PageRegion {
    PageRegion {
        id: "test".into(),
        contour: points.iter().map(|(x, y)| Point { x: *x, y: *y }).collect(),
        confidence: 1.,
        evidence: vec!["manual regression fixture".into()],
    }
}

fn lines(image: &mut RgbImage, start: u32, end: u32, middle: u32) {
    for x in start..end {
        for k in 0..5 {
            image.put_pixel(x, middle - 24 + 12 * k, Rgb([0; 3]));
        }
    }
}

#[test]
fn page_crop_preserves_pixels_and_uses_local_coordinates_and_relative_support() {
    let mut image = RgbImage::from_pixel(1600, 500, Rgb([245, 235, 210]));
    lines(&mut image, 1060, 1360, 210);
    lines(&mut image, 100, 800, 210);
    let region = polygon(&[(1000., 100.), (1400., 100.), (1400., 400.), (1000., 400.)]);
    let crop = crop_page(&image, &region).unwrap();
    assert_eq!(crop.dimensions(), (400, 300));
    assert_eq!(*crop.get_pixel(70, 110), *image.get_pixel(1070, 210));
    let obs = detect_page(&image, &region, 1800).unwrap();
    assert_eq!((obs.width, obs.height), (400, 300));
    assert_eq!(obs.staves.len(), 1);
    assert!(obs.staves[0].x[0] < 75.);
    assert!(*obs.staves[0].x.last().unwrap() > 345.);
    assert!((obs.staves[0].y[2][50] - 110.).abs() < 1.);
}

#[test]
fn one_pixel_region_barrier_cannot_be_bridged_by_coarse_or_dense_tracking() {
    let mut image = RgbImage::from_pixel(900, 400, Rgb([255; 3]));
    lines(&mut image, 40, 860, 220);
    // A connected page-shaped fixture with a thin excluded slit. It is a
    // deliberately stronger barrier test than two rectangular page crops.
    let region = polygon(&[
        (0., 0.),
        (900., 0.),
        (900., 400.),
        (450.5, 400.),
        (450.5, 80.),
        (449.5, 80.),
        (449.5, 400.),
        (0., 400.),
    ]);
    let obs = detect_page(&image, &region, 1800).unwrap();
    assert_eq!(
        obs.staves.len(),
        2,
        "both sides must retain independent seeds"
    );
    for s in &obs.staves {
        assert!(*s.x.last().unwrap() < 450. || s.x[0] > 450.);
        for (j, x) in s.x.iter().enumerate() {
            for line in &s.y {
                assert!(region.contains(*x, line[j]));
            }
        }
    }
}

#[test]
fn rasterized_concave_crop_matches_contour_membership() {
    let image = RgbImage::from_pixel(180, 180, Rgb([215, 205, 180]));
    let region = polygon(&[
        (10.2, 10.),
        (150., 10.),
        (150., 70.),
        (80.4, 70.),
        (80.4, 150.),
        (10.2, 150.),
    ]);
    let crop = crop_page(&image, &region).unwrap();
    let [x0, y0, _, _] = region.bounds();
    for (x, y, pixel) in crop.enumerate_pixels() {
        let inside = region.contains(f64::from(x + x0), f64::from(y + y0));
        assert_eq!(
            *pixel,
            if inside {
                Rgb([215, 205, 180])
            } else {
                Rgb([255; 3])
            },
            "mask mismatch at ({x},{y})"
        );
    }
}
