#![cfg(feature = "render")]
use image::{GrayImage, Luma};
use smude_geometry::{
    Barline, Observations, Staff,
    detection::stems::{HeadShape, detect_stems},
};

fn scene() -> (GrayImage, Observations) {
    let mut image = GrayImage::from_pixel(520, 280, Luma([255]));
    for k in -2..=2 {
        for x in 20..500 {
            image.put_pixel(x, (140 + k * 12) as u32, Luma([0]));
        }
    }
    let staff = Staff {
        x: vec![20., 499.],
        y: std::array::from_fn(|k| vec![116. + k as f64 * 12.; 2]),
    };
    (
        image,
        Observations {
            width: 520,
            height: 280,
            staves: vec![staff],
            barlines: vec![],
        },
    )
}
fn note(image: &mut GrayImage, x: f64, y: f64, down: bool, slope: f64, open: bool) {
    let end = if down { y + 42. } else { y - 42. };
    for yy in y.min(end) as i32..=y.max(end) as i32 {
        let xx = (x + slope * (yy as f64 - y)).round() as i32;
        for dx in 0..2 {
            image.put_pixel((xx + dx) as u32, yy as u32, Luma([0]));
        }
    }
    let cx = x + if down { 6. } else { -5. };
    for yy in (y - 7.) as i32..=(y + 7.) as i32 {
        for xx in (cx - 10.) as i32..=(cx + 10.) as i32 {
            let dx = xx as f64 - cx;
            let dy = yy as f64 - y + 0.25 * dx;
            let q = dx * dx / 49. + dy * dy / 25.;
            if q <= 1. && (!open || q >= 0.35) {
                image.put_pixel(xx as u32, yy as u32, Luma([0]));
            }
        }
    }
}
#[test]
fn filled_and_open_heads_support_stems_in_both_directions() {
    let (mut image, data) = scene();
    note(&mut image, 120., 140., false, 0., false);
    note(&mut image, 280., 140., true, 0., true);
    let found = detect_stems(&image, &data).unwrap();
    assert!(
        found.stems.iter().any(|s| s
            .heads
            .iter()
            .any(|h| (h.center.x - 115.).abs() < 8. && h.shape == HeadShape::Filled)),
        "{found:?}"
    );
    assert!(
        found.stems.iter().any(|s| s
            .heads
            .iter()
            .any(|h| (h.center.x - 286.).abs() < 8. && h.shape == HeadShape::Open)),
        "{found:?}"
    );
}
#[test]
fn stem_slope_is_measured_instead_of_replaced_by_a_vertical_line() {
    let (mut image, data) = scene();
    note(&mut image, 180., 140., false, 0.2, false);
    let found = detect_stems(&image, &data).unwrap();
    let stem = found
        .stems
        .iter()
        .find(|s| (s.heads[0].center.x - 175.).abs() < 8.)
        .expect("tilted stem");
    assert!((stem.slope - 0.2).abs() < 0.045, "{stem:?}");
    assert!(stem.points.last().unwrap().x - stem.points[0].x > 3.);
}
#[test]
fn naked_strokes_and_known_barlines_do_not_become_head_supported_stems() {
    let (mut image, mut data) = scene();
    for x in [100, 220, 380] {
        for y in 90..180 {
            for dx in 0..2 {
                image.put_pixel(x + dx, y, Luma([0]));
            }
        }
    }
    data.barlines.push(Barline {
        staff: 0,
        x: vec![380.5; 2],
        y: vec![90., 179.],
        aligned_staffs: vec![],
    });
    let found = detect_stems(&image, &data).unwrap();
    assert!(found.stems.is_empty(), "{found:?}");
}
#[test]
fn beamed_notes_keep_free_shaft_measurements_separate_from_the_beam() {
    let (mut image, data) = scene();
    note(&mut image, 120., 140., false, 0., false);
    note(&mut image, 180., 140., false, 0., false);
    for x in 120..182 {
        for y in 97..102 {
            image.put_pixel(x, y, Luma([0]));
        }
    }
    let found = detect_stems(&image, &data).unwrap();
    assert!(found.stems.len() >= 2, "{found:?}");
    assert!(
        found
            .stems
            .iter()
            .all(|s| s.points.iter().all(|p| p.y >= 102. || p.y < 97.))
    );
}
#[test]
fn image_coordinates_must_match_observation_coordinates() {
    let (_, data) = scene();
    let image = GrayImage::new(10, 10);
    assert!(detect_stems(&image, &data).is_err());
}

#[test]
fn head_like_letter_with_a_short_free_ascender_is_not_geometry_evidence() {
    let (mut image, data) = scene();
    // A d-like glyph: oval at a valid pitch height with an ascender. Total
    // component height can pass the line probe; the free shaft is too short.
    note(&mut image, 180., 140., false, 0., true);
    for y in 98..123 {
        for x in 178..184 {
            image.put_pixel(x, y, Luma([255]));
        }
    }
    let found = detect_stems(&image, &data).unwrap();
    assert!(found.stems.iter().all(|s| !s.geometry_usable), "{found:?}");
}

#[test]
fn shared_head_does_not_merge_two_distinct_opposite_stems() {
    let (mut image, data) = scene();
    note(&mut image, 180., 140., true, 0., false);
    note(&mut image, 192., 140., false, 0., false);
    let found = detect_stems(&image, &data).unwrap();
    assert!(
        found.stems.iter().any(|s| s.points[0].x < 185.),
        "{found:?}"
    );
    assert!(
        found.stems.iter().any(|s| s.points[0].x > 187.),
        "{found:?}"
    );
}

#[test]
fn three_note_chord_reports_all_attached_heads() {
    let (mut image, data) = scene();
    for y in 85..153 {
        for x in 180..182 {
            image.put_pixel(x, y, Luma([0]));
        }
    }
    for cy in [128., 140., 152.] {
        for y in (cy - 7.) as i32..=(cy + 7.) as i32 {
            for x in 165..185 {
                let dx = f64::from(x) - 175.;
                let dy = f64::from(y) - cy + 0.25 * dx;
                if dx * dx / 49. + dy * dy / 16. <= 1. {
                    image.put_pixel(x as u32, y as u32, Luma([0]));
                }
            }
        }
    }
    let found = detect_stems(&image, &data).unwrap();
    // Count via serialization so the regression also demonstrates that the old
    // single-head output cannot represent this chord at all.
    let v = serde_json::to_value(&found).unwrap();
    let count = v["stems"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|s| s["heads"].as_array())
        .map(Vec::len)
        .sum::<usize>();
    assert!(count >= 3, "chord head count {count}: {found:?}");
}
#[test]
fn multiple_beams_do_not_require_one_long_uninterrupted_free_run() {
    let (mut image, data) = scene();
    note(&mut image, 120., 140., false, 0., false);
    note(&mut image, 180., 140., false, 0., false);
    for x in 120..182 {
        for y0 in [98, 109, 120] {
            for y in y0..y0 + 4 {
                image.put_pixel(x, y, Luma([0]));
            }
        }
    }
    let found = detect_stems(&image, &data).unwrap();
    assert!(found.stems.len() >= 2, "{found:?}");
}

#[test]
fn photographed_clef_rests_and_accidentals_are_not_stem_geometry() {
    let image = image::load_from_memory(include_bytes!("fixtures/stems/top-staff.png"))
        .unwrap()
        .into_luma8();
    let data: Observations =
        serde_json::from_str(include_str!("fixtures/stems/top-staff.json")).unwrap();
    let found = detect_stems(&image, &data).unwrap();
    for (left, right) in [
        (345., 370.),
        (395., 410.),
        (835., 850.),
        (1024., 1042.),
        (1395., 1470.),
        (1625., 1644.),
    ] {
        assert!(
            found.stems.iter().all(|s| {
                let x = s.points[0].x;
                !(left..right).contains(&x)
            }),
            "non-note at {left}..{right}: {found:?}"
        );
    }
    for x in [770., 826., 899., 965., 1103., 1199., 1365., 1571.] {
        assert!(
            found.stems.iter().any(|s| (s.points[0].x - x).abs() < 5.),
            "lost real stem near {x}: {found:?}"
        );
    }
}

#[test]
fn overlapping_clef_fragments_cannot_promote_each_others_heads_to_geometry() {
    let image = image::load_from_memory(include_bytes!("fixtures/stems/last-staff.png"))
        .unwrap()
        .into_luma8();
    let data: Observations =
        serde_json::from_str(include_str!("fixtures/stems/last-staff.json")).unwrap();
    let found = detect_stems(&image, &data).unwrap();
    assert!(
        found
            .stems
            .iter()
            .filter(|s| s.points[0].x < 245.)
            .all(|s| !s.geometry_usable),
        "clef promoted to geometry: {found:?}"
    );
    assert!(
        found
            .vertical_evidence(&data)
            .iter()
            .all(|p| p.points[0].x >= 245.)
    );
    assert!(
        found
            .stems
            .iter()
            .filter(|s| s.points[0].x > 300. && s.geometry_usable)
            .count()
            >= 5,
        "lost real music shafts: {found:?}"
    );
}
