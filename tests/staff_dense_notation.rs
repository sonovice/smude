#![cfg(feature = "render")]

#[test]
fn ties_do_not_shift_the_staff_packet_to_the_adjacent_line() {
    let image = image::load_from_memory(include_bytes!("fixtures/pages/cropped-score.jpg"))
        .unwrap()
        .to_rgb8();
    let data = smude_geometry::detection::detect(&image, image.width()).unwrap();
    let top = &data.staves[0];
    // Independently checked middle-rule positions, on each side of the tied
    // chords. A one-gap phase slip retains four valid rules and used to pass
    // line-spacing and smoothness checks despite following the wrong packet.
    for (x, y) in [(420., 102.5), (630., 103.5), (770., 101.5)] {
        let i = top
            .x
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (*a - x).abs().total_cmp(&(*b - x).abs()))
            .unwrap()
            .0;
        assert!(
            (top.y[2][i] - y).abs() < 2.,
            "middle rule at x={x}: {}",
            top.y[2][i]
        );
    }
}
#[test]
fn photographed_beamed_staff_remains_one_full_width_observation() {
    let image = image::load_from_memory(include_bytes!("fixtures/staff/dense-marimba.png"))
        .unwrap()
        .to_rgb8();
    let data = smude_geometry::detection::detect(&image, image.width()).unwrap();
    assert_eq!(
        data.staves.len(),
        4,
        "the four printed staves must not split"
    );
    let bottom = data
        .staves
        .iter()
        .max_by(|a, b| a.y[2][a.x.len() / 2].total_cmp(&b.y[2][b.x.len() / 2]))
        .unwrap();
    assert!(bottom.x[0] < 110., "missing left measured rules");
    assert!(
        *bottom.x.last().unwrap() > 600.,
        "missing right measured rules"
    );
    for (j, x) in bottom.x.iter().enumerate() {
        if (220.0..450.0).contains(x) {
            let spacing = (bottom.y[4][j] - bottom.y[0][j]) / 4.;
            assert!(
                (3.0..6.0).contains(&spacing),
                "line identity slipped under beams"
            );
        }
    }
}

#[test]
fn photographed_top_pair_has_its_interior_barline() {
    let image = image::load_from_memory(include_bytes!("fixtures/staff/dense-marimba.png"))
        .unwrap()
        .to_rgb8();
    let data = smude_geometry::detection::detect(&image, image.width()).unwrap();
    assert!(
        data.barlines.iter().any(|b| b.staff == 0
            && b.aligned_staffs.contains(&1)
            && b.x.iter().sum::<f64>() / (b.x.len() as f64) > 245.
            && b.x.iter().sum::<f64>() / (b.x.len() as f64) < 265.),
        "{:?}",
        data.barlines
    );
}

#[test]
fn photographed_tied_chords_keep_staff_identity_and_both_measured_ends() {
    let image = image::load_from_memory(include_bytes!("fixtures/staff/tied-chords.png"))
        .unwrap()
        .to_rgb8();
    let data = smude_geometry::detection::detect(&image, image.width()).unwrap();
    assert_eq!(data.staves.len(), 2);
    let staff = &data.staves[0];
    assert!(
        staff.x[0] < 110.,
        "Left half lost after tie phase slip: {}",
        staff.x[0]
    );
    assert!(*staff.x.last().unwrap() > 1600.);
    for (x, center) in [
        (500., 133.),
        (780., 137.),
        (950., 140.),
        (1100., 141.),
        (1500., 142.),
    ] {
        let i = staff.x.iter().position(|v| *v >= x).unwrap();
        assert!(
            (staff.y[2][i] - center).abs() < 2.,
            "Wrong middle rule at x={x}: {}",
            staff.y[2][i]
        );
        let spacing = (staff.y[4][i] - staff.y[0][i]) / 4.;
        assert!(
            (15.0..18.0).contains(&spacing),
            "Compressed false packet at x={x}: {spacing}"
        );
    }
}
