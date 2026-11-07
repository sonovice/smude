#![cfg(feature = "render")]
use smude_geometry::detect;
#[test]
fn bass_clef_does_not_shift_the_staff_identity_at_the_start() {
    // Preserve the original coarse-probe x lattice: a narrow x crop changes
    // seed competition and does not reproduce this real clef/phase failure.
    let image = image::load_from_memory(include_bytes!("fixtures/staff-start-clef.png"))
        .unwrap()
        .to_rgb8();
    let found = detect(&image, 1800).unwrap();
    assert_eq!(found.staves.len(), 2);
    let bass = &found.staves[1];
    for (k, line) in bass.y.iter().enumerate() {
        let at = |x: f64| {
            let i = bass
                .x
                .partition_point(|v| *v < x)
                .clamp(1, bass.x.len() - 1);
            line[i - 1]
                + (line[i] - line[i - 1]) * (x - bass.x[i - 1]) / (bass.x[i] - bass.x[i - 1])
        };
        // Independently read rule positions at the beginning of this raster;
        // a consistently wrong five-line phase must not satisfy the test either.
        assert!((at(330.) - [660., 674., 688., 702., 716.][k]).abs() < 3.);
        assert!(
            (at(330.) - at(480.)).abs() < 4.,
            "bass start deviates: {} vs {}",
            at(330.),
            at(480.)
        );
    }
}
