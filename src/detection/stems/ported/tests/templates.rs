use super::*;

#[test]
fn the_left_half_of_a_ten_pixel_head_has_two_strokes_and_a_hollow() {
    let t = HalfHeadTemplate::new(HeadSide::Left, 10);
    assert_eq!(t.full.set.len() % 2, 0, "the strokes come in pairs");
    assert!(!t.full.clear.is_empty());
    assert!(t.full.wing.is_none() && t.narrow.wing.is_none());
    assert!(
        t.narrow.set.len() <= t.full.set.len(),
        "the narrow head is smaller"
    );
    let (left, bottom, right, top) = t.extent().unwrap();
    assert!(
        left < 0 && right <= 0 && bottom == 0 && top <= 10,
        "the left half lies left of the origin"
    );
}

#[test]
fn the_wing_lies_right_of_the_head_and_tolerates_more_ink_at_low_resolution() {
    let wing = |space: i32| {
        HalfHeadTemplate::new(HeadSide::Right, space)
            .full
            .wing
            .unwrap()
    };
    let w = wing(16);
    // An eighth of 16 high, half of 16 + 1 wide, from 20 columns and 6 rows on.
    assert_eq!(w.points.len(), 2 * 9);
    assert!(
        w.points
            .iter()
            .all(|p| (20..=28).contains(&p.x) && (6..8).contains(&p.up))
    );
    assert_eq!(w.threshold, 9);
    let low = HalfHeadTemplate::new(HeadSide::Right, 14)
        .full
        .wing
        .unwrap();
    assert_eq!(low.threshold, 8 / 2 + 1);
}

#[test]
fn a_staff_gets_small_templates_and_its_thresholds() {
    let t = OpenHeadTemplates::new(20);
    assert_eq!(
        (t.left.space, t.small_left.space, t.small_right.space),
        (20, 15, 15)
    );
    assert_eq!((t.pass, t.clear_pass), (PASS, CLEAR_PASS));
    assert_eq!(OpenHeadTemplates::new(12).pass, PASS_LOW_RESOLUTION);
}

#[test]
fn tiny_spaces_still_give_templates() {
    for space in 0..4 {
        let t = OpenHeadTemplates::new(space);
        assert!(t.left.extent().is_some() || space == 0);
        assert!(t.right.full.wing.is_some());
    }
}
