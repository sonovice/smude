#![cfg(feature = "render")]

use smude_geometry::{PageRegion, PageRegions, Point};

#[test]
fn clipped_page_boundary_and_manual_curved_contour_are_valid() {
    let region = PageRegion {
        id: "manual".into(),
        contour: vec![
            Point { x: 0., y: 0. },
            Point { x: 110., y: 0. },
            Point { x: 100., y: 80. },
            Point { x: 115., y: 160. },
            Point { x: 0., y: 160. },
        ],
        confidence: 1.,
        evidence: vec!["manual".into()],
    };
    assert!(region.contains(90., 80.));
    assert!(!region.contains(110., 80.));
    PageRegions {
        width: 120,
        height: 160,
        regions: vec![region.clone()],
    }
    .validate()
    .unwrap();
    let mut invalid = region;
    invalid.contour[1].x = f64::NAN;
    assert!(
        PageRegions {
            width: 120,
            height: 160,
            regions: vec![invalid]
        }
        .validate()
        .is_err()
    );
}

#[test]
fn self_crossing_or_duplicate_id_regions_are_rejected() {
    let region = PageRegion {
        id: "one".into(),
        contour: vec![
            Point { x: 0., y: 0. },
            Point { x: 100., y: 100. },
            Point { x: 100., y: 0. },
            Point { x: 0., y: 100. },
        ],
        confidence: 1.,
        evidence: vec![],
    };
    assert!(
        PageRegions {
            width: 100,
            height: 100,
            regions: vec![region]
        }
        .validate()
        .is_err()
    );
    let region = PageRegion {
        id: "one".into(),
        contour: vec![
            Point { x: 0., y: 0. },
            Point { x: 100., y: 0. },
            Point { x: 100., y: 100. },
            Point { x: 0., y: 100. },
        ],
        confidence: 1.,
        evidence: vec![],
    };
    assert!(
        PageRegions {
            width: 100,
            height: 100,
            regions: vec![region.clone(), region]
        }
        .validate()
        .is_err()
    );
}
