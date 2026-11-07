//! Output regressions exercise the public operations and exact distance primitives.
use super::border::*;
use super::distance::*;
use super::*;

#[cfg(test)]
mod distance_and_cache {
    #[test]
    fn parallel_distance_strips_match_exact_chebyshev_distance_across_halos() {
        let (w, h) = (17_usize, 600_usize);
        let points = [(7_usize, 255_usize), (3, 267), (12, 516), (1, 0), (15, 599)];
        for cap in [1_u16, 11, 65, 125] {
            let mut seeds = vec![cap; w * h];
            for (x, y) in points {
                seeds[y * w + x] = 0;
            }
            let actual = capped_distance(&seeds, w, h, cap, usize::from(cap));
            for y in 0..h {
                for x in 0..w {
                    let expected = points
                        .iter()
                        .map(|(a, b)| x.abs_diff(*a).max(y.abs_diff(*b)))
                        .min()
                        .unwrap()
                        .min(usize::from(cap)) as u16;
                    assert_eq!(actual[y * w + x], expected, "cap={cap}, ({x},{y})");
                }
            }
        }
    }
    #[test]
    fn shared_border_geometry_preserves_uncached_cleanup_pixels() {
        for (w, h) in [(64, 83), (257, 331)] {
            let gray = GrayImage::from_fn(w, h, |x, y| {
                let rim = x < 7 || x + 9 > w || y < 5 || y + 8 > h;
                let ink = (x * 17 + y * 31) % 97 < 7;
                image::Luma([if rim {
                    170 + ((x + y) % 70) as u8
                } else if ink {
                    35
                } else {
                    255
                }])
            });
            let contour = [
                crate::geometry::Point { x: 2., y: 2. },
                crate::geometry::Point {
                    x: f64::from(w) - 3.,
                    y: 4.,
                },
                crate::geometry::Point {
                    x: f64::from(w) - 5.,
                    y: f64::from(h) - 2.,
                },
                crate::geometry::Point {
                    x: 1.,
                    y: f64::from(h) - 5.,
                },
            ];
            let protected = [[5., 20., f64::from(w) - 10., 35.]];
            let raw = binarize_sauvola(&gray);
            let mut expected_binary = raw.clone();
            let expected_count = whiten_page_border(&mut expected_binary, &contour, 2);
            let mut expected = gray.clone();
            apply_binary_cleanup_with_contour(&mut expected, &raw, &expected_binary, &contour);
            whiten_gray_border(&mut expected, &contour, 2, &protected);
            let border = PageBorder::new(&contour);
            let mut actual_binary = raw.clone();
            assert_eq!(border.whiten_binary(&mut actual_binary, 2), expected_count);
            let mut actual = gray;
            border.clean_gray(&mut actual, &raw, &actual_binary, 2, &protected);
            assert_eq!(actual_binary, expected_binary);
            assert_eq!(actual, expected);
        }
    }
    #[test]
    fn rasterized_music_guards_match_inclusive_float_membership() {
        let rectangles = [
            [-2.5, 1.2, 12.8, 18.],
            [8., 12., 30., 30.],
            [34.1, -1., 38.9, 2.5],
            [4., 4., 3., 7.],
            [f64::NAN, 0., 50., 50.],
            [0., 20., f64::INFINITY, 21.],
        ];
        let mask = super::protected_pixel_mask(40, 32, &rectangles);
        for y in 0..32 {
            for x in 0..40 {
                let expected = rectangles.iter().any(|r| {
                    x as f64 >= r[0] && x as f64 <= r[2] && y as f64 >= r[1] && y as f64 <= r[3]
                });
                assert_eq!(mask[y * 40 + x] != 0, expected, "{x}, {y}");
            }
        }
    }

    use super::*;
    use image::Luma;

    #[test]
    fn shaded_ink_matches_direct_replicated_window_reference() {
        let gray = GrayImage::from_fn(57, 39, |x, y| {
            Luma([
                if (8..=48).contains(&x) && [6, 12, 18, 24, 30].contains(&y) {
                    18
                } else {
                    (90 + x * 2 + y / 2) as u8
                },
            ])
        });
        let binary = binarize_sauvola(&gray);
        for y in 0..gray.height() {
            for x in 0..gray.width() {
                let (mut total, mut squared) = (0., 0.);
                for dy in -25_i64..=25 {
                    for dx in -25_i64..=25 {
                        let xx = (i64::from(x) + dx).clamp(0, i64::from(gray.width()) - 1) as u32;
                        let yy = (i64::from(y) + dy).clamp(0, i64::from(gray.height()) - 1) as u32;
                        let v = f64::from(gray.get_pixel(xx, yy)[0]);
                        total += v;
                        squared += v * v;
                    }
                }
                let m = total / 2601.;
                let sd = (squared / 2601. - m * m).max(0.).sqrt();
                let expected =
                    if f64::from(gray.get_pixel(x, y)[0]) <= m * (1. + 0.25 * (sd / 128. - 1.)) {
                        0
                    } else {
                        255
                    };
                assert_eq!(binary.get_pixel(x, y)[0], expected, "{x},{y}");
            }
        }
        assert_eq!(binary.dimensions(), gray.dimensions());
        assert_eq!(binary.get_pixel(20, 18)[0], 0);
        assert_eq!(binary.get_pixel(20, 19)[0], 255);
    }

    #[test]
    fn tiny_constant_images_have_no_border_artifacts() {
        for tone in [0, 60, 128, 255] {
            let binary = binarize_sauvola(&GrayImage::from_pixel(2, 3, Luma([tone])));
            assert!(
                binary
                    .as_raw()
                    .iter()
                    .all(|p| *p == if tone == 0 { 0 } else { 255 })
            );
        }
    }
}

#[cfg(test)]
mod border_tests {
    use super::*;
    use crate::geometry::Point;
    use image::Luma;

    #[test]
    fn masks_outside_and_thin_border_without_flooding_attached_content() {
        let contour = vec![
            Point { x: 3., y: 3. },
            Point { x: 36., y: 3. },
            Point { x: 36., y: 36. },
            Point { x: 3., y: 36. },
        ];
        let mut binary = GrayImage::from_pixel(40, 40, Luma([255]));
        // A border stroke is connected to an interior symbol. Only the portion
        // in the excluded band must disappear, not the entire component.
        for x in 0..20 {
            binary.put_pixel(x, 20, Luma([0]));
        }
        binary.put_pixel(20, 22, Luma([0]));
        let count = whiten_page_border(&mut binary, &contour, 2);
        assert!(count > 0);
        for x in 0..5 {
            assert_eq!(binary.get_pixel(x, 20)[0], 255);
        }
        for x in 5..20 {
            assert_eq!(binary.get_pixel(x, 20)[0], 0);
        }
        assert_eq!(binary.get_pixel(20, 22)[0], 0);
        assert_eq!(binary.dimensions(), (40, 40));
    }

    #[test]
    fn curved_polygon_interior_and_zero_margin_are_respected() {
        let contour = vec![
            Point { x: 15., y: 2. },
            Point { x: 28., y: 15. },
            Point { x: 15., y: 28. },
            Point { x: 2., y: 15. },
        ];
        let mut binary = GrayImage::from_pixel(30, 30, Luma([0]));
        whiten_page_border(&mut binary, &contour, 0);
        assert_eq!(binary.get_pixel(15, 15)[0], 0);
        assert_eq!(binary.get_pixel(3, 3)[0], 255);
        assert_eq!(binary.get_pixel(26, 26)[0], 255);
    }
}

#[cfg(test)]
mod rim_regression {
    use super::*;
    use crate::geometry::Point;
    use image::Luma;

    #[test]
    fn short_curved_corner_frame_is_removed_as_one_object() {
        let contour = vec![
            Point { x: 0., y: 0. },
            Point { x: 599., y: 0. },
            Point { x: 599., y: 799. },
            Point { x: 0., y: 799. },
        ];
        let mut binary = GrayImage::from_pixel(600, 800, Luma([255]));
        for y in 700..783 {
            for x in 10..13 {
                binary.put_pixel(x, y, Luma([0]));
            }
        }
        for x in 10..90 {
            for y in 780..783 {
                binary.put_pixel(x, y, Luma([0]));
            }
        }
        whiten_page_border(&mut binary, &contour, 2);
        assert!(binary.as_raw().iter().all(|v| *v == 255));
    }

    #[test]
    fn short_fragment_is_removed_only_with_confirmed_aligned_rim_support() {
        let contour = vec![
            Point { x: 0., y: 0. },
            Point { x: 599., y: 0. },
            Point { x: 599., y: 799. },
            Point { x: 0., y: 799. },
        ];
        let mut binary = GrayImage::from_pixel(600, 800, Luma([255]));
        for y in 100..700 {
            binary.put_pixel(590, y, Luma([0]));
        }
        for y in 730..744 {
            binary.put_pixel(590, y, Luma([0]));
        }
        for y in 300..314 {
            binary.put_pixel(7, y, Luma([0]));
        }
        whiten_page_border(&mut binary, &contour, 4);
        for y in 100..700 {
            assert_eq!(binary.get_pixel(590, y)[0], 255);
        }
        for y in 730..744 {
            assert_eq!(binary.get_pixel(590, y)[0], 255);
        }
        for y in 300..314 {
            assert_eq!(binary.get_pixel(7, y)[0], 0);
        }
    }

    #[test]
    fn compact_near_edge_numeral_inside_uncertainty_band_is_preserved() {
        let contour = vec![
            Point { x: 0., y: 0. },
            Point { x: 599., y: 0. },
            Point { x: 599., y: 799. },
            Point { x: 0., y: 799. },
        ];
        let mut binary = GrayImage::from_pixel(600, 800, Luma([255]));
        for y in 200..208 {
            for x in 5..10 {
                if x == 5 || x == 9 || y == 200 || y == 207 {
                    binary.put_pixel(x, y, Luma([0]));
                }
            }
        }
        for y in 300..314 {
            binary.put_pixel(7, y, Luma([0]));
        }
        let saved = binary.clone();
        whiten_page_border(&mut binary, &contour, 4);
        for y in 200..208 {
            for x in 5..10 {
                assert_eq!(binary.get_pixel(x, y), saved.get_pixel(x, y));
            }
        }
        for y in 300..314 {
            assert_eq!(binary.get_pixel(7, y)[0], 0);
        }
    }

    #[test]
    fn raster_crop_edge_is_not_mistaken_for_a_paper_boundary() {
        let contour = vec![
            Point { x: -100., y: -100. },
            Point { x: 699., y: -100. },
            Point { x: 699., y: 899. },
            Point { x: -100., y: 899. },
        ];
        let mut binary = GrayImage::from_pixel(600, 800, Luma([255]));
        for y in 100..300 {
            binary.put_pixel(10, y, Luma([0]));
        }
        whiten_page_border(&mut binary, &contour, 2);
        for y in 100..300 {
            assert_eq!(binary.get_pixel(10, y)[0], 0);
        }
    }

    #[test]
    fn detached_wavy_rim_is_removed_whole_while_nearby_glyph_and_inner_ink_survive() {
        let contour = vec![
            Point { x: 0., y: 0. },
            Point { x: 599., y: 0. },
            Point { x: 599., y: 799. },
            Point { x: 0., y: 799. },
        ];
        let mut binary = GrayImage::from_pixel(600, 800, Luma([255]));
        let mut rim = vec![];
        for x in 40..560 {
            let y = 780 - (4. * (x as f64 / 70.).sin()).round() as u32;
            for yy in y..y + 3 {
                binary.put_pixel(x, yy, Luma([0]));
                rim.push((x, yy));
            }
        }
        // Compact page numeral in the same broad boundary band.
        for y in 760..774 {
            for x in 12..19 {
                if x == 12 || x == 18 || y == 760 || y == 773 {
                    binary.put_pixel(x, y, Luma([0]));
                }
            }
        }
        // A long edge-adjacent stroke enters actual page content and must stay.
        for x in 8..200 {
            binary.put_pixel(x, 400, Luma([0]));
        }
        // A thin isolated page numeral must not be mistaken for a short rim.
        for y in 730..744 {
            binary.put_pixel(15, y, Luma([0]));
        }
        let saved = binary.clone();
        whiten_page_border(&mut binary, &contour, 2);
        assert!(
            rim.into_iter()
                .all(|(x, y)| binary.get_pixel(x, y)[0] == 255)
        );
        for y in 760..774 {
            for x in 12..19 {
                assert_eq!(binary.get_pixel(x, y), saved.get_pixel(x, y));
            }
        }
        for x in 8..200 {
            assert_eq!(binary.get_pixel(x, 400)[0], 0);
        }
        for y in 730..744 {
            assert_eq!(binary.get_pixel(15, y)[0], 0);
        }
    }
}

#[cfg(test)]
mod crop_tests {
    use super::*;
    use image::Luma;
    #[test]
    fn crop_keeps_separate_page_number_and_ignores_corner_specks() {
        let mut image = GrayImage::from_pixel(100, 120, Luma([255]));
        for y in 30..90 {
            for x in 20..80 {
                image.put_pixel(x, y, Luma([0]));
            }
        }
        for y in 10..16 {
            image.put_pixel(70, y, Luma([0]));
        }
        image.put_pixel(0, 119, Luma([0]));
        assert_eq!(content_crop(&image, 5), [15, 5, 70, 90]);
        assert_eq!(image.get_pixel(0, 119)[0], 0);
    }
    #[test]
    fn blank_and_border_content_keep_valid_bounds() {
        let mut image = GrayImage::from_pixel(12, 9, Luma([255]));
        assert_eq!(content_crop(&image, 2), [0, 0, 12, 9]);
        for x in 0..6 {
            image.put_pixel(x, 0, Luma([0]));
        }
        assert_eq!(content_crop(&image, 2), [0, 0, 8, 3]);
    }
}

#[cfg(test)]
mod page_crop_tests {
    use super::*;
    use crate::geometry::Point;
    #[test]
    fn preserves_blank_paper_and_curved_boundary_extrema() {
        let contour = [
            Point { x: 10., y: 8. },
            Point { x: 90., y: 8. },
            Point { x: 98., y: 50. },
            Point { x: 90., y: 110. },
            Point { x: 10., y: 110. },
        ];
        assert_eq!(page_crop(&contour, 120, 130, 2), [8, 6, 93, 107]);
        // Framing has no dependency on a tight central music block.
        let mut binary = GrayImage::from_pixel(120, 130, image::Luma([255]));
        for x in 40..70 {
            binary.put_pixel(x, 50, image::Luma([0]));
        }
        assert_ne!(page_crop(&contour, 120, 130, 2), content_crop(&binary, 2));
    }
    #[test]
    fn missing_boundary_retains_canvas_instead_of_guessing_content_crop() {
        assert_eq!(page_crop(&[], 120, 130, 2), [0, 0, 120, 130]);
    }
}

#[cfg(test)]
mod print_tests {
    use super::*;
    use image::Luma;
    #[test]
    fn varying_paper_is_white_and_thin_ink_stays_high_contrast() {
        let input = GrayImage::from_fn(160, 90, |x, y| {
            Luma([if y == 45 && (20..140).contains(&x) {
                20
            } else {
                (90 + x / 2) as u8
            }])
        });
        let normalized = normalize_paper(&input);
        assert_eq!(normalized.get_pixel(30, 10)[0], 255);
        assert_eq!(normalized.get_pixel(130, 10)[0], 255);
        let print = antialias_print(&normalized);
        assert_eq!(print.dimensions(), input.dimensions());
        assert_eq!(print.get_pixel(80, 10)[0], 255);
        assert!(print.get_pixel(80, 45)[0] < 80);
        assert!((1..255).contains(&print.get_pixel(80, 44)[0]));
    }
    #[test]
    fn constant_paper_levels_are_white_and_black_is_not_inverted() {
        for tone in [40, 80, 120, 180, 240, 255] {
            let input = GrayImage::from_pixel(8, 9, Luma([tone]));
            assert!(normalize_paper(&input).as_raw().iter().all(|p| *p == 255));
        }
        assert!(
            normalize_paper(&GrayImage::from_pixel(8, 9, Luma([0])))
                .as_raw()
                .iter()
                .all(|p| *p == 0)
        );
    }
}

#[cfg(test)]
mod print_mask_tests {
    use super::*;
    use image::Luma;
    #[test]
    fn removes_rim_halo_but_protects_retained_antialiasing() {
        let mut gray = GrayImage::from_pixel(30, 10, Luma([255]));
        let mut raw = gray.clone();
        let mut clean = gray.clone();
        for y in 1..9 {
            raw.put_pixel(5, y, Luma([0]));
            gray.put_pixel(5, y, Luma([0]));
            gray.put_pixel(6, y, Luma([150]));
            raw.put_pixel(20, y, Luma([0]));
            clean.put_pixel(20, y, Luma([0]));
            gray.put_pixel(20, y, Luma([0]));
            gray.put_pixel(19, y, Luma([130]));
        }
        apply_binary_cleanup(&mut gray, &raw, &clean);
        assert_eq!(gray.get_pixel(5, 5)[0], 255);
        assert_eq!(gray.get_pixel(6, 5)[0], 255);
        assert_eq!(gray.get_pixel(20, 5)[0], 0);
        assert_eq!(gray.get_pixel(19, 5)[0], 130);
    }
}

#[cfg(test)]
mod gray_rim_reconstruction_tests {
    use super::*;
    use crate::geometry::Point;
    use image::Luma;
    #[test]
    fn faint_fragment_chain_is_removed_without_authorizing_dark_folio_or_music() {
        let mut gray = GrayImage::from_pixel(600, 800, Luma([255]));
        let contour = [
            Point { x: 10., y: 10. },
            Point { x: 590., y: 10. },
            Point { x: 590., y: 790. },
            Point { x: 10., y: 790. },
        ];
        let fragments = [(200, 214), (225, 239), (250, 264), (275, 286)];
        for (a, b) in fragments {
            for y in a..b {
                for x in 574..579 {
                    gray.put_pixel(x, y, Luma([175]));
                }
            }
        }
        // A black isolated folio in the same strip and within continuation
        // distance must not inherit the new faint-chain evidence.
        for y in 312..329 {
            for x in 574..579 {
                gray.put_pixel(x, y, Luma([40]));
            }
        }
        // A faint isolated glyph on another edge is not a chain.
        for y in 310..324 {
            for x in 23..28 {
                gray.put_pixel(x, y, Luma([175]));
            }
        }
        // Printed inner frame and protected music remain exact.
        for y in 180..350 {
            gray.put_pixel(550, y, Luma([30]));
        }
        for y in [190, 200, 210, 220, 230] {
            for x in 80..240 {
                gray.put_pixel(x, y, Luma([35]));
            }
        }
        let protected = [[75., 170., 245., 260.]];
        let before = gray.clone();
        whiten_gray_border(&mut gray, &contour, 4, &protected);
        for (a, b) in fragments {
            for y in a..b {
                for x in 574..579 {
                    assert_eq!(gray.get_pixel(x, y)[0], 255);
                }
            }
        }
        for y in 312..329 {
            for x in 574..579 {
                assert_eq!(gray.get_pixel(x, y), before.get_pixel(x, y));
            }
        }
        for y in 310..324 {
            for x in 23..28 {
                assert_eq!(gray.get_pixel(x, y), before.get_pixel(x, y));
            }
        }
        for y in 180..350 {
            assert_eq!(gray.get_pixel(550, y), before.get_pixel(550, y));
        }
        for y in 170..=260 {
            for x in 75..=245 {
                assert_eq!(gray.get_pixel(x, y), before.get_pixel(x, y));
            }
        }
    }
    #[test]
    fn short_faint_marks_without_three_collinear_components_stay() {
        let contour = [
            Point { x: 10., y: 10. },
            Point { x: 590., y: 10. },
            Point { x: 590., y: 790. },
            Point { x: 10., y: 790. },
        ];
        for columns in [vec![574, 574], vec![574, 565, 582]] {
            let mut gray = GrayImage::from_pixel(600, 800, Luma([255]));
            for (i, x) in columns.iter().enumerate() {
                for y in 200 + i as u32 * 25..214 + i as u32 * 25 {
                    for xx in *x..*x + 5 {
                        gray.put_pixel(xx, y, Luma([175]));
                    }
                }
            }
            let before = gray.clone();
            whiten_gray_border(&mut gray, &contour, 4, &[]);
            assert_eq!(
                gray, before,
                "small near-edge glyphs do not establish a boundary chain"
            );
        }
    }
    #[test]
    fn boundary_stroke_with_interior_contact_keeps_every_interior_pixel() {
        let mut gray = GrayImage::from_pixel(300, 400, Luma([255]));
        for y in 40..350 {
            gray.put_pixel(15, y, Luma([190]));
        }
        for x in 15..180 {
            gray.put_pixel(x, 100, Luma([120]));
        }
        let before = gray.clone();
        let contour = [
            Point { x: 10., y: 10. },
            Point { x: 290., y: 10. },
            Point { x: 290., y: 390. },
            Point { x: 10., y: 390. },
        ];
        whiten_gray_border(&mut gray, &contour, 1, &[]);
        assert_eq!(gray.get_pixel(15, 200)[0], 255);
        for y in 23..378 {
            for x in 23..278 {
                assert_eq!(gray.get_pixel(x, y), before.get_pixel(x, y));
            }
        }
    }
    #[test]
    fn rotated_solid_wedge_is_distinct_from_compact_bent_ink() {
        let mut triangle = vec![];
        for y in 10..40 {
            for x in 10..50 {
                // Rotated right triangle, rasterized by three half-plane tests.
                let p = (x as f64, y as f64);
                let v = [(12., 18.), (45., 12.), (23., 36.)];
                let cross = |a: (f64, f64), b: (f64, f64)| {
                    (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)
                };
                if cross(v[0], v[1]) >= 0. && cross(v[1], v[2]) >= 0. && cross(v[2], v[0]) >= 0. {
                    triangle.push(y * 100 + x);
                }
            }
        }
        assert!(solid_triangle(&triangle, 100));
        let mut bent = vec![];
        for y in 10..35 {
            for x in 20..23 {
                bent.push(y * 100 + x);
            }
        }
        for x in 12..20 {
            for y in 10..13 {
                bent.push(y * 100 + x);
            }
        }
        assert!(!solid_triangle(&bent, 100));
    }
    #[test]
    fn corner_wedge_is_removed_but_nearby_folio_survives() {
        let mut gray = GrayImage::from_pixel(1000, 1200, Luma([255]));
        for y in 0..24 {
            for x in 0..(18 - y * 18 / 24) {
                gray.put_pixel(25 + x, 35 + y, Luma([60]));
            }
        }
        for y in 50..61 {
            gray.put_pixel(42, y, Luma([190]));
        }
        // Near the same real corner: compact "1" shape, not a solid wedge.
        for y in 35..59 {
            for x in 50..53 {
                gray.put_pixel(x, y, Luma([90]));
            }
        }
        for x in 45..53 {
            gray.put_pixel(x, 35, Luma([90]));
        }
        let before = gray.clone();
        let contour = [
            Point { x: 20., y: 30. },
            Point { x: 980., y: 30. },
            Point { x: 980., y: 1180. },
            Point { x: 20., y: 1180. },
        ];
        whiten_gray_border(&mut gray, &contour, 2, &[]);
        assert_eq!(gray.get_pixel(26, 38)[0], 255);
        assert_eq!(gray.get_pixel(42, 55)[0], 255);
        for y in 35..59 {
            for x in 45..54 {
                assert_eq!(gray.get_pixel(x, y), before.get_pixel(x, y));
            }
        }
    }
    #[test]
    fn thick_three_sided_gray_frame_is_removed_without_changing_binary_default() {
        let mut gray = GrayImage::from_pixel(300, 400, Luma([255]));
        // Connected U shape: large in both dimensions, thicker than the old
        // perimeter*erosion-margin area limit, but wholly follows the page.
        for y in 20..380 {
            for x in 15..19 {
                gray.put_pixel(x, y, Luma([70]));
            }
            for x in 281..285 {
                gray.put_pixel(x, y, Luma([70]));
            }
        }
        for y in 378..382 {
            for x in 15..285 {
                gray.put_pixel(x, y, Luma([70]));
            }
        }
        // A weak compact folio and a boundary-adjacent protected music stroke.
        for y in 22..32 {
            for x in 80..87 {
                gray.put_pixel(x, y, Luma([180]));
            }
        }
        for x in 16..150 {
            gray.put_pixel(x, 100, Luma([120]));
        }
        let contour = [
            Point { x: 10., y: 10. },
            Point { x: 290., y: 10. },
            Point { x: 290., y: 390. },
            Point { x: 10., y: 390. },
        ];
        let before = gray.clone();
        let mut binary = GrayImage::from_fn(300, 400, |x, y| {
            Luma([if gray.get_pixel(x, y)[0] < 240 {
                0
            } else {
                255
            }])
        });
        whiten_page_border(&mut binary, &contour, 1);
        assert_eq!(
            binary.get_pixel(18, 200)[0],
            0,
            "binary default stays conservative"
        );
        whiten_gray_border(&mut gray, &contour, 1, &[[0., 90., 299., 110.]]);
        assert_eq!(gray.get_pixel(18, 200)[0], 255);
        assert_eq!(gray.get_pixel(150, 380)[0], 255);
        assert_eq!(gray.get_pixel(281, 200)[0], 255);
        assert_eq!(gray.get_pixel(83, 26)[0], 180);
        for y in 90..=110 {
            for x in 0..300 {
                assert_eq!(gray.get_pixel(x, y), before.get_pixel(x, y));
            }
        }
    }
    #[test]
    fn gray_only_rim_disappears_but_weak_folio_and_music_are_unchanged() {
        let mut gray = GrayImage::from_pixel(300, 400, Luma([255]));
        for y in 40..350 {
            gray.put_pixel(15, y, Luma([190]));
        }
        // A compact, faint page number and a real boundary-adjacent music stroke.
        for y in 18..28 {
            for x in 18..24 {
                gray.put_pixel(x, y, Luma([180]));
            }
        }
        for x in 20..270 {
            gray.put_pixel(x, 100, Luma([120]));
        }
        let contour = [
            crate::geometry::Point { x: 10., y: 10. },
            crate::geometry::Point { x: 290., y: 10. },
            crate::geometry::Point { x: 290., y: 390. },
            crate::geometry::Point { x: 10., y: 390. },
        ];
        let old = gray.clone();
        let empty = GrayImage::from_pixel(300, 400, Luma([255]));
        apply_binary_cleanup_with_contour(&mut gray, &empty, &empty, &contour);
        assert_eq!(
            gray.get_pixel(15, 200)[0],
            190,
            "old black-core path leaves this rim"
        );
        whiten_gray_border(&mut gray, &contour, 1, &[[0., 90., 299., 110.]]);
        assert_eq!(gray.get_pixel(15, 200)[0], 255);
        assert_eq!(gray.get_pixel(20, 22)[0], 180);
        for y in 90..=110 {
            for x in 0..300 {
                assert_eq!(gray.get_pixel(x, y), old.get_pixel(x, y));
            }
        }
    }
    #[test]
    fn long_soft_rim_is_removed_beyond_local_seed_radius_without_losing_nearby_ink() {
        let mut gray = GrayImage::from_pixel(200, 180, Luma([255]));
        let mut raw = gray.clone();
        let mut cleaned = gray.clone();
        for y in 30..160 {
            for x in 12..17 {
                gray.put_pixel(x, y, Luma([190]));
            }
        }
        for y in 140..150 {
            raw.put_pixel(13, y, Luma([0]));
            gray.put_pixel(13, y, Luma([0]));
        }
        // A retained glyph and its gray AA fringe are close to the halo.
        for y in 60..68 {
            for x in 20..25 {
                gray.put_pixel(x, y, Luma([0]));
                raw.put_pixel(x, y, Luma([0]));
                cleaned.put_pixel(x, y, Luma([0]));
            }
        }
        for y in 60..68 {
            gray.put_pixel(19, y, Luma([160]));
        }
        // An isolated weak glyph has no strong binary core at all.
        for y in 15..23 {
            for x in 12..17 {
                gray.put_pixel(x, y, Luma([185]));
            }
        }
        let contour = [
            Point { x: 10., y: 10. },
            Point { x: 190., y: 10. },
            Point { x: 190., y: 170. },
            Point { x: 10., y: 170. },
        ];
        apply_binary_cleanup_with_contour(&mut gray, &raw, &cleaned, &contour);
        assert_eq!(gray.get_pixel(13, 40)[0], 255);
        assert_eq!(gray.get_pixel(14, 158)[0], 255);
        assert_eq!(gray.get_pixel(20, 64)[0], 0);
        assert_eq!(gray.get_pixel(19, 64)[0], 160);
        assert_eq!(gray.get_pixel(14, 18)[0], 185);
    }
    #[test]
    fn reconstruction_stops_at_true_page_band_and_does_not_use_canvas_edges() {
        let mut gray = GrayImage::from_pixel(200, 180, Luma([255]));
        let mut raw = gray.clone();
        let cleaned = gray.clone();
        // Connected gray stroke extends from the paper rim into page content.
        for x in 42..90 {
            gray.put_pixel(x, 90, Luma([190]));
        }
        raw.put_pixel(43, 90, Luma([0]));
        // A separate canvas-edge component is not a page-boundary cue.
        for y in 20..160 {
            gray.put_pixel(1, y, Luma([190]));
        }
        raw.put_pixel(1, 140, Luma([0]));
        let contour = [
            Point { x: 40., y: 10. },
            Point { x: 190., y: 10. },
            Point { x: 190., y: 170. },
            Point { x: 40., y: 170. },
        ];
        apply_binary_cleanup_with_contour(&mut gray, &raw, &cleaned, &contour);
        assert_eq!(gray.get_pixel(44, 90)[0], 255);
        assert_eq!(gray.get_pixel(60, 90)[0], 190);
        assert_eq!(gray.get_pixel(1, 40)[0], 190);
    }
}
