//! Output regressions exercise the public operations and exact distance primitives.
use super::border::*;
use super::distance::*;
use super::*;

#[cfg(test)]
mod distance_and_cache {
    #[test]
    fn sparse_distance_sweeps_match_exact_chebyshev_distance() {
        // Seed rectangles as [x0, y0, x1, y1], inclusive: single pixels, solid
        // areas covering whole tiles, and rasters with large untouched parts.
        let cases: [(usize, usize, &[[usize; 4]]); 4] = [
            (
                17,
                600,
                &[
                    [7, 255, 7, 255],
                    [3, 267, 3, 267],
                    [1, 0, 1, 0],
                    [15, 599, 15, 599],
                ],
            ),
            (
                300,
                200,
                &[[60, 0, 200, 130], [299, 199, 299, 199], [0, 150, 0, 150]],
            ),
            (523, 140, &[[400, 70, 400, 70]]),
            (130, 70, &[]),
        ];
        for (w, h, seeds) in cases {
            for cap in [1_u8, 11, 65, 125, 200] {
                let mut narrow = vec![cap; w * h];
                for seed in seeds {
                    for y in seed[1]..=seed[3] {
                        narrow[y * w + seed[0]..=y * w + seed[2]].fill(0);
                    }
                }
                let mut wide: Vec<u16> = narrow.iter().map(|v| u16::from(*v)).collect();
                // Any number of bands yields the field of a single sweep.
                for bands in [2, 3, 16] {
                    let mut banded = narrow.clone();
                    capped_distance_in_bands(&mut banded, w, h, cap, bands);
                    let mut whole = narrow.clone();
                    capped_distance_in_bands(&mut whole, w, h, cap, 1);
                    assert_eq!(banded, whole, "cap={cap}, bands={bands}");
                }
                capped_distance(&mut narrow, w, h, cap);
                capped_distance(&mut wide, w, h, u16::from(cap));
                for y in 0..h {
                    for x in 0..w {
                        let outside = |v: usize, low: usize, high: usize| {
                            low.saturating_sub(v).max(v.saturating_sub(high))
                        };
                        let expected = seeds
                            .iter()
                            .map(|s| outside(x, s[0], s[2]).max(outside(y, s[1], s[3])))
                            .min()
                            .map_or(cap, |d| d.min(usize::from(cap)) as u8);
                        assert_eq!(narrow[y * w + x], expected, "cap={cap}, ({x},{y})");
                        assert_eq!(wide[y * w + x], u16::from(expected), "cap={cap}, ({x},{y})");
                    }
                }
            }
        }
    }
    #[test]
    fn mask_erosion_matches_the_square_window_inside_the_raster() {
        for (w, h) in [(1_usize, 1_usize), (9, 7), (40, 33), (64, 5)] {
            for radius in [1_usize, 2, 5, 40] {
                let source: Vec<u8> = (0..w * h)
                    .map(|i| u8::from(i * 7919 % 23 != 0 && i % w != 3 * (i / w)))
                    .collect();
                let mut actual = source.clone();
                erode_mask(&mut actual, w, h, radius);
                for y in 0..h {
                    for x in 0..w {
                        let inside = x >= radius && y >= radius && x + radius < w && y + radius < h;
                        let expected = inside
                            && (y - radius..=y + radius).all(|yy| {
                                (x - radius..=x + radius).all(|xx| source[yy * w + xx] != 0)
                            });
                        assert_eq!(
                            actual[y * w + x] != 0,
                            expected,
                            "{w}x{h} r{radius} ({x},{y})"
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn page_mask_matches_even_odd_row_crossings() {
        let point = |x: f64, y: f64| crate::geometry::Point { x, y };
        let contours = [
            vec![
                point(2.3, 1.7),
                point(40.6, 3.2),
                point(37.1, 28.9),
                point(4.4, 30.5),
            ],
            // Concave, with vertices on pixel rows and a horizontal edge.
            vec![
                point(5., 4.),
                point(30., 4.),
                point(18., 15.),
                point(31., 26.),
                point(6., 20.),
            ],
            // Partly outside the raster on every side.
            vec![
                point(-8.5, -3.),
                point(52., 10.),
                point(20.2, 44.),
                point(-2., 25.),
            ],
        ];
        let (w, h) = (44_usize, 34_usize);
        for contour in contours {
            let mask = page_mask(&contour, w, h);
            for y in 0..h {
                let yy = y as f64;
                let mut cuts: Vec<f64> = (contour.iter())
                    .zip(contour.iter().cycle().skip(1))
                    .filter(|(a, b)| (a.y > yy) != (b.y > yy))
                    .map(|(a, b)| a.x + (yy - a.y) * (b.x - a.x) / (b.y - a.y))
                    .collect();
                cuts.sort_by(f64::total_cmp);
                for x in 0..w {
                    let expected = cuts
                        .chunks_exact(2)
                        .any(|pair| pair[0] <= x as f64 && x as f64 <= pair[1]);
                    assert_eq!(mask[y * w + x] != 0, expected, "({x},{y})");
                }
            }
        }
    }
    #[test]
    fn shared_border_geometry_preserves_binary_cleanup_pixels() {
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
            let raw = binarize_sauvola(&gray);
            let mut expected_binary = raw.clone();
            let expected_count = whiten_page_border(&mut expected_binary, &contour, 2);
            let border = PageBorder::new(&contour);
            let mut actual_binary = raw.clone();
            assert_eq!(border.whiten_binary(&mut actual_binary, 2), expected_count);
            assert_eq!(actual_binary, expected_binary);
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
    #[cfg(feature = "application")]
    #[test]
    fn row_copies_equal_the_image_crate_crop_including_clamped_bounds() {
        let gray = GrayImage::from_fn(13, 9, |x, y| Luma([(x * 19 + y * 7) as u8]));
        let color = image::RgbImage::from_fn(13, 9, |x, y| {
            image::Rgb([(x * 19) as u8, (y * 27) as u8, (x + y) as u8])
        });
        for bounds in [
            [0, 0, 13, 9],
            [3, 2, 5, 4],
            [12, 8, 1, 1],
            [4, 0, 0, 9],
            [10, 7, 20, 20],
            [13, 9, 3, 3],
            [40, 40, 2, 2],
        ] {
            let [x, y, w, h] = bounds;
            assert_eq!(
                cropped(&gray, bounds),
                image::imageops::crop_imm(&gray, x, y, w, h).to_image()
            );
            assert_eq!(
                cropped(&color, bounds),
                image::imageops::crop_imm(&color, x, y, w, h).to_image()
            );
        }
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
    fn faint_staff_stem_dot_and_coverage_survive_paper_normalization() {
        let mut input = GrayImage::from_pixel(240, 180, Luma([240]));
        for x in 20..220 {
            input.put_pixel(x, 70, Luma([220]));
            input.put_pixel(x, 71, Luma([239]));
        }
        for y in 45..95 {
            input.put_pixel(90, y, Luma([225]));
        }
        input.put_pixel(120, 55, Luma([230]));
        let output = normalize_paper(&input);
        for (x, y) in [(50, 70), (50, 71), (90, 50), (120, 55)] {
            let expected = (255. * f64::from(input.get_pixel(x, y)[0]) / 240.).round() as u8;
            assert!(
                output.get_pixel(x, y)[0].abs_diff(expected) <= 1,
                "lost coverage at ({x},{y}): expected {expected}, got {}",
                output.get_pixel(x, y)[0]
            );
            assert!(output.get_pixel(x, y)[0] < 255);
        }
        assert_eq!(output.get_pixel(40, 40)[0], 255);
    }

    #[test]
    fn gray_border_does_not_erase_a_real_near_edge_staff_or_dot() {
        let contour = [
            crate::geometry::Point { x: 10., y: 10. },
            crate::geometry::Point { x: 289., y: 10. },
            crate::geometry::Point { x: 289., y: 389. },
            crate::geometry::Point { x: 10., y: 389. },
        ];
        let mut gray = GrayImage::from_pixel(300, 400, Luma([255]));
        for x in 30..270 {
            gray.put_pixel(x, 12, Luma([80]));
        }
        gray.put_pixel(20, 13, Luma([180]));
        let before = gray.clone();
        let raw = binarize_sauvola(&gray);
        let mut binary = raw.clone();
        let border = PageBorder::new(&contour);
        border.whiten_binary(&mut binary, 2);
        border.clean_gray(&mut gray, 2, &[[15., 10., 275., 18.]]);
        assert!(
            gray == before,
            "binary rim classification must not erase gray notation"
        );
    }

    #[test]
    fn final_gray_border_removes_inner_rim_without_losing_faint_print() {
        let contour = [
            crate::geometry::Point { x: 10., y: 10. },
            crate::geometry::Point { x: 290., y: 10. },
            crate::geometry::Point { x: 290., y: 390. },
            crate::geometry::Point { x: 10., y: 390. },
        ];
        let mut gray = GrayImage::from_pixel(300, 400, Luma([255]));
        for y in 40..350 {
            gray.put_pixel(15, y, Luma([190]));
        }
        for x in 15..285 {
            gray.put_pixel(x, 380, Luma([190]));
        }
        for x in 30..270 {
            gray.put_pixel(x, 100, Luma([235]));
        }
        gray.put_pixel(80, 105, Luma([244]));
        for y in 18..28 {
            for x in 24..30 {
                gray.put_pixel(x, y, Luma([180]));
            }
        }
        let before = gray.clone();
        PageBorder::new(&contour).clean_gray(&mut gray, 1, &[[25., 90., 275., 115.]]);
        assert_eq!(gray.get_pixel(15, 200)[0], 255, "inner page rim remains");
        assert_eq!(gray.get_pixel(150, 380)[0], 255, "bottom rim remains");
        for y in 90..115 {
            for x in 25..275 {
                assert_eq!(gray.get_pixel(x, y), before.get_pixel(x, y));
            }
        }
        for y in 18..28 {
            for x in 24..30 {
                assert_eq!(gray.get_pixel(x, y), before.get_pixel(x, y));
            }
        }
    }

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
        assert_eq!(normalized.dimensions(), input.dimensions());
        assert_eq!(normalized.get_pixel(80, 10)[0], 255);
        assert!(normalized.get_pixel(80, 45)[0] < 80);
        assert_eq!(normalized.get_pixel(80, 44)[0], 255);
    }

    #[test]
    fn large_shaded_page_keeps_weak_lines_across_analysis_blocks() {
        let input = GrayImage::from_fn(2051, 301, |x, y| {
            let paper = 100 + x * 140 / 2050;
            let ink = [30, 80, 130, 180, 230].contains(&y) || (x == 1024 && (60..150).contains(&y));
            Luma([(paper - if ink { 10 } else { 0 }) as u8])
        });
        let output = normalize_paper(&input);
        for x in 12..2040 {
            assert!(
                output.get_pixel(x, 15)[0] >= 252,
                "uncorrected paper at {x}"
            );
            for y in [30, 80, 130, 180, 230] {
                let paper = 100 + x * 140 / 2050;
                let expected = (255. * (paper - 10) as f32 / paper as f32).round() as u8;
                assert!(
                    output.get_pixel(x, y)[0].abs_diff(expected) <= 2,
                    "line coverage at {x},{y}"
                );
            }
        }
        // Partial last analysis block must not erase faint ink at the image edge.
        assert!(output.get_pixel(2050, 80)[0] < 250);
    }

    #[test]
    fn gray_border_only_whitens_beyond_the_exterior_uncertainty_band() {
        let contour = [
            crate::geometry::Point { x: 10., y: 10. },
            crate::geometry::Point { x: 89., y: 10. },
            crate::geometry::Point { x: 89., y: 89. },
            crate::geometry::Point { x: 10., y: 89. },
        ];
        let mut gray = GrayImage::from_pixel(100, 100, Luma([190]));
        PageBorder::new(&contour).clean_gray(&mut gray, 2, &[]);
        assert_eq!(gray.get_pixel(3, 50)[0], 255);
        assert_eq!(gray.get_pixel(8, 50)[0], 190);
        assert_eq!(gray.get_pixel(10, 50)[0], 190);
        assert_eq!(gray.get_pixel(50, 50)[0], 190);
        let before = gray.clone();
        PageBorder::new(&[]).clean_gray(&mut gray, 2, &[]);
        assert!(gray == before);
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
