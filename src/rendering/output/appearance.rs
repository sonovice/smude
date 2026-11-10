//! Conservative selection of photographic color versus paper-white notation.
use image::RgbImage;

pub(crate) fn preserve_color(image: &RgbImage, has_music: bool) -> bool {
    if has_music || image.width() < 2 || image.height() < 2 {
        return false;
    }
    // A white balance cast alone is insufficient. Require substantial chroma
    // over at least two percent of the interior; a colored desk/edge must not decide it.
    let step = (image.width().max(image.height()) / 256).max(1) as usize;
    let (mut colored, mut total) = (0, 0);
    for y in (image.height() / 10..image.height() * 9 / 10).step_by(step) {
        for x in (image.width() / 10..image.width() * 9 / 10).step_by(step) {
            let p = image.get_pixel(x, y).0;
            let hi = *p.iter().max().unwrap();
            let lo = *p.iter().min().unwrap();
            colored += usize::from(hi - lo > 32 && u16::from(lo) * 100 < u16::from(hi) * 72);
            total += 1;
        }
    }
    colored * 50 > total
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;
    #[test]
    fn a_small_colored_illustration_selects_rgb_without_changing_music() {
        let page = RgbImage::from_fn(400, 500, |x, y| {
            Rgb(if (260..330).contains(&x) && (100..170).contains(&y) {
                [70, 140, 170]
            } else {
                [195, 180, 165]
            })
        });
        assert!(
            preserve_color(&page, false),
            "small illustration was discarded"
        );
        assert!(!preserve_color(&page, true), "changed the music profile");
    }
    #[test]
    fn color_cover_is_preserved_but_cast_paper_and_music_are_not() {
        let cover = RgbImage::from_pixel(300, 400, Rgb([65, 95, 155]));
        assert!(preserve_color(&cover, false));
        assert!(!preserve_color(&cover, true));
        let paper = RgbImage::from_pixel(300, 400, Rgb([170, 180, 196]));
        assert!(!preserve_color(&paper, false));
        let edge = RgbImage::from_fn(300, 400, |x, _| {
            if x < 30 {
                Rgb([65, 95, 155])
            } else {
                Rgb([220, 220, 220])
            }
        });
        assert!(!preserve_color(&edge, false));
    }
}
