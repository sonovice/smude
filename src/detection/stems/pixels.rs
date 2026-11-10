//! Read-only labelled candidates and a sheared measurement view.
use crate::geometry::{Point, Staff, observations::interpolate};
use image::GrayImage;
use std::ops::Range;

/// One bit per pixel, stored by lines: the rows of a band, the columns of a
/// view. Every line ends in a spare empty word, so any 64 pixels that start
/// inside a line lie in two of its words.
struct Plane {
    stride: usize,
    words: Vec<u64>,
}
impl Plane {
    fn new(length: usize, lines: usize) -> Self {
        let stride = length.div_ceil(64) + 1;
        Self {
            stride,
            words: vec![0; stride * lines],
        }
    }
    fn set(&mut self, at: usize, line: usize) {
        self.words[line * self.stride + at / 64] |= 1 << (at % 64);
    }
    fn get(&self, at: usize, line: usize) -> bool {
        self.words[line * self.stride + at / 64] >> (at % 64) & 1 != 0
    }
    fn line(&self, line: usize) -> &[u64] {
        &self.words[line * self.stride..][..self.stride]
    }
    /// The 64 pixels of `line` from `at`: bit `i` is pixel `at + i`, unset
    /// where that leaves the line.
    #[inline]
    fn window(&self, at: i64, line: usize) -> u64 {
        let words = self.line(line);
        if at < 0 {
            return if at > -64 { words[0] << -at } else { 0 };
        }
        let (word, shift) = (at as usize / 64, at as u32 % 64);
        match words.get(word..word + 2) {
            // Two steps keep the shift defined for `shift == 0`.
            Some(pair) => pair[0] >> shift | pair[1] << 1 << (63 - shift),
            None => 0,
        }
    }
}

pub(super) struct Component {
    pub id: u32,
    pub bounds: [i32; 4],
}
pub(super) struct Pixels {
    pub width: usize,
    pub height: usize,
    pub origin: [i32; 2],
    pub labels: Vec<u32>,
    ink: Plane,
    /// Pixels on a measured staff rule.
    rule: Plane,
    rules: Vec<[f64; 5]>,
    line_radius: [f64; 5],
    pub components: Vec<Component>,
}
impl Pixels {
    pub fn new(image: &GrayImage, staff: &Staff, space: f64) -> Self {
        let w = image.width() as i32;
        let h = image.height() as i32;
        let left = staff.x[0].floor().max(0.) as i32;
        let right = (staff.x.last().unwrap().ceil() as i32 + 1).min(w);
        let top = (staff.y[0].iter().copied().fold(f64::INFINITY, f64::min) - 4. * space)
            .floor()
            .max(0.) as i32;
        let bottom = (staff.y[4].iter().copied().fold(f64::NEG_INFINITY, f64::max) + 4. * space)
            .ceil()
            .min(f64::from(h)) as i32;
        let width = (right - left) as usize;
        let height = (bottom - top) as usize;
        let black = |x: i32, y: i32| {
            x >= 0 && y >= 0 && x < w && y < h && image.get_pixel(x as u32, y as u32)[0] < 128
        };
        let rules: Vec<[f64; 5]> = (left..right)
            .map(|x| std::array::from_fn(|k| interpolate(&staff.x, &staff.y[k], f64::from(x))))
            .collect();
        let line_radius = std::array::from_fn(|k| {
            let mut lengths = vec![];
            for x in (left..right).step_by(space.ceil().max(1.) as usize) {
                let y = rules[(x - left) as usize][k].round() as i32;
                if !black(x, y) {
                    continue;
                }
                let mut lo = y;
                let mut hi = y;
                let limit = (space * 0.4).ceil() as i32;
                while y - lo < limit && black(x, lo - 1) {
                    lo -= 1;
                }
                while hi - y < limit && black(x, hi + 1) {
                    hi += 1;
                }
                lengths.push(hi - lo + 1);
            }
            lengths.sort_unstable();
            f64::from(lengths.get(lengths.len() / 4).copied().unwrap_or(1)) / 2. + 0.35
        });
        // The image is never edited. Rule support is only a neutral mask for
        // the head score; connected candidates retain every original ink pixel.
        let mut ink = Plane::new(width, height);
        let mut labels = vec![0; width * height];
        for (row, y) in (top..bottom).enumerate() {
            for (column, x) in (left..right).enumerate() {
                if black(x, y) {
                    ink.set(column, row);
                    labels[row * width + column] = u32::MAX;
                }
            }
        }
        let mut rule = Plane::new(width, height);
        for (column, course) in rules.iter().enumerate() {
            for (r, radius) in course.iter().zip(line_radius) {
                let rows = (r - radius).floor() as i32 - 1..=(r + radius).ceil() as i32 + 1;
                for y in rows.filter(|y| (top..bottom).contains(y)) {
                    if on_rule_row(*r, radius, y) {
                        rule.set(column, (y - top) as usize);
                    }
                }
            }
        }
        let mut components = vec![];
        let mut queue = vec![];
        let mut id = 0u32;
        for seed in 0..labels.len() {
            if labels[seed] != u32::MAX {
                continue;
            }
            id += 1;
            queue.clear();
            queue.push(seed);
            labels[seed] = id;
            let mut bounds = [right, bottom, left, top];
            let mut i = 0;
            while i < queue.len() {
                let p = queue[i];
                i += 1;
                let x = p % width;
                let y = p / width;
                bounds[0] = bounds[0].min(x as i32 + left);
                bounds[1] = bounds[1].min(y as i32 + top);
                bounds[2] = bounds[2].max(x as i32 + left + 1);
                bounds[3] = bounds[3].max(y as i32 + top + 1);
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let xx = x as i32 + dx;
                        let yy = y as i32 + dy;
                        if xx < 0 || yy < 0 || xx >= width as i32 || yy >= height as i32 {
                            continue;
                        }
                        let n = yy as usize * width + xx as usize;
                        if labels[n] == u32::MAX {
                            labels[n] = id;
                            queue.push(n);
                        }
                    }
                }
            }
            if f64::from(bounds[3] - bounds[1]) >= 1.6 * space {
                components.push(Component { id, bounds });
            }
        }
        Self {
            width,
            height,
            origin: [left, top],
            labels,
            ink,
            rule,
            rules,
            line_radius,
            components,
        }
    }
    /// The band row of page row `y`.
    fn row(&self, y: i32) -> Option<usize> {
        let y = y - self.origin[1];
        (y >= 0 && (y as usize) < self.height).then_some(y as usize)
    }
    /// The band position of a page pixel.
    fn cell(&self, x: i32, y: i32) -> Option<(usize, usize)> {
        let x = x - self.origin[0];
        (x >= 0 && (x as usize) < self.width)
            .then_some(x as usize)
            .zip(self.row(y))
    }
    pub fn on_rule(&self, x: i32, y: i32) -> bool {
        if let Some((x, y)) = self.cell(x, y) {
            return self.rule.get(x, y);
        }
        // Rules may leave the labelled band where the page clips it.
        let local = x - self.origin[0];
        local >= 0
            && self.rules.get(local as usize).is_some_and(|rule| {
                rule.iter()
                    .zip(self.line_radius)
                    .any(|(r, radius)| on_rule_row(*r, radius, y))
            })
    }
    pub fn ink(&self, x: i32, y: i32) -> bool {
        self.cell(x, y).is_some_and(|(x, y)| self.ink.get(x, y))
    }
    /// `[ink, observed]` of row `y` over the 64 columns from `x`, bit `i`
    /// being column `x + i`. A staff rule hides the pixels it covers; whatever
    /// lies off the band is observed paper.
    pub fn observed(&self, x: i32, y: i32) -> [u64; 2] {
        let local = i64::from(x) - i64::from(self.origin[0]);
        let (ink, hidden) = match self.row(y) {
            Some(row) => (self.ink.window(local, row), self.rule.window(local, row)),
            None => (
                0,
                (0..64)
                    .filter(|i| self.on_rule(x + i, y))
                    .fold(0, |hidden, i| hidden | 1 << i),
            ),
        };
        [ink & !hidden, !hidden]
    }
    /// The columns of page row `y` within `columns` that hold ink of
    /// component `id`, found from the ink rather than by asking every pixel.
    pub fn owned(&self, id: u32, columns: Range<i32>, y: i32) -> impl Iterator<Item = i32> + '_ {
        let row = self.row(y);
        let end = columns.end;
        row.into_iter()
            .flat_map(move |row| columns.clone().step_by(64).map(move |x| (row, x)))
            .flat_map(move |(row, x)| {
                let mut ink = self.ink.window(i64::from(x - self.origin[0]), row);
                if end - x < 64 {
                    ink &= (1 << (end - x)) - 1;
                }
                std::iter::from_fn(move || {
                    (ink != 0).then(|| {
                        let column = x + ink.trailing_zeros() as i32;
                        ink &= ink - 1;
                        column
                    })
                })
            })
            .filter(move |&x| self.own(id, x, y))
    }
    pub fn label(&self, x: i32, y: i32) -> u32 {
        self.cell(x, y)
            .map_or(0, |(x, y)| self.labels[y * self.width + x])
    }
    pub fn own(&self, id: u32, x: i32, y: i32) -> bool {
        id != 0 && self.label(x, y) == id
    }
}

fn on_rule_row(rule: f64, radius: f64, y: i32) -> bool {
    (rule - f64::from(y)).abs() <= radius
}

pub(super) struct View<'a> {
    pub pixels: &'a Pixels,
    pub component: &'a Component,
    pub shear: f64,
    pub padding: i32,
    /// The component's ink at every position of the view, by columns. The
    /// probes read each position many times, the shear is resolved once.
    own: Plane,
}
impl<'a> View<'a> {
    pub fn new(pixels: &'a Pixels, component: &'a Component, shear: f64, padding: i32) -> Self {
        let mut view = Self {
            pixels,
            component,
            shear,
            padding,
            own: Plane::new(0, 0),
        };
        let (width, height) = (view.width(), view.height());
        let [left, _, right, bottom] = component.bounds;
        let mut own = Plane::new(height as usize, width as usize);
        for y in 0..height {
            let page_y = bottom - 1 - y;
            let lean = view.lean(f64::from(page_y));
            if ((lean - lean.floor()) - 0.5).abs() > 1e-6 {
                // Away from half a column the rounding of `point` moves the
                // whole row alike, and the component ends at its bounds.
                let shift = left - padding + lean.round() as i32;
                pixels
                    .owned(component.id, left..right, page_y)
                    .map(|x| x - shift)
                    .filter(|x| (0..width).contains(x))
                    .for_each(|x| own.set(y as usize, x as usize));
            } else {
                (0..width)
                    .filter(|&x| view.sheared_own(x, y))
                    .for_each(|x| own.set(y as usize, x as usize));
            }
        }
        view.own = own;
        view
    }
    /// The component's ink down column `x`, bit `y` of the words being row
    /// `y`; they end in an empty word.
    pub fn column(&self, x: i32) -> &[u64] {
        self.own.line(x as usize)
    }
    pub fn width(&self) -> i32 {
        self.component.bounds[2] - self.component.bounds[0] + 2 * self.padding
    }
    pub fn height(&self) -> i32 {
        self.component.bounds[3] - self.component.bounds[1]
    }
    /// How far the shear moves page row `page_y` to the right.
    fn lean(&self, page_y: f64) -> f64 {
        let bounds = self.component.bounds;
        self.shear * (page_y - f64::from(bounds[1] + bounds[3] - 1) / 2.)
    }
    pub fn point(&self, x: f64, y: f64) -> Point {
        let bounds = self.component.bounds;
        let page_y = f64::from(bounds[3] - 1) - y;
        Point {
            x: f64::from(bounds[0] - self.padding) + x + self.lean(page_y),
            y: page_y,
        }
    }
    fn sheared_own(&self, x: i32, y: i32) -> bool {
        let p = self.point(f64::from(x), f64::from(y));
        self.pixels
            .own(self.component.id, p.x.round() as i32, p.y.round() as i32)
    }
    pub fn own(&self, x: i32, y: i32) -> bool {
        if x >= 0 && y >= 0 && x < self.width() && y < self.height() {
            self.own.get(y as usize, x as usize)
        } else {
            self.sheared_own(x, y)
        }
    }
    pub fn any_symbol_ink(&self, x: i32, y: i32) -> bool {
        let p = self.point(f64::from(x), f64::from(y));
        self.pixels.ink(p.x.round() as i32, p.y.round() as i32)
    }
    // There are no parser-created barriers in the standalone candidate plane.
    pub fn clear_of_barrier(&self, x: i32, y: i32) -> bool {
        !self.own(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{edge, probe};
    use super::*;
    // Same free-shaft/head-block geometry as the source measurements' fixture:
    // a two-pixel shaft above an eight-row head, read bottom-up.
    fn fixture() -> Pixels {
        let mut labels = vec![0; 20 * 45];
        for y in 0..40 {
            for x in 0..12 {
                if (5..7).contains(&x) || y >= 32 {
                    labels[y * 20 + x] = 1;
                }
            }
        }
        let mut ink = Plane::new(20, 45);
        for (i, _) in labels.iter().enumerate().filter(|(_, label)| **label != 0) {
            ink.set(i % 20, i / 20);
        }
        Pixels {
            width: 20,
            height: 45,
            origin: [0, 0],
            labels,
            ink,
            rule: Plane::new(20, 45),
            rules: vec![],
            line_radius: [0.; 5],
            components: vec![Component {
                id: 1,
                bounds: [0, 0, 12, 40],
            }],
        }
    }
    #[test]
    fn extracted_free_and_narrow_row_measurements_preserve_source_fixture_results() {
        let p = fixture();
        let v = View::new(&p, &p.components[0], 0., 0);
        let line = super::super::ported::types::StemLine {
            x: 5,
            left: 5,
            right: 6,
            y: 0,
            len: 40,
        };
        let free = edge::clear_run(&v, &line, 3, 8, false);
        assert_eq!((free.length, free.start), (32, 8));
        let narrow = edge::narrow_rows(
            &v,
            &super::super::ported::metrics::NoteMetrics::new(10),
            &line,
            10,
        );
        assert_eq!(
            (narrow.total, narrow.first_run, narrow.longest_run),
            (32, 0, 32)
        );
    }
    #[test]
    fn sheared_view_holds_the_ink_its_points_resolve_to() {
        let p = fixture();
        for shear in [-0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3] {
            for padding in [0, 3, 8] {
                let v = View::new(&p, &p.components[0], shear, padding);
                let mut inked = 0;
                for y in 0..v.height() {
                    for x in 0..v.width() {
                        assert_eq!(
                            v.own(x, y),
                            v.sheared_own(x, y),
                            "{shear} {padding} {x} {y}"
                        );
                        inked += usize::from(v.own(x, y));
                    }
                }
                assert!(inked > 0);
            }
        }
    }
    #[test]
    fn extracted_probe_keeps_candidate_geometry_and_abstains_on_capacity_exhaustion() {
        let p = fixture();
        let v = View::new(&p, &p.components[0], 0., 0);
        let found = probe::probe(&v, 10, 16, 5, None, 150);
        assert_eq!(
            found.lines,
            vec![probe::Line {
                left: 4,
                right: 9,
                top: 0,
                bottom: 40,
                centre: 6
            }]
        );
        assert!(probe::probe(&v, 10, 16, 5, None, 1).lines.is_empty());
    }
}
