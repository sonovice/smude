//! Adaptive inverse-coordinate map. The fitted geometry is never replaced:
//! evaluate it at grid vertices and probes, refine curved cells, and store
//! exact per-pixel coordinates wherever interpolation is insufficient.
use crate::geometry::fit::{Nearby, Scanline};
use crate::geometry::{Geometry, Point};
use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;

pub(super) const TILE: u32 = 32;
/// A node this small stores every pixel instead of subdividing further.
const BLOCK: u32 = 8;
/// Largest accepted interpolation error in source pixels. A fiftieth of a
/// pixel moves a bicubic sample by less than one grey level almost everywhere.
const TOLERANCE: f64 = 0.02;
const ORIGIN: Point = Point { x: 0., y: 0. };

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub(super) struct Node {
    pub corners: [[f32; 2]; 4],
    /// Of the first child, counted from the root of the tile, or of the first
    /// exact coordinate, counted from the first one of the tile.
    pub offset: u32,
    /// 0: bilinear cell, 1: four child nodes, 2: exact dense coordinates.
    pub mode: u32,
    padding: [u32; 2],
}

/// One row of tiles. No offset in it depends on the strips before it.
#[derive(Default)]
pub(super) struct Strip {
    pub nodes: Vec<Node>,
    pub dense: Vec<[f32; 2]>,
    /// The root node and the first exact coordinate of every tile.
    roots: Vec<[u32; 2]>,
    evaluations: usize,
    probe_error: f64,
}

/// Consecutive strips. Its nodes and exact coordinates are those of its
/// strips, one strip after another.
pub(super) struct Band {
    pub strips: Vec<Strip>,
    pub evaluations: usize,
    pub max_probe_error: f64,
}

/// The inverse geometry of an output raster as `rows` by `columns` tiles,
/// built band by band.
pub(super) struct Map<'a> {
    pub columns: u32,
    pub rows: u32,
    /// Those every band shares.
    pub evaluations: usize,
    exact: Exact<'a>,
    /// The corners, edge midpoints and centres of all tiles.
    corners: Lattice,
    /// An offset lattice, independent of the subdivision probes.
    probes: Lattice,
}

fn finite(p: [f32; 2]) -> bool {
    p.iter().all(|v| v.is_finite())
}
fn error(a: [f32; 2], b: Point) -> f64 {
    if !finite(a) || !b.x.is_finite() || !b.y.is_finite() {
        if !finite(a) && (!b.x.is_finite() || !b.y.is_finite()) {
            0.
        } else {
            f64::INFINITY
        }
    } else {
        (f64::from(a[0]) - b.x).hypot(f64::from(a[1]) - b.y)
    }
}
fn single(p: Point) -> [f32; 2] {
    [p.x as f32, p.y as f32]
}
fn mix(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
    std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t)
}
fn interpolate(c: [[f32; 2]; 4], x: f32, y: f32) -> [f32; 2] {
    mix(mix(c[0], c[1], x), mix(c[2], c[3], x), y)
}
fn unfolded(c: [[f32; 2]; 4]) -> bool {
    if !c.into_iter().all(finite) {
        return false;
    }
    let sub = |a: [f32; 2], b: [f32; 2]| [a[0] - b[0], a[1] - b[1]];
    let det = |a: [f32; 2], b: [f32; 2]| a[0] * b[1] - a[1] * b[0];
    let (top, bottom, left, right) = (
        sub(c[1], c[0]),
        sub(c[3], c[2]),
        sub(c[2], c[0]),
        sub(c[3], c[1]),
    );
    [
        det(top, left),
        det(top, right),
        det(bottom, left),
        det(bottom, right),
    ]
    .into_iter()
    .all(|d| d > 0.)
}

/// What the nodes of a tile, the root first, interpolate at the pixel
/// `(x, y)` in it, unless its coordinate is stored exactly.
fn interpolated(nodes: &[Node], x: u32, y: u32) -> Option<[f32; 2]> {
    let (mut i, mut size, mut ox, mut oy) = (0_usize, TILE, 0, 0);
    loop {
        let node = nodes[i];
        if node.mode == 2 {
            return None;
        }
        if node.mode == 0 {
            return Some(interpolate(
                node.corners,
                (x - ox) as f32 / size as f32,
                (y - oy) as f32 / size as f32,
            ));
        }
        size /= 2;
        let dx = u32::from(x >= ox + size);
        let dy = u32::from(y >= oy + size);
        ox += dx * size;
        oy += dy * size;
        i = (node.offset + dx + 2 * dy) as usize;
    }
}

/// The exact inverse at output pixels.
struct Exact<'a> {
    scanlines: Vec<Scanline<'a>>,
    /// Output x of pixel column 0 and of the last one, `width - 1`.
    x: [f64; 2],
    width: u32,
}
impl Exact<'_> {
    fn x(&self, x: u32) -> f64 {
        self.x[0] + (self.x[1] - self.x[0]) * f64::from(x) / f64::from(self.width - 1)
    }
}

/// The exact inverse at every `step`-th pixel from `first` on, in both
/// directions, each row evaluated in one batch.
struct Lattice {
    first: [u32; 2],
    step: [u32; 2],
    size: [u32; 2],
    points: Vec<Point>,
}
impl Lattice {
    /// Covers the pixels before `end`.
    fn new(exact: &Exact, first: [u32; 2], step: [u32; 2], end: [u32; 2]) -> Self {
        let size: [u32; 2] =
            std::array::from_fn(|k| end[k].saturating_sub(first[k]).div_ceil(step[k]));
        let xs: Vec<_> = (0..size[0])
            .map(|i| exact.x(first[0] + i * step[0]))
            .collect();
        let mut points = vec![ORIGIN; (size[0] * size[1]) as usize];
        points
            .par_chunks_mut(xs.len().max(1))
            .zip(
                exact
                    .scanlines
                    .par_iter()
                    .skip(first[1] as usize)
                    .step_by(step[1] as usize),
            )
            .for_each(|(row, scanline)| scanline.fill(&xs, row, &mut Nearby::default()));
        Self {
            first,
            step,
            size,
            points,
        }
    }
    /// The point `(i, j)` of the lattice.
    fn point(&self, i: u32, j: u32) -> Point {
        self.points[(j * self.size[0] + i) as usize]
    }
    /// The lattice pixels inside the tile at `(x, y)` with their points.
    fn within(&self, x: u32, y: u32) -> impl Iterator<Item = (u32, u32, Point)> + '_ {
        let range = |k: usize, lo: u32| {
            let index = |pixel: u32| pixel.saturating_sub(self.first[k]).div_ceil(self.step[k]);
            index(lo)..index(lo + TILE).min(self.size[k])
        };
        range(1, y).flat_map(move |j| {
            range(0, x).map(move |i| {
                let pixel = [i, j];
                let [x, y] = std::array::from_fn(|k| self.first[k] + pixel[k] * self.step[k]);
                (x, y, self.point(i, j))
            })
        })
    }
}

/// A run of pixels in each of `BLOCK` consecutive rows, part of a node of
/// `width` pixels that stores its exact coordinates from `first` on.
struct Span {
    x: u32,
    width: u32,
    first: usize,
    /// The row of the node the rows start with.
    row: u32,
}

/// A strip under construction, the row of tiles at `y`.
struct Builder<'a> {
    exact: &'a Exact<'a>,
    nearby: Nearby,
    y: u32,
    strip: Strip,
    /// The exact coordinates still to evaluate, for every `BLOCK` pixel rows.
    pending: [Vec<Span>; (TILE / BLOCK) as usize],
}

impl Builder<'_> {
    fn strip(mut self, columns: u32, corners: &Lattice, probes: &Lattice) -> Strip {
        for column in 0..columns {
            self.tile(column * TILE, corners, probes);
        }
        self.fill();
        self.strip
    }
    /// The tile at `x`, exact coordinates left pending.
    fn tile(&mut self, x: u32, corners: &Lattice, probes: &Lattice) {
        let (i, j) = (x / (TILE / 2), self.y / (TILE / 2));
        let [[a, top, b], [left, centre, right], [c, bottom, d]] =
            [0, 1, 2].map(|v| [0, 1, 2].map(|u| corners.point(i + u, j + v)));
        let tile = [self.strip.nodes.len(), self.strip.dense.len()];
        let pending = self.pending.each_ref().map(Vec::len);
        self.strip.roots.push(tile.map(|first| first as u32));
        self.strip.nodes.push(Node {
            corners: [a, b, c, d].map(single),
            ..Node::default()
        });
        self.build(
            tile,
            tile[0],
            x,
            self.y,
            TILE,
            [top, left, centre, right, bottom],
        );
        // Densify a tile failing on the second probe lattice rather than
        // letting a sampled local feature silently disappear between mesh
        // vertices.
        let mut probe_error = self.probe_error(tile[0], x, probes);
        if probe_error > TOLERANCE {
            self.strip.nodes.truncate(tile[0] + 1);
            self.strip.dense.truncate(tile[1]);
            for (pending, length) in self.pending.iter_mut().zip(pending) {
                pending.truncate(length);
            }
            let known = [a, top, left, centre].map(single);
            self.store(tile, tile[0], x, self.y, TILE, known);
            probe_error = self.probe_error(tile[0], x, probes);
        }
        self.strip.probe_error = self.strip.probe_error.max(probe_error);
    }
    /// Subdivide the node until bilinear interpolation between its corners
    /// reproduces `probes`, its exact edge midpoints and centre. `tile` is
    /// the root node and the first exact coordinate of its tile.
    fn build(
        &mut self,
        tile: [usize; 2],
        index: usize,
        x: u32,
        y: u32,
        size: u32,
        probes: [Point; 5],
    ) {
        let corners = self.strip.nodes[index].corners;
        let expected = [[0.5, 0.], [0., 0.5], [0.5, 0.5], [1., 0.5], [0.5, 1.]];
        let accurate = unfolded(corners)
            && probes
                .iter()
                .zip(expected)
                .all(|(p, t)| error(interpolate(corners, t[0], t[1]), *p) <= TOLERANCE);
        if accurate {
            return;
        }
        let p = probes.map(single);
        if size == BLOCK {
            // At the finest level evaluate every pixel, including odd pixel
            // positions between the interpolation probes. Close staff/text
            // knots can contain sub-cell structure even when the probes
            // coincide.
            self.store(tile, index, x, y, size, [corners[0], p[0], p[1], p[2]]);
            return;
        }
        let half = size / 2;
        let children = [
            [corners[0], p[0], p[1], p[2]],
            [p[0], corners[1], p[2], p[3]],
            [p[1], p[2], corners[2], p[4]],
            [p[2], p[3], p[4], corners[3]],
        ];
        let first = self.strip.nodes.len();
        self.strip.nodes[index].mode = 1;
        self.strip.nodes[index].offset = (first - tile[0]) as u32;
        self.strip.nodes.resize(first + 4, Node::default());
        let probes = self.child_probes(x, y, half / 2);
        self.strip.evaluations += 16;
        for (k, corners) in children.into_iter().enumerate() {
            let (index, x, y) = (
                first + k,
                x + (k as u32 % 2) * half,
                y + (k as u32 / 2) * half,
            );
            self.strip.nodes[index].corners = corners;
            self.build(tile, index, x, y, half, probes[k]);
        }
    }
    /// The pixels of row `y` in the given columns, at most a tile's worth.
    fn row(&mut self, y: u32, columns: impl Iterator<Item = u32>, out: &mut [Point]) {
        let mut xs = [0.; TILE as usize];
        let xs = &mut xs[..out.len()];
        for (x, value) in columns.zip(xs.iter_mut()) {
            *value = self.exact.x(x);
        }
        self.exact.scanlines[y as usize].fill(xs, out, &mut self.nearby);
    }
    /// The probes of the four children of the `4 * quarter` pixel node at
    /// `(x, y)`. Children share the probe on a common edge, and the probes of
    /// one row are evaluated together.
    fn child_probes(&mut self, x: u32, y: u32, quarter: u32) -> [[Point; 5]; 4] {
        let step = quarter as usize;
        let mut lattice = [[ORIGIN; 5]; 5];
        for ((b, row), j) in lattice.iter_mut().enumerate().zip((y..).step_by(step)) {
            if b % 2 == 1 {
                self.row(j, (x..).step_by(step), row);
            } else {
                // The even columns of an even row are the node's own corners
                // and probes.
                let mut odd = [ORIGIN; 2];
                self.row(j, (x + quarter..).step_by(2 * step), &mut odd);
                [row[1], row[3]] = odd;
            }
        }
        std::array::from_fn(|k| {
            let (a, b) = (2 * (k % 2), 2 * (k / 2));
            [
                lattice[b][a + 1],
                lattice[b + 1][a],
                lattice[b + 1][a + 1],
                lattice[b + 1][a + 2],
                lattice[b + 2][a + 1],
            ]
        })
    }
    /// Makes the node store the exact coordinate of every pixel. `known` are
    /// those of its first pixel, the pixels half its size to the right of and
    /// below that one, and its centre pixel; the others stay pending.
    fn store(
        &mut self,
        tile: [usize; 2],
        index: usize,
        x: u32,
        y: u32,
        size: u32,
        known: [[f32; 2]; 4],
    ) {
        let first = self.strip.dense.len();
        self.strip.nodes[index].mode = 2;
        self.strip.nodes[index].offset = (first - tile[1]) as u32;
        self.strip
            .dense
            .resize(first + (size * size) as usize, [0.; 2]);
        for (k, known) in (0..).zip(known) {
            self.strip.dense[first + ((k / 2 * size + k % 2) * (size / 2)) as usize] = known;
        }
        let block = ((y - self.y) / BLOCK) as usize;
        for (pending, row) in self.pending[block..]
            .iter_mut()
            .zip((0..size).step_by(BLOCK as usize))
        {
            pending.push(Span {
                x,
                width: size,
                first,
                row,
            });
        }
    }
    /// Evaluates the pending coordinates, all of a pixel row in one batch.
    fn fill(&mut self) {
        let (mut xs, mut slots, mut points) = (vec![], vec![], vec![]);
        let rows = self
            .pending
            .iter()
            .flat_map(|spans| (0..BLOCK).map(move |row| (spans, row)));
        for (y, (spans, block_row)) in (self.y..).zip(rows) {
            xs.clear();
            slots.clear();
            for span in spans {
                let (row, half) = (span.row + block_row, span.width / 2);
                let known = |v: u32| v.is_multiple_of(half);
                let unknown = |column: &u32| !(known(row) && known(*column));
                for column in (0..span.width).filter(unknown) {
                    xs.push(self.exact.x(span.x + column));
                    slots.push(span.first + (row * span.width + column) as usize);
                }
            }
            points.resize(xs.len(), ORIGIN);
            self.exact.scanlines[y as usize].fill(&xs, &mut points, &mut self.nearby);
            for (slot, point) in slots.iter().zip(&points) {
                self.strip.dense[*slot] = single(*point);
            }
            self.strip.evaluations += xs.len();
        }
    }
    /// Largest error of the tile at `x` with the root node `root` on the part
    /// of the offset probe lattice inside it.
    fn probe_error(&self, root: usize, x: u32, probes: &Lattice) -> f64 {
        probes
            .within(x, self.y)
            .map(|(i, j, exact)| {
                // An exact coordinate is still pending and will be this very
                // point.
                let stored = interpolated(&self.strip.nodes[root..], i - x, j - self.y)
                    .unwrap_or(single(exact));
                error(stored, exact)
            })
            .fold(0., f64::max)
    }
}

impl<'a> Map<'a> {
    pub fn new(geometry: &'a Geometry, width: u32, height: u32) -> Self {
        let columns = width.div_ceil(TILE);
        let rows = height.div_ceil(TILE);
        let [x0, y0, x1, y1] = geometry.report().bounds;
        let scanlines = (0..=rows * TILE)
            .into_par_iter()
            .map(|j| {
                geometry.inverse_scanline(y0 + (y1 - y0) * f64::from(j) / f64::from(height - 1))
            })
            .collect();
        let exact = Exact {
            scanlines,
            x: [x0, x1],
            width,
        };
        let corners = Lattice::new(
            &exact,
            [0; 2],
            [TILE / 2; 2],
            [columns * TILE + 1, rows * TILE + 1],
        );
        let probes = Lattice::new(&exact, [5, 3], [17, 19], [width, height]);
        Self {
            columns,
            rows,
            evaluations: corners.points.len(),
            exact,
            corners,
            probes,
        }
    }
    /// The strips of the given rows of tiles.
    pub fn band(&self, rows: std::ops::Range<u32>) -> Result<Band, String> {
        let strips: Vec<_> = rows
            .into_par_iter()
            .map(|row| {
                let builder = Builder {
                    exact: &self.exact,
                    nearby: Nearby::default(),
                    y: row * TILE,
                    strip: Strip::default(),
                    pending: Default::default(),
                };
                builder.strip(self.columns, &self.corners, &self.probes)
            })
            .collect();
        let max_probe_error = strips.iter().map(|s| s.probe_error).fold(0., f64::max);
        if max_probe_error > TOLERANCE {
            return Err("GPU map precision exceeds source-pixel tolerance".into());
        }
        Ok(Band {
            evaluations: strips.iter().map(|s| s.evaluations).sum(),
            strips,
            max_probe_error,
        })
    }
}

impl Band {
    pub fn nodes(&self) -> usize {
        self.strips.iter().map(|s| s.nodes.len()).sum()
    }
    pub fn dense(&self) -> usize {
        self.strips.iter().map(|s| s.dense.len()).sum()
    }
    /// The root node and the first exact coordinate of every tile, counted
    /// through all strips.
    pub fn roots(&self) -> impl Iterator<Item = [u32; 2]> + '_ {
        let mut next = [0; 2];
        self.strips.iter().flat_map(move |strip| {
            let first = next;
            next[0] += strip.nodes.len() as u32;
            next[1] += strip.dense.len() as u32;
            strip
                .roots
                .iter()
                .map(move |root| [first[0] + root[0], first[1] + root[1]])
        })
    }
}
