//! Adaptive inverse-coordinate map. The fitted geometry is never replaced:
//! evaluate it at grid vertices and probes, refine curved cells, and store
//! exact per-pixel coordinates wherever interpolation is insufficient.
use crate::geometry::fit::{Nearby, Scanline};
use crate::geometry::{Geometry, Point};
use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;

pub(super) const TILE: u32 = 32;
const TOLERANCE: f64 = 0.005;
const ORIGIN: Point = Point { x: 0., y: 0. };

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub(super) struct Node {
    pub corners: [[f32; 2]; 4],
    pub offset: u32,
    /// 0: bilinear cell, 1: four child nodes, 2: exact dense coordinates.
    pub mode: u32,
    padding: [u32; 2],
}

#[derive(Default)]
struct Tile {
    nodes: Vec<Node>,
    dense: Vec<[f32; 2]>,
    evaluations: usize,
}

pub(super) struct Map {
    pub roots: Vec<u32>,
    pub nodes: Vec<Node>,
    pub dense: Vec<[f32; 2]>,
    pub columns: u32,
    pub evaluations: usize,
    pub max_probe_error: f64,
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

/// The exact inverse at output pixels, for one tile at a time.
struct Exact<'a> {
    scanlines: &'a [Scanline<'a>],
    /// Output x of pixel column 0 and of the last one, `width - 1`.
    x: [f64; 2],
    width: u32,
    nearby: Nearby,
}
impl Exact<'_> {
    fn x(&self, x: u32) -> f64 {
        self.x[0] + (self.x[1] - self.x[0]) * f64::from(x) / f64::from(self.width - 1)
    }
    fn at(&mut self, x: u32, y: u32) -> Point {
        self.scanlines[y as usize].at(self.x(x), &mut self.nearby)
    }
    /// The pixels of row `y` in the given columns, at most a tile's worth.
    fn row(&mut self, y: u32, columns: impl Iterator<Item = u32>, out: &mut [Point]) {
        let mut xs = [0.; TILE as usize];
        let xs = &mut xs[..out.len()];
        for (x, value) in columns.zip(xs.iter_mut()) {
            *value = self.x(x);
        }
        self.scanlines[y as usize].fill(xs, out, &mut self.nearby);
    }
    /// The edge midpoints and centre of the `2 * half` pixel node at `(x, y)`
    /// together with its corners, row by row.
    fn lattice(&mut self, x: u32, y: u32, half: u32) -> [[Point; 3]; 3] {
        let mut lattice = [[ORIGIN; 3]; 3];
        for (row, j) in lattice.iter_mut().zip((y..).step_by(half as usize)) {
            self.row(j, (x..).step_by(half as usize), row);
        }
        lattice
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
}

impl Tile {
    fn dense(&mut self, index: usize, x: u32, y: u32, size: u32, exact: &mut Exact) {
        self.nodes[index].mode = 2;
        self.nodes[index].offset = self.dense.len() as u32;
        self.dense.reserve((size * size) as usize);
        let mut row = [ORIGIN; TILE as usize];
        let row = &mut row[..size as usize];
        for j in y..y + size {
            exact.row(j, x.., row);
            self.dense
                .extend(row.iter().map(|p| [p.x as f32, p.y as f32]));
        }
        self.evaluations += (size * size) as usize;
    }
    /// Subdivide the node until bilinear interpolation between its corners
    /// reproduces `probes`, its exact edge midpoints and centre.
    fn build(
        &mut self,
        index: usize,
        x: u32,
        y: u32,
        size: u32,
        probes: [Point; 5],
        exact: &mut Exact,
    ) {
        let corners = self.nodes[index].corners;
        let expected = [[0.5, 0.], [0., 0.5], [0.5, 0.5], [1., 0.5], [0.5, 1.]];
        let accurate = unfolded(corners)
            && probes
                .iter()
                .zip(expected)
                .all(|(p, t)| error(interpolate(corners, t[0], t[1]), *p) <= TOLERANCE);
        if accurate {
            return;
        }
        let half = size / 2;
        let p = probes.map(|p| [p.x as f32, p.y as f32]);
        let children = [
            [corners[0], p[0], p[1], p[2]],
            [p[0], corners[1], p[2], p[3]],
            [p[1], p[2], corners[2], p[4]],
            [p[2], p[3], p[4], corners[3]],
        ];
        let first = self.nodes.len();
        self.nodes[index].mode = 1;
        self.nodes[index].offset = first as u32;
        self.nodes.resize(first + 4, Node::default());
        // At the finest level evaluate every pixel, including odd pixel
        // positions between the interpolation probes. Close staff/text knots
        // can contain sub-cell structure even when the probes coincide.
        let probes = (half > 4).then(|| exact.child_probes(x, y, half / 2));
        self.evaluations += probes.map_or(0, |_| 16);
        for (k, corners) in children.into_iter().enumerate() {
            let (index, x, y) = (
                first + k,
                x + (k as u32 % 2) * half,
                y + (k as u32 / 2) * half,
            );
            self.nodes[index].corners = corners;
            match probes {
                Some(probes) => self.build(index, x, y, half, probes[k], exact),
                None => self.dense(index, x, y, half, exact),
            }
        }
    }
    /// Largest error on the part of the offset probe lattice inside this
    /// tile, whose origin is `(x, y)`.
    fn probe_error(&self, x: u32, y: u32, [width, height]: [u32; 2], exact: &mut Exact) -> f64 {
        let lattice = |first: u32, step: u32, lo: u32, end: u32| {
            let start = first + lo.saturating_sub(first).div_ceil(step) * step;
            (start..end.min(lo + TILE)).step_by(step as usize)
        };
        let mut maximum = 0_f64;
        for j in lattice(3, 19, y, height) {
            for i in lattice(5, 17, x, width) {
                maximum = maximum.max(error(self.lookup(i - x, j - y), exact.at(i, j)));
            }
        }
        maximum
    }
    fn lookup(&self, x: u32, y: u32) -> [f32; 2] {
        let (mut i, mut size, mut ox, mut oy) = (0_usize, TILE, 0, 0);
        loop {
            let node = self.nodes[i];
            if node.mode == 2 {
                return self.dense[node.offset as usize + ((y - oy) * size + x - ox) as usize];
            }
            if node.mode == 0 {
                return interpolate(
                    node.corners,
                    (x - ox) as f32 / size as f32,
                    (y - oy) as f32 / size as f32,
                );
            }
            size /= 2;
            let dx = u32::from(x >= ox + size);
            let dy = u32::from(y >= oy + size);
            ox += dx * size;
            oy += dy * size;
            i = (node.offset + dx + 2 * dy) as usize;
        }
    }
}

impl Map {
    pub fn build(geometry: &Geometry, width: u32, height: u32) -> Result<Self, String> {
        let columns = width.div_ceil(TILE);
        let rows = height.div_ceil(TILE);
        let [x0, y0, x1, y1] = geometry.report().bounds;
        let scanlines: Vec<_> = (0..=rows * TILE)
            .into_par_iter()
            .map(|j| {
                geometry.inverse_scanline(y0 + (y1 - y0) * f64::from(j) / f64::from(height - 1))
            })
            .collect();
        let tiles: Vec<_> = (0..columns * rows)
            .into_par_iter()
            .map(|id| {
                let (x, y) = ((id % columns) * TILE, (id / columns) * TILE);
                let mut exact = Exact {
                    scanlines: &scanlines,
                    x: [x0, x1],
                    width,
                    nearby: Nearby::default(),
                };
                let [[a, top, b], [left, centre, right], [c, bottom, d]] =
                    exact.lattice(x, y, TILE / 2);
                let corners = [a, b, c, d].map(|p| [p.x as f32, p.y as f32]);
                let mut tile = Tile {
                    nodes: vec![Node {
                        corners,
                        ..Node::default()
                    }],
                    evaluations: 9,
                    ..Tile::default()
                };
                tile.build(
                    0,
                    x,
                    y,
                    TILE,
                    [top, left, centre, right, bottom],
                    &mut exact,
                );
                // A second, offset probe lattice is independent of the
                // subdivision probes. Densify a failing tile rather than
                // letting a sampled local feature silently disappear between
                // mesh vertices.
                let mut probe_error = tile.probe_error(x, y, [width, height], &mut exact);
                if probe_error > TOLERANCE {
                    tile.nodes = vec![Node::default()];
                    tile.dense.clear();
                    tile.dense(0, x, y, TILE, &mut exact);
                    probe_error = tile.probe_error(x, y, [width, height], &mut exact);
                }
                (tile, probe_error)
            })
            .collect();
        let max_probe_error = tiles.iter().map(|(_, error)| *error).fold(0., f64::max);
        if max_probe_error > TOLERANCE {
            return Err("GPU map precision exceeds source-pixel tolerance".into());
        }
        let mut result = Self {
            roots: Vec::with_capacity(tiles.len()),
            nodes: vec![],
            dense: vec![],
            columns,
            evaluations: 0,
            max_probe_error,
        };
        for (tile, _) in tiles {
            let base = result.nodes.len() as u32;
            let dense_base = result.dense.len() as u32;
            result.roots.push(base);
            result.evaluations += tile.evaluations;
            result.nodes.extend(tile.nodes.into_iter().map(|mut n| {
                if n.mode == 1 {
                    n.offset += base;
                } else if n.mode == 2 {
                    n.offset += dense_base;
                }
                n
            }));
            result.dense.extend(tile.dense);
        }
        if result.dense.is_empty() {
            result.dense.push([0., 0.]);
        }
        Ok(result)
    }
}
