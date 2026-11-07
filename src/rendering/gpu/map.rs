//! Adaptive inverse-coordinate map. The fitted geometry is never replaced:
//! evaluate it at grid vertices and probes, refine curved cells, and store
//! exact per-pixel coordinates wherever interpolation is insufficient.
use crate::geometry::{Geometry, Point};
use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;

pub(super) const TILE: u32 = 32;
const TOLERANCE: f64 = 0.005;

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

impl Tile {
    fn dense(&mut self, index: usize, x: u32, y: u32, size: u32, at: &impl Fn(u32, u32) -> Point) {
        self.nodes[index].mode = 2;
        self.nodes[index].offset = self.dense.len() as u32;
        for j in y..y + size {
            for i in x..x + size {
                let p = at(i, j);
                self.dense.push([p.x as f32, p.y as f32]);
            }
        }
        self.evaluations += (size * size) as usize;
    }
    fn build(
        &mut self,
        index: usize,
        x: u32,
        y: u32,
        size: u32,
        corners: [[f32; 2]; 4],
        at: &impl Fn(u32, u32) -> Point,
    ) {
        self.nodes[index].corners = corners;
        // At the finest level evaluate every pixel, including odd pixel
        // positions between the interpolation probes. Close staff/text knots
        // can contain sub-cell structure even when the probes coincide.
        if size <= 4 {
            self.dense(index, x, y, size, at);
            return;
        }
        let half = size / 2;
        let positions = [
            (x + half, y),
            (x, y + half),
            (x + half, y + half),
            (x + size, y + half),
            (x + half, y + size),
        ];
        let expected = [[0.5, 0.], [0., 0.5], [0.5, 0.5], [1., 0.5], [0.5, 1.]];
        let probes = positions.map(|(x, y)| at(x, y));
        self.evaluations += 5;
        let accurate = unfolded(corners)
            && probes
                .iter()
                .zip(expected)
                .all(|(p, t)| error(interpolate(corners, t[0], t[1]), *p) <= TOLERANCE);
        if accurate {
            return;
        }
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
        for (k, c) in children.into_iter().enumerate() {
            self.build(
                first + k,
                x + (k as u32 % 2) * half,
                y + (k as u32 / 2) * half,
                half,
                c,
                at,
            );
        }
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
            .map(|j| {
                geometry.inverse_scanline(y0 + (y1 - y0) * f64::from(j) / f64::from(height - 1))
            })
            .collect();
        let at = |x: u32, y: u32| {
            scanlines[y as usize](x0 + (x1 - x0) * f64::from(x) / f64::from(width - 1))
        };
        let mut tiles: Vec<_> = (0..columns * rows)
            .into_par_iter()
            .map(|id| {
                let (x, y) = ((id % columns) * TILE, (id / columns) * TILE);
                let corners = [
                    at(x, y),
                    at(x + TILE, y),
                    at(x, y + TILE),
                    at(x + TILE, y + TILE),
                ]
                .map(|p| [p.x as f32, p.y as f32]);
                let mut tile = Tile {
                    nodes: vec![Node::default()],
                    evaluations: 4,
                    ..Tile::default()
                };
                tile.build(0, x, y, TILE, corners, &at);
                tile
            })
            .collect();
        // A second, offset probe lattice is independent of the subdivision
        // probes. Densify failing tiles rather than letting a sampled local
        // feature silently disappear between mesh vertices.
        let mut failed = vec![false; tiles.len()];
        for y in (3..height).step_by(19) {
            for x in (5..width).step_by(17) {
                let id = (y / TILE * columns + x / TILE) as usize;
                if error(tiles[id].lookup(x % TILE, y % TILE), at(x, y)) > TOLERANCE {
                    failed[id] = true;
                }
            }
        }
        tiles
            .par_iter_mut()
            .enumerate()
            .filter(|(i, _)| failed[*i])
            .for_each(|(i, tile)| {
                let (x, y) = ((i as u32 % columns) * TILE, (i as u32 / columns) * TILE);
                tile.nodes = vec![Node::default()];
                tile.dense.clear();
                tile.dense(0, x, y, TILE, &at);
            });
        let mut max_probe_error = 0_f64;
        for y in (3..height).step_by(19) {
            for x in (5..width).step_by(17) {
                let id = (y / TILE * columns + x / TILE) as usize;
                max_probe_error =
                    max_probe_error.max(error(tiles[id].lookup(x % TILE, y % TILE), at(x, y)));
            }
        }
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
        for tile in tiles {
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
