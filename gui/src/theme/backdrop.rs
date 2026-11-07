//! The soft light the web UI gave its backgrounds: CSS gradients and a
//! film of noise that keeps the faint gradients from banding.

use std::f32::consts::TAU;

use eframe::egui;
use eframe::egui::epaint::{Mesh, Vertex, WHITE_UV};
use eframe::egui::{
    Color32, ColorImage, Context, CornerRadius, Painter, Pos2, Rect, TextureHandle, TextureOptions,
    Vec2, pos2, vec2,
};

use super::color;

/// A radial gradient, as CSS `radial-gradient(ellipse <radii> at <center>,
/// <color> 0%, transparent <reach>)`.
#[derive(Clone, Copy, Debug)]
pub struct Glow {
    /// The center as a fraction of the rect.
    pub center: Vec2,
    /// The radii as a fraction of the rect's size.
    pub radii: Vec2,
    pub color: Color32,
    /// Where the glow has faded out, as a fraction of the radii.
    pub reach: f32,
}

impl Glow {
    /// The light falling from the top of every page.
    pub const PAGE: Glow = Glow {
        center: vec2(0.5, 0.0),
        radii: vec2(0.8, 0.6),
        color: accent_alpha(0.04),
        reach: 0.7,
    };
    /// The light behind the centered column of the start screens.
    pub const SPOTLIGHT: Glow = Glow {
        center: vec2(0.5, 0.5),
        radii: vec2(0.8, 0.6),
        color: accent_alpha(0.07),
        reach: 0.6,
    };
    /// The light at the left of the header, behind the brand.
    pub const HEADER: Glow = Glow {
        center: vec2(0.0, 0.5),
        radii: vec2(0.6, 1.0),
        color: accent_alpha(0.03),
        reach: 0.7,
    };

    /// Paint the glow over `rect`, clipped to it.
    pub fn paint(&self, painter: &Painter, rect: Rect) {
        const SEGMENTS: usize = 96;
        let center = rect.min + self.center * rect.size();
        let radii = self.reach * self.radii * rect.size();
        let mut mesh = Mesh::default();
        mesh.colored_vertex(center, self.color);
        for i in 0..SEGMENTS {
            let angle = i as f32 / SEGMENTS as f32 * TAU;
            mesh.colored_vertex(
                center + radii * vec2(angle.cos(), angle.sin()),
                Color32::TRANSPARENT,
            );
            let next = 1 + (i as u32 + 1) % SEGMENTS as u32;
            mesh.add_triangle(0, 1 + i as u32, next);
        }
        painter.with_clip_rect(rect).add(mesh);
    }
}

const fn accent_alpha(alpha: f32) -> Color32 {
    Color32::from_rgba_unmultiplied_const(0xff, 0xcc, 0x66, (alpha * 255.0 + 0.5) as u8)
}

/// The window background: the page color, its glow and the noise.
pub fn paint_page(painter: &Painter, rect: Rect) {
    painter.rect_filled(rect, 0, color::BG);
    Glow::PAGE.paint(painter, rect);
    paint_noise(painter, rect);
}

/// The fine noise the web UI laid over everything at 3.5% opacity.
pub fn paint_noise(painter: &Painter, rect: Rect) {
    let texture = noise_texture(painter.ctx());
    // One texel per physical pixel, as the browser drew it.
    let texels = rect.size() * painter.ctx().pixels_per_point() / NOISE_SIZE as f32;
    let uv = Rect::from_min_size(Pos2::ZERO, texels);
    painter.image(texture.id(), rect, uv, Color32::from_white_alpha(9));
}

const NOISE_SIZE: usize = 256;

/// Grey noise, made once per context; a hash keeps it the same every run.
fn noise_texture(ctx: &Context) -> TextureHandle {
    let id = egui::Id::new("wing_companion_noise");
    if let Some(texture) = ctx.data(|data| data.get_temp::<TextureHandle>(id)) {
        return texture;
    }
    let pixels = (0..NOISE_SIZE * NOISE_SIZE)
        .map(|i| {
            let mut x = i as u32 ^ 0x9e37_79b9;
            x = (x ^ (x >> 16)).wrapping_mul(0x7feb_352d);
            x = (x ^ (x >> 15)).wrapping_mul(0x846c_a68b);
            Color32::from_gray((x >> 24) as u8)
        })
        .collect();
    let image = ColorImage::new([NOISE_SIZE, NOISE_SIZE], pixels);
    let options = TextureOptions {
        wrap_mode: egui::TextureWrapMode::Repeat,
        ..TextureOptions::NEAREST
    };
    let texture = ctx.load_texture("noise", image, options);
    ctx.data_mut(|data| data.insert_temp(id, texture.clone()));
    texture
}

/// A rounded rect filled with a linear gradient at 135°, from `from` at the
/// top left corner to `to` at the bottom right, as the web UI's cards and
/// modals.
pub fn diagonal_gradient(
    rect: Rect,
    radius: impl Into<CornerRadius>,
    from: Color32,
    to: Color32,
) -> Mesh {
    // CSS measures the gradient along a line through the center at 135°,
    // long enough that the corners get exactly the end colors.
    let direction = vec2(1.0, 1.0).normalized();
    let length = (rect.width() + rect.height()) / 2f32.sqrt();
    let color_at = |pos: Pos2| {
        let t = (pos - rect.center()).dot(direction) / length + 0.5;
        lerp_premultiplied(from, to, t)
    };

    let mut mesh = Mesh::default();
    mesh.colored_vertex(rect.center(), color_at(rect.center()));
    let outline = rounded_rect_outline(rect, radius.into());
    for &pos in &outline {
        mesh.vertices.push(Vertex {
            pos,
            uv: WHITE_UV,
            color: color_at(pos),
        });
    }
    let n = outline.len() as u32;
    for i in 0..n {
        mesh.add_triangle(0, 1 + i, 1 + (i + 1) % n);
    }
    mesh
}

fn lerp_premultiplied(from: Color32, to: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let channel = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t).round() as u8;
    Color32::from_rgba_premultiplied(
        channel(from.r(), to.r()),
        channel(from.g(), to.g()),
        channel(from.b(), to.b()),
        channel(from.a(), to.a()),
    )
}

/// The points around a rounded rect, clockwise from its top left corner.
fn rounded_rect_outline(rect: Rect, radius: CornerRadius) -> Vec<Pos2> {
    const STEPS: usize = 8;
    let max = rect.width().min(rect.height()) / 2.0;
    let corners = [
        (pos2(rect.left(), rect.top()), radius.nw, 0.5),
        (pos2(rect.right(), rect.top()), radius.ne, 0.75),
        (pos2(rect.right(), rect.bottom()), radius.se, 0.0),
        (pos2(rect.left(), rect.bottom()), radius.sw, 0.25),
    ];
    let mut points = Vec::with_capacity(4 * (STEPS + 1));
    for (corner, r, start) in corners {
        let r = (r as f32).min(max);
        let inward = vec2(
            (rect.center().x - corner.x).signum(),
            (rect.center().y - corner.y).signum(),
        );
        let center = corner + inward * r;
        for step in 0..=STEPS {
            let angle = (start + step as f32 / STEPS as f32 * 0.25) * TAU;
            points.push(center + r * vec2(angle.cos(), angle.sin()));
        }
    }
    points
}
