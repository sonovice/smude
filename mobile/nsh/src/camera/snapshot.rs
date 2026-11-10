//! Bounded analysis-frame snapshots. Original camera photographs are never decoded here.
use image::{
    ImageEncoder, RgbImage,
    codecs::png::{CompressionType, FilterType, PngEncoder},
};
use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

#[cfg(any(target_os = "ios", target_os = "android", test))]
const MAX_FRAMES: usize = 4;
#[cfg(any(target_os = "ios", target_os = "android", test))]
static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static FRAMES: OnceLock<Mutex<VecDeque<Arc<CaptureFrame>>>> = OnceLock::new();

pub struct CaptureFrame {
    pub id: u64,
    pub image: RgbImage,
    pub polygons: Vec<Vec<[f32; 2]>>,
}
#[cfg(any(target_os = "ios", target_os = "android", test))]
pub fn remember(image: &RgbImage, polygons: Vec<Vec<[f32; 2]>>) -> Option<u64> {
    if image.width().max(image.height()) > 2048
        || polygons.is_empty()
        || polygons.len() > 2
        || polygons.iter().any(|p| bounds(p).is_none())
    {
        return None;
    }
    let frame = Arc::new(CaptureFrame {
        id: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        image: image.clone(),
        polygons,
    });
    let id = frame.id;
    let mut frames = FRAMES.get_or_init(Default::default).lock().ok()?;
    frames.push_back(frame);
    while frames.len() > MAX_FRAMES {
        frames.pop_front();
    }
    Some(id)
}
pub fn get(id: u64) -> Option<Arc<CaptureFrame>> {
    FRAMES
        .get()?
        .lock()
        .ok()?
        .iter()
        .find(|f| f.id == id)
        .cloned()
}
/// normalized bounding rectangle; malformed geometry never reaches the renderer.
pub fn bounds(polygon: &[[f32; 2]]) -> Option<[f32; 4]> {
    if polygon.len() < 3
        || polygon.len() > 128
        || polygon
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
    {
        return None;
    }
    let x = polygon.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
    let y = polygon.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
    let right = polygon.iter().map(|p| p[0]).fold(0., f32::max);
    let bottom = polygon.iter().map(|p| p[1]).fold(0., f32::max);
    (right - x > 0.005 && bottom - y > 0.005).then_some([x, y, right - x, bottom - y])
}
/// A session-scoped directory survives outstanding encoding jobs, then cleans up.
#[derive(PartialEq)]
pub struct SessionFiles {
    pub path: PathBuf,
}
impl SessionFiles {
    pub fn new(root: &Path) -> Self {
        Self {
            path: root
                .join(".camera-previews")
                .join(crate::model::id().to_string()),
        }
    }
}
impl Drop for SessionFiles {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
pub struct Snapshot {
    pub path: PathBuf,
    pub frame: Arc<CaptureFrame>,
    _session: Arc<SessionFiles>,
}
impl Drop for Snapshot {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
pub fn write(
    frame: Arc<CaptureFrame>,
    session: Arc<SessionFiles>,
    shot: u64,
) -> Result<Arc<Snapshot>, String> {
    std::fs::create_dir_all(&session.path).map_err(|e| e.to_string())?;
    let path = session.path.join(format!("{shot}.png"));
    let file = std::fs::File::create(&path).map_err(|e| e.to_string())?;
    // One small opaque image is shared by both page canvases; vector clipping
    // supplies transparent curved edges without rasterizing two extra bitmaps.
    PngEncoder::new_with_quality(file, CompressionType::Fast, FilterType::Sub)
        .write_image(
            frame.image.as_raw(),
            frame.image.width(),
            frame.image.height(),
            image::ExtendedColorType::Rgb8,
        )
        .map_err(|e| e.to_string())?;
    Ok(Arc::new(Snapshot {
        path,
        frame,
        _session: session,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shape_validation_rejects_unbounded_and_degenerate_crops() {
        assert!(bounds(&[[0., 0.], [1., 0.], [1., 1.]]).is_some());
        assert!(bounds(&[[0., 0.], [0., 0.], [0., 0.]]).is_none());
        assert!(bounds(&[[f32::NAN, 0.], [1., 0.], [1., 1.]]).is_none());
        assert!(bounds(&[[-1., 0.], [1., 0.], [1., 1.]]).is_none());
    }
    #[test]
    fn snapshot_preserves_pixels_and_cleans_up_after_the_last_owner() {
        let root =
            std::env::temp_dir().join(format!("smude-camera-snapshot-{}", crate::model::id()));
        let session = Arc::new(SessionFiles::new(&root));
        let directory = session.path.clone();
        let frame = Arc::new(CaptureFrame {
            id: 1,
            image: RgbImage::from_pixel(40, 60, image::Rgb([24, 80, 160])),
            polygons: vec![vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]]],
        });
        let snapshot = write(frame, session.clone(), 7).unwrap();
        let path = snapshot.path.clone();
        assert_eq!(
            image::open(&path).unwrap().to_rgb8().get_pixel(20, 30).0,
            [24, 80, 160]
        );
        drop(session);
        assert!(directory.exists());
        drop(snapshot);
        assert!(!path.exists());
        assert!(!directory.exists());
        let _ = std::fs::remove_dir_all(root);
    }
}
