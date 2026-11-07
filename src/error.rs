//! Errors shared by geometry, detection and rendering.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(&'static str),
    #[cfg(feature = "gpu-render")]
    #[error("rendering: {0}")]
    Rendering(String),
    #[cfg(feature = "onnx-pages")]
    #[error("ONNX page detection: {0}")]
    PageModel(String),
}
pub type Result<T> = std::result::Result<T, Error>;
