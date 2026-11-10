//! Thread-pool access for modules that also build without one.

/// `items.iter().map(f).collect()`, in the same order. The items are shared
/// among the pool's threads where the `render` feature provides a pool.
pub(crate) fn map<T: Sync, U: Send>(items: &[T], f: impl Fn(&T) -> U + Sync + Send) -> Vec<U> {
    #[cfg(feature = "render")]
    {
        use rayon::prelude::*;
        items.par_iter().map(f).collect()
    }
    #[cfg(not(feature = "render"))]
    {
        items.iter().map(f).collect()
    }
}
