//! Input decoding.
//!
//! Phone cameras end every row of JPEG blocks with a restart marker, after
//! which the entropy coding starts afresh. Such a file is cut into bands of
//! rows that the `image` crate decodes side by side; the result agrees with
//! its decode of the whole file sample for sample.

use std::{fs, io::Cursor, ops::Range, path::Path};

use image::{
    ColorType, ImageDecoder, ImageError, ImageFormat, ImageReader, ImageResult, RgbImage,
    codecs::jpeg::JpegDecoder,
};
use rayon::prelude::*;

/// `image::open(path)?.into_rgb8()`.
pub(super) fn open_rgb(path: &Path) -> ImageResult<RgbImage> {
    let bytes = fs::read(path).map_err(ImageError::IoError)?;
    let format = ImageFormat::from_path(path);
    let threads = rayon::current_num_threads();
    if threads > 1
        && matches!(format, Ok(ImageFormat::Jpeg))
        && let Some(image) = RestartRows::parse(&bytes).and_then(|rows| {
            // Four bands a thread even out their unequal cost. Each band
            // decodes two intervals beyond its own, so it stays much longer.
            rows.decode(rows.intervals.len().div_ceil(4 * threads).max(16))
        })
    {
        return Ok(image);
    }
    let mut reader = ImageReader::new(Cursor::new(bytes));
    if let Ok(format) = format {
        reader.set_format(format);
    }
    Ok(reader.decode()?.into_rgb8())
}

/// A baseline colour JPEG of one scan whose restart intervals are whole rows
/// of blocks.
struct RestartRows<'a> {
    /// The stream up to the entropy-coded data.
    head: &'a [u8],
    /// Where `head` states the number of pixel rows.
    height_at: usize,
    width: u32,
    height: u32,
    /// Pixel rows in a restart interval.
    rows: usize,
    /// The entropy-coded data of each restart interval, without its marker.
    intervals: Vec<&'a [u8]>,
}

impl<'a> RestartRows<'a> {
    fn parse(bytes: &'a [u8]) -> Option<Self> {
        const SCAN: usize = 1 << 16;
        let mut at = bytes.strip_prefix(&[0xFF, 0xD8]).map(|_| 2)?;
        let (mut frame, mut restart) = (None, 0);
        let scan_components = loop {
            let &[0xFF, marker, high, low] = bytes.get(at..at + 4)? else {
                return None;
            };
            let body = at + 4..at + 2 + usize::from(u16::from_be_bytes([high, low]));
            at = body.end;
            let body = bytes.get(body)?;
            match marker {
                0xC0 | 0xC1 if frame.is_none() => frame = Some((at - body.len(), body)),
                0xDD => restart = usize::from(u16::from_be_bytes(*body.first_chunk()?)),
                0xDA => break *body.first()?,
                // Tables, application data and comments.
                0xC4 | 0xDB | 0xE0..=0xEF | 0xFE => {}
                _ => return None,
            }
        };
        let (frame_at, frame) = frame?;
        let &[8, h1, h0, w1, w0, 3, ref components @ ..] = frame else {
            return None;
        };
        let (width, height) = (u16::from_be_bytes([w1, w0]), u16::from_be_bytes([h1, h0]));
        let (components, _) = components.as_chunks::<3>();
        let block = |shift: u8| components.iter().map(|c| c[1] >> shift & 15).max();
        let (block_width, block_height) = (8 * usize::from(block(4)?), 8 * usize::from(block(0)?));
        let blocks_across = usize::from(width).div_ceil(block_width.max(1));
        if scan_components != 3
            || components.len() != 3
            || block_width * block_height * usize::from(width) * usize::from(height) == 0
            || restart == 0
            || restart % blocks_across != 0
        {
            return None;
        }
        let rows = restart / blocks_across * block_height;
        // A stuffed zero follows every 0xFF of the coded data itself.
        let scan = &bytes[at..];
        let mut markers = (0..scan.len().div_ceil(SCAN))
            .into_par_iter()
            .flat_map_iter(|part| {
                let first = part * SCAN;
                let pairs = scan[first..].windows(2).take(SCAN).enumerate();
                pairs
                    .filter(|(_, pair)| pair[0] == 0xFF && pair[1] != 0)
                    .map(move |(at, pair)| (first + at, pair[1]))
            })
            .collect::<Vec<_>>()
            .into_iter();
        let mut intervals = Vec::with_capacity(usize::from(height).div_ceil(rows));
        let mut from = 0;
        loop {
            let (at, marker) = markers.next()?;
            let restarts = intervals.len();
            intervals.push(&scan[from..at]);
            from = at + 2;
            match marker {
                0xD9 => break,
                marker if marker == 0xD0 + (restarts % 8) as u8 => {}
                _ => return None,
            }
        }
        (intervals.len() == usize::from(height).div_ceil(rows)).then_some(Self {
            head: &bytes[..at],
            height_at: frame_at + 1,
            width: width.into(),
            height: height.into(),
            rows,
            intervals,
        })
    }

    /// The image, decoded as bands of `span` intervals. Chroma is interpolated
    /// from the block rows above and below, so every band is decoded together
    /// with the interval on either side and gives up those rows afterwards.
    fn decode(&self, span: usize) -> Option<RgbImage> {
        let row = 3 * self.width as usize;
        let mut pixels = vec![0; row * self.height as usize];
        pixels
            .par_chunks_mut(row * self.rows * span)
            .enumerate()
            .try_for_each_init(Vec::new, |band, (index, pixels)| {
                let first = index * span;
                let above = usize::from(first > 0);
                let intervals = first - above..(first + span + 1).min(self.intervals.len());
                let height = (intervals.end * self.rows).min(self.height as usize)
                    - intervals.start * self.rows;
                let decoder = JpegDecoder::new(Cursor::new(self.stream(intervals, height))).ok()?;
                if decoder.color_type() != ColorType::Rgb8
                    || decoder.dimensions() != (self.width, height as u32)
                {
                    return None;
                }
                band.resize(row * height, 0);
                decoder.read_image(band).ok()?;
                pixels.copy_from_slice(&band[row * self.rows * above..][..pixels.len()]);
                Some(())
            })?;
        RgbImage::from_raw(self.width, self.height, pixels)
    }

    /// A stream of the given intervals alone, `height` pixel rows tall.
    fn stream(&self, intervals: Range<usize>, height: usize) -> Vec<u8> {
        let mut stream = self.head.to_vec();
        stream[self.height_at..][..2].copy_from_slice(&(height as u16).to_be_bytes());
        for (index, interval) in self.intervals[intervals].iter().enumerate() {
            if index > 0 {
                stream.extend([0xFF, 0xD0 + ((index - 1) % 8) as u8]);
            }
            stream.extend_from_slice(interval);
        }
        stream.extend([0xFF, 0xD9]);
        stream
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    #[test]
    fn bands_of_restart_rows_reproduce_the_whole_decode() {
        for (name, intervals) in [
            ("decode/restart-rows.jpg", 18),
            ("decode/restart-row-pairs.jpg", 5),
            ("decode/restart-rows-422.jpg", 12),
            ("decode/restart-rows-440.jpg", 6),
            ("decode/restart-rows-444.jpg", 12),
        ] {
            let bytes = fs::read(fixture(name)).unwrap();
            let whole = image::load_from_memory(&bytes).unwrap().into_rgb8();
            let rows = RestartRows::parse(&bytes).unwrap();
            assert_eq!(rows.intervals.len(), intervals, "{name}");
            for span in [1, 2, 3, 7, intervals] {
                assert!(
                    rows.decode(span).unwrap() == whole,
                    "{name} in bands of {span} intervals"
                );
            }
        }
    }

    #[test]
    fn only_whole_rows_between_restarts_are_cut_into_bands() {
        let rows = fs::read(fixture("decode/restart-rows.jpg")).unwrap();
        let interval = rows
            .windows(2)
            .position(|pair| pair == [0xFF, 0xDD])
            .unwrap()
            + 5;
        let mut partial_rows = rows.clone();
        partial_rows[interval] -= 1;
        let mut missing_marker = rows.clone();
        let marker = rows.windows(2).rposition(|pair| pair == [0xFF, 0xD3]);
        missing_marker.drain(marker.unwrap()..marker.unwrap() + 2);
        let unmarked = fs::read(fixture("pages/text-only.jpg")).unwrap();
        for (name, bytes) in [
            ("restart interval of a partial row", &partial_rows[..]),
            ("missing restart marker", &missing_marker),
            ("no restart interval", &unmarked),
            ("truncated", &rows[..rows.len() / 2]),
            ("not a JPEG", b"\x89PNG\r\n\x1a\n"),
        ] {
            assert!(RestartRows::parse(bytes).is_none(), "{name}");
        }
    }

    #[test]
    fn files_open_as_the_image_crate_opens_them() {
        for name in [
            "decode/restart-rows.jpg",
            "pages/text-only.jpg",
            "staff-start-clef.png",
        ] {
            let path = fixture(name);
            let expected = image::open(&path).unwrap().into_rgb8();
            assert!(open_rgb(&path).unwrap() == expected, "{name}");
        }
        assert!(open_rgb(&fixture("decode/absent.jpg")).is_err());
    }
}
