# Smude v2

Rectify photographed sheet music with classical staff/barline detection and a
native Rust geometry pipeline. The default ONNX page detector supplies page
boundaries and aspect ratio; its interior dewarping output is not used. There is
no U-Net and no Python dependency in the normal runtime.

Development continues on branch `v2`. This is an alpha: the independent geometry
suite passes, but arbitrary paper deformation and page detection are not solved
universally. The recognition and OMR projects remain separate.

## Desktop app

```sh
cargo run -p smude-gui
# Optimized desktop executable:
cargo build --release -p smude-gui
# Optional macOS .app bundle (requires cargo-packager):
(cd gui && cargo packager --release --formats app)
```

The separate `gui/` crate uses egui/eframe and the same in-process processing
engine as the CLI. Add JPEG/PNG files or recursively import folders using the
native picker or drag and drop. Processing is automatic, including page detection;
the preview and PNG export always use the paper-white grayscale result. The file
queue can be paused after the current image. Compare original and result, zoom,
pan, and choose the output page for double-page photos. Export one page or every
completed page; existing files are preserved with numbered filenames.
The original preview receives live page, staff, barline, system-edge and text
overlays as their detector stages finish. A scan sweep and progressive line
reveal visualize this progress without delaying processing. The Erkennungen
button hides or shows overlays; exported images remain clean.
Zooming and panning follow corresponding music positions in both previews,
using the actual forward/inverse geometry including the final crop. Local
geometric scale is accounted for instead of equating source and output pixels.

The models and ONNX Runtime are embedded, so no CLI sidecar, runtime installation
or external model files are needed by the GUI. The CLI remains independently
buildable with its full processing and output options.

The desktop UI ports Wing Companion's typography, icon set, widgets, palette,
card gradients and background effects. Font and icon licences are included in
`gui/assets/fonts/` and available in the app's Info dialog.

## Run

```sh
cargo build --release
target/release/smude-v2 --image page.jpg --out outputs/page
```

Page detection runs by default and creates separate outputs for a book spread:

```sh
target/release/smude-v2 --image spread.jpg --out outputs/spread
```

The release executable includes both page-detection models and statically linked
ONNX Runtime. It needs no companion model files or ONNX Runtime installation.
Normal operating-system libraries remain required.
Left/right is the default spread
layout, independent of image shape. Use `--spread-layout stacked` for pages above
one another. Failed or cropped neighbouring pages are reported separately.

## Page model assets

The default models are embedded in the executable and loaded directly from memory.
`--page-models DIRECTORY` explicitly overrides them with external
`page-boundary.onnx` and `page-layout.onnx` files.

For a source build with `onnx-pages`, the two model files must be available under
`models/page-boundary/` at compile time. They are supplied separately and remain
Git-ignored. Cargo downloads the matching ONNX Runtime archive on the first
build and links it statically. Neither the source model directory nor network
access is needed when running the completed executable.

`--no-pages` skips page detection for an already cropped image. Supplying
`--geometry` also skips detection and reuses the supplied measurements. Builds
without the `onnx-pages` feature use direct staff detection by default.

`--page-layout single|double` overrides page-count classification.
`--page-regions regions.json` accepts manual source-image polygons instead of
running inference. Each page gets its own observations and warp; a failed sibling
does not discard successful pages. `--detect-only` exports observations without
rectification; `--geometry observations.json` reuses source-pixel measurements.

## Geometry

The default pipeline is:

1. Detect pages and prepare a bounded-resolution four-point analysis view when
   page mode supplies reliable corners and aspect ratio.
2. Detect five-line staves, complete bar strokes and staff starts/ends. Restore
   measurements to original-image coordinates.
3. Fit calibrated GCS or an evidence-supported page plane, using five-line
   equal spacing and the supplied paper aspect ratio.
4. Condition independent measurement noise, complete supported staff ends and
   flatten staff curves to horizontal, equally spaced lines.
5. Apply bounded, evidence-gated x-only correction to barlines/system starts,
   followed by local text correction outside protected music regions.
6. Sample the original RGB once, normalize grayscale contrast, remove confirmed
   paper rims and crop to the page with natural margins.

Detection uses at most 1800 pixels in width by default.
`--analysis-long-edge 1800` caps either orientation instead. Final reconstruction
uses the original resolution.

The recommended output is `rectified-clean.png`, a high-contrast grayscale page
on white paper. RGB, raw grayscale and strict Sauvola binary outputs are also
written, along with observations, mapped overlays, fit diagnostics and individual
stage profiles. Page mode writes one `page-*/` directory per accepted page.

Use `--no-gcs`, `--no-vertical-correction` or `--no-text-dewarp` for ablations.
`--no-perspective` selects staff flattening without the broad perspective/GCS
stage. The earlier unconditional `--system-shear` remains opt-in. Explicit
`--same-size '0,1;2,3'` groups require `--no-pages` or supplied geometry; never group an
ossia with regular-sized staves merely because it is nearby.

## Validation

```sh
cargo fmt --check
cargo test --workspace --all-targets
cargo test --no-default-features
cargo clippy --workspace --all-targets -- -D warnings
```

The development batch produced outputs for 146 of 154 photos; the eight known
page/staff-detection failures remain. Both independent six-case geometry tracks
passed their fixed gates. These are development checks, not universal accuracy
claims. Model weights, datasets and generated images are not committed.

## Repository layout

| Path | Purpose |
| --- | --- |
| `Cargo.toml`, `Cargo.lock` | Rust crate and locked dependencies |
| `src/` | Native geometry, detection, rendering and CLI |
| `tests/` | Rust regressions and required fixture |
| `gui/` | egui/eframe desktop app, preview and export |

Git ignores local assets, build products and generated results. Historical
Python code, research prototypes, synthetic generation and old development
files are archived outside this checkout. Local model weights and build products
are not committed. Gallery outputs and development tools live outside this
checkout.
The repository's [license](LICENSE.md) is unchanged.
