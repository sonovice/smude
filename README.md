# Smude v2

Rectify photographed sheet music with classical staff/barline detection and a
native Rust geometry pipeline. The default ONNX page detector supplies page
boundaries and aspect ratio; its interior dewarping output is not used. There is
no U-Net and no Python dependency in the normal runtime.

Development continues on branch `v2`. This is an alpha: the independent geometry
suite passes, but arbitrary paper deformation and page detection are not solved
universally. The recognition and OMR projects remain separate.

## Run

```sh
cargo build --release
target/release/smude-v2 --image page.jpg --out outputs/page
```

Page detection runs by default and creates separate outputs for a book spread:

```sh
target/release/smude-v2 --image spread.jpg --out outputs/spread
```

Default page detection needs locally prepared model weights. Cargo supplies and statically
links ONNX Runtime automatically; the first build needs network access.
Left/right is the default spread
layout, independent of image shape. Use `--spread-layout stacked` for pages above
one another. Failed or cropped neighbouring pages are reported separately.

## Page model assets

Page detection requires `page-boundary.onnx` and `page-layout.onnx` under
`models/page-boundary/`, or a directory supplied with `--page-models DIRECTORY`.
The contour head must return 96 ordered x/y perimeter points followed by the
height/width ratio (193 values). Model weights are supplied separately. A native
ONNX Runtime library does not belong in the model directory.

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
cargo test --all-targets
cargo test --no-default-features
cargo clippy --all-targets -- -D warnings
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

Git ignores local assets, build products and generated results. Historical
Python code, research prototypes, synthetic generation and old development
files are archived outside this checkout. Local model weights and build products
are not committed. Gallery outputs and development tools live outside this
checkout.
The repository's [license](LICENSE.md) is unchanged.
