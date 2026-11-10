# Smude v2

Rectify photographed sheet music with classical staff/barline detection and a
native Rust geometry pipeline. The default ONNX page model supplies boundaries,
aspect ratio and the complete interior page grid. It rectifies each page before
optional staff, barline and text refinement. Pages without music still produce
an output. If no complete page can be assigned, sufficiently supported score excerpts are processed from staff/bar geometry without an invented page boundary or physical aspect ratio. There is no U-Net or Python dependency in the normal runtime.

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
the preview and PNG export use paper-white notation; pages with substantial color and no detected staves retain their original RGB colors. The file
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

## Experimental stem observations

`detection::stems::detect_stems(binary, observations)` is an independent,
opt-in detector for head-supported stem shafts. It needs a black-on-white binary
image and staff/bar observations in the same pixel coordinates. It has no
external OMR dependency and does not modify either input.

The reduced geometric core retains candidate-owned line probes, narrow-row/edge
measurements and staff-scaled metrics. Head matching uses the original discrete
head templates with a smaller, standalone score: known staff-rule pixels count
as neither positive nor negative evidence. No staff-line-removed image is built.
Template support must remain sufficient outside those bands; exterior-paper
checks reject matches inside larger ink shapes.
Heads must attach to the end of a measured shaft or its chord cluster. A
second shaft fit rejects sparse/bent leftovers after rule/head exclusion.
Overlapping direction trials merge only if they also share a head; unrelated
clef fragments cannot donate confidence to each other.

Direction trials preserve measured shaft slope. A stem can report multiple
`heads`, including chord heads. Its `points` exclude rule/head intersections and
may have gaps. `geometry_usable` is a separate support gate; recognizing a short
candidate does not automatically authorize it to influence dewarping. Scores
and `known_support` are evidence measures, not calibrated probabilities.

This is a variant of the original geometric primitives, not a port of the full
semantic classifier. Short/grace stems can still be missed; text, rests, clefs
and severe deformation can produce false candidates. It searches for attached
heads around shaft candidates, not every stemless head on the page. There is no
pitch, rhythm or voice reconstruction. The default mesh uses the conservative
geometry subset as direction evidence; diagnostics include the stem candidates.

For controlled experiments, `StemDetection::vertical_evidence(observations)`
selects shafts with adequate head/shaft support and nearby direction agreement.
Map these points to the fit's source coordinates and pass them through
`geometry::GcsFitEvidence::vertical_paths` to
`Geometry::fit_profiled_gcs_with_evidence`. These measurements stay separate
from bars and system memberships. Their total influence is capped, independent
of note density. The regular pipeline passes no additional paths.

On the current photographed test page, adding 106 shafts while withholding 54
others produced no meaningful change in the withheld shaft angles or the final
image. A stronger-weight sensitivity run also showed no benefit. This is why
stem fitting remains opt-in; it adds detection cost without a demonstrated
output improvement on that page. These observations are not manually annotated
ground truth, and the result does not establish behavior on other pages.

## GCS-free direction mesh (default)

`smude-v2 --image score.jpg --out result` now uses the direction mesh by default.
Use `--gcs` to select the GCS comparison; `--direction-mesh` remains an explicit
alias for the default. Page detection, page ratio, output
preparation and the final single source-image sampling remain shared.

This path starts with the learned interior page map, uses all five staff
curves and their relative spacing for residual perspective and horizontal
flattening, then estimates a common 10 x 16 cubic B-spline x field from full bars,
aligned staff boundaries, independently measured five-line edge directions,
and head-supported shafts. There is no GCS fit and no
additional per-system vertical correction. Optional system shear remains an
explicit option. The mesh itself fixes y exactly.

Relative spacing along each staff also provides a weak local scale prior.
Different staff sizes remain independent unless explicitly grouped. This prior
is an approximate isotropy assumption, not a unique recovery of 3-D surface or
horizontal metric. Mean width is anchored against collapse; control differences
bound local x scale and shear globally. The model uses spatial/family evidence
budgets and curvature regularization rather than moving individual symbols.

On detected pages, consistent bar directions additionally provide a common
camera-space vertical ruling. Together with staff tangents and spacing, this
supplies a bounded relative-width prior near page bends. The fit preserves the
existing direction field, checks metric improvement and bar directions, and
keeps y fixed. Focal length is approximate; this does not uniquely reconstruct
arbitrary 3-D paper. Cropped-score excerpts do not use this prior because their
camera centre is unknown. `surface-metric.json` records support and acceptance.

Reports include `direction_mesh`, `direction_mesh_seconds` and
`stem_detection_seconds`; `gcs_seconds` stays zero. Stem analysis has bounded
resolution and restores original pixel-centre coordinates. `--diagnostics`
additionally writes `stems.json`.

When fewer than two bars per staff are available, one bounded second detection
pass runs on the provisional rectification (long edge at most 1800 pixels).
It reuses the measured staff identities, maps new bar paths back to source
coordinates, and deduplicates overlapping strokes. Only the x field is refit;
the page frame and staff y warp stay fixed. The candidate is accepted only if
aggregate normalized bar deviation improves and original bars stay within the
regression tolerance. `bar-refinement.json` records acceptance and measurements;
`bar_refinement_seconds` records the extra work. Final pixels are sampled from
the original image once, not from the detection preview. These measurements
are fit diagnostics, not independent quality scores.

New mobile pages and sessions default to Mesh; explicitly saved GCS choices are
preserved. The mobile test app also exposes a GCS/Mesh segmented control. Switching after
capture reuses the immutable imported photo. Each model is calculated once per
photo and its full-resolution result remains available for immediate switching
and export. A new photo invalidates both cached outputs; switching during a job
queues the selected model without relabeling an older result. The last chosen
model is remembered for the next photo.

## GPU image generation

CLI, desktop and mobile share the same GPU renderer, enabled by the default
`gpu-render` feature. It uses Metal on Apple devices, Vulkan on Android/Linux
and DirectX 12 on Windows. Automatic selection uses the GPU for final rasters
of at least one megapixel; analysis previews retain the CPU reference sampler.
GPU initialization overlaps detection. Page-model inference remains on the CPU.

```sh
# Automatic GPU selection and CPU fallback (default):
target/release/smude-v2 --image page.jpg --out outputs/page
# Force a backend for a reproducible comparison:
target/release/smude-v2 --image page.jpg --out outputs/page --render-backend cpu
# GPU errors are explicit when forced:
target/release/smude-v2 --image page.jpg --out outputs/page --render-backend gpu
# Build without GPU support:
cargo build --release --no-default-features --features cli,onnx-pages
```

The renderer evaluates the existing inverse geometry on an adaptive map, refines
curved cells and uses exact pixel coordinates at the finest level. Independent
probes enforce a sampled 0.005-source-pixel interpolation tolerance; this is not
an analytic bound between probes. Bicubic sampling, grayscale and diagnostic
binarization run on the GPU. Continuous paper correction runs in shared CPU code
after readback, so both backends use the same background estimator. Border
cleanup uses exact Chebyshev distance fields from a parallel CPU transform on
every platform. Gray output preserves page content and an outward
contour uncertainty band, independently of binary ink classification. Small
sampling differences from CPU floating-point arithmetic are expected.

Each page's `profile.json` records the actual adapter, backend, fallback reason,
map construction, transfers and GPU timestamps when available. GPU `render_seconds` includes grayscale and diagnostic binarization;
`gpu_paper_seconds` is the historical name for that threshold pass. Continuous
paper correction is recorded separately as `print_normalization_seconds` on
both backends. Compare the whole output stage as well as total processing time. Automatic mode falls back to CPU when
GPU initialization, resource limits or rendering fail.

## Android and iOS device test

`mobile/` is a separate Tauri application using the same Rust engine, embedded
models and native ONNX Runtime. No server is required on the phone. JPEG/PNG
and HEIC/HEIF import, native camera capture, automatic page detection,
full-resolution rectification, preview and PNG export are implemented. HEIC uses
ImageIO on iOS and ImageDecoder on Android, applies orientation once and writes
lossless PNG without reducing the source resolution. Native HEIC/camera decoding
accepts up to 64 megapixels and a 12000-pixel longest edge.
Phone EXIF orientation is normalized before detection; ordinary imports retain
the original file bytes. The displayed duration includes import and decoding the
result preview; `mobile-report.json` records the engine duration and per-stage
profiles separately.

On macOS, install Xcode, Node, Android SDK/NDK, Java 21, XcodeGen, CocoaPods and
`xz`. Supply the model files described below for source builds. Then:

```sh
rustup target add aarch64-linux-android aarch64-apple-ios aarch64-apple-ios-sim
cd mobile
npm ci
# Set ANDROID_HOME, NDK_HOME and JAVA_HOME to the installed toolchains.
# Opt-in local test signing and sandbox access; Rust still builds optimized:
ORG_GRADLE_PROJECT_smudeDeviceTest=true npm run tauri android build --apk --target aarch64
# Override the signing team when building with another developer account:
APPLE_DEVELOPMENT_TEAM=76CP95AA8F npm run tauri ios build --export-method debugging
```

Native projects are under `mobile/src-tauri/gen/`. Build outputs, generated
schemas and native libraries are ignored. The build hook prewarms Cargo's mobile
ONNX Runtime cache using checksum-verified archives matching the pinned `ort`
version; it does not put runtime binaries in the repository. The providers in
those archives are included for compatibility; inference currently uses CPU on
both platforms, as on desktop. Device acceleration is benchmarked explicitly,
including ONNX node placement and coordinate parity; a registered provider is
not treated as proof of GPU/NPU execution.

For the existing device test, place `test-input.jpg` in the Android app data
root or the iOS app's Documents directory. The app then exposes **Testbild
verwenden**. Android's private `run-test-once` marker or the iOS launch environment
`SMUDE_AUTORUN=1` runs it once automatically. Source photos are not bundled.
For renderer comparisons, set `SMUDE_RENDER_BACKEND=cpu|gpu|auto` in the iOS
launch environment, or put that value in Android app data file `render-backend`.
Remove the file after testing to restore automatic selection.
Device reports and PNGs are stored below the app's cache `processing/job-*`.
To compare CPU against the platform ONNX provider, add an Android app-data marker
`benchmark-models`, or launch iOS with `SMUDE_BENCH_MODELS=1`, then process a test
image. `processing/model-benchmark/report.json` records four runs, initialization,
node execution providers, coordinate differences and explicit backend errors.
This diagnostic does not change the production backend. On iOS,
`SMUDE_COREML_UNITS=gpu` excludes the Neural Engine (`cpu` selects CoreML CPU;
otherwise all units are allowed). The device check retains CPU as the default:
the CPU/GPU CoreML path meets the fixture coordinate tolerance but is slower,
while the all-units path fails parity and the tested Android NNAPI device rejects
the model. These are device observations, not claims about every phone.
The iOS UI export test
uses `SMUDE_VERIFY_EXPORT=1` to read the actual provider destination and compare
its bytes against the completed PNG before reporting success. Its result and
SHA-256 are written to `processing/last-export.json`.

After a Tauri iOS release build, the included native UI test can reuse the built
Rust library. It requires an unlocked, paired device with UI automation enabled:

```sh
xcodebuild test -project src-tauri/gen/apple/smude-mobile.xcodeproj \
  -scheme smude-mobileUITests -configuration release \
  -destination 'id=YOUR_DEVICE_UDID' -allowProvisioningUpdates \
  SMUDE_UI_TEST_REUSE_RUST=1
```

## Run

```sh
cargo build --release
target/release/smude-v2 --image page.jpg --out outputs/page
```

Page detection runs by default and creates separate outputs for a book spread:

```sh
target/release/smude-v2 --image spread.jpg --out outputs/spread
```

Normal processing writes only `rectified-clean.png` for each page, alongside
measurement and timing reports. Use `--diagnostics` to also save the source and
analysis images plus color, grayscale and binary output variants. The desktop
app uses the normal output path.

The release executable includes both page-detection models and statically linked
ONNX Runtime. It needs no companion model files or ONNX Runtime installation.
Normal operating-system libraries remain required.
Left/right is the default spread
layout, independent of image shape. Use `--spread-layout stacked` for pages above
one another. Failed or cropped neighbouring pages are reported separately.

## Page model assets

The default models are embedded in the executable and loaded directly from memory.
`--page-models DIRECTORY` explicitly overrides them with external
`page-boundary.onnx`, `page-layout.onnx` and `page-geometry.onnx` files.

For a source build with `onnx-pages`, those models and `page-geometry-order.json`
must be available under `models/page-boundary/` at compile time.
The compact boundary head remains available for live camera previews; batch
processing uses all 625 points from the full model. Model weights are supplied separately and remain
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

1. Detect pages and predict their full interior grids and aspect ratios. Prepare
   bounded analysis views through these reversible page maps, independently for
   both sides of a spread.
2. Detect five-line staves, complete bar strokes and staff starts/ends. Restore
   measurements to original-image coordinates. Staff tracking uses wider
   tangent-aligned ink probes and robust regression over staff-scaled windows
   to suppress short symbol-induced wobbles while retaining broad page curvature.
   Complete connected bar strokes are regularized across staff/gap junctions too.
3. Fit residual staff-spacing perspective on the already rectified page for the
   default direction mesh. `--gcs` selects a GCS residual instead.
   The mesh uses a shared x-direction field driven by accepted barlines, staff
   boundaries, coherent five-line edge directions and head-supported shafts;
   insufficient support leaves that field
   at identity rather than inventing vertical directions.
4. Condition independent measurement noise, complete supported staff ends and
   flatten staff curves to horizontal, equally spaced lines.
5. Apply bounded, evidence-gated x-only correction to barlines/system starts,
   followed by local text correction outside protected music regions. Short words
   contribute local slope observations too; confidence and measurement uncertainty
   limit their influence to measured support. Text analysis uses at most 1800 pixels
   on the long edge, independently of final output resolution.
6. Compose the learned map and residuals into one original-image sampling pass.
   Correct paper illumination and print contrast, remove contour-relative rim
   shapes with protected music/text, and retain the physical page crop. Without
   staves, keep the learned page map and independent text/output stages.

Detection uses at most 1800 pixels in width by default.
`--analysis-long-edge 1800` caps either orientation instead. Final reconstruction
uses the original resolution. Rendering reuses row-dependent map coefficients
without approximating the geometric inverse; staff tracing, bar proposals and
endpoint measurement run in parallel. The app uses up to eight workers while
reserving CPU capacity for its interface.

The default output is `rectified-clean.png`. Colorful pages without detected staves preserve RGB; notation uses continuous grayscale with corrected
paper illumination, plus observations, mapped geometry, fit diagnostics and
individual stage profiles. Alternative image outputs require `--diagnostics`. Page mode
writes one `page-*/` directory per accepted page.

Paper correction estimates a background with morphological closing on a max-pooled
analysis grid of at most 1024 pixels per side, then applies a bilinearly interpolated
gain to the full-resolution gray samples. After the final crop, print output also
estimates a local white point from the lower tail of each tile's bright population
and smoothly interpolates these points. A separate dark percentile supplies the
black point when there is enough dark content. This makes most paper texture
exactly white without a Sauvola mask, binary cleanup or a final blur. Intermediate
tones remain grayscale, but highlights overlapping the paper distribution can be
clipped; this is not a guarantee of preserving arbitrarily faint ink. The public
`normalize_paper` function remains the conservative illumination-only transform;
`enhance_print` adds the adaptive display contrast. Severe shadows, noise, dense
ink and inaccurate page boundaries remain limitations.
`rectified-gray.png` diagnostics retain uncorrected sampler tones (with the same
final crop). `--crop-mode content` still explicitly derives crop bounds from the
binary image; the default page crop uses physical page geometry.
The native mobile app continues to export gray by default; its explicit 1-bit
PDF option thresholds only after any requested export resize. New processing
uses a versioned cache folder so it cannot reuse pre-correction render results.
Existing saved page outputs are retained until reprocessed.

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
| `src/detection/` | Page regions/models, music tracking and text observations |
| `src/geometry/` | Observation types, page models, residual corrections and reversible fit |
| `src/rendering/` | CPU/GPU reconstruction, tone preparation, borders and final crop |
| `src/application/` | Shared file workflow, progress, profiles and preview mapping |
| `src/cli/` | Command-line arguments and entry point |
| `tests/` | Rust regressions and required fixture |
| `gui/` | egui/eframe desktop app, preview and export |
| `mobile/` | Tauri mobile app and native device integrations |

The library exposes `geometry`, `detection`, `rendering` and `application` as
separate modules. Existing root-level imports remain supported. Geometry models
and residual corrections are grouped under `geometry/models/` and
`geometry/corrections/`; `geometry/fit/` composes them. Rendering owns both
backends and `rendering/output/` handles tone, distance fields, page borders and
crop bounds. Its processed images never become detector input. `lib.rs` contains
module declarations and public re-exports; shared errors and numerical helpers
live in `error.rs` and the private `math.rs`.

Git ignores local assets, build products and generated results. Historical
Python code, research prototypes, synthetic generation and old development
files are archived outside this checkout. Local model weights and build products
are not committed. Gallery outputs and development tools live outside this
checkout.
The repository's [license](LICENSE.md) is unchanged.


DIN-nahe Seiten werden nach der Papierkorrektur automatisch auf ein exaktes A4-Seitenverhältnis gerahmt. Dafür muss die geschätzte Papierproportion höchstens 4 % von DIN A abweichen. Der vollständige Druckinhalt wird ohne weitere Bildskalierung auf einen weißen Rahmen mit mindestens 10 mm Rand gesetzt; gegenüberliegende Ränder sind symmetrisch. Die Ermittlung nutzt auch schwache Graupixel sowie erkannte Musik- und Textbereiche. Unbekannte Papierformate, reine Ausschnitte ohne Papierproportion, leere Seiten und farbige Cover bleiben unverändert. Mit `--no-a4` lässt sich die automatische Rahmung abschalten; `--no-final-crop` und `--crop-mode content` behalten ihre bisherigen Bedeutungen. `report.json` enthält die zusätzliche Rahmung unter `final_frame`, das Profil deren Zeit unter `final_framing_seconds`.
