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
an analytic bound between probes. Bicubic sampling, grayscale and paper-white
normalization run together on the GPU. Border cleanup also uses exact GPU
Chebyshev distance fields; Android retains the faster parallel CPU distance
transform after device profiling. Border content decisions remain unchanged. Small pixel
differences from CPU floating-point arithmetic are expected.

Each page's `profile.json` records the actual adapter, backend, fallback reason,
map construction, transfers, GPU timestamps when available and border-distance
calls. GPU `render_seconds` includes grayscale and paper normalization; their
separate CPU stage counters are zero in this path. Compare the whole output
stage as well as total processing time. Automatic mode falls back to CPU when
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
   measurements to original-image coordinates. Staff tracking uses wider
   tangent-aligned ink probes and robust regression over staff-scaled windows
   to suppress short symbol-induced wobbles while retaining broad page curvature.
   Complete connected bar strokes are regularized across staff/gap junctions too.
3. Fit calibrated GCS or an evidence-supported page plane, using five-line
   equal spacing and the supplied paper aspect ratio. Repeated system boundaries
   contribute shared-column evidence to this same camera fit, using observed
   staff ends or continuous boundary bars. Both sides must be supported by at
   least three common systems. Source-image consensus excludes
   intentional first-system indents, short final systems and incompatible edges.
4. Condition independent measurement noise, complete supported staff ends and
   flatten staff curves to horizontal, equally spaced lines.
5. Apply bounded, evidence-gated x-only correction to barlines/system starts,
   followed by local text correction outside protected music regions. Short words
   contribute local slope observations too; confidence and measurement uncertainty
   limit their influence to measured support. Text analysis uses at most 1800 pixels
   on the long edge, independently of final output resolution.
6. Sample the original RGB once, normalize grayscale contrast, remove confirmed
   paper rims and crop to the page with natural margins.

Detection uses at most 1800 pixels in width by default.
`--analysis-long-edge 1800` caps either orientation instead. Final reconstruction
uses the original resolution. Rendering reuses row-dependent map coefficients
without approximating the geometric inverse; staff tracing, bar proposals and
endpoint measurement run in parallel. The app uses up to eight workers while
reserving CPU capacity for its interface.

The default output is `rectified-clean.png`, a high-contrast grayscale page
on white paper, plus observations, mapped geometry, fit diagnostics and individual
stage profiles. Alternative image outputs require `--diagnostics`. Page mode
writes one `page-*/` directory per accepted page.

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
