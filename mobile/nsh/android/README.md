# Android host for the Rust UI

All application screens and the yellow page-contour overlay are Rust NSH. This
small Java host supplies CameraX and runtime permission/lifecycle handling. It
does not embed the previous WebView application.

## Build

Requires NSH **0.4.0**, Rust from `../rust-toolchain.toml`, the Android SDK with API
36/build tools and NDK **29.0.14206865**, and a JDK compatible with Gradle 9.6.1/AGP 9.3.1.
Set `ANDROID_HOME` (and optionally `ANDROID_NDK_HOME`) to the installed SDK/NDK.

```sh
./build.sh
```

The result is `app/build/outputs/apk/debug/app-debug.apk`, package
`com.gridnotelabs.smude.design`, separate from the existing mobile app. The build
first generates Smude's `libnsh.so`, then combines it with NSH's Java classes and
CameraX. **Do not install the intermediate NSH CLI APK:** the stock Activity does
not register this camera component. No stock `libnsh.so` is packaged alongside
the application library. The NDK's `libc++_shared.so` is also packaged, because
the linked ONNX Runtime needs it and Android does not provide it globally.

Only arm64-v8a is currently packaged. The app supports Android 9/API 28 and later,
targets API 35, and compiles against API 36. Existing NSH assets are copied into
the expected `nsh/assets/` tree. Generated outputs and SDK paths stay out of Git.

```sh
./gradlew :app:compileDebugJavaWithJavac :app:lintDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.gridnotelabs.smude.design/.MainActivity
```

## Camera contract

- `smude.camera` is registered before `NshView.runRustApp("nsh")`.
- CameraX 1.5.3 uses one shared ViewPort for preview, analysis and capture.
  Analysis crops to that ViewPort and physically applies sensor rotation before
  inference. The Rust canvas receives normalized coordinates in this oriented
  frame. Two page contours remain separate.
- Preview uses `TextureView` via `ImplementationMode.COMPATIBLE`, so it follows
  native clipping/navigation transitions. All chrome and drawing remain Rust.
- Analysis runs serially, keeps only the latest available frame, and caps its
  long edge at 640 px. Frames remain in native memory; JNI returns contour JSON.
- Two consistent valid detections enable capture. Valid results older than 1200 ms,
  lifecycle changes and rotation clear readiness. A previously stable page holds
  its contour and shutter through an invalid-frame gap of at most 600 ms, only
  while the 8×8 luminance fingerprint remains spatially consistent. Uniform
  exposure changes are ignored; strong motion clears immediately. The native capture
  command checks readiness again; Rust button state alone is not trusted.
- Captures explicitly request the highest available resolution, including slower
  high-resolution CameraX sizes, and prioritize quality. In-memory JPEG output
  preserves the full sensor raster instead of applying the preview ViewPort crop.
  EXIF records physical device orientation, independently of rotation lock.
  Private storage uses a temporary file renamed after successful writing.
  `captured` reports its absolute path and the detected page count.
- Physical orientation is observed independently of display rotation. The model
  sees upright pixels even under rotation lock; contours are transformed back
  to the unchanged preview. Unknown/flat poses retain the last known orientation.
- `active=false`, detach, lifecycle stop and component destruction unbind this
  component's own use cases. Analysis callbacks from earlier generations are
  discarded. Capture callbacks after destruction remove their incomplete file.

## Navigation

The host adds no global back callback and no gesture-exclusion rectangle. NSH's
native stack owns in-app back, including progress/cancel callbacks on API 34+.
The manifest enables `OnBackInvokedCallback`; root back remains system-owned.
Android back can start at either side, unlike iOS's leading-edge convention.

## Validation status

Verified on 2026-11-08:

- Rust arm64 release library, Debug APK and R8-minified release APK build.
  The release APK is unsigned; store signing has not been configured.
- Java compilation and Android Lint pass (zero errors; warnings concern target/
  dependency freshness, icons and release polish).
- Installed and launched on the **arm64 API 35 emulator**. Native Rust project
  list, page grid, viewer, camera and export screens render, including bundled
  image assets.
- An interior viewer swipe advances page 1 to page 2. A left-edge gesture
  visibly reveals the underlying project; returning the finger to the edge
  cancels and preserves page 2. Committing the edge gesture pops exactly once.
- Right-edge back exits the camera to the project. Root back leaves the task
  under OS control. Camera release is confirmed by `dumpsys media.camera`.
- Runtime camera permission, actual CameraX virtual-scene preview and disabled
  no-page shutter work. This is an emulator camera, **not a physical-camera
  alignment or successful document-capture test**.
- A subsequent 180-degree display-rotation listener addition was Java-compiled;
  its device rotation path still needs a physical test.

Screenshots from that run are in `/tmp/smude-nsh-android-visual/` (outside Git),
including an in-progress native back transition. No physical Android device was
connected. Actual page contours, their alignment through rotation, successful
single/double capture, permission denial/recovery and device-specific lifecycle
behavior remain physical-device acceptance checks. API 28–34 and 36+ navigation
behavior has not been runtime-tested in this pass.

Primary references: [NSH Android](https://nsh.dev/docs/android/),
[CameraX ViewPort](https://developer.android.com/media/camera/camerax/configuration#crop),
[RGBA API contract](https://developer.android.com/reference/androidx/camera/core/ImageAnalysis#OUTPUT_IMAGE_FORMAT_RGBA_8888()),
[Predictive back](https://developer.android.com/guide/navigation/custom-back/predictive-back-gesture).

## Native import and share services

`smude.services` is registered alongside the camera. It announces the durable
`filesDir/SmudeProjects` directory to Rust. `import` opens Android's native photo
picker (ordered multiple selection, gold accent); `import_files` opens the native
document picker. Both copy provider content while its temporary grant is valid.
Upright JPEGs retain their exact compressed bytes. Other supported images,
including HEIC and EXIF rotations/reflections, are decoded at full resolution and
written as canonical PNG; transparency is composited onto white. Imports are
serialized and reported only when the complete selected batch is durable. They
live under `staging/<uuid>` until the Rust repository adopts them.

`share` accepts only completed non-empty files under `SmudeProjects/exports`.
A restricted FileProvider supplies content URIs, ClipData and temporary read
grants to Android's chooser. Multiple JPEGs use ACTION_SEND_MULTIPLE; PDF/single
JPEG use ACTION_SEND. The callback reports `opened: true`, never delivery or a
successful message send. App UI remains Rust; only OS pickers/chooser are native.

Verification in this implementation pass:

- Three pure Java hysteresis tests pass (brief misses, strict deadline, motion,
  hard failures, exposure compensation).
- Five API-35-emulator instrumentation tests pass: native photo/files picker
  open and cancel; EXIF rotation at full resolution; upright JPEG direct-byte
  eligibility; alpha-to-white normalization; FileProvider byte identity,
  multi-file intents and rejection of non-export/private/empty paths.
- Java compilation, Android Lint and native APK packaging are checked separately
  from the parent Rust UI rebuild. Actual HEIC hardware decoding and successful
  real-document acquisition still require the physical-device pass.

The integrated Rust UI was subsequently rebuilt and exercised on the API 35
emulator with `input_fullsize.jpg`, selected through the real system photo
picker. The full flow passed: empty persistent library, new project with focused
name input and visible IME, IME-submit save, import, GCS processing, page viewer,
switch to Mesh, PDF export and the native share chooser. The GCS output was
3085×4207; the recomputed Mesh output was 3028×4134. Both were actual PNG files,
not bundled previews. A force-stop/relaunch preserved the project and result.

The full-resolution grayscale PDF was 1,314,686 bytes; the small black-and-white
PDF was 29,085 bytes. Poppler verified one valid PDF page and a 1172×1600,
one-bit image in the small export. The grayscale PDF was rendered and visually
checked. The chooser was dismissed without choosing a recipient; this tests
export and handoff, not delivery. The physical camera limitations above remain.


Launcher resources use a native adaptive icon for Android 8+ (all supported devices), with an explicit dark background and the shared full-resolution app artwork inset to fit circular launcher masks. Both icon and roundIcon select the adaptive resource. Gradle copies the unchanged source asset into generated launcher resources; no separate image export is needed when updating assets/app-icon.png. The emulator-only LauncherIconInstrumentedTest loads the installed application icon and checks its adaptive type, dark background, circular gold-artwork bounds and rendered rim.
