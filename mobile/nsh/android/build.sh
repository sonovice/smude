#!/usr/bin/env bash
set -euo pipefail
android_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ui_dir="$(dirname -- "$android_dir")"
nsh_bin="${NSH_BIN:-$HOME/.local/bin/nsh}"
if [[ ! -x "$nsh_bin" ]]; then nsh_bin="$(command -v nsh)"; fi
if [[ "$($nsh_bin --version)" != nsh\ 0.4.0* ]]; then
  echo "This host requires NSH 0.4.0 (Java runtime and Rust crates must match)." >&2
  exit 1
fi
cd -- "$ui_dir"
# This also generates the app-specific JNI library. Its stock APK is only an
# intermediate; Gradle supplies the actual CameraX Activity and permissions.
"$nsh_bin" package --target android --abi arm64-v8a --out .nsh/android-host-bootstrap
cd -- "$android_dir"
./gradlew :app:assembleDebug --console=plain
