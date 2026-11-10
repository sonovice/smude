#!/bin/sh
# Run from any directory. The NSH CLI has no custom Info.plist hook in 0.4.0.
set -eu
cd "$(dirname "$0")/../.."
target="${1:-ios-simulator}"
nsh_bin="${NSH_BIN:-$HOME/.local/bin/nsh}"
if [ ! -x "$nsh_bin" ]; then nsh_bin="$(command -v nsh)"; fi
"$nsh_bin" package --target "$target"
app=".nsh/out/Smude UI.app"
/usr/libexec/PlistBuddy -c 'Set :MinimumOSVersion 15.1' "$app/Info.plist"
# The standalone packager omits Xcode archive platform metadata. Without
# CFBundleSupportedPlatforms, exportArchive misidentifies this as a Mac app.
python3 - "$app/Info.plist" "$target" <<'PYMETA'
import pathlib, plistlib, subprocess, sys
path = pathlib.Path(sys.argv[1])
values = plistlib.loads(path.read_bytes())
values["CFBundleDevelopmentRegion"] = "en"
values["CFBundleLocalizations"] = ["en", "de", "pt"]
values["UISupportedInterfaceOrientations"] = ["UIInterfaceOrientationPortrait"]
simulator = sys.argv[2] == 'ios-simulator'
sdk = 'iphonesimulator' if simulator else 'iphoneos'
def sdk_value(flag):
    return subprocess.check_output(['xcrun', '--sdk', sdk, flag], text=True).strip()
version = sdk_value('--show-sdk-version')
build = sdk_value('--show-sdk-build-version')
xcode = subprocess.check_output(['xcodebuild', '-version'], text=True).splitlines()
parts = xcode[0].split()[1].split('.')
major, minor, patch = (list(map(int, parts)) + [0, 0])[:3]
values.update(
    CFBundleDisplayName='Smude',
    CFBundleSupportedPlatforms=['iPhoneSimulator' if simulator else 'iPhoneOS'],
    LSRequiresIPhoneOS=True,
    UIRequiredDeviceCapabilities=['arm64'],
    DTPlatformName=sdk, DTPlatformVersion=version, DTPlatformBuild=build,
    DTSDKName=sdk + version, DTSDKBuild=build,
    DTXcode=f'{major:02}{minor}{patch}', DTXcodeBuild=xcode[1].split()[-1],
    DTCompiler='com.apple.compilers.llvm.clang.1_0',
)
values['UISupportedInterfaceOrientations~ipad'] = [
    'UIInterfaceOrientationPortrait', 'UIInterfaceOrientationPortraitUpsideDown',
    'UIInterfaceOrientationLandscapeLeft', 'UIInterfaceOrientationLandscapeRight',
]
path.write_bytes(plistlib.dumps(values))
PYMETA
/usr/libexec/PlistBuddy -c 'Delete :NSCameraUsageDescription' "$app/Info.plist" 2>/dev/null || true
/usr/libexec/PlistBuddy -c 'Add :NSCameraUsageDescription string Take photos of sheet music to add to your collection.' "$app/Info.plist"
cp native/ios/PrivacyInfo.xcprivacy "$app/PrivacyInfo.xcprivacy"
if [ "$target" = ios-simulator ]; then
  codesign --force --sign - "$app"
else
  echo "Unsigned device app: $app. Sign with your development profile before installing."
fi

for language in en de pt; do
  cp -R "native/ios/$language.lproj" "$app/"
done
