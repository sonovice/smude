use std::{env, path::PathBuf, process::Command};
fn main() {
    println!("cargo:rerun-if-changed=native/ios/CameraView.swift");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("ios") {
        return;
    }
    let target = env::var("TARGET").unwrap();
    let sim = target.ends_with("-sim");
    let sdk = if sim { "iphonesimulator" } else { "iphoneos" };
    let sdkpath = String::from_utf8(
        Command::new("xcrun")
            .args(["--sdk", sdk, "--show-sdk-path"])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap();
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let triple = if sim {
        "arm64-apple-ios15.1-simulator"
    } else {
        "arm64-apple-ios15.1"
    };
    let mut objects = vec![];
    for name in ["CameraView", "ServicesView", "ControlsView"] {
        let source = format!("native/ios/{name}.swift");
        println!("cargo:rerun-if-changed={source}");
        let object = out.join(format!("{name}.o"));
        let status = Command::new("xcrun")
            .args([
                "swiftc",
                "-parse-as-library",
                "-emit-object",
                "-O",
                "-module-name",
                &format!("Smude{name}"),
                "-target",
                triple,
                "-sdk",
                sdkpath.trim(),
                &source,
                "-o",
            ])
            .arg(&object)
            .status()
            .unwrap();
        assert!(status.success(), "Swift platform adapter compile failed");
        objects.push(object);
    }
    let status = Command::new("xcrun")
        .args(["ar", "crs"])
        .arg(out.join("libSmudeCamera.a"))
        .args(objects)
        .status()
        .unwrap();
    assert!(status.success());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=SmudeCamera");
    for framework in [
        "AVFoundation",
        "CoreMedia",
        "CoreVideo",
        "CoreImage",
        "UIKit",
        "Foundation",
        "PhotosUI",
        "Photos",
        "ImageIO",
        "UniformTypeIdentifiers",
    ] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
    let dev = String::from_utf8(
        Command::new("xcode-select")
            .arg("-p")
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap();
    println!(
        "cargo:rustc-link-search=native={}/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/{}",
        dev.trim(),
        sdk
    );
    println!("cargo:rustc-link-search=native=/usr/lib/swift");
    for lib in [
        "swiftCore",
        "swiftFoundation",
        "swiftDispatch",
        "swiftCoreFoundation",
        "swiftCoreGraphics",
        "swiftDarwin",
        "swiftObjectiveC",
    ] {
        println!("cargo:rustc-link-lib={lib}");
    }
}
