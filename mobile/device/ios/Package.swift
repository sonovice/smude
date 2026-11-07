// swift-tools-version:5.7
import PackageDescription

let package = Package(
    name: "tauri-plugin-smude-device",
    platforms: [.iOS(.v16)],
    products: [.library(name: "tauri-plugin-smude-device", type: .static, targets: ["SmudeDevice"])],
    dependencies: [.package(name: "Tauri", path: "../.tauri/tauri-api")],
    targets: [.target(name: "SmudeDevice", dependencies: [.byName(name: "Tauri")], path: "Sources")]
)
