// Run the exact pure helper from CameraView.swift on the host, without a camera:
// { echo 'import Foundation'; sed -n '/BEGIN CAMERA STABILITY/,/END CAMERA STABILITY/p' native/ios/CameraView.swift;
//   cat native/ios/Tests/CameraStabilityChecks.swift; } > /tmp/smude-camera-stability.swift
// swift /tmp/smude-camera-stability.swift
let shape = [[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]]
let scene = (0..<64).map { Double($0 % 8) / 10 }
func acquired() -> CameraDetectionGate {
    var gate = CameraDetectionGate()
    assert(gate.accept(valid: true, polygons: shape, scene: scene, frameTime: 1, now: 1.08) == .show)
    assert(!gate.ready)
    _ = gate.accept(valid: true, polygons: shape, scene: scene, frameTime: 1.2, now: 1.28)
    assert(gate.ready)
    return gate
}
// A transient miss holds the last stable decision, including classifier count
// changes. Uniform auto-exposure must not look like moving the phone.
var flicker = acquired()
assert(flicker.accept(valid: false, polygons: [shape[0],shape[0]], scene: scene.map{$0+0.1}, frameTime: 1.35, now: 1.4) == .hold)
assert(flicker.canCapture(now: 1.7))
assert(!flicker.canCapture(now: 1.801))
assert(flicker.accept(valid: false, polygons: [], scene: scene, frameTime: 1.8, now: 1.85) == .show)
assert(!flicker.ready) // Repeated misses did not refresh the 600 ms window.
// Independent spatial change clears immediately even when geometry is invalid.
var changed = acquired()
let other = (0..<64).map { $0 < 32 ? 1.0 : 0.0 }
assert(changed.accept(valid: false, polygons: [], scene: other, frameTime: 1.3, now: 1.35) == .show)
assert(!changed.ready)
// Reliable moved geometry reacquires instead of keeping the old shutter state.
var moved = acquired()
let shifted = shape.map { $0.map { [$0[0]+0.1,$0[1]] } }
_ = moved.accept(valid: true, polygons: shifted, scene: scene, frameTime: 1.3, now: 1.35)
assert(!moved.ready)
_ = moved.accept(valid: true, polygons: shifted, scene: scene, frameTime: 1.5, now: 1.55)
assert(moved.ready)
// Old inference results and genuine backend errors never get the hold behavior.
var stale = acquired()
assert(stale.accept(valid: true, polygons: shape, scene: scene, frameTime: 1.3, now: 2.6) == .clear)
assert(!stale.ready)
var failed = acquired()
assert(failed.accept(valid: false, polygons: [], scene: scene, frameTime: 1.3, now: 1.35, hardFailure: true) == .clear)
assert(!failed.ready)
var lifecycle = acquired(); lifecycle.reset(); assert(!lifecycle.canCapture(now: 1.3))
var missingScene = acquired()
assert(missingScene.accept(valid: false, polygons: [], scene: [], frameTime: 1.3, now: 1.35) == .show)
assert(!missingScene.ready)
let normal = acquired(); assert(normal.canCapture(now: 2.3)); assert(normal.expired(now: 2.401))
print("Camera stability: exposure, classifier flicker, bounded hold, motion, error, lifecycle and source age passed")
