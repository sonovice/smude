// Run with the CAMERA ORIENTATION block from CameraView.swift (Foundation only).
var pose = CameraOrientation()
assert(pose.captureInterface(4) == 4) // Startup before any gravity sample.
pose.observe(3) // UIDevice.landscapeLeft, with the interface locked to portrait.
assert(pose.captureInterface(1) == 3, "Capture must follow the device despite portrait lock")
assert(pose.analysisTurns(1) == 3)
assert(pose.analysisTurns(3) == 0) // UIInterfaceOrientation.landscapeRight has raw value 3.
pose.observe(4)
assert(pose.captureInterface(1) == 4)
assert(pose.analysisTurns(1) == 1)
pose.observe(5); pose.observe(0); pose.observe(6) // Face-up/unknown/down preserve last pose.
assert(pose.captureInterface(1) == 4)
pose.observe(2)
assert(pose.analysisTurns(1) == 2)
pose.observe(1)
assert(pose.analysisTurns(1) == 0)
print("Camera orientation: locked/unlocked, both landscapes, inverted and flat poses passed")

for interface in 1...4 {
    for turns: UInt32 in 0...3 {
        let plan = CameraOrientation.capturePlan(interface, turns: turns)
        assert(plan.angle == CameraOrientation.angle(interface), "Photo must match preview, without consuming the model turn")
        assert(plan.rotation == Int(turns) * 90, "Accepted preview rotation must be persisted")
    }
}
