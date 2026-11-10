import XCTest
import UIKit

/// Exercise the actual platform adapter, including its material content layout.
@MainActor final class ProgressRenderingTests: XCTestCase {
    private func render(_ value: Double?, size: Int = 48) -> [UInt8] {
        let raw = smudeControlCreate(nil, { _, _ in })
        let view = Unmanaged<UIView>.fromOpaque(raw).takeRetainedValue()
        let config = "{\"kind\":\"progress\",\"value\":\(value ?? 0),\"indeterminate\":\(value == nil)}"
        config.withCString { smudeControlConfig(raw, $0) }
        view.frame = CGRect(x: 0, y: 0, width: size, height: size)
        view.setNeedsLayout(); view.layoutIfNeeded()
        let material = view.subviews.first as! UIVisualEffectView
        let pie = material.contentView.subviews.first!
        XCTAssertEqual(pie.frame, material.contentView.bounds, "Fill must reach the circle edge")
        pie.layoutIfNeeded()
        XCTAssertTrue(pie.layer.animationKeys()?.isEmpty ?? true, "No waiting spinner")
        var pixels = [UInt8](repeating: 0, count: size * size * 4)
        pixels.withUnsafeMutableBytes { buffer in
            let context = CGContext(data: buffer.baseAddress, width: size, height: size,
                bitsPerComponent: 8, bytesPerRow: size * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            pie.layer.render(in: context)
        }
        return pixels
    }
    func testQueuedPageHasNoFillOrSpinningAnimation() {
        XCTAssertTrue(render(nil).enumerated().filter { $0.offset % 4 == 3 }.allSatisfy { $0.element == 0 })
    }
    func testFullCompletionFillsToAllFourCircleEdges() {
        let pixels = render(100)
        for (x, y) in [(1, 24), (46, 24), (24, 1), (24, 46), (24, 24)] {
            let i = (y * 48 + x) * 4
            XCTAssertGreaterThan(pixels[i + 3], 230, "Unfilled circle edge at \(x),\(y)")
            XCTAssertGreaterThan(pixels[i], pixels[i + 2], "Completion must be gold")
        }
    }
}
