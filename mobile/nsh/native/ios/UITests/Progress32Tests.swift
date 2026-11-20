import XCTest
import UIKit

/// Simulator-only. Seed the regular fixture, then set page 27001 to Pending
/// with source=null: the retained preview persists without starting a worker.
final class Progress32Tests: XCTestCase {
    func testPendingProgressKeepsTileGeometry() {
        let app = XCUIApplication(bundleIdentifier: "com.notavium.smude")
        app.launch()
        let collection = app.descendants(matching: .any).matching(identifier: "project-27000").firstMatch
        XCTAssertTrue(collection.waitForExistence(timeout: 10))
        collection.tap()
        let active = app.buttons["page-27001"], ready = app.buttons["page-27002"]
        XCTAssertEqual(active.frame.width, ready.frame.width, accuracy: 0.5)
        XCTAssertEqual(active.frame.height, ready.frame.height, accuracy: 0.5)
        XCTAssertEqual(active.frame.minY, ready.frame.minY, accuracy: 0.5)
        // The page is one accessible button; its noninteractive progress
        // child is intentionally grouped inside it. Inspect the screenshot,
        // rather than querying an inaccessible child by its internal test ID.
        XCTAssertFalse(active.label.contains("…"), app.debugDescription)
        XCTAssertFalse((active.value as? String ?? "").isEmpty, app.debugDescription)
        XCTAssertLessThan(meanBrightness(active.screenshot().image),
                          meanBrightness(ready.screenshot().image) * 0.8,
                          "The unfinished preview must be visibly dimmed")
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.name = "34-material-progress-and-dimmed-preview"
        screenshot.lifetime = .keepAlways
        add(screenshot)
        active.tap()
        XCTAssertTrue(app.segmentedControls["viewer-mode"].waitForExistence(timeout: 5),
                      "The dimming and progress layers must not swallow page taps")
    }
    private func meanBrightness(_ image: UIImage) -> Double {
        var pixels = [UInt8](repeating: 0, count: 32 * 32 * 4)
        pixels.withUnsafeMutableBytes { buffer in
            let context = CGContext(data: buffer.baseAddress, width: 32, height: 32,
                bitsPerComponent: 8, bytesPerRow: 32 * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            context.draw(image.cgImage!, in: CGRect(x: 0, y: 0, width: 32, height: 32))
        }
        return stride(from: 0, to: pixels.count, by: 4).reduce(0.0) {
            $0 + Double(pixels[$1]) + Double(pixels[$1 + 1]) + Double(pixels[$1 + 2])
        } / Double(32 * 32 * 3)
    }

}
