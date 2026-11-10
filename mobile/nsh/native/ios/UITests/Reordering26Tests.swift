import XCTest

/// Simulator-only: Reorder26 is a 30-page collection of copied test fixtures.
final class Reordering26Tests: XCTestCase {
    let app = XCUIApplication(bundleIdentifier: "com.gridnotelabs.smude.design")
    func item(_ id: String) -> XCUIElement {
        app.descendants(matching: .any).matching(identifier: id).firstMatch
    }
    func testLongPressCanCarryAPageThroughAutomaticEdgeScrolling() {
        continueAfterFailure = false
        app.launchArguments = ["-AppleLanguages", "(de)", "-AppleLocale", "de_DE"]
        app.launch()
        XCTAssertTrue(item("project-26000").waitForExistence(timeout: 15))
        item("project-26000").tap()
        let page = item("page-26001")
        XCTAssertTrue(page.waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertEqual(page.frame.minY, item("page-26003").frame.minY, accuracy: 1, "All three columns must fit the actual native viewport")
        let before = page.label.components(separatedBy: " ✓")[0]
        let start = page.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        let edge = app.coordinate(withNormalizedOffset: CGVector(dx: 0.16, dy: 0.85))
        start.press(forDuration: 0.7, thenDragTo: edge, withVelocity: .slow, thenHoldForDuration: 1.7)
        XCTAssertTrue(item("selection-done").waitForExistence(timeout: 5), app.debugDescription)
        XCTAssertNotEqual(page.label.components(separatedBy: " ✓")[0], before, "Edge scrolling must move the held page into a later row")
        let shot = XCTAttachment(screenshot: app.screenshot())
        shot.name = "Page dropped after native edge scrolling"
        shot.lifetime = .keepAlways
        add(shot)
        let dropped = page.label.components(separatedBy: " ✓")[0]
        item("selection-done").tap()
        app.navigationBars.buttons["BackButton"].tap()
        item("project-26000").tap()
        XCTAssertEqual(item("page-26001").label.components(separatedBy: " ✓")[0], dropped, "Order survives reopening")
    }
}
