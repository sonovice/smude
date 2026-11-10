import XCTest

final class NavigationTests: XCTestCase {
    let app = XCUIApplication(bundleIdentifier: "com.gridnotelabs.smude.design")
    override func setUp() {
        continueAfterFailure = false
        app.launchArguments += ["-AppleLanguages", "(de)", "-AppleLocale", "de_DE"]
        app.launch()
        XCTAssertTrue(app.staticTexts["Sammlungen"].waitForExistence(timeout: 10))
    }
    func element(_ id: String) -> XCUIElement { app.descendants(matching: .any).matching(identifier: id).firstMatch }
    func capture(_ name: String) {
        let image = XCTAttachment(screenshot: app.screenshot())
        image.name = name; image.lifetime = .keepAlways; add(image)
    }
    func openProject() {
        let project = element("project-1"); XCTAssertTrue(project.waitForExistence(timeout: 5)); project.tap()
        XCTAssertTrue(element("page-1").waitForExistence(timeout: 5))
    }
    func testNativeScreensAndBack() {
        capture("projects")
        openProject(); capture("project")
        element("page-1").tap()
        XCTAssertTrue(app.staticTexts["1 / 4"].waitForExistence(timeout: 5)); capture("viewer")
        app.navigationBars.buttons["BackButton"].tap()
        element("collection-camera").tap()
        XCTAssertTrue(element("shutter").waitForExistence(timeout: 5)); capture("camera")
        XCTAssertFalse(element("shutter").isEnabled)
        app.navigationBars.buttons["BackButton"].tap()
        element("collection-more").tap();app.buttons["Teilen"].firstMatch.tap()
        XCTAssertTrue(element("share-filename").waitForExistence(timeout: 5)); capture("export")
    }
    func testEdgeSwipeCancelThenCommit() {
        openProject()
        let left = app.coordinate(withNormalizedOffset: CGVector(dx: 0.005, dy: 0.5))
        let short = app.coordinate(withNormalizedOffset: CGVector(dx: 0.15, dy: 0.5))
        left.press(forDuration: 0.1, thenDragTo: short, withVelocity: .slow, thenHoldForDuration: 0.6)
        XCTAssertTrue(element("page-1").exists, "An incomplete edge gesture must cancel without popping the route")
        capture("edge-cancel")
        let end = app.coordinate(withNormalizedOffset: CGVector(dx: 0.88, dy: 0.5))
        left.press(forDuration: 0.1, thenDragTo: end, withVelocity: .slow, thenHoldForDuration: 0.1)
        XCTAssertTrue(app.staticTexts["Sammlungen"].waitForExistence(timeout: 5))
        capture("edge-complete")
    }
    func testViewerInteriorPagesButEdgeReturns() {
        openProject(); element("page-1").tap()
        XCTAssertTrue(app.staticTexts["1 / 4"].waitForExistence(timeout: 5))
        let right = app.coordinate(withNormalizedOffset: CGVector(dx: 0.8, dy: 0.5))
        let left = app.coordinate(withNormalizedOffset: CGVector(dx: 0.25, dy: 0.5))
        right.press(forDuration: 0.1, thenDragTo: left, withVelocity: .slow, thenHoldForDuration: 0)
        XCTAssertTrue(app.staticTexts["2 / 4"].waitForExistence(timeout: 5))
        left.press(forDuration: 0.1, thenDragTo: right, withVelocity: .slow, thenHoldForDuration: 0)
        XCTAssertTrue(app.staticTexts["1 / 4"].waitForExistence(timeout: 5), "Interior right swipe should page, not pop on iOS 26+")
        let edge = app.coordinate(withNormalizedOffset: CGVector(dx: 0.005, dy: 0.5))
        edge.press(forDuration: 0.1, thenDragTo: right, withVelocity: .slow, thenHoldForDuration: 0.1)
        XCTAssertTrue(element("page-1").waitForExistence(timeout: 5))
    }
}
