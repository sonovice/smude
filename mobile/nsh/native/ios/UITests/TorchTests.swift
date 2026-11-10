import XCTest

final class TorchTests: XCTestCase {
    let app = XCUIApplication(bundleIdentifier: "com.gridnotelabs.smude.design")
    func testTorchStartsOffAndFollowsHardwareAcrossCameraAndBackground() {
        continueAfterFailure = false
        app.launchArguments += ["-AppleLanguages", "(de)", "-AppleLocale", "de_DE"]
        app.launch()
        let project = app.buttons["project-1794220037117065"]
        XCTAssertTrue(project.waitForExistence(timeout: 15))
        project.tap()
        let add = app.descendants(matching:.any).matching(identifier:"collection-camera").firstMatch
        XCTAssertTrue(add.waitForExistence(timeout: 5)); add.tap()
        let light = app.buttons["Licht"]
        XCTAssertTrue(light.waitForExistence(timeout: 10))
        let enabled = NSPredicate(format: "enabled == true")
        XCTAssertTrue(expectation(for: enabled, evaluatedWith: light).waitForFulfillment(timeout: 15))
        XCTAssertEqual(light.value as? String, "not checked")
        light.tap()
        let on = NSPredicate(format: "value == %@", "checked")
        XCTAssertTrue(expectation(for: on, evaluatedWith: light).waitForFulfillment(timeout: 5))
        light.tap()
        let off = NSPredicate(format: "value == %@", "not checked")
        XCTAssertTrue(expectation(for: off, evaluatedWith: light).waitForFulfillment(timeout: 5))
        light.tap()
        XCTAssertTrue(expectation(for: on, evaluatedWith: light).waitForFulfillment(timeout: 5))
        XCUIDevice.shared.press(.home)
        app.activate()
        XCTAssertTrue(expectation(for: off, evaluatedWith: light).waitForFulfillment(timeout: 10))
        app.navigationBars.buttons.element(boundBy: 0).tap()
    }
}

private extension XCTestExpectation {
    func waitForFulfillment(timeout: TimeInterval) -> Bool {
        XCTWaiter.wait(for: [self], timeout: timeout) == .completed
    }
}
