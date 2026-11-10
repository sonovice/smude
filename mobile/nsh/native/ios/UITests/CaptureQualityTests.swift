import XCTest

final class CaptureQualityTests: XCTestCase {
    let app = XCUIApplication(bundleIdentifier: "com.gridnotelabs.smude.design")
    func testLocalBlurWarningOffersRetakeAndKeep() {
        continueAfterFailure = false
        app.launchArguments += ["-AppleLanguages", "(de)", "-AppleLocale", "de_DE"]
        app.launch()
        let project = app.buttons.matching(NSPredicate(format: "label CONTAINS %@", "Schärfeprüfung13")).firstMatch
        XCTAssertTrue(project.waitForExistence(timeout: 15))
        project.tap()
        let page = app.buttons["page-9013002"]
        XCTAssertTrue(page.waitForExistence(timeout: 5))
        page.tap()
        let warning = app.staticTexts["Oben möglicherweise unscharf"]
        XCTAssertTrue(warning.waitForExistence(timeout: 5))
        app.buttons["retake-soft-page"].tap()
        XCTAssertTrue(app.navigationBars["Seite ersetzen"].waitForExistence(timeout: 5))
        app.navigationBars.buttons.element(boundBy: 0).tap()
        XCTAssertTrue(warning.waitForExistence(timeout: 5))
        app.buttons["Behalten"].tap()
        XCTAssertTrue(warning.waitForNonExistence(timeout: 2))
        XCTAssertTrue(app.navigationBars["Seite 1 von 2"].exists)
    }
}
