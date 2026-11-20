import XCTest

final class HeaderToastTests: XCTestCase {
    let app = XCUIApplication(bundleIdentifier: "com.notavium.smude")
    func testHeaderPlusCreatesProjectAndToastDismissesOnTap() {
        continueAfterFailure = false
        app.launchArguments += ["-AppleLanguages", "(de)", "-AppleLocale", "de_DE"]
        app.launch()
        let add = app.buttons["Neue Sammlung"].firstMatch
        XCTAssertTrue(add.waitForExistence(timeout: 15))
        XCTAssertEqual(add.label, "Neue Sammlung")
        XCTAssertLessThan(add.frame.midY, app.frame.minY + 160, "New project belongs in the header")
        add.tap()
        let name = app.textFields.firstMatch
        XCTAssertTrue(name.waitForExistence(timeout: 5))
        let title = "Header-Test-\(UUID().uuidString.prefix(6))"
        name.typeText(title + "\n")
        XCTAssertTrue(app.staticTexts["Noch keine Seiten"].waitForExistence(timeout: 5))
        app.navigationBars.buttons["BackButton"].tap()
        let project = app.buttons.matching(NSPredicate(format: "label CONTAINS %@", title)).firstMatch
        XCTAssertTrue(project.waitForExistence(timeout: 5))
        project.press(forDuration: 0.7)
        app.buttons["Löschen"].tap()
        let toast = app.descendants(matching: .any).matching(identifier: "toast").firstMatch
        XCTAssertTrue(toast.waitForExistence(timeout: 5))
        let bottomGap = app.frame.maxY - toast.frame.maxY
        XCTAssertGreaterThan(bottomGap, 20, "Toast must clear the home indicator")
        XCTAssertLessThan(bottomGap, 80, "Safe area must only be applied once")
        XCTAssertGreaterThanOrEqual(toast.frame.height, 44)
        app.buttons["Rückgängig"].tap()
        XCTAssertTrue(toast.waitForNonExistence(timeout: 2))
        XCTAssertTrue(project.waitForExistence(timeout: 5))
        project.press(forDuration: 0.7)
        app.buttons["Löschen"].tap()
        XCTAssertTrue(toast.waitForExistence(timeout: 5))
        toast.tap()
        XCTAssertTrue(toast.waitForNonExistence(timeout: 2))
    }
}
