import XCTest

/// Explicit four-page fixture installed in the simulator archive by the test harness.
final class ProjectGesturesTests: XCTestCase {
    let app = XCUIApplication(bundleIdentifier: "com.notavium.smude")
    func element(_ id: String) -> XCUIElement { app.descendants(matching: .any).matching(identifier: id).firstMatch }
    func testLongPressReorderSelectionShareAndDelete() {
        continueAfterFailure = false
        app.launchArguments += ["-AppleLanguages", "(de)", "-AppleLocale", "de_DE"]
        app.launch()
        let project = element("project-999000")
        XCTAssertTrue(project.waitForExistence(timeout: 15)); project.tap()
        XCTAssertFalse(app.buttons["Exportieren"].exists)
        XCTAssertFalse(app.buttons["Auswählen"].exists)
        let first = element("page-999001"), second = element("page-999002")
        XCTAssertTrue(first.waitForExistence(timeout: 5))
        let initialY = first.frame.minY
        first.press(forDuration: 0.7)
        XCTAssertTrue(element("page-selection-999001").waitForExistence(timeout: 5))
        XCTAssertEqual(first.frame.minY, initialY, accuracy: 1, "Selection must not insert content above the grid")
        XCTAssertFalse(app.staticTexts["1 ausgewählt"].exists)
        XCTAssertFalse(app.staticTexts["Alle auswählen"].exists)
        XCTAssertFalse(app.staticTexts["Zum Verschieben ziehen"].exists)
        XCTAssertTrue(first.isSelected)
        XCTAssertTrue(element("page-selection-999002").exists)
        first.press(forDuration: 0.1, thenDragTo: second)
        XCTAssertTrue(first.frame.minX > second.frame.minX, "Dragging inserts the page at its new position")
        app.buttons["Teilen"].tap()
        XCTAssertTrue(app.staticTexts["Auswahl: 2"].waitForExistence(timeout: 5))
        app.navigationBars.buttons.element(boundBy: 0).tap()
        app.buttons["Löschen"].tap()
        XCTAssertTrue(app.staticTexts["3 Seiten · auf diesem Gerät"].waitForExistence(timeout: 5))
        let toast = element("toast")
        XCTAssertTrue(toast.waitForExistence(timeout: 5))
        XCTAssertLessThan(toast.frame.maxY, app.frame.maxY - 20, "Toast remains above the home indicator")
        app.buttons["Rückgängig"].tap()
        XCTAssertTrue(app.staticTexts["4 Seiten · auf diesem Gerät"].waitForExistence(timeout: 5))
        app.buttons["Teilen"].tap()
        XCTAssertTrue(app.staticTexts["Auswahl: 1, 2, 3, 4"].waitForExistence(timeout: 5))
    }
    func testProjectLongPressDeleteAndToastExpiry() {
        continueAfterFailure = false
        app.launchArguments += ["-AppleLanguages", "(de)", "-AppleLocale", "de_DE"]
        app.launch()
        let project = element("project-999000")
        XCTAssertTrue(project.waitForExistence(timeout: 15))
        XCTAssertFalse(app.buttons["Löschen"].exists)
        let y = project.frame.minY
        project.press(forDuration: 0.7)
        XCTAssertTrue(app.buttons["Löschen"].waitForExistence(timeout: 5))
        XCTAssertEqual(project.frame.minY, y, accuracy: 1)
        XCTAssertTrue(project.isSelected)
        app.buttons["Löschen"].tap()
        XCTAssertTrue(app.buttons["Rückgängig"].waitForExistence(timeout: 5))
        app.buttons["Rückgängig"].tap()
        XCTAssertTrue(project.waitForExistence(timeout: 5))
        project.press(forDuration: 0.7)
        app.buttons["Löschen"].tap()
        let gone = XCTNSPredicateExpectation(predicate: NSPredicate(format: "exists == false"), object: element("toast"))
        XCTAssertEqual(XCTWaiter.wait(for: [gone], timeout: 10), .completed)
        XCTAssertFalse(app.buttons["Rückgängig"].exists)
    }
}
