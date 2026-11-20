import XCTest

final class KeyboardTests: XCTestCase {
    let app = XCUIApplication(bundleIdentifier: "com.notavium.smude")
    func testProjectSaveRemainsAboveTheSoftwareKeyboard() throws {
        continueAfterFailure = false
        app.launchArguments += ["-AppleLanguages", "(de)", "-AppleLocale", "de_DE"]
        app.launch()
        let add = app.navigationBars.buttons.firstMatch
        XCTAssertTrue(add.waitForExistence(timeout: 15))
        XCTAssertEqual(add.label, "Neue Sammlung")
        add.tap()
        let field = app.textFields.firstMatch
        XCTAssertTrue(field.waitForExistence(timeout: 5))
        let keyboard = app.keyboards.firstMatch
        XCTAssertTrue(keyboard.waitForExistence(timeout: 5))
        field.typeText("Keyboard-Test-\(UUID().uuidString.prefix(6))")
        try XCTSkipUnless(keyboard.isHittable, "Software keyboard unavailable (hardware keyboard attached)")
        XCTAssertLessThan(keyboard.frame.minY, app.frame.maxY - 150)
        print("Keyboard frame: \(keyboard.frame), app: \(app.frame)")
        let screenshot=XCTAttachment(screenshot: app.screenshot());screenshot.lifetime = .keepAlways;self.add(screenshot)
        let save = app.buttons["Speichern"]
        XCTAssertTrue(save.exists)
        XCTAssertLessThanOrEqual(save.frame.maxY, keyboard.frame.minY - 8,
                                 "Save must move above the keyboard without requiring Return")
        XCTAssertTrue(save.isHittable)
        save.tap()
        XCTAssertTrue(app.staticTexts["Noch keine Seiten"].waitForExistence(timeout: 5))
        XCTAssertTrue(keyboard.waitForNonExistence(timeout: 3))
    }
}
