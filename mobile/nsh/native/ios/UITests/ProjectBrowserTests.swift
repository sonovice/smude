import XCTest

final class ProjectBrowserTests: XCTestCase {
    let app = XCUIApplication(bundleIdentifier: "com.notavium.smude")
    func testCompactListAndPersistentGrid() {
        continueAfterFailure = false
        app.launchArguments += ["-AppleLanguages", "(de)", "-AppleLocale", "de_DE"]
        app.launch()
        let list = app.buttons["Listenansicht"]
        XCTAssertTrue(list.waitForExistence(timeout: 15)); list.tap()
        let cards = app.buttons.matching(NSPredicate(format: "identifier MATCHES %@", "project-[0-9]+"))
        XCTAssertTrue(cards.firstMatch.waitForExistence(timeout: 5))
        XCTAssertLessThan(cards.firstMatch.frame.height, 100)
        XCTAssertNotNil(cards.firstMatch.label.range(of: #"\d{2}\.\d{2}\.\d{4}"#, options: .regularExpression))
        let listImage = XCTAttachment(screenshot: app.screenshot())
        listImage.name = "Compact project list"; listImage.lifetime = .keepAlways; add(listImage)
        app.buttons["Rasteransicht"].tap()
        XCTAssertTrue(app.buttons["Rasteransicht"].isSelected)
        let first = cards.element(boundBy: 0), second = cards.element(boundBy: 1), third = cards.element(boundBy: 2), fourth = cards.element(boundBy: 3)
        XCTAssertEqual(first.frame.minY, second.frame.minY, accuracy: 1)
        XCTAssertGreaterThan(second.frame.minX, first.frame.minX)
        XCTAssertEqual(third.frame.minY, first.frame.minY, accuracy: 1)
        XCTAssertGreaterThan(third.frame.minX, second.frame.minX)
        XCTAssertGreaterThan(fourth.frame.minY, first.frame.minY)
        let gridImage = XCTAttachment(screenshot: app.screenshot())
        gridImage.name = "Project grid"; gridImage.lifetime = .keepAlways; add(gridImage)
        app.terminate(); app.launchArguments += ["-AppleLanguages", "(de)", "-AppleLocale", "de_DE"]
        app.launch()
        XCTAssertTrue(app.buttons["Rasteransicht"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.buttons["Rasteransicht"].isSelected)
        app.buttons["Listenansicht"].tap()
    }
}
