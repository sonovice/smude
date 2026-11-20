import XCTest

/// Simulator-only language smoke checks; no camera or user project mutation.
final class LocalizationTests: XCTestCase {
    let app = XCUIApplication(bundleIdentifier: "com.notavium.smude")
    func testNativeLanguageNegotiationAndCollectionName() {
        continueAfterFailure = false
        for (languages, region, title, addLabel) in [
            ("(en)", "en_GB", "Collections", "New collection"),
            ("(de)", "de_DE", "Sammlungen", "Neue Sammlung"),
            ("(pt-BR)", "pt_BR", "Coleções", "Nova coleção"),
            ("(pt-PT)", "pt_PT", "Coleções", "Nova coleção"),
            ("(fr, de)", "fr_FR", "Sammlungen", "Neue Sammlung")
        ] {
            app.terminate()
            app.launchArguments = ["-AppleLanguages", languages, "-AppleLocale", region]
            app.launch()
            XCTAssertTrue(app.staticTexts[title].waitForExistence(timeout: 20), app.debugDescription)
            let plus=app.buttons[addLabel].firstMatch
            XCTAssertTrue(plus.waitForExistence(timeout: 10),app.debugDescription)
            XCTAssertEqual(plus.label, addLabel)
            let capture=XCTAttachment(screenshot: app.screenshot());capture.name=languages;capture.lifetime = .keepAlways;add(capture)
        }
    }
}
