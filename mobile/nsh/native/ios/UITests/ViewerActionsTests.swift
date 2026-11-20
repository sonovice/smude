import XCTest

/// Uses an existing simulator page; does not rotate, replace or delete it.
final class ViewerActionsTests: XCTestCase {
    let app=XCUIApplication(bundleIdentifier:"com.notavium.smude")
    func testCompactPageActionsAndNativeMoreChooser() {
        continueAfterFailure=false
        app.launchArguments=["-AppleLanguages","(de)","-AppleLocale","de_DE"]
        app.launch()
        XCTAssertTrue(app.buttons["Listenansicht"].waitForExistence(timeout:15))
        app.buttons["Listenansicht"].tap()
        let collection=app.buttons.matching(NSPredicate(format:"label CONTAINS %@", "1 Seite,")).firstMatch
        XCTAssertTrue(collection.waitForExistence(timeout:5));collection.tap()
        let page=app.descendants(matching:.any).matching(NSPredicate(format:"identifier MATCHES %@","page-[0-9]+")).firstMatch
        XCTAssertTrue(page.waitForExistence(timeout:5));page.tap()
        for id in ["viewer-rotate","viewer-retake","viewer-more"] {
            XCTAssertTrue(app.descendants(matching:.any).matching(identifier:id).firstMatch.waitForExistence(timeout:5),app.debugDescription)
        }
        let screenshot=XCTAttachment(screenshot:app.screenshot());screenshot.name="Compact page actions";screenshot.lifetime = .keepAlways;add(screenshot)
        app.descendants(matching:.any).matching(identifier:"viewer-more").firstMatch.tap()
        XCTAssertTrue(app.buttons["Seite löschen"].waitForExistence(timeout:5))
        XCTAssertFalse(app.buttons["Entzerrung"].exists);XCTAssertFalse(app.buttons["GCS"].exists)
        XCTAssertTrue(app.buttons["Seite löschen"].exists)
        if app.buttons["Abbrechen"].exists { app.buttons["Abbrechen"].tap() }
        else { app.coordinate(withNormalizedOffset:CGVector(dx:0.5,dy:0.3)).tap() }
        let gone=XCTNSPredicateExpectation(predicate:NSPredicate(format:"exists == false"),object:app.buttons["Seite löschen"])
        XCTAssertEqual(XCTWaiter.wait(for:[gone],timeout:5),.completed)
        XCTAssertTrue(app.descendants(matching:.any).matching(identifier:"viewer-more").firstMatch.exists)
    }
}
