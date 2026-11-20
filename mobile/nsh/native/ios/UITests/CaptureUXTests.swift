import XCTest

/// Simulator UI only. Physical devices receive installation without tests.
final class CaptureUXTests:XCTestCase {
    let app=XCUIApplication(bundleIdentifier:"com.notavium.smude")
    func item(_ id:String)->XCUIElement{app.descendants(matching:.any).matching(identifier:id).firstMatch}
    func testCollectionHeaderComparisonPagerAndPhonePortrait(){
        continueAfterFailure=false
        app.launchArguments=["-AppleLanguages","(de)","-AppleLocale","de_DE"]
        app.launch()
        XCTAssertTrue(app.buttons["Listenansicht"].waitForExistence(timeout:15));app.buttons["Listenansicht"].tap()
        let collection=app.buttons.matching(NSPredicate(format:"label CONTAINS %@","1 Seite,")).firstMatch
        XCTAssertTrue(collection.waitForExistence(timeout:5))
        let name=collection.label.components(separatedBy:",").first!
        collection.tap()
        XCTAssertTrue(app.navigationBars.staticTexts[name].waitForExistence(timeout:5),app.debugDescription)
        XCTAssertFalse(app.staticTexts.matching(NSPredicate(format:"label CONTAINS %@","auf diesem Gerät")).firstMatch.exists)
        XCTAssertTrue(item("collection-camera").exists);XCTAssertTrue(item("collection-more").exists)
        XCTAssertFalse(app.buttons["Fotos importieren"].exists);XCTAssertFalse(app.buttons["Dateien"].exists)
        item("collection-camera").tap()
        XCTAssertTrue(app.navigationBars.buttons["Fertig"].waitForExistence(timeout:5))
        XCTAssertTrue(app.navigationBars.buttons["Fertig"].isEnabled)
        app.navigationBars.buttons["Fertig"].tap()
        let page=app.descendants(matching:.any).matching(NSPredicate(format:"identifier MATCHES %@","page-[0-9]+")).firstMatch
        XCTAssertTrue(page.waitForExistence(timeout:5));page.tap()
        XCTAssertTrue(app.buttons["Original"].waitForExistence(timeout:5));XCTAssertTrue(app.buttons["Bearbeitet"].exists)
        app.buttons["Original"].tap();app.buttons["Bearbeitet"].tap()
        XCTAssertEqual(item("page-previous").frame.midY,item("page-position").frame.midY,accuracy:1)
        XCTAssertEqual(item("page-next").frame.midY,item("page-position").frame.midY,accuracy:1)
        XCUIDevice.shared.orientation = .landscapeLeft
        XCTAssertTrue(app.wait(for:.runningForeground,timeout:5))
        XCTAssertLessThan(app.frame.width,app.frame.height)
        XCUIDevice.shared.orientation = .portrait
        let a=XCTAttachment(screenshot:app.screenshot());a.name="Capture UX viewer";a.lifetime = .keepAlways;add(a)
    }
}
