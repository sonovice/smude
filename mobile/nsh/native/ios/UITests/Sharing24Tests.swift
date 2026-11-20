import XCTest

/// Native simulator QA only; no launch or tests on physical devices.
final class Sharing24Tests:XCTestCase {
    let app=XCUIApplication(bundleIdentifier:"com.notavium.smude")
    func item(_ id:String)->XCUIElement{app.descendants(matching:.any).matching(identifier:id).firstMatch}
    func capture(_ name:String){let a=XCTAttachment(screenshot:app.screenshot());a.name=name;a.lifetime = .keepAlways;add(a)}
    func testCameraHeaderViewerAndMinimalPdfShare(){
        continueAfterFailure=false
        app.launchArguments=["-AppleLanguages","(de)","-AppleLocale","de_DE"]
        app.launch()
        XCTAssertTrue(item("home-camera").waitForExistence(timeout:15));XCTAssertTrue(item("new-project").exists)
        XCTAssertLessThan(item("home-camera").frame.midX,item("new-project").frame.midX)
        item("home-camera").tap();XCTAssertTrue(app.navigationBars.buttons["Fertig"].waitForExistence(timeout:5));app.navigationBars.buttons["Fertig"].tap()
        XCTAssertTrue(app.buttons["Listenansicht"].waitForExistence(timeout:5));app.buttons["Listenansicht"].tap()
        let collection=app.buttons.matching(NSPredicate(format:"label CONTAINS %@","1 Seite,")).firstMatch
        XCTAssertTrue(collection.waitForExistence(timeout:5));collection.tap()
        XCTAssertTrue(item("collection-camera").waitForExistence(timeout:5));XCTAssertTrue(item("collection-more").exists)
        XCTAssertFalse(app.buttons["Fotos importieren"].exists)
        let page=app.descendants(matching:.any).matching(NSPredicate(format:"identifier MATCHES %@","page-[0-9]+")).firstMatch
        XCTAssertTrue(page.waitForExistence(timeout:5));page.tap()
        XCTAssertTrue(item("viewer-rotate").waitForExistence(timeout:5))
        let top=app.buttons["Original"].frame.minY-app.navigationBars.firstMatch.frame.maxY
        XCTAssertGreaterThanOrEqual(top,10,"Comparison control needs breathing room")
        item("viewer-more").tap();XCTAssertTrue(app.buttons["Seite löschen"].waitForExistence(timeout:5));XCTAssertFalse(app.buttons["Entzerrung"].exists);XCTAssertFalse(app.buttons["GCS"].exists)
        if app.buttons["Abbrechen"].exists {app.buttons["Abbrechen"].tap()} else {app.coordinate(withNormalizedOffset:CGVector(dx:0.5,dy:0.3)).tap()}
        XCTAssertTrue(app.buttons["Seite löschen"].waitForNonExistence(timeout:5));capture("Viewer with rotation icon and spacing")
        app.navigationBars.buttons["Teilen"].tap();XCTAssertTrue(item("share-filename").waitForExistence(timeout:5))
        XCTAssertTrue(app.navigationBars.staticTexts["Teilen"].exists)
        XCTAssertTrue(app.staticTexts[".pdf"].exists);XCTAssertEqual(app.textFields.count,1)
        for id in ["share-size-0","share-size-1","share-size-2"] {XCTAssertTrue(item(id).exists)}
        for label in ["JPEG","Format","Darstellung","Exportvorschau","Auswahl:"] {XCTAssertFalse(app.staticTexts[label].exists)}
        let enabled=XCTNSPredicateExpectation(predicate:NSPredicate(format:"enabled == true"),object:item("share-pdf"))
        XCTAssertEqual(XCTWaiter.wait(for:[enabled],timeout:90),.completed,app.debugDescription)
        XCTAssertNotNil(item("share-size-2").label.range(of:#"[0-9,.]+ (kB|MB)"#,options:.regularExpression))
        capture("PDF sharing with exact file sizes")
        item("share-pdf").tap()
        XCTAssertTrue(app.otherElements["ActivityListView"].waitForExistence(timeout:20),app.debugDescription)
        capture("Native PDF share sheet")
        let close=app.buttons.matching(NSPredicate(format:"label == 'Schließen' OR label == 'Close' OR identifier == 'Close'")).firstMatch
        if close.exists { close.tap() } else { app.swipeDown() }
        XCTAssertTrue(item("share-filename").waitForExistence(timeout:5))
        for _ in 0..<3 {
            app.navigationBars.buttons["BackButton"].tap();XCTAssertTrue(app.navigationBars.buttons["Teilen"].waitForExistence(timeout:5))
            app.navigationBars.buttons["Teilen"].tap();XCTAssertTrue(item("share-filename").waitForExistence(timeout:5))
            XCTAssertTrue(app.navigationBars.buttons["BackButton"].exists)
        }
        app.navigationBars.buttons["BackButton"].tap();XCTAssertTrue(item("viewer-more").waitForExistence(timeout:5))
        let edge=app.coordinate(withNormalizedOffset:CGVector(dx:0.005,dy:0.5));let destination=app.coordinate(withNormalizedOffset:CGVector(dx:0.9,dy:0.5))
        let short=app.coordinate(withNormalizedOffset:CGVector(dx:0.15,dy:0.5))
        edge.press(forDuration:0.1,thenDragTo:short,withVelocity:.slow,thenHoldForDuration:0.6)
        XCTAssertTrue(item("viewer-more").exists,"A cancelled edge gesture must retain the viewer")
        edge.press(forDuration:0.1,thenDragTo:destination,withVelocity:.slow,thenHoldForDuration:0.1)
        XCTAssertTrue(item("collection-more").waitForExistence(timeout:5));capture("Native back returns to collection")
    }
}
