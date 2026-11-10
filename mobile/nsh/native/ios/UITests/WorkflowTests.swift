import XCTest

/// Runs against an installed app with a real photo in the simulator Photos library.
final class WorkflowTests: XCTestCase {
    let app = XCUIApplication(bundleIdentifier: "com.gridnotelabs.smude.design")
    func element(_ id: String) -> XCUIElement { app.descendants(matching: .any).matching(identifier: id).firstMatch }
    func capture(_ name: String) {
        let a = XCTAttachment(screenshot: app.screenshot()); a.name = name; a.lifetime = .keepAlways; add(a)
    }
    func testProjectImportAndExport() {
        continueAfterFailure = false
        app.launchArguments += ["-AppleLanguages", "(de)", "-AppleLocale", "de_DE"]
        app.launch()
        XCTAssertTrue(element("new-project").waitForExistence(timeout: 20))
        element("new-project").tap()
        let input = app.textFields.firstMatch
        XCTAssertTrue(input.waitForExistence(timeout: 10))
        XCTAssertTrue(app.keyboards.firstMatch.waitForExistence(timeout: 10), "Project name receives keyboard focus")
        input.typeText("iOS Funktionstest\n")
        XCTAssertTrue(element("collection-more").waitForExistence(timeout:10));element("collection-more").tap()
        XCTAssertTrue(app.buttons["Fotos importieren"].waitForExistence(timeout: 10))
        app.buttons["Fotos importieren"].tap()
        capture("photo-picker")
        print("PHOTO_PICKER_TREE\n" + app.debugDescription)
        let photos = app.images.matching(NSPredicate(format: "label CONTAINS[c] 'Photo' OR label CONTAINS[c] 'Foto'"))
        XCTAssertTrue(photos.firstMatch.waitForExistence(timeout: 15))
        photos.firstMatch.tap()
        let add = app.buttons.matching(NSPredicate(format: "identifier == 'Add' OR label == 'Done' OR label == 'Hinzufügen'")).firstMatch
        if add.waitForExistence(timeout: 3) { add.tap() }
        capture("imported")
        print("IMPORTED_TREE\n" + app.debugDescription)
        let page = app.descendants(matching: .any).matching(NSPredicate(format: "identifier BEGINSWITH 'page-'")).firstMatch
        XCTAssertTrue(page.waitForExistence(timeout: 120))
        page.tap()
        XCTAssertTrue(app.navigationBars.buttons["Teilen"].waitForExistence(timeout: 10))
        XCTAssertEqual(XCTWaiter.wait(for: [XCTNSPredicateExpectation(predicate: NSPredicate(format: "exists == false"), object: app.staticTexts["Wird entzerrt …"])], timeout: 180), .completed)
        capture("processed")
        print("PROCESSED_TREE\n" + app.debugDescription)
        element("viewer-rotate").tap()
        capture("mesh")
        app.navigationBars.buttons["Teilen"].firstMatch.tap()
        XCTAssertTrue(element("share-pdf").waitForExistence(timeout:10))
        let ready=XCTNSPredicateExpectation(predicate:NSPredicate(format:"enabled == true"),object:element("share-pdf"))
        XCTAssertEqual(XCTWaiter.wait(for:[ready],timeout:90),.completed)
        element("share-pdf").tap()
        let share = app.otherElements["ActivityListView"]
        XCTAssertTrue(share.waitForExistence(timeout: 90))
        capture("share")
    }
}
