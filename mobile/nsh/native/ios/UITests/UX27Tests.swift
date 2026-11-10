import XCTest

/// Run only against the simulator fixture prepared by native/ios/ux-fixture.py.
final class UX27Tests: XCTestCase {
 let app=XCUIApplication(bundleIdentifier:"com.gridnotelabs.smude.design")
 func el(_ id:String)->XCUIElement{app.descendants(matching:.any).matching(identifier:id).firstMatch}
 func tap(_ id:String){XCTAssertTrue(el(id).waitForExistence(timeout:8),app.debugDescription);el(id).tap()}
 func button(_ label:String){XCTAssertTrue(app.buttons[label].waitForExistence(timeout:5),app.debugDescription);app.buttons[label].firstMatch.tap()}
 func cancelMenu(){if app.buttons["Cancel"].exists{app.buttons["Cancel"].tap()}else{app.coordinate(withNormalizedOffset:CGVector(dx:0.5,dy:0.25)).tap()}}
 func snap(_ name:String){let a=XCTAttachment(screenshot:app.screenshot());a.name=name;a.lifetime = .keepAlways;add(a)}
 override func setUp(){continueAfterFailure=false;app.launchArguments=["-AppleLanguages","(en)","-AppleLocale","en_US"];app.launch();XCTAssertTrue(el("home-camera").waitForExistence(timeout:15))}
 func test01HomeSearchAndCompactScans(){
  tap("project-view-list");snap("27-home-list")
  let draft=el("project-27400");XCTAssertLessThan(draft.frame.height,100)
  tap("search");el("search").typeText("no-such-collection");XCTAssertTrue(el("search-empty").waitForExistence(timeout:5));snap("27-search-empty")
  XCTAssertFalse(app.staticTexts["Unfiled scans"].exists);tap("clear-search");XCTAssertTrue(draft.waitForExistence(timeout:5))
 }
 func test02CollectionAndSelectedShare(){
  tap("project-27000");snap("27-collection")
  let page=el("page-27001");XCTAssertEqual(page.label.trimmingCharacters(in:.whitespacesAndNewlines),"Page 1")
  tap("collection-more");XCTAssertTrue(app.buttons["Rename"].exists);snap("27-collection-menu");button("Rename")
  XCTAssertTrue(app.textFields.firstMatch.waitForExistence(timeout:5));snap("27-rename");app.navigationBars.buttons["BackButton"].tap()
  page.press(forDuration:0.7);tap("page-27002");snap("27-page-selection")
  XCTAssertTrue(app.buttons["Delete 2 pages"].exists)
  tap("collection-more");XCTAssertTrue(app.buttons["Share 2 pages"].exists);snap("27-selection-menu");button("Share 2 pages")
  XCTAssertTrue(app.navigationBars.staticTexts["Share 2 pages"].waitForExistence(timeout:5));snap("27-share-preparing")
  let wait=XCTNSPredicateExpectation(predicate:NSPredicate(format:"enabled == true"),object:el("share-pdf"));XCTAssertEqual(XCTWaiter.wait(for:[wait],timeout:90),.completed)
  snap("27-share-ready");tap("share-filename");snap("27-share-keyboard")
 }
 func test03MissingAndPreviousImages(){
  tap("project-27200");tap("page-27201");XCTAssertTrue(app.buttons["Original"].isSelected);XCTAssertFalse(app.buttons["Processed"].isEnabled);snap("27-failed-original")
  app.navigationBars.buttons["BackButton"].tap();app.navigationBars.buttons["BackButton"].tap()
  tap("project-28100");tap("page-28101");XCTAssertTrue(el("image-placeholder").waitForExistence(timeout:5));snap("27-missing-image")
  app.navigationBars.buttons["BackButton"].tap();app.navigationBars.buttons["BackButton"].tap()
  tap("project-28200");tap("page-28201");XCTAssertTrue(app.buttons["Previous result"].waitForExistence(timeout:5));snap("27-previous-result")
 }
 func test04DraftAndNewCollection(){
  tap("project-27400");XCTAssertTrue(el("draft-assign").waitForExistence(timeout:5));snap("27-draft")
  tap("draft-assign");XCTAssertTrue(app.buttons["＋ Create new collection…"].waitForExistence(timeout:5));snap("27-destination")
  button("＋ Create new collection…");XCTAssertTrue(app.navigationBars.staticTexts["New collection"].waitForExistence(timeout:5));snap("27-draft-name")
 }
 func test05CameraAndRetake(){
  tap("home-camera");XCTAssertTrue(el("shutter").waitForExistence(timeout:5));XCTAssertFalse(el("camera-review").exists)
  XCTAssertTrue(el("camera-problem").waitForExistence(timeout:8));snap("27-camera-unavailable")
  app.navigationBars.buttons["Done"].tap();tap("project-27000");tap("page-27002");tap("viewer-retake")
  XCTAssertTrue(app.navigationBars.staticTexts["Replace page 2"].waitForExistence(timeout:5));XCTAssertFalse(el("camera-review").exists);snap("27-retake")
 }
 func test06EmptyCollection(){tap("project-27300");XCTAssertTrue(el("empty-collection-camera").exists);XCTAssertTrue(el("empty-collection-import").exists);snap("27-empty-collection")}
 func test07HomeDeletionAndUndo(){
  el("project-27500").press(forDuration:0.7);XCTAssertTrue(el("delete-projects").waitForExistence(timeout:5));snap("27-home-selection")
  tap("delete-projects");XCTAssertTrue(app.buttons["Undo"].waitForExistence(timeout:5));button("Undo")
  XCTAssertTrue(el("project-27500").waitForExistence(timeout:5));el("project-27500").press(forDuration:0.7);XCTAssertTrue(el("delete-projects").isEnabled)
 }
 func test08CameraStackOpensCollectionAndIsDisabledWhenEmpty(){
  tap("home-camera");XCTAssertTrue(el("capture-stack").waitForExistence(timeout:5));XCTAssertFalse(el("capture-stack").isEnabled);snap("28-empty-camera-stack")
  app.navigationBars.buttons["Done"].tap();tap("project-27000");tap("collection-camera");XCTAssertTrue(el("capture-stack").isEnabled);snap("28-camera-stack")
  tap("capture-stack");XCTAssertTrue(el("collection-camera").waitForExistence(timeout:5));XCTAssertTrue(el("page-27001").exists);snap("28-stack-review")
 }

}
