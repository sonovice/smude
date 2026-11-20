import XCTest

/// Simulator-only; fixtures are seeded/restored outside the test runner.
final class NativeControls29Tests: XCTestCase {
 let app=XCUIApplication(bundleIdentifier:"com.notavium.smude")
 func el(_ id:String)->XCUIElement{app.descendants(matching:.any).matching(identifier:id).firstMatch}
 func tap(_ id:String){XCTAssertTrue(el(id).waitForExistence(timeout:8),app.debugDescription);el(id).tap()}
 func snap(_ name:String){let a=XCTAttachment(screenshot:app.screenshot());a.name=name;a.lifetime = .keepAlways;add(a)}
 override func setUp(){continueAfterFailure=false;app.launchArguments=["-AppleLanguages","(en)","-AppleLocale","en_US"];app.launch();XCTAssertTrue(el("home-camera").waitForExistence(timeout:15))}
 func test00HeaderUsesIndependentSystemBarButtons(){
  tap("project-27000")
  let bar=app.navigationBars.firstMatch
  XCTAssertEqual(bar.buttons.count,3,app.debugDescription)
  let camera=bar.buttons["collection-camera"],more=bar.buttons["collection-more"]
  XCTAssertTrue(camera.exists);XCTAssertTrue(more.exists)
  XCTAssertLessThanOrEqual(camera.frame.maxX,more.frame.minX)
  XCTAssertEqual(camera.frame.midY,more.frame.midY,accuracy:1)
  snap("30-independent-native-bar-buttons")
 }
 func test01CollectionMenuAnchorsToButtonAndSharesSelection(){
  tap("project-27000");tap("collection-more")
  let rename=app.buttons["Rename"].firstMatch;XCTAssertTrue(rename.waitForExistence(timeout:5),app.debugDescription)
  XCTAssertLessThan(rename.frame.midY,app.frame.height*0.55);snap("29-collection-native-menu")
  app.coordinate(withNormalizedOffset:CGVector(dx:0.2,dy:0.7)).tap()
  el("page-27001").press(forDuration:0.7);tap("page-27002");tap("collection-more")
  XCTAssertFalse(app.buttons["Rename"].exists);let share=app.buttons["Share 2 pages"].firstMatch;XCTAssertTrue(share.waitForExistence(timeout:5));snap("29-selection-native-menu");share.tap()
  XCTAssertTrue(app.navigationBars.staticTexts["Share 2 pages"].waitForExistence(timeout:5))
 }
 func test02ViewerUsesUIKitSegmentsAndAnchoredMenu(){
  tap("project-27000");tap("page-27001")
  let segments=app.segmentedControls.firstMatch;XCTAssertTrue(segments.waitForExistence(timeout:5),app.debugDescription)
  XCTAssertTrue(segments.buttons["Processed"].isSelected);segments.buttons["Original"].tap();XCTAssertTrue(segments.buttons["Original"].isSelected)
  segments.buttons["Processed"].tap();XCTAssertTrue(segments.buttons["Processed"].isSelected)
  XCTAssertTrue(app.buttons["viewer-rotate"].exists);tap("viewer-rotate");snap("29-viewer-native-controls")
  tap("viewer-more");let delete=app.buttons["Delete page"].firstMatch;XCTAssertTrue(delete.waitForExistence(timeout:5));XCTAssertLessThan(delete.frame.midY,app.frame.height*0.5);snap("29-viewer-native-menu")
  app.coordinate(withNormalizedOffset:CGVector(dx:0.2,dy:0.7)).tap();tap("viewer-retake");XCTAssertTrue(app.navigationBars.staticTexts["Replace page 1"].waitForExistence(timeout:5))
 }
 func test03UnavailableResultRemainsDisabled(){
  tap("project-27200");tap("page-27201");let segments=app.segmentedControls.firstMatch
  XCTAssertTrue(segments.waitForExistence(timeout:5));XCTAssertTrue(segments.buttons["Original"].isSelected);XCTAssertFalse(segments.buttons["Processed"].isEnabled);snap("29-native-disabled-result")
 }
 func test04NativeDeleteActionPreservesUndo(){
  tap("project-27000");tap("page-27006");tap("viewer-more");let delete=app.buttons["Delete page"].firstMatch
  XCTAssertTrue(delete.waitForExistence(timeout:5));delete.tap();XCTAssertTrue(el("collection-camera").waitForExistence(timeout:5));XCTAssertFalse(el("page-27006").exists)
  let undo=app.buttons["Undo"].firstMatch;XCTAssertTrue(undo.waitForExistence(timeout:5));undo.tap();XCTAssertTrue(el("page-27006").waitForExistence(timeout:5))
 }

 func test05HomeLayoutUsesSystemSegmentedControl(){
  let segments=app.segmentedControls["project-layout"]
  XCTAssertTrue(segments.waitForExistence(timeout:5),app.debugDescription)
  XCTAssertEqual(segments.buttons.count,2)
  segments.buttons.element(boundBy:1).tap()
  XCTAssertTrue(segments.buttons.element(boundBy:1).isSelected)
  snap("32-native-home-layout")
  segments.buttons.element(boundBy:0).tap()
  XCTAssertTrue(segments.buttons.element(boundBy:0).isSelected)
 }

}
