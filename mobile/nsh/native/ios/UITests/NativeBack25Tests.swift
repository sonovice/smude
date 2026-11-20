import XCTest
final class NativeBack25Tests:XCTestCase {
 let app=XCUIApplication(bundleIdentifier:"com.notavium.smude")
 func testBackUsesTheSystemChevronWithoutCustomOverlay(){
  continueAfterFailure=false;app.launchArguments=["-AppleLanguages","(de)","-AppleLocale","de_DE"];app.launch()
  let plus=app.buttons["new-project"];XCTAssertTrue(plus.waitForExistence(timeout:15));plus.tap()
  let back=app.navigationBars.buttons["BackButton"]
  XCTAssertTrue(back.waitForExistence(timeout:5),app.debugDescription)
  XCTAssertLessThanOrEqual(back.frame.width,60,"Minimal native back should contain a chevron, not a text label")
  XCTAssertFalse(app.descendants(matching:.any).matching(identifier:"screen-back").firstMatch.exists)
  let shot=XCTAttachment(screenshot:app.screenshot());shot.name="Native system chevron";shot.lifetime = .keepAlways;add(shot)
  back.tap();XCTAssertTrue(plus.waitForExistence(timeout:5))
 }
}
