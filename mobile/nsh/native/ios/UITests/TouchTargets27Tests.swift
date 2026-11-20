import XCTest
final class TouchTargets27Tests:XCTestCase {
 func testSystemHeaderAcceptsTapsBeyondThe36PointAccessibilityImage(){
  continueAfterFailure=false
  let app=XCUIApplication(bundleIdentifier:"com.notavium.smude")
  app.launchArguments=["-AppleLanguages","(en)","-AppleLocale","en_US"];app.launch()
  let camera=app.buttons["home-camera"];XCTAssertTrue(camera.waitForExistence(timeout:15))
  // 21pt from center lies within a 44pt target, beyond the visible 36pt AX frame.
  camera.coordinate(withNormalizedOffset:CGVector(dx:0.5,dy:0.5)).withOffset(CGVector(dx:-21,dy:0)).tap()
  XCTAssertTrue(app.buttons["shutter"].waitForExistence(timeout:5),"Native header hit target must extend beyond the icon")
  app.navigationBars.buttons["Done"].tap()
 }
}
