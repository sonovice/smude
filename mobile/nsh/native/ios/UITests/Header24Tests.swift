import XCTest
final class Header24Tests:XCTestCase {
 let app=XCUIApplication(bundleIdentifier:"com.notavium.smude")
 func item(_ id:String)->XCUIElement{app.descendants(matching:.any).matching(identifier:id).firstMatch}
 func testTwoHeaderButtons(){
  continueAfterFailure=false;app.launchArguments=["-AppleLanguages","(de)","-AppleLocale","de_DE"];app.launch()
  XCTAssertTrue(item("home-camera").waitForExistence(timeout:15));XCTAssertTrue(item("home-camera").isEnabled)
  item("new-project").tap();XCTAssertTrue(item("project-name").waitForExistence(timeout:5),app.debugDescription)
  app.navigationBars.buttons["BackButton"].tap();XCTAssertTrue(item("home-camera").waitForExistence(timeout:5))
  item("home-camera").tap();XCTAssertTrue(item("shutter").waitForExistence(timeout:5),app.debugDescription)

 }
}
