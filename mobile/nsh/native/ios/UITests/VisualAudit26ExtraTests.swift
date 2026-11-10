import XCTest
final class VisualAudit26ExtraTests:XCTestCase {
 let app=XCUIApplication(bundleIdentifier:"com.gridnotelabs.smude.design")
 func el(_ id:String)->XCUIElement {app.descendants(matching:.any).matching(identifier:id).firstMatch}
 func snap(_ name:String){let a=XCTAttachment(screenshot:XCUIScreen.main.screenshot());a.name=name;a.lifetime = .keepAlways;add(a);let d=XCTAttachment(string:app.debugDescription);d.name=name+"-hierarchy";d.lifetime = .keepAlways;add(d)}
 func tap(_ id:String){XCTAssertTrue(el(id).waitForExistence(timeout:8));el(id).tap()}
 override func setUp(){continueAfterFailure=false;app.launchArguments=["-AppleLanguages","(en)","-AppleLocale","en_US"];app.launch();XCTAssertTrue(el("home-camera").waitForExistence(timeout:15))}
 func test01Files(){tap("project-27300");tap("collection-more");app.buttons["Files"].tap();Thread.sleep(forTimeInterval:4);snap("71-ios-files-fullscreen")}
 func test02Photos(){tap("project-27300");tap("collection-more");app.buttons["Import photos"].tap();Thread.sleep(forTimeInterval:4);snap("72-ios-photos-fullscreen")}
 func test03Zoom(){tap("project-27000");tap("page-27001");el("page-pager").pinch(withScale:2,velocity:1);snap("73-ios-zoom");tap("viewer-rotate");snap("74-ios-rotated")}
 func test04Processing(){tap("project-27200");tap("page-27201");app.buttons["Try again"].tap();snap("75-ios-processing");Thread.sleep(forTimeInterval:2);snap("76-ios-processing-later")}
 func test05CameraPhotos(){tap("home-camera");app.buttons["Photos"].tap();Thread.sleep(forTimeInterval:3);snap("77-ios-camera-photos")}
 func test06Pending(){tap("project-28000");snap("79-ios-pending-grid");tap("page-28001");snap("80-ios-pending-viewer");app.navigationBars.buttons["BackButton"].tap();tap("page-28002");snap("81-ios-importing-viewer")}
 func test07Empty(){snap("82-ios-home-empty");app.terminate();app.launchArguments=["-AppleLanguages","(pt)","-AppleLocale","pt_PT"];app.launch();XCTAssertTrue(el("home-camera").waitForExistence(timeout:15));snap("83-ios-home-empty-pt");tap("new-project");el("project-name").typeText("Estudos para o ensaio da orquestra de primavera");snap("84-ios-name-long-pt")}
}
