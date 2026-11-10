import XCTest
final class VisualAudit26Tests:XCTestCase {
 let app=XCUIApplication(bundleIdentifier:"com.gridnotelabs.smude.design")
 func el(_ id:String)->XCUIElement {app.descendants(matching:.any).matching(identifier:id).firstMatch}
 func snap(_ name:String){Thread.sleep(forTimeInterval:0.5);let a=XCTAttachment(screenshot:app.screenshot());a.name=name;a.lifetime = .keepAlways;add(a)}
 func tap(_ id:String){XCTAssertTrue(el(id).waitForExistence(timeout:8),app.debugDescription);el(id).tap()}
 func button(_ label:String){if label == "Cancel" && !app.buttons[label].exists {app.coordinate(withNormalizedOffset:CGVector(dx:0.5,dy:0.3)).tap();return};XCTAssertTrue(app.buttons[label].waitForExistence(timeout:5),app.debugDescription);app.buttons[label].firstMatch.tap()}
 func back(){app.navigationBars.buttons["BackButton"].tap()}
 override func setUp(){continueAfterFailure=false;app.launchArguments=["-AppleLanguages","(en)","-AppleLocale","en_US"];app.launch();XCTAssertTrue(el("home-camera").waitForExistence(timeout:15))}
 func test01Home(){tap("project-view-list");snap("01-ios-home-list");tap("project-view-grid");snap("02-ios-home-grid");tap("search");el("search").typeText("zznomatch");snap("03-ios-search-empty-keyboard")}
 func test02Name(){tap("new-project");snap("04-ios-new-name-empty");el("project-name").typeText("Morning studies");snap("05-ios-new-name-keyboard");button("Save");snap("06-ios-collection-empty")}
 func test03Collection(){tap("project-27000");snap("07-ios-collection");tap("collection-more");snap("08-ios-collection-menu");button("Cancel");el("project-title").press(forDuration:1);snap("09-ios-title-context");button("Rename");snap("10-ios-rename-keyboard")}
 func test04Viewer(){tap("project-27000");tap("page-27001");snap("11-ios-viewer-result");button("Original");snap("12-ios-viewer-original");tap("viewer-more");snap("13-ios-viewer-menu");button("Cancel");tap("viewer-retake");snap("14-ios-camera-retake")}
 func test05Share(){tap("project-27000");tap("collection-more");button("Share");snap("15-ios-share-preparing");let ready=XCTNSPredicateExpectation(predicate:NSPredicate(format:"enabled == true"),object:el("share-pdf"));_ = XCTWaiter.wait(for:[ready],timeout:60);snap("16-ios-share-ready");tap("share-filename");snap("17-ios-share-keyboard");app.keyboards.buttons["Done"].firstMatch.tap();tap("share-pdf");Thread.sleep(forTimeInterval:2);snap("18-ios-system-share")}
 func test06Selection(){tap("project-27000");el("page-27001").press(forDuration:0.7);tap("page-27002");snap("19-ios-page-selection");tap("collection-more");snap("20-ios-selected-menu");button("Share");Thread.sleep(forTimeInterval:2);snap("21-ios-selected-share")}
 func test07Delete(){tap("project-27000");el("page-27012").press(forDuration:0.7);button("Delete");snap("22-ios-delete-undo")}
 func test08Camera(){tap("home-camera");Thread.sleep(forTimeInterval:2);snap("23-ios-camera-home")}
 func test09Soft(){tap("project-27100");tap("page-27101");snap("24-ios-focus-warning");button("Keep");snap("25-ios-focus-kept")}
 func test10Failed(){tap("project-27200");tap("page-27201");snap("26-ios-failed-page")}
 func test11Draft(){tap("project-27400");snap("27-ios-draft");tap("collection-more");snap("28-ios-draft-menu");button("Add to collection");snap("29-ios-draft-destination");button("New collection");snap("30-ios-draft-new-name")}
 func test12Import(){tap("project-27300");tap("collection-more");button("Files");Thread.sleep(forTimeInterval:1);snap("31-ios-system-files")}
 func test13Photos(){tap("project-27300");tap("collection-more");button("Import photos");Thread.sleep(forTimeInterval:1);snap("32-ios-system-photos")}
 func test14HomeSelection(){el("project-27500").press(forDuration:0.7);snap("33-ios-home-selection");tap("delete-projects");snap("34-ios-home-delete-undo")}
 func test15MoreStates(){tap("project-27000");tap("collection-camera");snap("35-ios-camera-collection");tap("camera-done");tap("page-27001");let pager=el("page-pager");pager.pinch(withScale:2,velocity:1);snap("36-ios-viewer-zoom");tap("viewer-rotate");snap("37-ios-viewer-rotated")}
 func test16Reorder(){tap("project-27000");let first=el("page-27001"),target=el("page-27003");first.press(forDuration:0.7,thenDragTo:target,withVelocity:.slow,thenHoldForDuration:1);snap("38-ios-reorder-after-drop")}
}
