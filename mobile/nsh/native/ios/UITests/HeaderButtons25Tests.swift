import XCTest

/// Native rendering regression: distinct actions must not share one capsule.
final class HeaderButtons25Tests:XCTestCase {
 let app=XCUIApplication(bundleIdentifier:"com.notavium.smude")
 func item(_ id:String)->XCUIElement{app.descendants(matching:.any).matching(identifier:id).firstMatch}
 func assertSeparate(_ first:String,_ second:String,file:StaticString=#filePath,line:UInt=#line){
  let left=item(first),right=item(second)
  XCTAssertTrue(left.waitForExistence(timeout:10),file:file,line:line);XCTAssertTrue(right.exists,file:file,line:line)
  XCTAssertLessThan(left.frame.maxX,right.frame.minX,file:file,line:line)
  print("HEADER_ACTION_FRAMES \(first)=\(left.frame) \(second)=\(right.frame)")
  XCTAssertTrue(left.isHittable,file:file,line:line);XCTAssertTrue(right.isHittable,file:file,line:line)
  let image=app.screenshot().image.cgImage!
  XCTAssertEqual(image.bitsPerPixel,32)
  let data=image.dataProvider!.data!;let bytes=CFDataGetBytePtr(data)!
  func shade(_ point:CGPoint)->Int{
   let x=Int(point.x*CGFloat(image.width)/app.frame.width),y=Int(point.y*CGFloat(image.height)/app.frame.height)
   let i=y*image.bytesPerRow+x*4
   return Int(min(bytes[i],bytes[i+1],bytes[i+2]))
  }
  let gap=shade(CGPoint(x:(left.frame.maxX+right.frame.minX)/2,y:left.frame.midY))
  let background=shade(CGPoint(x:5,y:left.frame.midY))
  XCTAssertGreaterThan(shade(CGPoint(x:left.frame.midX,y:left.frame.midY)),background+12,"Leading action must be visibly rendered",file:file,line:line)
  XCTAssertGreaterThan(shade(CGPoint(x:right.frame.midX,y:right.frame.midY)),background+12,"Trailing action must be visibly rendered",file:file,line:line)
  XCTAssertLessThanOrEqual(abs(gap-background),12,"A shared capsule still fills the gap: gap=\(gap), background=\(background)",file:file,line:line)
 }
 func testHomeAndCollectionUseSeparateNativeButtons(){
  continueAfterFailure=false;app.launchArguments=["-AppleLanguages","(de)","-AppleLocale","de_DE"];app.launch()
  assertSeparate("home-camera","new-project")
  let home=XCTAttachment(screenshot:app.screenshot());home.name="Separate native home buttons";home.lifetime = .keepAlways;add(home)
  app.buttons["Listenansicht"].tap()
  let collection=app.buttons.matching(NSPredicate(format:"label CONTAINS %@","1 Seite,")).firstMatch
  XCTAssertTrue(collection.waitForExistence(timeout:5));collection.tap()
  assertSeparate("collection-camera","collection-more")
  let shot=XCTAttachment(screenshot:app.screenshot());shot.name="Separate native collection buttons";shot.lifetime = .keepAlways;add(shot)
  item("collection-camera").tap();XCTAssertTrue(item("shutter").waitForExistence(timeout:5));app.navigationBars.buttons["Fertig"].tap()
  item("collection-more").tap();XCTAssertTrue(app.buttons["Fotos importieren"].waitForExistence(timeout:5));XCTAssertTrue(app.buttons["Dateien"].exists);XCTAssertTrue(app.buttons["Teilen"].exists)
 }
}
