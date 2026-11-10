import XCTest

final class SmudeMobileUITests: XCTestCase {
    func testSwitchModelsForAnExistingPhoto() throws {
        continueAfterFailure = false
        let app = XCUIApplication(bundleIdentifier: "com.gridnotelabs.smude.mobile")
        app.launchEnvironment["SMUDE_AUTORUN"] = "0"
        app.launch()
        XCTAssertTrue(app.switches["GCS"].waitForExistence(timeout: 15))
        app.switches["GCS"].tap()
        XCTAssertTrue(app.buttons["Testbild verwenden"].waitForExistence(timeout: 10))
        app.buttons["Testbild verwenden"].tap()
        let gcs = app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH %@ AND label ENDSWITH %@", "1 Seite ·", " · GCS")).firstMatch
        XCTAssertTrue(gcs.waitForExistence(timeout: 90))
        app.switches["Mesh"].tap()
        let mesh = app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH %@ AND label ENDSWITH %@", "1 Seite ·", " · Mesh")).firstMatch
        XCTAssertTrue(mesh.waitForExistence(timeout: 90))
        XCTAssertTrue(app.buttons["Exportieren ↗"].exists)
        let capture = XCTAttachment(screenshot: app.screenshot())
        capture.name = "Existing photo processed with direction mesh"
        capture.lifetime = .keepAlways; add(capture)
        app.switches["GCS"].tap()
        XCTAssertTrue(gcs.waitForExistence(timeout: 3), "Cached GCS result should switch immediately")
        app.switches["Mesh"].tap()
        XCTAssertTrue(mesh.waitForExistence(timeout: 3), "Cached mesh result should switch immediately")
    }

    func testModelBackendProbe() throws {
        continueAfterFailure = false
        let app = XCUIApplication(bundleIdentifier: "com.gridnotelabs.smude.mobile")
        app.launchEnvironment["SMUDE_AUTORUN"] = "1"
        app.launchEnvironment["SMUDE_BENCH_MODELS"] = "1"
        app.launchEnvironment["SMUDE_COREML_UNITS"] = "gpu"
        app.launch()
        let completed = app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH %@", "1 Seite ·")).firstMatch
        XCTAssertTrue(completed.waitForExistence(timeout: 90))
    }

    func testCameraInterface() throws {
        continueAfterFailure = false
        let app = XCUIApplication(bundleIdentifier: "com.gridnotelabs.smude.mobile")
        app.launchEnvironment["SMUDE_AUTORUN"] = "0"
        app.launch()
        XCTAssertTrue(app.buttons["Foto aufnehmen"].waitForExistence(timeout: 15))
        app.buttons["Foto aufnehmen"].tap()
        let system = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        let permission = system.buttons.matching(NSPredicate(format: "label IN %@", ["Allow", "Erlauben", "OK"])).firstMatch
        if permission.waitForExistence(timeout: 3) { permission.tap() }
        XCTAssertTrue(app.buttons["PhotoCapture"].waitForExistence(timeout: 10))
        let cancel = app.buttons["DismissImagePickerButton"]
        XCTAssertTrue(cancel.exists)
        let hierarchy = XCTAttachment(string: app.debugDescription)
        hierarchy.name = "Native camera hierarchy"; hierarchy.lifetime = .keepAlways; add(hierarchy)
        cancel.tap()
        XCTAssertTrue(app.buttons["Foto aufnehmen"].waitForExistence(timeout: 10))
    }

    func testCameraCaptureImportsFullResolutionPhoto() throws {
        continueAfterFailure = false
        let app = XCUIApplication(bundleIdentifier: "com.gridnotelabs.smude.mobile")
        app.launchEnvironment["SMUDE_AUTORUN"] = "0"
        app.launch()
        XCTAssertTrue(app.buttons["Foto aufnehmen"].waitForExistence(timeout: 15))
        app.buttons["Foto aufnehmen"].tap()
        XCTAssertTrue(app.buttons["PhotoCapture"].waitForExistence(timeout: 10))
        app.buttons["PhotoCapture"].tap()
        let use = app.buttons.matching(NSPredicate(format: "label IN %@", ["Use Photo", "Foto verwenden"])).firstMatch
        XCTAssertTrue(use.waitForExistence(timeout: 10))
        use.tap()
        let size = app.staticTexts.matching(NSPredicate(format: "label MATCHES %@", "[0-9]+ × [0-9]+")).firstMatch
        XCTAssertTrue(size.waitForExistence(timeout: 30))
        let dimensions = size.label.components(separatedBy: " × ").compactMap(Int.init)
        XCTAssertEqual(dimensions.count, 2)
        XCTAssertGreaterThan(dimensions.min() ?? 0, 1000)
    }

    func testFullResolutionProcessingAndNativeExportDialog() throws {
        continueAfterFailure = false
        let app = XCUIApplication(bundleIdentifier: "com.gridnotelabs.smude.mobile")
        app.launchEnvironment["SMUDE_AUTORUN"] = "1"
        app.launchEnvironment["SMUDE_VERIFY_EXPORT"] = "1"
        app.launch()
        XCTAssertTrue(app.buttons["Testbild verwenden"].waitForExistence(timeout: 15))
        let completed = app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH %@", "1 Seite ·")).firstMatch
        XCTAssertTrue(completed.waitForExistence(timeout: 45))
        let export = app.buttons["Exportieren ↗"]
        // WebKit's accessibility hit-test can briefly lack activation points
        // while the result scrolls into view. Finish the scroll with a native
        // gesture, then use the located button's frame rather than isHittable.
        app.swipeUp()
        XCTAssertTrue(export.waitForExistence(timeout: 10))
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "Processed score on iPhone"
        attachment.lifetime = .keepAlways
        add(attachment)
        export.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        let save = app.buttons.matching(NSPredicate(format: "label IN %@", ["Sichern", "Save"])).firstMatch
        XCTAssertTrue(save.waitForExistence(timeout: 10))
        // QA mode checks the actual provider file against the source PNG
        // before the app reports export success; it also records the hash.
        save.tap()
        let replace = app.buttons.matching(NSPredicate(format: "label IN %@", ["Replace", "Ersetzen"])).firstMatch
        if replace.waitForExistence(timeout: 2) { replace.tap() }
        XCTAssertTrue(app.staticTexts["Bild exportiert"].waitForExistence(timeout: 15))

    }
}
