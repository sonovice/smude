import Foundation
import UIKit
import PhotosUI
import ImageIO
import UniformTypeIdentifiers

func SmudeL(_ key: String) -> String { NSLocalizedString(key, comment: "") }

private typealias ServiceEvent = @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?) -> Void

/// Each import owns a private staging directory. Cancellation is shared with
/// serial image work; a callback can never recreate files after teardown.
private final class MediaOperation {
    enum Kind { case importing, sharing }
    let requestID: UInt64
    let kind: Kind
    var directory: URL?
    var paths: [String] = [] // main thread only
    var loading = false // picker has already handed over its selection
    private let lock = NSLock()
    private var cancelled = false
    init(_ requestID: UInt64, kind: Kind) { self.requestID = requestID; self.kind = kind }
    var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return cancelled }
    func cancel() { lock.lock(); cancelled = true; lock.unlock() }
}

/// OS pickers and sharing only. Projects, selection, processing and all app
/// screens belong to the Rust UI. Events are always emitted on the main thread.
private final class ServicesView: UIView, PHPickerViewControllerDelegate,
    UIDocumentPickerDelegate, UIAdaptivePresentationControllerDelegate {
    private var callback: ServiceEvent?
    private var userdata: UnsafeMutableRawPointer?
    private let work = DispatchQueue(label: "smude.media.import", qos: .userInitiated)
    private var root: URL?
    private var readySent = false
    private var operation: MediaOperation?
    private weak var presented: UIViewController?
    private var feedbackObservers: [NSObjectProtocol] = []
    private var dateRequests: [Double] = []

    init(userdata: UnsafeMutableRawPointer?, callback: @escaping ServiceEvent) {
        self.userdata = userdata; self.callback = callback
        super.init(frame: .zero)
        isUserInteractionEnabled = false
        feedbackObservers.append(NotificationCenter.default.addObserver(forName:Notification.Name("smude.capture"),object:nil,queue:.main){[weak self] note in
            if let payload=note.userInfo as? [String:Any]{self?.emit(payload)}
        })
        for name in [UIAccessibility.voiceOverStatusDidChangeNotification,
                     UIAccessibility.switchControlStatusDidChangeNotification,
                     UIApplication.didBecomeActiveNotification,
                     NSLocale.currentLocaleDidChangeNotification,
                     .NSSystemTimeZoneDidChange,
                     UIApplication.didEnterBackgroundNotification] {
            feedbackObservers.append(NotificationCenter.default.addObserver(forName: name,
                object: nil, queue: .main) { [weak self] _ in self?.feedbackPreferences(); self?.localization() })
        }
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) is not used") }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        guard window != nil else { return }
        // NSH installs the component event listener in the same mount batch.
        DispatchQueue.main.async { [weak self] in self?.sendReady() }
    }

    private func emit(_ payload: [String: Any]) {
        dispatchPrecondition(condition: .onQueue(.main))
        guard let callback,
              let bytes = try? JSONSerialization.data(withJSONObject: payload),
              let json = String(data: bytes, encoding: .utf8) else { return }
        json.withCString { callback(userdata, $0) }
    }

    private func sendReady(force: Bool = false) {
        guard callback != nil, !readySent || force else { return }
        do {
            let support = try FileManager.default.url(for: .applicationSupportDirectory,
                in: .userDomainMask, appropriateFor: nil, create: true)
            let directory = support.appendingPathComponent("SmudeProjects", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            root = directory.resolvingSymlinksInPath()
            readySent = true
            localization()
            emit(["event":"ready", "root":root!.path])
        } catch { emit(["event":"ready", "error":SmudeL("Could not open collection storage") + ": \(error.localizedDescription)"]) }
    }

    func command(_ name: String, arguments: [String: Any]) {
        dispatchPrecondition(condition: .onQueue(.main))
        guard callback != nil else { return }
        if name == "localization" {
            dateRequests = (arguments["dates"] as? [NSNumber] ?? []).map { $0.doubleValue }
            localization(); return
        }
        if name == "open_settings" {
            if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
            return
        }
        if name == "feedback_preferences" { feedbackPreferences(); return }
        if name == "ready" { sendReady(force: true); return }
        guard let requestID = (arguments["request_id"] as? NSNumber)?.uint64Value else { return }
        let event = name == "share" ? "shared" : "imported"
        guard operation == nil else {
            emit(["event":event, "request_id":requestID, "error":SmudeL("Close the current dialog first.")])
            return
        }
        guard let root else {
            emit(["event":event, "request_id":requestID, "error":SmudeL("Collection storage is not ready yet.")])
            return
        }
        guard let presenter = presenter(), presenter.viewIfLoaded?.window != nil,
              !presenter.isBeingDismissed else {
            emit(["event":event, "request_id":requestID, "error":SmudeL("Cannot open the dialog right now.")])
            return
        }
        switch name {
        case "import", "import_files":
            let op = MediaOperation(requestID, kind: .importing)
            let folder = root.appendingPathComponent("imports", isDirectory: true)
                .appendingPathComponent(UUID().uuidString, isDirectory: true)
            do { try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true) }
            catch {
                emit(["event":"imported", "request_id":requestID, "error":SmudeL("Could not prepare the import") + ": \(error.localizedDescription)"])
                return
            }
            op.directory = folder; operation = op
            presenter.view.endEditing(true)
            if name == "import" {
                var config = PHPickerConfiguration()
                config.filter = .images
                config.selectionLimit = 0
                config.selection = .ordered
                config.preferredAssetRepresentationMode = .current
                let picker = PHPickerViewController(configuration: config)
                picker.delegate = self
                present(picker, from: presenter)
            } else {
                let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.image], asCopy: true)
                picker.delegate = self
                picker.allowsMultipleSelection = true
                present(picker, from: presenter)
            }
        case "share":
            do {
                let paths = arguments["paths"] as? [String] ?? []
                guard !paths.isEmpty else { throw mediaFailure("Keine Exportdateien vorhanden.") }
                let urls = try paths.map { try checkedExport($0, root: root) }
                let op = MediaOperation(requestID, kind: .sharing); operation = op
                let controller = UIActivityViewController(activityItems: urls, applicationActivities: nil)
                controller.completionWithItemsHandler = { [weak self, weak op] _, completed, _, error in
                    DispatchQueue.main.async {
                        guard let self, let op, self.operation === op else { return }
                        self.finishShare(op, cancelled: !completed, error: error?.localizedDescription)
                    }
                }
                if let popover = controller.popoverPresentationController {
                    popover.sourceView = presenter.view
                    popover.sourceRect = CGRect(x: presenter.view.bounds.midX,
                        y: max(1, presenter.view.bounds.maxY - presenter.view.safeAreaInsets.bottom - 24), width: 1, height: 1)
                    popover.permittedArrowDirections = []
                }
                presenter.view.endEditing(true)
                present(controller, from: presenter)
            } catch { emit(["event":"shared", "request_id":requestID, "error":error.localizedDescription]) }
        default:
            emit(["event":event, "request_id":requestID, "error":"Unbekannte Medienaktion."])
        }
    }

    private func localization() {
        let formatter = DateFormatter()
        formatter.locale = .autoupdatingCurrent
        formatter.timeZone = .autoupdatingCurrent
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        var dates: [String:String] = [:]
        for ms in dateRequests where ms > 0 {
            dates[String(Int64(ms))] = formatter.string(from: Date(timeIntervalSince1970: ms / 1000))
        }
        emit(["event":"localization", "language":Bundle.main.preferredLocalizations.first ?? "en", "dates":dates])
    }

    private func presenter() -> UIViewController? {
        guard let root = window?.rootViewController else { return nil }
        func top(_ current: UIViewController) -> UIViewController {
            if let next = current.presentedViewController, !next.isBeingDismissed { return top(next) }
            if let nav = current as? UINavigationController, let next = nav.visibleViewController { return top(next) }
            if let tabs = current as? UITabBarController, let next = tabs.selectedViewController { return top(next) }
            return current
        }
        return top(root)
    }
    private func present(_ controller: UIViewController, from presenter: UIViewController) {
        presented = controller
        controller.presentationController?.delegate = self
        presenter.present(controller, animated: true) { [weak self, weak controller] in
            controller?.presentationController?.delegate = self
        }
    }

    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        guard let op = operation, op.kind == .importing else { picker.dismiss(animated: true); return }
        op.loading = true
        picker.dismiss(animated: true) { [weak self] in
            guard let self, self.operation === op else { return }
            self.presented = nil
            if results.isEmpty { self.finishImport(op, cancelled: true); return }
            self.loadPhoto(results, index: 0, operation: op)
        }
    }

    private func loadPhoto(_ results: [PHPickerResult], index: Int, operation op: MediaOperation) {
        guard operation === op, !op.isCancelled, let folder = op.directory else { return }
        guard index < results.count else { finishImport(op); return }
        let queue = work
        // The provider's URL expires when its completion returns. Copy it before
        // queuing the bounded, sequential full-resolution normalization work.
        results[index].itemProvider.loadFileRepresentation(forTypeIdentifier: UTType.image.identifier) {
            [weak self, op] source, error in
            guard !op.isCancelled else { return }
            let copied = folder.appendingPathComponent("source-\(index)")
            do {
                if let error { throw error }
                guard let source else { throw mediaFailure(SmudeL("Could not load the selected photo.")) }
                try FileManager.default.copyItem(at: source, to: copied)
            } catch {
                DispatchQueue.main.async { self?.finishImport(op, error: error.localizedDescription) }
                return
            }
            queue.async { [weak self, op] in
                guard !op.isCancelled else { try? FileManager.default.removeItem(at: copied); return }
                do {
                    let destination = folder.appendingPathComponent(String(format: "%04d.png", index + 1))
                    try autoreleasepool { try normalizeImage(source: copied, destination: destination) }
                    try? FileManager.default.removeItem(at: copied)
                    DispatchQueue.main.async {
                        guard let self, self.operation === op, !op.isCancelled else { return }
                        op.paths.append(destination.path)
                        self.loadPhoto(results, index: index + 1, operation: op)
                    }
                } catch {
                    DispatchQueue.main.async { self?.finishImport(op, error: "Foto \(index + 1): \(error.localizedDescription)") }
                }
            }
        }
    }

    func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        guard let op = operation, op.kind == .importing, let folder = op.directory else { return }
        op.loading = true
        presented = nil
        if urls.isEmpty { finishImport(op, cancelled: true); return }
        work.async { [weak self, op] in
            var paths: [String] = []
            do {
                for (index, source) in urls.enumerated() {
                    guard !op.isCancelled else { return }
                    let destination = folder.appendingPathComponent(String(format: "%04d.png", index + 1))
                    try autoreleasepool {
                        let access = source.startAccessingSecurityScopedResource()
                        defer { if access { source.stopAccessingSecurityScopedResource() } }
                        let coordinator = NSFileCoordinator()
                        var coordinationError: NSError?
                        var decodeError: Error?
                        coordinator.coordinate(readingItemAt: source, options: [], error: &coordinationError) { coordinated in
                            do { try normalizeImage(source: coordinated, destination: destination) }
                            catch { decodeError = error }
                        }
                        if let error = coordinationError ?? decodeError as NSError? { throw error }
                    }
                    paths.append(destination.path)
                }
                DispatchQueue.main.async {
                    guard let self, self.operation === op, !op.isCancelled else { return }
                    op.paths = paths; self.finishImport(op)
                }
            } catch { DispatchQueue.main.async { self?.finishImport(op, error: error.localizedDescription) } }
        }
    }
    func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
        if let op = operation, op.kind == .importing { finishImport(op, cancelled: true) }
    }
    func presentationControllerDidDismiss(_ presentationController: UIPresentationController) {
        guard let op = operation else { return }
        if op.kind == .sharing { finishShare(op, cancelled: true) }
        else if !op.loading { finishImport(op, cancelled: true) }
    }

    private func finishImport(_ op: MediaOperation, cancelled: Bool = false, error: String? = nil) {
        guard operation === op else { return }
        operation = nil; presented = nil
        if cancelled || error != nil {
            op.cancel(); clean(op)
            var data: [String: Any] = ["event":"imported", "request_id":op.requestID, "cancelled":cancelled]
            if let error { data["error"] = error }; emit(data)
        } else { emit(["event":"imported", "request_id":op.requestID, "paths":op.paths]) }
    }
    private func finishShare(_ op: MediaOperation, cancelled: Bool, error: String? = nil) {
        guard operation === op else { return }
        operation = nil; presented = nil
        var data: [String: Any] = ["event":"shared", "request_id":op.requestID, "cancelled":cancelled]
        if let error { data["error"] = error }; emit(data)
    }
    private func clean(_ op: MediaOperation) {
        guard let directory = op.directory else { return }
        work.async { try? FileManager.default.removeItem(at: directory) }
    }
    func shutdown() {
        dispatchPrecondition(condition: .onQueue(.main))
        feedbackObservers.forEach(NotificationCenter.default.removeObserver)
        feedbackObservers.removeAll()
        callback = nil; userdata = nil
        if let op = operation { op.cancel(); clean(op) }
        operation = nil
        presented?.dismiss(animated: false); presented = nil
    }

    private func feedbackPreferences() {
        emit(["event":"feedback_preferences", "info_ms":UIAccessibility.isVoiceOverRunning ? 8000 : 3000,
              "action_ms":6000,
              "persistent_action":UIAccessibility.isVoiceOverRunning || UIAccessibility.isSwitchControlRunning,
              "foreground":UIApplication.shared.applicationState == .active])
    }
}

private func mediaFailure(_ message: String) -> NSError {
    NSError(domain: "SmudeMedia", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
}

/// Full-resolution EXIF normalization, including HEIC and mirrored orientations.
/// PNG avoids another lossy encoding before dewarping and has orientation 1.
private func normalizeImage(source: URL, destination: URL) throws {
    guard let input = CGImageSourceCreateWithURL(source as CFURL, nil),
          let properties = CGImageSourceCopyPropertiesAtIndex(input, 0, nil) as? [CFString: Any],
          let width = properties[kCGImagePropertyPixelWidth] as? Int,
          let height = properties[kCGImagePropertyPixelHeight] as? Int,
          width > 0, height > 0, max(width, height) <= 12000, width * height <= 64_000_000 else {
        throw mediaFailure(SmudeL("The image is unreadable or exceeds 64 megapixels or 12,000 pixels per edge."))
    }
    let options: [CFString: Any] = [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: max(width, height),
        kCGImageSourceShouldCacheImmediately: true
    ]
    guard let image = CGImageSourceCreateThumbnailAtIndex(input, 0, options as CFDictionary) else {
        throw mediaFailure(SmudeL("Could not decode the image."))
    }
    let orientation = properties[kCGImagePropertyOrientation] as? Int ?? 1
    let swapped = (5...8).contains(orientation)
    guard image.width == (swapped ? height : width), image.height == (swapped ? width : height) else {
        throw mediaFailure(SmudeL("Could not load the full-resolution image."))
    }
    let temporary = destination.deletingLastPathComponent().appendingPathComponent(UUID().uuidString + ".part")
    defer { try? FileManager.default.removeItem(at: temporary) }
    guard let output = CGImageDestinationCreateWithURL(temporary as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw mediaFailure(SmudeL("Could not create the imported file."))
    }
    CGImageDestinationAddImage(output, image, [kCGImagePropertyOrientation:1] as CFDictionary)
    guard CGImageDestinationFinalize(output) else { throw mediaFailure(SmudeL("Could not save the imported file.")) }
    try FileManager.default.moveItem(at: temporary, to: destination)
}

private func checkedExport(_ path: String, root: URL) throws -> URL {
    let url = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
    guard url.path.hasPrefix(root.path + "/") else { throw mediaFailure(SmudeL("This export does not belong to this collection.")) }
    let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
    guard values.isRegularFile == true, (values.fileSize ?? 0) > 0,
          ["pdf", "jpg", "jpeg"].contains(url.pathExtension.lowercased()) else {
        throw mediaFailure(SmudeL("The export is missing or not ready yet."))
    }
    return url
}

@_cdecl("smude_services_create")
func createServices(_ userdata: UnsafeMutableRawPointer?, _ callback: @escaping @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?) -> Void) -> UnsafeMutableRawPointer {
    Unmanaged.passRetained(ServicesView(userdata: userdata, callback: callback)).toOpaque()
}
@_cdecl("smude_services_command")
func commandServices(_ view: UnsafeMutableRawPointer, _ name: UnsafePointer<CChar>, _ json: UnsafePointer<CChar>) {
    guard let bytes = String(cString: json).data(using: .utf8),
          let arguments = (try? JSONSerialization.jsonObject(with: bytes)) as? [String: Any] else { return }
    Unmanaged<ServicesView>.fromOpaque(view).takeUnretainedValue().command(String(cString: name), arguments: arguments)
}
@_cdecl("smude_services_shutdown")
func shutdownServices(_ view: UnsafeMutableRawPointer) { Unmanaged<ServicesView>.fromOpaque(view).takeUnretainedValue().shutdown() }
