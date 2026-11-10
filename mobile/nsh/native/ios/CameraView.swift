import UIKit

private func SmudeL(_ key: String) -> String { NSLocalizedString(key, comment: "") }
import AVFoundation
import CoreImage

@_silgen_name("smude_detect_bgra")
private func detectPage(_ bytes: UnsafePointer<UInt8>, _ length: UInt, _ width: UInt, _ height: UInt, _ stride: UInt, _ clockwiseTurns: UInt32) -> UnsafeMutablePointer<CChar>?
@_silgen_name("smude_detection_free")
private func freeDetection(_ json: UnsafeMutablePointer<CChar>)
private typealias Event = @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?) -> Void

// BEGIN CAMERA STABILITY
/// Pure time/geometry latch, shared in behavior with Android's DetectionHold.
/// Only missed/invalid measurements get a short hold; errors and scene changes
/// invalidate immediately. Source frame time, not result arrival, bounds age.
struct CameraDetectionGate {
    enum Decision { case show, hold, clear }
    static let sourceTTL = 1.2
    static let invalidHold = 0.6
    private(set) var ready = false
    private(set) var holding = false
    private var stableFrames = 0
    private var lastValidAt = -Double.infinity
    private var lastFrameAt = -Double.infinity
    private var previous: [[[Double]]] = []
    private var lastGoodScene: [Double] = []

    mutating func reset() { self = CameraDetectionGate() }
    func canCapture(now: Double, newerThan:Double = -Double.infinity) -> Bool {
        ready && lastValidAt >= newerThan && now >= lastValidAt && now - lastValidAt <= (holding ? Self.invalidHold : Self.sourceTTL)
    }
    func expired(now: Double) -> Bool {
        if ready { return !canCapture(now: now) }
        return lastFrameAt.isFinite && now - lastFrameAt > Self.sourceTTL
    }
    mutating func accept(valid: Bool, polygons: [[[Double]]], scene: [Double], frameTime: Double,
                         now: Double, hardFailure: Bool = false) -> Decision {
        guard !hardFailure, frameTime.isFinite, now.isFinite,
              now >= frameTime, now - frameTime <= Self.sourceTTL else {
            reset(); return .clear
        }
        lastFrameAt = frameTime
        let difference = Self.sceneDifference(lastGoodScene, scene)
        // An invalid contour/count jump can be the classifier flickering. It
        // isn't reliable motion evidence; the independent scene signal is.
        let moved = ready && (difference > 0.08 || (valid && Self.contourDistance(previous, polygons) > 0.06))
        if !valid && ready && !moved && now >= lastValidAt && now - lastValidAt <= Self.invalidHold
            && difference.isFinite && difference <= 0.08 {
            holding = true
            return .hold // A miss never advances lastValidAt.
        }
        holding = false
        if valid {
            stableFrames = !moved && (ready || Self.contourDistance(previous, polygons) < 0.025) ? stableFrames + 1 : 1
            previous = polygons
            lastValidAt = frameTime
            ready = stableFrames >= 2
            if ready { lastGoodScene = scene }
        } else {
            stableFrames = 0; previous = []; lastGoodScene = []; ready = false
        }
        return .show
    }
    static func sceneDifference(_ before: [Double], _ after: [Double]) -> Double {
        guard before.count == 64, after.count == 64,
              before.allSatisfy({ $0.isFinite }), after.allSatisfy({ $0.isFinite }) else { return .infinity }
        let exposure = zip(before, after).reduce(0) { $0 + $1.1 - $1.0 } / 64
        return zip(before, after).reduce(0) { $0 + abs($1.1 - $1.0 - exposure) } / 64
    }
    static func contourDistance(_ before: [[[Double]]], _ after: [[[Double]]]) -> Double {
        guard before.count == after.count, !before.isEmpty else { return .infinity }
        var distance = 0.0; var count = 0
        for (a,b) in zip(before, after) {
            guard a.count == b.count, !a.isEmpty else { return .infinity }
            for (p,q) in zip(a,b) {
                guard p.count == 2, q.count == 2, p.allSatisfy({$0.isFinite}), q.allSatisfy({$0.isFinite}) else { return .infinity }
                distance += hypot(p[0]-q[0],p[1]-q[1]); count += 1
            }
        }
        return count > 0 ? distance / Double(count) : .infinity
    }
}
// END CAMERA STABILITY

// BEGIN CAMERA ORIENTATION
/// Device pose and interface orientation differ when rotation lock is enabled.
struct CameraOrientation {
    private(set) var device = 0
    mutating func observe(_ value: Int) { if (1...4).contains(value) { device = value } }
    func captureInterface(_ interface: Int) -> Int {
        // UIKit's landscape names are reversed, but its raw values already
        // match UIDeviceOrientation and AVCaptureVideoOrientation (3/4).
        (1...4).contains(device) ? device : interface
    }
    static func capturePlan(_ interface: Int, turns: UInt32) -> (angle: Int, rotation: Int) {
        (angle(interface), Int(turns % 4) * 90)
    }
    static func angle(_ interface: Int) -> Int {
        switch interface { case 2: return 270; case 3: return 0; case 4: return 180; default: return 90 }
    }
    func analysisTurns(_ interface: Int) -> UInt {
        UInt((Self.angle(captureInterface(interface)) - Self.angle(interface) + 360) % 360 / 90)
    }
}
// END CAMERA ORIENTATION

// Platform adapter only: preview, frame delivery and full-resolution capture.
// All visible controls and the live contour painter belong to the Rust UI.
private final class CameraView: UIView, AVCaptureVideoDataOutputSampleBufferDelegate, AVCapturePhotoCaptureDelegate {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    private var preview: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "smude.camera.session", qos: .userInitiated)
    private let frames = DispatchQueue(label: "smude.camera.frames", qos: .userInitiated)
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let video = AVCaptureVideoDataOutput()
    private let photos = AVCapturePhotoOutput()
    private var configured = false // session queue only
    private var cameraDevice: AVCaptureDevice? // session queue only
    private var torchObservations: [NSKeyValueObservation] = []
    private var lastFrame: CFTimeInterval = 0 // frame queue only
    private var callback: Event?
    private var userdata: UnsafeMutableRawPointer?
    private var active = true
    private var generation: UInt = 0
    private var gate = CameraDetectionGate()
    private var ready:Bool {gate.canCapture(now:CACurrentMediaTime(),newerThan:captureAfter) && !capturing && pending.count < 3}
    private var capturing = false
    private var lastGoodStatus: [String: Any]?
    private var detectedPages = 1
    private struct PendingPhoto {
        let request: UInt64
        let path: String
        var writing = false
        var stored = false
        var sensorFinished = false
    }
    private var pending: [Int64:PendingPhoto] = [:]
    // Capture lifetime is independent of the mounted preview/context.
    private static var owners: [Int64:CameraView] = [:]
    private var acquiring: Int64?
    private var captureAfter = -Double.infinity
    private static let saves = DispatchQueue(label:"smude.camera.saves",qos:.userInitiated)
    private var expiry: Timer?
    private var observers: [NSObjectProtocol] = []
    private var orientation: UIInterfaceOrientation = .unknown
    private var frameOrientation = CameraOrientation()
    private var inferredTurns: UInt32?

    init(userdata: UnsafeMutableRawPointer?, callback: @escaping Event) {
        self.userdata = userdata; self.callback = callback
        super.init(frame: .zero)
        backgroundColor = .black
        preview.session = session; preview.videoGravity = .resizeAspectFill
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        frameOrientation.observe(UIDevice.current.orientation.rawValue)
        observers.append(NotificationCenter.default.addObserver(forName: UIDevice.orientationDidChangeNotification, object: nil, queue: .main) { [weak self] _ in
            guard let self else { return }
            let before = self.frameOrientation.device
            self.frameOrientation.observe(UIDevice.current.orientation.rawValue)
            if before != self.frameOrientation.device { self.updateOrientation(force: true) }
        })
        observers.append(NotificationCenter.default.addObserver(forName: UIApplication.willResignActiveNotification, object: nil, queue: .main) { [weak self] _ in self?.pause() })
        observers.append(NotificationCenter.default.addObserver(forName: UIApplication.didBecomeActiveNotification, object: nil, queue: .main) { [weak self] _ in self?.resume() })
        expiry = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            guard let self, self.gate.expired(now: CACurrentMediaTime()) else { return }
            self.invalidate(SmudeL("Hold the page steady in the frame"))
        }
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) is not used") }
    override func didMoveToWindow() { super.didMoveToWindow(); window == nil ? pause() : resume() }
    override func layoutSubviews() {
        super.layoutSubviews()
        updateOrientation()
    }
    private func updateOrientation(force: Bool = false) {
        let current = window?.windowScene?.interfaceOrientation ?? .portrait
        guard force || current != orientation else { return }
        orientation = current
        inferredTurns = nil
        invalidate(SmudeL("Place the page in the frame"))
        let angle = CGFloat(CameraOrientation.angle(current.rawValue))
        let capture = UIInterfaceOrientation(rawValue: frameOrientation.captureInterface(current.rawValue)) ?? current
        setOrientation(preview.connection, angle: angle, current: current)
        queue.async { [weak self] in
            guard let self else { return }
            self.setOrientation(self.video.connection(with: .video), angle: angle, current: current)
            self.setOrientation(self.photos.connection(with: .video), angle: CGFloat(CameraOrientation.angle(capture.rawValue)), current: capture)
        }
    }
    private func setOrientation(_ connection: AVCaptureConnection?, angle: CGFloat, current: UIInterfaceOrientation) {
        guard let connection else { return }
        if #available(iOS 17.0, *), connection.isVideoRotationAngleSupported(angle) { connection.videoRotationAngle = angle }
        else if connection.isVideoOrientationSupported {
            connection.videoOrientation = AVCaptureVideoOrientation(rawValue: current.rawValue) ?? .portrait
        }
        if connection.isVideoMirroringSupported { connection.automaticallyAdjustsVideoMirroring = false; connection.isVideoMirrored = false }
    }
    private func emit(_ payload: [String: Any]) {
        dispatchPrecondition(condition: .onQueue(.main))
        guard let callback, let data = try? JSONSerialization.data(withJSONObject: payload), let json = String(data: data, encoding: .utf8) else { return }
        json.withCString { callback(userdata, $0) }
    }
    private func invalidate(_ message: String, reason: String = "") {
        generation &+= 1; gate.reset(); lastGoodStatus = nil
        emit(["event":"status", "ready":false, "capturing":capturing, "message":message, "reason":reason, "width":1, "height":1, "polygons":[]])
    }
    private func resume() {
        guard active, window != nil, callback != nil, UIApplication.shared.applicationState == .active else { return }
        frameOrientation.observe(UIDevice.current.orientation.rawValue)
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: start()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] allowed in DispatchQueue.main.async { allowed ? self?.resume() : self?.invalidate(SmudeL("Allow camera access in Settings"), reason: "permission") } }
        default: invalidate(SmudeL("Allow camera access in Settings"), reason: "permission")
        }
    }
    private func start() {
        queue.async { [weak self] in
            guard let self else { return }
            if !self.configured {
                guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back), let input = try? AVCaptureDeviceInput(device: device) else {
                    DispatchQueue.main.async { self.invalidate(SmudeL("No camera available"), reason: "unavailable") }; return
                }
                self.session.beginConfiguration()
                // A 4:3 preview and photo use the same sensor field of view.
                self.session.sessionPreset = .photo
                guard self.session.canAddInput(input), self.session.canAddOutput(self.video), self.session.canAddOutput(self.photos) else {
                    self.session.commitConfiguration(); DispatchQueue.main.async { self.invalidate(SmudeL("Could not start the camera"), reason: "unavailable") }; return
                }
                self.session.addInput(input)
                self.cameraDevice = device
                for keyPath in [\AVCaptureDevice.isTorchAvailable, \AVCaptureDevice.isTorchActive] {
                    self.torchObservations.append(device.observe(keyPath, options: [.new]) { [weak self] _, _ in
                        self?.queue.async { [weak self] in self?.publishTorch() }
                    })
                }
                self.video.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
                self.video.alwaysDiscardsLateVideoFrames = true
                self.video.setSampleBufferDelegate(self, queue: self.frames)
                self.session.addOutput(self.video); self.session.addOutput(self.photos)
                self.photos.maxPhotoQualityPrioritization = .quality
                if #available(iOS 17.0, *), self.photos.isResponsiveCaptureSupported { self.photos.isResponsiveCaptureEnabled = true }
                self.session.commitConfiguration()
                if #available(iOS 16.0, *) {
                    if let maximum = device.activeFormat.supportedMaxPhotoDimensions.max(by: {
                        Int64($0.width) * Int64($0.height) < Int64($1.width) * Int64($1.height)
                    }) { self.photos.maxPhotoDimensions = maximum }
                } else {
                    self.photos.isHighResolutionCaptureEnabled = true
                }
                self.configured = true
            }
            self.session.startRunning()
            self.configureTorch(false)
            DispatchQueue.main.async { [weak self] in self?.orientation = .unknown; self?.setNeedsLayout() }
        }
    }
    private func pause() {
        inferredTurns = nil
        invalidate(SmudeL("Camera paused"))
        let stopSession = pending.isEmpty
        let stopTorch = !capturing
        queue.async { [weak self] in
            if stopTorch { self?.configureTorch(false) }
            if stopSession { self?.session.stopRunning() }
            self?.publishTorch()
        }
    }
    func shutdown() {
        callback = nil; userdata = nil; active = false; generation &+= 1
        expiry?.invalidate(); expiry = nil
        observers.forEach(NotificationCenter.default.removeObserver); observers = []
        UIDevice.current.endGeneratingDeviceOrientationNotifications()
        let stopSession = pending.isEmpty
        let stopTorch = !capturing
        queue.async { [self] in
            torchObservations = []
            if stopTorch { configureTorch(false) }
            video.setSampleBufferDelegate(nil, queue:nil)
            if stopSession { session.stopRunning() }
        }
    }
    private func publishTorch() {
        let available = session.isRunning && cameraDevice?.hasTorch == true && cameraDevice?.isTorchAvailable == true
        let on = available && cameraDevice?.isTorchActive == true
        DispatchQueue.main.async { [weak self] in self?.emit(["event":"torch", "available":available, "on":on]) }
    }
    private func configureTorch(_ on: Bool) {
        if let device = cameraDevice, device.hasTorch {
            do {
                try device.lockForConfiguration()
                defer { device.unlockForConfiguration() }
                if on && session.isRunning && device.isTorchAvailable {
                    try device.setTorchModeOn(level: AVCaptureDevice.maxAvailableTorchLevel)
                } else { device.torchMode = .off }
            } catch { /* Hardware state is reported below; never show a false on state. */ }
        }
        publishTorch()
    }
    func command(_ name: String, arguments: [String:Any] = [:]) {
        switch name {
        case "activate": active = true; resume()
        case "deactivate": active = false; pause()
        case "torch_on":
            guard active, window != nil, UIApplication.shared.applicationState == .active else { return }
            queue.async { [weak self] in self?.configureTorch(true) }
        case "torch_off": queue.async { [weak self] in self?.configureTorch(false) }
        case "capture":
            guard let request = (arguments["request_id"] as? NSNumber)?.uint64Value,
                  let path = arguments["path"] as? String else { return }
            guard ready, !capturing, pending.count < 3, (arguments["single_only"] as? Bool != true || detectedPages == 1) else {
                publishCapture(["event":"capture_failed","request_id":request,"reason":"not_ready"]);emit(["event":"capture_failed","request_id":request]);return
            }
            let plan = CameraOrientation.capturePlan(orientation.rawValue,
                turns: inferredTurns ?? UInt32(frameOrientation.analysisTurns(orientation.rawValue)))
            let settings = AVCapturePhotoSettings(format:[AVVideoCodecKey:AVVideoCodecType.jpeg])
            settings.flashMode = .off
            settings.photoQualityPrioritization = .quality
            if #available(iOS 16.0, *) { settings.maxPhotoDimensions = photos.maxPhotoDimensions }
            else { settings.isHighResolutionPhotoEnabled = true }
            let unique = settings.uniqueID
            pending[unique] = PendingPhoto(request:request,path:path)
            Self.owners[unique] = self; acquiring = unique; capturing = true
            let info:[String:Any] = ["event":"capture_started","request_id":request,"pages":detectedPages,"rotation":plan.rotation,"preview_frame":lastGoodStatus?["preview_frame"] ?? NSNull()]
            publishCapture(info);emit(info)
            var status = lastGoodStatus ?? [:];status["event"]="status";status["capturing"]=true;status["ready"]=false
            emit(status)
            queue.async { [self] in
                let orientation = [90:UIInterfaceOrientation.portrait,270:.portraitUpsideDown,0:.landscapeRight,180:.landscapeLeft][plan.angle] ?? .portrait
                setOrientation(photos.connection(with:.video),angle:CGFloat(plan.angle),current:orientation)
                photos.capturePhoto(with:settings,delegate:self)
            }
        default: break
        }
    }
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let now = CACurrentMediaTime()
        guard now - lastFrame >= 0.16, let source = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        lastFrame = now
        var ticket: UInt = 0; var enabled = false; var turns: UInt32 = 0; var displayAngle = 90
        DispatchQueue.main.sync {
            ticket = generation; enabled = active && window != nil && callback != nil && !capturing
            turns = inferredTurns ?? UInt32(frameOrientation.analysisTurns(orientation.rawValue))
            displayAngle = CameraOrientation.angle(orientation.rawValue)
        }
        guard enabled else { return }
        // Session connection updates are queued. Reject transitional buffers,
        // rather than interpreting old pixels using the new orientation.
        if #available(iOS 17.0, *) {
            guard abs(connection.videoRotationAngle - CGFloat(displayAngle)) < 0.1 else { return }
        } else {
            guard CameraOrientation.angle(connection.videoOrientation.rawValue) == displayAngle else { return }
        }
        let w = CVPixelBufferGetWidth(source), h = CVPixelBufferGetHeight(source)
        let scale = min(1, 640.0 / Double(max(w,h)))
        let width = max(2, Int(Double(w) * scale)), height = max(2, Int(Double(h) * scale))
        var resized: CVPixelBuffer?
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &resized) == kCVReturnSuccess, let resized else { return }
        context.render(CIImage(cvPixelBuffer: source).transformed(by: CGAffineTransform(scaleX: scale, y: scale)), to: resized)
        CVPixelBufferLockBaseAddress(resized, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(resized, .readOnly) }
        guard let bytes = CVPixelBufferGetBaseAddress(resized)?.assumingMemoryBound(to: UInt8.self) else { return }
        let stride = CVPixelBufferGetBytesPerRow(resized)
        guard let json = detectPage(bytes, UInt(stride * height), UInt(width), UInt(height), UInt(stride), turns) else { return }
        let data = Data(bytes: json, count: strlen(json)); freeDetection(json)
        guard let result = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return }
        DispatchQueue.main.async { [weak self] in
            guard let self, self.active, self.window != nil, self.generation == ticket, !self.capturing else { return }
            self.accept(result, frameTime: now)
        }
    }
    private func accept(_ result: [String: Any], frameTime: CFTimeInterval) {
        let polygons = result["polygons"] as? [[[Double]]] ?? []
        let scene = result["scene"] as? [Double] ?? []
        let valid = result["valid"] as? Bool == true && !polygons.isEmpty
        // Cache only the orientation hint, never readiness. Even a slow first
        // orientation search can seed a fresh, single-pass revalidation frame.
        if valid, let turns = result["analysis_turns"] as? UInt32 { inferredTurns = turns % 4 }
        let decision = gate.accept(valid: valid, polygons: polygons, scene: scene,
            frameTime: frameTime, now: CACurrentMediaTime(), hardFailure: result["error"] != nil)
        if decision == .clear {
            invalidate(SmudeL(result["error"] != nil ? "Page detection unavailable" : "Hold the camera steady"), reason: result["error"] != nil ? "detector" : "")
            return
        }
        if decision == .hold, var retained = lastGoodStatus {
            retained["ready"] = ready; retained["capturing"] = false
            emit(retained)
            return
        }
        detectedPages = valid ? (result["pages"] as? Int ?? 1) : 0
        var status = result
        status.removeValue(forKey: "scene") // Native latch only; not UI payload.
        status["event"] = "status"; status["ready"] = ready; status["capturing"] = capturing
        if valid && !ready { status["message"] = SmudeL("Hold the camera steady") }
        lastGoodStatus = ready ? status : nil
        emit(status)
    }
    private func publishCapture(_ info:[String:Any]) {
        NotificationCenter.default.post(name:Notification.Name("smude.capture"),object:nil,userInfo:info)
    }
    private func releaseShutter(_ unique:Int64) {
        guard acquiring == unique else { return }
        acquiring=nil;capturing=false;captureAfter=CACurrentMediaTime()
        if callback != nil {
            var status=lastGoodStatus ?? [:];status["event"]="status";status["capturing"]=false
            status["ready"]=ready
            emit(status)
        }
    }
    private func retire(_ unique:Int64) {
        guard let ticket=pending[unique],ticket.stored && ticket.sensorFinished else{return}
        pending.removeValue(forKey:unique);Self.owners.removeValue(forKey:unique)
        if !active || window == nil || callback == nil { pause() }
    }
    func photoOutput(_ output:AVCapturePhotoOutput,didCapturePhotoFor resolvedSettings:AVCaptureResolvedPhotoSettings) {
        if #available(iOS 17.0,*), output.isResponsiveCaptureEnabled {
            DispatchQueue.main.async { [self] in releaseShutter(resolvedSettings.uniqueID) }
        }
    }
    func photoOutput(_ output:AVCapturePhotoOutput,didFinishProcessingPhoto photo:AVCapturePhoto,error:Error?) {
        let data=photo.fileDataRepresentation()
        let unique=photo.resolvedSettings.uniqueID
        DispatchQueue.main.async { [self] in
            guard var ticket=pending[unique],!ticket.writing,!ticket.stored else{return}
            ticket.writing=true;pending[unique]=ticket
            releaseShutter(unique)
            Self.saves.async { [self] in
                var success=false
                if let data,error==nil {
                    do {
                        let file=URL(fileURLWithPath:ticket.path)
                        try FileManager.default.createDirectory(at:file.deletingLastPathComponent(),withIntermediateDirectories:true)
                        // Preserve full sensor JPEG and EXIF. Rust applies orientation
                        // during its existing final processing, without a camera re-encode.
                        try data.write(to:file,options:.atomic);success=true
                    } catch { success=false }
                }
                let saved=success
                DispatchQueue.main.async { [self] in
                    guard var current=pending[unique],!current.stored else{return}
                    current.stored=true;pending[unique]=current
                    publishCapture(["event":saved ? "capture_saved":"capture_failed","request_id":ticket.request])
                    retire(unique)
                }
            }
        }
    }
    func photoOutput(_ output:AVCapturePhotoOutput,didFinishCaptureFor resolvedSettings:AVCaptureResolvedPhotoSettings,error:Error?) {
        let unique=resolvedSettings.uniqueID
        DispatchQueue.main.async { [self] in
            guard var ticket=pending[unique]else{return}
            ticket.sensorFinished=true
            if !ticket.writing && !ticket.stored {
                ticket.stored=true
                publishCapture(["event":"capture_failed","request_id":ticket.request])
            }
            pending[unique]=ticket;releaseShutter(unique);retire(unique)
        }
    }

}

@_cdecl("smude_camera_create")
func createCamera(_ userdata: UnsafeMutableRawPointer?, _ callback: @escaping @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?) -> Void) -> UnsafeMutableRawPointer {
    Unmanaged.passRetained(CameraView(userdata: userdata, callback: callback)).toOpaque()
}
@_cdecl("smude_camera_command")
func commandCamera(_ view:UnsafeMutableRawPointer,_ command:UnsafePointer<CChar>,_ json:UnsafePointer<CChar>) {
    let arguments=(try? JSONSerialization.jsonObject(with:Data(String(cString:json).utf8))) as? [String:Any] ?? [:]
    Unmanaged<CameraView>.fromOpaque(view).takeUnretainedValue().command(String(cString:command),arguments:arguments)
}
@_cdecl("smude_camera_shutdown")
func shutdownCamera(_ view: UnsafeMutableRawPointer) { Unmanaged<CameraView>.fromOpaque(view).takeUnretainedValue().shutdown() }

// iOS 26 adds a whole-content pop gesture. The music viewer needs horizontal
// paging and image panning there; only its system edge-pop remains enabled.
private final class ViewerGesturePolicy: UIView {
    private weak var recognizer: UIGestureRecognizer?
    private var previousEnabled = true
    var active = true { didSet { updatePolicy() } }
    override func didMoveToWindow() { super.didMoveToWindow(); updatePolicy(); DispatchQueue.main.async { [weak self] in self?.updatePolicy() } }
    override func didMoveToSuperview() { super.didMoveToSuperview(); updatePolicy() }
    private func updatePolicy() {
        guard active, window != nil else { restore(); return }
        if #available(iOS 26.0, *) {
            var responder: UIResponder? = self
            while let current = responder {
                if let controller = current as? UIViewController, let nav = controller.navigationController {
                    if let content = nav.interactiveContentPopGestureRecognizer, recognizer !== content {
                        restore(); recognizer = content; previousEnabled = content.isEnabled
                    }
                    recognizer?.isEnabled = false
                    return
                }
                responder = current.next
            }
        }
    }
    func restore() { recognizer?.isEnabled = previousEnabled; recognizer = nil }
}
@_cdecl("smude_viewer_policy_create")
func createViewerPolicy() -> UnsafeMutableRawPointer { Unmanaged.passRetained(ViewerGesturePolicy()).toOpaque() }
@_cdecl("smude_viewer_policy_active")
func setViewerPolicy(_ view: UnsafeMutableRawPointer, _ active: Bool) { Unmanaged<ViewerGesturePolicy>.fromOpaque(view).takeUnretainedValue().active = active }
@_cdecl("smude_viewer_policy_shutdown")
func shutdownViewerPolicy(_ view: UnsafeMutableRawPointer) { Unmanaged<ViewerGesturePolicy>.fromOpaque(view).takeUnretainedValue().restore() }
