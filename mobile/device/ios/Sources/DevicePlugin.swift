import AVFoundation
import CryptoKit
import ImageIO
import Tauri
import UIKit
import UniformTypeIdentifiers

private struct ImageArgs: Decodable { let source: String; let destination: String }
private struct CaptureArgs: Decodable { let destination: String }
private struct ExportArgs: Decodable { let source: String; let directory: String?; let verify: Bool? }

class DevicePlugin: Plugin, UIImagePickerControllerDelegate, UINavigationControllerDelegate,
    UIDocumentPickerDelegate, UIAdaptivePresentationControllerDelegate {
    private var cameraCall: Invoke?
    private var cameraDestination: String?
    private var exportCall: Invoke?
    private var exportSource: URL?
    private var verifyExport = false

    @objc func decodeImage(_ invoke: Invoke) throws {
        let args = try invoke.parseArgs(ImageArgs.self)
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try self.normalize(source: URL(fileURLWithPath: args.source), destination: args.destination)
                invoke.resolve()
            } catch { invoke.reject(error.localizedDescription) }
        }
    }

    /// Decode at full resolution, apply EXIF once and write lossless PNG.
    private func normalize(source: URL, destination: String) throws {
        guard let input = CGImageSourceCreateWithURL(source as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(input, 0, nil) as? [CFString: Any],
              let width = props[kCGImagePropertyPixelWidth] as? Int,
              let height = props[kCGImagePropertyPixelHeight] as? Int,
              width > 0, height > 0, max(width, height) <= 12000, width * height <= 64_000_000
        else { throw failure("Bild nicht lesbar oder größer als 64 Megapixel / 12000 Pixel Kantenlänge.") }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(width, height),
            kCGImageSourceShouldCacheImmediately: true
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(input, 0, options as CFDictionary),
              let output = CGImageDestinationCreateWithURL(URL(fileURLWithPath: destination) as CFURL,
                  UTType.png.identifier as CFString, 1, nil)
        else { throw failure("Bild konnte nicht dekodiert werden.") }
        CGImageDestinationAddImage(output, image, nil)
        guard CGImageDestinationFinalize(output) else { throw failure("Bild konnte nicht gespeichert werden.") }
    }

    @objc func capturePhoto(_ invoke: Invoke) throws {
        let args = try invoke.parseArgs(CaptureArgs.self)
        DispatchQueue.main.async {
            guard self.cameraCall == nil else { invoke.reject("Kamera ist bereits geöffnet."); return }
            guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
                invoke.reject("Keine Kamera verfügbar."); return
            }
            self.cameraCall = invoke
            self.cameraDestination = args.destination
            AVCaptureDevice.requestAccess(for: .video) { allowed in
                DispatchQueue.main.async {
                    guard allowed else {
                        self.finishCamera(error: "Bitte Kamerazugriff in den iOS-Einstellungen erlauben.")
                        return
                    }
                    let picker = UIImagePickerController()
                    picker.sourceType = .camera
                    picker.cameraCaptureMode = .photo
                    picker.delegate = self
                    picker.modalPresentationStyle = .fullScreen
                    self.manager.viewController?.present(picker, animated: true)
                }
            }
        }
    }

    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true)
        finishCamera()
    }

    func imagePickerController(_ picker: UIImagePickerController,
        didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
        let url = info[.imageURL] as? URL
        let image = info[.originalImage] as? UIImage
        let destination = cameraDestination
        picker.dismiss(animated: true)
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                guard let destination = destination else { throw self.failure("Kameraziel fehlt.") }
                if let url = url {
                    try self.normalize(source: url, destination: destination)
                } else if let image = image {
                    let size = CGSize(width: image.size.width * image.scale, height: image.size.height * image.scale)
                    guard max(size.width, size.height) <= 12000, size.width * size.height <= 64_000_000 else {
                        throw self.failure("Kamerabild ist zu groß.")
                    }
                    let format = UIGraphicsImageRendererFormat(); format.scale = 1; format.opaque = true
                    let data = UIGraphicsImageRenderer(size: size, format: format).pngData { _ in
                        image.draw(in: CGRect(origin: .zero, size: size))
                    }
                    try data.write(to: URL(fileURLWithPath: destination), options: .atomic)
                } else { throw self.failure("Kamera hat kein Foto geliefert.") }
                DispatchQueue.main.async { self.finishCamera(path: destination) }
            } catch { DispatchQueue.main.async { self.finishCamera(error: error.localizedDescription) } }
        }
    }

    private func finishCamera(path: String? = nil, error: String? = nil) {
        let call = cameraCall; cameraCall = nil; cameraDestination = nil
        if let error = error { call?.reject(error) }
        else { call?.resolve(["path": path as Any? ?? NSNull()]) }
    }

    /// Export the real completed PNG. Never export an empty placeholder and
    /// subsequently try to write through a provider URL whose grant has ended.
    @objc func exportImage(_ invoke: Invoke) throws {
        let args = try invoke.parseArgs(ExportArgs.self)
        DispatchQueue.main.async {
            guard self.exportCall == nil else { invoke.reject("Export ist bereits geöffnet."); return }
            self.exportCall = invoke
            self.exportSource = URL(fileURLWithPath: args.source)
            self.verifyExport = args.verify ?? false
            let picker = UIDocumentPickerViewController(forExporting: [URL(fileURLWithPath: args.source)], asCopy: true)
            picker.delegate = self
            picker.presentationController?.delegate = self
            if let directory = args.directory { picker.directoryURL = URL(fileURLWithPath: directory) }
            self.manager.viewController?.present(picker, animated: true) {
                picker.presentationController?.delegate = self
            }
        }
    }

    func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        let call = exportCall
        let source = exportSource
        let verify = verifyExport
        exportCall = nil; exportSource = nil; verifyExport = false
        guard let destination = urls.first else { call?.resolve(["saved": false]); return }
        DispatchQueue.global(qos: .userInitiated).async {
            var result: [String: Any] = ["saved": true, "destination": destination.absoluteString]
            if verify {
                let access = destination.startAccessingSecurityScopedResource()
                defer { if access { destination.stopAccessingSecurityScopedResource() } }
                do {
                    let actual = try Data(contentsOf: destination)
                    guard let source = source else { throw self.failure("Exportquelle fehlt.") }
                    let expected = try Data(contentsOf: source)
                    let hash = SHA256.hash(data: actual).map { String(format: "%02x", $0) }.joined()
                    result["bytes"] = actual.count
                    result["sha256"] = hash
                    result["matches_source"] = actual == expected
                } catch { result["verification_error"] = error.localizedDescription }
            }
            call?.resolve(result)
        }
    }
    func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) { finishExport() }
    func presentationControllerDidDismiss(_ presentationController: UIPresentationController) { finishExport() }
    private func finishExport() {
        exportCall?.resolve(["saved": false]); exportCall = nil; exportSource = nil; verifyExport = false
    }
    private func failure(_ message: String) -> NSError {
        NSError(domain: "SmudeMedia", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
}

@_cdecl("init_plugin_smude_device")
public func initSmudeDevice() -> Plugin { DevicePlugin() }
