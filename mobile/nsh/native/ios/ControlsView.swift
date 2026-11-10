import UIKit
/// A solid sector, not a stroked ring: the stroke's inner radius is zero.
/// Core Animation interpolates strokeEnd smoothly without a per-frame timer.
private final class PieProgressView: UIView {
    private let sector = CAShapeLayer()
    private var fraction: CGFloat = 0
    override init(frame: CGRect) {
        super.init(frame: frame)
        isUserInteractionEnabled = false
        sector.fillColor = UIColor.clear.cgColor
        sector.strokeColor = UIColor(red: 1, green: 0.8, blue: 0.4, alpha: 1).cgColor
        sector.lineCap = .butt
        sector.strokeEnd = 0
        layer.addSublayer(sector)
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
    override func layoutSubviews() {
        super.layoutSubviews()
        let diameter = min(bounds.width, bounds.height)
        CATransaction.begin(); CATransaction.setDisableActions(true)
        sector.frame = bounds
        sector.path = UIBezierPath(arcCenter: CGPoint(x: bounds.midX, y: bounds.midY),
            radius: diameter / 4, startAngle: -.pi / 2, endAngle: 3 * .pi / 2, clockwise: true).cgPath
        sector.lineWidth = diameter / 2
        CATransaction.commit()
    }
    func update(value: Double, indeterminate: Bool) {
        let next = indeterminate ? CGFloat(0) : CGFloat(min(100, max(0, value)) / 100)
        if indeterminate { sector.removeAnimation(forKey: "completion") }
        guard next != fraction else { return }
        let previous = sector.presentation()?.strokeEnd ?? fraction
        fraction = next
        CATransaction.begin(); CATransaction.setDisableActions(true)
        sector.strokeEnd = next
        CATransaction.commit()
        if window != nil && !indeterminate && !UIAccessibility.isReduceMotionEnabled {
            let animation = CABasicAnimation(keyPath: "strokeEnd")
            animation.fromValue = previous; animation.toValue = next
            animation.duration = 0.2
            animation.timingFunction = CAMediaTimingFunction(name: .easeOut)
            sector.add(animation, forKey: "completion")
        }
    }
}

private typealias ControlEvent = @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?) -> Void
/// Platform controls only. Headers provide a UIBarButtonItem directly to NSH;
/// their placeholder view is never installed as a bar item's customView.
private final class SmudeControl: UIView {
    var callback: ControlEvent?
    var context: UnsafeMutableRawPointer?
    private var button: UIButton?
    private var segments: UISegmentedControl?
    private var progress: PieProgressView?
    private var progressBackdrop: UIVisualEffectView?
    private(set) var barItem: UIBarButtonItem?
    private var kind = ""
    private let gold = UIColor(red: 1, green: 0.8, blue: 0.4, alpha: 1)

    func configure(_ config: [String: Any]) {
        let next = config["bar_button"] as? Bool == true ? "bar" : config["kind"] as? String ?? "button"
        if kind != next {
            subviews.forEach { $0.removeFromSuperview() }
            button = nil; segments = nil; barItem = nil; progress = nil; progressBackdrop = nil; kind = next
            if next == "bar" {
                barItem = UIBarButtonItem(image: nil, style: .plain, target: self, action: #selector(press))
                if #available(iOS 26.0, *) { barItem?.sharesBackground = false }
            } else if next == "progress" {
                isUserInteractionEnabled = false
                let backdrop: UIVisualEffectView
                if #available(iOS 26.0, *) {
                    backdrop = UIVisualEffectView(effect: UIGlassEffect(style: .regular))
                    backdrop.cornerConfiguration = .capsule()
                } else {
                    backdrop = UIVisualEffectView(effect: UIBlurEffect(style: .systemThinMaterial))
                    backdrop.clipsToBounds = true
                }
                let control = PieProgressView(frame: .zero)
                backdrop.contentView.addSubview(control)
                addSubview(backdrop); progress = control; progressBackdrop = backdrop
            } else if next == "segments" {
                let control = UISegmentedControl(items: ["", ""])
                control.addTarget(self, action: #selector(change), for: .valueChanged)
                addSubview(control); segments = control
            } else {
                let control = UIButton(type: .system)
                if #available(iOS 26.0, *) { control.configuration = .glass() }
                else { control.configuration = .plain() }
                control.configuration?.cornerStyle = .capsule
                control.configuration?.baseForegroundColor = gold
                control.configuration?.contentInsets = .zero
                control.addTarget(self, action: #selector(press), for: .touchUpInside)
                addSubview(control); button = control
            }
        }
        let identifier = config["test_id"] as? String
        if let control = segments {
            let labels = config["labels"] as? [String] ?? ["", ""]
            let enabled = config["enabled"] as? [Bool] ?? [true, true]
            let symbols = config["symbols"] as? [String] ?? []
            for i in 0..<2 {
                if symbols.indices.contains(i) {
                    let image = UIImage(systemName: symbols[i])
                    image?.accessibilityLabel = labels.indices.contains(i) ? labels[i] : nil
                    control.setImage(image, forSegmentAt: i)
                } else {
                    control.setTitle(labels.indices.contains(i) ? labels[i] : "", forSegmentAt: i)
                }
                control.setEnabled(enabled.indices.contains(i) && enabled[i], forSegmentAt: i)
            }
            control.selectedSegmentIndex = config["selected"] as? Int ?? 0
            control.accessibilityIdentifier = identifier
        }
        if let control = progress {
            control.update(value: config["value"] as? Double ?? 0,
                indeterminate: config["indeterminate"] as? Bool == true)
        }
        if let control = button {
            control.isEnabled = config["enabled"] as? Bool ?? true
            control.accessibilityIdentifier = identifier
            control.accessibilityLabel = config["label"] as? String
            control.configuration?.image = UIImage(systemName: config["symbol"] as? String ?? "ellipsis")
            control.menu = menu(config)
            control.showsMenuAsPrimaryAction = control.menu != nil
        }
        if let item = barItem {
            item.image = UIImage(systemName: config["symbol"] as? String ?? "ellipsis")
            item.accessibilityIdentifier = identifier
            item.accessibilityLabel = config["label"] as? String
            item.isEnabled = config["enabled"] as? Bool ?? true
            item.menu = menu(config)
            // A menu-only bar item opens its native menu on the first tap.
            item.target = item.menu == nil ? self : nil
            item.action = item.menu == nil ? #selector(press) : nil
        }
        setNeedsLayout()
    }
    private func menu(_ config: [String: Any]) -> UIMenu? {
        let choices = config["choices"] as? [[String: Any]] ?? []
        let actions = choices.map { entry -> UIAction in
            let destructive = entry["destructive"] as? Bool == true
            return UIAction(title: entry["title"] as? String ?? "",
                image: UIImage(systemName: entry["symbol"] as? String ?? ""),
                attributes: destructive ? [.destructive] : []) { [weak self] _ in
                self?.send(["event": "select", "id": entry["id"] as? String ?? "", "payload": entry["payload"] ?? NSNull()])
            }
        }
        guard !actions.isEmpty else { return nil }
        let regular = actions.filter { !$0.attributes.contains(.destructive) }
        let destructive = actions.filter { $0.attributes.contains(.destructive) }
        let children: [UIMenuElement] = destructive.isEmpty || regular.isEmpty ? actions : [
            UIMenu(options: .displayInline, children: regular),
            UIMenu(options: .displayInline, children: destructive)
        ]
        return UIMenu(children: children)
    }
    override func layoutSubviews() {
        super.layoutSubviews()
        button?.frame = bounds
        segments?.frame = bounds.insetBy(dx: 0, dy: 4)
        if let backdrop = progressBackdrop {
            backdrop.frame = bounds
            if #unavailable(iOS 26.0) { backdrop.layer.cornerRadius = min(bounds.width, bounds.height) / 2 }
            backdrop.contentView.clipsToBounds = true
            backdrop.contentView.layer.cornerRadius = min(bounds.width, bounds.height) / 2
            progress?.frame = backdrop.contentView.bounds
        }
    }
    @objc private func press() {
        if button?.menu == nil && barItem?.menu == nil { send(["event": "select", "id": "press"]) }
    }
    @objc private func change() {
        if let control = segments { send(["event": "change", "value": control.selectedSegmentIndex]) }
    }
    private func send(_ payload: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
              let json = String(data: data, encoding: .utf8) else { return }
        json.withCString { callback?(context, $0) }
    }
    func shutdown() { callback = nil; context = nil; button?.menu = nil; barItem?.menu = nil; barItem?.target = nil }
}
@_cdecl("smude_control_create")
func smudeControlCreate(_ context: UnsafeMutableRawPointer?, _ callback: @escaping @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?) -> Void) -> UnsafeMutableRawPointer {
    let view = SmudeControl(); view.context = context; view.callback = callback
    return Unmanaged.passRetained(view).toOpaque()
}
@_cdecl("smude_control_config")
func smudeControlConfig(_ raw: UnsafeMutableRawPointer, _ json: UnsafePointer<CChar>) {
    let view = Unmanaged<SmudeControl>.fromOpaque(raw).takeUnretainedValue()
    guard let data = String(cString: json).data(using: .utf8),
          let config = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return }
    view.configure(config)
}
@_cdecl("smude_control_bar_item")
func smudeControlBarItem(_ raw: UnsafeMutableRawPointer) -> UnsafeMutableRawPointer? {
    guard let item = Unmanaged<SmudeControl>.fromOpaque(raw).takeUnretainedValue().barItem else { return nil }
    return Unmanaged.passRetained(item).toOpaque()
}
@_cdecl("smude_control_shutdown")
func smudeControlShutdown(_ raw: UnsafeMutableRawPointer) {
    Unmanaged<SmudeControl>.fromOpaque(raw).takeUnretainedValue().shutdown()
}
