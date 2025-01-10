//
//  MetalViewController.swift
//

#if os(iOS)
import UIKit
#elseif os(macOS)
import Cocoa
#endif
import MetalKit


#if os(iOS)
typealias OSViewController = UIViewController
typealias OSView = UIView
typealias OSPanGestureRecognizer = UIPanGestureRecognizer
#elseif os(macOS)
typealias OSViewController = NSViewController
typealias OSView = NSView
typealias OSPanGestureRecognizer = NSPanGestureRecognizer
#endif


class MetalViewController: OSViewController {
	let renderer: Renderer? = Renderer()
	let makeFallbackChild: () -> OSViewController

	private var _panGestureRecognizer: OSPanGestureRecognizer!

	init(makeFallbackChild: @escaping () -> OSViewController) {
		self.makeFallbackChild = makeFallbackChild
		super.init(nibName: nil, bundle: nil)
		self._panGestureRecognizer = .init(target: self, action: #selector(self._handlePanGesture(recognizer:)))
	}

	required init?(coder _: NSCoder) { fatalError("Unused") }

#if os(macOS)
	override var acceptsFirstResponder: Bool { true }
#endif
}


// MARK: - View delegate

extension MetalViewController {
	override func loadView() {
		if let renderer = self.renderer {
			let mtkView = MTKView(frame: .zero, device: renderer.device)
			mtkView.delegate = self
			mtkView.autoResizeDrawable = true
			self.view = mtkView
		} else {
			self.view = OSView()
		}
		self.view.translatesAutoresizingMaskIntoConstraints = false
	}

	override func viewDidLoad() {
		super.viewDidLoad()

		if self.renderer != nil {
			self.view.addGestureRecognizer(self._panGestureRecognizer)
		} else {
			let fallbackChild = self.makeFallbackChild()
			self.addChild(fallbackChild)
#if os(iOS)
			fallbackChild.view.autoresizingMask = [.flexibleWidth, .flexibleHeight]
#elseif os(macOS)
			fallbackChild.view.autoresizingMask = [.width, .height]
#endif
			self.view.addSubview(fallbackChild.view)
#if os(iOS)
			fallbackChild.didMove(toParent: self)
#endif
		}
	}

	private func _viewWillAppear() {
#if os(macOS)
		if let view = self.view as? MTKView {
			// TODO: This does not always work!?
			view.window?.makeFirstResponder(self)
			view.isPaused = false
		}

		DispatchQueue.main.async() { [weak self] in
			guard let `self` = self, let window = self.view.window else { return }
			window.makeFirstResponder(self)
		}
#endif
	}

	private func _viewDidDisappear() {
		if let view = self.view as? MTKView {
			self.resignFirstResponder()
			view.isPaused = true
		}
	}

	private func _viewDidLayout() {}

#if os(iOS)
	override func viewWillAppear(_ animated: Bool) { super.viewWillAppear(animated); self._viewWillAppear() }
	override func viewDidDisappear(_ animated: Bool) { super.viewDidDisappear(animated); self._viewDidDisappear() }
	override func viewDidLayoutSubviews() { super.viewDidLayoutSubviews(); self._viewDidLayout() }
#elseif os(macOS)
	override func viewWillAppear() { super.viewWillAppear(); self._viewWillAppear() }
	override func viewDidDisappear() { super.viewDidDisappear(); self._viewDidDisappear() }
	override func viewDidLayout() { super.viewDidLayout(); self._viewDidLayout() }
#endif
}


// MARK: Metal

extension MetalViewController: MTKViewDelegate {
	func mtkView(_: MTKView, drawableSizeWillChange _: CGSize) {}

	func draw(in view: MTKView) {
		autoreleasepool() {
			guard
				let renderer = self.renderer,
				let drawable = asserted(view.currentDrawable),
				let renderPassDescriptor = asserted(view.currentRenderPassDescriptor)
			else { return }

			renderer(subject: .init(), to: drawable, using: renderPassDescriptor, size: view.drawableSize)
		}
	}
}


// MARK: - Interaction

extension MetalViewController {

	@objc private func _handlePanGesture(recognizer: OSPanGestureRecognizer) {
		guard recognizer.state == .began else { return }
		print(self, #function, recognizer)
	}

#if os(iOS)
	override func pressesBegan(_ presses: Set<UIPress>, with _: UIPressesEvent?) {
		guard presses.count == 1, let key = presses.first!.key else { return }
		print(self, #function, key)
	}

	override func pressesEnded(_ presses: Set<UIPress>, with _: UIPressesEvent?) {
		guard presses.count == 1, let key = presses.first!.key else { return }
		print(self, #function, key)
	}
#elseif os(macOS)
	override func keyDown(with event: NSEvent) {
		print(self, #function, event)
	}

	override func keyUp(with event: NSEvent) {
		print(self, #function, event)
	}
#endif
}
