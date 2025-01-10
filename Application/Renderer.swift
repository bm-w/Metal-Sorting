//
//  Renderer.swift
//

import Metal


class Renderer {

	struct Subject {}

	let device: any MTLDevice
	private let _commandQueue: any MTLCommandQueue
	private let _renderPipelineState: any MTLRenderPipelineState

	private var _frameIndex: UInt = 0

	init?() {
		guard
			let device = asserted(MTLCreateSystemDefaultDevice()),
			let commandQueue = asserted(device.makeCommandQueue()),
			let renderPipelineState = asserted({ () -> (any MTLRenderPipelineState)? in
				guard
					let library = asserted(device.makeDefaultLibrary()),
					let vertexFunction = asserted(library.makeFunction(name: "main_vertex")),
					let fragmentFunction = asserted(library.makeFunction(name: "main_fragment"))
				else { return nil }
				let descriptor = MTLRenderPipelineDescriptor()
				descriptor.vertexFunction = vertexFunction
				descriptor.fragmentFunction = fragmentFunction
				descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
				do { return try device.makeRenderPipelineState(descriptor: descriptor) }
				catch { assertionFailure(String(describing: error)); return nil }
			}())
		else { return nil }

		self.device = device
		self._commandQueue = commandQueue
		self._renderPipelineState = renderPipelineState
	}

	func render(
		_ subject: Subject,
		to drawable: any MTLDrawable,
		using renderPassDescriptor: MTLRenderPassDescriptor,
		size drawableSize: CGSize
	) {
		defer { self._frameIndex += 1 }

		guard
			let commandBuffer = asserted(self._commandQueue.makeCommandBuffer()),
			let commandEncoder = asserted(commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor))
		else { return }

		commandEncoder.setFrontFacing(.counterClockwise)
		commandEncoder.setCullMode(.back)

		commandEncoder.setRenderPipelineState(self._renderPipelineState)
		var fragmentUniforms = FragmentUniforms(frameIndex: self._frameIndex)
		commandEncoder.setFragmentBytes(&fragmentUniforms, length: MemoryLayout<FragmentUniforms>.stride, index: 0)
		commandEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

		commandEncoder.endEncoding()

		commandBuffer.present(drawable)
		commandBuffer.commit()
	}

	func callAsFunction(
		subject: Subject,
		to drawable: any MTLDrawable,
		using renderPassDescriptor: MTLRenderPassDescriptor,
		size drawableSize: CGSize
	) {
		self.render(subject, to: drawable, using: renderPassDescriptor, size: drawableSize)
	}
}
