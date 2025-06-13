//
//  Renderer.swift
//

import Metal


class Renderer {

	struct Subject {}

	let device: any MTLDevice
	private let _commandQueue: any MTLCommandQueue
//	private let _computeBitwiseAndOr: ComputeBitwiseAndOr
//	private let _computeCheckSorted: ComputeCheckSorted
//	private let _computePrefixSums: ComputePrefixSums
	private let _computeRadixSort: ComputeRadixSort
	private let _computePipelineStates: (
//		debugRadixSort: (
//			pre: any MTLComputePipelineState,
//			post: any MTLComputePipelineState
//		),
		generateTexture: any MTLComputePipelineState,
		stepTextureSorting: any MTLComputePipelineState
	)
	private let _renderPipelineState: any MTLRenderPipelineState
	private let _texture: any MTLTexture

	private var _frameIndex: UInt = 0

	fileprivate struct _BatcherOddEvenSortingParameters {
		let n: UInt
		var p: UInt = 1, k: UInt = 1
	}
	private var _sortingParameters = _BatcherOddEvenSortingParameters(n: 65536)

	init?() {
		guard
			case let n = self._sortingParameters.n,
			let device = asserted(MTLCreateSystemDefaultDevice()),
			let library = asserted(device.makeDefaultLibrary()),
			let commandQueue = asserted(device.makeCommandQueue()),
			let computeRadixSort = asserted(ComputeRadixSort(device: device, library: library)),
			let generateTextureComputePipelineState = asserted({ () -> (any MTLComputePipelineState)? in
				guard let function = asserted(library.makeFunction(name: "main_gen_tex_kernel")) else { return nil }
				do { return try device.makeComputePipelineState(function: function) }
				catch { assertionFailure(String(describing: error)); return nil }
			}()),
			let stepTextureSortingComputePipelineState = asserted({ () -> (any MTLComputePipelineState)? in
				guard let function = asserted(library.makeFunction(name: "main_step_tex_sort_kernel")) else { return nil }
				do { return try device.makeComputePipelineState(function: function) }
				catch { assertionFailure(String(describing: error)); return nil }
			}()),
			let renderPipelineState = asserted({ () -> (any MTLRenderPipelineState)? in
				guard
					let vertexFunction = asserted(library.makeFunction(name: "main_vertex")),
					let fragmentFunction = asserted(library.makeFunction(name: "main_fragment"))
				else { return nil }
				let descriptor = MTLRenderPipelineDescriptor()
				descriptor.vertexFunction = vertexFunction
				descriptor.fragmentFunction = fragmentFunction
				descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
				do { return try device.makeRenderPipelineState(descriptor: descriptor) }
				catch { assertionFailure(String(describing: error)); return nil }
			}()),
			let texture = asserted({ () -> (any MTLTexture)? in
				let descriptor = MTLTextureDescriptor.texture2DDescriptor(
					pixelFormat: .bgra8Unorm,
					width: Int(usqrt(n)),
					height: Int(usqrt(n)),
					mipmapped: false
				)
				descriptor.usage = [.shaderRead, .shaderWrite]
				descriptor.storageMode = .private
				guard let texture = asserted(device.makeTexture(descriptor: descriptor)) else { return nil }
				texture.label = "Texture"
				return texture
			}())
		else { return nil }

		self.device = device
		self._commandQueue = commandQueue
		self._computeRadixSort = computeRadixSort
		self._computePipelineStates = (
			generateTextureComputePipelineState,
			stepTextureSortingComputePipelineState
		)
		self._renderPipelineState = renderPipelineState
		self._texture = texture
	}

	func setUp() {
		let captureManager = MTLCaptureManager.shared()
		let captureScope = captureManager.makeCaptureScope(commandQueue: self._commandQueue)
		captureScope.label = "Initial set-up"
		let captureDescriptor = MTLCaptureDescriptor()
		captureDescriptor.captureObject = captureScope
		try! captureManager.startCapture(with: captureDescriptor)
		captureScope.begin()
		defer { captureScope.end(); captureManager.stopCapture() }

		guard let commandBuffer = asserted(self._commandQueue.makeCommandBuffer()) else { return }

		// (Debug) Radix Sort
		do {
			let elementsCount = 1 << 20 + 1
			let keysBufferSize = elementsCount * MemoryLayout<UInt32>.stride
			let valuesStride = Int(log10(Double(elementsCount)).rounded(.up))
			let valuesBufferSize = elementsCount * valuesStride

			guard
				let keysBuffer = asserted(commandBuffer.device.makeBuffer(length: keysBufferSize, options: .storageModeShared)),
				let valuesBuffer = asserted(commandBuffer.device.makeBuffer(length: valuesBufferSize, options: .storageModeShared))
			else { return }
			keysBuffer.label = "Keys (Radix Sort Input/Output)"
			valuesBuffer.label = "Values (Radix Sort Input/Output)"

			var keysSet = Set<UInt32>()
			let scrambleOffset = 26561 % elementsCount
			let scrambleStride = nextCoprime(from: Int(1.61803398875 * Double(elementsCount)), relativeTo: elementsCount)
			for i in 0..<elementsCount {
				let key = UInt32((scrambleOffset + scrambleStride * i) % elementsCount)
				keysSet.insert(key)
				keysBuffer.contents().bindMemory(to: UInt32.self, capacity: elementsCount)[i] = key
				let valuePtr = valuesBuffer.contents().bindMemory(to: UInt8.self, capacity: elementsCount * valuesStride) + i * valuesStride
				let valueString = String(format: "%0\(valuesStride)d", key)
				for (j, b) in valueString.utf8.enumerated() {
					valuePtr[j] = b
				}

				switch key {
				case let k where k == elementsCount - 1: fallthrough
				case 0, 1337, 69420: print("\(i): \(key) -> \(valueString.debugDescription)")
				default: break
				}
			}
			assert(keysSet.count == elementsCount, "Keys buffer must contain all unique keys!")

			self._computeRadixSort.encode(to: commandBuffer, inoutBuffers: .keysAndValues(keys: keysBuffer, values: valuesBuffer, valuesStride: valuesStride))

			commandBuffer.addCompletedHandler() { _ in
				var incorrect = false
				for i in 0..<elementsCount {
					let value = keysBuffer.contents().bindMemory(to: UInt32.self, capacity: elementsCount)[i]
					if value != UInt32(i) { incorrect = true; break }
				}
				print(incorrect ? "Incorrect radix sort output..." : "Correct radix sort output! Yay!!!")

				for i in [0, 1337, 69420, elementsCount - 1] {
					if i >= elementsCount { continue }
					let keyPtr = keysBuffer.contents().bindMemory(to: UInt32.self, capacity: elementsCount) + i
					let valuePtr = valuesBuffer.contents().bindMemory(to: UInt8.self, capacity: elementsCount * valuesStride) + i * valuesStride
					let valueString = String.init(decoding: UnsafeBufferPointer(start: valuePtr, count: valuesStride), as: UTF8.self)
					print("\(i)th value in values buffer (key: \(keyPtr.pointee)): \(valueString.debugDescription)")
				}
			}
		}

		guard let commandEncoder = asserted(commandBuffer.makeComputeCommandEncoder()) else { return }

		commandEncoder.setComputePipelineState(self._computePipelineStates.generateTexture)
		commandEncoder.setTexture(self._texture, index: 0)
		commandEncoder.dispatchThreads(MTLSize(width: self._texture.width, height: self._texture.height, depth: 1), threadsPerThreadgroup: {
			let width = self._computePipelineStates.generateTexture.threadExecutionWidth
			return MTLSize(width: width, height: self._computePipelineStates.generateTexture.maxTotalThreadsPerThreadgroup / width, depth: 1)
		}())
		commandEncoder.endEncoding()

		commandBuffer.commit()
		commandBuffer.waitUntilCompleted()

		print("Finished set-up!")
	}

	func render(
		_ subject: Subject,
		to drawable: any MTLDrawable,
		using renderPassDescriptor: MTLRenderPassDescriptor,
		size drawableSize: CGSize
	) {
		defer { self._frameIndex += 1 }

		guard let commandBuffer = asserted(self._commandQueue.makeCommandBuffer()) else { return }

		if self._frameIndex > 0 && self._frameIndex % 10 == 0 && !self._sortingParameters.ended {
			print("k: \(self._sortingParameters.k), p: \(self._sortingParameters.p)")
			guard let commandEncoder = asserted(commandBuffer.makeComputeCommandEncoder()) else { return }

			commandEncoder.setComputePipelineState(self._computePipelineStates.stepTextureSorting)
			var uniforms = MainStepFragmentSortingUniforms(p: self._sortingParameters.p, k: self._sortingParameters.k)
			commandEncoder.setBytes(&uniforms, length: MemoryLayout.stride(ofValue: uniforms), index: 0)
			commandEncoder.setTexture(self._texture, index: 0)
			commandEncoder.dispatchThreads(MTLSize(width: self._texture.width, height: self._texture.height / 2, depth: 1), threadsPerThreadgroup: {
				let width = self._computePipelineStates.stepTextureSorting.threadExecutionWidth
				return MTLSize(width: width, height: self._computePipelineStates.stepTextureSorting.maxTotalThreadsPerThreadgroup / width, depth: 1)
			}())
			commandEncoder.endEncoding()
			self._sortingParameters.formNext()
		}

		guard let commandEncoder = asserted(commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)) else { return }

		commandEncoder.setFrontFacing(.counterClockwise)
		commandEncoder.setCullMode(.back)

		commandEncoder.setRenderPipelineState(self._renderPipelineState)
		var fragmentUniforms = MainFragmentUniforms(frameIndex: self._frameIndex)
		commandEncoder.setFragmentBytes(&fragmentUniforms, length: MemoryLayout.stride(ofValue: fragmentUniforms), index: 0)
		commandEncoder.setFragmentTexture(self._texture, index: 0)
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

extension Renderer._BatcherOddEvenSortingParameters {
	var ended: Bool { self.p == self.n }

	mutating func formNext() {
		if self.k > 1 {
			self.k /= 2
		} else if self.p < self.n {
			self.p *= 2
			self.k = self.p
		}
	}
}

func usqrt(_ x: UInt) -> UInt {
	var y: UInt = 1
	while y * y <= x {
		y += 1
	}
	return y - 1
}

func nextCoprime(from value: Int, relativeTo n: Int) -> Int {
	var candidate = value
	while true {
		if gcd(candidate, n) == 1 {
			return candidate
		}
		candidate += 1
	}
}

func gcd(_ m: Int, _ n: Int) -> Int {
	var a: Int = 0
	var b: Int = max(m, n)
	var r: Int = min(m, n)

	while r != 0 {
		a = b
		b = r
		r = a % b
	}
	return b
}
