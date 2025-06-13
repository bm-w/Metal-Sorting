//
//  RadixSort.swift
//  Metal-Sorting
//
//  Created by Bastiaan van de Weerd on 01-06-25.
//

import Metal


struct ComputeRadixSort {
	let nWay: Int
	private let _computeBitwiseAndOr: ComputeBitwiseAndOr
//	TODO: `private let _computeCheckSorted: ComputeCheckSorted`?
	private let _computePrefixSums: ComputePrefixSums
	private let _pipelineStates: (
		encode: any MTLComputePipelineState,
		local: any MTLComputePipelineState,
		global: any MTLComputePipelineState,
		finalCopy: any MTLComputePipelineState
	)
	private let _indirectEncodeCommandBuffer: any MTLIndirectCommandBuffer
	private let _encodeArgument0Buffer: any MTLBuffer
	private let _maxCommandCount: Int

	let maxTotalThreadsPerThreadgroup: Int

	init?(nWay: Int = 4, device: any MTLDevice, library: any MTLLibrary) {
		guard
			case 2..<32 = nWay,
			asserted(nWay.nonzeroBitCount == 1), // Must be a power of two
			case var nWayBytes = UInt8(nWay),
			case let constantValues = { () in
				let values = MTLFunctionConstantValues()
				values.setConstantValue(&nWayBytes, type: .uchar, index: 0)
				return values
			}(),
			let computeBitwiseAndOr = asserted(ComputeBitwiseAndOr(device: device, library: library)),
			let computePrefixSums = asserted(ComputePrefixSums(device: device, library: library, supportIndirectCommandBuffers: true)),
			let encodeKernelFunction = asserted({ () -> (any MTLFunction)? in
				do { return try library.makeFunction(name: "compute_radix_sort_encode", constantValues: constantValues) }
				catch { assertionFailure(String(describing: error)); return nil }
			}()),
			let encodePipelineState = asserted({ () -> (any MTLComputePipelineState)? in
				let descriptor = MTLComputePipelineDescriptor()
				descriptor.label = "Radix Sort / Encode"
				descriptor.computeFunction = encodeKernelFunction
				do { return try device.makeComputePipelineState(descriptor: descriptor, options: []).0 }
				catch { assertionFailure(String(describing: error)); return nil }
			}()),
			let localPipelineState = asserted({ () -> (any MTLComputePipelineState)? in
				let function: any MTLFunction
				do { function = try library.makeFunction(name: "compute_radix_sort_local", constantValues: constantValues) }
				catch { assertionFailure(String(describing: error)); return nil }
				let descriptor = MTLComputePipelineDescriptor()
				descriptor.label = "Radix Sort / Local"
				descriptor.computeFunction = function
				descriptor.supportIndirectCommandBuffers = true
				do { return try device.makeComputePipelineState(descriptor: descriptor, options: []).0 }
				catch { assertionFailure(String(describing: error)); return nil }
			}()),
			let globalPipelineState = asserted({ () -> (any MTLComputePipelineState)? in
				let function: any MTLFunction
				do { function = try library.makeFunction(name: "compute_radix_sort_global", constantValues: constantValues) }
				catch { assertionFailure(String(describing: error)); return nil }
				let descriptor = MTLComputePipelineDescriptor()
				descriptor.label = "Radix Sort / Global"
				descriptor.computeFunction = function
				descriptor.supportIndirectCommandBuffers = true
				do { return try device.makeComputePipelineState(descriptor: descriptor, options: []).0 }
				catch { assertionFailure(String(describing: error)); return nil }
			}()),
			let finalCopyPipelineState = asserted({ () -> (any MTLComputePipelineState)? in
				guard let function = asserted(library.makeFunction(name: "compute_radix_sort_final_copy")) else { return nil }
				let descriptor = MTLComputePipelineDescriptor()
				descriptor.label = "Radix Sort / Final Copy"
				descriptor.computeFunction = function
				descriptor.supportIndirectCommandBuffers = true
				do { return try device.makeComputePipelineState(descriptor: descriptor, options: []).0 }
				catch { assertionFailure(String(describing: error)); return nil }
			}()),
			let threadgroupSize = asserted(UInt16(exactly: [
				computeBitwiseAndOr.maxTotalThreadsPerThreadgroup,
				computePrefixSums.maxTotalThreadsPerThreadgroup,
				localPipelineState.maxTotalThreadsPerThreadgroup,
				globalPipelineState.maxTotalThreadsPerThreadgroup,
				finalCopyPipelineState.maxTotalThreadsPerThreadgroup,
			].min()!)),
			case let maxCommandCount = ((32 - 1) / nWay.trailingZeroBitCount + 1) * (2 + 7) + 1,
			let indirectEncodeCommandBuffer = asserted({ () -> (any MTLIndirectCommandBuffer)? in
				let descriptor = MTLIndirectCommandBufferDescriptor()
				descriptor.commandTypes = [.concurrentDispatchThreads]
				descriptor.inheritBuffers = false
				descriptor.maxKernelBufferBindCount = 6
				guard let buffer = asserted(device.makeIndirectCommandBuffer(descriptor: descriptor, maxCommandCount: maxCommandCount)) else { return nil }
				buffer.label = "Radix Sort / Indirect Encode Command Buffer"
				return buffer
			}()),
			let encodeArgument0Buffer = asserted({ () -> (any MTLBuffer)? in
				let argumentEncoder = encodeKernelFunction.makeArgumentEncoder(bufferIndex: 0)
				guard let buffer = asserted(device.makeBuffer(length: argumentEncoder.encodedLength, options: [.storageModeShared])) else { return nil }
				buffer.label = "Radix Sort / Encode Argument 0 Buffer"
				argumentEncoder.setArgumentBuffer(buffer, offset: 0)
				argumentEncoder.setIndirectCommandBuffer(indirectEncodeCommandBuffer, index: 0)
				argumentEncoder.setComputePipelineState(localPipelineState, index: 1)
				computePrefixSums.encodePipelineStates(using: argumentEncoder, indices: (local: 2, global: 3))
				argumentEncoder.setComputePipelineState(globalPipelineState, index: 4)
				argumentEncoder.setComputePipelineState(finalCopyPipelineState, index: 5)
				argumentEncoder.constantData(at: 6).bindMemory(to: UInt16.self, capacity: 1).pointee = threadgroupSize
				argumentEncoder.constantData(at: 7).bindMemory(to: UInt8.self, capacity: 1).pointee = UInt8(globalPipelineState.threadExecutionWidth) // TODO: Do I need to take the `min(…)` across pipeline states here as well?
				return buffer
			}())
		else { return nil }

		self.nWay = nWay
		self._computeBitwiseAndOr = computeBitwiseAndOr
		self._computePrefixSums = computePrefixSums
		self._pipelineStates = (encodePipelineState, localPipelineState, globalPipelineState, finalCopyPipelineState)
		self._indirectEncodeCommandBuffer = indirectEncodeCommandBuffer
		self._encodeArgument0Buffer = encodeArgument0Buffer
		self._maxCommandCount = maxCommandCount
		self.maxTotalThreadsPerThreadgroup = Int(threadgroupSize)
	}

	private func _additionalResultsCount(elementsCount: Int, threadgroupSize: Int! = nil) -> (actual: Int, padded: Int) {
		let threadgroupSize = threadgroupSize ?? self.maxTotalThreadsPerThreadgroup
		let result = additionalResultsCount(previousElementsCount: .init(elementsCount), threadgroupSize: .init(threadgroupSize))
		return (actual: Int(result.x), padded: Int(result.y))
	}

	private func _auxiliaryBufferSize(for elementsCount: Int, threadgroupSize: Int! = nil) -> (Int, orOffset: Int, uniformsOffset: Int, radixMasksOffset: Int) {
		let threadgroupSize = threadgroupSize ?? self.maxTotalThreadsPerThreadgroup
		let ownAuxiliaryBufferSize = self.nWay * Int(auxiliaryBufferSize(elementsCount: .init(elementsCount), threadgroupSize: .init(threadgroupSize)))
		let bitwiseAndOrResultsBufferSize = self._computeBitwiseAndOr.resultsBufferSize(for: elementsCount, threadgroupSize: threadgroupSize)

		let log2nWay = self.nWay.trailingZeroBitCount
		let uniformsSize = ((((32 - 1) / log2nWay + 1) * MemoryLayout<ComputeRadixSortUniforms>.stride - 1) / 16 + 1) * 16
		let radixMasksSize = ((((32 - 1) / log2nWay + 1) * MemoryLayout<UInt32>.stride - 1) / 16 + 1) * 16
		let bufferSize = max(ownAuxiliaryBufferSize, 2 * bitwiseAndOrResultsBufferSize) + /*checkSortedFinalResultSize + */uniformsSize + radixMasksSize
		let radixMasksOffset = bufferSize - radixMasksSize
		let uniformsOffset = radixMasksOffset - uniformsSize
		return (bufferSize, orOffset: bitwiseAndOrResultsBufferSize, uniformsOffset, radixMasksOffset)
	}

	enum Buffers {
		/// Values only, of type`UInt32`.
		case values(any MTLBuffer)
		/// Keys of type `UInt32` and arbitrary values with stride `valuesStride` (in bytes).
		case keysAndValues(keys: any MTLBuffer, values: any MTLBuffer, valuesStride: Int)
	}

	func encode(
		to commandBuffer: any MTLCommandBuffer,
		inoutBuffers: Buffers
	) {
		let elementsCount = inoutBuffers._elementsCount

		// Trivial cases
		if elementsCount <= 1 { return }

		let threadgroupSize = self.maxTotalThreadsPerThreadgroup

		let (auxiliaryBuffersSize, auxiliaryBufferOrOffset, auxiliaryBufferUniformsOffset, auxiliaryBufferRadixMasksOffset)
			= self._auxiliaryBufferSize(for: elementsCount, threadgroupSize: threadgroupSize)
		guard
			let doubledBuffers = asserted(inoutBuffers._makeDoubledBuffers(using: commandBuffer.device)),
			let auxiliaryBuffer = asserted({ () -> (any MTLBuffer)? in
				guard let buffer = asserted(commandBuffer.device.makeBuffer(length: auxiliaryBuffersSize, options: [.storageModePrivate])) else { return nil }
				buffer.label = "Radix Sort / Auxiliary"
				return buffer
			}()),
			let bitwiseAndOrResultOffsets = asserted(self._computeBitwiseAndOr.encode(to: commandBuffer, buffer: inoutBuffers._keys, andResultsBuffer: auxiliaryBuffer, orResultsBuffer: auxiliaryBuffer, orResultsBuffersOffset: auxiliaryBufferOrOffset))
		else { return }

		guard let encodeCommandEncoder = asserted(commandBuffer.makeComputeCommandEncoder()) else { return }
		encodeCommandEncoder.label = "Radix Sort / Encode"
		encodeCommandEncoder.setComputePipelineState(self._pipelineStates.encode)
		encodeCommandEncoder.useResource(self._indirectEncodeCommandBuffer, usage: .write)
		encodeCommandEncoder.setBuffer(self._encodeArgument0Buffer, offset: 0, index: 0)
		encodeCommandEncoder.setBuffer(inoutBuffers._keys, offset: 0, index: 1)
		encodeCommandEncoder.setBuffer(inoutBuffers._vals, offset: 0, index: 2)
		encodeCommandEncoder.setBuffer(doubledBuffers._keys, offset: 0, index: 3)
		encodeCommandEncoder.setBuffer(doubledBuffers._vals, offset: 0, index: 4)
		encodeCommandEncoder.setBuffer(auxiliaryBuffer, offset: 0, index: 5)
		var uniforms = ComputeRadixSortEncodeUniforms(
			bitwiseAndResultOffset: UInt32(bitwiseAndOrResultOffsets.and),
			bitwiseOrResultOffset: UInt32(bitwiseAndOrResultOffsets.or),
			uniformsOffset: UInt32(auxiliaryBufferUniformsOffset),
			radixMasksOffset: UInt32(auxiliaryBufferRadixMasksOffset),
			elementsCount: UInt32(elementsCount),
			valuesStride: UInt16(inoutBuffers._valuesStride),
		)
		encodeCommandEncoder.setBytes(&uniforms, length: MemoryLayout.stride(ofValue: uniforms), index: 6)
		encodeCommandEncoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
		encodeCommandEncoder.endEncoding()

		guard let executeCommandEncoder = asserted(commandBuffer.makeComputeCommandEncoder()) else { return }
		executeCommandEncoder.label = "Radix Sort / Execute"
		executeCommandEncoder.useResource(self._indirectEncodeCommandBuffer, usage: .read)
		if case .keysAndValues(keys: let buffer, _, _) = inoutBuffers {
			executeCommandEncoder.useResource(buffer, usage: [.read, .write])
		}
		executeCommandEncoder.useResource(inoutBuffers._vals, usage: [.read, .write])
		if case .keysAndValues(keys: let buffer, _, _) = doubledBuffers {
			executeCommandEncoder.useResource(buffer, usage: [.read, .write])
		}
		executeCommandEncoder.useResource(doubledBuffers._vals, usage: [.read, .write])
		executeCommandEncoder.useResource(auxiliaryBuffer, usage: [.read, .write])
		executeCommandEncoder.executeCommandsInBuffer(self._indirectEncodeCommandBuffer, range: 0..<self._maxCommandCount)
		executeCommandEncoder.endEncoding()
	}
}

fileprivate extension ComputeRadixSort.Buffers {

	var _elementsCount: Int {
		assert(self._keys.length % MemoryLayout<UInt32>.stride == 0, "Buffer length must be a multiple of 32-bit integer size")
		let elementsCount = self._keys.length / MemoryLayout<UInt32>.stride
		if case .keysAndValues(_, let valuesBuffer, let valuesStride) = self {
			assert(elementsCount * valuesStride <= valuesBuffer.length, "Values buffer length must be a at least \(elementsCount * valuesStride) byte\(elementsCount * valuesStride == 1 ? "" : "s")")
		}
		return elementsCount
	}

	var _keys: any MTLBuffer {
		switch self {
		case .values(let buffer): return buffer
		case .keysAndValues(let buffer, _, _): return buffer
		}
	}

	var _vals: any MTLBuffer {
		switch self {
		case .values(let buffer): return buffer
		case .keysAndValues(_, values: let buffer, _): return buffer
		}
	}

	func _makeDoubledBuffers(using device: any MTLDevice) -> Self? {
		switch self {
		case .values(let values):
			guard let doubledValues = asserted(device.makeBuffer(length: values.length, options: [])) else { return nil }
			doubledValues.label = "Radix Sort / Doubled Values"
			return .values(doubledValues)
		case let .keysAndValues(keys: keys, values: values, valuesStride: valuesStride):
			guard
				let doubledKeys = asserted(device.makeBuffer(length: keys.length, options: [])),
				let doubledValues = asserted(device.makeBuffer(length: values.length, options: []))
			else { return nil }
			doubledKeys.label = "Radix Sort / Doubled Keys"
			doubledValues.label = "Radix Sort / Doubled Values"
			return .keysAndValues(keys: doubledKeys, values: doubledValues, valuesStride: valuesStride)
		}
	}

	var _valuesStride: Int {
		switch self {
		case .values: return 0
		case .keysAndValues(keys: _, values: _, valuesStride: let valuesStride): return valuesStride
		}
	}
}
