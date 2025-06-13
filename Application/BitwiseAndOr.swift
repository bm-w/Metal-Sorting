//
//  BitwiseAndOr.swift
//

import Metal


struct ComputeBitwiseAndOr {
	private let _pipelineState: any MTLComputePipelineState

	var maxTotalThreadsPerThreadgroup: Int { self._pipelineState.maxTotalThreadsPerThreadgroup }

	init?(device: any MTLDevice, library: any MTLLibrary) {
		guard let function = asserted(library.makeFunction(name: "compute_bitwise_and_or")) else { return nil }
		do { self._pipelineState = try device.makeComputePipelineState(function: function) }
		catch { assertionFailure(String(describing: error)); return nil }
	}

	private func _additionalResultsBufferCount(elementsCount: Int, threadgroupSize: Int! = nil) -> (actual: Int, padded: Int) {
		let threadgroupSize = threadgroupSize ?? self._pipelineState.maxTotalThreadsPerThreadgroup
		let actualCount = (elementsCount - 1) / threadgroupSize + 1

		// Padding up to the next multiple of 4 elements (16 bytes), which is a
		// requirement for `….setBuffer(…, offset: …, …)`.
		return (actualCount, ((actualCount - 1) / 4 + 1) * 4)
	}

	func resultsBufferSize(for elementsCount: Int, threadgroupSize: Int! = nil) -> Int {
		let threadgroupSize = threadgroupSize ?? self._pipelineState.maxTotalThreadsPerThreadgroup
		var padded = 0
		var prevElementsCount = elementsCount
		while true {
			let (actualAdditonal, paddedAdditional) = self._additionalResultsBufferCount(elementsCount: prevElementsCount, threadgroupSize: threadgroupSize)
			prevElementsCount = actualAdditonal
			padded += paddedAdditional
			if actualAdditonal == 1 { break }
		}
		return padded * MemoryLayout<UInt32>.stride
	}

	private func _paddedSIMDResultsBufferSize(for threadgroupSize: Int, _ simdGroupSize: Int) -> Int {
		var count = 0
		var additional = threadgroupSize;
		repeat {
			additional = (additional - 1) / simdGroupSize + 1
			count += additional
		} while additional > 1
		return ((count * MemoryLayout<UInt32>.stride - 1) / 16 + 1) * 16
	}

	func encode(
		to commandBuffer: any MTLCommandBuffer,
		buffer: any MTLBuffer,
		andResultsBuffer: any MTLBuffer,
		andResultsBuffersOffset: Int = 0,
		orResultsBuffer: any MTLBuffer,
		orResultsBuffersOffset: Int = 0,
	) -> (and: Int, or: Int)? {
		let elementsCount = buffer.length / MemoryLayout<UInt32>.stride
		let resultsBuffersSize = self.resultsBufferSize(for: elementsCount)
		guard
			andResultsBuffer.length - andResultsBuffersOffset >= resultsBuffersSize,
			orResultsBuffer.length - orResultsBuffersOffset >= resultsBuffersSize
		else { return nil }

		return self._encode(to: commandBuffer, andBuffer: buffer, orBuffer: buffer, elementsCount: elementsCount, andResultsBuffer: andResultsBuffer, andResultsBuffersOffset: andResultsBuffersOffset, orResultsBuffer: orResultsBuffer, orResultsBuffersOffset: orResultsBuffersOffset)
	}

	private func _encode(
		to commandBuffer: any MTLCommandBuffer,
		iteration: Int = 0,
		andBuffer: any MTLBuffer,
		andBufferOffset: Int = 0,
		orBuffer: any MTLBuffer,
		orBufferOffset: Int = 0,
		elementsCount: Int,
		andResultsBuffer: any MTLBuffer,
		andResultsBuffersOffset: Int,
		orResultsBuffer: any MTLBuffer,
		orResultsBuffersOffset: Int,
	) -> (and: Int, or: Int)? {
		let threadgroupSize = self._pipelineState.maxTotalThreadsPerThreadgroup
		let simdGroupSize = self._pipelineState.threadExecutionWidth

		guard let commandEncoder = asserted(commandBuffer.makeComputeCommandEncoder()) else { return nil }
		commandEncoder.label = "Bitwise AND/OR (\(iteration): \(elementsCount) elements)"
		commandEncoder.setComputePipelineState(self._pipelineState)
		commandEncoder.setBuffer(andBuffer, offset: andBufferOffset, index: 0)
		commandEncoder.setBuffer(orBuffer, offset: orBufferOffset, index: 1)
		commandEncoder.setBuffer(andResultsBuffer, offset: andResultsBuffersOffset, index: 2)
		commandEncoder.setBuffer(orResultsBuffer, offset: orResultsBuffersOffset, index: 3)
		let simdGoupResultsBufferSize = self._paddedSIMDResultsBufferSize(for: threadgroupSize, simdGroupSize)
		commandEncoder.setThreadgroupMemoryLength(simdGoupResultsBufferSize, index: 0)
		commandEncoder.setThreadgroupMemoryLength(simdGoupResultsBufferSize, index: 1)
		commandEncoder.dispatchThreads(MTLSize(width: elementsCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
		commandEncoder.endEncoding()

		let threadgroupsCount = (elementsCount - 1) / threadgroupSize + 1
		if threadgroupsCount == 1 { return (andResultsBuffersOffset, orResultsBuffersOffset) } // Done!

		let andOffset = self._additionalResultsBufferCount(elementsCount: elementsCount, threadgroupSize: threadgroupSize).padded * MemoryLayout<UInt32>.stride
		let orOffset = self._additionalResultsBufferCount(elementsCount: elementsCount, threadgroupSize: threadgroupSize).padded * MemoryLayout<UInt32>.stride
		return self._encode(to: commandBuffer, iteration: iteration + 1, andBuffer: andResultsBuffer, andBufferOffset: andResultsBuffersOffset, orBuffer: orResultsBuffer, orBufferOffset: orResultsBuffersOffset, elementsCount: threadgroupsCount, andResultsBuffer: andResultsBuffer, andResultsBuffersOffset: andResultsBuffersOffset + andOffset, orResultsBuffer: orResultsBuffer, orResultsBuffersOffset: orResultsBuffersOffset + orOffset)
	}
}
