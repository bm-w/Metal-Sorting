//
//  CheckSorted.swift
//

import Metal


struct ComputeCheckSorted {
	private let _pipelineStates: (
		check: any MTLComputePipelineState,
		reduce: any MTLComputePipelineState
	)

	var maxTotalThreadsPerThreadgroup: Int { min(
		self._pipelineStates.check.maxTotalThreadsPerThreadgroup,
		self._pipelineStates.reduce.maxTotalThreadsPerThreadgroup,
	) }

	init?(device: any MTLDevice, library: any MTLLibrary, supportIndirectCommandBuffers: Bool = false) {
		guard
			let checkPipelineState = asserted({ () -> (any MTLComputePipelineState)? in
				guard let function = asserted(library.makeFunction(name: "compute_check_sorted")) else { return nil }
				let descriptor = MTLComputePipelineDescriptor()
				descriptor.label = "Check Sorted"
				descriptor.computeFunction = function
				descriptor.supportIndirectCommandBuffers = supportIndirectCommandBuffers
				do { return try device.makeComputePipelineState(descriptor: descriptor, options: []).0 }
				catch { assertionFailure(String(describing: error)); return nil }
			}()),
			let reducePipelineState = asserted({ () -> (any MTLComputePipelineState)? in
				guard let function = asserted(library.makeFunction(name: "compute_check_sorted_reduce")) else { return nil }
				let descriptor = MTLComputePipelineDescriptor()
				descriptor.label = "Check Sorted / Reduce"
				descriptor.computeFunction = function
				descriptor.supportIndirectCommandBuffers = supportIndirectCommandBuffers
				do { return try device.makeComputePipelineState(descriptor: descriptor, options: []).0 }
				catch { assertionFailure(String(describing: error)); return nil }
			}())
		else { return nil }
		self._pipelineStates = (checkPipelineState, reducePipelineState)
	}

	func encodePipelineStates(using argumentEncoder: any MTLArgumentEncoder, indices: (check: Int, reduce: Int)) {
		argumentEncoder.setComputePipelineState(self._pipelineStates.check, index: indices.check)
		argumentEncoder.setComputePipelineState(self._pipelineStates.reduce, index: indices.reduce)
	}

	/// Returns a sufficient size in bytes for a results buffer to contain the
	/// results of all threadgroups themselves, plus those of all groups of
	/// threadgroups if the number of threadgroups exceeds the number of threads
	/// per threadgroup, plus those of all groups of groups of threadgroups, etc.,
	/// such that the final result can finally be written to the very end of the buffer.
	func resultsBufferSize(for elementsCount: Int, threadgroupSize: Int! = nil) -> Int {
		if elementsCount == 0 { return 0 }

		let checkThreadgroupSize = threadgroupSize ?? self._pipelineStates.check.maxTotalThreadsPerThreadgroup
		let reduceThreadgroupSize = threadgroupSize ?? self._pipelineStates.reduce.maxTotalThreadsPerThreadgroup

		var neededAuxiliaryCount = (elementsCount - 1) / checkThreadgroupSize + 1
		if neededAuxiliaryCount == 1 { return MemoryLayout<Bool>.stride }

		var providedAuxiliaryCount = 0
		let elementsPer16Bytes = 16; assert(MemoryLayout<Bool>.stride == 1)
		while true {
			neededAuxiliaryCount = ((neededAuxiliaryCount - 1) / elementsPer16Bytes + 1) * elementsPer16Bytes
			let additionalAuxiliaryCount = (neededAuxiliaryCount - providedAuxiliaryCount - 1) / reduceThreadgroupSize + 1
			providedAuxiliaryCount = neededAuxiliaryCount
			neededAuxiliaryCount += additionalAuxiliaryCount
			if additionalAuxiliaryCount == 1 { break }
		}

		return neededAuxiliaryCount * MemoryLayout<Bool>.stride
	}

	private func _paddedSIMDResultsBufferSize(for threadgroupSize: Int, _ simdGroupSize: Int) -> Int {
		var count = 0
		var additional = threadgroupSize;
		repeat {
			additional = (additional - 1) / simdGroupSize + 1
			count += additional
		} while additional > 1
		return ((count * MemoryLayout<Bool>.stride - 1) / 16 + 1) * 16
	}

	/// Returns the offset of the final result in the `resultsBuffer`, or `nil` if
	/// the results could not be computed.
	func encode(
		to commandBuffer: any MTLCommandBuffer,
		buffer: any MTLBuffer,
		mask: UInt32 = .max,
		resultsBuffer: any MTLBuffer,
		resultsBufferOffset: Int = 0,
	) -> Int? {
		let elementsCount = buffer.length / MemoryLayout<UInt32>.stride
		let resultsBufferSize = self.resultsBufferSize(for: elementsCount)
		guard resultsBuffer.length - resultsBufferOffset >= resultsBufferSize else { return nil }

		return self._encode(to: commandBuffer, buffer: buffer, elementsCount: elementsCount, mask: mask, resultsBuffer: resultsBuffer, resultsBufferOffset: resultsBufferOffset)
	}

	private func _encode(
		to commandBuffer: any MTLCommandBuffer,
		buffer: any MTLBuffer,
		elementsCount: Int,
		mask: UInt32,
		resultsBuffer: any MTLBuffer,
		resultsBufferOffset: Int,
	) -> Int? {
		let threadgroupSize = self._pipelineStates.check.maxTotalThreadsPerThreadgroup
		let simdGroupSize = self._pipelineStates.check.threadExecutionWidth

		guard let commandEncoder = asserted(commandBuffer.makeComputeCommandEncoder()) else { return nil }
		commandEncoder.setComputePipelineState(self._pipelineStates.check)
		commandEncoder.setBuffer(buffer, offset: 0, index: 0)
		var mask = UInt32(mask)
		commandEncoder.setBytes(&mask, length: MemoryLayout.stride(ofValue: mask), index: 1)
		commandEncoder.setBuffer(resultsBuffer, offset: resultsBufferOffset, index: 2)
		commandEncoder.setThreadgroupMemoryLength(self._paddedSIMDResultsBufferSize(for: threadgroupSize, simdGroupSize), index: 0)
		commandEncoder.dispatchThreads(MTLSize(width: elementsCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
		commandEncoder.endEncoding()

		let threadgroupsCount = (elementsCount - 1) / threadgroupSize + 1
		if threadgroupsCount == 1 { return resultsBufferOffset } // Done!

		return self._encodeReduce(to: commandBuffer, elementsCount: threadgroupsCount, resultsBuffer: resultsBuffer, resultsBufferOffset: resultsBufferOffset)
	}

	private func _encodeReduce(to commandBuffer: any MTLCommandBuffer, elementsCount: Int, resultsBuffer: any MTLBuffer, resultsBufferOffset: Int) -> Int? {
		let threadgroupSize = self._pipelineStates.reduce.maxTotalThreadsPerThreadgroup
		let simdGroupSize = self._pipelineStates.reduce.threadExecutionWidth

		guard let commandEncoder = asserted(commandBuffer.makeComputeCommandEncoder()) else { return nil }
		commandEncoder.setComputePipelineState(self._pipelineStates.reduce)
		commandEncoder.setBuffer(resultsBuffer, offset: resultsBufferOffset, index: 0)
		commandEncoder.setThreadgroupMemoryLength(self._paddedSIMDResultsBufferSize(for: threadgroupSize, simdGroupSize), index: 0)
		commandEncoder.dispatchThreads(MTLSize(width: elementsCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
		commandEncoder.endEncoding()

		let threadgroupsCount = (elementsCount - 1) / threadgroupSize + 1
		if threadgroupsCount == 1 { return resultsBufferOffset + elementsCount } // Done!

		let offset = ((resultsBufferOffset - 1) / 16 + 1) * 16 // Round to multiple of 16 bytes
		return self._encodeReduce(to: commandBuffer, elementsCount: threadgroupsCount, resultsBuffer: resultsBuffer, resultsBufferOffset: resultsBufferOffset + offset)
	}
}
