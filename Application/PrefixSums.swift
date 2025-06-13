//
//  PrefixSums.swift
//

import Metal


struct ComputePrefixSums {
	private let _pipelineStates: (
		local: any MTLComputePipelineState,
		global: any MTLComputePipelineState
	)
	private let _dummyAuxiliaryBuffer: any MTLBuffer

	var maxTotalThreadsPerThreadgroup: Int { min(
		self._pipelineStates.local.maxTotalThreadsPerThreadgroup,
		self._pipelineStates.global.maxTotalThreadsPerThreadgroup,
	) }

	init?(device: any MTLDevice, library: any MTLLibrary, supportIndirectCommandBuffers: Bool = false) {
		guard
			let localPipelineState = asserted({ () -> (any MTLComputePipelineState)? in
				guard let function = asserted(library.makeFunction(name: "compute_prefix_sums_local")) else { return nil }
				let descriptor = MTLComputePipelineDescriptor()
				descriptor.label = "Prefix Sums / Local"
				descriptor.computeFunction = function
				descriptor.supportIndirectCommandBuffers = supportIndirectCommandBuffers
				do { return try device.makeComputePipelineState(descriptor: descriptor, options: []).0 }
				catch { assertionFailure(String(describing: error)); return nil }
			}()),
			let globalPipelineState = asserted({ () -> (any MTLComputePipelineState)? in
				guard let function = asserted(library.makeFunction(name: "compute_prefix_sums_global")) else { return nil }
				let descriptor = MTLComputePipelineDescriptor()
				descriptor.label = "Prefix Sums / Global"
				descriptor.computeFunction = function
				descriptor.supportIndirectCommandBuffers = supportIndirectCommandBuffers
				do { return try device.makeComputePipelineState(descriptor: descriptor, options: []).0 }
				catch { assertionFailure(String(describing: error)); return nil }
			}()),
			let dummyAuxiliaryBuffer = asserted(device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: []))
		else { return nil}

		self._pipelineStates = (localPipelineState, globalPipelineState)
		self._dummyAuxiliaryBuffer = dummyAuxiliaryBuffer
	}

	func encodePipelineStates(using argumentEncoder: any MTLArgumentEncoder, indices: (local: Int, global: Int)) {
		argumentEncoder.setComputePipelineState(self._pipelineStates.local, index: indices.local)
		argumentEncoder.setComputePipelineState(self._pipelineStates.global, index: indices.global)
	}

	func auxiliaryBufferSize(for elementsCount: Int, threadgroupSize: Int! = nil) -> Int {
		if elementsCount == 0 { return 0 }

		let threadgroupSize = threadgroupSize ?? self.maxTotalThreadsPerThreadgroup

		var neededAuxiliaryCount = (elementsCount - 1) / threadgroupSize + 1
		if neededAuxiliaryCount == 1 { return MemoryLayout<UInt32>.stride }

		var providedAuxiliaryCount = 0
		let elementsPer16Bytes = 4; assert(MemoryLayout<UInt32>.stride == 4)
		while true {
			neededAuxiliaryCount = ((neededAuxiliaryCount - 1) / elementsPer16Bytes + 1) * elementsPer16Bytes
			let additionalAuxiliaryCount = (neededAuxiliaryCount - providedAuxiliaryCount - 1) / threadgroupSize + 1
			providedAuxiliaryCount = neededAuxiliaryCount
			neededAuxiliaryCount += additionalAuxiliaryCount
			if additionalAuxiliaryCount == 1 { break }
		}
		return neededAuxiliaryCount * MemoryLayout<UInt32>.stride
	}
}
