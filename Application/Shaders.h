//
//  Shaders.h
//

#ifndef Shaders_h
#define Shaders_h

#ifdef __METAL_VERSION__
#define _SWIFT_NAME(name_) __attribute__((swift_name(#name_)))
#else
#include <simd/simd.h>
@import Foundation;
#define _SWIFT_NAME(name) NS_SWIFT_NAME(name)
typedef unsigned char uchar;
typedef simd_uint2 uint2;
#endif


struct MainStepFragmentSortingUniforms {
	unsigned long p, k;
};

struct MainFragmentUniforms {
	unsigned long frame_idx _SWIFT_NAME(frameIndex);
};

struct ComputeRadixSortEncodeUniforms {
	uint bitwise_and_result_offset _SWIFT_NAME(bitwiseAndResultOffset);
	uint bitwise_or_result_offset _SWIFT_NAME(bitwiseOrResultOffset);
	uint uniforms_offset _SWIFT_NAME(uniformsOffset);
	uint radix_masks_offset _SWIFT_NAME(radixMasksOffset);
	uint num_elements _SWIFT_NAME(elementsCount);
	ushort val_size _SWIFT_NAME(valuesStride);
};

struct ComputeRadixSortUniforms {
	ushort val_size _SWIFT_NAME(valuesStride);
	uchar radix_offset _SWIFT_NAME(radixOffset);
};

#ifdef __METAL_VERSION__
template<typename T> T simd_bitonic_key_sort(
	T val,
	thread uint& key,
	ushort lane_id,
	ushort lanes_width
);
uint simd_bitonic_sort(uint val, ushort simd_lane_id, ushort simd_lanes_width);

template<typename T> void bitonic_key_sort(
	threadgroup T *vals,
	threadgroup uint *keys,
	ushort thread_id,
	ushort threads_width,
	ushort simd_lanes_width
);
void bitonic_sort(threadgroup uint *vals, ushort thread_id, ushort threads_width, ushort simd_lanes_width);
#endif

/// Returns resp. actual and padded numbers of additional auxliary elements.
static inline uint2 num_add_aux(uint prev_num_elems, ushort threadgroup_size) _SWIFT_NAME(additionalResultsCount(previousElementsCount:threadgroupSize:)) {
	uint tgn = (prev_num_elems - 1) / threadgroup_size + 1;

	// Padding up to the next multiple of 16. Technically it needs to be to the
	// next multiple of 16 bytes (a requirement of `….setBuffer(…, offset: …, …)`,
	// and so the count padding may be tighter, but the difference should be
	// minimal or even zero in most cases.
	return (uint2){ tgn, ((tgn - 1) / 16 + 1) * 16 };
}

static inline uint aux_buffer_size(uint num_elems, ushort threadgroup_size) _SWIFT_NAME(auxiliaryBufferSize(elementsCount:threadgroupSize:)) {
	uint padded = 0;
	uint prev_num_elems = num_elems;
	while (true) {
		uint2 add = num_add_aux(prev_num_elems, threadgroup_size);
		prev_num_elems = add.x;
		padded += add.y;
		if (add.x == 1) { break; }
	}
	return padded * sizeof(uint);
}


//private func _additionalResultsCount(elementsCount: Int, threadgroupSize: Int! = nil) -> (actual: Int, padded: Int) {
//	   let threadgroupSize = threadgroupSize ?? self.maxTotalThreadsPerThreadgroup
//	   let threadgroupsCount = (elementsCount - 1) / threadgroupSize + 1
//
//	   // Padding up to the next multiple of 16. Technically it needs to be to the
//	   // next multiple of 16 bytes (a requirement of `….setBuffer(…, offset: …, …)`,
//	   // and so the count padding may be tighter, but the difference should be
//	   // minimal or even zero in most cases.
//	   return (threadgroupsCount, ((threadgroupsCount - 1) / 16 + 1) * 16)
//   }
//
//   func resultsBuffersSize(for elementsCount: Int, threadgroupSize: Int! = nil) -> Int {
//	   let threadgroupSize = threadgroupSize ?? self.maxTotalThreadsPerThreadgroup
//	   var padded = 0
//	   var prevElementsCount = elementsCount
//	   while true {
//		   let (actualAdditonal, paddedAdditional) = self._additionalResultsCount(elementsCount: prevElementsCount, threadgroupSize: threadgroupSize)
//		   prevElementsCount = actualAdditonal
//		   padded += paddedAdditional
//		   if actualAdditonal == 1 { break }
//	   }
//	   return padded * MemoryLayout<UInt32>.stride
//   }

#endif // Shaders_h
