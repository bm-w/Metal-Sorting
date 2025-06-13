//
//  Bitonic.metal
//

#include <metal_stdlib>
using namespace metal;

#include "Shaders.h"


#pragma mark SIMD Group

template<typename T>
T simd_bitonic_key_sort(T val, thread uint& key, ushort lane_id, ushort lanes_width) {
	for (ushort d = 1, s = 2; d < lanes_width; d = s, s <<= 1) {
		ushort other_lane_id = lane_id ^ (s - 1);
		while (d > 0) {
			uint other_key = simd_shuffle(key, other_lane_id);
			T other_val = simd_shuffle(val, other_lane_id);
			bool should_swap = select(
				other_key > key,
				other_lane_id < lanes_width && key > other_key,
				lane_id < other_lane_id
			);
			key = select(key, other_key, should_swap);
			val = select(val, other_val, should_swap);
			other_lane_id = lane_id ^ (d >>= 1);
		}
	}
	return val;
}

// TODO: Find some way to avoid this specialization? (For whatever reason in
// Metal’s flavor of C++ specializations are not automatically generated.)
template char simd_bitonic_key_sort(char, thread uint&, ushort, ushort);

uint simd_bitonic_sort(uint val, ushort lane_id, ushort lanes_width) {
	return simd_bitonic_key_sort<uint>(val, val, lane_id, lanes_width);
}


template <typename T>
T simd_bitonic_key_sort_split(T val, thread uint& key, ushort lane_id, ushort lanes_width) {
	for (ushort d = lanes_width >> 1; d > 0; d >>= 1) {
		ushort other_lane_id = lane_id ^ d;
		uint other_key = simd_shuffle(key, other_lane_id);
		T other_val = simd_shuffle(val, other_lane_id);
		bool should_swap = select(
			other_key > key,
			other_lane_id < lanes_width && key > other_key,
			lane_id < other_lane_id
		);
		key = select(key, other_key, should_swap);
		val = select(val, other_val, should_swap);
	}
	return val;
}


#pragma mark - Thread Group

template<typename T>
void bitonic_key_sort(
	threadgroup T *vals,
	threadgroup uint *keys,
	const ushort thread_id,
	const ushort threads_width,
	const ushort simd_lanes_width
) {
	bool vals_is_keys = is_same_v<T, uint> && (threadgroup uint*)vals == keys;

	// The first few steps are done faster within SIMD groups
	T val = vals[thread_id];
	uint key = keys[thread_id];
	val = simd_bitonic_key_sort(val, key, thread_id % simd_lanes_width, simd_lanes_width);
	if (!vals_is_keys) { vals[thread_id] = val; }
	keys[thread_id] = key;

	// The remaining steps are done here in the thread group
	for (ushort d = simd_lanes_width, s = d << 1; d < threads_width; d = s, s <<= 1) {
		ushort other_thread_id = thread_id ^ (s - 1);

		// The initial merge step and the first few split steps that span multiple
		// SIMD groups are done here in the thread group
		while (d >= simd_lanes_width) {
			threadgroup_barrier(mem_flags::mem_threadgroup);

			if (other_thread_id < threads_width) {
				uint other_key = keys[other_thread_id];
				if (select(other_key > key, key > other_key, thread_id < other_thread_id)) {
					T other_val = vals[other_thread_id];

					// The lower index’ thread will handle the swap, if any
					if (thread_id < other_thread_id) {
						if (!vals_is_keys) {
							vals[other_thread_id] = val;
							vals[thread_id] = other_val;
						}
						keys[other_thread_id] = key;
						keys[thread_id] = other_key;
					}

					val = other_val;
					key = other_key;
				}
			}

			other_thread_id = thread_id ^ (d >>= 1);
		}

		// The remaining split steps are done faster in SIMD groups
		val = simd_bitonic_key_sort_split(val, key, thread_id % simd_lanes_width, simd_lanes_width);
		if (!vals_is_keys) { vals[thread_id] = val; }
		keys[thread_id] = key;
	}
}

template void bitonic_key_sort(threadgroup ushort*, threadgroup uint*, const ushort, const ushort, const ushort);

void bitonic_sort(
	threadgroup uint *vals,
	const ushort thread_id,
	const ushort threads_width,
	const ushort simd_lanes_width
) {
	bitonic_key_sort(vals, vals, thread_id, threads_width, simd_lanes_width);
}

[[ kernel ]]
void compute_bitonic_sort_local(
	device uint *vals [[ buffer(0) ]],
	threadgroup uint *temp_vals [[ threadgroup(0) ]],
	uint lid [[ thread_position_in_threadgroup ]],
	uint tgs [[ threads_per_threadgroup ]],
	uint sgs [[ threads_per_simdgroup ]]
) {
	temp_vals[lid] = vals[lid];
	bitonic_key_sort(temp_vals, temp_vals, lid, tgs, sgs);
	vals[lid] = temp_vals[lid];
}
