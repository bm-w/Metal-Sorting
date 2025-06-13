//
//  PrefixSums.metal
//

#include <metal_stdlib>
using namespace metal;

#include "Shaders.h"


[[ kernel ]]
void compute_prefix_sums_local(
	device uint *inout [[ buffer(0) ]],
	uint gid [[ thread_position_in_grid ]],
	ushort tid [[ thread_position_in_threadgroup ]],
	ushort sgid [[ simdgroup_index_in_threadgroup ]],
	ushort sgn [[ simdgroups_per_threadgroup ]],
	ushort sgs [[ threads_per_simdgroup ]],
	uint tgid [[ threadgroup_position_in_grid ]],
	device uint *tg_sums [[ buffer(1) ]], // Per-threadroup sums
	threadgroup uint *sg_sums [[ threadgroup(0) ]]
) {
	uint val = inout[gid];
	uint prefix_sum = simd_prefix_exclusive_sum(val);
	uint sum = simd_sum(val);

	if (sgn > 1) {
		if (simd_is_first()) {
			sg_sums[sgid] = sum;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}

	for (uint s = 1, o = 0; sgn > s;) {
		uint n = (sgn - 1) / s + 1, ss = s * sgs, oo = o + n;
		if (tid < n) {
			ushort idx = o + tid;
			sum = sg_sums[idx];

			uint prefix_sum = simd_prefix_exclusive_sum(sum);
			sg_sums[idx] = prefix_sum;

			sum = simd_sum(sum);
			if (sgn > ss && simd_is_first()) {
				sg_sums[oo + sgid] = sum;
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);

		prefix_sum += sg_sums[o + sgid / s];

		s = ss, o = oo;
	}

	inout[gid] = prefix_sum;
	if (tid == 0) {
		tg_sums[tgid] = sum;
	}
}

[[ kernel ]]
void compute_prefix_sums_global(
	device uint *inout [[ buffer(0) ]], // Local prefix sums within threadgroups
	constant uint *aux [[ buffer(1) ]], // Global threadroup prefix sums
	uint gid [[ thread_position_in_grid ]],
	uint tgid [[ threadgroup_position_in_grid ]]
) {
	inout[gid] += aux[tgid];
}
