//
//  CheckSorted.metal
//

#include <metal_stdlib>
using namespace metal;


void reduce(bool is_sorted, ushort tid, ushort sgid, ushort sgn, ushort sgs, uint tgid, device bool *tg_is_sorted, threadgroup bool *sg_is_sorted);

[[ kernel ]]
void compute_check_sorted(
	constant uint *keys [[ buffer(0) ]],
	const uint gid [[ thread_position_in_grid ]],
	const ushort tid [[ thread_position_in_threadgroup ]],
	const ushort sgid [[ simdgroup_index_in_threadgroup ]],
	const ushort sgn [[ simdgroups_per_threadgroup ]],
	const ushort sgs [[ threads_per_simdgroup ]],
	const uint tgid [[ threadgroup_position_in_grid ]],
	constant uint &key_mask [[ buffer(1) ]],
	device bool *tg_is_sorted [[ buffer(2) ]],
	threadgroup bool *sg_is_sorted [[ threadgroup(0) ]]
) {
	uint key = keys[gid] & key_mask;
	uint prev_key = select(key, keys[gid - 1] & key_mask, gid > 0);
	bool is_sorted = prev_key <= key;

	reduce(is_sorted, tid, sgid, sgn, sgs, tgid, tg_is_sorted, sg_is_sorted);
}

[[ kernel ]]
void compute_check_sorted_reduce(
	device bool *tg_is_sorted [[ buffer(0) ]],
	const uint gid [[ thread_position_in_grid ]],
	const uint gn [[ threads_per_grid ]],
	const ushort tid [[ thread_position_in_threadgroup ]],
	const ushort sgid [[ simdgroup_index_in_threadgroup ]],
	const ushort sgn [[ simdgroups_per_threadgroup ]],
	const ushort sgs [[ threads_per_simdgroup ]],
	const uint tgid [[ threadgroup_position_in_grid ]],
	threadgroup bool *sg_is_sorted [[ threadgroup(0) ]]
) {
	bool is_sorted = tg_is_sorted[gid];
	uint o = ((gn - 1) / 16 + 1) * 16;
	reduce(is_sorted, tid, sgid, sgn, sgs, tgid, &tg_is_sorted[o], sg_is_sorted);
}

void reduce(
	bool is_sorted,
	const ushort tid,
	const ushort sgid,
	const ushort sgn,
	const ushort sgs,
	const uint tgid,
	device bool *tg_is_sorted,
	threadgroup bool *sg_is_sorted
) {
	is_sorted = simd_all(is_sorted);
	if (simd_is_first()) {
		sg_is_sorted[sgid] = is_sorted;
	}

	for (uint s = 1, o = 0; sgn > s;) {
		threadgroup_barrier(mem_flags::mem_threadgroup);

		uint n = (sgn - 1) / s + 1, ss = s * sgs, oo = o + n;
		if (tid < n) {
			is_sorted = sg_is_sorted[o + tid];
			is_sorted = simd_all(is_sorted);

			if (sgn > ss && simd_is_first()) {
				sg_is_sorted[oo + sgid] = is_sorted;
			}
		}
		s = ss, o = oo;
	}

	if (tid == 0) {
		if (sgn > 1) {
			tg_is_sorted[tgid] = is_sorted;
		} else {
			// Write the final results 4 times to ensure alignment
			*(device bool4 *)(&tg_is_sorted[tgid]) = bool4(is_sorted);
		}
	}
}
