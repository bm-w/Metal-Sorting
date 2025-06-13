//
//  BitwiseAndOr.metal
//

#include <metal_stdlib>
using namespace metal;


/// Computes bitwise-AND and -OR of resp. `and_keys` and `or_keys`, and stores
/// threadgroup results in `tg_and` and `tg_or`, resp.. The keys are separated
/// to support parallel reduction (they’ll be the same in the first iteration,
/// but in subsequent iterations they’ll be the previous intermediate results).
///
/// The final results can be passed to a single bitwise-XOR operation following
/// the following basic formula, where `bAND`, `bXOR`, and `bOR` are resp. the
/// `&` (bitwise-AND) `^` (bitwise-XOR), and `|` (bitwise-OR) operations:
///
/// `(bAND of all and_keys) bXOR (bOR of all or_keys)
///
/// The result of that final bitwise-XOR operation, `tg_and[fid] ^ tg_or[fid]` (where
/// `fid` is the index of the final parallel-reduced result), will have a `0`
/// bit where all bits in that position are identical, and a `1` bit where at
/// least one bit in that position is distinct.
///
/// The subsequent radix sort steps can be skipped whenever the radix mask only
/// has `0` bits the corresponding positions.
[[ kernel ]]
void compute_bitwise_and_or(
	constant uint *and_keys [[ buffer(0) ]],
	constant uint *or_keys [[ buffer(1) ]],
	uint gid [[ thread_position_in_grid ]],
	ushort tid [[ thread_position_in_threadgroup ]],
	ushort sgid [[ simdgroup_index_in_threadgroup ]],
	ushort sgn [[ simdgroups_per_threadgroup ]],
	ushort sgs [[ threads_per_simdgroup ]],
	uint tgid [[ threadgroup_position_in_grid ]],
	device uint *tg_and [[ buffer(2) ]],
	device uint *tg_or [[ buffer(3) ]],
	threadgroup uint *sg_and [[ threadgroup(0) ]],
	threadgroup uint *sg_or [[ threadgroup(1) ]]
) {
	uint and_key = and_keys[gid];
	uint or_key = or_keys[gid];
	and_key = simd_and(and_key);
	or_key = simd_or(or_key);

	if (simd_is_first()) {
		sg_and[sgid] = and_key;
		sg_or[sgid] = or_key;
	}

	for (uint s = 1, o = 0; sgn > s;) {
		threadgroup_barrier(mem_flags::mem_threadgroup);

		uint n = (sgn - 1) / s + 1, ss = s * sgs, oo = o + n;
		if (tid < n) {
			and_key = sg_and[o + tid];
			or_key = sg_or[o + tid];
			and_key = simd_and(and_key);
			or_key = simd_or(or_key);

			if (sgn > ss && simd_is_first()) {
				sg_and[oo + sgid] = and_key;
				sg_or[oo + sgid] = or_key;
			}
		}
		s = ss, o = oo;
	}

	if (tid == 0) {
		tg_and[tgid] = and_key;
		tg_or[tgid] = or_key;
	}
}
