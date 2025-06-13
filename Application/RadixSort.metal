//
//  RadixSort.metal
//

#include <metal_stdlib>
using namespace metal;

#include "Shaders.h"

constant uchar R [[ function_constant(0) ]];


struct EncodeArg0 {
	command_buffer cmd_buf [[ id(0) ]];
	compute_pipeline_state sort_local_pipeline [[ id(1) ]];
	compute_pipeline_state prefix_sums_local_sums_pipeline [[ id(2) ]];
	compute_pipeline_state prefix_sums_global_pipeline [[ id(3) ]];
	compute_pipeline_state sort_global_pipeline [[ id(4) ]];
	compute_pipeline_state sort_final_copy_pipeline [[ id(5) ]];
	ushort tgs [[ id(6) ]]; // Threads per threadgroup
	uchar sgs [[ id(7) ]]; // Threads per SIMD group
};


[[ kernel ]]
void compute_radix_sort_encode(
	device EncodeArg0 &arg0 [[ buffer(0) ]],
	device uint *keys_a [[ buffer(1) ]],
	device void *vals_a [[ buffer(2) ]],
	device uint *keys_b [[ buffer(3) ]],
	device void *vals_b [[ buffer(4) ]],
	device uchar *aux [[ buffer(5) ]],
	constant ComputeRadixSortEncodeUniforms &uniforms [[ buffer(6) ]]
) {
	uint keys_and = *(device uint*)&aux[uniforms.bitwise_and_result_offset];
	uint keys_or = *(device uint*)&aux[uniforms.bitwise_or_result_offset];
	uint keys_mask = keys_and ^ keys_or;
	if (keys_mask == 0) return; // All keys are identical

	uchar low = ctz(keys_mask) / ctz(R);
	uchar high = (32 - clz(keys_mask) - 1) / ctz(R) + 1;
	uint downshifted_radix_mask = (0xFFFFFFFF >> (32 - ctz(R)));

	uint sg_aux_len = 0;
	{
		uint sg_aux_len_add = arg0.tgs;
		do { sg_aux_len += (sg_aux_len_add = (sg_aux_len_add - 1) / arg0.sgs + 1); }
		while (sg_aux_len_add > 1);
	}
	sg_aux_len *= sizeof(uint);
	uint tgn = (uniforms.num_elements - 1) / arg0.tgs + 1;

	uchar k = 0, c = 0;
	for (uchar i = low; i < high; i++) {
		uint radix_mask = downshifted_radix_mask << (i * ctz(R));
		if ((keys_mask & radix_mask) == 0) continue;

		device ComputeRadixSortUniforms *uniforms_buffer = (device ComputeRadixSortUniforms *)(&aux[uniforms.uniforms_offset]) + i;
		*uniforms_buffer = (ComputeRadixSortUniforms){ .radix_offset = i, .val_size = uniforms.val_size };
		device uint *radix_mask_buffer = (device uint *)(&aux[uniforms.radix_masks_offset]) + i;
		*radix_mask_buffer = radix_mask;

		device uint *keys_in = k % 2 == 0 ? keys_a : keys_b;
		device void *vals_in = k % 2 == 0 ? vals_a : vals_b;
		device uint *keys_out = k % 2 == 0 ? keys_b : keys_a;
		device void *vals_out = k % 2 == 0 ? vals_b : vals_a;
		k++;

		compute_command local_cmd(arg0.cmd_buf, c++);
		local_cmd.set_compute_pipeline_state(arg0.sort_local_pipeline);
		local_cmd.set_barrier();
		local_cmd.set_kernel_buffer(keys_in, 0);
		local_cmd.set_kernel_buffer(vals_in, 1);
		local_cmd.set_kernel_buffer(uniforms_buffer, 2);
		local_cmd.set_kernel_buffer((device uint*)aux, 3);
		local_cmd.set_threadgroup_memory_length(R * sg_aux_len, 0);
		local_cmd.concurrent_dispatch_threads(uint3(uniforms.num_elements, 1, 1), uint3(arg0.tgs, 1, 1));

		thread uint3 prefix_sum_global_distr_params[5] = { 0 };

		uint n = R * tgn, o = 0, j = 0;
		while (true) {
			uint tgn = (n - 1) / arg0.tgs + 1;

			compute_command prefix_sums_local_cmd(arg0.cmd_buf, c++);
			prefix_sums_local_cmd.set_compute_pipeline_state(arg0.prefix_sums_local_sums_pipeline);
			prefix_sums_local_cmd.set_barrier();
			prefix_sums_local_cmd.set_kernel_buffer((device uint*)&aux[o], 0);
			uint oo = o + ((sizeof(uint) * n - 1) / 16 + 1) * 16;
			prefix_sums_local_cmd.set_kernel_buffer((device uint*)&aux[oo], 1);
			prefix_sums_local_cmd.set_threadgroup_memory_length(sg_aux_len, 0);
			prefix_sums_local_cmd.concurrent_dispatch_threads(uint3(n, 1, 1), uint3(arg0.tgs, 1, 1));

			if (tgn == 1) break;

			prefix_sum_global_distr_params[j++] = uint3(n, o, oo);
			o = oo;
			n = tgn;
		}

		while (j--) {
			uint n = prefix_sum_global_distr_params[j].x;
			uint o = prefix_sum_global_distr_params[j].y;
			uint oo = prefix_sum_global_distr_params[j].z;

			compute_command prefix_sums_global_cmd(arg0.cmd_buf, c++);
			prefix_sums_global_cmd.set_compute_pipeline_state(arg0.prefix_sums_global_pipeline);
			prefix_sums_global_cmd.set_barrier();
			prefix_sums_global_cmd.set_kernel_buffer((device uint*)&aux[o], 0);
			prefix_sums_global_cmd.set_kernel_buffer((device uint*)&aux[oo], 1);
			prefix_sums_global_cmd.set_threadgroup_memory_length(sg_aux_len, 0);
			prefix_sums_global_cmd.concurrent_dispatch_threads(uint3(n, 1, 1), uint3(arg0.tgs, 1, 1));
		}

		compute_command global_cmd(arg0.cmd_buf, c++);
		global_cmd.set_compute_pipeline_state(arg0.sort_global_pipeline);
		global_cmd.set_barrier();
		global_cmd.set_kernel_buffer(keys_in, 0);
		global_cmd.set_kernel_buffer(vals_in, 1);
		global_cmd.set_kernel_buffer(uniforms_buffer, 2);
		global_cmd.set_kernel_buffer((device uint*)aux, 3);
		global_cmd.set_kernel_buffer(keys_out, 4);
		global_cmd.set_kernel_buffer(vals_out, 5);
		global_cmd.set_threadgroup_memory_length(R * sg_aux_len, 0);
		global_cmd.concurrent_dispatch_threads(uint3(uniforms.num_elements, 1, 1), uint3(arg0.tgs, 1, 1));
	}

	if (k % 2 != 0) {
		// Re-using `val_size`, the first field of `ComputeRadixSortUniforms`, from iteration `i = 0` above
		device ushort *val_size_buffer = (device ushort *)(&aux[uniforms.uniforms_offset]);
		compute_command global_distr_cmd(arg0.cmd_buf, c++);
		global_distr_cmd.set_compute_pipeline_state(arg0.sort_final_copy_pipeline);
		global_distr_cmd.set_barrier();
		global_distr_cmd.set_kernel_buffer(keys_b, 0);
		global_distr_cmd.set_kernel_buffer(vals_b, 1);
		global_distr_cmd.set_kernel_buffer(val_size_buffer, 2);
		global_distr_cmd.set_kernel_buffer(keys_a, 3);
		global_distr_cmd.set_kernel_buffer(vals_a, 4);
		global_distr_cmd.concurrent_dispatch_threads(uint3(uniforms.num_elements, 1, 1), uint3(arg0.tgs, 1, 1));
	}
}

void copy_val(uint from_gid, uint to_gid, constant ushort &val_size, constant void *vals_in, device void *vals_out);

[[ kernel ]]
void compute_radix_sort_local(
	constant uint *keys [[ buffer(0) ]],
	constant void *vals [[ buffer(1) ]],
	constant ComputeRadixSortUniforms &uniforms [[ buffer(2) ]],
	uint gid [[ thread_position_in_grid ]],
	ushort tid [[ thread_position_in_threadgroup ]],
	ushort sgid [[ simdgroup_index_in_threadgroup ]],
	ushort sgn [[ simdgroups_per_threadgroup ]],
	ushort sgs [[ threads_per_simdgroup ]],
	uint tgid [[ threadgroup_position_in_grid ]],
	uint tgn [[ threadgroups_per_grid ]],
	device uint *tg_sums [[ buffer(3) ]],
	threadgroup ushort *sg_sums [[ threadgroup(0) ]]
) {
//	static_assert(R > 1 && (R & (R - 1)) == 0, "`R` must be a power of 2 greater than 1");

//	TODO: `bool vals_is_keys = uniforms.val_size == 0 || (device uint*)vals == keys;`
	uint key = keys[gid];

	uchar radix_shift = uniforms.radix_offset * ctz(R);
	uint radix_mask = (0xFFFFFFFF >> (32 - ctz(R))) << radix_shift;

	uint radix_key = key & radix_mask;

//	TODO: Turns out the local sort needs to be stable (the sort is not strictly
//	required, therefore it’s commented out, but supposedly it speeds up	memory
//	access & cache performance later on).

//	other_gids[tid] = gid;
//	uint unsorted_radix_key = key & radix_mask;
//	radix_keys[tid] = unsorted_radix_key;
//
//	// Local sort, supposedly for better memory access & cache performance later on
//	bitonic_key_sort(other_gids, radix_keys, tid, tgs, sgs);
//	threadgroup_barrier(mem_flags::mem_threadgroup);

//	// `thread_id` and `other_thread_id` are both within the same threadgroup, so
//	// there is no race condition between reading from one and writing to another
//	ushort other_gid = other_gids[tid];
//	if (!vals_is_keys) { vals[other_gid] = val; }
//	keys[other_gid] = key;

	// Count the number of elements with each radix key
	uchar shifted_radix_key = radix_key >> radix_shift;
	for (uchar srk = 0; srk < R; srk++) {
		bool is_radix_key = shifted_radix_key == srk;
		uchar sum = simd_sum((uchar)is_radix_key);
		if (sgn <= 1) {
			if (tid == 0) {
				tg_sums[srk * tgn + tgid] = sum;
			}
		} else if (simd_is_first()) {
			sg_sums[srk * sgn + sgid] = sum;
		}
	}

	// Sum up the sums of each SIMD group (if `sgn > 1`), then of each group of
	// SIMD groups (if `sgn > sgs`), etc., to finally get the threadgroup sum.
	for (uint s = 1, o = 0, n = sgn; sgn > s;) {
		threadgroup_barrier(mem_flags::mem_threadgroup);

		s *= sgs;
		uint next_o = o + n * R, next_n = (sgn - 1) / s + 1;
		if (tid < n) {
			for (uchar srk = 0; srk < R; srk++) {
				ushort sum = sg_sums[o + srk * n + tid];
				sum = simd_sum(sum);

				if (sgn <= s) {
					if (tid == 0) {
						tg_sums[srk * tgn + tgid] = sum;
					}
				} else if (simd_is_first()) {
					sg_sums[next_o + srk * next_n + sgid] = sum;
				}
			}
		}
		o = next_o, n = next_n;
	}
}

[[ kernel ]]
void compute_radix_sort_global(
	constant uint *keys_in [[ buffer(0) ]],
	constant void *vals_in [[ buffer(1) ]],
	constant ComputeRadixSortUniforms &uniforms [[ buffer(2) ]],
	constant uint *tg_prefix_sums [[ buffer(3) ]],
	uint gid [[ thread_position_in_grid ]],
	ushort tid [[ thread_position_in_threadgroup ]],
	ushort sgid [[ simdgroup_index_in_threadgroup ]],
	ushort sgn [[ simdgroups_per_threadgroup ]],
	ushort sgs [[ threads_per_simdgroup ]],
	uint tgid [[ threadgroup_position_in_grid ]],
	uint tgn [[ threadgroups_per_grid ]],
	device uint *keys_out [[ buffer(4) ]],
	device void *vals_out [[ buffer(5) ]],
	threadgroup ushort *sg_prefix_sums [[ threadgroup(0) ]]
) {
//	static_assert(R > 1 && (R & (R - 1)) == 0, "`R` must be a power of 2 greater than 1");

	bool vals_is_keys = uniforms.val_size == 0 || (constant uint*)vals_in == keys_in;

	uchar radix_shift = uniforms.radix_offset * ctz(R);
	uint radix_mask = (0xFFFFFFFF >> (32 - ctz(R))) << radix_shift;

	uint key = keys_in[gid];
	uchar shifted_radix_key = (key & radix_mask) >> radix_shift;

	uint other_gid = 0;
	for (uchar srk = 0; srk < R; srk++) {
		bool is_radix_key = shifted_radix_key == srk;
		uchar prefix_sum = simd_prefix_exclusive_sum((uchar)is_radix_key);
		if (is_radix_key) {
			other_gid += prefix_sum;
		}
		uchar sum = simd_sum((uchar)is_radix_key);
		if (sgn > 1 && simd_is_first()) {
			sg_prefix_sums[srk * sgn + sgid] = sum;
		}
	}
	if (sgn > 1) {
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}

	for (uint s = 1, o = 0, n = sgn; sgn > s;) {
		uint ss = s * sgs, oo = o + n * R, nn = (sgn - 1) / ss + 1;
		if (tid < n) {
			for (uchar srk = 0; srk < R; srk++) {
				ushort idx = o + srk * n + tid;
				ushort sum = sg_prefix_sums[idx];
				ushort prefix_sum = simd_prefix_exclusive_sum(sum);
				sg_prefix_sums[idx] = prefix_sum;

				if (sgn > ss) {
					sum = simd_sum(sum);
					if (simd_is_first()) {
						sg_prefix_sums[oo + srk * nn + sgid] = sum;
					}
				}
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);

		other_gid += sg_prefix_sums[o + shifted_radix_key * n + sgid / s];

		s = ss, o = oo, n = nn;
	}

	other_gid += tg_prefix_sums[shifted_radix_key * tgn + tgid];
//	if (other_gid == gid) return;
	keys_out[other_gid] = key;
	if (!vals_is_keys) {
		copy_val(gid, other_gid, uniforms.val_size, vals_in, vals_out);
	}
}

[[ kernel ]]
void compute_radix_sort_final_copy(
	constant uint *keys_in [[ buffer(0) ]],
	constant void *vals_in [[ buffer(1) ]],
	constant ushort &val_size [[ buffer(2) ]],
	device uint *keys_out [[ buffer(3) ]],
	device void *vals_out [[ buffer(4) ]],
	uint gid [[ thread_position_in_grid ]]
) {
	bool vals_is_keys = val_size == 0 || (constant uint*)vals_in == keys_in;

	keys_out[gid] = keys_in[gid];
	if (!vals_is_keys) {
		copy_val(gid, gid, val_size, vals_in, vals_out);
	}
}

void copy_val(uint from_gid, uint to_gid, constant ushort &val_size, constant void *vals_in, device void *vals_out) {
	constant uchar *val_in = &((constant uchar *)vals_in)[from_gid * val_size];
	device uchar *val_out = &((device uchar *)vals_out)[to_gid * val_size];
	for (ushort i = 0; i < val_size; i++) {
		val_out[i] = val_in[i];
	}
}
