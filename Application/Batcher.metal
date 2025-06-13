//
//  Batcher.metal
//

#include <metal_stdlib>
using namespace metal;


void batcher_sort_step(
	threadgroup void *ctx,
	void (*cmp_swap)(threadgroup void *, uint, uint),
	uint n,
	uint t,
	uint p,
	uint k
);

void batcher_sort(
	threadgroup void *ctx,
	void (*cmp_swap)(threadgroup void *, uint, uint),
	uint n,
	uint t
) {
	for (uint p = 1; p < n; p *= 2) {
		for (uint k = p; k >= 1; k /= 2) {
			batcher_sort_step(ctx, cmp_swap, n, t, p, k);
			threadgroup_barrier(mem_flags::mem_threadgroup);
		}
	}
}

void batcher_sort_step(
	threadgroup void *ctx,
	void (*cmp_swap)(threadgroup void *, uint, uint),
	uint n,
	uint t,
	uint p,
	uint k
) {
	uint i = t % k;
	uint j = k % p + 2 * k * (t / k);
	if (j >= n - k) return;

	uint l = i + j;
	uint r = i + j + k;
	if (l / 2 / p != r / 2 / p) return;

	cmp_swap(ctx, l, r);
}
