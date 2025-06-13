//
//  Main.metal
//

#include <metal_stdlib>
using namespace metal;

#include "Shaders.h"


uint hash(uint x);
unsigned long sq(uint x);


[[ kernel ]]
void main_gen_tex_kernel(
	texture2d<float, access::write> tex [[ texture(0) ]],
	uint2 gid [[ thread_position_in_grid ]]
) {
	uint t = gid.y * tex.get_width() + gid.x;
	tex.write(float4(float3((float)hash(t) / UINT_MAX), 1), gid);
}

[[ kernel ]]
void main_step_tex_sort_kernel(
	constant MainStepFragmentSortingUniforms &uniforms [[ buffer(0) ]],
	texture2d<float, access::read_write> tex [[ texture(0) ]],
	uint2 gid [[ thread_position_in_grid ]]
) {
	uint t = gid.y * tex.get_width() + gid.x;

	unsigned long i = t % uniforms.k;
	unsigned long j = uniforms.k % uniforms.p + 2 * uniforms.k * (t / uniforms.k);
	if (j >= sq(tex.get_width()) - uniforms.k) return;

	unsigned long l = i + j;
	unsigned long r = i + j + uniforms.k;
	if (l / 2 / uniforms.p != r / 2 / uniforms.p) return;

	uint2 lid = uint2(l % tex.get_width(), l / tex.get_width());
	uint2 rid = uint2(r % tex.get_width(), r / tex.get_width());

	float lv = tex.read(lid).r;
	float rv = tex.read(rid).r;

	if (lv < rv) return;

	tex.write(float4(float3(rv), 1), lid);
	tex.write(float4(float3(lv), 1), rid);
}


struct VertexOutput {
	/// Clip coordinates in vertex stage.
	float4 stage_pos [[ position ]];
	float2 pos;
	float2 uv;
};

[[ vertex ]]
VertexOutput main_vertex(unsigned int vertex_id [[ vertex_id ]]) {
	float2 uv = float2(vertex_id & 2, (vertex_id << 1) & 2);
	float2 pos = uv * float2(2, -2) + float2(-1, 1);
	return {
		.stage_pos = float4(pos, 0, 1),
		.pos = pos,
		.uv = uv,
	};
}


/// `stage_pos` becomes viewport coordinates in framgent stage.
typedef VertexOutput FragmentInput;

[[ fragment ]]
half4 main_fragment(
	FragmentInput interpolated [[ stage_in ]],
	constant MainFragmentUniforms &uniforms [[ buffer(0) ]],
	texture2d<float, access::sample> tex [[ texture(0) ]]
) {
	if (all(uint2(interpolated.stage_pos.xy / 16) == uint2(0))) {
		return half4(half3((float)(uniforms.frame_idx % 256) / 256), 1);
	} else {
		constexpr sampler tex_sampler(coord::normalized, address::repeat, filter::nearest);
		float val = tex.sample(tex_sampler, interpolated.uv).x;
		return half4(half2(val), 0, 1);
	}
}


#pragma mark - Utilities

// From: https://github.com/skeeto/hash-prospector/issues/19#issuecomment-1120105785
uint hash(uint x) {
	x ^= x >> 16;
	x *= 0x21f0aaad;
	x ^= x >> 15;
	x *= 0x735a2d97;
	x ^= x >> 16;
	return x;
}

unsigned long sq(uint x) {
	return (unsigned long)x * (unsigned long)x;
}
