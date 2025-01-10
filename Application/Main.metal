//
//  Main.metal
//

#include <metal_stdlib>
using namespace metal;

#include "Shaders.h"


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
	constant FragmentUniforms &uniforms [[ buffer(0) ]]
) {
	if (all(uint2(interpolated.stage_pos.xy / 16) == uint2(0))) {
		return half4(half3((float)(uniforms.frame_idx % 256) / 256), 1);
	} else {
		return half4(half2(interpolated.uv), 0, 1);
	}
}
