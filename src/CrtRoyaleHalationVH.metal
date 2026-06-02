// -----------------------------------------------------------------------------
// CRT-Royale MSL Port -- HALATION_V + HALATION_BLUR (slang passes 3 + 4).
//
// Original: crt-royale by TroggleMonkey (GPL v2+).
// Ported to Metal Shading Language for RetroVisor integration.
//
// Slang references:
//   blurs/shaders/royale/blur9fast-vertical.slang   (slang pass 3)
//   blurs/shaders/royale/blur9fast-horizontal.slang (slang pass 4)
//   include/blur-functions.h:416-444                (tex2Dblur9fast)
//   include/blur-functions.h:185                    (blur9_std_dev default)
//
// Pipeline position:
//   ... -> BLOOM_APPROX (pass 2) -> HALATION_V (this) -> HALATION_BLUR (this)
//                                -> apply-mask + bloom-final (consumers)
//
// Both passes inherit BLOOM_APPROX's framebuffer size via scale_type=source,
// scale=1.0 in the .slangp preset -> fixed 320 x 240.
//
// Consumers of HALATION_BLUR:
//   - apply-mask (slang pass 7): halation_weight lerp (default 0.0 -- no
//     visible effect at slang defaults, but the runtime parameter is exposed).
//   - bloom-h-reconstitute (slang pass 10): diffusion_weight lerp (default
//     0.075 -- ~7.5% halation always mixed into the final phosphor bloom).
//     This non-zero default was missed in earlier MSL iterations; the
//     correction is part of this iteration.
//
// Why 9-tap separable + bilinear-trick + blur9_std_dev:
//   - The slang blur uses tex2Dblur9fast (5 bilinear samples per pass, 9-tap
//     equivalent). Sigma = blur9_std_dev = 1.7533203125 from blur-functions.h
//     (USE_BINOMIAL_BLUR_STD_DEVS undefined -> "largest unused tail < 1/256"
//     branch). We share the same `tex2Dblur9fast` helper as bloom V/H -- it
//     lives in CrtRoyale.metal, signature unchanged.
//
// dxdy in slang:
//   blur_dxdy = (video_size/output_size) / texture_size
// With scale_type=source and scale=1.0 the three sizes coincide, so this
// reduces to 1/size on the relevant axis (y for V, x for H) and 0 on the
// other. We bake that in directly to avoid plumbing more uniforms.
//
// NOTE: This file is the standalone study mirror; the kernel that actually
// gets built lives in vendor/RetroVisor/RetroVisor/GPU/CrtRoyale.metal.
// -----------------------------------------------------------------------------

#include <metal_stdlib>
using namespace metal;

namespace crt_royale {

    // blur9_std_dev from include/blur-functions.h:185 (default branch).
    // Slang's "largest unused tail <= 1/256" calibration for 9-tap Gauss.
    constant float BLUR9_STD_DEV = 1.7533203125f;

    // tex2Dblur9fast: see CrtRoyale.metal. Shared with BLOOM_V/H.
    inline float3 tex2Dblur9fast_shared(
        texture2d<float, access::sample> tex, sampler sam,
        float2 tex_uv, float2 dxdy, float sigma);

    kernel void pass_halation_v(
        texture2d<float, access::sample> input   [[ texture(0) ]],  // BLOOM_APPROX
        texture2d<float, access::write>  output  [[ texture(1) ]],  // halation_v
        sampler                          sam     [[ sampler(0) ]],
        uint2                            gid     [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
        float2 out_size = float2(output.get_width(), output.get_height());
        float2 uv   = (float2(gid) + 0.5f) / out_size;
        float2 dxdy = float2(0.0f, 1.0f / out_size.y);
        float3 color = tex2Dblur9fast_shared(input, sam, uv, dxdy, BLUR9_STD_DEV);
        output.write(float4(color, 1.0f), gid);
    }

    kernel void pass_halation_h(
        texture2d<float, access::sample> input   [[ texture(0) ]],  // halation_v
        texture2d<float, access::write>  output  [[ texture(1) ]],  // HALATION_BLUR
        sampler                          sam     [[ sampler(0) ]],
        uint2                            gid     [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
        float2 out_size = float2(output.get_width(), output.get_height());
        float2 uv   = (float2(gid) + 0.5f) / out_size;
        float2 dxdy = float2(1.0f / out_size.x, 0.0f);
        float3 color = tex2Dblur9fast_shared(input, sam, uv, dxdy, BLUR9_STD_DEV);
        output.write(float4(color, 1.0f), gid);
    }

} // namespace crt_royale
