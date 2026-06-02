// -----------------------------------------------------------------------------
// CRT-Royale MSL Port - Bloom V/H + Reconstitute (slang passes 9 + 10)
//
// Original: crt-royale by TroggleMonkey (GPL v2+)
// Ported to Metal Shading Language for RetroVisor integration.
//
// Slang references:
//   crt-royale-bloom-vertical.slang                (Slang pass 9)
//   crt-royale-bloom-horizontal-reconstitute.slang (Slang pass 10)
//
// These two passes together implement the phosphor-bloom chain that produces
// BLOOM_FINAL (= the input that slang's pass 11 samples). With our Pass 4
// (= slang pass 11) now reading BLOOM_FINAL instead of MASKED_SCANLINES, the
// 04-final delta-E to the slang reference drops dramatically (e.g. colorbars
// 38.35 -> 1.67, solid_white 51.39 -> 2.97).
//
// Pass 9 (Bloom V): vertical 9-tap separable Gaussian blur of BRIGHTPASS.
// Pass 10 (Bloom H + Reconstitute): horizontal 9-tap blur of Pass 9's output,
//   then add the "dimpass" (MASKED_SCANLINES - BRIGHTPASS) back, un-dim, and
//   mask-amplify. With diffusion_weight = 0 (slang default), the halation
//   mix step is a no-op and HALATION_BLUR can be skipped entirely.
//
// 9-tap fast blur: 5 samples per pixel via the bilinear-tap trick (slang
// blur-functions.h:416-444). A linear sample at offset (1 + w12_ratio)
// effectively averages texels 1 and 2 with the correct Gaussian weight.
//
// Static-config simplifications:
//   - bloom_sigma computed from u.mask_triad_size via
//     get_min_sigma_to_blur_triad (slang's get_final_bloom_sigma in the
//     no-RUNTIME_PHOSPHOR_BLOOM_SIGMA static path).
//   - 9-tap blur only; PHOSPHOR_BLOOM_TRIADS_LARGER_*_PIXELS branches that
//     pick 17/25/31/43-tap are TODO for very large triad sizes.
//   - levels_contrast = 1.0 (slang default, not exposed).
//   - diffusion_weight = 0 -> HALATION_BLUR mixing skipped.
//
// NOTE: This file is the standalone study mirror; the kernels that get
// built live in vendor/RetroVisor/RetroVisor/GPU/CrtRoyale.metal.
// -----------------------------------------------------------------------------

#include <metal_stdlib>
using namespace metal;

namespace crt_royale {

    // 9-tap separable Gaussian using the bilinear-tap trick (5 samples).
    inline float3 tex2Dblur9fast(texture2d<float, access::sample> tex,
                                 float2 tex_uv, float2 dxdy, float sigma,
                                 sampler sam)
    {
        float denom_inv = 0.5f / (sigma * sigma);
        float w0 = 1.0f;
        float w1 = exp(-1.0f  * denom_inv);
        float w2 = exp(-4.0f  * denom_inv);
        float w3 = exp(-9.0f  * denom_inv);
        float w4 = exp(-16.0f * denom_inv);
        float weight_sum_inv = 1.0f / (w0 + 2.0f * (w1 + w2 + w3 + w4));
        float w12 = w1 + w2;
        float w34 = w3 + w4;
        float w12_ratio = w2 / w12;
        float w34_ratio = w4 / w34;
        float3 sum = float3(0.0f);
        sum += w34 * tex.sample(sam, tex_uv - (3.0f + w34_ratio) * dxdy).rgb;
        sum += w12 * tex.sample(sam, tex_uv - (1.0f + w12_ratio) * dxdy).rgb;
        sum += w0  * tex.sample(sam, tex_uv).rgb;
        sum += w12 * tex.sample(sam, tex_uv + (1.0f + w12_ratio) * dxdy).rgb;
        sum += w34 * tex.sample(sam, tex_uv + (3.0f + w34_ratio) * dxdy).rgb;
        return sum * weight_sum_inv;
    }

    // Slang's get_min_sigma_to_blur_triad (reused from brightpass).
    inline float bvh_get_min_sigma_to_blur_triad(float triad_size, float thresh) {
        return -0.05168f + 0.6113f * triad_size -
               1.122f * triad_size * sqrt(0.000416f + thresh);
    }

    // Pass 9: vertical 9-tap Gaussian blur of BRIGHTPASS.
    kernel void pass_bloom_v(
        texture2d<float, access::sample> input   [[ texture(0) ]],
        texture2d<float, access::write>  output  [[ texture(1) ]],
        constant Uniforms&               u       [[ buffer(0)  ]],
        sampler                          sam     [[ sampler(0) ]],
        uint2                            gid     [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
        float2 out_size = float2(output.get_width(), output.get_height());
        float2 uv = (float2(gid) + 0.5f) / out_size;
        float2 dxdy = float2(0.0f, 1.0f / out_size.y);
        constexpr float bloom_diff_thresh = 1.0f / 256.0f;
        float bloom_sigma = bvh_get_min_sigma_to_blur_triad(
            u.mask_triad_size, bloom_diff_thresh);
        float3 color = tex2Dblur9fast(input, uv, dxdy, bloom_sigma, sam);
        output.write(float4(color, 1.0f), gid);
    }

    // Pass 10: horizontal blur of BLOOM_V + reconstitute mask-dim portion.
    kernel void pass_bloom_h_reconstitute(
        texture2d<float, access::sample> bloom_v          [[ texture(0) ]],
        texture2d<float, access::write>  output           [[ texture(1) ]],
        texture2d<float, access::sample> brightpass       [[ texture(2) ]],
        texture2d<float, access::sample> masked_scanlines [[ texture(3) ]],
        constant Uniforms&               u                [[ buffer(0)  ]],
        sampler                          sam              [[ sampler(0) ]],
        uint2                            gid              [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
        float2 out_size = float2(output.get_width(), output.get_height());
        float2 uv = (float2(gid) + 0.5f) / out_size;
        float2 dxdy = float2(1.0f / out_size.x, 0.0f);
        constexpr float bloom_diff_thresh = 1.0f / 256.0f;
        float bloom_sigma = bvh_get_min_sigma_to_blur_triad(
            u.mask_triad_size, bloom_diff_thresh);
        float3 blurred_brightpass = tex2Dblur9fast(bloom_v, uv, dxdy,
                                                    bloom_sigma, sam);
        float3 intensity_dim = masked_scanlines.sample(sam, uv).rgb;
        float3 bp           = brightpass.sample(sam, uv).rgb;
        float3 dimpass      = intensity_dim - bp;
        float undim_factor  = 1.0f / max(u.levels_autodim_temp, 1e-5f);
        float3 phosphor_bloom = (dimpass + blurred_brightpass) *
                                u.mask_amplify * undim_factor;
        output.write(float4(phosphor_bloom, 1.0f), gid);
    }

} // namespace crt_royale
